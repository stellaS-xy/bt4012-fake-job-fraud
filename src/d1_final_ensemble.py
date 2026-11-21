"""Final notebook-13 ensemble fit/export and a single fixed-split test evaluation.

No work at import. Thresholds use development inner OOF predictions exclusively.
Calibration groups stay separate from component fitting; no post-calibration refit.
"""
import argparse
import ast
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from src.d1_advanced_models import _components, _configure_tf, calibrated_stage
from src.d1_candidate_comparison import ComparisonConfig
from src.evaluation import best_f1_threshold, make_folds, score_summary
from src.features import build_inputs, GROUP, TARGET
from src.reference13_runtime import training_lock

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SHA = '61117d0309733e4dcf28d1f9836588ffaf72b7b44479113a742533134cad57ce'
TEST_SHA = 'd3a9c9ba17208157f9f85fe01e78af22f31fe0f543d3343d60622262a32c1480'
MLP_CLASS_SHA = 'b46a0fad2eb68398c3b378496b6414ee5fc7a30c34fb531eafc6616aae0d2573'
CSR_FUNCTION_SHA = '3dd03d0476085f01fdc5e19b83b88abfb3dc8c5accb70ceacfef19bf807864dc'
ARTIFACT_FILES = ('mlp_features.joblib', 'mlp.keras', 'mlp_calibrator.joblib',
                  'trees_features.joblib', 'trees.joblib', 'trees_calibrator.joblib')
HISTORY_NOTE = ('Current posting/group separation passed the audit. Earlier project-wide '
                'test exposure remains unverified. This evaluation uses the audited current fixed split. '
                'Do not describe this as a proven never-accessed project-wide test.')


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def _release():
    import tensorflow as tf
    tf.keras.backend.clear_session(free_memory=True)
    gc.collect()


def load_fixed_split(root):
    train_path = Path(root)/'data/processed/d1_train.csv'
    test_path = Path(root)/'data/processed/d1_test.csv'
    if file_sha(train_path) != TRAIN_SHA or file_sha(test_path) != TEST_SHA:
        raise ValueError('The audited train/test files changed. Re-audit before final evaluation.')
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    if (not train.job_id.is_unique or not test.job_id.is_unique
            or train[GROUP].isna().any() or test[GROUP].isna().any()
            or not set(train.job_id).isdisjoint(test.job_id)
            or not set(train[GROUP]).isdisjoint(test[GROUP])):
        raise ValueError('Invalid IDs or train/test posting/group overlap.')
    if set(train[TARGET]) != {0, 1} or set(test[TARGET]) != {0, 1}:
        raise ValueError('Both train and test must have binary labels.')
    return train, test


def ast_fingerprint(node):
    """Portable syntax fingerprint: normalise only Python 3.12's empty type_params.

    ast.dump() itself is version-dependent. Nonempty type parameters and all
    other semantic fields remain checked, including defaults and method bodies.
    """
    def canonical(value):
        if isinstance(value, ast.AST):
            return {'node': type(value).__name__,
                    'fields': [[name, canonical(item)] for name, item in ast.iter_fields(value)
                               if not (name == 'type_params' and item == [])]}
        if isinstance(value, list):
            return [canonical(item) for item in value]
        return value
    content = json.dumps(canonical(node), sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(content).hexdigest()


def verify_selected_sources(root):
    """Freeze the estimator/calibration implementations that produced notebook 13."""
    root = Path(root)
    published = json.loads((root/'reports/13_reference_comparison/manifest.json').read_text())
    names = ('features.py', 'evaluation.py', 'd1_row_neural.py', 'd1_advanced_models.py',
             'd1_candidate_comparison.py')
    for name in names:
        if file_sha(root/'src'/name) != published['sources'][name]:
            raise ValueError(f'Selected implementation changed: {name}')
    tree = ast.parse((root/'src/d1_a2_mlp.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'A2MLP')
    if ast_fingerprint(node) != MLP_CLASS_SHA:
        raise ValueError('MLP estimator differs from the audited notebook 13 implementation.')
    csr = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_csr')
    if ast_fingerprint(csr) != CSR_FUNCTION_SHA:
        raise ValueError('MLP sparse-input conversion differs from the audited implementation.')
    return {name: file_sha(root/'src'/name) for name in names} | {
        'A2MLP_class_ast_sha256': MLP_CLASS_SHA, '_csr_ast_sha256': CSR_FUNCTION_SHA,
        'ast_fingerprint_format': 'canonical-v1-empty-type-params-normalised'}


def select_threshold(train, config, directory):
    """Only development labels enter the grouped OOF threshold selection."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    y, groups = train[TARGET].to_numpy(), train[GROUP].to_numpy()
    folds = make_folds(groups, y, n_splits=3, n_repeats=1, seed=42)
    oof = np.full(len(train), np.nan)
    for _, k, fitting, scoring in folds:
        print(f'Final threshold selection: inner fold {k+1}/3', flush=True)
        p, _ = calibrated_stage(train.iloc[fitting], train.iloc[scoring].drop(columns=TARGET),
                                 config, Path(directory)/f'inner_{k}.npz')
        oof[scoring] = p
        _release()
    cutoff = best_f1_threshold(y, oof)
    predictions = pd.DataFrame({'job_id': train.job_id, 'group_id': groups,
                                'fraudulent': y, 'inner_oof_score': oof})
    predictions.to_csv(Path(directory)/'threshold_oof.csv', index=False)
    return dict(threshold=cutoff, rule='maximum F1 on 3 grouped inner OOF folds',
                seed=42, inner_splits=3, source='development data only')


def fit_export(train, config, artifact, threshold):
    """Same core/calibration split and models as calibrated_stage; now retain weights."""
    artifact = Path(artifact)
    y, groups = train[TARGET].to_numpy(), train[GROUP].to_numpy()
    _, _, fitting, calibration = make_folds(groups, y, n_splits=5, n_repeats=1, seed=config.seed+300)[0]
    if not set(groups[fitting]).isdisjoint(groups[calibration]):
        raise ValueError('Calibration group overlap.')
    X = build_inputs(train.copy())
    artifact.mkdir(parents=True, exist_ok=True)
    expected, details = [], []
    probe = calibration[:32]
    for name, number in [('mlp', 0), ('trees', 1)]:
        print(f'Final fit/export: {name}; {len(fitting)} core / {len(calibration)} calibration rows', flush=True)
        # Construct each component sequentially to minimise peak memory.
        model = _components(config)[number]
        model.fit(X.iloc[fitting], y[fitting])
        p_cal = model.predict_proba(X.iloc[calibration])[:, 1]
        calibrator = LogisticRegression(C=1000., solver='lbfgs', max_iter=1000)
        calibrator.fit(logit(np.clip(p_cal.astype(float), 1e-6, 1-1e-6))[:, None], y[calibration])
        expected.append(calibrator.predict_proba(logit(np.clip(p_cal[:len(probe)].astype(float), 1e-6, 1-1e-6))[:, None])[:, 1])
        joblib.dump(model.features_, artifact/f'{name}_features.joblib')
        joblib.dump(calibrator, artifact/f'{name}_calibrator.joblib')
        if name == 'mlp':
            import tensorflow as tf
            # A fresh uncompiled wrapper shares the trained layers, excluding Adam state.
            inference_model = tf.keras.Model(model.adapter_.model.input, model.adapter_.model.output)
            inference_model.save(artifact/'mlp.keras')
            del inference_model
        else:
            joblib.dump(model.model_, artifact/'trees.joblib')
        details.append(dict(model=name, selected_epoch=getattr(model, 'selected_epoch_', None),
                            sigmoid_slope=float(calibrator.coef_[0, 0]), sigmoid_intercept=float(calibrator.intercept_[0])))
        del model, calibrator
        _release()
    runtime_versions = {name: importlib.metadata.version(name) for name in
        ['numpy', 'pandas', 'scikit-learn', 'scipy', 'tensorflow', 'keras', 'joblib', 'threadpoolctl']}
    (artifact/'runtime_requirements.txt').write_text(
        ''.join(f'{name}=={version}\n' for name, version in runtime_versions.items()), encoding='utf-8')
    metadata = dict(model='calibrated_mlp_extra_trees', weights=[.5, .5], threshold=threshold,
        core_rows=len(fitting), calibration_rows=len(calibration), refit_after_calibration=False,
        seed=config.seed, component_details=details, training_sha256=TRAIN_SHA,
        runtime_versions=runtime_versions,
        features_source_sha256=file_sha(Path(__file__).with_name('features.py')),
        file_hashes={name: file_sha(artifact/name) for name in ARTIFACT_FILES})
    write_json(artifact/'metadata.json', metadata)
    pd.DataFrame({'job_id': train.iloc[fitting].job_id, 'partition': 'core'}).to_csv(artifact/'core_ids.csv', index=False)
    pd.DataFrame({'job_id': train.iloc[calibration].job_id, 'partition': 'calibration'}).to_csv(artifact/'calibration_ids.csv', index=False)
    # Verify persistence on development calibration rows before ever scoring test.
    restored = predict_artifact(artifact, train.iloc[probe].drop(columns=TARGET))
    np.testing.assert_allclose(restored, .5*expected[0]+.5*expected[1], rtol=1e-5, atol=1e-6)
    write_json(artifact/'export_verified.json', {'development_probe_rows': len(probe), 'test_used': False})
    return metadata


def predict_artifact(artifact, frame):
    """Inference only; batch-densify neural inputs, never a full TF-IDF matrix."""
    import tensorflow as tf
    artifact = Path(artifact)
    metadata = json.loads((artifact/'metadata.json').read_text())
    if metadata['features_source_sha256'] != file_sha(Path(__file__).with_name('features.py')):
        raise ValueError('Raw feature construction differs from the trained model. Use its release code version.')
    for name, digest in metadata['file_hashes'].items():
        if file_sha(artifact/name) != digest:
            raise ValueError(f'Export checksum mismatch: {name}')
    X = build_inputs(frame.drop(columns=TARGET, errors='ignore').copy())
    output = np.zeros(len(X), dtype=float)
    for name, weight in [('mlp', .5), ('trees', .5)]:
        features = joblib.load(artifact/f'{name}_features.joblib')
        matrix = sparse.csr_matrix(features.transform(X), dtype=np.float32)
        calibrator = joblib.load(artifact/f'{name}_calibrator.joblib')
        if name == 'mlp':
            model = tf.keras.models.load_model(artifact/'mlp.keras', compile=False)
            p = np.concatenate([np.asarray(model.predict_on_batch(matrix[i:i+64].toarray())).reshape(-1)
                                for i in range(0, len(X), 64)])
        else:
            model = joblib.load(artifact/'trees.joblib')
            p = model.predict_proba(matrix)[:, 1]
        output += weight*calibrator.predict_proba(logit(np.clip(p.astype(float), 1e-6, 1-1e-6))[:, None])[:, 1]
        del model, features, matrix, calibrator
        _release()
    return output


def group_bootstrap(y, p, groups, threshold, *, repeats=1000, seed=2026):
    """Resample whole groups, keeping their postings together; never refit/tune."""
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    unique = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in unique]
    rng = np.random.default_rng(seed)
    keys = ['pr_auc', 'roc_auc', 'precision', 'recall', 'f1', 'false_positive_rate']
    rows = []
    for _ in range(repeats):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(unique), len(unique))])
        if len(np.unique(y[idx])) != 2:
            continue
        metric = score_summary(y[idx], p[idx], threshold=threshold)
        rows.append([metric[k] for k in keys])
    if not rows:
        return {'requested': repeats, 'valid': 0, 'intervals': {}}
    values = np.asarray(rows)
    return {'requested': repeats, 'valid': len(rows), 'seed': seed,
            'method': 'percentile 95% cluster bootstrap; groups resampled with replacement',
            'intervals': {k: {'low': float(np.quantile(values[:, j], .025)),
                              'high': float(np.quantile(values[:, j], .975))} for j, k in enumerate(keys)}}


def run(root, directory):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    if (directory/'test_metrics.json').exists():
        print((directory/'test_metrics.json').read_text())
        print('Existing final result loaded; no training or new test predictions.')
        return
    train, test = load_fixed_split(root)
    source_hashes = verify_selected_sources(root)
    versions = {p: importlib.metadata.version(p) for p in
                ['numpy', 'pandas', 'scikit-learn', 'scipy', 'tensorflow', 'keras', 'joblib', 'threadpoolctl']}
    config = ComparisonConfig(outer_limit=None)
    plan = dict(model='notebook13_fixed_calibrated_ensemble', config=asdict(config), weights=[.5, .5],
        training_sha256=TRAIN_SHA, test_sha256=TEST_SHA, source_hashes=source_hashes,
        final_runner_sha256=file_sha(Path(__file__)), versions=versions,
        threshold_policy='3 grouped inner OOF maximum F1 on development only',
        calibration_policy='5-way grouped holdout, seed 342; sigmoid, C=1000; no refit afterwards',
        bootstrap={'repeats': 1000, 'seed': 2026}, history_note=HISTORY_NOTE)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'frozen_plan.json').exists():
        if json.loads((directory/'frozen_plan.json').read_text()) != plan:
            raise ValueError('Frozen final-evaluation plan changed. Do not overwrite/re-tune this run.')
    else:
        write_json(directory/'frozen_plan.json', plan)
    started = time.monotonic()
    stages, artifact = directory/'training_stages', directory/'artifact'
    stages.mkdir(exist_ok=True)
    _configure_tf(config.threads)
    with threadpool_limits(limits=config.threads):
        selection_path = directory/'threshold_selection.json'
        if selection_path.exists():
            selection = json.loads(selection_path.read_text())
        else:
            selection = select_threshold(train, config, stages)
            write_json(selection_path, selection)
        print(f'Frozen training-selected threshold: {selection["threshold"]:.6f}', flush=True)
        if not (artifact/'export_verified.json').exists():
            fit_export(train, config, artifact, selection['threshold'])
        predictions_path = directory/'test_predictions.csv'
        if predictions_path.exists():
            saved = pd.read_csv(predictions_path)
            if (not np.array_equal(saved.job_id, test.job_id)
                    or not np.array_equal(saved[GROUP], test[GROUP])
                    or not np.array_equal(saved[TARGET], test[TARGET])):
                raise ValueError('Saved test prediction identities differ.')
            p = saved.score.to_numpy()
            print('Reusing immutable saved test predictions.', flush=True)
        else:
            print('Final model exported; predicting the fixed test set once.', flush=True)
            p = predict_artifact(artifact, test.drop(columns=TARGET))
            saved = pd.DataFrame({'job_id': test.job_id, 'group_id': test[GROUP],
                'fraudulent': test[TARGET], 'score': p, 'flagged': p >= selection['threshold']})
            temporary = predictions_path.with_suffix('.csv.tmp')
            saved.to_csv(temporary, index=False); temporary.replace(predictions_path)
        metrics = score_summary(test[TARGET].to_numpy(), p, threshold=selection['threshold'])
        print('Predictions frozen; calculating group bootstrap intervals.', flush=True)
        intervals = group_bootstrap(test[TARGET].to_numpy(), p, test[GROUP].to_numpy(), selection['threshold'])
    from src.final_ensemble_artifact import package_artifact
    archive = directory/'calibrated_mlp_extra_trees_v1.zip'
    archive_sha = package_artifact(artifact, archive)
    result = dict(model='Calibrated MLP + Extra Trees (50/50)', evaluation='fixed existing split; historical caveat disclosed',
        rows=len(test), fraudulent=int(test[TARGET].sum()), groups=int(test[GROUP].nunique()),
        metrics=metrics, bootstrap=intervals, elapsed_minutes=(time.monotonic()-started)/60,
        created_utc=datetime.now(timezone.utc).isoformat(), history_note=HISTORY_NOTE,
        test_predictions_sha256=file_sha(directory/'test_predictions.csv'),
        artifact=str(artifact), artifact_archive=str(archive), artifact_archive_sha256=archive_sha,
        frozen_plan_sha256=file_sha(directory/'frozen_plan.json'))
    write_json(directory/'test_metrics.json', result)
    pd.DataFrame([metrics]).to_csv(directory/'test_metrics.csv', index=False)
    print(json.dumps(result, indent=2), flush=True)
    print('Saved:', directory, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT/'data/processed/final_ensemble_test/v1')
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    try:
        with training_lock(args.run_dir/'run.lock'):
            run(ROOT, args.run_dir)
    except BlockingIOError:
        raise SystemExit('A final-evaluation worker is already running. Do not start another copy.')
