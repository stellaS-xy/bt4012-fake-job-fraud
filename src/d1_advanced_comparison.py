"""Three new approaches on the report's saved nested grouped development folds.

Each method has independent caches. Neural ensemble stages resume separately;
frozen embeddings can be built once and used for both semantic classifiers.
"""
from dataclasses import asdict
import gc
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from src import d1_targeted_improvements as targeted
from src import d1_advanced_models as models
from src.d1_semantic_features import TEXT_FIELDS, clean_text
from src.evaluation import (make_folds, best_f1_threshold, score_summary,
                            select_settings, paired_comparison)
from src.features import GROUP, TARGET, build_inputs

METHODS = ('catboost_native', 'semantic_only', 'semantic_structured', 'calibrated_ensemble')
NAMES = {**targeted.NAMES, 'catboost_native': 'CatBoost: native text + structured',
    'semantic_only': 'MiniLM + logistic regression',
    'semantic_structured': 'MiniLM + structured + logistic regression',
    'calibrated_ensemble': 'Calibrated MLP + Extra Trees (50/50)'}
REPRESENTATIONS = {'mlp_reference': 'A2, 30k TF-IDF', 'lr_reference': 'A2, 30k TF-IDF',
    'extra_trees_leaf1': 'A2, 30k TF-IDF', 'catboost_native': 'native categorical/numeric; native 5k text dictionary',
    'semantic_only': 'frozen 384-dimensional MiniLM', 'semantic_structured': 'frozen MiniLM + structured/style',
    'calibrated_ensemble': 'A2, 30k TF-IDF; fitting-only calibration holdout'}
PACKAGES = (*targeted.PACKAGES, 'catboost', 'sentence-transformers', 'torch', 'transformers')
SOURCES = ('d1_advanced_comparison.py', 'd1_advanced_models.py', 'd1_semantic_features.py',
           'd1_targeted_improvements.py', *targeted.SOURCES)


def experiment_directory(root, frame, config):
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    settings = asdict(config)
    for key in ('outer_limit', 'repeats'):
        settings.pop(key)
    digest = hashlib.sha256(frame.to_csv(index=False, lineterminator='\n').encode())
    digest.update(json.dumps({'config': settings, 'versions': versions}, sort_keys=True).encode())
    for name in SOURCES:
        digest.update((Path(__file__).parent / name).read_bytes())
    return Path(root) / 'data/processed/12_advanced_comparison' / digest.hexdigest()[:20], versions


def _method_directory(directory, key, embedding_manifest):
    if key.startswith('semantic_'):
        if embedding_manifest is None:
            raise ValueError('Prepare and validate frozen embeddings before running semantic classifiers.')
        suffix = hashlib.sha256(json.dumps(embedding_manifest, sort_keys=True).encode()).hexdigest()[:12]
        return directory / f'{key}_{suffix}'
    return directory / key


def _reference_rows(root, frame, config, folds):
    key, _ = targeted._fingerprint(frame, config, ('d1_targeted_improvements.py', *targeted.SOURCES))
    improved_directory = Path(root) / 'data/processed/11_targeted_improvements' / key
    rows = []
    for fold in folds:
        identity = targeted.baseline._identity(frame, fold)
        r, k, _, scoring = fold
        reference = targeted.baseline._reference_predictions(root, frame, config, fold)
        path = improved_directory / f'r{r}_f{k}_extra_trees_leaf1_{identity["fold_signature"][:16]}.npz'
        if not path.is_file():
            raise FileNotFoundError(f'Missing compatible improved tree reference: {path}')
        p, detail = targeted._read_prediction(path, frame.iloc[scoring].job_id.to_numpy())
        reference['extra_trees_leaf1'] = (p, detail, str(path))
        for name, (p, detail, path) in reference.items():
            rows.append({'key': name, 'model': NAMES[name], **identity,
                **score_summary(frame.iloc[scoring][TARGET], p, threshold=detail['threshold']),
                'selected_params': detail['params'], 'minutes': detail['minutes'], 'source': path})
    return rows


def method_spec(key, config):
    if key == 'catboost_native':
        return models.NativeCatBoost(seed=config.seed, threads=config.threads), {'depth': [4, 6]}
    if key in ('semantic_only', 'semantic_structured'):
        return models.SemanticLogistic(include_structured=key == 'semantic_structured', seed=config.seed), {'C': [.1, .3, 1.]}
    raise ValueError('Unknown ordinary classifier method.')


def validate_embeddings(frame, embeddings, manifest):
    if embeddings.ndim != 2 or len(embeddings) != len(frame) or not np.isfinite(embeddings).all():
        raise ValueError('Semantic vector shape/values differ from current development rows.')
    request = manifest['request']
    ids_sha = hashlib.sha256(frame.job_id.to_numpy(dtype='<i8').tobytes()).hexdigest()
    texts = frame[list(TEXT_FIELDS)].fillna('').astype(str).agg(' '.join, axis=1).map(clean_text)
    texts_sha = hashlib.sha256(texts.to_csv(index=False, lineterminator='\n').encode()).hexdigest()
    source_sha = hashlib.sha256((Path(__file__).parent / 'd1_semantic_features.py').read_bytes()).hexdigest()
    stream = io.BytesIO()
    np.save(stream, embeddings, allow_pickle=False)
    if (request['job_ids_sha256'] != ids_sha or request['texts_sha256'] != texts_sha
            or request['source_sha256'] != source_sha
            or manifest['array_sha256'] != hashlib.sha256(stream.getvalue()).hexdigest()):
        raise ValueError('Semantic vectors do not match the current posting order, text or preprocessing.')


def _ensemble_predictions(frame, fold, config, stages):
    r, k, fitting, scoring = fold
    training = frame.iloc[fitting]
    y, groups = training[TARGET].to_numpy(), training[GROUP].to_numpy()
    oof = np.full(len(training), np.nan)
    aps, stage_details = [], []
    inner = make_folds(groups, y, config.inner_splits, 1, config.seed + r * 1000 + k)
    for _, number, fit, val in inner:
        print(f'  Ensemble inner {number + 1}/{config.inner_splits}', flush=True)
        p, detail = models.calibrated_stage(training.iloc[fit], training.iloc[val].drop(columns=TARGET),
            config, stages / f'inner_{number}.npz')
        oof[val] = p
        aps.append(float(average_precision_score(y[val], p)))
        stage_details.append(detail)
    if not np.isfinite(oof).all():
        raise ValueError('Missing inner ensemble predictions.')
    threshold = best_f1_threshold(y, oof)
    print('  Ensemble outer fit and prediction', flush=True)
    p, final_detail = models.calibrated_stage(training, frame.iloc[scoring].drop(columns=TARGET),
        config, stages / 'outer.npz')
    return p, {'params': {'weights': [.5, .5], 'calibration_splits': 5},
        'threshold': threshold, 'selection_pr_auc': float(np.mean(aps)),
        'inner_aps': aps, 'inner_stage_details': stage_details, 'outer_stage_detail': final_detail}


def _summary_tables(scores, expected):
    rows, paired = [], []
    for key, table in scores.groupby('key', sort=False):
        rows.append({'key': key, 'model': NAMES[key], 'representation': REPRESENTATIONS[key],
            'evaluated_folds': len(table), 'expected_folds': expected, 'complete': len(table) == expected,
            **{metric: table[metric].mean() for metric in targeted.baseline.METRICS},
            'ap_sd': table.pr_auc.std(ddof=1), 'minutes_per_fold': table.minutes.mean()})
        if key in METHODS and len(table) >= 2:
            for reference in ('mlp_reference', 'extra_trees_leaf1'):
                ref = scores.loc[scores.key == reference]
                ap = paired_comparison(table, ref, 'pr_auc')
                paired.append({'candidate': NAMES[key], 'reference': NAMES[reference],
                    'ap_difference': ap['mean_diff'], 'ap_interval_low': ap['low'], 'ap_interval_high': ap['high'],
                    'share_folds_ap_better': ap['share_folds_a_better'],
                    **{metric + '_difference': table[metric].mean() - ref[metric].mean()
                       for metric in ('precision', 'recall', 'f1')}})
    return pd.DataFrame(rows).sort_values('pr_auc', ascending=False), pd.DataFrame(paired)


def run_comparison(root, frame, config=None, methods=METHODS, embeddings=None, embedding_manifest=None):
    config = config or targeted.baseline.ComparisonConfig(outer_limit=3)
    root, methods = Path(root), tuple(methods)
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(METHODS):
        raise ValueError('Choose unique supported methods.')
    needs_embeddings = any(key.startswith('semantic_') for key in methods)
    if needs_embeddings and (embeddings is None or embedding_manifest is None):
        raise ValueError('Frozen semantic vectors and their manifest are required.')
    if needs_embeddings:
        validate_embeddings(frame, embeddings, embedding_manifest)
    all_folds = targeted.baseline.restore_folds(frame, config, root / 'data/processed/model_cv_assignments.csv')
    folds = all_folds if config.outer_limit is None else all_folds[:config.outer_limit]
    reference_rows = _reference_rows(root, frame, config, folds)  # fail before any expensive new fit
    directory, versions = experiment_directory(root, frame, config)
    directory.mkdir(parents=True, exist_ok=True)
    X = build_inputs(frame.copy())
    X_semantic = None
    if needs_embeddings:
        X_semantic = pd.concat([X, pd.DataFrame(embeddings, index=X.index,
            columns=[f'sem_{i:03d}' for i in range(embeddings.shape[1])])], axis=1)
    y, groups = frame[TARGET].to_numpy(), frame[GROUP].to_numpy()
    rows = reference_rows.copy()
    for fold in folds:
        identity = targeted.baseline._identity(frame, fold)
        r, k, fitting, scoring = fold
        for key in methods:
            folder = _method_directory(directory, key, embedding_manifest)
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f'r{r}_f{k}_{identity["fold_signature"][:16]}.npz'
            if path.is_file():
                p, detail = targeted._read_prediction(path, frame.iloc[scoring].job_id.to_numpy())
                print(f'Repeat {r + 1}, fold {k + 1}: {NAMES[key]} — cached', flush=True)
            else:
                print(f'Repeat {r + 1}, fold {k + 1}: {NAMES[key]} — training', flush=True)
                started = time.monotonic()
                with threadpool_limits(limits=config.threads):
                    if key == 'calibrated_ensemble':
                        p, detail = _ensemble_predictions(frame, fold, config, folder / f'r{r}_f{k}_stages')
                    else:
                        matrix = X_semantic if key.startswith('semantic_') else X
                        pipeline, grid = method_spec(key, config)
                        choice = select_settings(lambda: pipeline, matrix.iloc[fitting], y[fitting], groups[fitting],
                            grid, inner_splits=config.inner_splits, seed=config.seed + r * 1000 + k)
                        fitted = clone(pipeline).set_params(**choice.params).fit(matrix.iloc[fitting], y[fitting])
                        p = fitted.predict_proba(matrix.iloc[scoring])[:, 1]
                        detail = {'params': choice.params, 'threshold': choice.threshold,
                            'selection_pr_auc': float(choice.candidates.mean_inner_pr_auc.max()),
                            'candidates': choice.candidates.to_dict(orient='records')}
                        del fitted
                        gc.collect()
                detail['minutes'] = (time.monotonic() - started) / 60
                score_summary(y[scoring], p, threshold=detail['threshold'])
                tmp = path.with_suffix('.tmp')
                with tmp.open('wb') as stream:
                    np.savez_compressed(stream, p=p, job_id=frame.iloc[scoring].job_id.to_numpy(), detail=json.dumps(detail))
                tmp.replace(path)
            measured = score_summary(y[scoring], p, threshold=detail['threshold'])
            rows.append({'key': key, 'model': NAMES[key], **identity, **measured,
                'minutes': detail['minutes'], 'selected_params': detail['params'], 'source': str(path)})
            print(f'  AP {measured["pr_auc"]:.4f}; precision {measured["precision"]:.4f}; '
                  f'recall {measured["recall"]:.4f}; F1 {measured["f1"]:.4f}', flush=True)
    scores = pd.DataFrame(rows)
    summary, paired = _summary_tables(scores, len(all_folds))
    # Distinct worker/report scopes avoid collisions when different methods run concurrently.
    report_dir = directory / 'reports' / '__'.join(sorted(methods))
    report_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(report_dir / 'fold_metrics.csv', index=False)
    summary.to_csv(report_dir / 'summary.csv', index=False)
    paired.to_csv(report_dir / 'paired_differences.csv', index=False)
    manifest = {'status': 'exploratory nested grouped development comparison', 'config': asdict(config),
        'methods': methods, 'versions': versions,
        'training_sha256': hashlib.sha256(frame.to_csv(index=False, lineterminator='\n').encode()).hexdigest(),
        'training_rows': len(frame), 'fraud_rows': int(y.sum()), 'evaluated_outer_folds': len(folds),
        'expected_outer_folds': len(all_folds), 'folds': [targeted.baseline._identity(frame, fold) for fold in folds],
        'selection': f'{config.inner_splits} grouped inner folds; mean AP selects settings; inner OOF F1 selects threshold',
        'ensemble': 'fixed equal weights; each inner/outer fit holds out training groups for component sigmoid calibration; no post-calibration refit',
        'ensemble_calibration_fraction': 'one of five grouped splits inside each fit',
        'ensemble_mlp_settings': 'original 09 architecture, batch 64, rate .001, max 50 epochs, patience 5, grouped checkpoint/refit',
        'embedding_manifest': embedding_manifest if needs_embeddings else None,
        'aggregation': 'arithmetic means of matching outer-fold metrics; repeated observations are not independent',
        'final_test_accessed': False}
    (report_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return {'scores': scores, 'summary': summary, 'paired': paired, 'manifest': manifest,
            'directory': directory, 'report_directory': report_dir}
