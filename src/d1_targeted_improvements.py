"""Two prespecified follow-ups to 10, with independent prediction caches.

Original files/caches remain read-only. Use saved 02 outer folds, training-only
three-fold setting/threshold selection, and matching original 09 MLP results.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile
import time

from joblib import Memory
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import precision_score, recall_score, f1_score
from threadpoolctl import threadpool_limits

from src import d1_candidate_comparison as baseline
from src.evaluation import paired_comparison, score_summary, select_settings
from src.features import GROUP, TARGET, build_inputs

TRIALS = ('extra_trees_leaf1', 'lightgbm_5k')
NAMES = {**baseline.NAMES, 'extra_trees_leaf1': 'Extra Trees: leaf size 1',
         'lightgbm_5k': 'LightGBM: 5,000 terms'}
PACKAGES = ('numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib', 'tensorflow', 'keras', 'lightgbm')
SOURCES = ('d1_candidate_comparison.py', 'features.py', 'evaluation.py', 'd1_a2_mlp.py', 'd1_row_neural.py')


def _fingerprint(frame, config, sources):
    settings = asdict(config)
    for field in ('outer_limit', 'repeats'):
        settings.pop(field)
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    key = hashlib.sha256(frame.to_csv(index=False, lineterminator='\n').encode())
    key.update(json.dumps({'config': settings, 'versions': versions}, sort_keys=True).encode())
    for name in sources:
        key.update((Path(__file__).parent / name).read_bytes())
    return key.hexdigest()[:20], versions


def baseline_directory(root, frame, config):
    """Reproduce 10's exact cache fingerprint, without rerunning/rewriting 10."""
    key, _ = _fingerprint(frame, config, SOURCES)
    return Path(root) / 'data/processed/10_candidate_comparison' / key


def _read_prediction(path, expected_ids):
    with np.load(path, allow_pickle=False) as saved:
        if not np.array_equal(saved['job_id'], expected_ids):
            raise ValueError('Prediction cache has different scoring posting IDs.')
        return saved['p'].copy(), json.loads(str(saved['detail'].item()))


def trial_spec(key, config, memory=None):
    if key == 'extra_trees_leaf1':
        model, _ = baseline.candidate_spec('extra_trees', config, memory)
        # A fixed new setting, with the same grouped inner OOF threshold rule.
        return model, {'min_samples_leaf': [1]}
    if key == 'lightgbm_5k':
        return baseline.candidate_spec('lightgbm', replace(config, lightgbm_max_features=5000), memory)
    raise ValueError('Unknown targeted trial.')


def _tables(scores, expected):
    summary, paired = [], []
    for key, table in scores.groupby('key', sort=False):
        summary.append({'key': key, 'model': NAMES[key], 'evaluated_folds': len(table),
            'expected_folds': expected, 'complete': len(table) == expected,
            'tfidf_max_features': int(table.tfidf_max_features.iloc[0]),
            **{metric: table[metric].mean() for metric in baseline.METRICS},
            'ap_sd': table.pr_auc.std(ddof=1), 'minutes_per_fold': table.minutes.mean()})
    for key in TRIALS:
        candidate = scores.loc[scores.key == key]
        if len(candidate) < 2:
            continue
        original = 'extra_trees' if key == 'extra_trees_leaf1' else 'lightgbm'
        for reference in (original, 'mlp_reference'):
            ref = scores.loc[scores.key == reference]
            ap = paired_comparison(candidate, ref, 'pr_auc')
            paired.append({'candidate': NAMES[key], 'reference': NAMES[reference],
                'ap_difference': ap['mean_diff'], 'ap_interval_low': ap['low'],
                'ap_interval_high': ap['high'], 'share_folds_ap_better': ap['share_folds_a_better'],
                **{metric + '_difference': candidate[metric].mean() - ref[metric].mean()
                   for metric in ('precision', 'recall', 'f1')}})
    return pd.DataFrame(summary).sort_values('pr_auc', ascending=False), pd.DataFrame(paired)


def _consensus(y, mlp_p, mlp_cutoff, tree_p, tree_cutoff, identity):
    """Fixed AND of ORIGINAL training-selected flags; no outer-label tuning."""
    mlp, tree = mlp_p >= mlp_cutoff, tree_p >= tree_cutoff
    rows = []
    for policy, flags in [('Original MLP', mlp), ('Both original models flag', mlp & tree)]:
        rows.append({**identity, 'policy': policy,
            'precision': precision_score(y, flags, zero_division=0),
            'recall': recall_score(y, flags), 'f1': f1_score(y, flags),
            'fraud_caught': int((flags & (y == 1)).sum()),
            'false_alarms': int((flags & (y == 0)).sum())})
    return rows


def run_comparison(root, frame, config=None, trials=TRIALS):
    config = config or baseline.ComparisonConfig(outer_limit=5)
    root, trials = Path(root), tuple(trials)
    if not trials or len(set(trials)) != len(trials) or set(trials) - set(TRIALS):
        raise ValueError('Choose unique supported targeted trials.')
    all_folds = baseline.restore_folds(frame, config, root / 'data/processed/model_cv_assignments.csv')
    folds = all_folds if config.outer_limit is None else all_folds[:config.outer_limit]
    old_directory = baseline_directory(root, frame, config)
    # Validate every requested comparator before starting any new fit.
    comparators = []
    for fold in folds:
        original = baseline._reference_predictions(root, frame, config, fold)
        identity = baseline._identity(frame, fold)
        r, k, _, scoring = fold
        for key in ('extra_trees', 'lightgbm'):
            path = old_directory / f'r{r}_f{k}_{key}_{identity["fold_signature"][:16]}.npz'
            if not path.is_file():
                raise FileNotFoundError(f'Missing compatible notebook 10 comparator: {path}. '
                                        'No new training has started.')
            p, detail = _read_prediction(path, frame.iloc[scoring].job_id.to_numpy())
            original[key] = (p, detail, str(path))
        comparators.append(original)
    cache_key, versions = _fingerprint(frame, config, ('d1_targeted_improvements.py', *SOURCES))
    directory = root / 'data/processed/11_targeted_improvements' / cache_key
    directory.mkdir(parents=True, exist_ok=True)
    X = build_inputs(frame.copy())
    y, groups = frame[TARGET].to_numpy(), frame[GROUP].to_numpy()
    rows, policies = [], []
    for fold, original in zip(folds, comparators):
        r, k, fitting, scoring = fold
        identity = baseline._identity(frame, fold)
        for key, (p, detail, path) in original.items():
            rows.append({'key': key, 'model': NAMES[key], **identity,
                **score_summary(y[scoring], p, threshold=detail['threshold']),
                'minutes': detail['minutes'], 'selected_params': detail['params'], 'source': path,
                'tfidf_max_features': config.lightgbm_max_features if key == 'lightgbm' else config.tfidf_max_features})
        pm, dm, _ = original['mlp_reference']
        pt, dt, _ = original['extra_trees']
        policies.extend(_consensus(y[scoring], pm, dm['threshold'], pt, dt['threshold'], identity))
        with tempfile.TemporaryDirectory(prefix='bt4012_targeted_') as temporary:
            memory = Memory(temporary, verbose=0)
            for key in trials:
                path = directory / f'r{r}_f{k}_{key}_{identity["fold_signature"][:16]}.npz'
                if path.is_file():
                    p, detail = _read_prediction(path, frame.iloc[scoring].job_id.to_numpy())
                    print(f'Repeat {r + 1}, fold {k + 1}: {NAMES[key]} — cached', flush=True)
                else:
                    print(f'Repeat {r + 1}, fold {k + 1}: {NAMES[key]} — training', flush=True)
                    started = time.monotonic()
                    pipeline, grid = trial_spec(key, config, memory)
                    with threadpool_limits(limits=config.threads):
                        choice = select_settings(lambda: pipeline, X.iloc[fitting], y[fitting],
                            groups[fitting], grid, inner_splits=config.inner_splits,
                            seed=config.seed + r * 1000 + k)
                        fitted = clone(pipeline).set_params(**choice.params).fit(X.iloc[fitting], y[fitting])
                        p = fitted.predict_proba(X.iloc[scoring])[:, 1]
                    detail = {'params': choice.params, 'threshold': choice.threshold,
                        'selection_pr_auc': float(choice.candidates.mean_inner_pr_auc.max()),
                        'candidates': choice.candidates.to_dict(orient='records'),
                        'minutes': (time.monotonic() - started) / 60}
                    score_summary(y[scoring], p, threshold=detail['threshold'])
                    tmp = path.with_suffix('.tmp')
                    with tmp.open('wb') as output:
                        np.savez_compressed(output, job_id=frame.iloc[scoring].job_id.to_numpy(),
                                            p=p, detail=json.dumps(detail))
                    tmp.replace(path)
                    del fitted
                    gc.collect()
                measured = score_summary(y[scoring], p, threshold=detail['threshold'])
                rows.append({'key': key, 'model': NAMES[key], **identity, **measured,
                    'minutes': detail['minutes'], 'selected_params': detail['params'], 'source': str(path),
                    'tfidf_max_features': 5000 if key == 'lightgbm_5k' else config.tfidf_max_features})
                print(f'  AP {measured["pr_auc"]:.4f}; precision {measured["precision"]:.4f}; '
                      f'recall {measured["recall"]:.4f}; F1 {measured["f1"]:.4f}', flush=True)
        scores = pd.DataFrame(rows)
        summary, paired = _tables(scores, len(all_folds))
        scores.to_csv(directory / 'fold_metrics.csv', index=False)
        summary.to_csv(directory / 'summary.csv', index=False)
        paired.to_csv(directory / 'paired_differences.csv', index=False)
    policy_scores = pd.DataFrame(policies)
    policy_summary = policy_scores.groupby('policy', sort=False)[['precision', 'recall', 'f1']].mean().reset_index()
    # Counts stay separate by repeat; repeated postings are never added together.
    policy_counts = policy_scores.groupby(['policy', 'repeat'], sort=False)[['fraud_caught', 'false_alarms']].sum().reset_index()
    policy_scores.to_csv(directory / 'consensus_fold_metrics.csv', index=False)
    policy_summary.to_csv(directory / 'consensus_summary.csv', index=False)
    policy_counts.to_csv(directory / 'consensus_counts_by_repeat.csv', index=False)
    manifest = {'status': 'exploratory development follow-up; hypotheses informed by notebook 10',
        'config': asdict(config), 'trials': trials, 'versions': versions,
        'training_sha256': hashlib.sha256(frame.to_csv(index=False, lineterminator='\n').encode()).hexdigest(),
        'training_rows': len(frame), 'fraud_rows': int(y.sum()),
        'baseline_directory': str(old_directory), 'evaluated_outer_folds': len(folds),
        'expected_outer_folds': len(all_folds), 'folds': [baseline._identity(frame, f) for f in folds],
        'selection': f'{config.inner_splits} grouped inner folds; AP selects settings; inner OOF F1 selects threshold',
        'extra_trees_trial': 'Leaf size fixed to 1; no other change from 10. Inner OOF predictions select its threshold.',
        'lightgbm_trial': 'TF-IDF ceiling fixed to 5000; leaf grid remains [15,31].',
        'consensus': 'Fixed AND of original MLP and original Extra Trees flags; separate policy, no new AP ranking',
        'aggregation': 'means of matching outer-fold metrics; counts separate by repeat',
        'neural_training_started': False, 'final_test_accessed': False}
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return {'scores': scores, 'summary': summary, 'paired': paired,
            'policy_scores': policy_scores, 'policy_summary': policy_summary, 'policy_counts': policy_counts,
            'manifest': manifest, 'directory': directory}
