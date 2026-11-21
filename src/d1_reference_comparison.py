"""Replay the supplied LightGBM protocol and compare the best calibrated ensemble.

Run with the isolated requirements-13 Python: python -m src.d1_reference_comparison
All 25 saved outer folds are mandatory. Each finished ensemble inner stage and
each outer prediction is checkpointed. Existing development files are untouched.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import tempfile
import time
import traceback

import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from src.reference13_runtime import training_lock, ensure_inputs
from src.d1_advanced_comparison import _ensemble_predictions
from src.d1_candidate_comparison import ComparisonConfig, _identity
from src.evaluation import fold_signature, paired_comparison, score_summary, select_settings, validate_folds
from src.features import GROUP, TARGET, build_inputs
from src.model_comparison import model_spec

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data/processed/13_reference_comparison'
METRICS = ('pr_auc', 'precision', 'recall', 'f1', 'precision_top5pct', 'recall_top5pct')
NAMES = {'lightgbm_replay': 'LightGBM A2 (2k TF-IDF), fresh replay',
         'calibrated_ensemble': 'Calibrated MLP + Extra Trees (50/50)'}
PACKAGES = ('numpy', 'pandas', 'scikit-learn', 'lightgbm', 'xgboost', 'tensorflow',
            'keras', 'scipy', 'joblib', 'threadpoolctl')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def restore_folds(frame, assignments, expected):
    """Restore saved positions by ID and enforce every original fold signature."""
    if not frame.job_id.is_unique or frame.job_id.isna().any():
        raise ValueError('Training job IDs must be unique and present.')
    if assignments.duplicated(['repeat', 'job_id']).any():
        raise ValueError('Duplicate saved assignments.')
    positions = pd.Series(np.arange(len(frame)), index=frame.job_id)
    folds = []
    for (r, k), rows in assignments.groupby(['repeat', 'fold'], sort=True):
        if not rows.job_id.isin(positions.index).all():
            raise ValueError('Unknown assignment IDs.')
        val = np.sort(positions.loc[rows.job_id].to_numpy())
        if not np.array_equal(rows.set_index('job_id').loc[frame.iloc[val].job_id, GROUP],
                              frame.iloc[val][GROUP]):
            raise ValueError('Saved groups differ from reconstructed groups.')
        fit = np.setdiff1d(np.arange(len(frame)), val)
        folds.append((int(r), int(k), fit, val))
    folds = validate_folds(folds, frame[TARGET], frame[GROUP])
    actual = {f'{r}:{k}': fold_signature(frame[TARGET], frame[GROUP], fit, val)
              for r, k, fit, val in folds}
    if actual != expected or len(folds) != 25:
        raise ValueError('The complete original 25 fold signatures must match.')
    return folds


def prepare():
    """Load the verified recovered development pool and original saved folds."""
    reference = ensure_inputs(ROOT, 'fafd14fbcf0257596f2755165c70fa0a69631d7ad0fe1eeca90bac0033346c4a')
    plan = json.loads((reference / 'lightgbm_a2_2k_plan.json').read_text())
    completion = json.loads((reference / 'lightgbm_a2_2k_complete.json').read_text())
    if sha((reference / 'outer_assignments.csv').read_bytes()) != plan['assignments_sha256']:
        raise ValueError('Original assignment checksum mismatch.')
    if sha((reference / 'lightgbm_a2_2k_fold_scores.csv').read_bytes()) != completion['scores_sha256']:
        raise ValueError('Historical score checksum mismatch.')
    for name, expected in plan['source_sha256'].items():
        if sha((ROOT / 'src' / name).read_bytes()) != expected:
            raise ValueError(f'Historical pipeline source changed: {name}')
    assignments = pd.read_csv(reference / 'outer_assignments.csv')
    provenance = json.loads((reference / 'recovery.json').read_text())
    content = (reference / 'recovered_train.csv').read_bytes()
    if sha(content) != provenance['recovered_training_file_sha256']:
        raise ValueError('Recovered training CSV checksum mismatch.')
    frame = pd.read_csv(io.BytesIO(content))
    counts = (len(frame), int(frame[TARGET].sum()), frame[GROUP].nunique(),
              frame.loc[frame[TARGET].eq(1), GROUP].nunique())
    if counts != (14439, 693, 3555, 331):
        raise ValueError(f'Report training counts differ: {counts}')
    folds = restore_folds(frame, assignments, plan['outer_signatures'])
    historical = pd.read_csv(reference / 'lightgbm_a2_2k_fold_scores.csv')
    if historical.duplicated(['repeat', 'fold']).any() or len(historical) != 25:
        raise ValueError('Historical score table is incomplete.')
    for fold in folds:
        identity = _identity(frame, fold)
        row = historical.loc[historical['repeat'].eq(fold[0]) & historical.fold.eq(fold[1])].iloc[0]
        if any(row[k] != v for k, v in identity.items()):
            raise ValueError('Historical score identities differ.')
    return frame, folds, plan, provenance


def runtime_context(plan, provenance, config):
    versions = {p: importlib.metadata.version(p) for p in PACKAGES}
    for name, version in plan['versions'].items():
        if versions[name] != version:
            raise ValueError(f'Use requirements-13 in the isolated environment: {name} needs {version}, found {versions[name]}.')
    sources = {p.name: sha(p.read_bytes()) for p in sorted((ROOT / 'src').glob('*.py'))}
    installed = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
    return {'config': asdict(config), 'versions': versions, 'installed_packages': installed, 'sources': sources,
            'provenance': provenance, 'methods': list(NAMES),
            'selection': 'Three grouped inner folds, seed 42 + repeat*1000 + fold; mean inner AP selects LightGBM leaves [15,31]; inner OOF maximum F1 selects thresholds for both.',
            'ensemble': 'Original best fixed 50/50 MLP + leaf-1 Extra Trees; 30k TF-IDF; each fit holds out 1/5 grouped rows for natural-prevalence sigmoid calibration; no post-calibration refit.',
            'mlp': 'Original 512/256/128/64 architecture; batch64, lr.001, max50 epochs, patience5, grouped checkpoint then refit.',
            'aggregation': 'Unweighted mean and sample SD of the same 25 outer fold scores; AP means average precision, not trapezoidal PR area.',
            'pipeline_comparison': 'A2 feature blocks in both; historical LightGBM uses 2k vocabulary, the previously selected ensemble uses 30k. This compares complete pipelines, not only algorithms.'}


def save_prediction(path, ids, p, detail):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, job_id=ids, p=p, detail=json.dumps(detail))
    tmp.replace(path)


def read_prediction(path, ids):
    with np.load(path, allow_pickle=False) as saved:
        if not np.array_equal(saved['job_id'], ids):
            raise ValueError('Cached prediction IDs differ.')
        return saved['p'].copy(), json.loads(saved['detail'].item())


def report(directory, rows, historical, status):
    scores = pd.DataFrame(rows)
    scores.to_csv(directory / 'fold_metrics.csv', index=False)
    summary_rows, paired_rows = [], []
    if len(scores):
        for key, table in scores.groupby('key', sort=False):
            summary_rows.append({'model': NAMES[key], 'key': key, 'completed_folds': len(table),
                'expected_folds': 25, 'complete': len(table) == 25,
                **{m: table[m].mean() for m in METRICS},
                **{m + '_sd': table[m].std(ddof=1) for m in METRICS},
                'minutes_per_fold': table.minutes.mean()})
        light = scores.loc[scores.key.eq('lightgbm_replay')]
        ensemble = scores.loc[scores.key.eq('calibrated_ensemble')]
        common = light[['repeat', 'fold']].merge(ensemble[['repeat', 'fold']])
        if len(common) >= 2:
            for metric in METRICS:
                difference = paired_comparison(ensemble.merge(common), light.merge(common), metric)
                paired_rows.append({'metric': metric, 'matching_folds': len(common), **difference})
        replay = light.merge(historical, on=['repeat', 'fold'], suffixes=('_replay', '_historical'))
        for m in METRICS:
            replay[m + '_difference'] = replay[m + '_replay'] - replay[m + '_historical']
        replay.to_csv(directory / 'historical_replay_check.csv', index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(directory / 'summary.csv', index=False)
    pd.DataFrame(paired_rows).to_csv(directory / 'paired_differences.csv', index=False)
    counts = {key: sum(row['key'] == key for row in rows) for key in NAMES}
    write_json(directory / 'progress.json', {'status': status, 'completed_folds': counts,
        'expected_folds_per_model': 25, 'updated_utc': datetime.now(timezone.utc).isoformat(),
        'pid': os.getpid(), 'complete': all(n == 25 for n in counts.values())})
    html = '<!doctype html><meta charset="utf-8"><title>13: matched comparison</title>'
    html += '<h1>LightGBM versus calibrated MLP + Extra Trees</h1>'
    html += f'<p>Status: {status}. Both pipelines must complete 25 matching folds before final interpretation.</p>'
    html += '<p>Recovered development pool: 14,439 postings, 693 frauds, 3,555 groups. Three grouped inner folds select settings and thresholds. No reserved test predictions.</p>'
    html += '<p>PR-AUC is implemented as average precision (AP). Main metrics: AP, fraud precision, recall, F1; top-5% review measures are secondary. Table means and SDs are across folds.</p>'
    html += summary.to_html(index=False, float_format=lambda x: f'{x:.4f}')
    html += '<p>Fresh LightGBM replay is the paired comparator. Historical results are audited separately because the original CSV file bytes could not be reproduced. Vocabulary sizes retain the selected pipelines: LightGBM 2k, ensemble 30k.</p>'
    html += pd.DataFrame(paired_rows).to_html(index=False, float_format=lambda x: f'{x:.4f}')
    html += '<p>Paired intervals use a repeated-CV correction and are approximate; folds are dependent. Prior model exploration means this is development evidence, not a new independent test.</p>'
    (directory / 'comparison.html').write_text(html)


def run(prepare_only=False):
    frame, folds, plan, provenance = prepare()
    if prepare_only:
        print(json.dumps(provenance, indent=2), flush=True)
        return
    config = ComparisonConfig(outer_limit=None)
    context = runtime_context(plan, provenance, config)
    fingerprint = sha(json.dumps(context, sort_keys=True).encode())[:20]
    directory = BASE / 'runs' / fingerprint
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / 'manifest.json', context)
    write_json(BASE / 'active_run.json', {'directory': str(directory), 'command': 'python -u -m src.d1_reference_comparison'})
    historical = pd.read_csv(BASE / 'reference/lightgbm_a2_2k_fold_scores.csv')
    X = build_inputs(frame.copy())
    y, groups = frame[TARGET].to_numpy(), frame[GROUP].to_numpy()
    rows = []
    print(f'Output: {directory}\nValidated all 25 saved folds; starting both complete pipelines.', flush=True)
    report(directory, rows, historical, 'running')
    try:
        for fold in folds:
            r, k, fit, val = fold
            identity = _identity(frame, fold)
            for key in NAMES:
                path = directory / key / f'r{r}_f{k}.npz'
                print(f'Repeat {r+1}/5 fold {k+1}/5: {NAMES[key]}', flush=True)
                if path.exists():
                    p, detail = read_prediction(path, frame.iloc[val].job_id.to_numpy())
                    print('  Reusing completed fold.', flush=True)
                else:
                    started = time.monotonic()
                    with threadpool_limits(limits=config.threads):
                        if key == 'calibrated_ensemble':
                            p, detail = _ensemble_predictions(frame, fold, config,
                                directory / key / f'r{r}_f{k}_stages')
                        else:
                            with tempfile.TemporaryDirectory(dir=BASE, prefix='features_') as cache:
                                pipeline, grid, _ = model_spec('lightgbm_a2', Memory(cache, verbose=0), config.threads)
                                choice = select_settings(lambda: pipeline, X.iloc[fit], y[fit], groups[fit], grid,
                                    inner_splits=3, seed=42 + r * 1000 + k)
                                fitted = clone(pipeline).set_params(**choice.params).fit(X.iloc[fit], y[fit])
                                p = fitted.predict_proba(X.iloc[val])[:, 1]
                                detail = {'params': choice.params, 'threshold': choice.threshold,
                                    'candidates': choice.candidates.to_dict(orient='records')}
                    detail['minutes'] = (time.monotonic() - started) / 60
                    score_summary(y[val], p, threshold=detail['threshold'])
                    save_prediction(path, frame.iloc[val].job_id.to_numpy(), p, detail)
                measured = score_summary(y[val], p, threshold=detail['threshold'])
                rows.append({'key': key, **identity, **measured, 'minutes': detail['minutes'],
                             'selected_params': json.dumps(detail['params'])})
                report(directory, rows, historical, 'running')
                print('  ' + '; '.join(f'{m}={measured[m]:.4f}' for m in METRICS[:4]), flush=True)
        report(directory, rows, historical, 'complete')
        print(f'COMPLETE: {directory / "comparison.html"}', flush=True)
    except BaseException:
        report(directory, rows, historical, 'interrupted or failed; resume with the same command')
        (directory / 'error.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    BASE.mkdir(parents=True, exist_ok=True)
    try:
        with training_lock(BASE / 'run.lock'):
            run(args.prepare_only)
    except BlockingIOError:
        raise SystemExit('Experiment 13 is already running; read progress.json instead.')
