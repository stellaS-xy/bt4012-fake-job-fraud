"""Run and report the repeated comparison without opening the reserved test set.

Run from the repository root: python -m src.model_comparison
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile
import time

from joblib import Memory
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from src.evaluation import cross_validate, fold_signature, make_folds, paired_comparison
from src.features import GROUP, SEED, TARGET, add_country, build_inputs, logistic_pipeline, make_features

ROOT = Path(__file__).resolve().parents[1]
MODEL_KEYS = ('logreg_tfidf', 'logreg_a2', 'lightgbm_a2', 'xgboost_a2')
MODEL_NAMES = {
    'logreg_tfidf': 'Logistic regression: text only',
    'logreg_a2': 'Logistic regression: A2',
    'lightgbm_a2': 'LightGBM: A2',
    'xgboost_a2': 'XGBoost: A2',
}


class BalancedClassifier(ClassifierMixin, BaseEstimator):
    """Compute class-balancing weights from the rows of each individual fit."""
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, sample_weight=compute_sample_weight('balanced', y))
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


def model_spec(key, memory=None, threads=2):
    """Small grids fixed before looking at outer results; no validation early stopping."""
    if key == 'logreg_tfidf':
        pipeline = logistic_pipeline(['tfidf'])
        pipeline.set_params(memory=memory, model__random_state=SEED)
        return pipeline, {'model__C': [0.1, 1.0]}, 30000
    if key == 'logreg_a2':
        pipeline = logistic_pipeline(['raw', 'engineered', 'text_style', 'tfidf'], tfidf_max_features=2000)
        pipeline.set_params(memory=memory, model__random_state=SEED)
        return pipeline, {'model__C': [0.1, 1.0]}, 2000
    if key == 'lightgbm_a2':
        from lightgbm import LGBMClassifier
        estimator = LGBMClassifier(n_estimators=200, learning_rate=0.05, max_bin=63,
                                   random_state=SEED, n_jobs=threads, verbosity=-1,
                                   deterministic=True, force_col_wise=True)
        grid = {'model__estimator__num_leaves': [15, 31]}
    elif key == 'xgboost_a2':
        from xgboost import XGBClassifier
        estimator = XGBClassifier(n_estimators=200, learning_rate=0.05, max_bin=64,
                                  tree_method='hist', eval_metric='logloss',
                                  random_state=SEED, n_jobs=threads)
        grid = {'model__estimator__max_depth': [3, 5]}
    else:
        raise ValueError(f'Unknown model: {key}')
    return Pipeline([('features', make_features('A2', tfidf_max_features=2000)),
                     ('model', BalancedClassifier(estimator))], memory=memory), grid, 2000


def _write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def run_one(key, X, y, groups, job_ids, folds, output, processed, inner_splits=3, threads=2):
    """Save each finished fold, plus final results; cache only within one outer fold."""
    print(f'\nStarting {MODEL_NAMES[key]}', flush=True)
    started = time.perf_counter()
    checkpoint, predictions = [], []
    with tempfile.TemporaryDirectory(prefix='bt4012_features_') as cache:
        memory = Memory(cache, verbose=0)
        pipeline, grid, _ = model_spec(key, memory, threads)

        def progress(row, val_idx, probabilities):
            saved = dict(row)
            saved['selected_params'] = json.dumps(saved['selected_params'], sort_keys=True)
            checkpoint.append(saved)
            pd.DataFrame(checkpoint).to_csv(output / f'{key}_fold_scores.csv', index=False)
            predictions.append(pd.DataFrame({
                'repeat': row['repeat'], 'fold': row['fold'], 'job_id': job_ids[val_idx],
                'probability': probabilities, 'threshold': row['threshold'],
                'flagged': probabilities >= row['threshold'],
            }))
            pd.concat(predictions, ignore_index=True).to_csv(processed / f'{key}_predictions.csv', index=False)
            _write_json(output / f'{key}_progress.json', {
                'model': key, 'completed_folds': len(checkpoint), 'total_folds': len(folds),
                'elapsed_seconds': round(time.perf_counter() - started, 1),
            })
            memory.clear(warn=False)

        with threadpool_limits(limits=threads):
            scores, _ = cross_validate(lambda: pipeline, X, y, groups, folds=folds, param_grid=grid,
                                       inner_splits=inner_splits, seed=SEED, on_fold=progress)
        candidates = scores.attrs['inner_candidates'].copy()
        candidates['params'] = candidates['params'].map(lambda p: json.dumps(p, sort_keys=True))
        candidates.to_csv(output / f'{key}_inner_settings.csv', index=False)
    _write_json(output / f'{key}_complete.json', {
        'model': key, 'folds': len(scores), 'elapsed_seconds': round(time.perf_counter() - started, 1),
    })
    print(f'Completed {key} in {(time.perf_counter() - started) / 60:.1f} minutes', flush=True)


def repeat_summary(scores):
    """One count of every ad per repeat; rates come from summed confusion counts."""
    rows = []
    count_cols = ['fraud_caught', 'fraud_missed', 'genuine_flagged', 'genuine_passed',
                  'review_count', 'review_fraud_caught', 'review_genuine_flagged']
    for repeat, folds in scores.groupby('repeat', sort=True):
        row = {c: int(folds[c].sum()) for c in count_cols}
        fraud = row['fraud_caught'] + row['fraud_missed']
        genuine = row['genuine_flagged'] + row['genuine_passed']
        flagged = row['fraud_caught'] + row['genuine_flagged']
        row.update(repeat=int(repeat), fraud_total=fraud, genuine_total=genuine,
                   recall=row['fraud_caught'] / fraud, false_positive_rate=row['genuine_flagged'] / genuine,
                   precision=row['fraud_caught'] / max(flagged, 1),
                   review_precision=row['review_fraud_caught'] / row['review_count'],
                   review_recall=row['review_fraud_caught'] / fraud,
                   mean_fold_pr_auc=folds['pr_auc'].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def report_results(output):
    """Build tables and an HTML report from completed runs, without fitting or reading test data."""
    output = Path(output)
    manifest = json.loads((output / 'run_plan.json').read_text(encoding='utf-8'))
    keys = manifest['models']
    scores = {key: pd.read_csv(output / f'{key}_fold_scores.csv') for key in keys}
    repeats = {key: repeat_summary(table) for key, table in scores.items()}
    pd.concat([r.assign(model=key) for key, r in repeats.items()], ignore_index=True).to_csv(
        output / 'results_by_repeat.csv', index=False)
    summary = pd.DataFrame({key: r.drop(columns='repeat').mean() for key, r in repeats.items()}).T
    summary.index.name = 'model'
    summary['pr_auc_sd_across_folds'] = pd.Series({key: s['pr_auc'].std(ddof=1) for key, s in scores.items()})
    summary.to_csv(output / 'comparison_summary.csv')
    paired = []
    metrics = {'pr_auc': True, 'recall': True, 'false_positive_rate': False,
               'precision_top5pct': True, 'recall_top5pct': True}
    for reference in ('logreg_tfidf', 'logreg_a2'):
        for key in keys:
            if key == reference:
                continue
            for metric, higher in metrics.items():
                comparison = paired_comparison(scores[key], scores[reference], metric=metric, higher_is_better=higher)
                paired.append({'model': key, 'reference': reference, 'metric': metric,
                               'higher_is_better': higher, **comparison})
    paired = pd.DataFrame(paired)
    paired.to_csv(output / 'paired_differences.csv', index=False)
    repeat_deltas = []
    for key in keys:
        if key == 'logreg_tfidf':
            continue
        for repeat in repeats[key]['repeat']:
            a = repeats[key].set_index('repeat').loc[repeat]
            b = repeats['logreg_tfidf'].set_index('repeat').loc[repeat]
            repeat_deltas.append({'model': key, 'repeat': repeat,
                                  **{f'{metric}_difference': float(a[metric] - b[metric]) for metric in (
                                      'mean_fold_pr_auc', 'recall', 'false_positive_rate', 'review_precision', 'review_recall')}})
    pd.DataFrame(repeat_deltas).to_csv(output / 'repeat_differences_vs_baseline.csv', index=False)
    _plot(summary, repeats, paired, output)
    friendly = pd.DataFrame(index=[MODEL_NAMES[k] for k in keys])
    friendly['Fake ads caught (of %d)' % manifest['fraud_rows']] = [f'{summary.loc[k,"fraud_caught"]:.1f}' for k in keys]
    friendly['Genuine ads wrongly flagged (of %d)' % manifest['genuine_rows']] = [f'{summary.loc[k,"genuine_flagged"]:.1f}' for k in keys]
    friendly['Flags that are fake'] = [f'{summary.loc[k,"precision"]:.1%}' for k in keys]
    friendly['Fake ads found in 5% review'] = [f'{summary.loc[k,"review_fraud_caught"]:.1f}' for k in keys]
    friendly['5% review: flags that are fake'] = [f'{summary.loc[k,"review_precision"]:.1%}' for k in keys]
    friendly['Mean PR-AUC'] = [f'{summary.loc[k,"mean_fold_pr_auc"]:.3f}' for k in keys]
    friendly.to_csv(output / 'plain_language_summary.csv')
    intervals = paired[(paired['reference'] == 'logreg_tfidf') & (paired['metric'] == 'pr_auc')].copy()
    intervals['conclusion'] = np.where(intervals['low'] > 0, 'Evidence of better ranking in these tests',
                                       np.where(intervals['high'] < 0, 'Evidence of worse ranking in these tests',
                                                'Difference remains uncertain'))
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Fake job model comparison</title>
<style>body{font:16px/1.6 system-ui;max-width:1250px;margin:40px auto;padding:0 24px;color:#172b3a}
table{border-collapse:collapse;font-size:14px;width:100%;margin:24px 0}td,th{padding:10px;border-bottom:1px solid #ccc;text-align:right}
th:first-child,td:first-child{text-align:left}img{max-width:100%}h1,h2{line-height:1.2}</style>
<h1>Which models catch fake job ads?</h1>'''
    html += f'<p>Completed {manifest["repeats"]} repeats of {manifest["splits"]} grouped tests: {manifest["repeats"]*manifest["splits"]} scoring rounds per model. Each repeat scores the same {manifest["training_rows"]:,} ads once. Related ads stay together. Settings and cutoffs are chosen using {manifest["inner_splits"]} smaller grouped splits of training ads only. The reserved final test file was never opened.</p>'
    html += '<p>Counts below are averages across repeats. The same ads are reused across repeats, so these are not extra independent datasets. The ordinary flags use each round’s training-chosen cutoff. The 5% review columns instead take the highest scores in each scoring batch; they are a separate review policy.</p>'
    html += friendly.to_html(escape=True)
    html += '<p>The text-only reference keeps the existing 30,000-term vocabulary limit. All three A2 models use the same 2,000-term limit plus the same structured and writing-style features, to keep tree training practical. This compares complete pipelines; use A2 logistic regression for a comparison with identical inputs to the tree models.</p>'
    html += '<img src="comparison.png" alt="Fraud caught, false alarms and review precision across repeats">'
    html += '<h2>Does an improvement hold up?</h2><p>Positive PR-AUC differences mean better ranking than the text-only reference. An interval crossing zero means these tests do not clearly establish a difference. Intervals are approximate and allow for overlap in training data; they are exploratory, not adjusted for all comparisons. Error bars in the chart show the range across repeats, not confidence intervals.</p>'
    html += intervals[['model', 'mean_diff', 'low', 'high', 'share_folds_a_better', 'conclusion']].to_html(index=False, float_format=lambda x:f'{x:.3f}', escape=True)
    html += '<img src="paired_uncertainty.png" alt="Approximate uncertainty in ranking differences">'
    html += '<h2>Results in each repeat</h2>' + pd.concat([r.assign(model=key) for key, r in repeats.items()], ignore_index=True).to_html(index=False, float_format=lambda x:f'{x:.3f}', escape=True)
    html += '<p>Detailed files: comparison_summary.csv, results_by_repeat.csv, paired_differences.csv, repeat_differences_vs_baseline.csv, each model’s fold scores and selected settings, and run_plan.json. Row-level predictions remain in the ignored processed-data folder. No final model has been selected or tested.</p></html>'
    (output / 'report.html').write_text(html, encoding='utf-8')
    print('\n' + friendly.to_string(), flush=True)
    print('\n' + intervals[['model','mean_diff','low','high','conclusion']].to_string(index=False), flush=True)
    return summary, paired


def _plot(summary, repeats, paired, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys = list(summary.index)
    labels = ['Text-only\nlogistic', 'A2\nlogistic', 'A2\nLightGBM', 'A2\nXGBoost']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, metric, title in zip(axes, ('recall', 'false_positive_rate', 'review_precision'),
                                 ('Fake ads caught', 'Genuine ads wrongly flagged', 'Within the 5% sent for review: fake ads')):
        means = summary.loc[keys, metric].to_numpy() * 100
        low = np.array([repeats[k][metric].min() for k in keys]) * 100
        high = np.array([repeats[k][metric].max() for k in keys]) * 100
        ax.bar(labels, means, color=['#6c8397','#4b9c91','#db9862','#7a73a5'],
               yerr=np.stack([means-low,high-means]), capsize=3)
        for i, k in enumerate(keys):
            ax.scatter(np.full(len(repeats[k]), i), repeats[k][metric]*100, color='#172b3a', s=12, zorder=3)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel('Percent')
        ax.set_ylim(bottom=0)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Same ads and testing rounds for every model; dots are separate repeats')
    fig.tight_layout()
    fig.savefig(output / 'comparison.png', dpi=180)
    plt.close(fig)
    p = paired[(paired.reference == 'logreg_tfidf') & (paired.metric == 'pr_auc')]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.errorbar(p.mean_diff, [MODEL_NAMES[k] for k in p.model],
                xerr=np.vstack([p.mean_diff-p.low,p.high-p.mean_diff]), fmt='o', capsize=4, color='#3d7c83')
    ax.axvline(0, color='grey', linestyle='--')
    ax.set(xlabel='PR-AUC difference from text-only logistic regression',
           title='Is the ranking improvement clear? Approximate 95% intervals')
    fig.tight_layout()
    fig.savefig(output / 'paired_uncertainty.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--splits', type=int, default=5)
    parser.add_argument('--inner-splits', type=int, default=3)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports' / 'model_comparison')
    parser.add_argument('--model', choices=MODEL_KEYS)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        report_results(args.output)
        return
    train_path = ROOT / 'data' / 'processed' / 'd1_train.csv'
    train = add_country(pd.read_csv(train_path))
    if not train['job_id'].is_unique:
        raise ValueError('Training job IDs must be unique.')
    X, y, groups = build_inputs(train), train[TARGET].to_numpy(), train[GROUP].to_numpy()
    folds = make_folds(groups, y, args.splits, args.repeats, SEED)
    processed = ROOT / 'data' / 'processed' / 'model_comparison'
    processed.mkdir(exist_ok=True)
    manifest = {
        'seed': SEED, 'splits': args.splits, 'repeats': args.repeats, 'inner_splits': args.inner_splits,
        'threads': args.threads, 'models': list(MODEL_KEYS), 'training_rows': len(train),
        'fraud_rows': int(y.sum()), 'genuine_rows': int((y==0).sum()),
        'training_sha256': hashlib.sha256(train_path.read_bytes()).hexdigest(),
        'source_sha256': {name: hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()
                          for name in ('model_comparison.py', 'evaluation.py', 'features.py')},
        'versions': {p: importlib.metadata.version(p) for p in ('numpy','pandas','scikit-learn','lightgbm','xgboost')},
        'model_specs': {key: {'parameters': model_spec(key, threads=args.threads)[0].get_params(deep=False)['steps'][-1][1].get_params(),
                              'grid': model_spec(key, threads=args.threads)[1],
                              'tfidf_max_features': model_spec(key, threads=args.threads)[2]} for key in MODEL_KEYS},
    }
    # Native estimators inside the weighting wrapper need a JSON representation.
    manifest = json.loads(json.dumps(manifest, default=str), parse_constant=str)
    plan_path = args.output / 'run_plan.json'
    if plan_path.exists():
        old = json.loads(plan_path.read_text(encoding='utf-8'))
        if {k:v for k,v in old.items() if k != 'created_utc'} != manifest:
            raise ValueError('The existing run uses different data, code or settings. Use another output folder.')
    else:
        _write_json(plan_path, {**manifest, 'created_utc': datetime.now(timezone.utc).isoformat()})
    fold_rows, assignments = [], []
    for r, k, fit, val in folds:
        fold_rows.append({'repeat':r,'fold':k,'train_rows':len(fit),'val_rows':len(val),
                          'fold_signature':fold_signature(y,groups,fit,val)})
        assignments.append(pd.DataFrame({'repeat':r,'fold':k,'job_id':train.iloc[val]['job_id'].to_numpy(),
                                          'group_id':groups[val]}))
    pd.DataFrame(fold_rows).to_csv(args.output/'fold_plan.csv',index=False)
    pd.concat(assignments,ignore_index=True).to_csv(processed/'fold_assignments.csv',index=False)
    print(f'{len(train):,} training ads; {len(folds)} rounds per model. Reserved test file stays closed.',flush=True)
    for key in ([args.model] if args.model else MODEL_KEYS):
        if (args.output/f'{key}_complete.json').exists():
            print(f'Already completed: {key}',flush=True)
            continue
        run_one(key,X,y,groups,train['job_id'].to_numpy(),folds,args.output,processed,args.inner_splits,args.threads)
    if all((args.output/f'{k}_complete.json').exists() for k in MODEL_KEYS):
        report_results(args.output)


if __name__ == '__main__':
    main()
