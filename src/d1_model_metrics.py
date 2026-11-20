"""Build 02-aligned metric tables, with optional file exports.

No fitting or final-test access. Historical results are excluded rather than
silently mixed when data or fold identities differ.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import d1_a2_mlp as experiment
from src.evaluation import fold_signature, paired_comparison

METRICS = ['pr_auc', 'roc_auc', 'precision', 'recall', 'f1', 'false_positive_rate',
           'precision_top5pct', 'recall_top5pct']
IDENTITY = ['fold_signature', 'train_rows', 'val_rows']


def _align(scores, reference):
    required = {'repeat', 'fold', *IDENTITY, *METRICS,
                'fraud_caught', 'fraud_missed', 'genuine_flagged', 'genuine_passed'}
    if not required.issubset(scores):
        raise ValueError('Missing fold-level metrics or identities.')
    table = scores.set_index(['repeat', 'fold']).sort_index()
    expected = reference.set_index(['repeat', 'fold']).sort_index()
    if table.index.has_duplicates or not expected.index.isin(table.index).all():
        raise ValueError('Missing or duplicate requested scoring folds.')
    table = table.loc[expected.index]
    if not table[IDENTITY].equals(expected[IDENTITY]):
        raise ValueError('Different scoring folds, labels/groups or training-pool size.')
    return table.reset_index()


def compare_models(root, frame, config, result):
    """Return comparison and tuning tables in memory without writing any files."""
    return export_comparison(root, frame, config, result, write_files=False)


def export_comparison(root, frame, config, result, output_dir=None, *, write_files=True):
    """Use all currently completed 09 folds; require each other model to match them."""
    root = Path(root)
    output = Path(output_dir) if output_dir else root / 'reports/tables/09_best_model'
    current = result['scores'].copy()
    reference = current.loc[current.model == experiment.MLP_NAME]
    if reference.empty or reference.duplicated(['repeat', 'fold']).any():
        raise ValueError('Need unique completed MLP scoring folds.')
    if Path(result['cache_dir']).name != experiment._cache_key(frame, config):
        raise ValueError('Current results do not match the supplied data/configuration.')
    y, groups = frame.fraudulent.to_numpy(), frame.group_id.to_numpy()
    planned = {(r, k): (fold_signature(y, groups, fit, score), len(fit), len(score))
               for r, k, fit, score in experiment.split_plan(frame, config)}
    for row in reference.itertuples():
        if planned.get((row.repeat, row.fold)) != (row.fold_signature, row.train_rows, row.val_rows):
            raise ValueError('Current results contain folds outside the configured plan.')
    included, sources = [], []

    def include(table, name, source, features, limit):
        aligned = _align(table, reference)
        aligned['model'], aligned['source'] = name, source
        aligned['feature_set'], aligned['tfidf_max_features'] = features, limit
        included.append(aligned)
        sources.append({'model': name, 'source': source, 'status': 'included',
                        'reason': f'{len(aligned)} matching outer folds'})

    for name in [experiment.MLP_NAME, experiment.LR_NAME]:
        include(current.loc[current.model == name], name, '09', 'A2', config.tfidf_max_features)

    external = [('02 TF-IDF logistic regression', root / 'reports/tables/model_logreg_tf_idf_only_fold_scores.csv',
                 '02', 'TF-IDF only', 30000)]
    data_file = root / 'data/processed/d1_train.csv'
    file_sha = hashlib.sha256(data_file.read_bytes()).hexdigest()
    for name, path, source, features, limit in external:
        reason = None
        if not path.is_file():
            reason = 'Saved model fold scores not present; rerun its notebook to include it.'
        if reason is None:
            try:
                include(pd.read_csv(path), name, source, features, limit)
            except ValueError as exc:
                reason = str(exc)
        if reason is not None:
            sources.append({'model': name, 'source': source, 'status': 'excluded', 'reason': reason})

    folds = pd.concat(included, ignore_index=True)
    summaries, repeated = [], []
    expected_count = config.outer_splits * config.repeats
    for name, table in folds.groupby('model', sort=False):
        first = table.iloc[0]
        summary = {'model': name, 'source': first.source, 'feature_set': first.feature_set,
                   'tfidf_max_features': int(first.tfidf_max_features), 'evaluated_folds': len(table),
                   'expected_folds': expected_count, 'complete': len(table) == expected_count,
                   'evaluated_repeats': table.repeat.nunique()}
        summary.update({metric: float(table[metric].mean()) for metric in METRICS})
        summary['pr_auc_sd'] = float(table.pr_auc.std(ddof=1))
        summaries.append(summary)
        for repeat, rows in table.groupby('repeat'):
            tp, fn = int(rows.fraud_caught.sum()), int(rows.fraud_missed.sum())
            fp, tn = int(rows.genuine_flagged.sum()), int(rows.genuine_passed.sum())
            repeated.append({'model': name, 'repeat': int(repeat) + 1,
                'evaluated_rows': int(rows.val_rows.sum()), 'fraud_total': tp + fn,
                'precision': tp / max(tp + fp, 1), 'recall': tp / max(tp + fn, 1),
                'f1': 2 * tp / max(2 * tp + fp + fn, 1), 'fraud_caught': tp,
                'fraud_missed': fn, 'genuine_flagged': fp, 'false_positive_rate': fp / max(fp + tn, 1)})
    summary = pd.DataFrame(summaries)
    tuning = current.loc[current.model == experiment.MLP_NAME,
        ['repeat', 'fold', 'selected_params', 'selected_epoch', 'threshold', 'selection_pr_auc']].copy()
    tuning[['repeat', 'fold']] += 1
    tuning['selected_params'] = tuning.selected_params.map(lambda value: json.dumps(value, sort_keys=True))
    paired = []
    if len(reference) >= 2:
        for name, table in folds.groupby('model', sort=False):
            if name != experiment.MLP_NAME:
                for metric in ['pr_auc', 'precision', 'recall', 'f1']:
                    paired.append({'model': experiment.MLP_NAME, 'reference': name, 'metric': metric,
                                   **paired_comparison(reference, table, metric)})
    tables = {'model_comparison': summary, 'combined_decisions_by_repeat': pd.DataFrame(repeated),
              'fold_metrics': folds, 'mlp_tuning_choices': tuning,
              'comparison_sources': pd.DataFrame(sources), 'paired_mlp_vs_others': pd.DataFrame(paired)}
    if not write_files:
        return tables
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        saved = table.copy()
        if name == 'fold_metrics':
            saved['selected_params'] = saved.selected_params.map(
                lambda value: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value)
        saved.to_csv(output / f'{name}.csv', index=False)
    provenance = {'training_file_sha256': file_sha, 'training_rows': len(frame),
        'fraud_rows': int(y.sum()), 'configuration': asdict(config), 'completed_outer_folds': len(reference),
        'expected_outer_folds': expected_count, 'final_test_accessed': False,
        'summary_aggregation': 'arithmetic mean of matched outer-fold metrics; AP is average precision',
        'count_aggregation': 'combined decisions separately per repeat; repeated postings are never added across repeats',
        'comparison_note': '02-aligned comparison only: A2 MLP, A2 LR and optional matching 02 text-only LR.',
        'versions': experiment._versions()}
    (output / 'run_manifest.json').write_text(json.dumps(provenance, indent=2))
    tables['output_dir'] = str(output)
    return tables
