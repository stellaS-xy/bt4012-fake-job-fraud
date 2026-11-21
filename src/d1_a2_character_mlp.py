"""Matched character-TF-IDF follow-up to 08's unchanged grouped A2 reference.

Reuse d1_a2_mlp's checkpoint/refit procedure without editing its source or
invalidating its completed fold caches. This module never reads final-test data.
"""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from src import d1_a2_mlp as reference
from src.evaluation import (fold_signature, paired_comparison, score_summary,
                            select_settings, summarise, validate_folds)
from src.features import GROUP, TARGET, build_inputs

CHAR_NAME = 'A2 + character deep MLP'


class CharacterTfidf(TransformerMixin, BaseEstimator):
    """Learn character vocabulary only from the normalized text of fitting rows."""
    def __init__(self, max_features=10000):
        self.max_features = max_features

    def fit(self, X, y=None):
        self.vectorizer_ = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5),
            min_df=3, max_features=self.max_features, sublinear_tf=True, dtype=np.float32)
        self.vectorizer_.fit(X['text_clean'])
        return self

    def transform(self, X):
        check_is_fitted(self, 'vectorizer_')
        return self.vectorizer_.transform(X['text_clean'])

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'vectorizer_')
        return self.vectorizer_.get_feature_names_out()


class A2CharacterMLP(reference.A2MLP):
    """The same hidden layers/selection procedure, with one extra feature branch."""
    def __init__(self, learning_rate=.001, max_epochs=50, patience=5, batch_size=64,
                 checkpoint_splits=5, tfidf_max_features=30000, dense_weight=1., seed=42,
                 char_features=10000):
        super().__init__(learning_rate=learning_rate, max_epochs=max_epochs, patience=patience,
            batch_size=batch_size, checkpoint_splits=checkpoint_splits,
            tfidf_max_features=tfidf_max_features, dense_weight=dense_weight, seed=seed)
        self.char_features = char_features

    def _features(self):
        return FeatureUnion([('a2', super()._features()),
                             ('character', CharacterTfidf(self.char_features))])


def _validate_reference(frame, config, folds, result):
    """Require the same data, feature/training settings and outer scoring rows."""
    if Path(result['cache_dir']).name != reference._cache_key(frame, config):
        raise ValueError('Reference data/settings/source versions differ. Run the reference cells with the same CONFIG.')
    y, groups = frame[TARGET].to_numpy(), frame[GROUP].to_numpy()
    expected = {(r, k): (fold_signature(y, groups, fit, score), len(fit), len(score))
                for r, k, fit, score in folds}
    for name in [reference.LR_NAME, reference.MLP_NAME]:
        rows = result['scores'].loc[result['scores'].model == name]
        actual = {(row.repeat, row.fold): (row.fold_signature, row.train_rows, row.val_rows)
                  for row in rows.itertuples()}
        if len(rows) != len(expected) or actual != expected:
            raise ValueError('Reference and character trial must evaluate the same requested outer folds.')


def run_comparison(frame, config, reference_result, cache_dir, folds=None, char_features=10000):
    """Evaluate one fixed character-feature addition; tune only inside training.

    The reference result is read-only. Cached candidates resume per outer fold.
    Expanding OUTER_LIMIT/repeats reuses matching completed candidate folds.
    """
    if not isinstance(char_features, int) or char_features < 1:
        raise ValueError('char_features must be a positive integer.')
    all_folds = reference.split_plan(frame, config) if folds is None else validate_folds(
        list(folds), frame[TARGET], frame[GROUP])
    requested = all_folds if config.outer_limit is None else all_folds[:config.outer_limit]
    _validate_reference(frame, config, requested, reference_result)
    settings = {'reference_key': reference._cache_key(frame, config), 'char_features': char_features}
    key = hashlib.sha256(Path(__file__).read_bytes())
    key.update(json.dumps(settings, sort_keys=True).encode())
    directory = Path(cache_dir) / key.hexdigest()[:20]
    y, groups = frame[TARGET].to_numpy(), frame[GROUP].to_numpy()
    X = build_inputs(frame.copy())
    oof = np.full((config.repeats, len(frame)), np.nan)
    decisions = np.full_like(oof, np.nan)
    rows, diagnostics = [], []

    def factory():
        return A2CharacterMLP(max_epochs=config.max_epochs, patience=config.patience,
            batch_size=config.batch_size, checkpoint_splits=config.checkpoint_splits,
            tfidf_max_features=config.tfidf_max_features, dense_weight=config.dense_weight,
            seed=config.seed, char_features=char_features)

    for repeat, fold, fitting, scoring in requested:
        print(f'Repeat {repeat + 1}, fold {fold + 1}: {CHAR_NAME}', flush=True)
        signature = fold_signature(y, groups, fitting, scoring)
        path = directory / f'r{repeat}_f{fold}_{signature[:16]}.npz'
        if path.is_file():
            with np.load(path, allow_pickle=False) as saved:
                if not np.array_equal(saved['job_id'], frame.iloc[scoring].job_id.to_numpy()):
                    raise ValueError('Character cache scoring IDs differ.')
                p = saved['p'].copy()
                detail = json.loads(str(saved['detail'].item()))
            print('  Reusing completed character-trial predictions.', flush=True)
        else:
            started = time.monotonic()
            with threadpool_limits(limits=2):
                selected = select_settings(factory, X.iloc[fitting], y[fitting], groups[fitting],
                    {'learning_rate': list(config.mlp_rates)}, inner_splits=config.inner_splits,
                    seed=config.seed + repeat * 1000 + fold)
                model = clone(factory()).set_params(**selected.params).fit(X.iloc[fitting], y[fitting])
                p = model.predict_proba(X.iloc[scoring])[:, 1]
            detail = {'params': selected.params, 'threshold': selected.threshold,
                'selected_epoch': model.selected_epoch_, 'feature_width': model.a2_width_,
                'selection_pr_auc': float(selected.candidates.mean_inner_pr_auc.max()),
                'inner_candidates': selected.candidates.to_dict(orient='records'),
                'checkpoint_history': model.checkpoint_history_, 'minutes': (time.monotonic() - started) / 60}
            directory.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as output:
                np.savez_compressed(output, p=p, job_id=frame.iloc[scoring].job_id.to_numpy(),
                                    detail=json.dumps(detail))
            temporary.replace(path)
            del model
            gc.collect()
        metrics = score_summary(y[scoring], p, threshold=detail['threshold'])
        rows.append({'model': CHAR_NAME, 'repeat': repeat, 'fold': fold,
            'fold_signature': signature, 'train_rows': len(fitting), 'val_rows': len(scoring),
            'val_fraud': int(y[scoring].sum()), 'threshold': detail['threshold'],
            'selected_params': detail['params'], 'selected_epoch': detail['selected_epoch'], **metrics})
        diagnostics.append({'repeat': repeat, 'fold': fold, **detail})
        oof[repeat, scoring] = p
        decisions[repeat, scoring] = p >= detail['threshold']
        print(f"  AP {metrics['pr_auc']:.4f}; F1 {metrics['f1']:.4f}; "
              f"{detail['minutes']:.1f} fit minutes.", flush=True)

    candidate_scores = pd.DataFrame(rows)
    scores = pd.concat([reference_result['scores'].copy(), candidate_scores], ignore_index=True)
    names = [reference.LR_NAME, reference.MLP_NAME, CHAR_NAME]
    means = pd.DataFrame([summarise(scores.loc[scores.model == name], name) for name in names])
    combined = []
    for repeat in sorted({r for r, *_ in requested}):
        mask = np.isfinite(oof[repeat])
        truth, flagged = y[mask], decisions[repeat, mask].astype(bool)
        tp = int((flagged & (truth == 1)).sum())
        fp = int((flagged & (truth == 0)).sum())
        fn = int((~flagged & (truth == 1)).sum())
        combined.append({'model': CHAR_NAME, 'repeat': repeat + 1, 'evaluated_rows': int(mask.sum()),
            'fraud_total': int(truth.sum()), 'precision': tp / max(tp + fp, 1),
            'recall': tp / max(tp + fn, 1), 'f1': 2 * tp / max(2 * tp + fp + fn, 1),
            'fraud_caught': tp, 'fraud_missed': fn, 'genuine_flagged': fp})
    paired = []
    if len(requested) >= 2:
        original_scores = reference_result['scores'].loc[reference_result['scores'].model == reference.MLP_NAME]
        paired = [{'metric': metric, **paired_comparison(candidate_scores, original_scores, metric)}
                  for metric in ['pr_auc', 'precision', 'recall', 'f1']]
    return {'scores': scores, 'means': means,
        'combined': pd.concat([reference_result['combined'].copy(), pd.DataFrame(combined)], ignore_index=True),
        'paired': pd.DataFrame(paired), 'diagnostics': pd.DataFrame(diagnostics),
        'oof': oof, 'decisions': decisions, 'complete': len(requested) == len(all_folds),
        'completed_outer_folds': len(requested), 'total_outer_folds': len(all_folds),
        'cache_dir': str(directory)}
