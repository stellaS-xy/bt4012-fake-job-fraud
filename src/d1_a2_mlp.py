"""A2 logistic regression versus the 07 MLP under notebook 02's grouped protocol.

Reads development data only. No reserved-test reader or report export is provided.
The neural estimator makes its own grouped checkpoint split, then refits A2 and
the network on all rows passed to fit() for the chosen number of epochs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from src import d1_row_neural as neural
from src.evaluation import (fold_signature, make_folds, paired_comparison, score_summary,
                            select_settings, summarise, validate_folds)
from src.features import GROUP, TARGET, build_inputs, logistic_pipeline, make_features

LR_NAME = 'A2 logistic regression'
MLP_NAME = 'A2 deep MLP'


@dataclass(frozen=True)
class GroupedConfig:
    seed: int = 42
    outer_splits: int = 5
    repeats: int = 1
    inner_splits: int = 3
    checkpoint_splits: int = 5
    outer_limit: int | None = 1
    tfidf_max_features: int = 30000
    dense_weight: float = 1.
    lr_Cs: tuple = (.1, .3, 1.)
    mlp_rates: tuple = (.001,)
    max_epochs: int = 50
    patience: int = 5
    batch_size: int = 64

    def __post_init__(self):
        if min(self.outer_splits, self.inner_splits, self.checkpoint_splits) < 2 or self.repeats < 1:
            raise ValueError('Use at least two folds at each level and at least one repeat.')
        if self.outer_limit is not None and self.outer_limit < 1:
            raise ValueError('outer_limit must be positive or None.')
        if min(self.tfidf_max_features, self.max_epochs, self.patience, self.batch_size) < 1:
            raise ValueError('Feature and training sizes must be positive.')
        if not np.isfinite(self.dense_weight) or self.dense_weight <= 0:
            raise ValueError('dense_weight must be positive and finite.')
        if not self.lr_Cs or not self.mlp_rates or any(
                not np.isfinite(x) or x <= 0 for x in (*self.lr_Cs, *self.mlp_rates)):
            raise ValueError('Use nonempty positive finite C and learning-rate candidates.')


def _csr(matrix):
    return sparse.csr_matrix(matrix, dtype=np.float32)


class A2MLP(ClassifierMixin, BaseEstimator):
    """Cloneable estimator: A2 preprocessing plus internally selected-duration MLP.

    The table still carries group_id for splitting, but make_features('A2')
    explicitly drops it from the neural inputs. All learned preprocessing is
    fitted again for checkpoint selection and for the final refit.
    """
    def __init__(self, learning_rate=.001, max_epochs=50, patience=5, batch_size=64,
                 checkpoint_splits=5, tfidf_max_features=30000, dense_weight=1., seed=42):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.checkpoint_splits = checkpoint_splits
        self.tfidf_max_features = tfidf_max_features
        self.dense_weight = dense_weight
        self.seed = seed

    def _features(self):
        return make_features('A2', tfidf_max_features=self.tfidf_max_features,
                             dense_weight=self.dense_weight)

    def fit(self, X, y):
        y = np.asarray(y, dtype=int)
        if len(X) != len(y) or set(np.unique(y)) != {0, 1} or GROUP not in X:
            raise ValueError('Need aligned binary labels and a group_id column.')
        groups = X[GROUP].to_numpy()
        _, _, fitting, stopping = make_folds(groups, y, n_splits=self.checkpoint_splits,
                                             n_repeats=1, seed=self.seed)[0]
        self.checkpoint_indices_ = (fitting.copy(), stopping.copy())
        settings = neural.RowConfig(seed=self.seed, word_features=self.tfidf_max_features,
                                    max_epochs=self.max_epochs, patience=self.patience,
                                    batch_size=self.batch_size, learning_rate=self.learning_rate)
        print(f'  MLP checkpoint: {len(fitting):,} fitting / {len(stopping):,} stopping rows; '
              'groups disjoint.', flush=True)
        features = self._features()
        fit_matrix = _csr(features.fit_transform(X.iloc[fitting], y[fitting]))
        stop_matrix = _csr(features.transform(X.iloc[stopping]))
        # Both returned predictions are on stopping rows; no assessment rows
        # enter the checkpoint routine. The checkpoint weights are discarded.
        _, _, details = neural.fit_network((fit_matrix, stop_matrix, stop_matrix),
                                           y[fitting], y[stopping], settings)
        self.selected_epoch_ = int(details['selected_epoch'])
        self.checkpoint_history_ = details['history']
        del features, fit_matrix, stop_matrix
        gc.collect()

        self.features_ = self._features()
        matrix = _csr(self.features_.fit_transform(X, y))
        self.adapter_ = neural.KerasAdapter(matrix.shape[1], settings)
        weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y)[y]
        rng = np.random.default_rng(self.seed)
        print(f'  Refit MLP on all {len(y):,} rows for {self.selected_epoch_} fixed epochs '
              f'({matrix.shape[1]:,} A2 columns).', flush=True)
        for _ in range(self.selected_epoch_):
            self.adapter_.reset_metrics()
            order = rng.permutation(len(y))
            for start in range(0, len(y), self.batch_size):
                idx = order[start:start + self.batch_size]
                self.adapter_.train_batch(matrix[idx], y[idx], weights[idx])
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        self.a2_width_ = matrix.shape[1]
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ['features_', 'adapter_', 'classes_'])
        matrix = _csr(self.features_.transform(X))
        p = neural.predict_batches(self.adapter_, matrix, self.batch_size)
        return np.column_stack([1 - p, p])


def load_development(path):
    """Load only the supplied development file; never call features.load_d1()."""
    path = Path(path)
    if path.name != 'd1_train.csv':
        raise ValueError('This experiment accepts the frozen d1_train.csv only.')
    frame = pd.read_csv(path)
    if not frame.job_id.is_unique or frame[GROUP].isna().any() or set(frame[TARGET]) != {0, 1}:
        raise ValueError('Need unique job IDs, nonmissing groups and both labels.')
    return frame


def split_plan(frame, config, team_assignments=None):
    """Restore saved outer folds by job ID; never regenerate the agreed split."""
    if team_assignments is None or not Path(team_assignments).is_file():
        raise FileNotFoundError('Saved model_cv_assignments.csv is required. Upload the agreed file from 02.')
    saved = pd.read_csv(team_assignments)
    required = {'repeat', 'fold', 'job_id', GROUP}
    if not required.issubset(saved):
        raise ValueError('Saved assignments need repeat, fold, job_id and group_id.')
    for column in ['repeat', 'fold']:
        values = pd.to_numeric(saved[column], errors='raise')
        if values.isna().any() or not np.isfinite(values).all() or (values < 0).any() or (values % 1 != 0).any():
            raise ValueError('Assignment repeat/fold identifiers must be nonnegative integers.')
        saved[column] = values.astype(int)
    if saved[['job_id', GROUP]].isna().any().any():
        raise ValueError('Saved assignments contain missing job/group IDs.')
    folds = []
    positions = np.arange(len(frame))
    indexed = frame.set_index('job_id')
    for repeat in range(config.repeats):
        rows = saved.loc[saved.repeat == repeat]
        if (rows.job_id.duplicated().any() or set(rows.job_id) != set(frame.job_id)
                or set(rows.fold) != set(range(config.outer_splits))):
            raise ValueError(f'Saved repeat {repeat} must assign every development job once across all folds.')
        expected = indexed.loc[rows.job_id, GROUP].to_numpy()
        if not np.array_equal(expected, rows[GROUP].to_numpy()):
            raise ValueError('Saved job/group identities differ from development data.')
        for fold in range(config.outer_splits):
            scoring = frame.job_id.isin(rows.loc[rows.fold == fold, 'job_id']).to_numpy()
            folds.append((repeat, fold, positions[~scoring], positions[scoring]))
    return validate_folds(folds, frame[TARGET], frame[GROUP])


def _versions():
    return {name: importlib.metadata.version(name) for name in
            ['numpy', 'pandas', 'scipy', 'scikit-learn', 'tensorflow', 'keras']}


def _cache_key(frame, config):
    # Changing only the number of requested folds/repeats can reuse identical
    # completed fold signatures; changes to any fitted setting invalidate them.
    settings = asdict(config)
    for key in ['outer_limit', 'repeats']:
        settings.pop(key)
    key = hashlib.sha256(frame.to_csv(index=False, lineterminator='\n').encode())
    key.update(json.dumps({'config': settings, 'versions': _versions()}, sort_keys=True).encode())
    for name in ['d1_a2_mlp.py', 'd1_row_neural.py', 'features.py', 'evaluation.py']:
        key.update((Path(__file__).parent / name).read_bytes())
    return key.hexdigest()[:20]


def run_comparison(frame, config, cache_dir, folds=None):
    """Nested grouped selection, resumable by model and completed outer fold.

    A partial screen intentionally leaves unevaluated OOF rows as NaN. It is
    never labelled a full CV result and no missing predictions are filled in.
    """
    if folds is None:
        raise ValueError('Pass the restored saved outer folds explicitly.')
    folds = validate_folds(list(folds), frame[TARGET], frame[GROUP])
    requested = folds if config.outer_limit is None else folds[:config.outer_limit]
    y, groups = frame[TARGET].to_numpy(), frame[GROUP].to_numpy()
    X = build_inputs(frame.copy())
    directory = Path(cache_dir) / _cache_key(frame, config)
    makers = {
        LR_NAME: (lambda: logistic_pipeline(['raw', 'engineered', 'text_style', 'tfidf'],
                    dense_weight=config.dense_weight, tfidf_max_features=config.tfidf_max_features),
                  {'model__C': list(config.lr_Cs)}),
        MLP_NAME: (lambda: A2MLP(max_epochs=config.max_epochs, patience=config.patience,
                    batch_size=config.batch_size, checkpoint_splits=config.checkpoint_splits,
                    tfidf_max_features=config.tfidf_max_features, dense_weight=config.dense_weight, seed=config.seed),
                  {'learning_rate': list(config.mlp_rates)}),
    }
    rows, candidates, checkpoint_rows = [], [], []
    oof = {name: np.full((config.repeats, len(frame)), np.nan) for name in makers}
    decisions = {name: np.full((config.repeats, len(frame)), np.nan) for name in makers}
    for repeat, fold, fitting, scoring in requested:
        if repeat >= config.repeats:
            raise ValueError('Supplied folds exceed configured repeats.')
        signature = fold_signature(y, groups, fitting, scoring)
        for number, (name, (factory, grid)) in enumerate(makers.items()):
            print(f'Repeat {repeat + 1}, fold {fold + 1}: {name}', flush=True)
            path = directory / f'r{repeat}_f{fold}_m{number}_{signature[:16]}.npz'
            if path.is_file():
                with np.load(path, allow_pickle=False) as saved:
                    if not np.array_equal(saved['job_id'], frame.iloc[scoring].job_id.to_numpy()):
                        raise ValueError('Cached scoring job IDs differ.')
                    p = saved['p'].copy()
                    detail = json.loads(str(saved['detail'].item()))
                print('  Reusing completed fold predictions.', flush=True)
            else:
                started = time.monotonic()
                with threadpool_limits(limits=2):
                    selection = select_settings(factory, X.iloc[fitting], y[fitting], groups[fitting], grid,
                        inner_splits=config.inner_splits, seed=config.seed + repeat * 1000 + fold)
                    model = clone(factory()).set_params(**selection.params).fit(X.iloc[fitting], y[fitting])
                    p = model.predict_proba(X.iloc[scoring])[:, 1]
                detail = {'params': selection.params, 'threshold': selection.threshold,
                          'selection_pr_auc': float(selection.candidates.mean_inner_pr_auc.max()),
                          'candidates': selection.candidates.to_dict(orient='records'),
                          'selected_epoch': getattr(model, 'selected_epoch_', None),
                          'checkpoint_history': getattr(model, 'checkpoint_history_', []),
                          'a2_width': getattr(model, 'a2_width_', None),
                          'minutes': (time.monotonic() - started) / 60}
                directory.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix('.tmp')
                with temporary.open('wb') as output:
                    np.savez_compressed(output, job_id=frame.iloc[scoring].job_id.to_numpy(),
                                        p=p, detail=json.dumps(detail))
                temporary.replace(path)
                del model
                gc.collect()
            metric = score_summary(y[scoring], p, threshold=detail['threshold'])
            rows.append({'model': name, 'repeat': repeat, 'fold': fold, 'fold_signature': signature,
                         'train_rows': len(fitting), 'val_rows': len(scoring), 'val_fraud': int(y[scoring].sum()),
                         'selected_params': detail['params'], 'selected_epoch': detail['selected_epoch'],
                         'threshold': detail['threshold'], 'selection_pr_auc': detail['selection_pr_auc'], **metric})
            candidates.extend({'model': name, 'repeat': repeat, 'fold': fold, **row} for row in detail['candidates'])
            checkpoint_rows.extend({'model': name, 'repeat': repeat, 'fold': fold, **row}
                                   for row in detail['checkpoint_history'])
            oof[name][repeat, scoring] = p
            decisions[name][repeat, scoring] = p >= detail['threshold']
            print(f"  AP {metric['pr_auc']:.4f}; F1 {metric['f1']:.4f}; "
                  f"cutoff {detail['threshold']:.4f}; {detail['minutes']:.1f} fit minutes.", flush=True)
    scores = pd.DataFrame(rows)
    means = pd.DataFrame([summarise(scores.loc[scores.model == name], name) for name in makers])
    combined = []
    for name in makers:
        for repeat in sorted({r for r, *_ in requested}):
            mask = np.isfinite(oof[name][repeat])
            truth, flagged = y[mask], decisions[name][repeat, mask].astype(bool)
            tp, fp = int((flagged & (truth == 1)).sum()), int((flagged & (truth == 0)).sum())
            fn = int((~flagged & (truth == 1)).sum())
            combined.append({'model': name, 'repeat': repeat + 1, 'evaluated_rows': int(mask.sum()),
                'fraud_total': int(truth.sum()), 'precision': tp / max(tp + fp, 1),
                'recall': tp / max(tp + fn, 1), 'f1': 2 * tp / max(2 * tp + fp + fn, 1),
                'fraud_caught': tp, 'fraud_missed': fn, 'genuine_flagged': fp})
    paired = []
    if len(requested) >= 2:
        for metric in ['pr_auc', 'precision', 'recall', 'f1']:
            paired.append({'metric': metric, **paired_comparison(scores.loc[scores.model == MLP_NAME],
                                                                 scores.loc[scores.model == LR_NAME], metric)})
    return {'scores': scores, 'means': means, 'combined': pd.DataFrame(combined),
            'paired': pd.DataFrame(paired), 'inner_candidates': pd.DataFrame(candidates),
            'checkpoint_history': pd.DataFrame(checkpoint_rows), 'oof': oof, 'decisions': decisions,
            'complete': len(requested) == len(folds), 'completed_outer_folds': len(requested),
            'total_outer_folds': len(folds), 'cache_dir': str(directory)}
