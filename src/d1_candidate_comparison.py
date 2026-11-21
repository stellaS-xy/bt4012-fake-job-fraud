"""New classical candidates under 02/09's nested grouped evaluation.

Only development data is read. Original MLP/LR predictions are reused only
when the data, source, packages, settings, scoring IDs and folds match exactly.
No neural training is started here. Partial runs are explicitly incomplete.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
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
from scipy import sparse
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from src import d1_a2_mlp as reference
from src.evaluation import (fold_signature, paired_comparison, score_summary,
                            select_settings, validate_folds)
from src.features import GROUP, TARGET, build_inputs, make_features

TRAINING_SHA = 'bb132ac17cadb39be336612523330fc85c1ccb6512ed44b6ac3ca00eb572f5e8'
CANDIDATES = ('linear_svm', 'svd_rbf_svm', 'extra_trees', 'group_balanced_lr', 'lightgbm')
NAMES = {'linear_svm': 'Linear SVM', 'svd_rbf_svm': 'SVD + RBF SVM',
         'extra_trees': 'Extra Trees', 'group_balanced_lr': 'Group-balanced logistic regression',
         'lightgbm': 'LightGBM (current split)',
         'mlp_reference': 'MLP reference (09)', 'lr_reference': 'Logistic regression reference (09)'}
METRICS = ('pr_auc', 'precision', 'recall', 'f1', 'precision_top5pct', 'recall_top5pct')


@dataclass(frozen=True)
class ComparisonConfig:
    repeats: int = 5
    outer_limit: int | None = 3
    seed: int = 42
    outer_splits: int = 5
    inner_splits: int = 3
    tfidf_max_features: int = 30000
    svd_components: int = 128
    tree_count: int = 128
    lightgbm_max_features: int = 2000
    threads: int = 2

    def __post_init__(self):
        if self.repeats < 1 or min(self.outer_splits, self.inner_splits) < 2:
            raise ValueError('Invalid repeat or split count.')
        if self.outer_limit is not None and self.outer_limit < 1:
            raise ValueError('outer_limit must be positive or None.')
        if min(self.tfidf_max_features, self.svd_components, self.tree_count,
               self.lightgbm_max_features, self.threads) < 1:
            raise ValueError('Feature, tree and thread counts must be positive.')


def load_training(root):
    frame = reference.load_development(Path(root) / 'data/processed/d1_train.csv')
    digest = hashlib.sha256(frame.to_csv(index=False, lineterminator='\n').encode()).hexdigest()
    if digest != TRAINING_SHA:
        raise ValueError('The agreed current frozen training data changed. Resolve the data version before comparing.')
    return frame


def restore_folds(frame, config, assignment_path):
    """Restore exact 02 assignments by job ID, never silently replace them."""
    saved = pd.read_csv(assignment_path)
    if not {'repeat', 'fold', 'job_id', GROUP}.issubset(saved):
        raise ValueError('Missing saved assignment columns.')
    for column in ('repeat', 'fold'):
        values = pd.to_numeric(saved[column], errors='raise')
        if values.isna().any() or (values < 0).any() or (values % 1 != 0).any():
            raise ValueError('Fold/repeat IDs must be nonnegative integers.')
        saved[column] = values.astype(int)
    positions = np.arange(len(frame))
    indexed = frame.set_index('job_id')
    if not indexed.index.is_unique or frame[GROUP].isna().any():
        raise ValueError('Need unique posting IDs and nonmissing groups.')
    folds = []
    for repeat in range(config.repeats):
        rows = saved.loc[saved.repeat == repeat]
        if (rows.job_id.duplicated().any() or set(rows.job_id) != set(frame.job_id)
                or set(rows.fold) != set(range(config.outer_splits))):
            raise ValueError('Each requested repeat must assign every training posting exactly once.')
        if not np.array_equal(indexed.loc[rows.job_id, GROUP].to_numpy(), rows[GROUP].to_numpy()):
            raise ValueError('Saved group identities differ from the training file.')
        for fold in range(config.outer_splits):
            scoring = frame.job_id.isin(rows.loc[rows.fold == fold, 'job_id']).to_numpy()
            folds.append((repeat, fold, positions[~scoring], positions[scoring]))
    return validate_folds(folds, frame[TARGET], frame[GROUP])


def group_balanced_weights(y, groups):
    """Inverse fitting-group size, then equal total weight for each class."""
    y, groups = np.asarray(y), np.asarray(groups)
    sizes = pd.Series(groups).value_counts()
    weights = 1. / pd.Series(groups).map(sizes).to_numpy(dtype=float)
    for label in (0, 1):
        mask = y == label
        if not mask.any():
            raise ValueError('Both classes are needed.')
        weights[mask] *= len(y) / (2 * weights[mask].sum())
    return weights


def _fit_features(X, y, limit):
    features = make_features('A2', tfidf_max_features=limit)
    matrix = sparse.csr_matrix(features.fit_transform(X, y), dtype=np.float32)
    return features, matrix


class A2Candidate(ClassifierMixin, BaseEstimator):
    """Fitting-only A2, optional projection, class weighting and a classifier.

    group_id only controls weights for the group-balanced candidate. It is
    never a predictor. SVM margins are monotonically mapped into [0,1] for the
    shared scoring interface; these values are NOT calibrated probabilities.
    """
    def __init__(self, kind='linear_svm', strength=1., min_samples_leaf=2,
                 tfidf_max_features=30000, svd_components=128, tree_count=128,
                 seed=42, threads=2, memory=None, lightgbm_max_features=2000, tree_leaves=15):
        self.kind = kind
        self.strength = strength
        self.min_samples_leaf = min_samples_leaf
        self.tfidf_max_features = tfidf_max_features
        self.svd_components = svd_components
        self.tree_count = tree_count
        self.seed = seed
        self.threads = threads
        self.memory = memory
        self.lightgbm_max_features = lightgbm_max_features
        self.tree_leaves = tree_leaves

    def fit(self, X, y):
        if self.kind not in CANDIDATES:
            raise ValueError('Unknown candidate.')
        y = np.asarray(y, dtype=int)
        if set(np.unique(y)) != {0, 1}:
            raise ValueError('Both classes are required.')
        prepare = _fit_features if self.memory is None else self.memory.cache(_fit_features)
        limit = self.lightgbm_max_features if self.kind == 'lightgbm' else self.tfidf_max_features
        self.features_, matrix = prepare(X, y, limit)
        weights = (group_balanced_weights(y, X[GROUP]) if self.kind == 'group_balanced_lr'
                   else compute_sample_weight('balanced', y))
        if self.kind == 'linear_svm':
            self.model_ = LinearSVC(C=self.strength, dual=False, max_iter=10000,
                                    random_state=self.seed)
        elif self.kind == 'svd_rbf_svm':
            width = min(self.svd_components, matrix.shape[1] - 1, matrix.shape[0] - 1)
            if width < 1:
                raise ValueError('Too few rows/features for SVD.')
            self.model_ = Pipeline([
                ('svd', TruncatedSVD(n_components=width, n_iter=3, random_state=self.seed)),
                ('scale', StandardScaler()),
                ('model', SVC(C=self.strength, kernel='rbf', gamma='scale',
                              probability=False, cache_size=256))], memory=self.memory)
            self.model_.fit(matrix, y, model__sample_weight=weights)
        elif self.kind == 'extra_trees':
            self.model_ = ExtraTreesClassifier(n_estimators=self.tree_count,
                min_samples_leaf=self.min_samples_leaf, max_features='sqrt',
                n_jobs=self.threads, random_state=self.seed)
        elif self.kind == 'lightgbm':
            from lightgbm import LGBMClassifier
            self.model_ = LGBMClassifier(n_estimators=200, learning_rate=.05,
                num_leaves=self.tree_leaves, max_bin=63, n_jobs=self.threads,
                random_state=self.seed, verbosity=-1, deterministic=True, force_col_wise=True)
        else:
            self.model_ = LogisticRegression(C=self.strength, solver='liblinear',
                max_iter=2000, random_state=self.seed)
        if self.kind != 'svd_rbf_svm':
            self.model_.fit(matrix, y, sample_weight=weights)
        if self.kind == 'linear_svm' and np.max(self.model_.n_iter_) >= self.model_.max_iter:
            raise RuntimeError('Linear SVM did not converge; do not interpret or cache this fit.')
        self.classes_ = np.array([0, 1])
        self.fitting_rows_ = len(y)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ['features_', 'model_'])
        matrix = sparse.csr_matrix(self.features_.transform(X), dtype=np.float32)
        if hasattr(self.model_, 'predict_proba'):
            return self.model_.predict_proba(matrix)
        p = expit(self.model_.decision_function(matrix))
        return np.column_stack([1 - p, p])


def candidate_spec(kind, config, memory=None):
    model = A2Candidate(kind=kind, tfidf_max_features=config.tfidf_max_features,
        svd_components=config.svd_components, tree_count=config.tree_count,
        seed=config.seed, threads=config.threads, memory=memory,
        lightgbm_max_features=config.lightgbm_max_features)
    grids = {'linear_svm': {'strength': [.1, 1.]},
             'svd_rbf_svm': {'strength': [1., 4.]},
             'extra_trees': {'min_samples_leaf': [2, 5]},
             'group_balanced_lr': {'strength': [.1, .3, 1.]},
             'lightgbm': {'tree_leaves': [15, 31]}}
    return model, grids[kind]


def _reference_predictions(root, frame, config, fold):
    """Only reuse the original 09 nested procedure, never the fast screen."""
    ref_config = reference.GroupedConfig(seed=config.seed, outer_splits=config.outer_splits,
        repeats=config.repeats, inner_splits=config.inner_splits, outer_limit=None,
        tfidf_max_features=config.tfidf_max_features)
    key = reference._cache_key(frame, ref_config)
    repeat, number, fitting, scoring = fold
    signature = fold_signature(frame[TARGET], frame[GROUP], fitting, scoring)
    results = {}
    for name, model_number in (('lr_reference', 0), ('mlp_reference', 1)):
        filename = f'r{repeat}_f{number}_m{model_number}_{signature[:16]}.npz'
        paths = [Path(root) / 'data/processed' / directory / key / filename
                 for directory in ('09_a2_mlp', '08_a2_mlp')]
        path = next((p for p in paths if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f'No compatible original 09 cache for {name}, repeat {repeat + 1}, '
                f'fold {number + 1}. Complete that fold in 09 or request fewer folds. '
                'Neural training has not been started.')
        with np.load(path, allow_pickle=False) as saved:
            if not np.array_equal(saved['job_id'], frame.iloc[scoring].job_id.to_numpy()):
                raise ValueError('Original reference scoring IDs differ.')
            results[name] = (saved['p'].copy(), json.loads(str(saved['detail'].item())), str(path))
    return results


def _identity(frame, fold):
    repeat, number, fitting, scoring = fold
    return {'repeat': repeat, 'fold': number,
        'fold_signature': fold_signature(frame[TARGET], frame[GROUP], fitting, scoring),
        'train_rows': len(fitting), 'val_rows': len(scoring)}


def comparison_tables(scores, expected_folds):
    summaries, comparisons = [], []
    ref = scores.loc[scores.key == 'mlp_reference']
    for key, table in scores.groupby('key', sort=False):
        summaries.append({'key': key, 'model': NAMES[key], 'evaluated_folds': len(table),
            'expected_folds': expected_folds, 'complete': len(table) == expected_folds,
            'tfidf_max_features': int(table.tfidf_max_features.iloc[0]),
            **{metric: table[metric].mean() for metric in METRICS},
            'pr_auc_sd': table.pr_auc.std(ddof=1), 'minutes_per_fold': table.minutes.mean()})
        if key in CANDIDATES and len(table) >= 2:
            ap = paired_comparison(table, ref, 'pr_auc')
            f1_delta = table.f1.mean() - ref.f1.mean()
            comparisons.append({'key': key, 'model': NAMES[key], 'ap_difference': ap['mean_diff'],
                'ap_interval_low': ap['low'], 'ap_interval_high': ap['high'],
                'share_folds_ap_better': ap['share_folds_a_better'], 'f1_difference': f1_delta,
                'precision_difference': table.precision.mean() - ref.precision.mean(),
                'recall_difference': table.recall.mean() - ref.recall.mean(),
                'worth_followup': bool(len(table) >= 3 and ap['mean_diff'] >= .03
                    and f1_delta >= 0 and ap['share_folds_a_better'] >= .6)})
    return pd.DataFrame(summaries).sort_values('pr_auc', ascending=False), pd.DataFrame(comparisons)


def run_comparison(root, frame, config=ComparisonConfig(), candidates=CANDIDATES):
    root, candidates = Path(root), tuple(candidates)
    if not candidates or len(set(candidates)) != len(candidates) or set(candidates) - set(CANDIDATES):
        raise ValueError('Choose unique supported candidates.')
    all_folds = restore_folds(frame, config, root / 'data/processed/model_cv_assignments.csv')
    folds = all_folds if config.outer_limit is None else all_folds[:config.outer_limit]
    # Fail BEFORE expensive classical training if the requested reference is unavailable.
    references = [_reference_predictions(root, frame, config, fold) for fold in folds]
    versions = {name: importlib.metadata.version(name) for name in
        ('numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib', 'tensorflow', 'keras', 'lightgbm')}
    settings = asdict(config)
    for field in ('outer_limit', 'repeats'):
        settings.pop(field)
    digest = hashlib.sha256(frame.to_csv(index=False, lineterminator='\n').encode())
    digest.update(json.dumps({'config': settings, 'versions': versions}, sort_keys=True).encode())
    for name in ('d1_candidate_comparison.py', 'features.py', 'evaluation.py', 'd1_a2_mlp.py', 'd1_row_neural.py'):
        digest.update((Path(__file__).parent / name).read_bytes())
    directory = root / 'data/processed/10_candidate_comparison' / digest.hexdigest()[:20]
    directory.mkdir(parents=True, exist_ok=True)
    y, groups = frame[TARGET].to_numpy(), frame[GROUP].to_numpy()
    X = build_inputs(frame.copy())
    rows = []
    for fold, cached_refs in zip(folds, references):
        identity = _identity(frame, fold)
        repeat, number, fitting, scoring = fold
        for key, (p, details, path) in cached_refs.items():
            rows.append({'key': key, 'model': NAMES[key], **identity,
                'tfidf_max_features': config.tfidf_max_features,
                **score_summary(y[scoring], p, threshold=details['threshold']),
                'minutes': details['minutes'], 'selected_params': details['params'],
                'selection_pr_auc': details['selection_pr_auc'], 'source': path})
        with tempfile.TemporaryDirectory(prefix='bt4012_candidates_') as temporary:
            memory = Memory(temporary, verbose=0)
            for key in candidates:
                path = directory / f'r{repeat}_f{number}_{key}_{identity["fold_signature"][:16]}.npz'
                if path.is_file():
                    with np.load(path, allow_pickle=False) as saved:
                        if not np.array_equal(saved['job_id'], frame.iloc[scoring].job_id.to_numpy()):
                            raise ValueError('Cached candidate scoring IDs differ.')
                        p, details = saved['p'].copy(), json.loads(str(saved['detail'].item()))
                    print(f'Repeat {repeat + 1}, fold {number + 1}: {NAMES[key]} — cached', flush=True)
                else:
                    print(f'Repeat {repeat + 1}, fold {number + 1}: {NAMES[key]} — inner selection', flush=True)
                    started = time.monotonic()
                    pipeline, grid = candidate_spec(key, config, memory)
                    with threadpool_limits(limits=config.threads):
                        selection = select_settings(lambda: pipeline, X.iloc[fitting], y[fitting],
                            groups[fitting], grid, inner_splits=config.inner_splits,
                            seed=config.seed + repeat * 1000 + number)
                        fitted = clone(pipeline).set_params(**selection.params).fit(X.iloc[fitting], y[fitting])
                        p = fitted.predict_proba(X.iloc[scoring])[:, 1]
                    details = {'params': selection.params, 'threshold': selection.threshold,
                        'selection_pr_auc': float(selection.candidates.mean_inner_pr_auc.max()),
                        'candidates': selection.candidates.to_dict(orient='records'),
                        'minutes': (time.monotonic() - started) / 60}
                    score_summary(y[scoring], p, threshold=details['threshold'])
                    tmp = path.with_suffix('.tmp')
                    with tmp.open('wb') as output:
                        np.savez_compressed(output, job_id=frame.iloc[scoring].job_id.to_numpy(),
                                            p=p, detail=json.dumps(details))
                    tmp.replace(path)
                    del fitted
                    gc.collect()
                metrics = score_summary(y[scoring], p, threshold=details['threshold'])
                rows.append({'key': key, 'model': NAMES[key], **identity, **metrics,
                    'tfidf_max_features': config.lightgbm_max_features if key == 'lightgbm' else config.tfidf_max_features,
                    'minutes': details['minutes'], 'selected_params': details['params'],
                    'selection_pr_auc': details['selection_pr_auc'], 'source': str(path)})
                print(f'  AP {metrics["pr_auc"]:.4f}; precision {metrics["precision"]:.4f}; '
                      f'recall {metrics["recall"]:.4f}; F1 {metrics["f1"]:.4f}', flush=True)
        # Refresh reviewable progress after each completed outer fold.
        scores = pd.DataFrame(rows)
        summary, paired = comparison_tables(scores, len(all_folds))
        scores.to_csv(directory / 'fold_metrics.csv', index=False)
        summary.to_csv(directory / 'summary.csv', index=False)
        paired.to_csv(directory / 'paired_vs_mlp.csv', index=False)
    manifest = {'status': 'development comparison', 'config': asdict(config), 'versions': versions,
        'training_sha256': hashlib.sha256(frame.to_csv(index=False, lineterminator='\n').encode()).hexdigest(),
        'training_rows': len(frame), 'fraud_rows': int(y.sum()), 'candidates': candidates,
        'evaluated_outer_folds': len(folds), 'expected_outer_folds': len(all_folds),
        'assignment_file_sha256': hashlib.sha256((root / 'data/processed/model_cv_assignments.csv').read_bytes()).hexdigest(),
        'folds': [_identity(frame, fold) for fold in folds],
        'selection': f'mean AP on {config.inner_splits} inner grouped folds; threshold maximises inner OOF F1',
        'aggregation': 'arithmetic mean of matching outer-fold metrics; repeats are not independent',
        'features': f'A2 with {config.tfidf_max_features}-term ceiling except LightGBM '
                    f'({config.lightgbm_max_features}); SVD additionally projects within each fit',
        'neural_training_started': False, 'final_test_accessed': False,
        'followup_rule': '>=3 folds; AP gain >=0.03; F1 not lower; AP better on >=60% of folds; heuristic, not significance'}
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return {'scores': scores, 'summary': summary, 'paired': paired, 'directory': directory,
            'manifest': manifest}
