"""Native CatBoost, frozen semantic classifiers and a calibrated score ensemble."""
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import logit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from src import d1_candidate_comparison as baseline
from src.d1_semantic_features import clean_text
from src.evaluation import make_folds
from src.features import GROUP, TARGET, RAW_CAT, build_inputs, make_features


class NativeCatBoost(ClassifierMixin, BaseEstimator):
    def __init__(self, depth=4, iterations=200, seed=42, threads=2):
        self.depth, self.iterations, self.seed, self.threads = depth, iterations, seed, threads

    def _table(self, X):
        table = X[self.columns_].copy()
        for name in self.categories_:
            table[name] = table[name].fillna('MISSING').astype(str)
        table['text_clean'] = table.text_clean.fillna('').map(clean_text).str.lower().replace('', 'emptytext')
        for name in set(self.columns_) - set(self.categories_) - {'text_clean'}:
            table[name] = pd.to_numeric(table[name], errors='coerce').astype(float).replace([np.inf, -np.inf], np.nan)
        return table

    def fit(self, X, y):
        from catboost import CatBoostClassifier
        self.columns_ = [c for c in X if c not in {GROUP, 'job_id', TARGET} and not c.startswith('sem_')]
        self.categories_ = [c for c in (*RAW_CAT, 'country') if c in self.columns_]
        self.model_ = CatBoostClassifier(iterations=self.iterations, depth=self.depth,
            learning_rate=.05, l2_leaf_reg=5., loss_function='Logloss', auto_class_weights='Balanced',
            random_seed=self.seed, thread_count=self.threads, verbose=False, allow_writing_files=False,
            cat_features=self.categories_, text_features=['text_clean'],
            text_processing={
                'tokenizers': [{'tokenizer_id': 'Space', 'separator_type': 'ByDelimiter', 'delimiter': ' '}],
                'dictionaries': [{'dictionary_id': 'Word', 'max_dictionary_size': '5000',
                                  'occurrence_lower_bound': '3', 'gram_order': '1'}],
                'feature_processing': {'default': [{'tokenizers_names': ['Space'],
                    'dictionaries_names': ['Word'], 'feature_calcers': ['BoW', 'NaiveBayes']}]}})
        self.model_.fit(self._table(X), y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        check_is_fitted(self, 'model_')
        return self.model_.predict_proba(self._table(X))


class SemanticLogistic(ClassifierMixin, BaseEstimator):
    def __init__(self, C=.3, include_structured=False, seed=42):
        self.C, self.include_structured, self.seed = C, include_structured, seed

    def _matrix(self, X, fitting=False, y=None):
        vectors = X[self.embedding_columns_].to_numpy(dtype=np.float32)
        if fitting:
            self.scaler_ = StandardScaler()
            vectors = self.scaler_.fit_transform(vectors)
        else:
            vectors = self.scaler_.transform(vectors)
        matrix = sparse.csr_matrix(vectors, dtype=np.float32)
        if self.include_structured:
            if fitting:
                self.structured_ = make_features(blocks=['raw', 'engineered', 'text_style'])
                metadata = self.structured_.fit_transform(X, y)
            else:
                metadata = self.structured_.transform(X)
            matrix = sparse.hstack([matrix, metadata], format='csr', dtype=np.float32)
        return matrix

    def fit(self, X, y):
        self.embedding_columns_ = sorted(c for c in X if c.startswith('sem_'))
        if not self.embedding_columns_:
            raise ValueError('Missing frozen semantic vectors.')
        matrix = self._matrix(X, fitting=True, y=y)
        self.model_ = LogisticRegression(C=self.C, solver='liblinear', class_weight='balanced',
                                         max_iter=3000, random_state=self.seed).fit(matrix, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        check_is_fitted(self, 'model_')
        return self.model_.predict_proba(self._matrix(X))


def _configure_tf(threads):
    import tensorflow as tf
    try:
        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        pass  # A notebook may already have initialised its TensorFlow runtime.


def _components(config):
    return (baseline.reference.A2MLP(tfidf_max_features=config.tfidf_max_features, seed=config.seed),
            baseline.A2Candidate(kind='extra_trees', min_samples_leaf=1,
                tfidf_max_features=config.tfidf_max_features, tree_count=config.tree_count,
                seed=config.seed, threads=config.threads))


def _frame_hash(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()).hexdigest()


def calibrated_stage(training, evaluation, config, cache_path):
    """Fit on training only; calibrate on its own held-out groups; predict evaluation.

    This whole procedure is rerun inside each outer training's three inner
    folds. Thus the eventual F1 threshold uses genuinely out-of-fold ensemble
    predictions, including the learned calibrators. Components are not refitted
    after calibration. Completed prediction stages resume without neural fits.
    """
    evaluation = evaluation.drop(columns=[TARGET], errors='ignore')
    if set(training.job_id) & set(evaluation.job_id) or set(training[GROUP]) & set(evaluation[GROUP]):
        raise ValueError('Ensemble fitting and evaluation rows/groups must be disjoint.')
    context = {'training_hash': _frame_hash(training), 'evaluation_hash': _frame_hash(evaluation)}
    cache_path = Path(cache_path)
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as saved:
            detail = json.loads(str(saved['detail'].item()))
            if detail['context'] != context or not np.array_equal(saved['job_id'], evaluation.job_id):
                raise ValueError('Calibrated ensemble prediction cache mismatch.')
            return saved['p'].copy(), detail
    y, groups = training[TARGET].to_numpy(), training[GROUP].to_numpy()
    _, _, fitting, calibration = make_folds(groups, y, n_splits=5, n_repeats=1,
                                           seed=config.seed + 300)[0]
    X = build_inputs(training.copy())
    X_eval = build_inputs(evaluation.copy())
    _configure_tf(config.threads)
    mlp, trees = _components(config)
    print(f'    Ensemble core: {len(fitting):,} fitting / {len(calibration):,} calibration groups held out', flush=True)
    predictions, calibration_details = [], []
    for name, model in [('MLP', mlp), ('Extra Trees', trees)]:
        print(f'    Fitting ensemble component: {name}', flush=True)
        model.fit(X.iloc[fitting], y[fitting])
        p_cal = model.predict_proba(X.iloc[calibration])[:, 1]
        p_eval = model.predict_proba(X_eval)[:, 1]
        # Natural-prevalence sigmoid calibration; no balancing of calibration labels.
        calibrator = LogisticRegression(C=1000., solver='lbfgs', max_iter=1000)
        calibrator.fit(logit(np.clip(p_cal.astype(float), 1e-6, 1 - 1e-6))[:, None], y[calibration])
        predictions.append(calibrator.predict_proba(
            logit(np.clip(p_eval.astype(float), 1e-6, 1 - 1e-6))[:, None])[:, 1])
        calibration_details.append({'model': name, 'selected_epoch': getattr(model, 'selected_epoch_', None),
            'sigmoid_slope': float(calibrator.coef_[0, 0]), 'sigmoid_intercept': float(calibrator.intercept_[0])})
    p = .5 * predictions[0] + .5 * predictions[1]
    detail = {'context': context, 'component_details': calibration_details,
        'core_rows': len(fitting), 'calibration_rows': len(calibration),
        'weights': [.5, .5], 'refit_after_calibration': False}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix('.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, p=p, job_id=evaluation.job_id.to_numpy(),
            fitting_job_id=training.iloc[fitting].job_id.to_numpy(),
            calibration_job_id=training.iloc[calibration].job_id.to_numpy(), detail=json.dumps(detail))
    tmp.replace(cache_path)
    framework = getattr(getattr(mlp, 'adapter_', None), 'tf', None)
    del model, mlp, trees
    if framework is not None:
        framework.keras.backend.clear_session(free_memory=True)
    gc.collect()
    return p, detail
