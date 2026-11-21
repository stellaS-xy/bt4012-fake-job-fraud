"""Notebook 06: sparse text + metadata classifiers, independently of 04/05.

TensorFlow is imported only when a neural network is actually constructed.
The CLI can prepare a Colab input ZIP without installing TensorFlow locally.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import zipfile

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from threadpoolctl import threadpool_limits

from src.features import GROUP, TARGET, SEED, build_inputs, make_features, prepare_text
from src.evaluation import best_f1_threshold, fold_signature, make_folds, score_summary

TEXT_LR = '1. Word TF-IDF LR'
A2_LR = '2. Text + metadata LR'
A2_NN = '3. Text + metadata MLP'
RICH_NN = '4. Word + character + metadata MLP'
BLEND = '5. Validation-selected LR + MLP blend'
MODELS = (TEXT_LR, A2_LR, A2_NN, RICH_NN, BLEND)
BUNDLE_FILES = ('src/d1_sparse_nn.py', 'src/features.py', 'src/evaluation.py',
                'data/processed/d1_train.csv')


@dataclass(frozen=True)
class NeuralConfig:
    outer_folds: tuple = (0,)
    seed: int = SEED
    word_features: int = 5000
    character_features: int = 15000
    max_epochs: int = 20
    patience: int = 3
    batch_size: int = 128
    dropout: float = .35
    l2: float = 1e-4
    learning_rate: float = 1e-3
    include_profile_text: bool = True
    blend_weights: tuple = (0., .25, .5, .75, 1.)

    def __post_init__(self):
        if self.seed != SEED:
            raise ValueError('Keep the shared seed at 42 for this comparison.')
        if not self.outer_folds or len(set(self.outer_folds)) != len(self.outer_folds):
            raise ValueError('Choose non-duplicated outer folds, e.g. (0,) or (0,1,2,3,4).')
        if any(f not in range(5) for f in self.outer_folds):
            raise ValueError('Outer fold indices must be from 0 to 4.')
        if min(self.word_features, self.character_features, self.max_epochs, self.patience, self.batch_size) < 1:
            raise ValueError('Feature counts, epochs, patience and batch size must be positive.')
        if not 0 <= self.dropout < 1 or self.l2 < 0 or self.learning_rate <= 0:
            raise ValueError('Invalid network regularization or learning rate.')
        if not self.blend_weights or any(not 0 <= w <= 1 for w in self.blend_weights):
            raise ValueError('Blend weights must be in [0, 1].')


def package_versions():
    return {name: importlib.metadata.version(name) for name in
            ['numpy', 'pandas', 'scipy', 'scikit-learn', 'tensorflow', 'keras', 'threadpoolctl']}


def load_training(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Missing frozen d1_train.csv. Run notebook 00 locally and prepare the Colab ZIP.')
    frame = pd.read_csv(path)
    if not frame.job_id.is_unique or frame[GROUP].isna().any() or set(frame[TARGET].unique()) != {0, 1}:
        raise ValueError('Need unique job IDs, frozen non-missing groups, and both fraud labels.')
    digest = hashlib.sha256(frame.to_csv(index=False, lineterminator='\n').encode()).hexdigest()
    return frame, digest


def split_plan(frame, config):
    y, groups = frame[TARGET].to_numpy(), frame[GROUP].to_numpy()
    outer_folds = make_folds(groups, y, 5, 1, config.seed)
    result = []
    for fold in config.outer_folds:
        _, _, outer, scoring = outer_folds[fold]
        _, _, local_fit, local_validation = make_folds(
            groups[outer], y[outer], 5, 1, config.seed + 100 + fold)[0]
        fitting, validation = outer[local_fit], outer[local_validation]
        for a, b in [(fitting, validation), (fitting, scoring), (validation, scoring)]:
            if not set(groups[a]).isdisjoint(groups[b]):
                raise ValueError('Related groups cross a split.')
        result.append((fold, fitting, validation, scoring, outer))
    return result


def feature_inputs(frame, config):
    inputs = build_inputs(frame.copy())
    inputs['department'] = frame['department']
    if not config.include_profile_text:
        without = frame.copy()
        without['company_profile'] = ''
        inputs['text_clean'] = prepare_text(without)[1]
    return inputs


def feature_builders(config):
    characters = ColumnTransformer([('characters', TfidfVectorizer(
        analyzer='char_wb', ngram_range=(3, 5), min_df=3, max_df=.95,
        max_features=config.character_features, sublinear_tf=True, dtype=np.float32), 'text_clean')],
        sparse_threshold=1.)
    department = ColumnTransformer([('department', Pipeline([
        ('fill', SimpleImputer(strategy='constant', fill_value='MISSING')),
        ('encode', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=20))]), ['department'])],
        sparse_threshold=1.)
    return {
        'text': make_features(blocks=['tfidf'], tfidf_max_features=config.word_features),
        'a2': make_features('A2', tfidf_max_features=config.word_features),
        'rich': FeatureUnion([
            ('a2', make_features('A2', tfidf_max_features=config.word_features)),
            ('character', characters), ('department', department)], n_jobs=1),
    }


def _probabilities(values, length):
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.shape != (length,) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Invalid/misaligned classifier probabilities.')
    return values


class KerasSparseAdapter:
    """Sparse first layer; only small hidden activations become dense."""
    def __init__(self, input_width, config, seed):
        import tensorflow as tf
        self.tf = tf
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        regularizer = tf.keras.regularizers.l2(config.l2)
        inputs = tf.keras.Input(shape=(input_width,), sparse=True, dtype='float32')
        x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizer)(inputs)
        x = tf.keras.layers.Dropout(config.dropout)(x)
        x = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizer)(x)
        x = tf.keras.layers.Dropout(config.dropout / 2)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        self.model = tf.keras.Model(inputs, outputs)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(config.learning_rate, clipnorm=1.),
                           loss=tf.keras.losses.BinaryCrossentropy())

    def _tensor(self, matrix):
        coo = matrix.tocoo()
        indices = np.column_stack([coo.row, coo.col]).astype(np.int64)
        value = self.tf.SparseTensor(indices=indices, values=coo.data.astype(np.float32),
                                     dense_shape=np.asarray(coo.shape, dtype=np.int64))
        return self.tf.sparse.reorder(value)

    def reset_metrics(self):
        self.model.reset_metrics()

    def train_batch(self, matrix, labels, weights):
        stats = self.model.train_on_batch(self._tensor(matrix), labels.astype(np.float32)[:, None],
                                          sample_weight=weights.astype(np.float32), return_dict=True)
        return float(stats['loss'])

    def predict_batch(self, matrix):
        return np.asarray(self.model.predict_on_batch(self._tensor(matrix))).reshape(-1)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)


def _predict_batches(adapter, matrix, batch_size):
    values = [adapter.predict_batch(matrix[start:start + batch_size])
              for start in range(0, matrix.shape[0], batch_size)]
    return _probabilities(np.concatenate(values), matrix.shape[0])


def fit_neural(matrices, fitting_labels, validation_labels, config, seed, adapter_factory=None):
    fitting, validation, scoring = matrices
    factory = adapter_factory or KerasSparseAdapter
    adapter = factory(fitting.shape[1], config, seed)
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=fitting_labels)
    weights = class_weights[fitting_labels.astype(int)]
    generator = np.random.default_rng(seed)
    history, best_weights, best_ap, best_epoch, stalled = [], None, -np.inf, None, 0
    for epoch in range(1, config.max_epochs + 1):
        adapter.reset_metrics()
        order = generator.permutation(len(fitting_labels))
        loss = np.nan
        for start in range(0, len(order), config.batch_size):
            indices = order[start:start + config.batch_size]
            loss = adapter.train_batch(fitting[indices], fitting_labels[indices], weights[indices])
        validation_p = _predict_batches(adapter, validation, config.batch_size)
        ap = float(average_precision_score(validation_labels, validation_p))
        cutoff = best_f1_threshold(validation_labels, validation_p)
        history.append({'epoch': epoch, 'fit_loss': float(loss), 'validation_pr_auc': ap,
                        'validation_f1': float(f1_score(validation_labels, validation_p >= cutoff))})
        print(f'  Epoch {epoch}/{config.max_epochs}: loss {loss:.4f}, validation AP {ap:.4f}', flush=True)
        if ap > best_ap:
            best_ap, best_epoch, stalled = ap, epoch, 0
            best_weights = [np.array(w, copy=True) for w in adapter.get_weights()]
        else:
            stalled += 1
        if stalled >= config.patience:
            break
    adapter.set_weights(best_weights)
    validation_p = _predict_batches(adapter, validation, config.batch_size)
    scoring_p = _predict_batches(adapter, scoring, config.batch_size)
    del adapter
    gc.collect()
    return validation_p, scoring_p, {'history': history, 'selected_epoch': best_epoch}


def choose_blend(validation_labels, probabilities, weights):
    candidates = []
    for network in [A2_NN, RICH_NN]:
        for weight in weights:
            values = (1 - weight) * probabilities[A2_LR] + weight * probabilities[network]
            cutoff = best_f1_threshold(validation_labels, values)
            metrics = score_summary(validation_labels, values, threshold=cutoff)
            candidates.append({'network': network, 'weight': float(weight), **metrics})
    # Stable iteration order favours A2 over the richer network when all keys tie.
    selected = max(candidates, key=lambda r: (r['f1'], r['pr_auc'], -r['weight']))
    return selected, candidates


def _cache_key(frame, config):
    digest = hashlib.sha256(frame.to_csv(index=False, lineterminator='\n').encode())
    for name in ['d1_sparse_nn.py', 'features.py', 'evaluation.py']:
        digest.update((Path(__file__).parent / name).read_bytes())
    # outer_folds controls how many rounds to run, not the training in each round.
    settings = asdict(config)
    settings.pop('outer_folds')
    digest.update(json.dumps({'config': settings, 'packages': package_versions(),
                              'python': platform.python_version()}, sort_keys=True).encode())
    return digest.hexdigest()


def run_experiment(frame, cache_dir, config=None):
    config = config or NeuralConfig()
    directory = Path(cache_dir) / _cache_key(frame, config)[:20]
    inputs = feature_inputs(frame, config)
    y, groups = frame[TARGET].to_numpy(), frame[GROUP].to_numpy()
    score_rows, records, histories, selections, dimensions = [], {}, [], [], []
    for fold, fitting, validation, scoring, outer in split_plan(frame, config):
        print(f'Fold {fold + 1}/5: fit {len(fitting)}, validation {len(validation)}, score {len(scoring)}', flush=True)
        path = directory / f'fold{fold + 1}.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                for key, expected in [('fitting', fitting), ('validation', validation), ('scoring', scoring)]:
                    if not np.array_equal(saved[key], expected):
                        raise ValueError('Cached assignments differ; move the cache aside and rerun.')
                val_p = {name: saved[f'v{i}'].copy() for i, name in enumerate(MODELS[:-1])}
                score_p = {name: saved[f'p{i}'].copy() for i, name in enumerate(MODELS[:-1])}
                details = json.loads(str(saved['details'].item()))
            print('  Reusing completed fold predictions.', flush=True)
        else:
            started = time.monotonic()
            matrices = {}
            width_rows = []
            for key, builder in feature_builders(config).items():
                a = sparse.csr_matrix(builder.fit_transform(inputs.iloc[fitting], y[fitting]), dtype=np.float32)
                b = sparse.csr_matrix(builder.transform(inputs.iloc[validation]), dtype=np.float32)
                c = sparse.csr_matrix(builder.transform(inputs.iloc[scoring]), dtype=np.float32)
                matrices[key] = (a, b, c)
                width_rows.append({'features': key, 'columns': int(a.shape[1]),
                                   'fit_nonzeros': int(a.nnz), 'fit_rows': int(a.shape[0])})
            val_p, score_p, model_details = {}, {}, {}
            for name, key in [(TEXT_LR, 'text'), (A2_LR, 'a2')]:
                print('  Fitting', name, flush=True)
                a, b, c = matrices[key]
                with threadpool_limits(limits=2):
                    classifier = LogisticRegression(C=1, class_weight='balanced', solver='liblinear',
                                                     max_iter=3000, random_state=config.seed).fit(a, y[fitting])
                    val_p[name], score_p[name] = classifier.predict_proba(b)[:, 1], classifier.predict_proba(c)[:, 1]
            for offset, (name, key) in enumerate([(A2_NN, 'a2'), (RICH_NN, 'rich')]):
                print('  Fitting', name, flush=True)
                vp, sp, extra = fit_neural(matrices[key], y[fitting], y[validation], config,
                                           seed=config.seed + fold * 10 + offset)
                val_p[name], score_p[name], model_details[name] = vp, sp, extra
            details = {'models': model_details, 'dimensions': width_rows}
            del matrices
            gc.collect()
            # Save only numeric probabilities + JSON; no fitted model or full feature matrices.
            directory.mkdir(parents=True, exist_ok=True)
            arrays = {f'v{i}': _probabilities(val_p[name], len(validation)) for i, name in enumerate(MODELS[:-1])}
            arrays.update({f'p{i}': _probabilities(score_p[name], len(scoring)) for i, name in enumerate(MODELS[:-1])})
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as output:
                np.savez_compressed(output, fitting=fitting, validation=validation, scoring=scoring,
                                    details=json.dumps(details), **arrays)
            temporary.replace(path)
            print(f'  Fold training completed in {(time.monotonic() - started)/60:.1f} minutes.', flush=True)
        for name in MODELS[:-1]:
            val_p[name] = _probabilities(val_p[name], len(validation))
            score_p[name] = _probabilities(score_p[name], len(scoring))
        selected, tried = choose_blend(y[validation], val_p, config.blend_weights)
        weight, network = selected['weight'], selected['network']
        val_p[BLEND] = (1 - weight) * val_p[A2_LR] + weight * val_p[network]
        score_p[BLEND] = (1 - weight) * score_p[A2_LR] + weight * score_p[network]
        selections.extend({'fold': fold + 1, 'selected': row == selected, **row} for row in tried)
        for name in MODELS:
            cutoff = selected['threshold'] if name == BLEND else best_f1_threshold(y[validation], val_p[name])
            metrics = score_summary(y[scoring], score_p[name], threshold=cutoff)
            metrics.update(model=name, fold=fold, repeat=0, train_rows=len(outer), val_rows=len(scoring),
                           fold_signature=fold_signature(y, groups, outer, scoring))
            score_rows.append(metrics)
            records[fold, name] = {'scoring_indices': scoring.copy(), 'probabilities': score_p[name].copy(),
                                   'decisions': score_p[name] >= cutoff, 'metrics': metrics}
        for name, detail in details['models'].items():
            histories.extend({'fold': fold + 1, 'model': name, **row} for row in detail['history'])
        dimensions.extend({'fold': fold + 1, **row} for row in details['dimensions'])
    return {'scores': pd.DataFrame(score_rows), 'records': records, 'history': pd.DataFrame(histories),
            'selection': pd.DataFrame(selections), 'dimensions': pd.DataFrame(dimensions),
            'cache_dir': str(directory), 'config': asdict(config)}


def pooled_results(result, frame):
    y = frame[TARGET].to_numpy()
    rows = []
    for name in MODELS:
        seen, decisions = np.zeros(len(y), dtype=int), np.zeros(len(y), dtype=bool)
        for (fold, model), entry in result['records'].items():
            if model == name:
                idx = entry['scoring_indices']
                seen[idx] += 1
                decisions[idx] = entry['decisions']
        if (seen > 1).any() or not (seen == 1).any():
            raise ValueError('Missing or duplicated scoring predictions.')
        idx = seen == 1
        tn, fp, fn, tp = confusion_matrix(y[idx], decisions[idx], labels=[0, 1]).ravel()
        rows.append({'model': name, 'evaluated_rows': int(idx.sum()), 'fraud_total': int(y[idx].sum()),
                     'precision': precision_score(y[idx], decisions[idx], zero_division=0),
                     'recall': recall_score(y[idx], decisions[idx], zero_division=0),
                     'f1': f1_score(y[idx], decisions[idx], zero_division=0),
                     'fraud_caught': tp, 'fraud_missed': fn, 'genuine_flagged': fp})
    return pd.DataFrame(rows).set_index('model')


def group_errors(result, frame, model=BLEND):
    parts = []
    for (fold, name), entry in result['records'].items():
        if name == model:
            idx, flags = entry['scoring_indices'], entry['decisions']
            labels = frame[TARGET].to_numpy()[idx]
            parts.append(pd.DataFrame({'fold': fold + 1, 'group': frame[GROUP].to_numpy()[idx],
                'fraud': labels, 'false_positives': (labels == 0) & flags, 'fraud_missed': (labels == 1) & ~flags}))
    return pd.concat(parts).groupby(['fold', 'group'], as_index=False).agg(
        rows=('fraud', 'size'), fraud=('fraud', 'sum'), false_positives=('false_positives', 'sum'),
        fraud_missed=('fraud_missed', 'sum')).sort_values('false_positives', ascending=False)


def prepare_colab_bundle(root):
    root = Path(root)
    payloads = {name: (root / name).read_bytes() for name in BUNDLE_FILES}
    manifest = {name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()}
    path = root / 'data' / 'processed' / '06_colab_inputs.zip'
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, content in payloads.items():
            bundle.writestr(name, content)
        bundle.writestr('manifest.json', json.dumps(manifest, indent=2))
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-colab', action='store_true')
    args = parser.parse_args()
    if args.prepare_colab:
        path = prepare_colab_bundle(Path(__file__).resolve().parents[1])
        print('Colab input ZIP:', path)
        print('Contains source code + frozen training CSV only; no final test or credentials.')
    else:
        parser.print_help()
