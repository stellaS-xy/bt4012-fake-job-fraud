"""Separate row-stratified, cleaned-text, validation-loss neural experiment.

Reads only the frozen development CSV, never the reserved final-test CSV.
Group overlap is reported rather than prohibited under this requested protocol.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from threadpoolctl import threadpool_limits

from src.evaluation import best_f1_threshold, score_summary

TEXT_COLUMNS = ('title', 'company_profile', 'description', 'requirements', 'benefits')
CAT_COLUMNS = ('location', 'department', 'salary_range', 'employment_type', 'required_experience',
               'required_education', 'industry', 'function')
BINARY_COLUMNS = ('telecommuting', 'has_company_logo', 'has_questions')
TARGET = 'fraudulent'
TEXT_LR = 'Cleaned TF-IDF LR'
COMBINED_LR = 'Cleaned TF-IDF + categorical LR'
NETWORK = 'Cleaned TF-IDF + categorical deep MLP'
BUNDLE_FILES = ('src/d1_row_neural.py', 'src/evaluation.py', 'data/processed/d1_train.csv')


@dataclass(frozen=True)
class RowConfig:
    seed: int = 42
    scoring_fraction: float = .2
    validation_fraction: float = .2  # Fraction of the remaining rows.
    word_features: int = 5000
    max_epochs: int = 50
    patience: int = 5
    batch_size: int = 64
    learning_rate: float = .001

    def __post_init__(self):
        if not 0 < self.scoring_fraction < 1 or not 0 < self.validation_fraction < 1:
            raise ValueError('Split fractions must be between zero and one.')
        if min(self.word_features, self.max_epochs, self.patience, self.batch_size) < 1 or self.learning_rate <= 0:
            raise ValueError('Feature/training settings must be positive.')


def package_versions():
    return {name: importlib.metadata.version(name) for name in
            ['numpy', 'pandas', 'scipy', 'scikit-learn', 'tensorflow', 'keras', 'nltk']}


def load_training(path):
    frame = pd.read_csv(path)
    if not frame.job_id.is_unique or set(frame[TARGET].unique()) != {0, 1} or frame.group_id.isna().any():
        raise ValueError('Need unique job IDs, both fraud labels and 00\'s groups for overlap diagnostics.')
    digest = hashlib.sha256(frame.to_csv(index=False, lineterminator='\n').encode()).hexdigest()
    return frame, digest


def stratified_rows(frame, config):
    indices, labels = np.arange(len(frame)), frame[TARGET].to_numpy()
    remaining, scoring = train_test_split(indices, test_size=config.scoring_fraction,
                                          random_state=config.seed, stratify=labels)
    fitting, validation = train_test_split(remaining, test_size=config.validation_fraction,
        random_state=config.seed + 1, stratify=labels[remaining])
    if len(np.unique(np.concatenate([fitting, validation, scoring]))) != len(frame):
        raise ValueError('Row assignments are duplicated or incomplete.')
    return fitting, validation, scoring


def group_overlap(frame, assignments):
    fitting, validation, scoring = assignments
    rows = []
    for name, left, right in [('fit → validation', fitting, validation), ('fit → score', fitting, scoring),
                               ('validation → score', validation, scoring)]:
        left_groups = set(frame.group_id.iloc[left])
        right_values = frame.group_id.iloc[right]
        rows.append({'pair': name, 'shared_groups': len(left_groups & set(right_values)),
                     'right_rows_in_shared_groups': int(right_values.isin(left_groups).sum()),
                     'right_row_share': float(right_values.isin(left_groups).mean())})
    return pd.DataFrame(rows)


class TextCleaner:
    """Match the supplied alphabetic/stopword/noun-lemmatization recipe."""
    def __init__(self, stop_words=None, lemmatize=None):
        if stop_words is None or lemmatize is None:
            import nltk
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer
            try:
                words = stopwords.words('english')
                WordNetLemmatizer().lemmatize('dogs')
            except LookupError:
                print('Downloading public NLTK stopwords and WordNet resources.', flush=True)
                for resource in ['stopwords', 'wordnet']:
                    nltk.download(resource, quiet=True, raise_on_error=True)
                words = stopwords.words('english')
            if stop_words is None:
                stop_words = set(words)
            if lemmatize is None:
                lemmatize = WordNetLemmatizer().lemmatize
        self.stop_words = set(stop_words)
        self.lemmatize = lru_cache(maxsize=100000)(lemmatize)

    def __call__(self, text):
        text = str(text).lower()
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'http[s]?://\S+', ' ', text)
        text = re.sub(r'\S+@\S+', ' ', text)
        text = re.sub(r'[^a-z\s]', ' ', text)
        # Only ASCII letters/whitespace remain, so split() avoids Punkt resources.
        return ' '.join(self.lemmatize(word) for word in text.split()
                        if word not in self.stop_words and len(word) > 2)


def cleaned_texts(frame, cache_dir, cleaner=None):
    raw = frame[list(TEXT_COLUMNS)].fillna('').astype(str).agg(' '.join, axis=1)
    key = hashlib.sha256(raw.to_csv(index=False, lineterminator='\n').encode())
    key.update(Path(__file__).read_bytes())
    key.update(importlib.metadata.version('nltk').encode())
    path = Path(cache_dir) / f'cleaned_{key.hexdigest()[:20]}.json'
    if path.is_file():
        texts = json.loads(path.read_text())
        if len(texts) != len(frame) or not all(isinstance(text, str) for text in texts):
            raise ValueError('Invalid cleaned-text cache.')
        print('Reusing cleaned texts.', flush=True)
        return texts
    cleaner = cleaner or TextCleaner()
    texts = []
    for index, text in enumerate(raw, 1):
        texts.append(cleaner(text))
        if index % 1000 == 0 or index == len(raw):
            print(f'Cleaned {index:,}/{len(raw):,} postings.', flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(texts))
    temporary.replace(path)
    return texts


def build_matrices(frame, texts, assignments, config):
    fitting, validation, scoring = assignments
    texts = np.asarray(texts, dtype=object)
    vectorizer = TfidfVectorizer(max_features=config.word_features, ngram_range=(1, 2),
                                 min_df=2, max_df=.95, dtype=np.float32)
    word_fit = vectorizer.fit_transform(texts[fitting])
    word_val = vectorizer.transform(texts[validation])
    word_score = vectorizer.transform(texts[scoring])
    categories = frame[list(CAT_COLUMNS)].fillna('missing').astype(str)
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float32)
    cat_fit = encoder.fit_transform(categories.iloc[fitting])
    cat_val = encoder.transform(categories.iloc[validation])
    cat_score = encoder.transform(categories.iloc[scoring])
    binary = frame[list(BINARY_COLUMNS)].fillna(0).to_numpy(dtype=np.float32)
    words = (word_fit.tocsr(), word_val.tocsr(), word_score.tocsr())
    combined = tuple(sparse.hstack([word, cat, sparse.csr_matrix(binary[idx])], format='csr', dtype=np.float32)
                     for word, cat, idx in zip(words, [cat_fit, cat_val, cat_score], assignments))
    return words, combined, {'word_columns': int(word_fit.shape[1]),
                             'categorical_binary_columns': int(combined[0].shape[1] - word_fit.shape[1]),
                             'total_columns': int(combined[0].shape[1])}


class KerasAdapter:
    def __init__(self, width, config):
        import tensorflow as tf
        self.tf = tf
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(config.seed)
        inputs = tf.keras.Input(shape=(width,), dtype='float32')
        x = inputs
        for units, dropout in [(512, .4), (256, .4), (128, .3), (64, .3)]:
            x = tf.keras.layers.Dense(units, activation='relu')(x)
            x = tf.keras.layers.Dropout(dropout)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        self.model = tf.keras.Model(inputs, outputs)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(config.learning_rate), loss='binary_crossentropy')

    def reset_metrics(self):
        self.model.reset_metrics()

    def train_batch(self, matrix, labels, weights):
        values = self.tf.convert_to_tensor(matrix.toarray(), dtype=self.tf.float32)
        return float(self.model.train_on_batch(values, labels.astype(np.float32)[:, None],
                     sample_weight=weights.astype(np.float32), return_dict=True)['loss'])

    def predict_batch(self, matrix):
        return np.asarray(self.model.predict_on_batch(
            self.tf.convert_to_tensor(matrix.toarray(), dtype=self.tf.float32))).reshape(-1)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)


def predict_batches(adapter, matrix, batch_size):
    p = np.concatenate([adapter.predict_batch(matrix[start:start + batch_size])
                        for start in range(0, matrix.shape[0], batch_size)])
    if p.shape != (matrix.shape[0],) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid probabilities.')
    return p


def fit_network(matrices, fitting_labels, validation_labels, config, adapter_factory=None):
    fitting, validation, scoring = matrices
    adapter = (adapter_factory or KerasAdapter)(fitting.shape[1], config)
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=fitting_labels)
    weights = class_weights[fitting_labels.astype(int)]
    rng = np.random.default_rng(config.seed)
    history, best_weights, best_loss, best_epoch, stalled = [], None, np.inf, None, 0
    for epoch in range(1, config.max_epochs + 1):
        adapter.reset_metrics()
        order = rng.permutation(len(fitting_labels))
        loss = np.nan
        for start in range(0, len(order), config.batch_size):
            idx = order[start:start + config.batch_size]
            loss = adapter.train_batch(fitting[idx], fitting_labels[idx], weights[idx])
        p = predict_batches(adapter, validation, config.batch_size)
        # Natural-prevalence, unweighted BCE; no L2 term is used in this model.
        validation_loss = float(log_loss(validation_labels, np.clip(p.astype(float), 1e-7, 1 - 1e-7), labels=[0, 1]))
        ap = float(average_precision_score(validation_labels, p))
        history.append({'epoch': epoch, 'fit_loss': loss, 'validation_loss': validation_loss, 'validation_pr_auc': ap})
        print(f'Epoch {epoch}/{config.max_epochs}: train loss {loss:.4f}, '
              f'validation loss {validation_loss:.4f}, validation AP {ap:.4f}', flush=True)
        if validation_loss < best_loss:
            best_loss, best_epoch, stalled = validation_loss, epoch, 0
            best_weights = [np.array(w, copy=True) for w in adapter.get_weights()]
        else:
            stalled += 1
        if stalled >= config.patience:
            break
    adapter.set_weights(best_weights)
    vp = predict_batches(adapter, validation, config.batch_size)
    sp = predict_batches(adapter, scoring, config.batch_size)
    del adapter
    gc.collect()
    return vp, sp, {'history': history, 'selected_epoch': best_epoch, 'checkpoint_monitor': 'validation_loss'}


def run_experiment(frame, texts, cache_dir, config=None):
    config = config or RowConfig()
    assignments = stratified_rows(frame, config)
    fitting, validation, scoring = assignments
    key = hashlib.sha256(frame.to_csv(index=False, lineterminator='\n').encode())
    key.update(json.dumps(list(texts)).encode())
    for name in ['d1_row_neural.py', 'evaluation.py']:
        key.update((Path(__file__).parent / name).read_bytes())
    key.update(json.dumps({'config': asdict(config), 'versions': package_versions()}, sort_keys=True).encode())
    path = Path(cache_dir) / f'row_screen_{key.hexdigest()[:20]}.npz'
    names = [TEXT_LR, COMBINED_LR, NETWORK]
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            for name, indices in zip(['fitting', 'validation', 'scoring'], assignments):
                if not np.array_equal(saved[name], indices):
                    raise ValueError('Cache row assignments differ.')
            vp = {name: saved[f'v{i}'].copy() for i, name in enumerate(names)}
            sp = {name: saved[f'p{i}'].copy() for i, name in enumerate(names)}
            details = json.loads(str(saved['details'].item()))
        print('Reusing completed row-split screen.', flush=True)
    else:
        word_matrices, combined_matrices, widths = build_matrices(frame, texts, assignments, config)
        y, vp, sp = frame[TARGET].to_numpy(), {}, {}
        for name, matrices in [(TEXT_LR, word_matrices), (COMBINED_LR, combined_matrices)]:
            print('Fitting', name, flush=True)
            with threadpool_limits(limits=2):
                model = LogisticRegression(C=1, class_weight='balanced', solver='liblinear', max_iter=3000,
                                           random_state=config.seed).fit(matrices[0], y[fitting])
                vp[name], sp[name] = model.predict_proba(matrices[1])[:, 1], model.predict_proba(matrices[2])[:, 1]
        vp[NETWORK], sp[NETWORK], neural = fit_network(combined_matrices, y[fitting], y[validation], config)
        details = {'widths': widths, **neural}
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f'v{i}': vp[name] for i, name in enumerate(names)}
        arrays.update({f'p{i}': sp[name] for i, name in enumerate(names)})
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as output:
            np.savez_compressed(output, fitting=fitting, validation=validation, scoring=scoring,
                                details=json.dumps(details), **arrays)
        temporary.replace(path)
    y, rows, thresholds = frame[TARGET].to_numpy(), [], {}
    for name in names:
        if vp[name].shape != (len(validation),) or sp[name].shape != (len(scoring),):
            raise ValueError('Prediction rows are misaligned.')
        cutoff = best_f1_threshold(y[validation], vp[name])
        thresholds[name] = cutoff
        rows.append({'model': name + ' (validation F1 threshold)', **score_summary(y[scoring], sp[name], threshold=cutoff)})
    rows.append({'model': NETWORK + ' (fixed 0.5)', **score_summary(y[scoring], sp[NETWORK], threshold=.5)})
    return {'scores': pd.DataFrame(rows).set_index('model'), 'history': pd.DataFrame(details['history']),
            'selected_epoch': details['selected_epoch'], 'widths': details['widths'], 'thresholds': thresholds,
            'probabilities': sp, 'assignments': assignments, 'cache_path': str(path)}


def prepare_colab_bundle(root):
    root = Path(root)
    payloads = {name: (root / name).read_bytes() for name in BUNDLE_FILES}
    manifest = {name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()}
    path = root / 'data' / 'processed' / '07_colab_inputs.zip'
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
        print('Prepared:', prepare_colab_bundle(Path(__file__).resolve().parents[1]))
    else:
        parser.print_help()
