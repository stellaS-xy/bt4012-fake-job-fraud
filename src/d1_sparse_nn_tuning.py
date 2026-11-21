"""Controlled notebook 06 follow-ups inspired by the supplied Kaggle screenshots.

The original helper stays unchanged so completed baseline caches remain usable.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from src import d1_sparse_nn as base
from src.evaluation import best_f1_threshold, fold_signature, score_summary
from src.features import GROUP, TARGET

RELAXED = '6. Relaxed-word rich MLP (128-32)'
DEEP = '7. Relaxed-word deep MLP (512-256-128-64)'
SELECTED = '8. Validation-selected original/new MLP blend'
REFERENCE = base.RICH_NN
MODELS = (base.A2_LR, REFERENCE, RELAXED, DEEP, SELECTED)
BUNDLE_FILES = (*base.BUNDLE_FILES, 'src/d1_sparse_nn_tuning.py')


def relaxed_features(config):
    """Match only the visible TF-IDF constructor; keep our cleaning/metadata."""
    builder = base.feature_builders(config)['rich']
    builder.set_params(a2__tfidf__min_df=2, a2__tfidf__max_df=.95,
                       a2__tfidf__stop_words=None, a2__tfidf__sublinear_tf=False)
    return builder


class DeepKerasAdapter(base.KerasSparseAdapter):
    """Screenshot-inspired shape/dropout; densify only individual mini-batches."""
    def __init__(self, input_width, config, seed):
        import tensorflow as tf
        self.tf = tf
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        regularizer = tf.keras.regularizers.l2(config.l2)
        inputs = tf.keras.Input(shape=(input_width,), dtype='float32')
        x = inputs
        for width, dropout in [(512, .4), (256, .4), (128, .3), (64, .3)]:
            x = tf.keras.layers.Dense(width, activation='relu', kernel_regularizer=regularizer)(x)
            x = tf.keras.layers.Dropout(dropout)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        self.model = tf.keras.Model(inputs, outputs)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(config.learning_rate, clipnorm=1.),
                           loss=tf.keras.losses.BinaryCrossentropy())

    def _tensor(self, matrix):
        # Dense GPU kernels may be faster; never densify the complete data matrix.
        return self.tf.convert_to_tensor(matrix.toarray(), dtype=self.tf.float32)


def choose_blend(labels, reference_p, candidates, weights):
    rows = []
    for name, candidate in candidates.items():
        for weight in weights:
            probabilities = (1 - weight) * reference_p + weight * candidate
            cutoff = best_f1_threshold(labels, probabilities)
            rows.append({'candidate': name, 'weight': float(weight),
                         **score_summary(labels, probabilities, threshold=cutoff)})
    # The zero option retains the unchanged original rich MLP; stable ties prefer compact.
    return max(rows, key=lambda row: (row['f1'], row['pr_auc'], -row['weight'])), rows


def run_followup(frame, reference_result, cache_dir):
    config = base.NeuralConfig(**reference_result['config'])
    expected_reference = base._cache_key(frame, config)[:20]
    if Path(reference_result['cache_dir']).name != expected_reference:
        raise ValueError('Reference data/code/config/packages differ from the current run. '
                         'Use the original completed result and its unchanged source/data.')
    key = hashlib.sha256((Path(__file__).read_text() + expected_reference).encode()).hexdigest()[:20]
    directory = Path(cache_dir) / key
    inputs = base.feature_inputs(frame, config)
    y, groups = frame[TARGET].to_numpy(), frame[GROUP].to_numpy()
    rows, records, history_rows, selection_rows, widths = [], {}, [], [], []
    for fold, fitting, validation, scoring, outer in base.split_plan(frame, config):
        print(f'Follow-up fold {fold + 1}/5; original networks will not be retrained.', flush=True)
        # Completed 06 stores validation probabilities in its ignored numeric cache.
        reference_path = Path(reference_result['cache_dir']) / f'fold{fold + 1}.npz'
        if not reference_path.is_file():
            raise FileNotFoundError('Original fold prediction cache is missing. Output tables alone '
                                    'cannot restore it; complete the original 06 run first.')
        with np.load(reference_path, allow_pickle=False) as saved:
            for key_name, expected in [('fitting', fitting), ('validation', validation), ('scoring', scoring)]:
                if not np.array_equal(saved[key_name], expected):
                    raise ValueError('Original prediction-cache assignments differ.')
            position = base.MODELS[:-1].index(REFERENCE)
            reference_validation = base._probabilities(saved[f'v{position}'], len(validation))
            reference_scoring = base._probabilities(saved[f'p{position}'], len(scoring))
        for name in [base.A2_LR, REFERENCE]:
            entry = reference_result['records'][(fold, name)]
            if not np.array_equal(entry['scoring_indices'], scoring):
                raise ValueError('The original result does not use the same scoring rows.')
            if entry['metrics']['fold_signature'] != fold_signature(y, groups, outer, scoring):
                raise ValueError('The original scoring labels/groups differ.')
            rows.append(entry['metrics'].copy())
            records[fold, name] = entry
        np.testing.assert_allclose(reference_scoring,
                                   reference_result['records'][(fold, REFERENCE)]['probabilities'])
        val_p, score_p, metadata = {}, {}, {}
        matrices = None
        for name, adapter, tag in [(RELAXED, base.KerasSparseAdapter, 'compact'),
                                    (DEEP, DeepKerasAdapter, 'deep')]:
            path = directory / f'fold{fold + 1}_{tag}.npz'
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    for key_name, expected in [('fitting', fitting), ('validation', validation), ('scoring', scoring)]:
                        if not np.array_equal(saved[key_name], expected):
                            raise ValueError('Follow-up prediction-cache assignments differ.')
                    vp, sp = saved['validation_p'].copy(), saved['scoring_p'].copy()
                    detail = json.loads(str(saved['details'].item()))
                print('  Reusing', name, flush=True)
            else:
                if matrices is None:
                    builder = relaxed_features(config)
                    a = sparse.csr_matrix(builder.fit_transform(inputs.iloc[fitting], y[fitting]), dtype=np.float32)
                    b = sparse.csr_matrix(builder.transform(inputs.iloc[validation]), dtype=np.float32)
                    c = sparse.csr_matrix(builder.transform(inputs.iloc[scoring]), dtype=np.float32)
                    matrices = (a, b, c)
                print('  Fitting', name, flush=True)
                started = time.monotonic()
                # Match the original rich MLP seed for both new candidates.
                vp, sp, detail = base.fit_neural(matrices, y[fitting], y[validation], config,
                    seed=config.seed + fold * 10 + 1, adapter_factory=adapter)
                detail['feature_columns'] = int(matrices[0].shape[1])
                vp, sp = base._probabilities(vp, len(validation)), base._probabilities(sp, len(scoring))
                directory.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix('.tmp')
                with temporary.open('wb') as output:
                    np.savez_compressed(output, fitting=fitting, validation=validation, scoring=scoring,
                                        validation_p=vp, scoring_p=sp, details=json.dumps(detail))
                temporary.replace(path)
                print(f'  Finished {name}: {(time.monotonic()-started)/60:.1f} minutes.', flush=True)
            val_p[name] = base._probabilities(vp, len(validation))
            score_p[name] = base._probabilities(sp, len(scoring))
            metadata[name] = detail
        del matrices
        gc.collect()
        chosen, candidates = choose_blend(y[validation], reference_validation, val_p, config.blend_weights)
        weight, name = chosen['weight'], chosen['candidate']
        val_p[SELECTED] = (1 - weight) * reference_validation + weight * val_p[name]
        score_p[SELECTED] = (1 - weight) * reference_scoring + weight * score_p[name]
        selection_rows.extend({'fold': fold + 1, 'selected': candidate == chosen, **candidate}
                              for candidate in candidates)
        for name in [RELAXED, DEEP, SELECTED]:
            cutoff = chosen['threshold'] if name == SELECTED else best_f1_threshold(y[validation], val_p[name])
            metrics = score_summary(y[scoring], score_p[name], threshold=cutoff)
            metrics.update(model=name, fold=fold, repeat=0, train_rows=len(outer), val_rows=len(scoring),
                           fold_signature=fold_signature(y, groups, outer, scoring))
            rows.append(metrics)
            records[fold, name] = {'scoring_indices': scoring.copy(), 'probabilities': score_p[name].copy(),
                                   'decisions': score_p[name] >= cutoff, 'metrics': metrics}
        for name, detail in metadata.items():
            history_rows.extend({'fold': fold + 1, 'model': name, **row} for row in detail['history'])
            widths.append({'fold': fold + 1, 'model': name, 'columns': detail['feature_columns'],
                           'selected_epoch': detail['selected_epoch']})
    return {'scores': pd.DataFrame(rows), 'records': records, 'history': pd.DataFrame(history_rows),
            'selection': pd.DataFrame(selection_rows), 'dimensions': pd.DataFrame(widths),
            'cache_dir': str(directory), 'config': asdict(config)}


def pooled_results(result, frame):
    labels, rows = frame[TARGET].to_numpy(), []
    for name in MODELS:
        seen, decisions = np.zeros(len(frame), dtype=int), np.zeros(len(frame), dtype=bool)
        for (fold, model), entry in result['records'].items():
            if model == name:
                idx = entry['scoring_indices']
                seen[idx] += 1
                decisions[idx] = entry['decisions']
        if (seen > 1).any() or not (seen == 1).any():
            raise ValueError('Incomplete or duplicated scoring predictions.')
        idx = seen == 1
        tn, fp, fn, tp = confusion_matrix(labels[idx], decisions[idx], labels=[0, 1]).ravel()
        rows.append({'model': name, 'evaluated_rows': int(idx.sum()), 'fraud_total': int(labels[idx].sum()),
                     'precision': precision_score(labels[idx], decisions[idx], zero_division=0),
                     'recall': recall_score(labels[idx], decisions[idx], zero_division=0),
                     'f1': f1_score(labels[idx], decisions[idx], zero_division=0),
                     'fraud_caught': tp, 'fraud_missed': fn, 'genuine_flagged': fp})
    return pd.DataFrame(rows).set_index('model')


def prepare_colab_bundle(root):
    root = Path(root)
    payloads = {name: (root / name).read_bytes() for name in BUNDLE_FILES}
    manifest = {name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()}
    path = root / 'data' / 'processed' / '06_colab_inputs.zip'
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, value in payloads.items():
            bundle.writestr(name, value)
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
