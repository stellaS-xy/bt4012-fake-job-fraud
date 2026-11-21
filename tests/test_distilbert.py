"""Protocol/token tests only: no torch import, downloads, forward passes or fits."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import d1_distilbert as experiment
from src.evaluation import make_folds


class FakeTokenizer:
    cls_token_id = 101
    sep_token_id = 102

    def encode(self, text, **kwargs):
        # Stable synthetic IDs; never uses a network or real model tokenizer.
        return [sum(map(ord, word)) for word in text.split()]


class DistilbertProtocolTests(unittest.TestCase):
    def test_late_fields_survive_long_description(self):
        row = {field: ' '.join(['long'] * 1000) for field in experiment.FIELDS}
        row['requirements'] = 'UNIQUE_REQUIREMENT'
        row['benefits'] = 'UNIQUE_BENEFIT'
        ids, clipped = experiment.encode_posting(row, FakeTokenizer())
        self.assertLessEqual(len(ids), 256)
        self.assertEqual(ids.count(102), 5)
        self.assertIn(sum(map(ord, 'UNIQUE_REQUIREMENT')), ids)
        self.assertIn(sum(map(ord, 'UNIQUE_BENEFIT')), ids)
        self.assertTrue(clipped[2])

    def test_head_tail_keeps_ending_without_overrun(self):
        self.assertEqual(experiment.head_tail(list(range(10)), 5), [0, 1, 2, 8, 9])
        self.assertEqual(experiment.head_tail([1, 2], 8), [1, 2])

    def test_missing_fields_are_explicit(self):
        row = {field: np.nan for field in experiment.FIELDS}
        ids, clipped = experiment.encode_posting(row, FakeTokenizer())
        self.assertEqual(ids.count(sum(map(ord, 'missing'))), 5)
        self.assertFalse(any(clipped))

    def test_invalid_memory_settings_rejected(self):
        for kwargs in [{'batch_size': 0}, {'max_length': 128}, {'accumulation': 0}, {'device': 'gpu'}]:
            with self.assertRaises(ValueError):
                experiment.Config(**kwargs)

    def test_outer_labels_never_enter_inner_training_or_threshold_selection(self):
        frame = pd.DataFrame({'job_id': np.arange(60),
                              'group_id': np.repeat(np.arange(30), 2),
                              'fraudulent': np.tile([0, 1], 30)})
        folds = make_folds(frame.group_id, frame.fraudulent, n_splits=5, n_repeats=1)
        outer_fit, outer_score = folds[0][2:]
        calls, threshold_labels = [], []

        def no_fit(frame, ids, pad, fitting, scoring, config, seed, path):
            self.assertTrue(set(frame.iloc[fitting].group_id).isdisjoint(frame.iloc[scoring].group_id))
            calls.append((fitting.copy(), scoring.copy()))
            return np.full(len(scoring), .25), {'selected_epoch': 1}

        original_threshold = experiment.best_f1_threshold

        def record_threshold(y, p):
            threshold_labels.append(y.copy())
            return original_threshold(y, p)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'data/processed').mkdir(parents=True)
            (root / 'data/processed/model_cv_assignments.csv').write_text('test fixture')
            with patch.object(experiment, '_fit_predict', side_effect=no_fit), \
                 patch.object(experiment, 'best_f1_threshold', side_effect=record_threshold), \
                 patch.object(experiment, 'device_name', return_value='test'), \
                 patch.object(experiment, 'versions', return_value={}):
                result = experiment.run(root, frame, folds, None, 0, experiment.Config(repeats=1), root / 'cache')
            self.assertEqual(len(calls), 4)  # 3 inner prediction phases + 1 outer
            for fitting, scoring in calls[:3]:
                self.assertTrue(set(fitting).issubset(set(outer_fit)))
                self.assertTrue(set(scoring).issubset(set(outer_fit)))
                self.assertTrue(set(scoring).isdisjoint(set(outer_score)))
            np.testing.assert_array_equal(calls[-1][1], outer_score)
            np.testing.assert_array_equal(threshold_labels[0], frame.fraudulent.to_numpy()[outer_fit])
            self.assertEqual(len(result['scores']), 1)
            self.assertFalse('delta_corrected_low' in result['comparison'])
            manifest = json.loads((Path(result['output_dir']) / 'run_manifest.json').read_text())
            self.assertEqual(manifest['completed_outer_folds'], 1)
            self.assertEqual(set(result['sources'].status), {'excluded'})

    def test_historical_different_folds_excluded(self):
        rows = pd.DataFrame({'repeat': [0], 'fold': [0], 'fold_signature': ['current'],
                             'train_rows': [8], 'val_rows': [2], 'pr_auc': [.5]})
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'reports/tables'
            path.mkdir(parents=True)
            historical = rows.copy()
            historical['fold_signature'] = 'different'
            historical.to_csv(path / 'model_logreg_a2_fold_scores.csv', index=False)
            summary, sources = experiment.compare_saved(root, rows)
        self.assertEqual(len(summary), 1)
        reason = sources.loc[sources.model == '02 A2 logistic 30k', 'reason'].item()
        self.assertIn('identities differ', reason)


if __name__ == '__main__':
    unittest.main()
