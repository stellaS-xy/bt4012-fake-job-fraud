"""Protect recovered fold identity and strict cache reuse in experiment 13."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.d1_reference_comparison import prepare, restore_folds, read_prediction, save_prediction
from src.evaluation import fold_signature, make_folds
from src.reference13_runtime import training_lock, worker_active, ensure_inputs


class RecoveredProtocolTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({'job_id': np.arange(100), 'group_id': np.arange(100),
                                   'fraudulent': np.tile([0, 1], 50)})
        self.folds = make_folds(self.frame.group_id, self.frame.fraudulent, 5, 5)
        self.assignments = pd.concat([self.frame.iloc[val][['job_id', 'group_id']].assign(repeat=r, fold=k)
                                      for r, k, _, val in self.folds], ignore_index=True)
        self.expected = {f'{r}:{k}': fold_signature(self.frame.fraudulent, self.frame.group_id, fit, val)
                         for r, k, fit, val in self.folds}

    def test_saved_assignments_survive_reordering(self):
        restored = restore_folds(self.frame, self.assignments.sample(frac=1, random_state=12), self.expected)
        self.assertEqual(len(restored), 25)
        for original, recovered in zip(self.folds, restored):
            np.testing.assert_array_equal(original[3], recovered[3])

    def test_wrong_groups_are_rejected(self):
        changed = self.assignments.copy()
        changed.loc[0, 'group_id'] = 999
        with self.assertRaises(ValueError):
            restore_folds(self.frame, changed, self.expected)

    def test_different_labels_are_rejected_even_with_identical_ids(self):
        changed = self.frame.copy()
        changed.loc[0, 'fraudulent'] = 1
        with self.assertRaises(ValueError):
            restore_folds(changed, self.assignments, self.expected)

    def test_duplicate_or_missing_assignment_is_rejected(self):
        for changed in (pd.concat([self.assignments, self.assignments.iloc[:1]]), self.assignments.iloc[1:]):
            with self.assertRaises(ValueError):
                restore_folds(self.frame, changed, self.expected)

    def test_cached_predictions_require_identical_row_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fold.npz'
            save_prediction(path, np.array([1, 2]), np.array([.1, .9]), {'threshold': .4})
            p, detail = read_prediction(path, np.array([1, 2]))
            np.testing.assert_equal(p, [.1, .9])
            self.assertEqual(detail['threshold'], .4)
            with self.assertRaises(ValueError):
                read_prediction(path, np.array([2, 1]))

    def test_worker_status_uses_lock_instead_of_stale_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / 'progress.json').write_text(json.dumps({'status': 'running'}))
            self.assertFalse(worker_active(base))
            with training_lock(base / 'run.lock'):
                self.assertTrue(worker_active(base))
                with self.assertRaises(BlockingIOError):
                    with training_lock(base / 'run.lock'):
                        pass
            self.assertFalse(worker_active(base))

    def test_input_archive_checksum_is_checked_before_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / '13_reference_training_inputs.zip').write_bytes(b'wrong input archive')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                ensure_inputs(base, '0' * 64)
            self.assertFalse((base / 'data/processed/13_reference_comparison/reference').exists())


if __name__ == '__main__':
    unittest.main()
