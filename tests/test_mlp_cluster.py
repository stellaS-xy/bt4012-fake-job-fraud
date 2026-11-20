"""CPU-only protocol checks: no TensorFlow import, notebook execution or fits."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import d1_a2_mlp as mlp


class SavedMLPFoldsTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / 'assignments.csv'
        self.frame = pd.DataFrame({'job_id': [10, 20, 30, 40, 50, 60, 70, 80],
                                   'group_id': [1, 1, 2, 2, 3, 3, 4, 4],
                                   'fraudulent': [0, 1, 0, 1, 0, 1, 0, 1]})
        self.saved = self.frame[['job_id', 'group_id']].copy()
        self.saved['repeat'] = 0
        self.saved['fold'] = [0, 0, 1, 1, 0, 0, 1, 1]
        self.config = mlp.GroupedConfig(outer_splits=2)
        self.saved.to_csv(self.path, index=False)

    def test_restore_by_job_id_without_generating_folds(self):
        # Reordered development rows and shuffled assignment rows must preserve identities.
        self.saved.sample(frac=1, random_state=3).to_csv(self.path, index=False)
        frame = self.frame.iloc[::-1].reset_index(drop=True)
        with patch.object(mlp, 'make_folds', side_effect=AssertionError('must not regenerate')):
            folds = mlp.split_plan(frame, self.config, self.path)
        self.assertEqual(set(frame.iloc[folds[0][3]].job_id), {10, 20, 50, 60})

    def test_missing_assignments_stop(self):
        with self.assertRaises(FileNotFoundError):
            mlp.split_plan(self.frame, self.config)

    def test_duplicate_jobs_rejected(self):
        pd.concat([self.saved, self.saved.iloc[:1]]).to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, 'every development job once'):
            mlp.split_plan(self.frame, self.config, self.path)

    def test_wrong_groups_rejected(self):
        self.saved.loc[0, 'group_id'] = 999
        self.saved.to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, 'job/group identities'):
            mlp.split_plan(self.frame, self.config, self.path)

    def test_group_leakage_rejected(self):
        self.saved.loc[[0, 2], 'fold'] = [1, 0]
        self.saved.to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, 'Related ads'):
            mlp.split_plan(self.frame, self.config, self.path)

    def test_requested_repeats_must_exist(self):
        with self.assertRaisesRegex(ValueError, 'repeat 1'):
            mlp.split_plan(self.frame, mlp.GroupedConfig(outer_splits=2, repeats=2), self.path)


if __name__ == '__main__':
    unittest.main()
