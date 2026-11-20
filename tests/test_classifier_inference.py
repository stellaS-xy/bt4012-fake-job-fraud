"""Provenance checks for exporting a completed comparison as an artifact."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src import classifier_inference as inference


class ComparisonSelectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / 'src').mkdir()
        self.source_hashes = {}
        for name in ('features.py', 'evaluation.py', 'model_comparison.py'):
            path = self.root / 'src' / name
            path.write_text(name)
            self.source_hashes[name] = inference.fingerprint(path)
        self.train = self.root / 'train.csv'
        self.train.write_text('training data only')
        self.results = self.root / 'results'
        self.results.mkdir()
        self.plan_path = self.results / 'lightgbm_a2_2k_plan.json'
        self.scores_path = self.results / 'lightgbm_a2_2k_fold_scores.csv'
        self.complete_path = self.results / 'lightgbm_a2_2k_complete.json'
        self.scores = pd.DataFrame([
            dict(repeat=0, fold=0, fold_signature='fold-a', threshold=0.2,
                 selected_params='{"model__estimator__num_leaves": 31}'),
            dict(repeat=0, fold=1, fold_signature='fold-b', threshold=0.4,
                 selected_params='{"model__estimator__num_leaves": 31}'),
        ])
        self.plan = dict(training_sha256=inference.fingerprint(self.train),
                         source_sha256=self.source_hashes,
                         outer_signatures={'0:0': 'fold-a', '0:1': 'fold-b'},
                         model='lightgbm_a2_2k', versions={'scikit-learn': 'recorded'})
        self.plan_path.write_text(json.dumps(self.plan))
        self.save_scores()
        self.patcher = patch.object(inference, 'ROOT', self.root)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def save_scores(self):
        self.scores.to_csv(self.scores_path, index=False)
        self.complete_path.write_text(json.dumps(dict(
            folds=len(self.scores), scores_sha256=inference.fingerprint(self.scores_path))))

    def test_selects_revised_results_and_records_their_provenance(self):
        _, params, cutoff, provenance = inference.comparison_selection(self.results, self.train)
        self.assertEqual(params, {'model__estimator__num_leaves': 31})
        self.assertAlmostEqual(cutoff, 0.3)
        self.assertEqual(provenance['comparison_model'], 'lightgbm_a2_2k')
        self.assertEqual(provenance['comparison_folds'], 2)
        self.assertEqual(provenance['comparison_scores_sha256'], inference.fingerprint(self.scores_path))

    def test_rejects_changed_training_data(self):
        self.train.write_text('different data')
        with self.assertRaisesRegex(ValueError, 'Training data differs'):
            inference.comparison_selection(self.results, self.train)

    def test_rejects_changed_model_or_feature_implementation(self):
        for name in self.source_hashes:
            with self.subTest(name=name):
                path = self.root / 'src' / name
                path.write_text('changed')
                with self.assertRaisesRegex(ValueError, 'differs from the saved comparison'):
                    inference.comparison_selection(self.results, self.train)
                path.write_text(name)

    def test_rejects_scores_modified_since_completion(self):
        with self.scores_path.open('a') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            inference.comparison_selection(self.results, self.train)

    def test_rejects_mismatched_fold_identity_even_with_updated_hash(self):
        self.scores.loc[0, 'fold_signature'] = 'different-fold'
        self.save_scores()
        with self.assertRaisesRegex(ValueError, 'outer fold plan'):
            inference.comparison_selection(self.results, self.train)

    def test_rejects_invalid_cutoffs(self):
        self.scores.loc[0, 'threshold'] = 1.5
        self.save_scores()
        with self.assertRaisesRegex(ValueError, 'cutoffs'):
            inference.comparison_selection(self.results, self.train)

    def test_rejects_duplicate_fold_records(self):
        self.scores.loc[1, 'fold'] = 0
        self.save_scores()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            inference.comparison_selection(self.results, self.train)


if __name__ == '__main__':
    unittest.main()
