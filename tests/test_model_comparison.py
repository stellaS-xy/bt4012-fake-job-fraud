import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from src.model_comparison import (BalancedClassifier, MODEL_KEYS, model_spec,
                                  repeat_summary, report_results)


class WeightRecorder(ClassifierMixin, BaseEstimator):
    def __init__(self, setting=1):
        self.setting = setting

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.array([0, 1])
        self.weights_ = np.asarray(sample_weight)
        return self

    def predict_proba(self, X):
        return np.tile([0.4, 0.6], (len(X), 1))


class ModelComparisonTests(unittest.TestCase):
    def test_class_weights_are_recomputed_from_each_fit(self):
        model = BalancedClassifier(WeightRecorder())
        X = np.zeros((4, 1))
        model.fit(X, np.array([0, 0, 0, 1]))
        np.testing.assert_allclose(model.estimator_.weights_, [2/3, 2/3, 2/3, 2])
        model.fit(X, np.array([0, 1, 1, 1]))
        np.testing.assert_allclose(model.estimator_.weights_, [2, 2/3, 2/3, 2/3])
        copied = clone(model).set_params(estimator__setting=3)
        self.assertEqual(copied.estimator.setting, 3)
        self.assertFalse(hasattr(copied, 'estimator_'))

    def test_repeat_rates_use_total_counts_not_unweighted_fold_averages(self):
        folds = pd.DataFrame([
            dict(repeat=0, fraud_caught=1, fraud_missed=1, genuine_flagged=1, genuine_passed=7,
                 review_count=1, review_fraud_caught=1, review_genuine_flagged=0, pr_auc=0.6),
            dict(repeat=0, fraud_caught=9, fraud_missed=0, genuine_flagged=9, genuine_passed=72,
                 review_count=4, review_fraud_caught=3, review_genuine_flagged=1, pr_auc=0.8),
        ])
        result = repeat_summary(folds).iloc[0]
        self.assertEqual(result['fraud_caught'], 10)
        self.assertEqual(result['genuine_flagged'], 10)
        self.assertAlmostEqual(result['recall'], 10/11)
        self.assertAlmostEqual(result['false_positive_rate'], 10/89)
        self.assertAlmostEqual(result['review_precision'], 4/5)

    def test_three_a2_models_have_matching_feature_settings(self):
        features = [model_spec(key)[0].named_steps['features']
                    for key in ('logreg_a2', 'lightgbm_a2', 'xgboost_a2')]
        # Their complete feature definitions, including vocabulary caps, match.
        self.assertEqual(repr(features[0]), repr(features[1]))
        self.assertEqual(repr(features[1]), repr(features[2]))
        self.assertEqual(model_spec('logreg_tfidf')[2], 30000)

    def test_report_produces_counts_repeat_results_and_uncertainty_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = {'models': list(MODEL_KEYS), 'repeats': 2, 'splits': 3, 'inner_splits': 2,
                    'training_rows': 72, 'fraud_rows': 36, 'genuine_rows': 36}
            (root/'run_plan.json').write_text(json.dumps(plan), encoding='utf-8')
            rows = []
            for repeat in range(2):
                for fold in range(3):
                    caught = 7 + repeat
                    rows.append(dict(repeat=repeat, fold=fold, fold_signature=f'{repeat}-{fold}',
                                     train_rows=48, val_rows=24, pr_auc=0.6 + repeat*0.01,
                                     fraud_caught=caught, fraud_missed=12-caught,
                                     genuine_flagged=2, genuine_passed=10,
                                     recall=caught/12, false_positive_rate=2/12,
                                     review_count=1, review_fraud_caught=1, review_genuine_flagged=0,
                                     precision_top5pct=1.0, recall_top5pct=1/12))
            for key in MODEL_KEYS:
                pd.DataFrame(rows).to_csv(root/f'{key}_fold_scores.csv', index=False)
            summary, paired = report_results(root)
            self.assertEqual(summary.loc['logreg_tfidf', 'fraud_caught'], 22.5)
            self.assertEqual(summary.loc['logreg_tfidf', 'genuine_flagged'], 6)
            self.assertEqual(len(paired), 30)
            for name in ('report.html', 'comparison.png', 'paired_uncertainty.png',
                         'comparison_summary.csv', 'results_by_repeat.csv',
                         'repeat_differences_vs_baseline.csv'):
                self.assertTrue((root/name).is_file(), name)


if __name__ == '__main__':
    unittest.main()
