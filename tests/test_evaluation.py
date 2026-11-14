"""Run with: python -m unittest discover -s tests -v."""
import unittest

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation import (best_f1_threshold, cross_validate, make_folds,
                            paired_comparison, pooled_metrics, score_summary,
                            select_settings, summarise, validate_folds)


class UnseenGroupsClassifier(ClassifierMixin, BaseEstimator):
    """Fails if ANY prediction used for tuning/scoring is on a fitted group."""
    fits = []

    def __init__(self, reverse=False):
        self.reverse = reverse

    def fit(self, X, y):
        self.groups_ = set(X["group"])
        self.classes_ = np.array([0, 1])
        type(self).fits.append(self.groups_)
        return self

    def predict_proba(self, X):
        if not self.groups_.isdisjoint(X["group"]):
            raise AssertionError("Prediction includes a group used in this fit.")
        p = X["signal"].to_numpy()
        if self.reverse:
            p = 1 - p
        return np.c_[1 - p, p]


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        # Each group has repeated ads; row-wise splitting would trigger the spy.
        self.groups = np.repeat(np.arange(36), 2)
        self.y = np.repeat(np.arange(36) % 2, 2)
        self.X = pd.DataFrame({"group": self.groups, "signal": 0.2 + 0.6 * self.y})
        self.folds = make_folds(self.groups, self.y, n_splits=3, n_repeats=2)
        UnseenGroupsClassifier.fits = []

    def evaluate(self, **kwargs):
        return cross_validate(UnseenGroupsClassifier, self.X, self.y, self.groups,
                              folds=self.folds, inner_splits=2, verbose=False, **kwargs)

    def test_nested_tuning_and_threshold_never_predict_fitted_groups(self):
        scores, oof = self.evaluate(param_grid={"reverse": [True, False]})
        self.assertEqual(oof.shape, (2, len(self.y)))
        self.assertTrue(np.isfinite(oof).all())
        self.assertTrue((scores["pr_auc"] == 1).all())
        self.assertTrue((scores["f1"] == 1).all())
        self.assertTrue((scores["threshold"] == 0.8).all())
        self.assertTrue(all(p == {"reverse": False} for p in scores["selected_params"]))
        # Reuse the winning candidate's OOF predictions rather than fitting it again.
        self.assertEqual(len(UnseenGroupsClassifier.fits), 6 * (2 * 2 + 1))
        self.assertEqual(len(scores.attrs["inner_candidates"]), 12)

    def test_outer_validation_changes_do_not_change_its_selected_cutoff(self):
        scores, _ = self.evaluate(param_grid={"reverse": [True, False]})
        changed_X = self.X.copy()
        val_idx = self.folds[0][3]
        # Change both their score scale and their answers. A cutoff chosen on
        # these validation rows would now differ from the original 0.8 cutoff.
        changed_X.loc[val_idx, "signal"] = 0.05 + 0.1 * self.y[val_idx]
        changed_y = self.y.copy()
        changed_y[val_idx] = 1 - changed_y[val_idx]
        changed, _ = cross_validate(UnseenGroupsClassifier, changed_X, changed_y, self.groups,
                                    folds=self.folds, param_grid={"reverse": [True, False]},
                                    inner_splits=2, verbose=False)
        self.assertEqual(scores.iloc[0]["threshold"], changed.iloc[0]["threshold"])
        self.assertEqual(scores.iloc[0]["selected_params"], changed.iloc[0]["selected_params"])

    def test_selected_threshold_matches_independent_inner_oof(self):
        _, _, fit, _ = self.folds[0]
        selection = select_settings(UnseenGroupsClassifier, self.X.iloc[fit], self.y[fit],
                                    self.groups[fit], inner_splits=2)
        manual = np.full(len(fit), np.nan)
        for _, _, inner_fit, inner_val in make_folds(self.groups[fit], self.y[fit], 2, 1):
            model = UnseenGroupsClassifier().fit(self.X.iloc[fit].iloc[inner_fit], self.y[fit][inner_fit])
            manual[inner_val] = model.predict_proba(self.X.iloc[fit].iloc[inner_val])[:, 1]
        np.testing.assert_array_equal(manual, selection.oof)
        self.assertEqual(selection.threshold, best_f1_threshold(self.y[fit], manual))

    def test_fold_leakage_is_rejected_before_any_fit(self):
        bad = [(0, 0, np.arange(1, len(self.y)), np.array([0]))]
        with self.assertRaisesRegex(ValueError, "Related ads"):
            cross_validate(UnseenGroupsClassifier, self.X, self.y, self.groups, folds=bad)
        self.assertEqual(len(UnseenGroupsClassifier.fits), 0)

    def test_missing_duplicate_and_one_class_validation_folds_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly once"):
            validate_folds(self.folds[:-1], self.y, self.groups)
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_folds(self.folds + [self.folds[0]], self.y, self.groups)
        negatives = np.flatnonzero(self.y == 0)
        positives = np.flatnonzero(self.y == 1)
        with self.assertRaisesRegex(ValueError, "both classes"):
            validate_folds([(0, 0, negatives, positives)], self.y, self.groups)

    def test_too_few_groups_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "Too few groups"):
            make_folds(np.array([0, 1, 2, 3]), np.array([0, 0, 1, 1]), n_splits=3)

    def test_no_implicit_threshold_and_review_metrics_always_available(self):
        scores, oof = self.evaluate(threshold=None)
        self.assertNotIn("threshold", scores)
        self.assertNotIn("f1", scores)
        self.assertIn("precision_top5pct", scores)
        self.assertNotIn("threshold", pooled_metrics(oof, self.y))
        metrics = score_summary(self.y, oof[0], review_share=0.25)
        self.assertEqual(metrics["review_count"], 18)
        self.assertEqual(metrics["precision_at_budget"], 1)
        self.assertNotIn("precision_top5pct", metrics)

    def test_fixed_threshold_and_invalid_inputs(self):
        scores, _ = self.evaluate(threshold=0.5)
        self.assertTrue((scores["threshold"] == 0.5).all())
        self.assertEqual(len(UnseenGroupsClassifier.fits), len(self.folds))
        for value in ("training", -0.1, np.nan):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.evaluate(threshold=value)
        with self.assertRaises(ValueError):
            score_summary(self.y, np.full(len(self.y), np.nan))

    def test_counts_and_false_alarm_rate_match_the_actual_ads(self):
        metrics = score_summary(np.array([0, 1, 1, 0, 0, 1]),
                                np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.6]),
                                threshold=0.5, review_share=0.5)
        for name, expected in {'fraud_caught': 2, 'fraud_missed': 1,
                               'genuine_flagged': 2, 'genuine_passed': 1,
                               'review_count': 3, 'review_fraud_caught': 1,
                               'review_genuine_flagged': 2}.items():
            self.assertEqual(metrics[name], expected)
        self.assertAlmostEqual(metrics['false_positive_rate'], 2 / 3)

    def test_progress_callback_receives_only_the_scored_rows(self):
        events = []
        scores, _ = self.evaluate(on_fold=lambda row, indices, p: events.append((row, indices, p)))
        self.assertEqual(len(events), len(self.folds))
        for (row, indices, p), (_, _, _, expected) in zip(events, self.folds):
            np.testing.assert_array_equal(indices, expected)
            self.assertEqual(len(p), row['val_rows'])
            self.assertEqual(row['fraud_caught'] + row['fraud_missed'], int(self.y[indices].sum()))

    def test_paired_comparison_aligns_rows_and_rejects_different_splits(self):
        a, _ = self.evaluate(threshold=None)
        b = a.iloc[::-1].copy()
        b["pr_auc"] -= 0.1
        comparison = paired_comparison(a, b)
        self.assertAlmostEqual(comparison["mean_diff"], 0.1)
        self.assertAlmostEqual(comparison["low"], 0.1)
        b.loc[b.index[0], "fold_signature"] = "different rows"
        with self.assertRaisesRegex(ValueError, "different rows"):
            paired_comparison(a, b)

    def test_paired_false_alarm_comparison_treats_fewer_as_better(self):
        a, _ = self.evaluate()
        b = a.copy()
        b['false_positive_rate'] += 0.1
        comparison = paired_comparison(a, b, metric='false_positive_rate', higher_is_better=False)
        self.assertAlmostEqual(comparison['mean_diff'], -0.1)
        self.assertEqual(comparison['share_folds_a_better'], 1)

    def test_real_sklearn_pipeline_is_refitted_and_runs_end_to_end(self):
        def factory():
            return Pipeline([("scale", StandardScaler()), ("model", LogisticRegression())])
        scores, oof = cross_validate(factory, self.X[["signal"]], self.y, self.groups,
                                     folds=self.folds, inner_splits=2,
                                     param_grid={"model__C": [0.1, 1.0]}, verbose=False)
        self.assertTrue(np.isfinite(oof).all())
        self.assertTrue((scores["pr_auc"] == 1).all())
        self.assertNotIn("selection_pr_auc", summarise(scores, "test"))
        self.assertEqual(len(scores["fold_signature"].unique()), len(self.folds))


if __name__ == "__main__":
    unittest.main()
