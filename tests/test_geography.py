import unittest

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.base import clone

from src.geography import (HierarchicalLocationCount, auxiliary_metrics,
                           corrected_difference, geographic_pipeline, location_keys)


class GeographyTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({
            'location': ['US, CA, SF'] * 5 + ['US, NY, NYC'] * 5 + ['US, CA, SD'],
            'group_id': [1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        })

    def test_parser_uses_hierarchy_and_preserves_missing(self):
        rows = location_keys([' us ,  Ca , San   Francisco ', 'US,,Unknown', None])
        self.assertEqual(rows[0], (('US',), ('US', 'ca'), ('US', 'ca', 'san francisco')))
        self.assertEqual(rows[1], (('US',), None, None))
        self.assertEqual(rows[2], (None, None, None))

    def test_unseen_rare_missing_and_backoff_are_distinct(self):
        encoder = HierarchicalLocationCount(levels=('state', 'city')).fit(self.train)
        query = pd.DataFrame({'location': ['US, CA, LA', 'US, CA, SD', 'US, TX, Austin', 'ZZ, X, Y', None]})
        result = encoder.describe(query)
        self.assertEqual(result.loc[0, 'geo_city_resolved_level'], 'state')
        self.assertAlmostEqual(result.loc[0, 'geo_city_log_count'], np.log1p(6))
        self.assertEqual(result.loc[0, 'geo_city_unseen'], 1)
        self.assertEqual(result.loc[1, 'geo_city_unseen'], 0)  # rare but observed
        self.assertEqual(result.loc[1, 'geo_city_resolved_level'], 'state')
        self.assertEqual(result.loc[2, 'geo_state_resolved_level'], 'country')
        self.assertEqual(result.loc[3, 'geo_city_resolved_level'], 'UNSEEN')
        self.assertEqual(result.loc[3, 'geo_city_log_count'], 0)
        self.assertEqual(result.loc[4, 'geo_city_resolved_level'], 'MISSING')
        self.assertEqual(result.loc[4, 'geo_city_missing'], 1)
        self.assertEqual(result.loc[4, 'geo_city_unseen'], 0)

    def test_group_deduplication_is_per_group_per_location(self):
        rows = HierarchicalLocationCount(levels=('state', 'city')).fit(self.train)
        groups = HierarchicalLocationCount(levels=('state', 'city'), group_deduplicate=True).fit(self.train)
        key = ('US', 'ca', 'sf')
        self.assertEqual(rows.counts_[2][key], 5)
        self.assertEqual(groups.counts_[2][key], 3)
        query = self.train.iloc[:1]
        self.assertEqual(rows.describe(query).iloc[0].geo_city_resolved_level, 'city')
        self.assertEqual(groups.describe(query).iloc[0].geo_city_resolved_level, 'country')
        # One group at two distinct places contributes once to each place.
        extra = pd.DataFrame({'location': ['US, CA, SF', 'US, NY, NYC'], 'group_id': [20, 20]})
        both = HierarchicalLocationCount(group_deduplicate=True).fit(extra)
        self.assertEqual(both.counts_[0][('US',)], 1)
        self.assertEqual(both.counts_[1][('US', 'ca')], 1)
        self.assertEqual(both.counts_[1][('US', 'ny')], 1)

    def test_validation_rows_and_labels_do_not_change_counts(self):
        encoder = HierarchicalLocationCount(levels=('state', 'city')).fit(self.train, np.zeros(len(self.train)))
        before = [dict(counts) for counts in encoder.counts_]
        encoder.transform(pd.DataFrame({'location': ['US, TX, Austin'] * 100}))
        self.assertEqual(before, [dict(counts) for counts in encoder.counts_])
        other = clone(encoder).fit(self.train, np.ones(len(self.train)))
        self.assertEqual(before, [dict(counts) for counts in other.counts_])
        self.assertNotIn(('US', 'tx'), encoder.counts_[1])

    def test_variants_keep_a2_and_existing_country_features(self):
        base = geographic_pipeline('V0').named_steps['features'].transformers
        for variant in ('V1', 'V2', 'V3'):
            pipe = geographic_pipeline(variant)
            clone(pipe)
            parts = pipe.named_steps['features'].transformers
            self.assertEqual([p[0] for p in parts], [p[0] for p in base] + ['geography'])
            self.assertEqual([p[2] for p in parts[:-1]], [p[2] for p in base])
            self.assertEqual(sum(p[0] == 'country' for p in parts), 1)
            self.assertEqual(pipe.named_steps['model'].class_weight, 'balanced')
        self.assertEqual(geographic_pipeline('V0').named_steps['features'].get_params()['tfidf__max_features'], 30000)

    def test_auxiliary_metrics_include_fixed_review_budgets(self):
        y = np.r_[np.ones(20), np.zeros(80)]
        scores = np.linspace(1, 0, 100)
        values = auxiliary_metrics(y, scores)
        self.assertEqual(values['precision_at_recall60'], 1)
        self.assertEqual(values['precision_top2pct'], 1)
        self.assertAlmostEqual(values['recall_top2pct'], .1)
        self.assertAlmostEqual(values['recall_top5pct'], .25)

    def test_corrected_interval_and_fold_identity(self):
        baseline = pd.DataFrame({'repeat': [0]*3, 'fold': [0, 1, 2],
            'fold_signature': ['a', 'b', 'c'], 'train_rows': [80]*3, 'val_rows': [20]*3,
            'pr_auc': [.4]*3})
        candidate = baseline.copy()
        candidate['pr_auc'] = [.5, .6, .7]
        result = corrected_difference(candidate, baseline)
        self.assertAlmostEqual(result['mean_diff'], .2)
        self.assertAlmostEqual(result['sd_diff'], .1)
        se = np.sqrt((1/3 + 1/4)*.01)
        self.assertAlmostEqual(result['corrected_se'], se)
        self.assertAlmostEqual(result['low'], .2-t.ppf(.975, 2)*se)
        candidate.loc[0, 'fold_signature'] = 'changed'
        with self.assertRaises(ValueError):
            corrected_difference(candidate, baseline)


if __name__ == '__main__':
    unittest.main()
