"""Protect recommendation mapping, undefined metrics and histogram counts."""
import unittest
from evaluator.dashboard_metrics import decision_metrics, score_distribution


class DashboardMetricTests(unittest.TestCase):
    def test_classifier_and_llm_are_separate_predictions(self):
        rows = [dict(fraudulent=1, classifier_prediction=False, recommendation='review_needed', agent_status='completed'),
                dict(fraudulent=0, classifier_prediction=True, recommendation='pass_no_review_needed', agent_status='completed')]
        self.assertEqual(decision_metrics(rows,'classifier')['f1'],0)
        result = decision_metrics(rows,'llm')
        self.assertEqual(result['precision'],1)
        self.assertEqual(result['recall'],1)
        self.assertEqual(result['f1'],1)
        self.assertEqual(result['true_negative'],1)

    def test_balanced_errors_produce_correct_prf(self):
        rows = [dict(fraudulent=y,classifier_prediction=p) for y,p in [(1,True),(1,False),(0,True),(0,False)]]
        result = decision_metrics(rows)
        self.assertEqual(result['evaluated'],4)
        for key in ('precision','recall','f1'):
            self.assertEqual(result[key],.5)

    def test_no_actual_fake_ads_does_not_invent_recall(self):
        rows = [dict(fraudulent=0,classifier_prediction=True),dict(fraudulent=0,classifier_prediction=False)]
        result = decision_metrics(rows)
        self.assertEqual(result['precision'],0)
        self.assertIsNone(result['recall'])
        self.assertEqual(result['f1'],0)
        result = decision_metrics(rows[1:])
        self.assertIsNone(result['precision'])
        self.assertIsNone(result['f1'])

    def test_missing_and_fallback_answers_are_not_llm_predictions(self):
        rows = [dict(fraudulent=1,agent_status='fallback',recommendation='review_needed'),
                dict(fraudulent=1,agent_status='completed',recommendation=None),
                dict(fraudulent=None,agent_status='completed',recommendation='review_needed'),
                dict(fraudulent=1,agent_status='completed',recommendation='review_needed')]
        result = decision_metrics(rows,'llm')
        self.assertEqual(result['evaluated'],1)
        self.assertEqual(result['total'],4)
        self.assertEqual(result['true_positive'],1)

    def test_histogram_includes_endpoints_and_excludes_invalid_and_test_scores(self):
        def row(score, **changes):
            output = dict(model_score=score,cutoff=.3,flagged=score >= .3,is_test_fixture=False) | changes
            return dict(case={'tool_trace':[dict(name='score_posting',status='success',output=output)]})
        result = score_distribution([row(0),row(.09),row(.3),row(1),row(-1),row(float('nan')),
                                     row(.2,is_test_fixture=True), row(.4,model_score=True)])
        self.assertEqual(result['count'],4)
        self.assertEqual(result['bins'][0]['below'],2)
        self.assertEqual(result['bins'][3]['flagged'],1)
        self.assertEqual(result['bins'][9]['flagged'],1)
        self.assertEqual(result['cutoffs'],[.3])
