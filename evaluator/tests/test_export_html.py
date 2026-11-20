"""Guard reporting denominators, score exclusions and untrusted text rendering."""
import unittest
from evaluator.export_html import render, classifier_panel, judge_panel


class HtmlReportTests(unittest.TestCase):
    def row(self, **changes):
        values = dict(case_id='1', fraudulent=0, classifier_prediction=False,
                      recommendation='pass_no_review_needed', agent_status='completed',
                      code={'all_passed': True, 'checks': {}}, judge_status='completed',
                      scoring={'effective_scores': {'evidence_grounding': 4}},
                      judgment={}, case={'posting': {'title': '<script>alert("x")</script>',
                          'description': 'Use < and > literally.'}, 'assessment': {}, 'tool_trace': []})
        return values | changes

    def test_matrix_uses_classifier_predictions_and_excludes_missing_ones(self):
        rows = [self.row(recommendation='review_needed'), self.row(case_id='2', classifier_prediction=True),
                self.row(case_id='3', fraudulent=1, classifier_prediction=None)]
        html = classifier_panel(rows)
        self.assertIn('2 scored postings', html)
        self.assertIn('1 predicted fake', html)
        self.assertIn('No fake ads in this selection', html)
        self.assertIn('N/A', html)
        self.assertIn('<strong>0%</strong>', html)
        self.assertLess(html.index('Precision'), html.index('<h2>Confusion matrix'))
        self.assertNotIn('Classifier metrics', html)

    def test_raw_judge_ratings_cannot_fill_excluded_scores(self):
        row = self.row(scoring={'effective_scores': {'evidence_grounding': None}},
                       judgment={'evidence_grounding': {'score': 4}})
        html = judge_panel([row], {})
        self.assertIn('N/A', html)
        self.assertNotIn('4.0', html)
        self.assertIn('0 scored evaluations', html)

    def test_untrusted_fields_are_escaped_and_removed_sections_stay_absent(self):
        html = render({'rows': [self.row()]})
        self.assertIn('&lt;script&gt;alert', html)
        self.assertNotIn('<script>alert', html)
        self.assertIn('Use &lt; and &gt; literally.', html)
        for text in ('What to note', 'Run summary', 'Inputs, actual tool outputs',
                     "Read the run's input source", 'Exploratory use:'):
            self.assertNotIn(text, html)
        self.assertEqual(html.count('class="section"'), 5)
        self.assertNotIn('<details class="section" open', html)

    def test_empty_report_has_no_invented_performance(self):
        html = render({'rows': []})
        self.assertIn('No labelled, valid classifier predictions', html)
        self.assertIn('0 of 0 shown', html)
        self.assertNotIn('<strong>100%</strong>', html)
