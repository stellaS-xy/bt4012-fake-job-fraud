"""Exercise both-tool pass and every reason a recommendation needs review."""
import unittest
from agent.core.policy import classifier_check, lexicon_check, recommendation_policy
from agent.core.schemas import Assessment, Posting, ToolTrace
from agent.core.validation import validate_assessment
from evaluator.checks import code_checks
from agent.core.schemas import ReviewCase


class RecommendationPolicyTests(unittest.TestCase):
    def setUp(self):
        self.posting = Posting(posting_id='policy', title='Assistant', description='Routine office duties.')
        self.score = dict(posting_id='policy', model_score=.1, cutoff=.3, flagged=False, is_test_fixture=False)
        self.lexicon = dict(is_test_fixture=False, context_status='completed', evidence=[])

    def trace(self, score=None, lexicon=None):
        return [ToolTrace(name=name, arguments={'posting_id': 'policy'}, status='success', output=output)
                for name, output in [('score_posting', self.score if score is None else score),
                                     ('match_lexicon', self.lexicon if lexicon is None else lexicon)]]

    def assessment(self, action):
        return Assessment(posting_id='policy', model_score=.1, cutoff=.3, flagged=False,
                          recommendation=action, rationale='Both checks passed; no review is needed.',
                          evidence_ids=[], limitations=['External facts were not checked.'],
                          next_steps=[], uses_test_fixtures=False)

    def test_both_tools_pass_without_optional_posting_fields(self):
        trace = self.trace()
        self.assertEqual(recommendation_policy(trace)['recommendation'], 'pass_no_review_needed')
        validate_assessment(self.assessment('pass_no_review_needed'), self.posting, trace)
        case = ReviewCase(posting=self.posting, scenario='ordinary', assessment=self.assessment('pass_no_review_needed'),
                          evidence=[], tool_trace=trace, model_backend='ollama_local', model_name='test-model')
        self.assertTrue(code_checks(self.posting, case)['all_passed'])

    def test_unnecessary_review_cannot_replace_a_pass(self):
        with self.assertRaisesRegex(ValueError, 'both tool checks'):
            validate_assessment(self.assessment('review_needed'), self.posting, self.trace())

    def test_classifier_flag_including_exact_cutoff_needs_review(self):
        for score in (.3, .9):
            check = recommendation_policy(self.trace(self.score | {'model_score': score, 'flagged': True}))
            self.assertEqual(check['recommendation'], 'review_needed')
            self.assertTrue(check['review_reasons'])

    def test_placeholder_missing_invalid_or_inconsistent_score_cannot_pass(self):
        for changes in ({'is_test_fixture': True}, {'model_score': None}, {'model_score': float('nan')},
                        {'model_score': True}, {'flagged': True}, {'cutoff': 2}):
            with self.subTest(changes=changes):
                self.assertFalse(classifier_check(self.score | changes)['passed'])

    def test_requests_uncertainty_and_unreviewed_candidates_need_review(self):
        for context in ('request', 'uncertain', 'candidate'):
            finding = dict(source='risk_lexicon', context=context, reviewed_by='llm', quote='Pay a deposit.')
            check = lexicon_check(self.lexicon | {'evidence': [finding]})
            self.assertFalse(check['passed'])
            self.assertIn('Pay a deposit.', check['review_reasons'][0])

    def test_confirmed_warnings_negations_and_background_do_not_fail(self):
        for context in ('warning', 'negated'):
            findings = [dict(source='risk_lexicon', context=context, reviewed_by='llm', quote='Never pay a fee.'),
                        dict(source='dataset_lexicon', context='background', kind='dataset_association'),
                        dict(source='posting', context='background', kind='missing_information')]
            self.assertTrue(lexicon_check(self.lexicon | {'evidence': findings})['passed'])

    def test_context_failure_or_missing_tool_cannot_pass(self):
        for status in ('failed', 'not_run'):
            self.assertEqual(recommendation_policy(self.trace(lexicon=self.lexicon | {'context_status': status}))['recommendation'], 'review_needed')
        for trace in ([], self.trace()[:1], self.trace()[1:]):
            self.assertEqual(recommendation_policy(trace)['recommendation'], 'review_needed')

    def test_claimed_pass_fields_cannot_override_actual_evidence(self):
        score = self.score | {'model_score': .9, 'flagged': True, 'passed': True}
        self.assertEqual(recommendation_policy(self.trace(score))['recommendation'], 'review_needed')
        with self.assertRaisesRegex(ValueError, 'both tool checks'):
            validate_assessment(self.assessment('pass_no_review_needed'), self.posting,
                                self.trace(lexicon=self.lexicon | {'context_status': 'failed'}))

    def test_old_actions_and_confirmation_fields_are_not_supported(self):
        for changes in ({'recommendation': 'propose_rejection'}, {'requires_confirmation': True}):
            with self.assertRaises(ValueError):
                Assessment.model_validate(self.assessment('pass_no_review_needed').model_dump() | changes)
