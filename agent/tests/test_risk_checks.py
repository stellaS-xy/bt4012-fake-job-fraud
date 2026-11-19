import unittest
from unittest.mock import Mock

from agent.tests.runtime_fixtures import offline_runtime
from agent.core.config import Settings
from agent.tools.context_review import ContextReport, review_context, validate_context, validate_interpretations
from agent.tools.risk_rules import TEXT_FIELDS, candidate_matches
from agent.core.schemas import Posting, Evidence
from agent.resources.fixtures import SCENARIOS
from agent.core.runner import run_review
from agent.core.validation import validate_assessment
from agent.tools.lexicon import match_lexicon
from agent.core.validation import is_risky_request


class RiskChecksTests(unittest.TestCase):
    def test_request_label_cannot_contradict_its_own_reason(self):
        report = self.report()
        report.candidate_reviews[0].reason = 'This is not a direct payment request from the employer.'
        with self.assertRaisesRegex(ValueError, 'contradicts'):
            validate_interpretations(report, self.posting, self.candidates)

    def setUp(self):
        runtime = offline_runtime()
        runtime.__enter__()
        self.addCleanup(runtime.__exit__, None, None, None)
        self.posting = Posting(title='Assistant', description='You must pay an application fee.')
        self.candidates = candidate_matches(self.posting)
        self.settings = Settings(backend='ollama_local')

    def report(self, interpretation='request', **changes):
        values = dict(posting_id=self.posting.posting_id,
            checked_fields=['title', 'description'], additional_findings=[],
            candidate_reviews=[dict(evidence_id=c['evidence_id'], interpretation=interpretation,
                                    reason='Interpretation of the supplied sentence.') for c in self.candidates])
        return ContextReport(**(values | changes))

    def test_rules_keep_negation_and_warning_in_the_quote(self):
        for text in ['You must never pay an application fee.',
                     'Teach applicants that "you must pay an application fee" is a scam.']:
            candidates = candidate_matches(Posting(title='Trainer', description=text))
            self.assertTrue(candidates)
            self.assertEqual(candidates[0]['quote'], text)

    def test_rules_cover_new_wording_and_all_input_fields(self):
        texts = ['Transfer a refundable registration deposit to our recruiter.',
                 'Send the one-time login code from your banking app to our hiring manager.']
        for field in TEXT_FIELDS:
            for text in texts:
                p = Posting(**({'title': 'Role', 'description': 'Routine duties.'} | {field: text}))
                self.assertTrue(any(c['field'] == field for c in candidate_matches(p)))

    def test_background_terms_and_urgency_never_support_rejection(self):
        p = Posting(title='Data entry administrative assistant', description='Apply now in the oil and gas industry.')
        result = match_lexicon(p)
        self.assertTrue(result['evidence'])
        self.assertTrue(all(e['source'] in ('dataset_lexicon', 'posting') for e in result['evidence']))
        self.assertFalse(any(is_risky_request(Evidence.model_validate(e)) for e in result['evidence']))

    def test_every_candidate_must_be_reviewed_once(self):
        review = self.report().candidate_reviews[0]
        for reviews in ([], [review, review]):
            with self.assertRaisesRegex(ValueError, 'every candidate'):
                validate_context(self.report(candidate_reviews=reviews), self.posting, self.candidates)
        with self.assertRaisesRegex(ValueError, 'text field'):
            validate_context(self.report(checked_fields=['title']), self.posting, self.candidates)

    def test_wrong_posting_and_invented_or_wrong_field_quotes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'another posting'):
            validate_context(self.report(posting_id='wrong'), self.posting, self.candidates)
        for field, quote in [('description', 'Invented request'), ('title', self.posting.description), ('description', ' ')]:
            report = self.report(additional_findings=[dict(field=field, quote=quote,
                category='payment_request', interpretation='request', reason='Claim.')])
            with self.assertRaisesRegex(ValueError, 'exact passage'):
                validate_context(report, self.posting, self.candidates)

    def test_only_reviewed_request_can_support_rejection(self):
        for interpretation in ('request', 'negated', 'warning', 'uncertain'):
            result = match_lexicon(self.posting, self.settings,
                                   lambda *args: self.report(interpretation))
            self.assertEqual(result['context_status'], 'completed')
            risk = next(e for e in result['evidence'] if e['source'] == 'risk_lexicon')
            self.assertEqual(is_risky_request(Evidence.model_validate(risk)), interpretation == 'request')

    def test_failed_or_incomplete_context_keeps_unconfirmed_candidates(self):
        for reviewer in (Mock(side_effect=RuntimeError('PRIVATE_ERROR')),
                         Mock(return_value=self.report(candidate_reviews=[]))):
            result = match_lexicon(self.posting, self.settings, reviewer)
            self.assertEqual(result['context_status'], 'failed')
            self.assertFalse(any(is_risky_request(Evidence.model_validate(e)) for e in result['evidence']))
            self.assertNotIn('PRIVATE_ERROR', str(result))

    def test_llm_can_supply_a_new_exact_quote_missed_by_rules(self):
        p = Posting(title='Role', description='Before an interview, a refundable eighty-dollar remittance is required.')
        report = ContextReport(posting_id=p.posting_id, checked_fields=['title', 'description'],
            candidate_reviews=[], additional_findings=[dict(field='description', quote=p.description,
                category='payment_request', interpretation='request', reason='Payment required for interview.')])
        result = match_lexicon(p, self.settings, lambda *args: report)
        self.assertTrue(any(e['source'] == 'llm_context' and is_risky_request(Evidence.model_validate(e))
                            for e in result['evidence']))

    def test_context_retry_is_bounded_and_returns_only_a_valid_report(self):
        model = Mock()
        structured = model.with_structured_output.return_value
        structured.invoke.side_effect = [self.report(checked_fields=[]), self.report()]
        result = review_context(self.posting, self.candidates, self.settings, model)
        self.assertEqual(result, self.report())
        self.assertEqual(structured.invoke.call_count, 2)
        structured.invoke.reset_mock()
        structured.invoke.side_effect = [self.report(checked_fields=[])] * 2
        with self.assertRaises(ValueError):
            review_context(self.posting, self.candidates, self.settings, model)
        self.assertEqual(structured.invoke.call_count, 2)

    def test_final_decision_cannot_bypass_context_review_with_a_rule_candidate(self):
        case = run_review(Posting(**SCENARIOS['suspicious']['posting']), 'suspicious', Settings(backend='ollama_local'))
        self.assertEqual(case.status, 'completed')
        proposal = case.assessment.model_copy(update={'recommendation': 'pass_no_review_needed'})
        with self.assertRaisesRegex(ValueError, 'both tool checks'):
            validate_assessment(proposal, case.posting, case.tool_trace)

    def test_model_request_verdict_cannot_override_negation_or_warning(self):
        for text in ('You must never pay an application fee.',
                     'Teach applicants that "you must pay an application fee" is a scam.'):
            p = Posting(title='Trainer', description=text)
            candidates = candidate_matches(p)
            # Deliberately wrong LLM answer must stay unconfirmed by the app.
            report = ContextReport(posting_id=p.posting_id, checked_fields=['title', 'description'],
                candidate_reviews=[dict(evidence_id=c['evidence_id'], interpretation='request',
                                        reason='Incorrect model verdict.') for c in candidates], additional_findings=[])
            result = match_lexicon(p, self.settings, lambda *args: report)
            risk = next(e for e in result['evidence'] if e['source'] == 'risk_lexicon')
            self.assertEqual(risk['model_interpretation'], 'request')
            self.assertEqual(risk['context'], 'uncertain')
            self.assertFalse(is_risky_request(Evidence.model_validate(risk)))

    def test_short_llm_quote_cannot_hide_the_surrounding_warning(self):
        p = Posting(title='Trainer', description='Teach applicants that "you must pay an application fee" is a scam.')
        report = ContextReport(posting_id=p.posting_id, checked_fields=['title', 'description'],
            candidate_reviews=[], additional_findings=[dict(field='description',
                quote='you must pay an application fee', category='payment_request',
                interpretation='request', reason='Incorrect claim from an incomplete quote.')])
        from agent.tools.context_review import interpreted_findings
        found = interpreted_findings(report, p, [])
        self.assertEqual(found[0]['interpretation'], 'uncertain')
        self.assertEqual(found[0]['context_guard'], 'warning')
