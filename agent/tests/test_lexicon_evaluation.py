import unittest

from agent.config import Settings
from agent.context_review import ContextReport
from agent.evaluate_lexicon import DEFAULT_CASES, evaluate, load_cases, metrics
from agent.risk_rules import TEXT_FIELDS


class LexiconEvaluationTests(unittest.TestCase):
    def test_every_method_gets_same_cases_and_repeats_without_receiving_gold(self):
        cases = load_cases(DEFAULT_CASES)[:2]
        seen = []

        def reviewer(posting, candidates, settings):
            seen.append((posting.posting_id, bool(candidates)))
            # Mock context interpretation for these two fixtures only.
            risky = posting.posting_id == 'dev02'
            return ContextReport(posting_id=posting.posting_id,
                checked_fields=[f for f in TEXT_FIELDS if getattr(posting, f)],
                candidate_reviews=[dict(evidence_id=c['evidence_id'], interpretation='request',
                    reason='Mock interpretation.') for c in candidates],
                additional_findings=[] if candidates or not risky else [dict(field='description',
                    quote=posting.description, category='payment_request', interpretation='request',
                    reason='Mock interpretation.')])

        result = evaluate(cases, Settings(backend='ollama_local'), repeats=2, reviewer=reviewer)
        self.assertEqual(len(result['rows']), 12)
        for method in ('rules', 'llm', 'combined'):
            rows = [r for r in result['rows'] if r['method'] == method]
            self.assertEqual({(r['case_id'], r['repeat']) for r in rows},
                             {(c['case_id'], n) for c in cases for n in range(2)})
            self.assertEqual(result['metrics'][method]['false_alarms'], 0)
        self.assertEqual(len(seen), 8)
        self.assertTrue(all(c['independent_cases'] == 2 for c in result['paired_comparisons']))

    def test_failed_llm_reviews_are_abstentions_not_correct_negative_predictions(self):
        def fail(*args):
            raise RuntimeError('PRIVATE_PROVIDER_TEXT')
        result = evaluate(load_cases(DEFAULT_CASES)[:2], Settings(backend='ollama_local'), reviewer=fail)
        for method in ('llm', 'combined'):
            self.assertEqual(result['metrics'][method]['failed_reviews'], 2)
            self.assertEqual(result['metrics'][method]['accuracy_counting_failures_as_wrong'], 0)
            self.assertEqual(result['metrics'][method]['requests_missed_or_unavailable'], 1)
        self.assertNotIn('PRIVATE_PROVIDER_TEXT', str(result))

    def test_unsupported_exact_quotes_count_against_evidence_quality(self):
        row = dict(expected_request=False, predicted_request=True, status='completed',
                   gold_requests=[], findings=[dict(field='title', quote='Data Entry', category='payment_request')])
        result = metrics([row])
        self.assertEqual(result['false_alarms'], 1)
        self.assertEqual(result['unsupported_findings'], 1)

    def test_rules_comparison_runs_without_model_and_covers_full_fixture_set(self):
        result = evaluate(load_cases(DEFAULT_CASES), Settings(backend='dummy'), mode='rules')
        self.assertEqual(result['independent_cases'], 12)
        self.assertEqual(result['metrics']['rules']['failed_reviews'], 0)
