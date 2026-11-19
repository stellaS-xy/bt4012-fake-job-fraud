import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import httpx
from langchain_openai import ChatOpenAI

from agent.tests.runtime_fixtures import offline_runtime
from agent.core.config import Settings
from agent.core.runner import run_review
from agent.core.schemas import Posting
from evaluator.checks import code_checks
from evaluator.config import load_judge_settings
from evaluator.data import load_examples
from evaluator.judge import judge_case, judge_payload, validate_judgment, answer_passages
from evaluator.runner import run_evaluation
from evaluator.schemas import DIMENSIONS, Example, Judgment
from evaluator.report import classification_metrics


class EvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.posting = Posting(posting_id='eval_case', title='Office Assistant',
                              description='You must pay an application fee before starting.')
        with offline_runtime():
            cls.case = run_review(cls.posting, 'ordinary', Settings(backend='ollama_local'))

    def verdict(self, **changes):
        values = dict(posting_id=self.posting.posting_id, recommendation_acceptable=True,
                      unsupported_claims=[], uncertainties=['Automated judgment only.'])
        values.update({d: dict(score=4, reason='The review recommendation follows the available evidence.',
                              evidence_ids=[], answer_quotes=[answer_passages(self.case)[0]]) for d in DIMENSIONS})
        return Judgment(**(values | changes))

    def test_code_checks_validate_a_genuine_saved_workflow(self):
        result = code_checks(self.posting, self.case)
        self.assertTrue(result['all_passed'], result)

    def test_fabricated_quote_unknown_citation_and_incorrect_pass_fail_checks(self):
        for change in ('quote', 'citation', 'incorrect_pass'):
            with self.subTest(change=change):
                case = self.case.model_copy(deep=True)
                if change == 'quote':
                    case.evidence[0].quote = 'Invented text that is absent.'
                elif change == 'citation':
                    case.assessment.evidence_ids = ['invented']
                else:
                    case.assessment.recommendation = 'pass_no_review_needed'
                self.assertFalse(code_checks(self.posting, case)['all_passed'])

    def test_csv_input_and_agent_and_judge_never_receive_annotations(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'development.csv'
            with path.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=['job_id', 'title', 'description', 'fraudulent', 'expected_action'])
                writer.writeheader()
                writer.writerow(dict(job_id='eval_case', title=self.posting.title,
                    description=self.posting.description, fraudulent=1, expected_action='review_needed'))
            examples = load_examples(path)
        agent = Mock(return_value=self.case)
        judge = Mock(return_value=self.verdict())
        result = run_evaluation(examples, Settings(backend='ollama_local'), Settings(backend='ollama_local'),
                                agent_fn=agent, judge_fn=judge)
        self.assertEqual(agent.call_args.args[0].model_dump(), self.posting.model_dump())
        payload = json.dumps(judge_payload(judge.call_args.args[0], judge.call_args.args[1]))
        for key in ('fraudulent', 'expected_action', 'scenario', 'model_name'):
            self.assertNotIn('"'+key+'":', payload)
        self.assertEqual(result['summary']['expected_action_matches'], 1)
        self.assertEqual(result['summary']['classifier_metrics']['status'], 'unavailable')
        self.assertEqual(result['summary']['recommendations_by_fraud_label']['1']['review_needed'], 1)

    def test_input_validation_does_not_silently_skip_duplicate_or_bad_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'data.jsonl'
            record = dict(case_id='one', posting=self.posting.model_dump(), fraudulent=2)
            path.write_text(json.dumps(record)+'\n')
            with self.assertRaises(ValueError):
                load_examples(path)
            record['fraudulent'] = 1
            path.write_text((json.dumps(record)+'\n')*2)
            with self.assertRaisesRegex(ValueError, 'unique'):
                load_examples(path)

    def test_frontend_export_can_be_replayed_without_agent_calls(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'case.json'
            path.write_text(json.dumps({'case': self.case.model_dump(mode='json'), 'reviewer_decision': None}))
            examples = load_examples(path, replay=True)
        no_agent = Mock(side_effect=AssertionError('Do not rerun the agent'))
        result = run_evaluation(examples, Settings(backend='ollama_local'), code_only=True, agent_fn=no_agent)
        no_agent.assert_not_called()
        self.assertEqual(result['summary']['judge_skipped'], 1)
        self.assertIsNone(result['summary']['mean_judge_scores']['evidence_grounding'])

    def test_missing_posting_text_is_counted_without_model_calls(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'data.jsonl'
            path.write_text(json.dumps(dict(case_id='empty', posting={'title': 'Role', 'description': ''}, fraudulent=1)))
            examples = load_examples(path)
        agent = Mock(side_effect=AssertionError('Invalid input must not reach the model'))
        result = run_evaluation(examples, Settings(backend='ollama_local'), code_only=True, agent_fn=agent)
        agent.assert_not_called()
        self.assertEqual(result['summary']['input_failures'], 1)
        self.assertEqual(result['summary']['evaluations'], 1)

    def test_classifier_metrics_use_predictions_not_recommendations(self):
        rows = [dict(fraudulent=y, classifier_prediction=p, recommendation='review_needed')
                for y, p in [(1, True), (0, True), (1, False), (0, False)]]
        rows.append(dict(fraudulent=1, classifier_prediction=None, recommendation='pass_no_review_needed'))
        result = classification_metrics(rows)
        self.assertEqual(result['evaluated'], 4)
        self.assertEqual(result['precision'], .5)
        self.assertEqual(result['recall'], .5)
        self.assertEqual(result['false_positive_rate'], .5)

    def test_valid_real_classifier_result_is_counted_even_if_the_llm_falls_back(self):
        case = self.case.model_copy(deep=True)
        case.status = 'fallback'
        score = next(t for t in case.tool_trace if t.name == 'score_posting')
        score.output['is_test_fixture'] = False
        example = Example(case_id='independent-classifier', posting=self.posting, fraudulent=0, case=case)
        result = run_evaluation([example], Settings(backend='ollama_local'), code_only=True)
        self.assertFalse(result['rows'][0]['code']['all_passed'])
        self.assertIs(result['rows'][0]['classifier_prediction'], False)
        self.assertEqual(result['summary']['classifier_metrics']['true_negative'], 1)

    def test_judge_failure_is_not_a_pass_or_a_zero_score(self):
        example = Example(case_id='one', posting=self.posting, fraudulent=1, case=self.case)
        judge = Mock(side_effect=RuntimeError('PRIVATE_PROVIDER_TEXT'))
        result = run_evaluation([example], Settings(backend='ollama_local'), Settings(backend='ollama_local'), judge_fn=judge)
        self.assertEqual(result['summary']['judge_failed'], 1)
        self.assertEqual(result['summary']['judge_completed'], 0)
        self.assertIsNone(result['summary']['mean_judge_scores']['evidence_grounding'])
        self.assertNotIn('PRIVATE_PROVIDER_TEXT', json.dumps(result))

    def test_bad_judge_ids_and_invented_criticism_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_judgment(self.verdict(posting_id='another'), self.posting, self.case)
        bad = self.verdict().model_copy(deep=True)
        bad.evidence_grounding.evidence_ids = ['invented']
        with self.assertRaisesRegex(ValueError, 'unknown'):
            validate_judgment(bad, self.posting, self.case)
        bad = self.verdict(unsupported_claims=[dict(quote='Agent never said this.', reason='Unsupported claim.')])
        with self.assertRaisesRegex(ValueError, 'quote'):
            validate_judgment(bad, self.posting, self.case)

    def test_tool_observation_cannot_be_quoted_as_the_agent_answer(self):
        bad = self.verdict().model_copy(deep=True)
        bad.explanation_clarity.answer_quotes = [self.case.evidence[0].observation]
        with self.assertRaisesRegex(ValueError, 'actual agent answer'):
            validate_judgment(bad, self.posting, self.case)

    def test_unsupported_limitation_claim_is_not_ignored(self):
        case = self.case.model_copy(deep=True)
        case.assessment.limitations.append('Such fees are sometimes legitimate if disclosed upfront.')
        value = self.verdict(unsupported_claims=[dict(quote=case.assessment.limitations[-1], reason='Not established by this posting.')])
        value.evidence_grounding.score = 2
        validate_judgment(value, self.posting, case)
        value.evidence_grounding.score = 4
        with self.assertRaisesRegex(ValueError, 'perfect grounding'):
            validate_judgment(value, self.posting, case)

    def test_fallback_can_be_safe_but_cannot_get_completed_answer_quality_scores(self):
        case = self.case.model_copy(deep=True)
        case.status = 'fallback'
        case.error_code = 'assessment_invalid'
        case.assessment.rationale = 'The review did not complete its checks. Escalate for manual review.'
        judgment = self.verdict()
        for dimension in DIMENSIONS:
            getattr(judgment, dimension).answer_quotes = [case.assessment.rationale]
        result = run_evaluation([Example(case_id='fallback', posting=self.posting, case=case)],
            Settings(backend='ollama_local'), Settings(backend='ollama_local'), judge_fn=lambda *args: judgment)
        row = result['rows'][0]
        self.assertTrue(row['code']['safety_checks_passed'])
        self.assertFalse(row['code']['all_passed'])
        self.assertEqual(row['judgment']['context_understanding']['score'], 4)
        self.assertIsNone(row['scoring']['effective_scores']['context_understanding'])
        self.assertEqual(result['summary']['safe_fallbacks'], 1)
        self.assertEqual(result['summary']['quality_scored_cases'], 0)
        self.assertIsNone(result['summary']['mean_judge_scores']['explanation_clarity'])

    def test_judge_validation_retry_is_bounded(self):
        model = Mock()
        structured = model.with_structured_output.return_value
        structured.invoke.side_effect = [self.verdict(posting_id='wrong'), self.verdict()]
        result = judge_case(self.posting, self.case, Settings(backend='ollama_local'), model)
        self.assertEqual(result.posting_id, self.posting.posting_id)
        self.assertEqual(structured.invoke.call_count, 2)
        structured.invoke.reset_mock()
        structured.invoke.side_effect = [self.verdict(posting_id='wrong')]*2
        with self.assertRaises(ValueError):
            judge_case(self.posting, self.case, Settings(backend='ollama_local'), model)
        self.assertEqual(structured.invoke.call_count, 2)

    def test_local_judge_defaults_do_not_reuse_a_cloud_key(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'evaluator').mkdir()
            (root/'.env').write_text('OLLAMA_API_KEY=PRIVATE_KEY\n')
            with patch('evaluator.config.EVALUATOR_DIR', root/'evaluator'), patch.dict('os.environ', {}, clear=True):
                settings = load_judge_settings()
                self.assertEqual(settings.backend, 'ollama_local')
                self.assertEqual(settings.api_key, 'ollama')
                self.assertEqual(settings.request_timeout, 180)
                self.assertNotIn('PRIVATE_KEY', repr(settings))
                for value in ('0', '-1', 'nan', 'inf'):
                    with patch.dict('os.environ', {'JUDGE_TIMEOUT_SECONDS': value}):
                        with self.assertRaises(ValueError):
                            load_judge_settings()

    def test_actual_sdk_judge_protocol_with_mocked_http(self):
        for backend in ('ollama_local', 'ollama'):
            with self.subTest(backend=backend):
                def respond(request):
                    body = json.loads(request.content)
                    self.assertNotIn('"fraudulent":', json.dumps(body))
                    message = dict(role='assistant', content=self.verdict().model_dump_json())
                    if backend == 'ollama':
                        self.assertNotIn('response_format', body)
                        message = dict(role='assistant', content=None, tool_calls=[dict(id='judge_call', type='function',
                            function=dict(name='Judgment', arguments=self.verdict().model_dump_json()))])
                    else:
                        self.assertEqual(body['response_format']['json_schema']['name'], 'Judgment')
                    return httpx.Response(200, json=dict(id='test', object='chat.completion', created=0,
                        model='test', choices=[dict(index=0, message=message,
                        finish_reason='tool_calls' if backend == 'ollama' else 'stop')]))
                with httpx.Client(transport=httpx.MockTransport(respond)) as client:
                    model = ChatOpenAI(model='test', api_key='TEST', base_url='http://localhost:11434/v1',
                                       http_client=client, max_retries=0, use_responses_api=False)
                    result = judge_case(self.posting, self.case, Settings(backend=backend, api_key='TEST'), model)
                    self.assertEqual(result.posting_id, self.posting.posting_id)

    def test_html_report_escapes_posting_text(self):
        from evaluator.export_html import render
        case = self.case.model_copy(deep=True)
        case.posting.title = '<script>alert(1)</script>'
        report = dict(summary={}, rows=[dict(case_id='example', fraudulent=0,
            case=case.model_dump(mode='json'), judge_status='skipped', judgment=None)])
        html = render(report)
        self.assertNotIn('<script>alert(1)</script>', html)
        self.assertIn('&lt;script&gt;', html)


if __name__ == '__main__':
    unittest.main()
