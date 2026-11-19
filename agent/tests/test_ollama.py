"""Exercise the real LangChain/ChatOpenAI/SDK protocol with HTTP mocked only."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from langchain_openai import ChatOpenAI

from agent.tests.runtime_fixtures import classifier_fixture
from agent.core.config import Settings, load_settings
from agent.resources.fixtures import SCENARIOS
from agent.core.middleware import REQUIRED_TOOLS
from agent.core.models import create_review_model
from agent.core.runner import run_review
from agent.core.schemas import Assessment, Posting
from agent.tools.context_review import ContextReport, review_context
from agent.tools.risk_rules import TEXT_FIELDS


def completion(calls):
    return {"id": "test-response", "object": "chat.completion", "created": 0,
            "model": "deepseek-v4.1-flash", "choices": [{"index": 0,
                "message": {"role": "assistant", "content": None, "tool_calls": calls},
                "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}}


def tool_call(name, args):
    return {"id": f"provider_{name}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


class OllamaTests(unittest.TestCase):
    def test_local_model_uses_loopback_and_never_sends_cloud_key(self):
        settings = Settings(backend="ollama_local", api_key="PRIVATE_CLOUD_KEY")
        self.assertEqual(settings.model, "qwen3:4b-instruct")
        with patch("langchain_openai.ChatOpenAI") as client:
            create_review_model(settings)
        self.assertEqual(client.call_args.kwargs["base_url"], "http://localhost:11434/v1")
        self.assertEqual(client.call_args.kwargs["api_key"], "ollama")
        self.assertNotIn("reasoning_effort", client.call_args.kwargs)
        with self.assertRaises(ValueError):
            Settings(backend="ollama_local", model="gpt-oss:20b-cloud")

    def setUp(self):
        self.posting = Posting(**SCENARIOS["suspicious"]["posting"])
        self.settings = Settings(backend="ollama", api_key="TEST_PROVIDER_KEY")

    def client(self, handler):
        client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(client.close)
        return ChatOpenAI(model=self.settings.model, base_url=self.settings.base_url,
                          api_key=self.settings.api_key, use_responses_api=False,
                          max_retries=0, http_client=client, reasoning_effort="none")

    def test_real_model_factory_preserves_user_endpoint_and_exact_model(self):
        with patch("langchain_openai.ChatOpenAI") as client:
            create_review_model(self.settings)
        args = client.call_args.kwargs
        self.assertEqual(args["base_url"], "https://ollama.com/v1")
        self.assertEqual(args["model"], "deepseek-v4.1-flash")
        self.assertFalse(args["use_responses_api"])
        self.assertNotIn("TEST_PROVIDER_KEY", repr(self.settings))

    def test_ollama_protocol_calls_tools_and_formats_a_valid_assessment(self):
        requests = []

        def respond(request):
            body = json.loads(request.content)
            requests.append(body)
            self.assertEqual(str(request.url), self.settings.base_url + "/chat/completions")
            self.assertEqual(request.headers['authorization'], 'Bearer ' + self.settings.api_key)
            self.assertEqual(body['model'], self.settings.model)
            local = self.settings.backend == 'ollama_local'
            if not local or len(requests) == 1:
                self.assertNotIn('response_format', body)  # Unsupported by Ollama cloud.
            names = [t['function']['name'] for t in body.get('tools', [])]
            if len(requests) == 1:
                self.assertEqual(set(names), set(REQUIRED_TOOLS))
                return httpx.Response(200, json=completion([
                    tool_call(name, {'posting_id': self.posting.posting_id}) for name in names]))
            self.assertEqual(names, [] if local else ['Assessment'])
            if getattr(self, 'omit_first_assessment', False) and len(requests) == 2:
                response = completion([])
                response['choices'][0]['message'] = {'role': 'assistant', 'content': 'I reviewed the tools.'}
                response['choices'][0]['finish_reason'] = 'stop'
                return httpx.Response(200, json=response)
            messages = [m for m in body['messages'] if m['role'] == 'tool']
            if not messages and getattr(self, 'omit_first_assessment', False):
                snapshot = next(json.loads(m['content']) for m in body['messages']
                                if m['role'] == 'user' and m['content'].startswith('{'))
                outputs = list(snapshot['tool_results'].values())
            else:
                self.assertGreaterEqual(len(messages), len(REQUIRED_TOOLS))
                outputs = [json.loads(m['content']) for m in messages[:len(REQUIRED_TOOLS)]]
            score = next(o for o in outputs if 'model_score' in o)
            evidence = [e for o in outputs for e in o.get('evidence', [])]
            assessment = Assessment(posting_id=self.posting.posting_id,
                model_score=score['model_score'], cutoff=score['cutoff'], flagged=score['flagged'],
                uses_test_fixtures=True, recommendation='review_needed', rationale='Provider-generated review based on the quoted payment request; business-tool results remain placeholders.',
                evidence_ids=[e['evidence_id'] for e in evidence],
                limitations=['Market salary and employer legitimacy were not checked.'],
                next_steps=['Resolve the failed checks before relying on the recommendation.'])
            if getattr(self, 'miscopy_first_score', False) and len(requests) == 2:
                assessment.model_score = 0.11
            if local:
                self.assertEqual(body['response_format']['type'], 'json_schema')
                response = completion([])
                response['choices'][0]['message'] = {'role': 'assistant', 'content': assessment.model_dump_json()}
                response['choices'][0]['finish_reason'] = 'stop'
                return httpx.Response(200, json=response)
            return httpx.Response(200, json=completion([tool_call('Assessment', assessment.model_dump())]))

        def context(posting, candidates, settings):
            return ContextReport(posting_id=posting.posting_id,
                checked_fields=[f for f in TEXT_FIELDS if getattr(posting, f)],
                candidate_reviews=[dict(evidence_id=c['evidence_id'], interpretation='request',
                                        reason='Mocked interpretation of explicit payment request.') for c in candidates],
                additional_findings=[])
        with patch('agent.tools.score_real', side_effect=lambda p,s: classifier_fixture(p, 'suspicious')), \
             patch('agent.core.runner.create_review_model', return_value=self.client(respond)), \
             patch('agent.tools.lexicon.review_context', side_effect=context):
            case = run_review(self.posting, 'suspicious', self.settings)
        self.assertEqual(case.status, 'completed')
        self.assertEqual(case.model_backend, self.settings.backend)
        self.assertEqual(case.model_name, self.settings.model)
        self.assertTrue(case.assessment.rationale.startswith('Provider-generated'))
        repaired = getattr(self, 'miscopy_first_score', False) or getattr(self, 'omit_first_assessment', False)
        self.assertEqual(len(requests), 3 if repaired else 2)
        self.assertEqual(case.recovery_attempted, bool(repaired))
        self.assertEqual(case.assessment.model_score, 0.92)
        self.assertEqual(len(case.llm_tool_calls), len(REQUIRED_TOOLS))
        self.assertTrue(all(t.call_id.startswith('provider_') for t in case.tool_trace))
        self.assertNotIn('TEST_PROVIDER_KEY', case.model_dump_json())

    def test_context_extraction_uses_real_sdk_with_mocked_http(self):
        for backend in ('ollama_local', 'ollama'):
            with self.subTest(backend=backend):
                self.settings = Settings(backend=backend, api_key='TEST_PROVIDER_KEY')
                def respond(request):
                    body = json.loads(request.content)
                    report = ContextReport(posting_id=self.posting.posting_id,
                        checked_fields=[f for f in TEXT_FIELDS if getattr(self.posting, f)],
                        candidate_reviews=[], additional_findings=[])
                    if backend == 'ollama':
                        self.assertNotIn('response_format', body)
                        return httpx.Response(200, json=completion([tool_call('ContextReport', report.model_dump())]))
                    self.assertEqual(body['response_format']['json_schema']['name'], 'ContextReport')
                    response = completion([])
                    response['choices'][0]['message'] = {'role': 'assistant', 'content': report.model_dump_json()}
                    response['choices'][0]['finish_reason'] = 'stop'
                    return httpx.Response(200, json=response)
                result = review_context(self.posting, [], self.settings, model=self.client(respond))
                self.assertEqual(result.posting_id, self.posting.posting_id)

    def test_model_gets_one_chance_to_correct_a_miscopied_score(self):
        self.miscopy_first_score = True
        self.test_ollama_protocol_calls_tools_and_formats_a_valid_assessment()

    def test_local_protocol_calls_tools_then_uses_json_schema(self):
        self.settings = Settings(backend='ollama_local')
        self.test_ollama_protocol_calls_tools_and_formats_a_valid_assessment()

    def test_missing_structured_assessment_is_recovered_without_rerunning_tools(self):
        self.settings = Settings(backend='ollama_local')
        self.omit_first_assessment = True
        self.test_ollama_protocol_calls_tools_and_formats_a_valid_assessment()

    def test_authentication_failure_is_visible_without_using_a_test_model(self):
        def unauthorized(request):
            return httpx.Response(401, json={'error': {'message': 'SECRET_FROM_PROVIDER', 'type': 'authentication_error'}})
        with patch('agent.core.runner.create_review_model', return_value=self.client(unauthorized)), \
             patch('agent.tests.chat_model_fixture.ScriptedChatModel', side_effect=AssertionError('No silent simulation')):
            case = run_review(self.posting, 'suspicious', self.settings)
        self.assertEqual(case.status, 'fallback')
        self.assertEqual(case.error_code, 'authentication_failed')
        self.assertEqual(case.model_backend, 'ollama')
        self.assertIsNone(case.assessment.model_score)
        self.assertEqual(case.llm_tool_calls, [])
        self.assertNotIn('SECRET_FROM_PROVIDER', case.model_dump_json())

    def test_configuration_merges_root_and_agent_env_without_mutating_environment(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'agent').mkdir()
            (root/'.env').write_text('OLLAMA_API_KEY=ROOT_TEST_KEY\n')
            (root/'agent'/'.env').write_text('AGENT_BACKEND=ollama\nOLLAMA_MODEL=deepseek-v4.1-flash\n')
            with patch('agent.core.config.AGENT_DIR', root/'agent'), patch.dict('os.environ', {}, clear=True):
                settings = load_settings()
                self.assertEqual(settings.backend, 'ollama')
                self.assertEqual(settings.api_key, 'ROOT_TEST_KEY')
                import os
                self.assertNotIn('OLLAMA_API_KEY', os.environ)
                with patch.dict('os.environ', {'OLLAMA_API_KEY': 'PROCESS_TEST_KEY'}):
                    self.assertEqual(load_settings().api_key, 'PROCESS_TEST_KEY')

    def test_credits_required_is_reported_without_exposing_provider_text(self):
        def payment_required(request):
            return httpx.Response(402, json={'error': {'message': 'PRIVATE_PROVIDER_TEXT'}})
        with patch('agent.core.runner.create_review_model', return_value=self.client(payment_required)):
            case = run_review(self.posting, 'suspicious', self.settings)
        self.assertEqual(case.status, 'fallback')
        self.assertEqual(case.error_code, 'credits_required')
        self.assertIsNone(case.assessment.model_score)
        self.assertEqual(case.llm_tool_calls, [])
        self.assertNotIn('PRIVATE_PROVIDER_TEXT', case.model_dump_json())

    def test_missing_key_or_untrusted_endpoint_is_rejected_before_network(self):
        with self.assertRaisesRegex(ValueError, 'OLLAMA_API_KEY'):
            Settings(backend='ollama')
        with self.assertRaisesRegex(ValueError, 'not sent'):
            Settings(backend='ollama', api_key='TEST', base_url='https://example.com/v1')


if __name__ == '__main__':
    unittest.main()
