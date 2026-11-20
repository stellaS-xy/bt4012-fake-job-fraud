from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import json
import hashlib

from agent.core.config import Settings
from agent.core.middleware import REQUIRED_TOOLS
from agent.core.schemas import Posting
from agent.tools import build_tools
from agent.tools.classifier import score_posting


class ClassifierToolTests(unittest.TestCase):
    def test_registry_exposes_only_the_two_planned_tools(self):
        posting = Posting(title='Role', description='Routine duties.')
        tools = build_tools(posting, 'ordinary', [], Settings(backend='ollama_local'))
        self.assertEqual({t.name for t in tools}, {'score_posting', 'match_lexicon'})
        self.assertEqual(set(REQUIRED_TOOLS), {t.name for t in tools})

    def test_real_classifier_does_not_fall_back_to_a_fixture(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = Settings(backend='ollama_local', classifier_artifact=Path(folder))
            with patch('agent.tools.score_real', side_effect=AssertionError('No substitution')):
                with self.assertRaises(FileNotFoundError):
                    score_posting(Posting(title='Role', description='Routine duties.'), settings)

    def test_real_tool_sends_posting_and_preserves_model_provenance(self):
        posting = Posting(title='Role', description='Routine duties.', has_company_logo=0)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'python.exe').touch()
            (root/'pipeline.joblib').touch()
            settings = Settings(backend='ollama_local', classifier_artifact=root, classifier_python=root/'python.exe')
            output = dict(posting_id=posting.posting_id, model_score=.7, cutoff=.3, flagged=True, is_test_fixture=False,
                          missing_input_fields=['industry'], model_version='model-test',
                          input_sha256=hashlib.sha256(json.dumps(posting.model_dump(exclude={'posting_id'}),
                              sort_keys=True,ensure_ascii=False).encode('utf-8')).hexdigest())
            with patch('agent.tools.classifier.subprocess.run', return_value=Mock(returncode=0, stdout=json.dumps(output))) as run:
                result = score_posting(posting, settings)
            sent = json.loads(run.call_args.kwargs['input'])
            self.assertEqual(sent['has_company_logo'], 0)
            self.assertNotIn('fraudulent', sent)
            self.assertEqual(run.call_args.kwargs['env']['PYTHONIOENCODING'], 'utf-8')
            self.assertEqual({k: result[k] for k in output}, output)
            self.assertFalse(result["passed"])
            self.assertTrue(result["review_reasons"])

    def test_real_classifier_is_selected_independently_of_the_llm_provider(self):
        self.assertEqual(Settings(backend='ollama_local').classifier_backend, 'lightgbm')
        self.assertEqual(Settings(backend='ollama', api_key='TEST').classifier_backend, 'lightgbm')
        self.assertEqual(Settings().classifier_backend, 'lightgbm')
