"""Guard real-only application settings and required model provenance."""
import unittest
from pydantic import ValidationError
from agent.core.config import Settings
from agent.core.schemas import ReviewCase


class RealRuntimeConfigTests(unittest.TestCase):
    def test_default_is_real_local_model_and_lightgbm(self):
        settings = Settings()
        self.assertEqual(settings.backend, 'ollama_local')
        self.assertEqual(settings.model, 'qwen3:4b-instruct')
        self.assertEqual(settings.classifier_backend, 'lightgbm')

    def test_simulated_modes_are_rejected_instead_of_renamed(self):
        for backend in ('dummy', 'test_fixture', 'simulated'):
            with self.subTest(backend=backend), self.assertRaises(ValueError):
                Settings(backend=backend)
            with self.subTest(classifier_backend=backend), self.assertRaises(ValueError):
                Settings(classifier_backend=backend)

    def test_missing_model_provenance_cannot_produce_a_case(self):
        with self.assertRaises(ValidationError) as error:
            ReviewCase.model_validate({})
        missing = {e['loc'][0] for e in error.exception.errors() if e['type'] == 'missing'}
        self.assertTrue({'model_backend', 'model_name'}.issubset(missing))
