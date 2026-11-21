"""Offline integration checks: no TensorFlow import, real predictions or LLM calls."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from agent.core.config import Settings
from agent.core.schemas import Posting
from evaluator import balanced
from src import ensemble_inference as inference
from src.final_ensemble_artifact import CORE, digest

# Load this adapter directly to avoid importing the LangChain tool registry.
spec = importlib.util.spec_from_file_location('ensemble_adapter_under_test', inference.ROOT/'agent/tools/classifier.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def make_artifact(folder):
    artifact = Path(folder)
    for name in CORE:
        (artifact/name).write_bytes(b'offline fixture, never deserialised')
    metadata = dict(model='calibrated_mlp_extra_trees', weights=[.5, .5],
        threshold=.20923746948802296, training_sha256='train-reference',
        runtime_versions={'tensorflow': 'test-version'},
        features_source_sha256=digest(inference.ROOT/'src/features.py'),
        file_hashes={name: digest(artifact/name) for name in CORE})
    (artifact/'metadata.json').write_text(json.dumps(metadata))
    (artifact/'runtime_requirements.txt').write_text('tensorflow==test-version\n')
    (artifact/'export_verified.json').write_text(json.dumps({'test_used': False, 'development_probe_rows': 32}))
    return metadata


class EnsembleInferenceTests(unittest.TestCase):
    def posting(self):
        return Posting(title='Role', description='Routine duties.',
                       telecommuting=0, has_company_logo=1, has_questions=0)

    def test_score_preserves_threshold_input_identity_and_missing_fields(self):
        posting = self.posting()
        with tempfile.TemporaryDirectory() as folder:
            metadata = make_artifact(folder)
            with patch.object(inference.importlib.metadata, 'version', return_value='test-version'), \
                 patch.object(inference, 'predict', return_value=[.4]) as predict:
                result = inference.score_payload({**posting.model_dump(), 'fraudulent': 1}, folder)
            frame = predict.call_args.args[1]
            self.assertNotIn('fraudulent', frame)
            self.assertNotIn('posting_id', frame)
            self.assertEqual(frame.telecommuting.iloc[0], 0)
            self.assertEqual(frame.has_company_logo.iloc[0], 1)
            expected = hashlib.sha256(json.dumps(posting.model_dump(exclude={'posting_id'}),
                sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            self.assertEqual(result['input_sha256'], expected)
            self.assertEqual(result['cutoff'], metadata['threshold'])
            self.assertTrue(result['flagged'])
            self.assertFalse(result['is_test_fixture'])
            self.assertIn('industry', result['missing_input_fields'])
            self.assertTrue(result['calibrated_probability'])

    def test_unknown_binaries_are_never_imputed_as_zero(self):
        for name in ('telecommuting', 'has_company_logo', 'has_questions'):
            posting = self.posting().model_dump()
            posting[name] = None
            with self.subTest(name=name), patch.object(inference, 'predict') as predict:
                with self.assertRaisesRegex(ValueError, name):
                    inference.score_payload(posting, '/nonexistent')
                predict.assert_not_called()

    def test_tampering_and_runtime_mismatch_fail_before_prediction(self):
        with tempfile.TemporaryDirectory() as folder:
            make_artifact(folder)
            with patch.object(inference.importlib.metadata, 'version', return_value='wrong'):
                with self.assertRaisesRegex(ValueError, 'version differs'):
                    inference.check_runtime(folder)
            (Path(folder)/'mlp.keras').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksums differ'):
                inference.check_runtime(folder)

    def test_invalid_scores_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            make_artifact(folder)
            for score in (float('nan'), float('inf'), -.1, 1.1):
                with self.subTest(score=score), \
                     patch.object(inference.importlib.metadata, 'version', return_value='test-version'), \
                     patch.object(inference, 'predict', return_value=[score]):
                    with self.assertRaisesRegex(ValueError, 'invalid score'):
                        inference.score_payload(self.posting().model_dump(), folder)

    def test_agent_dispatches_to_ensemble_and_rejects_wrong_input_identity(self):
        posting = self.posting()
        with tempfile.TemporaryDirectory() as folder:
            make_artifact(folder)
            python = Path(folder)/'python'; python.touch()
            settings = Settings(classifier_python=python, classifier_artifact=Path(folder))
            with patch.object(inference.importlib.metadata, 'version', return_value='test-version'), \
                 patch.object(inference, 'predict', return_value=[.1]):
                output = inference.score_payload(posting.model_dump(), folder)
            with patch.object(adapter.subprocess, 'run', return_value=Mock(returncode=0, stdout=json.dumps(output))) as run:
                result = adapter.score_posting(posting, settings)
                self.assertTrue(result['passed'])
                self.assertIn('src.ensemble_inference', run.call_args.args[0])
                self.assertEqual(run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '-1')
                output['input_sha256'] = 'wrong'
                run.return_value.stdout = json.dumps(output)
                with self.assertRaisesRegex(ValueError, 'input hash mismatch'):
                    adapter.score_posting(posting, settings)
                run.return_value.returncode = 1
                with self.assertRaises(RuntimeError):
                    adapter.score_posting(posting, settings)

    def test_evaluator_records_ensemble_checksums_and_training_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            metadata = make_artifact(folder)
            output = balanced.artifact_provenance(folder, {'split': 'heldout', 'training_sha256': 'train-reference'})
            self.assertEqual(output['file_hashes'], metadata['file_hashes'])
            self.assertEqual(output['threshold'], metadata['threshold'])
            with self.assertRaisesRegex(ValueError, 'training file differs'):
                balanced.artifact_provenance(folder, {'split': 'heldout', 'training_sha256': 'other-train'})

    def test_defaults_and_explicit_legacy_backend(self):
        self.assertEqual(Settings().classifier_backend, 'ensemble')
        self.assertEqual(Settings().classifier_python.parent.parent.name, '.venv-mlp')
        self.assertEqual(Settings(classifier_backend='lightgbm').classifier_artifact.name, 'agent_lightgbm')
