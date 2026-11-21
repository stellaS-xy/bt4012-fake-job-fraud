"""Real saved classifier inference in a separate modelling environment."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
from agent.core.schemas import Posting
from agent.core.policy import classifier_check

ROOT = Path(__file__).resolve().parents[2]


def score_posting(posting, settings):
    ensemble = settings.classifier_backend == 'ensemble'
    required = ('metadata.json', 'export_verified.json', 'runtime_requirements.txt',
                'mlp.keras', 'mlp_features.joblib', 'mlp_calibrator.joblib',
                'trees.joblib', 'trees_features.joblib', 'trees_calibrator.joblib') if ensemble else ('pipeline.joblib', 'metadata.json')
    if not settings.classifier_python.is_file() or any(
            not (settings.classifier_artifact/name).is_file() for name in required):
        raise FileNotFoundError('Classifier environment or model files missing. For ensemble v1, unpack the verified release ZIP and point CLASSIFIER_PYTHON to .venv-mlp/bin/python. No training is needed.')
    if ensemble and any(getattr(posting, name) is None for name in
                        ('telecommuting', 'has_company_logo', 'has_questions')):
        raise ValueError('Ensemble requires telecommuting, has_company_logo and has_questions as 0 or 1. Unknown is not zero; incomplete input requires review.')
    module = 'src.ensemble_inference' if ensemble else 'src.classifier_inference'
    result = subprocess.run([str(settings.classifier_python), '-m', module,
        '--artifact', str(settings.classifier_artifact)], input=posting.model_dump_json(),
        text=True, encoding='utf-8', capture_output=True, cwd=ROOT, timeout=90,
        env={**os.environ, 'PYTHONIOENCODING': 'utf-8',
             **({'CUDA_VISIBLE_DEVICES': '-1', 'TF_NUM_INTRAOP_THREADS': '2',
                 'TF_NUM_INTEROP_THREADS': '1', 'OMP_NUM_THREADS': '2'} if ensemble else {})})
    if result.returncode:
        raise RuntimeError('Real classifier inference failed; inspect the modelling environment and local artifact. No replacement score was substituted.')
    output = json.loads(result.stdout)
    if output.get('is_test_fixture') is not False or output.get('posting_id') != posting.posting_id:
        raise ValueError('Classifier returned the wrong posting or test-fixture status.')
    inputs = posting.model_dump(exclude={'posting_id'})
    expected_hash = hashlib.sha256(json.dumps(inputs, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()
    if output.get('input_sha256') != expected_hash:
        raise ValueError('Classifier input hash mismatch; the supplied posting was not preserved.')
    return {**output, **classifier_check(output)}
