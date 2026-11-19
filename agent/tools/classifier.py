"""Real LightGBM inference in the modelling environment."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
from agent.core.schemas import Posting
from agent.core.policy import classifier_check

ROOT = Path(__file__).resolve().parents[2]


def score_posting(posting, settings):
    if not settings.classifier_python.is_file() or not (settings.classifier_artifact/'pipeline.joblib').is_file():
        raise FileNotFoundError('Fit the classifier artifact first: python -m src.classifier_inference --fit')
    result = subprocess.run([str(settings.classifier_python), '-m', 'src.classifier_inference',
        '--artifact', str(settings.classifier_artifact)], input=posting.model_dump_json(),
        text=True, encoding='utf-8', capture_output=True, cwd=ROOT, timeout=90,
        env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
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
