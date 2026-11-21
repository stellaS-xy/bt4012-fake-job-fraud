"""Inference-only adapter for the released calibrated MLP + Extra Trees v1.

Reads one posting from stdin, emits one JSON object. Never fits a model, reads
training/test datasets, selects a threshold, or downloads weights.
"""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys

from src.final_ensemble_artifact import digest, verify_artifact

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT/'models/final_ensemble_v1'


def check_runtime(artifact):
    metadata = verify_artifact(artifact)
    if metadata.get('features_source_sha256') != digest(ROOT/'src/features.py'):
        raise ValueError('Preprocessing differs from the model release; use its matching source code.')
    threshold = metadata.get('threshold')
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Invalid saved classification threshold.')
    for name, expected in metadata['runtime_versions'].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f'{name} version differs from the export. Install runtime_requirements.txt in the classifier environment.')
    return metadata


def predict(artifact, frame):
    # CPU inference leaves the job GPU available to Ollama. Set before importing TF.
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
    os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
    from threadpoolctl import threadpool_limits
    from src.d1_final_ensemble import predict_artifact
    with threadpool_limits(limits=2):
        return predict_artifact(artifact, frame)


def score_payload(posting, artifact=DEFAULT_ARTIFACT):
    import numpy as np
    import pandas as pd
    from src.features import TEXT_COLS, RAW_BINARY, RAW_CAT, GROUP
    fields = tuple(dict.fromkeys(TEXT_COLS + RAW_BINARY + RAW_CAT + ['location', 'department', 'salary_range']))
    # Raw binary columns were passthrough during training: do not invent imputation.
    for name in RAW_BINARY:
        if posting.get(name) not in (0, 1):
            raise ValueError(f'{name} must be supplied as 0 or 1; unknown input requires review.')
    artifact = Path(artifact)
    metadata = check_runtime(artifact)
    inputs = {name: posting.get(name) for name in fields}
    row = {name: value if value is not None else np.nan for name, value in inputs.items()}
    row[GROUP] = 'inference-only'
    frame = pd.DataFrame([row])
    for name in RAW_BINARY:
        frame[name] = frame[name].astype(float)
    # Entirely absent categorical fields must retain the fitted object dtype.
    frame[RAW_CAT] = frame[RAW_CAT].astype(object)
    # Keep stdout machine-readable even if a dependency prints a progress message.
    with contextlib.redirect_stdout(sys.stderr):
        score = float(predict(artifact, frame)[0])
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('Ensemble returned an invalid score; no replacement was substituted.')
    cutoff = metadata['threshold']
    return dict(posting_id=posting['posting_id'], model_score=score, cutoff=cutoff,
        flagged=score >= cutoff, is_test_fixture=False,
        input_sha256=hashlib.sha256(json.dumps(inputs, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest(),
        model_version='calibrated-mlp-extra-trees-v1', model_family='Calibrated MLP + Extra Trees',
        calibrated_probability=True, calibration_method='group-heldout sigmoid per component; 50/50 average',
        artifact_sha256=digest(artifact/'metadata.json'),
        missing_input_fields=[name for name, value in inputs.items() if value is None or value == ''],
        threshold_policy='Frozen maximum-F1 threshold from 3 grouped development-only OOF folds.',
        note='Saved v1 model and threshold; no refitting. Sigmoid calibration does not guarantee real-world probability accuracy. Missing optional text/categories remain missing; three binary inputs must be supplied. A below-cutoff score does not establish safety.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument('--check', action='store_true', help='Verify files, source and installed versions without predicting.')
    args = parser.parse_args()
    if args.check:
        check_runtime(args.artifact)
        print('Ensemble v1 files, source and runtime versions OK; no training or prediction.')
    else:
        print(json.dumps(score_payload(json.load(sys.stdin), args.artifact), allow_nan=False))


if __name__ == '__main__':
    main()
