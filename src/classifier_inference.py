"""Fit a training-only LightGBM integration artifact and score individual ads.

Uses the original modelling environment; no LLM dependencies are required.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import sys

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.features import TEXT_COLS, RAW_BINARY, RAW_CAT, GROUP, TARGET, build_inputs
from src.model_comparison import model_spec

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT/'models'/'agent_lightgbm'
INPUT_FIELDS = tuple(dict.fromkeys(TEXT_COLS + RAW_BINARY + RAW_CAT + ['location', 'department', 'salary_range']))


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_artifact(destination=DEFAULT_ARTIFACT):
    destination = Path(destination)
    train_path = ROOT/'data/processed/d1_train.csv'
    results = ROOT/'reports/model_comparison'
    plan = json.loads((results/'run_plan.json').read_text(encoding='utf-8'))
    if fingerprint(train_path) != plan['training_sha256']:
        raise ValueError('Training data differs from the saved comparison; do not reuse its selected settings/cutoffs.')
    if fingerprint(ROOT/'src/features.py') != plan['source_sha256']['features.py']:
        raise ValueError('Feature code differs from the saved comparison; do not silently reuse its cutoffs.')
    scores = pd.read_csv(results/'lightgbm_a2_fold_scores.csv')
    selected = Counter(scores['selected_params'])
    params = json.loads(selected.most_common(1)[0][0])
    cutoff = statistics.median(scores['threshold'].astype(float))
    print('Reading training rows only...', flush=True)
    train = pd.read_csv(train_path)
    X = build_inputs(train.copy())
    pipeline, _, vocabulary_limit = model_spec('lightgbm_a2', threads=2)
    pipeline.set_params(**params)
    print(f'Fitting LightGBM on {len(train)} training rows; params={params}; cutoff={cutoff:.6f}', flush=True)
    with threadpool_limits(limits=2):
        pipeline.fit(X, train[TARGET].to_numpy())
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination/'pipeline.tmp.joblib'
    joblib.dump(pipeline, temporary)
    temporary.replace(destination/'pipeline.joblib')
    metadata = dict(model='lightgbm_a2', model_version='lightgbm-a2-agent-v1',
        created_at=datetime.now(timezone.utc).isoformat(), training_rows=len(train),
        training_sha256=plan['training_sha256'], features_sha256=fingerprint(ROOT/'src/features.py'),
        artifact_sha256=fingerprint(destination/'pipeline.joblib'),
        selected_params=params, tfidf_max_features=vocabulary_limit, cutoff=cutoff,
        selection_policy='Most frequent inner-selected parameter setting across the saved training-only comparison folds.',
        cutoff_policy='Median of the 25 saved inner-OOF-selected cutoffs; integration default, not a newly optimized final cutoff.',
        heldout_test_used=False, calibrated_probability=False,
        purpose='Real classifier integration candidate; not a final-model selection or new performance claim.',
        versions={p: importlib.metadata.version(p) for p in ('numpy','pandas','scikit-learn','lightgbm','joblib')})
    (destination/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('Saved:', destination, flush=True)
    return metadata


def score_payload(posting, artifact=DEFAULT_ARTIFACT):
    artifact = Path(artifact)
    metadata = json.loads((artifact/'metadata.json').read_text(encoding='utf-8'))
    if fingerprint(artifact/'pipeline.joblib') != metadata['artifact_sha256']:
        raise ValueError('Classifier artifact hash does not match its metadata.')
    if fingerprint(ROOT/'src/features.py') != metadata['features_sha256']:
        raise ValueError('Classifier preprocessing code has changed since fitting.')
    # Only allowlisted posting fields enter the predictor. Labels and IDs are
    # never features; GROUP is required by build_inputs but A2 does not use it.
    row = {f: posting.get(f) if posting.get(f) is not None else np.nan for f in INPUT_FIELDS}
    row[GROUP] = 'inference-only'
    frame = pd.DataFrame([row])
    for field in RAW_BINARY:
        frame[field] = pd.to_numeric(frame[field], errors='raise').astype(float)
    with threadpool_limits(limits=2):
        pipeline = joblib.load(artifact/'pipeline.joblib')
        matrix = build_inputs(frame)
        classes = list(pipeline.named_steps['model'].classes_)
        score = float(pipeline.predict_proba(matrix)[0, classes.index(1)])
    missing = [f for f in INPUT_FIELDS if posting.get(f) is None or posting.get(f) == '']
    return dict(posting_id=posting['posting_id'], model_score=score, cutoff=metadata['cutoff'],
        input_sha256=hashlib.sha256(json.dumps({f: posting.get(f) for f in INPUT_FIELDS},
                       sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest(),
        flagged=score >= metadata['cutoff'], is_test_fixture=False, model_version=metadata['model_version'],
        model_family='LightGBM', calibrated_probability=False, artifact_sha256=metadata['artifact_sha256'],
        missing_input_fields=missing, threshold_policy=metadata['cutoff_policy'],
        note='Real model score, not a calibrated fraud probability. Missing fields stay missing; categorical preprocessing uses the fitted missing category and LightGBM handles absent binary inputs as missing. Short incomplete examples may differ substantially from training ads.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', action='store_true')
    parser.add_argument('--artifact', type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    if args.fit:
        fit_artifact(args.artifact)
    else:
        print(json.dumps(score_payload(json.load(sys.stdin), args.artifact), allow_nan=False))


if __name__ == '__main__':
    main()
