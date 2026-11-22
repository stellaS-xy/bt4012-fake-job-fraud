# Run the agent with the released ensemble v1 on the SoC cluster

The default classifier is the saved, calibrated 50/50 MLP + Extra Trees v1.
No retraining, fold reruns or test-set predictions are needed to integrate it.
Ollama serves the agent/judge on the allocated GPU; classifier inference runs
on CPU in a separate `.venv-mlp` subprocess. The agent uses `agent/.venv`.
Do not run TensorFlow or Ollama on a login node.

For a single current entry point, use [deployment](../docs/DEPLOYMENT.md),
[dependencies](../docs/DEPENDENCIES.md) and [configuration](../docs/CONFIGURATION.md).
The [completed 100-case report](../reports/agent_evaluation/ensemble_v1_100cases/README.md)
is available without another run.

## 1. Prepare the model files once

After committing/pushing the integration code locally, run `git pull --ff-only`
from the cluster repository. Reuse the ZIP from the completed final-fit job,
or download the same ZIP and checksum from the project's GitHub Release when
published. Run the following from the repository root. This command uses only
the Python standard library; it does not load TensorFlow or run predictions.

```bash
python3 -m src.final_ensemble_artifact unpack \
  --archive data/processed/final_ensemble_test/v1/calibrated_mlp_extra_trees_v1.zip \
  --destination models/final_ensemble_v1 \
  --sha256 81d5bcef760008a1145d99038aee6e3fa2bd6442a483cffef526510528b39320
```

The destination must not already exist; the helper refuses to overwrite it.
The ZIP contains nine inference files, including `runtime_requirements.txt`.
Use only the trusted project release: joblib files are executable Python objects.
Model files and environments remain ignored by Git.

## 2. Reuse the existing working environments

The final-fit environment `.venv-mlp` already has the exported versions, including
TensorFlow 2.21.0 and scikit-learn 1.9.1. Do not reinstall it if it is intact.
The agent/evaluator environment `agent/.venv` and cached Ollama installation/model
are unchanged. No dependencies or weights need to be installed on your Mac.
For a fresh deployment, install the ZIP's `runtime_requirements.txt` into the
classifier environment and `agent/requirements-lock.txt` into a separate agent
environment. Follow cluster policies for package installation.

The supplied Slurm scripts explicitly override classifier settings to:

```dotenv
CLASSIFIER_BACKEND=ensemble
CLASSIFIER_PYTHON=.venv-mlp/bin/python
CLASSIFIER_ARTIFACT=models/final_ensemble_v1
```

Both paths become absolute in the job. Existing LightGBM values in `.env` do not
override these job settings. To reuse the original export without duplicating
files, submit with this override instead of unpacking:

```bash
export CLASSIFIER_ARTIFACT="$PWD/data/processed/final_ensemble_test/v1/artifact"
```

## 3. Submit the two-case smoke test

```bash
sbatch cluster/evaluator_smoke.slurm
# Replace JOB_ID with the submitted job number:
squeue -j JOB_ID
tail -f slurm-agent-smoke-JOB_ID.out
```

Preflight checks verify all six model/preprocessor/calibrator hashes, the
preprocessing source fingerprint, saved threshold and every exported runtime
version, before starting Ollama. A mismatch stops the job; it does not fit a
replacement. TensorFlow import checks happen on the allocated compute node.
Ollama uses the cached `qwen2.5:14b-instruct`; no weights are downloaded.
The two examples are synthetic and have explicitly supplied binary attributes;
this is a wiring check, not a fraud-performance estimate. Reports go to
`evaluator/outputs/slurm_smoke_JOB_ID/`.

## 4. Run the existing locked balanced quality audit

After the smoke test succeeds, use an already prepared/locked sample:

```bash
export EVAL_SAMPLE_DIR=/absolute/path/to/existing/locked_sample
sbatch cluster/evaluator_balanced.slurm
```

No sample is selected automatically. The 50+50 audit remains an agent-quality
audit, not an estimate at the dataset's natural prevalence. Outputs use a new
job-specific directory and record ensemble checksums, training identity,
threshold and adapter source hashes; historical LightGBM reports stay unchanged.

## Inputs and interpretation

The trained pipeline passes `telecommuting`, `has_company_logo`, and
`has_questions` directly to the MLP. All three must be supplied as 0/1; unknown
values are never silently converted to zero. Missing binary inputs cause the
classifier tool to fail, and the agent must return an incomplete/review-needed
assessment without inventing a score. Other optional fields retain their fitted
missing-value handling. The frontend exposes Yes/No/Unknown controls.

The adapter loads the saved sigmoid calibrators and averages their scores
50/50, preserving the saved development-selected threshold
`0.20923746948802296`. It does not optimise anything on new inputs. Calibration
is not a guarantee of real-world probability accuracy. Review-needed is not a
fraud label. A classification flag alone does not explain *why* a posting was
flagged; the lexicon/context tool supplies separate evidence.

Legacy LightGBM inference is still available only by explicitly setting
`CLASSIFIER_BACKEND=lightgbm` and the matching legacy artifact/interpreter paths
outside these ensemble-specific jobs.
