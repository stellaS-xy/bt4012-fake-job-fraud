# Deploy the saved ensemble and agent

Use the existing cluster export and environments to run the finished system.
No model training or further LLM evaluation is needed to read the saved results.
The agent and classifier execute on the same allocated compute node, using
separate Python environments. No TensorFlow installation is required on a Mac
that only submits jobs and reads reports.

## Reuse the completed cluster export

From the cluster repository root:

```bash
export CLASSIFIER_BACKEND=ensemble
export CLASSIFIER_PYTHON="$PWD/.venv-mlp/bin/python"
export CLASSIFIER_ARTIFACT="$PWD/data/processed/final_ensemble_test/v1/artifact"
```

This uses the original retained model directly; no extraction or refitting is
required. The agent itself uses `agent/.venv`. Model and source/version checks can
run without TensorFlow execution:

```bash
"$CLASSIFIER_PYTHON" -m src.ensemble_inference --check --artifact "$CLASSIFIER_ARTIFACT"
```

TensorFlow import, predictions and Ollama serving belong inside a Slurm job,
not on a login node. Existing launchers perform compute-node preflight checks.
They never install packages or download weights. Inference uses CPU, while
Ollama uses the allocated GPU. See [environment selection](DEPENDENCIES.md).

## Install a trusted release on another checkout

Open the project's [GitHub Releases](https://github.com/stellaS-xy/bt4012-fake-job-fraud/releases).
The expected assets are `calibrated_mlp_extra_trees_v1.zip` and
`calibrated_mlp_extra_trees_v1.sha256`. A published release URL is not assumed:
if the assets are absent, the maintainer must publish the already-generated ZIP
and checksum using the instructions below, or transfer the trusted files directly.

The known v1 ZIP SHA256 is:

```text
81d5bcef760008a1145d99038aee6e3fa2bd6442a483cffef526510528b39320
```

After saving the ZIP under `models/downloads/`, verify and extract once:

```bash
python3 -m src.final_ensemble_artifact unpack \
  --archive models/downloads/calibrated_mlp_extra_trees_v1.zip \
  --destination models/final_ensemble_v1 \
  --sha256 81d5bcef760008a1145d99038aee6e3fa2bd6442a483cffef526510528b39320
```

The helper uses the standard library, validates the ZIP and each model file,
rejects unexpected members and refuses to overwrite an existing directory.
Only load trusted project artifacts: joblib deserialisation executes Python.
The extracted directory contains six model/preprocessor/calibrator files plus
`metadata.json`, `runtime_requirements.txt` and `export_verified.json`.

A fresh classifier environment needs the exported actual dependencies:

```bash
python3 -m venv .venv-final
.venv-final/bin/python -m pip install -r models/final_ensemble_v1/runtime_requirements.txt
export CLASSIFIER_PYTHON="$PWD/.venv-final/bin/python"
export CLASSIFIER_ARTIFACT="$PWD/models/final_ensemble_v1"
```

Use a Python/platform compatible with the export; the original successful Linux
fit used Python 3.12.3. These are installation instructions for a new host, not a
request to alter the existing cluster environment. The original `runtime_versions`
and source fingerprint must match. Configure the separate agent environment
from its lockfile, and use cached Ollama weights; see [configuration](CONFIGURATION.md).

## Saved jobs and optional verification commands

The completed final-fit/test job was **1177635** (6m31s). The completed balanced
agent audit was **1177902** (1h16m14s). Their aggregate results are preserved in
[the reports index](../reports/README.md); no new submission is needed.

For a future deployment check, the existing two-case synthetic smoke launcher is:

```bash
sbatch cluster/evaluator_smoke.slurm
```

It makes live LLM calls. It is a wiring check, not a new accuracy evaluation.
The [100-case audit guide](../cluster/BALANCED_EVALUATION_README.md) documents the
completed design and commands for deliberate reproduction. Keep its frozen
sample and results distinct from any future modified system. The balanced CLI
returns exit code 1 if any code check fails, even when all cases and reports
complete; inspect coverage and failure counts before interpreting Slurm status.

## Model release checklist

1. Commit the matching inference/feature source and aggregate reports to Git.
2. In GitHub, open Releases and draft a release with a tag such as
   `final-ensemble-v1`, pointing to the matching committed source.
3. Attach the existing ZIP and `.sha256` assets. Do not put weights, raw postings,
   per-posting predictions or training caches into normal Git.
4. Include the SHA above, model name, threshold, required runtime and link to
   this guide in the release notes. Add the actual asset URL here once published.
5. Teammates download, verify and load the saved model; they do not rerun fitting.

Documentation edits do not publish a release or push a commit. Keep existing
source fingerprints intact; restructuring or editing trained feature code would
require a separately versioned, verified deployment.
