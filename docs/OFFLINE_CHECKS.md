# Offline software checks

The [GitHub Actions workflow](../.github/workflows/offline-checks.yml) runs on
pushes, pull requests and manual workflow dispatch. It installs test dependencies,
then performs local software checks. Tests make no live LLM calls, load no real
model weights, import no TensorFlow and read no private datasets. Pip installation
still requires package-registry access on the GitHub runner.

## Scope

The selected suite covers final protocol/source checks and portable packaging,
ensemble adapter input identity and validation, legacy classifier provenance,
balanced sampling and failure accounting, and runtime configuration. It uses
synthetic fixtures and mocked inference boundaries. These are software tests,
not another model experiment or performance evaluation.

CI also compiles Python sources and checks Slurm scripts with `bash -n`. It does
not run notebooks, contact Ollama, fit the final model, submit Slurm jobs, publish
releases or verify GPU runtime compatibility. Full agent/frontend/judge suites
have additional dependencies and are not represented as covered by this workflow.

## Reproduce the selected suite

Use a separate Python 3.12 environment; do not reinstall the working inference
environment just to run CI tests:

```bash
python3 -m venv /tmp/bt4012-offline-checks
/tmp/bt4012-offline-checks/bin/python -m pip install -r tests/requirements-offline.txt
/tmp/bt4012-offline-checks/bin/python -m unittest -v \
  tests.test_final_ensemble \
  tests.test_ensemble_inference \
  tests.test_classifier_inference \
  evaluator.tests.test_balanced \
  agent.tests.test_runtime_config
```

The same suite can run in an existing compatible scientific/test environment.
Pinned offline dependencies describe the tested suite, not the classifier
export: the model still requires its own exact `runtime_requirements.txt`.
A green CI run does not guarantee real provider availability or semantic
correctness of agent answers. A workflow added locally has no GitHub run until
it is committed and pushed.
