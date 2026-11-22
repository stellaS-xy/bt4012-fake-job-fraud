# Agent and evaluator configuration

Keep real values in ignored `.env` files. Copy the examples and edit paths and
model names for the machine executing the system. The existing cluster Slurm
scripts set their own process environment; editing a Mac `.env` does not change
a submitted cluster job.

## Agent settings

Agent precedence, from lowest to highest: Python defaults, project-root `.env`,
`agent/.env`, then supported process environment variables. Empty and placeholder
file values are skipped. Relative classifier paths resolve from the repository
root, not from `agent/`.

| Variable | Default when unset | Meaning |
|---|---|---|
| `AGENT_BACKEND` | `ollama_local` | Local Ollama, or `ollama` for Ollama cloud |
| `OLLAMA_MODEL` | `qwen3:4b-instruct` locally; `deepseek-v4.1-flash` for cloud | Model identifier; cluster launchers default to cached `qwen2.5:14b-instruct` |
| `OLLAMA_API_KEY` | No real key needed locally | Required for cloud only; local mode substitutes the local placeholder |
| `CLASSIFIER_BACKEND` | `auto`, resolved to `ensemble` | `ensemble` for final v1; explicit `lightgbm` retains the historical adapter |
| `CLASSIFIER_ARTIFACT` | `models/final_ensemble_v1` for ensemble | Directory containing all nine exported inference files |
| `CLASSIFIER_PYTHON` | `.venv-mlp/bin/python` on Linux for ensemble | Separate interpreter with the exported runtime versions |

Explicit legacy LightGBM uses `models/agent_lightgbm` and `.venv/bin/python` by
default. It is not a fallback for a missing ensemble. Windows defaults use
`Scripts/python.exe`. Neither backend automatically trains missing weights.

Local requests use `http://localhost:11434/v1`; cloud requests use
`https://ollama.com/v1`. Arbitrary base URLs and direct DeepSeek endpoints are
not supported by the environment loader. Agent request timeout (60 seconds)
and recursion limit (18) are Python settings, not supported `.env` keys.

## Judge settings

Judge precedence: defaults, project-root `.env`, `evaluator/.env`, then `JUDGE_*`
process variables. The judge configuration is independent of `agent/.env`.

| Variable | Default | Meaning |
|---|---|---|
| `JUDGE_BACKEND` | `ollama_local` | Local Ollama or Ollama cloud |
| `JUDGE_MODEL` | Backend default described above | Can differ from the actor model |
| `JUDGE_API_KEY` | Unset | Required only for a cloud judge; not inherited from `OLLAMA_API_KEY` |
| `JUDGE_TIMEOUT_SECONDS` | 180 locally, 60 in cloud | Positive per-request timeout |

The 100-case launcher sets both actor and judge to the same Qwen2.5 14B model,
with judge timeout 120 seconds unless already exported. This saves model memory
but permits shared actor/judge errors. It is not an independent human evaluator.

## Cluster launcher settings

| Variable | Usage |
|---|---|
| `EVAL_SAMPLE_DIR` | Required locked sample directory for the balanced job; no dataset is automatically chosen |
| `CLASSIFIER_ARTIFACT` | Overrides the default artifact directory in either evaluator launcher |
| `CLASSIFIER_PYTHON` | Overrides `.venv-mlp/bin/python` in either evaluator launcher |
| `OLLAMA_MODELS` | Cached weights directory; defaults to `~/ollama-models` |
| `OLLAMA_MODEL` | Cached actor model; launcher also assigns this model to the judge |
| `JUDGE_TIMEOUT_SECONDS` | Balanced launcher defaults to 120; smoke launcher defaults to 300 |
| `FINAL_ENSEMBLE_PYTHON` | Historical final-fit launcher interpreter override; not needed to use saved weights |

The launchers force local actor/judge backends and ensemble classifier backend.
They start a job-owned localhost Ollama server on port 11434, use a 16,384-token
context setting and allow one model/request at a time. Classifier subprocesses
use CPU. Supported overrides and full commands are in the
[deployment guide](DEPLOYMENT.md) and [balanced audit guide](../cluster/BALANCED_EVALUATION_README.md).

The balanced task requests eight hours on `gpu-long`, allows 240 seconds per
whole case, and reserves ten minutes for export/cleanup. These are resource
ceilings, not an instruction to rerun the completed evaluation. New job-specific
output paths preserve earlier results. There is no automatic balanced-run resume.
