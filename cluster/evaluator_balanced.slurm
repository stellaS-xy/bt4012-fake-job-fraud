#!/bin/bash
#SBATCH --job-name=agent-balanced20
#SBATCH --partition=gpu
#SBATCH --gpus=a100-40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-agent-balanced-%j.out

set -euo pipefail
SECONDS=0
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo 'Submit this script with sbatch from the repository root.' >&2
  exit 2
fi
cd "${SLURM_SUBMIT_DIR:?}"
PROJECT_DIR="$PWD"
test -f evaluator/balanced.py
: "${EVAL_SAMPLE_DIR:?Set EVAL_SAMPLE_DIR to an already locked sample directory. No dataset is selected automatically.}"
test -f "$EVAL_SAMPLE_DIR/sample.jsonl"
test -f "$EVAL_SAMPLE_DIR/sample_manifest.json"
test -f models/agent_lightgbm/pipeline.joblib
test -f models/agent_lightgbm/metadata.json
test -x "$HOME/ollama-local/bin/ollama"
export PATH="$HOME/ollama-local/bin:$PATH"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/ollama-models}"
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_CONTEXT_LENGTH=16384
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE=10m
export AGENT_BACKEND=ollama_local
export OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:14b-instruct}"
export JUDGE_BACKEND=ollama_local
export JUDGE_MODEL="$OLLAMA_MODEL"
export JUDGE_TIMEOUT_SECONDS="${JUDGE_TIMEOUT_SECONDS:-90}"
export CLASSIFIER_PYTHON="$PROJECT_DIR/.venv/bin/python"
export CLASSIFIER_ARTIFACT="$PROJECT_DIR/models/agent_lightgbm"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

OUTPUT_DIR="evaluator/outputs/balanced20_${SLURM_JOB_ID}"
test ! -e "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
echo "Node: $(hostname); job: $SLURM_JOB_ID; model: $OLLAMA_MODEL"
nvidia-smi
agent/.venv/bin/python -c 'import evaluator.balanced; print("Balanced evaluator imports OK")'
agent/.venv/bin/python -m pip freeze >"$OUTPUT_DIR/environment.txt"
.venv/bin/python -c 'import lightgbm, joblib; print("Classifier imports OK")'

# Never attach to or stop another job's existing localhost service.
agent/.venv/bin/python - <<'PY'
import socket
with socket.socket() as connection:
    if connection.connect_ex(('127.0.0.1', 11434)) == 0:
        raise SystemExit('Port 11434 is already occupied. Resubmit after resolving the conflict; do not stop another user/job service.')
PY

ollama serve >"$OUTPUT_DIR/ollama.log" 2>&1 &
OLLAMA_PID=$!
cleanup() {
  kill "$OLLAMA_PID" 2>/dev/null || true
  wait "$OLLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

READY=0
for attempt in {1..30}; do
  if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
    echo 'Ollama exited during startup:' >&2
    tail -n 50 "$OUTPUT_DIR/ollama.log" >&2
    exit 1
  fi
  if curl -fsS --max-time 1 http://127.0.0.1:11434/api/tags >"$OUTPUT_DIR/available_models.json" 2>/dev/null; then
    READY=1
    break
  fi
  sleep 1
done
if [[ "$READY" != 1 ]]; then
  echo 'Ollama did not become ready; inspect ollama.log.' >&2
  exit 1
fi

# Fail if cached weights are absent; never download a model in this smoke job.
agent/.venv/bin/python - "$OUTPUT_DIR/available_models.json" <<'PY'
import json, os, sys
models = json.load(open(sys.argv[1]))['models']
if not any(m.get('name') == os.environ['OLLAMA_MODEL'] for m in models):
    raise SystemExit('Requested model is absent from OLLAMA_MODELS. No download was attempted.')
PY

# Warm up before the agent's shorter per-request timeout starts applying.
agent/.venv/bin/python - "$OUTPUT_DIR" "$((300 - SECONDS))" <<'PY'
import json, os, pathlib, sys, urllib.request
folder = pathlib.Path(sys.argv[1])
request = urllib.request.Request('http://127.0.0.1:11434/api/generate',
    data=json.dumps(dict(model=os.environ['OLLAMA_MODEL'], prompt='Reply OK.',
                         stream=False, keep_alive='10m', options={'num_predict': 4})).encode(),
    headers={'Content-Type': 'application/json'})
remaining = float(sys.argv[2])
if remaining <= 0:
    raise SystemExit('Startup exceeded five-minute budget.')
with urllib.request.urlopen(request, timeout=remaining) as response:
    result = json.load(response)
if result.get('error') or not result.get('done'):
    raise SystemExit('Model warm-up failed: inspect the Ollama log.')
with urllib.request.urlopen('http://127.0.0.1:11434/api/ps', timeout=10) as response:
    resident = json.load(response)
(folder/'resident_models.json').write_text(json.dumps(resident, indent=2))
if not any(m.get('name') == os.environ['OLLAMA_MODEL'] and m.get('size_vram', 0) > 0
           for m in resident.get('models', [])):
    raise SystemExit('No GPU-resident model reported. Inspect Ollama/CUDA logs rather than silently evaluating on CPU.')
print('Model loaded on GPU; starting locked balanced quality audit.', flush=True)
PY

# Finish the evaluation loop by minute 55, reserving time for exports and cleanup.
REMAINING=$((3300 - SECONDS))
if (( REMAINING <= 0 )); then
  echo 'No runtime budget remains; no cases started.' >&2
  exit 1
fi
agent/.venv/bin/python -u -m evaluator.balanced run \
  --sample-dir "$EVAL_SAMPLE_DIR" \
  --output-dir "$OUTPUT_DIR/run" \
  --max-seconds "$REMAINING" --case-seconds 150
echo "Finished. Quality summary: $OUTPUT_DIR/run/quality_summary.csv"
echo "HTML report: $OUTPUT_DIR/run/review_report.html"
