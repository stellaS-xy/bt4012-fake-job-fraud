#!/bin/bash
#SBATCH --job-name=job-mlp
#SBATCH --partition=gpu
#SBATCH --gpus=a100-40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-mlp-%j.out

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo 'Submit with sbatch from the repository root.' >&2
  exit 2
fi
cd "${SLURM_SUBMIT_DIR:?}"
test -x .venv-mlp/bin/python
test -f data/processed/d1_train.csv
test -f data/processed/model_cv_assignments.csv
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TF_NUM_INTRAOP_THREADS="$OMP_NUM_THREADS"
export TF_NUM_INTEROP_THREADS=1
export MLP_REPEATS=5
case "${MLP_RUN_MODE:-diagnostic}" in
  diagnostic) export MLP_OUTER_LIMIT=1 ;;
  full) export MLP_OUTER_LIMIT=all ;;
  *) echo 'MLP_RUN_MODE must be diagnostic or full.' >&2; exit 2 ;;
esac
# Ensure the notebook kernel uses this dedicated environment, not a global kernel.
export JUPYTER_DATA_DIR="$PWD/.venv-mlp/share/jupyter"
.venv-mlp/bin/python -m ipykernel install --prefix "$PWD/.venv-mlp" --name python3 --display-name 'MLP cluster'
mkdir -p "reports/09_mlp_runs/$SLURM_JOB_ID"
.venv-mlp/bin/python -m pip freeze > "reports/09_mlp_runs/$SLURM_JOB_ID/environment.txt"
echo "Node: $(hostname); job: $SLURM_JOB_ID; mode: ${MLP_RUN_MODE:-diagnostic}"
nvidia-smi
# Fails on missing GPU before any model fits; the notebook repeats this check.
.venv-mlp/bin/python -c 'import tensorflow as tf; g = tf.config.list_physical_devices("GPU"); print("TensorFlow GPUs:", g); assert g, "No GPU detected: resolve CUDA/driver setup before training"'
.venv-mlp/bin/python -u cluster/run_mlp_notebook.py
