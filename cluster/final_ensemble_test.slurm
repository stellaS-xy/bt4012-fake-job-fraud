#!/bin/bash
#SBATCH --job-name=ensemble-final-test
#SBATCH --partition=gpu
#SBATCH --gpus=a100-40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-final-ensemble-%j.out

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo 'Submit using sbatch from the repository root.' >&2
  exit 2
fi
cd "${SLURM_SUBMIT_DIR:?}"
# Reuse the working TensorFlow environment; never install/alter packages in a job.
FINAL_ENSEMBLE_PYTHON="${FINAL_ENSEMBLE_PYTHON:-$PWD/.venv-mlp/bin/python}"
test -x "$FINAL_ENSEMBLE_PYTHON"
test -f data/processed/d1_train.csv
test -f data/processed/d1_test.csv
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=1
OUTPUT=data/processed/final_ensemble_test/v1
mkdir -p "$OUTPUT"
"$FINAL_ENSEMBLE_PYTHON" -m pip freeze > "$OUTPUT/environment_${SLURM_JOB_ID}.txt"
echo "Node: $(hostname); final fit/test job: $SLURM_JOB_ID"
nvidia-smi
"$FINAL_ENSEMBLE_PYTHON" -c 'import tensorflow as tf, sklearn; print("TensorFlow",tf.__version__,"sklearn",sklearn.__version__); assert sklearn.__version__ == "1.9.1", "Preserve the selected grouped splitter version"; assert tf.__version__ == "2.21.0", "Use the selected TensorFlow version"; assert tf.config.list_physical_devices("GPU"), "No TensorFlow GPU detected"'
"$FINAL_ENSEMBLE_PYTHON" -u -m src.d1_final_ensemble --run-dir "$OUTPUT"
