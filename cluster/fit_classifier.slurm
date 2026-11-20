#!/bin/bash
#SBATCH --job-name=fit-job-classifier
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-fit-classifier-%j.out

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo 'Submit this script with sbatch from the repository root.' >&2
  exit 2
fi
cd "${SLURM_SUBMIT_DIR:?}"
test -f src/classifier_inference.py
test -f data/processed/d1_train.csv
test -f reports/model_comparison_revised/lightgbm_a2_2k_complete.json
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export PYTHONUNBUFFERED=1
echo "Node: $(hostname); job: $SLURM_JOB_ID"
.venv/bin/python -u -m src.classifier_inference --fit \
  --comparison-dir reports/model_comparison_revised

