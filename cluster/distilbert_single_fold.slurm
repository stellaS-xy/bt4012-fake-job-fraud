#!/bin/bash
#SBATCH --job-name=distilbert-one
#SBATCH --partition=gpu
#SBATCH --gpus=a100-40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-distilbert-%j.out

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo 'Submit with sbatch from the repository root.' >&2
  exit 2
fi
cd "${SLURM_SUBMIT_DIR:?}"
DISTILBERT_PYTHON="${DISTILBERT_PYTHON:-$HOME/venvs/bt4012-distilbert/bin/python}"
test -x "$DISTILBERT_PYTHON"
test -f notebooks/12_distilbert_finetuning.ipynb
test -f src/d1_distilbert.py
test -f data/processed/d1_train.csv
test -f data/processed/model_cv_assignments.csv
test -f reports/tables/model_logreg_a2_fold_scores.csv
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
# Private kernelspec: no modification to the running MLP's environment/kernel.
DISTILBERT_PREFIX="$(dirname "$(dirname "$DISTILBERT_PYTHON")")"
export JUPYTER_DATA_DIR="$DISTILBERT_PREFIX/share/jupyter"
"$DISTILBERT_PYTHON" -m ipykernel install --prefix "$DISTILBERT_PREFIX" --name python3 --display-name 'DistilBERT cluster'
mkdir -p "reports/10_distilbert_runs/$SLURM_JOB_ID"
"$DISTILBERT_PYTHON" -m pip freeze > "reports/10_distilbert_runs/$SLURM_JOB_ID/environment.txt"
echo "Node: $(hostname); job: $SLURM_JOB_ID; one complete outer fold"
nvidia-smi
"$DISTILBERT_PYTHON" -c 'import torch; print("Torch:", torch.__version__, "CUDA:", torch.version.cuda); assert torch.cuda.is_available(), "CUDA unavailable: check allocated GPU, driver and PyTorch wheel"; print("GPU:", torch.cuda.get_device_name(0))'
"$DISTILBERT_PYTHON" -u cluster/run_distilbert_notebook.py
