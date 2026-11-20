# Run notebook 09 MLP on the SoC cluster

No local notebook execution or model training is needed. Submit all training through Slurm.

## 1. Upload the code and agreed inputs

Commit/push the changes and pull on the cluster, or copy the code files together. Required code includes `notebooks/09_MLP_model.ipynb`, `requirements-09.txt`, the complete `src/` directory and `cluster/`.

Data are ignored by Git. From your **Mac terminal**, upload the saved development folds (the training CSV was already uploaded; copy again only if needed):

```bash
scp -o ProxyJump=e1155697@sjump.comp.nus.edu.sg \
  "/Users/yiting/Desktop/nus/y4s1/bt4012/group project/data/processed/model_cv_assignments.csv" \
  e1155697@xlogin1.comp.nus.edu.sg:~/bt4012-fake-job-fraud/data/processed/
```

Required: the agreed `data/processed/d1_train.csv` and `data/processed/model_cv_assignments.csv`. Missing assignments stop the notebook; it does not generate replacement outer folds. The training file's SHA256 must match `61117d0309733e4dcf28d1f9836588ffaf72b7b44479113a742533134cad57ce`.

Optional existing reference results, copied with the same relative paths:

- `reports/tables/model_logreg_tf_idf_only_fold_scores.csv`
- `reports/tables/model_logreg_a2_fold_scores.csv`
- `reports/model_comparison_revised/lightgbm_a2_2k_fold_scores.csv`
- `reports/model_comparison_revised/lightgbm_a2_2k_plan.json`

These are read directly, never retrained. References with mismatched folds are excluded with reasons. The LightGBM plan must also match development data and preprocessing/evaluation source hashes. Outer-fold agreement alone does not imply identical historical inner tuning. No test CSV or locally installed LLM is needed.

## 2. Create a dedicated environment on the cluster

From `~/bt4012-fake-job-fraud` on the login node:

```bash
python3 --version
python3 -m venv .venv-mlp
mkdir -p "$HOME/tmp/pip-mlp"
TMPDIR="$HOME/tmp/pip-mlp" .venv-mlp/bin/python -m pip install --no-cache-dir -r requirements-09.txt
.venv-mlp/bin/python -m pip check
```

The cluster's Python 3.12 is supported by TensorFlow 2.21. Use this dedicated environment; do not install over `.venv` or `agent/.venv`. The requirements align scikit-learn with the current team environment and install Linux CUDA runtime dependencies via TensorFlow's `and-cuda` extra. Keras is resolved against TensorFlow rather than independently pinned. The job exports its resolved package versions for reproducibility. A compatible NVIDIA host driver is still required.

CUDA dependencies can consume several GB. Check your available quota before installation; changing `TMPDIR` or disabling pip caching does not relocate the installed environment or remove its storage requirement. No packages or models are installed by the Slurm script.

## 3. Submit one diagnostic outer fold first

```bash
sbatch cluster/mlp_model.slurm
```

Defaults: one A100 40 GB GPU, four CPUs, 32 GB host RAM, up to three hours. TensorFlow must see the allocated GPU; otherwise the job stops. This runs one complete outer fold (both freshly matched LR and MLP) with the full experiment's training settings, not a shortened epoch budget. The diagnostic is for checking resources and timing, not selecting the winning model.

Monitor using the returned job ID:

```bash
squeue -u "$USER"
tail -f slurm-mlp-JOBID.out
```

Notebook cell output is saved in the executed notebook; `nbclient` does not stream each cell's training output to the Slurm log. Completed fold caches can also be inspected while it runs. After completion:

```bash
sacct -j JOBID --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

## 4. Submit the full comparison

After the diagnostic succeeds and its elapsed time/resources have been checked:

```bash
sbatch --partition=gpu-long --time=3-00:00:00 \
  --export=ALL,MLP_RUN_MODE=full cluster/mlp_model.slurm
```

This requests up to the documented three-day limit, not an estimated runtime. Full mode evaluates 5 repeats × 5 outer folds. Diagnostic and full modes use identical fitted settings, so unchanged completed folds are reused. Do not run both modes concurrently against the same cache. Expect approximately 200 neural training phases and 250 LR fits before cache reuse. TF-IDF work still runs on CPUs. Model/architecture/epoch/inner-fold settings remain fixed.

## Outputs and interruption

- `data/processed/09_a2_mlp/<hash>/`: atomic compressed prediction cache after each complete model/outer fold. An interrupted in-progress model/fold restarts; no epoch-level recovery.
- `reports/tables/09_mlp/<hash>/<job ID>/`: comparison CSVs, MLP tuning choices, paired corrected intervals, excluded-reference reasons and provenance manifest.
- `reports/09_mlp_runs/<job ID>/`: executed notebook and resolved package versions.

An ordinary cell error saves the partially executed notebook with its traceback. A forced Slurm kill/time limit can prevent that final save, but previously completed fold caches remain. Re-submit with the same inputs and settings to resume. Code/package changes may invalidate caches. Reports are exported after the requested folds finish. These are development CV results and prediction caches, not deployable MLP weights; selecting the MLP would require a separate final fit/export. The final test is never read.
