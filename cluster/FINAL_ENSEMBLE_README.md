# Final MLP + Extra Trees model: cluster fit, fixed test evaluation, portable export

The selected model is notebook 13's fixed **sigmoid-calibrated 50/50 MLP + Extra Trees**, using A2 with a 30k TF-IDF cap, 128 Extra Trees with leaf size 1 and the original MLP settings. This is **not another 25-fold model comparison**. No training or test prediction was run locally when preparing these files.

## Protocol fixed before test scoring

1. Verify the exact audited train/test byte hashes, unique posting IDs and disjoint posting/group IDs. The files are 14,439 training postings and 3,441 test postings. Verify the selected component/calibration source implementations.
2. Use **development only** for three grouped inner fits of the complete calibrated ensemble. Select the final cutoff by maximum inner OOF F1. This is needed because notebook 13 has 25 fold-specific cutoffs, not one final deployable threshold.
3. Fit the final components on the training-side core groups. Hold out one of five grouped partitions (seed 342) to fit each component's natural-prevalence sigmoid calibrator. The MLP selects its epoch count using further groups inside the core, then refits there. Keep calibration groups out of component fitting; do **not** refit components after calibration. This reproduces notebook 13's procedure rather than fitting everything on all training rows.
4. Export both feature transformers, both models and both calibrators, the frozen cutoff and runtime versions. Verify exported inference against development calibration probes.
5. Only then score the current fixed test set. Save immutable predictions and calculate AP, ROC-AUC, precision/recall/F1, false-positive rate, counts and top-5% review metrics.
6. Report 1,000 percentile bootstrap replicates sampled by whole test groups (seed 2026). These are single-model performance intervals, **not paired superiority intervals**. Do not change model settings or cutoff after seeing test results.

The earlier audit found zero current train/test ID or group overlap, zero exact cleaned-text overlap and zero identical normalised long company profiles across this split. Historical project-wide test exposure is unverified, as documented in notebook 13. The result is a fixed-existing-split evaluation, not proof that these IDs were never encountered in any earlier project experiment. That caveat is retained in the frozen plan and final report.

## 1. Push code, pull on the cluster, upload the existing test CSV

Commit/push the new files using GitHub Desktop. In the cluster repository, inspect `git status` before pulling, and resolve local edits rather than overwriting them:

```bash
cd ~/bt4012-fake-job-fraud
git status
git pull --ff-only origin main
```

The training CSV is already there. If the test CSV is not yet uploaded, run this from the **Mac terminal**:

```bash
scp -o ProxyJump=e1155697@sjump.comp.nus.edu.sg \
  "/Users/yiting/Desktop/nus/y4s1/bt4012/group project/data/processed/d1_test.csv" \
  e1155697@xlogin1.comp.nus.edu.sg:~/bt4012-fake-job-fraud/data/processed/
```

## 2. Submit final fitting and test prediction

The script reuses the working cluster `.venv-mlp`; it installs nothing and does not alter it. It checks scikit-learn 1.9.1, TensorFlow 2.21.0 and a visible GPU. Other actual library versions are frozen into the run plan and model's `runtime_requirements.txt`. They may differ from notebook 13's original Mac environment, so exact numerical replication is not promised.

```bash
cd ~/bt4012-fake-job-fraud
sbatch cluster/final_ensemble_test.slurm
```

Default allocation: one A100 40 GB, four allocated CPUs (estimator thread settings remain two), 32 GB RAM, maximum three hours. The workflow has three training-side threshold stages and one final fit, rather than 25 outer folds. Runtime is not yet measured. Do not launch concurrent copies. A worker lock prevents writing the same result directory concurrently.

An optional different interpreter can be selected using `FINAL_ENSEMBLE_PYTHON`. `requirements-final-ensemble.txt` records the original selected numerical/neural versions for a separately prepared environment; Linux GPU use also requires compatible CUDA dependencies. Do not reinstall packages in `.venv-mlp` during other jobs.

```bash
squeue -j JOBID
tail -f slurm-final-ensemble-JOBID.out
sacct -j JOBID --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

All outputs are written under:

```text
data/processed/final_ensemble_test/v1/
```

The selected pipeline is frozen in `frozen_plan.json` before threshold training or test scoring. Existing completed inner prediction stages resume. In-progress component fits restart after an interruption. If a complete final result exists, rerunning displays it without training or new test predictions. A changed plan/source/runtime stops reuse instead of silently mixing runs.

## 3. Read the results

```bash
cat data/processed/final_ensemble_test/v1/test_metrics.json
```

Important files:

- `threshold_selection.json`: training-only threshold policy and cutoff.
- `training_stages/threshold_oof.csv`: development threshold predictions; not an independent performance result.
- `test_predictions.csv`: test IDs/groups/labels, scores and flags. Keep out of GitHub and the public model release.
- `test_metrics.json` / `.csv`: final metrics; JSON includes group-bootstrap intervals and the historical caveat.
- `artifact/`: retained models and preprocessing/calibration objects.
- `calibrated_mlp_extra_trees_v1.zip` and `.sha256`: portable, allowlisted model package and checksum.

Open `notebooks/15_final_ensemble_test_results.ipynb` to display downloaded saved results without fitting any model.

## 4. Download the model and results to your Mac

In the **Mac terminal**:

```bash
cd "/Users/yiting/Desktop/nus/y4s1/bt4012/group project"
mkdir -p models/final_ensemble_download

scp -o ProxyJump=e1155697@sjump.comp.nus.edu.sg \
  e1155697@xlogin1.comp.nus.edu.sg:~/bt4012-fake-job-fraud/data/processed/final_ensemble_test/v1/calibrated_mlp_extra_trees_v1.zip \
  e1155697@xlogin1.comp.nus.edu.sg:~/bt4012-fake-job-fraud/data/processed/final_ensemble_test/v1/calibrated_mlp_extra_trees_v1.sha256 \
  models/final_ensemble_download/

mkdir -p data/processed/final_ensemble_test/v1
scp -o ProxyJump=e1155697@sjump.comp.nus.edu.sg \
  e1155697@xlogin1.comp.nus.edu.sg:~/bt4012-fake-job-fraud/data/processed/final_ensemble_test/v1/test_metrics.json \
  e1155697@xlogin1.comp.nus.edu.sg:~/bt4012-fake-job-fraud/data/processed/final_ensemble_test/v1/test_metrics.csv \
  data/processed/final_ensemble_test/v1/
```

Verify the downloaded checksum, then extract into a new model directory:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from src.final_ensemble_artifact import unpack_artifact
folder = Path('models/final_ensemble_download')
checksum = (folder/'calibrated_mlp_extra_trees_v1.sha256').read_text().split()[0]
unpack_artifact(folder/'calibrated_mlp_extra_trees_v1.zip',
                Path('models/final_ensemble_v1'), checksum)
print('Model verified and extracted. No retraining performed.')
PY
```

The ZIP checksum is checked before extraction, unexpected members are rejected, and each model file is verified. Only use your own trusted project model package: joblib model files are executable serialised objects.

For local inference, create an isolated environment and install the **exported actual runtime requirements**, which contain no Linux CUDA extra. Do not train again:

```bash
python3.11 -m venv .venv-final
.venv-final/bin/python -m pip install --no-cache-dir \
  -r models/final_ensemble_v1/runtime_requirements.txt
```

The inference function is:

```python
from src.d1_final_ensemble import predict_artifact
scores = predict_artifact('models/final_ensemble_v1', posting_dataframe)
```

It expects the same raw posting fields used by the selected A2 pipeline. It is **not yet wired into the agent**: the agent's classifier tool still supports its existing LightGBM artifact. Do not call an agent evaluation "ensemble-based" until that adapter is changed and tested. The inference export itself can be reused without training.

## 5. Put the trained model on GitHub

Keep source, instructions and optional aggregate results in normal Git. Keep the ZIP and weights out of normal Git (`models/` and `data/processed/` are already ignored).

After the actual artifact exists:

1. Open the repository's **Releases → Draft a new release**.
2. Use a version tag such as `final-ensemble-v1`, tied to the committed inference code.
3. Attach `calibrated_mlp_extra_trees_v1.zip` and its `.sha256` file. Attach aggregate test metrics only if you want them published; do not attach posting data, per-posting labels, predictions or inner caches.
4. Record the actual download link and ZIP checksum in the repository instructions. Other teammates download that release, verify/extract it with the helper, and install the exported runtime requirements. They do not rerun model training.

No release is created or published by these scripts. GitHub blocks files above 100 MiB in normal Git and recommends Releases for distributing large binaries: [official large-file guidance](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github). Release assets are currently limited to under 2 GiB each: [release documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).
