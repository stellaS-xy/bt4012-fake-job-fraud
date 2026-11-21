# Notebook 10: one complete outer fold on the cluster

Keep MLP job 1173687 and its `.venv-mlp`, source files, notebook, kernels and caches unchanged. This launch adds new files and uses a separate environment outside the repository. Do not pull or copy all of `src/` while that MLP job is running. No cluster connection, installation, download or submission is performed by preparing these files locally.

## 1. Upload only the new files from the Mac terminal

```bash
cd "/Users/yiting/Desktop/nus/y4s1/bt4012/group project"

scp -o ProxyJump=e1155697@sjump.comp.nus.edu.sg \
  requirements-10.txt requirements-10-cluster.txt \
  e1155697@xlogin1.comp.nus.edu.sg:~/bt4012-fake-job-fraud/

scp -o ProxyJump=e1155697@sjump.comp.nus.edu.sg \
  src/d1_distilbert.py \
  e1155697@xlogin1.comp.nus.edu.sg:~/bt4012-fake-job-fraud/src/

scp -o ProxyJump=e1155697@sjump.comp.nus.edu.sg \
  notebooks/10_distilbert_finetuning.ipynb \
  e1155697@xlogin1.comp.nus.edu.sg:~/bt4012-fake-job-fraud/notebooks/

scp -o ProxyJump=e1155697@sjump.comp.nus.edu.sg \
  cluster/distilbert_single_fold.slurm cluster/run_distilbert_notebook.py \
  cluster/DISTILBERT_README.md \
  e1155697@xlogin1.comp.nus.edu.sg:~/bt4012-fake-job-fraud/cluster/
```

The existing training CSV, saved folds, A2 scores and shared Python helpers from the MLP setup are reused read-only. No dataset re-upload is needed. Saved LightGBM fold scores are optional. Do not replace any running MLP file or install into its environment. Notebook 10 local flags/outputs do not matter: the launcher validates and overrides a COPY in memory; the original notebook remains unchanged.

## 2. Install a separate environment in the cluster terminal

CUDA-enabled PyTorch and its dependencies require several GB of additional home storage; the small pretrained checkpoint is not the main installation cost. Check personal quota if your site provides it. Shared filesystem free space is not your personal quota. Stop on `Disk quota exceeded`; do not repeatedly retry or delete the MLP environment/models.

```bash
cd ~/bt4012-fake-job-fraud
python3 -m venv "$HOME/venvs/bt4012-distilbert"
mkdir -p "$HOME/tmp/pip-distilbert"

TMPDIR="$HOME/tmp/pip-distilbert" \
  "$HOME/venvs/bt4012-distilbert/bin/python" -m pip install \
  --no-cache-dir --timeout 120 --retries 5 \
  torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

TMPDIR="$HOME/tmp/pip-distilbert" \
  "$HOME/venvs/bt4012-distilbert/bin/python" -m pip install \
  --no-cache-dir --timeout 120 --retries 5 -r requirements-10-cluster.txt

"$HOME/venvs/bt4012-distilbert/bin/python" -m pip check
```

This selects PyTorch 2.6.0 CUDA 12.4 from the official PyTorch wheel index, with Transformers 4.57.3. It avoids installing a current CUDA 13 wheel by accident. A sufficiently new NVIDIA host driver is still required; the compute job checks GPU availability and fails before fitting if CUDA is unavailable. No GPU should be requested just to install packages. Do not test training on a login node.

Optional: download the fixed pretrained files now, to avoid spending allocated GPU time on downloading or requiring compute-node internet. This is only a file download, not model execution:

```bash
"$HOME/venvs/bt4012-distilbert/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    'distilbert/distilbert-base-uncased',
    revision='12040accade4e8a0f71eabdb258fecc2e7e948be',
    allow_patterns=['config.json', 'tokenizer.json', 'tokenizer_config.json',
                    'special_tokens_map.json', 'vocab.txt', 'model.safetensors'])
PY
```

The checkpoint is cached in the standard Hugging Face cache, separate from Ollama. Once fully downloaded, you can submit with `--export=ALL,HF_HUB_OFFLINE=1` if compute nodes lack internet access.

## 3. Submit ONE outer fold

```bash
cd ~/bt4012-fake-job-fraud
sbatch cluster/distilbert_single_fold.slurm
```

Requests one A100 40 GB GPU, 4 CPUs, 16 GB host RAM, maximum 3 hours. Queueing/resource allocation is independent of MLP. It can wait while the MLP runs. The script does not cancel or attach to that job. If resources permit simultaneous allocation, both jobs can run in their separate environments.

The launcher sets installation off, token preparation on, resource-only diagnostic off, training on, outer limit 1 and device CUDA in an in-memory copy of notebook 10. The actual training settings remain batch size 2, accumulation 8, max 256 tokens, maximum 3 epochs, grouped stopping, 3 grouped inner folds and seed 42. These deliberately conservative settings allow the same candidate to be compared with the local design. CPU/MPS fallback is not permitted in this cluster launcher. There is no full-25-fold launch mode here.

One outer fold includes three inner fits and an outer fit, each with stopping selection followed by fresh refit: up to eight training phases. No local timing guarantees cluster runtime. No test data are loaded. A2/LGBM are read from saved fold scores, not retrained.

## 4. Monitor and read results

Replace `JOBID` with the submitted job ID:

```bash
squeue -j JOBID

tail -n 60 slurm-distilbert-JOBID.out

# After the notebook kernel starts, this file streams the notebook training output:
tail -f reports/10_distilbert_runs/JOBID/progress.log

sacct -j JOBID --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

Look for `COMPLETED` and `0:0`. Use the path printed in the notebook/log to find the matched comparison:

- `reports/10_distilbert_runs/JOBID/10_distilbert.executed.ipynb`: executed copy, saved after completed cells and on ordinary cell errors; source notebook unchanged.
- `reports/10_distilbert_runs/JOBID/environment.txt`: resolved packages.
- `reports/classifier_distilbert/<hash>/model_comparison.csv`: same-fold AP, precision, recall and F1.
- `reports/classifier_distilbert/<hash>/fold_scores.csv`: DistilBERT outer-fold results.
- `reports/classifier_distilbert/<hash>/comparison_sources.csv`: included/excluded references.
- `data/processed/10_distilbert/<hash>/`: independent token/inner/outer prediction caches.

The default first-fold A2 AP is approximately 0.7531, not its 25-fold mean of 0.6049. One fold is an exploratory screen. No adoption claim/confidence interval is made from it. Stop this candidate if its practical results are disappointing; do not tune repeatedly against this same outer validation fold.

If the job hits its time limit, completed inner/outer prediction phases can resume with the same submission command. The interrupted phase restarts; there is no epoch-level checkpoint. A forced kill can prevent saving the latest notebook cell output, although previously written prediction caches/progress remain. This experiment exports predictions/metrics, not deployable model weights.

Installation reference: [official PyTorch previous-version commands](https://pytorch.org/get-started/previous-versions/).
