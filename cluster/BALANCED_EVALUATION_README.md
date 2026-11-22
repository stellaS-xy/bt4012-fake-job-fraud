# Balanced 100-case agent-quality audit

Protocol `balanced-agent-quality-v2` uses **50 genuine + 50 fraudulent postings
from 100 distinct near-duplicate groups**, fixed seed 2026, and one agent/judge
execution per case. It audits workflow and explanation quality, not performance
at natural fraud prevalence. Preparation makes no model calls. Existing locked
20-case v1 samples and their reports are never expanded or overwritten.

The completed run is job **1177902** (1h16m14s). Read the
[aggregate results and error analysis](../reports/agent_evaluation/ensemble_v1_100cases/README.md)
without submitting another job. The commands below document deliberate reproduction.

## Deployment and freeze

Use the [ensemble deployment guide](ENSEMBLE_AGENT_README.md) first. The launcher
uses saved ensemble v1 through `.venv-mlp`, plus the existing `agent/.venv` and
cached Qwen2.5 14B for both agent and judge. Classifier inference uses CPU;
Ollama uses the allocated GPU. No new fitting or model downloads are needed.

Commit/push these code changes, then pull them on the cluster when no running
job is using the affected files. No new packages are required for this change.
Freeze prompts, tools, model artifact and rubric before a formal heldout audit.
A frozen-version label records a declaration, not proof of historical isolation.

## Prepare a new locked sample

From the cluster repository root, choose **one** source. During system development:

```bash
agent/.venv/bin/python -m evaluator.balanced prepare \
  --data data/processed/d1_train.csv \
  --split development \
  --output-dir evaluator/outputs/balanced100_development_2026
```

This development audit can inform debugging, but its classifier results are not
an independent generalisation estimate because the classifier used development
data. Once the complete system is frozen, use the explicitly declared heldout
source instead (never run both commands into the same directory):

```bash
agent/.venv/bin/python -m evaluator.balanced prepare \
  --data data/processed/d1_test.csv \
  --split heldout --frozen-version ensemble-v1-agent-eval-v2 \
  --training-data data/processed/d1_train.csv \
  --output-dir evaluator/outputs/balanced100_heldout_2026
```

Heldout preparation checks group separation against the training file. It does
not prove the test file has never been accessed elsewhere in the project.
If the source cannot provide the required distinct groups, preparation fails
rather than weakening the design or substituting cases.

Sampling chooses a group, then one random posting of the requested label.
Mixed-label group allocation preserves both quotas without reusing groups.
The run alternates labels with a random starting class. There is no selection
based on predictions, errors, wording or length. Selected invalid inputs remain
failures; no replacement is drawn. Three binary classifier fields must be
available to score the posting; unknown values are not silently filled as zero.

The new directory contains:

- `sample.jsonl`: supported posting fields plus evaluator annotations. The loader
  strips labels and groups before passing postings to the agent; the judge also
  does not see the gold fraud label.
- `sample_manifest.json`: source/sample hashes, IDs/groups, lengths, split,
  protocol, seed and declared frozen version.
- `manual_review.csv`: **20 preselected cases (10 per class)**, seed 2027.
  Human-review fields start blank/pending. Review their evidence, interpretations
  and judge agreement after execution; the model does not perform this audit.

Samples are locked and ignored by Git. Do not redraw after seeing results.

## Submit and monitor

Reuse the final-fit artifact directly, without making another copy:

```bash
export CLASSIFIER_ARTIFACT="$PWD/data/processed/final_ensemble_test/v1/artifact"
export CLASSIFIER_PYTHON="$PWD/.venv-mlp/bin/python"
export EVAL_SAMPLE_DIR="$PWD/evaluator/outputs/balanced100_development_2026"
# For the frozen heldout design, set EVAL_SAMPLE_DIR to that locked directory instead.
sbatch cluster/evaluator_balanced.slurm
```

Without an artifact override, the script uses `models/final_ensemble_v1`.
It requires a locked v2 100-case manifest; an old 20-case sample is rejected
before loading the LLM. No dataset is automatically selected.

```bash
squeue -j JOBID
tail -f slurm-agent-balanced-JOBID.out
sacct -j JOBID --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

## Runtime and failures

- Allocation: `gpu-long`, A100 40 GB, four CPUs, 32 GB RAM, **8 hours maximum**,
  excluding queue wait. This is a ceiling, not an expected runtime. Earlier
  two-case timings suggest roughly 2–4 hours for 100 cases, with substantial
  uncertainty from text length, retries and timeouts.
- Whole-case deadline: **240 seconds**, including imports, classifier, all agent
  and context calls, judge and validation correction. The judge request timeout
  defaults to 120 seconds; the agent request timeout remains 60 seconds.
- Evaluation ends by minute 470 (7h50m), reserving ten minutes for export/cleanup.
  Startup and warm-up have an approximately five-minute budget.
- Agent results are checkpointed before judging. A judge timeout can preserve the
  completed agent/classifier result; earlier failures have no invented prediction.
- Timeouts, failures and unstarted cases remain in the denominator of **100**.
  Missing judge scores remain null, not zero. Score means carry eligible counts.
- Only the job-owned server/processes are stopped. Port conflicts cause failure
  rather than attaching to another job's Ollama instance.
- Reports are written incrementally. There is **no automatic resume, retry or
  replacement sampling**. Forced Slurm termination may interrupt the final write;
  preserve existing checkpoints and disclose any separately planned follow-up.

## Outputs and interpretation

Outputs: `evaluator/outputs/balanced100_JOBID/`, including Ollama/environment logs
and `run/` containing:

- `runtime_manifest.json`: model identities, source/rubric/package provenance,
  ensemble checksums and saved threshold, excluding secrets.
- `report.json`, `quality_summary.csv` and `.json`: overall and class-specific
  completion, hard-check success, judge coverage, eligible judge-score means,
  unsupported-claim case counts, failures/timeouts and latency.
- `review_report.html`: readable posting, tools, evidence, answer and judge report.
- `results.jsonl`, `cases.jsonl`, `case_XX/`: incremental results and checkpoints.

A nonzero job exit can mean incomplete agent/judge execution or failed hard
checks even when the report was saved. Inspect the report before assuming an
infrastructure failure. Low judge ratings alone do not fail the job.

Report all-case and per-class outcomes with denominators. Finish the 20-case
human review separately. The same agent/judge model can share errors. A balanced,
group-sampled audit does not estimate production precision or workload, and
review-needed is not a gold fraud label. Use the separate full-test classifier
results for its natural-distribution evaluation. One execution per case also
does not measure variation across repeated LLM responses.
