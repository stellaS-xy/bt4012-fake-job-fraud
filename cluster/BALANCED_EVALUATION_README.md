# Balanced 20-case agent-quality audit (one-hour cluster job)

This implements a **quality audit**, not model selection or a prevalence-representative accuracy estimate. It uses 10 genuine and 10 fraudulent real postings, 20 distinct near-duplicate groups, fixed sampling seed 2026 and one execution per case. No test file is automatically read. No LLMs, classifier training or model downloads are performed by preparing this code locally.

## What must be fixed first

The current agent tool still supports **LightGBM**. This launcher evaluates that existing artifact; it does not convert or deploy the MLP. To claim an MLP-based agent evaluation, finish that integration and freeze it before preparing the formal sample. Do not change prompts, tools, classifier artifacts or judge rubric after seeing heldout results. A `--frozen-version` is a declaration/provenance label, not proof that freezing occurred.

## 1. Transfer the implementation

Use GitHub Desktop to commit/push the new balanced evaluator, tests, documentation and the small evaluator/runner.py checkpoint change. On the cluster, inspect `git status` and pull normally if the working tree permits. Do not overwrite unrelated uncommitted cluster changes. If another job is actively using these evaluator files, wait for it to finish before updating them.

No new Python packages are required beyond the agent/evaluator environment already installed. Nothing is installed in `.venv-mlp` or the DistilBERT environment.

## 2. Prepare and lock the sample — explicit action, no model calls

Run preparation once, on the cluster, from `~/bt4012-fake-job-fraud`.

If the system is **still being developed**, use development data:

```bash
agent/.venv/bin/python -m evaluator.balanced prepare \
  --data data/processed/d1_train.csv \
  --split development \
  --output-dir evaluator/outputs/balanced20_development_2026
```

If the complete system is **frozen for final evaluation**, upload the agreed test CSV if it is not on the cluster, then use the explicit heldout command instead. This reads the test file and checks that its groups do not overlap training; do not execute it during prompt/model development:

```bash
agent/.venv/bin/python -m evaluator.balanced prepare \
  --data data/processed/d1_test.csv \
  --split heldout --frozen-version final-agent-v1 \
  --training-data data/processed/d1_train.csv \
  --output-dir evaluator/outputs/balanced20_heldout_2026
```

Preparation validates binary labels, unique posting IDs and nonmissing groups. It samples groups without replacement within labels, then one random posting of the requested label within each selected group. Mixed-label groups are allocated without repeating a group or exhausting the other label's quota. The order alternates labels with a random starting class. This is a group-distinct, class-balanced audit, not a uniformly random posting-level sample. No prediction, correctness, wording or length filter is used. Invalid posting text, if selected, remains an evaluation input failure; no replacement is drawn.

A new directory contains:

- `sample.jsonl`: only supported posting fields, IDs, groups and labels. The worker input loader strips annotations/group IDs before sending postings to the agent; the judge also never sees the gold label.
- `sample_manifest.json`: source/sample hashes, split, frozen declaration, group IDs, lengths, selected IDs and sampling rules.
- `manual_review.csv`: four cases preselected with seed 2027 (two per label); all human-review fields are initially blank/pending. Complete after inspecting their actual outputs. This is additional human time, not GPU time.

Existing sample directories are never overwritten. Keep using the same locked sample; do not draw alternatives based on the results. All these paths are ignored by Git through `evaluator/outputs/`.

## 3. Submit the one-hour job

For the development sample above:

```bash
sbatch --export=ALL,EVAL_SAMPLE_DIR=evaluator/outputs/balanced20_development_2026 \
  cluster/evaluator_balanced.slurm
```

For a formally prepared heldout sample, substitute `balanced20_heldout_2026`. The script does not select either dataset by default and does not prepare new samples.

Defaults: A100 40 GB, four CPUs, 32 GB RAM, Qwen2.5 14B for agent and judge, existing LightGBM classifier, one repeat. It starts its own Ollama server only if the local port is free and cached weights are present; it does not attach to or stop other services or download a model. A different actor/judge setup requires a separately frozen evaluation design.

## Runtime bounds

- Slurm maximum wall time: **60 minutes**, excluding queue wait.
- Target: complete 20 cases, approximately 30–45 minutes based on earlier smoke timings, not guaranteed.
- Each whole worker has **150 seconds**, including imports, classifier subprocess, all agent/context calls, judge and validation correction.
- The job starts the evaluator with the remaining time to minute 55, reserving time for report output and cleanup. It stops launching cases when there is insufficient remaining time for a complete case budget. Startup/warm-up is bounded at approximately five minutes.
- A timeout kills only that isolated worker process group, including its classifier subprocess. The shared job-owned Ollama service stays alive for subsequent cases. Inference cancellation on the server may not be instantaneous.
- An ordinary case/worker failure does not abort the other selected cases or trigger replacement sampling.
- Completed agent output is checkpointed **before the judge**. A judge-stage timeout preserves validated classifier predictions, agent status, evidence and code checks. A timeout before that checkpoint has no claimed classifier result.
- Unstarted cases and timeouts remain in the denominator of 20. Missing judge scores stay null, not zero and not a pass. Judge means always carry their eligible scored counts.

The wall-time cap is a resource limit, not a promise of 20 successful judgments. Unexpected OS/filesystem failures or forced Slurm termination can prevent the final update; previously written atomic reports and per-case checkpoints remain. There is no automatic retry/resume/replacement: preserve failures in this run and disclose any separately planned follow-up.

## 4. Monitor

Replace JOBID with the actual ID:

```bash
squeue -j JOBID

tail -f slurm-agent-balanced-JOBID.out

sacct -j JOBID --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

Outputs are under `evaluator/outputs/balanced20_JOBID/`:

- `environment.txt`, `available_models.json`, `resident_models.json`, `ollama.log`: environment/service provenance.
- `run/runtime_manifest.json`: allowlisted model settings, source/package/rubric and classifier artifact hashes; no API keys.
- `run/report.json`: all 20 cases, execution state, legacy evaluator summary and the balanced-quality summary.
- `run/quality_summary.csv` / `.json`: **primary results**, overall and by label; planned counts, completion/code-pass/coverage rates, effective dimension means and scored counts, unsupported-claim case count with judge denominator, fallback/timeout counts, median/maximum latency.
- `run/review_report.html`: readable posting/tool/answer/judge report with a prominent balanced-sampling warning and quality summary. Legacy classifier and recommendation-label panels remain descriptive only and must not be reported as population metrics.
- `run/results.jsonl`: incremental case results, including failures and unstarted cases.
- `run/cases.jsonl`: saved agent outputs that can be rejudged later without rerunning the agent, if a separate replay is intentionally planned and disclosed.
- `run/case_XX/`: worker log, result and pre-judge checkpoint when available. The label in metadata is never an agent text input.

The CLI exits nonzero when agent/judge execution or hard checks are incomplete/failed, even if all 20 cases were attempted and the report was successfully saved. **Inspect the report before treating a Slurm FAILED status as a missing evaluation.** Low judge ratings alone do not fail the job.

## How to report

Report all-case and class-specific completion/coverage, effective judge scores with denominators, unsupported-claim cases and timing. Do not average missing ratings as zero or drop failed cases from completion denominators. A judge that finishes after an agent fallback may still have some dimensions excluded by the existing eligibility rules.

The balanced dataset changes the base rate. It cannot support production precision, natural-distribution quality estimates or claims of overall fraud accuracy. `review_needed` is a recommendation, not a gold fraud decision. The same actor/judge model can reproduce shared mistakes. Twenty cases and one execution each are a limited audit, not statistical proof of deployment readiness. Finish the four-case human audit separately and label it as such.

Suggested methods description:

> We used a fixed-seed, class-balanced sample of 20 real postings from distinct near-duplicate groups (10 genuine, 10 fraudulent). One agent-and-judge evaluation was attempted per posting, with a 150-second whole-case deadline. Failures were retained and score coverage was reported. The sample audits workflow and explanation quality rather than population-level detection accuracy.
