# SoC Slurm smoke test

Run submission commands from the cloned repository root. These scripts do not
configure account/SSH access, install dependencies or download model weights.
Commit/push the local changes and pull them on the cluster first.

Prerequisites:

- Uploaded `data/processed/d1_train.csv`; no reserved test file is needed.
- Working modelling `.venv` and agent/evaluator `agent/.venv` environments.
- Existing Ollama at `$HOME/ollama-local/bin/ollama` and model cache at
  `$HOME/ollama-models`, containing `qwen2.5:14b-instruct`.
- The committed revised comparison reports and current artifact-builder code.

Submit the CPU-only, single classifier fit:

```bash
sbatch cluster/fit_classifier.slurm
```

After it completes successfully, check its `slurm-fit-classifier-JOBID.out` log
and verify `models/agent_lightgbm/pipeline.joblib` and `metadata.json` exist.
Then submit the GPU smoke test:

```bash
sbatch cluster/evaluator_smoke.slurm
```

Monitor with `squeue -u "$USER"`, `tail -f slurm-agent-smoke-JOBID.out`, and
`sacct -j JOBID --format=JobID,State,ExitCode,Elapsed`. The GPU script requests
one `a100-40` GPU in the `gpu` partition, according to the supplied cluster GRES
listing; change resource directives if your account allocation requires it.

The smoke test starts its own localhost Ollama service on the allocated compute
node, checks cached model availability, warms it up, and checks for GPU loading.
It runs the agent and judge sequentially on two synthetic cases and stops only
the server process it started. Port conflicts fail rather than attaching to an
unrelated service. No service runs on the login node. The server log is preserved.

The script explicitly uses the same Qwen2.5 model for agent and judge, overriding
the respective model environment settings for this smoke test. This reduces
storage requirements but introduces self-evaluation bias. The 16,384-token
context is a resource setting, not a guarantee that arbitrary long inputs fit.
Judge timeout is 300 seconds by default; the agent retains its configured
60-second request timeout. Warm-up reduces loading overhead but does not remove
the possibility of agent timeouts on difficult requests.

Outputs are under ignored `evaluator/outputs/slurm_smoke_JOBID/`: `ollama.log`,
`report.json`, `results.jsonl`, `cases.jsonl`, and `review_report.html`. Evaluator
exit code 1 can indicate agent/judge failures or failed hard checks; inspect the
saved report rather than treating it as an infrastructure failure automatically.
Re-running a job uses a new Slurm ID and does not overwrite previous evaluations.

This is a deployment/software smoke test, not a representative fraud benchmark.
The classifier is an integration candidate fitted once on training data, with
the revised experiment's modal setting and median cutoff. Testing that artifact
on its own training rows would not estimate generalisation. Keep the reserved
test set untouched until model, prompts, rules and evaluation policy are frozen.
