# Fake Job Posting Fraud Detection

BT4012 research prototype for screening job advertisements. A saved classifier
scores the posting; an LLM agent calls that classifier and a lexicon tool, then
explains the returned evidence. The system recommends `pass_no_review_needed`
only when both checks pass; otherwise it recommends `review_needed`. It does
not approve, reject, contact employers or execute business actions.

The final classifier is **Ensemble v1**, a fixed 50/50 average of sigmoid-calibrated
MLP and Extra Trees scores using A2 features and up to 30,000 word TF-IDF terms.
Its development-selected threshold is **0.20923746948802296**. The classifier,
lexicon and agent roles are distinct: a classifier flag is not an explanation,
and a review recommendation is not a fraud label.

## Start here

| Goal | Guide |
|---|---|
| Understand completed results | [Results index](reports/README.md) |
| Run the saved system on the cluster without training | [Deployment guide](docs/DEPLOYMENT.md) |
| Choose the correct Python environment | [Dependencies](docs/DEPENDENCIES.md) |
| Configure agent, judge and model paths | [Configuration reference](docs/CONFIGURATION.md) |
| Understand each experiment and its split | [Notebook index](notebooks/README.md) |
| Inspect the final agent evaluation and errors | [100-case report](reports/agent_evaluation/ensemble_v1_100cases/README.md) |
| Run software checks without models or LLMs | [Offline checks](docs/OFFLINE_CHECKS.md) |

## Completed evaluation

| Evaluation | Result | Interpretation |
|---|---|---|
| Notebook 13 development comparison | Ensemble mean AP **0.6891**, fold SD **0.0566**, 25 grouped outer folds | Repeated folds share observations. The corrected paired interval against the fresh LightGBM replay includes zero; superiority is not established |
| Fixed existing test split | AP **0.7848**, ROC-AUC **0.9503**, precision **0.6772**, recall **0.7399**, F1 **0.7072** | 3,441 postings, 173 frauds, 885 groups; threshold fixed from development data |
| Top 5% classifier review budget | 124 frauds among 172 postings; precision **0.7209**, recall **0.7168** | A ranking budget, separate from the saved classification threshold |
| Final agent quality audit | **100/100** agent and judge completions; **95/100** pass all code checks | 50 genuine and 50 fraudulent postings from 100 groups; five incomplete context reviews and four cases flagged by the judge for unsupported claims |

The test AP group-bootstrap 95% interval is **0.6236–0.8962**. Current train/test
posting and group separation passed audit, but historical project-wide test
exposure remains unverified. These results describe the audited existing split,
not a proven never-accessed test. The balanced agent audit is not representative
of natural fraud prevalence. Agent and judge both used Qwen2.5 14B, so judge
ratings are automated opinions with possible shared errors. See the linked
reports for denominators, sources and limitations.

## System components

```text
Posting → LLM agent → score_posting → saved Ensemble v1 score and threshold
                    → match_lexicon → rules + internal LLM context review
                                    + historical background observations
        → code-enforced recommendation policy → explanation and evidence
```

The context review distinguishes employer requests, warnings, negations and
uncertain wording. Historical dataset phrases are background associations, not
standalone fraud rules. Missing optional text alone does not fail the lexicon.
The final MLP requires explicit 0/1 values for `telecommuting`,
`has_company_logo` and `has_questions`; unknown values are not imputed as zero.
Incomplete checks require review. Employer legitimacy and market salary are not
verified.

Exactly two tools are exposed to the agent. The context-review LLM call is
internal to `match_lexicon`. The separate evaluator applies deterministic checks
and an LLM judge; gold fraud labels do not enter agent or judge inputs.

## Repository layout

| Path | Contents |
|---|---|
| `notebooks/` | Experiment interfaces and saved-results viewer; historical filenames preserved |
| `src/` | Feature construction, experiment helpers, training/export and inference adapters |
| `agent/core/` | LLM orchestration, schemas, validation and recommendation policy |
| `agent/tools/` | Classifier subprocess adapter, risk rules, lexicon and context review |
| `agent/resources/` | Frozen historical vocabulary and synthetic examples |
| `evaluator/` | Input loading, code checks, judge, scoring, reports and bounded audit runner |
| `frontend/` | Optional Streamlit interface |
| `cluster/` | Slurm launchers and detailed experiment/deployment instructions |
| `tests/` and component test folders | Offline software checks and test fixtures |
| `reports/` | Curated aggregate results, provenance and interpretation |
| `data/`, `models/`, `evaluator/outputs/` | Local datasets, weights and detailed run records; generally ignored by Git |

Model ZIPs belong in a GitHub Release; source and aggregate results belong in
Git. [Deployment instructions](docs/DEPLOYMENT.md) explain asset names, checksum
verification and the existing cluster artifact. A ZIP avoids retraining but
still needs the inference dependencies wherever predictions run. No Mac model
installation is needed when the complete system runs on the cluster.

## Research and reproduction

D1 is EMSCAD from Kaggle, `shivamb/real-or-fake-fake-jobposting-prediction`.
Notebook 00 expects `fake_job_postings.csv` directly under `data/raw/`.
Original and derived datasets are not distributed here. D2 is a supplementary
fraud-only source and is not part of the completed D1 evaluation.

The [notebook index](notebooks/README.md) distinguishes original, revised and
recovered split protocols. Do not treat different experiment directories as a
single matched leaderboard or compare a single fold to a 25-fold average.
Saved reports can be read without rerunning experiments. Reproduction commands
are documented for provenance; no further training or LLM evaluation is needed
to inspect the completed project.

## Software checks and team workflow

[GitHub Actions](.github/workflows/offline-checks.yml) runs a selected offline
suite and syntax checks without TensorFlow, model weights, datasets or live LLM
calls. It does not certify real cluster inference. Local commands and coverage
boundaries are in the [offline checks guide](docs/OFFLINE_CHECKS.md).

Keep `.env`, credentials, datasets, virtual environments and model binaries out
of commits. Preserve frozen experiment sources and manifests: changing paths or
code can invalidate their fingerprint checks. The numerical ML seed is normally
42; provider responses need not be identical even with temperature zero.

OpenAI Codex assisted with implementation, tests and documentation. The team
should review the work and complete the course's AI/external-code declaration.
