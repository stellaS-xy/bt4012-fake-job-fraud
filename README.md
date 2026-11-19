# Fake Job Posting Fraud Detection — BT4012

This project detects potentially fraudulent job advertisements and helps a human
reviewer understand suspicious wording. One dataset row represents one posting.
The classifier scores an ad; the LLM agent investigates its text and explains
returned evidence. A reviewer makes the final decision.

## Current status

| Component | Implemented | Remaining work |
|---|---|---|
| D1 data exploration | Missingness, duplicate grouping, frozen train/test split, training-only statistical analysis and phrase discovery | Further error analysis and sensitivity to grouping |
| Feature engineering | Raw fields, missingness/contact/salary features, text style, TF-IDF and a learned phrase-score option | Validate which signals generalize to new employers and job types |
| Fair model evaluation | Shared grouped folds; inner unseen-row predictions select settings and the flagging cutoff | Choose and freeze a final model and cutoff |
| Model comparison | Text-only logistic regression, A2 logistic regression, LightGBM and XGBoost; saved reports | Final held-out test evaluation after selection |
| Agent backend | Real LangChain tool calling with local or cloud Ollama; evidence validation, human confirmation and local audit records | Connect the saved trained classifier; its current tool score is a placeholder |
| Risk lexicon | Real payment/secret-credential patterns, LLM context checks, exact quotes and conservative warning/negation checks | Evaluate on representative human-annotated development ads |
| Frontend | Separate Streamlit application in `frontend/` | Further usability evaluation |
| D2 / explanations | D2 is a planned supplementary fraud-only source; the agent explains quoted evidence | D2 analysis if required; classifier-specific explanations such as SHAP |

The reserved D1 test set has not been used for the saved model-comparison run.
Agent development results use invented examples and must not be presented as
fraud-detection accuracy. Model scoring is implemented in the ML pipeline, but
that trained pipeline is **not yet connected to the agent**.

## Start the agent and frontend locally

Run these PowerShell commands from the repository root. Use Python 3.10+ and
keep the agent environment separate from the modelling `.venv`. If `python` is
not recognized, use the full path to your installed Python executable.

```powershell
python -m venv agent/.venv
.\agent\.venv\Scripts\python.exe -m pip install -r frontend/requirements.txt
```

Install the [Ollama Windows application](https://ollama.com/download/windows),
keep it running, and download the local model:

```powershell
ollama pull qwen3:4b-instruct
```

Create `agent/.env` with:

```dotenv
AGENT_BACKEND=ollama_local
OLLAMA_MODEL=qwen3:4b-instruct
```

No real API key is needed in local mode. Start the interface:

```powershell
.\agent\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

Open the local address printed by Streamlit. Enter a posting's title,
description and any supplied company profile, salary, requirements and benefits.
Click **Review posting**. Opening the interface alone does not call the model.
The current input form is manual; it does not automatically import dataset rows.
Never include the known `fraudulent` label in the LLM input.

The demo-scenario selector controls only the **placeholder classifier score**.
The live model and risk checks analyse the submitted text. Results include a
written rationale, quoted findings, context interpretations, limitations and a
recommendation to propose rejection or escalate. A named reviewer may confirm a
local decision and download a JSON report; no ad is removed from another system.

Local mode uses `http://localhost:11434/v1` without Ollama cloud API charges.
Optional `AGENT_BACKEND=ollama` uses Ollama cloud and requires an Ollama key and
account credits. Direct DeepSeek API integration is **not implemented**; changing
an API key alone does not switch providers. See [agent setup](agent/README.md)
and [frontend instructions](frontend/README.md) for details.

## Which lexicon is used?

| Item | Location | Role |
|---|---|---|
| Historical phrase discovery | Notebook 00 → `reports/tables/lexicon_confirmed.csv` | Finds dataset associations using separate training-group discovery and confirmation samples |
| Agent background vocabulary | `agent/resources/dataset_terms.csv` | Frozen copy of 33 historical terms, displayed as background only |
| Risk candidates | `agent/risk_rules.py` | Versioned English payment and secret-credential patterns |
| Context interpretation | `agent/context_review.py` | LLM distinguishes requests, negations, warnings and uncertainty, and can find additional quoted wording |
| Registered agent tool | `agent/tools/lexicon.py`, registered in `agent/tools/__init__.py` | Combines candidates, validated context and background vocabulary |

Notebook 00's discovery code was not replaced by the agent changes. Its exported
terms, such as `data entry` and `oil`, are not proof of fraud and cannot support
rejection. The agent reads a **snapshot**, not the report CSV directly; rerunning
Notebook 00 does not automatically refresh `agent/resources/dataset_terms.csv`.
See the [resource notes](agent/resources/README.md) before refreshing that copy.

Only a cited payment/secret-credential request with completed LLM context review
can support a rejection proposal. If explicit warning or negation wording
contradicts a model verdict, the finding becomes uncertain. Failed checks leave
candidates unconfirmed and require escalation. These controls do not establish
employer legitimacy, market salaries or real-world accuracy.

`agent/dummy_model.py` remains as an explicit **offline test model**. It is used
only with `AGENT_BACKEND=dummy`; normal local/cloud use calls a real model, and
failures never silently switch to the dummy model. This test model is separate
from the still-placeholder classifier tool in `agent/tools/classifier.py`.

## Reproduce the ML work

D1 is the labelled EMSCAD dataset from Kaggle:
`shivamb/real-or-fake-fake-jobposting-prediction` (about 17.9k ads, 4.8% fraud).
Download and extract `fake_job_postings.csv` directly into `data/raw/`.
Notebook 00 currently reads CSV files at that level, not inside `data/raw/d1/`.
Raw and processed data are ignored by Git.

D2, `srisaisuhassanisetty/fake-job-postings`, is a supplementary fraud-only
source. Its analysis is not part of the completed D1 pipeline. Keep any D2 files
separate from the top-level D1 CSV so Notebook 00 does not load the wrong file.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install statsmodels
.\.venv\Scripts\python.exe -m jupyterlab
```

Select the modelling environment as the notebook kernel and follow this order:

| Notebook | Work completed |
|---|---|
| [00: data understanding](notebooks/00_d1_data_understanding_clean.ipynb) | Data checks, related-posting groups, frozen split, training-only exploration and lexicon discovery |
| [01: feature engineering](notebooks/01_d1_feature_engineering_v2.ipynb) | Feature blocks, grouped comparisons and ablations; reusable code in `src/features.py` |
| [02: evaluation setup](notebooks/02_d1_modelling_setup.ipynb) | Shared grouped tests; model settings and cutoffs chosen from inner unseen-row predictions |
| [03: model comparison](notebooks/03_d1_model_comparison.ipynb) | Reads and explains the completed four-model comparison |

Every model uses the same scoring examples across five repeats of five grouped
folds. Three inner grouped folds select settings and a cutoff, using only the
training portion of each outer round. Related ads stay together. Features are
refitted inside each split. The cutoff maximizes F1 on the selected setting's
inner out-of-fold predictions; the scoring rows do not choose it.

The text-only logistic reference uses up to 30,000 TF-IDF terms. A2 logistic
regression, LightGBM and XGBoost share A2 inputs with a 2,000-term limit. Balancing
weights use each fit's training rows. Outer scoring data is not used for early
stopping. The saved comparison reads no reserved test data.

Read the [model report](reports/model_comparison/report.html) without retraining.
Its counts, averaged over repeats, include:

| Model | Fake ads caught / 693 | Genuine ads wrongly flagged / 13,746 | Fake ads found in the 5% review budget | Mean PR-AUC |
|---|---:|---:|---:|---:|
| Text-only logistic regression | 352.0 | 308.4 | 377.6 | 0.595 |
| A2 logistic regression | 376.8 | 443.2 | 370.8 | 0.555 |
| LightGBM A2 | 383.6 | 414.4 | 389.0 | 0.631 |
| XGBoost A2 | 336.6 | 398.0 | 344.8 | 0.566 |

The 5% budget ranks ads by score separately from the learned flagging cutoff.
The report includes per-round results and approximate paired uncertainty
intervals. Higher averages alone do not establish a reliable improvement.
Repeats reuse the same ads and are not independent new datasets.

```powershell
# Rebuild the report from saved completed results; no model training.
.\.venv\Scripts\python.exe -m src.model_comparison --report-only

# Full comparison; can take substantial time.
.\.venv\Scripts\python.exe -u -m src.model_comparison
```

Run plans, settings, scores and fingerprints are in `reports/model_comparison/`.
Row assignments and predictions are under ignored
`data/processed/model_comparison/`. Choose and freeze the final model and cutoff
before enabling the reserved-test section in Notebook 02.

## Agent comparison and checks

The [saved evidence comparison](reports/agent_evaluation/development_comparison.json)
uses 12 synthetic development examples, evaluated twice with local
`qwen3:4b-instruct`. Across each method's 24 evaluations, there were 12 actual
risky-request occurrences and 12 negatives:

| Method | Requests found / 12 | False alarms | Findings with wrong field/category |
|---|---:|---:|---:|
| Rules | 10 | 0 | 0 |
| LLM without rule candidates | 12 | 0 | 4 |
| Rules + LLM | 12 | 0 | 0 |

These examples were used during development, so they are regression checks,
not unseen validation data. The paired improvement interval includes zero.
Both LLM methods use exact-quote validation and the same contradiction guard.
More human-annotated development ads are needed before claiming an improvement.

```powershell
# Software checks; no live model calls or API credits.
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\agent\.venv\Scripts\python.exe -m unittest discover -s agent/tests -v
.\agent\.venv\Scripts\python.exe -m unittest discover -s frontend/tests -v

# Evidence baseline without an LLM.
.\agent\.venv\Scripts\python.exe -m agent.evaluate_lexicon --mode rules

# Live comparison using the configured model; may take several minutes locally.
.\agent\.venv\Scripts\python.exe -m agent.evaluate_lexicon --mode all --repeats 2
```

The last verified agent/frontend suites contained 43 passing tests. That count
excludes the separate ML tests and is not a measure of fraud-detection accuracy.
Evaluation options, custom JSONL annotations and result locations are documented
in [agent/README.md](agent/README.md).

## Repository layout

```text
notebooks/                    D1 exploration, features, evaluation and comparison
src/                          Reusable ML features and evaluation code
tests/                        ML evaluation and comparison checks
agent/                        LangChain backend, tools, config and agent checks
frontend/                     Streamlit UI, UI dependencies and interface checks
reports/tables/               Saved statistical and feature-analysis tables
reports/model_comparison/     Completed ML comparison reports and scores
reports/agent_evaluation/     Synthetic evidence-extraction comparison
agent/runtime/                Local decisions and ad-hoc reports (ignored)
data/raw/, data/processed/    Original and derived data (ignored)
```

## Team workflow and acknowledgement

Work on a branch, commit small changes and have a teammate review before merging
into `main`. Keep data, API keys, `.env`, virtual environments and large model
artifacts out of Git. The ML seed is 42; model-provider responses need not be
identical between runs, even with temperature zero.

OpenAI Codex assisted with evaluation code, agent/frontend implementation,
tests and documentation. The team should review this work and complete the
course's AI/external-code declaration with any additional tools and sources used.
