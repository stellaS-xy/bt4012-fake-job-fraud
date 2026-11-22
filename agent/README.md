# Job-ad review agent

A real Ollama model, running locally or in the cloud, uses LangChain to call
tools and write a review. This folder sits beside `notebooks/` and has its own
Python environment.

For the final completed run, see the [100-case evaluation and error analysis](../reports/agent_evaluation/ensemble_v1_100cases/README.md). Environment selection and supported variables are documented in [dependencies](../docs/DEPENDENCIES.md) and [configuration](../docs/CONFIGURATION.md).

The Streamlit frontend is in the sibling `frontend/` folder. See the
[frontend instructions](../frontend/README.md) for interface setup and the
[project README](../README.md) for the completed ML work. This folder contains
the agent logic; it can run without starting the interface.

The language model and tool-calling loop are live. Exactly two tools are exposed:
`score_posting` runs the saved calibrated MLP + Extra Trees v1 ensemble, and `match_lexicon` runs
risk patterns plus LLM context review. Missing-field and urgency observations
are internal parts of the lexicon result, not a third agent tool. The classifier
uses saved sigmoid calibration, which does not guarantee real-world probability accuracy; explanation quality still needs
representative development evaluation.

## Folder organization

The maintained code and resources are grouped into four folders:

```text
agent/
  core/                 Workflow, model setup, schemas, validation and recommendation policy
    config.py           Reads agent/.env; keeps runtime paths at the agent root
    models.py           Constructs the real local/cloud model
    runner.py           Runs the LangChain agent
  tools/                Classifier, lexicon, posting checks, risk rules and LLM context
  resources/            Historical terms, synthetic examples and demo fixtures
  tests/                Unit tests, test-only chat fixtures and development comparison CLI
  __main__.py           Entry point for python -m agent
  __init__.py
  README.md
  requirements.txt
  requirements-lock.txt
  .env                  Local settings and keys; ignored by Git
```

`.venv/`, `runtime/` and `__pycache__/` contain the local Python environment,
legacy generated records and interpreter caches. They remain ignored by Git.
Configuration stays in `agent/.env`, and the frontend launch command is unchanged.
The comparison CLI is now `python -m agent.tests.evaluate_lexicon`.

## First-time setup

Run from the repository root with Python 3.10+:

```powershell
python -m venv agent/.venv
.\agent\.venv\Scripts\python.exe -m pip install -r frontend/requirements.txt
```

This installs both the agent and interface dependencies. For backend-only use,
install `agent/requirements.txt` instead. The separate modelling environment is
`.venv-mlp/` for ensemble v1; do not mix the two sets of dependencies. If `python` is not recognized,
use your installed Python executable's full path.

## Free local use on Windows

Install the Ollama Windows app and keep it running. Download the local model:

```powershell
ollama pull qwen3:4b-instruct
```

Set these values in `agent/.env`:

```dotenv
AGENT_BACKEND=ollama_local
OLLAMA_MODEL=qwen3:4b-instruct
```

No cloud key is needed for local mode. Saved cloud credentials are not sent to
the local service. Requests go to `http://localhost:11434/v1` and use your
computer's resources with no Ollama cloud API charges. Do not select a cloud
model. Start the interface with the command below and click **Review posting**.
Restart Streamlit after changing settings. Local inference speed depends on your
computer. The classifier uses the saved ensemble artifact and its matching inference environment.

```powershell
.\agent\.venv\Scripts\python.exe -m streamlit run frontend/app.py --browser.gatherUsageStats false
```

For a real local terminal review:

```powershell
.\agent\.venv\Scripts\python.exe -m agent --backend ollama_local --scenario suspicious
```

The explicit local override selects the default local model. To use a custom
`OLLAMA_MODEL` from your `.env`, omit `--backend` and let configuration select it.

| Backend | Language model | Key needed? |
|---|---|---|
| `ollama_local` | Downloaded model at `localhost:11434` | No |
| `ollama` | Real model hosted by Ollama cloud | Ollama API key and sufficient account credits |

The example configuration uses local mode. Without configuration, the code's
fallback default is free local `ollama_local`. Choose a backend
explicitly in `.env`. Direct DeepSeek API support has only been discussed; it
has not been implemented. The cloud example below uses a DeepSeek model hosted
by **Ollama**, not the direct DeepSeek service.

## Optional paid cloud use on Windows

Save these settings in `agent/.env` (which Git ignores):

```dotenv
AGENT_BACKEND=ollama
OLLAMA_MODEL=deepseek-v4.1-flash
OLLAMA_API_KEY=<your Ollama API key>
```

The application also reads the project-root `.env`. Values in `agent/.env` take
priority, followed by process environment variables as the highest priority.
Use an Ollama key, not an OpenAI key. Requests go to `https://ollama.com/v1`;
no local Ollama server is required. Restart Streamlit after changing configuration.
The account must have access and sufficient credits for the selected model.

From the repository root:

```powershell
.\agent\.venv\Scripts\python.exe -m streamlit run frontend/app.py --browser.gatherUsageStats false
```

Or run a synthetic example in the terminal:

```powershell
.\agent\.venv\Scripts\python.exe -m agent --scenario suspicious
```

The CLI prints a JSON case report and exits with code 1 if the review fails.
An API failure is shown as an incomplete review; it never silently switches to
the scripted model. In particular, HTTP 402 means Ollama requires credits or a
subscription. Real-model failures remain visible; no simulated model is substituted.

To install on another computer, create a separate environment with Python 3.10+
and install the dependencies. Replace `python` with your installed interpreter
if needed. Keep the project's modelling environment separate.

```powershell
python -m venv agent/.venv
.\agent\.venv\Scripts\python.exe -m pip install -r agent/requirements.txt
```

For the interface as well, install `frontend/requirements.txt` instead; it
includes the agent dependencies and Streamlit. `requirements-lock.txt` records
the combined agent/frontend environment tested with Python 3.10.11. Use it to
reproduce that full environment. Newer Python versions may resolve different
versions from `requirements.txt`.

## How a review works

1. Submit a posting and select a demo scenario. One submission corresponds to
   one dataset row, using title, description, company profile, salary,
   requirements and benefits. The known `fraudulent` label is never an input.
   The form is manual; dataset-row import is available through the evaluator CLI.
   The scenario only fills an example; the real classifier computes its score.
2. LangChain sends the posting to the configured local or cloud Ollama model.
3. The model calls `score_posting` and `match_lexicon`. Both must complete.
   Basic missing-field/urgency observations are folded into the lexicon output.
   `match_lexicon` first finds payment/secret-credential candidates, then makes
   a separate structured LLM call to interpret each one and inspect all supplied
   text fields for wording missed by the rules. Each field and candidate must
   be accounted for; added findings must quote an exact passage in their field.
4. The model reads the tool results and returns a structured assessment. Local
   mode uses Ollama's JSON schema output; cloud mode uses an `Assessment`
   function call because cloud JSON schema output is not supported.
5. Validation checks scores, evidence IDs, quoted text and the two-tool policy.
6. Return a recommendation: **pass - no review needed** only if both tools pass; otherwise **review needed**, with reasons. No confirmation or escalation action is offered.

Normally there are about three model requests: choosing tools, the separate
context review, and the final assessment. Sequential tool calls and correction
attempts can increase this. The interface and JSON report distinguish model-issued calls from executed tool
results. They include the backend, model name, provider call IDs, and review
status. Each agent pass is limited to four orchestration-model calls and eight business-tool
calls. A missing, malformed or invalid final assessment gets one direct
structured-output recovery attempt using completed tool evidence. The business
tools are not rerun. Fixed IDs/scores and allowed recommendations are constrained
in the recovery schema, then the result is validated again. A second failure
returns a `review_needed` recommendation. Exports record `recovery_attempted` and `failure_stage`;
an invalid final answer has the safe error code `assessment_invalid`. Provider
requests use a 60-second timeout. Cloud requests may retry once; local requests
do not automatically repeat a slow inference. Generated final answers have
bounded lengths, with at most three limitations and three next steps. The context
review inside `match_lexicon` adds one structured model call and at most one
validation correction. Its result is cached for the case, including failures,
so repeated tool calls do not repeat that review. Temperature is set to zero;
this does not guarantee identical outputs across runs.
Context reviews also reject a request label that contradicts a warning,
negation or an explicit 'not a request' explanation. When all candidates have
protected warning/negation context, the response schema disallows request.
Uncertainty remains visible rather than being promoted to a confirmed request.

The two recommendations are `pass_no_review_needed` and `review_needed`.
The real classifier passes when its valid score is below its cutoff. The lexicon
passes when context review completes without risky requests, uncertain findings
or unreviewed candidates. Both must pass to recommend no review. A flag, missing
or failed tool, test-fixture score or incomplete context check requires review,
with reasons from the tools. Confirmed warnings/negations and historical terms
or missing optional fields alone do not fail the lexicon check. This policy is
implemented in `core/policy.py` and independently enforced during validation.
These are recommendations only: no rejection, human escalation or confirmation
action is executed. A pass does not verify an employer or guarantee safety.
Schema version 5 records this contract; saved older cases retain their original
meaning and must not be mixed into a new-policy evaluation.
An application guard also checks the containing sentence for explicit negation
or warning language, including context outside a short LLM quote. If that
contradicts the model's request verdict, the finding becomes uncertain. The
original model interpretation and guard reason stay in the evidence record.
This conservative guard can also suppress a real request in complicated text;
it is not a full language-understanding system.
Salary comparisons and employer legitimacy are unverified; missing supplied
information does not establish that an employer is fraudulent. Quoted instructions
inside a posting are treated as evidence, not instructions for the agent.

The supplied posting is sent to the local service in local mode and to Ollama's
cloud service in cloud mode. Optional LangSmith
uploads are disabled. The frontend can download a case report; it no longer
offers reviewer decisions or writes confirmation records. API keys are excluded from case exports and
application error messages.

## Notebook 00 lexicon versus the agent's checks

Notebook 00's discovery/confirmation analysis exports
`reports/tables/lexicon_confirmed.csv`. Its discovery code was not changed by
the agent implementation. The 33 terms currently in
[`resources/dataset_terms.csv`](resources/dataset_terms.csv) were copied from
that output and are matched as background only. The agent does not read the
report CSV at runtime and does not automatically refresh the copy after a
notebook rerun. Ordinary job and industry terms do not fail a check by themselves.

Actual risky-request candidates come from [`tools/risk_rules.py`](tools/risk_rules.py),
not from promoting those historical terms into fraud rules. The LLM context
review can also add exact-quoted findings the patterns missed. Separately,
`src/features.py` contains the ML pipeline's fold-trained `FraudPhraseScore`
feature; the agent does not load that fitted feature transformer or claim its
matches explain the classifier's score.

See [resource provenance and refresh instructions](resources/README.md) before
updating the historical terms. The main tool implementation is
[`tools/lexicon.py`](tools/lexicon.py); LangChain registration is in
[`tools/__init__.py`](tools/__init__.py).

## Files and responsibilities

| File | Responsibility |
|---|---|
| `../frontend/app.py` | Streamlit inputs, recommendation results and JSON download |
| `__main__.py` | Command-line entry point |
| `core/config.py` | Environment loading and configuration validation |
| `core/models.py` | Real Ollama model construction through LangChain ChatOpenAI |
| `core/runner.py` | Agent orchestration, validation and safe failure handling |
| `core/assessment.py` | One constrained final-answer recovery using completed tool outputs |
| `core/middleware.py` | Require both planned business tools before final assessment |
| `core/tracing.py` | Capture actual model-issued calls and match execution IDs |
| `core/errors.py` | Safe authentication, credits, connection and other error messages |
| `core/prompts.py` | Instructions and evidence boundaries |
| `tools/risk_rules.py` | Versioned English payment and secret-credential candidate patterns |
| `tools/context_review.py` | Structured LLM context review, complete coverage and exact quotes |
| `resources/` | Frozen background-only dataset terms and synthetic development examples |
| `tests/evaluate_lexicon.py` | Rules/LLM/combined comparisons on identical annotated examples |
| `core/schemas.py` | Posting, evidence and recommendation data contracts |
| `tools/__init__.py` | LangChain tool registration and execution traces |
| `tools/classifier.py` | Saved ensemble/legacy LightGBM subprocess adapter; no simulated classifier option |
| `tools/lexicon.py` | Risk candidates, LLM interpretation and background-only dataset terms |
| `tools/posting.py` | Internal missing-field/urgency checks included in the lexicon output |
| `core/validation.py` | Score consistency, required checks and evidence validation |
| `core/policy.py` | Both-tool pass checks and review reasons; no business actions |
| `resources/fixtures.py` | Example input postings; no fixed scores |
| `tests/chat_model_fixture.py` | Scripted chat responses imported only by automated tests |
| `tests/` | Agent workflow and mocked provider protocol checks |
| `../frontend/tests/` | Streamlit interface checks |

## Checks and offline tests

```powershell
.\agent\.venv\Scripts\python.exe -m unittest discover -s agent/tests -v
.\agent\.venv\Scripts\python.exe -m unittest discover -s frontend/tests -v
.\agent\.venv\Scripts\python.exe -m pip check
```

Tests do not spend API credits. Provider tests run the real LangChain/SDK loop
against mocked HTTP responses, checking tool calls, structured completion and
failure handling. These verify software behavior, not the model's explanation
quality or the classifier's accuracy.

Automated tests mock model and classifier boundaries using
`tests/chat_model_fixture.py` and `tests/runtime_fixtures.py`. Application code
never imports these files. There is no simulated backend or classifier CLI/UI
option. The normal runtime only supports `ollama_local` and `ollama`, and always
uses real saved classifier inference. Missing artifacts and failed requests remain
visible failures rather than silently substituting a score or explanation.

`ReviewCase.model_backend` and `model_name` are required provenance fields,
recording where the LLM ran and the exact model selected. They have no fake
defaults. `uses_test_fixtures` in assessments and `is_test_fixture` in tool
outputs explicitly distinguish mocked test data from real inference; normal
runs set these to false. They replace the earlier ambiguous provenance names.
Schema 5 reflects that contract change. Earlier saved reports are historical
snapshots; their input/output fields and source hashes have not been rewritten.

## Final ensemble v1 artifact

The default backend is `ensemble`. It loads the released calibrated 50/50 MLP +
Extra Trees model from `models/final_ensemble_v1`, using `.venv-mlp/bin/python`
(on Linux), separately from the agent environment. No fitting or dataset access
occurs during inference. The saved threshold is 0.20923746948802296.

For the cluster setup, verified ZIP extraction and Slurm commands, follow
[the ensemble deployment guide](../cluster/ENSEMBLE_AGENT_README.md). Reuse the
existing cluster environment; no model execution or dependency installation is
required on your Mac. Override `CLASSIFIER_ARTIFACT` and `CLASSIFIER_PYTHON` for
other deployment paths. The scripts explicitly select `CLASSIFIER_BACKEND=ensemble`.

The inference adapter verifies model file hashes, the preprocessing source and
exported runtime versions. It rejects missing binary inputs (`telecommuting`,
`has_company_logo`, `has_questions`) rather than assuming unknown means zero.
Incomplete classifier checks require review. Other optional fields keep their
fitted missing-value handling. Scores and cutoffs are copied exactly into the
agent output; calibration does not guarantee probability accuracy in deployment.

Legacy LightGBM support is retained via an explicit `CLASSIFIER_BACKEND=lightgbm`
plus its corresponding Python environment and artifact. It is not an automatic
fallback when the ensemble is unavailable.

The historical dataset terms are loaded from a versioned resource as background only.
Preserve posting IDs, quote validation and context provenance when extending
the real lexicon/context checks. Review the posting inspection rules as well.

Do not include synthetic scores in the project's model comparison metrics.
Evaluate explanation quality separately from classification. Normal inference
does not train models; the explicit artifact-fit command uses training data only.
The reserved test set is not evaluated as part of this integration.

## Compare the evidence tools

For full-agent recommendation and explanation scoring, use the separate
[`evaluator/`](../evaluator/README.md) package. It applies code checks plus a
configurable local/cloud LLM judge to dataset rows or saved agent outputs, keeps
gold annotations hidden and reports judge failures explicitly. It does not need
human action annotations. The comparison below instead tests risky-request
extraction against annotated spans; the two evaluations serve different roles.

Run the deterministic baseline without a model:

```powershell
.\agent\.venv\Scripts\python.exe -m agent.tests.evaluate_lexicon --mode rules
```

Compare rules alone, the configured LLM alone, and both together:

```powershell
.\agent\.venv\Scripts\python.exe -m agent.tests.evaluate_lexicon --mode all --repeats 3
```

The combined method supplies rule candidates to the same context-review prompt;
the LLM-only method gets the posting without those candidates. Both can find
additional exact-quoted requests. Neither receives the gold annotations.
Every method uses the same cases on every repeat. The JSON report includes an
input hash, versions, per-case predictions, missed requests, false alarms,
unsupported findings, failed reviews and paired accuracy intervals. Bootstrap
sampling keeps all repeats of each case together; repeats are not counted as
new independent examples. Failures are abstentions, never correct negatives.
Unsupported findings have no gold request in their field/category. A separate
count tracks quotes that do not include the annotated gold span; short but
valid quotes may need manual adjudication rather than being called fabricated.
The LLM and combined methods both use the same quote validation and contradiction
guard; "LLM alone" means no rule candidates are supplied, not unvalidated output.

The included 12 cases are synthetic regression examples, not a representative
fraud benchmark. Use `--cases path/to/annotated_development.jsonl` for your own
annotated development set. `--max-cases 6` limits a smoke run. Results are saved
to ignored `agent/runtime/lexicon_evaluation.json` by default; use `--output` to
choose another location. LLM comparisons use your configured backend, so local
mode is free of cloud API charges and cloud mode incurs provider usage.

The saved [development comparison](../reports/agent_evaluation/development_comparison.json)
uses local `qwen3:4b-instruct` on all 12 synthetic examples for two rounds.
Across the 24 evaluations per method (12 request-positive, 12 negative), rules
caught 10 requests; the LLM without candidate hints and the combined method
each caught 12, with no false alarms. The LLM without hints mislabeled four
credential findings as payment requests; the combined method had no such
field/category errors. No reviews failed. These are development examples used
while building the feature, not unseen validation data. The paired interval for
combined-minus-rules accuracy includes zero, so the result does not establish
a reliable improvement outside these examples.
