# Job-ad review agent

A real Ollama model, running locally or in the cloud, uses LangChain to call
tools and write a review. This folder sits beside `notebooks/` and has its own
Python environment.

The Streamlit frontend is in the sibling `frontend/` folder. See the
[frontend instructions](../frontend/README.md) for interface setup and the
[project README](../README.md) for the completed ML work. This folder contains
the agent logic; it can run without starting the interface.

The language model and tool-calling loop are live. The classifier still returns
a scenario-selected placeholder score. The lexicon tool now performs real
versioned risk-pattern matching and LLM context review, with exact quote and
coverage validation. The posting inspector checks missing fields and urgency.
These checks are not validated fraud predictions.

## First-time setup

Run from the repository root with Python 3.10+:

```powershell
python -m venv agent/.venv
.\agent\.venv\Scripts\python.exe -m pip install -r frontend/requirements.txt
```

This installs both the agent and interface dependencies. For backend-only use,
install `agent/requirements.txt` instead. The separate modelling environment is
`.venv/`; do not mix the two sets of dependencies. If `python` is not recognized,
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
computer. The classifier still returns a placeholder score.

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
| `dummy` | Scripted offline test model | No; context LLM is not run |

The example configuration uses local mode. Without configuration, the code's
fallback default is cloud `ollama` and it requires a key. Choose a backend
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
subscription, not that you should change to dummy mode.

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
   The form is manual; dataset-row import is not implemented. The scenario controls only the
   placeholder classifier score; editing the posting does not retrain a model.
2. LangChain sends the posting to the configured local or cloud Ollama model.
3. The model chooses tool calls to `score_posting`, `match_lexicon`, and
   `inspect_posting`. The application requires all three checks before completion.
   `match_lexicon` first finds payment/secret-credential candidates, then makes
   a separate structured LLM call to interpret each one and inspect all supplied
   text fields for wording missed by the rules. Each field and candidate must
   be accounted for; added findings must quote an exact passage in their field.
4. The model reads the tool results and returns a structured assessment. Local
   mode uses Ollama's JSON schema output; cloud mode uses an `Assessment`
   function call because cloud JSON schema output is not supported.
5. Validation checks scores, evidence IDs, quoted text, and rejection boundaries.
6. A human can confirm and save a local reviewer decision.

Normally there are about three model requests: choosing tools, the separate
context review, and the final assessment. Sequential tool calls and correction
attempts can increase this. The interface and JSON report distinguish model-issued calls from executed tool
results. They include the backend, model name, provider call IDs, and review
status. Each agent pass is limited to six orchestration-model calls and eight business-tool
calls. A completed assessment that fails evidence validation gets one correction
pass using the existing history; a second failure escalates to a human. Provider
retries are limited to one per request, with a 60-second timeout. The context
review inside `match_lexicon` adds one structured model call and at most one
validation correction. Its result is cached for the case, including failures,
so repeated tool calls do not repeat that review. Temperature is set to zero;
this does not guarantee identical outputs across runs.

The agent can propose rejection or escalate for review. It cannot remove an ad.
Rejection proposals require a cited request finding reviewed by the LLM.
Unconfirmed candidates, negations, warnings, uncertainty, ordinary job/industry
terms and scores cannot support a proposal. Failed context reviews leave the
candidates visible and require escalation. Even an LLM-reviewed request can be
misinterpreted and still needs a human to check its quoted context.
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
uploads are disabled. Local review decisions are saved only when confirmed, in
ignored `agent/runtime/reviews.jsonl`. API keys are excluded from case exports and
application error messages.

## Notebook 00 lexicon versus the agent's checks

Notebook 00's discovery/confirmation analysis exports
`reports/tables/lexicon_confirmed.csv`. Its discovery code was not changed by
the agent implementation. The 33 terms currently in
[`resources/dataset_terms.csv`](resources/dataset_terms.csv) were copied from
that output and are matched as background only. The agent does not read the
report CSV at runtime and does not automatically refresh the copy after a
notebook rerun. Ordinary job and industry terms cannot support rejection.

Actual risky-request candidates come from [`risk_rules.py`](risk_rules.py),
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
| `../frontend/app.py` | Streamlit inputs, results, reviewer confirmation and JSON download |
| `__main__.py` | Command-line entry point |
| `config.py` | Environment loading and configuration validation |
| `models.py` | Real Ollama model construction through LangChain ChatOpenAI |
| `runner.py` | Agent orchestration, validation and safe failure handling |
| `middleware.py` | Require the three business checks before final assessment |
| `tracing.py` | Capture actual model-issued calls and match execution IDs |
| `errors.py` | Safe authentication, credits, connection and other error messages |
| `prompts.py` | Instructions and evidence boundaries |
| `risk_rules.py` | Versioned English payment and secret-credential candidate patterns |
| `context_review.py` | Structured LLM context review, complete coverage and exact quotes |
| `resources/` | Frozen background-only dataset terms and synthetic development examples |
| `evaluate_lexicon.py` | Rules/LLM/combined comparisons on identical annotated examples |
| `schemas.py` | Posting, evidence, assessment and reviewer data contracts |
| `tools/__init__.py` | LangChain tool registration and execution traces |
| `tools/classifier.py` | Placeholder classifier adapter |
| `tools/lexicon.py` | Risk candidates, LLM interpretation and background-only dataset terms |
| `tools/posting.py` | Local demonstration inspection rules |
| `validation.py` | Score consistency, required checks and evidence validation |
| `reviewer.py`, `audit.py` | Human confirmation and local decision records |
| `fixtures.py` | Synthetic examples and fixed placeholder scores |
| `dummy_model.py` | Scripted model used only for explicit offline tests |
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

An explicit offline demonstration is available without a key:

```powershell
.\agent\.venv\Scripts\python.exe -m agent --backend dummy --scenario suspicious
```

`dummy_model.py` stays so automated tests can check the tool loop, form, failure
handling and review confirmation without a running LLM or API charges. It runs
only when `dummy` is explicitly selected. It is never a silent fallback for a
failed real-model request. In dummy mode, rule matching still runs, but context
is unconfirmed and cannot justify an automated rejection proposal.

The test language model and the classifier stub are separate. Even in real
local/cloud mode, `tools/classifier.py` still returns a scenario-selected
synthetic score. `tool_results_are_placeholders=true` remains in exported
assessments because of that classifier placeholder; individual evidence records
identify the real rule/context sources and review status.

## Connect the trained project tools later

Replace the adapter in `tools/classifier.py` with inference using the frozen
classifier and its saved preprocessing, cutoff and model version. The historical
dataset terms are now loaded from a versioned resource as background only.
Preserve posting IDs, quote validation and context provenance when extending
the real lexicon/context checks. Review the posting inspection rules as well.

Do not include synthetic scores in the project's model comparison metrics.
Evaluate explanation quality separately after connecting and validating the real
tools. This agent does not train models or evaluate the reserved test dataset.

## Compare the evidence tools

Run the deterministic baseline without a model:

```powershell
.\agent\.venv\Scripts\python.exe -m agent.evaluate_lexicon --mode rules
```

Compare rules alone, the configured LLM alone, and both together:

```powershell
.\agent\.venv\Scripts\python.exe -m agent.evaluate_lexicon --mode all --repeats 3
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
