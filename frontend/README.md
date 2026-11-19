# Job-ad review frontend

This folder contains the Streamlit user interface. The sibling
[`agent/`](../agent/README.md) package contains the LangChain workflow, model
configuration, tools, validation and local decision storage. The UI imports that
package directly; there is no separate API server or deployment step.

## Install and start

Run from the repository root using Python 3.10+:

```powershell
python -m venv agent/.venv
.\agent\.venv\Scripts\python.exe -m pip install -r frontend/requirements.txt
```

The existing environment stays in `agent/.venv`; there is no need for another
frontend environment. Keep the modelling `.venv` separate. If the environment
and dependencies are already installed, skip these two commands.

For local use, install and keep the Ollama application running, then download:

```powershell
ollama pull qwen3:4b-instruct
```

Save this in `agent/.env` (the project-root `.env` is also supported):

```dotenv
AGENT_BACKEND=ollama_local
OLLAMA_MODEL=qwen3:4b-instruct
```

No real API key is needed for local mode. Start the interface:

```powershell
.\agent\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

Open the local address printed by Streamlit. This starts the app on your
computer; it does not publish a website. Opening it alone makes no model
request. Clicking **Review posting** starts the model and tool calls.
Stop an older Streamlit process with Ctrl+C and restart after configuration or
resource changes. Local processing speed depends on the computer.

## What to enter

Submit one job posting, corresponding to one dataset row:

- Job title and advertisement/description are required.
- Company profile, salary range, requirements and benefits are optional.
- Leave unavailable information blank. Do not invent it.
- Do not include the known `fraudulent` label; keep it separately for evaluation.

The current form accepts manual input and does not automatically load dataset
rows. It is an evidence-review interface, not a general chatbot.

The **demo scenario** controls the classifier's fixed placeholder score. It does
not select a different language model or override the actual submitted wording.
Editing text changes the real risk checks, but does not change that fixed score.

## What the screen shows

| Area | Meaning |
|---|---|
| Model banner | Selected local/cloud model, or explicit offline test mode |
| Classifier score, cutoff and flag | Scenario-selected placeholders; trained classifier integration is pending |
| Rationale and recommendation | LLM explanation with a rejection proposal or escalation for human review |
| Risk findings and context | Exact text, field, source and interpretation: request, negation, warning, uncertainty or an unconfirmed candidate |
| Historical dataset terms | Notebook 00's vocabulary snapshot, displayed separately as background only |
| Limitations and next steps | Missing checks and suggested reviewer follow-up |
| Model calls and tool trace | Inspectable record of requested calls and executed tool outputs |
| Reviewer decision and download | Named human confirmation and a JSON case report |

The real lexicon/context tool checks payment requests and secret banking
passwords or codes. It can find new wording through the LLM as well as rule
matches. A guard keeps explicit warning/negation contradictions uncertain,
including context outside a short model quote. The original model verdict and
application check remain visible. These checks are not validated fraud scores.

If context review fails, matches remain unconfirmed and the interface shows a
warning. A real-model failure never silently switches to a scripted model.
Ordinary job titles, industry words, urgency and a high placeholder score cannot
support rejection. Missing supplied details do not prove an employer lacks them.
Market salaries and employer legitimacy are not checked externally.

## Human confirmation and saved data

Read the quoted context before confirming a decision. The agent cannot remove
an ad or contact an employer. A named reviewer can save a local decision;
records are written only on confirmation to ignored
`agent/runtime/reviews.jsonl`. The JSON download includes the submitted posting,
evidence, model/backend metadata, tool trace and any confirmed reviewer decision.
API keys are excluded.

Local mode sends the posting to the Ollama service on this computer and has no
Ollama cloud API charges. Cloud mode sends it to Ollama's hosted service and
requires an Ollama key and account credits. Direct DeepSeek API integration is
not currently supported; see the [agent configuration guide](../agent/README.md).

## Checks

```powershell
.\agent\.venv\Scripts\python.exe -m unittest discover -s frontend/tests -v
```

The two interface tests use mocked/offline model behavior, so they do not need
Ollama or spend API credits. They check the credit-error display, form submission,
scenario changes and explicit reviewer confirmation. The agent has separate
checks for evidence grounding, context completeness and rejection boundaries.
The combined agent/frontend suites last passed all 43 tests.

The [saved agent comparison](../reports/agent_evaluation/development_comparison.json)
uses synthetic development examples, not unseen real-world validation. See the
[project README](../README.md) for completed ML work and outstanding integration.
