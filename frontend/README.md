# Job-ad review frontend

This folder contains the Streamlit user interface. The sibling
[`agent/`](../agent/README.md) package contains the LangChain workflow, model
configuration, tools, validation and recommendation policy. The UI imports that
package directly; there is no separate API server or deployment step.

The backend is organized into `agent/core/` (workflow and shared contracts),
`agent/tools/` (risk checks), `agent/resources/` (vocabularies and examples), and
`agent/tests/` (tests and development evaluation). The UI imports the workflow
from `agent.core.runner`; settings still belong in `agent/.env`.

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

The example selector fills a posting. In normal local/cloud mode, LightGBM
computes a real score from the submitted fields. The optional classifier fields
include location, department, job categories and unknown/yes/no binary fields.
Leave unknown values unknown; the tool lists omitted inputs. Scores always come from the real saved classifier.

## What the screen shows

| Area | Meaning |
|---|---|
| Model banner | Selected local/cloud model, or explicit offline test mode |
| Classifier score, cutoff and flag | Real LightGBM score and saved integration cutoff; not a calibrated probability |
| Rationale and recommendation | LLM explanation: pass - no review needed, or review needed with reasons |
| Risk findings and context | Exact text, field, source and interpretation: request, negation, warning, uncertainty or an unconfirmed candidate |
| Historical dataset terms | Notebook 00's vocabulary snapshot, displayed separately as background only |
| Limitations and next steps | Limitations and suggested ways to resolve failed checks |
| Model calls and tool trace | Inspectable record of requested calls and executed tool outputs |
| Download | JSON report with the posting, recommendation and tool results |

The real lexicon/context tool checks payment requests and secret banking
passwords or codes. It can find new wording through the LLM as well as rule
matches. A guard keeps explicit warning/negation contradictions uncertain,
including context outside a short model quote. The original model verdict and
application check remain visible. These checks are not validated fraud scores.

If context review fails, matches remain unconfirmed and the interface shows a
warning. A real-model failure never silently switches to a scripted model.
Ordinary job titles, industry words, urgency and missing optional fields alone
do not fail a check. Placeholder scores cannot pass. Missing supplied details do not prove an employer lacks them.
Market salaries and employer legitimacy are not checked externally.

## Recommendations and downloads

Both checks must pass to recommend **pass - no review needed**. The real
classifier must score below its cutoff, and the lexicon context review must
complete without risky or unresolved findings. Otherwise the recommendation is
**review needed**, with reasons. All outputs are recommendations only; there
are no approval/rejection, escalation or reviewer-confirmation controls.
A pass does not verify an employer or establish that a posting is safe.

The JSON download contains the original posting, recommendation, evidence,
model/backend metadata and both tool traces. API keys are excluded. The UI
no longer saves named reviewer decisions. Older downloads use an earlier policy;
schema version 5 reports must be evaluated separately.

Downloaded case JSON can be evaluated without rerunning the agent:

```powershell
.\agent\.venv\Scripts\python.exe -m evaluator --cases path/to/review.json
```

The separate [evaluator guide](../evaluator/README.md) explains the code checks,
LLM judge and dataset-label comparisons. No judge is called just by opening this
interface; evaluation runs as a separate command.

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
scenario changes and absence of reviewer confirmation controls. The agent has separate
checks for evidence grounding, context completeness and recommendation policy.
The combined agent/frontend suites last passed all 49 tests.

The [saved agent comparison](../reports/agent_evaluation/development_comparison.json)
uses synthetic development examples, not unseen real-world validation. See the
[project README](../README.md) for completed ML work and outstanding integration.
