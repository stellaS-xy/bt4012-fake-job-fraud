# Agent evaluator

This package lives beside `agent/` and `frontend/`. It evaluates the agent with
**code checks and an LLM judge**, without requiring manual reviewer annotations
or confirmation. It does not execute business decisions or save reviewer audit
records. A judge's rating is an automated assessment, not verified ground truth.

## What is evaluated

| Part | Evaluation |
|---|---|
| Code checks | Original input preserved, consistent posting ID, unique evidence IDs, exact quotes, valid citations, evidence matching tool traces, unchanged classifier values and two-tool recommendation-policy compliance |
| LLM judge | Evidence grounding, interpretation of warnings/negations, recommendation consistency and explanation clarity, each scored 0–4 with a reason |
| Fraud labels | Precision/recall and false alarms only when a validated **non-placeholder classifier** prediction is available |
| Action annotations | Optional exact comparison with `expected_action`; not required for judge evaluation |

Normal agent runs now use a real LightGBM A2 classifier artifact. Fraud metrics
use valid real classifier tool outputs even if the LLM later falls back, so
failed final answers are not silently removed from classifier evaluation.
Automated tests use explicitly marked test fixtures; their classifier outputs are excluded from fraud metrics. Recommendations are not automatically converted to fraud labels.
`review_needed` can be a sensible action for either a genuine or fraudulent
posting. The report's action-by-fraud-label table is descriptive, not an accuracy
score. Dataset labels alone cannot establish whether an explanation is correct.

## Installation and local judge

Use the existing agent environment from the repository root:

```powershell
.\agent\.venv\Scripts\python.exe -m pip install -r evaluator/requirements.txt
```

The evaluator uses LangChain directly and needs no additional evaluation service
or hosted evaluation account. Keep Ollama running and download the judge model
if it is not already available:

```powershell
ollama pull qwen3:4b-instruct
```

The judge defaults to that local model without a real API key. Optional settings
can be placed in ignored `evaluator/.env`:

```dotenv
JUDGE_BACKEND=ollama_local
JUDGE_MODEL=qwen3:4b-instruct
JUDGE_TIMEOUT_SECONDS=180
```

Local judging defaults to a 180-second request timeout; cloud judging defaults
to 60 seconds. `JUDGE_TIMEOUT_SECONDS` can override this with a finite positive
number. Local requests do not automatically retry a slow inference. Invalid
structured output can still receive one explicit validation-correction attempt.

The project-root `.env`, then `evaluator/.env`, then `JUDGE_*` process variables
are read in that priority order. Judge configuration is separate from
`agent/.env`. A cloud judge requires explicit `JUDGE_BACKEND=ollama` and
`JUDGE_API_KEY`; it incurs Ollama cloud charges. Direct DeepSeek API support is
not implemented. Only real local/cloud models are supported.

Choose another downloaded model with `--judge-model <model-name>` if your
hardware permits. By default the actor and judge may be the same local model;
reports record matching configured model identifiers because shared mistakes
can make self-evaluation optimistic. A different model is not a guarantee of
correctness either.

## Evaluate saved agent outputs

Use a JSON case downloaded from the frontend, a raw `ReviewCase` JSON, or JSONL
records with a `case` property:

```powershell
.\agent\.venv\Scripts\python.exe -m evaluator --cases path/to/review.json
```

The evaluator reuses those outputs and calls only the judge. To run just code
checks without any model calls:

```powershell
.\agent\.venv\Scripts\python.exe -m evaluator --cases path/to/review.json --code-only
```

To compare different judges without rerunning the agent, replay a previous
run's `cases.jsonl`. An optional wrapper can preserve annotations separately:

```json
{"case_id":"ad_1","fraudulent":1,"case":{"posting":"... full saved ReviewCase object ..."}}
```

Replace the illustrative `case` value with the complete actual object. The
`fraudulent` annotation and optional `expected_action` remain outside that object
and are never sent to the judge.

## Run the agent on dataset rows and evaluate

A CSV needs `title` and `description`. Supported optional posting fields are
`company_profile`, `salary_range`, `requirements` and `benefits`. `job_id` or
`posting_id` identifies rows. The optional `fraudulent` column contains 0 or 1.
The supported classifier fields also include location, department, employment type, required experience/education, industry, function, telecommuting, company-logo and screening-question indicators. Unknown values stay missing; gold labels are never model inputs. Never put the gold label in
an input text field. Use development data while changing prompts or rules:

```powershell
.\agent\.venv\Scripts\python.exe -m evaluator --data path/to/development.csv --limit 10
```

JSONL is also supported; see `examples/postings.jsonl` for two invented examples:

```powershell
.\agent\.venv\Scripts\python.exe -m evaluator --data evaluator/examples/postings.jsonl
```

Gold annotations are retained only for reporting. Each row's selected posting
fields go to the agent. The agent receives the same `ordinary` example identifier for every row,
independently of the gold label. In real mode the identifier does not choose a
score: LightGBM always scores the actual fields. Test-only fixture scores are
excluded from fraud metrics.

To skip only the judge while still running the real agent:

```powershell
.\agent\.venv\Scripts\python.exe -m evaluator --data evaluator/examples/postings.jsonl --code-only
```

`--code-only` by itself skips the judge, not the real agent when using `--data`.
`--limit` selects a prefix for a smoke run, not a representative sample.
Duplicate IDs and invalid annotations fail input loading. Missing/invalid
required posting text is recorded as an input failure without a model call;
those rows are not silently counted as correct or dropped from the summary.

## Judge rubric and safeguards

The judge sees only the posting, agent assessment, evidence and completion/
placeholder status. It does not receive `fraudulent`, `expected_action`, the demo
scenario, reviewer decisions, model names or code-check verdicts. Posting text,
agent answers and tool observations are explicitly treated as untrusted data.

The v4 rubric requires `pass_no_review_needed` when both checks pass and
`review_needed` when either check is flagged, incomplete, unresolved or a
placeholder. The latter must explain its reasons. Code independently enforces
this policy from actual results rather than trusting claimed `passed` values.
The judge receives classifier score/cutoff/flag and context-completion status,
without model identities or deterministic verdicts. It checks whether the
explanation and next steps fit the recommendation, including unnecessary
mandatory review after a pass. Neither outcome executes a decision or requires
human confirmation. A pass does not establish employer legitimacy.

Each dimension is scored 0 (unusable/contradicted) through 4 (fully satisfactory).
The judge supplies a short reason, existing evidence IDs and `answer_quotes`
from what the agent actually wrote. The generated judge selects one verbatim
passage per dimension from the actual answer, and its reasons have bounded
lengths to avoid unnecessary local generation time. Scores 3–4 require an exact answer excerpt;
quoting only a tool observation cannot justify those scores. Criticisms can
quote the rationale, limitations or next steps, so unsupported claims cannot
hide in a limitations list. Perfect grounding is rejected if the judge also
reports unsupported claims. Code validates IDs, score ranges and quotes,
with one bounded correction attempt for invalid judge output. A failed judge has
**null scores**, an explicit failure code and zero evaluation coverage; it is
not treated as a pass or as a meaningful score of zero. Judge averages include
successful, eligible dimension scores only, so read the failure counts and
`scored_count_by_dimension` alongside them.

Execution and safety are separate. A fallback may have
`safety_checks_passed=true` while `review_completed=false` makes the full code
check fail. An incomplete answer or required context review cannot receive
completed-answer grounding, context or clarity scores. Those effective scores
are null, even if the raw judge awards 4. Recommendation consistency can still
describe a safe `review_needed` fallback. `judgment` preserves the raw model opinion;
`scoring` records effective scores, exclusions and policy adjustments. The
summary includes `agent_completed`, `safe_fallbacks`, `context_review_failures`
and `quality_scored_cases`; a safe fallback is never presented as a successful
substantive explanation. Hard grounding or policy failures also affect their
corresponding effective scores.

These guards validate output structure and references, not the truth of a
model's semantic opinion. There is no claim of human-validated judge accuracy.

## Outputs, repeats and held-out evaluation

Each run creates a new ignored `evaluator/outputs/run_.../` directory:

- `report.json`: summary, per-case code/judge results, labels, model metadata,
  rubric/input fingerprints and limitations.
- `results.jsonl`: every completed evaluation record, flushed incrementally,
  including failures.
- `cases.jsonl`: the first saved agent output per case, with annotations, for
  replay. Later repeat outputs remain in `results.jsonl`. Invalid-input or
  crashed-agent cases without a case object remain in the full results only.

These local files contain posting text and annotations. They are ignored by
Git. API keys and raw provider error messages are excluded. Optional LangSmith
uploads are disabled around judge calls, as they are in the agent.

```powershell
.\agent\.venv\Scripts\python.exe -m evaluator --data path/to/development.csv --repeats 2 --output-dir evaluator/outputs/my_run
```

Repeats rerun the agent for `--data` and rejudge saved outputs for `--cases`.
They are not new independent cases. Existing output files are never silently
overwritten. The CLI exits with code 1 for invalid inputs, agent/judge failures
or failed hard code checks; low judge ratings are reported without a hidden
pass threshold.

No dataset, including the reserved test file, is selected automatically. Freeze
the implementation before the final held-out evaluation, then record its version:

```powershell
.\agent\.venv\Scripts\python.exe -m evaluator --data data/processed/d1_test.csv --split heldout --frozen-version final-v1
```

That version is a provenance label, not an automatic check that the model was
frozen. Known reserved-test filenames require the held-out flags. Do not use
held-out results to tune prompts, rules or judge criteria. Connecting a real
classifier is still necessary before reporting end-to-end fraud-detection
metrics; evaluating explanations does not replace employer verification.

## Readable input/tool/answer report

Render a completed run without calling any models:

```powershell
.\agent\.venv\Scripts\python.exe -m evaluator.export_html --report evaluator/outputs/your_run/report.json --output reports/agent_evaluation/your_run.html
```

The standalone report separates the full input, real classifier score/cutoff,
lexicon/context findings, actual LLM answer and automated evaluation. Full tool
JSON and judge reasons are included in expandable sections. Posting text is
HTML-escaped. The two-tool LightGBM demonstration is saved in
[`reports/agent_evaluation/lightgbm_two_tools.html`](../reports/agent_evaluation/lightgbm_two_tools.html).

## Files and tests

| File | Responsibility |
|---|---|
| `config.py` | Independent judge configuration |
| `data.py`, `schemas.py` | Allowlisted dataset/replay inputs and structured outputs |
| `checks.py` | Deterministic grounding and policy checks |
| `judge.py` | LLM rubric, constrained response and validation |
| `runner.py` | Per-case execution, replay, failure accounting and label isolation |
| `export_html.py` | Standalone readable input/tool/answer/evaluation report |
| `report.py` | Code/judge summaries and eligible classifier metrics |
| `scoring.py` | Execution eligibility and explicit adjustments, preserving the raw judge opinion |
| `__main__.py` | CLI, split metadata and incremental local outputs |
| `tests/` | No-network tests of leakage, grounding, failures and SDK protocols |

```powershell
.\agent\.venv\Scripts\python.exe -m unittest discover -s evaluator/tests -v
```

The existing `agent/tests/evaluate_lexicon.py` remains a focused comparison of
rules, LLM-only extraction and combined extraction on annotated request spans.
This evaluator instead scores complete agent recommendations and explanations.

Schema version 5 agent cases use the new recommendation policy. Keep earlier
rejection/escalation reports as historical runs; rerun their input postings to
evaluate this policy instead of mixing old and new case contracts.
