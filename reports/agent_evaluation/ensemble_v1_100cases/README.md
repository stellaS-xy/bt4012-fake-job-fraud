# Ensemble v1 agent quality evaluation

The completed audit attempted one agent-and-judge execution for each of
**100 real postings: 50 genuine and 50 fraudulent, from 100 distinct groups**.
All agent and judge calls completed; 95 cases passed all deterministic checks.
Five context reviews were incomplete, and the judge flagged four other cases
for possible unsupported claims. These are different outcome categories.

## Protocol and provenance

- Slurm job: **1177902**, wall time **1h16m14s**; evaluation loop **4517.93 seconds**.
- Protocol: `balanced-agent-quality-v2`; split declared `heldout`, frozen version
  `ensemble-v1-agent-eval-v2`; sample seed 2026.
- Classifier: saved calibrated MLP + Extra Trees v1, 50/50; frozen development
  threshold 0.20923746948802296; no refitting during agent evaluation.
- Agent and judge: local Ollama **qwen2.5:14b-instruct**, same model identifier.
- Sampling: one posting per group, balanced labels, alternating run order;
  no selection based on model predictions, correctness, wording or length.
- Gold labels remain outside agent/judge inputs. Twenty cases, ten per class,
  were preselected for human review with seed 2027; that audit remains **pending**.

[Evaluation manifest](evaluation_manifest.json) records the source/sample/training
hashes and result counts. [Runtime manifest](runtime_manifest.json) records
models, source hashes, rubric and classifier artifact provenance. The declared
heldout source is the existing audited test split; historical project-wide test
exposure remains unverified. No new predictions or judgments were generated for
this report.

## Execution and quality

| Outcome | All | Genuine | Fraudulent |
|---|---:|---:|---:|
| Planned cases | 100 | 50 | 50 |
| Agent completed | 100 | 50 | 50 |
| Judge completed | 100 | 50 | 50 |
| All code checks passed | 95 | 48 | 47 |
| Judge reported unsupported claims | 4 | 2 | 2 |
| Timeouts, fallback, invalid or unstarted inputs | 0 | 0 | 0 |

All 100 recommendations passed the code-enforced recommendation-policy check,
and all 100 rows retained valid classifier predictions. Policy compliance does
not imply that every explanation or upstream context interpretation was correct.
Median case time was **44.18 seconds**, maximum **70.34 seconds**.

| Judge dimension | Effective mean out of 4 | Eligible cases |
|---|---:|---:|
| Evidence grounding | 3.8947 | 95 |
| Context understanding | 3.8842 | 95 |
| Recommendation consistency | 3.8200 | 100 |
| Explanation clarity | 3.8632 | 95 |

The three 95-case means exclude incomplete context reviews under the existing
scoring rules; they are not all-case means. Missing scores are null, not zero.
See [quality_summary.json](quality_summary.json) and
[quality_summary.csv](quality_summary.csv) for full counts and class breakdowns.
Slurm reports **FAILED, exit 1:0** because five hard checks failed, not because
the evaluation crashed. All 100 cases and the final report were saved.

## Context review failures

Cases **17572, 15110, 17695, 17822 and 33** failed only
`context_review_completed`; all other deterministic checks passed. Their lexicon
function returned output, but the internal LLM context step had
`context_status=failed`, `context_error=review_incomplete` and `passed=false`.
Retained background observations are not successfully validated context findings.

All five recommendations were `review_needed`. Cases 17572, 17695 and 17822 were
also classifier-flagged; 15110 and 33 were below the classifier cutoff. Thus the
failed lexicon check prevented a pass in two cases. The original exception was
not retained, so invalid structured output, coverage/quote-validation errors or
other technical causes cannot be distinguished retrospectively. Missing fields
alone do not establish the cause.

Case **15110** additionally attributes the context failure to missing profile
and salary information. The saved tool provides no such cause. This is an
analyst-identified unsupported causal explanation that the judge did not flag;
it is not added retrospectively to the original four-case judge count.

## Review of judge flagged cases

These interpretations compare the saved answers with their actual tool traces.
They are a post-run analyst review, not independent human annotation, new judge
calls or replacement scores.

| Case | Saved tool evidence | Interpretation |
|---|---|---|
| 8596 | Context completed, lexicon passed, no review reasons; evidence labelled warning | Agent's no-unresolved-findings statement matches operational tool state. The judge's claim that it lacks support is not established by that state |
| 17750 | Four findings labelled uncertain; lexicon failed, classifier below cutoff; observations describe normal wording | Agent follows the tool's unresolved status. Judge's assertion that no unresolved findings exist contradicts that status. The context tool itself is inconsistent between prose and labels |
| 12126 | Evidence labelled warning, not negated; lexicon passed | Agent describes the warning as negated. This wording conflates distinct categories, although the recommendation follows policy |
| 3050 | Tool observations explicitly mention title/responsibility mismatch warnings; evidence remains warning | Mismatch warnings were present in the tool, but the agent's claim that they were negated/background is not established. The final answer did not invent the mismatch observation itself |

The five failure cases and four judge-flagged cases do not overlap: **nine distinct
cases** were selected for focused inspection. A 4% judge flag rate is neither a
verified hallucination rate nor a claim that the other 96 cases are error-free.
[Curated case review](case_review.json) retains diagnostic IDs, states and analysis
without redistributing posting texts. Original detailed records remain in ignored
`evaluator/outputs/balanced100_1177902/run/`.

## Conclusions and limits

The audit supports complete workflow execution and conservative handling of
incomplete context checks, with a 95% full-code-pass rate. It also identifies
context reliability problems, warning/negation wording errors and disagreements
between judge criticism and tool state. High judge means do not establish
production safety, independent semantic accuracy or reduced reviewer workload.

The 1:1 group-distinct sample does not estimate natural-prevalence precision or
review volume. Actor and judge share a model, and only one execution was attempted
per posting. The planned human review is still pending. Original outputs and
ratings remain unchanged; no further runs or model/lexicon changes are part of
this documentation handover.
