# Risk-check resources

`dataset_terms.csv` is a frozen copy of the 33 terms in
`reports/tables/lexicon_confirmed.csv`, produced by notebook 00's training-only
discovery/confirmation analysis. Every term is marked `background_only`.
They are associations in this historical dataset, not general scam language,
independent fraud probabilities or explanations of the classifier's score.
The runtime does not discover or confirm terms from incoming ads.

## Refresh the historical snapshot

Rerunning Notebook 00 updates its report CSV, not this resource. If that analysis
is intentionally rerun:

1. Review the updated `reports/tables/lexicon_confirmed.csv` and its training-only
   discovery/confirmation results.
2. Copy its `feature` values into this file's `term` column, keeping
   `role=background_only` for every row. Do not copy odds ratios as per-ad fraud
   probabilities or promote job titles to actionable risk rules.
3. Review the resource diff, rerun agent tests and recheck the annotated
   development examples. Restart the app so its cached vocabulary is refreshed.

The agent work did not replace Notebook 00's discovery code. The separate risk
patterns and LLM context interpretation implement a different role: explaining
concrete requests in an individual posting. The ML pipeline's independently
learned `FraudPhraseScore` transformer is not loaded by the agent.

## Development examples and evaluation

`risk_examples.jsonl` contains 12 invented, manually specified development
examples. `gold_requests` marks explicit payment/secret-credential requests,
not whether a whole job advertisement is fraudulent. Empty gold lists include
ordinary job/industry terms, warnings, negation, urgency and normal payroll
details. Examples also cover paraphrases, prompt injection, requirements and
benefits. These examples were used in development and cannot establish detection
accuracy or generalization.

For a meaningful evaluation, prepare a separate set of human-annotated
development postings in the same JSONL format. Keep related postings together,
keep gold labels outside the LLM input, and do not use the project's reserved
test set for prompt/rule development. Annotate the exact supporting span, field
and category. The script rejects invalid gold spans and duplicate case IDs.

The English patterns are maintained separately in `agent/risk_rules.py`, with a
version identifier included in reports. Regex matches select candidates; they
do not verify employer intent. Extend patterns using development failures, then
evaluate on a fresh annotated development set. Preserve full context and exact
quotes when adding patterns. The rules are deliberately not comprehensive.
