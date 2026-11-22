# Results index

The final classifier evaluation and final agent-quality audit answer different
questions. Older experiment results remain as historical evidence and are not a
single directly comparable leaderboard. No model or LLM execution is required
to inspect these files.

## Final results

| Result | Files | Interpretation |
|---|---|---|
| Model-selection comparison | [Notebook 13 summary](13_reference_comparison/summary.csv), [paired differences](13_reference_comparison/paired_differences.csv), [manifest](13_reference_comparison/manifest.json), [HTML](13_reference_comparison/comparison.html) | Calibrated ensemble mean AP 0.6891, fold SD 0.0566 over 25 matched grouped outer folds; corrected superiority interval includes zero |
| Final classifier on fixed test split | [Classifier report](final_classifier/README.md), [aggregate metrics](final_classifier/test_metrics.json) | AP 0.7848 on 3,441 postings; current split separation audited, historical exposure unverified |
| Final 100-case agent audit | [Report and error analysis](agent_evaluation/ensemble_v1_100cases/README.md), [quality summary](agent_evaluation/ensemble_v1_100cases/quality_summary.json), [runtime](agent_evaluation/ensemble_v1_100cases/runtime_manifest.json) | Balanced 50+50 group-distinct audit; 100 agent/judge completions, 95 full code passes; shared actor/judge model |

## Earlier experiment evidence

| Directory | Scope |
|---|---|
| `tables/`, `figures/` | Data understanding, feature blocks and statistical analysis |
| `model_comparison/` | Original 03 comparison, with its own protocol and feature limits |
| `model_comparison_revised/` | Comparison restored to notebook 02 outer-fold assignments |
| `classifier_word_char/` | Word/character candidate experiment |
| `classifier_targeted_features/` | Error-motivated feature ablations |
| `classifier_semantic_embeddings/` | Frozen MiniLM representation experiment |
| `classifier_naive_bayes/` | Multinomial and Complement Naive Bayes comparison |
| `geography_ablation/` | State/city frequency-encoding ablation |
| `agent_evaluation/development_comparison.json` | Historical synthetic evidence-extraction development checks |
| `agent_evaluation/lightgbm_two_tools.*` | Historical LightGBM demonstration, not the final ensemble audit |

Notebook 10 MLP (historical 09) and notebook 12 DistilBERT (historical 10) runs also produced cluster/cache
outputs; use the [notebook index](../notebooks/README.md) and manifests rather
than assuming every complete report is committed here. A single-fold score
must not be compared to a 25-fold mean as evidence of improvement.

## Artifact boundaries

Curated aggregates, interpretation and configuration/source hashes belong in
`reports/`. Full per-case postings, traces, predictions, model ZIPs and training
caches remain in ignored run/artifact directories or a deliberate model Release.
The curated agent case review includes diagnostic case IDs and observations,
not the posting texts. Original score files are preserved; analysis does not
replace judge ratings or recompute test predictions.
