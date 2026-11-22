# Notebook index

The notebooks preserve the research sequence; their numbers are historical
identifiers, not a single execution pipeline. Two notebooks use 09, and 03 has
an original and revised version. Names are preserved to keep references,
launchers and fingerprints valid. Notebooks 11 and 12 are not present.

Read saved results first. Earlier notebooks can train models when their execution
flags are enabled; running every notebook is unnecessary for using Ensemble v1.
[Notebook 15](15_final_ensemble_test_results.ipynb) only displays saved final test
results. The [dependencies guide](../docs/DEPENDENCIES.md) maps environments.

## Experiment sequence

| Notebook | Purpose | Evaluation and outputs | Status |
|---|---|---|---|
| [00 data understanding](00_d1_data_understanding_clean.ipynb) | Cleaning, related-posting groups, initial split, EDA and phrase discovery | Creates processed D1 data and analysis tables; exploratory work uses development data | Data preparation history |
| [01 feature engineering](01_d1_feature_engineering_v2.ipynb) | Raw/engineered/text/phrase blocks, cumulative and drop-one ablations | Group-aware feature evaluation; reusable transformations in `src/features.py`, saved tables under `reports/tables/` | Feature-selection history |
| [02 modelling setup](02_d1_modelling_setup.ipynb) | Preliminary TF-IDF and A2 logistic baselines, cached folds and predictions | Five repeats of five grouped outer folds; inner grouped selection; saved fold assignments underpin later matched studies | Preliminary baseline reference |
| [03 original comparison](03_d1_model_comparison.ipynb) | Text LR, A2 LR, LightGBM and XGBoost comparison | Its own shared 25-fold comparison; A2 candidates use 2k TF-IDF, text-only reference 30k; `reports/model_comparison/` | Historical protocol; not directly interchangeable with revised results |
| [03 revised config](<03 (revised config).ipynb>) | Repeat the comparison against notebook 02's saved outer assignments | Restores matched 25 outer folds; model-specific text limits retained; `reports/model_comparison_revised/` | Matched comparison history |
| [04 word and character classifier](04_classifier_word_char.ipynb) | Test character patterns alongside word features | Restores 02 outer folds; `reports/classifier_word_char/` | Candidate experiment |
| [05 error analysis](05_error_analysis.ipynb) | Inspect repeated false positives/negatives and persistent errors | Reads cached development OOF predictions; no fitting or test access | Classifier error-analysis history, distinct from agent error analysis |
| [06 targeted features](06_targeted_feature_experiments.ipynb) | Test missingness and targeted feature hypotheses | Matched 02 folds and grouped inner selection; `reports/classifier_targeted_features/` | Development ablations |
| [07 semantic embeddings](07_semantic_embeddings.ipynb) | Frozen MiniLM field embeddings with logistic regression | Matched outer folds; encoder frozen, scaler/classifier fitted within folds; `reports/classifier_semantic_embeddings/` | Representation experiment; not Transformer fine-tuning |
| [08 Naive Bayes](08_naive_bayes_comparison.ipynb) | MultinomialNB and ComplementNB with nonnegative word TF-IDF | Matched 25 outer folds and inner alpha selection; `reports/classifier_naive_bayes/` | Classifier experiment |
| [09 MLP](09_MLP_model.ipynb) | Deep A2 MLP versus matched LR | Restores 02 assignments; inner group stopping split, fresh refit, class weighting; cluster guide documents saved paths | Neural comparison history; not the final calibrated ensemble |
| [09 geography](09_geography_ablation.ipynb) | State/city frequency encoding and group-deduplicated counts | Restores 02 folds; encoders fitted within training folds; `reports/geography_ablation/` | Negative feature ablation; geographical additions not adopted |
| [10 DistilBERT](10_distilbert_finetuning.ipynb) | Supervised text-only Transformer fine-tuning | Initial cluster run evaluated one saved outer fold, with training-side stopping/selection | Exploratory single-fold result; cannot compare its AP to a 25-fold mean |
| [13 reference comparison](13_d1_reference_comparison.ipynb) | Fresh LightGBM replay versus calibrated MLP + Extra Trees | Recovered development pool and 25 verified original outer signatures; `reports/13_reference_comparison/` | Final model-selection reference |
| [14 MLP development](14_d1_mlp_development.ipynb) | Additional controlled development configurations and blends | Separate declared common protocol; consult its frozen run plan and `RUN_NOTEBOOK_14.md` | Development history; does not supersede the frozen v1 model |
| [15 final test results](15_final_ensemble_test_results.ipynb) | Display downloaded final model metrics | Reads saved `data/processed/final_ensemble_test/v1/test_metrics.json`; never trains, downloads or predicts | Final saved-results viewer |

## Interpretation rules

A2 combines raw fields, engineered/missingness features, text style and word
TF-IDF. A3 additionally uses learned fraud-phrase scores; this is distinct from
the agent's lexicon evidence tool. Exact feature limits and fitted settings are
recorded in each run's plan, not implied by the feature-set name alone.

Comparisons isolate changes only when rows, groups, outer folds, inner selection
and feature settings align. Do not merge all saved model files into one ranking.
Original 03, revised 03 and recovered notebook 13 must retain their provenance.
Repeated grouped folds overlap statistically and are not 25 independent datasets.
The selected ensemble's corrected paired interval includes zero; its higher mean
is not proof of superiority. See [results](../reports/README.md).

The final fixed split passed current posting/group overlap checks. Earlier
project-wide test exposure is unverified. Classifier final-test results and the
balanced agent audit answer different questions and are documented separately.
No future model development or experiment reruns are required for this handover.
