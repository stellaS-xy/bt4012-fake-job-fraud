# Notebook index

The notebooks now use a unique consecutive sequence **00–15**. Original and
revised comparisons have separate numbers, and MLP and geography no longer
share 09. Numbers identify the current files; historical experiment IDs in
saved reports, cache directories, requirements files and module names remain
unchanged. The mapping below connects old references to current notebooks.

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
| [04 revised comparison](<04_d1_model_comparison_revised.ipynb>) | Repeat the comparison against notebook 02's saved outer assignments | Restores matched 25 outer folds; model-specific text limits retained; `reports/model_comparison_revised/` | Matched comparison history |
| [05 word and character classifier](05_classifier_word_char.ipynb) | Test character patterns alongside word features | Restores 02 outer folds; `reports/classifier_word_char/` | Candidate experiment |
| [06 error analysis](06_error_analysis.ipynb) | Inspect repeated false positives/negatives and persistent errors | Reads cached development OOF predictions; no fitting or test access | Classifier error-analysis history, distinct from agent error analysis |
| [07 targeted features](07_targeted_feature_experiments.ipynb) | Test missingness and targeted feature hypotheses | Matched 02 folds and grouped inner selection; `reports/classifier_targeted_features/` | Development ablations |
| [08 semantic embeddings](08_semantic_embeddings.ipynb) | Frozen MiniLM field embeddings with logistic regression | Matched outer folds; encoder frozen, scaler/classifier fitted within folds; `reports/classifier_semantic_embeddings/` | Representation experiment; not Transformer fine-tuning |
| [09 Naive Bayes](09_naive_bayes_comparison.ipynb) | MultinomialNB and ComplementNB with nonnegative word TF-IDF | Matched 25 outer folds and inner alpha selection; `reports/classifier_naive_bayes/` | Classifier experiment |
| [10 MLP](10_mlp_model.ipynb) | Deep A2 MLP versus matched LR | Restores 02 assignments; inner group stopping split, fresh refit, class weighting; cluster guide documents saved paths | Neural comparison history; not the final calibrated ensemble |
| [11 geography](11_geography_ablation.ipynb) | State/city frequency encoding and group-deduplicated counts | Restores 02 folds; encoders fitted within training folds; `reports/geography_ablation/` | Negative feature ablation; geographical additions not adopted |
| [12 DistilBERT](12_distilbert_finetuning.ipynb) | Supervised text-only Transformer fine-tuning | Initial cluster run evaluated one saved outer fold, with training-side stopping/selection | Exploratory single-fold result; cannot compare its AP to a 25-fold mean |
| [13 reference comparison](13_d1_reference_comparison.ipynb) | Fresh LightGBM replay versus calibrated MLP + Extra Trees | Recovered development pool and 25 verified original outer signatures; `reports/13_reference_comparison/` | Final model-selection reference |
| [14 MLP development](14_d1_mlp_development.ipynb) | Additional controlled development configurations and blends | Separate declared common protocol; consult its frozen run plan and `RUN_NOTEBOOK_14.md` | Development history; does not supersede the frozen v1 model |
| [15 final test results](15_final_ensemble_test_results.ipynb) | Display downloaded final model metrics | Reads saved `data/processed/final_ensemble_test/v1/test_metrics.json`; never trains, downloads or predicts | Final saved-results viewer |

## Historical filename mapping

| Previous filename | Current filename |
|---|---|
| `03 (revised config).ipynb` | [04_d1_model_comparison_revised.ipynb](04_d1_model_comparison_revised.ipynb) |
| `04_classifier_word_char.ipynb` | [05_classifier_word_char.ipynb](05_classifier_word_char.ipynb) |
| `05_error_analysis.ipynb` | [06_error_analysis.ipynb](06_error_analysis.ipynb) |
| `06_targeted_feature_experiments.ipynb` | [07_targeted_feature_experiments.ipynb](07_targeted_feature_experiments.ipynb) |
| `07_semantic_embeddings.ipynb` | [08_semantic_embeddings.ipynb](08_semantic_embeddings.ipynb) |
| `08_naive_bayes_comparison.ipynb` | [09_naive_bayes_comparison.ipynb](09_naive_bayes_comparison.ipynb) |
| `09_MLP_model.ipynb` | [10_mlp_model.ipynb](10_mlp_model.ipynb) |
| `09_geography_ablation.ipynb` | [11_geography_ablation.ipynb](11_geography_ablation.ipynb) |
| `10_distilbert_finetuning.ipynb` | [12_distilbert_finetuning.ipynb](12_distilbert_finetuning.ipynb) |

Notebooks 00–03 and 13–15 retain their filenames. The original 03 comparison is
still distinct from the revised comparison, now 04. Historical phrases such as
"notebook 09 MLP" in frozen report metadata refer to current notebook 10; use the
mapping rather than rewriting saved evidence. Notebook headings and their own
source-path lookups have been updated; model settings, cached results and outputs
were not regenerated. Self-fingerprinting notebook cells now refer to the new
paths, so a deliberate future rerun may produce a new source/cache fingerprint.

## Interpretation rules

A2 combines raw fields, engineered/missingness features, text style and word
TF-IDF. A3 additionally uses learned fraud-phrase scores; this is distinct from
the agent's lexicon evidence tool. Exact feature limits and fitted settings are
recorded in each run's plan, not implied by the feature-set name alone.

Comparisons isolate changes only when rows, groups, outer folds, inner selection
and feature settings align. Do not merge all saved model files into one ranking.
Original 03, revised comparison 04 (historical revised 03) and recovered notebook 13 must retain their provenance.
Repeated grouped folds overlap statistically and are not 25 independent datasets.
The selected ensemble's corrected paired interval includes zero; its higher mean
is not proof of superiority. See [results](../reports/README.md).

The final fixed split passed current posting/group overlap checks. Earlier
project-wide test exposure is unverified. Classifier final-test results and the
balanced agent audit answer different questions and are documented separately.
No future model development or experiment reruns are required for this handover.
