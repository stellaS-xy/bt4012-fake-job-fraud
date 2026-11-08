"""Feature engineering for Dataset 1 (EMSCAD fake job postings).

Everything the modelling notebooks need, taken unchanged from notebook 01.

Row-wise features (`build_rowwise`) use one posting at a time and learn nothing,
so they can be computed on any rows. Learned features (category encodings,
TF-IDF, the fraud-phrase score) live inside the sklearn pipelines from
`make_features` / `logistic_pipeline` and are refitted on the training part of
every fold. Never fit them on the test file.

Typical use
-----------
    from sklearn.model_selection import StratifiedGroupKFold
    from src.features import load_d1, build_inputs, logistic_pipeline, FEATURE_SETS, TARGET, GROUP

    train, test = load_d1("data/processed")
    X_train, X_test = build_inputs(train), build_inputs(test)
    y_train, groups = train[TARGET].to_numpy(), train[GROUP].to_numpy()

    for fit_idx, val_idx in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X_train, y_train, groups):
        pipe = logistic_pipeline(FEATURE_SETS["A3"]).fit(X_train.iloc[fit_idx], y_train[fit_idx])
        p = pipe.predict_proba(X_train.iloc[val_idx])[:, 1]

Use `make_features("A3")` instead of `logistic_pipeline` to put a different
model (e.g. gradient boosting) on the same features.
"""
from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

__all__ = [
    "TARGET", "GROUP", "SEED", "FEATURE_SETS", "PRIMARY_SET", "ROWWISE_SPEC", "ROWWISE_BINARY",
    "load_d1", "prepare_text", "build_rowwise", "build_inputs", "FraudPhraseScore",
    "make_features", "logistic_pipeline", "one_per_group", "weighted_log_odds",
]

SEED = 42
TARGET = "fraudulent"
GROUP = "group_id"
TEXT_COLS = ["title", "company_profile", "description", "requirements", "benefits"]
RAW_BINARY = ["telecommuting", "has_company_logo", "has_questions"]
RAW_CAT = ["employment_type", "required_experience", "required_education", "industry", "function"]
PLACEHOLDER = r"#(URL|EMAIL|PHONE)_[A-Za-z0-9]+#"
MISSING_FLAG_COLS = ["company_profile", "location", "department", "requirements", "benefits", "employment_type",
                     "required_experience", "required_education", "industry", "function"]

# Cumulative feature ladder. Change only this dictionary if the project's ladder differs.
FEATURE_SETS = {
    "A0": ["raw"],
    "A1": ["raw", "engineered"],
    "A2": ["raw", "engineered", "text_style", "tfidf"],
    "A3": ["raw", "engineered", "text_style", "tfidf", "lexicon"],
}

# The feature set every model is built on. Fixed after the ablation in notebook 01.
PRIMARY_SET = "A2"

DENSE_PARTS = {"salary_numeric", "text_style", "lexicon", "extra_numeric"}


# --------------------------------------------------------------------------- data
def add_country(df: pd.DataFrame) -> pd.DataFrame:
    """Add `country` (first part of `location`, upper case, 'UNK' if empty) if it is not there yet."""
    if "country" not in df.columns:
        df["country"] = df["location"].fillna("").str.split(",").str[0].str.strip().str.upper().replace("", "UNK")
    return df


def load_d1(processed_dir="data/processed"):
    """Read the frozen train and test files from notebook 00 and check that no group is in both."""
    processed_dir = Path(processed_dir)
    train = add_country(pd.read_csv(processed_dir / "d1_train.csv"))
    test = add_country(pd.read_csv(processed_dir / "d1_test.csv"))
    assert set(train[GROUP]).isdisjoint(set(test[GROUP])), "train and test share groups"
    return train, test


# ------------------------------------------------------------------ row-wise features
def prepare_text(df):
    """Return (raw combined text, cleaned text). The clean text has HTML decoded and masked
    contact details replaced by tokens (emailph, phoneph, urlph)."""
    text_all = df[TEXT_COLS].fillna("").astype(str).agg(" ".join, axis=1)
    text_clean = (text_all.map(html.unescape)
                  .str.replace(PLACEHOLDER, lambda m: f" {m.group(1).lower()}ph ", regex=True))
    return text_all, text_clean


def build_rowwise(df):
    """Compute every row-wise feature. Returns (features, spec), where spec maps
    feature name -> (block, kind) and kind is 'binary' or 'numeric'."""
    add_country(df)
    text_all, text_clean = prepare_text(df)
    n_chars = text_clean.str.len().clip(lower=1)
    n_words = text_clean.str.split().str.len().clip(lower=1)
    feats, spec = {}, {}

    def add(name, values, block, kind):
        feats[name] = values.astype("int8") if kind == "binary" else values.astype(float)
        spec[name] = (block, kind)

    # engineered: missing fields and masked contact details
    for col in MISSING_FLAG_COLS:
        add(f"missing_{col}", df[col].isna(), "engineered", "binary")
    optional = ["salary_range"] + MISSING_FLAG_COLS
    add("n_missing_fields", df[optional].isna().sum(axis=1), "engineered", "numeric")
    for kind in ("url", "email", "phone"):
        add(f"has_{kind}_placeholder", text_all.str.contains(f"#{kind.upper()}_", regex=False), "engineered", "binary")
    add("country_us", df["country"].eq("US"), "engineered", "binary")
    add("has_exclamation", text_all.str.contains("!", regex=False), "engineered", "binary")
    add("title_all_caps", df["title"].fillna("").str.isupper(), "engineered", "binary")

    # engineered: salary. Free text; some entries are spreadsheet date conversions such as '10-Oct'.
    sal = df["salary_range"].fillna("").astype(str).str.strip()
    numeric_range = sal.str.fullmatch(r"\d+\s*-\s*\d+")
    bounds = sal.where(numeric_range).str.extract(r"(\d+)\s*-\s*(\d+)").astype(float)
    lo, hi = bounds[0], bounds[1]
    add("has_salary", sal.ne(""), "engineered", "binary")
    add("salary_is_numeric_range", numeric_range, "engineered", "binary")
    add("salary_looks_like_date", sal.str.fullmatch(r"\d{1,2}-[A-Za-z]{3}|[A-Za-z]{3}-\d{1,2}"), "engineered", "binary")
    add("salary_min_log", np.log1p(lo).fillna(0), "engineered", "numeric")
    add("salary_max_log", np.log1p(hi).fillna(0), "engineered", "numeric")
    add("salary_span_log", np.log1p((hi - lo).clip(lower=0)).fillna(0), "engineered", "numeric")

    # text style
    for col in TEXT_COLS:
        add(f"words_{col}_log", np.log1p(df[col].fillna("").astype(str).str.split().str.len()), "text_style", "numeric")
    add("words_total_log", np.log1p(n_words), "text_style", "numeric")
    add("caps_ratio", text_clean.str.count(r"[A-Z]") / text_clean.str.count(r"[A-Za-z]").clip(lower=1), "text_style", "numeric")
    add("digit_ratio", text_clean.str.count(r"\d") / n_chars, "text_style", "numeric")
    add("exclaim_per_1k_chars", text_clean.str.count("!") / n_chars * 1000, "text_style", "numeric")
    add("money_symbols_log", np.log1p(text_clean.str.count(r"[$£€]")), "text_style", "numeric")
    add("long_caps_words_log", np.log1p(text_clean.str.count(r"\b[A-Z]{4,}\b")), "text_style", "numeric")
    add("avg_word_length", n_chars / n_words, "text_style", "numeric")
    add("unique_word_share", text_clean.str.lower().str.split().map(lambda w: len(set(w)) / max(len(w), 1)), "text_style", "numeric")
    add("html_entities_log", np.log1p(text_all.str.count(r"&(?:[A-Za-z]+|#\d+);")), "text_style", "numeric")
    add("url_placeholders_log", np.log1p(text_all.str.count("#URL_")), "text_style", "numeric")
    add("title_caps_share", df["title"].fillna("").map(lambda s: sum(c.isupper() for c in s) / max(sum(c.isalpha() for c in s), 1)), "text_style", "numeric")

    return pd.DataFrame(feats, index=df.index), spec


def _spec():
    """Feature spec from a one-row dummy frame, so the column groups are known without data."""
    cols = TEXT_COLS + RAW_BINARY + RAW_CAT + ["location", "department", "salary_range", "country"]
    dummy = pd.DataFrame({c: ["x"] for c in dict.fromkeys(cols)})
    return build_rowwise(dummy)[1]


ROWWISE_SPEC = _spec()
ROWWISE_BINARY = [c for c, (b, k) in ROWWISE_SPEC.items() if k == "binary"]
ROWWISE_NUMERIC_ENG = [c for c, (b, k) in ROWWISE_SPEC.items() if b == "engineered" and k == "numeric"]
ROWWISE_STYLE = [c for c, (b, k) in ROWWISE_SPEC.items() if b == "text_style"]


def build_inputs(df, rowwise=None):
    """The table every pipeline reads: row-wise features, raw fields, country, cleaned text, group id."""
    add_country(df)
    if rowwise is None:
        rowwise, _ = build_rowwise(df)
    _, text_clean = prepare_text(df)
    out = pd.concat([rowwise, df[RAW_BINARY + RAW_CAT + ["country", GROUP]]], axis=1)
    out["text_clean"] = text_clean
    return out


# ----------------------------------------------------------------- fraud-phrase score
def one_per_group(groups, seed):
    """Positions of one random row per group."""
    perm = np.random.default_rng(seed).permutation(len(groups))
    _, first = np.unique(groups[perm], return_index=True)
    return perm[first]


def weighted_log_odds(counts, y, alpha0=500.0, min_docs=5):
    """Weighted log-odds (fraud vs legitimate) with an informative Dirichlet prior
    (Monroe, Colaresi & Quinn, 2008). Phrases seen in fewer than `min_docs` documents get z = 0."""
    fraud, legit = counts[np.flatnonzero(y == 1)], counts[np.flatnonzero(y == 0)]
    yf = np.asarray(fraud.sum(axis=0)).ravel().astype(float)
    yl = np.asarray(legit.sum(axis=0)).ravel().astype(float)
    docs = np.asarray((counts > 0).sum(axis=0)).ravel()
    prior = yf + yl
    valid = (prior > 0) & (docs >= min_docs)
    alpha = alpha0 * prior / max(prior.sum(), 1.0)
    nf, nl = yf.sum(), yl.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = (np.log((yf + alpha) / (nf + alpha0 - yf - alpha))
                 - np.log((yl + alpha) / (nl + alpha0 - yl - alpha)))
        z = delta / np.sqrt(1.0 / (yf + alpha) + 1.0 / (yl + alpha))
    delta, z = np.where(valid, delta, 0.0), np.where(valid, z, 0.0)
    return delta, z, valid


class FraudPhraseScore(BaseEstimator, TransformerMixin):
    """Three features per posting: number of fraud-leaning phrases, number of legitimate-leaning
    phrases, and the summed log-odds weight. Input: DataFrame with `text_clean` and `group_id`.

    Training rows get out-of-fold scores (inner StratifiedGroupKFold) so their own labels never
    leak into their feature; other rows are scored with the list built from all training rows."""

    def __init__(self, n_fraud=50, n_legit=20, min_df=5, ngram_range=(1, 2), inner_splits=3, seed=42):
        self.n_fraud, self.n_legit, self.min_df = n_fraud, n_legit, min_df
        self.ngram_range, self.inner_splits, self.seed = ngram_range, inner_splits, seed

    def _lexicon(self, counts, y, groups):
        pick = one_per_group(groups, self.seed)
        delta, z, valid = weighted_log_odds(counts[pick], y[pick])
        fraud = np.argsort(-np.where(valid, z, -np.inf))[:self.n_fraud]
        legit = np.argsort(np.where(valid, z, np.inf))[:self.n_legit]
        fraud, legit = fraud[z[fraud] > 0], legit[z[legit] < 0]
        return fraud, legit, delta

    @staticmethod
    def _score(present, fraud, legit, delta):
        n_fraud = np.asarray(present[:, fraud].sum(axis=1)).ravel()
        n_legit = np.asarray(present[:, legit].sum(axis=1)).ravel()
        chosen = np.r_[fraud, legit].astype(int)
        weight = np.asarray(present[:, chosen] @ delta[chosen]).ravel()
        return np.c_[np.log1p(n_fraud), np.log1p(n_legit), weight]

    def fit_transform(self, X, y):
        texts, groups, y = X["text_clean"].to_numpy(), X[GROUP].to_numpy(), np.asarray(y)
        self.vec_ = CountVectorizer(ngram_range=self.ngram_range, min_df=self.min_df, lowercase=True, dtype=np.int32)
        counts = self.vec_.fit_transform(texts).tocsr()          # vocabulary only, no labels
        present = (counts > 0).astype(np.float32).tocsr()
        self.fraud_, self.legit_, self.delta_ = self._lexicon(counts, y, groups)
        out = np.zeros((len(y), 3))
        inner = StratifiedGroupKFold(n_splits=self.inner_splits, shuffle=True, random_state=self.seed)
        for fit_idx, score_idx in inner.split(texts, y, groups):
            fraud, legit, delta = self._lexicon(counts[fit_idx], y[fit_idx], groups[fit_idx])
            out[score_idx] = self._score(present[score_idx], fraud, legit, delta)
        return out

    def fit(self, X, y):
        self.fit_transform(X, y)
        return self

    def transform(self, X):
        present = (self.vec_.transform(X["text_clean"].to_numpy()) > 0).astype(np.float32).tocsr()
        return self._score(present, self.fraud_, self.legit_, self.delta_)

    def get_feature_names_out(self, input_features=None):
        return np.array(["fraud_phrase_count_log", "legit_phrase_count_log", "phrase_weight_sum"])


# -------------------------------------------------------------------------- pipelines
def one_hot(min_frequency):
    return Pipeline([
        ("fill", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency)),
    ])


def make_features(level=None, blocks=None, extra_numeric=(), dense_weight=1.0, tfidf_max_features=30000):
    """Unfitted feature ColumnTransformer for a ladder level (e.g. 'A3') or an explicit block list
    (any of 'raw', 'engineered', 'text_style', 'tfidf', 'lexicon'). Input: the table from `build_inputs`.
    `dense_weight` scales the standardised dense blocks relative to TF-IDF."""
    blocks = blocks if blocks is not None else FEATURE_SETS[level]
    parts = []
    if "raw" in blocks:
        parts += [("raw_binary", "passthrough", RAW_BINARY), ("raw_category", one_hot(50), RAW_CAT)]
    if "engineered" in blocks:
        parts += [("flags", "passthrough", ROWWISE_BINARY),
                  ("salary_numeric", StandardScaler(), ROWWISE_NUMERIC_ENG),
                  ("country", one_hot(100), ["country"])]
    if "text_style" in blocks:
        parts += [("text_style", StandardScaler(), ROWWISE_STYLE)]
    if "tfidf" in blocks:
        parts += [("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.5, max_features=tfidf_max_features,
                                            sublinear_tf=True, stop_words="english", dtype=np.float32), "text_clean")]
    if "lexicon" in blocks:
        parts += [("lexicon", Pipeline([("score", FraudPhraseScore()), ("scale", StandardScaler())]),
                   ["text_clean", GROUP])]
    if extra_numeric:
        parts += [("extra_numeric", StandardScaler(), list(extra_numeric))]
    weights = {name: dense_weight for name, *_ in parts if name in DENSE_PARTS}
    return ColumnTransformer(parts, sparse_threshold=1.0, remainder="drop", transformer_weights=weights or None)


def logistic_pipeline(blocks, extra_numeric=(), dense_weight=1.0, C=0.3, tfidf_max_features=30000):
    """Class-weighted logistic regression on the given blocks (the project baseline)."""
    return Pipeline([
        ("features", make_features(blocks=blocks, extra_numeric=extra_numeric, dense_weight=dense_weight,
                                   tfidf_max_features=tfidf_max_features)),
        ("model", LogisticRegression(solver="liblinear", class_weight="balanced", C=C, max_iter=2000)),
    ])
