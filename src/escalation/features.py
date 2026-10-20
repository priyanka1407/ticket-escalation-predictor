"""Feature engineering and preprocessing pipelines."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TEXT = "description"
CATEGORICAL = ["error_code", "service", "channel", "customer_tier"]
NUMERIC = ["is_after_cutoff", "is_weekend", "log_seen_before", "is_new_error", "has_runbook",
           "log_customers_affected", "logs_attached", "l1_experience_months", "reopened",
           "log_amount", "text_length"]
TARGET = "escalated"


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw ticket fields into features an engineer would reason with."""
    out = df.copy()
    out["is_after_cutoff"] = (out["hour"] >= 16).astype(int)            # payment cut-off pressure
    out["is_weekend"] = (out["weekday"] >= 5).astype(int)
    out["log_seen_before"] = np.log1p(out["signature_seen_before"])      # diminishing returns
    out["is_new_error"] = (out["signature_seen_before"] == 0).astype(int)
    out["log_customers_affected"] = np.log1p(out["customers_affected"])
    out["log_amount"] = np.log1p(out["amount"])                          # amounts are heavily skewed
    out["text_length"] = out[TEXT].str.len()
    return out


def sparse_preprocessor() -> ColumnTransformer:
    """For linear models: full TF-IDF vocabulary (sparse) + one-hot + scaled numbers."""
    return ColumnTransformer([
        ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True), TEXT),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
    ])


def dense_preprocessor(text_components: int = 40) -> ColumnTransformer:
    """For trees and the neural net: TF-IDF compressed with SVD (LSA) so everything is dense."""
    text = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)),
        ("svd", TruncatedSVD(n_components=text_components, random_state=42)),
    ])
    return ColumnTransformer([
        ("text", text, TEXT),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
    ], sparse_threshold=0.0)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feats = engineer(df)
    return feats[[TEXT] + CATEGORICAL + NUMERIC], feats[TARGET]
