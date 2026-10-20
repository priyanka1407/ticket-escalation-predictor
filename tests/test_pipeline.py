import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from escalation.data import generate_tickets  # noqa: E402
from escalation.features import NUMERIC, dense_preprocessor, engineer, sparse_preprocessor, split_xy  # noqa: E402
from escalation.metrics import evaluate, threshold_for_recall  # noqa: E402
from escalation.mlp import predict_proba, train_mlp  # noqa: E402


def test_data_is_reproducible_and_balanced_enough():
    a, b = generate_tickets(500, seed=7), generate_tickets(500, seed=7)
    assert a.equals(b)
    assert 0.15 < a["escalated"].mean() < 0.45


def test_engineered_features_exist_and_make_sense():
    df = engineer(generate_tickets(300, seed=1))
    assert set(NUMERIC) <= set(df.columns)
    assert (df.loc[df.signature_seen_before == 0, "is_new_error"] == 1).all()
    assert df["is_after_cutoff"].isin([0, 1]).all()


def test_threshold_meets_recall_target():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000)
    proba = np.clip(y * 0.4 + rng.random(1000) * 0.6, 0, 1)
    t = threshold_for_recall(y, proba, min_recall=0.85)
    assert evaluate(y, proba, t)["recall"] >= 0.85


def test_logistic_pipeline_learns_signal():
    X, y = split_xy(generate_tickets(2000, seed=3))
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    model = Pipeline([("prep", sparse_preprocessor()),
                      ("clf", LogisticRegression(class_weight="balanced", max_iter=2000))]).fit(X_tr, y_tr)
    assert roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]) > 0.75


def test_mlp_trains_and_outputs_probabilities():
    X, y = split_xy(generate_tickets(1200, seed=5))
    prep = dense_preprocessor().fit(X)
    Xd = prep.transform(X)
    model, history = train_mlp(Xd[:900], y.values[:900], Xd[900:], y.values[900:], epochs=5, patience=3)
    proba = predict_proba(model, Xd[900:])
    assert proba.min() >= 0 and proba.max() <= 1 and len(history) >= 1
