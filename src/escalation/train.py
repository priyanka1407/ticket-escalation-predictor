"""End-to-end training, model selection, threshold tuning and reporting.

    python run.py                 # generate data, compare models, write reports/
    python run.py --llm           # also run the optional LLM few-shot baseline (needs OPENAI_API_KEY)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, precision_recall_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from . import llm_baseline
from .data import HARD_PHRASES, generate_tickets
from .features import dense_preprocessor, engineer, sparse_preprocessor, split_xy
from .metrics import evaluate, threshold_for_recall
from .mlp import predict_proba as mlp_proba
from .mlp import train_mlp

SEED = 42
MIN_RECALL = 0.85


def candidate_models() -> dict[str, Pipeline]:
    return {
        "dummy_prior": Pipeline([("prep", sparse_preprocessor()), ("clf", DummyClassifier(strategy="prior"))]),
        "logistic_regression": Pipeline([("prep", sparse_preprocessor()),
                                         ("clf", LogisticRegression(class_weight="balanced", max_iter=3000))]),
        "random_forest": Pipeline([("prep", dense_preprocessor()),
                                   ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                                  class_weight="balanced_subsample",
                                                                  n_jobs=-1, random_state=SEED))]),
        "hist_gradient_boosting": Pipeline([("prep", dense_preprocessor()),
                                            ("clf", HistGradientBoostingClassifier(learning_rate=0.08, max_iter=300,
                                                                                   class_weight="balanced",
                                                                                   random_state=SEED))]),
    }


def rule_based(X: pd.DataFrame) -> np.ndarray:
    """What an L1 team lead might write as a rule of thumb, as a non-ML baseline."""
    hard_text = X["description"].str.lower().apply(lambda s: any(p in s for p in HARD_PHRASES))
    return ((X["is_new_error"] == 1) | (np.expm1(X["log_customers_affected"]) >= 3) | hard_text).astype(int).values


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default="reports")
    ap.add_argument("--llm", action="store_true", help="also evaluate an LLM few-shot classifier")
    ap.add_argument("--llm-sample", type=int, default=200)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Data: hold out a test set that is never used for any decision
    df = generate_tickets(args.n, args.seed)
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/tickets.csv", index=False)
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=args.seed)
    print(f"rows={len(df)} escalation_rate={y.mean():.1%} train={len(X_train)} test={len(X_test)}")

    # 2) Model selection with 5-fold stratified cross-validation on the training set only
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    models = candidate_models()
    grid = GridSearchCV(models["logistic_regression"], {"clf__C": [0.1, 0.3, 1.0, 3.0]},
                        scoring="average_precision", cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)
    models["logistic_regression"] = grid.best_estimator_
    print(f"logistic_regression best C={grid.best_params_['clf__C']}")

    cv_rows = []
    for name, model in models.items():
        s = cross_validate(model, X_train, y_train, cv=cv, n_jobs=-1,
                           scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1"})
        cv_rows.append({"model": name,
                        **{f"{m}_mean": round(s[f"test_{m}"].mean(), 3) for m in ("roc_auc", "pr_auc", "f1")},
                        **{f"{m}_std": round(s[f"test_{m}"].std(), 3) for m in ("roc_auc", "pr_auc", "f1")}})
        print(f"CV {name:24s} PR-AUC {cv_rows[-1]['pr_auc_mean']:.3f} ± {cv_rows[-1]['pr_auc_std']:.3f}")
    cv_table = pd.DataFrame(cv_rows).sort_values("pr_auc_mean", ascending=False)
    best_name = cv_table.iloc[0]["model"]
    print(f"selected: {best_name}")

    # 3) Threshold tuning on out-of-fold training predictions (never on the test set)
    test_results, curves = {}, {}
    for name in ("logistic_regression", "random_forest", "hist_gradient_boosting"):
        oof = cross_val_predict(clone(models[name]), X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        thr = threshold_for_recall(y_train, oof, MIN_RECALL)
        fitted = clone(models[name]).fit(X_train, y_train)
        proba = fitted.predict_proba(X_test)[:, 1]
        test_results[name] = evaluate(y_test, proba, thr)
        curves[name] = proba
        models[name] = fitted

    # 4) PyTorch MLP on the dense features, early-stopped on a validation split
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, stratify=y_train,
                                                random_state=args.seed)
    prep = dense_preprocessor().fit(X_tr)
    net, history = train_mlp(prep.transform(X_tr), y_tr.values, prep.transform(X_val), y_val.values, seed=args.seed)
    val_proba = mlp_proba(net, prep.transform(X_val))
    mlp_thr = threshold_for_recall(y_val, val_proba, MIN_RECALL)
    curves["pytorch_mlp"] = mlp_proba(net, prep.transform(X_test))
    test_results["pytorch_mlp"] = evaluate(y_test, curves["pytorch_mlp"], mlp_thr)
    print(f"MLP stopped after {len(history)} epochs, best val PR-AUC {max(h['val_pr_auc'] for h in history):.3f}")

    # 5) Non-ML baselines
    rule_pred = rule_based(X_test)
    test_results["rule_based"] = {"f1": round(f1_score(y_test, rule_pred), 3),
                                  "recall": round(float((rule_pred[y_test.values == 1]).mean()), 3),
                                  "precision": round(float(y_test.values[rule_pred == 1].mean()), 3)}
    test_results["dummy_prior"] = {"f1": 0.0, "recall": 0.0, "precision": 0.0,
                                   "pr_auc": round(float(y_test.mean()), 3), "roc_auc": 0.5}

    # 6) Optional LLM few-shot baseline on a sample
    llm_info = None
    if args.llm and llm_baseline.available():
        sample = df.loc[X_test.index].head(args.llm_sample)
        preds, llm_info = llm_baseline.llm_predict(sample.to_dict("records"))
        yt = y_test.loc[sample.index].values
        test_results["llm_few_shot"] = {"f1": round(f1_score(yt, preds), 3),
                                        "recall": round(float(np.array(preds)[yt == 1].mean()), 3), **llm_info}
    elif args.llm:
        print("LLM baseline skipped: OPENAI_API_KEY not set")

    # 7) Explainability: which raw inputs matter (permutation importance on the selected tree/linear model)
    explain_name = best_name if best_name in models and best_name != "dummy_prior" else "logistic_regression"
    sample_idx = X_test.sample(min(2000, len(X_test)), random_state=args.seed).index
    pi = permutation_importance(models[explain_name], X_test.loc[sample_idx], y_test.loc[sample_idx],
                                scoring="average_precision", n_repeats=5, random_state=args.seed, n_jobs=-1)
    importance = (pd.DataFrame({"feature": X_test.columns, "importance": pi.importances_mean})
                  .sort_values("importance", ascending=False).round(4))
    lr = models["logistic_regression"]
    names = lr.named_steps["prep"].get_feature_names_out()
    coefs = lr.named_steps["clf"].coef_[0]
    order = np.argsort(coefs)
    lr_terms = {"towards_escalation": [(names[i], round(coefs[i], 2)) for i in order[::-1][:10]],
                "towards_l1_resolution": [(names[i], round(coefs[i], 2)) for i in order[:10]]}

    # 8) Reports
    plt.figure(figsize=(6.5, 5))
    for name, proba in curves.items():
        p, r, _ = precision_recall_curve(y_test, proba)
        plt.plot(r, p, label=f"{name} (PR-AUC {test_results[name]['pr_auc']})")
    plt.axhline(y_test.mean(), ls="--", c="grey", label=f"no-skill ({y_test.mean():.2f})")
    plt.axvline(MIN_RECALL, ls=":", c="black", label=f"recall target {MIN_RECALL}")
    plt.xlabel("Recall (share of escalations caught)"); plt.ylabel("Precision"); plt.legend(fontsize=8)
    plt.title("Precision-recall on the held-out test set"); plt.tight_layout()
    plt.savefig(out / "pr_curves.png", dpi=140); plt.close()

    final = best_name if best_name in curves else "logistic_regression"
    pred = (curves[final] >= test_results[final]["threshold"]).astype(int)
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["L1 resolves", "escalated"], cmap="Blues")
    plt.title(f"{final} at threshold {test_results[final]['threshold']}"); plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=140); plt.close()

    summary = {"rows": len(df), "escalation_rate": round(float(y.mean()), 3), "selected_model": best_name,
               "logreg_best_C": grid.best_params_["clf__C"], "min_recall_target": MIN_RECALL,
               "cv": cv_table.to_dict("records"), "test": test_results, "mlp_history": history,
               "permutation_importance": importance.to_dict("records"), "logreg_top_terms": lr_terms}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    write_markdown(out / "results.md", summary, cv_table, test_results, importance, lr_terms, llm_info)
    print(f"reports written to {out}/")
    return summary


def write_markdown(path, summary, cv_table, test_results, importance, lr_terms, llm_info) -> None:
    lines = [f"# Results\n",
             f"Dataset: {summary['rows']} synthetic tickets, escalation rate {summary['escalation_rate']:.1%}. "
             f"80/20 stratified split; all model selection and threshold tuning on the training set only.\n",
             "## Cross-validation (5-fold, training set)\n",
             "| Model | ROC-AUC | PR-AUC | F1 @0.5 |", "| --- | --- | --- | --- |"]
    for r in cv_table.to_dict("records"):
        lines.append(f"| {r['model']} | {r['roc_auc_mean']} ± {r['roc_auc_std']} | {r['pr_auc_mean']} ± "
                     f"{r['pr_auc_std']} | {r['f1_mean']} ± {r['f1_std']} |")
    lines += [f"\nSelected by PR-AUC: **{summary['selected_model']}** (logistic regression C = {summary['logreg_best_C']}).\n",
              f"## Held-out test set (threshold tuned for recall ≥ {summary['min_recall_target']})\n",
              "| Model | Threshold | ROC-AUC | PR-AUC | Precision | Recall | F1 | Brier | FN (missed) | FP |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name in ("logistic_regression", "random_forest", "hist_gradient_boosting", "pytorch_mlp"):
        m = test_results[name]
        lines.append(f"| {name} | {m['threshold']} | {m['roc_auc']} | {m['pr_auc']} | {m['precision']} | "
                     f"{m['recall']} | {m['f1']} | {m['brier']} | {m['fn']} | {m['fp']} |")
    rb = test_results["rule_based"]
    lines += [f"| rule_based (no ML) | – | – | – | {rb['precision']} | {rb['recall']} | {rb['f1']} | – | – | – |",
              f"| dummy (predict base rate) | – | 0.5 | {test_results['dummy_prior']['pr_auc']} | – | 0 | 0 | – | – | – |"]
    if llm_info:
        m = test_results["llm_few_shot"]
        lines.append(f"\nLLM few-shot ({m['model']}, n={m['n']}): F1 {m['f1']}, recall {m['recall']}, "
                     f"tokens in/out {m['tokens_in']}/{m['tokens_out']}, avg latency {m['avg_latency_s']} s.")
    lines += ["\n![Precision-recall curves](pr_curves.png)\n", "![Confusion matrix](confusion_matrix.png)\n",
              "## What drives escalation (permutation importance, PR-AUC drop)\n",
              "| Input | Importance |", "| --- | --- |"]
    lines += [f"| {r['feature']} | {r['importance']} |" for r in importance.head(10).to_dict("records")]
    lines += ["\n## Logistic regression: strongest signals\n", "| Towards escalation | Weight | Towards L1 resolution | Weight |",
              "| --- | --- | --- | --- |"]
    for (a, wa), (b, wb) in zip(lr_terms["towards_escalation"], lr_terms["towards_l1_resolution"]):
        lines.append(f"| {a} | {wa} | {b} | {wb} |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
