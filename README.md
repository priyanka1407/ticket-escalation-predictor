# Ticket Escalation Predictor

Predicts whether a banking support ticket will be **escalated from L1 to L2/L3**, so L1 can route hard tickets early and handle the rest confidently. A classic supervised-learning project end to end: synthetic data, feature engineering, model selection with cross-validation, a PyTorch neural network, threshold tuning for a business goal, explainability, and an optional LLM few-shot baseline for comparison.

## Results (held-out test set, 2,000 tickets)

All model selection and threshold tuning used the training set only (5-fold stratified CV); the test set was touched once.

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| **Logistic regression (TF-IDF + features)**, selected | 0.930 | **0.824** | 0.649 | 0.831 | **0.729** |
| PyTorch MLP (LSA text + features) | 0.927 | 0.821 | 0.616 | 0.849 | 0.714 |
| Random forest | 0.914 | 0.800 | 0.588 | 0.822 | 0.685 |
| Histogram gradient boosting | 0.915 | 0.792 | 0.569 | 0.847 | 0.681 |
| Rule of thumb (no ML) | – | – | 0.436 | 0.751 | 0.552 |
| Dummy (predict base rate) | 0.5 | 0.218 | – | 0 | 0 |

Thresholds were tuned for **recall ≥ 0.85** on out-of-fold training predictions (a missed escalation is worse than a false alarm). Full tables, permutation importance and model weights: [reports/results.md](reports/results.md).

![Precision-recall curves](reports/pr_curves.png)

**Takeaways**

- The trained models beat the rule of thumb by **+0.18 F1** and catch more escalations with far fewer false alarms.
- The simplest model won: logistic regression on sparse TF-IDF features slightly beat random forest, gradient boosting and the neural network. The signal is mostly additive (wording, error code, history), and the tree and MLP models only see a 40-dimensional SVD compression of the text.
- Ticket wording carries most of the signal (permutation importance), followed by error code, how often the error signature was seen before, and whether a runbook exists.
- With the threshold chosen on training data, logistic regression reached 0.831 recall on the test set, just under the 0.85 target: a reminder that thresholds should be monitored and re-tuned on fresh data.

## How it works

```
generate_tickets()  →  engineer()  →  preprocessors  →  models  →  CV + threshold  →  test + reports
 10k synthetic         cut-off,        TF-IDF (sparse)   LogReg      5-fold PR-AUC      metrics.json
 banking tickets       log counts,     TF-IDF+SVD (LSA)  RF, HGB     recall-targeted    results.md
                       new-error flag  one-hot, scaling  PyTorch MLP threshold          PR curves, confusion
```

| Step | Detail | File |
| --- | --- | --- |
| Data | 10 error codes across payments, settlement, cards, accounts, auth; channel, customer tier, hour, amount, how often the error signature was seen, runbook, customers affected, L1 experience, reopened flag, free-text description. The label comes from a hidden complexity score plus noise | `src/escalation/data.py` |
| Feature engineering | After-cut-off flag (payments pressure), weekend, `log1p` of skewed counts and amounts, new-error flag, text length | `src/escalation/features.py` |
| Preprocessing | `ColumnTransformer`: TF-IDF uni/bi-grams, one-hot (unknown categories ignored), standard scaling; dense variant uses TruncatedSVD (LSA) for trees and the MLP | `src/escalation/features.py` |
| Model selection | Dummy, logistic regression (C tuned by `GridSearchCV`), random forest, histogram gradient boosting; class weights for the 22% minority class; ranked by PR-AUC | `src/escalation/train.py` |
| Neural network | PyTorch MLP 64-32-1, ReLU, dropout, `BCEWithLogitsLoss` with `pos_weight`, Adam with weight decay, early stopping on validation PR-AUC | `src/escalation/mlp.py` |
| Threshold | Highest precision with recall ≥ 0.85, chosen on out-of-fold predictions | `src/escalation/metrics.py` |
| Explainability | Permutation importance on raw inputs; logistic regression coefficients | `src/escalation/train.py` |
| LLM baseline (optional) | `gpt-4o-mini` few-shot JSON classifier on a sample; reports F1, tokens and latency | `src/escalation/llm_baseline.py` |

## Run it

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Linux/macOS: .venv/bin/pip
.venv/Scripts/python run.py                        # writes data/ and reports/
.venv/Scripts/python -m pytest -q tests            # 5 tests
```

Optional LLM comparison (not run for the published results):

```bash
pip install openai
export OPENAI_API_KEY=...        # PowerShell: $env:OPENAI_API_KEY="..."
python run.py --llm --llm-sample 200
```

## ML concepts used

| Concept | Where |
| --- | --- |
| Logistic regression: sigmoid of a weighted sum, trained by minimising log loss | Selected model; coefficients read as log-odds |
| Regularisation (C in logistic regression, weight decay, dropout) | Prevents overfitting to rare n-grams |
| Class imbalance (`class_weight`, `pos_weight`) | 22% positives; PR-AUC preferred over accuracy and ROC-AUC |
| Cross-validation and model selection | 5-fold stratified on training data only, no test-set leakage |
| Gradient descent, backpropagation, Adam | PyTorch MLP training loop |
| Dimensionality reduction (TruncatedSVD / LSA) | Dense text features for trees and the MLP |
| Precision, recall, F1, ROC-AUC, PR-AUC, Brier score | `metrics.py`; threshold chosen for a recall target |
| Permutation importance | Model-agnostic explanation of which inputs matter |

## Limitations and next steps

- **Synthetic data.** Real tickets contain customer data; on real data, add PII masking before feature extraction and expect noisier labels.
- Bigrams currently cross sentence boundaries (e.g. "reset no"); sentence-aware tokenisation would clean the vocabulary.
- Add probability calibration (isotonic or Platt) if scores are shown to users as risk percentages.
- Monitor recall and drift in production (new error codes after releases) and re-tune the threshold.
- Combine with the RAG assistant: tickets predicted to escalate get the assistant's suggested resolution and a pre-filled L2 summary.
