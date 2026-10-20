"""A small PyTorch neural network on the same (dense) features."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn


class EscalationMLP(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1),                       # one logit; sigmoid is applied in the loss / at predict time
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_mlp(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 60, batch_size: int = 256, lr: float = 1e-3, patience: int = 6,
              seed: int = 42) -> tuple[EscalationMLP, list[dict]]:
    torch.manual_seed(seed)
    model = EscalationMLP(X_train.shape[1])
    pos_weight = torch.tensor((y_train == 0).sum() / max((y_train == 1).sum(), 1), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)        # weights the rarer escalated class up
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)   # L2 regularisation

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    history, best_ap, best_state, bad_epochs = [], -1.0, None, 0
    gen = torch.Generator().manual_seed(seed)

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(Xt), generator=gen)
        total = 0.0
        for start in range(0, len(Xt), batch_size):
            idx = order[start:start + batch_size]
            optimiser.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()                                       # backpropagation
            optimiser.step()                                      # gradient descent step (Adam)
            total += loss.item() * len(idx)
        val_ap = average_precision_score(y_val, predict_proba(model, Xv))
        history.append({"epoch": epoch, "train_loss": round(total / len(Xt), 4), "val_pr_auc": round(val_ap, 4)})
        if val_ap > best_ap + 1e-4:                               # early stopping on validation PR-AUC
            best_ap, best_state, bad_epochs = val_ap, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    model.load_state_dict(best_state)
    return model, history


def predict_proba(model: EscalationMLP, X) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        X = X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
        return torch.sigmoid(model(X)).numpy()
