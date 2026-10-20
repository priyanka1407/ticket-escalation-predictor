"""Synthetic support-ticket dataset.

Real tickets contain customer data and cannot be published, so this module generates
realistic, reproducible tickets for a banking support desk. The escalation label comes
from a hidden "complexity" score built from the same signals an L2 engineer would weigh
(unknown error, customer impact, cut-off pressure, missing runbook, wording of the ticket),
plus noise, so no single feature gives the answer away.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# error code -> (service, how hard it usually is to fix at L1 [0..1], short description)
ERROR_CODES = {
    "PMT-422": ("payments", 0.15, "Payment file rejected: mandatory field missing"),
    "PMT-503": ("payments", 0.55, "Payment gateway returned 503 during submission"),
    "SET-210": ("settlement", 0.35, "Settlement file late: batch scheduler delay"),
    "SET-781": ("settlement", 0.85, "Settlement batch failed: duplicate value-date window"),
    "CRD-104": ("cards", 0.25, "Card authorisation declined at limit check"),
    "CRD-900": ("cards", 0.70, "Card transactions not posted to ledger"),
    "ACC-301": ("accounts", 0.30, "Balance mismatch on customer statement"),
    "ACC-777": ("accounts", 0.80, "Account frozen without compliance flag"),
    "AUT-401": ("auth", 0.10, "User locked out after password reset"),
    "AUT-555": ("auth", 0.60, "SSO token rejected by downstream service"),
}
CODE_WEIGHTS = np.array([0.16, 0.08, 0.10, 0.05, 0.13, 0.06, 0.12, 0.04, 0.18, 0.08])

HARD_PHRASES = ["data mismatch between systems", "customer funds impacted", "duplicate posting suspected",
                "regulatory deadline today", "no known workaround", "affects multiple customers"]
EASY_PHRASES = ["customer asked for status update", "known issue per runbook", "resubmitted successfully once",
                "single customer affected", "workaround applied"]
NOISE_PHRASES = ["please check", "urgent", "kindly assist", "screenshot attached", "seen since morning",
                 "raised by relationship manager", "customer called twice"]

CHANNELS = ["email", "phone", "portal", "monitoring_alert"]
TIERS = ["retail", "corporate", "private"]


def generate_tickets(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    codes = list(ERROR_CODES)
    code = rng.choice(codes, size=n, p=CODE_WEIGHTS / CODE_WEIGHTS.sum())
    service = np.array([ERROR_CODES[c][0] for c in code])
    difficulty = np.array([ERROR_CODES[c][1] for c in code])

    channel = rng.choice(CHANNELS, size=n, p=[0.40, 0.20, 0.25, 0.15])
    tier = rng.choice(TIERS, size=n, p=[0.55, 0.30, 0.15])
    hour = rng.integers(0, 24, size=n)
    weekday = rng.integers(0, 7, size=n)                       # 0 = Monday
    money_service = np.isin(service, ["payments", "settlement", "cards"])
    amount = np.where(money_service, np.round(rng.lognormal(8.5, 1.4, size=n), 2), 0.0)

    # Easy errors have been seen (and fixed) many times before; hard ones rarely
    seen_before = rng.poisson(lam=np.clip(40 * (1 - difficulty) ** 2, 0.3, None))
    has_runbook = (seen_before > 5) & (rng.random(n) < 0.8)
    customers_affected = 1 + rng.geometric(p=np.clip(0.75 - 0.4 * difficulty, 0.2, 0.9), size=n) - 1
    logs_attached = rng.random(n) < 0.55
    l1_experience_months = np.round(rng.gamma(shape=2.0, scale=10.0, size=n), 1)
    reopened = rng.random(n) < 0.08

    # Ticket wording: hard phrases appear more often on hard problems
    has_hard = rng.random(n) < (0.10 + 0.45 * difficulty)
    has_easy = (rng.random(n) < (0.50 - 0.35 * difficulty)) & ~has_hard
    descriptions = []
    for i in range(n):
        parts = [f"{code[i]}: {ERROR_CODES[code[i]][2]}."]
        if has_hard[i]:
            parts.append(rng.choice(HARD_PHRASES).capitalize() + ".")
        if has_easy[i]:
            parts.append(rng.choice(EASY_PHRASES).capitalize() + ".")
        parts.append(rng.choice(NOISE_PHRASES).capitalize() + ".")
        descriptions.append(" ".join(parts))

    # Hidden complexity score -> probability of escalation
    after_cutoff = hour >= 16
    z = (2.6 * difficulty
         + 1.0 * has_hard - 0.9 * has_easy
         - 0.30 * np.log1p(seen_before) - 0.6 * has_runbook
         + 0.5 * (after_cutoff & np.isin(service, ["payments", "settlement"]))
         + 0.5 * (tier == "private")
         + 0.35 * np.log1p(customers_affected)
         - 0.015 * l1_experience_months
         + 0.5 * reopened
         + 0.3 * (channel == "monitoring_alert")
         + rng.normal(0, 0.6, size=n)
         - 1.35)
    escalated = (rng.random(n) < 1 / (1 + np.exp(-2.2 * z))).astype(int)

    return pd.DataFrame({
        "ticket_id": [f"SUP-{100000 + i}" for i in range(n)],
        "error_code": code, "service": service, "channel": channel, "customer_tier": tier,
        "hour": hour, "weekday": weekday, "amount": amount,
        "signature_seen_before": seen_before, "has_runbook": has_runbook.astype(int),
        "customers_affected": customers_affected, "logs_attached": logs_attached.astype(int),
        "l1_experience_months": l1_experience_months, "reopened": reopened.astype(int),
        "description": descriptions, "escalated": escalated,
    })


if __name__ == "__main__":
    df = generate_tickets()
    print(df.head(3).to_string())
    print(f"\nrows={len(df)}  escalation rate={df.escalated.mean():.1%}")
