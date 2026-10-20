"""Optional: zero/few-shot LLM classifier, to compare against the trained models.

Runs only when OPENAI_API_KEY is set and --llm is passed. It classifies a small sample of test
tickets and reports F1 together with cost-relevant numbers (tokens, latency), because the real
question in production is not only "which is more accurate" but "which is accurate enough per dollar
and per millisecond".
"""
from __future__ import annotations

import json
import os
import time

FEW_SHOT = [
    ("AUT-401: User locked out after password reset. Known issue per runbook. Please check.", 0),
    ("SET-781: Settlement batch failed: duplicate value-date window. No known workaround. Urgent.", 1),
    ("PMT-422: Payment file rejected: mandatory field missing. Resubmitted successfully once.", 0),
    ("ACC-777: Account frozen without compliance flag. Customer funds impacted. Kindly assist.", 1),
]

SYSTEM = ("You triage banking support tickets. Decide if L1 support will need to escalate the ticket to "
          "L2/L3 engineers. Return JSON: {\"escalate\": true|false, \"reason\": \"<short>\"}.")


def llm_predict(tickets: list[dict], model: str = "gpt-4o-mini") -> tuple[list[int], dict]:
    from openai import OpenAI          # optional dependency

    client = OpenAI()
    shots = "\n".join(f"Ticket: {t}\nEscalate: {'true' if y else 'false'}" for t, y in FEW_SHOT)
    preds, tokens_in, tokens_out, started = [], 0, 0, time.perf_counter()
    for t in tickets:
        user = (f"Examples:\n{shots}\n\nTicket: {t['description']}\nService: {t['service']}, tier: "
                f"{t['customer_tier']}, seen before: {t['signature_seen_before']} times, customers affected: "
                f"{t['customers_affected']}, runbook: {bool(t['has_runbook'])}")
        r = client.chat.completions.create(
            model=model, temperature=0, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
        preds.append(int(bool(json.loads(r.choices[0].message.content).get("escalate"))))
        tokens_in += r.usage.prompt_tokens
        tokens_out += r.usage.completion_tokens
    elapsed = time.perf_counter() - started
    return preds, {"model": model, "n": len(tickets), "tokens_in": tokens_in, "tokens_out": tokens_out,
                   "avg_latency_s": round(elapsed / max(len(tickets), 1), 3)}


def available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))
