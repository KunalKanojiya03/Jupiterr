"""Stateless, tamper-evident run ledger.

The browser drives the pipeline one stage per request, so there is no server-side
session. Instead each response carries the run's cumulative spend signed with an
HMAC; the next request must return it unmodified, which lets the server enforce the
per-run budget and QA retry ceiling without a database.
"""
import hashlib
import hmac
import json
import os
import time
import uuid


def _secret():
    explicit = os.environ.get("LEDGER_SECRET")
    if explicit:
        return explicit.encode()
    seed = os.environ.get("APP_PASSCODE", "") + "|" + os.environ.get("GEMINI_API_KEY", "")
    return hashlib.sha256(seed.encode()).digest()


def _sign(ledger):
    # Canonicalise numbers first: the browser's JSON turns 0.0 into 0, which must not break the signature.
    canon = {k: (f"{v:.6f}" if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
             for k, v in ledger.items()}
    payload = json.dumps(canon, sort_keys=True, separators=(",", ":")).encode()
    return hmac.new(_secret(), payload, hashlib.sha256).hexdigest()


def new_ledger():
    ledger = {"run_id": uuid.uuid4().hex[:12], "started": int(time.time()), "calls": 0,
              "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "dev_rounds": 0, "feedback_rounds": 0}
    return ledger, _sign(ledger)


def verify(ledger, sig):
    if not isinstance(ledger, dict) or not sig or not hmac.compare_digest(_sign(ledger), sig):
        return None
    return dict(ledger)


def record(ledger, tokens_in, tokens_out, cost):
    ledger["calls"] += 1
    ledger["tokens_in"] += tokens_in
    ledger["tokens_out"] += tokens_out
    ledger["cost_usd"] = round(ledger["cost_usd"] + cost, 6)
    return ledger, _sign(ledger)
