"""LLM provider adapters with retry, exponential backoff and cross-model failover.

A model chain is an ordered list of specs. A spec is either a bare Gemini model id
("gemini-3.8-flash") or "provider:model" ("openai_compat:llama-3.3-70b-versatile").
The first spec is the primary; the rest are failover targets.
"""
import os
import random
import re
import time

import httpx

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
RETRYABLE = {408, 429, 500, 502, 503, 504}
MAX_ATTEMPTS = 2          # per model, before failing over
COOLDOWN_S = 120          # skip a model that just failed over for this long (per server instance)
DAILY_QUOTA_S = 3600      # a 429 asking us to wait longer than this is a daily quota, not a rate limit
MAX_NET_ATTEMPTS = 6      # DNS/connection errors hit every model on the same host, so wait longer instead
MAX_WAIT_S = 30           # longest backoff we will sit through before failing over instead


class ProviderError(Exception):
    def __init__(self, msg, status=None, retry_after=None, kind=None):
        super().__init__(msg)
        self.status = status
        self.retry_after = retry_after
        self.kind = kind  # "overloaded" | "rate_limited" | "daily_quota" | None


class AllModelsFailed(Exception):
    pass


_cooldown = {}  # spec -> (time.monotonic() until which the model is deprioritised, reason)


def _retry_delay(resp):
    header = resp.headers.get("retry-after")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    m = re.search(r'"retryDelay"\s*:\s*"([\d.]+)s"', resp.text)
    return float(m.group(1)) if m else None


def _error_message(resp):
    try:
        return resp.json()["error"]["message"]
    except Exception:
        return resp.text


def _human_wait(seconds):
    return f"{seconds / 3600:.0f}h" if seconds >= 3600 else f"{seconds / 60:.0f}m" if seconds >= 60 else f"{seconds:.0f}s"


def _http_error(resp):
    """Turn a provider error response into a short, classified ProviderError."""
    status, delay, body = resp.status_code, _retry_delay(resp), resp.text
    if status == 429 and ("PerDay" in body or (delay or 0) > DAILY_QUOTA_S):
        wait = f", resets in ~{_human_wait(delay)}" if delay else ""
        return ProviderError(f"daily quota used up{wait}", status, delay, kind="daily_quota")
    if status == 429:
        return ProviderError("rate limited (per-minute quota)", status, delay, kind="rate_limited")
    if status in (500, 502, 503, 504):
        return ProviderError(f"overloaded (HTTP {status})", status, delay, kind="overloaded")
    msg = " ".join(_error_message(resp).split())
    return ProviderError(f"HTTP {status}: {msg[:160]}", status, delay)


def _gemini(model, system, user, *, json_mode, max_tokens, thinking, timeout):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ProviderError("GEMINI_API_KEY is not set on the server", status=401)
    gen = {"maxOutputTokens": max_tokens}
    if json_mode:
        gen["responseMimeType"] = "application/json"
    if thinking:
        gen["thinkingConfig"] = {"thinkingLevel": thinking}
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": gen,
    }
    r = httpx.post(GEMINI_URL.format(model=model), headers={"x-goog-api-key": key},
                   json=body, timeout=timeout)
    if r.status_code != 200:
        raise _http_error(r)
    data = r.json()
    candidates = data.get("candidates") or []
    if not candidates:
        reason = (data.get("promptFeedback") or {}).get("blockReason", "unknown")
        raise ProviderError(f"no candidates returned (block reason: {reason})", status=422)
    cand = candidates[0]
    if cand.get("finishReason") == "MAX_TOKENS":
        raise ProviderError("output truncated at max output tokens", status=413)
    parts = (cand.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    usage = data.get("usageMetadata") or {}
    return {
        "text": text,
        "tokens_in": usage.get("promptTokenCount", 0),
        "tokens_out": usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0),
    }


def _openai_compat(model, system, user, *, json_mode, max_tokens, thinking, timeout):
    """Any OpenAI-compatible endpoint (Groq, OpenRouter, OpenAI, ...) as an extra failover tier."""
    base = os.environ.get("OPENAI_COMPAT_BASE_URL")
    key = os.environ.get("OPENAI_COMPAT_API_KEY")
    if not base or not key:
        raise ProviderError("OPENAI_COMPAT_BASE_URL / OPENAI_COMPAT_API_KEY not set", status=401)
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": max_tokens,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    r = httpx.post(base.rstrip("/") + "/chat/completions",
                   headers={"Authorization": f"Bearer {key}"}, json=body, timeout=timeout)
    if r.status_code != 200:
        raise _http_error(r)
    data = r.json()
    choice = data["choices"][0]
    if choice.get("finish_reason") == "length":
        raise ProviderError("output truncated at max output tokens", status=413)
    usage = data.get("usage") or {}
    return {
        "text": choice["message"]["content"] or "",
        "tokens_in": usage.get("prompt_tokens", 0),
        "tokens_out": usage.get("completion_tokens", 0),
    }


ADAPTERS = {"gemini": _gemini, "openai_compat": _openai_compat}


def parse_spec(spec):
    provider, sep, model = spec.partition(":")
    if sep and provider in ADAPTERS:
        return provider, model
    return "gemini", spec


def cooling(spec):
    """Return the reason a model is benched, or None if it is healthy."""
    until, reason = _cooldown.get(spec, (0, None))
    return reason if until > time.monotonic() else None


def status(chain):
    """Per-model health for the dashboard."""
    now = time.monotonic()
    out = []
    for spec in chain:
        until, reason = _cooldown.get(spec, (0, None))
        out.append({"model": spec, "status": reason if until > now else "ok",
                    "for_s": round(until - now) if until > now else 0})
    return out


def _bench(spec, err):
    if err.kind == "daily_quota":
        secs = min(err.retry_after or 6 * 3600, 24 * 3600)
        _cooldown[spec] = (time.monotonic() + secs, "daily quota used up")
    else:
        _cooldown[spec] = (time.monotonic() + COOLDOWN_S, err.kind or "failing")


def _ordered(chain):
    """Healthy models first, then briefly benched ones, then quota-exhausted ones (order kept)."""
    rank = {None: 0, "daily quota used up": 2}
    return sorted(chain, key=lambda spec: rank.get(cooling(spec), 1))


def generate(chain, system, user, *, json_mode=True, max_tokens=8192, thinking=None,
             deadline, log, attempt_timeout=240):
    """Call the first healthy model in `chain`. `deadline` is a time.monotonic() value."""
    failures = []
    ordered = _ordered(chain)
    benched = [f"{parse_spec(s)[1]} ({cooling(s)})" for s in chain if cooling(s)]
    if benched:
        log("skipping " + ", ".join(benched))
    for idx, spec in enumerate(ordered):
        provider, model = parse_spec(spec)
        fn = ADAPTERS[provider]
        think = thinking if provider == "gemini" else None
        last = idx == len(ordered) - 1
        attempt, limit = 0, MAX_ATTEMPTS
        while attempt < limit:
            remaining = deadline - time.monotonic()
            if remaining < 5:
                raise AllModelsFailed("step deadline reached; " + "; ".join(failures))
            # The last model in the chain may use all remaining time; earlier ones are capped
            # so a single hanging call cannot consume the whole step.
            timeout = remaining - 2 if last else min(remaining - 2, attempt_timeout)
            try:
                res = fn(model, system, user, json_mode=json_mode, max_tokens=max_tokens,
                         thinking=think, timeout=timeout)
                res.update(provider=provider, model=model)
                _cooldown.pop(spec, None)
                return res
            except httpx.TimeoutException:
                err = ProviderError(f"no response within {timeout:.0f}s", status=408)
                limit = attempt + 1  # a slow model is likely overloaded: fail over, don't wait again
            except httpx.HTTPError as e:
                err = ProviderError(f"network error: {e}", status=503)
                limit = MAX_NET_ATTEMPTS
            except ProviderError as e:
                err = e

            if err.status == 401 and len(chain) == 1:
                raise AllModelsFailed(str(err))
            if err.status == 400 and think and "think" in str(err).lower():
                log(f"{model}: thinking config not supported, retrying without it")
                think = None
                continue
            attempt += 1
            if err.kind == "daily_quota":
                attempt = limit  # waiting will not help today
            if err.status in RETRYABLE and attempt < limit:
                delay = err.retry_after or (min(2 ** attempt, 16) + random.random())
                if delay <= min(MAX_WAIT_S, remaining - 10):
                    log(f"{model}: {err}, retrying in {delay:.0f}s")
                    time.sleep(delay)
                    continue
            failures.append(f"{model}: {err}")
            if err.status in RETRYABLE:
                _bench(spec, err)
            nxt = f" -> trying {parse_spec(ordered[idx + 1])[1]}" if idx + 1 < len(ordered) else ""
            log(f"{model}: {err}{nxt}")
            break
    raise AllModelsFailed("all models in chain failed; " + "; ".join(failures))
