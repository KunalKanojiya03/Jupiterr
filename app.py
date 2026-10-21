"""Jupiterr - Autonomous Multi-Agent AI Council (POC).

The browser drives the council one stage per HTTPS request (manager -> architect ->
developer -> qa, looping developer/qa on rejection). Each request is short enough for
serverless limits, API keys never leave the server, and a signed ledger carries the
run's spend between requests so the budget cap is enforced server-side.
"""
import hmac
import io
import json
import os
import queue
import re
import threading
import time
import zipfile
from pathlib import Path
from typing import Literal, Optional

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent


def _load_dotenv():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

from council import agents, ledger as ledgers, providers, validate  # noqa: E402

RUN_BUDGET_USD = float(os.environ.get("RUN_BUDGET_USD", "0.50"))
MAX_QA_ROUNDS = int(os.environ.get("MAX_QA_ROUNDS", "2"))
MAX_FEEDBACK_ROUNDS = int(os.environ.get("MAX_FEEDBACK_ROUNDS", "6"))
STEP_DEADLINE_S = float(os.environ.get("STEP_DEADLINE_S", "270"))
# Estimated USD per 1M tokens. Placeholders: set these from your Gemini pricing page.
PRICE_IN = float(os.environ.get("PRICE_PER_MTOK_IN", "0.50"))
PRICE_OUT = float(os.environ.get("PRICE_PER_MTOK_OUT", "3.00"))
MAX_BRIEF_CHARS = 20000

app = FastAPI(title="Jupiterr AI Council")


def require_passcode(x_passcode: Optional[str] = Header(default=None)):
    expected = os.environ.get("APP_PASSCODE")
    if not expected:
        if os.environ.get("VERCEL"):
            raise HTTPException(503, "APP_PASSCODE must be configured before this deployment can be used")
        return
    if not x_passcode or not hmac.compare_digest(x_passcode.encode(), expected.encode()):
        raise HTTPException(401, "invalid passcode")


class FileItem(BaseModel):
    path: str
    language: Optional[str] = None
    content: str = ""


class StepRequest(BaseModel):
    stage: Literal["manager", "architect", "developer", "qa"]
    brief: str = Field(min_length=10, max_length=MAX_BRIEF_CHARS)
    plan: Optional[dict] = None
    architecture: Optional[dict] = None
    files: Optional[list[FileItem]] = None
    qa: Optional[dict] = None
    ledger: Optional[dict] = None
    sig: Optional[str] = None
    user_feedback: Optional[str] = Field(default=None, max_length=8000)


class ZipRequest(BaseModel):
    project_name: str = "council-project"
    files: list[FileItem]
    plan: Optional[dict] = None
    architecture: Optional[dict] = None
    qa: Optional[dict] = None
    ledger: Optional[dict] = None


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "auth_required": bool(os.environ.get("APP_PASSCODE")) or bool(os.environ.get("VERCEL")),
        "gemini_key_configured": bool(os.environ.get("GEMINI_API_KEY")),
        "models": {role: agents.model_chain(role) for role in agents.ROLES},
        "model_status": providers.status(agents.model_chain("developer")),
        "budget_usd": RUN_BUDGET_USD,
        "max_qa_rounds": MAX_QA_ROUNDS,
        "price_per_mtok": {"in": PRICE_IN, "out": PRICE_OUT},
    }


@app.get("/api/auth", dependencies=[Depends(require_passcode)])
def auth_check():
    return {"ok": True}


@app.get("/api/models", dependencies=[Depends(require_passcode)])
def list_models():
    """Diagnostic: which Gemini models this API key can call."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise HTTPException(503, "GEMINI_API_KEY is not set")
    r = httpx.get("https://generativelanguage.googleapis.com/v1beta/models",
                  headers={"x-goog-api-key": key}, params={"pageSize": 200}, timeout=20)
    if r.status_code != 200:
        raise HTTPException(502, f"Gemini ListModels failed: HTTP {r.status_code}")
    return sorted(m["name"].removeprefix("models/") for m in r.json().get("models", [])
                  if "generateContent" in m.get("supportedGenerationMethods", []))


def _sanitize_files(files):
    out, seen = [], set()
    for f in files:
        p = validate.safe_path(f.get("path", "") if isinstance(f, dict) else f.path)
        if not p or p in seen:
            continue
        seen.add(p)
        d = f if isinstance(f, dict) else f.model_dump()
        out.append({"path": p, "language": d.get("language"), "content": str(d.get("content", ""))})
    return out


def _prepare(req: StepRequest):
    """Checks that must fail with a real HTTP status, before the response starts streaming."""
    # Ledger: a manager call opens a new run; every later stage must present a valid one.
    if req.stage == "manager":
        led, _ = ledgers.new_ledger()
    else:
        led = ledgers.verify(req.ledger, req.sig)
        if led is None:
            raise HTTPException(400, "missing or tampered run ledger; start a new run")
        if req.plan is None:
            raise HTTPException(400, f"stage '{req.stage}' requires the plan")
        if req.stage in ("developer", "qa") and req.architecture is None:
            raise HTTPException(400, f"stage '{req.stage}' requires the architecture")
        if req.stage == "qa" and not req.files:
            raise HTTPException(400, "stage 'qa' requires files")
    if led["cost_usd"] >= RUN_BUDGET_USD:
        raise HTTPException(402, f"run budget of ${RUN_BUDGET_USD:.2f} exhausted (spent ${led['cost_usd']:.4f})")
    if req.stage == "developer" and req.user_feedback:
        # Fixing a problem the user reported has its own allowance, separate from QA revisions.
        if led.get("feedback_rounds", 0) >= MAX_FEEDBACK_ROUNDS:
            raise HTTPException(409, f"fix limit reached ({MAX_FEEDBACK_ROUNDS} developer passes on reported "
                                     "problems for this run); start a new run with the problem in the brief")
        led["feedback_rounds"] = led.get("feedback_rounds", 0) + 1
    elif req.stage == "developer":
        if led["dev_rounds"] >= MAX_QA_ROUNDS + 1:
            raise HTTPException(409, f"retry ceiling reached ({MAX_QA_ROUNDS} QA revisions)")
        led["dev_rounds"] += 1
    return led


def _run(req: StepRequest, led, emit, cancel):
    started = time.monotonic()
    deadline = started + STEP_DEADLINE_S
    events = []
    last_progress = [0.0]

    def log(msg):
        events.append(msg)
        emit({"type": "log", "msg": msg})

    def on_progress(model, text):
        now = time.monotonic()
        if text:  # throttle token updates; "waiting on model" notices always go through
            if now - last_progress[0] < 0.8:
                return
            last_progress[0] = now
        paths = PATH_RE.findall(text)
        emit({"type": "progress", "model": model, "chars": len(text),
              "file": paths[-1] if paths else None, "files_done": max(len(paths) - 1, 0)})

    files_in = _sanitize_files(req.files or [])
    req.files = [FileItem(**f) for f in files_in]
    validation = None
    if req.stage == "qa":
        validation = validate.validate_files(files_in, (req.architecture or {}).get("file_tree"))

    role = agents.ROLES[req.stage]
    chain = agents.model_chain(req.stage)
    user_msg = agents.build_user_message(req.stage, req, validation)
    tokens_in = tokens_out = 0
    output, used_model = None, None

    for attempt in range(2):  # one repair attempt if the JSON is malformed
        try:
            res = providers.generate(chain, agents.SYSTEM[req.stage], user_msg, json_mode=True,
                                     max_tokens=role["max_tokens"], thinking=role["thinking"],
                                     deadline=deadline, log=log,
                                     attempt_timeout=role["attempt_timeout"],
                                     on_progress=on_progress, cancel=cancel,
                                     schema=agents.SCHEMAS[req.stage])
        except providers.AllModelsFailed as e:
            raise HTTPException(503, {"message": str(e), "events": events})
        tokens_in += res["tokens_in"]
        tokens_out += res["tokens_out"]
        used_model = f"{res['provider']}:{res['model']}"
        try:
            parsed = agents.parse_json(res["text"])
            problem = agents.check_shape(req.stage, parsed)
        except (json.JSONDecodeError, ValueError) as e:
            parsed, problem = None, f"invalid JSON ({e})"
        if not problem:
            output = parsed
            break
        log(f"{res['model']}: {problem} - asking for a corrected response")
        user_msg += (f"\n\n=== FORMAT ERROR ===\nYour previous response was rejected: {problem}. "
                     "Respond again with ONLY the JSON object in the exact required shape.")
    cost = tokens_in / 1e6 * PRICE_IN + tokens_out / 1e6 * PRICE_OUT
    led, sig = ledgers.record(led, tokens_in, tokens_out, cost)
    if output is None:
        raise HTTPException(502, {"message": "model did not return valid structured output", "events": events,
                                  "ledger": led, "sig": sig})

    if req.stage == "developer":
        output["files"] = _sanitize_files(output.get("files") or [])
        output["validation"] = validate.validate_files(output["files"], (req.architecture or {}).get("file_tree"))
    if req.stage == "qa":
        output["validation"] = validation
        # Deterministic gate: the model cannot approve code that failed parsing or has blockers.
        blockers = [i for i in output.get("issues", []) if i.get("severity") in ("critical", "major")]
        hard_fail = validation["summary"]["fail"] > 0 or validation["missing"]
        if output.get("verdict") == "approve" and (blockers or hard_fail):
            output["verdict"] = "reject"
            output["gate_override"] = "auto-rejected: validation failures or critical/major issues present"
        output["rounds_left"] = MAX_QA_ROUNDS + 1 - led["dev_rounds"]

    return {
        "stage": req.stage,
        "output": output,
        "ledger": led,
        "sig": sig,
        "telemetry": {
            "role": role["title"], "model": used_model, "events": events,
            "tokens_in": tokens_in, "tokens_out": tokens_out, "cost_usd": round(cost, 6),
            "elapsed_s": round(time.monotonic() - started, 1),
        },
        "budget_usd": RUN_BUDGET_USD,
    }


PATH_RE = re.compile(r'"path"\s*:\s*"([^"]+)"')


@app.post("/api/step", dependencies=[Depends(require_passcode)])
def step(req: StepRequest):
    """Runs one council stage, streaming NDJSON events: log, progress, ping, then result or error.

    Streaming keeps the dashboard live during multi-minute stages, and the heartbeat stops
    corporate proxies from dropping a connection that would otherwise sit idle.
    """
    led = _prepare(req)
    events, cancel = queue.Queue(), threading.Event()

    def worker():
        try:
            events.put({"type": "result", "data": _run(req, led, events.put, cancel)})
        except HTTPException as e:
            events.put({"type": "error", "status": e.status_code, "detail": e.detail})
        except providers.Cancelled:
            pass
        except Exception as e:  # never leave the client hanging
            events.put({"type": "error", "status": 500, "detail": f"internal error: {e}"})
        finally:
            events.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def stream():
        try:
            while True:
                try:
                    item = events.get(timeout=10)
                except queue.Empty:
                    yield '{"type":"ping"}\n'
                    continue
                if item is None:
                    return
                yield json.dumps(item) + "\n"
        finally:
            cancel.set()  # stops a still-streaming model call if the client went away

    return StreamingResponse(stream(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


def _report_md(req: ZipRequest):
    plan, arch, qa = req.plan or {}, req.architecture or {}, req.qa or {}
    lines = [f"# {req.project_name} - Council Report", "", plan.get("summary", ""), ""]
    if plan.get("acceptance_criteria"):
        lines += ["## Acceptance criteria", *[f"- {c}" for c in plan["acceptance_criteria"]], ""]
    if plan.get("tasks"):
        lines += ["## Tasks", *[f"- **{t.get('id')}** {t.get('title')}: {t.get('detail')}" for t in plan["tasks"]], ""]
    for key, title in (("overview_md", "Architecture"), ("interfaces_md", "Interfaces"),
                       ("data_models_md", "Data models")):
        if arch.get(key):
            lines += [f"## {title}", arch[key], ""]
    if arch.get("run_instructions"):
        lines += ["## Run instructions", arch["run_instructions"], ""]
    if qa:
        lines += [f"## QA verdict: {qa.get('verdict', '?').upper()} (score {qa.get('score', '?')})",
                  qa.get("summary", ""), ""]
        for i in qa.get("issues", []):
            lines.append(f"- [{i.get('severity')}] `{i.get('file')}`: {i.get('description')}")
    if req.ledger:
        led = req.ledger
        lines += ["", f"_Run {led.get('run_id')}: {led.get('calls')} model calls, "
                      f"{led.get('tokens_in', 0) + led.get('tokens_out', 0):,} tokens, "
                      f"est. ${led.get('cost_usd', 0):.4f}_"]
    return "\n".join(lines)


@app.post("/api/zip", dependencies=[Depends(require_passcode)])
def make_zip(req: ZipRequest):
    name = validate.safe_path(req.project_name) or "council-project"
    name = name.replace("/", "-")
    files = _sanitize_files(req.files)
    if not files:
        raise HTTPException(400, "no files to package")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.writestr(f"{name}/{f['path']}", f["content"])
        z.writestr(f"{name}/COUNCIL_REPORT.md", _report_md(req))
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition": f'attachment; filename="{name}.zip"'})
