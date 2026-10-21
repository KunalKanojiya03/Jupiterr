"""Council roles: system prompts, per-role model chains and input assembly."""
import json
import os
import re

DEFAULT_CHAIN = ("gemini-3.8-flash,gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash,"
                 "gemini-3.5-flash-lite,gemini-3.1-flash-lite")

ROLES = {
    "manager":   {"title": "Project Manager",  "thinking": "high",   "max_tokens": 8192,  "attempt_timeout": 90},
    "architect": {"title": "System Architect", "thinking": "high",   "max_tokens": 16384, "attempt_timeout": 150},
    "developer": {"title": "Senior Developer", "thinking": "medium", "max_tokens": 65536, "attempt_timeout": 240},
    "qa":        {"title": "QA Reviewer",      "thinking": "high",   "max_tokens": 16384, "attempt_timeout": 150},
}


def model_chain(role):
    explicit = os.environ.get(f"{role.upper()}_MODELS")
    raw = explicit or os.environ.get("MODEL_CHAIN") or DEFAULT_CHAIN
    chain = [m.strip() for m in raw.split(",") if m.strip()]
    # A second provider, when configured, reviews first (a different model family catches
    # different mistakes) and is the last-resort failover for every other role.
    second = os.environ.get("OPENAI_COMPAT_MODEL")
    if second and not explicit and os.environ.get("OPENAI_COMPAT_API_KEY"):
        spec = f"openai_compat:{second}"
        chain = [spec] + chain if role == "qa" else chain + [spec]
    return chain


SYSTEM = {
    "manager": """You are the Project Manager of an autonomous AI engineering council (Architect, Developer, QA follow you).
Turn the user's brief into a concrete, buildable scope. Be decisive: where the brief is ambiguous, choose sensible
defaults and record them as assumptions instead of asking questions. Keep the scope deliverable in ONE code-generation
pass (typically 3-15 files). Prefer boring, well-known technology unless the brief demands otherwise.

Respond with JSON only, exactly this shape:
{
  "project_name": "kebab-case-name",
  "summary": "2-4 sentence description of what will be built",
  "assumptions": ["..."],
  "tech_stack": ["language/framework/library", "..."],
  "tasks": [{"id": "T1", "title": "...", "detail": "what must be done and how it is verified", "depends_on": []}],
  "acceptance_criteria": ["testable statement", "..."],
  "out_of_scope": ["..."]
}""",

    "architect": """You are the System Architect of an AI engineering council. Given the brief and the Project Manager's plan,
produce the technical blueprint the Developer will implement verbatim. Define every file, the contracts between modules
(function/class signatures, API routes, data shapes), data models, error handling strategy and edge cases.
The file_tree must be complete: include the dependency manifest (requirements.txt / package.json / etc.) and README.md.

Respond with JSON only, exactly this shape:
{
  "overview_md": "markdown: architecture overview, component responsibilities, data/control flow",
  "file_tree": [{"path": "relative/path.ext", "purpose": "..."}],
  "interfaces_md": "markdown: precise signatures, routes, schemas and contracts between files",
  "data_models_md": "markdown: entities, fields, types, validation rules (or 'n/a')",
  "edge_cases": ["..."],
  "dependencies": [{"name": "...", "version": "pinned or range", "why": "..."}],
  "run_instructions": "exact commands to install, run and test"
}""",

    "developer": """You are the Senior Developer of an AI engineering council. Implement the Architect's blueprint exactly.
Rules:
- Deliver EVERY file in the blueprint's file_tree, complete and runnable. No placeholders, no TODOs, no "..." elisions.
- Follow the interfaces exactly; idiomatic, modular, clean code with sensible error handling.
- Never hard-code secrets; read them from environment variables.
- If QA feedback is supplied, fix every listed issue and return the FULL updated file set (all files, not only changed ones).

Respond with JSON only, exactly this shape:
{
  "files": [{"path": "relative/path.ext", "language": "python|javascript|...", "content": "full file contents"}],
  "notes": "short notes on implementation decisions and what changed versus the previous revision"
}""",

    "qa": """You are the adversarial QA Reviewer of an AI engineering council. Assume the code is broken until you have
convinced yourself otherwise. You are the gate: nothing ships without your approval.
Check, in order:
1. Automated validation results (provided). Any "fail" or missing file is automatically a critical issue.
2. Every acceptance criterion: is it actually implemented? Trace it through the code.
3. Contract adherence to the Architect's interfaces (names, signatures, routes, data shapes).
4. Runtime bugs: bad imports, undefined names, wrong call signatures, async misuse, off-by-one, unhandled errors.
5. Security: injection, unsafe eval/exec, path traversal, hard-coded secrets, missing input validation.
Do not nitpick style. Severity: critical = will not run or is insecure; major = a requirement is broken; minor = polish.
6. If a USER-REPORTED PROBLEM is present, the user saw this fail for real: verify that exact problem is fixed.
Verdict must be "reject" if there is any critical or major issue, otherwise "approve".
Score calibration: you have only read the code, not run it, so be conservative. 90+ means you traced every
acceptance criterion and found nothing; 100 is reserved for trivially small code. Typical good code scores 75-90.
Each fix must be specific enough for the Developer to apply without guessing.

Respond with JSON only, exactly this shape:
{
  "verdict": "approve" | "reject",
  "score": 0-100,
  "summary": "2-4 sentence assessment",
  "criteria": [{"criterion": "...", "met": true, "note": "evidence"}],
  "issues": [{"severity": "critical|major|minor", "file": "path or 'general'", "description": "...", "fix": "..."}]
}""",
}


def _block(title, obj):
    body = obj if isinstance(obj, str) else json.dumps(obj, indent=2, ensure_ascii=False)
    return f"=== {title} ===\n{body}\n"


def _files_block(files):
    return "\n".join(f"--- FILE: {f.path} ---\n{f.content}\n--- END FILE ---" for f in files)


def build_user_message(stage, req, validation=None):
    parts = [_block("PROJECT BRIEF", req.brief)]
    if stage in ("architect", "developer", "qa"):
        parts.append(_block("PROJECT MANAGER PLAN", req.plan))
    if stage in ("developer", "qa"):
        parts.append(_block("ARCHITECT BLUEPRINT", req.architecture))
    rejected = bool(req.qa) and req.qa.get("verdict") == "reject"
    if stage == "developer" and req.files and (rejected or req.user_feedback):
        parts.append(_block("CURRENT CODE (to be fixed)", _files_block(req.files)))
        if rejected:
            parts.append(_block("QA FEEDBACK TO FIX", {"summary": req.qa.get("summary"),
                                                        "issues": req.qa.get("issues", [])}))
    if stage == "developer" and req.user_feedback:
        parts.append(_block("USER-REPORTED PROBLEM (highest priority)",
                            "The user ran this code and found the problem below. Find the root cause and fix it, "
                            "and check for the same mistake elsewhere in the code.\n\n" + req.user_feedback))
    if stage == "qa":
        parts.append(_block("SOURCE CODE", _files_block(req.files or [])))
        parts.append(_block("AUTOMATED VALIDATION RESULTS", validation))
        if req.user_feedback:
            parts.append(_block("USER-REPORTED PROBLEM",
                                "The user ran an earlier version of this code and hit the problem below. Confirm, "
                                "by tracing the code, that it is now fixed. If you cannot confirm the fix, reject "
                                "with a critical issue.\n\n" + req.user_feedback))
    return "\n".join(parts)


def parse_json(text):
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def _obj(props, required=None):
    return {"type": "object", "properties": props, "required": required or list(props)}


_STR, _STRS = {"type": "string"}, {"type": "array", "items": {"type": "string"}}

# Passed to Gemini as responseJsonSchema: decoding is constrained to this shape, so large
# code payloads inside JSON strings cannot come back with broken escaping.
SCHEMAS = {
    "manager": _obj({
        "project_name": _STR, "summary": _STR, "assumptions": _STRS, "tech_stack": _STRS,
        "tasks": {"type": "array", "items": _obj({"id": _STR, "title": _STR, "detail": _STR, "depends_on": _STRS})},
        "acceptance_criteria": _STRS, "out_of_scope": _STRS,
    }),
    "architect": _obj({
        "overview_md": _STR,
        "file_tree": {"type": "array", "items": _obj({"path": _STR, "purpose": _STR})},
        "interfaces_md": _STR, "data_models_md": _STR, "edge_cases": _STRS,
        "dependencies": {"type": "array", "items": _obj({"name": _STR, "version": _STR, "why": _STR})},
        "run_instructions": _STR,
    }),
    "developer": _obj({
        "files": {"type": "array", "items": _obj({"path": _STR, "language": _STR, "content": _STR})},
        "notes": _STR,
    }),
    "qa": _obj({
        "verdict": {"type": "string", "enum": ["approve", "reject"]},
        "score": {"type": "integer"},
        "summary": _STR,
        "criteria": {"type": "array", "items": _obj({"criterion": _STR, "met": {"type": "boolean"}, "note": _STR})},
        "issues": {"type": "array", "items": _obj({
            "severity": {"type": "string", "enum": ["critical", "major", "minor"]},
            "file": _STR, "description": _STR, "fix": _STR})},
    }),
}


REQUIRED = {
    "manager": ["project_name", "summary", "tasks", "acceptance_criteria"],
    "architect": ["overview_md", "file_tree"],
    "developer": ["files"],
    "qa": ["verdict", "issues"],
}


def check_shape(stage, obj):
    if not isinstance(obj, dict):
        return "response is not a JSON object"
    missing = [k for k in REQUIRED[stage] if k not in obj]
    return f"missing keys: {', '.join(missing)}" if missing else None
