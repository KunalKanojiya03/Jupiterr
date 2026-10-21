"""Deterministic validation layer that runs before the QA agent sees the code.

Real parsers where the standard library has one (Python, JSON, TOML, YAML), and a
string/comment-aware bracket-balance heuristic for C-family languages. Results are
"pass", "fail" (a real parser rejected it), "warn" (heuristic concern) or "skip".
"""
import ast
import json
import re
import tomllib
from html.parser import HTMLParser
from pathlib import PurePosixPath

import yaml

C_LIKE = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".css", ".scss", ".java", ".c", ".h",
          ".cpp", ".hpp", ".cc", ".cs", ".go", ".rs", ".kt", ".swift", ".php", ".dart", ".scala"}
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
             "source", "track", "wbr", "!doctype"}
STUB_PATTERNS = [
    re.compile(r"^\s*(#|//)\s*\.\.\.\s*$", re.M),
    re.compile(r"(TODO|FIXME)[:\s]+implement", re.I),
    re.compile(r"(rest of|remaining) (the )?(code|implementation)", re.I),
    re.compile(r"your (code|logic) here", re.I),
]
PAIRS = {")": "(", "]": "[", "}": "{"}


def safe_path(path):
    """Normalise a model-supplied path; return None if it escapes the project root."""
    p = PurePosixPath(str(path).replace("\\", "/").lstrip("/"))
    if not p.parts or any(part in ("..", "") for part in p.parts) or ":" in p.parts[0]:
        return None
    return str(p)


def _bracket_balance(src, hash_comments=False):
    stack, i, n, line = [], 0, len(src), 1
    while i < n:
        c = src[i]
        if c == "\n":
            line += 1
        elif src.startswith("//", i) or (hash_comments and c == "#"):
            j = src.find("\n", i)
            i = n if j == -1 else j
            continue
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            if j == -1:
                return f"unterminated block comment starting line {line}"
            line += src.count("\n", i, j)
            i = j + 2
            continue
        elif c in "\"'`":
            j = i + 1
            while j < n and src[j] != c:
                if src[j] == "\\":
                    j += 1
                elif src[j] == "\n" and c != "`":
                    break  # unterminated simple string: treat as end (could be a regex/char quirk)
                j += 1
            line += src.count("\n", i, j)
            i = j + 1
            continue
        elif c in "([{":
            stack.append((c, line))
        elif c in ")]}":
            if not stack or stack[-1][0] != PAIRS[c]:
                return f"unexpected '{c}' on line {line}"
            stack.pop()
        i += 1
    if stack:
        ch, ln = stack[-1]
        return f"unclosed '{ch}' opened on line {ln}"
    return None


class _TagChecker(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack, self.errors = [], []

    def handle_starttag(self, tag, attrs):
        if tag not in VOID_TAGS:
            self.stack.append((tag, self.getpos()[0]))

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        if tag in VOID_TAGS:
            return
        if tag in (t for t, _ in self.stack):
            while self.stack and self.stack[-1][0] != tag:
                t, ln = self.stack.pop()
                if t not in ("p", "li", "td", "tr", "th", "option"):  # optional end tags
                    self.errors.append(f"<{t}> on line {ln} not closed")
            self.stack.pop()
        else:
            self.errors.append(f"stray </{tag}> on line {self.getpos()[0]}")


def check_file(path, content):
    ext = PurePosixPath(path).suffix.lower()
    name = PurePosixPath(path).name.lower()
    if not content.strip():
        return "fail", "file is empty"
    try:
        if ext in (".py", ".pyw"):
            ast.parse(content, filename=path)
            result = ("pass", "python ast.parse ok")
        elif ext == ".json" or name in (".eslintrc", ".babelrc"):
            json.loads(content)
            result = ("pass", "valid JSON")
        elif ext == ".toml":
            tomllib.loads(content)
            result = ("pass", "valid TOML")
        elif ext in (".yaml", ".yml"):
            list(yaml.safe_load_all(content))
            result = ("pass", "valid YAML")
        elif ext in (".html", ".htm"):
            checker = _TagChecker()
            checker.feed(content)
            checker.close()
            errs = checker.errors + [f"<{t}> on line {ln} not closed" for t, ln in checker.stack
                                     if t not in ("p", "li", "html", "body", "head")]
            result = ("warn", "; ".join(errs[:3])) if errs else ("pass", "tags balanced")
        elif ext in C_LIKE:
            problem = _bracket_balance(content)
            result = ("warn", f"bracket heuristic: {problem}") if problem else ("pass", "brackets balanced (heuristic)")
        else:
            result = ("skip", "no validator for this file type")
    except SyntaxError as e:
        return "fail", f"SyntaxError line {e.lineno}: {e.msg}"
    except (json.JSONDecodeError, tomllib.TOMLDecodeError, yaml.YAMLError) as e:
        return "fail", f"parse error: {str(e)[:200]}"

    for pat in STUB_PATTERNS:
        m = pat.search(content)
        if m:
            snippet = m.group(0).strip()[:60]
            status = "warn" if result[0] in ("pass", "skip") else result[0]
            return status, f"{result[1]}; possible placeholder/stub: '{snippet}'"
    return result


def validate_files(files, file_tree=None):
    results = []
    delivered = set()
    for f in files:
        path = f.get("path", "")
        status, detail = check_file(path, f.get("content", ""))
        results.append({"path": path, "status": status, "detail": detail})
        delivered.add(path)
    missing = []
    for entry in file_tree or []:
        p = safe_path(entry.get("path", ""))
        if p and p not in delivered and not p.endswith("/"):
            missing.append(p)
    summary = {s: sum(1 for r in results if r["status"] == s) for s in ("pass", "fail", "warn", "skip")}
    summary["missing"] = len(missing)
    return {"results": results, "missing": missing, "summary": summary}
