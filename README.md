# Jupiterr — Autonomous Multi-Agent AI Council (POC)

One brief in, a reviewed multi-file project out. A council of role-based agents
(Project Manager → System Architect → Senior Developer → adversarial QA) plans,
builds and reviews code, with a deterministic validation gate and an automatic
QA-reject → refactor loop.

## How it works

```
browser (any device, HTTPS only)
  │  one short request per stage; the browser drives the pipeline
  ▼
FastAPI  /api/step ── manager ─► architect ─► developer ─► [validate] ─► qa
                                                ▲                         │
                                                └──── reject (≤ N rounds) ┘
         /api/zip  ── packages files + COUNCIL_REPORT.md
```

- **Serverless-safe:** each stage is its own request (≤ 300 s), so long runs never hit a function timeout.
- **Failover:** each role has a model chain (default `gemini-3.8-flash → 3.6-flash → 3.5-flash → 3.5-flash-lite`).
  429/5xx/timeouts back off exponentially (honouring Gemini's `retryDelay`), then fail over to the next model.
  Any OpenAI-compatible provider (Groq, OpenRouter) can be added to a chain as `openai_compat:<model>`.
- **Validation gate:** generated files are parsed server-side (Python `ast`, JSON, TOML, YAML; HTML tag and
  C-family bracket heuristics; stub/TODO detection; files missing from the blueprint). QA cannot approve code
  that fails parsing or carries critical/major issues; the server overrides the verdict.
- **Budget cap without a database:** the run's spend is carried in an HMAC-signed ledger that the browser
  returns on each request. The server refuses stages once `RUN_BUDGET_USD` is spent, and refuses more
  developer rounds past `MAX_QA_ROUNDS`.
- **Secrets stay server-side:** the Gemini key lives only in environment variables. The dashboard is locked by `APP_PASSCODE`.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env        # then fill in GEMINI_API_KEY
.venv\Scripts\python -m uvicorn app:app --port 8000
```

Open http://localhost:8000. Locally, a passcode is optional.

## Deploy to Vercel (reachable from any device or network)

```bash
vercel link
vercel env add GEMINI_API_KEY production
vercel env add APP_PASSCODE production
vercel --prod
```

A deployment refuses to run without `APP_PASSCODE`. Also set a budget alert or cap in
Google AI Studio or Cloud Billing; the in-app cap is per run.

## Configuration

See `.env.example`. Price-per-token values are **estimates** used for the meter and the cap.
Set them from the current Gemini pricing page. On the free tier, actual spend is $0.

`GET /api/models` (with passcode) lists the Gemini models your key can call. Use it to check the chain.

## Phase 2 ideas

Persistent run history and a global daily cap (Vercel KV / Postgres), a second provider in the QA
chain for real model diversity, sandboxed test execution, a vector store for project memory, and
streaming token output.
