# ਗੁਰਬਾਣੀ ਮਾਰਗਦਰਸ਼ਨ — Gurbani Guidance

> ੴ ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ ਨਿਰਭਉ ਨਿਰਵੈਰੁ ਅਕਾਲ ਮੂਰਤਿ ਅਜੂਨੀ ਸੈਭੰ ਗੁਰ ਪ੍ਰਸਾਦਿ ॥
>
> *One Universal Creator, Truth by name, Creative Being Personified, No Fear, No Hatred, Image of the Timeless One, Beyond Birth, Self-Existent, by Guru's Grace.*
> — ਮੂਲ ਮੰਤ੍ਰ (Mool Mantar), ਸ੍ਰੀ ਗੁਰੂ ਗ੍ਰੰਥ ਸਾਹਿਬ ਜੀ, Ang 1

A retrieval-augmented generation (RAG) system over the complete **ਸ੍ਰੀ ਗੁਰੂ ਗ੍ਰੰਥ ਸਾਹਿਬ ਜੀ** (Sri Guru Granth Sahib Ji) — all 1430 ਅੰਗ (angs).

Ask in **English, ਪੰਜਾਬੀ (Punjabi / Gurmukhi), romanized Punjabi, or Hinglish** — every answer is grounded strictly in ਗੁਰਬਾਣੀ (Gurbani), with verbatim quotes verified against the corpus before display. The model describes; it never rules.

---

## ਵਾਸਤੂਕਲਾ — Architecture

```
User question  (ਸਵਾਲ)
     │
     ▼
┌──────────────────┐   classifies   ┌───────────────────────┐
│  Router          │ ─────────────► │  Specialist agent     │
│  (regex + LLM)   │                │  (6 system prompts)   │
└──────────────────┘                └──────────┬────────────┘
                                               │
              ┌────────────────────────────────┘
              ▼
   ┌───────────────────────────────────────────────┐
   │  Retrieval (RETRIEVAL_MODE)                   │
   │  local (default): dense (bge-m3 + ChromaDB)   │
   │    + BM25 sparse  ──►  RRF top-8              │
   │    corpus synced once from the BaniDB API     │
   │  banidb (optional): live /search → /shabads   │
   └──────────────────────┬────────────────────────┘
                                      │
                                      ▼
                             ┌──────────────────┐
                             │  claude/ Gemini  │
                             │  (4000 tokens)   │
                             └────────┬─────────┘
                                      │ raw answer
                                      ▼
                    ┌─────────────────────────────────┐
                    │  3-layer quote verifier          │
                    │  1. <tuk> tags + ang validation  │
                    │  2. Untagged Gurmukhi runs ≥4w   │
                    │  3. Substring fallback           │
                    └────────────────┬────────────────┘
                                     │ verified answer
                                     ▼
                                 Response
```

### ਮਲਟੀ-ਏਜੰਟ ਰੂਟਿੰਗ — Multi-agent Routing

| ਏਜੰਟ (Agent) | Triggers | Behaviour |
|---|---|---|
| **Conceptual** — ਸੰਕਲਪ | Default | Theological explanation with Gurbani citations |
| **Situational** — ਸਥਿਤੀ | "I feel…", "I'm going through…", "I am suffering" | Compassionate framing; passages that speak to the experience |
| **Comparative** — ਤੁਲਨਾ | "Compare Nanak and Kabir…", "How do different Gurus describe…" | Retrieves per writer, organises answer by voice |
| **Rehat** — ਰਹਿਤ | "Is X allowed?", "Can Sikhs drink…" | Scriptural context + mandatory redirect to [Sikh Rehat Maryada](https://www.sgpc.net/rehat_maryada/) |
| **Fabrication** — ਨਿਰਮਾਣ | "Write a shabad…", "Compose a hymn…" | Immediate, graceful refusal — no invented ਗੁਰਮੁਖੀ |
| **Out of scope** — ਬਾਹਰ | "Birth story of Guru Nanak", "History of Sikh empire" | Redirect — biographical/historical facts lie outside SGGS |

### ਡੂੰਘਾ ਅਧਿਐਨ — Deep Study Mode

An optional mode (off by default; toggle in the web UI, or `"deep": true` in the
`/ask` payload) that trades latency for depth on substantive questions:

1. **Decompose** the question into 3–5 focused facets (one cheap LLM call) —
   e.g. *haumai* → its nature, its consequence, its remedy, manmukh vs. gurmukh.
2. **Retrieve per facet** + the original query, deduplicated by shabad (up to 16
   passages vs. 8 in standard mode).
3. **Synthesise** from the wider passage set through the same verified pipeline.

Only **Conceptual**, **Situational**, and **Comparative** questions use the depth
path; Rehat, Fabrication, and Out-of-scope delegate straight to the standard flow,
so refusals stay fast.

---

## ਮੁੱਖ ਵਿਸ਼ੇਸ਼ਤਾਵਾਂ — Key Features

| | |
|---|---|
| **Data source** | [BaniDB v2 API](https://api.banidb.com/v2/api-docs/) — proofread ground truth, synced **once** into the local corpus (canonical shabad boundaries, verseId ordering); optional live-search mode (`RETRIEVAL_MODE=banidb`) |
| **Data quality** | `python -m src.audit` — structural checks + reference-tuk verification against known-good Gurbani; non-zero exit for CI |
| **Multilingual** | English · ਪੰਜਾਬੀ (Gurmukhi) · Romanized Punjabi · Hinglish |
| **Retrieval** | Hybrid dense (BAAI/bge-m3 + ChromaDB) + sparse (BM25) fused with Reciprocal Rank Fusion |
| **Quote safety** | 3-layer verification — no fabricated ਗੁਰਬਾਣੀ ever reaches the user. **Scope:** the guarantee covers Gurmukhi-script quotes (`<tuk>` tags and danda-marked runs); transliterations and English translations are the model's own rendering and are not verified |
| **Chunking** | Shabad-scoped 12-line windows, 2-line overlap — never crosses ਸ਼ਬਦ boundary |
| **Routing** | 6-agent system; LLM (haiku) fallback for non-English queries |
| **Deep Study** | Optional mode — decomposes a question into 3–5 facets, retrieves per facet (up to 16 passages) for richer, multi-angle answers |
| **Providers** | Anthropic Claude (default) or Google Gemini — switch with a single `PROVIDER` env var |
| **History** | Rolling 10-turn conversation with REHAT stickiness |
| **API** | FastAPI · rate-limited (10 req/min/IP) · CORS-configurable |
| **Frontend** | Next.js 14 · Noto Sans Gurmukhi · Markdown rendering · GitHub Pages auto-deploy |
| **Container** | Docker + docker-compose |

---

## ਤੇਜ਼ ਸ਼ੁਰੂਆਤ — Quick Start (Local)

```bash
# 1. Clone and install
git clone https://github.com/akashdatageek/Gurbani-Guidance.git
cd Gurbani-Guidance
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY

# 3. One-time corpus sync from the BaniDB API (throttled + resumable),
#    then validate and index it
python -m src.ingest
python -m src.audit
python -m src.embed

# 4. Test the CLI  (ਪੁੱਛੋ — ask)
python -m src.rag "What does Gurbani say about haumai?"
python -m src.rag "ਨਾਮ ਸਿਮਰਨ ਬਾਰੇ ਕੀ ਕਿਹਾ ਗਿਆ ਹੈ?"

# 5. Start the API server
uvicorn src.app:app --reload

# 6. Start the frontend (separate terminal)
cd web
npm install
npm run dev
# Open http://localhost:3000
```

### Optional: live BaniDB search mode

If a local index can't be built (e.g. constrained hosting), set
`RETRIEVAL_MODE=banidb` to query the BaniDB search API live per question.
**Trade-off:** BaniDB search is lexical full-word matching — semantic
retrieval (dense embeddings, similarity gating) is bypassed, which degrades
situational, romanized-Punjabi, and Hinglish questions. The default local
mode is recommended.

```bash
RETRIEVAL_MODE=banidb uvicorn src.app:app   # no corpus/index needed
```

### Docker

```bash
docker compose up          # builds image + starts API on :8000
# Set ANTHROPIC_API_KEY in environment or a .env file
```

---

## ਤੈਨਾਤੀ — Deployment

### Backend (API + index)

The FastAPI server needs persistent disk for ChromaDB (`data/chroma/`) and ~1.5 GB RAM for the bge-m3 model.

**Railway (recommended)**
1. Push the repo; connect to Railway.
2. Set env vars: `ANTHROPIC_API_KEY`, `CORS_ORIGINS=https://akashdatageek.github.io`.
3. Add a volume mounted at `/app/data`.
4. Build: `pip install -r requirements.txt`; start: `uvicorn src.app:app --host 0.0.0.0 --port $PORT`.
5. Run `python -m src.ingest && python -m src.audit && python -m src.embed` once via a Railway "run" command (one-time BaniDB sync onto the volume).

**GCP Cloud Run** — see `PLAN.md §13` for full notes.

### Frontend (GitHub Pages)

The Next.js chat UI deploys automatically via GitHub Actions on every push to `main`.

1. **Enable GitHub Pages** → Settings → Pages → Source: "GitHub Actions".
2. **Add the `NEXT_PUBLIC_API_URL` secret** → Settings → Secrets → Actions:
   ```
   NEXT_PUBLIC_API_URL = https://your-backend.railway.app
   ```
3. Push to `main` — `.github/workflows/deploy.yml` builds a static export.
4. Live at: `https://akashdatageek.github.io/Gurbani-Guidance`

> **Local dev without a deployed backend**: `npm run dev` proxies `/api/*` → `localhost:8000` automatically.

---

## ਟੈਸਟ — Tests

```bash
# Unit tests (no index required)  — 32 tests
pytest tests/ -v --ignore=tests/test_retrieve.py --ignore=tests/test_rag.py

# Integration tests (require built index + ANTHROPIC_API_KEY)
pytest tests/ -v -m integration
```

Test coverage:
- `tests/test_corpus.py` — `make_windows`, shabad chunking
- `tests/test_verify.py` — 3-layer quote verification
- `tests/test_router.py` — all 6 `QuestionType` routes + false-positive prevention + REHAT stickiness

---

## ਮੁਲਾਂਕਣ — Evaluation

```bash
python eval/run_eval.py

# Skip adversarial manual-review section
python eval/run_eval.py --skip-adversarial
```

Runs 27 golden questions and prints a summary table.  
Exits non-zero if retrieval hit-rate < 80 % or any non-adversarial answer contains a stripped quote.

---

## ਅਟੱਲ ਡਿਜ਼ਾਈਨ ਸਿਧਾਂਤ — Non-negotiable Design Constraints

| # | Constraint | Why |
|---|---|---|
| 1 | **ਸ਼ਬਦ-ਸੀਮਿਤ ਚੰਕਿੰਗ** — Chunking is shabad-scoped | Windows never cross a ਸ਼ਬਦ boundary; each window is a coherent unit of ਗੁਰਬਾਣੀ |
| 2 | **ਹਵਾਲਾ ਜਾਂਚ ਲਾਜ਼ਮੀ** — Quote verification is mandatory | Every `<tuk>` tag is checked against the corpus before the answer reaches the user; fabricated quotes are stripped |
| 3 | **ਮਾਡਲ ਦੱਸਦਾ ਹੈ, ਰਾਜ ਨਹੀਂ ਕਰਦਾ** — The model describes; it never rules | Conduct questions always redirect to the [Sikh Rehat Maryada](https://www.sgpc.net/rehat_maryada/) |
| 4 | **ਗੁਰਮੁਖੀ ਪਹਿਲਾਂ** — Gurmukhi first, translation second | Original scripture in Gurmukhi script leads every citation |
| 5 | **ਕੇਵਲ ਸ੍ਰੀ ਗੁਰੂ ਗ੍ਰੰਥ ਸਾਹਿਬ ਜੀ** — SGGS only | Only SGGS content in the `sggs` ChromaDB collection |
| 6 | **BaniDB ਸਰੋਤ** — BaniDB is authoritative | The corpus is synced once from the proofread [BaniDB v2 API](https://api.banidb.com/v2/api-docs/) (canonical shabad boundaries, verseId order) and ships in the repo as `data/shabads.jsonl.gz` |
| 7 | **ਸ਼ੁੱਧਤਾ ਜਾਂਚ** — Corpus must pass the audit | `python -m src.audit` validates ang coverage, writers, raags, and reference tuks before the corpus is embedded |

---

## ਰਿਪੋਜ਼ਟਰੀ ਬਣਤਰ — Repository Layout

```
Gurbani-Guidance/
├── src/
│   ├── config.py        — all tunables (env-overridable)
│   ├── corpus.py        — Pydantic models + make_windows()
│   ├── banidb.py        — BaniDB v2 API client (angs/shabads/search)
│   ├── ingest.py        — one-time BaniDB sync → data/shabads.jsonl
│   ├── audit.py         — corpus data-quality audit (CI gate)
│   ├── retrieve_live.py — optional live-search backend (RETRIEVAL_MODE=banidb)
│   ├── embed.py         — bge-m3 → ChromaDB
│   ├── retrieve.py      — hybrid RRF retrieval
│   ├── verify.py        — 3-layer quote verification
│   ├── rag.py           — 6-agent RAG pipeline
│   └── app.py           — FastAPI server
├── data/
│   └── shabads.jsonl.gz — BaniDB-synced corpus snapshot (auto-inflated)
├── web/                 — Next.js 14 chat UI
│   ├── app/
│   │   ├── page.tsx     — main chat interface
│   │   ├── layout.tsx
│   │   └── components/
│   └── tailwind.config.ts
├── tests/
│   ├── test_corpus.py
│   ├── test_verify.py
│   └── test_router.py
├── eval/
│   └── run_eval.py      — 27 golden questions
├── .github/
│   └── workflows/
│       └── deploy.yml   — GitHub Pages auto-deploy
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── PLAN.md              — full design spec
```

---

## ਵਾਤਾਵਰਨ ਵੇਰੀਏਬਲ — Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PROVIDER` | `anthropic` | LLM provider: `anthropic` or `gemini` |
| `ANTHROPIC_API_KEY` | *(required if anthropic)* | Claude API key |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Claude generation model |
| `CLASSIFIER_MODEL` | `claude-haiku-4-5-20251001` | Claude router/classifier model |
| `GEMINI_API_KEY` | *(required if gemini)* | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-2.5-pro` | Gemini generation model |
| `GEMINI_CLASSIFIER_MODEL` | `gemini-2.5-flash` | Gemini classifier model |
| `MAX_TOKENS` | `4000` | Max generation tokens |
| `RETRIEVAL_MODE` | `local` | `local` = hybrid dense+BM25 semantic RAG (default); `banidb` = live lexical search, no index |
| `BANIDB_SEARCH_RESULTS` | `20` | Results per BaniDB search call (live mode / verification fallback) |
| `SIMILARITY_THRESHOLD` | `0.48` | Min cosine similarity; below → out-of-scope |
| `TOP_K` | `8` | Passages returned to the LLM |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated allowed origins |
| `RATE_LIMIT_MAX` | `10` | Max requests per IP per window |
| `RATE_LIMIT_WINDOW` | `60` | Rate-limit window in seconds |

---

## ਡਾਟਾ ਸਰੋਤ ਅਤੇ ਲਾਇਸੰਸ — Data Attribution & Licensing

The Gurbani text, translations (Bani DB, Manmohan Singh, Sant Singh Khalsa),
transliterations, and vyakhya (Prof. Sahib Singh's SGGS Darpan, Faridkot Wala
Teeka) in `data/shabads.jsonl.gz` come from **[BaniDB](https://banidb.com)** —
the community-maintained, proofread database of the Khalis Foundation that
also powers SikhiToTheMax. Deep gratitude to their sevadars.

> **Permission:** the BaniDB/Khalis Foundation team has granted this project
> permission to use their database, including the corpus snapshot committed in
> this repository (confirmed by email, July 2026). This project is a
> non-commercial seva/educational effort and claims no rights over the
> scripture, translations, or teekas; all credit for the proofread data
> belongs to the BaniDB sevadars.

---

*Built with ਸਤਿਕਾਰ (respect) for Sri Guru Granth Sahib Ji. All answers are grounded solely in ਗੁਰਬਾਣੀ — no interpretation beyond what the scripture says.*
