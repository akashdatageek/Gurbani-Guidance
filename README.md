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
   ┌──────────────────────┐     ┌────────────┐
   │  Dense retrieval     │     │  BM25      │
   │  (bge-m3 + ChromaDB) │ ──► │  sparse    │  ──► RRF top-8
   └──────────────────────┘     └────────────┘
                                      │
                                      ▼
                             ┌──────────────────┐
                             │  claude-sonnet   │
                             │  (2500 tokens)   │
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
| **Rehat** — ਰਹਿਤ | "Is X allowed?", "Can Sikhs drink…" | Scriptural context + mandatory redirect to [Sikh Rehat Maryada](https://www.sgpc.net/sikhism/sikh-rehat-maryada-section-one.asp) |
| **Fabrication** — ਨਿਰਮਾਣ | "Write a shabad…", "Compose a hymn…" | Immediate, graceful refusal — no invented ਗੁਰਮੁਖੀ |
| **Out of scope** — ਬਾਹਰ | "Birth story of Guru Nanak", "History of Sikh empire" | Redirect — biographical/historical facts lie outside SGGS |

---

## ਮੁੱਖ ਵਿਸ਼ੇਸ਼ਤਾਵਾਂ — Key Features

| | |
|---|---|
| **Corpus** | Complete ਸ੍ਰੀ ਗੁਰੂ ਗ੍ਰੰਥ ਸਾਹਿਬ ਜੀ — 1430 angs, ~2,890 shabads parsed from `SriGuruGranthSahibJiDarpanEnglish.pdf` (no internet required) |
| **Multilingual** | English · ਪੰਜਾਬੀ (Gurmukhi) · Romanized Punjabi · Hinglish |
| **Retrieval** | Hybrid dense (BAAI/bge-m3 + ChromaDB) + sparse (BM25) fused with Reciprocal Rank Fusion |
| **Quote safety** | 3-layer verification — no fabricated ਗੁਰਬਾਣੀ ever reaches the user |
| **Chunking** | Shabad-scoped 12-line windows, 2-line overlap — never crosses ਸ਼ਬਦ boundary |
| **Routing** | 6-agent system; LLM (haiku) fallback for non-English queries |
| **History** | Rolling 10-turn conversation with REHAT stickiness |
| **API** | FastAPI · rate-limited (10 req/min/IP) · CORS-configurable |
| **Frontend** | Next.js 14 · Noto Sans Gurmukhi · GitHub Pages auto-deploy |
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

# 3. Parse the SGGS PDF  (~6 seconds, no internet needed)
python -m src.ingest_pdf

# 4. Build the vector index
python -m src.embed

# 5. Test the CLI  (ਪੁੱਛੋ — ask)
python -m src.rag "What does Gurbani say about haumai?"
python -m src.rag "ਨਾਮ ਸਿਮਰਨ ਬਾਰੇ ਕੀ ਕਿਹਾ ਗਿਆ ਹੈ?"

# 6. Start the API server
uvicorn src.app:app --reload

# 7. Start the frontend (separate terminal)
cd web
npm install
npm run dev
# Open http://localhost:3000
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
5. Run `python -m src.ingest_pdf` + `python -m src.embed` once via a Railway "run" command.

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
| 3 | **ਮਾਡਲ ਦੱਸਦਾ ਹੈ, ਰਾਜ ਨਹੀਂ ਕਰਦਾ** — The model describes; it never rules | Conduct questions always redirect to the [Sikh Rehat Maryada](https://www.sgpc.net/sikhism/sikh-rehat-maryada-section-one.asp) |
| 4 | **ਗੁਰਮੁਖੀ ਪਹਿਲਾਂ** — Gurmukhi first, translation second | Original scripture in Gurmukhi script leads every citation |
| 5 | **ਕੇਵਲ ਸ੍ਰੀ ਗੁਰੂ ਗ੍ਰੰਥ ਸਾਹਿਬ ਜੀ** — SGGS only | Only SGGS content in the `sggs` ChromaDB collection |
| 6 | **PDF ਸਰੋਤ** — Offline PDF source | Corpus parsed from `SriGuruGranthSahibJiDarpanEnglish.pdf` — no external API calls needed |

---

## ਰਿਪੋਜ਼ਟਰੀ ਬਣਤਰ — Repository Layout

```
Gurbani-Guidance/
├── src/
│   ├── config.py        — all tunables (env-overridable)
│   ├── corpus.py        — Pydantic models + make_windows()
│   ├── ingest_pdf.py    — PDF parser → data/shabads.jsonl (primary)
│   ├── ingest.py        — BaniDB crawler (legacy fallback)
│   ├── embed.py         — bge-m3 → ChromaDB
│   ├── retrieve.py      — hybrid RRF retrieval
│   ├── verify.py        — 3-layer quote verification
│   ├── rag.py           — 6-agent RAG pipeline
│   ├── app.py           — FastAPI server
│   └── SriGuruGranthSahibJiDarpanEnglish.pdf  — SGGS source PDF
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
| `ANTHROPIC_API_KEY` | *(required)* | Claude API key |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Generation model |
| `CLASSIFIER_MODEL` | `claude-haiku-4-5-20251001` | Cheap router fallback for multilingual |
| `MAX_TOKENS` | `2500` | Max generation tokens |
| `SIMILARITY_THRESHOLD` | `0.35` | Min cosine similarity; below → out-of-scope |
| `TOP_K` | `8` | Passages returned to the LLM |
| `PDF_PATH` | `src/SriGuruGranthSahibJiDarpanEnglish.pdf` | Path to SGGS source PDF |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated allowed origins |
| `RATE_LIMIT_MAX` | `10` | Max requests per IP per window |
| `RATE_LIMIT_WINDOW` | `60` | Rate-limit window in seconds |

---

*Built with ਸਤਿਕਾਰ (respect) for Sri Guru Granth Sahib Ji. All answers are grounded solely in ਗੁਰਬਾਣੀ — no interpretation beyond what the scripture says.*
