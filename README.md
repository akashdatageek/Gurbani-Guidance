# Gurbani Guidance

A retrieval-augmented generation (RAG) system over the complete Sri Guru Granth Sahib Ji (1430 angs).  
Ask questions in English, Punjabi (Gurmukhi or romanized), or Hinglish — every answer is grounded strictly in Gurbani, with verbatim quotes that are verified against the corpus before display.

---

## Architecture

```
User question
     │
     ▼
┌─────────────┐   classifies   ┌──────────────────┐
│   Router    │ ─────────────► │  Specialist agent │
│  (regex)    │                │  (system prompt)  │
└─────────────┘                └──────────┬───────┘
                                          │
               ┌──────────────────────────┘
               ▼
    ┌────────────────────┐     ┌─────────────┐
    │  Hybrid retrieval  │     │   BM25      │
    │  (bge-m3 + Chroma) │ ──► │   (RRF)     │
    └────────────────────┘     └──────┬──────┘
                                      │ top-8 passages
                                      ▼
                             ┌─────────────────┐
                             │  claude-sonnet  │
                             │  -4-6 (1500 t)  │
                             └────────┬────────┘
                                      │ raw answer
                                      ▼
                             ┌─────────────────┐
                             │  Quote verifier  │
                             │  (corpus check)  │
                             └────────┬────────┘
                                      │ verified answer
                                      ▼
                                  Response
```

### Multi-agent routing

| Agent | Triggers | Behaviour |
|---|---|---|
| **Conceptual** | Default | Theological explanation with citations |
| **Situational** | "I'm going through…", "I feel…" | Compassionate framing, passages that speak to the experience |
| **Comparative** | "Compare…", "Guru X and Bhagat Y…" | Retrieves per writer, organises answer by voice |
| **Conduct/Rehat** | "Is X allowed?", "Can Sikhs…" | Scriptural context + mandatory redirect to Sikh Rehat Maryada |
| **Adversarial** | "Write a new shabad…", historical facts | Graceful refusal, no retrieval |

---

## Quick start (local)

```bash
# 1. Clone and install
git clone https://github.com/akashdatageek/Gurbani-Guidance.git
cd Gurbani-Guidance
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY

# 3. Crawl BaniDB (~1.5 hours, resumable — safe to Ctrl-C and re-run)
python -m src.ingest

# 4. Build the vector index
python -m src.embed

# 5. Test the CLI
python -m src.rag "What does Gurbani say about haumai?"

# 6. Start the API server
uvicorn src.app:app --reload

# 7. Start the frontend (separate terminal)
cd web
npm install
npm run dev
# Open http://localhost:3000
```

---

## Deployment

### Backend (API + index)

The FastAPI server needs persistent disk for ChromaDB (`data/chroma/`) and RAM for the bge-m3 model (~1.5 GB).

**Railway (recommended free tier)**
1. Push the repo; connect to Railway.
2. Set env vars: `ANTHROPIC_API_KEY`, `CORS_ORIGINS=https://akashdatageek.github.io`.
3. Add a volume mounted at `/app/data`.
4. Build: `pip install -r requirements.txt`; start: `uvicorn src.app:app --host 0.0.0.0 --port $PORT`.
5. Run ingest + embed in a one-off process (Railway "run" command).

**GCP Cloud Run** — see `PLAN.md §13` for full notes.

### Frontend (GitHub Pages)

The Next.js chat UI is deployed automatically via GitHub Actions on every push to `main`.

1. **Enable GitHub Pages** in repo Settings → Pages → Source: "GitHub Actions".
2. **Set the `API_URL` secret** in Settings → Secrets → Actions:
   ```
   API_URL = https://your-backend-url.railway.app
   ```
3. Push to `main` — the workflow in `.github/workflows/deploy.yml` builds a static export and deploys it.
4. The site will be live at `https://akashdatageek.github.io/Gurbani-Guidance`.

> **Local dev without a deployed backend**: `npm run dev` proxies `/api/*` to `localhost:8000` automatically — no env var needed.

---

## Running tests

```bash
# Unit tests (no index required)
pytest tests/ -v --ignore=tests/test_retrieve.py --ignore=tests/test_rag.py

# Integration tests (require built index)
pytest tests/ -v -m integration
```

---

## Evaluation

```bash
python eval/run_eval.py
```

Runs 27 golden questions and prints a summary table. Exits non-zero if retrieval hit-rate < 80% or any non-adversarial answer contains a stripped quote.

---

## Non-negotiable design constraints

1. **Chunking is shabad-scoped.** Windows never cross a shabad boundary.
2. **Quote verification is mandatory.** Every `<tuk>` tag is verified against the corpus frozenset before the answer reaches the user.
3. **The model describes; it never rules.** Conduct questions always redirect to the Sikh Rehat Maryada.
4. **Gurmukhi first, translation second.**
5. **SGGS only** in the `sggs` ChromaDB collection.
6. **BaniDB rate limits respected** (≥0.5 s/request, resumable crawl).
