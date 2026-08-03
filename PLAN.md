# Gurbani RAG — Implementation Plan

> **HISTORICAL DOCUMENT** — this is the original implementation plan, kept
> for context. The implemented system has evolved substantially beyond it:
> BaniDB-synced committed corpus (no PDF), shabad-scoped quote verification,
> SSE streaming with a streaming verifier, API spend protection, CI with a
> corpus audit and a model-output benchmark. See README.md for the current
> architecture and GAPS.md for the data-accuracy record.


## Overview
A Retrieval-Augmented Generation system for Sri Guru Granth Sahib Ji (SGGS), the eternal Guru of the Sikhs.
The system allows users to ask questions and receive answers grounded exclusively in Gurbani (scripture),
with mandatory quote verification, Gurmukhi-first output, and a respectful UI.

## Phases

### Phase 0 — Scaffolding
Create project structure: .gitignore, requirements.txt, .env.example, pyproject.toml, CLAUDE.md, PLAN.md.

### Phase 1 — Data Models & Crawling
- `src/config.py` — all tunables in one place, env-overridable
- `src/corpus.py` — Shabad/ShabadLine Pydantic models, load_shabads, make_windows, normalize_gurmukhi
- `src/ingest.py` — BaniDB crawler for all 1430 angs, resumable, rate-limited, groups into shabads.jsonl

### Phase 2 — Embedding & Indexing
- `src/embed.py` — BAAI/bge-m3 embeddings, ChromaDB upsert, passage windowing

### Phase 3 — Hybrid Retrieval
- `src/retrieve.py` — dense (bge-m3 + Chroma) + sparse (BM25) + RRF fusion

### Phase 4 — Quote Verification
- `src/verify.py` — regex-based tuk tag parser, corpus frozenset lookup, strip fabricated quotes

### Phase 5 — RAG Pipeline
- `src/rag.py` — orchestrates retrieval + Claude call + verification; CLI entry point

### Phase 6 — API Server
- `src/app.py` — FastAPI with /ask, /health, /stats; CORS; rate limiting; lifespan startup

### Phase 7 — Evaluation Harness
- `eval/golden_questions.yaml` — 25+ curated test questions
- `eval/run_eval.py` — hit-rate and verification metrics

### Phase 8 — Web Frontend
- `web/` — Next.js 14 App Router, TypeScript, Tailwind CSS
- Chat UI with Gurmukhi font rendering, citation chips, failed-quote notices

### Phase 9 — Tests
- `tests/test_corpus.py` — unit tests for data models and windowing
- `tests/test_verify.py` — unit tests for quote verification
- `tests/test_retrieve.py` — integration tests (require built index)
- `tests/test_rag.py` — integration tests (require Claude API key)

## Non-Negotiable Constraints
1. Chunking shabad-scoped: 12-line windows, 2-line overlap, never cross shabad boundary
2. Quote verification mandatory: every `<tuk>` tag verified against corpus frozenset
3. Model describes, never rules: Rehat questions redirect to Sikh Rehat Maryada
4. Gurmukhi first, translation second
5. SGGS only in `sggs` ChromaDB collection
6. BaniDB respect: ≥0.5s between requests, descriptive User-Agent, cache raw JSON, resumable

## Data Flow
```
BaniDB API → data/raw_angs/*.json
           → src/ingest.py → data/shabads.jsonl
           → src/embed.py  → data/chroma/ (ChromaDB)

User question → src/retrieve.py (dense + sparse + RRF)
             → src/rag.py (Claude API call)
             → src/verify.py (quote check)
             → FastAPI /ask → web/ Next.js
```
