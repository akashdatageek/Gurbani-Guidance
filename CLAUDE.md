# CLAUDE.md — Gurbani RAG Working Memory

## Project state
- Phase 0–9 implemented; BaniDB crawl needed before embedding/retrieval can run.

## Non-negotiable constraints
1. Chunking shabad-scoped (12-line windows, 2-line overlap, never cross shabad).
2. Quote verification mandatory: every <tuk> tag verified against corpus frozenset.
3. Model describes, never rules. Rehat questions → redirect to Sikh Rehat Maryada.
4. Gurmukhi first, translation second.
5. SGGS only in `sggs` collection.
6. BaniDB: ≥0.5s between requests, descriptive User-Agent, cache raw, resumable.

## Quick start
1. `pip install -r requirements.txt`
2. `python -m src.ingest` — crawls BaniDB (≈1.5 hours), saves to data/
3. `python -m src.embed` — builds ChromaDB index
4. `python -m src.rag "What does Gurbani say about haumai?"`
5. `uvicorn src.app:app` — start API server
6. `cd web && npm install && npm run dev` — start frontend

## Key paths
- BaniDB raw cache: data/raw_angs/0001.json … 1430.json
- Shabad corpus: data/shabads.jsonl
- Vector index: data/chroma/
- Config: src/config.py (all tunables, env-overridable)
