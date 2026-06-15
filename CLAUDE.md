# CLAUDE.md — Gurbani RAG Working Memory

## Project state
- Phase 0–9 implemented; PDF-based corpus builder (`src/ingest_pdf.py`) ready — no internet required.
- Run `python -m src.ingest_pdf` to parse the SGGS PDF into `data/shabads.jsonl` (~6 seconds).

## Non-negotiable constraints
1. Chunking shabad-scoped (12-line windows, 2-line overlap, never cross shabad).
2. Quote verification mandatory: every <tuk> tag verified against corpus frozenset.
3. Model describes, never rules. Rehat questions → redirect to Sikh Rehat Maryada.
4. Gurmukhi first, translation second.
5. SGGS only in `sggs` collection.
6. PDF source: `src/SriGuruGranthSahibJiDarpanEnglish.pdf` — GurbaniAkhar legacy encoding converted to Unicode on parse.

## Quick start
1. `pip install -r requirements.txt`
2. `python -m src.ingest_pdf` — parses PDF (~6 seconds), saves to data/shabads.jsonl
3. `python -m src.embed` — builds ChromaDB index
4. `python -m src.rag "What does Gurbani say about haumai?"`
5. `uvicorn src.app:app` — start API server
6. `cd web && npm install && npm run dev` — start frontend

## Key paths
- SGGS PDF source: src/SriGuruGranthSahibJiDarpanEnglish.pdf
- Shabad corpus: data/shabads.jsonl
- Vector index: data/chroma/
- Config: src/config.py (all tunables, env-overridable)
