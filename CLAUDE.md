# CLAUDE.md — Gurbani RAG Working Memory

## Project state
- Phase 0–9 implemented. PRIMARY corpus builder: `python -m src.ingest` — crawls
  the BaniDB v2 API (proofread ground truth, canonical shabad boundaries).
- `python -m src.ingest_pdf` is an OFFLINE FALLBACK only — its output fails the
  accuracy audit (see GAPS.md); never ship a corpus built from it without review.
- `python -m src.audit` — corpus data-quality gate; must pass before `src.embed`.

## Non-negotiable constraints
1. Chunking shabad-scoped (12-line windows, 2-line overlap, never cross shabad).
2. Quote verification mandatory: every <tuk> tag verified against corpus frozenset.
3. Model describes, never rules. Rehat questions → redirect to Sikh Rehat Maryada.
4. Gurmukhi first, translation second.
5. SGGS only in `sggs` collection.
6. Data source: BaniDB v2 API (https://api.banidb.com/v2) is authoritative.
   The PDF (`src/SriGuruGranthSahibJiDarpanEnglish.pdf`, GurbaniAkhar legacy
   encoding) is fallback only.
7. Corpus must pass `python -m src.audit` before embedding/serving.

## Quick start
1. `pip install -r requirements.txt`
2. `python -m src.ingest` — crawls BaniDB → data/shabads.jsonl (resumable; ~15 min)
   (offline fallback: `python -m src.ingest_pdf`, lower fidelity)
3. `python -m src.audit` — validate corpus accuracy (non-zero exit on failure)
4. `python -m src.embed` — builds ChromaDB index
5. `python -m src.rag "What does Gurbani say about haumai?"`
6. `uvicorn src.app:app` — start API server
7. `cd web && npm install && npm run dev` — start frontend

## Key paths
- BaniDB client: src/banidb.py (ang/shabad/search endpoints + verse extractors)
- Raw ang cache: data/raw_angs/ (resumable crawl cache)
- Corpus audit: src/audit.py
- SGGS PDF fallback source: src/SriGuruGranthSahibJiDarpanEnglish.pdf
- Shabad corpus: data/shabads.jsonl
- Vector index: data/chroma/
- Config: src/config.py (all tunables, env-overridable)
- Gap analysis: GAPS.md
