# CLAUDE.md — Gurbani RAG Working Memory

## Project state
- DEFAULT (`RETRIEVAL_MODE=local`): full semantic RAG — hybrid dense (bge-m3 +
  ChromaDB) + BM25 with RRF over a corpus built ONCE from the BaniDB v2 API
  (one-time sync, throttled + resumable: src.ingest → src.audit → src.embed).
  The BaniDB API is the authoritative data source; the sync is not an ongoing
  crawler.
- OPTIONAL (`RETRIEVAL_MODE=banidb`): live BaniDB search API per question —
  lexical full-word matching only, NO semantic retrieval (degrades situational/
  multilingual questions). Use only when a local index can't be built.
- Quote verification works in both modes: retrieved passages → local corpus →
  BaniDB search API fallback (fail-closed).
- The legacy PDF parser and source PDF have been REMOVED — the BaniDB-synced
  corpus snapshot (data/shabads.jsonl.gz, committed) is the only corpus source.

## Non-negotiable constraints
1. Chunking shabad-scoped (≤12-line windows, never cross shabad); shabad
   boundaries come from BaniDB's canonical shabadId.
2. Quote verification mandatory: every <tuk> tag verified (retrieved passages →
   local corpus → BaniDB search API). Unverifiable quotes are stripped.
3. Model describes, never rules. Rehat questions → redirect to Sikh Rehat Maryada.
4. Gurmukhi first, translation second.
5. SGGS only (BaniDB source id "G"; `sggs` ChromaDB collection).
6. Data source: BaniDB v2 API (https://api.banidb.com/v2) is authoritative —
   synced once into the local corpus (committed as data/shabads.jsonl.gz).
7. Corpus must pass `python -m src.audit` before embedding/serving.

## Quick start
1. `pip install -r requirements.txt`
2. `python -m src.ingest` — ONE-TIME BaniDB sync → data/shabads.jsonl (resumable)
3. `python -m src.audit` — data-quality gate (non-zero exit on failure)
4. `python -m src.embed` — build ChromaDB index
5. `python -m src.rag "What does Gurbani say about haumai?"`
6. `uvicorn src.app:app` — start API server
7. `cd web && npm install && npm run dev` — start frontend

## Key paths
- BaniDB client: src/banidb.py (angs/shabads/search endpoints + verse extractors)
- One-time corpus sync: src/ingest.py; audit: src/audit.py; index: src/embed.py
- Default retrieval: src/retrieve.py (hybrid dense+BM25+RRF)
- Optional live retrieval: src/retrieve_live.py (RETRIEVAL_MODE=banidb)
- Quote verification: src/verify.py (trusted passages → local corpus → BaniDB API)
- Config: src/config.py (RETRIEVAL_MODE etc., env-overridable)
- Gap analysis: GAPS.md
