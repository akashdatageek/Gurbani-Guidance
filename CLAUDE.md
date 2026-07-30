# CLAUDE.md — Gurbani RAG Working Memory

## Project state
- DEFAULT: live BaniDB API retrieval (`RETRIEVAL_MODE=banidb`) — questions are
  answered by querying the BaniDB v2 search API at ask time. NO crawler, NO
  corpus build, NO embedding step required. Quote verification runs against the
  retrieved passages + the BaniDB search API (fail-closed).
- OPTIONAL local mode (`RETRIEVAL_MODE=local`): hybrid dense+BM25 over a
  locally built corpus (src.ingest → src.audit → src.embed). Only use when
  explicitly requested; be considerate of BaniDB's servers.
- `python -m src.ingest_pdf` is a last-resort offline corpus builder — its
  output fails the accuracy audit (see GAPS.md).

## Non-negotiable constraints
1. Chunking shabad-scoped (≤12-line windows, never cross shabad) — in live mode
   the window is cut from a single BaniDB shabad, centred on the matched verse.
2. Quote verification mandatory: every <tuk> tag verified (retrieved passages →
   local corpus if built → BaniDB search API). Unverifiable quotes are stripped.
3. Model describes, never rules. Rehat questions → redirect to Sikh Rehat Maryada.
4. Gurmukhi first, translation second.
5. SGGS only (BaniDB source id "G"; `sggs` collection in local mode).
6. Data source: BaniDB v2 API (https://api.banidb.com/v2) is authoritative —
   used LIVE by default, never bulk-crawled unless local mode is explicitly chosen.
7. Local-mode corpus must pass `python -m src.audit` before embedding/serving.

## Quick start (default — live BaniDB API)
1. `pip install -r requirements.txt`
2. `python -m src.rag "What does Gurbani say about haumai?"`
3. `uvicorn src.app:app` — start API server
4. `cd web && npm install && npm run dev` — start frontend

## Key paths
- BaniDB client: src/banidb.py (angs/shabads/search endpoints + verse extractors)
- Live retrieval: src/retrieve_live.py (default backend)
- Local-mode pipeline (optional): src/ingest.py → src/audit.py → src/embed.py → src/retrieve.py
- Quote verification: src/verify.py (trusted passages → local corpus → BaniDB API)
- Config: src/config.py (RETRIEVAL_MODE etc., env-overridable)
- Gap analysis: GAPS.md
