# CLAUDE.md — Gurbani RAG Working Memory

## Project state
- DEFAULT (`RETRIEVAL_MODE=local`): full semantic RAG — hybrid dense (bge-m3 +
  ChromaDB) + BM25 with RRF over a corpus built ONCE from the BaniDB v2 API.
  The corpus snapshot is COMMITTED (data/shabads.jsonl.gz, auto-inflated on
  first use) — fresh clones need no sync. Rebuild path: src.ingest →
  src.audit → src.embed.
- OPTIONAL (`RETRIEVAL_MODE=banidb`): live BaniDB search API per question —
  lexical full-word matching only (degraded); use only when no index exists.
- Streaming: POST /ask/stream (SSE). StreamingVerifier holds <tuk> elements
  and Gurmukhi runs until verified — unverified Gurbani is never on the wire.
- BaniDB/Khalis Foundation permission for the data is on record (NOTICE).
- The legacy PDF parser and source PDF were REMOVED long ago; never reintroduce.

## Non-negotiable constraints
1. Chunking shabad-scoped (≤12-line windows, never cross shabad); shabad
   boundaries come from BaniDB's canonical shabadId.
2. Quote verification mandatory and SHABAD-SCOPED: corpus keyed by
   (shabad_id, line_idx, ang); multi-line quotes must be one shabad with
   adjacent lines (stitched quotes are rejected); ang corrections only for
   exact unambiguous matches; danda-free runs ≥6 words verified when
   attribution-preceded. Unverifiable quotes are stripped (fail-closed).
3. Model describes, never rules. Rehat questions → redirect to
   https://www.sgpc.net/rehat_maryada/
4. Gurmukhi first, translation second.
5. SGGS only (BaniDB source id "G"; `sggs` ChromaDB collection).
6. Data source: BaniDB v2 API is authoritative — synced once, committed
   as data/shabads.jsonl.gz. Attribution in NOTICE; code license MIT.
7. Corpus must pass `python -m src.audit`; retrieve.init() fails loudly on
   an empty Chroma collection — never serve silently degraded.

## Quick start
1. `pip install -r requirements.txt`
2. `python -m src.embed` — build the dense index from the committed corpus
   (one-time; the corpus itself ships in the repo)
3. `python -m src.rag "What does Gurbani say about haumai?"`
4. `uvicorn src.app:app` — API server (/ask, /ask/stream, /health, /stats)
5. `cd web && npm install && npm run dev` — frontend (streams, JSON fallback)

## Testing & quality gates (run before shipping)
- `pytest tests/ --ignore=tests/test_retrieve.py --ignore=tests/test_rag.py`
  — unit suite (what CI runs; no index/key needed)
- `python -m src.audit` — corpus accuracy gate (CI)
- `python eval/benchmark.py --mock --force-scan-retrieval` — end-to-end
  model-output benchmark, safety checks required at 100% (CI)
- `python eval/benchmark.py --runs 2` — real-model benchmark (needs API key)
- `python eval/run_eval.py --retrieval-only` — retrieval golden questions
  (needs built index; threshold 80%)

## Key paths
- BaniDB client: src/banidb.py (angs/shabads/search + tolerant extractors)
- One-time corpus sync: src/ingest.py; audit: src/audit.py; index: src/embed.py
- Default retrieval: src/retrieve.py (dense+BM25+RRF, pickled BM25 cache,
  window-level ang_start/ang_end filters, sentence-scoped similarity gate)
- Optional live retrieval: src/retrieve_live.py (RETRIEVAL_MODE=banidb)
- Verification: src/verify.py (shabad-scoped; StreamingVerifier for SSE)
- Pipeline: src/rag.py (ask / deep_ask / ask_stream; canned refusals)
- Server: src/app.py (auth token, daily cap, TRUST_PROXY, threadpool endpoints)
- Benchmark: eval/benchmark.py + eval/benchmark_questions.yaml
- Config: src/config.py (env-overridable); .env.example documents everything
- History: GAPS.md (data-accuracy record); PLAN.md (original spec, historical)
