# Project Gap Analysis — Data Accuracy Audit

**Date:** 2026-07-30
**Scope:** Full review of the Gurbani Guidance RAG pipeline, with focus on data
accuracy of the corpus (`data/shabads.jsonl`) and every stage that depends on it.

**Verdict:** The PDF-derived corpus is **not accurate enough to serve as the
source of truth** for a system whose core promise is "verbatim quotes verified
against the corpus." The BaniDB v2 API (https://api.banidb.com/v2/api-docs/) —
the proofread, community-maintained database behind SikhiToTheMax — is now the
**primary** data source (`python -m src.ingest`); the PDF parser remains an
offline fallback only. A new audit tool (`python -m src.audit`) makes corpus
accuracy measurable and CI-enforceable.

---

## 1. Data-accuracy gaps in the PDF corpus (measured)

All numbers below were produced by running `python -m src.ingest_pdf` on the
checked-in PDF and auditing the output (`python -m src.audit`).

### 1.1 Corrupted Gurmukhi — unconverted legacy characters *(fixed)*

1,143 lines contained `@` and 144 contained `\` — GurbaniAkhar legacy-font
codes the converter did not map:

| Legacy char | Correct Unicode | Example (before → after) |
|---|---|---|
| `@` | ੑ udaat (U+0A51) | ਸਾਮ**@**ੈ → ਸਾਮੑੈ |
| `\` | ਞ nyanya | ਸੁੰ**\**ੀ → ਸੁੰਞੀ |

Impact: ~1,300 scripture lines were displayed to users with garbage characters
**and** poisoned the quote-verification frozenset. Both mappings are now added
to `src/ingest_pdf.py`; the audit shows **0** ASCII residue after the fix.

### 1.2 Entire sections of SGGS are missing

10 angs have no content at all: **967, 1391, 1395, 1399, 1402–1404, 1408–1409,
1430**. This is not random — it maps exactly to compositions whose authors are
absent from the parser's marker tables:

- **Angs 1389–1409 — Savaiyye of the 11 Bhatts** (Bhatt Kalsahar, Mathura,
  Gyand, et al.): the parser recognises verses only via `mÚ N` (Mahalla) or a
  hardcoded bhagat-name list; Bhatt names are in neither, so their bani is
  silently dropped or mis-attributed.
- **Ang 967 — Ramkali Ki Vaar by Satta & Balwand**: same cause.
- **Ang 1430 — Raag Mala**: dropped.

The corpus has only **12 distinct writers**; Sri Guru Granth Sahib Ji has
**35+ contributors** (6 Gurus, 15 Bhagats, 11 Bhatts, Baba Sundar Ji,
Satta & Balwand, Bhai Mardana Ji). Missing entirely: all 11 Bhatts, Baba Sundar
Ji (Sadd), Satta & Balwand, and Bhagats Trilochan, Beni, Dhanna, Jaidev, Sadhna,
Sain, Pipa, Bhikhan, Parmanand, Surdas, Ramanand — every one of them present in
SGGS but absent or unattributed (205 shabads with empty writer) in the corpus.

### 1.3 Shabad boundaries are wrong — violates non-negotiable constraint #1

The parser groups verses into shabads with a heuristic (same writer + same raag
+ non-decreasing line number). Result: **2,890 shabads** where SGGS has
**~5,900**. Consecutive shabads by the same Guru in the same raag get merged —
the corpus contains "shabads" of 71 lines. Chunking windows built on these
merged units **cross real shabad boundaries**, directly violating the project's
own constraint #1 ("chunking never crosses shabad"). BaniDB's `shabadId` gives
canonical boundaries with no heuristics.

### 1.4 Famous tuks fail verbatim verification

Reference tuks that must exist letter-for-letter are missing or corrupted, e.g.:

- ਪਵਣੁ ਗੁਰੂ ਪਾਣੀ ਪਿਤਾ ਮਾਤਾ ਧਰਤਿ ਮਹਤੁ ॥ (Slok, Ang 8) — **missing**
- ਅੰਮ੍ਰਿਤ ਵੇਲਾ ਸਚੁ ਨਾਉ ਵਡਿਆਈ ਵੀਚਾਰੁ ॥ (Japji, Ang 2) — **missing**
- ਅਨੰਦੁ ਭਇਆ ਮੇਰੀ ਮਾਏ ਸਤਿਗੁਰੂ ਮੈ ਪਾਇਆ ॥ (Anand Sahib, Ang 917) — **missing**
- ਸਭਨਾ ਜੀਆ ਕਾ ਇਕੁ ਦਾਤਾ ਸੋ ਮੈ ਵਿਸਰਿ ਨ ਜਾਈ ॥ (Japji, Ang 2) — present only
  merged into an adjacent line, so exact-match verification fails.

This is the worst failure mode for this project: when the LLM quotes these
lines **correctly**, `verify.py` strips them as "could not be verified" —
the safety layer actively deletes true Gurbani because the corpus is wrong.

### 1.5 Other measured corpus defects

- 30 lines with empty Gurmukhi; 2,818 lines (5.2%) missing English translation;
  502 missing transliteration (translit/translation interleave heuristics
  misfire on multi-line layouts).
- **78 distinct raag labels** vs the actual **31 raags** (fragmented/misparsed
  names; 177 shabads with empty raag) — raag filters and citations are wrong.
- 16 shabads contain duplicated lines.

---

## 2. Code gaps found during the review

| # | Location | Gap | Status |
|---|---|---|---|
| 1 | `src/ingest.py` | Expected verses under `verses` key, but the BaniDB v2 ang endpoint returns them under `page` — the crawler could never ingest a single ang. | **Fixed** — tolerant `extract_verses()` handles both shapes. |
| 2 | `src/ingest.py` | Verses sorted by `lineNo`, which **restarts on every ang** — any shabad spanning an ang boundary got its lines re-ordered incorrectly (scripture order corruption). | **Fixed** — canonical ordering by global `verseId` with (ang, lineNo) fallback; covered by a regression test. |
| 3 | `src/retrieve.py` | `_lookup_passages_batch` restored RRF order with key `f"{shabad_id}-{ang}"`, but passage ids are `f"{shabad_id}-{win_idx}"` — the key never matched, so ranked order silently depended on ChromaDB's return order. | **Fixed** — order keyed by the actual passage id. |
| 4 | `src/rag.py` | `_KNOWN_WRITERS` hardcoded names (e.g. "Bhagat Farid Ji", Bhatt names) that don't match corpus metadata — comparative writer filters silently returned nothing; `_validate_writers` only logged a warning. | **Fixed** — writer patterns are now built from the writers actually present in the corpus; static list is only a fallback. |
| 5 | `src/ingest_pdf.py` | Missing `@`→ੑ and `\`→ਞ legacy mappings (§1.1). | **Fixed**, with tests. |
| 6 | Project-wide | No way to measure corpus accuracy — corruption shipped silently. | **Fixed** — `python -m src.audit` (structural checks + reference tuks, optional `--online` BaniDB cross-check), non-zero exit for CI. |

---

## 3. What changed in this PR

> **Update (same PR, after review discussion):** the BaniDB v2 API is the
> authoritative data source, consumed as a **one-time sync** (`src.ingest`,
> throttled + resumable — not an ongoing crawler) that feeds the existing
> semantic pipeline: bge-m3 dense + BM25 + RRF, similarity gating, writer/raag
> filters, deep-mode facets. This is the default (`RETRIEVAL_MODE=local`)
> because BaniDB's `/search` endpoint is lexical full-word matching — using it
> as the per-question retriever would bypass semantic retrieval and break
> situational/multilingual answering. A live-search mode
> (`RETRIEVAL_MODE=banidb`, `src/retrieve_live.py`) remains available for
> hosts that can't build an index, and the BaniDB search API additionally
> serves as the online quote-verification fallback (fail-closed) in both modes.

1. **`src/banidb.py` (new)** — BaniDB v2 API client: `fetch_ang`, `fetch_shabad`,
   `search` (the endpoint from the API docs), retry/backoff, and tolerant field
   extractors shared by the crawler and the audit.
2. **`src/ingest.py` (rewritten)** — BaniDB crawler is the primary corpus
   builder: canonical `shabadId` boundaries, `verseId` ordering, verse dedupe
   across ang payloads, resumable cache, per-line `verse_id` traceability.
3. **`src/audit.py` (new)** — data-quality gate described above.
4. **Bug fixes** — retrieve.py RRF ordering, rag.py writer patterns,
   ingest_pdf.py character mappings.
5. **Docs** — README/CLAUDE.md updated: BaniDB primary, PDF fallback; error
   messages across src/ point to `python -m src.ingest`.
6. **Tests** — 10 new unit tests (BaniDB parsing shapes, ang-boundary ordering,
   dedupe, audit failure detection, legacy-font conversion); all 55 pass.

**Note:** this sandbox has no network route to api.banidb.com, so the corpus
itself could not be re-crawled here. On any machine with internet:

```bash
python -m src.ingest        # crawl BaniDB (resumable, ~15 min at 0.5s/ang)
python -m src.audit         # must pass before embedding
python -m src.embed --reset # rebuild the vector index
```

---

## 4. Remaining gaps (recommended follow-ups)

1. **Re-crawl and commit a corpus snapshot** (or a build artifact) so deploys
   don't depend on parse-time heuristics; keep `data/raw_angs/` as the archival
   source.
2. **Wire `python -m src.audit` into CI** and into the Docker/Railway deploy
   step so a failing corpus can never ship.
3. **Embed Gurmukhi text too.** Passages are currently embedded from
   transliteration + English only; Gurmukhi queries rely on bge-m3's
   cross-lingual behavior. Adding the Unicode Gurmukhi to the embedded text
   should improve ਪੰਜਾਬੀ retrieval.
4. **Surface multiple translations.** BaniDB provides Bani DB, Manmohan Singh,
   and Sant Singh Khalsa English translations; the pipeline stores the extras
   (`translation_en_ms/ssk`) but the prompt/UI only uses the primary.
5. **Sehaj/vishraam and larivaar** data from BaniDB could improve display
   fidelity (BaniDB provides `visraam` metadata per verse).
6. **Eval golden set** should add per-ang expectations (e.g. Anand Sahib →
   Ang 917) now that citations can be trusted, and a check that Bhatt/Bhagat
   comparative queries return the right voices.
7. **Rate-limiter is per-process** (in-memory dict) and reads
   `request.client.host`, which is the proxy IP behind Railway/Cloud Run —
   consider honoring `X-Forwarded-For` from a trusted proxy.
