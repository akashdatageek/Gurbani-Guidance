"""Quote verification for Gurbani RAG.

Three layers of protection:
  1. <tuk> tags: every tagged quote verified; ang attribute validated.
  2. Untagged Gurmukhi: danda-marked runs of ≥4 words are always verified;
     danda-free runs of ≥6 words are verified when preceded by an
     attribution phrase ("Guru Ji says…", "Gurbani states…"). Short terms
     and unattributed Punjabi prose pass through (documented scope limit).
  3. Substring fallback: partial real quotes (e.g. half a line) pass; only
     fabricated text is stripped.

Verification is SHABAD-SCOPED: the corpus is keyed by
(shabad_id, line_idx, ang), and a quote spanning multiple lines only
verifies when all its segments belong to ONE shabad and are adjacent in
scripture order — two genuine lines stitched together from different
banis are rejected, closing the "stitched quote" attack.

Ang corrections are only issued for exact, unambiguous matches; substring
fragments and lines appearing at multiple locations never earn an
authoritative "[citation corrected]" note.

Verification sources, in order:
  a. The local corpus (data/shabads.jsonl), when built.
  b. `trusted_lines` — the exact lines of the passages retrieved for this
     answer (live mode), with the same location structure.
  c. The BaniDB search API (cached, fail-closed) when no local corpus exists.

Public API:
    verify_answer(answer, trusted_lines=None) -> tuple[str, list[str]]
    StreamingVerifier — incremental wrapper with identical guarantees.

Location values are (shabad_id, line_idx, ang) tuples; line_idx only needs
to be monotone within a shabad (global verseIds and window-relative indices
both qualify).
"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from functools import lru_cache

from src.config import SHABADS_FILE
from src.corpus import ensure_corpus, normalize_gurmukhi

logger = logging.getLogger(__name__)

# (shabad_id, line_idx, ang)
Loc = tuple[int, int, int]

# Matches <tuk ang="42">…</tuk>, <tuk ang='42'>…</tuk>, <tuk ang=42>…</tuk>, <tuk>…</tuk>
_TUK_RE = re.compile(
    r'<tuk(?:\s+ang=["\']?(\d+)["\']?)?\s*>(.*?)</tuk>',
    re.DOTALL,
)

# Gurmukhi Unicode block U+0A00–U+0A7F. NOTE: the dandas ।/॥ are U+0964/0965
# in the DEVANAGARI block (shared by Indic scripts) — a run regex limited to
# the Gurmukhi block silently cuts quotes off before their dandas.
_GURMUKHI_WORD_RE = re.compile(r"[਀-੿]+")
_GURMUKHI_RUN_RE = re.compile(r"[਀-੿][਀-੿\s।॥]*[਀-੿।॥]")

# Attribution phrases that mark the following Gurmukhi as a claimed quote —
# used to extend verification to danda-free runs (English + Punjabi cues).
_ATTRIBUTION_RE = re.compile(
    r"(?:\b(?:say|says|said|state|states|stated|write|writes|wrote|read|reads|"
    r"declare|declares|proclaim|proclaims|teach|teaches|taught|quote|quotes|quoted|"
    r"gurbani|bani|tuk|shabad|verse|line|scripture)\b|ਕਹਿੰਦੇ|ਆਖਦੇ|ਫੁਰਮਾ|ਫ਼ੁਰਮਾ|ਲਿਖ)"
    r"[^਀-੿]{0,20}$",
    re.IGNORECASE,
)

# Max line-index gap between consecutive segments of a multi-line quote —
# allows skipping one intervening line (e.g. a Rahao) but nothing more.
_MAX_ADJACENCY_GAP = 2

_FAILED_REPLACEMENT = "[quote removed — could not be verified against Sri Guru Granth Sahib Ji]"
_ANG_CORRECTED_TMPL = "{text} [citation corrected: Ang {correct}]"

# Track file mtime so the cache auto-invalidates if the corpus is rebuilt
_corpus_mtime: float = 0.0


def _get_corpus_lines() -> dict[str, tuple[Loc, ...]]:
    """Load corpus as {normalized_gurmukhi: (Loc, …)}. Auto-invalidates on file change."""
    global _corpus_mtime
    ensure_corpus(SHABADS_FILE)
    if not os.path.exists(SHABADS_FILE):
        logger.warning("Corpus not found — verification will reject all quotes.")
        return {}
    mtime = os.path.getmtime(SHABADS_FILE)
    if mtime != _corpus_mtime:
        _corpus_mtime = mtime
        _corpus_cache.cache_clear()
    return _build_corpus()


@lru_cache(maxsize=1)
def _build_corpus() -> dict[str, tuple[Loc, ...]]:
    corpus: dict[str, list[Loc]] = {}
    with open(SHABADS_FILE, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                shabad = json.loads(raw)
            except json.JSONDecodeError:
                continue
            sid = shabad.get("shabad_id", 0)
            shabad_ang = shabad.get("ang", 0)
            for idx, line_obj in enumerate(shabad.get("lines", [])):
                g = normalize_gurmukhi(line_obj.get("gurmukhi", ""))
                if g:
                    line_ang = line_obj.get("ang") or shabad_ang
                    corpus.setdefault(g, []).append((sid, idx, line_ang))
    logger.info("Loaded %d Gurmukhi lines into verify corpus.", len(corpus))
    return {k: tuple(v) for k, v in corpus.items()}


# Alias for test patching compatibility
_corpus_cache = _build_corpus


# ---------------------------------------------------------------------------
# Online (BaniDB) verification fallback — cached per process
# ---------------------------------------------------------------------------

# normalized text -> (exact_locs, substring_locs)
_online_cache: dict[str, tuple[tuple[Loc, ...], tuple[Loc, ...]]] = {}


def _banidb_find(norm: str) -> tuple[tuple[Loc, ...], tuple[Loc, ...]]:
    """Look a normalized Gurmukhi line up via the BaniDB search API.

    Uses (shabadId, verseId, pageNo) as the Loc — verseIds are globally
    sequential, so adjacency within a shabad still holds. Returns empty
    tuples when unreachable: fail-closed.
    """
    if norm in _online_cache:
        return _online_cache[norm]
    try:
        from src import banidb
        query = re.sub(r"[॥।]+|[੦-੯]+", " ", norm)
        query = re.sub(r"\s+", " ", query).strip()
        if not query:
            return (), ()
        resp = banidb.search(query, searchtype=banidb.SEARCH_FULL_WORD_GURMUKHI, results=20)
        exact: list[Loc] = []
        sub: list[Loc] = []
        for v in banidb.extract_verses(resp):
            vg = normalize_gurmukhi(banidb.verse_gurmukhi(v))
            loc = (banidb.verse_shabad_id(v) or 0, banidb.verse_id(v), banidb.verse_ang(v))
            if vg == norm:
                exact.append(loc)
            elif norm in vg:
                sub.append(loc)
        result = (tuple(exact), tuple(sub))
    except Exception as exc:  # noqa: BLE001 — any failure means "unverified"
        logger.warning("BaniDB verification lookup failed (%s) — quote will be stripped.", exc)
        result = ((), ())
    _online_cache[norm] = result
    return result


def _substring_locs(text: str, corpus: dict[str, tuple[Loc, ...]]) -> tuple[Loc, ...] | None:
    """Locs of corpus lines containing text as a contiguous substring (None = no match)."""
    norm = normalize_gurmukhi(text)
    found: list[Loc] = []
    matched = False
    for line, locs in corpus.items():
        if norm in line:
            matched = True
            found.extend(locs)
    return tuple(found) if matched else None


def _adjacent_chain(options: list[list[int]], max_gap: int = _MAX_ADJACENCY_GAP) -> bool:
    """True if one strictly-increasing index per segment exists with bounded gaps."""
    def dfs(seg: int, prev: int) -> bool:
        if seg == len(options):
            return True
        return any(
            dfs(seg + 1, idx)
            for idx in options[seg]
            if prev < idx <= prev + max_gap
        )
    return any(dfs(1, start) for start in options[0])


def _run_needs_verification(run: str, preceding: str) -> bool:
    """Decide whether an untagged Gurmukhi run must be verified.

    Danda-marked runs of ≥4 words are always claimed scripture. Danda-free
    runs are treated as quotes only when ≥6 words AND introduced by an
    attribution phrase — otherwise they are Punjabi prose, which this
    guarantee deliberately does not cover (see README scope note).
    """
    words = _GURMUKHI_WORD_RE.findall(run)
    if "॥" in run or "।" in run:
        return len(words) >= 4
    return len(words) >= 6 and bool(_ATTRIBUTION_RE.search(preceding[-80:]))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify_answer(
    answer: str,
    trusted_lines: dict[str, tuple[Loc, ...] | set | list] | None = None,
) -> tuple[str, list[str]]:
    """Verify all Gurmukhi citations in an answer.

    `trusted_lines` maps normalized Gurmukhi -> Loc tuples for the passages
    retrieved for this answer.

    Returns (cleaned_answer, failed_quotes).
    """
    if not answer:
        return answer, []

    local_corpus = _get_corpus_lines()
    # Online fallback only when no local corpus exists (pure live-API mode)
    use_online = len(local_corpus) == 0
    corpus: dict[str, tuple[Loc, ...]] = dict(local_corpus)
    for norm, locs in (trusted_lines or {}).items():
        locs = tuple(tuple(l) for l in locs)  # tolerate lists
        corpus[norm] = tuple(set(corpus.get(norm, ())) | set(locs))
    failed_quotes: list[str] = []

    def _lookup(text: str) -> tuple[tuple[Loc, ...], bool] | None:
        """Return (locs, exact) or None if not found anywhere."""
        norm = normalize_gurmukhi(text)
        if norm in corpus:
            return corpus[norm], True
        sub = _substring_locs(text, corpus)
        if sub is not None:
            return sub, False
        if use_online:
            exact_locs, sub_locs = _banidb_find(norm)
            if exact_locs:
                return exact_locs, True
            if sub_locs:
                return sub_locs, False
        return None

    def _verify_quote(text: str) -> tuple[tuple[Loc, ...], bool] | None:
        """Verify a quote; multi-line quotes must be one shabad, adjacent lines.

        Returns (locs, exact) or None. `exact` is False for substring/partial
        matches — those never earn ang corrections.
        """
        found = _lookup(text)
        if found is not None:
            return found

        segments = [
            s.strip() for s in re.split(r"[॥।]+", text)
            if s.strip() and not re.fullmatch(r"[੦-੯0-9\s]+|ਰਹਾਉ", s.strip())
        ]
        if len(segments) < 2:
            return None
        seg_results = []
        for seg in segments:
            seg_found = _lookup(seg)
            if seg_found is None:
                return None
            seg_results.append(seg_found)

        # All segments must share ONE shabad…
        common_sids = set.intersection(
            *({sid for sid, _, _ in locs} for locs, _ in seg_results)
        )
        matching_sids = []
        for sid in common_sids:
            # …with their lines adjacent in scripture order.
            options = [
                sorted(idx for s, idx, _ in locs if s == sid)
                for locs, _ in seg_results
            ]
            if _adjacent_chain(options):
                matching_sids.append(sid)
        if not matching_sids:
            logger.warning(
                "Rejected stitched/non-adjacent multi-line quote: %.60s…", text
            )
            return None

        # A multi-line quote that resolved to exactly ONE shabad with adjacent
        # lines is located precisely — that qualifies for ang correction even
        # though the danda-split segments matched as substrings. Ambiguous
        # (multi-shabad) resolutions never earn corrections.
        ambiguous = len(matching_sids) > 1
        sid = matching_sids[0]
        locs = tuple(
            loc for seg_locs, _ in seg_results for loc in seg_locs if loc[0] == sid
        )
        return locs, not ambiguous

    # ── Pass 1: process <tuk> tags ──────────────────────────────────────────
    def _replace_tuk(m: re.Match) -> str:
        ang_attr = m.group(1)   # may be None
        tuk_text = (m.group(2) or "").strip()

        result = _verify_quote(tuk_text)
        if result is None:
            failed_quotes.append(tuk_text)
            return _FAILED_REPLACEMENT

        locs, exact = result
        angs = {ang for _, _, ang in locs}
        if ang_attr and angs and int(ang_attr) not in angs:
            # Correct ONLY when the match is exact and the location unique —
            # substring fragments and multi-location lines must not receive
            # an authoritative correction.
            if exact and len(angs) == 1:
                correction = str(next(iter(angs)))
                logger.warning(
                    "Wrong ang in tuk: cited %s, correct %s for '%s…'",
                    ang_attr, correction, tuk_text[:30],
                )
                return _ANG_CORRECTED_TMPL.format(text=tuk_text, correct=correction)
            logger.warning(
                "Cited Ang %s not among matches %s for '%s…' — leaving uncorrected "
                "(partial or multi-location match).",
                ang_attr, sorted(angs)[:5], tuk_text[:30],
            )
        return tuk_text

    cleaned = _TUK_RE.sub(_replace_tuk, answer)

    # ── Pass 2: check untagged Gurmukhi runs ────────────────────────────────
    def _check_untagged(m: re.Match) -> str:
        run = m.group(0).strip()
        if not _run_needs_verification(run, m.string[: m.start()]):
            return run
        if _verify_quote(run) is not None:
            return run  # genuine Gurbani
        failed_quotes.append(run)
        return _FAILED_REPLACEMENT

    cleaned = _GURMUKHI_RUN_RE.sub(_check_untagged, cleaned)

    return cleaned, failed_quotes


# ---------------------------------------------------------------------------
# Streaming verification
# ---------------------------------------------------------------------------

# First character that could begin a protected region: a tag or Gurmukhi text
_HOLD_START_RE = re.compile(r"<|[਀-੿]")
# Run continuation: Gurmukhi block + whitespace + the shared-Indic dandas
_GURMUKHI_OR_SPACE_RE = re.compile(r"[਀-੿\s।॥]")
# Never buffer more than this without resolving — safety valve against a
# malformed/unclosed tag consuming the whole answer
_MAX_HOLD = 4000


class StreamingVerifier:
    """Incremental wrapper around verify_answer() for streamed answers.

    Plain prose is emitted as soon as it arrives. Anything that could be a
    scripture quote — a <tuk> element or a run of Gurmukhi text — is held
    back until it is complete, verified through exactly the same rules as
    the non-streaming path, and only then released. Unverifiable quotes
    stream out already replaced, so no unverified Gurbani is ever visible,
    not even transiently. Recently emitted text is tracked so the
    attribution-phrase rule for danda-free runs works across chunk
    boundaries.
    """

    def __init__(self, trusted_lines: dict | None = None):
        self._trusted = trusted_lines
        self._buf = ""
        self._context = ""   # tail of already-emitted text (for attribution rule)
        self.failed_quotes: list[str] = []

    def _verify_fragment(self, fragment: str) -> str:
        cleaned, failed = verify_answer(fragment, trusted_lines=self._trusted)
        self.failed_quotes.extend(failed)
        return cleaned

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        return self._drain(final=False)

    def close(self) -> str:
        return self._drain(final=True)

    def _drain(self, final: bool) -> str:
        out: list[str] = []
        while self._buf:
            m = _HOLD_START_RE.search(self._buf)
            if m is None:
                out.append(self._buf)
                self._buf = ""
                break
            if m.start() > 0:
                out.append(self._buf[: m.start()])
                self._buf = self._buf[m.start():]
                continue

            emitted, progressed = (
                self._consume_tag(final) if self._buf[0] == "<"
                else self._consume_gurmukhi(final)
            )
            out.append(emitted)
            if not progressed:
                break  # need more stream data
        text = "".join(out)
        self._context = (self._context + text)[-100:]
        return text

    def _consume_tag(self, final: bool) -> tuple[str, bool]:
        """Buffer starts with '<'. Returns (text_to_emit, made_progress)."""
        prefix = self._buf[:4]
        if prefix != "<tuk"[: len(prefix)]:
            # Not a tuk tag — release the '<' as plain text
            self._buf = self._buf[1:]
            return "<", True

        close_idx = self._buf.find("</tuk>")
        if close_idx == -1:
            if final or len(self._buf) > _MAX_HOLD:
                # Unclosed tag at end of stream: verify whatever we have
                fragment, self._buf = self._buf, ""
                return self._verify_fragment(fragment), True
            return "", False

        end = close_idx + len("</tuk>")
        element, self._buf = self._buf[:end], self._buf[end:]
        return self._verify_fragment(element), True

    def _consume_gurmukhi(self, final: bool) -> tuple[str, bool]:
        """Buffer starts with a Gurmukhi char. Returns (text_to_emit, made_progress)."""
        i = 0
        while i < len(self._buf) and _GURMUKHI_OR_SPACE_RE.match(self._buf[i]):
            i += 1
        if i == len(self._buf) and not final and len(self._buf) <= _MAX_HOLD:
            return "", False  # run may continue in the next chunk
        run, self._buf = self._buf[:i], self._buf[i:]
        core = run.strip()
        if not _run_needs_verification(core, self._context):
            return run, True
        # Force the run through pass-1 verification by wrapping it as a tuk;
        # the tags never reach the output (verified text or the replacement).
        lead = run[: len(run) - len(run.lstrip())]
        trail = run[len(run.rstrip()):]
        return lead + self._verify_fragment(f"<tuk>{core}</tuk>") + trail, True
