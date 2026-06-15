"""Gurbani RAG pipeline with multi-agent routing.

Router (classify_question) classifies each question into a type using
fast regex (language-agnostic pre-checks first) with a cheap LLM fallback
for non-English/Hinglish input.  A specialist agent (system prompt + retrieval
strategy) then handles the question.

Question types
--------------
CONCEPTUAL  — theology, definitions, concepts
SITUATIONAL — user in a personal situation seeking comfort/guidance
COMPARATIVE — compare teachings across Gurus/Bhagats/raags
REHAT       — conduct/practice/"is X allowed" questions; always redirects
FABRICATION — requests to compose/fabricate Gurmukhi (immediate refusal)
OUT_OF_SCOPE — out-of-corpus history/biography questions (immediate refusal)

Public API
----------
ask(question, history=None, **filters) -> dict
    {answer, failed_quotes, sources, question_type}
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from enum import Enum
from typing import Any

import anthropic

from src.config import (
    ANTHROPIC_API_KEY,
    CLASSIFIER_MODEL,
    CLAUDE_MODEL,
    GEMINI_API_KEY,
    GEMINI_CLASSIFIER_MODEL,
    GEMINI_MODEL,
    HISTORY_MAX_CHARS,
    HISTORY_MAX_TURNS,
    MAX_TOKENS,
    PROVIDER,
    SHABADS_FILE,
    TOP_K,
)
from src.retrieve import Passage, retrieve
from src.verify import verify_answer

logger = logging.getLogger(__name__)

# Module-level clients — created once, reused across requests
_anthropic_client: anthropic.Anthropic | None = None
_gemini_client: Any = None


def _get_anthropic_client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to .env or the environment.")
        _anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _anthropic_client


def _get_gemini_client() -> Any:
    global _gemini_client
    if _gemini_client is None:
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError("Run: pip install google-generativeai") from exc
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set. Add it to .env or the environment.")
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_client = genai
    return _gemini_client


def _llm_call(system: str, messages: list[dict], max_tokens: int = MAX_TOKENS) -> str:
    """Unified LLM call — routes to Anthropic or Gemini based on PROVIDER."""
    if PROVIDER == "gemini":
        genai = _get_gemini_client()
        import google.generativeai as _genai
        history_for_gemini = []
        for m in messages[:-1]:
            role = "user" if m["role"] == "user" else "model"
            history_for_gemini.append({"role": role, "parts": [m["content"]]})
        model = _genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=system,
        )
        chat = model.start_chat(history=history_for_gemini)
        resp = chat.send_message(messages[-1]["content"])
        return resp.text
    else:
        client = _get_anthropic_client()
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        return resp.content[0].text


def _llm_classify_call(prompt: str) -> str:
    """Cheap single-turn LLM call for classification."""
    if PROVIDER == "gemini":
        import google.generativeai as genai
        _get_gemini_client()
        model = genai.GenerativeModel(model_name=GEMINI_CLASSIFIER_MODEL)
        resp = model.generate_content(prompt)
        return resp.text.strip().upper()
    else:
        client = _get_anthropic_client()
        resp = client.messages.create(
            model=CLASSIFIER_MODEL,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip().upper()


# ---------------------------------------------------------------------------
# Question type enum
# ---------------------------------------------------------------------------

class QuestionType(str, Enum):
    CONCEPTUAL = "conceptual"
    SITUATIONAL = "situational"
    COMPARATIVE = "comparative"
    REHAT = "rehat"
    FABRICATION = "fabrication"    # compose / invent Gurmukhi
    OUT_OF_SCOPE = "out_of_scope"  # historical/biographical, outside SGGS


# ---------------------------------------------------------------------------
# Regex classifier (fast path — English)
# ---------------------------------------------------------------------------

_FABRICATION_RE = re.compile(
    r"\b(?:write\s+(?:a\s+)?new|compose\s+(?:a\s+)?|make\s+up\s+(?:a\s+)?|"
    r"fabricate\s+(?:a\s+)?|invent\s+(?:a\s+)?|create\s+(?:a\s+)?new|"
    r"generate\s+(?:a\s+)?)\s*(?:shabad|verse|tuk|bani|gurbani|scripture|hymn)\b",
    re.I,
)

_HISTORICAL_RE = re.compile(
    r"\b(?:janamsakh[i]?|birth\s+stor(?:y|ies)?|biography\s+of|historical\s+account\s+of|"
    r"when\s+was\s+guru\s+\w+\s+born|where\s+was\s+guru\s+\w+\s+born|"
    r"who\s+killed\s+guru|martyrdom\s+of\s+guru\s+(?!granth)|"
    r"life\s+of\s+guru\s+(?!granth)|political\s+history)\b",
    re.I,
)

# REHAT: must pair "can/is/are" with a conduct verb to avoid "Can I understand…"
_REHAT_RE = re.compile(
    r"\b(?:is\s+it\s+(?:allowed|permitted|ok|okay|right|wrong|permissible|haram|forbidden)|"
    r"(?:can|may)\s+(?:i|we|sikhs?|a\s+sikh)\s+\w{0,20}\s*(?:eat|drink|wear|cut|trim|marry|consume|practice|perform|do|use|have)|"
    r"are\s+sikhs?\s+(?:allowed|permitted|supposed|expected)\s+to|"
    r"rehat\s+maryada|sikh\s+code\s+of\s+conduct|panj\s+kakkars?|"
    r"5\s+k['s]?|cutting\s+hair|eating\s+meat|consuming\s+alcohol|"
    r"interfaith\s+marriage)\b",
    re.I,
)

_COMPARATIVE_RE = re.compile(
    r"\b(?:compare|comparison|contrast|differ(?:ence)?\s+between|"
    r"similarities?\s+between|how\s+do\s+different|what\s+did\s+each|"
    r"across\s+(?:the\s+)?gurus?|multiple\s+gurus?|"
    r"different\s+(?:writers?|gurus?|bhagats?)|various\s+(?:gurus?|bhagats?))\b|"
    r"\b(?:nanak|arjan|kabir|ravidas|namdev|farid|tegh)\b.{1,60}\b(?:and|vs\.?|versus|compared\s+to)\b.{1,60}\b(?:nanak|arjan|kabir|ravidas|namdev|farid|tegh|guru|bhagat)\b",
    re.I | re.S,
)

_SITUATIONAL_RE = re.compile(
    r"\bi(?:'m| am)\s+(?:feeling|going\s+through|struggling|suffering|grieving|"
    r"dealing\s+with|facing|lost|broken|alone|scared|helpless|hopeless|overwhelmed)\b|"
    r"\bi\s+feel\s+(?:so\s+)?(?:lost|alone|scared|helpless|overwhelmed|hopeless|broken|sad|empty)\b|"
    r"\bi(?:'ve| have)\s+(?:been\s+(?:feeling|struggling)|lost\s+(?:my|a\s+\w+)|"
    r"failed|made\s+a\s+mistake)\b|"
    r"\bhelp\s+me\s+(?:with|through|deal\s+with|cope\s+with|get\s+through)\b|"
    r"\bmy\s+(?:grief|loss|pain|suffering|depression|anxiety|fear|anger|"
    r"guilt|loneliness|sorrow|death\s+of|divorce)\b",
    re.I,
)

# Non-ASCII ratio threshold for non-English detection
_HINGLISH_RE = re.compile(
    r"\b(kya|hai|ka|ki|ko|mujhe|mera|tera|yeh|woh|aur|nahi|bhi|karo|bolo|"
    r"batao|kaise|kyun|kab|naam|simran|waheguru|gurbani|shabad|paath)\b",
    re.I,
)


def _looks_non_english(text: str) -> bool:
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if non_ascii / max(len(text), 1) > 0.12:
        return True
    return bool(_HINGLISH_RE.search(text))


def _llm_classify(question: str) -> QuestionType:
    """Cheap LLM classifier for non-English/ambiguous questions."""
    prompt = (
        "Classify this question about Gurbani into exactly one category.\n\n"
        "FABRICATION — asking to compose, write, or invent Gurmukhi verses\n"
        "OUT_OF_SCOPE — asking about historical facts, biographies, political events not in SGGS\n"
        "REHAT — asking about Sikh code of conduct, rules, 'is X allowed'\n"
        "COMPARATIVE — asking to compare teachings of different Gurus or Bhagats\n"
        "SITUATIONAL — user describes personal struggle and seeks spiritual comfort\n"
        "CONCEPTUAL — everything else: concepts, meanings, explanations from Gurbani\n\n"
        f"Question: {question}\n\n"
        "Reply with exactly one word from the list above."
    )
    try:
        label = _llm_classify_call(prompt)
        mapping = {
            "FABRICATION": QuestionType.FABRICATION,
            "OUT_OF_SCOPE": QuestionType.OUT_OF_SCOPE,
            "REHAT": QuestionType.REHAT,
            "COMPARATIVE": QuestionType.COMPARATIVE,
            "SITUATIONAL": QuestionType.SITUATIONAL,
            "CONCEPTUAL": QuestionType.CONCEPTUAL,
        }
        return mapping.get(label, QuestionType.CONCEPTUAL)
    except Exception as exc:
        logger.warning("LLM classifier failed (%s); defaulting to CONCEPTUAL", exc)
        return QuestionType.CONCEPTUAL


def classify_question(
    question: str,
    history: list[dict] | None = None,
) -> QuestionType:
    """Classify a question; uses regex fast-path, then LLM for non-English."""
    # Fast regex (English-optimised)
    if _FABRICATION_RE.search(question):
        return QuestionType.FABRICATION
    if _HISTORICAL_RE.search(question):
        return QuestionType.OUT_OF_SCOPE
    if _REHAT_RE.search(question):
        return QuestionType.REHAT
    if _COMPARATIVE_RE.search(question):
        return QuestionType.COMPARATIVE
    if _SITUATIONAL_RE.search(question):
        return QuestionType.SITUATIONAL

    # History-aware REHAT stickiness: if last assistant turn included Rehat redirect
    # and current question is a short follow-up, keep routing to REHAT
    if history and len(history) >= 2:
        last_assistant = next(
            (m["content"] for m in reversed(history) if m.get("role") == "assistant"),
            "",
        )
        if "sgpc.net/rehat" in last_assistant:
            short_followup = re.compile(
                r"^\s*(yes|no|okay|ok|but|just|please|so|then|tell me|why not|"
                r"what about|still|go ahead|anyway|give me)\b",
                re.I,
            )
            if short_followup.match(question) or len(question.split()) < 7:
                return QuestionType.REHAT

    # LLM fallback for non-English
    if _looks_non_english(question):
        return _llm_classify(question)

    return QuestionType.CONCEPTUAL


# ---------------------------------------------------------------------------
# Query condensation for follow-up questions
# ---------------------------------------------------------------------------

_ANAPHORA_RE = re.compile(
    r"^\s*(?:tell me more|what else|how about|and|also|explain|elaborate|"
    r"give me more|continue|what about|why|how so|can you|could you)\b",
    re.I,
)


def _condense_query(question: str, history: list[dict]) -> str:
    """Rewrite follow-up question into a standalone retrieval query."""
    if not history or len(history) < 2:
        return question
    # Only condense short or anaphoric questions
    if not _ANAPHORA_RE.match(question) and len(question.split()) > 8:
        return question
    last_user = next(
        (m["content"] for m in reversed(history) if m.get("role") == "user"),
        "",
    )
    if not last_user:
        return question
    # Merge: "previous question + follow-up"
    condensed = f"{last_user} {question}"
    logger.debug("Condensed query: '%s' → '%s'", question, condensed)
    return condensed


# ---------------------------------------------------------------------------
# Known writers list (used for comparative retrieval)
# ---------------------------------------------------------------------------

_KNOWN_WRITERS = [
    "Guru Nanak Dev Ji", "Guru Angad Dev Ji", "Guru Amar Das Ji",
    "Guru Ram Das Ji", "Guru Arjan Dev Ji", "Guru Tegh Bahadur Ji",
    "Bhagat Kabir Ji", "Bhagat Ravidas Ji", "Bhagat Namdev Ji",
    "Bhagat Farid Ji", "Bhagat Trilochan Ji", "Bhagat Dhanna Ji",
    "Bhagat Beni Ji", "Bhagat Parmanand Ji", "Bhagat Sain Ji",
    "Bhagat Pipa Ji", "Bhagat Sadhna Ji", "Bhagat Bhikhan Ji",
    "Bhagat Surdas Ji", "Bhagat Ramanand Ji",
    # Bhatt composers
    "Bhat Kalsahar", "Bhat Bal", "Bhat Bhalh", "Bhat Kirat",
    "Bhat Sal", "Bhat Bhal", "Bhat Nal", "Bhat Gyand",
    "Bhat Mathura", "Bhat Harbans", "Bhat Tal",
]

# Build patterns using the *unique* name token (Nanak, Arjan, Kabir…) not
# the category word (Guru, Bhagat) which would match almost every question
_WRITER_PATTERNS: dict[str, re.Pattern] = {}
for _w in _KNOWN_WRITERS:
    _parts = _w.replace(" Ji", "").replace("Bhat ", "").split()
    # Use the last distinctive word; skip common prefixes
    _skip = {"Guru", "Bhagat", "Bhat", "Dev"}
    _unique_parts = [p for p in _parts if p not in _skip]
    if _unique_parts:
        # Match any of the distinctive tokens
        _pattern = "|".join(r"\b" + re.escape(p) + r"\b" for p in _unique_parts)
        _WRITER_PATTERNS[_w] = re.compile(_pattern, re.I)

# Validate writer names against corpus at startup (lazy, logged once)
_writers_validated = False


def _validate_writers() -> None:
    global _writers_validated
    if _writers_validated or not os.path.exists(SHABADS_FILE):
        return
    corpus_writers: set[str] = set()
    with open(SHABADS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    corpus_writers.add(json.loads(line).get("writer", ""))
                except Exception:
                    pass
    unmatched = [w for w in _KNOWN_WRITERS if w not in corpus_writers]
    if unmatched:
        logger.warning(
            "Writer name mismatch — these names don't appear in corpus: %s. "
            "Check src/ingest_pdf.py writer/bhagat name maps.",
            unmatched,
        )
    _writers_validated = True


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_BASE_RULES = """\
## Core rules (non-negotiable)

1. **Answer ONLY from the passages provided.** Do not use outside knowledge \
   for scriptural claims. If no relevant passage covers the question, say so \
   plainly.

2. **Quote exactly.** Wrap every Gurmukhi citation in:
       <tuk ang="ANG_NUMBER">ਗੁਰਮੁਖੀ ਤੁਕ</tuk>
   Use the ang number shown in the passage header for that specific line. \
   Never paraphrase or reconstruct Gurmukhi.

3. **Gurmukhi first, translation second.** Show the original Gurmukhi before \
   its English translation, always.

4. **Describe, never rule.** Explain what the scripture says; do not issue \
   edicts on personal conduct or judge the user.

5. **Rehat questions → redirect.** For questions about Sikh code of conduct, \
   dress, ceremonies, or diet, always include: \
   "Questions about Sikh code of conduct are answered by the Sikh Rehat \
   Maryada: https://www.sgpc.net/rehat_maryada/"

6. **Tone.** Be humble, reverent, and warm.\
"""

_SYSTEM_PROMPTS: dict[QuestionType, str] = {
    QuestionType.CONCEPTUAL: f"""\
You are Gurbani Guidance — a respectful, scholarly assistant dedicated to \
Sri Guru Granth Sahib Ji (SGGS).

{_BASE_RULES}

## Format
- Brief framing of the concept.
- Gurmukhi passages with <tuk ang="N"> tags, translations, and citations \
  (Ang N, Raag X, written by Y).
- Closing synthesis in natural prose.
""",

    QuestionType.SITUATIONAL: f"""\
You are Gurbani Guidance — a compassionate, respectful presence drawing on \
Sri Guru Granth Sahib Ji (SGGS).

{_BASE_RULES}

## Guidance for personal/situational questions
- Begin by acknowledging the user's situation warmly and without judgment.
- Offer relevant passages with <tuk ang="N"> tags, translations, and gentle \
  framing: "Guru Ji reminds us…", "In this bani, we are told…"
- Close with an encouraging word rooted in the scripture — not generic advice.
""",

    QuestionType.COMPARATIVE: f"""\
You are Gurbani Guidance — a scholarly assistant skilled in comparative analysis \
of Sri Guru Granth Sahib Ji (SGGS).

{_BASE_RULES}

## Format for comparative questions
- One section per distinct author/voice present in the passages.
- In each section: quoted tuk with <tuk ang="N"> tag, translation, brief \
  interpretation of that author's emphasis.
- Closing synthesis paragraph noting the unity and any distinctive emphases.
- Only compare voices present in the provided passages.
""",

    QuestionType.REHAT: f"""\
You are Gurbani Guidance — a respectful assistant that grounds answers in \
Sri Guru Granth Sahib Ji (SGGS) while always directing conduct questions to \
the proper authority.

{_BASE_RULES}

## Handling conduct / Rehat questions
1. Provide passages that speak to the underlying values or principles at issue \
   (with <tuk ang="N"> tags, translations, citations).
2. ALWAYS close with exactly this notice:
   "Questions about specific rules of Sikh conduct — including diet, dress, \
    the Panj Kakkars, ceremonies, and daily discipline — are answered by the \
    Sikh Rehat Maryada, the official code of conduct approved by the Akal Takht. \
    Read it in full at: https://www.sgpc.net/rehat_maryada/ \
    For personal guidance, please consult a qualified Granthi or Giani."
3. Never say whether a specific practice is "allowed" or "forbidden".
""",

    QuestionType.FABRICATION: """\
You are Gurbani Guidance. A user has asked you to compose or invent Gurmukhi \
scripture. You must decline.

Respond with:
"I can only share what is already written in Sri Guru Granth Sahib Ji — \
I cannot compose, invent, or attribute new verses to the Gurus or Bhagats. \
Every Gurmukhi line I show you is verified against the corpus before being \
displayed to you.

If you have a question about what Gurbani teaches on a topic, I would be \
glad to help."
""",

    QuestionType.OUT_OF_SCOPE: """\
You are Gurbani Guidance. A user has asked about historical or biographical \
information that is outside Sri Guru Granth Sahib Ji.

Respond with:
"Sri Guru Granth Sahib Ji is a scripture of divine wisdom — it does not \
contain historical chronicles, biographical accounts of the Gurus' lives, or \
political history. My answers are grounded exclusively in the Gurbani of SGGS.

For questions about Sikh history, I'd suggest resources like the Sikh \
Encyclopedia (www.sikhiwiki.org) or consulting a qualified historian. \
If you have a question about what Gurbani teaches spiritually, I'm here to help."
""",
}


# ---------------------------------------------------------------------------
# Comparative multi-retrieval
# ---------------------------------------------------------------------------

def _comparative_retrieve(question: str, k: int) -> list[Passage]:
    """Retrieve per mentioned writer then merge, for comparative questions."""
    _validate_writers()
    mentioned = [w for w, pat in _WRITER_PATTERNS.items() if pat.search(question)]

    if mentioned:
        per_writer_k = max(3, k // max(len(mentioned), 1))
        passages_by_writer: dict[str, list[Passage]] = {}
        for writer in mentioned:
            results = retrieve(question, k=per_writer_k, writer=writer)
            if results:
                passages_by_writer[writer] = results

        if passages_by_writer:
            writers = list(passages_by_writer.keys())
            per_slot = max(1, k // len(writers))
            merged: list[Passage] = []
            for w in writers:
                merged.extend(passages_by_writer[w][:per_slot])
            # Pad with unfiltered results if needed
            if len(merged) < k:
                seen = {p.shabad_id for p in merged}
                extra = retrieve(question, k=k - len(merged))
                merged.extend(p for p in extra if p.shabad_id not in seen)
            return merged[:k]

    # No specific writers named — broad retrieve
    return retrieve(question, k=k + 4)[:k]


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def _format_passages(passages: list[Passage]) -> str:
    """Format passages for the LLM prompt. Uses per-line ang when available."""
    blocks: list[str] = []
    for i, p in enumerate(passages, 1):
        header = f"[Passage {i}] Ang {p.ang} · {p.raag} · {p.writer}"
        lines_block_parts = []
        for j, (g, e) in enumerate(zip(p.gurmukhi, p.translation_en)):
            # Use the exact ang for this line if available
            line_ang = p.line_angs[j] if j < len(p.line_angs) else p.ang
            lines_block_parts.append(
                f"  [Ang {line_ang}] Gurmukhi: {g}\n  Translation: {e}"
            )
        blocks.append(f"{header}\n" + "\n".join(lines_block_parts))
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# History sanitisation
# ---------------------------------------------------------------------------

def _sanitise_history(history: list[dict]) -> list[dict]:
    """Cap history length and per-message size; filter error messages."""
    if not history:
        return []
    # Filter out error messages that shouldn't be in LLM context
    filtered = [
        m for m in history
        if not (m.get("role") == "assistant" and
                m.get("content", "").startswith("Sorry, I encountered an error"))
    ]
    # Cap per-message length
    capped = [
        {**m, "content": m["content"][:HISTORY_MAX_CHARS]} for m in filtered
    ]
    # Keep last N turns (2 messages per turn: user + assistant)
    return capped[-(HISTORY_MAX_TURNS * 2):]


# ---------------------------------------------------------------------------
# Main RAG pipeline
# ---------------------------------------------------------------------------

def ask(
    question: str,
    history: list[dict] | None = None,
    writer: str | None = None,
    raag: str | None = None,
    ang_range: tuple[int, int] | None = None,
    k: int = TOP_K,
) -> dict:
    """Full RAG pipeline. Returns {answer, failed_quotes, sources, question_type}."""
    clean_history = _sanitise_history(history or [])
    qtype = classify_question(question, history=clean_history)
    logger.info("Question type: %s | question: %.80s", qtype.value, question)

    # Immediate refusals — no retrieval
    if qtype in (QuestionType.FABRICATION, QuestionType.OUT_OF_SCOPE):
        answer = _llm_call(
            system=_SYSTEM_PROMPTS[qtype],
            messages=[{"role": "user", "content": question}],
            max_tokens=300,
        )
        return {
            "answer": answer,
            "failed_quotes": [],
            "sources": [],
            "question_type": qtype.value,
        }

    # Condense follow-up questions for better retrieval
    retrieval_query = _condense_query(question, clean_history)

    # Retrieve passages
    extra_filters: dict[str, Any] = {}
    if writer:
        extra_filters["writer"] = writer
    if raag:
        extra_filters["raag"] = raag
    if ang_range:
        extra_filters["ang_range"] = ang_range

    if qtype == QuestionType.COMPARATIVE and not extra_filters:
        passages = _comparative_retrieve(retrieval_query, k=k)
    else:
        passages = retrieve(retrieval_query, k=k, **extra_filters)

    if not passages:
        return {
            "answer": (
                "I was unable to find relevant passages in Sri Guru Granth Sahib Ji "
                "for your question. This topic may not be addressed in SGGS, or try "
                "rephrasing. For guidance, consult a qualified Granthi."
            ),
            "failed_quotes": [],
            "sources": [],
            "question_type": qtype.value,
        }

    context_text = _format_passages(passages)
    user_content = (
        f"## Relevant passages from Sri Guru Granth Sahib Ji\n\n"
        f"{context_text}\n\n"
        f"---\n\n"
        f"## Question\n\n{question}"
    )

    messages: list[dict] = list(clean_history)
    messages.append({"role": "user", "content": user_content})

    try:
        raw_answer = _llm_call(
            system=_SYSTEM_PROMPTS[qtype],
            messages=messages,
        )
    except Exception as exc:
        logger.error("LLM API error (%s): %s", PROVIDER, exc)
        raise

    verified_answer, failed_quotes = verify_answer(raw_answer)

    if failed_quotes:
        logger.warning("Removed %d unverified quote(s): %s", len(failed_quotes), failed_quotes)

    # Deduplicate sources by shabad_id (keep highest score)
    seen_shabad: dict[int, dict] = {}
    for p in passages:
        sid = p.shabad_id
        if sid not in seen_shabad or p.score > seen_shabad[sid]["score"]:
            seen_shabad[sid] = {
                "shabad_id": sid,
                "ang": p.ang,
                "raag": p.raag,
                "writer": p.writer,
                "score": round(p.score, 4),
            }
    sources = sorted(seen_shabad.values(), key=lambda s: s["score"], reverse=True)

    return {
        "answer": verified_answer,
        "failed_quotes": failed_quotes,
        "sources": sources,
        "question_type": qtype.value,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if len(sys.argv) < 2:
        print("Usage: python -m src.rag \"<question>\"")
        sys.exit(1)
    question = " ".join(sys.argv[1:])
    print(f"\nQuestion: {question}\n{'─' * 60}")
    result = ask(question)
    print(f"[Agent: {result['question_type']}]\n")
    print(result["answer"])
    print(f"\n{'─' * 60}")
    print(f"Sources ({len(result['sources'])}):")
    for s in result["sources"]:
        print(f"  Ang {s['ang']} · {s['raag']} · {s['writer']}  (score={s['score']})")
    if result["failed_quotes"]:
        print(f"\n⚠  {len(result['failed_quotes'])} unverified quote(s) removed.")


if __name__ == "__main__":
    main()
