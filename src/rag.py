"""Gurbani RAG pipeline with multi-agent routing.

A lightweight router classifies each question, then a specialist agent
(distinct system prompt + retrieval strategy) handles it.

Question types
--------------
CONCEPTUAL  — theology, definitions, concepts (haumai, naam, hukam, seva …)
SITUATIONAL — user in a personal life situation seeking comfort or guidance
COMPARATIVE — compare teachings across Gurus, Bhagats, or raags
REHAT       — conduct/practice/"is X allowed" questions
ADVERSARIAL — fabrication requests, out-of-scope history

Public API
----------
ask(question, history=None, **filters) -> dict
    {answer, failed_quotes, sources, question_type}

CLI
---
python -m src.rag "What does Gurbani say about haumai?"
"""

from __future__ import annotations

import logging
import re
import sys
from enum import Enum
from typing import Any

import anthropic

from src.config import ANTHROPIC_API_KEY, CLAUDE_MODEL, MAX_TOKENS, TOP_K
from src.retrieve import Passage, retrieve
from src.verify import verify_answer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Question type enum
# ---------------------------------------------------------------------------


class QuestionType(str, Enum):
    CONCEPTUAL = "conceptual"
    SITUATIONAL = "situational"
    COMPARATIVE = "comparative"
    REHAT = "rehat"
    ADVERSARIAL = "adversarial"


# ---------------------------------------------------------------------------
# Classifier — fast regex, no extra API call
# ---------------------------------------------------------------------------

_ADVERSARIAL_RE = re.compile(
    r"\b(make up|fabricate|invent|compose|write a new|create a new?|generate a)\b"
    r".*\b(shabad|verse|tuk|line|gurbani|scripture|bani)\b"
    r"|\b(janamsakhi|janamsakhis|birth stor|historical account|biography of|life of guru"
    r"|when was guru|where was guru|who killed)\b",
    re.I | re.S,
)

_REHAT_RE = re.compile(
    r"\b(is it (allowed|permitted|ok|okay|right|wrong|permissible|acceptable|haram|forbidden))"
    r"|\b(can (i|we|sikhs?|a sikh))\b"
    r"|\b(are sikhs? (allowed|permitted|supposed|expected))\b"
    r"|\b(rehat|maryada|code of conduct|dress code|5 k[s']|panj kakkars?)\b"
    r"|\b(eating meat|cut.*hair|consuming alcohol|drink.*alcohol|premarital)\b",
    re.I,
)

_COMPARATIVE_RE = re.compile(
    r"\b(compare|comparison|contrast|differences? between|similarities? between|"
    r"how do (different|various)|what did (each|different)|"
    r"across (the )?gurus?|multiple gurus?|different (writers?|gurus?|bhagats?)|"
    r"(guru|bhagat|sant)\b.{1,40}\b(and|vs\.?|versus|compared to)\b.{1,40}\b(guru|bhagat|sant|ji)\b)",
    re.I | re.S,
)

_SITUATIONAL_RE = re.compile(
    r"\b(i[' ]?m (feeling|going through|struggling|suffering|grieving|dealing with|facing|lost|broken|alone|scared|helpless|hopeless))"
    r"|\b(i am (feeling|going through|struggling|suffering|grieving|facing|lost|scared))\b"
    r"|\b(i[' ]?ve (been feeling|been struggling|lost (my|a)|failed|made a mistake))\b"
    r"|\b(help me (with|through|deal with|cope with|get through|understand my))\b"
    r"|\bmy (grief|loss|pain|suffering|depression|anxiety|fear|anger|guilt|loneliness|sorrow|death of|divorce)\b",
    re.I,
)


def classify_question(question: str) -> QuestionType:
    """Classify a question into one of five types using fast regex matching."""
    if _ADVERSARIAL_RE.search(question):
        return QuestionType.ADVERSARIAL
    if _REHAT_RE.search(question):
        return QuestionType.REHAT
    if _COMPARATIVE_RE.search(question):
        return QuestionType.COMPARATIVE
    if _SITUATIONAL_RE.search(question):
        return QuestionType.SITUATIONAL
    return QuestionType.CONCEPTUAL


# ---------------------------------------------------------------------------
# Specialist system prompts
# ---------------------------------------------------------------------------

_BASE_RULES = """\
## Core rules (non-negotiable)

1. **Answer ONLY from the passages provided.** Do not draw on outside memory \
   or compose Gurmukhi lines yourself. If no relevant passage is given, say so \
   honestly.

2. **Quote exactly.** When you cite a line from SGGS, wrap it in:
       <tuk ang="ANG_NUMBER">ਗੁਰਮੁਖੀ ਤੁਕ</tuk>
   Use the ang number from the passage header. Copy the Gurmukhi exactly — never \
   paraphrase or reconstruct it.

3. **Gurmukhi first, translation second.** Always show the original Gurmukhi \
   before its English translation.

4. **Describe, never rule.** Explain what scripture says; do not issue personal \
   edicts or judge the user's choices.

5. **Rehat questions → redirect.** For questions about Sikh code of conduct, \
   dress, ceremonies, or rituals, always direct the user to the Sikh Rehat \
   Maryada: https://www.sgpc.net/rehat_maryada/

6. **Tone.** Be humble, thoughtful, and reverential at all times.\
"""

_SYSTEM_PROMPTS: dict[QuestionType, str] = {
    QuestionType.CONCEPTUAL: f"""\
You are Gurbani Guidance — a respectful, scholarly assistant dedicated to \
Sri Guru Granth Sahib Ji (SGGS).

{_BASE_RULES}

## Format for conceptual questions
- Open with a brief framing of the concept in Gurbani.
- Present the most relevant Gurmukhi passages with <tuk> tags, then their \
  English translations, then citation (Ang N, Raag X, written by Y).
- Close with a paragraph that synthesises what the passages together convey.
- Use natural, flowing prose — do not number every paragraph.
""",

    QuestionType.SITUATIONAL: f"""\
You are Gurbani Guidance — a compassionate, respectful presence drawing on \
Sri Guru Granth Sahib Ji (SGGS) to offer spiritual comfort and perspective.

{_BASE_RULES}

## Guidance for situational questions
The user is sharing a personal struggle or difficult moment. Your response should:
- Begin by acknowledging their situation with warmth and without judgment.
- Then offer passages from SGGS that speak directly to their experience, with \
  <tuk> tags + translations + citations.
- Frame each passage gently: "Guru Ji reminds us…", "In this bani, we are told…"
- Close with a grounded, encouraging word rooted in what the scripture says.
- Do NOT give generic self-help advice. All comfort must flow from the passages provided.
""",

    QuestionType.COMPARATIVE: f"""\
You are Gurbani Guidance — a scholarly assistant skilled in comparative analysis \
of Sri Guru Granth Sahib Ji (SGGS).

{_BASE_RULES}

## Format for comparative questions
The user wants to compare how different Gurus, Bhagats, or raags address a topic.
- Organise your answer by author/raag — one section per voice present in the passages.
- For each: quote a representative tuk with <tuk> tags, provide the translation, \
  then briefly interpret what that specific author emphasises.
- End with a short synthesis paragraph noting the unity and any distinctive \
  emphases across the voices.
- Only compare voices that are actually present in the provided passages.
""",

    QuestionType.REHAT: f"""\
You are Gurbani Guidance — a respectful assistant that grounds answers in \
Sri Guru Granth Sahib Ji (SGGS) while always directing conduct questions to \
the proper authority.

{_BASE_RULES}

## Handling conduct / Rehat questions
- First, provide any scriptural passages that speak to the underlying values or \
  principles at issue (with <tuk> tags + translations + citations).
- Then ALWAYS close with this notice (word for word):

  "Questions about specific rules of Sikh conduct — including the Panj Kakkars, \
   dietary practice, ceremonies, and daily discipline — are answered by the \
   Sikh Rehat Maryada, the official Sikh code of conduct approved by the Akal \
   Takht. You can read it in full at: https://www.sgpc.net/rehat_maryada/ \
   For personal guidance, please consult a qualified Granthi or Giani."

- The model must never say whether a specific practice is "allowed" or "forbidden".
""",

    QuestionType.ADVERSARIAL: "",  # not used — immediate refusal below
}


# ---------------------------------------------------------------------------
# Comparative retrieval helper
# ---------------------------------------------------------------------------

_KNOWN_WRITERS = [
    "Guru Nanak Dev Ji",
    "Guru Angad Dev Ji",
    "Guru Amar Das Ji",
    "Guru Ram Das Ji",
    "Guru Arjan Dev Ji",
    "Guru Tegh Bahadur Ji",
    "Bhagat Kabir Ji",
    "Bhagat Ravidas Ji",
    "Bhagat Namdev Ji",
    "Bhagat Farid Ji",
    "Bhagat Trilochan Ji",
    "Bhagat Dhanna Ji",
    "Bhagat Beni Ji",
]

_WRITER_PATTERNS = {
    w: re.compile(
        r"\b" + re.escape(w.split()[0]) + r"\b"  # match first word (Guru/Bhagat/name)
        + (r"|\b" + re.escape(w.split()[1]) + r"\b" if len(w.split()) > 1 else ""),
        re.I,
    )
    for w in _KNOWN_WRITERS
}


def _comparative_retrieve(question: str, k: int) -> list[Passage]:
    """For comparative questions, retrieve per mentioned writer then merge."""
    mentioned = [w for w, pat in _WRITER_PATTERNS.items() if pat.search(question)]

    passages_by_writer: dict[str, list[Passage]] = {}

    if mentioned:
        per_writer_k = max(3, k // max(len(mentioned), 1))
        for writer in mentioned:
            results = retrieve(question, k=per_writer_k, writer=writer)
            passages_by_writer[writer] = results
    else:
        # No specific writers named — broad retrieve with more results
        broad = retrieve(question, k=k + 4)
        for p in broad:
            passages_by_writer.setdefault(p.writer, []).append(p)

    # Merge: up to ceil(k / writers) per writer, total ≤ k
    merged: list[Passage] = []
    writers = list(passages_by_writer.keys())
    per_slot = max(1, k // max(len(writers), 1))
    for w in writers:
        merged.extend(passages_by_writer[w][:per_slot])
    merged = merged[:k]

    # Pad with unfiltered results if under k
    if len(merged) < k:
        extra = retrieve(question, k=k - len(merged))
        seen_ids = {p.shabad_id for p in merged}
        for p in extra:
            if p.shabad_id not in seen_ids:
                merged.append(p)
                if len(merged) >= k:
                    break

    return merged


# ---------------------------------------------------------------------------
# Passage → prompt context
# ---------------------------------------------------------------------------


def _format_passages(passages: list[Passage]) -> str:
    blocks: list[str] = []
    for i, p in enumerate(passages, 1):
        header = f"[Passage {i}] Ang {p.ang} · {p.raag} · {p.writer}"
        lines_block = "\n".join(
            f"  Gurmukhi: {g}\n  Translation: {e}"
            for g, e in zip(p.gurmukhi, p.translation_en)
        )
        blocks.append(f"{header}\n{lines_block}")
    return "\n\n".join(blocks)


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
    """Run the full RAG pipeline for a question.

    Args:
        question: User question in any language.
        history: Optional list of {role, content} dicts (cap: 10 turns).
        writer: Optional passage filter.
        raag: Optional passage filter.
        ang_range: Optional (start_ang, end_ang) filter.
        k: Number of passages to retrieve.

    Returns:
        dict: {answer, failed_quotes, sources, question_type}
    """
    qtype = classify_question(question)
    logger.info("Question classified as: %s", qtype.value)

    # --- Adversarial: immediate refusal, no retrieval ---
    if qtype == QuestionType.ADVERSARIAL:
        return {
            "answer": (
                "I'm here to help you explore what Sri Guru Granth Sahib Ji says "
                "on a topic — but I can't compose, fabricate, or attribute new verses "
                "to the Gurus or Bhagats. Every Gurmukhi line I show you must exist "
                "verbatim in the scripture.\n\n"
                "If you have a question about Gurbani's teachings, I'd be glad to help."
            ),
            "failed_quotes": [],
            "sources": [],
            "question_type": qtype.value,
        }

    # --- Retrieve passages (strategy depends on type) ---
    extra_filters: dict[str, Any] = {}
    if writer:
        extra_filters["writer"] = writer
    if raag:
        extra_filters["raag"] = raag
    if ang_range:
        extra_filters["ang_range"] = ang_range

    if qtype == QuestionType.COMPARATIVE and not extra_filters:
        passages = _comparative_retrieve(question, k=k)
    else:
        passages = retrieve(question, k=k, **extra_filters)

    if not passages:
        return {
            "answer": (
                "I was unable to find relevant passages in Sri Guru Granth Sahib Ji "
                "for your question. Please try rephrasing, or consult a Granthi for guidance."
            ),
            "failed_quotes": [],
            "sources": [],
            "question_type": qtype.value,
        }

    context_text = _format_passages(passages)

    # --- Build messages ---
    user_content = (
        f"## Relevant passages from Sri Guru Granth Sahib Ji\n\n"
        f"{context_text}\n\n"
        f"---\n\n"
        f"## Question\n\n{question}"
    )

    messages: list[dict] = []
    if history:
        messages.extend(history[-20:])
    messages.append({"role": "user", "content": user_content})

    # --- Call Claude with specialist system prompt ---
    system_prompt = _SYSTEM_PROMPTS[qtype]
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=messages,
        )
        raw_answer = response.content[0].text
    except anthropic.APIError as exc:
        logger.error("Anthropic API error: %s", exc)
        raise

    # --- Verify quotes ---
    verified_answer, failed_quotes = verify_answer(raw_answer)

    if failed_quotes:
        logger.warning(
            "Removed %d unverified quote(s): %s",
            len(failed_quotes),
            failed_quotes,
        )

    sources = [
        {
            "shabad_id": p.shabad_id,
            "ang": p.ang,
            "raag": p.raag,
            "writer": p.writer,
            "score": round(p.score, 4),
        }
        for p in passages
    ]

    return {
        "answer": verified_answer,
        "failed_quotes": failed_quotes,
        "sources": sources,
        "question_type": qtype.value,
    }


# ---------------------------------------------------------------------------
# CLI entry point
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
