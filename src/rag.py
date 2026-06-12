"""Gurbani RAG pipeline.

Orchestrates retrieval → Claude generation → quote verification.

Public API:
    ask(question, history=None, **filters) -> dict

Returns:
    {
        "answer": str,          # verified answer (fabricated tuk tags replaced)
        "failed_quotes": list,  # list of tuk texts that failed verification
        "sources": list[dict],  # passage metadata for the UI
    }

CLI:
    python -m src.rag "What does Gurbani say about haumai?"
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import anthropic

from src.config import ANTHROPIC_API_KEY, CLAUDE_MODEL, MAX_TOKENS, TOP_K
from src.retrieve import Passage, retrieve
from src.verify import verify_answer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are Gurbani Guidance — a respectful, scholarly assistant dedicated to \
Sri Guru Granth Sahib Ji (SGGS).

## Core rules (non-negotiable)

1. **Answer only from the passages provided.** Do not draw on outside memory \
   or compose Gurmukhi lines yourself.

2. **Quote exactly.** When you cite a line from SGGS, wrap it in:
       <tuk ang="ANG_NUMBER">ਗੁਰਮੁਖੀ ਤੁਕ</tuk>
   Use the ang number from the passage header. Copy the Gurmukhi exactly — \
   do not paraphrase or reconstruct it.

3. **Gurmukhi first, translation second.** Always present the original \
   Gurmukhi before its English translation.

4. **Describe, never rule.** Explain what the scripture says; do not issue \
   edicts on personal conduct or judge the user's choices.

5. **Rehat questions → redirect.** If the question is about Sikh code of \
   conduct, dress, ceremonies, or rituals, politely say: \
   "Questions about Sikh code of conduct are answered by the \
   Sikh Rehat Maryada, available at https://www.sgpc.net/rehat_maryada/"

6. **Scope.** If no relevant passage is provided, say so honestly — \
   do not speculate or fabricate.

7. **Tone.** Be humble, thoughtful, and reverential. \
   Begin answers to sincere spiritual questions with "Waheguru Ji Ka Khalsa, \
   Waheguru Ji Ki Fateh" only when the user has also greeted in that manner.

## Format
- Use clear paragraphs.
- Cite passages inline with <tuk> tags.
- After Gurmukhi citations, provide the English translation from the passage.
- Do not number every paragraph; use natural prose.
"""

# ---------------------------------------------------------------------------
# Passage → prompt context
# ---------------------------------------------------------------------------


def _format_passages(passages: list[Passage]) -> str:
    """Format retrieved passages into a numbered context block."""
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
# RAG pipeline
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
        question: User's question in any language.
        history: Optional list of previous {role, content} dicts (max 10 turns).
        writer: Optional passage filter — restrict to a specific writer.
        raag: Optional passage filter — restrict to a specific raag.
        ang_range: Optional (start_ang, end_ang) filter.
        k: Number of passages to retrieve.

    Returns:
        dict with keys: answer, failed_quotes, sources.
    """
    # 1. Retrieve passages
    passages = retrieve(
        question,
        k=k,
        writer=writer,
        raag=raag,
        ang_range=ang_range,
    )

    if not passages:
        return {
            "answer": (
                "I was unable to find relevant passages in Sri Guru Granth Sahib Ji "
                "for your question. Please try rephrasing, or consult a Granthi for guidance."
            ),
            "failed_quotes": [],
            "sources": [],
        }

    context_text = _format_passages(passages)

    # 2. Build messages
    user_content = (
        f"## Relevant passages from Sri Guru Granth Sahib Ji\n\n"
        f"{context_text}\n\n"
        f"---\n\n"
        f"## Question\n\n{question}"
    )

    messages: list[dict] = []

    # Add rolling history (cap at 10 turns = 20 messages)
    if history:
        messages.extend(history[-20:])

    messages.append({"role": "user", "content": user_content})

    # 3. Call Claude
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            messages=messages,
        )
        raw_answer = response.content[0].text
    except anthropic.APIError as exc:
        logger.error("Anthropic API error: %s", exc)
        raise

    # 4. Verify quotes
    verified_answer, failed_quotes = verify_answer(raw_answer)

    if failed_quotes:
        logger.warning(
            "Removed %d unverified quote(s) from answer: %s",
            len(failed_quotes),
            failed_quotes,
        )

    # 5. Build sources list for the UI
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

    print(result["answer"])
    print(f"\n{'─' * 60}")
    print(f"Sources ({len(result['sources'])}):")
    for s in result["sources"]:
        print(f"  Ang {s['ang']} · {s['raag']} · {s['writer']}  (score={s['score']})")

    if result["failed_quotes"]:
        print(f"\n⚠  {len(result['failed_quotes'])} unverified quote(s) removed.")


if __name__ == "__main__":
    main()
