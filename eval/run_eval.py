"""Evaluation runner for Gurbani RAG.

Metrics:
    1. Retrieval hit-rate: expected_term appears in top-8 passages (transliteration
       or translation_en), for questions with expected_terms.
    2. Verification rate: zero failed_quotes on non-adversarial items.

Exit codes:
    0 — all checks pass
    1 — hit-rate < 80% OR non-adversarial answer has a stripped quote

Usage:
    python eval/run_eval.py [--retrieval-only] [--yaml PATH]
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from src.retrieve import retrieve

# Optional: only import ask if running full eval
_ask = None


def _load_ask():
    global _ask
    if _ask is None:
        from src.rag import ask as _ask_fn
        _ask = _ask_fn
    return _ask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _passages_contain_term(passages, term: str) -> bool:
    """Return True if any passage contains the term (case-insensitive)."""
    term_lower = term.lower()
    for p in passages:
        combined = " ".join(p.translation_en + p.gurmukhi).lower()
        if term_lower in combined:
            return True
    return False


def _print_row(qid, category, status, detail=""):
    status_label = "PASS" if status else "FAIL"
    print(f"  [{status_label}] {qid} ({category}): {detail}")


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------


def run_retrieval_eval(questions: list[dict]) -> tuple[int, int]:
    """Return (hits, total) for questions with expected_terms."""
    hits = 0
    total = 0
    evaluable = [
        q for q in questions if q.get("expected_terms") and not q.get("out_of_scope")
    ]
    print(f"\n=== Retrieval Evaluation ({len(evaluable)} questions) ===\n")
    for q in evaluable:
        passages = retrieve(q["question"], k=8)
        expected = q["expected_terms"]
        hit = any(_passages_contain_term(passages, t) for t in expected)
        total += 1
        if hit:
            hits += 1
        _print_row(
            q["id"],
            q["category"],
            hit,
            detail=f"terms={expected[:3]}",
        )
    rate = (hits / total * 100) if total else 0.0
    print(f"\nRetrieval hit-rate: {hits}/{total} = {rate:.1f}%")
    return hits, total


def run_rag_eval(questions: list[dict]) -> tuple[int, int]:
    """Return (clean_answers, total) for non-adversarial questions."""
    ask = _load_ask()
    clean = 0
    total = 0
    non_adversarial = [q for q in questions if not q.get("adversarial")]
    print(f"\n=== RAG + Verification Evaluation ({len(non_adversarial)} questions) ===\n")
    for q in non_adversarial:
        result = ask(q["question"])
        total += 1
        has_failed = len(result.get("failed_quotes", [])) > 0
        if not has_failed:
            clean += 1
        _print_row(
            q["id"],
            q["category"],
            not has_failed,
            detail=(
                f"failed_quotes={result['failed_quotes']}"
                if has_failed
                else "all quotes verified"
            ),
        )
    rate = (clean / total * 100) if total else 0.0
    print(f"\nVerification rate: {clean}/{total} = {rate:.1f}%")
    return clean, total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gurbani RAG evaluation")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Only run retrieval hit-rate (skip Claude calls)",
    )
    parser.add_argument(
        "--yaml",
        default=os.path.join(os.path.dirname(__file__), "golden_questions.yaml"),
        help="Path to golden_questions.yaml",
    )
    args = parser.parse_args()

    with open(args.yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    questions = data["questions"]
    print(f"Loaded {len(questions)} golden questions from {args.yaml}")

    exit_code = 0

    # --- Retrieval evaluation ---
    hits, total = run_retrieval_eval(questions)
    if total > 0:
        hit_rate = hits / total
        if hit_rate < 0.80:
            print(f"\nFAIL: Retrieval hit-rate {hit_rate:.1%} < 80% threshold")
            exit_code = 1

    # --- RAG + verification evaluation ---
    if not args.retrieval_only:
        clean, total_rag = run_rag_eval(questions)
        if total_rag > 0 and clean < total_rag:
            print(
                f"\nFAIL: {total_rag - clean} non-adversarial answer(s) had stripped quotes"
            )
            exit_code = 1

    # --- Summary ---
    print("\n" + "=" * 60)
    if exit_code == 0:
        print("EVALUATION PASSED")
    else:
        print("EVALUATION FAILED — see failures above")
    print("=" * 60)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
