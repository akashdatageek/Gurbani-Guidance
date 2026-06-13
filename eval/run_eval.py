"""Evaluation runner for Gurbani RAG.

Metrics:
    1. Retrieval hit-rate: expected_term in top-8 passages OR expected_ang retrieved.
    2. Verification rate: zero failed_quotes on non-adversarial items.
    3. Adversarial behaviour: responses printed for manual review.

Exit codes: 0 pass | 1 fail
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from src.retrieve import retrieve

_ask = None


def _load_ask():
    global _ask
    if _ask is None:
        from src.rag import ask as fn
        _ask = fn
    return _ask


def _passages_contain_term(passages, term: str) -> bool:
    term_lower = term.lower()
    for p in passages:
        combined = " ".join(p.translation_en + p.gurmukhi).lower()
        if term_lower in combined:
            return True
    return False


def _passages_contain_ang(passages, ang: int) -> bool:
    return any(p.ang == ang or ang in p.line_angs for p in passages)


def _print_row(qid, category, status, detail=""):
    print(f"  [{'PASS' if status else 'FAIL'}] {qid} ({category}): {detail}")


def run_retrieval_eval(questions: list[dict]) -> tuple[int, int]:
    evaluable = [q for q in questions if not q.get("out_of_scope") and not q.get("adversarial")]
    print(f"\n=== Retrieval Evaluation ({len(evaluable)} questions) ===\n")
    hits = total = 0
    for q in evaluable:
        passages = retrieve(q["question"], k=8)
        terms_hit = any(_passages_contain_term(passages, t) for t in q.get("expected_terms", []))
        angs_hit = any(_passages_contain_ang(passages, a) for a in q.get("expected_angs", []))
        hit = terms_hit or angs_hit or (not q.get("expected_terms") and not q.get("expected_angs"))
        total += 1
        if hit:
            hits += 1
        _print_row(q["id"], q["category"], hit,
                   detail=f"terms={q.get('expected_terms', [])[:3]} angs={q.get('expected_angs', [])}")
    rate = hits / total * 100 if total else 0.0
    print(f"\nRetrieval hit-rate: {hits}/{total} = {rate:.1f}%")
    return hits, total


def run_rag_eval(questions: list[dict]) -> tuple[int, int]:
    ask = _load_ask()
    non_adversarial = [q for q in questions if not q.get("adversarial")]
    print(f"\n=== RAG + Verification Evaluation ({len(non_adversarial)} questions) ===\n")
    clean = total = 0
    for q in non_adversarial:
        result = ask(q["question"])
        total += 1
        has_failed = len(result.get("failed_quotes", [])) > 0
        if not has_failed:
            clean += 1
        _print_row(q["id"], q["category"], not has_failed,
                   detail="all quotes verified" if not has_failed else f"FAILED: {result['failed_quotes']}")
    rate = clean / total * 100 if total else 0.0
    print(f"\nVerification rate: {clean}/{total} = {rate:.1f}%")
    return clean, total


def run_adversarial_review(questions: list[dict]) -> None:
    """Print adversarial responses for manual behaviour review (spec Phase 7.2)."""
    ask = _load_ask()
    adversarial = [q for q in questions if q.get("adversarial")]
    if not adversarial:
        return
    print(f"\n=== Adversarial Behaviour Review ({len(adversarial)} questions — manual check) ===\n")
    for q in adversarial:
        result = ask(q["question"])
        print(f"  [{q['id']}] {q['category']} | agent={result.get('question_type')}")
        print(f"  Q: {q['question']}")
        print(f"  A: {textwrap.shorten(result['answer'], 300)}")
        print(f"  Notes: {q.get('notes', '')[:120]}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--yaml", default=os.path.join(os.path.dirname(__file__), "golden_questions.yaml"))
    parser.add_argument("--skip-adversarial", action="store_true")
    args = parser.parse_args()

    with open(args.yaml, "r", encoding="utf-8") as f:
        questions = yaml.safe_load(f)["questions"]
    print(f"Loaded {len(questions)} questions from {args.yaml}")

    exit_code = 0

    hits, total = run_retrieval_eval(questions)
    if total and hits / total < 0.80:
        print(f"\nFAIL: Retrieval hit-rate {hits/total:.1%} < 80%")
        exit_code = 1

    if not args.retrieval_only:
        clean, total_rag = run_rag_eval(questions)
        if total_rag and clean < total_rag:
            print(f"\nFAIL: {total_rag - clean} non-adversarial answer(s) had stripped quotes")
            exit_code = 1
        if not args.skip_adversarial:
            run_adversarial_review(questions)

    print("\n" + "=" * 60)
    print("EVALUATION PASSED" if exit_code == 0 else "EVALUATION FAILED — see above")
    print("=" * 60)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
