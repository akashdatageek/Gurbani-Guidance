"""End-to-end model-output benchmark for Gurbani Guidance.

Runs benchmark_questions.yaml through the FULL ask() pipeline in a proper
test loop and checks the ANSWERS a user would receive:

  Safety checks (every case, must be 100%):
    - no leaked <tuk> tags in the final answer
    - zero stripped quotes (failed_quotes empty)
    - every Gurmukhi run in the final answer independently RE-verified
      against the corpus (defense-in-depth re-check of the verifier)

  Behaviour checks (per-case expectations):
    - routing (question_type), Gurmukhi presence, sources, Rehat notice,
      refusal shape

  Content checks (real-model runs only — skipped in --mock):
    - expected terms / cited angs / writer voices, forbidden phrasings

Loop structure:
    for run in 1..RUNS:            # repeat to expose LLM nondeterminism
        for case in cases:
            for attempt in 1..RETRIES:   # transient API failures
                ask() -> check -> record
    aggregate -> per-case pass-rate -> flaky report -> summary -> exit code

Exit code 0 only when the overall pass-rate meets --threshold AND there are
ZERO safety-check failures across all runs.

Modes:
    (default)      real LLM (needs ANTHROPIC_API_KEY / GEMINI_API_KEY)
    --mock         deterministic fake LLM built from the actually-retrieved
                   passages — validates the whole pipeline without a key
    --force-scan-retrieval   pure-python lexical retriever (CI: no torch/
                   chromadb needed; auto-fallback if the index is missing)

Usage:
    python eval/benchmark.py [--mock] [--runs N] [--retries N]
                             [--threshold 0.8] [--only b001,b301]
                             [--deep] [--report PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from src.corpus import load_shabads, make_windows, normalize_gurmukhi
from src.config import WINDOW_SIZE, WINDOW_OVERLAP

_GURMUKHI_RUN = re.compile(r"[਀-੿][਀-੿\s।॥]*[਀-੿।॥]")
_PASSAGE_LINE = re.compile(r"\[Ang (\d+)\] Gurmukhi: (.+)")


# ---------------------------------------------------------------------------
# Mock LLM + scan retriever (deterministic, key-free)
# ---------------------------------------------------------------------------

def _mock_llm(system: str, messages: list[dict], max_tokens: int = 4000) -> str:
    """Deterministic 'model': quotes the first two retrieved passages back
    with correct <tuk> tags, honouring the Rehat-notice instruction."""
    user = messages[-1]["content"]
    pairs = _PASSAGE_LINE.findall(user)
    parts = ["Gurbani offers this guidance on your question."]
    for ang, gurmukhi in pairs[:2]:
        parts.append(f'<tuk ang="{ang}">{gurmukhi.strip()}</tuk>')
        parts.append("Translation: as the Guru teaches us.")
    if "sgpc.net/rehat_maryada" in system:
        parts.append(
            "Questions about specific rules of Sikh conduct are answered by the "
            "Sikh Rehat Maryada: https://www.sgpc.net/rehat_maryada/ "
            "For personal guidance, please consult a qualified Granthi or Giani."
        )
    parts.append("May Waheguru bless you with understanding.")
    return "\n\n".join(parts)


def _mock_classify(prompt: str, max_tokens: int = 16) -> str:
    return "CONCEPTUAL"


class _ScanRetriever:
    """Pure-python lexical fallback retriever (CI has no torch/chromadb).

    Scores shabads by content-word hits over transliteration+translation+
    gurmukhi and returns the first window of the top-k shabads.
    """

    def __init__(self):
        self._shabads = list(load_shabads())

    def __call__(self, question: str, k: int = 8, **filters):
        from src.retrieve import Passage
        words = [w for w in re.findall(r"[\w਀-੿]+", question.lower()) if len(w) > 2]
        scored = []
        for s in self._shabads:
            hay = f"{s.transliteration} {s.translation_en} {s.gurmukhi}".lower()
            score = sum(hay.count(w) for w in words)
            if score > 0:
                scored.append((score, s))
        scored.sort(key=lambda x: -x[0])
        passages = []
        for score, s in scored[:k]:
            window = make_windows(s.lines, WINDOW_SIZE, WINDOW_OVERLAP)[0]
            passages.append(Passage(
                shabad_id=s.shabad_id, ang=s.ang, raag=s.raag, writer=s.writer,
                gurmukhi=[l.gurmukhi for l in window],
                translation_en=[l.translation_en for l in window],
                line_angs=[l.ang for l in window],
                score=float(score),
            ))
        return passages


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

SAFETY_CHECKS = {"no_leaked_tuk_tags", "no_stripped_quotes", "gurmukhi_reverified"}


def check_case(case: dict, result: dict, mock: bool) -> list[dict]:
    from src.verify import verify_answer

    exp = case.get("expect") or {}
    answer = result.get("answer", "")
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str = ""):
        checks.append({"name": name, "ok": bool(ok), "detail": detail[:120]})

    # ── Safety (always) ─────────────────────────────────────────────────
    add("no_leaked_tuk_tags", "<tuk" not in answer)
    add("no_stripped_quotes", not result.get("failed_quotes"),
        str(result.get("failed_quotes"))[:100])
    # Defense-in-depth: run the final answer through verification again —
    # it must come back byte-identical with nothing newly stripped.
    recleaned, refailed = verify_answer(answer)
    add("gurmukhi_reverified", not refailed and recleaned == answer,
        f"refailed={refailed[:1]}" if refailed else "")

    # ── Behaviour ───────────────────────────────────────────────────────
    if "question_type" in exp:
        add("routing", result.get("question_type") == exp["question_type"],
            f"got {result.get('question_type')}")
    if exp.get("refusal"):
        add("refusal_no_sources", result.get("sources") == [])
    if exp.get("gurmukhi"):
        add("gurmukhi_present", bool(_GURMUKHI_RUN.search(answer)))
    if exp.get("sources"):
        add("sources_present", bool(result.get("sources")))
    if exp.get("rehat_notice"):
        add("rehat_notice", "sgpc.net/rehat_maryada" in answer)

    # ── Content (real model only) ───────────────────────────────────────
    if not mock:
        low = answer.lower()
        if exp.get("terms_any"):
            add("terms_any", any(t.lower() in low for t in exp["terms_any"]),
                str(exp["terms_any"]))
        if exp.get("angs_any"):
            cited = {int(a) for a in re.findall(r"[Aa]ng\s+(\d+)", answer)}
            cited |= {s.get("ang") for s in result.get("sources", [])}
            add("angs_any", bool(cited & {int(a) for a in exp["angs_any"]}),
                f"cited={sorted(cited)[:6]}")
        if exp.get("writers_any"):
            add("writers_any", any(w.lower() in low for w in exp["writers_any"]),
                str(exp["writers_any"]))
        if exp.get("forbid_terms"):
            hits = [t for t in exp["forbid_terms"] if t.lower() in low]
            add("forbidden_absent", not hits, str(hits))

    return checks


# ---------------------------------------------------------------------------
# Runner loop
# ---------------------------------------------------------------------------

def run_benchmark(args) -> int:
    with open(args.yaml, "r", encoding="utf-8") as f:
        cases = yaml.safe_load(f)["cases"]
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    print(f"Benchmark: {len(cases)} cases × {args.runs} run(s)"
          f"{' [MOCK model]' if args.mock else ''}"
          f"{' [scan retrieval]' if args.force_scan_retrieval else ''}\n")

    import src.rag as rag

    if args.mock:
        rag._llm_call = _mock_llm
        rag._llm_lightweight_call = _mock_classify

    if args.force_scan_retrieval:
        rag.retrieve = _ScanRetriever()
    else:
        # Auto-fallback: if the local index can't initialise, scan-retrieve
        try:
            from src.retrieve import init as retrieval_init
            retrieval_init()
        except Exception as exc:  # noqa: BLE001
            print(f"NOTE: local index unavailable ({exc}); using scan retrieval.\n")
            rag.retrieve = _ScanRetriever()

    records: list[dict] = []
    safety_failures = 0

    # The loop: runs × cases × retries
    for run_idx in range(1, args.runs + 1):
        print(f"── Run {run_idx}/{args.runs} " + "─" * 40)
        for case in cases:
            result, error = None, None
            for attempt in range(1, args.retries + 2):
                try:
                    t0 = time.monotonic()
                    result = (rag.deep_ask if args.deep else rag.ask)(case["question"])
                    latency = time.monotonic() - t0
                    break
                except Exception as exc:  # noqa: BLE001 — retry transient failures
                    error = f"{type(exc).__name__}: {exc}"
                    if attempt <= args.retries:
                        time.sleep(2 * attempt)

            if result is None:
                records.append({"run": run_idx, "id": case["id"],
                                "category": case["category"], "ok": False,
                                "error": error, "checks": []})
                print(f"  [ERR ] {case['id']} ({case['category']}): {error}")
                continue

            checks = check_case(case, result, mock=args.mock)
            failed = [c for c in checks if not c["ok"]]
            safety_failed = [c for c in failed if c["name"] in SAFETY_CHECKS]
            safety_failures += len(safety_failed)
            ok = not failed
            records.append({"run": run_idx, "id": case["id"],
                            "category": case["category"], "ok": ok,
                            "latency_s": round(latency, 2), "checks": checks})
            flag = "PASS" if ok else ("SAFE!" if safety_failed else "FAIL")
            detail = "" if ok else "; ".join(f"{c['name']}({c['detail']})" for c in failed[:3])
            print(f"  [{flag:5}] {case['id']} ({case['category']}) {detail}")

    # ── Aggregate ───────────────────────────────────────────────────────
    by_case: dict[str, list[bool]] = {}
    by_cat: dict[str, list[bool]] = {}
    for r in records:
        by_case.setdefault(r["id"], []).append(r["ok"])
        by_cat.setdefault(r["category"], []).append(r["ok"])

    total = len(records)
    passed = sum(1 for r in records if r["ok"])
    rate = passed / total if total else 0.0
    flaky = [cid for cid, oks in by_case.items() if any(oks) and not all(oks)]

    print("\n" + "=" * 60)
    print(f"{'Category':24} {'pass':>6}")
    for cat in sorted(by_cat):
        oks = by_cat[cat]
        print(f"{cat:24} {sum(oks)}/{len(oks)}")
    print("-" * 60)
    print(f"Overall: {passed}/{total} = {rate:.1%}   "
          f"safety failures: {safety_failures}   flaky: {flaky or 'none'}")

    report = {
        "mock": args.mock, "runs": args.runs, "deep": args.deep,
        "total": total, "passed": passed, "pass_rate": round(rate, 4),
        "safety_failures": safety_failures, "flaky_cases": flaky,
        "records": records,
    }
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report written to {args.report}")

    if safety_failures:
        print("\nBENCHMARK FAILED — safety checks must pass 100%.")
        return 1
    if rate < args.threshold:
        print(f"\nBENCHMARK FAILED — pass rate {rate:.1%} < {args.threshold:.0%}.")
        return 1
    print("\nBENCHMARK PASSED")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", default=os.path.join(os.path.dirname(__file__), "benchmark_questions.yaml"))
    parser.add_argument("--runs", type=int, default=1, help="repeat the full case loop N times")
    parser.add_argument("--retries", type=int, default=2, help="retries per case on transient errors")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--only", default="", help="comma-separated case ids")
    parser.add_argument("--deep", action="store_true", help="benchmark deep_ask()")
    parser.add_argument("--mock", action="store_true", help="deterministic fake LLM (no API key)")
    parser.add_argument("--force-scan-retrieval", action="store_true",
                        help="pure-python retriever (CI: no index needed)")
    parser.add_argument("--report", default=os.path.join(os.path.dirname(__file__), "benchmark_report.json"))
    args = parser.parse_args()
    sys.exit(run_benchmark(args))


if __name__ == "__main__":
    main()
