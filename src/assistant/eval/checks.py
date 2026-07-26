from __future__ import annotations

from assistant.eval.types import CheckResult, TurnOutcome


def _contains_any(text: str, needles: list[str]) -> bool:
    low = text.lower()
    return any(n.lower() in low for n in needles)


def _contains_all(text: str, needles: list[str]) -> bool:
    low = text.lower()
    return all(n.lower() in low for n in needles)


def evaluate_turn_checks(outcome: TurnOutcome, checks: dict) -> list[CheckResult]:
    """Rule-based checks only (no LLM judge)."""
    results: list[CheckResult] = []
    answer = outcome.answer or ""
    mem_blob = "\n".join(outcome.memory_texts)
    tools_blob = " ".join(
        f"{e.get('tool')} {e.get('result')}" for e in outcome.tool_events
    )
    combined = f"{answer}\n{mem_blob}\n{tools_blob}"

    if "answer_contains_any" in checks:
        needles = list(checks["answer_contains_any"])
        ok = _contains_any(answer, needles)
        results.append(
            CheckResult(
                "answer_contains_any",
                ok,
                f"need any of {needles!r} in answer={answer!r}",
            )
        )

    if "answer_contains_all" in checks:
        needles = list(checks["answer_contains_all"])
        ok = _contains_all(answer, needles)
        results.append(
            CheckResult(
                "answer_contains_all",
                ok,
                f"need all of {needles!r} in answer={answer!r}",
            )
        )

    if "must_recall" in checks:
        needles = list(checks["must_recall"])
        # Evidence must exist outside a bare scripted answer (LTM store, recall tool,
        # or injected recall ids), and the answer must also use the fact.
        evidence = f"{mem_blob}\n{tools_blob}"
        if outcome.recalled_memory_ids:
            evidence += "\n" + " ".join(outcome.recalled_memory_ids)
        in_evidence = _contains_all(evidence, needles)
        in_answer = _contains_all(answer, needles)
        ok = in_evidence and in_answer
        results.append(
            CheckResult(
                "must_recall",
                ok,
                f"need {needles!r} in evidence and answer "
                f"(evidence={in_evidence}, answer={in_answer})",
            )
        )

    if "must_abstain" in checks:
        # Model must not invent these claims.
        forbidden = list(checks["must_abstain"])
        invented = [f for f in forbidden if f.lower() in answer.lower()]
        ok = len(invented) == 0
        results.append(
            CheckResult(
                "must_abstain",
                ok,
                f"forbidden in answer: {invented}" if invented else "ok",
            )
        )

    if "forbid_claims" in checks:
        forbidden = list(checks["forbid_claims"])
        hit = [f for f in forbidden if f.lower() in combined.lower()]
        ok = len(hit) == 0
        results.append(
            CheckResult("forbid_claims", ok, f"found {hit}" if hit else "ok")
        )

    if "memory_texts_contain_any" in checks:
        needles = list(checks["memory_texts_contain_any"])
        ok = _contains_any(mem_blob, needles)
        results.append(
            CheckResult(
                "memory_texts_contain_any",
                ok,
                f"need any of {needles!r} in LTM",
            )
        )

    if "memory_count_max" in checks:
        limit = int(checks["memory_count_max"])
        n = len(outcome.memory_texts)
        ok = n <= limit
        results.append(CheckResult("memory_count_max", ok, f"count={n} max={limit}"))

    if "memory_count_min" in checks:
        limit = int(checks["memory_count_min"])
        n = len(outcome.memory_texts)
        ok = n >= limit
        results.append(CheckResult("memory_count_min", ok, f"count={n} min={limit}"))

    if "tool_called" in checks:
        want = set(checks["tool_called"])
        got = {e.get("tool") for e in outcome.tool_events}
        ok = want.issubset(got)
        results.append(
            CheckResult("tool_called", ok, f"want {want} got {got}")
        )

    return results
