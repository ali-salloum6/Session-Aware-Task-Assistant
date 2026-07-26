from __future__ import annotations

from assistant.eval.types import ModeMetrics, ScenarioOutcome

# Scenarios that should appear in the fair B0/B1/B2 comparison table.
SHARED_CORE_CATEGORIES = frozenset(
    {
        "single-session",
        "scratchpad",
        "context-efficiency",
        "abstention",
    }
)


def _is_shared_core(o: ScenarioOutcome) -> bool:
    return (
        not o.expected_fail
        and o.category in SHARED_CORE_CATEGORIES
    )


def compute_metrics(
    outcomes: list[ScenarioOutcome],
    *,
    scope: str = "shared_core",
) -> dict[str, ModeMetrics]:
    by_mode: dict[str, list[ScenarioOutcome]] = {}
    for o in outcomes:
        by_mode.setdefault(o.mode, []).append(o)

    # Context efficiency: shared context-efficiency scenarios that passed under
    # both full_history (B1) and memory (B2).
    b1_ok = {
        o.scenario_id: o
        for o in by_mode.get("full_history", [])
        if not o.expected_fail and o.passed and o.category == "context-efficiency"
    }
    b2_ok = {
        o.scenario_id: o
        for o in by_mode.get("memory", [])
        if not o.expected_fail and o.passed and o.category == "context-efficiency"
    }
    shared_ids = sorted(set(b1_ok) & set(b2_ok))
    b1_shared_mean = (
        sum(b1_ok[i].tokens_total for i in shared_ids) / len(shared_ids)
        if shared_ids
        else None
    )
    b2_shared_mean = (
        sum(b2_ok[i].tokens_total for i in shared_ids) / len(shared_ids)
        if shared_ids
        else None
    )
    shared_efficiency = None
    if b1_shared_mean and b2_shared_mean is not None and b1_shared_mean > 0:
        shared_efficiency = (b1_shared_mean - b2_shared_mean) / b1_shared_mean

    # Fair comparison: only scenarios run under all three baselines.
    triple_ids = set(o.scenario_id for o in outcomes if not o.expected_fail)
    for mode in ("stateless", "full_history", "memory"):
        triple_ids &= {
            o.scenario_id
            for o in by_mode.get(mode, [])
            if not o.expected_fail
        }

    metrics: dict[str, ModeMetrics] = {}
    for mode, items in by_mode.items():
        if scope == "all":
            normal = [o for o in items if not o.expected_fail]
        else:
            normal = [
                o
                for o in items
                if not o.expected_fail
                and o.scenario_id in triple_ids
                and _is_shared_core(o)
            ]
            if not normal:
                normal = [o for o in items if not o.expected_fail]

        expected_fails = [o for o in items if o.expected_fail]

        n = len(normal) or 1
        task_completion = sum(1 for o in normal if o.passed) / n

        recall_checks = []
        for o in normal:
            for t in o.turn_outcomes:
                for c in t.checks:
                    if c.name == "must_recall":
                        recall_checks.append(c.passed)
        if mode == "memory" and not recall_checks:
            for o in items:
                if o.expected_fail:
                    continue
                for t in o.turn_outcomes:
                    for c in t.checks:
                        if c.name == "must_recall":
                            recall_checks.append(c.passed)

        recall_accuracy = (
            sum(1 for p in recall_checks if p) / len(recall_checks)
            if recall_checks
            else 1.0
        )

        faith_flags = []
        for o in normal:
            turn_ok = True
            for t in o.turn_outcomes:
                for c in t.checks:
                    if c.name in {"forbid_claims", "must_abstain"} and not c.passed:
                        turn_ok = False
            faith_flags.append(turn_ok)
        faithfulness = (
            sum(1 for p in faith_flags if p) / len(faith_flags) if faith_flags else 1.0
        )

        passed_normal = [o for o in normal if o.passed]
        mean_tokens = (
            sum(o.tokens_total for o in passed_normal) / len(passed_normal)
            if passed_normal
            else 0.0
        )

        efficiency = shared_efficiency if mode == "memory" else None

        metrics[mode] = ModeMetrics(
            mode=mode,
            n_scenarios=len(normal),
            task_completion=task_completion,
            recall_accuracy=recall_accuracy,
            faithfulness=faithfulness,
            mean_tokens=mean_tokens,
            context_efficiency_vs_b1=efficiency,
            n_expected_fail_caught=sum(1 for o in expected_fails if o.passed),
            n_expected_fail=len(expected_fails),
        )
    return metrics


def metrics_markdown(
    metrics: dict[str, ModeMetrics],
    *,
    kind: str = "harness",
) -> str:
    if kind == "live":
        blurb = (
            "**Live system metrics** (real LLM via OpenRouter). "
            "These are the numbers for the report’s quality claims."
        )
    else:
        blurb = (
            "**Offline harness / regression** (scripted tool calls + observation-only "
            "finals). Use to prove the checker works — not as model quality scores. "
            "Shared-core rows = scenarios run on all three baselines."
        )
    lines = [
        blurb,
        "",
        "| Mode | N | Task completion | Recall accuracy | Faithfulness | Mean tokens | vs B1 |",
        "|------|---:|----------------:|----------------:|-------------:|------------:|------:|",
    ]
    order = ["stateless", "full_history", "memory"]
    for mode in order:
        if mode not in metrics:
            continue
        m = metrics[mode]
        eff = (
            f"{100 * m.context_efficiency_vs_b1:.1f}%"
            if m.context_efficiency_vs_b1 is not None
            else "—"
        )
        lines.append(
            f"| {m.mode} | {m.n_scenarios} | {100 * m.task_completion:.1f}% | "
            f"{100 * m.recall_accuracy:.1f}% | {100 * m.faithfulness:.1f}% | "
            f"{m.mean_tokens:.0f} | {eff} |"
        )
    for mode, m in metrics.items():
        if m.n_expected_fail:
            lines.append(
                f"\nNegative controls ({mode}): "
                f"{m.n_expected_fail_caught}/{m.n_expected_fail} expected failures correctly caught."
            )
    return "\n".join(lines)


def metrics_to_dict(metrics: dict[str, ModeMetrics]) -> dict:
    return {
        mode: {
            "mode": m.mode,
            "n_scenarios": m.n_scenarios,
            "task_completion": m.task_completion,
            "recall_accuracy": m.recall_accuracy,
            "faithfulness": m.faithfulness,
            "mean_tokens": m.mean_tokens,
            "context_efficiency_vs_b1": m.context_efficiency_vs_b1,
            "n_expected_fail_caught": m.n_expected_fail_caught,
            "n_expected_fail": m.n_expected_fail,
        }
        for mode, m in metrics.items()
    }
