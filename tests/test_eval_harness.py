"""Phase 8 eval harness: loader, checks, offline runner, metrics."""

from __future__ import annotations

import json
from pathlib import Path

from assistant.config import Settings
from assistant.eval.checks import evaluate_turn_checks
from assistant.eval.loader import load_scenario, load_suite
from assistant.eval.metrics import compute_metrics, metrics_markdown
from assistant.eval.runner import run_scenario
from assistant.eval.synthesize import synthesize_answer
from assistant.eval.types import TurnOutcome

_REPO = Path(__file__).resolve().parents[1]
_SCENARIOS = _REPO / "eval" / "scenarios"


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        api_key="test-key",
        base_url="http://example.invalid/v1",
        model="fake-model",
        max_steps=8,
        kb_dir=_REPO / "kb",
        traces_dir=tmp_path / "traces",
        max_context_tokens=700,
        keep_last_n_turns=1,
        summary_trigger_tokens=120,
        recall_k=1,
        context_mode="bounded",
        memory_dir=tmp_path / "memory",
        default_user_id="eval-user",
        ltm_recall_k=3,
        dedup_distance=0.18,
        reflect_memory=False,
        agent_mode="memory",
        agent_backend="loop",
        checkpoint_path=tmp_path / "checkpoints" / "t.sqlite",
        langfuse_public_key="",
        langfuse_secret_key="",
        langfuse_host="https://cloud.langfuse.com",
    )


def test_suite_has_expected_coverage():
    scenarios = load_suite(_SCENARIOS, suite="full")
    assert 20 <= len(scenarios) <= 30
    cats = {s.category for s in scenarios}
    for need in {
        "single-session",
        "cross-session",
        "deduplication",
        "abstention",
        "scratchpad",
        "context-efficiency",
        "preference",
    }:
        assert need in cats, f"missing category {need}"
    fails = [s for s in scenarios if s.expected_fail]
    assert len(fails) >= 3


def test_shared_core_runs_all_three_modes():
    scenarios = load_suite(_SCENARIOS, suite="full")
    shared = [
        s
        for s in scenarios
        if not s.expected_fail
        and s.category
        in {"single-session", "scratchpad", "context-efficiency", "abstention"}
        and set(s.modes) >= {"stateless", "full_history", "memory"}
    ]
    assert len(shared) >= 8


def test_smoke_suite_smaller_than_full():
    full = load_suite(_SCENARIOS, suite="full")
    smoke = load_suite(_SCENARIOS, suite="smoke")
    assert len(smoke) < len(full)
    assert any(s.expected_fail for s in smoke)


def test_checks_must_abstain_and_forbid():
    outcome = TurnOutcome(
        answer="Your favorite color is blue.",
        tool_events=[],
        recalled_memory_ids=[],
        tokens_in_est=10,
        memory_texts=[],
        checks=[],
    )
    results = evaluate_turn_checks(
        outcome,
        {
            "must_abstain": ["blue", "favorite color is"],
            "forbid_claims": ["favorite color is"],
        },
    )
    assert any(not r.passed for r in results)


def test_must_recall_requires_evidence_and_answer():
    # Answer-only mention is not enough.
    weak = TurnOutcome(
        answer="You prefer metric units.",
        tool_events=[],
        recalled_memory_ids=[],
        tokens_in_est=10,
        memory_texts=[],
        checks=[],
    )
    assert any(
        not r.passed
        for r in evaluate_turn_checks(weak, {"must_recall": ["metric"]})
    )
    strong = TurnOutcome(
        answer="You prefer metric units.",
        tool_events=[],
        recalled_memory_ids=["abc"],
        tokens_in_est=10,
        memory_texts=["User prefers metric units"],
        checks=[],
    )
    assert all(
        r.passed for r in evaluate_turn_checks(strong, {"must_recall": ["metric"]})
    )


def test_synthesize_does_not_invent_missing_facts():
    messages = [
        {"role": "system", "content": "You are a careful task assistant..."},
        {"role": "user", "content": "What is my favorite color?"},
    ]
    assert "blue" not in synthesize_answer(messages, task="What is my favorite color?").lower()
    assert "don't" in synthesize_answer(messages, task="What is my favorite color?").lower()


def test_synthesize_uses_calculator_observation():
    messages = [
        {"role": "user", "content": "What is 12*8?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "calculator", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "96"},
    ]
    out = synthesize_answer(messages, task="What is 12*8?")
    assert "96" in out


def test_expected_fail_scenario_caught(tmp_path):
    path = _SCENARIOS / "expected_fail_confabulate_01.json"
    sc = load_scenario(path)
    assert sc.expected_fail
    outcome = run_scenario(
        sc, mode="memory", settings=_settings(tmp_path), live=False
    )
    assert outcome is not None
    assert outcome.passed, outcome.detail


def test_cross_session_metric_passes_memory(tmp_path):
    sc = load_scenario(_SCENARIOS / "cross_session_metric_01.json")
    outcome = run_scenario(
        sc, mode="memory", settings=_settings(tmp_path), live=False
    )
    assert outcome is not None
    assert outcome.passed, json.dumps(
        [c.__dict__ for t in outcome.turn_outcomes for c in t.checks], indent=2
    )


def test_dedup_keeps_single_memory(tmp_path):
    sc = load_scenario(_SCENARIOS / "dedup_metric_01.json")
    outcome = run_scenario(
        sc, mode="memory", settings=_settings(tmp_path), live=False
    )
    assert outcome is not None
    assert outcome.passed, outcome.detail
    last = outcome.turn_outcomes[-1]
    assert len(last.memory_texts) == 1


def test_mode_skip_returns_none(tmp_path):
    sc = load_scenario(_SCENARIOS / "cross_session_metric_01.json")
    assert "stateless" not in sc.modes
    assert (
        run_scenario(sc, mode="stateless", settings=_settings(tmp_path), live=False)
        is None
    )


def test_context_long_history_modes_pass_with_mode_checks(tmp_path):
    sc = load_scenario(_SCENARIOS / "context_long_history_01.json")
    settings = _settings(tmp_path)
    b0 = run_scenario(sc, mode="stateless", settings=settings, live=False)
    b2 = run_scenario(sc, mode="memory", settings=settings, live=False)
    assert b0 is not None and b2 is not None
    assert b0.passed, b0.detail
    assert b2.passed, b2.detail
    assert "don't" in b0.turn_outcomes[-1].answer.lower() or "do not" in b0.turn_outcomes[-1].answer.lower()
    assert "ALPHA-7" in b2.turn_outcomes[-1].answer


def test_context_efficiency_hits_thirty_percent(tmp_path):
    scenarios = [
        load_scenario(_SCENARIOS / "context_token_stress_01.json"),
        load_scenario(_SCENARIOS / "context_long_history_01.json"),
    ]
    settings = _settings(tmp_path)
    outcomes = []
    for sc in scenarios:
        for mode in ("stateless", "full_history", "memory"):
            o = run_scenario(sc, mode=mode, settings=settings, live=False)
            assert o is not None and o.passed, (sc.id, mode, o.detail if o else None)
            outcomes.append(o)
    metrics = compute_metrics(outcomes)
    eff = metrics["memory"].context_efficiency_vs_b1
    assert eff is not None and eff >= 0.30, f"efficiency={eff}"


def test_report_suite_loads_manifest():
    scenarios = load_suite(_SCENARIOS, suite="report")
    assert 5 <= len(scenarios) <= 12
    ids = {s.id for s in scenarios}
    assert "single_session_calc_01" in ids
    assert "cross_session_metric_01" in ids


def test_negative_controls_are_caught(tmp_path):
    settings = _settings(tmp_path)
    for name in (
        "offline_fail_missing_calc_01.json",
        "offline_fail_wrong_search_01.json",
        "offline_fail_skip_remember_01.json",
    ):
        sc = load_scenario(_SCENARIOS / name)
        assert sc.expected_fail
        outcome = run_scenario(sc, mode="memory", settings=settings, live=False)
        assert outcome is not None and outcome.passed, (name, outcome.detail if outcome else None)


def test_metrics_table_includes_three_modes(tmp_path):
    smoke = load_suite(_SCENARIOS, suite="smoke")
    settings = _settings(tmp_path)
    outcomes = []
    for sc in smoke:
        for mode in ("stateless", "full_history", "memory"):
            o = run_scenario(sc, mode=mode, settings=settings, live=False)
            if o is not None:
                outcomes.append(o)
    metrics = compute_metrics(outcomes)
    md = metrics_markdown(metrics)
    assert "memory" in md
    assert "Shared-core" in md or "Offline harness" in md or "harness" in md.lower()
    ef = [o for o in outcomes if o.expected_fail]
    assert ef and all(o.passed for o in ef)
    # Equal N on shared-core triple modes when smoke has shared scenarios
    shared_ns = {
        m: metrics[m].n_scenarios
        for m in ("stateless", "full_history", "memory")
        if m in metrics
    }
    assert len(set(shared_ns.values())) == 1, shared_ns
