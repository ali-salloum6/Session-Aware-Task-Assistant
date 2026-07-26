from __future__ import annotations

import json
from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
from uuid import uuid4

from assistant.agent import run_agent
from assistant.config import Settings, make_long_term, make_short_term
from assistant.eval.checks import evaluate_turn_checks
from assistant.eval.synthesize import synthesize_answer
from assistant.eval.types import Scenario, ScenarioOutcome, TurnOutcome
from assistant.memory import Scratchpad


class _FakeCompletions:
    """Scripted tool calls, then an observation-only final (unless cheat_final is set)."""

    def __init__(
        self,
        tool_scripted: list[Any],
        *,
        task: str,
        cheat_final: str | None = None,
    ):
        self.tool_scripted = list(tool_scripted)
        self.task = task
        self.cheat_final = cheat_final

    def create(self, **kwargs):
        if self.tool_scripted:
            return self.tool_scripted.pop(0)
        if self.cheat_final is not None:
            return _resp(content=self.cheat_final)
        answer = synthesize_answer(kwargs.get("messages") or [], task=self.task)
        return _resp(content=answer)


class _FakeClient:
    def __init__(self, completions: _FakeCompletions):
        self.chat = SimpleNamespace(completions=completions)


def _resp(*, content: str | None = None, tool_calls: list | None = None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls,
                )
            )
        ]
    )


def _tool_call(name: str, arguments: dict, call_id: str):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def resolve_by_mode(payload: dict[str, Any], mode: str) -> dict[str, Any]:
    """Merge optional by_mode overrides into a base dict (offline scripts or checks)."""
    base = {k: v for k, v in (payload or {}).items() if k != "by_mode"}
    overrides = (payload or {}).get("by_mode") or {}
    if mode in overrides:
        base = {**base, **overrides[mode]}
    return base


def resolve_offline(offline: dict[str, Any], mode: str) -> dict[str, Any]:
    return resolve_by_mode(offline, mode)


def offline_tool_completions(offline: dict[str, Any]) -> list[Any]:
    """Only tool-call steps — finals are synthesized from observations."""
    scripted: list[Any] = []
    for i, call in enumerate(offline.get("tool_calls") or []):
        if isinstance(call, (list, tuple)):
            name, args = call[0], call[1]
        else:
            name, args = call["name"], call.get("arguments") or {}
        scripted.append(_resp(tool_calls=[_tool_call(name, args, f"call_{i}")]))
    return scripted


def eval_summarize(prompt: str) -> str:
    """Extractive summary that keeps concrete facts; drops repetitive padding."""
    lines = [
        line.strip()
        for line in prompt.splitlines()
        if line.startswith("user:") or line.startswith("assistant:")
    ]
    kept: list[str] = []
    for line in lines:
        low = line.lower()
        if "warehouse robot telemetry" in low and "secret" not in low and "alpha" not in low:
            continue
        if "additional context block" in low and "secret" not in low and "alpha" not in low:
            # Keep a short stub so the summary isn't empty, but don't retain padding bulk.
            kept.append(line[:80] + "…")
            continue
        kept.append(line[:240])
    text = " | ".join(kept)
    return text[:900] if text else "summary: (no durable facts)"


def run_scenario(
    scenario: Scenario,
    *,
    mode: str,
    settings: Settings,
    live: bool = False,
    save_trace: bool = False,
) -> ScenarioOutcome | None:
    """Run one scenario under one baseline mode. Returns None if mode is N/A."""
    if mode not in scenario.modes:
        return None

    mem_dir = settings.memory_dir / f"eval_{scenario.id}_{mode}_{uuid4().hex[:8]}"
    run_settings = replace(
        settings,
        memory_dir=mem_dir,
        agent_mode=mode,
        agent_backend="loop",
        reflect_memory=False,
        traces_dir=settings.traces_dir / "eval",
    )
    ltm = make_long_term(run_settings)
    stm = make_short_term(run_settings)
    pad = Scratchpad()

    turn_outcomes: list[TurnOutcome] = []
    all_ok = True

    for sess in scenario.sessions:
        if sess.fresh_stm:
            stm = make_short_term(run_settings)
        if sess.fresh_scratchpad:
            pad = Scratchpad()

        for turn in sess.turns:
            if live:
                result = run_agent(
                    turn.user,
                    settings=run_settings,
                    mode=mode,
                    backend="loop",
                    user_id=scenario.user_id,
                    session_id=sess.session_id,
                    long_term=ltm,
                    short_term=stm,
                    scratchpad=pad,
                    reflect=False,
                    save_trace=save_trace,
                    max_steps=run_settings.max_steps,
                )
            else:
                script = resolve_offline(turn.offline, mode)
                # Cheat finals only for intentional expected-fail scripts.
                cheat = None
                if scenario.expected_fail and script.get("final"):
                    cheat = str(script["final"])
                elif script.get("cheat_final"):
                    cheat = str(script["cheat_final"])
                completions = _FakeCompletions(
                    offline_tool_completions(script),
                    task=turn.user,
                    cheat_final=cheat,
                )
                client = _FakeClient(completions)
                with patch(
                    "assistant.agent.loop.make_client",
                    lambda _settings=None, _c=client: _c,
                ):
                    result = run_agent(
                        turn.user,
                        settings=run_settings,
                        mode=mode,
                        backend="loop",
                        user_id=scenario.user_id,
                        session_id=sess.session_id,
                        long_term=ltm,
                        short_term=stm,
                        scratchpad=pad,
                        reflect=False,
                        save_trace=save_trace,
                        max_steps=run_settings.max_steps,
                        summarize_fn=eval_summarize,
                    )

            tokens = sum(ts.tokens_in_est for ts in result.token_stats)
            memory_texts = [e.text for e in ltm.list_for_user(scenario.user_id)]
            outcome = TurnOutcome(
                answer=result.answer,
                tool_events=[
                    {
                        "step": e.step,
                        "tool": e.tool,
                        "arguments": e.arguments,
                        "result": e.result,
                    }
                    for e in result.tool_events
                ],
                recalled_memory_ids=list(result.recalled_memory_ids),
                tokens_in_est=tokens,
                memory_texts=memory_texts,
                checks=[],
            )
            checks = resolve_by_mode(turn.checks, mode)
            outcome.checks = evaluate_turn_checks(outcome, checks)
            if outcome.checks and not all(c.passed for c in outcome.checks):
                all_ok = False
            turn_outcomes.append(outcome)

    if scenario.expected_fail:
        passed = not all_ok
        detail = (
            "expected failure detected" if passed else "expected failure did not fail"
        )
    else:
        passed = all_ok
        detail = "ok" if passed else "one or more checks failed"

    return ScenarioOutcome(
        scenario_id=scenario.id,
        category=scenario.category,
        mode=mode,
        expected_fail=scenario.expected_fail,
        turn_outcomes=turn_outcomes,
        passed=passed,
        tokens_total=sum(t.tokens_in_est for t in turn_outcomes),
        detail=detail,
    )


def outcomes_to_jsonable(outcomes: list[ScenarioOutcome]) -> list[dict]:
    return [
        {
            "scenario_id": o.scenario_id,
            "category": o.category,
            "mode": o.mode,
            "expected_fail": o.expected_fail,
            "passed": o.passed,
            "tokens_total": o.tokens_total,
            "detail": o.detail,
            "turns": [
                {
                    "answer": t.answer,
                    "tokens_in_est": t.tokens_in_est,
                    "recalled_memory_ids": t.recalled_memory_ids,
                    "memory_texts": t.memory_texts,
                    "tool_events": t.tool_events,
                    "checks": [asdict(c) for c in t.checks],
                }
                for t in o.turn_outcomes
            ],
        }
        for o in outcomes
    ]
