from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TurnSpec:
    user: str
    offline: dict[str, Any] = field(default_factory=dict)
    checks: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionSpec:
    session_id: str
    turns: list[TurnSpec]
    fresh_stm: bool = True
    fresh_scratchpad: bool = True


@dataclass
class Scenario:
    id: str
    category: str
    sessions: list[SessionSpec]
    user_id: str = "eval-user"
    expected_fail: bool = False
    description: str = ""
    # Optional: only run under these modes (default all)
    modes: list[str] = field(default_factory=lambda: ["stateless", "full_history", "memory"])


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class TurnOutcome:
    answer: str
    tool_events: list[dict[str, Any]]
    recalled_memory_ids: list[str]
    tokens_in_est: int
    memory_texts: list[str]
    checks: list[CheckResult]


@dataclass
class ScenarioOutcome:
    scenario_id: str
    category: str
    mode: str
    expected_fail: bool
    turn_outcomes: list[TurnOutcome]
    passed: bool
    tokens_total: int
    detail: str = ""


@dataclass
class ModeMetrics:
    mode: str
    n_scenarios: int
    task_completion: float
    recall_accuracy: float
    faithfulness: float
    mean_tokens: float
    context_efficiency_vs_b1: float | None
    n_expected_fail_caught: int
    n_expected_fail: int
