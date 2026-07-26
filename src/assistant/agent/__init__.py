from __future__ import annotations

from typing import Any, Callable

from assistant.agent.common import AgentResult, StepTokenStats, ToolEvent
from assistant.agent.graph import build_checkpointer, build_graph, run_agent_graph
from assistant.agent.loop import run_agent_loop
from assistant.config import Settings, get_settings
from assistant.memory import LongTermMemory, Scratchpad, ShortTermMemory, WriteResult


def run_agent(
    task: str,
    *,
    settings: Settings | None = None,
    max_steps: int | None = None,
    save_trace: bool = True,
    scratchpad: Scratchpad | None = None,
    short_term: ShortTermMemory | None = None,
    long_term: LongTermMemory | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    summarize_fn: Callable[[str], str] | None = None,
    reflect_fn: Callable[[str], str] | None = None,
    record_turn: bool = True,
    reflect: bool | None = None,
    mode: str | None = None,
    backend: str | None = None,
    checkpointer: Any | None = None,
    resume: bool = False,
    interrupt_before_tools: bool = False,
) -> AgentResult:
    """Run the task agent (LangGraph by default; loop fallback via AGENT_BACKEND=loop)."""
    s = settings or get_settings()
    chosen = (backend or s.agent_backend).strip().lower()
    kwargs = dict(
        task=task,
        settings=s,
        max_steps=max_steps,
        save_trace=save_trace,
        scratchpad=scratchpad,
        short_term=short_term,
        long_term=long_term,
        user_id=user_id,
        session_id=session_id,
        summarize_fn=summarize_fn,
        reflect_fn=reflect_fn,
        record_turn=record_turn,
        reflect=reflect,
        mode=mode,
    )
    if chosen == "loop":
        return run_agent_loop(**kwargs)
    return run_agent_graph(
        **kwargs,
        checkpointer=checkpointer,
        resume=resume,
        interrupt_before_tools=interrupt_before_tools,
    )


__all__ = [
    "AgentResult",
    "ToolEvent",
    "StepTokenStats",
    "WriteResult",
    "run_agent",
    "run_agent_loop",
    "run_agent_graph",
    "build_graph",
    "build_checkpointer",
]
