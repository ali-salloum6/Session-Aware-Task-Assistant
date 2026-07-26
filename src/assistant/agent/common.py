from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal
from uuid import uuid4

from assistant.agent.prompts import SYSTEM_PROMPT
from assistant.config import Settings
from assistant.llm import complete
from assistant.memory import (
    LongTermMemory,
    Scratchpad,
    ShortTermMemory,
    WriteResult,
    apply_reflection,
    reflect_memories,
)
from assistant.memory.reflection import REFLECT_SYSTEM

AgentMode = Literal["stateless", "full_history", "memory"]


@dataclass
class ToolEvent:
    step: int
    tool: str
    arguments: dict[str, Any]
    result: str


@dataclass
class StepTokenStats:
    step: int
    tokens_in_est: int
    latency_ms: float | None = None


@dataclass
class AgentResult:
    answer: str
    stopped_reason: str
    steps: int
    tool_events: list[ToolEvent] = field(default_factory=list)
    scratchpad: Scratchpad | None = None
    short_term: ShortTermMemory | None = None
    token_stats: list[StepTokenStats] = field(default_factory=list)
    compressed: bool = False
    user_id: str = "default"
    session_id: str = ""
    recalled_memory_ids: list[str] = field(default_factory=list)
    reflection_writes: list[WriteResult] = field(default_factory=list)
    trace_path: str | None = None
    mode: str = "memory"
    backend: str = "loop"
    latency_ms: float | None = None
    langfuse_trace_id: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def system_message(scratchpad: Scratchpad) -> dict[str, str]:
    return {
        "role": "system",
        "content": SYSTEM_PROMPT + "\n\n" + scratchpad.format(),
    }


def ltm_context_message(
    long_term: LongTermMemory, user_id: str, query: str, k: int
) -> tuple[dict[str, str] | None, list[str]]:
    entries = long_term.recall(query, user_id=user_id, k=k)
    if not entries:
        return None, []
    body = "Long-term memories recalled for this user:\n" + long_term.format_recall(
        entries
    )
    return {"role": "system", "content": body}, [e.id for e in entries]


def message_to_dict(message: Any) -> dict[str, Any]:
    data: dict[str, Any] = {"role": message.role}
    if message.content:
        data["content"] = message.content
    if getattr(message, "tool_calls", None):
        data["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in message.tool_calls
        ]
    return data


def default_summarize(settings: Settings) -> Callable[[str], str]:
    def _summarize(prompt: str) -> str:
        return complete(
            [
                {
                    "role": "system",
                    "content": "You compress conversation history into a factual running summary.",
                },
                {"role": "user", "content": prompt},
            ],
            settings=settings,
        )

    return _summarize


def default_reflect(settings: Settings) -> Callable[[str], str]:
    def _reflect(user_prompt: str) -> str:
        return complete(
            [
                {"role": "system", "content": REFLECT_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            settings=settings,
        )

    return _reflect


def tool_trace_summary(events: list[ToolEvent]) -> str:
    if not events:
        return "(none)"
    lines: list[str] = []
    for event in events:
        args = json.dumps(event.arguments, ensure_ascii=True)
        lines.append(f"- {event.tool}({args}) -> {event.result[:160]}")
    return "\n".join(lines)


def normalize_mode(mode: str | None, settings: Settings) -> AgentMode:
    raw = (mode or getattr(settings, "agent_mode", None) or "memory").strip().lower()
    if raw in {"stateless", "b0"}:
        return "stateless"
    if raw in {"full_history", "full", "b1"}:
        return "full_history"
    return "memory"


def apply_mode_to_stm(stm: ShortTermMemory, mode: AgentMode) -> ShortTermMemory:
    if mode == "stateless":
        stm.mode = "bounded"
        stm.reset()
    elif mode == "full_history":
        stm.mode = "full"
    else:
        if stm.mode not in {"bounded", "full"}:
            stm.mode = "bounded"
    return stm


def tool_schemas_for_mode(mode: AgentMode) -> list[dict[str, Any]]:
    """Schemas must match build_registry_for_mode (no advertised-but-missing tools)."""
    from assistant.tools import OPENAI_TOOL_SCHEMAS

    if mode == "memory":
        return list(OPENAI_TOOL_SCHEMAS)
    # B0 + B1: no long-term memory tools
    allowed = {"calculator", "search", "update_scratchpad"}
    return [t for t in OPENAI_TOOL_SCHEMAS if t["function"]["name"] in allowed]


def build_registry_for_mode(
    *,
    kb_dir: Path,
    scratchpad: Scratchpad,
    long_term: LongTermMemory | None,
    user_id: str,
    mode: AgentMode,
):
    from assistant.tools import build_tool_registry

    ltm = None if mode == "stateless" else long_term
    # full_history baseline: no LTM tools (append-all context only)
    if mode == "full_history":
        ltm = None
    return build_tool_registry(kb_dir, scratchpad, long_term=ltm, user_id=user_id)


def run_reflection(
    *,
    task: str,
    answer: str,
    tool_events: list[ToolEvent],
    long_term: LongTermMemory,
    user_id: str,
    reflect_fn: Callable[[str], str],
    enabled: bool,
) -> list[WriteResult]:
    if not enabled:
        return []
    proposals = reflect_memories(
        task=task,
        answer=answer,
        tool_events_summary=tool_trace_summary(tool_events),
        reflect_fn=reflect_fn,
    )
    return apply_reflection(long_term, user_id=user_id, proposals=proposals)


def write_trace(
    traces_dir: Path,
    task: str,
    result: AgentResult,
    raw_trace: list[dict[str, Any]],
    *,
    episode_start: int,
    settings: Settings | None = None,
) -> str:
    from assistant.observability import build_trace_payload, emit_langfuse

    traces_dir.mkdir(parents=True, exist_ok=True)
    path = (
        traces_dir
        / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}.json"
    )
    payload = build_trace_payload(
        task, result, raw_trace, episode_start=episode_start
    )
    if settings is not None:
        langfuse_id = emit_langfuse(payload, settings)
        if langfuse_id:
            payload["langfuse_trace_id"] = langfuse_id
            result.langfuse_trace_id = langfuse_id
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)
