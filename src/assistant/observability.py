from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from assistant.agent.common import AgentResult, utc_now
from assistant.config import Settings


def build_trace_payload(
    task: str,
    result: AgentResult,
    raw_trace: list[dict[str, Any]],
    *,
    episode_start: int,
) -> dict[str, Any]:
    return {
        "created_at": utc_now(),
        "task": task,
        "input": task,
        "user_id": result.user_id,
        "session_id": result.session_id,
        "mode": result.mode,
        "backend": result.backend,
        "stopped_reason": result.stopped_reason,
        "steps": result.steps,
        "answer": result.answer,
        "output": result.answer,
        "compressed": result.compressed,
        "recalled_memory_ids": result.recalled_memory_ids,
        "latency_ms": result.latency_ms,
        "reflection_writes": [
            {
                "action": w.action,
                "reason": w.reason,
                "entry": {
                    "id": w.entry.id,
                    "text": w.entry.text,
                    "memory_type": w.entry.memory_type,
                    "source": w.entry.source,
                },
            }
            for w in result.reflection_writes
        ],
        "episode_start": episode_start,
        "token_stats": [asdict(s) for s in result.token_stats],
        "tool_events": [asdict(e) for e in result.tool_events],
        "scratchpad": result.scratchpad.to_dict() if result.scratchpad else None,
        "short_term": result.short_term.to_dict() if result.short_term else None,
        "raw": raw_trace,
    }


def emit_langfuse(
    payload: dict[str, Any],
    settings: Settings,
) -> str | None:
    """Push a run to Langfuse when keys are configured. Returns trace id or None."""
    if not settings.langfuse_enabled:
        return None
    try:
        from langfuse import Langfuse
    except ImportError:
        return None

    client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
    metadata = {
        "mode": payload.get("mode"),
        "backend": payload.get("backend"),
        "stopped_reason": payload.get("stopped_reason"),
        "steps": payload.get("steps"),
        "latency_ms": payload.get("latency_ms"),
        "recalled_memory_ids": payload.get("recalled_memory_ids"),
        "token_stats": payload.get("token_stats"),
        "user_id": payload.get("user_id"),
        "session_id": payload.get("session_id"),
    }
    root = client.start_observation(
        name="assistant-run",
        as_type="agent",
        input=payload.get("input"),
        metadata=metadata,
    )
    try:
        for event in payload.get("tool_events") or []:
            child = root.start_observation(
                name=f"tool:{event.get('tool')}",
                as_type="tool",
                input=event.get("arguments"),
                output=event.get("result"),
                metadata={"step": event.get("step")},
            )
            child.end()
        root.update(output=payload.get("output"))
        root.set_trace_io(input=payload.get("input"), output=payload.get("output"))
    finally:
        root.end()
        client.flush()

    return getattr(root, "trace_id", None)


def format_trace_summary(payload: dict[str, Any]) -> str:
    lines = [
        f"task: {payload.get('task')}",
        f"session: {payload.get('session_id')} | user: {payload.get('user_id')}",
        f"mode={payload.get('mode')} backend={payload.get('backend')} "
        f"stopped={payload.get('stopped_reason')} steps={payload.get('steps')}",
        f"latency_ms={payload.get('latency_ms')}",
        f"recalled_memory_ids={payload.get('recalled_memory_ids')}",
        "token_stats:",
    ]
    for ts in payload.get("token_stats") or []:
        extra = ""
        if "latency_ms" in ts:
            extra = f" latency_ms={ts['latency_ms']}"
        lines.append(f"  step {ts.get('step')}: tokens_in_est={ts.get('tokens_in_est')}{extra}")
    lines.append("tool_events:")
    for event in payload.get("tool_events") or []:
        lines.append(
            f"  step {event.get('step')}: {event.get('tool')}({json.dumps(event.get('arguments'))}) "
            f"-> {str(event.get('result'))[:160]}"
        )
    lines.append(f"answer: {payload.get('answer')}")
    return "\n".join(lines)
