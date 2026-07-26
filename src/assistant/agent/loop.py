from __future__ import annotations

import json
import time
from typing import Any, Callable
from uuid import uuid4

from assistant.agent.common import (
    AgentResult,
    StepTokenStats,
    ToolEvent,
    apply_mode_to_stm,
    build_registry_for_mode,
    default_reflect,
    default_summarize,
    ltm_context_message,
    message_to_dict,
    normalize_mode,
    run_reflection,
    system_message,
    tool_schemas_for_mode,
    write_trace,
)
from assistant.config import Settings, get_settings, make_long_term, make_short_term
from assistant.llm import make_client
from assistant.memory import LongTermMemory, Scratchpad, ShortTermMemory, estimate_messages
from assistant.tools import dispatch_tool


def run_agent_loop(
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
) -> AgentResult:
    """Original ReAct loop (fallback backend)."""
    t0 = time.perf_counter()
    s = settings or get_settings()
    agent_mode = normalize_mode(mode, s)
    budget = max_steps if max_steps is not None else s.max_steps
    client = make_client(s)
    pad = scratchpad if scratchpad is not None else Scratchpad()
    stm = apply_mode_to_stm(
        short_term if short_term is not None else make_short_term(s), agent_mode
    )
    ltm = long_term if long_term is not None else make_long_term(s)
    uid = (user_id or s.default_user_id).strip() or s.default_user_id
    sid = session_id or uuid4().hex[:12]
    registry = build_registry_for_mode(
        kb_dir=s.kb_dir,
        scratchpad=pad,
        long_term=ltm,
        user_id=uid,
        mode=agent_mode,
    )
    schemas = tool_schemas_for_mode(agent_mode)
    summarize = summarize_fn or default_summarize(s)
    do_reflect = (
        False
        if agent_mode != "memory"
        else (s.reflect_memory if reflect is None else reflect)
    )
    reflect_call = reflect_fn or default_reflect(s)

    history = [] if agent_mode == "stateless" else stm.build_history_messages(task)
    recalled_ids: list[str] = []
    prefix: list[dict[str, Any]] = [system_message(pad)]
    if agent_mode == "memory":
        ltm_msg, recalled_ids = ltm_context_message(ltm, uid, task, s.ltm_recall_k)
        if ltm_msg is not None:
            prefix.append(ltm_msg)
    prefix.extend(history)

    messages: list[dict[str, Any]] = [*prefix, {"role": "user", "content": task}]
    episode_start = len(prefix)

    tool_events: list[ToolEvent] = []
    token_stats: list[StepTokenStats] = []
    raw_trace: list[dict[str, Any]] = []

    def _finish(answer: str, stopped_reason: str, steps: int) -> AgentResult:
        compressed = False
        if record_turn and agent_mode != "stateless":
            stm.add("user", task)
            stm.add("assistant", answer)
            compressed = stm.compress_if_needed(summarize)

        reflection_writes = (
            run_reflection(
                task=task,
                answer=answer,
                tool_events=tool_events,
                long_term=ltm,
                user_id=uid,
                reflect_fn=reflect_call,
                enabled=do_reflect and stopped_reason == "final_answer",
            )
            if agent_mode == "memory"
            else []
        )

        result = AgentResult(
            answer=answer,
            stopped_reason=stopped_reason,
            steps=steps,
            tool_events=tool_events,
            scratchpad=pad,
            short_term=stm,
            token_stats=token_stats,
            compressed=compressed,
            user_id=uid,
            session_id=sid,
            recalled_memory_ids=recalled_ids,
            reflection_writes=reflection_writes,
            mode=agent_mode,
            backend="loop",
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
        if save_trace:
            result.trace_path = write_trace(
                s.traces_dir,
                task,
                result,
                raw_trace,
                episode_start=episode_start,
                settings=s,
            )
        return result

    for step in range(1, budget + 1):
        messages[0] = system_message(pad)
        tokens_in = estimate_messages(messages)
        step_t0 = time.perf_counter()

        response = client.chat.completions.create(
            model=s.model,
            messages=messages,
            tools=schemas,
            tool_choice="auto",
            temperature=0.0,
        )
        step_ms = round((time.perf_counter() - step_t0) * 1000, 1)
        token_stats.append(
            StepTokenStats(step=step, tokens_in_est=tokens_in, latency_ms=step_ms)
        )
        message = response.choices[0].message
        messages.append(message_to_dict(message))
        raw_trace.append(
            {
                "step": step,
                "tokens_in_est": tokens_in,
                "latency_ms": step_ms,
                "scratchpad": pad.snapshot(),
                "assistant": message_to_dict(message),
            }
        )

        if message.tool_calls:
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments or "{}")
                    if not isinstance(args, dict):
                        args = {}
                        result = f"tool error ({name}): arguments must be a JSON object"
                    else:
                        result = dispatch_tool(registry, name, args)
                except json.JSONDecodeError as exc:
                    args = {"_raw": tool_call.function.arguments}
                    result = f"tool error ({name}): invalid JSON arguments ({exc})"

                tool_events.append(
                    ToolEvent(step=step, tool=name, arguments=args, result=result)
                )
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
                messages.append(tool_msg)
                raw_trace.append(
                    {
                        "step": step,
                        "tool_result": tool_msg,
                        "scratchpad": pad.snapshot(),
                    }
                )
            continue

        answer = (message.content or "").strip() or "(empty model response)"
        return _finish(answer, "final_answer", step)

    return _finish(
        "Stopped: step budget exhausted before a final answer.",
        "max_steps",
        budget,
    )
