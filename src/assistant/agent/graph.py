from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, TypedDict
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from assistant.agent.common import (
    AgentMode,
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
from assistant.memory import WriteResult
from assistant.memory.long_term import MemoryEntry
from assistant.config import Settings, get_settings, make_long_term, make_short_term
from assistant.llm import make_client
from assistant.memory import LongTermMemory, Scratchpad, ShortTermMemory, estimate_messages
from assistant.tools import dispatch_tool

Route = Literal["tools", "finalize", "agent"]


class GraphState(TypedDict, total=False):
    task: str
    messages: list[dict[str, Any]]
    step_count: int
    max_steps: int
    answer: str
    stopped_reason: str
    tool_events: list[dict[str, Any]]
    token_stats: list[dict[str, Any]]
    recalled_memory_ids: list[str]
    reflection_writes: list[dict[str, Any]]
    episode_start: int
    raw_trace: list[dict[str, Any]]
    pending_route: Route


@dataclass
class GraphRuntime:
    settings: Settings
    mode: AgentMode
    pad: Scratchpad
    stm: ShortTermMemory
    ltm: LongTermMemory
    user_id: str
    session_id: str
    registry: dict
    schemas: list[dict[str, Any]]
    summarize_fn: Callable[[str], str]
    reflect_fn: Callable[[str], str]
    do_reflect: bool
    record_turn: bool
    client: Any


def _open_sqlite_saver(path: Path) -> SqliteSaver:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn)


def build_checkpointer(settings: Settings, *, in_memory: bool = False):
    if in_memory:
        return MemorySaver()
    return _open_sqlite_saver(settings.checkpoint_path)


def build_graph(runtime: GraphRuntime):
    """assemble_context → agent ⇄ tools → finalize → reflect_memory → END"""

    def assemble_context(state: GraphState) -> dict[str, Any]:
        task = state["task"]
        history = (
            []
            if runtime.mode == "stateless"
            else runtime.stm.build_history_messages(task)
        )
        recalled_ids: list[str] = []
        prefix: list[dict[str, Any]] = [system_message(runtime.pad)]
        if runtime.mode == "memory":
            ltm_msg, recalled_ids = ltm_context_message(
                runtime.ltm, runtime.user_id, task, runtime.settings.ltm_recall_k
            )
            if ltm_msg is not None:
                prefix.append(ltm_msg)
        prefix.extend(history)
        messages = [*prefix, {"role": "user", "content": task}]
        return {
            "messages": messages,
            "episode_start": len(prefix),
            "recalled_memory_ids": recalled_ids,
            "step_count": state.get("step_count", 0),
            "tool_events": state.get("tool_events", []),
            "token_stats": state.get("token_stats", []),
            "raw_trace": state.get("raw_trace", []),
            "reflection_writes": [],
            "answer": "",
            "stopped_reason": "",
        }

    def agent_node(state: GraphState) -> dict[str, Any]:
        step = int(state.get("step_count", 0)) + 1
        max_steps = int(state["max_steps"])
        messages = list(state["messages"])
        messages[0] = system_message(runtime.pad)
        tokens_in = estimate_messages(messages)
        token_stats = list(state.get("token_stats", []))
        raw_trace = list(state.get("raw_trace", []))

        if step > max_steps:
            token_stats.append(
                {"step": step, "tokens_in_est": tokens_in, "latency_ms": 0.0}
            )
            return {
                "step_count": step,
                "answer": "Stopped: step budget exhausted before a final answer.",
                "stopped_reason": "max_steps",
                "pending_route": "finalize",
                "token_stats": token_stats,
                "messages": messages,
            }

        step_t0 = time.perf_counter()
        response = runtime.client.chat.completions.create(
            model=runtime.settings.model,
            messages=messages,
            tools=runtime.schemas,
            tool_choice="auto",
            temperature=0.0,
        )
        step_ms = round((time.perf_counter() - step_t0) * 1000, 1)
        token_stats.append(
            {"step": step, "tokens_in_est": tokens_in, "latency_ms": step_ms}
        )
        message = response.choices[0].message
        messages.append(message_to_dict(message))
        raw_trace.append(
            {
                "step": step,
                "tokens_in_est": tokens_in,
                "latency_ms": step_ms,
                "scratchpad": runtime.pad.snapshot(),
                "assistant": message_to_dict(message),
            }
        )

        if message.tool_calls:
            return {
                "messages": messages,
                "step_count": step,
                "token_stats": token_stats,
                "raw_trace": raw_trace,
                "pending_route": "tools",
            }

        answer = (message.content or "").strip() or "(empty model response)"
        return {
            "messages": messages,
            "step_count": step,
            "token_stats": token_stats,
            "raw_trace": raw_trace,
            "answer": answer,
            "stopped_reason": "final_answer",
            "pending_route": "finalize",
        }

    def tools_node(state: GraphState) -> dict[str, Any]:
        messages = list(state["messages"])
        step = int(state.get("step_count", 1))
        tool_events = list(state.get("tool_events", []))
        raw_trace = list(state.get("raw_trace", []))
        last = messages[-1]
        for tc in last.get("tool_calls") or []:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
                if not isinstance(args, dict):
                    args = {}
                    result = f"tool error ({name}): arguments must be a JSON object"
                else:
                    result = dispatch_tool(runtime.registry, name, args)
            except json.JSONDecodeError as exc:
                args = {"_raw": tc["function"].get("arguments")}
                result = f"tool error ({name}): invalid JSON arguments ({exc})"

            tool_events.append(
                {"step": step, "tool": name, "arguments": args, "result": result}
            )
            tool_msg = {
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            }
            messages.append(tool_msg)
            raw_trace.append(
                {
                    "step": step,
                    "tool_result": tool_msg,
                    "scratchpad": runtime.pad.snapshot(),
                }
            )
        return {
            "messages": messages,
            "tool_events": tool_events,
            "raw_trace": raw_trace,
            "pending_route": "agent",
        }

    def finalize_node(state: GraphState) -> dict[str, Any]:
        answer = state.get("answer") or ""
        stopped = state.get("stopped_reason") or "final_answer"
        return {"answer": answer, "stopped_reason": stopped}

    def reflect_memory_node(state: GraphState) -> dict[str, Any]:
        events = [
            ToolEvent(
                step=e["step"],
                tool=e["tool"],
                arguments=e["arguments"],
                result=e["result"],
            )
            for e in state.get("tool_events", [])
        ]
        enabled = (
            runtime.mode == "memory"
            and runtime.do_reflect
            and state.get("stopped_reason") == "final_answer"
        )
        writes = run_reflection(
            task=state["task"],
            answer=state.get("answer") or "",
            tool_events=events,
            long_term=runtime.ltm,
            user_id=runtime.user_id,
            reflect_fn=runtime.reflect_fn,
            enabled=enabled,
        )
        if runtime.record_turn and runtime.mode != "stateless":
            runtime.stm.add("user", state["task"])
            runtime.stm.add("assistant", state.get("answer") or "")
            runtime.stm.compress_if_needed(runtime.summarize_fn)

        return {
            "reflection_writes": [
                {
                    "action": w.action,
                    "reason": w.reason,
                    "entry": {
                        "id": w.entry.id,
                        "text": w.entry.text,
                        "memory_type": w.entry.memory_type,
                        "source": w.entry.source,
                        "timestamp": w.entry.timestamp,
                        "user_id": w.entry.user_id,
                    },
                }
                for w in writes
            ]
        }

    def route_after_agent(state: GraphState) -> str:
        return state.get("pending_route") or "finalize"

    def route_after_tools(state: GraphState) -> str:
        return "agent"

    g = StateGraph(GraphState)
    g.add_node("assemble_context", assemble_context)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_node("finalize", finalize_node)
    g.add_node("reflect_memory", reflect_memory_node)

    g.add_edge(START, "assemble_context")
    g.add_edge("assemble_context", "agent")
    g.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "finalize": "finalize"},
    )
    g.add_conditional_edges("tools", route_after_tools, {"agent": "agent"})
    g.add_edge("finalize", "reflect_memory")
    g.add_edge("reflect_memory", END)
    return g


def run_agent_graph(
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
    checkpointer: Any | None = None,
    resume: bool = False,
    interrupt_before_tools: bool = False,
) -> AgentResult:
    """LangGraph orchestration with optional sqlite checkpointer."""
    t0 = time.perf_counter()
    s = settings or get_settings()
    agent_mode = normalize_mode(mode, s)
    budget = max_steps if max_steps is not None else s.max_steps
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
    do_reflect = (
        False
        if agent_mode != "memory"
        else (s.reflect_memory if reflect is None else reflect)
    )
    runtime = GraphRuntime(
        settings=s,
        mode=agent_mode,
        pad=pad,
        stm=stm,
        ltm=ltm,
        user_id=uid,
        session_id=sid,
        registry=registry,
        schemas=tool_schemas_for_mode(agent_mode),
        summarize_fn=summarize_fn or default_summarize(s),
        reflect_fn=reflect_fn or default_reflect(s),
        do_reflect=do_reflect,
        record_turn=record_turn,
        client=make_client(s),
    )

    saver = checkpointer if checkpointer is not None else build_checkpointer(s)
    builder = build_graph(runtime)
    compile_kwargs: dict[str, Any] = {"checkpointer": saver}
    if interrupt_before_tools:
        compile_kwargs["interrupt_before"] = ["tools"]
    app = builder.compile(**compile_kwargs)

    config = {"configurable": {"thread_id": sid}}
    if resume:
        final_state = app.invoke(None, config)
    else:
        final_state = app.invoke(
            {
                "task": task,
                "max_steps": budget,
                "step_count": 0,
                "messages": [],
                "tool_events": [],
                "token_stats": [],
                "raw_trace": [],
            },
            config,
        )

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    # Interrupted before tools: return partial result for resume tests.
    if interrupt_before_tools and final_state.get("pending_route") == "tools":
        snapshot = app.get_state(config)
        if snapshot.next:
            tool_events = [
                ToolEvent(**e) for e in final_state.get("tool_events", [])
            ]
            return AgentResult(
                answer="",
                stopped_reason="interrupted",
                steps=int(final_state.get("step_count", 0)),
                tool_events=tool_events,
                scratchpad=pad,
                short_term=stm,
                token_stats=[
                    StepTokenStats(**t) for t in final_state.get("token_stats", [])
                ],
                user_id=uid,
                session_id=sid,
                recalled_memory_ids=list(final_state.get("recalled_memory_ids", [])),
                mode=agent_mode,
                backend="graph",
                latency_ms=latency_ms,
            )

    tool_events = [ToolEvent(**e) for e in final_state.get("tool_events", [])]
    token_stats = [StepTokenStats(**t) for t in final_state.get("token_stats", [])]
    reflection_writes: list[WriteResult] = []
    for w in final_state.get("reflection_writes", []):
        entry = MemoryEntry(
            id=w["entry"]["id"],
            text=w["entry"]["text"],
            memory_type=w["entry"]["memory_type"],
            timestamp=w["entry"].get("timestamp", ""),
            user_id=w["entry"].get("user_id", uid),
            source=w["entry"]["source"],
        )
        reflection_writes.append(
            WriteResult(action=w["action"], entry=entry, reason=w["reason"])
        )

    result = AgentResult(
        answer=final_state.get("answer") or "",
        stopped_reason=final_state.get("stopped_reason") or "final_answer",
        steps=int(final_state.get("step_count", 0)),
        tool_events=tool_events,
        scratchpad=pad,
        short_term=stm,
        token_stats=token_stats,
        compressed=False,
        user_id=uid,
        session_id=sid,
        recalled_memory_ids=list(final_state.get("recalled_memory_ids", [])),
        reflection_writes=reflection_writes,
        mode=agent_mode,
        backend="graph",
        latency_ms=latency_ms,
    )
    if save_trace and result.stopped_reason != "interrupted":
        result.trace_path = write_trace(
            s.traces_dir,
            task,
            result,
            list(final_state.get("raw_trace", [])),
            episode_start=int(final_state.get("episode_start", 0)),
            settings=s,
        )
    return result
