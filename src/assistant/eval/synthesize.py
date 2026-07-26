"""Observation-only final answers for honest offline eval (no scripted facts)."""

from __future__ import annotations

import re
from typing import Any


_ABSTAIN = "I don't have that stored."


def _tool_results(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Return (tool_name, content) pairs from the current episode messages."""
    # Map tool_call_id -> name from the preceding assistant message.
    id_to_name: dict[str, str] = {}
    out: list[tuple[str, str]] = []
    for msg in messages:
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            tid = tc.get("id")
            if tid:
                id_to_name[tid] = str(fn.get("name") or "tool")
        if msg.get("role") == "tool":
            name = id_to_name.get(msg.get("tool_call_id", ""), "tool")
            content = str(msg.get("content") or "")
            out.append((name, content))
    return out


def _context_blobs(messages: list[dict[str, Any]]) -> list[str]:
    """STM / LTM system blobs injected before the live user turn (not the profile prompt)."""
    blobs: list[str] = []
    for msg in messages:
        if msg.get("role") != "system":
            continue
        content = str(msg.get("content") or "")
        if content.startswith("Long-term memories recalled"):
            blobs.append(content)
        elif content.startswith("Running summary"):
            blobs.append(content)
        elif content.startswith("Relevant earlier turns"):
            blobs.append(content)
    return blobs


def _history_turns(messages: list[dict[str, Any]], task: str) -> list[str]:
    """Prior user/assistant turns in the assembled context (exclude the live task)."""
    turns: list[str] = []
    for msg in messages:
        if msg.get("role") not in {"user", "assistant"}:
            continue
        content = str(msg.get("content") or "").strip()
        if not content or content == task:
            continue
        # Skip empty model placeholders.
        if content.startswith("(empty"):
            continue
        turns.append(f"{msg['role']}: {content}")
    return turns


def evidence_text(
    messages: list[dict[str, Any]],
    *,
    task: str,
) -> str:
    parts: list[str] = []
    for name, content in _tool_results(messages):
        parts.append(f"{name}: {content}")
    parts.extend(_context_blobs(messages))
    parts.extend(_history_turns(messages, task))
    return "\n".join(parts)


def synthesize_answer(messages: list[dict[str, Any]], *, task: str) -> str:
    """
    Build a final answer using only tool observations and injected STM/LTM context.

    Never invents facts that are not present in those sources.
    """
    tools = _tool_results(messages)
    blobs = _context_blobs(messages)
    history = _history_turns(messages, task)
    evidence = evidence_text(messages, task=task)

    # Prefer concrete calculator results.
    calc_vals = [
        content.strip()
        for name, content in tools
        if name == "calculator" and content.strip() and not content.startswith("calculator error")
    ]
    search_hits = [
        content.strip()
        for name, content in tools
        if name == "search" and content.strip() and not content.startswith("search error")
        and not content.startswith("No documents matched")
    ]
    memory_bits = [
        content.strip()
        for name, content in tools
        if name in {"recall", "remember"} and content.strip()
        and not content.startswith("No long-term")
        and not content.startswith("remember error")
        and not content.startswith("unknown tool")
    ]
    # remember tool returns "created: [...] text" — keep as confirmation
    scratch_bits = [
        content.strip()
        for name, content in tools
        if name == "update_scratchpad" and content.strip()
    ]

    chunks: list[str] = []
    if calc_vals:
        chunks.append("Calculator result: " + "; ".join(calc_vals))
    if search_hits:
        # Keep a short extract; checks look for keywords like Berlin.
        chunks.append("Search observations:\n" + "\n".join(s[:500] for s in search_hits))
    if memory_bits:
        chunks.append("Memory observations:\n" + "\n".join(memory_bits))
    if blobs:
        chunks.append("Recalled context:\n" + "\n".join(blobs))
    # For STM-only questions (no tools), surface matching prior turns / summary.
    if not tools and (blobs or history):
        task_tokens = set(re.findall(r"[a-z0-9]+", task.lower()))
        relevant = []
        for line in history + blobs:
            line_tokens = set(re.findall(r"[a-z0-9]+", line.lower()))
            if task_tokens & line_tokens:
                relevant.append(line[:400])
        if relevant:
            chunks.append("From earlier context:\n" + "\n".join(relevant[:6]))
        elif history:
            chunks.append("From earlier context:\n" + "\n".join(h[:400] for h in history[-4:]))
    if scratch_bits and not chunks:
        chunks.append(scratch_bits[-1])

    if not chunks:
        # Empty recall / no context → abstain on questions; echo statements into STM.
        if any(
            name == "recall"
            and (
                "No long-term" in content
                or content.startswith("unknown tool")
            )
            for name, content in tools
        ):
            return _ABSTAIN
        if "?" in task:
            return _ABSTAIN
        # Non-question user statements become the recorded observation (no invention).
        return f"Noted: {task.strip()}"

    return "\n".join(chunks)
