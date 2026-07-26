from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

from assistant.memory.long_term import VALID_TYPES, LongTermMemory, WriteResult

ReflectFn = Callable[[str], str]

REFLECT_SYSTEM = """\
You extract durable long-term memories from a finished task.
Return ONLY valid JSON of the form:
{"memories":[{"text":"...","memory_type":"semantic|episodic|procedural"}]}

Rules:
- Include 0–3 memories. Prefer preferences, constraints, and lasting decisions.
- Skip ephemeral arithmetic, one-off tool outputs, and scratchpad noise.
- If nothing durable was learned, return {"memories":[]}.
- Do not invent facts the user did not state or confirm.
"""


@dataclass(frozen=True)
class ProposedMemory:
    text: str
    memory_type: str


def parse_reflection_json(raw: str) -> list[ProposedMemory]:
    """Parse model JSON; abstain (empty list) on failure or empty memories."""
    text = raw.strip()
    if not text:
        return []
    # Allow fenced code blocks.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    items = data.get("memories") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    out: list[ProposedMemory] = []
    for item in items[:3]:
        if not isinstance(item, dict):
            continue
        mem_text = str(item.get("text") or "").strip()
        mem_type = str(item.get("memory_type") or "semantic").strip()
        if not mem_text:
            continue
        if mem_type not in VALID_TYPES:
            mem_type = "semantic"
        out.append(ProposedMemory(text=mem_text, memory_type=mem_type))
    return out


def reflect_memories(
    *,
    task: str,
    answer: str,
    tool_events_summary: str,
    reflect_fn: ReflectFn,
) -> list[ProposedMemory]:
    user_prompt = (
        f"User task:\n{task}\n\n"
        f"Assistant answer:\n{answer}\n\n"
        f"Tool trace (brief):\n{tool_events_summary or '(none)'}\n"
    )
    raw = reflect_fn(user_prompt)
    return parse_reflection_json(raw)


def apply_reflection(
    long_term: LongTermMemory,
    *,
    user_id: str,
    proposals: list[ProposedMemory],
) -> list[WriteResult]:
    """Write proposed memories with dedup policy. Empty proposals → no writes."""
    results: list[WriteResult] = []
    for prop in proposals:
        results.append(
            long_term.remember_policy(
                prop.text,
                user_id=user_id,
                memory_type=prop.memory_type,
                source="reflection",
            )
        )
    return results


MEMORY_POLICY_BLURB = (
    "Long-term memory is written deliberately: on explicit user request via the "
    "remember tool, or after a task via a single reflection pass that may propose "
    "0–3 durable facts (preferences, constraints, decisions) and must abstain when "
    "nothing lasting was learned. Near-duplicates are skipped by cosine distance "
    "so repeated preferences do not multiply. Ephemeral arithmetic and tool noise "
    "are not stored; empty recall must be answered as unknown, never confabulated."
)
