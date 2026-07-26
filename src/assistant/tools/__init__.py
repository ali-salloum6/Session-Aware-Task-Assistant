from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from assistant.memory.long_term import LongTermMemory, VALID_TYPES
from assistant.memory.scratchpad import Scratchpad
from assistant.tools.calculator import calculator
from assistant.tools.search import ScopedSearch

ToolFn = Callable[..., str]


def build_tool_registry(
    kb_dir: Path,
    scratchpad: Scratchpad,
    *,
    long_term: LongTermMemory | None = None,
    user_id: str = "default",
) -> dict[str, ToolFn]:
    searcher = ScopedSearch(kb_dir)

    def update_scratchpad(
        plan: str | None = None,
        done: str | None = None,
        next: str | None = None,
    ) -> str:
        return scratchpad.update(plan=plan, done=done, next=next)

    registry: dict[str, ToolFn] = {
        "calculator": calculator,
        "search": searcher.search,
        "update_scratchpad": update_scratchpad,
    }

    if long_term is not None:

        def remember(text: str, memory_type: str = "semantic") -> str:
            try:
                result = long_term.remember_policy(
                    text,
                    user_id=user_id,
                    memory_type=memory_type,
                    source="agent",
                )
            except ValueError as exc:
                return f"remember error: {exc}"
            return result.format()

        def recall(query: str, k: int = 3) -> str:
            try:
                top_k = int(k)
            except (TypeError, ValueError):
                top_k = 3
            entries = long_term.recall(query, user_id=user_id, k=top_k)
            return long_term.format_recall(entries)

        registry["remember"] = remember
        registry["recall"] = recall

    return registry


OPENAI_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Evaluate a basic arithmetic expression with + - * / ** // % and "
                "parentheses. Use for any numeric computation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression, e.g. '2*(3+4)' or '480*3'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search the local project knowledge base (markdown docs about Acme "
                "Robotics). Use for company facts, products, and policies. Not the web."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Short keyword query, e.g. 'Model X battery'.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_scratchpad",
            "description": (
                "Update the structured working scratchpad (PLAN / DONE / NEXT). "
                "Call this at the start of a multi-step task and after each meaningful "
                "progress. Omit a field to leave it unchanged; pass an empty string to clear."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "string",
                        "description": "Full plan or revised plan for the task.",
                    },
                    "done": {
                        "type": "string",
                        "description": "What has been completed so far.",
                    },
                    "next": {
                        "type": "string",
                        "description": "Immediate next action.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": (
                "Store a durable fact in long-term memory for this user (preferences, "
                "constraints, decisions). Do not store ephemeral arithmetic or one-off "
                f"tool results. memory_type must be one of: {', '.join(sorted(VALID_TYPES))}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The durable fact to store.",
                    },
                    "memory_type": {
                        "type": "string",
                        "enum": sorted(VALID_TYPES),
                        "description": "episodic | semantic | procedural",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": (
                "Retrieve relevant long-term memories for this user by semantic similarity. "
                "Use when the answer may depend on prior preferences or decisions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look up in long-term memory.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Max memories to return (default 3).",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def dispatch_tool(registry: dict[str, ToolFn], name: str, arguments: dict[str, Any]) -> str:
    fn = registry.get(name)
    if fn is None:
        return f"unknown tool: {name}"
    try:
        if name == "calculator":
            return fn(arguments.get("expression", ""))
        if name == "search":
            return fn(arguments.get("query", ""))
        if name == "update_scratchpad":
            return fn(
                plan=arguments.get("plan"),
                done=arguments.get("done"),
                next=arguments.get("next"),
            )
        if name == "remember":
            return fn(
                arguments.get("text", ""),
                memory_type=arguments.get("memory_type", "semantic"),
            )
        if name == "recall":
            return fn(arguments.get("query", ""), k=arguments.get("k", 3))
        return fn(**arguments)
    except TypeError as exc:
        return f"tool error ({name}): {exc}"
