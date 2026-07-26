#!/usr/bin/env python3
"""Run the task agent (Phase 2: scratchpad + tools)."""

from __future__ import annotations

import argparse
import sys

from assistant.agent import run_agent
from assistant.config import get_settings


DEFAULT_TASK = (
    "Multi-step task: (1) search the knowledge base for Model X battery capacity, "
    "(2) search for Acme Robotics headquarters city, "
    "(3) use the calculator to compute battery_wh * 3. "
    "Maintain PLAN/DONE/NEXT with update_scratchpad as you go. "
    "Final answer must include city, per-unit Wh, and total Wh for 3 units."
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Phase 2 task agent")
    parser.add_argument("task", nargs="?", default=DEFAULT_TASK, help="User task")
    parser.add_argument("--max-steps", type=int, default=None, help="Override MAX_STEPS")
    parser.add_argument("--no-trace", action="store_true", help="Do not write a trace file")
    args = parser.parse_args()

    try:
        get_settings()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    result = run_agent(
        args.task,
        max_steps=args.max_steps,
        save_trace=not args.no_trace,
    )

    print("=== tool calls ===")
    if not result.tool_events:
        print("(none)")
    for event in result.tool_events:
        print(f"step {event.step}: {event.tool}({event.arguments}) -> {event.result[:200]}")

    if result.scratchpad is not None:
        print("\n=== scratchpad ===")
        print(result.scratchpad.format())
        if result.scratchpad.history:
            print(f"(updates: {len(result.scratchpad.history)})")

    print("\n=== final answer ===")
    print(result.answer)
    print(f"\nstopped: {result.stopped_reason} | steps: {result.steps}")
    if result.latency_ms is not None:
        print(f"latency_ms: {result.latency_ms}")
    if result.trace_path:
        print(f"trace: {result.trace_path}")
    if result.langfuse_trace_id:
        print(f"langfuse: {result.langfuse_trace_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
