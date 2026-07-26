#!/usr/bin/env python3
"""Phase 4 demo: remember in session A, recall after reopen in session B."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from assistant.agent import run_agent
from assistant.config import get_settings
from assistant.memory import LongTermMemory, Scratchpad, ShortTermMemory


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4 cross-session memory demo")
    parser.add_argument("--user-id", default="demo-user")
    parser.add_argument(
        "--memory-dir",
        default="",
        help="Override MEMORY_DIR (default: settings / temp demo dir)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete the memory dir before running",
    )
    parser.add_argument("--no-live", action="store_true", help="Skip live LLM calls")
    args = parser.parse_args()

    try:
        settings = get_settings()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    memory_dir = Path(args.memory_dir) if args.memory_dir else settings.memory_dir / "phase4_demo"
    if args.fresh and memory_dir.exists():
        shutil.rmtree(memory_dir)

    print(f"memory_dir={memory_dir}")
    print(f"user_id={args.user_id}")

    # --- Session A: write preference ---
    ltm_a = LongTermMemory(memory_dir)
    if args.no_live:
        entry = ltm_a.remember(
            "User prefers metric units for all measurements.",
            user_id=args.user_id,
            memory_type="semantic",
            source="user",
        )
        print(f"session A wrote: {entry.format()}")
    else:
        result_a = run_agent(
            "Please remember that I prefer metric units for all measurements.",
            settings=settings,
            long_term=ltm_a,
            user_id=args.user_id,
            session_id="session-a",
            scratchpad=Scratchpad(),
            short_term=ShortTermMemory(
                max_context_tokens=settings.max_context_tokens,
                keep_last_n_turns=settings.keep_last_n_turns,
                summary_trigger_tokens=settings.summary_trigger_tokens,
                recall_k=settings.recall_k,
                mode="bounded",
            ),
            max_steps=6,
        )
        print("=== session A ===")
        for event in result_a.tool_events:
            print(f"{event.tool}: {event.result[:180]}")
        print(result_a.answer)

    # --- Simulate process restart: new LTM client, fresh STM/scratchpad ---
    ltm_b = LongTermMemory(memory_dir)
    other = ltm_b.recall("units preference", user_id="other-user", k=3)
    hits = ltm_b.recall("preferred measurement units", user_id=args.user_id, k=3)
    print("\n=== after reopen ===")
    print(f"other-user hits: {len(other)}")
    print("demo-user hits:")
    print(ltm_b.format_recall(hits))
    if not hits or not any("metric" in h.text.lower() for h in hits):
        print("error: preference not recalled for demo user", file=sys.stderr)
        return 2
    if other:
        print("error: other-user should not see demo memories", file=sys.stderr)
        return 3

    if not args.no_live:
        result_b = run_agent(
            "What measurement units do I prefer? Use memory if needed.",
            settings=settings,
            long_term=ltm_b,
            user_id=args.user_id,
            session_id="session-b",
            scratchpad=Scratchpad(),
            short_term=ShortTermMemory(
                max_context_tokens=settings.max_context_tokens,
                keep_last_n_turns=settings.keep_last_n_turns,
                summary_trigger_tokens=settings.summary_trigger_tokens,
                recall_k=settings.recall_k,
                mode="bounded",
            ),
            max_steps=6,
        )
        print("\n=== session B ===")
        print(f"recalled_ids: {result_b.recalled_memory_ids}")
        print(result_b.answer)
        if "metric" not in result_b.answer.lower():
            print("error: session B answer missing metric", file=sys.stderr)
            return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
