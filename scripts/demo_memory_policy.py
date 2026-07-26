#!/usr/bin/env python3
"""Phase 5 demo: reflection abstains on math; stores prefs; dedups repeats."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from assistant.agent import run_agent
from assistant.config import get_settings
from assistant.memory import LongTermMemory, Scratchpad, ShortTermMemory


def _fresh_stm(settings) -> ShortTermMemory:
    return ShortTermMemory(
        max_context_tokens=settings.max_context_tokens,
        keep_last_n_turns=settings.keep_last_n_turns,
        summary_trigger_tokens=settings.summary_trigger_tokens,
        recall_k=settings.recall_k,
        mode="bounded",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 5 memory-policy demo")
    parser.add_argument("--user-id", default="phase5-user")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    try:
        settings = get_settings()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    memory_dir = Path(settings.memory_dir) / "phase5_demo"
    if args.fresh and memory_dir.exists():
        shutil.rmtree(memory_dir)

    ltm = LongTermMemory(memory_dir, dedup_distance=settings.dedup_distance)
    print(f"memory_dir={memory_dir}")

    # 1) Ephemeral arithmetic — expect 0 memories from reflection.
    r1 = run_agent(
        "Use the calculator: what is 24 * 7?",
        settings=settings,
        long_term=ltm,
        user_id=args.user_id,
        session_id="math",
        scratchpad=Scratchpad(),
        short_term=_fresh_stm(settings),
        max_steps=6,
    )
    print("\n=== math task ===")
    print(r1.answer)
    print(f"reflection_writes: {[w.action for w in r1.reflection_writes]}")
    print(f"count: {ltm.count_for_user(args.user_id)}")
    if ltm.count_for_user(args.user_id) != 0:
        print(
            "error: math task must leave long-term memory empty "
            f"(tool remember calls={[e.result for e in r1.tool_events if e.tool == 'remember']}; "
            f"reflection={[w.format() for w in r1.reflection_writes]})",
            file=sys.stderr,
        )
        return 2

    before = ltm.count_for_user(args.user_id)

    # 2) Preference — expect >=1 semantic/durable memory.
    r2 = run_agent(
        "Please remember that I prefer metric units for all measurements.",
        settings=settings,
        long_term=ltm,
        user_id=args.user_id,
        session_id="pref1",
        scratchpad=Scratchpad(),
        short_term=_fresh_stm(settings),
        max_steps=6,
    )
    print("\n=== preference ===")
    print(r2.answer)
    print(f"tool remember: {[e.result[:120] for e in r2.tool_events if e.tool == 'remember']}")
    print(f"reflection_writes: {[w.format() for w in r2.reflection_writes]}")
    after_pref = ltm.count_for_user(args.user_id)
    print(f"count: {after_pref}")
    if after_pref < before + 1:
        print("error: expected at least one new memory for preference", file=sys.stderr)
        return 3

    # 3) Repeat preference — expect no net new memory (dedup).
    r3 = run_agent(
        "Remember again: I prefer metric units for all measurements.",
        settings=settings,
        long_term=ltm,
        user_id=args.user_id,
        session_id="pref2",
        scratchpad=Scratchpad(),
        short_term=_fresh_stm(settings),
        max_steps=6,
    )
    print("\n=== repeat preference ===")
    print(r3.answer)
    print(f"tool remember: {[e.result[:120] for e in r3.tool_events if e.tool == 'remember']}")
    print(f"reflection_writes: {[w.format() for w in r3.reflection_writes]}")
    after_dup = ltm.count_for_user(args.user_id)
    print(f"count: {after_dup}")
    if after_dup > after_pref:
        print("error: duplicate preference created extra memories", file=sys.stderr)
        return 4

    print("\n=== stored memories ===")
    for entry in ltm.list_for_user(args.user_id):
        print(entry.format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
