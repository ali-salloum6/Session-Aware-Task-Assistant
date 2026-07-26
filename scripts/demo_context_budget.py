#!/usr/bin/env python3
"""Phase 3 demo: padded multi-turn session stays bounded; early fact still recoverable."""

from __future__ import annotations

import argparse
import sys

from assistant.agent import run_agent
from assistant.config import get_settings, make_short_term
from assistant.memory import ShortTermMemory


def _extractive_summarize(prompt: str) -> str:
    lines = [
        line.strip()
        for line in prompt.splitlines()
        if line.startswith("user:") or line.startswith("assistant:")
    ]
    return "Running summary: " + " | ".join(lines)[:800]


def build_padded_session(*, live_summary: bool) -> ShortTermMemory:
    settings = get_settings()
    stm = make_short_term(settings)
    # Tighten budgets so padding triggers compression in the demo.
    stm.keep_last_n_turns = 4
    stm.summary_trigger_tokens = 400
    stm.max_context_tokens = 900
    stm.recall_k = 3

    stm.add("user", "Please remember: the warehouse codeword is ORBIT-9.")
    stm.add("assistant", "Understood — warehouse codeword is ORBIT-9.")

    for i in range(30):
        stm.add(
            "user",
            f"Filler topic {i}: tell me something generic about robots and warehouses. "
            + ("padding " * 25),
        )
        stm.add(
            "assistant",
            f"Filler reply {i}: robots move boxes efficiently in warehouses. "
            + ("padding " * 25),
        )

    if live_summary:
        from assistant.llm import complete

        def summarize(prompt: str) -> str:
            return complete(
                [
                    {
                        "role": "system",
                        "content": "Compress conversation history; keep concrete facts.",
                    },
                    {"role": "user", "content": prompt},
                ],
                settings=settings,
            )

        while stm.needs_compression():
            stm.compress_if_needed(summarize)
    else:
        while stm.needs_compression():
            stm.compress_if_needed(_extractive_summarize)
    return stm


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 context-budget demo")
    parser.add_argument(
        "--live-summary",
        action="store_true",
        help="Use the LLM to write the rolling summary (costs tokens)",
    )
    parser.add_argument(
        "--live-answer",
        action="store_true",
        help="Ask the live agent the recall question (costs tokens)",
    )
    args = parser.parse_args()

    try:
        get_settings()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    stm = build_padded_session(live_summary=args.live_summary)
    full = ShortTermMemory(mode="full", turns=list(stm.turns))
    query = "What is the warehouse codeword?"
    bounded_tokens = stm.estimate_assembled_tokens(query)
    full_tokens = full.estimate_assembled_tokens(query)
    recalled = stm.selective_recall(query)

    print("=== session stats ===")
    print(f"turns: {len(stm.turns)}")
    print(f"verbatim turns: {len(stm.verbatim_turns())}")
    print(f"summary chars: {len(stm.summary)}")
    print(f"bounded assembled tokens (est): {bounded_tokens}")
    print(f"full-history tokens (est): {full_tokens}")
    print(f"reduction: {100 * (1 - bounded_tokens / full_tokens):.1f}%")
    print(f"under max_context_tokens ({stm.max_context_tokens}): {bounded_tokens <= stm.max_context_tokens}")
    print("\n=== selective recall ===")
    for turn in recalled:
        print(f"- turn {turn.turn_id} [{turn.role}]: {turn.content[:120]}")

    if not any("ORBIT-9" in t.content for t in recalled) and "ORBIT-9" not in stm.summary:
        print("warning: ORBIT-9 missing from recall and summary", file=sys.stderr)
        return 2

    if args.live_answer:
        print("\n=== live agent ===")
        result = run_agent(
            query,
            short_term=stm,
            summarize_fn=_extractive_summarize if not args.live_summary else None,
            max_steps=4,
        )
        print(result.answer)
        if result.token_stats:
            print(f"tokens_in_est (step1): {result.token_stats[0].tokens_in_est}")
        if result.trace_path:
            print(f"trace: {result.trace_path}")
        if "ORBIT-9" not in result.answer:
            print("warning: answer missing ORBIT-9", file=sys.stderr)
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
