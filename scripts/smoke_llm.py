#!/usr/bin/env python3
"""Phase 0 smoke test: one LLM completion."""

from __future__ import annotations

import sys

from assistant.llm import complete


def main() -> int:
    try:
        text = complete(
            [
                {"role": "system", "content": "Reply with one short sentence."},
                {"role": "user", "content": "Say hello."},
            ]
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
