#!/usr/bin/env python3
"""Pretty-print a local JSON agent trace (Phase 7 local viewer)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from assistant.observability import format_trace_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="View an agent trace JSON file")
    parser.add_argument("trace", nargs="?", help="Path to traces/*.json")
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the newest file in traces/",
    )
    args = parser.parse_args()

    if args.latest or not args.trace:
        traces = sorted(Path("traces").glob("*.json"))
        if not traces:
            print("error: no traces found in traces/", file=sys.stderr)
            return 1
        path = traces[-1]
    else:
        path = Path(args.trace)

    payload = json.loads(path.read_text(encoding="utf-8"))
    print(f"file: {path}")
    print(format_trace_summary(payload))
    if payload.get("langfuse_trace_id"):
        print(f"langfuse_trace_id: {payload['langfuse_trace_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
