#!/usr/bin/env python3
"""Run Phase 8 eval: offline harness and/or live system metrics for the report."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from assistant.config import Settings, get_settings
from assistant.eval import (
    compute_metrics,
    load_suite,
    metrics_markdown,
    metrics_to_dict,
    run_scenario,
)
from assistant.eval.runner import outcomes_to_jsonable

_REPO = Path(__file__).resolve().parents[1]
_SCENARIOS = _REPO / "eval" / "scenarios"
_RESULTS = _REPO / "eval" / "results"
_MODES = ("stateless", "full_history", "memory")


def _offline_settings(tmp_root: Path) -> Settings:
    return Settings(
        api_key="eval-offline",
        base_url="http://example.invalid/v1",
        model="fake-model",
        max_steps=8,
        kb_dir=_REPO / "kb",
        traces_dir=tmp_root / "traces",
        max_context_tokens=700,
        keep_last_n_turns=1,
        summary_trigger_tokens=120,
        recall_k=1,
        context_mode="bounded",
        memory_dir=tmp_root / "memory",
        default_user_id="eval-user",
        ltm_recall_k=3,
        dedup_distance=0.18,
        reflect_memory=False,
        agent_mode="memory",
        agent_backend="loop",
        checkpoint_path=tmp_root / "checkpoints" / "eval.sqlite",
        langfuse_public_key="",
        langfuse_secret_key="",
        langfuse_host="https://cloud.langfuse.com",
    )


def _run_batch(
    *,
    scenarios,
    modes: list[str],
    settings: Settings,
    live: bool,
) -> list:
    outcomes = []
    for sc in scenarios:
        run_modes = list(sc.modes)
        if modes:
            run_modes = [m for m in modes if m in sc.modes]
        for mode in run_modes:
            outcome = run_scenario(
                sc, mode=mode, settings=settings, live=live, save_trace=False
            )
            if outcome is None:
                continue
            status = "PASS" if outcome.passed else "FAIL"
            print(
                f"[{status}] {outcome.mode:13} {outcome.scenario_id} "
                f"({outcome.category}) {outcome.detail}"
            )
            outcomes.append(outcome)
    return outcomes


def _write_results(
    *,
    out_dir: Path,
    kind: str,
    suite: str,
    live: bool,
    outcomes,
    extra_md: str = "",
) -> tuple[Path, Path]:
    metrics = compute_metrics(
        outcomes, scope="all" if live else "shared_core"
    )
    table = metrics_markdown(metrics, kind="live" if live else "harness")
    print("\n" + table)

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "kind": kind,
        "suite": suite,
        "live": live,
        "timestamp": stamp,
        "metrics": metrics_to_dict(metrics),
        "outcomes": outcomes_to_jsonable(outcomes),
    }
    stem = f"{kind}_{suite}_{stamp}"
    out_json = out_dir / f"{stem}.json"
    out_md = out_dir / f"{stem}.md"
    latest_json = out_dir / f"latest_{kind}.json"
    latest_md = out_dir / f"latest_{kind}.md"

    text = json.dumps(payload, indent=2)
    out_json.write_text(text + "\n", encoding="utf-8")
    latest_json.write_text(text + "\n", encoding="utf-8")
    title = "Live system metrics" if live else "Offline harness / regression"
    md = f"# {title} ({suite})\n\n{table}\n{extra_md}"
    out_md.write_text(md, encoding="utf-8")
    latest_md.write_text(md, encoding="utf-8")
    print(f"wrote {out_json}")
    print(f"wrote {latest_md}")
    return latest_md, latest_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Run memory benchmark suite")
    parser.add_argument(
        "--suite",
        choices=("full", "smoke", "report"),
        default="full",
        help="full/smoke = offline harness; report = frozen live subset",
    )
    parser.add_argument(
        "--modes",
        default="",
        help="Optional comma-separated mode filter (default: each scenario's modes)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Call the real LLM (requires LLM_API_KEY)",
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Run offline full harness + live report suite; write REPORT.md",
    )
    parser.add_argument("--scenarios-dir", type=Path, default=_SCENARIOS)
    parser.add_argument("--out-dir", type=Path, default=_RESULTS)
    args = parser.parse_args()

    mode_filter = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in mode_filter:
        if m not in _MODES:
            print(f"error: unknown mode {m!r}", file=sys.stderr)
            return 2

    if args.both:
        # 1) Offline harness
        print("=== OFFLINE HARNESS (full) ===")
        harness_sc = load_suite(args.scenarios_dir, suite="full")
        harness_out = _run_batch(
            scenarios=harness_sc,
            modes=mode_filter,
            settings=_offline_settings(_REPO / "data" / "memory" / "eval_runs"),
            live=False,
        )
        h_md, h_json = _write_results(
            out_dir=args.out_dir,
            kind="harness",
            suite="full",
            live=False,
            outcomes=harness_out,
            extra_md=(
                "\n## How to read this\n\n"
                "Tool calls are scripted; finals are observation-only. "
                "Negative controls must fail. Do **not** cite these % as model quality.\n"
            ),
        )

        # 2) Live system
        print("\n=== LIVE SYSTEM (report subset) ===")
        try:
            base = get_settings()
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        live_settings = replace(
            base,
            memory_dir=_REPO / "data" / "memory" / "eval_live",
            reflect_memory=False,
            agent_backend="loop",
            max_context_tokens=700,
            keep_last_n_turns=1,
            summary_trigger_tokens=120,
            recall_k=1,
        )
        live_sc = load_suite(args.scenarios_dir, suite="report")
        live_out = _run_batch(
            scenarios=live_sc,
            modes=mode_filter,
            settings=live_settings,
            live=True,
        )
        l_md, l_json = _write_results(
            out_dir=args.out_dir,
            kind="live",
            suite="report",
            live=True,
            outcomes=live_out,
            extra_md=(
                "\n## How to read this\n\n"
                "Real OpenRouter model. Cite **these** numbers for task completion, "
                "recall, and faithfulness in the report. "
                "Context efficiency ≥30% is measured on the offline harness "
                "long-history scripts (structural STM vs append-all).\n"
            ),
        )

        report = args.out_dir / "REPORT.md"
        report.write_text(
            "# Evaluation report data\n\n"
            "Generated by `python scripts/run_eval.py --both`.\n\n"
            "## 1. Offline harness (regression / checker)\n\n"
            + h_md.read_text(encoding="utf-8")
            + "\n\n## 2. Live system metrics (report)\n\n"
            + l_md.read_text(encoding="utf-8")
            + f"\n\nArtifacts: `{h_json.name}`, `{l_json.name}`.\n",
            encoding="utf-8",
        )
        # Convenience pointers
        (args.out_dir / "latest.md").write_text(
            report.read_text(encoding="utf-8"), encoding="utf-8"
        )
        print(f"\nwrote {report}")

        live_fails = [o for o in live_out if not o.passed]
        if live_fails:
            print(f"\n{len(live_fails)} live scenario×mode failure(s)", file=sys.stderr)
            return 1
        return 0

    # Single-suite path
    live = args.live or args.suite == "report"
    if args.suite == "report" and not args.live:
        # report suite implies live
        live = True

    scenarios = load_suite(args.scenarios_dir, suite=args.suite)
    if not scenarios:
        print("error: no scenarios found", file=sys.stderr)
        return 2

    if live:
        try:
            base = get_settings()
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        settings = replace(
            base,
            memory_dir=_REPO / "data" / "memory" / "eval_live",
            reflect_memory=False,
            agent_backend="loop",
            max_context_tokens=700,
            keep_last_n_turns=1,
            summary_trigger_tokens=120,
            recall_k=1,
        )
        kind = "live"
    else:
        settings = _offline_settings(_REPO / "data" / "memory" / "eval_runs")
        kind = "harness"

    outcomes = _run_batch(
        scenarios=scenarios,
        modes=mode_filter,
        settings=settings,
        live=live,
    )
    _write_results(
        out_dir=args.out_dir,
        kind=kind,
        suite=args.suite,
        live=live,
        outcomes=outcomes,
    )
    # Keep latest.md as alias for whatever was just run
    latest_kind = args.out_dir / f"latest_{kind}.md"
    if latest_kind.exists():
        (args.out_dir / "latest.md").write_text(
            latest_kind.read_text(encoding="utf-8"), encoding="utf-8"
        )

    hard_fails = [o for o in outcomes if not o.passed]
    if hard_fails:
        print(f"\n{len(hard_fails)} scenario×mode failure(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
