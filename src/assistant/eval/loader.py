from __future__ import annotations

import json
from pathlib import Path

from assistant.eval.types import Scenario, SessionSpec, TurnSpec


def load_scenario(path: Path) -> Scenario:
    data = json.loads(path.read_text(encoding="utf-8"))
    sessions: list[SessionSpec] = []
    for i, sess in enumerate(data["sessions"]):
        turns = [
            TurnSpec(
                user=t["user"],
                offline=dict(t.get("offline") or {}),
                checks=dict(t.get("checks") or {}),
            )
            for t in sess["turns"]
        ]
        sessions.append(
            SessionSpec(
                session_id=sess.get("session_id", f"s{i+1}"),
                turns=turns,
                fresh_stm=bool(sess.get("fresh_stm", True)),
                fresh_scratchpad=bool(sess.get("fresh_scratchpad", True)),
            )
        )
    return Scenario(
        id=data["id"],
        category=data["category"],
        sessions=sessions,
        user_id=data.get("user_id", "eval-user"),
        expected_fail=bool(data.get("expected_fail", False)),
        description=data.get("description", ""),
        modes=list(data.get("modes") or ["stateless", "full_history", "memory"]),
    )


def load_suite(
    scenarios_dir: Path,
    *,
    suite: str = "full",
    report_manifest: Path | None = None,
) -> list[Scenario]:
    by_id: dict[str, Scenario] = {}
    for path in sorted(scenarios_dir.glob("*.json")):
        sc = load_scenario(path)
        by_id[sc.id] = sc
    scenarios = list(by_id.values())

    if suite == "full":
        return scenarios
    if suite == "smoke":
        by_cat: dict[str, Scenario] = {}
        fails: list[Scenario] = []
        for sc in scenarios:
            if sc.expected_fail:
                fails.append(sc)
                continue
            by_cat.setdefault(sc.category, sc)
        return list(by_cat.values()) + fails
    if suite == "report":
        manifest_path = report_manifest or (
            scenarios_dir.parent / "suites" / "report_live.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        out: list[Scenario] = []
        for entry in manifest["scenarios"]:
            sc = by_id.get(entry["id"])
            if sc is None:
                raise FileNotFoundError(f"report suite missing scenario {entry['id']}")
            modes = list(entry.get("modes") or sc.modes)
            out.append(
                Scenario(
                    id=sc.id,
                    category=sc.category,
                    sessions=sc.sessions,
                    user_id=sc.user_id,
                    expected_fail=sc.expected_fail,
                    description=sc.description,
                    modes=modes,
                )
            )
        return out
    raise ValueError(f"unknown suite {suite!r}; use full|smoke|report")
