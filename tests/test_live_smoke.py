"""Optional live smoke (skipped unless RUN_LIVE=1). Default CI stays offline."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

pytestmark = pytest.mark.live

RUN_LIVE = os.getenv("RUN_LIVE", "").strip().lower() in {"1", "true", "yes"}


@pytest.mark.skipif(not RUN_LIVE, reason="Set RUN_LIVE=1 with LLM_API_KEY for live smoke")
def test_live_calc_search_bad_calc_observation_and_trace_file(tmp_path):
    from assistant.agent import run_agent
    from assistant.config import get_settings
    from assistant.observability import format_trace_summary
    from assistant.tools.calculator import calculator

    settings = replace(
        get_settings(),
        traces_dir=tmp_path / "traces",
        memory_dir=tmp_path / "memory",
        checkpoint_path=tmp_path / "ckpt.sqlite",
        reflect_memory=False,
        agent_backend="loop",
        agent_mode="stateless",
    )

    good = run_agent(
        "Search for Model X battery, then calculate 480*3. Final number only.",
        settings=settings,
        max_steps=8,
        reflect=False,
        mode="stateless",
        backend="loop",
    )
    assert good.stopped_reason == "final_answer"
    assert "1440" in good.answer
    assert good.trace_path and Path(good.trace_path).is_file()
    payload = json.loads(Path(good.trace_path).read_text(encoding="utf-8"))
    assert payload["latency_ms"] is not None
    assert payload["tool_events"]
    assert "calculator" in format_trace_summary(payload)

    bad = calculator("not-a-number")
    assert bad.startswith("calculator error:")
    assert "unsupported or unsafe" in bad.lower() or "invalid syntax" in bad.lower()
