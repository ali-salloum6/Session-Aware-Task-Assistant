import json
from pathlib import Path
from types import SimpleNamespace

from assistant.agent.common import AgentResult, StepTokenStats, write_trace
from assistant.config import Settings
from assistant.observability import build_trace_payload, format_trace_summary


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        api_key="test-key",
        base_url="http://example.invalid/v1",
        model="fake-model",
        max_steps=4,
        kb_dir=tmp_path / "kb",
        traces_dir=tmp_path / "traces",
        max_context_tokens=2000,
        keep_last_n_turns=4,
        summary_trigger_tokens=1200,
        recall_k=3,
        context_mode="bounded",
        memory_dir=tmp_path / "memory",
        default_user_id="u",
        ltm_recall_k=3,
        dedup_distance=0.18,
        reflect_memory=False,
        agent_mode="memory",
        agent_backend="loop",
        checkpoint_path=tmp_path / "ckpt.sqlite",
        langfuse_public_key="",
        langfuse_secret_key="",
        langfuse_host="https://cloud.langfuse.com",
    )


def test_trace_payload_includes_required_observability_fields(tmp_path):
    settings = _settings(tmp_path)
    (settings.kb_dir).mkdir(parents=True, exist_ok=True)
    result = AgentResult(
        answer="42",
        stopped_reason="final_answer",
        steps=2,
        tool_events=[],
        token_stats=[StepTokenStats(step=1, tokens_in_est=10, latency_ms=1.5)],
        user_id="u",
        session_id="s1",
        recalled_memory_ids=["abc"],
        mode="memory",
        backend="loop",
        latency_ms=12.3,
    )
    # attach empty tool event via SimpleNamespace-compatible ToolEvent through asdict path
    from assistant.agent.common import ToolEvent

    result.tool_events = [
        ToolEvent(step=1, tool="calculator", arguments={"expression": "6*7"}, result="42")
    ]
    path = write_trace(
        settings.traces_dir,
        "What is 6*7?",
        result,
        raw_trace=[{"step": 1}],
        episode_start=1,
        settings=settings,
    )
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in (
        "input",
        "output",
        "tool_events",
        "recalled_memory_ids",
        "token_stats",
        "latency_ms",
    ):
        assert key in payload, key
    assert payload["input"] == "What is 6*7?"
    assert payload["latency_ms"] == 12.3
    assert payload["recalled_memory_ids"] == ["abc"]
    assert payload["tool_events"][0]["result"] == "42"
    assert payload["token_stats"][0]["latency_ms"] == 1.5
    summary = format_trace_summary(payload)
    assert "latency_ms=12.3" in summary
    assert "calculator" in summary
