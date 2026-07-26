import json
from pathlib import Path
from types import SimpleNamespace

from langgraph.checkpoint.memory import MemorySaver

from assistant.agent.common import normalize_mode
from assistant.agent.graph import GraphRuntime, build_graph, run_agent_graph
from assistant.config import Settings
from assistant.memory import LongTermMemory, Scratchpad, ShortTermMemory
from assistant.tools import build_tool_registry, OPENAI_TOOL_SCHEMAS


def _settings(tmp_path: Path, **overrides) -> Settings:
    kb = tmp_path / "kb"
    kb.mkdir(exist_ok=True)
    (kb / "product_specs.md").write_text(
        "# Specs\n\nModel X battery capacity: **480 Wh**\n",
        encoding="utf-8",
    )
    base = dict(
        api_key="test-key",
        base_url="http://example.invalid/v1",
        model="fake-model",
        max_steps=8,
        kb_dir=kb,
        traces_dir=tmp_path / "traces",
        max_context_tokens=2000,
        keep_last_n_turns=4,
        summary_trigger_tokens=1200,
        recall_k=3,
        context_mode="bounded",
        memory_dir=tmp_path / "memory",
        default_user_id="test-user",
        ltm_recall_k=3,
        dedup_distance=0.18,
        reflect_memory=False,
        agent_mode="memory",
        agent_backend="graph",
        checkpoint_path=tmp_path / "checkpoints" / "test.sqlite",
        langfuse_public_key="",
        langfuse_secret_key="",
        langfuse_host="https://cloud.langfuse.com",
    )
    base.update(overrides)
    return Settings(**base)


class _FakeCompletions:
    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if not self.scripted:
            raise AssertionError("unexpected extra LLM call")
        return self.scripted.pop(0)


class _FakeClient:
    def __init__(self, scripted):
        self.chat = SimpleNamespace(completions=_FakeCompletions(scripted))


def _resp(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _tool_call(name: str, arguments: str, call_id: str = "call_1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def test_normalize_mode_aliases():
    s = _settings(Path("/tmp"))  # path unused beyond construction
    assert normalize_mode("b0", s) == "stateless"
    assert normalize_mode("b1", s) == "full_history"
    assert normalize_mode("b2", s) == "memory"


def test_graph_search_then_calc(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    scripted = [
        _resp(
            SimpleNamespace(
                role="assistant",
                content=None,
                tool_calls=[
                    _tool_call("search", json.dumps({"query": "Model X battery"}), "c1")
                ],
            )
        ),
        _resp(
            SimpleNamespace(
                role="assistant",
                content=None,
                tool_calls=[
                    _tool_call("calculator", json.dumps({"expression": "480*3"}), "c2")
                ],
            )
        ),
        _resp(
            SimpleNamespace(
                role="assistant",
                content="Model X needs 1440 Wh for 3 units.",
                tool_calls=None,
            )
        ),
    ]
    fake = _FakeClient(scripted)
    monkeypatch.setattr("assistant.agent.graph.make_client", lambda _s=None: fake)

    result = run_agent_graph(
        "battery for 3 Model X",
        settings=settings,
        checkpointer=MemorySaver(),
        summarize_fn=lambda p: "summary",
        reflect=False,
    )
    assert result.backend == "graph"
    assert result.stopped_reason == "final_answer"
    assert "1440" in result.answer
    assert [e.tool for e in result.tool_events] == ["search", "calculator"]


def test_mode_schemas_match_registry_names(tmp_path):
    """B0/B1/B2 must not advertise tools the harness cannot dispatch."""
    from assistant.agent.common import build_registry_for_mode, tool_schemas_for_mode

    settings = _settings(tmp_path)
    pad = Scratchpad()
    ltm = LongTermMemory(settings.memory_dir)
    for mode, expect_ltm in (
        ("stateless", False),
        ("full_history", False),
        ("memory", True),
    ):
        schemas = {
            t["function"]["name"] for t in tool_schemas_for_mode(mode)  # type: ignore[arg-type]
        }
        registry = build_registry_for_mode(
            kb_dir=settings.kb_dir,
            scratchpad=pad,
            long_term=ltm,
            user_id="u",
            mode=mode,  # type: ignore[arg-type]
        )
        assert schemas == set(registry), (mode, schemas, set(registry))
        assert ("remember" in schemas) is expect_ltm
        assert ("recall" in schemas) is expect_ltm


def test_interrupt_before_tools_then_resume(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    scripted = [
        _resp(
            SimpleNamespace(
                role="assistant",
                content=None,
                tool_calls=[
                    _tool_call("calculator", json.dumps({"expression": "6*7"}), "c1")
                ],
            )
        ),
        _resp(
            SimpleNamespace(
                role="assistant",
                content="42",
                tool_calls=None,
            )
        ),
    ]
    fake = _FakeClient(scripted)
    monkeypatch.setattr("assistant.agent.graph.make_client", lambda _s=None: fake)

    pad = Scratchpad()
    stm = ShortTermMemory()
    ltm = LongTermMemory(settings.memory_dir)
    registry = build_tool_registry(
        settings.kb_dir, pad, long_term=None, user_id="u"
    )
    runtime = GraphRuntime(
        settings=settings,
        mode="stateless",
        pad=pad,
        stm=stm,
        ltm=ltm,
        user_id="u",
        session_id="thread-resume",
        registry=registry,
        schemas=[
            t
            for t in OPENAI_TOOL_SCHEMAS
            if t["function"]["name"] in {"calculator", "search", "update_scratchpad"}
        ],
        summarize_fn=lambda p: "summary",
        reflect_fn=lambda p: '{"memories":[]}',
        do_reflect=False,
        record_turn=False,
        client=fake,
    )
    saver = MemorySaver()
    app = build_graph(runtime).compile(
        checkpointer=saver, interrupt_before=["tools"]
    )
    config = {"configurable": {"thread_id": "thread-resume"}}
    partial = app.invoke(
        {
            "task": "What is 6*7?",
            "max_steps": 5,
            "step_count": 0,
            "messages": [],
            "tool_events": [],
            "token_stats": [],
            "raw_trace": [],
        },
        config,
    )
    snap = app.get_state(config)
    assert snap.next == ("tools",)
    assert partial.get("pending_route") == "tools"

    final = app.invoke(None, config)
    assert final.get("stopped_reason") == "final_answer"
    assert final.get("answer") == "42"
    assert final.get("tool_events")[0]["result"] == "42"
