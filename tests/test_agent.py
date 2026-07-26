import json
from pathlib import Path
from types import SimpleNamespace

from assistant.agent import run_agent
from assistant.config import Settings
from assistant.memory import Scratchpad


class _FakeCompletions:
    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.calls = 0
        self.seen_messages = []

    def create(self, **kwargs):
        self.calls += 1
        self.seen_messages.append(kwargs["messages"])
        if not self.scripted:
            raise AssertionError("unexpected extra LLM call")
        return self.scripted.pop(0)


class _FakeChat:
    def __init__(self, scripted):
        self.completions = _FakeCompletions(scripted)


class _FakeClient:
    def __init__(self, scripted):
        self.chat = _FakeChat(scripted)


def _resp(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _tool_call(name: str, arguments: str, call_id: str = "call_1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _settings(tmp_path: Path) -> Settings:
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "product_specs.md").write_text(
        "# Specs\n\nModel X battery capacity: **480 Wh**\n",
        encoding="utf-8",
    )
    return Settings(
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
        agent_backend="loop",
        checkpoint_path=tmp_path / "checkpoints" / "test.sqlite",
        langfuse_public_key="",
        langfuse_secret_key="",
        langfuse_host="https://cloud.langfuse.com",
    )


def test_run_agent_search_then_calc(tmp_path, monkeypatch):
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
    monkeypatch.setattr("assistant.agent.loop.make_client", lambda _settings=None: fake)

    result = run_agent("battery for 3 Model X", settings=settings)
    assert result.stopped_reason == "final_answer"
    assert "1440" in result.answer
    assert [e.tool for e in result.tool_events] == ["search", "calculator"]
    assert "480" in result.tool_events[0].result
    assert result.tool_events[1].result == "1440"
    assert result.trace_path is not None
    payload = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
    assert payload["tool_events"][1]["result"] == "1440"
    assert fake.chat.completions.calls == 3
    assert "SCRATCHPAD" in fake.chat.completions.seen_messages[0][0]["content"]


def test_run_agent_updates_and_injects_scratchpad(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    scripted = [
        _resp(
            SimpleNamespace(
                role="assistant",
                content=None,
                tool_calls=[
                    _tool_call(
                        "update_scratchpad",
                        json.dumps(
                            {
                                "plan": "1) search battery 2) calc *3",
                                "done": "",
                                "next": "search",
                            }
                        ),
                        "c1",
                    )
                ],
            )
        ),
        _resp(
            SimpleNamespace(
                role="assistant",
                content=None,
                tool_calls=[
                    _tool_call("search", json.dumps({"query": "Model X battery"}), "c2")
                ],
            )
        ),
        _resp(
            SimpleNamespace(
                role="assistant",
                content=None,
                tool_calls=[
                    _tool_call(
                        "update_scratchpad",
                        json.dumps(
                            {
                                "done": "battery=480 Wh",
                                "next": "calculator 480*3",
                            }
                        ),
                        "c3",
                    )
                ],
            )
        ),
        _resp(
            SimpleNamespace(
                role="assistant",
                content=None,
                tool_calls=[
                    _tool_call("calculator", json.dumps({"expression": "480*3"}), "c4")
                ],
            )
        ),
        _resp(
            SimpleNamespace(
                role="assistant",
                content="1440 Wh total for 3 Model X units.",
                tool_calls=None,
            )
        ),
    ]
    fake = _FakeClient(scripted)
    monkeypatch.setattr("assistant.agent.loop.make_client", lambda _settings=None: fake)

    result = run_agent("multi-step battery task", settings=settings, max_steps=8)
    assert result.stopped_reason == "final_answer"
    assert result.scratchpad is not None
    assert "search battery" in result.scratchpad.plan
    assert "480" in result.scratchpad.done
    assert "calculator" in result.scratchpad.next
    assert len(result.scratchpad.history) == 2

    # Second LLM call should already see the first scratchpad write.
    second_system = fake.chat.completions.seen_messages[1][0]["content"]
    assert "PLAN: 1) search battery 2) calc *3" in second_system
    third_system = fake.chat.completions.seen_messages[2][0]["content"]
    assert "DONE: battery=480 Wh" in third_system

    payload = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
    assert payload["scratchpad"]["current"]["done"] == "battery=480 Wh"


def test_run_agent_reuses_incoming_scratchpad(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    pad = Scratchpad(plan="carry over", done="prior turn", next="continue")
    scripted = [
        _resp(
            SimpleNamespace(
                role="assistant",
                content="Continuing from scratchpad.",
                tool_calls=None,
            )
        )
    ]
    fake = _FakeClient(scripted)
    monkeypatch.setattr("assistant.agent.loop.make_client", lambda _settings=None: fake)

    result = run_agent("follow-up", settings=settings, scratchpad=pad)
    assert result.scratchpad is pad
    system = fake.chat.completions.seen_messages[0][0]["content"]
    assert "PLAN: carry over" in system
    assert "DONE: prior turn" in system


def test_run_agent_respects_max_steps(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    settings = Settings(
        api_key=settings.api_key,
        base_url=settings.base_url,
        model=settings.model,
        max_steps=2,
        kb_dir=settings.kb_dir,
        traces_dir=settings.traces_dir,
        max_context_tokens=settings.max_context_tokens,
        keep_last_n_turns=settings.keep_last_n_turns,
        summary_trigger_tokens=settings.summary_trigger_tokens,
        recall_k=settings.recall_k,
        context_mode=settings.context_mode,
        memory_dir=settings.memory_dir,
        default_user_id=settings.default_user_id,
        ltm_recall_k=settings.ltm_recall_k,
        dedup_distance=settings.dedup_distance,
        reflect_memory=settings.reflect_memory,
        agent_mode=settings.agent_mode,
        agent_backend=settings.agent_backend,
        checkpoint_path=settings.checkpoint_path,
        langfuse_public_key="",
        langfuse_secret_key="",
        langfuse_host="https://cloud.langfuse.com",
    )
    forever_tool = _resp(
        SimpleNamespace(
            role="assistant",
            content=None,
            tool_calls=[_tool_call("search", json.dumps({"query": "hello"}), "c")],
        )
    )
    fake = _FakeClient([forever_tool, forever_tool])
    monkeypatch.setattr("assistant.agent.loop.make_client", lambda _settings=None: fake)

    result = run_agent("loop", settings=settings, max_steps=2)
    assert result.stopped_reason == "max_steps"
    assert result.steps == 2


def test_run_agent_injects_short_term_history(tmp_path, monkeypatch):
    from assistant.memory import ShortTermMemory

    settings = _settings(tmp_path)
    stm = ShortTermMemory(
        keep_last_n_turns=2,
        summary_trigger_tokens=10_000,
        recall_k=2,
        mode="bounded",
    )
    stm.add("user", "The dock PIN is 2468.")
    stm.add("assistant", "Noted dock PIN 2468.")
    for i in range(4):
        stm.add("user", f"filler {i}")
        stm.add("assistant", f"ok {i}")

    scripted = [
        _resp(
            SimpleNamespace(
                role="assistant",
                content="The dock PIN is 2468.",
                tool_calls=None,
            )
        )
    ]
    fake = _FakeClient(scripted)
    monkeypatch.setattr("assistant.agent.loop.make_client", lambda _settings=None: fake)

    result = run_agent(
        "What is the dock PIN?",
        settings=settings,
        short_term=stm,
        summarize_fn=lambda prompt: "summary",
    )
    assert "2468" in result.answer
    first_msgs = fake.chat.completions.seen_messages[0]
    blob = json.dumps(first_msgs)
    assert "2468" in blob or "PIN" in blob
    assert result.token_stats
    assert result.token_stats[0].tokens_in_est > 0
    assert result.short_term is stm
    assert len(stm.turns) >= 2


def test_run_agent_remember_and_recall_tools(tmp_path, monkeypatch):
    from assistant.memory import LongTermMemory

    settings = _settings(tmp_path)
    ltm = LongTermMemory(settings.memory_dir)
    scripted = [
        _resp(
            SimpleNamespace(
                role="assistant",
                content=None,
                tool_calls=[
                    _tool_call(
                        "remember",
                        json.dumps(
                            {
                                "text": "User prefers metric units.",
                                "memory_type": "semantic",
                            }
                        ),
                        "c1",
                    )
                ],
            )
        ),
        _resp(
            SimpleNamespace(
                role="assistant",
                content="I'll remember that you prefer metric units.",
                tool_calls=None,
            )
        ),
    ]
    fake = _FakeClient(scripted)
    monkeypatch.setattr("assistant.agent.loop.make_client", lambda _settings=None: fake)

    result = run_agent(
        "Please remember I prefer metric units.",
        settings=settings,
        long_term=ltm,
        user_id="alice",
        session_id="s1",
        summarize_fn=lambda prompt: "summary",
    )
    assert result.tool_events[0].tool == "remember"
    assert "metric" in result.tool_events[0].result.lower()
    assert "created" in result.tool_events[0].result.lower()
    hits = ltm.recall("units", user_id="alice", k=3)
    assert any("metric" in h.text.lower() for h in hits)

    # New session, same user — auto-injected LTM context should include the memory.
    scripted2 = [
        _resp(
            SimpleNamespace(
                role="assistant",
                content="You prefer metric units.",
                tool_calls=None,
            )
        )
    ]
    fake2 = _FakeClient(scripted2)
    monkeypatch.setattr("assistant.agent.loop.make_client", lambda _settings=None: fake2)
    result2 = run_agent(
        "What units do I prefer?",
        settings=settings,
        long_term=LongTermMemory(settings.memory_dir),
        user_id="alice",
        session_id="s2",
        summarize_fn=lambda prompt: "summary",
    )
    assert result2.recalled_memory_ids
    blob = json.dumps(fake2.chat.completions.seen_messages[0])
    assert "metric" in blob.lower()
