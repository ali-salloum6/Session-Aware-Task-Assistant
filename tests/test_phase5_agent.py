import json
from types import SimpleNamespace

from assistant.agent import run_agent
from assistant.config import Settings
from assistant.memory import LongTermMemory


def _settings(tmp_path) -> Settings:
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "doc.md").write_text("# Doc\n\nhello\n", encoding="utf-8")
    return Settings(
        api_key="test-key",
        base_url="http://example.invalid/v1",
        model="fake-model",
        max_steps=6,
        kb_dir=kb,
        traces_dir=tmp_path / "traces",
        max_context_tokens=2000,
        keep_last_n_turns=4,
        summary_trigger_tokens=1200,
        recall_k=3,
        context_mode="bounded",
        memory_dir=tmp_path / "memory",
        default_user_id="alice",
        ltm_recall_k=3,
        dedup_distance=0.25,
        reflect_memory=True,
        agent_mode="memory",
        agent_backend="loop",
        checkpoint_path=tmp_path / "checkpoints" / "test.sqlite",
        langfuse_public_key="",
        langfuse_secret_key="",
        langfuse_host="https://cloud.langfuse.com",
    )


class _FakeCompletions:
    def __init__(self, scripted):
        self.scripted = list(scripted)

    def create(self, **kwargs):
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


def test_ephemeral_task_reflection_writes_nothing(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    ltm = LongTermMemory(settings.memory_dir, dedup_distance=0.25)
    scripted = [
        _resp(
            SimpleNamespace(
                role="assistant",
                content=None,
                tool_calls=[
                    _tool_call("calculator", json.dumps({"expression": "2+2"}), "c1")
                ],
            )
        ),
        _resp(
            SimpleNamespace(
                role="assistant",
                content="4",
                tool_calls=None,
            )
        ),
    ]
    monkeypatch.setattr(
        "assistant.agent.loop.make_client", lambda _settings=None: _FakeClient(scripted)
    )
    result = run_agent(
        "What is 2+2? Use the calculator.",
        settings=settings,
        long_term=ltm,
        user_id="alice",
        summarize_fn=lambda p: "summary",
        reflect_fn=lambda p: '{"memories":[]}',
    )
    assert result.answer == "4"
    assert result.reflection_writes == []
    assert ltm.count_for_user("alice") == 0


def test_preference_reflection_creates_then_dedups(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    ltm = LongTermMemory(settings.memory_dir, dedup_distance=0.25)

    scripted = [
        _resp(
            SimpleNamespace(
                role="assistant",
                content="I'll remember your metric preference.",
                tool_calls=None,
            )
        )
    ]
    monkeypatch.setattr(
        "assistant.agent.loop.make_client", lambda _settings=None: _FakeClient(scripted)
    )
    result = run_agent(
        "I prefer metric units.",
        settings=settings,
        long_term=ltm,
        user_id="alice",
        summarize_fn=lambda p: "summary",
        reflect_fn=lambda p: json.dumps(
            {
                "memories": [
                    {
                        "text": "User prefers metric units.",
                        "memory_type": "semantic",
                    }
                ]
            }
        ),
    )
    assert len(result.reflection_writes) == 1
    assert result.reflection_writes[0].action == "created"
    assert ltm.count_for_user("alice") == 1

    scripted2 = [
        _resp(
            SimpleNamespace(
                role="assistant",
                content="Noted again.",
                tool_calls=None,
            )
        )
    ]
    monkeypatch.setattr(
        "assistant.agent.loop.make_client", lambda _settings=None: _FakeClient(scripted2)
    )
    result2 = run_agent(
        "Please remember I prefer metric units.",
        settings=settings,
        long_term=ltm,
        user_id="alice",
        summarize_fn=lambda p: "summary",
        reflect_fn=lambda p: json.dumps(
            {
                "memories": [
                    {
                        "text": "User prefers metric units for measurements.",
                        "memory_type": "semantic",
                    }
                ]
            }
        ),
    )
    assert result2.reflection_writes[0].action == "skipped"
    assert ltm.count_for_user("alice") == 1
