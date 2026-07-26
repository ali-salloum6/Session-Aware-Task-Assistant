from assistant.memory.long_term import LongTermMemory
from assistant.memory.reflection import (
    MEMORY_POLICY_BLURB,
    apply_reflection,
    parse_reflection_json,
    reflect_memories,
)
from assistant.memory.reflection import ProposedMemory


def test_parse_reflection_json_ok():
    raw = '{"memories":[{"text":"User prefers metric","memory_type":"semantic"}]}'
    props = parse_reflection_json(raw)
    assert len(props) == 1
    assert props[0].text == "User prefers metric"


def test_parse_reflection_abstain_and_bad_json():
    assert parse_reflection_json('{"memories":[]}') == []
    assert parse_reflection_json("not json") == []
    assert parse_reflection_json("") == []


def test_parse_reflection_caps_at_three():
    items = ",".join(
        f'{{"text":"fact {i}","memory_type":"semantic"}}' for i in range(5)
    )
    props = parse_reflection_json('{"memories":[' + items + "]}")
    assert len(props) == 3


def test_dedup_skips_near_duplicate(tmp_path):
    ltm = LongTermMemory(tmp_path / "mem", dedup_distance=0.25)
    first = ltm.remember_policy(
        "User prefers metric units for all measurements.",
        user_id="alice",
        memory_type="semantic",
    )
    assert first.action == "created"
    second = ltm.remember_policy(
        "The user prefers metric units for all measurements.",
        user_id="alice",
        memory_type="semantic",
    )
    assert second.action == "skipped"
    assert ltm.count_for_user("alice") == 1


def test_apply_reflection_writes_and_dedups(tmp_path):
    ltm = LongTermMemory(tmp_path / "mem", dedup_distance=0.25)
    props = [
        ProposedMemory("User prefers metric units.", "semantic"),
        ProposedMemory("User prefers metric units for measurements.", "semantic"),
    ]
    writes = apply_reflection(ltm, user_id="alice", proposals=props)
    assert writes[0].action == "created"
    assert writes[1].action == "skipped"
    assert ltm.count_for_user("alice") == 1


def test_reflect_memories_uses_fn():
    def fake_reflect(_prompt: str) -> str:
        return '{"memories":[]}'

    assert (
        reflect_memories(
            task="2+2",
            answer="4",
            tool_events_summary="- calculator",
            reflect_fn=fake_reflect,
        )
        == []
    )


def test_policy_blurb_present():
    assert "abstain" in MEMORY_POLICY_BLURB.lower() or "0–3" in MEMORY_POLICY_BLURB
