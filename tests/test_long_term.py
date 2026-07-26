from assistant.memory.long_term import LongTermMemory


def test_remember_recall_same_user(tmp_path):
    ltm = LongTermMemory(tmp_path / "mem")
    entry = ltm.remember(
        "User prefers metric units for all measurements.",
        user_id="alice",
        memory_type="semantic",
        source="user",
    )
    assert entry.user_id == "alice"

    hits = ltm.recall("preferred measurement units", user_id="alice", k=3)
    assert hits
    assert any("metric" in h.text.lower() for h in hits)
    assert all(h.user_id == "alice" for h in hits)


def test_recall_scoped_by_user_id(tmp_path):
    ltm = LongTermMemory(tmp_path / "mem")
    ltm.remember("Alice likes metric units.", user_id="alice", memory_type="semantic")
    ltm.remember("Bob likes imperial units.", user_id="bob", memory_type="semantic")

    alice_hits = ltm.recall("units preference", user_id="alice", k=5)
    bob_hits = ltm.recall("units preference", user_id="bob", k=5)
    assert alice_hits
    assert bob_hits
    assert all(h.user_id == "alice" for h in alice_hits)
    assert all(h.user_id == "bob" for h in bob_hits)
    assert any("metric" in h.text.lower() for h in alice_hits)
    assert any("imperial" in h.text.lower() for h in bob_hits)
    assert not any("imperial" in h.text.lower() for h in alice_hits)


def test_persistence_across_reopen(tmp_path):
    path = tmp_path / "mem"
    LongTermMemory(path).remember(
        "Warehouse codeword is ORBIT-9.",
        user_id="alice",
        memory_type="episodic",
        source="agent",
    )
    # New client / "session" — same persist dir.
    ltm2 = LongTermMemory(path)
    hits = ltm2.recall("warehouse codeword", user_id="alice", k=3)
    assert any("ORBIT-9" in h.text for h in hits)


def test_remember_rejects_bad_type(tmp_path):
    ltm = LongTermMemory(tmp_path / "mem")
    try:
        ltm.remember("x", user_id="alice", memory_type="vibes")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "memory_type" in str(exc)
