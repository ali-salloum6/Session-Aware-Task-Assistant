from assistant.memory.short_term import ShortTermMemory, estimate_tokens


def _fold_summarize(prompt: str) -> str:
    # Deterministic stand-in for unit tests — keeps key phrases.
    return "SUMMARY: " + " | ".join(
        line.strip()
        for line in prompt.splitlines()
        if line.startswith("user:") or line.startswith("assistant:")
    )[:500]


def test_selective_recall_recovers_early_codeword_after_filler():
    stm = ShortTermMemory(keep_last_n_turns=2, recall_k=2)
    stm.add("user", "Remember the warehouse codeword is ORBIT-9.")
    stm.add("assistant", "Got it, codeword ORBIT-9 stored for this session.")
    for i in range(10):
        stm.add("user", f"filler question number {i} about weather and lunch")
        stm.add("assistant", f"filler answer {i} with no useful content")

    recalled = stm.selective_recall("what is the warehouse codeword?")
    assert any("ORBIT-9" in t.content for t in recalled)


def test_compression_keeps_exactly_last_n_verbatim_and_preserves_fact_via_recall():
    stm = ShortTermMemory(
        keep_last_n_turns=2,
        summary_trigger_tokens=50,
        recall_k=3,
    )
    stm.add("user", "My preferred unit system is metric.")
    stm.add("assistant", "Understood, metric units.")
    for i in range(6):
        stm.add("user", f"padding turn {i} " + ("word " * 40))
        stm.add("assistant", f"ack {i} " + ("text " * 40))

    assert stm.needs_compression()
    assert stm.compress_if_needed(_fold_summarize) is True
    assert len(stm.verbatim_turns()) == 2
    assert "metric" in stm.summary.lower()
    recalled = stm.selective_recall("preferred unit system metric")
    assert any("metric" in t.content.lower() for t in recalled)


def test_bounded_assembled_tokens_strictly_less_than_full_history():
    stm = ShortTermMemory(
        mode="bounded",
        keep_last_n_turns=2,
        summary_trigger_tokens=80,
        max_context_tokens=300,
        recall_k=1,
    )
    stm.add("user", "Secret project name is NEBULA.")
    stm.add("assistant", "Noted NEBULA.")
    for i in range(20):
        stm.add("user", f"long filler user {i} " + ("padding " * 30))
        stm.add("assistant", f"long filler assistant {i} " + ("padding " * 30))
    assert stm.compress_if_needed(_fold_summarize)

    query = "What is the secret project name NEBULA?"
    bounded = stm.estimate_assembled_tokens(query)
    full = ShortTermMemory(
        mode="full",
        turns=list(stm.turns),
        keep_last_n_turns=stm.keep_last_n_turns,
    )
    full_tokens = full.estimate_assembled_tokens(query)
    assert bounded < full_tokens


def test_after_compression_verbatim_window_equals_keep_last_n():
    stm = ShortTermMemory(
        mode="bounded",
        keep_last_n_turns=4,
        summary_trigger_tokens=60,
        recall_k=1,
    )
    for i in range(30):
        stm.add("user", f"u{i} " + ("pad " * 20))
        stm.add("assistant", f"a{i} " + ("pad " * 20))
    assert stm.compress_if_needed(_fold_summarize)
    assert len(stm.verbatim_turns()) == 4


def test_estimate_tokens_uses_four_chars_per_token():
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 40) == 10
