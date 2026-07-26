from assistant.memory.scratchpad import MAX_FIELD_CHARS, Scratchpad


def test_scratchpad_format_empty():
    pad = Scratchpad()
    text = pad.format()
    assert "PLAN: (empty)" in text
    assert "DONE: (empty)" in text
    assert "NEXT: (empty)" in text


def test_scratchpad_partial_update_and_history():
    pad = Scratchpad()
    pad.update(plan="1) search 2) calc", next="search battery")
    assert pad.plan.startswith("1)")
    assert pad.done == ""
    assert pad.next == "search battery"
    pad.update(done="found 480 Wh", next="calc 480*3")
    assert pad.done == "found 480 Wh"
    assert len(pad.history) == 2
    assert pad.history[0]["next"] == "search battery"


def test_scratchpad_clips_long_fields():
    pad = Scratchpad()
    pad.update(plan="x" * (MAX_FIELD_CHARS + 50))
    assert len(pad.plan) == MAX_FIELD_CHARS
    assert pad.plan.endswith("…")


def test_scratchpad_clear_with_empty_string():
    pad = Scratchpad(plan="keep", done="wipe me", next="keep")
    pad.update(done="")
    assert pad.plan == "keep"
    assert pad.done == ""
    assert pad.next == "keep"
