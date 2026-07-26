"""Tool contract tests: pure functions + dispatch harness (Observation, not exception)."""

from __future__ import annotations

from pathlib import Path

from assistant.memory import Scratchpad
from assistant.memory.long_term import MAX_MEMORY_CHARS, LongTermMemory
from assistant.tools import build_tool_registry, dispatch_tool
from assistant.tools.calculator import MAX_EXPR_CHARS, calculator
from assistant.tools.search import ScopedSearch


def _kb(tmp_path: Path) -> Path:
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "doc.md").write_text("# Doc\n\nhello Model X battery 480\n", encoding="utf-8")
    return kb


def _registry(tmp_path: Path, *, with_ltm: bool = False):
    pad = Scratchpad()
    ltm = LongTermMemory(tmp_path / "mem") if with_ltm else None
    return build_tool_registry(_kb(tmp_path), pad, long_term=ltm, user_id="u")


def assert_observation_error(out: str, *, prefix: str, must_contain: str) -> None:
    """Harness contract: string Observation the model can read; never a raised exception."""
    assert isinstance(out, str), type(out)
    assert out.startswith(prefix), out
    assert must_contain.lower() in out.lower(), out
    assert "Traceback" not in out


# --- calculator (pure) ---


def test_calculator_happy_path_returns_exact_result():
    assert calculator("2*(3+4)") == "14"
    assert calculator("480*3") == "1440"


def test_calculator_empty_expression_is_observation_not_exception():
    out = calculator("")
    assert_observation_error(out, prefix="calculator error:", must_contain="empty")


def test_calculator_division_by_zero_is_observation_not_exception():
    out = calculator("1/0")
    assert_observation_error(
        out, prefix="calculator error:", must_contain="division by zero"
    )


def test_calculator_too_long_expression_is_observation_not_exception():
    out = calculator("1+" * (MAX_EXPR_CHARS + 10) + "1")
    assert_observation_error(out, prefix="calculator error:", must_contain="too long")


def test_calculator_unsafe_name_is_observation_not_exception():
    out = calculator("__import__('os').system('x')")
    assert_observation_error(
        out, prefix="calculator error:", must_contain="unsupported or unsafe"
    )


# --- calculator via dispatch (harness) ---


def test_dispatch_calculator_division_by_zero_returns_observation(tmp_path):
    registry = _registry(tmp_path)
    out = dispatch_tool(registry, "calculator", {"expression": "1/0"})
    assert_observation_error(
        out, prefix="calculator error:", must_contain="division by zero"
    )


def test_dispatch_calculator_happy_path(tmp_path):
    registry = _registry(tmp_path)
    assert dispatch_tool(registry, "calculator", {"expression": "6*7"}) == "42"


# --- search (pure + dispatch) ---


def test_search_happy_path_returns_allowlisted_doc(tmp_path):
    out = ScopedSearch(_kb(tmp_path)).search("Model X battery")
    assert "doc.md" in out
    assert "480" in out


def test_search_path_like_query_is_observation_not_exception(tmp_path):
    out = ScopedSearch(_kb(tmp_path)).search("../secrets")
    assert_observation_error(out, prefix="search error:", must_contain="path-like")


def test_search_query_too_long_is_observation_not_exception(tmp_path):
    out = ScopedSearch(_kb(tmp_path)).search("x" * 300)
    assert_observation_error(out, prefix="search error:", must_contain="too long")


def test_dispatch_search_path_probe_returns_observation(tmp_path):
    out = dispatch_tool(_registry(tmp_path), "search", {"query": "../../etc/passwd"})
    assert_observation_error(out, prefix="search error:", must_contain="path-like")


# --- remember (policy + dispatch) ---


def test_dispatch_remember_rejects_empty_text_as_observation(tmp_path):
    out = dispatch_tool(
        _registry(tmp_path, with_ltm=True),
        "remember",
        {"text": "  ", "memory_type": "semantic"},
    )
    assert_observation_error(out, prefix="remember error:", must_contain="non-empty")


def test_dispatch_remember_rejects_too_long_text_as_observation(tmp_path):
    out = dispatch_tool(
        _registry(tmp_path, with_ltm=True),
        "remember",
        {"text": "m" * (MAX_MEMORY_CHARS + 20), "memory_type": "semantic"},
    )
    assert_observation_error(out, prefix="remember error:", must_contain="too long")


def test_dispatch_unknown_tool_is_observation_not_exception(tmp_path):
    out = dispatch_tool(_registry(tmp_path), "nope", {})
    assert out.startswith("unknown tool:")
