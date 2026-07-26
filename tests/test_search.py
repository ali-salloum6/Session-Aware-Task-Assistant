from pathlib import Path

from assistant.tools.search import ScopedSearch

KB = Path(__file__).resolve().parents[1] / "kb"


def test_search_finds_battery():
    searcher = ScopedSearch(KB)
    out = searcher.search("Model X battery")
    assert "480 Wh" in out
    assert "product_specs.md" in out


def test_search_finds_hq():
    searcher = ScopedSearch(KB)
    out = searcher.search("headquarters Berlin")
    assert "Berlin" in out


def test_search_no_match():
    searcher = ScopedSearch(KB)
    out = searcher.search("unicorn teleportation")
    assert out.startswith("No documents matched")


def test_search_empty_query():
    searcher = ScopedSearch(KB)
    assert searcher.search("").startswith("search error:")
