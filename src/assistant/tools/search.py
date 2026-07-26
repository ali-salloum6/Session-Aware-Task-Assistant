from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_TOKEN = re.compile(r"[a-z0-9]+")
MAX_QUERY_CHARS = 200
ALLOWED_SUFFIXES = (".md",)


@dataclass(frozen=True)
class Doc:
    doc_id: str
    title: str
    text: str


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def _is_allowed_doc(path: Path, kb_root: Path) -> bool:
    try:
        resolved = path.resolve()
        resolved.relative_to(kb_root.resolve())
    except ValueError:
        return False
    return resolved.suffix.lower() in ALLOWED_SUFFIXES and resolved.is_file()


def load_kb(kb_dir: Path) -> list[Doc]:
    kb_root = Path(kb_dir)
    if not kb_root.is_dir():
        raise FileNotFoundError(f"Knowledge base directory not found: {kb_root}")
    docs: list[Doc] = []
    for path in sorted(kb_root.glob("*.md")):
        if not _is_allowed_doc(path, kb_root):
            continue
        text = path.read_text(encoding="utf-8")
        title = path.stem.replace("_", " ")
        for line in text.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        docs.append(Doc(doc_id=path.name, title=title, text=text))
    if not docs:
        raise FileNotFoundError(f"No allow-listed markdown documents in {kb_root}")
    return docs


class ScopedSearch:
    """Keyword search over a fixed in-repo markdown corpus (no open web)."""

    def __init__(self, kb_dir: Path):
        self.kb_dir = Path(kb_dir).resolve()
        self.docs = load_kb(self.kb_dir)
        self.allowlist = {d.doc_id for d in self.docs}

    def search(self, query: str, k: int = 3) -> str:
        if not isinstance(query, str):
            return "search error: query must be a string"
        q = query.strip()
        if not q:
            return "search error: empty query"
        if len(q) > MAX_QUERY_CHARS:
            return (
                f"search error: query too long "
                f"({len(q)} chars, max {MAX_QUERY_CHARS})"
            )
        # Reject path-like probes; corpus is allow-listed filenames only.
        if ".." in q or "/" in q or "\\" in q:
            return (
                "search error: path-like queries are not allowed; "
                "use keywords (e.g. 'Model X battery')"
            )
        q_tokens = _tokenize(q)
        if not q_tokens:
            return "search error: no searchable tokens in query"

        try:
            top_k = int(k)
        except (TypeError, ValueError):
            return "search error: k must be an integer"
        if top_k <= 0:
            return "search error: k must be >= 1"
        top_k = min(top_k, 10)

        scored: list[tuple[int, Doc]] = []
        for doc in self.docs:
            if doc.doc_id not in self.allowlist:
                continue
            doc_tokens = _tokenize(doc.title + " " + doc.text)
            score = len(q_tokens & doc_tokens)
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda item: (-item[0], item[1].doc_id))

        if not scored:
            return f"No documents matched {q!r}."

        chunks: list[str] = []
        for score, doc in scored[:top_k]:
            body = doc.text.strip()
            if len(body) > 800:
                body = body[:800] + "…"
            chunks.append(f"[{doc.doc_id} | score={score}] {body}")
        return "\n\n---\n\n".join(chunks)
