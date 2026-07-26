from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

import chromadb

MemoryType = Literal["episodic", "semantic", "procedural"]
MemorySource = Literal["user", "agent", "reflection"]
WriteAction = Literal["created", "skipped", "updated"]

VALID_TYPES = frozenset({"episodic", "semantic", "procedural"})
VALID_SOURCES = frozenset({"user", "agent", "reflection"})

# Chroma cosine space: distance ~= 1 - cosine_similarity; lower = more similar.
DEFAULT_DEDUP_DISTANCE = 0.18
MAX_MEMORY_CHARS = 500


@dataclass(frozen=True)
class MemoryEntry:
    id: str
    text: str
    memory_type: str
    timestamp: str
    user_id: str
    source: str
    distance: float | None = None

    def format(self) -> str:
        dist = f" dist={self.distance:.4f}" if self.distance is not None else ""
        return (
            f"[{self.id[:8]} | {self.memory_type} | {self.source} | {self.timestamp}{dist}] "
            f"{self.text}"
        )


@dataclass(frozen=True)
class WriteResult:
    action: WriteAction
    entry: MemoryEntry
    reason: str

    def format(self) -> str:
        return f"{self.action}: {self.entry.format()} ({self.reason})"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LongTermMemory:
    """Persistent vector memory scoped by user_id (Chroma)."""

    def __init__(
        self,
        persist_dir: Path,
        *,
        collection_name: str = "long_term",
        dedup_distance: float = DEFAULT_DEDUP_DISTANCE,
        max_text_chars: int = MAX_MEMORY_CHARS,
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.dedup_distance = dedup_distance
        self.max_text_chars = max_text_chars
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def remember(
        self,
        text: str,
        *,
        user_id: str,
        memory_type: str = "semantic",
        source: str = "agent",
    ) -> MemoryEntry:
        """Unconditional write (no dedup). Prefer remember_policy() for agent writes."""
        cleaned, memory_type, source, user_id = self._validate(
            text, memory_type=memory_type, source=source, user_id=user_id
        )
        entry = MemoryEntry(
            id=uuid4().hex,
            text=cleaned,
            memory_type=memory_type,
            timestamp=_utc_now(),
            user_id=user_id,
            source=source,
        )
        self._collection.add(
            ids=[entry.id],
            documents=[entry.text],
            metadatas=[
                {
                    "memory_type": entry.memory_type,
                    "timestamp": entry.timestamp,
                    "user_id": entry.user_id,
                    "source": entry.source,
                }
            ],
        )
        return entry

    def remember_policy(
        self,
        text: str,
        *,
        user_id: str,
        memory_type: str = "semantic",
        source: str = "agent",
    ) -> WriteResult:
        """Write with near-duplicate skip (store little, well)."""
        cleaned, memory_type, source, user_id = self._validate(
            text, memory_type=memory_type, source=source, user_id=user_id
        )
        near = self.recall(cleaned, user_id=user_id, k=1)
        if (
            near
            and near[0].distance is not None
            and near[0].distance <= self.dedup_distance
        ):
            return WriteResult(
                action="skipped",
                entry=near[0],
                reason=f"near-duplicate of {near[0].id[:8]} (dist={near[0].distance:.4f})",
            )
        entry = self.remember(
            cleaned,
            user_id=user_id,
            memory_type=memory_type,
            source=source,
        )
        return WriteResult(action="created", entry=entry, reason="new durable fact")

    def recall(
        self,
        query: str,
        *,
        user_id: str,
        k: int = 3,
    ) -> list[MemoryEntry]:
        q = query.strip()
        if not q or k <= 0:
            return []
        if not user_id.strip():
            return []

        count = self._collection.count()
        if count == 0:
            return []

        result = self._collection.query(
            query_texts=[q],
            n_results=min(k, count),
            where={"user_id": user_id},
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]

        entries: list[MemoryEntry] = []
        for i, mem_id in enumerate(ids):
            meta = metas[i] or {}
            if meta.get("user_id") != user_id:
                continue
            entries.append(
                MemoryEntry(
                    id=mem_id,
                    text=docs[i] or "",
                    memory_type=str(meta.get("memory_type", "semantic")),
                    timestamp=str(meta.get("timestamp", "")),
                    user_id=str(meta.get("user_id", "")),
                    source=str(meta.get("source", "agent")),
                    distance=float(dists[i]) if dists and dists[i] is not None else None,
                )
            )
        return entries

    def count_for_user(self, user_id: str) -> int:
        if not user_id.strip():
            return 0
        got = self._collection.get(where={"user_id": user_id}, include=[])
        return len(got.get("ids") or [])

    def list_for_user(self, user_id: str) -> list[MemoryEntry]:
        got = self._collection.get(
            where={"user_id": user_id},
            include=["documents", "metadatas"],
        )
        entries: list[MemoryEntry] = []
        for i, mem_id in enumerate(got.get("ids") or []):
            meta = (got.get("metadatas") or [])[i] or {}
            docs = got.get("documents") or []
            entries.append(
                MemoryEntry(
                    id=mem_id,
                    text=docs[i] or "",
                    memory_type=str(meta.get("memory_type", "semantic")),
                    timestamp=str(meta.get("timestamp", "")),
                    user_id=str(meta.get("user_id", "")),
                    source=str(meta.get("source", "agent")),
                )
            )
        return entries

    def format_recall(self, entries: list[MemoryEntry]) -> str:
        if not entries:
            return "No long-term memories matched."
        return "\n".join(e.format() for e in entries)

    def _validate(
        self,
        text: str,
        *,
        memory_type: str,
        source: str,
        user_id: str,
    ) -> tuple[str, str, str, str]:
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("remember() requires non-empty text")
        if len(cleaned) > self.max_text_chars:
            raise ValueError(
                f"text too long ({len(cleaned)} chars, max {self.max_text_chars})"
            )
        if not user_id.strip():
            raise ValueError("remember() requires user_id")
        if memory_type not in VALID_TYPES:
            raise ValueError(
                f"memory_type must be one of {sorted(VALID_TYPES)}, got {memory_type!r}"
            )
        if source not in VALID_SOURCES:
            raise ValueError(
                f"source must be one of {sorted(VALID_SOURCES)}, got {source!r}"
            )
        return cleaned, memory_type, source, user_id.strip()
