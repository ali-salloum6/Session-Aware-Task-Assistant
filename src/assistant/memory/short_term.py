from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

_TOKEN = re.compile(r"[a-z0-9]+")


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token). Good enough for budgets."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def estimate_messages(messages: list[dict]) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content)
        tool_calls = msg.get("tool_calls") or []
        for tc in tool_calls:
            fn = tc.get("function") or {}
            total += estimate_tokens(str(fn.get("name", "")))
            total += estimate_tokens(str(fn.get("arguments", "")))
    return total


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


@dataclass
class Turn:
    turn_id: int
    role: str
    content: str


SummarizeFn = Callable[[str], str]


@dataclass
class ShortTermMemory:
    """Session short-term memory: rolling summary + selective turn recall + recent verbatim."""

    max_context_tokens: int = 2000
    keep_last_n_turns: int = 4
    summary_trigger_tokens: int = 1200
    recall_k: int = 3
    mode: str = "bounded"  # "bounded" | "full" (B1 append-all)

    turns: list[Turn] = field(default_factory=list)
    summary: str = ""
    _verbatim_start: int = 0
    _next_id: int = 0

    def add(self, role: str, content: str) -> Turn:
        turn = Turn(turn_id=self._next_id, role=role, content=content.strip())
        self._next_id += 1
        self.turns.append(turn)
        return turn

    def reset(self) -> None:
        self.turns.clear()
        self.summary = ""
        self._verbatim_start = 0
        self._next_id = 0

    def verbatim_turns(self) -> list[Turn]:
        return self.turns[self._verbatim_start :]

    def older_turns(self) -> list[Turn]:
        """Turns outside the recent verbatim window (eligible for selective recall)."""
        if self.keep_last_n_turns <= 0:
            return list(self.turns)
        cutoff = max(0, len(self.turns) - self.keep_last_n_turns)
        return self.turns[:cutoff]

    def selective_recall(self, query: str, k: int | None = None) -> list[Turn]:
        """Lexical overlap retrieval over older turns (no embedding API yet)."""
        top_k = self.recall_k if k is None else k
        q = _tokenize(query)
        if not q or top_k <= 0:
            return []
        scored: list[tuple[int, Turn]] = []
        for turn in self.older_turns():
            score = len(q & _tokenize(turn.content))
            if score > 0:
                scored.append((score, turn))
        scored.sort(key=lambda item: (-item[0], -item[1].turn_id))
        return [turn for _, turn in scored[:top_k]]

    def estimate_assembled_tokens(self, query: str = "") -> int:
        return estimate_messages(self.build_history_messages(query))

    def build_history_messages(self, query: str) -> list[dict]:
        """History prefix inserted after the live system/scratchpad message."""
        if self.mode == "full":
            return [{"role": t.role, "content": t.content} for t in self.turns]

        messages: list[dict] = []
        if self.summary.strip():
            messages.append(
                {
                    "role": "system",
                    "content": "Running summary of earlier conversation:\n" + self.summary.strip(),
                }
            )

        recalled = self.selective_recall(query)
        if recalled:
            block = "\n\n".join(
                f"[turn {t.turn_id} | {t.role}] {t.content}" for t in recalled
            )
            messages.append(
                {
                    "role": "system",
                    "content": "Relevant earlier turns (selective recall):\n" + block,
                }
            )

        for turn in self.verbatim_turns():
            messages.append({"role": turn.role, "content": turn.content})
        return messages

    def needs_compression(self) -> bool:
        if self.mode == "full":
            return False
        verbatim = self.verbatim_turns()
        if len(verbatim) <= self.keep_last_n_turns:
            return False
        # Compress when verbatim region (plus summary) exceeds the trigger.
        tokens = estimate_tokens(self.summary)
        tokens += sum(estimate_tokens(t.content) for t in verbatim)
        return tokens >= self.summary_trigger_tokens

    def compress_if_needed(self, summarize_fn: SummarizeFn) -> bool:
        """Fold older verbatim turns into the rolling summary. Returns True if compressed."""
        if not self.needs_compression():
            return False

        fold_end = len(self.turns) - self.keep_last_n_turns
        to_fold = self.turns[self._verbatim_start : fold_end]
        if not to_fold:
            return False

        transcript = "\n".join(f"{t.role}: {t.content}" for t in to_fold)
        prompt = (
            "Update the running summary of a conversation. Preserve concrete facts, "
            "names, numbers, preferences, and decisions. Be concise.\n\n"
            f"Prior summary:\n{self.summary or '(none)'}\n\n"
            f"New turns to fold in:\n{transcript}\n\n"
            "Return only the updated summary."
        )
        self.summary = summarize_fn(prompt).strip()
        self._verbatim_start = fold_end
        return True

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "summary": self.summary,
            "verbatim_start": self._verbatim_start,
            "max_context_tokens": self.max_context_tokens,
            "keep_last_n_turns": self.keep_last_n_turns,
            "summary_trigger_tokens": self.summary_trigger_tokens,
            "recall_k": self.recall_k,
            "turns": [
                {"turn_id": t.turn_id, "role": t.role, "content": t.content}
                for t in self.turns
            ],
            "assembled_tokens_empty_query": self.estimate_assembled_tokens(""),
        }
