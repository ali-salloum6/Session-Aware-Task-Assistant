from __future__ import annotations

from dataclasses import dataclass, field


MAX_FIELD_CHARS = 400


def _clip(value: str, limit: int = MAX_FIELD_CHARS) -> str:
    text = value.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


@dataclass
class Scratchpad:
    """Structured short-term working state (PLAN / DONE / NEXT)."""

    plan: str = ""
    done: str = ""
    next: str = ""
    history: list[dict[str, str]] = field(default_factory=list)

    def format(self) -> str:
        return (
            "SCRATCHPAD (working state — keep this updated on multi-step tasks):\n"
            f"PLAN: {self.plan or '(empty)'}\n"
            f"DONE: {self.done or '(empty)'}\n"
            f"NEXT: {self.next or '(empty)'}"
        )

    def snapshot(self) -> dict[str, str]:
        return {"plan": self.plan, "done": self.done, "next": self.next}

    def update(
        self,
        *,
        plan: str | None = None,
        done: str | None = None,
        next: str | None = None,
    ) -> str:
        """Apply partial updates. Omit a field to leave it unchanged; pass '' to clear."""
        if plan is not None:
            self.plan = _clip(plan)
        if done is not None:
            self.done = _clip(done)
        if next is not None:
            self.next = _clip(next)
        snap = self.snapshot()
        self.history.append(snap)
        return "scratchpad updated:\n" + self.format()

    def to_dict(self) -> dict:
        return {"current": self.snapshot(), "history": list(self.history)}
