from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model: str
    max_steps: int
    kb_dir: Path
    traces_dir: Path
    max_context_tokens: int
    keep_last_n_turns: int
    summary_trigger_tokens: int
    recall_k: int
    context_mode: str  # bounded | full
    memory_dir: Path
    default_user_id: str
    ltm_recall_k: int
    dedup_distance: float
    reflect_memory: bool
    agent_mode: str  # stateless | full_history | memory
    agent_backend: str  # graph | loop
    checkpoint_path: Path
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_host: str

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


def get_settings() -> Settings:
    api_key = os.getenv("LLM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "LLM_API_KEY is missing. Copy .env.example to .env and set your key."
        )
    return Settings(
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1").rstrip(
            "/"
        ),
        model=os.getenv("LLM_MODEL", "google/gemini-2.5-flash"),
        max_steps=int(os.getenv("MAX_STEPS", "8")),
        kb_dir=_REPO_ROOT / "kb",
        traces_dir=_REPO_ROOT / "traces",
        max_context_tokens=int(os.getenv("MAX_CONTEXT_TOKENS", "2000")),
        keep_last_n_turns=int(os.getenv("KEEP_LAST_N_TURNS", "4")),
        summary_trigger_tokens=int(os.getenv("SUMMARY_TRIGGER_TOKENS", "1200")),
        recall_k=int(os.getenv("RECALL_K", "3")),
        context_mode=os.getenv("CONTEXT_MODE", "bounded").strip().lower(),
        memory_dir=Path(os.getenv("MEMORY_DIR", str(_REPO_ROOT / "data" / "memory"))),
        default_user_id=os.getenv("DEFAULT_USER_ID", "default").strip() or "default",
        ltm_recall_k=int(os.getenv("LTM_RECALL_K", "3")),
        dedup_distance=float(os.getenv("DEDUP_DISTANCE", "0.18")),
        reflect_memory=os.getenv("REFLECT_MEMORY", "1").strip().lower()
        not in {"0", "false", "no", "off"},
        agent_mode=os.getenv("AGENT_MODE", "memory").strip().lower(),
        agent_backend=os.getenv("AGENT_BACKEND", "graph").strip().lower(),
        checkpoint_path=Path(
            os.getenv(
                "CHECKPOINT_PATH",
                str(_REPO_ROOT / "data" / "checkpoints" / "langgraph.sqlite"),
            )
        ),
        langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "").strip(),
        langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY", "").strip(),
        langfuse_host=(
            os.getenv("LANGFUSE_HOST")
            or os.getenv("LANGFUSE_BASE_URL")
            or "https://cloud.langfuse.com"
        ).strip().rstrip("/"),
    )


def make_short_term(settings: Settings | None = None) -> "ShortTermMemory":
    from assistant.memory.short_term import ShortTermMemory

    s = settings or get_settings()
    return ShortTermMemory(
        max_context_tokens=s.max_context_tokens,
        keep_last_n_turns=s.keep_last_n_turns,
        summary_trigger_tokens=s.summary_trigger_tokens,
        recall_k=s.recall_k,
        mode=s.context_mode if s.context_mode in {"bounded", "full"} else "bounded",
    )


def make_long_term(settings: Settings | None = None) -> "LongTermMemory":
    from assistant.memory.long_term import LongTermMemory

    s = settings or get_settings()
    return LongTermMemory(s.memory_dir, dedup_distance=s.dedup_distance)
