from __future__ import annotations

from openai import OpenAI

from assistant.config import Settings, get_settings


def make_client(settings: Settings | None = None) -> OpenAI:
    s = settings or get_settings()
    return OpenAI(api_key=s.api_key, base_url=s.base_url)


def complete(
    messages: list[dict],
    *,
    temperature: float = 0.0,
    settings: Settings | None = None,
) -> str:
    """Single-turn chat completion (no tools). Used for smoke tests."""
    s = settings or get_settings()
    client = make_client(s)
    response = client.chat.completions.create(
        model=s.model,
        messages=messages,
        temperature=temperature,
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("LLM returned an empty completion.")
    return content
