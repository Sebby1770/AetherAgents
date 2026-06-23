"""Runtime configuration.

Loads settings from environment variables (prefixed ``AETHER_``) plus a couple
of well-known keys like ``OPENAI_API_KEY``. Works with or without the optional
``pydantic-settings`` package - it falls back to reading ``os.environ`` directly,
so importing the framework never fails on a missing dependency.
"""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import BaseModel


class Settings(BaseModel):
    """Framework-wide configuration."""

    default_model: str = "gpt-4o"
    temperature: float = 0.7
    max_steps: int = 8
    chroma_path: str = "./chroma_db"
    enable_tracing: bool = False
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> Settings:
        env = environ if environ is not None else os.environ
        data: dict[str, object] = {}
        mapping = {
            "default_model": ("AETHER_DEFAULT_MODEL", "LITELLM_MODEL"),
            "chroma_path": ("AETHER_CHROMA_PATH", "CHROMA_PATH"),
            "openai_api_key": ("OPENAI_API_KEY",),
            "anthropic_api_key": ("ANTHROPIC_API_KEY",),
        }
        for field, keys in mapping.items():
            for key in keys:
                if env.get(key):
                    data[field] = env[key]
                    break
        if env.get("AETHER_TEMPERATURE"):
            data["temperature"] = float(env["AETHER_TEMPERATURE"])
        if env.get("AETHER_MAX_STEPS"):
            data["max_steps"] = int(env["AETHER_MAX_STEPS"])
        if env.get("AETHER_ENABLE_TRACING"):
            data["enable_tracing"] = env["AETHER_ENABLE_TRACING"].lower() in ("1", "true", "yes")
        return cls(**data)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, loaded once from the environment."""
    return Settings.from_env()


# Backwards-compatible module-level handle.
settings = get_settings()
