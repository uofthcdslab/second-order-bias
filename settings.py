"""Centralized settings loader.

All pipeline tunables live in `config.yaml`. Import them like:

    from settings import settings
    for m in settings.models:
        print(m["id"])
    concurrency = settings.run_sob["concurrency"]
    prompts = settings.prompts  # dict[str, str]
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load .env once so os.environ is populated for downstream consumers.
load_dotenv()

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
_CACHE: dict[str, Any] | None = None


def load_config(path: Path | str | None = None, *, reload: bool = False) -> dict[str, Any]:
    """Read config.yaml and cache the result."""
    global _CACHE
    if _CACHE is not None and not reload:
        return _CACHE

    cfg_path = Path(path) if path else _CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml not found at {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    _CACHE = data
    return data


class Settings:
    """Eagerly-loaded, attribute-style access to the YAML config."""

    def __init__(self, config: dict[str, Any] | None = None):
        self._cfg = config if config is not None else load_config()

    # ── top-level groups ────────────────────────────────────────────────────
    @property
    def api(self) -> dict[str, Any]:
        return self._cfg["api"]

    @property
    def paths(self) -> dict[str, Any]:
        return self._cfg["paths"]

    @property
    def run_sob(self) -> dict[str, Any]:
        return self._cfg["run_sob"]

    @property
    def parse_sob(self) -> dict[str, Any]:
        return self._cfg["parse_sob"]

    @property
    def address_parsing(self) -> dict[str, Any]:
        return self._cfg["address_parsing"]

    @property
    def models(self) -> list[dict[str, Any]]:
        """All registered models, filtered by `run_sob.selected_models` if set."""
        all_models = self._cfg["models"]
        selected = self._cfg.get("run_sob", {}).get("selected_models", []) or []
        if not selected:
            return all_models
        keep = set(selected)
        return [m for m in all_models if m["label"] in keep]

    @property
    def all_models(self) -> list[dict[str, Any]]:
        """All registered models, ignoring the `selected_models` filter."""
        return self._cfg["models"]

    @property
    def prompts(self) -> dict[str, str]:
        return self._cfg["prompts"]

    # ── convenience accessors ───────────────────────────────────────────────
    @property
    def api_key(self) -> str:
        env_name = self.api.get("api_key_env", "OPENROUTER_API_KEY")
        return os.environ.get(env_name, "")

    @property
    def base_url(self) -> str:
        return self.api["base_url"]

    @property
    def model_labels(self) -> list[str]:
        """Labels of all models registered in the `models:` block."""
        return [m["label"] for m in self.models]


# Module-level singleton, importable as `from settings import settings`.
settings = Settings()


__all__ = ["Settings", "load_config", "settings"]
