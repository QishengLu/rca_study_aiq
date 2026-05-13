"""PromptManager — caching loader for agent prompt YAMLs.

100% aligned with ThinkDepthAI's manager. Each agent ships its own
`prompts/agents/langgraph/rca.yaml` containing 4 keys:

    RCA_ANALYSIS_SP
    RCA_ANALYSIS_UP
    COMPRESS_FINDINGS_SP
    COMPRESS_FINDINGS_UP

Placeholders filled at format-time:
    {date}, {incident_description}, {agent_contract}
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class PromptManager:
    """Caching prompt loader. Same singleton pattern as ThinkDepthAI."""

    _cache: dict[str, dict[str, str]] = {}

    @classmethod
    def get_prompts(cls, prompt_path: str | Path) -> dict[str, str]:
        """Return the full prompt dict for a given yaml file."""
        key = str(Path(prompt_path).resolve())
        if key not in cls._cache:
            with open(prompt_path) as f:
                cls._cache[key] = yaml.safe_load(f) or {}
        return cls._cache[key]

    @classmethod
    def clear(cls) -> None:
        cls._cache.clear()


def load_rca_prompts(agent_pkg_dir: Path) -> dict[str, str]:
    """Convenience: load `prompts/agents/langgraph/rca.yaml` from an agent pkg.

    Args:
        agent_pkg_dir: e.g. Path(__file__).parent for an agent module that
            wants its own prompts.
    """
    p = Path(agent_pkg_dir) / "prompts" / "agents" / "langgraph" / "rca.yaml"
    return PromptManager.get_prompts(p)
