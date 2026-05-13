"""AIQ (NVIDIA AIRA) eval agent — wraps legacy agent_runner.py."""
from __future__ import annotations

from pathlib import Path

from sota_rca.legacy_agent import LegacySubprocessAgent, LegacyAgentConfig

_FW_DIR = Path(__file__).resolve().parent.parent.parent.parent  # frameworks/aiq


class AIQEvalAgent(LegacySubprocessAgent):
    @staticmethod
    def name() -> str:
        return "aiq"

    @classmethod
    def _default_config(cls) -> LegacyAgentConfig:
        return LegacyAgentConfig(
            framework_name="aiq",
            framework_dir=_FW_DIR,
            cmd=["uv", "run", "python", "agent_runner.py"],
            prompt_yaml=_FW_DIR / "src" / "aiq_rca" / "prompts" / "agents" / "langgraph" / "rca.yaml",
            timeout=1800,
            multi_agent=False,
        )
