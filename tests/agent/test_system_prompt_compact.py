"""Compact system-prompt mode tests."""

from run_agent import AIAgent
from agent.system_prompt import build_system_prompt


def test_compact_system_prompt_omits_default_bloat():
    agent = AIAgent(
        model="glm-5.2",
        api_key="inspect-only",
        base_url="https://api.z.ai/api/anthropic",
        provider="zai",
        api_mode="anthropic_messages",
        quiet_mode=True,
        save_trajectories=False,
        platform="telegram",
        enabled_toolsets=["terminal", "memory", "session_search", "vision"],
    )
    agent._compact_system_prompt = True

    prompt = build_system_prompt(agent)

    assert "# Hermes Dispatcher" in prompt
    assert "Conversation started:" in prompt
    assert "<available_skills>" not in prompt
    assert "# Project Context" not in prompt
    assert "MEMORY (your personal notes)" not in prompt
    assert "USER PROFILE (who the user is)" not in prompt
    assert "# Tool-use enforcement" not in prompt
    assert len(prompt.encode("utf-8")) < 2500
