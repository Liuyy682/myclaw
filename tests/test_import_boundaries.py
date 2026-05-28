def test_public_api_exports_stay_stable():
    import myclaw
    from myclaw import (
        AgentConfig,
        AgentLoop,
        AgentRunner,
        FakeProvider,
        OpenAICompatibleProvider,
        RunResult,
    )

    assert not hasattr(myclaw, "Agent")
    assert AgentConfig.__name__ == "AgentConfig"
    assert AgentLoop.__name__ == "AgentLoop"
    assert AgentRunner.__name__ == "AgentRunner"
    assert FakeProvider.__name__ == "FakeProvider"
    assert OpenAICompatibleProvider.__name__ == "OpenAICompatibleProvider"
    assert RunResult.__name__ == "RunResult"


def test_nanobot_style_internal_module_boundaries_are_available():
    from myclaw.agent.loop import AgentLoop
    from myclaw.agent.runner import AgentRunner
    from myclaw.agent.types import AgentRunSpec
    from myclaw.cli.commands import build_agent_loop
    from myclaw.config.env import load_env_file
    from myclaw.providers.fake import FakeProvider
    from myclaw.providers.openai_compat import OpenAICompatibleProvider

    assert AgentLoop.__name__ == "AgentLoop"
    assert AgentRunner.__name__ == "AgentRunner"
    assert AgentRunSpec.__name__ == "AgentRunSpec"
    assert build_agent_loop.__name__ == "build_agent_loop"
    assert load_env_file.__name__ == "load_env_file"
    assert FakeProvider.__name__ == "FakeProvider"
    assert OpenAICompatibleProvider.__name__ == "OpenAICompatibleProvider"
