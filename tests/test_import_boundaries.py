def test_public_api_exports_stay_stable():
    from myclaw import Agent, AgentConfig, FakeProvider, OpenAICompatibleProvider, RunResult

    assert Agent.__name__ == "Agent"
    assert AgentConfig.__name__ == "AgentConfig"
    assert FakeProvider.__name__ == "FakeProvider"
    assert OpenAICompatibleProvider.__name__ == "OpenAICompatibleProvider"
    assert RunResult.__name__ == "RunResult"


def test_nanobot_style_internal_module_boundaries_are_available():
    from myclaw.agent.runner import Agent
    from myclaw.cli.commands import build_agent
    from myclaw.config.env import load_env_file
    from myclaw.providers.fake import FakeProvider
    from myclaw.providers.openai_compat import OpenAICompatibleProvider

    assert Agent.__name__ == "Agent"
    assert build_agent.__name__ == "build_agent"
    assert load_env_file.__name__ == "load_env_file"
    assert FakeProvider.__name__ == "FakeProvider"
    assert OpenAICompatibleProvider.__name__ == "OpenAICompatibleProvider"
