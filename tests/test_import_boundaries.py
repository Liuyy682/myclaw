def test_public_api_exports_stay_stable():
    import myclaw
    from myclaw import (
        AgentConfig,
        AgentLoop,
        AgentRunner,
        FakeProvider,
        FunctionTool,
        LLMResponse,
        OpenAICompatibleProvider,
        RunResult,
        Session,
        SessionManager,
        Tool,
        ToolCallRequest,
        ToolRegistry,
    )

    assert not hasattr(myclaw, "Agent")
    assert AgentConfig.__name__ == "AgentConfig"
    assert AgentLoop.__name__ == "AgentLoop"
    assert AgentRunner.__name__ == "AgentRunner"
    assert FakeProvider.__name__ == "FakeProvider"
    assert FunctionTool.__name__ == "FunctionTool"
    assert LLMResponse.__name__ == "LLMResponse"
    assert OpenAICompatibleProvider.__name__ == "OpenAICompatibleProvider"
    assert RunResult.__name__ == "RunResult"
    assert Session.__name__ == "Session"
    assert SessionManager.__name__ == "SessionManager"
    assert Tool.__name__ == "Tool"
    assert ToolCallRequest.__name__ == "ToolCallRequest"
    assert ToolRegistry.__name__ == "ToolRegistry"


def test_nanobot_style_internal_module_boundaries_are_available():
    from myclaw.agent.dispatcher import AgentDispatcher
    from myclaw.agent.loop import AgentLoop
    from myclaw.agent.runner import AgentRunner
    from myclaw.agent.types import AgentRunSpec
    from myclaw.bus import InboundMessage, MessageBus, OutboundMessage
    from myclaw.cli.commands import build_agent_loop
    from myclaw.config.env import load_env_file
    from myclaw.providers import LLMResponse
    from myclaw.providers.fake import FakeProvider
    from myclaw.providers.openai_compat import OpenAICompatibleProvider
    from myclaw.session import Session, SessionManager
    from myclaw.tools import FunctionTool, Tool, ToolCallRequest, ToolRegistry

    assert AgentDispatcher.__name__ == "AgentDispatcher"
    assert AgentLoop.__name__ == "AgentLoop"
    assert AgentRunner.__name__ == "AgentRunner"
    assert AgentRunSpec.__name__ == "AgentRunSpec"
    assert InboundMessage.__name__ == "InboundMessage"
    assert MessageBus.__name__ == "MessageBus"
    assert OutboundMessage.__name__ == "OutboundMessage"
    assert build_agent_loop.__name__ == "build_agent_loop"
    assert load_env_file.__name__ == "load_env_file"
    assert LLMResponse.__name__ == "LLMResponse"
    assert FakeProvider.__name__ == "FakeProvider"
    assert FunctionTool.__name__ == "FunctionTool"
    assert OpenAICompatibleProvider.__name__ == "OpenAICompatibleProvider"
    assert Session.__name__ == "Session"
    assert SessionManager.__name__ == "SessionManager"
    assert Tool.__name__ == "Tool"
    assert ToolCallRequest.__name__ == "ToolCallRequest"
    assert ToolRegistry.__name__ == "ToolRegistry"
