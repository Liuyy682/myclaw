import subprocess
import sys

def test_root_package_does_not_aggregate_runtime_exports():
    import myclaw

    aggregated_names = {
        "AgentConfig",
        "AgentLoop",
        "FakeProvider",
        "HttpGatewayServer",
        "ToolRegistry",
        "run_gateway",
    }
    assert aggregated_names.isdisjoint(vars(myclaw))


def test_explicit_domain_boundaries_are_available():
    from myclaw.agent import AgentConfig, AgentDispatcher, AgentLoop, AgentRunner, DispatcherRuntime
    from myclaw.bus import InboundMessage, MessageBus, OutboundMessage
    from myclaw.gateway.server import HttpGatewayServer, run_gateway
    from myclaw.providers import FakeProvider, LLMResponse, OpenAICompatibleProvider
    from myclaw.session import Session, SessionManager
    from myclaw.tools import FunctionTool, Tool, ToolCallRequest, ToolRegistry, build_default_tool_registry

    exported = (
        AgentConfig,
        AgentDispatcher,
        AgentLoop,
        AgentRunner,
        DispatcherRuntime,
        InboundMessage,
        MessageBus,
        OutboundMessage,
        HttpGatewayServer,
        run_gateway,
        FakeProvider,
        LLMResponse,
        OpenAICompatibleProvider,
        Session,
        SessionManager,
        FunctionTool,
        Tool,
        ToolCallRequest,
        ToolRegistry,
        build_default_tool_registry,
    )
    assert all(item.__name__ for item in exported)


def test_gateway_package_does_not_reexport_server_symbols():
    import myclaw.gateway

    assert not hasattr(myclaw.gateway, "HttpGatewayServer")
    assert not hasattr(myclaw.gateway, "run_gateway")


def test_domain_packages_import_in_a_fresh_process():
    script = """
from myclaw.tools import ToolRegistry
from myclaw.providers import FakeProvider
from myclaw.agent import AgentLoop
from myclaw.gateway.server import HttpGatewayServer
assert all((ToolRegistry, FakeProvider, AgentLoop, HttpGatewayServer))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
