"""Minimal stdio MCP server used by tests. Exposes a single echo tool."""

from mcp.server.fastmcp import FastMCP

server = FastMCP("echo-server")


@server.tool()
def echo(text: str) -> str:
    """Echo the provided text back to the caller."""
    return f"echo: {text}"


if __name__ == "__main__":
    server.run("stdio")
