import shutil
import logging
import json
from typing import Dict, Any
from pydantic_ai.mcp import MCPServerStreamableHTTP, MCPServerSSE, MCPServerStdio
logger = logging.getLogger(__name__)

class MCPServer:
    """Factory for creating MCP server instances without initializing sessions."""

    @staticmethod
    def create(server_name: str, server_config: Dict[str, Any]):
        protocol = server_config.get("protocol", "stdio")

        if protocol == "stdio":
            command = shutil.which("npx") if server_config.get("command") == "npx" else server_config.get("command")
            if not command:
                raise ValueError(f"Invalid command for server '{server_name}'")

            return MCPServerStdio(
                command=command,
                args=server_config.get("args", []),
                env=server_config.get("env"),
                timeout=30
                )

        elif protocol == "sse":
            url = server_config.get("url")
            if not url:
                raise ValueError(f"URL required for SSE server '{server_name}'")

            return MCPServerSSE(
                url=url,
                headers=server_config.get("headers", {}),
                timeout=30
            )

        elif protocol == "http-stream":
            url = server_config.get("url")
            if not url:
                raise ValueError(f"URL required for HTTP stream server '{server_name}'")

            return MCPServerStreamableHTTP(
                url=url,
                headers=server_config.get("headers", {}),
                timeout=30
            )

        else:
            raise ValueError(f"Unsupported protocol: {protocol}")


async def load_mcp_servers(config_path: str = "mcp_config.json") -> Dict[str, Any]:
    """Load all MCP servers from configuration without initializing them."""
    logger.info(f"Loading MCP servers config from {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    logger.debug(f"Parsed config: {config}")

    servers = {}
    mcp_servers_config = config.get("mcpServers", {})

    for server_name, server_config in mcp_servers_config.items():
        try:
            logger.debug(f"Creating server '{server_name}' with config: {server_config}")
            server = MCPServer.create(server_name, server_config)
            servers[server_name] = server
            logger.info(f" Loaded server '{server_name}' ({server_config.get('protocol', 'stdio')})")
        except Exception as e:
            logger.exception(f"Failed to load server {server_name}")
            continue

    return servers
