import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from fastmcp import FastMCP, Context
import asyncio
import json
import os
from dotenv import load_dotenv

load_dotenv()

from ..db.memory_db import Mem0Context,get_mem0_client,test_db_connection

logger = logging.getLogger(__name__)


@asynccontextmanager
async def mem0_lifespan(server: FastMCP) -> AsyncIterator[Mem0Context]:
    """
    Manages the Mem0 client lifecycle.
    
    Args:
        server: The FastMCP server instance
        
    Yields:
        Mem0Context: The context containing the Mem0 client
    """
    # Test DB connection
    db_ok = test_db_connection()
    if not db_ok:
        raise RuntimeError("Database connection failed!")
    
    mem0_client = get_mem0_client()
    
    try:
        yield Mem0Context(mem0_client=mem0_client)
    finally:
        pass

mcp = FastMCP(
    name="memory_mcp",
    version="0.1.0",
    lifespan=mem0_lifespan,
    host="0.0.0.0",
    port=8050,
    json_response=True,
)


@mcp.tool()
async def save_memory(ctx: Context,user_id: str, text: str) -> str:
    """Save information to your long-term memory.

    This tool is designed to store any type of information that might be useful in the future.
    The content will be processed and indexed for later retrieval through semantic search.

    Args:
        ctx: The MCP server provided context which includes the Mem0 client
        user_id: The user ID to associate the memory with
        text: The content to store in memory, including any relevant details and context
    """
    try:
        mem0_client = ctx.request_context.lifespan_context.mem0_client
        messages = [{"role": "user", "content": text}]
        mem0_client.add(messages, user_id=user_id)
        return f"Successfully saved memory: {text[:100]}..." if len(text) > 100 else f"Successfully saved memory: {text}"
    except Exception as e:
        return f"Error saving memory: {str(e)}"


@mcp.tool()
async def get_all_memories(ctx: Context,user_id:str) -> str:
    """Get all stored memories for the user.
    
    Call this tool when you need complete context of all previously memories.

    Args:
        ctx: The MCP server provided context which includes the Mem0 client
        user_id: The user ID to retrieve memories for

    Returns a JSON formatted list of all stored memories, including when they were created
    and their content. Results are paginated with a default of 50 items per page.
    """
    try:
        mem0_client = ctx.request_context.lifespan_context.mem0_client
        memories = mem0_client.get_all(user_id=user_id)
        if isinstance(memories, dict) and "results" in memories:
            flattened_memories = [memory["memory"] for memory in memories["results"]]
        else:
            flattened_memories = memories
        return json.dumps(flattened_memories, indent=2)
    except Exception as e:
        return f"Error retrieving memories: {str(e)}"



@mcp.tool()
async def search_memory(ctx: Context, query: str, user_id: str, limit: int = 3) -> str:
    """Search for memories using a query string.

    This tool searches through stored memories using semantic search to find
    relevant information based on the provided query.

    Args:
        ctx: The MCP server provided context which includes the Mem0 client
        query: The search query to find relevant memories
        user_id: The user ID to search memories for
        limit: Maximum number of memories to return (default: 3)

    Returns:
        JSON formatted list of matching memories
    """
    try:
        mem0_client = ctx.request_context.lifespan_context.mem0_client
        memories = mem0_client.search(query, user_id=user_id, limit=limit)

        if isinstance(memories, dict) and "results" in memories:
            flattened_memories = [memory["memory"] for memory in memories["results"]]
        else:
            flattened_memories = memories
        return json.dumps(flattened_memories, indent=2)
    except Exception as e:
        return f"Error searching memories: {str(e)}"


@mcp.tool()
async def delele_memory(ctx: Context, user_id: str, query: str, limit: int = 3) -> str:
    """Delete memories matching a search query.

    This tool searches for memories matching the provided query and deletes
    all matching memories for the specified user.

    Args:
        ctx: The MCP server provided context which includes the Mem0 client
        user_id: The user ID to delete memories for
        query: The search query to find memories to delete
        limit: Maximum number of memories to search and delete (default: 3)

    Returns:
        String indicating how many memories were deleted
    """
    try:
        mem0_client = ctx.request_context.lifespan_context.mem0_client
        find_memories = mem0_client.search(query, user_id=user_id, limit=limit)

        if isinstance(find_memories, dict) and "results" in find_memories:
            memory_ids = [memory["id"] for memory in find_memories["results"]]

        else:
            results = find_memories.get("results", [])
            memory_ids = [m["id"] for m in results]

        for memory in memory_ids:
            mem0_client.delete(memory)

        return f"Deleted {len(memory_ids)} memories."

    except Exception as e:
        return f"Error deleting memories: {e}"


@mcp.tool()
async def delete_all_memory(ctx: Context, user_id: str) -> str:
    """Delete all memories for a specific user.

    This tool removes all stored memories for the specified user ID.
    Use with caution as this action cannot be undone.

    Args:
        ctx: The MCP server provided context which includes the Mem0 client
        user_id: The user ID to delete all memories for

    Returns:
        String indicating how many memories were deleted
    """
    try:
        mem0_client = ctx.request_context.lifespan_context.mem0_client
        result = mem0_client.delete_all(user_id=user_id)
        return f"Deleted {user_id} - {result} memories."

    except Exception as e:
        return f"Error deleting memories: {e}"


@mcp.tool()
async def update_memory(ctx: Context, query: str, data: str, user_id: str) -> str:
    """Update memories matching a search query with new data.

    This tool searches for memories matching the provided query and updates
    them with the new data for the specified user.

    Args:
        ctx: The MCP server provided context which includes the Mem0 client
        query: The search query to find memories to update
        data: The new data to replace the existing memory content
        user_id: The user ID to update memories for

    Returns:
        String indicating how many memories were updated
    """
    try:
        mem0_client = ctx.request_context.lifespan_context.mem0_client
        find_memories = mem0_client.search(query, user_id=user_id)

        if isinstance(find_memories, dict) and "results" in find_memories:
            memory_ids = [memory["id"] for memory in find_memories["results"]]

        else:
            results = find_memories.get("results", [])
            memory_ids = [m["id"] for m in results]

        for memory in memory_ids:
            mem0_client.update(memory, data)

        return f"updated {len(memory_ids)} memories."

    except Exception as e:
        return f"Error update memories: {e}"


async def main():
    """Main function to run the Memory MCP server.

    Starts the FastMCP server with the configured transport method.
    The transport method can be set via the MEMORY_MCP_TRANSPORT environment variable.
    """
    transport = os.getenv("MEMORY_MCP_TRANSPORT", "streamable-http")
    await mcp.run_async(transport=transport)


if __name__ == "__main__":
    asyncio.run(main())
