

import logging
import os
from typing import List
import chainlit as cl
import httpx
import json

from dotenv import load_dotenv



load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration
API_HOST = os.getenv("API_HOST", "localhost")
API_PORT = int(os.getenv("API_PORT", 8000))
API_BASE_URL = os.getenv("API_URL", f"http://{API_HOST}:{API_PORT}")


class APIClient:
    """ API client for interacting with the backend service."""

    
    @staticmethod
    async def stream_chat(message: str, user_id: str, session_id: str):
        """Stream chat response from the API."""
        async with httpx.AsyncClient(timeout=None) as client:
            url = f"{API_BASE_URL}/chat/stream"
            async with client.stream(
                "POST",
                url,
                json={
                    "message": message,
                    "user_id": user_id,
                    "session_id": session_id,
                    "metadata": {},
                },
            ) as response:
                response.raise_for_status()
                async for raw_line in response.aiter_lines():
                    line = (raw_line or "").strip()
                    if not line:
                        continue

                    if line.startswith("data:"):
                        data = line[5:].strip()
                    else:
                        data = line

                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        cl.logger.warning(f"Invalid JSON line: {line}")
    @staticmethod
    async def user_exists(user_id: str) -> bool:
        """Check if user exists."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            url = f"{API_BASE_URL}/users/{user_id}/exists"
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            return {
            "exists": data.get("exists", False),
            "user_id": data.get("user_id"),
            "session_count": data.get("session_count", 0)
            }
    @staticmethod        
    async def create_session(user_id: str):
        """Create a new session for the user."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            url = f"{API_BASE_URL}/users/{user_id}/sessions"
            response = await client.post(url)
            response.raise_for_status()
            return response.json().get("session_id")
    @staticmethod
    async def check_api_health() -> bool:
        """Check API health."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            url = f"{API_BASE_URL}/health"
            try:
                response = await client.get(url)
                response.raise_for_status()
                return True
            except httpx.RequestError:
                return False
    @staticmethod
    async def upload_files(file_paths: List[str]):
        """Upload multiple files to the API."""
        async with httpx.AsyncClient(timeout=300.0) as client:
            url = f"{API_BASE_URL}/documents/upload"
            
            files_to_upload = []
            file_handles = []
            
            try:
                for file_path in file_paths:
                    f = open(file_path, "rb")
                    file_handles.append(f)
                    files_to_upload.append(
                        ("files", (os.path.basename(file_path), f, "application/octet-stream"))
                    )
                
                response = await client.post(url, files=files_to_upload)
                response.raise_for_status()
                return response.json()
            
            finally:
                for f in file_handles:
                    f.close()


@cl.password_auth_callback
async def auth_callback(username: str, password: str):
    """Basic user login."""
    if not username.strip() or not password.strip():
        return None
    response = await APIClient.user_exists(username.strip())
    if not response.get("exists"):
        return None

    return cl.User(
        display_name=username.strip(),
        identifier=response.get("user_id", username.strip()),
        metadata={"session_count": response.get("session_count", 0)}
    )


@cl.on_chat_start
async def on_chat_start():
    """Start a new chat."""
    if not await APIClient.check_api_health():
        await cl.Message(content="❌ Unable to connect to API server. Please ensure the API is running.", author="System").send()
        return
    
    user = cl.user_session.get("user")
    user_id = user.identifier if user else "guest"
    cl.user_session.set("user_id", user_id)
    session_id= await APIClient.create_session(user_id)
    cl.user_session.set("session_id", session_id)


@cl.on_message
async def on_message(message: cl.Message):
    """Process chat messages."""
    user_id = cl.user_session.get("user_id")
    session_id = cl.user_session.get("session_id")
    content = message.content.strip()

    response_msg = cl.Message(content="", author="Assistant")
    await response_msg.send()

    token_buffer = []
    token_count = 0

    if message.elements:
        file_paths = []
        for element in message.elements:
            if isinstance(element, cl.File):
                file_paths.append(element.path)
        
        if file_paths:
            loading_msg = cl.Message(content="⏳ Documents exctracting and uploading...", author="System")
            await loading_msg.send()
            
            try:
                print(file_paths)
                await APIClient.upload_files(file_paths)
                await loading_msg.remove()
            except Exception as e:
                await loading_msg.remove()
                await cl.Message(content=f"Error: {str(e)}").send()
                return
        
    
    try:
        async for item in APIClient.stream_chat(content, user_id,session_id):
            item_type = item.get("type")

            if item_type == "text":
                token = item.get("content", "")
                await response_msg.stream_token(token)
                token_buffer.append(token)
                token_count += 1

                if token_count % 5 == 0:
                    await response_msg.update()
            if item_type == "tools":
                token = item.get("tools", [])
                token_str = " \n".join([f"[Tool: {tool.get('tool_name', '')}] Args: {tool.get('args', {})}" for tool in token])
                await response_msg.stream_token(token_str)

            elif item_type == "error":
                error_msg = item.get("content", "Unknown error")
                await cl.Message(content=f"❌ Error: {error_msg}", author="System").send()
                return

            elif item_type == "complete":
                break

        await response_msg.update()

    except Exception as e:
        await cl.Message(content=f"❌ Message could not be processed: {str(e)}", author="System").send()
        cl.logger.error(f"Chat error: {e}")


@cl.on_chat_end
async def on_chat_end():
    """Log when chat ends."""
    session_id = cl.user_session.get("session_id")
    cl.logger.info(f"Chat ended: {session_id}")
