import asyncio
import aiohttp
import streamlit as st
import requests
import json
from dotenv import load_dotenv
import os
from typing import Dict, List, Any

load_dotenv()

# Application configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8081))
API_URL = os.getenv("API_URL", f"http://localhost:{API_PORT}")

st.set_page_config(page_title="Agenctic RAG", page_icon="🤖", layout="wide")

USER_ID = "user"

# Session state init
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []


def check_health(base_url: str):
    """Check if the API is healthy."""
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        if resp.status_code == 200:
            return True
        return False
    except Exception:
        return False

def upload_documents(files: List, config: Dict[str, Any], base_url: str) -> Dict[str, Any]:
    """Upload documents with configuration."""
    try:
        # Prepare files for upload
        upload_files = []
        for uploaded_file in files:
            upload_files.append(
                ("files", (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type))
            )
        
        # Prepare headers with configuration
        headers = {
            "X-Ingestion-Config": json.dumps(config)
        }
        
        # Upload files
        response = requests.post(
            f"{base_url}/documents/upload",
            files=upload_files,
            headers=headers
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}



async def stream_chat(message: str, base_url: str):
    request_data = {
        "message": message,
        "session_id": st.session_state.session_id,
        "user_id": USER_ID,
        "search_type": "hybrid"
    }

    response_box = st.empty()
    full_response = ""

    async with aiohttp.ClientSession() as session:
        async with session.post(f"{base_url}/chat/stream", json=request_data) as resp:
            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue

                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                if data.get("type") == "session":
                    st.session_state.session_id = data.get("session_id")

                elif data.get("type") == "text":
                    content = data.get("content", "")
                    full_response += content
                    response_box.write(full_response)
                    
                elif data.get("type") == "tools":
                    tools = data.get("tools", [])
                    for tool in tools:
                        tool_name = tool.get("tool_name", "")
                        tool_args = tool.get("args", {})
                        full_response += f"\n [Tool: {tool_name}] \n Args: {tool_args}"
                    response_box.write(full_response)
                    
                elif data.get("type") == "end":
                    break

    response_box.write(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

def run_async(message: str, base_url: str):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(stream_chat(message, base_url))



# Sidebar for all controls
with st.sidebar:
    st.header("🔧 API Settings")
    base_url = st.text_input("API URL", value=API_URL)
    
    if st.button("🏥 Check Health"):
        if check_health(base_url):
            st.success("✅ API is healthy")
        else:
            st.error("❌ API not reachable")
    
    st.divider()
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()
    
    st.divider()
    
    st.header("📤 Document Upload")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Select files",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'doc', 'txt', 'md', 'png', 'jpg', 'jpeg'],
        help="Supported: PDF, DOCX, DOC, TXT, MD, PNG, JPG, JPEG"
    )
    
    if uploaded_files:
        st.write(f"📁 {len(uploaded_files)} files selected")
    
    st.divider()
    
    st.header("⚙️ Settings")
    
    # Basic settings
    chunk_size = st.slider("Chunk Size", min_value=200, max_value=2000, value=850, step=50)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=150, step=25)
    use_ocr = st.checkbox("Use OCR", value=True)
    use_semantic = st.checkbox("Use Semantic Chunker", value=True)
    
    st.divider()
    
    # Upload button
    if st.button("🚀 Upload & Process", type="primary", disabled=not uploaded_files):
        if uploaded_files:
            # Prepare configuration
            config = {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "enable_ocr": use_ocr,
                "use_semantic_chunking": use_semantic,
                "enable_vlm": True,
                "enable_table_extraction": True,
                "enable_image_extraction": True,
                "save_to_files": True,
                "output_format": "markdown"
            }
            
            with st.spinner("📤 Uploading documents..."):
                result = upload_documents(uploaded_files, config, base_url)
                
                if result.get("success"):
                    processed = result.get("processed_documents") or result.get("processed") or 0
                    success = result.get("ingest_success") or 0
                    st.success(f"✅ Processed: {processed}, Successful: {success}")

                else:
                    st.error(f"❌ Upload failed: {result.get('error')}")

# Main chat
st.title("Agentic RAG")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
if prompt := st.chat_input("Ask something..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            run_async(prompt, base_url)