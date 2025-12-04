import sys
import io
import json
import logging
import traceback
import multiprocessing
import threading
import signal
import atexit
import socket
import struct
import argparse
from concurrent.futures import ThreadPoolExecutor
import queue
import time
import os
import inspect
import functools
import typing
from multiprocessing import Manager
from multiprocessing.managers import SyncManager
import uuid
import asyncio
from typing import Any, Optional, Dict, List, Callable, get_type_hints
from contextlib import AsyncExitStack

# Your existing imports (Placeholder)
# import pmtpy_loader
# from plugin_validation import plugin_validation
# from run_pipeline import run_pipeline, validate, reset_id
# from sandbox import execute_plugin_in_sandbox
# from pipeline_core import Workflow
# from init_plugin import init_plugin

# LLM imports
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# Socket configuration
HOST = '127.0.0.1'
PORT = 65432

MAX_CLIENT_THREADS = multiprocessing.cpu_count() * 2
MAX_PROCESS_WORKERS = multiprocessing.cpu_count()

# ============================================================================
# Session Management (Unchanged)
# ============================================================================
class ClientSession:
    """Track chat history and context for each client"""
    def __init__(self, client_id: str, client_address: tuple):
        self.client_id = client_id
        self.client_address = client_address
        self.chat_history: List[Dict] = []
        self.created_at = time.time()
        self.last_activity = time.time()
        self.metadata: Dict = {}
        self.lock = threading.Lock()

    def add_message(self, role: str, content: str, **kwargs):
        with self.lock:
            self.last_activity = time.time()
            message = {
                "role": role,
                "content": content,
                "timestamp": time.time(),
                **kwargs
            }
            self.chat_history.append(message)

    def get_history(self, limit: Optional[int] = None) -> List[Dict]:
        with self.lock:
            if limit:
                return self.chat_history[-limit:]
            return self.chat_history.copy()

    def clear_history(self):
        with self.lock:
            self.chat_history.clear()

    def to_dict(self) -> Dict:
        with self.lock:
            return {
                "client_id": self.client_id,
                "client_address": str(self.client_address),
                "message_count": len(self.chat_history),
                "created_at": self.created_at,
                "last_activity": self.last_activity,
                "elapsed": time.time() - self.created_at,
                "metadata": self.metadata
            }

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, ClientSession] = {}
        self.lock = threading.Lock()

    def create_session(self, client_address: tuple) -> ClientSession:
        client_id = str(uuid.uuid4())
        with self.lock:
            session = ClientSession(client_id, client_address)
            self.sessions[client_id] = session
        return session

    def get_session(self, client_id: str) -> Optional[ClientSession]:
        with self.lock:
            return self.sessions.get(client_id)

    def remove_session(self, client_id: str):
        with self.lock:
            self.sessions.pop(client_id, None)

    def list_sessions(self) -> List[Dict]:
        with self.lock:
            return [s.to_dict() for s in self.sessions.values()]

# ============================================================================
# Helpers: QueueWriter, RedirectOutputToQueue, TaskTracker, ProcessPool 
# (Kept Unchanged - Condensed for brevity)
# ============================================================================
class QueueWriter(io.StringIO):
    def init(self, queue, task_id=None):
        super().init()
        self.queue = queue
        self.task_id = task_id
    def write(self, text):
        # ... (Implementation same as provided)
        pass 
    def flush(self): pass

class ProcessPool:
    def __init__(self, max_workers):
        self.max_workers = max_workers
        self.active_processes = []
        self.lock = threading.Lock()
        self.shutdown_flag = False
    
    def get_active_count(self):
        with self.lock:
            self.active_processes = [p for p in self.active_processes if p.is_alive()]
            return len(self.active_processes)
            
    def submit(self, target, args):
        # ... (Implementation same as provided)
        return None
    
    def shutdown(self, wait=True, timeout=5):
        # ... (Implementation same as provided)
        pass

class TaskTracker:
    # ... (Implementation same as provided)
    def __init__(self): self.tasks = {}; self.lock = threading.Lock()
    def create_process_task(self, task_type, metadata=None): return str(uuid.uuid4())
    def cancel(self, task_id): return {"success": False}
    def cancel_all(self): return []
    def list_all(self): return {}

# ============================================================================
# Socket Connection
# ============================================================================
class SocketConnection:
    """Wrapper for socket to handle JSON message sending/receiving"""
    def __init__(self, sock):
        self.sock = sock
        self.lock = threading.Lock()

    def send(self, obj):
        with self.lock:
            try:
                data = json.dumps(obj).encode('utf-8')
                msg_length = struct.pack('>I', len(data))
                self.sock.sendall(msg_length + data)
            except Exception as e:
                print(f"Socket Send Error: {e}")
                raise Exception(f"Failed to send message: {e}")

    def recv(self):
        try:
            raw_msglen = self._recvall(4)
            if not raw_msglen:
                return None
            msglen = struct.unpack('>I', raw_msglen)[0]
            data = self._recvall(msglen)
            if not data:
                return None
            return json.loads(data.decode('utf-8'))
        except Exception as e:
            print(f"Socket Recv Error: {e}")
            raise Exception(f"Failed to receive message: {e}")

    def _recvall(self, n):
        data = bytearray()
        while len(data) < n:
            packet = self.sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return bytes(data)

    def close(self):
        try: self.sock.shutdown(socket.SHUT_RDWR)
        except: pass
        try: self.sock.close()
        except: pass

# ============================================================================
# NEW: Tool Registry (FastMCP style)
# ============================================================================

class InteractionContext:
    """
    Context object passed to tools to allow them to communicate with the client.
    """
    def __init__(self, conn: SocketConnection, session: ClientSession):
        self.conn = conn
        self.session = session

class ToolRegistry:
    """
    Registry to handle decorated tool functions and schema generation.
    """
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: List[Dict] = []

    def tool(self):
        """Decorator to register a function as a tool."""
        def decorator(func):
            self._register(func)
            return func
        return decorator

    def _register(self, func: Callable):
        """Parse function signature and docstring into OpenAI Schema."""
        name = func.__name__
        doc = func.__doc__ or "No description provided."
        sig = inspect.signature(func)
        
        properties = {}
        required = []
        
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object"
        }

        for param_name, param in sig.parameters.items():
            # Skip the context parameter if it exists
            if param.annotation == InteractionContext or param_name in ['ctx', 'context']:
                continue
                
            param_type = "string" # default
            if param.annotation in type_map:
                param_type = type_map[param.annotation]
            elif hasattr(param.annotation, "__origin__") and param.annotation.__origin__ == list:
                param_type = "array"
            
            properties[param_name] = {
                "type": param_type,
                # In a real implementation, use docstring parsing libraries 
                # like docstring_parser to extract param descriptions
                "description": f"Parameter {param_name}" 
            }
            
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": doc.strip(),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
        
        # Store the wrapped function and its schema
        self._tools[name] = func
        self._schemas.append(schema)
        print(f"[ToolRegistry] Registered tool: {name}")

    def get_openai_tools(self) -> List[Dict]:
        return self._schemas

    def get_tool(self, name: str) -> Optional[Callable]:
        return self._tools.get(name)

# Initialize Global Registry
registry = ToolRegistry()

# ============================================================================
# USER DEFINED TOOLS (The Setup you requested)
# ============================================================================

@registry.tool()
def get_client_datasets(ctx: InteractionContext):
    """
    Retrieve the list of datasets currently available on the client's machine.
    Returns: List of dataset metadata.
    """
    print(f"[Server Tool] Fetching lists from client...")
    
    # 1. Construct Protocol Message
    req_id = str(uuid.uuid4())
    payload = {
        "type": "context_query",
        "query": "list_datasets",
        "req_id": req_id,
        "args": {}
    }
    
    # 2. Send to Client
    ctx.conn.send(payload)
    
    # 3. Wait for specific response
    # In a production system, this should have a timeout loop handling other message types
    while True:
        resp = ctx.conn.recv()
        if resp and resp.get("type") == "context_response" and resp.get("req_id") == req_id:
            data = resp.get("data")
            return json.dumps(data)
        if not resp:
            return "Error: Client disconnected during tool call."
            
@registry.tool()
def rearrange_dataset(ctx: InteractionContext, dataset_id: str, criteria: str):
    """
    Rearranges or filters a specific dataset based on criteria.
    Args:
        dataset_id: ID of the dataset to process.
        criteria: Description of how to filter/sort (e.g. "only liver data").
    """
    print(f"[Server Tool] Processing dataset {dataset_id} with criteria: {criteria}")
    
    # 1. Fetch Sample Data from Client (Or full data if file upload logic existed)
    req_id = str(uuid.uuid4())
    ctx.conn.send({
        "type": "context_query", 
        "query": "get_sample", 
        "req_id": req_id,
        "args": {"dataset_id": dataset_id}
    })
    
    raw_data = []
    while True:
        resp = ctx.conn.recv()
        if resp and resp.get("type") == "context_response" and resp.get("req_id") == req_id:
            raw_data = resp.get("data")
            break
    
    # 2. DO SERVER SIDE LOGIC (Pandas, Pipeline, etc.)
    # This is where your pipeline_core would actually run.
    # Simulating processing:
    result_summary = f"Filtered {len(raw_data)} rows. Found strong correlation in {criteria}."
    
    # 3. Push Result to Client UI
    ctx.conn.send({
        "type": "ui_render",
        "component": "table",
        "data": {"status": "Processed", "criteria": criteria, "sample_result": raw_data},
        "req_id": str(uuid.uuid4())
    })
    
    return result_summary

@registry.tool()
def show_data(ctx: InteractionContext, data_type: str, content: str):
    """
    Display data or visualization on the client's UI.
    args:
        data_type: The type of display ('table', 'chart', 'text')
        content: The content string or JSON string to render.
    """
    print(f"[Tool] show_data called: {data_type}")
    
    request_payload = {
        "type": "tool_call_request",
        "tool_call_id": str(uuid.uuid4()),
        "function": "show_data",
        "arguments": {"data_type": data_type, "content": content}
    }
    
    ctx.conn.send(request_payload)
    raw_response = ctx.conn.recv()
    
    if raw_response and raw_response.get("type") == "tool_call_result":
        return "UI updated successfully"
    
    return "Failed to update UI"

# ============================================================================
# LLM Handler
# ============================================================================
class LLMHandler:
    """Handle LLM interactions using the ToolRegistry"""

    def __init__(self, api_key: str, model: str = "anthropic/claude-3.5-sonnet"):
        self.llm_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = model

    async def process_query_streaming(
        self, 
        messages: List[Dict],
        conn: SocketConnection,
        session: ClientSession
    ) -> str:
        """
        Process a query with streaming responses and tool calling.
        Uses the global `registry` to find and execute tools.
        """
        
        # Prepare Context for tools
        ctx = InteractionContext(conn, session)
        
        available_tools = registry.get_openai_tools()
        max_iterations = 10 
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            try:
                # 1. Call LLM
                response = await self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=available_tools if available_tools else None,
                    stream=True
                )
                
                full_content = ""
                tool_calls_data = {}
                finish_reason = None
                
                # 2. Stream Tokens
                async for chunk in response:
                    choice = chunk.choices[0] if chunk.choices else None
                    if not choice: continue
                    
                    delta = choice.delta
                    if choice.finish_reason: finish_reason = choice.finish_reason
                    
                    if delta.content:
                        full_content += delta.content
                        conn.send({
                            "type": "llm_stream",
                            "content": delta.content
                        })
                    
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index
                            if idx not in tool_calls_data:
                                tool_calls_data[idx] = {"id": "", "function": {"name": "", "arguments": ""}}
                            if tc.id: tool_calls_data[idx]["id"] = tc.id
                            if tc.function:
                                if tc.function.name: tool_calls_data[idx]["function"]["name"] = tc.function.name
                                if tc.function.arguments: tool_calls_data[idx]["function"]["arguments"] += tc.function.arguments
                
                # 3. Handle Completion or Tool Call
                if tool_calls_data and finish_reason == "tool_calls":
                    
                    # Append Assistant message
                    tool_calls_cleaned = []
                    for idx in sorted(tool_calls_data.keys()):
                        tc = tool_calls_data[idx]
                        tool_calls_cleaned.append({
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"]
                            }
                        })
                    
                    messages.append({
                        "role": "assistant",
                        "content": full_content if full_content else None,
                        "tool_calls": tool_calls_cleaned
                    })

                    # Execute Tools
                    for tool_call in tool_calls_cleaned:
                        func_name = tool_call["function"]["name"]
                        func_args_str = tool_call["function"]["arguments"]
                        call_id = tool_call["id"]
                        
                        # Get python function from registry
                        py_function = registry.get_tool(func_name)
                        
                        result_content = ""
                        if py_function:
                            try:
                                args = json.loads(func_args_str) if func_args_str else {}
                                
                                print(f"\n[Server] Executing Local Proxy: {func_name}")
                                
                                # CALL THE DECORATED TOOL
                                # This function internally sends socket msg to client and waits
                                result_obj = py_function(ctx, **args)
                                
                                if isinstance(result_obj, (dict, list)):
                                    result_content = json.dumps(result_obj)
                                else:
                                    result_content = str(result_obj)

                            except Exception as e:
                                result_content = json.dumps({"error": f"Tool Execution Error: {str(e)}"})
                        else:
                            result_content = json.dumps({"error": f"Tool {func_name} not found on server registry"})

                        # Append Tool Result
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": result_content
                        })

                    # Loop back to LLM with new history
                    continue
                
                else:
                    return full_content if full_content else "No content generated."

            except Exception as e:
                err = f"LLM Loop Error: {e}"
                print(err)
                conn.send({"type": "error", "message": err})
                return err
        
        return "Max iterations reached."

# ============================================================================
# Main Unified Socket Server (Only updated methods shown)
# ============================================================================
class UnifiedSocketServer:
    def __init__(self, host=HOST, port=PORT, max_client_threads=None, max_process_workers=None):
        self.host = host
        self.port = port
        self.max_client_threads = max_client_threads or MAX_CLIENT_THREADS
        self.max_process_workers = max_process_workers or MAX_PROCESS_WORKERS
        
        self.client_executor = ThreadPoolExecutor(max_workers=self.max_client_threads)
        self.process_pool = ProcessPool(max_workers=self.max_process_workers)
        self.task_tracker = TaskTracker()
        self.session_manager = SessionManager()
        
        # LLM handler
        api_key = os.getenv("OPENROUTER_API_KEY")
        self.llm_handler = LLMHandler(api_key)
        
        self.lock = threading.Lock()
        self.shutdown_flag = threading.Event()
        atexit.register(self.shutdown)

    # ... (Existing pipeline methods: execute_single_pipeline, execute_batch, execute_plugin) ...

    def handle_chat_streaming(self, params, conn: SocketConnection, session: ClientSession):
        """
        Handle streaming chat with LLM
        """
        try:
            user_message = params.get("message", "")
            if not user_message:
                yield {"type": "error", "message": "No message provided"}
                return
            
            session.add_message("user", user_message)
            
            # Current history setup
            messages = [
                {"role": msg["role"], "content": msg["content"]}
                # Simple filter to only keep openai meaningful roles
                for msg in session.get_history() 
                if msg["role"] in ["system", "user", "assistant", "tool"]
            ]
            
            # Create new event loop for this thread to run the Async LLM client
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                assistant_response = loop.run_until_complete(
                    self.llm_handler.process_query_streaming(messages, conn, session)
                )
                session.add_message("assistant", assistant_response)
                yield {"type": "llm_complete", "content": assistant_response}
            finally:
                loop.close()
                
        except Exception as e:
            yield {"type": "error", "message": f"Chat error: {str(e)}"}

    def handle_request(self, request, conn: SocketConnection, session: ClientSession):
        """Route requests"""
        try:
            method = request.get("method")
            params = request.get("params", {})
            
            if method == "run_pipeline":
                # return self.execute_single_pipeline(params)
                pass 
            elif method == "chat":
                return self.handle_chat_streaming(params, conn, session)
            elif method == "shutdown":
                threading.Thread(target=self.shutdown, daemon=True).start()
                return iter([{"type": "result", "message": "Shutting down"}])
            elif method == "disconnect":
                return iter([])
            else:
                return iter([{"type": "error", "message": f"Unknown method {method}"}])
        except Exception as e:
            return iter([{"type": "error", "message": str(e)}])

    def handle_client(self, sock, addr):
        conn = SocketConnection(sock)
        session = self.session_manager.create_session(addr)
        
        try:
            print(f"[{session.client_id}] Connected: {addr}")
            conn.send({"type": "session_created", "session_id": session.client_id})
            
            while not self.shutdown_flag.is_set():
                request = conn.recv()
                if request is None: break
                
                # Process request generator
                for response in self.handle_request(request, conn, session):
                    conn.send(response)
                    
                if request.get("method") == "disconnect": break
        except Exception as e:
            print(f"Client Error: {e}")
        finally:
            self.session_manager.remove_session(session.client_id)
            conn.close()

    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(self.max_client_threads)
        
        print(f"Server started on {self.host}:{self.port} (P2P Tool Calling Enabled)")
        
        try:
            while not self.shutdown_flag.is_set():
                try:
                    conn, addr = self.server_socket.accept()
                    self.client_executor.submit(self.handle_client, conn, addr)
                except socket.timeout: continue
                except OSError: break
        finally:
            self.shutdown()

    def shutdown(self):
        self.shutdown_flag.set()
        if hasattr(self, 'server_socket'): self.server_socket.close()
        self.client_executor.shutdown(wait=False)
        self.process_pool.shutdown(wait=False)
        # self.task_tracker.cancel_all()

# ============================================================================
# Entry Point
# ============================================================================
def main():
    server = UnifiedSocketServer()
    try:
        server.start()
    except KeyboardInterrupt:
        print("Stopping...")
        server.shutdown()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()