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
from multiprocessing import Manager
from multiprocessing.managers import SyncManager
import uuid
import asyncio
from typing import Any, Optional, Dict, List
from contextlib import AsyncExitStack

# Your existing imports
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
# Session Management
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
        """Add a message to chat history"""
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
        """Get chat history (optionally limited)"""
        with self.lock:
            if limit:
                return self.chat_history[-limit:]
            return self.chat_history.copy()
    
    def clear_history(self):
        """Clear chat history"""
        with self.lock:
            self.chat_history.clear()
    
    def to_dict(self) -> Dict:
        """Serialize session info"""
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
    """Manage all client sessions"""
    
    def __init__(self):
        self.sessions: Dict[str, ClientSession] = {}
        self.lock = threading.Lock()
    
    def create_session(self, client_address: tuple) -> ClientSession:
        """Create a new session for a client"""
        client_id = str(uuid.uuid4())
        with self.lock:
            session = ClientSession(client_id, client_address)
            self.sessions[client_id] = session
        return session
    
    def get_session(self, client_id: str) -> Optional[ClientSession]:
        """Get session by ID"""
        with self.lock:
            return self.sessions.get(client_id)
    
    def remove_session(self, client_id: str):
        """Remove a session"""
        with self.lock:
            self.sessions.pop(client_id, None)
    
    def list_sessions(self) -> List[Dict]:
        """List all active sessions"""
        with self.lock:
            return [s.to_dict() for s in self.sessions.values()]
    
    def cleanup_inactive(self, timeout: float = 3600):
        """Remove sessions inactive for more than timeout seconds"""
        current_time = time.time()
        with self.lock:
            to_remove = [
                sid for sid, session in self.sessions.items()
                if current_time - session.last_activity > timeout
            ]
            for sid in to_remove:
                del self.sessions[sid]
        return len(to_remove)


# ============================================================================
# Your Existing Helper Classes (keeping them unchanged)
# ============================================================================

class QueueWriter(io.StringIO):
    def __init__(self, queue, task_id=None):
        super().__init__()
        self.queue = queue
        self.task_id = task_id
    
    def write(self, text):
        if text and text.strip():
            try:
                msg = {
                    "type": "output",
                    "line": text.rstrip('\n')
                }
                if self.task_id is not None:
                    msg["task_id"] = self.task_id
                
                self.queue.put(msg, timeout=0.5) 
            except (queue.Full, Exception):
                pass 
        return len(text)
    
    def flush(self):
        pass


class RedirectOutputToQueue:
    """Context manager to redirect stdout/stderr to queue"""
    
    def __init__(self, output_queue, task_id=None):
        self.output_queue = output_queue
        self.task_id = task_id
        self.old_stdout = None
        self.old_stderr = None
    
    def __enter__(self):
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        
        sys.stdout = QueueWriter(self.output_queue, self.task_id)
        sys.stderr = QueueWriter(self.output_queue, self.task_id)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        return False


class TaskTracker:
    """Unified tracker for both thread and process tasks"""
    
    def __init__(self):
        self.tasks = {}
        self.lock = threading.Lock()
    
    def create_process_task(self, task_type, metadata=None):
        task_id = str(uuid.uuid4())
        with self.lock:
            self.tasks[task_id] = {
                "worker": None,
                "cancel_flag": None,
                "type": task_type,
                "start_time": time.time(),
                "metadata": metadata or {}
            }
        return task_id
    
    def set_worker(self, task_id, worker):
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id]["worker"] = worker
    
    def cancel(self, task_id):
        with self.lock:
            if task_id not in self.tasks:
                return {
                    "success": False,
                    "message": f"Task {task_id} not found"
                }
            
            task = self.tasks[task_id]
            worker = task["worker"]
        
        try:
            if worker and worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
                if worker.is_alive():
                    worker.kill()
                
                return {
                    "success": True,
                    "message": f"Process terminated for task {task_id}",
                    "elapsed": time.time() - task["start_time"]
                }
            else:
                return {
                    "success": False,
                    "message": f"Process for task {task_id} is not running"
                }
        
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to cancel task {task_id}: {str(e)}"
            }
    
    def get(self, task_id):
        with self.lock:
            return self.tasks.get(task_id)
    
    def remove(self, task_id):
        with self.lock:
            self.tasks.pop(task_id, None)
    
    def list_all(self):
        with self.lock:
            result = {}
            for task_id, task in self.tasks.items():
                worker = task["worker"]
                result[task_id] = {
                    "type": task["type"],
                    "elapsed": time.time() - task["start_time"],
                    "is_alive": worker.is_alive() if worker else False,
                    "metadata": task["metadata"]
                }
            return result
    
    def cancel_all(self):
        with self.lock:
            task_ids = list(self.tasks.keys())
        
        results = []
        for task_id in task_ids:
            result = self.cancel(task_id)
            results.append({"task_id": task_id, **result})
        
        return results


class ProcessPool:
    """Simple process pool that limits concurrent processes"""
    
    def __init__(self, max_workers):
        self.max_workers = max_workers
        self.active_processes = []
        self.lock = threading.Lock()
        self.shutdown_flag = False
    
    def _cleanup_finished(self):
        self.active_processes = [p for p in self.active_processes if p.is_alive()]
    
    def get_active_count(self):
        with self.lock:
            self._cleanup_finished()
            return len(self.active_processes)
    
    def has_capacity(self):
        with self.lock:
            self._cleanup_finished()
            return len(self.active_processes) < self.max_workers
    
    def submit(self, target, args):
        if self.shutdown_flag:
            return None
        
        with self.lock:
            self._cleanup_finished()
            if len(self.active_processes) < self.max_workers:
                process = multiprocessing.Process(target=target, args=args)
                process.daemon = True
                self.active_processes.append(process)
                process.start()
                return process
            return None
    
    def shutdown(self, wait=True, timeout=5):
        self.shutdown_flag = True
        
        with self.lock:
            processes = list(self.active_processes)
        
        if wait:
            for process in processes:
                if process.is_alive():
                    process.join(timeout=1)
        
        with self.lock:
            for process in self.active_processes:
                if process.is_alive():
                    try:
                        process.terminate()
                    except:
                        pass
            
            for process in self.active_processes:
                if process.is_alive():
                    try:
                        process.kill()
                    except:
                        pass
            
            self.active_processes.clear()


# ============================================================================
# Enhanced Socket Connection with Streaming
# ============================================================================

class SocketConnection:
    """Wrapper for socket to handle JSON message sending/receiving"""
    
    def __init__(self, sock):
        self.sock = sock
        self.lock = threading.Lock()
    
    def send(self, obj):
        """Send a JSON-serializable object"""
        with self.lock:
            try:
                data = json.dumps(obj).encode('utf-8')
                msg_length = struct.pack('>I', len(data))
                self.sock.sendall(msg_length + data)
            except Exception as e:
                raise Exception(f"Failed to send message: {e}")
    
    def recv(self):
        """Receive a JSON object"""
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
            raise Exception(f"Failed to receive message: {e}")
    
    def _recvall(self, n):
        """Helper to receive exactly n bytes"""
        data = bytearray()
        while len(data) < n:
            packet = self.sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return bytes(data)
    
    def close(self):
        """Close the socket"""
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except:
            pass
        try:
            self.sock.close()
        except:
            pass


# ============================================================================
# LLM Handler with Tool Calling
# ============================================================================

class LLMHandler:
    """Handle LLM interactions with tool calling capabilities"""
    
    def __init__(self, api_key: str, model: str = "anthropic/claude-3.5-sonnet"):
        self.llm_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = model
    
    def define_client_tools(self) -> List[Dict]:
        """
        Define tools that will be executed on the CLIENT side
        These are P2P tools where server calls client functions
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_dataset_info",
                    "description": "Get information about available datasets on the client",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "dataset_name": {
                                "type": "string",
                                "description": "Name of the dataset (optional, if empty returns all)"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "show_data",
                    "description": "Display data or visualization on the client UI",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "data_type": {
                                "type": "string",
                                "enum": ["table", "chart", "text"],
                                "description": "Type of data to display"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to display"
                            }
                        },
                        "required": ["data_type", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_pipeline",
                    "description": "Execute a data processing pipeline on the server",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pipeline_json": {
                                "type": "string",
                                "description": "Path to pipeline JSON file"
                            },
                            "mode": {
                                "type": "string",
                                "enum": ["complete", "partial"],
                                "description": "Execution mode"
                            }
                        },
                        "required": ["pipeline_json"]
                    }
                }
            }
        ]
    
    async def process_query_streaming(
        self, 
        messages: List[Dict],
        conn: SocketConnection,
        session: ClientSession
    ) -> str:
        """
        Process a query with streaming responses and tool calling
        
        Args:
            messages: Chat history
            conn: Socket connection to send streaming updates
            session: Client session for tracking
        
        Yields streaming tokens and handles tool calls
        """
        available_tools = self.define_client_tools()
        
        while True:
            try:
                # Call LLM with streaming
                response = await self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=available_tools,
                    stream=True
                )
                
                # Collect streaming response
                full_content = ""
                tool_calls_data = {}
                current_tool_call = None
                
                async for chunk in response:
                    delta = chunk.choices[0].delta
                    
                    # Stream text content
                    if delta.content:
                        full_content += delta.content
                        conn.send({
                            "type": "llm_stream",
                            "content": delta.content
                        })
                    
                    # Collect tool calls
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index
                            if idx not in tool_calls_data:
                                tool_calls_data[idx] = {
                                    "id": tc.id or "",
                                    "function": {"name": "", "arguments": ""}
                                }
                            
                            if tc.id:
                                tool_calls_data[idx]["id"] = tc.id
                            if tc.function:
                                if tc.function.name:
                                    tool_calls_data[idx]["function"]["name"] = tc.function.name
                                if tc.function.arguments:
                                    tool_calls_data[idx]["function"]["arguments"] += tc.function.arguments
                
                # Check if we have tool calls
                if tool_calls_data:
                    # Reconstruct tool calls
                    tool_calls = []
                    for idx in sorted(tool_calls_data.keys()):
                        tc_data = tool_calls_data[idx]
                        tool_calls.append({
                            "id": tc_data["id"],
                            "type": "function",
                            "function": {
                                "name": tc_data["function"]["name"],
                                "arguments": tc_data["function"]["arguments"]
                            }
                        })
                    
                    # Add assistant message with tool calls
                    messages.append({
                        "role": "assistant",
                        "content": full_content if full_content else None,
                        "tool_calls": tool_calls
                    })
                    
                    # Process each tool call
                    for tool_call in tool_calls:
                        func_name = tool_call["function"]["name"]
                        func_args = json.loads(tool_call["function"]["arguments"])
                        
                        # Send tool call request to client
                        conn.send({
                            "type": "tool_call_request",
                            "tool_call_id": tool_call["id"],
                            "function": func_name,
                            "arguments": func_args
                        })
                        
                        # Wait for tool result from client
                        tool_result = conn.recv()
                        
                        if tool_result and tool_result.get("type") == "tool_call_result":
                            result_content = tool_result.get("result", "No result")
                        else:
                            result_content = "Tool execution failed"
                        
                        # Add tool result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": result_content
                        })
                    
                    # Continue loop to get final response
                    continue
                
                else:
                    # No tool calls, return final response
                    return full_content
                    
            except Exception as e:
                error_msg = f"LLM Error: {str(e)}"
                conn.send({
                    "type": "error",
                    "message": error_msg
                })
                return error_msg


# ============================================================================
# Main Unified Socket Server
# ============================================================================

class UnifiedSocketServer:
    """
    Unified server handling:
    1. CPU-bound pipeline/plugin execution (blocking threads + processes)
    2. LLM chat with tool calling (async)
    3. P2P architecture (server can call client tools)
    4. Session management per client
    """
    
    def __init__(self, host=HOST, port=PORT, max_client_threads=None, max_process_workers=None):
        self.host = host
        self.port = port
        self.max_client_threads = max_client_threads or MAX_CLIENT_THREADS
        self.max_process_workers = max_process_workers or MAX_PROCESS_WORKERS
        
        # Thread pool for handling client connections
        self.client_executor = ThreadPoolExecutor(max_workers=self.max_client_threads)
        
        # Process pool for CPU-bound tasks
        self.process_pool = ProcessPool(max_workers=self.max_process_workers)
        
        # Task tracker
        self.task_tracker = TaskTracker()
        
        # Session manager
        self.session_manager = SessionManager()
        
        # LLM handler
        api_key = os.getenv("OPENROUTER_API_KEY", "sk-or-v1-your-key-here")
        self.llm_handler = LLMHandler(api_key)
        
        # Track active tasks
        self.active_tasks = 0
        self.lock = threading.Lock()
        
        # Server socket
        self.server_socket = None
        
        # Shared memory manager
        self.manager = SyncManager()
        self.manager.start()
        
        # Shutdown flag
        self.shutdown_flag = threading.Event()
        
        # Register cleanup
        atexit.register(self.shutdown)
    
    # ========================================================================
    # Your existing pipeline/plugin methods (keeping them)
    # ========================================================================
    
    def execute_single_pipeline(self, params):
        """Execute a single pipeline (your existing code)"""
        # ... (keep your implementation)
        pass
    
    def execute_batch_parallel(self, pipeline_json_list, params):
        """Execute multiple pipelines in parallel (your existing code)"""
        # ... (keep your implementation)
        pass
    
    def execute_plugin(self, plugin_name, args, cache_dir):
        """Execute a plugin (your existing code)"""
        # ... (keep your implementation)
        pass
    
    # ========================================================================
    # NEW: LLM Chat Methods
    # ========================================================================
    
    def handle_chat_streaming(self, params, conn: SocketConnection, session: ClientSession):
        """
        Handle streaming chat with LLM
        
        Args:
            params: {
                "message": str,  # User message
                "session_id": str  # Optional, for continuing conversation
            }
            conn: Socket connection
            session: Client session
        """
        try:
            user_message = params.get("message", "")
            
            if not user_message:
                yield {
                    "type": "error",
                    "message": "No message provided"
                }
                return
            
            # Add user message to session
            session.add_message("user", user_message)
            
            # Get chat history for LLM context
            messages = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in session.get_history()
            ]
            
            # Create async event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Run async LLM processing
                assistant_response = loop.run_until_complete(
                    self.llm_handler.process_query_streaming(messages, conn, session)
                )
                
                # Add assistant response to session
                session.add_message("assistant", assistant_response)
                
                # Send completion
                yield {
                    "type": "llm_complete",
                    "content": assistant_response
                }
                
            finally:
                loop.close()
                
        except Exception as e:
            yield {
                "type": "error",
                "message": f"Chat error: {str(e)}",
                "traceback": traceback.format_exc()
            }
    
    # ========================================================================
    # Request Router
    # ========================================================================
    
    def handle_request(self, request, conn: SocketConnection, session: ClientSession):
        """
        Route requests to appropriate handlers
        
        Supported methods:
        - run_pipeline: Execute single pipeline
        - run_pipeline_batch: Execute multiple pipelines
        - run_plugin: Execute plugin
        - chat: Chat with LLM (streaming, with tool calling)
        - get_session_info: Get session information
        - clear_session: Clear chat history
        - list_sessions: List all active sessions
        - terminate_task: Cancel a running task
        - list_tasks: List all tasks
        - ping: Health check
        - shutdown: Shutdown server
        """
        try:
            method = request.get("method")
            params = request.get("params", {})
            
            # ============================================================
            # Pipeline/Plugin Methods (your existing code)
            # ============================================================
            
            if method == "run_pipeline":
                return self.execute_single_pipeline(params)
            
            elif method == "run_pipeline_batch":
                pipeline_json_list = params.get("pipeline_json_list", [])
                if not pipeline_json_list:
                    return iter([{
                        "type": "error",
                        "message": "No pipelines provided"
                    }])
                return self.execute_batch_parallel(pipeline_json_list, params)
            
            elif method == "run_plugin":
                plugin_name = params.get("plugin_name")
                args_list = params.get("args", "").split(" ")
                cache_dir = params.get("cache_dir")
                return self.execute_plugin(plugin_name, args_list, cache_dir)
            
            # ============================================================
            # NEW: LLM Chat Methods
            # ============================================================
            
            elif method == "chat":
                """
                Chat with LLM (streaming)
                
                params: {
                    "message": "Your question here"
                }
                """
                return self.handle_chat_streaming(params, conn, session)
            
            elif method == "get_session_info":
                """Get current session information"""
                return iter([{
                    "type": "result",
                    "data": session.to_dict()
                }])
            
            elif method == "clear_session":
                """Clear chat history for this session"""
                session.clear_history()
                return iter([{
                    "type": "result",
                    "message": "Session history cleared"
                }])
            
            elif method == "list_sessions":
                """List all active sessions"""
                sessions = self.session_manager.list_sessions()
                return iter([{
                    "type": "result",
                    "data": sessions
                }])
            
            # ============================================================
            # Task Management (your existing code)
            # ============================================================
            
            elif method == "terminate_task":
                task_id = params.get("task_id")
                if not task_id:
                    return iter([{"type": "error", "message": "task_id required"}])
                
                result = self.task_tracker.cancel(task_id)
                return iter([{
                    "type": "result",
                    "data": result
                }])
            
            elif method == "list_tasks":
                tasks = self.task_tracker.list_all()
                return iter([{
                    "type": "result",
                    "data": tasks
                }])
            
            elif method == "ping":
                return iter([{"type": "result", "message": "pong"}])
            
            elif method == "get_server_stats":
                with self.lock:
                    tasks = self.task_tracker.list_all()
                    sessions = self.session_manager.list_sessions()
                    return iter([{
                        "type": "result",
                        "data": {
                            "max_client_threads": self.max_client_threads,
                            "max_process_workers": self.max_process_workers,
                            "active_tasks": self.active_tasks,
                            "active_processes": self.process_pool.get_active_count(),
                            "tracked_tasks": len(tasks),
                            "active_sessions": len(sessions)
                        }
                    }])
            
            elif method == "shutdown":
                threading.Thread(target=self.shutdown, daemon=True).start()
                return iter([{"type": "result", "message": "Server shutting down"}])
            
            else:
                return iter([{"type": "error", "message": f"Unknown method: {method}"}])
                
        except Exception as e:
            return iter([{
                "type": "error",
                "message": str(e),
                "traceback": traceback.format_exc()
            }])
    
    # ========================================================================
    # Client Handler
    # ========================================================================
    
    def handle_client(self, sock, addr):
        """Handle a single client connection"""
        conn = SocketConnection(sock)
        
        # Create session for this client
        session = self.session_manager.create_session(addr)
        
        try:
            print(f"[{session.client_id}] Client connected from {addr}")
            
            # Send session info to client
            conn.send({
                "type": "session_created",
                "session_id": session.client_id
            })
            
            # Handle requests from this client
            while not self.shutdown_flag.is_set():
                # Receive request
                request = conn.recv()
                
                if request is None:
                    break
                
                # Process request and stream responses
                for response in self.handle_request(request, conn, session):
                    conn.send(response)
                
                # Check if client requested disconnect
                if request.get("method") == "disconnect":
                    break
            
        except Exception as e:
            if not self.shutdown_flag.is_set():
                print(f"[{session.client_id}] Error: {e}")
                try:
                    conn.send({
                        "type": "error",
                        "message": f"Server error: {str(e)}"
                    })
                except:
                    pass
        finally:
            print(f"[{session.client_id}] Client disconnected")
            self.session_manager.remove_session(session.client_id)
            conn.close()
    
    # ========================================================================
    # Server Lifecycle
    # ========================================================================
    
    def start(self):
        """Start the server"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(self.max_client_threads)
        self.server_socket.settimeout(1.0)
        
        print("=" * 70)
        print(f"Unified Socket Server Started")
        print(f"Host: {self.host}:{self.port}")
        print(f"Max client threads: {self.max_client_threads}")
        print(f"Max process workers: {self.max_process_workers}")
        print(f"Features: Pipeline Execution + LLM Chat + Tool Calling")
        print("=" * 70)
        print("\nWaiting for connections... (Press Ctrl+C to stop)\n")
        
        try:
            while not self.shutdown_flag.is_set():
                try:
                    conn, addr = self.server_socket.accept()
                    
                    if self.shutdown_flag.is_set():
                        conn.close()
                        break
                    
                    # Submit to thread pool
                    self.client_executor.submit(self.handle_client, conn, addr)
                    
                except socket.timeout:
                    continue
                except KeyboardInterrupt:
                    print("\n\nReceived interrupt signal...")
                    break
                except OSError:
                    if not self.shutdown_flag.is_set():
                        print("Error accepting connection")
                    break
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Shutdown the server"""
        if self.shutdown_flag.is_set():
            return
        
        print("\nStarting shutdown sequence...")
        self.shutdown_flag.set()
        
        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        # Cancel all tasks
        print("Cancelling all tasks...")
        self.task_tracker.cancel_all()
        
        # Shutdown pools
        print("Shutting down thread pool...")
        try:
            self.client_executor.shutdown(wait=True, cancel_futures=True)
        except:
            self.client_executor.shutdown(wait=False)
        
        print("Shutting down process pool...")
        self.process_pool.shutdown(wait=True, timeout=3)
        
        print("Shutting down manager...")
        try:
            self.manager.shutdown()
        except:
            pass
        
        print("Server shutdown complete.")


# ============================================================================
# Main Entry Point
# ============================================================================

def main(host=HOST, port=PORT, max_client_threads=None, max_process_workers=None):
    """Start the unified server"""
    server = UnifiedSocketServer(host, port, max_client_threads, max_process_workers)
    
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}")
        server.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        server.start()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")
    finally:
        server.shutdown()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Unified Socket Server (Pipeline + LLM)")
    parser.add_argument('--host', type=str, default=HOST, help=f"Host (default: {HOST})")
    parser.add_argument('--port', type=int, default=PORT, help=f"Port (default: {PORT})")
    parser.add_argument('--max-client-threads', type=int, default=None)
    parser.add_argument('--max-process-workers', type=int, default=None)
    
    args = parser.parse_args()
    
    main(host=args.host, port=args.port,
         max_client_threads=args.max_client_threads,
         max_process_workers=args.max_process_workers)