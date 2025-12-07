"""
Unified Socket Server combining pipeline execution and LLM chat functionality
"""
import sys
import os
import json
import socket
import struct
import signal
import atexit
import threading
import traceback
import multiprocessing
import argparse
import time
from typing import Dict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.managers import SyncManager

from config import HOST, PORT, MAX_CLIENT_THREADS, MAX_PROCESS_WORKERS, OPENROUTER_API_KEY
from process_utils import (
    subprocess_cleanup_monitor,
    ProcessPool,
    terminate_process_tree
)
from session_manager import SessionManager, SessionType, TaskStatus
from pipeline_handler import PipelineHandler
from llm_handler import LLMHandler

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
                print(f"Socket Send Error: {e}")
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
            print(f"Socket Recv Error: {e}")
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


class UnifiedSocketServer:
    """
    Unified Socket Server that handles both:
    - Pipeline/Plugin execution requests (stateless or session-based)
    - LLM Chat requests (stateful, session-based)
    """

    def __init__(self, host=HOST, port=PORT, max_client_threads=None, max_process_workers=None):
        self.host = host
        self.port = port
        self.max_client_threads = max_client_threads or MAX_CLIENT_THREADS
        self.max_process_workers = max_process_workers or MAX_PROCESS_WORKERS

        # Thread pool for handling client connections
        self.client_executor = ThreadPoolExecutor(max_workers=self.max_client_threads)

        # Custom process pool for executing tasks
        self.process_pool = ProcessPool(max_workers=self.max_process_workers)

        # Track active stateful connections for graceful shutdown
        self.active_connections: Dict[str, SocketConnection] = {}  # session_id -> conn
        self.connections_lock = threading.Lock()

        # Socket server
        self.server_socket = None

        # Shared memory manager
        self.manager = SyncManager()
        self.manager.start(initializer=subprocess_cleanup_monitor, initargs=(os.getpid(),))

        # Shutdown flag
        self.shutdown_flag = threading.Event()

        # Unified Session manager (handles both chat and pipeline sessions)
        self.session_manager = SessionManager(
            session_timeout=3600,  # 1 hour
            cleanup_interval=300,  # 5 minutes
            terminate_process_func=terminate_process_tree
        )

        # LLM Handler
        self.llm_handler = LLMHandler(OPENROUTER_API_KEY)

        # Pipeline Handler
        self.pipeline_handler = PipelineHandler(
            self.process_pool,
            self.session_manager,
            self.manager,
            self.shutdown_flag
        )

        # Register cleanup
        atexit.register(self.shutdown)

    def handle_pipeline_request(self, request, session):
        """Handle pipeline-related requests"""
        method = request.get("method")
        params = request.get("params", {})

        if method == "run_pipeline_batch":
            pipeline_json_list = params.get("pipeline_json_list", [])

            if not pipeline_json_list:
                return iter([{
                    "type": "error",
                    "status": "error",
                    "session_id": session.session_id,
                    "message": "No pipelines provided in pipeline_json_list"
                }])

            return self.pipeline_handler.execute_batch_parallel(session, params)

        elif method == "run_plugin":
            plugin_name = params.get("plugin_name", None)
            args_list = params.get("args", "").split(" ")
            cache_dir = params.get("cache_dir", None)

            return self.pipeline_handler.execute_plugin(session, plugin_name, args_list, cache_dir)

        elif method == "plugin_validation":
            return self.pipeline_handler.handle_plugin_validation(params)

        elif method == "init_plugin":
            return self.pipeline_handler.handle_init_plugin(params)

        elif method == "reset_id":
            return self.pipeline_handler.handle_reset_id(params)

        elif method == "validate":
            return self.pipeline_handler.handle_validate(params)

        elif method == "terminate_task":
            task_id = params.get("task_id")

            if not task_id:
                return iter([{
                    "type": "error",
                    "status": "error",
                    "message": "task_id is required"
                }])

            # First try to find in current session
            result = session.terminate_task(task_id)
            if not result["success"] and "not found" in result["message"]:
                # Try global search
                result = self.session_manager.terminate_task(task_id)

            return iter([{
                "type": "result",
                "status": "success" if result["success"] else "error",
                "message": result["message"],
                "data": result
            }])

        elif method == "list_tasks":
            # Option to list all tasks or just session tasks
            scope = params.get("scope", "session")  # "session" or "all"
            
            if scope == "all":
                tasks = self.session_manager.list_all_tasks()
            else:
                tasks = session.list_tasks()

            return iter([{
                "type": "result",
                "status": "success",
                "session_id": session.session_id,
                "message": f"Found {len(tasks)} tasks",
                "data": tasks
            }])

        elif method == "get_task_status":
            task_id = params.get("task_id")

            if not task_id:
                return iter([{
                    "type": "error",
                    "status": "error",
                    "message": "task_id is required"
                }])

            # First check current session
            task_info = session.get_task(task_id)
            if task_info:
                return iter([{
                    "type": "result",
                    "status": "success",
                    "session_id": session.session_id,
                    "data": task_info.to_dict()
                }])
            
            # Try global search
            task_dict = self.session_manager.get_task(task_id)
            if task_dict:
                return iter([{
                    "type": "result",
                    "status": "success",
                    "data": task_dict
                }])

            return iter([{
                "type": "error",
                "status": "error",
                "message": f"Task {task_id} not found"
            }])

        elif method == "list_sessions":
            sessions = self.session_manager.list_sessions()
            return iter([{
                "type": "result",
                "status": "success",
                "message": f"Found {len(sessions)} sessions",
                "data": sessions
            }])

        elif method == "ping":
            return iter([{
                "type": "result",
                "status": "success",
                "session_id": session.session_id,
                "message": "pong"
            }])

        elif method == "get_server_stats":
            stats = self.session_manager.get_stats()
            stats.update({
                "max_client_threads": self.max_client_threads,
                "max_process_workers": self.max_process_workers,
                "active_processes": self.process_pool.get_active_count()
            })
            
            return iter([{
                "type": "result",
                "status": "success",
                "data": stats
            }])

        elif method == "shutdown":
            threading.Thread(target=self.shutdown, daemon=True).start()
            return iter([{
                "type": "result",
                "status": "success",
                "message": "Server is shutting down"
            }])

        else:
            return iter([{
                "type": "error",
                "status": "error",
                "message": f"Unknown method: {method}"
            }])

    def handle_client(self, sock, addr):
        """
        Unified client handler that determines connection type from first message
        """
        conn = SocketConnection(sock)
        session = None
        
        try:
            # Receive first message to determine connection type
            request = conn.recv()

            if request is None:
                return

            method = request.get("method", "")
            connection_type = request.get("connection_type", "stateless")
            provided_session_id = request.get("session_id")

            # Determine session type
            if connection_type == "chat" or method in ["chat", "connect_chat"]:
                session_type = SessionType.CHAT
            elif connection_type == "hybrid":
                session_type = SessionType.HYBRID
            else:
                session_type = SessionType.PIPELINE

            # Get or create session
            session = self.session_manager.get_or_create_session(
                provided_session_id, addr, session_type
            )

            print(f"[Server] Client connected: {addr} (Session: {session.session_id}, Type: {session_type.value})")

            # Send session confirmation
            conn.send({
                "type": "session_created",
                "session_id": session.session_id,
                "session_type": session_type.value
            })

            # Handle based on connection type
            if session_type == SessionType.CHAT or session_type == SessionType.HYBRID:
                # Stateful connection - keep connection open for multiple requests
                self._handle_stateful_connection(conn, session, request)
            else:
                # Stateless connection - single request/response
                self._handle_stateless_request(conn, session, request)

        except Exception as e:
            if not self.shutdown_flag.is_set():
                print(f"Client Error ({addr}): {e}")
                traceback.print_exc()
                error_response = {
                    "type": "error",
                    "status": "error",
                    "message": f"Server error: {str(e)}",
                    "traceback": traceback.format_exc()
                }
                if session:
                    error_response["session_id"] = session.session_id
                try:
                    conn.send(error_response)
                except:
                    pass
        finally:
            conn.close()
            if session:
                # For stateless connections, remove session after request
                # For stateful, keep session alive for reconnection
                if session.session_type == SessionType.PIPELINE:
                    # Only remove if no running tasks
                    if not session.get_running_tasks():
                        self.session_manager.remove_session(session.session_id)
                print(f"[Server] Client disconnected: {addr} (Session: {session.session_id})")

    def _notify_clients_shutdown(self):
        """Notify all connected stateful clients that server is shutting down"""
        with self.connections_lock:
            connections = list(self.active_connections.items())
        
        for session_id, conn in connections:
            try:
                conn.send({
                    "type": "server_shutdown",
                    "session_id": session_id,
                    "message": "Server is shutting down"
                })
            except Exception as e:
                print(f"Failed to notify client {session_id}: {e}")

    def _handle_stateful_connection(self, conn, session, first_request):
        """Handle a stateful (chat/hybrid) connection"""
        # Register this connection for server-initiated shutdown
        with self.connections_lock:
            self.active_connections[session.session_id] = conn

        try:
            # Process first request if it's a chat message
            method = first_request.get("method")
            if method == "chat":
                msg = first_request.get("params", {}).get("message")
                if msg:
                    session.add_message("user", msg)
                    self.llm_handler.run_chat_sync(session, conn)
            elif method not in ["connect_chat", None]:
                # It's a pipeline request in hybrid mode
                for response in self.handle_pipeline_request(first_request, session):
                    conn.send(response)

            # Continue with request loop
            while not self.shutdown_flag.is_set():
                req = conn.recv()
                if not req:
                    break

                method = req.get("method")
                params = req.get("params", {})

                if method == "chat":
                    msg = params.get("message")
                    if msg:
                        session.add_message("user", msg)
                        self.llm_handler.run_chat_sync(session, conn)

                elif method == "clear_history":
                    session.clear_history()
                    conn.send({
                        "type": "result",
                        "session_id": session.session_id,
                        "message": "History cleared"
                    })

                elif method == "get_history":
                    limit = params.get("limit")
                    conn.send({
                        "type": "result",
                        "session_id": session.session_id,
                        "history": session.get_history(limit)
                    })

                elif method == "get_session_info":
                    conn.send({
                        "type": "result",
                        "session_id": session.session_id,
                        "data": session.to_dict()
                    })

                elif method == "disconnect":
                    # Client requested disconnect - send acknowledgment
                    conn.send({
                        "type": "disconnected",
                        "session_id": session.session_id,
                        "message": "Disconnected successfully"
                    })
                    break

                elif method == "disconnect_ack":
                    # Client acknowledged our shutdown request
                    break

                else:
                    # Handle as pipeline request (for hybrid mode)
                    for response in self.handle_pipeline_request(req, session):
                        conn.send(response)
        finally:
            # Unregister connection
            with self.connections_lock:
                self.active_connections.pop(session.session_id, None)

    def _handle_stateless_request(self, conn, session, request):
        """Handle a stateless pipeline request"""
        for response in self.handle_pipeline_request(request, session):
            conn.send(response)

    def start(self):
        """Start the Socket server"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(self.max_client_threads)
        self.server_socket.settimeout(1.0)

        print("=" * 60)
        print(f"Unified Socket Server started on {self.host}:{self.port}")
        print("=" * 60)
        print(f"Transport: TCP Socket")
        print(f"Max client threads: {self.max_client_threads}")
        print(f"Max process workers: {self.max_process_workers}")
        print(f"Session timeout: {self.session_manager.session_timeout}s")
        print(f"Supported modes: Pipeline (stateless), Chat (stateful), Hybrid")
        print(f"Waiting for connections... (Press Ctrl+C to stop)\n")

        try:
            while not self.shutdown_flag.is_set():
                try:
                    conn, addr = self.server_socket.accept()

                    if self.shutdown_flag.is_set():
                        conn.close()
                        break

                    print(f"Connection accepted from {addr}")

                    self.client_executor.submit(self.handle_client, conn, addr)

                except socket.timeout:
                    continue
                except KeyboardInterrupt:
                    print("\n\nReceived interrupt signal...")
                    break
                except OSError as e:
                    if not self.shutdown_flag.is_set():
                        print(f"Error accepting connection: {e}")
                    break
                except Exception as e:
                    if not self.shutdown_flag.is_set():
                        print(f"Error accepting connection: {e}")
                    break
        finally:
            print("Shutting down server...")
            self.shutdown()

    def shutdown(self):
        """Shutdown the server"""
        if self.shutdown_flag.is_set():
            return

        print("Starting shutdown sequence...")
        self.shutdown_flag.set()

        # Notify all connected clients FIRST
        print("Notifying connected clients...")
        self._notify_clients_shutdown()

        # Give clients a moment to disconnect gracefully
        time.sleep(0.5)

        # Stop session cleanup thread
        print("Stopping session cleanup...")
        self.session_manager.stop_cleanup()

        # Terminate all tasks in all sessions
        print("Terminating all tasks...")
        self.session_manager.terminate_all()

        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
            self.server_socket = None

        # Shutdown thread pool
        print("Waiting for client threads to finish...")
        try:
            self.client_executor.shutdown(wait=True, cancel_futures=True)
        except TypeError:
            self.client_executor.shutdown(wait=True)
        except:
            self.client_executor.shutdown(wait=False)

        # Shutdown process pool
        print("Terminating worker processes...")
        self.process_pool.shutdown(wait=True, timeout=3)

        # Shutdown manager
        print("Shutting down shared memory manager...")
        try:
            self.manager.shutdown()
        except:
            pass

        print("Server shutdown complete.")


def main(host=HOST, port=PORT, max_client_threads=None, max_process_workers=None):
    """
    Start the Unified Socket server

    Args:
        host: Host to bind to (default: 127.0.0.1)
        port: Port to listen on (default: 65432)
        max_client_threads: Maximum number of concurrent client connections
        max_process_workers: Maximum number of concurrent task processes
    """
    server = UnifiedSocketServer(host, port, max_client_threads, max_process_workers)

    # Setup signal handlers
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
    # Required for multiprocessing on Windows
    multiprocessing.freeze_support()

    # Force spawn method to ensure clean process state
    multiprocessing.set_start_method('spawn', force=True)

    # Parse command-line args
    parser = argparse.ArgumentParser(description="Start unified socket server")
    parser.add_argument('--host', type=str, default=HOST,
                        help=f"Host to bind to (default: {HOST})")
    parser.add_argument('--port', type=int, default=PORT,
                        help=f"Port to listen on (default: {PORT})")
    parser.add_argument('--max-client-threads', type=int, default=None,
                        help="Maximum client threads (optional)")
    parser.add_argument('--max-process-workers', type=int, default=None,
                        help="Maximum process workers (optional)")

    args = parser.parse_args()

    main(
        host=args.host,
        port=args.port,
        max_client_threads=args.max_client_threads,
        max_process_workers=args.max_process_workers
    )