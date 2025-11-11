import socket
import struct
import json
import threading
import multiprocessing
import time

HOST = '127.0.0.1'
PORT = 65432
WORKER_PORT = 65433


class SocketConnection:
    """Wrapper for socket to handle JSON message sending/receiving"""
    
    def __init__(self, sock):
        self.sock = sock
        self.lock = threading.Lock()
    
    def send(self, obj):
        """Send a JSON-serializable object"""
        with self.lock:
            data = json.dumps(obj).encode('utf-8')
            msg_length = struct.pack('>I', len(data))
            self.sock.sendall(msg_length + data)
    
    def recv(self):
        """Receive a JSON object"""
        raw_msglen = self._recvall(4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('>I', raw_msglen)[0]
        
        data = self._recvall(msglen)
        if not data:
            return None
        
        return json.loads(data.decode('utf-8'))
    
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


def worker_process(task_data, host, port, conn_id):
    """
    Worker function that runs in a separate process.
    It connects BACK to the parent via a callback socket.
    """
    print(f"[Worker Process {conn_id}] Started")
    conn = None
    
    try:
        # Connect back to parent's worker server
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        conn = SocketConnection(sock)
        
        # Identify ourselves
        print(f"[Worker Process {conn_id}] Connecting back to {host}:{port}")
        conn.send({"type": "worker_connect", "conn_id": conn_id})
        
        # Simulate some work with progress updates
        task_name = task_data.get("task_name", "unknown")
        iterations = task_data.get("iterations", 5)
        
        for i in range(iterations):
            time.sleep(0.5)  # Simulate work
            
            # Send progress update
            conn.send({
                "type": "progress",
                "message": f"Processing {task_name}: step {i+1}/{iterations}"
            })
            print(f"[Worker Process {conn_id}] Sent progress {i+1}/{iterations}")
        
        # Send final result
        result = f"Task '{task_name}' completed with {iterations} iterations"
        conn.send({
            "type": "result",
            "status": "success",
            "data": result
        })
        print(f"[Worker Process {conn_id}] Sent final result")
        
    except Exception as e:
        print(f"[Worker Process {conn_id}] Error: {e}")
        if conn:
            conn.send({
                "type": "error",
                "status": "error",
                "message": str(e)
            })
    finally:
        if conn:
            conn.close()
        print(f"[Worker Process {conn_id}] Finished")


class SimpleServer:
    """Simple server demonstrating the socket architecture"""
    
    def __init__(self, host=HOST, port=PORT):
        self.host = host
        self.port = port
        self.worker_port = port + 1  # WORKER_PORT
        
        # Main server socket (accepts client requests)
        self.server_socket = None
        
        # Worker callback server (receives messages from worker processes)
        self.worker_server_socket = None
        self.worker_connections = {}  # conn_id -> list of messages
        self.worker_server_thread = None
        
        self.shutdown_flag = threading.Event()
    
    def _start_worker_server(self):
        """Start the callback server for worker processes"""
        self.worker_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.worker_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.worker_server_socket.bind((self.host, self.worker_port))
        self.worker_server_socket.listen(5)
        self.worker_server_socket.settimeout(1.0)
        
        print(f"[Worker Server] Started on {self.host}:{self.worker_port}")
        
        def accept_workers():
            """Accept worker connections in a loop"""
            while not self.shutdown_flag.is_set():
                try:
                    conn, addr = self.worker_server_socket.accept()
                    print(f"[Worker Server] Accepted connection from {addr}")
                    # Handle each worker in a separate thread
                    threading.Thread(
                        target=self._handle_worker_connection, 
                        args=(conn,), 
                        daemon=True
                    ).start()
                except socket.timeout:
                    continue
                except Exception as e:
                    if not self.shutdown_flag.is_set():
                        print(f"[Worker Server] Error: {e}")
                    break
        
        self.worker_server_thread = threading.Thread(target=accept_workers, daemon=True)
        self.worker_server_thread.start()
    
    def _handle_worker_connection(self, sock):
        """Handle messages from a single worker process"""
        conn = SocketConnection(sock)
        conn_id = None
        
        try:
            # First message: worker identification
            msg = conn.recv()
            if msg and msg.get("type") == "worker_connect":
                conn_id = msg.get("conn_id")
                print(f"[Worker Server] Worker {conn_id} connected")
                
                # Create message queue for this worker
                if conn_id not in self.worker_connections:
                    self.worker_connections[conn_id] = []
                
                # Receive messages from worker
                while not self.shutdown_flag.is_set():
                    msg = conn.recv()
                    if msg is None:
                        break
                    
                    print(f"[Worker Server] Received from {conn_id}: {msg.get('type')}")
                    
                    # Store message for the main handler thread
                    self.worker_connections[conn_id].append(msg)
                    
                    # Stop if final message
                    if msg.get("type") in ["result", "error"]:
                        break
                        
        except Exception as e:
            print(f"[Worker Server] Error handling {conn_id}: {e}")
        finally:
            conn.close()
            print(f"[Worker Server] Worker {conn_id} disconnected")
    
    def handle_client(self, sock):
        """Handle a client request (runs in main server thread)"""
        conn = SocketConnection(sock)
        
        try:
            # Receive the request
            request = conn.recv()
            if request is None:
                return
            
            print(f"[Client Handler] Received request: {request}")
            
            task_data = request.get("task_data", {})
            
            # Generate unique connection ID for this task
            conn_id = f"worker_{threading.get_ident()}_{id(request)}"
            
            # Initialize message queue for this worker
            self.worker_connections[conn_id] = []
            
            # Start worker process
            print(f"[Client Handler] Starting worker process {conn_id}")
            process = multiprocessing.Process(
                target=worker_process,
                args=(task_data, self.host, self.worker_port, conn_id)
            )
            process.start()
            
            # Stream messages from worker back to client
            final_received = False
            while not self.shutdown_flag.is_set() and not final_received:
                # Check for messages from worker
                if conn_id in self.worker_connections and self.worker_connections[conn_id]:
                    messages = self.worker_connections[conn_id][:]
                    self.worker_connections[conn_id].clear()
                    
                    for message in messages:
                        print(f"[Client Handler] Forwarding to client: {message.get('type')}")
                        conn.send(message)  # Forward to client
                        
                        if message.get("type") in ["result", "error"]:
                            final_received = True
                            break
                
                # Check if process is still alive
                if not process.is_alive():
                    time.sleep(0.5)  # Wait for final messages
                    
                    # Get any remaining messages
                    if conn_id in self.worker_connections and self.worker_connections[conn_id]:
                        messages = self.worker_connections[conn_id][:]
                        self.worker_connections[conn_id].clear()
                        
                        for message in messages:
                            conn.send(message)
                            if message.get("type") in ["result", "error"]:
                                final_received = True
                                break
                    
                    if not final_received:
                        conn.send({
                            "type": "error",
                            "status": "error",
                            "message": "Worker terminated unexpectedly"
                        })
                    break
                
                time.sleep(0.1)  # Avoid busy waiting
            
            # Wait for process to finish
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
            
        except Exception as e:
            print(f"[Client Handler] Error: {e}")
            conn.send({"type": "error", "status": "error", "message": str(e)})
        finally:
            # Cleanup
            if conn_id in self.worker_connections:
                del self.worker_connections[conn_id]
            conn.close()
    
    def start(self):
        """Start the server"""
        # Start worker callback server first
        self._start_worker_server()
        
        # Start main server
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.server_socket.settimeout(1.0)
        
        print(f"[Main Server] Started on {self.host}:{self.port}")
        print("=" * 60)
        print("ARCHITECTURE:")
        print(f"  1. Main Server (port {self.port}): Accepts client requests")
        print(f"  2. Worker Server (port {self.worker_port}): Receives worker callbacks")
        print(f"  3. Worker Process: Connects back to port {self.worker_port}")
        print("=" * 60)
        print()
        
        try:
            while not self.shutdown_flag.is_set():
                try:
                    conn, addr = self.server_socket.accept()
                    print(f"[Main Server] Client connected from {addr}")
                    
                    # Handle client in same thread for simplicity
                    # (In real code, use ThreadPoolExecutor)
                    threading.Thread(
                        target=self.handle_client, 
                        args=(conn,), 
                        daemon=True
                    ).start()
                    
                except socket.timeout:
                    continue
                except KeyboardInterrupt:
                    print("\n[Main Server] Shutting down...")
                    break
                    
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Shutdown the server"""
        self.shutdown_flag.set()
        
        if self.server_socket:
            self.server_socket.close()
        
        if self.worker_server_socket:
            self.worker_server_socket.close()
        
        print("[Server] Shutdown complete")


def test_client():
    """Test client that sends a request"""
    time.sleep(1)  # Wait for server to start
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((HOST, PORT))
        conn = SocketConnection(sock)
        
        print("[Client] Connected to server")
        
        # Send request
        request = {
            "method": "process_task",
            "task_data": {
                "task_name": "MyTask",
                "iterations": 3
            }
        }
        
        print(f"[Client] Sending request: {request}")
        conn.send(request)
        
        # Receive streaming responses
        while True:
            response = conn.recv()
            if response is None:
                break
            
            print(f"[Client] Received: {response}")
            
            if response.get("type") in ["result", "error"]:
                break
        
        conn.close()
        print("[Client] Done")
        
    except Exception as e:
        print(f"[Client] Error: {e}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    # Start server in main thread
    server = SimpleServer()
    
    # Start test client in separate thread
    client_thread = threading.Thread(target=test_client, daemon=True)
    client_thread.start()
    
    # Run server
    try:
        server.start()
    except KeyboardInterrupt:
        print("\nShutting down...")