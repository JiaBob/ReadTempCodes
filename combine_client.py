import socket
import struct
import json
import sys
import threading
import queue
from typing import Optional, Callable


class SocketConnection:
    """Socket wrapper for JSON communication"""
    
    def __init__(self, sock):
        self.sock = sock
    
    def send(self, obj):
        """Send JSON object"""
        data = json.dumps(obj).encode('utf-8')
        msg_length = struct.pack('>I', len(data))
        self.sock.sendall(msg_length + data)
    
    def recv(self):
        """Receive JSON object"""
        raw_msglen = self._recvall(4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('>I', raw_msglen)[0]
        
        data = self._recvall(msglen)
        if not data:
            return None
        
        return json.loads(data.decode('utf-8'))
    
    def _recvall(self, n):
        data = bytearray()
        while len(data) < n:
            packet = self.sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return bytes(data)
    
    def close(self):
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except:
            pass
        try:
            self.sock.close()
        except:
            pass


class UnifiedClient:
    """
    Unified client supporting:
    1. Pipeline execution requests
    2. LLM chat with streaming
    3. Tool execution (responding to server's tool calls)
    """
    
    def __init__(self, host='127.0.0.1', port=65432):
        self.host = host
        self.port = port
        self.sock = None
        self.conn: Optional[SocketConnection] = None
        self.session_id: Optional[str] = None
        
        # Tool handlers registry
        self.tool_handlers = {}
        self.register_default_tools()
    
    def connect(self):
        """Connect to server"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        self.conn = SocketConnection(self.sock)
        
        # Receive session info
        session_msg = self.conn.recv()
        if session_msg and session_msg.get("type") == "session_created":
            self.session_id = session_msg.get("session_id")
            print(f"Connected to server. Session ID: {self.session_id}\n")
    
    def disconnect(self):
        """Disconnect from server"""
        if self.conn:
            try:
                self.conn.send({"method": "disconnect"})
            except:
                pass
            self.conn.close()
    
    # ========================================================================
    # Tool Handlers (P2P: Server calls these client-side functions)
    # ========================================================================
    
    def register_tool(self, name: str, handler: Callable):
        """Register a tool handler"""
        self.tool_handlers[name] = handler
    
    def register_default_tools(self):
        """Register default client-side tools"""
        
        def get_dataset_info(dataset_name: Optional[str] = None):
            """Mock: Get dataset information"""
            datasets = {
                "sales_data": {"rows": 10000, "columns": 5, "size": "2.3 MB"},
                "customer_data": {"rows": 5000, "columns": 8, "size": "1.1 MB"}
            }
            
            if dataset_name:
                return json.dumps(datasets.get(dataset_name, {"error": "Dataset not found"}))
            return json.dumps(datasets)
        
        def show_data(data_type: str, content: str):
            """Mock: Display data on client UI"""
            print(f"\n[CLIENT UI - {data_type.upper()}]")
            print("=" * 50)
            print(content)
            print("=" * 50)
            return json.dumps({"status": "displayed", "type": data_type})
        
        def execute_pipeline(pipeline_json: str, mode: str = "complete"):
            """Execute pipeline (delegates back to server)"""
            # This creates a nested request back to server
            result = self.send_request("run_pipeline", {
                "pipeline_json": pipeline_json,
                "mode": mode,
                "output_path": "test_cache",
                "debug": False,
                "log_level": "info"
            })
            return json.dumps({"status": "pipeline_executed", "result": str(result)})
        
        self.register_tool("get_dataset_info", get_dataset_info)
        self.register_tool("show_data", show_data)
        self.register_tool("execute_pipeline", execute_pipeline)
    
    def handle_tool_call(self, tool_call_msg):
        """Handle a tool call request from server"""
        func_name = tool_call_msg.get("function")
        arguments = tool_call_msg.get("arguments", {})
        tool_call_id = tool_call_msg.get("tool_call_id")
        
        print(f"\n[TOOL CALL] {func_name}({arguments})")
        
        # Execute tool
        if func_name in self.tool_handlers:
            try:
                result = self.tool_handlers[func_name](**arguments)
            except Exception as e:
                result = json.dumps({"error": str(e)})
        else:
            result = json.dumps({"error": f"Unknown tool: {func_name}"})
        
        # Send result back to server
        self.conn.send({
            "type": "tool_call_result",
            "tool_call_id": tool_call_id,
            "result": result
        })
    
    # ========================================================================
    # Request Methods
    # ========================================================================
    
    def send_request(self, method: str, params: dict = None):
        """Send a request and collect all responses"""
        request = {
            "method": method,
            "params": params or {}
        }
        
        self.conn.send(request)
        
        responses = []
        while True:
            response = self.conn.recv()
            
            if response is None:
                break
            
            response_type = response.get("type")
            
            # Handle tool call requests
            if response_type == "tool_call_request":
                self.handle_tool_call(response)
                continue
            
            responses.append(response)
            
            # Check for completion
            if response_type in ["result", "error", "llm_complete"]:
                break
        
        return responses
    
    def chat(self, message: str, on_stream: Optional[Callable] = None):
        """
        Send a chat message and handle streaming response
        
        Args:
            message: User message
            on_stream: Callback for streaming tokens (optional)
        """
        request = {
            "method": "chat",
            "params": {"message": message}
        }
        
        self.conn.send(request)
        
        full_response = ""
        
        while True:
            response = self.conn.recv()
            
            if response is None:
                break
            
            response_type = response.get("type")
            
            # Handle streaming tokens
            if response_type == "llm_stream":
                token = response.get("content", "")
                full_response += token
                
                if on_stream:
                    on_stream(token)
                else:
                    print(token, end="", flush=True)
            
            # Handle tool calls
            elif response_type == "tool_call_request":
                self.handle_tool_call(response)
            
            # Handle completion
            elif response_type == "llm_complete":
                print()  # New line after streaming
                return response.get("content")
            
            # Handle errors
            elif response_type == "error":
                print(f"\nError: {response.get('message')}")
                return None
        
        return full_response
    
    def run_pipeline(self, pipeline_json: str, mode: str = "complete"):
        """Execute a pipeline on the server"""
        return self.send_request("run_pipeline", {
            "pipeline_json": pipeline_json,
            "mode": mode,
            "output_path": "test_cache",
            "debug": False,
            "log_level": "info"
        })
    
    def ping(self):
        """Ping server"""
        responses = self.send_request("ping")
        return responses[0] if responses else None
    
    def get_stats(self):
        """Get server statistics"""
        responses = self.send_request("get_server_stats")
        return responses[0].get("data") if responses else None
    
    # ========================================================================
    # Interactive Chat Loop
    # ========================================================================
    
    def interactive_chat(self):
        """Run an interactive chat session"""
        print("=" * 70)
        print("Unified Client - Interactive Mode")
        print("=" * 70)
        print("Commands:")
        print("  /ping          - Ping server")
        print("  /stats         - Show server stats")
        print("  /clear         - Clear chat history")
        print("  /sessions      - List active sessions")
        print("  /quit          - Exit")
        print("=" * 70)
        print()
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    if user_input == "/quit":
                        break
                    elif user_input == "/ping":
                        result = self.ping()
                        print(f"✓ {result.get('message')}")
                    elif user_input == "/stats":
                        stats = self.get_stats()
                        print(json.dumps(stats, indent=2))
                    elif user_input == "/clear":
                        responses = self.send_request("clear_session")
                        print(f"✓ {responses[0].get('message')}")
                    elif user_input == "/sessions":
                        responses = self.send_request("list_sessions")
                        print(json.dumps(responses[0].get('data'), indent=2))
                    else:
                        print(f"Unknown command: {user_input}")
                    continue
                
                # Send chat message
                print("\n🤖 Assistant: ", end="", flush=True)
                self.chat(user_input)
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {e}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Client")
    parser.add_argument('--host', type=str, default='127.0.0.1', help="Server host")
    parser.add_argument('--port', type=int, default=65432, help="Server port")
    parser.add_argument('--mode', type=str, choices=['chat', 'pipeline'], default='chat',
                        help="Client mode")
    
    args = parser.parse_args()
    
    client = UnifiedClient(host=args.host, port=args.port)
    
    try:
        client.connect()
        
        if args.mode == 'chat':
            client.interactive_chat()
        elif args.mode == 'pipeline':
            # Example pipeline execution
            result = client.run_pipeline("path/to/pipeline.json")
            print(json.dumps(result, indent=2))
        
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()