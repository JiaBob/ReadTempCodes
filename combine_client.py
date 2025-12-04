import socket
import struct
import json
import sys
import threading
import time
from typing import Optional, Dict, Any

class SocketProtocol:
    """Handles the low-level formatting of JSON messages."""
    
    @staticmethod
    def send(sock, data: dict):
        try:
            json_data = json.dumps(data).encode('utf-8')
            # Prefix with 4-byte Header (legacy network byte order)
            msg_length = struct.pack('>I', len(json_data))
            sock.sendall(msg_length + json_data)
        except BrokenPipeError:
            pass

    @staticmethod
    def recv(sock) -> Optional[dict]:
        try:
            # Read 4-byte Header
            raw_msglen = SocketProtocol._recvall(sock, 4)
            if not raw_msglen: return None
            msglen = struct.unpack('>I', raw_msglen)[0]
            
            # Read Body
            data = SocketProtocol._recvall(sock, msglen)
            if not data: return None
            
            return json.loads(data.decode('utf-8'))
        except Exception:
            return None

    @staticmethod
    def _recvall(sock, n):
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet: return None
            data.extend(packet)
        return bytes(data)

class MockClientState:
    """
    Simulates the Client's Local State (e.g., Browser Data, API State).
    The Server will 'query' this.
    """
    def __init__(self):
        # This represents data sitting on the Client Machine
        self.available_datasets = [
            {
                "id": "ds_heart_001",
                "name": "Heart Disease Records",
                "columns": ["age", "sex", "cp", "trestbps", "chol", "target"],
                "rows": 303
            },
            {
                "id": "ds_liver_002",
                "name": "Liver Patient Dataset",
                "columns": ["age", "gender", "total_bilirubin", "albumin", "liver_score"],
                "rows": 583
            }
        ]

    def get_all_datasets(self):
        return self.available_datasets

    def get_dataset_sample(self, dataset_id):
        # Simulate fetching raw data for the server
        if dataset_id == "ds_liver_002":
            return [
                {"age": 65, "gender": "Female", "total_bilirubin": 0.7, "liver_score": 1},
                {"age": 62, "gender": "Male", "total_bilirubin": 10.9, "liver_score": 2}
            ]
        return []

class ProductionClient:
    def __init__(self, host='127.0.0.1', port=65432):
        self.host = host
        self.port = port
        self.sock = None
        self.running = False
        self.state = MockClientState()

    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            print(f"✅ Connected to Server at {self.host}:{self.port}")
            self.running = True
            
            # Start Listener Thread
            listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
            listen_thread.start()
            
            # Start Input Loop (Main Thread)
            self._input_loop()
            
        except ConnectionRefusedError:
            print("❌ Could not connect to server.")
        except KeyboardInterrupt:
            self.disconnect()
        finally:
            self.disconnect()

    def disconnect(self):
        self.running = False
        if self.sock:
            try:
                SocketProtocol.send(self.sock, {"method": "disconnect"})
                self.sock.close()
            except: pass
        print("\n🔌 Disconnected.")
        sys.exit(0)

    def _listen_loop(self):
        """
        The 'Passive' loop.
        It blindly reacts to commands from the Server (The Brain).
        """
        while self.running:
            msg = SocketProtocol.recv(self.sock)
            if not msg:
                print("\n⚠️ Server closed connection.")
                self.running = False
                break
            
            msg_type = msg.get("type")

            # 1. LLM Output (Stream or Final)
            if msg_type == "llm_stream":
                print(msg.get("content"), end="", flush=True)
            
            elif msg_type == "llm_complete":
                print("\n" + "-"*50) # End of turn visual separator

            # 2. Error Handling
            elif msg_type == "error":
                print(f"\n❌ Error: {msg.get('message')}")

            # 3. Protocol Request: Server needs Client Info (P2P Data Fetch)
            elif msg_type == "context_query":
                self._handle_context_query(msg)

            # 4. Protocol Request: Server wants to update Client UI
            elif msg_type == "ui_render":
                self._handle_ui_render(msg)

    def _handle_context_query(self, msg):
        """
        Server says: "I need data X". 
        Client says: "Here is data X".
        """
        query_key = msg.get("query")
        print(f"\n[System] Server requesting local context: {query_key}...", end="")
        
        response_data = None
        
        if query_key == "list_datasets":
            response_data = self.state.get_all_datasets()
        elif query_key == "get_sample":
            ds_id = msg.get("args", {}).get("dataset_id")
            response_data = self.state.get_dataset_sample(ds_id)
            
        # Send result back regardless of success or failure
        SocketProtocol.send(self.sock, {
            "type": "context_response",
            "req_id": msg.get("req_id"), # Sync ID
            "data": response_data
        })
        print(" Sent. ✅")

    def _handle_ui_render(self, msg):
        """
        Server says: "Show a chart/table".
        Client says: "Okay, drawing it."
        """
        component = msg.get("component")
        data = msg.get("data")
        
        print(f"\n\n🖥️  [CLIENT UI RENDER: {component.upper()}]")
        if component == "table":
            # Mock rendering a table
            print(json.dumps(data, indent=2))
        elif component == "chart":
            print(f"📊 Drawing chart for: {data.get('title', 'Untitled')}")
        print("-" * 30 + "\n")
        
        # Acknowledge (Optional, depends on server logic)
        SocketProtocol.send(self.sock, {
            "type": "ui_ack",
            "req_id": msg.get("req_id")
        })

    def _input_loop(self):
        print("\n💬 Chat ready. Type '/quit' to exit.\n")
        while self.running:
            try:
                user_input = input("You > ")
                if user_input.strip() == "": continue
                
                if user_input.lower() in ["/quit", "/exit"]:
                    self.disconnect()
                    
                SocketProtocol.send(self.sock, {
                    "method": "chat",
                    "params": {"message": user_input}
                })
                # Wait a moment for UI purposes
                time.sleep(0.1)
            except EOFError:
                self.disconnect()

if __name__ == "__main__":
    client = ProductionClient()
    client.connect()