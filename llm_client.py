import asyncio
import sys

HOST = '127.0.0.1'
PORT = 8888
END_TURN_SIGNAL = "||WAITING_INPUT||"

async def run_client():
    try:
        reader, writer = await asyncio.open_connection(HOST, PORT)
    except ConnectionRefusedError:
        print("Could not connect to server. Is it running?")
        return

    print(f"Connected to {HOST}:{PORT}")
    
    # Buffer for incoming data
    buffer = ""
    
    while True:
        try:
            # Read byte by byte or small chunks for smooth streaming
            chunk = await reader.read(100)
            if not chunk:
               print("\nServer closed connection.")
               break
            
            text = chunk.decode('utf-8', errors='ignore')
            buffer += text
            
            # Check if buffer contains the "My turn" signal
            if END_TURN_SIGNAL in buffer:
                # Split: content before signal, content after
                pre, post = buffer.split(END_TURN_SIGNAL, 1)
                
                # Print the final bit of text before the signal
                print(pre, end="", flush=True)
                
                # Reset buffer (in case post contains partial next message, unlikely here)
                buffer = post 
                
                # --- USER INPUT BLOCK ---
                # The server is done, it's our turn.
                try:
                    user_input = input("\nYou: ").strip()
                except EOFError:
                    break

                if not user_input:
                    user_input = "..." # Sends non-empty to keep logic flowing

                writer.write(user_input.encode('utf-8'))
                await writer.drain()
                
                if user_input.lower() == 'quit':
                    break
            else:
                # Just standard streaming content
                # We print buffer and clear it to avoid re-printing
                # (In a robust app we'd wait for newline, but here we print raw)
                print(buffer, end="", flush=True)
                buffer = ""
                
        except Exception as e:
            print(f"\nError: {e}")
            break

    writer.close()
    await writer.wait_closed()

if __name__ == "__main__":
    # Windows specific fix for asyncio
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    try:
        asyncio.run(run_client())
    except KeyboardInterrupt:
        pass