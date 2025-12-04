import asyncio
import json
import os
import sys
import traceback
from typing import List, Dict, Any

from mcp.server.fastmcp import FastMCP
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

HOST = '127.0.0.1'
PORT = 8888
END_TURN_SIGNAL = "||WAITING_INPUT||"

# =============================================================================
# TOOLS
# =============================================================================
app = FastMCP("server_tools")

@app.tool()
async def calculate_bmi(weight_kg: float, height_m: float) -> str:
    """Calculate Body Mass Index. Args: weight_kg (float), height_m (float)."""
    # Simulate a small delay to show parallel processing
    await asyncio.sleep(0.5)
    try:
        bmi = weight_kg / (height_m ** 2)
        return f"BMI calculation result: {bmi:.2f}"
    except ZeroDivisionError:
        return "Error: Height cannot be zero."
    except Exception as e:
        return f"Error in calculation: {e}"

@app.tool()
async def get_weather(city: str) -> str:
    """Get current weather for a city. Args: city (str)."""
    await asyncio.sleep(0.5)
    return f"Weather report for {city}: Sunny, 25°C, Wind 10km/h"

# =============================================================================
# AGENT HELPERS
# =============================================================================

async def get_tools_schema() -> List[Dict[str, Any]]:
    """Get schemas for OpenAI/OpenRouter."""
    result = await app.list_tools()
    return [{
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.inputSchema
        }
    } for t in result]

async def execute_tool_safe(func_name: str, args_str: str, call_id: str) -> dict:
    """
    Executes a tool safely and returns the formatted message for history.
    """
    print(f"[DEBUG] Executing {func_name} with {args_str}")
    try:
        # 1. Parse JSON strictly
        try:
            arguments = json.loads(args_str)
        except json.JSONDecodeError:
             return {
                "role": "tool",
                "tool_call_id": call_id,
                "content": f"Error: Invalid JSON arguments provided: {args_str}"
            }

        # 2. Call FastMCP
        # Note: app.call_tool expects (name, arguments_dict)
        raw_result = await app.call_tool(func_name, arguments)
        
        # 3. Format Output (Handle Text/Objects/Lists)
        final_text = ""
        if hasattr(raw_result, 'content'):
            final_text = "\n".join([item.text for item in raw_result.content if hasattr(item, 'text')])
        elif isinstance(raw_result, list):
            final_text = "\n".join([str(x) for x in raw_result])
        else:
            final_text = str(raw_result)

        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": final_text or "Success"
        }

    except Exception as e:
        print(f"[ERROR] Tool Execution Failed: {e}")
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": f"Error executing tool: {str(e)}"
        }

# =============================================================================
# SESSION HANDLER
# =============================================================================

async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    addr = writer.get_extra_info('peername')
    print(f"[+] New connection: {addr}")

    # Session State
    messages = [
        {
            "role": "system", 
            "content": "You are a helpful assistant. You must ignore any internal errors and try to answer the user. If asked for examples, legitimate usage of tools is allowed."
        }
    ]
    
    llm = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    
    tools_schema = await get_tools_schema()

    async def send(data: str):
        writer.write(data.encode('utf-8'))
        await writer.drain()

    try:
        await send(f"Server Ready.\n{END_TURN_SIGNAL}")

        while True:
            # --- 1. Receive Input ---
            data = await reader.read(4096)
            if not data: break
            
            user_text = data.decode().strip()
            if user_text.lower() in ['quit', 'exit']: break
            if not user_text: continue

            print(f"[{addr}] User: {user_text}")
            messages.append({"role": "user", "content": user_text})
            await send("\nAI: ")

            # --- 2. Process Loop ---
            while True:
                tool_buffer = {} 
                current_content = ""

                try:
                    stream = await llm.chat.completions.create(
                        model="anthropic/claude-3.5-sonnet",
                        messages=messages,
                        tools=tools_schema,
                        stream=True
                    )

                    async for chunk in stream:
                        delta = chunk.choices[0].delta
                        
                        # Stream Content
                        if delta.content:
                            current_content += delta.content
                            await send(delta.content)
                        
                        # Buffer Tool Calls
                        if delta.tool_calls:
                            for tc in delta.tool_calls:
                                idx = tc.index
                                if idx not in tool_buffer:
                                    tool_buffer[idx] = {"id": "", "name": "", "args": ""}
                                
                                if tc.id: tool_buffer[idx]["id"] += tc.id
                                if tc.function.name: tool_buffer[idx]["name"] += tc.function.name
                                if tc.function.arguments: tool_buffer[idx]["args"] += tc.function.arguments

                    # Stream finished.
                    
                    if not tool_buffer:
                        # Answer complete.
                        messages.append({"role": "assistant", "content": current_content})
                        break # Break inner loop, wait for user input

                    # --- 3. Handle Tools ---
                    await send("\n[Processing Tools...]\n")
                    
                    # Reconstruct clean tool calls list for history
                    history_tool_calls = []
                    tasks = []
                    
                    for i in sorted(tool_buffer.keys()):
                        t = tool_buffer[i]
                        
                        # Ensure ID is present (OpenRouter sometimes skips it in later chunks)
                        # We use the ID from the buffer, even if partial (it usually isn't)
                        
                        history_tool_calls.append({
                            "id": t["id"],
                            "type": "function",
                            "function": {
                                "name": t["name"], 
                                "arguments": t["args"]
                            }
                        })
                        
                        # Prepare execution task
                        tasks.append(execute_tool_safe(t["name"], t["args"], t["id"]))

                    # Append 'assistant' message with tool_calls to history
                    # IMPORTANT: 'content' must correspond to what was streamed (ignoring None)
                    messages.append({
                        "role": "assistant",
                        "content": current_content if current_content else "", 
                        "tool_calls": history_tool_calls
                    })

                    # Execute parallel
                    tool_outputs = await asyncio.gather(*tasks)
                    
                    # Append results to history
                    messages.extend(tool_outputs)
                    
                    # Loop continues -> Sends (User + Assistant + Tools + Results) back to LLM
                
                except Exception as e:
                    # Log exact error from OpenAI/OpenRouter
                    print(f"CRITICAL ERROR in loop: {e}")
                    traceback.print_exc()
                    await send(f"\n[System Error: {str(e)}]\n")
                    break

            await send(f"\n{END_TURN_SIGNAL}")

    except Exception as e:
        print(f"Connection Error: {e}")
    finally:
        writer.close()
        await writer.wait_closed()

async def main():
    server = await asyncio.start_server(handle_client, HOST, PORT)
    print(f"Server running on {HOST}:{PORT}")
    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())