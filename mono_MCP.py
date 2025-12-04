import asyncio
import json
import os
import sys
from typing import List, Dict, Any

# 1. Import FastMCP
from mcp.server.fastmcp import FastMCP

# 2. Import AI Libraries
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# PART A: DEFINE TOOLS
# =============================================================================
app = FastMCP("my_internal_tools")

@app.tool()
def calculate_bmi(weight_kg: float, height_m: float) -> str:
    """Calculate Body Mass Index."""
    try:
        bmi = weight_kg / (height_m ** 2)
        category = "Normal"
        if bmi < 18.5: category = "Underweight"
        if bmi > 25: category = "Overweight"
        return f"BMI is {bmi:.2f} ({category})"
    except Exception as e:
        return f"Error calculating BMI: {e}"

@app.tool()
def plan_trip(destination: str, days: int, activity_type: str = "leisure") -> str:
    """Plan a generic trip itinerary."""
    return json.dumps({
        "destination": destination,
        "duration": f"{days} days",
        "style": activity_type,
        "itinerary": ["Day 1: Arrival", "Day 2: Main Activity", "Day 3: Departure"]
    })

# =============================================================================
# PART B: HELPERS
# =============================================================================

async def get_openai_tools(mcp_app: FastMCP) -> List[Dict[str, Any]]:
    """Extract schemas from FastMCP to send to OpenAI."""
    result = await mcp_app.list_tools()
    return [{
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputSchema
        }
    } for tool in result]

async def execute_tool_locally(mcp_app: FastMCP, name: str, arguments: dict) -> str:
    """
    Executes tool safely handling the Tuple/List/Object return variations.
    """
    try:
        raw_result = await mcp_app.call_tool(name, arguments)
        
        content_list = []
        
        # 1. Handle Tuple return (Fixes the error you saw)
        if isinstance(raw_result, tuple):
            content_list = raw_result[0]
        # 2. Handle Object with .content
        elif hasattr(raw_result, 'content'):
            content_list = raw_result.content
        # 3. Handle List directly
        elif isinstance(raw_result, list):
            content_list = raw_result
        else:
            return str(raw_result)

        # Extract actual text
        final_text = []
        for item in content_list:
            if hasattr(item, 'text'):
                final_text.append(item.text)
            elif isinstance(item, dict) and 'text' in item:
                final_text.append(item['text'])
            else:
                final_text.append(str(item))
                
        return "\n".join(final_text)
    except Exception as e:
        return f"Tool execution failed: {str(e)}"

# =============================================================================
# PART C: MAIN STREAMING LOOP
# =============================================================================

async def main():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set.")
        return
        
    llm = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={"HTTP-Referer": "http://localhost", "X-Title": "LocalStream"}
    )
    
    system_tools = await get_openai_tools(app)
    messages = []

    print("\n--- Streaming Agent (Type 'quit' to exit) ---")
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            break
            
        messages.append({"role": "user", "content": user_input})
        print("\nAI: ", end="", flush=True)
        
        # --- AGENT LOOP (Handles Multi-Step Logic) ---
        while True:
            try:
                stream = await llm.chat.completions.create(
                    model="anthropic/claude-3.5-sonnet",
                    messages=messages,
                    tools=system_tools,
                    stream=True # ENABLE STREAMING
                )

                final_content = ""
                tool_calls_buffer = {} # Dict to re-assemble chunked tool calls
                
                # --- STREAM PROCESSING ---
                async for chunk in stream:
                    delta = chunk.choices[0].delta

                    # 1. Stream Text to Console
                    if delta.content:
                        print(delta.content, end="", flush=True)
                        final_content += delta.content

                    # 2. Buffer Tool Calls (They arrive in pieces)
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index
                            if idx not in tool_calls_buffer:
                                tool_calls_buffer[idx] = {"id": "", "name": "", "args": ""}
                            
                            if tc.id: tool_calls_buffer[idx]["id"] += tc.id
                            if tc.function.name: tool_calls_buffer[idx]["name"] += tc.function.name
                            if tc.function.arguments: tool_calls_buffer[idx]["args"] += tc.function.arguments

                # --- END OF ONE RESPONSE TURN ---
                
                # If we have tool calls, we must process them and loop again
                if tool_calls_buffer:
                    print("\n\n[Processing tools...]")
                    
                    # 1. Reconstruct OpenAI-compliant tool_calls list
                    reconstructed_calls = []
                    for idx in sorted(tool_calls_buffer.keys()):
                        t = tool_calls_buffer[idx]
                        reconstructed_calls.append({
                            "id": t["id"],
                            "type": "function",
                            "function": {"name": t["name"], "arguments": t["args"]}
                        })

                    # 2. Add Assistant's "Intent" to history
                    messages.append({
                        "role": "assistant",
                        "content": final_content if final_content else None,
                        "tool_calls": reconstructed_calls
                    })

                    # 3. Execute Tools Locally
                    for tool_call in reconstructed_calls:
                        func_name = tool_call["function"]["name"]
                        args_str = tool_call["function"]["arguments"]
                        call_id = tool_call["id"]
                        
                        print(f" > Executing {func_name}...")
                        
                        try:
                            args = json.loads(args_str)
                            # Call the robust local executor
                            result_text = await execute_tool_locally(app, func_name, args)
                        except Exception as e:
                            result_text = f"Error parsing arguments: {e}"

                        # 4. Add Tool Result to history
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": result_text
                        })
                    
                    print("[Sending results back to AI...]\n")
                    print("AI: ", end="", flush=True)
                    # LOOP CONTINUES (The inner while True repeats)
                    continue

                else:
                    # No tools called, this was a final text response
                    messages.append({"role": "assistant", "content": final_content})
                    break # Break inner loop, wait for user

            except Exception as e:
                print(f"\nError in stream: {e}")
                break

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())