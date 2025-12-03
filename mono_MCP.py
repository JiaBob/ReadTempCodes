import asyncio
import json
import os
from typing import List, Dict, Any

from mcp.server.fastmcp import FastMCP
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# PART A: TOOLS
# =============================================================================
app = FastMCP("my_internal_tools")

@app.tool()
def calculate_bmi(weight_kg: float, height_m: float) -> str:
    """Calculate Body Mass Index."""
    bmi = weight_kg / (height_m ** 2)
    category = "Normal"
    if bmi < 18.5: category = "Underweight"
    if bmi > 25: category = "Overweight"
    return f"BMI is {bmi:.2f} ({category})"

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
# PART B: IMPROVED EXECUTION LOGIC
# =============================================================================

async def get_openai_tools(mcp_app: FastMCP) -> List[Dict[str, Any]]:
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
    Robustly executes a tool and cleans the output.
    """
    raw_result = await mcp_app.call_tool(name, arguments)

    # DEBUG: Uncomment if you want to see exactly what comes back
    # print(f"DEBUG RAW: {type(raw_result)} - {raw_result}")

    content_list = []
    
    # 1. Handle Tuple return (The issue you saw: ([TextContent], MetaDict))
    if isinstance(raw_result, tuple):
        # The first item is usually the list of content objects
        content_list = raw_result[0]
    
    # 2. Handle Object with .content attribute (Standard MCP)
    elif hasattr(raw_result, 'content'):
        content_list = raw_result.content
        
    # 3. Handle direct list
    elif isinstance(raw_result, list):
        content_list = raw_result
        
    # 4. Handle direct string/other
    else:
        return str(raw_result)

    # Extract text from the list of content objects
    final_text = []
    for item in content_list:
        # Check for object attribute
        if hasattr(item, 'text'):
            final_text.append(item.text)
        # Check for dictionary key
        elif isinstance(item, dict) and 'text' in item:
            final_text.append(item['text'])
        else:
            final_text.append(str(item))
            
    return "\n".join(final_text)

# =============================================================================
# PART C: AGENT LOOP (WITH ITERATIVE EXECUTION)
# =============================================================================

async def main():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found.")
        return
        
    llm = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "LocalApp"
        }
    )
    
    system_tools = await get_openai_tools(app)
    messages = []

    print("\n--- Start Chatting (Type 'quit' to exit) ---")
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            break
            
        messages.append({"role": "user", "content": user_input})
        
        # --- INNER LOOP: Keep processing until the AI stops calling tools ---
        while True:
            try:
                response = await llm.chat.completions.create(
                    model="anthropic/claude-3.5-sonnet",
                    messages=messages,
                    tools=system_tools
                )
                
                msg = response.choices[0].message
                
                # Check if the AI wants to call tools
                if msg.tool_calls:
                    # 1. Print the AI's "Thoughts" before the tool call (if any)
                    if msg.content:
                        print(f"\nAI: {msg.content}")
                    
                    # 2. Add the "Call Intent" to history
                    messages.append(msg)
                    
                    print("\n[AI is executing tools...]")
                    
                    # 3. Execute all requested tools
                    for tool_call in msg.tool_calls:
                        func_name = tool_call.function.name
                        args = json.loads(tool_call.function.arguments)
                        
                        print(f" > Calling: {func_name}({args})")
                        
                        # Execute cleanly
                        tool_output = await execute_tool_locally(app, func_name, args)
                        
                        print(f" > Output: {tool_output[:100]}...") # Print preview only
                        
                        # Add result to history
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_output
                        })
                    
                    # 4. CONTINUE LOOP: Send results back to LLM and see if it wants more tools
                    print("[Sending results back to AI process...]")
                    continue 

                else:
                    # No tool calls? We are done.
                    print(f"\nAI: {msg.content}")
                    messages.append(msg)
                    break # Break inner loop, wait for user input

            except Exception as e:
                print(f"API Error: {e}")
                break

if __name__ == "__main__":
    asyncio.run(main())