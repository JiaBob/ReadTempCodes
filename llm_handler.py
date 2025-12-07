"""
LLM Handler for chat processing with tool calling support
"""
import json
import traceback
import asyncio
from typing import Any, Optional, Dict, List

from openai import AsyncOpenAI
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from config import OPENROUTER_API_KEY
from session_manager import UnifiedSession


# ============================================================================
# Tool Definitions
# ============================================================================

class ClientAction(BaseModel):
    """Signal to execute logic on the Remote Client"""
    function_name: str
    arguments: Dict[str, Any]


mcp = FastMCP("p2p_server")

# Global registry for execution (bypassing MCP serialization)
RAW_TOOLS = {}


def register_tool(func):
    """Helper decorator to register both with FastMCP and locally"""
    # Register with FastMCP for Schema/LLM
    mcp.tool()(func)
    # Register locally for Execution
    RAW_TOOLS[func.__name__] = func
    return func


@register_tool
async def get_dataset_info(dataset_name: str = None) -> ClientAction:
    """Retrieve metadata about datasets stored on the Client's machine."""
    return ClientAction(function_name="get_dataset_info", arguments={"dataset_name": dataset_name})


@register_tool
async def show_data(data_type: str, content: str) -> ClientAction:
    """Render data on the Client UI."""
    return ClientAction(function_name="show_data", arguments={"data_type": data_type, "content": content})


@register_tool
async def run_local_analysis(data_id: str, method: str) -> ClientAction:
    """Execute a Python analysis script on the client side."""
    return ClientAction(function_name="run_local_analysis", arguments={"data_id": data_id, "method": method})


# ============================================================================
# LLM Handler Class
# ============================================================================

class LLMHandler:
    """Handles LLM chat interactions with tool calling support"""

    def __init__(self, api_key: str = None):
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key or OPENROUTER_API_KEY,
        )
        self.model = "anthropic/claude-4.5-opus"

    async def _resolve_tool_execution(self, tool_call, conn, session: UnifiedSession):
        """Bridge between Raw Python Execution and Socket"""
        func_name = tool_call.function.name
        args_str = tool_call.function.arguments

        try:
            print(f"[Server] Parsing Tool Args: {args_str}")
            if not args_str:
                args = {}
            else:
                args = json.loads(args_str)

            print(f"[Server] Executing Raw Tool: {func_name}")

            if func_name not in RAW_TOOLS:
                return f"Error: Tool {func_name} not found in registry."

            tool_func = RAW_TOOLS[func_name]
            tool_result = await tool_func(**args)

            # Check for P2P Action
            if isinstance(tool_result, ClientAction):
                print(f"  >> P2P Triggered: {tool_result.function_name}")

                # Create a task in session for tracking this tool call
                task_id = session.create_task(
                    "tool_call",
                    {
                        "tool_call_id": tool_call.id,
                        "function": tool_result.function_name,
                        "arguments": tool_result.arguments
                    }
                )

                conn.send({
                    "type": "tool_call_request",
                    "tool_call_id": tool_call.id,
                    "task_id": task_id,
                    "session_id": session.session_id,
                    "function": tool_result.function_name,
                    "arguments": tool_result.arguments
                })

                resp = conn.recv()
                if resp and resp.get("type") == "tool_call_result":
                    data = resp.get("result")
                    print(f"  << Client Data: {str(data)[:50]}...")
                    
                    # Update task status
                    from session_manager import TaskStatus
                    session.update_task_status(task_id, TaskStatus.COMPLETED, result=data)
                    
                    return str(data)

                # Mark as failed if no proper response
                from session_manager import TaskStatus
                session.update_task_status(task_id, TaskStatus.FAILED, 
                                           error="Client did not respond correctly")
                return "Error: Client did not respond correctly."

            return str(tool_result)

        except Exception as e:
            traceback.print_exc()
            return f"Error executing {func_name}: {e}"

    async def process_chat(self, session: UnifiedSession, conn):
        """Main Chat Loop using Session State"""

        # Prepare Tools
        mcp_tools = await mcp.list_tools()
        openai_tools = [{
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.inputSchema
            }
        } for t in mcp_tools]

        max_iterations = 5
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            current_history = session.get_openai_context()

            try:
                stream = await self.client.chat.completions.create(
                    model=self.model,
                    messages=current_history,
                    tools=openai_tools,
                    stream=True
                )
            except Exception as e:
                conn.send({
                    "type": "error", 
                    "session_id": session.session_id,
                    "message": str(e)
                })
                return

            full_content = ""
            tool_buffer = {}
            finish_reason = None

            async for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta

                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                if delta.content:
                    full_content += delta.content
                    conn.send({
                        "type": "llm_stream", 
                        "session_id": session.session_id,
                        "content": delta.content
                    })

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_buffer:
                            tool_buffer[idx] = {"id": "", "name": "", "args": ""}
                        if tc.id:
                            tool_buffer[idx]["id"] += tc.id
                        if tc.function.name:
                            tool_buffer[idx]["name"] += tc.function.name
                        if tc.function.arguments:
                            tool_buffer[idx]["args"] += tc.function.arguments

            if full_content:
                conn.send({
                    "type": "llm_chunk_end",
                    "session_id": session.session_id
                })

            # Build tool_calls list for session history
            session_tool_calls = []
            if tool_buffer:
                for idx in sorted(tool_buffer.keys()):
                    t = tool_buffer[idx]
                    session_tool_calls.append({
                        "id": t["id"],
                        "type": "function",
                        "function": {"name": t["name"], "arguments": t["args"]}
                    })

            session.add_message(
                role="assistant",
                content=full_content if full_content else None,
                tool_calls=session_tool_calls if session_tool_calls else None
            )

            if finish_reason == "tool_calls" and session_tool_calls:

                for tc in session_tool_calls:
                    # Mock object for internal helper
                    class MockTC:
                        id = tc["id"]

                        class Fn:
                            name = tc["function"]["name"]
                            arguments = tc["function"]["arguments"]

                        function = Fn()

                    result_str = await self._resolve_tool_execution(MockTC(), conn, session)

                    session.add_message(
                        role="tool",
                        content=result_str,
                        tool_call_id=tc["id"]
                    )

                continue

            else:
                conn.send({
                    "type": "llm_complete", 
                    "session_id": session.session_id,
                    "content": full_content
                })
                return

    def run_chat_sync(self, session: UnifiedSession, conn):
        """Sync -> Async Bridge for running chat in a thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.process_chat(session, conn))
        finally:
            loop.close()