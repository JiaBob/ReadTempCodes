"""
Self-Organizing Multi-Agent Playground

A dynamic multi-agent system where the coordinator agent analyzes incoming queries
and automatically creates specialized agent teams to handle them.

Flow:
1. User query arrives
2. Team Creator agent analyzes query complexity and requirements  
3. Team Creator calls setup_agent_team tool to define the team
4. System instantiates the proposed agents dynamically
5. The instantiated team executes the query
6. Results are collected and returned
"""

import json
import asyncio
from typing import Any, Optional, Dict, List, Callable
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from config import OPENROUTER_API_KEY
from session_manager import UnifiedSession, TaskStatus
from mcp_instance import ClientAction, RAW_TOOLS
from data_schema.agent_schema import AgentConfig, AgentTeamConfig, CommunicationLink
from tools.agent_pipeline_converter import get_state_manager


# =============================================================================
# Dynamic Agent
# =============================================================================

class DynamicAgent:
    """
    An agent that can be dynamically created and configured.
    Supports bi-directional inter-agent communication.
    """
    
    def __init__(
        self,
        config: AgentConfig,
        llm_client: AsyncOpenAI,
        playground: "SelfOrganizingPlayground",
    ):
        self.config = config
        self.name = config.name
        self.client = llm_client
        self.playground = playground
        
        # Message history
        self.messages: List[Dict] = [
            {"role": "system", "content": self._build_system_prompt()}
        ]
        
        # Pending messages from other agents
        self._pending_messages: List[Dict] = []
        
        # Cached tool schemas
        self._tools_cache: Optional[List[Dict]] = None
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with communication instructions"""
        base_prompt = self.config.system_prompt
        
        # Get connected agents
        team_config = self.playground.current_team_config
        if team_config:
            connected = team_config.get_connected_agents(self.name)
            if connected:
                comm_instructions = f"""

## Inter-Agent Communication
You can communicate with these agents: {', '.join(connected)}

Use the `send_message_to_agent` tool to communicate with them.
When you receive a message from another agent, it will be included in your context.
After completing your task, provide a clear response.
"""
                return base_prompt + comm_instructions
        
        return base_prompt
    
    def add_message(self, role: str, content: str, **kwargs):
        """Add a message to conversation history"""
        msg = {"role": role}
        if content is not None:
            msg["content"] = content
        msg.update({k: v for k, v in kwargs.items() if v is not None})
        self.messages.append(msg)
    
    def receive_message(self, from_agent: str, content: str):
        """Receive a message from another agent"""
        self._pending_messages.append({
            "from": from_agent,
            "content": content,
        })
    
    def _get_messages_with_pending(self) -> List[Dict]:
        """Get messages with pending inter-agent messages injected"""
        messages = self.messages.copy()
        
        if self._pending_messages:
            pending_text = "\n\n".join([
                f"[Message from {m['from']}]:\n{m['content']}"
                for m in self._pending_messages
            ])
            messages.append({
                "role": "system",
                "content": f"[Inter-Agent Messages]\n{pending_text}"
            })
            self._pending_messages.clear()
        
        return messages
    
    async def get_tools(self) -> Optional[List[Dict]]:
        """Get all tools available to this agent"""
        if not self.config.tools_enabled:
            # Still include communication tool even if other tools disabled
            return [self._get_communication_tool_schema()]
        
        if self._tools_cache is not None:
            return self._tools_cache
        
        tools = []
        
        # Add communication tool
        tools.append(self._get_communication_tool_schema())
        
        # Add regular tools based on config
        from llm_handler import tool_registry
        
        if self.config.allowed_tools is not None:
            regular_tools = await tool_registry.get_tools_by_names(self.config.allowed_tools)
        elif self.config.excluded_tools is not None:
            regular_tools = await tool_registry.get_tools_except(self.config.excluded_tools)
        else:
            regular_tools = await tool_registry.get_all_tools()
        
        if regular_tools:
            tools.extend(regular_tools)
        
        self._tools_cache = tools
        return tools
    
    def _get_communication_tool_schema(self) -> Dict:
        """Get the schema for the inter-agent communication tool"""
        # Get connected agents
        connected = []
        if self.playground.current_team_config:
            connected = self.playground.current_team_config.get_connected_agents(self.name)
        
        return {
            "type": "function",
            "function": {
                "name": "send_message_to_agent",
                "description": f"Send a message to another agent and wait for their response. You can communicate with: {', '.join(connected) if connected else 'no agents connected'}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_agent": {
                            "type": "string",
                            "description": f"Name of the agent to send message to. Must be one of: {', '.join(connected) if connected else 'none'}",
                            "enum": connected if connected else [],
                        },
                        "message": {
                            "type": "string",
                            "description": "The message or task to send to the agent",
                        },
                        "wait_for_response": {
                            "type": "boolean",
                            "description": "Whether to wait for a response. Default: true",
                            "default": True,
                        },
                    },
                    "required": ["target_agent", "message"],
                },
            },
        }
    
    async def process_task(
        self,
        task: str,
        conn,
        session: UnifiedSession,
        skip_user_message: bool = False,
    ) -> str:
        """Process a task with LLM and tools"""
        if not skip_user_message:
            self.add_message("user", task)
        
        tools = await self.get_tools()
        iteration = 0
        final_content = ""
        
        while iteration < self.config.max_iterations:
            iteration += 1
            
            try:
                kwargs = {
                    "model": self.config.model,
                    "messages": self._get_messages_with_pending(),
                    "temperature": self.config.temperature,
                    "stream": True,
                }
                if tools:
                    kwargs["tools"] = tools
                
                stream = await self.client.chat.completions.create(**kwargs)
                
            except Exception as e:
                error_msg = f"LLM Error: {e}"
                self._send_to_client(conn, session, "error", {"message": error_msg})
                return error_msg
            
            full_content, tool_calls, finish_reason = await self._process_stream(
                stream, conn, session
            )
            
            self.add_message(
                "assistant",
                full_content if full_content else None,
                tool_calls=tool_calls if tool_calls else None,
            )
            
            if finish_reason == "tool_calls" and tool_calls:
                for tc in tool_calls:
                    result = await self._execute_tool(tc, conn, session)
                    self.add_message("tool", result, tool_call_id=tc["id"])
                continue
            
            final_content = full_content
            break
        
        self._send_to_client(conn, session, "llm_complete", {
            "content": final_content,
            "agent": self.name,
        })
        return final_content
    
    async def _process_stream(
        self,
        stream,
        conn,
        session: UnifiedSession,
    ) -> tuple[str, List[Dict], Optional[str]]:
        """Process streaming response from LLM"""
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
                self._send_to_client(conn, session, "llm_stream", {
                    "content": delta.content,
                    "agent": self.name,
                    "model": self.config.model,
                })
            
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_buffer:
                        tool_buffer[idx] = {"id": "", "name": "", "args": ""}
                    if tc.id:
                        tool_buffer[idx]["id"] += tc.id
                    if tc.function and tc.function.name:
                        tool_buffer[idx]["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        tool_buffer[idx]["args"] += tc.function.arguments
        
        tool_calls = []
        for idx in sorted(tool_buffer.keys()):
            t = tool_buffer[idx]
            tool_calls.append({
                "id": t["id"],
                "type": "function",
                "function": {"name": t["name"], "arguments": t["args"]},
            })
        
        return full_content, tool_calls, finish_reason
    
    async def _execute_tool(
        self,
        tool_call: Dict,
        conn,
        session: UnifiedSession,
    ) -> str:
        """Execute a tool call"""
        func_name = tool_call["function"]["name"]
        args_str = tool_call["function"]["arguments"]
        
        try:
            args = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError as e:
            return f"Error parsing arguments: {e}"
        
        # Handle inter-agent communication
        if func_name == "send_message_to_agent":
            return await self._handle_agent_communication(args, conn, session)
        
        # Handle regular tools
        return await self._execute_regular_tool(func_name, args, tool_call, conn, session)
    
    async def _handle_agent_communication(
        self,
        args: Dict,
        conn,
        session: UnifiedSession,
    ) -> str:
        """Handle sending message to another agent"""
        target_name = args.get("target_agent")
        message = args.get("message", "")
        wait_for_response = args.get("wait_for_response", True)
        
        # Validate target agent
        if self.playground.current_team_config:
            connected = self.playground.current_team_config.get_connected_agents(self.name)
            if target_name not in connected:
                return f"Error: Cannot communicate with '{target_name}'. Connected agents: {connected}"
        
        target_agent = self.playground.get_agent(target_name)
        if not target_agent:
            return f"Error: Agent '{target_name}' not found"
        
        self._send_to_client(conn, session, "agent_communication", {
            "from_agent": self.name,
            "to_agent": target_name,
            "message": message[:100] + "..." if len(message) > 100 else message,
        })
        
        if wait_for_response:
            # Process the message with target agent and get response
            response = await target_agent.process_task(
                f"[Message from {self.name}]: {message}",
                conn,
                session,
            )
            return f"Response from {target_name}: {response}"
        else:
            # Just notify the agent (add to pending)
            target_agent.receive_message(self.name, message)
            return f"Message sent to {target_name}"
    
    async def _execute_regular_tool(
        self,
        func_name: str,
        args: Dict,
        tool_call: Dict,
        conn,
        session: UnifiedSession,
    ) -> str:
        """Execute a regular (non-communication) tool"""
        # Get tool function
        tool_func = RAW_TOOLS.get(func_name)
        if not tool_func:
            return f"Error: Tool '{func_name}' not found"
        
        # Check permissions
        if self.config.allowed_tools is not None:
            if func_name not in self.config.allowed_tools:
                return f"Error: Agent '{self.name}' cannot use tool '{func_name}'"
        
        if self.config.excluded_tools is not None:
            if func_name in self.config.excluded_tools:
                return f"Error: Tool '{func_name}' is excluded for '{self.name}'"
        
        self._send_to_client(conn, session, "tool_call_started", {
            "tool_name": func_name,
            "agent": self.name,
            "arguments": args,
        })
        
        try:
            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(**args)
            else:
                result = tool_func(**args)
            
            # Handle ClientAction (requires client response)
            if isinstance(result, ClientAction):
                return await self._handle_client_action(result, tool_call, conn, session)
            
            self._send_to_client(conn, session, "tool_call_completed", {
                "tool_name": func_name,
                "agent": self.name,
                "success": True,
            })
            
            return str(result)
            
        except Exception as e:
            return f"Error executing {func_name}: {e}"
    
    async def _handle_client_action(
        self,
        action: ClientAction,
        tool_call: Dict,
        conn,
        session: UnifiedSession,
    ) -> str:
        """Handle a ClientAction that requires client response"""
        task_id = session.create_task(
            "tool_call",
            {
                "tool_call_id": tool_call["id"],
                "function": action.function_name,
                "arguments": action.arguments,
                "agent": self.name,
            },
            agent=self.name,
        )
        
        conn.send({
            "type": "tool_call_request",
            "tool_call_id": tool_call["id"],
            "task_id": task_id,
            "session_id": session.session_id,
            "agent": self.name,
            "function": action.function_name,
            "arguments": action.arguments,
        })
        
        resp = conn.recv()
        if resp and resp.get("type") == "tool_call_result":
            data = resp.get("result")
            session.update_task_status(task_id, TaskStatus.COMPLETED, result=data)
            return str(data)
        
        session.update_task_status(task_id, TaskStatus.FAILED, error="No response")
        return "Error: Client did not respond"
    
    def _send_to_client(self, conn, session: UnifiedSession, msg_type: str, data: Dict):
        """Send message to client"""
        try:
            conn.send({
                "type": msg_type,
                "session_id": session.session_id,
                **data,
            })
        except Exception as e:
            print(f"Error sending to client: {e}")
    
    def clear_history(self, keep_system: bool = True):
        """Clear message history"""
        if keep_system and self.messages and self.messages[0].get("role") == "system":
            self.messages = [self.messages[0]]
        else:
            self.messages = []
        self._pending_messages.clear()


# =============================================================================
# Team Creator Prompt
# =============================================================================

TEAM_CREATOR_PROMPT = """You are the Team Creator, a strategic agent that analyzes incoming queries and designs optimal agent teams to handle them.

## Your Role
When you receive a query, you must:
1. Analyze the query's complexity, domain, and requirements
2. Design an optimal team of specialized agents
3. Use the `setup_agent_team` tool to create the team
4. The system will then execute the query with your designed team

## Team Design Guidelines

### Query Analysis
- What is the main task? (analysis, processing, generation, research, etc.)
- What domains are involved? (data, code, medical, creative, etc.)
- What tools might be needed?
- How complex is the task?

### Team Size Guidelines
1. **Simple Query** (1 agent): Direct answers, simple tool usage
2. **Moderate Query** (2-3 agents): Coordinator + specialist(s)
3. **Complex Query** (3-5 agents): Coordinator + multiple specialists + QA

### Agent Design Principles
- Keep teams minimal - don't over-engineer
- Each agent should have a clear, focused role
- Entry agent (coordinator) orchestrates; specialists execute
- Write detailed, specific system prompts for each agent

### Communication Links
- Links are bi-directional: both agents can send messages to each other
- Connect agents that need to collaborate
- Entry agent usually connects to all other agents
- Specialists can connect to each other if they need to collaborate

## Example Team Structures

### Simple Research Team

Coordinator <-> Researcher



### Analysis Team
Coordinator <-> Analyst
Coordinator <-> Reviewer



### Complex Pipeline
Coordinator <-> Researcher
Coordinator <-> Analyst
Coordinator <-> Writer
Analyst <-> Writer (for data handoff)



## Important
- Always use the `setup_agent_team` tool to create your team
- Provide detailed system prompts that explain each agent's role
- The entry_agent receives the user's query first
- Use model "anthropic/claude-sonnet-4" for most agents

Now analyze the incoming query and design the optimal agent team.
"""


# =============================================================================
# Self-Organizing Playground
# =============================================================================

class SelfOrganizingPlayground:
    """
    A self-organizing multi-agent system that dynamically creates agent teams
    based on query analysis.
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or OPENROUTER_API_KEY
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        
        # Current team
        self.agents: Dict[str, DynamicAgent] = {}
        self.current_team_config: Optional[AgentTeamConfig] = None
        self.entry_agent: Optional[str] = None
        
        # Team creator agent
        self.team_creator = self._create_team_creator()
        
        # History
        self.team_history: List[Dict] = []
    
    def _create_team_creator(self) -> DynamicAgent:
        """Create the team creator agent"""
        config = AgentConfig(
            name="team_creator",
            system_prompt=TEAM_CREATOR_PROMPT,
            model="anthropic/claude-sonnet-4",
            temperature=0.7,
            max_iterations=5,
            tools_enabled=True,
            allowed_tools=["setup_agent_team", "get_team_info", "list_teams"],
        )
        
        return DynamicAgent(
            config=config,
            llm_client=self.client,
            playground=self,
        )
    
    def setup_team(self, team_config: AgentTeamConfig) -> bool:
        """
        Instantiate agents from team configuration.
        
        This method is called after setup_agent_team tool creates the config.
        It creates the actual DynamicAgent objects.
        """
        try:
            # Clear existing team
            self.agents.clear()
            
            # Store config
            self.current_team_config = team_config
            self.entry_agent = team_config.entry_agent
            
            # Create agents
            for agent_config in team_config.agents:
                agent = DynamicAgent(
                    config=agent_config,
                    llm_client=self.client,
                    playground=self,
                )
                self.agents[agent_config.name] = agent
                print(f"Created agent: {agent_config.name}")
            
            # Log connections
            for agent_name in self.agents:
                connected = team_config.get_connected_agents(agent_name)
                if connected:
                    print(f"  {agent_name} <-> {connected}")
            
            # Record history
            self.team_history.append({
                "team_name": team_config.team_name,
                "agents": [a.name for a in team_config.agents],
                "entry_agent": team_config.entry_agent,
            })
            
            print(f"Team '{team_config.team_name}' ready with {len(self.agents)} agents")
            return True
            
        except Exception as e:
            print(f"Error setting up team: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_agent(self, name: str) -> Optional[DynamicAgent]:
        """Get an agent by name"""
        return self.agents.get(name)
    
    def list_agents(self) -> List[str]:
        """List all current agent names"""
        return list(self.agents.keys())
    
    async def process_query(
        self,
        query: str,
        conn,
        session: UnifiedSession,
        reuse_team: bool = False,
    ) -> str:
        """
        Process a query using self-organizing agent team.
        
        Args:
            query: User's query
            conn: Socket connection for client communication
            session: Session manager
            reuse_team: If True, reuse current team instead of creating new one
        
        Returns:
            Final response string
        """
        # Step 1: Design/reuse team
        if not reuse_team or not self.current_team_config:
            conn.send({
                "type": "team_design_started",
                "session_id": session.session_id,
                "message": "Analyzing query and designing agent team...",
            })
            
            # Have team creator design the team
            design_prompt = f"""Analyze this query and design the optimal agent team:

**User Query:**
{query}

Use the `setup_agent_team` tool to create your team.
"""
            
            await self.team_creator.process_task(design_prompt, conn, session)
            
            # Check if team was created via tool call
            # The tool stores config in state manager
            state_manager = get_state_manager()
            teams = state_manager.list_teams()
            
            if teams:
                # Get the most recently created team
                latest_team = state_manager.get_team(teams[-1])
                if latest_team:
                    self.setup_team(latest_team)
            
            # Fallback if no team created
            if not self.current_team_config:
                conn.send({
                    "type": "team_design_failed", 
                    "session_id": session.session_id,
                    "message": "Using fallback single-agent team",
                })
                
                fallback = AgentTeamConfig(
                    team_name="fallback",
                    agents=[AgentConfig(
                        name="assistant",
                        system_prompt="You are a helpful assistant. Answer the user's query thoroughly.",
                    )],
                    links=[],
                    entry_agent="assistant",
                )
                self.setup_team(fallback)
            
            conn.send({
                "type": "team_instantiated",
                "session_id": session.session_id,
                "team_name": self.current_team_config.team_name,
                "agents": self.list_agents(),
                "entry_agent": self.entry_agent,
            })
        else:
            conn.send({
                "type": "team_reused",
                "session_id": session.session_id,
                "team_name": self.current_team_config.team_name,
                "agents": self.list_agents(),
            })
        
        # Step 2: Execute query with entry agent
        conn.send({
            "type": "team_execution_started",
            "session_id": session.session_id,
            "entry_agent": self.entry_agent,
        })
        
        entry = self.agents.get(self.entry_agent)
        if not entry:
            error_msg = f"Error: Entry agent '{self.entry_agent}' not found"
            conn.send({
                "type": "error",
                "session_id": session.session_id,
                "message": error_msg,
            })
            return error_msg
        
        result = await entry.process_task(query, conn, session)
        
        conn.send({
            "type": "team_execution_completed",
            "session_id": session.session_id,
            "team_name": self.current_team_config.team_name,
        })
        
        return result
    
    def process_query_sync(
        self,
        query: str,
        conn,
        session: UnifiedSession,
        reuse_team: bool = False,
    ) -> str:
        """Synchronous wrapper for process_query"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.process_query(query, conn, session, reuse_team)
            )
        finally:
            loop.close()
    
    def clear_team(self):
        """Clear the current team"""
        self.agents.clear()
        self.current_team_config = None
        self.entry_agent = None
    
    def get_team_info(self) -> Dict:
        """Get current team information"""
        if not self.current_team_config:
            return {"team_name": None, "agents": [], "entry_agent": None}
        
        return {
            "team_name": self.current_team_config.team_name,
            "entry_agent": self.entry_agent,
            "agents": self.list_agents(),
            "links": [l.to_dict() for l in self.current_team_config.links],
        }


# =============================================================================
# LLMHandler-Compatible Wrapper
# =============================================================================

class SelfOrganizingLLMHandler:
    """Drop-in replacement for LLMHandler using self-organizing multi-agent system"""
    
    def __init__(self, api_key: str = None):
        self.playground = SelfOrganizingPlayground(api_key)
        self._reuse_team = False
    
    @property
    def client(self):
        return self.playground.client
    
    async def process_chat(self, session: UnifiedSession, conn):
        """Process chat with self-organizing team"""
        user_messages = [m for m in session.chat_history if m.get("role") == "user"]
        if not user_messages:
            return
        
        latest = user_messages[-1].get("content", "")
        result = await self.playground.process_query(
            latest, conn, session, reuse_team=self._reuse_team
        )
        session.add_message("assistant", result)
    
    def run_chat_sync(self, session: UnifiedSession, conn):
        """Synchronous chat processing"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.process_chat(session, conn))
        finally:
            loop.close()
    
    def set_reuse_team(self, reuse: bool):
        """Set whether to reuse team between messages"""
        self._reuse_team = reuse
    
    def get_current_team(self) -> Dict:
        return self.playground.get_team_info()
    
    def clear_team(self):
        self.playground.clear_team()