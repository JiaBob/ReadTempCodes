"""
Unified Session management for both LLM chat and pipeline tasks
"""
import time
import threading
import uuid
from typing import Any, Optional, Dict, List
from dataclasses import dataclass, field
from enum import Enum


class SessionType(Enum):
    CHAT = "chat"
    PIPELINE = "pipeline"
    HYBRID = "hybrid"  # Supports both chat and pipeline


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


@dataclass
class TaskInfo:
    """Information about a single task within a session"""
    task_id: str
    task_type: str  # "pipeline", "plugin", "batch_pipeline"
    status: TaskStatus = TaskStatus.PENDING
    worker: Any = None  # Process or Thread reference
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    metadata: Dict = field(default_factory=dict)
    result: Any = None
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "elapsed": (self.end_time or time.time()) - self.start_time,
            "is_alive": self.worker.is_alive() if self.worker else False,
            "metadata": self.metadata,
            "error": self.error
        }


class UnifiedSession:
    """
    Unified session that supports both LLM chat and pipeline task tracking.
    
    Usage Patterns:
    1. Chat-only: Client connects, sends chat messages, maintains history
    2. Pipeline-only: Client sends pipeline requests, tracks task status
    3. Hybrid: Client can do both (e.g., ask LLM to run a pipeline)
    """

    # Keys allowed by standard LLM APIs (OpenAI spec)
    _VALID_API_KEYS = {"role", "content", "name", "tool_calls", "tool_call_id"}

    def __init__(self, session_id: str, client_address: tuple,
                 session_type: SessionType = SessionType.HYBRID):
        self.session_id = session_id
        self.client_address = client_address
        self.session_type = session_type
        self.created_at = time.time()
        self.last_activity = time.time()

        # Chat state
        self.chat_history: List[Dict] = []

        # Pipeline task state
        self.tasks: Dict[str, TaskInfo] = {}

        # Shared metadata
        self.metadata: Dict[str, Any] = {}

        # Thread safety
        self.lock = threading.RLock()  # RLock allows nested locking
        
        # Reference to terminate function (injected to avoid circular import)
        self._terminate_process_func = None

    def set_terminate_func(self, func):
        """Set the terminate process function"""
        self._terminate_process_func = func

    # =========================================================================
    # Activity Tracking
    # =========================================================================

    def touch(self):
        """Update last activity timestamp"""
        with self.lock:
            self.last_activity = time.time()

    def is_idle(self, timeout_seconds: float = 3600) -> bool:
        """Check if session has been idle for longer than timeout"""
        with self.lock:
            return (time.time() - self.last_activity) > timeout_seconds

    # =========================================================================
    # Chat Methods (LLM)
    # =========================================================================

    def add_message(self, role: str, content: str, **kwargs):
        """Add a message to chat history"""
        with self.lock:
            self.touch()
            message = {
                "role": role,
                "content": content,
                "timestamp": time.time(),
                **kwargs
            }
            self.chat_history.append(message)

    def get_history(self, limit: Optional[int] = None) -> List[Dict]:
        """Get full chat history (for UI/logging)"""
        with self.lock:
            history = self.chat_history[-limit:] if limit else self.chat_history
            return [msg.copy() for msg in history]

    def get_openai_context(self, limit: Optional[int] = None) -> List[Dict]:
        """Get sanitized chat history for LLM API"""
        with self.lock:
            history_slice = self.chat_history[-limit:] if limit else self.chat_history
            return [
                {k: v for k, v in msg.items() if k in self._VALID_API_KEYS and v is not None}
                for msg in history_slice
            ]

    def clear_history(self):
        """Clear chat history"""
        with self.lock:
            self.chat_history.clear()
            self.touch()

    # =========================================================================
    # Task Methods (Pipeline)
    # =========================================================================

    def create_task(self, task_type: str, metadata: Optional[Dict] = None) -> str:
        """
        Create a new task within this session.
        
        Args:
            task_type: Type of task ("pipeline", "plugin", "batch_pipeline")
            metadata: Optional metadata for the task
            
        Returns:
            task_id: Unique identifier for this task
        """
        task_id = str(uuid.uuid4())

        with self.lock:
            self.touch()
            self.tasks[task_id] = TaskInfo(
                task_id=task_id,
                task_type=task_type,
                metadata=metadata or {}
            )

        return task_id

    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """Get task info by ID"""
        with self.lock:
            return self.tasks.get(task_id)

    def set_task_worker(self, task_id: str, worker):
        """Set the worker (process/thread) for a task"""
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].worker = worker
                self.tasks[task_id].status = TaskStatus.RUNNING

    def update_task_status(self, task_id: str, status: TaskStatus,
                           result: Any = None, error: str = None):
        """Update task status"""
        with self.lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = status
                if status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.TERMINATED):
                    task.end_time = time.time()
                if result is not None:
                    task.result = result
                if error is not None:
                    task.error = error
                self.touch()

    def terminate_task(self, task_id: str) -> Dict:
        """Terminate a specific task"""
        with self.lock:
            if task_id not in self.tasks:
                return {"success": False, "message": f"Task {task_id} not found"}

            task = self.tasks[task_id]

            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.TERMINATED):
                return {"success": False, "message": f"Task {task_id} already finished"}

            if not task.worker or not task.worker.is_alive():
                task.status = TaskStatus.COMPLETED
                return {"success": False, "message": f"Task {task_id} is not running"}

        # Terminate outside lock to avoid deadlock
        try:
            if self._terminate_process_func:
                self._terminate_process_func(task.worker)
            else:
                # Fallback
                task.worker.terminate()
                task.worker.join(timeout=5)
                if task.worker.is_alive():
                    task.worker.kill()

            with self.lock:
                task.status = TaskStatus.TERMINATED
                task.end_time = time.time()

            return {
                "success": True,
                "message": f"Task {task_id} terminated",
                "elapsed": task.end_time - task.start_time
            }
        except Exception as e:
            return {"success": False, "message": f"Failed to terminate: {str(e)}"}

    def list_tasks(self, status_filter: Optional[TaskStatus] = None) -> Dict[str, Dict]:
        """List all tasks in this session"""
        with self.lock:
            result = {}
            for task_id, task in self.tasks.items():
                if status_filter is None or task.status == status_filter:
                    result[task_id] = task.to_dict()
            return result

    def get_running_tasks(self) -> List[str]:
        """Get list of running task IDs"""
        with self.lock:
            return [
                task_id for task_id, task in self.tasks.items()
                if task.status == TaskStatus.RUNNING and task.worker and task.worker.is_alive()
            ]

    def cleanup_finished_tasks(self, max_age_seconds: float = 3600) -> int:
        """Remove finished tasks older than max_age"""
        with self.lock:
            current_time = time.time()
            to_remove = []

            for task_id, task in self.tasks.items():
                if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.TERMINATED):
                    if task.end_time and (current_time - task.end_time) > max_age_seconds:
                        to_remove.append(task_id)

            for task_id in to_remove:
                del self.tasks[task_id]

            return len(to_remove)

    def remove_task(self, task_id: str):
        """Remove a task from tracking"""
        with self.lock:
            self.tasks.pop(task_id, None)

    # =========================================================================
    # Termination
    # =========================================================================

    def terminate_all_tasks(self) -> List[Dict]:
        """Terminate all running tasks in this session"""
        running_task_ids = self.get_running_tasks()

        results = []
        for task_id in running_task_ids:
            result = self.terminate_task(task_id)
            results.append({"task_id": task_id, **result})

        return results

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict:
        """Convert session to dictionary"""
        with self.lock:
            running_tasks = self.get_running_tasks()
            return {
                "session_id": self.session_id,
                "client_address": self.client_address,
                "session_type": self.session_type.value,
                "created_at": self.created_at,
                "last_activity": self.last_activity,
                "idle_time": time.time() - self.last_activity,
                "message_count": len(self.chat_history),
                "total_tasks": len(self.tasks),
                "running_tasks": len(running_tasks),
                "metadata": self.metadata
            }


class SessionManager:
    """Manages multiple unified sessions"""

    def __init__(self, session_timeout: float = 3600, cleanup_interval: float = 300,
                 terminate_process_func=None):
        self.sessions: Dict[str, UnifiedSession] = {}
        self.lock = threading.RLock()
        self.session_timeout = session_timeout
        self.cleanup_interval = cleanup_interval
        self._terminate_process_func = terminate_process_func

        # Start cleanup thread
        self._stop_cleanup = threading.Event()
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

    def _cleanup_loop(self):
        """Periodically cleanup idle sessions and finished tasks"""
        while not self._stop_cleanup.wait(self.cleanup_interval):
            self.cleanup_idle_sessions()
            # Also cleanup finished tasks in active sessions
            with self.lock:
                for session in self.sessions.values():
                    session.cleanup_finished_tasks()

    def stop_cleanup(self):
        """Stop the cleanup thread"""
        self._stop_cleanup.set()

    def create_session(self, client_address: tuple,
                       session_type: SessionType = SessionType.HYBRID) -> UnifiedSession:
        """Create a new session"""
        session_id = str(uuid.uuid4())

        with self.lock:
            session = UnifiedSession(session_id, client_address, session_type)
            session.set_terminate_func(self._terminate_process_func)
            self.sessions[session_id] = session

        return session

    def get_session(self, session_id: str) -> Optional[UnifiedSession]:
        """Get session by ID"""
        with self.lock:
            return self.sessions.get(session_id)

    def get_or_create_session(self, session_id: Optional[str],
                              client_address: tuple,
                              session_type: SessionType = SessionType.HYBRID) -> UnifiedSession:
        """Get existing session or create new one"""
        if session_id:
            session = self.get_session(session_id)
            if session:
                session.touch()
                return session

        return self.create_session(client_address, session_type)

    def remove_session(self, session_id: str) -> bool:
        """Remove a session (terminates all its tasks first)"""
        with self.lock:
            session = self.sessions.get(session_id)
            if not session:
                return False

        # Terminate all tasks before removing
        session.terminate_all_tasks()

        with self.lock:
            self.sessions.pop(session_id, None)

        return True

    def cleanup_idle_sessions(self) -> int:
        """Remove sessions that have been idle too long"""
        with self.lock:
            idle_sessions = [
                sid for sid, session in self.sessions.items()
                if session.is_idle(self.session_timeout) and not session.get_running_tasks()
            ]

        removed = 0
        for session_id in idle_sessions:
            if self.remove_session(session_id):
                removed += 1

        return removed

    def list_sessions(self) -> List[Dict]:
        """List all sessions"""
        with self.lock:
            return [session.to_dict() for session in self.sessions.values()]

    def get_session_by_task(self, task_id: str) -> Optional[UnifiedSession]:
        """Find which session owns a specific task"""
        with self.lock:
            for session in self.sessions.values():
                if task_id in session.tasks:
                    return session
        return None

    def terminate_task(self, task_id: str) -> Dict:
        """Terminate a task by ID (searches all sessions)"""
        session = self.get_session_by_task(task_id)
        if not session:
            return {"success": False, "message": f"Task {task_id} not found in any session"}

        result = session.terminate_task(task_id)
        result["session_id"] = session.session_id
        return result

    def get_task(self, task_id: str) -> Optional[Dict]:
        """Get task info by ID (searches all sessions)"""
        session = self.get_session_by_task(task_id)
        if not session:
            return None
        
        task = session.get_task(task_id)
        if task:
            task_dict = task.to_dict()
            task_dict["session_id"] = session.session_id
            return task_dict
        return None

    def list_all_tasks(self) -> Dict[str, Dict]:
        """List all tasks across all sessions"""
        with self.lock:
            all_tasks = {}
            for session in self.sessions.values():
                for task_id, task in session.tasks.items():
                    task_dict = task.to_dict()
                    task_dict["session_id"] = session.session_id
                    all_tasks[task_id] = task_dict
            return all_tasks

    def terminate_all(self) -> List[Dict]:
        """Terminate all tasks in all sessions"""
        with self.lock:
            session_ids = list(self.sessions.keys())

        results = []
        for session_id in session_ids:
            session = self.get_session(session_id)
            if session:
                session_results = session.terminate_all_tasks()
                for r in session_results:
                    r["session_id"] = session_id
                results.extend(session_results)

        return results

    def get_stats(self) -> Dict:
        """Get overall statistics"""
        with self.lock:
            total_sessions = len(self.sessions)
            total_tasks = sum(len(s.tasks) for s in self.sessions.values())
            running_tasks = sum(len(s.get_running_tasks()) for s in self.sessions.values())
            total_messages = sum(len(s.chat_history) for s in self.sessions.values())

            by_type = {
                SessionType.CHAT.value: 0,
                SessionType.PIPELINE.value: 0,
                SessionType.HYBRID.value: 0
            }
            for session in self.sessions.values():
                by_type[session.session_type.value] += 1

            return {
                "total_sessions": total_sessions,
                "sessions_by_type": by_type,
                "total_tasks": total_tasks,
                "running_tasks": running_tasks,
                "total_messages": total_messages,
                "session_timeout": self.session_timeout
            }