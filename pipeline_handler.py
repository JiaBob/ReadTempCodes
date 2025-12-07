"""
Pipeline execution handler
"""
import sys
import os
import time
import queue
import threading
import traceback
import multiprocessing
from multiprocessing.managers import SyncManager

# Your existing imports
import pmtpy_loader
from plugin_validation import plugin_validation
from run_pipeline import run_pipeline, validate, reset_id
from sandbox import execute_plugin_in_sandbox
from pipeline_core import Workflow
from init_plugin import init_plugin

from process_utils import (
    subprocess_cleanup_monitor,
    RedirectOutputToQueue,
    ProcessPool,
    task_observer
)
from session_manager import SessionManager, UnifiedSession, TaskStatus
from config import MAX_PROCESS_WORKERS


def _pipeline_worker(shared_dict, output_queue, task_id=None):
    """Worker function for parallel batch processing"""
    if 'parent_pid' in shared_dict:
        subprocess_cleanup_monitor(shared_dict['parent_pid'])

    with RedirectOutputToQueue(output_queue, task_id):
        start_time = time.time()
        try:
            pipeline_json = shared_dict[task_id]
            mode = shared_dict['mode']
            output_path = shared_dict['output_path']
            debug = shared_dict['debug']
            log_level = shared_dict['log_level']

            run_pipeline(pipeline_json, mode, output_path, debug, log_level)

            execution_time = time.time() - start_time

            output_queue.put({
                "type": "result",
                "task_id": task_id,
                "status": "success",
                "execution_time": execution_time,
                "message": f"Task {task_id} completed successfully"
            })

        except Exception as e:
            execution_time = time.time() - start_time
            output_queue.put({
                "type": "error",
                "task_id": task_id,
                "status": "error",
                "execution_time": execution_time,
                "message": str(e),
                "traceback": traceback.format_exc()
            })


def _plugin_worker(plugin_name, args, cache_dir, output_queue):
    """Worker function for plugin execution"""
    with RedirectOutputToQueue(output_queue):
        try:
            execute_plugin_in_sandbox(plugin_name, args, cache_dir)

            output_queue.put({
                "type": "result",
                "status": "success",
                "message": "Plugin completed successfully"
            })
        except Exception as e:
            output_queue.put({
                "type": "error",
                "status": "error",
                "message": str(e),
                "traceback": traceback.format_exc()
            })


class PipelineHandler:
    """Handles pipeline and plugin execution using unified sessions"""

    def __init__(self, process_pool: ProcessPool, session_manager: SessionManager,
                 manager: SyncManager, shutdown_flag: threading.Event):
        self.process_pool = process_pool
        self.session_manager = session_manager
        self.manager = manager
        self.shutdown_flag = shutdown_flag
        self.lock = threading.Lock()
        self.active_tasks = 0


    @task_observer
    def execute_batch_parallel(self, session: UnifiedSession, params):
        """Execute multiple pipelines in parallel using ProcessPool"""

        shared_dict = self.manager.dict()
        shared_dict['parent_pid'] = os.getpid()
        shared_dict['mode'] = params.get('mode', 'complete')
        shared_dict['output_path'] = params.get('output_path', 'test_cache')
        shared_dict['debug'] = params.get('debug', False)
        shared_dict['log_level'] = params.get('log_level', 'info')
        pipeline_json_list = params.get("pipeline_json_list", [])

        output_queue = multiprocessing.Queue()

        total_pipelines = len(pipeline_json_list)
        pipeline_tasks = {}  # pipeline_index -> task_id
        submitted_processes = {}  # pipeline_index -> process
        next_to_submit = 0

        while next_to_submit < total_pipelines or len(submitted_processes) > 0:
            if self.shutdown_flag.is_set():
                break

            while next_to_submit < total_pipelines:
                idx = next_to_submit

                # Create task in session
                task_id = session.create_task(
                    "batch_pipeline",
                    {"pipeline_index": idx, "total_batch": total_pipelines}
                )
                shared_dict[task_id] = pipeline_json_list[idx]
                process = self.process_pool.submit(
                    target=_pipeline_worker,
                    args=(shared_dict, output_queue, task_id)
                )

                if process is None:  # No capacity
                    # Remove the task we just created since we couldn't start it
                    session.remove_task(task_id)
                    break

                # Set worker on session task
                session.set_task_worker(task_id, process)
                pipeline_tasks[idx] = task_id
                submitted_processes[idx] = process

                yield {
                    "type": "task_started",
                    "task_id": task_id,
                    "session_id": session.session_id,
                    "pipeline_index": idx,
                    "message": f"Pipeline {idx} task started"
                }

                next_to_submit += 1

            # Clean up finished processes
            for idx in list(submitted_processes.keys()):
                if not submitted_processes[idx].is_alive():
                    del submitted_processes[idx]

            # Collect messages
            try:
                message = output_queue.get(timeout=0.1)
                
                task_id = message.get("task_id")
                message["session_id"] = session.session_id

                yield message

                # Update task status
                if task_id:
                    if message.get("type") == "result":
                        session.update_task_status(task_id, TaskStatus.COMPLETED)
                    elif message.get("type") == "error":
                        session.update_task_status(task_id, TaskStatus.FAILED,
                                                   error=message.get("message"))

            except queue.Empty:
                continue

        # Drain remaining messages
        try:
            while True:
                message = output_queue.get_nowait()
                task_id = message.get("task_id")
                message["session_id"] = session.session_id
                
                if task_id:
                    if message.get("type") == "result":
                        session.update_task_status(task_id, TaskStatus.COMPLETED)
                    elif message.get("type") == "error":
                        session.update_task_status(task_id, TaskStatus.FAILED,
                                                   error=message.get("message"))
                
                yield message
        except queue.Empty:
            pass

    @task_observer
    def execute_plugin(self, session: UnifiedSession, plugin_name, args, cache_dir):
        """Execute a plugin in sandboxed environment"""
        
        # Create task in session
        task_id = session.create_task(
            "plugin",
            {"plugin_name": plugin_name, "args": args, "cache_dir": cache_dir}
        )

        yield {
            "type": "task_started",
            "task_id": task_id,
            "session_id": session.session_id,
            "message": "Plugin task started"
        }

        output_queue = multiprocessing.Queue()

        process = self.process_pool.submit(
            target=_plugin_worker,
            args=(plugin_name, args, cache_dir, output_queue)
        )

        if process is None:
            session.update_task_status(task_id, TaskStatus.FAILED, 
                                       error="No process capacity available")
            yield {
                "type": "error",
                "task_id": task_id,
                "session_id": session.session_id,
                "status": "error",
                "message": "No process capacity available"
            }
            return

        # Set worker on session task
        session.set_task_worker(task_id, process)

        while not self.shutdown_flag.is_set():
            try:
                message = output_queue.get(timeout=0.1)
                message["task_id"] = task_id
                message["session_id"] = session.session_id
                yield message

                if message.get("type") == "result":
                    session.update_task_status(task_id, TaskStatus.COMPLETED)
                    break
                elif message.get("type") == "error":
                    session.update_task_status(task_id, TaskStatus.FAILED,
                                               error=message.get("message"))
                    break

            except queue.Empty:
                if not process.is_alive():
                    session.update_task_status(task_id, TaskStatus.FAILED,
                                               error="Process terminated unexpectedly")
                    yield {
                        "type": "error",
                        "task_id": task_id,
                        "session_id": session.session_id,
                        "status": "error",
                        "message": "Process terminated unexpectedly"
                    }
                    break
                continue

        process.join(timeout=2)
        if process.is_alive():
            process.terminate()

    def handle_plugin_validation(self, params):
        """Handle plugin validation request"""
        name = params.get("plugin", None)
        from_file = params.get("from_file", False)
        try:
            if from_file:
                results = plugin_validation(None, from_file)
            else:
                results = plugin_validation(name, None)
            return iter([{
                "type": "result",
                "status": "success",
                "message": results
            }])
        except Exception as e:
            return iter([{
                "type": "error",
                "status": "error",
                "message": str(e)
            }])

    def handle_init_plugin(self, params):
        """Handle init plugin request"""
        path = params.get("path", None)
        description = params.get("description", None)
        version = params.get("version", None)
        author = params.get("author", None)

        try:
            init_plugin(path, description, version, author)
            return iter([{
                "type": "result",
                "status": "success",
                "message": f"Plugin initialized on {path}"
            }])
        except Exception as e:
            return iter([{
                "type": "error",
                "status": "error",
                "message": str(e)
            }])

    def handle_reset_id(self, params):
        """Handle reset_id request"""
        pipeline_json = params.get("pipeline_json")
        id_val = params.get("id")
        return iter([{
            "type": "result",
            "status": "success",
            "message": reset_id(pipeline_json, id_val)
        }])

    def handle_validate(self, params):
        """Handle validate request"""
        pipeline_json = params.get("pipeline_json")
        return iter([{
            "type": "result",
            "status": "success",
            "message": validate(pipeline_json)
        }])