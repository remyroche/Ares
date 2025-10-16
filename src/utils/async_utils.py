from collections.abc import Coroutine, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Union
import asyncio
import contextlib
import json
import os
import signal
import time

import aiofiles

from .logger import system_logger
from .warning_symbols import invalid, failed, missing

from logging import warning
from src.utils.decorators import handles_errors
# src/utils/async_utils.py

class AsyncFileManager:
    """
    Enhanced async file manager with comprehensive error handling and type safety.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize async file manager with enhanced type safety.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("AsyncFileManager")

        # File management
        self.file_cache: dict[str, Any] = {}
        self.max_cache_size: int = 100
        self.cache_enabled: bool = True

        # Configuration
        self.file_config: dict[str, Any] = self.config.get("async_file_manager", {})
        self.max_cache_size = int(self.file_config.get("max_cache_size", 100))
        self.cache_enabled = bool(self.file_config.get("cache_enabled", True))
        self.default_encoding: str = str(
            self.file_config.get("default_encoding", "utf-8")
        )

    @handles_errors(
        error_handlers={
            ValueError: (False, "Invalid async file manager configuration"),
            AttributeError: (False, "Missing required file parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return = False,
        context="async file manager initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize async file manager with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        self.logger.info("Initializing Async File Manager...")

        # Load file configuration
        await self._load_file_configuration()

        # Validate configuration
        if not self._validate_configuration():
            self.logger.error(invalid("Invalid configuration for async file manager"))
            return False

        self.logger.info("✅ Async File Manager initialization completed successfully")
        return True

    @handles_errors(fallback = None)
    async def _load_file_configuration(self) -> None:
        """Load file configuration."""
        # Set default file parameters
        self.file_config.setdefault("max_cache_size", 100)
        self.file_config.setdefault("cache_enabled", True)
        self.file_config.setdefault("default_encoding", "utf-8")
        self.file_config.setdefault("chunk_size", 8192)
        self.file_config.setdefault("timeout", 30)

        # Update configuration
        self.max_cache_size = int(self.file_config["max_cache_size"])
        self.cache_enabled = bool(self.file_config["cache_enabled"])
        self.default_encoding = str(self.file_config["default_encoding"])

        self.logger.info("File configuration loaded successfully")

    @handles_errors(fallback = False)
    def _validate_configuration(self) -> bool:
        """
        Validate file configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        # Validate cache size
        if self.max_cache_size <= 0:
            self.logger.error(invalid("Invalid max cache size"))
            return False

        # Validate encoding
        if not self.default_encoding:
            self.logger.error(invalid("Invalid default encoding"))
            return False

        self.logger.info("Configuration validation successful")
        return True

    @handles_errors(
        default_return = None,
        context="file reading",
    )
    async def read_file(
        self, file_path: str, encoding: str | None = None
    ) -> str | None:
        """
        Read file asynchronously.

        Args:
            file_path: Path to the file
            encoding: File encoding (defaults to configured encoding)

        Returns:
            Optional[str]: File content or None if failed
        """
        # Check cache first
        if self.cache_enabled and file_path in self.file_cache:
            self.logger.info(f"Reading {file_path} from cache")
            return str(self.file_cache[file_path])

        # Read file
        chosen_encoding = encoding or self.default_encoding
        async with aiofiles.open(file_path, encoding = chosen_encoding) as f:
            content = await f.read()

        # Cache the content
        if self.cache_enabled:
            self._add_to_cache(file_path, content)

        self.logger.info(f"Read file: {file_path}")
        return content

    @handles_errors(
        default_return = False,
        context="file writing",
    )
    async def write_file(
        self, file_path: str, content: str, encoding: str | None = None
    ) -> bool:
        """
        Write file asynchronously.

        Args:
            file_path: Path to the file
            content: Content to write
            encoding: File encoding (defaults to configured encoding)

        Returns:
            bool: True if successful, False otherwise
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok = True)

        # Write file
        chosen_encoding = encoding or self.default_encoding
        async with aiofiles.open(file_path, "w", encoding = chosen_encoding) as f:
            await f.write(content)

        # Update cache
        if self.cache_enabled:
            self._add_to_cache(file_path, content)

        self.logger.info(f"Wrote file: {file_path}")
        return True

    @handles_errors(
        default_return = None,
        context="JSON file reading",
    )
    async def read_json(self, file_path: str) -> dict[str, Any] | None:
        """
        Read JSON file asynchronously.

        Args:
            file_path: Path to the JSON file

        Returns:
            Optional[Dict[str, Any]]: JSON data or None if failed
        """
        content = await self.read_file(file_path)
        if content is None:
            # Fallback implementation for content
            return None

        data: dict[str, Any] = json.loads(content)
        self.logger.info(f"Read JSON file: {file_path}")
        return data

    @handles_errors(
        default_return = False,
        context="JSON file writing",
    )
    async def write_json(
        self, file_path: str, data: dict[str, Any], indent: int = 2
    ) -> bool:
        """
        Write JSON file asynchronously.

        Args:
            file_path: Path to the JSON file
            data: Data to write
            indent: JSON indentation

        Returns:
            bool: True if successful, False otherwise
        """
        content = json.dumps(data, indent = indent, default = str)
        success = await self.write_file(file_path, content)
        if success:
            self.logger.info(f"Wrote JSON file: {file_path}")
        return success

    @handles_errors(fallback = None)
    def _add_to_cache(self, file_path: str, content: str) -> None:
        """
        Add file content to cache.

        Args:
            file_path: File path
            content: File content
        """
        # Remove oldest entry if cache is full
        if len(self.file_cache) >= self.max_cache_size:
            oldest_key = next(iter(self.file_cache))
            del self.file_cache[oldest_key]
            self.logger.debug(f"Removed {oldest_key} from cache")

        # Add to cache
        self.file_cache[file_path] = content
        self.logger.debug(f"Added {file_path} to cache")

    @handles_errors(fallback = None)
    def clear_cache(self) -> None:
        """Clear the file cache."""
        cache_size = len(self.file_cache)
        self.file_cache.clear()
        self.logger.info(f"Cleared cache ({cache_size} entries)")

    def get_cache_status(self) -> dict[str, Any]:
        """
        Get cache status information.

        Returns:
            Dict[str, Any]: Cache status
        """
        return {
            "cache_enabled": self.cache_enabled,
            "max_cache_size": self.max_cache_size,
            "current_cache_size": len(self.file_cache),
            "cached_files": list(self.file_cache.keys()),
        }

    @handles_errors(fallback = None)
    async def stop(self) -> None:
        """Stop the async file manager."""
        self.logger.info("🛑 Stopping Async File Manager...")
        self.clear_cache()
        self.logger.info("✅ Async File Manager stopped successfully")

class AsyncTaskManager:
    """
    Enhanced async task manager with comprehensive error handling and type safety.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize async task manager with enhanced type safety.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("AsyncTaskManager")

        # Task management
        self.active_tasks: dict[str, asyncio.Task[Any]] = {}
        self.task_results: dict[str, Any] = {}
        self.max_concurrent_tasks: int = 10

        # Configuration
        self.task_config: dict[str, Any] = self.config.get("async_task_manager", {})
        self.max_concurrent_tasks = int(
            self.task_config.get("max_concurrent_tasks", 10)
        )
        self.task_timeout: int = int(self.task_config.get("task_timeout", 300))

    @handles_errors(
        error_handlers={
            ValueError: (False, "Invalid async task manager configuration"),
            AttributeError: (False, "Missing required task parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return = False,
        context="async task manager initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize async task manager with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        self.logger.info("Initializing Async Task Manager...")

        # Load task configuration
        await self._load_task_configuration()

        # Validate configuration
        if not self._validate_configuration():
            self.logger.error(invalid("Invalid configuration for async task manager"))
            return False

        self.logger.info("✅ Async Task Manager initialization completed successfully")
        return True

    @handles_errors(fallback = None)
    async def _load_task_configuration(self) -> None:
        """Load task configuration."""
        # Set default task parameters
        self.task_config.setdefault("max_concurrent_tasks", 10)
        self.task_config.setdefault("task_timeout", 300)
        self.task_config.setdefault("enable_task_monitoring", True)
        self.task_config.setdefault("auto_cleanup_failed_tasks", True)

        # Update configuration
        self.max_concurrent_tasks = int(self.task_config["max_concurrent_tasks"])
        self.task_timeout = int(self.task_config["task_timeout"])

        self.logger.info("Task configuration loaded successfully")

    @handles_errors(fallback = False)
    def _validate_configuration(self) -> bool:
        """
        Validate task configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        # Validate max concurrent tasks
        if self.max_concurrent_tasks <= 0:
            self.logger.error(invalid("Invalid max concurrent tasks"))
            return False

        # Validate task timeout
        if self.task_timeout <= 0:
            self.logger.error(invalid("Invalid task timeout"))
            return False

        self.logger.info("Configuration validation successful")
        return True

    @handles_errors(fallback = None)
    async def execute_task(
        self,
        task_name: str,
        coro: Coroutine[Any, Any, Any],
        timeout: int | None = None,
    ) -> Any | None:
        """
        Execute a task with timeout and error handling.

        Args:
            task_name: Name of the task
            coro: Coroutine to execute
            timeout: Task timeout (defaults to configured timeout)

        Returns:
            Optional[Any]: Task result or None if failed
        """
        # Check if we can run more tasks
        if len(self.active_tasks) >= self.max_concurrent_tasks:
            self.logger.warning(
                f"Maximum concurrent tasks reached ({self.max_concurrent_tasks})",
            )
            return None

        # Create task
        chosen_timeout = timeout or self.task_timeout
        task = asyncio.create_task(coro, name = task_name)
        self.active_tasks[task_name] = task

        self.logger.info(f"Started task: {task_name}")

        try:
            # Execute with timeout
            result = await asyncio.wait_for(task, timeout = chosen_timeout)
            self.task_results[task_name] = result
            self.logger.info(f"Task completed: {task_name}")
            return result
        except asyncio.TimeoutError:
            self.logger.warning(f"Task timed out after {chosen_timeout}s: {task_name}")
            if not task.done():
                task.cancel()
            try:
                await task  # Wait for cancellation to complete
            except asyncio.CancelledError:
                pass  # Expected when cancelling
            return None
        except asyncio.CancelledError:
            self.logger.info(f"Task was cancelled: {task_name}")
            return None
        except (AttributeError, TypeError) as e:
            self.logger.error(f"Task configuration error for {task_name}: {e}")
            if not task.done():
                task.cancel()
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in task {task_name}: {e}")
            if not task.done():
                task.cancel()
            return None
        finally:
            # Remove from active tasks
            if task_name in self.active_tasks:
                del self.active_tasks[task_name]

    @handles_errors(fallback = False)
    async def cancel_task(self, task_name: str) -> bool:
        """
        Cancel a running task.

        Args:
            task_name: Name of the task to cancel

        Returns:
            bool: True if successful, False otherwise
        """
        if task_name not in self.active_tasks:
            self.logger.warning(missing(f"Task not found: {task_name}"))
            return False

        task = self.active_tasks[task_name]
        task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await task

        del self.active_tasks[task_name]
        self.logger.info(f"Cancelled task: {task_name}")
        return True

    @handles_errors(fallback = None)
    async def cancel_all_tasks(self) -> None:
        """Cancel all running tasks."""
        if not self.active_tasks:
            self.logger.info("No active tasks to cancel")
            return

        self.logger.info(f"Cancelling {len(self.active_tasks)} active tasks...")

        for task in list(self.active_tasks.values()):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        self.active_tasks.clear()
        self.logger.info("All tasks cancelled")

    def get_task_status(self) -> dict[str, Any]:
        """
        Get task manager status information.

        Returns:
            Dict[str, Any]: Task manager status
        """
        return {
            "active_tasks_count": len(self.active_tasks),
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "task_timeout": self.task_timeout,
            "active_task_names": list(self.active_tasks.keys()),
            "completed_tasks_count": len(self.task_results),
        }

    @handles_errors(fallback = None)
    async def stop(self) -> None:
        """Stop the async task manager."""
        self.logger.info("🛑 Stopping Async Task Manager...")
        await self.cancel_all_tasks()
        self.task_results.clear()
        self.logger.info("✅ Async Task Manager stopped successfully")

# Global instances
async_file_manager: AsyncFileManager | None = None
async_task_manager: AsyncTaskManager | None = None

class AsyncProcessesManager:
    """
    Manager for async processes with comprehensive error handling.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = config or {}
        self.logger = system_logger.getChild("AsyncProcessesManager")
        self.processes: dict[str, asyncio.subprocess.Process] = {}
        self.max_processes: int = int(self.config.get("max_processes", 10))

    async def start_process(
        self,
        name: str,
        command: list[str],
        cwd: str | None = None,
    ) -> asyncio.subprocess.Process | None:
        """Start an async process."""
        if len(self.processes) >= self.max_processes:
            self.logger.warning(
                warning(f"Maximum processes ({self.max_processes}) reached")
            )
            return None

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd = cwd,
                stdout = asyncio.subprocess.PIPE,
                stderr = asyncio.subprocess.PIPE,
            )
        except Exception as e:  # noqa: BLE001
            self.logger.exception(failed(f"Failed to start process '{name}': {e}"))
            return None

        self.processes[name] = process
        self.logger.info(f"Started process '{name}' with PID {process.pid}")
        return process

    async def stop_process(self, name: str) -> bool:
        """Stop a specific process."""
        if name not in self.processes:
            self.logger.warning(missing(f"Process '{name}' not found"))
            return False

        process = self.processes[name]
        try:
            process.terminate()
            await asyncio.wait_for(process.wait(), timeout = 5.0)
        except TimeoutError:
            process.kill()
            await process.wait()
        except Exception as e:  # noqa: BLE001
            self.logger.exception(failed(f"Failed to stop process '{name}': {e}"))
            return False

        del self.processes[name]
        self.logger.info(f"Stopped process '{name}'")
        return True

    async def stop_all_processes(self) -> None:
        """Stop all managed processes."""
        for name in list(self.processes.keys()):
            await self.stop_process(name)

    def get_process_status(self) -> dict[str, Any]:
        """Get status of all processes."""
        return {
            "total_processes": len(self.processes),
            "max_processes": self.max_processes,
            "processes": {
                name: {"pid": proc.pid, "returncode": proc.returncode}
                for name, proc in self.processes.items()
            },
        }

# Create a global instance for backward compatibility
async_processes_manager = AsyncProcessesManager()

# =============================================================================
# ENHANCED ASYNC UTILITIES
# =============================================================================

class TaskState(Enum):
    """Enumeration of task execution states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class TaskResult:
    """Result container for async task execution."""
    task_name: str
    state: TaskState
    result: Any = None
    error: Optional[Exception] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    execution_time: Optional[float] = None

    def __post_init__(self):
        if self.end_time:
            self.execution_time = self.end_time - self.start_time

class EnhancedAsyncManager:
    """Enhanced async task manager with comprehensive error handling and cancellation support."""

    def __init__(self, max_concurrent_tasks: int = 10, default_timeout: float = 300.0):
        """Initialize enhanced async manager.

        Args:
            max_concurrent_tasks: Maximum number of concurrent tasks
            default_timeout: Default timeout in seconds for tasks
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.default_timeout = default_timeout
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.task_results: Dict[str, TaskResult] = {}
        self.logger = system_logger.getChild("EnhancedAsyncManager")

        # Graceful shutdown handling
        self._shutdown_event = asyncio.Event()
        self._cleanup_tasks: Set[asyncio.Task] = set()

    async def execute_task(self,
                          task_name: str,
                          coro: Coroutine[Any, Any, Any],
                          timeout: Optional[float] = None,
                          priority: int = 0) -> TaskResult:
        """Execute a coroutine with enhanced error handling and cancellation support.

        Args:
            task_name: Unique name for the task
            coro: Coroutine to execute
            timeout: Optional timeout in seconds
            priority: Task priority (higher = more important)

        Returns:
            TaskResult: Execution result with state and metadata
        """
        timeout = timeout or self.default_timeout
        start_time = time.time()

        # Check if task is already running
        if task_name in self.active_tasks:
            existing_result = TaskResult(
                task_name=task_name,
                state=TaskState.FAILED,
                error=RuntimeError(f"Task '{task_name}' is already running")
            )
            self.task_results[task_name] = existing_result
            return existing_result

        # Create task result placeholder
        task_result = TaskResult(task_name=task_name, state=TaskState.PENDING)
        self.task_results[task_name] = task_result

        async def _wrapped_task():
            """Wrapped task with comprehensive error handling."""
            try:
                async with self.semaphore:
                    task_result.state = TaskState.RUNNING
                    task_result.start_time = time.time()

                    try:
                        if timeout:
                            result = await asyncio.wait_for(coro, timeout=timeout)
                        else:
                            result = await coro

                        task_result.state = TaskState.COMPLETED
                        task_result.result = result
                        task_result.end_time = time.time()

                        self.logger.info(f"Task completed successfully: {task_name}")
                        return result

                    except asyncio.TimeoutError:
                        task_result.state = TaskState.TIMEOUT
                        task_result.error = asyncio.TimeoutError(f"Task timed out after {timeout}s")
                        task_result.end_time = time.time()
                        self.logger.warning(f"Task timed out: {task_name}")
                        raise

                    except asyncio.CancelledError:
                        task_result.state = TaskState.CANCELLED
                        task_result.error = asyncio.CancelledError("Task was cancelled")
                        task_result.end_time = time.time()
                        self.logger.info(f"Task was cancelled: {task_name}")
                        raise

                    except (AttributeError, TypeError) as e:
                        task_result.state = TaskState.FAILED
                        task_result.error = e
                        task_result.end_time = time.time()
                        self.logger.error(f"Task configuration error: {task_name} - {e}")
                        raise

                    except Exception as e:
                        task_result.state = TaskState.FAILED
                        task_result.error = e
                        task_result.end_time = time.time()
                        self.logger.error(f"Task failed: {task_name} - {e}")
                        raise

            except Exception as e:
                if not hasattr(task_result, 'error') or task_result.error is None:
                    task_result.error = e
                if task_result.end_time is None:
                    task_result.end_time = time.time()

                if isinstance(e, (asyncio.TimeoutError, asyncio.CancelledError)):
                    raise  # Re-raise cancellation/timeout errors
                else:
                    self.logger.error(f"Unexpected error in task wrapper: {e}")

        # Create and start the task
        try:
            task = asyncio.create_task(_wrapped_task())
            self.active_tasks[task_name] = task

            # Wait for completion with proper cleanup
            try:
                await task
            except asyncio.CancelledError:
                # Task was cancelled, cleanup
                if task_name in self.active_tasks:
                    del self.active_tasks[task_name]
                raise
            finally:
                # Always cleanup
                if task_name in self.active_tasks:
                    del self.active_tasks[task_name]

        except Exception as e:
            self.logger.error(f"Failed to create/execute task {task_name}: {e}")
            task_result.error = e
            task_result.state = TaskState.FAILED
            task_result.end_time = time.time()

        return task_result

    async def cancel_task(self, task_name: str, graceful_timeout: float = 5.0) -> bool:
        """Cancel a running task with graceful shutdown support.

        Args:
            task_name: Name of task to cancel
            graceful_timeout: Time to wait for graceful cancellation

        Returns:
            bool: True if cancelled successfully
        """
        if task_name not in self.active_tasks:
            self.logger.warning(f"Task not found for cancellation: {task_name}")
            return False

        task = self.active_tasks[task_name]
        task.cancel()

        try:
            # Wait for graceful cancellation
            await asyncio.wait_for(task, timeout=graceful_timeout)
            self.logger.info(f"Task cancelled gracefully: {task_name}")
            return True
        except asyncio.TimeoutError:
            # Force cancellation if graceful timeout exceeded
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            self.logger.warning(f"Task forcefully cancelled after timeout: {task_name}")
            return True
        except asyncio.CancelledError:
            self.logger.info(f"Task was already cancelled: {task_name}")
            return True
        except Exception as e:
            self.logger.error(f"Error during task cancellation: {task_name} - {e}")
            return False
        finally:
            if task_name in self.active_tasks:
                del self.active_tasks[task_name]

    async def cancel_all_tasks(self, graceful_timeout: float = 5.0) -> int:
        """Cancel all running tasks.

        Args:
            graceful_timeout: Time to wait for graceful cancellation per task

        Returns:
            int: Number of tasks cancelled
        """
        if not self.active_tasks:
            self.logger.info("No active tasks to cancel")
            return 0

        cancelled_count = 0
        tasks_to_cancel = list(self.active_tasks.keys())

        for task_name in tasks_to_cancel:
            if await self.cancel_task(task_name, graceful_timeout):
                cancelled_count += 1

        self.logger.info(f"Cancelled {cancelled_count}/{len(tasks_to_cancel)} tasks")
        return cancelled_count

    async def gather_with_timeout(self,
                                 coros: List[Coroutine[Any, Any, Any]],
                                 timeout: Optional[float] = None,
                                 return_exceptions: bool = True) -> List[Any]:
        """Enhanced gather with timeout and proper cancellation.

        Args:
            coros: List of coroutines to execute
            timeout: Overall timeout for all coroutines
            return_exceptions: Whether to return exceptions or raise them

        Returns:
            List of results or exceptions
        """
        if not coros:
            return []

        try:
            if timeout:
                return await asyncio.wait_for(
                    asyncio.gather(*coros, return_exceptions=return_exceptions),
                    timeout=timeout
                )
            else:
                return await asyncio.gather(*coros, return_exceptions=return_exceptions)
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout in gather_with_timeout after {timeout}s")
            # Cancel all running tasks
            for task in asyncio.all_tasks():
                if task != asyncio.current_task() and not task.done():
                    task.cancel()
            raise

    def get_task_status(self, task_name: str) -> Optional[TaskResult]:
        """Get status of a specific task.

        Args:
            task_name: Name of the task

        Returns:
            TaskResult if found, None otherwise
        """
        return self.task_results.get(task_name)

    def get_all_task_status(self) -> Dict[str, TaskResult]:
        """Get status of all tasks (active and completed).

        Returns:
            Dictionary mapping task names to their results
        """
        return self.task_results.copy()

    async def shutdown(self, graceful_timeout: float = 10.0):
        """Gracefully shutdown the async manager.

        Args:
            graceful_timeout: Time to wait for graceful shutdown
        """
        self.logger.info("Starting graceful shutdown of EnhancedAsyncManager")

        # Cancel all active tasks
        await self.cancel_all_tasks(graceful_timeout)

        # Wait for cleanup tasks to complete
        if self._cleanup_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._cleanup_tasks, return_exceptions=True),
                    timeout=graceful_timeout
                )
            except asyncio.TimeoutError:
                self.logger.warning("Timeout waiting for cleanup tasks")

        self._shutdown_event.set()
        self.logger.info("EnhancedAsyncManager shutdown complete")

# Global enhanced async manager instance
enhanced_async_manager = EnhancedAsyncManager()

# =============================================================================
# ASYNC UTILITY FUNCTIONS
# =============================================================================

async def safe_await(coro: Awaitable[T],
                    timeout: Optional[float] = None,
                    default: T = None) -> T:
    """Safely await a coroutine with timeout and default value.

    Args:
        coro: Coroutine to await
        timeout: Optional timeout in seconds
        default: Default value if timeout or error occurs

    Returns:
        Result of coroutine or default value
    """
    try:
        if timeout:
            return await asyncio.wait_for(coro, timeout=timeout)
        return await coro
    except asyncio.TimeoutError:
        logger = system_logger.getChild("safe_await")
        logger.warning(f"Timeout awaiting coroutine after {timeout}s")
        return default
    except asyncio.CancelledError:
        logger = system_logger.getChild("safe_await")
        logger.info("Coroutine was cancelled")
        return default
    except Exception as e:
        logger = system_logger.getChild("safe_await")
        logger.error(f"Error awaiting coroutine: {e}")
        return default

async def run_with_timeout(coro: Coroutine[Any, Any, T],
                          timeout: float,
                          default: T = None) -> T:
    """Run a coroutine with timeout and return default on timeout.

    Args:
        coro: Coroutine to run
        timeout: Timeout in seconds
        default: Default value if timeout occurs

    Returns:
        Result of coroutine or default value
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger = system_logger.getChild("run_with_timeout")
        logger.warning(f"Timeout after {timeout}s")
        return default
    except Exception as e:
        logger = system_logger.getChild("run_with_timeout")
        logger.error(f"Error in coroutine: {e}")
        return default

async def retry_async(func: Callable[..., Awaitable[T]],
                     max_retries: int = 3,
                     delay: float = 1.0,
                     backoff: float = 2.0,
                     exceptions: tuple = (Exception,)) -> T:
    """Retry an async function with exponential backoff.

    Args:
        func: Async function to retry
        max_retries: Maximum number of retries
        delay: Initial delay between retries
        backoff: Backoff multiplier
        exceptions: Tuple of exceptions to catch and retry

    Returns:
        Result of the function

    Raises:
        Last exception if all retries fail
    """
    logger = system_logger.getChild("retry_async")
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries:
                wait_time = delay * (backoff ** attempt)
                logger.warning(f"Retry {attempt + 1}/{max_retries + 1} after {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"All {max_retries + 1} retries failed")

    raise last_exception
