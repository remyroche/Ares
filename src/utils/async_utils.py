# src/utils/async_utils.py

import asyncio
import json
import os
from collections.abc import Coroutine
from typing import Any

import aiofiles

from src.utils.error_handler import (
    handle_errors,
    handle_file_operations,
    handle_specific_errors,
)
from src.utils.logger import system_logger


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
        self.max_cache_size = self.file_config.get("max_cache_size", 100)
        self.cache_enabled = self.file_config.get("cache_enabled", True)
        self.default_encoding: str = self.file_config.get("default_encoding", "utf-8")

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid async file manager configuration"),
            AttributeError: (False, "Missing required file parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="async file manager initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize async file manager with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Async File Manager...")

            # Load file configuration
            await self._load_file_configuration()

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for async file manager")
                return False

            self.logger.info(
                "✅ Async File Manager initialization completed successfully",
            )
            return True

        except Exception as e:
            self.logger.error(f"❌ Async File Manager initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="file configuration loading",
    )
    async def _load_file_configuration(self) -> None:
        """Load file configuration."""
        try:
            # Set default file parameters
            self.file_config.setdefault("max_cache_size", 100)
            self.file_config.setdefault("cache_enabled", True)
            self.file_config.setdefault("default_encoding", "utf-8")
            self.file_config.setdefault("chunk_size", 8192)
            self.file_config.setdefault("timeout", 30)

            # Update configuration
            self.max_cache_size = self.file_config["max_cache_size"]
            self.cache_enabled = self.file_config["cache_enabled"]
            self.default_encoding = self.file_config["default_encoding"]

            self.logger.info("File configuration loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading file configuration: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate file configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate cache size
            if self.max_cache_size <= 0:
                self.logger.error("Invalid max cache size")
                return False

            # Validate encoding
            if not self.default_encoding:
                self.logger.error("Invalid default encoding")
                return False

            self.logger.info("Configuration validation successful")
            return True

        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    @handle_file_operations(
        default_return=None,
        context="file reading",
    )
    async def read_file(
        self,
        file_path: str,
        encoding: str | None = None,
    ) -> str | None:
        """
        Read file asynchronously.

        Args:
            file_path: Path to the file
            encoding: File encoding (defaults to configured encoding)

        Returns:
            Optional[str]: File content or None if failed
        """
        try:
            # Check cache first
            if self.cache_enabled and file_path in self.file_cache:
                self.logger.info(f"Reading {file_path} from cache")
                return self.file_cache[file_path]

            # Read file
            encoding = encoding or self.default_encoding
            async with aiofiles.open(file_path, encoding=encoding) as f:
                content = await f.read()

            # Cache the content
            if self.cache_enabled:
                self._add_to_cache(file_path, content)

            self.logger.info(f"Read file: {file_path}")
            return content

        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            return None

    @handle_file_operations(
        default_return=None,
        context="file writing",
    )
    async def write_file(
        self,
        file_path: str,
        content: str,
        encoding: str | None = None,
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
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Write file
            encoding = encoding or self.default_encoding
            async with aiofiles.open(file_path, "w", encoding=encoding) as f:
                await f.write(content)

            # Update cache
            if self.cache_enabled:
                self._add_to_cache(file_path, content)

            self.logger.info(f"Wrote file: {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error writing file {file_path}: {e}")
            return False

    @handle_file_operations(
        default_return=None,
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
        try:
            content = await self.read_file(file_path)
            if content is None:
                return None

            data = json.loads(content)
            self.logger.info(f"Read JSON file: {file_path}")
            return data

        except Exception as e:
            self.logger.error(f"Error reading JSON file {file_path}: {e}")
            return None

    @handle_file_operations(
        default_return=None,
        context="JSON file writing",
    )
    async def write_json(
        self,
        file_path: str,
        data: dict[str, Any],
        indent: int = 2,
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
        try:
            content = json.dumps(data, indent=indent, default=str)
            success = await self.write_file(file_path, content)

            if success:
                self.logger.info(f"Wrote JSON file: {file_path}")

            return success

        except Exception as e:
            self.logger.error(f"Error writing JSON file {file_path}: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="cache management",
    )
    def _add_to_cache(self, file_path: str, content: str) -> None:
        """
        Add file content to cache.

        Args:
            file_path: File path
            content: File content
        """
        try:
            # Remove oldest entry if cache is full
            if len(self.file_cache) >= self.max_cache_size:
                oldest_key = next(iter(self.file_cache))
                del self.file_cache[oldest_key]
                self.logger.debug(f"Removed {oldest_key} from cache")

            # Add to cache
            self.file_cache[file_path] = content
            self.logger.debug(f"Added {file_path} to cache")

        except Exception as e:
            self.logger.error(f"Error adding to cache: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="cache clearing",
    )
    def clear_cache(self) -> None:
        """Clear the file cache."""
        try:
            cache_size = len(self.file_cache)
            self.file_cache.clear()
            self.logger.info(f"Cleared cache ({cache_size} entries)")

        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")

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

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="async file manager cleanup",
    )
    async def stop(self) -> None:
        """Stop the async file manager."""
        self.logger.info("🛑 Stopping Async File Manager...")

        try:
            # Clear cache
            self.clear_cache()

            self.logger.info("✅ Async File Manager stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping async file manager: {e}")


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
        self.active_tasks: dict[str, asyncio.Task] = {}
        self.task_results: dict[str, Any] = {}
        self.max_concurrent_tasks: int = 10

        # Configuration
        self.task_config: dict[str, Any] = self.config.get("async_task_manager", {})
        self.max_concurrent_tasks = self.task_config.get("max_concurrent_tasks", 10)
        self.task_timeout: int = self.task_config.get("task_timeout", 300)

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid async task manager configuration"),
            AttributeError: (False, "Missing required task parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="async task manager initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize async task manager with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Async Task Manager...")

            # Load task configuration
            await self._load_task_configuration()

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for async task manager")
                return False

            self.logger.info(
                "✅ Async Task Manager initialization completed successfully",
            )
            return True

        except Exception as e:
            self.logger.error(f"❌ Async Task Manager initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="task configuration loading",
    )
    async def _load_task_configuration(self) -> None:
        """Load task configuration."""
        try:
            # Set default task parameters
            self.task_config.setdefault("max_concurrent_tasks", 10)
            self.task_config.setdefault("task_timeout", 300)
            self.task_config.setdefault("enable_task_monitoring", True)
            self.task_config.setdefault("auto_cleanup_failed_tasks", True)

            # Update configuration
            self.max_concurrent_tasks = self.task_config["max_concurrent_tasks"]
            self.task_timeout = self.task_config["task_timeout"]

            self.logger.info("Task configuration loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading task configuration: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate task configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate max concurrent tasks
            if self.max_concurrent_tasks <= 0:
                self.logger.error("Invalid max concurrent tasks")
                return False

            # Validate task timeout
            if self.task_timeout <= 0:
                self.logger.error("Invalid task timeout")
                return False

            self.logger.info("Configuration validation successful")
            return True

        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="task execution",
    )
    async def execute_task(
        self,
        task_name: str,
        coro: Coroutine,
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
        try:
            # Check if we can run more tasks
            if len(self.active_tasks) >= self.max_concurrent_tasks:
                self.logger.warning(
                    f"Maximum concurrent tasks reached ({self.max_concurrent_tasks})",
                )
                return None

            # Create task
            timeout = timeout or self.task_timeout
            task = asyncio.create_task(coro, name=task_name)
            self.active_tasks[task_name] = task

            self.logger.info(f"Started task: {task_name}")

            # Execute with timeout
            try:
                result = await asyncio.wait_for(task, timeout=timeout)
                self.task_results[task_name] = result
                self.logger.info(f"Task completed: {task_name}")
                return result
            except TimeoutError:
                self.logger.error(f"Task timed out: {task_name}")
                task.cancel()
                return None
            except Exception as e:
                self.logger.error(f"Task failed: {task_name} - {e}")
                return None
            finally:
                # Remove from active tasks
                if task_name in self.active_tasks:
                    del self.active_tasks[task_name]

        except Exception as e:
            self.logger.error(f"Error executing task {task_name}: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="task cancellation",
    )
    async def cancel_task(self, task_name: str) -> bool:
        """
        Cancel a running task.

        Args:
            task_name: Name of the task to cancel

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if task_name not in self.active_tasks:
                self.logger.warning(f"Task not found: {task_name}")
                return False

            task = self.active_tasks[task_name]
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

            del self.active_tasks[task_name]
            self.logger.info(f"Cancelled task: {task_name}")
            return True

        except Exception as e:
            self.logger.error(f"Error cancelling task {task_name}: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="all tasks cancellation",
    )
    async def cancel_all_tasks(self) -> None:
        """Cancel all running tasks."""
        try:
            if not self.active_tasks:
                self.logger.info("No active tasks to cancel")
                return

            self.logger.info(f"Cancelling {len(self.active_tasks)} active tasks...")

            for task_name, task in self.active_tasks.items():
                try:
                    task.cancel()
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    self.logger.error(f"Error cancelling task {task_name}: {e}")

            self.active_tasks.clear()
            self.logger.info("All tasks cancelled")

        except Exception as e:
            self.logger.error(f"Error cancelling all tasks: {e}")

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

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="async task manager cleanup",
    )
    async def stop(self) -> None:
        """Stop the async task manager."""
        self.logger.info("🛑 Stopping Async Task Manager...")

        try:
            # Cancel all active tasks
            await self.cancel_all_tasks()

            # Clear results
            self.task_results.clear()

            self.logger.info("✅ Async Task Manager stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping async task manager: {e}")


# Global instances
async_file_manager: AsyncFileManager | None = None
async_task_manager: AsyncTaskManager | None = None


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="async utils setup",
)
async def setup_async_utils(
    config: dict[str, Any] | None = None,
) -> tuple[AsyncFileManager | None, AsyncTaskManager | None]:
    """
    Setup global async utilities.

    Args:
        config: Optional configuration dictionary

    Returns:
        Tuple[Optional[AsyncFileManager], Optional[AsyncTaskManager]]: Global instances
    """
    try:
        global async_file_manager, async_task_manager

        if config is None:
            config = {
                "async_file_manager": {
                    "max_cache_size": 100,
                    "cache_enabled": True,
                    "default_encoding": "utf-8",
                    "chunk_size": 8192,
                    "timeout": 30,
                },
                "async_task_manager": {
                    "max_concurrent_tasks": 10,
                    "task_timeout": 300,
                    "enable_task_monitoring": True,
                    "auto_cleanup_failed_tasks": True,
                },
            }

        # Create async file manager
        async_file_manager = AsyncFileManager(config)
        file_success = await async_file_manager.initialize()

        # Create async task manager
        async_task_manager = AsyncTaskManager(config)
        task_success = await async_task_manager.initialize()

        if file_success and task_success:
            return async_file_manager, async_task_manager
        return None, None

    except Exception as e:
        print(f"Error setting up async utils: {e}")
        return None, None


class AsyncProcessesManager:
    """
    Manager for async processes with comprehensive error handling.
    """

    def __init__(self, config: dict[str, Any] = None):
        self.config = config or {}
        self.logger = system_logger.getChild("AsyncProcessesManager")
        self.processes: dict[str, asyncio.subprocess.Process] = {}
        self.max_processes: int = self.config.get("max_processes", 10)

    async def start_process(
        self,
        name: str,
        command: list[str],
        cwd: str | None = None,
    ) -> asyncio.subprocess.Process | None:
        """Start an async process."""
        try:
            if len(self.processes) >= self.max_processes:
                self.logger.warning(f"Maximum processes ({self.max_processes}) reached")
                return None

            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            self.processes[name] = process
            self.logger.info(f"Started process '{name}' with PID {process.pid}")
            return process

        except Exception as e:
            self.logger.error(f"Failed to start process '{name}': {e}")
            return None

    async def stop_process(self, name: str) -> bool:
        """Stop a specific process."""
        try:
            if name not in self.processes:
                self.logger.warning(f"Process '{name}' not found")
                return False

            process = self.processes[name]
            process.terminate()

            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except TimeoutError:
                process.kill()
                await process.wait()

            del self.processes[name]
            self.logger.info(f"Stopped process '{name}'")
            return True

        except Exception as e:
            self.logger.error(f"Failed to stop process '{name}': {e}")
            return False

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
                name: {"pid": process.pid, "returncode": process.returncode}
                for name, process in self.processes.items()
            },
        }


# Create a global instance for backward compatibility
async_processes_manager = AsyncProcessesManager()
