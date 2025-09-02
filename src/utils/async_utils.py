# src/utils/async_utils.py

from collections.abc import Coroutine
from typing import Any, Dict, Optional, Union, List
from src.utils.logger import system_logger
import aiofiles
import asyncio
import contextlib
import json
import os

from src.utils.error_handler import (
    handle_errors,
    handle_file_operations,
    handle_specific_errors,
    error,
    failed,
    invalid,
    missing,
    warning,
)


class AsyncFileManager:
    """
    Enhanced async file manager with comprehensive error handling and type safety.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize AsyncFileManager.
        
        Args:
            config: Configuration dictionary
        """
        self.config: Dict[str, Any] = config or {}
        self.logger = system_logger.getChild("AsyncFileManager")
        
        # File management
        self.file_cache: Dict[str, Any] = {}
        self.max_cache_size: int = 100
        self.cache_enabled: bool = True
        
        # Configuration
        self.file_config: Dict[str, Any] = self.config.get("async_file_manager", {})
        self.max_cache_size = int(self.file_config.get("max_cache_size", 100))
        self.cache_enabled = bool(self.file_config.get("cache_enabled", True))
        self.default_encoding: str = str(self.file_config.get("default_encoding", "utf-8"))

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
        """Initialize the async file manager.
        
        Returns:
            True if initialization successful, False otherwise
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

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="file configuration loading",
    )
    async def _load_file_configuration(self) -> None:
        """Load and validate file configuration."""
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

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """Validate configuration parameters.
        
        Returns:
            True if configuration is valid, False otherwise
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

    @handle_file_operations(
        default_return=None,
        context="file reading",
    )
    async def read_file(self, file_path: str, encoding: Optional[str] = None) -> Optional[str]:
        """Read file content asynchronously.
        
        Args:
            file_path: Path to the file to read
            encoding: File encoding (uses default if not specified)
            
        Returns:
            File content as string or None if failed
        """
        # Check cache first
        if self.cache_enabled and file_path in self.file_cache:
            self.logger.info(f"Reading {file_path} from cache")
            return str(self.file_cache[file_path])
        
        # Read file
        chosen_encoding = encoding or self.default_encoding
        async with aiofiles.open(file_path, mode="r", encoding=chosen_encoding) as f:
            content = await f.read()
        
        # Cache the content
        if self.cache_enabled:
            self._add_to_cache(file_path, content)
        
        self.logger.info(f"Read file: {file_path}")
        return content

    @handle_file_operations(
        default_return=False,
        context="file writing",
    )
    async def write_file(self, file_path: str, content: str, encoding: Optional[str] = None) -> bool:
        """Write content to file asynchronously.
        
        Args:
            file_path: Path to the file to write
            content: Content to write
            encoding: File encoding (uses default if not specified)
            
        Returns:
            True if successful, False otherwise
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        
        # Write file
        chosen_encoding = encoding or self.default_encoding
        async with aiofiles.open(file_path, "w", encoding=chosen_encoding) as f:
            await f.write(content)
        
        # Update cache
        if self.cache_enabled:
            self._add_to_cache(file_path, content)
        
        self.logger.info(f"Wrote file: {file_path}")
        return True

    @handle_file_operations(
        default_return=None,
        context="JSON file reading",
    )
    async def read_json(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Read JSON file content asynchronously.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Parsed JSON data or None if failed
        """
        content = await self.read_file(file_path)
        if content is None:
            return None
        
        data: Dict[str, Any] = json.loads(content)
        self.logger.info(f"Read JSON file: {file_path}")
        return data

    @handle_file_operations(
        default_return=False,
        context="JSON file writing",
    )
    async def write_json(self, file_path: str, data: Dict[str, Any], indent: int = 2) -> bool:
        """Write data to JSON file asynchronously.
        
        Args:
            file_path: Path to the JSON file
            data: Data to write
            indent: JSON indentation
            
        Returns:
            True if successful, False otherwise
        """
        content = json.dumps(data, indent=indent, default=str)
        success = await self.write_file(file_path, content)
        if success:
            self.logger.info(f"Wrote JSON file: {file_path}")
        return success

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="cache management",
    )
    def _add_to_cache(self, file_path: str, content: str) -> None:
        """Add file content to cache.
        
        Args:
            file_path: File path as cache key
            content: File content to cache
        """
        # Remove oldest entry if cache is full
        if len(self.file_cache) >= self.max_cache_size:
            oldest_key = next(iter(self.file_cache))
            del self.file_cache[oldest_key]
            self.logger.debug(f"Removed {oldest_key} from cache")
        
        # Add to cache
        self.file_cache[file_path] = content
        self.logger.debug(f"Added {file_path} to cache")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="cache clearing",
    )
    def clear_cache(self) -> None:
        """Clear the file cache."""
        cache_size = len(self.file_cache)
        self.file_cache.clear()
        self.logger.info(f"Cleared cache ({cache_size} entries)")

    def get_cache_status(self) -> Dict[str, Any]:
        """Get cache status information.
        
        Returns:
            Dictionary with cache status
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
        self.clear_cache()
        self.logger.info("✅ Async File Manager stopped successfully")


class AsyncTaskManager:
    """
    Enhanced async task manager with comprehensive error handling and type safety.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize AsyncTaskManager.
        
        Args:
            config: Configuration dictionary
        """
        self.config: Dict[str, Any] = config or {}
        self.logger = system_logger.getChild("AsyncTaskManager")
        
        # Task management
        self.active_tasks: Dict[str, asyncio.Task[Any]] = {}
        self.task_results: Dict[str, Any] = {}
        self.max_concurrent_tasks: int = 10
        
        # Configuration
        self.task_config: Dict[str, Any] = self.config.get("async_task_manager", {})
        self.max_concurrent_tasks = int(self.task_config.get("max_concurrent_tasks", 10))
        self.task_timeout: int = int(self.task_config.get("task_timeout", 300))

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
        """Initialize the async task manager.
        
        Returns:
            True if initialization successful, False otherwise
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

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="task configuration loading",
    )
    async def _load_task_configuration(self) -> None:
        """Load and validate task configuration."""
        # Set default task parameters
        self.task_config.setdefault("max_concurrent_tasks", 10)
        self.task_config.setdefault("task_timeout", 300)
        self.task_config.setdefault("enable_task_monitoring", True)
        self.task_config.setdefault("auto_cleanup_failed_tasks", True)
        
        # Update configuration
        self.max_concurrent_tasks = int(self.task_config["max_concurrent_tasks"])
        self.task_timeout = int(self.task_config["task_timeout"])
        
        self.logger.info("Task configuration loaded successfully")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """Validate configuration parameters.
        
        Returns:
            True if configuration is valid, False otherwise
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

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="task execution",
    )
    async def execute_task(self, coro: Coroutine, task_name: str, timeout: Optional[int] = None) -> Optional[Any]:
        """Execute an async coroutine as a managed task.
        
        Args:
            coro: Coroutine to execute
            task_name: Name for the task
            timeout: Task timeout in seconds
            
        Returns:
            Task result or None if failed
        """
        # Check if we can run more tasks
        if len(self.active_tasks) >= self.max_concurrent_tasks:
            self.logger.warning(
                f"Maximum concurrent tasks reached ({self.max_concurrent_tasks})",
            )
            return None
        
        # Create task
        chosen_timeout = timeout or self.task_timeout
        task = asyncio.create_task(coro, name=task_name)
        self.active_tasks[task_name] = task
        
        self.logger.info(f"Started task: {task_name}")
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(task, timeout=chosen_timeout)
            self.task_results[task_name] = result
            self.logger.info(f"Task completed: {task_name}")
            return result
        except asyncio.TimeoutError:
            self.logger.error(failed(f"Task timed out: {task_name}"))
            task.cancel()
            return None
        except Exception as e:  # noqa: BLE001
            self.logger.error(failed(f"Task failed: {task_name} - {e}"))
            return None
        finally:
            # Remove from active tasks
            if task_name in self.active_tasks:
                del self.active_tasks[task_name]

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="task cancellation",
    )
    async def cancel_task(self, task_name: str) -> bool:
        """Cancel a running task.
        
        Args:
            task_name: Name of the task to cancel
            
        Returns:
            True if task was cancelled, False otherwise
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

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="all tasks cancellation",
    )
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

    def get_task_status(self) -> Dict[str, Any]:
        """Get task manager status information.
        
        Returns:
            Dictionary with task status
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
        await self.cancel_all_tasks()
        self.task_results.clear()
        self.logger.info("✅ Async Task Manager stopped successfully")


# Global instances
async_file_manager: Optional[AsyncFileManager] = None
async_task_manager: Optional[AsyncTaskManager] = None


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="async utils setup",
)
async def setup_async_utils(config: Optional[Dict[str, Any]] = None) -> tuple[Optional[AsyncFileManager], Optional[AsyncTaskManager]]:
    """Setup global async utility instances.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (AsyncFileManager, AsyncTaskManager) instances
    """
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


class AsyncProcessesManager:
    """
    Manager for async processes with comprehensive error handling.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize AsyncProcessesManager.
        
        Args:
            config: Configuration dictionary
        """
        self.config: Dict[str, Any] = config or {}
        self.logger = system_logger.getChild("AsyncProcessesManager")
        self.processes: Dict[str, asyncio.subprocess.Process] = {}
        self.max_processes: int = int(self.config.get("max_processes", 10))

    async def start_process(self, name: str, command: List[str], cwd: Optional[str] = None) -> Optional[asyncio.subprocess.Process]:
        """Start an async subprocess.
        
        Args:
            name: Name identifier for the process
            command: Command and arguments to execute
            cwd: Working directory for the process
            
        Returns:
            Process object or None if failed
        """
        if len(self.processes) >= self.max_processes:
            self.logger.warning(warning(f"Maximum processes ({self.max_processes}) reached"))
            return None
        
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as e:  # noqa: BLE001
            self.logger.error(failed(f"Failed to start process '{name}': {e}"))
            return None
        
        self.processes[name] = process
        self.logger.info(f"Started process '{name}' with PID {process.pid}")
        return process

    async def stop_process(self, name: str) -> bool:
        """Stop a running process.
        
        Args:
            name: Name of the process to stop
            
        Returns:
            True if process was stopped, False otherwise
        """
        if name not in self.processes:
            self.logger.warning(missing(f"Process '{name}' not found"))
            return False
        
        process = self.processes[name]
        try:
            process.terminate()
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
        except Exception as e:  # noqa: BLE001
            self.logger.error(failed(f"Failed to stop process '{name}': {e}"))
            return False
        
        del self.processes[name]
        self.logger.info(f"Stopped process '{name}'")
        return True

    async def stop_all_processes(self) -> None:
        """Stop all running processes."""
        for name in list(self.processes.keys()):
            await self.stop_process(name)

    def get_process_status(self) -> Dict[str, Any]:
        """Get process manager status information.
        
        Returns:
            Dictionary with process status
        """
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
