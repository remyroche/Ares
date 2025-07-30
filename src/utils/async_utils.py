# src/utils/async_utils.py

import asyncio
import aiofiles
import aiohttp
import json
import os
import hashlib
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from src.utils.logger import system_logger
from src.utils.error_handler import (
    handle_errors,
    handle_file_operations,
    handle_network_operations,
    error_context,
    ErrorRecoveryStrategies
)

class AsyncFileManager:
    """Async file operations manager"""
    
    @staticmethod
    @handle_file_operations(
        default_return="",
        context="async_read_file"
    )
    async def read_file(file_path: str, encoding: str = 'utf-8') -> str:
        """Async file read operation"""
        try:
            async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
                return await f.read()
        except Exception as e:
            system_logger.error(f"Failed to read file {file_path}: {e}")
            return ""

    @staticmethod
    @handle_file_operations(
        default_return=False,
        context="async_write_file"
    )
    async def write_file(file_path: str, content: str, encoding: str = 'utf-8') -> bool:
        """Async file write operation"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            async with aiofiles.open(file_path, 'w', encoding=encoding) as f:
                await f.write(content)
            return True
        except Exception as e:
            system_logger.error(f"Failed to write file {file_path}: {e}")
            return False

    @staticmethod
    @handle_file_operations(
        default_return=False,
        context="async_append_file"
    )
    async def append_file(file_path: str, content: str, encoding: str = 'utf-8') -> bool:
        """Async file append operation"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            async with aiofiles.open(file_path, 'a', encoding=encoding) as f:
                await f.write(content)
            return True
        except Exception as e:
            system_logger.error(f"Failed to append to file {file_path}: {e}")
            return False

    @staticmethod
    @handle_file_operations(
        default_return=b"",
        context="async_read_binary"
    )
    async def read_binary(file_path: str) -> bytes:
        """Async binary file read operation"""
        try:
            async with aiofiles.open(file_path, 'rb') as f:
                return await f.read()
        except Exception as e:
            system_logger.error(f"Failed to read binary file {file_path}: {e}")
            return b""

    @staticmethod
    @handle_file_operations(
        default_return=False,
        context="async_write_binary"
    )
    async def write_binary(file_path: str, content: bytes) -> bool:
        """Async binary file write operation"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(content)
            return True
        except Exception as e:
            system_logger.error(f"Failed to write binary file {file_path}: {e}")
            return False

    @staticmethod
    @handle_file_operations(
        default_return="",
        context="async_read_json"
    )
    async def read_json(file_path: str) -> Dict[str, Any]:
        """Async JSON file read operation"""
        try:
            content = await AsyncFileManager.read_file(file_path)
            if content:
                return json.loads(content)
            return {}
        except json.JSONDecodeError as e:
            system_logger.error(f"Failed to parse JSON from {file_path}: {e}")
            return {}

    @staticmethod
    @handle_file_operations(
        default_return=False,
        context="async_write_json"
    )
    async def write_json(file_path: str, data: Dict[str, Any], indent: int = 2) -> bool:
        """Async JSON file write operation"""
        try:
            content = json.dumps(data, indent=indent)
            return await AsyncFileManager.write_file(file_path, content)
        except Exception as e:
            system_logger.error(f"Failed to write JSON to {file_path}: {e}")
            return False

    @staticmethod
    @handle_file_operations(
        default_return="",
        context="async_calculate_checksum"
    )
    async def calculate_checksum(file_path: str) -> str:
        """Async file checksum calculation"""
        try:
            content = await AsyncFileManager.read_binary(file_path)
            return hashlib.md5(content).hexdigest()
        except Exception as e:
            system_logger.error(f"Failed to calculate checksum for {file_path}: {e}")
            return ""

class AsyncProcessManager:
    """Async process management utilities"""
    
    @staticmethod
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="async_run_process"
    )
    async def run_process(command: List[str], timeout: int = 30) -> Optional[Dict[str, Any]]:
        """Run a subprocess asynchronously"""
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            
            return {
                "returncode": process.returncode,
                "stdout": stdout.decode('utf-8') if stdout else "",
                "stderr": stderr.decode('utf-8') if stderr else "",
                "success": process.returncode == 0
            }
        except asyncio.TimeoutError:
            system_logger.error(f"Process {command} timed out after {timeout} seconds")
            return None
        except Exception as e:
            system_logger.error(f"Failed to run process {command}: {e}")
            return None

    @staticmethod
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="async_run_python_script"
    )
    async def run_python_script(script_path: str, args: List[str] = None, timeout: int = 30) -> Optional[Dict[str, Any]]:
        """Run a Python script asynchronously"""
        command = [sys.executable, script_path]
        if args:
            command.extend(args)
        return await AsyncProcessManager.run_process(command, timeout)

class AsyncNetworkManager:
    """Async network utilities"""
    
    @staticmethod
    @handle_network_operations(
        max_retries=3,
        default_return=None,
        context="async_http_request"
    )
    async def http_request(
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Dict[str, Any]] = None,
        timeout: int = 30
    ) -> Optional[Dict[str, Any]]:
        """Make an async HTTP request"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=method.upper(),
                    url=url,
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    response.raise_for_status()
                    return {
                        "status": response.status,
                        "headers": dict(response.headers),
                        "data": await response.json() if response.content_type == 'application/json' else await response.text()
                    }
        except Exception as e:
            system_logger.error(f"HTTP request failed {method} {url}: {e}")
            return None

class AsyncDelay:
    """Async delay utilities to replace time.sleep"""
    
    @staticmethod
    async def sleep(seconds: float):
        """Async sleep replacement for time.sleep"""
        await asyncio.sleep(seconds)

    @staticmethod
    async def sleep_with_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0):
        """Async sleep with exponential backoff"""
        delay = min(base_delay * (2 ** attempt), max_delay)
        await asyncio.sleep(delay)

    @staticmethod
    async def sleep_until(timestamp: float):
        """Sleep until a specific timestamp"""
        now = asyncio.get_event_loop().time()
        if timestamp > now:
            await asyncio.sleep(timestamp - now)

class AsyncLockManager:
    """Async lock management for thread-safe operations"""
    
    def __init__(self):
        self._locks: Dict[str, asyncio.Lock] = {}
    
    async def acquire_lock(self, lock_name: str) -> asyncio.Lock:
        """Get or create a lock for a specific name"""
        if lock_name not in self._locks:
            self._locks[lock_name] = asyncio.Lock()
        return self._locks[lock_name]
    
    async def with_lock(self, lock_name: str, coro):
        """Execute a coroutine with a specific lock"""
        lock = await self.acquire_lock(lock_name)
        async with lock:
            return await coro

# Global instances
async_file_manager = AsyncFileManager()
async_process_manager = AsyncProcessManager()
async_network_manager = AsyncNetworkManager()
async_delay = AsyncDelay()
async_lock_manager = AsyncLockManager() 