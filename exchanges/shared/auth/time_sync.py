"""
Time Synchronization Utilities

Handles clock skew detection and correction for exchange API calls.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from src.utils.logger import system_logger


@dataclass
class TimeSyncInfo:
    """Time synchronization information"""
    exchange_name: str
    server_time: int
    local_time: int
    clock_skew: int
    last_sync: datetime
    sync_accuracy: float  # milliseconds
    is_synced: bool = False


class TimeSyncManager:
    """
    Manages time synchronization with exchange servers to handle clock skew.
    """
    
    def __init__(self, exchange_name: str, max_skew_ms: int = 5000):
        self.exchange_name = exchange_name
        self.max_skew_ms = max_skew_ms
        self.logger = system_logger.getChild(f"TimeSyncManager.{exchange_name}")
        self.sync_info: Optional[TimeSyncInfo] = None
        self.sync_history: List[TimeSyncInfo] = []
        self.auto_sync_interval = 300  # 5 minutes
        self.sync_task: Optional[asyncio.Task] = None
        
    async def start_auto_sync(self, sync_function) -> None:
        """Start automatic time synchronization."""
        if self.sync_task and not self.sync_task.done():
            self.logger.warning("Auto sync already running")
            return
            
        self.sync_task = asyncio.create_task(self._auto_sync_loop(sync_function))
        self.logger.info("Started automatic time synchronization")
    
    async def stop_auto_sync(self) -> None:
        """Stop automatic time synchronization."""
        if self.sync_task:
            self.sync_task.cancel()
            try:
                await self.sync_task
            except asyncio.CancelledError:
                pass
            self.sync_task = None
            self.logger.info("Stopped automatic time synchronization")
    
    async def _auto_sync_loop(self, sync_function) -> None:
        """Automatic synchronization loop."""
        while True:
            try:
                await self.sync_time(sync_function)
                await asyncio.sleep(self.auto_sync_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in auto sync loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def sync_time(self, sync_function) -> bool:
        """
        Synchronize time with exchange server.
        
        Args:
            sync_function: Async function that returns server time in milliseconds
            
        Returns:
            True if sync was successful
        """
        try:
            # Get local time before API call
            local_time_before = int(time.time() * 1000)
            
            # Get server time
            server_time = await sync_function()
            if not server_time:
                self.logger.warning("Failed to get server time")
                return False
                
            # Get local time after API call
            local_time_after = int(time.time() * 1000)
            
            # Calculate average local time (accounting for network latency)
            local_time_avg = (local_time_before + local_time_after) // 2
            
            # Calculate clock skew
            clock_skew = server_time - local_time_avg
            
            # Calculate sync accuracy (network latency estimate)
            sync_accuracy = local_time_after - local_time_before
            
            # Create sync info
            sync_info = TimeSyncInfo(
                exchange_name=self.exchange_name,
                server_time=server_time,
                local_time=local_time_avg,
                clock_skew=clock_skew,
                last_sync=datetime.now(),
                sync_accuracy=sync_accuracy,
                is_synced=abs(clock_skew) <= self.max_skew_ms
            )
            
            # Update sync info
            self.sync_info = sync_info
            self.sync_history.append(sync_info)
            
            # Keep only last 100 sync records
            if len(self.sync_history) > 100:
                self.sync_history = self.sync_history[-100:]
            
            if sync_info.is_synced:
                self.logger.info(f"Time synced successfully. Clock skew: {clock_skew}ms")
            else:
                self.logger.warning(f"Clock skew too large: {clock_skew}ms (max: {self.max_skew_ms}ms)")
            
            return sync_info.is_synced
            
        except Exception as e:
            self.logger.error(f"Error syncing time: {e}")
            return False
    
    def get_adjusted_timestamp(self, local_timestamp: Optional[int] = None) -> int:
        """
        Get timestamp adjusted for clock skew.
        
        Args:
            local_timestamp: Local timestamp in milliseconds. If None, uses current time.
            
        Returns:
            Adjusted timestamp for API calls
        """
        if local_timestamp is None:
            local_timestamp = int(time.time() * 1000)
            
        if not self.sync_info or not self.sync_info.is_synced:
            self.logger.warning("Time not synced, using local timestamp")
            return local_timestamp
            
        # Adjust for clock skew
        adjusted_timestamp = local_timestamp + self.sync_info.clock_skew
        return adjusted_timestamp
    
    def get_server_time_estimate(self) -> int:
        """
        Get estimated server time based on last sync.
        
        Returns:
            Estimated server time in milliseconds
        """
        if not self.sync_info:
            return int(time.time() * 1000)
            
        # Calculate time elapsed since last sync
        time_elapsed = int((datetime.now() - self.sync_info.last_sync).total_seconds() * 1000)
        
        # Estimate current server time
        estimated_server_time = self.sync_info.server_time + time_elapsed
        return estimated_server_time
    
    def is_time_synced(self) -> bool:
        """Check if time is currently synced."""
        if not self.sync_info:
            return False
            
        # Check if sync is recent (within 10 minutes)
        time_since_sync = datetime.now() - self.sync_info.last_sync
        is_recent = time_since_sync < timedelta(minutes=10)
        
        return self.sync_info.is_synced and is_recent
    
    def get_clock_skew(self) -> Optional[int]:
        """Get current clock skew in milliseconds."""
        if not self.sync_info:
            return None
        return self.sync_info.clock_skew
    
    def get_sync_accuracy(self) -> Optional[float]:
        """Get last sync accuracy in milliseconds."""
        if not self.sync_info:
            return None
        return self.sync_info.sync_accuracy
    
    def get_sync_statistics(self) -> Dict[str, Any]:
        """Get time synchronization statistics."""
        if not self.sync_history:
            return {
                "total_syncs": 0,
                "successful_syncs": 0,
                "average_skew": 0,
                "max_skew": 0,
                "min_skew": 0,
                "average_accuracy": 0,
                "last_sync": None,
                "is_currently_synced": False
            }
        
        successful_syncs = [s for s in self.sync_history if s.is_synced]
        skews = [s.clock_skew for s in self.sync_history]
        accuracies = [s.sync_accuracy for s in self.sync_history]
        
        return {
            "total_syncs": len(self.sync_history),
            "successful_syncs": len(successful_syncs),
            "average_skew": sum(skews) / len(skews) if skews else 0,
            "max_skew": max(skews) if skews else 0,
            "min_skew": min(skews) if skews else 0,
            "average_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0,
            "last_sync": self.sync_info.last_sync.isoformat() if self.sync_info else None,
            "is_currently_synced": self.is_time_synced()
        }
    
    def should_sync_now(self) -> bool:
        """Check if time synchronization is needed."""
        if not self.sync_info:
            return True
            
        # Sync if last sync was more than 10 minutes ago
        time_since_sync = datetime.now() - self.sync_info.last_sync
        return time_since_sync > timedelta(minutes=10)
    
    def get_recommended_timestamp_for_request(self) -> int:
        """
        Get recommended timestamp for API request.
        
        Returns:
            Timestamp adjusted for clock skew and network latency
        """
        if not self.is_time_synced():
            self.logger.warning("Time not synced, using local timestamp")
            return int(time.time() * 1000)
        
        # Use server time estimate with small buffer for network latency
        server_time = self.get_server_time_estimate()
        buffer_ms = 1000  # 1 second buffer
        return server_time + buffer_ms
    
    def validate_timestamp(self, timestamp: int, tolerance_ms: int = 5000) -> bool:
        """
        Validate if a timestamp is within acceptable range.
        
        Args:
            timestamp: Timestamp to validate
            tolerance_ms: Tolerance in milliseconds
            
        Returns:
            True if timestamp is valid
        """
        if not self.sync_info:
            return True  # Can't validate without sync info
            
        server_time = self.get_server_time_estimate()
        time_diff = abs(timestamp - server_time)
        
        return time_diff <= tolerance_ms
    
    async def force_sync(self, sync_function) -> bool:
        """Force immediate time synchronization."""
        self.logger.info("Forcing time synchronization")
        return await self.sync_time(sync_function)
    
    def cleanup_old_sync_history(self, keep_last: int = 50) -> int:
        """Clean up old sync history records."""
        if len(self.sync_history) <= keep_last:
            return 0
            
        removed_count = len(self.sync_history) - keep_last
        self.sync_history = self.sync_history[-keep_last:]
        
        self.logger.info(f"Cleaned up {removed_count} old sync records")
        return removed_count