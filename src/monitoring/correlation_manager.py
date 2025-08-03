#!/usr/bin/env python3
"""
Correlation Manager

This module provides centralized correlation ID management and request/response
correlation tracking for the Ares trading bot.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors, handle_specific_errors


class CorrelationStatus(Enum):
    """Correlation status enumeration."""
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class CorrelationRequest:
    """Correlation request tracking."""
    correlation_id: str
    request_timestamp: datetime
    status: CorrelationStatus
    component_path: List[str]
    request_data: Dict[str, Any]
    response_timestamp: Optional[datetime] = None
    response_data: Optional[Dict[str, Any]] = None
    error_info: Optional[Dict[str, Any]] = None
    performance_metrics: Dict[str, float] = None
    metadata: Dict[str, Any] = None


class CorrelationManager:
    """
    Centralized correlation ID management and request/response correlation tracking.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize correlation manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("CorrelationManager")
        
        # Correlation configuration
        self.correlation_config = config.get("correlation_manager", {})
        self.enable_correlation_tracking = self.correlation_config.get("enable_correlation_tracking", True)
        self.correlation_timeout = self.correlation_config.get("correlation_timeout", 300)  # 5 minutes
        self.max_correlation_history = self.correlation_config.get("max_correlation_history", 10000)
        
        # Correlation storage
        self.correlation_requests: Dict[str, CorrelationRequest] = {}
        self.is_tracking = False
        
        self.logger.info("🔗 Correlation Manager initialized")

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid correlation configuration"),
            AttributeError: (False, "Missing required correlation parameters"),
        },
        default_return=False,
        context="correlation manager initialization",
    )
    async def initialize(self) -> bool:
        """Initialize the correlation manager."""
        try:
            self.logger.info("Initializing Correlation Manager...")
            
            # Initialize correlation storage
            self.correlation_requests.clear()
            
            self.logger.info("✅ Correlation Manager initialization completed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Correlation Manager initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="correlation request tracking",
    )
    async def track_correlation_request(
        self,
        correlation_id: str,
        component_path: List[str],
        request_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Track a new correlation request.
        
        Args:
            correlation_id: Unique correlation ID
            component_path: List of components in the request path
            request_data: Request data
            metadata: Optional metadata
        """
        try:
            correlation_request = CorrelationRequest(
                correlation_id=correlation_id,
                request_timestamp=datetime.now(),
                status=CorrelationStatus.ACTIVE,
                component_path=component_path,
                request_data=request_data,
                metadata=metadata or {},
            )
            
            self.correlation_requests[correlation_id] = correlation_request
            
            self.logger.debug(f"Tracking correlation request: {correlation_id}")
            
        except Exception as e:
            self.logger.error(f"Error tracking correlation request: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="correlation response tracking",
    )
    async def track_correlation_response(
        self,
        correlation_id: str,
        response_data: Dict[str, Any],
        error_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Track a correlation response.
        
        Args:
            correlation_id: Correlation ID
            response_data: Response data
            error_info: Optional error information
        """
        try:
            if correlation_id not in self.correlation_requests:
                self.logger.warning(f"Correlation ID not found: {correlation_id}")
                return
            
            correlation_request = self.correlation_requests[correlation_id]
            correlation_request.response_timestamp = datetime.now()
            correlation_request.response_data = response_data
            correlation_request.error_info = error_info
            
            if error_info:
                correlation_request.status = CorrelationStatus.FAILED
            else:
                correlation_request.status = CorrelationStatus.COMPLETED
            
            # Calculate performance metrics
            if correlation_request.request_timestamp and correlation_request.response_timestamp:
                duration_ms = (
                    correlation_request.response_timestamp - correlation_request.request_timestamp
                ).total_seconds() * 1000
                
                correlation_request.performance_metrics = {
                    "total_duration_ms": duration_ms,
                    "component_count": len(correlation_request.component_path),
                }
            
            self.logger.debug(f"Tracked correlation response: {correlation_id}")
            
        except Exception as e:
            self.logger.error(f"Error tracking correlation response: {e}")

    def get_correlation_request(self, correlation_id: str) -> Optional[CorrelationRequest]:
        """Get a correlation request by ID."""
        return self.correlation_requests.get(correlation_id)

    def get_correlation_statistics(self) -> Dict[str, Any]:
        """Get correlation statistics."""
        try:
            total_requests = len(self.correlation_requests)
            active_requests = len([
                req for req in self.correlation_requests.values()
                if req.status == CorrelationStatus.ACTIVE
            ])
            
            completed_requests = len([
                req for req in self.correlation_requests.values()
                if req.status == CorrelationStatus.COMPLETED
            ])
            
            failed_requests = len([
                req for req in self.correlation_requests.values()
                if req.status == CorrelationStatus.FAILED
            ])
            
            # Calculate performance metrics
            avg_duration = 0
            if completed_requests:
                durations = [
                    req.performance_metrics.get("total_duration_ms", 0)
                    for req in self.correlation_requests.values()
                    if req.status == CorrelationStatus.COMPLETED and req.performance_metrics
                ]
                avg_duration = sum(durations) / len(durations) if durations else 0
            
            return {
                "total_requests": total_requests,
                "active_requests": active_requests,
                "completed_requests": completed_requests,
                "failed_requests": failed_requests,
                "success_rate": completed_requests / total_requests if total_requests > 0 else 0,
                "average_duration_ms": avg_duration,
                "correlation_tracking_enabled": self.enable_correlation_tracking,
            }
            
        except Exception as e:
            self.logger.error(f"Error getting correlation statistics: {e}")
            return {}

    def export_correlation_data(self, correlation_id: Optional[str] = None,
                              format: str = "json") -> str:
        """Export correlation data."""
        try:
            if correlation_id:
                correlation_data = self.get_correlation_request(correlation_id)
                if not correlation_data:
                    return "Correlation not found"
                
                if format == "json":
                    return json.dumps(asdict(correlation_data), indent=2, default=str)
                else:
                    return str(correlation_data)
            else:
                # Export all correlations
                all_correlations = {
                    corr_id: asdict(request) for corr_id, request in self.correlation_requests.items()
                }
                
                if format == "json":
                    return json.dumps(all_correlations, indent=2, default=str)
                else:
                    return str(all_correlations)
                    
        except Exception as e:
            self.logger.error(f"Error exporting correlation data: {e}")
            return ""

    @handle_specific_errors(
        error_handlers={
            Exception: (False, "Correlation manager start failed"),
        },
        default_return=False,
        context="correlation manager start",
    )
    async def start(self) -> bool:
        """Start the correlation manager."""
        try:
            self.is_tracking = True
            self.logger.info("🚀 Correlation Manager started")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting correlation manager: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="correlation manager stop",
    )
    async def stop(self) -> None:
        """Stop the correlation manager."""
        try:
            self.is_tracking = False
            self.logger.info("🛑 Correlation Manager stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping correlation manager: {e}")


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="correlation manager setup",
)
async def setup_correlation_manager(config: Dict[str, Any]) -> CorrelationManager | None:
    """
    Setup and initialize correlation manager.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        CorrelationManager instance or None if setup failed
    """
    try:
        correlation_manager = CorrelationManager(config)
        
        if await correlation_manager.initialize():
            return correlation_manager
        else:
            return None
            
    except Exception as e:
        system_logger.error(f"Error setting up correlation manager: {e}")
        return None 