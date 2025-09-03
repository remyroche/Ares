#!/usr/bin/env python3
"""Correlation Manager.

Centralized correlation ID management and request/response correlation tracking for the
Ares trading bot.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from src.core.decorators import handles_errors
from src.utils.logger import system_logger


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
    """Centralized correlation ID management and request/response correlation
    tracking."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("CorrelationManager")

        # Correlation configuration
        self.correlation_config = config.get("correlation_manager", {})
        self.enable_correlation_tracking: bool = bool(
            self.correlation_config.get("enable_correlation_tracking", True)
        )
        self.correlation_timeout: int = int(
            self.correlation_config.get("correlation_timeout", 300)
        )
        self.max_correlation_history: int = int(
            self.correlation_config.get("max_correlation_history", 10000)
        )

        # Correlation storage
        self.correlation_requests: Dict[str, CorrelationRequest] = {}
        self.is_tracking: bool = False

        self.logger.info("🔗 Correlation Manager initialized")

    @handles_errors(
        error_handlers={
            ValueError: (False, "Invalid correlation configuration"),
            AttributeError: (False, "Missing required correlation parameters"),
        },
        default_return=False,
        context="correlation_manager.initialize",
    )
    async def initialize(self) -> bool:
        """Initialize the correlation manager."""
        self.logger.info("Initializing Correlation Manager...")
        self.correlation_requests.clear()
        self.is_tracking = True
        self.logger.info("✅ Correlation Manager initialization completed")
        return True

    @handles_errors(fallback=None)
    async def track_correlation_request(
        self,
        correlation_id: str,
        component_path: List[str],
        request_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        request = CorrelationRequest(
            correlation_id=correlation_id,
            request_timestamp=datetime.now(),
            status=CorrelationStatus.ACTIVE,
            component_path=list(component_path),
            request_data=dict(request_data),
            performance_metrics={},
            metadata=dict(metadata or {}),
        )
        self.correlation_requests[correlation_id] = request

        # Enforce history limit
        if len(self.correlation_requests) > self.max_correlation_history:
            oldest_key = next(iter(self.correlation_requests))
            self.correlation_requests.pop(oldest_key, None)

    @handles_errors(fallback=None)
    async def complete_correlation_request(
        self,
        correlation_id: str,
        response_data: Optional[Dict[str, Any]] = None,
        error_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        req = self.correlation_requests.get(correlation_id)
        if not req:
            return
        req.response_timestamp = datetime.now()
        req.response_data = dict(response_data or {})
        req.error_info = dict(error_info or {}) if error_info else None
        req.status = (
            CorrelationStatus.FAILED if error_info else CorrelationStatus.COMPLETED
        )

    def get_request(self, correlation_id: str) -> Optional[CorrelationRequest]:
        return self.correlation_requests.get(correlation_id)

    def list_requests(self) -> List[CorrelationRequest]:
        return list(self.correlation_requests.values())
