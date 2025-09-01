#!/usr/bin/env python3
"""
Correlation Manager

Centralized correlation ID management and request/response correlation tracking
for the Ares trading bot.
"""


from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from src.utils.error_handler import handle_errors, handle_specific_errors
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


