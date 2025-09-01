#!/usr/bin/env python3
"""
Report Scheduler (minimal scaffold)

Automated report scheduling scaffolding.
"""


from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from src.utils.logger import system_logger


class ReportType(Enum):
    PERFORMANCE_SUMMARY = "performance_summary"
    MODEL_ANALYSIS = "model_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    EXECUTIVE_SUMMARY = "executive_summary"
    CONTINUOUS_IMPROVEMENT = "continuous_improvement"


class ReportSchedule(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class ReportFormat(Enum):
    JSON = "json"
    HTML = "html"


@dataclass
class ReportConfig:
    report_type: ReportType
    schedule: ReportSchedule
    format: ReportFormat
    recipients: List[str]
    enabled: bool = True


@dataclass
class ReportHistory:
    report_id: str
    report_type: ReportType
    generated_at: datetime
    schedule_type: ReportSchedule
    recipients: List[str]
    file_path: str
    status: str


