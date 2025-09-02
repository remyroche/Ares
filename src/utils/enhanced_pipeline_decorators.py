"""
Enhanced Pipeline Decorators for Training Manager
Provides comprehensive decorators, detailed reporting, and consistent storage for all pipeline steps.
"""

import asyncio
import functools
import hashlib
import json
import os
import time
import traceback
import uuid
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

# Handle optional dependencies
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

from src.utils.logger import system_logger
from src.utils.warning_symbols import critical, error, success, warning


class StepStatus(Enum):
    """Step execution status."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    VALIDATED = "validated"


class ReportLevel(Enum):
    """Report detail levels."""

    BASIC = "basic"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"
    DEBUG = "debug"


class EnhancedPipelineDecorator:
    """Enhanced decorator for pipeline steps with comprehensive monitoring and reporting."""

    def __init__(self, step_name: str, report_level: ReportLevel = ReportLevel.DETAILED):
        self.step_name = step_name
        self.report_level = report_level
        self.logger = system_logger.getChild(f"EnhancedPipeline.{step_name}")
        self.reports_dir = Path("reports/enhanced_training_pipeline")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, func: Callable) -> Callable:
        """Apply the enhanced decorator to a function."""

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await self._execute_with_enhanced_monitoring(func, args, kwargs, is_async=True)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(self._execute_with_enhanced_monitoring(func, args, kwargs, is_async=False))

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    async def _execute_with_enhanced_monitoring(self, func: Callable, args: tuple, kwargs: dict, is_async: bool) -> Any:
        """Execute function with comprehensive monitoring and reporting."""

        # Generate unique execution ID
        execution_id = str(uuid.uuid4())
        start_time = datetime.now()
        step_start_time = time.time()

        # Initialize step report
        step_report = {
            "execution_id": execution_id,
            "step_name": self.step_name,
            "start_time": start_time.isoformat(),
            "status": StepStatus.RUNNING.value,
            "report_level": self.report_level.value,
            "pre_execution": {},
            "execution": {},
            "post_execution": {},
            "artifacts": {},
            "performance_metrics": {},
            "validation_results": {},
            "errors": [],
            "warnings": [],
            "recommendations": [],
        }

        try:
            # Pre-execution monitoring
            await self._pre_execution_monitoring(step_report, args, kwargs)

            # Execute the function
            if is_async:
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Post-execution monitoring
            await self._post_execution_monitoring(step_report, result, step_start_time)

            # Update status to success
            step_report["status"] = StepStatus.SUCCESS.value
            step_report["execution"]["result"] = self._serialize_result(result)

            # Generate and store detailed report
            await self._generate_and_store_report(step_report)

            return result

        except Exception as e:
            # Handle execution failure
            step_report["status"] = StepStatus.FAILED.value
            step_report["errors"].append(
                {
                    "type": type(e).__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                    "timestamp": datetime.now().isoformat(),
                }
            )

            # Generate failure report
            await self._generate_and_store_report(step_report)

            # Re-raise the exception
            raise

    async def _pre_execution_monitoring(self, step_report: Dict[str, Any], args: tuple, kwargs: dict):
        """Perform pre-execution monitoring and validation."""

        self.logger.info(f"🚀 [ENHANCED] Starting {self.step_name} with execution ID: {step_report['execution_id']}")

        # System resource monitoring
        if PSUTIL_AVAILABLE:
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent()

            step_report["pre_execution"]["system_resources"] = {
                "memory_usage_percent": memory_info.percent,
                "memory_available_gb": memory_info.available / (1024**3),
                "cpu_usage_percent": cpu_percent,
                "disk_usage_percent": psutil.disk_usage("/").percent,
            }

            # Check for resource warnings
            if memory_info.percent > 80:
                warning_msg = f"High memory usage: {memory_info.percent:.1f}%"
                step_report["warnings"].append(warning_msg)
                self.logger.warning(f"⚠️ {warning_msg}")

        # Input validation
        step_report["pre_execution"]["input_validation"] = {
            "args_count": len(args),
            "kwargs_count": len(kwargs),
            "kwargs_keys": list(kwargs.keys()) if kwargs else [],
        }

        # Data quality checks for pandas DataFrames
        if PANDAS_AVAILABLE:
            data_quality_info = await self._check_data_quality(args, kwargs)
            step_report["pre_execution"]["data_quality"] = data_quality_info

    async def _post_execution_monitoring(self, step_report: Dict[str, Any], result: Any, step_start_time: float):
        """Perform post-execution monitoring and analysis."""

        execution_time = time.time() - step_start_time

        # Performance metrics
        step_report["performance_metrics"] = {
            "execution_time_seconds": execution_time,
            "end_time": datetime.now().isoformat(),
            "duration_formatted": str(timedelta(seconds=execution_time)),
        }

        # System resource monitoring after execution
        if PSUTIL_AVAILABLE:
            memory_info = psutil.virtual_memory()
            step_report["post_execution"]["system_resources"] = {
                "memory_usage_percent": memory_info.percent,
                "memory_available_gb": memory_info.available / (1024**3),
                "cpu_usage_percent": psutil.cpu_percent(),
            }

        # Result analysis
        step_report["post_execution"]["result_analysis"] = await self._analyze_result(result)

        # Performance recommendations
        if execution_time > 300:  # 5 minutes
            step_report["recommendations"].append(
                "Consider optimizing step performance - execution time exceeds 5 minutes"
            )

        if PSUTIL_AVAILABLE and psutil.virtual_memory().percent > 85:
            step_report["recommendations"].append("High memory usage detected - consider memory optimization")

    async def _check_data_quality(self, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Check data quality for pandas DataFrames in arguments."""

        data_quality_info = {}

        try:
            # Check args for DataFrames
            for i, arg in enumerate(args):
                if PANDAS_AVAILABLE and isinstance(arg, pd.DataFrame):
                    data_quality_info[f"arg_{i}"] = {
                        "shape": arg.shape,
                        "memory_usage_mb": arg.memory_usage(deep=True).sum() / (1024**2),
                        "null_counts": arg.isnull().sum().to_dict(),
                        "dtypes": arg.dtypes.to_dict(),
                    }

            # Check kwargs for DataFrames
            for key, value in kwargs.items():
                if PANDAS_AVAILABLE and isinstance(value, pd.DataFrame):
                    data_quality_info[f"kwarg_{key}"] = {
                        "shape": value.shape,
                        "memory_usage_mb": value.memory_usage(deep=True).sum() / (1024**2),
                        "null_counts": value.isnull().sum().to_dict(),
                        "dtypes": value.dtypes.to_dict(),
                    }

        except Exception as e:
            data_quality_info["error"] = str(e)

        return data_quality_info

    async def _analyze_result(self, result: Any) -> Dict[str, Any]:
        """Analyze the result of the step execution."""

        analysis = {"result_type": type(result).__name__, "result_size": None, "result_summary": None}

        try:
            if PANDAS_AVAILABLE and isinstance(result, pd.DataFrame):
                analysis.update(
                    {
                        "result_size": result.shape,
                        "result_summary": {
                            "columns": list(result.columns),
                            "memory_usage_mb": result.memory_usage(deep=True).sum() / (1024**2),
                            "null_counts": result.isnull().sum().to_dict(),
                        },
                    }
                )
            elif isinstance(result, dict):
                analysis.update(
                    {
                        "result_size": len(result),
                        "result_summary": {
                            "keys": list(result.keys()),
                            "nested_structure": self._analyze_dict_structure(result),
                        },
                    }
                )
            elif isinstance(result, (list, tuple)):
                analysis.update(
                    {
                        "result_size": len(result),
                        "result_summary": {
                            "element_types": [type(item).__name__ for item in result[:10]]  # First 10 elements
                        },
                    }
                )
            elif isinstance(result, bool):
                analysis["result_summary"] = {"boolean_value": result}
            else:
                analysis["result_summary"] = {"value": str(result)[:100]}  # Truncate long strings

        except Exception as e:
            analysis["error"] = str(e)

        return analysis

    def _analyze_dict_structure(self, data: dict, max_depth: int = 3, current_depth: int = 0) -> dict:
        """Recursively analyze dictionary structure."""
        if current_depth >= max_depth:
            return {"type": "max_depth_reached"}

        structure = {}
        for key, value in data.items():
            if isinstance(value, dict):
                structure[key] = {
                    "type": "dict",
                    "size": len(value),
                    "structure": self._analyze_dict_structure(value, max_depth, current_depth + 1),
                }
            elif PANDAS_AVAILABLE and isinstance(value, pd.DataFrame):
                structure[key] = {"type": "DataFrame", "shape": value.shape, "columns": list(value.columns)}
            else:
                structure[key] = {
                    "type": type(value).__name__,
                    "value_preview": str(value)[:50] if value is not None else None,
                }

        return structure

    def _serialize_result(self, result: Any) -> Any:
        """Safely serialize result for JSON storage."""
        try:
            if PANDAS_AVAILABLE and isinstance(result, pd.DataFrame):
                return {
                    "type": "DataFrame",
                    "shape": result.shape,
                    "columns": list(result.columns),
                    "sample_data": result.head(5).to_dict() if not result.empty else {},
                }
            elif isinstance(result, dict):
                return {"type": "dict", "keys": list(result.keys()), "size": len(result)}
            elif isinstance(result, (list, tuple)):
                return {
                    "type": type(result).__name__,
                    "size": len(result),
                    "element_types": [type(item).__name__ for item in result[:5]],
                }
            else:
                return {"type": type(result).__name__, "value": str(result)[:200]}  # Truncate long values
        except Exception:
            return {"type": "unserializable", "error": "Failed to serialize result"}

    async def _generate_and_store_report(self, step_report: Dict[str, Any]):
        """Generate and store the detailed step report."""

        # Add completion timestamp
        step_report["completion_time"] = datetime.now().isoformat()

        # Generate report filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.step_name}_{timestamp}_{step_report['execution_id'][:8]}.json"
        report_path = self.reports_dir / filename

        try:
            # Save detailed JSON report
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(step_report, f, indent=2, ensure_ascii=False, default=str)

            # Generate summary report
            summary_report = self._generate_summary_report(step_report)
            summary_filename = f"{self.step_name}_{timestamp}_{step_report['execution_id'][:8]}_summary.txt"
            summary_path = self.reports_dir / summary_filename

            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary_report)

            # Log completion
            status_emoji = "✅" if step_report["status"] == StepStatus.SUCCESS.value else "❌"
            self.logger.info(f"{status_emoji} [ENHANCED] {self.step_name} completed - Report saved to {report_path}")

            # Store report metadata
            await self._store_report_metadata(step_report, report_path, summary_path)

        except Exception as e:
            self.logger.error(f"❌ Failed to save report for {self.step_name}: {e}")

    def _generate_summary_report(self, step_report: Dict[str, Any]) -> str:
        """Generate a human-readable summary report."""

        summary = []
        summary.append("=" * 80)
        summary.append(f"ENHANCED PIPELINE STEP REPORT")
        summary.append("=" * 80)
        summary.append(f"Step Name: {step_report['step_name']}")
        summary.append(f"Execution ID: {step_report['execution_id']}")
        summary.append(f"Status: {step_report['status'].upper()}")
        summary.append(f"Start Time: {step_report['start_time']}")
        summary.append(f"Completion Time: {step_report['completion_time']}")
        summary.append(f"Report Level: {step_report['report_level']}")
        summary.append("")

        # Performance metrics
        if step_report.get("performance_metrics"):
            metrics = step_report["performance_metrics"]
            summary.append("PERFORMANCE METRICS:")
            summary.append("-" * 40)
            summary.append(f"Execution Time: {metrics.get('duration_formatted', 'N/A')}")
            summary.append(f"Duration (seconds): {metrics.get('execution_time_seconds', 'N/A')}")
            summary.append("")

        # System resources
        if step_report.get("pre_execution", {}).get("system_resources"):
            resources = step_report["pre_execution"]["system_resources"]
            summary.append("SYSTEM RESOURCES (Pre-execution):")
            summary.append("-" * 40)
            summary.append(f"Memory Usage: {resources.get('memory_usage_percent', 'N/A')}%")
            summary.append(f"CPU Usage: {resources.get('cpu_usage_percent', 'N/A')}%")
            summary.append(f"Available Memory: {resources.get('memory_available_gb', 'N/A'):.2f} GB")
            summary.append("")

        # Data quality summary
        if step_report.get("pre_execution", {}).get("data_quality"):
            data_quality = step_report["pre_execution"]["data_quality"]
            summary.append("DATA QUALITY SUMMARY:")
            summary.append("-" * 40)
            for key, info in data_quality.items():
                if isinstance(info, dict) and "shape" in info:
                    summary.append(f"{key}: Shape {info['shape']}, Memory {info.get('memory_usage_mb', 'N/A'):.2f} MB")
            summary.append("")

        # Result analysis
        if step_report.get("post_execution", {}).get("result_analysis"):
            analysis = step_report["post_execution"]["result_analysis"]
            summary.append("RESULT ANALYSIS:")
            summary.append("-" * 40)
            summary.append(f"Result Type: {analysis.get('result_type', 'N/A')}")
            if analysis.get("result_size"):
                summary.append(f"Result Size: {analysis['result_size']}")
            summary.append("")

        # Warnings and errors
        if step_report.get("warnings"):
            summary.append("WARNINGS:")
            summary.append("-" * 40)
            for warning in step_report["warnings"]:
                summary.append(f"⚠️ {warning}")
            summary.append("")

        if step_report.get("errors"):
            summary.append("ERRORS:")
            summary.append("-" * 40)
            for error in step_report["errors"]:
                summary.append(f"❌ {error.get('type', 'Unknown')}: {error.get('message', 'No message')}")
            summary.append("")

        # Recommendations
        if step_report.get("recommendations"):
            summary.append("RECOMMENDATIONS:")
            summary.append("-" * 40)
            for rec in step_report["recommendations"]:
                summary.append(f"💡 {rec}")
            summary.append("")

        summary.append("=" * 80)
        summary.append("End of Report")
        summary.append("=" * 80)

        return "\n".join(summary)

    async def _store_report_metadata(self, step_report: Dict[str, Any], report_path: Path, summary_path: Path):
        """Store metadata about the report for indexing and retrieval."""

        metadata = {
            "step_name": step_report["step_name"],
            "execution_id": step_report["execution_id"],
            "status": step_report["status"],
            "start_time": step_report["start_time"],
            "completion_time": step_report["completion_time"],
            "report_path": str(report_path),
            "summary_path": str(summary_path),
            "report_level": step_report["report_level"],
            "performance_metrics": step_report.get("performance_metrics", {}),
            "errors_count": len(step_report.get("errors", [])),
            "warnings_count": len(step_report.get("warnings", [])),
            "recommendations_count": len(step_report.get("recommendations", [])),
        }

        # Store in metadata index
        metadata_file = self.reports_dir / "reports_metadata.json"
        try:
            if metadata_file.exists():
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata_index = json.load(f)
            else:
                metadata_index = []

            metadata_index.append(metadata)

            # Keep only last 1000 reports
            if len(metadata_index) > 1000:
                metadata_index = metadata_index[-1000:]

            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata_index, f, indent=2, ensure_ascii=False)

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to update metadata index: {e}")


# Convenience decorators for different report levels
def enhanced_pipeline_step(step_name: str, report_level: ReportLevel = ReportLevel.DETAILED):
    """Enhanced pipeline step decorator with comprehensive monitoring and reporting."""
    return EnhancedPipelineDecorator(step_name, report_level)


def basic_pipeline_step(step_name: str):
    """Basic pipeline step decorator with minimal reporting."""
    return EnhancedPipelineDecorator(step_name, ReportLevel.BASIC)


def detailed_pipeline_step(step_name: str):
    """Detailed pipeline step decorator with comprehensive reporting."""
    return EnhancedPipelineDecorator(step_name, ReportLevel.DETAILED)


def comprehensive_pipeline_step(step_name: str):
    """Comprehensive pipeline step decorator with full debugging information."""
    return EnhancedPipelineDecorator(step_name, ReportLevel.COMPREHENSIVE)


def debug_pipeline_step(step_name: str):
    """Debug pipeline step decorator with maximum detail."""
    return EnhancedPipelineDecorator(step_name, ReportLevel.DEBUG)


# Utility functions for report management
async def get_step_reports(step_name: str = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve step reports from the metadata index."""

    reports_dir = Path("reports/enhanced_training_pipeline")
    metadata_file = reports_dir / "reports_metadata.json"

    if not metadata_file.exists():
        return []

    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata_index = json.load(f)

        # Filter by step name if provided
        if step_name:
            metadata_index = [m for m in metadata_index if m["step_name"] == step_name]

        # Sort by completion time (newest first) and limit
        metadata_index.sort(key=lambda x: x["completion_time"], reverse=True)
        return metadata_index[:limit]

    except Exception as e:
        system_logger.error(f"Failed to retrieve step reports: {e}")
        return []


async def get_latest_step_report(step_name: str) -> Optional[Dict[str, Any]]:
    """Get the latest report for a specific step."""

    reports = await get_step_reports(step_name, limit=1)
    return reports[0] if reports else None


async def cleanup_old_reports(days_to_keep: int = 30):
    """Clean up old reports to save disk space."""

    reports_dir = Path("reports/enhanced_training_pipeline")
    if not reports_dir.exists():
        return

    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    cleaned_count = 0

    try:
        metadata_file = reports_dir / "reports_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata_index = json.load(f)

            # Filter out old reports
            old_reports = []
            for metadata in metadata_index:
                report_date = datetime.fromisoformat(metadata["completion_time"])
                if report_date < cutoff_date:
                    old_reports.append(metadata)

            # Remove old report files
            for old_report in old_reports:
                try:
                    report_path = Path(old_report["report_path"])
                    summary_path = Path(old_report["summary_path"])

                    if report_path.exists():
                        report_path.unlink()
                        cleaned_count += 1

                    if summary_path.exists():
                        summary_path.unlink()
                        cleaned_count += 1

                except Exception as e:
                    system_logger.warning(f"Failed to remove old report file: {e}")

            # Update metadata index
            metadata_index = [m for m in metadata_index if m not in old_reports]
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata_index, f, indent=2, ensure_ascii=False)

            system_logger.info(f"🧹 Cleaned up {cleaned_count} old report files")

    except Exception as e:
        system_logger.error(f"Failed to cleanup old reports: {e}")


# Export the main decorator for easy import
__all__ = [
    "enhanced_pipeline_step",
    "basic_pipeline_step",
    "detailed_pipeline_step",
    "comprehensive_pipeline_step",
    "debug_pipeline_step",
    "get_step_reports",
    "get_latest_step_report",
    "cleanup_old_reports",
    "StepStatus",
    "ReportLevel",
]
