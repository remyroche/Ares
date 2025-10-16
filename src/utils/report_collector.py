#!/usr/bin/env python3
from src.utils.tprint import tprint

from .logger import system_logger
"""Comprehensive Report Collector for Ares Trading System.

This module provides a centralized system to collect and organize ALL reports
generated throughout the pipeline execution, ensuring they are all stored in
the same reports/run_DATETIME/ folder with proper naming conventions.

Features:
- Intercepts all report generation across the pipeline
- Redirects reports to centralized location
- Maintains original report content and formatting
- Provides fallback for existing report generation methods
- Ensures no reports are missed during pipeline execution
"""

from pathlib import Path
import shutil
import functools

from .utils.common_operations import (
    get_current_datetime, format_datetime, ensure_directory,
)
import logging
import time
import typing

class ReportCollector:
    """Centralized collector for all pipeline reports."""

    def __init__(self, base_reports_dir: str = "reports"):
        """Initialize the report collector."""
        self.base_reports_dir = Path(base_reports_dir)
        self.logger = system_logger.getChild("ReportCollector")
        self.current_run_dir = None
        self.run_timestamp = None
        self.collected_reports = []
        self._initialize_run_directory()

        # Track original functions to restore later
        self.original_functions = {}

    def _initialize_run_directory(self):
        """Initialize the current run directory with timestamp."""
        self.run_timestamp = format_datetime(get_current_datetime(), '%Y%m%d_%H%M%S')
        self.current_run_dir = self.base_reports_dir / f"run_{self.run_timestamp}"

        self.logger.info(f"📁 Report collector initialized: {self.current_run_dir}")
        tprint(f"📁 Report collector directory: {self.current_run_dir}")

    def get_run_directory(self) -> Path:
        """Get the current run directory path."""
        return self.current_run_dir

    def get_run_timestamp(self) -> str:
        """Get the current run timestamp."""
        return self.run_timestamp

    def collect_report(
        self,
        report_content: str,
        report_name: str,
        report_type: str = "general",
        symbol: str = "UNKNOWN",
        exchange: str = "UNKNOWN",
        step_name: str = None
    ) -> Path:
        """Collect a report and save it to the centralized location.

        Args:
            report_content: The report content (string)
            report_name: Name for the report file
            report_type: Type of report (step, ml_interpretability, general)
            symbol: Trading symbol
            exchange: Trading exchange
            step_name: Pipeline step name (for step reports)

        Returns:
            Path to the saved report file
        """
        try:
            # Determine file extension
            if report_content.strip().startswith('{'):
                # JSON content
                file_extension = "json"
            elif report_content.strip().startswith('#'):
                # Markdown content
                file_extension = "md"
            else:
                # Assume text content
                file_extension = "txt"

            # Create standardized filename
            if report_type == "step" and step_name:
                filename = f"step_report_{step_name}_{symbol}_{exchange}.{file_extension}"
            elif report_type == "ml_interpretability":
                filename = f"ml_interpretability_{report_name}_{symbol}_{exchange}.{file_extension}"
            else:
                filename = f"{report_name}_{symbol}_{exchange}.{file_extension}"

            report_path = self.current_run_dir / filename

            # Save the report
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)

            # Track the collected report
            self.collected_reports.append({
                "filename": filename,
                "report_type": report_type,
                "step_name": step_name,
                "symbol": symbol,
                "exchange": exchange,
                "size_bytes": report_path.stat().st_size,
                "generated_at": format_datetime(get_current_datetime())
            })

            self.logger.info(f"📄 Collected report: {filename}")
            tprint(f"📄 Collected report: {filename}")
            return report_path

        except Exception as e:
            self.logger.error(f"❌ Failed to collect report {report_name}: {e}")
            return None

    def intercept_report_generation(self, module_name: str, function_name: str):
        """Intercept report generation in a specific module/function.

        Args:
            module_name: Name of the module containing the report function
            function_name: Name of the report generation function
        """
        try:
            # Import the module
            module = __import__(module_name, fromlist=[function_name])
            original_func = getattr(module, function_name)

            # Store original function
            self.original_functions[f"{module_name}.{function_name}"] = original_func

            # Create intercepted version
            @functools.wraps(original_func)
            def intercepted_func(*args, **kwargs):
                # Call original function
                result = original_func(*args, **kwargs)

                # If result is a string (report content), collect it
                if isinstance(result, str) and len(result) > 100:  # Likely a report
                    # Try to extract context from args/kwargs
                    symbol = kwargs.get('symbol', 'UNKNOWN')
                    exchange = kwargs.get('exchange', 'UNKNOWN')
                    step_name = kwargs.get('step_name', None)

                    # Determine report type based on function name
                    if 'step' in function_name.lower():
                        report_type = "step"
                    elif 'ml' in function_name.lower() or 'interpretability' in function_name.lower():
                        report_type = "ml_interpretability"
                    else:
                        report_type = "general"

                    # Collect the report
                    self.collect_report(
                        report_content = result,
                        report_name = function_name,
                        report_type = report_type,
                        symbol = symbol,
                        exchange = exchange,
                        step_name = step_name
                    )

                return result

            # Replace the function
            setattr(module, function_name, intercepted_func)
            self.logger.info(f"🔗 Intercepted {module_name}.{function_name}")

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to intercept {module_name}.{function_name}: {e}")

    def copy_existing_report(
        self,
        source_path: Union[str, Path],
        report_name: str,
        report_type: str = "general",
        symbol: str = "UNKNOWN",
        exchange: str = "UNKNOWN",
        step_name: str = None
    ) -> Path:
        """Copy an existing report to the centralized location.

        Args:
            source_path: Path to the existing report
            report_name: Name for the copied report
            report_type: Type of report
            symbol: Trading symbol
            exchange: Trading exchange
            step_name: Pipeline step name

        Returns:
            Path to the copied report file
        """
        try:
            source_path = Path(source_path)
            if not source_path.exists():
                self.logger.warning(f"⚠️ Source report not found: {source_path}")
                return None

            # Determine file extension
            file_extension = source_path.suffix.lstrip('.')
            if not file_extension:
                file_extension = "txt"

            # Create standardized filename
            if report_type == "step" and step_name:
                filename = f"step_report_{step_name}_{symbol}_{exchange}.{file_extension}"
            elif report_type == "ml_interpretability":
                filename = f"ml_interpretability_{report_name}_{symbol}_{exchange}.{file_extension}"
            else:
                filename = f"{report_name}_{symbol}_{exchange}.{file_extension}"

            target_path = self.current_run_dir / filename

            # Copy the file
            shutil.copy2(source_path, target_path)

            # Track the collected report
            self.collected_reports.append({
                "filename": filename,
                "report_type": report_type,
                "step_name": step_name,
                "symbol": symbol,
                "exchange": exchange,
                "size_bytes": target_path.stat().st_size,
                "generated_at": format_datetime(get_current_datetime()),
                "source_path": str(source_path)
            })

            self.logger.info(f"📋 Copied report: {filename}")
            tprint(f"📋 Copied report: {filename}")
            return target_path

        except Exception as e:
            self.logger.error(f"❌ Failed to copy report {source_path}: {e}")
            return None

    def generate_collection_summary(self, symbol: str, exchange: str) -> Path:
        """Generate a summary of all collected reports.

        Args:
            symbol: Trading symbol
            exchange: Trading exchange

        Returns:
            Path to the collection summary file
        """
        try:
            # Group reports by type
            report_groups = {}
            for report in self.collected_reports:
                report_type = report["report_type"]
                if report_type not in report_groups:
                    report_groups[report_type] = []
                report_groups[report_type].append(report)

            # Generate summary content
            lines = [
                "=" * 80,
                "COMPREHENSIVE REPORT COLLECTION SUMMARY",
                "=" * 80,
                "",
                "📋 COLLECTION INFORMATION",
                "-" * 40,
                f"Run Timestamp:    {self.run_timestamp}",
                f"Symbol:           {symbol}",
                f"Exchange:         {exchange}",
                f"Total Reports:    {len(self.collected_reports)}",
                f"Collection Directory: {self.current_run_dir}",
                f"Generated:        {format_datetime(get_current_datetime())}",
                "",
                "📊 REPORT BREAKDOWN BY TYPE",
                "-" * 40
            ]

            for report_type, reports in report_groups.items():
                lines.append(f"{report_type.replace('_', ' ').title()}: {len(reports)}")

            lines.extend([
                "",
                "📁 COLLECTED REPORTS",
                "-" * 40
            ])

            for i, report in enumerate(self.collected_reports, 1):
                lines.append(f"{i:2d}. {report['filename']} ({report['size_bytes']} bytes)")
                if report.get('source_path'):
                    lines.append(f"     Source: {report['source_path']}")

            lines.extend([
                "",
                "=" * 80,
                "Report generated by Ares Trading System Report Collector v1.0",
                "=" * 80
            ])

            # Save summary
            summary_path = self.current_run_dir / f"report_collection_summary_{symbol}_{exchange}.txt"
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))

            self.logger.info(f"📊 Collection summary generated: {summary_path}")
            tprint(f"📊 Collection summary generated: {summary_path}")
            return summary_path

        except Exception as e:
            self.logger.error(f"❌ Failed to generate collection summary: {e}")
            return None

    def setup_pipeline_interception(self, symbol: str, exchange: str):
        """Set up interception for common report generation functions.

        Args:
            symbol: Trading symbol
            exchange: Trading exchange
        """
        try:
            # List of common report generation functions to intercept
            interceptions = [
                # Data quality monitoring
                ("src.training.steps.market_analysis.step1.data_quality_monitor", "generate_monitoring_report"),
                ("src.training.steps.market_analysis.step1.data_gap_detector", "generate_missing_data_report"),

                # Step-specific reports
                ("src.training.steps.model_training.step05_labeling", "_generate_labeling_reports"),
                ("src.training.steps.model_training.step10_unified_regime_intelligence_validator", "_generate_validation_report"),

                # Optimization reports
                ("src.training.steps.model_training.validation.step17_final_parameters_optimization", "_generate_optimization_report"),
                ("src.training.steps.optimisation.step17_final_parameters_optimization_new", "_generate_optimization_report"),
                ("src.training.steps.market_analysis.step17_final_parameters_optimization.regime_specific_triple_barrier_optimization", "_generate_optimization_report"),
                ("src.training.steps.market_analysis.step17_final_parameters_optimization.step17_probabilistic_bayesian_optimization", "_generate_final_report"),
                ("src.training.steps.market_analysis.step17_final_parameters_optimization.sr_optuna_optimization", "generate_optimization_report"),

                # Feature selection reports
                ("src.training.steps.market_analysis.step08_advanced_feature_selection", "_generate_interpretability_report"),
            ]

            for module_name, function_name in interceptions:
                try:
                    self.intercept_report_generation(module_name, function_name)
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not intercept {module_name}.{function_name}: {e}")

            self.logger.info(f"🔗 Set up interception for {len(interceptions)} report functions")

        except Exception as e:
            self.logger.error(f"❌ Failed to setup pipeline interception: {e}")

    def restore_original_functions(self):
        """Restore original functions after interception."""
        try:
            for func_path, original_func in self.original_functions.items():
                module_name, function_name = func_path.rsplit('.', 1)
                module = __import__(module_name, fromlist=[function_name])
                setattr(module, function_name, original_func)

            self.logger.info(f"🔄 Restored {len(self.original_functions)} original functions")

        except Exception as e:
            self.logger.error(f"❌ Failed to restore original functions: {e}")

# Global report collector instance
_global_report_collector = None

def get_report_collector() -> ReportCollector:
    """Get the global report collector instance."""
    global _global_report_collector
    if _global_report_collector is None:
        _global_report_collector = ReportCollector()
    return _global_report_collector

def initialize_report_collector(base_reports_dir: str = "reports") -> ReportCollector:
    """Initialize the global report collector with custom base directory."""
    global _global_report_collector
    _global_report_collector = ReportCollector(base_reports_dir)
    return _global_report_collector

def collect_report(
    report_content: str,
    report_name: str,
    report_type: str = "general",
    symbol: str = "UNKNOWN",
    exchange: str = "UNKNOWN",
    step_name: str = None
) -> Optional[Path]:
    """Convenience function to collect a report using the global collector."""
    collector = get_report_collector()
    return collector.collect_report(
        report_content = report_content,
        report_name = report_name,
        report_type = report_type,
        symbol = symbol,
        exchange = exchange,
        step_name = step_name
    )
