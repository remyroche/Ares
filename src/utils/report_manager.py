#!/usr/bin/env python3
"""Centralized Report Manager for Ares Trading System.

This module provides a unified interface for managing all reports generated
during pipeline execution, ensuring they are organized in a consistent
reports/run_DATETIME/ folder structure with proper naming conventions.

Features:
- Centralized report directory management
- Consistent naming conventions for all report types
- Support for step reports, ML interpretability reports, and other reports
- Automatic timestamp-based folder creation
- Report aggregation and summary generation
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import shutil

from src.utils.common_operations import (
    get_current_datetime, format_datetime, ensure_directory,
    safe_json_dump, safe_json_load, safe_file_exists
)
from src.utils.logger import system_logger


class ReportManager:
    """Centralized manager for all report generation and organization."""
    
    def __init__(self, base_reports_dir: str = "reports"):
        """Initialize the report manager.
        
        Args:
            base_reports_dir: Base directory for all reports
        """
        self.base_reports_dir = Path(base_reports_dir)
        self.logger = system_logger.getChild("ReportManager")
        self.current_run_dir = None
        self.run_timestamp = None
        self._initialize_run_directory()
    
    def _initialize_run_directory(self):
        """Initialize the current run directory with timestamp."""
        self.run_timestamp = format_datetime(get_current_datetime(), '%Y%m%d_%H%M%S')
        self.current_run_dir = self.base_reports_dir / f"run_{self.run_timestamp}"
        ensure_directory(str(self.current_run_dir))
        
        self.logger.info(f"📁 Initialized report directory: {self.current_run_dir}")
        print(f"📁 Report directory: {self.current_run_dir}")
    
    def get_run_directory(self) -> Path:
        """Get the current run directory path."""
        return self.current_run_dir
    
    def get_run_timestamp(self) -> str:
        """Get the current run timestamp."""
        return self.run_timestamp
    
    def create_step_report_path(
        self, 
        step_name: str, 
        symbol: str, 
        exchange: str, 
        file_extension: str = "json"
    ) -> Path:
        """Create a standardized path for step reports.
        
        Args:
            step_name: Name of the pipeline step
            symbol: Trading symbol (e.g., ETHUSDT)
            exchange: Trading exchange (e.g., BINANCE)
            file_extension: File extension (json, md, txt, html)
            
        Returns:
            Path object for the step report
        """
        filename = f"step_report_{step_name}_{symbol}_{exchange}.{file_extension}"
        return self.current_run_dir / filename
    
    def create_ml_interpretability_report_path(
        self,
        model_type: str,
        symbol: str,
        exchange: str,
        file_extension: str = "json"
    ) -> Path:
        """Create a standardized path for ML interpretability reports.
        
        Args:
            model_type: Type of model (e.g., hmm, tactician, analyst)
            symbol: Trading symbol (e.g., ETHUSDT)
            exchange: Trading exchange (e.g., BINANCE)
            file_extension: File extension (json, md, txt, html)
            
        Returns:
            Path object for the ML interpretability report
        """
        filename = f"ml_interpretability_{model_type}_{symbol}_{exchange}.{file_extension}"
        return self.current_run_dir / filename
    
    def create_general_report_path(
        self,
        report_type: str,
        symbol: str,
        exchange: str,
        file_extension: str = "json"
    ) -> Path:
        """Create a standardized path for general reports.
        
        Args:
            report_type: Type of report (e.g., pipeline_summary, validation)
            symbol: Trading symbol (e.g., ETHUSDT)
            exchange: Trading exchange (e.g., BINANCE)
            file_extension: File extension (json, md, txt, html)
            
        Returns:
            Path object for the general report
        """
        filename = f"{report_type}_{symbol}_{exchange}.{file_extension}"
        return self.current_run_dir / filename
    
    def save_step_report(
        self,
        step_name: str,
        symbol: str,
        exchange: str,
        report_data: Dict[str, Any],
        file_extension: str = "json"
    ) -> Path:
        """Save a step report with standardized naming and location.
        
        Args:
            step_name: Name of the pipeline step
            symbol: Trading symbol
            exchange: Trading exchange
            report_data: Report data to save
            file_extension: File extension
            
        Returns:
            Path to the saved report file
        """
        report_path = self.create_step_report_path(step_name, symbol, exchange, file_extension)
        
        # Add metadata to report
        enhanced_report_data = {
            "report_metadata": {
                "report_type": "step_report",
                "step_name": step_name,
                "symbol": symbol,
                "exchange": exchange,
                "generated_at": format_datetime(get_current_datetime()),
                "run_timestamp": self.run_timestamp,
                "report_manager_version": "1.0"
            },
            "report_content": report_data
        }
        
        if file_extension == "json":
            safe_json_dump(enhanced_report_data, str(report_path), indent=2)
        else:
            with open(report_path, 'w', encoding='utf-8') as f:
                if file_extension == "md":
                    f.write(self._format_markdown_report(enhanced_report_data))
                elif file_extension == "txt":
                    f.write(self._format_text_report(enhanced_report_data))
                else:
                    f.write(str(enhanced_report_data))
        
        self.logger.info(f"💾 Step report saved: {report_path}")
        print(f"💾 Step report saved: {report_path}")
        return report_path
    
    def save_ml_interpretability_report(
        self,
        model_type: str,
        symbol: str,
        exchange: str,
        report_data: Dict[str, Any],
        file_extension: str = "json"
    ) -> Path:
        """Save an ML interpretability report with standardized naming and location.
        
        Args:
            model_type: Type of model
            symbol: Trading symbol
            exchange: Trading exchange
            report_data: Report data to save
            file_extension: File extension
            
        Returns:
            Path to the saved report file
        """
        report_path = self.create_ml_interpretability_report_path(model_type, symbol, exchange, file_extension)
        
        # Add metadata to report
        enhanced_report_data = {
            "report_metadata": {
                "report_type": "ml_interpretability",
                "model_type": model_type,
                "symbol": symbol,
                "exchange": exchange,
                "generated_at": format_datetime(get_current_datetime()),
                "run_timestamp": self.run_timestamp,
                "report_manager_version": "1.0"
            },
            "report_content": report_data
        }
        
        if file_extension == "json":
            safe_json_dump(enhanced_report_data, str(report_path), indent=2)
        else:
            with open(report_path, 'w', encoding='utf-8') as f:
                if file_extension == "md":
                    f.write(self._format_markdown_report(enhanced_report_data))
                elif file_extension == "txt":
                    f.write(self._format_text_report(enhanced_report_data))
                else:
                    f.write(str(enhanced_report_data))
        
        self.logger.info(f"💾 ML interpretability report saved: {report_path}")
        print(f"💾 ML interpretability report saved: {report_path}")
        return report_path
    
    def save_general_report(
        self,
        report_type: str,
        symbol: str,
        exchange: str,
        report_data: Dict[str, Any],
        file_extension: str = "json"
    ) -> Path:
        """Save a general report with standardized naming and location.
        
        Args:
            report_type: Type of report
            symbol: Trading symbol
            exchange: Trading exchange
            report_data: Report data to save
            file_extension: File extension
            
        Returns:
            Path to the saved report file
        """
        report_path = self.create_general_report_path(report_type, symbol, exchange, file_extension)
        
        # Add metadata to report
        enhanced_report_data = {
            "report_metadata": {
                "report_type": report_type,
                "symbol": symbol,
                "exchange": exchange,
                "generated_at": format_datetime(get_current_datetime()),
                "run_timestamp": self.run_timestamp,
                "report_manager_version": "1.0"
            },
            "report_content": report_data
        }
        
        if file_extension == "json":
            safe_json_dump(enhanced_report_data, str(report_path), indent=2)
        else:
            with open(report_path, 'w', encoding='utf-8') as f:
                if file_extension == "md":
                    f.write(self._format_markdown_report(enhanced_report_data))
                elif file_extension == "txt":
                    f.write(self._format_text_report(enhanced_report_data))
                else:
                    f.write(str(enhanced_report_data))
        
        self.logger.info(f"💾 General report saved: {report_path}")
        print(f"💾 General report saved: {report_path}")
        return report_path
    
    def copy_existing_report(
        self,
        source_path: Union[str, Path],
        target_name: str,
        symbol: str,
        exchange: str
    ) -> Path:
        """Copy an existing report to the current run directory with standardized naming.
        
        Args:
            source_path: Path to the existing report
            target_name: Name for the copied report (without extension)
            symbol: Trading symbol
            exchange: Trading exchange
            
        Returns:
            Path to the copied report file
        """
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source report not found: {source_path}")
        
        # Determine file extension
        file_extension = source_path.suffix.lstrip('.')
        if not file_extension:
            file_extension = "txt"
        
        # Create target path
        target_path = self.current_run_dir / f"{target_name}_{symbol}_{exchange}.{file_extension}"
        
        # Copy the file
        shutil.copy2(source_path, target_path)
        
        self.logger.info(f"📋 Copied report: {source_path} -> {target_path}")
        print(f"📋 Copied report: {source_path} -> {target_path}")
        return target_path
    
    def generate_run_summary(self, symbol: str, exchange: str) -> Path:
        """Generate a summary report of all reports in the current run.
        
        Args:
            symbol: Trading symbol
            exchange: Trading exchange
            
        Returns:
            Path to the summary report
        """
        # Collect all reports in the current run directory
        reports = []
        for file_path in self.current_run_dir.glob("*"):
            if file_path.is_file():
                reports.append({
                    "filename": file_path.name,
                    "size_bytes": file_path.stat().st_size,
                    "modified_time": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    "file_type": file_path.suffix.lstrip('.')
                })
        
        # Generate summary data
        summary_data = {
            "run_summary": {
                "run_timestamp": self.run_timestamp,
                "symbol": symbol,
                "exchange": exchange,
                "total_reports": len(reports),
                "run_directory": str(self.current_run_dir),
                "generated_at": format_datetime(get_current_datetime())
            },
            "reports": reports,
            "report_categories": {
                "step_reports": len([r for r in reports if r["filename"].startswith("step_report_")]),
                "ml_interpretability_reports": len([r for r in reports if r["filename"].startswith("ml_interpretability_")]),
                "other_reports": len([r for r in reports if not r["filename"].startswith(("step_report_", "ml_interpretability_"))])
            }
        }
        
        # Save summary report
        summary_path = self.create_general_report_path("run_summary", symbol, exchange, "json")
        safe_json_dump(summary_data, str(summary_path), indent=2)
        
        self.logger.info(f"📊 Run summary generated: {summary_path}")
        print(f"📊 Run summary generated: {summary_path}")
        return summary_path
    
    def _format_markdown_report(self, report_data: Dict[str, Any]) -> str:
        """Format report data as markdown."""
        metadata = report_data.get("report_metadata", {})
        content = report_data.get("report_content", {})
        
        markdown_lines = [
            f"# {metadata.get('report_type', 'Report').title()} Report",
            "",
            f"**Generated:** {metadata.get('generated_at', 'N/A')}",
            f"**Run Timestamp:** {metadata.get('run_timestamp', 'N/A')}",
            f"**Symbol:** {metadata.get('symbol', 'N/A')}",
            f"**Exchange:** {metadata.get('exchange', 'N/A')}",
            "",
            "## Report Content",
            "",
            "```json",
            json.dumps(content, indent=2, default=str),
            "```"
        ]
        
        return "\n".join(markdown_lines)
    
    def _format_text_report(self, report_data: Dict[str, Any]) -> str:
        """Format report data as plain text."""
        metadata = report_data.get("report_metadata", {})
        content = report_data.get("report_content", {})
        
        text_lines = [
            f"{metadata.get('report_type', 'Report').title()} Report",
            "=" * 50,
            f"Generated: {metadata.get('generated_at', 'N/A')}",
            f"Run Timestamp: {metadata.get('run_timestamp', 'N/A')}",
            f"Symbol: {metadata.get('symbol', 'N/A')}",
            f"Exchange: {metadata.get('exchange', 'N/A')}",
            "",
            "Report Content:",
            "-" * 20,
            json.dumps(content, indent=2, default=str)
        ]
        
        return "\n".join(text_lines)


# Global report manager instance
_global_report_manager = None


def get_report_manager() -> ReportManager:
    """Get the global report manager instance."""
    global _global_report_manager
    if _global_report_manager is None:
        _global_report_manager = ReportManager()
    return _global_report_manager


def initialize_report_manager(base_reports_dir: str = "reports") -> ReportManager:
    """Initialize the global report manager with custom base directory."""
    global _global_report_manager
    _global_report_manager = ReportManager(base_reports_dir)
    return _global_report_manager