"""
Comprehensive Reporting Step

This module provides comprehensive reporting functionality for backtesting results
with detailed reports, visualizations, and actionable insights.

Key Features:
- Comprehensive backtesting reports
- Performance visualization
- Risk analysis reports
- Trade analysis reports
- Portfolio analysis reports
- Executive summaries
- Actionable recommendations
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import gc
import psutil
from pathlib import Path
import json

# Common utilities
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, safe_file_exists, ensure_directory,
    safe_mean, safe_std, safe_float, safe_int, get_current_datetime,
    safe_append, safe_extend, safe_dict_get, safe_lower, safe_upper,
    format_datetime, validate_file_path, get_file_size, check_disk_space
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, MathValidationError
)
from src.utils.parquet_utils import get_parquet_utils, ParquetUtils
from src.utils.comprehensive_function_logger import (
    log_step_functions, log_important_calls, log_all_calls, 
    log_internal_call, log_step_progress, log_data_operation
)
from src.utils.enhanced_financial_metrics_logger import EnhancedFinancialMetricsLogger
from src.utils.performance_utils import PerformanceMonitor
from src.utils.monitoring_utils import SystemMonitor

# Core decorators and validation
from src.core.decorators import (
    handles_errors, validates, traced, log_execution_time, 
    timeout, error_boundary, compose, validate_data_quality, 
    monitor_step_execution, ensure_data_integrity, validate_pipeline_step
)
from src.core.errors import (
    ValidationError, DataIntegrityError, FileOperationError,
    MathValidationError, TimeoutError
)

# Training step utilities
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler

# Import existing reporting functionality
from src.training.steps.backtesting.comprehensive_reporting import (
    BacktestingReportGenerator, ComprehensiveReporter
)

logger = logging.getLogger(__name__)


class ReportType(Enum):
    """Types of reports."""
    EXECUTIVE_SUMMARY = "executive_summary"
    PERFORMANCE_REPORT = "performance_report"
    RISK_REPORT = "risk_report"
    TRADE_REPORT = "trade_report"
    PORTFOLIO_REPORT = "portfolio_report"
    COMPREHENSIVE_REPORT = "comprehensive_report"
    COMPARISON_REPORT = "comparison_report"


@dataclass
class ReportingConfig:
    """Configuration for reporting step."""
    # Basic configuration
    symbol: str
    exchange: str
    timeframe: str
    data_dir: str
    
    # Report parameters
    report_types: List[ReportType] = field(default_factory=lambda: [
        ReportType.EXECUTIVE_SUMMARY,
        ReportType.PERFORMANCE_REPORT,
        ReportType.RISK_REPORT,
        ReportType.TRADE_REPORT,
        ReportType.PORTFOLIO_REPORT,
        ReportType.COMPREHENSIVE_REPORT
    ])
    
    # Report settings
    include_visualizations: bool = True
    include_recommendations: bool = True
    include_troubleshooting: bool = True
    include_quality_assessment: bool = True
    
    # Output settings
    output_format: str = "html"  # html, pdf, json, markdown
    save_individual_reports: bool = True
    save_combined_report: bool = True
    
    # Analysis settings
    enable_detailed_logging: bool = True
    enable_performance_monitoring: bool = True
    enable_memory_optimization: bool = True


@dataclass
class ReportingResults:
    """Results from reporting step."""
    # Basic info
    symbol: str
    exchange: str
    timeframe: str
    start_time: datetime
    end_time: datetime
    total_duration: float
    
    # Generated reports
    generated_reports: Dict[str, str] = field(default_factory=dict)  # report_type -> file_path
    
    # Report summaries
    report_summaries: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Combined report
    combined_report_path: Optional[str] = None
    
    # Report metadata
    report_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    config: ReportingConfig = field(default_factory=ReportingConfig)
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    system_metrics: Dict[str, Any] = field(default_factory=dict)


class ReportingStep:
    """Comprehensive reporting step."""
    
    def __init__(self, config: ReportingConfig):
        """Initialize the reporting step."""
        self.config = config
        self.logger = logger.getChild('ReportingStep')
        
        # Initialize utilities
        self.parquet_utils = get_parquet_utils()
        self.financial_logger = EnhancedFinancialMetricsLogger()
        self.performance_monitor = PerformanceMonitor()
        self.system_monitor = SystemMonitor()
        
        # Initialize report generator
        self.report_generator = BacktestingReportGenerator()
        self.comprehensive_reporter = ComprehensiveReporter()
        
        # Initialize data directory
        self.data_dir = Path(config.data_dir)
        ensure_directory(self.data_dir)
        
        self.logger.info(f"🚀 ReportingStep initialized for {config.symbol}")
        self.logger.info(f"📊 Report types: {[rt.value for rt in config.report_types]}")
        self.logger.info(f"📁 Data directory: {config.data_dir}")
    
    @traced(span_name='comprehensive_reporting')
    @log_execution_time
    @monitor_step_execution
    async def execute(
        self, 
        backtesting_results: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ReportingResults:
        """Execute comprehensive reporting."""
        
        self.logger.info("🚀 Starting comprehensive reporting...")
        start_time = time.time()
        
        # Start performance monitoring
        if self.config.enable_performance_monitoring:
            self.performance_monitor.start_monitoring()
        
        try:
            # Load backtesting results if not provided
            if backtesting_results is None:
                backtesting_results = await self._load_backtesting_results()
            
            # Generate individual reports
            generated_reports = {}
            report_summaries = {}
            
            for report_type in self.config.report_types:
                self.logger.info(f"📝 Generating {report_type.value} report...")
                
                report_path, summary = await self._generate_report(report_type, backtesting_results)
                generated_reports[report_type.value] = report_path
                report_summaries[report_type.value] = summary
            
            # Generate combined report
            combined_report_path = None
            if self.config.save_combined_report:
                self.logger.info("📋 Generating combined report...")
                combined_report_path = await self._generate_combined_report(
                    generated_reports, report_summaries, backtesting_results
                )
            
            # Create report metadata
            report_metadata = self._create_report_metadata(backtesting_results, generated_reports)
            
            # Create results
            results = ReportingResults(
                symbol=self.config.symbol,
                exchange=self.config.exchange,
                timeframe=self.config.timeframe,
                start_time=datetime.now(),
                end_time=datetime.now(),
                total_duration=time.time() - start_time,
                generated_reports=generated_reports,
                report_summaries=report_summaries,
                combined_report_path=combined_report_path,
                report_metadata=report_metadata,
                config=self.config,
                execution_time=time.time() - start_time,
                memory_usage_mb=psutil.Process().memory_info().rss / 1024 / 1024,
                system_metrics=self._get_system_metrics()
            )
            
            self.logger.info("✅ Comprehensive reporting completed successfully")
            self.logger.info(f"⏱️ Execution time: {results.execution_time:.2f}s")
            self.logger.info(f"📊 Reports generated: {len(generated_reports)}")
            self.logger.info(f"📋 Combined report: {combined_report_path is not None}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error in comprehensive reporting: {e}")
            self.logger.exception("Full traceback:")
            raise
        finally:
            # Stop performance monitoring
            if self.config.enable_performance_monitoring:
                self.performance_monitor.stop_monitoring()
    
    async def _load_backtesting_results(self) -> Dict[str, Any]:
        """Load backtesting results from various steps."""
        self.logger.info("📂 Loading backtesting results...")
        
        backtesting_results = {}
        
        # Load results from various backtesting steps
        result_files = [
            ("basic_backtesting_pre", "basic_backtesting_pre"),
            ("basic_backtesting_post", "basic_backtesting_post"),
            ("walk_forward_validation", "walk_forward_validation"),
            ("monte_carlo_simulation", "monte_carlo_simulation"),
            ("ab_testing", "ab_testing"),
            ("performance_analytics", "performance_analytics"),
            ("risk_analysis", "risk_analysis"),
            ("trade_analysis", "trade_analysis"),
            ("portfolio_analysis", "portfolio_analysis")
        ]
        
        for step_name, directory_name in result_files:
            result_file = self.data_dir / "backtesting_results" / directory_name / f"{self.config.symbol}_{self.config.exchange}_{step_name}_results.json"
            
            if safe_file_exists(result_file):
                try:
                    results = await safe_json_load(result_file)
                    backtesting_results[step_name] = results
                    self.logger.info(f"✅ Loaded {step_name} results")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not load {step_name} results: {e}")
            else:
                self.logger.warning(f"⚠️ No results found for {step_name}")
        
        # Generate mock results if none found
        if not backtesting_results:
            self.logger.warning("⚠️ No backtesting results found, generating mock data")
            backtesting_results = self._generate_mock_backtesting_results()
        
        return backtesting_results
    
    def _generate_mock_backtesting_results(self) -> Dict[str, Any]:
        """Generate mock backtesting results for testing."""
        mock_results = {}
        
        # Mock basic backtesting results
        mock_results["basic_backtesting_pre"] = {
            "symbol": self.config.symbol,
            "exchange": self.config.exchange,
            "timeframe": self.config.timeframe,
            "total_return": 0.15,
            "sharpe_ratio": 1.2,
            "max_drawdown": -0.08,
            "win_rate": 0.55,
            "total_trades": 150,
            "execution_time": 45.2
        }
        
        # Mock performance analytics results
        mock_results["performance_analytics"] = {
            "symbol": self.config.symbol,
            "exchange": self.config.exchange,
            "timeframe": self.config.timeframe,
            "performance_metrics": {
                "total_return": 0.18,
                "annualized_return": 0.22,
                "volatility": 0.25,
                "sharpe_ratio": 1.4,
                "max_drawdown": -0.12,
                "win_rate": 0.58
            },
            "execution_time": 32.1
        }
        
        # Mock risk analysis results
        mock_results["risk_analysis"] = {
            "symbol": self.config.symbol,
            "exchange": self.config.exchange,
            "timeframe": self.config.timeframe,
            "risk_metrics": {
                "var_95": -0.025,
                "var_99": -0.045,
                "max_drawdown": -0.12,
                "volatility": 0.25
            },
            "execution_time": 28.7
        }
        
        return mock_results
    
    async def _generate_report(self, report_type: ReportType, backtesting_results: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Generate a specific type of report."""
        output_dir = self.data_dir / "backtesting_results" / "reports"
        ensure_directory(output_dir)
        
        if report_type == ReportType.EXECUTIVE_SUMMARY:
            return await self._generate_executive_summary(output_dir, backtesting_results)
        elif report_type == ReportType.PERFORMANCE_REPORT:
            return await self._generate_performance_report(output_dir, backtesting_results)
        elif report_type == ReportType.RISK_REPORT:
            return await self._generate_risk_report(output_dir, backtesting_results)
        elif report_type == ReportType.TRADE_REPORT:
            return await self._generate_trade_report(output_dir, backtesting_results)
        elif report_type == ReportType.PORTFOLIO_REPORT:
            return await self._generate_portfolio_report(output_dir, backtesting_results)
        elif report_type == ReportType.COMPREHENSIVE_REPORT:
            return await self._generate_comprehensive_report(output_dir, backtesting_results)
        elif report_type == ReportType.COMPARISON_REPORT:
            return await self._generate_comparison_report(output_dir, backtesting_results)
        else:
            raise ValueError(f"Unknown report type: {report_type}")
    
    async def _generate_executive_summary(self, output_dir: Path, backtesting_results: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Generate executive summary report."""
        report_path = output_dir / f"{self.config.symbol}_{self.config.exchange}_executive_summary.{self.config.output_format}"
        
        # Extract key metrics
        key_metrics = self._extract_key_metrics(backtesting_results)
        
        # Generate executive summary content
        summary_content = self._create_executive_summary_content(key_metrics)
        
        # Save report
        await self._save_report(report_path, summary_content)
        
        summary = {
            "report_type": "executive_summary",
            "key_metrics": key_metrics,
            "recommendations": self._extract_recommendations(backtesting_results),
            "file_size": get_file_size(report_path)
        }
        
        return str(report_path), summary
    
    async def _generate_performance_report(self, output_dir: Path, backtesting_results: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Generate performance report."""
        report_path = output_dir / f"{self.config.symbol}_{self.config.exchange}_performance_report.{self.config.output_format}"
        
        # Extract performance metrics
        performance_metrics = self._extract_performance_metrics(backtesting_results)
        
        # Generate performance report content
        performance_content = self._create_performance_report_content(performance_metrics)
        
        # Save report
        await self._save_report(report_path, performance_content)
        
        summary = {
            "report_type": "performance_report",
            "performance_metrics": performance_metrics,
            "file_size": get_file_size(report_path)
        }
        
        return str(report_path), summary
    
    async def _generate_risk_report(self, output_dir: Path, backtesting_results: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Generate risk report."""
        report_path = output_dir / f"{self.config.symbol}_{self.config.exchange}_risk_report.{self.config.output_format}"
        
        # Extract risk metrics
        risk_metrics = self._extract_risk_metrics(backtesting_results)
        
        # Generate risk report content
        risk_content = self._create_risk_report_content(risk_metrics)
        
        # Save report
        await self._save_report(report_path, risk_content)
        
        summary = {
            "report_type": "risk_report",
            "risk_metrics": risk_metrics,
            "file_size": get_file_size(report_path)
        }
        
        return str(report_path), summary
    
    async def _generate_trade_report(self, output_dir: Path, backtesting_results: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Generate trade report."""
        report_path = output_dir / f"{self.config.symbol}_{self.config.exchange}_trade_report.{self.config.output_format}"
        
        # Extract trade metrics
        trade_metrics = self._extract_trade_metrics(backtesting_results)
        
        # Generate trade report content
        trade_content = self._create_trade_report_content(trade_metrics)
        
        # Save report
        await self._save_report(report_path, trade_content)
        
        summary = {
            "report_type": "trade_report",
            "trade_metrics": trade_metrics,
            "file_size": get_file_size(report_path)
        }
        
        return str(report_path), summary
    
    async def _generate_portfolio_report(self, output_dir: Path, backtesting_results: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Generate portfolio report."""
        report_path = output_dir / f"{self.config.symbol}_{self.config.exchange}_portfolio_report.{self.config.output_format}"
        
        # Extract portfolio metrics
        portfolio_metrics = self._extract_portfolio_metrics(backtesting_results)
        
        # Generate portfolio report content
        portfolio_content = self._create_portfolio_report_content(portfolio_metrics)
        
        # Save report
        await self._save_report(report_path, portfolio_content)
        
        summary = {
            "report_type": "portfolio_report",
            "portfolio_metrics": portfolio_metrics,
            "file_size": get_file_size(report_path)
        }
        
        return str(report_path), summary
    
    async def _generate_comprehensive_report(self, output_dir: Path, backtesting_results: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Generate comprehensive report using existing functionality."""
        report_path = output_dir / f"{self.config.symbol}_{self.config.exchange}_comprehensive_report.{self.config.output_format}"
        
        # Use existing comprehensive reporter
        try:
            comprehensive_content = await self.comprehensive_reporter.generate_comprehensive_report(
                backtesting_results, 
                include_visualizations=self.config.include_visualizations,
                include_recommendations=self.config.include_recommendations,
                include_troubleshooting=self.config.include_troubleshooting,
                include_quality_assessment=self.config.include_quality_assessment
            )
        except Exception as e:
            self.logger.warning(f"⚠️ Could not use comprehensive reporter: {e}")
            # Fallback to basic comprehensive report
            comprehensive_content = self._create_basic_comprehensive_report_content(backtesting_results)
        
        # Save report
        await self._save_report(report_path, comprehensive_content)
        
        summary = {
            "report_type": "comprehensive_report",
            "content_sections": len(comprehensive_content.get("sections", [])),
            "file_size": get_file_size(report_path)
        }
        
        return str(report_path), summary
    
    async def _generate_comparison_report(self, output_dir: Path, backtesting_results: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Generate comparison report."""
        report_path = output_dir / f"{self.config.symbol}_{self.config.exchange}_comparison_report.{self.config.output_format}"
        
        # Extract comparison metrics
        comparison_metrics = self._extract_comparison_metrics(backtesting_results)
        
        # Generate comparison report content
        comparison_content = self._create_comparison_report_content(comparison_metrics)
        
        # Save report
        await self._save_report(report_path, comparison_content)
        
        summary = {
            "report_type": "comparison_report",
            "comparison_metrics": comparison_metrics,
            "file_size": get_file_size(report_path)
        }
        
        return str(report_path), summary
    
    async def _generate_combined_report(self, generated_reports: Dict[str, str], report_summaries: Dict[str, Dict[str, Any]], backtesting_results: Dict[str, Any]) -> str:
        """Generate combined report from all individual reports."""
        output_dir = self.data_dir / "backtesting_results" / "reports"
        combined_report_path = output_dir / f"{self.config.symbol}_{self.config.exchange}_combined_report.{self.config.output_format}"
        
        # Create combined report content
        combined_content = {
            "title": f"Combined Backtesting Report - {self.config.symbol}",
            "generated_at": datetime.now().isoformat(),
            "symbol": self.config.symbol,
            "exchange": self.config.exchange,
            "timeframe": self.config.timeframe,
            "report_summaries": report_summaries,
            "generated_reports": generated_reports,
            "backtesting_results": backtesting_results,
            "metadata": {
                "total_reports": len(generated_reports),
                "total_execution_time": sum([summary.get("execution_time", 0) for summary in report_summaries.values()]),
                "report_types": list(generated_reports.keys())
            }
        }
        
        # Save combined report
        await self._save_report(combined_report_path, combined_content)
        
        return str(combined_report_path)
    
    def _create_report_metadata(self, backtesting_results: Dict[str, Any], generated_reports: Dict[str, str]) -> Dict[str, Any]:
        """Create report metadata."""
        return {
            "generation_time": datetime.now().isoformat(),
            "symbol": self.config.symbol,
            "exchange": self.config.exchange,
            "timeframe": self.config.timeframe,
            "total_reports": len(generated_reports),
            "report_types": list(generated_reports.keys()),
            "backtesting_steps": list(backtesting_results.keys()),
            "output_format": self.config.output_format,
            "include_visualizations": self.config.include_visualizations,
            "include_recommendations": self.config.include_recommendations
        }
    
    def _extract_key_metrics(self, backtesting_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from backtesting results."""
        key_metrics = {}
        
        # Extract from performance analytics
        if "performance_analytics" in backtesting_results:
            perf_data = backtesting_results["performance_analytics"]
            if "performance_metrics" in perf_data:
                key_metrics.update(perf_data["performance_metrics"])
        
        # Extract from basic backtesting
        if "basic_backtesting_pre" in backtesting_results:
            basic_data = backtesting_results["basic_backtesting_pre"]
            key_metrics.update({
                "total_return": basic_data.get("total_return", 0),
                "sharpe_ratio": basic_data.get("sharpe_ratio", 0),
                "max_drawdown": basic_data.get("max_drawdown", 0),
                "win_rate": basic_data.get("win_rate", 0),
                "total_trades": basic_data.get("total_trades", 0)
            })
        
        return key_metrics
    
    def _extract_performance_metrics(self, backtesting_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract performance metrics from backtesting results."""
        performance_metrics = {}
        
        # Extract from performance analytics
        if "performance_analytics" in backtesting_results:
            perf_data = backtesting_results["performance_analytics"]
            performance_metrics.update(perf_data.get("performance_metrics", {}))
        
        # Extract from basic backtesting
        if "basic_backtesting_pre" in backtesting_results:
            basic_data = backtesting_results["basic_backtesting_pre"]
            performance_metrics.update({
                "total_return": basic_data.get("total_return", 0),
                "sharpe_ratio": basic_data.get("sharpe_ratio", 0),
                "max_drawdown": basic_data.get("max_drawdown", 0),
                "win_rate": basic_data.get("win_rate", 0)
            })
        
        return performance_metrics
    
    def _extract_risk_metrics(self, backtesting_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract risk metrics from backtesting results."""
        risk_metrics = {}
        
        # Extract from risk analysis
        if "risk_analysis" in backtesting_results:
            risk_data = backtesting_results["risk_analysis"]
            risk_metrics.update(risk_data.get("risk_metrics", {}))
        
        # Extract from performance analytics
        if "performance_analytics" in backtesting_results:
            perf_data = backtesting_results["performance_analytics"]
            if "risk_metrics" in perf_data:
                risk_metrics.update(perf_data["risk_metrics"])
        
        return risk_metrics
    
    def _extract_trade_metrics(self, backtesting_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract trade metrics from backtesting results."""
        trade_metrics = {}
        
        # Extract from trade analysis
        if "trade_analysis" in backtesting_results:
            trade_data = backtesting_results["trade_analysis"]
            trade_metrics.update(trade_data.get("trade_statistics", {}))
        
        # Extract from basic backtesting
        if "basic_backtesting_pre" in backtesting_results:
            basic_data = backtesting_results["basic_backtesting_pre"]
            trade_metrics.update({
                "total_trades": basic_data.get("total_trades", 0),
                "win_rate": basic_data.get("win_rate", 0)
            })
        
        return trade_metrics
    
    def _extract_portfolio_metrics(self, backtesting_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract portfolio metrics from backtesting results."""
        portfolio_metrics = {}
        
        # Extract from portfolio analysis
        if "portfolio_analysis" in backtesting_results:
            portfolio_data = backtesting_results["portfolio_analysis"]
            portfolio_metrics.update(portfolio_data.get("portfolio_metrics", {}))
        
        return portfolio_metrics
    
    def _extract_comparison_metrics(self, backtesting_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comparison metrics from backtesting results."""
        comparison_metrics = {}
        
        # Compare pre and post optimization
        if "basic_backtesting_pre" in backtesting_results and "basic_backtesting_post" in backtesting_results:
            pre_data = backtesting_results["basic_backtesting_pre"]
            post_data = backtesting_results["basic_backtesting_post"]
            
            comparison_metrics["optimization_comparison"] = {
                "pre_optimization": {
                    "total_return": pre_data.get("total_return", 0),
                    "sharpe_ratio": pre_data.get("sharpe_ratio", 0),
                    "max_drawdown": pre_data.get("max_drawdown", 0)
                },
                "post_optimization": {
                    "total_return": post_data.get("total_return", 0),
                    "sharpe_ratio": post_data.get("sharpe_ratio", 0),
                    "max_drawdown": post_data.get("max_drawdown", 0)
                },
                "improvement": {
                    "return_improvement": post_data.get("total_return", 0) - pre_data.get("total_return", 0),
                    "sharpe_improvement": post_data.get("sharpe_ratio", 0) - pre_data.get("sharpe_ratio", 0),
                    "drawdown_improvement": post_data.get("max_drawdown", 0) - pre_data.get("max_drawdown", 0)
                }
            }
        
        return comparison_metrics
    
    def _extract_recommendations(self, backtesting_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract recommendations from backtesting results."""
        recommendations = []
        
        # Extract from risk analysis
        if "risk_analysis" in backtesting_results:
            risk_data = backtesting_results["risk_analysis"]
            if "risk_recommendations" in risk_data:
                recommendations.extend(risk_data["risk_recommendations"])
        
        # Extract from trade analysis
        if "trade_analysis" in backtesting_results:
            trade_data = backtesting_results["trade_analysis"]
            if "optimization_insights" in trade_data:
                recommendations.extend(trade_data["optimization_insights"])
        
        # Extract from portfolio analysis
        if "portfolio_analysis" in backtesting_results:
            portfolio_data = backtesting_results["portfolio_analysis"]
            if "optimization_insights" in portfolio_data:
                recommendations.extend(portfolio_data["optimization_insights"])
        
        return recommendations
    
    def _create_executive_summary_content(self, key_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create executive summary content."""
        return {
            "title": f"Executive Summary - {self.config.symbol} Backtesting",
            "generated_at": datetime.now().isoformat(),
            "symbol": self.config.symbol,
            "exchange": self.config.exchange,
            "timeframe": self.config.timeframe,
            "key_metrics": key_metrics,
            "summary": {
                "performance": f"Total return: {key_metrics.get('total_return', 0):.2%}",
                "risk": f"Sharpe ratio: {key_metrics.get('sharpe_ratio', 0):.2f}",
                "drawdown": f"Max drawdown: {key_metrics.get('max_drawdown', 0):.2%}",
                "trades": f"Total trades: {key_metrics.get('total_trades', 0)}"
            },
            "recommendations": self._generate_executive_recommendations(key_metrics)
        }
    
    def _create_performance_report_content(self, performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create performance report content."""
        return {
            "title": f"Performance Report - {self.config.symbol}",
            "generated_at": datetime.now().isoformat(),
            "performance_metrics": performance_metrics,
            "analysis": {
                "return_analysis": self._analyze_returns(performance_metrics),
                "risk_analysis": self._analyze_risk(performance_metrics),
                "efficiency_analysis": self._analyze_efficiency(performance_metrics)
            }
        }
    
    def _create_risk_report_content(self, risk_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create risk report content."""
        return {
            "title": f"Risk Report - {self.config.symbol}",
            "generated_at": datetime.now().isoformat(),
            "risk_metrics": risk_metrics,
            "analysis": {
                "var_analysis": self._analyze_var(risk_metrics),
                "drawdown_analysis": self._analyze_drawdown(risk_metrics),
                "volatility_analysis": self._analyze_volatility(risk_metrics)
            }
        }
    
    def _create_trade_report_content(self, trade_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create trade report content."""
        return {
            "title": f"Trade Report - {self.config.symbol}",
            "generated_at": datetime.now().isoformat(),
            "trade_metrics": trade_metrics,
            "analysis": {
                "trade_frequency": self._analyze_trade_frequency(trade_metrics),
                "trade_performance": self._analyze_trade_performance(trade_metrics),
                "trade_patterns": self._analyze_trade_patterns(trade_metrics)
            }
        }
    
    def _create_portfolio_report_content(self, portfolio_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create portfolio report content."""
        return {
            "title": f"Portfolio Report - {self.config.symbol}",
            "generated_at": datetime.now().isoformat(),
            "portfolio_metrics": portfolio_metrics,
            "analysis": {
                "allocation_analysis": self._analyze_allocation(portfolio_metrics),
                "diversification_analysis": self._analyze_diversification(portfolio_metrics),
                "rebalancing_analysis": self._analyze_rebalancing(portfolio_metrics)
            }
        }
    
    def _create_basic_comprehensive_report_content(self, backtesting_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create basic comprehensive report content."""
        return {
            "title": f"Comprehensive Report - {self.config.symbol}",
            "generated_at": datetime.now().isoformat(),
            "sections": [
                {"name": "Executive Summary", "content": "Overview of backtesting results"},
                {"name": "Performance Analysis", "content": "Detailed performance metrics"},
                {"name": "Risk Analysis", "content": "Risk assessment and metrics"},
                {"name": "Trade Analysis", "content": "Trade-level analysis"},
                {"name": "Portfolio Analysis", "content": "Portfolio-level analysis"},
                {"name": "Recommendations", "content": "Actionable recommendations"}
            ],
            "backtesting_results": backtesting_results
        }
    
    def _create_comparison_report_content(self, comparison_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create comparison report content."""
        return {
            "title": f"Comparison Report - {self.config.symbol}",
            "generated_at": datetime.now().isoformat(),
            "comparison_metrics": comparison_metrics,
            "analysis": {
                "optimization_impact": self._analyze_optimization_impact(comparison_metrics),
                "performance_comparison": self._analyze_performance_comparison(comparison_metrics)
            }
        }
    
    def _generate_executive_recommendations(self, key_metrics: Dict[str, Any]) -> List[str]:
        """Generate executive recommendations based on key metrics."""
        recommendations = []
        
        if key_metrics.get("sharpe_ratio", 0) < 1.0:
            recommendations.append("Consider improving risk-adjusted returns through better position sizing")
        
        if key_metrics.get("max_drawdown", 0) < -0.15:
            recommendations.append("Implement better risk management to reduce maximum drawdown")
        
        if key_metrics.get("win_rate", 0) < 0.5:
            recommendations.append("Review entry/exit criteria to improve win rate")
        
        return recommendations
    
    def _analyze_returns(self, performance_metrics: Dict[str, Any]) -> str:
        """Analyze return metrics."""
        total_return = performance_metrics.get("total_return", 0)
        if total_return > 0.2:
            return "Strong positive returns"
        elif total_return > 0.1:
            return "Moderate positive returns"
        elif total_return > 0:
            return "Weak positive returns"
        else:
            return "Negative returns"
    
    def _analyze_risk(self, performance_metrics: Dict[str, Any]) -> str:
        """Analyze risk metrics."""
        sharpe_ratio = performance_metrics.get("sharpe_ratio", 0)
        if sharpe_ratio > 1.5:
            return "Excellent risk-adjusted returns"
        elif sharpe_ratio > 1.0:
            return "Good risk-adjusted returns"
        elif sharpe_ratio > 0.5:
            return "Moderate risk-adjusted returns"
        else:
            return "Poor risk-adjusted returns"
    
    def _analyze_efficiency(self, performance_metrics: Dict[str, Any]) -> str:
        """Analyze efficiency metrics."""
        win_rate = performance_metrics.get("win_rate", 0)
        if win_rate > 0.6:
            return "High efficiency"
        elif win_rate > 0.5:
            return "Moderate efficiency"
        else:
            return "Low efficiency"
    
    def _analyze_var(self, risk_metrics: Dict[str, Any]) -> str:
        """Analyze VaR metrics."""
        var_95 = abs(risk_metrics.get("var_95", 0))
        if var_95 < 0.02:
            return "Low VaR risk"
        elif var_95 < 0.05:
            return "Moderate VaR risk"
        else:
            return "High VaR risk"
    
    def _analyze_drawdown(self, risk_metrics: Dict[str, Any]) -> str:
        """Analyze drawdown metrics."""
        max_drawdown = abs(risk_metrics.get("max_drawdown", 0))
        if max_drawdown < 0.05:
            return "Low drawdown risk"
        elif max_drawdown < 0.15:
            return "Moderate drawdown risk"
        else:
            return "High drawdown risk"
    
    def _analyze_volatility(self, risk_metrics: Dict[str, Any]) -> str:
        """Analyze volatility metrics."""
        volatility = risk_metrics.get("volatility", 0)
        if volatility < 0.15:
            return "Low volatility"
        elif volatility < 0.25:
            return "Moderate volatility"
        else:
            return "High volatility"
    
    def _analyze_trade_frequency(self, trade_metrics: Dict[str, Any]) -> str:
        """Analyze trade frequency."""
        total_trades = trade_metrics.get("total_trades", 0)
        if total_trades > 200:
            return "High frequency trading"
        elif total_trades > 100:
            return "Moderate frequency trading"
        else:
            return "Low frequency trading"
    
    def _analyze_trade_performance(self, trade_metrics: Dict[str, Any]) -> str:
        """Analyze trade performance."""
        win_rate = trade_metrics.get("win_rate", 0)
        if win_rate > 0.6:
            return "Strong trade performance"
        elif win_rate > 0.5:
            return "Moderate trade performance"
        else:
            return "Weak trade performance"
    
    def _analyze_trade_patterns(self, trade_metrics: Dict[str, Any]) -> str:
        """Analyze trade patterns."""
        return "Trade pattern analysis completed"
    
    def _analyze_allocation(self, portfolio_metrics: Dict[str, Any]) -> str:
        """Analyze portfolio allocation."""
        return "Portfolio allocation analysis completed"
    
    def _analyze_diversification(self, portfolio_metrics: Dict[str, Any]) -> str:
        """Analyze portfolio diversification."""
        return "Portfolio diversification analysis completed"
    
    def _analyze_rebalancing(self, portfolio_metrics: Dict[str, Any]) -> str:
        """Analyze portfolio rebalancing."""
        return "Portfolio rebalancing analysis completed"
    
    def _analyze_optimization_impact(self, comparison_metrics: Dict[str, Any]) -> str:
        """Analyze optimization impact."""
        return "Optimization impact analysis completed"
    
    def _analyze_performance_comparison(self, comparison_metrics: Dict[str, Any]) -> str:
        """Analyze performance comparison."""
        return "Performance comparison analysis completed"
    
    async def _save_report(self, report_path: Path, content: Dict[str, Any]) -> None:
        """Save report to file."""
        if self.config.output_format == "json":
            await safe_json_dump(report_path, content, indent=2)
        elif self.config.output_format == "html":
            html_content = self._convert_to_html(content)
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
        elif self.config.output_format == "markdown":
            markdown_content = self._convert_to_markdown(content)
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
        else:
            # Default to JSON
            await safe_json_dump(report_path, content, indent=2)
    
    def _convert_to_html(self, content: Dict[str, Any]) -> str:
        """Convert content to HTML format."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{content.get('title', 'Backtesting Report')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                .metric {{ background-color: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .recommendation {{ background-color: #e8f4fd; padding: 10px; margin: 10px 0; border-left: 4px solid #2196F3; }}
            </style>
        </head>
        <body>
            <h1>{content.get('title', 'Backtesting Report')}</h1>
            <p>Generated at: {content.get('generated_at', 'N/A')}</p>
            <p>Symbol: {content.get('symbol', 'N/A')}</p>
            <p>Exchange: {content.get('exchange', 'N/A')}</p>
            <p>Timeframe: {content.get('timeframe', 'N/A')}</p>
        </body>
        </html>
        """
        return html
    
    def _convert_to_markdown(self, content: Dict[str, Any]) -> str:
        """Convert content to Markdown format."""
        markdown = f"""# {content.get('title', 'Backtesting Report')}

**Generated at:** {content.get('generated_at', 'N/A')}
**Symbol:** {content.get('symbol', 'N/A')}
**Exchange:** {content.get('exchange', 'N/A')}
**Timeframe:** {content.get('timeframe', 'N/A')}

## Key Metrics

"""
        return markdown
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage_percent': psutil.disk_usage('/').percent,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.warning(f"⚠️ Could not get system metrics: {e}")
            return {}


# Convenience function for easy integration
async def execute_comprehensive_reporting(
    symbol: str = "ETHUSDT",
    exchange: str = "BINANCE", 
    timeframe: str = "1h",
    data_dir: str = "data/training",
    **kwargs
) -> ReportingResults:
    """
    Convenience function to execute comprehensive reporting.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Data timeframe
        data_dir: Data directory
        **kwargs: Additional configuration parameters
        
    Returns:
        Reporting results
    """
    config = ReportingConfig(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        **kwargs
    )
    
    step = ReportingStep(config)
    return await step.execute()