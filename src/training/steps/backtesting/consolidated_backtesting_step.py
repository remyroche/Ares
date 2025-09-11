"""
Consolidated Backtesting Step with Comprehensive Analytics

This module provides a single, comprehensive backtesting step that combines:
1. Walk-forward backtesting with M1 optimizations
2. Monte Carlo simulations with GPU acceleration
3. A/B testing with statistical validation
4. Model saving with comprehensive metadata
5. Detailed analytics and reporting

All functionality from steps 18-21 is consolidated into this single step.
"""

import asyncio
import logging

# Optional imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import gc

# Optional imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None
from pathlib import Path

# Import consolidated backtesting utilities
from src.utils.common_ml.backtesting import (
    BacktestingEngine, BacktestingConfig, BacktestingResults,
    MonteCarloEngine, MonteCarloConfig, MonteCarloResults,
    ABTestingEngine, ABTestConfig, ABTestResults,
    ModelSaver, ModelSaveConfig, ModelMetadata,
    AnalyticsReporter, AnalyticsConfig, PerformanceMetrics, RiskMetrics
)

# M1 Optimization imports
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer

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
from src.core.decorators import (
    handles_errors, validates, traced, log_execution_time, 
    timeout, error_boundary, compose, validate_data_quality, 
    monitor_step_execution, ensure_data_integrity, validate_pipeline_step
)
from src.core.errors import (
    ValidationError, DataIntegrityError, FileOperationError,
    MathValidationError, TimeoutError
)

logger = logging.getLogger(__name__)


class BacktestingMode(Enum):
    """Backtesting execution modes."""
    BACKTESTING_ONLY = "backtesting_only"
    MONTE_CARLO_ONLY = "monte_carlo_only"
    AB_TESTING_ONLY = "ab_testing_only"
    MODEL_SAVING_ONLY = "model_saving_only"
    COMPREHENSIVE = "comprehensive"  # All components


@dataclass
class ConsolidatedBacktestingConfig:
    """Configuration for consolidated backtesting step."""
    # Basic configuration
    symbol: str
    exchange: str
    timeframe: str
    data_dir: str
    output_dir: str
    
    # Execution mode
    mode: BacktestingMode = BacktestingMode.COMPREHENSIVE
    
    # Backtesting configuration
    backtesting_config: Optional[BacktestingConfig] = None
    
    # Monte Carlo configuration
    monte_carlo_config: Optional[MonteCarloConfig] = None
    
    # A/B testing configuration
    ab_test_config: Optional[ABTestConfig] = None
    
    # Model saving configuration
    model_save_config: Optional[ModelSaveConfig] = None
    
    # Analytics configuration
    analytics_config: Optional[AnalyticsConfig] = None
    
    # M1 optimization settings
    enable_gpu_acceleration: bool = True
    enable_memory_optimization: bool = True
    enable_parallel_processing: bool = True
    memory_limit_gb: float = 8.0
    max_workers: Optional[int] = None
    
    # Performance settings
    enable_caching: bool = True
    cache_size_mb: int = 100
    enable_profiling: bool = False
    
    # Validation settings
    validate_inputs: bool = True
    min_data_points: int = 100
    max_execution_time: int = 3600  # 1 hour
    
    # Output settings
    save_detailed_results: bool = True
    generate_reports: bool = True
    cleanup_temp_files: bool = True


@dataclass
class ConsolidatedBacktestingResults:
    """Results from consolidated backtesting step."""
    # Basic info
    symbol: str
    exchange: str
    timeframe: str
    mode: BacktestingMode
    start_time: datetime
    end_time: datetime
    total_duration: float
    
    # Component results
    backtesting_results: Optional[BacktestingResults] = None
    monte_carlo_results: Optional[MonteCarloResults] = None
    ab_test_results: Optional[ABTestResults] = None
    model_metadata: Optional[ModelMetadata] = None
    analytics_report: Optional[Dict[str, Any]] = None
    
    # Overall metrics
    overall_performance: str = "Unknown"
    risk_assessment: str = "Unknown"
    recommendation: str = "No recommendation available"
    
    # Execution info
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    optimization_used: List[str] = field(default_factory=list)
    
    # File paths
    output_files: List[str] = field(default_factory=list)
    
    # Status
    status: str = "Unknown"
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ConsolidatedBacktestingStep:
    """Consolidated backtesting step with comprehensive functionality."""
    
    def __init__(self, config: ConsolidatedBacktestingConfig):
        """Initialize consolidated backtesting step."""
        self.config = config
        self.logger = logger.getChild('ConsolidatedBacktestingStep')
        
        # Initialize M1 optimizers
        self.m1_gpu = get_m1_gpu_manager() if config.enable_gpu_acceleration else None
        self.m1_memory = get_m1_memory_optimizer(
            memory_limit_gb=config.memory_limit_gb
        ) if config.enable_memory_optimization else None
        self.m1_cpu = get_m1_cpu_optimizer(
            max_workers=config.max_workers
        ) if config.enable_parallel_processing else None
        
        # Initialize utilities
        self.parquet_utils = get_parquet_utils()
        
        # Initialize component engines
        self._initialize_engines()
        
        # Ensure output directory exists
        ensure_directory(config.output_dir)
        
        self.logger.info(f"🚀 ConsolidatedBacktestingStep initialized for {config.symbol}")
        self.logger.info(f"⚡ GPU acceleration: {config.enable_gpu_acceleration}")
        self.logger.info(f"🧠 Memory optimization: {config.enable_memory_optimization}")
        self.logger.info(f"🔄 Parallel processing: {config.enable_parallel_processing}")
        self.logger.info(f"🎯 Execution mode: {config.mode.value}")
    
    def _initialize_engines(self):
        """Initialize component engines with default configurations."""
        
        # Initialize backtesting engine
        if self.config.backtesting_config is None:
            self.config.backtesting_config = BacktestingConfig(
                symbol=self.config.symbol,
                exchange=self.config.exchange,
                timeframe=self.config.timeframe,
                data_dir=self.config.data_dir,
                enable_gpu_acceleration=self.config.enable_gpu_acceleration,
                enable_memory_optimization=self.config.enable_memory_optimization,
                enable_parallel_processing=self.config.enable_parallel_processing,
                memory_limit_gb=self.config.memory_limit_gb,
                max_workers=self.config.max_workers
            )
        
        self.backtesting_engine = BacktestingEngine(self.config.backtesting_config)
        
        # Initialize Monte Carlo engine
        if self.config.monte_carlo_config is None:
            self.config.monte_carlo_config = MonteCarloConfig(
                symbol=self.config.symbol,
                exchange=self.config.exchange,
                timeframe=self.config.timeframe,
                data_dir=self.config.data_dir,
                enable_gpu_acceleration=self.config.enable_gpu_acceleration,
                enable_memory_optimization=self.config.enable_memory_optimization,
                enable_parallel_processing=self.config.enable_parallel_processing,
                memory_limit_gb=self.config.memory_limit_gb,
                max_workers=self.config.max_workers
            )
        
        self.monte_carlo_engine = MonteCarloEngine(self.config.monte_carlo_config)
        
        # Initialize A/B testing engine
        if self.config.ab_test_config is None:
            self.config.ab_test_config = ABTestConfig(
                symbol=self.config.symbol,
                exchange=self.config.exchange,
                timeframe=self.config.timeframe,
                data_dir=self.config.data_dir,
                test_name=f"{self.config.symbol}_ab_test",
                test_description=f"A/B test for {self.config.symbol}",
                enable_gpu_acceleration=self.config.enable_gpu_acceleration,
                enable_memory_optimization=self.config.enable_memory_optimization,
                enable_parallel_processing=self.config.enable_parallel_processing,
                memory_limit_gb=self.config.memory_limit_gb,
                max_workers=self.config.max_workers
            )
        
        self.ab_testing_engine = ABTestingEngine(self.config.ab_test_config)
        
        # Initialize model saver
        if self.config.model_save_config is None:
            self.config.model_save_config = ModelSaveConfig(
                model_name=f"{self.config.symbol}_model",
                output_dir=f"{self.config.output_dir}/models",
                enable_memory_optimization=self.config.enable_memory_optimization,
                memory_limit_gb=self.config.memory_limit_gb,
                max_workers=self.config.max_workers
            )
        
        self.model_saver = ModelSaver(self.config.model_save_config)
        
        # Initialize analytics reporter
        if self.config.analytics_config is None:
            self.config.analytics_config = AnalyticsConfig(
                symbol=self.config.symbol,
                exchange=self.config.exchange,
                timeframe=self.config.timeframe,
                output_dir=f"{self.config.output_dir}/analytics",
                enable_gpu_acceleration=self.config.enable_gpu_acceleration,
                enable_memory_optimization=self.config.enable_memory_optimization,
                enable_parallel_processing=self.config.enable_parallel_processing,
                memory_limit_gb=self.config.memory_limit_gb,
                max_workers=self.config.max_workers
            )
        
        self.analytics_reporter = AnalyticsReporter(self.config.analytics_config)
    
    @traced(span_name='execute_consolidated_backtesting')
    @log_execution_time
    async def execute(
        self, 
        data: pd.DataFrame,
        model: Optional[Any] = None,
        strategy_func: Optional[Callable] = None,
        control_data: Optional[pd.DataFrame] = None,
        treatment_data: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> ConsolidatedBacktestingResults:
        """Execute consolidated backtesting with all components."""
        
        self.logger.info("🚀 Starting consolidated backtesting execution...")
        start_time = time.time()
        
        # Validate inputs
        if self.config.validate_inputs:
            self._validate_inputs(data, model, strategy_func, control_data, treatment_data)
        
        # Memory optimization context
        if self.m1_memory:
            with self.m1_memory.optimization_context():
                results = await self._execute_consolidated_backtesting(
                    data, model, strategy_func, control_data, treatment_data, **kwargs
                )
        else:
            results = await self._execute_consolidated_backtesting(
                data, model, strategy_func, control_data, treatment_data, **kwargs
            )
        
        execution_time = time.time() - start_time
        results.execution_time = execution_time
        
        # Log memory usage
        if self.m1_memory:
            results.memory_usage_mb = self.m1_memory.get_current_memory_usage_mb()
        
        self.logger.info(f"✅ Consolidated backtesting completed in {execution_time:.2f}s")
        self.logger.info(f"📊 Overall performance: {results.overall_performance}")
        self.logger.info(f"⚠️ Risk assessment: {results.risk_assessment}")
        self.logger.info(f"💡 Recommendation: {results.recommendation}")
        
        return results
    
    def _validate_inputs(
        self, 
        data: pd.DataFrame, 
        model: Optional[Any], 
        strategy_func: Optional[Callable],
        control_data: Optional[pd.DataFrame],
        treatment_data: Optional[pd.DataFrame]
    ) -> None:
        """Validate input data and parameters."""
        
        if data.empty:
            raise ValidationError("Input data is empty")
        
        if len(data) < self.config.min_data_points:
            raise ValidationError(f"Insufficient data points: {len(data)} < {self.config.min_data_points}")
        
        # Validate required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValidationError(f"Missing required columns: {missing_columns}")
        
        # Validate mode-specific requirements
        if self.config.mode in [BacktestingMode.BACKTESTING_ONLY, BacktestingMode.COMPREHENSIVE]:
            if strategy_func is None:
                raise ValidationError("Strategy function is required for backtesting")
        
        if self.config.mode in [BacktestingMode.AB_TESTING_ONLY, BacktestingMode.COMPREHENSIVE]:
            if control_data is None or treatment_data is None:
                self.logger.warning("Control and treatment data not provided for A/B testing")
        
        if self.config.mode in [BacktestingMode.MODEL_SAVING_ONLY, BacktestingMode.COMPREHENSIVE]:
            if model is None:
                self.logger.warning("Model not provided for saving")
    
    async def _execute_consolidated_backtesting(
        self, 
        data: pd.DataFrame,
        model: Optional[Any],
        strategy_func: Optional[Callable],
        control_data: Optional[pd.DataFrame],
        treatment_data: Optional[pd.DataFrame],
        **kwargs
    ) -> ConsolidatedBacktestingResults:
        """Execute the actual consolidated backtesting logic."""
        
        results = ConsolidatedBacktestingResults(
            symbol=self.config.symbol,
            exchange=self.config.exchange,
            timeframe=self.config.timeframe,
            mode=self.config.mode,
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_duration=0.0,
            optimization_used=self._get_optimization_used()
        )
        
        try:
            # Execute backtesting
            if self.config.mode in [BacktestingMode.BACKTESTING_ONLY, BacktestingMode.COMPREHENSIVE]:
                if strategy_func is not None:
                    self.logger.info("🔄 Executing backtesting...")
                    results.backtesting_results = await self.backtesting_engine.execute(
                        data, strategy_func, **kwargs
                    )
                    self.logger.info("✅ Backtesting completed")
                else:
                    results.warnings.append("Strategy function not provided, skipping backtesting")
            
            # Execute Monte Carlo simulation
            if self.config.mode in [BacktestingMode.MONTE_CARLO_ONLY, BacktestingMode.COMPREHENSIVE]:
                self.logger.info("🔄 Executing Monte Carlo simulation...")
                results.monte_carlo_results = await self.monte_carlo_engine.simulate(
                    data, **kwargs
                )
                self.logger.info("✅ Monte Carlo simulation completed")
            
            # Execute A/B testing
            if self.config.mode in [BacktestingMode.AB_TESTING_ONLY, BacktestingMode.COMPREHENSIVE]:
                if control_data is not None and treatment_data is not None:
                    self.logger.info("🔄 Executing A/B testing...")
                    # Extract metric columns (assuming returns or similar metrics)
                    metric_columns = ['close']  # Default metric
                    if 'returns' in control_data.columns:
                        metric_columns = ['returns']
                    
                    results.ab_test_results = await self.ab_testing_engine.execute(
                        control_data, treatment_data, metric_columns, **kwargs
                    )
                    self.logger.info("✅ A/B testing completed")
                else:
                    results.warnings.append("Control and treatment data not provided, skipping A/B testing")
            
            # Save model
            if self.config.mode in [BacktestingMode.MODEL_SAVING_ONLY, BacktestingMode.COMPREHENSIVE]:
                if model is not None:
                    self.logger.info("🔄 Saving model...")
                    
                    # Create metadata
                    metadata = ModelMetadata(
                        model_name=self.config.model_save_config.model_name,
                        model_type=self.model_saver._detect_model_type(model),
                        version="1.0.0",
                        created_at=datetime.now(),
                        model_class=model.__class__.__name__
                    )
                    
                    # Add performance metrics if available
                    if results.backtesting_results:
                        metadata.performance_metrics = {
                            'total_return': results.backtesting_results.total_return,
                            'sharpe_ratio': results.backtesting_results.sharpe_ratio,
                            'max_drawdown': results.backtesting_results.max_drawdown,
                            'win_rate': results.backtesting_results.win_rate
                        }
                    
                    results.model_metadata = await self.model_saver.save_model(
                        model, metadata, **kwargs
                    )
                    self.logger.info("✅ Model saved")
                else:
                    results.warnings.append("Model not provided, skipping model saving")
            
            # Generate analytics report
            if self.config.generate_reports:
                self.logger.info("🔄 Generating analytics report...")
                results.analytics_report = await self.analytics_reporter.generate_report(
                    backtesting_results=results.backtesting_results,
                    monte_carlo_results=results.monte_carlo_results,
                    ab_test_results=results.ab_test_results,
                    **kwargs
                )
                self.logger.info("✅ Analytics report generated")
            
            # Determine overall assessment
            results.overall_performance = self._assess_overall_performance(results)
            results.risk_assessment = self._assess_risk_level(results)
            results.recommendation = self._generate_recommendation(results)
            
            # Save results
            if self.config.save_detailed_results:
                await self._save_results(results)
            
            results.status = "SUCCESS"
            
        except Exception as e:
            self.logger.error(f"❌ Error in consolidated backtesting: {e}")
            results.status = "FAILED"
            results.errors.append(str(e))
            raise
        
        finally:
            results.end_time = datetime.now()
            results.total_duration = (results.end_time - results.start_time).total_seconds()
        
        return results
    
    def _assess_overall_performance(self, results: ConsolidatedBacktestingResults) -> str:
        """Assess overall performance based on results."""
        
        if results.backtesting_results:
            sharpe_ratio = results.backtesting_results.sharpe_ratio
            if sharpe_ratio > 1.0:
                return "Excellent"
            elif sharpe_ratio > 0.5:
                return "Good"
            elif sharpe_ratio > 0.0:
                return "Fair"
            else:
                return "Poor"
        
        if results.monte_carlo_results:
            mean_return = results.monte_carlo_results.mean_return
            if mean_return > 0.1:
                return "Excellent"
            elif mean_return > 0.05:
                return "Good"
            elif mean_return > 0.0:
                return "Fair"
            else:
                return "Poor"
        
        return "Unknown"
    
    def _assess_risk_level(self, results: ConsolidatedBacktestingResults) -> str:
        """Assess risk level based on results."""
        
        if results.backtesting_results:
            max_drawdown = abs(results.backtesting_results.max_drawdown)
            if max_drawdown > 0.2:
                return "High"
            elif max_drawdown > 0.1:
                return "Medium"
            else:
                return "Low"
        
        if results.monte_carlo_results:
            var_95 = abs(results.monte_carlo_results.var_95)
            if var_95 > 0.1:
                return "High"
            elif var_95 > 0.05:
                return "Medium"
            else:
                return "Low"
        
        return "Unknown"
    
    def _generate_recommendation(self, results: ConsolidatedBacktestingResults) -> str:
        """Generate actionable recommendation based on results."""
        
        recommendations = []
        
        if results.backtesting_results:
            if results.backtesting_results.sharpe_ratio > 1.0:
                recommendations.append("Strategy shows excellent performance - consider implementation")
            elif results.backtesting_results.sharpe_ratio > 0.5:
                recommendations.append("Strategy shows good performance - monitor closely")
            elif results.backtesting_results.sharpe_ratio > 0.0:
                recommendations.append("Strategy shows fair performance - consider optimization")
            else:
                recommendations.append("Strategy shows poor performance - requires revision")
        
        if results.ab_test_results:
            if "Significant difference" in results.ab_test_results.overall_conclusion:
                recommendations.append(results.ab_test_results.recommendation)
        
        if results.monte_carlo_results:
            if results.monte_carlo_results.convergence_achieved:
                recommendations.append("Monte Carlo simulation converged - results are reliable")
            else:
                recommendations.append("Monte Carlo simulation did not converge - increase iterations")
        
        if not recommendations:
            return "No specific recommendations available"
        
        return "; ".join(recommendations)
    
    def _get_optimization_used(self) -> List[str]:
        """Get list of optimizations used."""
        optimizations = []
        
        if self.config.enable_gpu_acceleration and self.m1_gpu:
            optimizations.append("m1_gpu_acceleration")
        
        if self.config.enable_memory_optimization and self.m1_memory:
            optimizations.append("m1_memory_optimization")
        
        if self.config.enable_parallel_processing and self.m1_cpu:
            optimizations.append("m1_parallel_processing")
        
        return optimizations
    
    async def _save_results(self, results: ConsolidatedBacktestingResults) -> None:
        """Save comprehensive results to disk."""
        
        # Save main results
        results_file = f"{self.config.output_dir}/{self.config.symbol}_consolidated_backtesting_results.json"
        await safe_json_dump(results_file, results.__dict__)
        results.output_files.append(results_file)
        self.logger.info(f"💾 Results saved to {results_file}")
        
        # Save individual component results
        if results.backtesting_results:
            backtesting_file = f"{self.config.output_dir}/{self.config.symbol}_backtesting_results.json"
            await safe_json_dump(backtesting_file, results.backtesting_results.__dict__)
            results.output_files.append(backtesting_file)
        
        if results.monte_carlo_results:
            mc_file = f"{self.config.output_dir}/{self.config.symbol}_monte_carlo_results.json"
            await safe_json_dump(mc_file, results.monte_carlo_results.__dict__)
            results.output_files.append(mc_file)
        
        if results.ab_test_results:
            ab_file = f"{self.config.output_dir}/{self.config.symbol}_ab_test_results.json"
            await safe_json_dump(ab_file, results.ab_test_results.__dict__)
            results.output_files.append(ab_file)
        
        if results.analytics_report:
            analytics_file = f"{self.config.output_dir}/{self.config.symbol}_analytics_report.json"
            await safe_json_dump(analytics_file, results.analytics_report)
            results.output_files.append(analytics_file)
        
        # Save summary report
        summary = {
            'symbol': results.symbol,
            'execution_mode': results.mode.value,
            'overall_performance': results.overall_performance,
            'risk_assessment': results.risk_assessment,
            'recommendation': results.recommendation,
            'execution_time': results.execution_time,
            'optimization_used': results.optimization_used,
            'output_files': results.output_files,
            'status': results.status,
            'errors': results.errors,
            'warnings': results.warnings
        }
        
        summary_file = f"{self.config.output_dir}/{self.config.symbol}_backtesting_summary.json"
        await safe_json_dump(summary_file, summary)
        results.output_files.append(summary_file)
        self.logger.info(f"💾 Summary saved to {summary_file}")
    
    async def cleanup(self) -> None:
        """Cleanup temporary files and resources."""
        if self.config.cleanup_temp_files:
            self.logger.info("🧹 Cleaning up temporary files...")
            
            # Cleanup M1 optimizers
            if self.m1_memory:
                self.m1_memory.cleanup()
            
            # Force garbage collection
            gc.collect()
            
            self.logger.info("✅ Cleanup completed")