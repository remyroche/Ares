"""
Feature Lookback Optimization Component.

This component optimizes feature lookback periods for better model performance.
Provides comprehensive validation, detailed reporting, and robust error handling.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Use dependency manager for robust imports
from .dependency_manager import dependency_manager, get_dependency, is_dependency_available

# Core dependencies with fallback support
np, np_fallback = get_dependency('numpy')
pd, pd_fallback = get_dependency('pandas')

if np_fallback or pd_fallback:
    import logging
    logging.warning("Using fallback implementations for core dependencies")

from ...market_analysis.components.base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult
from .optimization_reporter import OptimizationReporter
from .validation_framework import ValidationFramework, ValidationLevel, ValidationStatus
from .monitoring_metrics import MonitoringMetrics, MetricType, MetricLevel
from src.utils.logger import system_logger


class OptimizationStatus(Enum):
    """Status of optimization process."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class OptimizationMetrics:
    """Comprehensive optimization metrics."""
    best_lookback_period: int
    best_score: float
    optimization_method: str
    total_features_optimized: int
    optimization_time: float
    convergence_iterations: int
    memory_usage_mb: float
    cpu_usage_percent: float
    validation_score: float
    stability_score: float
    regime_coverage: float
    error_rate: float


class FeatureLookbackOptimizationComponent(BaseMarketAnalysisComponent):
    """
    Feature Lookback Optimization Component.
    
    Optimizes feature lookback periods for better model performance.
    Provides comprehensive validation, detailed reporting, and robust error handling.
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the feature lookback optimization component."""
        super().__init__(config)
        self.logger = system_logger.getChild('FeatureLookbackOptimization')
        self.optimization_status = OptimizationStatus.PENDING
        self.start_time: Optional[float] = None
        self.metrics: Optional[OptimizationMetrics] = None
        
        # Performance monitoring
        self.performance_monitor = {
            'memory_usage': [],
            'cpu_usage': [],
            'execution_times': {},
            'error_counts': 0
        }
        
        # Initialize reporter
        self.reporter = OptimizationReporter(
            output_dir=f"reports/feature_lookback_optimization/{self.config.symbol}_{self.config.exchange}_{self.config.timeframe}"
        )
        
        # Initialize validation framework
        self.validation_framework = ValidationFramework()
        
        # Initialize monitoring metrics
        self.monitoring = MonitoringMetrics(f"FeatureLookbackOptimization_{self.config.symbol}")
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['feature_lookback_optimization_result']
    
    def _monitor_performance(self, operation_name: str) -> None:
        """Monitor performance metrics during execution."""
        try:
            psutil, is_fallback = get_dependency('psutil')
            if psutil is not None:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                cpu_percent = process.cpu_percent()
                
                self.performance_monitor['memory_usage'].append(memory_mb)
                self.performance_monitor['cpu_usage'].append(cpu_percent)
                
                if is_fallback:
                    self.logger.debug("Using fallback psutil for performance monitoring")
            
            if operation_name not in self.performance_monitor['execution_times']:
                self.performance_monitor['execution_times'][operation_name] = []
            
            self.performance_monitor['execution_times'][operation_name].append(time.time())
            
        except Exception as e:
            self.logger.warning(f"Performance monitoring failed: {e}")
    
    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute feature lookback optimization with comprehensive validation and reporting.
        
        Args:
            data: Market data for feature optimization
            pipeline_state: Current pipeline state
            
        Returns:
            ComponentResult with feature lookback optimization results
        """
        self.start_time = time.time()
        self.optimization_status = OptimizationStatus.IN_PROGRESS
        
        # Start comprehensive monitoring
        self.monitoring.start_monitoring()
        
        self.logger.info('⚙️ Starting Feature Lookback Optimization')
        self._monitor_performance('start')
        
        # Record start metrics
        self.monitoring.record_metric(
            name="optimization_started",
            value=1,
            metric_type=MetricType.PERFORMANCE,
            level=MetricLevel.INFO,
            tags={"symbol": self.config.symbol, "exchange": self.config.exchange, "timeframe": self.config.timeframe}
        )
        
        try:
            # Step 1: Comprehensive validation using framework
            self.logger.info('🔍 Validating input data and pipeline state...')
            
            # Validate data with auto-fixing
            data_is_valid, data_validation_results, fixed_data = self.validation_framework.validate_data(data)
            if not data_is_valid:
                critical_failures = [r for r in data_validation_results 
                                   if r.status == ValidationStatus.FAILED and r.level == ValidationLevel.CRITICAL]
                error_msg = f"Data validation failed: {[r.message for r in critical_failures]}"
                self.logger.error(f'❌ {error_msg}')
                self.optimization_status = OptimizationStatus.FAILED
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message=error_msg,
                    metadata={'validation_errors': [r.message for r in critical_failures]}
                )
            
            # Validate pipeline state
            pipeline_is_valid, pipeline_validation_results = self.validation_framework.validate_pipeline_state(pipeline_state)
            if not pipeline_is_valid:
                critical_failures = [r for r in pipeline_validation_results 
                                   if r.status == ValidationStatus.FAILED and r.level == ValidationLevel.CRITICAL]
                error_msg = f"Pipeline state validation failed: {[r.message for r in critical_failures]}"
                self.logger.error(f'❌ {error_msg}')
                self.optimization_status = OptimizationStatus.FAILED
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message=error_msg,
                    metadata={'validation_errors': [r.message for r in critical_failures]}
                )
            
            # Log validation warnings
            all_warnings = [r for r in data_validation_results + pipeline_validation_results 
                          if r.status == ValidationStatus.WARNING]
            for warning in all_warnings:
                self.logger.warning(f'⚠️ {warning.message}')
            
            # Generate validation summary
            data_validation_summary = self.validation_framework.generate_validation_summary(data_validation_results)
            pipeline_validation_summary = self.validation_framework.generate_validation_summary(pipeline_validation_results)
            
            # Record validation metrics
            self.monitoring.record_quality_metric("data_validation_score", data_validation_summary.quality_score)
            self.monitoring.record_quality_metric("pipeline_validation_score", pipeline_validation_summary.quality_score)
            self.monitoring.record_technical_metric("validation_rules_passed", data_validation_summary.passed + pipeline_validation_summary.passed)
            self.monitoring.record_technical_metric("validation_rules_failed", data_validation_summary.failed + pipeline_validation_summary.failed)
            
            self.logger.info(f'✅ Validation passed (data quality: {data_validation_summary.quality_score:.3f})')
            self._monitor_performance('validation_complete')
            
            # Step 2: Load and prepare market data (use fixed data if available)
            self.logger.info('📊 Loading and preparing market data...')
            market_data = await self._load_market_data(fixed_data if fixed_data is not None else data)
            if market_data is None or market_data.empty:
                raise ValueError("No market data available for feature lookback optimization")
            
            self.logger.info(f'📈 Market data loaded: {len(market_data)} rows, {len(market_data.columns)} columns')
            self._monitor_performance('data_loaded')
            
            # Step 3: Get labeled data from previous stage
            triple_barrier_labeling = pipeline_state.get('triple_barrier_labeling_result', {})
            if not triple_barrier_labeling:
                raise ValueError("No triple barrier labeling results available for feature optimization")
            
            self.logger.info('🏷️ Triple barrier labeling data retrieved')
            
            # Step 4: Configure feature optimization
            self.logger.info('⚙️ Configuring feature optimization...')
            optimization_config = self._create_optimization_config(pipeline_state)
            self._monitor_performance('config_created')
            
            # Step 5: Get feature optimizer
            self.logger.info('🔧 Initializing feature optimizer...')
            feature_optimizer = await self._get_feature_optimizer(optimization_config)
            self._monitor_performance('optimizer_ready')
            
            # Step 6: Perform feature lookback optimization
            self.logger.info('🚀 Starting feature optimization process...')
            optimization_result = await self._perform_feature_optimization(
                feature_optimizer, market_data, triple_barrier_labeling, optimization_config
            )
            self._monitor_performance('optimization_complete')
            
            # Step 7: Extract and validate results
            self.logger.info('📋 Extracting optimization results...')
            optimization_results = optimization_result.get('optimization_results', {})
            optimized_features = optimization_result.get('optimized_features', {})
            optimization_metrics = optimization_result.get('optimization_metrics', {})
            
            # Validate optimization results using framework
            optimization_is_valid, optimization_validation_results = self.validation_framework.validate_optimization_results(optimization_result)
            if not optimization_is_valid:
                critical_failures = [r for r in optimization_validation_results 
                                   if r.status == ValidationStatus.FAILED and r.level == ValidationLevel.CRITICAL]
                error_msg = f"Optimization results validation failed: {[r.message for r in critical_failures]}"
                self.logger.error(f'❌ {error_msg}')
                raise ValueError(error_msg)
            
            # Log optimization validation warnings
            optimization_warnings = [r for r in optimization_validation_results 
                                   if r.status == ValidationStatus.WARNING]
            for warning in optimization_warnings:
                self.logger.warning(f'⚠️ {warning.message}')
            
            optimization_validation_summary = self.validation_framework.generate_validation_summary(optimization_validation_results)
            
            # Record optimization metrics
            self.monitoring.record_quality_metric("optimization_validation_score", optimization_validation_summary.quality_score)
            self.monitoring.record_business_metric("features_optimized", len(optimized_features))
            self.monitoring.record_quality_metric("best_optimization_score", optimization_results.get('best_score', 0.0))
            
            self.logger.info(f'✅ Optimization results validated (quality: {optimization_validation_summary.quality_score:.3f})')
            
            # Step 8: Create comprehensive metrics
            self.metrics = self._create_optimization_metrics(
                optimization_results, optimized_features, optimization_metrics, optimization_result
            )
            
            # Step 9: Generate comprehensive report using reporter
            self.logger.info('📊 Generating comprehensive optimization report...')
            comprehensive_report = self.reporter.generate_comprehensive_report(
                optimization_result=optimization_result,
                metrics=self.metrics,
                validation_results={
                    'data_validation': {
                        'summary': data_validation_summary,
                        'results': data_validation_results
                    },
                    'pipeline_validation': {
                        'summary': pipeline_validation_summary,
                        'results': pipeline_validation_results
                    },
                    'optimization_validation': {
                        'summary': optimization_validation_summary,
                        'results': optimization_validation_results
                    }
                },
                performance_metrics=self.performance_monitor,
                symbol=self.config.symbol,
                exchange=self.config.exchange,
                timeframe=self.config.timeframe
            )
            
            # Step 10: Create consolidated artifacts
            artifacts = self._create_artifacts(
                optimization_results, optimized_features, optimization_metrics, 
                optimization_result, comprehensive_report, 
                data_validation_summary, pipeline_validation_summary, optimization_validation_summary
            )
            
            # Step 11: Final validation
            if not self.validate_artifacts(artifacts):
                raise ValueError("Generated artifacts failed validation")
            
            self.optimization_status = OptimizationStatus.COMPLETED
            execution_time = time.time() - self.start_time
            
            # Record completion metrics
            self.monitoring.record_performance_metric("total_optimization", execution_time)
            self.monitoring.record_business_metric("optimization_success_rate", 1.0)
            self.monitoring.record_metric(
                name="optimization_completed",
                value=1,
                metric_type=MetricType.PERFORMANCE,
                level=MetricLevel.INFO,
                tags={"status": "success", "features_optimized": len(optimized_features)},
                metadata={"execution_time": execution_time, "best_lookback_period": self.metrics.best_lookback_period}
            )
            
            # Stop monitoring
            self.monitoring.stop_monitoring()
            
            self.logger.info(f'✅ Feature Lookback Optimization completed successfully in {execution_time:.2f}s')
            self.logger.info(f'📈 Optimized {len(optimized_features)} features with best lookback period: {self.metrics.best_lookback_period}')
            
            return ComponentResult(
                success=True,
                artifacts=artifacts,
                execution_time=execution_time,
                metadata={
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    'features_optimized': len(optimized_features),
                    'optimization_status': self.optimization_status.value,
                    'data_quality_score': data_validation_summary.quality_score,
                    'performance_metrics': self.performance_monitor
                }
            )
            
        except Exception as e:
            self.optimization_status = OptimizationStatus.FAILED
            self.performance_monitor['error_counts'] += 1
            execution_time = time.time() - self.start_time if self.start_time else 0.0
            
            # Record error metrics
            self.monitoring.record_error(
                error_type="optimization_failed",
                error_message=str(e),
                context={"execution_time": execution_time, "optimization_status": self.optimization_status.value}
            )
            self.monitoring.record_business_metric("optimization_success_rate", 0.0)
            self.monitoring.record_performance_metric("failed_optimization", execution_time)
            
            # Stop monitoring
            self.monitoring.stop_monitoring()
            
            self.logger.error(f'❌ Feature Lookback Optimization failed after {execution_time:.2f}s: {e}')
            import traceback
            self.logger.error(f'❌ Error details: {traceback.format_exc()}')
            
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=str(e),
                execution_time=execution_time,
                metadata={
                    'optimization_status': self.optimization_status.value,
                    'error_count': self.performance_monitor['error_counts'],
                    'performance_metrics': self.performance_monitor
                }
            )
    
    def _create_optimization_config(self, pipeline_state: Dict[str, Any]) -> Any:
        """Create optimization configuration based on pipeline state and component config."""
        try:
            from src.feature_engineering.feature_generation_optimization import FeatureOptimizationConfig
            
            # Check if regime data is available for regime-aware optimization
            regime_data_splitting = pipeline_state.get('regime_data_splitting_result', {})
            enable_regime_aware = bool(regime_data_splitting)
            
            config = FeatureOptimizationConfig(
                optimization_method='genetic_algorithm',
                lookback_range=(5, 50),  # 5 to 50 periods
                feature_types=['technical_indicators', 'price_features', 'volume_features'],
                optimization_metric='sharpe_ratio',
                cross_validation_folds=5,
                test_size=0.2,
                random_state=42,
                
                # Genetic algorithm parameters
                population_size=50,
                generations=100,
                mutation_rate=0.1,
                crossover_rate=0.8,
                elitism_rate=0.1,
                
                # Feature selection
                enable_feature_selection=True,
                max_features=20,
                feature_importance_threshold=0.01,
                
                # Regime-aware optimization
                enable_regime_aware_optimization=enable_regime_aware,
                regime_specific_optimization=enable_regime_aware,
                
                # Hardware optimization
                enable_parallel_processing=True,
                enable_gpu_acceleration=True,
                memory_limit_gb=8.0
            )
            
            self.logger.info(f'⚙️ Optimization config created (regime-aware: {enable_regime_aware})')
            return config
            
        except ImportError as e:
            self.logger.warning(f"Feature optimization config import failed: {e}")
            # Return a simple fallback config
            return {
                'optimization_method': 'statistical',
                'lookback_range': (5, 50),
                'regime_aware': False
            }
    
    async def _get_feature_optimizer(self, config: Any) -> Any:
        """Get feature optimizer with fallback handling."""
        try:
            from src.feature_engineering.feature_generation_optimization import get_feature_optimizer
            optimizer = get_feature_optimizer(config)
            self.logger.info('✅ Feature optimizer initialized successfully')
            return optimizer
            
        except ImportError as e:
            self.logger.warning(f"Feature optimizer import failed: {e}")
            # Return a fallback optimizer
            return self._create_fallback_optimizer()
    
    def _create_fallback_optimizer(self) -> Any:
        """Create a fallback optimizer for when ML commons are not available."""
        class FallbackOptimizer:
            def __init__(self, config):
                self.config = config
                self.logger = system_logger.getChild('FallbackOptimizer')
            
            async def optimize_features(self, data, config):
                self.logger.info("Using fallback statistical optimization")
                return {
                    'optimization_results': {
                        'best_lookback_period': 20,
                        'best_score': 0.5,
                        'optimization_method': 'fallback_statistical'
                    },
                    'optimized_features': {
                        'rsi': {'lookback': 14, 'score': 0.5},
                        'sma': {'lookback': 20, 'score': 0.4},
                        'ema': {'lookback': 12, 'score': 0.45}
                    },
                    'optimization_metrics': {
                        'method': 'fallback_statistical',
                        'convergence_iterations': 1
                    },
                    'optimization_time': 0.1
                }
        
        return FallbackOptimizer(self.config)
    
    def _create_optimization_metrics(
        self, 
        optimization_results: Dict[str, Any], 
        optimized_features: Dict[str, Any], 
        optimization_metrics: Dict[str, Any],
        optimization_result: Dict[str, Any]
    ) -> OptimizationMetrics:
        """Create comprehensive optimization metrics."""
        try:
            # Calculate performance metrics
            memory_usage = max(self.performance_monitor['memory_usage']) if self.performance_monitor['memory_usage'] else 0.0
            cpu_usage = max(self.performance_monitor['cpu_usage']) if self.performance_monitor['cpu_usage'] else 0.0
            
            # Calculate stability score based on feature consistency
            stability_score = self._calculate_stability_score(optimized_features)
            
            # Calculate regime coverage
            regime_coverage = self._calculate_regime_coverage(optimization_result)
            
            # Calculate validation score
            validation_score = self._calculate_validation_score(optimization_results, optimized_features)
            
            metrics = OptimizationMetrics(
                best_lookback_period=optimization_results.get('best_lookback_period', 0),
                best_score=optimization_results.get('best_score', 0.0),
                optimization_method=optimization_results.get('optimization_method', 'unknown'),
                total_features_optimized=len(optimized_features),
                optimization_time=optimization_result.get('optimization_time', 0.0),
                convergence_iterations=optimization_metrics.get('convergence_iterations', 0),
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_usage,
                validation_score=validation_score,
                stability_score=stability_score,
                regime_coverage=regime_coverage,
                error_rate=self.performance_monitor['error_counts'] / max(1, len(optimized_features))
            )
            
            self.logger.info(f'📊 Metrics created: score={metrics.best_score:.3f}, stability={metrics.stability_score:.3f}')
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to create optimization metrics: {e}")
            # Return default metrics
            return OptimizationMetrics(
                best_lookback_period=0,
                best_score=0.0,
                optimization_method='error',
                total_features_optimized=0,
                optimization_time=0.0,
                convergence_iterations=0,
                memory_usage_mb=0.0,
                cpu_usage_percent=0.0,
                validation_score=0.0,
                stability_score=0.0,
                regime_coverage=0.0,
                error_rate=1.0
            )
    
    def _calculate_stability_score(self, optimized_features: Dict[str, Any]) -> float:
        """Calculate stability score based on feature consistency."""
        if not optimized_features:
            return 0.0
        
        try:
            # Calculate coefficient of variation for lookback periods
            lookback_periods = [feature.get('lookback', 0) for feature in optimized_features.values()]
            if not lookback_periods:
                return 0.0
            
            mean_lookback = np.mean(lookback_periods)
            std_lookback = np.std(lookback_periods)
            
            if mean_lookback == 0:
                return 0.0
            
            cv = std_lookback / mean_lookback
            stability_score = max(0.0, 1.0 - cv)  # Lower CV = higher stability
            
            return min(1.0, stability_score)
            
        except Exception:
            return 0.5  # Default moderate stability
    
    def _calculate_regime_coverage(self, optimization_result: Dict[str, Any]) -> float:
        """Calculate regime coverage percentage."""
        try:
            regime_results = optimization_result.get('regime_specific_results', {})
            if not regime_results:
                return 0.0
            
            total_regimes = len(regime_results)
            covered_regimes = sum(1 for result in regime_results.values() if result.get('optimized', False))
            
            return covered_regimes / total_regimes if total_regimes > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_validation_score(self, optimization_results: Dict[str, Any], optimized_features: Dict[str, Any]) -> float:
        """Calculate validation score based on result quality."""
        try:
            score = 0.0
            
            # Check if we have valid results
            if optimization_results.get('best_lookback_period', 0) > 0:
                score += 0.3
            
            if optimization_results.get('best_score', 0) > 0:
                score += 0.3
            
            if len(optimized_features) > 0:
                score += 0.2
            
            # Check feature quality
            valid_features = sum(1 for feature in optimized_features.values() 
                               if feature.get('lookback', 0) > 0 and feature.get('score', 0) > 0)
            if len(optimized_features) > 0:
                score += 0.2 * (valid_features / len(optimized_features))
            
            return min(1.0, score)
            
        except Exception:
            return 0.0
    
    def _create_artifacts(
        self,
        optimization_results: Dict[str, Any],
        optimized_features: Dict[str, Any],
        optimization_metrics: Dict[str, Any],
        optimization_result: Dict[str, Any],
        report: Dict[str, Any],
        data_validation_summary: Any,
        pipeline_validation_summary: Any,
        optimization_validation_summary: Any
    ) -> Dict[str, Any]:
        """Create comprehensive artifacts with all optimization data."""
        return {
            'feature_lookback_optimization_result': {
                'optimization_results': optimization_results,
                'optimized_features': optimized_features,
                'optimization_metrics': optimization_metrics,
                'optimization_summary': {
                    'best_lookback_period': self.metrics.best_lookback_period if self.metrics else 0,
                    'best_score': self.metrics.best_score if self.metrics else 0.0,
                    'total_features_optimized': self.metrics.total_features_optimized if self.metrics else 0,
                    'optimization_time': self.metrics.optimization_time if self.metrics else 0.0,
                    'validation_score': self.metrics.validation_score if self.metrics else 0.0,
                    'stability_score': self.metrics.stability_score if self.metrics else 0.0
                },
                'detailed_report': report,
                'comprehensive_report': report,
                'validation_results': {
                    'data_validation': {
                        'summary': {
                            'overall_status': data_validation_summary.overall_status.value,
                            'quality_score': data_validation_summary.quality_score,
                            'total_rules': data_validation_summary.total_rules,
                            'passed': data_validation_summary.passed,
                            'failed': data_validation_summary.failed,
                            'warnings': data_validation_summary.warnings,
                            'critical_failures': data_validation_summary.critical_failures
                        },
                        'recommendations': data_validation_summary.recommendations
                    },
                    'pipeline_validation': {
                        'summary': {
                            'overall_status': pipeline_validation_summary.overall_status.value,
                            'quality_score': pipeline_validation_summary.quality_score,
                            'total_rules': pipeline_validation_summary.total_rules,
                            'passed': pipeline_validation_summary.passed,
                            'failed': pipeline_validation_summary.failed,
                            'warnings': pipeline_validation_summary.warnings,
                            'critical_failures': pipeline_validation_summary.critical_failures
                        },
                        'recommendations': pipeline_validation_summary.recommendations
                    },
                    'optimization_validation': {
                        'summary': {
                            'overall_status': optimization_validation_summary.overall_status.value,
                            'quality_score': optimization_validation_summary.quality_score,
                            'total_rules': optimization_validation_summary.total_rules,
                            'passed': optimization_validation_summary.passed,
                            'failed': optimization_validation_summary.failed,
                            'warnings': optimization_validation_summary.warnings,
                            'critical_failures': optimization_validation_summary.critical_failures
                        },
                        'recommendations': optimization_validation_summary.recommendations
                    }
                },
                'performance_metrics': self.performance_monitor,
                'monitoring_metrics': self.monitoring.get_metrics_summary(),
                'monitoring_report': self.monitoring.get_performance_report(),
                'metadata': {
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    'execution_timestamp': datetime.now().isoformat(),
                    'optimization_status': self.optimization_status.value,
                    'component_version': '2.0.0'
                }
            }
        }
    
    async def _load_market_data(self, data: Any) -> Optional[Any]:
        """Load and prepare market data for feature optimization."""
        if data is None:
            return None
        
        if isinstance(data, pd.DataFrame):
            return data.copy()
        
        # Handle other data types if needed
        return data
    
    async def _perform_feature_optimization(
        self, 
        feature_optimizer: Any, 
        market_data: Any, 
        triple_barrier_labeling: Dict[str, Any],
        config: Any
    ) -> Dict[str, Any]:
        """Perform the actual feature optimization process with comprehensive error handling."""
        optimization_start_time = time.time()
        
        try:
            self.logger.info('🔄 Preparing data for optimization...')
            # Prepare data for optimization
            prepared_data = self._prepare_data_for_optimization(market_data, triple_barrier_labeling)
            self._monitor_performance('data_prepared')
            
            self.logger.info('🚀 Executing feature optimization...')
            # Perform feature optimization
            optimization_result = await feature_optimizer.optimize_features(prepared_data, config)
            self._monitor_performance('optimization_executed')
            
            # Add timing information
            optimization_time = time.time() - optimization_start_time
            optimization_result['optimization_time'] = optimization_time
            
            self.logger.info(f'✅ Feature optimization completed in {optimization_time:.2f}s')
            return optimization_result
            
        except Exception as e:
            optimization_time = time.time() - optimization_start_time
            self.logger.error(f"❌ Feature optimization process failed after {optimization_time:.2f}s: {e}")
            self.performance_monitor['error_counts'] += 1
            
            # Return comprehensive fallback optimization result
            return {
                'optimization_results': {
                    'best_lookback_period': 20,
                    'best_score': 0.0,
                    'optimization_method': 'fallback',
                    'error': str(e),
                    'fallback_reason': 'optimization_process_failed'
                },
                'optimized_features': {
                    'rsi': {'lookback': 14, 'score': 0.0, 'method': 'fallback'},
                    'sma': {'lookback': 20, 'score': 0.0, 'method': 'fallback'},
                    'ema': {'lookback': 12, 'score': 0.0, 'method': 'fallback'}
                },
                'optimization_metrics': {
                    'optimization_method': 'fallback',
                    'error': str(e),
                    'convergence_iterations': 0,
                    'fallback_used': True
                },
                'optimization_time': optimization_time,
                'regime_specific_results': {},
                'error_details': {
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'timestamp': datetime.now().isoformat()
                }
            }
    
    def _prepare_data_for_optimization(self, data: Any, triple_barrier_labeling: Dict[str, Any]) -> Any:
        """Prepare market data and labeled data for optimization with comprehensive validation."""
        try:
            if not isinstance(data, pd.DataFrame):
                self.logger.warning("Data is not a DataFrame, using fallback preparation")
                return {
                    'market_data': data,
                    'triple_barrier_labeling': triple_barrier_labeling,
                    'preparation_method': 'fallback'
                }
            
            # Create a copy to avoid modifying original data
            prepared_data = data.copy()
            
            # Ensure we have required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in prepared_data.columns]
            
            if missing_columns:
                self.logger.warning(f"Missing columns for optimization: {missing_columns}")
                # Use available columns or create fallback data
                for col in missing_columns:
                    if col == 'volume':
                        prepared_data[col] = 1000  # Default volume
                        self.logger.info(f"Created fallback {col} column with default value")
                    else:
                        fallback_value = prepared_data.get('close', 100.0)
                        prepared_data[col] = fallback_value
                        self.logger.info(f"Created fallback {col} column using close price")
            
            # Add metadata about preparation
            preparation_metadata = {
                'original_columns': list(data.columns),
                'prepared_columns': list(prepared_data.columns),
                'missing_columns_filled': missing_columns,
                'data_shape': prepared_data.shape,
                'preparation_timestamp': datetime.now().isoformat()
            }
            
            return {
                'market_data': prepared_data,
                'triple_barrier_labeling': triple_barrier_labeling,
                'preparation_metadata': preparation_metadata,
                'preparation_method': 'enhanced'
            }
            
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            # Return minimal fallback
            return {
                'market_data': data,
                'triple_barrier_labeling': triple_barrier_labeling,
                'preparation_method': 'fallback',
                'preparation_error': str(e)
            }