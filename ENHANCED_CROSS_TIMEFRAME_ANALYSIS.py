"""
Enhanced Cross Timeframe Analysis Component

This is an improved version of the cross timeframe analysis that addresses:
1. Silent failure prevention
2. Enhanced error handling
3. Comprehensive reporting
4. Streamlined code structure
5. Resource management
"""

import asyncio
import json
import logging
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import psutil
import gc

# Handle optional dependencies gracefully
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

from .base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult
from src.utils.logger import system_logger


class FailureMode(Enum):
    """Defines how the analysis should handle errors."""
    STRICT = "strict"      # Fail fast on any error
    GRACEFUL = "graceful"  # Continue with partial results
    FALLBACK = "fallback"  # Use fallback methods


class AnalysisError(Exception):
    """Base exception for analysis errors with context."""
    def __init__(self, error_type: str, message: str, context: Dict[str, Any] = None):
        self.error_type = error_type
        self.message = message
        self.context = context or {}
        super().__init__(f"[{error_type}] {message}")


class DataLoadingError(AnalysisError):
    """Raised when data loading fails."""
    pass


class FeatureGenerationError(AnalysisError):
    """Raised when feature generation fails."""
    pass


class ValidationError(AnalysisError):
    """Raised when validation fails."""
    pass


@dataclass
class EnhancedCrossTimeframeConfig(ComponentConfig):
    """Enhanced configuration for cross timeframe analysis."""
    # Analysis parameters
    timeframes: List[str] = field(default_factory=lambda: ['1m', '5m', '15m', '30m'])
    base_timeframe: str = '1m'
    
    # Feature engineering
    interaction_features: List[str] = field(default_factory=lambda: ['correlation', 'momentum', 'volatility', 'volume'])
    lookback_periods: List[int] = field(default_factory=lambda: [3, 5, 10, 15, 20])
    
    # Quality thresholds
    correlation_threshold: float = 0.6
    min_observations: int = 50
    min_required_timeframes: int = 2
    min_required_features: int = 10
    
    # Error handling
    failure_mode: FailureMode = FailureMode.STRICT
    enable_fallback_analysis: bool = False
    
    # Performance
    enable_parallel_processing: bool = True
    max_workers: int = 4
    memory_limit_gb: float = 8.0
    chunk_size: int = 10000
    
    # Reporting
    enable_detailed_reporting: bool = True
    report_interval_seconds: int = 30


@dataclass
class MemoryStatus:
    """Memory usage status."""
    usage_gb: float
    limit_gb: float
    usage_percent: float
    is_critical: bool


@dataclass
class ValidationReport:
    """Validation report for analysis results."""
    is_valid: bool
    issues: List[str]
    quality_score: float
    recommendations: List[str] = field(default_factory=list)


@dataclass
class AnalysisReport:
    """Comprehensive analysis report."""
    execution_summary: Dict[str, Any]
    progress_checkpoints: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    quality_metrics: Dict[str, Any]
    recommendations: List[str]


class MemoryManager:
    """Manages memory usage and cleanup."""
    
    def __init__(self, limit_gb: float):
        self.limit_gb = limit_gb
        self.logger = system_logger.getChild('MemoryManager')
    
    def check_memory_usage(self) -> MemoryStatus:
        """Check current memory usage and return status."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            usage_gb = memory_info.rss / (1024**3)
            
            return MemoryStatus(
                usage_gb=usage_gb,
                limit_gb=self.limit_gb,
                usage_percent=(usage_gb / self.limit_gb) * 100,
                is_critical=usage_gb >= self.limit_gb * 0.9
            )
        except Exception as e:
            self.logger.warning(f"Failed to check memory usage: {e}")
            return MemoryStatus(0.0, self.limit_gb, 0.0, False)
    
    def cleanup_if_needed(self):
        """Clean up memory if usage is high."""
        status = self.check_memory_usage()
        if status.is_critical:
            gc.collect()
            self.logger.warning(f"🧹 Memory cleanup performed. Usage: {status.usage_gb:.2f}GB")
        return status


class AnalysisProgressReporter:
    """Reports progress and generates comprehensive reports."""
    
    def __init__(self, config: EnhancedCrossTimeframeConfig):
        self.config = config
        self.logger = system_logger.getChild('AnalysisProgressReporter')
        self.start_time = time.time()
        self.checkpoints = []
        self.metrics = {}
    
    def report_checkpoint(self, step: str, status: str, details: Dict[str, Any]):
        """Report progress at key checkpoints."""
        checkpoint = {
            'step': step,
            'status': status,
            'timestamp': time.time(),
            'elapsed_time': time.time() - self.start_time,
            'details': details
        }
        self.checkpoints.append(checkpoint)
        
        # Log progress
        self.logger.info(f"📊 [{step}] {status} - {details}")
        
        # Report to external monitoring if enabled
        if self.config.enable_detailed_reporting:
            self._report_to_monitoring(checkpoint)
    
    def track_metric(self, name: str, value: float, unit: str = ""):
        """Track performance and quality metrics."""
        self.metrics[name] = {
            'value': value,
            'unit': unit,
            'timestamp': time.time()
        }
    
    def _report_to_monitoring(self, checkpoint: Dict[str, Any]):
        """Report to external monitoring systems."""
        # This could be extended to send to external monitoring systems
        pass
    
    def generate_final_report(self, result: ComponentResult) -> AnalysisReport:
        """Generate comprehensive final report."""
        return AnalysisReport(
            execution_summary={
                'total_time': time.time() - self.start_time,
                'success': result.success,
                'features_generated': len(result.artifacts.get('cross_timeframe_features', {})),
                'data_quality_score': result.metadata.get('data_quality_score', 0.0),
                'error_count': len(result.metadata.get('errors', []))
            },
            progress_checkpoints=self.checkpoints,
            performance_metrics=self.metrics,
            quality_metrics=self._calculate_quality_metrics(result),
            recommendations=self._generate_recommendations(result)
        )
    
    def _calculate_quality_metrics(self, result: ComponentResult) -> Dict[str, Any]:
        """Calculate quality metrics from the result."""
        return {
            'data_quality_score': result.metadata.get('data_quality_score', 0.0),
            'feature_completeness': len(result.artifacts.get('cross_timeframe_features', {})) / self.config.min_required_features,
            'error_rate': len(result.metadata.get('errors', [])) / max(1, len(self.checkpoints))
        }
    
    def _generate_recommendations(self, result: ComponentResult) -> List[str]:
        """Generate recommendations based on the analysis results."""
        recommendations = []
        
        if result.metadata.get('data_quality_score', 1.0) < 0.8:
            recommendations.append("Consider improving data quality - current score below 0.8")
        
        if len(result.artifacts.get('cross_timeframe_features', {})) < self.config.min_required_features:
            recommendations.append(f"Increase feature generation - only {len(result.artifacts.get('cross_timeframe_features', {}))} features generated")
        
        if len(result.metadata.get('errors', [])) > 0:
            recommendations.append("Review and address errors in the analysis pipeline")
        
        return recommendations


class AnalysisResultValidator:
    """Validates analysis results to prevent silent failures."""
    
    def __init__(self, config: EnhancedCrossTimeframeConfig):
        self.config = config
        self.logger = system_logger.getChild('AnalysisResultValidator')
    
    def validate_result(self, result: ComponentResult) -> ValidationReport:
        """Validate that the analysis result meets quality standards."""
        issues = []
        recommendations = []
        
        # Check required artifacts
        if not result.artifacts.get('cross_timeframe_features'):
            issues.append("Missing cross_timeframe_features artifact")
        
        # Check feature quality
        features = result.artifacts.get('cross_timeframe_features', {})
        if len(features) < self.config.min_required_features:
            issues.append(f"Insufficient features: {len(features)} < {self.config.min_required_features}")
            recommendations.append("Increase feature generation or reduce min_required_features threshold")
        
        # Check data quality
        data_quality_score = result.metadata.get('data_quality_score', 1.0)
        if data_quality_score < 0.8:
            issues.append(f"Data quality below threshold: {data_quality_score} < 0.8")
            recommendations.append("Improve data quality or adjust quality thresholds")
        
        # Check for critical errors
        errors = result.metadata.get('errors', [])
        critical_errors = [e for e in errors if 'critical' in str(e).lower()]
        if critical_errors:
            issues.append(f"Critical errors found: {len(critical_errors)}")
            recommendations.append("Address critical errors before using results")
        
        quality_score = self._calculate_quality_score(result)
        
        return ValidationReport(
            is_valid=len(issues) == 0,
            issues=issues,
            quality_score=quality_score,
            recommendations=recommendations
        )
    
    def _calculate_quality_score(self, result: ComponentResult) -> float:
        """Calculate overall quality score."""
        score = 1.0
        
        # Penalize for missing features
        features = result.artifacts.get('cross_timeframe_features', {})
        if len(features) < self.config.min_required_features:
            score *= 0.5
        
        # Penalize for data quality issues
        data_quality_score = result.metadata.get('data_quality_score', 1.0)
        score *= data_quality_score
        
        # Penalize for errors
        errors = result.metadata.get('errors', [])
        if errors:
            score *= max(0.1, 1.0 - (len(errors) * 0.1))
        
        return score


class EnhancedCrossTimeframeAnalysisComponent(BaseMarketAnalysisComponent):
    """
    Enhanced Cross Timeframe Analysis Component.
    
    Addresses silent failures, provides comprehensive reporting, and implements
    robust error handling.
    """
    
    def __init__(self, config: Optional[EnhancedCrossTimeframeConfig] = None):
        """Initialize the enhanced cross timeframe analysis component."""
        super().__init__(config)
        self.config = config or EnhancedCrossTimeframeConfig()
        self.logger = system_logger.getChild('EnhancedCrossTimeframeAnalysis')
        
        # Initialize managers
        self.memory_manager = MemoryManager(self.config.memory_limit_gb)
        self.progress_reporter = AnalysisProgressReporter(self.config)
        self.result_validator = AnalysisResultValidator(self.config)
        
        # Track errors and metrics
        self.errors = []
        self.metrics = {}
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['cross_timeframe_analysis_result']
    
    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute enhanced cross timeframe analysis with comprehensive error handling.
        """
        self.logger.info('🌐 Starting Enhanced Cross Timeframe Analysis')
        self.progress_reporter.report_checkpoint('start', 'analysis_started', {
            'symbol': self.config.symbol,
            'exchange': self.config.exchange,
            'timeframe': self.config.timeframe
        })
        
        try:
            # Check memory usage
            memory_status = self.memory_manager.check_memory_usage()
            self.progress_reporter.track_metric('initial_memory_gb', memory_status.usage_gb, 'GB')
            
            # Load and validate market data
            market_data = await self._load_and_validate_market_data(data)
            self.progress_reporter.report_checkpoint('data_loading', 'completed', {
                'data_points': len(market_data) if market_data is not None else 0
            })
            
            # Get feature optimization results
            feature_lookback_optimization = await self._get_feature_optimization_results(pipeline_state)
            self.progress_reporter.report_checkpoint('feature_optimization', 'retrieved', {
                'optimization_available': bool(feature_lookback_optimization)
            })
            
            # Perform cross timeframe analysis
            analysis_result = await self._perform_enhanced_cross_timeframe_analysis(
                market_data, feature_lookback_optimization
            )
            self.progress_reporter.report_checkpoint('analysis', 'completed', {
                'features_generated': len(analysis_result.get('cross_timeframe_features', {})),
                'analysis_time': analysis_result.get('analysis_time', 0.0)
            })
            
            # Validate results
            validation_result = await self._validate_analysis_results(analysis_result)
            self.progress_reporter.report_checkpoint('validation', 'completed', {
                'is_valid': validation_result.is_valid,
                'quality_score': validation_result.quality_score,
                'issues_count': len(validation_result.issues)
            })
            
            # Create artifacts
            artifacts = await self._create_artifacts(analysis_result, validation_result)
            
            # Generate final report
            final_report = self.progress_reporter.generate_final_report(
                ComponentResult(success=True, artifacts=artifacts, metadata={
                    'data_quality_score': validation_result.quality_score,
                    'errors': self.errors,
                    'validation_report': validation_result
                })
            )
            
            self.progress_reporter.report_checkpoint('completion', 'success', {
                'total_features': len(artifacts.get('cross_timeframe_analysis_result', {}).get('cross_timeframe_features', {})),
                'quality_score': validation_result.quality_score
            })
            
            return ComponentResult(
                success=True,
                artifacts=artifacts,
                metadata={
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    'data_quality_score': validation_result.quality_score,
                    'errors': self.errors,
                    'validation_report': validation_result,
                    'final_report': final_report
                }
            )
            
        except Exception as e:
            self.logger.error(f'❌ Enhanced Cross Timeframe Analysis failed: {e}')
            self.logger.error(f'❌ Error details: {traceback.format_exc()}')
            
            # Add error to tracking
            self.errors.append({
                'type': 'execution_error',
                'message': str(e),
                'timestamp': time.time(),
                'traceback': traceback.format_exc()
            })
            
            # Generate failure report
            failure_report = self.progress_reporter.generate_final_report(
                ComponentResult(success=False, artifacts={}, metadata={'errors': self.errors})
            )
            
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=str(e),
                metadata={
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    'errors': self.errors,
                    'failure_report': failure_report
                }
            )
    
    async def _load_and_validate_market_data(self, data: Any) -> Any:
        """Load and validate market data with comprehensive error handling."""
        try:
            if data is None:
                raise DataLoadingError(
                    error_type='data_loading',
                    message="No market data provided",
                    context={'data_type': type(data).__name__}
                )
            
            if not PANDAS_AVAILABLE:
                raise DataLoadingError(
                    error_type='data_loading',
                    message="Pandas not available for data processing",
                    context={'pandas_available': False}
                )
            
            if not isinstance(data, pd.DataFrame):
                raise DataLoadingError(
                    error_type='data_loading',
                    message=f"Expected pandas DataFrame, got {type(data).__name__}",
                    context={'data_type': type(data).__name__}
                )
            
            if data.empty:
                raise DataLoadingError(
                    error_type='data_loading',
                    message="Market data is empty",
                    context={'data_shape': data.shape}
                )
            
            # Validate required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise DataLoadingError(
                    error_type='data_loading',
                    message=f"Missing required columns: {missing_columns}",
                    context={'missing_columns': missing_columns, 'available_columns': list(data.columns)}
                )
            
            # Check for data quality issues
            data_quality_issues = []
            for col in required_columns:
                if data[col].isna().sum() > len(data) * 0.1:  # More than 10% NaN
                    data_quality_issues.append(f"Column {col} has {data[col].isna().sum()} NaN values")
                
                if data[col].dtype in ['object']:
                    data_quality_issues.append(f"Column {col} has non-numeric data type: {data[col].dtype}")
            
            if data_quality_issues:
                self.logger.warning(f"Data quality issues detected: {data_quality_issues}")
                self.errors.append({
                    'type': 'data_quality',
                    'message': 'Data quality issues detected',
                    'issues': data_quality_issues,
                    'timestamp': time.time()
                })
            
            return data.copy()
            
        except DataLoadingError:
            raise
        except Exception as e:
            raise DataLoadingError(
                error_type='data_loading',
                message=f"Unexpected error during data loading: {e}",
                context={'error': str(e), 'data_type': type(data).__name__}
            )
    
    async def _get_feature_optimization_results(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get feature optimization results with validation."""
        feature_lookback_optimization = pipeline_state.get('feature_lookback_optimization_result', {})
        
        if not feature_lookback_optimization:
            if self.config.failure_mode == FailureMode.STRICT:
                raise ValidationError(
                    error_type='validation',
                    message="No feature lookback optimization results available",
                    context={'pipeline_state_keys': list(pipeline_state.keys())}
                )
            else:
                self.logger.warning("No feature lookback optimization results available, continuing without them")
                return {}
        
        return feature_lookback_optimization
    
    async def _perform_enhanced_cross_timeframe_analysis(
        self, 
        market_data: Any, 
        feature_lookback_optimization: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform the actual cross timeframe analysis with enhanced error handling."""
        try:
            # Import cross timeframe analysis utilities
            from src.feature_engineering.cross_timeframe_features import CrossTimeframeFeatureGenerator, CrossTimeframeConfig
            
            # Configure cross timeframe analysis
            analysis_config = CrossTimeframeConfig(
                target_timeframe=self.config.timeframe,
                analysis_timeframes=self.config.timeframes,
                feature_types=self.config.interaction_features,
                interaction_depth=2,
                correlation_threshold=self.config.correlation_threshold,
                significance_threshold=0.05,
                
                # Feature generation
                enable_momentum_features='momentum' in self.config.interaction_features,
                enable_volume_features='volume' in self.config.interaction_features,
                enable_volatility_features='volatility' in self.config.interaction_features,
                enable_trend_features='trend' in self.config.interaction_features,
                
                # Statistical analysis
                enable_correlation_analysis=True,
                enable_causality_analysis=True,
                enable_regime_analysis=True,
                
                # Feature selection
                enable_feature_selection=True,
                max_features_per_timeframe=10,
                feature_importance_threshold=0.01,
                
                # Hardware optimization
                enable_parallel_processing=self.config.enable_parallel_processing,
                enable_gpu_acceleration=False,  # Disable for now
                memory_limit_gb=self.config.memory_limit_gb
            )
            
            # Create cross timeframe feature generator
            feature_generator = CrossTimeframeFeatureGenerator(analysis_config)
            
            # Prepare data for analysis
            prepared_data = self._prepare_data_for_analysis(market_data, feature_lookback_optimization)
            
            # Perform cross timeframe analysis
            start_time = time.time()
            analysis_result = await feature_generator.generate_cross_timeframe_features(prepared_data, analysis_config)
            analysis_time = time.time() - start_time
            
            # Add timing information
            analysis_result['analysis_time'] = analysis_time
            
            # Track performance metrics
            self.progress_reporter.track_metric('analysis_time_seconds', analysis_time, 'seconds')
            self.progress_reporter.track_metric('features_generated', len(analysis_result.get('cross_timeframe_features', {})), 'count')
            
            return analysis_result
            
        except Exception as e:
            if self.config.failure_mode == FailureMode.STRICT:
                raise FeatureGenerationError(
                    error_type='feature_generation',
                    message=f"Cross timeframe analysis failed: {e}",
                    context={'error': str(e), 'config': str(analysis_config)}
                )
            else:
                self.logger.error(f"Cross timeframe analysis failed, using fallback: {e}")
                self.errors.append({
                    'type': 'feature_generation',
                    'message': str(e),
                    'timestamp': time.time()
                })
                
                # Return minimal fallback result
                return {
                    'cross_timeframe_features': {},
                    'analysis_metrics': {
                        'analysis_method': 'fallback',
                        'error': str(e)
                    },
                    'feature_interactions': {
                        'significant_interactions': [],
                        'correlation_matrix': {}
                    },
                    'analysis_time': 0.0
                }
    
    async def _validate_analysis_results(self, analysis_result: Dict[str, Any]) -> ValidationReport:
        """Validate analysis results to ensure quality."""
        return self.result_validator.validate_result(
            ComponentResult(
                success=True,
                artifacts={'cross_timeframe_analysis_result': analysis_result},
                metadata={'data_quality_score': 1.0, 'errors': self.errors}
            )
        )
    
    async def _create_artifacts(self, analysis_result: Dict[str, Any], validation_result: ValidationReport) -> Dict[str, Any]:
        """Create artifacts with comprehensive metadata."""
        cross_timeframe_features = analysis_result.get('cross_timeframe_features', {})
        analysis_metrics = analysis_result.get('analysis_metrics', {})
        feature_interactions = analysis_result.get('feature_interactions', {})
        
        return {
            'cross_timeframe_analysis_result': {
                'cross_timeframe_features': cross_timeframe_features,
                'analysis_metrics': analysis_metrics,
                'feature_interactions': feature_interactions,
                'analysis_summary': {
                    'total_timeframes_analyzed': len(self.config.timeframes),
                    'total_features_generated': len(cross_timeframe_features),
                    'significant_interactions': len(feature_interactions.get('significant_interactions', [])),
                    'analysis_time': analysis_result.get('analysis_time', 0.0),
                    'quality_score': validation_result.quality_score,
                    'validation_passed': validation_result.is_valid
                },
                'metadata': {
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    'execution_timestamp': datetime.now().isoformat(),
                    'config': {
                        'timeframes': self.config.timeframes,
                        'interaction_features': self.config.interaction_features,
                        'failure_mode': self.config.failure_mode.value
                    },
                    'validation_report': {
                        'is_valid': validation_result.is_valid,
                        'issues': validation_result.issues,
                        'recommendations': validation_result.recommendations
                    }
                }
            }
        }
    
    def _prepare_data_for_analysis(self, data: Any, feature_lookback_optimization: Dict[str, Any]) -> Any:
        """Prepare market data and feature optimization results for analysis."""
        if not PANDAS_AVAILABLE or not isinstance(data, pd.DataFrame):
            self.logger.warning("Pandas not available or data is not a DataFrame, using fallback")
            return {
                'market_data': data,
                'feature_lookback_optimization': feature_lookback_optimization
            }
        
        # Ensure we have required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            self.logger.warning(f"Missing columns for analysis: {missing_columns}")
            # Use available columns or create fallback data
            for col in missing_columns:
                if col == 'volume':
                    data[col] = 1000  # Default volume
                else:
                    data[col] = data.get('close', 100.0)  # Use close price as fallback
        
        return {
            'market_data': data,
            'feature_lookback_optimization': feature_lookback_optimization
        }