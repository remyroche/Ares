"""
Cross Timeframe Analysis Component.

This component performs cross timeframe interaction feature analysis with enhanced
error handling, comprehensive reporting, and silent failure prevention.
"""

import asyncio
import json
import logging
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from enum import Enum

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


class CrossTimeframeAnalysisComponent(BaseMarketAnalysisComponent):
    """
    Cross Timeframe Analysis Component.
    
    Performs cross timeframe interaction feature analysis with enhanced
    error handling, comprehensive reporting, and silent failure prevention.
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the cross timeframe analysis component."""
        super().__init__(config)
        self.logger = system_logger.getChild('CrossTimeframeAnalysis')
        
        # Enhanced configuration with error handling
        self.failure_mode = getattr(config, 'failure_mode', FailureMode.STRICT) if config else FailureMode.STRICT
        self.min_required_features = getattr(config, 'min_required_features', 10) if config else 10
        self.enable_detailed_reporting = getattr(config, 'enable_detailed_reporting', True) if config else True
        
        # Track errors and metrics
        self.errors = []
        self.metrics = {}
        self.start_time = None
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['cross_timeframe_analysis_result']
    
    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute cross timeframe analysis with enhanced error handling and reporting.
        
        Args:
            data: Market data for cross timeframe analysis
            pipeline_state: Current pipeline state
            
        Returns:
            ComponentResult with cross timeframe analysis results
        """
        self.start_time = time.time()
        self.logger.info('🌐 Starting Enhanced Cross Timeframe Analysis')
        self._report_checkpoint('start', 'analysis_started', {
            'symbol': self.config.symbol,
            'exchange': self.config.exchange,
            'timeframe': self.config.timeframe,
            'failure_mode': self.failure_mode.value
        })
        
        try:
            # Import cross timeframe analysis utilities
            from src.feature_generation.utils.cross_timeframe_features import CrossTimeframeFeatureGenerator, CrossTimeframeConfig
            
            # Get market data with enhanced validation
            market_data = await self._load_and_validate_market_data(data)
            self._report_checkpoint('data_loading', 'completed', {
                'data_points': len(market_data) if market_data is not None else 0,
                'data_quality_score': self._calculate_data_quality_score(market_data)
            })
            
            # Get feature optimization results from previous stage with validation
            feature_lookback_optimization = await self._get_feature_optimization_results(pipeline_state)
            self._report_checkpoint('feature_optimization', 'retrieved', {
                'optimization_available': bool(feature_lookback_optimization)
            })
            
            # Configure cross timeframe analysis
            analysis_config = CrossTimeframeConfig(
                target_timeframe=self.config.timeframe,
                analysis_timeframes=['1m', '5m', '15m', '1h', '4h', '1d'],
                feature_types=['price_momentum', 'volume_profile', 'volatility_regime', 'trend_alignment'],
                interaction_depth=2,  # 2nd order interactions
                correlation_threshold=0.7,
                significance_threshold=0.05,
                
                # Feature generation
                enable_momentum_features=True,
                enable_volume_features=True,
                enable_volatility_features=True,
                enable_trend_features=True,
                
                # Statistical analysis
                enable_correlation_analysis=True,
                enable_causality_analysis=True,
                enable_regime_analysis=True,
                
                # Feature selection
                enable_feature_selection=True,
                max_features_per_timeframe=10,
                feature_importance_threshold=0.01,
                
                # Hardware optimization
                enable_parallel_processing=True,
                enable_gpu_acceleration=True,
                memory_limit_gb=8.0
            )
            
            # Create cross timeframe feature generator
            feature_generator = CrossTimeframeFeatureGenerator(analysis_config)
            
            # Perform cross timeframe analysis with enhanced error handling
            analysis_result = await self._perform_cross_timeframe_analysis(
                feature_generator, market_data, feature_lookback_optimization, analysis_config
            )
            self._report_checkpoint('analysis', 'completed', {
                'features_generated': len(analysis_result.get('cross_timeframe_features', {})),
                'analysis_time': analysis_result.get('analysis_time', 0.0)
            })
            
            # Extract results
            cross_timeframe_features = analysis_result.get('cross_timeframe_features', {})
            analysis_metrics = analysis_result.get('analysis_metrics', {})
            feature_interactions = analysis_result.get('feature_interactions', {})
            
            # Validate that we have analysis results with enhanced validation
            validation_result = await self._validate_analysis_results(analysis_result)
            self._report_checkpoint('validation', 'completed', {
                'is_valid': validation_result['is_valid'],
                'quality_score': validation_result['quality_score'],
                'issues_count': len(validation_result['issues'])
            })
            
            if not validation_result['is_valid'] and self.failure_mode == FailureMode.STRICT:
                raise ValidationError(
                    error_type='validation',
                    message="Analysis results failed validation",
                    context={'issues': validation_result['issues']}
                )
            
            # Create single consolidated artifact with enhanced metadata
            artifacts = await self._create_enhanced_artifacts(
                analysis_result, validation_result, market_data, analysis_config
            )
            
            # Generate final report
            final_report = self._generate_final_report(artifacts, validation_result)
            self._report_checkpoint('completion', 'success', {
                'total_features': len(cross_timeframe_features),
                'quality_score': validation_result['quality_score'],
                'execution_time': time.time() - self.start_time
            })
            
            self.logger.info(f'✅ Enhanced Cross Timeframe Analysis completed: {len(cross_timeframe_features)} features generated')
            return ComponentResult(
                success=True,
                artifacts=artifacts,
                metadata={
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    'features_generated': len(cross_timeframe_features),
                    'data_quality_score': validation_result['quality_score'],
                    'errors': self.errors,
                    'validation_report': validation_result,
                    'final_report': final_report,
                    'execution_time': time.time() - self.start_time
                }
            )
            
        except Exception as e:
            # Track error with context
            error_context = {
                'type': 'execution_error',
                'message': str(e),
                'timestamp': time.time(),
                'traceback': traceback.format_exc(),
                'execution_time': time.time() - self.start_time if self.start_time else 0
            }
            self.errors.append(error_context)
            
            self.logger.error(f'❌ Enhanced Cross Timeframe Analysis failed: {e}')
            self.logger.error(f'❌ Error details: {traceback.format_exc()}')
            
            # Generate failure report
            failure_report = self._generate_failure_report(error_context)
            self._report_checkpoint('completion', 'failed', {
                'error_type': error_context['type'],
                'execution_time': error_context['execution_time']
            })
            
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=str(e),
                metadata={
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    'errors': self.errors,
                    'failure_report': failure_report,
                    'execution_time': error_context['execution_time']
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
    
    async def _perform_cross_timeframe_analysis(
        self, 
        feature_generator: Any, 
        market_data: Any, 
        feature_lookback_optimization: Dict[str, Any],
        config: Any
    ) -> Dict[str, Any]:
        """Perform the actual cross timeframe analysis process with enhanced error handling."""
        try:
            # Prepare data for analysis
            prepared_data = self._prepare_data_for_analysis(market_data, feature_lookback_optimization)
            
            # Perform cross timeframe analysis
            start_time = time.time()
            analysis_result = await feature_generator.generate_cross_timeframe_features(prepared_data, config)
            analysis_time = time.time() - start_time
            
            # Add timing information
            analysis_result['analysis_time'] = analysis_time
            
            # Track performance metrics
            self.metrics['analysis_time_seconds'] = analysis_time
            self.metrics['features_generated'] = len(analysis_result.get('cross_timeframe_features', {}))
            
            return analysis_result
            
        except Exception as e:
            error_context = {
                'type': 'feature_generation',
                'message': str(e),
                'timestamp': time.time(),
                'traceback': traceback.format_exc()
            }
            self.errors.append(error_context)
            
            if self.failure_mode == FailureMode.STRICT:
                raise FeatureGenerationError(
                    error_type='feature_generation',
                    message=f"Cross timeframe analysis failed: {e}",
                    context={'error': str(e), 'config': str(config)}
                )
            else:
                self.logger.error(f"Cross timeframe analysis failed, using fallback: {e}")
                
                # Return minimal fallback result with error tracking
                return {
                    'cross_timeframe_features': {},
                    'analysis_metrics': {
                        'analysis_method': 'fallback',
                        'error': str(e),
                        'error_context': error_context
                    },
                    'feature_interactions': {
                        'significant_interactions': [],
                        'correlation_matrix': {}
                    },
                    'analysis_time': 0.0
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
    
    async def _get_feature_optimization_results(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get feature optimization results with validation."""
        feature_lookback_optimization = pipeline_state.get('feature_lookback_optimization_result', {})
        
        if not feature_lookback_optimization:
            if self.failure_mode == FailureMode.STRICT:
                raise ValidationError(
                    error_type='validation',
                    message="No feature lookback optimization results available",
                    context={'pipeline_state_keys': list(pipeline_state.keys())}
                )
            else:
                self.logger.warning("No feature lookback optimization results available, continuing without them")
                return {}
        
        return feature_lookback_optimization
    
    async def _validate_analysis_results(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate analysis results to ensure quality."""
        issues = []
        recommendations = []
        
        # Check required artifacts
        if not analysis_result.get('cross_timeframe_features'):
            issues.append("Missing cross_timeframe_features in analysis result")
        
        # Check feature quality
        features = analysis_result.get('cross_timeframe_features', {})
        if len(features) < self.min_required_features:
            issues.append(f"Insufficient features: {len(features)} < {self.min_required_features}")
            recommendations.append("Increase feature generation or reduce min_required_features threshold")
        
        # Check for critical errors
        analysis_metrics = analysis_result.get('analysis_metrics', {})
        if analysis_metrics.get('analysis_method') == 'fallback':
            issues.append("Analysis used fallback method due to errors")
            recommendations.append("Review and address errors in the analysis pipeline")
        
        quality_score = self._calculate_quality_score(analysis_result)
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'quality_score': quality_score,
            'recommendations': recommendations
        }
    
    def _calculate_quality_score(self, analysis_result: Dict[str, Any]) -> float:
        """Calculate overall quality score."""
        score = 1.0
        
        # Penalize for missing features
        features = analysis_result.get('cross_timeframe_features', {})
        if len(features) < self.min_required_features:
            score *= 0.5
        
        # Penalize for fallback analysis
        analysis_metrics = analysis_result.get('analysis_metrics', {})
        if analysis_metrics.get('analysis_method') == 'fallback':
            score *= 0.3
        
        # Penalize for errors
        if self.errors:
            score *= max(0.1, 1.0 - (len(self.errors) * 0.1))
        
        return score
    
    def _calculate_data_quality_score(self, data: Any) -> float:
        """Calculate data quality score."""
        if not PANDAS_AVAILABLE or not isinstance(data, pd.DataFrame):
            return 0.0
        
        score = 1.0
        
        # Check for missing values
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col in data.columns:
                nan_ratio = data[col].isna().sum() / len(data)
                score *= (1.0 - nan_ratio)
        
        # Check for data types
        for col in required_columns:
            if col in data.columns and data[col].dtype == 'object':
                score *= 0.5
        
        return max(0.0, score)
    
    async def _create_enhanced_artifacts(
        self, 
        analysis_result: Dict[str, Any], 
        validation_result: Dict[str, Any], 
        market_data: Any, 
        analysis_config: Any
    ) -> Dict[str, Any]:
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
                    'total_timeframes_analyzed': len(analysis_config.analysis_timeframes),
                    'total_features_generated': len(cross_timeframe_features),
                    'significant_interactions': len(feature_interactions.get('significant_interactions', [])),
                    'analysis_time': analysis_result.get('analysis_time', 0.0),
                    'quality_score': validation_result['quality_score'],
                    'validation_passed': validation_result['is_valid']
                },
                'metadata': {
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    'data_points': len(market_data) if market_data is not None else 0,
                    'execution_timestamp': datetime.now().isoformat(),
                    'config': {
                        'timeframes': analysis_config.analysis_timeframes,
                        'interaction_features': analysis_config.feature_types,
                        'failure_mode': self.failure_mode.value
                    },
                    'validation_report': validation_result,
                    'performance_metrics': self.metrics,
                    'errors': self.errors
                }
            }
        }
    
    def _generate_final_report(self, artifacts: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        return {
            'execution_summary': {
                'total_time': time.time() - self.start_time if self.start_time else 0,
                'success': True,
                'features_generated': len(artifacts.get('cross_timeframe_analysis_result', {}).get('cross_timeframe_features', {})),
                'data_quality_score': validation_result['quality_score'],
                'error_count': len(self.errors)
            },
            'performance_metrics': self.metrics,
            'quality_metrics': {
                'data_quality_score': validation_result['quality_score'],
                'feature_completeness': len(artifacts.get('cross_timeframe_analysis_result', {}).get('cross_timeframe_features', {})) / self.min_required_features,
                'error_rate': len(self.errors) / max(1, 5)  # Assuming 5 main steps
            },
            'recommendations': validation_result.get('recommendations', [])
        }
    
    def _generate_failure_report(self, error_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate failure report."""
        return {
            'execution_summary': {
                'total_time': error_context.get('execution_time', 0),
                'success': False,
                'features_generated': 0,
                'data_quality_score': 0.0,
                'error_count': len(self.errors)
            },
            'error_details': error_context,
            'recommendations': [
                "Review error logs for detailed failure information",
                "Check data quality and availability",
                "Verify configuration parameters",
                "Consider using graceful failure mode for partial results"
            ]
        }
    
    def _report_checkpoint(self, step: str, status: str, details: Dict[str, Any]):
        """Report progress at key checkpoints."""
        if self.enable_detailed_reporting:
            self.logger.info(f"📊 [{step}] {status} - {details}")
        
        # Track metrics
        self.metrics[f'{step}_status'] = status
        self.metrics[f'{step}_timestamp'] = time.time()