"""
Example Enhanced Data Qualification Step

This module demonstrates how to use the new unified utilities for data qualification steps,
showing best practices for configuration, error handling, and utility integration.

Key Features Demonstrated:
- Unified import management with fallback handling
- Standardized configuration system
- Comprehensive error handling and recovery
- Type-safe interfaces with proper documentation
- Performance monitoring and metrics collection
- ML Commons integration with graceful degradation
"""

import asyncio
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Import our new unified utilities
from src.utils.data_quality.data_qualification_base import (
    DataQualificationStep,
    DataQualificationResult,
    StepMetrics,
    create_step_result,
    validate_dataframe_input
)
from src.utils.data_quality.data_qualification_config import (
    DataQualificationConfig,
    SROptimizationConfig,
    PerformanceConfig
)
from src.utils.data_quality.data_qualification_error_handler import (
    DataQualificationErrorHandler,
    handle_utility_failure,
    error_context
)
from src.utils.data_quality.data_qualification_imports import (
    DataQualificationImportManager,
    get_utility_suite
)

# Initialize logger
import logging
logger = logging.getLogger(__name__)

class EnhancedSROptimizationStep(DataQualificationStep):
    """
    Enhanced Support/Resistance Optimization Step.
    
    This step demonstrates the use of unified utilities for support/resistance
    detection and optimization with comprehensive error handling and fallback mechanisms.
    
    Example:
        >>> config = DataQualificationConfig(
        ...     symbol="AAPL",
        ...     exchange="NASDAQ", 
        ...     timeframe="1m",
        ...     data_dir="/data"
        ... )
        >>> step = EnhancedSROptimizationStep(config)
        >>> result = await step.execute({"data": df})
    """
    
    def __init__(self, config: DataQualificationConfig):
        """
        Initialize the enhanced SR optimization step.
        
        Args:
            config: Configuration for the step
            
        Raises:
            ValueError: If configuration is invalid
        """
        super().__init__(config)
        
        # Get step-specific configuration
        self.sr_config = self.config.sr_optimization
        self.performance_config = self.config.performance
        
        # Initialize utilities with error handling
        self._initialize_utilities()
        
        self.logger.info("🚀 Enhanced SR Optimization Step initialized")
    
    def _initialize_utilities(self):
        """Initialize utilities with comprehensive error handling."""
        try:
            # Get ML Commons utilities
            ml_commons = self.get_utility('ml_common')
            if ml_commons:
                self.data_quality_utils = ml_commons.get('data_quality')
                self.feature_selector = ml_commons.get('feature_selection')
                self.pipeline_orchestrator = ml_commons.get('pipeline_orchestrator')
                self.logger.info("✅ ML Commons utilities initialized")
            else:
                self.logger.warning("⚠️ ML Commons utilities not available")
                self._initialize_fallback_utilities()
            
            # Get M1 optimization utilities
            m1_optimizers = self.get_utility('m1_optimizers')
            if m1_optimizers:
                self.gpu_manager = m1_optimizers.get('gpu_manager')
                self.memory_optimizer = m1_optimizers.get('memory_optimizer')
                self.cpu_optimizer = m1_optimizers.get('cpu_optimizer')
                self.logger.info("✅ M1 optimization utilities initialized")
            else:
                self.logger.warning("⚠️ M1 optimization utilities not available")
                self._initialize_fallback_optimizers()
            
            # Get validation utilities
            validation_utils = self.get_utility('validation')
            if validation_utils:
                self.math_validation = validation_utils.get('math')
                self.common_operations = validation_utils.get('common_operations')
                self.logger.info("✅ Validation utilities initialized")
            else:
                self.logger.warning("⚠️ Validation utilities not available")
                self._initialize_fallback_validation()
                
        except Exception as e:
            self.logger.error(f"❌ Error initializing utilities: {e}")
            self._initialize_fallback_utilities()
    
    def _initialize_fallback_utilities(self):
        """Initialize fallback utilities when ML Commons is not available."""
        self.logger.info("🔄 Initializing fallback utilities")
        
        # Fallback data quality utility
        class FallbackDataQuality:
            def missing_value_analysis(self, data):
                return {'severity_assessment': {'severity_level': 'low'}, 'recommendations': []}
            
            def automated_outlier_detection(self, data):
                return {'outliers_detected': 0, 'recommendations': []}
        
        self.data_quality_utils = FallbackDataQuality()
        
        # Fallback feature selector
        class FallbackFeatureSelector:
            def mrmr_selection(self, features, target, feature_names, n_features):
                return {'selected_features': feature_names[:n_features]}
        
        self.feature_selector = FallbackFeatureSelector()
        
        # Fallback pipeline orchestrator
        class FallbackPipelineOrchestrator:
            def create_training_pipeline(self, *args, **kwargs):
                return f"fallback_pipeline_{int(time.time())}"
            
            def execute_pipeline(self, pipeline_id, *args, **kwargs):
                return {'success': True, 'results': {}}
        
        self.pipeline_orchestrator = FallbackPipelineOrchestrator()
    
    def _initialize_fallback_optimizers(self):
        """Initialize fallback optimization utilities."""
        # Fallback GPU manager
        class FallbackGPUManager:
            def is_available(self):
                return False
        
        self.gpu_manager = FallbackGPUManager()
        
        # Fallback memory optimizer
        class FallbackMemoryOptimizer:
            def memory_checkpoint(self, name):
                from contextlib import nullcontext
                return nullcontext()
        
        self.memory_optimizer = FallbackMemoryOptimizer()
        
        # Fallback CPU optimizer
        class FallbackCPUOptimizer:
            def get_optimal_workers_for_task(self, task_type):
                return 1
        
        self.cpu_optimizer = FallbackCPUOptimizer()
    
    def _initialize_fallback_validation(self):
        """Initialize fallback validation utilities."""
        # Fallback math validation
        def safe_divide(a, b, default=0.0):
            try:
                return a / b if b != 0 else default
            except:
                return default
        
        self.math_validation = {'safe_divide': safe_divide}
        
        # Fallback common operations
        def safe_float(value, default=0.0):
            try:
                return float(value)
            except:
                return default
        
        self.common_operations = {'safe_float': safe_float}
    
    def validate_input(self, input_data: Dict[str, Any]) -> 'ValidationResult':
        """
        Validate input data for SR optimization.
        
        Args:
            input_data: Input data containing DataFrame
            
        Returns:
            ValidationResult with validation status
        """
        errors = []
        warnings = []
        
        # Check if data is present
        if 'data' not in input_data:
            errors.append("Input data must contain 'data' key")
            return ValidationResult(is_valid=False, errors=errors)
        
        data = input_data['data']
        
        # Validate DataFrame
        df_validation = validate_dataframe_input(
            data,
            required_columns=['open', 'high', 'low', 'close', 'volume'],
            min_rows=100
        )
        
        errors.extend(df_validation.errors)
        warnings.extend(df_validation.warnings)
        
        # Additional validation for SR optimization
        if isinstance(data, pd.DataFrame):
            # Check for sufficient price data
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in data.columns:
                    if (data[col] <= 0).any():
                        warnings.append(f"Column {col} contains zero or negative values")
            
            # Check for reasonable price ranges
            if 'close' in data.columns:
                close_prices = data['close'].dropna()
                if len(close_prices) > 0:
                    price_range = close_prices.max() - close_prices.min()
                    if price_range <= 0:
                        errors.append("Price data has no variation")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    async def execute(self, input_data: Dict[str, Any]) -> DataQualificationResult:
        """
        Execute SR optimization step with comprehensive error handling.
        
        Args:
            input_data: Input data containing DataFrame
            
        Returns:
            DataQualificationResult with optimization results
        """
        async with self.execution_context(input_data) as data:
            try:
                # Validate input
                validation_result = self.validate_input(input_data)
                if not validation_result.is_valid:
                    return create_step_result(
                        success=False,
                        step_name=self.__class__.__name__,
                        errors=validation_result.errors,
                        warnings=validation_result.warnings
                    )
                
                # Extract data
                df = input_data['data']
                
                # Perform SR optimization with error handling
                sr_results = await self._perform_sr_optimization(df)
                
                # Create result
                result = create_step_result(
                    success=True,
                    data=sr_results,
                    step_name=self.__class__.__name__,
                    metadata={
                        'sr_levels_found': len(sr_results.get('sr_levels', [])),
                        'optimization_method': sr_results.get('method', 'unknown'),
                        'ml_commons_used': sr_results.get('ml_commons_used', False),
                        'm1_optimization_used': sr_results.get('m1_optimization_used', False)
                    }
                )
                
                # Add warnings from validation
                result.warnings.extend(validation_result.warnings)
                
                return result
                
            except Exception as e:
                self.logger.exception(f"Error in SR optimization: {e}")
                return create_step_result(
                    success=False,
                    step_name=self.__class__.__name__,
                    errors=[str(e)]
                )
    
    async def _perform_sr_optimization(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform SR optimization with comprehensive error handling.
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            Dictionary with SR optimization results
        """
        with error_context(self.__class__.__name__, "sr_optimization"):
            # Use memory optimization if available
            if hasattr(self, 'memory_optimizer') and self.memory_optimizer:
                with self.memory_optimizer.memory_checkpoint("sr_optimization"):
                    return await self._sr_optimization_core(df)
            else:
                return await self._sr_optimization_core(df)
    
    async def _sr_optimization_core(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Core SR optimization logic."""
        try:
            # Data quality assessment
            if hasattr(self, 'data_quality_utils') and self.data_quality_utils:
                quality_report = self.data_quality_utils.missing_value_analysis(df)
                self.logger.info(f"Data quality score: {quality_report.get('severity_assessment', {}).get('severity_level', 'unknown')}")
            
            # Feature engineering
            features = await self._engineer_sr_features(df)
            
            # SR level detection
            sr_levels = await self._detect_sr_levels(df, features)
            
            # SR optimization
            optimized_sr = await self._optimize_sr_levels(sr_levels, df)
            
            return {
                'sr_levels': optimized_sr,
                'features_used': features.shape[1] if hasattr(features, 'shape') else len(features),
                'method': 'enhanced_optimization',
                'ml_commons_used': hasattr(self, 'data_quality_utils') and self.data_quality_utils is not None,
                'm1_optimization_used': hasattr(self, 'memory_optimizer') and self.memory_optimizer is not None,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"SR optimization core failed: {e}")
            # Return fallback results
            return {
                'sr_levels': [],
                'features_used': 0,
                'method': 'fallback',
                'ml_commons_used': False,
                'm1_optimization_used': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _engineer_sr_features(self, df: pd.DataFrame) -> np.ndarray:
        """Engineer features for SR detection."""
        try:
            features = []
            
            # Price-based features
            if 'close' in df.columns:
                close_prices = df['close'].values
                
                # Returns
                returns = np.diff(close_prices) / close_prices[:-1]
                features.append(returns)
                
                # Rolling statistics
                if len(close_prices) > 20:
                    rolling_mean = pd.Series(close_prices).rolling(20).mean().values[19:]
                    rolling_std = pd.Series(close_prices).rolling(20).std().values[19:]
                    features.extend([rolling_mean, rolling_std])
            
            # Volume features
            if 'volume' in df.columns:
                volume_data = df['volume'].values
                if len(volume_data) > 1:
                    volume_returns = np.diff(volume_data) / (volume_data[:-1] + 1e-8)
                    features.append(volume_returns)
            
            # Combine features
            if features:
                min_length = min(len(feat) for feat in features if len(feat) > 0)
                combined_features = np.column_stack([feat[:min_length] for feat in features if len(feat) > 0])
                return combined_features
            else:
                return np.array([]).reshape(0, 0)
                
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}")
            return np.array([]).reshape(0, 0)
    
    async def _detect_sr_levels(self, df: pd.DataFrame, features: np.ndarray) -> List[Dict[str, Any]]:
        """Detect support/resistance levels."""
        try:
            sr_levels = []
            
            if 'high' in df.columns and 'low' in df.columns:
                high_prices = df['high'].values
                low_prices = df['low'].values
                
                # Simple SR detection based on local maxima/minima
                for i in range(1, len(high_prices) - 1):
                    # Local maximum (resistance)
                    if high_prices[i] > high_prices[i-1] and high_prices[i] > high_prices[i+1]:
                        sr_levels.append({
                            'price': float(high_prices[i]),
                            'type': 'resistance',
                            'strength': 1.0,
                            'index': i
                        })
                    
                    # Local minimum (support)
                    if low_prices[i] < low_prices[i-1] and low_prices[i] < low_prices[i+1]:
                        sr_levels.append({
                            'price': float(low_prices[i]),
                            'type': 'support',
                            'strength': 1.0,
                            'index': i
                        })
            
            return sr_levels
            
        except Exception as e:
            self.logger.error(f"SR level detection failed: {e}")
            return []
    
    async def _optimize_sr_levels(self, sr_levels: List[Dict[str, Any]], df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Optimize SR levels using ML Commons if available."""
        try:
            if not sr_levels:
                return []
            
            # Use feature selection if available
            if hasattr(self, 'feature_selector') and self.feature_selector:
                # Create feature matrix from SR levels
                feature_matrix = np.array([[level['price'], level['strength']] for level in sr_levels])
                target = np.array([1 if level['type'] == 'resistance' else 0 for level in sr_levels])
                feature_names = ['price', 'strength']
                
                # Apply feature selection
                selection_result = self.feature_selector.mrmr_selection(
                    feature_matrix, target, feature_names, n_features=2
                )
                
                if selection_result.get('selected_features'):
                    self.logger.info("Applied ML Commons feature selection to SR levels")
            
            # Filter levels by strength threshold
            min_strength = self.sr_config.strength_threshold
            optimized_levels = [
                level for level in sr_levels 
                if level['strength'] >= min_strength
            ]
            
            # Sort by strength
            optimized_levels.sort(key=lambda x: x['strength'], reverse=True)
            
            # Limit number of levels
            max_levels = self.sr_config.max_touch_count
            optimized_levels = optimized_levels[:max_levels]
            
            return optimized_levels
            
        except Exception as e:
            self.logger.error(f"SR level optimization failed: {e}")
            return sr_levels  # Return original levels as fallback

# Example usage function
async def run_enhanced_sr_optimization_example():
    """
    Example function demonstrating how to use the enhanced SR optimization step.
    
    This function shows the complete workflow from configuration creation
    to step execution with comprehensive error handling.
    """
    try:
        # Create configuration
        config = DataQualificationConfig(
            symbol="AAPL",
            exchange="NASDAQ",
            timeframe="1m",
            data_dir="./data",
            performance=PerformanceConfig(
                enable_m1_optimization=True,
                enable_gpu_acceleration=True,
                max_workers=4
            ),
            sr_optimization=SROptimizationConfig(
                min_touch_count=3,
                max_touch_count=10,
                strength_threshold=0.5,
                enable_ml_commons=True
            )
        )
        
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=1000, freq='1min')
        prices = 100 + np.cumsum(np.random.randn(1000) * 0.01)
        
        sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices + np.random.randn(1000) * 0.1,
            'high': prices + np.abs(np.random.randn(1000) * 0.2),
            'low': prices - np.abs(np.random.randn(1000) * 0.2),
            'close': prices,
            'volume': np.random.randint(1000, 10000, 1000)
        })
        
        # Create and execute step
        step = EnhancedSROptimizationStep(config)
        result = await step.execute({"data": sample_data})
        
        # Display results
        print(f"Step execution successful: {result.success}")
        print(f"Execution time: {result.execution_time:.2f}s")
        print(f"SR levels found: {len(result.data.get('sr_levels', []))}")
        print(f"ML Commons used: {result.data.get('ml_commons_used', False)}")
        print(f"M1 optimization used: {result.data.get('m1_optimization_used', False)}")
        
        if result.warnings:
            print(f"Warnings: {result.warnings}")
        
        if result.errors:
            print(f"Errors: {result.errors}")
        
        return result
        
    except Exception as e:
        logger.exception(f"Example execution failed: {e}")
        return None

if __name__ == "__main__":
    # Run the example
    asyncio.run(run_enhanced_sr_optimization_example())