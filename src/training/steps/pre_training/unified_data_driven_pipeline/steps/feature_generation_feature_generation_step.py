"""
Enhanced Feature Generation Step

This step generates features using the AutoOptimizedFeatureGenerator
with VectorBT optimization and comprehensive feature categories.
"""

from __future__ import annotations

import logging
import json
import warnings
import pandas as pd
import numpy as np
import copy
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

from src.training.steps.pre_training.components.base_component import (
    BasePreTrainingComponent, ComponentConfig, ComponentResult
)
from src.utils.common_operations import safe_dataframe_operation
from src.utils.matrix_operations import safe_matrix_multiply, optimize_dataframe

# Import advanced feature generation components
try:
    from src.feature_generation.core.auto_optimized_feature_generator import (
        AutoOptimizedFeatureGenerator
    )
    from src.feature_generation.core.feature_generator import (
        FeatureGenerator, FeatureConfig, FeatureCategory
    )
    from src.feature_generation.core.auto_optimization_config import (
        AutoOptimizationConfig, OptimizationLevel
    )
    from src.feature_generation.core.vectorbt_feature_generator import (
        VectorBTFeatureGenerator
    )
    from src.feature_generation.core.feature_bank import FeatureBank
    FEATURE_GENERATION_AVAILABLE = True
except ImportError:
    FEATURE_GENERATION_AVAILABLE = False
    AutoOptimizedFeatureGenerator = None
    FeatureGenerator = None
    FeatureConfig = None
    FeatureCategory = None
    AutoOptimizationConfig = None
    OptimizationLevel = None
    VectorBTFeatureGenerator = None
    FeatureBank = None

@dataclass
class FeatureGenerationResult:
    """Enhanced result of feature generation step."""

    success: bool
    generated_features: pd.DataFrame
    feature_metadata: Dict[str, Any]
    generation_metrics: Dict[str, Any]
    optimization_stats: Dict[str, Any]
    feature_categories: List[str]
    vectorbt_optimizations: Dict[str, Any]
    artifacts: Dict[str, Any]
    error_message: Optional[str] = None

class FeatureGenerationStep(BasePreTrainingComponent):
    """Enhanced feature generation step using AutoOptimizedFeatureGenerator."""

    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the enhanced feature generation step."""
        super().__init__(config or ComponentConfig())
        self.logger = logging.getLogger(__name__)
        
        # Initialize feature generation components
        if FEATURE_GENERATION_AVAILABLE:
            # Create feature configuration with default values
            self.feature_config = FeatureConfig(
                name="enhanced_features",
                category=FeatureCategory.VOLATILITY,  # Default category
                description="Enhanced feature generation with VectorBT optimization",
                required_columns=["open", "high", "low", "close", "volume"],
                optional_columns=["timestamp"],
                default_lookback=20,
                min_lookback=1,
                max_lookback=252,
                use_vectorbt=True,
                enable_gpu=True,
                enable_parallel=True
            )
            
            # Create auto-optimization configuration
            self.auto_optimization_config = AutoOptimizationConfig(
                optimization_level=OptimizationLevel.BALANCED,
                enable_auto_optimization=False,  # Disable auto-optimization to prevent data corruption
                enable_vectorbt_optimization=False,  # Disable VectorBT optimization to prevent data corruption
                enable_memory_optimization=False  # Disable memory optimization to prevent data corruption
            )
            
            # Initialize feature generators
            self.auto_optimized_generator = AutoOptimizedFeatureGenerator(
                self.feature_config, 
                self.auto_optimization_config
            )
            
        else:
            self.logger.warning("Advanced feature generation components not available, using fallback")
            self.auto_optimized_generator = None

    async def execute(self,
                     data: pd.DataFrame,
                     symbol: str = "ETHUSDT",
                     timeframe: str = "15m",
                     direction: str = "longs",
                     custom_overrides: Optional[Dict[str, Any]] = None) -> FeatureGenerationResult:
        """Execute enhanced feature generation step using AutoOptimizedFeatureGenerator."""

        self.logger.info("Starting enhanced feature generation step with auto-optimization")

        try:
            # DEBUG: Check data quality at the start of execute
            import numpy as np
            print(f"🔍 [DEBUG] Execute method - Data shape: {data.shape}")
            print(f"🔍 [DEBUG] Execute method - Non-finite values: {(~np.isfinite(data.select_dtypes(include=[np.number])).values).sum()}")
            for col in data.select_dtypes(include=[np.number]).columns:
                non_finite = (~np.isfinite(data[col])).sum()
                if non_finite > 0:
                    print(f"🔍 [DEBUG] Execute method - {col}: {non_finite} non-finite values")

            # Update feature configuration with actual values
            if FEATURE_GENERATION_AVAILABLE:
                self.feature_config.symbol = symbol
                self.feature_config.timeframe = timeframe

            # Validate input data
            if data is None or data.empty:
                raise ValueError("Input data is None or empty")
            
            required_columns = ['close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}. Available columns: {list(data.columns)}")
            if not FEATURE_GENERATION_AVAILABLE:
                # Fallback to basic feature generation
                return await self._fallback_feature_generation(
                    data, symbol, timeframe, direction, custom_overrides
                )

            # Perform comprehensive feature generation
            generation_result = await self._perform_enhanced_feature_generation(
                data, symbol, timeframe, direction, custom_overrides
            )

            if generation_result.success:
                self.logger.info(f"Enhanced feature generation completed successfully")
                self.logger.info(f"Generated {len(generation_result.generated_features.columns)} features")
                self.logger.info(f"Categories: {', '.join(generation_result.feature_categories)}")
                self.logger.info(f"Optimization stats: {generation_result.optimization_stats}")
            else:
                self.logger.error(f"Feature generation failed: {generation_result.error_message}")

            return generation_result

        except Exception as e:
            self.logger.error(f"Enhanced feature generation step failed with exception: {e}")
            return FeatureGenerationResult(
                success=False,
                generated_features=pd.DataFrame(),
                feature_metadata={},
                generation_metrics={},
                optimization_stats={},
                feature_categories=[],
                vectorbt_optimizations={},
                artifacts={},
                error_message=str(e)
            )

    async def _perform_enhanced_feature_generation(self, data: pd.DataFrame, symbol: str,
                                                   timeframe: str, direction: str,
                                                   custom_overrides: Optional[Dict[str, Any]]) -> FeatureGenerationResult:
        """Perform enhanced feature generation using AutoOptimizedFeatureGenerator."""
        
        try:
            # Create a fresh copy of the configuration to avoid race conditions
            feature_config = copy.deepcopy(self.feature_config)
            
            # Update configuration with custom overrides
            if custom_overrides:
                feature_config.update_from_dict(custom_overrides)
            
            # Generate features using auto-optimized generator with fresh config
            feature_result = self.auto_optimized_generator.generate(
                data=data,
                symbol=symbol,
                timeframe=timeframe,
                direction=direction,
                config=feature_config
            )
            
            # Get optimization statistics with null check
            optimization_stats = self.auto_optimized_generator.get_optimization_stats()
            if optimization_stats is None:
                optimization_stats = {'status': 'unavailable', 'message': 'Optimization stats not available'}
            
            # Get feature categories used
            feature_categories = [cat.value for cat in feature_config.categories]
            
            # Get VectorBT optimization details
            vectorbt_optimizations = {
                'vectorbt_enabled': feature_config.enable_vectorbt,
                'optimization_level': self.auto_optimization_config.level.value,
                'parallel_processing': self.auto_optimization_config.enable_parallel_processing,
                'memory_optimization': self.auto_optimization_config.enable_memory_optimization
            }
            
            # Compile comprehensive result
            return FeatureGenerationResult(
                success=feature_result.success,
                generated_features=feature_result.features,
                feature_metadata=feature_result.feature_metadata,
                generation_metrics=feature_result.metrics,
                optimization_stats=optimization_stats,
                feature_categories=feature_categories,
                vectorbt_optimizations=vectorbt_optimizations,
                artifacts={
                    'feature_result': feature_result.__dict__,
                    'optimization_stats': optimization_stats,
                    'config': self._serialize_config(feature_config),
                    'auto_optimization_config': self._serialize_config(self.auto_optimization_config)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Enhanced feature generation failed: {e}")
            return FeatureGenerationResult(
                success=False,
                generated_features=pd.DataFrame(),
                feature_metadata={},
                generation_metrics={},
                optimization_stats={},
                feature_categories=[],
                vectorbt_optimizations={},
                artifacts={},
                error_message=str(e)
            )

    async def _fallback_feature_generation(self, data: pd.DataFrame, symbol: str,
                                          timeframe: str, direction: str,
                                          custom_overrides: Optional[Dict[str, Any]]) -> FeatureGenerationResult:
        """Fallback feature generation when advanced components are not available."""
        
        try:
            # Basic feature generation
            basic_features = pd.DataFrame(index=data.index)
            
            # Simple technical indicators with proper min_periods
            basic_features['sma_20'] = data['close'].rolling(20, min_periods=20).mean()
            basic_features['sma_50'] = data['close'].rolling(50, min_periods=50).mean()
            basic_features['rsi_14'] = self._calculate_rsi(data['close'], 14)
            basic_features['bb_upper'] = data['close'].rolling(20, min_periods=20).mean() + 2 * data['close'].rolling(20, min_periods=20).std()
            basic_features['bb_lower'] = data['close'].rolling(20, min_periods=20).mean() - 2 * data['close'].rolling(20, min_periods=20).std()
            basic_features['volume_sma'] = data['volume'].rolling(20, min_periods=20).mean()
            
            # Remove NaN values
            basic_features = basic_features.dropna()
            
            return FeatureGenerationResult(
                success=True,
                generated_features=basic_features,
                feature_metadata={'method': 'fallback_basic', 'symbol': symbol, 'timeframe': timeframe},
                generation_metrics={'feature_count': len(basic_features.columns)},
                optimization_stats={'method': 'fallback'},
                feature_categories=['basic_technical'],
                vectorbt_optimizations={'vectorbt_enabled': False},
                artifacts={'fallback_features': basic_features.columns.tolist()}
            )
            
        except Exception as e:
            return FeatureGenerationResult(
                success=False,
                generated_features=pd.DataFrame(),
                feature_metadata={},
                generation_metrics={},
                optimization_stats={},
                feature_categories=[],
                vectorbt_optimizations={},
                artifacts={},
                error_message=str(e)
            )

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator using Wilder's smoothing to avoid division by zero."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Use Wilder's smoothing (exponential weighted mean)
        alpha = 1.0 / period
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
        
        # Guard against division by zero
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Neutral RSI when no data

    # Required utility methods for BasePreTrainingComponent
    def safe_dataframe_operation(self, operation_func, *args, **kwargs):
        """Safe dataframe operation wrapper."""
        return safe_dataframe_operation(operation_func, *args, **kwargs)

    def safe_matrix_multiply(self, a, b):
        """Safe matrix multiplication."""
        return safe_matrix_multiply(a, b)

    def optimize_dataframe_for_matrix_ops(self, df):
        """Optimize dataframe for matrix operations."""
        return optimize_dataframe(df)

    def _serialize_config(self, config) -> Dict[str, Any]:
        """Serialize configuration object to plain types for JSON serialization."""
        serialized = {}
        for key, value in config.__dict__.items():
            if hasattr(value, 'value'):  # Enum
                serialized[key] = value.value
            elif hasattr(value, '__dict__'):  # Object with __dict__
                serialized[key] = self._serialize_config(value)
            elif isinstance(value, (list, tuple)):
                serialized[key] = [
                    item.value if hasattr(item, 'value') else item 
                    for item in value
                ]
            else:
                serialized[key] = value
        return serialized

    def process(self, data: Any) -> Any:
        """Process the input data and return the result."""
        # This method is required by the abstract base class
        # The actual processing is done in the execute method
        return data

    def validate(self, data: Any) -> bool:
        """Validate the input data."""
        # This method is required by the abstract base class
        # Basic validation - check if data is not None and has required columns
        if data is None:
            return False
        if hasattr(data, 'empty') and data.empty:
            return False
        return True

# Command handler for ares_launcher integration
async def handle_feature_generation_step(
    data: Optional[pd.DataFrame] = None,
    symbol: str = "ETHUSDT",
    timeframe: str = "15m",
    direction: str = "longs",
    custom_overrides: Optional[Dict[str, Any]] = None,
    **kwargs
) -> FeatureGenerationResult:
    """
    Handle enhanced feature generation step command.

    Args:
        data: Input DataFrame with OHLCV data (optional, will generate sample if None)
        symbol: Trading symbol (default: "ETHUSDT")
        timeframe: Timeframe (default: "15m")
        direction: Direction (default: "longs")
        custom_overrides: Custom configuration overrides (optional)
        **kwargs: Additional arguments

    Returns:
        Enhanced FeatureGenerationResult with comprehensive generation results
    """
    # Create sample data for feature generation (only for testing/demo)
    # In production, this should be replaced with actual data loading
    if data is None:
        sample_data = pd.DataFrame({
            'open': np.random.randn(1000).cumsum() + 100,
            'high': np.random.randn(1000).cumsum() + 105,
            'low': np.random.randn(1000).cumsum() + 95,
            'close': np.random.randn(1000).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 1000)
        })
    else:
        sample_data = data

    # Create enhanced step instance and execute
    step = FeatureGenerationStep()

    return await step.execute(
        data=sample_data,
        symbol=symbol,
        timeframe=timeframe,
        direction=direction,
        custom_overrides=custom_overrides
    )

# Register component with factory
def _register_feature_generation_step():
    """Register the FeatureGenerationStep component with the factory."""
    try:
        from src.training.steps.pre_training.components import ComponentFactory
        ComponentFactory.register_component(
            'feature_generation_step',
            FeatureGenerationStep
        )
    except ImportError:
        # Component factory not available, skip registration
        pass

# Register the component when module is imported
_register_feature_generation_step()
