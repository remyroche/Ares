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
import asyncio
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
                use_vectorbt=False,  # Align with auto-optimization config
                enable_gpu=False,  # Align with auto-optimization config
                enable_parallel=False  # Align with auto-optimization config
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
            self.logger.debug("Execute - data shape: %s", data.shape)
            numeric = data.select_dtypes(include=[np.number])
            non_finite_total = (~np.isfinite(numeric)).to_numpy().sum()
            self.logger.debug("Execute - non-finite total: %d", non_finite_total)
            for col in numeric.columns:
                nf = (~np.isfinite(numeric[col])).sum()
                if nf:
                    self.logger.debug("Execute - %s: %d non-finite", col, nf)

            # Clone feature configuration to avoid mutating shared config
            if FEATURE_GENERATION_AVAILABLE:
                base_cfg = copy.deepcopy(self.feature_config)
                base_cfg.symbol = symbol
                base_cfg.timeframe = timeframe
            else:
                base_cfg = None

            # Validate input data
            if data is None or len(data) == 0:
                raise ValueError("Input data is None or empty")
            
            # Use proper validation that matches FeatureConfig requirements
            required_columns = getattr(self.feature_config, 'required_columns', ['open', 'high', 'low', 'close', 'volume'])
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}. Available: {list(data.columns)}")
            if not FEATURE_GENERATION_AVAILABLE:
                # Fallback to basic feature generation
                return await self._fallback_feature_generation(
                    data, symbol, timeframe, direction, custom_overrides
                )

            # Perform comprehensive feature generation
            generation_result = await self._perform_enhanced_feature_generation(
                data, symbol, timeframe, direction, custom_overrides, base_cfg
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
                                                   custom_overrides: Optional[Dict[str, Any]],
                                                   base_config: Optional[FeatureConfig] = None) -> FeatureGenerationResult:
        """Perform enhanced feature generation using AutoOptimizedFeatureGenerator."""
        
        try:
            # Use the provided base config or create a fresh copy
            if base_config is not None:
                feature_config = copy.deepcopy(base_config)
            else:
                feature_config = copy.deepcopy(self.feature_config)
            
            # Update configuration with custom overrides
            if custom_overrides:
                feature_config.update_from_dict(custom_overrides)
                # Sanity checks after overrides
                if not getattr(feature_config, 'required_columns', None):
                    raise ValueError("feature_config.required_columns cannot be empty after overrides.")
            
            # Generate features using auto-optimized generator with fresh config
            # Handle potential async generate method
            res = self.auto_optimized_generator.generate(
                data=data,
                symbol=symbol,
                timeframe=timeframe,
                direction=direction,
                config=feature_config
            )
            
            # Check if result is a coroutine (async method)
            if asyncio.iscoroutine(res):
                feature_result = await res
            else:
                feature_result = res
            
            # Get optimization statistics with null check
            try:
                optimization_stats = self.auto_optimized_generator.get_optimization_stats()
                if optimization_stats is None:
                    optimization_stats = {'status': 'unavailable', 'message': 'Optimization stats not available'}
            except Exception as e:
                self.logger.warning(f"Failed to get optimization stats: {e}")
                optimization_stats = {'status': 'error', 'message': f'Failed to get optimization stats: {e}'}
            
            # Get feature categories used - handle both singular and plural
            if hasattr(feature_config, 'categories') and feature_config.categories:
                feature_categories = [getattr(cat, 'value', str(cat)) for cat in feature_config.categories]
            elif hasattr(feature_config, 'category') and feature_config.category is not None:
                feature_categories = [getattr(feature_config.category, 'value', str(feature_config.category))]
            else:
                feature_categories = []
            
            # Get VectorBT optimization details with safe attribute access
            vectorbt_optimizations = {
                'vectorbt_enabled': bool(getattr(feature_config, 'enable_vectorbt', False)),
                'optimization_level': getattr(self.auto_optimization_config, 'optimization_level', None),
                'parallel_processing': bool(getattr(self.auto_optimization_config, 'enable_parallel', False)),
                'memory_optimization': bool(getattr(self.auto_optimization_config, 'enable_memory_optimization', False))
            }
            
            # Compile comprehensive result with safe attribute access
            return FeatureGenerationResult(
                success=getattr(feature_result, 'success', False),
                generated_features=getattr(feature_result, 'features', pd.DataFrame()),
                feature_metadata=getattr(feature_result, 'feature_metadata', {}),
                generation_metrics=getattr(feature_result, 'metrics', {}),
                optimization_stats=optimization_stats,
                feature_categories=feature_categories,
                vectorbt_optimizations=vectorbt_optimizations,
                artifacts={
                    'feature_result': getattr(feature_result, '__dict__', {}),
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
            # Basic feature generation with prefixed fallback columns
            basic_features = pd.DataFrame(index=data.index)
            
            # Simple technical indicators with proper min_periods
            basic_features['fb_sma_20'] = data['close'].rolling(20, min_periods=20).mean()
            basic_features['fb_sma_50'] = data['close'].rolling(50, min_periods=50).mean()
            basic_features['fb_rsi_14'] = self._calculate_rsi(data['close'], 14)
            basic_features['fb_bb_upper'] = data['close'].rolling(20, min_periods=20).mean() + 2 * data['close'].rolling(20, min_periods=20).std()
            basic_features['fb_bb_lower'] = data['close'].rolling(20, min_periods=20).mean() - 2 * data['close'].rolling(20, min_periods=20).std()
            basic_features['fb_volume_sma'] = data['volume'].rolling(20, min_periods=20).mean()
            
            # Remove NaN values
            basic_features = basic_features.dropna()
            
            return FeatureGenerationResult(
                success=True,
                generated_features=basic_features,
                feature_metadata={'method': 'fallback_basic', 'symbol': symbol, 'timeframe': timeframe},
                generation_metrics={'feature_count': len(basic_features.columns), 'shape': basic_features.shape},
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
        return rsi.clip(0, 100).fillna(50)  # Clip to [0,100] and fill NaNs with neutral RSI

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

    def _serialize_config(self, config, _depth=0) -> Dict[str, Any]:
        """Serialize configuration object to plain types for JSON serialization with recursion guard."""
        if _depth > 3:  # prevent runaway recursion
            return str(config)
            
        serialized = {}
        for key, value in config.__dict__.items():
            if hasattr(value, 'value'):  # Enum
                serialized[key] = value.value
            elif isinstance(value, (list, tuple, set)):
                serialized[key] = [self._serialize_config(x, _depth+1) for x in value]
            elif hasattr(value, '__dict__'):  # Object with __dict__
                serialized[key] = self._serialize_config(value, _depth+1)
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
        if not isinstance(data, pd.DataFrame) or data.empty:
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
