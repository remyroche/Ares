"""
Enhanced Feature Vectorization Step

This step performs advanced feature vectorization using VectorizedFeatureGenerator
with matrix integration and VectorBT optimization.
"""

from __future__ import annotations

import logging
import pandas as pd
import numpy as np
import asyncio
import inspect
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from src.training.steps.pre_training.components.base_component import (
    BasePreTrainingComponent, ComponentConfig, ComponentResult
)

# Import advanced vectorization components
try:
    from src.feature_generation.matrix_integration.matrix_processor import (
        MatrixFeatureProcessor, VectorizedFeatureGenerator
    )
    from src.feature_generation.core.vectorbt_feature_generator import (
        VectorBTFeatureGenerator
    )
    from src.utils.ml_common.unified_vectorization_manager import (
        UnifiedVectorizationManager
    )
    VECTORIZATION_COMPONENTS_AVAILABLE = True
except ImportError:
    VECTORIZATION_COMPONENTS_AVAILABLE = False
    MatrixFeatureProcessor = None
    VectorizedFeatureGenerator = None
    VectorBTFeatureGenerator = None
    UnifiedVectorizationManager = None


@dataclass
class VectorizationResult:
    """Enhanced result of vectorization step."""

    success: bool
    vectorized_features: int
    vectorization_metadata: Dict[str, Any]
    matrix_operations: Dict[str, Any]
    vectorbt_optimizations: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    artifacts: Dict[str, Any]
    error_message: Optional[str] = None

class FeatureGenerationVectorizationStep(BasePreTrainingComponent):
    """Enhanced vectorization step using VectorizedFeatureGenerator."""

    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the enhanced vectorization step."""
        super().__init__(config or ComponentConfig())
        self.logger = logging.getLogger(__name__)
        
        # Initialize vectorization components
        if VECTORIZATION_COMPONENTS_AVAILABLE:
            # Initialize matrix feature processor
            self.matrix_processor = MatrixFeatureProcessor()
            
            # Initialize vectorized feature generator
            self.vectorized_generator = VectorizedFeatureGenerator()
            
            # Initialize VectorBT feature generator
            self.vectorbt_generator = VectorBTFeatureGenerator()
            
            # Initialize unified vectorization manager
            self.vectorization_manager = UnifiedVectorizationManager()
        else:
            self.logger.warning("⚠️ Advanced vectorization components not available, using fallback")
            self.matrix_processor = None
            self.vectorized_generator = None
            self.vectorbt_generator = None
            self.vectorization_manager = None

    async def execute(self,
                     training_input: Dict[str, Any],
                     pipeline_state: Dict[str, Any]) -> ComponentResult:
        """Execute enhanced vectorization step using VectorizedFeatureGenerator."""

        self.logger.info("⚡ Starting enhanced vectorization step with matrix integration and VectorBT optimization")

        # Validate input data
        data = training_input.get('data')
        if data is None or not isinstance(data, pd.DataFrame) or data.empty:
            msg = "Input 'data' must be a non-empty pandas DataFrame."
            self.logger.error(msg)
            return ComponentResult(success=False, artifacts={}, metadata={}, error_message=msg)

        # Extract parameters from training_input
        symbol = training_input.get('symbol', 'ETHUSDT')
        timeframe = training_input.get('timeframe', '15m')
        direction = training_input.get('direction', 'longs')
        intensity = training_input.get('intensity', 'blank')
        lookback_days = training_input.get('lookback_days')
        start_date = training_input.get('start_date')
        end_date = training_input.get('end_date')
        exchange = training_input.get('exchange', 'binance')
        custom_overrides = training_input.get('custom_overrides')

        try:
            if not VECTORIZATION_COMPONENTS_AVAILABLE:
                # Fast fail instead of fallback
                msg = "Advanced vectorization components not available. Install required dependencies."
                self.logger.error(msg)
                return ComponentResult(success=False, artifacts={}, metadata={}, error_message=msg)

            # Perform comprehensive vectorization
            vectorization_result = await self._perform_enhanced_vectorization(
                data, symbol, timeframe, direction, custom_overrides
            )

            if not vectorization_result.success:
                return ComponentResult(success=False, artifacts={}, metadata={}, error_message=vectorization_result.error_message)

            # Extract final_features for consistent artifact
            final_features = vectorization_result.vectorization_metadata.get('unified_management', {}).get('final_features')
            if final_features is None and 'unified_result' in vectorization_result.artifacts:
                unified_dict = vectorization_result.artifacts['unified_result']
                final_features = getattr(unified_dict, 'final_features', None) if hasattr(unified_dict, 'final_features') else unified_dict.get('final_features')

            # Ensure we only store serializable structures or DataFrames
            artifacts = {}
            if isinstance(final_features, pd.DataFrame):
                artifacts['final_features'] = final_features

            # Clean, JSON-safe metadata only
            metadata = {
                'vectorized_features': vectorization_result.vectorized_features,
                'vectorization_metadata': vectorization_result.vectorization_metadata,
                'matrix_operations': vectorization_result.matrix_operations,
                'vectorbt_optimizations': vectorization_result.vectorbt_optimizations,
                'performance_metrics': vectorization_result.performance_metrics,
                'inputs': {
                    'symbol': symbol, 'timeframe': timeframe, 'direction': direction, 'intensity': intensity,
                    'lookback_days': lookback_days, 'start_date': start_date, 'end_date': end_date, 'exchange': exchange
                }
            }

            self.logger.info("✅ Enhanced vectorization completed successfully")
            self.logger.info(f"📊 Vectorized {vectorization_result.vectorized_features} features")

            return ComponentResult(success=True, artifacts=artifacts, metadata=metadata, error_message=None)

        except Exception as e:
            self.logger.error(f"❌ Enhanced vectorization step failed with exception: {e}")
            return ComponentResult(
                success=False,
                artifacts={},
                metadata={},
                error_message=str(e)
            )

    async def _maybe_await(self, fn, *args, **kwargs):
        """Helper to handle both sync and async component methods."""
        if inspect.iscoroutinefunction(fn):
            return await fn(*args, **kwargs)
        return await asyncio.to_thread(fn, *args, **kwargs)

    async def _perform_enhanced_vectorization(self, data: pd.DataFrame, symbol: str,
                                              timeframe: str, direction: str,
                                              custom_overrides: Optional[Dict[str, Any]]) -> VectorizationResult:
        """Perform enhanced vectorization using VectorizedFeatureGenerator."""
        
        try:
            # Step 1: Matrix feature processing
            matrix_result = await self._maybe_await(
                self.matrix_processor.process_features, data, symbol=symbol, timeframe=timeframe
            )
            
            # Step 2: Vectorized feature generation
            vectorized_result = await self._maybe_await(
                self.vectorized_generator.generate_vectorized_features,
                matrix_result.processed_data, symbol=symbol, timeframe=timeframe, direction=direction
            )
            
            # Step 3: VectorBT optimization
            vectorbt_result = await self._maybe_await(
                self.vectorbt_generator.optimize_vectorization,
                vectorized_result.features, symbol=symbol, timeframe=timeframe
            )
            
            # Step 4: Unified vectorization management
            unified_result = await self._maybe_await(
                self.vectorization_manager.manage_vectorization,
                vectorbt_result.optimized_features, symbol=symbol, timeframe=timeframe
            )
            
            # Extract final_features for consistent artifact
            final_features = getattr(unified_result, 'final_features', None)
            
            # Compile comprehensive result
            return VectorizationResult(
                success=unified_result.success,
                vectorized_features=0 if final_features is None else final_features.shape[1],
                vectorization_metadata={
                    'matrix_processing': getattr(matrix_result, 'metadata', {}),
                    'vectorized_generation': getattr(vectorized_result, 'metadata', {}),
                    'vectorbt_optimization': getattr(vectorbt_result, 'metadata', {}),
                    'unified_management': getattr(unified_result, 'metadata', {}),
                },
                matrix_operations=getattr(matrix_result, 'operations', {}),
                vectorbt_optimizations=getattr(vectorbt_result, 'optimizations', {}),
                performance_metrics=getattr(unified_result, 'performance_metrics', {}),
                artifacts={'final_features': final_features}
            )
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced vectorization failed: {e}")
            return VectorizationResult(
                success=False,
                vectorized_features=0,
                vectorization_metadata={},
                matrix_operations={},
                vectorbt_optimizations={},
                performance_metrics={},
                artifacts={},
                error_message=str(e)
            )



# Command handler for ares_launcher integration
async def handle_feature_generation_vectorization_step(
    symbol: str = "ETHUSDT",
    timeframe: str = "15m",
    direction: str = "longs",
    intensity: str = "blank",
    lookback_days: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    exchange: str = "binance",
    custom_overrides: Optional[Dict[str, Any]] = None,
    **kwargs
) -> ComponentResult:
    """
    Handle feature generation vectorization step command.

    Args:
        symbol: Trading symbol (default: "ETHUSDT")
        timeframe: Timeframe (default: "15m")
        direction: Direction (default: "longs")
        intensity: Pipeline intensity (default: "blank")
        lookback_days: Lookback days (optional)
        start_date: Start date (optional)
        end_date: End date (optional)
        exchange: Exchange (default: "binance")
        custom_overrides: Custom configuration overrides (optional)
        **kwargs: Additional arguments

    Returns:
        ComponentResult with vectorization results
    """
    # Determinism for sample data
    rng = np.random.default_rng(seed=42)
    sample_data = pd.DataFrame({
        'open': rng.normal(size=1000).cumsum() + 100,
        'high': rng.normal(size=1000).cumsum() + 105,
        'low': rng.normal(size=1000).cumsum() + 95,
        'close': rng.normal(size=1000).cumsum() + 100,
        'volume': rng.integers(1000, 10000, size=1000)
    })

    training_input = {
        'data': sample_data,
        'symbol': symbol,
        'timeframe': timeframe,
        'direction': direction,
        'intensity': intensity,
        'lookback_days': lookback_days,
        'start_date': start_date,
        'end_date': end_date,
        'exchange': exchange,
        'custom_overrides': custom_overrides,
    }

    step = FeatureGenerationVectorizationStep()
    pipeline_state: Dict[str, Any] = {}
    return await step.execute(training_input, pipeline_state)

# Register component with factory
def _register_feature_generation_vectorization_step():
    """Register the feature generation vectorization step component with the factory."""
    try:
        from src.training.steps.pre_training.components import ComponentFactory
        ComponentFactory.register_component(
            'feature_generation_vectorization_step',
            FeatureGenerationVectorizationStep
        )
    except ImportError:
        # Component factory not available, skip registration
        pass

# Register the component when module is imported
_register_feature_generation_vectorization_step()