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

from src.training.steps.pre_training.unified_data_driven_pipeline.core.modular_architecture import (
    ModularComponent
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
class FeatureGenerationVectorizationStep(ModularComponent):
    """Enhanced vectorization step using VectorizedFeatureGenerator."""

    def __init__(self, name: str = "step", config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        """Initialize the enhanced vectorization step."""
        super().__init__(name, config or {}, logger)
            name="feature_generation_vectorization_step",
            config=config_dict,
            logger=logging.getLogger(__name__)
        )
        
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

    # Required abstract methods from ModularComponent
    def _initialize_resources(self) -> bool:
        """Initialize component-specific resources."""
        try:
            # Initialize vectorization components
            if VECTORIZATION_COMPONENTS_AVAILABLE:
                self.matrix_processor = MatrixFeatureProcessor()
                self.vectorized_generator = VectorizedFeatureGenerator()
                self.vectorbt_generator = VectorBTFeatureGenerator()
                self.unified_manager = UnifiedVectorizationManager()
            else:
                self.matrix_processor = None
                self.vectorized_generator = None
                self.vectorbt_generator = None
                self.unified_manager = None
            
            # Set initial state
            self.set_state('initialized_at', datetime.now().isoformat())
            self.set_state('vectorization_count', 0)
            return True
        except Exception as e:
            self.logger.error(f"Resource initialization failed: {e}")
            return False
    
    def _cleanup_resources(self) -> None:
        """Cleanup component-specific resources."""
        self.set_state('cleaned_up_at', datetime.now().isoformat())
        self.set_state('vectorization_count', 0)
    
    def _process_data(self, data: Any, **kwargs) -> Any:
        """Process data with component logic."""
        # Increment vectorization count
        count = self.get_state('vectorization_count', 0)
        self.set_state('vectorization_count', count + 1)
        
        # Basic processing - return data as-is for now
        # The actual vectorization is done in the execute method
        return data
    
    def _get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for this component."""
        return {
            'min_size': 100,
            'max_size': 1000000,
            'required_attributes': ['open', 'high', 'low', 'close', 'volume'],
            'data_types': ['pandas.DataFrame'],
            'max_nan_ratio': 0.1,
            'min_unique_values': 2
        }
    
    def _validate_component_specific(self, data: Any) -> Dict[str, Any]:
        """Validate data with component-specific rules."""
        errors = []
        warnings = []
        metadata = {}
        
        if isinstance(data, pd.DataFrame):
            # Check required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                errors.append(f"Missing required columns: {missing_columns}")
            
            # Check data size
            if len(data) < 100:
                warnings.append("Data size is small (< 100 rows)")
            
            # Check for NaN values
            nan_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            if nan_ratio > 0.1:
                warnings.append(f"High NaN ratio: {nan_ratio:.2%}")
            
            metadata['data_shape'] = data.shape
            metadata['nan_ratio'] = nan_ratio
            metadata['columns'] = list(data.columns)
        
        return {'errors': errors, 'warnings': warnings, 'metadata': metadata}

# Command handler for ares_launcher integration
