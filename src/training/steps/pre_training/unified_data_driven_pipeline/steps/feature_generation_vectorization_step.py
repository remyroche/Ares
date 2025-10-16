"""
Enhanced Feature Vectorization Step

This step performs advanced feature vectorization using VectorizedFeatureGenerator
with matrix integration and VectorBT optimization.
"""

from __future__ import annotations

import warnings
import logging
import json
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

from src.training.steps.pre_training.components.base_component import (
    BasePreTrainingComponent, ComponentConfig, ComponentResult
)
from src.utils.common_operations import safe_dataframe_operation
from src.utils.matrix_operations import safe_matrix_multiply, optimize_dataframe

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

# Import tprint utilities for enhanced logging
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug,
        tprint_performance, tprint_step, tprint_result
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)
    def tprint_performance(*args, **kwargs): print("PERFORMANCE:", *args, **kwargs)
    def tprint_step(*args, **kwargs): print("STEP:", *args, **kwargs)
    def tprint_result(*args, **kwargs): print("RESULT:", *args, **kwargs)

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

        # Extract parameters from training_input
        data = training_input.get('data')
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
                # Fallback to basic vectorization
                return await self._fallback_vectorization(data, training_input, pipeline_state)

            # Perform comprehensive vectorization
            vectorization_result = await self._perform_enhanced_vectorization(
                data, symbol, timeframe, direction, custom_overrides
            )

            # Convert result to ComponentResult
            component_result = ComponentResult(
                success=vectorization_result.success,
                artifacts=vectorization_result.artifacts,
                metadata={
                    'vectorized_features': vectorization_result.vectorized_features,
                    'vectorization_metadata': vectorization_result.vectorization_metadata,
                    'matrix_operations': vectorization_result.matrix_operations,
                    'vectorbt_optimizations': vectorization_result.vectorbt_optimizations,
                    'performance_metrics': vectorization_result.performance_metrics
                },
                error_message=vectorization_result.error_message
            )

            if component_result.success:
                self.logger.info(f"✅ Enhanced vectorization completed successfully")
                self.logger.info(f"📊 Vectorized {vectorization_result.vectorized_features} features")
                self.logger.info(f"🔧 Matrix operations: {vectorization_result.matrix_operations}")
                self.logger.info(f"⚡ VectorBT optimizations: {vectorization_result.vectorbt_optimizations}")
            else:
                self.logger.error(f"❌ Vectorization failed: {component_result.error_message}")

            return component_result

        except Exception as e:
            self.logger.error(f"❌ Enhanced vectorization step failed with exception: {e}")
            return ComponentResult(
                success=False,
                artifacts={},
                metadata={},
                error_message=str(e)
            )

    async def _perform_enhanced_vectorization(self, data: pd.DataFrame, symbol: str,
                                              timeframe: str, direction: str,
                                              custom_overrides: Optional[Dict[str, Any]]) -> VectorizationResult:
        """Perform enhanced vectorization using VectorizedFeatureGenerator."""
        
        try:
            # Step 1: Matrix feature processing
            matrix_result = await self.matrix_processor.process_features(
                data, symbol=symbol, timeframe=timeframe
            )
            
            # Step 2: Vectorized feature generation
            vectorized_result = await self.vectorized_generator.generate_vectorized_features(
                matrix_result.processed_data,
                symbol=symbol,
                timeframe=timeframe,
                direction=direction
            )
            
            # Step 3: VectorBT optimization
            vectorbt_result = await self.vectorbt_generator.optimize_vectorization(
                vectorized_result.features,
                symbol=symbol,
                timeframe=timeframe
            )
            
            # Step 4: Unified vectorization management
            unified_result = await self.vectorization_manager.manage_vectorization(
                vectorbt_result.optimized_features,
                symbol=symbol,
                timeframe=timeframe
            )
            
            # Compile comprehensive result
            return VectorizationResult(
                success=unified_result.success,
                vectorized_features=len(unified_result.final_features.columns),
                vectorization_metadata={
                    'matrix_processing': matrix_result.metadata,
                    'vectorized_generation': vectorized_result.metadata,
                    'vectorbt_optimization': vectorbt_result.metadata,
                    'unified_management': unified_result.metadata
                },
                matrix_operations=matrix_result.operations,
                vectorbt_optimizations=vectorbt_result.optimizations,
                performance_metrics=unified_result.performance_metrics,
                artifacts={
                    'matrix_result': matrix_result.__dict__,
                    'vectorized_result': vectorized_result.__dict__,
                    'vectorbt_result': vectorbt_result.__dict__,
                    'unified_result': unified_result.__dict__
                }
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

    async def _fallback_vectorization(self, data: pd.DataFrame, training_input: Dict[str, Any],
                                     pipeline_state: Dict[str, Any]) -> ComponentResult:
        """Fallback vectorization when advanced components are not available."""
        
        try:
            # Basic vectorization using numpy
            vectorized_data = data.values  # Convert to numpy array
            
            # Simple feature scaling
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(vectorized_data)
            
            # Create vectorized features dataframe
            vectorized_df = pd.DataFrame(
                scaled_data, 
                index=data.index, 
                columns=[f'vec_{i}' for i in range(scaled_data.shape[1])]
            )
            
            return ComponentResult(
                success=True,
                artifacts={'vectorized_data': vectorized_df},
                metadata={
                    'vectorized_features': len(vectorized_df.columns),
                    'vectorization_metadata': {'method': 'fallback_scaling'},
                    'matrix_operations': {'scaling': 'standard'},
                    'vectorbt_optimizations': {'vectorbt_enabled': False},
                    'performance_metrics': {'method': 'fallback'}
                },
                error_message=None
            )
            
        except Exception as e:
            return ComponentResult(
                success=False,
                artifacts={},
                metadata={},
                error_message=str(e)
            )

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
) -> VectorizationResult:
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
        VectorizationResult with vectorization results
    """
    # Create sample data for vectorization (in real usage, this would come from data loading)
    sample_data = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 105,
        'low': np.random.randn(1000).cumsum() + 95,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    })

    # Create step instance and execute
    step = FeatureGenerationVectorizationStep()

    return await step.execute(
        data=sample_data,
        symbol=symbol,
        timeframe=timeframe,
        direction=direction,
        intensity=intensity,
        lookback_days=lookback_days,
        start_date=start_date,
        end_date=end_date,
        exchange=exchange,
        custom_overrides=custom_overrides
    )

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