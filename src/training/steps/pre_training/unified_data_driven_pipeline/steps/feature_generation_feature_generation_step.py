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

from src.training.steps.pre_training.unified_data_driven_pipeline.core.modular_architecture import (
    ModularComponent
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
class FeatureGenerationStep(ModularComponent):
    """Enhanced feature generation step using AutoOptimizedFeatureGenerator."""

    def __init__(self, name: str = "step", config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        """Initialize the enhanced feature generation step."""
        super().__init__(name, config or {}, logger)
            name="feature_generation_step",
            config=config_dict,
            logger=logging.getLogger(__name__)
        )
        
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
                use_vectorbt=True,  # Enable VectorBT optimization
                enable_gpu=True,  # Enable GPU acceleration
                enable_parallel=True  # Enable parallel processing
            )
            
            # Create auto-optimization configuration
            self.auto_optimization_config = AutoOptimizationConfig(
                optimization_level=OptimizationLevel.BALANCED,
                enable_auto_optimization=True,  # Enable auto-optimization
                enable_vectorbt_optimization=True,  # Enable VectorBT optimization
                enable_memory_optimization=True,  # Enable memory optimization
                enable_gpu_acceleration=True  # Enable GPU acceleration
            )
            
            # Initialize feature generators
            self.auto_optimized_generator = AutoOptimizedFeatureGenerator(
                self.feature_config, 
                self.auto_optimization_config
            )
            
        else:
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
                # Fast fail if enhanced components are not available
                raise RuntimeError("Enhanced feature generation components are not available")

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
                'vectorbt_enabled': bool(getattr(feature_config, 'use_vectorbt', False)),
                'optimization_level': getattr(self.auto_optimization_config, 'optimization_level', None),
                'parallel_processing': bool(getattr(feature_config, 'enable_parallel', False)),
                'memory_optimization': bool(getattr(self.auto_optimization_config, 'enable_memory_optimization', False)),
                'gpu_acceleration': bool(getattr(feature_config, 'enable_gpu', False))
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
            # Fast fail - no fallback, just raise the error
            raise RuntimeError(f"Feature generation failed: {e}") from e



    # Required utility methods for BasePreTrainingComponent
