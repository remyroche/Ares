# src/training/vectorized_training_pipeline.py

"""Vectorized Training Pipeline for enhanced ML training processes.
Integrates matrix enhancements with existing training workflows to improve
performance, accuracy, and computational efficiency.
"""

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.training.matrix_enhancement_manager import MatrixEnhancementManager
from src.training.steps.vectorized_advanced_feature_engineering import (
    VectorizedAdvancedFeatureEngineering,
)
from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger


@dataclass
class VectorizedTrainingConfig:
    """Configuration for vectorized training pipeline."""

    # Matrix enhancement settings
    enable_matrix_enhancement: bool = True
    enable_vectorized_features: bool = True
    enable_parallel_processing: bool = True

    # Training optimization
    enable_batch_processing: bool = True
    enable_memory_optimization: bool = True
    enable_gpu_acceleration: bool = False

    # Quality settings
    enable_quality_gates: bool = True
    enable_performance_monitoring: bool = True

    # Integration settings
    integrate_with_existing_pipeline: bool = True
    preserve_original_features: bool = True


class VectorizedTrainingPipeline:
    """Vectorized training pipeline with matrix enhancements."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize vectorized training pipeline."""
        self.config = VectorizedTrainingConfig(**config.get("vectorized_training", {}))
        self.logger = system_logger.getChild("VectorizedTrainingPipeline")

        # Initialize components
        self.matrix_enhancement = MatrixEnhancementManager(config)
        self.vectorized_features = None

        if self.config.enable_vectorized_features:
            self.vectorized_features = VectorizedAdvancedFeatureEngineering(config)

        # Pipeline state
        self.pipeline_results = {}
        self.performance_metrics = {}

    @handle_errors(exceptions=(ValueError, AttributeError), default_return=False)
    @handle_errors(exceptions=(ValueError, AttributeError), default_return=None)
    async def enhance_training_data(
        self, training_data: dict[str, Any],
        step_name: str = "vectorized_enhancement",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Enhance training data with vectorized and matrix operations.

        Args:
            training_data: Input training data dictionary
            step_name: Name of the processing step

        Returns:
            Tuple of (enhanced_training_data, enhancement_metadata)

        """
        try:
            start_time = time.time()
            self.logger.info(f"🔄 Starting vectorized enhancement for {step_name}")

            # 1. Copy input data
            enhanced_data = training_data.copy()
            enhancement_metadata = {}

            # 2. Apply vectorized features if enabled
            if self.config.enable_vectorized_features and self.vectorized_features:
                if "features" in training_data:
                    features_df = training_data["features"]

                    enhanced_features, feature_metadata = await self._apply_vectorized_features(features_df)
                    enhanced_data["features"] = enhanced_features
                    enhancement_metadata["vectorized_features"] = feature_metadata

            # 3. Apply matrix enhancement if enabled
            if self.config.enable_matrix_enhancement:
                if "features" in enhanced_data:
                    features_df = enhanced_data["features"]

                    matrix_enhanced_features, matrix_metadata = (
                        self.matrix_enhancement.enhance_training_features(features_df)
                    )
                    enhanced_data["features"] = matrix_enhanced_features
                    enhancement_metadata["matrix_enhancement"] = matrix_metadata

            # 4. Apply quality gates if enabled
            if self.config.enable_quality_gates:
                quality_passed = await self._apply_quality_gates(enhanced_data)
                if not quality_passed:
                    self.logger.warning(
                        "⚠️ Quality gates failed, reverting to original data",
                    )
                    return training_data, enhancement_metadata

            # 5. Performance monitoring
            total_time = time.time() - start_time
            enhancement_metadata["processing_time"] = total_time
            enhancement_metadata["step_name"] = step_name

            self.logger.info(f"✅ Vectorized enhancement completed in {total_time:.2f}s")
            return enhanced_data, enhancement_metadata

        except Exception as e:
            self.logger.exception(
                f"❌ Vectorized training data enhancement failed: {e}",
            )
            return {"error": str(e)}

    @handle_errors(exceptions=(ValueError, AttributeError), default_return=None)
    async def _apply_vectorized_features(
        self, features_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Apply vectorized feature engineering."""
        try:
            if not self.vectorized_features:
                return features_df, {"status": "skipped", "reason": "vectorized_features_disabled"}

            enhanced_features, metadata = await self.vectorized_features.enhance_features(features_df)
            return enhanced_features, metadata

        except Exception as e:
            self.logger.exception(f"❌ Vectorized feature application failed: {e}")
            return features_df, {"error": str(e)}

    @handle_errors(exceptions=(ValueError, AttributeError), default_return=False)
    async def _apply_quality_gates(self, enhanced_data: dict[str, Any]) -> bool:
        """Apply quality gates to enhanced data."""
        try:
            # Basic quality checks
            if "features" not in enhanced_data:
                return False

            features_df = enhanced_data["features"]

            # Check for NaN values
            if features_df.isnull().any().any():
                self.logger.warning("⚠️ Quality gate failed: NaN values detected")
                return False

            # Check for infinite values
            if np.isinf(features_df.select_dtypes(include=[np.number])).any().any():
                self.logger.warning("⚠️ Quality gate failed: Infinite values detected")
                return False

            # Check for empty dataframe
            if features_df.empty:
                self.logger.warning("⚠️ Quality gate failed: Empty features dataframe")
                return False

            return True

        except Exception as e:
            self.logger.exception(f"❌ Quality gate application failed: {e}")
            return False

    @handle_errors(exceptions=(ValueError, AttributeError), default_return=None)
    @handle_errors(exceptions=(ValueError, AttributeError), default_return=None)
    @handle_errors(exceptions=(ValueError, AttributeError), default_return=None)