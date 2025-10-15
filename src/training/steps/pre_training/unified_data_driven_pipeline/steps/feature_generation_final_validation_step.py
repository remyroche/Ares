"""
Feature Generation Final Validation Step

This step performs final validation and quality check as the last step in the
unified data-driven pipeline.
"""

from __future__ import annotations

import logging
import json
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

from src.training.steps.pre_training.unified_data_driven_pipeline.core.config import UnifiedPipelineConfig


@dataclass
class FinalValidationResult:
    """Result of final validation step."""
    
    success: bool
    final_dataset: pd.DataFrame
    validation_summary: Dict[str, Any]
    quality_metrics: Dict[str, Any]
    pipeline_summary: Dict[str, Any]
    error_message: Optional[str] = None
    artifacts: Optional[Dict[str, Any]] = None


class FeatureGenerationFinalValidationStep:
    """Final validation step for feature generation pipeline."""
    
    def __init__(self, config: UnifiedPipelineConfig, logger: Optional[logging.Logger] = None):
        """Initialize the final validation step.
        
        Args:
            config: Unified pipeline configuration
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
    
    async def execute(self, 
                     market_data: pd.DataFrame,
                     artifacts_dir: str,
                     previous_artifacts: Optional[Dict[str, Any]] = None,
                     **kwargs) -> FinalValidationResult:
        """Execute final validation step.
        
        Args:
            market_data: Input market data
            artifacts_dir: Directory to save artifacts
            previous_artifacts: Artifacts from previous steps
            **kwargs: Additional arguments
            
        Returns:
            FinalValidationResult with final dataset
        """
        self.logger.info("✅ Starting final validation step...")
        
        try:
            # Create artifacts directory
            artifacts_path = Path(artifacts_dir) / "feature_generation_final_validation_step"
            artifacts_path.mkdir(parents=True, exist_ok=True)
            
            # Load labeled data from previous step
            labeled_data = await self._load_labeled_data(previous_artifacts)
            
            if labeled_data.empty:
                self.logger.warning("⚠️ No labeled data available for final validation")
                return FinalValidationResult(
                    success=False,
                    final_dataset=pd.DataFrame(),
                    validation_summary={},
                    quality_metrics={},
                    pipeline_summary={},
                    error_message="No labeled data available for final validation"
                )
            
            # Perform final validation
            validation_summary = await self._perform_final_validation(labeled_data)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(labeled_data)
            
            # Generate pipeline summary
            pipeline_summary = self._generate_pipeline_summary(previous_artifacts, labeled_data)
            
            # Create final dataset
            final_dataset = self._create_final_dataset(labeled_data, market_data)
            
            # Save artifacts
            artifacts = await self._save_artifacts(
                artifacts_path, final_dataset, validation_summary, 
                quality_metrics, pipeline_summary
            )
            
            self.logger.info(f"✅ Final validation completed with {len(final_dataset.columns)} features")
            
            return FinalValidationResult(
                success=True,
                final_dataset=final_dataset,
                validation_summary=validation_summary,
                quality_metrics=quality_metrics,
                pipeline_summary=pipeline_summary,
                artifacts=artifacts
            )
            
        except Exception as e:
            self.logger.error(f"❌ Final validation failed: {e}")
            return FinalValidationResult(
                success=False,
                final_dataset=pd.DataFrame(),
                validation_summary={},
                quality_metrics={},
                pipeline_summary={},
                error_message=str(e)
            )
    
    async def _load_labeled_data(self, 
                                previous_artifacts: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """Load labeled data from previous step artifacts."""
        if not previous_artifacts:
            return pd.DataFrame()
        
        # Try to find labeled data from previous step
        for key, path in previous_artifacts.items():
            if 'labeled_data' in key and path.endswith('.parquet'):
                try:
                    return pd.read_parquet(path)
                except Exception as e:
                    self.logger.error(f"Failed to load labeled data from {key}: {e}")
                    continue
        
        return pd.DataFrame()
    
    async def _perform_final_validation(self, 
                                       labeled_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform final validation on the dataset.
        
        Args:
            labeled_data: Labeled data to validate
            
        Returns:
            Validation summary
        """
        validation_results = {
            "data_shape": labeled_data.shape,
            "missing_values": labeled_data.isnull().sum().sum(),
            "infinite_values": np.isinf(labeled_data.select_dtypes(include=[np.number])).sum().sum(),
            "duplicate_rows": labeled_data.duplicated().sum(),
            "memory_usage_mb": labeled_data.memory_usage(deep=True).sum() / 1024 / 1024,
            "validation_timestamp": datetime.now().isoformat()
        }
        
        # Check for label columns
        label_columns = [col for col in labeled_data.columns if 'label' in col.lower()]
        validation_results["label_columns"] = label_columns
        validation_results["label_column_count"] = len(label_columns)
        
        # Validate data quality
        validation_results["data_quality_score"] = self._calculate_data_quality_score(labeled_data)
        
        return validation_results
    
    def _calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate overall data quality score.
        
        Args:
            data: Data to evaluate
            
        Returns:
            Quality score between 0 and 1
        """
        scores = []
        
        # Check for missing values
        missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        scores.append(1.0 - missing_ratio)
        
        # Check for infinite values
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            infinite_ratio = np.isinf(numeric_data).sum().sum() / (numeric_data.shape[0] * numeric_data.shape[1])
            scores.append(1.0 - infinite_ratio)
        else:
            scores.append(1.0)
        
        # Check for duplicate rows
        duplicate_ratio = data.duplicated().sum() / len(data)
        scores.append(1.0 - duplicate_ratio)
        
        return np.mean(scores) if scores else 0.0
    
    def _calculate_quality_metrics(self, 
                                  labeled_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate quality metrics for the final dataset."""
        return {
            "total_features": len(labeled_data.columns),
            "total_samples": len(labeled_data),
            "feature_types": labeled_data.dtypes.value_counts().to_dict(),
            "memory_usage_mb": labeled_data.memory_usage(deep=True).sum() / 1024 / 1024,
            "data_quality_score": self._calculate_data_quality_score(labeled_data),
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_pipeline_summary(self, 
                                  previous_artifacts: Optional[Dict[str, Any]],
                                  labeled_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive pipeline summary.
        
        Args:
            previous_artifacts: Artifacts from all previous steps
            labeled_data: Final labeled data
            
        Returns:
            Pipeline summary
        """
        summary = {
            "pipeline_name": "unified_data_driven_pipeline",
            "execution_timestamp": datetime.now().isoformat(),
            "total_steps": 9,  # Number of steps in the pipeline
            "final_dataset_shape": labeled_data.shape,
            "final_feature_count": len(labeled_data.columns),
            "steps_completed": []
        }
        
        # Add information about completed steps based on artifacts
        if previous_artifacts:
            step_mapping = {
                "data_quality_report": "data_validation",
                "generated_features": "feature_generation",
                "selected_features": "feature_selection",
                "optimal_periods": "period_optimization",
                "optimal_lookbacks": "lookback_optimization",
                "interaction_features": "interaction_generation",
                "vectorized_features": "vectorization",
                "labeled_data": "labeling_integration"
            }
            
            for artifact_key, step_name in step_mapping.items():
                if any(artifact_key in key for key in previous_artifacts.keys()):
                    summary["steps_completed"].append(step_name)
        
        return summary
    
    def _create_final_dataset(self, 
                             labeled_data: pd.DataFrame,
                             market_data: pd.DataFrame) -> pd.DataFrame:
        """Create the final dataset by combining features with market data.
        
        Args:
            labeled_data: Labeled features
            market_data: Original market data
            
        Returns:
            Final combined dataset
        """
        if market_data.empty:
            return labeled_data
        
        # Align indices if possible
        try:
            # Find common index
            common_index = labeled_data.index.intersection(market_data.index)
            if len(common_index) > 0:
                aligned_features = labeled_data.loc[common_index]
                aligned_market = market_data.loc[common_index]
                return pd.concat([aligned_market, aligned_features], axis=1)
            else:
                return labeled_data
        except Exception as e:
            self.logger.warning(f"Could not align with market data: {e}")
            return labeled_data
    
    async def _save_artifacts(self,
                             artifacts_path: Path,
                             final_dataset: pd.DataFrame,
                             validation_summary: Dict[str, Any],
                             quality_metrics: Dict[str, Any],
                             pipeline_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Save final validation artifacts."""
        artifacts = {}
        
        # Save final dataset
        if not final_dataset.empty:
            final_dataset_path = artifacts_path / "final_dataset.parquet"
            final_dataset.to_parquet(final_dataset_path)
            artifacts["final_dataset"] = str(final_dataset_path)
        
        # Save validation summary
        validation_summary_path = artifacts_path / "validation_summary.json"
        with open(validation_summary_path, 'w') as f:
            json.dump(validation_summary, f, indent=2)
        artifacts["validation_summary"] = str(validation_summary_path)
        
        # Save quality metrics
        quality_metrics_path = artifacts_path / "quality_metrics.json"
        with open(quality_metrics_path, 'w') as f:
            json.dump(quality_metrics, f, indent=2)
        artifacts["quality_metrics"] = str(quality_metrics_path)
        
        # Save pipeline summary
        pipeline_summary_path = artifacts_path / "pipeline_summary.json"
        with open(pipeline_summary_path, 'w') as f:
            json.dump(pipeline_summary, f, indent=2)
        artifacts["pipeline_summary"] = str(pipeline_summary_path)
        
        return artifacts


# Command handler for ares_launcher integration
async def handle_feature_generation_final_validation_step(
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
) -> FinalValidationResult:
    """Handle feature_generation_final_validation_step command."""
    from src.training.steps.pre_training.unified_data_driven_pipeline.core.simplified_config import (
        SimplifiedConfig
    )
    
    # Create configuration
    simplified_config = SimplifiedConfig()
    simplified_config.set_intensity(intensity)
    
    if custom_overrides:
        simplified_config.apply_custom_overrides(custom_overrides)
    
    config = simplified_config.get_config()
    
    # Create step instance
    step = FeatureGenerationFinalValidationStep(config)
    
    # Load market data (placeholder)
    market_data = pd.DataFrame()
    
    # Execute step
    return await step.execute(market_data, "artifacts")