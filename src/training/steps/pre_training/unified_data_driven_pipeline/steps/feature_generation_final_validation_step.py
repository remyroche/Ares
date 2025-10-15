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
        
        # Generate human-readable report
        await self._generate_human_readable_report(artifacts_path, final_dataset, validation_summary, quality_metrics, pipeline_summary)
        
        return artifacts
    
    async def _generate_human_readable_report(self,
                                            artifacts_path: Path,
                                            final_dataset: pd.DataFrame,
                                            validation_summary: Dict[str, Any],
                                            quality_metrics: Dict[str, Any],
                                            pipeline_summary: Dict[str, Any]) -> None:
        """Generate human-readable report in outcomes/ directory.
        
        Args:
            artifacts_path: Path to save artifacts
            final_dataset: Final dataset
            validation_summary: Validation summary
            quality_metrics: Quality metrics
            pipeline_summary: Pipeline summary
        """
        # Create outcomes directory
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(exist_ok=True)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"final_validation_report_{timestamp}.md"
        report_path = outcomes_dir / report_filename
        
        # Calculate final statistics
        total_features = len(final_dataset.columns)
        total_samples = len(final_dataset)
        memory_usage_mb = final_dataset.memory_usage(deep=True).sum() / 1024 / 1024 if not final_dataset.empty else 0
        
        # Generate report content
        report_content = f"""# Final Validation Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
- **Final Dataset Shape**: {final_dataset.shape[0]} rows × {final_dataset.shape[1]} columns
- **Total Features**: {total_features}
- **Memory Usage**: {memory_usage_mb:.2f} MB
- **Validation Status**: {'✅ SUCCESS' if total_features > 0 else '❌ FAILED'}

## Pipeline Summary
- **Pipeline Name**: {pipeline_summary.get('pipeline_name', 'N/A')}
- **Execution Timestamp**: {pipeline_summary.get('execution_timestamp', 'N/A')}
- **Total Steps**: {pipeline_summary.get('total_steps', 0)}
- **Final Dataset Shape**: {pipeline_summary.get('final_dataset_shape', 'N/A')}
- **Final Feature Count**: {pipeline_summary.get('final_feature_count', 0)}

## Completed Steps
"""
        
        # Add completed steps
        completed_steps = pipeline_summary.get('steps_completed', [])
        for i, step in enumerate(completed_steps, 1):
            report_content += f"{i}. {step.replace('_', ' ').title()}
"
        
        if not completed_steps:
            report_content += "- No steps completed
"
        
        report_content += f"""

## Validation Summary
- **Data Shape**: {validation_summary.get('data_shape', 'N/A')}
- **Missing Values**: {validation_summary.get('missing_values', 0)}
- **Infinite Values**: {validation_summary.get('infinite_values', 0)}
- **Duplicate Rows**: {validation_summary.get('duplicate_rows', 0)}
- **Memory Usage**: {validation_summary.get('memory_usage_mb', 0.0):.2f} MB
- **Data Quality Score**: {validation_summary.get('data_quality_score', 0.0):.3f}

## Quality Metrics
- **Total Features**: {quality_metrics.get('total_features', 0)}
- **Total Samples**: {quality_metrics.get('total_samples', 0)}
- **Feature Types**: {quality_metrics.get('feature_types', {})}
- **Memory Usage**: {quality_metrics.get('memory_usage_mb', 0.0):.2f} MB
- **Data Quality Score**: {quality_metrics.get('data_quality_score', 0.0):.3f}

## Dataset Overview
- **Feature Names**: {', '.join(final_dataset.columns[:10])}{'...' if len(final_dataset.columns) > 10 else ''}
- **Data Types**: {dict(final_dataset.dtypes.value_counts()) if not final_dataset.empty else 'N/A'}
- **Missing Values**: {final_dataset.isnull().sum().sum() if not final_dataset.empty else 0}
- **Memory per Feature**: {memory_usage_mb / total_features:.4f} MB (average) if total_features > 0 else 'N/A'}

## Quality Assessment
- **Overall Quality**: {'Excellent' if quality_metrics.get('data_quality_score', 0) > 0.8 else 'Good' if quality_metrics.get('data_quality_score', 0) > 0.6 else 'Fair' if quality_metrics.get('data_quality_score', 0) > 0.4 else 'Poor'}
- **Data Completeness**: {((final_dataset.count().sum() / (final_dataset.shape[0] * final_dataset.shape[1])) * 100):.2f}% if not final_dataset.empty else 'N/A'}
- **Memory Efficiency**: {'Excellent' if memory_usage_mb < 100 else 'Good' if memory_usage_mb < 500 else 'High' if memory_usage_mb < 1000 else 'Very High'}
- **Feature Diversity**: {len(set(final_dataset.dtypes))} different data types

## Pipeline Completion
- **Steps Completed**: {len(completed_steps)}/{pipeline_summary.get('total_steps', 0)}
- **Completion Rate**: {(len(completed_steps) / pipeline_summary.get('total_steps', 1)) * 100:.1f}%
- **Pipeline Status**: {'Complete' if len(completed_steps) == pipeline_summary.get('total_steps', 0) else 'Partial'}

## Next Steps
1. Review final dataset for quality
2. Use dataset for model training
3. Consider additional preprocessing if needed

---
*Report generated by Feature Generation Final Validation Step*
"""
        
        # Write report
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        self.logger.info(f"📊 Human-readable report saved: {report_path}")


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