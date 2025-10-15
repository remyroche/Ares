"""
Feature Generation Interaction Generation Step

This step generates feature interactions as part of the
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
from src.training.steps.pre_training.unified_data_driven_pipeline.stages.optimization_stage import (
    TemplateInteractionGenerator
)


@dataclass
class InteractionGenerationResult:
    """Result of interaction generation step."""
    
    success: bool
    interaction_features: pd.DataFrame
    interaction_metadata: Dict[str, Any]
    generation_metrics: Dict[str, Any]
    error_message: Optional[str] = None
    artifacts: Optional[Dict[str, Any]] = None


class FeatureGenerationInteractionGenerationStep:
    """Interaction generation step for feature generation pipeline."""
    
    def __init__(self, config: UnifiedPipelineConfig, logger: Optional[logging.Logger] = None):
        """Initialize the interaction generation step.
        
        Args:
            config: Unified pipeline configuration
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize interaction generator
        self.interaction_generator = TemplateInteractionGenerator()
    
    async def execute(self, 
                     market_data: pd.DataFrame,
                     artifacts_dir: str,
                     previous_artifacts: Optional[Dict[str, Any]] = None,
                     **kwargs) -> InteractionGenerationResult:
        """Execute interaction generation step.
        
        Args:
            market_data: Input market data
            artifacts_dir: Directory to save artifacts
            previous_artifacts: Artifacts from previous steps
            **kwargs: Additional arguments
            
        Returns:
            InteractionGenerationResult with generated interactions
        """
        self.logger.info("🔗 Starting interaction generation step...")
        
        try:
            # Create artifacts directory
            artifacts_path = Path(artifacts_dir) / "feature_generation_interaction_generation_step"
            artifacts_path.mkdir(parents=True, exist_ok=True)
            
            # Load features from previous step
            features = await self._load_features_from_previous_step(previous_artifacts)
            
            if features.empty:
                self.logger.warning("⚠️ No features available for interaction generation")
                return InteractionGenerationResult(
                    success=False,
                    interaction_features=pd.DataFrame(),
                    interaction_metadata={},
                    generation_metrics={},
                    error_message="No features available for interaction generation"
                )
            
            # Generate interactions
            interaction_features = await self._generate_interactions(features, market_data)
            
            # Generate interaction metadata
            interaction_metadata = self._generate_interaction_metadata(interaction_features, features)
            
            # Calculate generation metrics
            generation_metrics = self._calculate_generation_metrics(interaction_features, features)
            
            # Save artifacts
            artifacts = await self._save_artifacts(
                artifacts_path, interaction_features, interaction_metadata, generation_metrics
            )
            
            self.logger.info(f"✅ Interaction generation completed with {len(interaction_features.columns)} interaction features")
            
            return InteractionGenerationResult(
                success=True,
                interaction_features=interaction_features,
                interaction_metadata=interaction_metadata,
                generation_metrics=generation_metrics,
                artifacts=artifacts
            )
            
        except Exception as e:
            self.logger.error(f"❌ Interaction generation failed: {e}")
            return InteractionGenerationResult(
                success=False,
                interaction_features=pd.DataFrame(),
                interaction_metadata={},
                generation_metrics={},
                error_message=str(e)
            )
    
    async def _load_features_from_previous_step(self, 
                                               previous_artifacts: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """Load features from previous step artifacts."""
        if not previous_artifacts or "selected_features" not in previous_artifacts:
            return pd.DataFrame()
        
        try:
            features_path = previous_artifacts["selected_features"]
            return pd.read_parquet(features_path)
        except Exception as e:
            self.logger.error(f"Failed to load features from previous step: {e}")
            return pd.DataFrame()
    
    async def _generate_interactions(self, 
                                   features: pd.DataFrame,
                                   market_data: pd.DataFrame) -> pd.DataFrame:
        """Generate feature interactions.
        
        Args:
            features: Features to generate interactions from
            market_data: Market data for context
            
        Returns:
            DataFrame with interaction features
        """
        interaction_features = []
        interaction_names = []
        
        feature_columns = list(features.columns)
        
        # Generate pairwise interactions (placeholder implementation)
        for i, col1 in enumerate(feature_columns):
            for j, col2 in enumerate(feature_columns[i+1:], i+1):
                try:
                    # Create different types of interactions
                    interaction_name = f"{col1}_{col2}_mult"
                    interaction_feature = features[col1] * features[col2]
                    interaction_features.append(interaction_feature)
                    interaction_names.append(interaction_name)
                    
                    # Ratio interaction
                    if features[col2].abs().min() > 1e-8:  # Avoid division by zero
                        interaction_name = f"{col1}_{col2}_ratio"
                        interaction_feature = features[col1] / features[col2]
                        interaction_features.append(interaction_feature)
                        interaction_names.append(interaction_name)
                    
                    # Difference interaction
                    interaction_name = f"{col1}_{col2}_diff"
                    interaction_feature = features[col1] - features[col2]
                    interaction_features.append(interaction_feature)
                    interaction_names.append(interaction_name)
                    
                except Exception as e:
                    self.logger.error(f"Failed to generate interaction between {col1} and {col2}: {e}")
                    continue
        
        if interaction_features:
            return pd.DataFrame(dict(zip(interaction_names, interaction_features)))
        else:
            return pd.DataFrame()
    
    def _generate_interaction_metadata(self, 
                                     interaction_features: pd.DataFrame,
                                     original_features: pd.DataFrame) -> Dict[str, Any]:
        """Generate interaction metadata.
        
        Args:
            interaction_features: Generated interaction features
            original_features: Original features
            
        Returns:
            Interaction metadata
        """
        return {
            "step_name": "feature_generation_interaction_generation_step",
            "timestamp": datetime.now().isoformat(),
            "original_feature_count": len(original_features.columns),
            "interaction_feature_count": len(interaction_features.columns),
            "interaction_types": self._get_interaction_types(interaction_features),
            "interaction_definitions": self._get_interaction_definitions(interaction_features)
        }
    
    def _get_interaction_types(self, interaction_features: pd.DataFrame) -> Dict[str, int]:
        """Get interaction types and their counts."""
        interaction_types = {"multiplicative": 0, "ratio": 0, "difference": 0}
        
        for col in interaction_features.columns:
            if "_mult" in col:
                interaction_types["multiplicative"] += 1
            elif "_ratio" in col:
                interaction_types["ratio"] += 1
            elif "_diff" in col:
                interaction_types["difference"] += 1
        
        return interaction_types
    
    def _get_interaction_definitions(self, interaction_features: pd.DataFrame) -> Dict[str, str]:
        """Get interaction definitions."""
        definitions = {}
        
        for col in interaction_features.columns:
            if "_mult" in col:
                base_name = col.replace("_mult", "")
                definitions[col] = f"Multiplicative interaction: {base_name}"
            elif "_ratio" in col:
                base_name = col.replace("_ratio", "")
                definitions[col] = f"Ratio interaction: {base_name}"
            elif "_diff" in col:
                base_name = col.replace("_diff", "")
                definitions[col] = f"Difference interaction: {base_name}"
        
        return definitions
    
    def _calculate_generation_metrics(self, 
                                    interaction_features: pd.DataFrame,
                                    original_features: pd.DataFrame) -> Dict[str, Any]:
        """Calculate generation metrics."""
        return {
            "original_feature_count": len(original_features.columns),
            "interaction_feature_count": len(interaction_features.columns),
            "interaction_ratio": len(interaction_features.columns) / len(original_features.columns) if len(original_features.columns) > 0 else 0.0,
            "generation_timestamp": datetime.now().isoformat()
        }
    
    async def _save_artifacts(self,
                             artifacts_path: Path,
                             interaction_features: pd.DataFrame,
                             interaction_metadata: Dict[str, Any],
                             generation_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Save interaction generation artifacts."""
        artifacts = {}
        
        # Save interaction features
        if not interaction_features.empty:
            interaction_features_path = artifacts_path / "interaction_features.parquet"
            interaction_features.to_parquet(interaction_features_path)
            artifacts["interaction_features"] = str(interaction_features_path)
        
        # Save interaction metadata
        interaction_metadata_path = artifacts_path / "interaction_metadata.json"
        with open(interaction_metadata_path, 'w') as f:
            json.dump(interaction_metadata, f, indent=2)
        artifacts["interaction_metadata"] = str(interaction_metadata_path)
        
        # Save generation metrics
        generation_metrics_path = artifacts_path / "generation_metrics.json"
        with open(generation_metrics_path, 'w') as f:
            json.dump(generation_metrics, f, indent=2)
        artifacts["generation_metrics"] = str(generation_metrics_path)
        
        # Generate human-readable report
        report_path = await self._generate_human_readable_report(artifacts_path, interaction_features, interaction_metadata, generation_metrics)
        if report_path:
            artifacts["human_readable_report"] = str(report_path)
        
        return artifacts
    
    async def _generate_human_readable_report(self,
                                            artifacts_path: Path,
                                            interaction_features: pd.DataFrame,
                                            interaction_metadata: Dict[str, Any],
                                            generation_metrics: Dict[str, Any]) -> Optional[Path]:
        """Generate human-readable report in outcomes/ directory.
        
        Args:
            artifacts_path: Path to save artifacts
            interaction_features: Generated interaction features
            interaction_metadata: Interaction metadata
            generation_metrics: Generation metrics
            
        Returns:
            Path to the generated report file
        """
        # Create outcomes directory
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(exist_ok=True)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"interaction_generation_report_{timestamp}.md"
        report_path = outcomes_dir / report_filename
        
        # Calculate interaction statistics
        total_interactions = len(interaction_features.columns)
        memory_usage_mb = interaction_features.memory_usage(deep=True).sum() / 1024 / 1024 if not interaction_features.empty else 0
        
        # Generate report content
        report_content = f"""# Interaction Generation Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
- **Interaction Features Generated**: {total_interactions}
- **Data Shape**: {interaction_features.shape[0]} rows × {interaction_features.shape[1]} columns
- **Memory Usage**: {memory_usage_mb:.2f} MB
- **Generation Status**: {'✅ SUCCESS' if total_interactions > 0 else '❌ FAILED'}

## Interaction Types
- **Multiplicative**: {interaction_metadata.get('interaction_types', {}).get('multiplicative', 0)}
- **Ratio**: {interaction_metadata.get('interaction_types', {}).get('ratio', 0)}
- **Difference**: {interaction_metadata.get('interaction_types', {}).get('difference', 0)}

## Generation Metrics
- **Original Features**: {generation_metrics.get('original_feature_count', 0)}
- **Interaction Features**: {generation_metrics.get('interaction_feature_count', 0)}
- **Interaction Ratio**: {generation_metrics.get('interaction_ratio', 0.0):.2f}
- **Generation Timestamp**: {generation_metrics.get('generation_timestamp', 'N/A')}

## Sample Interactions
"""
        
        # Add sample interaction definitions
        interaction_definitions = interaction_metadata.get('interaction_definitions', {})
        sample_interactions = list(interaction_definitions.items())[:10]
        for feature, definition in sample_interactions:
            report_content += f"- **{feature}**: {definition}
"
        
        if len(interaction_definitions) > 10:
            report_content += f"... and {len(interaction_definitions) - 10} more interactions
"
        
        report_content += f"""

## Feature Overview
- **Feature Names**: {', '.join(interaction_features.columns[:10])}{'...' if len(interaction_features.columns) > 10 else ''}
- **Data Types**: {dict(interaction_features.dtypes.value_counts()) if not interaction_features.empty else 'N/A'}
- **Missing Values**: {interaction_features.isnull().sum().sum() if not interaction_features.empty else 0}

## Quality Assessment
- **Feature Completeness**: {((interaction_features.count().sum() / (interaction_features.shape[0] * interaction_features.shape[1])) * 100):.2f}% if not interaction_features.empty else 'N/A'}
- **Memory Efficiency**: {'Good' if memory_usage_mb < 100 else 'High' if memory_usage_mb < 500 else 'Very High'}
- **Interaction Diversity**: {len(set(interaction_features.dtypes))} different data types

## Next Steps
1. Review generated interactions for relevance
2. Proceed to vectorization step
3. Consider interaction filtering if needed

---
*Report generated by Feature Generation Interaction Generation Step*
"""
        
        # Write report
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        self.logger.info(f"📊 Human-readable report saved: {report_path}")
        
        return report_path


# Command handler for ares_launcher integration
async def handle_feature_generation_interaction_generation_step(
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
) -> InteractionGenerationResult:
    """Handle feature_generation_interaction_generation_step command."""
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
    step = FeatureGenerationInteractionGenerationStep(config)
    
    # Load market data (placeholder)
    market_data = pd.DataFrame()
    
    # Execute step
    return await step.execute(market_data, "artifacts")