"""
Feature Generation Vectorization Step

This step optimizes feature vectorization for performance as part of the
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
class VectorizationResult:
    """Result of vectorization step."""
    
    success: bool
    vectorized_features: pd.DataFrame
    vectorization_metadata: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    error_message: Optional[str] = None
    artifacts: Optional[Dict[str, Any]] = None


class FeatureGenerationVectorizationStep:
    """Vectorization step for feature generation pipeline."""
    
    def __init__(self, config: UnifiedPipelineConfig, logger: Optional[logging.Logger] = None):
        """Initialize the vectorization step.
        
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
                     **kwargs) -> VectorizationResult:
        """Execute vectorization step.
        
        Args:
            market_data: Input market data
            artifacts_dir: Directory to save artifacts
            previous_artifacts: Artifacts from previous steps
            **kwargs: Additional arguments
            
        Returns:
            VectorizationResult with vectorized features
        """
        self.logger.info("⚡ Starting vectorization step...")
        
        try:
            # Create artifacts directory
            artifacts_path = Path(artifacts_dir) / "feature_generation_vectorization_step"
            artifacts_path.mkdir(parents=True, exist_ok=True)
            
            # Load features from previous steps
            individual_features = await self._load_individual_features(previous_artifacts)
            interaction_features = await self._load_interaction_features(previous_artifacts)
            
            if individual_features.empty and interaction_features.empty:
                self.logger.warning("⚠️ No features available for vectorization")
                return VectorizationResult(
                    success=False,
                    vectorized_features=pd.DataFrame(),
                    vectorization_metadata={},
                    performance_metrics={},
                    error_message="No features available for vectorization"
                )
            
            # Combine and vectorize features
            vectorized_features = await self._vectorize_features(individual_features, interaction_features)
            
            # Generate vectorization metadata
            vectorization_metadata = self._generate_vectorization_metadata(
                vectorized_features, individual_features, interaction_features
            )
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                vectorized_features, individual_features, interaction_features
            )
            
            # Save artifacts
            artifacts = await self._save_artifacts(
                artifacts_path, vectorized_features, vectorization_metadata, performance_metrics
            )
            
            self.logger.info(f"✅ Vectorization completed with {len(vectorized_features.columns)} vectorized features")
            
            return VectorizationResult(
                success=True,
                vectorized_features=vectorized_features,
                vectorization_metadata=vectorization_metadata,
                performance_metrics=performance_metrics,
                artifacts=artifacts
            )
            
        except Exception as e:
            self.logger.error(f"❌ Vectorization failed: {e}")
            return VectorizationResult(
                success=False,
                vectorized_features=pd.DataFrame(),
                vectorization_metadata={},
                performance_metrics={},
                error_message=str(e)
            )
    
    async def _load_individual_features(self, 
                                       previous_artifacts: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """Load individual features from previous step artifacts."""
        if not previous_artifacts or "selected_features" not in previous_artifacts:
            return pd.DataFrame()
        
        try:
            features_path = previous_artifacts["selected_features"]
            return pd.read_parquet(features_path)
        except Exception as e:
            self.logger.error(f"Failed to load individual features: {e}")
            return pd.DataFrame()
    
    async def _load_interaction_features(self, 
                                        previous_artifacts: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """Load interaction features from previous step artifacts."""
        if not previous_artifacts or "interaction_features" not in previous_artifacts:
            return pd.DataFrame()
        
        try:
            interaction_path = previous_artifacts["interaction_features"]
            return pd.read_parquet(interaction_path)
        except Exception as e:
            self.logger.error(f"Failed to load interaction features: {e}")
            return pd.DataFrame()
    
    async def _vectorize_features(self, 
                                 individual_features: pd.DataFrame,
                                 interaction_features: pd.DataFrame) -> pd.DataFrame:
        """Vectorize features for optimal performance.
        
        Args:
            individual_features: Individual features
            interaction_features: Interaction features
            
        Returns:
            DataFrame with vectorized features
        """
        vectorized_features = []
        
        # Process individual features
        if not individual_features.empty:
            # Optimize data types for memory efficiency
            individual_optimized = individual_features.copy()
            for col in individual_optimized.columns:
                if individual_optimized[col].dtype == 'float64':
                    individual_optimized[col] = individual_optimized[col].astype('float32')
                elif individual_optimized[col].dtype == 'int64':
                    individual_optimized[col] = individual_optimized[col].astype('int32')
            
            vectorized_features.append(individual_optimized)
        
        # Process interaction features
        if not interaction_features.empty:
            # Optimize data types for memory efficiency
            interaction_optimized = interaction_features.copy()
            for col in interaction_optimized.columns:
                if interaction_optimized[col].dtype == 'float64':
                    interaction_optimized[col] = interaction_optimized[col].astype('float32')
                elif interaction_optimized[col].dtype == 'int64':
                    interaction_optimized[col] = interaction_optimized[col].astype('int32')
            
            vectorized_features.append(interaction_optimized)
        
        # Combine all features
        if vectorized_features:
            return pd.concat(vectorized_features, axis=1)
        else:
            return pd.DataFrame()
    
    def _generate_vectorization_metadata(self, 
                                        vectorized_features: pd.DataFrame,
                                        individual_features: pd.DataFrame,
                                        interaction_features: pd.DataFrame) -> Dict[str, Any]:
        """Generate vectorization metadata.
        
        Args:
            vectorized_features: Vectorized features
            individual_features: Individual features
            interaction_features: Interaction features
            
        Returns:
            Vectorization metadata
        """
        return {
            "step_name": "feature_generation_vectorization_step",
            "timestamp": datetime.now().isoformat(),
            "individual_feature_count": len(individual_features.columns),
            "interaction_feature_count": len(interaction_features.columns),
            "total_vectorized_features": len(vectorized_features.columns),
            "memory_usage_mb": vectorized_features.memory_usage(deep=True).sum() / 1024 / 1024,
            "data_types": vectorized_features.dtypes.to_dict()
        }
    
    def _calculate_performance_metrics(self, 
                                     vectorized_features: pd.DataFrame,
                                     individual_features: pd.DataFrame,
                                     interaction_features: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance metrics."""
        return {
            "vectorization_timestamp": datetime.now().isoformat(),
            "memory_usage_mb": vectorized_features.memory_usage(deep=True).sum() / 1024 / 1024,
            "feature_count": len(vectorized_features.columns),
            "row_count": len(vectorized_features),
            "optimization_applied": True
        }
    
    async def _save_artifacts(self,
                             artifacts_path: Path,
                             vectorized_features: pd.DataFrame,
                             vectorization_metadata: Dict[str, Any],
                             performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Save vectorization artifacts."""
        artifacts = {}
        
        # Save vectorized features
        if not vectorized_features.empty:
            vectorized_features_path = artifacts_path / "vectorized_features.parquet"
            vectorized_features.to_parquet(vectorized_features_path)
            artifacts["vectorized_features"] = str(vectorized_features_path)
        
        # Save vectorization metadata
        vectorization_metadata_path = artifacts_path / "vectorization_metadata.json"
        with open(vectorization_metadata_path, 'w') as f:
            json.dump(vectorization_metadata, f, indent=2)
        artifacts["vectorization_metadata"] = str(vectorization_metadata_path)
        
        # Save performance metrics
        performance_metrics_path = artifacts_path / "performance_metrics.json"
        with open(performance_metrics_path, 'w') as f:
            json.dump(performance_metrics, f, indent=2)
        artifacts["performance_metrics"] = str(performance_metrics_path)
        
        # Generate human-readable report
        await self._generate_human_readable_report(artifacts_path, vectorized_features, vectorization_metadata, performance_metrics)
        
        return artifacts
    
    async def _generate_human_readable_report(self,
                                            artifacts_path: Path,
                                            vectorized_features: pd.DataFrame,
                                            vectorization_metadata: Dict[str, Any],
                                            performance_metrics: Dict[str, Any]) -> None:
        """Generate human-readable report in outcomes/ directory.
        
        Args:
            artifacts_path: Path to save artifacts
            vectorized_features: Vectorized features
            vectorization_metadata: Vectorization metadata
            performance_metrics: Performance metrics
        """
        # Create outcomes directory
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(exist_ok=True)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"vectorization_report_{timestamp}.md"
        report_path = outcomes_dir / report_filename
        
        # Calculate vectorization statistics
        total_features = len(vectorized_features.columns)
        memory_usage_mb = vectorized_features.memory_usage(deep=True).sum() / 1024 / 1024 if not vectorized_features.empty else 0
        
        # Generate report content
        report_content = f"""# Vectorization Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
- **Vectorized Features**: {total_features}
- **Data Shape**: {vectorized_features.shape[0]} rows × {vectorized_features.shape[1]} columns
- **Memory Usage**: {memory_usage_mb:.2f} MB
- **Vectorization Status**: {'✅ SUCCESS' if total_features > 0 else '❌ FAILED'}

## Vectorization Metrics
- **Individual Features**: {vectorization_metadata.get('individual_feature_count', 0)}
- **Interaction Features**: {vectorization_metadata.get('interaction_feature_count', 0)}
- **Total Vectorized Features**: {vectorization_metadata.get('total_vectorized_features', 0)}
- **Memory Usage**: {vectorization_metadata.get('memory_usage_mb', 0.0):.2f} MB

## Data Types
"""
        
        # Add data type information
        data_types = vectorization_metadata.get('data_types', {})
        for dtype, count in data_types.items():
            report_content += f"- **{dtype}**: {count} features
"
        
        report_content += f"""

## Performance Metrics
- **Vectorization Timestamp**: {performance_metrics.get('vectorization_timestamp', 'N/A')}
- **Memory Usage**: {performance_metrics.get('memory_usage_mb', 0.0):.2f} MB
- **Feature Count**: {performance_metrics.get('feature_count', 0)}
- **Row Count**: {performance_metrics.get('row_count', 0)}
- **Optimization Applied**: {performance_metrics.get('optimization_applied', False)}

## Feature Overview
- **Feature Names**: {', '.join(vectorized_features.columns[:10])}{'...' if len(vectorized_features.columns) > 10 else ''}
- **Data Types**: {dict(vectorized_features.dtypes.value_counts()) if not vectorized_features.empty else 'N/A'}
- **Missing Values**: {vectorized_features.isnull().sum().sum() if not vectorized_features.empty else 0}

## Quality Assessment
- **Feature Completeness**: {((vectorized_features.count().sum() / (vectorized_features.shape[0] * vectorized_features.shape[1])) * 100):.2f}% if not vectorized_features.empty else 'N/A'}
- **Memory Efficiency**: {'Excellent' if memory_usage_mb < 50 else 'Good' if memory_usage_mb < 200 else 'High' if memory_usage_mb < 500 else 'Very High'}
- **Data Type Optimization**: {'Optimized' if performance_metrics.get('optimization_applied', False) else 'Not Optimized'}

## Next Steps
1. Review vectorized features for quality
2. Proceed to labeling integration step
3. Consider additional optimizations if needed

---
*Report generated by Feature Generation Vectorization Step*
"""
        
        # Write report
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        self.logger.info(f"📊 Human-readable report saved: {report_path}")


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
    """Handle feature_generation_vectorization_step command."""
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
    step = FeatureGenerationVectorizationStep(config)
    
    # Load market data (placeholder)
    market_data = pd.DataFrame()
    
    # Execute step
    return await step.execute(market_data, "artifacts")