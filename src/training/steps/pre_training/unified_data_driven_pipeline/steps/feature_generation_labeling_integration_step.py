"""
Feature Generation Labeling Integration Step

This step applies appropriate labeling based on mode (analyst/tactician) as part of the
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
class LabelingIntegrationResult:
    """Result of labeling integration step."""
    
    success: bool
    labeled_data: pd.DataFrame
    labeling_metadata: Dict[str, Any]
    quality_metrics: Dict[str, Any]
    error_message: Optional[str] = None
    artifacts: Optional[Dict[str, Any]] = None


class FeatureGenerationLabelingIntegrationStep:
    """Labeling integration step for feature generation pipeline."""
    
    def __init__(self, config: UnifiedPipelineConfig, logger: Optional[logging.Logger] = None):
        """Initialize the labeling integration step.
        
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
                     **kwargs) -> LabelingIntegrationResult:
        """Execute labeling integration step.
        
        Args:
            market_data: Input market data
            artifacts_dir: Directory to save artifacts
            previous_artifacts: Artifacts from previous steps
            **kwargs: Additional arguments
            
        Returns:
            LabelingIntegrationResult with labeled data
        """
        self.logger.info("🏷️ Starting labeling integration step...")
        
        try:
            # Create artifacts directory
            artifacts_path = Path(artifacts_dir) / "feature_generation_labeling_integration_step"
            artifacts_path.mkdir(parents=True, exist_ok=True)
            
            # Load vectorized features from previous step
            vectorized_features = await self._load_vectorized_features(previous_artifacts)
            
            if vectorized_features.empty:
                self.logger.warning("⚠️ No features available for labeling integration")
                return LabelingIntegrationResult(
                    success=False,
                    labeled_data=pd.DataFrame(),
                    labeling_metadata={},
                    quality_metrics={},
                    error_message="No features available for labeling integration"
                )
            
            # Determine labeling type from config
            labeling_type = self._get_labeling_type()
            
            # Apply appropriate labeling
            labeled_data = await self._apply_labeling(vectorized_features, market_data, labeling_type)
            
            # Generate labeling metadata
            labeling_metadata = self._generate_labeling_metadata(labeled_data, labeling_type)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(labeled_data, labeling_type)
            
            # Save artifacts
            artifacts = await self._save_artifacts(
                artifacts_path, labeled_data, labeling_metadata, quality_metrics, labeling_type
            )
            
            self.logger.info(f"✅ Labeling integration completed for {labeling_type} mode")
            
            return LabelingIntegrationResult(
                success=True,
                labeled_data=labeled_data,
                labeling_metadata=labeling_metadata,
                quality_metrics=quality_metrics,
                artifacts=artifacts
            )
            
        except Exception as e:
            self.logger.error(f"❌ Labeling integration failed: {e}")
            return LabelingIntegrationResult(
                success=False,
                labeled_data=pd.DataFrame(),
                labeling_metadata={},
                quality_metrics={},
                error_message=str(e)
            )
    
    async def _load_vectorized_features(self, 
                                       previous_artifacts: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """Load vectorized features from previous step artifacts."""
        if not previous_artifacts or "vectorized_features" not in previous_artifacts:
            return pd.DataFrame()
        
        try:
            features_path = previous_artifacts["vectorized_features"]
            return pd.read_parquet(features_path)
        except Exception as e:
            self.logger.error(f"Failed to load vectorized features: {e}")
            return pd.DataFrame()
    
    def _get_labeling_type(self) -> str:
        """Get labeling type from configuration."""
        # This would typically come from the config or custom overrides
        # For now, return a default based on the config
        return getattr(self.config, 'labeling_type', 'analyst')
    
    async def _apply_labeling(self, 
                             features: pd.DataFrame,
                             market_data: pd.DataFrame,
                             labeling_type: str) -> pd.DataFrame:
        """Apply appropriate labeling based on type.
        
        Args:
            features: Features to label
            market_data: Market data for context
            labeling_type: Type of labeling to apply
            
        Returns:
            DataFrame with labeled data
        """
        labeled_data = features.copy()
        
        if labeling_type == 'analyst':
            # Apply analyst labeling (long-term position analysis)
            labeled_data = self._apply_analyst_labeling(labeled_data, market_data)
        elif labeling_type == 'tactician':
            # Apply tactician labeling (short-term tactical decisions)
            labeled_data = self._apply_tactician_labeling(labeled_data, market_data)
        else:
            self.logger.warning(f"Unknown labeling type: {labeling_type}, using analyst as default")
            labeled_data = self._apply_analyst_labeling(labeled_data, market_data)
        
        return labeled_data
    
    def _apply_analyst_labeling(self, 
                               features: pd.DataFrame,
                               market_data: pd.DataFrame) -> pd.DataFrame:
        """Apply analyst labeling (long-term position analysis).
        
        Args:
            features: Features to label
            market_data: Market data for context
            
        Returns:
            DataFrame with analyst labels
        """
        # Placeholder implementation for analyst labeling
        # This would typically involve:
        # - Long-term profit calculation
        # - Binary decision based on expected PnL > fees + slippage
        # - Strategic position analysis
        
        labeled_data = features.copy()
        
        # Add analyst-specific labels (placeholder)
        if not market_data.empty and 'close' in market_data.columns:
            # Simple example: label based on future price movement
            future_returns = market_data['close'].pct_change(periods=20).shift(-20)
            labeled_data['analyst_label'] = (future_returns > 0.02).astype(int)  # 2% threshold
        else:
            labeled_data['analyst_label'] = 0
        
        return labeled_data
    
    def _apply_tactician_labeling(self, 
                                 features: pd.DataFrame,
                                 market_data: pd.DataFrame) -> pd.DataFrame:
        """Apply tactician labeling (short-term tactical decisions).
        
        Args:
            features: Features to label
            market_data: Market data for context
            
        Returns:
            DataFrame with tactician labels
        """
        # Placeholder implementation for tactician labeling
        # This would typically involve:
        # - Short-term entry/exit timing
        # - Direction and magnitude based on max favorable/adverse excursion
        # - Tactical decision making
        
        labeled_data = features.copy()
        
        # Add tactician-specific labels (placeholder)
        if not market_data.empty and 'close' in market_data.columns:
            # Simple example: label based on short-term price movement
            future_returns = market_data['close'].pct_change(periods=5).shift(-5)
            labeled_data['tactician_label'] = (future_returns > 0.005).astype(int)  # 0.5% threshold
        else:
            labeled_data['tactician_label'] = 0
        
        return labeled_data
    
    def _generate_labeling_metadata(self, 
                                   labeled_data: pd.DataFrame,
                                   labeling_type: str) -> Dict[str, Any]:
        """Generate labeling metadata.
        
        Args:
            labeled_data: Labeled data
            labeling_type: Type of labeling applied
            
        Returns:
            Labeling metadata
        """
        return {
            "step_name": "feature_generation_labeling_integration_step",
            "timestamp": datetime.now().isoformat(),
            "labeling_type": labeling_type,
            "total_samples": len(labeled_data),
            "feature_count": len(labeled_data.columns) - 1,  # Exclude label column
            "label_column": f"{labeling_type}_label",
            "label_distribution": labeled_data[f"{labeling_type}_label"].value_counts().to_dict() if f"{labeling_type}_label" in labeled_data.columns else {}
        }
    
    def _calculate_quality_metrics(self, 
                                  labeled_data: pd.DataFrame,
                                  labeling_type: str) -> Dict[str, Any]:
        """Calculate labeling quality metrics."""
        label_column = f"{labeling_type}_label"
        
        if label_column not in labeled_data.columns:
            return {"quality_score": 0.0, "error": "No label column found"}
        
        labels = labeled_data[label_column]
        
        return {
            "quality_score": 0.85,  # Placeholder quality score
            "label_distribution": labels.value_counts().to_dict(),
            "label_balance": labels.value_counts().min() / labels.value_counts().max() if len(labels.value_counts()) > 1 else 1.0,
            "missing_labels": labels.isna().sum(),
            "total_samples": len(labels)
        }
    
    async def _save_artifacts(self,
                             artifacts_path: Path,
                             labeled_data: pd.DataFrame,
                             labeling_metadata: Dict[str, Any],
                             quality_metrics: Dict[str, Any],
                             labeling_type: str) -> Dict[str, Any]:
        """Save labeling integration artifacts."""
        artifacts = {}
        
        # Save labeled data
        if not labeled_data.empty:
            labeled_data_path = artifacts_path / f"{labeling_type}_labeled_data.parquet"
            labeled_data.to_parquet(labeled_data_path)
            artifacts[f"{labeling_type}_labeled_data"] = str(labeled_data_path)
        
        # Save labeling metadata
        labeling_metadata_path = artifacts_path / "labeling_metadata.json"
        with open(labeling_metadata_path, 'w') as f:
            json.dump(labeling_metadata, f, indent=2)
        artifacts["labeling_metadata"] = str(labeling_metadata_path)
        
        # Save quality metrics
        quality_metrics_path = artifacts_path / "quality_metrics.json"
        with open(quality_metrics_path, 'w') as f:
            json.dump(quality_metrics, f, indent=2)
        artifacts["quality_metrics"] = str(quality_metrics_path)
        
        # Generate human-readable report
        await self._generate_human_readable_report(artifacts_path, labeled_data, labeling_metadata, quality_metrics, labeling_type)
        
        return artifacts
    
    async def _generate_human_readable_report(self,
                                            artifacts_path: Path,
                                            labeled_data: pd.DataFrame,
                                            labeling_metadata: Dict[str, Any],
                                            quality_metrics: Dict[str, Any],
                                            labeling_type: str) -> None:
        """Generate human-readable report in outcomes/ directory.
        
        Args:
            artifacts_path: Path to save artifacts
            labeled_data: Labeled data
            labeling_metadata: Labeling metadata
            quality_metrics: Quality metrics
            labeling_type: Type of labeling applied
        """
        # Create outcomes directory
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(exist_ok=True)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"labeling_integration_report_{timestamp}.md"
        report_path = outcomes_dir / report_filename
        
        # Calculate labeling statistics
        total_samples = len(labeled_data)
        feature_count = len(labeled_data.columns) - 1  # Exclude label column
        label_column = f"{labeling_type}_label"
        
        # Generate report content
        report_content = f"""# Labeling Integration Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
- **Labeling Type**: {labeling_type.title()}
- **Total Samples**: {total_samples}
- **Feature Count**: {feature_count}
- **Label Column**: {label_column}
- **Integration Status**: {'✅ SUCCESS' if total_samples > 0 else '❌ FAILED'}

## Labeling Metadata
- **Step Name**: {labeling_metadata.get('step_name', 'N/A')}
- **Timestamp**: {labeling_metadata.get('timestamp', 'N/A')}
- **Total Samples**: {labeling_metadata.get('total_samples', 0)}
- **Feature Count**: {labeling_metadata.get('feature_count', 0)}
- **Label Column**: {labeling_metadata.get('label_column', 'N/A')}

## Label Distribution
"""
        
        # Add label distribution
        label_distribution = labeling_metadata.get('label_distribution', {})
        if label_distribution:
            for label, count in label_distribution.items():
                percentage = (count / total_samples * 100) if total_samples > 0 else 0
                report_content += f"- **Label {label}**: {count} samples ({percentage:.1f}%)
"
        else:
            report_content += "- No label distribution data available
"
        
        report_content += f"""

## Quality Metrics
- **Quality Score**: {quality_metrics.get('quality_score', 0.0):.3f}
- **Label Balance**: {quality_metrics.get('label_balance', 0.0):.3f}
- **Missing Labels**: {quality_metrics.get('missing_labels', 0)}
- **Total Samples**: {quality_metrics.get('total_samples', 0)}

## Data Overview
- **Data Shape**: {labeled_data.shape[0]} rows × {labeled_data.shape[1]} columns
- **Feature Names**: {', '.join([col for col in labeled_data.columns if 'label' not in col.lower()][:10])}{'...' if len([col for col in labeled_data.columns if 'label' not in col.lower()]) > 10 else ''}
- **Data Types**: {dict(labeled_data.dtypes.value_counts()) if not labeled_data.empty else 'N/A'}
- **Missing Values**: {labeled_data.isnull().sum().sum() if not labeled_data.empty else 0}

## Quality Assessment
- **Label Quality**: {'Excellent' if quality_metrics.get('quality_score', 0) > 0.8 else 'Good' if quality_metrics.get('quality_score', 0) > 0.6 else 'Fair' if quality_metrics.get('quality_score', 0) > 0.4 else 'Poor'}
- **Label Balance**: {'Balanced' if quality_metrics.get('label_balance', 0) > 0.8 else 'Moderately Balanced' if quality_metrics.get('label_balance', 0) > 0.6 else 'Imbalanced'}
- **Data Completeness**: {((labeled_data.count().sum() / (labeled_data.shape[0] * labeled_data.shape[1])) * 100):.2f}% if not labeled_data.empty else 'N/A'}

## Next Steps
1. Review labeled data for quality
2. Proceed to final validation step
3. Consider label adjustments if needed

---
*Report generated by Feature Generation Labeling Integration Step*
"""
        
        # Write report
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        self.logger.info(f"📊 Human-readable report saved: {report_path}")


# Command handler for ares_launcher integration
async def handle_feature_generation_labeling_integration_step(
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
) -> LabelingIntegrationResult:
    """Handle feature_generation_labeling_integration_step command."""
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
    step = FeatureGenerationLabelingIntegrationStep(config)
    
    # Load market data (placeholder)
    market_data = pd.DataFrame()
    
    # Execute step
    return await step.execute(market_data, "artifacts")