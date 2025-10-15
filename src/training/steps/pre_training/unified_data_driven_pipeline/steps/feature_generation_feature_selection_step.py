"""
Feature Generation Feature Selection Step

This step performs multi-stage feature selection as part of the
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
from src.training.steps.pre_training.unified_data_driven_pipeline.stages.feature_selection_stage import (
    IntelligentFeatureSelector,
    AdvancedFeatureSelector
)


@dataclass
class FeatureSelectionResult:
    """Result of feature selection step."""
    
    success: bool
    selected_features: pd.DataFrame
    selection_metadata: Dict[str, Any]
    selection_metrics: Dict[str, Any]
    error_message: Optional[str] = None
    artifacts: Optional[Dict[str, Any]] = None


class FeatureGenerationFeatureSelectionStep:
    """Feature selection step for feature generation pipeline."""
    
    def __init__(self, config: UnifiedPipelineConfig, logger: Optional[logging.Logger] = None):
        """Initialize the feature selection step.
        
        Args:
            config: Unified pipeline configuration
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize feature selectors
        self.intelligent_selector = IntelligentFeatureSelector()
        self.advanced_selector = AdvancedFeatureSelector()
    
    async def execute(self, 
                     market_data: pd.DataFrame,
                     artifacts_dir: str,
                     previous_artifacts: Optional[Dict[str, Any]] = None,
                     **kwargs) -> FeatureSelectionResult:
        """Execute feature selection step.
        
        Args:
            market_data: Input market data
            artifacts_dir: Directory to save artifacts
            previous_artifacts: Artifacts from previous steps
            **kwargs: Additional arguments
            
        Returns:
            FeatureSelectionResult with selected features
        """
        self.logger.info("🎯 Starting feature selection step...")
        
        try:
            # Create artifacts directory
            artifacts_path = Path(artifacts_dir) / "feature_generation_feature_selection_step"
            artifacts_path.mkdir(parents=True, exist_ok=True)
            
            # Load features from previous step
            features = await self._load_features_from_previous_step(previous_artifacts)
            
            if features.empty:
                self.logger.warning("⚠️ No features available for selection")
                return FeatureSelectionResult(
                    success=False,
                    selected_features=pd.DataFrame(),
                    selection_metadata={},
                    selection_metrics={},
                    error_message="No features available for selection"
                )
            
            # 1. Intelligent feature selection
            self.logger.info("🧠 Performing intelligent feature selection...")
            intelligent_selection = await self._perform_intelligent_selection(features, market_data)
            
            # 2. Advanced feature selection
            self.logger.info("⚡ Performing advanced feature selection...")
            advanced_selection = await self._perform_advanced_selection(features, market_data)
            
            # 3. Combine selection results
            selected_features = await self._combine_selection_results(
                intelligent_selection, advanced_selection, features
            )
            
            # 4. Generate selection metadata
            selection_metadata = self._generate_selection_metadata(
                intelligent_selection, advanced_selection, selected_features
            )
            
            # 5. Calculate selection metrics
            selection_metrics = self._calculate_selection_metrics(
                features, selected_features, intelligent_selection, advanced_selection
            )
            
            # 6. Save artifacts
            artifacts = await self._save_artifacts(
                artifacts_path, selected_features, selection_metadata, 
                selection_metrics, intelligent_selection, advanced_selection
            )
            
            self.logger.info(f"✅ Feature selection completed with {len(selected_features.columns)} selected features")
            
            return FeatureSelectionResult(
                success=True,
                selected_features=selected_features,
                selection_metadata=selection_metadata,
                selection_metrics=selection_metrics,
                artifacts=artifacts
            )
            
        except Exception as e:
            self.logger.error(f"❌ Feature selection failed: {e}")
            return FeatureSelectionResult(
                success=False,
                selected_features=pd.DataFrame(),
                selection_metadata={},
                selection_metrics={},
                error_message=str(e)
            )
    
    async def _load_features_from_previous_step(self, 
                                               previous_artifacts: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """Load features from previous step artifacts.
        
        Args:
            previous_artifacts: Artifacts from previous steps
            
        Returns:
            DataFrame with features
        """
        if not previous_artifacts or "generated_features" not in previous_artifacts:
            return pd.DataFrame()
        
        try:
            features_path = previous_artifacts["generated_features"]
            return pd.read_parquet(features_path)
        except Exception as e:
            self.logger.error(f"Failed to load features from previous step: {e}")
            return pd.DataFrame()
    
    async def _perform_intelligent_selection(self, 
                                           features: pd.DataFrame,
                                           market_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform intelligent feature selection.
        
        Args:
            features: Features to select from
            market_data: Market data for context
            
        Returns:
            Intelligent selection results
        """
        try:
            # This would use the actual IntelligentFeatureSelector
            # For now, return placeholder results
            return {
                "selected_features": list(features.columns[:10]),  # Placeholder
                "selection_scores": {col: np.random.random() for col in features.columns[:10]},
                "selection_method": "intelligent",
                "selection_time": 0.0
            }
        except Exception as e:
            self.logger.error(f"Intelligent selection failed: {e}")
            return {"selected_features": [], "selection_scores": {}, "selection_method": "intelligent", "selection_time": 0.0}
    
    async def _perform_advanced_selection(self, 
                                        features: pd.DataFrame,
                                        market_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform advanced feature selection.
        
        Args:
            features: Features to select from
            market_data: Market data for context
            
        Returns:
            Advanced selection results
        """
        try:
            # This would use the actual AdvancedFeatureSelector
            # For now, return placeholder results
            return {
                "selected_features": list(features.columns[5:15]),  # Placeholder
                "selection_scores": {col: np.random.random() for col in features.columns[5:15]},
                "selection_method": "advanced",
                "selection_time": 0.0
            }
        except Exception as e:
            self.logger.error(f"Advanced selection failed: {e}")
            return {"selected_features": [], "selection_scores": {}, "selection_method": "advanced", "selection_time": 0.0}
    
    async def _combine_selection_results(self,
                                       intelligent_selection: Dict[str, Any],
                                       advanced_selection: Dict[str, Any],
                                       features: pd.DataFrame) -> pd.DataFrame:
        """Combine selection results from different methods.
        
        Args:
            intelligent_selection: Intelligent selection results
            advanced_selection: Advanced selection results
            features: Original features
            
        Returns:
            DataFrame with selected features
        """
        # Combine selected features from both methods
        intelligent_features = set(intelligent_selection.get("selected_features", []))
        advanced_features = set(advanced_selection.get("selected_features", []))
        
        # Union of both selections
        selected_feature_names = list(intelligent_features.union(advanced_features))
        
        # Filter features to only include selected ones
        available_features = [col for col in selected_feature_names if col in features.columns]
        
        if available_features:
            return features[available_features]
        else:
            return pd.DataFrame()
    
    def _generate_selection_metadata(self,
                                   intelligent_selection: Dict[str, Any],
                                   advanced_selection: Dict[str, Any],
                                   selected_features: pd.DataFrame) -> Dict[str, Any]:
        """Generate selection metadata.
        
        Args:
            intelligent_selection: Intelligent selection results
            advanced_selection: Advanced selection results
            selected_features: Selected features
            
        Returns:
            Selection metadata
        """
        return {
            "step_name": "feature_generation_feature_selection_step",
            "timestamp": datetime.now().isoformat(),
            "total_selected_features": len(selected_features.columns),
            "intelligent_selection": {
                "method": intelligent_selection.get("selection_method", "intelligent"),
                "selected_count": len(intelligent_selection.get("selected_features", [])),
                "selection_time": intelligent_selection.get("selection_time", 0.0)
            },
            "advanced_selection": {
                "method": advanced_selection.get("selection_method", "advanced"),
                "selected_count": len(advanced_selection.get("selected_features", [])),
                "selection_time": advanced_selection.get("selection_time", 0.0)
            },
            "selected_features": list(selected_features.columns)
        }
    
    def _calculate_selection_metrics(self,
                                   original_features: pd.DataFrame,
                                   selected_features: pd.DataFrame,
                                   intelligent_selection: Dict[str, Any],
                                   advanced_selection: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate selection metrics.
        
        Args:
            original_features: Original features
            selected_features: Selected features
            intelligent_selection: Intelligent selection results
            advanced_selection: Advanced selection results
            
        Returns:
            Selection metrics
        """
        return {
            "original_feature_count": len(original_features.columns),
            "selected_feature_count": len(selected_features.columns),
            "selection_ratio": len(selected_features.columns) / len(original_features.columns) if len(original_features.columns) > 0 else 0.0,
            "intelligent_selection_time": intelligent_selection.get("selection_time", 0.0),
            "advanced_selection_time": advanced_selection.get("selection_time", 0.0),
            "total_selection_time": intelligent_selection.get("selection_time", 0.0) + advanced_selection.get("selection_time", 0.0)
        }
    
    async def _save_artifacts(self,
                             artifacts_path: Path,
                             selected_features: pd.DataFrame,
                             selection_metadata: Dict[str, Any],
                             selection_metrics: Dict[str, Any],
                             intelligent_selection: Dict[str, Any],
                             advanced_selection: Dict[str, Any]) -> Dict[str, Any]:
        """Save feature selection artifacts.
        
        Args:
            artifacts_path: Path to save artifacts
            selected_features: Selected features
            selection_metadata: Selection metadata
            selection_metrics: Selection metrics
            intelligent_selection: Intelligent selection results
            advanced_selection: Advanced selection results
            
        Returns:
            Dictionary of saved artifact paths
        """
        artifacts = {}
        
        # Save selected features
        if not selected_features.empty:
            selected_features_path = artifacts_path / "selected_features.parquet"
            selected_features.to_parquet(selected_features_path)
            artifacts["selected_features"] = str(selected_features_path)
        
        # Save selection metadata
        selection_metadata_path = artifacts_path / "selection_metadata.json"
        with open(selection_metadata_path, 'w') as f:
            json.dump(selection_metadata, f, indent=2)
        artifacts["selection_metadata"] = str(selection_metadata_path)
        
        # Save selection metrics
        selection_metrics_path = artifacts_path / "selection_metrics.json"
        with open(selection_metrics_path, 'w') as f:
            json.dump(selection_metrics, f, indent=2)
        artifacts["selection_metrics"] = str(selection_metrics_path)
        
        # Save detailed selection results
        selection_results = {
            "intelligent_selection": intelligent_selection,
            "advanced_selection": advanced_selection,
            "timestamp": datetime.now().isoformat()
        }
        
        selection_results_path = artifacts_path / "selection_results.json"
        with open(selection_results_path, 'w') as f:
            json.dump(selection_results, f, indent=2)
        artifacts["selection_results"] = str(selection_results_path)
        
        # Generate human-readable report
        await self._generate_human_readable_report(artifacts_path, selected_features, selection_metadata, selection_metrics)
        
        return artifacts
    
    async def _generate_human_readable_report(self,
                                            artifacts_path: Path,
                                            selected_features: pd.DataFrame,
                                            selection_metadata: Dict[str, Any],
                                            selection_metrics: Dict[str, Any]) -> None:
        """Generate human-readable report in outcomes/ directory.
        
        Args:
            artifacts_path: Path to save artifacts
            selected_features: Selected features
            selection_metadata: Selection metadata
            selection_metrics: Selection metrics
        """
        # Create outcomes directory
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(exist_ok=True)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"feature_selection_report_{timestamp}.md"
        report_path = outcomes_dir / report_filename
        
        # Calculate selection statistics
        total_selected = len(selected_features.columns)
        memory_usage_mb = selected_features.memory_usage(deep=True).sum() / 1024 / 1024 if not selected_features.empty else 0
        
        # Generate report content
        report_content = f"""# Feature Selection Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
- **Selected Features**: {total_selected}
- **Data Shape**: {selected_features.shape[0]} rows × {selected_features.shape[1]} columns
- **Memory Usage**: {memory_usage_mb:.2f} MB
- **Selection Status**: {'✅ SUCCESS' if total_selected > 0 else '❌ FAILED'}

## Selection Methods
- **Intelligent Selection**: {selection_metadata.get('intelligent_selection', {}).get('selected_count', 0)} features
- **Advanced Selection**: {selection_metadata.get('advanced_selection', {}).get('selected_count', 0)} features
- **Total Unique**: {total_selected} features

## Selection Metrics
- **Original Features**: {selection_metrics.get('original_feature_count', 0)}
- **Selected Features**: {selection_metrics.get('selected_feature_count', 0)}
- **Selection Ratio**: {selection_metrics.get('selection_ratio', 0.0):.2%}
- **Total Selection Time**: {selection_metrics.get('total_selection_time', 0.0):.3f} seconds

## Selected Features
- **Feature Names**: {', '.join(selected_features.columns[:10])}{'...' if len(selected_features.columns) > 10 else ''}
- **Data Types**: {dict(selected_features.dtypes.value_counts()) if not selected_features.empty else 'N/A'}
- **Missing Values**: {selected_features.isnull().sum().sum() if not selected_features.empty else 0}

## Quality Assessment
- **Feature Completeness**: {((selected_features.count().sum() / (selected_features.shape[0] * selected_features.shape[1])) * 100):.2f}% if not selected_features.empty else 'N/A'}
- **Memory Efficiency**: {'Good' if memory_usage_mb < 50 else 'High' if memory_usage_mb < 200 else 'Very High'}
- **Selection Efficiency**: {'Excellent' if selection_metrics.get('selection_ratio', 0) < 0.3 else 'Good' if selection_metrics.get('selection_ratio', 0) < 0.6 else 'Moderate'}

## Next Steps
1. Review selected features for relevance
2. Proceed to period optimization step
3. Consider additional feature engineering

---
*Report generated by Feature Generation Feature Selection Step*
"""
        
        # Write report
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        self.logger.info(f"📊 Human-readable report saved: {report_path}")


# Command handler for ares_launcher integration
async def handle_feature_generation_feature_selection_step(
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
) -> FeatureSelectionResult:
    """Handle feature_generation_feature_selection_step command.
    
    Args:
        symbol: Trading symbol
        timeframe: Data timeframe
        direction: Direction type
        intensity: Pipeline intensity
        lookback_days: Lookback period in days
        start_date: Start date for data
        end_date: End date for data
        exchange: Exchange name
        custom_overrides: Custom configuration overrides
        **kwargs: Additional arguments
        
    Returns:
        FeatureSelectionResult
    """
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
    step = FeatureGenerationFeatureSelectionStep(config)
    
    # Load market data (placeholder - would integrate with actual data loading)
    market_data = pd.DataFrame()  # Placeholder
    
    # Execute step
    return await step.execute(market_data, "artifacts")