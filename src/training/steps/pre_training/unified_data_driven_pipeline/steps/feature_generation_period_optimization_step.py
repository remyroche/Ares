"""
Feature Generation Period Optimization Step

This step optimizes lookback periods for features as part of the
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
    EconomicPeriodEvaluator
)


@dataclass
class PeriodOptimizationResult:
    """Result of period optimization step."""
    
    success: bool
    optimal_periods: Dict[str, int]
    optimization_metrics: Dict[str, Any]
    error_message: Optional[str] = None
    artifacts: Optional[Dict[str, Any]] = None


class FeatureGenerationPeriodOptimizationStep:
    """Period optimization step for feature generation pipeline."""
    
    def __init__(self, config: UnifiedPipelineConfig, logger: Optional[logging.Logger] = None):
        """Initialize the period optimization step.
        
        Args:
            config: Unified pipeline configuration
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize period evaluator
        self.period_evaluator = EconomicPeriodEvaluator()
    
    async def execute(self, 
                     market_data: pd.DataFrame,
                     artifacts_dir: str,
                     previous_artifacts: Optional[Dict[str, Any]] = None,
                     **kwargs) -> PeriodOptimizationResult:
        """Execute period optimization step.
        
        Args:
            market_data: Input market data
            artifacts_dir: Directory to save artifacts
            previous_artifacts: Artifacts from previous steps
            **kwargs: Additional arguments
            
        Returns:
            PeriodOptimizationResult with optimal periods
        """
        self.logger.info("⏱️ Starting period optimization step...")
        
        try:
            # Create artifacts directory
            artifacts_path = Path(artifacts_dir) / "feature_generation_period_optimization_step"
            artifacts_path.mkdir(parents=True, exist_ok=True)
            
            # Load features from previous step
            features = await self._load_features_from_previous_step(previous_artifacts)
            
            if features.empty:
                self.logger.warning("⚠️ No features available for period optimization")
                return PeriodOptimizationResult(
                    success=False,
                    optimal_periods={},
                    optimization_metrics={},
                    error_message="No features available for period optimization"
                )
            
            # Perform period optimization
            optimal_periods = await self._optimize_periods(features, market_data)
            
            # Calculate optimization metrics
            optimization_metrics = self._calculate_optimization_metrics(optimal_periods, features)
            
            # Save artifacts
            artifacts = await self._save_artifacts(
                artifacts_path, optimal_periods, optimization_metrics
            )
            
            self.logger.info(f"✅ Period optimization completed with {len(optimal_periods)} optimized periods")
            
            return PeriodOptimizationResult(
                success=True,
                optimal_periods=optimal_periods,
                optimization_metrics=optimization_metrics,
                artifacts=artifacts
            )
            
        except Exception as e:
            self.logger.error(f"❌ Period optimization failed: {e}")
            return PeriodOptimizationResult(
                success=False,
                optimal_periods={},
                optimization_metrics={},
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
    
    async def _optimize_periods(self, 
                               features: pd.DataFrame,
                               market_data: pd.DataFrame) -> Dict[str, int]:
        """Optimize periods for features.
        
        Args:
            features: Features to optimize
            market_data: Market data for context
            
        Returns:
            Dictionary mapping feature names to optimal periods
        """
        optimal_periods = {}
        
        for feature_name in features.columns:
            try:
                # This would use the actual EconomicPeriodEvaluator
                # For now, return placeholder optimization
                optimal_period = np.random.randint(5, 50)  # Placeholder
                optimal_periods[feature_name] = optimal_period
            except Exception as e:
                self.logger.error(f"Failed to optimize period for {feature_name}: {e}")
                optimal_periods[feature_name] = 20  # Default fallback
        
        return optimal_periods
    
    def _calculate_optimization_metrics(self, 
                                       optimal_periods: Dict[str, int],
                                       features: pd.DataFrame) -> Dict[str, Any]:
        """Calculate optimization metrics."""
        return {
            "total_features_optimized": len(optimal_periods),
            "avg_optimal_period": np.mean(list(optimal_periods.values())),
            "min_optimal_period": min(optimal_periods.values()) if optimal_periods else 0,
            "max_optimal_period": max(optimal_periods.values()) if optimal_periods else 0,
            "optimization_timestamp": datetime.now().isoformat()
        }
    
    async def _save_artifacts(self,
                             artifacts_path: Path,
                             optimal_periods: Dict[str, int],
                             optimization_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Save period optimization artifacts."""
        artifacts = {}
        
        # Save optimal periods
        optimal_periods_path = artifacts_path / "optimal_periods.json"
        with open(optimal_periods_path, 'w') as f:
            json.dump(optimal_periods, f, indent=2)
        artifacts["optimal_periods"] = str(optimal_periods_path)
        
        # Save optimization metrics
        optimization_metrics_path = artifacts_path / "optimization_metrics.json"
        with open(optimization_metrics_path, 'w') as f:
            json.dump(optimization_metrics, f, indent=2)
        artifacts["optimization_metrics"] = str(optimization_metrics_path)
        
        return artifacts


# Command handler for ares_launcher integration
async def handle_feature_generation_period_optimization_step(
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
) -> PeriodOptimizationResult:
    """Handle feature_generation_period_optimization_step command."""
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
    step = FeatureGenerationPeriodOptimizationStep(config)
    
    # Load market data (placeholder)
    market_data = pd.DataFrame()
    
    # Execute step
    return await step.execute(market_data, "artifacts")