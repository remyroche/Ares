"""
Analyst-Tactician Feature Selection Adapter

This adapter provides the interface between Analyst and Tactician training modes:
1. For Analyst (5m): Combines long and short signals without directional differentiation
2. For Tactician (1m): Separates long and short signals for independent optimization

Integrates with existing directional_feature_selection_adapter.py for optimal feature selection.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from src.utils.logger import get_logger
from src.utils.tprint import tprint
from src.utils.math_validation import safe_divide, validate_finite

# Import existing directional adapter
from .directional_feature_selection_adapter import (
    DirectionalFeatureSelectionAdapter, DirectionalFeatureSelectionConfig,
    DirectionalFeatureSelectionResult
)


class TrainingMode(Enum):
    """Training mode for feature selection."""
    ANALYST = "analyst"  # 5m timeframe, combined signals
    TACTICIAN_LONG = "tactician_long"  # 1m timeframe, long-only
    TACTICIAN_SHORT = "tactician_short"  # 1m timeframe, short-only


@dataclass
class AnalystTacticianConfig:
    """Configuration for Analyst-Tactician feature selection."""
    training_mode: TrainingMode
    combined_features_target: int = 80  # Target for combined features (Analyst)
    long_features_target: int = 50      # Target for long-only features (Tactician)
    short_features_target: int = 50     # Target for short-only features (Tactician)

    # Directional weighting for combined mode
    long_weight: float = 0.5
    short_weight: float = 0.5

    # Optimization settings
    enable_directional_optimization: bool = True
    min_feature_quality_score: float = 0.7
    max_correlation_threshold: float = 0.95


class AnalystTacticianFeatureAdapter:
    """
    Adapter for Analyst vs Tactician training data preparation.

    For Analyst (5m): Combines long and short signals for unified training
    For Tactician (1m): Separates signals and trains separate long/short models
    """

    def __init__(self, config: Optional[AnalystTacticianConfig] = None):
        self.config = config or AnalystTacticianConfig(TrainingMode.ANALYST)
        self.logger = get_logger('AnalystTacticianFeatureAdapter')

        # Initialize directional adapter
        self.directional_adapter = DirectionalFeatureSelectionAdapter(
            DirectionalFeatureSelectionConfig(
                target_total_features=self.config.combined_features_target,
                maintain_directional_balance=True,
                min_mutual_info_score=self.config.min_feature_quality_score,
                max_correlation_threshold=self.config.max_correlation_threshold
            )
        )

        self.logger.info(f"🚀 Analyst-Tactician Feature Adapter initialized for {self.config.training_mode.value} mode")

    def prepare_analyst_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for Analyst training (5m timeframe).
        Combines long and short signals without directional differentiation.

        Args:
            data: Input data with directional columns

        Returns:
            DataFrame with combined opportunity scores for unified training
        """
        self.logger.info("🔄 Preparing data for Analyst training (5m) - combining directional signals")

        if self.config.training_mode != TrainingMode.ANALYST:
            self.logger.warning(f"⚠️ Adapter configured for {self.config.training_mode.value}, not ANALYST mode")

        # Create combined opportunity scores
        combined_data = data.copy()

        # Find all directional opportunity columns
        long_cols = [col for col in data.columns if '_long_opportunity' in col or '_long_prob' in col]
        short_cols = [col for col in data.columns if '_short_opportunity' in col or '_short_prob' in col]

        if not long_cols and not short_cols:
            self.logger.warning("⚠️ No directional columns found - using original data structure")
            return data

        # Calculate combined opportunity for each horizon
        horizons = ['immediate', 'short', 'overall']
        for horizon in horizons:
            long_col = f'long_{horizon}_opportunity'
            short_col = f'short_{horizon}_opportunity'

            if long_col in data.columns and short_col in data.columns:
                # Weighted combination based on signal strength
                combined_col = f'{horizon}_opportunity'
                combined_data[combined_col] = (
                    data[long_col] * self.config.long_weight +
                    data[short_col] * self.config.short_weight
                )

                self.logger.info(f"✅ Combined {horizon} opportunities: long={data[long_col].mean():.4f}, short={data[short_col].mean():.4f}")

        # Calculate overall combined opportunity score
        opportunity_cols = [col for col in combined_data.columns if '_opportunity' in col]
        if opportunity_cols:
            combined_data['combined_opportunity_score'] = combined_data[opportunity_cols].mean(axis=1)
            self.logger.info(f"✅ Created combined opportunity score: mean={combined_data['combined_opportunity_score'].mean():.4f}")

        # Remove directional columns to prevent confusion in training
        directional_cols_to_remove = [col for col in combined_data.columns
                                    if '_long_' in col or '_short_' in col]
        combined_data = combined_data.drop(columns=directional_cols_to_remove, errors='ignore')

        self.logger.info(f"✅ Analyst data preparation completed: {len(combined_data)} samples, {len(combined_data.columns)} features")
        return combined_data

    def prepare_tactician_data(self, data: pd.DataFrame, direction: str = 'both') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for Tactician training (1m timeframe).
        Separates long and short signals for independent optimization.

        Args:
            data: Input data with directional columns
            direction: 'long', 'short', or 'both'

        Returns:
            Tuple of (long_data, short_data) or single dataset if direction specified
        """
        self.logger.info(f"🔄 Preparing data for Tactician training (1m) - separating directional signals ({direction})")

        if self.config.training_mode in [TrainingMode.ANALYST]:
            self.logger.warning(f"⚠️ Adapter configured for {self.config.training_mode.value}, not TACTICIAN mode")

        if direction == 'both':
            long_data = self._extract_directional_data(data, 'long')
            short_data = self._extract_directional_data(data, 'short')

            # Apply feature selection for each direction
            if self.config.enable_directional_optimization:
                long_data = self._optimize_directional_features(long_data, 'long')
                short_data = self._optimize_directional_features(short_data, 'short')

            self.logger.info(f"✅ Tactician data preparation completed: Long ({len(long_data)}) and Short ({len(short_data)}) datasets")
            return long_data, short_data

        elif direction == 'long':
            long_data = self._extract_directional_data(data, 'long')
            if self.config.enable_directional_optimization:
                long_data = self._optimize_directional_features(long_data, 'long')
            self.logger.info(f"✅ Tactician long data preparation completed: {len(long_data)} samples")
            return long_data, None

        elif direction == 'short':
            short_data = self._extract_directional_data(data, 'short')
            if self.config.enable_directional_optimization:
                short_data = self._optimize_directional_features(short_data, 'short')
            self.logger.info(f"✅ Tactician short data preparation completed: {len(short_data)} samples")
            return None, short_data

        else:
            raise ValueError(f"Invalid direction: {direction}. Must be 'long', 'short', or 'both'")

    def _extract_directional_data(self, data: pd.DataFrame, direction: str) -> pd.DataFrame:
        """
        Extract directional-specific features from data.

        Args:
            data: Input data
            direction: 'long' or 'short'

        Returns:
            DataFrame with directional-specific features
        """
        directional_data = data.copy()

        # Find directional columns
        if direction == 'long':
            dir_cols = [col for col in data.columns if '_long_' in col]
            # Add base features that are direction-neutral
            base_cols = [col for col in data.columns if '_long_' not in col and '_short_' not in col]
            selected_cols = base_cols + dir_cols
        else:  # short
            dir_cols = [col for col in data.columns if '_short_' in col]
            # Add base features that are direction-neutral
            base_cols = [col for col in data.columns if '_long_' not in col and '_short_' not in col]
            selected_cols = base_cols + dir_cols

        # Select only relevant columns
        available_cols = [col for col in selected_cols if col in data.columns]
        directional_data = directional_data[available_cols]

        # Calculate directional strength metrics
        if direction == 'long':
            opp_cols = [col for col in available_cols if '_opportunity' in col and '_long_' in col]
            if opp_cols:
                directional_data['directional_strength'] = directional_data[opp_cols].mean(axis=1)
        else:
            opp_cols = [col for col in available_cols if '_opportunity' in col and '_short_' in col]
            if opp_cols:
                directional_data['directional_strength'] = directional_data[opp_cols].mean(axis=1)

        return directional_data

    def _optimize_directional_features(self, data: pd.DataFrame, direction: str) -> pd.DataFrame:
        """
        Optimize features for specific direction using the directional adapter.

        Args:
            data: Input feature data
            direction: 'long' or 'short'

        Returns:
            Optimized feature data
        """
        self.logger.info(f"🔧 Optimizing features for {direction} direction")

        try:
            # Create dummy directional result for feature selection
            # In practice, this would come from actual directional optimization
            from .directional_lookback_optimizer import DirectionalOptimizationResult, DirectionalFeatureResult

            # For now, create a simplified optimization result
            dummy_result = DirectionalOptimizationResult()

            # Add all features as directional features with basic metrics
            for col in data.columns:
                if col != 'directional_strength':  # Skip our added column
                    feature_result = DirectionalFeatureResult(
                        feature_name=col,
                        optimal_lookback_period=20,  # Default lookback
                        mutual_info_score=0.5,  # Default score
                        stability_score=0.7,
                        cross_validation_score=0.6,
                        data_quality_score=0.8,
                        sample_count=len(data),
                        convergence_achieved=True
                    )

                    if direction == 'long':
                        dummy_result.long_features[col] = feature_result
                    else:
                        dummy_result.short_features[col] = feature_result

            # Apply feature selection
            selection_config = DirectionalFeatureSelectionConfig(
                target_total_features=getattr(self.config, f'{direction}_features_target', 50),
                maintain_directional_balance=False,  # We're already in directional mode
                min_mutual_info_score=self.config.min_feature_quality_score,
                max_correlation_threshold=self.config.max_correlation_threshold
            )

            selection_result = self.directional_adapter.select_optimal_directional_features(
                dummy_result, data, target_column='directional_strength'
            )

            # Select only the chosen features
            if direction == 'long':
                selected_features = selection_result.selected_long_features
            else:
                selected_features = selection_result.selected_short_features

            # Keep directional_strength if it exists
            if 'directional_strength' in data.columns:
                selected_features.append('directional_strength')

            # Filter data to selected features
            available_features = [f for f in selected_features if f in data.columns]
            optimized_data = data[available_features]

            self.logger.info(f"✅ Feature optimization completed for {direction}: {len(optimized_data.columns)} features selected")
            return optimized_data

        except Exception as e:
            self.logger.warning(f"⚠️ Feature optimization failed for {direction}: {e}, using original data")
            return data

    def create_training_targets(self, data: pd.DataFrame, direction: str = None) -> pd.Series:
        """
        Create appropriate training targets based on mode.

        Args:
            data: Input data
            direction: For Tactician mode, specify 'long' or 'short'

        Returns:
            Series with training targets
        """
        if self.config.training_mode == TrainingMode.ANALYST:
            # Analyst: Use combined opportunity score
            return data.get('combined_opportunity_score', data.iloc[:, -1])

        elif self.config.training_mode in [TrainingMode.TACTICIAN_LONG, TrainingMode.TACTICIAN_SHORT]:
            # Tactician: Use directional-specific targets
            if direction == 'long':
                target_cols = [col for col in data.columns if '_long_opportunity' in col]
            elif direction == 'short':
                target_cols = [col for col in data.columns if '_short_opportunity' in col]
            else:
                target_cols = [col for col in data.columns if '_opportunity' in col]

            if target_cols:
                # Use the strongest directional signal as target
                return data[target_cols].max(axis=1)
            else:
                # Fallback to last column
                return data.iloc[:, -1]

    def get_adapter_summary(self) -> Dict[str, Any]:
        """Get summary of adapter configuration and results."""
        return {
            'training_mode': self.config.training_mode.value,
            'combined_features_target': self.config.combined_features_target,
            'long_features_target': self.config.long_features_target,
            'short_features_target': self.config.short_features_target,
            'directional_optimization_enabled': self.config.enable_directional_optimization,
            'long_weight': self.config.long_weight,
            'short_weight': self.config.short_weight
        }


# Convenience functions for easy integration
def prepare_data_for_analyst(data: pd.DataFrame, config: Optional[AnalystTacticianConfig] = None) -> pd.DataFrame:
    """Prepare data for Analyst training (5m, combined signals)."""
    adapter = AnalystTacticianFeatureAdapter(config)
    return adapter.prepare_analyst_data(data)


def prepare_data_for_tactician(data: pd.DataFrame, direction: str = 'both',
                             config: Optional[AnalystTacticianConfig] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare data for Tactician training (1m, directional separation)."""
    adapter = AnalystTacticianFeatureAdapter(config)
    return adapter.prepare_tactician_data(data, direction)