"""
Analyst-Tactician Multi-Horizon Profit Labeler Adapter

This adapter provides directional-aware labeling that can:
1. For Analyst: Remove directional differentiation (combined signals)
2. For Tactician: Separate long and short signals for independent optimization

Integrates with existing multi-horizon profit labeler for optimal labeling.
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

# Import existing multi-horizon profit labeler
try:
    from .multi_horizon_profit_labeler import (
        MultiHorizonProfitLabeler, MultiHorizonConfig
    )
    LABELER_AVAILABLE = True
except ImportError as e:
    LABELER_AVAILABLE = False
    tprint(f"⚠️ Multi-horizon profit labeler not available: {e}")


class LabelerTrainingMode(Enum):
    """Training mode for multi-horizon profit labeling."""
    ANALYST = "analyst"  # 5m timeframe, combined signals
    TACTICIAN_LONG = "tactician_long"  # 1m timeframe, long-only
    TACTICIAN_SHORT = "tactician_short"  # 1m timeframe, short-only


@dataclass
class LabelerAnalystTacticianConfig:
    """Configuration for Analyst-Tactician multi-horizon labeling."""
    training_mode: LabelerTrainingMode

    # Directional weighting for combined mode
    long_weight: float = 0.5
    short_weight: float = 0.5

    # Profit targets for different timeframes
    analyst_profit_targets: Dict[str, float] = None
    tactician_profit_targets: Dict[str, float] = None

    # Time horizons for different timeframes
    analyst_time_horizons: Dict[str, int] = None
    tactician_time_horizons: Dict[str, int] = None

    # Transaction cost (fee-aware labeling)
    transaction_cost: float = 0.0008

    # Quality scoring settings
    enable_quality_scoring: bool = True
    enable_quality_validation: bool = True

    def __post_init__(self):
        if self.analyst_profit_targets is None:
            self.analyst_profit_targets = {
                'micro': 0.003,    # 0.3% (net: 0.22% after fees)
                'small': 0.005,    # 0.5% (net: 0.42% after fees)
                'medium': 0.007,   # 0.7% (net: 0.62% after fees)
                'good': 0.010      # 1.0% (net: 0.92% after fees)
            }

        if self.tactician_profit_targets is None:
            # More aggressive targets for 1m timeframe
            self.tactician_profit_targets = {
                'micro': 0.001,    # 0.1% (net: 0.02% after fees)
                'small': 0.002,    # 0.2% (net: 0.12% after fees)
                'medium': 0.003,   # 0.3% (net: 0.22% after fees)
                'good': 0.005      # 0.5% (net: 0.42% after fees)
            }

        if self.analyst_time_horizons is None:
            # 5m timeframe - longer horizons
            self.analyst_time_horizons = {
                'immediate': 2,    # 10 minutes (2 * 5m)
                'short': 4         # 20 minutes (4 * 5m)
            }

        if self.tactician_time_horizons is None:
            # 1m timeframe - shorter horizons
            self.tactician_time_horizons = {
                'immediate': 5,    # 5 minutes (5 * 1m)
                'short': 10        # 10 minutes (10 * 1m)
            }


class LabelerAnalystTacticianAdapter:
    """
    Adapter for multi-horizon profit labeling in Analyst vs Tactician modes.

    For Analyst (5m): Combines long and short signals for unified training
    For Tactician (1m): Separates signals and creates directional-specific targets
    """

    def __init__(self, config: Optional[LabelerAnalystTacticianConfig] = None):
        self.config = config or LabelerAnalystTacticianConfig(LabelerTrainingMode.ANALYST)
        self.logger = get_logger('LabelerAnalystTacticianAdapter')

        # Initialize labeler components
        self.analyst_labeler = None
        self.tactician_labeler = None

        if LABELER_AVAILABLE:
            try:
                # Analyst labeler (5m timeframe)
                analyst_config = MultiHorizonConfig(
                    profit_targets=self.config.analyst_profit_targets,
                    time_horizons=self.config.analyst_time_horizons,
                    transaction_cost=self.config.transaction_cost,
                    enable_quality_scoring=self.config.enable_quality_scoring,
                    enable_quality_validation=self.config.enable_quality_validation
                )
                self.analyst_labeler = MultiHorizonProfitLabeler(analyst_config)

                # Tactician labeler (1m timeframe)
                tactician_config = MultiHorizonConfig(
                    profit_targets=self.config.tactician_profit_targets,
                    time_horizons=self.config.tactician_time_horizons,
                    transaction_cost=self.config.transaction_cost,
                    enable_quality_scoring=self.config.enable_quality_scoring,
                    enable_quality_validation=self.config.enable_quality_validation
                )
                self.tactician_labeler = MultiHorizonProfitLabeler(tactician_config)

                self.logger.info("✅ Multi-horizon profit labelers initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize labelers: {e}")
        else:
            self.logger.warning("⚠️ Multi-horizon profit labeler not available")

        self.logger.info(f"🚀 Labeler Analyst-Tactician Adapter initialized for {self.config.training_mode.value} mode")

    def generate_analyst_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate labels for Analyst training (5m timeframe).
        Creates combined labels without directional differentiation.

        Args:
            data: Input market data

        Returns:
            DataFrame with combined labels for unified training
        """
        self.logger.info("🔄 Generating labels for Analyst training (5m) - combined approach")

        if self.config.training_mode != LabelerTrainingMode.ANALYST:
            self.logger.warning(f"⚠️ Adapter configured for {self.config.training_mode.value}, not ANALYST mode")

        if not self.analyst_labeler:
            self.logger.warning("⚠️ Analyst labeler not available, using fallback labeling")
            return self._generate_combined_labels_fallback(data)

        try:
            # Generate labels using analyst labeler
            labeled_data = self.analyst_labeler.generate_labels(data)

            # Create combined opportunity scores for Analyst training
            combined_data = self._create_combined_analyst_targets(labeled_data)

            self.logger.info(f"✅ Analyst labeling completed: {len(combined_data)} samples with {len(combined_data.columns)} features")
            return combined_data

        except Exception as e:
            self.logger.warning(f"⚠️ Analyst labeling failed: {e}, using fallback")
            return self._generate_combined_labels_fallback(data)

    def generate_tactician_labels(self, data: pd.DataFrame, direction: str = 'both') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate labels for Tactician training (1m timeframe).
        Creates directional-specific labels for independent optimization.

        Args:
            data: Input market data
            direction: 'long', 'short', or 'both'

        Returns:
            Tuple of (long_data, short_data) or single dataset if direction specified
        """
        self.logger.info(f"🔄 Generating labels for Tactician training (1m) - directional separation ({direction})")

        if self.config.training_mode in [LabelerTrainingMode.ANALYST]:
            self.logger.warning(f"⚠️ Adapter configured for {self.config.training_mode.value}, not TACTICIAN mode")

        if not self.tactician_labeler:
            self.logger.warning("⚠️ Tactician labeler not available, using fallback labeling")
            return self._generate_directional_labels_fallback(data, direction)

        try:
            if direction == 'both':
                long_data = self._generate_directional_labels(data, 'long')
                short_data = self._generate_directional_labels(data, 'short')

                self.logger.info(f"✅ Tactician labeling completed: Long ({len(long_data)}) and Short ({len(short_data)}) datasets")
                return long_data, short_data

            elif direction == 'long':
                long_data = self._generate_directional_labels(data, 'long')
                self.logger.info(f"✅ Tactician long labeling completed: {len(long_data)} samples")
                return long_data, None

            elif direction == 'short':
                short_data = self._generate_directional_labels(data, 'short')
                self.logger.info(f"✅ Tactician short labeling completed: {len(short_data)} samples")
                return None, short_data

            else:
                raise ValueError(f"Invalid direction: {direction}. Must be 'long', 'short', or 'both'")

        except Exception as e:
            self.logger.warning(f"⚠️ Tactician labeling failed: {e}, using fallback")
            return self._generate_directional_labels_fallback(data, direction)

    def _generate_directional_labels(self, data: pd.DataFrame, direction: str) -> pd.DataFrame:
        """
        Generate directional-specific labels using tactician labeler.

        Args:
            data: Input market data
            direction: 'long' or 'short'

        Returns:
            DataFrame with directional labels
        """
        self.logger.info(f"🔧 Generating {direction} directional labels")

        try:
            # Generate base labels
            labeled_data = self.tactician_labeler.generate_labels(data)

            # Create directional-specific targets
            directional_data = self._create_directional_targets(labeled_data, direction)

            return directional_data

        except Exception as e:
            self.logger.warning(f"⚠️ Directional labeling failed: {e}")
            raise

    def _create_combined_analyst_targets(self, labeled_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create combined targets for Analyst training.

        Args:
            labeled_data: Labeled data with directional columns

        Returns:
            DataFrame with combined targets
        """
        combined_data = labeled_data.copy()

        # Find all directional opportunity columns
        long_cols = [col for col in labeled_data.columns if '_long_opportunity' in col]
        short_cols = [col for col in labeled_data.columns if '_short_opportunity' in col]

        # Calculate combined opportunity for each horizon
        horizons = ['immediate', 'short', 'overall']
        for horizon in horizons:
            long_col = f'long_{horizon}_opportunity'
            short_col = f'short_{horizon}_opportunity'

            if long_col in labeled_data.columns and short_col in labeled_data.columns:
                # Weighted combination based on signal strength
                combined_col = f'{horizon}_opportunity'
                combined_data[combined_col] = (
                    labeled_data[long_col] * self.config.long_weight +
                    labeled_data[short_col] * self.config.short_weight
                )

                self.logger.info(f"✅ Combined {horizon} opportunities: long={labeled_data[long_col].mean():.4f}, short={labeled_data[short_col].mean():.4f}")

        # Calculate overall combined opportunity score
        opportunity_cols = [col for col in combined_data.columns if '_opportunity' in col]
        if opportunity_cols:
            combined_data['combined_opportunity_score'] = combined_data[opportunity_cols].mean(axis=1)
            self.logger.info(f"✅ Created combined opportunity score: mean={combined_data['combined_opportunity_score'].mean():.4f}")

        # Remove directional columns for Analyst training
        directional_cols_to_remove = [col for col in combined_data.columns
                                    if '_long_' in col or '_short_' in col]
        combined_data = combined_data.drop(columns=directional_cols_to_remove, errors='ignore')

        return combined_data

    def _create_directional_targets(self, labeled_data: pd.DataFrame, direction: str) -> pd.DataFrame:
        """
        Create directional-specific targets.

        Args:
            labeled_data: Labeled data with directional columns
            direction: 'long' or 'short'

        Returns:
            DataFrame with directional targets
        """
        directional_data = labeled_data.copy()

        # Find directional opportunity columns
        if direction == 'long':
            opp_cols = [col for col in labeled_data.columns if '_long_opportunity' in col]
            # Keep only long-specific columns
            short_cols_to_remove = [col for col in directional_data.columns if '_short_' in col]
        else:
            opp_cols = [col for col in labeled_data.columns if '_short_opportunity' in col]
            # Keep only short-specific columns
            long_cols_to_remove = [col for col in directional_data.columns if '_long_' in col]

        if opp_cols:
            # Create directional strength indicator
            directional_data[f'{direction}_directional_strength'] = directional_data[opp_cols].mean(axis=1)

            # Create directional opportunity score
            directional_data[f'{direction}_opportunity_score'] = directional_data[opp_cols].max(axis=1)

        # Remove opposite directional columns
        if direction == 'long' and 'short_cols_to_remove' in locals():
            directional_data = directional_data.drop(columns=short_cols_to_remove, errors='ignore')
        elif direction == 'short' and 'long_cols_to_remove' in locals():
            directional_data = directional_data.drop(columns=long_cols_to_remove, errors='ignore')

        return directional_data

    def _generate_combined_labels_fallback(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fallback labeling for combined Analyst mode.

        Args:
            data: Input market data

        Returns:
            DataFrame with basic combined labels
        """
        self.logger.info("🔧 Generating fallback combined labels")

        combined_data = data.copy()

        # Create simple return-based targets
        for horizon in [10, 20]:  # 10 and 20 periods
            # Calculate future returns
            combined_data[f'future_return_{horizon}'] = combined_data['close'].shift(-horizon) / combined_data['close'] - 1

            # Create opportunity score based on return magnitude
            combined_data[f'return_opportunity_{horizon}'] = np.abs(combined_data[f'future_return_{horizon}'])

            # Create binary opportunity indicator
            combined_data[f'opportunity_{horizon}'] = (combined_data[f'future_return_{horizon}'].abs() > 0.002).astype(int)

        # Create combined opportunity score
        opportunity_cols = [col for col in combined_data.columns if 'opportunity_' in col]
        if opportunity_cols:
            combined_data['combined_opportunity_score'] = combined_data[opportunity_cols].mean(axis=1)

        self.logger.info(f"✅ Fallback combined labeling completed: {len(combined_data.columns)} features")
        return combined_data

    def _generate_directional_labels_fallback(self, data: pd.DataFrame, direction: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fallback labeling for directional Tactician mode.

        Args:
            data: Input market data
            direction: 'long', 'short', or 'both'

        Returns:
            Tuple of (long_data, short_data)
        """
        self.logger.info(f"🔧 Generating fallback directional labels for {direction}")

        if direction == 'both':
            long_data = self._generate_direction_fallback_labels(data, 'long')
            short_data = self._generate_direction_fallback_labels(data, 'short')
            return long_data, short_data
        elif direction == 'long':
            return self._generate_direction_fallback_labels(data, 'long'), None
        elif direction == 'short':
            return None, self._generate_direction_fallback_labels(data, 'short')
        else:
            raise ValueError(f"Invalid direction: {direction}")

    def _generate_direction_fallback_labels(self, data: pd.DataFrame, direction: str) -> pd.DataFrame:
        """
        Generate fallback labels for specific direction.

        Args:
            data: Input market data
            direction: 'long' or 'short'

        Returns:
            DataFrame with directional labels
        """
        directional_data = data.copy()

        # Create direction-specific labels
        for horizon in [5, 10]:  # Shorter horizons for 1m timeframe
            # Calculate future returns
            directional_data[f'future_return_{horizon}'] = directional_data['close'].shift(-horizon) / directional_data['close'] - 1

            # Create directional opportunity scores
            if direction == 'long':
                # Long opportunities: positive returns
                directional_data[f'{direction}_opportunity_{horizon}'] = np.maximum(
                    directional_data[f'future_return_{horizon}'], 0
                )
            else:
                # Short opportunities: negative returns (absolute value)
                directional_data[f'{direction}_opportunity_{horizon}'] = np.maximum(
                    -directional_data[f'future_return_{horizon}'], 0
                )

        # Create directional strength indicator
        opp_cols = [col for col in directional_data.columns if f'_{direction}_opportunity_' in col]
        if opp_cols:
            directional_data[f'{direction}_directional_strength'] = directional_data[opp_cols].mean(axis=1)
            directional_data[f'{direction}_opportunity_score'] = directional_data[opp_cols].max(axis=1)

        self.logger.info(f"✅ Fallback {direction} labeling completed: {len(directional_data.columns)} features")
        return directional_data

    def get_adapter_summary(self) -> Dict[str, Any]:
        """Get summary of adapter configuration."""
        return {
            'training_mode': self.config.training_mode.value,
            'analyst_profit_targets': self.config.analyst_profit_targets,
            'tactician_profit_targets': self.config.tactician_profit_targets,
            'analyst_time_horizons': self.config.analyst_time_horizons,
            'tactician_time_horizons': self.config.tactician_time_horizons,
            'transaction_cost': self.config.transaction_cost,
            'enable_quality_scoring': self.config.enable_quality_scoring,
            'enable_quality_validation': self.config.enable_quality_validation,
            'labeler_available': LABELER_AVAILABLE and self.analyst_labeler is not None,
            'long_weight': self.config.long_weight,
            'short_weight': self.config.short_weight
        }


# Convenience functions for easy integration
def generate_labels_for_analyst(data: pd.DataFrame,
                              config: Optional[LabelerAnalystTacticianConfig] = None) -> pd.DataFrame:
    """Generate labels for Analyst training (5m, combined)."""
    adapter = LabelerAnalystTacticianAdapter(config)
    return adapter.generate_analyst_labels(data)


def generate_labels_for_tactician(data: pd.DataFrame, direction: str = 'both',
                                config: Optional[LabelerAnalystTacticianConfig] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate labels for Tactician training (1m, directional)."""
    adapter = LabelerAnalystTacticianAdapter(config)
    return adapter.generate_tactician_labels(data, direction)