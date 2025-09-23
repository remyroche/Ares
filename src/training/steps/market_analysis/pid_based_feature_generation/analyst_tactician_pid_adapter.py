"""
Analyst-Tactician PID-Based Feature Generation Adapter

This adapter provides directional-aware PID-based feature generation that can:
1. For Analyst: Remove directional differentiation (combined signals)
2. For Tactician: Separate long and short signals for independent optimization

Integrates with existing PID-based feature generation for optimal feature creation.
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

# Import existing PID-based feature generation
try:
    from .pid_based_feature_generation_component import (
        PIDBasedFeatureGenerationComponent, GenerationStatus
    )
    PID_AVAILABLE = True
except ImportError as e:
    PID_AVAILABLE = False
    tprint(f"⚠️ PID-based feature generation not available: {e}")


class PIDTrainingMode(Enum):
    """Training mode for PID-based feature generation."""
    ANALYST = "analyst"  # 5m timeframe, combined signals
    TACTICIAN_LONG = "tactician_long"  # 1m timeframe, long-only
    TACTICIAN_SHORT = "tactician_short"  # 1m timeframe, short-only


@dataclass
class PIDAnalystTacticianConfig:
    """Configuration for Analyst-Tactician PID-based feature generation."""
    training_mode: PIDTrainingMode
    max_interaction_features: int = 100
    max_polynomial_features: int = 50
    max_cross_timeframe_features: int = 50

    # Directional weighting for combined mode
    long_weight: float = 0.5
    short_weight: float = 0.5

    # Feature generation settings
    enable_interaction_features: bool = True
    enable_polynomial_features: bool = True
    enable_cross_timeframe_features: bool = True

    # Optimization settings
    enable_parallel_processing: bool = True
    enable_gpu_acceleration: bool = True
    memory_limit_gb: float = 8.0


class PIDAnalystTacticianAdapter:
    """
    Adapter for PID-based feature generation in Analyst vs Tactician modes.

    For Analyst (5m): Combines long and short signals for unified feature generation
    For Tactician (1m): Separates signals and generates directional-specific features
    """

    def __init__(self, config: Optional[PIDAnalystTacticianConfig] = None):
        self.config = config or PIDAnalystTacticianConfig(PIDTrainingMode.ANALYST)
        self.logger = get_logger('PIDAnalystTacticianAdapter')

        # Initialize PID component if available
        self.pid_component = None
        if PID_AVAILABLE:
            try:
                from src.training.steps.market_analysis.components.base_component import ComponentConfig
                pid_config = ComponentConfig(
                    symbol='ADAPTER',
                    exchange='ADAPTER',
                    timeframe='5m'  # Default timeframe
                )
                self.pid_component = PIDBasedFeatureGenerationComponent(pid_config)
                self.logger.info("✅ PID-based feature generation component initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize PID component: {e}")
        else:
            self.logger.warning("⚠️ PID-based feature generation not available")

        self.logger.info(f"🚀 PID Analyst-Tactician Adapter initialized for {self.config.training_mode.value} mode")

    def generate_analyst_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features for Analyst training (5m timeframe).
        Creates combined features without directional differentiation.

        Args:
            data: Input market data

        Returns:
            DataFrame with combined features for unified training
        """
        self.logger.info("🔄 Generating features for Analyst training (5m) - combined approach")

        if self.config.training_mode != PIDTrainingMode.ANALYST:
            self.logger.warning(f"⚠️ Adapter configured for {self.config.training_mode.value}, not ANALYST mode")

        if not self.pid_component:
            self.logger.warning("⚠️ PID component not available, using fallback feature generation")
            return self._generate_combined_features_fallback(data)

        try:
            # Use PID component with combined configuration
            combined_data = self._generate_combined_pid_features(data)

            # Remove directional columns for Analyst training
            directional_cols_to_remove = [col for col in combined_data.columns
                                        if '_long_' in col or '_short_' in col]
            combined_data = combined_data.drop(columns=directional_cols_to_remove, errors='ignore')

            self.logger.info(f"✅ Analyst feature generation completed: {len(combined_data)} samples, {len(combined_data.columns)} features")
            return combined_data

        except Exception as e:
            self.logger.warning(f"⚠️ PID feature generation failed: {e}, using fallback")
            return self._generate_combined_features_fallback(data)

    def generate_tactician_features(self, data: pd.DataFrame, direction: str = 'both') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate features for Tactician training (1m timeframe).
        Creates directional-specific features for independent optimization.

        Args:
            data: Input market data
            direction: 'long', 'short', or 'both'

        Returns:
            Tuple of (long_data, short_data) or single dataset if direction specified
        """
        self.logger.info(f"🔄 Generating features for Tactician training (1m) - directional separation ({direction})")

        if self.config.training_mode in [PIDTrainingMode.ANALYST]:
            self.logger.warning(f"⚠️ Adapter configured for {self.config.training_mode.value}, not TACTICIAN mode")

        if not self.pid_component:
            self.logger.warning("⚠️ PID component not available, using fallback feature generation")
            return self._generate_directional_features_fallback(data, direction)

        try:
            if direction == 'both':
                long_data = self._generate_directional_pid_features(data, 'long')
                short_data = self._generate_directional_pid_features(data, 'short')

                self.logger.info(f"✅ Tactician feature generation completed: Long ({len(long_data)}) and Short ({len(short_data)}) datasets")
                return long_data, short_data

            elif direction == 'long':
                long_data = self._generate_directional_pid_features(data, 'long')
                self.logger.info(f"✅ Tactician long feature generation completed: {len(long_data)} samples")
                return long_data, None

            elif direction == 'short':
                short_data = self._generate_directional_pid_features(data, 'short')
                self.logger.info(f"✅ Tactician short feature generation completed: {len(short_data)} samples")
                return None, short_data

            else:
                raise ValueError(f"Invalid direction: {direction}. Must be 'long', 'short', or 'both'")

        except Exception as e:
            self.logger.warning(f"⚠️ PID feature generation failed: {e}, using fallback")
            return self._generate_directional_features_fallback(data, direction)

    def _generate_combined_pid_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate combined features using PID component.

        Args:
            data: Input market data

        Returns:
            DataFrame with combined features
        """
        self.logger.info("🔧 Generating combined PID features")

        # Create pipeline state for PID component
        pipeline_state = {
            'symbol': 'COMBINED',
            'exchange': 'COMBINED',
            'timeframe': '5m',
            'training_mode': 'analyst'
        }

        try:
            # Execute PID component
            result = self.pid_component.execute(data, pipeline_state)

            if result.status == 'success' and result.data is not None:
                combined_features = result.data.copy()

                # Add combined opportunity score
                opportunity_cols = [col for col in combined_features.columns if '_opportunity' in col]
                if opportunity_cols:
                    combined_features['combined_opportunity_score'] = combined_features[opportunity_cols].mean(axis=1)

                return combined_features
            else:
                raise Exception(f"PID component failed: {result.message}")

        except Exception as e:
            self.logger.warning(f"⚠️ PID component execution failed: {e}")
            raise

    def _generate_directional_pid_features(self, data: pd.DataFrame, direction: str) -> pd.DataFrame:
        """
        Generate directional-specific features using PID component.

        Args:
            data: Input market data
            direction: 'long' or 'short'

        Returns:
            DataFrame with directional features
        """
        self.logger.info(f"🔧 Generating {direction} directional PID features")

        # Create pipeline state for directional training
        pipeline_state = {
            'symbol': f'{direction.upper()}',
            'exchange': 'DIRECTIONAL',
            'timeframe': '1m',
            'training_mode': f'tactician_{direction}',
            'direction': direction
        }

        try:
            # Execute PID component
            result = self.pid_component.execute(data, pipeline_state)

            if result.status == 'success' and result.data is not None:
                directional_features = result.data.copy()

                # Add directional strength indicator
                opp_cols = [col for col in directional_features.columns if f'_{direction}_opportunity' in col]
                if opp_cols:
                    directional_features[f'{direction}_directional_strength'] = directional_features[opp_cols].mean(axis=1)

                return directional_features
            else:
                raise Exception(f"PID component failed: {result.message}")

        except Exception as e:
            self.logger.warning(f"⚠️ PID component execution failed: {e}")
            raise

    def _generate_combined_features_fallback(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fallback feature generation for combined Analyst mode.

        Args:
            data: Input market data

        Returns:
            DataFrame with basic combined features
        """
        self.logger.info("🔧 Generating fallback combined features")

        combined_data = data.copy()

        # Create simple combined features
        # Moving averages
        for window in [10, 20, 50]:
            combined_data[f'sma_{window}'] = combined_data['close'].rolling(window).mean()
            combined_data[f'ema_{window}'] = combined_data['close'].ewm(span=window).mean()

        # Volatility features
        combined_data['volatility_10'] = combined_data['close'].rolling(10).std()
        combined_data['volatility_20'] = combined_data['close'].rolling(20).std()

        # Price momentum
        combined_data['momentum_5'] = combined_data['close'].pct_change(5)
        combined_data['momentum_10'] = combined_data['close'].pct_change(10)

        # Volume features
        if 'volume' in combined_data.columns:
            combined_data['volume_ma_10'] = combined_data['volume'].rolling(10).mean()
            combined_data['volume_ratio'] = combined_data['volume'] / combined_data['volume_ma_10']

        # Create combined opportunity score
        opportunity_features = [col for col in combined_data.columns if 'opportunity' in col or 'momentum' in col]
        if opportunity_features:
            combined_data['combined_opportunity_score'] = combined_data[opportunity_features].mean(axis=1)

        self.logger.info(f"✅ Fallback combined feature generation completed: {len(combined_data.columns)} features")
        return combined_data

    def _generate_directional_features_fallback(self, data: pd.DataFrame, direction: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fallback feature generation for directional Tactician mode.

        Args:
            data: Input market data
            direction: 'long', 'short', or 'both'

        Returns:
            Tuple of (long_data, short_data)
        """
        self.logger.info(f"🔧 Generating fallback directional features for {direction}")

        if direction == 'both':
            long_data = self._generate_direction_fallback_features(data, 'long')
            short_data = self._generate_direction_fallback_features(data, 'short')
            return long_data, short_data
        elif direction == 'long':
            return self._generate_direction_fallback_features(data, 'long'), None
        elif direction == 'short':
            return None, self._generate_direction_fallback_features(data, 'short')
        else:
            raise ValueError(f"Invalid direction: {direction}")

    def _generate_direction_fallback_features(self, data: pd.DataFrame, direction: str) -> pd.DataFrame:
        """
        Generate fallback features for specific direction.

        Args:
            data: Input market data
            direction: 'long' or 'short'

        Returns:
            DataFrame with directional features
        """
        directional_data = data.copy()

        # Generate direction-specific features
        for window in [5, 10, 20]:
            # Price-based features
            directional_data[f'{direction}_sma_{window}'] = directional_data['close'].rolling(window).mean()
            directional_data[f'{direction}_ema_{window}'] = directional_data['close'].ewm(span=window).mean()
            directional_data[f'{direction}_momentum_{window}'] = directional_data['close'].pct_change(window)

            # Volatility features
            directional_data[f'{direction}_volatility_{window}'] = directional_data['close'].rolling(window).std()

            # Directional bias features
            if direction == 'long':
                # Long features emphasize upward momentum
                directional_data[f'{direction}_upward_trend_{window}'] = (
                    directional_data['close'] > directional_data[f'{direction}_sma_{window}']
                ).astype(int)
            else:
                # Short features emphasize downward momentum
                directional_data[f'{direction}_downward_trend_{window}'] = (
                    directional_data['close'] < directional_data[f'{direction}_sma_{window}']
                ).astype(int)

        # Directional strength indicator
        opp_cols = [col for col in directional_data.columns if f'_{direction}_' in col and 'momentum' in col]
        if opp_cols:
            directional_data[f'{direction}_directional_strength'] = directional_data[opp_cols].mean(axis=1)

        self.logger.info(f"✅ Fallback {direction} feature generation completed: {len(directional_data.columns)} features")
        return directional_data

    def get_adapter_summary(self) -> Dict[str, Any]:
        """Get summary of adapter configuration."""
        return {
            'training_mode': self.config.training_mode.value,
            'max_interaction_features': self.config.max_interaction_features,
            'max_polynomial_features': self.config.max_polynomial_features,
            'max_cross_timeframe_features': self.config.max_cross_timeframe_features,
            'enable_interaction_features': self.config.enable_interaction_features,
            'enable_polynomial_features': self.config.enable_polynomial_features,
            'enable_cross_timeframe_features': self.config.enable_cross_timeframe_features,
            'pid_component_available': PID_AVAILABLE and self.pid_component is not None,
            'long_weight': self.config.long_weight,
            'short_weight': self.config.short_weight
        }


# Convenience functions for easy integration
def generate_features_for_analyst(data: pd.DataFrame,
                                config: Optional[PIDAnalystTacticianConfig] = None) -> pd.DataFrame:
    """Generate features for Analyst training (5m, combined)."""
    adapter = PIDAnalystTacticianAdapter(config)
    return adapter.generate_analyst_features(data)


def generate_features_for_tactician(data: pd.DataFrame, direction: str = 'both',
                                  config: Optional[PIDAnalystTacticianConfig] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate features for Tactician training (1m, directional)."""
    adapter = PIDAnalystTacticianAdapter(config)
    return adapter.generate_tactician_features(data, direction)