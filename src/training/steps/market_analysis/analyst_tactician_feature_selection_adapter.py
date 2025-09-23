"""
Analyst-Tactician Final Feature Selection Adapter

This adapter provides directional-aware final feature selection that can:
1. For Analyst: Remove directional differentiation (combined signals)
2. For Tactician: Separate long and short signals for independent optimization

Integrates with existing final feature selection step for optimal feature selection.
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

# Import existing final feature selection step
try:
    from .final_feature_selection_step import (
        FinalFeatureSelectionStep
    )
    FEATURE_SELECTION_AVAILABLE = True
except ImportError as e:
    FEATURE_SELECTION_AVAILABLE = False
    tprint(f"⚠️ Final feature selection step not available: {e}")


class FeatureSelectionTrainingMode(Enum):
    """Training mode for final feature selection."""
    ANALYST = "analyst"  # 5m timeframe, combined signals
    TACTICIAN_LONG = "tactician_long"  # 1m timeframe, long-only
    TACTICIAN_SHORT = "tactician_short"  # 1m timeframe, short-only


@dataclass
class FeatureSelectionAnalystTacticianConfig:
    """Configuration for Analyst-Tactician final feature selection."""
    training_mode: FeatureSelectionTrainingMode
    combined_features_target: int = 80  # Target for combined features (Analyst)
    long_features_target: int = 50      # Target for long-only features (Tactician)
    short_features_target: int = 50     # Target for short-only features (Tactician)

    # Directional weighting for combined mode
    long_weight: float = 0.5
    short_weight: float = 0.5

    # Feature selection settings
    rf_n_estimators: int = 100
    cv_folds: int = 5
    enable_quality_validation: bool = True
    enable_outlier_detection: bool = True
    outlier_threshold: float = 3.0
    min_sample_quality_score: float = 0.7


class FeatureSelectionAnalystTacticianAdapter:
    """
    Adapter for final feature selection in Analyst vs Tactician modes.

    For Analyst (5m): Combines long and short signals for unified feature selection
    For Tactician (1m): Separates signals and selects features for each direction independently
    """

    def __init__(self, config: Optional[FeatureSelectionAnalystTacticianConfig] = None):
        self.config = config or FeatureSelectionAnalystTacticianConfig(FeatureSelectionTrainingMode.ANALYST)
        self.logger = get_logger('FeatureSelectionAnalystTacticianAdapter')

        # Initialize feature selection components
        self.analyst_selector = None
        self.tactician_long_selector = None
        self.tactician_short_selector = None

        if FEATURE_SELECTION_AVAILABLE:
            try:
                # Analyst selector (5m timeframe, combined)
                analyst_config = {
                    'initial_features': 120,
                    'stage_1_target': 100,
                    'stage_2_target': self.config.combined_features_target,
                    'stage_3_target': self.config.combined_features_target,
                    'rf_n_estimators': self.config.rf_n_estimators,
                    'cv_folds': self.config.cv_folds,
                    'save_analysis': True,
                    'output_directory': "outcomes/market_analysis",
                    'verbose': True
                }
                self.analyst_selector = FinalFeatureSelectionStep(analyst_config)

                # Tactician selectors (1m timeframe, directional)
                long_config = analyst_config.copy()
                long_config.update({
                    'stage_2_target': self.config.long_features_target,
                    'stage_3_target': self.config.long_features_target,
                    'output_directory': "outcomes/market_analysis/long"
                })
                self.tactician_long_selector = FinalFeatureSelectionStep(long_config)

                short_config = analyst_config.copy()
                short_config.update({
                    'stage_2_target': self.config.short_features_target,
                    'stage_3_target': self.config.short_features_target,
                    'output_directory': "outcomes/market_analysis/short"
                })
                self.tactician_short_selector = FinalFeatureSelectionStep(short_config)

                self.logger.info("✅ Final feature selection components initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize feature selectors: {e}")
        else:
            self.logger.warning("⚠️ Final feature selection step not available")

        self.logger.info(f"🚀 Feature Selection Analyst-Tactician Adapter initialized for {self.config.training_mode.value} mode")

    async def select_analyst_features(self, symbol: str, exchange: str, data_dir: str) -> pd.DataFrame:
        """
        Select features for Analyst training (5m timeframe).
        Selects optimal features from combined signals.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory path

        Returns:
            DataFrame with selected features for Analyst training
        """
        self.logger.info("🔄 Selecting features for Analyst training (5m) - combined approach")

        if self.config.training_mode != FeatureSelectionTrainingMode.ANALYST:
            self.logger.warning(f"⚠️ Adapter configured for {self.config.training_mode.value}, not ANALYST mode")

        if not self.analyst_selector:
            self.logger.warning("⚠️ Analyst selector not available, using fallback selection")
            return await self._select_combined_features_fallback(symbol, exchange, data_dir)

        try:
            # Execute feature selection
            success = await self.analyst_selector.execute_final_feature_selection(
                symbol=symbol,
                exchange=exchange,
                timeframe='5m',
                data_dir=data_dir
            )

            if success:
                # Load selected features
                selected_data = await self._load_selected_features(symbol, exchange, '5m', 'combined')
                self.logger.info(f"✅ Analyst feature selection completed: {len(selected_data.columns)} features selected")
                return selected_data
            else:
                raise Exception("Analyst feature selection failed")

        except Exception as e:
            self.logger.warning(f"⚠️ Analyst feature selection failed: {e}, using fallback")
            return await self._select_combined_features_fallback(symbol, exchange, data_dir)

    async def select_tactician_features(self, symbol: str, exchange: str, data_dir: str,
                                      direction: str = 'both') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Select features for Tactician training (1m timeframe).
        Selects directional-specific features for independent optimization.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory path
            direction: 'long', 'short', or 'both'

        Returns:
            Tuple of (long_data, short_data) or single dataset if direction specified
        """
        self.logger.info(f"🔄 Selecting features for Tactician training (1m) - directional separation ({direction})")

        if self.config.training_mode in [FeatureSelectionTrainingMode.ANALYST]:
            self.logger.warning(f"⚠️ Adapter configured for {self.config.training_mode.value}, not TACTICIAN mode")

        if not self.tactician_long_selector or not self.tactician_short_selector:
            self.logger.warning("⚠️ Tactician selectors not available, using fallback selection")
            return await self._select_directional_features_fallback(symbol, exchange, data_dir, direction)

        try:
            if direction == 'both':
                # Select features for both directions
                long_success = await self.tactician_long_selector.execute_final_feature_selection(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe='1m',
                    data_dir=f"{data_dir}/long"
                )

                short_success = await self.tactician_short_selector.execute_final_feature_selection(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe='1m',
                    data_dir=f"{data_dir}/short"
                )

                if long_success and short_success:
                    long_data = await self._load_selected_features(symbol, exchange, '1m', 'long')
                    short_data = await self._load_selected_features(symbol, exchange, '1m', 'short')

                    self.logger.info(f"✅ Tactician feature selection completed: Long ({len(long_data.columns)}) and Short ({len(short_data.columns)}) features")
                    return long_data, short_data
                else:
                    raise Exception(f"Directional feature selection failed: long={long_success}, short={short_success}")

            elif direction == 'long':
                success = await self.tactician_long_selector.execute_final_feature_selection(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe='1m',
                    data_dir=data_dir
                )

                if success:
                    long_data = await self._load_selected_features(symbol, exchange, '1m', 'long')
                    self.logger.info(f"✅ Tactician long feature selection completed: {len(long_data.columns)} features")
                    return long_data, None
                else:
                    raise Exception("Long feature selection failed")

            elif direction == 'short':
                success = await self.tactician_short_selector.execute_final_feature_selection(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe='1m',
                    data_dir=data_dir
                )

                if success:
                    short_data = await self._load_selected_features(symbol, exchange, '1m', 'short')
                    self.logger.info(f"✅ Tactician short feature selection completed: {len(short_data.columns)} features")
                    return None, short_data
                else:
                    raise Exception("Short feature selection failed")

            else:
                raise ValueError(f"Invalid direction: {direction}. Must be 'long', 'short', or 'both'")

        except Exception as e:
            self.logger.warning(f"⚠️ Tactician feature selection failed: {e}, using fallback")
            return await self._select_directional_features_fallback(symbol, exchange, data_dir, direction)

    async def _load_selected_features(self, symbol: str, exchange: str, timeframe: str, mode: str) -> pd.DataFrame:
        """
        Load selected features from feature selection results.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe
            mode: 'combined', 'long', or 'short'

        Returns:
            DataFrame with selected features
        """
        try:
            # Try to load from feature selection results
            import os
            from pathlib import Path

            # Determine the correct data directory based on mode
            if mode == 'combined':
                data_dir = "outcomes/market_analysis"
            else:
                data_dir = f"outcomes/market_analysis/{mode}"

            # Look for selected features file
            selected_features_file = Path(data_dir) / f"{symbol.lower()}_{timeframe}_final_features.json"

            if selected_features_file.exists():
                import json
                with open(selected_features_file, 'r') as f:
                    feature_data = json.load(f)

                selected_features = feature_data.get('final_features', [])

                # Load original data and filter to selected features
                original_data_file = Path(data_dir) / f"{symbol.lower()}_{timeframe}_features.parquet"
                if original_data_file.exists():
                    original_data = pd.read_parquet(original_data_file)

                    # Filter to selected features
                    available_features = [f for f in selected_features if f in original_data.columns]
                    if available_features:
                        return original_data[available_features]

            # Fallback: return empty DataFrame
            self.logger.warning(f"⚠️ Could not load selected features for {mode}, returning empty DataFrame")
            return pd.DataFrame()

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load selected features: {e}")
            return pd.DataFrame()

    async def _select_combined_features_fallback(self, symbol: str, exchange: str, data_dir: str) -> pd.DataFrame:
        """
        Fallback feature selection for combined Analyst mode.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory path

        Returns:
            DataFrame with fallback selected features
        """
        self.logger.info("🔧 Performing fallback combined feature selection")

        try:
            # Load feature data
            import os
            from pathlib import Path

            feature_file = Path(data_dir) / f"{symbol.lower()}_5m_features.parquet"
            if not feature_file.exists():
                self.logger.error(f"❌ Feature file not found: {feature_file}")
                return pd.DataFrame()

            data = pd.read_parquet(feature_file)

            # Simple fallback selection based on correlation and variance
            selected_features = self._fallback_feature_selection(data, self.config.combined_features_target)

            self.logger.info(f"✅ Fallback combined feature selection completed: {len(selected_features)} features")
            return data[selected_features]

        except Exception as e:
            self.logger.error(f"❌ Fallback combined feature selection failed: {e}")
            return pd.DataFrame()

    async def _select_directional_features_fallback(self, symbol: str, exchange: str, data_dir: str,
                                                 direction: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fallback feature selection for directional Tactician mode.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory path
            direction: 'long', 'short', or 'both'

        Returns:
            Tuple of (long_data, short_data)
        """
        self.logger.info(f"🔧 Performing fallback directional feature selection for {direction}")

        if direction == 'both':
            long_data = await self._select_direction_fallback_features(symbol, exchange, data_dir, 'long')
            short_data = await self._select_direction_fallback_features(symbol, exchange, data_dir, 'short')
            return long_data, short_data
        elif direction == 'long':
            long_data = await self._select_direction_fallback_features(symbol, exchange, data_dir, 'long')
            return long_data, None
        elif direction == 'short':
            short_data = await self._select_direction_fallback_features(symbol, exchange, data_dir, 'short')
            return None, short_data
        else:
            raise ValueError(f"Invalid direction: {direction}")

    async def _select_direction_fallback_features(self, symbol: str, exchange: str, data_dir: str,
                                                direction: str) -> pd.DataFrame:
        """
        Fallback feature selection for specific direction.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory path
            direction: 'long' or 'short'

        Returns:
            DataFrame with fallback selected features for direction
        """
        try:
            # Load feature data
            import os
            from pathlib import Path

            feature_file = Path(data_dir) / f"{symbol.lower()}_1m_features.parquet"
            if not feature_file.exists():
                self.logger.error(f"❌ Feature file not found: {feature_file}")
                return pd.DataFrame()

            data = pd.read_parquet(feature_file)

            # Apply directional filtering and selection
            target_count = getattr(self.config, f'{direction}_features_target', 50)
            selected_features = self._fallback_feature_selection(data, target_count)

            self.logger.info(f"✅ Fallback {direction} feature selection completed: {len(selected_features)} features")
            return data[selected_features]

        except Exception as e:
            self.logger.error(f"❌ Fallback {direction} feature selection failed: {e}")
            return pd.DataFrame()

    def _fallback_feature_selection(self, data: pd.DataFrame, target_count: int) -> List[str]:
        """
        Simple fallback feature selection based on variance and correlation.

        Args:
            data: Input feature data
            target_count: Target number of features to select

        Returns:
            List of selected feature names
        """
        try:
            # Remove non-numeric columns
            numeric_data = data.select_dtypes(include=[np.number]).copy()

            # Remove columns with too many missing values
            missing_threshold = 0.1  # 10% missing values
            missing_ratios = numeric_data.isnull().mean()
            valid_columns = missing_ratios[missing_ratios <= missing_threshold].index.tolist()

            if len(valid_columns) <= target_count:
                return valid_columns

            # Calculate variance for feature importance
            variances = numeric_data[valid_columns].var().sort_values(ascending=False)

            # Select top features by variance
            top_by_variance = variances.head(target_count).index.tolist()

            # If we have more features, apply correlation filtering
            if len(valid_columns) > target_count:
                # Calculate correlation matrix
                corr_matrix = numeric_data[top_by_variance].corr().abs()

                # Remove highly correlated features (keep the one with higher variance)
                selected_features = []
                for feature in top_by_variance:
                    if not selected_features:
                        selected_features.append(feature)
                        continue

                    # Check correlation with already selected features
                    correlated = False
                    for selected in selected_features:
                        if corr_matrix.loc[feature, selected] > 0.95:  # 95% correlation threshold
                            correlated = True
                            break

                    if not correlated:
                        selected_features.append(feature)

                return selected_features[:target_count]
            else:
                return top_by_variance.tolist()

        except Exception as e:
            self.logger.warning(f"⚠️ Fallback feature selection failed: {e}, using first {target_count} columns")
            return list(data.columns)[:target_count]

    def get_adapter_summary(self) -> Dict[str, Any]:
        """Get summary of adapter configuration."""
        return {
            'training_mode': self.config.training_mode.value,
            'combined_features_target': self.config.combined_features_target,
            'long_features_target': self.config.long_features_target,
            'short_features_target': self.config.short_features_target,
            'rf_n_estimators': self.config.rf_n_estimators,
            'cv_folds': self.config.cv_folds,
            'enable_quality_validation': self.config.enable_quality_validation,
            'enable_outlier_detection': self.config.enable_outlier_detection,
            'outlier_threshold': self.config.outlier_threshold,
            'min_sample_quality_score': self.config.min_sample_quality_score,
            'feature_selection_available': FEATURE_SELECTION_AVAILABLE and self.analyst_selector is not None,
            'long_weight': self.config.long_weight,
            'short_weight': self.config.short_weight
        }


# Convenience functions for easy integration
async def select_features_for_analyst(symbol: str, exchange: str, data_dir: str,
                                    config: Optional[FeatureSelectionAnalystTacticianConfig] = None) -> pd.DataFrame:
    """Select features for Analyst training (5m, combined)."""
    adapter = FeatureSelectionAnalystTacticianAdapter(config)
    return await adapter.select_analyst_features(symbol, exchange, data_dir)


async def select_features_for_tactician(symbol: str, exchange: str, data_dir: str, direction: str = 'both',
                                      config: Optional[FeatureSelectionAnalystTacticianConfig] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Select features for Tactician training (1m, directional)."""
    adapter = FeatureSelectionAnalystTacticianAdapter(config)
    return await adapter.select_tactician_features(symbol, exchange, data_dir, direction)