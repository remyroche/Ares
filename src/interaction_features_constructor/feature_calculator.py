"""
Feature Calculator

Main orchestrator for calculating selected features from base features.
Handles:
- Feature retrieval from feature_bank
- Variant transformations
- Cross-timeframe calculations
- Mathematical operations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

try:
    import polars as pl
except Exception:
    pl = None

from src.interaction_features_constructor.feature_decomposer import FeatureDecomposer
from src.interaction_features_constructor.feature_metadata_store import FeatureMetadataStore


class FeatureCalculator:
    """
    Calculates selected features from base features in feature_bank.

    Usage:
        # From selected features
        calculator = FeatureCalculator(selected_features)
        features = calculator.calculate(ohlcv_data, feature_bank)

        # From saved metadata
        calculator = FeatureCalculator.from_metadata_file('path/to/metadata.json')
        features = calculator.calculate(ohlcv_data, feature_bank)
    """

    def __init__(self, selected_features: Optional[List[str]] = None):
        """
        Initialize the feature calculator.

        Args:
            selected_features: List of selected feature names to calculate
        """
        self.decomposer = FeatureDecomposer()
        self.selected_features = selected_features or []
        self.metadata_store: Optional[FeatureMetadataStore] = None

        if self.selected_features:
            self._decompose_features()

    def _decompose_features(self):
        """Decompose selected features into their components."""
        self.feature_components = self.decomposer.batch_decompose(self.selected_features)
        self.base_features_required = self.decomposer.get_all_base_features(self.selected_features)

    @classmethod
    def from_metadata_file(cls, filepath: str) -> 'FeatureCalculator':
        """
        Create a FeatureCalculator from a saved metadata file.

        Args:
            filepath: Path to the metadata JSON file

        Returns:
            FeatureCalculator instance
        """
        metadata_store = FeatureMetadataStore.load(filepath)
        calculator = cls(metadata_store.get_selected_features())
        calculator.metadata_store = metadata_store
        return calculator

    @classmethod
    def from_metadata(cls, metadata_store: FeatureMetadataStore) -> 'FeatureCalculator':
        """
        Create a FeatureCalculator from a FeatureMetadataStore instance.

        Args:
            metadata_store: FeatureMetadataStore instance

        Returns:
            FeatureCalculator instance
        """
        calculator = cls(metadata_store.get_selected_features())
        calculator.metadata_store = metadata_store
        return calculator

    def get_required_base_features(self) -> List[str]:
        """
        Get the list of base features required from feature_bank.

        Returns:
            List of base feature names
        """
        return self.base_features_required

    def calculate(
        self,
        ohlcv_data: pd.DataFrame,
        base_features: Union[pd.DataFrame, Dict[str, pd.Series]],
        return_type: str = 'dataframe'
    ) -> Union[pd.DataFrame, Dict[str, pd.Series]]:
        """
        Calculate all selected features from base features and OHLCV data.

        Args:
            ohlcv_data: DataFrame with OHLCV data (required for transformations)
            base_features: DataFrame or dict with base features from feature_bank
            return_type: 'dataframe' or 'dict'

        Returns:
            DataFrame or dict with calculated features
        """
        # Normalize OHLCV input to pandas DataFrame (supports optional Polars input)
        if pl is not None and isinstance(ohlcv_data, pl.DataFrame):
            ohlcv_data_pd = ohlcv_data.to_pandas()
        else:
            ohlcv_data_pd = ohlcv_data

        # Convert base_features to dict if it's a DataFrame (supports optional Polars input)
        if isinstance(base_features, dict):
            base_features_dict = base_features
        else:
            if pl is not None and isinstance(base_features, pl.DataFrame):
                base_features_pd = base_features.to_pandas()
            else:
                base_features_pd = base_features

            base_features_dict = {col: base_features_pd[col] for col in base_features_pd.columns}

        # Storage for calculated features
        calculated_features = {}

        # Process each selected feature
        for feature_name in self.selected_features:
            try:
                calculated = self._calculate_single_feature(
                    feature_name,
                    base_features_dict,
                    ohlcv_data_pd
                )
                calculated_features[feature_name] = calculated
            except Exception as e:
                print(f"Warning: Failed to calculate {feature_name}: {e}")
                # Create NaN series as placeholder
                calculated_features[feature_name] = pd.Series(
                    np.nan,
                    index=ohlcv_data_pd.index if hasattr(ohlcv_data_pd, 'index') else range(len(ohlcv_data_pd))
                )

        # Return as DataFrame or dict
        if return_type == 'dataframe':
            return pd.DataFrame(calculated_features)
        else:
            return calculated_features

    def _calculate_single_feature(
        self,
        feature_name: str,
        base_features: Dict[str, pd.Series],
        ohlcv_data: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate a single feature from its components.

        Args:
            feature_name: Name of the feature to calculate
            base_features: Dict of base features
            ohlcv_data: OHLCV data

        Returns:
            Calculated feature as a Series
        """
        components = self.feature_components[feature_name]

        # Process calculation steps in order
        intermediate_results = {}

        for step in components.calculation_steps:
            step_type = step['step']

            if step_type == 'get_base_feature':
                feature = step['feature']
                if feature in base_features:
                    intermediate_results[feature] = base_features[feature]
                else:
                    raise ValueError(f"Base feature '{feature}' not found in feature_bank")

            elif step_type == 'apply_variant':
                variant_type = step['variant_type']
                input_feature = step['input']
                input_data = intermediate_results.get(input_feature) or base_features.get(input_feature)

                if input_data is None:
                    raise ValueError(f"Input feature '{input_feature}' not found")

                variant_result = self._apply_variant_transformation(
                    input_data,
                    variant_type,
                    ohlcv_data
                )
                variant_name = f"{input_feature}_{variant_type}"
                intermediate_results[variant_name] = variant_result

            elif step_type == 'apply_timeframe_ratio':
                multiplier = step['multiplier']
                input_feature = step['input']
                input_data = intermediate_results.get(input_feature) or base_features.get(input_feature)

                if input_data is None:
                    raise ValueError(f"Input feature '{input_feature}' not found")

                ratio_result = self._apply_timeframe_ratio(
                    input_data,
                    multiplier
                )
                ratio_name = f"{input_feature}_{multiplier}x_ratio"
                intermediate_results[ratio_name] = ratio_result

            elif step_type == 'apply_operator':
                operator = step['operator']
                left_name = step['left']
                right_name = step['right']

                left_data = intermediate_results.get(left_name) or base_features.get(left_name)
                right_data = intermediate_results.get(right_name) or base_features.get(right_name)

                if left_data is None or right_data is None:
                    raise ValueError(f"Operator inputs not found: {left_name}, {right_name}")

                result = self._apply_operator(left_data, right_data, operator)
                intermediate_results[feature_name] = result

        # Return the final calculated feature
        return intermediate_results.get(feature_name, intermediate_results.get(components.feature_name))

    def _apply_variant_transformation(
        self,
        feature: pd.Series,
        variant_type: str,
        ohlcv_data: pd.DataFrame
    ) -> pd.Series:
        """
        Apply variant transformation to a feature.

        Args:
            feature: Input feature series
            variant_type: Type of transformation (volnorm, vwap, trend_adj)
            ohlcv_data: OHLCV data for transformations

        Returns:
            Transformed feature
        """
        if variant_type == 'volnorm':
            return self._apply_volatility_normalization(feature, ohlcv_data)
        elif variant_type == 'vwap':
            return self._apply_vwap_weighting(feature, ohlcv_data)
        elif variant_type == 'trend_adj':
            return self._apply_trend_adjustment(feature, ohlcv_data)
        elif variant_type == 'base':
            return feature
        else:
            raise ValueError(f"Unknown variant type: {variant_type}")

    def _apply_volatility_normalization(
        self,
        feature: pd.Series,
        ohlcv_data: pd.DataFrame,
        lookback_period: int = 20
    ) -> pd.Series:
        """Normalize feature by volatility."""
        volatility = ohlcv_data['close'].pct_change().rolling(
            window=lookback_period,
            min_periods=max(1, lookback_period // 2)
        ).std()

        # Avoid division by zero
        volatility = volatility.replace(0, np.nan)

        return feature / (volatility + 1e-8)

    def _apply_vwap_weighting(
        self,
        feature: pd.Series,
        ohlcv_data: pd.DataFrame,
        lookback_period: int = 20
    ) -> pd.Series:
        """Weight feature by VWAP ratio."""
        if 'volume' not in ohlcv_data.columns:
            return feature

        # Calculate VWAP
        typical_price = (ohlcv_data['high'] + ohlcv_data['low'] + ohlcv_data['close']) / 3
        vwap = (
            (typical_price * ohlcv_data['volume'])
            .rolling(window=lookback_period, min_periods=max(1, lookback_period // 2))
            .sum()
        ) / (
            ohlcv_data['volume']
            .rolling(window=lookback_period, min_periods=max(1, lookback_period // 2))
            .sum()
        )

        # Weight feature by price/VWAP ratio
        price_vwap_ratio = ohlcv_data['close'] / (vwap + 1e-8)
        return feature * price_vwap_ratio

    def _apply_trend_adjustment(
        self,
        feature: pd.Series,
        ohlcv_data: pd.DataFrame,
        lookback_period: int = 20
    ) -> pd.Series:
        """Adjust feature by trend strength."""
        # Calculate trend strength using price momentum
        price_momentum = ohlcv_data['close'] - ohlcv_data['close'].shift(lookback_period)
        trend_strength = abs(price_momentum) / (ohlcv_data['close'].shift(lookback_period) + 1e-8)
        trend_direction = np.sign(price_momentum)

        return feature * trend_strength * trend_direction

    def _apply_timeframe_ratio(
        self,
        feature: pd.Series,
        multiplier: int
    ) -> pd.Series:
        """
        Calculate ratio between feature and its smoothed (extended timeframe) version.

        Args:
            feature: Base feature
            multiplier: Timeframe multiplier (3, 6, 9, 27)

        Returns:
            Ratio feature
        """
        # Simulate extended timeframe by smoothing
        extended_feature = feature.rolling(
            window=multiplier,
            min_periods=max(1, multiplier // 3)
        ).mean()

        # Calculate ratio
        ratio = feature / (extended_feature + 1e-8)

        # Replace infinite values
        ratio = ratio.replace([np.inf, -np.inf], np.nan)

        return ratio

    def _apply_operator(
        self,
        left: pd.Series,
        right: pd.Series,
        operator: str
    ) -> pd.Series:
        """
        Apply mathematical operator between two features.

        Args:
            left: Left operand
            right: Right operand
            operator: Operator type

        Returns:
            Result of operation
        """
        if operator == 'multiply':
            return left * right
        elif operator == 'divide':
            return left / (right + 1e-8)
        elif operator == 'subtract':
            return left - right
        elif operator == 'add':
            return left + right
        elif operator == 'log':
            return np.log(np.abs(right) + 1) * np.sign(right)
        elif operator == 'log_ratio':
            return np.log(np.abs(left / (right + 1e-8)) + 1)
        else:
            raise ValueError(f"Unknown operator: {operator}")

    def save_metadata(
        self,
        filepath: str,
        symbol: str = None,
        exchange: str = None,
        timeframe: str = None,
        direction: str = None,
        model: str = None
    ) -> None:
        """
        Save feature metadata to a file.

        Args:
            filepath: Path to save metadata
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            direction: Trading direction
            model: Model type
        """
        store = FeatureMetadataStore()
        store.create_from_selection(
            self.selected_features,
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            direction=direction,
            model=model
        )
        store.save(filepath)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FeatureCalculator(" \
            f"selected_features={len(self.selected_features)}, " \
            f"base_features_required={len(self.base_features_required)})"
        )


class FeatureCalculatorPolars(FeatureCalculator):
    def calculate(
        self,
        ohlcv_data: "pl.DataFrame",
        base_features: Union["pl.DataFrame", Dict[str, "pl.Series"]],
        return_type: str = "dataframe",
    ):
        if pl is None:
            raise RuntimeError("polars is not available")

        ohlcv_pd = ohlcv_data.to_pandas()

        if isinstance(base_features, dict):
            base_pd = {name: series.to_pandas() for name, series in base_features.items()}
        else:
            base_pd = base_features.to_pandas()

        result = super().calculate(ohlcv_pd, base_pd, return_type="dataframe")

        if return_type == "dataframe":
            return pl.from_pandas(result)

        return {
            name: pl.Series(name=name, values=series.to_numpy())
            for name, series in result.items()
        }
