# src/training/feature_integration.py

"""
Feature Integration Module for ML Training Pipeline.
Ensures liquidity features from advanced feature engineering are properly integrated
into the ML model training process.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    warning,
    critical,
    problem,
    failed,
    invalid,
    missing,
    timeout,
    connection_error,
    validation_error,
    initialization_error,
    execution_error,
)


class FeatureIntegrationManager:
    """
    Manages integration of advanced features (including liquidity features)
    into the ML training pipeline.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("FeatureIntegrationManager")

        # Configuration
        self.feature_config = config.get("feature_integration", {})
        self.enable_liquidity_features = self.feature_config.get(
            "enable_liquidity_features",
            True,
        )
        self.enable_advanced_features = self.feature_config.get(
            "enable_advanced_features",
            True,
        )
        self.feature_selection_method = self.feature_config.get(
            "feature_selection_method",
            "correlation",
        )

        # Feature engineering components
        self.advanced_feature_engineering = None
        self.feature_scaler = StandardScaler()
        self.feature_pca = PCA(n_components=0.95)  # Keep 95% variance

        self.is_initialized = False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="feature integration initialization",
    )
    async def initialize(self) -> bool:
        """Initialize feature integration manager."""
        try:
            self.logger.info("🚀 Initializing feature integration manager...")

            # Initialize advanced feature engineering
            if self.enable_advanced_features:
                from src.analyst.advanced_feature_engineering import (
                    AdvancedFeatureEngineering,
                )

                self.advanced_feature_engineering = AdvancedFeatureEngineering(
                    self.config,
                )
                await self.advanced_feature_engineering.initialize()

            self.is_initialized = True
            self.logger.info("✅ Feature integration manager initialized successfully")
            return True

        except Exception as e:
            self.logger.exception(
                f"❌ Error initializing feature integration manager: {e}",
            )
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="feature integration",
    )
    async def integrate_features(
        self,
        historical_data: pd.DataFrame,
        market_data: pd.DataFrame,
        order_flow_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Integrate advanced features (including liquidity features) into training data.

        Args:
            historical_data: Historical price and volume data
            market_data: Current market data
            order_flow_data: Order flow data (optional)

        Returns:
            DataFrame with integrated features
        """
        try:
            if not self.is_initialized:
                self.print(
                    initialization_error("Feature integration manager not initialized")
                )
                return historical_data

            # Start with original data
            integrated_data = historical_data.copy()

            # Add advanced features including liquidity features
            if self.advanced_feature_engineering:
                advanced_features = await self._add_advanced_features(
                    historical_data,
                    market_data,
                    order_flow_data,
                )
                integrated_data = pd.concat(
                    [integrated_data, advanced_features],
                    axis=1,
                )

            # Add liquidity-specific features
            if self.enable_liquidity_features:
                liquidity_features = await self._add_liquidity_features(
                    historical_data,
                    market_data,
                    order_flow_data,
                )
                integrated_data = pd.concat(
                    [integrated_data, liquidity_features],
                    axis=1,
                )

            # Feature selection and dimensionality reduction
            selected_features = self._select_optimal_features(integrated_data)

            self.logger.info(f"✅ Integrated {len(selected_features.columns)} features")
            return selected_features

        except Exception as e:
            self.print(error("Error integrating features: {e}"))
            return historical_data

    async def _add_advanced_features(
        self,
        historical_data: pd.DataFrame,
        market_data: pd.DataFrame,
        order_flow_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Add advanced features from advanced feature engineering."""
        try:
            # Prepare data for advanced feature engineering
            price_data = historical_data[["open", "high", "low", "close"]].copy()
            volume_data = historical_data[["volume"]].copy()

            # Get advanced features
            advanced_features = (
                await self.advanced_feature_engineering.engineer_features(
                    price_data=price_data,
                    volume_data=volume_data,
                    order_flow_data=order_flow_data,
                )
            )

            # Convert to DataFrame
            features_df = pd.DataFrame([advanced_features])

            # Replicate for all rows in historical data
            features_df = pd.concat(
                [features_df] * len(historical_data),
                ignore_index=True,
            )
            features_df.index = historical_data.index

            return features_df

        except Exception as e:
            self.print(error("Error adding advanced features: {e}"))
            return pd.DataFrame()

    async def _add_liquidity_features(
        self,
        historical_data: pd.DataFrame,
        market_data: pd.DataFrame,
        order_flow_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Add liquidity-specific features."""
        try:
            liquidity_features = {}

            # Calculate basic liquidity metrics
            volume = historical_data["volume"]
            price_changes = historical_data["close"].pct_change()

            # Volume-based liquidity
            liquidity_features["volume_liquidity"] = volume / volume.rolling(20).mean()

            # Price impact
            liquidity_features["price_impact"] = np.abs(price_changes) / volume
            liquidity_features["price_impact"] = (
                liquidity_features["price_impact"].rolling(20).mean()
            )

            # Amihud illiquidity
            liquidity_features["amihud_illiquidity"] = np.abs(price_changes) / volume
            liquidity_features["amihud_illiquidity"] = (
                liquidity_features["amihud_illiquidity"].rolling(20).mean()
            )

            # Kyle's lambda
            liquidity_features["kyle_lambda"] = (
                np.abs(price_changes).rolling(50).mean() / volume.rolling(50).mean()
            )

            # Volume rate of change
            liquidity_features["volume_roc"] = volume.pct_change(5)

            # Volume moving average ratio
            liquidity_features["volume_ma_ratio"] = volume / volume.rolling(20).mean()

            # Liquidity regime classification
            liquidity_percentile = liquidity_features["volume_liquidity"].rank(pct=True)
            liquidity_features["liquidity_regime"] = liquidity_percentile.apply(
                lambda x: "high" if x > 0.8 else "low" if x < 0.2 else "medium",
            )
            liquidity_features["liquidity_percentile"] = liquidity_percentile

            # Liquidity stress calculation
            liquidity_features["liquidity_stress"] = self._calculate_liquidity_stress(
                liquidity_features,
            )

            # Convert to DataFrame
            return pd.DataFrame(liquidity_features)

        except Exception as e:
            self.print(error("Error adding liquidity features: {e}"))
            return pd.DataFrame()

    def _calculate_liquidity_stress(
        self,
        liquidity_features: dict[str, pd.Series],
    ) -> pd.Series:
        """Calculate liquidity stress score."""
        try:
            stress_factors = []

            # Volume-based stress
            volume_liquidity = liquidity_features.get(
                "volume_liquidity",
                pd.Series(1.0),
            )
            volume_stress = pd.Series(0.1, index=volume_liquidity.index)
            volume_stress[volume_liquidity < 0.5] = 0.8
            volume_stress[(volume_liquidity >= 0.5) & (volume_liquidity < 0.8)] = 0.4
            stress_factors.append(volume_stress)

            # Price impact stress
            price_impact = liquidity_features.get("price_impact", pd.Series(0.0))
            impact_stress = pd.Series(0.1, index=price_impact.index)
            impact_stress[price_impact > 0.001] = 0.9
            impact_stress[(price_impact > 0.0005) & (price_impact <= 0.001)] = 0.5
            stress_factors.append(impact_stress)

            # Amihud illiquidity stress
            amihud_illiquidity = liquidity_features.get(
                "amihud_illiquidity",
                pd.Series(0.0),
            )
            amihud_stress = pd.Series(0.1, index=amihud_illiquidity.index)
            amihud_stress[amihud_illiquidity > 0.01] = 0.9
            amihud_stress[
                (amihud_illiquidity > 0.005) & (amihud_illiquidity <= 0.01)
            ] = 0.5
            stress_factors.append(amihud_stress)

            # Calculate average stress score
            if stress_factors:
                stress_df = pd.concat(stress_factors, axis=1)
                return stress_df.mean(axis=1)

            return pd.Series(
                0.0,
                index=liquidity_features.get("volume_liquidity", pd.Series()).index,
            )

        except Exception as e:
            self.print(error("Error calculating liquidity stress: {e}"))
            return pd.Series(0.0)

    def _select_optimal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Select optimal features using correlation analysis and PCA."""
        try:
            # Remove NaN values
            data_clean = data.dropna()

            if data_clean.empty:
                return data

            # Remove constant features
            data_clean = data_clean.loc[:, data_clean.std() > 0]

            # Remove highly correlated features
            if len(data_clean.columns) > 1:
                correlation_matrix = data_clean.corr()
                upper_triangle = np.triu(np.ones_like(correlation_matrix, dtype=bool))
                high_correlation = np.abs(correlation_matrix) > 0.95
                high_correlation = high_correlation & upper_triangle

                to_drop = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i + 1, len(correlation_matrix.columns)):
                        if high_correlation.iloc[i, j]:
                            to_drop.append(correlation_matrix.columns[j])

                data_clean = data_clean.drop(columns=list(set(to_drop)))

            # Apply PCA for dimensionality reduction if needed
            if len(data_clean.columns) > 50:  # Only if we have many features
                try:
                    # Scale and reduce within CV folds or train-only sections to avoid lookahead
                    scaled_features = self.feature_scaler.fit_transform(data_clean)
                    pca_features = self.feature_pca.fit_transform(scaled_features)

                    # Create new DataFrame with PCA features
                    pca_df = pd.DataFrame(
                        pca_features,
                        index=data_clean.index,
                        columns=[
                            f"pca_component_{i}" for i in range(pca_features.shape[1])
                        ],
                    )

                    self.logger.info(
                        f"Applied PCA: {len(data_clean.columns)} -> {pca_features.shape[1]} features",
                    )
                    return pca_df

                except Exception as e:
                    self.logger.error(f"PCA failed, using original features: {e}")
                    return data_clean

            return data_clean

        except Exception as e:
            self.logger.error(f"Error selecting optimal features: {e}")
            return data

    def get_feature_importance(
        self,
        model,
        feature_names: list[str],
    ) -> dict[str, float]:
        """Get feature importance from trained model."""
        try:
            if hasattr(model, "feature_importances_"):
                importance_dict = dict(
                    zip(feature_names, model.feature_importances_, strict=False),
                )
                return dict(
                    sorted(importance_dict.items(), key=lambda x: x[1], reverse=True),
                )
            if hasattr(model, "coef_"):
                importance_dict = dict(
                    zip(feature_names, np.abs(model.coef_[0]), strict=False),
                )
                return dict(
                    sorted(importance_dict.items(), key=lambda x: x[1], reverse=True),
                )
            return {}

        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            return {}

    def get_liquidity_feature_summary(self, data: pd.DataFrame) -> dict[str, Any]:
        """Get summary of liquidity features in the dataset."""
        try:
            liquidity_features = [
                "volume_liquidity",
                "price_impact",
                "spread_liquidity",
                "liquidity_regime",
                "liquidity_percentile",
                "kyle_lambda",
                "amihud_illiquidity",
                "order_flow_imbalance",
                "large_order_ratio",
                "vwap",
                "volume_roc",
                "volume_ma_ratio",
                "liquidity_stress",
            ]

            available_features = [f for f in liquidity_features if f in data.columns]

            summary = {
                "total_liquidity_features": len(available_features),
                "available_liquidity_features": available_features,
                "liquidity_feature_coverage": len(available_features)
                / len(liquidity_features),
                "feature_statistics": {},
            }

            for feature in available_features:
                if feature in data.columns:
                    feature_data = data[feature].dropna()
                    if not feature_data.empty:
                        summary["feature_statistics"][feature] = {
                            "mean": feature_data.mean(),
                            "std": feature_data.std(),
                            "min": feature_data.min(),
                            "max": feature_data.max(),
                            "missing_pct": (data[feature].isna().sum() / len(data))
                            * 100,
                        }

            return summary

        except Exception as e:
            self.print(error("Error getting liquidity feature summary: {e}"))
            return {}
