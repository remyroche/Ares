"""
Tree-Based Clustering Regime Detection

Advanced tree-based clustering for market regime detection with:
- Data-driven clustering strategy selection
- Market-specific data analysis
- Clustering quality metrics
- Multiple tree model support
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
from datetime import datetime
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class ClusteringRegimeConfig:
    """Configuration for clustering-based regime detection."""

    # Clustering strategy
    clustering_strategy: str = "auto"  # "auto", "complementary", "ensemble", "sequential", "single"
    n_regimes: int = 8

    # Data analysis thresholds
    tabular_threshold: float = 0.7
    sequential_threshold: float = 0.5
    complexity_threshold: float = 0.8
    volatility_threshold: float = 0.3
    volume_ratio_threshold: float = 2.0

    # Tree models to use
    tree_models: List[str] = field(default_factory=lambda: [
        "random_forest", "xgboost", "lightgbm", "extra_trees",
        "gradient_boosting", "ngboost", "quantile_gbdt", "dart", "deepgbm", "node"
    ])

    # Clustering metrics to calculate
    clustering_metrics: List[str] = field(default_factory=lambda: [
        "silhouette_score", "calinski_harabasz_score", "davies_bouldin_score"
    ])

    # Market data assumptions
    price_column: str = "close"
    volume_column: str = "volume"
    high_column: str = "high"
    low_column: str = "low"

    # Feature selection
    enable_feature_selection: bool = True
    max_features_per_model: int = 50
    min_feature_importance: float = 0.01


class TreeBasedClusteringRegimeDetector:
    """Tree-based clustering regime detector with advanced data-driven strategies."""

    def __init__(self, config: ClusteringRegimeConfig):
        """Initialize tree-based clustering detector."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Available tree models
        self.tree_models = {
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "extra_trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
            "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
        }

        # Clustering algorithms
        self.clustering_algorithms = {
            "kmeans": KMeans(n_clusters=config.n_regimes, random_state=42),
            "dbscan": DBSCAN(eps=0.5, min_samples=5),
            "gmm": GaussianMixture(n_components=config.n_regimes, random_state=42),
            "agglomerative": AgglomerativeClustering(n_clusters=config.n_regimes)
        }

        self.logger.info("✅ Tree-Based Clustering Regime Detector initialized")

    def detect_regimes(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect market regimes using tree-based clustering.

        Args:
            market_data: Market data (OHLCV)

        Returns:
            Regime detection results
        """
        self.logger.info("🚀 Starting tree-based regime detection...")

        try:
            # Step 1: Analyze market data characteristics
            data_characteristics = self._analyze_market_data(market_data)
            self.logger.info(f"📊 Data characteristics: {data_characteristics}")

            # Step 2: Choose clustering strategy
            clustering_strategy = self._choose_clustering_strategy(data_characteristics)
            self.logger.info(f"🎯 Selected clustering strategy: {clustering_strategy}")

            # Step 3: Perform clustering based on strategy
            if clustering_strategy == 'complementary':
                results = self._complementary_clustering(market_data, data_characteristics)
            elif clustering_strategy == 'ensemble':
                results = self._ensemble_clustering(market_data)
            elif clustering_strategy == 'sequential':
                results = self._sequential_clustering(market_data)
            elif clustering_strategy == 'single':
                results = self._single_model_clustering(market_data)
            else:
                results = self._auto_clustering(market_data, data_characteristics)

            # Step 4: Calculate clustering metrics
            results['clustering_metrics'] = self._calculate_clustering_metrics(
                results['features'], results['labels']
            )

            # Step 5: Add metadata
            results.update({
                'strategy': clustering_strategy,
                'data_characteristics': data_characteristics,
                'timestamp': datetime.now().isoformat(),
                'method': 'tree_based_clustering'
            })

            self.logger.info("✅ Tree-based regime detection completed")
            return results

        except Exception as e:
            self.logger.error(f"❌ Tree-based regime detection failed: {e}")
            raise

    def _analyze_market_data(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market data characteristics to guide clustering strategy."""
        try:
            # Basic data characteristics
            n_samples = len(market_data)
            n_features = len(market_data.columns)

            # Calculate ratios for strategy selection
            tabular_ratio = self._calculate_tabular_ratio(market_data)
            sequential_ratio = self._calculate_sequential_ratio(market_data)
            complexity_ratio = self._calculate_complexity_ratio(market_data)

            # Calculate market characteristics
            volatility = market_data[self.config.price_column].pct_change().std()
            volume_ratio = (
                market_data[self.config.volume_column].mean() /
                market_data[self.config.volume_column].std()
            )
            price_range = (
                (market_data[self.config.high_column].max() - market_data[self.config.low_column].min()) /
                market_data[self.config.price_column].mean()
            )

            characteristics = {
                'n_samples': n_samples,
                'n_features': n_features,
                'tabular_ratio': tabular_ratio,
                'sequential_ratio': sequential_ratio,
                'complexity_ratio': complexity_ratio,
                'volatility': volatility,
                'volume_ratio': volume_ratio,
                'price_range': price_range,
                'is_tabular_dominant': tabular_ratio > self.config.tabular_threshold,
                'is_sequential_dominant': sequential_ratio > self.config.sequential_threshold,
                'is_complex_dominant': complexity_ratio > self.config.complexity_threshold,
                'is_volatile': volatility > self.config.volatility_threshold,
                'has_high_volume_ratio': volume_ratio > self.config.volume_ratio_threshold
            }

            return characteristics

        except Exception as e:
            self.logger.warning(f"Data analysis failed: {e}")
            return {'n_samples': len(market_data), 'n_features': len(market_data.columns)}

    def _calculate_tabular_ratio(self, market_data: pd.DataFrame) -> float:
        """Calculate ratio of tabular features in market data."""
        try:
            position = np.arange(len(market_data))
            correlations = []

            for column in market_data.columns:
                if market_data[column].dtype in ['float64', 'int64']:
                    corr = np.corrcoef(market_data[column].values, position)[0, 1]
                    correlations.append(abs(corr))

            # Tabular features have low correlation with time
            tabular_features = sum(1 for corr in correlations if corr < 0.3)
            return tabular_features / len(correlations) if correlations else 0.5

        except Exception as e:
            tprint_warning(f"⚠️ Failed to calculate tabular feature score: {e}")
            return 0.5

    def _calculate_sequential_ratio(self, market_data: pd.DataFrame) -> float:
        """Calculate ratio of sequential features in market data."""
        try:
            # Calculate autocorrelation for price and volume
            price_autocorr = market_data[self.config.price_column].autocorr(lag=1)
            volume_autocorr = market_data[self.config.volume_column].autocorr(lag=1)

            # Sequential features have high autocorrelation
            sequential_features = sum(1 for ac in [price_autocorr, volume_autocorr] if abs(ac) > 0.3)
            return sequential_features / 2.0

        except Exception as e:
            tprint_warning(f"⚠️ Failed to calculate sequential feature score: {e}")
            return 0.3

    def _calculate_complexity_ratio(self, market_data: pd.DataFrame) -> float:
        """Calculate ratio of complex features in market data."""
        try:
            complexities = []

            for column in market_data.columns:
                if market_data[column].dtype in ['float64', 'int64']:
                    feature = market_data[column].values
                    variance = np.var(feature)
                    # Simple non-linearity measure
                    sorted_feature = np.sort(feature)
                    non_linearity = np.var(np.diff(sorted_feature))
                    complexity = variance * non_linearity
                    complexities.append(complexity)

            # Normalize and calculate ratio
            max_complexity = max(complexities) if complexities else 1.0
            complex_features = sum(1 for c in complexities if c > 0.5 * max_complexity)
            return complex_features / len(complexities) if complexities else 0.5

        except Exception as e:
            tprint_warning(f"⚠️ Failed to calculate tabular feature score: {e}")
            return 0.5

    def _choose_clustering_strategy(self, data_characteristics: Dict[str, Any]) -> str:
        """Choose the best clustering strategy based on data characteristics."""
        try:
            # Use data characteristics to determine strategy
            if data_characteristics.get('is_tabular_dominant', False):
                return 'complementary'  # Tree for tabular, ensemble for patterns
            elif data_characteristics.get('is_sequential_dominant', False):
                return 'sequential'  # Sequential processing with trees
            elif data_characteristics.get('is_complex_dominant', False):
                return 'ensemble'  # Combine multiple tree approaches
            elif data_characteristics.get('is_volatile', False):
                return 'single'  # Single robust model for volatile markets
            else:
                return self.config.clustering_strategy if self.config.clustering_strategy != 'auto' else 'ensemble'

        except Exception as e:
            self.logger.warning(f"Strategy selection failed: {e}")
            return 'ensemble'  # Default to ensemble

    def _complementary_clustering(self, market_data: pd.DataFrame, data_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Perform complementary clustering using multiple tree approaches."""
        self.logger.info("🔍 Starting complementary clustering...")

        try:
            # Step 1: Feature selection using Random Forest
            features = self._extract_features(market_data)
            rf_model = self.tree_models['random_forest']

            # Use Random Forest for feature importance
            rf_model.fit(features, np.arange(len(market_data)) % self.config.n_regimes)
            feature_importances = rf_model.feature_importances_

            # Select top features
            top_feature_indices = np.argsort(feature_importances)[-self.config.max_features_per_model:]
            selected_features = features[:, top_feature_indices]

            # Step 2: Clustering with selected features
            kmeans = KMeans(n_clusters=self.config.n_regimes, random_state=42)
            labels = kmeans.fit_predict(selected_features)
            centers = kmeans.cluster_centers_

            return {
                'features': selected_features,
                'labels': labels,
                'cluster_centers': centers,
                'feature_importances': feature_importances,
                'selected_feature_indices': top_feature_indices,
                'method': 'complementary'
            }

        except Exception as e:
            self.logger.error(f"Complementary clustering failed: {e}")
            raise

    def _ensemble_clustering(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform ensemble clustering combining multiple tree approaches."""
        self.logger.info("🔍 Starting ensemble clustering...")

        try:
            features = self._extract_features(market_data)

            # Get predictions from multiple tree models
            ensemble_predictions = []

            for model_name, model in self.tree_models.items():
                try:
                    # Create dummy labels for training
                    dummy_labels = np.arange(len(market_data)) % self.config.n_regimes
                    model.fit(features, dummy_labels)
                    predictions = model.predict_proba(features)
                    ensemble_predictions.append(predictions)
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to predict with model: {e}")
                    continue

            if not ensemble_predictions:
                raise ValueError("No tree models could be trained")

            # Average predictions from ensemble
            avg_predictions = np.mean(ensemble_predictions, axis=0)
            labels = np.argmax(avg_predictions, axis=1)

            # Use KMeans on ensemble predictions for final clustering
            kmeans = KMeans(n_clusters=self.config.n_regimes, random_state=42)
            final_labels = kmeans.fit_predict(avg_predictions)
            centers = kmeans.cluster_centers_

            return {
                'features': avg_predictions,
                'labels': final_labels,
                'cluster_centers': centers,
                'ensemble_predictions': avg_predictions,
                'method': 'ensemble'
            }

        except Exception as e:
            self.logger.error(f"Ensemble clustering failed: {e}")
            raise

    def _sequential_clustering(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform sequential clustering using tree-first approach."""
        self.logger.info("🔍 Starting sequential clustering...")

        try:
            # Step 1: Initial clustering with Random Forest
            features = self._extract_features(market_data)
            rf_model = self.tree_models['random_forest']

            dummy_labels = np.arange(len(market_data)) % self.config.n_regimes
            rf_model.fit(features, dummy_labels)
            rf_predictions = rf_model.predict_proba(features)

            # Step 2: Refine with Gradient Boosting
            gb_model = self.tree_models['gradient_boosting']
            gb_model.fit(rf_predictions, dummy_labels)
            gb_predictions = gb_model.predict_proba(rf_predictions)

            # Step 3: Final clustering
            kmeans = KMeans(n_clusters=self.config.n_regimes, random_state=42)
            labels = kmeans.fit_predict(gb_predictions)
            centers = kmeans.cluster_centers_

            return {
                'features': gb_predictions,
                'labels': labels,
                'cluster_centers': centers,
                'rf_predictions': rf_predictions,
                'gb_predictions': gb_predictions,
                'method': 'sequential'
            }

        except Exception as e:
            self.logger.error(f"Sequential clustering failed: {e}")
            raise

    def _single_model_clustering(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform clustering using single best tree model."""
        self.logger.info("🔍 Starting single model clustering...")

        try:
            features = self._extract_features(market_data)

            # Use Random Forest as it's generally most robust
            model = self.tree_models['random_forest']
            dummy_labels = np.arange(len(market_data)) % self.config.n_regimes
            model.fit(features, dummy_labels)
            predictions = model.predict_proba(features)

            # Final clustering
            kmeans = KMeans(n_clusters=self.config.n_regimes, random_state=42)
            labels = kmeans.fit_predict(predictions)
            centers = kmeans.cluster_centers_

            return {
                'features': predictions,
                'labels': labels,
                'cluster_centers': centers,
                'method': 'single'
            }

        except Exception as e:
            self.logger.error(f"Single model clustering failed: {e}")
            raise

    def _auto_clustering(self, market_data: pd.DataFrame, data_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Auto-select clustering strategy based on data characteristics."""
        self.logger.info("🔍 Starting auto clustering...")

        try:
            # Try ensemble first as it's most robust
            return self._ensemble_clustering(market_data)

        except Exception as e:
            self.logger.warning(f"Auto clustering failed, falling back to single: {e}")
            return self._single_model_clustering(market_data)

    def _extract_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Extract features from market data."""
        try:
            # Basic price features
            close_prices = market_data[self.config.price_column].values.reshape(-1, 1)

            # Returns features
            returns = np.diff(close_prices.ravel(), prepend=close_prices[0])
            returns = returns.reshape(-1, 1)

            # Volatility features
            volatility = pd.Series(close_prices.ravel()).rolling(window=10).std().values
            volatility = volatility.reshape(-1, 1)

            # Volume features
            volume = market_data[self.config.volume_column].values.reshape(-1, 1)

            # Combine features
            features = np.hstack([close_prices, returns, volatility, volume])

            # Remove NaN values
            mask = ~np.isnan(features).any(axis=1)
            features = features[mask]

            # Standardize features
            scaler = StandardScaler()
            features = scaler.fit_transform(features)

            return features

        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            # Return basic features if extraction fails
            close_prices = market_data[self.config.price_column].values.reshape(-1, 1)
            return StandardScaler().fit_transform(close_prices)

    def _calculate_clustering_metrics(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate clustering quality metrics."""
        try:
            metrics = {}

            # Silhouette score
            try:
                metrics['silhouette_score'] = silhouette_score(features, labels)
            except Exception as e:
                tprint_warning(f"⚠️ Failed to calculate silhouette score: {e}")
                metrics['silhouette_score'] = 0.0

            # Calinski-Harabasz score
            try:
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(features, labels)
            except Exception as e:
                tprint_warning(f"⚠️ Failed to calculate Calinski-Harabasz score: {e}")
                metrics['calinski_harabasz_score'] = 0.0

            # Davies-Bouldin score
            try:
                metrics['davies_bouldin_score'] = davies_bouldin_score(features, labels)
            except Exception as e:
                tprint_warning(f"⚠️ Failed to calculate Davies-Bouldin score: {e}")
                metrics['davies_bouldin_score'] = 0.0

            return metrics

        except Exception as e:
            self.logger.warning(f"Clustering metrics calculation failed: {e}")
            return {'silhouette_score': 0.0, 'calinski_harabasz_score': 0.0, 'davies_bouldin_score': 0.0}


# Convenience functions
def create_clustering_regime_detector(n_regimes: int = 8, strategy: str = "auto") -> TreeBasedClusteringRegimeDetector:
    """Create a tree-based clustering regime detector."""
    config = ClusteringRegimeConfig(
        clustering_strategy=strategy,
        n_regimes=n_regimes
    )
    return TreeBasedClusteringRegimeDetector(config)


def quick_clustering_detection(market_data: pd.DataFrame, n_regimes: int = 8) -> Dict[str, Any]:
    """Quick tree-based clustering regime detection."""
    detector = create_clustering_regime_detector(n_regimes)
    return detector.detect_regimes(market_data)