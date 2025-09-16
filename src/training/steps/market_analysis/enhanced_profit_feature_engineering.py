"""
Enhanced Profit Potential Feature Engineering for ML Models

This module provides comprehensive feature engineering capabilities specifically
designed to leverage the enhanced profit potential labels from the triple barrier
method. It creates rich, ML-friendly features that help models learn from
profit magnitude, confidence, and category information.

Key Features:
- Profit category-based features (one-hot, ordinal, embeddings)
- Profit magnitude features (continuous, normalized, transformed)
- Confidence-based features (reliability, uncertainty, calibration)
- Regime-specific profit features
- Volatility-adjusted profit features
- Interaction features between profit components
- Time-series profit features (momentum, trends, patterns)
- Risk-adjusted profit features
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime

import pandas as pd
import numpy as np

from src.utils.tprint import tprint
from src.utils.logger import get_logger

# Import scikit-learn for advanced feature engineering
try:
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
    from sklearn.feature_selection import mutual_info_regression, f_regression
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Import PyTorch for neural network features
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

@dataclass
class EnhancedProfitFeatureConfig:
    """Configuration for enhanced profit feature engineering."""
    
    # Feature categories to generate
    enable_category_features: bool = True
    enable_magnitude_features: bool = True
    enable_confidence_features: bool = True
    enable_regime_features: bool = True
    enable_volatility_features: bool = True
    enable_interaction_features: bool = True
    enable_timeseries_features: bool = True
    enable_risk_features: bool = True
    
    # Advanced feature engineering
    enable_embedding_features: bool = True
    enable_clustering_features: bool = True
    enable_pca_features: bool = True
    enable_neural_features: bool = True
    
    # Feature selection and scaling
    enable_feature_selection: bool = True
    enable_feature_scaling: bool = True
    scaling_method: str = "robust"  # "standard", "robust", "minmax"
    
    # Time-series parameters
    timeseries_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    momentum_windows: List[int] = field(default_factory=lambda: [3, 7, 14])
    
    # Clustering parameters
    n_clusters: int = 5
    clustering_features: List[str] = field(default_factory=lambda: [
        'profit_magnitude_score', 'confidence_score', 'potential_profit_pct'
    ])
    
    # PCA parameters
    n_pca_components: int = 3
    pca_features: List[str] = field(default_factory=lambda: [
        'profit_magnitude_score', 'confidence_score', 'potential_profit_pct',
        'profit_magnitude_log', 'confidence_log'
    ])
    
    # Neural network parameters
    embedding_dim: int = 8
    hidden_dims: List[int] = field(default_factory=lambda: [16, 8])
    
    # Feature selection parameters
    selection_method: str = "mutual_info"  # "mutual_info", "f_score", "correlation"
    max_features: Optional[int] = None
    selection_threshold: float = 0.01

class EnhancedProfitFeatureEngineering:
    """Enhanced profit feature engineering for ML models."""
    
    def __init__(self, config: Optional[EnhancedProfitFeatureConfig] = None):
        """Initialize the enhanced profit feature engineering system."""
        self.config = config or EnhancedProfitFeatureConfig()
        self.logger = get_logger('EnhancedProfitFeatureEngineering')
        
        # Initialize scalers
        if self.config.enable_feature_scaling:
            if self.config.scaling_method == "standard":
                self.scaler = StandardScaler()
            elif self.config.scaling_method == "robust":
                self.scaler = RobustScaler()
            elif self.config.scaling_method == "minmax":
                self.scaler = MinMaxScaler()
            else:
                self.scaler = RobustScaler()  # Default
        else:
            self.scaler = None
        
        # Initialize feature selection results
        self.selected_features = []
        self.feature_importance_scores = {}
        
        # Initialize models for advanced features
        self.pca_model = None
        self.clustering_model = None
        self.embedding_model = None
        
        self.logger.info("🔧 Enhanced Profit Feature Engineering initialized")
        tprint("🔧 Enhanced Profit Feature Engineering initialized")
    
    def apply_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply all enhanced profit features to the data."""
        start_time = time.time()
        
        tprint("🚀 Starting Enhanced Profit Feature Engineering")
        self.logger.info("🚀 Starting Enhanced Profit Feature Engineering")
        
        # Validate input data
        required_columns = ['profit_category', 'profit_magnitude_score', 'confidence_score']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        result_data = data.copy()
        
        # Apply feature categories
        if self.config.enable_category_features:
            tprint("📊 Generating category features...")
            result_data = self._apply_category_features(result_data)
            tprint("✅ Category features generated")
        
        if self.config.enable_magnitude_features:
            tprint("📊 Generating magnitude features...")
            result_data = self._apply_magnitude_features(result_data)
            tprint("✅ Magnitude features generated")
        
        if self.config.enable_confidence_features:
            tprint("📊 Generating confidence features...")
            result_data = self._apply_confidence_features(result_data)
            tprint("✅ Confidence features generated")
        
        if self.config.enable_regime_features and 'hmm_regime' in data.columns:
            tprint("📊 Generating regime features...")
            result_data = self._apply_regime_features(result_data)
            tprint("✅ Regime features generated")
        
        if self.config.enable_volatility_features:
            tprint("📊 Generating volatility features...")
            result_data = self._apply_volatility_features(result_data)
            tprint("✅ Volatility features generated")
        
        if self.config.enable_interaction_features:
            tprint("📊 Generating interaction features...")
            result_data = self._apply_interaction_features(result_data)
            tprint("✅ Interaction features generated")
        
        if self.config.enable_timeseries_features:
            tprint("📊 Generating time-series features...")
            result_data = self._apply_timeseries_features(result_data)
            tprint("✅ Time-series features generated")
        
        if self.config.enable_risk_features:
            tprint("📊 Generating risk features...")
            result_data = self._apply_risk_features(result_data)
            tprint("✅ Risk features generated")
        
        # Apply advanced features
        if self.config.enable_clustering_features and SKLEARN_AVAILABLE:
            tprint("📊 Generating clustering features...")
            result_data = self._apply_clustering_features(result_data)
            tprint("✅ Clustering features generated")
        
        if self.config.enable_pca_features and SKLEARN_AVAILABLE:
            tprint("📊 Generating PCA features...")
            result_data = self._apply_pca_features(result_data)
            tprint("✅ PCA features generated")
        
        if self.config.enable_embedding_features and TORCH_AVAILABLE:
            tprint("📊 Generating embedding features...")
            result_data = self._apply_embedding_features(result_data)
            tprint("✅ Embedding features generated")
        
        # Apply feature selection
        if self.config.enable_feature_selection:
            tprint("📊 Applying feature selection...")
            result_data = self._apply_feature_selection(result_data)
            tprint("✅ Feature selection applied")
        
        # Apply feature scaling
        if self.config.enable_feature_scaling:
            tprint("📊 Applying feature scaling...")
            result_data = self._apply_feature_scaling(result_data)
            tprint("✅ Feature scaling applied")
        
        # Calculate performance metrics
        processing_time = time.time() - start_time
        features_added = len(result_data.columns) - len(data.columns)
        
        tprint(f"✅ Enhanced profit feature engineering completed")
        tprint(f"   Duration: {processing_time:.2f}s")
        tprint(f"   Features added: {features_added}")
        tprint(f"   Total features: {len(result_data.columns)}")
        
        self.logger.info(f"✅ Enhanced profit feature engineering completed in {processing_time:.2f}s")
        
        return result_data
    
    def _apply_category_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply profit category-based features."""
        
        # One-hot encoding for profit categories
        if 'profit_category' in data.columns:
            category_dummies = pd.get_dummies(data['profit_category'], prefix='profit_cat')
            data = pd.concat([data, category_dummies], axis=1)
        
        # One-hot encoding for confidence categories
        if 'confidence_category' in data.columns:
            confidence_dummies = pd.get_dummies(data['confidence_category'], prefix='conf_cat')
            data = pd.concat([data, confidence_dummies], axis=1)
        
        # Ordinal encoding for profit categories
        if 'profit_category_ordinal' in data.columns:
            data['profit_category_ordinal_scaled'] = data['profit_category_ordinal'] / 8.0  # Scale to 0-1
        
        # Category frequency features
        if 'profit_category' in data.columns:
            category_counts = data['profit_category'].value_counts()
            data['profit_category_frequency'] = data['profit_category'].map(category_counts) / len(data)
        
        # Category transition features (if time series)
        if 'profit_category' in data.columns and len(data) > 1:
            data['profit_category_changed'] = (data['profit_category'] != data['profit_category'].shift(1)).astype(int)
            data['profit_category_streak'] = self._calculate_category_streaks(data['profit_category'])
        
        return data
    
    def _apply_magnitude_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply profit magnitude-based features."""
        
        if 'profit_magnitude_score' not in data.columns:
            return data
        
        magnitude = data['profit_magnitude_score']
        
        # Basic transformations
        data['profit_magnitude_log'] = np.log1p(magnitude)
        data['profit_magnitude_sqrt'] = np.sqrt(magnitude)
        data['profit_magnitude_squared'] = magnitude ** 2
        data['profit_magnitude_cubed'] = magnitude ** 3
        
        # Normalized features
        data['profit_magnitude_normalized'] = (magnitude - magnitude.mean()) / magnitude.std()
        data['profit_magnitude_percentile'] = magnitude.rank(pct=True)
        
        # Magnitude categories
        data['profit_magnitude_high'] = (magnitude >= 7).astype(int)
        data['profit_magnitude_medium'] = ((magnitude >= 4) & (magnitude < 7)).astype(int)
        data['profit_magnitude_low'] = (magnitude < 4).astype(int)
        
        # Magnitude ratios
        if 'potential_profit_pct' in data.columns:
            profit_pct = data['potential_profit_pct']
            data['profit_magnitude_to_pct_ratio'] = np.where(
                profit_pct != 0, magnitude / np.abs(profit_pct), 0
            )
        
        return data
    
    def _apply_confidence_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply confidence-based features."""
        
        if 'confidence_score' not in data.columns:
            return data
        
        confidence = data['confidence_score']
        
        # Basic transformations
        data['confidence_log'] = np.log1p(confidence)
        data['confidence_sqrt'] = np.sqrt(confidence)
        data['confidence_squared'] = confidence ** 2
        
        # Confidence categories
        data['confidence_very_high'] = (confidence >= 0.8).astype(int)
        data['confidence_high'] = ((confidence >= 0.6) & (confidence < 0.8)).astype(int)
        data['confidence_medium'] = ((confidence >= 0.4) & (confidence < 0.6)).astype(int)
        data['confidence_low'] = ((confidence >= 0.2) & (confidence < 0.4)).astype(int)
        data['confidence_very_low'] = (confidence < 0.2).astype(int)
        
        # Uncertainty features
        data['uncertainty_score'] = 1.0 - confidence
        data['uncertainty_log'] = np.log1p(data['uncertainty_score'])
        
        # Confidence stability (rolling confidence)
        if len(data) > 10:
            data['confidence_stability'] = confidence.rolling(window=10).std()
            data['confidence_trend'] = confidence.rolling(window=10).mean().diff()
        
        return data
    
    def _apply_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply regime-specific profit features."""
        
        if 'hmm_regime' not in data.columns:
            return data
        
        regime = data['hmm_regime']
        
        # Regime one-hot encoding
        regime_dummies = pd.get_dummies(regime, prefix='regime')
        data = pd.concat([data, regime_dummies], axis=1)
        
        # Regime-specific profit features
        if 'profit_magnitude_score' in data.columns:
            for regime_val in regime.unique():
                if not pd.isna(regime_val):
                    regime_mask = regime == regime_val
                    regime_magnitude = data.loc[regime_mask, 'profit_magnitude_score']
                    if len(regime_magnitude) > 0:
                        data[f'profit_magnitude_regime_{int(regime_val)}'] = np.where(
                            regime_mask, regime_magnitude, 0
                        )
        
        # Regime transitions
        if len(data) > 1:
            data['regime_changed'] = (regime != regime.shift(1)).astype(int)
            data['regime_streak'] = self._calculate_regime_streaks(regime)
        
        return data
    
    def _apply_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply volatility-based profit features."""
        
        if 'close' not in data.columns:
            return data
        
        # Calculate price volatility
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=20).std()
        
        # Volatility features
        data['price_volatility'] = volatility
        data['volatility_percentile'] = volatility.rank(pct=True)
        data['high_volatility'] = (volatility > volatility.quantile(0.8)).astype(int)
        data['low_volatility'] = (volatility < volatility.quantile(0.2)).astype(int)
        
        # Volatility-adjusted profit features
        if 'profit_magnitude_score' in data.columns:
            data['profit_magnitude_vol_adjusted'] = np.where(
                volatility > 0, data['profit_magnitude_score'] / volatility, 0
            )
        
        if 'confidence_score' in data.columns:
            data['confidence_vol_adjusted'] = np.where(
                volatility > 0, data['confidence_score'] / volatility, 0
            )
        
        return data
    
    def _apply_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply interaction features between profit components."""
        
        # Magnitude-Confidence interactions
        if 'profit_magnitude_score' in data.columns and 'confidence_score' in data.columns:
            magnitude = data['profit_magnitude_score']
            confidence = data['confidence_score']
            
            data['profit_confidence_interaction'] = magnitude * confidence
            data['profit_confidence_ratio'] = np.where(
                confidence > 0, magnitude / confidence, 0
            )
            data['profit_confidence_sum'] = magnitude + confidence
            data['profit_confidence_diff'] = magnitude - confidence
        
        # Category-Magnitude interactions
        if 'profit_category_ordinal' in data.columns and 'profit_magnitude_score' in data.columns:
            data['category_magnitude_interaction'] = (
                data['profit_category_ordinal'] * data['profit_magnitude_score']
            )
        
        # Regime-Confidence interactions
        if 'hmm_regime' in data.columns and 'confidence_score' in data.columns:
            data['regime_confidence_interaction'] = (
                data['hmm_regime'].fillna(0) * data['confidence_score']
            )
        
        # Volatility-Confidence interactions
        if 'price_volatility' in data.columns and 'confidence_score' in data.columns:
            data['volatility_confidence_interaction'] = (
                data['price_volatility'].fillna(0) * data['confidence_score']
            )
        
        return data
    
    def _apply_timeseries_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply time-series profit features."""
        
        # Profit magnitude time-series features
        if 'profit_magnitude_score' in data.columns:
            magnitude = data['profit_magnitude_score']
            
            for window in self.config.timeseries_windows:
                # Rolling statistics
                data[f'profit_magnitude_ma_{window}'] = magnitude.rolling(window=window).mean()
                data[f'profit_magnitude_std_{window}'] = magnitude.rolling(window=window).std()
                data[f'profit_magnitude_max_{window}'] = magnitude.rolling(window=window).max()
                data[f'profit_magnitude_min_{window}'] = magnitude.rolling(window=window).min()
                
                # Momentum features
                data[f'profit_magnitude_momentum_{window}'] = magnitude.diff(window)
                data[f'profit_magnitude_acceleration_{window}'] = magnitude.diff(window).diff()
        
        # Confidence time-series features
        if 'confidence_score' in data.columns:
            confidence = data['confidence_score']
            
            for window in self.config.timeseries_windows:
                data[f'confidence_ma_{window}'] = confidence.rolling(window=window).mean()
                data[f'confidence_std_{window}'] = confidence.rolling(window=window).std()
                data[f'confidence_trend_{window}'] = confidence.rolling(window=window).mean().diff()
        
        # Category time-series features
        if 'profit_category_ordinal' in data.columns:
            category_ordinal = data['profit_category_ordinal']
            
            for window in self.config.timeseries_windows:
                data[f'category_ma_{window}'] = category_ordinal.rolling(window=window).mean()
                data[f'category_std_{window}'] = category_ordinal.rolling(window=window).std()
        
        return data
    
    def _apply_risk_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply risk-adjusted profit features."""
        
        # Risk-adjusted magnitude
        if 'profit_magnitude_score' in data.columns and 'price_volatility' in data.columns:
            magnitude = data['profit_magnitude_score']
            volatility = data['price_volatility'].fillna(magnitude.std())
            
            data['profit_magnitude_sharpe'] = np.where(
                volatility > 0, magnitude / volatility, 0
            )
        
        # Downside risk features
        if 'potential_profit_pct' in data.columns:
            profit_pct = data['potential_profit_pct']
            downside_returns = np.where(profit_pct < 0, profit_pct, 0)
            
            data['downside_risk'] = pd.Series(downside_returns).rolling(window=20).std()
            data['downside_deviation'] = np.sqrt(np.mean(downside_returns ** 2))
        
        # Maximum drawdown features
        if 'profit_magnitude_score' in data.columns:
            magnitude = data['profit_magnitude_score']
            cumulative = magnitude.cumsum()
            running_max = cumulative.expanding().max()
            drawdown = cumulative - running_max
            
            data['max_drawdown'] = drawdown.min()
            data['current_drawdown'] = drawdown.iloc[-1] if len(drawdown) > 0 else 0
        
        # Value at Risk (VaR) features
        if 'potential_profit_pct' in data.columns:
            profit_pct = data['potential_profit_pct']
            
            for percentile in [0.05, 0.1, 0.25]:
                data[f'var_{int(percentile*100)}'] = profit_pct.rolling(window=50).quantile(percentile)
        
        return data
    
    def _apply_clustering_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply clustering-based features."""
        
        if not SKLEARN_AVAILABLE:
            return data
        
        # Prepare clustering features
        clustering_data = data[self.config.clustering_features].fillna(0)
        
        if len(clustering_data) < self.config.n_clusters:
            return data
        
        # Fit clustering model
        self.clustering_model = KMeans(n_clusters=self.config.n_clusters, random_state=42)
        cluster_labels = self.clustering_model.fit_predict(clustering_data)
        
        # Add cluster features
        data['profit_cluster'] = cluster_labels
        
        # Cluster one-hot encoding
        cluster_dummies = pd.get_dummies(cluster_labels, prefix='cluster')
        data = pd.concat([data, cluster_dummies], axis=1)
        
        # Distance to cluster centers
        distances = self.clustering_model.transform(clustering_data)
        for i in range(self.config.n_clusters):
            data[f'distance_to_cluster_{i}'] = distances[:, i]
        
        return data
    
    def _apply_pca_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply PCA-based features."""
        
        if not SKLEARN_AVAILABLE:
            return data
        
        # Prepare PCA features
        pca_data = data[self.config.pca_features].fillna(0)
        
        if len(pca_data) < self.config.n_pca_components:
            return data
        
        # Fit PCA model
        self.pca_model = PCA(n_components=self.config.n_pca_components)
        pca_features = self.pca_model.fit_transform(pca_data)
        
        # Add PCA features
        for i in range(self.config.n_pca_components):
            data[f'profit_pca_{i}'] = pca_features[:, i]
        
        # Explained variance features
        data['pca_explained_variance_ratio'] = self.pca_model.explained_variance_ratio_[0]
        
        return data
    
    def _apply_embedding_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply neural network embedding features."""
        
        if not TORCH_AVAILABLE:
            return data
        
        # This is a simplified version - in practice, you'd train a proper embedding model
        # For now, we'll create some basic neural-inspired features
        
        if 'profit_category_ordinal' in data.columns and 'confidence_score' in data.columns:
            # Simple embedding-like features
            category_ordinal = data['profit_category_ordinal'].fillna(4)  # Default to break-even
            confidence = data['confidence_score'].fillna(0.5)
            
            # Create embedding-like features using simple transformations
            for i in range(self.config.embedding_dim):
                data[f'profit_embedding_{i}'] = (
                    np.sin(category_ordinal * (i + 1) * np.pi / 8) * confidence +
                    np.cos(confidence * (i + 1) * np.pi) * category_ordinal / 8
                )
        
        return data
    
    def _apply_feature_selection(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply feature selection to reduce dimensionality."""
        
        if not SKLEARN_AVAILABLE:
            return data
        
        # Identify profit-related features
        profit_features = [col for col in data.columns if any(
            keyword in col.lower() for keyword in [
                'profit', 'confidence', 'category', 'magnitude', 'cluster', 'pca', 'embedding'
            ]
        )]
        
        if len(profit_features) < 2:
            return data
        
        # Prepare data for feature selection
        X = data[profit_features].fillna(0)
        
        # Use potential_profit_pct as target if available
        if 'potential_profit_pct' in data.columns:
            y = data['potential_profit_pct'].fillna(0)
        elif 'profit_magnitude_score' in data.columns:
            y = data['profit_magnitude_score'].fillna(0)
        else:
            return data
        
        # Apply feature selection
        if self.config.selection_method == "mutual_info":
            scores = mutual_info_regression(X, y)
        elif self.config.selection_method == "f_score":
            scores, _ = f_regression(X, y)
        else:  # correlation
            scores = np.abs(X.corrwith(pd.Series(y, index=X.index))).fillna(0).values
        
        # Select features
        feature_scores = pd.Series(scores, index=profit_features)
        selected_features = feature_scores[feature_scores > self.config.selection_threshold].index.tolist()
        
        # Limit number of features
        if self.config.max_features and len(selected_features) > self.config.max_features:
            selected_features = feature_scores.nlargest(self.config.max_features).index.tolist()
        
        self.selected_features = selected_features
        self.feature_importance_scores = feature_scores.to_dict()
        
        # Keep only selected features (plus original columns)
        original_columns = [col for col in data.columns if col not in profit_features]
        selected_data = data[original_columns + selected_features]
        
        return selected_data
    
    def _apply_feature_scaling(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply feature scaling to numerical features."""
        
        if self.scaler is None:
            return data
        
        # Identify numerical features to scale
        numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude original columns and target variables
        exclude_columns = ['close', 'open', 'high', 'low', 'volume', 'potential_profit_pct']
        numerical_features = [col for col in numerical_features if col not in exclude_columns]
        
        if len(numerical_features) == 0:
            return data
        
        # Apply scaling
        data[numerical_features] = self.scaler.fit_transform(data[numerical_features])
        
        return data
    
    def _calculate_category_streaks(self, categories: pd.Series) -> pd.Series:
        """Calculate streaks of consecutive categories."""
        streaks = []
        current_streak = 1
        
        for i in range(len(categories)):
            if i == 0:
                streaks.append(1)
            elif categories.iloc[i] == categories.iloc[i-1]:
                current_streak += 1
                streaks.append(current_streak)
            else:
                current_streak = 1
                streaks.append(1)
        
        return pd.Series(streaks, index=categories.index)
    
    def _calculate_regime_streaks(self, regimes: pd.Series) -> pd.Series:
        """Calculate streaks of consecutive regimes."""
        return self._calculate_category_streaks(regimes)
    
    def get_feature_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get summary of enhanced profit features."""
        
        # Categorize features
        feature_categories = {
            "category_features": [],
            "magnitude_features": [],
            "confidence_features": [],
            "regime_features": [],
            "volatility_features": [],
            "interaction_features": [],
            "timeseries_features": [],
            "risk_features": [],
            "clustering_features": [],
            "pca_features": [],
            "embedding_features": []
        }
        
        for col in data.columns:
            if 'profit_cat' in col or 'conf_cat' in col:
                feature_categories["category_features"].append(col)
            elif 'magnitude' in col:
                feature_categories["magnitude_features"].append(col)
            elif 'confidence' in col or 'uncertainty' in col:
                feature_categories["confidence_features"].append(col)
            elif 'regime' in col:
                feature_categories["regime_features"].append(col)
            elif 'volatility' in col:
                feature_categories["volatility_features"].append(col)
            elif 'interaction' in col or 'ratio' in col:
                feature_categories["interaction_features"].append(col)
            elif any(f'_{w}' in col for w in self.config.timeseries_windows):
                feature_categories["timeseries_features"].append(col)
            elif 'risk' in col or 'var_' in col or 'drawdown' in col:
                feature_categories["risk_features"].append(col)
            elif 'cluster' in col:
                feature_categories["clustering_features"].append(col)
            elif 'pca' in col:
                feature_categories["pca_features"].append(col)
            elif 'embedding' in col:
                feature_categories["embedding_features"].append(col)
        
        return {
            "total_features": len(data.columns),
            "feature_categories": feature_categories,
            "selected_features": self.selected_features,
            "feature_importance_scores": self.feature_importance_scores
        }

# Convenience functions
def create_enhanced_profit_feature_engineering(
    enable_category_features: bool = True,
    enable_magnitude_features: bool = True,
    enable_confidence_features: bool = True,
    enable_regime_features: bool = True,
    enable_volatility_features: bool = True,
    enable_interaction_features: bool = True,
    enable_timeseries_features: bool = True,
    enable_risk_features: bool = True,
    enable_clustering_features: bool = True,
    enable_pca_features: bool = True,
    enable_embedding_features: bool = True,
    enable_feature_selection: bool = True,
    enable_feature_scaling: bool = True,
    scaling_method: str = "robust"
) -> EnhancedProfitFeatureEngineering:
    """Create enhanced profit feature engineering with specified parameters."""
    config = EnhancedProfitFeatureConfig(
        enable_category_features=enable_category_features,
        enable_magnitude_features=enable_magnitude_features,
        enable_confidence_features=enable_confidence_features,
        enable_regime_features=enable_regime_features,
        enable_volatility_features=enable_volatility_features,
        enable_interaction_features=enable_interaction_features,
        enable_timeseries_features=enable_timeseries_features,
        enable_risk_features=enable_risk_features,
        enable_clustering_features=enable_clustering_features,
        enable_pca_features=enable_pca_features,
        enable_embedding_features=enable_embedding_features,
        enable_feature_selection=enable_feature_selection,
        enable_feature_scaling=enable_feature_scaling,
        scaling_method=scaling_method
    )
    
    return EnhancedProfitFeatureEngineering(config)

def apply_enhanced_profit_feature_engineering(
    data: pd.DataFrame,
    enable_category_features: bool = True,
    enable_magnitude_features: bool = True,
    enable_confidence_features: bool = True,
    enable_regime_features: bool = True,
    enable_volatility_features: bool = True,
    enable_interaction_features: bool = True,
    enable_timeseries_features: bool = True,
    enable_risk_features: bool = True,
    enable_clustering_features: bool = True,
    enable_pca_features: bool = True,
    enable_embedding_features: bool = True,
    enable_feature_selection: bool = True,
    enable_feature_scaling: bool = True,
    scaling_method: str = "robust"
) -> pd.DataFrame:
    """Apply enhanced profit feature engineering to data."""
    feature_eng = create_enhanced_profit_feature_engineering(
        enable_category_features=enable_category_features,
        enable_magnitude_features=enable_magnitude_features,
        enable_confidence_features=enable_confidence_features,
        enable_regime_features=enable_regime_features,
        enable_volatility_features=enable_volatility_features,
        enable_interaction_features=enable_interaction_features,
        enable_timeseries_features=enable_timeseries_features,
        enable_risk_features=enable_risk_features,
        enable_clustering_features=enable_clustering_features,
        enable_pca_features=enable_pca_features,
        enable_embedding_features=enable_embedding_features,
        enable_feature_selection=enable_feature_selection,
        enable_feature_scaling=enable_feature_scaling,
        scaling_method=scaling_method
    )
    
    return feature_eng.apply_all_features(data)

if __name__ == '__main__':
    # Test the enhanced profit feature engineering
    tprint('🧪 Testing Enhanced Profit Feature Engineering')
    
    # Create test data with enhanced profit labels
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    data = pd.DataFrame({
        'open': np.random.uniform(100, 110, 1000),
        'high': np.random.uniform(105, 115, 1000),
        'low': np.random.uniform(95, 105, 1000),
        'close': np.random.uniform(100, 110, 1000),
        'volume': np.random.uniform(1000, 10000, 1000),
        'hmm_regime': np.random.choice([0, 1, 2, 3], 1000),
        'profit_category': np.random.choice([
            'extreme_loss', 'large_loss', 'medium_loss', 'small_loss', 'break_even',
            'low_profit', 'medium_profit', 'high_profit', 'extreme_profit'
        ], 1000),
        'profit_magnitude_score': np.random.uniform(0, 10, 1000),
        'confidence_score': np.random.uniform(0, 1, 1000),
        'potential_profit_pct': np.random.uniform(-0.05, 0.05, 1000)
    }, index=dates)
    
    # Test enhanced feature engineering
    tprint('\n📊 Testing enhanced profit feature engineering...')
    result = apply_enhanced_profit_feature_engineering(data)
    
    tprint(f'✅ Enhanced profit feature engineering completed')
    tprint(f'   Original features: {len(data.columns)}')
    tprint(f'   Enhanced features: {len(result.columns)}')
    tprint(f'   Features added: {len(result.columns) - len(data.columns)}')
    
    # Show feature summary
    feature_eng = EnhancedProfitFeatureEngineering()
    summary = feature_eng.get_feature_summary(result)
    tprint(f'\n📋 Feature Summary:')
    for category, features in summary['feature_categories'].items():
        if features:
            tprint(f'   {category}: {len(features)} features')
    
    tprint('✅ Enhanced Profit Feature Engineering test completed!')