#!/usr/bin/env python3
"""
Feature Selection for HMM Regime Discovery

This module implements systematic feature selection for HMM clustering:
- Mutual Information-Based Ranking
- Recursive Feature Elimination
- Feature Selection Pipeline using existing tools
- Enhanced feature engineering for larger feature sets

Author: AI Assistant
Date: 2024-01-XX
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
import time
import logging
from pathlib import Path
import json

# Sklearn imports
try:
    from sklearn.feature_selection import (
        mutual_info_classif, mutual_info_regression,
        RFE, SelectKBest, f_classif, f_regression,
        SelectFromModel, VarianceThreshold
    )
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LassoCV, ElasticNetCV
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Import existing feature selection tools
try:
    from src.utils.feature_selection.step08_unified_complete import (
        FeatureSelectionFramework, FinancialMetrics, RegimeDataSplitter
    )
    EXISTING_FEATURE_SELECTION_AVAILABLE = True
except ImportError:
    EXISTING_FEATURE_SELECTION_AVAILABLE = False

from src.utils.logger import system_logger

@dataclass
class FeatureSelectionResult:
    """Result of feature selection"""
    selected_features: List[str]
    feature_scores: pd.DataFrame
    selection_method: str
    selection_time: float
    n_features_before: int
    n_features_after: int

class EnhancedFeatureEngineer:
    """Enhanced feature engineering for larger feature sets"""
    
    def __init__(self, logger=None):
        self.logger = logger or system_logger.getChild('EnhancedFeatureEngineer')
    
    def create_comprehensive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a comprehensive set of features for regime detection
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with comprehensive features
        """
        self.logger.info("🔧 Creating comprehensive feature set...")
        features = pd.DataFrame()
        features['timestamp'] = df['timestamp'] if 'timestamp' in df.columns else df.index
        
        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Price-based features
        self._add_price_features(features, df)
        
        # Volume-based features
        self._add_volume_features(features, df)
        
        # Volatility features
        self._add_volatility_features(features, df)
        
        # Technical indicators
        self._add_technical_indicators(features, df)
        
        # Momentum features
        self._add_momentum_features(features, df)
        
        # Support/Resistance features
        self._add_sr_features(features, df)
        
        # Statistical features
        self._add_statistical_features(features, df)
        
        # Cross-asset features (if multiple symbols)
        self._add_cross_asset_features(features, df)
        
        # Time-based features
        self._add_time_features(features, df)
        
        # Feature interactions
        self._add_feature_interactions(features)
        
        # Clean features
        features = self._clean_features(features)
        
        self.logger.info(f"✅ Created {len(features.columns)} comprehensive features")
        return features
    
    def _add_price_features(self, features: pd.DataFrame, df: pd.DataFrame) -> None:
        """Add price-based features"""
        # Basic price features
        features['price_change'] = df['close'].pct_change()
        features['price_range'] = (df['high'] - df['low']) / df['close']
        features['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Price ratios
        features['high_close_ratio'] = df['high'] / df['close']
        features['low_close_ratio'] = df['low'] / df['close']
        features['open_close_ratio'] = df['open'] / df['close']
        
        # Price gaps
        features['gap_up'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        features['gap_down'] = (df['close'].shift(1) - df['open']) / df['close'].shift(1)
        
        # Price patterns
        features['doji'] = (abs(df['open'] - df['close']) / (df['high'] - df['low'])) < 0.1
        features['hammer'] = ((df['close'] - df['low']) > 2 * (df['open'] - df['close'])) & \
                            ((df['high'] - df['close']) < 0.1 * (df['close'] - df['low']))
        
        # Multiple timeframe price features
        for window in [5, 10, 20, 50]:
            features[f'price_ma_{window}'] = df['close'].rolling(window).mean()
            features[f'price_ema_{window}'] = df['close'].ewm(span=window).mean()
            features[f'price_std_{window}'] = df['close'].rolling(window).std()
            features[f'price_min_{window}'] = df['close'].rolling(window).min()
            features[f'price_max_{window}'] = df['close'].rolling(window).max()
            
            # Price vs moving averages
            features[f'price_vs_ma_{window}'] = (df['close'] - features[f'price_ma_{window}']) / features[f'price_ma_{window}']
            features[f'price_vs_ema_{window}'] = (df['close'] - features[f'price_ema_{window}']) / features[f'price_ema_{window}']
    
    def _add_volume_features(self, features: pd.DataFrame, df: pd.DataFrame) -> None:
        """Add volume-based features"""
        # Basic volume features
        features['volume_change'] = df['volume'].pct_change()
        features['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Volume-price relationship
        features['volume_price_trend'] = (df['close'] - df['close'].shift(1)) * df['volume']
        features['volume_price_correlation'] = df['close'].rolling(20).corr(df['volume'])
        
        # Volume patterns
        features['volume_spike'] = df['volume'] > df['volume'].rolling(20).mean() * 2
        features['volume_dry_up'] = df['volume'] < df['volume'].rolling(20).mean() * 0.5
        
        # Multiple timeframe volume features
        for window in [5, 10, 20, 50]:
            features[f'volume_ma_{window}'] = df['volume'].rolling(window).mean()
            features[f'volume_std_{window}'] = df['volume'].rolling(window).std()
            features[f'volume_ratio_{window}'] = df['volume'] / features[f'volume_ma_{window}']
    
    def _add_volatility_features(self, features: pd.DataFrame, df: pd.DataFrame) -> None:
        """Add volatility features"""
        # Rolling volatility
        for window in [5, 10, 20, 50]:
            features[f'volatility_{window}'] = df['close'].pct_change().rolling(window).std()
            features[f'volatility_ewma_{window}'] = df['close'].pct_change().ewm(span=window).std()
        
        # Volatility ratios
        features['volatility_ratio_5_20'] = features['volatility_5'] / features['volatility_20']
        features['volatility_ratio_10_50'] = features['volatility_10'] / features['volatility_50']
        
        # Volatility momentum
        features['volatility_momentum'] = features['volatility_20'] - features['volatility_20'].shift(5)
        features['volatility_acceleration'] = features['volatility_momentum'].diff()
        
        # GARCH-like features
        features['volatility_clustering'] = (df['close'].pct_change() ** 2).rolling(20).mean()
        features['volatility_persistence'] = features['volatility_clustering'].rolling(10).corr(
            features['volatility_clustering'].shift(1)
        )
    
    def _add_technical_indicators(self, features: pd.DataFrame, df: pd.DataFrame) -> None:
        """Add technical indicators"""
        # RSI
        for window in [14, 21, 30]:
            features[f'rsi_{window}'] = self._calculate_rsi(df['close'], window)
        
        # MACD
        features['macd'] = self._calculate_macd(df['close'])
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        for window in [20, 50]:
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df['close'], window)
            features[f'bb_upper_{window}'] = bb_upper
            features[f'bb_middle_{window}'] = bb_middle
            features[f'bb_lower_{window}'] = bb_lower
            features[f'bb_width_{window}'] = (bb_upper - bb_lower) / bb_middle
            features[f'bb_position_{window}'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # ATR
        features['atr_14'] = self._calculate_atr(df)
        features['atr_ratio'] = features['atr_14'] / df['close']
        
        # ADX
        features['adx_14'] = self._calculate_adx(df)
    
    def _add_momentum_features(self, features: pd.DataFrame, df: pd.DataFrame) -> None:
        """Add momentum features"""
        # Price momentum
        for window in [1, 2, 3, 5, 10, 20, 50]:
            features[f'momentum_{window}'] = df['close'].pct_change(window)
            features[f'momentum_ma_{window}'] = features[f'momentum_{window}'].rolling(10).mean()
        
        # Volume momentum
        for window in [1, 2, 3, 5, 10, 20]:
            features[f'volume_momentum_{window}'] = df['volume'].pct_change(window)
        
        # Momentum ratios
        features['momentum_ratio_5_20'] = features['momentum_5'] / features['momentum_20']
        features['momentum_ratio_10_50'] = features['momentum_10'] / features['momentum_50']
    
    def _add_sr_features(self, features: pd.DataFrame, df: pd.DataFrame) -> None:
        """Add support/resistance features"""
        # Pivot points
        features['pivot_point'] = (df['high'] + df['low'] + df['close']) / 3
        features['support_1'] = 2 * features['pivot_point'] - df['high']
        features['resistance_1'] = 2 * features['pivot_point'] - df['low']
        features['support_2'] = features['pivot_point'] - (df['high'] - df['low'])
        features['resistance_2'] = features['pivot_point'] + (df['high'] - df['low'])
        
        # Distance to S/R levels
        features['distance_to_support'] = (df['close'] - features['support_1']) / df['close']
        features['distance_to_resistance'] = (features['resistance_1'] - df['close']) / df['close']
        
        # S/R strength
        features['sr_strength'] = self._calculate_sr_strength(df)
        
        # Swing highs and lows
        for window in [10, 20, 50]:
            features[f'swing_high_{window}'] = df['high'].rolling(window, center=True).max()
            features[f'swing_low_{window}'] = df['low'].rolling(window, center=True).min()
            features[f'distance_to_swing_high_{window}'] = (features[f'swing_high_{window}'] - df['close']) / df['close']
            features[f'distance_to_swing_low_{window}'] = (df['close'] - features[f'swing_low_{window}']) / df['close']
    
    def _add_statistical_features(self, features: pd.DataFrame, df: pd.DataFrame) -> None:
        """Add statistical features"""
        # Skewness and kurtosis
        for window in [20, 50]:
            features[f'skewness_{window}'] = df['close'].pct_change().rolling(window).skew()
            features[f'kurtosis_{window}'] = df['close'].pct_change().rolling(window).kurt()
        
        # Quantiles
        for window in [20, 50]:
            for q in [0.25, 0.5, 0.75, 0.9, 0.95]:
                features[f'quantile_{q}_{window}'] = df['close'].rolling(window).quantile(q)
                features[f'price_vs_quantile_{q}_{window}'] = (df['close'] - features[f'quantile_{q}_{window}']) / df['close']
        
        # Autocorrelation
        for window in [20, 50]:
            features[f'autocorr_{window}'] = df['close'].pct_change().rolling(window).apply(
                lambda x: x.autocorr(lag=1) if len(x) > 1 else 0
            )
    
    def _add_cross_asset_features(self, features: pd.DataFrame, df: pd.DataFrame) -> None:
        """Add cross-asset features (if multiple symbols available)"""
        # This would be implemented if multiple symbols are available
        # For now, we'll add some placeholder features
        pass
    
    def _add_time_features(self, features: pd.DataFrame, df: pd.DataFrame) -> None:
        """Add time-based features"""
        if 'timestamp' in features.columns:
            timestamp = pd.to_datetime(features['timestamp'])
            features['hour'] = timestamp.dt.hour
            features['day_of_week'] = timestamp.dt.dayofweek
            features['day_of_month'] = timestamp.dt.day
            features['month'] = timestamp.dt.month
            
            # Cyclical encoding
            features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
            features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
            features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
    
    def _add_feature_interactions(self, features: pd.DataFrame) -> None:
        """Add feature interactions"""
        # Price-volume interactions
        if 'price_change' in features.columns and 'volume_change' in features.columns:
            features['price_volume_interaction'] = features['price_change'] * features['volume_change']
        
        # Volatility-momentum interactions
        if 'volatility_20' in features.columns and 'momentum_10' in features.columns:
            features['volatility_momentum_interaction'] = features['volatility_20'] * features['momentum_10']
        
        # RSI-momentum interactions
        if 'rsi_14' in features.columns and 'momentum_5' in features.columns:
            features['rsi_momentum_interaction'] = features['rsi_14'] * features['momentum_5']
    
    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate features"""
        self.logger.info("🧹 Cleaning features...")
        
        # Remove timestamp column for HMM training
        if 'timestamp' in features.columns:
            features = features.drop('timestamp', axis=1)
        
        # Handle infinite values
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill technical indicators
        technical_cols = [col for col in features.columns if any(indicator in col for indicator in 
                       ['rsi', 'macd', 'bb_', 'atr', 'adx', 'sr_strength'])]
        for col in technical_cols:
            if col in features.columns:
                features[col] = features[col].ffill()
        
        # Fill remaining NaN values
        features = features.fillna(0)
        
        # Remove constant features
        constant_features = features.columns[features.nunique() <= 1]
        if len(constant_features) > 0:
            self.logger.info(f"   Removing {len(constant_features)} constant features")
            features = features.drop(constant_features, axis=1)
        
        self.logger.info(f"✅ Feature cleaning completed: {len(features.columns)} features")
        return features
    
    # Technical indicator calculation methods
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - 100 / (1 + rs)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        return ema_fast - ema_slow
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = sma + std * num_std
        lower = sma - std * num_std
        return upper, sma, lower
    
    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate ATR"""
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window).mean()
    
    def _calculate_adx(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate ADX"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        dm_plus = high - high.shift(1)
        dm_minus = low.shift(1) - low
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
        
        tr_smooth = tr.rolling(window).mean()
        dm_plus_smooth = dm_plus.rolling(window).mean()
        dm_minus_smooth = dm_minus.rolling(window).mean()
        
        di_plus = 100 * (dm_plus_smooth / tr_smooth)
        di_minus = 100 * (dm_minus_smooth / tr_smooth)
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        
        return dx.rolling(window).mean()
    
    def _calculate_sr_strength(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate support/resistance strength"""
        high_swing = df['high'].rolling(window, center=True).max()
        low_swing = df['low'].rolling(window, center=True).min()
        current_price = df['close']
        
        high_strength = (high_swing - current_price) / high_swing
        low_strength = (current_price - low_swing) / low_swing
        
        return (high_strength + low_strength) / 2

class FeatureSelector:
    """Systematic feature selection for HMM clustering"""
    
    def __init__(self, logger=None):
        self.logger = logger or system_logger.getChild('FeatureSelector')
        self.selection_history = []
    
    def mutual_information_ranking(self, features: pd.DataFrame, regime_labels: np.ndarray) -> FeatureSelectionResult:
        """
        Rank features by mutual information with regime labels
        
        Args:
            features: Input features
            regime_labels: Regime labels for each sample
            
        Returns:
            FeatureSelectionResult with ranked features
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn not available")
        
        start_time = time.time()
        self.logger.info("🔍 Ranking features by mutual information...")
        
        # Calculate mutual information scores
        mi_scores = mutual_info_classif(features, regime_labels, random_state=42)
        
        # Create feature importance dataframe
        feature_scores = pd.DataFrame({
            'feature': features.columns,
            'mutual_info_score': mi_scores,
            'rank': range(1, len(features.columns) + 1)
        }).sort_values('mutual_info_score', ascending=False)
        
        # Select top features (keep all for now, can be filtered later)
        selected_features = feature_scores['feature'].tolist()
        
        selection_time = time.time() - start_time
        
        result = FeatureSelectionResult(
            selected_features=selected_features,
            feature_scores=feature_scores,
            selection_method='mutual_information_ranking',
            selection_time=selection_time,
            n_features_before=len(features.columns),
            n_features_after=len(selected_features)
        )
        
        self.selection_history.append(result)
        self.logger.info(f"✅ Mutual information ranking completed: {len(selected_features)} features")
        
        return result
    
    def recursive_feature_elimination(self, features: pd.DataFrame, regime_labels: np.ndarray, 
                                    n_features: int = 20) -> FeatureSelectionResult:
        """
        Use recursive feature elimination to find most important features
        
        Args:
            features: Input features
            regime_labels: Regime labels for each sample
            n_features: Number of features to select
            
        Returns:
            FeatureSelectionResult with selected features
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn not available")
        
        start_time = time.time()
        self.logger.info(f"🔍 Recursive feature elimination for {n_features} features...")
        
        # Use Random Forest as base estimator
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Recursive feature elimination
        rfe = RFE(estimator=rf, n_features_to_select=n_features)
        rfe.fit(features, regime_labels)
        
        # Get selected features
        selected_features = features.columns[rfe.support_].tolist()
        feature_ranking = pd.DataFrame({
            'feature': features.columns,
            'selected': rfe.support_,
            'ranking': rfe.ranking_
        }).sort_values('ranking')
        
        selection_time = time.time() - start_time
        
        result = FeatureSelectionResult(
            selected_features=selected_features,
            feature_scores=feature_ranking,
            selection_method='recursive_feature_elimination',
            selection_time=selection_time,
            n_features_before=len(features.columns),
            n_features_after=len(selected_features)
        )
        
        self.selection_history.append(result)
        self.logger.info(f"✅ RFE completed: {len(selected_features)} features selected")
        
        return result
    
    def variance_threshold_selection(self, features: pd.DataFrame, threshold: float = 0.01) -> FeatureSelectionResult:
        """
        Remove low-variance features
        
        Args:
            features: Input features
            threshold: Variance threshold
            
        Returns:
            FeatureSelectionResult with selected features
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn not available")
        
        start_time = time.time()
        self.logger.info(f"🔍 Variance threshold selection (threshold={threshold})...")
        
        # Apply variance threshold
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(features)
        
        # Get selected features
        selected_features = features.columns[selector.get_support()].tolist()
        
        # Create feature scores dataframe
        feature_scores = pd.DataFrame({
            'feature': features.columns,
            'variance': selector.variances_,
            'selected': selector.get_support()
        }).sort_values('variance', ascending=False)
        
        selection_time = time.time() - start_time
        
        result = FeatureSelectionResult(
            selected_features=selected_features,
            feature_scores=feature_scores,
            selection_method='variance_threshold_selection',
            selection_time=selection_time,
            n_features_before=len(features.columns),
            n_features_after=len(selected_features)
        )
        
        self.selection_history.append(result)
        self.logger.info(f"✅ Variance threshold selection completed: {len(selected_features)} features selected")
        
        return result
    
    def model_based_selection(self, features: pd.DataFrame, regime_labels: np.ndarray, 
                            n_features: int = 20) -> FeatureSelectionResult:
        """
        Use model-based feature selection
        
        Args:
            features: Input features
            regime_labels: Regime labels for each sample
            n_features: Number of features to select
            
        Returns:
            FeatureSelectionResult with selected features
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn not available")
        
        start_time = time.time()
        self.logger.info(f"🔍 Model-based feature selection for {n_features} features...")
        
        # Use Lasso for feature selection
        lasso = LassoCV(cv=5, random_state=42)
        lasso.fit(features, regime_labels)
        
        # Select features with non-zero coefficients
        selector = SelectFromModel(lasso, max_features=n_features)
        selector.fit(features, regime_labels)
        
        # Get selected features
        selected_features = features.columns[selector.get_support()].tolist()
        
        # Create feature scores dataframe
        feature_scores = pd.DataFrame({
            'feature': features.columns,
            'coefficient': lasso.coef_,
            'selected': selector.get_support()
        }).sort_values('coefficient', key=abs, ascending=False)
        
        selection_time = time.time() - start_time
        
        result = FeatureSelectionResult(
            selected_features=selected_features,
            feature_scores=feature_scores,
            selection_method='model_based_selection',
            selection_time=selection_time,
            n_features_before=len(features.columns),
            n_features_after=len(selected_features)
        )
        
        self.selection_history.append(result)
        self.logger.info(f"✅ Model-based selection completed: {len(selected_features)} features selected")
        
        return result
    
    def comprehensive_feature_selection(self, features: pd.DataFrame, regime_labels: np.ndarray,
                                      n_features: int = 20) -> FeatureSelectionResult:
        """
        Comprehensive feature selection combining multiple methods
        
        Args:
            features: Input features
            regime_labels: Regime labels for each sample
            n_features: Number of features to select
            
        Returns:
            FeatureSelectionResult with selected features
        """
        start_time = time.time()
        self.logger.info(f"🔍 Comprehensive feature selection for {n_features} features...")
        
        # Step 1: Remove low variance features
        var_result = self.variance_threshold_selection(features, threshold=0.01)
        features_filtered = features[var_result.selected_features]
        
        # Step 2: Mutual information ranking
        mi_result = self.mutual_information_ranking(features_filtered, regime_labels)
        
        # Step 3: Select top features based on mutual information
        top_features = mi_result.feature_scores.head(n_features)['feature'].tolist()
        
        # Step 4: Final validation with RFE
        rfe_result = self.recursive_feature_elimination(
            features[top_features], regime_labels, n_features=min(n_features, len(top_features))
        )
        
        selection_time = time.time() - start_time
        
        result = FeatureSelectionResult(
            selected_features=rfe_result.selected_features,
            feature_scores=mi_result.feature_scores,
            selection_method='comprehensive_feature_selection',
            selection_time=selection_time,
            n_features_before=len(features.columns),
            n_features_after=len(rfe_result.selected_features)
        )
        
        self.selection_history.append(result)
        self.logger.info(f"✅ Comprehensive selection completed: {len(rfe_result.selected_features)} features selected")
        
        return result
    
    def get_selection_summary(self) -> Dict[str, Any]:
        """Get summary of all feature selection runs"""
        if not self.selection_history:
            return {"message": "No feature selection runs recorded"}
        
        summary = {
            "total_runs": len(self.selection_history),
            "methods_used": list(set(r.selection_method for r in self.selection_history)),
            "total_selection_time": sum(r.selection_time for r in self.selection_history),
            "runs": []
        }
        
        for i, result in enumerate(self.selection_history):
            summary["runs"].append({
                "run_id": i,
                "method": result.selection_method,
                "n_features_before": result.n_features_before,
                "n_features_after": result.n_features_after,
                "selection_time": result.selection_time
            })
        
        return summary

# Example usage and testing
def test_feature_selection():
    """Test the feature selection functionality"""
    # Generate sample data
    np.random.seed(42)
    n_samples, n_features = 1000, 50
    features = pd.DataFrame(np.random.randn(n_samples, n_features), 
                          columns=[f'feature_{i}' for i in range(n_features)])
    
    # Generate regime labels
    regime_labels = np.random.randint(0, 3, n_samples)
    
    # Test feature selection
    selector = FeatureSelector()
    
    # Test mutual information ranking
    print("Testing mutual information ranking...")
    result1 = selector.mutual_information_ranking(features, regime_labels)
    print(f"Selected {len(result1.selected_features)} features")
    
    # Test recursive feature elimination
    print("\nTesting recursive feature elimination...")
    result2 = selector.recursive_feature_elimination(features, regime_labels, n_features=20)
    print(f"Selected {len(result2.selected_features)} features")
    
    # Test comprehensive selection
    print("\nTesting comprehensive selection...")
    result3 = selector.comprehensive_feature_selection(features, regime_labels, n_features=20)
    print(f"Selected {len(result3.selected_features)} features")
    
    # Print summary
    print("\nFeature Selection Summary:")
    summary = selector.get_selection_summary()
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    test_feature_selection()