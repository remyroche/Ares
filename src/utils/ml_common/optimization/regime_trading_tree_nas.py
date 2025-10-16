"""
import warnings
Regime Detection and Trading Optimized Pure Tree NAS

This module provides tree-based NAS specifically optimized for:
1. Regime detection and qualification
2. Trading applications using the most appropriate models

Key Features:
- Regime-specific tree models (bull/bear/sideways/volatile/trending)
- Trading-optimized tree architectures
- Financial feature engineering
- Regime transition detection
- Trading signal generation
- Risk management trees
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime
from src.utils.tprint import (tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_progress, tprint_performance, tprint_timer)

# Tree models optimized for financial data
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

# Advanced tree models for financial data
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    cb = None

logger = logging.getLogger(__name__)

@dataclass
class RegimeTradingTreeNASConfig:
    """Configuration for regime detection and trading optimized tree NAS."""

    # Regime detection specific models
    regime_models: List[str] = field(default_factory=lambda: [
        'regime_classifier', 'regime_transition_detector', 'regime_quality_assessor',
        'regime_persistence_predictor', 'regime_volatility_estimator'
    ])

    # Trading specific models
    trading_models: List[str] = field(default_factory=lambda: [
        'signal_generator', 'position_sizer', 'risk_manager', 'entry_exit_detector',
        'momentum_predictor', 'mean_reversion_detector', 'trend_follower'
    ])

    # Regime types to detect
    regime_types: List[str] = field(default_factory=lambda: [
        'bull', 'bear', 'sideways', 'volatile', 'trending', 'crisis', 'recovery'
    ])

    # Trading strategies
    trading_strategies: List[str] = field(default_factory=lambda: [
        'momentum', 'mean_reversion', 'trend_following', 'arbitrage', 'scalping',
        'swing_trading', 'position_trading', 'hedging'
    ])

    # Financial feature engineering
    feature_engineering: Dict[str, Any] = field(default_factory=lambda: {
        'technical_indicators': True,
        'price_features': True,
        'volume_features': True,
        'volatility_features': True,
        'momentum_features': True,
        'trend_features': True,
        'regime_features': True,
        'trading_features': True
    })

    # Regime detection parameters
    regime_detection: Dict[str, Any] = field(default_factory=lambda: {
        'min_regime_duration': 5,
        'max_regime_duration': 100,
        'regime_stability_threshold': 0.7,
        'transition_sensitivity': 0.5,
        'quality_thresholds': {
            'min_silhouette_score': 0.3,
            'min_persistence': 0.6,
            'min_separation': 0.5,
            'min_consistency': 0.7
        }
    })

    # Trading parameters
    trading_config: Dict[str, Any] = field(default_factory=lambda: {
        'signal_threshold': 0.6,
        'position_sizing_method': 'kelly',
        'risk_tolerance': 0.02,
        'max_position_size': 0.1,
        'stop_loss_pct': 0.05,
        'take_profit_pct': 0.15
    })

    # Tree model configurations
    tree_configs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'regime_classifier': {
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'criterion': 'gini'
        },
        'signal_generator': {
            'max_depth': 8,
            'min_samples_split': 3,
            'min_samples_leaf': 1,
            'criterion': 'entropy'
        },
        'risk_manager': {
            'max_depth': 6,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'criterion': 'gini'
        }
    })

    # Optimization settings
    n_trials: int = 100
    timeout_seconds: int = 3600
    cv_folds: int = 5

    # Performance settings
    n_jobs: int = -1
    memory_limit_gb: float = 8.0

class RegimeDetectionTree:
    """Tree model optimized for regime detection."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize regime detection tree."""
        self.config = config
        self.model = None
        self.regime_centers = {}
        self.regime_boundaries = {}
        self.is_trained = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train regime detection tree."""
        try:
            # Use Extra Trees for regime detection (better for clustering-like tasks)
            self.model = ExtraTreesClassifier(
                n_estimators=self.config.get('n_estimators', 100),
                max_depth=self.config.get('max_depth', 10),
                min_samples_split=self.config.get('min_samples_split', 5),
                min_samples_leaf=self.config.get('min_samples_leaf', 2),
                random_state=42
            )

            self.model.fit(X, y)

            # Calculate regime centers and boundaries
            unique_regimes = np.unique(y)
            for regime in unique_regimes:
                regime_mask = y == regime
                regime_data = X[regime_mask]

                self.regime_centers[regime] = np.mean(regime_data, axis=0)
                self.regime_boundaries[regime] = np.std(regime_data, axis=0)

            self.is_trained = True

        except Exception as e:
            logger.error(f"Regime detection tree training failed: {e}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict regimes."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict regime probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        return self.model.predict_proba(X)

    def get_regime_quality(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculate regime quality metrics."""
        try:
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

            # Calculate clustering quality metrics
            silhouette = silhouette_score(X, y)
            calinski_harabasz = calinski_harabasz_score(X, y)
            davies_bouldin = davies_bouldin_score(X, y)

            # Calculate regime persistence
            persistence = self._calculate_regime_persistence(y)

            # Calculate regime separation
            separation = self._calculate_regime_separation(X, y)

            return {
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski_harabasz,
                'davies_bouldin_score': davies_bouldin,
                'persistence': persistence,
                'separation': separation,
                'overall_quality': (silhouette + persistence + separation) / 3.0
            }

        except Exception as e:
            logger.warning(f"Regime quality calculation failed: {e}")
            return {'overall_quality': 0.0}

    def _calculate_regime_persistence(self, y: np.ndarray) -> float:
        """Calculate regime persistence."""
        try:
            # Calculate consecutive periods in same regime
            consecutive_periods = []
            current_period = 1

            for i in range(1, len(y)):
                if y[i] == y[i-1]:
                    current_period += 1
                else:
                    consecutive_periods.append(current_period)
                    current_period = 1

            consecutive_periods.append(current_period)

            # Persistence as ratio of longest consecutive period to total length
            max_consecutive = max(consecutive_periods)
            persistence = max_consecutive / len(y)

            return float(persistence)

        except Exception as e:
            logger.warning(f"Regime persistence calculation failed: {e}")
            return 0.0

    def _calculate_regime_separation(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate regime separation."""
        try:
            unique_regimes = np.unique(y)
            if len(unique_regimes) < 2:
                return 0.0

            # Calculate distances between regime centers
            regime_centers = []
            for regime in unique_regimes:
                regime_mask = y == regime
                regime_data = X[regime_mask]
                if len(regime_data) > 0:
                    regime_centers.append(np.mean(regime_data, axis=0))

            if len(regime_centers) < 2:
                return 0.0

            # Calculate minimum distance between regime centers
            min_distance = float('inf')
            for i in range(len(regime_centers)):
                for j in range(i + 1, len(regime_centers)):
                    distance = np.linalg.norm(regime_centers[i] - regime_centers[j])
                    min_distance = min(min_distance, distance)

            # Normalize by maximum possible distance
            max_possible_distance = np.sqrt(X.shape[1])
            separation = min(1.0, min_distance / max_possible_distance)

            return float(separation)

        except Exception as e:
            logger.warning(f"Regime separation calculation failed: {e}")
            return 0.0

class TradingSignalTree:
    """Tree model optimized for trading signal generation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize trading signal tree."""
        self.config = config
        self.model = None
        self.signal_threshold = config.get('signal_threshold', 0.6)
        self.is_trained = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train trading signal tree."""
        try:
            # Use Gradient Boosting for trading signals (good for sequential patterns)
            self.model = GradientBoostingClassifier(
                n_estimators=self.config.get('n_estimators', 100),
                max_depth=self.config.get('max_depth', 8),
                learning_rate=self.config.get('learning_rate', 0.1),
                min_samples_split=self.config.get('min_samples_split', 3),
                min_samples_leaf=self.config.get('min_samples_leaf', 1),
                random_state=42
            )

            self.model.fit(X, y)
            self.is_trained = True

        except Exception as e:
            logger.error(f"Trading signal tree training failed: {e}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate trading signals."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        # Get signal probabilities
        signal_proba = self.model.predict_proba(X)

        # Apply threshold to generate signals
        signals = np.zeros(len(X))
        for i, proba in enumerate(signal_proba):
            if len(proba) > 1:  # Binary classification
                if proba[1] > self.signal_threshold:
                    signals[i] = 1  # Buy signal
                elif proba[0] > self.signal_threshold:
                    signals[i] = -1  # Sell signal
                else:
                    signals[i] = 0  # Hold signal
            else:  # Regression
                if proba[0] > self.signal_threshold:
                    signals[i] = 1
                elif proba[0] < -self.signal_threshold:
                    signals[i] = -1
                else:
                    signals[i] = 0

        return signals

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get signal probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        return self.model.predict_proba(X)

    def get_signal_strength(self, X: np.ndarray) -> np.ndarray:
        """Get signal strength (confidence)."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        signal_proba = self.model.predict_proba(X)
        if len(signal_proba[0]) > 1:  # Binary classification
            return np.max(signal_proba, axis=1)
        else:  # Regression
            return np.abs(signal_proba.flatten())

class RiskManagementTree:
    """Tree model optimized for risk management."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize risk management tree."""
        self.config = config
        self.model = None
        self.risk_threshold = config.get('risk_threshold', 0.02)
        self.is_trained = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train risk management tree."""
        try:
            # Use Random Forest for risk management (robust to outliers)
            self.model = RandomForestClassifier(
                n_estimators=self.config.get('n_estimators', 100),
                max_depth=self.config.get('max_depth', 6),
                min_samples_split=self.config.get('min_samples_split', 10),
                min_samples_leaf=self.config.get('min_samples_leaf', 5),
                random_state=42
            )

            self.model.fit(X, y)
            self.is_trained = True

        except Exception as e:
            logger.error(f"Risk management tree training failed: {e}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict risk levels."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict risk probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        return self.model.predict_proba(X)

    def get_risk_score(self, X: np.ndarray) -> np.ndarray:
        """Get risk scores."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        risk_proba = self.model.predict_proba(X)
        # Return probability of high risk
        if len(risk_proba[0]) > 1:
            return risk_proba[:, 1]  # High risk probability
        else:
            return risk_proba.flatten()

class PositionSizingTree:
    """Tree model optimized for position sizing."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize position sizing tree."""
        self.config = config
        self.model = None
        self.max_position_size = config.get('max_position_size', 0.1)
        self.is_trained = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train position sizing tree."""
        try:
            # Use XGBoost for position sizing (good for regression)
            if XGBOOST_AVAILABLE:
                self.model = xgb.XGBRegressor(
                    n_estimators=self.config.get('n_estimators', 100),
                    max_depth=self.config.get('max_depth', 6),
                    learning_rate=self.config.get('learning_rate', 0.1),
                    random_state=42
                )
            else:
                # Fallback to Gradient Boosting
                self.model = GradientBoostingRegressor(
                    n_estimators=self.config.get('n_estimators', 100),
                    max_depth=self.config.get('max_depth', 6),
                    learning_rate=self.config.get('learning_rate', 0.1),
                    random_state=42
                )

            self.model.fit(X, y)
            self.is_trained = True

        except Exception as e:
            logger.error(f"Position sizing tree training failed: {e}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict position sizes."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        position_sizes = self.model.predict(X)

        # Apply constraints
        position_sizes = np.clip(position_sizes, 0, self.max_position_size)

        return position_sizes

    def get_position_recommendation(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get position recommendations."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        position_sizes = self.predict(X)

        # Generate recommendations
        recommendations = []
        for size in position_sizes:
            if size > 0.05:
                recommendations.append('BUY')
            elif size < -0.05:
                recommendations.append('SELL')
            else:
                recommendations.append('HOLD')

        return {
            'position_sizes': position_sizes,
            'recommendations': np.array(recommendations),
            'confidence': np.abs(position_sizes)
        }

class RegimeTradingTreeNAS:
    """Regime Detection and Trading Optimized Pure Tree NAS."""

    def __init__(self, config: RegimeTradingTreeNASConfig):
        """Initialize regime trading tree NAS."""
        tprint("🚀 [REGIME_TRADING_TREE_NAS] Initializing Regime Trading Tree NAS", color="cyan", bold=True)
        tprint(f"📊 [REGIME_TRADING_TREE_NAS] Regime types: {config.regime_types}", color="blue")
        tprint(f"📊 [REGIME_TRADING_TREE_NAS] Tree configs: {len(config.tree_configs)}", color="blue")
        self.config = config
        self.logger = logger.getChild('RegimeTradingTreeNAS')

        # Initialize specialized models
        tprint("🔍 [REGIME_TRADING_TREE_NAS] Initializing regime detector", color="yellow")
        self.regime_detector = RegimeDetectionTree(config.tree_configs['regime_classifier'])
        tprint("📈 [REGIME_TRADING_TREE_NAS] Initializing signal generator", color="yellow")
        self.signal_generator = TradingSignalTree(config.tree_configs['signal_generator'])
        tprint("⚠️ [REGIME_TRADING_TREE_NAS] Initializing risk manager", color="yellow")
        self.risk_manager = RiskManagementTree(config.tree_configs['risk_manager'])
        tprint("💰 [REGIME_TRADING_TREE_NAS] Initializing position sizer", color="yellow")
        self.position_sizer = PositionSizingTree(config.tree_configs.get('position_sizer', {}))

        # Results storage
        tprint("💾 [REGIME_TRADING_TREE_NAS] Setting up results storage", color="blue")
        self.regime_results = None
        self.trading_results = None
        self.combined_results = None

        tprint("✅ [REGIME_TRADING_TREE_NAS] Regime Trading Tree NAS initialized successfully", color="green")
        self.logger.info("✅ Regime Trading Tree NAS initialized")

    def detect_regimes(self, market_data: pd.DataFrame, timestamps: np.ndarray) -> Dict[str, Any]:
        """Detect market regimes using tree-based models."""
        tprint("🔍 [REGIME_TRADING_TREE_NAS] Detecting market regimes", color="yellow")
        tprint(f"📊 [REGIME_TRADING_TREE_NAS] Market data shape: {market_data.shape}", color="blue")
        self.logger.info("🔍 Detecting market regimes...")

        try:
            # Prepare features for regime detection
            tprint("🔧 [REGIME_TRADING_TREE_NAS] Preparing regime detection features", color="yellow")
            X, feature_names = self._prepare_regime_features(market_data)
            tprint(f"✅ [REGIME_TRADING_TREE_NAS] Prepared {X.shape[1]} features for regime detection", color="green")

            # Train regime detection model
            # For demonstration, we'll create synthetic regime labels
            # In practice, you'd use unsupervised clustering or labeled data
            n_samples = len(X)
            n_regimes = len(self.config.regime_types)
            regime_labels = np.random.randint(0, n_regimes, n_samples)
            tprint(f"📊 [REGIME_TRADING_TREE_NAS] Generated {n_regimes} regime labels for {n_samples} samples", color="cyan")

            # Train regime detector
            tprint("🧠 [REGIME_TRADING_TREE_NAS] Training regime detector", color="yellow")
            self.regime_detector.fit(X, regime_labels)
            tprint("✅ [REGIME_TRADING_TREE_NAS] Regime detector trained successfully", color="green")

            # Get regime predictions
            tprint("🔮 [REGIME_TRADING_TREE_NAS] Generating regime predictions", color="yellow")
            regime_predictions = self.regime_detector.predict(X)
            regime_probabilities = self.regime_detector.predict_proba(X)
            tprint(f"✅ [REGIME_TRADING_TREE_NAS] Generated predictions for {len(regime_predictions)} samples", color="green")

            # Calculate regime quality
            tprint("📊 [REGIME_TRADING_TREE_NAS] Calculating regime quality metrics", color="yellow")
            regime_quality = self.regime_detector.get_regime_quality(X, regime_predictions)
            tprint(f"✅ [REGIME_TRADING_TREE_NAS] Regime quality calculated: {regime_quality}", color="green")

            # Create regime results
            self.regime_results = {
                'regime_predictions': regime_predictions,
                'regime_probabilities': regime_probabilities,
                'regime_quality': regime_quality,
                'feature_names': feature_names,
                'regime_centers': self.regime_detector.regime_centers,
                'regime_boundaries': self.regime_detector.regime_boundaries
            }

            self.logger.info(f"✅ Detected {len(np.unique(regime_predictions))} regimes")
            self.logger.info(f"📊 Regime quality: {regime_quality['overall_quality']:.4f}")

            return self.regime_results

        except Exception as e:
            self.logger.error(f"Regime detection failed: {e}")
            raise

    def generate_trading_signals(self, market_data: pd.DataFrame, regime_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals using tree-based models."""
        self.logger.info("📈 Generating trading signals...")

        try:
            # Prepare features for trading
            X, feature_names = self._prepare_trading_features(market_data, regime_data)

            # Create synthetic trading signals for training
            # In practice, you'd use historical trading performance
            n_samples = len(X)
            signal_labels = np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.4, 0.3])  # Hold, Buy, Sell

            # Train signal generator
            self.signal_generator.fit(X, signal_labels)

            # Generate signals
            signals = self.signal_generator.predict(X)
            signal_strengths = self.signal_generator.get_signal_strength(X)

            # Train risk manager
            risk_labels = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])  # Low risk, High risk
            self.risk_manager.fit(X, risk_labels)

            # Get risk assessments
            risk_levels = self.risk_manager.predict(X)
            risk_scores = self.risk_manager.get_risk_score(X)

            # Train position sizer
            position_sizes = np.random.uniform(-0.1, 0.1, n_samples)  # Synthetic position sizes
            self.position_sizer.fit(X, position_sizes)

            # Get position recommendations
            position_recommendations = self.position_sizer.get_position_recommendation(X)

            # Create trading results
            self.trading_results = {
                'signals': signals,
                'signal_strengths': signal_strengths,
                'risk_levels': risk_levels,
                'risk_scores': risk_scores,
                'position_sizes': position_recommendations['position_sizes'],
                'position_recommendations': position_recommendations['recommendations'],
                'confidence': position_recommendations['confidence']
            }

            self.logger.info(f"✅ Generated {len(signals)} trading signals")
            self.logger.info(f"📊 Signal distribution: {np.bincount(signals + 1)}")

            return self.trading_results

        except Exception as e:
            self.logger.error(f"Trading signal generation failed: {e}")
            raise

    def _prepare_regime_features(self, market_data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Prepare features for regime detection."""
        try:
            features = []
            feature_names = []

            # Price-based features
            if 'close' in market_data.columns:
                # Returns
                returns = market_data['close'].pct_change().fillna(0)
                features.append(returns.values)
                feature_names.append('returns')

                # Log returns
                log_returns = np.log(market_data['close'] / market_data['close'].shift(1)).fillna(0)
                features.append(log_returns.values)
                feature_names.append('log_returns')

                # Price momentum
                momentum_5 = market_data['close'].pct_change(5).fillna(0)
                momentum_10 = market_data['close'].pct_change(10).fillna(0)
                momentum_20 = market_data['close'].pct_change(20).fillna(0)
                features.extend([momentum_5.values, momentum_10.values, momentum_20.values])
                feature_names.extend(['momentum_5', 'momentum_10', 'momentum_20'])

                # Moving averages
                ma_5 = market_data['close'].rolling(5).mean().fillna(market_data['close'])
                ma_10 = market_data['close'].rolling(10).mean().fillna(market_data['close'])
                ma_20 = market_data['close'].rolling(20).mean().fillna(market_data['close'])
                features.extend([ma_5.values, ma_10.values, ma_20.values])
                feature_names.extend(['ma_5', 'ma_10', 'ma_20'])

                # Price ratios
                price_ratios = (market_data['close'] / ma_20).fillna(1)
                features.append(price_ratios.values)
                feature_names.append('price_ratio_ma20')

            # Volatility features
            if 'high' in market_data.columns and 'low' in market_data.columns:
                # True range
                high_low = market_data['high'] - market_data['low']
                high_close = np.abs(market_data['high'] - market_data['close'].shift(1))
                low_close = np.abs(market_data['low'] - market_data['close'].shift(1))
                true_range = np.maximum(high_low, np.maximum(high_close, low_close))
                features.append(true_range.values)
                feature_names.append('true_range')

                # Volatility (rolling standard deviation)
                volatility_5 = returns.rolling(5).std().fillna(0)
                volatility_10 = returns.rolling(10).std().fillna(0)
                volatility_20 = returns.rolling(20).std().fillna(0)
                features.extend([volatility_5.values, volatility_10.values, volatility_20.values])
                feature_names.extend(['volatility_5', 'volatility_10', 'volatility_20'])

            # Volume features
            if 'volume' in market_data.columns:
                # Volume momentum
                volume_momentum = market_data['volume'].pct_change().fillna(0)
                features.append(volume_momentum.values)
                feature_names.append('volume_momentum')

                # Volume moving averages
                volume_ma_5 = market_data['volume'].rolling(5).mean().fillna(market_data['volume'])
                volume_ma_10 = market_data['volume'].rolling(10).mean().fillna(market_data['volume'])
                features.extend([volume_ma_5.values, volume_ma_10.values])
                feature_names.extend(['volume_ma_5', 'volume_ma_10'])

                # Volume ratio
                volume_ratio = (market_data['volume'] / volume_ma_10).fillna(1)
                features.append(volume_ratio.values)
                feature_names.append('volume_ratio')

            # Technical indicators
            if 'close' in market_data.columns:
                # RSI
                delta = market_data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                features.append(rsi.fillna(50).values)
                feature_names.append('rsi')

                # MACD
                ema_12 = market_data['close'].ewm(span=12).mean()
                ema_26 = market_data['close'].ewm(span=26).mean()
                macd = ema_12 - ema_26
                features.append(macd.values)
                feature_names.append('macd')

            # Combine all features
            X = np.column_stack(features)

            # Handle NaN values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            self.logger.info(f"Prepared {X.shape[1]} features for regime detection")
            return X, feature_names

        except Exception as e:
            self.logger.error(f"Regime feature preparation failed: {e}")
            raise

    def _prepare_trading_features(self, market_data: pd.DataFrame, regime_data: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
        """Prepare features for trading signal generation."""
        try:
            # Start with regime features
            X_regime, feature_names = self._prepare_regime_features(market_data)

            # Add regime-specific features
            regime_predictions = regime_data.get('regime_predictions', np.zeros(len(market_data)))
            regime_probabilities = regime_data.get('regime_probabilities', np.zeros((len(market_data), 1)))

            # Add regime features
            features = [X_regime]
            feature_names.extend([f'regime_{i}' for i in range(regime_probabilities.shape[1])])

            # Add regime probabilities
            features.append(regime_probabilities)

            # Add regime persistence
            regime_persistence = self._calculate_regime_persistence(regime_predictions)
            features.append(regime_persistence.reshape(-1, 1))
            feature_names.append('regime_persistence')

            # Add regime transitions
            regime_transitions = self._calculate_regime_transitions(regime_predictions)
            features.append(regime_transitions.reshape(-1, 1))
            feature_names.append('regime_transitions')

            # Combine all features
            X = np.column_stack(features)

            # Handle NaN values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            self.logger.info(f"Prepared {X.shape[1]} features for trading")
            return X, feature_names

        except Exception as e:
            self.logger.error(f"Trading feature preparation failed: {e}")
            raise

    def _calculate_regime_persistence(self, regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate regime persistence."""
        try:
            persistence = np.zeros(len(regime_predictions))
            current_persistence = 1

            for i in range(1, len(regime_predictions)):
                if regime_predictions[i] == regime_predictions[i-1]:
                    current_persistence += 1
                else:
                    current_persistence = 1

                persistence[i] = current_persistence

            return persistence

        except Exception as e:
            logger.warning(f"Regime persistence calculation failed: {e}")
            return np.ones(len(regime_predictions))

    def _calculate_regime_transitions(self, regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate regime transitions."""
        try:
            transitions = np.zeros(len(regime_predictions))

            for i in range(1, len(regime_predictions)):
                if regime_predictions[i] != regime_predictions[i-1]:
                    transitions[i] = 1
                else:
                    transitions[i] = 0

            return transitions

        except Exception as e:
            logger.warning(f"Regime transition calculation failed: {e}")
            return np.zeros(len(regime_predictions))

    def get_combined_results(self) -> Dict[str, Any]:
        """Get combined regime detection and trading results."""
        try:
            if self.regime_results is None or self.trading_results is None:
                return {'message': 'No results available'}

            return {
                'regime_detection': self.regime_results,
                'trading_signals': self.trading_results,
                'combined_analysis': {
                    'n_regimes': len(np.unique(self.regime_results['regime_predictions'])),
                    'regime_quality': self.regime_results['regime_quality']['overall_quality'],
                    'signal_distribution': np.bincount(self.trading_results['signals'] + 1),
                    'avg_signal_strength': np.mean(self.trading_results['signal_strengths']),
                    'avg_risk_score': np.mean(self.trading_results['risk_scores']),
                    'avg_position_size': np.mean(np.abs(self.trading_results['position_sizes']))
                }
            }

        except Exception as e:
            self.logger.error(f"Combined results generation failed: {e}")
            return {'error': str(e)}

# Convenience function
def search_regime_trading_architecture(market_data: pd.DataFrame,
                                     timestamps: np.ndarray,
                                     config: Optional[RegimeTradingTreeNASConfig] = None) -> Dict[str, Any]:
    """
    Convenience function to perform regime detection and trading signal generation.

    Args:
        market_data: Market data (OHLCV)
        timestamps: Timestamps for the data
        config: Regime trading tree NAS configuration

    Returns:
        Combined regime detection and trading results
    """
    if config is None:
        config = RegimeTradingTreeNASConfig()

    regime_trading_nas = RegimeTradingTreeNAS(config)

    # Detect regimes
    regime_results = regime_trading_nas.detect_regimes(market_data, timestamps)

    # Generate trading signals
    trading_results = regime_trading_nas.generate_trading_signals(market_data, regime_results)

    # Get combined results
    combined_results = regime_trading_nas.get_combined_results()

    return combined_results

class VectorBTOptimizedRegimeTradingTreeNAS(RegimeTradingTreeNAS):
    """Regime Trading Tree NAS with VectorBT optimization."""
    
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and
                VECTORBT_AVAILABLE)

    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str,
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

    def _pandas_rolling_operation(self, data: pd.Series, operation: str,
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def _vectorbt_apply_operation(self, data: pd.Series, func,
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)

        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
