"""
Specialized Trading Tree Models

This module provides specialized tree models optimized for different trading strategies
and market regimes, including regime-specific models and trading strategy trees.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from abc import ABC, abstractmethod

# Tree models for financial applications
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

# Advanced tree models
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

class BullMarketTree:
    """Tree model optimized for bull market conditions."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize bull market tree."""
        self.config = config
        self.model = None
        self.is_trained = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train bull market tree."""
        try:
            # Use XGBoost for bull markets (good for trend following)
            if XGBOOST_AVAILABLE:
                self.model = xgb.XGBClassifier(
                    n_estimators=self.config.get('n_estimators', 100),
                    max_depth=self.config.get('max_depth', 8),
                    learning_rate=self.config.get('learning_rate', 0.1),
                    subsample=self.config.get('subsample', 0.8),
                    colsample_bytree=self.config.get('colsample_bytree', 0.8),
                    random_state=42
                )
            else:
                # Fallback to Gradient Boosting
                self.model = GradientBoostingClassifier(
                    n_estimators=self.config.get('n_estimators', 100),
                    max_depth=self.config.get('max_depth', 8),
                    learning_rate=self.config.get('learning_rate', 0.1),
                    random_state=42
                )

            self.model.fit(X, y)
            self.is_trained = True

        except Exception as e:
            logger.error(f"Bull market tree training failed: {e}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict bull market signals."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict bull market probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        return self.model.predict_proba(X)

    def get_momentum_signals(self, X: np.ndarray) -> np.ndarray:
        """Get momentum-based signals for bull markets."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        # Get momentum features (assuming first few features are momentum)
        momentum_features = X[:, :5] if X.shape[1] >= 5 else X

        # Calculate momentum score
        momentum_score = np.mean(momentum_features, axis=1)

        # Generate signals based on momentum
        signals = np.zeros(len(X))
        signals[momentum_score > 0.1] = 1  # Strong buy
        signals[(momentum_score > 0.05) & (momentum_score <= 0.1)] = 0.5  # Buy
        signals[(momentum_score >= -0.05) & (momentum_score <= 0.05)] = 0  # Hold
        signals[momentum_score < -0.05] = -0.5  # Sell

        return signals

class BearMarketTree:
    """Tree model optimized for bear market conditions."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize bear market tree."""
        self.config = config
        self.model = None
        self.is_trained = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train bear market tree."""
        try:
            # Use Random Forest for bear markets (robust to volatility)
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
            logger.error(f"Bear market tree training failed: {e}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict bear market signals."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict bear market probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        return self.model.predict_proba(X)

    def get_risk_signals(self, X: np.ndarray) -> np.ndarray:
        """Get risk-based signals for bear markets."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        # Get volatility features (assuming features 5-10 are volatility)
        volatility_features = X[:, 5:10] if X.shape[1] >= 10 else X[:, 5:]

        # Calculate risk score
        risk_score = np.mean(volatility_features, axis=1)

        # Generate signals based on risk
        signals = np.zeros(len(X))
        signals[risk_score > 0.15] = -1  # Strong sell (high risk)
        signals[(risk_score > 0.1) & (risk_score <= 0.15)] = -0.5  # Sell
        signals[(risk_score >= 0.05) & (risk_score <= 0.1)] = 0  # Hold
        signals[risk_score < 0.05] = 0.5  # Buy (low risk)

        return signals

class SidewaysMarketTree:
    """Tree model optimized for sideways market conditions."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize sideways market tree."""
        self.config = config
        self.model = None
        self.is_trained = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train sideways market tree."""
        try:
            # Use Extra Trees for sideways markets (good for mean reversion)
            self.model = ExtraTreesClassifier(
                n_estimators=self.config.get('n_estimators', 100),
                max_depth=self.config.get('max_depth', 10),
                min_samples_split=self.config.get('min_samples_split', 5),
                min_samples_leaf=self.config.get('min_samples_leaf', 2),
                random_state=42
            )

            self.model.fit(X, y)
            self.is_trained = True

        except Exception as e:
            logger.error(f"Sideways market tree training failed: {e}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict sideways market signals."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict sideways market probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        return self.model.predict_proba(X)

    def get_mean_reversion_signals(self, X: np.ndarray) -> np.ndarray:
        """Get mean reversion signals for sideways markets."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        # Get price ratio features (assuming features 10-15 are price ratios)
        price_ratio_features = X[:, 10:15] if X.shape[1] >= 15 else X[:, 10:]

        # Calculate mean reversion score
        mean_reversion_score = np.mean(price_ratio_features, axis=1)

        # Generate signals based on mean reversion
        signals = np.zeros(len(X))
        signals[mean_reversion_score > 1.05] = -0.5  # Sell (overbought)
        signals[(mean_reversion_score > 1.02) & (mean_reversion_score <= 1.05)] = -0.25  # Weak sell
        signals[(mean_reversion_score >= 0.98) & (mean_reversion_score <= 1.02)] = 0  # Hold
        signals[(mean_reversion_score >= 0.95) & (mean_reversion_score < 0.98)] = 0.25  # Weak buy
        signals[mean_reversion_score < 0.95] = 0.5  # Buy (oversold)

        return signals

class VolatileMarketTree:
    """Tree model optimized for volatile market conditions."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize volatile market tree."""
        self.config = config
        self.model = None
        self.is_trained = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train volatile market tree."""
        try:
            # Use LightGBM for volatile markets (fast and robust)
            if LIGHTGBM_AVAILABLE:
                self.model = lgb.LGBMClassifier(
                    n_estimators=self.config.get('n_estimators', 100),
                    max_depth=self.config.get('max_depth', 8),
                    learning_rate=self.config.get('learning_rate', 0.1),
                    num_leaves=self.config.get('num_leaves', 31),
                    subsample=self.config.get('subsample', 0.8),
                    colsample_bytree=self.config.get('colsample_bytree', 0.8),
                    random_state=42,
                    verbose=-1
                )
            else:
                # Fallback to Gradient Boosting
                self.model = GradientBoostingClassifier(
                    n_estimators=self.config.get('n_estimators', 100),
                    max_depth=self.config.get('max_depth', 8),
                    learning_rate=self.config.get('learning_rate', 0.1),
                    random_state=42
                )

            self.model.fit(X, y)
            self.is_trained = True

        except Exception as e:
            logger.error(f"Volatile market tree training failed: {e}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict volatile market signals."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict volatile market probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        return self.model.predict_proba(X)

    def get_volatility_signals(self, X: np.ndarray) -> np.ndarray:
        """Get volatility-based signals for volatile markets."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        # Get volatility features
        volatility_features = X[:, 5:10] if X.shape[1] >= 10 else X[:, 5:]

        # Calculate volatility score
        volatility_score = np.mean(volatility_features, axis=1)

        # Generate signals based on volatility
        signals = np.zeros(len(X))
        signals[volatility_score > 0.2] = 0  # Hold (too volatile)
        signals[(volatility_score > 0.1) & (volatility_score <= 0.2)] = 0.25  # Weak buy
        signals[(volatility_score >= 0.05) & (volatility_score <= 0.1)] = 0.5  # Buy
        signals[volatility_score < 0.05] = 0.25  # Weak buy (low volatility)

        return signals

class MomentumTradingTree:
    """Tree model optimized for momentum trading strategies."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize momentum trading tree."""
        self.config = config
        self.model = None
        self.is_trained = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train momentum trading tree."""
        try:
            # Use XGBoost for momentum trading (good for trend following)
            if XGBOOST_AVAILABLE:
                self.model = xgb.XGBRegressor(
                    n_estimators=self.config.get('n_estimators', 100),
                    max_depth=self.config.get('max_depth', 8),
                    learning_rate=self.config.get('learning_rate', 0.1),
                    subsample=self.config.get('subsample', 0.8),
                    colsample_bytree=self.config.get('colsample_bytree', 0.8),
                    random_state=42
                )
            else:
                # Fallback to Gradient Boosting
                self.model = GradientBoostingRegressor(
                    n_estimators=self.config.get('n_estimators', 100),
                    max_depth=self.config.get('max_depth', 8),
                    learning_rate=self.config.get('learning_rate', 0.1),
                    random_state=42
                )

            self.model.fit(X, y)
            self.is_trained = True

        except Exception as e:
            logger.error(f"Momentum trading tree training failed: {e}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict momentum signals."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        return self.model.predict(X)

    def get_momentum_signals(self, X: np.ndarray) -> np.ndarray:
        """Get momentum-based trading signals."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        # Get momentum features
        momentum_features = X[:, :5] if X.shape[1] >= 5 else X

        # Calculate momentum score
        momentum_score = np.mean(momentum_features, axis=1)

        # Generate momentum signals
        signals = np.zeros(len(X))
        signals[momentum_score > 0.1] = 1  # Strong buy
        signals[(momentum_score > 0.05) & (momentum_score <= 0.1)] = 0.5  # Buy
        signals[(momentum_score >= -0.05) & (momentum_score <= 0.05)] = 0  # Hold
        signals[(momentum_score >= -0.1) & (momentum_score < -0.05)] = -0.5  # Sell
        signals[momentum_score < -0.1] = -1  # Strong sell

        return signals

class MeanReversionTradingTree:
    """Tree model optimized for mean reversion trading strategies."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize mean reversion trading tree."""
        self.config = config
        self.model = None
        self.is_trained = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train mean reversion trading tree."""
        try:
            # Use Extra Trees for mean reversion (good for overfitting prevention)
            self.model = ExtraTreesRegressor(
                n_estimators=self.config.get('n_estimators', 100),
                max_depth=self.config.get('max_depth', 10),
                min_samples_split=self.config.get('min_samples_split', 5),
                min_samples_leaf=self.config.get('min_samples_leaf', 2),
                random_state=42
            )

            self.model.fit(X, y)
            self.is_trained = True

        except Exception as e:
            logger.error(f"Mean reversion trading tree training failed: {e}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict mean reversion signals."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        return self.model.predict(X)

    def get_mean_reversion_signals(self, X: np.ndarray) -> np.ndarray:
        """Get mean reversion trading signals."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        # Get price ratio features
        price_ratio_features = X[:, 10:15] if X.shape[1] >= 15 else X[:, 10:]

        # Calculate mean reversion score
        mean_reversion_score = np.mean(price_ratio_features, axis=1)

        # Generate mean reversion signals
        signals = np.zeros(len(X))
        signals[mean_reversion_score > 1.05] = -0.5  # Sell (overbought)
        signals[(mean_reversion_score > 1.02) & (mean_reversion_score <= 1.05)] = -0.25  # Weak sell
        signals[(mean_reversion_score >= 0.98) & (mean_reversion_score <= 1.02)] = 0  # Hold
        signals[(mean_reversion_score >= 0.95) & (mean_reversion_score < 0.98)] = 0.25  # Weak buy
        signals[mean_reversion_score < 0.95] = 0.5  # Buy (oversold)

        return signals

class TrendFollowingTree:
    """Tree model optimized for trend following strategies."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize trend following tree."""
        self.config = config
        self.model = None
        self.is_trained = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train trend following tree."""
        try:
            # Use CatBoost for trend following (good for categorical features)
            if CATBOOST_AVAILABLE:
                self.model = cb.CatBoostRegressor(
                    iterations=self.config.get('iterations', 100),
                    depth=self.config.get('depth', 8),
                    learning_rate=self.config.get('learning_rate', 0.1),
                    random_seed=42,
                    verbose=False
                )
            else:
                # Fallback to Gradient Boosting
                self.model = GradientBoostingRegressor(
                    n_estimators=self.config.get('n_estimators', 100),
                    max_depth=self.config.get('max_depth', 8),
                    learning_rate=self.config.get('learning_rate', 0.1),
                    random_state=42
                )

            self.model.fit(X, y)
            self.is_trained = True

        except Exception as e:
            logger.error(f"Trend following tree training failed: {e}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict trend following signals."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        return self.model.predict(X)

    def get_trend_signals(self, X: np.ndarray) -> np.ndarray:
        """Get trend-based trading signals."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        # Get trend features
        trend_features = X[:, 15:20] if X.shape[1] >= 20 else X[:, 15:]

        # Calculate trend score
        trend_score = np.mean(trend_features, axis=1)

        # Generate trend signals
        signals = np.zeros(len(X))
        signals[trend_score > 0.1] = 1  # Strong buy
        signals[(trend_score > 0.05) & (trend_score <= 0.1)] = 0.5  # Buy
        signals[(trend_score >= -0.05) & (trend_score <= 0.05)] = 0  # Hold
        signals[(trend_score >= -0.1) & (trend_score < -0.05)] = -0.5  # Sell
        signals[trend_score < -0.1] = -1  # Strong sell

        return signals

class RiskManagementTree:
    """Tree model optimized for risk management."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize risk management tree."""
        self.config = config
        self.model = None
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

    def get_risk_adjusted_signals(self, X: np.ndarray, base_signals: np.ndarray) -> np.ndarray:
        """Get risk-adjusted trading signals."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        # Get risk probabilities
        risk_proba = self.model.predict_proba(X)
        high_risk_prob = risk_proba[:, 1] if len(risk_proba[0]) > 1 else risk_proba.flatten()

        # Adjust signals based on risk
        adjusted_signals = base_signals.copy()

        # Reduce signal strength for high risk
        high_risk_mask = high_risk_prob > 0.7
        adjusted_signals[high_risk_mask] *= 0.5

        # Increase signal strength for low risk
        low_risk_mask = high_risk_prob < 0.3
        adjusted_signals[low_risk_mask] *= 1.2

        # Clip signals to valid range
        adjusted_signals = np.clip(adjusted_signals, -1, 1)

        return adjusted_signals

class PositionSizingTree:
    """Tree model optimized for position sizing."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize position sizing tree."""
        self.config = config
        self.model = None
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
        max_position = self.config.get('max_position_size', 0.1)
        position_sizes = np.clip(position_sizes, -max_position, max_position)

        return position_sizes

    def get_kelly_position_sizes(self, X: np.ndarray, win_rate: float, avg_win: float, avg_loss: float) -> np.ndarray:
        """Get Kelly criterion position sizes."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        # Get base position sizes
        base_sizes = self.predict(X)

        # Calculate Kelly criterion
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win

        # Apply Kelly criterion
        kelly_sizes = base_sizes * kelly_fraction

        # Apply constraints
        max_position = self.config.get('max_position_size', 0.1)
        kelly_sizes = np.clip(kelly_sizes, -max_position, max_position)

        return kelly_sizes

class RegimeSpecificTreeFactory:
    """Factory for creating regime-specific tree models."""

    @staticmethod
    def create_regime_tree(regime_type: str, config: Dict[str, Any]):
        """Create tree model for specific regime type."""
        if regime_type == 'bull':
            return BullMarketTree(config)
        elif regime_type == 'bear':
            return BearMarketTree(config)
        elif regime_type == 'sideways':
            return SidewaysMarketTree(config)
        elif regime_type == 'volatile':
            return VolatileMarketTree(config)
        else:
            raise ValueError(f"Unknown regime type: {regime_type}")

    @staticmethod
    def create_trading_tree(strategy_type: str, config: Dict[str, Any]):
        """Create tree model for specific trading strategy."""
        if strategy_type == 'momentum':
            return MomentumTradingTree(config)
        elif strategy_type == 'mean_reversion':
            return MeanReversionTradingTree(config)
        elif strategy_type == 'trend_following':
            return TrendFollowingTree(config)
        elif strategy_type == 'risk_management':
            return RiskManagementTree(config)
        elif strategy_type == 'position_sizing':
            return PositionSizingTree(config)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

class AdaptiveTradingTree:
    """Adaptive tree model that switches between strategies based on market conditions."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize adaptive trading tree."""
        self.config = config
        self.regime_trees = {}
        self.trading_trees = {}
        self.is_trained = False

        # Initialize regime-specific trees
        for regime_type in ['bull', 'bear', 'sideways', 'volatile']:
            self.regime_trees[regime_type] = RegimeSpecificTreeFactory.create_regime_tree(
                regime_type, config.get('regime_configs', {}).get(regime_type, {})
            )

        # Initialize trading strategy trees
        for strategy_type in ['momentum', 'mean_reversion', 'trend_following']:
            self.trading_trees[strategy_type] = RegimeSpecificTreeFactory.create_trading_tree(
                strategy_type, config.get('trading_configs', {}).get(strategy_type, {})
            )

    def fit(self, X: np.ndarray, y: np.ndarray, regime_labels: np.ndarray):
        """Train adaptive trading tree."""
        try:
            # Train regime-specific trees
            for regime_type, tree in self.regime_trees.items():
                regime_mask = regime_labels == regime_type
                if np.sum(regime_mask) > 0:
                    tree.fit(X[regime_mask], y[regime_mask])

            # Train trading strategy trees
            for strategy_type, tree in self.trading_trees.items():
                tree.fit(X, y)

            self.is_trained = True

        except Exception as e:
            logger.error(f"Adaptive trading tree training failed: {e}")
            raise

    def predict(self, X: np.ndarray, regime_predictions: np.ndarray) -> np.ndarray:
        """Predict using adaptive strategy."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        # Get predictions from regime-specific trees
        regime_predictions_dict = {}
        for regime_type, tree in self.regime_trees.items():
            if hasattr(tree, 'predict'):
                regime_predictions_dict[regime_type] = tree.predict(X)

        # Get predictions from trading strategy trees
        strategy_predictions_dict = {}
        for strategy_type, tree in self.trading_trees.items():
            if hasattr(tree, 'predict'):
                strategy_predictions_dict[strategy_type] = tree.predict(X)

        # Combine predictions based on regime
        final_predictions = np.zeros(len(X))
        for i, regime in enumerate(regime_predictions):
            if regime in regime_predictions_dict:
                final_predictions[i] = regime_predictions_dict[regime][i]
            else:
                # Use momentum strategy as default
                if 'momentum' in strategy_predictions_dict:
                    final_predictions[i] = strategy_predictions_dict['momentum'][i]

        return final_predictions

    def get_adaptive_signals(self, X: np.ndarray, regime_predictions: np.ndarray) -> np.ndarray:
        """Get adaptive trading signals."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        # Get signals from regime-specific trees
        regime_signals = {}
        for regime_type, tree in self.regime_trees.items():
            if hasattr(tree, 'get_momentum_signals'):
                regime_signals[regime_type] = tree.get_momentum_signals(X)
            elif hasattr(tree, 'get_risk_signals'):
                regime_signals[regime_type] = tree.get_risk_signals(X)
            elif hasattr(tree, 'get_mean_reversion_signals'):
                regime_signals[regime_type] = tree.get_mean_reversion_signals(X)
            elif hasattr(tree, 'get_volatility_signals'):
                regime_signals[regime_type] = tree.get_volatility_signals(X)

        # Get signals from trading strategy trees
        strategy_signals = {}
        for strategy_type, tree in self.trading_trees.items():
            if hasattr(tree, 'get_momentum_signals'):
                strategy_signals[strategy_type] = tree.get_momentum_signals(X)
            elif hasattr(tree, 'get_mean_reversion_signals'):
                strategy_signals[strategy_type] = tree.get_mean_reversion_signals(X)
            elif hasattr(tree, 'get_trend_signals'):
                strategy_signals[strategy_type] = tree.get_trend_signals(X)

        # Combine signals based on regime
        final_signals = np.zeros(len(X))
        for i, regime in enumerate(regime_predictions):
            if regime in regime_signals:
                final_signals[i] = regime_signals[regime][i]
            else:
                # Use momentum strategy as default
                if 'momentum' in strategy_signals:
                    final_signals[i] = strategy_signals['momentum'][i]

        return final_signals
