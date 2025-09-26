"""
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
            
            # Validate regime generation parameters
            if n_samples < 10:
                tprint_error(f"❌ [REGIME_TRADING_TREE_NAS] Insufficient samples: {n_samples}")
                raise ValueError(f"Insufficient samples for regime detection: {n_samples}")
            
            if n_regimes < 2:
                tprint_error(f"❌ [REGIME_TRADING_TREE_NAS] Insufficient regime types: {n_regimes}")
                raise ValueError(f"Insufficient regime types: {n_regimes}")
            
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
            
            # Validate signal generation parameters
            if n_samples < 5:
                tprint_error(f"❌ [REGIME_TRADING_TREE_NAS] Insufficient samples for signal generation: {n_samples}")
                raise ValueError(f"Insufficient samples for signal generation: {n_samples}")
            
            signal_labels = np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.4, 0.3])  # Hold, Buy, Sell
            tprint(f"📊 [REGIME_TRADING_TREE_NAS] Generated {len(signal_labels)} trading signals", color="cyan")
            
            # Train signal generator
            self.signal_generator.fit(X, signal_labels)
            
            # Generate signals
            signals = self.signal_generator.predict(X)
            signal_strengths = self.signal_generator.get_signal_strength(X)
            
            # Train risk manager
            risk_labels = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])  # Low risk, High risk
            tprint(f"📊 [REGIME_TRADING_TREE_NAS] Generated {len(risk_labels)} risk labels", color="cyan")
            self.risk_manager.fit(X, risk_labels)
            
            # Get risk assessments
            risk_levels = self.risk_manager.predict(X)
            risk_scores = self.risk_manager.get_risk_score(X)
            
            # Train position sizer
            position_sizes = np.random.uniform(-0.1, 0.1, n_samples)  # Synthetic position sizes
            tprint(f"📊 [REGIME_TRADING_TREE_NAS] Generated {len(position_sizes)} position sizes", color="cyan")
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
        """Calculate regime persistence with advanced analysis."""
        try:
            if len(regime_predictions) == 0:
                tprint_error(f"❌ [REGIME_TRADING_TREE_NAS] Empty regime predictions for persistence calculation")
                raise ValueError("Empty regime predictions for persistence calculation")
            
            # Import utilities for advanced calculations
            from src.utils.nas_tas.shared_utils.math_validation_bridge import safe_mean, safe_std, validate_numeric_array
            from src.utils.nas_tas.shared_utils.common_operations_bridge import safe_divide
            
            # Validate input
            regime_predictions = validate_numeric_array(regime_predictions, "regime_predictions")
            
            n_samples = len(regime_predictions)
            persistence = np.zeros(n_samples)
            current_persistence = 1
            
            # Calculate basic persistence
            for i in range(1, n_samples):
                if regime_predictions[i] == regime_predictions[i-1]:
                    current_persistence += 1
                else:
                    current_persistence = 1
                
                persistence[i] = current_persistence
            
            # Advanced persistence analysis
            unique_regimes = np.unique(regime_predictions)
            regime_persistence_stats = {}
            
            for regime in unique_regimes:
                regime_mask = regime_predictions == regime
                regime_persistence_values = persistence[regime_mask]
                
                if len(regime_persistence_values) > 0:
                    regime_persistence_stats[regime] = {
                        'mean_persistence': safe_mean(regime_persistence_values),
                        'std_persistence': safe_std(regime_persistence_values),
                        'max_persistence': np.max(regime_persistence_values),
                        'min_persistence': np.min(regime_persistence_values),
                        'stability_score': safe_divide(safe_mean(regime_persistence_values), 
                                                     safe_std(regime_persistence_values), 0.0)
                    }
            
            # Calculate normalized persistence scores
            max_possible_persistence = n_samples
            normalized_persistence = persistence / max_possible_persistence
            
            # Calculate regime transition indicators
            transition_points = np.zeros(n_samples, dtype=bool)
            for i in range(1, n_samples):
                if regime_predictions[i] != regime_predictions[i-1]:
                    transition_points[i] = True
            
            # Calculate persistence quality metrics
            overall_mean_persistence = safe_mean(persistence)
            overall_std_persistence = safe_std(persistence)
            persistence_consistency = safe_divide(overall_mean_persistence, 
                                                overall_std_persistence, 0.0)
            
            # Store advanced metrics for later use
            self._regime_persistence_metrics = {
                'basic_persistence': persistence,
                'normalized_persistence': normalized_persistence,
                'transition_points': transition_points,
                'regime_stats': regime_persistence_stats,
                'overall_mean': overall_mean_persistence,
                'overall_std': overall_std_persistence,
                'consistency_score': persistence_consistency,
                'n_transitions': np.sum(transition_points),
                'transition_rate': safe_divide(np.sum(transition_points), n_samples, 0.0)
            }
            
            tprint_debug(f"📊 [REGIME_TRADING_TREE_NAS] Calculated advanced regime persistence for {n_samples} samples")
            tprint_debug(f"📊 [REGIME_TRADING_TREE_NAS] Mean persistence: {overall_mean_persistence:.2f}, "
                        f"Consistency: {persistence_consistency:.2f}, "
                        f"Transitions: {np.sum(transition_points)}")
            
            return persistence
            
        except Exception as e:
            tprint_error(f"❌ [REGIME_TRADING_TREE_NAS] Regime persistence calculation failed: {e}")
            logger.error(f"Regime persistence calculation failed: {e}")
            raise RuntimeError(f"Regime persistence calculation failed: {e}") from e
    
    def _calculate_regime_transitions(self, regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate regime transitions with advanced analysis."""
        try:
            if len(regime_predictions) == 0:
                tprint_error(f"❌ [REGIME_TRADING_TREE_NAS] Empty regime predictions for transition calculation")
                raise ValueError("Empty regime predictions for transition calculation")
            
            # Import utilities for advanced calculations
            from src.utils.nas_tas.shared_utils.math_validation_bridge import safe_mean, safe_std, validate_numeric_array
            from src.utils.nas_tas.shared_utils.common_operations_bridge import safe_divide
            
            # Validate input
            regime_predictions = validate_numeric_array(regime_predictions, "regime_predictions")
            
            n_samples = len(regime_predictions)
            transitions = np.zeros(n_samples)
            
            # Calculate basic transitions
            for i in range(1, n_samples):
                if regime_predictions[i] != regime_predictions[i-1]:
                    transitions[i] = 1
                else:
                    transitions[i] = 0
            
            # Advanced transition analysis
            unique_regimes = np.unique(regime_predictions)
            transition_matrix = np.zeros((len(unique_regimes), len(unique_regimes)))
            regime_to_index = {regime: idx for idx, regime in enumerate(unique_regimes)}
            
            # Build transition matrix
            for i in range(1, n_samples):
                if regime_predictions[i] != regime_predictions[i-1]:
                    from_regime = regime_predictions[i-1]
                    to_regime = regime_predictions[i]
                    from_idx = regime_to_index[from_regime]
                    to_idx = regime_to_index[to_regime]
                    transition_matrix[from_idx, to_idx] += 1
            
            # Calculate transition probabilities
            transition_probabilities = np.zeros_like(transition_matrix)
            for i in range(len(unique_regimes)):
                row_sum = np.sum(transition_matrix[i, :])
                if row_sum > 0:
                    transition_probabilities[i, :] = transition_matrix[i, :] / row_sum
            
            # Calculate transition statistics
            n_transitions = np.sum(transitions)
            transition_rate = safe_divide(n_transitions, n_samples, 0.0)
            
            # Calculate transition patterns
            transition_patterns = {}
            for i in range(len(unique_regimes)):
                for j in range(len(unique_regimes)):
                    if i != j and transition_matrix[i, j] > 0:
                        pattern_key = f"{unique_regimes[i]}_to_{unique_regimes[j]}"
                        transition_patterns[pattern_key] = {
                            'count': int(transition_matrix[i, j]),
                            'probability': transition_probabilities[i, j],
                            'from_regime': unique_regimes[i],
                            'to_regime': unique_regimes[j]
                        }
            
            # Calculate transition stability metrics
            transition_stability = {}
            for regime in unique_regimes:
                regime_idx = regime_to_index[regime]
                outgoing_transitions = np.sum(transition_matrix[regime_idx, :])
                incoming_transitions = np.sum(transition_matrix[:, regime_idx])
                
                transition_stability[regime] = {
                    'outgoing_transitions': int(outgoing_transitions),
                    'incoming_transitions': int(incoming_transitions),
                    'stability_score': safe_divide(outgoing_transitions, 
                                                outgoing_transitions + incoming_transitions, 0.0)
                }
            
            # Calculate transition clustering
            transition_intervals = []
            current_interval = 0
            for i in range(1, n_samples):
                if transitions[i] == 1:
                    transition_intervals.append(current_interval)
                    current_interval = 0
                else:
                    current_interval += 1
            transition_intervals.append(current_interval)
            
            # Calculate transition timing statistics
            if transition_intervals:
                mean_interval = safe_mean(np.array(transition_intervals))
                std_interval = safe_std(np.array(transition_intervals))
                transition_regularity = safe_divide(mean_interval, std_interval, 0.0)
            else:
                mean_interval = 0.0
                std_interval = 0.0
                transition_regularity = 0.0
            
            # Store advanced transition metrics
            self._regime_transition_metrics = {
                'basic_transitions': transitions,
                'transition_matrix': transition_matrix,
                'transition_probabilities': transition_probabilities,
                'transition_patterns': transition_patterns,
                'transition_stability': transition_stability,
                'n_transitions': n_transitions,
                'transition_rate': transition_rate,
                'transition_intervals': transition_intervals,
                'mean_interval': mean_interval,
                'std_interval': std_interval,
                'transition_regularity': transition_regularity,
                'unique_regimes': unique_regimes.tolist()
            }
            
            tprint_debug(f"📊 [REGIME_TRADING_TREE_NAS] Calculated advanced regime transitions for {n_samples} samples")
            tprint_debug(f"📊 [REGIME_TRADING_TREE_NAS] Transitions: {n_transitions}, "
                        f"Rate: {transition_rate:.3f}, "
                        f"Regularity: {transition_regularity:.2f}")
            
            return transitions
            
        except Exception as e:
            tprint_error(f"❌ [REGIME_TRADING_TREE_NAS] Regime transition calculation failed: {e}")
            logger.error(f"Regime transition calculation failed: {e}")
            raise RuntimeError(f"Regime transition calculation failed: {e}") from e
    
    def get_combined_results(self) -> Dict[str, Any]:
        """Get combined regime detection and trading results with comprehensive validation."""
        try:
            if self.regime_results is None or self.trading_results is None:
                tprint_warning("⚠️ [REGIME_TRADING_TREE_NAS] No results available for combination")
                return {'message': 'No results available', 'success': False}
            
            # Import utilities for validation and analysis
            from src.utils.nas_tas.shared_utils.math_validation_bridge import safe_mean, safe_std, validate_numeric_array
            from src.utils.nas_tas.shared_utils.common_operations_bridge import safe_divide, safe_weighted_average
            from src.utils.common_utilities import validate_dataframe_columns
            
            # Validate regime results
            regime_validation = self._validate_regime_results()
            if not regime_validation['valid']:
                tprint_error(f"❌ [REGIME_TRADING_TREE_NAS] Regime results validation failed: {regime_validation['error']}")
                return {'error': f"Regime validation failed: {regime_validation['error']}", 'success': False}
            
            # Validate trading results
            trading_validation = self._validate_trading_results()
            if not trading_validation['valid']:
                tprint_error(f"❌ [REGIME_TRADING_TREE_NAS] Trading results validation failed: {trading_validation['error']}")
                return {'error': f"Trading validation failed: {trading_validation['error']}", 'success': False}
            
            # Extract regime metrics
            regime_predictions = self.regime_results['regime_predictions']
            regime_probabilities = self.regime_results['regime_probabilities']
            regime_quality = self.regime_results['regime_quality']
            
            # Extract trading metrics
            signals = self.trading_results['signals']
            signal_strengths = self.trading_results['signal_strengths']
            risk_scores = self.trading_results['risk_scores']
            position_sizes = self.trading_results['position_sizes']
            
            # Calculate comprehensive combined analysis
            combined_analysis = self._calculate_combined_analysis(
                regime_predictions, regime_probabilities, regime_quality,
                signals, signal_strengths, risk_scores, position_sizes
            )
            
            # Calculate regime-trading alignment
            regime_trading_alignment = self._calculate_regime_trading_alignment(
                regime_predictions, signals, signal_strengths
            )
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                regime_quality, signal_strengths, risk_scores, position_sizes
            )
            
            # Calculate risk-adjusted metrics
            risk_adjusted_metrics = self._calculate_risk_adjusted_metrics(
                signal_strengths, risk_scores, position_sizes
            )
            
            # Create comprehensive results
            combined_results = {
                'regime_detection': self.regime_results,
                'trading_signals': self.trading_results,
                'combined_analysis': combined_analysis,
                'regime_trading_alignment': regime_trading_alignment,
                'performance_metrics': performance_metrics,
                'risk_adjusted_metrics': risk_adjusted_metrics,
                'validation_status': {
                    'regime_validation': regime_validation,
                    'trading_validation': trading_validation,
                    'overall_valid': regime_validation['valid'] and trading_validation['valid']
                },
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'n_regimes': len(np.unique(regime_predictions)),
                    'n_samples': len(regime_predictions),
                    'regime_types': self.config.regime_types,
                    'trading_strategies': self.config.trading_strategies
                },
                'success': True
            }
            
            tprint_success(f"✅ [REGIME_TRADING_TREE_NAS] Combined results generated successfully")
            tprint_info(f"📊 [REGIME_TRADING_TREE_NAS] Regimes: {len(np.unique(regime_predictions))}, "
                       f"Quality: {regime_quality['overall_quality']:.3f}, "
                       f"Signals: {len(signals)}")
            
            return combined_results
            
        except Exception as e:
            tprint_error(f"❌ [REGIME_TRADING_TREE_NAS] Combined results generation failed: {e}")
            self.logger.error(f"Combined results generation failed: {e}")
            return {'error': str(e), 'success': False}
    
    def _validate_regime_results(self) -> Dict[str, Any]:
        """Validate regime detection results."""
        try:
            if not self.regime_results:
                return {'valid': False, 'error': 'No regime results available'}
            
            required_keys = ['regime_predictions', 'regime_probabilities', 'regime_quality']
            missing_keys = [key for key in required_keys if key not in self.regime_results]
            
            if missing_keys:
                return {'valid': False, 'error': f'Missing keys: {missing_keys}'}
            
            # Validate regime predictions
            regime_predictions = self.regime_results['regime_predictions']
            if not isinstance(regime_predictions, np.ndarray) or len(regime_predictions) == 0:
                return {'valid': False, 'error': 'Invalid regime predictions'}
            
            # Validate regime probabilities
            regime_probabilities = self.regime_results['regime_probabilities']
            if not isinstance(regime_probabilities, np.ndarray) or regime_probabilities.shape[0] != len(regime_predictions):
                return {'valid': False, 'error': 'Invalid regime probabilities'}
            
            # Validate regime quality
            regime_quality = self.regime_results['regime_quality']
            if not isinstance(regime_quality, dict) or 'overall_quality' not in regime_quality:
                return {'valid': False, 'error': 'Invalid regime quality metrics'}
            
            return {'valid': True, 'error': None}
            
        except Exception as e:
            return {'valid': False, 'error': f'Validation error: {e}'}
    
    def _validate_trading_results(self) -> Dict[str, Any]:
        """Validate trading signal results."""
        try:
            if not self.trading_results:
                return {'valid': False, 'error': 'No trading results available'}
            
            required_keys = ['signals', 'signal_strengths', 'risk_scores', 'position_sizes']
            missing_keys = [key for key in required_keys if key not in self.trading_results]
            
            if missing_keys:
                return {'valid': False, 'error': f'Missing keys: {missing_keys}'}
            
            # Validate signals
            signals = self.trading_results['signals']
            if not isinstance(signals, np.ndarray) or len(signals) == 0:
                return {'valid': False, 'error': 'Invalid signals'}
            
            # Validate signal strengths
            signal_strengths = self.trading_results['signal_strengths']
            if not isinstance(signal_strengths, np.ndarray) or len(signal_strengths) != len(signals):
                return {'valid': False, 'error': 'Invalid signal strengths'}
            
            return {'valid': True, 'error': None}
            
        except Exception as e:
            return {'valid': False, 'error': f'Validation error: {e}'}
    
    def _calculate_combined_analysis(self, regime_predictions, regime_probabilities, regime_quality,
                                   signals, signal_strengths, risk_scores, position_sizes) -> Dict[str, Any]:
        """Calculate comprehensive combined analysis."""
        try:
            from src.utils.nas_tas.shared_utils.math_validation_bridge import safe_mean, safe_std
            
            # Basic metrics
            n_regimes = len(np.unique(regime_predictions))
            n_samples = len(regime_predictions)
            
            # Regime analysis
            regime_analysis = {
                'n_regimes': n_regimes,
                'regime_quality': regime_quality['overall_quality'],
                'regime_distribution': np.bincount(regime_predictions),
                'regime_probability_stats': {
                    'mean_probability': safe_mean(regime_probabilities.flatten()),
                    'std_probability': safe_std(regime_probabilities.flatten()),
                    'max_probability': np.max(regime_probabilities),
                    'min_probability': np.min(regime_probabilities)
                }
            }
            
            # Trading analysis
            trading_analysis = {
                'signal_distribution': np.bincount(signals + 1),  # +1 to handle negative signals
                'signal_strength_stats': {
                    'mean_strength': safe_mean(signal_strengths),
                    'std_strength': safe_std(signal_strengths),
                    'max_strength': np.max(signal_strengths),
                    'min_strength': np.min(signal_strengths)
                },
                'risk_stats': {
                    'mean_risk': safe_mean(risk_scores),
                    'std_risk': safe_std(risk_scores),
                    'max_risk': np.max(risk_scores),
                    'min_risk': np.min(risk_scores)
                },
                'position_stats': {
                    'mean_position_size': safe_mean(np.abs(position_sizes)),
                    'std_position_size': safe_std(np.abs(position_sizes)),
                    'max_position_size': np.max(np.abs(position_sizes)),
                    'min_position_size': np.min(np.abs(position_sizes))
                }
            }
            
            return {
                'regime_analysis': regime_analysis,
                'trading_analysis': trading_analysis,
                'n_samples': n_samples,
                'data_quality': {
                    'regime_quality_score': regime_quality['overall_quality'],
                    'signal_quality_score': safe_mean(signal_strengths),
                    'overall_quality_score': (regime_quality['overall_quality'] + safe_mean(signal_strengths)) / 2
                }
            }
            
        except Exception as e:
            self.logger.error(f"Combined analysis calculation failed: {e}")
            return {'error': str(e)}
    
    def _calculate_regime_trading_alignment(self, regime_predictions, signals, signal_strengths) -> Dict[str, Any]:
        """Calculate alignment between regime detection and trading signals."""
        try:
            from src.utils.nas_tas.shared_utils.math_validation_bridge import safe_correlation, safe_mean
            
            # Calculate regime-signal correlation
            regime_signal_correlation = safe_correlation(regime_predictions, signals)
            
            # Calculate regime-signal strength correlation
            regime_strength_correlation = safe_correlation(regime_predictions, signal_strengths)
            
            # Calculate alignment by regime
            unique_regimes = np.unique(regime_predictions)
            regime_alignment = {}
            
            for regime in unique_regimes:
                regime_mask = regime_predictions == regime
                regime_signals = signals[regime_mask]
                regime_strengths = signal_strengths[regime_mask]
                
                regime_alignment[regime] = {
                    'n_samples': np.sum(regime_mask),
                    'signal_distribution': np.bincount(regime_signals + 1),
                    'mean_signal_strength': safe_mean(regime_strengths),
                    'signal_consistency': safe_std(regime_strengths)
                }
            
            return {
                'regime_signal_correlation': regime_signal_correlation,
                'regime_strength_correlation': regime_strength_correlation,
                'regime_alignment': regime_alignment,
                'overall_alignment_score': (abs(regime_signal_correlation) + abs(regime_strength_correlation)) / 2
            }
            
        except Exception as e:
            self.logger.error(f"Regime-trading alignment calculation failed: {e}")
            return {'error': str(e)}
    
    def _calculate_performance_metrics(self, regime_quality, signal_strengths, risk_scores, position_sizes) -> Dict[str, Any]:
        """Calculate performance metrics."""
        try:
            from src.utils.nas_tas.shared_utils.math_validation_bridge import safe_mean, safe_std, safe_divide
            
            # Signal performance
            signal_performance = {
                'mean_signal_strength': safe_mean(signal_strengths),
                'signal_consistency': safe_divide(safe_mean(signal_strengths), safe_std(signal_strengths), 0.0),
                'signal_quality_score': safe_mean(signal_strengths)
            }
            
            # Risk performance
            risk_performance = {
                'mean_risk_score': safe_mean(risk_scores),
                'risk_consistency': safe_divide(safe_mean(risk_scores), safe_std(risk_scores), 0.0),
                'risk_management_score': 1.0 - safe_mean(risk_scores)  # Lower risk is better
            }
            
            # Position performance
            position_performance = {
                'mean_position_size': safe_mean(np.abs(position_sizes)),
                'position_consistency': safe_divide(safe_mean(np.abs(position_sizes)), safe_std(np.abs(position_sizes)), 0.0),
                'position_management_score': safe_mean(np.abs(position_sizes))
            }
            
            # Overall performance
            overall_performance = {
                'regime_quality': regime_quality['overall_quality'],
                'signal_quality': signal_performance['signal_quality_score'],
                'risk_quality': risk_performance['risk_management_score'],
                'position_quality': position_performance['position_management_score'],
                'combined_score': (
                    regime_quality['overall_quality'] * 0.3 +
                    signal_performance['signal_quality_score'] * 0.3 +
                    risk_performance['risk_management_score'] * 0.2 +
                    position_performance['position_management_score'] * 0.2
                )
            }
            
            return {
                'signal_performance': signal_performance,
                'risk_performance': risk_performance,
                'position_performance': position_performance,
                'overall_performance': overall_performance
            }
            
        except Exception as e:
            self.logger.error(f"Performance metrics calculation failed: {e}")
            return {'error': str(e)}
    
    def _calculate_risk_adjusted_metrics(self, signal_strengths, risk_scores, position_sizes) -> Dict[str, Any]:
        """Calculate risk-adjusted metrics."""
        try:
            from src.utils.nas_tas.shared_utils.math_validation_bridge import safe_mean, safe_std, safe_divide
            
            # Risk-adjusted signal strength
            risk_adjusted_signals = signal_strengths / (1.0 + risk_scores)
            
            # Risk-adjusted position sizes
            risk_adjusted_positions = position_sizes / (1.0 + risk_scores)
            
            # Calculate risk-adjusted metrics
            risk_adjusted_metrics = {
                'risk_adjusted_signal_strength': safe_mean(risk_adjusted_signals),
                'risk_adjusted_position_size': safe_mean(np.abs(risk_adjusted_positions)),
                'risk_adjustment_factor': safe_mean(1.0 / (1.0 + risk_scores)),
                'risk_efficiency': safe_divide(safe_mean(signal_strengths), safe_mean(risk_scores), 0.0),
                'position_efficiency': safe_divide(safe_mean(np.abs(position_sizes)), safe_mean(risk_scores), 0.0)
            }
            
            return risk_adjusted_metrics
            
        except Exception as e:
            self.logger.error(f"Risk-adjusted metrics calculation failed: {e}")
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