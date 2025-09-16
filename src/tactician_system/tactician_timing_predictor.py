"""
Tactician Timing Predictor System

This module implements the Tactician system that runs every 30 seconds on 1-minute base
timeframe data, deciding WHEN to trade when the Analyst gives a green light.
Trained on all regimes but only on periods where Analyst gives green light.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import joblib
import os

from src.utils.tprint import tprint
from src.utils.logger import system_logger
from src.core.decorators import handles_errors


@dataclass
class TacticianConfig:
    """Configuration for Tactician timing prediction system."""
    base_timeframe: str = "1m"
    run_interval_seconds: int = 30
    target_threshold: float = 0.5  # 0.5% price change target
    lookback_periods: int = 60  # 1 hour of 1m data
    cross_timeframe_periods: List[int] = None  # Will be set in __post_init__
    models: Dict[str, str] = None  # Will be set in __post_init__
    meta_learner: str = "lightgbm"
    train_test_split: float = 0.8
    random_state: int = 42
    min_confidence_threshold: float = 0.6
    max_position_size: float = 0.1
    max_leverage: float = 3.0
    
    def __post_init__(self):
        if self.cross_timeframe_periods is None:
            # Cross-timeframe periods: 5m, 15m, 30m, 1h, 2h, 4h
            self.cross_timeframe_periods = [5, 15, 30, 60, 120, 240]
        
        if self.models is None:
            self.models = {
                "xgboost": "XGBoost",
                "randomforest": "RandomForestRegressor", 
                "catboost": "CatBoostRegressor",           
                "elastic_net": "Elastic Net"
            }


@dataclass
class TacticianPrediction:
    """Container for Tactician predictions and metadata."""
    timestamp: datetime
    should_enter: bool
    entry_confidence: float
    expected_return: float
    risk_score: float
    position_size: float
    leverage: float
    base_model_predictions: Dict[str, float]
    meta_learner_prediction: float
    feature_importance: Dict[str, float]
    market_timing: Dict[str, Any]


class TacticianTimingPredictor:
    """
    Tactician system for deciding WHEN to trade.
    
    This system:
    - Runs every 30 seconds on 1-minute base timeframe data
    - Uses 50+ cross-timeframe features
    - Integrates HMM outputs and Analyst predictions
    - Trained on all regimes but only on green light periods
    - Finds optimal entry timing for 0.5% price changes
    """
    
    def __init__(self, config: TacticianConfig):
        """Initialize the Tactician timing predictor."""
        self.config = config
        self.logger = system_logger.getChild('TacticianTimingPredictor')
        self.scaler = StandardScaler()
        self.models: Dict[str, Any] = {}
        self.meta_learner: Optional[Any] = None
        self.feature_names: List[str] = []
        self.is_trained = False
        self.last_run_time: Optional[datetime] = None
        
    @handles_errors
    def extract_timing_features(self, data: pd.DataFrame, 
                               hmm_outputs: Optional[Dict[str, Any]] = None,
                               analyst_outputs: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Extract 50+ timing-focused features for entry prediction.
        
        Features include:
        - High-frequency technical indicators
        - Cross-timeframe momentum and volatility
        - Market microstructure features
        - HMM regime probabilities and characteristics
        - Analyst model outputs and confidence
        - Entry timing signals
        - Risk and position sizing indicators
        """
        tprint("Extracting timing features for Tactician...")
        
        features = pd.DataFrame(index=data.index)
        
        # === HIGH-FREQUENCY PRICE FEATURES ===
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        features['price_change'] = data['close'] - data['open']
        features['price_range'] = data['high'] - data['low']
        features['body_size'] = abs(data['close'] - data['open'])
        features['upper_shadow'] = data['high'] - np.maximum(data['open'], data['close'])
        features['lower_shadow'] = np.minimum(data['open'], data['close']) - data['low']
        features['body_ratio'] = features['body_size'] / features['price_range']
        features['shadow_ratio'] = (features['upper_shadow'] + features['lower_shadow']) / features['price_range']
        
        # === SHORT-TERM MOMENTUM ===
        for period in [2, 3, 5, 10, 15, 30]:
            features[f'rsi_{period}'] = self._calculate_rsi(data['close'], period)
            features[f'momentum_{period}'] = data['close'] / data['close'].shift(period) - 1
            features[f'roc_{period}'] = data['close'].pct_change(period)
            features[f'price_acceleration_{period}'] = features[f'momentum_{period}'].diff()
            
        # === HIGH-FREQUENCY VOLATILITY ===
        for period in [2, 5, 10, 15, 30]:
            features[f'atr_{period}'] = self._calculate_atr(data, period)
            features[f'volatility_{period}'] = data['close'].rolling(period).std()
            features[f'volatility_ratio_{period}'] = features[f'volatility_{period}'] / features[f'volatility_{period}'].rolling(20).mean()
            
        # === VOLUME ANALYSIS ===
        if 'volume' in data.columns:
            for period in [2, 5, 10, 15, 30]:
                features[f'volume_ma_{period}'] = data['volume'].rolling(period).mean()
                features[f'volume_ratio_{period}'] = data['volume'] / features[f'volume_ma_{period}']
                features[f'volume_acceleration_{period}'] = features[f'volume_ratio_{period}'].diff()
                
            features['obv'] = self._calculate_obv(data['close'], data['volume'])
            features['vpt'] = self._calculate_vpt(data['close'], data['volume'])
            features['mfi'] = self._calculate_mfi(data, 14)
            features['vwap'] = self._calculate_vwap(data)
            features['volume_price_correlation'] = data['close'].rolling(10).corr(data['volume'])
            
        # === CROSS-TIMEFRAME FEATURES ===
        for period in self.config.cross_timeframe_periods:
            if len(data) > period:
                # Price momentum across timeframes
                features[f'ctf_momentum_{period}'] = data['close'] / data['close'].shift(period) - 1
                features[f'ctf_volatility_{period}'] = data['close'].rolling(period).std()
                features[f'ctf_range_{period}'] = (data['high'].rolling(period).max() - data['low'].rolling(period).min()) / data['close']
                
                # Volume analysis across timeframes
                if 'volume' in data.columns:
                    features[f'ctf_volume_{period}'] = data['volume'].rolling(period).mean()
                    features[f'ctf_volume_ratio_{period}'] = data['volume'] / features[f'ctf_volume_{period}']
                    
                # Technical indicators across timeframes
                features[f'ctf_rsi_{period}'] = self._calculate_rsi(data['close'], period)
                features[f'ctf_atr_{period}'] = self._calculate_atr(data, period)
                
        # === ENTRY TIMING SIGNALS ===
        features['price_breakout'] = self._detect_price_breakout(data)
        features['volume_breakout'] = self._detect_volume_breakout(data)
        features['momentum_divergence'] = self._detect_momentum_divergence(data)
        features['volatility_breakout'] = self._detect_volatility_breakout(data)
        
        # === MARKET MICROSTRUCTURE ===
        features['bid_ask_spread'] = (data['high'] - data['low']) / data['close']
        features['price_impact'] = abs(data['close'] - data['open']) / data['close']
        features['intraday_volatility'] = (data['high'] - data['low']) / data['open']
        features['gap_up'] = data['open'] > data['close'].shift(1)
        features['gap_down'] = data['open'] < data['close'].shift(1)
        features['price_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low'])
        
        # === TREND ANALYSIS ===
        for period in [5, 10, 20, 30]:
            features[f'sma_{period}'] = data['close'].rolling(period).mean()
            features[f'ema_{period}'] = data['close'].ewm(span=period).mean()
            features[f'trend_{period}'] = (data['close'] - features[f'sma_{period}']) / features[f'sma_{period}']
            features[f'trend_strength_{period}'] = features[f'trend_{period}'].rolling(10).std()
            features[f'trend_acceleration_{period}'] = features[f'trend_{period}'].diff()
            
        # === OSCILLATORS ===
        features['stoch_k'] = self._calculate_stochastic_k(data)
        features['stoch_d'] = self._calculate_stochastic_d(data)
        features['williams_r'] = self._calculate_williams_r(data)
        features['cci'] = self._calculate_cci(data)
        features['roc'] = self._calculate_roc(data)
        
        # === HMM INTEGRATION ===
        if hmm_outputs is not None:
            if 'regime_probs' in hmm_outputs:
                regime_probs = hmm_outputs['regime_probs']
                for i, prob in enumerate(regime_probs):
                    features[f'hmm_regime_{i}_prob'] = prob
                    
            if 'dominant_regime' in hmm_outputs:
                features['hmm_dominant_regime'] = hmm_outputs['dominant_regime']
                
            if 'regime_characteristics' in hmm_outputs:
                regime_chars = hmm_outputs['regime_characteristics']
                for key, value in regime_chars.items():
                    features[f'hmm_{key}'] = value
                    
        # === ANALYST INTEGRATION ===
        if analyst_outputs is not None:
            if 'base_model_predictions' in analyst_outputs:
                for model_name, prediction in analyst_outputs['base_model_predictions'].items():
                    features[f'analyst_{model_name}'] = prediction
                    
            if 'meta_learner_prediction' in analyst_outputs:
                features['analyst_meta_prediction'] = analyst_outputs['meta_learner_prediction']
                
            if 'confidence' in analyst_outputs:
                features['analyst_confidence'] = analyst_outputs['confidence']
                
            if 'market_conditions' in analyst_outputs:
                market_conds = analyst_outputs['market_conditions']
                for key, value in market_conds.items():
                    features[f'analyst_{key}'] = value
                    
        # === RISK INDICATORS ===
        features['risk_score'] = self._calculate_risk_score(data)
        features['position_size_indicator'] = self._calculate_position_size_indicator(data)
        features['leverage_indicator'] = self._calculate_leverage_indicator(data)
        features['drawdown_risk'] = self._calculate_drawdown_risk(data)
        
        # === TIMING INDICATORS ===
        features['entry_signal_strength'] = self._calculate_entry_signal_strength(data)
        features['timing_confidence'] = self._calculate_timing_confidence(data)
        features['market_timing_score'] = self._calculate_market_timing_score(data)
        
        # Drop NaN values
        features = features.dropna()
        
        # Ensure we have enough features
        if len(features.columns) < 50:
            tprint(f"Warning: Only {len(features.columns)} features extracted, target is 50+")
        elif len(features.columns) > 100:
            # Select most important features
            feature_vars = features.var().sort_values(ascending=False)
            selected_features = feature_vars.head(100).index.tolist()
            features = features[selected_features]
            
        self.feature_names = features.columns.tolist()
        tprint(f"Extracted {len(self.feature_names)} timing features for Tactician")
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=period).mean()
    
    def _calculate_obv(self, prices: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume."""
        price_change = prices.diff()
        obv = np.where(price_change > 0, volume, 
                      np.where(price_change < 0, -volume, 0)).cumsum()
        return pd.Series(obv, index=prices.index)
    
    def _calculate_vpt(self, prices: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Volume Price Trend."""
        price_change = prices.pct_change()
        vpt = (price_change * volume).cumsum()
        return vpt
    
    def _calculate_mfi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        money_flow = typical_price * data['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(period).sum()
        
        mfi = 100 - (100 / (1 + positive_flow / negative_flow))
        return mfi
    
    def _calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        return (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
    
    def _calculate_stochastic_k(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Stochastic %K."""
        lowest_low = data['low'].rolling(window=period).min()
        highest_high = data['high'].rolling(window=period).max()
        return 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
    
    def _calculate_stochastic_d(self, data: pd.DataFrame, period: int = 14, smooth: int = 3) -> pd.Series:
        """Calculate Stochastic %D."""
        k = self._calculate_stochastic_k(data, period)
        return k.rolling(window=smooth).mean()
    
    def _calculate_williams_r(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        highest_high = data['high'].rolling(window=period).max()
        lowest_low = data['low'].rolling(window=period).min()
        return -100 * ((highest_high - data['close']) / (highest_high - lowest_low))
    
    def _calculate_cci(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        return (typical_price - sma_tp) / (0.015 * mad)
    
    def _calculate_roc(self, data: pd.DataFrame, period: int = 10) -> pd.Series:
        """Calculate Rate of Change."""
        return data['close'].pct_change(period) * 100
    
    def _detect_price_breakout(self, data: pd.DataFrame) -> pd.Series:
        """Detect price breakout signals."""
        sma_20 = data['close'].rolling(20).mean()
        std_20 = data['close'].rolling(20).std()
        upper_band = sma_20 + 2 * std_20
        lower_band = sma_20 - 2 * std_20
        
        breakout_up = data['close'] > upper_band
        breakout_down = data['close'] < lower_band
        
        return np.where(breakout_up, 1, np.where(breakout_down, -1, 0))
    
    def _detect_volume_breakout(self, data: pd.DataFrame) -> pd.Series:
        """Detect volume breakout signals."""
        if 'volume' not in data.columns:
            return pd.Series(0, index=data.index)
            
        volume_ma = data['volume'].rolling(20).mean()
        volume_std = data['volume'].rolling(20).std()
        volume_threshold = volume_ma + 2 * volume_std
        
        return (data['volume'] > volume_threshold).astype(int)
    
    def _detect_momentum_divergence(self, data: pd.DataFrame) -> pd.Series:
        """Detect momentum divergence signals."""
        price_momentum = data['close'].pct_change(10)
        rsi = self._calculate_rsi(data['close'], 14)
        
        # Simple divergence detection
        price_trend = price_momentum.rolling(5).mean()
        rsi_trend = rsi.rolling(5).mean()
        
        divergence = np.where(
            (price_trend > 0) & (rsi_trend < 0), 1,  # Bearish divergence
            np.where((price_trend < 0) & (rsi_trend > 0), -1, 0)  # Bullish divergence
        )
        
        return pd.Series(divergence, index=data.index)
    
    def _detect_volatility_breakout(self, data: pd.DataFrame) -> pd.Series:
        """Detect volatility breakout signals."""
        volatility = data['close'].rolling(20).std()
        vol_ma = volatility.rolling(20).mean()
        vol_std = volatility.rolling(20).std()
        
        vol_breakout = volatility > (vol_ma + 2 * vol_std)
        return vol_breakout.astype(int)
    
    def _calculate_risk_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate risk score for position sizing."""
        volatility = data['close'].rolling(20).std()
        max_drawdown = (data['close'].rolling(20).max() - data['close']) / data['close'].rolling(20).max()
        
        # Normalize and combine
        vol_score = volatility / volatility.rolling(100).max()
        dd_score = max_drawdown / max_drawdown.rolling(100).max()
        
        return (vol_score + dd_score) / 2
    
    def _calculate_position_size_indicator(self, data: pd.DataFrame) -> pd.Series:
        """Calculate position size indicator."""
        risk_score = self._calculate_risk_score(data)
        confidence = self._calculate_timing_confidence(data)
        
        # Position size inversely related to risk, directly to confidence
        return (1 - risk_score) * confidence
    
    def _calculate_leverage_indicator(self, data: pd.DataFrame) -> pd.Series:
        """Calculate leverage indicator."""
        volatility = data['close'].rolling(20).std()
        momentum = data['close'].pct_change(10)
        
        # Higher leverage for lower volatility and higher momentum
        vol_factor = 1 / (volatility / volatility.rolling(100).mean())
        mom_factor = abs(momentum) / momentum.rolling(100).std()
        
        return np.clip(vol_factor * mom_factor, 0.1, self.config.max_leverage)
    
    def _calculate_drawdown_risk(self, data: pd.DataFrame) -> pd.Series:
        """Calculate drawdown risk indicator."""
        rolling_max = data['close'].rolling(20).max()
        drawdown = (rolling_max - data['close']) / rolling_max
        return drawdown.rolling(10).max()
    
    def _calculate_entry_signal_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate entry signal strength."""
        momentum = data['close'].pct_change(5)
        volume_ratio = data['volume'] / data['volume'].rolling(20).mean() if 'volume' in data.columns else 1
        volatility = data['close'].rolling(10).std()
        
        # Combine signals
        signal = abs(momentum) * volume_ratio / (volatility + 1e-8)
        return signal.rolling(5).mean()
    
    def _calculate_timing_confidence(self, data: pd.DataFrame) -> pd.Series:
        """Calculate timing confidence score."""
        rsi = self._calculate_rsi(data['close'], 14)
        stoch_k = self._calculate_stochastic_k(data)
        williams_r = self._calculate_williams_r(data)
        
        # Combine oscillator signals
        rsi_signal = 1 - abs(rsi - 50) / 50
        stoch_signal = 1 - abs(stoch_k - 50) / 50
        williams_signal = 1 - abs(williams_r + 50) / 50
        
        return (rsi_signal + stoch_signal + williams_signal) / 3
    
    def _calculate_market_timing_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate overall market timing score."""
        trend_score = self._calculate_trend_score(data)
        momentum_score = self._calculate_momentum_score(data)
        volume_score = self._calculate_volume_score(data)
        
        return (trend_score + momentum_score + volume_score) / 3
    
    def _calculate_trend_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate trend score."""
        sma_short = data['close'].rolling(10).mean()
        sma_long = data['close'].rolling(30).mean()
        return (sma_short - sma_long) / sma_long
    
    def _calculate_momentum_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate momentum score."""
        momentum_5 = data['close'].pct_change(5)
        momentum_10 = data['close'].pct_change(10)
        return (momentum_5 + momentum_10) / 2
    
    def _calculate_volume_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate volume score."""
        if 'volume' not in data.columns:
            return pd.Series(0, index=data.index)
            
        volume_ratio = data['volume'] / data['volume'].rolling(20).mean()
        return (volume_ratio - 1) / 2  # Normalize around 0
    
    @handles_errors
    def train_models(self, data: pd.DataFrame, analyst_green_lights: np.ndarray,
                    hmm_outputs: Optional[Dict[str, Any]] = None,
                    analyst_outputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train Tactician models on all regimes but only on green light periods.
        
        Args:
            data: Market data
            analyst_green_lights: Boolean array indicating when Analyst gave green light
            hmm_outputs: HMM regime detection outputs
            analyst_outputs: Analyst prediction outputs
        """
        tprint("Training Tactician models on green light periods...")
        
        # Extract features
        features = self.extract_timing_features(data, hmm_outputs, analyst_outputs)
        
        # Align features with green light signals
        min_len = min(len(features), len(analyst_green_lights))
        features = features.iloc[:min_len]
        green_lights = analyst_green_lights[:min_len]
        
        # Only use data where Analyst gave green light
        green_light_mask = green_lights == True
        if not np.any(green_light_mask):
            raise ValueError("No green light periods found for training")
        
        green_features = features[green_light_mask]
        green_data = data.iloc[:min_len][green_light_mask]
        
        # Create target variable (0.5% price change)
        target = (green_data['close'].shift(-1) / green_data['close'] - 1) * 100  # Convert to percentage
        target = target.dropna()
        
        # Align features with target
        min_len_green = min(len(green_features), len(target))
        green_features = green_features.iloc[:min_len_green]
        target = target.iloc[:min_len_green]
        
        if len(green_features) < 10:
            raise ValueError("Insufficient green light data for training")
        
        # Scale features
        features_scaled = self.scaler.fit_transform(green_features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, target, 
            test_size=1-self.config.train_test_split,
            random_state=self.config.random_state
        )
        
        # Train base models
        base_models = {}
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=self.config.random_state
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        base_models['xgboost'] = xgb_pred
        
        # Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=self.config.random_state
        )
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        base_models['randomforest'] = rf_pred
        
        # CatBoost
        catboost_model = CatBoostRegressor(
            iterations=100,
            learning_rate=0.1,
            depth=6,
            random_seed=self.config.random_state,
            verbose=False
        )
        catboost_model.fit(X_train, y_train)
        catboost_pred = catboost_model.predict(X_test)
        base_models['catboost'] = catboost_pred
        
        # Elastic Net
        elastic_model = ElasticNet(alpha=0.1, random_state=self.config.random_state)
        elastic_model.fit(X_train, y_train)
        elastic_pred = elastic_model.predict(X_test)
        base_models['elastic_net'] = elastic_pred
        
        # Train meta-learner (LightGBM)
        meta_features = np.column_stack(list(base_models.values()))
        meta_learner = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=self.config.random_state,
            verbose=-1
        )
        meta_learner.fit(meta_features, y_test)
        
        # Store models
        self.models = {
            'xgboost': xgb_model,
            'randomforest': rf_model,
            'catboost': catboost_model,
            'elastic_net': elastic_model
        }
        self.meta_learner = meta_learner
        
        # Calculate performance metrics
        meta_pred = meta_learner.predict(meta_features)
        mse = mean_squared_error(y_test, meta_pred)
        r2 = r2_score(y_test, meta_pred)
        
        # Calculate additional metrics
        accuracy = np.mean(np.abs(meta_pred - y_test) < 0.1)  # Within 0.1% accuracy
        directional_accuracy = np.mean(np.sign(meta_pred) == np.sign(y_test))
        
        training_results = {
            'mse': mse,
            'r2': r2,
            'accuracy': accuracy,
            'directional_accuracy': directional_accuracy,
            'n_green_light_samples': len(green_features),
            'base_model_predictions': {k: v.tolist() for k, v in base_models.items()}
        }
        
        self.is_trained = True
        tprint(f"Tactician training completed: MSE={mse:.4f}, R²={r2:.4f}, "
              f"Accuracy={accuracy:.3f}, Directional={directional_accuracy:.3f}")
        
        return training_results
    
    @handles_errors
    def predict_entry_timing(self, data: pd.DataFrame, 
                           hmm_outputs: Optional[Dict[str, Any]] = None,
                           analyst_outputs: Optional[Dict[str, Any]] = None) -> TacticianPrediction:
        """
        Predict optimal entry timing for trading.
        
        Returns entry decision, confidence, and position sizing recommendations.
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        # Extract features for the latest data point
        features = self.extract_timing_features(data.tail(self.config.lookback_periods), 
                                              hmm_outputs, analyst_outputs)
        
        if len(features) == 0:
            raise ValueError("Insufficient data for prediction")
        
        # Use the latest data point
        latest_features = features.iloc[-1:].values
        
        # Scale features
        features_scaled = self.scaler.transform(latest_features)
        
        # Get base model predictions
        base_predictions = {}
        for model_name, model in self.models.items():
            pred = model.predict(features_scaled)[0]
            base_predictions[model_name] = pred
        
        # Get meta-learner prediction
        meta_features = np.array(list(base_predictions.values())).reshape(1, -1)
        meta_prediction = self.meta_learner.predict(meta_features)[0]
        
        # Determine if we should enter
        should_enter = abs(meta_prediction) >= self.config.target_threshold
        
        # Calculate confidence
        prediction_consistency = 1 - np.std(list(base_predictions.values())) / (np.mean(np.abs(list(base_predictions.values()))) + 1e-8)
        entry_confidence = min(abs(meta_prediction) / self.config.target_threshold, 1.0) * prediction_consistency
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(data).iloc[-1] if len(data) > 20 else 0.5
        
        # Calculate position size and leverage
        position_size = min(self._calculate_position_size_indicator(data).iloc[-1] * self.config.max_position_size, 
                           self.config.max_position_size) if len(data) > 20 else 0.05
        
        leverage = min(self._calculate_leverage_indicator(data).iloc[-1], 
                      self.config.max_leverage) if len(data) > 20 else 1.0
        
        # Calculate expected return
        expected_return = meta_prediction
        
        # Calculate feature importance (simplified)
        feature_importance = dict(zip(self.feature_names, 
                                    np.abs(features_scaled[0])))
        
        # Market timing analysis
        market_timing = {
            'entry_signal_strength': self._calculate_entry_signal_strength(data).iloc[-1] if len(data) > 5 else 0,
            'timing_confidence': self._calculate_timing_confidence(data).iloc[-1] if len(data) > 14 else 0.5,
            'market_timing_score': self._calculate_market_timing_score(data).iloc[-1] if len(data) > 30 else 0,
            'risk_score': risk_score,
            'volatility': data['close'].rolling(20).std().iloc[-1] if len(data) > 20 else 0
        }
        
        return TacticianPrediction(
            timestamp=datetime.now(),
            should_enter=should_enter,
            entry_confidence=entry_confidence,
            expected_return=expected_return,
            risk_score=risk_score,
            position_size=position_size,
            leverage=leverage,
            base_model_predictions=base_predictions,
            meta_learner_prediction=meta_prediction,
            feature_importance=feature_importance,
            market_timing=market_timing
        )
    
    @handles_errors
    def should_run(self) -> bool:
        """Check if it's time to run the Tactician analysis."""
        if self.last_run_time is None:
            return True
        
        time_since_last_run = datetime.now() - self.last_run_time
        return time_since_last_run >= timedelta(seconds=self.config.run_interval_seconds)
    
    @handles_errors
    def run_analysis(self, data: pd.DataFrame, 
                    hmm_outputs: Optional[Dict[str, Any]] = None,
                    analyst_outputs: Optional[Dict[str, Any]] = None) -> Optional[TacticianPrediction]:
        """
        Run the complete Tactician analysis if it's time.
        
        This method should be called every 30 seconds.
        """
        if not self.should_run():
            return None
        
        if not self.is_trained:
            tprint("Tactician models not trained")
            return None
        
        result = self.predict_entry_timing(data, hmm_outputs, analyst_outputs)
        self.last_run_time = datetime.now()
        
        status = "ENTER" if result.should_enter else "WAIT"
        tprint(f"Tactician analysis completed: {status} "
              f"(confidence: {result.entry_confidence:.3f}, "
              f"expected return: {result.expected_return:.3f}%, "
              f"position size: {result.position_size:.3f})")
        
        return result
    
    def save_models(self, filepath: str) -> None:
        """Save trained models to disk."""
        if not self.is_trained:
            raise ValueError("No trained models to save")
        
        model_data = {
            'config': self.config,
            'scaler': self.scaler,
            'models': self.models,
            'meta_learner': self.meta_learner,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)
        tprint(f"Tactician models saved to {filepath}")
    
    def load_models(self, filepath: str) -> None:
        """Load trained models from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.config = model_data['config']
        self.scaler = model_data['scaler']
        self.models = model_data['models']
        self.meta_learner = model_data['meta_learner']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        
        tprint(f"Tactician models loaded from {filepath}")


# Factory function for easy instantiation
def create_tactician_timing_predictor(config: Optional[TacticianConfig] = None) -> TacticianTimingPredictor:
    """Create and return a new Tactician timing predictor instance."""
    if config is None:
        config = TacticianConfig()
    return TacticianTimingPredictor(config)