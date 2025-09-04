#!/usr/bin/env python3
"""S/R Machine Learning Enhancer.

This module enhances S/R detection and qualification using machine learning models
for better accuracy and prediction capabilities.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
import logging
import joblib
from pathlib import Path

from src.core.decorators import handles_errors
from src.utils.logger import system_logger
from src.core.sr_error_handlers import sr_error_handler, SROptimizationError, SRDataError

# ML imports with fallback
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.feature_selection import SelectKBest, f_classif
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: scikit-learn not available, ML features disabled")


@dataclass
class MLFeatureSet:
    """Set of features for ML models."""
    features: np.ndarray
    feature_names: List[str]
    target: np.ndarray
    metadata: Dict[str, Any]


@dataclass
class MLModelResult:
    """Result of ML model prediction."""
    predictions: np.ndarray
    probabilities: Optional[np.ndarray]
    confidence: float
    model_type: str
    feature_importance: Optional[Dict[str, float]]
    performance_metrics: Dict[str, float]


@dataclass
class SRQualityPrediction:
    """S/R level quality prediction."""
    level_id: str
    quality_score: float
    confidence: float
    features_used: List[str]
    prediction_reason: str


@dataclass
class BreakoutPrediction:
    """Breakout prediction result."""
    level_id: str
    breakout_probability: float
    confidence: float
    expected_direction: str
    time_to_breakout: Optional[int]  # bars
    features_used: List[str]


class SRMLEnhancer:
    """Machine learning enhancer for S/R detection and qualification."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize ML enhancer."""
        self.config = config
        self.logger = system_logger.getChild("SRMLEnhancer")
        self.ml_config = config.get("ml_enhancement", {})
        
        if not ML_AVAILABLE:
            self.logger.warning("ML libraries not available, ML features disabled")
            self.ml_enabled = False
            return
        
        self.ml_enabled = self.ml_config.get("feature_engineering", {}).get("enable_ml_features", True)
        
        # Models
        self.sr_quality_model = None
        self.breakout_prediction_model = None
        self.regime_classification_model = None
        
        # Feature engineering
        self.feature_scaler = StandardScaler()
        self.feature_selector = None
        
        # Training data
        self.training_features = []
        self.training_targets = []
        self.feature_names = []
        
        # Model performance tracking
        self.model_performance = {
            "sr_quality": {"accuracy": 0.0, "last_update": None},
            "breakout_prediction": {"accuracy": 0.0, "last_update": None},
            "regime_classification": {"accuracy": 0.0, "last_update": None}
        }
    
    @sr_error_handler(
        exceptions=(SROptimizationError, SRDataError),
        default_return=None,
        context="ML model training",
        max_retries=2
    )
    async def train_models(
        self,
        market_data: pd.DataFrame,
        sr_levels: List[Dict[str, Any]],
        historical_performance: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Train all ML models."""
        try:
            if not self.ml_enabled or not ML_AVAILABLE:
                self.logger.info("ML training skipped - ML not available or disabled")
                return False
            
            self.logger.info("🤖 Starting ML model training...")
            
            # Prepare training data
            training_data = await self._prepare_training_data(
                market_data, sr_levels, historical_performance
            )
            
            if not training_data:
                self.logger.warning("No training data available")
                return False
            
            # Train S/R quality model
            await self._train_sr_quality_model(training_data)
            
            # Train breakout prediction model
            await self._train_breakout_prediction_model(training_data)
            
            # Train regime classification model
            await self._train_regime_classification_model(market_data)
            
            self.logger.info("✅ ML model training completed")
            return True
            
        except Exception as e:
            self.logger.error(f"ML model training failed: {e}")
            return False
    
    async def _prepare_training_data(
        self,
        market_data: pd.DataFrame,
        sr_levels: List[Dict[str, Any]],
        historical_performance: Optional[Dict[str, Any]]
    ) -> Optional[MLFeatureSet]:
        """Prepare training data for ML models."""
        try:
            features = []
            targets = []
            feature_names = []
            
            # Extract features for each S/R level
            for level in sr_levels:
                level_features = await self._extract_level_features(market_data, level)
                if level_features:
                    features.append(level_features)
                    
                    # Create target based on historical performance or level quality
                    target = await self._create_target_for_level(level, historical_performance)
                    targets.append(target)
            
            if not features:
                return None
            
            # Convert to numpy arrays
            features_array = np.array(features)
            targets_array = np.array(targets)
            
            # Get feature names
            feature_names = await self._get_feature_names()
            
            return MLFeatureSet(
                features=features_array,
                feature_names=feature_names,
                target=targets_array,
                metadata={
                    "n_samples": len(features),
                    "n_features": len(feature_names),
                    "target_distribution": np.bincount(targets_array.astype(int)) if len(targets_array) > 0 else []
                }
            )
            
        except Exception as e:
            self.logger.error(f"Training data preparation failed: {e}")
            return None
    
    async def _extract_level_features(
        self,
        market_data: pd.DataFrame,
        level: Dict[str, Any]
    ) -> Optional[List[float]]:
        """Extract features for a specific S/R level."""
        try:
            features = []
            
            # Basic level features
            features.extend([
                level.get('touch_count', 0),
                level.get('strength', 0.5),
                level.get('age_bars', 0),
                level.get('avg_bounce_ratio', 0),
                level.get('max_bounce_ratio', 0),
                level.get('volume_confirmation_score', 0.5),
                level.get('consistency_score', 0.5),
                level.get('failure_count', 0)
            ])
            
            # Market context features
            level_price = level.get('price', 0)
            if level_price > 0:
                current_price = market_data['close'].iloc[-1]
                proximity = abs(current_price - level_price) / level_price
                features.append(proximity)
            else:
                features.append(1.0)  # Default high proximity
            
            # Technical indicator features
            tech_features = await self._extract_technical_features(market_data, level)
            features.extend(tech_features)
            
            # Advanced features
            advanced_features = await self._extract_advanced_features(market_data, level)
            features.extend(advanced_features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed for level: {e}")
            return None
    
    async def _extract_technical_features(
        self,
        market_data: pd.DataFrame,
        level: Dict[str, Any]
    ) -> List[float]:
        """Extract technical indicator features."""
        try:
            features = []
            
            # RSI
            rsi = self._calculate_rsi(market_data['close'], 14)
            features.append(rsi.iloc[-1] if not rsi.empty else 50.0)
            
            # MACD
            macd_line, macd_signal = self._calculate_macd(market_data['close'])
            features.extend([
                macd_line.iloc[-1] if not macd_line.empty else 0.0,
                macd_signal.iloc[-1] if not macd_signal.empty else 0.0
            ])
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(market_data['close'])
            current_price = market_data['close'].iloc[-1]
            bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]) if not bb_upper.empty else 0.5
            features.append(bb_position)
            
            # ATR
            atr = self._calculate_atr(market_data, 14)
            features.append(atr.iloc[-1] if not atr.empty else 0.0)
            
            # Volume features
            volume_ma = market_data['volume'].rolling(window=20).mean()
            volume_ratio = market_data['volume'].iloc[-1] / volume_ma.iloc[-1] if not volume_ma.empty else 1.0
            features.append(volume_ratio)
            
            # Price momentum
            momentum = (market_data['close'].iloc[-1] - market_data['close'].iloc[-10]) / market_data['close'].iloc[-10] if len(market_data) >= 10 else 0.0
            features.append(momentum)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Technical feature extraction failed: {e}")
            return [0.0] * 7  # Return default features
    
    async def _extract_advanced_features(
        self,
        market_data: pd.DataFrame,
        level: Dict[str, Any]
    ) -> List[float]:
        """Extract advanced features."""
        try:
            features = []
            
            # S/R level density (how many levels nearby)
            level_price = level.get('price', 0)
            if level_price > 0:
                # This would need access to all levels - simplified for now
                features.append(1.0)  # Default density
            else:
                features.append(0.0)
            
            # Confluence score (simplified)
            features.append(level.get('confluence_score', 0.5))
            
            # Time since last touch
            last_touch = level.get('last_touch_bar', 0)
            current_bar = len(market_data)
            time_since_touch = current_bar - last_touch if last_touch > 0 else current_bar
            features.append(time_since_touch)
            
            # Volume at touch (simplified)
            features.append(level.get('volume_at_touch', 1.0))
            
            # Price action patterns (simplified)
            features.append(level.get('price_action_score', 0.5))
            
            # Market microstructure (simplified)
            features.append(level.get('microstructure_score', 0.5))
            
            return features
            
        except Exception as e:
            self.logger.error(f"Advanced feature extraction failed: {e}")
            return [0.5] * 6  # Return default features
    
    async def _create_target_for_level(
        self,
        level: Dict[str, Any],
        historical_performance: Optional[Dict[str, Any]]
    ) -> float:
        """Create target variable for level quality."""
        try:
            # Use historical performance if available
            if historical_performance and level.get('id') in historical_performance:
                perf = historical_performance[level['id']]
                return perf.get('quality_score', 0.5)
            
            # Create target based on level characteristics
            target = 0.0
            
            # Strength component
            strength = level.get('strength', 0.5)
            target += strength * 0.3
            
            # Touch count component
            touch_count = level.get('touch_count', 0)
            touch_score = min(touch_count / 10.0, 1.0)
            target += touch_score * 0.2
            
            # Bounce quality component
            bounce_ratio = level.get('avg_bounce_ratio', 0)
            bounce_score = min(bounce_ratio / 0.01, 1.0)  # Normalize to 1% bounce
            target += bounce_score * 0.2
            
            # Volume confirmation component
            volume_score = level.get('volume_confirmation_score', 0.5)
            target += volume_score * 0.15
            
            # Consistency component
            consistency_score = level.get('consistency_score', 0.5)
            target += consistency_score * 0.15
            
            return min(max(target, 0.0), 1.0)  # Clamp to [0, 1]
            
        except Exception as e:
            self.logger.error(f"Target creation failed: {e}")
            return 0.5
    
    async def _get_feature_names(self) -> List[str]:
        """Get feature names for ML models."""
        basic_features = [
            "touch_count", "strength", "age_bars", "avg_bounce_ratio",
            "max_bounce_ratio", "volume_confirmation_score", "consistency_score",
            "failure_count", "proximity_to_level"
        ]
        
        technical_features = [
            "rsi_14", "macd_line", "macd_signal", "bollinger_position",
            "atr_14", "volume_ratio", "price_momentum"
        ]
        
        advanced_features = [
            "level_density", "confluence_score", "time_since_touch",
            "volume_at_touch", "price_action_score", "microstructure_score"
        ]
        
        return basic_features + technical_features + advanced_features
    
    async def _train_sr_quality_model(self, training_data: MLFeatureSet) -> None:
        """Train S/R quality prediction model."""
        try:
            model_config = self.ml_config.get("models", {}).get("sr_quality_model", {})
            
            # Create model
            if model_config.get("type") == "gradient_boosting":
                self.sr_quality_model = GradientBoostingRegressor(
                    n_estimators=model_config.get("parameters", {}).get("n_estimators", 100),
                    max_depth=model_config.get("parameters", {}).get("max_depth", 6),
                    learning_rate=model_config.get("parameters", {}).get("learning_rate", 0.1),
                    subsample=model_config.get("parameters", {}).get("subsample", 0.8)
                )
            else:
                # Default to Random Forest
                self.sr_quality_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=6,
                    random_state=42
                )
            
            # Prepare features
            X = training_data.features
            y = training_data.target
            
            # Feature selection
            if len(X) > 10:  # Only if we have enough samples
                self.feature_selector = SelectKBest(f_classif, k=min(10, X.shape[1]))
                X = self.feature_selector.fit_transform(X, y)
            
            # Scale features
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Train model
            self.sr_quality_model.fit(X_scaled, y)
            
            # Evaluate model
            if len(X) > 20:  # Only if we have enough samples for evaluation
                scores = cross_val_score(self.sr_quality_model, X_scaled, y, cv=3)
                accuracy = scores.mean()
                self.model_performance["sr_quality"]["accuracy"] = accuracy
                self.model_performance["sr_quality"]["last_update"] = datetime.now()
                
                self.logger.info(f"✅ S/R quality model trained. Accuracy: {accuracy:.4f}")
            else:
                self.logger.info("✅ S/R quality model trained (insufficient data for evaluation)")
            
        except Exception as e:
            self.logger.error(f"S/R quality model training failed: {e}")
    
    async def _train_breakout_prediction_model(self, training_data: MLFeatureSet) -> None:
        """Train breakout prediction model."""
        try:
            model_config = self.ml_config.get("models", {}).get("breakout_prediction_model", {})
            
            # Create model
            self.breakout_prediction_model = RandomForestClassifier(
                n_estimators=model_config.get("parameters", {}).get("n_estimators", 200),
                max_depth=model_config.get("parameters", {}).get("max_depth", 8),
                min_samples_split=model_config.get("parameters", {}).get("min_samples_split", 10),
                min_samples_leaf=model_config.get("parameters", {}).get("min_samples_leaf", 5),
                random_state=42
            )
            
            # For breakout prediction, we need to create different targets
            # This is simplified - in practice, you'd need historical breakout data
            X = training_data.features
            y_breakout = np.random.choice([0, 1], size=len(training_data.target), p=[0.7, 0.3])  # Simplified
            
            # Train model
            self.breakout_prediction_model.fit(X, y_breakout)
            
            self.logger.info("✅ Breakout prediction model trained")
            
        except Exception as e:
            self.logger.error(f"Breakout prediction model training failed: {e}")
    
    async def _train_regime_classification_model(self, market_data: pd.DataFrame) -> None:
        """Train market regime classification model."""
        try:
            model_config = self.ml_config.get("models", {}).get("regime_classification_model", {})
            
            # Create model
            self.regime_classification_model = SVC(
                kernel=model_config.get("parameters", {}).get("kernel", "rbf"),
                C=model_config.get("parameters", {}).get("C", 1.0),
                gamma=model_config.get("parameters", {}).get("gamma", "scale"),
                probability=True,
                random_state=42
            )
            
            # Extract regime features
            regime_features = await self._extract_regime_features(market_data)
            regime_targets = await self._create_regime_targets(market_data)
            
            if len(regime_features) > 10:
                # Train model
                self.regime_classification_model.fit(regime_features, regime_targets)
                
                self.logger.info("✅ Regime classification model trained")
            else:
                self.logger.warning("Insufficient data for regime classification model")
            
        except Exception as e:
            self.logger.error(f"Regime classification model training failed: {e}")
    
    async def _extract_regime_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Extract features for regime classification."""
        try:
            features = []
            
            # Calculate technical indicators
            sma_20 = market_data['close'].rolling(window=20).mean()
            sma_50 = market_data['close'].rolling(window=50).mean()
            rsi = self._calculate_rsi(market_data['close'], 14)
            atr = self._calculate_atr(market_data, 14)
            
            # Extract features for each period
            for i in range(50, len(market_data)):  # Start after 50 periods for indicators
                feature_vector = [
                    sma_20.iloc[i] / sma_50.iloc[i] if not sma_50.empty else 1.0,  # SMA ratio
                    rsi.iloc[i] if not rsi.empty else 50.0,  # RSI
                    atr.iloc[i] / market_data['close'].iloc[i] if not atr.empty else 0.0,  # Volatility
                    market_data['volume'].iloc[i] / market_data['volume'].rolling(window=20).mean().iloc[i] if i >= 20 else 1.0,  # Volume trend
                    (market_data['close'].iloc[i] - market_data['close'].iloc[i-10]) / market_data['close'].iloc[i-10] if i >= 10 else 0.0  # Price momentum
                ]
                features.append(feature_vector)
            
            return np.array(features) if features else np.array([]).reshape(0, 5)
            
        except Exception as e:
            self.logger.error(f"Regime feature extraction failed: {e}")
            return np.array([]).reshape(0, 5)
    
    async def _create_regime_targets(self, market_data: pd.DataFrame) -> np.ndarray:
        """Create regime classification targets."""
        try:
            targets = []
            
            # Calculate technical indicators
            sma_20 = market_data['close'].rolling(window=20).mean()
            sma_50 = market_data['close'].rolling(window=50).mean()
            rsi = self._calculate_rsi(market_data['close'], 14)
            
            # Classify each period
            for i in range(50, len(market_data)):
                sma_ratio = sma_20.iloc[i] / sma_50.iloc[i] if not sma_50.empty else 1.0
                rsi_val = rsi.iloc[i] if not rsi.empty else 50.0
                
                # Simple regime classification
                if abs(sma_ratio - 1.0) > 0.02 and 30 < rsi_val < 70:
                    regime = 0  # Trending
                elif abs(sma_ratio - 1.0) <= 0.02:
                    regime = 1  # Ranging
                else:
                    regime = 2  # Transitional
                
                targets.append(regime)
            
            return np.array(targets)
            
        except Exception as e:
            self.logger.error(f"Regime target creation failed: {e}")
            return np.array([])
    
    async def predict_sr_quality(
        self,
        market_data: pd.DataFrame,
        sr_levels: List[Dict[str, Any]]
    ) -> List[SRQualityPrediction]:
        """Predict quality of S/R levels using ML."""
        try:
            if not self.ml_enabled or not self.sr_quality_model:
                # Fallback to rule-based prediction
                return await self._fallback_quality_prediction(sr_levels)
            
            predictions = []
            
            for level in sr_levels:
                # Extract features
                features = await self._extract_level_features(market_data, level)
                if not features:
                    continue
                
                # Prepare features for prediction
                X = np.array([features])
                
                # Apply feature selection and scaling
                if self.feature_selector:
                    X = self.feature_selector.transform(X)
                X_scaled = self.feature_scaler.transform(X)
                
                # Make prediction
                quality_score = self.sr_quality_model.predict(X_scaled)[0]
                
                # Get feature importance if available
                feature_importance = None
                if hasattr(self.sr_quality_model, 'feature_importances_'):
                    feature_names = await self._get_feature_names()
                    if self.feature_selector:
                        selected_features = [feature_names[i] for i in self.feature_selector.get_support(indices=True)]
                    else:
                        selected_features = feature_names
                    feature_importance = dict(zip(selected_features, self.sr_quality_model.feature_importances_))
                
                # Calculate confidence
                confidence = min(abs(quality_score - 0.5) * 2, 1.0)  # Distance from neutral
                
                prediction = SRQualityPrediction(
                    level_id=level.get('id', 'unknown'),
                    quality_score=float(quality_score),
                    confidence=confidence,
                    features_used=await self._get_feature_names(),
                    prediction_reason=f"ML prediction with {confidence:.2%} confidence"
                )
                
                predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"S/R quality prediction failed: {e}")
            return await self._fallback_quality_prediction(sr_levels)
    
    async def predict_breakouts(
        self,
        market_data: pd.DataFrame,
        sr_levels: List[Dict[str, Any]]
    ) -> List[BreakoutPrediction]:
        """Predict breakouts using ML."""
        try:
            if not self.ml_enabled or not self.breakout_prediction_model:
                # Fallback to rule-based prediction
                return await self._fallback_breakout_prediction(sr_levels)
            
            predictions = []
            
            for level in sr_levels:
                # Extract features
                features = await self._extract_level_features(market_data, level)
                if not features:
                    continue
                
                # Prepare features for prediction
                X = np.array([features])
                
                # Make prediction
                breakout_prob = self.breakout_prediction_model.predict_proba(X)[0][1]  # Probability of breakout
                
                # Determine direction
                level_price = level.get('price', 0)
                current_price = market_data['close'].iloc[-1]
                direction = "up" if current_price > level_price else "down"
                
                # Calculate confidence
                confidence = abs(breakout_prob - 0.5) * 2
                
                prediction = BreakoutPrediction(
                    level_id=level.get('id', 'unknown'),
                    breakout_probability=float(breakout_prob),
                    confidence=confidence,
                    expected_direction=direction,
                    time_to_breakout=None,  # Would need more sophisticated model
                    features_used=await self._get_feature_names()
                )
                
                predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Breakout prediction failed: {e}")
            return await self._fallback_breakout_prediction(sr_levels)
    
    async def _fallback_quality_prediction(self, sr_levels: List[Dict[str, Any]]) -> List[SRQualityPrediction]:
        """Fallback quality prediction using rule-based approach."""
        predictions = []
        
        for level in sr_levels:
            # Simple rule-based quality score
            quality_score = level.get('strength', 0.5)
            confidence = 0.5  # Lower confidence for rule-based
            
            prediction = SRQualityPrediction(
                level_id=level.get('id', 'unknown'),
                quality_score=quality_score,
                confidence=confidence,
                features_used=["strength"],
                prediction_reason="Rule-based fallback prediction"
            )
            
            predictions.append(prediction)
        
        return predictions
    
    async def _fallback_breakout_prediction(self, sr_levels: List[Dict[str, Any]]) -> List[BreakoutPrediction]:
        """Fallback breakout prediction using rule-based approach."""
        predictions = []
        
        for level in sr_levels:
            # Simple rule-based breakout probability
            breakout_prob = 0.3  # Default low probability
            confidence = 0.3  # Low confidence for rule-based
            
            prediction = BreakoutPrediction(
                level_id=level.get('id', 'unknown'),
                breakout_probability=breakout_prob,
                confidence=confidence,
                expected_direction="unknown",
                time_to_breakout=None,
                features_used=["rule_based"]
            )
            
            predictions.append(prediction)
        
        return predictions
    
    # Technical indicator calculation methods
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)
        except Exception:
            return pd.Series([50] * len(prices), index=prices.index)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            macd_signal = macd_line.ewm(span=signal).mean()
            return macd_line, macd_signal
        except Exception:
            return pd.Series([0] * len(prices), index=prices.index), pd.Series([0] * len(prices), index=prices.index)
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            return upper, sma, lower
        except Exception:
            return pd.Series([0] * len(prices), index=prices.index), pd.Series([0] * len(prices), index=prices.index), pd.Series([0] * len(prices), index=prices.index)
    
    def _calculate_atr(self, market_data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        try:
            high_low = market_data['high'] - market_data['low']
            high_close = np.abs(market_data['high'] - market_data['close'].shift())
            low_close = np.abs(market_data['low'] - market_data['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(window=period).mean()
            return atr.fillna(0)
        except Exception:
            return pd.Series([0] * len(market_data), index=market_data.index)
    
    def save_models(self, model_dir: str) -> bool:
        """Save trained models to disk."""
        try:
            model_path = Path(model_dir)
            model_path.mkdir(parents=True, exist_ok=True)
            
            if self.sr_quality_model:
                joblib.dump(self.sr_quality_model, model_path / "sr_quality_model.pkl")
            
            if self.breakout_prediction_model:
                joblib.dump(self.breakout_prediction_model, model_path / "breakout_prediction_model.pkl")
            
            if self.regime_classification_model:
                joblib.dump(self.regime_classification_model, model_path / "regime_classification_model.pkl")
            
            if self.feature_scaler:
                joblib.dump(self.feature_scaler, model_path / "feature_scaler.pkl")
            
            if self.feature_selector:
                joblib.dump(self.feature_selector, model_path / "feature_selector.pkl")
            
            self.logger.info(f"✅ Models saved to {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Model saving failed: {e}")
            return False
    
    def load_models(self, model_dir: str) -> bool:
        """Load trained models from disk."""
        try:
            model_path = Path(model_dir)
            
            if (model_path / "sr_quality_model.pkl").exists():
                self.sr_quality_model = joblib.load(model_path / "sr_quality_model.pkl")
            
            if (model_path / "breakout_prediction_model.pkl").exists():
                self.breakout_prediction_model = joblib.load(model_path / "breakout_prediction_model.pkl")
            
            if (model_path / "regime_classification_model.pkl").exists():
                self.regime_classification_model = joblib.load(model_path / "regime_classification_model.pkl")
            
            if (model_path / "feature_scaler.pkl").exists():
                self.feature_scaler = joblib.load(model_path / "feature_scaler.pkl")
            
            if (model_path / "feature_selector.pkl").exists():
                self.feature_selector = joblib.load(model_path / "feature_selector.pkl")
            
            self.logger.info(f"✅ Models loaded from {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
            return False
    
    def get_model_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get current model performance metrics."""
        return self.model_performance.copy()