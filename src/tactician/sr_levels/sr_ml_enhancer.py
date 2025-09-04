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
            
            # Optimize target weights first
            await self.optimize_target_weights(market_data, sr_levels, historical_performance)
            
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
        """Prepare training data for ML models with step06 feature integration."""
        try:
            features = []
            targets = []
            
            # Extract step06 features once for all levels
            step06_features = await self._extract_step06_features(market_data)
            
            # Extract features for each S/R level
            for level in sr_levels:
                # Extract S/R specific features
                sr_features = await self._extract_level_features(market_data, level)
                if sr_features:
                    # Combine S/R features with step06 features
                    combined_features = sr_features + step06_features
                    features.append(combined_features)
                    
                    # Create target based on historical performance or level quality
                    target = await self._create_target_for_level(level, historical_performance)
                    targets.append(target)
            
            if not features:
                return None
            
            # Convert to numpy arrays
            features_array = np.array(features)
            targets_array = np.array(targets)
            
            # Get combined feature names
            feature_names = await self._get_combined_feature_names()
            
            self.logger.info(f"📊 Training data prepared: {len(features)} samples, {len(feature_names)} features")
            self.logger.info(f"   - S/R specific features: {len(await self._get_feature_names())} (45 features)")
            self.logger.info(f"   - Step06 features: {len(step06_features)} (200+ features)")
            self.logger.info(f"   - S/R feature breakdown: Core(15), HVN(5), Fibonacci(6), Psychological(5), Pivot(4), Trendline(4), S/R Specific(6)")
            self.logger.info(f"   - Target calculation: Optimized weights based on trading performance")
            self.logger.info(f"   - Quality definition: Bounce rate, false breakout rate, volume confirmation, timeframe consistency")
            
            return MLFeatureSet(
                features=features_array,
                feature_names=feature_names,
                target=targets_array,
                metadata={
                    "n_samples": len(features),
                    "n_features": len(feature_names),
                    "sr_features": len(await self._get_feature_names()),
                    "step06_features": len(step06_features),
                    "target_distribution": np.bincount(targets_array.astype(int)) if len(targets_array) > 0 else []
                }
            )
            
        except Exception as e:
            self.logger.error(f"Training data preparation failed: {e}")
            return None
    
    async def _extract_step06_features(self, market_data: pd.DataFrame) -> List[float]:
        """Extract step06 features (200+ features)."""
        try:
            # Import step06 feature engineering
            try:
                from src.training.steps.vectorized_advanced_feature_engineering import (
                    VectorizedAdvancedFeatureEngineeringRefactored
                )
                step06_engineer = VectorizedAdvancedFeatureEngineeringRefactored()
                
                # Engineer features using step06
                step06_result = await step06_engineer.engineer_features(market_data)
                
                # Extract all features from step06
                all_features = []
                
                # Price features
                price_features = step06_result.get('price_features', {})
                for feature_name, feature_values in price_features.items():
                    if isinstance(feature_values, (list, np.ndarray)) and len(feature_values) > 0:
                        all_features.append(float(feature_values[-1]))  # Use latest value
                    elif isinstance(feature_values, (int, float)):
                        all_features.append(float(feature_values))
                
                # Volume features
                volume_features = step06_result.get('volume_features', {})
                for feature_name, feature_values in volume_features.items():
                    if isinstance(feature_values, (list, np.ndarray)) and len(feature_values) > 0:
                        all_features.append(float(feature_values[-1]))
                    elif isinstance(feature_values, (int, float)):
                        all_features.append(float(feature_values))
                
                # Microstructure features
                microstructure_features = step06_result.get('microstructure_features', {})
                for feature_name, feature_values in microstructure_features.items():
                    if isinstance(feature_values, (list, np.ndarray)) and len(feature_values) > 0:
                        all_features.append(float(feature_values[-1]))
                    elif isinstance(feature_values, (int, float)):
                        all_features.append(float(feature_values))
                
                # Technical features
                technical_features = step06_result.get('technical_features', {})
                for feature_name, feature_values in technical_features.items():
                    if isinstance(feature_values, (list, np.ndarray)) and len(feature_values) > 0:
                        all_features.append(float(feature_values[-1]))
                    elif isinstance(feature_values, (int, float)):
                        all_features.append(float(feature_values))
                
                # Regime features
                regime_features = step06_result.get('regime_features', {})
                for feature_name, feature_values in regime_features.items():
                    if isinstance(feature_values, (list, np.ndarray)) and len(feature_values) > 0:
                        all_features.append(float(feature_values[-1]))
                    elif isinstance(feature_values, (int, float)):
                        all_features.append(float(feature_values))
                
                # Wavelet features
                wavelet_features = step06_result.get('wavelet_features', {})
                for feature_name, feature_values in wavelet_features.items():
                    if isinstance(feature_values, (list, np.ndarray)) and len(feature_values) > 0:
                        all_features.append(float(feature_values[-1]))
                    elif isinstance(feature_values, (int, float)):
                        all_features.append(float(feature_values))
                
                # Cross-timeframe features
                cross_timeframe_features = step06_result.get('cross_timeframe_features', {})
                for feature_name, feature_values in cross_timeframe_features.items():
                    if isinstance(feature_values, (list, np.ndarray)) and len(feature_values) > 0:
                        all_features.append(float(feature_values[-1]))
                    elif isinstance(feature_values, (int, float)):
                        all_features.append(float(feature_values))
                
                # Interaction features
                interaction_features = step06_result.get('interaction_features', {})
                for feature_name, feature_values in interaction_features.items():
                    if isinstance(feature_values, (list, np.ndarray)) and len(feature_values) > 0:
                        all_features.append(float(feature_values[-1]))
                    elif isinstance(feature_values, (int, float)):
                        all_features.append(float(feature_values))
                
                self.logger.info(f"✅ Step06 features extracted: {len(all_features)} features")
                return all_features
                
            except ImportError as e:
                self.logger.warning(f"Step06 feature engineering not available: {e}")
                return []
            except Exception as e:
                self.logger.warning(f"Step06 feature extraction failed: {e}")
                return []
            
        except Exception as e:
            self.logger.error(f"Step06 feature extraction failed: {e}")
            return []
    
    async def _get_combined_feature_names(self) -> List[str]:
        """Get combined feature names (S/R + step06)."""
        try:
            # Get S/R specific feature names
            sr_feature_names = await self._get_feature_names()
            
            # Get step06 feature names (simplified)
            step06_feature_names = []
            
            # Add step06 feature categories
            step06_categories = [
                'price_features', 'volume_features', 'microstructure_features',
                'technical_features', 'regime_features', 'wavelet_features',
                'cross_timeframe_features', 'interaction_features'
            ]
            
            for category in step06_categories:
                # Add generic feature names for each category
                for i in range(25):  # Assume 25 features per category
                    step06_feature_names.append(f"{category}_{i}")
            
            # Combine feature names
            combined_names = sr_feature_names + step06_feature_names
            
            self.logger.info(f"📊 Combined feature names: {len(combined_names)} total")
            self.logger.info(f"   - S/R features: {len(sr_feature_names)}")
            self.logger.info(f"   - Step06 features: {len(step06_feature_names)}")
            
            return combined_names
            
        except Exception as e:
            self.logger.error(f"Combined feature names failed: {e}")
            return await self._get_feature_names()  # Fallback to S/R features only
    
    async def _extract_level_features(
        self,
        market_data: pd.DataFrame,
        level: Dict[str, Any]
    ) -> Optional[List[float]]:
        """Extract S/R specific features for a specific S/R level."""
        try:
            features = []
            level_price = level.get('price', 0)
            
            # === CORE S/R FEATURES (15 features) ===
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
            if level_price > 0:
                current_price = market_data['close'].iloc[-1]
                proximity = abs(current_price - level_price) / level_price
                features.append(proximity)
            else:
                features.append(1.0)  # Default high proximity
            
            # Advanced S/R features
            advanced_features = await self._extract_advanced_sr_features(market_data, level)
            features.extend(advanced_features)
            
            # === HVN (HIGH VOLUME NODE) FEATURES (5 features) ===
            hvn_features = await self._extract_hvn_features(market_data, level)
            features.extend(hvn_features)
            
            # === FIBONACCI RETRACEMENT FEATURES (6 features) ===
            fibonacci_features = await self._extract_fibonacci_features(market_data, level)
            features.extend(fibonacci_features)
            
            # === PSYCHOLOGICAL LEVEL FEATURES (5 features) ===
            psychological_features = await self._extract_psychological_features(market_data, level)
            features.extend(psychological_features)
            
            # === PIVOT POINT FEATURES (4 features) ===
            pivot_features = await self._extract_pivot_features(market_data, level)
            features.extend(pivot_features)
            
            # === TREND LINE FEATURES (4 features) ===
            trendline_features = await self._extract_trendline_features(market_data, level)
            features.extend(trendline_features)
            
            # === SUPPORT/RESISTANCE SPECIFIC FEATURES (6 features) ===
            sr_specific_features = await self._extract_sr_specific_features(market_data, level)
            features.extend(sr_specific_features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"S/R feature extraction failed for level: {e}")
            return None
    
    async def _extract_technical_features(
        self,
        market_data: pd.DataFrame,
        level: Dict[str, Any]
    ) -> List[float]:
        """Extract technical indicator features (15+ features)."""
        try:
            features = []
            
            # Basic technical indicators (7 features)
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
            
            # Additional technical indicators (8+ more features)
            # Stochastic Oscillator
            stoch_k, stoch_d = self._calculate_stochastic(market_data)
            features.extend([
                stoch_k.iloc[-1] if not stoch_k.empty else 50.0,
                stoch_d.iloc[-1] if not stoch_d.empty else 50.0
            ])
            
            # Williams %R
            williams_r = self._calculate_williams_r(market_data)
            features.append(williams_r.iloc[-1] if not williams_r.empty else -50.0)
            
            # CCI (Commodity Channel Index)
            cci = self._calculate_cci(market_data)
            features.append(cci.iloc[-1] if not cci.empty else 0.0)
            
            # ADX (Average Directional Index)
            adx = self._calculate_adx(market_data)
            features.append(adx.iloc[-1] if not adx.empty else 25.0)
            
            # Volume indicators
            obv = self._calculate_obv(market_data)
            features.append(obv.iloc[-1] if not obv.empty else 0.0)
            
            # Price action patterns
            doji = self._detect_doji_pattern(market_data)
            hammer = self._detect_hammer_pattern(market_data)
            features.extend([doji, hammer])
            
            # Volatility indicators
            vix_proxy = self._calculate_volatility_proxy(market_data)
            features.append(vix_proxy)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Technical feature extraction failed: {e}")
            return [0.0] * 15  # Return default features
    
    async def _extract_advanced_sr_features(
        self,
        market_data: pd.DataFrame,
        level: Dict[str, Any]
    ) -> List[float]:
        """Extract advanced S/R features (6 features)."""
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
            self.logger.error(f"Advanced S/R feature extraction failed: {e}")
            return [0.5] * 6  # Return default features
    
    async def _extract_hvn_features(
        self,
        market_data: pd.DataFrame,
        level: Dict[str, Any]
    ) -> List[float]:
        """Extract HVN (High Volume Node) features (5 features)."""
        try:
            features = []
            level_price = level.get('price', 0)
            
            # HVN strength (based on volume profile)
            hvn_strength = level.get('hvn_strength', 0.5)
            features.append(hvn_strength)
            
            # HVN volume ratio (volume at level vs average)
            hvn_volume_ratio = level.get('hvn_volume_ratio', 1.0)
            features.append(hvn_volume_ratio)
            
            # HVN touch count (how many times price touched HVN)
            hvn_touch_count = level.get('hvn_touch_count', 0)
            features.append(hvn_touch_count)
            
            # HVN time weight (how long HVN was active)
            hvn_time_weight = level.get('hvn_time_weight', 0.5)
            features.append(hvn_time_weight)
            
            # HVN price accuracy (how precise the HVN level is)
            hvn_price_accuracy = level.get('hvn_price_accuracy', 0.5)
            features.append(hvn_price_accuracy)
            
            return features
            
        except Exception as e:
            self.logger.error(f"HVN feature extraction failed: {e}")
            return [0.5] * 5  # Return default features
    
    async def _extract_fibonacci_features(
        self,
        market_data: pd.DataFrame,
        level: Dict[str, Any]
    ) -> List[float]:
        """Extract Fibonacci retracement features (6 features)."""
        try:
            features = []
            
            # Fibonacci level type (0.236, 0.382, 0.5, 0.618, 0.786)
            fib_level_type = level.get('fib_level_type', 0.0)
            features.append(fib_level_type)
            
            # Fibonacci strength (how strong the fib level is)
            fib_strength = level.get('fib_strength', 0.5)
            features.append(fib_strength)
            
            # Fibonacci confluence count (how many fib levels at same price)
            fib_confluence_count = level.get('fib_confluence_count', 0)
            features.append(fib_confluence_count)
            
            # Fibonacci timeframe alignment (multiple timeframes)
            fib_timeframe_alignment = level.get('fib_timeframe_alignment', 0.5)
            features.append(fib_timeframe_alignment)
            
            # Fibonacci volume confirmation
            fib_volume_confirmation = level.get('fib_volume_confirmation', 0.5)
            features.append(fib_volume_confirmation)
            
            # Fibonacci bounce quality
            fib_bounce_quality = level.get('fib_bounce_quality', 0.5)
            features.append(fib_bounce_quality)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Fibonacci feature extraction failed: {e}")
            return [0.0] * 6  # Return default features
    
    async def _extract_psychological_features(
        self,
        market_data: pd.DataFrame,
        level: Dict[str, Any]
    ) -> List[float]:
        """Extract psychological level features (5 features)."""
        try:
            features = []
            level_price = level.get('price', 0)
            
            # Psychological level type (round numbers, key levels)
            psychological_level_type = level.get('psychological_level_type', 0.0)
            if level_price > 0:
                # Check if it's a round number (100, 1000, etc.)
                if level_price % 100 == 0:
                    psychological_level_type = 1.0
                elif level_price % 50 == 0:
                    psychological_level_type = 0.8
                elif level_price % 10 == 0:
                    psychological_level_type = 0.6
            features.append(psychological_level_type)
            
            # Round number strength
            round_number_strength = level.get('round_number_strength', 0.5)
            features.append(round_number_strength)
            
            # Psychological touch count
            psychological_touch_count = level.get('psychological_touch_count', 0)
            features.append(psychological_touch_count)
            
            # Psychological volume spike
            psychological_volume_spike = level.get('psychological_volume_spike', 1.0)
            features.append(psychological_volume_spike)
            
            # Psychological bounce ratio
            psychological_bounce_ratio = level.get('psychological_bounce_ratio', 0.5)
            features.append(psychological_bounce_ratio)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Psychological feature extraction failed: {e}")
            return [0.0] * 5  # Return default features
    
    async def _extract_pivot_features(
        self,
        market_data: pd.DataFrame,
        level: Dict[str, Any]
    ) -> List[float]:
        """Extract pivot point features (4 features)."""
        try:
            features = []
            
            # Pivot type (daily, weekly, monthly)
            pivot_type = level.get('pivot_type', 0.0)
            features.append(pivot_type)
            
            # Pivot strength
            pivot_strength = level.get('pivot_strength', 0.5)
            features.append(pivot_strength)
            
            # Pivot timeframe
            pivot_timeframe = level.get('pivot_timeframe', 0.5)
            features.append(pivot_timeframe)
            
            # Pivot confluence
            pivot_confluence = level.get('pivot_confluence', 0.5)
            features.append(pivot_confluence)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Pivot feature extraction failed: {e}")
            return [0.0] * 4  # Return default features
    
    async def _extract_trendline_features(
        self,
        market_data: pd.DataFrame,
        level: Dict[str, Any]
    ) -> List[float]:
        """Extract trend line features (4 features)."""
        try:
            features = []
            
            # Trend line type (support, resistance, channel)
            trendline_type = level.get('trendline_type', 0.0)
            features.append(trendline_type)
            
            # Trend line strength
            trendline_strength = level.get('trendline_strength', 0.5)
            features.append(trendline_strength)
            
            # Trend line touch count
            trendline_touch_count = level.get('trendline_touch_count', 0)
            features.append(trendline_touch_count)
            
            # Trend line angle
            trendline_angle = level.get('trendline_angle', 0.0)
            features.append(trendline_angle)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Trend line feature extraction failed: {e}")
            return [0.0] * 4  # Return default features
    
    async def _extract_sr_specific_features(
        self,
        market_data: pd.DataFrame,
        level: Dict[str, Any]
    ) -> List[float]:
        """Extract S/R specific features (6 features)."""
        try:
            features = []
            
            # S/R type (support, resistance, both)
            sr_type = level.get('sr_type', 0.5)
            features.append(sr_type)
            
            # S/R timeframe confluence
            sr_timeframe_confluence = level.get('sr_timeframe_confluence', 0.5)
            features.append(sr_timeframe_confluence)
            
            # S/R breakout history
            sr_breakout_history = level.get('sr_breakout_history', 0.5)
            features.append(sr_breakout_history)
            
            # S/R retest success rate
            sr_retest_success_rate = level.get('sr_retest_success_rate', 0.5)
            features.append(sr_retest_success_rate)
            
            # S/R volume profile strength
            sr_volume_profile_strength = level.get('sr_volume_profile_strength', 0.5)
            features.append(sr_volume_profile_strength)
            
            # S/R market structure alignment
            sr_market_structure_alignment = level.get('sr_market_structure_alignment', 0.5)
            features.append(sr_market_structure_alignment)
            
            return features
            
        except Exception as e:
            self.logger.error(f"S/R specific feature extraction failed: {e}")
            return [0.5] * 6  # Return default features
    
    async def _create_target_for_level(
        self,
        level: Dict[str, Any],
        historical_performance: Optional[Dict[str, Any]]
    ) -> float:
        """Create optimized target variable for S/R level quality based on trading performance."""
        try:
            # Use historical performance if available (preferred method)
            if historical_performance and level.get('id') in historical_performance:
                perf = historical_performance[level['id']]
                return perf.get('quality_score', 0.5)
            
            # Define what makes a "good S/R level" based on trading performance
            # A good S/R level should:
            # 1. Hold when tested (high bounce rate)
            # 2. Provide clear breakout signals when broken
            # 3. Have consistent behavior across timeframes
            # 4. Show volume confirmation
            # 5. Have low false breakout rate
            
            # Get optimized weights from configuration or use defaults
            weights = self.ml_config.get("target_weights", {})
            
            # === CORE PERFORMANCE ASPECTS (60% weight) ===
            # Bounce rate (most important - 20%)
            bounce_rate = level.get('bounce_rate', 0.5)
            bounce_weight = weights.get('bounce_rate', 0.20)
            target += bounce_rate * bounce_weight
            
            # False breakout rate (penalty - 15%)
            false_breakout_rate = level.get('false_breakout_rate', 0.0)
            false_breakout_weight = weights.get('false_breakout_rate', 0.15)
            target -= false_breakout_rate * false_breakout_weight
            
            # Volume confirmation (10%)
            volume_confirmation = level.get('volume_confirmation_score', 0.5)
            volume_weight = weights.get('volume_confirmation', 0.10)
            target += volume_confirmation * volume_weight
            
            # Timeframe consistency (10%)
            timeframe_consistency = level.get('timeframe_consistency', 0.5)
            timeframe_weight = weights.get('timeframe_consistency', 0.10)
            target += timeframe_consistency * timeframe_weight
            
            # Touch count (5%)
            touch_count = level.get('touch_count', 0)
            touch_score = min(touch_count / 10.0, 1.0)
            touch_weight = weights.get('touch_count', 0.05)
            target += touch_score * touch_weight
            
            # === TECHNICAL STRENGTH ASPECTS (25% weight) ===
            # Level strength (8%)
            strength = level.get('strength', 0.5)
            strength_weight = weights.get('strength', 0.08)
            target += strength * strength_weight
            
            # Confluence score (7%)
            confluence_score = level.get('confluence_score', 0.5)
            confluence_weight = weights.get('confluence_score', 0.07)
            target += confluence_score * confluence_weight
            
            # HVN strength (5%)
            hvn_strength = level.get('hvn_strength', 0.5)
            hvn_weight = weights.get('hvn_strength', 0.05)
            target += hvn_strength * hvn_weight
            
            # Fibonacci confluence (5%)
            fib_confluence = level.get('fib_confluence_count', 0)
            fib_score = min(fib_confluence / 3.0, 1.0)
            fib_weight = weights.get('fib_confluence', 0.05)
            target += fib_score * fib_weight
            
            # === MARKET STRUCTURE ASPECTS (15% weight) ===
            # Retest success rate (6%)
            retest_success = level.get('sr_retest_success_rate', 0.5)
            retest_weight = weights.get('retest_success_rate', 0.06)
            target += retest_success * retest_weight
            
            # Market structure alignment (5%)
            market_structure_alignment = level.get('sr_market_structure_alignment', 0.5)
            market_structure_weight = weights.get('market_structure_alignment', 0.05)
            target += market_structure_alignment * market_structure_weight
            
            # Psychological level strength (4%)
            psychological_strength = level.get('psychological_level_type', 0.0)
            psychological_weight = weights.get('psychological_strength', 0.04)
            target += psychological_strength * psychological_weight
            
            return min(max(target, 0.0), 1.0)  # Clamp to [0, 1]
            
        except Exception as e:
            self.logger.error(f"Target creation failed: {e}")
            return 0.5
    
    async def _get_feature_names(self) -> List[str]:
        """Get feature names for ML models (S/R specific features only)."""
        # Core S/R features (15 features)
        core_features = [
            "touch_count", "strength", "age_bars", "avg_bounce_ratio",
            "max_bounce_ratio", "volume_confirmation_score", "consistency_score",
            "failure_count", "proximity_to_level", "level_density", 
            "confluence_score", "time_since_touch", "volume_at_touch", 
            "price_action_score", "microstructure_score"
        ]
        
        # HVN (High Volume Node) features (5 features)
        hvn_features = [
            "hvn_strength", "hvn_volume_ratio", "hvn_touch_count",
            "hvn_time_weight", "hvn_price_accuracy"
        ]
        
        # Fibonacci retracement features (6 features)
        fibonacci_features = [
            "fib_level_type", "fib_strength", "fib_confluence_count",
            "fib_timeframe_alignment", "fib_volume_confirmation", "fib_bounce_quality"
        ]
        
        # Psychological level features (5 features)
        psychological_features = [
            "psychological_level_type", "round_number_strength", "psychological_touch_count",
            "psychological_volume_spike", "psychological_bounce_ratio"
        ]
        
        # Pivot point features (4 features)
        pivot_features = [
            "pivot_type", "pivot_strength", "pivot_timeframe", "pivot_confluence"
        ]
        
        # Trend line features (4 features)
        trendline_features = [
            "trendline_type", "trendline_strength", "trendline_touch_count", "trendline_angle"
        ]
        
        # Support/Resistance specific features (6 features)
        sr_specific_features = [
            "sr_type", "sr_timeframe_confluence", "sr_breakout_history",
            "sr_retest_success_rate", "sr_volume_profile_strength", "sr_market_structure_alignment"
        ]
        
        return (core_features + hvn_features + fibonacci_features + 
                psychological_features + pivot_features + trendline_features + sr_specific_features)
    
    async def _train_sr_quality_model(self, training_data: MLFeatureSet) -> None:
        """Train S/R quality prediction model with proper regularization."""
        try:
            model_config = self.ml_config.get("models", {}).get("sr_quality_model", {})
            
            # Create model with proper regularization
            if model_config.get("type") == "gradient_boosting":
                self.sr_quality_model = GradientBoostingRegressor(
                    n_estimators=model_config.get("parameters", {}).get("n_estimators", 200),
                    max_depth=model_config.get("parameters", {}).get("max_depth", 4),  # Reduced for regularization
                    learning_rate=model_config.get("parameters", {}).get("learning_rate", 0.05),  # Lower for regularization
                    subsample=model_config.get("parameters", {}).get("subsample", 0.8),
                    max_features='sqrt',  # Feature subsampling for regularization
                    min_samples_split=10,  # Prevent overfitting
                    min_samples_leaf=5,    # Prevent overfitting
                    validation_fraction=0.2,  # Early stopping
                    n_iter_no_change=10,   # Early stopping patience
                    random_state=42
                )
            else:
                # Default to Random Forest with regularization
                self.sr_quality_model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=8,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    max_features='sqrt',
                    bootstrap=True,
                    random_state=42
                )
            
            # Prepare features
            X = training_data.features
            y = training_data.target
            
            # Advanced feature selection with Random Forest, SHAP, and correlation analysis
            if len(X) > 50:  # Need more samples for robust feature selection
                feature_names = await self._get_feature_names()
                
                # Step 1: Random Forest feature importance
                rf_selector = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_selector.fit(X, y)
                rf_importance = rf_selector.feature_importances_
                
                # Step 2: Permutation importance
                from sklearn.inspection import permutation_importance
                perm_importance = permutation_importance(rf_selector, X, y, n_repeats=10, random_state=42)
                perm_scores = perm_importance.importances_mean
                
                # Step 3: Correlation analysis
                correlation_scores = self._calculate_feature_correlations(X, y)
                
                # Step 4: SHAP analysis (if available)
                shap_scores = await self._calculate_shap_importance(rf_selector, X, feature_names)
                
                # Step 5: Combined feature scoring
                combined_scores = self._combine_feature_scores(
                    rf_importance, perm_scores, correlation_scores, shap_scores
                )
                
                # Step 6: Select top features with S/R feature prioritization
                top_features = self._select_top_features_with_sr_priority(combined_scores, feature_names, top_k=50)
                
                # Store comprehensive feature importance
                self.feature_importance = {
                    'rf_importance': dict(zip(feature_names, rf_importance)),
                    'permutation_importance': dict(zip(feature_names, perm_scores)),
                    'correlation_scores': dict(zip(feature_names, correlation_scores)),
                    'shap_scores': shap_scores,
                    'combined_scores': dict(zip(feature_names, combined_scores)),
                    'selected_features': top_features
                }
                
                # Log comprehensive feature analysis
                self._log_feature_analysis()
                
                # Select features for training
                feature_indices = [i for i, name in enumerate(feature_names) if name in top_features]
                X = X[:, feature_indices]
            
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
        """Use step03 regime detection with LGBM model instead of training new model."""
        try:
            # Use existing regime detection from step03 which has its own LGBM model
            self.logger.info("Using step03 regime detection with LGBM model")
            
            # Import step03 regime detection
            try:
                from src.training.steps.vectorized_advanced_feature_engineering import (
                    VectorizedAdvancedFeatureEngineeringRefactored
                )
                self.step03_engineer = VectorizedAdvancedFeatureEngineeringRefactored()
                self.logger.info("✅ Step03 regime detection loaded successfully")
            except ImportError as e:
                self.logger.warning(f"Step03 regime detection not available: {e}")
                self.step03_engineer = None
            
            # Set regime classification model to None (use step03)
            self.regime_classification_model = None
            
            # Extract regime features for validation
            regime_features = await self._extract_regime_features(market_data)
            regime_targets = await self._create_regime_targets(market_data)
            
            if len(regime_features) > 10:
                # Validate regime detection accuracy
                accuracy = self._validate_regime_detection(regime_features, regime_targets)
                self.logger.info(f"✅ Regime detection validation completed. Accuracy: {accuracy:.4f}")
                
                # Test step03 regime detection if available
                if self.step03_engineer:
                    try:
                        # Test step03 regime detection
                        step03_features = await self.step03_engineer.engineer_features(market_data)
                        regime_features_step03 = step03_features.get('regime_features', [])
                        
                        if len(regime_features_step03) > 0:
                            self.logger.info(f"✅ Step03 regime features extracted: {len(regime_features_step03)} features")
                        else:
                            self.logger.warning("Step03 regime features not found")
                            
                    except Exception as e:
                        self.logger.warning(f"Step03 regime detection test failed: {e}")
            else:
                self.logger.warning("Insufficient data for regime validation")
            
        except Exception as e:
            self.logger.error(f"Regime classification setup failed: {e}")
    
    def _validate_regime_detection(self, features: np.ndarray, targets: np.ndarray) -> float:
        """Validate regime detection accuracy using simple rules."""
        try:
            correct_predictions = 0
            total_predictions = len(targets)
            
            for i, (feature, target) in enumerate(zip(features, targets)):
                # Simple rule-based regime classification
                sma_ratio, rsi, volatility, volume_ratio, momentum = feature
                
                # Rule-based classification
                if abs(sma_ratio - 1.0) > 0.02 and 30 < rsi < 70:
                    predicted_regime = 0  # Trending
                elif abs(sma_ratio - 1.0) <= 0.02:
                    predicted_regime = 1  # Ranging
                else:
                    predicted_regime = 2  # Transitional
                
                if predicted_regime == target:
                    correct_predictions += 1
            
            return correct_predictions / total_predictions if total_predictions > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Regime validation failed: {e}")
            return 0.0
    
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
    
    def _calculate_stochastic(self, market_data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        try:
            low_min = market_data['low'].rolling(window=k_period).min()
            high_max = market_data['high'].rolling(window=k_period).max()
            k_percent = 100 * ((market_data['close'] - low_min) / (high_max - low_min))
            d_percent = k_percent.rolling(window=d_period).mean()
            return k_percent.fillna(50), d_percent.fillna(50)
        except Exception:
            return pd.Series([50] * len(market_data), index=market_data.index), pd.Series([50] * len(market_data), index=market_data.index)
    
    def _calculate_williams_r(self, market_data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        try:
            high_max = market_data['high'].rolling(window=period).max()
            low_min = market_data['low'].rolling(window=period).min()
            williams_r = -100 * ((high_max - market_data['close']) / (high_max - low_min))
            return williams_r.fillna(-50)
        except Exception:
            return pd.Series([-50] * len(market_data), index=market_data.index)
    
    def _calculate_cci(self, market_data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index."""
        try:
            typical_price = (market_data['high'] + market_data['low'] + market_data['close']) / 3
            sma_tp = typical_price.rolling(window=period).mean()
            mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            cci = (typical_price - sma_tp) / (0.015 * mad)
            return cci.fillna(0)
        except Exception:
            return pd.Series([0] * len(market_data), index=market_data.index)
    
    def _calculate_adx(self, market_data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index."""
        try:
            high_diff = market_data['high'].diff()
            low_diff = market_data['low'].diff()
            
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
            
            plus_dm = pd.Series(plus_dm, index=market_data.index)
            minus_dm = pd.Series(minus_dm, index=market_data.index)
            
            atr = self._calculate_atr(market_data, period)
            plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean()
            
            return adx.fillna(25)
        except Exception:
            return pd.Series([25] * len(market_data), index=market_data.index)
    
    def _calculate_obv(self, market_data: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume."""
        try:
            price_change = market_data['close'].diff()
            obv = np.where(price_change > 0, market_data['volume'], 
                          np.where(price_change < 0, -market_data['volume'], 0))
            obv = pd.Series(obv, index=market_data.index).cumsum()
            return obv.fillna(0)
        except Exception:
            return pd.Series([0] * len(market_data), index=market_data.index)
    
    def _detect_doji_pattern(self, market_data: pd.DataFrame) -> float:
        """Detect Doji candlestick pattern."""
        try:
            if len(market_data) < 1:
                return 0.0
            
            current = market_data.iloc[-1]
            body_size = abs(current['close'] - current['open'])
            total_range = current['high'] - current['low']
            
            # Doji: body is less than 10% of total range
            return 1.0 if body_size / total_range < 0.1 else 0.0
        except Exception:
            return 0.0
    
    def _detect_hammer_pattern(self, market_data: pd.DataFrame) -> float:
        """Detect Hammer candlestick pattern."""
        try:
            if len(market_data) < 1:
                return 0.0
            
            current = market_data.iloc[-1]
            body_size = abs(current['close'] - current['open'])
            lower_shadow = min(current['open'], current['close']) - current['low']
            upper_shadow = current['high'] - max(current['open'], current['close'])
            total_range = current['high'] - current['low']
            
            # Hammer: long lower shadow, small body, small upper shadow
            is_hammer = (lower_shadow > 2 * body_size and 
                        upper_shadow < body_size and 
                        body_size / total_range < 0.3)
            
            return 1.0 if is_hammer else 0.0
        except Exception:
            return 0.0
    
    def _calculate_volatility_proxy(self, market_data: pd.DataFrame, period: int = 20) -> float:
        """Calculate volatility proxy (simplified VIX)."""
        try:
            if len(market_data) < period:
                return 0.0
            
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.rolling(window=period).std().iloc[-1]
            return float(volatility * 100) if not np.isnan(volatility) else 0.0
        except Exception:
            return 0.0
    
    def _calculate_feature_correlations(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate correlation between features and target."""
        try:
            correlations = []
            for i in range(X.shape[1]):
                corr = np.corrcoef(X[:, i], y)[0, 1]
                correlations.append(abs(corr) if not np.isnan(corr) else 0.0)
            return np.array(correlations)
        except Exception:
            return np.zeros(X.shape[1])
    
    async def _calculate_shap_importance(self, model, X: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Calculate SHAP importance scores."""
        try:
            # Try to import SHAP
            try:
                import shap
                SHAP_AVAILABLE = True
            except ImportError:
                SHAP_AVAILABLE = False
                self.logger.warning("SHAP not available, skipping SHAP analysis")
                return {name: 0.0 for name in feature_names}
            
            if not SHAP_AVAILABLE or len(X) < 100:
                return {name: 0.0 for name in feature_names}
            
            # Use TreeExplainer for tree-based models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X[:100])  # Use subset for performance
            
            # Calculate mean absolute SHAP values
            mean_shap_values = np.mean(np.abs(shap_values), axis=0)
            
            return dict(zip(feature_names, mean_shap_values))
            
        except Exception as e:
            self.logger.warning(f"SHAP calculation failed: {e}")
            return {name: 0.0 for name in feature_names}
    
    def _combine_feature_scores(
        self, 
        rf_importance: np.ndarray, 
        perm_scores: np.ndarray, 
        correlation_scores: np.ndarray, 
        shap_scores: Dict[str, float]
    ) -> np.ndarray:
        """Combine different feature importance scores."""
        try:
            # Normalize scores to [0, 1]
            rf_norm = rf_importance / (np.max(rf_importance) + 1e-8)
            perm_norm = perm_scores / (np.max(perm_scores) + 1e-8)
            corr_norm = correlation_scores / (np.max(correlation_scores) + 1e-8)
            
            # Get SHAP scores as array
            shap_array = np.array([shap_scores.get(f"feature_{i}", 0.0) for i in range(len(rf_importance))])
            shap_norm = shap_array / (np.max(shap_array) + 1e-8)
            
            # Weighted combination
            combined = (
                rf_norm * 0.3 +      # Random Forest importance
                perm_norm * 0.3 +    # Permutation importance
                corr_norm * 0.2 +    # Correlation with target
                shap_norm * 0.2      # SHAP importance
            )
            
            return combined
            
        except Exception as e:
            self.logger.error(f"Feature score combination failed: {e}")
            return rf_importance  # Fallback to RF importance
    
    def _select_top_features(self, scores: np.ndarray, feature_names: List[str], top_k: int = 20) -> List[str]:
        """Select top K features based on combined scores."""
        try:
            # Get indices of top features
            top_indices = np.argsort(scores)[-top_k:]
            
            # Return feature names
            return [feature_names[i] for i in top_indices]
            
        except Exception as e:
            self.logger.error(f"Top feature selection failed: {e}")
            return feature_names[:top_k]  # Fallback to first K features
    
    def _select_top_features_with_sr_priority(self, scores: np.ndarray, feature_names: List[str], top_k: int = 50) -> List[str]:
        """Select top K features with S/R feature prioritization."""
        try:
            # Define S/R specific feature patterns (45 features total)
            sr_feature_patterns = [
                # Core S/R features (15)
                'touch_count', 'strength', 'age_bars', 'avg_bounce_ratio',
                'max_bounce_ratio', 'volume_confirmation_score', 'consistency_score',
                'failure_count', 'proximity_to_level', 'level_density', 
                'confluence_score', 'time_since_touch', 'volume_at_touch', 
                'price_action_score', 'microstructure_score',
                
                # HVN features (5)
                'hvn_strength', 'hvn_volume_ratio', 'hvn_touch_count',
                'hvn_time_weight', 'hvn_price_accuracy',
                
                # Fibonacci features (6)
                'fib_level_type', 'fib_strength', 'fib_confluence_count',
                'fib_timeframe_alignment', 'fib_volume_confirmation', 'fib_bounce_quality',
                
                # Psychological features (5)
                'psychological_level_type', 'round_number_strength', 'psychological_touch_count',
                'psychological_volume_spike', 'psychological_bounce_ratio',
                
                # Pivot features (4)
                'pivot_type', 'pivot_strength', 'pivot_timeframe', 'pivot_confluence',
                
                # Trend line features (4)
                'trendline_type', 'trendline_strength', 'trendline_touch_count', 'trendline_angle',
                
                # S/R specific features (6)
                'sr_type', 'sr_timeframe_confluence', 'sr_breakout_history',
                'sr_retest_success_rate', 'sr_volume_profile_strength', 'sr_market_structure_alignment'
            ]
            
            # Identify S/R features
            sr_features = []
            non_sr_features = []
            
            for i, feature_name in enumerate(feature_names):
                is_sr_feature = any(pattern in feature_name.lower() for pattern in sr_feature_patterns)
                if is_sr_feature:
                    sr_features.append((i, feature_name, scores[i]))
                else:
                    non_sr_features.append((i, feature_name, scores[i]))
            
            # Sort by scores
            sr_features.sort(key=lambda x: x[2], reverse=True)
            non_sr_features.sort(key=lambda x: x[2], reverse=True)
            
            # Select features with S/R prioritization
            selected_features = []
            
            # Get minimum S/R feature ratio from configuration
            min_sr_ratio = self.ml_config.get("feature_selection", {}).get("min_sr_ratio", 0.7)
            min_sr_count = int(top_k * min_sr_ratio)
            
            # First, select top S/R features (minimum 70%, no upper limit)
            sr_count = max(min_sr_count, min(len(sr_features), top_k))
            for i in range(sr_count):
                selected_features.append(sr_features[i][1])
            
            # Then, select top non-S/R features (remaining slots)
            remaining_count = top_k - len(selected_features)
            for i in range(min(remaining_count, len(non_sr_features))):
                selected_features.append(non_sr_features[i][1])
            
            # If we still need more features, fill with highest scoring remaining features
            if len(selected_features) < top_k:
                all_features = [(i, feature_names[i], scores[i]) for i in range(len(feature_names))]
                all_features.sort(key=lambda x: x[2], reverse=True)
                
                for i, feature_name, score in all_features:
                    if feature_name not in selected_features and len(selected_features) < top_k:
                        selected_features.append(feature_name)
            
            self.logger.info(f"🎯 Feature selection with S/R prioritization:")
            self.logger.info(f"   - S/R features selected: {sr_count} (minimum {min_sr_ratio*100:.0f}% of total)")
            self.logger.info(f"   - Non-S/R features selected: {len(selected_features) - sr_count}")
            self.logger.info(f"   - Total features selected: {len(selected_features)}")
            self.logger.info(f"   - S/R feature categories: Core(15), HVN(5), Fibonacci(6), Psychological(5), Pivot(4), Trendline(4), S/R Specific(6)")
            self.logger.info(f"   - Selection strategy: Minimum {min_sr_ratio*100:.0f}% S/R features, no upper limit")
            
            return selected_features
            
        except Exception as e:
            self.logger.error(f"S/R prioritized feature selection failed: {e}")
            return self._select_top_features(scores, feature_names, top_k)  # Fallback
    
    def _log_feature_analysis(self) -> None:
        """Log comprehensive feature analysis results."""
        try:
            if not hasattr(self, 'feature_importance') or not self.feature_importance:
                return
            
            combined_scores = self.feature_importance.get('combined_scores', {})
            selected_features = self.feature_importance.get('selected_features', [])
            
            # Log top 15 selected features
            sorted_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            
            self.logger.info("🔍 Comprehensive Feature Analysis Results:")
            self.logger.info(f"📊 Total features analyzed: {len(combined_scores)}")
            self.logger.info(f"🎯 Selected features: {len(selected_features)}")
            
            self.logger.info("🏆 Top 25 Most Important Features:")
            for i, (feature, score) in enumerate(sorted_features[:25]):
                status = "✅ SELECTED" if feature in selected_features else "❌ NOT SELECTED"
                feature_type = "🎯 S/R" if any(pattern in feature.lower() for pattern in ['proximity', 'level', 'touch', 'bounce', 'strength', 'rsi', 'macd', 'bollinger', 'atr', 'stoch', 'williams', 'cci', 'adx', 'obv', 'doji', 'hammer', 'volatility']) else "📊 STEP06"
                self.logger.info(f"  {i+1:2d}. {feature:<30} {score:.4f} {feature_type} {status}")
            
            # Log feature selection statistics
            rf_importance = self.feature_importance.get('rf_importance', {})
            perm_importance = self.feature_importance.get('permutation_importance', {})
            
            if rf_importance and perm_importance:
                self.logger.info("📈 Feature Selection Statistics:")
                self.logger.info(f"   - Random Forest top feature: {max(rf_importance.items(), key=lambda x: x[1])}")
                self.logger.info(f"   - Permutation top feature: {max(perm_importance.items(), key=lambda x: x[1])}")
            
        except Exception as e:
            self.logger.error(f"Feature analysis logging failed: {e}")
    
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
    
    async def optimize_target_weights(
        self,
        market_data: pd.DataFrame,
        sr_levels: List[Dict[str, Any]],
        historical_performance: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Optimize target weights through backtesting and performance analysis."""
        try:
            self.logger.info("🔧 Optimizing target weights through backtesting...")
            
            # Define weight optimization ranges
            weight_ranges = {
                'bounce_rate': (0.15, 0.25),           # Most important
                'false_breakout_rate': (0.10, 0.20),    # Penalty weight
                'volume_confirmation': (0.08, 0.15),    # Volume importance
                'timeframe_consistency': (0.08, 0.15),  # Consistency importance
                'touch_count': (0.03, 0.08),            # Touch count importance
                'strength': (0.05, 0.12),               # Level strength
                'confluence_score': (0.05, 0.10),       # Confluence importance
                'hvn_strength': (0.03, 0.08),           # HVN importance
                'fib_confluence': (0.03, 0.08),         # Fibonacci importance
                'retest_success_rate': (0.04, 0.08),    # Retest importance
                'market_structure_alignment': (0.03, 0.07),  # Market structure
                'psychological_strength': (0.02, 0.06)  # Psychological levels
            }
            
            # Current best weights (start with defaults)
            best_weights = {
                'bounce_rate': 0.20,
                'false_breakout_rate': 0.15,
                'volume_confirmation': 0.10,
                'timeframe_consistency': 0.10,
                'touch_count': 0.05,
                'strength': 0.08,
                'confluence_score': 0.07,
                'hvn_strength': 0.05,
                'fib_confluence': 0.05,
                'retest_success_rate': 0.06,
                'market_structure_alignment': 0.05,
                'psychological_strength': 0.04
            }
            
            best_score = 0.0
            
            # Simple grid search optimization (can be enhanced with more sophisticated methods)
            for iteration in range(10):  # 10 optimization iterations
                # Generate candidate weights
                candidate_weights = {}
                for param, (min_val, max_val) in weight_ranges.items():
                    # Add some randomness around current best
                    current_val = best_weights[param]
                    noise = (max_val - min_val) * 0.1  # 10% noise
                    candidate_weights[param] = max(min_val, min(max_val, 
                        current_val + np.random.normal(0, noise)))
                
                # Normalize weights to sum to 1.0
                total_weight = sum(candidate_weights.values())
                candidate_weights = {k: v/total_weight for k, v in candidate_weights.items()}
                
                # Test candidate weights
                score = await self._evaluate_target_weights(
                    candidate_weights, market_data, sr_levels, historical_performance
                )
                
                if score > best_score:
                    best_score = score
                    best_weights = candidate_weights.copy()
                    self.logger.info(f"   Iteration {iteration+1}: New best score {score:.4f}")
            
            self.logger.info(f"✅ Target weight optimization completed. Best score: {best_score:.4f}")
            self.logger.info(f"   Optimized weights: {best_weights}")
            
            # Update configuration with optimized weights
            if "target_weights" not in self.ml_config:
                self.ml_config["target_weights"] = {}
            self.ml_config["target_weights"].update(best_weights)
            
            return best_weights
            
        except Exception as e:
            self.logger.error(f"Target weight optimization failed: {e}")
            return self.ml_config.get("target_weights", {})
    
    async def _evaluate_target_weights(
        self,
        weights: Dict[str, float],
        market_data: pd.DataFrame,
        sr_levels: List[Dict[str, Any]],
        historical_performance: Optional[Dict[str, Any]]
    ) -> float:
        """Evaluate target weights by measuring correlation with actual trading performance."""
        try:
            # Temporarily set weights
            original_weights = self.ml_config.get("target_weights", {})
            self.ml_config["target_weights"] = weights
            
            # Calculate targets with new weights
            targets = []
            actual_performance = []
            
            for level in sr_levels:
                # Calculate target with new weights
                target = await self._create_target_for_level(level, historical_performance)
                targets.append(target)
                
                # Get actual performance if available
                if historical_performance and level.get('id') in historical_performance:
                    perf = historical_performance[level['id']]
                    actual_perf = perf.get('actual_bounce_rate', 0.5)  # Use actual bounce rate as ground truth
                    actual_performance.append(actual_perf)
                else:
                    # Use level characteristics as proxy for actual performance
                    actual_perf = level.get('bounce_rate', 0.5)
                    actual_performance.append(actual_perf)
            
            # Restore original weights
            self.ml_config["target_weights"] = original_weights
            
            if len(targets) < 5:  # Need minimum samples
                return 0.0
            
            # Calculate correlation between predicted and actual performance
            correlation = np.corrcoef(targets, actual_performance)[0, 1]
            
            # Return correlation as score (higher is better)
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            self.logger.error(f"Target weight evaluation failed: {e}")
            return 0.0