"""
SR Breakout Predictor - Specialized ML model for predicting breakout/bounce probabilities
when price approaches support/resistance zones.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Dict, Any, Optional, Tuple
import joblib
import os
from datetime import datetime

from src.utils.logger import setup_logging
from src.utils.error_handler import handle_errors, handle_data_processing_errors
from src.analyst.sr_analyzer import SRLevelAnalyzer


class SRBreakoutPredictor:
    """
    Specialized ML model for predicting breakout/bounce probabilities when price approaches SR zones.
    
    Architecture:
    1. SR Analyzer identifies zones (traditional analysis)
    2. This model predicts breakout/bounce odds when near zones
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the SR Breakout Predictor."""
        self.config = config
        self.logger = setup_logging("SRBreakoutPredictor")
        
        # Model configuration
        self.model_config = config.get("sr_breakout_predictor", {})
        self.model_path = self.model_config.get("model_path", "models/sr_breakout_model.pkl")
        self.feature_importance_path = self.model_config.get("feature_importance_path", "models/sr_breakout_feature_importance.json")
        
        # Training configuration
        self.lookback_days = self.model_config.get("lookback_days", 730)  # 2 years
        self.min_samples = self.model_config.get("min_samples", 1000)
        self.test_size = self.model_config.get("test_size", 0.2)
        self.random_state = self.model_config.get("random_state", 42)
        
        # SR Analyzer for zone detection
        self.sr_analyzer = SRLevelAnalyzer(config.get("analyst", {}).get("sr_analyzer", {}))
        
        # Model state
        self.model = None
        self.feature_names = []
        self.is_trained = False
        
    @handle_errors(exceptions=(Exception,), default_return=False, context="SR breakout predictor initialization")
    async def initialize(self) -> bool:
        """Initialize the SR Breakout Predictor."""
        try:
            self.logger.info("Initializing SR Breakout Predictor...")
            
            # Initialize SR Analyzer
            await self.sr_analyzer.initialize()
            
            # Try to load existing model
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.is_trained = True
                self.logger.info("✅ Loaded existing SR breakout model")
            else:
                self.logger.info("No existing model found. Will train new model.")
                
            self.logger.info("✅ SR Breakout Predictor initialization completed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ SR Breakout Predictor initialization failed: {e}")
            return False
    
    @handle_data_processing_errors(default_return=None, context="prepare breakout training data")
    def _prepare_breakout_training_data(self, df: pd.DataFrame, sr_levels: list) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
        """
        Prepare training data for breakout/bounce prediction.
        
        Args:
            df: OHLCV data with technical indicators
            sr_levels: List of SR levels from SR Analyzer
            
        Returns:
            Tuple of (features, targets) or None if insufficient data
        """
        try:
            if df.empty or not sr_levels:
                self.logger.warning("Insufficient data for breakout training")
                return None
                
            # Create features for each timestamp
            features_list = []
            targets_list = []
            
            for i in range(1, len(df)):  # Start from 1 to have previous data
                current_row = df.iloc[i]
                previous_row = df.iloc[i-1]
                
                # Check if we're near any SR level
                current_price = current_row['close']
                near_sr_zone = False
                sr_context = {}
                
                for level in sr_levels:
                    distance = abs(current_price - level.get('price', 0)) / current_price
                    if distance < 0.02:  # Within 2% of SR level
                        near_sr_zone = True
                        sr_context = {
                            'level_price': level.get('price', 0),
                            'level_type': level.get('type', 'unknown'),
                            'level_strength': level.get('strength', 0.0),
                            'distance_to_level': distance
                        }
                        break
                
                # Only create training samples when near SR zones
                if near_sr_zone:
                    # Create features
                    features = self._extract_breakout_features(df, i, sr_context)
                    features_list.append(features)
                    
                    # Create target (breakout vs bounce)
                    target = self._determine_breakout_target(df, i, sr_context)
                    targets_list.append(target)
            
            if len(features_list) < self.min_samples:
                self.logger.warning(f"Insufficient samples for training: {len(features_list)} < {self.min_samples}")
                return None
                
            # Convert to DataFrame
            features_df = pd.DataFrame(features_list)
            targets_series = pd.Series(targets_list)
            
            self.logger.info(f"Prepared {len(features_df)} training samples for breakout prediction")
            return features_df, targets_series
            
        except Exception as e:
            self.logger.error(f"Error preparing breakout training data: {e}")
            return None
    
    def _extract_breakout_features(self, df: pd.DataFrame, idx: int, sr_context: dict) -> dict:
        """Extract features for breakout/bounce prediction."""
        features = {}
        
        # Current price context
        current_price = df.iloc[idx]['close']
        features['price'] = current_price
        features['price_change'] = (current_price - df.iloc[idx-1]['close']) / df.iloc[idx-1]['close']
        
        # SR context features
        features['distance_to_sr'] = sr_context.get('distance_to_level', 1.0)
        features['sr_strength'] = sr_context.get('level_strength', 0.0)
        features['sr_type'] = 1 if sr_context.get('level_type') == 'resistance' else 0
        
        # Technical indicators (if available)
        for indicator in ['RSI', 'MACD', 'ATR', 'ADX', 'volume']:
            if indicator in df.columns:
                features[f'{indicator.lower()}'] = df.iloc[idx][indicator]
            else:
                features[f'{indicator.lower()}'] = 0.0
        
        # Price momentum features
        if idx >= 5:
            price_momentum_5 = (current_price - df.iloc[idx-5]['close']) / df.iloc[idx-5]['close']
            features['momentum_5'] = price_momentum_5
        else:
            features['momentum_5'] = 0.0
            
        if idx >= 10:
            price_momentum_10 = (current_price - df.iloc[idx-10]['close']) / df.iloc[idx-10]['close']
            features['momentum_10'] = price_momentum_10
        else:
            features['momentum_10'] = 0.0
        
        # Volume features
        if 'volume' in df.columns:
            current_volume = df.iloc[idx]['volume']
            avg_volume = df.iloc[max(0, idx-20):idx]['volume'].mean()
            features['volume_ratio'] = current_volume / avg_volume if avg_volume > 0 else 1.0
        else:
            features['volume_ratio'] = 1.0
        
        # Volatility features
        if idx >= 20:
            price_volatility = df.iloc[idx-20:idx]['close'].std() / df.iloc[idx-20:idx]['close'].mean()
            features['volatility'] = price_volatility
        else:
            features['volatility'] = 0.0
        
        # Market structure features
        high_20 = df.iloc[max(0, idx-20):idx]['high'].max()
        low_20 = df.iloc[max(0, idx-20):idx]['low'].min()
        features['price_position'] = (current_price - low_20) / (high_20 - low_20) if high_20 > low_20 else 0.5
        
        return features
    
    def _determine_breakout_target(self, df: pd.DataFrame, idx: int, sr_context: dict) -> int:
        """
        Determine if a breakout occurred (1) or bounce (0).
        
        Args:
            df: OHLCV data
            idx: Current index
            sr_context: SR level context
            
        Returns:
            1 for breakout, 0 for bounce
        """
        try:
            # Look ahead 5 periods to determine outcome
            look_ahead = min(5, len(df) - idx - 1)
            if look_ahead < 2:
                return 0  # Default to bounce if insufficient future data
            
            current_price = df.iloc[idx]['close']
            sr_price = sr_context.get('level_price', current_price)
            sr_type = sr_context.get('level_type', 'unknown')
            
            # Check future price movement
            future_prices = df.iloc[idx+1:idx+look_ahead+1]['close']
            max_future_price = future_prices.max()
            min_future_price = future_prices.min()
            
            if sr_type == 'resistance':
                # Breakout: price goes significantly above resistance
                breakout_threshold = sr_price * 1.005  # 0.5% above resistance
                if max_future_price > breakout_threshold:
                    return 1  # Breakout
                else:
                    return 0  # Bounce
                    
            elif sr_type == 'support':
                # Breakout: price goes significantly below support
                breakout_threshold = sr_price * 0.995  # 0.5% below support
                if min_future_price < breakout_threshold:
                    return 1  # Breakout
                else:
                    return 0  # Bounce
                    
            else:
                return 0  # Default to bounce
                
        except Exception as e:
            self.logger.warning(f"Error determining breakout target: {e}")
            return 0
    
    @handle_errors(exceptions=(Exception,), default_return=False, context="train breakout predictor")
    async def train(self, df: pd.DataFrame) -> bool:
        """
        Train the SR breakout predictor.
        
        Args:
            df: OHLCV data with technical indicators
            
        Returns:
            bool: True if training successful
        """
        try:
            self.logger.info("Training SR Breakout Predictor...")
            
            # Analyze SR levels
            sr_levels = await self.sr_analyzer.analyze(df)
            
            # Prepare training data
            training_data = self._prepare_breakout_training_data(df, sr_levels)
            if training_data is None:
                self.logger.error("Failed to prepare training data")
                return False
                
            features_df, targets_series = training_data
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                features_df, targets_series, 
                test_size=self.test_size, 
                random_state=self.random_state,
                stratify=targets_series
            )
            
            # Train LightGBM model
            model_params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': self.random_state
            }
            
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
            
            self.model = lgb.train(
                model_params,
                train_data,
                valid_sets=[valid_data],
                num_boost_round=100,
                callbacks=[lgb.early_stopping(stopping_rounds=10)]
            )
            
            # Store feature names
            self.feature_names = list(features_df.columns)
            self.is_trained = True
            
            # Evaluate model
            train_preds = self.model.predict(X_train)
            test_preds = self.model.predict(X_test)
            
            from sklearn.metrics import accuracy_score, classification_report
            train_accuracy = accuracy_score(y_train, train_preds > 0.5)
            test_accuracy = accuracy_score(y_test, test_preds > 0.5)
            
            self.logger.info(f"Training accuracy: {train_accuracy:.3f}")
            self.logger.info(f"Test accuracy: {test_accuracy:.3f}")
            self.logger.info(f"Classification report:\n{classification_report(y_test, test_preds > 0.5)}")
            
            # Save model
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.model, self.model_path)
            
            # Save feature importance
            feature_importance = dict(zip(self.feature_names, self.model.feature_importance()))
            import json
            with open(self.feature_importance_path, 'w') as f:
                json.dump(feature_importance, f, indent=2)
            
            self.logger.info("✅ SR Breakout Predictor training completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ SR Breakout Predictor training failed: {e}")
            return False
    
    @handle_errors(exceptions=(Exception,), default_return=None, context="predict breakout probability")
    async def predict_breakout_probability(self, df: pd.DataFrame, current_price: float) -> Optional[Dict[str, Any]]:
        """
        Predict breakout probability for current market conditions.
        
        Args:
            df: Recent OHLCV data with technical indicators
            current_price: Current market price
            
        Returns:
            Dict with breakout probability and confidence
        """
        try:
            if not self.is_trained or self.model is None:
                self.logger.warning("Model not trained. Cannot make predictions.")
                return None
            
            # Analyze SR levels
            sr_levels = await self.sr_analyzer.analyze(df)
            
            # Find nearest SR level
            nearest_level = None
            min_distance = float('inf')
            
            for level in sr_levels:
                distance = abs(current_price - level.get('price', 0)) / current_price
                if distance < min_distance:
                    min_distance = distance
                    nearest_level = level
            
            # Only predict if near SR zone
            if min_distance > 0.02:  # Not near SR zone
                return {
                    'breakout_probability': 0.5,
                    'confidence': 0.0,
                    'near_sr_zone': False,
                    'message': 'Not near SR zone'
                }
            
            # Extract features for prediction
            features = self._extract_breakout_features(df, len(df)-1, {
                'level_price': nearest_level.get('price', 0),
                'level_type': nearest_level.get('type', 'unknown'),
                'level_strength': nearest_level.get('strength', 0.0),
                'distance_to_level': min_distance
            })
            
            # Create feature vector
            feature_vector = pd.DataFrame([features])
            
            # Ensure all required features are present
            for feature in self.feature_names:
                if feature not in feature_vector.columns:
                    feature_vector[feature] = 0.0
            
            # Reorder columns to match training data
            feature_vector = feature_vector[self.feature_names]
            
            # Make prediction
            breakout_prob = self.model.predict(feature_vector)[0]
            bounce_prob = 1 - breakout_prob
            
            # Calculate confidence based on model certainty
            confidence = abs(breakout_prob - 0.5) * 2  # Scale to 0-1
            
            return {
                'breakout_probability': breakout_prob,
                'bounce_probability': bounce_prob,
                'confidence': confidence,
                'near_sr_zone': True,
                'sr_level_type': nearest_level.get('type', 'unknown'),
                'sr_level_strength': nearest_level.get('strength', 0.0),
                'distance_to_sr': min_distance,
                'recommendation': 'BREAKOUT' if breakout_prob > 0.6 else 'BOUNCE' if bounce_prob > 0.6 else 'UNCERTAIN'
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting breakout probability: {e}")
            return None
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get training status and model information."""
        return {
            'is_trained': self.is_trained,
            'model_path': self.model_path,
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names[:10] + ['...'] if len(self.feature_names) > 10 else self.feature_names
        } 