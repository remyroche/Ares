#!/usr/bin/env python3
"""
Enhanced HMM Regime Manager

This module provides a comprehensive, non-redundant implementation of HMM-based
regime discovery and prediction. It ensures proper cluster generation and
regime change prediction with optimized performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import warnings

from src.utils.logger import system_logger
from src.utils.centralized_decorators import (
    handle_errors,
    validate_data_structure,
    monitor_feature_engineering,
    memory_efficient
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class RegimeType(Enum):
    """Market regime types."""
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    CONSOLIDATION = "consolidation"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"


@dataclass
class RegimeState:
    """Regime state information."""
    regime_id: int
    regime_type: RegimeType
    confidence: float
    duration: int
    volatility: float
    momentum: float
    volume_profile: float
    metadata: Dict[str, Any]


@dataclass
class RegimeTransition:
    """Regime transition information."""
    from_regime: int
    to_regime: int
    probability: float
    trigger_features: Dict[str, float]
    timestamp: pd.Timestamp


class EnhancedHMMRegimeManager:
    """
    Enhanced HMM Regime Manager that provides comprehensive regime discovery
    and prediction capabilities with proper cluster generation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("EnhancedHMMRegimeManager")
        
        # HMM Configuration
        self.n_states = config.get("hmm_n_states", 5)
        self.n_clusters = config.get("hmm_n_clusters", 20)
        self.sequence_length = config.get("hmm_sequence_length", 50)
        self.transition_threshold = config.get("regime_transition_threshold", 0.6)
        
        # Models
        self.hmm_model = None
        self.cluster_model = None
        self.transition_model = None
        self.scaler = None
        
        # State tracking
        self.current_regime = None
        self.regime_history: List[RegimeState] = []
        self.transition_history: List[RegimeTransition] = []
        
        # Cache
        self._feature_cache: Dict[str, np.ndarray] = {}
        self._prediction_cache: Dict[str, Dict[str, Any]] = {}
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize HMM and clustering models."""
        try:
            # Import required libraries
            try:
                from hmmlearn import hmm
                from sklearn.cluster import KMeans
                from sklearn.preprocessing import StandardScaler
                from sklearn.ensemble import RandomForestClassifier
                
                self.hmm_available = True
                self.logger.info("✅ HMM libraries available")
                
            except ImportError as e:
                self.hmm_available = False
                self.logger.warning(f"⚠️ HMM libraries not available: {e}")
                return
            
            # Initialize scaler
            self.scaler = StandardScaler()
            
            # Initialize transition model
            self.transition_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            self.logger.info("✅ Enhanced HMM Regime Manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing HMM models: {e}")
    
    @handle_errors(
        exceptions=(Exception,),
        default_return={"success": False, "error": "HMM training failed"},
        context="hmm_regime_training"
    )
    @validate_data_structure(required_columns=["close", "volume"])
    @memory_efficient
    async def train_regime_models(
        self, 
        df: pd.DataFrame,
        features: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Train HMM regime discovery models.
        
        Args:
            df: OHLCV DataFrame
            features: Optional pre-computed features
            
        Returns:
            Training results
        """
        if not self.hmm_available:
            return {"success": False, "error": "HMM libraries not available"}
        
        if df.empty or len(df) < self.sequence_length:
            return {"success": False, "error": "Insufficient data for training"}
        
        try:
            self.logger.info("🎓 Starting HMM regime model training...")
            
            # 1. Prepare features
            if features is None:
                features = self._generate_comprehensive_features(df)
            
            if features.empty:
                return {"success": False, "error": "No features generated"}
            
            # 2. Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # 3. Train HMM model
            self.logger.info(f"🧠 Training HMM with {self.n_states} states...")
            from hmmlearn import hmm
            
            self.hmm_model = hmm.GaussianHMM(
                n_components=self.n_states,
                n_iter=200,
                random_state=42,
                covariance_type="full",
                init_params="stmc",
                params="stmc"
            )
            
            self.hmm_model.fit(features_scaled)
            
            # 4. Generate HMM states
            hmm_states = self.hmm_model.predict(features_scaled)
            hmm_probs = self.hmm_model.predict_proba(features_scaled)
            
            # 5. Train clustering model
            self.logger.info(f"🎯 Training clustering model with {self.n_clusters} clusters...")
            from sklearn.cluster import KMeans
            
            # Create composite features
            composite_features = self._create_composite_features(features, hmm_states, hmm_probs)
            
            self.cluster_model = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10,
                max_iter=300
            )
            
            cluster_labels = self.cluster_model.fit_predict(composite_features)
            
            # 6. Train transition model
            self.logger.info("🔄 Training regime transition model...")
            self._train_transition_model(features_scaled, hmm_states, cluster_labels)
            
            # 7. Analyze and interpret regimes
            regime_analysis = self._analyze_regime_characteristics(
                features, hmm_states, cluster_labels
            )
            
            # 8. Generate training report
            training_report = self._generate_training_report(
                features, hmm_states, cluster_labels, regime_analysis
            )
            
            self.logger.info("✅ HMM regime model training completed successfully")
            
            return {
                "success": True,
                "hmm_model": self.hmm_model,
                "cluster_model": self.cluster_model,
                "transition_model": self.transition_model,
                "scaler": self.scaler,
                "regime_analysis": regime_analysis,
                "training_report": training_report,
                "n_states": self.n_states,
                "n_clusters": self.n_clusters
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error in HMM training: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_comprehensive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive features for HMM analysis."""
        try:
            features = pd.DataFrame(index=df.index)
            
            # Price-based features
            features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            features['price_momentum'] = df['close'].pct_change(5)
            features['price_acceleration'] = features['price_momentum'].diff()
            
            # Volatility features
            features['volatility_20'] = features['log_returns'].rolling(20).std()
            features['volatility_50'] = features['log_returns'].rolling(50).std()
            features['volatility_ratio'] = features['volatility_20'] / features['volatility_50']
            
            # Volume features
            features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            features['volume_momentum'] = df['volume'].pct_change(5)
            
            # Technical indicators
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            features['macd'] = exp1 - exp2
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            features['macd_histogram'] = features['macd'] - features['macd_signal']
            
            # Bollinger Bands
            bb_middle = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            features['bb_position'] = (df['close'] - bb_middle) / bb_std
            features['bb_width'] = bb_std / bb_middle
            
            # ATR
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            features['atr'] = tr.rolling(14).mean()
            
            # ADX
            features['adx'] = self._calculate_adx(df)
            
            # Regime-specific features
            features['trend_strength'] = abs(features['price_momentum']) / features['volatility_20']
            features['volume_trend_alignment'] = np.sign(features['price_momentum']) * np.sign(features['volume_momentum'])
            
            # Fill NaN values
            features = features.fillna(method='ffill').fillna(0)
            
            # Remove infinite values
            features = features.replace([np.inf, -np.inf], 0)
            
            self.logger.info(f"📊 Generated {len(features.columns)} comprehensive features")
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Error generating features: {e}")
            return pd.DataFrame()
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index."""
        try:
            # Calculate True Range
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            
            # Calculate Directional Movement
            up_move = df['high'] - df['high'].shift()
            down_move = df['low'].shift() - df['low']
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # Smooth the values
            tr_smooth = tr.rolling(period).mean()
            plus_di = pd.Series(plus_dm).rolling(period).mean() / tr_smooth * 100
            minus_di = pd.Series(minus_dm).rolling(period).mean() / tr_smooth * 100
            
            # Calculate ADX
            dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
            adx = dx.rolling(period).mean()
            
            return adx.fillna(0)
            
        except Exception:
            return pd.Series([0] * len(df), index=df.index)
    
    def _create_composite_features(
        self, 
        features: pd.DataFrame, 
        hmm_states: np.ndarray, 
        hmm_probs: np.ndarray
    ) -> np.ndarray:
        """Create composite features combining original features with HMM information."""
        try:
            # Convert HMM states to one-hot encoding
            state_encoder = pd.get_dummies(hmm_states, prefix='hmm_state')
            
            # Combine original features with HMM states and probabilities
            composite_features = pd.concat([
                features,
                state_encoder,
                pd.DataFrame(hmm_probs, columns=[f'hmm_prob_{i}' for i in range(hmm_probs.shape[1])])
            ], axis=1)
            
            # Fill any remaining NaN values
            composite_features = composite_features.fillna(0)
            
            return composite_features.values
            
        except Exception as e:
            self.logger.error(f"❌ Error creating composite features: {e}")
            return features.values
    
    def _train_transition_model(
        self, 
        features: np.ndarray, 
        hmm_states: np.ndarray, 
        cluster_labels: np.ndarray
    ) -> None:
        """Train regime transition prediction model."""
        try:
            # Create transition labels
            transition_labels = []
            feature_sequences = []
            
            for i in range(self.sequence_length, len(features)):
                # Check if regime changed
                current_regime = cluster_labels[i]
                previous_regime = cluster_labels[i-1]
                
                if current_regime != previous_regime:
                    transition_labels.append(1)  # Regime change
                else:
                    transition_labels.append(0)  # No change
                
                # Use recent feature sequence
                feature_sequences.append(features[i-self.sequence_length:i].flatten())
            
            if len(transition_labels) > 0:
                X = np.array(feature_sequences)
                y = np.array(transition_labels)
                
                # Train transition model
                self.transition_model.fit(X, y)
                
                self.logger.info(f"🔄 Transition model trained with {len(transition_labels)} samples")
            
        except Exception as e:
            self.logger.error(f"❌ Error training transition model: {e}")
    
    def _analyze_regime_characteristics(
        self, 
        features: pd.DataFrame, 
        hmm_states: np.ndarray, 
        cluster_labels: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze characteristics of discovered regimes."""
        try:
            regime_analysis = {}
            
            unique_clusters = np.unique(cluster_labels)
            
            for cluster_id in unique_clusters:
                cluster_mask = cluster_labels == cluster_id
                cluster_data = features[cluster_mask]
                
                if len(cluster_data) == 0:
                    continue
                
                # Calculate regime characteristics
                characteristics = {
                    'count': len(cluster_data),
                    'percentage': len(cluster_data) / len(features) * 100,
                    'avg_volatility': cluster_data['volatility_20'].mean(),
                    'avg_momentum': cluster_data['price_momentum'].mean(),
                    'avg_volume_ratio': cluster_data['volume_ratio'].mean(),
                    'avg_rsi': cluster_data['rsi'].mean(),
                    'avg_trend_strength': cluster_data['trend_strength'].mean(),
                    'regime_type': self._classify_regime_type(cluster_data)
                }
                
                regime_analysis[f'cluster_{cluster_id}'] = characteristics
            
            return regime_analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing regime characteristics: {e}")
            return {}
    
    def _classify_regime_type(self, cluster_data: pd.DataFrame) -> RegimeType:
        """Classify regime type based on characteristics."""
        try:
            avg_momentum = cluster_data['price_momentum'].mean()
            avg_volatility = cluster_data['volatility_20'].mean()
            avg_trend_strength = cluster_data['trend_strength'].mean()
            
            # Classification logic
            if avg_momentum > 0.01 and avg_trend_strength > 0.5:
                return RegimeType.BULL_TREND
            elif avg_momentum < -0.01 and avg_trend_strength > 0.5:
                return RegimeType.BEAR_TREND
            elif avg_volatility > 0.02:
                return RegimeType.VOLATILE
            elif abs(avg_momentum) < 0.005:
                return RegimeType.SIDEWAYS
            else:
                return RegimeType.CONSOLIDATION
                
        except Exception:
            return RegimeType.CONSOLIDATION
    
    def _generate_training_report(
        self, 
        features: pd.DataFrame, 
        hmm_states: np.ndarray, 
        cluster_labels: np.ndarray, 
        regime_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive training report."""
        try:
            # Calculate model scores
            hmm_score = self.hmm_model.score(self.scaler.transform(features)) if self.hmm_model else 0
            
            # Calculate cluster quality metrics
            from sklearn.metrics import silhouette_score, calinski_harabasz_score
            
            composite_features = self._create_composite_features(features, hmm_states, self.hmm_model.predict_proba(self.scaler.transform(features)))
            
            silhouette = silhouette_score(composite_features, cluster_labels) if len(np.unique(cluster_labels)) > 1 else 0
            calinski = calinski_harabasz_score(composite_features, cluster_labels) if len(np.unique(cluster_labels)) > 1 else 0
            
            # Regime distribution
            cluster_counts = pd.Series(cluster_labels).value_counts()
            
            report = {
                'hmm_score': hmm_score,
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski,
                'n_states': self.n_states,
                'n_clusters': self.n_clusters,
                'total_samples': len(features),
                'regime_distribution': cluster_counts.to_dict(),
                'regime_analysis': regime_analysis,
                'training_metadata': {
                    'sequence_length': self.sequence_length,
                    'transition_threshold': self.transition_threshold
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Error generating training report: {e}")
            return {}
    
    @handle_errors(
        exceptions=(Exception,),
        default_return={"success": False, "error": "Regime prediction failed"},
        context="regime_prediction"
    )
    async def predict_regime_changes(
        self, 
        df: pd.DataFrame,
        features: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Predict regime changes and current regime.
        
        Args:
            df: Recent OHLCV data
            features: Optional pre-computed features
            
        Returns:
            Regime prediction results
        """
        if not self.hmm_model or not self.cluster_model:
            return {"success": False, "error": "Models not trained"}
        
        if df.empty or len(df) < self.sequence_length:
            return {"success": False, "error": "Insufficient data for prediction"}
        
        try:
            # Generate features if not provided
            if features is None:
                features = self._generate_comprehensive_features(df)
            
            if features.empty:
                return {"success": False, "error": "No features generated"}
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Predict HMM states
            hmm_states = self.hmm_model.predict(features_scaled)
            hmm_probs = self.hmm_model.predict_proba(features_scaled)
            
            # Predict clusters
            composite_features = self._create_composite_features(features, hmm_states, hmm_probs)
            cluster_labels = self.cluster_model.predict(composite_features)
            
            # Predict regime transitions
            transition_probabilities = self._predict_transitions(features_scaled)
            
            # Analyze current regime
            current_regime = self._analyze_current_regime(
                features.iloc[-1], cluster_labels[-1], hmm_probs[-1]
            )
            
            # Update regime history
            self._update_regime_history(current_regime)
            
            # Check for regime changes
            regime_changes = self._detect_regime_changes(cluster_labels, transition_probabilities)
            
            return {
                "success": True,
                "current_regime": current_regime,
                "regime_changes": regime_changes,
                "transition_probabilities": transition_probabilities,
                "hmm_states": hmm_states,
                "cluster_labels": cluster_labels,
                "confidence": current_regime.confidence
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error in regime prediction: {e}")
            return {"success": False, "error": str(e)}
    
    def _predict_transitions(self, features: np.ndarray) -> List[float]:
        """Predict regime transition probabilities."""
        try:
            if not self.transition_model or len(features) < self.sequence_length:
                return [0.0] * len(features)
            
            transition_probs = []
            
            for i in range(self.sequence_length, len(features)):
                feature_sequence = features[i-self.sequence_length:i].flatten()
                prob = self.transition_model.predict_proba([feature_sequence])[0][1]  # Probability of transition
                transition_probs.append(prob)
            
            # Pad beginning with zeros
            transition_probs = [0.0] * self.sequence_length + transition_probs
            
            return transition_probs
            
        except Exception as e:
            self.logger.error(f"❌ Error predicting transitions: {e}")
            return [0.0] * len(features)
    
    def _analyze_current_regime(
        self, 
        current_features: pd.Series, 
        cluster_label: int, 
        hmm_probs: np.ndarray
    ) -> RegimeState:
        """Analyze current regime state."""
        try:
            # Calculate confidence based on HMM probabilities
            confidence = np.max(hmm_probs)
            
            # Determine regime type
            regime_type = self._classify_regime_type(pd.DataFrame([current_features]))
            
            # Calculate duration (how long in current regime)
            duration = 1
            if self.regime_history:
                last_regime = self.regime_history[-1]
                if last_regime.regime_id == cluster_label:
                    duration = last_regime.duration + 1
            
            return RegimeState(
                regime_id=cluster_label,
                regime_type=regime_type,
                confidence=confidence,
                duration=duration,
                volatility=current_features.get('volatility_20', 0),
                momentum=current_features.get('price_momentum', 0),
                volume_profile=current_features.get('volume_ratio', 1),
                metadata={
                    'hmm_probs': hmm_probs.tolist(),
                    'features': current_features.to_dict()
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing current regime: {e}")
            return RegimeState(
                regime_id=0,
                regime_type=RegimeType.CONSOLIDATION,
                confidence=0.0,
                duration=1,
                volatility=0.0,
                momentum=0.0,
                volume_profile=1.0,
                metadata={}
            )
    
    def _update_regime_history(self, current_regime: RegimeState) -> None:
        """Update regime history."""
        if self.regime_history and self.regime_history[-1].regime_id == current_regime.regime_id:
            # Update duration of existing regime
            self.regime_history[-1] = current_regime
        else:
            # Add new regime
            self.regime_history.append(current_regime)
        
        # Keep only recent history
        max_history = self.config.get("max_regime_history", 100)
        if len(self.regime_history) > max_history:
            self.regime_history = self.regime_history[-max_history:]
    
    def _detect_regime_changes(
        self, 
        cluster_labels: np.ndarray, 
        transition_probabilities: List[float]
    ) -> List[RegimeTransition]:
        """Detect regime changes based on cluster labels and transition probabilities."""
        try:
            regime_changes = []
            
            for i in range(1, len(cluster_labels)):
                current_cluster = cluster_labels[i]
                previous_cluster = cluster_labels[i-1]
                
                # Check for cluster change
                if current_cluster != previous_cluster:
                    transition_prob = transition_probabilities[i] if i < len(transition_probabilities) else 0.0
                    
                    if transition_prob > self.transition_threshold:
                        # Create transition record
                        transition = RegimeTransition(
                            from_regime=previous_cluster,
                            to_regime=current_cluster,
                            probability=transition_prob,
                            trigger_features={},  # Could be enhanced with feature analysis
                            timestamp=pd.Timestamp.now()  # Could use actual timestamp if available
                        )
                        
                        regime_changes.append(transition)
                        self.transition_history.append(transition)
            
            return regime_changes
            
        except Exception as e:
            self.logger.error(f"❌ Error detecting regime changes: {e}")
            return []
    
    def get_regime_summary(self) -> Dict[str, Any]:
        """Get summary of current regime state."""
        try:
            if not self.regime_history:
                return {"error": "No regime history available"}
            
            current_regime = self.regime_history[-1]
            
            return {
                "current_regime": {
                    "id": current_regime.regime_id,
                    "type": current_regime.regime_type.value,
                    "confidence": current_regime.confidence,
                    "duration": current_regime.duration,
                    "volatility": current_regime.volatility,
                    "momentum": current_regime.momentum,
                    "volume_profile": current_regime.volume_profile
                },
                "regime_history": {
                    "total_regimes": len(self.regime_history),
                    "recent_transitions": len(self.transition_history[-10:]) if self.transition_history else 0
                },
                "model_info": {
                    "n_states": self.n_states,
                    "n_clusters": self.n_clusters,
                    "models_trained": all([self.hmm_model, self.cluster_model, self.transition_model])
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting regime summary: {e}")
            return {"error": str(e)}
    
    def clear_cache(self) -> None:
        """Clear prediction cache."""
        self._feature_cache.clear()
        self._prediction_cache.clear()
        self.logger.info("HMM regime cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "feature_cache_size": len(self._feature_cache),
            "prediction_cache_size": len(self._prediction_cache),
            "regime_history_size": len(self.regime_history),
            "transition_history_size": len(self.transition_history)
        }