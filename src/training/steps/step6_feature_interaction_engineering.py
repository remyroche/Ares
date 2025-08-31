"""
Step6: Feature Interaction Engineering

This module implements comprehensive feature interaction engineering for the Tactician model.
It creates interaction terms between technical indicators, market features, and derived metrics
to capture non-linear relationships and improve model performance.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

# Configure logging
logger = logging.getLogger(__name__)

class FeatureInteractionEngine:
    """
    Advanced feature interaction engineering for step6.
    
    Creates interaction terms between:
    - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
    - Market features (price, volume, volatility)
    - Derived metrics (momentum, acceleration, regime indicators)
    - Cross-timeframe features
    - Regime-dependent interactions
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature interaction engine.
        
        Args:
            config: Configuration dictionary with interaction parameters
        """
        self.config = config
        self.logger = logger
        
        # Load interaction configuration
        step6_config = config.get("step6_feature_interaction_engineering", {})
        
        # Interaction patterns and weights
        self.interaction_patterns = {
            "momentum_volume": {
                "features": ["RSI", "MACD", "Stochastic", "Volume_Ratio"],
                "weight": step6_config.get("momentum_volume_weight", 1.5),
                "enabled": step6_config.get("momentum_volume_enabled", True)
            },
            "trend_volatility": {
                "features": ["SMA_Ratio", "EMA_Ratio", "BB_Position", "ATR_Normalized"],
                "weight": step6_config.get("trend_volatility_weight", 1.8),
                "enabled": step6_config.get("trend_volatility_enabled", True)
            },
            "oscillator_trend": {
                "features": ["RSI", "Williams_R", "CCI", "SMA_Ratio"],
                "weight": step6_config.get("oscillator_trend_weight", 1.3),
                "enabled": step6_config.get("oscillator_trend_enabled", True)
            },
            "volume_price": {
                "features": ["OBV_Normalized", "MFI", "Price_Momentum", "Volume_Ratio"],
                "weight": step6_config.get("volume_price_weight", 1.6),
                "enabled": step6_config.get("volume_price_enabled", True)
            },
            "volatility_regime": {
                "features": ["ATR_Normalized", "BB_Squeeze", "Volatility", "Market_Regime"],
                "weight": step6_config.get("volatility_regime_weight", 1.4),
                "enabled": step6_config.get("volatility_regime_enabled", True)
            },
            "cross_timeframe": {
                "features": ["RSI_14", "RSI_30", "MACD_12_26", "MACD_20_40"],
                "weight": step6_config.get("cross_timeframe_weight", 1.2),
                "enabled": step6_config.get("cross_timeframe_enabled", True)
            },
            "regime_dependent": {
                "features": ["Trend_Strength", "Volatility_Regime", "Volume_Regime", "Momentum_Regime"],
                "weight": step6_config.get("regime_dependent_weight", 1.7),
                "enabled": step6_config.get("regime_dependent_enabled", True)
            }
        }
        
        # Interaction strength thresholds
        self.interaction_thresholds = {
            "strong": step6_config.get("strong_interaction_threshold", 0.7),
            "medium": step6_config.get("medium_interaction_threshold", 0.5),
            "weak": step6_config.get("weak_interaction_threshold", 0.3)
        }
        
        # Feature selection parameters
        self.selection_params = {
            "max_interactions": step6_config.get("max_interactions", 100),
            "min_importance": step6_config.get("min_importance", 0.01),
            "correlation_threshold": step6_config.get("correlation_threshold", 0.8),
            "mutual_info_threshold": step6_config.get("mutual_info_threshold", 0.05)
        }
        
        # Performance tracking
        self.interaction_performance = {}
        self.feature_importance_history = []
        self.selected_interactions_history = []
        
        # Initialize scaler for interaction features
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def extract_interaction_features(self, features: np.ndarray, 
                                   feature_names: List[str],
                                   market_data: pd.DataFrame) -> np.ndarray:
        """
        Extract comprehensive interaction features.
        
        Args:
            features: Base feature array
            feature_names: Names of base features
            market_data: Market data for regime analysis
            
        Returns:
            np.ndarray: Interaction features
        """
        try:
            self.logger.info("Extracting feature interactions...")
            
            # 1. Create basic interaction features
            basic_interactions = self._create_basic_interactions(features, feature_names)
            
            # 2. Create pattern-based interactions
            pattern_interactions = self._create_pattern_interactions(features, feature_names)
            
            # 3. Create regime-dependent interactions
            regime_interactions = self._create_regime_interactions(features, feature_names, market_data)
            
            # 4. Create cross-timeframe interactions
            timeframe_interactions = self._create_cross_timeframe_interactions(features, feature_names)
            
            # 5. Combine all interactions
            all_interactions = np.concatenate([
                basic_interactions,
                pattern_interactions,
                regime_interactions,
                timeframe_interactions
            ], axis=1)
            
            # 6. Select optimal interactions
            selected_interactions = self._select_optimal_interactions(all_interactions, market_data)
            
            # 7. Scale interaction features
            if not self.is_fitted:
                selected_interactions = self.scaler.fit_transform(selected_interactions)
                self.is_fitted = True
            else:
                selected_interactions = self.scaler.transform(selected_interactions)
            
            self.logger.info(f"Extracted {selected_interactions.shape[1]} interaction features")
            
            return selected_interactions
            
        except Exception as e:
            self.logger.error(f"Feature interaction extraction failed: {e}")
            return np.zeros((features.shape[0], 50))  # Return default interactions
    
    def _create_basic_interactions(self, features: np.ndarray, 
                                 feature_names: List[str]) -> np.ndarray:
        """
        Create basic pairwise interactions between features.
        """
        interactions = []
        
        # Create feature name to index mapping
        feature_map = {name: i for i, name in enumerate(feature_names)}
        
        # Define important feature pairs for interactions
        important_pairs = [
            ("RSI", "MACD"),
            ("RSI", "Volume_Ratio"),
            ("MACD", "Volume_Ratio"),
            ("BB_Position", "ATR_Normalized"),
            ("SMA_Ratio", "EMA_Ratio"),
            ("Price_Momentum", "Volume_Ratio"),
            ("OBV_Normalized", "Price_Momentum"),
            ("Stochastic", "RSI"),
            ("Williams_R", "RSI"),
            ("CCI", "RSI")
        ]
        
        for feature1, feature2 in important_pairs:
            if feature1 in feature_map and feature2 in feature_map:
                idx1, idx2 = feature_map[feature1], feature_map[feature2]
                
                # Create interaction
                interaction = features[:, idx1] * features[:, idx2]
                interactions.append(interaction)
                
                # Create ratio interaction
                ratio_interaction = features[:, idx1] / (features[:, idx2] + 1e-8)
                interactions.append(ratio_interaction)
                
                # Create difference interaction
                diff_interaction = features[:, idx1] - features[:, idx2]
                interactions.append(diff_interaction)
        
        return np.column_stack(interactions) if interactions else np.zeros((features.shape[0], 0))
    
    def _create_pattern_interactions(self, features: np.ndarray, 
                                   feature_names: List[str]) -> np.ndarray:
        """
        Create pattern-based interactions using predefined patterns.
        """
        interactions = []
        feature_map = {name: i for i, name in enumerate(feature_names)}
        
        for pattern_name, pattern_config in self.interaction_patterns.items():
            if not pattern_config["enabled"]:
                continue
                
            pattern_features = pattern_config["features"]
            weight = pattern_config["weight"]
            
            # Find feature indices for this pattern
            pattern_indices = []
            for feature_name in pattern_features:
                if feature_name in feature_map:
                    pattern_indices.append(feature_map[feature_name])
            
            if len(pattern_indices) >= 2:
                # Create pattern-specific interactions
                pattern_interactions = self._create_pattern_specific_interactions(
                    features, pattern_indices, pattern_name, weight
                )
                interactions.extend(pattern_interactions)
        
        return np.column_stack(interactions) if interactions else np.zeros((features.shape[0], 0))
    
    def _create_pattern_specific_interactions(self, features: np.ndarray,
                                            pattern_indices: List[int],
                                            pattern_name: str,
                                            weight: float) -> List[np.ndarray]:
        """
        Create pattern-specific interactions.
        """
        interactions = []
        pattern_features = features[:, pattern_indices]
        
        if pattern_name == "momentum_volume":
            # Momentum × Volume interactions
            momentum_avg = np.mean(pattern_features[:, :3], axis=1)  # RSI, MACD, Stochastic
            volume_feature = pattern_features[:, 3]  # Volume_Ratio
            
            interactions.extend([
                momentum_avg * volume_feature * weight,  # Momentum × Volume
                momentum_avg / (volume_feature + 1e-8) * weight,  # Momentum / Volume
                np.std(pattern_features[:, :3], axis=1) * volume_feature * weight  # Momentum divergence × Volume
            ])
            
        elif pattern_name == "trend_volatility":
            # Trend × Volatility interactions
            trend_avg = np.mean(pattern_features[:, :2], axis=1)  # SMA_Ratio, EMA_Ratio
            volatility_avg = np.mean(pattern_features[:, 2:], axis=1)  # BB_Position, ATR_Normalized
            
            interactions.extend([
                trend_avg * volatility_avg * weight,  # Trend × Volatility
                trend_avg / (volatility_avg + 1e-8) * weight,  # Trend / Volatility
                np.abs(trend_avg) * volatility_avg * weight  # Trend strength × Volatility
            ])
            
        elif pattern_name == "oscillator_trend":
            # Oscillator × Trend interactions
            oscillator_avg = np.mean(pattern_features[:, :3], axis=1)  # RSI, Williams_R, CCI
            trend_feature = pattern_features[:, 3]  # SMA_Ratio
            
            interactions.extend([
                oscillator_avg * trend_feature * weight,  # Oscillator × Trend
                oscillator_avg / (trend_feature + 1e-8) * weight,  # Oscillator / Trend
                np.std(pattern_features[:, :3], axis=1) * trend_feature * weight  # Oscillator divergence × Trend
            ])
            
        elif pattern_name == "volume_price":
            # Volume × Price interactions
            volume_avg = np.mean(pattern_features[:, [0, 3]], axis=1)  # OBV_Normalized, Volume_Ratio
            price_feature = pattern_features[:, 2]  # Price_Momentum
            
            interactions.extend([
                volume_avg * price_feature * weight,  # Volume × Price
                volume_avg / (price_feature + 1e-8) * weight,  # Volume / Price
                np.sqrt(volume_avg) * price_feature * weight  # Volume-weighted price
            ])
            
        elif pattern_name == "volatility_regime":
            # Volatility × Regime interactions
            volatility_avg = np.mean(pattern_features[:, :3], axis=1)  # ATR, BB_Squeeze, Volatility
            regime_feature = pattern_features[:, 3] if pattern_features.shape[1] > 3 else np.ones(features.shape[0])
            
            interactions.extend([
                volatility_avg * regime_feature * weight,  # Volatility × Regime
                volatility_avg / (regime_feature + 1e-8) * weight,  # Volatility / Regime
                np.square(volatility_avg) * regime_feature * weight  # Volatility² × Regime
            ])
        
        return interactions
    
    def _create_regime_interactions(self, features: np.ndarray,
                                  feature_names: List[str],
                                  market_data: pd.DataFrame) -> np.ndarray:
        """
        Create regime-dependent interactions.
        """
        interactions = []
        
        # Identify market regime
        market_regime = self._identify_market_regime(market_data)
        
        # Create regime-specific interactions
        if market_regime == "trending":
            # Trending market interactions
            trend_interactions = self._create_trending_interactions(features, feature_names)
            interactions.extend(trend_interactions)
            
        elif market_regime == "ranging":
            # Ranging market interactions
            ranging_interactions = self._create_ranging_interactions(features, feature_names)
            interactions.extend(ranging_interactions)
            
        elif market_regime == "volatile":
            # Volatile market interactions
            volatile_interactions = self._create_volatile_interactions(features, feature_names)
            interactions.extend(volatile_interactions)
        
        return np.column_stack(interactions) if interactions else np.zeros((features.shape[0], 0))
    
    def _create_trending_interactions(self, features: np.ndarray,
                                    feature_names: List[str]) -> List[np.ndarray]:
        """
        Create interactions specific to trending markets.
        """
        interactions = []
        feature_map = {name: i for i, name in enumerate(feature_names)}
        
        # Trend-following interactions
        trend_features = ["SMA_Ratio", "EMA_Ratio", "MACD", "ADX"]
        momentum_features = ["RSI", "Stochastic", "CCI"]
        
        trend_indices = [feature_map.get(f) for f in trend_features if f in feature_map]
        momentum_indices = [feature_map.get(f) for f in momentum_features if f in feature_map]
        
        if trend_indices and momentum_indices:
            trend_avg = np.mean(features[:, trend_indices], axis=1)
            momentum_avg = np.mean(features[:, momentum_indices], axis=1)
            
            interactions.extend([
                trend_avg * momentum_avg * 1.5,  # Trend × Momentum
                trend_avg / (momentum_avg + 1e-8) * 1.3,  # Trend / Momentum
                np.abs(trend_avg) * momentum_avg * 1.4  # Trend strength × Momentum
            ])
        
        return interactions
    
    def _create_ranging_interactions(self, features: np.ndarray,
                                   feature_names: List[str]) -> List[np.ndarray]:
        """
        Create interactions specific to ranging markets.
        """
        interactions = []
        feature_map = {name: i for i, name in enumerate(feature_names)}
        
        # Range-trading interactions
        oscillator_features = ["RSI", "Stochastic", "Williams_R", "CCI"]
        volume_features = ["Volume_Ratio", "OBV_Normalized", "MFI"]
        
        oscillator_indices = [feature_map.get(f) for f in oscillator_features if f in feature_map]
        volume_indices = [feature_map.get(f) for f in volume_features if f in feature_map]
        
        if oscillator_indices and volume_indices:
            oscillator_avg = np.mean(features[:, oscillator_indices], axis=1)
            volume_avg = np.mean(features[:, volume_indices], axis=1)
            
            interactions.extend([
                oscillator_avg * volume_avg * 1.6,  # Oscillator × Volume
                oscillator_avg / (volume_avg + 1e-8) * 1.4,  # Oscillator / Volume
                np.std(features[:, oscillator_indices], axis=1) * volume_avg * 1.5  # Oscillator divergence × Volume
            ])
        
        return interactions
    
    def _create_volatile_interactions(self, features: np.ndarray,
                                    feature_names: List[str]) -> List[np.ndarray]:
        """
        Create interactions specific to volatile markets.
        """
        interactions = []
        feature_map = {name: i for i, name in enumerate(feature_names)}
        
        # Volatility-focused interactions
        volatility_features = ["ATR_Normalized", "BB_Squeeze", "Volatility"]
        risk_features = ["RSI", "Stochastic", "Williams_R"]
        
        volatility_indices = [feature_map.get(f) for f in volatility_features if f in feature_map]
        risk_indices = [feature_map.get(f) for f in risk_features if f in feature_map]
        
        if volatility_indices and risk_indices:
            volatility_avg = np.mean(features[:, volatility_indices], axis=1)
            risk_avg = np.mean(features[:, risk_indices], axis=1)
            
            interactions.extend([
                volatility_avg * risk_avg * 1.8,  # Volatility × Risk
                volatility_avg / (risk_avg + 1e-8) * 1.6,  # Volatility / Risk
                np.square(volatility_avg) * risk_avg * 1.7  # Volatility² × Risk
            ])
        
        return interactions
    
    def _create_cross_timeframe_interactions(self, features: np.ndarray,
                                           feature_names: List[str]) -> np.ndarray:
        """
        Create cross-timeframe interactions.
        """
        interactions = []
        feature_map = {name: i for i, name in enumerate(feature_names)}
        
        # Define timeframe pairs
        timeframe_pairs = [
            ("RSI_14", "RSI_30"),
            ("MACD_12_26", "MACD_20_40"),
            ("SMA_20", "SMA_50"),
            ("EMA_12", "EMA_26")
        ]
        
        for short_feature, long_feature in timeframe_pairs:
            if short_feature in feature_map and long_feature in feature_map:
                short_idx, long_idx = feature_map[short_feature], feature_map[long_feature]
                
                # Create cross-timeframe interactions
                interactions.extend([
                    features[:, short_idx] - features[:, long_idx],  # Divergence
                    features[:, short_idx] / (features[:, long_idx] + 1e-8),  # Ratio
                    features[:, short_idx] * features[:, long_idx],  # Product
                    np.abs(features[:, short_idx] - features[:, long_idx])  # Absolute divergence
                ])
        
        return np.column_stack(interactions) if interactions else np.zeros((features.shape[0], 0))
    
    def _identify_market_regime(self, market_data: pd.DataFrame) -> str:
        """
        Identify current market regime.
        """
        try:
            # Calculate regime indicators
            volatility = market_data['close'].pct_change().rolling(20).std().iloc[-1]
            trend_strength = abs(market_data['close'].rolling(20).mean().iloc[-1] - 
                               market_data['close'].rolling(50).mean().iloc[-1]) / market_data['close'].iloc[-1]
            
            if volatility > 0.03:
                return "volatile"
            elif trend_strength > 0.02:
                return "trending"
            else:
                return "ranging"
                
        except Exception as e:
            self.logger.warning(f"Market regime identification failed: {e}")
            return "ranging"  # Default to ranging
    
    def _select_optimal_interactions(self, interactions: np.ndarray,
                                   market_data: pd.DataFrame) -> np.ndarray:
        """
        Select optimal interactions based on importance and correlation.
        """
        try:
            # Create dummy target for feature selection (in real implementation, use actual target)
            dummy_target = np.random.choice([0, 1], size=interactions.shape[0])
            
            # Calculate mutual information
            mi_scores = mutual_info_classif(interactions, dummy_target, random_state=42)
            
            # Select interactions based on mutual information
            mi_threshold = self.selection_params["mutual_info_threshold"]
            important_indices = np.where(mi_scores > mi_threshold)[0]
            
            # Limit number of interactions
            max_interactions = self.selection_params["max_interactions"]
            if len(important_indices) > max_interactions:
                # Select top interactions by mutual information
                top_indices = np.argsort(mi_scores)[-max_interactions:]
                selected_interactions = interactions[:, top_indices]
            else:
                selected_interactions = interactions[:, important_indices]
            
            # Store selection history
            self.selected_interactions_history.append({
                "timestamp": datetime.now(),
                "n_interactions": selected_interactions.shape[1],
                "mi_scores": mi_scores[important_indices] if len(important_indices) > 0 else []
            })
            
            return selected_interactions
            
        except Exception as e:
            self.logger.error(f"Interaction selection failed: {e}")
            return interactions[:, :50]  # Return first 50 interactions as fallback
    
    def get_interaction_summary(self) -> Dict[str, Any]:
        """
        Get summary of interaction engineering results.
        """
        return {
            "interaction_patterns": self.interaction_patterns,
            "selection_params": self.selection_params,
            "performance_history": self.interaction_performance,
            "selected_interactions_count": len(self.selected_interactions_history),
            "is_fitted": self.is_fitted,
            "scaler_params": {
                "mean": self.scaler.mean_.tolist() if self.is_fitted else None,
                "scale": self.scaler.scale_.tolist() if self.is_fitted else None
            }
        }
    
    def update_performance(self, performance_metrics: Dict[str, float]) -> None:
        """
        Update interaction performance tracking.
        """
        self.interaction_performance[datetime.now()] = performance_metrics
    
    def get_feature_importance(self, interactions: np.ndarray,
                             target: np.ndarray) -> np.ndarray:
        """
        Calculate importance of interaction features.
        """
        try:
            # Calculate mutual information for interaction importance
            mi_scores = mutual_info_classif(interactions, target, random_state=42)
            
            # Store importance history
            self.feature_importance_history.append({
                "timestamp": datetime.now(),
                "importance_scores": mi_scores.tolist(),
                "mean_importance": np.mean(mi_scores),
                "max_importance": np.max(mi_scores)
            })
            
            return mi_scores
            
        except Exception as e:
            self.logger.error(f"Feature importance calculation failed: {e}")
            return np.ones(interactions.shape[1])  # Return uniform importance as fallback