"""HMM Components for regime discovery.

This module contains the core components used by the HMM regime discovery step,
extracted from the original large file for better modularity.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from src.utils.logger import system_logger


class HMMRegimeAnalyzer:
    """Analyzes market regimes using Hidden Markov Models."""
    
    def __init__(self, n_regimes: int, config: Dict[str, Any]):
        """Initialize HMM analyzer.
        
        Args:
            n_regimes: Number of regimes to identify
            config: Configuration dictionary
        """
        self.n_regimes = n_regimes
        self.config = config
        self.logger = system_logger.getChild("HMMRegimeAnalyzer")
        self.hmm_model = None
        
    async def analyze(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Analyze features to identify market regimes.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Analysis results dictionary
        """
        try:
            # Try to use hmmlearn if available
            from hmmlearn import hmm
            
            # Prepare data
            X = features.values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Initialize and fit HMM
            self.hmm_model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=100
            )
            
            self.hmm_model.fit(X_scaled)
            
            # Predict states
            states = self.hmm_model.predict(X_scaled)
            state_probs = self.hmm_model.predict_proba(X_scaled)
            
            # Calculate model score
            score = self.hmm_model.score(X_scaled)
            
            return {
                "success": True,
                "n_states": self.n_regimes,
                "regime_labels": states,
                "regime_probabilities": state_probs,
                "transition_matrix": self.hmm_model.transmat_,
                "model_score": score,
                "means": self.hmm_model.means_,
                "covariances": self.hmm_model.covars_
            }
            
        except ImportError:
            self.logger.warning("hmmlearn not available, using fallback clustering")
            return await self._fallback_analysis(features)
        except Exception as e:
            self.logger.error(f"HMM analysis failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _fallback_analysis(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Fallback analysis using K-means clustering.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Analysis results dictionary
        """
        try:
            # Prepare data
            X = features.values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Use K-means as fallback
            kmeans = KMeans(n_clusters=self.n_regimes, random_state=42)
            states = kmeans.fit_predict(X_scaled)
            
            # Calculate pseudo-probabilities based on distances
            distances = kmeans.transform(X_scaled)
            # Convert distances to probabilities (softmax)
            exp_distances = np.exp(-distances)
            state_probs = exp_distances / exp_distances.sum(axis=1, keepdims=True)
            
            # Calculate pseudo transition matrix
            trans_matrix = self._calculate_transition_matrix(states)
            
            return {
                "success": True,
                "n_states": self.n_regimes,
                "regime_labels": states,
                "regime_probabilities": state_probs,
                "transition_matrix": trans_matrix,
                "model_score": -kmeans.inertia_,  # Negative inertia as pseudo-score
                "cluster_centers": kmeans.cluster_centers_,
                "method": "kmeans_fallback"
            }
            
        except Exception as e:
            self.logger.error(f"Fallback analysis failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _calculate_transition_matrix(self, states: np.ndarray) -> np.ndarray:
        """Calculate transition matrix from state sequence.
        
        Args:
            states: Array of state labels
            
        Returns:
            Transition matrix
        """
        trans_matrix = np.zeros((self.n_regimes, self.n_regimes))
        
        for i in range(len(states) - 1):
            trans_matrix[states[i], states[i + 1]] += 1
        
        # Normalize rows
        row_sums = trans_matrix.sum(axis=1)
        trans_matrix = trans_matrix / np.maximum(row_sums[:, np.newaxis], 1)
        
        return trans_matrix


class FeatureEngineer:
    """Engineers features for regime analysis."""
    
    def __init__(self, feature_config: Dict[str, Any]):
        """Initialize feature engineer.
        
        Args:
            feature_config: Feature configuration
        """
        self.feature_config = feature_config
        self.logger = system_logger.getChild("FeatureEngineer")
        
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from raw market data.
        
        Args:
            data: Raw market data
            
        Returns:
            DataFrame with engineered features
        """
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features["returns"] = data["close"].pct_change()
        features["log_returns"] = np.log(data["close"] / data["close"].shift(1))
        features["price_change"] = data["close"] - data["close"].shift(1)
        
        # Technical indicators
        if self.feature_config.get("technical_indicators", True):
            features = self._add_technical_indicators(features, data)
        
        # Volume features
        if self.feature_config.get("volume_features", True) and "volume" in data.columns:
            features = self._add_volume_features(features, data)
        
        # Volatility features
        if self.feature_config.get("volatility_features", True):
            features = self._add_volatility_features(features, data)
        
        # Momentum features
        if self.feature_config.get("momentum_features", True):
            features = self._add_momentum_features(features, data)
        
        # Drop NaN values
        features = features.dropna()
        
        self.logger.info(f"✅ Engineered {len(features.columns)} features")
        
        return features
    
    def _add_technical_indicators(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators.
        
        Args:
            features: Feature DataFrame
            data: Raw market data
            
        Returns:
            Updated features
        """
        # RSI
        features["rsi"] = self._calculate_rsi(data["close"])
        
        # MACD
        macd_results = self._calculate_macd(data["close"])
        features["macd"] = macd_results["macd"]
        features["macd_signal"] = macd_results["signal"]
        features["macd_hist"] = macd_results["histogram"]
        
        # Bollinger Bands
        bb_results = self._calculate_bollinger_bands(data["close"])
        features["bb_upper"] = bb_results["upper"]
        features["bb_lower"] = bb_results["lower"]
        features["bb_position"] = (data["close"] - bb_results["lower"]) / (bb_results["upper"] - bb_results["lower"])
        
        # ATR
        features["atr"] = self._calculate_atr(data)
        
        return features
    
    def _add_volume_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features.
        
        Args:
            features: Feature DataFrame
            data: Raw market data
            
        Returns:
            Updated features
        """
        features["volume"] = data["volume"]
        features["volume_sma"] = data["volume"].rolling(20).mean()
        features["volume_ratio"] = data["volume"] / features["volume_sma"]
        features["volume_change"] = data["volume"].pct_change()
        
        # On-Balance Volume
        features["obv"] = (np.sign(data["close"].diff()) * data["volume"]).cumsum()
        
        return features
    
    def _add_volatility_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features.
        
        Args:
            features: Feature DataFrame
            data: Raw market data
            
        Returns:
            Updated features
        """
        # Rolling volatility
        for window in [5, 10, 20]:
            features[f"volatility_{window}"] = features["returns"].rolling(window).std()
        
        # Parkinson volatility (using high-low)
        if "high" in data.columns and "low" in data.columns:
            hl_ratio = np.log(data["high"] / data["low"])
            features["parkinson_vol"] = hl_ratio.rolling(20).apply(
                lambda x: np.sqrt(np.mean(x**2) / (4 * np.log(2)))
            )
        
        # Realized volatility
        features["realized_vol"] = np.sqrt(
            features["returns"].rolling(20).apply(lambda x: np.sum(x**2))
        )
        
        return features
    
    def _add_momentum_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add momentum features.
        
        Args:
            features: Feature DataFrame
            data: Raw market data
            
        Returns:
            Updated features
        """
        # Price momentum
        for period in [5, 10, 20]:
            features[f"momentum_{period}"] = data["close"] / data["close"].shift(period) - 1
        
        # Rate of change
        features["roc"] = (data["close"] - data["close"].shift(10)) / data["close"].shift(10) * 100
        
        # Stochastic oscillator
        if "high" in data.columns and "low" in data.columns:
            lowest_low = data["low"].rolling(14).min()
            highest_high = data["high"].rolling(14).max()
            features["stoch_k"] = 100 * (data["close"] - lowest_low) / (highest_high - lowest_low)
            features["stoch_d"] = features["stoch_k"].rolling(3).mean()
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD indicator."""
        exp_fast = prices.ewm(span=fast).mean()
        exp_slow = prices.ewm(span=slow).mean()
        macd = exp_fast - exp_slow
        macd_signal = macd.ewm(span=signal).mean()
        return {
            "macd": macd,
            "signal": macd_signal,
            "histogram": macd - macd_signal
        }
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        return {
            "upper": sma + num_std * std,
            "lower": sma - num_std * std,
            "middle": sma
        }
    
    def _calculate_atr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        if not all(col in data.columns for col in ["high", "low", "close"]):
            return pd.Series(index=data.index)
        
        high_low = data["high"] - data["low"]
        high_close = np.abs(data["high"] - data["close"].shift())
        low_close = np.abs(data["low"] - data["close"].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window).mean()


class RegimeCharacterizer:
    """Characterizes identified market regimes."""
    
    def __init__(self):
        """Initialize regime characterizer."""
        self.logger = system_logger.getChild("RegimeCharacterizer")
        
    async def characterize(
        self, 
        features: pd.DataFrame, 
        hmm_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Characterize market regimes based on features and HMM results.
        
        Args:
            features: Feature DataFrame
            hmm_results: HMM analysis results
            
        Returns:
            Regime characteristics
        """
        regime_labels = hmm_results.get("regime_labels", [])
        n_regimes = hmm_results.get("n_states", 0)
        
        characteristics = {}
        
        for regime in range(n_regimes):
            mask = regime_labels == regime
            if np.any(mask):
                regime_features = features[mask]
                
                # Calculate basic statistics
                regime_stats = {
                    "count": int(np.sum(mask)),
                    "percentage": float(np.mean(mask) * 100),
                    "duration_stats": self._calculate_duration_stats(regime_labels, regime)
                }
                
                # Feature statistics
                for feature in features.columns:
                    if feature in regime_features:
                        regime_stats[f"{feature}_mean"] = float(regime_features[feature].mean())
                        regime_stats[f"{feature}_std"] = float(regime_features[feature].std())
                
                # Determine regime type
                regime_type = self._determine_regime_type(regime_stats, features.columns)
                regime_stats["type"] = regime_type
                regime_stats["label"] = self._get_regime_label(regime_type)
                
                characteristics[f"regime_{regime}"] = regime_stats
        
        # Add transition analysis
        characteristics["transitions"] = self._analyze_transitions(
            regime_labels, 
            hmm_results.get("transition_matrix")
        )
        
        return characteristics
    
    def _calculate_duration_stats(self, regime_labels: np.ndarray, regime: int) -> Dict[str, float]:
        """Calculate duration statistics for a regime."""
        durations = []
        current_duration = 0
        
        for label in regime_labels:
            if label == regime:
                current_duration += 1
            elif current_duration > 0:
                durations.append(current_duration)
                current_duration = 0
        
        if current_duration > 0:
            durations.append(current_duration)
        
        if durations:
            return {
                "mean": float(np.mean(durations)),
                "std": float(np.std(durations)),
                "min": float(np.min(durations)),
                "max": float(np.max(durations))
            }
        else:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}
    
    def _determine_regime_type(self, stats: Dict[str, Any], feature_names: List[str]) -> str:
        """Determine the type of regime based on statistics."""
        # Simple heuristic based on volatility and returns
        vol_mean = stats.get("volatility_20_mean", stats.get("volatility_10_mean", 0))
        return_mean = stats.get("returns_mean", 0)
        
        if vol_mean > 0.02:  # High volatility threshold
            if return_mean > 0:
                return "volatile_bullish"
            else:
                return "volatile_bearish"
        elif vol_mean < 0.01:  # Low volatility threshold
            if abs(return_mean) < 0.001:
                return "ranging"
            elif return_mean > 0:
                return "steady_bullish"
            else:
                return "steady_bearish"
        else:  # Normal volatility
            if return_mean > 0.001:
                return "bullish"
            elif return_mean < -0.001:
                return "bearish"
            else:
                return "neutral"
    
    def _get_regime_label(self, regime_type: str) -> str:
        """Get human-readable label for regime type."""
        labels = {
            "volatile_bullish": "Volatile Bull",
            "volatile_bearish": "Volatile Bear",
            "ranging": "Range-Bound",
            "steady_bullish": "Steady Bull",
            "steady_bearish": "Steady Bear",
            "bullish": "Bullish",
            "bearish": "Bearish",
            "neutral": "Neutral"
        }
        return labels.get(regime_type, "Unknown")
    
    def _analyze_transitions(
        self, 
        regime_labels: np.ndarray, 
        transition_matrix: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Analyze regime transitions."""
        transitions = []
        
        for i in range(len(regime_labels) - 1):
            if regime_labels[i] != regime_labels[i + 1]:
                transitions.append({
                    "from": int(regime_labels[i]),
                    "to": int(regime_labels[i + 1]),
                    "index": i
                })
        
        # Calculate transition frequencies
        trans_freq = {}
        for trans in transitions:
            key = f"{trans['from']}_to_{trans['to']}"
            trans_freq[key] = trans_freq.get(key, 0) + 1
        
        return {
            "total_transitions": len(transitions),
            "transition_rate": len(transitions) / len(regime_labels) if len(regime_labels) > 0 else 0,
            "frequencies": trans_freq,
            "matrix": transition_matrix.tolist() if transition_matrix is not None else None
        }