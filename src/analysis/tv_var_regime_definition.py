"""
TV-VAR 8-Feature Regime Definition System

This module implements the regime detection system based on the 8 core features:
1. rv_z_short: Z-score of realized vol (1-2 days)
2. rv_z_long: Z-score of realized vol (10-20 days)
3. vol_ratio: rv_short / rv_long
4. volume_z: Rolling Z of traded volume
5. spread_proxy_z: High-low / close deviation
6. trend_slope_z: Z of linear regression slope (1-3 days)
7. trend_strength: Slope magnitude
8. drawdown_z: Z of rolling drawdown

These features define 6 distinct market regimes for TV-VAR analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regimes based on 8-feature definition."""
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    STRESS_REGIME = "STRESS_REGIME"
    TREND_REGIME = "TREND_REGIME"
    LIQUIDITY_REGIME = "LIQUIDITY_REGIME"
    NEUTRAL = "NEUTRAL"

@dataclass
class RegimeThresholds:
    """Thresholds for regime detection."""
    high_volatility_rv_z_short: float = 1.5
    high_volatility_vol_ratio: float = 1.2
    low_volatility_rv_z_short: float = 0.5
    low_volatility_vol_ratio: float = 0.8
    stress_drawdown_z: float = 2.0
    stress_spread_proxy_z: float = 1.5
    trend_slope_z: float = 1.0
    trend_strength: float = 0.7
    liquidity_volume_z: float = 1.5
    liquidity_spread_proxy_z: float = 0.5

@dataclass
class RegimeCharacteristics:
    """Characteristics of each regime for TV-VAR priors."""
    volatility_level: float
    correlation_strength: float
    parameter_variance: float
    regime_persistence: float
    specialist_weights: Dict[str, float]

class EightFeatureRegimeDetector:
    """
    Advanced regime detection using the 8 core features.
    
    This system defines regimes based on comprehensive market conditions
    including volatility, liquidity, trend, and stress factors.
    """
    
    def __init__(self, 
                 thresholds: Optional[RegimeThresholds] = None,
                 use_clustering: bool = False,
                 adaptive_thresholds: bool = False):
        """
        Initialize 8-feature regime detector.
        
        Args:
            thresholds: Custom thresholds for regime detection
            use_clustering: Use K-means clustering for regime discovery
            adaptive_thresholds: Adapt thresholds based on recent market conditions
        """
        self.thresholds = thresholds or RegimeThresholds()
        self.use_clustering = use_clustering
        self.adaptive_thresholds = adaptive_thresholds
        
        # Feature weights for regime scoring
        self.feature_weights = {
            'rv_z_short': 0.15,      # Short-term volatility
            'rv_z_long': 0.15,       # Long-term volatility
            'vol_ratio': 0.10,       # Volatility expansion/contraction
            'volume_z': 0.15,        # Participation proxy
            'spread_proxy_z': 0.10,   # Execution friction
            'trend_slope_z': 0.15,    # Directional bias
            'trend_strength': 0.10,   # Trend magnitude
            'drawdown_z': 0.10        # Risk-off detection
        }
        
        # Regime characteristics for TV-VAR priors
        self.regime_characteristics = self._initialize_regime_characteristics()
        
        # Adaptive threshold history
        self.threshold_history = []
        
        logger.info("✅ 8-Feature Regime Detector initialized")
    
    def _initialize_regime_characteristics(self) -> Dict[MarketRegime, RegimeCharacteristics]:
        """Initialize regime characteristics for TV-VAR Bayesian priors."""
        
        return {
            MarketRegime.HIGH_VOLATILITY: RegimeCharacteristics(
                volatility_level=2.0,
                correlation_strength=0.7,
                parameter_variance=0.3,
                regime_persistence=0.6,
                specialist_weights={
                    'risk_specialist': 0.8,
                    'volatility_specialist': 0.7,
                    'momentum_specialist': 0.4,
                    'liquidity_specialist': 0.3
                }
            ),
            MarketRegime.LOW_VOLATILITY: RegimeCharacteristics(
                volatility_level=0.5,
                correlation_strength=0.3,
                parameter_variance=0.1,
                regime_persistence=0.8,
                specialist_weights={
                    'trend_specialist': 0.7,
                    'liquidity_specialist': 0.6,
                    'risk_specialist': 0.2,
                    'momentum_specialist': 0.5
                }
            ),
            MarketRegime.STRESS_REGIME: RegimeCharacteristics(
                volatility_level=3.0,
                correlation_strength=0.9,
                parameter_variance=0.5,
                regime_persistence=0.4,
                specialist_weights={
                    'risk_specialist': 0.9,
                    'liquidity_specialist': 0.4,
                    'trend_specialist': 0.1,
                    'momentum_specialist': 0.2
                }
            ),
            MarketRegime.TREND_REGIME: RegimeCharacteristics(
                volatility_level=1.2,
                correlation_strength=0.5,
                parameter_variance=0.2,
                regime_persistence=0.7,
                specialist_weights={
                    'trend_specialist': 0.8,
                    'momentum_specialist': 0.7,
                    'risk_specialist': 0.3,
                    'liquidity_specialist': 0.4
                }
            ),
            MarketRegime.LIQUIDITY_REGIME: RegimeCharacteristics(
                volatility_level=0.8,
                correlation_strength=0.4,
                parameter_variance=0.15,
                regime_persistence=0.6,
                specialist_weights={
                    'liquidity_specialist': 0.8,
                    'volume_specialist': 0.7,
                    'risk_specialist': 0.2,
                    'trend_specialist': 0.4
                }
            ),
            MarketRegime.NEUTRAL: RegimeCharacteristics(
                volatility_level=1.0,
                correlation_strength=0.5,
                parameter_variance=0.2,
                regime_persistence=0.5,
                specialist_weights={
                    'risk_specialist': 0.5,
                    'trend_specialist': 0.5,
                    'liquidity_specialist': 0.5,
                    'momentum_specialist': 0.5
                }
            )
        }
    
    def detect_regimes(self, features_df: pd.DataFrame) -> pd.Series:
        """
        Detect regimes using the 8-feature definition.
        
        Args:
            features_df: DataFrame with the 8 core features
            
        Returns:
            Series of regime assignments
        """
        logger.info("🔍 Detecting regimes using 8-feature definition")
        
        # Validate features
        self._validate_features(features_df)
        
        # Apply adaptive thresholds if enabled
        if self.adaptive_thresholds:
            current_thresholds = self._update_adaptive_thresholds(features_df)
        else:
            current_thresholds = self.thresholds
        
        # Use clustering or rule-based detection
        if self.use_clustering:
            regimes = self._detect_regimes_clustering(features_df)
        else:
            regimes = self._detect_regimes_rule_based(features_df, current_thresholds)
        
        # Post-process regimes for temporal consistency
        regimes = self._post_process_regimes(regimes)
        
        # Log regime distribution
        regime_counts = regimes.value_counts()
        logger.info(f"📊 Regime distribution: {dict(regime_counts)}")
        
        return regimes
    
    def _validate_features(self, features_df: pd.DataFrame) -> None:
        """Validate that all 8 required features are present."""
        
        required_features = {
            'rv_z_short', 'rv_z_long', 'vol_ratio',
            'volume_z', 'spread_proxy_z',
            'trend_slope_z', 'trend_strength', 'drawdown_z'
        }
        
        missing_features = required_features - set(features_df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Check for sufficient data
        if len(features_df) < 50:
            raise ValueError(f"Insufficient data: need at least 50 samples, got {len(features_df)}")
        
        # Check for extreme values
        for col in features_df.columns:
            if np.isinf(features_df[col]).any():
                logger.warning(f"Infinite values found in {col}, replacing with NaN")
                features_df[col] = features_df[col].replace([np.inf, -np.inf], np.nan)
        
        # Handle missing values
        if features_df.isnull().any().any():
            logger.warning("Missing values found, using forward fill")
            features_df = features_df.fillna(method='ffill').fillna(0)
    
    def _detect_regimes_rule_based(self, 
                                  features_df: pd.DataFrame, 
                                  thresholds: RegimeThresholds) -> pd.Series:
        """
        Detect regimes using rule-based approach with the 8 features.
        
        Priority order (most restrictive first):
        1. STRESS_REGIME
        2. HIGH_VOLATILITY
        3. LOW_VOLATILITY
        4. TREND_REGIME
        5. LIQUIDITY_REGIME
        6. NEUTRAL (default)
        """
        
        regimes = pd.Series(index=features_df.index, data=MarketRegime.NEUTRAL.value)
        
        # 1. Stress Regime (highest priority)
        stress_mask = (
            (features_df['drawdown_z'] > thresholds.stress_drawdown_z) &
            (features_df['spread_proxy_z'] > thresholds.stress_spread_proxy_z)
        )
        regimes[stress_mask] = MarketRegime.STRESS_REGIME.value
        
        # 2. High Volatility Regime
        high_vol_mask = (
            (features_df['rv_z_short'] > thresholds.high_volatility_rv_z_short) &
            (features_df['vol_ratio'] > thresholds.high_volatility_vol_ratio) &
            (~stress_mask)  # Not already stress
        )
        regimes[high_vol_mask] = MarketRegime.HIGH_VOLATILITY.value
        
        # 3. Low Volatility Regime
        low_vol_mask = (
            (features_df['rv_z_short'] < thresholds.low_volatility_rv_z_short) &
            (features_df['vol_ratio'] < thresholds.low_volatility_vol_ratio) &
            (~stress_mask) & (~high_vol_mask)  # Not already stress or high vol
        )
        regimes[low_vol_mask] = MarketRegime.LOW_VOLATILITY.value
        
        # 4. Trend Regime
        trend_mask = (
            (features_df['trend_slope_z'] > thresholds.trend_slope_z) &
            (features_df['trend_strength'] > thresholds.trend_strength) &
            (~stress_mask) & (~high_vol_mask) & (~low_vol_mask)  # Not already assigned
        )
        regimes[trend_mask] = MarketRegime.TREND_REGIME.value
        
        # 5. Liquidity Regime
        liquidity_mask = (
            (features_df['volume_z'] > thresholds.liquidity_volume_z) &
            (features_df['spread_proxy_z'] < thresholds.liquidity_spread_proxy_z) &
            (~stress_mask) & (~high_vol_mask) & (~low_vol_mask) & (~trend_mask)  # Not already assigned
        )
        regimes[liquidity_mask] = MarketRegime.LIQUIDITY_REGIME.value
        
        return regimes
    
    def _detect_regimes_clustering(self, features_df: pd.DataFrame) -> pd.Series:
        """
        Detect regimes using K-means clustering on the 8 features.
        
        This is an alternative to rule-based detection that can discover
        new regime patterns automatically.
        """
        logger.info("🔄 Using K-means clustering for regime detection")
        
        try:
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_df)
            
            # Apply K-means with 6 clusters (6 regimes)
            kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_scaled)
            
            # Map clusters to regime names based on cluster characteristics
            cluster_to_regime = self._map_clusters_to_regimes(features_df, cluster_labels)
            
            regimes = pd.Series(
                index=features_df.index,
                data=[cluster_to_regime[label] for label in cluster_labels]
            )
            
            return regimes
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}, falling back to rule-based detection")
            return self._detect_regimes_rule_based(features_df, self.thresholds)
    
    def _map_clusters_to_regimes(self, 
                                features_df: pd.DataFrame, 
                                cluster_labels: np.ndarray) -> Dict[int, str]:
        """Map K-means clusters to regime names based on feature characteristics."""
        
        cluster_characteristics = {}
        
        for cluster_id in range(6):
            cluster_mask = cluster_labels == cluster_id
            cluster_features = features_df[cluster_mask]
            
            if len(cluster_features) > 0:
                # Calculate cluster characteristics
                avg_rv_z_short = cluster_features['rv_z_short'].mean()
                avg_vol_ratio = cluster_features['vol_ratio'].mean()
                avg_drawdown_z = cluster_features['drawdown_z'].mean()
                avg_trend_slope = cluster_features['trend_slope_z'].mean()
                avg_volume_z = cluster_features['volume_z'].mean()
                avg_spread = cluster_features['spread_proxy_z'].mean()
                
                # Determine regime based on characteristics
                if avg_drawdown_z > 1.5 and avg_spread > 1.2:
                    regime = MarketRegime.STRESS_REGIME.value
                elif avg_rv_z_short > 1.2 and avg_vol_ratio > 1.1:
                    regime = MarketRegime.HIGH_VOLATILITY.value
                elif avg_rv_z_short < 0.7 and avg_vol_ratio < 0.9:
                    regime = MarketRegime.LOW_VOLATILITY.value
                elif avg_trend_slope > 0.8 and cluster_features['trend_strength'].mean() > 0.6:
                    regime = MarketRegime.TREND_REGIME.value
                elif avg_volume_z > 1.2 and avg_spread < 0.7:
                    regime = MarketRegime.LIQUIDITY_REGIME.value
                else:
                    regime = MarketRegime.NEUTRAL.value
                
                cluster_characteristics[cluster_id] = regime
            else:
                cluster_characteristics[cluster_id] = MarketRegime.NEUTRAL.value
        
        return cluster_characteristics
    
    def _post_process_regimes(self, regimes: pd.Series) -> pd.Series:
        """
        Post-process regimes for temporal consistency and noise reduction.
        
        Args:
            regimes: Initial regime assignments
            
        Returns:
            Processed regime assignments
        """
        
        # Remove very short regime transitions (noise filtering)
        min_regime_duration = 5  # Minimum periods for regime persistence
        
        processed_regimes = regimes.copy()
        
        # Find regime transitions
        regime_changes = regimes != regimes.shift(1)
        change_indices = np.where(regime_changes)[0]
        
        # Filter short-lived regimes
        for i in range(1, len(change_indices) - 1):
            start_idx = change_indices[i]
            end_idx = change_indices[i + 1]
            
            if end_idx - start_idx < min_regime_duration:
                # This regime is too short, replace with surrounding regime
                if start_idx > 0 and end_idx < len(regimes):
                    surrounding_regime = regimes.iloc[start_idx - 1]
                    processed_regimes.iloc[start_idx:end_idx] = surrounding_regime
        
        return processed_regimes
    
    def _update_adaptive_thresholds(self, features_df: pd.DataFrame) -> RegimeThresholds:
        """
        Update thresholds based on recent market conditions.
        
        This makes the regime detection adaptive to changing market dynamics.
        """
        
        # Calculate recent statistics (last 100 periods)
        recent_features = features_df.tail(100)
        
        # Adaptive thresholds based on recent percentiles
        adaptive_thresholds = RegimeThresholds(
            high_volatility_rv_z_short=np.percentile(recent_features['rv_z_short'], 85),
            high_volatility_vol_ratio=np.percentile(recent_features['vol_ratio'], 80),
            low_volatility_rv_z_short=np.percentile(recent_features['rv_z_short'], 15),
            low_volatility_vol_ratio=np.percentile(recent_features['vol_ratio'], 20),
            stress_drawdown_z=np.percentile(recent_features['drawdown_z'], 95),
            stress_spread_proxy_z=np.percentile(recent_features['spread_proxy_z'], 90),
            trend_slope_z=np.percentile(recent_features['trend_slope_z'], 80),
            trend_strength=np.percentile(recent_features['trend_strength'], 75),
            liquidity_volume_z=np.percentile(recent_features['volume_z'], 80),
            liquidity_spread_proxy_z=np.percentile(recent_features['spread_proxy_z'], 25)
        )
        
        # Store threshold history
        self.threshold_history.append({
            'timestamp': pd.Timestamp.now(),
            'thresholds': adaptive_thresholds
        })
        
        # Keep only last 10 updates
        if len(self.threshold_history) > 10:
            self.threshold_history = self.threshold_history[-10:]
        
        logger.info("📊 Updated adaptive thresholds based on recent market conditions")
        
        return adaptive_thresholds
    
    def get_regime_characteristics(self, regime: MarketRegime) -> RegimeCharacteristics:
        """Get characteristics for a specific regime."""
        return self.regime_characteristics.get(regime, self.regime_characteristics[MarketRegime.NEUTRAL])
    
    def calculate_regime_scores(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate regime scores for each time point.
        
        Returns a DataFrame with scores for each regime, useful for
        understanding regime confidence and transition probabilities.
        """
        
        scores = pd.DataFrame(index=features_df.index)
        
        # Calculate scores for each regime
        for regime in MarketRegime:
            regime_char = self.regime_characteristics[regime]
            
            # Calculate weighted feature scores
            score = 0.0
            
            if regime == MarketRegime.HIGH_VOLATILITY:
                score = (
                    features_df['rv_z_short'] * self.feature_weights['rv_z_short'] +
                    features_df['vol_ratio'] * self.feature_weights['vol_ratio']
                ) / (self.feature_weights['rv_z_short'] + self.feature_weights['vol_ratio'])
                
            elif regime == MarketRegime.LOW_VOLATILITY:
                score = (
                    (2.0 - features_df['rv_z_short']) * self.feature_weights['rv_z_short'] +
                    (2.0 - features_df['vol_ratio']) * self.feature_weights['vol_ratio']
                ) / (self.feature_weights['rv_z_short'] + self.feature_weights['vol_ratio'])
                
            elif regime == MarketRegime.STRESS_REGIME:
                score = (
                    features_df['drawdown_z'] * self.feature_weights['drawdown_z'] +
                    features_df['spread_proxy_z'] * self.feature_weights['spread_proxy_z']
                ) / (self.feature_weights['drawdown_z'] + self.feature_weights['spread_proxy_z'])
                
            elif regime == MarketRegime.TREND_REGIME:
                score = (
                    features_df['trend_slope_z'] * self.feature_weights['trend_slope_z'] +
                    features_df['trend_strength'] * self.feature_weights['trend_strength']
                ) / (self.feature_weights['trend_slope_z'] + self.feature_weights['trend_strength'])
                
            elif regime == MarketRegime.LIQUIDITY_REGIME:
                score = (
                    features_df['volume_z'] * self.feature_weights['volume_z'] +
                    (2.0 - features_df['spread_proxy_z']) * self.feature_weights['spread_proxy_z']
                ) / (self.feature_weights['volume_z'] + self.feature_weights['spread_proxy_z'])
                
            else:  # NEUTRAL
                # Neutral score is inverse of extreme scores
                extreme_scores = [
                    abs(features_df['rv_z_short'] - 1.0),
                    abs(features_df['drawdown_z']),
                    abs(features_df['trend_slope_z'] - 0.5)
                ]
                score = 2.0 - np.mean(extreme_scores, axis=0)
            
            scores[regime.value] = score
        
        return scores
    
    def analyze_regime_transitions(self, regimes: pd.Series) -> Dict[str, Any]:
        """
        Analyze regime transitions and persistence.
        
        Args:
            regimes: Series of regime assignments
            
        Returns:
            Dictionary with transition analysis
        """
        
        transition_analysis = {}
        
        # Transition matrix
        transition_counts = pd.crosstab(
            regimes.shift(1), 
            regimes, 
            margins=True
        )
        
        # Transition probabilities
        transition_probs = transition_counts.div(transition_counts.sum(axis=1), axis=0)
        
        # Regime persistence (average duration)
        persistence = {}
        for regime in regimes.unique():
            regime_mask = regimes == regime
            regime_changes = regime_mask.diff().fillna(False)
            
            # Find regime starts
            starts = np.where(regime_changes & regime_mask)[0]
            
            if len(starts) > 0:
                # Calculate durations
                durations = []
                for start in starts:
                    end = start
                    while end < len(regimes) and regimes.iloc[end] == regime:
                        end += 1
                    durations.append(end - start)
                
                persistence[regime] = {
                    'mean_duration': np.mean(durations),
                    'median_duration': np.median(durations),
                    'max_duration': np.max(durations),
                    'min_duration': np.min(durations)
                }
        
        transition_analysis = {
            'transition_matrix': transition_probs,
            'regime_persistence': persistence,
            'total_transitions': (regimes != regimes.shift(1)).sum(),
            'most_persistent_regime': max(persistence.items(), key=lambda x: x[1]['mean_duration'])[0] if persistence else None,
            'least_persistent_regime': min(persistence.items(), key=lambda x: x[1]['mean_duration'])[0] if persistence else None
        }
        
        return transition_analysis
    
    def generate_regime_report(self, 
                             features_df: pd.DataFrame, 
                             regimes: pd.Series) -> str:
        """Generate comprehensive regime analysis report."""
        
        # Calculate regime scores
        regime_scores = self.calculate_regime_scores(features_df)
        
        # Analyze transitions
        transition_analysis = self.analyze_regime_transitions(regimes)
        
        # Generate report
        report = f"""
# 8-Feature Regime Analysis Report

## Summary
- **Total Periods**: {len(features_df)}
- **Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Detection Method**: {'Clustering' if self.use_clustering else 'Rule-based'}
- **Adaptive Thresholds**: {'Enabled' if self.adaptive_thresholds else 'Disabled'}

## Regime Distribution
{regimes.value_counts().to_string()}

## Regime Characteristics
"""
        
        for regime in MarketRegime:
            if regime.value in regimes.value_counts().index:
                char = self.regime_characteristics[regime]
                count = regimes[regimes == regime.value].sum()
                percentage = (count / len(regimes)) * 100
                
                report += f"""
### {regime.value}
- **Frequency**: {count} ({percentage:.1f}%)
- **Volatility Level**: {char.volatility_level:.2f}
- **Correlation Strength**: {char.correlation_strength:.2f}
- **Parameter Variance**: {char.parameter_variance:.2f}
- **Regime Persistence**: {char.regime_persistence:.2f}
- **Specialist Weights**: {char.specialist_weights}
"""
        
        report += f"""
## Transition Analysis
- **Total Transitions**: {transition_analysis['total_transitions']}
- **Most Persistent**: {transition_analysis['most_persistent_regime']}
- **Least Persistent**: {transition_analysis['least_persistent_regime']}

### Transition Matrix
{transition_analysis['transition_matrix'].round(3).to_string()}

### Regime Persistence
"""
        
        for regime, persistence in transition_analysis['regime_persistence'].items():
            report += f"""
**{regime}**:
- Mean Duration: {persistence['mean_duration']:.1f} periods
- Median Duration: {persistence['median_duration']:.1f} periods
- Max Duration: {persistence['max_duration']} periods
"""
        
        report += f"""
## Feature Statistics by Regime
"""
        
        # Feature statistics by regime
        feature_stats = features_df.groupby(regimes).agg(['mean', 'std', 'min', 'max'])
        report += feature_stats.round(3).to_string()
        
        report += f"""

## Thresholds Used
"""
        
        if self.adaptive_thresholds and self.threshold_history:
            latest_thresholds = self.threshold_history[-1]['thresholds']
            report += f"""
*Adaptive thresholds based on recent market conditions:*
- High Volatility rv_z_short: {latest_thresholds.high_volatility_rv_z_short:.2f}
- High Volatility vol_ratio: {latest_thresholds.high_volatility_vol_ratio:.2f}
- Low Volatility rv_z_short: {latest_thresholds.low_volatility_rv_z_short:.2f}
- Low Volatility vol_ratio: {latest_thresholds.low_volatility_vol_ratio:.2f}
- Stress drawdown_z: {latest_thresholds.stress_drawdown_z:.2f}
- Stress spread_proxy_z: {latest_thresholds.stress_spread_proxy_z:.2f}
- Trend slope_z: {latest_thresholds.trend_slope_z:.2f}
- Trend strength: {latest_thresholds.trend_strength:.2f}
- Liquidity volume_z: {latest_thresholds.liquidity_volume_z:.2f}
- Liquidity spread_proxy_z: {latest_thresholds.liquidity_spread_proxy_z:.2f}
"""
        else:
            report += f"""
*Fixed thresholds:*
- High Volatility rv_z_short: {self.thresholds.high_volatility_rv_z_short:.2f}
- High Volatility vol_ratio: {self.thresholds.high_volatility_vol_ratio:.2f}
- Low Volatility rv_z_short: {self.thresholds.low_volatility_rv_z_short:.2f}
- Low Volatility vol_ratio: {self.thresholds.low_volatility_vol_ratio:.2f}
- Stress drawdown_z: {self.thresholds.stress_drawdown_z:.2f}
- Stress spread_proxy_z: {self.thresholds.stress_spread_proxy_z:.2f}
- Trend slope_z: {self.thresholds.trend_slope_z:.2f}
- Trend strength: {self.thresholds.trend_strength:.2f}
- Liquidity volume_z: {self.thresholds.liquidity_volume_z:.2f}
- Liquidity spread_proxy_z: {self.thresholds.liquidity_spread_proxy_z:.2f}
"""
        
        return report

# Convenience function for quick regime detection
def detect_market_regimes(features_df: pd.DataFrame, 
                         use_clustering: bool = False,
                         adaptive_thresholds: bool = False) -> pd.Series:
    """
    Convenience function for quick regime detection.
    
    Args:
        features_df: DataFrame with 8 core features
        use_clustering: Use K-means clustering
        adaptive_thresholds: Use adaptive thresholds
        
    Returns:
        Series of regime assignments
    """
    detector = EightFeatureRegimeDetector(
        use_clustering=use_clustering,
        adaptive_thresholds=adaptive_thresholds
    )
    
    return detector.detect_regimes(features_df)

if __name__ == "__main__":
    # Example usage
    print("8-Feature Regime Definition System - Ready for integration")
    print("Features:")
    print("- rv_z_short, rv_z_long, vol_ratio (Volatility Regime)")
    print("- volume_z, spread_proxy_z (Liquidity/Participation)")
    print("- trend_slope_z, trend_strength (Trend/Directional)")
    print("- drawdown_z (Stress/Tail Risk)")
    print("\nRegimes:")
    for regime in MarketRegime:
        print(f"- {regime.value}")
