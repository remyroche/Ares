#!/usr/bin/env python3
"""Hierarchical Multi-Scale Regime Detection.

This module implements regime detection at multiple timeframes (5m, 15m, 30m, 1h)
with hierarchical alignment and cross-timeframe validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import asyncio
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class HierarchicalRegimeDetector:
    """Multi-scale regime detection with hierarchical alignment."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Timeframe configuration
        self.timeframes = self.config.get('timeframes', ['5m', '15m', '30m', '1h'])
        self.base_timeframe = self.config.get('base_timeframe', '1m')
        
        # Regime detection parameters
        self.min_regimes = self.config.get('min_regimes', 2)
        self.max_regimes = self.config.get('max_regimes', 6)
        self.regime_stability_threshold = self.config.get('regime_stability_threshold', 0.7)
        
        # Hierarchical alignment parameters
        self.alignment_window = self.config.get('alignment_window', 10)
        self.cross_timeframe_weight = self.config.get('cross_timeframe_weight', 0.3)
        
        # Storage for regime results
        self.regime_hierarchy = {}
        self.aligned_regimes = None
        self.regime_confidence = None
        
    def detect_hierarchical_regimes(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect regimes at multiple timeframes and align them hierarchically."""
        print("🔍 Starting hierarchical multi-scale regime detection...")
        
        # Step 1: Detect regimes at each timeframe
        self.regime_hierarchy = {}
        for timeframe in self.timeframes:
            print(f"  📊 Detecting regimes at {timeframe} timeframe...")
            regimes = self._detect_regimes_at_timeframe(data, timeframe)
            self.regime_hierarchy[timeframe] = regimes
        
        # Step 2: Align regimes across timeframes
        print("  🔗 Aligning regimes across timeframes...")
        aligned_regimes = self._align_hierarchical_regimes(data)
        
        # Step 3: Calculate regime confidence
        print("  📈 Calculating regime confidence...")
        regime_confidence = self._calculate_regime_confidence()
        
        # Step 4: Validate hierarchical consistency
        print("  ✅ Validating hierarchical consistency...")
        validation_results = self._validate_hierarchical_consistency()
        
        return {
            'regime_hierarchy': self.regime_hierarchy,
            'aligned_regimes': aligned_regimes,
            'regime_confidence': regime_confidence,
            'validation_results': validation_results,
            'timeframes': self.timeframes,
            'hierarchical_quality_score': self._calculate_hierarchical_quality_score()
        }
    
    def _detect_regimes_at_timeframe(self, data: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """Detect regimes at a specific timeframe."""
        # Resample data to target timeframe
        resampled_data = self._resample_data(data, timeframe)
        
        if len(resampled_data) < 100:  # Need minimum data points
            return {
                'regimes': np.zeros(len(resampled_data)),
                'n_regimes': 1,
                'confidence': 0.0,
                'timeframe': timeframe
            }
        
        # Extract features for this timeframe
        features = self._extract_timeframe_features(resampled_data, timeframe)
        
        # Determine optimal number of regimes
        optimal_n_regimes = self._optimize_regime_count(features, timeframe)
        
        # Detect regimes using HMM
        regimes = self._detect_regimes_hmm(features, optimal_n_regimes)
        
        # Calculate regime quality metrics
        quality_metrics = self._calculate_regime_quality_metrics(resampled_data, regimes)
        
        return {
            'regimes': regimes,
            'n_regimes': optimal_n_regimes,
            'confidence': quality_metrics['overall_confidence'],
            'timeframe': timeframe,
            'quality_metrics': quality_metrics,
            'features': features,
            'data': resampled_data
        }
    
    def _resample_data(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample data to target timeframe."""
        # Convert timeframe to pandas frequency
        timeframe_map = {
            '5m': '5T',
            '15m': '15T', 
            '30m': '30T',
            '1h': '1H'
        }
        
        freq = timeframe_map.get(timeframe, '1H')
        
        # Resample OHLCV data
        resampled = data.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return resampled
    
    def _extract_timeframe_features(self, data: pd.DataFrame, timeframe: str) -> np.ndarray:
        """Extract features specific to the timeframe."""
        features = []
        
        # Price-based features
        returns = data['close'].pct_change().dropna()
        features.extend([
            returns.mean(),
            returns.std(),
            returns.skew(),
            returns.kurtosis(),
            returns.rolling(20).std().mean()  # Average volatility
        ])
        
        # Volume-based features
        volume = data['volume']
        features.extend([
            volume.mean(),
            volume.std(),
            (volume > volume.rolling(20).mean()).sum() / len(volume)
        ])
        
        # Timeframe-specific features
        timeframe_minutes = self._get_timeframe_minutes(timeframe)
        
        # Intraday patterns (if timeframe is short enough)
        if timeframe_minutes <= 60:  # 1 hour or less
            hour_of_day = data.index.hour
            features.extend([
                hour_of_day.mean(),
                hour_of_day.std()
            ])
        
        # Volatility clustering
        volatility = returns.rolling(20).std()
        features.extend([
            volatility.autocorr(lag=1) if len(volatility) > 1 else 0,
            volatility.autocorr(lag=5) if len(volatility) > 5 else 0
        ])
        
        return np.array(features)
    
    def _get_timeframe_minutes(self, timeframe: str) -> int:
        """Get timeframe in minutes."""
        timeframe_map = {
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60
        }
        return timeframe_map.get(timeframe, 60)
    
    def _optimize_regime_count(self, features: np.ndarray, timeframe: str) -> int:
        """Optimize number of regimes for a specific timeframe."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        if len(features) < 50:
            return 2
        
        # Try different numbers of regimes
        best_score = -1
        best_n_regimes = 2
        
        for n_regimes in range(self.min_regimes, min(self.max_regimes + 1, len(features) // 10)):
            try:
                kmeans = KMeans(n_clusters=n_regimes, random_state=42)
                labels = kmeans.fit_predict(features.reshape(-1, 1))
                
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(features.reshape(-1, 1), labels)
                    if score > best_score:
                        best_score = score
                        best_n_regimes = n_regimes
            except:
                continue
        
        return best_n_regimes
    
    def _detect_regimes_hmm(self, features: np.ndarray, n_regimes: int) -> np.ndarray:
        """Detect regimes using Hidden Markov Model."""
        try:
            from hmmlearn.hmm import GaussianHMM
            
            # Prepare features for HMM
            X = features.reshape(-1, 1)
            
            # Fit HMM
            model = GaussianHMM(
                n_components=n_regimes,
                covariance_type="full",
                random_state=42
            )
            model.fit(X)
            
            # Predict regimes
            regimes = model.predict(X)
            
            return regimes
            
        except Exception as e:
            print(f"Error in HMM regime detection: {e}")
            # Fallback to simple clustering
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_regimes, random_state=42)
            regimes = kmeans.fit_predict(features.reshape(-1, 1))
            return regimes
    
    def _calculate_regime_quality_metrics(self, data: pd.DataFrame, regimes: np.ndarray) -> Dict[str, Any]:
        """Calculate quality metrics for regimes."""
        if len(regimes) == 0:
            return {'overall_confidence': 0.0}
        
        # Regime separation
        regime_separation = self._calculate_regime_separation(data, regimes)
        
        # Regime stability
        regime_stability = self._calculate_regime_stability(regimes)
        
        # Economic significance
        economic_significance = self._calculate_economic_significance(data, regimes)
        
        # Overall confidence
        overall_confidence = (regime_separation + regime_stability + economic_significance) / 3
        
        return {
            'regime_separation': regime_separation,
            'regime_stability': regime_stability,
            'economic_significance': economic_significance,
            'overall_confidence': overall_confidence
        }
    
    def _calculate_regime_separation(self, data: pd.DataFrame, regimes: np.ndarray) -> float:
        """Calculate how well-separated the regimes are."""
        if 'close' not in data.columns:
            return 0.0
        
        returns = data['close'].pct_change().dropna()
        if len(returns) == 0:
            return 0.0
        
        # Calculate mean returns for each regime
        regime_returns = {}
        for regime in np.unique(regimes):
            regime_mask = regimes[:len(returns)] == regime
            if np.sum(regime_mask) > 0:
                regime_returns[regime] = returns[regime_mask].mean()
        
        if len(regime_returns) < 2:
            return 0.0
        
        # Calculate separation as variance of regime means
        regime_means = list(regime_returns.values())
        separation = np.var(regime_means)
        
        return min(separation * 1000, 1.0)  # Scale and cap at 1.0
    
    def _calculate_regime_stability(self, regimes: np.ndarray) -> float:
        """Calculate regime stability (how often regimes change)."""
        if len(regimes) < 2:
            return 1.0
        
        # Count regime changes
        regime_changes = np.sum(np.diff(regimes) != 0)
        stability = 1.0 - (regime_changes / (len(regimes) - 1))
        
        return max(0.0, stability)
    
    def _calculate_economic_significance(self, data: pd.DataFrame, regimes: np.ndarray) -> float:
        """Calculate economic significance of regimes."""
        if 'close' not in data.columns or len(regimes) == 0:
            return 0.0
        
        returns = data['close'].pct_change().dropna()
        if len(returns) == 0:
            return 0.0
        
        # Calculate Sharpe ratio for each regime
        regime_sharpe_ratios = []
        for regime in np.unique(regimes):
            regime_mask = regimes[:len(returns)] == regime
            if np.sum(regime_mask) > 5:  # Need minimum samples
                regime_returns = returns[regime_mask]
                if regime_returns.std() > 0:
                    sharpe = regime_returns.mean() / regime_returns.std()
                    regime_sharpe_ratios.append(sharpe)
        
        if len(regime_sharpe_ratios) < 2:
            return 0.0
        
        # Economic significance as variance of Sharpe ratios
        significance = np.var(regime_sharpe_ratios)
        return min(significance, 1.0)
    
    def _align_hierarchical_regimes(self, data: pd.DataFrame) -> np.ndarray:
        """Align regimes across timeframes hierarchically."""
        if not self.regime_hierarchy:
            return np.zeros(len(data))
        
        # Start with the finest timeframe
        finest_timeframe = min(self.timeframes, key=lambda x: self._get_timeframe_minutes(x))
        base_regimes = self.regime_hierarchy[finest_timeframe]['regimes']
        
        # Align with coarser timeframes
        aligned_regimes = base_regimes.copy()
        
        for timeframe in self.timeframes:
            if timeframe == finest_timeframe:
                continue
            
            # Get regime data for this timeframe
            regime_data = self.regime_hierarchy[timeframe]
            coarse_regimes = regime_data['regimes']
            
            # Align coarse regimes with fine regimes
            aligned_regimes = self._align_timeframe_regimes(
                aligned_regimes, coarse_regimes, finest_timeframe, timeframe
            )
        
        return aligned_regimes
    
    def _align_timeframe_regimes(self, fine_regimes: np.ndarray, coarse_regimes: np.ndarray, 
                                fine_tf: str, coarse_tf: str) -> np.ndarray:
        """Align regimes between two timeframes."""
        # Calculate alignment ratio
        fine_minutes = self._get_timeframe_minutes(fine_tf)
        coarse_minutes = self._get_timeframe_minutes(coarse_tf)
        ratio = coarse_minutes // fine_minutes
        
        if ratio <= 1:
            return fine_regimes
        
        # Align coarse regimes to fine timeframe
        aligned_regimes = fine_regimes.copy()
        
        for i, coarse_regime in enumerate(coarse_regimes):
            start_idx = i * ratio
            end_idx = min((i + 1) * ratio, len(aligned_regimes))
            
            # Use coarse regime for this period if it has higher confidence
            if i < len(self.regime_hierarchy[coarse_tf]['confidence']):
                coarse_confidence = self.regime_hierarchy[coarse_tf]['confidence']
                if coarse_confidence > 0.5:  # Threshold for using coarse regime
                    aligned_regimes[start_idx:end_idx] = coarse_regime
        
        return aligned_regimes
    
    def _calculate_regime_confidence(self) -> np.ndarray:
        """Calculate confidence scores for aligned regimes."""
        if not self.regime_hierarchy:
            return np.array([])
        
        # Initialize confidence array
        max_length = max(len(r['regimes']) for r in self.regime_hierarchy.values())
        confidence = np.zeros(max_length)
        
        # Weight confidence by timeframe (shorter timeframes get higher weight)
        timeframe_weights = {}
        for tf in self.timeframes:
            minutes = self._get_timeframe_minutes(tf)
            timeframe_weights[tf] = 1.0 / minutes  # Inverse relationship
        
        # Normalize weights
        total_weight = sum(timeframe_weights.values())
        for tf in timeframe_weights:
            timeframe_weights[tf] /= total_weight
        
        # Calculate weighted confidence
        for timeframe, regime_data in self.regime_hierarchy.items():
            regimes = regime_data['regimes']
            tf_confidence = regime_data['confidence']
            weight = timeframe_weights[timeframe]
            
            # Add weighted confidence
            for i in range(min(len(confidence), len(regimes))):
                confidence[i] += weight * tf_confidence
        
        return confidence
    
    def _validate_hierarchical_consistency(self) -> Dict[str, Any]:
        """Validate consistency across timeframes."""
        if len(self.regime_hierarchy) < 2:
            return {'consistent': True, 'consistency_score': 1.0}
        
        # Calculate consistency metrics
        consistency_scores = []
        
        for i, tf1 in enumerate(self.timeframes):
            for tf2 in self.timeframes[i+1:]:
                score = self._calculate_timeframe_consistency(tf1, tf2)
                consistency_scores.append(score)
        
        avg_consistency = np.mean(consistency_scores) if consistency_scores else 1.0
        
        return {
            'consistent': avg_consistency > self.regime_stability_threshold,
            'consistency_score': avg_consistency,
            'individual_scores': consistency_scores
        }
    
    def _calculate_timeframe_consistency(self, tf1: str, tf2: str) -> float:
        """Calculate consistency between two timeframes."""
        if tf1 not in self.regime_hierarchy or tf2 not in self.regime_hierarchy:
            return 0.0
        
        regimes1 = self.regime_hierarchy[tf1]['regimes']
        regimes2 = self.regime_hierarchy[tf2]['regimes']
        
        # Calculate correlation between regime sequences
        min_length = min(len(regimes1), len(regimes2))
        if min_length < 10:
            return 0.0
        
        correlation = np.corrcoef(regimes1[:min_length], regimes2[:min_length])[0, 1]
        return max(0.0, correlation) if not np.isnan(correlation) else 0.0
    
    def _calculate_hierarchical_quality_score(self) -> float:
        """Calculate overall quality score for hierarchical regime detection."""
        if not self.regime_hierarchy:
            return 0.0
        
        # Collect quality metrics
        quality_scores = []
        for timeframe, regime_data in self.regime_hierarchy.items():
            quality_metrics = regime_data.get('quality_metrics', {})
            overall_confidence = quality_metrics.get('overall_confidence', 0.0)
            quality_scores.append(overall_confidence)
        
        # Calculate weighted average (shorter timeframes get higher weight)
        timeframe_weights = {}
        for tf in self.timeframes:
            minutes = self._get_timeframe_minutes(tf)
            timeframe_weights[tf] = 1.0 / minutes
        
        total_weight = sum(timeframe_weights.values())
        weighted_score = 0.0
        
        for i, tf in enumerate(self.timeframes):
            if i < len(quality_scores):
                weight = timeframe_weights[tf] / total_weight
                weighted_score += weight * quality_scores[i]
        
        return weighted_score