"""
Economic Validator and Regime Profiler

This module provides comprehensive economic validation and profiling capabilities
for HDBSCAN-based regime discovery, including statistical analysis, regime
characteristics, and trading recommendations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
from sklearn.metrics import silhouette_score
import logging

# Import tprint utilities
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance, tprint_progress, tprint_timer,
    tprint_logged, LogLevel
)

logger = logging.getLogger(__name__)

@dataclass
class RegimeProfile:
    """Economic profile for a discovered regime."""
    regime_id: int
    name: str
    key_stats: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    avg_duration: float
    transitions: Dict[str, int]
    works_best_for: List[str]
    risk_caveats: List[str]
    radar_plot_data: Dict[str, float]
    # Enhanced regime profiling
    regime_type: str  # 'bull', 'bear', 'sideways', 'volatile', 'trending'
    market_conditions: Dict[str, Any]  # Market conditions when this regime occurs
    statistical_significance: Dict[str, float]  # Statistical tests results
    regime_stability: float  # Stability measure
    transition_probabilities: Dict[str, float]  # Probabilities of transitioning to other regimes
    economic_indicators: Dict[str, float]  # Economic indicators during this regime

@dataclass
class EconomicValidationResult:
    """Result of economic validation and profiling."""
    profiles: List[RegimeProfile]
    validation_metrics: Dict[str, float]
    regime_quality_score: float
    trading_recommendations: Dict[str, Any]
    # Enhanced validation results
    regime_correlation_matrix: Optional[np.ndarray] = None
    regime_transition_matrix: Optional[np.ndarray] = None
    market_regime_analysis: Dict[str, Any] = None
    statistical_tests: Dict[str, Any] = None
    regime_persistence_analysis: Dict[str, Any] = None

class EconomicValidator:
    """
    Economic validator and regime profiler for HDBSCAN regime discovery.
    
    Provides comprehensive analysis of discovered regimes including:
    - Statistical characterization of each regime
    - Economic significance validation
    - Trading strategy recommendations
    - Risk assessment and caveats
    """
    
    @tprint_logged(LogLevel.INFO, include_args=True)
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize economic validator.
        
        Args:
            config: Configuration parameters for economic validation
        """
        tprint_info("Initializing EconomicValidator")
        
        self.config = config or {}
        
        # Default configuration
        self.min_regime_duration = self.config.get('min_regime_duration', 10)
        self.confidence_level = self.config.get('confidence_level', 0.95)
        self.min_sharpe_ratio = self.config.get('min_sharpe_ratio', 0.5)
        self.max_volatility = self.config.get('max_volatility', 0.3)
        self.min_regime_samples = self.config.get('min_regime_samples', 20)
        
        tprint_debug(f"Config: min_regime_duration={self.min_regime_duration}, confidence_level={self.confidence_level}")
        tprint_success("✅ EconomicValidator initialized")
        
    @tprint_logged(LogLevel.INFO, include_args=True)
    def validate_and_profile(self, 
                           cluster_labels: np.ndarray,
                           market_data: pd.DataFrame,
                           returns: Optional[np.ndarray] = None,
                           features_df: Optional[pd.DataFrame] = None) -> EconomicValidationResult:
        """
        Validate and profile discovered regimes.
        
        Args:
            cluster_labels: Cluster labels from HDBSCAN
            market_data: Market data with OHLCV columns
            returns: Pre-computed returns (optional)
            features_df: Feature matrix used for clustering (optional)
            
        Returns:
            EconomicValidationResult with profiles and validation metrics
        """
        try:
            tprint_info("🔍 Starting economic validation and profiling...")
            tprint_debug(f"Input shapes: cluster_labels={cluster_labels.shape}, market_data={market_data.shape}")
            
            # Calculate returns if not provided
            if returns is None:
                with tprint_timer("Returns calculation"):
                    returns = self._calculate_returns(market_data)
                    tprint_debug(f"Calculated returns shape: {returns.shape}")
            else:
                tprint_debug(f"Using provided returns shape: {returns.shape}")
            
            # Get unique regimes (excluding noise)
            unique_regimes = np.unique(cluster_labels)
            unique_regimes = unique_regimes[unique_regimes != -1]
            tprint_debug(f"Found {len(unique_regimes)} unique regimes: {unique_regimes}")
            
            if len(unique_regimes) == 0:
                tprint_warning("⚠️ No valid regimes found for economic validation")
                return self._create_empty_result()
            
            # Create regime profiles
            profiles = []
            all_regime_stats = []  # Collect stats for data-driven classification
            with tprint_progress("Creating regime profiles", len(unique_regimes)):
                for i, regime_id in enumerate(unique_regimes):
                    tprint_debug(f"Creating profile for regime {regime_id} ({i+1}/{len(unique_regimes)})")
                    profile = self._create_regime_profile(
                        regime_id, cluster_labels, market_data, returns, features_df
                    )
                    profiles.append(profile)
                    all_regime_stats.append(profile.key_stats)  # Collect for data-driven analysis
                    tprint_progress("Creating regime profiles", i + 1)
            
            # Store all regime stats for data-driven classification
            self._all_regime_stats = all_regime_stats
            
            # Calculate validation metrics
            with tprint_timer("Validation metrics calculation"):
                validation_metrics = self._calculate_validation_metrics(
                    cluster_labels, returns, features_df
                )
                tprint_debug(f"Validation metrics: {validation_metrics}")
            
            # Calculate regime quality score
            with tprint_timer("Quality score calculation"):
                regime_quality_score = self._calculate_regime_quality_score(profiles, validation_metrics)
                tprint_debug(f"Regime quality score: {regime_quality_score:.3f}")
            
            # Generate trading recommendations
            with tprint_timer("Trading recommendations generation"):
                trading_recommendations = self._generate_trading_recommendations(profiles)
                tprint_debug(f"Trading recommendations: {trading_recommendations}")
            
            # Enhanced regime analysis
            with tprint_timer("Enhanced regime analysis"):
                regime_correlation_matrix = self._calculate_regime_correlations(profiles, returns)
                regime_transition_matrix = self._calculate_regime_transitions(cluster_labels)
                market_regime_analysis = self._analyze_market_regime_patterns(profiles, market_data)
                statistical_tests = self._perform_statistical_tests(profiles, returns)
                regime_persistence_analysis = self._analyze_regime_persistence(cluster_labels, profiles)
                tprint_debug(f"Enhanced analysis completed")
            
            tprint_success(f"✅ Economic validation completed. Quality score: {regime_quality_score:.3f}")
            
            return EconomicValidationResult(
                profiles=profiles,
                validation_metrics=validation_metrics,
                regime_quality_score=regime_quality_score,
                trading_recommendations=trading_recommendations,
                regime_correlation_matrix=regime_correlation_matrix,
                regime_transition_matrix=regime_transition_matrix,
                market_regime_analysis=market_regime_analysis,
                statistical_tests=statistical_tests,
                regime_persistence_analysis=regime_persistence_analysis
            )
            
        except Exception as e:
            tprint_error(f"❌ Economic validation failed: {e}")
            return self._create_empty_result()
    
    @tprint_logged(LogLevel.DEBUG, include_args=True)
    def _calculate_returns(self, market_data: pd.DataFrame) -> np.ndarray:
        """Calculate returns from market data."""
        try:
            tprint_debug(f"Calculating returns from market data with shape: {market_data.shape}")
            
            if 'close' in market_data.columns:
                prices = market_data['close'].values
                tprint_debug("Using 'close' column for price data")
            elif 'Close' in market_data.columns:
                prices = market_data['Close'].values
                tprint_debug("Using 'Close' column for price data")
            else:
                # Try to find price column
                price_cols = [col for col in market_data.columns if 'close' in col.lower()]
                if price_cols:
                    prices = market_data[price_cols[0]].values
                    tprint_debug(f"Using '{price_cols[0]}' column for price data")
                else:
                    raise ValueError("No price column found in market data")
            
            returns = np.diff(prices) / prices[:-1]
            result = np.concatenate([[0], returns])  # Add 0 for first period
            
            tprint_debug(f"Returns calculated: shape={result.shape}, mean={np.mean(result):.6f}, std={np.std(result):.6f}")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Returns calculation failed: {e}")
            return np.zeros(len(market_data))
    
    @tprint_logged(LogLevel.DEBUG, include_args=True)
    def _create_regime_profile(self, 
                             regime_id: int,
                             cluster_labels: np.ndarray,
                             market_data: pd.DataFrame,
                             returns: np.ndarray,
                             features_df: Optional[pd.DataFrame]) -> RegimeProfile:
        """Create detailed profile for a specific regime."""
        try:
            tprint_debug(f"Creating profile for regime {regime_id}")
            
            # Get regime mask
            regime_mask = cluster_labels == regime_id
            regime_data = market_data[regime_mask]
            regime_returns = returns[regime_mask]
            
            tprint_debug(f"Regime {regime_id}: {len(regime_data)} samples, {len(regime_returns)} returns")
            
            if len(regime_data) < self.min_regime_samples:
                tprint_warning(f"⚠️ Regime {regime_id} has insufficient samples: {len(regime_data)}")
            
            # Calculate key statistics
            with tprint_timer(f"Key stats calculation for regime {regime_id}"):
                key_stats = self._calculate_key_stats(regime_returns, regime_data)
                tprint_debug(f"Key stats for regime {regime_id}: {key_stats}")
            
            # Calculate confidence intervals
            with tprint_timer(f"Confidence intervals for regime {regime_id}"):
                confidence_intervals = self._calculate_confidence_intervals(regime_returns, key_stats)
                tprint_debug(f"Confidence intervals for regime {regime_id}: {confidence_intervals}")
            
            # Calculate average duration
            with tprint_timer(f"Duration calculation for regime {regime_id}"):
                avg_duration = self._calculate_avg_duration(cluster_labels, regime_id)
                tprint_debug(f"Average duration for regime {regime_id}: {avg_duration:.2f}")
            
            # Calculate transitions
            with tprint_timer(f"Transitions calculation for regime {regime_id}"):
                transitions = self._calculate_transitions(cluster_labels, regime_id)
                tprint_debug(f"Transitions for regime {regime_id}: {transitions}")
            
            # Generate trading recommendations
            with tprint_timer(f"Regime analysis for regime {regime_id}"):
                works_best_for, risk_caveats = self._analyze_regime_characteristics(
                    key_stats, regime_data, regime_returns
                )
                tprint_debug(f"Works best for regime {regime_id}: {works_best_for}")
                tprint_debug(f"Risk caveats for regime {regime_id}: {risk_caveats}")
            
            # Create radar plot data
            with tprint_timer(f"Radar plot data for regime {regime_id}"):
                radar_plot_data = self._create_radar_plot_data(key_stats, regime_data)
                tprint_debug(f"Radar plot data for regime {regime_id}: {radar_plot_data}")
            
            # Generate regime name
            regime_name = self._generate_regime_name(key_stats, regime_id)
            tprint_debug(f"Generated name for regime {regime_id}: {regime_name}")
            
            # Enhanced regime profiling
            with tprint_timer(f"Enhanced profiling for regime {regime_id}"):
                regime_type = self._classify_regime_type(key_stats, regime_data)
                market_conditions = self._analyze_market_conditions(regime_data, key_stats)
                statistical_significance = self._calculate_statistical_significance(regime_returns, key_stats)
                regime_stability = self._calculate_regime_stability_score(cluster_labels, regime_id)
                transition_probabilities = self._calculate_transition_probabilities(cluster_labels, regime_id)
                economic_indicators = self._calculate_economic_indicators(regime_data, key_stats)
                tprint_debug(f"Enhanced profiling completed for regime {regime_id}")
            
            tprint_success(f"✅ Profile created for regime {regime_id}: {regime_name}")
            
            return RegimeProfile(
                regime_id=regime_id,
                name=regime_name,
                key_stats=key_stats,
                confidence_intervals=confidence_intervals,
                avg_duration=avg_duration,
                transitions=transitions,
                works_best_for=works_best_for,
                risk_caveats=risk_caveats,
                radar_plot_data=radar_plot_data,
                regime_type=regime_type,
                market_conditions=market_conditions,
                statistical_significance=statistical_significance,
                regime_stability=regime_stability,
                transition_probabilities=transition_probabilities,
                economic_indicators=economic_indicators
            )
            
        except Exception as e:
            tprint_error(f"❌ Failed to create profile for regime {regime_id}: {e}")
            return self._create_empty_profile(regime_id)
    
    @tprint_logged(LogLevel.DEBUG, include_args=True)
    def _calculate_key_stats(self, returns: np.ndarray, regime_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate key statistical measures for a regime."""
        try:
            tprint_debug(f"Calculating key stats for {len(returns)} returns")
            
            # Basic return statistics
            avg_return = np.mean(returns)
            volatility = np.std(returns)
            sharpe_ratio = avg_return / volatility if volatility > 0 else 0
            
            tprint_debug(f"Basic stats: avg_return={avg_return:.6f}, volatility={volatility:.6f}, sharpe={sharpe_ratio:.3f}")
            
            # Additional statistics
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)
            max_drawdown = self._calculate_max_drawdown(returns)
            
            tprint_debug(f"Distribution stats: skewness={skewness:.3f}, kurtosis={kurtosis:.3f}, max_dd={max_drawdown:.3f}")
            
            # Volume statistics (if available)
            volume_stats = self._calculate_volume_stats(regime_data)
            tprint_debug(f"Volume stats: {volume_stats}")
            
            # Volatility clustering
            volatility_clustering = self._calculate_volatility_clustering(returns)
            tprint_debug(f"Volatility clustering: {volatility_clustering:.3f}")
            
            result = {
                'avg_return': avg_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'max_drawdown': max_drawdown,
                'volatility_clustering': volatility_clustering,
                **volume_stats
            }
            
            tprint_debug(f"Key stats calculation completed: {len(result)} metrics")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Key stats calculation failed: {e}")
            return {
                'avg_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0,
                'max_drawdown': 0.0,
                'volatility_clustering': 0.0
            }
    
    def _calculate_confidence_intervals(self, 
                                     returns: np.ndarray, 
                                     key_stats: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for key statistics."""
        try:
            confidence_intervals = {}
            alpha = 1 - self.confidence_level
            
            # Return confidence interval
            if len(returns) > 1:
                ci_low, ci_high = stats.t.interval(
                    self.confidence_level, 
                    len(returns) - 1,
                    loc=key_stats['avg_return'],
                    scale=stats.sem(returns)
                )
                confidence_intervals['avg_return'] = (ci_low, ci_high)
            
            # Volatility confidence interval (using chi-square distribution)
            if len(returns) > 1:
                n = len(returns)
                var = key_stats['volatility'] ** 2
                chi2_low = stats.chi2.ppf(alpha/2, n-1)
                chi2_high = stats.chi2.ppf(1-alpha/2, n-1)
                
                vol_ci_low = np.sqrt((n-1) * var / chi2_high)
                vol_ci_high = np.sqrt((n-1) * var / chi2_low)
                confidence_intervals['volatility'] = (vol_ci_low, vol_ci_high)
            
            return confidence_intervals
            
        except Exception as e:
            logger.error(f"❌ Confidence interval calculation failed: {e}")
            return {}
    
    def _calculate_avg_duration(self, cluster_labels: np.ndarray, regime_id: int) -> float:
        """Calculate average duration of regime periods."""
        try:
            regime_mask = cluster_labels == regime_id
            regime_changes = np.diff(np.concatenate([[False], regime_mask, [False]]).astype(int))
            
            # Find start and end points
            starts = np.where(regime_changes == 1)[0]
            ends = np.where(regime_changes == -1)[0]
            
            if len(starts) == 0:
                return 0.0
            
            # Handle case where regime continues to end
            if len(ends) < len(starts):
                ends = np.concatenate([ends, [len(cluster_labels)]])
            
            durations = ends - starts
            return np.mean(durations) if len(durations) > 0 else 0.0
            
        except Exception as e:
            logger.error(f"❌ Duration calculation failed: {e}")
            return 0.0
    
    def _calculate_transitions(self, cluster_labels: np.ndarray, regime_id: int) -> Dict[str, int]:
        """Calculate transition statistics for a regime."""
        try:
            regime_mask = cluster_labels == regime_id
            transitions = {
                'from_other': 0,
                'to_other': 0,
                'self_transitions': 0
            }
            
            # Find regime changes
            regime_changes = np.diff(np.concatenate([[False], regime_mask, [False]]).astype(int))
            
            # Count transitions into this regime
            transitions['from_other'] = np.sum(regime_changes == 1)
            
            # Count transitions out of this regime
            transitions['to_other'] = np.sum(regime_changes == -1)
            
            # Count self-transitions (regime appears and disappears in same period)
            # This is a simplified calculation
            transitions['self_transitions'] = max(0, transitions['from_other'] - transitions['to_other'])
            
            return transitions
            
        except Exception as e:
            logger.error(f"❌ Transition calculation failed: {e}")
            return {'from_other': 0, 'to_other': 0, 'self_transitions': 0}
    
    def _analyze_regime_characteristics(self, 
                                      key_stats: Dict[str, float],
                                      regime_data: pd.DataFrame,
                                      returns: np.ndarray) -> Tuple[List[str], List[str]]:
        """Analyze regime characteristics for trading recommendations."""
        try:
            works_best_for = []
            risk_caveats = []
            
            # Analyze return characteristics
            if key_stats['sharpe_ratio'] > 1.0:
                works_best_for.append("Trend following strategies")
            elif key_stats['sharpe_ratio'] < 0:
                works_best_for.append("Mean reversion strategies")
            
            # Analyze volatility
            if key_stats['volatility'] > 0.2:
                risk_caveats.append("High volatility regime")
                works_best_for.append("Volatility trading")
            elif key_stats['volatility'] < 0.05:
                works_best_for.append("Low-risk strategies")
            
            # Analyze skewness
            if key_stats['skewness'] > 1.0:
                risk_caveats.append("Positive skew - fat tail risk")
            elif key_stats['skewness'] < -1.0:
                risk_caveats.append("Negative skew - crash risk")
            
            # Analyze kurtosis
            if key_stats['kurtosis'] > 3.0:
                risk_caveats.append("High kurtosis - extreme events likely")
            
            # Analyze drawdown
            if key_stats['max_drawdown'] > 0.1:
                risk_caveats.append("High maximum drawdown")
            
            # Default recommendations
            if not works_best_for:
                works_best_for.append("General trading strategies")
            
            return works_best_for, risk_caveats
            
        except Exception as e:
            logger.error(f"❌ Regime analysis failed: {e}")
            return ["General trading strategies"], ["Analysis failed"]
    
    def _create_radar_plot_data(self, 
                              key_stats: Dict[str, float], 
                              regime_data: pd.DataFrame) -> Dict[str, float]:
        """Create data for radar plot visualization."""
        try:
            # Normalize statistics to 0-1 scale for radar plot
            radar_data = {}
            
            # Normalize return (assuming -0.1 to 0.1 range)
            radar_data['return'] = np.clip((key_stats['avg_return'] + 0.1) / 0.2, 0, 1)
            
            # Normalize volatility (assuming 0 to 0.3 range)
            radar_data['volatility'] = np.clip(key_stats['volatility'] / 0.3, 0, 1)
            
            # Normalize Sharpe ratio (assuming -2 to 2 range)
            radar_data['sharpe'] = np.clip((key_stats['sharpe_ratio'] + 2) / 4, 0, 1)
            
            # Normalize skewness (assuming -3 to 3 range)
            radar_data['skewness'] = np.clip((key_stats['skewness'] + 3) / 6, 0, 1)
            
            # Normalize kurtosis (assuming 0 to 10 range)
            radar_data['kurtosis'] = np.clip(key_stats['kurtosis'] / 10, 0, 1)
            
            return radar_data
            
        except Exception as e:
            logger.error(f"❌ Radar plot data creation failed: {e}")
            return {}
    
    def _generate_regime_name(self, key_stats: Dict[str, float], regime_id: int) -> str:
        """Generate data-driven descriptive name for regime based on characteristics."""
        try:
            # Get all regime stats for comparison
            all_regime_stats = getattr(self, '_all_regime_stats', [])
            
            # Calculate data-driven thresholds
            sharpe_thresholds = self._calculate_sharpe_thresholds(all_regime_stats)
            volatility_thresholds = self._calculate_volatility_thresholds(all_regime_stats)
            return_thresholds = self._calculate_return_thresholds(all_regime_stats)
            
            # Data-driven regime naming
            sharpe = key_stats['sharpe_ratio']
            volatility = key_stats['volatility']
            avg_return = key_stats['avg_return']
            
            # Multi-dimensional classification for naming
            if sharpe > sharpe_thresholds['high'] and volatility < volatility_thresholds['low']:
                return f"Elite_{regime_id}"
            elif sharpe > sharpe_thresholds['high'] and volatility < volatility_thresholds['high']:
                return f"High_Quality_{regime_id}"
            elif volatility > volatility_thresholds['high']:
                return f"Volatile_{regime_id}"
            elif avg_return > return_thresholds['high']:
                return f"Bullish_{regime_id}"
            elif avg_return < -return_thresholds['high']:
                return f"Bearish_{regime_id}"
            elif volatility < volatility_thresholds['low'] and abs(avg_return) < return_thresholds['low']:
                return f"Stable_{regime_id}"
            elif sharpe < sharpe_thresholds['low']:
                return f"Poor_Performance_{regime_id}"
            else:
                return f"Neutral_{regime_id}"
                
        except Exception as e:
            logger.error(f"❌ Regime naming failed: {e}")
            return f"Regime_{regime_id}"
    
    def _calculate_volume_stats(self, regime_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume-related statistics."""
        try:
            volume_stats = {}
            
            # Find volume column
            volume_cols = [col for col in regime_data.columns if 'volume' in col.lower()]
            if volume_cols:
                volume = regime_data[volume_cols[0]].values
                volume_stats['avg_volume'] = np.mean(volume)
                volume_stats['volume_volatility'] = np.std(volume)
            else:
                volume_stats['avg_volume'] = 0.0
                volume_stats['volume_volatility'] = 0.0
            
            return volume_stats
            
        except Exception as e:
            logger.error(f"❌ Volume stats calculation failed: {e}")
            return {'avg_volume': 0.0, 'volume_volatility': 0.0}
    
    def _calculate_volatility_clustering(self, returns: np.ndarray) -> float:
        """Calculate volatility clustering measure."""
        try:
            if len(returns) < 10:
                return 0.0
            
            # Calculate rolling volatility
            window = min(10, len(returns) // 4)
            rolling_vol = pd.Series(returns).rolling(window=window).std().dropna()
            
            if len(rolling_vol) < 2:
                return 0.0
            
            # Calculate autocorrelation of volatility
            autocorr = rolling_vol.autocorr(lag=1)
            return autocorr if not np.isnan(autocorr) else 0.0
            
        except Exception as e:
            logger.error(f"❌ Volatility clustering calculation failed: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        try:
            if len(returns) == 0:
                return 0.0
            
            # Calculate cumulative returns
            cumulative = np.cumprod(1 + returns)
            
            # Calculate running maximum
            running_max = np.maximum.accumulate(cumulative)
            
            # Calculate drawdown
            drawdown = (cumulative - running_max) / running_max
            
            return np.min(drawdown)
            
        except Exception as e:
            logger.error(f"❌ Max drawdown calculation failed: {e}")
            return 0.0
    
    @tprint_logged(LogLevel.DEBUG, include_args=True)
    def _calculate_validation_metrics(self, 
                                    cluster_labels: np.ndarray,
                                    returns: np.ndarray,
                                    features_df: Optional[pd.DataFrame]) -> Dict[str, float]:
        """Calculate validation metrics for regime discovery."""
        try:
            tprint_debug(f"Calculating validation metrics for {len(cluster_labels)} labels")
            
            metrics = {}
            
            # Basic cluster metrics
            unique_regimes = np.unique(cluster_labels)
            unique_regimes = unique_regimes[unique_regimes != -1]
            
            metrics['n_regimes'] = len(unique_regimes)
            metrics['noise_ratio'] = np.sum(cluster_labels == -1) / len(cluster_labels)
            
            tprint_debug(f"Basic metrics: n_regimes={metrics['n_regimes']}, noise_ratio={metrics['noise_ratio']:.3f}")
            
            # Calculate silhouette score if features available
            if features_df is not None and len(unique_regimes) > 1:
                tprint_debug("Calculating silhouette score with features")
                valid_mask = cluster_labels != -1
                if valid_mask.sum() > 1:
                    valid_labels = cluster_labels[valid_mask]
                    valid_features = features_df[valid_mask]
                    
                    if len(np.unique(valid_labels)) > 1:
                        with tprint_timer("Silhouette score calculation"):
                            metrics['silhouette_score'] = silhouette_score(valid_features, valid_labels)
                        tprint_debug(f"Silhouette score: {metrics['silhouette_score']:.3f}")
                    else:
                        metrics['silhouette_score'] = 0.0
                        tprint_debug("Silhouette score: 0.0 (only one cluster)")
                else:
                    metrics['silhouette_score'] = 0.0
                    tprint_debug("Silhouette score: 0.0 (insufficient valid samples)")
            else:
                metrics['silhouette_score'] = 0.0
                tprint_debug("Silhouette score: 0.0 (no features or insufficient regimes)")
            
            # Calculate regime stability
            with tprint_timer("Regime stability calculation"):
                metrics['regime_stability'] = self._calculate_regime_stability(cluster_labels)
                tprint_debug(f"Regime stability: {metrics['regime_stability']:.3f}")
            
            tprint_debug(f"Validation metrics completed: {metrics}")
            return metrics
            
        except Exception as e:
            tprint_error(f"❌ Validation metrics calculation failed: {e}")
            return {'n_regimes': 0, 'noise_ratio': 1.0, 'silhouette_score': 0.0, 'regime_stability': 0.0}
    
    def _calculate_regime_stability(self, cluster_labels: np.ndarray) -> float:
        """Calculate regime stability measure."""
        try:
            # Count regime changes
            regime_changes = np.sum(np.diff(cluster_labels) != 0)
            
            # Calculate stability as inverse of change frequency
            if len(cluster_labels) > 1:
                stability = 1.0 - (regime_changes / (len(cluster_labels) - 1))
                return max(0.0, stability)
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"❌ Regime stability calculation failed: {e}")
            return 0.0
    
    def _calculate_regime_quality_score(self, 
                                      profiles: List[RegimeProfile],
                                      validation_metrics: Dict[str, float]) -> float:
        """Calculate overall regime quality score."""
        try:
            if not profiles:
                return 0.0
            
            # Base score from validation metrics
            base_score = validation_metrics.get('silhouette_score', 0.0)
            
            # Adjust for regime characteristics
            regime_scores = []
            for profile in profiles:
                regime_score = 0.0
                
                # Reward good Sharpe ratios
                if profile.key_stats['sharpe_ratio'] > 0.5:
                    regime_score += 0.2
                
                # Reward reasonable volatility
                if 0.05 < profile.key_stats['volatility'] < 0.2:
                    regime_score += 0.2
                
                # Reward reasonable duration
                if profile.avg_duration > 10:
                    regime_score += 0.2
                
                # Penalize extreme characteristics
                if abs(profile.key_stats['skewness']) > 2:
                    regime_score -= 0.1
                
                if profile.key_stats['kurtosis'] > 5:
                    regime_score -= 0.1
                
                regime_scores.append(regime_score)
            
            # Combine scores
            avg_regime_score = np.mean(regime_scores) if regime_scores else 0.0
            quality_score = (base_score + avg_regime_score) / 2
            
            return np.clip(quality_score, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"❌ Quality score calculation failed: {e}")
            return 0.0
    
    def _generate_trading_recommendations(self, profiles: List[RegimeProfile]) -> Dict[str, Any]:
        """Generate data-driven trading recommendations based on regime profiles."""
        try:
            recommendations = {
                'best_regimes': [],
                'avoid_regimes': [],
                'overall_strategy': 'Conservative',
                'risk_level': 'Medium'
            }
            
            if not profiles:
                return recommendations
            
            # Get all regime stats for data-driven analysis
            all_regime_stats = getattr(self, '_all_regime_stats', [])
            
            # Calculate data-driven thresholds
            sharpe_ratios = [p.key_stats['sharpe_ratio'] for p in profiles]
            volatilities = [p.key_stats['volatility'] for p in profiles]
            returns = [p.key_stats['avg_return'] for p in profiles]
            
            # Data-driven regime classification
            sharpe_thresholds = self._calculate_sharpe_thresholds(all_regime_stats)
            volatility_thresholds = self._calculate_volatility_thresholds(all_regime_stats)
            return_thresholds = self._calculate_return_thresholds(all_regime_stats)
            
            # Analyze each regime with data-driven thresholds
            for profile in profiles:
                sharpe = profile.key_stats['sharpe_ratio']
                volatility = profile.key_stats['volatility']
                avg_return = profile.key_stats['avg_return']
                
                # Multi-dimensional data-driven classification
                if sharpe > sharpe_thresholds['high'] and volatility < volatility_thresholds['high']:
                    recommendations['best_regimes'].append(profile.regime_id)
                elif sharpe < sharpe_thresholds['low'] or volatility > volatility_thresholds['high']:
                    recommendations['avoid_regimes'].append(profile.regime_id)
            
            # Data-driven overall strategy
            avg_sharpe = np.mean(sharpe_ratios)
            avg_vol = np.mean(volatilities)
            sharpe_std = np.std(sharpe_ratios)
            
            if avg_sharpe > sharpe_thresholds['high'] and sharpe_std < np.std([s.get('sharpe_ratio', 0) for s in all_regime_stats]) * 0.5:
                recommendations['overall_strategy'] = 'Aggressive'
                recommendations['risk_level'] = 'Low'
            elif avg_sharpe > sharpe_thresholds['low'] and avg_vol < volatility_thresholds['high']:
                recommendations['overall_strategy'] = 'Moderate'
                recommendations['risk_level'] = 'Medium'
            elif avg_sharpe < sharpe_thresholds['low'] and avg_vol > volatility_thresholds['high']:
                recommendations['overall_strategy'] = 'Defensive'
                recommendations['risk_level'] = 'High'
            else:
                recommendations['overall_strategy'] = 'Conservative'
                recommendations['risk_level'] = 'Medium'
            
            # Additional data-driven insights
            recommendations['regime_diversity'] = len(set([p.regime_type for p in profiles]))
            recommendations['performance_consistency'] = 1.0 - (np.std(volatilities) / (avg_vol + 1e-10))
            recommendations['data_driven_thresholds'] = {
                'sharpe_thresholds': sharpe_thresholds,
                'volatility_thresholds': volatility_thresholds,
                'return_thresholds': return_thresholds
            }
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Trading recommendations generation failed: {e}")
            return {'best_regimes': [], 'avoid_regimes': [], 'overall_strategy': 'Conservative', 'risk_level': 'Medium'}
    
    def _create_empty_profile(self, regime_id: int) -> RegimeProfile:
        """Create empty profile for error cases."""
        return RegimeProfile(
            regime_id=regime_id,
            name=f"Regime_{regime_id}",
            key_stats={'avg_return': 0.0, 'volatility': 0.0, 'sharpe_ratio': 0.0},
            confidence_intervals={},
            avg_duration=0.0,
            transitions={'from_other': 0, 'to_other': 0, 'self_transitions': 0},
            works_best_for=[],
            risk_caveats=[],
            radar_plot_data={}
        )
    
    def _create_empty_result(self) -> EconomicValidationResult:
        """Create empty result for error cases."""
        return EconomicValidationResult(
            profiles=[],
            validation_metrics={'n_regimes': 0, 'noise_ratio': 1.0, 'silhouette_score': 0.0, 'regime_stability': 0.0},
            regime_quality_score=0.0,
            trading_recommendations={'best_regimes': [], 'avoid_regimes': [], 'overall_strategy': 'Conservative', 'risk_level': 'Medium'},
            regime_correlation_matrix=None,
            regime_transition_matrix=None,
            market_regime_analysis=None,
            statistical_tests=None,
            regime_persistence_analysis=None
        )
    
    def _classify_regime_type(self, key_stats: Dict[str, float], regime_data: pd.DataFrame) -> str:
        """Classify regime type based on data-driven statistical characteristics."""
        try:
            # Get all regime data for comparison
            all_regime_stats = self._get_all_regime_statistics()
            
            # Determine regime type based on relative characteristics
            avg_return = key_stats['avg_return']
            volatility = key_stats['volatility']
            sharpe_ratio = key_stats['sharpe_ratio']
            skewness = key_stats['skewness']
            
            # Calculate data-driven thresholds
            volatility_thresholds = self._calculate_volatility_thresholds(all_regime_stats)
            return_thresholds = self._calculate_return_thresholds(all_regime_stats)
            sharpe_thresholds = self._calculate_sharpe_thresholds(all_regime_stats)
            
            # Data-driven classification
            if volatility > volatility_thresholds['high']:
                return 'volatile'
            elif volatility < volatility_thresholds['low']:
                if abs(avg_return) < return_thresholds['low']:
                    return 'sideways'
                else:
                    return 'trending'
            else:
                # Medium volatility - check return characteristics
                if avg_return > return_thresholds['high']:
                    return 'bull' if sharpe_ratio > sharpe_thresholds['high'] else 'trending'
                elif avg_return < return_thresholds['low']:
                    return 'bear' if sharpe_ratio < sharpe_thresholds['low'] else 'trending'
                else:
                    return 'trending'
            
        except Exception as e:
            logger.error(f"❌ Regime type classification failed: {e}")
            return 'unknown'
    
    def _get_all_regime_statistics(self) -> List[Dict[str, float]]:
        """Get statistics for all regimes to enable data-driven classification."""
        try:
            # This would be populated during the validation process
            # For now, return empty list - will be enhanced when called from main validation
            return getattr(self, '_all_regime_stats', [])
        except Exception as e:
            logger.debug(f"Getting all regime statistics failed: {e}")
            return []
    
    def _calculate_volatility_thresholds(self, all_regime_stats: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate data-driven volatility thresholds."""
        try:
            if not all_regime_stats:
                # Default thresholds if no data available
                return {'low': 0.05, 'high': 0.15}
            
            volatilities = [stats.get('volatility', 0) for stats in all_regime_stats]
            volatilities = [v for v in volatilities if v > 0]  # Remove invalid values
            
            if len(volatilities) < 2:
                return {'low': 0.05, 'high': 0.15}
            
            # Calculate percentiles for data-driven thresholds
            low_threshold = np.percentile(volatilities, 33)  # Bottom third
            high_threshold = np.percentile(volatilities, 67)  # Top third
            
            return {'low': low_threshold, 'high': high_threshold}
            
        except Exception as e:
            logger.debug(f"Volatility threshold calculation failed: {e}")
            return {'low': 0.05, 'high': 0.15}
    
    def _calculate_return_thresholds(self, all_regime_stats: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate data-driven return thresholds."""
        try:
            if not all_regime_stats:
                return {'low': 0.005, 'high': 0.01}
            
            returns = [stats.get('avg_return', 0) for stats in all_regime_stats]
            abs_returns = [abs(r) for r in returns]
            
            if len(abs_returns) < 2:
                return {'low': 0.005, 'high': 0.01}
            
            # Calculate percentiles for data-driven thresholds
            low_threshold = np.percentile(abs_returns, 33)
            high_threshold = np.percentile(abs_returns, 67)
            
            return {'low': low_threshold, 'high': high_threshold}
            
        except Exception as e:
            logger.debug(f"Return threshold calculation failed: {e}")
            return {'low': 0.005, 'high': 0.01}
    
    def _calculate_sharpe_thresholds(self, all_regime_stats: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate data-driven Sharpe ratio thresholds."""
        try:
            if not all_regime_stats:
                return {'low': -0.5, 'high': 0.5}
            
            sharpe_ratios = [stats.get('sharpe_ratio', 0) for stats in all_regime_stats]
            sharpe_ratios = [s for s in sharpe_ratios if not np.isnan(s)]  # Remove NaN values
            
            if len(sharpe_ratios) < 2:
                return {'low': -0.5, 'high': 0.5}
            
            # Calculate percentiles for data-driven thresholds
            low_threshold = np.percentile(sharpe_ratios, 33)
            high_threshold = np.percentile(sharpe_ratios, 67)
            
            return {'low': low_threshold, 'high': high_threshold}
            
        except Exception as e:
            logger.debug(f"Sharpe threshold calculation failed: {e}")
            return {'low': -0.5, 'high': 0.5}
    
    def _analyze_market_conditions(self, regime_data: pd.DataFrame, key_stats: Dict[str, float]) -> Dict[str, Any]:
        """Analyze market conditions during this regime using data-driven thresholds."""
        try:
            conditions = {}
            
            # Get all regime stats for comparison
            all_regime_stats = getattr(self, '_all_regime_stats', [])
            
            # Volume analysis
            volume_cols = [col for col in regime_data.columns if 'volume' in col.lower()]
            if volume_cols:
                volume = regime_data[volume_cols[0]].values
                conditions['avg_volume'] = np.mean(volume)
                
                # Data-driven volume trend analysis
                if len(volume) > 4:
                    recent_volume = np.mean(volume[-len(volume)//4:])
                    early_volume = np.mean(volume[:len(volume)//4])
                    volume_change = (recent_volume - early_volume) / early_volume if early_volume > 0 else 0
                    conditions['volume_trend'] = 'increasing' if volume_change > 0.1 else 'decreasing' if volume_change < -0.1 else 'stable'
                    conditions['volume_change_pct'] = volume_change
                else:
                    conditions['volume_trend'] = 'unknown'
                    conditions['volume_change_pct'] = 0.0
            else:
                conditions['avg_volume'] = 0.0
                conditions['volume_trend'] = 'unknown'
                conditions['volume_change_pct'] = 0.0
            
            # Price range analysis
            if 'high' in regime_data.columns and 'low' in regime_data.columns:
                price_range = (regime_data['high'] - regime_data['low']).mean()
                conditions['avg_price_range'] = price_range
                
                # Data-driven price volatility classification
                if len(all_regime_stats) > 1:
                    price_ranges = []
                    for stats in all_regime_stats:
                        if 'price_range' in stats:
                            price_ranges.append(stats['price_range'])
                    
                    if price_ranges:
                        range_threshold = np.percentile(price_ranges, 67)
                        conditions['price_volatility'] = 'high' if price_range > range_threshold else 'low'
                    else:
                        conditions['price_volatility'] = 'high' if price_range > regime_data['close'].mean() * 0.02 else 'low'
                else:
                    conditions['price_volatility'] = 'high' if price_range > regime_data['close'].mean() * 0.02 else 'low'
            else:
                conditions['avg_price_range'] = 0.0
                conditions['price_volatility'] = 'unknown'
            
            # Data-driven regime characteristics
            volatility_thresholds = self._calculate_volatility_thresholds(all_regime_stats)
            return_thresholds = self._calculate_return_thresholds(all_regime_stats)
            sharpe_thresholds = self._calculate_sharpe_thresholds(all_regime_stats)
            
            conditions['regime_volatility'] = 'high' if key_stats['volatility'] > volatility_thresholds['high'] else 'low'
            conditions['regime_trend'] = 'bullish' if key_stats['avg_return'] > return_thresholds['high'] else 'bearish' if key_stats['avg_return'] < -return_thresholds['high'] else 'neutral'
            
            # Data-driven quality assessment
            if key_stats['sharpe_ratio'] > sharpe_thresholds['high']:
                conditions['regime_quality'] = 'high'
            elif key_stats['sharpe_ratio'] < sharpe_thresholds['low']:
                conditions['regime_quality'] = 'low'
            else:
                conditions['regime_quality'] = 'medium'
            
            # Additional data-driven metrics
            conditions['relative_volatility'] = key_stats['volatility'] / np.mean([s.get('volatility', 0) for s in all_regime_stats]) if all_regime_stats else 1.0
            conditions['relative_return'] = key_stats['avg_return'] / np.mean([s.get('avg_return', 0) for s in all_regime_stats]) if all_regime_stats else 1.0
            conditions['relative_sharpe'] = key_stats['sharpe_ratio'] / np.mean([s.get('sharpe_ratio', 0) for s in all_regime_stats]) if all_regime_stats else 1.0
            
            return conditions
            
        except Exception as e:
            logger.error(f"❌ Market conditions analysis failed: {e}")
            return {}
    
    def _calculate_statistical_significance(self, returns: np.ndarray, key_stats: Dict[str, float]) -> Dict[str, float]:
        """Calculate statistical significance tests for regime characteristics."""
        try:
            significance = {}
            
            # T-test for mean return
            if len(returns) > 1:
                t_stat, p_value = stats.ttest_1samp(returns, 0)
                significance['return_t_stat'] = t_stat
                significance['return_p_value'] = p_value
                significance['return_significant'] = p_value < 0.05
            else:
                significance['return_t_stat'] = 0.0
                significance['return_p_value'] = 1.0
                significance['return_significant'] = False
            
            # Normality test (Shapiro-Wilk for small samples, Kolmogorov-Smirnov for larger)
            if len(returns) >= 3:
                if len(returns) <= 5000:
                    try:
                        shapiro_stat, shapiro_p = stats.shapiro(returns)
                        significance['normality_test'] = 'shapiro'
                        significance['normality_stat'] = shapiro_stat
                        significance['normality_p_value'] = shapiro_p
                    except:
                        significance['normality_test'] = 'failed'
                        significance['normality_stat'] = 0.0
                        significance['normality_p_value'] = 1.0
                else:
                    ks_stat, ks_p = stats.kstest(returns, 'norm', args=(np.mean(returns), np.std(returns)))
                    significance['normality_test'] = 'ks'
                    significance['normality_stat'] = ks_stat
                    significance['normality_p_value'] = ks_p
                
                significance['is_normal'] = significance['normality_p_value'] > 0.05
            else:
                significance['normality_test'] = 'insufficient_data'
                significance['normality_stat'] = 0.0
                significance['normality_p_value'] = 1.0
                significance['is_normal'] = False
            
            # Autocorrelation test
            if len(returns) > 10:
                try:
                    from statsmodels.stats.diagnostic import acorr_ljungbox
                    lb_stat, lb_p = acorr_ljungbox(returns, lags=1, return_df=False)
                    significance['autocorr_lb_stat'] = lb_stat[0]
                    significance['autocorr_lb_p_value'] = lb_p[0]
                    significance['has_autocorr'] = lb_p[0] < 0.05
                except:
                    significance['autocorr_lb_stat'] = 0.0
                    significance['autocorr_lb_p_value'] = 1.0
                    significance['has_autocorr'] = False
            else:
                significance['autocorr_lb_stat'] = 0.0
                significance['autocorr_lb_p_value'] = 1.0
                significance['has_autocorr'] = False
            
            return significance
            
        except Exception as e:
            logger.error(f"❌ Statistical significance calculation failed: {e}")
            return {}
    
    def _calculate_regime_stability_score(self, cluster_labels: np.ndarray, regime_id: int) -> float:
        """Calculate stability score for a specific regime."""
        try:
            # Get regime periods
            regime_mask = cluster_labels == regime_id
            regime_changes = np.diff(np.concatenate([[False], regime_mask, [False]]).astype(int))
            
            # Find regime periods
            starts = np.where(regime_changes == 1)[0]
            ends = np.where(regime_changes == -1)[0]
            
            if len(starts) == 0:
                return 0.0
            
            # Handle case where regime continues to end
            if len(ends) < len(starts):
                ends = np.concatenate([ends, [len(cluster_labels)]])
            
            # Calculate stability as consistency of regime duration
            durations = ends - starts
            if len(durations) == 0:
                return 0.0
            
            # Stability is inverse of coefficient of variation
            mean_duration = np.mean(durations)
            std_duration = np.std(durations)
            if mean_duration > 0:
                cv = std_duration / mean_duration
                stability = 1.0 / (1.0 + cv)  # Normalize to 0-1
            else:
                stability = 0.0
            
            return min(stability, 1.0)
            
        except Exception as e:
            logger.error(f"❌ Regime stability calculation failed: {e}")
            return 0.0
    
    def _calculate_transition_probabilities(self, cluster_labels: np.ndarray, regime_id: int) -> Dict[str, float]:
        """Calculate transition probabilities from this regime to others."""
        try:
            # Get unique regimes
            unique_regimes = np.unique(cluster_labels)
            unique_regimes = unique_regimes[unique_regimes != -1]
            
            if len(unique_regimes) <= 1:
                return {}
            
            # Find regime transitions
            regime_mask = cluster_labels == regime_id
            transitions = {}
            
            for other_regime in unique_regimes:
                if other_regime == regime_id:
                    continue
                
                # Count transitions from this regime to other regime
                transitions_from = 0
                total_transitions = 0
                
                for i in range(len(cluster_labels) - 1):
                    if cluster_labels[i] == regime_id and cluster_labels[i + 1] != regime_id:
                        total_transitions += 1
                        if cluster_labels[i + 1] == other_regime:
                            transitions_from += 1
                
                # Calculate probability
                if total_transitions > 0:
                    prob = transitions_from / total_transitions
                    transitions[f'to_regime_{other_regime}'] = prob
                else:
                    transitions[f'to_regime_{other_regime}'] = 0.0
            
            # Add probability of staying in same regime
            same_regime_count = 0
            total_periods = 0
            
            for i in range(len(cluster_labels) - 1):
                if cluster_labels[i] == regime_id:
                    total_periods += 1
                    if cluster_labels[i + 1] == regime_id:
                        same_regime_count += 1
            
            if total_periods > 0:
                transitions['stay_in_regime'] = same_regime_count / total_periods
            else:
                transitions['stay_in_regime'] = 0.0
            
            return transitions
            
        except Exception as e:
            logger.error(f"❌ Transition probabilities calculation failed: {e}")
            return {}
    
    def _calculate_economic_indicators(self, regime_data: pd.DataFrame, key_stats: Dict[str, float]) -> Dict[str, float]:
        """Calculate economic indicators during this regime."""
        try:
            indicators = {}
            
            # Basic economic indicators
            indicators['avg_return'] = key_stats['avg_return']
            indicators['volatility'] = key_stats['volatility']
            indicators['sharpe_ratio'] = key_stats['sharpe_ratio']
            indicators['max_drawdown'] = key_stats['max_drawdown']
            
            # Risk-adjusted returns
            if key_stats['volatility'] > 0:
                indicators['risk_adjusted_return'] = key_stats['avg_return'] / key_stats['volatility']
            else:
                indicators['risk_adjusted_return'] = 0.0
            
            # Volatility clustering
            indicators['volatility_clustering'] = key_stats.get('volatility_clustering', 0.0)
            
            # Market efficiency (approximate)
            if len(regime_data) > 10:
                prices = regime_data['close'].values if 'close' in regime_data.columns else regime_data.iloc[:, 0].values
                returns = np.diff(prices) / prices[:-1]
                if len(returns) > 1:
                    # Calculate first-order autocorrelation as efficiency measure
                    autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
                    indicators['market_efficiency'] = 1.0 - abs(autocorr) if not np.isnan(autocorr) else 0.0
                else:
                    indicators['market_efficiency'] = 0.0
            else:
                indicators['market_efficiency'] = 0.0
            
            # Regime persistence
            indicators['regime_persistence'] = key_stats.get('regime_persistence', 0.0)
            
            return indicators
            
        except Exception as e:
            logger.error(f"❌ Economic indicators calculation failed: {e}")
            return {}
    
    def _calculate_regime_correlations(self, profiles: List[RegimeProfile], returns: np.ndarray) -> Optional[np.ndarray]:
        """Calculate correlation matrix between regimes."""
        try:
            if len(profiles) < 2:
                return None
            
            # Extract regime returns
            regime_returns = []
            for profile in profiles:
                # This is a simplified approach - in practice, you'd need to track regime periods
                regime_returns.append([profile.key_stats['avg_return']])
            
            # Calculate correlation matrix
            regime_returns = np.array(regime_returns)
            correlation_matrix = np.corrcoef(regime_returns)
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"❌ Regime correlation calculation failed: {e}")
            return None
    
    def _calculate_regime_transitions(self, cluster_labels: np.ndarray) -> Optional[np.ndarray]:
        """Calculate regime transition matrix."""
        try:
            unique_regimes = np.unique(cluster_labels)
            unique_regimes = unique_regimes[unique_regimes != -1]
            
            if len(unique_regimes) < 2:
                return None
            
            # Create transition matrix
            n_regimes = len(unique_regimes)
            transition_matrix = np.zeros((n_regimes, n_regimes))
            
            # Count transitions
            for i in range(len(cluster_labels) - 1):
                current_regime = cluster_labels[i]
                next_regime = cluster_labels[i + 1]
                
                if current_regime != -1 and next_regime != -1:
                    current_idx = np.where(unique_regimes == current_regime)[0][0]
                    next_idx = np.where(unique_regimes == next_regime)[0][0]
                    transition_matrix[current_idx, next_idx] += 1
            
            # Normalize to probabilities
            row_sums = transition_matrix.sum(axis=1, keepdims=True)
            transition_matrix = np.divide(transition_matrix, row_sums, out=np.zeros_like(transition_matrix), where=row_sums!=0)
            
            return transition_matrix
            
        except Exception as e:
            logger.error(f"❌ Regime transition calculation failed: {e}")
            return None
    
    def _analyze_market_regime_patterns(self, profiles: List[RegimeProfile], market_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data-driven market regime patterns and characteristics."""
        try:
            analysis = {}
            
            # Regime distribution
            regime_types = [profile.regime_type for profile in profiles]
            type_counts = {}
            for regime_type in regime_types:
                type_counts[regime_type] = type_counts.get(regime_type, 0) + 1
            analysis['regime_type_distribution'] = type_counts
            
            # Data-driven regime characteristics analysis
            volatilities = [profile.key_stats['volatility'] for profile in profiles]
            sharpe_ratios = [profile.key_stats['sharpe_ratio'] for profile in profiles]
            returns = [profile.key_stats['avg_return'] for profile in profiles]
            durations = [profile.avg_duration for profile in profiles]
            stabilities = [profile.regime_stability for profile in profiles]
            
            # Calculate data-driven percentiles
            volatility_percentiles = np.percentile(volatilities, [25, 50, 75])
            sharpe_percentiles = np.percentile(sharpe_ratios, [25, 50, 75])
            return_percentiles = np.percentile(returns, [25, 50, 75])
            
            # Average regime characteristics
            avg_characteristics = {
                'avg_duration': np.mean(durations),
                'avg_volatility': np.mean(volatilities),
                'avg_sharpe': np.mean(sharpe_ratios),
                'avg_stability': np.mean(stabilities),
                'volatility_percentiles': volatility_percentiles.tolist(),
                'sharpe_percentiles': sharpe_percentiles.tolist(),
                'return_percentiles': return_percentiles.tolist()
            }
            analysis['avg_characteristics'] = avg_characteristics
            
            # Data-driven regime quality assessment
            high_quality_regimes = [p for p in profiles if p.key_stats['sharpe_ratio'] > sharpe_percentiles[2] and p.key_stats['volatility'] < volatility_percentiles[0]]
            low_quality_regimes = [p for p in profiles if p.key_stats['sharpe_ratio'] < sharpe_percentiles[0] or p.key_stats['volatility'] > volatility_percentiles[2]]
            
            analysis['high_quality_regimes'] = len(high_quality_regimes)
            analysis['low_quality_regimes'] = len(low_quality_regimes)
            analysis['quality_ratio'] = len(high_quality_regimes) / len(profiles) if profiles else 0
            
            # Market regime analysis
            analysis['total_regimes'] = len(profiles)
            analysis['regime_diversity'] = len(set(regime_types))
            analysis['most_common_type'] = max(type_counts, key=type_counts.get) if type_counts else 'unknown'
            
            # Data-driven regime stability analysis
            analysis['regime_stability_std'] = np.std(stabilities)
            analysis['regime_stability_cv'] = np.std(stabilities) / (np.mean(stabilities) + 1e-10)
            analysis['most_stable_regime'] = profiles[np.argmax(stabilities)].regime_id if profiles else None
            
            # Regime transition analysis
            analysis['regime_transition_frequency'] = self._calculate_regime_transition_frequency(profiles)
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Market regime pattern analysis failed: {e}")
            return {}
    
    def _calculate_regime_transition_frequency(self, profiles: List[RegimeProfile]) -> Dict[str, float]:
        """Calculate data-driven regime transition frequency."""
        try:
            if not profiles:
                return {}
            
            # Calculate transition probabilities for each regime
            transition_frequencies = {}
            for profile in profiles:
                regime_id = profile.regime_id
                transition_probs = profile.transition_probabilities
                
                if transition_probs:
                    # Calculate average transition probability
                    avg_transition_prob = np.mean(list(transition_probs.values()))
                    transition_frequencies[f'regime_{regime_id}'] = avg_transition_prob
                else:
                    transition_frequencies[f'regime_{regime_id}'] = 0.0
            
            # Calculate overall transition metrics
            all_transition_probs = list(transition_frequencies.values())
            transition_frequencies['avg_transition_frequency'] = np.mean(all_transition_probs)
            transition_frequencies['transition_volatility'] = np.std(all_transition_probs)
            
            return transition_frequencies
            
        except Exception as e:
            logger.debug(f"Regime transition frequency calculation failed: {e}")
            return {}
    
    def _perform_statistical_tests(self, profiles: List[RegimeProfile], returns: np.ndarray) -> Dict[str, Any]:
        """Perform statistical tests on regime characteristics."""
        try:
            tests = {}
            
            # Test for regime differences
            if len(profiles) >= 2:
                # Extract regime statistics
                regime_returns = []
                regime_volatilities = []
                
                for profile in profiles:
                    regime_returns.append(profile.key_stats['avg_return'])
                    regime_volatilities.append(profile.key_stats['volatility'])
                
                # ANOVA test for return differences
                try:
                    from scipy.stats import f_oneway
                    f_stat, p_value = f_oneway(*[regime_returns])
                    tests['regime_returns_anova'] = {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
                except:
                    tests['regime_returns_anova'] = {'error': 'insufficient_data'}
                
                # Kruskal-Wallis test for volatility differences
                try:
                    from scipy.stats import kruskal
                    h_stat, p_value = kruskal(*[regime_volatilities])
                    tests['regime_volatility_kruskal'] = {
                        'h_statistic': h_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
                except:
                    tests['regime_volatility_kruskal'] = {'error': 'insufficient_data'}
            
            # Overall market tests
            if len(returns) > 10:
                # Test for market efficiency (random walk)
                try:
                    from statsmodels.stats.diagnostic import acorr_ljungbox
                    lb_stat, lb_p = acorr_ljungbox(returns, lags=5, return_df=False)
                    tests['market_efficiency'] = {
                        'lb_statistic': lb_stat[0],
                        'p_value': lb_p[0],
                        'efficient': lb_p[0] > 0.05
                    }
                except:
                    tests['market_efficiency'] = {'error': 'test_failed'}
            
            return tests
            
        except Exception as e:
            logger.error(f"❌ Statistical tests failed: {e}")
            return {}
    
    def _analyze_regime_persistence(self, cluster_labels: np.ndarray, profiles: List[RegimeProfile]) -> Dict[str, Any]:
        """Analyze regime persistence and stability patterns."""
        try:
            analysis = {}
            
            # Calculate regime durations
            regime_durations = []
            for profile in profiles:
                regime_mask = cluster_labels == profile.regime_id
                regime_changes = np.diff(np.concatenate([[False], regime_mask, [False]]).astype(int))
                starts = np.where(regime_changes == 1)[0]
                ends = np.where(regime_changes == -1)[0]
                
                if len(ends) < len(starts):
                    ends = np.concatenate([ends, [len(cluster_labels)]])
                
                durations = ends - starts
                regime_durations.extend(durations.tolist())
            
            if regime_durations:
                analysis['avg_duration'] = np.mean(regime_durations)
                analysis['duration_std'] = np.std(regime_durations)
                analysis['min_duration'] = np.min(regime_durations)
                analysis['max_duration'] = np.max(regime_durations)
                analysis['duration_consistency'] = 1.0 / (1.0 + np.std(regime_durations) / np.mean(regime_durations))
            else:
                analysis['avg_duration'] = 0.0
                analysis['duration_std'] = 0.0
                analysis['min_duration'] = 0.0
                analysis['max_duration'] = 0.0
                analysis['duration_consistency'] = 0.0
            
            # Regime stability analysis
            stability_scores = [profile.regime_stability for profile in profiles]
            analysis['avg_stability'] = np.mean(stability_scores)
            analysis['stability_std'] = np.std(stability_scores)
            analysis['most_stable_regime'] = profiles[np.argmax(stability_scores)].regime_id if profiles else None
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Regime persistence analysis failed: {e}")
            return {}
    
    def get_data_driven_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Get data-driven thresholds used for regime classification."""
        try:
            all_regime_stats = getattr(self, '_all_regime_stats', [])
            
            return {
                'volatility_thresholds': self._calculate_volatility_thresholds(all_regime_stats),
                'return_thresholds': self._calculate_return_thresholds(all_regime_stats),
                'sharpe_thresholds': self._calculate_sharpe_thresholds(all_regime_stats)
            }
        except Exception as e:
            logger.error(f"❌ Failed to get data-driven thresholds: {e}")
            return {}
    
    def get_regime_statistics_summary(self) -> Dict[str, Any]:
        """Get comprehensive regime statistics summary."""
        try:
            all_regime_stats = getattr(self, '_all_regime_stats', [])
            
            if not all_regime_stats:
                return {}
            
            # Extract all statistics
            volatilities = [stats.get('volatility', 0) for stats in all_regime_stats]
            returns = [stats.get('avg_return', 0) for stats in all_regime_stats]
            sharpe_ratios = [stats.get('sharpe_ratio', 0) for stats in all_regime_stats]
            
            # Calculate comprehensive statistics
            summary = {
                'n_regimes': len(all_regime_stats),
                'volatility_stats': {
                    'mean': np.mean(volatilities),
                    'std': np.std(volatilities),
                    'min': np.min(volatilities),
                    'max': np.max(volatilities),
                    'percentiles': np.percentile(volatilities, [25, 50, 75]).tolist()
                },
                'return_stats': {
                    'mean': np.mean(returns),
                    'std': np.std(returns),
                    'min': np.min(returns),
                    'max': np.max(returns),
                    'percentiles': np.percentile(returns, [25, 50, 75]).tolist()
                },
                'sharpe_stats': {
                    'mean': np.mean(sharpe_ratios),
                    'std': np.std(sharpe_ratios),
                    'min': np.min(sharpe_ratios),
                    'max': np.max(sharpe_ratios),
                    'percentiles': np.percentile(sharpe_ratios, [25, 50, 75]).tolist()
                },
                'data_driven_thresholds': self.get_data_driven_thresholds()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Failed to get regime statistics summary: {e}")
            return {}