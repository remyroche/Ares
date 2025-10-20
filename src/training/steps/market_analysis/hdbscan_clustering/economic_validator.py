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

@dataclass
class EconomicValidationResult:
    """Result of economic validation and profiling."""
    profiles: List[RegimeProfile]
    validation_metrics: Dict[str, float]
    regime_quality_score: float
    trading_recommendations: Dict[str, Any]

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
            with tprint_progress("Creating regime profiles", len(unique_regimes)):
                for i, regime_id in enumerate(unique_regimes):
                    tprint_debug(f"Creating profile for regime {regime_id} ({i+1}/{len(unique_regimes)})")
                    profile = self._create_regime_profile(
                        regime_id, cluster_labels, market_data, returns, features_df
                    )
                    profiles.append(profile)
                    tprint_progress("Creating regime profiles", i + 1)
            
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
            
            tprint_success(f"✅ Economic validation completed. Quality score: {regime_quality_score:.3f}")
            
            return EconomicValidationResult(
                profiles=profiles,
                validation_metrics=validation_metrics,
                regime_quality_score=regime_quality_score,
                trading_recommendations=trading_recommendations
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
                radar_plot_data=radar_plot_data
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
    
    def validate_regime_quality(self, market_data: pd.DataFrame, 
                              regime_labels: np.ndarray) -> Dict[str, Any]:
        """
        Enhanced validation of regime quality with comprehensive metrics.
        
        Args:
            market_data: Market data DataFrame
            regime_labels: Regime labels from clustering
            
        Returns:
            Dictionary with comprehensive validation results
        """
        try:
            logger.info("🔍 Starting comprehensive regime quality validation")
            
            validation_results = {
                'regime_profiling': {},
                'statistical_analysis': {},
                'economic_validation': {},
                'cross_validation': {},
                'overall_score': 0.0,
                'is_valid': True,
                'issues': []
            }
            
            # Validate regime profiling logic
            regime_profiling_results = self._validate_regime_profiling_logic(
                market_data, regime_labels
            )
            validation_results['regime_profiling'] = regime_profiling_results
            
            # Validate statistical analysis
            statistical_results = self._validate_statistical_analysis(
                market_data, regime_labels
            )
            validation_results['statistical_analysis'] = statistical_results
            
            # Validate economic metrics
            economic_results = self._validate_economic_metrics(
                market_data, regime_labels
            )
            validation_results['economic_validation'] = economic_results
            
            # Cross-validation
            cv_results = self._cross_validate_regime_discovery(
                market_data, regime_labels
            )
            validation_results['cross_validation'] = cv_results
            
            # Calculate overall score
            overall_score = self._calculate_overall_validation_score(validation_results)
            validation_results['overall_score'] = overall_score
            
            # Determine if system is valid
            validation_results['is_valid'] = (
                regime_profiling_results.get('is_valid', False) and
                statistical_results.get('is_valid', False) and
                economic_results.get('is_valid', False) and
                overall_score > 0.5
            )
            
            logger.info(f"✅ Regime quality validation completed. Overall score: {overall_score:.3f}")
            return validation_results
            
        except Exception as e:
            logger.error(f"❌ Regime quality validation failed: {e}")
            return {'error': str(e), 'is_valid': False}
    
    def _validate_regime_profiling_logic(self, 
                                       market_data: pd.DataFrame,
                                       regime_labels: np.ndarray) -> Dict[str, Any]:
        """Validate regime profiling logic."""
        try:
            results = {
                'regime_count': 0,
                'regime_durations': [],
                'regime_transitions': 0,
                'regime_stability': 0.0,
                'is_valid': True,
                'issues': []
            }
            
            # Count regimes
            unique_regimes = np.unique(regime_labels)
            unique_regimes = unique_regimes[unique_regimes != -1]  # Remove noise
            results['regime_count'] = len(unique_regimes)
            
            if results['regime_count'] == 0:
                results['issues'].append("No valid regimes found")
                results['is_valid'] = False
                return results
            
            # Calculate regime durations
            regime_durations = []
            for regime in unique_regimes:
                regime_mask = regime_labels == regime
                regime_indices = np.where(regime_mask)[0]
                
                if len(regime_indices) > 0:
                    # Find consecutive periods
                    consecutive_periods = self._find_consecutive_periods(regime_indices)
                    regime_durations.extend(consecutive_periods)
            
            results['regime_durations'] = regime_durations
            
            # Check minimum duration
            min_duration = min(regime_durations) if regime_durations else 0
            if min_duration < self.config.min_regime_duration:
                results['issues'].append(f"Minimum regime duration too short: {min_duration}")
                results['is_valid'] = False
            
            # Calculate regime transitions
            transitions = self._calculate_regime_transitions(regime_labels)
            results['regime_transitions'] = transitions
            
            # Calculate regime stability
            stability = self._calculate_regime_stability(regime_labels)
            results['regime_stability'] = stability
            
            if stability < 0.8:
                results['issues'].append(f"Regime stability too low: {stability:.3f}")
                results['is_valid'] = False
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Regime profiling logic validation failed: {e}")
            return {'error': str(e), 'is_valid': False}
    
    def _validate_statistical_analysis(self, 
                                     market_data: pd.DataFrame,
                                     regime_labels: np.ndarray) -> Dict[str, Any]:
        """Validate statistical analysis of regime characteristics."""
        try:
            results = {
                'confidence_intervals': {},
                'statistical_tests': {},
                'is_valid': True,
                'issues': []
            }
            
            # Test confidence interval calculations
            try:
                validation_result = self.validate_and_profile(market_data, regime_labels)
                
                if validation_result and 'regime_profiles' in validation_result:
                    for profile in validation_result['regime_profiles']:
                        regime_name = profile['regime_name']
                        
                        # Check if confidence intervals are present
                        if 'confidence_intervals' in profile:
                            ci_data = profile['confidence_intervals']
                            results['confidence_intervals'][regime_name] = ci_data
                        else:
                            results['issues'].append(f"No confidence intervals for regime {regime_name}")
                            results['is_valid'] = False
                
            except Exception as e:
                results['issues'].append(f"Statistical analysis validation failed: {e}")
                results['is_valid'] = False
            
            # Test statistical significance
            results['statistical_tests'] = self._test_statistical_significance(
                market_data, regime_labels
            )
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Statistical analysis validation failed: {e}")
            return {'error': str(e), 'is_valid': False}
    
    def _validate_economic_metrics(self, 
                                 market_data: pd.DataFrame,
                                 regime_labels: np.ndarray) -> Dict[str, Any]:
        """Validate economic metrics and calculations."""
        try:
            results = {
                'sharpe_ratios': {},
                'volatilities': {},
                'drawdowns': {},
                'is_valid': True,
                'issues': []
            }
            
            # Test economic validator
            try:
                validation_result = self.validate_and_profile(market_data, regime_labels)
                
                if validation_result and 'regime_profiles' in validation_result:
                    for profile in validation_result['regime_profiles']:
                        regime_name = profile['regime_name']
                        
                        # Validate Sharpe ratios
                        if 'sharpe_ratio' in profile:
                            sharpe = profile['sharpe_ratio']
                            results['sharpe_ratios'][regime_name] = sharpe
                            
                            if sharpe < 0.5:
                                results['issues'].append(f"Low Sharpe ratio for {regime_name}: {sharpe:.3f}")
                                results['is_valid'] = False
                        
                        # Validate volatilities
                        if 'volatility' in profile:
                            vol = profile['volatility']
                            results['volatilities'][regime_name] = vol
                            
                            if vol < 0.01 or vol > 0.5:
                                results['issues'].append(f"Volatility out of range for {regime_name}: {vol:.3f}")
                                results['is_valid'] = False
                        
                        # Validate drawdowns
                        if 'max_drawdown' in profile:
                            dd = abs(profile['max_drawdown'])
                            results['drawdowns'][regime_name] = dd
                            
                            if dd > 0.2:
                                results['issues'].append(f"High drawdown for {regime_name}: {dd:.3f}")
                                results['is_valid'] = False
                
            except Exception as e:
                results['issues'].append(f"Economic metrics validation failed: {e}")
                results['is_valid'] = False
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Economic metrics validation failed: {e}")
            return {'error': str(e), 'is_valid': False}
    
    def _cross_validate_regime_discovery(self, 
                                       market_data: pd.DataFrame,
                                       regime_labels: np.ndarray) -> Dict[str, Any]:
        """Perform cross-validation for regime discovery."""
        try:
            results = {
                'cv_scores': [],
                'stability_scores': [],
                'mean_cv_score': 0.0,
                'cv_std': 0.0,
                'is_valid': True
            }
            
            # Simple cross-validation using time series split
            n_samples = len(regime_labels)
            n_splits = 5
            
            cv_scores = []
            stability_scores = []
            
            for i in range(n_splits):
                try:
                    # Split data
                    split_size = n_samples // n_splits
                    start_idx = i * split_size
                    end_idx = (i + 1) * split_size if i < n_splits - 1 else n_samples
                    
                    train_labels = regime_labels[start_idx:end_idx]
                    
                    # Calculate stability
                    stability = self._calculate_regime_stability(train_labels)
                    stability_scores.append(stability)
                    
                    # Calculate CV score (simplified)
                    cv_score = stability
                    cv_scores.append(cv_score)
                    
                except Exception as e:
                    logger.debug(f"CV fold {i} failed: {e}")
                    continue
            
            if cv_scores:
                results['cv_scores'] = cv_scores
                results['stability_scores'] = stability_scores
                results['mean_cv_score'] = np.mean(cv_scores)
                results['cv_std'] = np.std(cv_scores)
                
                # Validate CV scores
                if results['mean_cv_score'] < 0.5:
                    results['is_valid'] = False
            else:
                results['is_valid'] = False
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Cross-validation failed: {e}")
            return {'error': str(e), 'is_valid': False}
    
    def _find_consecutive_periods(self, indices: np.ndarray) -> List[int]:
        """Find consecutive periods in regime indices."""
        if len(indices) == 0:
            return []
        
        consecutive_periods = []
        current_length = 1
        
        for i in range(1, len(indices)):
            if indices[i] == indices[i-1] + 1:
                current_length += 1
            else:
                consecutive_periods.append(current_length)
                current_length = 1
        
        consecutive_periods.append(current_length)
        return consecutive_periods
    
    def _calculate_regime_transitions(self, regime_labels: np.ndarray) -> int:
        """Calculate number of regime transitions."""
        transitions = 0
        for i in range(1, len(regime_labels)):
            if regime_labels[i] != regime_labels[i-1]:
                transitions += 1
        return transitions
    
    def _calculate_regime_stability(self, regime_labels: np.ndarray) -> float:
        """Calculate regime stability score."""
        if len(regime_labels) == 0:
            return 0.0
        
        # Calculate stability as 1 - (transitions / total_periods)
        transitions = self._calculate_regime_transitions(regime_labels)
        stability = 1.0 - (transitions / len(regime_labels))
        return max(0.0, stability)
    
    def _test_statistical_significance(self, 
                                     market_data: pd.DataFrame,
                                     regime_labels: np.ndarray) -> Dict[str, Any]:
        """Test statistical significance of regime differences."""
        try:
            results = {
                't_tests': {},
                'is_significant': True
            }
            
            # Test returns across regimes
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change().dropna()
                
                # Align returns with regime labels
                min_len = min(len(returns), len(regime_labels))
                returns = returns.iloc[:min_len]
                labels = regime_labels[:min_len]
                
                # Remove noise
                non_noise_mask = labels != -1
                if np.sum(non_noise_mask) < 2:
                    return results
                
                returns_clean = returns[non_noise_mask]
                labels_clean = labels[non_noise_mask]
                
                # T-tests between regimes
                unique_regimes = np.unique(labels_clean)
                if len(unique_regimes) >= 2:
                    regime_returns = [returns_clean[labels_clean == regime] for regime in unique_regimes]
                    
                    # Pairwise t-tests
                    for i, regime1 in enumerate(unique_regimes):
                        for j, regime2 in enumerate(unique_regimes[i+1:], i+1):
                            try:
                                from scipy import stats
                                t_stat, p_value = stats.ttest_ind(regime_returns[i], regime_returns[j])
                                results['t_tests'][f'regime_{regime1}_vs_regime_{regime2}'] = {
                                    't_statistic': t_stat,
                                    'p_value': p_value,
                                    'is_significant': p_value < 0.05
                                }
                            except Exception as e:
                                logger.debug(f"T-test failed for regimes {regime1} vs {regime2}: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Statistical significance testing failed: {e}")
            return {'error': str(e), 'is_significant': False}
    
    def _calculate_overall_validation_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate overall validation score."""
        try:
            scores = []
            
            # Regime profiling score
            regime_score = 1.0 if validation_results['regime_profiling'].get('is_valid', False) else 0.0
            scores.append(regime_score)
            
            # Statistical analysis score
            stat_score = 1.0 if validation_results['statistical_analysis'].get('is_valid', False) else 0.0
            scores.append(stat_score)
            
            # Economic validation score
            econ_score = 1.0 if validation_results['economic_validation'].get('is_valid', False) else 0.0
            scores.append(econ_score)
            
            # Cross-validation score
            cv_score = validation_results['cross_validation'].get('mean_cv_score', 0.0)
            scores.append(cv_score)
            
            # Calculate weighted average
            weights = [0.3, 0.3, 0.2, 0.2]  # Regime, Statistical, Economic, CV
            overall_score = np.average(scores, weights=weights)
            
            return overall_score
            
        except Exception as e:
            logger.error(f"❌ Overall score calculation failed: {e}")
            return 0.0
            
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
        """Generate descriptive name for regime based on characteristics."""
        try:
            # Determine regime type based on key characteristics
            if key_stats['sharpe_ratio'] > 1.0 and key_stats['volatility'] < 0.1:
                return f"High_Quality_{regime_id}"
            elif key_stats['volatility'] > 0.2:
                return f"High_Vol_{regime_id}"
            elif key_stats['avg_return'] > 0.01:
                return f"Bullish_{regime_id}"
            elif key_stats['avg_return'] < -0.01:
                return f"Bearish_{regime_id}"
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
        """Generate trading recommendations based on regime profiles."""
        try:
            recommendations = {
                'best_regimes': [],
                'avoid_regimes': [],
                'overall_strategy': 'Conservative',
                'risk_level': 'Medium'
            }
            
            if not profiles:
                return recommendations
            
            # Analyze each regime
            for profile in profiles:
                if profile.key_stats['sharpe_ratio'] > 1.0 and profile.key_stats['volatility'] < 0.15:
                    recommendations['best_regimes'].append(profile.regime_id)
                elif profile.key_stats['sharpe_ratio'] < -0.5 or profile.key_stats['volatility'] > 0.25:
                    recommendations['avoid_regimes'].append(profile.regime_id)
            
            # Determine overall strategy
            avg_sharpe = np.mean([p.key_stats['sharpe_ratio'] for p in profiles])
            avg_vol = np.mean([p.key_stats['volatility'] for p in profiles])
            
            if avg_sharpe > 0.5 and avg_vol < 0.15:
                recommendations['overall_strategy'] = 'Aggressive'
                recommendations['risk_level'] = 'Low'
            elif avg_sharpe < 0 and avg_vol > 0.2:
                recommendations['overall_strategy'] = 'Defensive'
                recommendations['risk_level'] = 'High'
            
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
            trading_recommendations={'best_regimes': [], 'avoid_regimes': [], 'overall_strategy': 'Conservative', 'risk_level': 'Medium'}
        )