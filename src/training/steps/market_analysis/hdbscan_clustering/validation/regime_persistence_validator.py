"""
Regime Persistence and Economic Coherence Validator

This module provides validation for regime stability and economic coherence,
ensuring that regimes maintain their economic identity over time and exhibit
meaningful persistence patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from scipy import stats
from scipy.signal import find_peaks
from itertools import groupby
import warnings

logger = logging.getLogger(__name__)

@dataclass
class RegimePersistenceConfig:
    """Configuration for regime persistence validation."""
    # Persistence metrics
    enable_lifespan_analysis: bool = True
    enable_transition_analysis: bool = True
    enable_economic_coherence: bool = True
    enable_volatility_persistence: bool = True
    
    # Lifespan parameters
    min_regime_lifespan: int = 5
    max_regime_lifespan: int = 1000
    expected_lifespan: int = 50
    
    # Transition parameters
    max_transition_frequency: float = 0.1  # Max 10% of periods can be transitions
    min_stability_periods: int = 10
    
    # Economic coherence parameters
    return_correlation_threshold: float = 0.3
    volatility_correlation_threshold: float = 0.2
    volume_correlation_threshold: float = 0.1
    
    # Volatility persistence parameters
    volatility_regime_threshold: float = 0.02
    min_volatility_consistency: float = 0.7
    
    # Statistical validation
    enable_statistical_tests: bool = True
    significance_level: float = 0.05
    bootstrap_samples: int = 1000

@dataclass
class RegimePersistenceResult:
    """Result of regime persistence validation."""
    # Overall scores
    overall_persistence_score: float
    lifespan_score: float
    transition_score: float
    economic_coherence_score: float
    volatility_persistence_score: float
    
    # Detailed metrics
    lifespan_metrics: Dict[str, Any]
    transition_metrics: Dict[str, Any]
    economic_coherence_metrics: Dict[str, Any]
    volatility_persistence_metrics: Dict[str, Any]
    
    # Statistical validation
    statistical_tests: Dict[str, Any]
    
    # Metadata
    n_regimes: int
    n_transitions: int
    avg_regime_lifespan: float
    validation_time: float

class RegimePersistenceValidator:
    """
    Validator for regime persistence and economic coherence.
    
    Ensures that regimes maintain their economic identity over time and
    exhibit meaningful persistence patterns that are economically significant.
    """
    
    def __init__(self, config: Optional[RegimePersistenceConfig] = None):
        """Initialize regime persistence validator."""
        self.config = config or RegimePersistenceConfig()
        
    def validate_persistence(self, 
                           cluster_labels: np.ndarray,
                           market_data: pd.DataFrame,
                           features: Optional[np.ndarray] = None,
                           feature_names: Optional[List[str]] = None) -> RegimePersistenceResult:
        """
        Validate regime persistence and economic coherence.
        
        Args:
            cluster_labels: Cluster labels to validate
            market_data: Market data with price, volume, etc.
            features: Optional feature matrix
            feature_names: Optional feature names
            
        Returns:
            RegimePersistenceResult with comprehensive persistence metrics
        """
        try:
            import time
            start_time = time.time()
            
            logger.info("🔍 Starting regime persistence validation...")
            
            # Validate input data
            cluster_labels, market_data = self._validate_input(cluster_labels, market_data)
            
            # Calculate lifespan metrics
            lifespan_metrics = self._calculate_lifespan_metrics(cluster_labels)
            
            # Calculate transition metrics
            transition_metrics = self._calculate_transition_metrics(cluster_labels)
            
            # Calculate economic coherence metrics
            economic_coherence_metrics = self._calculate_economic_coherence_metrics(
                cluster_labels, market_data, features, feature_names
            )
            
            # Calculate volatility persistence metrics
            volatility_persistence_metrics = self._calculate_volatility_persistence_metrics(
                cluster_labels, market_data
            )
            
            # Perform statistical tests
            statistical_tests = self._perform_statistical_tests(
                cluster_labels, market_data, lifespan_metrics, transition_metrics
            )
            
            # Calculate overall scores
            scores = self._calculate_overall_scores(
                lifespan_metrics, transition_metrics, economic_coherence_metrics, volatility_persistence_metrics
            )
            
            # Create result
            result = RegimePersistenceResult(
                overall_persistence_score=scores['overall'],
                lifespan_score=scores['lifespan'],
                transition_score=scores['transition'],
                economic_coherence_score=scores['economic_coherence'],
                volatility_persistence_score=scores['volatility_persistence'],
                lifespan_metrics=lifespan_metrics,
                transition_metrics=transition_metrics,
                economic_coherence_metrics=economic_coherence_metrics,
                volatility_persistence_metrics=volatility_persistence_metrics,
                statistical_tests=statistical_tests,
                n_regimes=len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
                n_transitions=transition_metrics.get('n_transitions', 0),
                avg_regime_lifespan=lifespan_metrics.get('avg_lifespan', 0),
                validation_time=time.time() - start_time
            )
            
            logger.info(f"✅ Regime persistence validation completed in {result.validation_time:.2f}s")
            logger.info(f"📊 Overall persistence score: {result.overall_persistence_score:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Regime persistence validation failed: {e}")
            raise
    
    def _validate_input(self, cluster_labels: np.ndarray, market_data: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """Validate input data."""
        try:
            # Check cluster labels
            if len(cluster_labels) == 0:
                raise ValueError("Empty cluster labels")
            
            # Check market data
            if len(market_data) == 0:
                raise ValueError("Empty market data")
            
            # Ensure same length
            if len(cluster_labels) != len(market_data):
                raise ValueError(f"Length mismatch: labels={len(cluster_labels)}, data={len(market_data)}")
            
            # Check required columns
            required_columns = ['close']
            missing_columns = [col for col in required_columns if col not in market_data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            return cluster_labels, market_data
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            raise
    
    def _calculate_lifespan_metrics(self, cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Calculate regime lifespan metrics."""
        try:
            if not self.config.enable_lifespan_analysis:
                return {'enabled': False}
            
            # Remove noise points
            valid_labels = cluster_labels[cluster_labels != -1]
            
            if len(valid_labels) == 0:
                return {'enabled': True, 'error': 'No valid regimes'}
            
            # Calculate regime lifespans
            regime_lifespans = []
            current_regime = valid_labels[0]
            current_length = 1
            
            for i in range(1, len(valid_labels)):
                if valid_labels[i] == current_regime:
                    current_length += 1
                else:
                    regime_lifespans.append(current_length)
                    current_regime = valid_labels[i]
                    current_length = 1
            
            # Add last regime
            regime_lifespans.append(current_length)
            
            if not regime_lifespans:
                return {'enabled': True, 'error': 'No regime lifespans calculated'}
            
            # Calculate statistics
            avg_lifespan = np.mean(regime_lifespans)
            median_lifespan = np.median(regime_lifespans)
            std_lifespan = np.std(regime_lifespans)
            min_lifespan = np.min(regime_lifespans)
            max_lifespan = np.max(regime_lifespans)
            
            # Calculate lifespan score
            lifespan_score = self._calculate_lifespan_score(regime_lifespans)
            
            # Calculate regime count
            n_regimes = len(regime_lifespans)
            
            return {
                'enabled': True,
                'regime_lifespans': regime_lifespans,
                'avg_lifespan': avg_lifespan,
                'median_lifespan': median_lifespan,
                'std_lifespan': std_lifespan,
                'min_lifespan': min_lifespan,
                'max_lifespan': max_lifespan,
                'lifespan_score': lifespan_score,
                'n_regimes': n_regimes,
                'lifespan_distribution': {
                    'short_regimes': sum(1 for x in regime_lifespans if x < self.config.min_regime_lifespan),
                    'medium_regimes': sum(1 for x in regime_lifespans if self.config.min_regime_lifespan <= x <= self.config.max_regime_lifespan),
                    'long_regimes': sum(1 for x in regime_lifespans if x > self.config.max_regime_lifespan)
                }
            }
            
        except Exception as e:
            logger.warning(f"Lifespan metrics calculation failed: {e}")
            return {'enabled': True, 'error': str(e)}
    
    def _calculate_lifespan_score(self, regime_lifespans: List[int]) -> float:
        """Calculate lifespan quality score."""
        try:
            if not regime_lifespans:
                return 0.0
            
            # Penalize very short regimes
            short_penalty = sum(1 for x in regime_lifespans if x < self.config.min_regime_lifespan) / len(regime_lifespans)
            
            # Penalize very long regimes (potential over-smoothing)
            long_penalty = sum(1 for x in regime_lifespans if x > self.config.max_regime_lifespan) / len(regime_lifespans)
            
            # Reward regimes close to expected lifespan
            expected_lifespan = self.config.expected_lifespan
            closeness_score = 1 - np.mean([abs(x - expected_lifespan) / expected_lifespan for x in regime_lifespans])
            
            # Overall score
            lifespan_score = max(0, closeness_score - short_penalty - long_penalty)
            
            return lifespan_score
            
        except Exception as e:
            logger.warning(f"Lifespan score calculation failed: {e}")
            return 0.0
    
    def _calculate_transition_metrics(self, cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Calculate regime transition metrics."""
        try:
            if not self.config.enable_transition_analysis:
                return {'enabled': False}
            
            # Remove noise points
            valid_labels = cluster_labels[cluster_labels != -1]
            
            if len(valid_labels) < 2:
                return {'enabled': True, 'error': 'Insufficient data for transition analysis'}
            
            # Calculate transitions
            transitions = []
            for i in range(1, len(valid_labels)):
                if valid_labels[i] != valid_labels[i-1]:
                    transitions.append({
                        'from_regime': valid_labels[i-1],
                        'to_regime': valid_labels[i],
                        'position': i
                    })
            
            n_transitions = len(transitions)
            transition_frequency = n_transitions / len(valid_labels)
            
            # Calculate transition score
            transition_score = self._calculate_transition_score(transitions, len(valid_labels))
            
            # Calculate transition patterns
            transition_patterns = self._analyze_transition_patterns(transitions)
            
            return {
                'enabled': True,
                'transitions': transitions,
                'n_transitions': n_transitions,
                'transition_frequency': transition_frequency,
                'transition_score': transition_score,
                'transition_patterns': transition_patterns,
                'stability_periods': len(valid_labels) - n_transitions
            }
            
        except Exception as e:
            logger.warning(f"Transition metrics calculation failed: {e}")
            return {'enabled': True, 'error': str(e)}
    
    def _calculate_transition_score(self, transitions: List[Dict[str, Any]], total_periods: int) -> float:
        """Calculate transition quality score."""
        try:
            if not transitions:
                return 1.0  # No transitions is good for stability
            
            transition_frequency = len(transitions) / total_periods
            
            # Penalize high transition frequency
            frequency_penalty = max(0, transition_frequency - self.config.max_transition_frequency)
            
            # Reward moderate transition frequency (some change is good)
            if transition_frequency <= self.config.max_transition_frequency:
                frequency_score = 1.0
            else:
                frequency_score = 1.0 - frequency_penalty
            
            # Calculate transition diversity (avoid ping-ponging)
            transition_diversity = self._calculate_transition_diversity(transitions)
            
            # Overall score
            transition_score = (frequency_score + transition_diversity) / 2
            
            return max(0, transition_score)
            
        except Exception as e:
            logger.warning(f"Transition score calculation failed: {e}")
            return 0.0
    
    def _calculate_transition_diversity(self, transitions: List[Dict[str, Any]]) -> float:
        """Calculate transition diversity score."""
        try:
            if len(transitions) < 2:
                return 1.0
            
            # Count unique transition pairs
            transition_pairs = [(t['from_regime'], t['to_regime']) for t in transitions]
            unique_pairs = len(set(transition_pairs))
            
            # Calculate diversity score
            max_possible_pairs = len(set([t['from_regime'] for t in transitions])) * len(set([t['to_regime'] for t in transitions]))
            diversity_score = unique_pairs / max_possible_pairs if max_possible_pairs > 0 else 0
            
            return diversity_score
            
        except Exception as e:
            logger.warning(f"Transition diversity calculation failed: {e}")
            return 0.0
    
    def _analyze_transition_patterns(self, transitions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze transition patterns."""
        try:
            if not transitions:
                return {'patterns': [], 'summary': {}}
            
            # Group transitions by from_regime
            from_regime_groups = {}
            for transition in transitions:
                from_regime = transition['from_regime']
                if from_regime not in from_regime_groups:
                    from_regime_groups[from_regime] = []
                from_regime_groups[from_regime].append(transition)
            
            # Analyze patterns
            patterns = []
            for from_regime, regime_transitions in from_regime_groups.items():
                to_regimes = [t['to_regime'] for t in regime_transitions]
                unique_to_regimes = list(set(to_regimes))
                
                patterns.append({
                    'from_regime': from_regime,
                    'to_regimes': unique_to_regimes,
                    'n_transitions': len(regime_transitions),
                    'most_common_to': max(set(to_regimes), key=to_regimes.count) if to_regimes else None
                })
            
            # Summary statistics
            summary = {
                'n_unique_from_regimes': len(from_regime_groups),
                'avg_transitions_per_regime': np.mean([len(transitions) for transitions in from_regime_groups.values()]),
                'max_transitions_from_single_regime': max([len(transitions) for transitions in from_regime_groups.values()]) if from_regime_groups else 0
            }
            
            return {
                'patterns': patterns,
                'summary': summary
            }
            
        except Exception as e:
            logger.warning(f"Transition pattern analysis failed: {e}")
            return {'patterns': [], 'summary': {}}
    
    def _calculate_economic_coherence_metrics(self, 
                                            cluster_labels: np.ndarray,
                                            market_data: pd.DataFrame,
                                            features: Optional[np.ndarray],
                                            feature_names: Optional[List[str]]) -> Dict[str, Any]:
        """Calculate economic coherence metrics."""
        try:
            if not self.config.enable_economic_coherence:
                return {'enabled': False}
            
            # Remove noise points
            valid_mask = cluster_labels != -1
            valid_labels = cluster_labels[valid_mask]
            valid_market_data = market_data.iloc[valid_mask]
            
            if len(valid_labels) < 10:
                return {'enabled': True, 'error': 'Insufficient data for economic coherence analysis'}
            
            # Calculate returns
            if 'close' in valid_market_data.columns:
                returns = valid_market_data['close'].pct_change().dropna()
                returns = returns.iloc[1:]  # Remove first NaN
                
                # Align with labels
                min_len = min(len(returns), len(valid_labels))
                returns = returns.iloc[:min_len]
                labels = valid_labels[:min_len]
                
                # Calculate return coherence
                return_coherence = self._calculate_return_coherence(labels, returns)
                
                # Calculate volatility coherence
                volatility_coherence = self._calculate_volatility_coherence(labels, returns)
                
                # Calculate volume coherence (if available)
                volume_coherence = self._calculate_volume_coherence(labels, valid_market_data)
                
                # Calculate overall economic coherence score
                coherence_scores = [return_coherence, volatility_coherence, volume_coherence]
                valid_scores = [s for s in coherence_scores if s is not None]
                overall_coherence = np.mean(valid_scores) if valid_scores else 0.0
                
                return {
                    'enabled': True,
                    'return_coherence': return_coherence,
                    'volatility_coherence': volatility_coherence,
                    'volume_coherence': volume_coherence,
                    'overall_coherence': overall_coherence,
                    'n_regimes': len(set(labels))
                }
            else:
                return {'enabled': True, 'error': 'No price data available for economic coherence analysis'}
                
        except Exception as e:
            logger.warning(f"Economic coherence metrics calculation failed: {e}")
            return {'enabled': True, 'error': str(e)}
    
    def _calculate_return_coherence(self, labels: np.ndarray, returns: pd.Series) -> Optional[float]:
        """Calculate return coherence within regimes."""
        try:
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return None
            
            # Calculate return correlation within each regime
            regime_correlations = []
            
            for label in unique_labels:
                regime_mask = labels == label
                regime_returns = returns[regime_mask]
                
                if len(regime_returns) > 5:  # Minimum samples for correlation
                    # Calculate autocorrelation
                    autocorr = regime_returns.autocorr(lag=1)
                    if not pd.isna(autocorr):
                        regime_correlations.append(abs(autocorr))
            
            if regime_correlations:
                avg_correlation = np.mean(regime_correlations)
                return min(1.0, avg_correlation / self.config.return_correlation_threshold)
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Return coherence calculation failed: {e}")
            return None
    
    def _calculate_volatility_coherence(self, labels: np.ndarray, returns: pd.Series) -> Optional[float]:
        """Calculate volatility coherence within regimes."""
        try:
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return None
            
            # Calculate volatility for each regime
            regime_volatilities = []
            
            for label in unique_labels:
                regime_mask = labels == label
                regime_returns = returns[regime_mask]
                
                if len(regime_returns) > 5:  # Minimum samples for volatility
                    regime_vol = regime_returns.std()
                    if not pd.isna(regime_vol):
                        regime_volatilities.append(regime_vol)
            
            if len(regime_volatilities) > 1:
                # Calculate volatility consistency (lower coefficient of variation is better)
                vol_mean = np.mean(regime_volatilities)
                vol_std = np.std(regime_volatilities)
                vol_cv = vol_std / (vol_mean + 1e-10)
                
                # Convert to coherence score (lower CV = higher coherence)
                coherence_score = max(0, 1 - vol_cv / self.config.volatility_correlation_threshold)
                return coherence_score
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Volatility coherence calculation failed: {e}")
            return None
    
    def _calculate_volume_coherence(self, labels: np.ndarray, market_data: pd.DataFrame) -> Optional[float]:
        """Calculate volume coherence within regimes."""
        try:
            if 'volume' not in market_data.columns:
                return None
            
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return None
            
            # Calculate volume consistency for each regime
            regime_volumes = []
            
            for label in unique_labels:
                regime_mask = labels == label
                regime_volume = market_data['volume'][regime_mask]
                
                if len(regime_volume) > 5:  # Minimum samples for volume analysis
                    regime_vol_mean = regime_volume.mean()
                    if not pd.isna(regime_vol_mean):
                        regime_volumes.append(regime_vol_mean)
            
            if len(regime_volumes) > 1:
                # Calculate volume consistency
                vol_mean = np.mean(regime_volumes)
                vol_std = np.std(regime_volumes)
                vol_cv = vol_std / (vol_mean + 1e-10)
                
                # Convert to coherence score
                coherence_score = max(0, 1 - vol_cv / self.config.volume_correlation_threshold)
                return coherence_score
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Volume coherence calculation failed: {e}")
            return None
    
    def _calculate_volatility_persistence_metrics(self, cluster_labels: np.ndarray, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volatility persistence metrics."""
        try:
            if not self.config.enable_volatility_persistence:
                return {'enabled': False}
            
            if 'close' not in market_data.columns:
                return {'enabled': True, 'error': 'No price data available for volatility persistence analysis'}
            
            # Calculate returns
            returns = market_data['close'].pct_change().dropna()
            
            # Align with labels
            min_len = min(len(returns), len(cluster_labels))
            returns = returns.iloc[:min_len]
            labels = cluster_labels[:min_len]
            
            # Remove noise points
            valid_mask = labels != -1
            valid_returns = returns[valid_mask]
            valid_labels = labels[valid_mask]
            
            if len(valid_returns) < 10:
                return {'enabled': True, 'error': 'Insufficient data for volatility persistence analysis'}
            
            # Calculate rolling volatility
            vol_window = 20
            volatility = valid_returns.rolling(vol_window).std().dropna()
            vol_labels = valid_labels[vol_window-1:]
            
            # Calculate volatility regimes
            vol_regimes = self._calculate_volatility_regimes(volatility)
            
            # Calculate volatility persistence score
            persistence_score = self._calculate_volatility_persistence_score(vol_regimes, vol_labels)
            
            # Calculate volatility consistency
            consistency_score = self._calculate_volatility_consistency(vol_regimes, vol_labels)
            
            return {
                'enabled': True,
                'volatility_regimes': vol_regimes,
                'persistence_score': persistence_score,
                'consistency_score': consistency_score,
                'overall_score': (persistence_score + consistency_score) / 2,
                'n_volatility_regimes': len(set(vol_regimes))
            }
            
        except Exception as e:
            logger.warning(f"Volatility persistence metrics calculation failed: {e}")
            return {'enabled': True, 'error': str(e)}
    
    def _calculate_volatility_regimes(self, volatility: pd.Series) -> np.ndarray:
        """Calculate volatility regimes (high/medium/low)."""
        try:
            # Calculate percentiles
            vol_33 = volatility.quantile(0.33)
            vol_67 = volatility.quantile(0.67)
            
            # Assign regimes
            regimes = np.ones(len(volatility))  # Medium by default
            regimes[volatility <= vol_33] = 0  # Low volatility
            regimes[volatility >= vol_67] = 2  # High volatility
            
            return regimes
            
        except Exception as e:
            logger.warning(f"Volatility regime calculation failed: {e}")
            return np.ones(len(volatility))
    
    def _calculate_volatility_persistence_score(self, vol_regimes: np.ndarray, cluster_labels: np.ndarray) -> float:
        """Calculate volatility persistence score."""
        try:
            # Calculate regime changes
            vol_changes = np.sum(vol_regimes[1:] != vol_regimes[:-1])
            cluster_changes = np.sum(cluster_labels[1:] != cluster_labels[:-1])
            
            # Calculate persistence ratio
            if cluster_changes > 0:
                persistence_ratio = 1 - (vol_changes / cluster_changes)
            else:
                persistence_ratio = 1.0
            
            return max(0, persistence_ratio)
            
        except Exception as e:
            logger.warning(f"Volatility persistence score calculation failed: {e}")
            return 0.0
    
    def _calculate_volatility_consistency(self, vol_regimes: np.ndarray, cluster_labels: np.ndarray) -> float:
        """Calculate volatility consistency within clusters."""
        try:
            unique_labels = np.unique(cluster_labels)
            consistency_scores = []
            
            for label in unique_labels:
                mask = cluster_labels == label
                regime_vol_regimes = vol_regimes[mask]
                
                if len(regime_vol_regimes) > 5:  # Minimum samples
                    # Calculate regime consistency
                    regime_consistency = 1 - (np.sum(regime_vol_regimes[1:] != regime_vol_regimes[:-1]) / len(regime_vol_regimes))
                    consistency_scores.append(regime_consistency)
            
            if consistency_scores:
                return np.mean(consistency_scores)
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Volatility consistency calculation failed: {e}")
            return 0.0
    
    def _perform_statistical_tests(self, 
                                 cluster_labels: np.ndarray,
                                 market_data: pd.DataFrame,
                                 lifespan_metrics: Dict[str, Any],
                                 transition_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical tests for regime persistence."""
        try:
            if not self.config.enable_statistical_tests:
                return {'enabled': False}
            
            tests = {'enabled': True}
            
            # Test regime lifespan distribution
            if 'regime_lifespans' in lifespan_metrics:
                regime_lifespans = lifespan_metrics['regime_lifespans']
                
                # Test against exponential distribution (random regime changes)
                try:
                    from scipy.stats import kstest
                    # Generate exponential distribution with same mean
                    mean_lifespan = np.mean(regime_lifespans)
                    exp_dist = np.random.exponential(mean_lifespan, len(regime_lifespans))
                    
                    ks_stat, p_value = kstest(regime_lifespans, lambda x: 1 - np.exp(-x/mean_lifespan))
                    
                    tests['lifespan_distribution'] = {
                        'ks_statistic': ks_stat,
                        'p_value': p_value,
                        'significant': p_value < self.config.significance_level,
                        'interpretation': 'Significant if p < 0.05 (regimes are not random)'
                    }
                except Exception as e:
                    tests['lifespan_distribution'] = {'error': str(e)}
            
            # Test transition patterns
            if 'transitions' in transition_metrics:
                transitions = transition_metrics['transitions']
                
                if len(transitions) > 5:
                    # Test for transition clustering (non-random transitions)
                    try:
                        transition_positions = [t['position'] for t in transitions]
                        
                        # Calculate inter-transition intervals
                        intervals = np.diff(transition_positions)
                        
                        if len(intervals) > 2:
                            # Test against uniform distribution
                            from scipy.stats import chi2_contingency
                            
                            # Create bins for chi-square test
                            n_bins = min(5, len(intervals))
                            bin_edges = np.linspace(min(intervals), max(intervals), n_bins + 1)
                            observed, _ = np.histogram(intervals, bins=bin_edges)
                            expected = np.full(n_bins, len(intervals) / n_bins)
                            
                            chi2_stat, p_value, _, _ = chi2_contingency([observed, expected])
                            
                            tests['transition_patterns'] = {
                                'chi2_statistic': chi2_stat,
                                'p_value': p_value,
                                'significant': p_value < self.config.significance_level,
                                'interpretation': 'Significant if p < 0.05 (transitions are not random)'
                            }
                    except Exception as e:
                        tests['transition_patterns'] = {'error': str(e)}
            
            return tests
            
        except Exception as e:
            logger.warning(f"Statistical tests failed: {e}")
            return {'enabled': True, 'error': str(e)}
    
    def _calculate_overall_scores(self, 
                                lifespan_metrics: Dict[str, Any],
                                transition_metrics: Dict[str, Any],
                                economic_coherence_metrics: Dict[str, Any],
                                volatility_persistence_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall persistence scores."""
        try:
            scores = {}
            
            # Lifespan score
            if lifespan_metrics.get('enabled', False) and 'lifespan_score' in lifespan_metrics:
                scores['lifespan'] = lifespan_metrics['lifespan_score']
            else:
                scores['lifespan'] = 0.0
            
            # Transition score
            if transition_metrics.get('enabled', False) and 'transition_score' in transition_metrics:
                scores['transition'] = transition_metrics['transition_score']
            else:
                scores['transition'] = 0.0
            
            # Economic coherence score
            if economic_coherence_metrics.get('enabled', False) and 'overall_coherence' in economic_coherence_metrics:
                scores['economic_coherence'] = economic_coherence_metrics['overall_coherence']
            else:
                scores['economic_coherence'] = 0.0
            
            # Volatility persistence score
            if volatility_persistence_metrics.get('enabled', False) and 'overall_score' in volatility_persistence_metrics:
                scores['volatility_persistence'] = volatility_persistence_metrics['overall_score']
            else:
                scores['volatility_persistence'] = 0.0
            
            # Overall score
            overall_score = np.mean(list(scores.values()))
            scores['overall'] = overall_score
            
            return scores
            
        except Exception as e:
            logger.warning(f"Overall score calculation failed: {e}")
            return {'overall': 0.0, 'lifespan': 0.0, 'transition': 0.0, 'economic_coherence': 0.0, 'volatility_persistence': 0.0}