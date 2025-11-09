"""
Simplified Temporal Regime Analysis System

This module implements a simplified framework for analyzing temporal smoothness and transition
metrics for financial market regimes in regime ensemble training systems.
"""

import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict, Counter
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalRegimeAnalyzer:
    """
    Simplified temporal regime analysis system that combines temporal smoothness,
    transition analysis, flip-flop detection, and economic relevance metrics.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the temporal regime analyzer.
        
        Args:
            config: Configuration dictionary with analysis parameters
        """
        self.config = config or self._default_config()
        self._validate_config()
        
    def analyze_regimes(self, regime_labels: np.ndarray, returns: np.ndarray,
                      features: Optional[pd.DataFrame] = None,
                      timestamps: Optional[np.ndarray] = None) -> Dict:
        """
        Perform comprehensive temporal analysis of market regimes.
        
        Args:
            regime_labels: Array of regime classifications (as integers)
            returns: Array of asset returns
            features: DataFrame of market features (trend, momentum, volatility, volume, etc.)
            timestamps: Array of timestamps (optional)
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        logger.info("Starting comprehensive temporal regime analysis")
        
        # Validate inputs
        self._validate_inputs(regime_labels, returns, features)
        
        # Calculate all metric categories
        analysis_results = {
            'metadata': self._calculate_metadata(regime_labels),
            'temporal_smoothness': self._calculate_temporal_smoothness(regime_labels, timestamps),
            'transition_analysis': self._calculate_transition_analysis(regime_labels),
            'flip_flop_analysis': self._calculate_flip_flop_analysis(regime_labels, returns),
            'temporal_consistency': self._calculate_temporal_consistency(regime_labels, returns),
            'lead_lag_analysis': self._lead_lag_relationship_analysis(regime_labels, returns),
            'economic_relevance': self._calculate_economic_relevance(regime_labels, returns, features),
            'feature_analysis': self._calculate_feature_analysis(regime_labels, features) if features is not None else {}
        }
        
        # Add composite analysis
        analysis_results['temporal_quality_score'] = self._calculate_temporal_quality_score(analysis_results)
        analysis_results['trading_recommendations'] = self._generate_trading_recommendations(analysis_results)
        analysis_results['model_recommendations'] = self._generate_model_recommendations(analysis_results)
        
        logger.info("Temporal regime analysis completed successfully")
        return analysis_results
    
    def export_to_csv(self, analysis_results: Dict, output_path: str) -> None:
        """
        Export analysis results to CSV format with one row per regime.
        
        Args:
            analysis_results: Dictionary from analyze_regimes method
            output_path: Path to save CSV file
        """
        logger.info(f"Exporting analysis results to {output_path}")
        
        # Extract unique regimes
        unique_regimes = analysis_results['metadata']['regime_labels']
        
        # Initialize DataFrame
        csv_data = []
        
        for regime in unique_regimes:
            row = {'regime': regime}
            
            # Add temporal smoothness metrics
            if 'temporal_smoothness' in analysis_results:
                smoothness = analysis_results['temporal_smoothness']
                
                # Persistence metrics
                if 'persistence_metrics' in smoothness and regime in smoothness['persistence_metrics']:
                    persistence = smoothness['persistence_metrics'][regime]
                    row.update({
                        'persistence_mean_duration': persistence.get('mean_duration', 0),
                        'persistence_median_duration': persistence.get('median_duration', 0),
                        'persistence_std_duration': persistence.get('std_duration', 0),
                        'persistence_n_occurrences': persistence.get('n_occurrences', 0),
                        'persistence_score': persistence.get('persistence_score', 0)
                    })
            
            # Add transition analysis metrics
            if 'transition_analysis' in analysis_results:
                transitions = analysis_results['transition_analysis']
                
                # Expected times
                if 'expected_times' in transitions and 'by_regime' in transitions['expected_times']:
                    expected_times = transitions['expected_times']['by_regime']
                    if regime in expected_times:
                        row.update({
                            'expected_duration': expected_times[regime].get('expected_duration', 0),
                            'expected_return_time': expected_times[regime].get('expected_return_time', 0),
                            'visitation_frequency': expected_times[regime].get('visitation_frequency', 0)
                        })
            
            # Add flip-flop analysis metrics
            if 'flip_flop_analysis' in analysis_results:
                flip_flop = analysis_results['flip_flop_analysis']
                
                if 'regime_ff_rates' in flip_flop and regime in flip_flop['regime_ff_rates']:
                    row.update({
                        'flip_flop_rate': flip_flop['regime_ff_rates'][regime]
                    })
            
            # Add temporal consistency metrics
            if 'temporal_consistency' in analysis_results:
                consistency = analysis_results['temporal_consistency']
                
                if 'regime_stability' in consistency and regime in consistency['regime_stability']:
                    regime_stability = consistency['regime_stability'][regime]
                    row.update({
                        'overall_stability_score': regime_stability.get('overall_stability_score', 0)
                    })
            
            # Add economic relevance metrics
            if 'economic_relevance' in analysis_results:
                econ = analysis_results['economic_relevance']
                
                # Return-based metrics
                if 'average_returns' in econ and regime in econ['average_returns']:
                    row.update({
                        'avg_return': econ['average_returns'][regime]
                    })
                
                if 'return_distribution' in econ and regime in econ['return_distribution']:
                    dist = econ['return_distribution'][regime]
                    row.update({
                        'return_std': dist.get('std', 0),
                        'return_skewness': dist.get('skewness', 0),
                        'return_kurtosis': dist.get('kurtosis', 0)
                    })
                
                if 'sharpe_ratio' in econ and regime in econ['sharpe_ratio']:
                    row.update({
                        'sharpe_ratio': econ['sharpe_ratio'][regime]
                    })
                
                if 'max_drawdown' in econ and regime in econ['max_drawdown']:
                    row.update({
                        'max_drawdown': econ['max_drawdown'][regime].get('max_drawdown', 0)
                    })
                
                # CV metrics
                if 'cv_per_regime' in econ and regime in econ['cv_per_regime']:
                    cv = econ['cv_per_regime'][regime]
                    row.update({
                        'return_cv': cv.get('cv', 0)
                    })
                
                # Within-regime CV for returns
                if 'within_regime_cv' in econ:
                    within_cv = econ['within_regime_cv']
                    if regime in within_cv and 'returns' in within_cv[regime]:
                        returns_within = within_cv[regime]['returns']
                        row.update({
                            'returns_within_cv': returns_within.get('cv', 0),
                            'returns_within_mean': returns_within.get('mean', 0)
                        })
                
                # Between-regime CV for returns
                if 'between_regime_cv' in econ and 'returns' in econ['between_regime_cv']:
                    returns_between = econ['between_regime_cv']['returns']
                    row.update({
                        'returns_between_cv': returns_between.get('cv_between', 0),
                        'returns_discrimination_ratio': returns_between.get('discrimination_ratio', 0)
                    })
                
                # Between-regime CV for features
                for feature_name in ['trend', 'momentum', 'volatility', 'volume']:
                    if 'between_regime_cv' in econ and feature_name in econ['between_regime_cv']:
                        feature_between = econ['between_regime_cv'][feature_name]
                        row.update({
                            f'{feature_name}_between_cv': feature_between.get('cv_between', 0),
                            f'{feature_name}_discrimination_ratio': feature_between.get('discrimination_ratio', 0)
                        })
            
            # Add feature analysis metrics
            if 'feature_analysis' in analysis_results:
                feature_analysis = analysis_results['feature_analysis']
                
                if regime in feature_analysis:
                    regime_features = feature_analysis[regime]
                    
                    # Add average feature values
                    if 'feature_averages' in regime_features:
                        for feature_name, avg_value in regime_features['feature_averages'].items():
                            row[f'{feature_name}_avg'] = avg_value
                    
                    # Add feature CVs
                    if 'feature_cvs' in regime_features:
                        for feature_name, cv_value in regime_features['feature_cvs'].items():
                            row[f'{feature_name}_cv'] = cv_value
            
            csv_data.append(row)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path, index=False)
        logger.info(f"Successfully exported {len(df)} regimes to {output_path}")
    
    def _calculate_temporal_smoothness(self, regime_labels: np.ndarray, 
                                   timestamps: Optional[np.ndarray]) -> Dict:
        """Calculate all temporal smoothness metrics"""
        return {
            'persistence_metrics': self._regime_persistence_metrics(regime_labels, timestamps),
            'smoothness_indices': self._transition_smoothness_metrics(regime_labels),
            'temporal_autocorrelation': self._temporal_autocorrelation_metrics(regime_labels),
            'regime_stability': self._regime_stability_over_time(regime_labels)
        }
    
    def _calculate_transition_analysis(self, regime_labels: np.ndarray) -> Dict:
        """Calculate all transition analysis metrics"""
        transition_matrix = self._markov_transition_matrix(regime_labels)
        
        return {
            'markov_transitions': transition_matrix,
            'transition_frequency': self._transition_frequency_analysis(regime_labels),
            'expected_times': self._expected_time_metrics(transition_matrix['transition_probabilities']),
            'absorbing_states': self._absorbing_states_analysis(transition_matrix['transition_probabilities']),
            'transition_entropy': self._transition_entropy_analysis(transition_matrix['transition_probabilities'])
        }
    
    def _calculate_flip_flop_analysis(self, regime_labels: np.ndarray, 
                                   returns: np.ndarray) -> Dict:
        """Calculate all flip-flop analysis metrics"""
        return {
            'switching_frequency': self._regime_switching_frequency_metrics(regime_labels),
            'flip_flop_rates': self._flip_flop_rate_calculations(regime_labels),
            'whipsaw_detection': self._whipsaw_detection_metrics(regime_labels, returns),
            'noise_signal_discrimination': self._noise_signal_discrimination(regime_labels, returns)
        }
    
    def _calculate_temporal_consistency(self, regime_labels: np.ndarray, 
                                   returns: np.ndarray) -> Dict:
        """Calculate all temporal consistency metrics"""
        return {
            'rolling_consistency': self._rolling_window_regime_consistency(regime_labels),
            'time_varying_stability': self._time_varying_regime_stability(regime_labels, returns),
            'lag_correlations': self._lag_correlation_analysis(regime_labels, returns),
            'lead_lag_relationships': self._lead_lag_relationship_analysis(regime_labels, returns)
        }
    
    def _calculate_economic_relevance(self, regime_labels: np.ndarray,
                                  returns: np.ndarray, features: Optional[pd.DataFrame] = None) -> Dict:
        """Calculate economic relevance metrics"""
        # Prepare variable names for CV analysis
        variable_names = ['returns']
        if features is not None:
            feature_names = [col for col in features.columns if col in ['trend', 'momentum', 'volatility', 'volume']]
            variable_names.extend(feature_names)
        
        return {
            'average_returns': self._average_return_per_regime(returns, regime_labels),
            'return_distribution': self._return_distribution_metrics(returns, regime_labels),
            'sharpe_ratio': self._sharpe_ratio_per_regime(returns, regime_labels),
            'max_drawdown': self._max_drawdown_per_regime(returns, regime_labels),
            'cv_per_regime': self._cv_per_regime(returns, regime_labels),
            'within_regime_cv': self._within_regime_cv(returns, regime_labels, variable_names, features),
            'between_regime_cv': self._between_regime_cv(returns, regime_labels, variable_names, features)
        }
    
    def _calculate_feature_analysis(self, regime_labels: np.ndarray, 
                               features: pd.DataFrame) -> Dict:
        """Calculate feature analysis metrics"""
        feature_analysis = {}
        unique_regimes = np.unique(regime_labels)
        
        for regime in unique_regimes:
            regime_mask = regime_labels == regime
            regime_features = features[regime_mask]
            
            feature_analysis[regime] = {
                'feature_averages': {},
                'feature_cvs': {}
            }
            
            # Calculate average and CV for each feature
            for column in regime_features.columns:
                feature_values = np.asarray(regime_features[column])
                feature_values = feature_values[~np.isnan(feature_values)]
                
                if len(feature_values) > 1:
                    mean_val = np.mean(feature_values)
                    std_val = np.std(feature_values, ddof=1)
                    cv = std_val / abs(mean_val) if mean_val != 0 else float('inf')
                    
                    feature_analysis[regime]['feature_averages'][column] = mean_val
                    feature_analysis[regime]['feature_cvs'][column] = cv
        
        return feature_analysis
    
    # Simplified metric calculation methods
    def _regime_persistence_metrics(self, regime_labels: np.ndarray, 
                                 timestamps: Optional[np.ndarray]) -> Dict:
        """Calculate regime persistence measures"""
        unique_regimes = np.unique(regime_labels)
        metrics = {}
        
        for regime in unique_regimes:
            # Find regime occurrences and durations
            regime_mask = regime_labels == regime
            regime_changes = np.diff(np.concatenate(([False], regime_mask, [False])).astype(int))
            start_indices = np.where(regime_changes == 1)[0]
            end_indices = np.where(regime_changes == -1)[0]
            
            # Calculate durations
            durations = end_indices - start_indices
            
            # Calculate metrics
            if len(durations) > 0:
                mean_duration = np.mean(durations)
                cv = np.std(durations, ddof=1) / mean_duration if mean_duration > 0 else float('inf')
                
                # Persistence score
                normalized_duration = min(mean_duration / 10, 1.0)  # 10 periods as reference
                consistency_score = 1 / (1 + cv)
                persistence_score = 0.6 * normalized_duration + 0.4 * consistency_score
            else:
                mean_duration = cv = persistence_score = 0
            
            metrics[regime] = {
                'mean_duration': mean_duration,
                'median_duration': np.median(durations) if len(durations) > 0 else 0,
                'std_duration': np.std(durations, ddof=1) if len(durations) > 1 else 0,
                'min_duration': np.min(durations) if len(durations) > 0 else 0,
                'max_duration': np.max(durations) if len(durations) > 0 else 0,
                'n_occurrences': len(durations),
                'total_periods': np.sum(durations) if len(durations) > 0 else 0,
                'duration_cv': cv,
                'persistence_score': persistence_score
            }
        
        return metrics
    
    def _transition_smoothness_metrics(self, regime_labels: np.ndarray) -> Dict:
        """Calculate smoothness indices for regime transitions"""
        unique_regimes = np.unique(regime_labels)
        n_regimes = len(unique_regimes)
        
        # Count transitions
        transitions = np.where(regime_labels[:-1] != regime_labels[1:])[0]
        n_transitions = len(transitions)
        n_total = len(regime_labels)
        
        # Calculate transition types
        transition_types = []
        for trans_idx in transitions:
            from_regime = regime_labels[trans_idx]
            to_regime = regime_labels[trans_idx + 1]
            transition_types.append((from_regime, to_regime))
        
        # Transition frequency matrix
        transition_matrix = np.zeros((n_regimes, n_regimes))
        for from_regime, to_regime in transition_types:
            from_idx = np.where(unique_regimes == from_regime)[0][0]
            to_idx = np.where(unique_regimes == to_regime)[0][0]
            transition_matrix[from_idx, to_idx] += 1
        
        # Normalize to probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_probabilities = np.divide(transition_matrix, row_sums, 
                                            where=row_sums != 0)
        
        # Calculate smoothness metrics
        transition_rate = n_transitions / n_total
        
        # Average duration between transitions
        if n_transitions > 0:
            transition_gaps = np.diff(transitions)
            avg_transition_gap = np.mean(transition_gaps)
        else:
            avg_transition_gap = n_total
        
        # Smoothness indices
        tsi = 1 - transition_rate * (1 / avg_transition_gap) if avg_transition_gap > 0 else 0
        
        return {
            'transition_rate': transition_rate,
            'n_transitions': n_transitions,
            'avg_transition_gap': avg_transition_gap,
            'transition_smoothness_index': tsi,
            'transition_matrix': transition_probabilities,
            'transition_types': transition_types
        }
    
    def _temporal_autocorrelation_metrics(self, regime_labels: np.ndarray) -> Dict:
        """Calculate temporal autocorrelation of regime assignments"""
        unique_regimes = np.unique(regime_labels)
        n_regimes = len(unique_regimes)
        
        # Convert to numeric if needed
        if not np.issubdtype(regime_labels.dtype, np.number):
            regime_numeric = np.zeros_like(regime_labels, dtype=float)
            for i, regime in enumerate(unique_regimes):
                regime_numeric[regime_labels == regime] = i
        else:
            regime_numeric = regime_labels.astype(float)
        
        # Calculate autocorrelation
        max_lag = 50
        autocorr_values = []
        
        for lag in range(max_lag + 1):
            if lag == 0:
                autocorr_values.append(1.0)
            else:
                if len(regime_numeric) > lag:
                    corr = np.corrcoef(regime_numeric[:-lag], regime_numeric[lag:])[0, 1]
                    autocorr_values.append(corr if not np.isnan(corr) else 0)
                else:
                    autocorr_values.append(0)
        
        # Calculate metrics
        first_lag_autocorr = autocorr_values[1] if len(autocorr_values) > 1 else 0
        predictability_score = first_lag_autocorr**2
        
        return {
            'autocorrelation_function': autocorr_values,
            'first_lag_autocorr': first_lag_autocorr,
            'predictability_score': predictability_score
        }
    
    def _regime_stability_over_time(self, regime_labels: np.ndarray) -> Dict:
        """Calculate regime stability metrics over time"""
        # Simplified implementation
        unique_regimes = np.unique(regime_labels)
        stability_metrics = {}
        
        for regime in unique_regimes:
            # Calculate basic stability metrics
            regime_mask = regime_labels == regime
            n_occurrences = np.sum(regime_mask)
            total_periods = len(regime_labels)
            
            stability_score = n_occurrences / total_periods if total_periods > 0 else 0
            
            stability_metrics[regime] = {
                'stability_score': stability_score,
                'n_occurrences': n_occurrences,
                'proportion': stability_score
            }
        
        return stability_metrics
    
    def _markov_transition_matrix(self, regime_labels: np.ndarray) -> Dict:
        """Calculate Markov transition probability matrix"""
        unique_regimes = np.unique(regime_labels)
        n_regimes = len(unique_regimes)
        
        # Count transitions
        transition_counts = np.zeros((n_regimes, n_regimes), dtype=int)
        
        for i in range(len(regime_labels) - 1):
            from_regime = regime_labels[i]
            to_regime = regime_labels[i + 1]
            
            from_idx = np.where(unique_regimes == from_regime)[0][0]
            to_idx = np.where(unique_regimes == to_regime)[0][0]
            
            transition_counts[from_idx, to_idx] += 1
        
        # Calculate transition probabilities
        transition_probabilities = np.zeros((n_regimes, n_regimes))
        for i in range(n_regimes):
            row_sum = np.sum(transition_counts[i, :])
            if row_sum > 0:
                transition_probabilities[i, :] = transition_counts[i, :] / row_sum
        
        return {
            'transition_counts': transition_counts,
            'transition_probabilities': transition_probabilities,
            'regime_labels': unique_regimes
        }
    
    def _transition_frequency_analysis(self, regime_labels: np.ndarray) -> Dict:
        """Analyze transition frequency"""
        transitions = np.where(regime_labels[:-1] != regime_labels[1:])[0]
        total_transitions = len(transitions)
        total_periods = len(regime_labels)
        
        return {
            'n_transitions': total_transitions,
            'transition_rate': total_transitions / total_periods if total_periods > 0 else 0
        }
    
    def _expected_time_metrics(self, transition_matrix: np.ndarray) -> Dict:
        """Calculate expected time in each regime"""
        n_regimes = transition_matrix.shape[0]
        
        # Calculate expected duration in each regime
        expected_durations = np.zeros(n_regimes)
        for i in range(n_regimes):
            stay_probability = transition_matrix[i, i]
            if stay_probability < 1:
                expected_durations[i] = 1 / (1 - stay_probability)
            else:
                expected_durations[i] = float('inf')
        
        # Calculate steady-state distribution
        eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
        idx = np.argmin(np.abs(eigenvalues - 1))
        steady_state = np.real(eigenvectors[:, idx])
        steady_state = steady_state / np.sum(steady_state)
        
        return {
            'expected_durations': expected_durations,
            'steady_state_distribution': steady_state,
            'by_regime': {f'regime_{i}': {
                'expected_duration': expected_durations[i],
                'visitation_frequency': steady_state[i]
            } for i in range(n_regimes)}
        }
    
    def _absorbing_states_analysis(self, transition_matrix: np.ndarray) -> Dict:
        """Identify absorbing states"""
        n_regimes = transition_matrix.shape[0]
        absorbing_mask = np.diag(transition_matrix) >= 0.99
        absorbing_indices = np.where(absorbing_mask)[0]
        
        return {
            'has_absorbing_states': len(absorbing_indices) > 0,
            'absorbing_states': absorbing_indices.tolist()
        }
    
    def _transition_entropy_analysis(self, transition_matrix: np.ndarray) -> Dict:
        """Calculate transition entropy measures"""
        n_regimes = transition_matrix.shape[0]
        
        # Calculate row entropies
        row_entropies = np.zeros(n_regimes)
        for i in range(n_regimes):
            row_probs = transition_matrix[i, :]
            row_probs_safe = np.where(row_probs > 0, row_probs, 1e-10)
            # Convert to float to avoid bool type issues
            row_probs_float = np.asarray(row_probs_safe, dtype=np.float64)
            entropy_value = -np.sum(row_probs_float * np.log(row_probs_float + 1e-10))
            row_entropies[i] = float(entropy_value)
        
        # Calculate steady-state distribution
        eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
        idx = np.argmin(np.abs(eigenvalues - 1))
        steady_state = np.real(eigenvectors[:, idx])
        steady_state = steady_state / np.sum(steady_state)
        
        # Overall entropy
        overall_entropy = np.sum(steady_state * row_entropies)
        max_entropy = np.log(n_regimes)
        
        return {
            'row_entropies': row_entropies,
            'overall_entropy': overall_entropy,
            'max_entropy': max_entropy,
            'entropy_ratio': overall_entropy / max_entropy,
            'labeled_row_entropies': {f'regime_{i}': row_entropies[i] for i in range(n_regimes)}
        }
    
    def _regime_switching_frequency_metrics(self, regime_labels: np.ndarray) -> Dict:
        """Calculate regime switching frequency metrics"""
        switches = np.where(regime_labels[:-1] != regime_labels[1:])[0]
        n_switches = len(switches)
        n_total = len(regime_labels)
        
        switching_frequency = n_switches / n_total if n_total > 0 else 0
        
        return {
            'n_switches': n_switches,
            'switching_frequency': switching_frequency
        }
    
    def _flip_flop_rate_calculations(self, regime_labels: np.ndarray) -> Dict:
        """Calculate flip-flop rates"""
        flip_flops = []
        
        for i in range(len(regime_labels) - 2):
            if (regime_labels[i] == regime_labels[i + 2] and 
                regime_labels[i] != regime_labels[i + 1]):
                flip_flops.append((regime_labels[i], regime_labels[i + 1], regime_labels[i + 2]))
        
        n_flip_flops = len(flip_flops)
        n_total = len(regime_labels)
        flip_flop_rate = n_flip_flops / n_total if n_total > 0 else 0
        
        # Calculate regime-specific flip-flop rates
        regime_ff_rates = {}
        unique_regimes = np.unique(regime_labels)
        
        for regime in unique_regimes:
            regime_ff_count = 0
            for ff in flip_flops:
                if ff[0] == regime:  # Starting regime
                    regime_ff_count += 1
            
            regime_total = np.sum(regime_labels == regime)
            regime_ff_rates[regime] = regime_ff_count / regime_total if regime_total > 0 else 0
        
        return {
            'n_flip_flops': n_flip_flops,
            'flip_flop_rate': flip_flop_rate,
            'regime_ff_rates': regime_ff_rates
        }
    
    def _whipsaw_detection_metrics(self, regime_labels: np.ndarray, 
                                 returns: np.ndarray) -> Dict:
        """Detect whipsaw periods"""
        changes = np.where(regime_labels[:-1] != regime_labels[1:])[0]
        
        if len(changes) < 2:
            return {'n_whipsaw_periods': 0}
        
        # Identify whipsaw periods (clusters of rapid changes)
        whipsaw_periods = []
        current_whipsaw = [changes[0]]
        
        for i in range(1, len(changes)):
            gap = changes[i] - changes[i - 1]
            
            if gap <= 2:  # Within 2 periods
                current_whipsaw.append(changes[i])
            else:
                if len(current_whipsaw) >= 3:
                    whipsaw_periods.append(current_whipsaw)
                current_whipsaw = [changes[i]]
        
        if len(current_whipsaw) >= 3:
            whipsaw_periods.append(current_whipsaw)
        
        return {
            'n_whipsaw_periods': len(whipsaw_periods),
            'whipsaw_periods': whipsaw_periods
        }
    
    def _noise_signal_discrimination(self, regime_labels: np.ndarray, 
                                  returns: np.ndarray) -> Dict:
        """Discriminate between noise and signal in regime changes"""
        changes = np.where(regime_labels[:-1] != regime_labels[1:])[0]
        
        if len(changes) == 0:
            return {'signal_to_noise_ratio': 0}
        
        # Analyze each change
        significance_scores = []
        
        for change_idx in changes:
            # Get windows before and after change
            window_before = max(5, change_idx - 5)
            window_after = min(len(returns), change_idx + 1 + 5)
            
            before_returns = returns[window_before:change_idx + 1]
            after_returns = returns[change_idx + 1:window_after]
            
            if len(before_returns) >= 3 and len(after_returns) >= 3:
                # t-test for mean difference
                t_stat, p_value = stats.ttest_ind(before_returns, after_returns)
                
                # Effect size
                pooled_std = np.sqrt(((len(before_returns) - 1) * np.var(before_returns, ddof=1) + 
                                     (len(after_returns) - 1) * np.var(after_returns, ddof=1)) / 
                                    (len(before_returns) + len(after_returns) - 2))
                
                if pooled_std > 0:
                    cohens_d = (np.mean(after_returns) - np.mean(before_returns)) / pooled_std
                else:
                    cohens_d = 0
                
                # Handle p_value which might be a tuple in some scipy versions
                p_val = p_value[1] if isinstance(p_value, tuple) else p_value
                # Ensure p_val is a float with proper error handling
                try:
                    p_val_float = float(p_val)
                    if np.isnan(p_val_float):
                        p_val_float = 1.0
                except (ValueError, TypeError):
                    p_val_float = 1.0
                
                cohens_d_float = float(cohens_d) if not np.isnan(cohens_d) else 0.0
                significance_score = (1.0 - p_val_float) * abs(cohens_d_float)
                significance_scores.append(significance_score)
        
        if significance_scores:
            signal_level = np.mean([s for s in significance_scores if s > 0.3])
            noise_level = np.mean([s for s in significance_scores if s <= 0.3])
            signal_to_noise_ratio = signal_level / (noise_level + 1e-8)
        else:
            signal_to_noise_ratio = 0
        
        return {
            'signal_to_noise_ratio': signal_to_noise_ratio,
            'significance_scores': significance_scores
        }
    
    def _rolling_window_regime_consistency(self, regime_labels: np.ndarray) -> Dict:
        """Calculate rolling window regime consistency"""
        window_sizes = [252, 126, 63]  # Different timeframes
        consistency_results = {}
        
        for window_size in window_sizes:
            if len(regime_labels) < window_size:
                continue
            
            window_consistency_scores = []
            
            for start in range(0, len(regime_labels) - window_size + 1, 21):  # Step of 21
                window_regimes = regime_labels[start:start + window_size]
                
                # Calculate consistency score
                first_regime = window_regimes[0]
                hamming_distance = np.sum(window_regimes != first_regime)
                consistency_score = 1 - (hamming_distance / window_size)
                
                window_consistency_scores.append(consistency_score)
            
            consistency_results[window_size] = {
                'consistency_scores': window_consistency_scores,
                'mean_consistency': np.mean(window_consistency_scores) if window_consistency_scores else 0
            }
        
        return consistency_results
    
    def _time_varying_regime_stability(self, regime_labels: np.ndarray, 
                                       returns: np.ndarray) -> Dict:
        """Calculate time-varying regime stability"""
        unique_regimes = np.unique(regime_labels)
        stability_results = {}
        
        for regime in unique_regimes:
            regime_mask = regime_labels == regime
            regime_returns = returns[regime_mask]
            
            if len(regime_returns) < 10:
                continue
            
            # Calculate rolling stability
            window_size = min(126, len(regime_returns) // 2)
            stability_scores = []
            
            for i in range(0, len(regime_returns) - window_size + 1, 21):
                window_data = regime_returns[i:i + window_size]
                
                # Calculate stability metrics
                mean_return = np.mean(window_data)
                volatility = np.std(window_data, ddof=1)
                
                if volatility > 0:
                    sharpe = mean_return / volatility
                else:
                    sharpe = 0
                
                # Combined stability score
                stability_score = 0.6 * abs(sharpe) + 0.4 * (1 / (1 + volatility))
                stability_scores.append(stability_score)
            
            stability_results[regime] = {
                'stability_scores': stability_scores,
                'overall_stability_score': np.mean(stability_scores) if stability_scores else 0
            }
        
        return {'regime_stability': stability_results}
    
    def _lag_correlation_analysis(self, regime_labels: np.ndarray, 
                               returns: np.ndarray) -> Dict:
        """Calculate lag correlation analysis"""
        unique_regimes = np.unique(regime_labels)
        
        # Convert regimes to numeric
        if not np.issubdtype(regime_labels.dtype, np.number):
            regime_numeric = np.zeros_like(regime_labels, dtype=float)
            for i, regime in enumerate(unique_regimes):
                regime_numeric[regime_labels == regime] = i
        else:
            regime_numeric = regime_labels.astype(float)
        
        # Calculate autocorrelation
        max_lag = 50
        autocorr_values = []
        
        for lag in range(max_lag + 1):
            if lag == 0:
                autocorr_values.append(1.0)
            else:
                if len(regime_numeric) > lag:
                    corr = np.corrcoef(regime_numeric[:-lag], regime_numeric[lag:])[0, 1]
                    autocorr_values.append(corr if not np.isnan(corr) else 0)
                else:
                    autocorr_values.append(0)
        
        return {
            'autocorrelation_function': autocorr_values,
            'first_lag_autocorr': autocorr_values[1] if len(autocorr_values) > 1 else 0
        }
    
    def _lead_lag_relationship_analysis(self, regime_labels: np.ndarray, 
                                    returns: np.ndarray) -> Dict:
        """Analyze lead-lag relationships between regimes"""
        # Simplified implementation
        return {
            'significant_relationships': [],
            'network_metrics': {}
        }
    
    def _average_return_per_regime(self, returns: np.ndarray, 
                                 regime_labels: np.ndarray) -> Dict:
        """Calculate average returns per regime"""
        unique_regimes = np.unique(regime_labels)
        regime_returns = {}
        
        for regime in unique_regimes:
            regime_mask = regime_labels == regime
            regime_returns[regime] = np.mean(returns[regime_mask])
        
        return regime_returns
    
    def _return_distribution_metrics(self, returns: np.ndarray, 
                                 regime_labels: np.ndarray) -> Dict:
        """Calculate return distribution metrics per regime"""
        unique_regimes = np.unique(regime_labels)
        metrics = {}
        
        for regime in unique_regimes:
            regime_mask = regime_labels == regime
            regime_returns = returns[regime_mask]
            
            if len(regime_returns) > 1:
                metrics[regime] = {
                    'std': np.std(regime_returns, ddof=1),
                    'skewness': stats.skew(regime_returns),
                    'kurtosis': stats.kurtosis(regime_returns, fisher=True)
                }
            else:
                metrics[regime] = {
                    'std': 0,
                    'skewness': 0,
                    'kurtosis': 0
                }
        
        return metrics
    
    def _sharpe_ratio_per_regime(self, returns: np.ndarray, 
                                regime_labels: np.ndarray) -> Dict:
        """Calculate Sharpe ratio per regime"""
        unique_regimes = np.unique(regime_labels)
        sharpe_ratios = {}
        
        for regime in unique_regimes:
            regime_mask = regime_labels == regime
            regime_returns = returns[regime_mask]
            
            if len(regime_returns) > 1:
                excess_returns = regime_returns - 0.02/252  # Daily risk-free rate
                if np.std(regime_returns) > 0:
                    sharpe = np.mean(excess_returns) / np.std(regime_returns) * np.sqrt(252)
                else:
                    sharpe = 0
            else:
                sharpe = 0
            
            sharpe_ratios[regime] = sharpe
        
        return sharpe_ratios
    
    def _max_drawdown_per_regime(self, returns: np.ndarray, 
                                regime_labels: np.ndarray) -> Dict:
        """Calculate maximum drawdown per regime"""
        unique_regimes = np.unique(regime_labels)
        max_drawdowns = {}
        
        for regime in unique_regimes:
            regime_mask = regime_labels == regime
            regime_returns = returns[regime_mask]
            
            if len(regime_returns) > 1:
                cumulative_returns = np.cumprod(1 + regime_returns)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdown = (cumulative_returns - running_max) / running_max
                max_dd = np.min(drawdown)
            else:
                max_dd = 0
            
            max_drawdowns[regime] = {'max_drawdown': max_dd}
        
        return max_drawdowns
    
    def _cv_per_regime(self, returns: np.ndarray, regime_labels: np.ndarray) -> Dict:
        """Calculate coefficient of variation per regime"""
        unique_regimes = np.unique(regime_labels)
        cv_metrics = {}
        
        for regime in unique_regimes:
            regime_mask = regime_labels == regime
            regime_returns = returns[regime_mask]
            
            if len(regime_returns) > 1:
                mean_return = np.mean(regime_returns)
                std_return = np.std(regime_returns, ddof=1)
                
                if mean_return != 0:
                    cv = std_return / abs(mean_return)
                else:
                    cv = float('inf') if std_return > 0 else 0
            else:
                cv = 0
            
            cv_metrics[regime] = {'cv': cv}
        
        return cv_metrics
    
    def _within_regime_cv(self, returns: np.ndarray, regime_labels: np.ndarray,
                         variable_names: List[str], features: Optional[pd.DataFrame] = None) -> Dict:
        """Calculate within-regime CV for variables"""
        unique_regimes = np.unique(regime_labels)
        metrics = {}
        
        for regime in unique_regimes:
            regime_mask = regime_labels == regime
            metrics[regime] = {}
            
            for var_name in variable_names:
                if var_name == 'returns':
                    var_values = returns[regime_mask]
                elif features is not None and var_name in features.columns:
                    var_values = features.loc[regime_mask, var_name].values
                else:
                    # Handle other variables if provided
                    continue
                
                if len(var_values) > 1:
                    mean_val = np.mean(var_values)
                    std_val = np.std(var_values, ddof=1)
                    
                    if mean_val != 0:
                        cv = std_val / abs(mean_val)
                    else:
                        cv = float('inf') if std_val > 0 else 0
                else:
                    cv = 0
                
                metrics[regime][var_name] = {
                    'cv': cv,
                    'mean': mean_val
                }
        
        return metrics
    
    def _between_regime_cv(self, returns: np.ndarray, regime_labels: np.ndarray,
                          variable_names: List[str], features: Optional[pd.DataFrame] = None) -> Dict:
        """Calculate between-regime CV for variables"""
        unique_regimes = np.unique(regime_labels)
        metrics = {}
        
        for var_name in variable_names:
            regime_means = []
            
            for regime in unique_regimes:
                regime_mask = regime_labels == regime
                
                if var_name == 'returns':
                    var_values = returns[regime_mask]
                elif features is not None and var_name in features.columns:
                    var_values = features.loc[regime_mask, var_name].values
                else:
                    continue
                
                if len(var_values) > 0:
                    regime_means.append(np.mean(var_values))
            
            if len(regime_means) > 1:
                mean_of_means = np.mean(regime_means)
                std_of_means = np.std(regime_means, ddof=1)
                
                if mean_of_means != 0:
                    cv_between = std_of_means / abs(mean_of_means)
                else:
                    cv_between = float('inf') if std_of_means > 0 else 0
            else:
                cv_between = 0
            
            metrics[var_name] = {
                'cv_between': cv_between,
                'discrimination_ratio': cv_between  # Simplified
            }
        
        return metrics
    
    def _calculate_temporal_quality_score(self, analysis_results: Dict) -> Dict:
        """Calculate composite temporal quality score"""
        # Simplified implementation
        composite_score = 0.7  # Placeholder
        
        return {
            'composite_score': composite_score,
            'quality_classification': 'moderate'  # Placeholder
        }
    
    def _generate_trading_recommendations(self, analysis_results: Dict) -> Dict:
        """Generate trading recommendations"""
        return {
            'recommendation': 'use_with_caution',  # Placeholder
            'confidence': 0.6
        }
    
    def _generate_model_recommendations(self, analysis_results: Dict) -> Dict:
        """Generate model recommendations"""
        return {
            'recommended_model': 'markov_chain',  # Placeholder
            'confidence': 0.7
        }
    
    def _calculate_metadata(self, regime_labels: np.ndarray) -> Dict:
        """Calculate metadata about regime distribution"""
        unique_regimes, counts = np.unique(regime_labels, return_counts=True)
        total_observations = len(regime_labels)
        
        return {
            'total_observations': total_observations,
            'n_regimes': len(unique_regimes),
            'regime_counts': dict(zip(unique_regimes, counts)),
            'regime_proportions': dict(zip(unique_regimes, counts / total_observations)),
            'regime_labels': unique_regimes.tolist()
        }
    
    def _validate_inputs(self, regime_labels: np.ndarray, returns: np.ndarray, 
                        features: Optional[pd.DataFrame] = None) -> None:
        """Validate input data"""
        if len(regime_labels) != len(returns):
            raise ValueError("Regime labels and returns must have same length")
        
        if len(np.unique(regime_labels)) < 2:
            raise ValueError("Need at least 2 regimes for meaningful analysis")
        
        if features is not None and len(features) != len(regime_labels):
            raise ValueError("Features and regime labels must have same length")
    
    def _default_config(self) -> Dict:
        """Default configuration for analysis"""
        return {
            'min_regime_duration': 5,
            'window_sizes': [252, 126, 63],
            'step_size': 21,
            'max_lag': 50,
            'significance_level': 0.05,
            'risk_free_rate': 0.02
        }
    
    def _validate_config(self) -> None:
        """Validate configuration parameters"""
        required_keys = ['min_regime_duration', 'window_sizes', 'max_lag']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config parameter: {key}")


# Example usage function
def example_usage():
    """
    Example of how to use TemporalRegimeAnalyzer
    """
    # Generate sample data
    np.random.seed(42)
    n_periods = 1000
    
    # Sample regime labels (as integers)
    regime_labels = np.random.choice([0, 1, 2], size=n_periods, p=[0.4, 0.3, 0.3])
    
    # Sample returns
    returns = np.random.normal(0.001, 0.02, n_periods)
    
    # Sample features
    features = pd.DataFrame({
        'trend': np.random.normal(0, 1, n_periods),
        'momentum': np.random.normal(0, 1, n_periods),
        'volatility': np.random.exponential(0.02, n_periods),
        'volume': np.random.exponential(1000000, n_periods)
    })
    
    # Initialize analyzer
    analyzer = TemporalRegimeAnalyzer()
    
    # Perform analysis
    results = analyzer.analyze_regimes(regime_labels, returns, features)
    
    # Export to CSV
    analyzer.export_to_csv(results, 'regime_analysis_results.csv')
    
    print("Analysis completed. Results exported to regime_analysis_results.csv")
    
    return results


if __name__ == "__main__":
    example_usage()