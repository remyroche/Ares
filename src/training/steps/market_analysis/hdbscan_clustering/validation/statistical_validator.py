"""
Statistical Validation Tools for HDBSCAN Economic Profiling System

This module provides comprehensive validation tools for:
- Statistical measure accuracy
- Regime profiling validation
- Economic validation metrics
- Cross-validation for regime discovery
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import time
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

# Import utilities
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance, tprint_progress, tprint_timer,
    tprint_logged, LogLevel
)

logger = logging.getLogger(__name__)

@dataclass
class ValidationConfig:
    """Configuration for statistical validation."""
    # Cross-validation settings
    n_splits: int = 5
    test_size: float = 0.2
    random_state: int = 42
    
    # Statistical validation thresholds
    min_confidence_level: float = 0.95
    max_p_value: float = 0.05
    min_correlation: float = 0.7
    
    # Regime validation
    min_regime_duration: int = 10
    max_regime_transitions: int = 50
    min_regime_stability: float = 0.8
    
    # Economic validation
    min_sharpe_ratio: float = 0.5
    max_drawdown_threshold: float = 0.2
    min_volatility: float = 0.01
    max_volatility: float = 0.5

class StatisticalValidator:
    """
    Comprehensive statistical validation for the HDBSCAN economic profiling system.
    """
    
    @tprint_logged(LogLevel.INFO, include_args=True)
    def __init__(self, config: Optional[ValidationConfig] = None):
        """Initialize the statistical validator."""
        self.config = config or ValidationConfig()
        self.validation_results = {}
        self.regime_profiles = {}
        self.economic_metrics = {}
    
    def validate_regime_profiling(self, 
                                market_data: pd.DataFrame,
                                regime_labels: np.ndarray,
                                economic_validator: Any) -> Dict[str, Any]:
        """
        Validate regime profiling logic and statistical analysis.
        
        Args:
            market_data: Market data DataFrame with OHLCV data
            regime_labels: Regime labels from clustering
            economic_validator: EconomicValidator instance
            
        Returns:
            Dictionary with validation results
        """
        try:
            tprint_info("Starting regime profiling validation")
            
            validation_results = {
                'regime_profiling': {},
                'statistical_analysis': {},
                'economic_validation': {},
                'cross_validation': {},
                'overall_score': 0.0
            }
            
            # Validate regime profiling logic
            regime_profiling_results = self._validate_regime_profiling_logic(
                market_data, regime_labels, economic_validator
            )
            validation_results['regime_profiling'] = regime_profiling_results
            
            # Validate statistical analysis
            statistical_results = self._validate_statistical_analysis(
                market_data, regime_labels, economic_validator
            )
            validation_results['statistical_analysis'] = statistical_results
            
            # Validate economic metrics
            economic_results = self._validate_economic_metrics(
                market_data, regime_labels, economic_validator
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
            
            self.validation_results = validation_results
            
            tprint_success(f"Regime profiling validation completed. Overall score: {overall_score:.3f}")
            return validation_results
            
        except Exception as e:
            tprint_error(f"Regime profiling validation failed: {e}")
            return {'error': str(e), 'success': False}
    
    def _validate_regime_profiling_logic(self, 
                                       market_data: pd.DataFrame,
                                       regime_labels: np.ndarray,
                                       economic_validator: Any) -> Dict[str, Any]:
        """Validate regime profiling logic."""
        try:
            results = {
                'regime_count': 0,
                'regime_durations': [],
                'regime_transitions': 0,
                'regime_stability': 0.0,
                'regime_naming': {},
                'trading_recommendations': {},
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
            
            if transitions > self.config.max_regime_transitions:
                results['issues'].append(f"Too many regime transitions: {transitions}")
                results['is_valid'] = False
            
            # Calculate regime stability
            stability = self._calculate_regime_stability(regime_labels)
            results['regime_stability'] = stability
            
            if stability < self.config.min_regime_stability:
                results['issues'].append(f"Regime stability too low: {stability:.3f}")
                results['is_valid'] = False
            
            # Test economic validator functionality
            try:
                validation_result = economic_validator.validate_and_profile(
                    market_data, regime_labels
                )
                
                if validation_result and 'regime_profiles' in validation_result:
                    results['regime_naming'] = {
                        profile['regime_name']: profile['characteristics']
                        for profile in validation_result['regime_profiles']
                    }
                
                if validation_result and 'trading_recommendations' in validation_result:
                    results['trading_recommendations'] = validation_result['trading_recommendations']
                
            except Exception as e:
                results['issues'].append(f"Economic validator failed: {e}")
                results['is_valid'] = False
            
            return results
            
        except Exception as e:
            tprint_error(f"Regime profiling logic validation failed: {e}")
            return {'error': str(e), 'is_valid': False}
    
    def _validate_statistical_analysis(self, 
                                     market_data: pd.DataFrame,
                                     regime_labels: np.ndarray,
                                     economic_validator: Any) -> Dict[str, Any]:
        """Validate statistical analysis of regime characteristics."""
        try:
            results = {
                'confidence_intervals': {},
                'statistical_tests': {},
                'correlation_analysis': {},
                'distribution_tests': {},
                'is_valid': True,
                'issues': []
            }
            
            # Test confidence interval calculations
            try:
                validation_result = economic_validator.validate_and_profile(
                    market_data, regime_labels
                )
                
                if validation_result and 'regime_profiles' in validation_result:
                    for profile in validation_result['regime_profiles']:
                        regime_name = profile['regime_name']
                        
                        # Check if confidence intervals are present
                        if 'confidence_intervals' in profile:
                            ci_data = profile['confidence_intervals']
                            results['confidence_intervals'][regime_name] = ci_data
                            
                            # Validate confidence interval structure
                            required_keys = ['mean_return', 'volatility', 'sharpe_ratio']
                            for key in required_keys:
                                if key not in ci_data:
                                    results['issues'].append(f"Missing confidence interval for {key}")
                                    results['is_valid'] = False
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
            
            # Test correlations
            results['correlation_analysis'] = self._test_correlations(
                market_data, regime_labels
            )
            
            # Test distributions
            results['distribution_tests'] = self._test_distributions(
                market_data, regime_labels
            )
            
            return results
            
        except Exception as e:
            tprint_error(f"Statistical analysis validation failed: {e}")
            return {'error': str(e), 'is_valid': False}
    
    def _validate_economic_metrics(self, 
                                 market_data: pd.DataFrame,
                                 regime_labels: np.ndarray,
                                 economic_validator: Any) -> Dict[str, Any]:
        """Validate economic metrics and calculations."""
        try:
            results = {
                'sharpe_ratios': {},
                'volatilities': {},
                'drawdowns': {},
                'volume_analysis': {},
                'is_valid': True,
                'issues': []
            }
            
            # Test economic validator
            try:
                validation_result = economic_validator.validate_and_profile(
                    market_data, regime_labels
                )
                
                if validation_result and 'regime_profiles' in validation_result:
                    for profile in validation_result['regime_profiles']:
                        regime_name = profile['regime_name']
                        
                        # Validate Sharpe ratios
                        if 'sharpe_ratio' in profile:
                            sharpe = profile['sharpe_ratio']
                            results['sharpe_ratios'][regime_name] = sharpe
                            
                            if sharpe < self.config.min_sharpe_ratio:
                                results['issues'].append(f"Low Sharpe ratio for {regime_name}: {sharpe:.3f}")
                                results['is_valid'] = False
                        
                        # Validate volatilities
                        if 'volatility' in profile:
                            vol = profile['volatility']
                            results['volatilities'][regime_name] = vol
                            
                            if vol < self.config.min_volatility or vol > self.config.max_volatility:
                                results['issues'].append(f"Volatility out of range for {regime_name}: {vol:.3f}")
                                results['is_valid'] = False
                        
                        # Validate drawdowns
                        if 'max_drawdown' in profile:
                            dd = abs(profile['max_drawdown'])
                            results['drawdowns'][regime_name] = dd
                            
                            if dd > self.config.max_drawdown_threshold:
                                results['issues'].append(f"High drawdown for {regime_name}: {dd:.3f}")
                                results['is_valid'] = False
                        
                        # Validate volume analysis
                        if 'volume_stats' in profile:
                            results['volume_analysis'][regime_name] = profile['volume_stats']
                
            except Exception as e:
                results['issues'].append(f"Economic metrics validation failed: {e}")
                results['is_valid'] = False
            
            return results
            
        except Exception as e:
            tprint_error(f"Economic metrics validation failed: {e}")
            return {'error': str(e), 'is_valid': False}
    
    def _cross_validate_regime_discovery(self, 
                                       market_data: pd.DataFrame,
                                       regime_labels: np.ndarray) -> Dict[str, Any]:
        """Perform cross-validation for regime discovery."""
        try:
            results = {
                'cv_scores': [],
                'stability_scores': [],
                'consistency_scores': [],
                'mean_cv_score': 0.0,
                'cv_std': 0.0,
                'is_valid': True
            }
            
            # Use time series split for cross-validation
            tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
            
            # Prepare features (simplified - in practice, use proper feature extraction)
            features = market_data[['close', 'volume']].values
            
            cv_scores = []
            stability_scores = []
            consistency_scores = []
            
            for train_idx, test_idx in tscv.split(features):
                try:
                    # Split data
                    train_features = features[train_idx]
                    test_features = features[test_idx]
                    train_labels = regime_labels[train_idx]
                    test_labels = regime_labels[test_idx]
                    
                    # Calculate stability (how consistent are the regimes)
                    stability = self._calculate_regime_stability(train_labels)
                    stability_scores.append(stability)
                    
                    # Calculate consistency (how similar are train/test regimes)
                    consistency = self._calculate_regime_consistency(
                        train_labels, test_labels
                    )
                    consistency_scores.append(consistency)
                    
                    # Calculate CV score (combination of stability and consistency)
                    cv_score = (stability + consistency) / 2
                    cv_scores.append(cv_score)
                    
                except Exception as e:
                    tprint_debug(f"CV fold failed: {e}")
                    continue
            
            if cv_scores:
                results['cv_scores'] = cv_scores
                results['stability_scores'] = stability_scores
                results['consistency_scores'] = consistency_scores
                results['mean_cv_score'] = np.mean(cv_scores)
                results['cv_std'] = np.std(cv_scores)
                
                # Validate CV scores
                if results['mean_cv_score'] < 0.5:
                    results['is_valid'] = False
            else:
                results['is_valid'] = False
            
            return results
            
        except Exception as e:
            tprint_error(f"Cross-validation failed: {e}")
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
    
    def _calculate_regime_consistency(self, 
                                    train_labels: np.ndarray, 
                                    test_labels: np.ndarray) -> float:
        """Calculate consistency between train and test regime labels."""
        # Simple consistency measure - in practice, use more sophisticated methods
        train_regimes = set(train_labels)
        test_regimes = set(test_labels)
        
        if len(train_regimes) == 0 or len(test_regimes) == 0:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(train_regimes.intersection(test_regimes))
        union = len(train_regimes.union(test_regimes))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _test_statistical_significance(self, 
                                     market_data: pd.DataFrame,
                                     regime_labels: np.ndarray) -> Dict[str, Any]:
        """Test statistical significance of regime differences."""
        try:
            results = {
                't_tests': {},
                'anova_tests': {},
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
                                t_stat, p_value = stats.ttest_ind(regime_returns[i], regime_returns[j])
                                results['t_tests'][f'regime_{regime1}_vs_regime_{regime2}'] = {
                                    't_statistic': t_stat,
                                    'p_value': p_value,
                                    'is_significant': p_value < self.config.max_p_value
                                }
                            except Exception as e:
                                tprint_debug(f"T-test failed for regimes {regime1} vs {regime2}: {e}")
                    
                    # ANOVA test
                    try:
                        f_stat, p_value = stats.f_oneway(*regime_returns)
                        results['anova_tests']['all_regimes'] = {
                            'f_statistic': f_stat,
                            'p_value': p_value,
                            'is_significant': p_value < self.config.max_p_value
                        }
                    except Exception as e:
                        tprint_debug(f"ANOVA test failed: {e}")
            
            return results
            
        except Exception as e:
            tprint_error(f"Statistical significance testing failed: {e}")
            return {'error': str(e), 'is_significant': False}
    
    def _test_correlations(self, 
                          market_data: pd.DataFrame,
                          regime_labels: np.ndarray) -> Dict[str, Any]:
        """Test correlations between market variables and regime labels."""
        try:
            results = {
                'correlations': {},
                'is_correlated': True
            }
            
            # Test correlation between regime labels and market variables
            numeric_columns = market_data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                try:
                    # Align data
                    min_len = min(len(market_data[col]), len(regime_labels))
                    data_col = market_data[col].iloc[:min_len]
                    labels_col = regime_labels[:min_len]
                    
                    # Remove noise
                    non_noise_mask = labels_col != -1
                    if np.sum(non_noise_mask) < 2:
                        continue
                    
                    data_clean = data_col[non_noise_mask]
                    labels_clean = labels_col[non_noise_mask]
                    
                    # Calculate correlation
                    correlation = np.corrcoef(data_clean, labels_clean)[0, 1]
                    results['correlations'][col] = {
                        'correlation': correlation,
                        'is_correlated': abs(correlation) > self.config.min_correlation
                    }
                    
                except Exception as e:
                    tprint_debug(f"Correlation test failed for {col}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            tprint_error(f"Correlation testing failed: {e}")
            return {'error': str(e), 'is_correlated': False}
    
    def _test_distributions(self, 
                           market_data: pd.DataFrame,
                           regime_labels: np.ndarray) -> Dict[str, Any]:
        """Test distribution properties of regime characteristics."""
        try:
            results = {
                'normality_tests': {},
                'distribution_fits': {},
                'is_normal': True
            }
            
            # Test normality of returns by regime
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change().dropna()
                
                # Align data
                min_len = min(len(returns), len(regime_labels))
                returns = returns.iloc[:min_len]
                labels = regime_labels[:min_len]
                
                # Remove noise
                non_noise_mask = labels != -1
                if np.sum(non_noise_mask) < 2:
                    return results
                
                returns_clean = returns[non_noise_mask]
                labels_clean = labels[non_noise_mask]
                
                # Test normality for each regime
                unique_regimes = np.unique(labels_clean)
                for regime in unique_regimes:
                    regime_returns = returns_clean[labels_clean == regime]
                    
                    if len(regime_returns) < 3:
                        continue
                    
                    try:
                        # Shapiro-Wilk test for normality
                        shapiro_stat, shapiro_p = stats.shapiro(regime_returns)
                        
                        # Kolmogorov-Smirnov test
                        ks_stat, ks_p = stats.kstest(regime_returns, 'norm')
                        
                        results['normality_tests'][f'regime_{regime}'] = {
                            'shapiro_statistic': shapiro_stat,
                            'shapiro_p_value': shapiro_p,
                            'ks_statistic': ks_stat,
                            'ks_p_value': ks_p,
                            'is_normal': shapiro_p > self.config.max_p_value
                        }
                        
                    except Exception as e:
                        tprint_debug(f"Normality test failed for regime {regime}: {e}")
            
            return results
            
        except Exception as e:
            tprint_error(f"Distribution testing failed: {e}")
            return {'error': str(e), 'is_normal': False}
    
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
            tprint_error(f"Overall score calculation failed: {e}")
            return 0.0
    
    def generate_validation_report(self) -> str:
        """Generate a comprehensive validation report."""
        try:
            if not self.validation_results:
                return "No validation results available. Run validation first."
            
            report = []
            report.append("=" * 80)
            report.append("HDBSCAN ECONOMIC PROFILING SYSTEM - VALIDATION REPORT")
            report.append("=" * 80)
            report.append("")
            
            # Overall score
            overall_score = self.validation_results.get('overall_score', 0.0)
            report.append(f"OVERALL VALIDATION SCORE: {overall_score:.3f}")
            report.append("")
            
            # Regime profiling results
            regime_results = self.validation_results.get('regime_profiling', {})
            report.append("REGIME PROFILING VALIDATION:")
            report.append(f"  Valid: {regime_results.get('is_valid', False)}")
            report.append(f"  Regime Count: {regime_results.get('regime_count', 0)}")
            report.append(f"  Regime Stability: {regime_results.get('regime_stability', 0.0):.3f}")
            report.append(f"  Regime Transitions: {regime_results.get('regime_transitions', 0)}")
            
            if regime_results.get('issues'):
                report.append("  Issues:")
                for issue in regime_results['issues']:
                    report.append(f"    - {issue}")
            report.append("")
            
            # Statistical analysis results
            stat_results = self.validation_results.get('statistical_analysis', {})
            report.append("STATISTICAL ANALYSIS VALIDATION:")
            report.append(f"  Valid: {stat_results.get('is_valid', False)}")
            
            if stat_results.get('issues'):
                report.append("  Issues:")
                for issue in stat_results['issues']:
                    report.append(f"    - {issue}")
            report.append("")
            
            # Economic validation results
            econ_results = self.validation_results.get('economic_validation', {})
            report.append("ECONOMIC VALIDATION:")
            report.append(f"  Valid: {econ_results.get('is_valid', False)}")
            
            if econ_results.get('issues'):
                report.append("  Issues:")
                for issue in econ_results['issues']:
                    report.append(f"    - {issue}")
            report.append("")
            
            # Cross-validation results
            cv_results = self.validation_results.get('cross_validation', {})
            report.append("CROSS-VALIDATION:")
            report.append(f"  Valid: {cv_results.get('is_valid', False)}")
            report.append(f"  Mean CV Score: {cv_results.get('mean_cv_score', 0.0):.3f}")
            report.append(f"  CV Std: {cv_results.get('cv_std', 0.0):.3f}")
            report.append("")
            
            # Recommendations
            report.append("RECOMMENDATIONS:")
            if overall_score < 0.5:
                report.append("  - System needs significant improvements")
            elif overall_score < 0.7:
                report.append("  - System is functional but needs optimization")
            elif overall_score < 0.9:
                report.append("  - System is good with minor improvements needed")
            else:
                report.append("  - System is excellent and ready for production")
            
            report.append("")
            report.append("=" * 80)
            
            return "\n".join(report)
            
        except Exception as e:
            tprint_error(f"Report generation failed: {e}")
            return f"Report generation failed: {e}"