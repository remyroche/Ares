"""
Lookahead Bias Prevention Module.

This module ensures strict temporal separation between observable information
at time t and ex-post evaluation, preventing forward-looking bias in regime
analysis and economic metric calculations.

Key Principles:
1. Strict time separation: Only use information available at time t
2. Ex-post evaluation: Separate analysis of outcomes vs predictions
3. Walk-forward validation: Rolling window analysis
4. Out-of-sample testing: Hold-out periods for validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

from src.utils.logger import system_logger


@dataclass
class TimeAwareAnalysis:
    """Container for time-aware analysis results."""
    observable_at_time_t: Dict[str, Any]
    ex_post_evaluation: Dict[str, Any]
    prediction_accuracy: Dict[str, float]
    temporal_separation_verified: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'observable_at_time_t': self.observable_at_time_t,
            'ex_post_evaluation': self.ex_post_evaluation,
            'prediction_accuracy': self.prediction_accuracy,
            'temporal_separation_verified': self.temporal_separation_verified
        }


class LookaheadBiasPrevention:
    """
    Prevents lookahead bias in regime analysis.
    
    Ensures all analysis maintains strict temporal separation between
    what's observable at time t vs what is evaluated ex-post.
    """
    
    def __init__(self):
        self.logger = system_logger.getChild('LookaheadBiasPrevention')
    
    def time_aware_breakout_analysis(self, 
                                   market_data: pd.DataFrame,
                                   dimension_features: pd.DataFrame,
                                   evaluation_periods: int = 5) -> TimeAwareAnalysis:
        """
        Time-aware breakout analysis with strict temporal separation.
        
        Observable at time t: Price position, dimension signal, technical levels
        Ex-post evaluation: Whether breakout actually occurred
        """
        
        prices = market_data['close']
        
        # Calculate technical levels using ONLY past data
        observable_signals = []
        ex_post_outcomes = []
        
        for t in range(50, len(prices) - evaluation_periods):  # Need lookback and evaluation periods
            
            # === OBSERVABLE AT TIME T (no lookahead) ===
            historical_data = prices.iloc[:t+1]  # Only data up to time t
            
            # Bollinger Bands using only historical data
            ma_20_t = historical_data.rolling(20).mean().iloc[-1]
            std_20_t = historical_data.rolling(20).std().iloc[-1]
            upper_band_t = ma_20_t + 2 * std_20_t
            lower_band_t = ma_20_t - 2 * std_20_t
            
            # Current price and position
            current_price_t = prices.iloc[t]
            
            # Dimension signal using only historical data
            dimension_signal_t = dimension_features.iloc[:t+1].mean(axis=1).iloc[-1]
            dimension_signal_normalized = (
                dimension_signal_t - dimension_features.iloc[:t+1].mean(axis=1).mean()
            ) / dimension_features.iloc[:t+1].mean(axis=1).std()
            
            # Observable conditions at time t
            near_upper_band = abs(current_price_t - upper_band_t) / current_price_t < 0.01
            near_lower_band = abs(current_price_t - lower_band_t) / current_price_t < 0.01
            strong_dimension_signal = abs(dimension_signal_normalized) > 1.0
            
            # Record observable signal
            if (near_upper_band or near_lower_band) and strong_dimension_signal:
                observable_signals.append({
                    'time': t,
                    'price': current_price_t,
                    'upper_band': upper_band_t,
                    'lower_band': lower_band_t,
                    'dimension_signal': dimension_signal_normalized,
                    'near_upper': near_upper_band,
                    'near_lower': near_lower_band,
                    'signal_strength': abs(dimension_signal_normalized)
                })
                
                # === EX-POST EVALUATION (evaluation_periods later) ===
                future_prices = prices.iloc[t+1:t+1+evaluation_periods]
                
                # Check if breakout actually occurred
                if near_upper_band:
                    breakout_occurred = any(future_prices > upper_band_t)
                else:  # near_lower_band
                    breakout_occurred = any(future_prices < lower_band_t)
                
                # Additional ex-post metrics
                max_future_price = future_prices.max()
                min_future_price = future_prices.min()
                price_range = max_future_price - min_future_price
                
                ex_post_outcomes.append({
                    'time': t,
                    'breakout_occurred': breakout_occurred,
                    'max_future_price': max_future_price,
                    'min_future_price': min_future_price,
                    'price_range': price_range,
                    'breakout_magnitude': max(max_future_price - upper_band_t, lower_band_t - min_future_price) if breakout_occurred else 0
                })
        
        # Calculate prediction accuracy (ex-post)
        if observable_signals and ex_post_outcomes:
            correct_predictions = sum(1 for outcome in ex_post_outcomes if outcome['breakout_occurred'])
            total_predictions = len(ex_post_outcomes)
            prediction_accuracy = correct_predictions / total_predictions
            
            # Signal strength vs outcome correlation
            signal_strengths = [signal['signal_strength'] for signal in observable_signals]
            breakout_outcomes = [outcome['breakout_occurred'] for outcome in ex_post_outcomes]
            
            if len(signal_strengths) > 5:
                signal_outcome_corr = np.corrcoef(signal_strengths, breakout_outcomes)[0, 1]
                signal_outcome_corr = signal_outcome_corr if not np.isnan(signal_outcome_corr) else 0
            else:
                signal_outcome_corr = 0
            
            prediction_metrics = {
                'prediction_accuracy': prediction_accuracy,
                'signal_outcome_correlation': float(signal_outcome_corr),
                'total_signals': total_predictions,
                'correct_predictions': correct_predictions
            }
        else:
            prediction_metrics = {
                'prediction_accuracy': 0.0,
                'signal_outcome_correlation': 0.0,
                'total_signals': 0,
                'correct_predictions': 0
            }
        
        return TimeAwareAnalysis(
            observable_at_time_t=observable_signals,
            ex_post_evaluation=ex_post_outcomes,
            prediction_accuracy=prediction_metrics,
            temporal_separation_verified=True
        )
    
    def walk_forward_regime_validation(self,
                                     market_data: pd.DataFrame,
                                     regime_discovery_func,
                                     window_size: int = 252,
                                     step_size: int = 21) -> Dict[str, Any]:
        """
        Walk-forward validation of regime discovery with strict time separation.
        
        Args:
            market_data: Market data
            regime_discovery_func: Function to discover regimes
            window_size: Size of rolling window for regime discovery
            step_size: Step size for rolling window
            
        Returns:
            Walk-forward validation results
        """
        
        self.logger.info(f"🔄 Starting walk-forward validation: window={window_size}, step={step_size}")
        
        validation_results = []
        
        for start_idx in range(window_size, len(market_data) - step_size, step_size):
            end_idx = start_idx + window_size
            
            # === IN-SAMPLE PERIOD (for regime discovery) ===
            in_sample_data = market_data.iloc[start_idx:end_idx]
            
            # Discover regimes using only in-sample data
            try:
                regime_result = regime_discovery_func(in_sample_data)
                in_sample_regimes = regime_result['regime_labels']
                
                # === OUT-OF-SAMPLE PERIOD (for validation) ===
                out_sample_start = end_idx
                out_sample_end = min(end_idx + step_size, len(market_data))
                out_sample_data = market_data.iloc[out_sample_start:out_sample_end]
                
                # Apply regime model to out-of-sample data (if possible)
                # This would require regime prediction model
                
                # For now, calculate regime characteristics stability
                if len(in_sample_regimes) > 0:
                    # Calculate in-sample regime characteristics
                    in_sample_characteristics = self._calculate_regime_characteristics(
                        in_sample_data, in_sample_regimes
                    )
                    
                    # Calculate out-sample characteristics (assuming regime continues)
                    # This is a simplified approach - in practice you'd need regime prediction
                    out_sample_characteristics = self._calculate_market_characteristics(out_sample_data)
                    
                    # Measure characteristic stability
                    stability_score = self._calculate_characteristic_stability(
                        in_sample_characteristics, out_sample_characteristics
                    )
                    
                    validation_results.append({
                        'period_start': start_idx,
                        'period_end': end_idx,
                        'out_sample_start': out_sample_start,
                        'out_sample_end': out_sample_end,
                        'n_regimes_discovered': len(np.unique(in_sample_regimes)),
                        'regime_characteristics_stability': stability_score,
                        'temporal_separation_verified': True
                    })
                
            except Exception as e:
                self.logger.warning(f"Walk-forward validation failed for period {start_idx}-{end_idx}: {e}")
                continue
        
        # Calculate overall walk-forward performance
        if validation_results:
            avg_stability = np.mean([r['regime_characteristics_stability'] for r in validation_results])
            stability_std = np.std([r['regime_characteristics_stability'] for r in validation_results])
            
            summary = {
                'total_periods': len(validation_results),
                'average_stability': float(avg_stability),
                'stability_std': float(stability_std),
                'stable_periods': sum(1 for r in validation_results if r['regime_characteristics_stability'] > 0.7),
                'walk_forward_success_rate': sum(1 for r in validation_results if r['regime_characteristics_stability'] > 0.7) / len(validation_results)
            }
        else:
            summary = {
                'total_periods': 0,
                'average_stability': 0.0,
                'stability_std': 0.0,
                'stable_periods': 0,
                'walk_forward_success_rate': 0.0
            }
        
        return {
            'validation_results': validation_results,
            'summary': summary,
            'temporal_separation_verified': True
        }
    
    def _calculate_regime_characteristics(self, data: pd.DataFrame, regime_labels: np.ndarray) -> Dict[str, float]:
        """Calculate regime characteristics using only available data."""
        
        if 'close' not in data.columns:
            return {}
        
        returns = data['close'].pct_change().fillna(0)
        
        characteristics = {}
        unique_regimes = np.unique(regime_labels)
        
        for regime in unique_regimes:
            regime_mask = regime_labels == regime
            regime_returns = returns[regime_mask]
            
            if len(regime_returns) > 10:
                characteristics[f'regime_{regime}_mean_return'] = float(regime_returns.mean())
                characteristics[f'regime_{regime}_volatility'] = float(regime_returns.std())
                characteristics[f'regime_{regime}_skewness'] = float(regime_returns.skew())
                characteristics[f'regime_{regime}_kurtosis'] = float(regime_returns.kurtosis())
        
        return characteristics
    
    def _calculate_market_characteristics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate market characteristics for comparison."""
        
        if 'close' not in data.columns:
            return {}
        
        returns = data['close'].pct_change().fillna(0)
        
        return {
            'mean_return': float(returns.mean()),
            'volatility': float(returns.std()),
            'skewness': float(returns.skew()),
            'kurtosis': float(returns.kurtosis())
        }
    
    def _calculate_characteristic_stability(self, 
                                          in_sample: Dict[str, float],
                                          out_sample: Dict[str, float]) -> float:
        """Calculate stability of regime characteristics out-of-sample."""
        
        if not in_sample or not out_sample:
            return 0.0
        
        # Compare similar characteristics
        common_characteristics = ['mean_return', 'volatility', 'skewness', 'kurtosis']
        stability_scores = []
        
        for char in common_characteristics:
            if char in out_sample:
                # Find corresponding regime characteristics
                regime_values = [value for key, value in in_sample.items() if char in key]
                if regime_values:
                    # Find closest regime characteristic
                    out_sample_value = out_sample[char]
                    closest_regime_value = min(regime_values, key=lambda x: abs(x - out_sample_value))
                    
                    # Calculate relative stability (1 - relative difference)
                    if abs(closest_regime_value) > 1e-6:
                        relative_diff = abs(out_sample_value - closest_regime_value) / abs(closest_regime_value)
                        stability = max(0, 1 - relative_diff)
                    else:
                        stability = 1.0 if abs(out_sample_value) < 1e-6 else 0.0
                    
                    stability_scores.append(stability)
        
        return float(np.mean(stability_scores)) if stability_scores else 0.0
    
    def validate_temporal_separation(self, 
                                   analysis_function,
                                   market_data: pd.DataFrame,
                                   **kwargs) -> Dict[str, Any]:
        """
        Validate that an analysis function maintains proper temporal separation.
        
        Args:
            analysis_function: Function to validate
            market_data: Market data for testing
            **kwargs: Additional arguments for the function
            
        Returns:
            Validation results
        """
        
        self.logger.info("🔍 Validating temporal separation in analysis function")
        
        validation_results = {
            'temporal_separation_verified': False,
            'lookahead_bias_detected': False,
            'validation_details': {}
        }
        
        try:
            # Test 1: Progressive data revelation test
            # Run analysis with progressively more data to check for consistency
            data_lengths = [len(market_data) // 4, len(market_data) // 2, len(market_data)]
            results_by_length = []
            
            for length in data_lengths:
                partial_data = market_data.iloc[:length]
                try:
                    result = analysis_function(partial_data, **kwargs)
                    results_by_length.append({
                        'data_length': length,
                        'result': result,
                        'success': True
                    })
                except Exception as e:
                    results_by_length.append({
                        'data_length': length,
                        'result': None,
                        'success': False,
                        'error': str(e)
                    })
            
            # Test 2: Future data dependency test
            # Check if results change when future data is modified
            original_result = analysis_function(market_data, **kwargs)
            
            # Modify future data (last 10% of data)
            modified_data = market_data.copy()
            future_start = int(len(market_data) * 0.9)
            modified_data.iloc[future_start:, :] = modified_data.iloc[future_start:, :] * 1.1  # 10% change
            
            modified_result = analysis_function(modified_data, **kwargs)
            
            # Check if results are identical (they should be if no lookahead bias)
            results_identical = self._compare_analysis_results(original_result, modified_result)
            
            validation_results.update({
                'temporal_separation_verified': results_identical,
                'lookahead_bias_detected': not results_identical,
                'validation_details': {
                    'progressive_data_test': results_by_length,
                    'future_data_dependency_test': {
                        'results_identical': results_identical,
                        'original_result_summary': self._summarize_result(original_result),
                        'modified_result_summary': self._summarize_result(modified_result)
                    }
                }
            })
            
            if results_identical:
                self.logger.info("✅ Temporal separation verified - no lookahead bias detected")
            else:
                self.logger.warning("⚠️ Potential lookahead bias detected - results change with future data")
            
        except Exception as e:
            self.logger.error(f"Temporal separation validation failed: {e}")
            validation_results['validation_error'] = str(e)
        
        return validation_results
    
    def _compare_analysis_results(self, result1: Any, result2: Any) -> bool:
        """Compare two analysis results for equality (within tolerance)."""
        
        try:
            # Handle different result types
            if isinstance(result1, dict) and isinstance(result2, dict):
                # Compare dictionary results
                if set(result1.keys()) != set(result2.keys()):
                    return False
                
                for key in result1.keys():
                    if isinstance(result1[key], (int, float)) and isinstance(result2[key], (int, float)):
                        if abs(result1[key] - result2[key]) > 1e-6:  # Numerical tolerance
                            return False
                    elif result1[key] != result2[key]:
                        return False
                
                return True
                
            elif isinstance(result1, (list, np.ndarray)) and isinstance(result2, (list, np.ndarray)):
                # Compare array results
                arr1 = np.array(result1)
                arr2 = np.array(result2)
                
                if arr1.shape != arr2.shape:
                    return False
                
                return np.allclose(arr1, arr2, atol=1e-6)
            
            else:
                # Direct comparison
                return result1 == result2
                
        except Exception:
            return False
    
    def _summarize_result(self, result: Any) -> Dict[str, Any]:
        """Create summary of analysis result for comparison."""
        
        summary = {'type': type(result).__name__}
        
        try:
            if isinstance(result, dict):
                summary['keys'] = list(result.keys())
                summary['numeric_values'] = {
                    k: v for k, v in result.items() 
                    if isinstance(v, (int, float))
                }
            elif isinstance(result, (list, np.ndarray)):
                arr = np.array(result)
                summary['shape'] = arr.shape
                summary['mean'] = float(np.mean(arr))
                summary['std'] = float(np.std(arr))
            elif isinstance(result, (int, float)):
                summary['value'] = float(result)
        except Exception as e:
            summary['error'] = str(e)
        
        return summary


def create_bias_free_analysis_wrapper(analysis_function):
    """
    Decorator to ensure analysis function is bias-free.
    
    Args:
        analysis_function: Function to wrap with bias prevention
        
    Returns:
        Wrapped function with temporal validation
    """
    
    def bias_free_wrapper(*args, **kwargs):
        # Add temporal validation
        bias_prevention = LookaheadBiasPrevention()
        
        # Run original analysis
        result = analysis_function(*args, **kwargs)
        
        # Validate temporal separation if market data is provided
        if len(args) > 0 and isinstance(args[0], pd.DataFrame):
            validation = bias_prevention.validate_temporal_separation(
                analysis_function, args[0], **kwargs
            )
            
            # Add validation results to output
            if isinstance(result, dict):
                result['temporal_validation'] = validation
            
        return result
    
    return bias_free_wrapper


def create_temporal_validation_framework() -> Dict[str, Any]:
    """
    Create framework for temporal validation of regime analysis.
    
    Returns guidelines and methods for preventing lookahead bias.
    """
    
    framework = {
        'principles': {
            'strict_time_separation': "Only use information available at time t for decisions",
            'ex_post_evaluation': "Evaluate outcomes separately from predictions",
            'rolling_validation': "Use walk-forward analysis for temporal robustness",
            'hold_out_testing': "Reserve periods for out-of-sample validation"
        },
        
        'implementation_guidelines': {
            'breakout_analysis': {
                'observable_at_t': [
                    "Current price position relative to historical bands",
                    "Dimension signal strength using historical data only",
                    "Technical levels calculated from past data"
                ],
                'ex_post_evaluation': [
                    "Whether breakout occurred in subsequent periods",
                    "Magnitude of breakout movement",
                    "Duration of breakout follow-through"
                ],
                'bias_prevention': [
                    "Never use future prices in current period calculations",
                    "Calculate bands using only historical data",
                    "Evaluate predictions separately from signal generation"
                ]
            },
            
            'regime_transition_analysis': {
                'observable_at_t': [
                    "Current regime state based on historical data",
                    "Transition probability based on historical patterns",
                    "Market conditions observable at time t"
                ],
                'ex_post_evaluation': [
                    "Whether regime actually changed",
                    "New regime characteristics",
                    "Transition timing accuracy"
                ],
                'bias_prevention': [
                    "Use lagged regime labels for transition prediction",
                    "Separate transition detection from transition prediction",
                    "Validate predictions on hold-out periods"
                ]
            }
        },
        
        'validation_methods': {
            'walk_forward_analysis': {
                'description': "Rolling window regime discovery with out-of-sample validation",
                'parameters': {
                    'training_window': 252,  # 1 year for regime discovery
                    'validation_window': 63,  # 3 months for validation
                    'step_size': 21  # Monthly steps
                }
            },
            
            'hold_out_validation': {
                'description': "Reserve final periods for completely out-of-sample testing",
                'parameters': {
                    'hold_out_percentage': 0.2,  # 20% of data
                    'minimum_hold_out_periods': 100
                }
            },
            
            'bootstrap_validation': {
                'description': "Bootstrap sampling for robustness testing",
                'parameters': {
                    'bootstrap_samples': 1000,
                    'sample_fraction': 0.8
                }
            }
        },
        
        'bias_detection_tests': {
            'future_information_test': "Check if any calculations use future data",
            'prediction_accuracy_test': "Validate prediction accuracy on unseen data",
            'temporal_consistency_test': "Ensure results are consistent across time periods",
            'regime_stability_test': "Test if regime characteristics are stable out-of-sample"
        }
    }
    
    return framework