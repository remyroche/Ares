"""
Deflated Sharpe Ratio Calculator

Implements deflated Sharpe ratio calculation to correct for multiple testing
and prevent overconfidence in feature selection and model validation.

Key Features:
- Multiple testing correction
- Deflation factor calculation
- Statistical significance testing
- Reality check integration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import logging
import time
from scipy import stats
from scipy.stats import norm, t
import warnings

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

logger = logging.getLogger(__name__)


@dataclass
class DeflatedSharpeConfig:
    """Configuration for deflated Sharpe ratio calculation."""
    
    # Multiple testing parameters
    n_features_tested: int = 100
    n_observations: int = 1000
    confidence_level: float = 0.95
    
    # Deflation parameters
    deflation_method: str = 'bailey_lopez'  # 'bailey_lopez', 'harvey_liu', 'custom'
    custom_deflation_factor: Optional[float] = None
    
    # Statistical parameters
    min_observations: int = 30
    max_observations: int = 10000
    skewness_adjustment: bool = True
    kurtosis_adjustment: bool = True
    
    # Performance parameters
    enable_parallel: bool = True
    max_workers: int = 4
    chunk_size: int = 1000
    
    # Validation parameters
    validate_inputs: bool = True
    strict_validation: bool = False


@dataclass
class DeflatedSharpeResult:
    """Result from deflated Sharpe ratio calculation."""
    
    # Deflated Sharpe ratios
    deflated_sharpe_ratios: Dict[str, float]
    original_sharpe_ratios: Dict[str, float]
    deflation_factors: Dict[str, float]
    
    # Statistical significance
    significant_features: List[str]
    p_values: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    
    # Multiple testing correction
    corrected_p_values: Dict[str, float]
    fdr_corrected: Dict[str, float]
    bonferroni_corrected: Dict[str, float]
    
    # Summary statistics
    n_features_tested: int
    n_significant_features: int
    significance_rate: float
    average_deflation_factor: float
    
    # Performance metrics
    calculation_time: float
    memory_usage_mb: float
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class DeflatedSharpeCalculator:
    """
    Calculator for deflated Sharpe ratios with multiple testing correction.
    
    This class implements various methods for calculating deflated Sharpe ratios
    to correct for multiple testing and prevent overconfidence in feature selection.
    """
    
    def __init__(self, config: Optional[DeflatedSharpeConfig] = None):
        """
        Initialize the deflated Sharpe calculator.
        
        Args:
            config: Configuration for deflated Sharpe calculation
        """
        self.config = config or DeflatedSharpeConfig()
        self.logger = logger
        
        tprint_info("📊 Deflated Sharpe Calculator initialized")
        tprint_debug(f"📊 Features tested: {self.config.n_features_tested}")
        tprint_debug(f"📊 Observations: {self.config.n_observations}")
        tprint_debug(f"📊 Deflation method: {self.config.deflation_method}")
    
    def calculate_deflated_sharpe(self, 
                                sharpe_ratios: Dict[str, float],
                                returns: Optional[Dict[str, pd.Series]] = None) -> DeflatedSharpeResult:
        """
        Calculate deflated Sharpe ratios for multiple features.
        
        Args:
            sharpe_ratios: Dictionary of feature names to Sharpe ratios
            returns: Optional dictionary of feature returns for detailed calculation
            
        Returns:
            DeflatedSharpeResult with deflated Sharpe ratios and significance tests
        """
        start_time = time.time()
        
        tprint_info("📊 Calculating deflated Sharpe ratios...")
        tprint_debug(f"📊 Features: {len(sharpe_ratios)}")
        
        try:
            # Validate inputs
            if self.config.validate_inputs:
                self._validate_inputs(sharpe_ratios, returns)
            
            # Calculate deflation factors
            deflation_factors = self._calculate_deflation_factors(sharpe_ratios, returns)
            
            # Calculate deflated Sharpe ratios
            deflated_sharpe_ratios = {}
            for feature, sharpe in sharpe_ratios.items():
                deflation_factor = deflation_factors.get(feature, 1.0)
                deflated_sharpe_ratios[feature] = sharpe * deflation_factor
            
            # Calculate statistical significance
            p_values = self._calculate_p_values(deflated_sharpe_ratios, returns)
            confidence_intervals = self._calculate_confidence_intervals(deflated_sharpe_ratios, returns)
            
            # Apply multiple testing correction
            corrected_p_values = self._apply_multiple_testing_correction(p_values)
            fdr_corrected = self._apply_fdr_correction(p_values)
            bonferroni_corrected = self._apply_bonferroni_correction(p_values)
            
            # Identify significant features
            significant_features = self._identify_significant_features(
                deflated_sharpe_ratios, corrected_p_values
            )
            
            # Calculate summary statistics
            summary_stats = self._calculate_summary_statistics(
                deflated_sharpe_ratios, deflation_factors, significant_features
            )
            
            # Calculate performance metrics
            calculation_time = time.time() - start_time
            memory_usage = self._estimate_memory_usage(sharpe_ratios, returns)
            
            result = DeflatedSharpeResult(
                deflated_sharpe_ratios=deflated_sharpe_ratios,
                original_sharpe_ratios=sharpe_ratios,
                deflation_factors=deflation_factors,
                significant_features=significant_features,
                p_values=p_values,
                confidence_intervals=confidence_intervals,
                corrected_p_values=corrected_p_values,
                fdr_corrected=fdr_corrected,
                bonferroni_corrected=bonferroni_corrected,
                n_features_tested=len(sharpe_ratios),
                n_significant_features=len(significant_features),
                significance_rate=len(significant_features) / len(sharpe_ratios),
                average_deflation_factor=np.mean(list(deflation_factors.values())),
                calculation_time=calculation_time,
                memory_usage_mb=memory_usage,
                metadata={
                    'config': self.config.__dict__,
                    'deflation_method': self.config.deflation_method
                }
            )
            
            tprint_success(f"✅ Deflated Sharpe calculation completed in {calculation_time:.3f}s")
            tprint_info(f"📊 Significant features: {len(significant_features)}/{len(sharpe_ratios)}")
            tprint_info(f"📊 Significance rate: {result.significance_rate:.3f}")
            tprint_info(f"📊 Average deflation factor: {result.average_deflation_factor:.3f}")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Deflated Sharpe calculation failed: {e}")
            return self._create_error_result(start_time, str(e))
    
    def _validate_inputs(self, 
                        sharpe_ratios: Dict[str, float],
                        returns: Optional[Dict[str, pd.Series]]) -> None:
        """Validate input parameters."""
        if not sharpe_ratios:
            raise ValueError("Sharpe ratios dictionary cannot be empty")
        
        if self.config.strict_validation:
            # Check for finite values
            for feature, sharpe in sharpe_ratios.items():
                if not np.isfinite(sharpe):
                    raise ValueError(f"Non-finite Sharpe ratio for feature {feature}: {sharpe}")
            
            # Check returns if provided
            if returns is not None:
                for feature, series in returns.items():
                    if not isinstance(series, pd.Series):
                        raise ValueError(f"Returns for {feature} must be pandas Series")
                    if len(series) < self.config.min_observations:
                        raise ValueError(f"Insufficient observations for {feature}: {len(series)}")
    
    def _calculate_deflation_factors(self, 
                                   sharpe_ratios: Dict[str, float],
                                   returns: Optional[Dict[str, pd.Series]]) -> Dict[str, float]:
        """Calculate deflation factors for each feature."""
        deflation_factors = {}
        
        try:
            if self.config.deflation_method == 'bailey_lopez':
                deflation_factors = self._bailey_lopez_deflation(sharpe_ratios, returns)
            elif self.config.deflation_method == 'harvey_liu':
                deflation_factors = self._harvey_liu_deflation(sharpe_ratios, returns)
            elif self.config.deflation_method == 'custom':
                deflation_factors = self._custom_deflation(sharpe_ratios, returns)
            else:
                raise ValueError(f"Unknown deflation method: {self.config.deflation_method}")
            
        except Exception as e:
            tprint_error(f"❌ Deflation factor calculation failed: {e}")
            # Return default deflation factors
            for feature in sharpe_ratios.keys():
                deflation_factors[feature] = 1.0
        
        return deflation_factors
    
    def _bailey_lopez_deflation(self, 
                              sharpe_ratios: Dict[str, float],
                              returns: Optional[Dict[str, pd.Series]]) -> Dict[str, float]:
        """Calculate deflation factors using Bailey-Lopez method."""
        deflation_factors = {}
        
        try:
            # Bailey-Lopez deflation factor
            n = self.config.n_observations
            m = self.config.n_features_tested
            
            # Base deflation factor
            base_deflation = np.sqrt(np.log(m))
            
            for feature, sharpe in sharpe_ratios.items():
                # Calculate feature-specific deflation
                if returns is not None and feature in returns:
                    series = returns[feature]
                    if len(series) >= self.config.min_observations:
                        # Use actual observations for this feature
                        n_feature = len(series)
                        deflation_factor = np.sqrt(np.log(m) * n / n_feature)
                    else:
                        deflation_factor = base_deflation
                else:
                    deflation_factor = base_deflation
                
                # Apply skewness and kurtosis adjustments if enabled
                if self.config.skewness_adjustment or self.config.kurtosis_adjustment:
                    if returns is not None and feature in returns:
                        series = returns[feature]
                        if len(series) >= self.config.min_observations:
                            skewness = series.skew()
                            kurtosis = series.kurtosis()
                            
                            # Skewness adjustment
                            if self.config.skewness_adjustment:
                                skewness_adj = 1 + (skewness / 6) * (sharpe / np.sqrt(n_feature))
                                deflation_factor *= skewness_adj
                            
                            # Kurtosis adjustment
                            if self.config.kurtosis_adjustment:
                                kurtosis_adj = 1 + ((kurtosis - 3) / 24) * (sharpe**2 / n_feature)
                                deflation_factor *= kurtosis_adj
                
                deflation_factors[feature] = deflation_factor
            
        except Exception as e:
            tprint_error(f"❌ Bailey-Lopez deflation failed: {e}")
            # Return default deflation factors
            for feature in sharpe_ratios.keys():
                deflation_factors[feature] = 1.0
        
        return deflation_factors
    
    def _harvey_liu_deflation(self, 
                            sharpe_ratios: Dict[str, float],
                            returns: Optional[Dict[str, pd.Series]]) -> Dict[str, float]:
        """Calculate deflation factors using Harvey-Liu method."""
        deflation_factors = {}
        
        try:
            # Harvey-Liu deflation factor
            n = self.config.n_observations
            m = self.config.n_features_tested
            
            # Base deflation factor
            base_deflation = np.sqrt(2 * np.log(m))
            
            for feature, sharpe in sharpe_ratios.items():
                # Calculate feature-specific deflation
                if returns is not None and feature in returns:
                    series = returns[feature]
                    if len(series) >= self.config.min_observations:
                        n_feature = len(series)
                        deflation_factor = np.sqrt(2 * np.log(m) * n / n_feature)
                    else:
                        deflation_factor = base_deflation
                else:
                    deflation_factor = base_deflation
                
                deflation_factors[feature] = deflation_factor
            
        except Exception as e:
            tprint_error(f"❌ Harvey-Liu deflation failed: {e}")
            # Return default deflation factors
            for feature in sharpe_ratios.keys():
                deflation_factors[feature] = 1.0
        
        return deflation_factors
    
    def _custom_deflation(self, 
                        sharpe_ratios: Dict[str, float],
                        returns: Optional[Dict[str, pd.Series]]) -> Dict[str, float]:
        """Calculate custom deflation factors."""
        deflation_factors = {}
        
        try:
            if self.config.custom_deflation_factor is not None:
                # Use custom deflation factor for all features
                for feature in sharpe_ratios.keys():
                    deflation_factors[feature] = self.config.custom_deflation_factor
            else:
                # Use default deflation factor
                for feature in sharpe_ratios.keys():
                    deflation_factors[feature] = 1.0
            
        except Exception as e:
            tprint_error(f"❌ Custom deflation failed: {e}")
            # Return default deflation factors
            for feature in sharpe_ratios.keys():
                deflation_factors[feature] = 1.0
        
        return deflation_factors
    
    def _calculate_p_values(self, 
                          deflated_sharpe_ratios: Dict[str, float],
                          returns: Optional[Dict[str, pd.Series]]) -> Dict[str, float]:
        """Calculate p-values for deflated Sharpe ratios."""
        p_values = {}
        
        try:
            for feature, deflated_sharpe in deflated_sharpe_ratios.items():
                # Calculate p-value using t-distribution
                if returns is not None and feature in returns:
                    series = returns[feature]
                    if len(series) >= self.config.min_observations:
                        n = len(series)
                        # Use t-distribution with n-1 degrees of freedom
                        t_stat = deflated_sharpe * np.sqrt(n)
                        p_value = 2 * (1 - t.cdf(abs(t_stat), n - 1))
                    else:
                        # Use normal approximation
                        p_value = 2 * (1 - norm.cdf(abs(deflated_sharpe)))
                else:
                    # Use normal approximation
                    p_value = 2 * (1 - norm.cdf(abs(deflated_sharpe)))
                
                p_values[feature] = p_value
            
        except Exception as e:
            tprint_error(f"❌ P-value calculation failed: {e}")
            # Return default p-values
            for feature in deflated_sharpe_ratios.keys():
                p_values[feature] = 1.0
        
        return p_values
    
    def _calculate_confidence_intervals(self, 
                                      deflated_sharpe_ratios: Dict[str, float],
                                      returns: Optional[Dict[str, pd.Series]]) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for deflated Sharpe ratios."""
        confidence_intervals = {}
        
        try:
            alpha = 1 - self.config.confidence_level
            z_critical = norm.ppf(1 - alpha / 2)
            
            for feature, deflated_sharpe in deflated_sharpe_ratios.items():
                if returns is not None and feature in returns:
                    series = returns[feature]
                    if len(series) >= self.config.min_observations:
                        n = len(series)
                        # Use t-distribution
                        t_critical = t.ppf(1 - alpha / 2, n - 1)
                        se = 1 / np.sqrt(n)
                        margin_error = t_critical * se
                    else:
                        # Use normal approximation
                        se = 1 / np.sqrt(self.config.n_observations)
                        margin_error = z_critical * se
                else:
                    # Use normal approximation
                    se = 1 / np.sqrt(self.config.n_observations)
                    margin_error = z_critical * se
                
                ci_lower = deflated_sharpe - margin_error
                ci_upper = deflated_sharpe + margin_error
                confidence_intervals[feature] = (ci_lower, ci_upper)
            
        except Exception as e:
            tprint_error(f"❌ Confidence interval calculation failed: {e}")
            # Return default confidence intervals
            for feature in deflated_sharpe_ratios.keys():
                confidence_intervals[feature] = (0.0, 0.0)
        
        return confidence_intervals
    
    def _apply_multiple_testing_correction(self, p_values: Dict[str, float]) -> Dict[str, float]:
        """Apply multiple testing correction to p-values."""
        corrected_p_values = {}
        
        try:
            if not p_values:
                return corrected_p_values
            
            # Get p-values and sort
            features = list(p_values.keys())
            p_vals = list(p_values.values())
            
            # Apply Bonferroni correction
            n_tests = len(p_vals)
            bonferroni_corrected = [min(1.0, p * n_tests) for p in p_vals]
            
            for feature, corrected_p in zip(features, bonferroni_corrected):
                corrected_p_values[feature] = corrected_p
            
        except Exception as e:
            tprint_error(f"❌ Multiple testing correction failed: {e}")
            # Return original p-values
            corrected_p_values = p_values.copy()
        
        return corrected_p_values
    
    def _apply_fdr_correction(self, p_values: Dict[str, float]) -> Dict[str, float]:
        """Apply FDR (False Discovery Rate) correction."""
        fdr_corrected = {}
        
        try:
            if not p_values:
                return fdr_corrected
            
            # Get p-values and sort
            features = list(p_values.keys())
            p_vals = list(p_values.values())
            
            # Sort p-values
            sorted_indices = np.argsort(p_vals)
            sorted_p_vals = [p_vals[i] for i in sorted_indices]
            
            # Apply FDR correction (Benjamini-Hochberg)
            n_tests = len(p_vals)
            fdr_corrected_vals = [0.0] * n_tests
            
            for i in range(n_tests):
                rank = i + 1
                fdr_corrected_vals[i] = min(1.0, sorted_p_vals[i] * n_tests / rank)
            
            # Apply monotonicity constraint
            for i in range(n_tests - 2, -1, -1):
                fdr_corrected_vals[i] = min(fdr_corrected_vals[i], fdr_corrected_vals[i + 1])
            
            # Map back to original order
            for i, original_index in enumerate(sorted_indices):
                fdr_corrected[features[original_index]] = fdr_corrected_vals[i]
            
        except Exception as e:
            tprint_error(f"❌ FDR correction failed: {e}")
            # Return original p-values
            fdr_corrected = p_values.copy()
        
        return fdr_corrected
    
    def _apply_bonferroni_correction(self, p_values: Dict[str, float]) -> Dict[str, float]:
        """Apply Bonferroni correction."""
        bonferroni_corrected = {}
        
        try:
            if not p_values:
                return bonferroni_corrected
            
            n_tests = len(p_values)
            
            for feature, p_value in p_values.items():
                corrected_p = min(1.0, p_value * n_tests)
                bonferroni_corrected[feature] = corrected_p
            
        except Exception as e:
            tprint_error(f"❌ Bonferroni correction failed: {e}")
            # Return original p-values
            bonferroni_corrected = p_values.copy()
        
        return bonferroni_corrected
    
    def _identify_significant_features(self, 
                                     deflated_sharpe_ratios: Dict[str, float],
                                     corrected_p_values: Dict[str, float]) -> List[str]:
        """Identify statistically significant features."""
        significant_features = []
        
        try:
            alpha = 1 - self.config.confidence_level
            
            for feature, corrected_p in corrected_p_values.items():
                if corrected_p < alpha:
                    significant_features.append(feature)
            
        except Exception as e:
            tprint_error(f"❌ Significant feature identification failed: {e}")
        
        return significant_features
    
    def _calculate_summary_statistics(self, 
                                    deflated_sharpe_ratios: Dict[str, float],
                                    deflation_factors: Dict[str, float],
                                    significant_features: List[str]) -> Dict[str, Any]:
        """Calculate summary statistics."""
        try:
            return {
                'n_features_tested': len(deflated_sharpe_ratios),
                'n_significant_features': len(significant_features),
                'significance_rate': len(significant_features) / len(deflated_sharpe_ratios),
                'average_deflation_factor': np.mean(list(deflation_factors.values())),
                'median_deflation_factor': np.median(list(deflation_factors.values())),
                'max_deflation_factor': np.max(list(deflation_factors.values())),
                'min_deflation_factor': np.min(list(deflation_factors.values()))
            }
        except Exception:
            return {
                'n_features_tested': 0,
                'n_significant_features': 0,
                'significance_rate': 0.0,
                'average_deflation_factor': 0.0,
                'median_deflation_factor': 0.0,
                'max_deflation_factor': 0.0,
                'min_deflation_factor': 0.0
            }
    
    def _estimate_memory_usage(self, 
                             sharpe_ratios: Dict[str, float],
                             returns: Optional[Dict[str, pd.Series]]) -> float:
        """Estimate memory usage in MB."""
        try:
            memory_usage = 0.0
            
            # Add memory for Sharpe ratios
            memory_usage += len(sharpe_ratios) * 8 / 1024 / 1024  # 8 bytes per float
            
            # Add memory for returns if provided
            if returns is not None:
                for series in returns.values():
                    memory_usage += series.memory_usage(deep=True) / 1024 / 1024
            
            return memory_usage
            
        except Exception:
            return 0.0
    
    def _create_error_result(self, start_time: float, error_message: str) -> DeflatedSharpeResult:
        """Create error result for failed calculation."""
        return DeflatedSharpeResult(
            deflated_sharpe_ratios={},
            original_sharpe_ratios={},
            deflation_factors={},
            significant_features=[],
            p_values={},
            confidence_intervals={},
            corrected_p_values={},
            fdr_corrected={},
            bonferroni_corrected={},
            n_features_tested=0,
            n_significant_features=0,
            significance_rate=0.0,
            average_deflation_factor=0.0,
            calculation_time=time.time() - start_time,
            memory_usage_mb=0.0,
            metadata={'error': True, 'error_message': error_message}
        )


# Convenience functions
def calculate_deflated_sharpe(sharpe_ratios: Dict[str, float],
                            returns: Optional[Dict[str, pd.Series]] = None,
                            config: Optional[DeflatedSharpeConfig] = None) -> DeflatedSharpeResult:
    """
    Convenience function to calculate deflated Sharpe ratios.
    
    Args:
        sharpe_ratios: Dictionary of feature names to Sharpe ratios
        returns: Optional dictionary of feature returns
        config: Configuration for deflated Sharpe calculation
        
    Returns:
        DeflatedSharpeResult with deflated Sharpe ratios
    """
    calculator = DeflatedSharpeCalculator(config)
    return calculator.calculate_deflated_sharpe(sharpe_ratios, returns)


# Export main classes and functions
__all__ = [
    'DeflatedSharpeCalculator',
    'DeflatedSharpeConfig',
    'DeflatedSharpeResult',
    'calculate_deflated_sharpe'
]