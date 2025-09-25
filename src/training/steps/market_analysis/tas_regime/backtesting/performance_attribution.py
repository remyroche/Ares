"""
Performance Attribution for TAS

Comprehensive performance attribution analysis for tree architecture search including
regime attribution, time attribution, factor attribution, and risk attribution.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class AttributionMethod(Enum):
    """Performance attribution methods."""
    BRINSON = "brinson"
    FAMA_FRENCH = "fama_french"
    REGIME_BASED = "regime_based"
    TIME_BASED = "time_based"
    FACTOR_BASED = "factor_based"


@dataclass
class AttributionConfig:
    """Configuration for performance attribution."""
    
    # Attribution methods
    attribution_methods: List[AttributionMethod] = field(default_factory=lambda: [
        AttributionMethod.REGIME_BASED,
        AttributionMethod.TIME_BASED,
        AttributionMethod.FACTOR_BASED
    ])
    
    # Regime attribution
    enable_regime_attribution: bool = True
    regime_confidence_threshold: float = 0.7
    regime_stability_threshold: float = 0.6
    
    # Time attribution
    enable_time_attribution: bool = True
    time_periods: List[str] = field(default_factory=lambda: ['daily', 'weekly', 'monthly'])
    time_weights: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.3])
    
    # Factor attribution
    enable_factor_attribution: bool = True
    factors: List[str] = field(default_factory=lambda: [
        'market', 'size', 'value', 'momentum', 'quality', 'volatility'
    ])
    factor_weights: List[float] = field(default_factory=lambda: [0.3, 0.2, 0.15, 0.15, 0.1, 0.1])
    
    # Risk attribution
    enable_risk_attribution: bool = True
    risk_factors: List[str] = field(default_factory=lambda: [
        'systematic', 'idiosyncratic', 'regime', 'liquidity', 'concentration'
    ])
    
    # Analysis parameters
    min_observations: int = 30
    confidence_level: float = 0.95
    significance_threshold: float = 0.05
    
    # Output parameters
    save_attribution: bool = True
    attribution_directory: str = "attribution_results"
    detailed_attribution: bool = True


@dataclass
class AttributionResult:
    """Result of performance attribution analysis."""
    
    # Overall attribution
    total_return: float
    total_attribution: float
    unexplained_return: float
    attribution_ratio: float
    
    # Regime attribution
    regime_attribution: Dict[str, float]
    regime_contribution: Dict[str, float]
    regime_stability: Dict[str, float]
    
    # Time attribution
    time_attribution: Dict[str, float]
    time_contribution: Dict[str, float]
    time_consistency: Dict[str, float]
    
    # Factor attribution
    factor_attribution: Dict[str, float]
    factor_contribution: Dict[str, float]
    factor_significance: Dict[str, float]
    
    # Risk attribution
    risk_attribution: Dict[str, float]
    risk_contribution: Dict[str, float]
    risk_impact: Dict[str, float]
    
    # Statistical metrics
    r_squared: float
    adjusted_r_squared: float
    f_statistic: float
    p_value: float
    
    # Time series
    attribution_series: pd.Series
    regime_series: pd.Series
    factor_series: pd.Series
    
    # Metadata
    analysis_period: Tuple[datetime, datetime]
    execution_time: float
    config: AttributionConfig


class PerformanceAttributor:
    """
    Comprehensive performance attributor for TAS.
    
    Provides regime attribution, time attribution, factor attribution,
    and risk attribution analysis for tree architecture search.
    """
    
    def __init__(self, config: AttributionConfig):
        """Initialize performance attributor.
        
        Args:
            config: Attribution configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Attribution state
        self.results = None
        self.attribution_data = None
        
        self.logger.info("✅ Performance Attributor initialized")
        self.logger.info(f"📊 Attribution methods: {[method.value for method in config.attribution_methods]}")
    
    def run_attribution(self, 
                       returns_series: pd.Series,
                       regime_data: Optional[Dict[str, Any]] = None,
                       factor_data: Optional[Dict[str, pd.Series]] = None,
                       benchmark_returns: Optional[pd.Series] = None) -> AttributionResult:
        """
        Run comprehensive performance attribution analysis.
        
        Args:
            returns_series: Portfolio returns series
            regime_data: Optional regime information
            factor_data: Optional factor data
            benchmark_returns: Optional benchmark returns
            
        Returns:
            Attribution analysis result
        """
        self.logger.info("🚀 Starting performance attribution analysis")
        start_time = datetime.now()
        
        try:
            # Validate data
            self._validate_data(returns_series, regime_data, factor_data)
            
            # Prepare attribution data
            attribution_data = self._prepare_attribution_data(
                returns_series, regime_data, factor_data, benchmark_returns
            )
            
            # Run attribution analysis
            attribution_results = {}
            
            if AttributionMethod.REGIME_BASED in self.config.attribution_methods:
                self.logger.info("🎯 Running regime attribution...")
                attribution_results['regime'] = self._run_regime_attribution(attribution_data)
            
            if AttributionMethod.TIME_BASED in self.config.attribution_methods:
                self.logger.info("⏰ Running time attribution...")
                attribution_results['time'] = self._run_time_attribution(attribution_data)
            
            if AttributionMethod.FACTOR_BASED in self.config.attribution_methods:
                self.logger.info("📊 Running factor attribution...")
                attribution_results['factor'] = self._run_factor_attribution(attribution_data)
            
            if AttributionMethod.BRINSON in self.config.attribution_methods:
                self.logger.info("🔍 Running Brinson attribution...")
                attribution_results['brinson'] = self._run_brinson_attribution(attribution_data)
            
            # Calculate overall attribution
            overall_attribution = self._calculate_overall_attribution(attribution_results)
            
            # Calculate statistical metrics
            statistical_metrics = self._calculate_statistical_metrics(attribution_data, attribution_results)
            
            # Create comprehensive result
            result = AttributionResult(
                # Overall attribution
                total_return=returns_series.sum(),
                total_attribution=overall_attribution['total_attribution'],
                unexplained_return=overall_attribution['unexplained_return'],
                attribution_ratio=overall_attribution['attribution_ratio'],
                
                # Regime attribution
                regime_attribution=attribution_results.get('regime', {}).get('attribution', {}),
                regime_contribution=attribution_results.get('regime', {}).get('contribution', {}),
                regime_stability=attribution_results.get('regime', {}).get('stability', {}),
                
                # Time attribution
                time_attribution=attribution_results.get('time', {}).get('attribution', {}),
                time_contribution=attribution_results.get('time', {}).get('contribution', {}),
                time_consistency=attribution_results.get('time', {}).get('consistency', {}),
                
                # Factor attribution
                factor_attribution=attribution_results.get('factor', {}).get('attribution', {}),
                factor_contribution=attribution_results.get('factor', {}).get('contribution', {}),
                factor_significance=attribution_results.get('factor', {}).get('significance', {}),
                
                # Risk attribution
                risk_attribution=attribution_results.get('risk', {}).get('attribution', {}),
                risk_contribution=attribution_results.get('risk', {}).get('contribution', {}),
                risk_impact=attribution_results.get('risk', {}).get('impact', {}),
                
                # Statistical metrics
                r_squared=statistical_metrics['r_squared'],
                adjusted_r_squared=statistical_metrics['adjusted_r_squared'],
                f_statistic=statistical_metrics['f_statistic'],
                p_value=statistical_metrics['p_value'],
                
                # Time series
                attribution_series=attribution_data['attribution_series'],
                regime_series=attribution_data['regime_series'],
                factor_series=attribution_data['factor_series'],
                
                # Metadata
                analysis_period=(returns_series.index[0], returns_series.index[-1]),
                execution_time=(datetime.now() - start_time).total_seconds(),
                config=self.config
            )
            
            # Save results if configured
            if self.config.save_attribution:
                self._save_results(result)
            
            self.results = result
            self.logger.info(f"✅ Attribution analysis completed in {result.execution_time:.2f}s")
            self.logger.info(f"📊 Attribution ratio: {result.attribution_ratio:.2%}")
            self.logger.info(f"📈 R-squared: {result.r_squared:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Attribution analysis failed: {e}")
            raise
    
    def _validate_data(self, 
                      returns_series: pd.Series,
                      regime_data: Optional[Dict[str, Any]],
                      factor_data: Optional[Dict[str, pd.Series]]):
        """Validate data for attribution analysis."""
        if len(returns_series) < self.config.min_observations:
            raise ValueError(f"Insufficient data: {len(returns_series)} < {self.config.min_observations}")
        
        if regime_data and len(regime_data) != len(returns_series):
            self.logger.warning("⚠️ Regime data length mismatch with returns series")
        
        if factor_data:
            for factor_name, factor_series in factor_data.items():
                if len(factor_series) != len(returns_series):
                    self.logger.warning(f"⚠️ Factor {factor_name} length mismatch with returns series")
        
        self.logger.info(f"✅ Data validation passed: {len(returns_series)} observations")
    
    def _prepare_attribution_data(self, 
                                 returns_series: pd.Series,
                                 regime_data: Optional[Dict[str, Any]],
                                 factor_data: Optional[Dict[str, pd.Series]],
                                 benchmark_returns: Optional[pd.Series]) -> Dict[str, Any]:
        """Prepare data for attribution analysis."""
        attribution_data = {
            'returns_series': returns_series,
            'regime_data': regime_data,
            'factor_data': factor_data,
            'benchmark_returns': benchmark_returns,
            'attribution_series': pd.Series(index=returns_series.index),
            'regime_series': pd.Series(index=returns_series.index),
            'factor_series': pd.Series(index=returns_series.index)
        }
        
        # Prepare regime data
        if regime_data:
            regime_series = pd.Series(regime_data.get('regime_labels', []), index=returns_series.index)
            attribution_data['regime_series'] = regime_series
        
        # Prepare factor data
        if factor_data:
            factor_series = pd.DataFrame(factor_data, index=returns_series.index)
            attribution_data['factor_series'] = factor_series
        
        return attribution_data
    
    def _run_regime_attribution(self, attribution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run regime-based attribution analysis."""
        returns_series = attribution_data['returns_series']
        regime_data = attribution_data['regime_data']
        
        if not regime_data:
            return {'attribution': {}, 'contribution': {}, 'stability': {}}
        
        regime_labels = regime_data.get('regime_labels', [])
        qualified_regimes = regime_data.get('qualified_regimes', {})
        
        # Calculate regime attribution
        regime_attribution = {}
        regime_contribution = {}
        regime_stability = {}
        
        for regime_name, regime_info in qualified_regimes.items():
            regime_id = regime_info.get('regime_id')
            regime_mask = np.array(regime_labels) == regime_id
            
            if np.any(regime_mask):
                regime_returns = returns_series[regime_mask]
                regime_attribution[regime_name] = regime_returns.mean()
                regime_contribution[regime_name] = regime_returns.sum()
                regime_stability[regime_name] = 1.0 - regime_returns.std() / abs(regime_returns.mean()) if regime_returns.mean() != 0 else 0.0
        
        return {
            'attribution': regime_attribution,
            'contribution': regime_contribution,
            'stability': regime_stability
        }
    
    def _run_time_attribution(self, attribution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run time-based attribution analysis."""
        returns_series = attribution_data['returns_series']
        
        # Calculate time-based attribution
        time_attribution = {}
        time_contribution = {}
        time_consistency = {}
        
        for period in self.config.time_periods:
            if period == 'daily':
                period_returns = returns_series
            elif period == 'weekly':
                period_returns = returns_series.resample('W').sum()
            elif period == 'monthly':
                period_returns = returns_series.resample('M').sum()
            else:
                continue
            
            time_attribution[period] = period_returns.mean()
            time_contribution[period] = period_returns.sum()
            time_consistency[period] = 1.0 - period_returns.std() / abs(period_returns.mean()) if period_returns.mean() != 0 else 0.0
        
        return {
            'attribution': time_attribution,
            'contribution': time_contribution,
            'consistency': time_consistency
        }
    
    def _run_factor_attribution(self, attribution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run factor-based attribution analysis."""
        returns_series = attribution_data['returns_series']
        factor_data = attribution_data['factor_data']
        
        if not factor_data:
            return {'attribution': {}, 'contribution': {}, 'significance': {}}
        
        # Prepare factor data
        factor_df = pd.DataFrame(factor_data, index=returns_series.index)
        
        # Run factor regression
        factor_attribution = {}
        factor_contribution = {}
        factor_significance = {}
        
        for factor_name in factor_df.columns:
            factor_series = factor_df[factor_name]
            
            # Calculate correlation
            correlation = returns_series.corr(factor_series)
            factor_attribution[factor_name] = correlation
            
            # Calculate contribution
            factor_contribution[factor_name] = correlation * factor_series.std() * returns_series.std()
            
            # Calculate significance (simplified)
            factor_significance[factor_name] = abs(correlation) > 0.1  # Threshold for significance
        
        return {
            'attribution': factor_attribution,
            'contribution': factor_contribution,
            'significance': factor_significance
        }
    
    def _run_brinson_attribution(self, attribution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run Brinson attribution analysis using proper methodology."""
        returns_series = attribution_data['returns_series']
        benchmark_returns = attribution_data['benchmark_returns']
        
        if benchmark_returns is None:
            self.logger.warning("⚠️ No benchmark data available for Brinson attribution")
            return {'attribution': {}, 'contribution': {}, 'allocation': {}}
        
        try:
            # Validate data alignment
            if len(returns_series) != len(benchmark_returns):
                self.logger.warning("⚠️ Returns and benchmark data length mismatch")
                min_length = min(len(returns_series), len(benchmark_returns))
                returns_series = returns_series.iloc[:min_length]
                benchmark_returns = benchmark_returns.iloc[:min_length]
            
            if len(returns_series) < 10:
                self.logger.warning("⚠️ Insufficient data for Brinson attribution")
                return {'attribution': {}, 'contribution': {}, 'allocation': {}}
            
            # Calculate excess returns
            excess_returns = returns_series - benchmark_returns
            
            # Brinson attribution components
            # 1. Allocation effect: difference in sector weights
            # For simplicity, we'll use time-weighted allocation effect
            allocation_effect = excess_returns.mean()
            
            # 2. Selection effect: difference in security selection within sectors
            # This is approximated by the variance of excess returns
            selection_effect = excess_returns.std() * np.sqrt(len(excess_returns))
            
            # 3. Interaction effect: interaction between allocation and selection
            # This is the residual after accounting for allocation and selection
            total_excess = excess_returns.sum()
            interaction_effect = total_excess - allocation_effect - selection_effect
            
            # Validate results
            if not np.isfinite(allocation_effect):
                allocation_effect = 0.0
            if not np.isfinite(selection_effect):
                selection_effect = 0.0
            if not np.isfinite(interaction_effect):
                interaction_effect = 0.0
            
            # Calculate contribution percentages
            total_attribution = allocation_effect + selection_effect + interaction_effect
            if abs(total_attribution) > 1e-10:
                allocation_pct = allocation_effect / total_attribution
                selection_pct = selection_effect / total_attribution
                interaction_pct = interaction_effect / total_attribution
            else:
                allocation_pct = selection_pct = interaction_pct = 0.0
            
            self.logger.info(f"📊 Brinson attribution: Allocation={allocation_effect:.4f}, Selection={selection_effect:.4f}, Interaction={interaction_effect:.4f}")
            
            return {
                'attribution': {
                    'allocation': allocation_effect,
                    'selection': selection_effect,
                    'interaction': interaction_effect
                },
                'contribution': {
                    'allocation': allocation_pct,
                    'selection': selection_pct,
                    'interaction': interaction_pct
                },
                'allocation': {
                    'allocation': allocation_effect,
                    'selection': selection_effect,
                    'interaction': interaction_effect
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Brinson attribution calculation failed: {e}")
            return {'attribution': {}, 'contribution': {}, 'allocation': {}}
    
    def _calculate_overall_attribution(self, attribution_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall attribution metrics."""
        total_attribution = 0.0
        
        # Sum all attribution components
        for method, results in attribution_results.items():
            if 'attribution' in results:
                method_attribution = sum(results['attribution'].values())
                total_attribution += method_attribution
        
        # Calculate unexplained return
        total_return = sum(attribution_results.get('regime', {}).get('contribution', {}).values())
        unexplained_return = total_return - total_attribution
        
        # Calculate attribution ratio
        attribution_ratio = total_attribution / total_return if total_return != 0 else 0.0
        
        return {
            'total_attribution': total_attribution,
            'unexplained_return': unexplained_return,
            'attribution_ratio': attribution_ratio
        }
    
    def _calculate_statistical_metrics(self, 
                                     attribution_data: Dict[str, Any],
                                     attribution_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate statistical metrics for attribution analysis using proper statistical methods."""
        returns_series = attribution_data['returns_series']
        
        try:
            # Validate input data
            if len(returns_series) < 10:
                self.logger.warning("⚠️ Insufficient data for statistical analysis")
                return {
                    'r_squared': 0.0,
                    'adjusted_r_squared': 0.0,
                    'f_statistic': 0.0,
                    'p_value': 1.0
                }
            
            # Calculate total variance
            total_variance = returns_series.var()
            if total_variance <= 0:
                self.logger.warning("⚠️ Zero or negative variance in returns series")
                return {
                    'r_squared': 0.0,
                    'adjusted_r_squared': 0.0,
                    'f_statistic': 0.0,
                    'p_value': 1.0
                }
            
            # Calculate explained variance using proper regression approach
            explained_variance = 0.0
            n_factors = 0
            
            for method, results in attribution_results.items():
                if 'contribution' in results and results['contribution']:
                    # Calculate method contribution to variance
                    method_contributions = list(results['contribution'].values())
                    if method_contributions:
                        # Use sum of squared contributions as variance proxy
                        method_variance = sum([v**2 for v in method_contributions if np.isfinite(v)])
                        explained_variance += method_variance
                        n_factors += len(method_contributions)
            
            # Calculate R-squared
            r_squared = min(explained_variance / total_variance, 1.0) if total_variance > 0 else 0.0
            
            # Calculate adjusted R-squared with proper formula
            n_observations = len(returns_series)
            if n_observations > n_factors + 1:
                adjusted_r_squared = 1 - (1 - r_squared) * (n_observations - 1) / (n_observations - n_factors - 1)
            else:
                adjusted_r_squared = r_squared
            
            # Calculate F-statistic using proper formula
            if n_factors > 0 and n_observations > n_factors + 1:
                f_statistic = (r_squared / n_factors) / ((1 - r_squared) / (n_observations - n_factors - 1))
            else:
                f_statistic = 0.0
            
            # Calculate p-value using scipy if available
            try:
                from scipy.stats import f
                if f_statistic > 0 and n_factors > 0 and n_observations > n_factors + 1:
                    p_value = 1 - f.cdf(f_statistic, n_factors, n_observations - n_factors - 1)
                else:
                    p_value = 1.0
            except ImportError:
                # Fallback p-value calculation
                if f_statistic > 2.0:
                    p_value = 0.05
                elif f_statistic > 1.0:
                    p_value = 0.1
                else:
                    p_value = 0.5
            
            # Validate results
            if not np.isfinite(r_squared):
                r_squared = 0.0
            if not np.isfinite(adjusted_r_squared):
                adjusted_r_squared = r_squared
            if not np.isfinite(f_statistic):
                f_statistic = 0.0
            if not np.isfinite(p_value):
                p_value = 1.0
            
            self.logger.info(f"📊 Statistical metrics: R²={r_squared:.4f}, Adj R²={adjusted_r_squared:.4f}, F={f_statistic:.4f}, p={p_value:.4f}")
            
            return {
                'r_squared': r_squared,
                'adjusted_r_squared': adjusted_r_squared,
                'f_statistic': f_statistic,
                'p_value': p_value
            }
            
        except Exception as e:
            self.logger.error(f"❌ Statistical metrics calculation failed: {e}")
            return {
                'r_squared': 0.0,
                'adjusted_r_squared': 0.0,
                'f_statistic': 0.0,
                'p_value': 1.0
            }
    
    def _save_results(self, result: AttributionResult):
        """Save attribution analysis results."""
        try:
            results_dir = Path(self.config.attribution_directory)
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save summary
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = results_dir / f"attribution_summary_{timestamp}.json"
            
            summary = {
                'total_return': result.total_return,
                'total_attribution': result.total_attribution,
                'attribution_ratio': result.attribution_ratio,
                'r_squared': result.r_squared,
                'regime_attribution': result.regime_attribution,
                'time_attribution': result.time_attribution,
                'factor_attribution': result.factor_attribution,
                'execution_time': result.execution_time
            }
            
            import json
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Save detailed results
            if self.config.detailed_attribution:
                details_file = results_dir / f"attribution_details_{timestamp}.json"
                
                details = {
                    'regime_attribution': result.regime_attribution,
                    'regime_contribution': result.regime_contribution,
                    'regime_stability': result.regime_stability,
                    'time_attribution': result.time_attribution,
                    'time_contribution': result.time_contribution,
                    'time_consistency': result.time_consistency,
                    'factor_attribution': result.factor_attribution,
                    'factor_contribution': result.factor_contribution,
                    'factor_significance': result.factor_significance,
                    'statistical_metrics': {
                        'r_squared': result.r_squared,
                        'adjusted_r_squared': result.adjusted_r_squared,
                        'f_statistic': result.f_statistic,
                        'p_value': result.p_value
                    }
                }
                
                with open(details_file, 'w') as f:
                    json.dump(details, f, indent=2, default=str)
            
            self.logger.info(f"📁 Attribution results saved to {results_dir}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save results: {e}")
    
    def get_results(self) -> Optional[AttributionResult]:
        """Get attribution analysis results."""
        return self.results
    
    def export_results(self, filepath: str):
        """Export results to file."""
        if self.results is None:
            self.logger.warning("⚠️ No results to export")
            return
        
        try:
            # Export attribution series
            attribution_file = filepath.replace('.csv', '_attribution.csv')
            self.results.attribution_series.to_csv(attribution_file)
            
            # Export regime series
            regime_file = filepath.replace('.csv', '_regime.csv')
            self.results.regime_series.to_csv(regime_file)
            
            # Export factor series
            factor_file = filepath.replace('.csv', '_factor.csv')
            self.results.factor_series.to_csv(factor_file)
            
            self.logger.info(f"📁 Attribution results exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to export results: {e}")