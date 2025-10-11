"""
VectorBT Research Utilities

Specialized utilities for integrating VectorBT with research frameworks,
providing enhanced capabilities for profit labeling, economic relevance,
and volatility impact research.
"""

import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import warnings

from src.utils.logger import get_logger
from src.utils.math_validation import validate_finite, safe_divide
from .vectorbt_backtesting_engine import VectorBTBacktestingEngine, VectorBTConfig

logger = get_logger('VectorBTResearchUtils')


class ResearchMode(Enum):
    """Research analysis modes."""
    QUICK = "quick"  # Fast analysis for rapid insights
    STANDARD = "standard"  # Balanced analysis
    COMPREHENSIVE = "comprehensive"  # Detailed analysis with full metrics


@dataclass
class ResearchConfig:
    """Configuration for VectorBT research utilities."""
    # Core settings
    research_mode: ResearchMode = ResearchMode.STANDARD
    initial_capital: float = 100000.0
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    
    # Research-specific settings
    min_data_points: int = 1000
    validation_split: float = 0.2
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    
    # Performance settings
    enable_parallel: bool = True
    n_jobs: int = -1
    chunked_processing: bool = True
    
    # Output settings
    save_intermediate_results: bool = True
    generate_plots: bool = True
    output_dir: str = "vectorbt_research_results"


class VectorBTResearchUtils:
    """
    VectorBT utilities for research frameworks.
    
    This class provides specialized methods for integrating VectorBT with
    research workflows, including profit labeling, economic relevance analysis,
    and volatility impact research.
    """
    
    def __init__(self, config: Optional[ResearchConfig] = None):
        """Initialize VectorBT research utilities."""
        self.config = config or ResearchConfig()
        self.logger = logger.getChild('VectorBTResearchUtils')
        
        # Initialize VectorBT backtesting engine
        vectorbt_config = VectorBTConfig(
            initial_capital=self.config.initial_capital,
            commission_rate=self.config.commission_rate,
            slippage_rate=self.config.slippage_rate,
            enable_parallel=self.config.enable_parallel,
            n_jobs=self.config.n_jobs,
            chunked=self.config.chunked_processing
        )
        self.backtesting_engine = VectorBTBacktestingEngine(vectorbt_config)
        
        self.logger.info("✅ VectorBT Research Utils initialized")
        self.logger.info(f"📊 Research mode: {self.config.research_mode.value}")
    
    def analyze_profit_labeling_effectiveness(self, 
                                            market_data: pd.DataFrame,
                                            profit_labels: pd.Series,
                                            labeling_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze effectiveness of profit labeling using VectorBT portfolio simulation.
        
        Args:
            market_data: OHLCV market data
            profit_labels: Profit labels (1 for profitable, 0 for not)
            labeling_config: Configuration for profit labeling
            
        Returns:
            Dictionary with analysis results
        """
        self.logger.info("🔬 Analyzing profit labeling effectiveness with VectorBT")
        
        try:
            # Validate inputs
            if len(market_data) != len(profit_labels):
                raise ValueError("Market data and profit labels length mismatch")
            
            if len(market_data) < self.config.min_data_points:
                self.logger.warning(f"Insufficient data: {len(market_data)} < {self.config.min_data_points}")
            
            # Create signals from profit labels
            signals = self._create_signals_from_labels(profit_labels, labeling_config)
            
            # Run VectorBT backtest
            results = self.backtesting_engine.run_backtest(
                market_data['close'], 
                signals,
                mode=vbt.settings['array_wrapper']['mode']
            )
            
            # Analyze labeling effectiveness
            effectiveness_metrics = self._calculate_labeling_effectiveness_metrics(
                results, profit_labels, signals
            )
            
            # Bootstrap validation
            if self.config.research_mode in [ResearchMode.STANDARD, ResearchMode.COMPREHENSIVE]:
                bootstrap_results = self._bootstrap_labeling_analysis(
                    market_data, profit_labels, labeling_config
                )
                effectiveness_metrics['bootstrap_validation'] = bootstrap_results
            
            # Economic significance test
            economic_significance = self._test_economic_significance(
                results, effectiveness_metrics
            )
            effectiveness_metrics['economic_significance'] = economic_significance
            
            self.logger.info("✅ Profit labeling effectiveness analysis completed")
            return effectiveness_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Profit labeling analysis failed: {e}")
            raise
    
    def analyze_economic_relevance(self,
                                 market_data: pd.DataFrame,
                                 dimension_features: pd.DataFrame,
                                 dimension_name: str,
                                 pattern_type: str) -> Dict[str, Any]:
        """
        Analyze economic relevance of market dimensions using VectorBT.
        
        Args:
            market_data: OHLCV market data
            dimension_features: Feature matrix for the dimension
            dimension_name: Name of the dimension being analyzed
            pattern_type: Type of price pattern to analyze
            
        Returns:
            Dictionary with economic relevance analysis
        """
        self.logger.info(f"🔬 Analyzing economic relevance of {dimension_name} for {pattern_type}")
        
        try:
            # Create dimension-based signals
            dimension_signals = self._create_dimension_signals(
                dimension_features, market_data, pattern_type
            )
            
            # Run VectorBT backtest with dimension signals
            dimension_results = self.backtesting_engine.run_backtest(
                market_data['close'],
                dimension_signals
            )
            
            # Create baseline (random) signals for comparison
            random_signals = self._create_random_signals(len(dimension_signals))
            baseline_results = self.backtesting_engine.run_backtest(
                market_data['close'],
                random_signals
            )
            
            # Calculate economic relevance metrics
            relevance_metrics = self._calculate_economic_relevance_metrics(
                dimension_results, baseline_results, dimension_name, pattern_type
            )
            
            # Statistical significance testing
            if self.config.research_mode in [ResearchMode.STANDARD, ResearchMode.COMPREHENSIVE]:
                significance_test = self._test_statistical_significance(
                    dimension_results, baseline_results
                )
                relevance_metrics['statistical_significance'] = significance_test
            
            # Factor analysis for comprehensive mode
            if self.config.research_mode == ResearchMode.COMPREHENSIVE:
                factor_analysis = self._analyze_dimension_factors(
                    dimension_features, market_data, dimension_results
                )
                relevance_metrics['factor_analysis'] = factor_analysis
            
            self.logger.info(f"✅ Economic relevance analysis completed for {dimension_name}")
            return relevance_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Economic relevance analysis failed: {e}")
            raise
    
    def analyze_volatility_impact(self,
                                market_data: pd.DataFrame,
                                volatility_measures: Dict[str, pd.Series],
                                impact_type: str) -> Dict[str, Any]:
        """
        Analyze volatility impact on price patterns using VectorBT.
        
        Args:
            market_data: OHLCV market data
            volatility_measures: Dictionary of volatility measures
            impact_type: Type of impact to analyze
            
        Returns:
            Dictionary with volatility impact analysis
        """
        self.logger.info(f"🔬 Analyzing volatility impact: {impact_type}")
        
        try:
            impact_results = {}
            
            # Analyze each volatility measure
            for vol_name, vol_series in volatility_measures.items():
                self.logger.info(f"   → Analyzing {vol_name}")
                
                # Create volatility-based signals
                vol_signals = self._create_volatility_signals(
                    vol_series, market_data, impact_type
                )
                
                # Run VectorBT backtest
                vol_results = self.backtesting_engine.run_backtest(
                    market_data['close'],
                    vol_signals
                )
                
                # Calculate impact metrics
                impact_metrics = self._calculate_volatility_impact_metrics(
                    vol_results, vol_name, impact_type
                )
                
                impact_results[vol_name] = impact_metrics
            
            # Comparative analysis
            if len(volatility_measures) > 1:
                comparative_analysis = self._compare_volatility_measures(impact_results)
                impact_results['comparative_analysis'] = comparative_analysis
            
            # Regime analysis for comprehensive mode
            if self.config.research_mode == ResearchMode.COMPREHENSIVE:
                regime_analysis = self._analyze_volatility_regimes(
                    market_data, volatility_measures, impact_results
                )
                impact_results['regime_analysis'] = regime_analysis
            
            self.logger.info("✅ Volatility impact analysis completed")
            return impact_results
            
        except Exception as e:
            self.logger.error(f"❌ Volatility impact analysis failed: {e}")
            raise
    
    def run_research_validation(self,
                               research_results: Dict[str, Any],
                               market_data: pd.DataFrame,
                               validation_method: str = "out_of_sample") -> Dict[str, Any]:
        """
        Validate research results using VectorBT portfolio simulation.
        
        Args:
            research_results: Results from research analysis
            market_data: Market data for validation
            validation_method: Method for validation (out_of_sample, bootstrap, etc.)
            
        Returns:
            Dictionary with validation results
        """
        self.logger.info(f"🔬 Running research validation: {validation_method}")
        
        try:
            if validation_method == "out_of_sample":
                validation_results = self._out_of_sample_validation(
                    research_results, market_data
                )
            elif validation_method == "bootstrap":
                validation_results = self._bootstrap_validation(
                    research_results, market_data
                )
            elif validation_method == "walk_forward":
                validation_results = self._walk_forward_validation(
                    research_results, market_data
                )
            else:
                raise ValueError(f"Unknown validation method: {validation_method}")
            
            self.logger.info("✅ Research validation completed")
            return validation_results
            
        except Exception as e:
            self.logger.error(f"❌ Research validation failed: {e}")
            raise
    
    def _create_signals_from_labels(self, 
                                   profit_labels: pd.Series, 
                                   labeling_config: Dict[str, Any]) -> pd.Series:
        """Create trading signals from profit labels."""
        # Simple strategy: buy when label indicates profit opportunity
        signals = pd.Series(0, index=profit_labels.index)
        
        # Use labeling configuration to determine signal strength
        threshold = labeling_config.get('threshold', 0.5)
        signal_strength = labeling_config.get('signal_strength', 1.0)
        
        # Create signals based on profit labels
        signals[profit_labels > threshold] = signal_strength
        signals[profit_labels < (1 - threshold)] = -signal_strength
        
        return signals
    
    def _create_dimension_signals(self,
                                 dimension_features: pd.DataFrame,
                                 market_data: pd.DataFrame,
                                 pattern_type: str) -> pd.Series:
        """Create trading signals from dimension features."""
        # Combine dimension features into composite signal
        if len(dimension_features.columns) == 1:
            composite_signal = dimension_features.iloc[:, 0]
        else:
            # Use PCA or weighted average for multiple features
            composite_signal = dimension_features.mean(axis=1)
        
        # Normalize signal
        composite_signal = (composite_signal - composite_signal.mean()) / composite_signal.std()
        
        # Create trading signals based on pattern type
        signals = pd.Series(0, index=composite_signal.index)
        
        if pattern_type == "momentum":
            signals[composite_signal > 1] = 1  # Strong positive momentum
            signals[composite_signal < -1] = -1  # Strong negative momentum
        elif pattern_type == "mean_reversion":
            signals[composite_signal > 2] = -1  # Overbought - sell
            signals[composite_signal < -2] = 1  # Oversold - buy
        else:
            # Default: use threshold-based signals
            signals[composite_signal > 0.5] = 1
            signals[composite_signal < -0.5] = -1
        
        return signals
    
    def _create_volatility_signals(self,
                                  volatility_series: pd.Series,
                                  market_data: pd.DataFrame,
                                  impact_type: str) -> pd.Series:
        """Create trading signals based on volatility measures."""
        # Calculate volatility percentiles
        vol_percentiles = volatility_series.rolling(100).rank(pct=True)
        
        signals = pd.Series(0, index=volatility_series.index)
        
        if impact_type == "trend_persistence":
            # High volatility = trend continuation, Low volatility = trend reversal
            signals[vol_percentiles > 0.8] = 1  # High vol - trend following
            signals[vol_percentiles < 0.2] = -1  # Low vol - mean reversion
        elif impact_type == "breakout_probability":
            # High volatility = higher breakout probability
            signals[vol_percentiles > 0.7] = 1  # High vol - breakout
            signals[vol_percentiles < 0.3] = 0  # Low vol - no signal
        else:
            # Default: volatility-based mean reversion
            signals[vol_percentiles > 0.8] = -1  # High vol - sell
            signals[vol_percentiles < 0.2] = 1  # Low vol - buy
        
        return signals
    
    def _create_random_signals(self, length: int) -> pd.Series:
        """Create random signals for baseline comparison."""
        np.random.seed(42)  # For reproducibility
        random_values = np.random.randn(length)
        signals = pd.Series(0, index=range(length))
        signals[random_values > 0.5] = 1
        signals[random_values < -0.5] = -1
        return signals
    
    def _calculate_labeling_effectiveness_metrics(self,
                                                 results,
                                                 profit_labels: pd.Series,
                                                 signals: pd.Series) -> Dict[str, Any]:
        """Calculate effectiveness metrics for profit labeling."""
        # Basic performance metrics
        total_return = results.performance_metrics.get('total_return', 0)
        sharpe_ratio = results.performance_metrics.get('sharpe_ratio', 0)
        win_rate = results.performance_metrics.get('win_rate', 0)
        
        # Labeling-specific metrics
        signal_accuracy = self._calculate_signal_accuracy(signals, profit_labels)
        label_utilization = self._calculate_label_utilization(signals, profit_labels)
        
        # Economic significance
        economic_value = self._calculate_economic_value(results)
        
        return {
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe_ratio),
            'win_rate': float(win_rate),
            'signal_accuracy': float(signal_accuracy),
            'label_utilization': float(label_utilization),
            'economic_value': float(economic_value),
            'total_trades': results.performance_metrics.get('total_trades', 0)
        }
    
    def _calculate_economic_relevance_metrics(self,
                                            dimension_results,
                                            baseline_results,
                                            dimension_name: str,
                                            pattern_type: str) -> Dict[str, Any]:
        """Calculate economic relevance metrics."""
        # Performance comparison
        dimension_return = dimension_results.performance_metrics.get('total_return', 0)
        baseline_return = baseline_results.performance_metrics.get('total_return', 0)
        
        dimension_sharpe = dimension_results.performance_metrics.get('sharpe_ratio', 0)
        baseline_sharpe = baseline_results.performance_metrics.get('sharpe_ratio', 0)
        
        # Calculate improvement metrics
        return_improvement = dimension_return - baseline_return
        sharpe_improvement = dimension_sharpe - baseline_sharpe
        
        # Information ratio (excess return per unit of risk)
        excess_return = return_improvement
        tracking_error = abs(dimension_sharpe - baseline_sharpe) if baseline_sharpe != 0 else 0
        information_ratio = safe_divide(excess_return, tracking_error, default=0)
        
        # Economic significance
        is_economically_significant = (
            return_improvement > 0.05 and  # 5% improvement
            sharpe_improvement > 0.1 and   # 0.1 Sharpe improvement
            information_ratio > 0.5        # 0.5 information ratio
        )
        
        return {
            'dimension_name': dimension_name,
            'pattern_type': pattern_type,
            'dimension_return': float(dimension_return),
            'baseline_return': float(baseline_return),
            'return_improvement': float(return_improvement),
            'sharpe_improvement': float(sharpe_improvement),
            'information_ratio': float(information_ratio),
            'is_economically_significant': is_economically_significant,
            'economic_relevance_score': float(
                (return_improvement * 0.4 + sharpe_improvement * 0.4 + information_ratio * 0.2)
            )
        }
    
    def _calculate_volatility_impact_metrics(self,
                                           vol_results,
                                           vol_name: str,
                                           impact_type: str) -> Dict[str, Any]:
        """Calculate volatility impact metrics."""
        # Basic performance metrics
        total_return = vol_results.performance_metrics.get('total_return', 0)
        sharpe_ratio = vol_results.performance_metrics.get('sharpe_ratio', 0)
        max_drawdown = vol_results.performance_metrics.get('max_drawdown', 0)
        
        # Volatility-specific metrics
        volatility_utilization = self._calculate_volatility_utilization(vol_results)
        impact_strength = self._calculate_impact_strength(vol_results, impact_type)
        
        return {
            'volatility_measure': vol_name,
            'impact_type': impact_type,
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'volatility_utilization': float(volatility_utilization),
            'impact_strength': float(impact_strength),
            'is_impactful': impact_strength > 0.3  # 30% impact threshold
        }
    
    def _calculate_signal_accuracy(self, signals: pd.Series, labels: pd.Series) -> float:
        """Calculate accuracy of signals against labels."""
        if len(signals) == 0 or len(labels) == 0:
            return 0.0
        
        # Align signals and labels
        aligned_data = pd.concat([signals, labels], axis=1).dropna()
        if len(aligned_data) == 0:
            return 0.0
        
        signal_col, label_col = aligned_data.columns
        
        # Calculate accuracy for non-zero signals
        non_zero_signals = aligned_data[aligned_data[signal_col] != 0]
        if len(non_zero_signals) == 0:
            return 0.0
        
        # Accuracy = percentage of signals that align with labels
        correct_signals = (
            (non_zero_signals[signal_col] > 0) & (non_zero_signals[label_col] > 0.5)
        ) | (
            (non_zero_signals[signal_col] < 0) & (non_zero_signals[label_col] < 0.5)
        )
        
        return float(correct_signals.mean())
    
    def _calculate_label_utilization(self, signals: pd.Series, labels: pd.Series) -> float:
        """Calculate how well signals utilize available labels."""
        if len(signals) == 0 or len(labels) == 0:
            return 0.0
        
        # Count non-zero signals and positive labels
        non_zero_signals = (signals != 0).sum()
        positive_labels = (labels > 0.5).sum()
        
        if positive_labels == 0:
            return 0.0
        
        # Utilization = ratio of signals to available opportunities
        return float(min(non_zero_signals / positive_labels, 1.0))
    
    def _calculate_economic_value(self, results) -> float:
        """Calculate economic value of the strategy."""
        total_return = results.performance_metrics.get('total_return', 0)
        sharpe_ratio = results.performance_metrics.get('sharpe_ratio', 0)
        max_drawdown = abs(results.performance_metrics.get('max_drawdown', 0))
        
        # Economic value = return adjusted for risk
        if max_drawdown > 0:
            risk_adjusted_return = total_return / max_drawdown
        else:
            risk_adjusted_return = total_return
        
        return float(risk_adjusted_return * sharpe_ratio)
    
    def _calculate_volatility_utilization(self, results) -> float:
        """Calculate how well the strategy utilizes volatility information."""
        # This would require more detailed analysis of volatility-based signals
        # For now, return a simple metric based on trade frequency
        total_trades = results.performance_metrics.get('total_trades', 0)
        return float(min(total_trades / 100, 1.0))  # Normalize by expected trade count
    
    def _calculate_impact_strength(self, results, impact_type: str) -> float:
        """Calculate strength of volatility impact."""
        # Base impact on performance metrics
        total_return = results.performance_metrics.get('total_return', 0)
        sharpe_ratio = results.performance_metrics.get('sharpe_ratio', 0)
        
        # Impact strength varies by type
        if impact_type == "trend_persistence":
            return float(abs(total_return) * sharpe_ratio)
        elif impact_type == "breakout_probability":
            win_rate = results.performance_metrics.get('win_rate', 0)
            return float(win_rate * abs(total_return))
        else:
            return float(abs(total_return))
    
    def _bootstrap_labeling_analysis(self,
                                   market_data: pd.DataFrame,
                                   profit_labels: pd.Series,
                                   labeling_config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform bootstrap analysis for labeling effectiveness."""
        bootstrap_results = []
        
        for i in range(self.config.bootstrap_samples):
            # Create bootstrap sample
            sample_indices = np.random.choice(
                len(market_data), 
                size=len(market_data), 
                replace=True
            )
            
            sample_data = market_data.iloc[sample_indices]
            sample_labels = profit_labels.iloc[sample_indices]
            
            # Analyze sample
            sample_signals = self._create_signals_from_labels(sample_labels, labeling_config)
            sample_results = self.backtesting_engine.run_backtest(
                sample_data['close'], 
                sample_signals
            )
            
            bootstrap_results.append({
                'total_return': sample_results.performance_metrics.get('total_return', 0),
                'sharpe_ratio': sample_results.performance_metrics.get('sharpe_ratio', 0),
                'win_rate': sample_results.performance_metrics.get('win_rate', 0)
            })
        
        # Calculate bootstrap statistics
        returns = [r['total_return'] for r in bootstrap_results]
        sharpe_ratios = [r['sharpe_ratio'] for r in bootstrap_results]
        win_rates = [r['win_rate'] for r in bootstrap_results]
        
        return {
            'mean_return': float(np.mean(returns)),
            'std_return': float(np.std(returns)),
            'mean_sharpe': float(np.mean(sharpe_ratios)),
            'std_sharpe': float(np.std(sharpe_ratios)),
            'mean_win_rate': float(np.mean(win_rates)),
            'std_win_rate': float(np.std(win_rates)),
            'confidence_interval_lower': float(np.percentile(returns, 2.5)),
            'confidence_interval_upper': float(np.percentile(returns, 97.5))
        }
    
    def _test_economic_significance(self, results, effectiveness_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Test economic significance of results."""
        total_return = effectiveness_metrics.get('total_return', 0)
        sharpe_ratio = effectiveness_metrics.get('sharpe_ratio', 0)
        economic_value = effectiveness_metrics.get('economic_value', 0)
        
        # Economic significance criteria
        criteria = {
            'positive_return': total_return > 0.05,  # 5% return
            'positive_sharpe': sharpe_ratio > 0.5,   # 0.5 Sharpe ratio
            'positive_economic_value': economic_value > 0.1,  # 0.1 economic value
            'sufficient_trades': effectiveness_metrics.get('total_trades', 0) > 10
        }
        
        is_significant = sum(criteria.values()) >= 3  # At least 3 out of 4 criteria
        
        return {
            'is_economically_significant': is_significant,
            'criteria': criteria,
            'significance_score': sum(criteria.values()) / len(criteria)
        }
    
    def _test_statistical_significance(self, dimension_results, baseline_results) -> Dict[str, Any]:
        """Test statistical significance of dimension results."""
        # This would require more sophisticated statistical testing
        # For now, return basic comparison metrics
        dimension_return = dimension_results.performance_metrics.get('total_return', 0)
        baseline_return = baseline_results.performance_metrics.get('total_return', 0)
        
        improvement = dimension_return - baseline_return
        
        return {
            'return_improvement': float(improvement),
            'is_positive_improvement': improvement > 0,
            'improvement_magnitude': float(abs(improvement))
        }
    
    def _analyze_dimension_factors(self, dimension_features, market_data, results) -> Dict[str, Any]:
        """Analyze factor exposures of dimension features."""
        # This would require factor data and more sophisticated analysis
        return {
            'factor_analysis': 'Requires additional factor data',
            'dimension_complexity': len(dimension_features.columns),
            'feature_correlation': float(dimension_features.corr().mean().mean())
        }
    
    def _compare_volatility_measures(self, impact_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare different volatility measures."""
        measures = [k for k in impact_results.keys() if k != 'comparative_analysis']
        
        if len(measures) < 2:
            return {'comparison': 'Insufficient measures for comparison'}
        
        # Compare impact strength
        impact_strengths = {
            measure: impact_results[measure].get('impact_strength', 0)
            for measure in measures
        }
        
        best_measure = max(impact_strengths, key=impact_strengths.get)
        
        return {
            'best_measure': best_measure,
            'impact_strengths': impact_strengths,
            'relative_performance': {
                measure: strength / impact_strengths[best_measure]
                for measure, strength in impact_strengths.items()
            }
        }
    
    def _analyze_volatility_regimes(self, market_data, volatility_measures, impact_results) -> Dict[str, Any]:
        """Analyze volatility regimes and their impact."""
        # This would require regime detection and analysis
        return {
            'regime_analysis': 'Requires regime detection implementation',
            'volatility_measures_analyzed': len(volatility_measures),
            'regime_complexity': 'High'
        }
    
    def _out_of_sample_validation(self, research_results, market_data) -> Dict[str, Any]:
        """Perform out-of-sample validation."""
        # Split data for validation
        split_idx = int(len(market_data) * (1 - self.config.validation_split))
        train_data = market_data.iloc[:split_idx]
        test_data = market_data.iloc[split_idx:]
        
        # This would require implementing the research methodology on test data
        return {
            'validation_method': 'out_of_sample',
            'train_period': f"{train_data.index[0]} to {train_data.index[-1]}",
            'test_period': f"{test_data.index[0]} to {test_data.index[-1]}",
            'validation_status': 'Requires implementation'
        }
    
    def _bootstrap_validation(self, research_results, market_data) -> Dict[str, Any]:
        """Perform bootstrap validation."""
        return {
            'validation_method': 'bootstrap',
            'bootstrap_samples': self.config.bootstrap_samples,
            'validation_status': 'Requires implementation'
        }
    
    def _walk_forward_validation(self, research_results, market_data) -> Dict[str, Any]:
        """Perform walk-forward validation."""
        return {
            'validation_method': 'walk_forward',
            'validation_status': 'Requires implementation'
        }


# Convenience functions
def analyze_profit_labeling_with_vectorbt(market_data, profit_labels, labeling_config, config=None):
    """Convenience function for profit labeling analysis."""
    utils = VectorBTResearchUtils(config)
    return utils.analyze_profit_labeling_effectiveness(market_data, profit_labels, labeling_config)


def analyze_economic_relevance_with_vectorbt(market_data, dimension_features, dimension_name, pattern_type, config=None):
    """Convenience function for economic relevance analysis."""
    utils = VectorBTResearchUtils(config)
    return utils.analyze_economic_relevance(market_data, dimension_features, dimension_name, pattern_type)


def analyze_volatility_impact_with_vectorbt(market_data, volatility_measures, impact_type, config=None):
    """Convenience function for volatility impact analysis."""
    utils = VectorBTResearchUtils(config)
    return utils.analyze_volatility_impact(market_data, volatility_measures, impact_type)