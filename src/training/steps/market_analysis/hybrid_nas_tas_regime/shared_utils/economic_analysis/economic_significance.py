"""
Economic Significance Analyzer for Regime Detection Systems.

This module provides utilities for assessing the economic significance of
regimes and architectures detected by both NAS and TAS systems.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from src.utils.logger import system_logger


@dataclass
class EconomicSignificanceResult:
    """Result of economic significance analysis."""
    significance_score: float
    confidence_level: float
    economic_metrics: Dict[str, float]
    statistical_tests: Dict[str, float]
    risk_adjusted_metrics: Dict[str, float]
    recommendation: str
    details: Dict[str, Any]


class EconomicSignificanceAnalyzer:
    """
    Analyzer for assessing economic significance of regimes and architectures.

    This class provides comprehensive economic significance analysis that can be
    used by both NAS and TAS systems to evaluate the practical value of detected
    regimes and architectures.
    """

    def __init__(self, significance_threshold: float = 0.01):
        """
        Initialize the economic significance analyzer.

        Args:
            significance_threshold: Minimum significance threshold for economic impact
        """
        self.logger = system_logger.getChild('EconomicSignificanceAnalyzer')
        self.significance_threshold = significance_threshold

    def analyze_regime_significance(self,
                                  regime_data: pd.DataFrame,
                                  market_data: pd.DataFrame,
                                  regime_column: str = 'regime') -> EconomicSignificanceResult:
        """
        Analyze economic significance of detected regimes.

        Args:
            regime_data: DataFrame with regime assignments
            market_data: Market data with OHLCV
            regime_column: Name of regime column

        Returns:
            EconomicSignificanceResult with analysis
        """
        try:
            self.logger.info("📊 Analyzing economic significance of regimes")

            # Calculate regime returns and performance metrics
            regime_metrics = self._calculate_regime_metrics(regime_data, market_data, regime_column)

            # Perform statistical significance tests
            statistical_tests = self._perform_statistical_tests(regime_data, market_data, regime_column)

            # Calculate risk-adjusted metrics
            risk_adjusted_metrics = self._calculate_risk_adjusted_metrics(regime_metrics)

            # Calculate overall significance score
            significance_score = self._calculate_overall_significance(regime_metrics, statistical_tests)

            # Generate recommendation
            recommendation = self._generate_regime_recommendation(significance_score, regime_metrics)

            return EconomicSignificanceResult(
                significance_score=significance_score,
                confidence_level=self._calculate_confidence_level(statistical_tests),
                economic_metrics=regime_metrics,
                statistical_tests=statistical_tests,
                risk_adjusted_metrics=risk_adjusted_metrics,
                recommendation=recommendation,
                details={'regime_count': len(regime_data[regime_column].unique())}
            )

        except Exception as e:
            self.logger.error(f"❌ Regime significance analysis failed: {e}")
            return EconomicSignificanceResult(
                significance_score=0.0,
                confidence_level=0.0,
                economic_metrics={},
                statistical_tests={},
                risk_adjusted_metrics={},
                recommendation="Analysis failed",
                details={'error': str(e)}
            )

    def analyze_architecture_significance(self,
                                        architecture: Any,
                                        performance_data: Dict[str, Any]) -> EconomicSignificanceResult:
        """
        Analyze economic significance of an architecture.

        Args:
            architecture: Architecture to analyze
            performance_data: Performance data and metrics

        Returns:
            EconomicSignificanceResult with analysis
        """
        try:
            self.logger.info("📊 Analyzing economic significance of architecture")

            # Extract architecture metrics
            arch_metrics = self._extract_architecture_metrics(architecture, performance_data)

            # Calculate economic impact
            economic_impact = self._calculate_architecture_economic_impact(arch_metrics)

            # Calculate significance score
            significance_score = self._calculate_architecture_significance(arch_metrics, economic_impact)

            # Generate recommendation
            recommendation = self._generate_architecture_recommendation(significance_score, arch_metrics)

            return EconomicSignificanceResult(
                significance_score=significance_score,
                confidence_level=self._calculate_confidence_level({}),
                economic_metrics=arch_metrics,
                statistical_tests={},
                risk_adjusted_metrics=economic_impact,
                recommendation=recommendation,
                details={'architecture_type': type(architecture).__name__}
            )

        except Exception as e:
            self.logger.error(f"❌ Architecture significance analysis failed: {e}")
            return EconomicSignificanceResult(
                significance_score=0.0,
                confidence_level=0.0,
                economic_metrics={},
                statistical_tests={},
                risk_adjusted_metrics={},
                recommendation="Analysis failed",
                details={'error': str(e)}
            )

    def _calculate_regime_metrics(self,
                                regime_data: pd.DataFrame,
                                market_data: pd.DataFrame,
                                regime_column: str) -> Dict[str, float]:
        """
        Calculate economic metrics for each regime.

        Args:
            regime_data: DataFrame with regime assignments
            market_data: Market data
            regime_column: Name of regime column

        Returns:
            Dictionary of regime metrics
        """
        try:
            metrics = {}
            unique_regimes = regime_data[regime_column].unique()

            for regime in unique_regimes:
                regime_mask = regime_data[regime_column] == regime
                regime_returns = market_data.loc[regime_mask, 'close'].pct_change()

                # Calculate regime-specific metrics
                regime_metrics = {
                    f'regime_{regime}_return_mean': regime_returns.mean(),
                    f'regime_{regime}_return_std': regime_returns.std(),
                    f'regime_{regime}_sharpe': self._calculate_sharpe_ratio(regime_returns),
                    f'regime_{regime}_max_drawdown': self._calculate_max_drawdown(regime_returns),
                    f'regime_{regime}_win_rate': (regime_returns > 0).mean(),
                    f'regime_{regime}_sample_count': regime_mask.sum()
                }
                metrics.update(regime_metrics)

            # Add overall metrics
            overall_returns = market_data['close'].pct_change()
            metrics.update({
                'overall_sharpe': self._calculate_sharpe_ratio(overall_returns),
                'overall_max_drawdown': self._calculate_max_drawdown(overall_returns),
                'overall_volatility': overall_returns.std(),
                'total_regimes': len(unique_regimes),
                'total_samples': len(regime_data)
            })

            return metrics

        except Exception as e:
            self.logger.warning(f"⚠️ Regime metrics calculation failed: {e}")
            return {}

    def _perform_statistical_tests(self,
                                 regime_data: pd.DataFrame,
                                 market_data: pd.DataFrame,
                                 regime_column: str) -> Dict[str, float]:
        """
        Perform statistical significance tests on regimes.

        Args:
            regime_data: DataFrame with regime assignments
            market_data: Market data
            regime_column: Name of regime column

        Returns:
            Dictionary of statistical test results
        """
        try:
            # Simple statistical tests for regime significance
            returns = market_data['close'].pct_change().dropna()
            regime_returns = []

            for regime in regime_data[regime_column].unique():
                regime_mask = regime_data[regime_column] == regime
                regime_ret = market_data.loc[regime_mask, 'close'].pct_change().dropna()
                if len(regime_ret) > 10:  # Minimum sample size
                    regime_returns.append(regime_ret)

            # Perform basic statistical tests
            tests = {}

            if len(regime_returns) >= 2:
                # Test for significant differences between regimes
                from scipy.stats import f_oneway, kruskal

                try:
                    # ANOVA test for equal means
                    if len(regime_returns) >= 2:
                        f_stat, p_value = f_oneway(*regime_returns)
                        tests['anova_f_stat'] = f_stat
                        tests['anova_p_value'] = p_value
                except:
                    pass

                try:
                    # Kruskal-Wallis test (non-parametric)
                    h_stat, p_value = kruskal(*regime_returns)
                    tests['kruskal_h_stat'] = h_stat
                    tests['kruskal_p_value'] = p_value
                except:
                    pass

            # Test for stationarity and other properties
            tests.update({
                'regime_count': len(regime_data[regime_column].unique()),
                'min_regime_samples': regime_data[regime_column].value_counts().min(),
                'max_regime_samples': regime_data[regime_column].value_counts().max(),
                'regime_balance_ratio': regime_data[regime_column].value_counts().min() / regime_data[regime_column].value_counts().max()
            })

            return tests

        except Exception as e:
            self.logger.warning(f"⚠️ Statistical tests failed: {e}")
            return {}

    def _calculate_risk_adjusted_metrics(self, regime_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate risk-adjusted metrics for regimes.

        Args:
            regime_metrics: Raw regime metrics

        Returns:
            Dictionary of risk-adjusted metrics
        """
        try:
            risk_metrics = {}

            # Calculate risk-adjusted returns for each regime
            for key in regime_metrics:
                if 'return_mean' in key and 'return_std' in key.replace('mean', 'std'):
                    regime_id = key.replace('_return_mean', '')
                    mean_key = f'{regime_id}_return_mean'
                    std_key = f'{regime_id}_return_std'

                    if mean_key in regime_metrics and std_key in regime_metrics:
                        mean_return = regime_metrics[mean_key]
                        std_return = regime_metrics[std_key]

                        # Risk-adjusted return (Sharpe-like ratio)
                        if std_return > 0:
                            risk_adjusted = mean_return / std_return
                            risk_metrics[f'{regime_id}_risk_adjusted'] = risk_adjusted

            return risk_metrics

        except Exception as e:
            self.logger.warning(f"⚠️ Risk-adjusted metrics calculation failed: {e}")
            return {}

    def _calculate_overall_significance(self,
                                      regime_metrics: Dict[str, float],
                                      statistical_tests: Dict[str, float]) -> float:
        """
        Calculate overall economic significance score.

        Args:
            regime_metrics: Regime economic metrics
            statistical_tests: Statistical test results

        Returns:
            Overall significance score (0-1)
        """
        try:
            score_components = []

            # Component 1: Average regime Sharpe ratio
            sharpe_ratios = [v for k, v in regime_metrics.items() if 'sharpe' in k.lower()]
            if sharpe_ratios:
                avg_sharpe = np.mean([s for s in sharpe_ratios if s > 0])
                score_components.append(min(avg_sharpe, 2.0) / 2.0)  # Normalize to 0-1

            # Component 2: Statistical significance
            if 'anova_p_value' in statistical_tests:
                p_value = statistical_tests['anova_p_value']
                significance_score = 1.0 - min(p_value, 1.0)  # Lower p-value = higher significance
                score_components.append(significance_score)

            # Component 3: Regime balance
            if 'regime_balance_ratio' in statistical_tests:
                balance = statistical_tests['regime_balance_ratio']
                balance_score = min(balance * 2, 1.0)  # Prefer balanced regimes
                score_components.append(balance_score)

            # Component 4: Sample adequacy
            if 'min_regime_samples' in statistical_tests:
                min_samples = statistical_tests['min_regime_samples']
                sample_score = min(min_samples / 100.0, 1.0)  # Adequate samples
                score_components.append(sample_score)

            # Calculate weighted average
            if score_components:
                significance_score = np.mean(score_components)
            else:
                significance_score = 0.0

            return max(0.0, min(1.0, significance_score))

        except Exception as e:
            self.logger.warning(f"⚠️ Overall significance calculation failed: {e}")
            return 0.0

    def _calculate_confidence_level(self, statistical_tests: Dict[str, float]) -> float:
        """
        Calculate confidence level based on statistical tests.

        Args:
            statistical_tests: Statistical test results

        Returns:
            Confidence level (0-1)
        """
        try:
            confidence = 0.5  # Base confidence

            # Adjust based on statistical significance
            if 'anova_p_value' in statistical_tests:
                p_value = statistical_tests['anova_p_value']
                if p_value < 0.05:
                    confidence += 0.3
                elif p_value < 0.1:
                    confidence += 0.15

            # Adjust based on sample size
            if 'min_regime_samples' in statistical_tests:
                min_samples = statistical_tests['min_regime_samples']
                if min_samples > 100:
                    confidence += 0.2
                elif min_samples > 50:
                    confidence += 0.1

            return max(0.0, min(1.0, confidence))

        except Exception as e:
            self.logger.warning(f"⚠️ Confidence level calculation failed: {e}")
            return 0.5

    def _generate_regime_recommendation(self,
                                     significance_score: float,
                                     regime_metrics: Dict[str, float]) -> str:
        """
        Generate recommendation based on regime analysis.

        Args:
            significance_score: Overall significance score
            regime_metrics: Regime metrics

        Returns:
            Recommendation string
        """
        try:
            if significance_score > 0.7:
                return "Strong economic significance - recommended for trading"
            elif significance_score > 0.5:
                return "Moderate economic significance - proceed with caution"
            elif significance_score > 0.3:
                return "Weak economic significance - further analysis needed"
            else:
                return "Insufficient economic significance - not recommended"

        except Exception as e:
            self.logger.warning(f"⚠️ Recommendation generation failed: {e}")
            return "Unable to generate recommendation"

    def _extract_architecture_metrics(self, architecture: Any, performance_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract metrics from architecture for economic analysis.

        Args:
            architecture: Architecture object
            performance_data: Performance data

        Returns:
            Dictionary of architecture metrics
        """
        try:
            metrics = {}

            # Extract basic architecture properties
            if hasattr(architecture, 'parameters_count'):
                metrics['parameters'] = architecture.parameters_count
            if hasattr(architecture, 'fitness_score'):
                metrics['fitness_score'] = architecture.fitness_score
            if hasattr(architecture, 'regime_accuracy'):
                metrics['regime_accuracy'] = architecture.regime_accuracy

            # Extract performance metrics
            metrics.update(performance_data)

            return metrics

        except Exception as e:
            self.logger.warning(f"⚠️ Architecture metrics extraction failed: {e}")
            return performance_data

    def _calculate_architecture_economic_impact(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate economic impact of architecture.

        Args:
            metrics: Architecture metrics

        Returns:
            Dictionary of economic impact metrics
        """
        try:
            impact = {}

            # Calculate efficiency metrics
            if 'parameters' in metrics and 'regime_accuracy' in metrics:
                param_efficiency = metrics['regime_accuracy'] / max(metrics['parameters'], 1)
                impact['parameter_efficiency'] = param_efficiency

            # Calculate performance impact
            if 'fitness_score' in metrics:
                impact['performance_impact'] = metrics['fitness_score']

            return impact

        except Exception as e:
            self.logger.warning(f"⚠️ Economic impact calculation failed: {e}")
            return {}

    def _calculate_architecture_significance(self,
                                           metrics: Dict[str, float],
                                           economic_impact: Dict[str, float]) -> float:
        """
        Calculate significance score for architecture.

        Args:
            metrics: Architecture metrics
            economic_impact: Economic impact metrics

        Returns:
            Significance score (0-1)
        """
        try:
            score_components = []

            # Performance component
            if 'regime_accuracy' in metrics:
                score_components.append(metrics['regime_accuracy'])

            # Efficiency component
            if 'parameter_efficiency' in economic_impact:
                score_components.append(economic_impact['parameter_efficiency'])

            # Overall fitness component
            if 'fitness_score' in metrics:
                score_components.append(metrics['fitness_score'])

            if score_components:
                return np.mean(score_components)
            else:
                return 0.0

        except Exception as e:
            self.logger.warning(f"⚠️ Architecture significance calculation failed: {e}")
            return 0.0

    def _generate_architecture_recommendation(self,
                                           significance_score: float,
                                           metrics: Dict[str, float]) -> str:
        """
        Generate recommendation for architecture.

        Args:
            significance_score: Significance score
            metrics: Architecture metrics

        Returns:
            Recommendation string
        """
        try:
            if significance_score > 0.7:
                return "High significance architecture - recommended for deployment"
            elif significance_score > 0.5:
                return "Moderate significance - suitable for further testing"
            elif significance_score > 0.3:
                return "Low significance - needs improvement"
            else:
                return "Insufficient significance - not recommended"

        except Exception as e:
            self.logger.warning(f"⚠️ Architecture recommendation generation failed: {e}")
            return "Unable to generate recommendation"

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """
        Calculate Sharpe ratio for returns series.

        Args:
            returns: Returns series

        Returns:
            Sharpe ratio
        """
        try:
            if len(returns) < 2:
                return 0.0

            mean_return = returns.mean()
            std_return = returns.std()

            if std_return == 0:
                return 0.0

            # Annualized Sharpe ratio (assuming daily returns)
            sharpe = mean_return / std_return * np.sqrt(252)
            return sharpe

        except Exception as e:
            self.logger.warning(f"⚠️ Sharpe ratio calculation failed: {e}")
            return 0.0

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        Calculate maximum drawdown for returns series.

        Args:
            returns: Returns series

        Returns:
            Maximum drawdown (as positive value)
        """
        try:
            if len(returns) < 2:
                return 0.0

            # Calculate cumulative returns
            cum_returns = (1 + returns).cumprod()

            # Calculate running maximum
            running_max = cum_returns.expanding().max()

            # Calculate drawdown
            drawdown = (cum_returns - running_max) / running_max

            # Return maximum drawdown (positive value)
            max_dd = abs(drawdown.min())
            return max_dd

        except Exception as e:
            self.logger.warning(f"⚠️ Max drawdown calculation failed: {e}")
            return 0.0