"""
Trading Viability Assessor for Regime Detection Systems.

This module provides utilities for assessing the trading viability of
regimes and architectures detected by both NAS and TAS systems.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from src.utils.logger import system_logger


@dataclass
class TradingViabilityResult:
    """Result of trading viability assessment."""
    viability_score: float
    risk_assessment: Dict[str, float]
    market_conditions: Dict[str, float]
    implementation_requirements: Dict[str, Any]
    recommendation: str
    details: Dict[str, Any]


class TradingViabilityAssessor:
    """
    Assessor for evaluating trading viability of regimes and architectures.

    This class provides comprehensive trading viability assessment that can be
    used by both NAS and TAS systems to evaluate the practical trading value
    of detected regimes and architectures.
    """

    def __init__(self, risk_tolerance: float = 0.5):
        """
        Initialize the trading viability assessor.

        Args:
            risk_tolerance: Risk tolerance level (0-1, where 1 is highest tolerance)
        """
        self.logger = system_logger.getChild('TradingViabilityAssessor')
        self.risk_tolerance = risk_tolerance

    def assess_regime_viability(self,
                              regime_data: pd.DataFrame,
                              market_data: pd.DataFrame,
                              regime_column: str = 'regime') -> TradingViabilityResult:
        """
        Assess trading viability of detected regimes.

        Args:
            regime_data: DataFrame with regime assignments
            market_data: Market data with OHLCV
            regime_column: Name of regime column

        Returns:
            TradingViabilityResult with assessment
        """
        try:
            self.logger.info("📊 Assessing trading viability of regimes")

            # Assess market conditions for trading
            market_conditions = self._assess_market_conditions(market_data)

            # Calculate risk metrics for each regime
            risk_assessment = self._calculate_regime_risk_metrics(regime_data, market_data, regime_column)

            # Assess implementation requirements
            implementation_reqs = self._assess_implementation_requirements(regime_data, regime_column)

            # Calculate overall viability score
            viability_score = self._calculate_regime_viability_score(risk_assessment, market_conditions)

            # Generate recommendation
            recommendation = self._generate_regime_viability_recommendation(viability_score, risk_assessment)

            return TradingViabilityResult(
                viability_score=viability_score,
                risk_assessment=risk_assessment,
                market_conditions=market_conditions,
                implementation_requirements=implementation_reqs,
                recommendation=recommendation,
                details={'regime_count': len(regime_data[regime_column].unique())}
            )

        except Exception as e:
            self.logger.error(f"❌ Regime viability assessment failed: {e}")
            return TradingViabilityResult(
                viability_score=0.0,
                risk_assessment={},
                market_conditions={},
                implementation_requirements={},
                recommendation="Assessment failed",
                details={'error': str(e)}
            )

    def assess_architecture_viability(self,
                                    architecture: Any,
                                    performance_data: Dict[str, Any]) -> TradingViabilityResult:
        """
        Assess trading viability of an architecture.

        Args:
            architecture: Architecture to assess
            performance_data: Performance data and metrics

        Returns:
            TradingViabilityResult with assessment
        """
        try:
            self.logger.info("📊 Assessing trading viability of architecture")

            # Assess market conditions for architecture deployment
            market_conditions = self._assess_architecture_market_conditions(architecture, performance_data)

            # Calculate risk metrics for architecture
            risk_assessment = self._calculate_architecture_risk_metrics(architecture, performance_data)

            # Assess implementation requirements
            implementation_reqs = self._assess_architecture_requirements(architecture)

            # Calculate viability score
            viability_score = self._calculate_architecture_viability_score(risk_assessment, market_conditions)

            # Generate recommendation
            recommendation = self._generate_architecture_viability_recommendation(viability_score, risk_assessment)

            return TradingViabilityResult(
                viability_score=viability_score,
                risk_assessment=risk_assessment,
                market_conditions=market_conditions,
                implementation_requirements=implementation_reqs,
                recommendation=recommendation,
                details={'architecture_type': type(architecture).__name__}
            )

        except Exception as e:
            self.logger.error(f"❌ Architecture viability assessment failed: {e}")
            return TradingViabilityResult(
                viability_score=0.0,
                risk_assessment={},
                market_conditions={},
                implementation_requirements={},
                recommendation="Assessment failed",
                details={'error': str(e)}
            )

    def _assess_market_conditions(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """
        Assess current market conditions for trading viability.

        Args:
            market_data: Market data with OHLCV

        Returns:
            Dictionary of market condition metrics
        """
        try:
            conditions = {}

            # Basic market metrics
            returns = market_data['close'].pct_change()
            conditions['volatility'] = returns.std()
            conditions['trend_strength'] = abs(returns.rolling(20).mean().iloc[-1])
            conditions['volume_trend'] = market_data['volume'].pct_change().rolling(10).mean().iloc[-1]

            # Market regime indicators
            conditions['market_efficiency'] = self._calculate_market_efficiency(market_data)
            conditions['liquidity_score'] = self._calculate_liquidity_score(market_data)

            # Risk indicators
            conditions['downside_risk'] = self._calculate_downside_risk(returns)
            conditions['tail_risk'] = self._calculate_tail_risk(returns)

            return conditions

        except Exception as e:
            self.logger.warning(f"⚠️ Market conditions assessment failed: {e}")
            return {}

    def _calculate_regime_risk_metrics(self,
                                     regime_data: pd.DataFrame,
                                     market_data: pd.DataFrame,
                                     regime_column: str) -> Dict[str, float]:
        """
        Calculate risk metrics for each regime.

        Args:
            regime_data: DataFrame with regime assignments
            market_data: Market data
            regime_column: Name of regime column

        Returns:
            Dictionary of risk metrics
        """
        try:
            risk_metrics = {}
            unique_regimes = regime_data[regime_column].unique()

            for regime in unique_regimes:
                regime_mask = regime_data[regime_column] == regime
                regime_returns = market_data.loc[regime_mask, 'close'].pct_change().dropna()

                if len(regime_returns) > 10:  # Minimum sample size
                    risk_metrics.update({
                        f'regime_{regime}_volatility': regime_returns.std(),
                        f'regime_{regime}_downside_risk': self._calculate_downside_risk(regime_returns),
                        f'regime_{regime}_var_95': self._calculate_var(regime_returns, 0.95),
                        f'regime_{regime}_max_loss': regime_returns.min(),
                        f'regime_{regime}_drawdown_risk': self._calculate_drawdown_risk(regime_returns)
                    })

            # Overall risk metrics
            overall_returns = market_data['close'].pct_change().dropna()
            risk_metrics.update({
                'overall_volatility': overall_returns.std(),
                'overall_downside_risk': self._calculate_downside_risk(overall_returns),
                'overall_var_95': self._calculate_var(overall_returns, 0.95),
                'regime_dispersion': len(unique_regimes),
                'regime_stability': 1.0 / len(unique_regimes)  # Higher for fewer regimes
            })

            return risk_metrics

        except Exception as e:
            self.logger.warning(f"⚠️ Regime risk metrics calculation failed: {e}")
            return {}

    def _assess_implementation_requirements(self,
                                          regime_data: pd.DataFrame,
                                          regime_column: str) -> Dict[str, Any]:
        """
        Assess implementation requirements for regime-based trading.

        Args:
            regime_data: DataFrame with regime assignments
            regime_column: Name of regime column

        Returns:
            Dictionary of implementation requirements
        """
        try:
            requirements = {}

            # Regime characteristics
            regime_counts = regime_data[regime_column].value_counts()
            requirements['total_regimes'] = len(regime_counts)
            requirements['min_regime_samples'] = int(regime_counts.min())
            requirements['max_regime_samples'] = int(regime_counts.max())
            requirements['regime_balance'] = regime_counts.min() / regime_counts.max()

            # Implementation complexity
            complexity_score = len(regime_counts) * 0.3 + (1.0 - requirements['regime_balance']) * 0.7
            requirements['implementation_complexity'] = complexity_score

            # Data requirements
            requirements['historical_data_needed'] = max(1000, len(regime_data) * 2)
            requirements['training_samples_required'] = max(500, int(len(regime_data) * 0.7))

            # Technical requirements
            requirements['real_time_processing'] = len(regime_counts) > 5
            requirements['memory_estimation_mb'] = len(regime_data) * len(regime_counts) * 8 / (1024 * 1024)

            return requirements

        except Exception as e:
            self.logger.warning(f"⚠️ Implementation requirements assessment failed: {e}")
            return {}

    def _calculate_regime_viability_score(self,
                                        risk_assessment: Dict[str, float],
                                        market_conditions: Dict[str, float]) -> float:
        """
        Calculate overall viability score for regimes.

        Args:
            risk_assessment: Risk metrics for regimes
            market_conditions: Current market conditions

        Returns:
            Viability score (0-1)
        """
        try:
            score_components = []

            # Risk component (lower risk = higher viability)
            if 'overall_volatility' in risk_assessment:
                volatility_score = 1.0 / (1.0 + risk_assessment['overall_volatility'])
                score_components.append(volatility_score)

            # Market efficiency component
            if 'market_efficiency' in market_conditions:
                efficiency = market_conditions['market_efficiency']
                score_components.append(efficiency)

            # Liquidity component
            if 'liquidity_score' in market_conditions:
                liquidity = market_conditions['liquidity_score']
                score_components.append(liquidity)

            # Regime stability component
            if 'regime_stability' in risk_assessment:
                stability = risk_assessment['regime_stability']
                score_components.append(stability)

            # Calculate weighted average
            if score_components:
                viability_score = np.mean(score_components)
            else:
                viability_score = 0.0

            return max(0.0, min(1.0, viability_score))

        except Exception as e:
            self.logger.warning(f"⚠️ Regime viability score calculation failed: {e}")
            return 0.0

    def _generate_regime_viability_recommendation(self,
                                               viability_score: float,
                                               risk_assessment: Dict[str, float]) -> str:
        """
        Generate recommendation based on regime viability analysis.

        Args:
            viability_score: Overall viability score
            risk_assessment: Risk metrics

        Returns:
            Recommendation string
        """
        try:
            if viability_score > 0.7:
                return "High trading viability - suitable for live trading"
            elif viability_score > 0.5:
                return "Moderate trading viability - suitable for paper trading"
            elif viability_score > 0.3:
                return "Low trading viability - needs risk management improvements"
            else:
                return "Insufficient trading viability - not recommended for trading"

        except Exception as e:
            self.logger.warning(f"⚠️ Recommendation generation failed: {e}")
            return "Unable to generate recommendation"

    def _assess_architecture_market_conditions(self,
                                            architecture: Any,
                                            performance_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Assess market conditions for architecture deployment.

        Args:
            architecture: Architecture to assess
            performance_data: Performance data

        Returns:
            Dictionary of market condition metrics
        """
        try:
            conditions = {}

            # Architecture-specific market conditions
            conditions['architecture_complexity'] = len(getattr(architecture, 'layers', []))
            conditions['performance_stability'] = performance_data.get('stability', 0.5)
            conditions['adaptation_needed'] = performance_data.get('adaptation_required', 0.0)

            # Market environment factors
            conditions['competitive_advantage'] = self._calculate_competitive_advantage(architecture)
            conditions['deployment_feasibility'] = self._calculate_deployment_feasibility(architecture)

            return conditions

        except Exception as e:
            self.logger.warning(f"⚠️ Architecture market conditions assessment failed: {e}")
            return {}

    def _calculate_architecture_risk_metrics(self,
                                           architecture: Any,
                                           performance_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate risk metrics for architecture.

        Args:
            architecture: Architecture to assess
            performance_data: Performance data

        Returns:
            Dictionary of risk metrics
        """
        try:
            risk_metrics = {}

            # Architecture complexity risks
            complexity = len(getattr(architecture, 'layers', []))
            risk_metrics['complexity_risk'] = min(complexity / 10.0, 1.0)

            # Performance risks
            risk_metrics['performance_risk'] = 1.0 - performance_data.get('accuracy', 0.5)
            risk_metrics['stability_risk'] = 1.0 - performance_data.get('stability', 0.5)

            # Operational risks
            risk_metrics['implementation_risk'] = performance_data.get('implementation_complexity', 0.5)
            risk_metrics['maintenance_risk'] = performance_data.get('maintenance_required', 0.3)

            return risk_metrics

        except Exception as e:
            self.logger.warning(f"⚠️ Architecture risk metrics calculation failed: {e}")
            return {}

    def _assess_architecture_requirements(self, architecture: Any) -> Dict[str, Any]:
        """
        Assess implementation requirements for architecture.

        Args:
            architecture: Architecture to assess

        Returns:
            Dictionary of implementation requirements
        """
        try:
            requirements = {}

            # Resource requirements
            parameters = getattr(architecture, 'parameters_count', 1000000)
            requirements['memory_mb'] = parameters * 4 / (1024 * 1024)
            requirements['compute_flops'] = parameters * 1000  # Rough estimate

            # Development requirements
            complexity = len(getattr(architecture, 'layers', []))
            requirements['development_time_days'] = complexity * 2
            requirements['testing_effort'] = complexity * 3

            # Operational requirements
            requirements['monitoring_needed'] = complexity > 5
            requirements['backup_frequency'] = 'daily' if complexity > 3 else 'weekly'

            return requirements

        except Exception as e:
            self.logger.warning(f"⚠️ Architecture requirements assessment failed: {e}")
            return {}

    def _calculate_architecture_viability_score(self,
                                              risk_assessment: Dict[str, float],
                                              market_conditions: Dict[str, float]) -> float:
        """
        Calculate viability score for architecture.

        Args:
            risk_assessment: Risk metrics for architecture
            market_conditions: Market conditions

        Returns:
            Viability score (0-1)
        """
        try:
            score_components = []

            # Risk-adjusted score (lower risk = higher viability)
            for risk_key in ['performance_risk', 'stability_risk', 'implementation_risk']:
                if risk_key in risk_assessment:
                    risk_score = 1.0 - risk_assessment[risk_key]
                    score_components.append(risk_score)

            # Market condition score
            if 'competitive_advantage' in market_conditions:
                score_components.append(market_conditions['competitive_advantage'])

            if 'deployment_feasibility' in market_conditions:
                score_components.append(market_conditions['deployment_feasibility'])

            if score_components:
                viability_score = np.mean(score_components)
            else:
                viability_score = 0.0

            return max(0.0, min(1.0, viability_score))

        except Exception as e:
            self.logger.warning(f"⚠️ Architecture viability score calculation failed: {e}")
            return 0.0

    def _generate_architecture_viability_recommendation(self,
                                                      viability_score: float,
                                                      risk_assessment: Dict[str, float]) -> str:
        """
        Generate recommendation for architecture viability.

        Args:
            viability_score: Viability score
            risk_assessment: Risk metrics

        Returns:
            Recommendation string
        """
        try:
            if viability_score > 0.7:
                return "High viability - ready for production deployment"
            elif viability_score > 0.5:
                return "Moderate viability - suitable for controlled deployment"
            elif viability_score > 0.3:
                return "Low viability - requires significant improvements"
            else:
                return "Insufficient viability - not recommended for deployment"

        except Exception as e:
            self.logger.warning(f"⚠️ Architecture recommendation generation failed: {e}")
            return "Unable to generate recommendation"

    # Helper methods
    def _calculate_market_efficiency(self, market_data: pd.DataFrame) -> float:
        """Calculate market efficiency score."""
        try:
            # Simple efficiency measure based on price randomness
            returns = market_data['close'].pct_change().dropna()
            # Higher autocorrelation = lower efficiency
            autocorrelation = abs(returns.autocorr(lag=1))
            efficiency = 1.0 - autocorrelation
            return max(0.0, min(1.0, efficiency))
        except:
            return 0.5

    def _calculate_liquidity_score(self, market_data: pd.DataFrame) -> float:
        """Calculate liquidity score."""
        try:
            # Liquidity based on volume and price impact
            avg_volume = market_data['volume'].mean()
            price_range = (market_data['high'] - market_data['low']).mean()
            liquidity = min(avg_volume / (price_range * 1000), 1.0)
            return max(0.0, min(1.0, liquidity))
        except:
            return 0.3

    def _calculate_downside_risk(self, returns: pd.Series) -> float:
        """Calculate downside risk (semi-deviation)."""
        try:
            negative_returns = returns[returns < 0]
            if len(negative_returns) == 0:
                return 0.0
            return negative_returns.std()
        except:
            return 0.0

    def _calculate_tail_risk(self, returns: pd.Series) -> float:
        """Calculate tail risk (VaR approximation)."""
        try:
            return self._calculate_var(returns, 0.95)
        except:
            return 0.0

    def _calculate_var(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Value at Risk."""
        try:
            if len(returns) < 10:
                return 0.0
            return abs(np.percentile(returns, (1 - confidence) * 100))
        except:
            return 0.0

    def _calculate_drawdown_risk(self, returns: pd.Series) -> float:
        """Calculate drawdown risk."""
        try:
            # Simplified drawdown risk
            cum_returns = (1 + returns).cumprod()
            peak = cum_returns.expanding().max()
            drawdown = (cum_returns - peak) / peak
            max_dd = abs(drawdown.min())
            return max_dd
        except:
            return 0.0

    def _calculate_competitive_advantage(self, architecture: Any) -> float:
        """Calculate competitive advantage score."""
        try:
            # Based on architecture properties
            layers = getattr(architecture, 'layers', [])
            complexity = len(layers)
            performance = getattr(architecture, 'fitness_score', 0.5)

            advantage = (performance * 0.7) + (min(complexity / 5.0, 1.0) * 0.3)
            return max(0.0, min(1.0, advantage))
        except:
            return 0.5

    def _calculate_deployment_feasibility(self, architecture: Any) -> float:
        """Calculate deployment feasibility score."""
        try:
            # Based on architecture properties
            parameters = getattr(architecture, 'parameters_count', 1000000)
            complexity = len(getattr(architecture, 'layers', []))

            # Lower parameters and complexity = higher feasibility
            param_score = max(0.0, 1.0 - (parameters / 10000000))
            complexity_score = max(0.0, 1.0 - (complexity / 10.0))

            feasibility = (param_score * 0.6) + (complexity_score * 0.4)
            return feasibility
        except:
            return 0.3