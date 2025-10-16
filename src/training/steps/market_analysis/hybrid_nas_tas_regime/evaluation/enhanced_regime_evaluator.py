"""
Enhanced Regime Evaluation for NAS-TAS System

This module provides comprehensive regime evaluation metrics including:
- Return and volatility per regime
- Sharpe and Sortino ratios
- Maximum drawdown
- Hit rate and pay-off ratio
- Risk-adjusted performance measures
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

except ImportError as e:
    print(f"⚠️ WARNING: tprint utilities not available: {e}")
    # Fallback functions
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_progress(*args, **kwargs): print("PROGRESS:", *args, **kwargs)
    def tprint_performance(*args, **kwargs): print("PERFORMANCE:", *args, **kwargs)
    def tprint_timer(*args, **kwargs): print("TIMER:", *args, **kwargs)

logger = logging.getLogger(__name__)

@dataclass
class RegimeMetrics:
    """Comprehensive regime evaluation metrics."""
    regime_id: int
    size: int
    mean_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    hit_rate: float
    payoff_ratio: float
    calmar_ratio: float
    information_ratio: float
    var_95: float
    cvar_95: float
    skewness: float
    kurtosis: float
    economic_significance: float
    trading_viability: float
    stability_score: float
    risk_score: float
    performance_score: float

@dataclass
class RegimeEvaluationResult:
    """Result from regime evaluation."""
    regime_metrics: List[RegimeMetrics]
    regime_rankings: Dict[str, List[int]]
    overall_quality_score: float
    regime_transitions: List[Dict[str, Any]]
    risk_adjusted_rankings: Dict[str, List[int]]
    economic_rankings: Dict[str, List[int]]
    trading_rankings: Dict[str, List[int]]
    metadata: Dict[str, Any]

class EnhancedRegimeEvaluator:
    """
    Enhanced regime evaluator with comprehensive metrics for NAS-TAS system.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize enhanced regime evaluator."""
        tprint_info("🚀 Initializing Enhanced Regime Evaluator")
        tprint_debug(f"Configuration: {config}")

        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Evaluation parameters
        tprint_debug("⚙️ Setting evaluation parameters...")
        self.risk_free_rate = config.get('risk_free_rate', 0.02)  # 2% annual
        self.confidence_level = config.get('confidence_level', 0.95)
        self.min_regime_size = config.get('min_regime_size', 10)
        self.lookback_periods = config.get('lookback_periods', [1, 5, 10, 20])
        self.volatility_window = config.get('volatility_window', 20)
        tprint_success("✅ Evaluation parameters configured")

        # Performance thresholds
        tprint_debug("📊 Setting performance thresholds...")
        self.min_sharpe_threshold = config.get('min_sharpe_threshold', 0.5)
        self.min_sortino_threshold = config.get('min_sortino_threshold', 0.3)
        self.max_drawdown_threshold = config.get('max_drawdown_threshold', 0.2)
        self.min_hit_rate_threshold = config.get('min_hit_rate_threshold', 0.4)
        self.min_payoff_ratio_threshold = config.get('min_payoff_ratio_threshold', 1.0)
        tprint_success("✅ Performance thresholds configured")

        tprint_success("✅ Enhanced Regime Evaluator initialized")
        self.logger.info("✅ Enhanced Regime Evaluator initialized")

    def evaluate_regimes(self,
                        market_data: pd.DataFrame,
                        regime_labels: np.ndarray,
                        returns: Optional[np.ndarray] = None) -> RegimeEvaluationResult:
        """
        Evaluate regimes with comprehensive metrics.

        Args:
            market_data: Market data DataFrame
            regime_labels: Regime labels for each observation
            returns: Optional pre-calculated returns

        Returns:
            RegimeEvaluationResult with comprehensive regime metrics
        """
        try:
            tprint("🔍 [REGIME_EVALUATION] Starting enhanced regime evaluation", color="blue", bold=True)
            tprint_debug(f"📊 [REGIME_EVALUATION] Market data shape: {market_data.shape}")
            tprint_debug(f"📊 [REGIME_EVALUATION] Regime labels shape: {regime_labels.shape}")
            tprint_debug(f"📊 [REGIME_EVALUATION] Unique regimes: {len(set(regime_labels))}")
            self.logger.info("🔍 Starting enhanced regime evaluation...")

            # Calculate returns if not provided
            if returns is None:
                tprint("📈 [REGIME_EVALUATION] Calculating returns", color="cyan")
                returns = self._calculate_returns(market_data)
                tprint_success(f"✅ [REGIME_EVALUATION] Returns calculated: {len(returns)} observations")
            else:
                tprint_success(f"✅ [REGIME_EVALUATION] Using provided returns: {len(returns)} observations")

            # Calculate regime metrics
            tprint("📊 [REGIME_EVALUATION] Calculating regime metrics", color="cyan")
            regime_metrics = self._calculate_regime_metrics(market_data, regime_labels, returns)
            tprint_success(f"✅ [REGIME_EVALUATION] Regime metrics calculated for {len(regime_metrics)} regimes")

            # Calculate regime rankings
            tprint("🏆 [REGIME_EVALUATION] Calculating regime rankings", color="cyan")
            regime_rankings = self._calculate_regime_rankings(regime_metrics)
            tprint_success("✅ [REGIME_EVALUATION] Regime rankings calculated")

            # Calculate overall quality score
            tprint("📈 [REGIME_EVALUATION] Calculating overall quality score", color="cyan")
            overall_quality_score = self._calculate_overall_quality_score(regime_metrics)
            tprint_success(f"✅ [REGIME_EVALUATION] Overall quality score: {overall_quality_score:.3f}")

            # Analyze regime transitions
            tprint("🔄 [REGIME_EVALUATION] Analyzing regime transitions", color="cyan")
            regime_transitions = self._analyze_regime_transitions(regime_labels)
            tprint_success(f"✅ [REGIME_EVALUATION] Regime transitions analyzed: {len(regime_transitions)} transitions")

            # Calculate specialized rankings
            tprint("📊 [REGIME_EVALUATION] Calculating specialized rankings", color="cyan")
            risk_adjusted_rankings = self._calculate_risk_adjusted_rankings(regime_metrics)
            economic_rankings = self._calculate_economic_rankings(regime_metrics)
            trading_rankings = self._calculate_trading_rankings(regime_metrics)
            tprint_success("✅ [REGIME_EVALUATION] Specialized rankings calculated")

            tprint_success(f"🎉 [REGIME_EVALUATION] Enhanced regime evaluation completed successfully")
            tprint_performance(f"⚡ [REGIME_EVALUATION] Final result: {len(regime_metrics)} regimes evaluated")

            return RegimeEvaluationResult(
                regime_metrics=regime_metrics,
                regime_rankings=regime_rankings,
                overall_quality_score=overall_quality_score,
                regime_transitions=regime_transitions,
                risk_adjusted_rankings=risk_adjusted_rankings,
                economic_rankings=economic_rankings,
                trading_rankings=trading_rankings,
                metadata={
                    'n_regimes': len(regime_metrics),
                    'n_observations': len(regime_labels),
                    'evaluation_timestamp': datetime.now().isoformat(),
                    'config': self.config
                }
            )

        except Exception as e:
            tprint_error(f"❌ [REGIME_EVALUATION] Enhanced regime evaluation failed: {e}")
            tprint_debug(f"🔍 [REGIME_EVALUATION] Error details: {str(e)}")
            self.logger.error(f"Enhanced regime evaluation failed: {e}")
            raise

    def _calculate_returns(self, market_data: pd.DataFrame) -> np.ndarray:
        """Calculate returns from market data."""
        try:
            if 'close' in market_data.columns:
                prices = market_data['close'].values
                returns = np.diff(prices, prepend=prices[0]) / prices
                return returns[1:]  # Remove first element (always 0)
            else:
                # Fallback to first numeric column
                numeric_cols = market_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    prices = market_data[numeric_cols[0]].values
                    returns = np.diff(prices, prepend=prices[0]) / prices
                    return returns[1:]
                else:
                    raise ValueError("No suitable price column found for return calculation")
        except Exception as e:
            self.logger.warning(f"Return calculation failed: {e}")
            return np.zeros(len(market_data))

    def _calculate_regime_metrics(self,
                                 market_data: pd.DataFrame,
                                 regime_labels: np.ndarray,
                                 returns: np.ndarray) -> List[RegimeMetrics]:
        """Calculate comprehensive metrics for each regime."""
        try:
            regime_metrics = []
            unique_regimes = sorted(set(regime_labels))

            for regime_id in unique_regimes:
                regime_mask = regime_labels == regime_id
                regime_returns = returns[regime_mask]
                regime_data = market_data[regime_mask]

                if len(regime_returns) < self.min_regime_size:
                    tprint_warning(f"⚠️ [REGIME_EVALUATION] Regime {regime_id} has insufficient data: {len(regime_returns)} < {self.min_regime_size}")
                    continue

                # Basic statistics
                mean_return = np.mean(regime_returns)
                volatility = np.std(regime_returns)

                # Risk-adjusted metrics
                sharpe_ratio = self._calculate_sharpe_ratio(regime_returns)
                sortino_ratio = self._calculate_sortino_ratio(regime_returns)
                max_drawdown = self._calculate_max_drawdown(regime_returns)
                calmar_ratio = self._calculate_calmar_ratio(regime_returns, max_drawdown)

                # Trading metrics
                hit_rate, payoff_ratio = self._calculate_trading_metrics(regime_returns)

                # Risk metrics
                var_95, cvar_95 = self._calculate_risk_metrics(regime_returns)
                information_ratio = self._calculate_information_ratio(regime_returns)

                # Distribution metrics
                skewness = stats.skew(regime_returns)
                kurtosis = stats.kurtosis(regime_returns)

                # Economic and trading viability
                economic_significance = self._calculate_economic_significance(regime_data, regime_returns)
                trading_viability = self._calculate_trading_viability(hit_rate, payoff_ratio, sharpe_ratio)
                stability_score = self._calculate_stability_score(regime_returns)
                risk_score = self._calculate_risk_score(volatility, max_drawdown, var_95)
                performance_score = self._calculate_performance_score(
                    sharpe_ratio, sortino_ratio, hit_rate, payoff_ratio
                )

                regime_metric = RegimeMetrics(
                    regime_id=regime_id,
                    size=len(regime_returns),
                    mean_return=mean_return,
                    volatility=volatility,
                    sharpe_ratio=sharpe_ratio,
                    sortino_ratio=sortino_ratio,
                    max_drawdown=max_drawdown,
                    hit_rate=hit_rate,
                    payoff_ratio=payoff_ratio,
                    calmar_ratio=calmar_ratio,
                    information_ratio=information_ratio,
                    var_95=var_95,
                    cvar_95=cvar_95,
                    skewness=skewness,
                    kurtosis=kurtosis,
                    economic_significance=economic_significance,
                    trading_viability=trading_viability,
                    stability_score=stability_score,
                    risk_score=risk_score,
                    performance_score=performance_score
                )

                regime_metrics.append(regime_metric)

            return regime_metrics

        except Exception as e:
            self.logger.error(f"Regime metrics calculation failed: {e}")
            return []

    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        try:
            if len(returns) == 0 or np.std(returns) == 0:
                return 0.0

            excess_returns = returns - (self.risk_free_rate / 252)  # Daily risk-free rate
            return np.mean(excess_returns) / np.std(returns) * np.sqrt(252)  # Annualized
        except:
            return 0.0

    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        try:
            if len(returns) == 0:
                return 0.0

            excess_returns = returns - (self.risk_free_rate / 252)
            downside_returns = excess_returns[excess_returns < 0]

            if len(downside_returns) == 0 or np.std(downside_returns) == 0:
                return 0.0

            return np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
        except:
            return 0.0

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        try:
            if len(returns) == 0:
                return 0.0

            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            return abs(np.min(drawdown))
        except:
            return 0.0

    def _calculate_calmar_ratio(self, returns: np.ndarray, max_drawdown: float) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)."""
        try:
            if max_drawdown == 0:
                return 0.0

            annual_return = np.mean(returns) * 252
            return annual_return / max_drawdown
        except:
            return 0.0

    def _calculate_trading_metrics(self, returns: np.ndarray) -> Tuple[float, float]:
        """Calculate hit rate and payoff ratio."""
        try:
            if len(returns) == 0:
                return 0.0, 0.0

            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]

            hit_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0.0

            avg_gain = np.mean(positive_returns) if len(positive_returns) > 0 else 0.0
            avg_loss = abs(np.mean(negative_returns)) if len(negative_returns) > 0 else 0.0

            payoff_ratio = avg_gain / avg_loss if avg_loss > 0 else 0.0

            return hit_rate, payoff_ratio
        except:
            return 0.0, 0.0

    def _calculate_risk_metrics(self, returns: np.ndarray) -> Tuple[float, float]:
        """Calculate VaR and CVaR at 95% confidence level."""
        try:
            if len(returns) == 0:
                return 0.0, 0.0

            var_95 = np.percentile(returns, (1 - self.confidence_level) * 100)
            cvar_95 = np.mean(returns[returns <= var_95]) if np.any(returns <= var_95) else var_95

            return var_95, cvar_95
        except:
            return 0.0, 0.0

    def _calculate_information_ratio(self, returns: np.ndarray) -> float:
        """Calculate information ratio."""
        try:
            if len(returns) == 0 or np.std(returns) == 0:
                return 0.0

            # Use benchmark return of 0 (market-neutral)
            benchmark_return = 0.0
            excess_returns = returns - benchmark_return
            tracking_error = np.std(excess_returns)

            return np.mean(excess_returns) / tracking_error if tracking_error > 0 else 0.0
        except:
            return 0.0

    def _calculate_economic_significance(self, regime_data: pd.DataFrame, returns: np.ndarray) -> float:
        """Calculate economic significance score."""
        try:
            significance_factors = []

            # Return significance
            if len(returns) > 0:
                return_significance = min(abs(np.mean(returns)) * 100, 1.0)
                significance_factors.append(return_significance)

            # Volatility significance
            if len(returns) > 1:
                vol_significance = min(np.std(returns) * 10, 1.0)
                significance_factors.append(vol_significance)

            # Volume significance (if available)
            if 'volume' in regime_data.columns:
                volume = regime_data['volume'].values
                if len(volume) > 0:
                    volume_volatility = np.std(volume) / np.mean(volume) if np.mean(volume) > 0 else 0
                    volume_significance = min(volume_volatility, 1.0)
                    significance_factors.append(volume_significance)

            return np.mean(significance_factors) if significance_factors else 0.5
        except:
            return 0.5

    def _calculate_trading_viability(self, hit_rate: float, payoff_ratio: float, sharpe_ratio: float) -> float:
        """Calculate trading viability score."""
        try:
            # Weighted combination of trading metrics
            viability = (
                0.4 * min(hit_rate, 1.0) +
                0.3 * min(payoff_ratio / 2.0, 1.0) +  # Normalize payoff ratio
                0.3 * min(max(sharpe_ratio, 0) / 2.0, 1.0)  # Normalize Sharpe ratio
            )
            return min(viability, 1.0)
        except:
            return 0.5

    def _calculate_stability_score(self, returns: np.ndarray) -> float:
        """Calculate regime stability score."""
        try:
            if len(returns) < 10:
                return 0.5

            # Rolling volatility stability
            window_size = min(10, len(returns) // 2)
            rolling_vol = pd.Series(returns).rolling(window=window_size).std()
            vol_stability = 1.0 - (rolling_vol.std() / rolling_vol.mean()) if rolling_vol.mean() > 0 else 0.0

            # Return consistency
            return_consistency = 1.0 - (np.std(returns) / abs(np.mean(returns))) if np.mean(returns) != 0 else 0.0

            return (vol_stability + return_consistency) / 2.0
        except:
            return 0.5

    def _calculate_risk_score(self, volatility: float, max_drawdown: float, var_95: float) -> float:
        """Calculate risk score (lower is better)."""
        try:
            # Normalize risk metrics
            vol_score = min(volatility * 10, 1.0)
            dd_score = min(max_drawdown * 5, 1.0)
            var_score = min(abs(var_95) * 20, 1.0)

            # Combined risk score (lower is better, so invert)
            risk_score = (vol_score + dd_score + var_score) / 3.0
            return 1.0 - risk_score  # Invert so higher is better
        except:
            return 0.5

    def _calculate_performance_score(self, sharpe_ratio: float, sortino_ratio: float,
                                   hit_rate: float, payoff_ratio: float) -> float:
        """Calculate overall performance score."""
        try:
            # Normalize metrics
            sharpe_norm = min(max(sharpe_ratio, 0) / 2.0, 1.0)
            sortino_norm = min(max(sortino_ratio, 0) / 2.0, 1.0)
            hit_rate_norm = min(hit_rate, 1.0)
            payoff_norm = min(payoff_ratio / 2.0, 1.0)

            # Weighted performance score
            performance = (
                0.3 * sharpe_norm +
                0.3 * sortino_norm +
                0.2 * hit_rate_norm +
                0.2 * payoff_norm
            )

            return min(performance, 1.0)
        except:
            return 0.5

    def _calculate_regime_rankings(self, regime_metrics: List[RegimeMetrics]) -> Dict[str, List[int]]:
        """Calculate various regime rankings."""
        try:
            rankings = {}

            # Performance rankings
            sharpe_ranking = sorted(range(len(regime_metrics)),
                                 key=lambda i: regime_metrics[i].sharpe_ratio, reverse=True)
            sortino_ranking = sorted(range(len(regime_metrics)),
                                   key=lambda i: regime_metrics[i].sortino_ratio, reverse=True)
            performance_ranking = sorted(range(len(regime_metrics)),
                                       key=lambda i: regime_metrics[i].performance_score, reverse=True)

            # Risk rankings (lower risk is better)
            risk_ranking = sorted(range(len(regime_metrics)),
                                key=lambda i: regime_metrics[i].risk_score, reverse=True)
            drawdown_ranking = sorted(range(len(regime_metrics)),
                                    key=lambda i: regime_metrics[i].max_drawdown)

            # Trading rankings
            hit_rate_ranking = sorted(range(len(regime_metrics)),
                                   key=lambda i: regime_metrics[i].hit_rate, reverse=True)
            payoff_ranking = sorted(range(len(regime_metrics)),
                                  key=lambda i: regime_metrics[i].payoff_ratio, reverse=True)

            # Economic rankings
            economic_ranking = sorted(range(len(regime_metrics)),
                                   key=lambda i: regime_metrics[i].economic_significance, reverse=True)
            trading_viability_ranking = sorted(range(len(regime_metrics)),
                                            key=lambda i: regime_metrics[i].trading_viability, reverse=True)

            rankings = {
                'sharpe_ratio': sharpe_ranking,
                'sortino_ratio': sortino_ranking,
                'performance_score': performance_ranking,
                'risk_score': risk_ranking,
                'max_drawdown': drawdown_ranking,
                'hit_rate': hit_rate_ranking,
                'payoff_ratio': payoff_ranking,
                'economic_significance': economic_ranking,
                'trading_viability': trading_viability_ranking
            }

            return rankings
        except Exception as e:
            self.logger.warning(f"Regime rankings calculation failed: {e}")
            return {}

    def _calculate_overall_quality_score(self, regime_metrics: List[RegimeMetrics]) -> float:
        """Calculate overall quality score for all regimes."""
        try:
            if not regime_metrics:
                return 0.0

            # Weighted average of key metrics
            scores = []
            weights = []

            for metric in regime_metrics:
                # Performance weight
                perf_score = metric.performance_score
                scores.append(perf_score)
                weights.append(0.3)

                # Risk-adjusted weight
                risk_adj_score = (metric.sharpe_ratio + metric.sortino_ratio) / 2.0
                risk_adj_score = min(max(risk_adj_score, 0) / 2.0, 1.0)
                scores.append(risk_adj_score)
                weights.append(0.25)

                # Trading viability weight
                trading_score = metric.trading_viability
                scores.append(trading_score)
                weights.append(0.25)

                # Stability weight
                stability_score = metric.stability_score
                scores.append(stability_score)
                weights.append(0.2)

            # Calculate weighted average
            if weights:
                overall_score = np.average(scores, weights=weights)
                return min(overall_score, 1.0)
            else:
                return 0.0
        except Exception as e:
            self.logger.warning(f"Overall quality score calculation failed: {e}")
            return 0.5

    def _analyze_regime_transitions(self, regime_labels: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze regime transitions."""
        try:
            transitions = []
            unique_regimes = set(regime_labels)

            for i in range(1, len(regime_labels)):
                if regime_labels[i] != regime_labels[i-1]:
                    transition = {
                        'from_regime': regime_labels[i-1],
                        'to_regime': regime_labels[i],
                        'transition_point': i,
                        'transition_type': f"{regime_labels[i-1]} -> {regime_labels[i]}"
                    }
                    transitions.append(transition)

            return transitions
        except Exception as e:
            self.logger.warning(f"Regime transition analysis failed: {e}")
            return []

    def _calculate_risk_adjusted_rankings(self, regime_metrics: List[RegimeMetrics]) -> Dict[str, List[int]]:
        """Calculate risk-adjusted rankings."""
        try:
            # Sharpe ratio ranking
            sharpe_ranking = sorted(range(len(regime_metrics)),
                                 key=lambda i: regime_metrics[i].sharpe_ratio, reverse=True)

            # Sortino ratio ranking
            sortino_ranking = sorted(range(len(regime_metrics)),
                                   key=lambda i: regime_metrics[i].sortino_ratio, reverse=True)

            # Calmar ratio ranking
            calmar_ranking = sorted(range(len(regime_metrics)),
                                  key=lambda i: regime_metrics[i].calmar_ratio, reverse=True)

            # Information ratio ranking
            info_ranking = sorted(range(len(regime_metrics)),
                                key=lambda i: regime_metrics[i].information_ratio, reverse=True)

            return {
                'sharpe_ratio': sharpe_ranking,
                'sortino_ratio': sortino_ranking,
                'calmar_ratio': calmar_ranking,
                'information_ratio': info_ranking
            }
        except Exception as e:
            self.logger.warning(f"Risk-adjusted rankings calculation failed: {e}")
            return {}

    def _calculate_economic_rankings(self, regime_metrics: List[RegimeMetrics]) -> Dict[str, List[int]]:
        """Calculate economic rankings."""
        try:
            # Economic significance ranking
            economic_ranking = sorted(range(len(regime_metrics)),
                                    key=lambda i: regime_metrics[i].economic_significance, reverse=True)

            # Trading viability ranking
            trading_ranking = sorted(range(len(regime_metrics)),
                                   key=lambda i: regime_metrics[i].trading_viability, reverse=True)

            # Stability ranking
            stability_ranking = sorted(range(len(regime_metrics)),
                                     key=lambda i: regime_metrics[i].stability_score, reverse=True)

            return {
                'economic_significance': economic_ranking,
                'trading_viability': trading_ranking,
                'stability_score': stability_ranking
            }
        except Exception as e:
            self.logger.warning(f"Economic rankings calculation failed: {e}")
            return {}

    def _calculate_trading_rankings(self, regime_metrics: List[RegimeMetrics]) -> Dict[str, List[int]]:
        """Calculate trading-specific rankings."""
        try:
            # Hit rate ranking
            hit_rate_ranking = sorted(range(len(regime_metrics)),
                                    key=lambda i: regime_metrics[i].hit_rate, reverse=True)

            # Payoff ratio ranking
            payoff_ranking = sorted(range(len(regime_metrics)),
                                  key=lambda i: regime_metrics[i].payoff_ratio, reverse=True)

            # Risk score ranking (higher is better)
            risk_ranking = sorted(range(len(regime_metrics)),
                                key=lambda i: regime_metrics[i].risk_score, reverse=True)

            return {
                'hit_rate': hit_rate_ranking,
                'payoff_ratio': payoff_ranking,
                'risk_score': risk_ranking
            }
        except Exception as e:
            self.logger.warning(f"Trading rankings calculation failed: {e}")
            return {}

def create_enhanced_regime_evaluator(config: Dict[str, Any]) -> EnhancedRegimeEvaluator:
    """Create enhanced regime evaluator."""
    return EnhancedRegimeEvaluator(config)

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and
                VECTORBT_AVAILABLE)

    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str,
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

    def _pandas_rolling_operation(self, data: pd.Series, operation: str,
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def _vectorbt_apply_operation(self, data: pd.Series, func,
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)

        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
