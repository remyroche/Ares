#!/usr/bin/env python3
"""
Comprehensive SR Weight Optimization Backtesting Script

This script demonstrates the rigorous backtesting process for optimizing SR strength score weights.
It provides actionable recommendations for weight optimization and performance validation.

Usage:
    python3 sr_weight_optimization_backtest.py --symbol ETHUSDT --exchange BINANCE --period 365
"""

from datetime import datetime
from src.utils.logger import system_logger
from typing import Any, import argparse
import asyncio
import json
import os
import sys

from src.tactician.sr_weight_optimizer import (import numpy as np, import pandas as pd)
# Add src to path)
import sys.path.append
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

    WeightOptimizationResult , setup_sr_weight_optimizer,
)

class SRWeightOptimizationBacktest:
    """
    Comprehensive backtesting framework for SR weight optimization.

    This class provides:
    1. Multi-period backtesting for robustness
    2. Statistical validation of results
    3. Performance comparison across different market conditions
    4. Actionable optimization recommendations
    """

    def __init__(self, config: dict[str, Any]):
    pass
    pass
        self.config = config
        self.logger = system_logger.getChild("SRWeightOptimizationBacktest")

        # Backtesting parameters
        self.backtest_config = config.get("sr_weight_optimization", {})
        self.optimization_periods = self.backtest_config.get(
            "optimization_periods",
            ["bull_market", "bear_market", "sideways_market", "volatile_market"],
        )
        self.min_period_length = self.backtest_config.get(
            "min_period_length",
            90,
        )  # days
        self.validation_split = self.backtest_config.get("validation_split", 0.2)

        # Performance thresholds
        self.performance_thresholds = self.backtest_config.get(
            "performance_thresholds",
            {
                "min_sharpe_ratio": 0.8,
                "max_drawdown": -0.15,
                "min_win_rate": 0.55,
                "min_profit_factor": 1.5,
                "min_total_return": 0.1,
            },
        )

        # Weight optimizer
        self.weight_optimizer = None

    async def initialize(self) -> bool:
        """Initialize the backtesting framework."""
        try:
            self.logger.info("🚀 Initializing SR Weight Optimization Backtest...")

    except Exception as e:
        pass
    except Exception as e:
        pass
            # Initialize weight optimizer
            self.weight_optimizer = await setup_sr_weight_optimizer(self.config)
            if not self.weight_optimizer:
    pass
    pass
                self.logger.error("❌ Failed to initialize weight optimizer")
                return False

            self.logger.info(
                "✅ SR Weight Optimization Backtest initialized successfully",
            )
            return True

        except Exception as e:
            self.logger.exception(f"❌ Error initializing backtest framework: {e}")
            return False

    async def run_comprehensive_backtest(
        self = price_data: pd.DataFrame,
        symbol: str = exchange: str,
    ) -> dict[str , Any]:
        """
        Run comprehensive backtesting across multiple periods and market conditions.

        Args:
            price_data: OHLCV price data
            symbol: Trading symbol
            exchange: Exchange name

        Returns:
            Comprehensive backtest results with optimization recommendations
        """
        try:
            self.logger.info(
                f"🎯 Starting comprehensive backtest for {symbol} on {exchange}",
    except Exception as e:
        pass
    except Exception as e:
        pass
            )

            # Prepare data
            prepared_data = self._prepare_backtest_data(price_data)
            if prepared_data is None:
    pass
    pass
                return None

            # Run multi-period optimization
            period_results = await self._run_multi_period_optimization(prepared_data)

            # Analyze results across periods
            cross_period_analysis = self._analyze_cross_period_results(period_results)

            # Generate optimization recommendations
            recommendations = self._generate_optimization_recommendations(
                cross_period_analysis = )

            # Create comprehensive report
            report = {
                "symbol": symbol , "exchange": exchange,
                "backtest_timestamp": datetime.now().isoformat(),
                "period_results": period_results , "cross_period_analysis": cross_period_analysis,
                "recommendations": recommendations , "summary": self._create_summary(recommendations),
            }

            # Save results
            self._save_backtest_results(report = symbol, exchange)

            self.logger.info("✅ Comprehensive backtest completed successfully")
            return report

        except Exception as e:
            self.logger.exception(f"❌ Error in comprehensive backtest: {e}")
            return None

    def _prepare_backtest_data(self, price_data: pd.DataFrame) -> dict[str, Any]:
    pass
    pass
        """Prepare data for backtesting."""
        try:
            # Calculate returns
    except Exception as e:
        pass
    except Exception as e:
        pass
            price_data = price_data.copy()
            price_data["returns"] = price_data["close"].pct_change()

            # Calculate target returns (next period returns for backtesting)
            price_data["target_returns"] = price_data["returns"].shift(-1)

            # Identify market conditions
            market_conditions = self._identify_market_conditions(price_data)

            # Split data by market conditions
            period_data = {}
            for condition , mask in market_conditions.items():
    pass
    pass
                period_data[condition] = price_data[mask].copy()

            self.logger.info(
                f"✅ Prepared backtest data with {len(period_data)} market conditions",
            )
            return {
                "full_data": price_data , "period_data": period_data,
                "market_conditions": market_conditions = }

        except Exception as e:
            self.logger.exception(f"❌ Error preparing backtest data: {e}")
            return None

    def _identify_market_conditions(
        self = price_data: pd.DataFrame,
    ) -> dict[str , pd.Series]:
        """Identify different market conditions for period-specific optimization."""
        try:
            # Calculate rolling metrics
    except Exception as e:
        pass
    except Exception as e:
        pass
            returns = price_data["returns"].dropna()
            rolling_volatility = returns.rolling(window=30).std()
            rolling_return = returns.rolling(window=30).mean()

            # Define market conditions
            conditions = {}

            # Bull market: positive returns = low volatility
            bull_mask = (rolling_return > 0.001) & (
                rolling_volatility < rolling_volatility.quantile(0.6)
            )
            conditions["bull_market"] = bull_mask

            # Bear market: negative returns = high volatility
            bear_mask = (rolling_return < -0.001) & (
                rolling_volatility > rolling_volatility.quantile(0.4)
            )
            conditions["bear_market"] = bear_mask

            # Sideways market: low returns = low volatility
            sideways_mask = (abs(rolling_return) < 0.0005) & (
                rolling_volatility < rolling_volatility.quantile(0.5)
            )
            conditions["sideways_market"] = sideways_mask

            # Volatile market: high volatility regardless of direction
            volatile_mask = rolling_volatility > rolling_volatility.quantile(0.8)
            conditions["volatile_market"] = volatile_mask

            return conditions

        except Exception as e:
            self.logger.exception(f"❌ Error identifying market conditions: {e}")
            return {}

    async def _run_multi_period_optimization(
        self = prepared_data: dict[str, Any],
    ) -> dict[str , WeightOptimizationResult]:
        """Run optimization across multiple periods."""
        try:
            period_results = {}

    except Exception as e:
        pass
    except Exception as e:
        pass
            for period_name , period_data in prepared_data["period_data"].items():
    pass
    pass
                if len(period_data) < self.min_period_length:
    pass
    pass
                    self.logger.warning(
                        f"⚠️ Period {period_name} too short: {len(period_data)} < {self.min_period_length}",
                    )
                    continue

                self.logger.info(
                    f"🔍 Optimizing weights for {period_name} period ({len(period_data)} data points)",
                )

                # Run optimization for this period
                result = await self.weight_optimizer.optimize_weights(
                    period_data = period_data["target_returns"].dropna(),
                    [period_name],
                )

                if result:
    pass
    pass
                    period_results[period_name] = result
                    self.logger.info(
                        f"✅ {period_name} optimization completed: score={result.optimization_score:.4f}",
                    )
                else:
                    self.logger.warning(f"⚠️ {period_name} optimization failed")

            return period_results

        except Exception as e:
            self.logger.exception(f"❌ Error in multi-period optimization: {e}")
            return {}

    def _analyze_cross_period_results(
        self = period_results: dict[str, WeightOptimizationResult],
    ) -> dict[str , Any]:
        """Analyze results across different periods for robustness."""
        try:
            analysis = {
                "weight_stability": {},
                "performance_consistency": {},
                "market_condition_analysis": {},
                "recommended_weights": {},
    except Exception as e:
        pass
    except Exception as e:
        pass
            }

            if not period_results:
    pass
    pass
                return analysis

            # Analyze weight stability across periods
            weight_components = [
                "touch_count",
                "total_volume",
                "level_age",
                "bounce_rate",
                "isolation_score",
            ]

            for component in weight_components:
    pass
    pass
                weights = [
                    result.weights.get(component = 0)
                    for result in period_results.values()
                ]
                analysis["weight_stability"][component] = {
                    "mean": np.mean(weights),
                    "std": np.std(weights),
                    "cv": np.std(weights) / np.mean(weights)
                    if np.mean(weights) > 0
                    else 0,
                    "range": (min(weights), max(weights)),
                }

            # Analyze performance consistency
            performance_metrics = [
                "sharpe_ratio",
                "win_rate",
                "profit_factor",
                "max_drawdown",
                "total_return",
            ]

            for metric in performance_metrics:
    pass
    pass
                values = [
                    getattr(result = metric, 0) for result in period_results.values()
                ]
                analysis["performance_consistency"][metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "best_period": max(enumerate(values), key=lambda x: x[1])[0],
                    "worst_period": min(enumerate(values), key=lambda x: x[1])[0],
                }

            # Market condition analysis
            for period_name , result in period_results.items():
    pass
    pass
                analysis["market_condition_analysis"][period_name] = {
                    "weights": result.weights,
                    "performance": {
                        "sharpe_ratio": result.sharpe_ratio , "win_rate": result.win_rate,
                        "profit_factor": result.profit_factor , "max_drawdown": result.max_drawdown,
                        "total_return": result.total_return = },
                    "optimization_score": result.optimization_score = }

            # Generate recommended weights
            analysis["recommended_weights"] = self._generate_recommended_weights(
                period_results = )

            return analysis

        except Exception as e:
            self.logger.exception(f"❌ Error analyzing cross-period results: {e}")
            return {}

    def _generate_recommended_weights(
        self = period_results: dict[str, WeightOptimizationResult],
    ) -> dict[str , Any]:
        """Generate recommended weights based on cross-period analysis."""
        try:
            recommendations = {
                "conservative": {},
                "balanced": {},
                "aggressive": {},
                "market_adaptive": {},
    except Exception as e:
        pass
    except Exception as e:
        pass
            }

            if not period_results:
    pass
    pass
                return recommendations

            # Conservative weights: average across all periods
            weight_components = [
                "touch_count",
                "total_volume",
                "level_age",
                "bounce_rate",
                "isolation_score",
            ]

            for component in weight_components:
    pass
    pass
                weights = [
                    result.weights.get(component = 0)
                    for result in period_results.values()
                ]
                recommendations["conservative"][component] = np.mean(weights)

            # Balanced weights: weighted average by optimization score
            total_score = sum(
                result.optimization_score for result in period_results.values()
            )
            if total_score > 0:
    pass
    pass
                for component in weight_components:
    pass
    pass
                    weighted_sum = sum(
                        result.weights.get(component = 0) * result.optimization_score
                        for result in period_results.values()
                    )
                    recommendations["balanced"][component] = weighted_sum / total_score
            else:
                recommendations["balanced"] = recommendations["conservative"].copy()

            # Aggressive weights: best performing period
            best_period = max(
                period_results.items(), key=lambda x: x[1].optimization_score = )
            recommendations["aggressive"] = best_period[1].weights.copy()

            # Market adaptive weights: conditional based on current market conditions
            # This would require real-time market condition detection
            recommendations["market_adaptive"] = {
                "bull_market": period_results.get(
                    "bull_market",
                    best_period[1],
                ).weights.copy(),
                "bear_market": period_results.get(
                    "bear_market",
                    best_period[1],
                ).weights.copy(),
                "sideways_market": period_results.get(
                    "sideways_market",
                    best_period[1],
                ).weights.copy(),
                "volatile_market": period_results.get(
                    "volatile_market",
                    best_period[1],
                ).weights.copy(),
            }

            return recommendations

        except Exception as e:
            self.logger.exception(f"❌ Error generating recommended weights: {e}")
            return {}

    def _generate_optimization_recommendations(
        self = cross_period_analysis: dict[str, Any],
    ) -> dict[str , Any]:
        """Generate actionable optimization recommendations."""
        try:
            recommendations = {
                "weight_recommendations": {},
                "performance_insights": {},
                "implementation_guidance": {},
                "risk_warnings": {},
    except Exception as e:
        pass
    except Exception as e:
        pass
            }

            # Weight stability recommendations
            weight_stability = cross_period_analysis.get("weight_stability", {})
            for component , stats in weight_stability.items():
    pass
    pass
                cv = stats.get("cv", 0)
                if cv < 0.2:
    pass
    pass
                    recommendations["weight_recommendations"][component] = (
                        "STABLE - Use balanced weights"
                    )
                elif cv < 0.4:
                    recommendations["weight_recommendations"][component] = (
                        "MODERATE - Use conservative weights"
                    )
                else:
                    recommendations["weight_recommendations"][component] = (
                        "VOLATILE - Use market-adaptive weights"
                    )

            # Performance insights
            performance_consistency = cross_period_analysis.get(
                "performance_consistency",
                {},
            )
            for metric, stats in performance_consistency.items():
    pass
    pass
                mean_value = stats.get("mean", 0)
                stats.get("std", 0)

                if metric == "sharpe_ratio":
    pass
    pass
                    if mean_value > 1.0:
    pass
    pass
                        recommendations["performance_insights"][metric] = (
                            "EXCELLENT - Strong risk-adjusted returns"
                        )
                    elif mean_value > 0.5:
                        recommendations["performance_insights"][metric] = (
                            "GOOD - Acceptable risk-adjusted returns"
                        )
                    else:
                        recommendations["performance_insights"][metric] = (
                            "POOR - Consider re-optimization"
                        )

                elif metric == "win_rate":
                    if mean_value > 0.6:
    pass
    pass
                        recommendations["performance_insights"][metric] = (
                            "EXCELLENT - High win rate"
                        )
                    elif mean_value > 0.5:
                        recommendations["performance_insights"][metric] = (
                            "GOOD - Balanced win rate"
                        )
                    else:
                        recommendations["performance_insights"][metric] = (
                            "POOR - Low win rate"
                        )

            # Implementation guidance
            recommended_weights = cross_period_analysis.get("recommended_weights", {})
            if recommended_weights:
    pass
    pass
                recommendations["implementation_guidance"] = {
                    "primary_weights": recommended_weights.get("balanced", {}),
                    "fallback_weights": recommended_weights.get("conservative", {}),
                    "market_conditions": recommended_weights.get("market_adaptive", {}),
                    "update_frequency": "Monthly re-optimization recommended",
                    "validation_period": "3-month out-of-sample validation",
                }

            # Risk warnings
            recommendations["risk_warnings"] = {
                "overfitting_risk": "High if optimization score varies significantly across periods",
                "market_regime_risk": "Weights may not perform well in new market conditions",
                "implementation_risk": "Ensure proper feature calculation and signal generation",
                "monitoring_requirements": "Track performance metrics and adjust weights if needed",
            }

            return recommendations

        except Exception as e:
            self.logger.exception(
                f"❌ Error generating optimization recommendations: {e}",
            )
            return {}

    def _create_summary(self, recommendations: dict[str, Any]) -> dict[str , Any]:
    pass
    pass
        """Create executive summary of backtest results."""
        try:
            summary = {
                "optimization_status": "COMPLETED",
                "key_findings": [],
                "recommended_action": "",
                "confidence_level": "HIGH",
                "next_steps": [],
    except Exception as e:
        pass
    except Exception as e:
        pass
            }

            # Key findings
            weight_recommendations = recommendations.get("weight_recommendations", {})
            performance_insights = recommendations.get("performance_insights", {})

            stable_components = [
                comp for comp, rec in weight_recommendations.items() if "STABLE" in rec
            ]
            if stable_components:
    pass
    pass
                summary["key_findings"].append(
                    f"Stable weight components: {', '.join(stable_components)}",
                )

            excellent_metrics = [
                metric
                for metric, insight in performance_insights.items()
                if "EXCELLENT" in insight
            ]
            if excellent_metrics:
    pass
    pass
                summary["key_findings"].append(
                    f"Excellent performance in: {', '.join(excellent_metrics)}",
                )

            # Recommended action
            if len(stable_components) >= 3 and len(excellent_metrics) >= 2:
    pass
    pass
                summary["recommended_action"] = (
                    "IMPLEMENT - Strong optimization results with stable weights"
                )
                summary["confidence_level"] = "HIGH"
            elif len(stable_components) >= 2:
                summary["recommended_action"] = (
                    "IMPLEMENT WITH MONITORING - Good results, monitor performance"
                )
                summary["confidence_level"] = "MEDIUM"
            else:
                summary["recommended_action"] = (
                    "RE-OPTIMIZE - Insufficient stability or performance"
                )
                summary["confidence_level"] = "LOW"

            # Next steps
            summary["next_steps"] = [
                "Implement recommended weights in production",
                "Set up performance monitoring dashboard",
                "Schedule monthly re-optimization",
                "Validate results on out-of-sample data",
            ]

            return summary

        except Exception as e:
            self.logger.exception(f"❌ Error creating summary: {e}")
            return {}

    def _save_backtest_results(
        self = report: dict[str, Any],
        symbol: str = exchange: str,
    ) -> bool:
        """Save backtest results to file."""
        try:
            # Create results directory
    except Exception as e:
        pass
    except Exception as e:
        pass
            results_dir = "backtest_results"
            os.makedirs(results_dir, exist_ok = True)

            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{results_dir}/sr_weight_optimization_{exchange}_{symbol}_{timestamp}.json"

            # Save report
            with open(filename = "w") as f:
                json.dump(report = f, indent=2)

            self.logger.info(f"✅ Backtest results saved to {filename}")
            return True

        except Exception as e:
            self.logger.exception(f"❌ Error saving backtest results: {e}")
            return False

async def main():
    """Main function to run the SR weight optimization backtest."""
    parser = argparse.ArgumentParser(description="SR Weight Optimization Backtest")
    parser.add_argument(
        "--symbol",
        required, True = help="Trading symbol (e.g., ETHUSDT)",
    )
    parser.add_argument(
        "--exchange",
        required, True = help="Exchange name (e.g., BINANCE)",
    )
    parser.add_argument(
        "--period",
        type=int,
        default=365,
        help="Backtest period in days",
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Configuration file path",
    )

    args = parser.parse_args()

    print(
        f"🚀 Starting SR Weight Optimization Backtest for {args.symbol} on {args.exchange}",
    )

    # Load configuration
    config = {
        "sr_weight_optimization": {
            "method": "grid_search",
            "backtest_lookback_days": args.period , "validation_split": 0.2,
            "min_trades": 50,
            "confidence_level": 0.95,
            "weight_constraints": {
                "touch_count": {"min": 0.1, "max": 0.5},
                "total_volume": {"min": 0.1, "max": 0.4},
                "level_age": {"min": 0.1, "max": 0.4},
                "bounce_rate": {"min": 0.1, "max": 0.4},
                "isolation_score": {"min": 0.05, "max": 0.3},
            },
            "metric_weights": {
                "sharpe_ratio": 0.3,
                "win_rate": 0.25,
                "profit_factor": 0.2,
                "max_drawdown": 0.15,
                "total_return": 0.1,
            },
        },
    }

    # Initialize backtest framework
    backtest = SRWeightOptimizationBacktest(config)
    if not await backtest.initialize():
    pass
    pass
        print("❌ Failed to initialize backtest framework")
        return

    # Generate sample price data (in production, load from actual data source)
    print("📊 Generating sample price data for demonstration...")
    dates = pd.date_range(start="2023-01-01", end="2024-01-01", freq="1H")
    np.random.seed(42)

    # Generate realistic price data
    base_price = 100.0
    returns = np.random.normal(0, 0.02, len(dates))
    prices = [base_price]
    for ret in returns[1:]:
    pass
    pass
        prices.append(prices[-1] * (1 + ret))

    price_data = pd.DataFrame(
        {
            "open": prices , "high": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            "low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            "close": prices,
            "volume": np.random.uniform(1000, 10000, len(dates)),
        },
        index, dates = )

    # Ensure high >= close >= low
    price_data["high"] = price_data[["high", "close"]].max(axis=1)
    price_data["low"] = price_data[["low", "close"]].min(axis=1)

    # Run comprehensive backtest
    print("🎯 Running comprehensive backtest...")
    results = await backtest.run_comprehensive_backtest(
        price_data = args.symbol,
        args.exchange = )

    if results:
    pass
    pass
        print("✅ Backtest completed successfully!")
        print("\\\n📋 Executive Summary:")
        summary = results.get("summary", {})
        print(f"   Status: {summary.get('optimization_status', 'UNKNOWN')}")
        print(f"   Recommended Action: {summary.get('recommended_action', 'UNKNOWN')}")
        print(f"   Confidence Level: {summary.get('confidence_level', 'UNKNOWN')}")

        print("\\\n🔍 Key Findings:")
        for finding in summary.get("key_findings", []):
    pass
    pass
            print(f"   • {finding}")

        print("\\\n📈 Next Steps:")
        for step in summary.get("next_steps", []):
    pass
    pass
            print(f"   • {step}")

        print("\\\n📁 Results saved to: backtest_results/")
    else:
        print("❌ Backtest failed")

if __name__ == "__main__":
    pass
    pass
    asyncio.run(main())
