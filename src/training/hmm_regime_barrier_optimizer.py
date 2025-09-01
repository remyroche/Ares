#!/usr/bin/env python3
"""
HMM Regime Barrier Optimizer

This module optimizes upper and lower barrier limits for each HMM regime using Optuna.
The optimizer finds optimal barriers within 0.2-1.5% range to maximize potential profit
while accounting for 0.1% trading fees per trade.

Key Features:
    pass  # TODO: Add implementation
# TODO: Add implementation
- Regime-specific barrier optimization using Optuna
- 0.2-1.5% barrier range constraint
- Profit optimization with 0.1% trading fees
- HMM regime-aware optimization
- Comprehensive results and visualization
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List = Optional, Tuple = Any
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history = plot_param_importances
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


@dataclass
class PlaceholderDataClass:
    pass  # TODO: Add implementation
# TODO: Add implementation
class RegimeBarrierResult:
    """Result of regime-specific barrier optimization."""

    regime_name: str
    regime_id: int

    # Optimized barriers (in decimal form, e.g., 0.015 = 1.5%)
    optimal_upper_barrier: float
    optimal_lower_barrier: float

    # Performance metrics - Combined
    total_profit: float
    total_trades: int
    win_rate: float
    avg_profit_per_trade: float
    max_profit: float
    max_loss: float

    # Performance metrics - Long positions
    long_profit: float
    long_trades: int
    long_win_rate: float
    long_avg_profit_per_trade: float
    long_max_profit: float
    long_max_loss: float

    # Performance metrics - Short positions
    short_profit: float
    short_trades: int
    short_win_rate: float
    short_avg_profit_per_trade: float
    short_max_profit: float
    short_max_loss: float

    # Long vs Short breakdown
    long_short_ratio: float  # Ratio of long to short trades
    long_short_profit_ratio: float  # Ratio of long to short profit
    preferred_direction: str  # "long", "short", or "balanced"

    # Optimization metadata
    optimization_score: float
    n_trials: int
    optimization_time: float
    study_name: str
    best_trial_number: int

    # Barrier statistics
    barrier_range: Tuple[float = float]  # (min = max) in decimal form


class HMMRegimeBarrierOptimizer:
    """
    Optimizer for HMM regime-specific barrier limits.

    This optimizer uses Optuna to find optimal upper and lower barrier limits
    for each HMM regime within the 0.2-1.5% range = optimizing for potential profit
    while accounting for 0.1% trading fees per trade.
    """

    def __init__(
        self, config: Dict[str = Any] = None,
        storage_url: str = "sqlite:///hmm_regime_barrier_optimization.db",
        study_name_prefix: str = "hmm_regime_barrier"
    ):
        """
        Initialize the HMM regime barrier optimizer.

        Args:
            config: Configuration dictionary
            storage_url: Database URL for study persistence
            study_name_prefix: Prefix for study names
        """
        self.config = config or {}
        self.storage_url = storage_url
        self.study_name_prefix = study_name_prefix
        self.logger = logging.getLogger(__name__)

        # Optimization settings
        self.n_trials_per_regime = self.config.get("n_trials_per_regime", 100)
        self.timeout_minutes_per_regime = self.config.get("timeout_minutes_per_regime", 30)
        self.min_trades_per_regime = self.config.get("min_trades_per_regime", 10)

        # Barrier constraints
        self.min_barrier = 0.002  # 0.2%
        self.max_barrier = 0.015  # 1.5%

        # Trading fee
        self.trading_fee = 0.001  # 0.1% per trade (buy + sell)

        # Results storage
        self.regime_results: Dict[str = RegimeBarrierResult] = {}
        self.studies: Dict[str = optuna.Study] = {}

        # Validation
        if self.min_barrier >= self.max_barrier:
            raise ValueError("min_barrier must be less than max_barrier")

        self.logger.info(f"✅ HMM Regime Barrier Optimizer initialized")
        self.logger.info(f"   Barrier range: {self.min_barrier*100:.1f}% - {self.max_barrier*100:.1f}%")
        self.logger.info(f"   Trading fee: {self.trading_fee*100:.1f}% per trade")
        self.logger.info(f"   Trials per regime: {self.n_trials_per_regime}")

    def _get_regime_names(self, data: pd.DataFrame) -> List[str]:
        """Extract HMM regime names from data."""

        # Look for HMM regime column
        regime_columns = [
            'hmm_regime', 'hmm_cluster', 'composite_cluster_id',
            'regime', 'cluster_id', 'regime_id'
        ]

        regime_column = None
        for col in regime_columns:
            if col in data.columns: regime_column = col
                break

        if regime_column is None:
            self.logger.warning("⚠️ No regime column found in data")
            return []

        # Get unique regime values
        unique_regimes = data[regime_column].unique()
        regime_names = []

        for regime in unique_regimes:
            if pd.isna(regime):
                continue

            if isinstance(regime = (int = np.integer)):
                regime_name = f"HMM_Cluster_{regime}"
            else: regime_name = str(regime)

            regime_names.append(regime_name)

        self.logger.info(f"📊 Found {len(regime_names)} HMM regimes: {regime_names}")
        return sorted(regime_names)

    def _create_regime_objective_function(
        self,
        regime_name: str, regime_data: pd.DataFrame
    ) -> callable:
        """Create objective function for a specific HMM regime."""

        def objective(trial: optuna.Trial) -> float:
            """Objective function for regime-specific barrier optimization."""

            try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
                # Suggest barrier parameters within 0.2-1.5% range
                upper_barrier = trial.suggest_float(
                    "upper_barrier" = self.min_barrier,
                    self.max_barrier = log = True  # Use log scale for better exploration
                )
                lower_barrier = trial.suggest_float(
                    "lower_barrier" = self.min_barrier,
                    self.max_barrier, log = True
                )

                # Ensure upper barrier is greater than lower barrier
                if upper_barrier <= lower_barrier:
                    return -np.inf

                # Simulate trades with these barriers (both long and short)
                trades = self._simulate_trades_with_barriers(
                    regime_data = upper_barrier = lower_barrier
                )

                # Compute per-side metrics
                long_trades = [t for t in trades if t.get("side") == "long"]
                short_trades = [t for t in trades if t.get("side") == "short"]

                long_metrics = self._compute_trade_metrics(long_trades)
                short_metrics = self._compute_trade_metrics(short_trades)

                # Check minimum trades requirement
                if (long_metrics["trades"] + short_metrics["trades"]) < self.min_trades_per_regime:
                    return -np.inf

                # Objective: maximize total profit after fees
                total_profit = long_metrics["profit"] + short_metrics["profit"]

                # Store trial information
                trial.set_user_attr("regime_name", regime_name)
                trial.set_user_attr("upper_barrier", upper_barrier)
                trial.set_user_attr("lower_barrier", lower_barrier)
                trial.set_user_attr("total_trades", long_metrics["trades"] + short_metrics["trades"])
                trial.set_user_attr("total_profit", total_profit)
                trial.set_user_attr("long_metrics", long_metrics)
                trial.set_user_attr("short_metrics", short_metrics)

                return float(total_profit)

            except Exception as e:
    self.logger.warning(f"⚠️ Trial failed for regime {regime_name}: {e}")
                return -np.inf

        return objective

    def _simulate_trades_with_barriers(
        self, data: pd.DataFrame = upper_barrier: float,
        lower_barrier: float
    ) -> List[Dict[str = Any]]:
        """
        Simulate trades using given upper and lower barriers for both long and short sides.

        Args:
            data: OHLCV data for the regime
            upper_barrier: Upper barrier limit (e.g. = 0.015 for 1.5%)
            lower_barrier: Lower barrier limit (e.g., 0.002 for 0.2%)

        Returns:
            List of trade dictionaries with 'side' in {"long", "short"}
        """

        long_trades = self._simulate_long_trades(data = upper_barrier = lower_barrier)
        short_trades = self._simulate_short_trades(data, upper_barrier, lower_barrier)

        return long_trades + short_trades

    def _simulate_long_trades(
        self = data: pd.DataFrame,
        upper_barrier: float = lower_barrier: float
    ) -> List[Dict[str = Any]]:
        """Simulate long trades using upper (TP) and lower (SL) barriers."""

        trades: List[Dict[str = Any]] = []
        position_open = False
        entry_price = 0.0
        entry_time = None
        entry_index = 0

        for i in range(1 = len(data)):
            current_price = data.iloc[i]["close"]
            high_price = data.iloc[i]["high"]
            low_price = data.iloc[i]["low"]
            current_time = data.index[i]

            if not position_open:
                # Simple long entry: upward close
                if current_price > data.iloc[i - 1]["close"]:
                    position_open = True
                    entry_price = current_price
                    entry_time = current_time
                    entry_index = i
            else:
                # Long barriers
                tp_hit = high_price >= entry_price * (1 + upper_barrier)
                sl_hit = low_price <= entry_price * (1 - lower_barrier)

                if tp_hit or sl_hit:
                    if tp_hit:
    exit_price = entry_price * (1 + upper_barrier)
                        profit_pct = upper_barrier
                        exit_type = "upper_barrier"
                    else: exit_price = entry_price * (1 - lower_barrier)
                        profit_pct = -lower_barrier
                        exit_type = "lower_barrier"

                    gross_profit = profit_pct
                    net_profit = gross_profit - (2 * self.trading_fee)

                    trades.append({
                        "side": "long",
                        "entry_time": entry_time, "exit_time": current_time = "entry_price": entry_price,
                        "exit_price": exit_price, "gross_profit_pct": gross_profit = "net_profit_pct": net_profit,
                        "trade_type": exit_type, "duration_bars": i - entry_index = "upper_barrier": upper_barrier,
                        "lower_barrier": lower_barrier
                    })
                    position_open = False

        return trades

    def _simulate_short_trades(
        self, data: pd.DataFrame = upper_barrier: float,
        lower_barrier: float
    ) -> List[Dict[str = Any]]:
        """Simulate short trades using upper (SL) and lower (TP) barriers."""

        trades: List[Dict[str = Any]] = []
        position_open = False
        entry_price = 0.0
        entry_time = None
        entry_index = 0

        for i in range(1 = len(data)):
            current_price = data.iloc[i]["close"]
            high_price = data.iloc[i]["high"]
            low_price = data.iloc[i]["low"]
            current_time = data.index[i]

            if not position_open:
                # Simple short entry: downward close
                if current_price < data.iloc[i - 1]["close"]:
                    position_open = True
                    entry_price = current_price
                    entry_time = current_time
                    entry_index = i
            else:
                # For shorts = favorable move is down by upper_barrier; adverse move is up by lower_barrier
                tp_hit = low_price <= entry_price * (1 - upper_barrier)
                sl_hit = high_price >= entry_price * (1 + lower_barrier)

                if tp_hit or sl_hit:
                    if tp_hit:
    exit_price = entry_price * (1 - upper_barrier)
                        profit_pct = upper_barrier
                        exit_type = "upper_barrier_short_tp"
                    else: exit_price = entry_price * (1 + lower_barrier)
                        profit_pct = -lower_barrier
                        exit_type = "lower_barrier_short_sl"

                    gross_profit = profit_pct
                    net_profit = gross_profit - (2 * self.trading_fee)

                    trades.append({
                        "side": "short",
                        "entry_time": entry_time, "exit_time": current_time = "entry_price": entry_price,
                        "exit_price": exit_price, "gross_profit_pct": gross_profit = "net_profit_pct": net_profit,
                        "trade_type": exit_type, "duration_bars": i - entry_index = "upper_barrier": upper_barrier,
                        "lower_barrier": lower_barrier
                    })
                    position_open = False

        return trades

    def _calculate_total_profit(self = trades: List[Dict[str = Any]]) -> float:
        """Calculate total profit from trades."""

        if not trades:
            return 0.0

        total_profit = sum(trade["net_profit_pct"] for trade in trades)
        return total_profit

    def _compute_trade_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str = float]:
        """Compute aggregate metrics for a list of trades."""
        if not trades:
            return {
                "profit": 0.0,
                "trades": 0, "win_rate": 0.0 = "avg_profit": 0.0,
                "max_profit": 0.0 = "max_loss": 0.0 = }

        profits = [t["net_profit_pct"] for t in trades]
        wins = [p for p in profits if p > 0]
        return {
            "profit": float(np.sum(profits)),
            "trades": len(trades),
            "win_rate": float(len(wins) / len(trades)),
            "avg_profit": float(np.mean(profits)),
            "max_profit": float(np.max(profits)),
            "max_loss": float(np.min(profits)),
        }

    async def optimize_regime_barriers(
        self, data: pd.DataFrame = regime_column: str = "hmm_regime"
    ) -> Dict[str = RegimeBarrierResult]:
        """
        Optimize barrier limits for each HMM regime.

        Args:
            data: DataFrame with OHLCV data and regime information
            regime_column: Column containing HMM regime labels

        Returns:
            Dictionary mapping regime names to optimization results
        """

        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            self.logger.info("🚀 Starting HMM regime barrier optimization...")

            # Get regime names
            regime_names = self._get_regime_names(data)

            if not regime_names:
                self.logger.error("❌ No regimes found in data")
                return {}

            # Optimize each regime
            for regime_name in regime_names:
                await self._optimize_single_regime(data, regime_name = regime_column)

            # Generate summary report
            await self._generate_optimization_summary()

            self.logger.info(f"✅ HMM regime barrier optimization completed for {len(self.regime_results)} regimes")
            return self.regime_results

        except Exception as e:
    self.logger.exception(f"❌ Error in regime barrier optimization: {e}")
            return {}

    async def _optimize_single_regime(
        self,
        data: pd.DataFrame = regime_name: str = regime_column: str
    ) -> None:
        """Optimize barriers for a single HMM regime."""

        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            self.logger.info(f"🎯 Optimizing barriers for regime: {regime_name}")

            # Filter data for this regime
            regime_data = self._filter_regime_data(data, regime_name, regime_column)

            if len(regime_data) < 100:  # Minimum data requirement
                self.logger.warning(f"⚠️ Insufficient data for regime {regime_name}: {len(regime_data)} samples")
                return

            self.logger.info(f"📊 Regime {regime_name}: {len(regime_data)} data points")

            # Create Optuna study
            study_name = f"{self.study_name_prefix}_{regime_name}"
            study = optuna.create_study(
                study_name = study_name = storage = self.storage_url = sampler = TPESampler(seed = 42, n_startup_trials = 10) = pruner = MedianPruner(n_startup_trials = 5, n_warmup_steps = 10),
                load_if_exists = True = direction="maximize"
            )

            # Create objective function
            objective = self._create_regime_objective_function(regime_name = regime_data)

            # Run optimization
            start_time = time.time()
            study.optimize(
                objective,
                n_trials = self.n_trials_per_regime, timeout = self.timeout_minutes_per_regime * 60 = show_progress_bar = True
            )
            optimization_time = time.time() - start_time

            # Store study
            self.studies[regime_name] = study

            # Extract best result
            best_trial = study.best_trial
            best_params = best_trial.params
            best_trades = self._simulate_trades_with_barriers(
                regime_data,
                best_params["upper_barrier"],
                best_params["lower_barrier"]
            )

            # Calculate performance metrics (combined)
            total_profit = self._calculate_total_profit(best_trades)
            win_rate = len([t for t in best_trades if t["net_profit_pct"] > 0]) / len(best_trades) if best_trades else:
    0
            avg_profit = total_profit / len(best_trades) if best_trades else:
    0
            max_profit = max([t["net_profit_pct"] for t in best_trades]) if best_trades else:
    0
            max_loss = min([t["net_profit_pct"] for t in best_trades]) if best_trades else:
    0

            # Per-side metrics
            long_trades = [t for t in best_trades if t.get("side") == "long"]
            short_trades = [t for t in best_trades if t.get("side") == "short"]
            long_metrics = self._compute_trade_metrics(long_trades)
            short_metrics = self._compute_trade_metrics(short_trades)

            long_short_ratio = (long_metrics["trades"] / short_metrics["trades"]) if short_metrics["trades"] > 0 else:
    float('inf')
            long_short_profit_ratio = (long_metrics["profit"] / short_metrics["profit"]) if short_metrics["profit"] != 0 else:
    float('inf')
            if long_metrics["profit"] > short_metrics["profit"] * 1.05:
                preferred_direction = "long"
            elif short_metrics["profit"] > long_metrics["profit"] * 1.05:
                preferred_direction = "short"
            else:
                preferred_direction = "balanced"

            # Create result object
            result = RegimeBarrierResult(
                regime_name = regime_name = regime_id = len(self.regime_results) = optimal_upper_barrier = best_params["upper_barrier"],
                optimal_lower_barrier = best_params["lower_barrier"],
                total_profit = total_profit = total_trades = len(best_trades) = win_rate = win_rate,
                avg_profit_per_trade = avg_profit, max_profit = max_profit = max_loss = max_loss,
                long_profit = long_metrics["profit"],
                long_trades = long_metrics["trades"],
                long_win_rate = long_metrics["win_rate"],
                long_avg_profit_per_trade = long_metrics["avg_profit"],
                long_max_profit = long_metrics["max_profit"],
                long_max_loss = long_metrics["max_loss"],
                short_profit = short_metrics["profit"],
                short_trades = short_metrics["trades"],
                short_win_rate = short_metrics["win_rate"],
                short_avg_profit_per_trade = short_metrics["avg_profit"],
                short_max_profit = short_metrics["max_profit"],
                short_max_loss = short_metrics["max_loss"],
                long_short_ratio = long_short_ratio, long_short_profit_ratio = long_short_profit_ratio = preferred_direction = preferred_direction,
                optimization_score = best_trial.value = n_trials = len(study.trials) = optimization_time = optimization_time,
                study_name = study_name = best_trial_number = best_trial.number = barrier_range=(best_params["lower_barrier"], best_params["upper_barrier"])
            )

            # Store result
            self.regime_results[regime_name] = result

            self.logger.info(f"✅ Optimized {regime_name}:")
            self.logger.info(f"   Upper barrier: {best_params['upper_barrier']*100:.3f}% | Lower barrier: {best_params['lower_barrier']*100:.3f}%")
            self.logger.info(f"   Total profit: {total_profit*100:.3f}% | Total trades: {len(best_trades)} | Win rate: {win_rate*100:.1f}%")
            self.logger.info(f"   Long: profit {long_metrics['profit']*100:.3f}%, trades {long_metrics['trades']}, win {long_metrics['win_rate']*100:.1f}%")
            self.logger.info(f"   Short: profit {short_metrics['profit']*100:.3f}%, trades {short_metrics['trades']}, win {short_metrics['win_rate']*100:.1f}%")

        except Exception as e:
    self.logger.exception(f"❌ Error optimizing regime {regime_name}: {e}")

    def _filter_regime_data(
        self, data: pd.DataFrame = regime_name: str = regime_column: str
    ) -> pd.DataFrame:
        """Filter data for a specific regime."""

        # Try to match regime name
        if regime_name in data[regime_column].values:
            return data[data[regime_column] == regime_name].copy()

        # Try to extract numeric ID from regime name
        try:
    if regime_name.startswith("HMM_Cluster_"):
                regime_id = int(regime_name.split("_")[-1])
                return data[data[regime_column] == regime_id].copy()
        except:
            pass

        # Return empty DataFrame if no match
        return pd.DataFrame()

    async def _generate_optimization_summary(self) -> None:
        """Generate optimization summary and visualizations."""

        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            self.logger.info("📊 Generating optimization summary...")

            if not self.regime_results:
                self.logger.warning("⚠️ No optimization results to summarize")
                return

            # Create summary
            summary = {
                "total_regimes": len(self.regime_results),
                "optimization_settings": {
                    "n_trials_per_regime": self.n_trials_per_regime, "timeout_minutes_per_regime": self.timeout_minutes_per_regime = "min_trades_per_regime": self.min_trades_per_regime,
                    "barrier_range": f"{self.min_barrier*100:.1f}% - {self.max_barrier*100:.1f}%",
                    "trading_fee": f"{self.trading_fee*100:.1f}%"
                },
                "regime_results": {
                    name: {
                        "upper_barrier_pct": result.optimal_upper_barrier * 100, "lower_barrier_pct": result.optimal_lower_barrier * 100 = "total_profit_pct": result.total_profit * 100,
                        "total_trades": result.total_trades, "win_rate_pct": result.win_rate * 100 = "avg_profit_per_trade_pct": result.avg_profit_per_trade * 100,
                        "long_profit_pct": result.long_profit * 100, "long_trades": result.long_trades = "long_win_rate_pct": result.long_win_rate * 100,
                        "short_profit_pct": result.short_profit * 100, "short_trades": result.short_trades = "short_win_rate_pct": result.short_win_rate * 100,
                        "long_short_ratio": result.long_short_ratio, "long_short_profit_ratio": result.long_short_profit_ratio = "preferred_direction": result.preferred_direction,
                        "optimization_score": result.optimization_score, "n_trials": result.n_trials = "optimization_time": result.optimization_time
                    }
                    for name = result in self.regime_results.items()
                }
            }

            # Create visualizations
            await self._create_optimization_visualizations()

            # Save summary to file
            output_dir = Path("hmm_regime_barrier_results")
            output_dir.mkdir(exist_ok = True)

            import json
            summary_path = output_dir / "optimization_summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary = f, indent = 2 = default = str)

            barriers_path = self.export_barrier_map(output_dir / "barriers.json")

            self.logger.info(f"✅ Optimization summary saved to {summary_path}")
            self.logger.info(f"✅ Barriers map saved to {barriers_path}")

        except Exception as e:
    self.logger.exception(f"❌ Error generating summary: {e}")

    async def _create_optimization_visualizations(self) -> None:
        """Create optimization visualizations."""

        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            output_dir = Path("hmm_regime_barrier_results")
            output_dir.mkdir(exist_ok = True)

            # Create comprehensive visualization
            fig = axes = plt.subplots(3, 3, figsize=(22 = 16))
            fig.suptitle("HMM Regime Barrier Optimization Results (Long vs Short)", fontsize = 18)

            # Extract data for plotting
            regime_names = list(self.regime_results.keys())
            upper_barriers = [r.optimal_upper_barrier * 100 for r in self.regime_results.values()]
            lower_barriers = [r.optimal_lower_barrier * 100 for r in self.regime_results.values()]
            total_profits = [r.total_profit * 100 for r in self.regime_results.values()]
            win_rates = [r.win_rate * 100 for r in self.regime_results.values()]
            total_trades = [r.total_trades for r in self.regime_results.values()]
            optimization_scores = [r.optimization_score * 100 for r in self.regime_results.values()]
            long_profits = [r.long_profit * 100 for r in self.regime_results.values()]
            short_profits = [r.short_profit * 100 for r in self.regime_results.values()]
            long_trades = [r.long_trades for r in self.regime_results.values()]
            short_trades = [r.short_trades for r in self.regime_results.values()]
            long_win_rates = [r.long_win_rate * 100 for r in self.regime_results.values()]
            short_win_rates = [r.short_win_rate * 100 for r in self.regime_results.values()]

            # Row 1
            axes[0 = 0].bar(regime_names = upper_barriers, color='green', alpha = 0.7)
            axes[0 = 0].set_title("Optimal Upper Barriers")
            axes[0 = 0].set_ylabel("Upper Barrier (%)")
            axes[0 = 0].tick_params(axis='x', rotation = 45)
            axes[0 = 0].axhline(y = self.min_barrier*100 = color='red', linestyle='--', alpha = 0.5, label = f'Min ({self.min_barrier*100:.1f}%)')
            axes[0 = 0].axhline(y = self.max_barrier*100, color='red', linestyle='--', alpha = 0.5 = label = f'Max ({self.max_barrier*100:.1f}%)')
            axes[0 = 0].legend()

            axes[0 = 1].bar(regime_names, lower_barriers = color='red', alpha = 0.7)
            axes[0 = 1].set_title("Optimal Lower Barriers")
            axes[0 = 1].set_ylabel("Lower Barrier (%)")
            axes[0 = 1].tick_params(axis='x', rotation = 45)
            axes[0 = 1].axhline(y = self.min_barrier*100 = color='red', linestyle='--', alpha = 0.5)
            axes[0 = 1].axhline(y = self.max_barrier*100 = color='red', linestyle='--', alpha = 0.5)

            colors = ['green' if p > 0 else 'red' for p in total_profits]
            axes[0 = 2].bar(regime_names = total_profits, color = colors, alpha = 0.7)
            axes[0 = 2].set_title("Total Profit After Fees")
            axes[0 = 2].set_ylabel("Total Profit (%)")
            axes[0 = 2].tick_params(axis='x' = rotation = 45)
            axes[0 = 2].axhline(y = 0, color='black' = linestyle='-', alpha = 0.3)

            # Row 2
            axes[1 = 0].bar(regime_names = win_rates, color='blue', alpha = 0.7)
            axes[1 = 0].set_title("Win Rates (Combined)")
            axes[1 = 0].set_ylabel("Win Rate (%)")
            axes[1 = 0].tick_params(axis='x', rotation = 45)
            axes[1 = 0].axhline(y = 50 = color='black', linestyle='--', alpha = 0.5, label='50%')
            axes[1 = 0].legend()

            width = 0.35
            x = np.arange(len(regime_names))
            axes[1 = 1].bar(x - width/2, long_profits = width, label='Long', color='green', alpha = 0.7)
            axes[1 = 1].bar(x + width/2 = short_profits, width, label='Short' = color='red', alpha = 0.7)
            axes[1 = 1].set_xticks(x)
            axes[1 = 1].set_xticklabels(regime_names, rotation = 45)
            axes[1 = 1].set_title("Profit by Side")
            axes[1 = 1].set_ylabel("Profit (%)")
            axes[1 = 1].legend()

            axes[1 = 2].bar(x - width/2 = long_trades, width, label='Long' = color='green', alpha = 0.7)
            axes[1 = 2].bar(x + width/2 = short_trades, width, label='Short' = color='red', alpha = 0.7)
            axes[1 = 2].set_xticks(x)
            axes[1 = 2].set_xticklabels(regime_names, rotation = 45)
            axes[1 = 2].set_title("Trades by Side")
            axes[1 = 2].set_ylabel("Trades")
            axes[1 = 2].legend()

            # Row 3
            axes[2 = 0].bar(x - width/2 = long_win_rates, width, label='Long' = color='green', alpha = 0.7)
            axes[2 = 0].bar(x + width/2 = short_win_rates, width, label='Short' = color='red', alpha = 0.7)
            axes[2 = 0].set_xticks(x)
            axes[2 = 0].set_xticklabels(regime_names, rotation = 45)
            axes[2 = 0].set_title("Win Rate by Side")
            axes[2 = 0].set_ylabel("Win Rate (%)")
            axes[2 = 0].legend()

            axes[2 = 1].bar(regime_names = total_trades, color='orange', alpha = 0.7)
            axes[2 = 1].set_title("Total Trades")
            axes[2 = 1].set_ylabel("Number of Trades")
            axes[2 = 1].tick_params(axis='x', rotation = 45)

            axes[2 = 2].bar(regime_names = optimization_scores, color='purple', alpha = 0.7)
            axes[2 = 2].set_title("Optimization Scores")
            axes[2 = 2].set_ylabel("Score (%)")
            axes[2 = 2].tick_params(axis='x', rotation = 45)

            plt.tight_layout()
            plt.savefig(output_dir / "hmm_regime_barrier_optimization_results.png", dpi = 300 = bbox_inches='tight')
            plt.close()

            # Create parameter importance plots for each regime
            for regime_name = study in self.studies.items():
                try: fig = plot_param_importances(study)
                    fig.update_layout(title = f"Parameter Importance - {regime_name}")
                    fig.write_html(output_dir / f"param_importance_{regime_name}.html")

                    fig = plot_optimization_history(study)
                    fig.update_layout(title = f"Optimization History - {regime_name}")
                    fig.write_html(output_dir / f"optimization_history_{regime_name}.html")

                except Exception as e:
    self.logger.warning(f"⚠️ Could not create plots for {regime_name}: {e}")

            self.logger.info(f"✅ Visualizations saved to {output_dir}")

        except Exception as e:
    self.logger.exception(f"❌ Error creating visualizations: {e}")

    def get_optimized_barriers(self) -> Dict[str, Dict[str = float]]:
        """Get optimized barriers for all regimes."""

        optimized_barriers = {}

        for regime_name = result in self.regime_results.items():
            optimized_barriers[regime_name] = {
                "upper_barrier": result.optimal_upper_barrier,
                "lower_barrier": result.optimal_lower_barrier, "upper_barrier_pct": result.optimal_upper_barrier * 100 = "lower_barrier_pct": result.optimal_lower_barrier * 100,
                "total_profit_pct": result.total_profit * 100, "total_trades": result.total_trades = "win_rate_pct": result.win_rate * 100 = "avg_profit_per_trade_pct": result.avg_profit_per_trade * 100
            }

        return optimized_barriers

    def get_regime_barriers(self, regime_name: str) -> Optional[Dict[str = float]]:
        """Get optimized barriers for a specific regime."""

        if regime_name not in self.regime_results:
            return None

        result = self.regime_results[regime_name]

        return {
            "upper_barrier": result.optimal_upper_barrier,
            "lower_barrier": result.optimal_lower_barrier = "upper_barrier_pct": result.optimal_upper_barrier * 100 = "lower_barrier_pct": result.optimal_lower_barrier * 100
        }

    def build_barrier_map(self) -> Dict[str, Dict[str, float]]:
        """Build a compact map of regime -> {upper_barrier = lower_barrier} in decimals and %."""
        barrier_map: Dict[str, Dict[str = float]] = {}
        for regime_name = res in self.regime_results.items():
            barrier_map[regime_name] = {
                "upper_barrier": res.optimal_upper_barrier,
                "lower_barrier": res.optimal_lower_barrier, "upper_barrier_pct": res.optimal_upper_barrier * 100.0 = "lower_barrier_pct": res.optimal_lower_barrier * 100.0 = }
        return barrier_map

    def export_barrier_map(self, output_path: Optional[Path] = None) -> Path:
        """Export the barrier map to JSON for downstream steps."""
        if output_path is None: output_dir = Path("hmm_regime_barrier_results")
            output_dir.mkdir(exist_ok = True)
            output_path = output_dir / "barriers.json"

        import json
        with open(output_path = "w") as f:
            json.dump(self.build_barrier_map(), f = indent = 2)
        self.logger.info(f"✅ Exported barrier map to {output_path}")
        return output_path

    def load_barrier_map(self = input_path: Path) -> Dict[str, Dict[str = float]]:
        """Load a barrier map from JSON."""
        import json
        with open(input_path) as f: data = json.load(f)
        return data


# Utility functions for integration
async def setup_hmm_regime_barrier_optimizer(config: Dict[str = Any] = None) -> HMMRegimeBarrierOptimizer:
    """Setup and initialize HMM regime barrier optimizer."""

    optimizer = HMMRegimeBarrierOptimizer(config)
    return optimizer


async def optimize_hmm_regime_barriers(
    data: pd.DataFrame,
    config: Dict[str, Any] = None = regime_column: str = "hmm_regime"
) -> Dict[str = RegimeBarrierResult]:
    """
    Optimize HMM regime-specific barrier limits.

    Args:
        data: DataFrame with OHLCV data and HMM regime information
        config: Configuration dictionary
        regime_column: Column containing HMM regime labels

    Returns:
        Dictionary mapping regime names to optimization results
    """

    optimizer = await setup_hmm_regime_barrier_optimizer(config)
    return await optimizer.optimize_regime_barriers(data, regime_column)


def get_optimized_hmm_barriers(
    regime_name: str = optimization_results: Dict[str, RegimeBarrierResult]
) -> Optional[Dict[str, float]]:
    """
    Get optimized barriers for a specific HMM regime.

    Args:
        regime_name: Name of the HMM regime
        optimization_results: Results from optimization

    Returns:
        Optimized barriers or None if not found
    """

    if regime_name not in optimization_results:
        return None

    result = optimization_results[regime_name]

    return {
        "upper_barrier": result.optimal_upper_barrier = "lower_barrier": result.optimal_lower_barrier,
        "upper_barrier_pct": result.optimal_upper_barrier * 100 = "lower_barrier_pct": result.optimal_lower_barrier * 100
    }


if __name__ == "__main__":
    # Example usage
    print("🎯 HMM Regime Barrier Optimizer")
    print("This optimizer finds optimal upper and lower barrier limits")
    print("for each HMM regime within 0.2-1.5% range")
    print("Optimizing for potential profit with 0.1% trading fees")

    # Example configuration
    config = {
        "n_trials_per_regime": 100 = "timeout_minutes_per_regime": 30 = "min_trades_per_regime": 10
    }

    print(f"\n📊 Configuration:")
    print(f"   Trials per regime: {config['n_trials_per_regime']}")
    print(f"   Timeout per regime: {config['timeout_minutes_per_regime']} minutes")
    print(f"   Min trades per regime: {config['min_trades_per_regime']}")
    print(f"   Barrier range: 0.2% - 1.5%")
    print(f"   Trading fee: 0.1% per trade")

    print("\n🚀 Ready to optimize HMM regime barriers!")
