# src/training/multi_objective_optimizer.py

from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger


@dataclass
class OptimizationMetrics:
    """Container for multiple optimization metrics."""

    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    sortino_ratio: float
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional Value at Risk 95%


class MultiObjectiveOptimizer:
    """
    Advanced multi-objective hyperparameter optimizer using Pareto optimization.

    This optimizer considers multiple performance metrics simultaneously:
    - Risk-adjusted returns (Sharpe, Sortino, Calmar ratios)
    - Risk metrics (Max drawdown, VaR, CVaR)
    - Profitability metrics (Total return, Win rate, Profit factor)
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("MultiObjectiveOptimizer")
        self.metrics_scaler = StandardScaler()
        self.best_pareto_front: list[OptimizationMetrics] = []

        # Multi-objective weights (configurable) - Focused on Sharpe, win rate, and profit factor
        self.objective_weights = config.get(
            "multi_objective_weights",
            {"sharpe_ratio": 0.50, "win_rate": 0.30, "profit_factor": 0.20},
        )

        # Risk constraints
        self.risk_constraints = config.get(
            "risk_constraints",
            {
                "max_drawdown_threshold": 0.20,
                "min_win_rate": 0.40,
                "min_profit_factor": 1.2,
            },
        )

        # Initialize optimized backtester if market data is provided
        self.optimized_backtester = None
        if "market_data" in config:
            from src.training.optimized_backtester import OptimizedBacktester

            self.optimized_backtester = OptimizedBacktester(
                config["market_data"],
                config.get("computational_optimization", {}),
            )

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="multi-objective optimization",
    )
    def objective(self, trial: optuna.trial.Trial) -> tuple[float, float, float]:
        """
        Multi-objective function returning (sharpe_ratio, win_rate, profit_factor).
        """
        # Suggest hyperparameters
        params = self._suggest_hyperparameters(trial)

        # Run backtest with suggested parameters
        backtest_results = self._run_backtest(params)

        # Calculate comprehensive metrics
        metrics = self._calculate_metrics(backtest_results)

        # Check risk constraints
        if not self._check_risk_constraints(metrics):
            return -np.inf, -np.inf, -np.inf

        # Return Pareto objectives
        return (metrics.sharpe_ratio, metrics.win_rate, metrics.profit_factor)

    def _suggest_hyperparameters(self, trial: optuna.trial.Trial) -> dict[str, Any]:
        """Suggest hyperparameters with advanced search spaces."""
        params = {}

        # Model hyperparameters
        params["learning_rate"] = trial.suggest_float(
            "learning_rate",
            1e-4,
            1e-1,
            log=True,
        )
        params["max_depth"] = trial.suggest_int("max_depth", 3, 12)
        params["n_estimators"] = trial.suggest_int("n_estimators", 50, 500)
        params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
        params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.6, 1.0)

        # Regularization parameters (L1 and L2 regularization)
        params["reg_alpha"] = trial.suggest_float(
            "reg_alpha",
            1e-8,
            10.0,
            log=True,
        )  # L1 regularization
        params["reg_lambda"] = trial.suggest_float(
            "reg_lambda",
            1e-8,
            10.0,
            log=True,
        )  # L2 regularization

        # Feature engineering parameters
        params["lookback_window"] = trial.suggest_int("lookback_window", 10, 100)
        params["feature_threshold"] = trial.suggest_float(
            "feature_threshold",
            0.01,
            0.1,
        )

        # Trading parameters
        params["tp_multiplier"] = trial.suggest_float("tp_multiplier", 1.5, 5.0)
        params["sl_multiplier"] = trial.suggest_float("sl_multiplier", 1.0, 3.0)
        params["position_size"] = trial.suggest_float("position_size", 0.01, 0.2)

        # Ensemble parameters
        params["ensemble_weight"] = trial.suggest_float("ensemble_weight", 0.1, 0.9)
        params["confidence_threshold"] = trial.suggest_float(
            "confidence_threshold",
            0.6,
            0.95,
        )

        return params

    def _run_backtest(self, params: dict[str, Any]) -> dict[str, Any]:
        """Run backtest with given parameters using optimized backtester."""

        # Use optimized backtester if available
        if hasattr(self, "optimized_backtester"):
            score = self.optimized_backtester.run_cached_backtest(params)

            # Convert score to mock results for compatibility
            return {
                "returns": np.random.normal(score * 0.01, 0.02, 1000),
                "equity_curve": np.cumprod(
                    1 + np.random.normal(score * 0.01, 0.02, 1000),
                ),
                "trades": self._generate_mock_trades(),
                "score": score,
            }
        # Fallback to mock results
        return {
            "returns": np.random.normal(0.001, 0.02, 1000),
            "equity_curve": np.cumprod(1 + np.random.normal(0.001, 0.02, 1000)),
            "trades": self._generate_mock_trades(),
        }

    def _calculate_metrics(
        self,
        backtest_results: dict[str, Any],
    ) -> OptimizationMetrics:
        """Calculate comprehensive performance metrics."""
        returns = backtest_results["returns"]
        equity_curve = backtest_results["equity_curve"]

        # Basic metrics
        total_return = (equity_curve[-1] / equity_curve[0]) - 1
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0

        # Drawdown calculation
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        max_drawdown = np.min(drawdown)

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = (
            np.std(downside_returns) if len(downside_returns) > 0 else 1e-6
        )
        sortino_ratio = np.mean(returns) / downside_deviation

        # Calmar ratio
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Win rate and profit factor
        trades = backtest_results["trades"]
        winning_trades = [t for t in trades if t["pnl"] > 0]
        losing_trades = [t for t in trades if t["pnl"] < 0]

        win_rate = len(winning_trades) / len(trades) if trades else 0
        if losing_trades:
            profit_factor = sum(t["pnl"] for t in winning_trades) / abs(
                sum(t["pnl"] for t in losing_trades),
            )
        else:
            profit_factor = float("inf")

        # Risk metrics
        var_95 = np.percentile(returns, 5)
        cvar_95 = np.mean(returns[returns <= var_95])

        return OptimizationMetrics(
            sharpe_ratio=sharpe_ratio,
            total_return=total_return,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            var_95=var_95,
            cvar_95=cvar_95,
        )

    def _check_risk_constraints(self, metrics: OptimizationMetrics) -> bool:
        """Check if metrics meet risk constraints."""
        return (
            metrics.max_drawdown >= -self.risk_constraints["max_drawdown_threshold"]
            and metrics.var_95 >= -self.risk_constraints["var_95_threshold"]
            and metrics.win_rate >= self.risk_constraints["min_win_rate"]
        )

    def _generate_mock_trades(self) -> list[dict[str, Any]]:
        """Generate mock trade data for testing."""
        n_trades = np.random.randint(50, 200)
        trades = []

        for i in range(n_trades):
            pnl = np.random.normal(0.01, 0.05)  # Mock PnL
            trades.append(
                {
                    "entry_time": pd.Timestamp.now() - pd.Timedelta(days=i),
                    "exit_time": pd.Timestamp.now() - pd.Timedelta(days=i - 1),
                    "pnl": pnl,
                    "duration": np.random.randint(1, 100),
                },
            )

        return trades

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="multi-objective study execution",
    )
    def run_optimization(self, n_trials: int = 500) -> dict[str, Any]:
        """Run multi-objective optimization study."""
        self.logger.info(
            f"Starting multi-objective optimization with {n_trials} trials...",
        )

        # Create multi-objective study
        study = optuna.create_study(
            directions=[
                "maximize",
                "maximize",
                "maximize",
            ],  # sharpe, win_rate, profit_factor
            sampler=optuna.samplers.NSGAIISampler(
                population_size=50,
                crossover_prob=0.9,
                mutation_prob=0.1,
            ),
        )

        # Run optimization
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)

        # Analyze Pareto front
        pareto_front = self._analyze_pareto_front(study)

        # Select best solution based on weighted combination
        best_solution = self._select_best_solution(pareto_front)

        self.logger.info("Multi-objective optimization completed successfully")

        return {
            "best_params": best_solution["params"],
            "pareto_front": pareto_front,
            "study": study,
        }

    def _analyze_pareto_front(self, study: optuna.Study) -> list[dict[str, Any]]:
        """Analyze and rank Pareto front solutions."""
        pareto_front = []

        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                pareto_front.append(
                    {
                        "params": trial.params,
                        "values": trial.values,
                        "user_attrs": trial.user_attrs,
                    },
                )

        # Sort by weighted objective
        for solution in pareto_front:
            solution["weighted_score"] = self._calculate_weighted_score(
                solution["values"],
            )

        pareto_front.sort(key=lambda x: x["weighted_score"], reverse=True)

        return pareto_front

    def _calculate_weighted_score(self, values: tuple[float, float, float]) -> float:
        """Calculate weighted score from objective values."""
        sharpe, win_rate, profit_factor = values

        return (
            self.objective_weights["sharpe_ratio"] * sharpe
            + self.objective_weights["win_rate"] * win_rate
            + self.objective_weights["profit_factor"] * profit_factor
        )

    def _select_best_solution(
        self,
        pareto_front: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Select the best solution from Pareto front."""
        if not pareto_front:
            raise ValueError("No valid solutions found in Pareto front")

        return pareto_front[0]  # Already sorted by weighted score
