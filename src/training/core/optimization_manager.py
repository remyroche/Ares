"""Optimization manager extracted from enhanced training manager.

This module handles hyperparameter optimization, backtesting,
and performance evaluation.
"""

from typing import Any, Dict, List, Optional, Callable
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from datetime import datetime

from src.utils.logger import system_logger
from src.core.decorators import handles_errors


class OptimizationManager:
    """Manages optimization processes for the training pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize optimization manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("OptimizationManager")
        
        # Optimization configuration
        self.opt_config = config.get("optimization", {})
        self.n_trials = self.opt_config.get("n_trials", 100)
        self.enable_early_stopping = self.opt_config.get("enable_early_stopping", True)
        self.enable_parallelization = self.opt_config.get("enable_parallelization", True)
        self.max_workers = self.opt_config.get("max_workers", 4)
        
        # Caching
        self.cache_dir = Path("cache/optimization")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.result_cache = {}
        
    @handles_errors(
        exceptions=(Exception,),
        default_return={"best_params": {}, "best_score": -np.inf},
        context="hyperparameter optimization"
    )
    async def optimize_hyperparameters(
        self,
        objective_function: Callable,
        param_space: Dict[str, Any],
        market_data: pd.DataFrame,
        n_trials: Optional[int] = None
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using Bayesian optimization.
        
        Args:
            objective_function: Function to optimize
            param_space: Parameter search space
            market_data: Market data for evaluation
            n_trials: Number of optimization trials
            
        Returns:
            Optimization results
        """
        n_trials = n_trials or self.n_trials
        self.logger.info(f"🔍 Starting hyperparameter optimization ({n_trials} trials)")
        
        # Try to use Optuna if available
        try:
            import optuna
            return await self._optimize_with_optuna(
                objective_function, param_space, market_data, n_trials
            )
        except ImportError:
            self.logger.warning("Optuna not available, using grid search")
            return await self._optimize_with_grid_search(
                objective_function, param_space, market_data, n_trials
            )
    
    async def _optimize_with_optuna(
        self,
        objective_function: Callable,
        param_space: Dict[str, Any],
        market_data: pd.DataFrame,
        n_trials: int
    ) -> Dict[str, Any]:
        """Optimize using Optuna."""
        import optuna
        
        # Create study
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner() if self.enable_early_stopping else None
        )
        
        # Define objective wrapper
        def optuna_objective(trial):
            # Sample parameters
            params = {}
            for param_name, param_config in param_space.items():
                if param_config["type"] == "int":
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config["low"],
                        param_config["high"]
                    )
                elif param_config["type"] == "float":
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config["low"],
                        param_config["high"]
                    )
                elif param_config["type"] == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config["choices"]
                    )
            
            # Evaluate
            score = objective_function(params, market_data)
            
            # Report for pruning
            if self.enable_early_stopping:
                trial.report(score, 0)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return score
        
        # Optimize
        study.optimize(
            optuna_objective,
            n_trials=n_trials,
            n_jobs=self.max_workers if self.enable_parallelization else 1
        )
        
        # Get results
        best_trial = study.best_trial
        
        return {
            "best_params": best_trial.params,
            "best_score": best_trial.value,
            "n_trials": len(study.trials),
            "optimization_history": [
                {"params": t.params, "score": t.value}
                for t in study.trials if t.value is not None
            ]
        }
    
    async def _optimize_with_grid_search(
        self,
        objective_function: Callable,
        param_space: Dict[str, Any],
        market_data: pd.DataFrame,
        n_trials: int
    ) -> Dict[str, Any]:
        """Fallback grid search optimization."""
        # Generate parameter grid
        param_grid = self._generate_param_grid(param_space, n_trials)
        
        best_params = None
        best_score = -np.inf
        history = []
        
        # Evaluate each parameter set
        for i, params in enumerate(param_grid):
            try:
                score = objective_function(params, market_data)
                history.append({"params": params, "score": score})
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    self.logger.info(f"📈 New best score: {score:.4f} at trial {i+1}")
                    
            except Exception as e:
                self.logger.warning(f"Trial {i+1} failed: {e}")
        
        return {
            "best_params": best_params,
            "best_score": best_score,
            "n_trials": len(history),
            "optimization_history": history
        }
    
    def _generate_param_grid(
        self,
        param_space: Dict[str, Any],
        max_combinations: int
    ) -> List[Dict[str, Any]]:
        """Generate parameter grid for search."""
        import itertools
        
        # Create parameter lists
        param_lists = {}
        for param_name, param_config in param_space.items():
            if param_config["type"] == "int":
                param_lists[param_name] = np.linspace(
                    param_config["low"],
                    param_config["high"],
                    min(5, param_config.get("n_samples", 5)),
                    dtype=int
                ).tolist()
            elif param_config["type"] == "float":
                param_lists[param_name] = np.linspace(
                    param_config["low"],
                    param_config["high"],
                    min(5, param_config.get("n_samples", 5))
                ).tolist()
            elif param_config["type"] == "categorical":
                param_lists[param_name] = param_config["choices"]
        
        # Generate combinations
        all_combinations = list(itertools.product(*param_lists.values()))
        param_names = list(param_lists.keys())
        
        # Create parameter dictionaries
        param_grid = [
            dict(zip(param_names, combo))
            for combo in all_combinations
        ]
        
        # Limit to max_combinations
        if len(param_grid) > max_combinations:
            # Random sample
            indices = np.random.choice(len(param_grid), max_combinations, replace=False)
            param_grid = [param_grid[i] for i in indices]
        
        return param_grid
    
    @handles_errors(
        exceptions=(Exception,),
        default_return={"sharpe": 0, "returns": 0, "max_drawdown": 0},
        context="backtesting"
    )
    async def backtest_strategy(
        self,
        strategy_params: Dict[str, Any],
        market_data: pd.DataFrame,
        initial_capital: float = 10000
    ) -> Dict[str, float]:
        """Backtest a trading strategy.
        
        Args:
            strategy_params: Strategy parameters
            market_data: Market data
            initial_capital: Starting capital
            
        Returns:
            Backtest metrics
        """
        # Generate cache key
        cache_key = joblib.hash((strategy_params, len(market_data), initial_capital))
        
        # Check cache
        if cache_key in self.result_cache:
            return self.result_cache[cache_key]
        
        # Simple backtest implementation
        returns = market_data['close'].pct_change().fillna(0)
        
        # Generate signals based on strategy params
        # This is a placeholder - implement actual strategy logic
        signals = self._generate_signals(market_data, strategy_params)
        
        # Calculate portfolio returns
        portfolio_returns = returns * signals.shift(1)
        
        # Calculate metrics
        total_return = (1 + portfolio_returns).prod() - 1
        sharpe_ratio = portfolio_returns.mean() / (portfolio_returns.std() + 1e-10) * np.sqrt(252)
        
        # Calculate drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        results = {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": (portfolio_returns > 0).mean(),
            "avg_return": portfolio_returns.mean(),
            "volatility": portfolio_returns.std()
        }
        
        # Cache results
        self.result_cache[cache_key] = results
        
        return results
    
    def _generate_signals(
        self,
        market_data: pd.DataFrame,
        strategy_params: Dict[str, Any]
    ) -> pd.Series:
        """Generate trading signals based on strategy parameters."""
        # Placeholder signal generation
        # In practice, this would use the actual strategy logic
        
        # Example: Simple moving average crossover
        fast_period = strategy_params.get("fast_period", 10)
        slow_period = strategy_params.get("slow_period", 30)
        
        fast_ma = market_data['close'].rolling(fast_period).mean()
        slow_ma = market_data['close'].rolling(slow_period).mean()
        
        signals = pd.Series(0, index=market_data.index)
        signals[fast_ma > slow_ma] = 1
        signals[fast_ma < slow_ma] = -1
        
        return signals
    
    async def optimize_portfolio_weights(
        self,
        returns_matrix: pd.DataFrame,
        optimization_method: str = "sharpe"
    ) -> Dict[str, float]:
        """Optimize portfolio weights.
        
        Args:
            returns_matrix: DataFrame of asset returns
            optimization_method: Method to use (sharpe, min_variance, etc.)
            
        Returns:
            Optimal weights
        """
        n_assets = len(returns_matrix.columns)
        
        if optimization_method == "equal_weight":
            weights = np.ones(n_assets) / n_assets
        
        elif optimization_method == "sharpe":
            # Maximum Sharpe ratio
            from scipy.optimize import minimize
            
            def neg_sharpe(weights):
                portfolio_return = np.sum(returns_matrix.mean() * weights)
                portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns_matrix.cov(), weights)))
                return -portfolio_return / (portfolio_std + 1e-10)
            
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for _ in range(n_assets))
            initial_weights = np.ones(n_assets) / n_assets
            
            result = minimize(
                neg_sharpe,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            weights = result.x
        
        elif optimization_method == "min_variance":
            # Minimum variance
            from scipy.optimize import minimize
            
            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(returns_matrix.cov(), weights))
            
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for _ in range(n_assets))
            initial_weights = np.ones(n_assets) / n_assets
            
            result = minimize(
                portfolio_variance,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            weights = result.x
        
        else:
            # Default to equal weight
            weights = np.ones(n_assets) / n_assets
        
        # Create weight dictionary
        weight_dict = {
            asset: weight 
            for asset, weight in zip(returns_matrix.columns, weights)
        }
        
        return weight_dict
    
    def save_optimization_results(
        self,
        results: Dict[str, Any],
        filename: str
    ) -> None:
        """Save optimization results to disk.
        
        Args:
            results: Optimization results
            filename: Output filename
        """
        output_path = self.cache_dir / f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        joblib.dump(results, output_path)
        self.logger.info(f"💾 Saved optimization results to {output_path}")
    
    def load_optimization_results(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load optimization results from disk.
        
        Args:
            filename: Input filename
            
        Returns:
            Optimization results or None
        """
        try:
            results = joblib.load(self.cache_dir / filename)
            self.logger.info(f"📂 Loaded optimization results from {filename}")
            return results
        except Exception as e:
            self.logger.error(f"Failed to load optimization results: {e}")
            return None