#!/usr/bin/env python3
"""
Per-HMM Regime Triple Barrier Thresholds and TPSL Parameters Optimization

This module provides comprehensive optimization of triple barrier thresholds and
Take Profit/Stop Loss (TPSL) parameters for each HMM regime identified by the
HMM regime discovery system.

The optimizer uses:
1. HMM regime identification from step3_hmm_regime_discovery
2. Regime-specific triple barrier parameter optimization
3. Regime-specific TPSL parameter optimization
4. Cross-validation and backtesting for parameter validation
5. Dynamic parameter adjustment based on regime characteristics

Key Features:
- Per-regime triple barrier threshold optimization
- Per-regime TPSL parameter optimization
- Regime transition handling
- Real-time parameter adjustment
- Comprehensive backtesting validation
- Risk-adjusted performance metrics
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import CONFIG
from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.logger import system_logger
from src.utils.hmm_composite_manager import HMMCompositeManager
from src.training.steps.step4_analyst_labeling_feature_engineering_components.optimized_triple_barrier_labeling import (
    OptimizedTripleBarrierLabeling,
)
from src.utils.warning_symbols import (
    error,
    failed,
    initialization_error,
    warning,
    success,
)


class PerHMMRegimeTPSLOptimizer:
    """
    Comprehensive per-HMM regime triple barrier thresholds and TPSL parameters optimizer.
    
    This optimizer provides regime-specific optimization of:
    1. Triple barrier thresholds (profit take, stop loss, time barrier)
    2. TPSL parameters (take profit %, stop loss %, risk-reward ratios)
    3. Regime-specific entry/exit conditions
    4. Dynamic parameter adjustment based on regime characteristics
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the per-HMM regime TPSL optimizer.
        
        Args:
            config: Configuration dictionary containing optimization parameters
        """
        self.config = config
        self.logger = system_logger.getChild("PerHMMRegimeTPSLOptimizer")
        
        # HMM regime management
        self.hmm_manager = HMMCompositeManager()
        
        # Optimization configuration
        self.optimization_config = config.get("per_hmm_regime_tpsl_optimizer", {})
        self.n_trials = self.optimization_config.get("n_trials", 200)
        self.min_trades_per_regime = self.optimization_config.get("min_trades_per_regime", 30)
        self.cv_folds = self.optimization_config.get("cv_folds", 5)
        self.optimization_metric = self.optimization_config.get("optimization_metric", "sharpe_ratio")
        
        # Regime-specific parameter bounds
        self.regime_parameter_bounds = {
            # Triple barrier parameters
            "triple_barrier": {
                "profit_take_multiplier": (0.001, 0.01),  # 0.1% to 1%
                "stop_loss_multiplier": (0.0005, 0.005),  # 0.05% to 0.5%
                "time_barrier_minutes": (15, 120),  # 15 minutes to 2 hours
                "max_lookahead": (50, 200),  # 50 to 200 bars
            },
            # TPSL parameters
            "tpsl": {
                "target_pct": (0.002, 0.02),  # 0.2% to 2%
                "stop_pct": (0.001, 0.01),  # 0.1% to 1%
                "risk_reward_ratio": (1.5, 4.0),  # 1.5:1 to 4:1
                "position_sizing_pct": (0.01, 0.05),  # 1% to 5% of capital
            },
            # Regime-specific adjustments
            "regime_adjustments": {
                "volatility_multiplier": (0.5, 2.0),  # Volatility-based scaling
                "momentum_multiplier": (0.8, 1.5),  # Momentum-based scaling
                "regime_confidence_threshold": (0.3, 0.8),  # Minimum confidence for regime
            }
        }
        
        # Regime-specific default parameters (will be optimized per regime)
        self.regime_defaults = {
            "hmm_cluster_0": {
                "name": "Low Volatility Sideways",
                "triple_barrier": {"profit_take_multiplier": 0.003, "stop_loss_multiplier": 0.002, "time_barrier_minutes": 45},
                "tpsl": {"target_pct": 0.005, "stop_pct": 0.003, "risk_reward_ratio": 1.67},
                "characteristics": {"volatility": "low", "trend": "sideways", "frequency": "high"}
            },
            "hmm_cluster_1": {
                "name": "Moderate Volatility Trending",
                "triple_barrier": {"profit_take_multiplier": 0.005, "stop_loss_multiplier": 0.003, "time_barrier_minutes": 60},
                "tpsl": {"target_pct": 0.008, "stop_pct": 0.004, "risk_reward_ratio": 2.0},
                "characteristics": {"volatility": "moderate", "trend": "trending", "frequency": "medium"}
            },
            "hmm_cluster_2": {
                "name": "High Volatility Breakout",
                "triple_barrier": {"profit_take_multiplier": 0.008, "stop_loss_multiplier": 0.004, "time_barrier_minutes": 30},
                "tpsl": {"target_pct": 0.012, "stop_pct": 0.006, "risk_reward_ratio": 2.0},
                "characteristics": {"volatility": "high", "trend": "breakout", "frequency": "low"}
            },
            "hmm_cluster_3": {
                "name": "Extreme Volatility Crisis",
                "triple_barrier": {"profit_take_multiplier": 0.015, "stop_loss_multiplier": 0.008, "time_barrier_minutes": 20},
                "tpsl": {"target_pct": 0.02, "stop_pct": 0.01, "risk_reward_ratio": 2.0},
                "characteristics": {"volatility": "extreme", "trend": "crisis", "frequency": "very_low"}
            },
            "hmm_cluster_4": {
                "name": "Low Volatility Trending",
                "triple_barrier": {"profit_take_multiplier": 0.004, "stop_loss_multiplier": 0.002, "time_barrier_minutes": 90},
                "tpsl": {"target_pct": 0.006, "stop_pct": 0.003, "risk_reward_ratio": 2.0},
                "characteristics": {"volatility": "low", "trend": "trending", "frequency": "medium"}
            },
            "hmm_cluster_5": {
                "name": "Moderate Volatility Sideways",
                "triple_barrier": {"profit_take_multiplier": 0.004, "stop_loss_multiplier": 0.003, "time_barrier_minutes": 60},
                "tpsl": {"target_pct": 0.007, "stop_pct": 0.004, "risk_reward_ratio": 1.75},
                "characteristics": {"volatility": "moderate", "trend": "sideways", "frequency": "high"}
            },
            "hmm_cluster_6": {
                "name": "High Volatility Sideways",
                "triple_barrier": {"profit_take_multiplier": 0.006, "stop_loss_multiplier": 0.004, "time_barrier_minutes": 45},
                "tpsl": {"target_pct": 0.01, "stop_pct": 0.005, "risk_reward_ratio": 2.0},
                "characteristics": {"volatility": "high", "trend": "sideways", "frequency": "medium"}
            },
            "hmm_cluster_7": {
                "name": "Moderate Volatility Breakout",
                "triple_barrier": {"profit_take_multiplier": 0.006, "stop_loss_multiplier": 0.004, "time_barrier_minutes": 40},
                "tpsl": {"target_pct": 0.009, "stop_pct": 0.005, "risk_reward_ratio": 1.8},
                "characteristics": {"volatility": "moderate", "trend": "breakout", "frequency": "low"}
            }
        }
        
        # Model storage
        self.model_dir = os.path.join(CONFIG["CHECKPOINT_DIR"], "per_hmm_regime_tpsl_models")
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Optimization results cache
        self.optimization_results: Dict[str, Dict[str, Any]] = {}
        self.last_optimization_time: Optional[datetime] = None
        self.regime_statistics: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.regime_transition_history: List[Dict[str, Any]] = []

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid per-HMM regime TPSL optimization configuration"),
            AttributeError: (False, "Missing required optimization parameters"),
        },
        default_return=False,
        context="per-HMM regime TPSL optimizer initialization",
    )
    async def initialize(self) -> bool:
        """Initialize the per-HMM regime TPSL optimizer.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("🚀 Initializing Per-HMM Regime TPSL Optimizer...")
            
            # Initialize HMM manager
            if not self.hmm_manager:
                self.logger.error("❌ Failed to initialize HMM manager")
                return False
            
            # Load existing optimization results
            await self._load_optimization_results()
            
            # Load regime statistics
            await self._load_regime_statistics()
            
            self.logger.info(f"✅ Per-HMM Regime TPSL Optimizer initialized successfully")
            self.logger.info(f"   - Optimization trials: {self.n_trials}")
            self.logger.info(f"   - CV folds: {self.cv_folds}")
            self.logger.info(f"   - Optimization metric: {self.optimization_metric}")
            self.logger.info(f"   - Min trades per regime: {self.min_trades_per_regime}")
            
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Failed to initialize Per-HMM Regime TPSL Optimizer: {e}")
            return False

    async def _load_optimization_results(self) -> None:
        """Load existing optimization results from disk."""
        try:
            results_file = os.path.join(self.model_dir, "per_hmm_optimization_results.json")
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    self.optimization_results = json.load(f)
                    self.logger.info(f"✅ Loaded {len(self.optimization_results)} regime optimization results")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not load optimization results: {e}")

    async def _save_optimization_results(self) -> None:
        """Save optimization results to disk."""
        try:
            results_file = os.path.join(self.model_dir, "per_hmm_optimization_results.json")
            with open(results_file, 'w') as f:
                json.dump(self.optimization_results, f, indent=2, default=str)
            self.logger.info("✅ Saved per-HMM optimization results")
        except Exception as e:
            self.logger.error(f"❌ Failed to save optimization results: {e}")

    async def _load_regime_statistics(self) -> None:
        """Load regime statistics from disk."""
        try:
            stats_file = os.path.join(self.model_dir, "regime_statistics.json")
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    self.regime_statistics = json.load(f)
                    self.logger.info(f"✅ Loaded regime statistics for {len(self.regime_statistics)} regimes")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not load regime statistics: {e}")

    async def _save_regime_statistics(self) -> None:
        """Save regime statistics to disk."""
        try:
            stats_file = os.path.join(self.model_dir, "regime_statistics.json")
            with open(stats_file, 'w') as f:
                json.dump(self.regime_statistics, f, indent=2, default=str)
            self.logger.info("✅ Saved regime statistics")
        except Exception as e:
            self.logger.error(f"❌ Failed to save regime statistics: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="HMM regime identification",
    )
    async def identify_current_hmm_regime(
        self, 
        current_data: pd.DataFrame,
        exchange: str,
        symbol: str,
        timeframe: str
    ) -> Tuple[str, float, Dict[str, Any]]:
        """Identify the current HMM regime using the HMM composite manager.
        
        Args:
            current_data: Current market OHLCV data
            exchange: Exchange name
            symbol: Symbol name
            timeframe: Timeframe string
            
        Returns:
            Tuple of (regime_label, confidence, additional_info)
        """
        try:
            # Get HMM regime data
            regime_data = self.hmm_manager.get_hmm_composite_clusters(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe
            )
            
            if regime_data is None or regime_data.empty:
                self.logger.warning("⚠️ No HMM regime data available, using default regime")
                return "hmm_cluster_0", 0.5, {"method": "default", "error": "No HMM data"}
            
            # Get the most recent regime
            latest_regime = regime_data.iloc[-1] if len(regime_data) > 0 else None
            
            if latest_regime is None:
                return "hmm_cluster_0", 0.5, {"method": "default", "error": "No recent regime data"}
            
            # Extract regime information
            regime_label = f"hmm_cluster_{latest_regime.get('hmm_composite_cluster_id', 0)}"
            confidence = float(latest_regime.get('hmm_composite_intensity', 0.5))
            
            # Additional regime information
            regime_info = {
                "method": "hmm_composite",
                "regime_id": int(latest_regime.get('hmm_composite_cluster_id', 0)),
                "intensity": confidence,
                "timestamp": latest_regime.name.isoformat() if hasattr(latest_regime.name, 'isoformat') else str(latest_regime.name),
                "regime_characteristics": self.regime_defaults.get(regime_label, {}).get("characteristics", {})
            }
            
            self.logger.info(f"🎯 Identified HMM regime: {regime_label} (confidence: {confidence:.3f})")
            
            return regime_label, confidence, regime_info
            
        except Exception as e:
            self.logger.error(f"❌ Error identifying HMM regime: {e}")
            return "hmm_cluster_0", 0.5, {"method": "fallback", "error": str(e)}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="per-HMM regime optimization",
    )
    async def optimize_regime_parameters(
        self,
        regime: str,
        historical_data: pd.DataFrame,
        current_data: pd.DataFrame,
        force_optimization: bool = False
    ) -> Dict[str, Any]:
        """Optimize triple barrier and TPSL parameters for a specific HMM regime.
        
        Args:
            regime: HMM regime to optimize for
            historical_data: Historical data for optimization
            current_data: Current market data
            force_optimization: Force re-optimization even if cached
            
        Returns:
            Dictionary with optimized parameters
        """
        try:
            self.logger.info(f"🎯 Optimizing parameters for regime: {regime}")
            
            # Check if we have cached results for this regime
            if not force_optimization and regime in self.optimization_results:
                cached_params = self.optimization_results[regime]
                self.logger.info(f"📋 Using cached parameters for {regime}")
                return cached_params
            
            # Get base parameters for this regime
            base_params = self.regime_defaults.get(regime, self.regime_defaults["hmm_cluster_0"])
            
            # Create optimization study
            study = optuna.create_study(
                direction="maximize",
                study_name=f"per_hmm_tpsl_optimization_{regime}",
                sampler=optuna.samplers.TPESampler(seed=42)
            )
            
            # Define objective function
            def objective(trial):
                return self._evaluate_regime_parameters(
                    trial,
                    regime,
                    historical_data,
                    base_params
                )
            
            # Run optimization
            self.logger.info(f"🔄 Running optimization for {regime} with {self.n_trials} trials...")
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
            
            # Get best parameters
            best_params = study.best_params
            best_value = study.best_value
            
            # Combine with base parameters
            optimized_params = {
                **base_params,
                "optimized_triple_barrier": best_params.get("triple_barrier", {}),
                "optimized_tpsl": best_params.get("tpsl", {}),
                "optimized_regime_adjustments": best_params.get("regime_adjustments", {}),
                "optimization_score": best_value,
                "optimization_trials": self.n_trials,
                "optimization_time": datetime.now().isoformat(),
                "study_summary": {
                    "best_value": best_value,
                    "n_trials": len(study.trials),
                    "optimization_history": [trial.value for trial in study.trials if trial.value is not None]
                }
            }
            
            # Cache results
            self.optimization_results[regime] = optimized_params
            await self._save_optimization_results()
            
            self.logger.info(f"✅ Optimized parameters for {regime}: score={best_value:.4f}")
            return optimized_params
            
        except Exception as e:
            self.logger.error(f"❌ Error optimizing parameters for regime {regime}: {e}")
            return self.regime_defaults.get(regime, self.regime_defaults["hmm_cluster_0"])

    def _evaluate_regime_parameters(
        self,
        trial: optuna.Trial,
        regime: str,
        historical_data: pd.DataFrame,
        base_params: Dict[str, Any]
    ) -> float:
        """Evaluate regime parameters using cross-validation and backtesting.
        
        Args:
            trial: Optuna trial object
            regime: Market regime
            historical_data: Historical data for backtesting
            base_params: Base parameters for the regime
            
        Returns:
            float: Optimization score (higher is better)
        """
        try:
            # Suggest triple barrier parameters
            tb_params = {
                "profit_take_multiplier": trial.suggest_float(
                    "tb_profit_take_multiplier",
                    self.regime_parameter_bounds["triple_barrier"]["profit_take_multiplier"][0],
                    self.regime_parameter_bounds["triple_barrier"]["profit_take_multiplier"][1]
                ),
                "stop_loss_multiplier": trial.suggest_float(
                    "tb_stop_loss_multiplier",
                    self.regime_parameter_bounds["triple_barrier"]["stop_loss_multiplier"][0],
                    self.regime_parameter_bounds["triple_barrier"]["stop_loss_multiplier"][1]
                ),
                "time_barrier_minutes": trial.suggest_int(
                    "tb_time_barrier_minutes",
                    self.regime_parameter_bounds["triple_barrier"]["time_barrier_minutes"][0],
                    self.regime_parameter_bounds["triple_barrier"]["time_barrier_minutes"][1]
                ),
                "max_lookahead": trial.suggest_int(
                    "tb_max_lookahead",
                    self.regime_parameter_bounds["triple_barrier"]["max_lookahead"][0],
                    self.regime_parameter_bounds["triple_barrier"]["max_lookahead"][1]
                )
            }
            
            # Suggest TPSL parameters
            tpsl_params = {
                "target_pct": trial.suggest_float(
                    "tpsl_target_pct",
                    self.regime_parameter_bounds["tpsl"]["target_pct"][0],
                    self.regime_parameter_bounds["tpsl"]["target_pct"][1]
                ),
                "stop_pct": trial.suggest_float(
                    "tpsl_stop_pct",
                    self.regime_parameter_bounds["tpsl"]["stop_pct"][0],
                    self.regime_parameter_bounds["tpsl"]["stop_pct"][1]
                ),
                "risk_reward_ratio": trial.suggest_float(
                    "tpsl_risk_reward_ratio",
                    self.regime_parameter_bounds["tpsl"]["risk_reward_ratio"][0],
                    self.regime_parameter_bounds["tpsl"]["risk_reward_ratio"][1]
                ),
                "position_sizing_pct": trial.suggest_float(
                    "tpsl_position_sizing_pct",
                    self.regime_parameter_bounds["tpsl"]["position_sizing_pct"][0],
                    self.regime_parameter_bounds["tpsl"]["position_sizing_pct"][1]
                )
            }
            
            # Suggest regime adjustments
            regime_adjustments = {
                "volatility_multiplier": trial.suggest_float(
                    "regime_volatility_multiplier",
                    self.regime_parameter_bounds["regime_adjustments"]["volatility_multiplier"][0],
                    self.regime_parameter_bounds["regime_adjustments"]["volatility_multiplier"][1]
                ),
                "momentum_multiplier": trial.suggest_float(
                    "regime_momentum_multiplier",
                    self.regime_parameter_bounds["regime_adjustments"]["momentum_multiplier"][0],
                    self.regime_parameter_bounds["regime_adjustments"]["momentum_multiplier"][1]
                ),
                "regime_confidence_threshold": trial.suggest_float(
                    "regime_confidence_threshold",
                    self.regime_parameter_bounds["regime_adjustments"]["regime_confidence_threshold"][0],
                    self.regime_parameter_bounds["regime_adjustments"]["regime_confidence_threshold"][1]
                )
            }
            
            # Validate parameter constraints
            if tpsl_params["target_pct"] <= tpsl_params["stop_pct"]:
                return -1.0
            
            if tb_params["profit_take_multiplier"] <= tb_params["stop_loss_multiplier"]:
                return -1.0
            
            # Run cross-validation
            cv_scores = []
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            
            for train_idx, test_idx in tscv.split(historical_data):
                train_data = historical_data.iloc[train_idx]
                test_data = historical_data.iloc[test_idx]
                
                # Evaluate on test fold
                score = self._evaluate_single_fold(
                    test_data, tb_params, tpsl_params, regime_adjustments, regime
                )
                cv_scores.append(score)
            
            # Return mean CV score
            mean_score = np.mean(cv_scores) if cv_scores else -1.0
            return mean_score
            
        except Exception as e:
            self.logger.error(f"❌ Error in parameter evaluation: {e}")
            return -1.0

    def _evaluate_single_fold(
        self,
        data: pd.DataFrame,
        tb_params: Dict[str, Any],
        tpsl_params: Dict[str, Any],
        regime_adjustments: Dict[str, Any],
        regime: str
    ) -> float:
        """Evaluate parameters on a single fold of data.
        
        Args:
            data: Test data fold
            tb_params: Triple barrier parameters
            tpsl_params: TPSL parameters
            regime_adjustments: Regime adjustment parameters
            regime: Market regime
            
        Returns:
            float: Performance score
        """
        try:
            # Generate triple barrier labels
            tb_labeler = OptimizedTripleBarrierLabeling(
                profit_take_multiplier=tb_params["profit_take_multiplier"],
                stop_loss_multiplier=tb_params["stop_loss_multiplier"],
                time_barrier_minutes=tb_params["time_barrier_minutes"],
                max_lookahead=tb_params["max_lookahead"],
                binary_classification=True
            )
            
            # Apply regime adjustments
            adjusted_tb_params = self._apply_regime_adjustments(tb_params, regime_adjustments, data)
            
            # Generate labels with adjusted parameters
            labels = tb_labeler.generate_labels(data)
            
            if labels is None or labels.empty:
                return -1.0
            
            # Simulate trading with TPSL parameters
            trades = self._simulate_regime_trades(data, labels, tpsl_params, regime_adjustments)
            
            if len(trades) < self.min_trades_per_regime:
                return -1.0
            
            # Calculate performance metrics
            returns = [trade["return"] for trade in trades]
            total_return = sum(returns)
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8)
            win_rate = len([r for r in returns if r > 0]) / len(returns)
            max_drawdown = self._calculate_max_drawdown(returns)
            
            # Combine metrics based on optimization target
            if self.optimization_metric == "sharpe_ratio":
                score = sharpe_ratio
            elif self.optimization_metric == "total_return":
                score = total_return
            elif self.optimization_metric == "win_rate":
                score = win_rate
            elif self.optimization_metric == "calmar_ratio":
                score = total_return / (max_drawdown + 1e-8)
            else:
                # Composite score
                score = (
                    sharpe_ratio * 0.4 +
                    total_return * 0.3 +
                    win_rate * 0.2 +
                    (1 / (max_drawdown + 1e-8)) * 0.1
                )
            
            return score
            
        except Exception as e:
            self.logger.error(f"❌ Error in single fold evaluation: {e}")
            return -1.0

    def _apply_regime_adjustments(
        self,
        params: Dict[str, Any],
        adjustments: Dict[str, Any],
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Apply regime-specific adjustments to parameters.
        
        Args:
            params: Base parameters
            adjustments: Regime adjustments
            data: Market data for calculating adjustments
            
        Returns:
            Dict[str, Any]: Adjusted parameters
        """
        try:
            adjusted_params = params.copy()
            
            # Calculate volatility-based adjustments
            if len(data) > 20:
                volatility = data["close"].pct_change().std()
                volatility_adjustment = adjustments["volatility_multiplier"] * volatility
                
                # Adjust profit take and stop loss based on volatility
                adjusted_params["profit_take_multiplier"] *= (1 + volatility_adjustment)
                adjusted_params["stop_loss_multiplier"] *= (1 + volatility_adjustment)
            
            # Calculate momentum-based adjustments
            if len(data) > 10:
                momentum = (data["close"].iloc[-1] / data["close"].iloc[-10] - 1)
                momentum_adjustment = adjustments["momentum_multiplier"] * abs(momentum)
                
                # Adjust time barrier based on momentum
                adjusted_params["time_barrier_minutes"] = int(
                    adjusted_params["time_barrier_minutes"] * (1 + momentum_adjustment)
                )
            
            return adjusted_params
            
        except Exception as e:
            self.logger.error(f"❌ Error applying regime adjustments: {e}")
            return params

    def _simulate_regime_trades(
        self,
        data: pd.DataFrame,
        labels: pd.Series,
        tpsl_params: Dict[str, Any],
        regime_adjustments: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Simulate trades using regime-specific TPSL parameters.
        
        Args:
            data: Market data
            labels: Triple barrier labels
            tpsl_params: TPSL parameters
            regime_adjustments: Regime adjustments
            
        Returns:
            List[Dict[str, Any]]: List of trade dictionaries
        """
        trades = []
        position_open = False
        entry_price = 0.0
        entry_time = None
        position_size = 0.0
        
        for i in range(1, len(data)):
            current_price = data.iloc[i]["close"]
            high_price = data.iloc[i]["high"]
            low_price = data.iloc[i]["low"]
            
            if not position_open:
                # Entry condition based on triple barrier labels
                if labels.iloc[i-1] == 1:  # Buy signal
                    position_open = True
                    entry_price = current_price
                    entry_time = data.index[i-1]
                    position_size = tpsl_params["position_sizing_pct"]
                elif labels.iloc[i-1] == -1:  # Sell signal
                    position_open = True
                    entry_price = current_price
                    entry_time = data.index[i-1]
                    position_size = -tpsl_params["position_sizing_pct"]
            else:
                # Check for TP/SL
                if position_size > 0:  # Long position
                    if high_price >= entry_price * (1 + tpsl_params["target_pct"]):
                        # Take profit hit
                        trades.append({
                            "entry_time": entry_time,
                            "exit_time": data.index[i],
                            "entry_price": entry_price,
                            "exit_price": entry_price * (1 + tpsl_params["target_pct"]),
                            "return": tpsl_params["target_pct"] * position_size,
                            "type": "TP",
                            "position_size": position_size
                        })
                        position_open = False
                    elif low_price <= entry_price * (1 - tpsl_params["stop_pct"]):
                        # Stop loss hit
                        trades.append({
                            "entry_time": entry_time,
                            "exit_time": data.index[i],
                            "entry_price": entry_price,
                            "exit_price": entry_price * (1 - tpsl_params["stop_pct"]),
                            "return": -tpsl_params["stop_pct"] * position_size,
                            "type": "SL",
                            "position_size": position_size
                        })
                        position_open = False
                else:  # Short position
                    if low_price <= entry_price * (1 - tpsl_params["target_pct"]):
                        # Take profit hit
                        trades.append({
                            "entry_time": entry_time,
                            "exit_time": data.index[i],
                            "entry_price": entry_price,
                            "exit_price": entry_price * (1 - tpsl_params["target_pct"]),
                            "return": tpsl_params["target_pct"] * abs(position_size),
                            "type": "TP",
                            "position_size": position_size
                        })
                        position_open = False
                    elif high_price >= entry_price * (1 + tpsl_params["stop_pct"]):
                        # Stop loss hit
                        trades.append({
                            "entry_time": entry_time,
                            "exit_time": data.index[i],
                            "entry_price": entry_price,
                            "exit_price": entry_price * (1 + tpsl_params["stop_pct"]),
                            "return": -tpsl_params["stop_pct"] * abs(position_size),
                            "type": "SL",
                            "position_size": position_size
                        })
                        position_open = False
        
        return trades

    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from a list of returns.
        
        Args:
            returns: List of returns
            
        Returns:
            float: Maximum drawdown
        """
        try:
            cumulative = np.cumprod([1 + r for r in returns])
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            return abs(np.min(drawdown))
        except Exception:
            return 1.0

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="per-HMM regime TPSL prediction",
    )
    async def get_optimized_parameters(
        self,
        current_data: pd.DataFrame,
        historical_data: pd.DataFrame,
        exchange: str,
        symbol: str,
        timeframe: str,
        force_optimization: bool = False
    ) -> Dict[str, Any]:
        """Get optimized triple barrier and TPSL parameters for the current HMM regime.
        
        Args:
            current_data: Current market data (OHLCV)
            historical_data: Historical data for optimization
            exchange: Exchange name
            symbol: Symbol name
            timeframe: Timeframe string
            force_optimization: Force re-optimization even if cached
            
        Returns:
            Dictionary with optimized parameters
        """
        try:
            # Identify current HMM regime
            regime, confidence, regime_info = await self.identify_current_hmm_regime(
                current_data, exchange, symbol, timeframe
            )
            
            # Check if we have cached results for this regime
            if not force_optimization and regime in self.optimization_results:
                cached_params = self.optimization_results[regime]
                self.logger.info(f"📋 Using cached parameters for {regime}")
                return {
                    **cached_params,
                    "regime": regime,
                    "confidence": confidence,
                    "regime_info": regime_info,
                    "source": "cached"
                }
            
            # Optimize for current regime
            optimized_params = await self.optimize_regime_parameters(
                regime,
                historical_data,
                current_data,
                force_optimization
            )
            
            # Update regime statistics
            await self._update_regime_statistics(regime, optimized_params, confidence)
            
            return {
                **optimized_params,
                "regime": regime,
                "confidence": confidence,
                "regime_info": regime_info,
                "source": "optimized"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting optimized parameters: {e}")
            # Return default parameters
            return {
                **self.regime_defaults["hmm_cluster_0"],
                "regime": "hmm_cluster_0",
                "confidence": 0.5,
                "regime_info": {"method": "fallback", "error": str(e)},
                "source": "fallback"
            }

    async def _update_regime_statistics(
        self,
        regime: str,
        optimized_params: Dict[str, Any],
        confidence: float
    ) -> None:
        """Update regime statistics with new optimization results.
        
        Args:
            regime: Regime name
            optimized_params: Optimized parameters
            confidence: Regime confidence
        """
        try:
            if regime not in self.regime_statistics:
                self.regime_statistics[regime] = {
                    "optimization_count": 0,
                    "last_optimization": None,
                    "average_confidence": 0.0,
                    "best_score": -1.0,
                    "parameter_history": []
                }
            
            stats = self.regime_statistics[regime]
            stats["optimization_count"] += 1
            stats["last_optimization"] = datetime.now().isoformat()
            stats["average_confidence"] = (
                (stats["average_confidence"] * (stats["optimization_count"] - 1) + confidence) /
                stats["optimization_count"]
            )
            
            score = optimized_params.get("optimization_score", -1.0)
            if score > stats["best_score"]:
                stats["best_score"] = score
            
            # Store parameter history (keep last 10)
            param_summary = {
                "timestamp": datetime.now().isoformat(),
                "score": score,
                "confidence": confidence,
                "triple_barrier": optimized_params.get("optimized_triple_barrier", {}),
                "tpsl": optimized_params.get("optimized_tpsl", {})
            }
            stats["parameter_history"].append(param_summary)
            if len(stats["parameter_history"]) > 10:
                stats["parameter_history"] = stats["parameter_history"][-10:]
            
            await self._save_regime_statistics()
            
        except Exception as e:
            self.logger.error(f"❌ Error updating regime statistics: {e}")

    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about regime-specific optimization.
        
        Returns:
            Dictionary with optimization statistics
        """
        return {
            "optimized_regimes": list(self.optimization_results.keys()),
            "total_optimizations": len(self.optimization_results),
            "last_optimization_time": self.last_optimization_time,
            "regime_statistics": self.regime_statistics,
            "performance_summary": {
                "total_regimes": len(self.regime_defaults),
                "optimized_regimes": len(self.optimization_results),
                "optimization_rate": len(self.optimization_results) / len(self.regime_defaults),
                "average_confidence": np.mean([
                    stats.get("average_confidence", 0.0) 
                    for stats in self.regime_statistics.values()
                ]) if self.regime_statistics else 0.0
            }
        }

    def get_regime_parameter_summary(self) -> Dict[str, Any]:
        """Get a summary of optimized parameters for all regimes.
        
        Returns:
            Dictionary with parameter summary
        """
        summary = {}
        
        for regime, params in self.optimization_results.items():
            summary[regime] = {
                "name": params.get("name", f"Regime {regime}"),
                "optimization_score": params.get("optimization_score", -1.0),
                "last_optimization": params.get("optimization_time", "Unknown"),
                "triple_barrier": params.get("optimized_triple_barrier", {}),
                "tpsl": params.get("optimized_tpsl", {}),
                "characteristics": params.get("characteristics", {})
            }
        
        return summary