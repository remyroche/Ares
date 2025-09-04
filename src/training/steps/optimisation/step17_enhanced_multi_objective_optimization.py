"""
Enhanced Step17 Multi-Objective Optimization with Block-Based Approach

This implementation addresses:
1. Curse of dimensionality through parameter block optimization
2. Logical block ordering for efficient optimization
3. Multi-objective optimization (PnL, win rate, Sharpe ratio, etc.)
4. Computational efficiency through hierarchical optimization
"""

import asyncio
import json
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

import optuna
import numpy as np
import pandas as pd
from optuna.samplers import NSGAIISampler, TPESampler
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner

from src.config.config_manager import get_config_manager, get_optimizable_parameters, get_search_space, update_optimizable_config
from src.core.decorators import handles_errors
from src.utils.logger import system_logger


class OptimizationBlock(Enum):
    """Logical blocks for parameter optimization to avoid curse of dimensionality."""
    
    # Block 1: Market analysis (regime transitions only - S/R and technical indicators removed)
    MARKET_ANALYSIS = "market_analysis"
    
    # Block 2: Core intensity parameters
    CORE_INTENSITY = "core_intensity"
    
    # Block 3: Signal processing
    SIGNAL_PROCESSING = "signal_processing"
    
    # Block 4: Core confidence parameters
    CORE_CONFIDENCE = "core_confidence"
    
    # Block 5: Position management
    POSITION_MANAGEMENT = "position_management"
    
    # Block 6: Risk management
    RISK_MANAGEMENT = "risk_management"


@dataclass
class OptimizationObjective:
    """Multi-objective optimization targets."""
    name: str
    weight: float
    direction: str  # "maximize" or "minimize"
    target_value: Optional[float] = None


@dataclass
class BlockOptimizationResult:
    """Result of block optimization."""
    block_name: str
    best_params: Dict[str, Any]
    objectives: Dict[str, float]
    n_trials: int
    optimization_time: float
    convergence_score: float


class EnhancedStep17Optimizer:
    """
    Enhanced Step17 optimizer with block-based multi-objective optimization.
    
    Features:
    - Block-based optimization to avoid curse of dimensionality
    - Logical block ordering for efficient optimization
    - Multi-objective optimization with weighted objectives
    - Computational efficiency through hierarchical optimization
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("EnhancedStep17Optimizer")
        self.config_manager = get_config_manager()
        
        # Optimization configuration
        self.optimization_config = config.get("step17_enhanced_optimization", {})
        
        # Multi-objective configuration
        self.objectives = self._setup_objectives()
        
        # Block configuration
        self.optimization_blocks = self._setup_optimization_blocks()
        
        # Optimization state
        self.optimization_results: Dict[str, BlockOptimizationResult] = {}
        self.global_best_score = float('-inf')
        
        # Performance tracking
        self.optimization_history: List[Dict[str, Any]] = []
        
    def _setup_objectives(self) -> List[OptimizationObjective]:
        """Setup multi-objective optimization targets."""
        objectives_config = self.optimization_config.get("objectives", {})
        
        return [
            OptimizationObjective(
                name="profit_factor",
                weight=objectives_config.get("profit_factor_weight", 0.5),
                direction="maximize",
                target_value=objectives_config.get("target_profit_factor", 1.5)
            ),
            OptimizationObjective(
                name="sharpe_ratio",
                weight=objectives_config.get("sharpe_weight", 0.125),
                direction="maximize",
                target_value=objectives_config.get("target_sharpe", 2.0)
            ),
            OptimizationObjective(
                name="win_rate",
                weight=objectives_config.get("win_rate_weight", 0.125),
                direction="maximize",
                target_value=objectives_config.get("target_win_rate", 0.6)
            ),
            OptimizationObjective(
                name="max_drawdown",
                weight=objectives_config.get("drawdown_weight", 0.125),
                direction="minimize",
                target_value=objectives_config.get("target_drawdown", 0.1)
            ),
            OptimizationObjective(
                name="total_return",
                weight=objectives_config.get("return_weight", 0.125),
                direction="maximize",
                target_value=objectives_config.get("target_return", 0.3)
            )
        ]
    
    def _setup_optimization_blocks(self) -> Dict[OptimizationBlock, Dict[str, Any]]:
        """Setup logical optimization blocks to avoid curse of dimensionality."""
        
        return {
            # Block 1: Market analysis (regime transitions only - S/R and technical indicators removed)
            OptimizationBlock.MARKET_ANALYSIS: {
                "categories": ["regime_transitions"],
                "parameters": [
                    # Regime transitions parameters only
                    "transition_intensity_threshold", "min_combined_intensity", "max_regimes_to_consider",
                    "transition_confidence_threshold", "step9_5_weight", "step10_weight", "regime_expert_weight",
                    "transition_lookback_periods", "transition_risk_multiplier"
                ],
                "n_trials": 60,
                "timeout": 480,  # 8 minutes
                "sampler": "tpe",
                "pruner": "median",
                "description": "Regime transitions optimization (S/R and technical indicators removed - optimized in step2_5)"
            },
            
            # Block 2: Core intensity parameters
            OptimizationBlock.CORE_INTENSITY: {
                "categories": ["intensity"],
                "parameters": [
                    "transition_intensity_threshold", "min_combined_intensity", "signal_intensity_threshold",
                    "intensity_reliability_weight", "intensity_decay_rate", "intensity_boost_factor",
                    "regime_transition_intensity", "regime_stability_threshold", "regime_change_boost",
                    "breakout_intensity_threshold", "volume_intensity_threshold", "momentum_intensity_threshold",
                    "intensity_position_multiplier", "high_intensity_boost", "low_intensity_reduction",
                    "intensity_nms_threshold", "intensity_overlap_threshold", "intensity_time_decay",
                    "intensity_persistence"
                ],
                "n_trials": 80,
                "timeout": 480,  # 8 minutes
                "sampler": "tpe",
                "pruner": "median",
                "description": "Intensity thresholds and weighting parameters"
            },
            
            # Block 3: Signal processing
            OptimizationBlock.SIGNAL_PROCESSING: {
                "categories": ["ensemble", "signal_aggregation"],
                "parameters": [
                    # Ensemble parameters
                    "ensemble_method", "base_models", "meta_model", "weights", "cross_validation_folds",
                    "sharpe_ratio", "max_drawdown", "win_rate", "profit_factor", "total_return",
                    "barrier_hit_rate", "online_learning", "regime_awareness", "uncertainty_weighting",
                    "learning_rate", "performance_window", "weight_combination",
                    # Signal aggregation parameters
                    "analyst_weight", "tactician_weight", "scenario_weight", "sr_breakout_weight",
                    "use_multiplicative", "conflict_penalty", "signal_quality_threshold"
                ],
                "n_trials": 100,
                "timeout": 600,  # 10 minutes
                "sampler": "tpe",
                "pruner": "median",
                "description": "Ensemble and signal aggregation optimization"
            },
            
            # Block 4: Core confidence parameters
            OptimizationBlock.CORE_CONFIDENCE: {
                "categories": ["confidence"],
                "parameters": [
                    "base_entry_threshold", "analyst_confidence_threshold", "tactician_confidence_threshold",
                    "position_scale_up_threshold", "position_scale_down_threshold", "position_close_threshold",
                    "ensemble_agreement_threshold", "neutral_signal_threshold", "tactician_close_threshold",
                    "model_performance_threshold", "model_degradation_threshold", "model_retrain_threshold",
                    "min_sr_confidence", "high_confidence_threshold", "confidence_decay_rate",
                    "ensemble_confidence_threshold", "breakout_confidence_threshold", "false_breakout_filter",
                    "confidence_min_threshold", "confidence_max_threshold", "confidence_min_multiplier",
                    "confidence_max_multiplier", "entry_risk_threshold", "profit_confidence_threshold",
                    "confidence_scaling_factor", "risk_scaling_factor", "profit_scaling_factor"
                ],
                "n_trials": 100,
                "timeout": 600,  # 10 minutes
                "sampler": "tpe",
                "pruner": "median",
                "description": "Core confidence thresholds and linear scaling parameters"
            },
            
            # Block 5: Position management
            OptimizationBlock.POSITION_MANAGEMENT: {
                "categories": ["position_sizing", "leverage"],
                "parameters": [
                    # Position sizing parameters
                    "kelly_multiplier", "max_position_size", "min_position_size", "confidence_threshold",
                    "positionsize_combined_threshold", "ml_weight", "base_position_size",
                    "confidence_based_scaling", "low_confidence_multiplier", "medium_confidence_multiplier",
                    "high_confidence_multiplier", "very_high_confidence_multiplier",
                    # Leverage parameters
                    "min_leverage", "max_leverage", "leverage_combined_threshold", "liquidation_buffer",
                    "leverage_multiplier", "max_risk_leverage", "liquidation_weight"
                ],
                "n_trials": 120,
                "timeout": 900,  # 15 minutes
                "sampler": "nsga2",
                "pruner": "successive_halving",
                "description": "Position sizing and leverage optimization"
            },
            
            # Block 6: Risk management
            OptimizationBlock.RISK_MANAGEMENT: {
                "categories": ["tpsl"],
                "parameters": [
                    "stop_loss_atr_multiplier", "trailing_stop_atr_multiplier", "stop_loss_confidence_threshold",
                    "enable_dynamic_stop_loss", "volatility_based_sl", "regime_based_sl",
                    "sl_tightening_threshold", "sl_loosening_threshold", "max_drawdown_threshold",
                    "max_daily_loss", "atr_multiplier", "confidence_threshold", "min_hold_time",
                    "stop_loss_multiplier", "take_profit_multiplier", "trailing_stop_enabled",
                    "trailing_stop_distance", "max_hold_time"
                ],
                "n_trials": 60,
                "timeout": 360,  # 6 minutes
                "sampler": "tpe",
                "pruner": "median",
                "description": "Take profit and stop loss optimization"
            }
        }
    
    @handles_errors(fallback=False)
    async def initialize(self) -> bool:
        """Initialize the enhanced optimizer."""
        self.logger.info("🚀 Initializing Enhanced Step17 Multi-Objective Optimizer...")
        
        # Validate configuration
        if not self._validate_configuration():
            return False
        
        # Setup optimization storage
        self._setup_optimization_storage()
        
        self.logger.info("✅ Enhanced Step17 Optimizer initialized successfully")
        return True
    
    @handles_errors(fallback={})
    async def execute_optimization(
        self, 
        training_input: Dict[str, Any], 
        pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute block-based multi-objective optimization.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Dict containing optimization results
        """
        try:
            self.logger.info("🔄 Starting Enhanced Step17 Multi-Objective Optimization...")
            start_time = datetime.now()
            
            # Load calibration results
            calibration_results = await self._load_calibration_results(training_input)
            if not calibration_results:
                raise FileNotFoundError("Calibration results not found")
            
            # Load previous optimization results for warm start
            previous_results = await self._load_previous_optimization_results(training_input)
            
            # Execute block-based optimization
            optimization_results = await self._execute_block_optimization(
                calibration_results, 
                previous_results
            )
            
            # Perform final global optimization
            final_results = await self._execute_global_optimization(
                optimization_results, 
                calibration_results
            )
            
            # Generate optimization report
            optimization_report = self._generate_optimization_report(
                optimization_results, 
                final_results, 
                start_time
            )
            
            # Save results
            await self._save_optimization_results(optimization_results, final_results)
            
            self.logger.info("✅ Enhanced Step17 Optimization completed successfully")
            return optimization_report
            
        except Exception as e:
            self.logger.error(f"Error in enhanced optimization: {e}")
            raise
    
    async def _execute_block_optimization(
        self, 
        calibration_results: Dict[str, Any], 
        previous_results: Dict[str, Any]
    ) -> Dict[str, BlockOptimizationResult]:
        """Execute optimization for each block in logical order."""
        
        optimization_results = {}
        
        # Execute blocks in logical dependency order
        for block in OptimizationBlock:
            self.logger.info(f"🔄 Optimizing block: {block.value}")
            block_start_time = datetime.now()
            
            try:
                block_config = self.optimization_blocks[block]
                block_result = await self._optimize_block(
                    block, 
                    block_config, 
                    calibration_results, 
                    previous_results
                )
                
                optimization_results[block.value] = block_result
                
                # Update global configuration with best parameters
                self._update_global_config(block_result.best_params, block_config["categories"])
                
                block_time = (datetime.now() - block_start_time).total_seconds()
                self.logger.info(f"✅ Block {block.value} completed in {block_time:.1f}s")
                
            except Exception as e:
                self.logger.error(f"Error optimizing block {block.value}: {e}")
                continue
        
        return optimization_results
    
    async def _optimize_block(
        self, 
        block: OptimizationBlock, 
        block_config: Dict[str, Any], 
        calibration_results: Dict[str, Any], 
        previous_results: Dict[str, Any]
    ) -> BlockOptimizationResult:
        """Optimize a single block with multi-objective optimization."""
        
        # Create multi-objective study
        study = self._create_multi_objective_study(block, block_config)
        
        # Define objective function
        def objective(trial):
            return self._multi_objective_function(
                trial, 
                block_config["categories"], 
                calibration_results
            )
        
        # Run optimization
        study.optimize(
            objective, 
            n_trials=block_config["n_trials"], 
            timeout=block_config["timeout"]
        )
        
        # Get best parameters and objectives
        best_trial = self._get_best_trial(study)
        best_params = best_trial.params
        objectives = self._extract_objectives(best_trial)
        
        # Calculate convergence score
        convergence_score = self._calculate_convergence_score(study)
        
        return BlockOptimizationResult(
            block_name=block.value,
            best_params=best_params,
            objectives=objectives,
            n_trials=block_config["n_trials"],
            optimization_time=block_config["timeout"],
            convergence_score=convergence_score
        )
    
    def _create_multi_objective_study(
        self, 
        block: OptimizationBlock, 
        block_config: Dict[str, Any]
    ) -> optuna.Study:
        """Create multi-objective study with appropriate sampler and pruner."""
        
        # Choose sampler based on block configuration
        if block_config["sampler"] == "nsga2":
            sampler = NSGAIISampler(
                population_size=50,
                mutation_prob=0.1,
                crossover_prob=0.8
            )
        else:  # tpe
            sampler = TPESampler(
                n_startup_trials=10,
                n_ei_candidates=24,
                gamma=0.25
            )
        
        # Choose pruner based on block configuration
        if block_config["pruner"] == "successive_halving":
            pruner = SuccessiveHalvingPruner(
                min_resource=1,
                reduction_factor=4,
                min_early_stopping_rate=0
            )
        else:  # median
            pruner = MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            )
        
        # Create study
        study_name = f"step17_enhanced_{block.value}"
        study = optuna.create_study(
            study_name=study_name,
            directions=["maximize"] * len(self.objectives),
            sampler=sampler,
            pruner=pruner,
            storage="sqlite:///optuna_enhanced_studies.db",
            load_if_exists=True
        )
        
        return study
    
    def _multi_objective_function(
        self, 
        trial: optuna.Trial, 
        categories: List[str], 
        calibration_results: Dict[str, Any]
    ) -> List[float]:
        """Multi-objective function for optimization."""
        
        # Suggest parameters for all categories in this block
        params = {}
        for category in categories:
            search_space = get_search_space(category)
            if search_space:
                for param_name, param_config in search_space.items():
                    if param_config["type"] == "float":
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_config["min"],
                            param_config["max"]
                        )
                    elif param_config["type"] == "int":
                        params[param_name] = trial.suggest_int(
                            param_name,
                            param_config["min"],
                            param_config["max"]
                        )
        
        # Update configuration
        for category in categories:
            category_params = {k: v for k, v in params.items() 
                             if k in get_search_space(category)}
            if category_params:
                update_optimizable_config(category, category_params)
        
        # Evaluate multi-objective performance
        objectives = self._evaluate_multi_objective_performance(
            categories, 
            params, 
            calibration_results
        )
        
        return objectives
    
    def _evaluate_multi_objective_performance(
        self, 
        categories: List[str], 
        params: Dict[str, Any], 
        calibration_results: Dict[str, Any]
    ) -> List[float]:
        """Evaluate multi-objective performance for given parameters."""
        
        # Run backtest with current parameters
        backtest_results = self._run_backtest(categories, params, calibration_results)
        
        # Extract objective values
        objectives = []
        for objective in self.objectives:
            value = backtest_results.get(objective.name, 0.0)
            
            # Apply direction (minimize objectives are negated)
            if objective.direction == "minimize":
                value = -value
            
            objectives.append(value)
        
        return objectives
    
    def _get_best_trial(self, study: optuna.Study) -> optuna.Trial:
        """Get the best trial based on weighted objectives."""
        
        best_trial = None
        best_weighted_score = float('-inf')
        
        for trial in study.trials:
            if trial.values is None:
                continue
            
            # Calculate weighted score
            weighted_score = sum(
                objective.weight * value 
                for objective, value in zip(self.objectives, trial.values)
            )
            
            if weighted_score > best_weighted_score:
                best_weighted_score = weighted_score
                best_trial = trial
        
        return best_trial or study.best_trials[0]
    
    def _extract_objectives(self, trial: optuna.Trial) -> Dict[str, float]:
        """Extract objective values from trial."""
        objectives = {}
        for i, objective in enumerate(self.objectives):
            if trial.values and i < len(trial.values):
                value = trial.values[i]
                # Convert back from minimize (negated) values
                if objective.direction == "minimize":
                    value = -value
                objectives[objective.name] = value
        return objectives
    
    def _calculate_convergence_score(self, study: optuna.Study) -> float:
        """Calculate convergence score for the study."""
        if len(study.trials) < 10:
            return 0.0
        
        # Calculate improvement over last 20% of trials
        recent_trials = study.trials[-max(1, len(study.trials) // 5):]
        early_trials = study.trials[:max(1, len(study.trials) // 5)]
        
        if not recent_trials or not early_trials:
            return 0.0
        
        # Calculate average improvement
        recent_scores = [sum(trial.values) for trial in recent_trials if trial.values]
        early_scores = [sum(trial.values) for trial in early_trials if trial.values]
        
        if not recent_scores or not early_scores:
            return 0.0
        
        improvement = (np.mean(recent_scores) - np.mean(early_scores)) / abs(np.mean(early_scores))
        return max(0.0, min(1.0, improvement))
    
    async def _execute_global_optimization(
        self, 
        block_results: Dict[str, BlockOptimizationResult], 
        calibration_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute final global optimization across all blocks."""
        
        self.logger.info("🔄 Executing final global optimization...")
        
        # Create global study with all parameters
        study = optuna.create_study(
            study_name="step17_global_optimization",
            direction="maximize",
            sampler=TPESampler(n_startup_trials=20),
            pruner=MedianPruner(),
            storage="sqlite:///optuna_enhanced_studies.db",
            load_if_exists=True
        )
        
        # Define global objective function
        def global_objective(trial):
            return self._global_objective_function(trial, calibration_results)
        
        # Run global optimization
        study.optimize(global_objective, n_trials=200, timeout=1800)  # 30 minutes
        
        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials),
            "convergence_score": self._calculate_convergence_score(study)
        }
    
    def _global_objective_function(
        self, 
        trial: optuna.Trial, 
        calibration_results: Dict[str, Any]
    ) -> float:
        """Global objective function for final optimization."""
        
        # Suggest parameters for all categories
        all_params = {}
        for block in OptimizationBlock:
            block_config = self.optimization_blocks[block]
            for category in block_config["categories"]:
                search_space = get_search_space(category)
                if search_space:
                    for param_name, param_config in search_space.items():
                        if param_name not in all_params:  # Avoid duplicates
                            if param_config["type"] == "float":
                                all_params[param_name] = trial.suggest_float(
                                    param_name,
                                    param_config["min"],
                                    param_config["max"]
                                )
                            elif param_config["type"] == "int":
                                all_params[param_name] = trial.suggest_int(
                                    param_name,
                                    param_config["min"],
                                    param_config["max"]
                                )
        
        # Update all configurations
        for block in OptimizationBlock:
            block_config = self.optimization_blocks[block]
            for category in block_config["categories"]:
                category_params = {k: v for k, v in all_params.items() 
                                 if k in get_search_space(category)}
                if category_params:
                    update_optimizable_config(category, category_params)
        
        # Evaluate global performance
        backtest_results = self._run_backtest(
            [cat for block in OptimizationBlock 
             for cat in self.optimization_blocks[block]["categories"]], 
            all_params, 
            calibration_results
        )
        
        # Calculate weighted score
        weighted_score = sum(
            objective.weight * backtest_results.get(objective.name, 0.0)
            for objective in self.objectives
        )
        
        return weighted_score
    
    def _run_backtest(
        self, 
        categories: List[str], 
        params: Dict[str, Any], 
        calibration_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """Run backtest with given parameters and return performance metrics."""
        
        # This would integrate with your existing backtesting system
        # For now, return mock results - replace with actual backtesting
        
        # Mock performance metrics (replace with actual backtesting)
        return {
            "sharpe_ratio": np.random.normal(1.5, 0.3),
            "win_rate": np.random.normal(0.55, 0.1),
            "profit_factor": np.random.normal(1.3, 0.2),
            "max_drawdown": np.random.normal(0.12, 0.05),
            "total_return": np.random.normal(0.25, 0.1)
        }
    
    def _update_global_config(self, params: Dict[str, Any], categories: List[str]):
        """Update global configuration with optimized parameters."""
        for category in categories:
            category_params = {k: v for k, v in params.items() 
                             if k in get_search_space(category)}
            if category_params:
                update_optimizable_config(category, category_params)
    
    def _generate_optimization_report(
        self, 
        block_results: Dict[str, BlockOptimizationResult], 
        final_results: Dict[str, Any], 
        start_time: datetime
    ) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate overall performance
        overall_objectives = {}
        for objective in self.objectives:
            overall_objectives[objective.name] = final_results.get("best_value", 0.0)
        
        # Generate block performance summary
        block_summary = {}
        for block_name, result in block_results.items():
            block_summary[block_name] = {
                "objectives": result.objectives,
                "n_trials": result.n_trials,
                "optimization_time": result.optimization_time,
                "convergence_score": result.convergence_score
            }
        
        return {
            "optimization_type": "enhanced_multi_objective_block_based",
            "total_optimization_time": total_time,
            "overall_objectives": overall_objectives,
            "block_results": block_summary,
            "final_results": final_results,
            "optimization_summary": {
                "total_blocks": len(block_results),
                "total_trials": sum(result.n_trials for result in block_results.values()),
                "average_convergence": np.mean([result.convergence_score for result in block_results.values()]),
                "best_block": max(block_results.keys(), key=lambda k: block_results[k].convergence_score)
            }
        }
    
    async def _load_calibration_results(self, training_input: Dict[str, Any]) -> Dict[str, Any]:
        """Load calibration results for optimization."""
        # Implementation would load actual calibration results
        return {"calibration_data": "mock_data"}
    
    async def _load_previous_optimization_results(self, training_input: Dict[str, Any]) -> Dict[str, Any]:
        """Load previous optimization results for warm start."""
        # Implementation would load actual previous results
        return {}
    
    async def _save_optimization_results(
        self, 
        block_results: Dict[str, BlockOptimizationResult], 
        final_results: Dict[str, Any]
    ):
        """Save optimization results."""
        # Implementation would save results to storage
        pass
    
    def _validate_configuration(self) -> bool:
        """Validate optimization configuration."""
        # Implementation would validate configuration
        return True
    
    def _setup_optimization_storage(self):
        """Setup optimization storage."""
        # Implementation would setup storage
        pass


# Factory function for backward compatibility
def create_enhanced_step17_optimizer(config: Dict[str, Any]) -> EnhancedStep17Optimizer:
    """Create enhanced Step17 optimizer instance."""
    return EnhancedStep17Optimizer(config)