# src/tactician/sr_detection_optimization.py

"""
Enhanced S/R Detection Optimization Module

This module implements comprehensive optimization strategies for S/R detection
specifically optimized for 1-30m timeframes. It includes:

1. Multi-Method Ensemble Optimization
2. Advanced Strength Scoring Optimization  
3. Multi-Timeframe Confluence Optimization
4. Advanced S/R Method Optimization
5. DBSCAN Clustering Optimization with real data testing
6. Timeframe-specific parameter optimization

The optimized parameters are then used by the main S/R predictor.
"""

import asyncio
import json
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Optimization libraries
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: optuna not available, using basic optimization")

try:
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score
    from sklearn.model_selection import TimeSeriesSplit
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available, clustering optimization disabled")

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors, handle_specific_errors
from src.tactician.sr_breakout_predictor import SRBreakoutPredictor
from src.tactician.sr_data_integration_simple import SRDataIntegrationSimple, create_sr_data_integration_simple


@dataclass
class OptimizationResult:
    """Result of S/R detection optimization."""
    
    # Optimized parameters
    method_weights: Dict[str, float] = field(default_factory=dict)
    strength_weights: Dict[str, float] = field(default_factory=dict)
    dbscan_params: Dict[str, Any] = field(default_factory=dict)
    timeframe_weights: Dict[str, float] = field(default_factory=dict)
    advanced_params: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    optimization_score: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    signal_clarity: float = 0.0
    
    # Validation metrics
    cross_validation_score: float = 0.0
    out_of_sample_score: float = 0.0
    statistical_significance: float = 0.0
    
    # S/R specific metrics
    sr_validation_score: float = 0.0
    bounce_rate: float = 0.0
    false_breakout_rate: float = 0.0
    volume_confirmation_rate: float = 0.0
    level_detection_accuracy: float = 0.0
    
    # Optimization metadata
    optimization_time: float = 0.0
    n_trials: int = 0
    best_trial_number: int = 0
    optimization_method: str = ""
    market_regime: str = ""
    timeframe_optimized: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "method_weights": self.method_weights,
            "strength_weights": self.strength_weights,
            "dbscan_params": self.dbscan_params,
            "timeframe_weights": self.timeframe_weights,
            "advanced_params": self.advanced_params,
            "performance_metrics": {
                "optimization_score": self.optimization_score,
                "sharpe_ratio": self.sharpe_ratio,
                "win_rate": self.win_rate,
                "max_drawdown": self.max_drawdown,
                "profit_factor": self.profit_factor,
                "signal_clarity": self.signal_clarity,
            },
            "sr_metrics": {
                "sr_validation_score": self.sr_validation_score,
                "bounce_rate": self.bounce_rate,
                "false_breakout_rate": self.false_breakout_rate,
                "volume_confirmation_rate": self.volume_confirmation_rate,
                "level_detection_accuracy": self.level_detection_accuracy,
            },
            "validation_metrics": {
                "cross_validation_score": self.cross_validation_score,
                "out_of_sample_score": self.out_of_sample_score,
                "statistical_significance": self.statistical_significance,
            },
            "metadata": {
                "optimization_time": self.optimization_time,
                "n_trials": self.n_trials,
                "best_trial_number": self.best_trial_number,
                "optimization_method": self.optimization_method,
                "market_regime": self.market_regime,
                "timeframe_optimized": self.timeframe_optimized,
                "timestamp": datetime.now().isoformat(),
            }
        }


class SRDetectionOptimizer:
    """
    Enhanced S/R Detection Optimizer for 1-30m timeframes.
    
    Implements multiple optimization strategies:
    1. Multi-Method Ensemble Optimization
    2. Advanced Strength Scoring Optimization
    3. Multi-Timeframe Confluence Optimization
    4. Advanced S/R Method Optimization
    5. DBSCAN Clustering Optimization with real data testing
    6. Timeframe-specific parameter optimization
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the S/R detection optimizer."""
        self.config = config
        self.logger = system_logger.getChild("SRDetectionOptimizer")
        
        # Optimization configuration
        self.opt_config = config.get("sr_detection_optimization", {})
        self.n_trials = self.opt_config.get("n_trials", 100)
        self.cv_folds = self.opt_config.get("cv_folds", 5)
        self.test_size = self.opt_config.get("test_size", 0.2)
        self.optimization_timeout = self.opt_config.get("optimization_timeout", 3600)  # 1 hour
        
        # Timeframe-specific configuration for 1-30m
        self.timeframe_config = self.opt_config.get("timeframe_config", {
            "1m": {
                "touch_threshold": 0.0005,  # 0.05% for 1m
                "bounce_threshold": 0.002,  # 0.2% for 1m
                "breakout_threshold": 0.005,  # 0.5% for 1m
                "min_touches": 3,
                "volume_spike_threshold": 1.3,
            },
            "5m": {
                "touch_threshold": 0.001,  # 0.1% for 5m
                "bounce_threshold": 0.003,  # 0.3% for 5m
                "breakout_threshold": 0.008,  # 0.8% for 5m
                "min_touches": 3,
                "volume_spike_threshold": 1.4,
            },
            "15m": {
                "touch_threshold": 0.0015,  # 0.15% for 15m
                "bounce_threshold": 0.005,  # 0.5% for 15m
                "breakout_threshold": 0.01,  # 1% for 15m
                "min_touches": 2,
                "volume_spike_threshold": 1.5,
            },
            "30m": {
                "touch_threshold": 0.002,  # 0.2% for 30m
                "bounce_threshold": 0.008,  # 0.8% for 30m
                "breakout_threshold": 0.015,  # 1.5% for 30m
                "min_touches": 2,
                "volume_spike_threshold": 1.6,
            }
        })
        
        # Performance thresholds for 1-30m timeframes
        self.performance_thresholds = self.opt_config.get("performance_thresholds", {
            "min_sr_validation_score": 0.6,  # Lower threshold for shorter timeframes
            "min_bounce_rate": 0.5,  # 50% minimum bounce rate
            "max_false_breakout_rate": 0.4,  # 40% max false breakouts
            "min_volume_confirmation": 0.4,  # 40% volume confirmation
            "min_level_detection_accuracy": 0.3,  # 30% level detection accuracy
        })
        
        # Optimization state
        self.optimization_results: List[OptimizationResult] = []
        self.best_result: Optional[OptimizationResult] = None
        self.optimization_history: List[Dict[str, Any]] = []
        
        # S/R predictor for testing
        self.sr_predictor: Optional[SRBreakoutPredictor] = None
        
        # Data integration
        self.data_integration: Optional[SRDataIntegrationSimple] = None
        
        # Data storage
        self.training_data: Optional[pd.DataFrame] = None
        self.validation_data: Optional[pd.DataFrame] = None
        self.multi_timeframe_data: Optional[Dict[str, pd.DataFrame]] = None
        
    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid optimization configuration"),
            AttributeError: (False, "Missing required components"),
        },
        default_return=False,
        context="S/R detection optimizer initialization"
    )
    async def initialize(self) -> bool:
        """Initialize the S/R detection optimizer."""
        try:
            self.logger.info("🚀 Initializing Enhanced S/R Detection Optimizer for 1-30m timeframes...")
            
            # Initialize S/R predictor
            self.sr_predictor = SRBreakoutPredictor(self.config)
            if not await self.sr_predictor.initialize():
                self.logger.error("Failed to initialize S/R predictor")
                return False
            
            # Validate configuration
            if not self._validate_configuration():
                return False
            
            self.logger.info("✅ Enhanced S/R Detection Optimizer initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize S/R detection optimizer: {e}")
            return False
    
    def _validate_configuration(self) -> bool:
        """Validate optimization configuration."""
        try:
            if self.n_trials <= 0:
                self.logger.error("n_trials must be positive")
                return False
            
            if self.cv_folds < 2:
                self.logger.error("cv_folds must be at least 2")
                return False
            
            if not 0 < self.test_size < 1:
                self.logger.error("test_size must be between 0 and 1")
                return False
            
            # Validate timeframe configuration
            for timeframe, config in self.timeframe_config.items():
                required_keys = ["touch_threshold", "bounce_threshold", "breakout_threshold", "min_touches"]
                for key in required_keys:
                    if key not in config:
                        self.logger.error(f"Missing {key} in {timeframe} configuration")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    @handle_specific_errors(
        error_handlers={
            ValueError: (None, "Invalid data for optimization"),
            AttributeError: (None, "Optimizer not properly initialized"),
        },
        default_return=None,
        context="comprehensive S/R optimization"
    )
    async def optimize_sr_detection(
        self,
        market_data: pd.DataFrame,
        multi_timeframe_data: Optional[Dict[str, pd.DataFrame]] = None,
        target_data: Optional[pd.Series] = None,
        target_timeframe: str = "15m"
    ) -> Optional[OptimizationResult]:
        """
        Run comprehensive S/R detection optimization for specific timeframe.
        
        Args:
            market_data: Main market data for optimization
            multi_timeframe_data: Multi-timeframe data for confluence optimization
            target_data: Target data for supervised optimization (optional)
            target_timeframe: Target timeframe for optimization (1m, 5m, 15m, 30m)
            
        Returns:
            OptimizationResult: Optimized parameters and performance metrics
        """
        try:
            self.logger.info(f"🎯 Starting comprehensive S/R detection optimization for {target_timeframe} timeframe...")
            
            # Validate target timeframe
            if target_timeframe not in self.timeframe_config:
                self.logger.error(f"Invalid target timeframe: {target_timeframe}")
                return None
            
            # Prepare data
            self.training_data = market_data
            self.multi_timeframe_data = multi_timeframe_data or {}
            
            # Split data for validation
            split_idx = int(len(market_data) * (1 - self.test_size))
            self.validation_data = market_data.iloc[split_idx:]
            training_data = market_data.iloc[:split_idx]
            
            # Update configuration for target timeframe
            await self._update_timeframe_config(target_timeframe)
            
            # Run optimization
            if OPTUNA_AVAILABLE:
                result = await self._run_optuna_optimization(training_data, target_data, target_timeframe)
            else:
                result = await self._run_basic_optimization(training_data, target_data, target_timeframe)
            
            if result:
                # Validate on out-of-sample data
                await self._validate_optimization_result(result, target_timeframe)
                
                # Store results
                self.optimization_results.append(result)
                if not self.best_result or result.optimization_score > self.best_result.optimization_score:
                    self.best_result = result
                
                self.logger.info(f"✅ Optimization completed for {target_timeframe}. Best score: {result.optimization_score:.4f}")
                return result
            
            return None
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            return None
    
    async def _update_timeframe_config(self, target_timeframe: str) -> None:
        """Update configuration for specific timeframe."""
        try:
            timeframe_config = self.timeframe_config[target_timeframe]
            
            # Update S/R predictor configuration
            if self.sr_predictor:
                # Update touch and bounce thresholds
                self.sr_predictor.sr_proximity_threshold = timeframe_config["touch_threshold"]
                
                # Update backtesting configuration
                if hasattr(self.sr_predictor, 'backtest_config'):
                    self.sr_predictor.backtest_config.update({
                        "touch_threshold": timeframe_config["touch_threshold"],
                        "bounce_threshold": timeframe_config["bounce_threshold"],
                        "breakout_threshold": timeframe_config["breakout_threshold"],
                        "min_touches": timeframe_config["min_touches"],
                        "volume_spike_threshold": timeframe_config["volume_spike_threshold"],
                    })
            
            self.logger.info(f"Updated configuration for {target_timeframe} timeframe")
            
        except Exception as e:
            self.logger.error(f"Failed to update timeframe configuration: {e}")
    
    async def _run_optuna_optimization(
        self,
        training_data: pd.DataFrame,
        target_data: Optional[pd.Series],
        target_timeframe: str
    ) -> Optional[OptimizationResult]:
        """Run optimization using Optuna with timeframe-specific parameters."""
        try:
            # For now, fall back to basic optimization to avoid asyncio issues
            self.logger.info("Optuna optimization temporarily disabled due to asyncio compatibility issues. Using basic optimization.")
            return await self._run_basic_optimization(training_data, target_data, target_timeframe)
            
        except Exception as e:
            self.logger.error(f"Optuna optimization failed: {e}")
            return None
    
    async def _run_basic_optimization(
        self,
        training_data: pd.DataFrame,
        target_data: Optional[pd.Series],
        target_timeframe: str
    ) -> Optional[OptimizationResult]:
        """Run basic optimization without Optuna."""
        try:
            self.logger.info(f"Running basic optimization for {target_timeframe} (Optuna not available)")
            
            # Define parameter ranges for specific timeframe
            param_ranges = self._get_timeframe_parameter_ranges(target_timeframe)
            
            best_result = None
            best_score = -np.inf
            
            # Grid search over parameter ranges
            for i, params in enumerate(self._generate_parameter_combinations(param_ranges)):
                if i >= self.n_trials:
                    break
                
                score = await self._evaluate_parameters_basic(params, training_data, target_data, target_timeframe)
                
                if score > best_score:
                    best_score = score
                    best_result = OptimizationResult(
                        method_weights=self._extract_method_weights(params),
                        strength_weights=self._extract_strength_weights(params),
                        dbscan_params=self._extract_dbscan_params(params),
                        timeframe_weights=self._extract_timeframe_weights(params),
                        advanced_params=self._extract_advanced_params(params),
                        optimization_score=score,
                        n_trials=i + 1,
                        best_trial_number=i,
                        optimization_method="basic_grid_search",
                        timeframe_optimized=target_timeframe
                    )
                
                if i % 10 == 0:
                    self.logger.info(f"Basic optimization progress: {i}/{min(len(param_ranges), self.n_trials)}")
            
            return best_result
            
        except Exception as e:
            self.logger.error(f"Basic optimization failed: {e}")
            return None
    
    def _get_timeframe_parameter_ranges(self, target_timeframe: str) -> Dict[str, List[Any]]:
        """Get parameter ranges optimized for specific timeframe."""
        base_ranges = {
            "fractal_weight": [0.2, 0.3, 0.4, 0.5, 0.6],
            "volume_weight": [0.2, 0.3, 0.4, 0.5],
            "pivot_weight": [0.1, 0.2, 0.3, 0.4],
            "atr_weight": [0.05, 0.1, 0.15, 0.2],
            "touch_count_weight": [0.2, 0.3, 0.4, 0.5],
            "total_volume_weight": [0.1, 0.2, 0.3, 0.4],
            "level_age_weight": [0.1, 0.2, 0.3, 0.4],
            "bounce_rate_weight": [0.1, 0.2, 0.3, 0.4],
            "isolation_score_weight": [0.05, 0.1, 0.15, 0.2],
        }
        
        # Adjust ranges based on timeframe
        if target_timeframe == "1m":
            # More sensitive parameters for 1m
            base_ranges.update({
                "dbscan_eps": [0.002, 0.005, 0.008, 0.01],
                "dbscan_min_samples": [2, 3, 4],
            })
        elif target_timeframe == "5m":
            base_ranges.update({
                "dbscan_eps": [0.005, 0.008, 0.01, 0.015],
                "dbscan_min_samples": [2, 3, 4, 5],
            })
        elif target_timeframe == "15m":
            base_ranges.update({
                "dbscan_eps": [0.008, 0.01, 0.015, 0.02],
                "dbscan_min_samples": [3, 4, 5, 6],
            })
        elif target_timeframe == "30m":
            # Less sensitive parameters for 30m
            base_ranges.update({
                "dbscan_eps": [0.01, 0.015, 0.02, 0.025],
                "dbscan_min_samples": [4, 5, 6],
            })
        
        return base_ranges
    
    async def _evaluate_parameters(
        self,
        trial: optuna.Trial,
        training_data: pd.DataFrame,
        target_data: Optional[pd.Series],
        target_timeframe: str
    ) -> float:
        """Evaluate parameters using Optuna trial with timeframe-specific suggestions."""
        try:
            # Suggest parameters with timeframe-specific ranges
            params = self._suggest_timeframe_parameters(trial, target_timeframe)
            
            # Evaluate parameters
            return await self._evaluate_parameters_basic(params, training_data, target_data, target_timeframe)
            
        except Exception as e:
            self.logger.error(f"Parameter evaluation failed: {e}")
            return -np.inf
    
    def _suggest_timeframe_parameters(self, trial: optuna.Trial, target_timeframe: str) -> Dict[str, Any]:
        """Suggest parameters optimized for specific timeframe."""
        params = {}
        
        # Method weights (same for all timeframes)
        params["fractal_weight"] = trial.suggest_float("fractal_weight", 0.1, 0.6)
        params["volume_weight"] = trial.suggest_float("volume_weight", 0.1, 0.5)
        params["pivot_weight"] = trial.suggest_float("pivot_weight", 0.1, 0.4)
        params["atr_weight"] = trial.suggest_float("atr_weight", 0.05, 0.3)
        
        # Strength weights (same for all timeframes)
        params["touch_count_weight"] = trial.suggest_float("touch_count_weight", 0.2, 0.5)
        params["total_volume_weight"] = trial.suggest_float("total_volume_weight", 0.1, 0.4)
        params["level_age_weight"] = trial.suggest_float("level_age_weight", 0.1, 0.4)
        params["bounce_rate_weight"] = trial.suggest_float("bounce_rate_weight", 0.1, 0.4)
        params["isolation_score_weight"] = trial.suggest_float("isolation_score_weight", 0.05, 0.3)
        
        # DBSCAN parameters (timeframe-specific)
        if target_timeframe == "1m":
            params["dbscan_eps"] = trial.suggest_float("dbscan_eps", 0.002, 0.01)
            params["dbscan_min_samples"] = trial.suggest_int("dbscan_min_samples", 2, 4)
        elif target_timeframe == "5m":
            params["dbscan_eps"] = trial.suggest_float("dbscan_eps", 0.005, 0.015)
            params["dbscan_min_samples"] = trial.suggest_int("dbscan_min_samples", 2, 5)
        elif target_timeframe == "15m":
            params["dbscan_eps"] = trial.suggest_float("dbscan_eps", 0.008, 0.02)
            params["dbscan_min_samples"] = trial.suggest_int("dbscan_min_samples", 3, 6)
        elif target_timeframe == "30m":
            params["dbscan_eps"] = trial.suggest_float("dbscan_eps", 0.01, 0.025)
            params["dbscan_min_samples"] = trial.suggest_int("dbscan_min_samples", 4, 6)
        
        # Timeframe weights (emphasize target timeframe)
        params["tf_1m_weight"] = trial.suggest_float("tf_1m_weight", 0.05, 0.2)
        params["tf_5m_weight"] = trial.suggest_float("tf_5m_weight", 0.1, 0.25)
        params["tf_15m_weight"] = trial.suggest_float("tf_15m_weight", 0.15, 0.3)
        params["tf_1h_weight"] = trial.suggest_float("tf_1h_weight", 0.2, 0.35)
        params["tf_4h_weight"] = trial.suggest_float("tf_4h_weight", 0.15, 0.3)
        params["tf_1d_weight"] = trial.suggest_float("tf_1d_weight", 0.05, 0.2)
        
        # Advanced parameters (timeframe-specific)
        if target_timeframe in ["1m", "5m"]:
            # More sensitive for shorter timeframes
            params["fibonacci_sensitivity"] = trial.suggest_float("fibonacci_sensitivity", 0.6, 0.9)
            params["elliott_confidence_threshold"] = trial.suggest_float("elliott_confidence_threshold", 0.5, 0.8)
            params["order_flow_hvn_threshold"] = trial.suggest_float("order_flow_hvn_threshold", 1.1, 1.8)
        else:
            # Less sensitive for longer timeframes
            params["fibonacci_sensitivity"] = trial.suggest_float("fibonacci_sensitivity", 0.5, 0.8)
            params["elliott_confidence_threshold"] = trial.suggest_float("elliott_confidence_threshold", 0.4, 0.7)
            params["order_flow_hvn_threshold"] = trial.suggest_float("order_flow_hvn_threshold", 1.3, 2.0)
        
        return params
    
    async def _evaluate_parameters_basic(
        self,
        params: Dict[str, Any],
        training_data: pd.DataFrame,
        target_data: Optional[pd.Series],
        target_timeframe: str
    ) -> float:
        """Evaluate parameters using basic approach with enhanced S/R validation."""
        try:
            # Update S/R predictor with new parameters
            await self._update_sr_predictor_params(params)
            
            # Run cross-validation
            cv_scores = []
            
            if len(training_data) >= self.cv_folds * 50:  # Ensure enough data
                tscv = TimeSeriesSplit(n_splits=self.cv_folds)
                
                for train_idx, val_idx in tscv.split(training_data):
                    train_data = training_data.iloc[train_idx]
                    val_data = training_data.iloc[val_idx]
                    
                    # Get S/R context for validation
                    current_price = val_data['close'].iloc[-1]
                    sr_context = await self.sr_predictor.get_sr_context(val_data, current_price)
                    
                    # Calculate performance metrics with enhanced S/R validation
                    score = await self._calculate_enhanced_performance_score(sr_context, val_data, target_data, target_timeframe)
                    cv_scores.append(score)
                
                # Return mean CV score
                return np.mean(cv_scores) if cv_scores else -np.inf
            else:
                # Use simple validation if not enough data
                current_price = training_data['close'].iloc[-1]
                sr_context = await self.sr_predictor.get_sr_context(training_data, current_price)
                return await self._calculate_enhanced_performance_score(sr_context, training_data, target_data, target_timeframe)
            
        except Exception as e:
            self.logger.error(f"Parameter evaluation failed: {e}")
            return -np.inf
    
    async def _update_sr_predictor_params(self, params: Dict[str, Any]) -> None:
        """Update S/R predictor with new parameters."""
        try:
            if not self.sr_predictor:
                return
            
            # Update method weights
            method_weights = {
                "fractal": params.get("fractal_weight", 0.4),
                "volume": params.get("volume_weight", 0.3),
                "pivot": params.get("pivot_weight", 0.2),
                "atr": params.get("atr_weight", 0.1),
            }
            
            # Normalize weights
            total_weight = sum(method_weights.values())
            if total_weight > 0:
                method_weights = {k: v / total_weight for k, v in method_weights.items()}
            
            self.sr_predictor.model_weights = method_weights
            
            # Update strength weights
            strength_weights = {
                "touch_count": params.get("touch_count_weight", 0.3),
                "total_volume": params.get("total_volume_weight", 0.2),
                "level_age": params.get("level_age_weight", 0.2),
                "bounce_rate": params.get("bounce_rate_weight", 0.2),
                "isolation_score": params.get("isolation_score_weight", 0.1),
            }
            
            # Normalize weights
            total_weight = sum(strength_weights.values())
            if total_weight > 0:
                strength_weights = {k: v / total_weight for k, v in strength_weights.items()}
            
            self.sr_predictor.strength_score_weights = strength_weights
            
            # Update DBSCAN parameters
            self.sr_predictor.dbscan_eps = params.get("dbscan_eps", 0.01)
            self.sr_predictor.dbscan_min_samples = params.get("dbscan_min_samples", 3)
            
            # Update advanced parameters
            # Note: These would need to be added to the SRBreakoutPredictor class
            # For now, we'll store them for later use
            
        except Exception as e:
            self.logger.error(f"Failed to update S/R predictor parameters: {e}")
    
    async def _calculate_enhanced_performance_score(
        self,
        sr_context: Dict[str, Any],
        market_data: pd.DataFrame,
        target_data: Optional[pd.Series],
        target_timeframe: str
    ) -> float:
        """Calculate enhanced performance score with comprehensive S/R validation."""
        try:
            # Import backtesting validator
            from src.tactician.sr_backtesting_validator import setup_sr_backtesting_validator
            
            # Initialize backtesting validator
            validator = await setup_sr_backtesting_validator(self.config)
            if not validator:
                self.logger.warning("Backtesting validator not available, using fallback scoring")
                return self._calculate_fallback_score(sr_context, market_data, target_data)
            
            # Extract S/R levels from context
            support_levels = sr_context.get("support_levels", [])
            resistance_levels = sr_context.get("resistance_levels", [])
            all_levels = support_levels + resistance_levels
            
            if not all_levels:
                return 0.0
            
            # Get current price
            current_price = market_data['close'].iloc[-1]
            
            # Validate S/R levels through backtesting
            backtest_result = await validator.validate_sr_levels(
                market_data=market_data,
                sr_levels=all_levels,
                current_price=current_price
            )
            
            if not backtest_result:
                return 0.0
            
            # Calculate enhanced performance score
            performance_score = self._calculate_timeframe_specific_score(backtest_result, target_timeframe)
            
            # Store backtesting results for analysis
            if not hasattr(self, 'backtest_results'):
                self.backtest_results = []
            self.backtest_results.append({
                'backtest_result': backtest_result,
                'sr_context': sr_context,
                'target_timeframe': target_timeframe,
                'timestamp': pd.Timestamp.now()
            })
            
            return performance_score
            
        except Exception as e:
            self.logger.error(f"Enhanced performance score calculation failed: {e}")
            # Fallback to basic scoring
            return self._calculate_fallback_score(sr_context, market_data, target_data)
    
    def _calculate_timeframe_specific_score(self, backtest_result, target_timeframe: str) -> float:
        """Calculate performance score optimized for specific timeframe."""
        try:
            # Base S/R validation score
            base_score = backtest_result.sr_validation_score
            
            # Timeframe-specific adjustments
            timeframe_adjustments = {
                "1m": {
                    "bounce_rate_weight": 0.4,
                    "volume_weight": 0.3,
                    "accuracy_weight": 0.2,
                    "false_breakout_weight": 0.1,
                },
                "5m": {
                    "bounce_rate_weight": 0.35,
                    "volume_weight": 0.3,
                    "accuracy_weight": 0.25,
                    "false_breakout_weight": 0.1,
                },
                "15m": {
                    "bounce_rate_weight": 0.3,
                    "volume_weight": 0.25,
                    "accuracy_weight": 0.3,
                    "false_breakout_weight": 0.15,
                },
                "30m": {
                    "bounce_rate_weight": 0.25,
                    "volume_weight": 0.2,
                    "accuracy_weight": 0.35,
                    "false_breakout_weight": 0.2,
                }
            }
            
            weights = timeframe_adjustments.get(target_timeframe, timeframe_adjustments["15m"])
            
            # Calculate weighted score
            weighted_score = (
                backtest_result.overall_bounce_rate * weights["bounce_rate_weight"] +
                backtest_result.avg_volume_confirmation_rate * weights["volume_weight"] +
                backtest_result.level_detection_accuracy * weights["accuracy_weight"] +
                (1 - backtest_result.overall_false_breakout_rate) * weights["false_breakout_weight"]
            )
            
            # Combine base score with weighted score
            final_score = (base_score * 0.6) + (weighted_score * 0.4)
            
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            self.logger.error(f"Timeframe-specific score calculation failed: {e}")
            return backtest_result.sr_validation_score if backtest_result else 0.0
    
    def _calculate_fallback_score(
        self,
        sr_context: Dict[str, Any],
        market_data: pd.DataFrame,
        target_data: Optional[pd.Series]
    ) -> float:
        """Fallback performance score calculation when backtesting is not available."""
        try:
            score = 0.0
            
            # Base score from S/R context quality
            if sr_context:
                # Number of levels detected
                support_levels = sr_context.get("support_levels", [])
                resistance_levels = sr_context.get("resistance_levels", [])
                total_levels = len(support_levels) + len(resistance_levels)
                
                if total_levels > 0:
                    score += min(total_levels / 10.0, 1.0) * 0.3  # Max 30% for level count
                
                # Average strength
                avg_strength = 0.0
                if support_levels:
                    avg_strength += np.mean([level.get("enhanced_strength", level.get("strength", 0.5)) for level in support_levels])
                if resistance_levels:
                    avg_strength += np.mean([level.get("enhanced_strength", level.get("strength", 0.5)) for level in resistance_levels])
                
                if total_levels > 0:
                    avg_strength /= total_levels
                    score += avg_strength * 0.3  # 30% for strength
                
                # Clustering quality
                clustering_result = sr_context.get("clustering_result", {})
                if clustering_result.get("n_clusters", 0) > 0:
                    score += min(clustering_result["n_clusters"] / 5.0, 1.0) * 0.2  # 20% for clustering
                
                # Advanced analysis quality
                fibonacci_levels = sr_context.get("fibonacci_levels", {})
                elliott_wave_levels = sr_context.get("elliott_wave_levels", {})
                order_flow_analysis = sr_context.get("order_flow_analysis", {})
                
                advanced_score = 0.0
                if fibonacci_levels:
                    advanced_score += 0.3
                if elliott_wave_levels.get("pattern_type") != "incomplete":
                    advanced_score += 0.3
                if order_flow_analysis.get("poc"):
                    advanced_score += 0.4
                
                score += advanced_score * 0.2  # 20% for advanced analysis
            
            # If target data is provided, calculate supervised score
            if target_data is not None and len(target_data) > 0:
                try:
                    features = self._extract_sr_features(sr_context, market_data)
                    if features and len(features) == len(target_data):
                        correlation = np.corrcoef(features, target_data)[0, 1]
                        if not np.isnan(correlation):
                            score += abs(correlation) * 0.5  # 50% bonus for supervised learning
                except Exception as e:
                    self.logger.debug(f"Supervised scoring failed: {e}")
            
            return max(0.0, min(1.0, score))  # Ensure score is between 0 and 1
            
        except Exception as e:
            self.logger.error(f"Fallback performance score calculation failed: {e}")
            return 0.0
    
    def _extract_sr_features(
        self,
        sr_context: Dict[str, Any],
        market_data: pd.DataFrame
    ) -> Optional[np.ndarray]:
        """Extract features from S/R context for supervised learning."""
        try:
            if not sr_context:
                return None
            
            features = []
            
            for i in range(len(market_data)):
                # Basic S/R features
                support_proximity = sr_context.get("support_proximity", 0.0)
                resistance_proximity = sr_context.get("resistance_proximity", 0.0)
                support_strength = sr_context.get("support_strength", 0.5)
                resistance_strength = sr_context.get("resistance_strength", 0.5)
                
                # Combine features
                feature_value = (
                    support_proximity * 0.3 +
                    resistance_proximity * 0.3 +
                    support_strength * 0.2 +
                    resistance_strength * 0.2
                )
                
                features.append(feature_value)
            
            return np.array(features)
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return None
    
    async def _validate_optimization_result(self, result: OptimizationResult, target_timeframe: str) -> None:
        """Validate optimization result on out-of-sample data."""
        try:
            if self.validation_data is None:
                return
            
            # Update S/R predictor with optimized parameters
            await self._update_sr_predictor_params({
                **result.method_weights,
                **result.strength_weights,
                **result.dbscan_params,
                **result.timeframe_weights,
                **result.advanced_params,
            })
            
            # Test on validation data
            current_price = self.validation_data['close'].iloc[-1]
            sr_context = await self.sr_predictor.get_sr_context(self.validation_data, current_price)
            
            # Calculate out-of-sample score
            oos_score = await self._calculate_enhanced_performance_score(sr_context, self.validation_data, None, target_timeframe)
            result.out_of_sample_score = oos_score
            
            # Calculate statistical significance (simplified)
            if len(self.optimization_results) > 1:
                scores = [r.optimization_score for r in self.optimization_results]
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                if std_score > 0:
                    result.statistical_significance = (result.optimization_score - mean_score) / std_score
                else:
                    result.statistical_significance = 0.0
            
            self.logger.info(f"Validation completed for {target_timeframe}. OOS score: {oos_score:.4f}")
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
    
    def _extract_method_weights(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Extract method weights from parameters."""
        return {
            "fractal": params.get("fractal_weight", 0.4),
            "volume": params.get("volume_weight", 0.3),
            "pivot": params.get("pivot_weight", 0.2),
            "atr": params.get("atr_weight", 0.1),
        }
    
    def _extract_strength_weights(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Extract strength weights from parameters."""
        return {
            "touch_count": params.get("touch_count_weight", 0.3),
            "total_volume": params.get("total_volume_weight", 0.2),
            "level_age": params.get("level_age_weight", 0.2),
            "bounce_rate": params.get("bounce_rate_weight", 0.2),
            "isolation_score": params.get("isolation_score_weight", 0.1),
        }
    
    def _extract_dbscan_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract DBSCAN parameters from parameters."""
        return {
            "eps": params.get("dbscan_eps", 0.01),
            "min_samples": params.get("dbscan_min_samples", 3),
        }
    
    def _extract_timeframe_weights(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Extract timeframe weights from parameters."""
        return {
            "1m": params.get("tf_1m_weight", 0.1),
            "5m": params.get("tf_5m_weight", 0.15),
            "15m": params.get("tf_15m_weight", 0.2),
            "1h": params.get("tf_1h_weight", 0.25),
            "4h": params.get("tf_4h_weight", 0.2),
            "1d": params.get("tf_1d_weight", 0.1),
        }
    
    def _extract_advanced_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract advanced parameters from parameters."""
        return {
            "fibonacci_sensitivity": params.get("fibonacci_sensitivity", 0.7),
            "elliott_confidence_threshold": params.get("elliott_confidence_threshold", 0.6),
            "order_flow_hvn_threshold": params.get("order_flow_hvn_threshold", 1.5),
        }
    
    def _get_basic_parameter_ranges(self) -> Dict[str, List[Any]]:
        """Get parameter ranges for basic optimization."""
        return {
            "fractal_weight": [0.2, 0.3, 0.4, 0.5, 0.6],
            "volume_weight": [0.2, 0.3, 0.4, 0.5],
            "pivot_weight": [0.1, 0.2, 0.3, 0.4],
            "atr_weight": [0.05, 0.1, 0.15, 0.2],
            "touch_count_weight": [0.2, 0.3, 0.4, 0.5],
            "total_volume_weight": [0.1, 0.2, 0.3, 0.4],
            "level_age_weight": [0.1, 0.2, 0.3, 0.4],
            "bounce_rate_weight": [0.1, 0.2, 0.3, 0.4],
            "isolation_score_weight": [0.05, 0.1, 0.15, 0.2],
            "dbscan_eps": [0.005, 0.01, 0.015, 0.02],
            "dbscan_min_samples": [2, 3, 4, 5],
        }
    
    def _generate_parameter_combinations(self, param_ranges: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate parameter combinations for grid search."""
        import itertools
        
        # Get all combinations
        keys = list(param_ranges.keys())
        values = list(param_ranges.values())
        
        combinations = []
        for combination in itertools.product(*values):
            params = dict(zip(keys, combination))
            
            # Normalize weights
            method_weights = [params["fractal_weight"], params["volume_weight"], 
                            params["pivot_weight"], params["atr_weight"]]
            total_weight = sum(method_weights)
            if total_weight > 0:
                params["fractal_weight"] /= total_weight
                params["volume_weight"] /= total_weight
                params["pivot_weight"] /= total_weight
                params["atr_weight"] /= total_weight
            
            strength_weights = [params["touch_count_weight"], params["total_volume_weight"],
                              params["level_age_weight"], params["bounce_rate_weight"],
                              params["isolation_score_weight"]]
            total_weight = sum(strength_weights)
            if total_weight > 0:
                params["touch_count_weight"] /= total_weight
                params["total_volume_weight"] /= total_weight
                params["level_age_weight"] /= total_weight
                params["bounce_rate_weight"] /= total_weight
                params["isolation_score_weight"] /= total_weight
            
            combinations.append(params)
        
        return combinations
    
    def get_optimized_parameters(self) -> Optional[Dict[str, Any]]:
        """Get the best optimized parameters."""
        if self.best_result:
            return {
                "method_weights": self.best_result.method_weights,
                "strength_weights": self.best_result.strength_weights,
                "dbscan_params": self.best_result.dbscan_params,
                "timeframe_weights": self.best_result.timeframe_weights,
                "advanced_params": self.best_result.advanced_params,
            }
        return None
    
    def save_optimization_results(self, filepath: str) -> bool:
        """Save optimization results to file."""
        try:
            results = {
                "best_result": self.best_result.to_dict() if self.best_result else None,
                "all_results": [r.to_dict() for r in self.optimization_results],
                "optimization_history": self.optimization_history,
                "config": self.config,
            }
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"✅ Optimization results saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save optimization results: {e}")
            return False
    
    def load_optimization_results(self, filepath: str) -> bool:
        """Load optimization results from file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if data.get("best_result"):
                self.best_result = OptimizationResult(**data["best_result"])
            
            if data.get("all_results"):
                self.optimization_results = [OptimizationResult(**r) for r in data["all_results"]]
            
            if data.get("optimization_history"):
                self.optimization_history = data["optimization_history"]
            
            self.logger.info(f"✅ Optimization results loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load optimization results: {e}")
            return False


# Setup function for easy integration
async def setup_sr_detection_optimizer(config: Dict[str, Any]) -> Optional[SRDetectionOptimizer]:
    """Setup S/R detection optimizer."""
    try:
        optimizer = SRDetectionOptimizer(config)
        if await optimizer.initialize():
            return optimizer
        return None
    except Exception as e:
        system_logger.error(f"Failed to setup S/R detection optimizer: {e}")
        return None