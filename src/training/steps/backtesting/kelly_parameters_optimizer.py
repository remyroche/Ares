"""
Kelly Parameters Optimizer - Nested Regime-Aware Optimization

Optimizes dampened Kelly parameters using nested optimization strategy:
1. Global optimization (150 trials): Optimize global fallback parameters
2. Per-regime refinement (50 trials each): Tune per-regime starting from global
3. Hierarchical regularization: L2 penalty toward global for sparse regimes

Multi-objective optimization with 6 objectives:
- geometric_mean (maximize)
- sharpe_ratio (maximize)
- max_drawdown (minimize)
- high_leverage_frequency (minimize)
- calibration_error (minimize)
- bin_coverage (maximize)

Can be integrated into step17 (final_parameters_optimization) or run standalone.
"""

import optuna
import numpy as np
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from copy import deepcopy
from dataclasses import dataclass

from src.utils.logger import system_logger
from src.core.decorators import handles_errors
from src.utils.tprint import tprint_info, tprint_warning, tprint_success, tprint_error
from src.utils.common_operations import calculate_sharpe_ratio, calculate_max_drawdown

# Import Kelly components
from src.trading.sizing.dampened_kelly_engine import DampenedKellyEngine
from src.trading.sizing.kelly_history_tracker import KellyHistoryTracker
from src.training.steps.backtesting.walk_forward_kelly_validation import WalkForwardKellyValidator

logger = system_logger.getChild('KellyParametersOptimizer')


@dataclass
class OptimizationConfig:
    """Configuration for Kelly optimization."""
    global_trials: int = 150
    per_regime_trials: int = 50
    l2_penalty: float = 0.1
    min_regime_samples: int = 50
    enable_meta_learning: bool = True
    n_folds: int = 5
    train_window_months: int = 24
    test_window_months: int = 6
    timeout_hours: int = 16
    parallel_jobs: int = 4


class KellyParametersOptimizer:
    """
    Nested regime-aware optimizer for Kelly parameters.
    
    Uses hierarchical optimization:
    1. Optimize global parameters
    2. Refine per-regime with L2 regularization toward global
    3. Meta-learning across symbols (optional)
    """
    
    def __init__(
        self,
        kelly_config: Dict[str, Any],
        optimization_config: Optional[OptimizationConfig] = None,
        output_dir: str = "checkpoints/kelly_sizing"
    ):
        """
        Initialize Kelly parameters optimizer.
        
        Args:
            kelly_config: Base Kelly configuration
            optimization_config: Optimization settings
            output_dir: Output directory for artifacts
        """
        self.kelly_config = kelly_config
        self.opt_config = optimization_config or OptimizationConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger.getChild('Optimizer')
        
        # Store optimization history
        self.global_study: Optional[optuna.Study] = None
        self.regime_studies: Dict[str, optuna.Study] = {}
        
        # Best parameters
        self.best_global_params: Optional[Dict[str, Any]] = None
        self.best_regime_params: Dict[str, Dict[str, Any]] = {}
        
        tprint_info("✅ Kelly Parameters Optimizer initialized")
        tprint_info(f"  Global trials: {self.opt_config.global_trials}")
        tprint_info(f"  Per-regime trials: {self.opt_config.per_regime_trials}")
        tprint_info(f"  L2 penalty: {self.opt_config.l2_penalty}")
    
    def _create_global_objective(
        self,
        data: Any,
        signals: Any,
        returns: Any,
        regimes: Optional[Any],
        confidences: Optional[Any]
    ):
        """
        Create objective function for global parameter optimization.
        
        Returns a function that optuna will optimize.
        """
        def objective(trial: optuna.Trial) -> float:
            """
            Objective function for global parameters.
            
            Returns: Combined score from multi-objective optimization
            """
            # Sample global parameters
            params = {
                'lambda_base': trial.suggest_float('lambda_base', 0.1, 0.6),
                'beta_position': trial.suggest_float('beta_position', 0.4, 2.0),
                'beta_leverage': trial.suggest_float('beta_leverage', 0.4, 2.0),
                'prior_alpha': trial.suggest_float('prior_alpha', 5.0, 200.0, log=True),
                'ess_threshold': trial.suggest_float('ess_threshold', 10.0, 200.0),
                'entropy_threshold': trial.suggest_float('entropy_threshold', 0.5, 1.5),
                'n_min_samples': trial.suggest_int('n_min_samples', 5, 50),
                'f_floor': trial.suggest_float('f_floor', 0.001, 0.02),
                'lev_floor': trial.suggest_float('lev_floor', 1.0, 2.0),
                'decay_theta': trial.suggest_float('decay_theta', 0.85, 0.98)
            }
            
            # Sample lambda_eff components
            params['ess_sigmoid_kappa'] = trial.suggest_float('ess_sigmoid_kappa', 0.05, 0.3)
            params['entropy_scale'] = trial.suggest_float('entropy_scale', 0.3, 0.8)
            params['variance_penalty'] = trial.suggest_float('variance_penalty', 1.0, 5.0)
            
            # Sample safety parameters
            params['max_kelly_fraction'] = trial.suggest_float('max_kelly_fraction', 0.3, 0.7)
            
            # Create config with these parameters
            test_config = self._create_test_config(params, is_global=True)
            
            # Run walk-forward validation
            validator = WalkForwardKellyValidator(
                test_config,
                train_window_months=self.opt_config.train_window_months,
                test_window_months=self.opt_config.test_window_months,
                n_folds=self.opt_config.n_folds
            )
            
            # Validate full system variant
            fold_results = validator.validate_variant(
                variant_name='full_system',
                data=data,
                signals=signals,
                returns=returns,
                regimes=regimes,
                confidences=confidences
            )
            
            # Calculate multi-objective score
            score = self._calculate_multi_objective_score(fold_results, trial)
            
            return score
        
        return objective
    
    def _create_regime_objective(
        self,
        regime_id: int,
        global_params: Dict[str, Any],
        data: Any,
        signals: Any,
        returns: Any,
        regimes: Any,
        confidences: Optional[Any]
    ):
        """
        Create objective function for per-regime optimization.
        
        Includes L2 regularization toward global parameters.
        """
        def objective(trial: optuna.Trial) -> float:
            """Objective for regime-specific parameters."""
            # Sample regime-specific parameters
            params = {
                'lambda_base': trial.suggest_float('lambda_base', 0.1, 0.6),
                'beta_position': trial.suggest_float('beta_position', 0.4, 2.0),
                'beta_leverage': trial.suggest_float('beta_leverage', 0.4, 2.0),
                'prior_alpha': trial.suggest_float('prior_alpha', 5.0, 200.0, log=True),
                'ess_threshold': trial.suggest_float('ess_threshold', 10.0, 200.0),
                'entropy_threshold': trial.suggest_float('entropy_threshold', 0.5, 1.5),
                'n_min_samples': trial.suggest_int('n_min_samples', 5, 50),
                'f_floor': trial.suggest_float('f_floor', 0.001, 0.02),
                'lev_floor': trial.suggest_float('lev_floor', 1.0, 2.0),
                'decay_theta': trial.suggest_float('decay_theta', 0.85, 0.98)
            }
            
            # Create config with regime-specific parameters
            test_config = self._create_test_config(params, is_global=False, regime_id=regime_id)
            
            # Filter data to this regime only (if regime data available)
            regime_mask = (regimes == regime_id) if regimes is not None else None
            
            if regime_mask is not None and regime_mask.sum() < self.opt_config.min_regime_samples:
                # Not enough samples for this regime, use global params with high penalty
                return -1.0
            
            # Run validation (could be simplified for regime-specific)
            # For now, use walk-forward on full data but with regime-specific params
            validator = WalkForwardKellyValidator(test_config)
            fold_results = validator.validate_variant(
                'full_system', data, signals, returns, regimes, confidences
            )
            
            # Calculate base score
            base_score = self._calculate_multi_objective_score(fold_results, trial)
            
            # Add L2 regularization toward global
            l2_penalty = self._calculate_l2_penalty(params, global_params)
            
            # Combined score (subtract penalty)
            final_score = base_score - (self.opt_config.l2_penalty * l2_penalty)
            
            return final_score
        
        return objective
    
    def _calculate_multi_objective_score(
        self,
        fold_results: List[Any],
        trial: optuna.Trial
    ) -> float:
        """
        Calculate multi-objective score from fold results.
        
        Combines 6 objectives with weights:
        - geometric_mean (1.0, maximize)
        - sharpe_ratio (0.8, maximize)
        - max_drawdown (1.2, minimize)
        - high_leverage_frequency (0.5, minimize)
        - calibration_error (0.7, minimize)
        - bin_coverage (0.6, maximize)
        
        Args:
            fold_results: List of FoldResult objects
            trial: Optuna trial for logging
            
        Returns:
            Combined score (higher is better)
        """
        if not fold_results:
            return -1.0
        
        # Extract metrics
        geo_returns = [f.geometric_return for f in fold_results]
        sharpes = [f.sharpe_ratio for f in fold_results]
        max_dds = [f.max_drawdown for f in fold_results]
        cal_errors = [f.calibration_error for f in fold_results]
        coverages = [f.bin_coverage_pct for f in fold_results]
        
        # Calculate high-leverage frequency (placeholder - would need actual data)
        high_lev_freqs = [f.high_leverage_trades / f.total_trades if f.total_trades > 0 else 0.0 for f in fold_results]
        
        # Aggregate across folds (use median for robustness)
        geo_return = np.median(geo_returns)
        sharpe = np.median(sharpes)
        max_dd = np.median(max_dds)
        cal_error = np.mean(cal_errors)
        coverage = np.mean(coverages)
        high_lev_freq = np.mean(high_lev_freqs)
        
        # Log to trial
        trial.set_user_attr('geometric_return', geo_return)
        trial.set_user_attr('sharpe_ratio', sharpe)
        trial.set_user_attr('max_drawdown', max_dd)
        trial.set_user_attr('calibration_error', cal_error)
        trial.set_user_attr('bin_coverage', coverage)
        trial.set_user_attr('high_leverage_freq', high_lev_freq)
        
        # Normalize and weight objectives
        # Maximize: geometric_return, sharpe, bin_coverage
        # Minimize: max_dd, high_lev_freq, cal_error
        
        geo_score = np.clip(geo_return / 0.5, 0, 1) * 1.0  # weight=1.0
        sharpe_score = np.clip(sharpe / 3.0, 0, 1) * 0.8  # weight=0.8
        dd_score = (1 - np.clip(max_dd / 0.20, 0, 1)) * 1.2  # weight=1.2, minimize
        coverage_score = coverage * 0.6  # weight=0.6
        cal_score = (1 - np.clip(cal_error / 0.20, 0, 1)) * 0.7  # weight=0.7, minimize
        freq_score = (1 - np.clip(high_lev_freq / 0.3, 0, 1)) * 0.5  # weight=0.5, minimize
        
        # Combined score
        total_score = geo_score + sharpe_score + dd_score + coverage_score + cal_score + freq_score
        
        # Normalize to [0, 1]
        max_possible = 1.0 + 0.8 + 1.2 + 0.6 + 0.7 + 0.5
        normalized_score = total_score / max_possible
        
        return normalized_score
    
    def _calculate_l2_penalty(
        self,
        regime_params: Dict[str, Any],
        global_params: Dict[str, Any]
    ) -> float:
        """
        Calculate L2 penalty for deviation from global parameters.
        
        Args:
            regime_params: Regime-specific parameters
            global_params: Global parameters
            
        Returns:
            L2 penalty (higher = more deviation)
        """
        penalty = 0.0
        n_params = 0
        
        for key in regime_params:
            if key in global_params:
                # Normalize difference by global value to handle different scales
                global_val = global_params[key]
                regime_val = regime_params[key]
                
                if global_val != 0:
                    normalized_diff = (regime_val - global_val) / abs(global_val)
                    penalty += normalized_diff ** 2
                    n_params += 1
        
        # Average penalty
        return penalty / n_params if n_params > 0 else 0.0
    
    def _create_test_config(
        self,
        params: Dict[str, Any],
        is_global: bool = True,
        regime_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create Kelly config for testing with given parameters.
        
        Args:
            params: Parameters to test
            is_global: Whether this is global optimization
            regime_id: Regime ID (if per-regime optimization)
            
        Returns:
            Test configuration
        """
        config = deepcopy(self.kelly_config)
        
        if is_global:
            # Update global fallback
            config['global_fallback'].update(params)
            
            # Also update lambda_eff components if present
            if 'ess_sigmoid_kappa' in params:
                if 'lambda_eff_components' not in config:
                    config['lambda_eff_components'] = {}
                config['lambda_eff_components']['ess_sigmoid_kappa'] = params['ess_sigmoid_kappa']
                config['lambda_eff_components']['entropy_scale'] = params.get('entropy_scale', 0.5)
                config['lambda_eff_components']['variance_penalty'] = params.get('variance_penalty', 2.0)
            
            # Update safety limits if present
            if 'max_kelly_fraction' in params:
                if 'safety_limits' not in config:
                    config['safety_limits'] = {}
                config['safety_limits']['max_kelly_fraction'] = params['max_kelly_fraction']
        
        else:
            # Update regime-specific parameters
            regime_key = f"regime_{regime_id}"
            if 'regime_params' not in config:
                config['regime_params'] = {}
            if regime_key not in config['regime_params']:
                config['regime_params'][regime_key] = {}
            
            config['regime_params'][regime_key].update(params)
        
        return config
    
    @handles_errors
    def optimize_global_parameters(
        self,
        data: Any,
        signals: Any,
        returns: Any,
        regimes: Optional[Any] = None,
        confidences: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Optimize global fallback parameters.
        
        Args:
            data: Market data
            signals: Trading signals
            returns: Forward returns
            regimes: Regime labels
            confidences: Model confidences
            
        Returns:
            Best global parameters
        """
        tprint_info("\n" + "="*80)
        tprint_info("🎯 STEP 1: Global Parameter Optimization")
        tprint_info("="*80)
        tprint_info(f"Running {self.opt_config.global_trials} trials...")
        
        # Create study
        study_name = f"kelly_global_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.global_study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Create objective
        objective = self._create_global_objective(data, signals, returns, regimes, confidences)
        
        # Optimize
        self.global_study.optimize(
            objective,
            n_trials=self.opt_config.global_trials,
            timeout=self.opt_config.timeout_hours * 1800,  # Half time for global
            n_jobs=self.opt_config.parallel_jobs,
            show_progress_bar=True
        )
        
        # Get best parameters
        self.best_global_params = self.global_study.best_params
        
        tprint_success(f"✅ Global optimization complete")
        tprint_info(f"   Best score: {self.global_study.best_value:.4f}")
        tprint_info(f"   Best params: {self.best_global_params}")
        
        return self.best_global_params
    
    @handles_errors
    def optimize_regime_parameters(
        self,
        regime_ids: List[int],
        data: Any,
        signals: Any,
        returns: Any,
        regimes: Any,
        confidences: Optional[Any] = None
    ) -> Dict[int, Dict[str, Any]]:
        """
        Optimize per-regime parameters starting from global optimum.
        
        Args:
            regime_ids: List of regime IDs to optimize
            data: Market data
            signals: Trading signals
            returns: Forward returns
            regimes: Regime labels
            confidences: Model confidences
            
        Returns:
            Dictionary of regime_id -> best_params
        """
        if self.best_global_params is None:
            raise ValueError("Must run global optimization first")
        
        tprint_info("\n" + "="*80)
        tprint_info("🎯 STEP 2: Per-Regime Parameter Refinement")
        tprint_info("="*80)
        
        regime_params = {}
        
        for regime_id in regime_ids:
            tprint_info(f"\n📊 Optimizing Regime {regime_id}...")
            
            # Check if regime has enough samples
            regime_mask = regimes == regime_id
            regime_samples = regime_mask.sum() if hasattr(regime_mask, 'sum') else 0
            
            if regime_samples < self.opt_config.min_regime_samples:
                tprint_warning(f"  ⚠️ Regime {regime_id} has only {regime_samples} samples, using global params")
                regime_params[regime_id] = self.best_global_params.copy()
                continue
            
            # Create study for this regime
            study_name = f"kelly_regime_{regime_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            study = optuna.create_study(
                study_name=study_name,
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=42 + regime_id)
            )
            
            # Create objective with L2 regularization
            objective = self._create_regime_objective(
                regime_id, self.best_global_params, data, signals, returns, regimes, confidences
            )
            
            # Optimize
            study.optimize(
                objective,
                n_trials=self.opt_config.per_regime_trials,
                timeout=self.opt_config.timeout_hours * 900,  # Remaining time split across regimes
                n_jobs=max(1, self.opt_config.parallel_jobs // 2),
                show_progress_bar=True
            )
            
            # Store study and best params
            self.regime_studies[regime_id] = study
            regime_params[regime_id] = study.best_params
            
            tprint_success(f"  ✅ Regime {regime_id} complete: score={study.best_value:.4f}")
        
        self.best_regime_params = regime_params
        
        return regime_params
    
    def generate_pareto_configs(
        self,
        n_configs: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate Pareto frontier configurations (conservative, balanced, aggressive).
        
        Args:
            n_configs: Number of configurations to generate
            
        Returns:
            List of configuration dictionaries
        """
        tprint_info("\n" + "="*80)
        tprint_info("📊 Generating Pareto Frontier Configurations")
        tprint_info("="*80)
        
        if self.global_study is None or not self.best_global_params:
            raise ValueError("Must run optimization first")
        
        # Get top N trials from global study
        top_trials = sorted(
            self.global_study.trials,
            key=lambda t: t.value if t.value is not None else -np.inf,
            reverse=True
        )[:n_configs * 3]  # Get more to filter
        
        # Filter for diversity in risk profiles
        configs = []
        
        # Conservative: Low DD, high Sharpe, low high-lev frequency
        conservative_trials = [
            t for t in top_trials
            if t.user_attrs.get('max_drawdown', 1.0) < 0.10
            and t.user_attrs.get('sharpe_ratio', 0.0) > 1.5
        ]
        if conservative_trials:
            configs.append(self._create_config_from_trial(conservative_trials[0], 'conservative'))
        
        # Balanced: Middle ground
        balanced_trials = [
            t for t in top_trials
            if 0.10 <= t.user_attrs.get('max_drawdown', 0.0) <= 0.15
            and t.user_attrs.get('sharpe_ratio', 0.0) > 1.0
        ]
        if balanced_trials:
            configs.append(self._create_config_from_trial(balanced_trials[0], 'balanced'))
        
        # Aggressive: High return, higher DD acceptable
        aggressive_trials = [
            t for t in top_trials
            if t.user_attrs.get('geometric_return', 0.0) > 0.30
            and t.user_attrs.get('max_drawdown', 0.0) < 0.20
        ]
        if aggressive_trials:
            configs.append(self._create_config_from_trial(aggressive_trials[0], 'aggressive'))
        
        # If we don't have all 3, fill with best trials
        while len(configs) < n_configs and len(top_trials) > len(configs):
            trial = top_trials[len(configs)]
            config_type = ['conservative', 'balanced', 'aggressive'][len(configs)]
            configs.append(self._create_config_from_trial(trial, config_type))
        
        # Save configs
        self._save_pareto_configs(configs)
        
        for i, config in enumerate(configs):
            tprint_info(f"\n  Config {i+1}: {config['config_type']}")
            tprint_info(f"    Sharpe: {config['metrics']['sharpe']:.2f}")
            tprint_info(f"    Geo Return: {config['metrics']['geometric_return']:.2%}")
            tprint_info(f"    Max DD: {config['metrics']['max_drawdown']:.2%}")
        
        return configs
    
    def _create_config_from_trial(self, trial: optuna.Trial, config_type: str) -> Dict[str, Any]:
        """Create full config from trial."""
        return {
            'config_type': config_type,
            'global_params': trial.params,
            'regime_params': self.best_regime_params.copy() if self.best_regime_params else {},
            'metrics': {
                'sharpe': trial.user_attrs.get('sharpe_ratio', 0.0),
                'geometric_return': trial.user_attrs.get('geometric_return', 0.0),
                'max_drawdown': trial.user_attrs.get('max_drawdown', 0.0),
                'calibration_error': trial.user_attrs.get('calibration_error', 0.0),
                'bin_coverage': trial.user_attrs.get('bin_coverage', 0.0),
                'high_leverage_freq': trial.user_attrs.get('high_leverage_freq', 0.0)
            },
            'trial_number': trial.number,
            'score': trial.value
        }
    
    def _save_pareto_configs(self, configs: List[Dict[str, Any]]) -> Path:
        """Save Pareto configurations."""
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = self.output_dir / f"kelly_pareto_configs_{timestamp_str}.json"
        
        with open(filepath, 'w') as f:
            json.dump(configs, f, indent=2)
        
        tprint_success(f"✅ Pareto configs saved: {filepath}")
        return filepath


def optimize_kelly_parameters(
    symbol: str,
    timeframe: str,
    data: Any,
    signals: Any,
    returns: Any,
    regimes: Optional[Any] = None,
    confidences: Optional[Any] = None,
    kelly_config_path: str = "src/config/kelly_sizing_config.yaml",
    optimization_config: Optional[OptimizationConfig] = None
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Run complete Kelly parameter optimization.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        data: Market data
        signals: Trading signals
        returns: Forward returns
        regimes: Regime labels
        confidences: Model confidences
        kelly_config_path: Path to Kelly config
        optimization_config: Optimization settings
        
    Returns:
        Tuple of (best_global_params, pareto_configs)
    """
    # Load config
    with open(kelly_config_path, 'r') as f:
        kelly_config = yaml.safe_load(f)['dampened_kelly']
    
    # Create optimizer
    optimizer = KellyParametersOptimizer(kelly_config, optimization_config)
    
    # Step 1: Optimize global parameters
    global_params = optimizer.optimize_global_parameters(data, signals, returns, regimes, confidences)
    
    # Step 2: Optimize per-regime (if regimes available)
    if regimes is not None:
        unique_regimes = np.unique(regimes)
        regime_params = optimizer.optimize_regime_parameters(
            regime_ids=list(unique_regimes),
            data=data,
            signals=signals,
            returns=returns,
            regimes=regimes,
            confidences=confidences
        )
    
    # Step 3: Generate Pareto configs
    pareto_configs = optimizer.generate_pareto_configs(n_configs=3)
    
    return global_params, pareto_configs

