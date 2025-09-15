"""
Enhanced Final Parameters Optimization with Non-Linear Transformations

This module extends the existing final_parameters_optimization.py with advanced
non-linear optimization techniques including logarithmic, fractional power,
and adaptive transformations.

Key Enhancements:
- Log-space parameter sampling for better exploration
- Fractional power transformations for non-linear scaling
- Adaptive optimization strategies
- Multi-scale parameter optimization
- Enhanced convergence monitoring
"""

import numpy as np
import optuna
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import time
from datetime import datetime

# Import the original optimizer
from src.training.steps.backtesting.final_parameters_optimization import FinalParametersOptimizer

logger = logging.getLogger(__name__)

@dataclass
class NonLinearConfig:
    """Configuration for non-linear optimization methods."""
    use_log_sampling: bool = True
    use_fractional_powers: bool = True
    use_adaptive_transforms: bool = True
    power_exponents: List[float] = None
    log_threshold: float = 10.0  # Use log sampling for ranges > this value
    sigmoid_range: Tuple[float, float] = (-6.0, 6.0)
    
    def __post_init__(self):
        if self.power_exponents is None:
            self.power_exponents = [0.3, 0.5, 0.7, 0.9]

class EnhancedFinalParametersOptimizer(FinalParametersOptimizer):
    """
    Enhanced final parameters optimizer with non-linear transformations.
    
    Extends the original FinalParametersOptimizer with advanced non-linear
    optimization techniques for better parameter exploration and convergence.
    """
    
    def __init__(self, config: Dict[str, Any], nonlinear_config: Optional[NonLinearConfig] = None):
        """Initialize the enhanced optimizer."""
        super().__init__(config)
        
        self.nonlinear_config = nonlinear_config or NonLinearConfig()
        self.logger = logger.getChild('EnhancedFinalParametersOptimizer')
        
        self.logger.info("🚀 Enhanced Final Parameters Optimizer initialized")
        self.logger.info(f"📊 Non-linear config: log_sampling={self.nonlinear_config.use_log_sampling}, "
                        f"fractional_powers={self.nonlinear_config.use_fractional_powers}, "
                        f"adaptive={self.nonlinear_config.use_adaptive_transforms}")
        
        # Enhanced search spaces with non-linear transformations
        self.enhanced_search_spaces = self._create_enhanced_search_spaces()
    
    def _create_enhanced_search_spaces(self) -> Dict[str, Dict[str, Any]]:
        """Create enhanced search spaces with non-linear transformations."""
        enhanced_spaces = {}
        
        for category, space in self.default_search_spaces.items():
            enhanced_space = {}
            
            for param_name, param_config in space.items():
                if param_config['type'] == 'float':
                    min_val = param_config['min']
                    max_val = param_config['max']
                    range_size = max_val - min_val
                    
                    # Determine optimal transformation based on parameter characteristics
                    if range_size > self.nonlinear_config.log_threshold and min_val > 0:
                        # Use log sampling for large positive ranges
                        enhanced_space[param_name] = {
                            'type': 'log_float',
                            'min': min_val,
                            'max': max_val,
                            'log_min': np.log(min_val),
                            'log_max': np.log(max_val)
                        }
                    elif 0 < min_val < 1 and max_val > 1:
                        # Use fractional power for [0,1] to larger ranges
                        enhanced_space[param_name] = {
                            'type': 'fractional_power_float',
                            'min': min_val,
                            'max': max_val,
                            'power': 0.5
                        }
                    elif min_val >= 0 and max_val <= 1:
                        # Use sigmoid for bounded [0,1] ranges
                        enhanced_space[param_name] = {
                            'type': 'sigmoid_float',
                            'min': min_val,
                            'max': max_val,
                            'sigmoid_range': self.nonlinear_config.sigmoid_range
                        }
                    else:
                        # Use adaptive transformation
                        enhanced_space[param_name] = {
                            'type': 'adaptive_float',
                            'min': min_val,
                            'max': max_val,
                            'range_size': range_size
                        }
                else:
                    # Keep integer and boolean parameters as-is
                    enhanced_space[param_name] = param_config
            
            enhanced_spaces[category] = enhanced_space
        
        return enhanced_spaces
    
    def _sample_enhanced_parameter(self, trial: optuna.Trial, param_name: str, 
                                 param_config: Dict[str, Any]) -> Any:
        """Sample parameter using enhanced non-linear transformations."""
        param_type = param_config['type']
        
        if param_type == 'log_float':
            # Logarithmic sampling
            log_param = trial.suggest_float(
                f"log_{param_name}",
                param_config['log_min'],
                param_config['log_max']
            )
            return np.exp(log_param)
        
        elif param_type == 'fractional_power_float':
            # Fractional power sampling
            raw_param = trial.suggest_float(f"raw_{param_name}", 0.0, 1.0)
            power = param_config.get('power', 0.5)
            transformed = raw_param ** power
            return param_config['min'] + transformed * (param_config['max'] - param_config['min'])
        
        elif param_type == 'sigmoid_float':
            # Sigmoid sampling
            raw_param = trial.suggest_float(
                f"raw_{param_name}",
                param_config['sigmoid_range'][0],
                param_config['sigmoid_range'][1]
            )
            sigmoid_param = 1 / (1 + np.exp(-raw_param))
            return param_config['min'] + sigmoid_param * (param_config['max'] - param_config['min'])
        
        elif param_type == 'adaptive_float':
            # Adaptive transformation based on range characteristics
            range_size = param_config['range_size']
            min_val = param_config['min']
            max_val = param_config['max']
            
            if range_size > self.nonlinear_config.log_threshold and min_val > 0:
                # Use log sampling
                log_min = np.log(min_val)
                log_max = np.log(max_val)
                log_param = trial.suggest_float(f"log_{param_name}", log_min, log_max)
                return np.exp(log_param)
            elif range_size < 1.0:
                # Use sigmoid for small ranges
                raw_param = trial.suggest_float(f"raw_{param_name}", -6, 6)
                sigmoid_param = 1 / (1 + np.exp(-raw_param))
                return min_val + sigmoid_param * (max_val - min_val)
            else:
                # Use fractional power for medium ranges
                raw_param = trial.suggest_float(f"raw_{param_name}", 0.0, 1.0)
                power = 0.7  # Slightly more aggressive than 0.5
                transformed = raw_param ** power
                return min_val + transformed * (max_val - min_val)
        
        else:
            # Fallback to original sampling
            if param_config['type'] == 'float':
                return trial.suggest_float(param_name, param_config['min'], param_config['max'])
            elif param_config['type'] == 'int':
                return trial.suggest_int(param_name, param_config['min'], param_config['max'])
            elif param_config['type'] == 'bool':
                return trial.suggest_categorical(param_name, [True, False])
    
    def _enhanced_objective_function(self, trial: optuna.Trial, category: str, 
                                   search_space: Dict[str, Dict[str, Any]], 
                                   calibration_results: Dict[str, Any]) -> float:
        """
        Enhanced objective function with non-linear parameter sampling.
        
        Args:
            trial: Optuna trial object
            category: Parameter category being optimized
            search_space: Enhanced search space for the category
            calibration_results: Results from confidence calibration
            
        Returns:
            Optimization score (higher is better)
        """
        try:
            params = {}
            
            # Sample parameters using enhanced transformations
            for param_name, param_config in search_space.items():
                params[param_name] = self._sample_enhanced_parameter(trial, param_name, param_config)
            
            # Evaluate configuration
            score = self._evaluate_configuration(category, params, calibration_results)
            
            # Apply non-linear scoring enhancements
            enhanced_score = self._apply_nonlinear_scoring(score, params, category)
            
            return enhanced_score
            
        except Exception as e:
            self.logger.error(f"Error in enhanced objective function for {category}: {e}")
            return -999.0
    
    def _apply_nonlinear_scoring(self, base_score: float, params: Dict[str, Any], 
                               category: str) -> float:
        """Apply non-linear scoring enhancements."""
        try:
            enhanced_score = base_score
            
            # Apply confidence-based non-linear scaling
            if category == 'confidence' and 'base_entry_threshold' in params:
                threshold = params['base_entry_threshold']
                # Non-linear confidence scaling: higher confidence gets exponential bonus
                confidence_bonus = np.exp(threshold - 0.5) - 1
                enhanced_score += confidence_bonus * 0.1
            
            # Apply position sizing non-linear scaling
            elif category == 'position_sizing' and 'base_position_size' in params:
                position_size = params['base_position_size']
                # Risk-adjusted scoring: penalize very large positions exponentially
                risk_penalty = (position_size ** 2.5) * 0.2
                enhanced_score -= risk_penalty
            
            # Apply leverage non-linear scaling
            elif category == 'leverage' and 'safe_leverage_multiplier' in params:
                leverage = params['safe_leverage_multiplier']
                # Leverage scoring: optimal around 0.7-0.8, penalize extremes
                optimal_leverage = 0.75
                leverage_penalty = ((leverage - optimal_leverage) ** 2) * 2.0
                enhanced_score -= leverage_penalty
            
            # Apply ensemble weight non-linear scaling
            elif category == 'ensemble':
                weights = [params.get(key, 0) for key in ['analyst_weight', 'tactician_weight', 'strategist_weight']]
                if all(w > 0 for w in weights):
                    # Diversity bonus: reward balanced weights
                    weight_entropy = -sum(w * np.log(w) for w in weights if w > 0)
                    diversity_bonus = weight_entropy * 0.1
                    enhanced_score += diversity_bonus
            
            return enhanced_score
            
        except Exception as e:
            self.logger.warning(f"Non-linear scoring failed: {e}")
            return base_score
    
    async def _optimize_category_enhanced(self, category: str, calibration_results: Dict[str, Any], 
                                        previous_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enhanced category optimization with non-linear transformations.
        
        Args:
            category: Parameter category to optimize
            calibration_results: Results from confidence calibration
            previous_results: Previous optimization results for this category
            
        Returns:
            Dict containing enhanced optimization results for the category
        """
        try:
            self.logger.info(f"🔍 Enhanced optimization for category: {category}")
            
            # Use enhanced search space
            search_space = self.enhanced_search_spaces.get(category, {})
            if not search_space:
                self.logger.warning(f"⚠️ No enhanced search space found for category: {category}")
                # Fallback to original method
                return await self._optimize_category(category, calibration_results, previous_results)
            
            self.logger.info(f"📊 Enhanced search space parameters: {len(search_space)}")
            for param_name, param_config in search_space.items():
                self.logger.debug(f"   • {param_name}: {param_config['type']} "
                                f"[{param_config.get('min', 'N/A')}-{param_config.get('max', 'N/A')}]")
            
            study_name = f'{self.study_name}_{category}_enhanced'
            self.logger.info(f"📝 Creating enhanced Optuna study: {study_name}")
            
            # Create study with enhanced sampler
            sampler = optuna.samplers.TPESampler(
                n_startup_trials=10,
                n_ei_candidates=24,
                gamma=lambda x: min(int(0.25 * x), 25),
                prior_weight=1.0,
                consider_magic_clip=True,
                consider_endpoints=True
            )
            
            study = optuna.create_study(
                study_name=study_name,
                direction='maximize',
                sampler=sampler,
                storage='sqlite:///optuna_studies_enhanced.db',
                load_if_exists=True
            )
            
            self.logger.info(f"🎯 Starting enhanced optimization with {self.n_trials} trials")
            
            def objective(trial):
                return self._enhanced_objective_function(trial, category, search_space, calibration_results)
            
            # Run optimization with enhanced monitoring
            start_time = time.time()
            study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
            optimization_time = time.time() - start_time
            
            best_params = study.best_params
            best_value = study.best_value
            
            # Convert log parameters back to original space for reporting
            converted_params = self._convert_parameters_to_original_space(best_params, search_space)
            
            self.logger.info(f"🏆 Enhanced optimization results for {category}:")
            for param, value in converted_params.items():
                self.logger.info(f"   • {param}: {value}")
            self.logger.info(f"📈 Best objective value: {best_value:.4f}")
            self.logger.info(f"⏱️ Optimization time: {optimization_time:.2f}s")
            
            # Enhanced convergence analysis
            convergence_analysis = self._analyze_convergence(study)
            
            return {
                'best_params': converted_params,
                'best_value': best_value,
                'study_name': study_name,
                'n_trials': self.n_trials,
                'optimization_time': optimization_time,
                'convergence_analysis': convergence_analysis,
                'enhanced_methods_used': self._get_used_enhancement_methods(search_space)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced optimization failed for category {category}: {e}")
            self.logger.exception("Full traceback:")
            # Fallback to original optimization
            return await self._optimize_category(category, calibration_results, previous_results)
    
    def _convert_parameters_to_original_space(self, params: Dict[str, Any], 
                                           search_space: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Convert enhanced parameters back to original parameter space."""
        converted_params = {}
        
        for param_name, value in params.items():
            # Remove prefixes added by enhanced sampling
            original_name = param_name
            if param_name.startswith('log_'):
                original_name = param_name[4:]
            elif param_name.startswith('raw_'):
                original_name = param_name[4:]
            
            if original_name in search_space:
                param_config = search_space[original_name]
                
                if param_config['type'] in ['log_float', 'fractional_power_float', 'sigmoid_float', 'adaptive_float']:
                    # These are already converted to original space in the objective function
                    converted_params[original_name] = value
                else:
                    converted_params[original_name] = value
            else:
                converted_params[param_name] = value
        
        return converted_params
    
    def _analyze_convergence(self, study: optuna.Study) -> Dict[str, Any]:
        """Analyze convergence characteristics of the optimization."""
        try:
            if len(study.trials) < 5:
                return {'convergence_quality': 'insufficient_data'}
            
            values = [t.value for t in study.trials if t.value is not None]
            if not values:
                return {'convergence_quality': 'no_valid_trials'}
            
            # Calculate convergence metrics
            best_values = []
            current_best = float('-inf')
            for value in values:
                if value > current_best:
                    current_best = value
                best_values.append(current_best)
            
            # Improvement rate
            total_improvement = best_values[-1] - best_values[0]
            improvement_rate = total_improvement / len(values) if len(values) > 0 else 0
            
            # Convergence stability (variance in last 20% of trials)
            last_portion = int(len(best_values) * 0.2)
            if last_portion > 1:
                recent_values = best_values[-last_portion:]
                convergence_variance = np.var(recent_values)
            else:
                convergence_variance = 0
            
            # Convergence quality assessment
            if improvement_rate > 0.01 and convergence_variance < 0.001:
                convergence_quality = 'excellent'
            elif improvement_rate > 0.005 and convergence_variance < 0.01:
                convergence_quality = 'good'
            elif improvement_rate > 0.001:
                convergence_quality = 'fair'
            else:
                convergence_quality = 'poor'
            
            return {
                'convergence_quality': convergence_quality,
                'total_improvement': total_improvement,
                'improvement_rate': improvement_rate,
                'convergence_variance': convergence_variance,
                'final_best_value': best_values[-1],
                'n_trials': len(values)
            }
            
        except Exception as e:
            self.logger.warning(f"Convergence analysis failed: {e}")
            return {'convergence_quality': 'analysis_failed', 'error': str(e)}
    
    def _get_used_enhancement_methods(self, search_space: Dict[str, Dict[str, Any]]) -> List[str]:
        """Get list of enhancement methods used in the search space."""
        methods = set()
        for param_config in search_space.values():
            if param_config['type'] in ['log_float', 'fractional_power_float', 'sigmoid_float', 'adaptive_float']:
                methods.add(param_config['type'])
        return list(methods)
    
    async def optimize_all_parameters_enhanced(self, calibration_results: Dict[str, Any], 
                                            previous_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enhanced optimization of all parameters with non-linear transformations.
        
        Args:
            calibration_results: Results from confidence calibration
            previous_results: Previous optimization results for warm start
            
        Returns:
            Dict containing enhanced optimization results by category
        """
        try:
            self.logger.info("🔧 Starting enhanced final parameters optimization...")
            self.logger.info(f"📊 Calibration results available: {len(calibration_results)} keys")
            self.logger.info(f"🔄 Previous results available: {previous_results is not None}")
            
            optimization_results = {}
            start_time = time.time()
            
            for i, category in enumerate(self.categories, 1):
                self.logger.info(f"🔄 Enhanced optimization {category} parameters ({i}/{len(self.categories)})...")
                category_start = time.time()
                
                category_results = await self._optimize_category_enhanced(
                    category, calibration_results, 
                    previous_results.get(category) if previous_results else None
                )
                
                category_duration = time.time() - category_start
                optimization_results[category] = category_results
                
                if category_results and 'best_value' in category_results:
                    self.logger.info(f"✅ {category} enhanced optimization completed in {category_duration:.2f}s - "
                                   f"Best value: {category_results['best_value']:.4f}")
                    if 'enhanced_methods_used' in category_results:
                        methods = category_results['enhanced_methods_used']
                        self.logger.info(f"   🚀 Enhancement methods used: {', '.join(methods)}")
                else:
                    self.logger.warning(f"⚠️ {category} enhanced optimization completed in {category_duration:.2f}s - "
                                      f"No results obtained")
            
            total_duration = time.time() - start_time
            self.logger.info("✅ Enhanced final parameters optimization completed")
            self.logger.info(f"⏱️ Total optimization time: {total_duration:.2f}s")
            self.logger.info(f"📊 Categories optimized: {len(optimization_results)}")
            
            # Add enhancement summary
            optimization_results['_enhancement_summary'] = {
                'total_optimization_time': total_duration,
                'categories_optimized': len(optimization_results),
                'enhancement_config': {
                    'use_log_sampling': self.nonlinear_config.use_log_sampling,
                    'use_fractional_powers': self.nonlinear_config.use_fractional_powers,
                    'use_adaptive_transforms': self.nonlinear_config.use_adaptive_transforms
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"❌ Error in enhanced final parameters optimization: {e}")
            self.logger.exception("Full traceback:")
            raise

# Convenience function for enhanced optimization
async def optimize_final_parameters_enhanced(calibration_results: Dict[str, Any], 
                                           config: Dict[str, Any],
                                           nonlinear_config: Optional[NonLinearConfig] = None,
                                           symbol: str = "ETHUSDT",
                                           exchange: str = "BINANCE",
                                           data_dir: str = "data/training") -> Dict[str, Any]:
    """
    Enhanced convenience function to optimize final parameters with non-linear transformations.
    
    Args:
        calibration_results: Results from confidence calibration
        config: Configuration dictionary
        nonlinear_config: Non-linear optimization configuration
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory
        
    Returns:
        Enhanced optimization results
    """
    optimizer = EnhancedFinalParametersOptimizer(config, nonlinear_config)
    
    # Load previous results for warm start
    previous_results = await optimizer.load_optimization_results(symbol, exchange, data_dir)
    
    # Optimize all parameters with enhancements
    optimization_results = await optimizer.optimize_all_parameters_enhanced(
        calibration_results, previous_results
    )
    
    # Validate results
    validation_passed = await optimizer.validate_optimization_results(optimization_results)
    if not validation_passed:
        logger.warning('⚠️ Enhanced optimization results validation failed, using fallback parameters')
    
    # Save results
    await optimizer.save_optimization_results(optimization_results, symbol, exchange, data_dir)
    
    # Generate enhanced report
    start_time = datetime.now()
    report = await optimizer.generate_optimization_report(optimization_results, start_time)
    
    return {
        'final_parameters': optimization_results,
        'optimization_report': report,
        'validation_passed': validation_passed,
        'enhancement_summary': optimization_results.get('_enhancement_summary', {})
    }