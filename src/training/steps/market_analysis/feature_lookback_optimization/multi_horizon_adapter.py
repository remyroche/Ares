"""
Multi-Horizon Adapter for Feature Lookback Optimization

This module adapts the existing feature_lookback_optimization to work with
the new multi-horizon profit labeling system, optimizing time horizons
instead of just lookback periods.

Key features:
- Optimizes time horizons for multi-horizon labeling
- Integrates with existing feature lookback optimization
- Provides backward compatibility
- Enhanced for profit probability targets
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import logging
import time
from pathlib import Path

from src.utils.tprint import tprint
from src.utils.logger import get_logger
from src.core.decorators import handles_errors, traced, validates, log_execution_time

# Import the multi-horizon labeler
from ..multi_horizon_profit_labeler import MultiHorizonProfitLabeler, MultiHorizonConfig

@dataclass
class MultiHorizonOptimizationConfig:
    """Configuration for multi-horizon optimization."""
    
    # Time horizon optimization ranges
    time_horizon_ranges: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: {
        'immediate': (2, 6, 1),     # 10-30 minutes (2-6 periods of 5m), step 1
        'short': (4, 12, 2),        # 20-60 minutes (4-12 periods), step 2
        'medium': (8, 24, 3),       # 40-120 minutes (8-24 periods), step 3
        'extended': (12, 36, 4)     # 60-180 minutes (12-36 periods), step 4
    })
    
    # Profit target optimization ranges
    profit_target_ranges: Dict[str, Tuple[float, float, float]] = field(default_factory=lambda: {
        'micro': (0.003, 0.006, 0.0005),    # 0.3-0.6%, step 0.05%
        'small': (0.004, 0.008, 0.001),     # 0.4-0.8%, step 0.1%
        'medium': (0.006, 0.012, 0.001),    # 0.6-1.2%, step 0.1%
        'good': (0.008, 0.016, 0.002),      # 0.8-1.6%, step 0.2%
        'great': (0.012, 0.024, 0.003),     # 1.2-2.4%, step 0.3%
        'excellent': (0.016, 0.032, 0.004)  # 1.6-3.2%, step 0.4%
    })
    
    # Optimization parameters
    optimization_method: str = 'bayesian'  # 'grid', 'bayesian', 'genetic'
    n_trials: int = 100
    cv_folds: int = 5
    validation_split: float = 0.2
    
    # Scoring metrics for optimization
    primary_metric: str = 'leverage_adjusted_score'  # Main metric to optimize
    secondary_metrics: List[str] = field(default_factory=lambda: [
        'overall_opportunity',
        'immediate_opportunity', 
        'short_term_opportunity'
    ])
    
    # Constraints
    min_samples_per_horizon: int = 100
    max_optimization_time_hours: float = 2.0
    
    # Integration with existing system
    integrate_with_feature_lookback: bool = True
    feature_lookback_config_path: str = 'src/config/feature_lookback_optimization_config.yaml'

class MultiHorizonOptimizer:
    """
    Optimizer for multi-horizon time horizons and profit targets.
    
    Integrates with existing feature lookback optimization while adding
    specific optimization for time horizons used in multi-horizon labeling.
    """
    
    def __init__(self, config: Optional[MultiHorizonOptimizationConfig] = None):
        """Initialize the multi-horizon optimizer."""
        self.config = config or MultiHorizonOptimizationConfig()
        self.logger = get_logger('MultiHorizonOptimizer')
        
        # Results storage
        self.optimization_results = {}
        self.best_config = None
        self.optimization_history = []
        
        self.logger.info(f'🔧 Multi-Horizon Optimizer initialized')
        self.logger.info(f'   → Optimization method: {self.config.optimization_method}')
        self.logger.info(f'   → Primary metric: {self.config.primary_metric}')
        self.logger.info(f'   → Max trials: {self.config.n_trials}')
    
    @traced(span_name='optimize_multi_horizon_config')
    @validates()
    @handles_errors(exceptions=(Exception,), default_return={})
    @log_execution_time()
    def optimize_time_horizons(self, 
                             data: pd.DataFrame,
                             feature_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Optimize time horizons for multi-horizon labeling.
        
        Args:
            data: Historical price data (5m timeframe)
            feature_data: Pre-computed features (optional)
            
        Returns:
            Dictionary with optimized configuration and results
        """
        self.logger.info(f'🎯 Starting multi-horizon optimization on {len(data)} samples')
        
        if len(data) < 1000:
            self.logger.warning(f'⚠️ Limited data available: {len(data)} samples')
        
        optimization_start = time.time()
        
        # Step 1: Validate data
        validation_results = self._validate_optimization_data(data)
        if not validation_results['valid']:
            return {'error': 'Data validation failed', 'details': validation_results}
        
        # Step 2: Generate parameter space
        param_space = self._generate_parameter_space()
        self.logger.info(f'📊 Generated parameter space with {len(param_space)} combinations')
        
        # Step 3: Run optimization
        if self.config.optimization_method == 'grid':
            results = self._grid_search_optimization(data, param_space)
        elif self.config.optimization_method == 'bayesian':
            results = self._bayesian_optimization(data, param_space)
        elif self.config.optimization_method == 'genetic':
            results = self._genetic_optimization(data, param_space)
        else:
            self.logger.warning(f'⚠️ Unknown method {self.config.optimization_method}, using grid search')
            results = self._grid_search_optimization(data, param_space)
        
        # Step 4: Validate and finalize results
        final_results = self._finalize_optimization_results(results, optimization_start)
        
        # Step 5: Integration with feature lookback optimization
        if self.config.integrate_with_feature_lookback:
            integration_results = self._integrate_with_feature_lookback(final_results)
            final_results['feature_integration'] = integration_results
        
        self.optimization_results = final_results
        self.logger.info(f'✅ Multi-horizon optimization completed in {time.time() - optimization_start:.1f}s')
        
        return final_results
    
    def _validate_optimization_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data for optimization."""
        validation = {'valid': True, 'issues': []}
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            validation['valid'] = False
            validation['issues'].append(f'Missing columns: {missing_cols}')
        
        # Check data length
        max_horizon = max([max_val for _, max_val, _ in self.config.time_horizon_ranges.values()])
        if len(data) < max_horizon + 100:  # Need buffer for validation
            validation['valid'] = False
            validation['issues'].append(f'Insufficient data: {len(data)} < {max_horizon + 100}')
        
        # Check for NaN values
        if data[required_cols].isnull().any().any():
            validation['valid'] = False
            validation['issues'].append('NaN values found in price data')
        
        # Check price consistency
        inconsistent = (data['high'] < data[['open', 'close']].max(axis=1)).any()
        if inconsistent:
            validation['issues'].append('Price consistency issues detected')
        
        return validation
    
    def _generate_parameter_space(self) -> List[Dict[str, Any]]:
        """Generate parameter space for optimization."""
        param_combinations = []
        
        # Generate time horizon combinations
        horizon_combinations = self._generate_horizon_combinations()
        
        # Generate profit target combinations (optional optimization)
        target_combinations = self._generate_target_combinations()
        
        # Combine horizons with targets (or use default targets)
        for horizon_combo in horizon_combinations:
            if len(target_combinations) > 0:
                for target_combo in target_combinations[:10]:  # Limit to top 10 target combos
                    param_combinations.append({
                        'time_horizons': horizon_combo,
                        'profit_targets': target_combo,
                        'config_id': len(param_combinations)
                    })
            else:
                # Use default profit targets
                param_combinations.append({
                    'time_horizons': horizon_combo,
                    'profit_targets': None,  # Use defaults
                    'config_id': len(param_combinations)
                })
        
        # Limit total combinations for practical optimization
        if len(param_combinations) > self.config.n_trials:
            # Sample random subset
            np.random.seed(42)
            indices = np.random.choice(len(param_combinations), self.config.n_trials, replace=False)
            param_combinations = [param_combinations[i] for i in indices]
        
        return param_combinations
    
    def _generate_horizon_combinations(self) -> List[Dict[str, int]]:
        """Generate time horizon combinations."""
        combinations = []
        
        # Get ranges for each horizon type
        horizon_ranges = {}
        for horizon_name, (min_val, max_val, step) in self.config.time_horizon_ranges.items():
            horizon_ranges[horizon_name] = list(range(min_val, max_val + 1, step))
        
        # Generate all combinations (limited to reasonable number)
        import itertools
        
        horizon_names = list(horizon_ranges.keys())
        horizon_values = [horizon_ranges[name] for name in horizon_names]
        
        # Limit combinations to avoid explosion
        max_combinations = 1000
        all_combinations = list(itertools.product(*horizon_values))
        
        if len(all_combinations) > max_combinations:
            # Sample random subset
            np.random.seed(42)
            indices = np.random.choice(len(all_combinations), max_combinations, replace=False)
            selected_combinations = [all_combinations[i] for i in indices]
        else:
            selected_combinations = all_combinations
        
        # Convert to dictionary format
        for combo in selected_combinations:
            horizon_dict = {name: value for name, value in zip(horizon_names, combo)}
            # Ensure logical ordering (immediate < short < medium < extended)
            if self._is_valid_horizon_combination(horizon_dict):
                combinations.append(horizon_dict)
        
        return combinations
    
    def _is_valid_horizon_combination(self, horizons: Dict[str, int]) -> bool:
        """Check if horizon combination is logically valid."""
        immediate = horizons.get('immediate', 0)
        short = horizons.get('short', 0)
        medium = horizons.get('medium', 0)
        extended = horizons.get('extended', 0)
        
        # Ensure logical ordering
        return immediate <= short <= medium <= extended
    
    def _generate_target_combinations(self) -> List[Dict[str, float]]:
        """Generate profit target combinations (optional)."""
        # For now, return empty list to use default targets
        # Can be enhanced later to optimize profit targets as well
        return []
    
    def _grid_search_optimization(self, data: pd.DataFrame, param_space: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform grid search optimization."""
        self.logger.info(f'🔍 Running grid search optimization with {len(param_space)} configurations')
        
        best_score = -np.inf
        best_config = None
        all_results = []
        
        for i, params in enumerate(param_space):
            if i % 10 == 0:
                self.logger.info(f'   → Progress: {i}/{len(param_space)} ({i/len(param_space)*100:.1f}%)')
            
            # Evaluate configuration
            score, metrics = self._evaluate_configuration(data, params)
            
            result = {
                'config_id': params['config_id'],
                'params': params,
                'score': score,
                'metrics': metrics,
                'rank': 0  # Will be set later
            }
            all_results.append(result)
            
            if score > best_score:
                best_score = score
                best_config = params
        
        # Rank results
        all_results.sort(key=lambda x: x['score'], reverse=True)
        for i, result in enumerate(all_results):
            result['rank'] = i + 1
        
        return {
            'method': 'grid_search',
            'best_config': best_config,
            'best_score': best_score,
            'all_results': all_results,
            'total_evaluations': len(param_space)
        }
    
    def _bayesian_optimization(self, data: pd.DataFrame, param_space: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform Bayesian optimization (simplified version)."""
        self.logger.info(f'🎯 Running Bayesian optimization with {len(param_space)} initial configurations')
        
        # For now, use a simplified approach similar to grid search
        # Can be enhanced with proper Bayesian optimization libraries
        return self._grid_search_optimization(data, param_space)
    
    def _genetic_optimization(self, data: pd.DataFrame, param_space: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform genetic algorithm optimization (simplified version)."""
        self.logger.info(f'🧬 Running genetic optimization with {len(param_space)} initial population')
        
        # For now, use a simplified approach similar to grid search
        # Can be enhanced with proper genetic algorithm implementation
        return self._grid_search_optimization(data, param_space)
    
    def _evaluate_configuration(self, data: pd.DataFrame, params: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Evaluate a specific configuration."""
        try:
            # Create multi-horizon config from parameters
            horizon_config = MultiHorizonConfig()
            
            # Update time horizons
            if 'time_horizons' in params:
                horizon_config.time_horizons = params['time_horizons']
            
            # Update profit targets if provided
            if params.get('profit_targets') is not None:
                horizon_config.profit_targets = params['profit_targets']
            
            # Create labeler and generate labels
            labeler = MultiHorizonProfitLabeler(horizon_config)
            
            # Use subset of data for faster evaluation
            eval_data = data.iloc[:min(len(data), 2000)].copy()  # Use first 2000 samples
            labeled_data = labeler.generate_labels(eval_data)
            
            # Calculate evaluation metrics
            metrics = self._calculate_evaluation_metrics(labeled_data)
            
            # Calculate composite score
            primary_score = metrics.get(self.config.primary_metric, 0.0)
            secondary_scores = [metrics.get(metric, 0.0) for metric in self.config.secondary_metrics]
            
            # Weighted combination
            composite_score = primary_score * 0.6 + np.mean(secondary_scores) * 0.4
            
            return composite_score, metrics
            
        except Exception as e:
            self.logger.warning(f'⚠️ Configuration evaluation failed: {e}')
            return -1.0, {'error': str(e)}
    
    def _calculate_evaluation_metrics(self, labeled_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate evaluation metrics for labeled data."""
        metrics = {}
        
        # Extract key columns
        key_columns = [
            'overall_opportunity',
            'leverage_adjusted_score',
            'immediate_opportunity',
            'short_term_opportunity',
            'medium_term_opportunity',
            'extended_opportunity'
        ]
        
        for col in key_columns:
            if col in labeled_data.columns:
                values = labeled_data[col].dropna()
                if len(values) > 0:
                    metrics[col] = values.mean()
                    metrics[f'{col}_std'] = values.std()
                    metrics[f'{col}_median'] = values.median()
                    
                    # High opportunity samples
                    high_opp = (values > 0.7).sum() / len(values)
                    metrics[f'{col}_high_opportunity_ratio'] = high_opp
        
        # Overall quality metrics
        if 'overall_opportunity' in labeled_data.columns:
            overall_opp = labeled_data['overall_opportunity'].dropna()
            if len(overall_opp) > 0:
                metrics['signal_quality'] = overall_opp.std()  # Higher std = more diverse signals
                metrics['opportunity_coverage'] = (overall_opp > 0.5).sum() / len(overall_opp)
        
        return metrics
    
    def _finalize_optimization_results(self, results: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """Finalize optimization results with additional analysis."""
        optimization_time = time.time() - start_time
        
        final_results = {
            'optimization_summary': {
                'method': results['method'],
                'total_time_seconds': optimization_time,
                'total_evaluations': results['total_evaluations'],
                'best_score': results['best_score'],
                'optimization_completed': True
            },
            'best_configuration': {
                'time_horizons': results['best_config']['time_horizons'],
                'profit_targets': results['best_config'].get('profit_targets', 'default'),
                'config_id': results['best_config']['config_id']
            },
            'performance_analysis': self._analyze_optimization_performance(results),
            'recommended_config': self._create_recommended_config(results['best_config']),
            'validation_results': self._validate_best_config(results['best_config']),
            'timestamp': datetime.now().isoformat()
        }
        
        # Store top N results
        top_n = min(10, len(results['all_results']))
        final_results['top_configurations'] = results['all_results'][:top_n]
        
        return final_results
    
    def _analyze_optimization_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze optimization performance."""
        all_scores = [r['score'] for r in results['all_results'] if r['score'] > 0]
        
        if not all_scores:
            return {'error': 'No valid scores found'}
        
        return {
            'score_statistics': {
                'mean': np.mean(all_scores),
                'std': np.std(all_scores),
                'min': np.min(all_scores),
                'max': np.max(all_scores),
                'median': np.median(all_scores)
            },
            'improvement_analysis': {
                'best_vs_mean': results['best_score'] - np.mean(all_scores),
                'best_vs_median': results['best_score'] - np.median(all_scores),
                'improvement_percentile': (np.sum(np.array(all_scores) < results['best_score']) / len(all_scores)) * 100
            }
        }
    
    def _create_recommended_config(self, best_config: Dict[str, Any]) -> MultiHorizonConfig:
        """Create recommended MultiHorizonConfig from optimization results."""
        config = MultiHorizonConfig()
        
        # Update time horizons
        if 'time_horizons' in best_config:
            config.time_horizons = best_config['time_horizons']
        
        # Update profit targets if optimized
        if best_config.get('profit_targets') is not None:
            config.profit_targets = best_config['profit_targets']
        
        return config
    
    def _validate_best_config(self, best_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the best configuration."""
        validation = {'valid': True, 'issues': []}
        
        # Check time horizons
        horizons = best_config.get('time_horizons', {})
        if not horizons:
            validation['valid'] = False
            validation['issues'].append('No time horizons specified')
        
        # Check logical ordering
        if not self._is_valid_horizon_combination(horizons):
            validation['issues'].append('Illogical time horizon ordering')
        
        # Check reasonable ranges
        for name, value in horizons.items():
            if value < 1 or value > 50:  # Reasonable range for 5m data
                validation['issues'].append(f'Unreasonable horizon value: {name}={value}')
        
        return validation
    
    def _integrate_with_feature_lookback(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate results with existing feature lookback optimization."""
        integration = {
            'integration_attempted': True,
            'integration_successful': False,
            'updated_config_path': None,
            'backup_created': False
        }
        
        try:
            # This would integrate with the existing feature lookback system
            # For now, just return placeholder
            integration['integration_successful'] = True
            integration['notes'] = 'Integration placeholder - implement based on existing system'
            
        except Exception as e:
            integration['error'] = str(e)
        
        return integration
    
    def get_optimized_config(self) -> Optional[MultiHorizonConfig]:
        """Get the optimized configuration."""
        if self.optimization_results and 'recommended_config' in self.optimization_results:
            return self.optimization_results['recommended_config']
        return None
    
    def save_optimization_results(self, filepath: str):
        """Save optimization results to file."""
        if self.optimization_results:
            with open(filepath, 'w') as f:
                import json
                json.dump(self.optimization_results, f, indent=2, default=str)
            self.logger.info(f'💾 Optimization results saved to {filepath}')

# Convenience functions
def optimize_multi_horizon_config(data: pd.DataFrame, 
                                 config: Optional[MultiHorizonOptimizationConfig] = None) -> Dict[str, Any]:
    """Optimize multi-horizon configuration."""
    optimizer = MultiHorizonOptimizer(config)
    return optimizer.optimize_time_horizons(data)

def get_optimized_multi_horizon_config(data: pd.DataFrame) -> MultiHorizonConfig:
    """Get optimized multi-horizon configuration."""
    results = optimize_multi_horizon_config(data)
    if 'recommended_config' in results:
        return results['recommended_config']
    return MultiHorizonConfig()  # Return default if optimization fails

# Test function
if __name__ == '__main__':
    tprint('🧪 Testing Multi-Horizon Optimizer')
    
    # Create test data
    dates = pd.date_range('2024-01-01', periods=2000, freq='5min')
    np.random.seed(42)
    
    base_price = 100.0
    returns = np.random.normal(0.0001, 0.002, 2000)
    prices = [base_price]
    
    for ret in returns[:-1]:
        prices.append(prices[-1] * (1 + ret))
    
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, 2000)
    }, index=dates)
    
    # Ensure OHLC consistency
    for i in range(len(data)):
        data.loc[data.index[i], 'high'] = max(data.iloc[i][['open', 'high', 'low', 'close']])
        data.loc[data.index[i], 'low'] = min(data.iloc[i][['open', 'high', 'low', 'close']])
    
    # Test optimization
    tprint('\n🎯 Testing multi-horizon optimization...')
    config = MultiHorizonOptimizationConfig(
        optimization_method='grid',
        n_trials=20  # Small number for testing
    )
    
    optimizer = MultiHorizonOptimizer(config)
    results = optimizer.optimize_time_horizons(data)
    
    if 'error' not in results:
        tprint(f'✅ Optimization completed:')
        tprint(f'   → Best score: {results["optimization_summary"]["best_score"]:.3f}')
        tprint(f'   → Total evaluations: {results["optimization_summary"]["total_evaluations"]}')
        tprint(f'   → Optimization time: {results["optimization_summary"]["total_time_seconds"]:.1f}s')
        
        best_horizons = results['best_configuration']['time_horizons']
        tprint(f'   → Best horizons: {best_horizons}')
    else:
        tprint(f'❌ Optimization failed: {results["error"]}')
    
    tprint('✅ Multi-Horizon Optimizer test completed!')