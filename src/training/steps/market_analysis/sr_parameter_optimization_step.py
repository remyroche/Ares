"""
SR Parameter Optimization Step

BaseStep-based implementation for optimizing Support/Resistance detection parameters.
Migrated from the component pattern to use the new BaseStep architecture.
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path
import warnings

from src.training.steps.base_step import BaseStep
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error
from src.utils.logger import system_logger

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Import optimization libraries with error handling
OPTIMIZATION_LIBRARIES_AVAILABLE = False
OPTIMIZATION_IMPORT_ERRORS = []

try:
    from sklearn.model_selection import ParameterGrid
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    OPTIMIZATION_LIBRARIES_AVAILABLE = True
    tprint("✅ Optimization libraries imported successfully", "SUCCESS")
except ImportError as e:
    OPTIMIZATION_IMPORT_ERRORS.append(f"sklearn: {e}")
    tprint(f"❌ Failed to import optimization libraries: {e}", "ERROR")

# Import SR clustering components with error handling
SR_CLUSTERING_AVAILABLE = False
try:
    from src.utils.sr_clustering.sr_backtesting_engine import SRBacktestingEngine, BacktestConfig
    from src.utils.sr_clustering.parameter_optimization_engine import get_parameter_optimization_engine, ParameterOptimizationConfig
    SR_CLUSTERING_AVAILABLE = True
    tprint("✅ SR clustering components imported successfully", "SUCCESS")
except ImportError as e:
    SR_CLUSTERING_AVAILABLE = False
    tprint(f"⚠️ SR clustering components not available: {e}", "WARNING")


class SRParameterOptimizationStep(BaseStep):
    """
    SR Parameter Optimization Step using BaseStep pattern.
    
    Optimizes Support/Resistance detection parameters using backtesting.
    """
    
    def __init__(self, step_name: str = "sr_parameter_optimization"):
        """Initialize the SR parameter optimization step."""
        super().__init__(step_name)
        self.logger = system_logger.getChild('SRParameterOptimizationStep')
        
        if not OPTIMIZATION_LIBRARIES_AVAILABLE:
            tprint("⚠️ Optimization libraries not available - functionality may be limited", "WARNING")
        
        if not SR_CLUSTERING_AVAILABLE:
            tprint("⚠️ SR clustering components not available - using fallback optimization", "WARNING")
        
        tprint("✅ SRParameterOptimizationStep initialized", "SUCCESS")
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute SR parameter optimization step.
        
        Args:
            config: Configuration dictionary with parameters:
                - symbol: Trading symbol (e.g., 'ETHUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Timeframe (e.g., '15m')
                - data_dir: Data directory path
                - start_date: Start date (optional)
                - end_date: End date (optional)
                - market_data: Market data (optional)
                - sr_levels: SR levels (optional)
                - parameter_grid: Parameter grid for optimization (optional)
                - optimization_metric: Metric to optimize (default: 'accuracy')
                
        Returns:
            Dictionary with optimization results and best parameters
        """
        start_time = datetime.now()
        
        try:
            tprint(f"🔍 Starting SR parameter optimization for {config.get('symbol', 'UNKNOWN')}", "INFO")
            
            # Set context for artifact management
            self._set_context(
                symbol=config.get('symbol', 'ETHUSDT'),
                exchange=config.get('exchange', 'binance'),
                direction=config.get('direction', 'both'),
                model=config.get('model', 'default')
            )
            
            # Load market data
            market_data = self._load_market_data(config)
            if market_data is None:
                raise ValueError("No market data found")
            
            tprint(f"✅ Loaded market data: {market_data.shape[0]} rows, {market_data.shape[1]} columns", "SUCCESS")
            
            # Load SR levels
            sr_levels = self._load_sr_levels(config)
            if sr_levels is None:
                raise ValueError("No SR levels found")
            
            tprint(f"✅ Loaded SR levels: {len(sr_levels)} levels", "SUCCESS")
            
            # Get parameter grid
            parameter_grid = self._get_parameter_grid(config)
            
            # Perform optimization
            optimization_results = self._optimize_parameters(market_data, sr_levels, parameter_grid, config)
            
            # Save optimization results
            self._save_optimization_results(optimization_results, config)
            
            # Calculate metrics
            metrics = self._calculate_optimization_metrics(optimization_results, start_time, config)
            
            # Create outcome report
            outcome_report = self._create_outcome_report(optimization_results, metrics, config)
            
            tprint(f"✅ SR parameter optimization completed", "SUCCESS")
            
            return {
                'success': True,
                'optimization_results': optimization_results,
                'metrics': metrics,
                'outcome_report': outcome_report,
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
            
        except Exception as e:
            error_msg = f"SR parameter optimization failed: {str(e)}"
            tprint(f"❌ {error_msg}", "ERROR")
            self.logger.error(error_msg)
            
            return {
                'success': False,
                'error': error_msg,
                'optimization_results': {},
                'metrics': {},
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
    
    def _load_market_data(self, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Load market data from artifacts or config."""
        try:
            # Try to load from artifacts first
            market_data = self._load_dataframe('market_data')
            if market_data is not None:
                return market_data
            
            # Try alternative artifact names
            market_data = self._load_dataframe('processed_data') or self._load_dataframe('data')
            if market_data is not None:
                return market_data
            
            # Try to load from config
            if 'market_data' in config:
                return pd.DataFrame(config['market_data'])
            
            return None
            
        except Exception as e:
            tprint(f"⚠️ Failed to load market data: {e}", "WARNING")
            return None
    
    def _load_sr_levels(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Load SR levels from artifacts or config."""
        try:
            # Try to load from artifacts first
            sr_levels = self._get_artifact('sr_levels')
            if sr_levels is not None:
                return sr_levels
            
            # Try alternative artifact names
            sr_levels = self._get_artifact('support_resistance_levels') or self._get_artifact('sr_data')
            if sr_levels is not None:
                return sr_levels
            
            # Try to load from config
            if 'sr_levels' in config:
                return config['sr_levels']
            
            return None
            
        except Exception as e:
            tprint(f"⚠️ Failed to load SR levels: {e}", "WARNING")
            return None
    
    def _get_parameter_grid(self, config: Dict[str, Any]) -> Dict[str, List[Any]]:
        """Get parameter grid for optimization."""
        default_grid = {
            'min_touches': [2, 3, 4, 5],
            'tolerance_pct': [0.3, 0.5, 0.7, 1.0],
            'lookback_periods': [50, 100, 150, 200],
            'min_strength': [0.1, 0.2, 0.3, 0.4],
            'max_levels': [10, 20, 30, 50]
        }
        
        # Use config parameter grid if provided
        if 'parameter_grid' in config:
            return config['parameter_grid']
        
        return default_grid
    
    def _optimize_parameters(self, market_data: pd.DataFrame, sr_levels: Dict[str, Any], parameter_grid: Dict[str, List[Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize SR parameters using grid search."""
        try:
            if SR_CLUSTERING_AVAILABLE:
                return self._optimize_with_sr_clustering(market_data, sr_levels, parameter_grid, config)
            else:
                return self._optimize_with_fallback(market_data, sr_levels, parameter_grid, config)
                
        except Exception as e:
            tprint(f"❌ Failed to optimize parameters: {e}", "ERROR")
            raise
    
    def _optimize_with_sr_clustering(self, market_data: pd.DataFrame, sr_levels: Dict[str, Any], parameter_grid: Dict[str, List[Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize parameters using SR clustering engine."""
        try:
            tprint("🔧 Using SR clustering engine for optimization", "INFO")
            
            # Create backtest config
            backtest_config = BacktestConfig(
                symbol=config.get('symbol', 'ETHUSDT'),
                exchange=config.get('exchange', 'binance'),
                timeframe=config.get('timeframe', '15m'),
                start_date=config.get('start_date'),
                end_date=config.get('end_date')
            )
            
            # Create parameter optimization config
            opt_config = ParameterOptimizationConfig(
                parameter_grid=parameter_grid,
                optimization_metric=config.get('optimization_metric', 'accuracy'),
                cv_folds=config.get('cv_folds', 3),
                n_trials=config.get('n_trials', 10)
            )
            
            # Get optimization engine
            optimization_engine = get_parameter_optimization_engine()
            
            # Run optimization
            results = optimization_engine.optimize(
                market_data=market_data,
                sr_levels=sr_levels,
                backtest_config=backtest_config,
                optimization_config=opt_config
            )
            
            return results
            
        except Exception as e:
            tprint(f"⚠️ SR clustering optimization failed, using fallback: {e}", "WARNING")
            return self._optimize_with_fallback(market_data, sr_levels, parameter_grid, config)
    
    def _optimize_with_fallback(self, market_data: pd.DataFrame, sr_levels: Dict[str, Any], parameter_grid: Dict[str, List[Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback optimization using simple grid search."""
        try:
            tprint("🔧 Using fallback optimization method", "INFO")
            
            best_score = -np.inf
            best_params = None
            all_results = []
            
            # Generate parameter combinations
            param_names = list(parameter_grid.keys())
            param_values = list(parameter_grid.values())
            
            # Simple grid search
            for i, param_combo in enumerate(self._generate_parameter_combinations(param_values)):
                params = dict(zip(param_names, param_combo))
                
                try:
                    # Evaluate parameters
                    score = self._evaluate_parameters(market_data, sr_levels, params, config)
                    
                    result = {
                        'parameters': params,
                        'score': score,
                        'iteration': i + 1
                    }
                    all_results.append(result)
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                    
                    tprint(f"📊 Iteration {i+1}: Score {score:.3f} for params {params}", "INFO")
                    
                except Exception as e:
                    tprint(f"⚠️ Failed to evaluate parameters {params}: {e}", "WARNING")
                    continue
            
            return {
                'best_parameters': best_params,
                'best_score': best_score,
                'all_results': all_results,
                'n_iterations': len(all_results),
                'optimization_method': 'fallback_grid_search'
            }
            
        except Exception as e:
            tprint(f"❌ Fallback optimization failed: {e}", "ERROR")
            raise
    
    def _generate_parameter_combinations(self, param_values: List[List[Any]]) -> List[Tuple[Any, ...]]:
        """Generate all combinations of parameter values."""
        if not param_values:
            return [()]
        
        if len(param_values) == 1:
            return [(val,) for val in param_values[0]]
        
        combinations = []
        for val in param_values[0]:
            for combo in self._generate_parameter_combinations(param_values[1:]):
                combinations.append((val,) + combo)
        
        return combinations
    
    def _evaluate_parameters(self, market_data: pd.DataFrame, sr_levels: Dict[str, Any], params: Dict[str, Any], config: Dict[str, Any]) -> float:
        """Evaluate parameter set using simple scoring."""
        try:
            # Simple evaluation based on SR level quality
            if not sr_levels or 'levels' not in sr_levels:
                return 0.0
            
            levels = sr_levels['levels']
            if not levels:
                return 0.0
            
            # Calculate quality metrics
            n_levels = len(levels)
            avg_strength = np.mean([level.get('strength', 0) for level in levels]) if levels else 0
            avg_touches = np.mean([level.get('touches', 0) for level in levels]) if levels else 0
            
            # Simple scoring function
            score = (
                min(n_levels / params.get('max_levels', 50), 1.0) * 0.3 +  # Level count score
                min(avg_strength, 1.0) * 0.4 +  # Strength score
                min(avg_touches / params.get('min_touches', 2), 1.0) * 0.3  # Touch count score
            )
            
            return score
            
        except Exception as e:
            tprint(f"⚠️ Failed to evaluate parameters: {e}", "WARNING")
            return 0.0
    
    def _save_optimization_results(self, results: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Save optimization results using artifact manager."""
        try:
            # Save optimization results
            self._save_artifact('sr_optimization_results', results)
            
            # Save best parameters
            if 'best_parameters' in results:
                self._save_artifact('best_sr_parameters', results['best_parameters'])
            
            # Save optimization metadata
            optimization_metadata = {
                'optimization_timestamp': datetime.now().isoformat(),
                'n_iterations': results.get('n_iterations', 0),
                'best_score': results.get('best_score', 0.0),
                'optimization_method': results.get('optimization_method', 'unknown'),
                'config': config
            }
            self._save_metadata(optimization_metadata)
            
            tprint("✅ Optimization results saved to artifacts", "SUCCESS")
            
        except Exception as e:
            tprint(f"⚠️ Failed to save optimization results: {e}", "WARNING")
    
    def _calculate_optimization_metrics(self, results: Dict[str, Any], start_time: datetime, config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimization metrics."""
        try:
            processing_time = (datetime.now() - start_time).total_seconds()
            
            metrics = {
                'processing_time_seconds': processing_time,
                'best_score': results.get('best_score', 0.0),
                'n_iterations': results.get('n_iterations', 0),
                'optimization_method': results.get('optimization_method', 'unknown'),
                'best_parameters': results.get('best_parameters', {}),
                'success': True
            }
            
            return metrics
            
        except Exception as e:
            tprint(f"⚠️ Failed to calculate metrics: {e}", "WARNING")
            return {'success': False, 'error': str(e)}
    
    def _create_outcome_report(self, results: Dict[str, Any], metrics: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Create outcome report markdown."""
        try:
            report = f"""# SR Parameter Optimization Outcome Report

## Execution Summary
- **Symbol**: {config.get('symbol', 'UNKNOWN')}
- **Exchange**: {config.get('exchange', 'UNKNOWN')}
- **Timeframe**: {config.get('timeframe', 'UNKNOWN')}
- **Processing Time**: {metrics.get('processing_time_seconds', 0):.2f} seconds
- **Success**: {'✅ Yes' if metrics.get('success', False) else '❌ No'}

## Optimization Results
- **Best Score**: {metrics.get('best_score', 0):.3f}
- **Iterations**: {metrics.get('n_iterations', 0)}
- **Method**: {metrics.get('optimization_method', 'unknown')}

## Best Parameters
"""
            
            best_params = metrics.get('best_parameters', {})
            for param, value in best_params.items():
                report += f"- **{param}**: {value}\n"
            
            report += f"""
## Optimization Details
- **Parameter Grid Size**: {len(results.get('all_results', []))}
- **SR Clustering Available**: {'✅ Yes' if SR_CLUSTERING_AVAILABLE else '❌ No'}
- **Optimization Libraries Available**: {'✅ Yes' if OPTIMIZATION_LIBRARIES_AVAILABLE else '❌ No'}

## Generated Artifacts
- Optimization results
- Best parameters
- Optimization metadata

---
*Generated by SR Parameter Optimization Step at {datetime.now().isoformat()}*
"""
            
            return report
            
        except Exception as e:
            tprint(f"⚠️ Failed to create outcome report: {e}", "WARNING")
            return f"# SR Parameter Optimization Outcome Report\n\nError creating report: {str(e)}"


# Register the step
def register_sr_parameter_optimization_step():
    """Register the SR parameter optimization step."""
    from src.training.steps.base_step import step_registry
    
    step_registry.register("sr_parameter_optimization", SRParameterOptimizationStep)
    tprint("✅ SR parameter optimization step registered", "SUCCESS")


# Auto-register when module is imported
register_sr_parameter_optimization_step()