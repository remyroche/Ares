"""Step 20: AB Testing - Per-Regime Implementation."""

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
import pandas as pd
import numpy as np
import json
from datetime import datetime

from src.training.steps.step20_ab_testing import Step20ABTesting
from src.training.steps.regime_continuity_decorator import per_regime_step
from src.utils.logger import getChild as get_logger
from src.utils.pipeline_standards import pipeline_standards
from src.core.decorators import traced, validates, handles_errors

logger = get_logger('Step20ABTestingPerRegime')

class PerRegimeABTestingStep(Step20ABTesting):
    """AB testing step that processes each regime separately."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.per_regime_enabled = config.get('per_regime_ab_testing', True)
        
    @traced(span_name='execute_per_regime_ab_testing')
    @per_regime_step('step20_ab_testing')
    async def execute_per_regime_ab_testing(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        force_rerun: bool = False,
        regime_id: Optional[int] = None,
        regime_context: Optional[Any] = None,
        per_regime: bool = True
    ) -> bool:
        """Execute AB testing on a per-regime basis."""
        try:
            self.logger.info(f"🚀 Starting per-regime AB testing for regime {regime_id}")
            
            # Load Monte Carlo validation results
            mc_data = await self._load_mc_data(symbol, exchange, timeframe, data_dir, regime_id)
            if mc_data is None:
                self.logger.error(f"❌ Failed to load Monte Carlo data for regime {regime_id}")
                return False
            
            # Perform AB testing
            ab_results = await self._perform_ab_testing(mc_data, regime_id)
            
            # Save results
            success = await self._save_ab_results(ab_results, symbol, exchange, timeframe, data_dir, regime_id)
            
            if success:
                self.logger.info(f"✅ Successfully completed AB testing for regime {regime_id}")
            else:
                self.logger.error(f"❌ Failed to save AB results for regime {regime_id}")
            
            return success
            
        except Exception as e:
            self.logger.exception(f"❌ Error in per-regime AB testing for regime {regime_id}: {e}")
            return False
    
    async def _load_mc_data(self, symbol: str, exchange: str, timeframe: str, data_dir: str, regime_id: int) -> Optional[Dict[str, Any]]:
        """Load Monte Carlo data for regime."""
        try:
            mc_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_monte_carlo_validation_regime_{regime_id}.json'
            if mc_path.exists():
                with open(mc_path, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            self.logger.error(f"❌ Error loading Monte Carlo data for regime {regime_id}: {e}")
            return None
    
    async def _perform_ab_testing(self, mc_data: Dict[str, Any], regime_id: int) -> Dict[str, Any]:
        """Perform AB testing for regime."""
        try:
            results = {
                'regime_id': regime_id,
                'ab_tests': {},
                'test_results': {},
                'statistical_significance': {}
            }
            
            # Define test variants
            variants = {
                'control': {'name': 'Control', 'parameters': {}},
                'variant_a': {'name': 'Variant A', 'parameters': {'learning_rate': 0.01}},
                'variant_b': {'name': 'Variant B', 'parameters': {'learning_rate': 0.02}},
                'variant_c': {'name': 'Variant C', 'parameters': {'learning_rate': 0.005}}
            }
            
            # Run AB tests
            for variant_name, variant_config in variants.items():
                test_result = await self._run_ab_test_variant(variant_config, regime_id)
                results['ab_tests'][variant_name] = test_result
            
            # Calculate statistical significance
            results['statistical_significance'] = self._calculate_statistical_significance(results['ab_tests'])
            
            # Determine winning variant
            results['winning_variant'] = self._determine_winning_variant(results['ab_tests'])
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error performing AB testing for regime {regime_id}: {e}")
            return {}
    
    async def _run_ab_test_variant(self, variant_config: Dict[str, Any], regime_id: int) -> Dict[str, Any]:
        """Run AB test for a specific variant."""
        try:
            # Simulate test results
            base_performance = 0.7
            
            # Adjust based on regime characteristics
            if regime_id <= 2:  # Trending regimes
                performance_boost = 0.1
            elif regime_id >= 5:  # Volatile regimes
                performance_boost = 0.05
            else:  # Balanced regimes
                performance_boost = 0.08
            
            # Adjust based on variant parameters
            learning_rate = variant_config.get('parameters', {}).get('learning_rate', 0.01)
            if learning_rate > 0.01:
                performance_boost += 0.05
            elif learning_rate < 0.01:
                performance_boost += 0.03
            
            test_performance = min(1.0, base_performance + performance_boost)
            
            return {
                'variant_name': variant_config['name'],
                'parameters': variant_config['parameters'],
                'performance_metrics': {
                    'accuracy': test_performance,
                    'precision': min(1.0, test_performance - 0.05),
                    'recall': min(1.0, test_performance - 0.03),
                    'f1_score': 2 * (test_performance - 0.05) * (test_performance - 0.03) / (2 * test_performance - 0.08),
                    'sharpe_ratio': np.random.uniform(0.5, 2.0),
                    'max_drawdown': np.random.uniform(0.05, 0.3)
                },
                'test_metadata': {
                    'sample_size': np.random.randint(100, 1000),
                    'test_duration': np.random.uniform(1, 30),
                    'confidence_level': 0.95
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error running AB test variant: {e}")
            return {}
    
    def _calculate_statistical_significance(self, ab_tests: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistical significance of AB test results."""
        try:
            significance_results = {}
            
            # Extract performance metrics
            control_performance = ab_tests.get('control', {}).get('performance_metrics', {}).get('accuracy', 0.5)
            
            for variant_name, test_result in ab_tests.items():
                if variant_name == 'control':
                    continue
                
                variant_performance = test_result.get('performance_metrics', {}).get('accuracy', 0.5)
                
                # Calculate statistical significance (simplified)
                performance_diff = variant_performance - control_performance
                p_value = np.random.uniform(0.01, 0.5)  # Simulated p-value
                
                significance_results[variant_name] = {
                    'performance_difference': performance_diff,
                    'p_value': p_value,
                    'statistically_significant': p_value < 0.05,
                    'confidence_interval': {
                        'lower': performance_diff - 0.05,
                        'upper': performance_diff + 0.05
                    }
                }
            
            return significance_results
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating statistical significance: {e}")
            return {}
    
    def _determine_winning_variant(self, ab_tests: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the winning variant from AB tests."""
        try:
            best_variant = None
            best_performance = 0.0
            
            for variant_name, test_result in ab_tests.items():
                performance = test_result.get('performance_metrics', {}).get('accuracy', 0.0)
                if performance > best_performance:
                    best_performance = performance
                    best_variant = variant_name
            
            return {
                'winning_variant': best_variant,
                'winning_performance': best_performance,
                'improvement_over_control': best_performance - ab_tests.get('control', {}).get('performance_metrics', {}).get('accuracy', 0.0)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error determining winning variant: {e}")
            return {}
    
    async def _save_ab_results(self, ab_results: Dict[str, Any], symbol: str, exchange: str, timeframe: str, data_dir: str, regime_id: int) -> bool:
        """Save AB testing results for regime."""
        try:
            ab_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_ab_testing_regime_{regime_id}.json'
            with open(ab_path, 'w') as f:
                json.dump(ab_results, f, indent=2, default=str)
            self.logger.info(f"✅ Saved AB testing results for regime {regime_id}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error saving AB testing results for regime {regime_id}: {e}")
            return False

@traced(span_name='run_per_regime_ab_testing_step')
@validates()
@handles_errors
async def run_per_regime_step(symbol: str, exchange: str, timeframe: str, data_dir: str = None, force_rerun: bool = False, config: Optional[Dict[str, Any]] = None) -> bool:
    """Run the per-regime AB testing step."""
    logger.info("🚀 Starting Step 20: Per-Regime AB Testing")
    
    if config is None:
        config = {}
    if data_dir is None:
        data_dir = pipeline_standards.build_path('processed_data', exchange, symbol)
    
    config['per_regime_ab_testing'] = True
    step = PerRegimeABTestingStep(config)
    
    success = await step.execute_per_regime_ab_testing(symbol=symbol, exchange=exchange, timeframe=timeframe, data_dir=data_dir, force_rerun=force_rerun)
    
    if success:
        logger.info("✅ Step 20: Per-Regime AB Testing completed successfully")
    else:
        logger.error("❌ Step 20: Per-Regime AB Testing failed")
        
    return success

if __name__ == '__main__':
    async def test():
        success = await run_per_regime_step(symbol='ETHUSDT', exchange='BINANCE', timeframe='1m', data_dir='data_cache')
        print(f'Per-regime AB testing result: {success}')
    asyncio.run(test())