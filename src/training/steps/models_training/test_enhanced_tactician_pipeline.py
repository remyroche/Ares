"""
Enhanced Tactician Pre-ML Orchestration Pipeline Validation

This script validates the enhanced tactician_pre_ml_orchestration.py implementation
with per-regime optimization and proper integration with regime_data_splitting results.

Key Validation Points:
1. Per-regime data splitting functionality
2. Regime-specific feature engineering optimization
3. Integration with regime data splitting results
4. Enhanced entry labeling for Tactician 5m optimization
5. Multi-method ensemble approach for entry optimization
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from models_training.tactician_pre_ml_orchestration import (
        TacticianPreMLOrchestrator, TacticianPreMLConfig, TacticianLabelingConfig
    )
    from models_training.tactician_5m_entry_optimizer import (
        Tactician5mEntryOptimizer, Tactician5mConfig, optimize_tactician_entries
    )
    from src.utils.logger import system_logger
    from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_error
    IMPORTS_SUCCESS = True
except ImportError as e:
    print(f"❌ Import error: {e}")
    IMPORTS_SUCCESS = False


class TacticianPipelineValidator:
    """Validator for the enhanced Tactician pre-ML orchestration pipeline."""

    def __init__(self):
        """Initialize the validator."""
        self.logger = system_logger.getChild('TacticianPipelineValidator')
        self.test_results = {}

    def generate_mock_market_data(self, num_periods: int = 1000) -> pd.DataFrame:
        """Generate mock market data for testing."""
        tprint_info(f"📊 Generating {num_periods} periods of mock market data...")

        # Create datetime index
        start_time = datetime.now() - timedelta(minutes=num_periods * 15)
        timestamps = [start_time + timedelta(minutes=i * 15) for i in range(num_periods)]

        # Generate realistic price data
        np.random.seed(42)  # For reproducible results

        # Base price movement
        price_changes = np.random.normal(0, 0.002, num_periods)  # ~0.2% std dev
        prices = 50000 * np.exp(np.cumsum(price_changes))  # Start around $50k

        # Generate OHLC data
        data = []
        for i in range(num_periods):
            open_price = prices[i]
            close_price = prices[i] * (1 + np.random.normal(0, 0.001))

            # Generate high/low around open/close
            volatility = abs(close_price - open_price) / open_price
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, volatility/2)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, volatility/2)))

            volume = np.random.lognormal(10, 1)  # Realistic volume distribution

            data.append({
                'timestamp': timestamps[i],
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)

        tprint_info(f"✅ Generated market data: {len(df)} periods, price range ${df['low'].min()".1f"}-${df['high'].max()".1f"}")
        return df

    def generate_mock_analyst_signals(self, market_data: pd.DataFrame) -> pd.Series:
        """Generate mock Analyst signals for testing."""
        tprint_info("🎯 Generating mock Analyst confidence signals...")

        # Create signals that occasionally go above threshold (0.4%)
        np.random.seed(123)
        signals = np.random.normal(0.002, 0.001, len(market_data))  # Mean 0.2%, std 0.1%

        # Add some periods of higher confidence (green lights)
        green_periods = []
        for i in range(0, len(signals), 50):  # Every ~12.5 hours
            if i + 10 < len(signals):
                signals[i:i+10] = np.random.normal(0.006, 0.001, 10)  # 0.6% mean for 10 periods
                green_periods.append((i, i+10))

        analyst_signals = pd.Series(signals, index=market_data.index)

        tprint_info(f"✅ Generated Analyst signals: {len(green_periods)} green periods")
        tprint_info(f"📊 Signal range: {analyst_signals.min()".3f"} to {analyst_signals.max()".3f"}")

        return analyst_signals

    def generate_mock_regime_data(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate mock regime data splitting results."""
        tprint_info("🏷️ Generating mock regime data splitting results...")

        # Create mock regime assignments (3 regimes: trending, ranging, volatile)
        np.random.seed(456)
        regime_assignments = np.random.choice([0, 1, 2], size=len(market_data), p=[0.5, 0.3, 0.2])

        # Create unified regime data structure
        unified_data = {
            'regime_assignments': pd.Series(regime_assignments, index=market_data.index),
            'regime_labels': {0: 'trending', 1: 'ranging', 2: 'volatile'},
            'regime_stats': {
                'regime_0': {'samples': (regime_assignments == 0).sum(), 'percentage': 50.0},
                'regime_1': {'samples': (regime_assignments == 1).sum(), 'percentage': 30.0},
                'regime_2': {'samples': (regime_assignments == 2).sum(), 'percentage': 20.0}
            },
            'total_samples': len(market_data)
        }

        regime_splits_result = {
            'unified_data': unified_data,
            'regime_stats': unified_data['regime_stats'],
            'metadata': {
                'symbol': 'ETHUSDT',
                'exchange': 'binance',
                'timeframe': '15m',
                'generation_time': datetime.now().isoformat()
            }
        }

        tprint_info("✅ Generated regime data with 3 mock regimes")
        tprint_info(f"🏷️ Regime distribution: {[f'{k}: {v[\"samples\"]} samples' for k, v in unified_data['regime_stats'].items()]}")

        return regime_splits_result

    async def test_per_regime_orchestration(self) -> Dict[str, Any]:
        """Test the per-regime orchestration functionality."""
        tprint_info("🧪 Testing per-regime orchestration...")

        try:
            # Generate test data
            market_data_15m = self.generate_mock_market_data(500)
            analyst_signals = self.generate_mock_analyst_signals(market_data_15m)
            regime_data = self.generate_mock_regime_data(market_data_15m)

            # Create orchestrator with per-regime optimization enabled
            config = TacticianPreMLConfig(
                symbol="ETHUSDT",
                exchange="binance",
                timeframe="15m",
                enable_per_regime_optimization=True,
                enable_per_cluster_optimization=True
            )

            orchestrator = TacticianPreMLOrchestrator(config)

            # Test orchestration
            result = await orchestrator.orchestrate(
                training_data=market_data_15m,
                analyst_predictions=pd.DataFrame({'analyst_signal': analyst_signals}),
                regime_data_splitting_result=regime_data
            )

            # Validate results
            test_result = {
                'success': result.success,
                'execution_time': result.execution_time,
                'total_samples_before': result.total_samples_before_filter,
                'total_samples_after': result.total_samples_after_filter,
                'filter_ratio': result.filter_ratio,
                'final_feature_count': result.final_feature_count,
                'phase': result.phase.value if hasattr(result.phase, 'value') else str(result.phase)
            }

            if result.success:
                tprint_success("✅ Per-regime orchestration test PASSED")
                tprint_info(f"📊 Execution time: {result.execution_time:.2f}s")
                tprint_info(f"📊 Final features: {result.final_feature_count}")
                tprint_info(f"📊 Data retention: {result.filter_ratio:.2%}")
            else:
                tprint_error(f"❌ Per-regime orchestration test FAILED: {result.error_message}")

            return test_result

        except Exception as e:
            tprint_error(f"❌ Per-regime orchestration test ERROR: {e}")
            return {'success': False, 'error': str(e)}

    async def test_tactician_5m_entry_optimizer(self) -> Dict[str, Any]:
        """Test the Tactician 5m entry optimizer."""
        tprint_info("🎯 Testing Tactician 5m entry optimizer...")

        try:
            # Generate test data
            data_5m = self.generate_mock_market_data(1000)  # More data for 5m
            analyst_signals_15m = self.generate_mock_analyst_signals(data_5m)

            # Create 5m entry optimizer
            config = Tactician5mConfig(
                analyst_timeframe="15m",
                tactician_timeframe="5m",
                optimization_method="hybrid_ensemble"
            )

            optimizer = Tactician5mEntryOptimizer(config)

            # Test entry optimization
            result = optimizer.optimize_entries(data_5m, analyst_signals_15m)

            # Validate results
            test_result = {
                'success': result.success,
                'execution_time': result.execution_time,
                'green_periods_analyzed': result.green_periods_analyzed,
                'total_entries_found': result.total_entries_found,
                'optimal_entries_count': len(result.optimal_entries),
                'avg_entry_quality': result.avg_entry_quality,
                'best_entry_score': result.best_entry_score,
                'method_used': result.method_used.value if result.method_used else None
            }

            if result.success:
                tprint_success("✅ Tactician 5m entry optimizer test PASSED")
                tprint_info(f"📊 Found {len(result.optimal_entries)} optimal entries")
                tprint_info(f"📊 Average quality: {result.avg_entry_quality:.3f}")
                tprint_info(f"📊 Best score: {result.best_entry_score:.3f}")
            else:
                tprint_error(f"❌ Tactician 5m entry optimizer test FAILED: {result.error_message}")

            return test_result

        except Exception as e:
            tprint_error(f"❌ Tactician 5m entry optimizer test ERROR: {e}")
            return {'success': False, 'error': str(e)}

    def test_regime_data_integration(self) -> Dict[str, Any]:
        """Test integration with regime data splitting results."""
        tprint_info("🔗 Testing regime data integration...")

        try:
            # Generate test data
            market_data = self.generate_mock_market_data(300)
            regime_data = self.generate_mock_regime_data(market_data)

            # Create orchestrator
            orchestrator = TacticianPreMLOrchestrator()

            # Test regime data preparation
            regime_datasets = orchestrator._prepare_training_data_per_regime(market_data, regime_data)

            # Validate regime splitting
            test_result = {
                'success': True,
                'regime_count': len(regime_datasets),
                'total_samples_across_regimes': sum(len(df) for df in regime_datasets.values()),
                'regime_names': list(regime_datasets.keys()),
                'samples_per_regime': {name: len(df) for name, df in regime_datasets.items()}
            }

            tprint_success("✅ Regime data integration test PASSED")
            tprint_info(f"🏷️ Created {len(regime_datasets)} regime datasets")
            tprint_info(f"📊 Total samples: {test_result['total_samples_across_regimes']}")

            for regime_name, samples in test_result['samples_per_regime'].items():
                tprint_info(f"🏷️ {regime_name}: {samples} samples")

            return test_result

        except Exception as e:
            tprint_error(f"❌ Regime data integration test ERROR: {e}")
            return {'success': False, 'error': str(e)}

    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation of the enhanced pipeline."""
        tprint_info("🚀 Starting comprehensive Tactician pipeline validation...")

        if not IMPORTS_SUCCESS:
            return {'success': False, 'error': 'Import failures detected'}

        validation_results = {}

        try:
            # Test 1: Regime data integration
            validation_results['regime_integration'] = self.test_regime_data_integration()

            # Test 2: Per-regime orchestration
            validation_results['per_regime_orchestration'] = await self.test_per_regime_orchestration()

            # Test 3: Tactician 5m entry optimizer
            validation_results['tactician_5m_optimizer'] = await self.test_tactician_5m_entry_optimizer()

            # Overall assessment
            overall_success = all([
                validation_results['regime_integration']['success'],
                validation_results['per_regime_orchestration']['success'],
                validation_results['tactician_5m_optimizer']['success']
            ])

            summary = {
                'overall_success': overall_success,
                'total_tests': len(validation_results),
                'passed_tests': sum(1 for r in validation_results.values() if r.get('success', False)),
                'validation_timestamp': datetime.now().isoformat(),
                'components_tested': list(validation_results.keys())
            }

            if overall_success:
                tprint_success("🎉 COMPREHENSIVE VALIDATION PASSED!")
                tprint_info(f"✅ {summary['passed_tests']}/{summary['total_tests']} tests passed")
            else:
                tprint_error("❌ COMPREHENSIVE VALIDATION FAILED!")
                tprint_info(f"❌ {summary['passed_tests']}/{summary['total_tests']} tests passed")

            return {
                'success': overall_success,
                'summary': summary,
                'detailed_results': validation_results
            }

        except Exception as e:
            tprint_error(f"❌ Comprehensive validation ERROR: {e}")
            return {
                'success': False,
                'error': str(e),
                'summary': {'overall_success': False, 'error': str(e)}
            }


async def main():
    """Main validation function."""
    tprint("🧪 Tactician Enhanced Pipeline Validation")
    tprint("=" * 50)

    validator = TacticianPipelineValidator()
    results = await validator.run_comprehensive_validation()

    # Print summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)

    if results['success']:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")

    summary = results['summary']
    print(f"📊 Tests run: {summary['total_tests']}")
    print(f"✅ Passed: {summary['passed_tests']}")
    print(f"❌ Failed: {summary['total_tests'] - summary['passed_tests']}")
    print(f"⏰ Completed: {summary['validation_timestamp']}")

    print("\n📋 COMPONENTS TESTED:")
    for component in summary['components_tested']:
        status = "✅ PASS" if results['detailed_results'][component]['success'] else "❌ FAIL"
        print(f"  • {component}: {status}")

    return results


if __name__ == "__main__":
    # Run validation
    results = asyncio.run(main())

    # Exit with appropriate code
    if results['success']:
        exit(0)
    else:
        exit(1)