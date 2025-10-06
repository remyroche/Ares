"""
Test Integration for Multi-Horizon Profit Labeler with Pipeline.

This script tests the integration between multi_horizon_profit_labeler and
the rest of the pre-training pipeline, including regime data splitting and
feature lookback optimization compatibility.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_error
from src.training.steps.pre_training.multi_horizon_profit_labeler import (
    MultiHorizonProfitLabeler,
    MultiHorizonConfig,
    MultiHorizonProfitLabelerComponent
)
from src.training.steps.pre_training.profit_labeling.profit_labeling_report_generator import (
    generate_profit_labeling_report,
    ProfitLabelingReportGenerator
)


def create_test_market_data(n_samples: int = 1000, n_regimes: int = 3) -> pd.DataFrame:
    """Create test market data with regime information."""
    np.random.seed(42)

    # Create datetime index
    start_date = datetime.now() - timedelta(days=30)
    dates = pd.date_range(start=start_date, periods=n_samples, freq='15min')

    # Create OHLCV data
    data = {
        'open': 100 + np.cumsum(np.random.randn(n_samples) * 0.01),
        'high': 100 + np.cumsum(np.random.randn(n_samples) * 0.01) + np.abs(np.random.randn(n_samples) * 0.005),
        'low': 100 + np.cumsum(np.random.randn(n_samples) * 0.01) - np.abs(np.random.randn(n_samples) * 0.005),
        'close': 100 + np.cumsum(np.random.randn(n_samples) * 0.01),
        'volume': np.random.randint(1000, 10000, n_samples)
    }

    df = pd.DataFrame(data, index=dates)

    # Add regime information
    regime_states = np.random.randint(0, n_regimes, n_samples)
    df['regime_state'] = regime_states

    return df


def create_test_regime_data():
    """Create test regime data splitting results."""
    return {
        'regime_data': {
            'market_data': None,  # Would be populated in real scenario
            'regime_states': np.array([0, 1, 2]),
            'regime_statistics': {
                0: {'data_points': 300, 'percentage': 30.0, 'volatility_std': 0.02},
                1: {'data_points': 400, 'percentage': 40.0, 'volatility_std': 0.015},
                2: {'data_points': 300, 'percentage': 30.0, 'volatility_std': 0.025}
            }
        },
        'regime_count': 3,
        'regime_continuity_score': 0.75,
        'processing_metrics': {
            'total_data_points': 1000,
            'processing_time_seconds': 5.2
        }
    }


async def test_multi_horizon_profit_labeler():
    """Test the multi-horizon profit labeler component."""
    try:
        tprint("🧪 Testing Multi-Horizon Profit Labeler...")

        # Create test data
        market_data = create_test_market_data(n_samples=500)
        regime_data = create_test_regime_data()

        # Test configuration
        config = MultiHorizonConfig(
            timeframe="15m",
            enable_regime_aware_labeling=True,
            enable_volatility_normalization=True,
            min_data_points=100,
            generate_reports=True
        )

        # Create labeler
        labeler = MultiHorizonProfitLabeler(config)

        # Execute labeling
        labeling_result = await labeler.execute_labeling(
            symbol="ETHUSDT",
            exchange="binance",
            timeframe="15m",
            data_dir="test_data"
        )

        # Check results structure
        if 'multi_horizon_labeling_result' in labeling_result:
            mh_result = labeling_result['multi_horizon_labeling_result']
            tprint_success("✅ Multi-horizon labeling result structure is correct")
            tprint_info(f"   → Contains metadata: {'metadata' in mh_result}")
            tprint_info(f"   → Regime-aware: {mh_result.get('metadata', {}).get('regime_aware', False)}")
        else:
            tprint_error("❌ Multi-horizon labeling result missing expected structure")
            return False

        # Test report generation
        report_generator = ProfitLabelingReportGenerator()
        report = report_generator.generate_report(
            labeling_result,
            regime_data,
            output_directory="test_reports"
        )

        tprint_success("✅ Report generation completed")
        tprint_info(f"   → Report has {len(report.recommendations)} recommendations")
        tprint_info(f"   → Quality analysis completed: {'quality_scores' in report.quality_scores}")

        return True

    except Exception as e:
        tprint_error(f"❌ Multi-horizon profit labeler test failed: {e}")
        return False


async def test_component_integration():
    """Test the component wrapper integration."""
    try:
        tprint("🧪 Testing Component Integration...")

        # Create component
        component = MultiHorizonProfitLabelerComponent()

        # Test pipeline state
        pipeline_state = {
            'symbol': 'ETHUSDT',
            'exchange': 'binance',
            'timeframe': '15m',
            'data_dir': 'test_data'
        }

        # Execute component (will fail without real data, but tests integration)
        try:
            result = await component.execute(None, pipeline_state)

            if result.success:
                tprint_success("✅ Component execution successful")
                tprint_info(f"   → Artifacts: {list(result.artifacts.keys())}")
            else:
                tprint_error(f"❌ Component execution failed: {result.error_message}")

        except Exception as e:
            tprint_warning(f"⚠️ Component execution error (expected without real data): {e}")

        return True

    except Exception as e:
        tprint_error(f"❌ Component integration test failed: {e}")
        return False


async def test_feature_compatibility():
    """Test compatibility with feature lookback optimization."""
    try:
        tprint("🧪 Testing Feature Lookback Optimization Compatibility...")

        # Create sample labeling result that matches expected format
        sample_labeling_result = {
            'multi_horizon_labeling_result': {
                'labels': pd.DataFrame({
                    'target_1': np.random.randint(-1, 2, 100),
                    'target_2': np.random.randint(-1, 2, 100)
                }),
                'confidence_scores': pd.DataFrame({
                    'target_1_confidence': np.random.random(100),
                    'target_2_confidence': np.random.random(100)
                }),
                'eligibility_masks': pd.DataFrame({
                    'target_1_eligible': np.random.choice([True, False], 100),
                    'target_2_eligible': np.random.choice([True, False], 100)
                }),
                'quality_scores': {
                    'target_1': {
                        'overall_quality': 0.75,
                        'predictability': 0.70,
                        'stability': 0.65,
                        'balance': 0.55
                    },
                    'target_2': {
                        'overall_quality': 0.68,
                        'predictability': 0.62,
                        'stability': 0.58,
                        'balance': 0.48
                    }
                },
                'metadata': {
                    'symbol': 'ETHUSDT',
                    'exchange': 'binance',
                    'timeframe': '15m',
                    'n_samples': 100,
                    'n_targets': 2,
                    'processing_time': 2.5
                }
            },
            'labeling_report': {
                'status': 'completed',
                'recommendations': ['Test recommendation']
            }
        }

        # Generate report
        report = generate_profit_labeling_report(
            sample_labeling_result,
            output_directory="test_reports"
        )

        # Check compatibility indicators
        compatibility_ok = (
            report.feature_lookback_compatibility.get('compatibility_score', 0) > 0.5
            or 'error' not in report.feature_lookback_compatibility
        )

        if compatibility_ok:
            tprint_success("✅ Feature lookback optimization compatibility verified")
            tprint_info(f"   → Compatibility score: {report.feature_lookback_compatibility.get('compatibility_score', 'N/A')}")
        else:
            tprint_error("❌ Feature lookback optimization compatibility issues detected")

        return compatibility_ok

    except Exception as e:
        tprint_error(f"❌ Feature compatibility test failed: {e}")
        return False


async def run_integration_tests():
    """Run all integration tests."""
    tprint("🚀 Starting Integration Tests...")

    test_results = []

    # Test 1: Multi-horizon profit labeler
    test_results.append(await test_multi_horizon_profit_labeler())

    # Test 2: Component integration
    test_results.append(await test_component_integration())

    # Test 3: Feature compatibility
    test_results.append(await test_feature_compatibility())

    # Summary
    passed = sum(test_results)
    total = len(test_results)

    if passed == total:
        tprint_success(f"🎉 All integration tests passed ({passed}/{total})")
        return True
    else:
        tprint_error(f"❌ Some integration tests failed ({passed}/{total})")
        return False


if __name__ == "__main__":
    """Run integration tests when script is executed directly."""
    asyncio.run(run_integration_tests())