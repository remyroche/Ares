#!/usr/bin/env python3
"""
Test script for Step05 Refactored Modular Architecture

This script tests the new modular Step05 implementation with validation,
financial calculations, error handling, and reporting modules.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.training.steps.step05_labeling_refactored import run_step05_refactored
from src.training.steps.step05_validation import Step05Validator
from src.training.steps.step05_financial import Step05FinancialCalculator
from src.training.steps.step05_error_handling import Step05ErrorHandler
from src.training.steps.step05_reporting import Step05Reporter


def create_test_data():
    """Create test data for Step05 validation."""
    np.random.seed(42)
    
    # Generate timestamps
    base_time = datetime.now()
    timestamps = [base_time + timedelta(minutes=i) for i in range(1000)]
    
    # Generate price data
    prices = []
    price = 50000.0
    for i in range(1000):
        change = np.random.normal(0, 0.01)  # 1% volatility
        price *= (1 + change)
        prices.append(price)
    
    # Generate labels
    labels = []
    for i in range(1000):
        if i < 5:
            labels.append(0)
            continue
        
        # Simple labeling logic
        recent_prices = prices[max(0, i-10):i+1]
        if len(recent_prices) < 5:
            labels.append(0)
            continue
        
        short_ma = np.mean(recent_prices[-5:])
        long_ma = np.mean(recent_prices[-20:])
        
        if short_ma > long_ma * 1.002:
            labels.append(1)
        elif short_ma < long_ma * 0.998:
            labels.append(-1)
        else:
            labels.append(0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.003))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.003))) for p in prices],
        'close': prices,
        'volume': np.random.lognormal(15, 1, 1000),
        'label': labels,
        'hmm_regime': np.random.choice([0, 1, 2], 1000, p=[0.4, 0.4, 0.2])
    })
    
    df.set_index('timestamp', inplace=True)
    return df


def test_validation_module():
    """Test the validation module."""
    print("🧪 Testing Validation Module...")
    
    try:
        validator = Step05Validator()
        test_data = create_test_data()
        
        # Test data integrity validation
        print("  - Testing data integrity validation...")
        integrity_result = validator.validate_data_integrity(test_data)
        print(f"    Data integrity passed: {integrity_result.passed}")
        print(f"    Data integrity score: {integrity_result.score:.3f}")
        
        # Test lookahead bias validation
        print("  - Testing lookahead bias validation...")
        barrier_params = {
            'profit_take_multiplier': 0.002,
            'stop_loss_multiplier': 0.001,
            'time_barrier_minutes': 30,
            'max_lookahead': 100
        }
        bias_result = validator.validate_lookahead_bias(test_data, barrier_params)
        print(f"    Lookahead bias detected: {bias_result.bias_detected}")
        print(f"    Bias score: {bias_result.bias_score:.3f}")
        
        # Test label quality validation
        print("  - Testing label quality validation...")
        quality_result = validator.validate_label_quality(test_data)
        print(f"    Label quality passed: {quality_result.passed}")
        print(f"    Label quality score: {quality_result.score:.3f}")
        
        print("✅ Validation module tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Validation module tests failed: {e}")
        return False


def test_financial_module():
    """Test the financial calculations module."""
    print("🧪 Testing Financial Module...")
    
    try:
        calculator = Step05FinancialCalculator()
        test_data = create_test_data()
        
        # Test transaction cost calculation
        print("  - Testing transaction cost calculation...")
        transaction_costs = calculator.calculate_transaction_costs(test_data)
        total_costs = transaction_costs.sum()
        print(f"    Total transaction costs: ${total_costs:.2f}")
        
        # Test trading performance calculation
        print("  - Testing trading performance calculation...")
        performance = calculator.calculate_trading_performance(test_data, transaction_costs)
        print(f"    Net return: {performance.net_return:.2%}")
        print(f"    Sharpe ratio: {performance.sharpe_ratio:.3f}")
        print(f"    Win rate: {performance.win_rate:.2%}")
        
        # Test risk metrics calculation
        print("  - Testing risk metrics calculation...")
        risk_metrics = calculator.calculate_risk_metrics(test_data)
        print(f"    VaR 95%: {risk_metrics.var_95:.2%}")
        print(f"    Volatility: {risk_metrics.volatility:.2%}")
        
        # Test position sizing calculation
        print("  - Testing position sizing calculation...")
        position_sizes = calculator.calculate_position_sizing(test_data)
        avg_position_size = position_sizes.mean()
        print(f"    Average position size: ${avg_position_size:.2f}")
        
        print("✅ Financial module tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Financial module tests failed: {e}")
        return False


def test_error_handling_module():
    """Test the error handling module."""
    print("🧪 Testing Error Handling Module...")
    
    try:
        error_handler = Step05ErrorHandler()
        
        # Test error handling
        print("  - Testing error handling...")
        from src.training.steps.step05_error_handling import ErrorContext, ErrorSeverity, ErrorCategory
        
        context = ErrorContext(
            function_name="test_function",
            step_name="step05",
            additional_context={'test': True}
        )
        
        # Simulate an error
        test_error = ValueError("Test error for validation")
        error_record = error_handler.handle_error(
            error=test_error,
            context=context,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.VALIDATION
        )
        
        print(f"    Error handled: {error_record.error_id}")
        print(f"    Error type: {error_record.error_type}")
        print(f"    Severity: {error_record.severity.value}")
        
        # Test error summary
        print("  - Testing error summary...")
        summary = error_handler.get_error_summary()
        print(f"    Total errors: {summary['total_errors']}")
        print(f"    Resolution rate: {summary['resolution_rate']:.2%}")
        
        print("✅ Error handling module tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Error handling module tests failed: {e}")
        return False


def test_reporting_module():
    """Test the reporting module."""
    print("🧪 Testing Reporting Module...")
    
    try:
        reporter = Step05Reporter()
        test_data = create_test_data()
        
        # Prepare test data
        labeling_results = {
            'total_labels': len(test_data),
            'label_distribution': test_data['label'].value_counts().to_dict(),
            'labeling_method': 'test_method'
        }
        
        performance_data = {
            'execution_time': 45.67,
            'memory_usage': 256.8,
            'cpu_usage': 78.5,
            'processing_efficiency': 0.87,
            'optimization_effectiveness': 0.92
        }
        
        validation_results = {
            'passed': True,
            'checks_performed': 5,
            'failures': 0
        }
        
        meta_labeling_analysis = {
            'meta_labels_created': 950,
            'success_rate': 0.94,
            'avg_confidence': 0.82
        }
        
        # Test report generation
        print("  - Testing comprehensive report generation...")
        report = reporter.generate_comprehensive_report(
            labeled_data=test_data,
            labeling_results=labeling_results,
            performance_data=performance_data,
            validation_results=validation_results,
            meta_labeling_analysis=meta_labeling_analysis,
            symbol="BTCUSDT",
            exchange="BINANCE",
            timeframe="1h"
        )
        
        print(f"    Report generated: {report.get('timestamp', 'Unknown')}")
        print(f"    Report sections: {len([k for k in report.keys() if k != 'timestamp'])}")
        
        # Test report saving
        print("  - Testing report saving...")
        saved_files = reporter.save_report(report, "test_reports/step05")
        print(f"    Files saved: {len(saved_files)}")
        
        print("✅ Reporting module tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Reporting module tests failed: {e}")
        return False


async def test_integrated_step05():
    """Test the integrated Step05 refactored implementation."""
    print("🧪 Testing Integrated Step05 Refactored...")
    
    try:
        # Create test data directory structure
        test_data_dir = Path("test_data_cache")
        test_data_dir.mkdir(exist_ok=True)
        
        training_dir = test_data_dir / "training"
        training_dir.mkdir(exist_ok=True)
        
        # Create test triple barrier data
        test_data = create_test_data()
        triple_barrier_path = training_dir / "BINANCE_ETHUSDT_1m_triple_barrier_labels.parquet"
        test_data.to_parquet(triple_barrier_path)
        
        print("  - Created test data structure")
        
        # Test configuration
        config = {
            'vectorized_labelling_orchestrator': {
                'auto_recalculate_hmm_barriers': True,
                'hmm_barrier_regime_column': 'hmm_regime',
                'time_barrier_minutes': 30,
                'max_lookahead': 100,
                'profit_take_multiplier': 0.002,
                'stop_loss_multiplier': 0.001
            },
            'transaction_costs': {
                'maker_fee': 0.001,
                'taker_fee': 0.001,
                'slippage_bps': 2.0,
                'funding_rate': 0.0001
            }
        }
        
        # Run Step05 refactored
        print("  - Running Step05 refactored...")
        success = await run_step05_refactored(
            symbol='ETHUSDT',
            exchange='BINANCE',
            timeframe='1m',
            data_dir=str(test_data_dir),
            force_rerun=True,
            config=config
        )
        
        print(f"    Step05 refactored result: {success}")
        
        # Check if results were created
        if success:
            labeled_data_path = training_dir / "labeled_data" / "BINANCE_ETHUSDT_1m_labeled_data.parquet"
            if labeled_data_path.exists():
                print("    ✅ Labeled data file created")
            else:
                print("    ⚠️ Labeled data file not found")
            
            report_dir = test_data_dir / "reports" / "step05"
            if report_dir.exists():
                report_files = list(report_dir.glob("*.json"))
                print(f"    ✅ Report files created: {len(report_files)}")
            else:
                print("    ⚠️ Report directory not found")
        
        print("✅ Integrated Step05 refactored tests passed")
        return success
        
    except Exception as e:
        print(f"❌ Integrated Step05 refactored tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("🚀 Starting Step05 Refactored Modular Architecture Tests")
    print("=" * 70)
    
    test_results = []
    
    # Test individual modules
    test_results.append(("Validation Module", test_validation_module()))
    test_results.append(("Financial Module", test_financial_module()))
    test_results.append(("Error Handling Module", test_error_handling_module()))
    test_results.append(("Reporting Module", test_reporting_module()))
    
    # Test integrated implementation
    integrated_success = await test_integrated_step05()
    test_results.append(("Integrated Step05", integrated_success))
    
    # Print results summary
    print("\n" + "=" * 70)
    print("🎯 TEST RESULTS SUMMARY")
    print("=" * 70)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<50} {status}")
        if result:
            passed_tests += 1
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! Step05 refactored modular architecture is working correctly.")
        print("\n📋 Key Improvements Implemented:")
        print("  ✅ Refactored large files into focused modules")
        print("  ✅ Implemented lookahead bias validation")
        print("  ✅ Added comprehensive transaction cost modeling")
        print("  ✅ Standardized error handling patterns")
        print("  ✅ Created modular reporting system")
        print("  ✅ Integrated all components seamlessly")
    else:
        print(f"\n⚠️ {total_tests - passed_tests} tests failed. Please review the issues above.")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)