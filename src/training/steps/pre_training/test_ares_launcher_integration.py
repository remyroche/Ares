"""
Comprehensive Test for Ares Launcher Integration

This module tests the integration between ares_launcher and the feature
lookback optimization and interactive feature generation systems to ensure
that the 20-day lookback period in "light" mode is properly applied.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
)

# Import integration modules
from src.utils.data.ares_launcher_data_loader import AresLauncherDataLoader
from src.training.steps.pre_training.feature_lookback_optimization.ares_launcher_integration import (
    AresLauncherFeatureLookbackOptimizer
)
from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.ares_launcher_integration import (
    AresLauncherInteractiveFeatureGenerator
)
from src.config.pipeline_modes import get_light_mode_config, get_blank_mode_config, get_full_mode_config


class AresLauncherIntegrationTester:
    """Comprehensive tester for ares launcher integration."""
    
    def __init__(self):
        """Initialize the integration tester."""
        self.data_loader = AresLauncherDataLoader()
        self.optimizer = AresLauncherFeatureLookbackOptimizer()
        self.generator = AresLauncherInteractiveFeatureGenerator()
        self.test_results = {}
        
        tprint("🧪 Initializing Ares Launcher Integration Tester")
    
    def test_mode_detection(self) -> Dict[str, Any]:
        """Test mode detection logic."""
        tprint("\n🔍 Testing Mode Detection Logic")
        
        test_cases = [
            # Explicit mode
            {'execution_mode': 'light', 'expected': 'light'},
            {'execution_mode': 'blank', 'expected': 'blank'},
            {'execution_mode': 'full', 'expected': 'full'},
            
            # Inferred from lookback days
            {'lookback_days': 20, 'expected': 'light'},
            {'lookback_days': 180, 'expected': 'blank'},
            {'lookback_days': 1460, 'expected': 'full'},
            
            # Inferred from intensity
            {'intensity_percentage': 0.025, 'expected': 'light'},
            {'intensity_percentage': 0.1, 'expected': 'blank'},
            {'intensity_percentage': 1.0, 'expected': 'full'},
            
            # Default fallback
            {},  # Should default to 'light'
        ]
        
        results = {'passed': 0, 'failed': 0, 'details': []}
        
        for i, test_case in enumerate(test_cases):
            expected = test_case.pop('expected')
            
            # Test optimizer mode detection
            detected_mode = self.optimizer.detect_execution_mode(test_case)
            
            if detected_mode == expected:
                results['passed'] += 1
                results['details'].append({
                    'test_case': i + 1,
                    'input': test_case,
                    'expected': expected,
                    'detected': detected_mode,
                    'status': 'PASS'
                })
                tprint_success(f"✅ Test {i + 1}: {test_case} -> {detected_mode}")
            else:
                results['failed'] += 1
                results['details'].append({
                    'test_case': i + 1,
                    'input': test_case,
                    'expected': expected,
                    'detected': detected_mode,
                    'status': 'FAIL'
                })
                tprint_error(f"❌ Test {i + 1}: {test_case} -> {detected_mode} (expected {expected})")
        
        tprint_info(f"📊 Mode Detection Results: {results['passed']} passed, {results['failed']} failed")
        return results
    
    def test_date_calculation(self) -> Dict[str, Any]:
        """Test date calculation for different modes."""
        tprint("\n📅 Testing Date Calculation")
        
        modes = ['light', 'blank', 'full']
        results = {'passed': 0, 'failed': 0, 'details': []}
        
        for mode in modes:
            try:
                # Get expected lookback days
                if mode == 'light':
                    expected_days = get_light_mode_config().lookback_days
                elif mode == 'blank':
                    expected_days = get_blank_mode_config().lookback_days
                else:  # full
                    expected_days = get_full_mode_config().lookback_days
                
                # Calculate dates
                start_date, end_date = self.data_loader.get_lookback_dates(mode)
                calculated_days = (end_date - start_date).days
                
                if calculated_days == expected_days:
                    results['passed'] += 1
                    results['details'].append({
                        'mode': mode,
                        'expected_days': expected_days,
                        'calculated_days': calculated_days,
                        'start_date': start_date.date(),
                        'end_date': end_date.date(),
                        'status': 'PASS'
                    })
                    tprint_success(f"✅ {mode.upper()}: {calculated_days} days ({start_date.date()} to {end_date.date()})")
                else:
                    results['failed'] += 1
                    results['details'].append({
                        'mode': mode,
                        'expected_days': expected_days,
                        'calculated_days': calculated_days,
                        'start_date': start_date.date(),
                        'end_date': end_date.date(),
                        'status': 'FAIL'
                    })
                    tprint_error(f"❌ {mode.upper()}: {calculated_days} days (expected {expected_days})")
                    
            except Exception as e:
                results['failed'] += 1
                results['details'].append({
                    'mode': mode,
                    'error': str(e),
                    'status': 'ERROR'
                })
                tprint_error(f"❌ {mode.upper()}: Error - {e}")
        
        tprint_info(f"📊 Date Calculation Results: {results['passed']} passed, {results['failed']} failed")
        return results
    
    def test_parameter_adaptation(self) -> Dict[str, Any]:
        """Test parameter adaptation for different modes."""
        tprint("\n⚙️ Testing Parameter Adaptation")
        
        modes = ['light', 'blank', 'full']
        results = {'passed': 0, 'failed': 0, 'details': []}
        
        for mode in modes:
            try:
                pipeline_state = {'execution_mode': mode}
                
                # Test optimizer parameters
                opt_params = self.optimizer.get_optimization_parameters(pipeline_state)
                
                # Test generator parameters
                gen_params = self.generator.get_generation_parameters(pipeline_state)
                
                # Validate parameters
                valid = True
                issues = []
                
                # Check mode consistency
                if opt_params['mode'] != mode or gen_params['mode'] != mode:
                    valid = False
                    issues.append("Mode mismatch")
                
                # Check lookback days consistency
                if opt_params['lookback_days'] != gen_params['lookback_days']:
                    valid = False
                    issues.append("Lookback days mismatch")
                
                # Check feature budget progression (light < blank < full)
                if mode == 'light' and gen_params['feature_budget_pre'] >= 100:
                    valid = False
                    issues.append("Light mode feature budget too high")
                elif mode == 'blank' and gen_params['feature_budget_pre'] >= 150:
                    valid = False
                    issues.append("Blank mode feature budget too high")
                elif mode == 'full' and gen_params['feature_budget_pre'] < 100:
                    valid = False
                    issues.append("Full mode feature budget too low")
                
                if valid:
                    results['passed'] += 1
                    results['details'].append({
                        'mode': mode,
                        'optimizer_params': opt_params,
                        'generator_params': gen_params,
                        'status': 'PASS'
                    })
                    tprint_success(f"✅ {mode.upper()}: Parameters adapted correctly")
                else:
                    results['failed'] += 1
                    results['details'].append({
                        'mode': mode,
                        'issues': issues,
                        'optimizer_params': opt_params,
                        'generator_params': gen_params,
                        'status': 'FAIL'
                    })
                    tprint_error(f"❌ {mode.upper()}: Parameter issues - {', '.join(issues)}")
                    
            except Exception as e:
                results['failed'] += 1
                results['details'].append({
                    'mode': mode,
                    'error': str(e),
                    'status': 'ERROR'
                })
                tprint_error(f"❌ {mode.upper()}: Error - {e}")
        
        tprint_info(f"📊 Parameter Adaptation Results: {results['passed']} passed, {results['failed']} failed")
        return results
    
    def test_data_loading(self) -> Dict[str, Any]:
        """Test data loading for different modes."""
        tprint("\n📊 Testing Data Loading")
        
        # Test with a common symbol and timeframe
        symbol = "ETHUSDT"
        timeframe = "15m"
        modes = ['light', 'blank', 'full']
        
        results = {'passed': 0, 'failed': 0, 'details': []}
        
        for mode in modes:
            try:
                # Test data loader
                data = self.data_loader.load_data_with_mode(symbol, timeframe, mode)
                
                if data is not None and not data.empty:
                    # Check data attributes
                    has_mode_attr = hasattr(data, 'attrs') and 'ares_mode' in data.attrs
                    has_lookback_attr = hasattr(data, 'attrs') and 'lookback_days' in data.attrs
                    
                    if has_mode_attr and has_lookback_attr:
                        results['passed'] += 1
                        results['details'].append({
                            'mode': mode,
                            'records': len(data),
                            'date_range': f"{data.index.min().date()} to {data.index.max().date()}",
                            'ares_mode': data.attrs.get('ares_mode'),
                            'lookback_days': data.attrs.get('lookback_days'),
                            'status': 'PASS'
                        })
                        tprint_success(f"✅ {mode.upper()}: {len(data)} records loaded")
                    else:
                        results['failed'] += 1
                        results['details'].append({
                            'mode': mode,
                            'records': len(data),
                            'missing_attrs': {
                                'ares_mode': not has_mode_attr,
                                'lookback_days': not has_lookback_attr
                            },
                            'status': 'FAIL'
                        })
                        tprint_error(f"❌ {mode.upper()}: Missing data attributes")
                else:
                    results['failed'] += 1
                    results['details'].append({
                        'mode': mode,
                        'data_loaded': False,
                        'status': 'FAIL'
                    })
                    tprint_error(f"❌ {mode.upper()}: No data loaded")
                    
            except Exception as e:
                results['failed'] += 1
                results['details'].append({
                    'mode': mode,
                    'error': str(e),
                    'status': 'ERROR'
                })
                tprint_error(f"❌ {mode.upper()}: Error - {e}")
        
        tprint_info(f"📊 Data Loading Results: {results['passed']} passed, {results['failed']} failed")
        return results
    
    def test_integration_consistency(self) -> Dict[str, Any]:
        """Test consistency between different integration components."""
        tprint("\n🔄 Testing Integration Consistency")
        
        pipeline_state = {'execution_mode': 'light'}
        symbol = "ETHUSDT"
        timeframe = "15m"
        
        results = {'passed': 0, 'failed': 0, 'details': []}
        
        try:
            # Test mode detection consistency
            opt_mode = self.optimizer.detect_execution_mode(pipeline_state)
            gen_mode = self.generator.detect_execution_mode(pipeline_state)
            
            if opt_mode == gen_mode:
                results['passed'] += 1
                results['details'].append({
                    'test': 'mode_detection_consistency',
                    'optimizer_mode': opt_mode,
                    'generator_mode': gen_mode,
                    'status': 'PASS'
                })
                tprint_success("✅ Mode detection consistent between components")
            else:
                results['failed'] += 1
                results['details'].append({
                    'test': 'mode_detection_consistency',
                    'optimizer_mode': opt_mode,
                    'generator_mode': gen_mode,
                    'status': 'FAIL'
                })
                tprint_error(f"❌ Mode detection inconsistent: {opt_mode} vs {gen_mode}")
            
            # Test parameter consistency
            opt_params = self.optimizer.get_optimization_parameters(pipeline_state)
            gen_params = self.generator.get_generation_parameters(pipeline_state)
            
            if opt_params['lookback_days'] == gen_params['lookback_days']:
                results['passed'] += 1
                results['details'].append({
                    'test': 'lookback_days_consistency',
                    'optimizer_lookback': opt_params['lookback_days'],
                    'generator_lookback': gen_params['lookback_days'],
                    'status': 'PASS'
                })
                tprint_success("✅ Lookback days consistent between components")
            else:
                results['failed'] += 1
                results['details'].append({
                    'test': 'lookback_days_consistency',
                    'optimizer_lookback': opt_params['lookback_days'],
                    'generator_lookback': gen_params['lookback_days'],
                    'status': 'FAIL'
                })
                tprint_error(f"❌ Lookback days inconsistent: {opt_params['lookback_days']} vs {gen_params['lookback_days']}")
            
            # Test date range consistency
            opt_start, opt_end = opt_params['start_date'], opt_params['end_date']
            gen_start, gen_end = gen_params['start_date'], gen_params['end_date']
            
            if opt_start == gen_start and opt_end == gen_end:
                results['passed'] += 1
                results['details'].append({
                    'test': 'date_range_consistency',
                    'optimizer_range': f"{opt_start.date()} to {opt_end.date()}",
                    'generator_range': f"{gen_start.date()} to {gen_end.date()}",
                    'status': 'PASS'
                })
                tprint_success("✅ Date range consistent between components")
            else:
                results['failed'] += 1
                results['details'].append({
                    'test': 'date_range_consistency',
                    'optimizer_range': f"{opt_start.date()} to {opt_end.date()}",
                    'generator_range': f"{gen_start.date()} to {gen_end.date()}",
                    'status': 'FAIL'
                })
                tprint_error(f"❌ Date range inconsistent")
                
        except Exception as e:
            results['failed'] += 1
            results['details'].append({
                'test': 'integration_consistency',
                'error': str(e),
                'status': 'ERROR'
            })
            tprint_error(f"❌ Integration consistency test failed: {e}")
        
        tprint_info(f"📊 Integration Consistency Results: {results['passed']} passed, {results['failed']} failed")
        return results
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        tprint("🚀 Starting Comprehensive Ares Launcher Integration Tests")
        tprint("=" * 60)
        
        start_time = time.time()
        
        # Run all tests
        test_results = {
            'mode_detection': self.test_mode_detection(),
            'date_calculation': self.test_date_calculation(),
            'parameter_adaptation': self.test_parameter_adaptation(),
            'data_loading': self.test_data_loading(),
            'integration_consistency': self.test_integration_consistency()
        }
        
        # Calculate summary
        total_passed = sum(result['passed'] for result in test_results.values())
        total_failed = sum(result['failed'] for result in test_results.values())
        total_tests = total_passed + total_failed
        
        execution_time = time.time() - start_time
        
        # Create summary
        summary = {
            'total_tests': total_tests,
            'passed': total_passed,
            'failed': total_failed,
            'success_rate': (total_passed / total_tests * 100) if total_tests > 0 else 0,
            'execution_time': execution_time,
            'test_results': test_results
        }
        
        # Print summary
        tprint("\n📊 COMPREHENSIVE TEST RESULTS")
        tprint("=" * 60)
        tprint(f"Total Tests: {total_tests}")
        tprint(f"Passed: {total_passed}")
        tprint(f"Failed: {total_failed}")
        tprint(f"Success Rate: {summary['success_rate']:.1f}%")
        tprint(f"Execution Time: {execution_time:.3f}s")
        
        # Print detailed results
        for test_name, result in test_results.items():
            status = "✅ PASS" if result['failed'] == 0 else "❌ FAIL"
            tprint(f"\n{test_name.replace('_', ' ').title()}: {status}")
            tprint(f"   Passed: {result['passed']}, Failed: {result['failed']}")
        
        return summary
    
    def print_detailed_results(self, summary: Dict[str, Any]):
        """Print detailed test results."""
        tprint("\n📋 DETAILED TEST RESULTS")
        tprint("=" * 60)
        
        for test_name, result in summary['test_results'].items():
            tprint(f"\n🔍 {test_name.replace('_', ' ').title()}")
            tprint("-" * 40)
            
            for detail in result['details']:
                if detail['status'] == 'PASS':
                    tprint_success(f"✅ {detail}")
                elif detail['status'] == 'FAIL':
                    tprint_error(f"❌ {detail}")
                else:  # ERROR
                    tprint_error(f"💥 {detail}")


# Convenience function for running tests
async def run_ares_launcher_integration_tests():
    """Run comprehensive ares launcher integration tests."""
    tester = AresLauncherIntegrationTester()
    
    try:
        summary = await tester.run_comprehensive_tests()
        tester.print_detailed_results(summary)
        return summary
    except Exception as e:
        tprint_error(f"❌ Test execution failed: {e}")
        return None


# Example usage
if __name__ == "__main__":
    async def main():
        # Run comprehensive tests
        summary = await run_ares_launcher_integration_tests()
        
        if summary:
            success_rate = summary['success_rate']
            if success_rate >= 90:
                tprint_success("🎉 Integration tests passed! Ares launcher integration is working correctly.")
            elif success_rate >= 70:
                tprint_warning("⚠️ Integration tests mostly passed, but some issues need attention.")
            else:
                tprint_error("❌ Integration tests failed. Please review the issues and fix them.")
        else:
            tprint_error("❌ Test execution failed completely.")
    
    asyncio.run(main())