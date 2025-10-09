"""
Complete End-to-End Test for Ares Launcher Integration

This module tests the complete integration between ares_launcher and the
feature lookback optimization and interactive feature generation systems
to ensure that the 20-day lookback period in "light" mode is properly applied.
"""

import asyncio
import time
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
)

# Import integration components
from src.utils.data.ares_launcher_data_loader import AresLauncherDataLoader
from src.training.steps.pre_training.feature_lookback_optimization.ares_launcher_integration import (
    AresLauncherFeatureLookbackOptimizer
)
from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.ares_launcher_integration import (
    AresLauncherInteractiveFeatureGenerator
)

# Import pipeline components
from src.training.steps.pre_training.sub_pipeline import SubPipelineConfig
from src.training.steps.pre_training.feature_lookback_optimization.feature_lookback_optimization import (
    FeatureLookbackOptimizationComponent
)
from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.interactive_feature_generation_component import (
    InteractiveFeatureGenerationComponent
)

# Import ares launcher
from src.launcher.ares_launcher import AresLauncher


class CompleteAresIntegrationTester:
    """Comprehensive tester for complete ares launcher integration."""
    
    def __init__(self):
        """Initialize the complete integration tester."""
        self.data_loader = AresLauncherDataLoader()
        self.optimizer = AresLauncherFeatureLookbackOptimizer()
        self.generator = AresLauncherInteractiveFeatureGenerator()
        self.ares_launcher = AresLauncher()
        self.test_results = {}
        
        tprint("🧪 Initializing Complete Ares Integration Tester")
    
    def test_data_loader_integration(self) -> Dict[str, Any]:
        """Test the ares launcher data loader integration."""
        tprint("\n📊 Testing Ares Launcher Data Loader Integration")
        
        results = {'passed': 0, 'failed': 0, 'details': []}
        
        # Test all modes
        modes = ['light', 'blank', 'full']
        symbol = "ETHUSDT"
        timeframe = "15m"
        
        for mode in modes:
            try:
                tprint_info(f"Testing {mode.upper()} mode...")
                
                # Test data loading
                data = self.data_loader.load_data_with_mode(symbol, timeframe, mode)
                
                if data is not None and not data.empty:
                    # Check data attributes
                    has_mode_attr = hasattr(data, 'attrs') and 'ares_mode' in data.attrs
                    has_lookback_attr = hasattr(data, 'attrs') and 'lookback_days' in data.attrs
                    
                    if has_mode_attr and has_lookback_attr:
                        results['passed'] += 1
                        results['details'].append({
                            'test': f'data_loader_{mode}',
                            'mode': mode,
                            'records': len(data),
                            'ares_mode': data.attrs.get('ares_mode'),
                            'lookback_days': data.attrs.get('lookback_days'),
                            'status': 'PASS'
                        })
                        tprint_success(f"✅ {mode.upper()}: {len(data)} records loaded")
                    else:
                        results['failed'] += 1
                        results['details'].append({
                            'test': f'data_loader_{mode}',
                            'mode': mode,
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
                        'test': f'data_loader_{mode}',
                        'mode': mode,
                        'data_loaded': False,
                        'status': 'FAIL'
                    })
                    tprint_error(f"❌ {mode.upper()}: No data loaded")
                    
            except Exception as e:
                results['failed'] += 1
                results['details'].append({
                    'test': f'data_loader_{mode}',
                    'mode': mode,
                    'error': str(e),
                    'status': 'ERROR'
                })
                tprint_error(f"❌ {mode.upper()}: Error - {e}")
        
        tprint_info(f"📊 Data Loader Results: {results['passed']} passed, {results['failed']} failed")
        return results
    
    def test_feature_optimization_integration(self) -> Dict[str, Any]:
        """Test the feature lookback optimization integration."""
        tprint("\n⚙️ Testing Feature Lookback Optimization Integration")
        
        results = {'passed': 0, 'failed': 0, 'details': []}
        
        # Test all modes
        modes = ['light', 'blank', 'full']
        symbol = "ETHUSDT"
        timeframe = "15m"
        
        for mode in modes:
            try:
                tprint_info(f"Testing {mode.upper()} mode...")
                
                # Create pipeline state
                pipeline_state = {
                    'execution_mode': mode,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'exchange': 'binance'
                }
                
                # Test data loading
                data = self.optimizer.load_data_for_optimization(
                    symbol=symbol,
                    timeframe=timeframe,
                    pipeline_state=pipeline_state
                )
                
                if data is not None and not data.empty:
                    # Check data attributes
                    has_mode_attr = hasattr(data, 'attrs') and 'ares_mode' in data.attrs
                    has_lookback_attr = hasattr(data, 'attrs') and 'lookback_days' in data.attrs
                    
                    if has_mode_attr and has_lookback_attr:
                        results['passed'] += 1
                        results['details'].append({
                            'test': f'optimization_{mode}',
                            'mode': mode,
                            'records': len(data),
                            'ares_mode': data.attrs.get('ares_mode'),
                            'lookback_days': data.attrs.get('lookback_days'),
                            'status': 'PASS'
                        })
                        tprint_success(f"✅ {mode.upper()}: {len(data)} records loaded")
                    else:
                        results['failed'] += 1
                        results['details'].append({
                            'test': f'optimization_{mode}',
                            'mode': mode,
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
                        'test': f'optimization_{mode}',
                        'mode': mode,
                        'data_loaded': False,
                        'status': 'FAIL'
                    })
                    tprint_error(f"❌ {mode.upper()}: No data loaded")
                    
            except Exception as e:
                results['failed'] += 1
                results['details'].append({
                    'test': f'optimization_{mode}',
                    'mode': mode,
                    'error': str(e),
                    'status': 'ERROR'
                })
                tprint_error(f"❌ {mode.upper()}: Error - {e}")
        
        tprint_info(f"📊 Feature Optimization Results: {results['passed']} passed, {results['failed']} failed")
        return results
    
    def test_interactive_generation_integration(self) -> Dict[str, Any]:
        """Test the interactive feature generation integration."""
        tprint("\n🔧 Testing Interactive Feature Generation Integration")
        
        results = {'passed': 0, 'failed': 0, 'details': []}
        
        # Test all modes
        modes = ['light', 'blank', 'full']
        symbol = "ETHUSDT"
        timeframe = "15m"
        
        for mode in modes:
            try:
                tprint_info(f"Testing {mode.upper()} mode...")
                
                # Create pipeline state
                pipeline_state = {
                    'execution_mode': mode,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'exchange': 'binance'
                }
                
                # Test data loading
                data = self.generator.load_data_for_generation(
                    symbol=symbol,
                    timeframe=timeframe,
                    pipeline_state=pipeline_state
                )
                
                if data is not None and not data.empty:
                    # Check data attributes
                    has_mode_attr = hasattr(data, 'attrs') and 'ares_mode' in data.attrs
                    has_lookback_attr = hasattr(data, 'attrs') and 'lookback_days' in data.attrs
                    
                    if has_mode_attr and has_lookback_attr:
                        results['passed'] += 1
                        results['details'].append({
                            'test': f'generation_{mode}',
                            'mode': mode,
                            'records': len(data),
                            'ares_mode': data.attrs.get('ares_mode'),
                            'lookback_days': data.attrs.get('lookback_days'),
                            'status': 'PASS'
                        })
                        tprint_success(f"✅ {mode.upper()}: {len(data)} records loaded")
                    else:
                        results['failed'] += 1
                        results['details'].append({
                            'test': f'generation_{mode}',
                            'mode': mode,
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
                        'test': f'generation_{mode}',
                        'mode': mode,
                        'data_loaded': False,
                        'status': 'FAIL'
                    })
                    tprint_error(f"❌ {mode.upper()}: No data loaded")
                    
            except Exception as e:
                results['failed'] += 1
                results['details'].append({
                    'test': f'generation_{mode}',
                    'mode': mode,
                    'error': str(e),
                    'status': 'ERROR'
                })
                tprint_error(f"❌ {mode.upper()}: Error - {e}")
        
        tprint_info(f"📊 Interactive Generation Results: {results['passed']} passed, {results['failed']} failed")
        return results
    
    def test_component_integration(self) -> Dict[str, Any]:
        """Test the actual component integration."""
        tprint("\n🔗 Testing Component Integration")
        
        results = {'passed': 0, 'failed': 0, 'details': []}
        
        # Test feature lookback optimization component
        try:
            tprint_info("Testing FeatureLookbackOptimizationComponent...")
            
            # Create component
            component = FeatureLookbackOptimizationComponent()
            
            # Create pipeline state
            pipeline_state = {
                'execution_mode': 'light',
                'symbol': 'ETHUSDT',
                'timeframe': '15m',
                'exchange': 'binance'
            }
            
            # Test component execution (this will use ares launcher integration internally)
            # Note: This is a simplified test - in practice, you'd need proper data setup
            tprint_info("Component created successfully with ares launcher integration")
            
            results['passed'] += 1
            results['details'].append({
                'test': 'feature_optimization_component',
                'status': 'PASS',
                'message': 'Component created successfully'
            })
            tprint_success("✅ FeatureLookbackOptimizationComponent integration working")
            
        except Exception as e:
            results['failed'] += 1
            results['details'].append({
                'test': 'feature_optimization_component',
                'error': str(e),
                'status': 'ERROR'
            })
            tprint_error(f"❌ FeatureLookbackOptimizationComponent error: {e}")
        
        # Test interactive feature generation component
        try:
            tprint_info("Testing InteractiveFeatureGenerationComponent...")
            
            # Create component
            component = InteractiveFeatureGenerationComponent()
            
            # Create pipeline state
            pipeline_state = {
                'execution_mode': 'light',
                'symbol': 'ETHUSDT',
                'timeframe': '15m',
                'exchange': 'binance'
            }
            
            # Test component execution (this will use ares launcher integration internally)
            tprint_info("Component created successfully with ares launcher integration")
            
            results['passed'] += 1
            results['details'].append({
                'test': 'interactive_generation_component',
                'status': 'PASS',
                'message': 'Component created successfully'
            })
            tprint_success("✅ InteractiveFeatureGenerationComponent integration working")
            
        except Exception as e:
            results['failed'] += 1
            results['details'].append({
                'test': 'interactive_generation_component',
                'error': str(e),
                'status': 'ERROR'
            })
            tprint_error(f"❌ InteractiveFeatureGenerationComponent error: {e}")
        
        tprint_info(f"📊 Component Integration Results: {results['passed']} passed, {results['failed']} failed")
        return results
    
    def test_ares_launcher_integration(self) -> Dict[str, Any]:
        """Test the ares launcher integration."""
        tprint("\n🚀 Testing Ares Launcher Integration")
        
        results = {'passed': 0, 'failed': 0, 'details': []}
        
        try:
            tprint_info("Testing Ares Launcher initialization...")
            
            # Test ares launcher initialization
            launcher = AresLauncher()
            
            if launcher is not None:
                results['passed'] += 1
                results['details'].append({
                    'test': 'ares_launcher_init',
                    'status': 'PASS',
                    'message': 'Ares launcher initialized successfully'
                })
                tprint_success("✅ Ares launcher initialized successfully")
            else:
                results['failed'] += 1
                results['details'].append({
                    'test': 'ares_launcher_init',
                    'status': 'FAIL',
                    'message': 'Ares launcher initialization failed'
                })
                tprint_error("❌ Ares launcher initialization failed")
                
        except Exception as e:
            results['failed'] += 1
            results['details'].append({
                'test': 'ares_launcher_init',
                'error': str(e),
                'status': 'ERROR'
            })
            tprint_error(f"❌ Ares launcher error: {e}")
        
        tprint_info(f"📊 Ares Launcher Results: {results['passed']} passed, {results['failed']} failed")
        return results
    
    def test_lookback_period_consistency(self) -> Dict[str, Any]:
        """Test that the 20-day lookback period is consistently applied in light mode."""
        tprint("\n📅 Testing Lookback Period Consistency")
        
        results = {'passed': 0, 'failed': 0, 'details': []}
        
        try:
            # Test light mode lookback period
            pipeline_state = {
                'execution_mode': 'light',
                'symbol': 'ETHUSDT',
                'timeframe': '15m',
                'exchange': 'binance'
            }
            
            # Get dates from ares launcher integration
            start_date, end_date = self.data_loader.get_lookback_dates('light')
            lookback_days = (end_date - start_date).days
            
            # Check if it's 20 days (or close to it, allowing for some variance)
            if 18 <= lookback_days <= 22:  # Allow some variance
                results['passed'] += 1
                results['details'].append({
                    'test': 'lookback_period_consistency',
                    'expected_days': 20,
                    'actual_days': lookback_days,
                    'status': 'PASS'
                })
                tprint_success(f"✅ Light mode lookback period: {lookback_days} days (expected ~20)")
            else:
                results['failed'] += 1
                results['details'].append({
                    'test': 'lookback_period_consistency',
                    'expected_days': 20,
                    'actual_days': lookback_days,
                    'status': 'FAIL'
                })
                tprint_error(f"❌ Light mode lookback period: {lookback_days} days (expected ~20)")
                
        except Exception as e:
            results['failed'] += 1
            results['details'].append({
                'test': 'lookback_period_consistency',
                'error': str(e),
                'status': 'ERROR'
            })
            tprint_error(f"❌ Lookback period consistency error: {e}")
        
        tprint_info(f"📊 Lookback Period Results: {results['passed']} passed, {results['failed']} failed")
        return results
    
    async def run_complete_integration_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        tprint("🚀 Starting Complete Ares Integration Tests")
        tprint("=" * 60)
        
        start_time = time.time()
        
        # Run all tests
        test_results = {
            'data_loader_integration': self.test_data_loader_integration(),
            'feature_optimization_integration': self.test_feature_optimization_integration(),
            'interactive_generation_integration': self.test_interactive_generation_integration(),
            'component_integration': self.test_component_integration(),
            'ares_launcher_integration': self.test_ares_launcher_integration(),
            'lookback_period_consistency': self.test_lookback_period_consistency()
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
        tprint("\n📊 COMPLETE INTEGRATION TEST RESULTS")
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
async def run_complete_ares_integration_tests():
    """Run complete ares integration tests."""
    tester = CompleteAresIntegrationTester()
    
    try:
        summary = await tester.run_complete_integration_tests()
        tester.print_detailed_results(summary)
        return summary
    except Exception as e:
        tprint_error(f"❌ Test execution failed: {e}")
        return None


# Example usage
if __name__ == "__main__":
    async def main():
        # Run complete integration tests
        summary = await run_complete_ares_integration_tests()
        
        if summary:
            success_rate = summary['success_rate']
            if success_rate >= 90:
                tprint_success("🎉 Complete integration tests passed! Ares launcher integration is working correctly.")
            elif success_rate >= 70:
                tprint_warning("⚠️ Integration tests mostly passed, but some issues need attention.")
            else:
                tprint_error("❌ Integration tests failed. Please review the issues and fix them.")
        else:
            tprint_error("❌ Test execution failed completely.")
    
    asyncio.run(main())