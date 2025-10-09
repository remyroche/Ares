"""
Test Ares Integration with Thorough Tprint Logging

This module demonstrates the comprehensive tprint logging that has been
integrated throughout the ares launcher integration system.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any

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


class AresIntegrationTprintDemo:
    """Demonstration of ares integration with thorough tprint logging."""
    
    def __init__(self):
        """Initialize the demo."""
        self.data_loader = AresLauncherDataLoader()
        self.optimizer = AresLauncherFeatureLookbackOptimizer()
        self.generator = AresLauncherInteractiveFeatureGenerator()
        
        tprint("🚀 [DEMO] Initializing Ares Integration Tprint Demo")
        tprint_info("📊 [DEMO] This demo will show comprehensive tprint logging throughout the integration")
    
    def demo_data_loader_logging(self):
        """Demonstrate data loader tprint logging."""
        tprint("\n" + "="*80)
        tprint("📊 [DEMO] Testing AresLauncherDataLoader with tprint logging")
        tprint("="*80)
        
        # Test mode detection and date calculation
        tprint("🔍 [DEMO] Testing mode detection and date calculation...")
        for mode in ['light', 'blank', 'full']:
            tprint(f"\n--- Testing {mode.upper()} mode ---")
            start_date, end_date = self.data_loader.get_lookback_dates(mode)
            tprint_info(f"✅ [DEMO] {mode.upper()} mode dates calculated successfully")
        
        # Test data loading (this will show extensive logging)
        tprint("\n📥 [DEMO] Testing data loading with extensive logging...")
        symbol = "ETHUSDT"
        timeframe = "15m"
        
        for mode in ['light', 'blank', 'full']:
            tprint(f"\n--- Testing data loading for {mode.upper()} mode ---")
            try:
                data = self.data_loader.load_data_with_mode(symbol, timeframe, mode)
                if data is not None:
                    tprint_success(f"✅ [DEMO] Data loaded successfully for {mode.upper()} mode")
                else:
                    tprint_warning(f"⚠️ [DEMO] No data loaded for {mode.upper()} mode")
            except Exception as e:
                tprint_error(f"❌ [DEMO] Error loading data for {mode.upper()} mode: {e}")
    
    def demo_optimizer_logging(self):
        """Demonstrate optimizer tprint logging."""
        tprint("\n" + "="*80)
        tprint("⚙️ [DEMO] Testing AresLauncherFeatureLookbackOptimizer with tprint logging")
        tprint("="*80)
        
        symbol = "ETHUSDT"
        timeframe = "15m"
        
        # Test different pipeline states
        test_states = [
            {'execution_mode': 'light'},
            {'lookback_days': 20},
            {'intensity_percentage': 0.025},
            {'execution_mode': 'blank', 'lookback_days': 180},
            {'execution_mode': 'full', 'intensity_percentage': 1.0}
        ]
        
        for i, pipeline_state in enumerate(test_states, 1):
            tprint(f"\n--- Test Case {i}: {pipeline_state} ---")
            try:
                # This will show extensive logging for mode detection and data loading
                data = self.optimizer.load_data_for_optimization(
                    symbol=symbol,
                    timeframe=timeframe,
                    pipeline_state=pipeline_state
                )
                if data is not None:
                    tprint_success(f"✅ [DEMO] Optimizer data loaded successfully for test case {i}")
                else:
                    tprint_warning(f"⚠️ [DEMO] No data loaded for test case {i}")
            except Exception as e:
                tprint_error(f"❌ [DEMO] Error in test case {i}: {e}")
    
    def demo_generator_logging(self):
        """Demonstrate generator tprint logging."""
        tprint("\n" + "="*80)
        tprint("🔧 [DEMO] Testing AresLauncherInteractiveFeatureGenerator with tprint logging")
        tprint("="*80)
        
        symbol = "ETHUSDT"
        timeframe = "15m"
        
        # Test different pipeline states
        test_states = [
            {'execution_mode': 'light'},
            {'lookback_days': 20},
            {'intensity_percentage': 0.025},
            {'execution_mode': 'blank', 'lookback_days': 180},
            {'execution_mode': 'full', 'intensity_percentage': 1.0}
        ]
        
        for i, pipeline_state in enumerate(test_states, 1):
            tprint(f"\n--- Test Case {i}: {pipeline_state} ---")
            try:
                # This will show extensive logging for mode detection and data loading
                data = self.generator.load_data_for_generation(
                    symbol=symbol,
                    timeframe=timeframe,
                    pipeline_state=pipeline_state
                )
                if data is not None:
                    tprint_success(f"✅ [DEMO] Generator data loaded successfully for test case {i}")
                else:
                    tprint_warning(f"⚠️ [DEMO] No data loaded for test case {i}")
            except Exception as e:
                tprint_error(f"❌ [DEMO] Error in test case {i}: {e}")
    
    def demo_parameter_adaptation_logging(self):
        """Demonstrate parameter adaptation logging."""
        tprint("\n" + "="*80)
        tprint("⚙️ [DEMO] Testing Parameter Adaptation with tprint logging")
        tprint("="*80)
        
        # Test optimizer parameters
        tprint("🔍 [DEMO] Testing optimizer parameter adaptation...")
        for mode in ['light', 'blank', 'full']:
            pipeline_state = {'execution_mode': mode}
            tprint(f"\n--- Testing {mode.upper()} mode parameters ---")
            try:
                params = self.optimizer.get_optimization_parameters(pipeline_state)
                tprint_success(f"✅ [DEMO] {mode.upper()} mode parameters retrieved successfully")
                tprint_info(f"   → Lookback days: {params['lookback_days']}")
                tprint_info(f"   → Intensity: {params['intensity_percentage']:.1%}")
                tprint_info(f"   → Max trials: {params['max_trials']}")
            except Exception as e:
                tprint_error(f"❌ [DEMO] Error getting {mode.upper()} parameters: {e}")
        
        # Test generator parameters
        tprint("\n🔧 [DEMO] Testing generator parameter adaptation...")
        for mode in ['light', 'blank', 'full']:
            pipeline_state = {'execution_mode': mode}
            tprint(f"\n--- Testing {mode.upper()} mode parameters ---")
            try:
                params = self.generator.get_generation_parameters(pipeline_state)
                tprint_success(f"✅ [DEMO] {mode.upper()} mode parameters retrieved successfully")
                tprint_info(f"   → Lookback days: {params['lookback_days']}")
                tprint_info(f"   → Feature budget (pre): {params['feature_budget_pre']}")
                tprint_info(f"   → Feature budget (post): {params['feature_budget_post']}")
                tprint_info(f"   → Interactions cap: {params['interactions_cap']}")
                tprint_info(f"   → Max workers: {params['max_workers']}")
            except Exception as e:
                tprint_error(f"❌ [DEMO] Error getting {mode.upper()} parameters: {e}")
    
    def demo_mode_detection_logging(self):
        """Demonstrate mode detection logging."""
        tprint("\n" + "="*80)
        tprint("🔍 [DEMO] Testing Mode Detection with tprint logging")
        tprint("="*80)
        
        # Test different detection scenarios
        test_cases = [
            {'name': 'Explicit mode', 'state': {'execution_mode': 'light'}},
            {'name': 'Lookback days inference', 'state': {'lookback_days': 20}},
            {'name': 'Intensity inference', 'state': {'intensity_percentage': 0.025}},
            {'name': 'Mixed indicators', 'state': {'execution_mode': 'blank', 'lookback_days': 180}},
            {'name': 'Conflicting indicators', 'state': {'execution_mode': 'light', 'lookback_days': 1000}},
            {'name': 'Empty state', 'state': {}},
        ]
        
        for test_case in test_cases:
            tprint(f"\n--- {test_case['name']} ---")
            tprint_debug(f"   → Pipeline state: {test_case['state']}")
            
            # Test optimizer mode detection
            try:
                mode = self.optimizer.detect_execution_mode(test_case['state'])
                tprint_success(f"✅ [DEMO] Optimizer detected mode: {mode.upper()}")
            except Exception as e:
                tprint_error(f"❌ [DEMO] Optimizer mode detection error: {e}")
            
            # Test generator mode detection
            try:
                mode = self.generator.detect_execution_mode(test_case['state'])
                tprint_success(f"✅ [DEMO] Generator detected mode: {mode.upper()}")
            except Exception as e:
                tprint_error(f"❌ [DEMO] Generator mode detection error: {e}")
    
    def demo_error_handling_logging(self):
        """Demonstrate error handling logging."""
        tprint("\n" + "="*80)
        tprint("❌ [DEMO] Testing Error Handling with tprint logging")
        tprint("="*80)
        
        # Test invalid inputs
        invalid_cases = [
            {'name': 'Invalid symbol', 'symbol': 'INVALID', 'timeframe': '15m', 'mode': 'light'},
            {'name': 'Invalid timeframe', 'symbol': 'ETHUSDT', 'timeframe': 'invalid', 'mode': 'light'},
            {'name': 'Invalid mode', 'symbol': 'ETHUSDT', 'timeframe': '15m', 'mode': 'invalid'},
            {'name': 'None symbol', 'symbol': None, 'timeframe': '15m', 'mode': 'light'},
            {'name': 'Empty pipeline state', 'symbol': 'ETHUSDT', 'timeframe': '15m', 'mode': 'light', 'state': {}},
        ]
        
        for case in invalid_cases:
            tprint(f"\n--- Testing {case['name']} ---")
            try:
                if 'state' in case:
                    pipeline_state = case['state']
                else:
                    pipeline_state = {'execution_mode': case['mode']}
                
                # This should trigger error handling and show detailed logging
                data = self.data_loader.load_data_with_mode(
                    symbol=case['symbol'],
                    interval=case['timeframe'],
                    mode=case['mode']
                )
                
                if data is not None:
                    tprint_success(f"✅ [DEMO] Unexpected success for {case['name']}")
                else:
                    tprint_warning(f"⚠️ [DEMO] Expected failure for {case['name']}")
                    
            except Exception as e:
                tprint_error(f"❌ [DEMO] Expected error for {case['name']}: {e}")
                tprint_debug(f"   → Exception type: {type(e).__name__}")
    
    def run_complete_demo(self):
        """Run the complete demonstration."""
        tprint("🚀 [DEMO] Starting Complete Ares Integration Tprint Demo")
        tprint("="*80)
        tprint("📊 [DEMO] This demo will show comprehensive tprint logging throughout")
        tprint("📊 [DEMO] the ares launcher integration system, demonstrating:")
        tprint("   → Mode detection and logging")
        tprint("   → Data loading with detailed progress")
        tprint("   → Parameter adaptation logging")
        tprint("   → Error handling and debugging")
        tprint("   → Component integration logging")
        tprint("="*80)
        
        try:
            # Run all demos
            self.demo_mode_detection_logging()
            self.demo_data_loader_logging()
            self.demo_optimizer_logging()
            self.demo_generator_logging()
            self.demo_parameter_adaptation_logging()
            self.demo_error_handling_logging()
            
            tprint("\n" + "="*80)
            tprint("🎉 [DEMO] Complete Ares Integration Tprint Demo Finished")
            tprint("="*80)
            tprint_success("✅ [DEMO] All demonstrations completed successfully")
            tprint_info("📊 [DEMO] The tprint logging provides comprehensive visibility into:")
            tprint_info("   → Mode detection process and confidence levels")
            tprint_info("   → Data loading progress and parameters")
            tprint_info("   → Parameter adaptation based on execution mode")
            tprint_info("   → Error handling and debugging information")
            tprint_info("   → Component integration and data flow")
            tprint("="*80)
            
        except Exception as e:
            tprint_error(f"❌ [DEMO] Demo execution failed: {e}")
            tprint_debug(f"   → Exception type: {type(e).__name__}")
            tprint_debug(f"   → Exception details: {str(e)}")


# Convenience function for running the demo
def run_ares_integration_tprint_demo():
    """Run the ares integration tprint demo."""
    demo = AresIntegrationTprintDemo()
    demo.run_complete_demo()


# Example usage
if __name__ == "__main__":
    run_ares_integration_tprint_demo()