#!/usr/bin/env python3
"""
Training Pipeline Debug Script

This script runs comprehensive validation and debugging of the training pipeline
to identify and resolve silent failures.
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_error, tprint_warning
except ImportError:
    def tprint(msg, **kwargs):
        print(f"[{datetime.now()}] {msg}")
    tprint_info = tprint_success = tprint_error = tprint_warning = tprint

try:
    from src.training.utils.debug_utilities import TrainingDebugger, quick_dependency_check, quick_data_check, quick_system_check
    DEBUG_UTILITIES_AVAILABLE = True
except ImportError as e:
    tprint_error(f"❌ Debug utilities not available: {e}")
    DEBUG_UTILITIES_AVAILABLE = False

try:
    from src.training.steps.model_training.sub_pipeline import ModelTrainingSubPipeline, SubPipelineConfig, ExecutionMode
    MODEL_TRAINING_AVAILABLE = True
except ImportError as e:
    tprint_error(f"❌ Model training sub-pipeline not available: {e}")
    MODEL_TRAINING_AVAILABLE = False

def print_banner():
    """Print debug session banner."""
    tprint("=" * 80)
    tprint("🔍 TRAINING PIPELINE DEBUG SESSION")
    tprint("=" * 80)
    tprint(f"⏰ Started at: {datetime.now()}")
    tprint("")

async def run_comprehensive_validation():
    """Run comprehensive validation of the entire training pipeline."""
    tprint_info("🔍 Running comprehensive validation...")
    
    validation_results = {}
    
    # 1. Quick dependency check
    tprint_info("1️⃣ Checking dependencies...")
    try:
        deps_ok = quick_dependency_check()
        validation_results['dependencies'] = deps_ok
        if deps_ok:
            tprint_success("   ✅ Dependencies check passed")
        else:
            tprint_error("   ❌ Dependencies check failed")
    except Exception as e:
        tprint_error(f"   ❌ Dependencies check error: {e}")
        validation_results['dependencies'] = False
    
    # 2. Quick system check
    tprint_info("2️⃣ Checking system resources...")
    try:
        system_ok = quick_system_check()
        validation_results['system'] = system_ok
        if system_ok:
            tprint_success("   ✅ System resources check passed")
        else:
            tprint_error("   ❌ System resources check failed")
    except Exception as e:
        tprint_error(f"   ❌ System check error: {e}")
        validation_results['system'] = False
    
    # 3. Quick data check
    tprint_info("3️⃣ Checking data files...")
    try:
        data_ok = quick_data_check("historical_data", "ETHUSDT", "15m", "binance")
        validation_results['data'] = data_ok
        if data_ok:
            tprint_success("   ✅ Data files check passed")
        else:
            tprint_error("   ❌ Data files check failed")
    except Exception as e:
        tprint_error(f"   ❌ Data check error: {e}")
        validation_results['data'] = False
    
    # 4. Model training pipeline availability
    tprint_info("4️⃣ Checking model training pipeline...")
    validation_results['model_training_pipeline'] = MODEL_TRAINING_AVAILABLE
    if MODEL_TRAINING_AVAILABLE:
        tprint_success("   ✅ Model training pipeline available")
    else:
        tprint_error("   ❌ Model training pipeline not available")
    
    # 5. Debug utilities availability
    tprint_info("5️⃣ Checking debug utilities...")
    validation_results['debug_utilities'] = DEBUG_UTILITIES_AVAILABLE
    if DEBUG_UTILITIES_AVAILABLE:
        tprint_success("   ✅ Debug utilities available")
    else:
        tprint_error("   ❌ Debug utilities not available")
    
    return validation_results

async def test_training_step(step_name: str, mode: str = "light"):
    """Test a specific training step with enhanced debugging."""
    tprint_info(f"🧪 Testing training step: {step_name}")
    
    if not MODEL_TRAINING_AVAILABLE:
        tprint_error("   ❌ Model training pipeline not available")
        return False
    
    try:
        # Create configuration
        config = SubPipelineConfig(
            mode=ExecutionMode.LIGHT if mode == "light" else ExecutionMode.FULL,
            symbol="ETHUSDT",
            exchange="binance",
            timeframe="15m",
            data_dir="historical_data"
        )
        
        # Initialize pipeline
        pipeline = ModelTrainingSubPipeline(config)
        
        # Execute the step
        tprint_info(f"   🚀 Executing {step_name}...")
        result = await pipeline.execute_sub_pipeline(step_name, config)
        
        if result.status.value == "completed":
            tprint_success(f"   ✅ {step_name} completed successfully")
            tprint_info(f"   ⏱️  Duration: {result.duration_seconds:.2f}s")
            tprint_info(f"   📊 Artifacts: {len(result.artifacts)} types")
            return True
        else:
            tprint_error(f"   ❌ {step_name} failed: {result.error_message}")
            return False
            
    except Exception as e:
        tprint_error(f"   ❌ {step_name} test failed with exception: {str(e)}")
        tprint_error(f"   📋 Traceback: {traceback.format_exc()}")
        return False

async def run_training_pipeline_tests():
    """Run tests on all training pipeline steps."""
    tprint_info("🧪 Running training pipeline tests...")
    
    # Define test steps
    test_steps = [
        "analyst_model_training",
        "analyst_ensemble_training", 
        "tactician_models_training",
        "tactician_ensemble_training"
    ]
    
    results = {}
    
    for step in test_steps:
        tprint_info(f"Testing {step}...")
        try:
            success = await test_training_step(step, mode="light")
            results[step] = success
        except Exception as e:
            tprint_error(f"Test for {step} failed: {e}")
            results[step] = False
    
    return results

def save_debug_report(validation_results: dict, test_results: dict):
    """Save comprehensive debug report."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path(f"debug_reports/comprehensive_debug_report_{timestamp}.json")
        report_path.parent.mkdir(exist_ok=True)
        
        report = {
            "timestamp": timestamp,
            "validation_results": validation_results,
            "test_results": test_results,
            "summary": {
                "validation_passed": all(validation_results.values()),
                "tests_passed": all(test_results.values()) if test_results else False,
                "total_validations": len(validation_results),
                "passed_validations": sum(validation_results.values()),
                "total_tests": len(test_results),
                "passed_tests": sum(test_results.values()) if test_results else 0
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        tprint_success(f"📝 Debug report saved to: {report_path}")
        return report_path
        
    except Exception as e:
        tprint_error(f"Failed to save debug report: {e}")
        return None

async def main():
    """Main debug session."""
    print_banner()
    
    try:
        # Step 1: Run comprehensive validation
        tprint_info("PHASE 1: Comprehensive Validation")
        tprint("-" * 40)
        validation_results = await run_comprehensive_validation()
        
        tprint("")
        tprint_info("VALIDATION SUMMARY:")
        for check, passed in validation_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            tprint(f"   {check}: {status}")
        
        validation_passed = all(validation_results.values())
        if validation_passed:
            tprint_success("🎉 All validations passed!")
        else:
            tprint_error("❌ Some validations failed!")
        
        tprint("")
        
        # Step 2: Run training pipeline tests (only if basic validation passes)
        test_results = {}
        if validation_results.get('model_training_pipeline', False) and validation_results.get('debug_utilities', False):
            tprint_info("PHASE 2: Training Pipeline Tests")
            tprint("-" * 40)
            test_results = await run_training_pipeline_tests()
            
            tprint("")
            tprint_info("TEST SUMMARY:")
            for test, passed in test_results.items():
                status = "✅ PASS" if passed else "❌ FAIL"
                tprint(f"   {test}: {status}")
            
            tests_passed = all(test_results.values()) if test_results else False
            if tests_passed:
                tprint_success("🎉 All tests passed!")
            else:
                tprint_error("❌ Some tests failed!")
        else:
            tprint_warning("⚠️  Skipping training pipeline tests due to validation failures")
        
        tprint("")
        
        # Step 3: Save comprehensive report
        tprint_info("PHASE 3: Generating Debug Report")
        tprint("-" * 40)
        report_path = save_debug_report(validation_results, test_results)
        
        # Final summary
        tprint("")
        tprint("=" * 80)
        tprint("🏁 DEBUG SESSION SUMMARY")
        tprint("=" * 80)
        
        total_checks = len(validation_results) + len(test_results)
        passed_checks = sum(validation_results.values()) + sum(test_results.values())
        
        tprint(f"📊 Overall Status: {passed_checks}/{total_checks} checks passed")
        tprint(f"✅ Validations: {sum(validation_results.values())}/{len(validation_results)}")
        tprint(f"🧪 Tests: {sum(test_results.values())}/{len(test_results)}")
        
        if report_path:
            tprint(f"📝 Report: {report_path}")
        
        if passed_checks == total_checks:
            tprint_success("🎉 ALL CHECKS PASSED - Training pipeline is ready!")
        else:
            tprint_error("❌ SOME CHECKS FAILED - See report for details")
        
        tprint("=" * 80)
        
    except Exception as e:
        tprint_error(f"❌ Debug session failed: {str(e)}")
        tprint_error(f"📋 Traceback: {traceback.format_exc()}")
        return False
    
    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        tprint_warning("⚠️  Debug session interrupted by user")
        sys.exit(1)
    except Exception as e:
        tprint_error(f"❌ Fatal error: {str(e)}")
        sys.exit(1)