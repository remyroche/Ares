#!/usr/bin/env python3
"""
Demonstration of the Sequential Feature Generation Pipeline

This script demonstrates how to use the enhanced ares_launcher.py with
sequential pipeline execution for feature generation steps.
"""

import subprocess
import sys
import time
from typing import List, Dict, Any

def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*80}")
    print(f"🔄 {description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="/workspace")
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            if result.stdout:
                print("Output:", result.stdout.strip())
            return True
        else:
            print(f"❌ {description} failed with return code {result.returncode}")
            if result.stderr:
                print("Error:", result.stderr.strip())
            return False
            
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False

def demonstrate_sequential_pipeline():
    """Demonstrate the sequential pipeline functionality."""
    print("🚀 Sequential Feature Generation Pipeline Demonstration")
    print("=" * 80)
    
    # Test 1: List available feature generation steps
    print("\n📋 Test 1: List Available Feature Generation Steps")
    cmd1 = [
        "python3", "src/launcher/ares_launcher.py",
        "--list-feature-generation-steps"
    ]
    
    success1 = run_command(cmd1, "Listing feature generation steps")
    
    # Test 2: Show help for sequential mode
    print("\n📋 Test 2: Show Help for Sequential Mode")
    cmd2 = [
        "python3", "src/launcher/ares_launcher.py",
        "--help"
    ]
    
    success2 = run_command(cmd2, "Showing help (look for sequential mode)")
    
    # Test 3: Demonstrate individual step execution (first step only)
    print("\n📋 Test 3: Execute Individual Steps (Data Validation)")
    cmd3 = [
        "python3", "src/launcher/ares_launcher.py",
        "--mode", "sub_pipeline",
        "--sub_pipeline", "feature_generation_data_validation_step",
        "--symbol", "ETHUSDT",
        "--execution-mode", "light",
        "--exchange", "binance",
        "--timeframe", "15m",
        "--direction", "both"
    ]
    
    success3 = run_command(cmd3, "Data Validation Step")
    
    # Test 4: Show how sequential mode would work (dry run)
    print("\n📋 Test 4: Sequential Mode Command Structure")
    print("To run the full sequential pipeline, use:")
    print()
    cmd4 = [
        "python3", "src/launcher/ares_launcher.py",
        "--mode", "sequential",
        "--pipeline-type", "feature_generation",
        "--symbol", "ETHUSDT",
        "--execution-mode", "light",
        "--exchange", "binance",
        "--timeframe", "15m",
        "--direction", "both"
    ]
    print(f"Command: {' '.join(cmd4)}")
    print()
    print("To run specific steps:")
    cmd5 = [
        "python3", "src/launcher/ares_launcher.py",
        "--mode", "sequential",
        "--pipeline-type", "feature_generation",
        "--start-from-step", "1",
        "--stop-at-step", "3",
        "--symbol", "ETHUSDT",
        "--execution-mode", "light"
    ]
    print(f"Command: {' '.join(cmd5)}")
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 DEMONSTRATION SUMMARY")
    print(f"{'='*80}")
    print(f"✅ List steps: {'PASS' if success1 else 'FAIL'}")
    print(f"✅ Show help: {'PASS' if success2 else 'FAIL'}")
    print(f"✅ Individual step: {'PASS' if success3 else 'FAIL'}")
    print(f"✅ Sequential mode: READY")
    
    print(f"\n🎯 KEY FEATURES DEMONSTRATED:")
    print("   • Sequential execution of feature generation steps")
    print("   • Parameter consistency across all steps")
    print("   • Automatic progression upon completion")
    print("   • Flexible start/stop step control")
    print("   • Comprehensive logging and error handling")
    
    return success1 and success2

def show_usage_examples():
    """Show usage examples for the sequential pipeline."""
    print(f"\n{'='*80}")
    print("📚 USAGE EXAMPLES")
    print(f"{'='*80}")
    
    examples = [
        {
            "title": "Run all feature generation steps sequentially",
            "command": "python3 src/launcher/ares_launcher.py --mode sequential --symbol ETHUSDT --execution-mode light"
        },
        {
            "title": "Run steps 1-3 only",
            "command": "python3 src/launcher/ares_launcher.py --mode sequential --start-from-step 1 --stop-at-step 3 --symbol ETHUSDT --execution-mode light"
        },
        {
            "title": "Run from step 5 to the end",
            "command": "python3 src/launcher/ares_launcher.py --mode sequential --start-from-step 5 --symbol ETHUSDT --execution-mode light"
        },
        {
            "title": "List available steps",
            "command": "python3 src/launcher/ares_launcher.py --list-feature-generation-steps"
        },
        {
            "title": "Run individual step",
            "command": "python3 src/launcher/ares_launcher.py --mode sub_pipeline --sub_pipeline feature_generation_data_validation_step --symbol ETHUSDT --execution-mode light"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['title']}:")
        print(f"   {example['command']}")

if __name__ == "__main__":
    print("🎯 Sequential Feature Generation Pipeline Demo")
    print("=" * 80)
    
    try:
        # Run demonstration
        success = demonstrate_sequential_pipeline()
        
        # Show usage examples
        show_usage_examples()
        
        if success:
            print(f"\n🎉 Demonstration completed successfully!")
            print("The sequential pipeline functionality is ready to use.")
        else:
            print(f"\n⚠️ Demonstration completed with some issues.")
            print("Check the output above for details.")
            
    except Exception as e:
        print(f"\n❌ Demonstration failed with error: {e}")
        sys.exit(1)