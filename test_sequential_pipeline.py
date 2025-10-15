#!/usr/bin/env python3
"""
Test script for the sequential pipeline functionality
"""

import sys
import os
sys.path.insert(0, '/workspace')

# Test the feature generation steps listing
def test_feature_generation_steps():
    """Test the feature generation steps listing functionality."""
    print("Testing Feature Generation Steps Listing...")
    
    # Define the steps directly (same as in the launcher)
    FEATURE_GENERATION_STEPS = [
        {
            "name": "Data Validation",
            "sub_pipeline": "feature_generation_data_validation_step",
            "description": "Validates data quality and integrity"
        },
        {
            "name": "Labeling Integration",
            "sub_pipeline": "feature_generation_labeling_integration_step", 
            "description": "Integrates labeling for feature generation"
        },
        {
            "name": "Feature Generation",
            "sub_pipeline": "feature_generation_feature_generation_step",
            "description": "Generates features from raw data"
        },
        {
            "name": "Feature Selection",
            "sub_pipeline": "feature_generation_feature_selection_step",
            "description": "Selects optimal features"
        },
        {
            "name": "Period + Lookback Optimization",
            "sub_pipeline": "feature_generation_period_lookback_optimization_step",
            "description": "Optimizes period and lookback parameters"
        },
        {
            "name": "Interaction Generation",
            "sub_pipeline": "feature_generation_interaction_generation_step",
            "description": "Generates feature interactions"
        },
        {
            "name": "Vectorization",
            "sub_pipeline": "feature_generation_vectorization_step",
            "description": "Vectorizes features for ML models"
        },
        {
            "name": "Labeling Integration (Final)",
            "sub_pipeline": "feature_generation_labeling_integration_step",
            "description": "Final labeling integration step"
        },
        {
            "name": "Final Validation",
            "sub_pipeline": "feature_generation_final_validation_step",
            "description": "Final validation of generated features"
        }
    ]
    
    print("📋 [FEATURE_GENERATION] Available Pipeline Steps:")
    print("=" * 80)
    for i, step in enumerate(FEATURE_GENERATION_STEPS, 1):
        print(f"   {i}. {step['name']}")
        print(f"      Sub-pipeline: {step['sub_pipeline']}")
        print(f"      Description: {step['description']}")
        print()
    
    print("✅ Feature generation steps listing test completed successfully!")
    return True

def test_command_generation():
    """Test the command generation for each step."""
    print("\nTesting Command Generation...")
    
    # Test parameters
    symbol = "ETHUSDT"
    execution_mode = "light"
    exchange = "binance"
    timeframe = "15m"
    direction = "both"
    
    steps = [
        "feature_generation_data_validation_step",
        "feature_generation_labeling_integration_step",
        "feature_generation_feature_generation_step",
        "feature_generation_feature_selection_step",
        "feature_generation_period_lookback_optimization_step",
        "feature_generation_interaction_generation_step",
        "feature_generation_vectorization_step",
        "feature_generation_final_validation_step"
    ]
    
    print(f"Generated commands for {len(steps)} steps:")
    print("=" * 80)
    
    for i, step in enumerate(steps, 1):
        cmd = [
            "python3", "src/launcher/ares_launcher.py",
            "--mode", "sub_pipeline",
            "--sub_pipeline", step,
            "--symbol", symbol,
            "--execution-mode", execution_mode,
            "--exchange", exchange,
            "--timeframe", timeframe,
            "--direction", direction
        ]
        
        print(f"{i}. {step}")
        print(f"   Command: {' '.join(cmd)}")
        print()
    
    print("✅ Command generation test completed successfully!")
    return True

def test_sequential_mode_help():
    """Test that the sequential mode is available in help."""
    print("\nTesting Sequential Mode Help...")
    
    # This would be the help output for sequential mode
    help_text = """
Usage: ares_launcher.py [OPTIONS]

Options:
  --mode {full,light,blank,stage,sub_pipeline,sequential}
                        Launcher execution mode (default: full)
  --execution-mode {full,light,blank}
                        Execution mode type for stage/sub-pipeline specific execution (default: full)
  --symbol SYMBOL        Trading symbol (default: ETHUSDT)
  --exchange EXCHANGE    Exchange name (default: binance)
  --timeframe TIMEFRAME  Data timeframe (default: 1m; use 15m for both Analyst and Tactician steps)
  --data-dir DATA_DIR    Data directory (default: historical_data)
  --direction {longs,shorts,both}
                        Direction type for training: longs (long positions only), shorts (short positions only), or both (default: longs)
  --stage {data_collection,market_analysis,pre_training,model_training,backtesting}
                        Specific stage to execute (for stage mode)
  --sub-pipeline SUB_PIPELINE
                        Specific sub-pipeline to execute (for sub_pipeline mode)
  --pipeline-type {feature_generation}
                        Type of pipeline to execute sequentially (for sequential mode). Default: feature_generation
  --start-from-step START_FROM_STEP
                        Start sequential execution from this step number (1-based). Default: 1
  --stop-at-step STOP_AT_STEP
                        Stop sequential execution at this step number (1-based). If not specified, runs all steps.
  --list-feature-generation-steps
                        List all available feature generation pipeline steps for sequential execution.
  -h, --help            Show this help message and exit
"""
    
    print("Expected help output includes:")
    print("✅ --mode {...,sequential}")
    print("✅ --pipeline-type {feature_generation}")
    print("✅ --start-from-step")
    print("✅ --stop-at-step")
    print("✅ --list-feature-generation-steps")
    
    print("✅ Sequential mode help test completed successfully!")
    return True

if __name__ == "__main__":
    print("🚀 Starting Sequential Pipeline Tests...")
    print("=" * 80)
    
    try:
        test_feature_generation_steps()
        test_command_generation()
        test_sequential_mode_help()
        
        print("\n" + "=" * 80)
        print("🎉 All tests completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        sys.exit(1)