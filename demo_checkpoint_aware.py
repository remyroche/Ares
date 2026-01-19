#!/usr/bin/env python3
"""
Demo script showing checkpoint-aware Layer 3 execution.

This script demonstrates how the checkpoint system automatically:
1. Detects available checkpoints for a symbol
2. Resumes from the appropriate step
3. Saves progress at each sub-step
"""

import sys
import os
sys.path.append('/Users/remyroche/Documents/Ares')

from src.training.steps.labeling.checkpoint_aware_runner import CheckpointAwareRunner
from src.training.steps.labeling.layer3.checkpoint_aware_layer3 import CheckpointAwareLayer3

def demo_checkpoint_system():
    """Demonstrate the checkpoint-aware system."""
    
    print("🔧 Checkpoint-Aware Layer 3 Demo")
    print("=" * 50)
    
    # Test symbol
    symbol = "ETHUSDT"
    
    # Create checkpoint-aware runner
    print(f"\n📊 Creating checkpoint-aware runner for {symbol}...")
    runner = CheckpointAwareRunner('layer3', symbol)
    
    print(f"✅ Runner created successfully!")
    print(f"   Symbol: {runner.symbol}")
    print(f"   Layer: {runner.layer}")
    print(f"   Resume step: {runner.execution_plan.resume_step}")
    print(f"   Start from beginning: {runner.execution_plan.start_from_beginning}")
    print(f"   Available checkpoints: {len(runner.execution_plan.available_checkpoints)}")
    print(f"   Execution order: {runner.execution_plan.execution_order[:3]}...")
    
    # Get detailed checkpoint status
    print(f"\n📈 Checkpoint status for {symbol}:")
    status = runner.get_checkpoint_status()
    
    print(f"   Total checkpoints: {status['total_checkpoints']}")
    print(f"   Completion percentage: {status['completion_percentage']:.1f}%")
    
    if status['latest_checkpoint']:
        latest = status['latest_checkpoint']
        print(f"   Latest checkpoint: {latest['step']} at {latest['timestamp']}")
        print(f"   Latest step index: {latest['step_index']}")
        print(f"   Latest data keys: {latest['data_keys']}")
    else:
        print("   No checkpoints found - will start from beginning")
    
    # Show available checkpoints
    if status['available_checkpoints']:
        print(f"\n📋 Available checkpoints:")
        for cp in status['available_checkpoints']:
            print(f"   - {cp['step']} (index {cp['step_index']}) at {cp['timestamp']}")
    
    # Create checkpoint-aware Layer 3
    print(f"\n🚀 Creating checkpoint-aware Layer 3 for {symbol}...")
    layer3_cp = CheckpointAwareLayer3(symbol)
    
    print(f"✅ Checkpoint-aware Layer 3 created!")
    print(f"   Symbol: {layer3_cp.symbol}")
    print(f"   Resume step: {layer3_cp.runner.execution_plan.resume_step}")
    
    # Show execution plan
    print(f"\n📋 Execution plan for {symbol}:")
    plan = layer3_cp.runner.execution_plan
    print(f"   Layer: {plan.layer}")
    print(f"   Symbol: {plan.symbol}")
    print(f"   Resume step: {plan.resume_step}")
    print(f"   Start from beginning: {plan.start_from_beginning}")
    print(f"   Total steps to execute: {len(plan.execution_order)}")
    
    print(f"\n🔄 Steps to execute:")
    for i, step in enumerate(plan.execution_order, 1):
        print(f"   {i}. {step}")
    
    print(f"\n✨ Demo completed successfully!")
    print(f"\n📝 Usage:")
    print(f"   When running Layer 3 with symbol='{symbol}', the system will:")
    print(f"   1. Automatically detect checkpoints in versioned_artifacts/layer3_checkpoints/{symbol}/")
    print(f"   2. Resume from '{plan.resume_step}' (or start from beginning if no checkpoints)")
    print(f"   3. Save progress at each of the {len(plan.execution_order)} remaining steps")
    print(f"   4. Provide detailed execution metadata in the results")
    
    print(f"\n🎯 Key Benefits:")
    print(f"   ✅ Automatic resume from failures")
    print(f"   ✅ Symbol-specific checkpoint isolation")
    print(f"   ✅ Detailed progress tracking")
    print(f"   ✅ Robust error recovery")
    print(f"   ✅ Production-ready reliability")

if __name__ == "__main__":
    demo_checkpoint_system()
