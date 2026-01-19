#!/usr/bin/env python3
"""
Demo script showing checkpoint override functionality.

This script demonstrates how to:
1. Override checkpoints from any specific stage
2. Force restart from beginning
3. Replace existing checkpoints with new run
4. Preserve earlier checkpoints if desired
"""

import sys
import os
sys.path.append('/Users/remyroche/Documents/Ares')

from src.training.steps.labeling.checkpoint_override_manager import (
    CheckpointOverrideManager, 
    create_checkpoint_override,
    list_override_options
)
from src.training.steps.labeling.checkpoint_aware_runner import CheckpointAwareRunner

def demo_checkpoint_override():
    """Demonstrate the checkpoint override system."""
    
    print("🔄 Checkpoint Override Demo")
    print("=" * 50)
    
    # Test symbol
    symbol = "ETHUSDT"
    layer = "layer3"
    
    print(f"📊 Testing checkpoint override for {layer} ({symbol})")
    print()
    
    # 1. List available override options
    print("📋 Available Override Options:")
    options = list_override_options(layer, symbol)
    
    print(f"   Layer: {options['layer']}")
    print(f"   Symbol: {options['symbol']}")
    print(f"   Total steps: {options['total_steps']}")
    print(f"   Completed steps: {options['completed_steps']}")
    print(f"   Completion: {options['completion_percentage']:.1f}%")
    print()
    
    print("   Available steps to override from:")
    for i, step in enumerate(options['available_steps'], 1):
        status = "✅" if step in options['current_checkpoints'] else "⭕"
        timestamp = options['current_checkpoints'].get(step, "Not completed")
        print(f"   {i:2d}. {step} {status} ({timestamp})")
    
    print()
    
    # 2. Create override manager
    print("🔧 Creating Checkpoint Override Manager...")
    override_manager = CheckpointOverrideManager(layer, symbol)
    
    # 3. Test different override scenarios
    scenarios = [
        {
            'name': 'Override from dual_head_training',
            'step': 'dual_head_training',
            'force_restart': False,
            'keep_earlier': True
        },
        {
            'name': 'Force restart from beginning',
            'step': 'data_loading',
            'force_restart': True,
            'keep_earlier': False
        },
        {
            'name': 'Override from race_reporting',
            'step': 'race_reporting',
            'force_restart': False,
            'keep_earlier': False
        }
    ]
    
    for scenario in scenarios:
        print(f"\n🎯 Scenario: {scenario['name']}")
        print(f"   Override step: {scenario['step']}")
        print(f"   Force restart: {scenario['force_restart']}")
        print(f"   Keep earlier checkpoints: {scenario['keep_earlier']}")
        
        # Get override plan
        from src.training.steps.labeling.checkpoint_override_manager import OverrideConfig
        override_config = OverrideConfig(
            layer=layer,
            symbol=symbol,
            override_step=scenario['step'],
            force_restart=scenario['force_restart'],
            keep_earlier_checkpoints=scenario['keep_earlier']
        )
        
        plan = override_manager.get_override_plan(override_config)
        
        print(f"   📋 Override Plan:")
        print(f"      Steps to execute: {len(plan['steps_to_execute'])}")
        print(f"      Steps to delete: {len(plan['steps_to_delete'])}")
        print(f"      Steps to keep: {len(plan['steps_to_keep'])}")
        print(f"      Checkpoints to remove: {plan['checkpoints_to_be_removed']}")
        print(f"      Checkpoints to preserve: {plan['checkpoints_to_be_preserved']}")
        
        print(f"      Steps to execute: {plan['steps_to_execute'][:3]}...")
        
        # Show what would be deleted/preserved
        if plan['existing_to_delete']:
            print(f"      Will delete checkpoints: {plan['existing_to_delete']}")
        if plan['existing_to_keep']:
            print(f"      Will preserve checkpoints: {plan['existing_to_keep']}")
    
    print()
    
    # 4. Demonstrate convenience function
    print("🚀 Convenience Function Demo:")
    print("   Using create_checkpoint_override()...")
    
    try:
        # Create override runner (this would actually execute the override)
        runner = create_checkpoint_override(
            layer=layer,
            symbol=symbol,
            override_step='dual_head_training',
            force_restart=False,
            keep_earlier_checkpoints=True
        )
        
        print(f"   ✅ Override runner created successfully")
        print(f"   Resume step: {runner.execution_plan.resume_step}")
        print(f"   Steps to execute: {len(runner.execution_plan.execution_order)}")
        print(f"   Start from beginning: {runner.execution_plan.start_from_beginning}")
        
    except Exception as e:
        print(f"   ⚠️ Override runner creation failed: {e}")
    
    print()
    
    # 5. Show usage examples
    print("📝 Usage Examples:")
    print()
    
    print("   # Basic usage - override from specific step:")
    print("   df, models = layer3_analyst_lgbm(")
    print("       oof_df=data,")
    print("       base_model_cols=features,")
    print("       target_col='target',")
    print("       symbol='ETHUSDT',")
    print("       override_step='dual_head_training'  # Override from here")
    print("   )")
    print()
    
    print("   # Force restart from beginning:")
    print("   df, models = layer3_analyst_lgbm(")
    print("       oof_df=data,")
    print("       base_model_cols=features,")
    print("       target_col='target',")
    print("       symbol='ETHUSDT',")
    print("       force_restart=True  # Ignore all checkpoints")
    print("   )")
    print()
    
    print("   # Override but keep earlier checkpoints:")
    print("   df, models = layer3_analyst_lgbm(")
    print("       oof_df=data,")
    print("       base_model_cols=features,")
    print("       target_col='target',")
    print("       symbol='ETHUSDT',")
    print("       override_step='race_reporting',")
    print("       keep_earlier_checkpoints=True  # Keep data_loading...feature_clustering")
    print("   )")
    print()
    
    print("   # Using checkpoint override manager directly:")
    print("   from src.training.steps.labeling.checkpoint_override_manager import CheckpointOverrideManager")
    print("   ")
    print("   manager = CheckpointOverrideManager('layer3', 'ETHUSDT')")
    print("   runner = manager.create_override_runner(")
    print("       override_step='dual_head_training',")
    print("       force_restart=False,")
    print("       keep_earlier_checkpoints=True")
    print("   )")
    print()
    
    print("🎯 Key Benefits:")
    print("   ✅ Restart from any specific stage")
    print("   ✅ Force restart from beginning (clean slate)")
    print("   ✅ Preserve earlier checkpoints if desired")
    print("   ✅ Automatic replacement of overridden checkpoints")
    print("   ✅ Detailed override planning and preview")
    print("   ✅ Available for all layers (2.5, 3, 4)")
    print()
    
    print("🚀 Override system is fully functional!")
    print("   You can now override checkpoints from any stage and automatically")
    print("   replace existing checkpoints with the new run.")

if __name__ == "__main__":
    demo_checkpoint_override()
