#!/usr/bin/env python3
"""
Enable Gate Feature Protection

This script enables gate feature protection across the entire pre-training pipeline.
Run this script to automatically patch all relevant components.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def enable_gate_protection():
    """Enable gate feature protection across the pipeline."""
    print("🛡️ Enabling Gate Feature Protection...")
    
    try:
        # Import and apply patches
        from .gate_feature_integration import apply_all_gate_protection_patches
        apply_all_gate_protection_patches()
        
        print("✅ Gate feature protection enabled successfully!")
        print("📋 Patches applied to:")
        print("   - final_feature_selection_pipeline.py")
        print("   - final_feature_selection_step.py") 
        print("   - analyst_pre_ml_orchestration.py")
        print("   - tactician_pre_ml_orchestration.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to enable gate feature protection: {e}")
        return False

def create_gate_aware_config():
    """Create a gate-aware configuration."""
    return {
        'gate_protection': {
            'enabled': True,
            'max_gate_features_per_base': 3,
            'min_gate_ic_improvement': 0.005,
            'min_gate_stability': 0.4,
            'gate_correlation_threshold': 0.95,
            'gate_importance_weight': 1.5,
            'gate_regime_bonus': 0.1,
            'validate_gate_contribution': True,
            'min_gate_contribution': 0.01,
            'enable_gate_interaction_validation': True
        },
        'feature_selection': {
            'correlation_thresholds': [0.92, 0.96, 0.98],
            'enable_rfe': True,
            'enable_variance_filtering': True,
            'enable_gate_protection': True
        }
    }

def main():
    """Main function to enable gate protection."""
    print("🚀 Gate Feature Protection Setup")
    print("=" * 50)
    
    # Enable protection
    success = enable_gate_protection()
    
    if success:
        print("\n📝 Usage Examples:")
        print("1. Use in Analyst orchestration:")
        print("   config = AnalystPreMLConfig(enable_gate_protection=True)")
        print("   orchestrator = AnalystPreMLOrchestrator(config)")
        
        print("\n2. Use in Tactician orchestration:")
        print("   config = TacticianPreMLConfig(enable_gate_protection=True)")
        print("   orchestrator = TacticianPreMLOrchestrator(config)")
        
        print("\n3. Use gate-aware pipeline manager:")
        print("   from gate_feature_integration import GateFeaturePipelineManager")
        print("   manager = GateFeaturePipelineManager()")
        
        print("\n✅ Gate feature protection is now active!")
        print("🛡️ Gate features will be protected during final feature selection")
        
    else:
        print("\n❌ Failed to enable gate feature protection")
        print("Please check the error messages above and try again")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())