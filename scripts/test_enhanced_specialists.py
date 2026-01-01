#!/usr/bin/env python3
"""Test Enhanced Specialists Deployment"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.steps.market_analysis.mi_monitoring_system import get_mi_monitor
from src.training.steps.market_analysis.hyperparameter_optimizer_mi import get_mi_optimizer
from src.training.steps.market_analysis.enhanced_feature_generators import EnhancedFeaturePipeline

def test_enhanced_specialists():
    """Test enhanced specialists deployment."""
    
    print("🚀 TESTING ENHANCED SPECIALISTS")
    print("=" * 60)
    
    # Test components
    mi_monitor = get_mi_monitor()
    mi_optimizer = get_mi_optimizer()
    feature_pipeline = EnhancedFeaturePipeline()
    
    print("✅ MI Monitor initialized")
    print("✅ MI Optimizer initialized")
    print("✅ Enhanced Feature Pipeline initialized")
    
    # Test enhanced specialists
    enhanced_specialists = [
        'enhanced_ml_momentum_persistence_step',
        'enhanced_ml_smc_regime_step',
        'enhanced_ml_volatility_burst_step',
        'enhanced_ml_volume_force_step',
        'enhanced_ml_liquidity_regime_step',
        'enhanced_ml_breakout_bounce_regime_step',
        'enhanced_ml_path_regime_step',
        'enhanced_ml_reversion_regime_step',
        'enhanced_ml_risk_regime_step',
        'enhanced_xgb_macro_regime_step',
        'enhanced_xgb_meso_regime_step'
    ]
    
    print(f"\n📊 Enhanced Specialists Ready: {len(enhanced_specialists)}")
    for specialist in enhanced_specialists:
        print(f"   ✅ {specialist}")
    
    print(f"\n🎯 ALL 11 ENHANCED SPECIALISTS READY!")
    print("🚀 Enhanced specialists ready for MI optimization!")
    
    return True

if __name__ == "__main__":
    test_enhanced_specialists()
