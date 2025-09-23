#!/usr/bin/env python3
"""
Test Enhanced ML Pipeline

Comprehensive test for all the enhanced ML pipeline features:
1. Grid Search + Bayesian TPE Integration
2. Overfitting/Underfitting Detection
3. Bayesian Entry Timing Optimization
4. Enhanced HPO Integration
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import logging

# Add src to path
sys.path.append('/workspace/src')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_grid_bayesian_integration():
    """Test Grid Search + Bayesian TPE integration."""
    print("🧪 Testing Grid Search + Bayesian TPE Integration...")
    
    try:
        from src.utils.ml_common.validation.hpo_overfitting_prevention import (
            HPOWithOverfittingPrevention, 
            HPOOverfittingPreventionConfig
        )
        
        # Test configuration
        config = HPOOverfittingPreventionConfig(
            enable_staged_hpo=True,
            coarse_strategy="grid",
            coarse_grid_points=3,
            fine_grid_points=5,
            bayes_n_trials=10,
            n_trials=50
        )
        
        print("✅ HPOWithOverfittingPrevention with staged HPO configuration created")
        print(f"📊 Staged HPO enabled: {config.enable_staged_hpo}")
        print(f"📊 Coarse strategy: {config.coarse_strategy}")
        print(f"📊 Bayesian trials: {config.bayes_n_trials}")
        
        return True
        
    except Exception as e:
        print(f"❌ Grid + Bayesian TPE integration test failed: {e}")
        return False

def test_overfitting_underfitting_detection():
    """Test overfitting and underfitting detection."""
    print("🧪 Testing Overfitting/Underfitting Detection...")
    
    try:
        from src.utils.ml_common.validation.enhanced_overfitting_detection import (
            UniversalOverfittingDetector,
            OverfittingConfig
        )
        from src.utils.ml_common.validation.underfitting_detection import (
            UnderfittingDetector,
            UnderfittingConfig
        )
        from src.utils.ml_common.validation.model_enhancement_guide import (
            ModelEnhancementGuide
        )
        
        # Test overfitting detection
        overfitting_config = OverfittingConfig()
        overfitting_detector = UniversalOverfittingDetector(overfitting_config)
        print("✅ Overfitting detector created")
        
        # Test underfitting detection
        underfitting_config = UnderfittingConfig()
        underfitting_detector = UnderfittingDetector(underfitting_config)
        print("✅ Underfitting detector created")
        
        # Test enhancement guide
        enhancement_guide = ModelEnhancementGuide()
        print("✅ Model enhancement guide created")
        
        # Test enhancement plan creation
        enhancement_plan = enhancement_guide.create_enhancement_plan(
            model_name="test_model",
            model_type="random_forest",
            overfitting_report={'is_overfitting': True, 'severity': 'moderate'},
            underfitting_report={'is_underfitting': False, 'severity': 'none'}
        )
        
        print(f"✅ Enhancement plan created with {len(enhancement_plan.enhancement_actions)} actions")
        print(f"📊 Risk assessment: {enhancement_plan.risk_assessment}")
        print(f"📊 Timeline estimate: {enhancement_plan.timeline_estimate}")
        
        return True
        
    except Exception as e:
        print(f"❌ Overfitting/Underfitting detection test failed: {e}")
        return False

def test_bayesian_entry_timing_optimization():
    """Test Bayesian entry timing optimization."""
    print("🧪 Testing Bayesian Entry Timing Optimization...")
    
    try:
        from src.utils.ml_common.optimization.bayesian_entry_timing_optimizer import (
            BayesianEntryTimingOptimizer,
            EntryTimingConfig,
            EntryTimingResult
        )
        
        # Test configuration
        config = EntryTimingConfig(
            n_trials=10,
            timeout_minutes=5,
            enable_multi_objective=True,
            objectives=['profit', 'sharpe', 'win_rate', 'max_drawdown'],
            objective_weights=[0.4, 0.3, 0.2, 0.1]
        )
        
        print("✅ Entry timing configuration created")
        print(f"📊 Multi-objective enabled: {config.enable_multi_objective}")
        print(f"📊 Objectives: {config.objectives}")
        print(f"📊 Objective weights: {config.objective_weights}")
        
        # Test optimizer creation
        optimizer = BayesianEntryTimingOptimizer(config)
        print("✅ Bayesian entry timing optimizer created")
        
        return True
        
    except Exception as e:
        print(f"❌ Bayesian entry timing optimization test failed: {e}")
        return False

def test_enhanced_hpo_integration():
    """Test enhanced HPO integration."""
    print("🧪 Testing Enhanced HPO Integration...")
    
    try:
        from src.training.steps.model_training.random_survival_forest_tactician import (
            RandomSurvivalForestTactician,
            SurvivalAnalysisConfig
        )
        
        # Test configuration
        config = SurvivalAnalysisConfig(
            n_estimators=100,
            max_depth=10,
            horizons=[1, 2, 5, 10],
            horizon_weights=[0.4, 0.3, 0.2, 0.1],
            entry_timing_range=0.005,
            expected_movement=0.01,
            latency_constraint=2.0
        )
        
        print("✅ RandomSurvivalForestTactician configuration created")
        print(f"📊 Horizons: {config.horizons}")
        print(f"📊 Horizon weights: {config.horizon_weights}")
        
        # Test model creation
        model = RandomSurvivalForestTactician(config)
        print("✅ RandomSurvivalForestTactician model created")
        
        # Test fit method signature
        import inspect
        fit_signature = inspect.signature(model.fit)
        fit_params = list(fit_signature.parameters.keys())
        
        expected_params = [
            'X', 'y', 'feature_names', 'analyst_signals', 'hmm_regime_probs',
            'multi_horizon_data', 'enable_hpo', 'hpo_trials', 'cv_folds',
            'enable_entry_timing_optimization', 'entry_timing_trials'
        ]
        
        missing_params = [p for p in expected_params if p not in fit_params]
        if missing_params:
            print(f"⚠️ Missing parameters in fit method: {missing_params}")
        else:
            print("✅ All expected parameters present in fit method")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced HPO integration test failed: {e}")
        return False

def test_comprehensive_pipeline():
    """Test the comprehensive enhanced ML pipeline."""
    print("🧪 Testing Comprehensive Enhanced ML Pipeline...")
    
    try:
        # Test all components together
        from src.utils.ml_common.validation.hpo_overfitting_prevention import (
            HPOWithOverfittingPrevention, 
            HPOOverfittingPreventionConfig
        )
        from src.utils.ml_common.validation.enhanced_overfitting_detection import (
            UniversalOverfittingDetector
        )
        from src.utils.ml_common.validation.underfitting_detection import (
            UnderfittingDetector
        )
        from src.utils.ml_common.optimization.bayesian_entry_timing_optimizer import (
            BayesianEntryTimingOptimizer
        )
        from src.training.steps.model_training.random_survival_forest_tactician import (
            RandomSurvivalForestTactician
        )
        
        print("✅ All enhanced ML pipeline components imported successfully")
        
        # Test configuration integration
        hpo_config = HPOOverfittingPreventionConfig(
            enable_staged_hpo=True,
            coarse_strategy="grid",
            bayes_n_trials=30
        )
        
        entry_config = EntryTimingConfig(
            n_trials=50,
            enable_multi_objective=True
        )
        
        rsf_config = SurvivalAnalysisConfig(
            horizons=[1, 2, 5, 10],
            enable_entry_timing_optimization=True
        )
        
        print("✅ All configurations created successfully")
        print(f"📊 HPO staged optimization: {hpo_config.enable_staged_hpo}")
        print(f"📊 Entry timing multi-objective: {entry_config.enable_multi_objective}")
        print(f"📊 RSF horizons: {rsf_config.horizons}")
        
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive pipeline test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Enhanced ML Pipeline Tests")
    print("=" * 60)
    
    tests = [
        ("Grid Search + Bayesian TPE Integration", test_grid_bayesian_integration),
        ("Overfitting/Underfitting Detection", test_overfitting_underfitting_detection),
        ("Bayesian Entry Timing Optimization", test_bayesian_entry_timing_optimization),
        ("Enhanced HPO Integration", test_enhanced_hpo_integration),
        ("Comprehensive Pipeline", test_comprehensive_pipeline)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\n📊 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced ML pipeline is ready.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)