#!/usr/bin/env python3
"""
Test script to verify unified utilities can be imported without errors.
"""

import sys
import os

# Add the workspace to the Python path
sys.path.insert(0, '/workspace')

def test_imports():
    """Test importing unified utilities."""
    try:
        print("Testing unified utilities imports...")
        
        # Test individual imports
        print("1. Testing UnifiedEconomicSignificanceEvaluator...")
        from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils.unified_economic_evaluator import (
            UnifiedEconomicSignificanceEvaluator, EconomicEvaluationConfig, EconomicSignificanceResult
        )
        print("   ✅ UnifiedEconomicSignificanceEvaluator imported successfully")
        
        print("2. Testing UnifiedTradingViabilityEvaluator...")
        from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils.unified_trading_viability_evaluator import (
            UnifiedTradingViabilityEvaluator, TradingViabilityConfig, TradingViabilityResult
        )
        print("   ✅ UnifiedTradingViabilityEvaluator imported successfully")
        
        print("3. Testing UnifiedMultiObjectiveOptimizer...")
        from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils.unified_multi_objective_optimizer import (
            UnifiedMultiObjectiveOptimizer, OptimizationConfig, OptimizationResult
        )
        print("   ✅ UnifiedMultiObjectiveOptimizer imported successfully")
        
        print("4. Testing UnifiedHardwareOptimizer...")
        from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils.unified_hardware_optimizer import (
            UnifiedHardwareOptimizer, HardwareConfig, PerformanceMetrics
        )
        print("   ✅ UnifiedHardwareOptimizer imported successfully")
        
        print("5. Testing UnifiedRegimeAnalyzer...")
        from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils.unified_regime_analyzer import (
            UnifiedRegimeAnalyzer, RegimeAnalysisConfig, RegimeAnalysisResult
        )
        print("   ✅ UnifiedRegimeAnalyzer imported successfully")
        
        print("6. Testing UnifiedConfigManager...")
        from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils.unified_config_manager import (
            UnifiedConfigManager, UnifiedRegimeConfig
        )
        print("   ✅ UnifiedConfigManager imported successfully")
        
        print("7. Testing UnifiedValidationSystem...")
        from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils.unified_validation_system import (
            UnifiedValidationSystem, ValidationConfig, ValidationResult
        )
        print("   ✅ UnifiedValidationSystem imported successfully")
        
        # Test convenience functions
        print("8. Testing convenience functions...")
        from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils.unified_economic_evaluator import (
            create_unified_economic_evaluator, quick_economic_evaluation
        )
        from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils.unified_trading_viability_evaluator import (
            create_unified_trading_viability_evaluator, quick_trading_viability_evaluation
        )
        from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils.unified_multi_objective_optimizer import (
            create_unified_multi_objective_optimizer, quick_multi_objective_optimization
        )
        from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils.unified_hardware_optimizer import (
            create_unified_hardware_optimizer, quick_hardware_optimization
        )
        from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils.unified_regime_analyzer import (
            create_unified_regime_analyzer, quick_regime_analysis
        )
        from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils.unified_config_manager import (
            create_unified_config_manager, load_config_from_file, create_environment_config
        )
        from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils.unified_validation_system import (
            create_unified_validation_system, quick_validation
        )
        print("   ✅ All convenience functions imported successfully")
        
        # Test main module import
        print("9. Testing main shared_utils module import...")
        from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils import (
            UnifiedEconomicSignificanceEvaluator,
            UnifiedTradingViabilityEvaluator,
            UnifiedMultiObjectiveOptimizer,
            UnifiedHardwareOptimizer,
            UnifiedRegimeAnalyzer,
            UnifiedConfigManager,
            UnifiedValidationSystem,
            create_unified_economic_evaluator,
            quick_economic_evaluation,
            create_unified_trading_viability_evaluator,
            quick_trading_viability_evaluation,
            create_unified_multi_objective_optimizer,
            quick_multi_objective_optimization,
            create_unified_hardware_optimizer,
            quick_hardware_optimization,
            create_unified_regime_analyzer,
            quick_regime_analysis,
            create_unified_config_manager,
            load_config_from_file,
            create_environment_config,
            create_unified_validation_system,
            quick_validation
        )
        print("   ✅ Main shared_utils module imported successfully")
        
        print("\n🎉 All unified utilities imported successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)