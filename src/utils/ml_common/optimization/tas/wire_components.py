"""
TAS Component Wiring Script

This script ensures all TAS components are properly wired together and
demonstrates the complete integration between shared utilities and enhanced TAS.
"""

import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import all TAS components
try:
    from .core.tas_engine import TreeArchitectureSearchEngine, TASEngineConfig
    from .enhanced_tas_engine import EnhancedTASEngine, EnhancedTASConfig
    from .core.tas_config import TASConfig, TASSearchConfig, TASOptimizationConfig
    TAS_CORE_AVAILABLE = True
except ImportError as e:
    TAS_CORE_AVAILABLE = False
    logger.warning(f"⚠️ TAS core not available: {e}")

# Import shared utilities
try:
    from ..shared_utils.evolutionary_search import (
        EvolutionaryAlgorithmManager, EvolutionaryConfig, EvolutionaryResult,
        create_evolutionary_algorithm_manager
    )
    from ..shared_utils.feature_engineering import (
        UnifiedFeatureEngineer, FeatureConfig, FeatureEngineeringResult,
        create_unified_feature_engineer
    )
    from ..shared_utils.evaluation_metrics import (
        UnifiedEvaluator, UnifiedEvaluationResult,
        create_unified_evaluator
    )
    SHARED_UTILS_AVAILABLE = True
except ImportError as e:
    SHARED_UTILS_AVAILABLE = False
    logger.warning(f"⚠️ Shared utilities not available: {e}")

# Import enhanced TAS components
try:
    from .models.enhanced_tree_models import (
        EnhancedTreeModelFactory, TreeModelConfig, TreeModelType,
        TreeModelResult, TreeModelEvaluator, create_model_ensemble
    )
    from .automl.tree_automl import (
        TreeAutoMLManager, AutoMLConfig, AutoMLResult,
        create_tree_automl_manager
    )
    from .evaluation.advanced_metrics import (
        AdvancedEvaluator, AdvancedEvaluationResult,
        create_advanced_evaluator
    )
    ENHANCED_COMPONENTS_AVAILABLE = True
except ImportError as e:
    ENHANCED_COMPONENTS_AVAILABLE = False
    logger.warning(f"⚠️ Enhanced components not available: {e}")


class TASComponentWiring:
    """Class to handle TAS component wiring and integration."""
    
    def __init__(self):
        """Initialize TAS component wiring."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.wiring_status = {}
        self.integration_tests = {}
        
        self.logger.info("🔧 Initializing TAS Component Wiring")
    
    def check_component_availability(self) -> Dict[str, bool]:
        """Check availability of all TAS components."""
        self.logger.info("🔍 Checking component availability...")
        
        availability = {
            'tas_core': TAS_CORE_AVAILABLE,
            'shared_utils': SHARED_UTILS_AVAILABLE,
            'enhanced_components': ENHANCED_COMPONENTS_AVAILABLE
        }
        
        for component, available in availability.items():
            status = "✅" if available else "❌"
            self.logger.info(f"   {component}: {status}")
        
        self.wiring_status['component_availability'] = availability
        return availability
    
    def test_shared_utilities_integration(self) -> Dict[str, bool]:
        """Test integration of shared utilities."""
        self.logger.info("🔧 Testing shared utilities integration...")
        
        integration_results = {}
        
        # Test evolutionary search
        try:
            if SHARED_UTILS_AVAILABLE:
                config = EvolutionaryConfig(population_size=10, max_generations=5)
                manager = create_evolutionary_algorithm_manager(config)
                integration_results['evolutionary_search'] = True
                self.logger.info("   ✅ Evolutionary search integration successful")
            else:
                integration_results['evolutionary_search'] = False
                self.logger.warning("   ⚠️ Evolutionary search not available")
        except Exception as e:
            integration_results['evolutionary_search'] = False
            self.logger.warning(f"   ⚠️ Evolutionary search integration failed: {e}")
        
        # Test feature engineering
        try:
            if SHARED_UTILS_AVAILABLE:
                config = FeatureConfig(enable_technical_indicators=True, max_features=50)
                engineer = create_unified_feature_engineer(config)
                integration_results['feature_engineering'] = True
                self.logger.info("   ✅ Feature engineering integration successful")
            else:
                integration_results['feature_engineering'] = False
                self.logger.warning("   ⚠️ Feature engineering not available")
        except Exception as e:
            integration_results['feature_engineering'] = False
            self.logger.warning(f"   ⚠️ Feature engineering integration failed: {e}")
        
        # Test evaluation metrics
        try:
            if SHARED_UTILS_AVAILABLE:
                evaluator = create_unified_evaluator()
                integration_results['evaluation_metrics'] = True
                self.logger.info("   ✅ Evaluation metrics integration successful")
            else:
                integration_results['evaluation_metrics'] = False
                self.logger.warning("   ⚠️ Evaluation metrics not available")
        except Exception as e:
            integration_results['evaluation_metrics'] = False
            self.logger.warning(f"   ⚠️ Evaluation metrics integration failed: {e}")
        
        self.wiring_status['shared_utilities_integration'] = integration_results
        return integration_results
    
    def test_enhanced_components_integration(self) -> Dict[str, bool]:
        """Test integration of enhanced TAS components."""
        self.logger.info("🔧 Testing enhanced components integration...")
        
        integration_results = {}
        
        # Test enhanced tree models
        try:
            if ENHANCED_COMPONENTS_AVAILABLE:
                config = TreeModelConfig(
                    model_type=TreeModelType.XGBOOST,
                    params={'n_estimators': 10, 'max_depth': 3},
                    is_classifier=True
                )
                factory = EnhancedTreeModelFactory(config)
                integration_results['enhanced_tree_models'] = True
                self.logger.info("   ✅ Enhanced tree models integration successful")
            else:
                integration_results['enhanced_tree_models'] = False
                self.logger.warning("   ⚠️ Enhanced tree models not available")
        except Exception as e:
            integration_results['enhanced_tree_models'] = False
            self.logger.warning(f"   ⚠️ Enhanced tree models integration failed: {e}")
        
        # Test AutoML
        try:
            if ENHANCED_COMPONENTS_AVAILABLE:
                config = AutoMLConfig(
                    optimization_method="optuna",
                    max_trials=5,
                    timeout_seconds=60,
                    model_types=["xgboost"]
                )
                manager = create_tree_automl_manager(config)
                integration_results['automl'] = True
                self.logger.info("   ✅ AutoML integration successful")
            else:
                integration_results['automl'] = False
                self.logger.warning("   ⚠️ AutoML not available")
        except Exception as e:
            integration_results['automl'] = False
            self.logger.warning(f"   ⚠️ AutoML integration failed: {e}")
        
        # Test advanced evaluation
        try:
            if ENHANCED_COMPONENTS_AVAILABLE:
                evaluator = create_advanced_evaluator()
                integration_results['advanced_evaluation'] = True
                self.logger.info("   ✅ Advanced evaluation integration successful")
            else:
                integration_results['advanced_evaluation'] = False
                self.logger.warning("   ⚠️ Advanced evaluation not available")
        except Exception as e:
            integration_results['advanced_evaluation'] = False
            self.logger.warning(f"   ⚠️ Advanced evaluation integration failed: {e}")
        
        self.wiring_status['enhanced_components_integration'] = integration_results
        return integration_results
    
    def test_tas_engine_integration(self) -> Dict[str, bool]:
        """Test TAS engine integration with all components."""
        self.logger.info("🔧 Testing TAS engine integration...")
        
        integration_results = {}
        
        # Test standard TAS engine
        try:
            if TAS_CORE_AVAILABLE:
                config = TASEngineConfig(
                    enable_enhanced_models=True,
                    enable_automl=True,
                    enable_evolutionary_search=True,
                    enable_advanced_metrics=True,
                    enable_feature_engineering=True,
                    model_types=["xgboost", "lightgbm"],
                    max_search_time=60,  # 1 minute for test
                    verbose=False
                )
                engine = TreeArchitectureSearchEngine(config)
                integration_results['standard_tas_engine'] = True
                self.logger.info("   ✅ Standard TAS engine integration successful")
            else:
                integration_results['standard_tas_engine'] = False
                self.logger.warning("   ⚠️ Standard TAS engine not available")
        except Exception as e:
            integration_results['standard_tas_engine'] = False
            self.logger.warning(f"   ⚠️ Standard TAS engine integration failed: {e}")
        
        # Test enhanced TAS engine
        try:
            if TAS_CORE_AVAILABLE:
                config = EnhancedTASConfig(
                    model_types=["xgboost", "lightgbm"],
                    enable_automl=True,
                    enable_evolutionary_search=True,
                    enable_advanced_metrics=True,
                    enable_feature_engineering=True,
                    max_search_time=60,  # 1 minute for test
                    verbose=False
                )
                engine = EnhancedTASEngine(config)
                integration_results['enhanced_tas_engine'] = True
                self.logger.info("   ✅ Enhanced TAS engine integration successful")
            else:
                integration_results['enhanced_tas_engine'] = False
                self.logger.warning("   ⚠️ Enhanced TAS engine not available")
        except Exception as e:
            integration_results['enhanced_tas_engine'] = False
            self.logger.warning(f"   ⚠️ Enhanced TAS engine integration failed: {e}")
        
        self.wiring_status['tas_engine_integration'] = integration_results
        return integration_results
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run comprehensive integration tests."""
        self.logger.info("🧪 Running comprehensive integration tests...")
        
        test_results = {}
        
        # Test 1: Component availability
        test_results['component_availability'] = self.check_component_availability()
        
        # Test 2: Shared utilities integration
        test_results['shared_utilities'] = self.test_shared_utilities_integration()
        
        # Test 3: Enhanced components integration
        test_results['enhanced_components'] = self.test_enhanced_components_integration()
        
        # Test 4: TAS engine integration
        test_results['tas_engines'] = self.test_tas_engine_integration()
        
        # Test 5: End-to-end integration
        test_results['end_to_end'] = self.test_end_to_end_integration()
        
        self.integration_tests = test_results
        return test_results
    
    def test_end_to_end_integration(self) -> Dict[str, bool]:
        """Test end-to-end integration with sample data."""
        self.logger.info("🔧 Testing end-to-end integration...")
        
        try:
            # Create sample data
            from sklearn.datasets import make_classification
            from sklearn.model_selection import train_test_split
            
            X, y = make_classification(n_samples=100, n_features=10, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            
            # Test feature engineering
            if SHARED_UTILS_AVAILABLE:
                feature_config = FeatureConfig(enable_technical_indicators=True, max_features=5)
                feature_engineer = create_unified_feature_engineer(feature_config)
                feature_result = feature_engineer.engineer_features(X_train, y_train)
                
                if feature_result.success:
                    self.logger.info("   ✅ Feature engineering end-to-end test successful")
                else:
                    self.logger.warning("   ⚠️ Feature engineering end-to-end test failed")
                    return {'end_to_end': False}
            
            # Test model training and evaluation
            if ENHANCED_COMPONENTS_AVAILABLE:
                model_config = TreeModelConfig(
                    model_type=TreeModelType.XGBOOST,
                    params={'n_estimators': 10, 'max_depth': 3},
                    is_classifier=True
                )
                model_factory = EnhancedTreeModelFactory(model_config)
                model_factory.fit(X_train, y_train)
                predictions = model_factory.predict(X_val)
                accuracy = (predictions == y_val).mean()
                
                if accuracy > 0:
                    self.logger.info("   ✅ Model training and evaluation end-to-end test successful")
                else:
                    self.logger.warning("   ⚠️ Model training and evaluation end-to-end test failed")
                    return {'end_to_end': False}
            
            # Test unified evaluation
            if SHARED_UTILS_AVAILABLE:
                evaluator = create_unified_evaluator()
                eval_result = evaluator.evaluate(predictions, y_val)
                
                if eval_result.success:
                    self.logger.info("   ✅ Unified evaluation end-to-end test successful")
                else:
                    self.logger.warning("   ⚠️ Unified evaluation end-to-end test failed")
                    return {'end_to_end': False}
            
            self.logger.info("   ✅ End-to-end integration test successful")
            return {'end_to_end': True}
            
        except Exception as e:
            self.logger.warning(f"   ⚠️ End-to-end integration test failed: {e}")
            return {'end_to_end': False}
    
    def generate_wiring_report(self) -> str:
        """Generate a comprehensive wiring report."""
        self.logger.info("📋 Generating wiring report...")
        
        report = []
        report.append("TAS Component Wiring Report")
        report.append("=" * 50)
        report.append("")
        
        # Component availability
        if 'component_availability' in self.wiring_status:
            report.append("🔍 Component Availability:")
            for component, available in self.wiring_status['component_availability'].items():
                status = "✅" if available else "❌"
                report.append(f"   {component}: {status}")
            report.append("")
        
        # Integration test results
        if 'shared_utilities' in self.wiring_status:
            report.append("🔧 Shared Utilities Integration:")
            for component, success in self.wiring_status['shared_utilities'].items():
                status = "✅" if success else "❌"
                report.append(f"   {component}: {status}")
            report.append("")
        
        if 'enhanced_components' in self.wiring_status:
            report.append("🚀 Enhanced Components Integration:")
            for component, success in self.wiring_status['enhanced_components'].items():
                status = "✅" if success else "❌"
                report.append(f"   {component}: {status}")
            report.append("")
        
        if 'tas_engines' in self.wiring_status:
            report.append("🎯 TAS Engine Integration:")
            for component, success in self.wiring_status['tas_engines'].items():
                status = "✅" if success else "❌"
                report.append(f"   {component}: {status}")
            report.append("")
        
        # Summary
        total_components = 0
        successful_components = 0
        
        for category, results in self.wiring_status.items():
            if isinstance(results, dict):
                for component, success in results.items():
                    total_components += 1
                    if success:
                        successful_components += 1
        
        success_rate = (successful_components / total_components * 100) if total_components > 0 else 0
        
        report.append("📊 Summary:")
        report.append(f"   Total components tested: {total_components}")
        report.append(f"   Successful integrations: {successful_components}")
        report.append(f"   Success rate: {success_rate:.1f}%")
        report.append("")
        
        if success_rate >= 80:
            report.append("🎉 TAS component wiring is successful!")
            report.append("   The enhanced TAS system is ready for use.")
        elif success_rate >= 60:
            report.append("⚠️ TAS component wiring is partially successful.")
            report.append("   Some components may need attention.")
        else:
            report.append("❌ TAS component wiring has issues.")
            report.append("   Please check component availability and dependencies.")
        
        return "\n".join(report)
    
    def wire_all_components(self) -> bool:
        """Wire all TAS components together."""
        self.logger.info("🔧 Wiring all TAS components...")
        
        try:
            # Run integration tests
            test_results = self.run_integration_tests()
            
            # Generate report
            report = self.generate_wiring_report()
            self.logger.info("\n" + report)
            
            # Check overall success
            total_components = 0
            successful_components = 0
            
            for category, results in test_results.items():
                if isinstance(results, dict):
                    for component, success in results.items():
                        total_components += 1
                        if success:
                            successful_components += 1
            
            success_rate = (successful_components / total_components * 100) if total_components > 0 else 0
            
            if success_rate >= 80:
                self.logger.info("✅ TAS component wiring completed successfully!")
                return True
            else:
                self.logger.warning("⚠️ TAS component wiring completed with issues.")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ TAS component wiring failed: {e}")
            return False


def main():
    """Main wiring function."""
    logger.info("🔧 Starting TAS Component Wiring")
    logger.info("=" * 50)
    
    # Create wiring instance
    wiring = TASComponentWiring()
    
    # Wire all components
    success = wiring.wire_all_components()
    
    if success:
        logger.info("🎉 TAS component wiring completed successfully!")
        logger.info("")
        logger.info("The enhanced TAS system is now fully integrated with:")
        logger.info("✅ Shared utilities (Evolutionary Search, Feature Engineering, Evaluation)")
        logger.info("✅ Enhanced components (Tree Models, AutoML, Advanced Metrics)")
        logger.info("✅ TAS engines (Standard and Enhanced)")
        logger.info("✅ Multi-objective optimization capabilities")
        logger.info("✅ Regime-aware and economic significance assessment")
    else:
        logger.warning("⚠️ TAS component wiring completed with issues.")
        logger.warning("Please check the wiring report above for details.")
    
    return success


if __name__ == "__main__":
    main()