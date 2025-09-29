"""
Test Script for Regime HPO Integration

This script validates the HPO integration with regime training configurations,
ensuring all components work correctly together.

Test Coverage:
- RegimeHPOWrapper functionality
- RegimeHPOIntegration pipeline
- Search space generation
- Model factory creation
- Optimization execution
- Results validation
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
import logging
import time
from datetime import datetime
import yaml
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

# Import regime HPO components
from src.utils.ml_common.optimization.regime_hpo_wrapper import (
    RegimeHPOWrapper, 
    RegimeHPOConfig, 
    RegimeHPOResult,
    optimize_regime_models,
    create_regime_hpo_config
)

from src.training.steps.market_analysis.hybrid_nas_tas_regime.automatic_training.regime_hpo_integration import (
    RegimeHPOIntegration,
    run_regime_optimization,
    create_regime_hpo_integration_config
)

# Import regime training components
from src.training.steps.market_analysis.hybrid_nas_tas_regime.config.hybrid_regime_config import HybridRegimeConfig
from src.training.steps.market_analysis.hybrid_nas_tas_regime.automatic_training.regime_training_pipeline import RegimeTrainingConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RegimeHPOTestSuite:
    """Test suite for regime HPO integration."""
    
    def __init__(self):
        """Initialize test suite."""
        self.logger = logging.getLogger(__name__)
        self.test_results = {}
        self.synthetic_data = self._create_synthetic_data()
        
    def _create_synthetic_data(self) -> pd.DataFrame:
        """Create synthetic market data for testing."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        
        # Create synthetic market data
        data = {
            'price': np.random.randn(n_samples).cumsum() + 100,
            'volume': np.random.exponential(1000, n_samples),
            'volatility': np.random.exponential(0.02, n_samples),
            'momentum': np.random.randn(n_samples),
            'rsi': np.random.uniform(0, 100, n_samples),
            'macd': np.random.randn(n_samples),
            'bollinger_upper': np.random.randn(n_samples) + 2,
            'bollinger_lower': np.random.randn(n_samples) - 2,
            'sma_20': np.random.randn(n_samples) + 100,
            'ema_12': np.random.randn(n_samples) + 100,
        }
        
        # Add more features
        for i in range(10):
            data[f'feature_{i}'] = np.random.randn(n_samples)
        
        return pd.DataFrame(data)
    
    def test_regime_hpo_wrapper_initialization(self) -> bool:
        """Test RegimeHPOWrapper initialization."""
        self.logger.info("🧪 Testing RegimeHPOWrapper initialization...")
        
        try:
            # Test with default config
            wrapper = RegimeHPOWrapper()
            assert wrapper is not None
            assert wrapper.hpo_config is not None
            assert wrapper.regime_search_spaces is not None
            
            # Test with custom config
            custom_config = RegimeHPOConfig(
                base_model_n_trials=50,
                meta_model_n_trials=25,
                enable_meta_feature_optimization=True
            )
            wrapper_custom = RegimeHPOWrapper(hpo_config=custom_config)
            assert wrapper_custom.hpo_config.base_model_n_trials == 50
            
            self.logger.info("✅ RegimeHPOWrapper initialization test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ RegimeHPOWrapper initialization test failed: {e}")
            return False
    
    def test_search_space_generation(self) -> bool:
        """Test search space generation."""
        self.logger.info("🧪 Testing search space generation...")
        
        try:
            wrapper = RegimeHPOWrapper()
            search_spaces = wrapper.regime_search_spaces
            
            # Check that search spaces are generated
            assert len(search_spaces) > 0
            assert 'catboost' in search_spaces or 'extratrees' in search_spaces
            
            # Check search space structure
            for model_type, search_space in search_spaces.items():
                assert isinstance(search_space, dict)
                for param_name, param_config in search_space.items():
                    assert 'type' in param_config
                    assert param_config['type'] in ['int', 'float', 'categorical']
                    if param_config['type'] in ['int', 'float']:
                        assert 'low' in param_config
                        assert 'high' in param_config
            
            self.logger.info("✅ Search space generation test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Search space generation test failed: {e}")
            return False
    
    def test_model_factory_creation(self) -> bool:
        """Test model factory creation."""
        self.logger.info("🧪 Testing model factory creation...")
        
        try:
            wrapper = RegimeHPOWrapper()
            
            # Test CatBoost factory
            catboost_factory = wrapper._create_catboost_factory()
            assert callable(catboost_factory)
            
            # Test ExtraTrees factory
            extratrees_factory = wrapper._create_extratrees_factory()
            assert callable(extratrees_factory)
            
            # Test LightGBM meta factory
            lightgbm_factory = wrapper._create_lightgbm_meta_factory()
            assert callable(lightgbm_factory)
            
            self.logger.info("✅ Model factory creation test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Model factory creation test failed: {e}")
            return False
    
    def test_regime_hpo_integration_initialization(self) -> bool:
        """Test RegimeHPOIntegration initialization."""
        self.logger.info("🧪 Testing RegimeHPOIntegration initialization...")
        
        try:
            # Test with default configs
            integration = RegimeHPOIntegration()
            assert integration is not None
            assert integration.hpo_wrapper is not None
            assert integration.regime_pipeline is not None
            
            # Test with custom configs
            regime_config = HybridRegimeConfig()
            hpo_config = create_regime_hpo_integration_config(
                optimization_strategy='hierarchical',
                base_model_n_trials=50
            )
            integration_custom = RegimeHPOIntegration(
                regime_config=regime_config,
                hpo_config=hpo_config
            )
            assert integration_custom is not None
            
            self.logger.info("✅ RegimeHPOIntegration initialization test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ RegimeHPOIntegration initialization test failed: {e}")
            return False
    
    def test_data_preparation(self) -> bool:
        """Test data preparation for optimization."""
        self.logger.info("🧪 Testing data preparation...")
        
        try:
            integration = RegimeHPOIntegration()
            
            # Create synthetic regime labels
            regime_labels = np.random.randint(0, 3, len(self.synthetic_data))
            
            # Test data preparation
            X, y = integration._prepare_optimization_data(
                self.synthetic_data, 
                regime_labels
            )
            
            assert X.shape[0] == len(self.synthetic_data)
            assert len(y) == len(regime_labels)
            assert X.shape[1] > 0
            
            self.logger.info("✅ Data preparation test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Data preparation test failed: {e}")
            return False
    
    def test_optimization_execution(self) -> bool:
        """Test optimization execution (simplified)."""
        self.logger.info("🧪 Testing optimization execution...")
        
        try:
            # Create small test data
            X = np.random.randn(100, 10)
            y = np.random.randint(0, 3, 100)
            
            # Test base model optimization (simplified)
            wrapper = RegimeHPOWrapper()
            
            # Test with minimal trials for speed
            test_config = RegimeHPOConfig(
                base_model_n_trials=2,
                meta_model_n_trials=2,
                enable_meta_feature_optimization=False
            )
            test_wrapper = RegimeHPOWrapper(hpo_config=test_config)
            
            # Test optimization (this might take a while)
            self.logger.info("⏳ Running optimization test (this may take a few minutes)...")
            results = test_wrapper.optimize_regime_base_models(X, y)
            
            assert 'results' in results
            assert 'optimization_time' in results
            
            self.logger.info("✅ Optimization execution test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Optimization execution test failed: {e}")
            return False
    
    def test_configuration_loading(self) -> bool:
        """Test configuration loading from files."""
        self.logger.info("🧪 Testing configuration loading...")
        
        try:
            # Test regime base config loading
            base_config_path = "src/config/regime_base_training_config.yaml"
            if os.path.exists(base_config_path):
                wrapper = RegimeHPOWrapper(regime_base_config_path=base_config_path)
                assert wrapper.regime_base_config is not None
            
            # Test regime meta-model config loading
            metamodel_config_path = "src/config/regime_metamodel_training_config.yaml"
            if os.path.exists(metamodel_config_path):
                wrapper = RegimeHPOWrapper(regime_metamodel_config_path=metamodel_config_path)
                assert wrapper.regime_metamodel_config is not None
            
            # Test HPO config loading
            hpo_config_path = "src/config/regime_hpo_config.yaml"
            if os.path.exists(hpo_config_path):
                with open(hpo_config_path, 'r') as f:
                    hpo_config_data = yaml.safe_load(f)
                assert hpo_config_data is not None
                assert 'base_model_optimization' in hpo_config_data
            
            self.logger.info("✅ Configuration loading test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Configuration loading test failed: {e}")
            return False
    
    def test_results_serialization(self) -> bool:
        """Test results serialization and saving."""
        self.logger.info("🧪 Testing results serialization...")
        
        try:
            # Create mock results
            mock_results = RegimeHPOResult(
                base_model_results={'test': {'best_score': 0.8}},
                base_model_best_params={'test': {'param1': 1, 'param2': 0.5}},
                base_model_best_scores={'test': 0.8},
                meta_model_results={'best_score': 0.85},
                meta_model_best_params={'param1': 1, 'param2': 0.5},
                meta_model_best_score=0.85,
                total_optimization_time=120.5,
                optimization_strategy='hierarchical',
                n_total_trials=150,
                convergence_info={'converged': True}
            )
            
            # Test serialization
            wrapper = RegimeHPOWrapper()
            test_filepath = "test_optimization_results.yaml"
            wrapper.save_optimization_results(mock_results, test_filepath)
            
            # Verify file was created
            assert os.path.exists(test_filepath)
            
            # Clean up
            os.remove(test_filepath)
            
            self.logger.info("✅ Results serialization test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Results serialization test failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests in the suite."""
        self.logger.info("🚀 Starting Regime HPO Integration Test Suite")
        start_time = time.time()
        
        tests = [
            ("RegimeHPOWrapper Initialization", self.test_regime_hpo_wrapper_initialization),
            ("Search Space Generation", self.test_search_space_generation),
            ("Model Factory Creation", self.test_model_factory_creation),
            ("RegimeHPOIntegration Initialization", self.test_regime_hpo_integration_initialization),
            ("Data Preparation", self.test_data_preparation),
            ("Configuration Loading", self.test_configuration_loading),
            ("Results Serialization", self.test_results_serialization),
            ("Optimization Execution", self.test_optimization_execution),
        ]
        
        results = {}
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            self.logger.info(f"🧪 Running test: {test_name}")
            try:
                result = test_func()
                results[test_name] = result
                if result:
                    passed += 1
                    self.logger.info(f"✅ {test_name} PASSED")
                else:
                    self.logger.error(f"❌ {test_name} FAILED")
            except Exception as e:
                self.logger.error(f"❌ {test_name} FAILED with exception: {e}")
                results[test_name] = False
        
        total_time = time.time() - start_time
        
        # Summary
        self.logger.info("=" * 60)
        self.logger.info("🏆 REGIME HPO INTEGRATION TEST SUITE SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"📊 Tests Passed: {passed}/{total}")
        self.logger.info(f"⏱️ Total Time: {total_time:.2f}s")
        self.logger.info(f"📈 Success Rate: {passed/total*100:.1f}%")
        
        if passed == total:
            self.logger.info("🎉 ALL TESTS PASSED! HPO integration is working correctly.")
        else:
            self.logger.warning(f"⚠️ {total-passed} tests failed. Please check the implementation.")
        
        self.logger.info("=" * 60)
        
        return results

def main():
    """Run the test suite."""
    print("🧪 Regime HPO Integration Test Suite")
    print("=" * 50)
    
    test_suite = RegimeHPOTestSuite()
    results = test_suite.run_all_tests()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"regime_hpo_test_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"📄 Test results saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    main()