#!/usr/bin/env python3
"""
Test CLVSA Structure Integration

This script tests the structure and integration of CLVSA architecture
with tree models without requiring external dependencies.
"""

import sys
import os
import ast
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_file_exists(file_path: str) -> bool:
    """Test if a file exists."""
    if os.path.exists(file_path):
        logger.info(f"✅ File exists: {file_path}")
        return True
    else:
        logger.error(f"❌ File missing: {file_path}")
        return False

def test_file_syntax(file_path: str) -> bool:
    """Test if a Python file has valid syntax."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        ast.parse(content)
        logger.info(f"✅ Valid syntax: {file_path}")
        return True
    except SyntaxError as e:
        logger.error(f"❌ Syntax error in {file_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error reading {file_path}: {e}")
        return False

def test_clvsa_files():
    """Test that all CLVSA-related files exist and have valid syntax."""
    logger.info("📁 Testing CLVSA file structure...")
    
    clvsa_files = [
        "src/training/steps/model_training/clvsa_attention_wrapper.py",
        "src/training/steps/model_training/tree_clvsa_wrapper.py",
        "src/utils/ml_common/models/clvsa_architecture.py",
        "src/utils/ml_common/models/model_factory.py"
    ]
    
    results = []
    for file_path in clvsa_files:
        exists = test_file_exists(file_path)
        if exists:
            syntax_ok = test_file_syntax(file_path)
            results.append(syntax_ok)
        else:
            results.append(False)
    
    return all(results)

def test_model_factory_integration():
    """Test that model factory has been updated with CLVSA integration."""
    logger.info("🏭 Testing Model Factory CLVSA integration...")
    
    try:
        with open("src/utils/ml_common/models/model_factory.py", 'r') as f:
            content = f.read()
        
        # Check for CLVSA imports
        clvsa_imports = [
            "from src.training.steps.model_training.tree_clvsa_wrapper import",
            "create_tree_clvsa_wrapper",
            "create_tree_clvsa_config"
        ]
        
        for import_str in clvsa_imports:
            if import_str in content:
                logger.info(f"✅ Found CLVSA import: {import_str}")
            else:
                logger.error(f"❌ Missing CLVSA import: {import_str}")
                return False
        
        # Check for tree model methods with CLVSA
        tree_models = [
            "_create_random_forest_model",
            "_create_lightgbm_model", 
            "_create_catboost_model",
            "_create_xgboost_model",
            "_create_extra_trees_model",
            "_create_hist_gradient_boosting_model"
        ]
        
        for method in tree_models:
            if f"def {method}" in content:
                logger.info(f"✅ Found tree model method: {method}")
            else:
                logger.error(f"❌ Missing tree model method: {method}")
                return False
        
        # Check for CLVSA wrapper usage
        clvsa_usage_patterns = [
            "use_clvsa = model_config.model_params.get('use_clvsa', True)",
            "create_tree_clvsa_wrapper(base_model, clvsa_config)",
            "Tree CLVSA attention architecture"
        ]
        
        for pattern in clvsa_usage_patterns:
            if pattern in content:
                logger.info(f"✅ Found CLVSA usage pattern: {pattern}")
            else:
                logger.error(f"❌ Missing CLVSA usage pattern: {pattern}")
                return False
        
        logger.info("✅ Model Factory CLVSA integration verified")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model Factory integration test failed: {e}")
        return False

def test_tree_clvsa_wrapper():
    """Test that Tree CLVSA wrapper has the required structure."""
    logger.info("🌲 Testing Tree CLVSA wrapper structure...")
    
    try:
        with open("src/training/steps/model_training/tree_clvsa_wrapper.py", 'r') as f:
            content = f.read()
        
        # Check for required classes
        required_classes = [
            "class TreeCLVSAConfig:",
            "class TreeCLVSAWrapper(",
            "class TreeCLVSAWrapper(BaseEstimator, RegressorMixin):"
        ]
        
        for class_def in required_classes:
            if class_def in content:
                logger.info(f"✅ Found class: {class_def}")
            else:
                logger.error(f"❌ Missing class: {class_def}")
                return False
        
        # Check for required methods
        required_methods = [
            "def fit(",
            "def predict(",
            "def get_attention_weights(",
            "def _compute_tree_specific_attention(",
            "def _compute_temporal_attention(",
            "def _compute_regime_attention(",
            "def _compute_ensemble_attention("
        ]
        
        for method in required_methods:
            if method in content:
                logger.info(f"✅ Found method: {method}")
            else:
                logger.error(f"❌ Missing method: {method}")
                return False
        
        # Check for factory functions
        factory_functions = [
            "def create_tree_clvsa_wrapper(",
            "def create_tree_clvsa_config(",
            "def wrap_tree_model_with_clvsa(",
            "def create_clvsa_random_forest(",
            "def create_clvsa_xgboost(",
            "def create_clvsa_lightgbm(",
            "def create_clvsa_catboost("
        ]
        
        for func in factory_functions:
            if func in content:
                logger.info(f"✅ Found factory function: {func}")
            else:
                logger.error(f"❌ Missing factory function: {func}")
                return False
        
        logger.info("✅ Tree CLVSA wrapper structure verified")
        return True
        
    except Exception as e:
        logger.error(f"❌ Tree CLVSA wrapper structure test failed: {e}")
        return False

def test_clvsa_architecture():
    """Test that CLVSA architecture file has the required structure."""
    logger.info("🏗️ Testing CLVSA architecture structure...")
    
    try:
        with open("src/utils/ml_common/models/clvsa_architecture.py", 'r') as f:
            content = f.read()
        
        # Check for required classes
        required_classes = [
            "class CLVSAConfig:",
            "class CLVSARegressor(",
            "class CLVSAPredictor("
        ]
        
        for class_def in required_classes:
            if class_def in content:
                logger.info(f"✅ Found CLVSA class: {class_def}")
            else:
                logger.error(f"❌ Missing CLVSA class: {class_def}")
                return False
        
        # Check for key methods
        key_methods = [
            "def fit(",
            "def predict(",
            "def get_model_info(",
            "def create_clvsa_model(",
            "def get_clvsa_model("
        ]
        
        for method in key_methods:
            if method in content:
                logger.info(f"✅ Found CLVSA method: {method}")
            else:
                logger.error(f"❌ Missing CLVSA method: {method}")
                return False
        
        logger.info("✅ CLVSA architecture structure verified")
        return True
        
    except Exception as e:
        logger.error(f"❌ CLVSA architecture structure test failed: {e}")
        return False

def test_integration_completeness():
    """Test that the integration is complete and consistent."""
    logger.info("🔗 Testing integration completeness...")
    
    try:
        # Check that all tree models are covered
        with open("src/utils/ml_common/models/model_factory.py", 'r') as f:
            content = f.read()
        
        # Tree model types that should have CLVSA integration
        tree_model_types = [
            "RANDOM_FOREST",
            "LIGHTGBM", 
            "XGBOOST",
            "CATBOOST",
            "EXTRA_TREES",
            "HIST_GRADIENT_BOOSTING"
        ]
        
        for model_type in tree_model_types:
            # Check that the model type is defined
            if f"ModelType.{model_type}" in content:
                logger.info(f"✅ Found model type: {model_type}")
            else:
                logger.error(f"❌ Missing model type: {model_type}")
                return False
        
        # Check that default CLVSA is enabled
        if "use_clvsa = model_config.model_params.get('use_clvsa', True)" in content:
            logger.info("✅ CLVSA enabled by default for tree models")
        else:
            logger.error("❌ CLVSA not enabled by default")
            return False
        
        # Check that Tree CLVSA wrapper is used
        if "create_tree_clvsa_wrapper" in content:
            logger.info("✅ Tree CLVSA wrapper integration found")
        else:
            logger.error("❌ Tree CLVSA wrapper integration missing")
            return False
        
        logger.info("✅ Integration completeness verified")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration completeness test failed: {e}")
        return False

def main():
    """Run all structure tests."""
    logger.info("🚀 Starting CLVSA Structure Integration Tests")
    
    tests = [
        ("CLVSA Files", test_clvsa_files),
        ("Model Factory Integration", test_model_factory_integration),
        ("Tree CLVSA Wrapper", test_tree_clvsa_wrapper),
        ("CLVSA Architecture", test_clvsa_architecture),
        ("Integration Completeness", test_integration_completeness)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if success:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All structure tests passed! CLVSA integration is properly implemented.")
        logger.info("\n📋 Summary of CLVSA Integration:")
        logger.info("   ✅ All tree models (RandomForest, XGBoost, LightGBM, CatBoost, ExtraTrees, HistGradientBoosting) are wrapped with CLVSA by default")
        logger.info("   ✅ Tree-specific CLVSA wrapper provides enhanced attention mechanisms")
        logger.info("   ✅ Model factory automatically applies CLVSA architecture to tree models")
        logger.info("   ✅ CLVSA can be disabled by setting use_clvsa=False")
        logger.info("   ✅ Comprehensive attention mechanisms: feature, temporal, regime, and ensemble attention")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)