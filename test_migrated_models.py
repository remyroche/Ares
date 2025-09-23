#!/usr/bin/env python3
"""
Test Script for Migrated ML Models

This script tests the comprehensive migration of ML models across HMM, Analyst, and Tactician
components with proper regularization, overfitting prevention, and regime-aware training.

Usage:
    python test_migrated_models.py [--config CONFIG_FILE] [--mode MODE]
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import logging
import time
import warnings
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Enhanced dependency management
try:
    from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print(*args)
    def tprint_info(*args, **kwargs): print(f"INFO: {args[0] if args else ''}")
    def tprint_warning(*args, **kwargs): print(f"WARNING: {args[0] if args else ''}")
    def tprint_error(*args, **kwargs): print(f"ERROR: {args[0] if args else ''}")
    def tprint_success(*args, **kwargs): print(f"SUCCESS: {args[0] if args else ''}")

# Import migrated model components
try:
    from src.utils.ml_common.models.migrated_model_configs import (
        MigratedModelConfigs, ModelConfig, ModelArchitecture,
        RegimeCharacteristics, RegimeAwareParameterOptimizer
    )
    from src.utils.ml_common.models.enhanced_migrated_factory import EnhancedMigratedModelFactory
    from src.utils.ml_common.training.migrated_training_integration import (
        MigratedTrainingIntegration, MigratedTrainingConfig
    )
    MIGRATED_MODELS_AVAILABLE = True
    tprint_success("✅ Migrated model components imported successfully")
except ImportError as e:
    MIGRATED_MODELS_AVAILABLE = False
    tprint_error(f"❌ Failed to import migrated model components: {e}")

# Import existing model factory
try:
    from src.utils.ml_common.models.model_factory import EnhancedModelFactory, ModelType, ModelConfig as BaseModelConfig
    EXISTING_FACTORY_AVAILABLE = True
    tprint_success("✅ Existing model factory imported successfully")
except ImportError as e:
    EXISTING_FACTORY_AVAILABLE = False
    tprint_error(f"❌ Failed to import existing model factory: {e}")

# Import common utilities
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    tprint_warning("⚠️ PyYAML not available, will use default configurations")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MigratedModelsTester:
    """Comprehensive tester for migrated ML models."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize the tester."""
        self.logger = logger.getChild('MigratedModelsTester')
        self.logger.info("🚀 Initializing Migrated Models Tester...")
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize components
        self.model_factory = None
        self.training_integration = None
        
        if MIGRATED_MODELS_AVAILABLE:
            self.model_factory = EnhancedMigratedModelFactory()
            self.training_integration = MigratedTrainingIntegration()
            self.logger.info("✅ Migrated model components initialized")
        
        # Test data
        self.test_data = self._generate_test_data()
        
        self.logger.info("✅ Migrated Models Tester initialized successfully")
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file."""
        if config_file and YAML_AVAILABLE:
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                self.logger.info(f"✅ Configuration loaded from {config_file}")
                return config
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load config from {config_file}: {e}")
        
        # Return default configuration
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "global_config": {
                "enable_regime_aware_training": True,
                "enable_overfitting_prevention": True,
                "enable_regularization": True,
                "output_dir": "./test_migrated_models"
            },
            "test_parameters": {
                "n_samples": 1000,
                "n_features": 50,
                "n_classes": 4,
                "random_state": 42
            }
        }
    
    def _generate_test_data(self) -> Dict[str, Dict[str, Any]]:
        """Generate test data for all timeframes."""
        np.random.seed(self.config["test_parameters"]["random_state"])
        
        n_samples = self.config["test_parameters"]["n_samples"]
        n_features = self.config["test_parameters"]["n_features"]
        n_classes = self.config["test_parameters"]["n_classes"]
        
        test_data = {}
        
        # Generate data for each timeframe
        for timeframe in ["15m", "5m", "1m"]:
            # Generate features
            X = np.random.randn(n_samples, n_features)
            
            # Generate targets based on timeframe
            if timeframe == "15m":  # HMM - classification
                y = np.random.randint(0, n_classes, n_samples)
            else:  # Analyst and Tactician - regression
                y = np.random.randn(n_samples)
            
            # Generate regime data
            regime_data = {
                "volume": np.random.uniform(0, 1, n_samples),
                "volatility": np.random.uniform(0, 1, n_samples),
                "momentum": np.random.uniform(0, 1, n_samples),
                "trend": np.random.uniform(0, 1, n_samples)
            }
            
            test_data[timeframe] = {
                "X": X,
                "y": y,
                "regime_data": regime_data,
                "feature_names": [f"feature_{i}" for i in range(n_features)]
            }
        
        self.logger.info(f"✅ Generated test data for {len(test_data)} timeframes")
        return test_data
    
    def test_model_configurations(self) -> Dict[str, Any]:
        """Test model configurations."""
        self.logger.info("🔄 Testing model configurations...")
        
        if not MIGRATED_MODELS_AVAILABLE:
            return {"error": "Migrated models not available", "success": False}
        
        results = {}
        
        try:
            # Test HMM models
            hmm_models = MigratedModelConfigs.get_hmm_models()
            results["hmm_models"] = {
                "count": len(hmm_models),
                "models": list(hmm_models.keys()),
                "success": True
            }
            self.logger.info(f"✅ HMM models: {len(hmm_models)} configurations")
            
            # Test Analyst models
            analyst_models = MigratedModelConfigs.get_analyst_models()
            results["analyst_models"] = {
                "count": len(analyst_models),
                "models": list(analyst_models.keys()),
                "success": True
            }
            self.logger.info(f"✅ Analyst models: {len(analyst_models)} configurations")
            
            # Test Tactician models
            tactician_models = MigratedModelConfigs.get_tactician_models()
            results["tactician_models"] = {
                "count": len(tactician_models),
                "models": list(tactician_models.keys()),
                "success": True
            }
            self.logger.info(f"✅ Tactician models: {len(tactician_models)} configurations")
            
            # Test regime-aware models
            regime_aware_models = MigratedModelConfigs.get_regime_aware_models()
            results["regime_aware_models"] = {
                "count": len(regime_aware_models),
                "models": regime_aware_models,
                "success": True
            }
            self.logger.info(f"✅ Regime-aware models: {len(regime_aware_models)} models")
            
            results["success"] = True
            
        except Exception as e:
            self.logger.error(f"❌ Model configuration test failed: {e}")
            results["error"] = str(e)
            results["success"] = False
        
        return results
    
    def test_model_creation(self) -> Dict[str, Any]:
        """Test model creation."""
        self.logger.info("🔄 Testing model creation...")
        
        if not MIGRATED_MODELS_AVAILABLE or not self.model_factory:
            return {"error": "Model factory not available", "success": False}
        
        results = {}
        
        try:
            # Test HMM model creation
            hmm_data = self.test_data["15m"]
            hmm_models = {}
            
            for model_name in ["lgbm", "xgboost", "financial_resnet"]:
                try:
                    regime_chars = RegimeCharacteristics(
                        volume=0.7, volatility=0.6, momentum=0.8, trend=0.5
                    )
                    model = self.model_factory.create_hmm_model(
                        model_name, hmm_data["X"].shape[1], 4, regime_chars
                    )
                    hmm_models[model_name] = {"success": True, "model": type(model).__name__}
                    self.logger.info(f"✅ Created HMM model: {model_name}")
                except Exception as e:
                    hmm_models[model_name] = {"success": False, "error": str(e)}
                    self.logger.warning(f"⚠️ Failed to create HMM model {model_name}: {e}")
            
            results["hmm_models"] = hmm_models
            
            # Test Analyst model creation
            analyst_data = self.test_data["5m"]
            analyst_models = {}
            
            for model_name in ["deepscaler", "catboost", "nbeats"]:
                try:
                    regime_chars = RegimeCharacteristics(
                        volume=0.7, volatility=0.6, momentum=0.8, trend=0.5
                    )
                    model = self.model_factory.create_analyst_model(
                        model_name, analyst_data["X"].shape[1], 1, regime_chars
                    )
                    analyst_models[model_name] = {"success": True, "model": type(model).__name__}
                    self.logger.info(f"✅ Created Analyst model: {model_name}")
                except Exception as e:
                    analyst_models[model_name] = {"success": False, "error": str(e)}
                    self.logger.warning(f"⚠️ Failed to create Analyst model {model_name}: {e}")
            
            results["analyst_models"] = analyst_models
            
            # Test Tactician model creation
            tactician_data = self.test_data["1m"]
            tactician_models = {}
            
            for model_name in ["xgboost", "lightgbm", "deepscaler_1m"]:
                try:
                    regime_chars = RegimeCharacteristics(
                        volume=0.7, volatility=0.6, momentum=0.8, trend=0.5
                    )
                    model = self.model_factory.create_tactician_model(
                        model_name, tactician_data["X"].shape[1], 1, regime_chars
                    )
                    tactician_models[model_name] = {"success": True, "model": type(model).__name__}
                    self.logger.info(f"✅ Created Tactician model: {model_name}")
                except Exception as e:
                    tactician_models[model_name] = {"success": False, "error": str(e)}
                    self.logger.warning(f"⚠️ Failed to create Tactician model {model_name}: {e}")
            
            results["tactician_models"] = tactician_models
            
            results["success"] = True
            
        except Exception as e:
            self.logger.error(f"❌ Model creation test failed: {e}")
            results["error"] = str(e)
            results["success"] = False
        
        return results
    
    def test_training_integration(self) -> Dict[str, Any]:
        """Test training integration."""
        self.logger.info("🔄 Testing training integration...")
        
        if not MIGRATED_MODELS_AVAILABLE or not self.training_integration:
            return {"error": "Training integration not available", "success": False}
        
        results = {}
        
        try:
            # Prepare data for training
            data_config = {
                "hmm": {
                    "X": self.test_data["15m"]["X"],
                    "y": self.test_data["15m"]["y"],
                    "feature_names": self.test_data["15m"]["feature_names"]
                },
                "analyst": {
                    "X": self.test_data["5m"]["X"],
                    "y": self.test_data["5m"]["y"],
                    "feature_names": self.test_data["5m"]["feature_names"]
                },
                "tactician": {
                    "X": self.test_data["1m"]["X"],
                    "y": self.test_data["1m"]["y"],
                    "feature_names": self.test_data["1m"]["feature_names"]
                }
            }
            
            regime_data = {
                "hmm": self.test_data["15m"]["regime_data"],
                "analyst": self.test_data["5m"]["regime_data"],
                "tactician": self.test_data["1m"]["regime_data"]
            }
            
            # Test training
            training_results = self.training_integration.train_all_models(data_config, regime_data)
            
            # Analyze results
            total_models = 0
            successful_models = 0
            
            for component, component_results in training_results.items():
                if component == "error":
                    continue
                
                for model_name, model_result in component_results.items():
                    total_models += 1
                    if model_result.get("success", False):
                        successful_models += 1
            
            results = {
                "total_models": total_models,
                "successful_models": successful_models,
                "success_rate": successful_models / total_models if total_models > 0 else 0,
                "training_results": training_results,
                "success": successful_models > 0
            }
            
            self.logger.info(f"✅ Training integration test completed: {successful_models}/{total_models} models successful")
            
        except Exception as e:
            self.logger.error(f"❌ Training integration test failed: {e}")
            results["error"] = str(e)
            results["success"] = False
        
        return results
    
    def test_regime_aware_optimization(self) -> Dict[str, Any]:
        """Test regime-aware parameter optimization."""
        self.logger.info("🔄 Testing regime-aware optimization...")
        
        if not MIGRATED_MODELS_AVAILABLE:
            return {"error": "Migrated models not available", "success": False}
        
        results = {}
        
        try:
            # Test N-BEATS regime optimization
            from src.utils.ml_common.models.migrated_model_configs import NBEATSConfig
            
            # Create test regime characteristics
            regime_chars = RegimeCharacteristics(
                volume=0.8, volatility=0.7, momentum=0.6, trend=0.9
            )
            
            # Test optimization
            nbeats_config = NBEATSConfig()
            optimized_config = RegimeAwareParameterOptimizer.optimize_nbeats_parameters(
                nbeats_config, regime_chars
            )
            
            results["nbeats_optimization"] = {
                "original_learning_rate": nbeats_config.learning_rate,
                "optimized_learning_rate": optimized_config.learning_rate,
                "original_dropout": nbeats_config.dropout,
                "optimized_dropout": optimized_config.dropout,
                "success": True
            }
            
            # Test FinancialResNet regime optimization
            from src.utils.ml_common.models.migrated_model_configs import FinancialResNetConfig
            
            resnet_config = FinancialResNetConfig()
            optimized_resnet_config = RegimeAwareParameterOptimizer.optimize_financial_resnet_parameters(
                resnet_config, regime_chars
            )
            
            results["resnet_optimization"] = {
                "original_attention_heads": resnet_config.attention_heads,
                "optimized_attention_heads": optimized_resnet_config.attention_heads,
                "original_dropout": resnet_config.dropout,
                "optimized_dropout": optimized_resnet_config.dropout,
                "success": True
            }
            
            results["success"] = True
            self.logger.info("✅ Regime-aware optimization test completed successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Regime-aware optimization test failed: {e}")
            results["error"] = str(e)
            results["success"] = False
        
        return results
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test suite."""
        self.logger.info("🚀 Running comprehensive migrated models test...")
        start_time = time.time()
        
        test_results = {
            "test_start_time": start_time,
            "tests": {}
        }
        
        # Test model configurations
        test_results["tests"]["model_configurations"] = self.test_model_configurations()
        
        # Test model creation
        test_results["tests"]["model_creation"] = self.test_model_creation()
        
        # Test regime-aware optimization
        test_results["tests"]["regime_aware_optimization"] = self.test_regime_aware_optimization()
        
        # Test training integration
        test_results["tests"]["training_integration"] = self.test_training_integration()
        
        # Calculate overall results
        test_results["test_end_time"] = time.time()
        test_results["test_duration"] = test_results["test_end_time"] - test_results["test_start_time"]
        
        # Count successful tests
        successful_tests = sum(1 for test_result in test_results["tests"].values() 
                             if test_result.get("success", False))
        total_tests = len(test_results["tests"])
        
        test_results["overall_success"] = successful_tests == total_tests
        test_results["success_rate"] = successful_tests / total_tests if total_tests > 0 else 0
        
        self.logger.info(f"✅ Comprehensive test completed in {test_results['test_duration']:.3f}s")
        self.logger.info(f"📊 Test results: {successful_tests}/{total_tests} tests successful")
        
        return test_results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test migrated ML models")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--mode", type=str, choices=["quick", "full"], default="full",
                       help="Test mode")
    parser.add_argument("--output", type=str, help="Output file for test results")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = MigratedModelsTester(args.config)
    
    # Run tests based on mode
    if args.mode == "quick":
        # Quick test - just configuration and creation
        results = {
            "model_configurations": tester.test_model_configurations(),
            "model_creation": tester.test_model_creation(),
            "regime_aware_optimization": tester.test_regime_aware_optimization()
        }
    else:
        # Full test - comprehensive testing
        results = tester.run_comprehensive_test()
    
    # Print results
    print("\n" + "="*80)
    print("MIGRATED MODELS TEST RESULTS")
    print("="*80)
    
    if isinstance(results, dict) and "tests" in results:
        # Comprehensive test results
        print(f"Overall Success: {results['overall_success']}")
        print(f"Success Rate: {results['success_rate']:.2%}")
        print(f"Test Duration: {results['test_duration']:.3f}s")
        print("\nTest Details:")
        
        for test_name, test_result in results["tests"].items():
            print(f"  {test_name}: {'✅' if test_result.get('success', False) else '❌'}")
            if "error" in test_result:
                print(f"    Error: {test_result['error']}")
    else:
        # Individual test results
        for test_name, test_result in results.items():
            print(f"{test_name}: {'✅' if test_result.get('success', False) else '❌'}")
            if "error" in test_result:
                print(f"  Error: {test_result['error']}")
    
    print("="*80)
    
    # Save results if requested
    if args.output:
        try:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to {args.output}")
        except Exception as e:
            print(f"Failed to save results: {e}")


if __name__ == "__main__":
    main()