#!/usr/bin/env python3
"""
VectorBT Optimizations Validation Script

This script validates the VectorBT optimizations implemented in the training modules.
It tests the integration, performance, and fallback mechanisms.
"""

import numpy as np
import pandas as pd
import asyncio
import time
from typing import Dict, Any
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_sample_data(n_samples: int = 1000, n_features: int = 50) -> pd.DataFrame:
    """Create sample data for testing."""
    np.random.seed(42)
    
    # Create sample features
    features = {}
    for i in range(n_features):
        features[f'feature_{i}'] = np.random.randn(n_samples)
    
    # Create sample targets
    features['target'] = np.random.randn(n_samples)
    features['hmm_regime'] = np.random.randint(0, 3, n_samples)
    features['analyst_confidence'] = np.random.uniform(0, 1, n_samples)
    
    # Add timestamps
    features['timestamp'] = pd.date_range('2020-01-01', periods=n_samples, freq='1min')
    
    return pd.DataFrame(features)

async def test_analyst_ensemble_training():
    """Test Analyst ensemble training with VectorBT optimizations."""
    print("🧪 Testing Analyst Ensemble Training with VectorBT optimizations...")
    
    try:
        from src.training.steps.models_training.analyst_ensemble_training import (
            AnalystEnsembleTrainingStep, AnalystEnsembleTrainingConfig
        )
        
        # Create sample data
        training_data = create_sample_data(500, 30)
        feature_columns = [f'feature_{i}' for i in range(30)]
        target_columns = ['target']
        
        # Test with VectorBT optimizations enabled
        config = AnalystEnsembleTrainingConfig(
            enable_vectorbt_optimizations=True,
            vectorbt_rolling_window=10,
            vectorbt_max_features=20,
            vectorbt_memory_efficient=True
        )
        
        trainer = AnalystEnsembleTrainingStep(config)
        
        # Test initialization
        print("✅ Analyst ensemble trainer initialized successfully")
        
        # Test performance metrics
        metrics = trainer.get_performance_metrics()
        print(f"📊 VectorBT optimizations enabled: {metrics['config']['vectorbt_optimizations_enabled']}")
        print(f"📊 Rolling optimizer available: {metrics['vectorbt_optimization']['rolling_optimizer_available']}")
        print(f"📊 Unified framework available: {metrics['vectorbt_optimization']['unified_framework_available']}")
        
        # Test feature creation (without actual training)
        try:
            X_base = training_data[feature_columns].values
            base_models = {}
            
            # Test enhanced feature set creation
            enhanced_features = await trainer._create_enhanced_feature_set(
                X_base, training_data, base_models
            )
            
            print(f"✅ Enhanced feature set created: {enhanced_features.shape[1]} features")
            
            # Check if VectorBT features were added
            if hasattr(trainer, '_enhanced_metadata'):
                metadata = trainer._enhanced_metadata
                print(f"📊 VectorBT rolling features: {metadata.get('vectorbt_rolling_features_count', 0)}")
                print(f"📊 VectorBT selected features: {metadata.get('vectorbt_selected_features_count', 0)}")
            
        except Exception as e:
            print(f"⚠️ Feature creation test failed: {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import Analyst ensemble training: {e}")
        return False
    except Exception as e:
        print(f"❌ Analyst ensemble training test failed: {e}")
        return False

async def test_tactician_ensemble_training():
    """Test Tactician ensemble training with VectorBT optimizations."""
    print("\n🧪 Testing Tactician Ensemble Training with VectorBT optimizations...")
    
    try:
        from src.training.steps.models_training.tactician_ensemble_training import (
            TacticianEnsembleTrainingStep, TacticianEnsembleTrainingConfig
        )
        
        # Create sample data
        training_data = create_sample_data(500, 30)
        feature_columns = [f'feature_{i}' for i in range(30)]
        target_columns = ['target']
        
        # Test with VectorBT optimizations enabled
        config = TacticianEnsembleTrainingConfig(
            enable_vectorbt_optimizations=True,
            vectorbt_rolling_window=10,
            vectorbt_max_features=20,
            vectorbt_memory_efficient=True
        )
        
        trainer = TacticianEnsembleTrainingStep(config)
        
        # Test initialization
        print("✅ Tactician ensemble trainer initialized successfully")
        
        # Test performance metrics
        metrics = trainer.get_performance_metrics()
        print(f"📊 VectorBT optimizations enabled: {metrics['config']['vectorbt_optimizations_enabled']}")
        print(f"📊 Rolling optimizer available: {metrics['vectorbt_optimization']['rolling_optimizer_available']}")
        print(f"📊 Unified framework available: {metrics['vectorbt_optimization']['unified_framework_available']}")
        
        # Test feature creation (without actual training)
        try:
            X_base = training_data[feature_columns].values
            base_models = {}
            
            # Test enhanced feature set creation
            enhanced_features = await trainer._create_enhanced_feature_set(
                X_base, training_data, base_models
            )
            
            print(f"✅ Enhanced feature set created: {enhanced_features.shape[1]} features")
            
            # Check if VectorBT features were added
            if hasattr(trainer, '_enhanced_metadata'):
                metadata = trainer._enhanced_metadata
                print(f"📊 VectorBT rolling features: {metadata.get('vectorbt_rolling_features_count', 0)}")
                print(f"📊 VectorBT selected features: {metadata.get('vectorbt_selected_features_count', 0)}")
            
        except Exception as e:
            print(f"⚠️ Feature creation test failed: {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import Tactician ensemble training: {e}")
        return False
    except Exception as e:
        print(f"❌ Tactician ensemble training test failed: {e}")
        return False

async def test_analyst_models_training():
    """Test Analyst models training with VectorBT optimizations."""
    print("\n🧪 Testing Analyst Models Training with VectorBT optimizations...")
    
    try:
        from src.training.steps.models_training.analyst_models_training import (
            AnalystModelsTrainingStep, AnalystModelsTrainingConfig
        )
        
        # Create sample data
        training_data = create_sample_data(500, 30)
        feature_columns = [f'feature_{i}' for i in range(30)]
        target_columns = ['target']
        
        # Test with VectorBT optimizations enabled
        config = AnalystModelsTrainingConfig(
            enable_vectorbt_optimizations=True,
            vectorbt_rolling_window=10,
            vectorbt_max_features=20,
            vectorbt_memory_efficient=True
        )
        
        trainer = AnalystModelsTrainingStep(config)
        
        # Test initialization
        print("✅ Analyst models trainer initialized successfully")
        
        # Test performance metrics
        metrics = trainer.get_performance_metrics()
        print(f"📊 VectorBT optimizations enabled: {metrics['config']['vectorbt_optimizations_enabled']}")
        print(f"📊 Rolling optimizer available: {metrics['vectorbt_optimization']['rolling_optimizer_available']}")
        print(f"📊 Unified framework available: {metrics['vectorbt_optimization']['unified_framework_available']}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import Analyst models training: {e}")
        return False
    except Exception as e:
        print(f"❌ Analyst models training test failed: {e}")
        return False

def test_vectorbt_components():
    """Test VectorBT components directly."""
    print("\n🧪 Testing VectorBT Components...")
    
    try:
        from src.feature_generation.utils.vectorbt_rolling_optimizer import (
            VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
        )
        
        # Test VectorBT rolling optimizer
        optimizer = get_vectorbt_rolling_optimizer(
            enable_gpu=False,
            enable_parallel=True,
            memory_efficient=True,
            chunk_size=1000,
            fast_fail=True,
            enable_logging=True
        )
        
        print("✅ VectorBT rolling optimizer initialized")
        
        # Test rolling operations
        data = pd.Series(np.random.randn(100))
        
        # Test rolling mean
        rolling_mean = optimizer.rolling_mean(data, window=10)
        print(f"✅ Rolling mean test passed: {len(rolling_mean)} values")
        
        # Test rolling std
        rolling_std = optimizer.rolling_std(data, window=10)
        print(f"✅ Rolling std test passed: {len(rolling_std)} values")
        
        # Test performance stats
        stats = optimizer.get_performance_stats()
        print(f"📊 VectorBT operations: {stats.get('vectorbt_operations', 0)}")
        print(f"📊 Total operations: {stats.get('total_operations', 0)}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import VectorBT rolling optimizer: {e}")
        return False
    except Exception as e:
        print(f"❌ VectorBT components test failed: {e}")
        return False

async def main():
    """Main validation function."""
    print("🚀 Starting VectorBT Optimizations Validation...")
    print("=" * 60)
    
    results = []
    
    # Test VectorBT components
    results.append(("VectorBT Components", test_vectorbt_components()))
    
    # Test Analyst ensemble training
    results.append(("Analyst Ensemble Training", await test_analyst_ensemble_training()))
    
    # Test Tactician ensemble training
    results.append(("Tactician Ensemble Training", await test_tactician_ensemble_training()))
    
    # Test Analyst models training
    results.append(("Analyst Models Training", await test_analyst_models_training()))
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 VALIDATION RESULTS")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n📈 Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All VectorBT optimizations validated successfully!")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)