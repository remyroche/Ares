"""
OptimizedTrainer Examples and Usage Guide

This file demonstrates comprehensive usage of the OptimizedTrainer class
with various machine learning scenarios and Apple Silicon optimization.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Import the OptimizedTrainer from the consolidated location
from src.utils.nas_tas.hardware_accelerator import (
    NASHardwareAccelerator, TASHardwareAccelerator, CLVSAHardwareOptimizer,
    HardwareAccelerationConfig, create_nas_hardware_accelerator,
    create_tas_hardware_accelerator, create_cvlsa_hardware_optimizer
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_1_basic_training():
    """Example 1: Basic training with M1 optimization."""
    print("\n" + "="*60)
    print("Example 1: Basic Training with M1 Optimization")
    print("="*60)
    
    try:
        # Create configuration
        config = TrainingConfig(
            max_epochs=20,
            batch_size=32,
            learning_rate=0.001,
            patience=5,
            enable_gpu=True,
            enable_memory_optimization=True,
            enable_parallel=True,
            output_dir="examples/basic_training"
        )
        
        # Create trainer
        trainer = OptimizedTrainer(config)
        
        # Generate sample data
        np.random.seed(42)
        X = np.random.randn(1000, 20)
        y = np.random.randint(0, 2, 1000)
        
        print(f"📊 Generated data: {X.shape}, {y.shape}")
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(X, y)
        
        # Setup a simple model (placeholder for actual model)
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Train model
        results = trainer.train(X_train, y_train, X_val, y_val)
        
        print(f"✅ Training completed!")
        print(f"📊 Best epoch: {results['best_epoch']}")
        print(f"📊 Best metric: {results['best_metric']:.4f}")
        print(f"📊 Total time: {results['total_time_s']:.2f}s")
        
        # Get performance report
        report = trainer.get_performance_report()
        print(f"📊 Performance report generated")
        
        # Cleanup
        trainer.cleanup()
        
        return results
        
    except Exception as e:
        print(f"❌ Example 1 failed: {e}")
        return None

def example_2_hyperparameter_optimization():
    """Example 2: Hyperparameter optimization with different methods."""
    print("\n" + "="*60)
    print("Example 2: Hyperparameter Optimization")
    print("="*60)
    
    try:
        # Create configuration
        config = TrainingConfig(
            enable_gpu=True,
            enable_memory_optimization=True,
            enable_parallel=True,
            output_dir="examples/hyperparameter_opt"
        )
        
        trainer = OptimizedTrainer(config)
        
        # Generate sample data
        np.random.seed(42)
        X = np.random.randn(800, 15)
        y = np.random.randint(0, 2, 800)
        
        print(f"📊 Generated data: {X.shape}, {y.shape}")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Grid Search
        print("\n🔍 Grid Search Optimization:")
        grid_results = trainer.hyperparameter_optimization(
            X, y, param_grid, method='grid', cv_folds=3
        )
        print(f"✅ Best parameters: {grid_results['best_params']}")
        print(f"✅ Best score: {grid_results['best_score']:.4f}")
        
        # Random Search
        print("\n🎲 Random Search Optimization:")
        random_results = trainer.hyperparameter_optimization(
            X, y, param_grid, method='random', cv_folds=3, n_trials=20
        )
        print(f"✅ Best parameters: {random_results['best_params']}")
        print(f"✅ Best score: {random_results['best_score']:.4f}")
        
        # Bayesian Optimization (if Optuna is available)
        try:
            print("\n🧠 Bayesian Optimization:")
            bayesian_results = trainer.hyperparameter_optimization(
                X, y, param_grid, method='bayesian', n_trials=30, timeout=300
            )
            print(f"✅ Best parameters: {bayesian_results['best_params']}")
            print(f"✅ Best score: {bayesian_results['best_score']:.4f}")
        except Exception as e:
            print(f"⚠️ Bayesian optimization not available: {e}")
        
        trainer.cleanup()
        
        return {
            'grid': grid_results,
            'random': random_results,
            'bayesian': bayesian_results if 'bayesian_results' in locals() else None
        }
        
    except Exception as e:
        print(f"❌ Example 2 failed: {e}")
        return None

def example_3_cross_validation():
    """Example 3: Advanced cross-validation techniques."""
    print("\n" + "="*60)
    print("Example 3: Cross-Validation Techniques")
    print("="*60)
    
    try:
        config = TrainingConfig(
            enable_gpu=True,
            enable_memory_optimization=True,
            output_dir="examples/cross_validation"
        )
        
        trainer = OptimizedTrainer(config)
        
        # Generate sample data
        np.random.seed(42)
        X = np.random.randn(1200, 25)
        y = np.random.randint(0, 3, 1200)  # Multi-class
        
        print(f"📊 Generated data: {X.shape}, {y.shape}")
        
        # Standard Cross-Validation
        print("\n🔄 Standard Cross-Validation:")
        cv_results = trainer.cross_validate(X, y, cv_folds=5, scoring='accuracy')
        print(f"✅ CV Score: {cv_results['mean_test_score']:.4f} ± {cv_results['std_test_score']:.4f}")
        print(f"✅ Train Score: {cv_results['mean_train_score']:.4f} ± {cv_results['std_train_score']:.4f}")
        
        # Time Series Cross-Validation (Lookahead)
        print("\n👀 Lookahead Validation:")
        lookahead_results = trainer.lookahead_validation(X, y, lookahead_steps=10)
        print(f"✅ Lookahead Score: {lookahead_results['mean_score']:.4f} ± {lookahead_results['std_score']:.4f}")
        
        trainer.cleanup()
        
        return {
            'standard_cv': cv_results,
            'lookahead': lookahead_results
        }
        
    except Exception as e:
        print(f"❌ Example 3 failed: {e}")
        return None

def example_4_memory_optimization():
    """Example 4: Memory optimization with large datasets."""
    print("\n" + "="*60)
    print("Example 4: Memory Optimization with Large Datasets")
    print("="*60)
    
    try:
        # Create configuration with memory limits
        config = TrainingConfig(
            memory_limit_gb=2.0,  # Limit memory usage
            chunk_size_mb=128,    # Smaller chunks for memory efficiency
            max_memory_percent=0.6,  # Conservative memory usage
            enable_memory_optimization=True,
            enable_gpu=True,
            output_dir="examples/memory_optimization"
        )
        
        trainer = OptimizedTrainer(config)
        
        # Generate large dataset
        print("📊 Generating large dataset...")
        np.random.seed(42)
        X = np.random.randn(5000, 50)  # Larger dataset
        y = np.random.randint(0, 2, 5000)
        
        print(f"📊 Generated large data: {X.shape}, {y.shape}")
        print(f"📊 Memory usage: {X.nbytes / (1024**2):.1f} MB")
        
        # Prepare data with memory optimization
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(X, y)
        
        # Perform operations that demonstrate memory optimization
        print("\n🔄 Performing memory-intensive operations...")
        
        # Cross-validation with memory monitoring
        cv_results = trainer.cross_validate(X_train, y_train, cv_folds=3)
        print(f"✅ CV completed with memory optimization")
        
        # Hyperparameter optimization with memory constraints
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'min_samples_split': [2, 5]
        }
        
        opt_results = trainer.hyperparameter_optimization(
            X_train, y_train, param_grid, method='random', n_trials=10
        )
        print(f"✅ Hyperparameter optimization completed with memory optimization")
        
        # Get memory statistics
        if trainer.memory_optimizer:
            memory_stats = trainer.memory_optimizer.get_memory_stats()
            print(f"📊 Memory stats: {memory_stats}")
        
        trainer.cleanup()
        
        return {
            'cv_results': cv_results,
            'opt_results': opt_results,
            'memory_stats': memory_stats if 'memory_stats' in locals() else None
        }
        
    except Exception as e:
        print(f"❌ Example 4 failed: {e}")
        return None

def example_5_pytorch_integration():
    """Example 5: PyTorch model integration (if available)."""
    print("\n" + "="*60)
    print("Example 5: PyTorch Integration")
    print("="*60)
    
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        # Simple neural network
        class SimpleNN(nn.Module):
            def __init__(self, input_size=20, hidden_size=64, output_size=2):
                super(SimpleNN, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.fc3 = nn.Linear(hidden_size, output_size)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.2)
                
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return x
        
        config = TrainingConfig(
            max_epochs=10,
            batch_size=64,
            learning_rate=0.001,
            enable_gpu=True,
            enable_memory_optimization=True,
            output_dir="examples/pytorch_integration"
        )
        
        trainer = OptimizedTrainer(config)
        
        # Generate data
        np.random.seed(42)
        X = np.random.randn(1000, 20).astype(np.float32)
        y = np.random.randint(0, 2, 1000)
        
        print(f"📊 Generated data: {X.shape}, {y.shape}")
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(X, y)
        
        # Create model
        model = SimpleNN(input_size=20, hidden_size=64, output_size=2)
        
        # Setup model with trainer
        trainer.setup_model(
            model=model,
            optimizer_class=optim.Adam,
            scheduler_class=optim.lr_scheduler.StepLR,
            lr=0.001,
            step_size=5,
            gamma=0.1
        )
        
        print("✅ PyTorch model setup completed")
        print(f"✅ Model device: {'MPS' if trainer.gpu_manager and trainer.gpu_manager.mps_available else 'CPU'}")
        
        # Train model (this would contain actual PyTorch training loop in real implementation)
        print("🚀 Starting PyTorch training...")
        
        # For demo purposes, we'll simulate training
        results = trainer.train(X_train, y_train, X_val, y_val)
        
        print(f"✅ PyTorch training completed!")
        print(f"📊 Training results: {results}")
        
        # Save model
        trainer.save_model("examples/pytorch_model.pth", format="torch")
        
        trainer.cleanup()
        
        return results
        
    except ImportError:
        print("⚠️ PyTorch not available, skipping PyTorch integration example")
        return None
    except Exception as e:
        print(f"❌ Example 5 failed: {e}")
        return None

def example_6_comprehensive_workflow():
    """Example 6: Comprehensive ML workflow."""
    print("\n" + "="*60)
    print("Example 6: Comprehensive ML Workflow")
    print("="*60)
    
    try:
        # Create comprehensive configuration
        config = TrainingConfig(
            max_epochs=50,
            batch_size=32,
            learning_rate=0.001,
            patience=10,
            enable_gpu=True,
            enable_memory_optimization=True,
            enable_parallel=True,
            enable_hyperparameter_optimization=True,
            optimization_trials=20,
            enable_cross_validation=True,
            cv_folds=5,
            enable_lookahead_validation=True,
            lookahead_steps=5,
            enable_monitoring=True,
            log_interval=5,
            checkpoint_interval=10,
            output_dir="examples/comprehensive_workflow"
        )
        
        # Create trainer
        trainer = OptimizedTrainer(config)
        
        # Generate complex dataset
        np.random.seed(42)
        n_samples, n_features = 2000, 30
        
        # Create features with different scales and types
        X = np.random.randn(n_samples, n_features)
        
        # Add some categorical-like features
        X[:, 0] = np.random.randint(0, 5, n_samples)  # Categorical
        X[:, 1] = np.random.choice([0, 1], n_samples)  # Binary
        
        # Create target with some complexity
        y = (X[:, 0] + X[:, 1] + np.sum(X[:, 2:5], axis=1) + 
             np.random.randn(n_samples) * 0.1 > 0).astype(int)
        
        print(f"📊 Generated complex data: {X.shape}, {y.shape}")
        print(f"📊 Target distribution: {np.bincount(y)}")
        
        # Step 1: Data preparation
        print("\n📊 Step 1: Data Preparation")
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(X, y)
        
        # Step 2: Hyperparameter optimization
        print("\n🔍 Step 2: Hyperparameter Optimization")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        opt_results = trainer.hyperparameter_optimization(
            X_train, y_train, param_grid, method='random', n_trials=15
        )
        print(f"✅ Best parameters: {opt_results['best_params']}")
        print(f"✅ Best score: {opt_results['best_score']:.4f}")
        
        # Step 3: Cross-validation with best parameters
        print("\n🔄 Step 3: Cross-Validation")
        cv_results = trainer.cross_validate(X_train, y_train, cv_folds=5)
        print(f"✅ CV Score: {cv_results['mean_test_score']:.4f} ± {cv_results['std_test_score']:.4f}")
        
        # Step 4: Lookahead validation
        print("\n👀 Step 4: Lookahead Validation")
        lookahead_results = trainer.lookahead_validation(X_train, y_train, lookahead_steps=5)
        print(f"✅ Lookahead Score: {lookahead_results['mean_score']:.4f} ± {lookahead_results['std_score']:.4f}")
        
        # Step 5: Final training with best parameters
        print("\n🚀 Step 5: Final Training")
        from sklearn.ensemble import RandomForestClassifier
        
        # Create model with best parameters
        best_model = RandomForestClassifier(**opt_results['best_params'], random_state=42)
        trainer.setup_model(best_model)
        
        # Train final model
        final_results = trainer.train(X_train, y_train, X_val, y_val)
        
        # Step 6: Model evaluation and saving
        print("\n📊 Step 6: Model Evaluation and Saving")
        
        # Evaluate on test set
        best_model.fit(X_train, y_train)
        test_predictions = best_model.predict(X_test)
        test_accuracy = np.mean(test_predictions == y_test)
        
        print(f"✅ Test Accuracy: {test_accuracy:.4f}")
        
        # Save model and results
        trainer.save_model("examples/comprehensive_model.pkl")
        
        # Save comprehensive results
        comprehensive_results = {
            'hyperparameter_optimization': opt_results,
            'cross_validation': cv_results,
            'lookahead_validation': lookahead_results,
            'final_training': final_results,
            'test_accuracy': test_accuracy,
            'performance_report': trainer.get_performance_report()
        }
        
        # Save results to JSON
        import json
        with open("examples/comprehensive_results.json", "w") as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        print("✅ Comprehensive workflow completed!")
        print(f"📊 Results saved to: examples/comprehensive_results.json")
        
        trainer.cleanup()
        
        return comprehensive_results
        
    except Exception as e:
        print(f"❌ Example 6 failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_all_examples():
    """Run all examples."""
    print("🚀 OptimizedTrainer - Comprehensive Examples")
    print("=" * 80)
    
    results = {}
    
    # Run examples
    results['example_1'] = example_1_basic_training()
    results['example_2'] = example_2_hyperparameter_optimization()
    results['example_3'] = example_3_cross_validation()
    results['example_4'] = example_4_memory_optimization()
    results['example_5'] = example_5_pytorch_integration()
    results['example_6'] = example_6_comprehensive_workflow()
    
    # Summary
    print("\n" + "="*80)
    print("📊 EXAMPLES SUMMARY")
    print("="*80)
    
    for example_name, result in results.items():
        status = "✅ SUCCESS" if result is not None else "❌ FAILED"
        print(f"{example_name}: {status}")
    
    successful_examples = sum(1 for r in results.values() if r is not None)
    total_examples = len(results)
    
    print(f"\n📊 Overall: {successful_examples}/{total_examples} examples completed successfully")
    
    return results

if __name__ == "__main__":
    # Create examples directory
    Path("examples").mkdir(exist_ok=True)
    
    # Run all examples
    results = run_all_examples()
    
    print("\n🎉 All examples completed!")
    print("Check the 'examples/' directory for outputs and results.")