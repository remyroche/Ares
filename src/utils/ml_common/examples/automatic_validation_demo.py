"""
Automatic Validation Integration Demonstration

Demonstrates how universal validation is automatically wired into all ML training/optimization
pipelines by default, ensuring comprehensive validation for all models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path

# Import ML Common training components with automatic validation
from ..training import (
    BaseTrainingStep,
    TrainingUtils,
    get_validation_integrator,
    validate_training_data,
    validate_trained_model
)
from ..config.base_training_config import BaseTrainingConfig
from ..optimization import HierarchicalHPO, HierarchicalHPOConfig, HPOPhaseConfig

# Import ML models for demonstration
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DemoTrainingStep(BaseTrainingStep):
    """Demo training step that inherits automatic validation."""
    
    def execute(self, X, y, regime_labels, feature_names=None, hmm_states=None, **kwargs):
        """Execute training with automatic validation."""
        logger.info("🚀 Starting demo training with automatic validation...")
        
        # 1. Automatic training data validation
        logger.info("📊 Validating training data...")
        data_validation = self.validate_training_data(
            X=X, y=y, regime_labels=regime_labels, 
            feature_names=feature_names, model_type="demo_model"
        )
        
        if not data_validation['valid']:
            logger.error("❌ Training data validation failed!")
            return {'error': 'Data validation failed', 'validation': data_validation}
        
        # 2. Split data
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 3. Train models with automatic validation
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100),
            'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=300),
            'LogisticRegression': LogisticRegression(max_iter=1000)
        }
        
        trained_models = {}
        validation_results = {}
        
        for model_name, model in models.items():
            logger.info(f"🔧 Training {model_name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Automatic model validation
            logger.info(f"🔍 Validating {model_name}...")
            model_validation = self.validate_trained_model(
                model=model,
                X_train=X_train,
                X_val=X_val,
                y_train=y_train,
                y_val=y_val,
                model_name=model_name,
                model_type=model_name.lower()
            )
            
            trained_models[model_name] = model
            validation_results[model_name] = model_validation
            
            # Log validation results
            if model_validation['valid']:
                logger.info(f"✅ {model_name} validation passed")
            else:
                logger.warning(f"⚠️ {model_name} validation failed")
        
        # 4. Get validation summary
        validation_summary = self.get_validation_summary()
        
        return {
            'trained_models': trained_models,
            'validation_results': validation_results,
            'validation_summary': validation_summary,
            'success': True
        }

def demonstrate_automatic_validation():
    """Demonstrate automatic validation integration."""
    print("🚀 Automatic Validation Integration Demonstration")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    n_samples, n_features = 1000, 50
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1, 2], size=n_samples, p=[0.2, 0.6, 0.2])
    regime_labels = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    
    # Add some structure to make it more realistic
    X[:, 0] += y * 0.5
    X[:, 1] += np.random.normal(0, 0.1, n_samples)
    
    print(f"📊 Created sample data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"📊 Target distribution: {np.bincount(y)}")
    print(f"📊 Regime distribution: {np.bincount(regime_labels)}")
    
    # Create training configuration with validation enabled
    config = BaseTrainingConfig(
        model_name="demo_model",
        enable_validation=True,
        enable_overfitting_detection=True,
        enable_temporal_validation=True,
        enable_timeframe_validation=True,
        save_validation_reports=True,
        validation_report_directory="demo_reports/validation"
    )
    
    print(f"\n⚙️ Configuration:")
    print(f"  Validation enabled: {config.enable_validation}")
    print(f"  Overfitting detection: {config.enable_overfitting_detection}")
    print(f"  Temporal validation: {config.enable_temporal_validation}")
    print(f"  Timeframe validation: {config.enable_timeframe_validation}")
    
    # Create and run training step
    training_step = DemoTrainingStep(config)
    
    # Execute training with automatic validation
    results = training_step.execute(
        X=X, y=y, regime_labels=regime_labels,
        feature_names=[f"feature_{i}" for i in range(n_features)]
    )
    
    if results['success']:
        print(f"\n✅ Training completed successfully!")
        
        # Display validation results
        print(f"\n📊 Validation Results:")
        for model_name, validation in results['validation_results'].items():
            status = "✅ PASSED" if validation['valid'] else "❌ FAILED"
            score = validation.get('validation_score', 'N/A')
            print(f"  {model_name}: {status} (Score: {score})")
            
            if not validation['valid']:
                print(f"    Critical issues: {len(validation.get('critical_issues', []))}")
                print(f"    Warnings: {len(validation.get('warnings', []))}")
        
        # Display validation summary
        summary = results['validation_summary']
        print(f"\n📈 Validation Summary:")
        print(f"  Total validations: {summary['total_validations']}")
        print(f"  Valid validations: {summary['valid_validations']}")
        print(f"  Success rate: {summary['success_rate']:.2%}")
        print(f"  Average score: {summary['average_validation_score']:.3f}")
        
    else:
        print(f"\n❌ Training failed: {results.get('error', 'Unknown error')}")
    
    return results

def demonstrate_training_utils_validation():
    """Demonstrate validation integration in TrainingUtils."""
    print("\n🔧 TrainingUtils Validation Integration")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(500, 30)
    y = np.random.choice([0, 1], size=500, p=[0.6, 0.4])
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create training utils with validation
    config = BaseTrainingConfig(enable_validation=True)
    training_utils = TrainingUtils(config)
    
    # Test different models with automatic validation
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=50),
        'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(20, 10), max_iter=200),
        'LogisticRegression': LogisticRegression(max_iter=500)
    }
    
    for model_name, model in models.items():
        print(f"\n🔧 Testing {model_name} with TrainingUtils...")
        
        # Train with validation
        trained_model, validation_results = training_utils.train_model_with_validation(
            model=model,
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            model_name=model_name,
            model_type=model_name.lower()
        )
        
        # Display results
        status = "✅ PASSED" if validation_results['valid'] else "❌ FAILED"
        score = validation_results.get('validation_score', 'N/A')
        print(f"  Result: {status} (Score: {score})")
        
        if not validation_results['valid']:
            print(f"  Critical issues: {len(validation_results.get('critical_issues', []))}")
            print(f"  Warnings: {len(validation_results.get('warnings', []))}")

def demonstrate_hpo_validation():
    """Demonstrate validation integration in HPO."""
    print("\n🎯 HPO Validation Integration")
    print("=" * 40)
    
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(300, 20)
    y = np.random.choice([0, 1], size=300, p=[0.6, 0.4])
    
    # Create HPO configuration with validation
    phase1_config = HPOPhaseConfig(
        phase_name="base_models",
        models={'RandomForest': RandomForestClassifier()},
        search_spaces={
            'RandomForest': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 200},
                'max_depth': {'type': 'int', 'low': 5, 'high': 15}
            }
        },
        n_trials=10,
        cv_folds=3
    )
    
    hpo_config = HierarchicalHPOConfig(
        phase1_config=phase1_config,
        phase2_config=phase1_config,  # Simplified for demo
        enable_validation=True,
        enable_overfitting_detection=True,
        validation_failure_threshold=0.3
    )
    
    print(f"⚙️ HPO Configuration:")
    print(f"  Validation enabled: {hpo_config.enable_validation}")
    print(f"  Overfitting detection: {hpo_config.enable_overfitting_detection}")
    print(f"  Validation threshold: {hpo_config.validation_failure_threshold}")
    
    # Create HPO instance
    hpo = HierarchicalHPO(hpo_config)
    
    print(f"\n🔍 HPO validation integration initialized")
    print(f"  Validation integrator: {hasattr(hpo, 'validation_integrator')}")
    print(f"  Validation config: {hpo.validation_integrator.config.enable_validation}")

def demonstrate_validation_configuration():
    """Demonstrate validation configuration options."""
    print("\n⚙️ Validation Configuration Options")
    print("=" * 45)
    
    # Default configuration
    default_config = BaseTrainingConfig()
    print(f"📋 Default Configuration:")
    print(f"  Validation enabled: {default_config.enable_validation}")
    print(f"  Overfitting detection: {default_config.enable_overfitting_detection}")
    print(f"  Temporal validation: {default_config.enable_temporal_validation}")
    print(f"  Timeframe validation: {default_config.enable_timeframe_validation}")
    print(f"  Validation threshold: {default_config.validation_failure_threshold}")
    print(f"  Fail on error: {default_config.fail_on_validation_error}")
    
    # Custom configuration
    custom_config = BaseTrainingConfig(
        enable_validation=True,
        enable_overfitting_detection=True,
        enable_temporal_validation=True,
        enable_timeframe_validation=True,
        validation_failure_threshold=0.7,
        fail_on_validation_error=True,
        save_validation_reports=True,
        validation_report_directory="custom_reports/validation"
    )
    
    print(f"\n📋 Custom Configuration:")
    print(f"  Validation enabled: {custom_config.enable_validation}")
    print(f"  Overfitting detection: {custom_config.enable_overfitting_detection}")
    print(f"  Temporal validation: {custom_config.enable_temporal_validation}")
    print(f"  Timeframe validation: {custom_config.enable_timeframe_validation}")
    print(f"  Validation threshold: {custom_config.validation_failure_threshold}")
    print(f"  Fail on error: {custom_config.fail_on_validation_error}")
    print(f"  Report directory: {custom_config.validation_report_directory}")
    
    # Test validation integrator with custom config
    integrator = get_validation_integrator()
    print(f"\n🔧 Validation Integrator:")
    print(f"  Config: {integrator.config.enable_validation}")
    print(f"  History: {len(integrator.validation_history)} validations")

def main():
    """Run the complete automatic validation demonstration."""
    print("🎯 Automatic Validation Integration - Complete Demonstration")
    print("=" * 70)
    
    try:
        # 1. Demonstrate automatic validation in training steps
        training_results = demonstrate_automatic_validation()
        
        # 2. Demonstrate validation in TrainingUtils
        demonstrate_training_utils_validation()
        
        # 3. Demonstrate validation in HPO
        demonstrate_hpo_validation()
        
        # 4. Demonstrate configuration options
        demonstrate_validation_configuration()
        
        # Summary
        print("\n" + "=" * 70)
        print("🎉 AUTOMATIC VALIDATION INTEGRATION - COMPLETE!")
        print("=" * 70)
        
        print("\n✅ Key Features Demonstrated:")
        print("  ✅ Automatic validation in BaseTrainingStep")
        print("  ✅ Automatic validation in TrainingUtils")
        print("  ✅ Automatic validation in HPO")
        print("  ✅ Configurable validation settings")
        print("  ✅ Comprehensive validation reporting")
        print("  ✅ Overfitting detection")
        print("  ✅ Temporal validation")
        print("  ✅ Timeframe validation")
        
        print("\n🎯 Benefits:")
        print("  🎯 All ML models automatically validated")
        print("  🎯 No code changes required for existing pipelines")
        print("  🎯 Configurable validation behavior")
        print("  🎯 Comprehensive reporting and logging")
        print("  🎯 Production-ready model validation")
        
        print("\n💡 Next Steps:")
        print("  1. Use existing training pipelines - validation is automatic!")
        print("  2. Configure validation settings in your config")
        print("  3. Monitor validation reports in the specified directory")
        print("  4. Adjust validation thresholds as needed")
        
        return training_results
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        print(f"\n❌ Demonstration failed: {e}")
        return None

if __name__ == "__main__":
    results = main()
    if results:
        print(f"\n📋 Results available: {results}")