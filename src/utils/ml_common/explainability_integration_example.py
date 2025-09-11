#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Example: Model Explainability Integration with ML Commons

This example demonstrates how the new model-focused explainability system
integrates with ML commons training and model registry.

Key Features Demonstrated:
- Automatic explainability during model training
- Model registry integration with explanations
- Model-specific explainers (not component-specific)
- Explanation caching and retrieval
- Integration with existing ML commons utilities
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import logging

# Import ML commons utilities
from .model_training import EnhancedModelTrainer
from .model_registry import ModelRegistry
from .model_explainability import ModelExplainabilityManager, explain_model_quick
from .base_safeguards import MLTrainingSafeguards

# Optional sklearn imports for demonstration
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    tprint("⚠️ Scikit-learn not available - using mock data for demonstration")


def create_sample_data(n_samples: int = 1000, n_features: int = 20) -> tuple:
    """Create sample data for demonstration."""
    if SKLEARN_AVAILABLE:
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=15,
            n_redundant=5,
            n_classes=2,
            random_state=42
        )
        feature_names = [f"feature_{i}" for i in range(n_features)]
        return X, y, feature_names
    else:
        # Mock data for demonstration
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        feature_names = [f"feature_{i}" for i in range(n_features)]
        return X, y, feature_names


def demonstrate_automatic_explainability():
    """Demonstrate automatic explainability integration during training."""
    tprint("🚀 Demonstrating Automatic Explainability Integration")
    tprint("=" * 60)
    
    # Create sample data
    X, y, feature_names = create_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    tprint(f"📊 Data shapes - Train: {X_train.shape}, Test: {X_test.shape}")
    tprint(f"📊 Features: {len(feature_names)}")
    
    # Initialize enhanced model trainer with explainability enabled
    config = {
        'enable_model_explanations': True,
        'explainability': {
            'enable_auto_explanations': True,
            'enable_explanation_caching': True,
            'auto_explain_on_training': True
        }
    }
    
    trainer = EnhancedModelTrainer(config)
    
    if SKLEARN_AVAILABLE:
        # Train a model with automatic explainability
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        tprint("\n🔄 Training model with automatic explainability...")
        results = trainer.train_and_evaluate_model(
            model=model,
            model_name="demo_random_forest",
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names
        )
        
        if results['success']:
            tprint("✅ Model training completed successfully!")
            tprint(f"📊 Accuracy: {results['basic_metrics'].get('accuracy', 'N/A'):.3f}")
            
            # Check if explanations were generated
            if 'model_explanations' in results and 'error' not in results['model_explanations']:
                explanations = results['model_explanations']
                tprint(f"🧠 Model explanations generated:")
                tprint(f"   - Explanation confidence: {explanations.get('explanation_confidence', 0):.3f}")
                tprint(f"   - Processing time: {explanations.get('processing_time_ms', 0):.1f}ms")
                tprint(f"   - Feature importance available: {'feature_importance' in explanations}")
                tprint(f"   - SHAP values available: {explanations.get('shap_values') is not None}")
                tprint(f"   - LIME explanation available: {explanations.get('lime_explanation') is not None}")
            else:
                tprint("⚠️ Model explanations not generated or failed")
        else:
            tprint(f"❌ Model training failed: {results.get('error', 'Unknown error')}")
    else:
        tprint("⚠️ Skipping model training - scikit-learn not available")


def demonstrate_model_registry_integration():
    """Demonstrate model registry integration with explanations."""
    tprint("\n🚀 Demonstrating Model Registry Integration")
    tprint("=" * 60)
    
    # Initialize model registry
    registry = ModelRegistry(registry_path="./demo_model_registry")
    
    # Initialize explainability manager with registry integration
    explainability_manager = ModelExplainabilityManager(
        config={'enable_explanation_caching': True},
        model_registry=registry
    )
    
    if SKLEARN_AVAILABLE:
        # Create and train a simple model
        X, y, feature_names = create_sample_data(n_samples=500, n_features=10)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Generate explanations
        tprint("🔄 Generating model explanations...")
        explanation = explainability_manager.explain_model(
            model=model,
            X_train=X_train,
            X_test=X_test,
            model_id="demo_registry_model",
            model_type="RandomForestClassifier",
            feature_names=feature_names
        )
        
        tprint(f"✅ Explanation generated:")
        tprint(f"   - Model ID: {explanation.model_id}")
        tprint(f"   - Model type: {explanation.model_type}")
        tprint(f"   - Explanation confidence: {explanation.explanation_confidence:.3f}")
        tprint(f"   - Processing time: {explanation.processing_time_ms:.1f}ms")
        
        # Save model with metadata to registry
        metadata = {
            'model_type': 'RandomForestClassifier',
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_count': len(feature_names),
            'accuracy': model.score(X_test, y_test),
            'explanation_available': True
        }
        
        tprint("\n💾 Saving model to registry...")
        save_result = registry.save_model_with_metadata(
            model=model,
            metadata=metadata,
            model_name="demo_registry_model"
        )
        
        if save_result['success']:
            tprint(f"✅ Model saved successfully: {save_result['model_id']}")
            
            # Load model back with explanations
            tprint("\n📂 Loading model with explanations...")
            load_result = registry.load_model_with_validation(
                model_id=save_result['model_id'],
                version='latest'
            )
            
            if load_result['success']:
                tprint("✅ Model loaded successfully!")
                if 'explanation' in load_result:
                    explanation_data = load_result['explanation']
                    tprint(f"🧠 Explanation loaded:")
                    tprint(f"   - Model ID: {explanation_data.get('model_id', 'N/A')}")
                    tprint(f"   - Explanation confidence: {explanation_data.get('explanation_confidence', 0):.3f}")
                else:
                    tprint("⚠️ No explanation found in loaded model")
            else:
                tprint(f"❌ Model loading failed: {load_result.get('error', 'Unknown error')}")
        else:
            tprint(f"❌ Model saving failed: {save_result.get('error', 'Unknown error')}")
    else:
        tprint("⚠️ Skipping registry demonstration - scikit-learn not available")


def demonstrate_quick_explanations():
    """Demonstrate quick explanation generation."""
    tprint("\n🚀 Demonstrating Quick Explanation Generation")
    tprint("=" * 60)
    
    if SKLEARN_AVAILABLE:
        # Create sample data
        X, y, feature_names = create_sample_data(n_samples=300, n_features=8)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train a simple model
        model = RandomForestClassifier(n_estimators=30, random_state=42)
        model.fit(X_train, y_train)
        
        tprint("🔄 Generating quick explanation...")
        
        # Use the convenience function for quick explanations
        explanation = explain_model_quick(
            model=model,
            X_train=X_train,
            X_test=X_test,
            model_id="quick_demo_model"
        )
        
        tprint(f"✅ Quick explanation generated:")
        tprint(f"   - Model ID: {explanation.model_id}")
        tprint(f"   - Model type: {explanation.model_type}")
        tprint(f"   - Explanation confidence: {explanation.explanation_confidence:.3f}")
        tprint(f"   - Processing time: {explanation.processing_time_ms:.1f}ms")
        tprint(f"   - Features explained: {len(explanation.feature_names)}")
        
        # Show cache statistics
        manager = ModelExplainabilityManager()
        cache_stats = manager.get_cache_stats()
        tprint(f"\n📊 Cache statistics:")
        tprint(f"   - Cache size: {cache_stats['cache_size']}")
        tprint(f"   - Cache hits: {cache_stats['cache_hits']}")
        tprint(f"   - Cache misses: {cache_stats['cache_misses']}")
        tprint(f"   - Hit rate: {cache_stats['hit_rate']:.3f}")
    else:
        tprint("⚠️ Skipping quick explanation demonstration - scikit-learn not available")


def demonstrate_model_focused_approach():
    """Demonstrate the model-focused approach vs component-specific approach."""
    tprint("\n🚀 Demonstrating Model-Focused Approach")
    tprint("=" * 60)
    
    tprint("📋 Key Differences:")
    tprint("   OLD (Component-specific):")
    tprint("   - TacticianExplainer, AnalystExplainer, SRExplainer, HMMExplainer")
    tprint("   - Separate explainers for each trading component")
    tprint("   - Manual integration required")
    tprint("   - Component-specific explanation formats")
    
    tprint("\n   NEW (Model-focused):")
    tprint("   - ModelExplainabilityManager handles all model types")
    tprint("   - Automatic integration with ML commons")
    tprint("   - Model-specific explanations (RandomForest, Neural Network, etc.)")
    tprint("   - Unified explanation format")
    tprint("   - Automatic caching and persistence")
    
    if SKLEARN_AVAILABLE:
        # Demonstrate different model types
        X, y, feature_names = create_sample_data(n_samples=200, n_features=6)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        manager = ModelExplainabilityManager()
        
        # Test different model types
        model_types = [
            ("RandomForestClassifier", RandomForestClassifier(n_estimators=20, random_state=42)),
        ]
        
        for model_name, model in model_types:
            tprint(f"\n🔄 Testing {model_name}...")
            model.fit(X_train, y_train)
            
            explanation = manager.explain_model(
                model=model,
                X_train=X_train,
                X_test=X_test,
                model_id=f"demo_{model_name.lower()}",
                model_type=model_name,
                feature_names=feature_names
            )
            
            tprint(f"   ✅ {model_name} explanation generated")
            tprint(f"   📊 Confidence: {explanation.explanation_confidence:.3f}")
            tprint(f"   ⏱️ Time: {explanation.processing_time_ms:.1f}ms")
    else:
        tprint("⚠️ Skipping model type demonstration - scikit-learn not available")


def main():
    """Run all demonstrations."""
    tprint("🧠 Model Explainability Integration Demonstration")
    tprint("=" * 80)
    tprint("This demonstration shows how the new model-focused explainability")
    tprint("system integrates with ML commons training and model registry.")
    tprint("=" * 80)
    
    try:
        # Run all demonstrations
        demonstrate_automatic_explainability()
        demonstrate_model_registry_integration()
        demonstrate_quick_explanations()
        demonstrate_model_focused_approach()
        
        tprint("\n" + "=" * 80)
        tprint("✅ All demonstrations completed successfully!")
        tprint("=" * 80)
        
        tprint("\n📋 Summary of Integration Benefits:")
        tprint("   ✅ Automatic explainability during model training")
        tprint("   ✅ Model registry integration with explanations")
        tprint("   ✅ Model-focused approach (not component-specific)")
        tprint("   ✅ Explanation caching and retrieval")
        tprint("   ✅ Integration with existing ML commons utilities")
        tprint("   ✅ Unified explanation format across all model types")
        
    except Exception as e:
        tprint(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()