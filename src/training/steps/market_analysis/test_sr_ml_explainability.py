#!/usr/bin/env python3
"""
Test script for SR ML Learning Explainability Integration

This script tests the new explainability integration in the SR ML learning pipeline.
"""

import sys
import os
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, Any

# Add the workspace to the path
sys.path.insert(0, '/workspace')

def create_mock_data():
    """Create mock data for testing."""
    # Create mock clustered levels
    clustered_levels = {
        'clusters': [
            {
                'levels': [
                    {'price': 100.0, 'strength': 0.8},
                    {'price': 101.0, 'strength': 0.7},
                    {'price': 99.5, 'strength': 0.6}
                ],
                'center_price': 100.0,
                'cluster_strength': 0.7,
                'cluster_type': 'support'
            },
            {
                'levels': [
                    {'price': 105.0, 'strength': 0.9},
                    {'price': 106.0, 'strength': 0.8}
                ],
                'center_price': 105.5,
                'cluster_strength': 0.85,
                'cluster_type': 'resistance'
            }
        ]
    }
    
    # Create mock features data
    features_data = pd.DataFrame({
        'price': np.random.randn(100) * 10 + 100,
        'volume': np.random.randn(100) * 1000 + 5000,
        'rsi': np.random.rand(100) * 100,
        'macd': np.random.randn(100) * 2,
        'bollinger_upper': np.random.randn(100) * 5 + 105,
        'bollinger_lower': np.random.randn(100) * 5 + 95,
        'atr': np.random.rand(100) * 5 + 1
    })
    
    return clustered_levels, features_data

async def test_sr_ml_explainability():
    """Test the SR ML learning explainability integration."""
    print("🧪 Testing SR ML Learning Explainability Integration")
    print("=" * 60)
    
    try:
        # Import the SR ML learning step
        from src.training.steps.market_analysis.sr_ml_learning import SRMLLearningStep
        
        # Create mock data
        clustered_levels, features_data = create_mock_data()
        
        # Create pipeline state
        pipeline_state = {
            'clustered_levels': clustered_levels,
            'dataframe': features_data
        }
        
        # Initialize the SR ML learning step
        config = {
            'ml_learning': {
                'test_size': 0.3,
                'random_state': 42
            },
            'sr_optimization': {
                'min_touches': 2,
                'tolerance_pct': 0.5,
                'lookback_periods': 10
            }
        }
        
        print("🔄 Initializing SR ML Learning Step...")
        sr_ml_step = SRMLLearningStep(config)
        
        # Check if explainability manager was initialized
        if hasattr(sr_ml_step, 'explainability_manager') and sr_ml_step.explainability_manager is not None:
            print("✅ Model explainability manager initialized successfully")
        else:
            print("⚠️ Model explainability manager not available")
        
        print("🔄 Executing SR ML Learning...")
        result = await sr_ml_step.execute({}, pipeline_state)
        
        if result['success']:
            print("✅ SR ML Learning executed successfully")
            
            ml_results = result.get('ml_results', {})
            model_results = ml_results.get('model_results', {})
            
            print(f"📊 Models trained: {len(model_results)}")
            
            # Check explainability integration
            explanations_count = 0
            for model_name, model_result in model_results.items():
                if 'model_explanation' in model_result and model_result['model_explanation'] is not None:
                    explanations_count += 1
                    explanation = model_result['model_explanation']
                    print(f"🧠 {model_name}: Explanation generated")
                    print(f"   • Model ID: {explanation.model_id}")
                    print(f"   • Model type: {explanation.model_type}")
                    print(f"   • Explanation confidence: {explanation.explanation_confidence:.3f}")
                    print(f"   • Processing time: {explanation.processing_time_ms:.1f}ms")
                else:
                    print(f"⚠️ {model_name}: No explanation generated")
            
            print(f"📊 Summary: {explanations_count}/{len(model_results)} models have explanations")
            
            # Test cache statistics if available
            if hasattr(sr_ml_step, 'explainability_manager') and sr_ml_step.explainability_manager is not None:
                cache_stats = sr_ml_step.explainability_manager.get_cache_stats()
                print(f"📊 Cache statistics:")
                print(f"   • Cache size: {cache_stats['cache_size']}")
                print(f"   • Cache hits: {cache_stats['cache_hits']}")
                print(f"   • Cache misses: {cache_stats['cache_misses']}")
                print(f"   • Hit rate: {cache_stats['hit_rate']:.3f}")
            
        else:
            print(f"❌ SR ML Learning failed: {result.get('error', 'Unknown error')}")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("This might be due to missing dependencies (numpy, pandas, sklearn)")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def test_imports():
    """Test that the new imports work correctly."""
    print("🧪 Testing Import Integration")
    print("=" * 40)
    
    try:
        # Test ML commons imports
        from src.utils.ml_common import (
            ModelExplainabilityManager,
            ModelExplanationResult
        )
        print("✅ ModelExplainabilityManager imported successfully")
        print("✅ ModelExplanationResult imported successfully")
        
        # Test SR ML learning import
        from src.training.steps.market_analysis.sr_ml_learning import SRMLLearningStep
        print("✅ SRMLLearningStep imported successfully")
        
        # Test initialization
        manager = ModelExplainabilityManager()
        print("✅ ModelExplainabilityManager initialized successfully")
        
        cache_stats = manager.get_cache_stats()
        print(f"✅ Cache stats retrieved: {cache_stats}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Test error: {e}")

def main():
    """Run all tests."""
    print("🧠 SR ML Learning Explainability Integration Test")
    print("=" * 80)
    
    # Test imports first
    test_imports()
    
    print("\n" + "=" * 80)
    
    # Test full integration
    try:
        asyncio.run(test_sr_ml_explainability())
    except Exception as e:
        print(f"❌ Async test failed: {e}")
    
    print("\n" + "=" * 80)
    print("✅ Integration test completed!")

if __name__ == "__main__":
    main()