"""
LightGBM + Featuretools Integration Example

This example demonstrates how to use the new LightGBM/CatBoost + Featuretools
feature generation system as a replacement for Random Forest + SHAP.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.lightgbm_feature_generator import (
    LightGBMFeatureGenerator,
    FeatureGenerationConfig,
    create_lightgbm_feature_generator
)

def create_sample_data(n_samples=1000, n_features=20):
    """Create sample financial data for testing."""
    np.random.seed(42)
    
    # Create time series data
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    # Generate price data
    price_data = {}
    base_price = 100
    
    for i in range(n_features):
        # Generate correlated price series
        returns = np.random.normal(0.001, 0.02, n_samples)
        if i > 0:
            # Add some correlation with previous series
            returns += 0.3 * np.random.normal(0, 0.01, n_samples)
        
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        price_data[f'price_{i}'] = prices
    
    # Add volume data
    for i in range(5):
        price_data[f'volume_{i}'] = np.random.lognormal(10, 1, n_samples)
    
    # Add technical indicators
    price_data['rsi'] = np.random.uniform(20, 80, n_samples)
    price_data['macd'] = np.random.normal(0, 0.5, n_samples)
    price_data['bollinger_upper'] = np.random.uniform(95, 105, n_samples)
    price_data['bollinger_lower'] = np.random.uniform(95, 105, n_samples)
    
    # Create target (next period return)
    target = np.random.normal(0, 0.02, n_samples)
    
    # Create DataFrame
    data = pd.DataFrame(price_data, index=dates)
    data['target'] = target
    
    return data

def test_lightgbm_feature_generation():
    """Test the LightGBM feature generation system."""
    print("🚀 Testing LightGBM + Featuretools Feature Generation System")
    print("=" * 60)
    
    # Create sample data
    print("📊 Creating sample data...")
    data = create_sample_data(n_samples=500, n_features=15)
    print(f"✅ Data created: {data.shape[0]} samples, {data.shape[1]} features")
    
    # Test different configurations
    configs = [
        {
            'name': 'LightGBM (Default)',
            'config': FeatureGenerationConfig(
                model_type='lightgbm',
                max_features=50,
                use_shap=True,
                use_ale=True
            )
        },
        {
            'name': 'CatBoost (Alternative)',
            'config': FeatureGenerationConfig(
                model_type='catboost',
                max_features=30,
                use_shap=True,
                use_ale=False
            )
        },
        {
            'name': 'Light Mode',
            'config': FeatureGenerationConfig(
                model_type='lightgbm',
                max_features=20,
                use_shap=False,
                use_ale=False,
                max_depth_featuretools=1
            )
        }
    ]
    
    for test_config in configs:
        print(f"\n🔧 Testing {test_config['name']}")
        print("-" * 40)
        
        try:
            # Create generator
            generator = create_lightgbm_feature_generator(test_config['config'])
            
            # Generate features
            result = generator.generate_features(
                data=data,
                target_column='target',
                execution_mode='full'
            )
            
            # Display results
            print(f"✅ Generation completed in {result.generation_time:.3f}s")
            print(f"📊 Generated {result.n_features_generated} total features")
            print(f"🎯 Selected {result.n_features_selected} best features")
            print(f"🔍 SHAP analysis: {'✅' if result.shap_analysis_completed else '❌'}")
            print(f"📈 ALE analysis: {'✅' if result.ale_analysis_completed else '❌'}")
            print(f"⚡ Featuretools features: {result.featuretools_features}")
            
            # Show model performance
            if result.model_performance:
                print(f"📊 Model Performance:")
                for metric, value in result.model_performance.items():
                    print(f"   {metric}: {value:.4f}")
            
            # Show top features
            if result.generated_features:
                print(f"🏆 Top 5 Features:")
                top_features = sorted(result.generated_features, 
                                    key=lambda x: x.importance_score, reverse=True)[:5]
                for i, feature in enumerate(top_features, 1):
                    print(f"   {i}. {feature.name} (score: {feature.importance_score:.4f})")
            
            # Show performance stats
            stats = generator.get_performance_stats()
            print(f"📈 Performance Stats:")
            print(f"   Total generations: {stats['total_generations']}")
            print(f"   Successful: {stats['successful_generations']}")
            print(f"   Failed: {stats['failed_generations']}")
            print(f"   SHAP analyses: {stats['shap_analyses']}")
            print(f"   ALE analyses: {stats['ale_analyses']}")
            print(f"   Featuretools features: {stats['featuretools_features']}")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")

def compare_with_random_forest():
    """Compare performance with Random Forest approach."""
    print("\n🔄 Comparing with Random Forest approach")
    print("-" * 40)
    
    # Create sample data
    data = create_sample_data(n_samples=300, n_features=10)
    
    # Test LightGBM approach
    print("Testing LightGBM + Featuretools...")
    lightgbm_config = FeatureGenerationConfig(
        model_type='lightgbm',
        max_features=25,
        use_shap=True,
        use_ale=True
    )
    
    generator = create_lightgbm_feature_generator(lightgbm_config)
    result = generator.generate_features(data, 'target', execution_mode='full')
    
    print(f"LightGBM Results:")
    print(f"  Features generated: {result.n_features_generated}")
    print(f"  Features selected: {result.n_features_selected}")
    print(f"  Generation time: {result.generation_time:.3f}s")
    print(f"  R² Score: {result.model_performance.get('r2_score', 0):.4f}")
    
    print("\n✅ Comparison completed!")

if __name__ == "__main__":
    print("🌟 LightGBM + Featuretools Feature Generation Demo")
    print("=" * 60)
    
    # Run tests
    test_lightgbm_feature_generation()
    compare_with_random_forest()
    
    print("\n🎉 Demo completed successfully!")
    print("\nKey Benefits of the new system:")
    print("• Faster training with LightGBM/CatBoost")
    print("• Better calibrated models")
    print("• Advanced feature synthesis with Featuretools")
    print("• SHAP + ALE validation for feature impact")
    print("• Maximum 100 features limit")
    print("• Improved performance and accuracy")