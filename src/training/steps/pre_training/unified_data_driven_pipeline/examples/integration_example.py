#!/usr/bin/env python3
"""
Integration Example: Unified Pipeline with Existing Feature Generation

This example demonstrates how to integrate the new unified data-driven pipeline
with the existing src/feature_generation/ system.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

def create_sample_market_data(n_samples=1000):
    """Create sample market data for demonstration."""
    print("Creating sample market data...")
    
    # Create date index
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    # Generate realistic market data
    np.random.seed(42)
    
    # Price data with trend and volatility
    price = 100
    prices = []
    returns = []
    
    for i in range(n_samples):
        # Add trend and volatility
        trend = 0.0001 * i  # Slight upward trend
        volatility = 0.02 * (1 + 0.5 * np.sin(i * 0.1))  # Varying volatility
        return_val = np.random.normal(trend, volatility)
        
        price *= (1 + return_val)
        prices.append(price)
        returns.append(return_val)
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_samples)
    }, index=dates)
    
    # Create targets (future returns)
    targets = pd.Series(returns[1:] + [0], index=dates)  # Shift returns forward
    
    print(f"Created market data: {data.shape}, targets: {targets.shape}")
    return data, targets

def example_basic_integration():
    """Example 1: Basic integration using existing feature generation."""
    print("\n" + "="*60)
    print("EXAMPLE 1: BASIC INTEGRATION")
    print("="*60)
    
    # Create sample data
    data, targets = create_sample_market_data(500)
    
    try:
        # Import unified pipeline
        from src.training.steps.pre_training.unified_data_driven_pipeline import process_features, create_default_config
        
        # Import feature generation (if available)
        try:
            from src.feature_generation.core.factory import get_feature_bank
            from src.feature_generation.categories.momentum import MomentumFeatures
            from src.feature_generation.categories.volatility import VolatilityFeatures
            
            print("✓ Feature generation system available")
            use_existing_features = True
        except ImportError:
            print("⚠️ Feature generation system not available, using synthetic features")
            use_existing_features = False
        
        if use_existing_features:
            # Generate features using existing system
            print("\n1. Generating features using existing feature generation system...")
            
            # Create feature generators
            momentum_gen = MomentumFeatures()
            volatility_gen = VolatilityFeatures()
            
            # Generate features
            momentum_result = momentum_gen.generate(data)
            volatility_result = volatility_gen.generate(data)
            
            # Combine features
            all_features = {}
            all_features.update(momentum_result.features)
            all_features.update(volatility_result.features)
            
            features_df = pd.DataFrame(all_features, index=data.index)
            print(f"✓ Generated {len(features_df.columns)} features")
            
        else:
            # Create synthetic features for demonstration
            print("\n1. Creating synthetic features for demonstration...")
            
            features = {}
            for i in range(20):
                features[f'momentum_{i}'] = data['close'].pct_change(i+1)
                features[f'volatility_{i}'] = data['close'].rolling(i+1).std()
                features[f'volume_{i}'] = data['volume'].rolling(i+1).mean()
            
            features_df = pd.DataFrame(features, index=data.index)
            features_df = features_df.fillna(method='ffill').fillna(0)
            print(f"✓ Created {len(features_df.columns)} synthetic features")
        
        # Select optimal features using unified pipeline
        print("\n2. Selecting optimal features using unified pipeline...")
        
        config = create_default_config()
        config.feature_selection.multi_objective.max_features = 10
        config.feature_selection.multi_objective.min_features = 3
        
        result = process_features(features_df, targets, config=config)
        
        print(f"\nResults:")
        print(f"- Selected features: {len(result.selected_features)}")
        print(f"- Processing time: {result.processing_time:.2f}s")
        print(f"- Out-of-sample Sharpe: {result.out_of_sample_sharpe:.3f}")
        print(f"- Max drawdown: {result.max_drawdown:.3f}")
        print(f"- Stability score: {result.stability_score:.3f}")
        print(f"- Diversity score: {result.diversity_score:.3f}")
        
        print(f"\nSelected features: {result.selected_features[:5]}...")  # Show first 5
        
    except Exception as e:
        print(f"❌ Basic integration failed: {e}")
        import traceback
        traceback.print_exc()

def example_advanced_integration():
    """Example 2: Advanced integration with custom pipeline class."""
    print("\n" + "="*60)
    print("EXAMPLE 2: ADVANCED INTEGRATION")
    print("="*60)
    
    # Create sample data
    data, targets = create_sample_market_data(800)
    
    try:
        # Import required modules
        from src.training.steps.pre_training.unified_data_driven_pipeline import create_unified_pipeline, create_high_performance_config
        
        class IntegratedFeaturePipeline:
            """Integrated pipeline that generates and selects features."""
            
            def __init__(self):
                print("🔧 Initializing integrated pipeline...")
                
                # Create unified pipeline
                self.config = create_high_performance_config()
                self.pipeline = create_unified_pipeline(self.config)
                
                # Try to import feature generation
                try:
                    from src.feature_generation.core.factory import get_feature_bank
                    self.feature_bank = get_feature_bank()
                    self.use_existing_features = True
                    print("✓ Feature generation system available")
                except ImportError:
                    self.feature_bank = None
                    self.use_existing_features = False
                    print("⚠️ Using synthetic feature generation")
            
            def generate_features(self, data, categories=None):
                """Generate features using existing or synthetic methods."""
                if categories is None:
                    categories = ['momentum', 'volatility', 'volume']
                
                features = {}
                
                if self.use_existing_features and self.feature_bank:
                    # Use existing feature generation
                    for category in categories:
                        try:
                            generators = self.feature_bank.get_generators_by_category(category)
                            for generator in generators[:3]:  # Limit to 3 per category
                                result = generator.generate(data)
                                features.update(result.features)
                        except Exception as e:
                            print(f"Warning: {category} generation failed: {e}")
                else:
                    # Use synthetic feature generation
                    for category in categories:
                        if category == 'momentum':
                            for i in range(5):
                                features[f'momentum_{i}'] = data['close'].pct_change(i+1)
                        elif category == 'volatility':
                            for i in range(5):
                                features[f'volatility_{i}'] = data['close'].rolling(i+1).std()
                        elif category == 'volume':
                            for i in range(5):
                                features[f'volume_{i}'] = data['volume'].rolling(i+1).mean()
                
                features_df = pd.DataFrame(features, index=data.index)
                features_df = features_df.fillna(method='ffill').fillna(0)
                
                return features_df
            
            def process_with_categories(self, data, targets):
                """Process features with category-aware optimization."""
                print("\n📊 Processing features by category...")
                
                # Generate features by category
                category_features = {}
                categories = ['momentum', 'volatility', 'volume']
                
                for category in categories:
                    print(f"  Generating {category} features...")
                    category_data = self.generate_features(data, [category])
                    if not category_data.empty:
                        category_features[category] = category_data
                        print(f"    ✓ Generated {len(category_data.columns)} {category} features")
                
                # Combine all features
                if category_features:
                    all_features = pd.concat(category_features.values(), axis=1)
                    print(f"  ✓ Combined to {len(all_features.columns)} total features")
                else:
                    print("  ⚠️ No features generated")
                    return None
                
                # Select optimal features
                print("\n🎯 Selecting optimal features...")
                result = self.pipeline.process(all_features, targets)
                
                # Analyze by category
                category_breakdown = self._analyze_by_category(result.selected_features, category_features)
                
                return {
                    'selected_features': result.selected_features,
                    'objective_values': result.objective_values,
                    'category_breakdown': category_breakdown,
                    'processing_time': result.processing_time
                }
            
            def _analyze_by_category(self, selected_features, category_features):
                """Analyze selected features by category."""
                breakdown = {}
                for category, features_df in category_features.items():
                    category_selected = [f for f in selected_features if f in features_df.columns]
                    breakdown[category] = {
                        'selected': len(category_selected),
                        'total': len(features_df.columns),
                        'features': category_selected
                    }
                return breakdown
        
        # Create and run integrated pipeline
        print("\n🚀 Running integrated pipeline...")
        pipeline = IntegratedFeaturePipeline()
        result = pipeline.process_with_categories(data, targets)
        
        if result:
            print(f"\nResults:")
            print(f"- Selected features: {len(result['selected_features'])}")
            print(f"- Processing time: {result['processing_time']:.2f}s")
            print(f"- Objective values: {result['objective_values']}")
            
            print(f"\nCategory breakdown:")
            for category, breakdown in result['category_breakdown'].items():
                print(f"  {category}: {breakdown['selected']}/{breakdown['total']} features selected")
                if breakdown['features']:
                    print(f"    Selected: {breakdown['features'][:3]}...")  # Show first 3
        
    except Exception as e:
        print(f"❌ Advanced integration failed: {e}")
        import traceback
        traceback.print_exc()

def example_streaming_integration():
    """Example 3: Streaming integration for real-time processing."""
    print("\n" + "="*60)
    print("EXAMPLE 3: STREAMING INTEGRATION")
    print("="*60)
    
    try:
        from src.training.steps.pre_training.unified_data_driven_pipeline import create_unified_pipeline, create_fast_config
        
        class StreamingFeaturePipeline:
            """Pipeline for streaming data with incremental feature generation."""
            
            def __init__(self):
                print("🔧 Initializing streaming pipeline...")
                
                # Use fast config for streaming
                self.config = create_fast_config()
                self.pipeline = create_unified_pipeline(self.config)
                self.feature_cache = {}
                self.is_initialized = False
            
            def initialize(self, data_sample):
                """Initialize with sample data."""
                print("📊 Initializing with sample data...")
                
                # Generate initial features
                self.feature_cache = self._generate_synthetic_features(data_sample)
                self.is_initialized = True
                print(f"✓ Initialized with {len(self.feature_cache)} features")
            
            def _generate_synthetic_features(self, data):
                """Generate synthetic features for demonstration."""
                features = {}
                
                # Momentum features
                for i in range(3):
                    features[f'momentum_{i}'] = data['close'].pct_change(i+1)
                
                # Volatility features
                for i in range(3):
                    features[f'volatility_{i}'] = data['close'].rolling(i+1).std()
                
                # Volume features
                for i in range(3):
                    features[f'volume_{i}'] = data['volume'].rolling(i+1).mean()
                
                return features
            
            def process_streaming_data(self, new_data, targets=None):
                """Process streaming data."""
                if not self.is_initialized:
                    self.initialize(new_data)
                    return None
                
                print(f"📈 Processing streaming data: {len(new_data)} samples")
                
                # Generate features for new data
                new_features = self._generate_synthetic_features(new_data)
                
                # Add to feature cache
                self.feature_cache.update(new_features)
                
                # Convert to DataFrame
                features_df = pd.DataFrame(self.feature_cache, index=new_data.index)
                features_df = features_df.fillna(method='ffill').fillna(0)
                
                # Select features if targets provided
                if targets is not None:
                    print("🎯 Selecting features...")
                    result = self.pipeline.process(features_df, targets)
                    return result
                
                return features_df
        
        # Create streaming pipeline
        pipeline = StreamingFeaturePipeline()
        
        # Simulate streaming data
        print("\n📊 Simulating streaming data...")
        
        # Initial batch
        initial_data, initial_targets = create_sample_market_data(100)
        pipeline.initialize(initial_data)
        
        # Streaming batches
        for i in range(3):
            batch_data, batch_targets = create_sample_market_data(50)
            result = pipeline.process_streaming_data(batch_data, batch_targets)
            
            if result:
                print(f"  Batch {i+1}: Selected {len(result.selected_features)} features")
        
        print("✓ Streaming integration completed")
        
    except Exception as e:
        print(f"❌ Streaming integration failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all integration examples."""
    print("UNIFIED PIPELINE + FEATURE GENERATION INTEGRATION EXAMPLES")
    print("=" * 60)
    
    try:
        # Run examples
        example_basic_integration()
        example_advanced_integration()
        example_streaming_integration()
        
        print("\n" + "="*60)
        print("🎉 ALL INTEGRATION EXAMPLES COMPLETED!")
        print("="*60)
        print("\nKey Integration Points:")
        print("1. Use existing feature_generation/ for feature creation")
        print("2. Use unified pipeline for feature selection and optimization")
        print("3. Combine both systems for comprehensive feature engineering")
        print("4. Support both batch and streaming processing")
        
    except Exception as e:
        print(f"\n❌ Integration examples failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()