"""
Demonstration of Log Interactions and Category-Based Interactions

This script demonstrates the new log interaction types and category-based
interaction generation (both within-category and between-category interactions).
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
import time

# Import the enhanced interaction generator
try:
    from src.feature_generation.utils.data_driven_interaction_generator import (
        DataDrivenInteractionGenerator,
        EnhancedInteractionConfig
    )
    from src.feature_generation.utils.enhanced_data_driven_interaction_generator import (
        EnhancedDataDrivenInteractionGenerator,
        EnhancedDataDrivenConfig
    )
    GENERATORS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Import error: {e}")
    GENERATORS_AVAILABLE = False

def create_sample_financial_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample financial data with features from different categories."""
    np.random.seed(42)
    
    # Generate base price data
    price = 100 + np.cumsum(np.random.randn(n_samples) * 0.01)
    
    data = pd.DataFrame({
        'close': price,
        'high': price + np.abs(np.random.randn(n_samples) * 0.5),
        'low': price - np.abs(np.random.randn(n_samples) * 0.5),
        'volume': np.random.randint(1000, 10000, n_samples),
        'open': np.roll(price, 1),
    })
    
    # Add momentum features
    data['rsi_14'] = 50 + np.random.randn(n_samples) * 10
    data['momentum_10'] = data['close'].pct_change(10) * 100
    data['roc_5'] = data['close'].pct_change(5) * 100
    
    # Add volatility features
    data['volatility_20'] = data['close'].rolling(20).std()
    data['atr_14'] = np.abs(data['high'] - data['low']) * 0.5
    data['bb_upper'] = data['close'].rolling(20).mean() + 2 * data['volatility_20']
    
    # Add trend features
    data['sma_20'] = data['close'].rolling(20).mean()
    data['ema_12'] = data['close'].ewm(span=12).mean()
    data['macd'] = data['ema_12'] - data['close'].ewm(span=26).mean()
    
    # Add volume features
    data['volume_sma_20'] = data['volume'].rolling(20).mean()
    data['volume_ratio'] = data['volume'] / data['volume_sma_20']
    data['obv'] = (data['volume'] * np.sign(data['close'].diff())).cumsum()
    
    # Add return features
    data['return_1'] = data['close'].pct_change()
    data['log_return_1'] = np.log(data['close'] / data['close'].shift(1))
    data['return_5'] = data['close'].pct_change(5)
    
    # Add oscillator features
    data['stoch_k'] = 50 + np.random.randn(n_samples) * 15
    data['williams_r'] = -50 + np.random.randn(n_samples) * 15
    data['cci_20'] = np.random.randn(n_samples) * 50
    
    # Add support/resistance features
    data['pivot_point'] = (data['high'] + data['low'] + data['close']) / 3
    data['resistance_1'] = 2 * data['pivot_point'] - data['low']
    data['support_1'] = 2 * data['pivot_point'] - data['high']
    
    # Add time features
    data['hour'] = np.random.randint(0, 24, n_samples)
    data['day_of_week'] = np.random.randint(0, 7, n_samples)
    data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
    
    # Add microstructure features
    data['bid_ask_spread'] = np.random.uniform(0.01, 0.05, n_samples)
    data['tick_volume'] = np.random.randint(1, 100, n_samples)
    
    # Add entropy features
    data['entropy_10'] = np.random.uniform(0, 1, n_samples)
    data['shannon_entropy'] = np.random.uniform(0, 2, n_samples)
    
    # Add regime features
    data['regime_state'] = np.random.randint(0, 3, n_samples)
    data['regime_change'] = (data['regime_state'] != data['regime_state'].shift(1)).astype(int)
    
    # Add acceleration features
    data['acceleration'] = data['close'].diff().diff()
    data['jerk'] = data['acceleration'].diff()
    
    # Add advanced statistical features
    data['skewness_20'] = data['close'].rolling(20).skew()
    data['kurtosis_20'] = data['close'].rolling(20).kurt()
    data['quantile_90'] = data['close'].rolling(20).quantile(0.9)
    
    # Add spectral features
    data['spectral_entropy'] = np.random.uniform(0, 1, n_samples)
    data['wavelet_energy'] = np.random.uniform(0, 10, n_samples)
    
    # Fill NaN values
    data = data.fillna(method='bfill').fillna(0)
    
    return data

def demonstrate_log_interactions():
    """Demonstrate the new log interaction types."""
    print("🔢 LOG INTERACTIONS DEMONSTRATION")
    print("=" * 50)
    
    if not GENERATORS_AVAILABLE:
        print("❌ Generators not available, cannot demonstrate")
        return
    
    # Create sample data
    data = create_sample_financial_data(500)
    targets = data['close'].pct_change().shift(-1)  # Next period return
    
    print(f"📊 Sample data shape: {data.shape}")
    print(f"📊 Features: {list(data.columns)}")
    
    # Initialize generator with log interactions
    config = EnhancedInteractionConfig(
        max_interactions=50,
        utility_threshold=0.05,
        enable_vectorbt=True,
        enable_parallel=True
    )
    
    generator = DataDrivenInteractionGenerator(config=config)
    
    print("\n🔧 Available Interaction Types:")
    for name, interaction_type in generator.interaction_types.items():
        print(f"   • {name}: {interaction_type.description}")
    
    print(f"\n📊 Total interaction types: {len(generator.interaction_types)}")
    
    # Generate interactions
    print("\n⚡ Generating interactions...")
    start_time = time.time()
    
    interactions = generator.generate_interactions(data, targets)
    
    generation_time = time.time() - start_time
    
    print(f"\n✅ Generated {len(interactions)} interactions in {generation_time:.2f}s")
    
    # Analyze log interactions
    log_interactions = [i for i in interactions if 'log' in i.interaction_type]
    print(f"\n📊 Log interactions generated: {len(log_interactions)}")
    
    if log_interactions:
        print("\n🔍 Log Interaction Examples:")
        for i, interaction in enumerate(log_interactions[:5]):
            print(f"   {i+1}. {interaction.feature_name}")
            print(f"      Type: {interaction.interaction_type}")
            print(f"      Utility: {interaction.utility_score:.4f}")
            print(f"      Parents: {interaction.parent_features}")
            print()
    
    # Performance stats
    stats = generator.get_performance_stats()
    print("📊 Performance Statistics:")
    print(f"   Total processing time: {stats['total_processing_time']:.2f}s")
    print(f"   VectorBT operations: {stats['vectorbt_operations']}")
    print(f"   Cache hit rate: {stats['cache_hit_rate']:.1%}")
    print(f"   Memory usage: {stats['memory_usage_mb']:.1f} MB")

def demonstrate_category_interactions():
    """Demonstrate category-based interactions (within and between categories)."""
    print("\n🏷️ CATEGORY-BASED INTERACTIONS DEMONSTRATION")
    print("=" * 50)
    
    if not GENERATORS_AVAILABLE:
        print("❌ Generators not available, cannot demonstrate")
        return
    
    # Create sample data with diverse features
    data = create_sample_financial_data(300)
    targets = data['close'].pct_change().shift(-1)
    
    print(f"📊 Sample data shape: {data.shape}")
    
    # Show feature categorization
    generator = DataDrivenInteractionGenerator()
    feature_categories = generator._categorize_features(list(data.columns))
    
    print("\n🏷️ Feature Categorization:")
    category_distribution = generator._get_category_distribution(feature_categories)
    for category, count in sorted(category_distribution.items()):
        features = [f for f, c in feature_categories.items() if c == category]
        print(f"   {category}: {count} features")
        print(f"      Examples: {features[:3]}")
    
    # Generate category-based combinations
    print("\n🔄 Generating category-based combinations...")
    feature_combinations = generator._generate_feature_combinations(list(data.columns))
    
    print(f"📊 Total combinations generated: {len(feature_combinations)}")
    
    # Analyze within vs between category combinations
    within_category = 0
    between_category = 0
    
    for feat1, feat2 in feature_combinations:
        cat1 = feature_categories.get(feat1, 'unknown')
        cat2 = feature_categories.get(feat2, 'unknown')
        
        if cat1 == cat2:
            within_category += 1
        else:
            between_category += 1
    
    print(f"\n📊 Category Interaction Analysis:")
    print(f"   Within-category combinations: {within_category}")
    print(f"   Between-category combinations: {between_category}")
    print(f"   Within-category ratio: {within_category/len(feature_combinations):.1%}")
    print(f"   Between-category ratio: {between_category/len(feature_combinations):.1%}")
    
    # Show examples of each type
    print(f"\n🔍 Within-Category Examples:")
    within_examples = []
    for feat1, feat2 in feature_combinations:
        cat1 = feature_categories.get(feat1, 'unknown')
        cat2 = feature_categories.get(feat2, 'unknown')
        if cat1 == cat2 and len(within_examples) < 5:
            within_examples.append((feat1, feat2, cat1))
    
    for i, (feat1, feat2, category) in enumerate(within_examples):
        print(f"   {i+1}. {feat1} × {feat2} ({category})")
    
    print(f"\n🔍 Between-Category Examples:")
    between_examples = []
    for feat1, feat2 in feature_combinations:
        cat1 = feature_categories.get(feat1, 'unknown')
        cat2 = feature_categories.get(feat2, 'unknown')
        if cat1 != cat2 and len(between_examples) < 5:
            between_examples.append((feat1, feat2, cat1, cat2))
    
    for i, (feat1, feat2, cat1, cat2) in enumerate(between_examples):
        print(f"   {i+1}. {feat1} ({cat1}) × {feat2} ({cat2})")

def demonstrate_enhanced_generator():
    """Demonstrate the enhanced generator with feature pre-selection."""
    print("\n🚀 ENHANCED GENERATOR DEMONSTRATION")
    print("=" * 50)
    
    if not GENERATORS_AVAILABLE:
        print("❌ Generators not available, cannot demonstrate")
        return
    
    # Create larger dataset
    data = create_sample_financial_data(1000)
    targets = data['close'].pct_change().shift(-1)
    
    print(f"📊 Dataset shape: {data.shape}")
    print(f"📊 Features available: {len(data.columns)}")
    
    # Initialize enhanced generator
    config = EnhancedDataDrivenConfig(
        target_feature_count=20,
        max_interactions=30,
        utility_threshold=0.05,
        enable_vectorbt=True
    )
    
    generator = EnhancedDataDrivenInteractionGenerator(config)
    
    print(f"\n🎯 Target feature count: {config.target_feature_count}")
    print(f"🎯 Max interactions: {config.max_interactions}")
    
    # Generate interactions
    print("\n⚡ Generating interactions with enhanced generator...")
    start_time = time.time()
    
    result = generator.generate_interactions(data, targets)
    
    generation_time = time.time() - start_time
    
    print(f"\n✅ Enhanced generation completed in {generation_time:.2f}s")
    print(f"📊 Selected features: {result.final_feature_count}")
    print(f"📊 Generated interactions: {result.final_interaction_count}")
    
    # Show feature selection results
    if hasattr(result, 'selected_features') and result.selected_features:
        print(f"\n🏷️ Selected Features by Category:")
        category_counts = {}
        for feature in result.selected_features:
            category = getattr(feature, 'category', 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        for category, count in sorted(category_counts.items()):
            print(f"   {category}: {count} features")
    
    # Show interaction types
    if hasattr(result, 'interactions') and result.interactions:
        interaction_types = {}
        for interaction in result.interactions:
            itype = interaction.interaction_type
            interaction_types[itype] = interaction_types.get(itype, 0) + 1
        
        print(f"\n🔧 Generated Interaction Types:")
        for itype, count in sorted(interaction_types.items()):
            print(f"   {itype}: {count} interactions")
    
    # Performance stats
    stats = generator.get_performance_stats()
    print(f"\n📊 Performance Statistics:")
    print(f"   Total processing time: {stats['total_processing_time']:.2f}s")
    print(f"   Feature selection time: {stats['feature_selection_time']:.2f}s")
    print(f"   Interaction generation time: {stats['interaction_generation_time']:.2f}s")
    print(f"   Categories used: {stats['feature_categories_used']}")

def main():
    """Main demonstration function."""
    print("🚀 LOG INTERACTIONS AND CATEGORY-BASED INTERACTIONS DEMO")
    print("=" * 70)
    
    try:
        # Demonstrate log interactions
        demonstrate_log_interactions()
        
        # Demonstrate category-based interactions
        demonstrate_category_interactions()
        
        # Demonstrate enhanced generator
        demonstrate_enhanced_generator()
        
        print("\n✅ DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        print("\n🎯 KEY FEATURES DEMONSTRATED:")
        print("   ✅ Log interactions (log_product, log_ratio, log_sum, log_difference)")
        print("   ✅ Log return interactions (log_return_product, log_return_ratio)")
        print("   ✅ Within-category interactions (same feature category)")
        print("   ✅ Between-category interactions (different feature categories)")
        print("   ✅ Intelligent feature categorization")
        print("   ✅ Enhanced generator with feature pre-selection")
        print("   ✅ Comprehensive performance monitoring")
        
        print("\n📈 BENEFITS:")
        print("   • Log interactions capture multiplicative relationships in financial data")
        print("   • Category-based interactions ensure diverse feature combinations")
        print("   • Within-category interactions capture intra-category patterns")
        print("   • Between-category interactions capture cross-category relationships")
        print("   • Intelligent categorization reduces manual feature grouping")
        print("   • Enhanced performance with VectorBT optimizations")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()