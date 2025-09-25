#!/usr/bin/env python3
"""
Example: Price Level Bank Usage

This script demonstrates the complete workflow for using the Price Level Bank:
1. Building the bank from historical data
2. Querying and analyzing the data
3. Using it in ML training
4. Integrating with feature generation

Run this as a complete example of the system in action.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from feature_generation.core.price_level_bank import PriceLevelBank, get_global_price_level_bank
from feature_generation.categories.support_resistance import (
    HistoricalPriceLevelCrossingGenerator,
    HistoricalPriceLevelBounceGenerator,
    HistoricalVolumeAtPriceLevelGenerator
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def example_1_build_and_query():
    """Example 1: Build bank and query data."""
    logger.info("="*60)
    logger.info("EXAMPLE 1: Building and Querying Price Level Bank")
    logger.info("="*60)

    # Initialize bank
    bank = get_global_price_level_bank()
    logger.info(f"Bank initialized with {bank.get_statistics()['total_levels']} levels")

    # Check if we need to build the bank
    stats = bank.get_statistics()
    if stats['total_levels'] == 0:
        logger.info("Bank is empty, demonstrating with sample data...")

        # For demonstration, we'll add some sample data
        sample_levels = create_sample_levels()
        level_ids = bank.add_levels(sample_levels)
        logger.info(f"Added {len(sample_levels)} sample levels")

    # Query examples
    logger.info("\n1. Getting most significant levels:")
    significant_levels = bank.get_most_significant_levels('BTCUSDT', '1h', top_k=5)
    for i, level in enumerate(significant_levels, 1):
        logger.info(f"   {i}. Price: ${level.price".2f"}, "
                   f"Significance: {level.significance_level".2f"}, "
                   f"Crossings: {level.historical_crossings}")

    logger.info("\n2. Querying by price range:")
    range_levels = bank.query_levels(
        symbol='BTCUSDT',
        min_price=45000,
        max_price=55000,
        min_significance=0.5,
        limit=3
    )
    for level in range_levels:
        logger.info(f"   Price: ${level.price".2f"}, "
                   f"Success Rate: {level.historical_success_rate".2f"}")

    logger.info("\n3. Getting bank statistics:")
    stats = bank.get_statistics()
    logger.info(f"   Total levels: {stats['total_levels']}")
    logger.info(f"   Symbols: {stats['symbols']}")
    logger.info(f"   Timeframes: {stats['timeframes']}")

def example_2_feature_generation():
    """Example 2: Using bank in feature generation."""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 2: Feature Generation with Bank Integration")
    logger.info("="*60)

    # Create sample data for demonstration
    dates = pd.date_range('2024-01-01', periods=100, freq='1h')
    prices = 50000 + np.cumsum(np.random.normal(0, 100, 100))  # Random walk around 50k
    volumes = np.random.exponential(1000, 100) + 100

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * 0.999,
        'high': prices * 1.002,
        'low': prices * 0.998,
        'close': prices,
        'volume': volumes
    })
    df.set_index('timestamp', inplace=True)

    logger.info(f"Created sample data: {len(df)} periods, price range ${df['low'].min()".0f"} - ${df['high'].max()".0f"}")

    # Initialize generators (they'll use the bank automatically)
    crossing_gen = HistoricalPriceLevelCrossingGenerator(level_pct=0.2, window=50)
    bounce_gen = HistoricalPriceLevelBounceGenerator(level_pct=0.2, window=50)
    volume_gen = HistoricalVolumeAtPriceLevelGenerator(level_pct=0.2, window=50)

    # Generate features
    logger.info("Generating historical price level features...")

    crossings = crossing_gen._generate_feature(df, symbol='BTCUSDT', timeframe='1h')
    bounces = bounce_gen._generate_feature(df, symbol='BTCUSDT', timeframe='1h')
    volumes = volume_gen._generate_feature(df, symbol='BTCUSDT', timeframe='1h')

    logger.info(f"Generated features:")
    logger.info(f"  Crossings: {len(crossings.dropna())} valid values")
    logger.info(f"  Bounces: {len(bounces.dropna())} valid values")
    logger.info(f"  Volumes: {len(volumes.dropna())} valid values")

    # Show sample of generated features
    recent_data = pd.DataFrame({
        'crossings': crossings,
        'bounces': bounces,
        'volumes': volumes
    }).tail(10)

    logger.info("\nRecent feature values:")
    logger.info(recent_data.to_string())

def example_3_ml_training():
    """Example 3: Using bank data for ML training."""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 3: ML Training with Bank Data")
    logger.info("="*60)

    # Get bank data for training
    bank = get_global_price_level_bank()
    levels = bank.query_levels(symbol='BTCUSDT', limit=1000)

    if not levels:
        logger.warning("No levels available for ML training example")
        return

    # Convert to DataFrame
    training_data = []
    for level in levels:
        training_data.append({
            'price': level.price,
            'level_pct': level.level_pct,
            'historical_crossings': level.historical_crossings,
            'historical_bounces': level.historical_bounces,
            'historical_volume': level.historical_volume,
            'historical_touch_density': level.historical_touch_density,
            'historical_success_rate': level.historical_success_rate,
            'significance_level': level.significance_level,
            'total_activity': (level.historical_crossings +
                             level.historical_bounces +
                             level.historical_touch_density * 10)
        })

    df = pd.DataFrame(training_data)
    logger.info(f"Prepared {len(df)} samples for ML training")
    logger.info(f"Feature columns: {list(df.columns)}")

    # Prepare features and target
    feature_cols = ['historical_crossings', 'historical_bounces', 'historical_volume',
                   'historical_touch_density', 'total_activity']
    target_col = 'historical_success_rate'

    X = df[feature_cols]
    y = df[target_col]

    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info("
Model Performance:")
    logger.info(f"  Mean Squared Error: {mse".4f"}")
    logger.info(f"  R² Score: {r2".4f"}")

    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    logger.info("
Feature Importance:")
    for _, row in importance.iterrows():
        logger.info(f"  {row['feature']}: {row['importance']".4f"}")

def create_sample_levels():
    """Create sample price level data for demonstration."""
    from feature_generation.core.price_level_bank import PriceLevelData
    import pandas as pd

    base_prices = [45000, 46000, 47000, 48000, 49000, 50000, 51000, 52000, 53000, 54000]

    levels = []
    for price in base_prices:
        for pct in [0.1, 0.2, 0.5, 1.0]:
            level = PriceLevelData(
                price=price,
                level_pct=pct,
                symbol='BTCUSDT',
                timeframe='1h',
                timestamp=pd.Timestamp('2024-01-01'),
                historical_crossings=np.random.randint(5, 50),
                historical_bounces=np.random.randint(2, 20),
                historical_volume=np.random.uniform(10000, 100000),
                historical_touch_density=np.random.uniform(0.1, 1.0),
                historical_time_decay=np.random.uniform(0.3, 0.9),
                historical_success_rate=np.random.uniform(0.4, 0.8),
                significance_level=np.random.uniform(0.5, 0.9),
                session_type=np.random.choice(['asian', 'european', 'us']),
                day_of_week=np.random.randint(0, 7),
                hour_of_day=np.random.randint(0, 24)
            )
            levels.append(level)

    return levels

def main():
    """Run all examples."""
    logger.info("Starting Price Level Bank Examples")
    logger.info("This demonstrates the complete workflow")

    try:
        example_1_build_and_query()
        example_2_feature_generation()
        example_3_ml_training()

        logger.info("\n" + "="*60)
        logger.info("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        logger.info("="*60)

        print("\n📚 Key Takeaways:")
        print("• Price Level Bank provides pre-computed historical analysis")
        print("• Eliminates redundant calculations in feature generation")
        print("• Enables efficient ML training with consistent data")
        print("• Supports advanced querying and filtering")
        print("• Integrates seamlessly with existing feature generators")

        print("\n🔧 Next Steps:")
        print("1. Build bank with your historical data:")
        print("   python build_price_level_bank.py --symbol BTCUSDT --timeframe 1h --start-date 2023-01-01 --end-date 2024-01-01")
        print("2. Query and analyze:")
        print("   python query_price_level_bank.py --symbol BTCUSDT --top 10")
        print("3. Use in ML training or feature generation")

    except Exception as e:
        logger.error(f"Example failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()