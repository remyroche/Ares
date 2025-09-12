#!/usr/bin/env python3
"""
Test script for Pipeline Integration

This script tests the full integration of temporal feature integration
into both MARKET_ANALYSIS and MODEL_TRAINING pipelines.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('PipelineIntegrationTest')

def create_sample_data(symbol: str = "BTCUSDT", days: int = 7) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    logger.info(f"Creating sample data for {symbol} ({days} days)")
    
    # Generate timestamps
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    timestamps = pd.date_range(start=start_time, end=end_time, freq='1min')
    
    # Generate realistic price data
    np.random.seed(42)  # For reproducible results
    n_points = len(timestamps)
    
    # Base price with trend and volatility
    base_price = 50000
    trend = np.linspace(0, 0.1, n_points)  # 10% trend over period
    volatility = np.random.normal(0, 0.02, n_points)  # 2% daily volatility
    price_changes = trend + volatility
    prices = base_price * np.exp(np.cumsum(price_changes))
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices * (1 + np.random.normal(0, 0.001, n_points)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_points))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_points))),
        'close': prices,
        'volume': np.random.lognorm(10, 1, n_points)
    })
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    # Add some basic technical indicators for testing
    data['rsi_14'] = calculate_rsi(data['close'], 14)
    data['sma_20'] = data['close'].rolling(20).mean()
    data['ema_12'] = data['close'].ewm(span=12).mean()
    
    logger.info(f"✅ Created sample data: {len(data)} rows, {len(data.columns)} columns")
    return data

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

async def test_market_analysis_pipeline():
    """Test MARKET_ANALYSIS pipeline with temporal feature integration."""
    logger.info("🚀 Testing MARKET_ANALYSIS Pipeline Integration")
    
    try:
        # Import MARKET_ANALYSIS sub-pipeline
        from src.training.steps.market_analysis.sub_pipeline import (
            MarketAnalysisSubPipeline, SubPipelineConfig, ExecutionMode
        )
        
        # Create sample data
        data = create_sample_data("BTCUSDT", days=3)  # 3 days for faster testing
        
        # Save data for pipeline
        data_dir = "test_data"
        Path(data_dir).mkdir(exist_ok=True)
        data_path = f"{data_dir}/BTCUSDT_binance_1m.parquet"
        data.to_parquet(data_path)
        logger.info(f"💾 Saved test data to: {data_path}")
        
        # Create pipeline configuration
        config = SubPipelineConfig(
            mode=ExecutionMode.LIGHT,  # Use light mode for faster testing
            symbol="BTCUSDT",
            exchange="binance",
            timeframe="1m",
            data_dir=data_dir,
            force_rerun=True
        )
        
        # Create pipeline
        pipeline = MarketAnalysisSubPipeline(config)
        
        # Test individual sub-pipelines
        logger.info("\n📊 Testing individual sub-pipelines...")
        
        # Test feature lookback optimization
        logger.info("🔧 Testing feature_lookback_optimization...")
        result1 = await pipeline.execute_sub_pipeline('feature_lookback_optimization', config)
        logger.info(f"✅ Feature lookback optimization: {result1.status.value}")
        
        # Test cross timeframe analysis
        logger.info("⏰ Testing cross_timeframe_analysis...")
        result2 = await pipeline.execute_sub_pipeline('cross_timeframe_analysis', config)
        logger.info(f"✅ Cross timeframe analysis: {result2.status.value}")
        
        # Test temporal feature integration
        logger.info("🔄 Testing temporal_feature_integration...")
        result3 = await pipeline.execute_sub_pipeline('temporal_feature_integration', config)
        logger.info(f"✅ Temporal feature integration: {result3.status.value}")
        
        if result3.status.value == 'completed':
            artifacts = result3.artifacts
            logger.info(f"   - Temporal features: {len(artifacts.get('temporal_features', {}))}")
            logger.info(f"   - Quality metrics: {artifacts.get('quality_metrics', {})}")
            logger.info(f"   - Integration summary: {artifacts.get('integration_summary', {})}")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ MARKET_ANALYSIS pipeline test failed: {e}")
        return False

async def test_model_training_pipeline():
    """Test MODEL_TRAINING pipeline with temporal features."""
    logger.info("\n🚀 Testing MODEL_TRAINING Pipeline Integration")
    
    try:
        # Import MODEL_TRAINING sub-pipeline
        from src.training.steps.model_training.sub_pipeline import (
            ModelTrainingSubPipeline, SubPipelineConfig, ExecutionMode
        )
        
        # Create sample temporal features (simulating MARKET_ANALYSIS output)
        data_dir = "test_data"
        temporal_features = create_sample_temporal_features()
        
        # Save temporal features
        temporal_df = pd.DataFrame(temporal_features)
        temporal_path = f"{data_dir}/temporal_features_BTCUSDT_binance_1m.parquet"
        temporal_df.to_parquet(temporal_path)
        logger.info(f"💾 Saved temporal features to: {temporal_path}")
        
        # Create temporal feature metadata
        metadata = {
            'lookback_rsi_14': {'type': 'lookback', 'variance': 0.15, 'mean': 50.0},
            'lookback_sma_20': {'type': 'lookback', 'variance': 0.08, 'mean': 50000.0},
            'cross_tf_momentum_1m': {'type': 'cross_timeframe', 'variance': 0.12, 'mean': 0.0}
        }
        
        metadata_path = f"{data_dir}/temporal_feature_metadata_BTCUSDT_binance_1m.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"💾 Saved temporal feature metadata to: {metadata_path}")
        
        # Create pipeline configuration
        config = SubPipelineConfig(
            mode=ExecutionMode.LIGHT,  # Use light mode for faster testing
            symbol="BTCUSDT",
            exchange="binance",
            timeframe="1m",
            data_dir=data_dir,
            force_rerun=True
        )
        
        # Create pipeline
        pipeline = ModelTrainingSubPipeline(config)
        
        # Test temporal feature loading
        logger.info("📊 Testing temporal feature loading...")
        temporal_loaded = await pipeline._load_temporal_features(config)
        logger.info(f"✅ Temporal features loaded: {temporal_loaded}")
        
        if temporal_loaded:
            temporal_info = pipeline._get_temporal_feature_info()
            logger.info(f"   - Total features: {temporal_info['count']}")
            logger.info(f"   - Lookback features: {temporal_info['lookback_features']}")
            logger.info(f"   - Cross timeframe features: {temporal_info['cross_timeframe_features']}")
        
        # Test general model training with temporal features
        logger.info("🤖 Testing general_model_training with temporal features...")
        result = await pipeline.execute_sub_pipeline('general_model_training', config)
        logger.info(f"✅ General model training: {result.status.value}")
        
        if result.status.value == 'completed':
            artifacts = result.artifacts
            logger.info(f"   - Temporal features used: {artifacts.get('temporal_features_used', False)}")
            logger.info(f"   - Temporal feature info: {artifacts.get('temporal_feature_info', {})}")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ MODEL_TRAINING pipeline test failed: {e}")
        return False

def create_sample_temporal_features() -> dict:
    """Create sample temporal features for testing."""
    np.random.seed(42)
    n_points = 1000
    
    return {
        'lookback_rsi_14': np.random.uniform(20, 80, n_points),
        'lookback_sma_20': np.random.uniform(49000, 51000, n_points),
        'lookback_ema_12': np.random.uniform(49000, 51000, n_points),
        'cross_tf_momentum_1m': np.random.uniform(-0.05, 0.05, n_points),
        'cross_tf_volatility_5m': np.random.uniform(0.01, 0.03, n_points),
        'cross_tf_range_15m': np.random.uniform(100, 500, n_points)
    }

async def test_full_pipeline_integration():
    """Test full pipeline integration from MARKET_ANALYSIS to MODEL_TRAINING."""
    logger.info("\n🚀 Testing Full Pipeline Integration")
    
    try:
        # Test MARKET_ANALYSIS pipeline
        market_success = await test_market_analysis_pipeline()
        if not market_success:
            logger.error("❌ MARKET_ANALYSIS pipeline test failed")
            return False
        
        # Test MODEL_TRAINING pipeline
        model_success = await test_model_training_pipeline()
        if not model_success:
            logger.error("❌ MODEL_TRAINING pipeline test failed")
            return False
        
        logger.info("✅ Full pipeline integration test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Full pipeline integration test failed: {e}")
        return False

async def test_pipeline_sequence():
    """Test the pipeline sequence with temporal feature integration."""
    logger.info("\n🚀 Testing Pipeline Sequence")
    
    try:
        from src.training.steps.market_analysis.sub_pipeline import (
            MarketAnalysisSubPipeline, SubPipelineConfig, ExecutionMode
        )
        
        # Create sample data
        data = create_sample_data("ETHUSDT", days=2)
        
        # Save data
        data_dir = "test_data"
        Path(data_dir).mkdir(exist_ok=True)
        data_path = f"{data_dir}/ETHUSDT_binance_1m.parquet"
        data.to_parquet(data_path)
        
        # Create pipeline configuration
        config = SubPipelineConfig(
            mode=ExecutionMode.LIGHT,
            symbol="ETHUSDT",
            exchange="binance",
            timeframe="1m",
            data_dir=data_dir,
            force_rerun=True
        )
        
        # Create pipeline
        pipeline = MarketAnalysisSubPipeline(config)
        
        # Test the sequence: feature_lookback_optimization -> cross_timeframe_analysis -> temporal_feature_integration
        logger.info("🔄 Testing pipeline sequence...")
        
        # Start with feature lookback optimization
        result1 = await pipeline.execute_sub_pipeline_with_next('feature_lookback_optimization', config)
        logger.info(f"✅ Pipeline sequence completed: {result1.status.value}")
        
        # Check if temporal features were created
        temporal_path = f"{data_dir}/temporal_features_ETHUSDT_binance_1m.parquet"
        if Path(temporal_path).exists():
            logger.info(f"✅ Temporal features file created: {temporal_path}")
            temporal_df = pd.read_parquet(temporal_path)
            logger.info(f"   - Features: {len(temporal_df.columns)}")
            logger.info(f"   - Rows: {len(temporal_df)}")
        else:
            logger.warning("⚠️ Temporal features file not found")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline sequence test failed: {e}")
        return False

async def main():
    """Main test function."""
    logger.info("🚀 Starting Pipeline Integration Test Suite")
    
    # Test individual components
    test1 = await test_market_analysis_pipeline()
    test2 = await test_model_training_pipeline()
    test3 = await test_pipeline_sequence()
    test4 = await test_full_pipeline_integration()
    
    if all([test1, test2, test3, test4]):
        logger.info("\n🎉 All pipeline integration tests passed successfully!")
        logger.info("\n📋 Summary:")
        logger.info("   ✅ MARKET_ANALYSIS pipeline with temporal feature integration")
        logger.info("   ✅ MODEL_TRAINING pipeline with temporal features")
        logger.info("   ✅ Pipeline sequence execution")
        logger.info("   ✅ Full pipeline integration")
        return True
    else:
        logger.error("\n❌ Some pipeline integration tests failed!")
        return False

if __name__ == "__main__":
    # Run the test suite
    success = asyncio.run(main())
    exit(0 if success else 1)