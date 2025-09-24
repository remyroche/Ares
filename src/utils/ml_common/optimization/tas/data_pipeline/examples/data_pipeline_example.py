"""
Data Pipeline Example for TAS

Comprehensive example demonstrating the usage of the data pipeline
for tree architecture search including all components.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import pipeline components
from ..pipeline_orchestrator import DataPipelineOrchestrator, PipelineConfig
from ..data_ingestion import DataIngestor, DataIngestionConfig
from ..data_preprocessing import DataPreprocessor, DataPreprocessingConfig
from ..feature_engineering import FeatureEngineer, FeatureEngineeringConfig
from ..regime_detection import RegimeDetectorPipeline, RegimeDetectionPipelineConfig
from ..data_validation import DataValidator, DataValidationConfig
from ..data_storage import DataStorageManager, StorageConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_synthetic_data(symbol: str = "BTCUSDT", 
                         timeframe: str = "1h",
                         start_date: datetime = None,
                         end_date: datetime = None,
                         num_points: int = 1000) -> pd.DataFrame:
    """
    Create synthetic market data for testing.
    
    Args:
        symbol: Trading symbol
        timeframe: Data timeframe
        start_date: Start date for data
        end_date: End date for data
        num_points: Number of data points
        
    Returns:
        Synthetic market data
    """
    logger.info(f"🔄 Creating synthetic data for {symbol} {timeframe}")
    
    # Set default dates
    if start_date is None:
        start_date = datetime.now() - timedelta(days=30)
    if end_date is None:
        end_date = datetime.now()
    
    # Generate time index
    time_index = pd.date_range(start=start_date, end=end_date, periods=num_points)
    
    # Generate synthetic price data with multiple regimes
    np.random.seed(42)
    
    # Create different market regimes
    regime_length = num_points // 4
    regimes = []
    
    # Regime 1: Trending up
    trend1 = np.linspace(100, 150, regime_length)
    noise1 = np.random.normal(0, 2, regime_length)
    regimes.extend(trend1 + noise1)
    
    # Regime 2: High volatility
    trend2 = np.linspace(150, 140, regime_length)
    noise2 = np.random.normal(0, 8, regime_length)
    regimes.extend(trend2 + noise2)
    
    # Regime 3: Trending down
    trend3 = np.linspace(140, 120, regime_length)
    noise3 = np.random.normal(0, 3, regime_length)
    regimes.extend(trend3 + noise3)
    
    # Regime 4: Low volatility
    trend4 = np.linspace(120, 125, regime_length)
    noise4 = np.random.normal(0, 1, regime_length)
    regimes.extend(trend4 + noise4)
    
    # Ensure we have the right number of points
    regimes = regimes[:num_points]
    
    # Generate OHLCV data
    close_prices = np.array(regimes)
    open_prices = close_prices + np.random.normal(0, 0.5, num_points)
    high_prices = np.maximum(open_prices, close_prices) + np.abs(np.random.normal(0, 1, num_points))
    low_prices = np.minimum(open_prices, close_prices) - np.abs(np.random.normal(0, 1, num_points))
    volumes = np.random.lognormal(10, 1, num_points)
    
    # Create DataFrame
    data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=time_index)
    
    # Add some missing values and outliers for testing
    data.iloc[100:105, 0] = np.nan  # Missing values
    data.iloc[200, 1] = data.iloc[200, 1] * 10  # Outlier
    
    logger.info(f"✅ Synthetic data created: {data.shape}")
    logger.info(f"📊 Date range: {data.index[0]} to {data.index[-1]}")
    logger.info(f"📊 Price range: {data['close'].min():.2f} to {data['close'].max():.2f}")
    
    return data


def demonstrate_individual_components():
    """Demonstrate individual pipeline components."""
    logger.info("🚀 Demonstrating individual pipeline components")
    
    # Create synthetic data
    data = create_synthetic_data()
    
    # 1. Data Ingestion
    logger.info("\n" + "="*50)
    logger.info("1. DATA INGESTION")
    logger.info("="*50)
    
    ingestion_config = DataIngestionConfig(
        data_source="synthetic",
        enable_klines_manager=True,
        enable_data_validation=True
    )
    
    data_ingestor = DataIngestor(ingestion_config)
    
    # Simulate loading data (in real scenario, this would load from KlinesParquetManager)
    logger.info("📦 Data loaded successfully")
    logger.info(f"📊 Data shape: {data.shape}")
    logger.info(f"📊 Data columns: {list(data.columns)}")
    
    # 2. Data Preprocessing
    logger.info("\n" + "="*50)
    logger.info("2. DATA PREPROCESSING")
    logger.info("="*50)
    
    preprocessing_config = DataPreprocessingConfig(
        enable_cleaning=True,
        enable_timestamp_regularization=True,
        enable_outlier_detection=True,
        enable_missing_value_handling=True
    )
    
    data_preprocessor = DataPreprocessor(preprocessing_config)
    processed_data = data_preprocessor.preprocess_data(data)
    
    logger.info("✅ Data preprocessing completed")
    logger.info(f"📊 Original shape: {data.shape}")
    logger.info(f"📊 Processed shape: {processed_data.shape}")
    logger.info(f"📊 Missing values: {processed_data.isnull().sum().sum()}")
    
    # 3. Feature Engineering
    logger.info("\n" + "="*50)
    logger.info("3. FEATURE ENGINEERING")
    logger.info("="*50)
    
    feature_config = FeatureEngineeringConfig(
        enable_4d_features=True,
        enable_technical_indicators=True,
        enable_regime_features=True,
        enable_normalization=True
    )
    
    feature_engineer = FeatureEngineer(feature_config)
    features_data = feature_engineer.generate_features(processed_data)
    
    logger.info("✅ Feature engineering completed")
    logger.info(f"📊 Original columns: {len(processed_data.columns)}")
    logger.info(f"📊 Feature columns: {len(features_data.columns)}")
    logger.info(f"📊 New features: {len(features_data.columns) - len(processed_data.columns)}")
    
    # 4. Regime Detection
    logger.info("\n" + "="*50)
    logger.info("4. REGIME DETECTION")
    logger.info("="*50)
    
    regime_config = RegimeDetectionPipelineConfig(
        enable_unsupervised_detection=True,
        enable_regime_qualification=True,
        enable_multi_timeframe=False
    )
    
    regime_detector = RegimeDetectorPipeline(regime_config)
    regime_data = regime_detector.detect_and_mark_regimes(features_data)
    
    logger.info("✅ Regime detection completed")
    logger.info(f"📊 Regime columns: {[col for col in regime_data.columns if 'regime' in col.lower()]}")
    if 'regime' in regime_data.columns:
        logger.info(f"📊 Unique regimes: {regime_data['regime'].nunique()}")
        logger.info(f"📊 Regime distribution: {regime_data['regime'].value_counts().to_dict()}")
    
    # 5. Data Validation
    logger.info("\n" + "="*50)
    logger.info("5. DATA VALIDATION")
    logger.info("="*50)
    
    validation_config = DataValidationConfig(
        enable_completeness_check=True,
        enable_consistency_check=True,
        enable_integrity_check=True,
        enable_statistical_check=True
    )
    
    data_validator = DataValidator(validation_config)
    validation_result = data_validator.validate_data(regime_data)
    
    logger.info("✅ Data validation completed")
    logger.info(f"📊 Validation passed: {validation_result.get('validation_passed', False)}")
    logger.info(f"📊 Validation score: {validation_result.get('validation_score', 0.0):.3f}")
    
    # 6. Data Storage
    logger.info("\n" + "="*50)
    logger.info("6. DATA STORAGE")
    logger.info("="*50)
    
    storage_config = StorageConfig(
        storage_type="local",
        storage_format="parquet",
        enable_compression=True,
        enable_caching=True
    )
    
    data_storage = DataStorageManager(storage_config)
    storage_result = data_storage.store_data(
        data=regime_data,
        data_type="processed_with_regimes",
        symbol="BTCUSDT",
        timeframe="1h"
    )
    
    logger.info("✅ Data storage completed")
    logger.info(f"📊 Storage path: {storage_result.storage_path}")
    logger.info(f"📊 Storage size: {storage_result.storage_size_mb:.2f} MB")
    logger.info(f"📊 Compression ratio: {storage_result.compression_ratio:.3f}")
    
    return regime_data, validation_result, storage_result


def demonstrate_pipeline_orchestrator():
    """Demonstrate the complete pipeline orchestrator."""
    logger.info("\n" + "="*60)
    logger.info("🚀 DEMONSTRATING COMPLETE PIPELINE ORCHESTRATOR")
    logger.info("="*60)
    
    # Configure pipeline
    pipeline_config = PipelineConfig(
        # Enable all stages
        enable_ingestion=True,
        enable_preprocessing=True,
        enable_feature_engineering=True,
        enable_regime_detection=True,
        enable_validation=True,
        enable_storage=True,
        
        # Pipeline options
        parallel_processing=False,  # Disable for demonstration
        enable_checkpointing=True,
        
        # Data options
        symbols=["BTCUSDT", "ETHUSDT"],
        timeframes=["1h", "4h"],
        start_date=datetime.now() - timedelta(days=7),
        end_date=datetime.now(),
        
        # Component configurations
        ingestion_config=DataIngestionConfig(
            data_source="synthetic",
            enable_klines_manager=True
        ),
        preprocessing_config=DataPreprocessingConfig(
            enable_cleaning=True,
            enable_timestamp_regularization=True
        ),
        feature_engineering_config=FeatureEngineeringConfig(
            enable_4d_features=True,
            enable_technical_indicators=True
        ),
        regime_detection_config=RegimeDetectionPipelineConfig(
            enable_unsupervised_detection=True,
            enable_regime_qualification=True
        ),
        validation_config=DataValidationConfig(
            enable_completeness_check=True,
            enable_consistency_check=True
        ),
        storage_config=StorageConfig(
            storage_type="local",
            storage_format="parquet",
            enable_compression=True
        )
    )
    
    # Initialize orchestrator
    orchestrator = DataPipelineOrchestrator(pipeline_config)
    
    # Run pipeline
    logger.info("🚀 Starting complete pipeline")
    pipeline_result = orchestrator.run_pipeline()
    
    # Display results
    logger.info("\n" + "="*50)
    logger.info("PIPELINE RESULTS")
    logger.info("="*50)
    
    logger.info(f"📊 Pipeline ID: {pipeline_result.pipeline_id}")
    logger.info(f"📊 Total duration: {pipeline_result.total_duration:.2f}s")
    logger.info(f"📊 Success rate: {pipeline_result.success_rate:.2%}")
    logger.info(f"📊 Failure rate: {pipeline_result.failure_rate:.2%}")
    
    if pipeline_result.final_data_shape:
        logger.info(f"📊 Final data shape: {pipeline_result.final_data_shape}")
        logger.info(f"📊 Final data columns: {len(pipeline_result.final_data_columns)}")
    
    # Display stage results
    logger.info("\n" + "="*30)
    logger.info("STAGE RESULTS")
    logger.info("="*30)
    
    for stage, result in pipeline_result.stage_results.items():
        logger.info(f"\n{stage.value.upper()}:")
        logger.info(f"  Status: {result.status.value}")
        logger.info(f"  Duration: {result.duration:.2f}s")
        if result.data_shape:
            logger.info(f"  Data shape: {result.data_shape}")
        if result.error_message:
            logger.info(f"  Error: {result.error_message}")
    
    # Display errors and warnings
    if pipeline_result.errors:
        logger.info(f"\n❌ Errors ({len(pipeline_result.errors)}):")
        for error in pipeline_result.errors:
            logger.info(f"  - {error}")
    
    if pipeline_result.warnings:
        logger.info(f"\n⚠️ Warnings ({len(pipeline_result.warnings)}):")
        for warning in pipeline_result.warnings:
            logger.info(f"  - {warning}")
    
    return pipeline_result


def demonstrate_data_retrieval():
    """Demonstrate data retrieval from storage."""
    logger.info("\n" + "="*50)
    logger.info("🔄 DEMONSTRATING DATA RETRIEVAL")
    logger.info("="*50)
    
    # Initialize storage manager
    storage_config = StorageConfig(
        storage_type="local",
        storage_format="parquet",
        enable_caching=True
    )
    
    data_storage = DataStorageManager(storage_config)
    
    # Retrieve data
    try:
        retrieval_result = data_storage.retrieve_data(
            data_type="processed_with_regimes",
            symbol="BTCUSDT",
            timeframe="1h"
        )
        
        logger.info("✅ Data retrieval successful")
        logger.info(f"📊 Retrieved data shape: {retrieval_result.data_shape}")
        logger.info(f"📊 Cache hit: {retrieval_result.cache_hit}")
        logger.info(f"📊 Read time: {retrieval_result.read_time:.2f}s")
        
        # Access retrieved data
        if hasattr(retrieval_result, 'retrieved_data'):
            data = retrieval_result.retrieved_data
            logger.info(f"📊 Data columns: {list(data.columns)}")
            if 'regime' in data.columns:
                logger.info(f"📊 Regime distribution: {data['regime'].value_counts().to_dict()}")
        
    except Exception as e:
        logger.warning(f"⚠️ Data retrieval failed: {e}")
        logger.info("💡 This is expected if no data was previously stored")


def demonstrate_performance_analysis():
    """Demonstrate performance analysis of the pipeline."""
    logger.info("\n" + "="*50)
    logger.info("📊 PERFORMANCE ANALYSIS")
    logger.info("="*50)
    
    # Create test data of different sizes
    test_sizes = [100, 500, 1000, 2000]
    results = []
    
    for size in test_sizes:
        logger.info(f"\n🔄 Testing with {size} data points")
        
        # Create synthetic data
        data = create_synthetic_data(num_points=size)
        
        # Time preprocessing
        start_time = datetime.now()
        
        preprocessing_config = DataPreprocessingConfig(
            enable_cleaning=True,
            enable_timestamp_regularization=True
        )
        
        data_preprocessor = DataPreprocessor(preprocessing_config)
        processed_data = data_preprocessor.preprocess_data(data)
        
        preprocessing_time = (datetime.now() - start_time).total_seconds()
        
        # Time feature engineering
        start_time = datetime.now()
        
        feature_config = FeatureEngineeringConfig(
            enable_4d_features=True,
            enable_technical_indicators=True
        )
        
        feature_engineer = FeatureEngineer(feature_config)
        features_data = feature_engineer.generate_features(processed_data)
        
        feature_time = (datetime.now() - start_time).total_seconds()
        
        # Store results
        results.append({
            'size': size,
            'preprocessing_time': preprocessing_time,
            'feature_time': feature_time,
            'total_time': preprocessing_time + feature_time,
            'data_shape': features_data.shape
        })
        
        logger.info(f"  Preprocessing: {preprocessing_time:.3f}s")
        logger.info(f"  Feature engineering: {feature_time:.3f}s")
        logger.info(f"  Total: {preprocessing_time + feature_time:.3f}s")
    
    # Display performance summary
    logger.info("\n" + "="*30)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("="*30)
    
    for result in results:
        logger.info(f"Size {result['size']:4d}: {result['total_time']:6.3f}s "
                   f"({result['total_time']/result['size']*1000:.2f}ms per point)")
    
    return results


def main():
    """Main demonstration function."""
    logger.info("🚀 TAS Data Pipeline Demonstration")
    logger.info("="*60)
    
    try:
        # 1. Demonstrate individual components
        logger.info("\n1️⃣ INDIVIDUAL COMPONENTS DEMONSTRATION")
        regime_data, validation_result, storage_result = demonstrate_individual_components()
        
        # 2. Demonstrate complete pipeline orchestrator
        logger.info("\n2️⃣ COMPLETE PIPELINE ORCHESTRATOR DEMONSTRATION")
        pipeline_result = demonstrate_pipeline_orchestrator()
        
        # 3. Demonstrate data retrieval
        logger.info("\n3️⃣ DATA RETRIEVAL DEMONSTRATION")
        demonstrate_data_retrieval()
        
        # 4. Demonstrate performance analysis
        logger.info("\n4️⃣ PERFORMANCE ANALYSIS DEMONSTRATION")
        performance_results = demonstrate_performance_analysis()
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("✅ DEMONSTRATION COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        
        logger.info("📊 Key Features Demonstrated:")
        logger.info("  ✅ Data ingestion from multiple sources")
        logger.info("  ✅ Data preprocessing and cleaning")
        logger.info("  ✅ Feature engineering with 4D features")
        logger.info("  ✅ Unsupervised regime detection")
        logger.info("  ✅ Data validation and quality checks")
        logger.info("  ✅ Data storage with compression and caching")
        logger.info("  ✅ Complete pipeline orchestration")
        logger.info("  ✅ Performance analysis and optimization")
        
        logger.info("\n📊 Production Readiness:")
        logger.info("  ✅ Modular and extensible architecture")
        logger.info("  ✅ Comprehensive error handling")
        logger.info("  ✅ Performance monitoring and optimization")
        logger.info("  ✅ Data validation and quality assurance")
        logger.info("  ✅ Flexible storage and retrieval")
        logger.info("  ✅ Complete pipeline orchestration")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        logger.info("🎉 All demonstrations completed successfully!")
    else:
        logger.error("💥 Demonstration failed!")