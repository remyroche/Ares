"""
Pipeline Quality Integration - Example Implementation

This module demonstrates how to integrate quality verification into existing pipeline steps
for both aggtrades and klines data, ensuring quality checks at data collection completion
and at the beginning of each stage.

Key Features Demonstrated:
- Data collection completion quality verification
- Stage beginning quality verification
- Quality gate enforcement
- Configuration management
- Integration with existing pipeline steps
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Any

from .pipeline_quality_integration import (
    get_quality_integration,
    verify_data_collection_quality,
    verify_stage_beginning_quality,
    enforce_quality_gate
)
from .unified_quality_verification import DataType, VerificationStage

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExampleDataCollectionStep:
    """Example data collection step with quality verification."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logger.getChild('DataCollection')
    
    @verify_data_collection_quality("binance", "BTCUSDT", DataType.AGGRADES)
    async def collect_aggtrades_data(self, exchange: str, symbol: str, 
                                   start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Collect aggtrades data with automatic quality verification.
        
        Args:
            exchange: Exchange name
            symbol: Symbol name
            start_time: Start time for data collection
            end_time: End time for data collection
            
        Returns:
            Collected aggtrades data
        """
        self.logger.info(f"📊 Collecting aggtrades data for {exchange}_{symbol}")
        
        # Simulate data collection
        data = self._simulate_aggtrades_collection(start_time, end_time)
        
        self.logger.info(f"✅ Collected {len(data)} aggtrades records")
        return data
    
    @verify_data_collection_quality("binance", "BTCUSDT", DataType.KLINES)
    async def collect_klines_data(self, exchange: str, symbol: str, 
                                timeframe: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Collect klines data with automatic quality verification.
        
        Args:
            exchange: Exchange name
            symbol: Symbol name
            timeframe: Timeframe (1m, 5m, etc.)
            start_time: Start time for data collection
            end_time: End time for data collection
            
        Returns:
            Collected klines data
        """
        self.logger.info(f"📊 Collecting klines data for {exchange}_{symbol} ({timeframe})")
        
        # Simulate data collection
        data = self._simulate_klines_collection(timeframe, start_time, end_time)
        
        self.logger.info(f"✅ Collected {len(data)} klines records")
        return data
    
    def _simulate_aggtrades_collection(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Simulate aggtrades data collection with some quality issues."""
        # Generate timestamps
        timestamps = []
        current_time = start_time
        while current_time < end_time:
            timestamps.append(current_time)
            current_time += timedelta(milliseconds=np.random.randint(50, 200))
        
        # Generate data with some quality issues
        data = pd.DataFrame({
            'timestamp': timestamps,
            'price': np.random.uniform(45000, 55000, len(timestamps)),
            'quantity': np.random.exponential(1.0, len(timestamps)),
            'first_trade_id': range(1000, 1000 + len(timestamps)),
            'last_trade_id': range(1000, 1000 + len(timestamps)),
            'is_buyer_maker': np.random.choice([True, False], len(timestamps))
        })
        
        # Introduce some quality issues for testing
        if len(data) > 10:
            # Add some timestamp gaps
            gap_indices = np.random.choice(len(data), size=3, replace=False)
            for idx in gap_indices:
                data.loc[idx, 'timestamp'] += timedelta(seconds=2)
            
            # Add some duplicates
            duplicate_indices = np.random.choice(len(data), size=2, replace=False)
            for idx in duplicate_indices:
                data.loc[idx] = data.iloc[0]
            
            # Add some negative prices
            negative_indices = np.random.choice(len(data), size=1, replace=False)
            for idx in negative_indices:
                data.loc[idx, 'price'] = -100.0
        
        return data
    
    def _simulate_klines_collection(self, timeframe: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Simulate klines data collection with some quality issues."""
        # Calculate timeframe in minutes
        timeframe_minutes = int(timeframe.replace('m', '').replace('h', '00').replace('d', '0000'))
        
        # Generate timestamps
        timestamps = []
        current_time = start_time
        while current_time < end_time:
            timestamps.append(current_time)
            current_time += timedelta(minutes=timeframe_minutes)
        
        # Generate OHLCV data
        base_price = 50000
        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': base_price + np.random.normal(0, 100, len(timestamps)),
            'high': base_price + np.random.normal(0, 100, len(timestamps)) + np.random.uniform(0, 200, len(timestamps)),
            'low': base_price + np.random.normal(0, 100, len(timestamps)) - np.random.uniform(0, 200, len(timestamps)),
            'close': base_price + np.random.normal(0, 100, len(timestamps)),
            'volume': np.random.exponential(10.0, len(timestamps))
        })
        
        # Ensure OHLC consistency
        for i in range(len(data)):
            open_price = data.loc[i, 'open']
            close_price = data.loc[i, 'close']
            data.loc[i, 'high'] = max(data.loc[i, 'high'], open_price, close_price)
            data.loc[i, 'low'] = min(data.loc[i, 'low'], open_price, close_price)
        
        # Introduce some quality issues for testing
        if len(data) > 5:
            # Add some timestamp gaps
            gap_indices = np.random.choice(len(data), size=2, replace=False)
            for idx in gap_indices:
                data.loc[idx, 'timestamp'] += timedelta(minutes=timeframe_minutes * 2)
            
            # Add some OHLC inconsistencies
            inconsistent_indices = np.random.choice(len(data), size=1, replace=False)
            for idx in inconsistent_indices:
                data.loc[idx, 'high'] = data.loc[idx, 'low'] - 100  # Invalid: high < low
        
        return data


class ExamplePreprocessingStep:
    """Example preprocessing step with quality verification."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logger.getChild('Preprocessing')
    
    @verify_stage_beginning_quality("preprocessing")
    async def preprocess_aggtrades_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess aggtrades data with automatic quality verification.
        
        Args:
            data: Raw aggtrades data
            
        Returns:
            Preprocessed aggtrades data
        """
        self.logger.info(f"🔧 Preprocessing aggtrades data ({len(data)} rows)")
        
        # Simulate preprocessing
        processed_data = data.copy()
        
        # Sort by timestamp
        processed_data = processed_data.sort_values('timestamp').reset_index(drop=True)
        
        # Remove obvious outliers (prices > 2x median)
        median_price = processed_data['price'].median()
        processed_data = processed_data[processed_data['price'] <= median_price * 2]
        
        self.logger.info(f"✅ Preprocessed aggtrades data ({len(processed_data)} rows)")
        return processed_data
    
    @verify_stage_beginning_quality("preprocessing")
    async def preprocess_klines_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess klines data with automatic quality verification.
        
        Args:
            data: Raw klines data
            
        Returns:
            Preprocessed klines data
        """
        self.logger.info(f"🔧 Preprocessing klines data ({len(data)} rows)")
        
        # Simulate preprocessing
        processed_data = data.copy()
        
        # Sort by timestamp
        processed_data = processed_data.sort_values('timestamp').reset_index(drop=True)
        
        # Fix OHLC inconsistencies
        for i in range(len(processed_data)):
            row = processed_data.iloc[i]
            open_price = row['open']
            close_price = row['close']
            high_price = row['high']
            low_price = row['low']
            
            # Ensure high >= max(open, close)
            processed_data.loc[i, 'high'] = max(high_price, open_price, close_price)
            # Ensure low <= min(open, close)
            processed_data.loc[i, 'low'] = min(low_price, open_price, close_price)
        
        self.logger.info(f"✅ Preprocessed klines data ({len(processed_data)} rows)")
        return processed_data


class ExampleFeatureEngineeringStep:
    """Example feature engineering step with quality verification."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logger.getChild('FeatureEngineering')
    
    @verify_stage_beginning_quality("feature_engineering")
    async def engineer_aggtrades_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from aggtrades data with automatic quality verification.
        
        Args:
            data: Preprocessed aggtrades data
            
        Returns:
            Data with engineered features
        """
        self.logger.info(f"⚙️ Engineering aggtrades features ({len(data)} rows)")
        
        # Simulate feature engineering
        features_data = data.copy()
        
        # Add some basic features
        features_data['price_change'] = features_data['price'].pct_change()
        features_data['volume_weighted_price'] = features_data['price'] * features_data['quantity']
        features_data['buy_volume'] = features_data['quantity'] * features_data['is_buyer_maker'].astype(int)
        features_data['sell_volume'] = features_data['quantity'] * (~features_data['is_buyer_maker']).astype(int)
        
        self.logger.info(f"✅ Engineered aggtrades features ({len(features_data)} rows, {len(features_data.columns)} columns)")
        return features_data
    
    @verify_stage_beginning_quality("feature_engineering")
    async def engineer_klines_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from klines data with automatic quality verification.
        
        Args:
            data: Preprocessed klines data
            
        Returns:
            Data with engineered features
        """
        self.logger.info(f"⚙️ Engineering klines features ({len(data)} rows)")
        
        # Simulate feature engineering
        features_data = data.copy()
        
        # Add some basic features
        features_data['price_range'] = features_data['high'] - features_data['low']
        features_data['body_size'] = abs(features_data['close'] - features_data['open'])
        features_data['upper_shadow'] = features_data['high'] - features_data[['open', 'close']].max(axis=1)
        features_data['lower_shadow'] = features_data[['open', 'close']].min(axis=1) - features_data['low']
        features_data['price_change'] = features_data['close'].pct_change()
        features_data['volume_change'] = features_data['volume'].pct_change()
        
        self.logger.info(f"✅ Engineered klines features ({len(features_data)} rows, {len(features_data.columns)} columns)")
        return features_data


class ExampleModelTrainingStep:
    """Example model training step with quality gate enforcement."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logger.getChild('ModelTraining')
    
    @enforce_quality_gate(0.8, "model_training")
    async def train_model(self, data: pd.DataFrame, model_type: str = "regression") -> Dict[str, Any]:
        """
        Train model with quality gate enforcement.
        
        Args:
            data: Feature-engineered data
            model_type: Type of model to train
            
        Returns:
            Trained model information
        """
        self.logger.info(f"🤖 Training {model_type} model ({len(data)} rows, {len(data.columns)} features)")
        
        # Simulate model training
        model_info = {
            'model_type': model_type,
            'training_samples': len(data),
            'features_count': len(data.columns),
            'training_time': np.random.uniform(10, 60),  # seconds
            'model_accuracy': np.random.uniform(0.7, 0.95),
            'created_at': datetime.now().isoformat()
        }
        
        self.logger.info(f"✅ Model training completed (accuracy: {model_info['model_accuracy']:.3f})")
        return model_info


class ExamplePipeline:
    """Example complete pipeline with quality verification."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logger.getChild('Pipeline')
        
        # Initialize steps
        self.data_collection = ExampleDataCollectionStep(config)
        self.preprocessing = ExamplePreprocessingStep(config)
        self.feature_engineering = ExampleFeatureEngineeringStep(config)
        self.model_training = ExampleModelTrainingStep(config)
        
        # Get quality integration
        self.quality_integration = get_quality_integration(config)
    
    async def run_aggtrades_pipeline(self, exchange: str, symbol: str, 
                                   start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """
        Run complete aggtrades pipeline with quality verification.
        
        Args:
            exchange: Exchange name
            symbol: Symbol name
            start_time: Start time for data collection
            end_time: End time for data collection
            
        Returns:
            Pipeline results
        """
        self.logger.info(f"🚀 Starting aggtrades pipeline for {exchange}_{symbol}")
        
        try:
            # Step 1: Data collection with quality verification
            raw_data = await self.data_collection.collect_aggtrades_data(
                exchange, symbol, start_time, end_time
            )
            
            # Step 2: Preprocessing with quality verification
            processed_data = await self.preprocessing.preprocess_aggtrades_data(raw_data)
            
            # Step 3: Feature engineering with quality verification
            features_data = await self.feature_engineering.engineer_aggtrades_features(processed_data)
            
            # Step 4: Model training with quality gate
            model_info = await self.model_training.train_model(features_data, "aggtrades_regression")
            
            # Get quality verification summary
            quality_summary = self.quality_integration.get_verification_summary()
            
            results = {
                'pipeline_type': 'aggtrades',
                'exchange': exchange,
                'symbol': symbol,
                'data_rows': len(features_data),
                'model_info': model_info,
                'quality_summary': quality_summary,
                'success': True
            }
            
            self.logger.info(f"✅ Aggtrades pipeline completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Aggtrades pipeline failed: {e}")
            return {
                'pipeline_type': 'aggtrades',
                'exchange': exchange,
                'symbol': symbol,
                'success': False,
                'error': str(e)
            }
    
    async def run_klines_pipeline(self, exchange: str, symbol: str, timeframe: str,
                                start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """
        Run complete klines pipeline with quality verification.
        
        Args:
            exchange: Exchange name
            symbol: Symbol name
            timeframe: Timeframe (1m, 5m, etc.)
            start_time: Start time for data collection
            end_time: End time for data collection
            
        Returns:
            Pipeline results
        """
        self.logger.info(f"🚀 Starting klines pipeline for {exchange}_{symbol} ({timeframe})")
        
        try:
            # Step 1: Data collection with quality verification
            raw_data = await self.data_collection.collect_klines_data(
                exchange, symbol, timeframe, start_time, end_time
            )
            
            # Step 2: Preprocessing with quality verification
            processed_data = await self.preprocessing.preprocess_klines_data(raw_data)
            
            # Step 3: Feature engineering with quality verification
            features_data = await self.feature_engineering.engineer_klines_features(processed_data)
            
            # Step 4: Model training with quality gate
            model_info = await self.model_training.train_model(features_data, "klines_classification")
            
            # Get quality verification summary
            quality_summary = self.quality_integration.get_verification_summary()
            
            results = {
                'pipeline_type': 'klines',
                'exchange': exchange,
                'symbol': symbol,
                'timeframe': timeframe,
                'data_rows': len(features_data),
                'model_info': model_info,
                'quality_summary': quality_summary,
                'success': True
            }
            
            self.logger.info(f"✅ Klines pipeline completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Klines pipeline failed: {e}")
            return {
                'pipeline_type': 'klines',
                'exchange': exchange,
                'symbol': symbol,
                'timeframe': timeframe,
                'success': False,
                'error': str(e)
            }


async def run_example_pipelines():
    """Run example pipelines to demonstrate quality verification."""
    logger.info("🚀 Starting example pipeline demonstrations")
    
    # Configuration
    config = {
        'enable_auto_verification': True,
        'auto_fix_enabled': True,
        'export_reports': True,
        'reports_directory': 'reports/quality_examples'
    }
    
    # Initialize pipeline
    pipeline = ExamplePipeline(config)
    
    # Define time range
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=1)
    
    # Run aggtrades pipeline
    logger.info("=" * 60)
    logger.info("RUNNING AGGTRADES PIPELINE")
    logger.info("=" * 60)
    
    aggtrades_results = await pipeline.run_aggtrades_pipeline(
        "binance", "BTCUSDT", start_time, end_time
    )
    
    logger.info(f"Aggtrades pipeline results: {aggtrades_results['success']}")
    if aggtrades_results['success']:
        logger.info(f"  Data rows: {aggtrades_results['data_rows']}")
        logger.info(f"  Model accuracy: {aggtrades_results['model_info']['model_accuracy']:.3f}")
        logger.info(f"  Quality verifications: {aggtrades_results['quality_summary']['total_verifications']}")
    
    # Run klines pipeline
    logger.info("=" * 60)
    logger.info("RUNNING KLINES PIPELINE")
    logger.info("=" * 60)
    
    klines_results = await pipeline.run_klines_pipeline(
        "binance", "BTCUSDT", "1m", start_time, end_time
    )
    
    logger.info(f"Klines pipeline results: {klines_results['success']}")
    if klines_results['success']:
        logger.info(f"  Data rows: {klines_results['data_rows']}")
        logger.info(f"  Model accuracy: {klines_results['model_info']['model_accuracy']:.3f}")
        logger.info(f"  Quality verifications: {klines_results['quality_summary']['total_verifications']}")
    
    # Export quality verification summary
    quality_integration = get_quality_integration()
    quality_integration.export_verification_summary('reports/quality_examples/verification_summary.json')
    
    logger.info("✅ Example pipeline demonstrations completed")


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_example_pipelines())