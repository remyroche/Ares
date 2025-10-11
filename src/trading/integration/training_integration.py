"""
Training Integration

Integration utilities for connecting trading operations
with the training pipeline for data synchronization and model updates.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success
from ..utils.error_handling import TradingError, TradingErrorSeverity, trading_error_handler
from ..utils.validation import validate_market_data

logger = system_logger.getChild('TrainingIntegration')

class TrainingDataProvider:
    """
    Provides training pipeline features and data to trading operations.
    """
    
    def __init__(self):
        self.logger = logger.getChild('TrainingDataProvider')
        self.feature_cache: Dict[str, Any] = {}
        self.last_update: Optional[datetime] = None
        
    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.MEDIUM,
        raise_on_error=True
    )
    async def get_training_features(
        self,
        market_data: pd.DataFrame,
        feature_set: str = "default",
        symbol: str = "ETHUSDT"
    ) -> pd.DataFrame:
        """
        Get features using the same feature engineering from training pipeline.
        
        Args:
            market_data: Market data DataFrame
            feature_set: Feature set name from training pipeline
            symbol: Trading symbol
            
        Returns:
            DataFrame with engineered features
        """
        tprint_info(f"🔄 Generating training features for {symbol} (set: {feature_set})")
        
        try:
            # Validate input data
            validate_market_data(market_data, min_rows=50)
            
            # Import feature engineering from training pipeline
            features_df = await self._apply_training_feature_engineering(
                market_data, feature_set, symbol
            )
            
            tprint_success(f"✅ Generated {len(features_df.columns)} training features")
            return features_df
            
        except Exception as e:
            raise TradingError(
                f"Failed to get training features: {e}",
                error_code="TRAINING_FEATURES_ERROR",
                severity=TradingErrorSeverity.MEDIUM,
                context={
                    'feature_set': feature_set,
                    'symbol': symbol,
                    'data_shape': market_data.shape
                }
            )
    
    async def _apply_training_feature_engineering(
        self,
        market_data: pd.DataFrame,
        feature_set: str,
        symbol: str
    ) -> pd.DataFrame:
        """Apply feature engineering from training pipeline."""
        try:
            # Import feature engineering components from training pipeline
            from src.feature_generation.utils.feature_engineering_orchestrator import FeatureEngineeringOrchestrator
            
            # Create feature engineering orchestrator
            feature_orchestrator = FeatureEngineeringOrchestrator(
                symbol=symbol,
                exchange="binance",
                timeframe="1m"
            )
            
            # Initialize orchestrator
            await feature_orchestrator.initialize()
            
            # Apply feature engineering
            features_df = await feature_orchestrator.engineer_features(market_data)
            
            return features_df
            
        except ImportError as e:
            tprint_warning(f"⚠️ Training feature engineering not available: {e}")
            # Fallback to basic features
            return self._create_basic_features(market_data)
        
        except Exception as e:
            tprint_warning(f"⚠️ Training feature engineering failed: {e}")
            # Fallback to basic features
            return self._create_basic_features(market_data)
    
    def _create_basic_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Create basic features as fallback."""
        try:
            features_df = market_data.copy()
            
            # Basic technical indicators
            if 'close' in features_df.columns:
                # Simple moving averages
                features_df['sma_5'] = features_df['close'].rolling(5).mean()
                features_df['sma_20'] = features_df['close'].rolling(20).mean()
                features_df['sma_50'] = features_df['close'].rolling(50).mean()
                
                # Price ratios
                features_df['price_sma5_ratio'] = features_df['close'] / features_df['sma_5']
                features_df['price_sma20_ratio'] = features_df['close'] / features_df['sma_20']
                
                # Returns
                features_df['returns_1'] = features_df['close'].pct_change(1)
                features_df['returns_5'] = features_df['close'].pct_change(5)
                features_df['returns_20'] = features_df['close'].pct_change(20)
                
                # Volatility
                features_df['volatility_5'] = features_df['returns_1'].rolling(5).std()
                features_df['volatility_20'] = features_df['returns_1'].rolling(20).std()
            
            # Volume features
            if 'volume' in features_df.columns:
                features_df['volume_sma_20'] = features_df['volume'].rolling(20).mean()
                features_df['volume_ratio'] = features_df['volume'] / features_df['volume_sma_20']
            
            tprint_info("📊 Created basic fallback features")
            return features_df
            
        except Exception as e:
            tprint_error(f"❌ Basic feature creation failed: {e}")
            return market_data
    
    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.LOW,
        raise_on_error=False
    )
    async def sync_with_training_pipeline(
        self,
        trading_data: Dict[str, Any]
    ) -> bool:
        """
        Sync trading data with training pipeline for model updates.
        
        Args:
            trading_data: Trading performance and decision data
            
        Returns:
            True if sync successful
        """
        tprint_info("🔄 Syncing trading data with training pipeline...")
        
        try:
            # Export trading data for training pipeline consumption
            await self._export_trading_performance(trading_data)
            
            # Update feature cache if needed
            await self._update_feature_cache()
            
            # Check for model updates
            await self._check_model_updates()
            
            self.last_update = datetime.now()
            
            tprint_success("✅ Successfully synced with training pipeline")
            return True
            
        except Exception as e:
            tprint_warning(f"⚠️ Training pipeline sync failed: {e}")
            return False
    
    async def _export_trading_performance(self, trading_data: Dict[str, Any]):
        """Export trading performance data for training pipeline."""
        try:
            # Prepare data for training pipeline
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'trading_performance': trading_data,
                'export_type': 'trading_performance'
            }
            
            # Save to training pipeline data directory
            import json
            import os

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
            
            export_dir = "data_cache/training_sync"
            os.makedirs(export_dir, exist_ok=True)
            
            filename = f"trading_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(export_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            tprint_info(f"📤 Exported trading performance to {filepath}")
            
        except Exception as e:
            tprint_warning(f"⚠️ Trading performance export failed: {e}")
    
    async def _update_feature_cache(self):
        """Update feature cache from training pipeline."""
        try:
            # This would check for updated feature definitions
            # from the training pipeline
            tprint_info("🔄 Checking for feature updates...")
            
            # Placeholder for feature cache update logic
            
        except Exception as e:
            tprint_warning(f"⚠️ Feature cache update failed: {e}")
    
    async def _check_model_updates(self):
        """Check for updated models from training pipeline."""
        try:
            # This would check for newly trained models
            # that should be loaded into trading
            tprint_info("🔄 Checking for model updates...")
            
            # Placeholder for model update check logic
            
        except Exception as e:
            tprint_warning(f"⚠️ Model update check failed: {e}")

class TradingDataExporter:
    """
    Exports trading data for use in training pipeline.
    """
    
    def __init__(self):
        self.logger = logger.getChild('TradingDataExporter')
        
    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.LOW,
        raise_on_error=False
    )
    async def export_trading_data(
        self,
        trading_decisions: List[Dict[str, Any]],
        performance_metrics: Dict[str, Any],
        market_data: pd.DataFrame,
        export_format: str = "parquet"
    ) -> bool:
        """
        Export trading data for training pipeline consumption.
        
        Args:
            trading_decisions: List of trading decisions
            performance_metrics: Performance metrics
            market_data: Market data used
            export_format: Export format ('parquet', 'json', 'csv')
            
        Returns:
            True if export successful
        """
        tprint_info("📤 Exporting trading data for training pipeline...")
        
        try:
            # Prepare export directory
            export_dir = "data_cache/training_export"
            os.makedirs(export_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Export trading decisions
            decisions_df = pd.DataFrame(trading_decisions)
            if not decisions_df.empty:
                if export_format == "parquet":
                    decisions_file = os.path.join(export_dir, f"trading_decisions_{timestamp}.parquet")
                    decisions_df.to_parquet(decisions_file, index=False)
                elif export_format == "csv":
                    decisions_file = os.path.join(export_dir, f"trading_decisions_{timestamp}.csv")
                    decisions_df.to_csv(decisions_file, index=False)
                
                tprint_success(f"✅ Exported trading decisions to {decisions_file}")
            
            # Export performance metrics
            metrics_file = os.path.join(export_dir, f"performance_metrics_{timestamp}.json")
            with open(metrics_file, 'w') as f:
                json.dump(performance_metrics, f, indent=2, default=str)
            
            tprint_success(f"✅ Exported performance metrics to {metrics_file}")
            
            # Export market data
            if not market_data.empty:
                if export_format == "parquet":
                    market_file = os.path.join(export_dir, f"market_data_{timestamp}.parquet")
                    market_data.to_parquet(market_file, index=False)
                elif export_format == "csv":
                    market_file = os.path.join(export_dir, f"market_data_{timestamp}.csv")
                    market_data.to_csv(market_file, index=False)
                
                tprint_success(f"✅ Exported market data to {market_file}")
            
            return True
            
        except Exception as e:
            tprint_error(f"❌ Trading data export failed: {e}")
            return False
    
    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.LOW,
        raise_on_error=False
    )
    async def prepare_training_data(
        self,
        trading_history: List[Dict[str, Any]],
        feature_data: pd.DataFrame,
        target_column: str = "success"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare trading data for training pipeline consumption.
        
        Args:
            trading_history: Historical trading data
            feature_data: Feature data DataFrame
            target_column: Target column name
            
        Returns:
            Tuple of (features_df, targets_series)
        """
        tprint_info("🔄 Preparing trading data for training pipeline...")
        
        try:
            # Convert trading history to DataFrame
            history_df = pd.DataFrame(trading_history)
            
            if history_df.empty or feature_data.empty:
                tprint_warning("⚠️ Empty data provided for training preparation")
                return pd.DataFrame(), pd.Series()
            
            # Align data by timestamp
            if 'timestamp' in history_df.columns and 'timestamp' in feature_data.columns:
                # Convert timestamps
                history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
                feature_data['timestamp'] = pd.to_datetime(feature_data['timestamp'])
                
                # Merge on timestamp
                merged_df = pd.merge(feature_data, history_df, on='timestamp', how='inner')
            else:
                # If no timestamps, assume aligned by index
                merged_df = pd.concat([feature_data, history_df], axis=1)
            
            # Separate features and targets
            feature_columns = [col for col in merged_df.columns 
                             if col not in ['timestamp', target_column] and 
                             not col.startswith('trade_') and
                             not col.startswith('decision_')]
            
            features_df = merged_df[feature_columns]
            
            if target_column in merged_df.columns:
                targets_series = merged_df[target_column]
            else:
                # Create default target based on PnL
                if 'pnl' in merged_df.columns:
                    targets_series = (merged_df['pnl'] > 0).astype(int)
                else:
                    tprint_warning(f"⚠️ Target column {target_column} not found, creating dummy targets")
                    targets_series = pd.Series([0] * len(features_df))
            
            # Remove any remaining NaN values
            mask = ~(features_df.isna().any(axis=1) | targets_series.isna())
            features_df = features_df[mask]
            targets_series = targets_series[mask]
            
            tprint_success(f"✅ Prepared {len(features_df)} samples with {len(feature_columns)} features")
            
            return features_df, targets_series
            
        except Exception as e:
            tprint_error(f"❌ Training data preparation failed: {e}")
            return pd.DataFrame(), pd.Series()

# Global instances
training_data_provider = TrainingDataProvider()
trading_data_exporter = TradingDataExporter()

# Convenience functions
async def get_training_features(
    market_data: pd.DataFrame,
    feature_set: str = "default",
    symbol: str = "ETHUSDT"
) -> pd.DataFrame:
    """Get features using training pipeline feature engineering."""
    return await training_data_provider.get_training_features(market_data, feature_set, symbol)

async def sync_with_training_pipeline(trading_data: Dict[str, Any]) -> bool:
    """Sync trading data with training pipeline."""
    return await training_data_provider.sync_with_training_pipeline(trading_data)

async def export_trading_data(
    trading_decisions: List[Dict[str, Any]],
    performance_metrics: Dict[str, Any],
    market_data: pd.DataFrame,
    export_format: str = "parquet"
) -> bool:
    """Export trading data for training pipeline."""
    return await trading_data_exporter.export_trading_data(
        trading_decisions, performance_metrics, market_data, export_format
    )

async def prepare_training_data(
    trading_history: List[Dict[str, Any]],
    feature_data: pd.DataFrame,
    target_column: str = "success"
) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare trading data for training pipeline."""
    return await trading_data_exporter.prepare_training_data(
        trading_history, feature_data, target_column
    )

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
