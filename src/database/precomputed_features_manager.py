from __future__ import annotations
from datetime import datetime
from typing import Any, Iterable
import json
from src.utils.logger import system_logger
from src.core.decorators import handles_errors
from src.utils.warning_symbols import error, failed, warning
import asyncio
import pandas as pd
import copy
import os
try:
    from src.database.influxdb_manager import InfluxDBManager
    INFLUXDB_AVAILABLE = True
except Exception:
    InfluxDBManager = None
    INFLUXDB_AVAILABLE = False

class PrecomputedFeaturesManager:
    """
    Manages precomputed features with standardized naming convention and database storage.

    Feature naming convention: {category}_{timeframe}_{name}
    Categories: candle, volatility, volume, momentum, technical, price, time,
    ml_enhanced, triple_barrier, autoencoder
    Timeframes: 1m, 5m, 15m, 30m

    Examples:
    - candle_1m_doji_present
    - volatility_5m_atr
    - momentum_15m_rsi
    - price_30m_change_pct
    - triple_barrier_1m_profit_take_hit
    - autoencoder_5m_reconstruction_error
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild('PrecomputedFeaturesManager')
        if INFLUXDB_AVAILABLE:
            self.db_manager: InfluxDBManager | None = InfluxDBManager()
        else:
            self.db_manager = None
            self.logger.warning('InfluxDB not available - features will be stored locally only')
        self.feature_categories: dict[str, str] = {'candle': 'Candlestick patterns and formations', 'volatility': 'Volatility-based indicators and regimes', 'volume': 'Volume-based analysis and flow', 'momentum': 'Momentum and oscillator indicators', 'technical': 'Technical analysis indicators', 'price': 'Price-based features and changes', 'time': 'Time-based and cyclical features', 'ml_enhanced': 'Machine learning enhanced features', 'triple_barrier': 'Triple barrier labeling results', 'autoencoder': 'Autoencoder-generated features'}
        self.timeframes: list[str] = ['1m', '5m', '15m', '30m']
        self.price_difference_features: set[str] = {'price_change', 'price_momentum', 'gap_size', 'dist_to_resistance', 'dist_to_support', 'price_return', 'log_return'}

    @handles_errors(fallback=False)
    async def initialize(self) -> bool:
        """Initialize the precomputed features manager."""
        self.logger.info('🚀 Initializing PrecomputedFeaturesManager...')
        await self._create_feature_metadata_tables()
        self.logger.info('✅ PrecomputedFeaturesManager initialized successfully')
        return True

    def generate_feature_name(self, category: str, timeframe: str, name: str) -> str:
        """
        Generate standardized feature name.

        Args:
            category: Feature category
            timeframe: Timeframe (1m, 5m, 15m, 30m)
            name: Feature name

        Returns:
            Standardized feature name
        """
        if category not in self.feature_categories:
            msg = f'Invalid category: {category}. Valid categories: {list(self.feature_categories.keys())}'
            raise ValueError(msg)
        if timeframe not in self.timeframes:
            msg = f'Invalid timeframe: {timeframe}. Valid timeframes: {self.timeframes}'
            raise ValueError(msg)
        return f'{category}_{timeframe}_{name}'

    def parse_feature_name(self, feature_name: str) -> tuple[str, str, str]:
        """
        Parse standardized feature name into components.

        Args:
            feature_name: Standardized feature name

        Returns:
            Tuple of (category, timeframe, name)
        """
        parts = feature_name.split('_', 2)
        if len(parts) != 3:
            msg = f'Invalid feature name format: {feature_name}'
            raise ValueError(msg)
        category, timeframe, name = parts
        if category not in self.feature_categories:
            msg = f'Invalid category in feature name: {category}'
            raise ValueError(msg)
        if timeframe not in self.timeframes:
            msg = f'Invalid timeframe in feature name: {timeframe}'
            raise ValueError(msg)
        return (category, timeframe, name)

    @handles_errors(fallback=False)
    async def store_features(self, features_df: pd.DataFrame, symbol: str, metadata: dict[str, Any] | None=None) -> bool:
        """
        Store precomputed features in the database.

        Args:
            features_df: DataFrame with features using standardized naming
            symbol: Trading symbol
            metadata: Additional metadata about the features

        Returns:
            Success status
        """
        if features_df.empty:
            self.logger.warning(warning('Empty features DataFrame provided'))
            return False
        self.logger.info(f'Storing {len(features_df.columns)} features for {symbol}')
        features_df = self._ensure_price_differences(features_df)
        features_df_copy = features_df.copy()
        features_df_copy['symbol'] = symbol
        features_df_copy['computation_timestamp'] = datetime.now().isoformat()
        if metadata:
            features_df_copy['metadata'] = json.dumps(metadata)
        if self.db_manager is not None:
            self.db_manager.write_api.write(bucket=self.db_manager.bucket, record=features_df_copy, data_frame_measurement_name='precomputed_features', data_frame_tag_columns=['symbol'])
        await self._store_feature_metadata(features_df.columns.tolist(), symbol, metadata)
        self.logger.info(f'✅ Successfully stored features for {symbol}')
        return True

    @handles_errors(fallback=pd.DataFrame())
    async def retrieve_features(self, symbol: str, feature_names: list[str] | None=None, start_time: str | None=None, end_time: str | None=None, category_filter: str | None=None, timeframe_filter: str | None=None) -> pd.DataFrame:
        """
        Retrieve precomputed features from the database.

        Args:
            symbol: Trading symbol
            feature_names: Specific feature names to retrieve
            start_time: Start time for data retrieval
            end_time: End time for data retrieval
            category_filter: Filter by feature category
            timeframe_filter: Filter by timeframe

        Returns:
            DataFrame with requested features
        """
        if self.db_manager is None:
            self.logger.warning(warning('InfluxDB not available; cannot retrieve features'))
            return pd.DataFrame()
        query_filters = [f'r["symbol"] == "{symbol}"']
        if feature_names:
            field_filter = ' or '.join([f'r["_field"] == "{name}"' for name in feature_names])
            query_filters.append(f'({field_filter})')
        time_range = ''
        if start_time and end_time:
            time_range = f'|> range(start: {start_time}, stop: {end_time})'
        elif start_time:
            time_range = f'|> range(start: {start_time})'
        elif end_time:
            time_range = f'|> range(stop: {end_time})'
        query = f'''\n        from(bucket: "{self.db_manager.bucket}")\n          {time_range}\n          |> filter(fn: (r) => r["_measurement"] == "precomputed_features")\n          |> filter(fn: (r) => {' and '.join(query_filters)})\n          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")\n        '''
        df = self.db_manager.query_api.query_data_frame(query, org=self.db_manager.org)
        if isinstance(df, list):
            if not df:
                return pd.DataFrame()
            df = pd.concat(df, ignore_index=True)
        if df.empty:
            return pd.DataFrame()
        if category_filter or timeframe_filter:
            df = self._apply_feature_filters(df, category_filter, timeframe_filter)
        if '_time' in df.columns:
            df['_time'] = pd.to_datetime(df['_time'])
            df = df.set_index('_time')
        self.logger.info(f'Retrieved {len(df)} rows with {len(df.columns)} features for {symbol}')
        return df

    def _ensure_price_differences(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure price-based features use differences rather than absolute values.

        Args:
            df: Features DataFrame

        Returns:
            DataFrame with price differences applied
        """
        df_copy = df.copy()
        for col in df_copy.columns:
            try:
                category, timeframe, name = self.parse_feature_name(col)
            except ValueError:
                continue
            if category == 'price' and any((price_feat in name for price_feat in self.price_difference_features)):
                continue
            if category == 'price' and any((abs_feat in name for abs_feat in ['open', 'high', 'low', 'close'])):
                if name.endswith(('_close', '_open')):
                    df_copy[col] = df_copy[col].pct_change()
                elif name.endswith(('_high', '_low')):
                    close_col = col.replace(name.split('_')[-1], 'close')
                    if close_col in df_copy.columns:
                        df_copy[col] = (df_copy[col] - df_copy[close_col]) / df_copy[close_col]
                    else:
                        df_copy[col] = df_copy[col].pct_change()
        return df_copy.fillna(0)

    def _apply_feature_filters(self, df: pd.DataFrame, category_filter: str | None=None, timeframe_filter: str | None=None) -> pd.DataFrame:
        """Apply category and timeframe filters to the DataFrame."""
        filtered_columns: list[str] = []
        for col in df.columns:
            try:
                category, timeframe, name = self.parse_feature_name(col)
            except ValueError:
                filtered_columns.append(col)
                continue
            if category_filter and category != category_filter:
                continue
            if timeframe_filter and timeframe != timeframe_filter:
                continue
            filtered_columns.append(col)
        return df[filtered_columns]

    async def _create_feature_metadata_tables(self) -> None:
        """Create tables for storing feature metadata."""
        self.logger.info('Feature metadata storage configured')

    async def _store_feature_metadata(self, feature_names: list[str], symbol: str, metadata: dict[str, Any] | None=None) -> None:
        """Store metadata about the features."""
        if self.db_manager is None:
            return
        metadata_records: list[dict[str, Any]] = []
        for feature_name in feature_names:
            try:
                category, timeframe, name = self.parse_feature_name(feature_name)
            except ValueError:
                continue
            record: dict[str, Any] = {'feature_name': feature_name, 'category': category, 'timeframe': timeframe, 'name': name, 'symbol': symbol, 'created_at': datetime.now().isoformat(), 'description': self.feature_categories.get(category, 'Unknown category')}
            if metadata:
                record.update(metadata)
            metadata_records.append(record)
        if not metadata_records:
            return
        metadata_df = pd.DataFrame(metadata_records)
        metadata_df['timestamp'] = datetime.now()
        metadata_df = metadata_df.set_index('timestamp')
        self.db_manager.write_api.write(bucket=self.db_manager.bucket, record=metadata_df, data_frame_measurement_name='feature_metadata', data_frame_tag_columns=['symbol', 'category', 'timeframe'])

    def get_available_features(self, category: str | None=None, timeframe: str | None=None) -> list[str]:
        """
        Get list of available feature names based on filters.

        Args:
            category: Filter by category
            timeframe: Filter by timeframe

        Returns:
            List of available feature names
        """
        categories = [category] if category else list(self.feature_categories.keys())
        timeframes = [timeframe] if timeframe else self.timeframes
        example_features: list[str] = []
        for cat in categories:
            for tf in timeframes:
                if cat == 'candle':
                    example_features.extend([f'{cat}_{tf}_doji_present', f'{cat}_{tf}_hammer_present', f'{cat}_{tf}_engulfing_bullish'])
                elif cat == 'volatility':
                    example_features.extend([f'{cat}_{tf}_atr', f'{cat}_{tf}_volatility_regime', f'{cat}_{tf}_vol_ratio'])
                elif cat == 'momentum':
                    example_features.extend([f'{cat}_{tf}_rsi', f'{cat}_{tf}_macd_signal', f'{cat}_{tf}_stoch_k'])
                elif cat == 'triple_barrier':
                    example_features.extend([f'{cat}_{tf}_profit_take_hit', f'{cat}_{tf}_stop_loss_hit', f'{cat}_{tf}_time_barrier_hit'])
                elif cat == 'autoencoder':
                    example_features.extend([f'{cat}_{tf}_reconstruction_error', f'{cat}_{tf}_latent_feature_1', f'{cat}_{tf}_latent_feature_2'])
        return example_features

    def get_feature_statistics(self) -> dict[str, Any]:
        """Get statistics about stored features."""
        return {'categories': self.feature_categories, 'timeframes': self.timeframes, 'total_feature_types': len(self.feature_categories) * len(self.timeframes), 'price_difference_features': list(self.price_difference_features)}