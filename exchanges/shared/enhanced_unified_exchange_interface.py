"""
Enhanced Unified Exchange Interface

This module provides a comprehensive, unified interface for all exchanges that ensures
complete equivalency in OHLCV data format and full compatibility with src/utils/data/ utilities.

Features:
- Single interface for all exchanges (binance, bingx, okx, mexc)
- Complete data equivalency across all exchanges
- Full integration with src/utils/data/ processing pipeline
- Comprehensive error handling and validation
- Memory-efficient data processing
- Backward compatibility with existing systems
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import our unified standardizer
from .unified_exchange_standardizer import (
    UnifiedExchangeStandardizer, StandardizedOHLCVData, ExchangeType, DataQualityLevel,
    standardize_exchange_ohlcv, validate_ohlcv_equivalency
)

# Import src/utils/data utilities
from src.utils.data import (
    DataProcessor, DataQualityFramework, DataCleaner,
    validate_and_fix_data_quality, optimize_dataframe_dtypes,
    check_dataframe_health, regularize_timestamps
)
from src.utils.logger import system_logger

logger = logging.getLogger(__name__)


@dataclass
class ExchangeConfig:
    """Configuration for exchange connections"""
    name: str
    api_key: str
    api_secret: str
    base_url: str
    sandbox: bool = False
    rate_limits: Dict[str, int] = None
    timeout: int = 30
    max_retries: int = 3


class IEnhancedUnifiedExchange(ABC):
    """
    Enhanced unified exchange interface that all exchanges must implement.
    
    This interface ensures complete equivalency in data format across all exchanges.
    """
    
    @abstractmethod
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Get standardized klines data.
        
        Returns:
            DataFrame with standardized OHLCV data compatible with src/utils/data/
        """
        pass
    
    @abstractmethod
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get ticker information"""
        pass
    
    @abstractmethod
    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Get order book data"""
        pass
    
    @abstractmethod
    async def get_trades(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Get recent trades data"""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        pass
    
    @abstractmethod
    async def get_balance(self, currency: str = None) -> Dict[str, Any]:
        """Get balance information"""
        pass


class EnhancedUnifiedExchangeAdapter:
    """
    Enhanced unified adapter that wraps individual exchange implementations
    and ensures complete data equivalency.
    """
    
    def __init__(self, exchange_instance: Any, exchange_type: ExchangeType, quality_level: DataQualityLevel = DataQualityLevel.STANDARD):
        """
        Initialize the enhanced unified adapter.
        
        Args:
            exchange_instance: The actual exchange implementation
            exchange_type: Type of exchange (binance, bingx, okx, mexc)
            quality_level: Data quality validation level
        """
        self.exchange_instance = exchange_instance
        self.exchange_type = exchange_type
        self.quality_level = quality_level
        self.logger = system_logger.getChild(f"EnhancedUnifiedExchangeAdapter-{exchange_type.value.upper()}")
        
        # Initialize unified standardizer
        self.standardizer = UnifiedExchangeStandardizer(quality_level)
        
        # Initialize data processing utilities
        self.data_processor = DataProcessor()
        self.quality_framework = DataQualityFramework()
        
        self.logger.info(f"✅ Enhanced unified adapter initialized for {exchange_type.value} with {quality_level.value} quality level")
    
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Get standardized klines data from the exchange.
        
        This method ensures complete equivalency with other exchanges.
        """
        try:
            # Get raw data from exchange
            raw_data = await self._get_raw_klines(
                symbol, interval, start_time, end_time, limit
            )
            
            if not raw_data or (isinstance(raw_data, pd.DataFrame) and raw_data.empty):
                self.logger.warning(f"No data returned for {symbol} {interval}")
                return pd.DataFrame()
            
            # Standardize data using unified standardizer
            standardized_df = self.standardizer.standardize_to_dataframe(
                raw_data, self.exchange_type, symbol, interval
            )
            
            # Apply additional processing for src/utils/data/ compatibility
            standardized_df = self._apply_compatibility_processing(standardized_df)
            
            # Validate data quality
            self._validate_data_quality(standardized_df, f"{self.exchange_type.value} klines")
            
            self.logger.info(f"✅ Retrieved {len(standardized_df)} klines for {symbol} {interval}")
            return standardized_df
            
        except Exception as e:
            self.logger.error(f"Failed to get klines: {e}")
            raise
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get standardized ticker information"""
        try:
            # Get raw ticker data
            raw_ticker = await self._get_raw_ticker(symbol)
            
            # Standardize ticker format
            standardized_ticker = self._standardize_ticker(raw_ticker, symbol)
            
            return standardized_ticker
            
        except Exception as e:
            self.logger.error(f"Failed to get ticker: {e}")
            raise
    
    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Get standardized order book data"""
        try:
            # Get raw order book data
            raw_orderbook = await self._get_raw_orderbook(symbol, limit)
            
            # Standardize order book format
            standardized_orderbook = self._standardize_orderbook(raw_orderbook, symbol)
            
            return standardized_orderbook
            
        except Exception as e:
            self.logger.error(f"Failed to get order book: {e}")
            raise
    
    async def get_trades(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Get standardized trades data"""
        try:
            # Get raw trades data
            raw_trades = await self._get_raw_trades(symbol, limit)
            
            if not raw_trades or (isinstance(raw_trades, pd.DataFrame) and raw_trades.empty):
                return pd.DataFrame()
            
            # Standardize trades format
            standardized_trades = self._standardize_trades(raw_trades, symbol)
            
            return standardized_trades
            
        except Exception as e:
            self.logger.error(f"Failed to get trades: {e}")
            raise
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get standardized account information"""
        try:
            # Get raw account info
            raw_account = await self._get_raw_account_info()
            
            # Standardize account format
            standardized_account = self._standardize_account_info(raw_account)
            
            return standardized_account
            
        except Exception as e:
            self.logger.error(f"Failed to get account info: {e}")
            raise
    
    async def get_balance(self, currency: str = None) -> Dict[str, Any]:
        """Get standardized balance information"""
        try:
            # Get raw balance data
            raw_balance = await self._get_raw_balance(currency)
            
            # Standardize balance format
            standardized_balance = self._standardize_balance(raw_balance, currency)
            
            return standardized_balance
            
        except Exception as e:
            self.logger.error(f"Failed to get balance: {e}")
            raise
    
    async def _get_raw_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        limit: int
    ) -> Union[List[Dict], List[List], pd.DataFrame]:
        """Get raw klines data from exchange instance"""
        # Try different method names that exchanges might use
        method_names = ['get_klines', '_get_klines', 'get_klines_raw', '_get_klines_raw']
        
        for method_name in method_names:
            if hasattr(self.exchange_instance, method_name):
                method = getattr(self.exchange_instance, method_name)
                try:
                    return await method(symbol, interval, start_time, end_time, limit)
                except Exception as e:
                    self.logger.debug(f"Method {method_name} failed: {e}")
                    continue
        
        raise ValueError(f"No suitable klines method found on exchange instance")
    
    async def _get_raw_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get raw ticker data from exchange instance"""
        method_names = ['get_ticker', '_get_ticker', 'get_ticker_raw']
        
        for method_name in method_names:
            if hasattr(self.exchange_instance, method_name):
                method = getattr(self.exchange_instance, method_name)
                try:
                    return await method(symbol)
                except Exception as e:
                    self.logger.debug(f"Method {method_name} failed: {e}")
                    continue
        
        raise ValueError(f"No suitable ticker method found on exchange instance")
    
    async def _get_raw_orderbook(self, symbol: str, limit: int) -> Dict[str, Any]:
        """Get raw order book data from exchange instance"""
        method_names = ['get_orderbook', '_get_orderbook', 'get_order_book']
        
        for method_name in method_names:
            if hasattr(self.exchange_instance, method_name):
                method = getattr(self.exchange_instance, method_name)
                try:
                    return await method(symbol, limit)
                except Exception as e:
                    self.logger.debug(f"Method {method_name} failed: {e}")
                    continue
        
        raise ValueError(f"No suitable order book method found on exchange instance")
    
    async def _get_raw_trades(self, symbol: str, limit: int) -> Union[List[Dict], pd.DataFrame]:
        """Get raw trades data from exchange instance"""
        method_names = ['get_trades', '_get_trades', 'get_recent_trades']
        
        for method_name in method_names:
            if hasattr(self.exchange_instance, method_name):
                method = getattr(self.exchange_instance, method_name)
                try:
                    return await method(symbol, limit)
                except Exception as e:
                    self.logger.debug(f"Method {method_name} failed: {e}")
                    continue
        
        raise ValueError(f"No suitable trades method found on exchange instance")
    
    async def _get_raw_account_info(self) -> Dict[str, Any]:
        """Get raw account info from exchange instance"""
        method_names = ['get_account_info', '_get_account_info', 'get_account']
        
        for method_name in method_names:
            if hasattr(self.exchange_instance, method_name):
                method = getattr(self.exchange_instance, method_name)
                try:
                    return await method()
                except Exception as e:
                    self.logger.debug(f"Method {method_name} failed: {e}")
                    continue
        
        raise ValueError(f"No suitable account info method found on exchange instance")
    
    async def _get_raw_balance(self, currency: str = None) -> Dict[str, Any]:
        """Get raw balance data from exchange instance"""
        method_names = ['get_balance', '_get_balance', 'get_balances']
        
        for method_name in method_names:
            if hasattr(self.exchange_instance, method_name):
                method = getattr(self.exchange_instance, method_name)
                try:
                    if currency:
                        return await method(currency)
                    else:
                        return await method()
                except Exception as e:
                    self.logger.debug(f"Method {method_name} failed: {e}")
                    continue
        
        raise ValueError(f"No suitable balance method found on exchange instance")
    
    def _apply_compatibility_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply processing to ensure compatibility with src/utils/data/ utilities"""
        try:
            # Regularize timestamps
            df = self.data_processor.regularize_timestamps(df)
            
            # Optimize data types
            df = self.data_processor.optimize_dataframe_dtypes(df)
            
            # Apply feature-specific optimizations
            df = self.data_processor.apply_feature_specific_optimization(df)
            
            return df
            
        except Exception as e:
            self.logger.warning(f"Compatibility processing failed: {e}")
            return df
    
    def _validate_data_quality(self, df: pd.DataFrame, context: str) -> None:
        """Validate data quality using src/utils/data/ framework"""
        try:
            # Use the quality framework for validation
            quality_result = self.quality_framework.validate_dataframe_quality(df, context)
            
            if not quality_result.passed:
                self.logger.warning(f"Data quality validation failed: {quality_result.issues}")
            
            # Log quality metrics
            self.logger.info(f"Data quality score: {quality_result.quality_score:.2f}")
            
        except Exception as e:
            self.logger.warning(f"Data quality validation failed: {e}")
    
    def _standardize_ticker(self, raw_ticker: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Standardize ticker data format"""
        return {
            'symbol': symbol,
            'exchange': self.exchange_type.value,
            'timestamp': datetime.now(timezone.utc),
            'last_price': raw_ticker.get('last_price', raw_ticker.get('lastPrice', 0.0)),
            'bid_price': raw_ticker.get('bid_price', raw_ticker.get('bidPrice', 0.0)),
            'ask_price': raw_ticker.get('ask_price', raw_ticker.get('askPrice', 0.0)),
            'volume_24h': raw_ticker.get('volume_24h', raw_ticker.get('volume24h', 0.0)),
            'price_change_24h': raw_ticker.get('price_change_24h', raw_ticker.get('priceChange24h', 0.0)),
            'price_change_percent_24h': raw_ticker.get('price_change_percent_24h', raw_ticker.get('priceChangePercent24h', 0.0))
        }
    
    def _standardize_orderbook(self, raw_orderbook: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Standardize order book data format"""
        return {
            'symbol': symbol,
            'exchange': self.exchange_type.value,
            'timestamp': datetime.now(timezone.utc),
            'bids': raw_orderbook.get('bids', []),
            'asks': raw_orderbook.get('asks', []),
            'bid_count': len(raw_orderbook.get('bids', [])),
            'ask_count': len(raw_orderbook.get('asks', []))
        }
    
    def _standardize_trades(self, raw_trades: Union[List[Dict], pd.DataFrame], symbol: str) -> pd.DataFrame:
        """Standardize trades data format"""
        if isinstance(raw_trades, pd.DataFrame):
            df = raw_trades.copy()
        else:
            df = pd.DataFrame(raw_trades)
        
        # Standardize column names
        column_mapping = {
            'id': 'trade_id',
            'price': 'price',
            'qty': 'quantity',
            'quantity': 'quantity',
            'amount': 'amount',
            'time': 'timestamp',
            'timestamp': 'timestamp',
            'side': 'side',
            'is_buyer_maker': 'is_buyer_maker'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Add metadata
        df['symbol'] = symbol
        df['exchange'] = self.exchange_type.value
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        
        return df
    
    def _standardize_account_info(self, raw_account: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize account information format"""
        return {
            'exchange': self.exchange_type.value,
            'account_type': raw_account.get('account_type', raw_account.get('accountType', 'spot')),
            'can_trade': raw_account.get('can_trade', raw_account.get('canTrade', False)),
            'can_withdraw': raw_account.get('can_withdraw', raw_account.get('canWithdraw', False)),
            'can_deposit': raw_account.get('can_deposit', raw_account.get('canDeposit', False)),
            'update_time': raw_account.get('update_time', raw_account.get('updateTime', datetime.now(timezone.utc)))
        }
    
    def _standardize_balance(self, raw_balance: Dict[str, Any], currency: str = None) -> Dict[str, Any]:
        """Standardize balance information format"""
        if currency:
            # Single currency balance
            return {
                'currency': currency,
                'exchange': self.exchange_type.value,
                'free': raw_balance.get('free', 0.0),
                'used': raw_balance.get('used', 0.0),
                'total': raw_balance.get('total', 0.0)
            }
        else:
            # All balances
            return {
                'exchange': self.exchange_type.value,
                'balances': raw_balance.get('balances', raw_balance.get('info', {})),
                'total_balance': raw_balance.get('total_balance', 0.0)
            }


class EnhancedUnifiedExchangeManager:
    """
    Manager for multiple enhanced unified exchange adapters.
    
    This class provides a single interface to access multiple exchanges
    with guaranteed data equivalency.
    """
    
    def __init__(self, quality_level: DataQualityLevel = DataQualityLevel.STANDARD):
        """Initialize the enhanced unified exchange manager"""
        self.logger = system_logger.getChild("EnhancedUnifiedExchangeManager")
        self.adapters: Dict[ExchangeType, EnhancedUnifiedExchangeAdapter] = {}
        self.quality_level = quality_level
        
        self.logger.info(f"✅ EnhancedUnifiedExchangeManager initialized with {quality_level.value} quality level")
    
    def register_exchange(
        self, 
        exchange_instance: Any, 
        exchange_type: ExchangeType,
        quality_level: Optional[DataQualityLevel] = None
    ) -> None:
        """Register an exchange instance"""
        quality = quality_level or self.quality_level
        adapter = EnhancedUnifiedExchangeAdapter(exchange_instance, exchange_type, quality)
        self.adapters[exchange_type] = adapter
        self.logger.info(f"✅ Registered exchange: {exchange_type.value} with {quality.value} quality level")
    
    def get_adapter(self, exchange_type: ExchangeType) -> EnhancedUnifiedExchangeAdapter:
        """Get adapter for specific exchange"""
        if exchange_type not in self.adapters:
            raise ValueError(f"Exchange {exchange_type.value} not registered")
        return self.adapters[exchange_type]
    
    async def get_klines_from_all(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> Dict[ExchangeType, pd.DataFrame]:
        """
        Get klines data from all registered exchanges.
        
        Returns:
            Dictionary mapping exchange types to standardized DataFrames
        """
        results = {}
        
        for exchange_type, adapter in self.adapters.items():
            try:
                df = await adapter.get_klines(symbol, interval, start_time, end_time, limit)
                results[exchange_type] = df
            except Exception as e:
                self.logger.error(f"Failed to get klines from {exchange_type.value}: {e}")
                results[exchange_type] = pd.DataFrame()
        
        return results
    
    def validate_equivalency(
        self,
        data1: pd.DataFrame,
        data2: pd.DataFrame,
        tolerance: float = 1e-6
    ) -> Dict[str, Any]:
        """Validate that two DataFrames are equivalent"""
        return validate_ohlcv_equivalency(data1, data2, tolerance)
    
    def get_available_exchanges(self) -> List[ExchangeType]:
        """Get list of available exchanges"""
        return list(self.adapters.keys())
    
    def compare_exchange_data(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> Dict[str, Any]:
        """
        Compare data from all exchanges to ensure equivalency.
        
        Returns:
            Comparison results with equivalency validation
        """
        comparison_results = {
            'symbol': symbol,
            'interval': interval,
            'exchanges_compared': [],
            'equivalency_results': {},
            'overall_equivalent': True,
            'issues': []
        }
        
        try:
            # Get data from all exchanges
            all_data = asyncio.run(self.get_klines_from_all(symbol, interval, start_time, end_time, limit))
            
            comparison_results['exchanges_compared'] = list(all_data.keys())
            
            # Compare each pair of exchanges
            exchange_list = list(all_data.keys())
            for i, exchange1 in enumerate(exchange_list):
                for exchange2 in exchange_list[i+1:]:
                    if not all_data[exchange1].empty and not all_data[exchange2].empty:
                        equivalency_result = self.validate_equivalency(
                            all_data[exchange1], all_data[exchange2]
                        )
                        
                        pair_key = f"{exchange1.value}_vs_{exchange2.value}"
                        comparison_results['equivalency_results'][pair_key] = equivalency_result
                        
                        if not equivalency_result['equivalent']:
                            comparison_results['overall_equivalent'] = False
                            comparison_results['issues'].extend(equivalency_result['errors'])
            
            return comparison_results
            
        except Exception as e:
            comparison_results['overall_equivalent'] = False
            comparison_results['issues'].append(f"Comparison failed: {str(e)}")
            return comparison_results


# Global instance for easy access
enhanced_unified_exchange_manager = EnhancedUnifiedExchangeManager()


# Convenience functions
async def get_enhanced_standardized_klines(
    exchange_instance: Any,
    exchange_type: str,
    symbol: str,
    interval: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 1000,
    quality_level: str = "standard"
) -> pd.DataFrame:
    """
    Convenience function to get enhanced standardized klines data.
    
    Args:
        exchange_instance: The exchange implementation instance
        exchange_type: Exchange type (binance, bingx, okx, mexc)
        symbol: Trading symbol
        interval: Data interval
        start_time: Start time for data
        end_time: End time for data
        limit: Maximum number of records
        quality_level: Quality validation level
        
    Returns:
        Standardized DataFrame compatible with src/utils/data/
    """
    try:
        exchange_type_enum = ExchangeType(exchange_type.lower())
        quality_level_enum = DataQualityLevel(quality_level.lower())
        
        adapter = EnhancedUnifiedExchangeAdapter(exchange_instance, exchange_type_enum, quality_level_enum)
        return await adapter.get_klines(symbol, interval, start_time, end_time, limit)
        
    except ValueError as e:
        raise ValueError(f"Invalid exchange type or quality level: {e}")


def create_enhanced_unified_adapter(
    exchange_instance: Any, 
    exchange_type: str, 
    quality_level: str = "standard"
) -> EnhancedUnifiedExchangeAdapter:
    """
    Create an enhanced unified adapter for an exchange instance.
    
    Args:
        exchange_instance: The exchange implementation instance
        exchange_type: Exchange type (binance, bingx, okx, mexc)
        quality_level: Quality validation level
        
    Returns:
        EnhancedUnifiedExchangeAdapter instance
    """
    try:
        exchange_type_enum = ExchangeType(exchange_type.lower())
        quality_level_enum = DataQualityLevel(quality_level.lower())
        return EnhancedUnifiedExchangeAdapter(exchange_instance, exchange_type_enum, quality_level_enum)
        
    except ValueError as e:
        raise ValueError(f"Invalid exchange type or quality level: {e}")


# Export the main classes and functions
__all__ = [
    'EnhancedUnifiedExchangeAdapter',
    'EnhancedUnifiedExchangeManager',
    'IEnhancedUnifiedExchange',
    'ExchangeConfig',
    'get_enhanced_standardized_klines',
    'create_enhanced_unified_adapter',
    'enhanced_unified_exchange_manager'
]