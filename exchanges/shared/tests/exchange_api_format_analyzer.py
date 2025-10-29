#!/usr/bin/env python3
"""
Exchange API Format Analyzer

Comprehensive testing and analysis tool for exchange APIs that:
1. Tests exchange API calls systematically
2. Collects and analyzes response formats from different exchanges
3. Identifies format differences and inconsistencies
4. Generates adapter code recommendations for unified formatting

This script works alongside enhanced_position_test_suite.py to ensure
proper exchange integration and data format standardization.
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from collections import defaultdict
import pandas as pd

# Add workspace to path
workspace = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(workspace))

from exchanges.exchange_dispatcher import ExchangeDispatcher, ExchangeConfig, ExchangeType, TradingMode
from exchanges.base_exchange.exchange_interface import IExchange, ExchangeStatus
from src.utils.logger import system_logger
from src.utils.tprint import tprint_info, tprint_success, tprint_error, tprint_debug, tprint_warning

logger = logging.getLogger(__name__)


class ResponseType(Enum):
    """Types of API responses to analyze"""
    TICKER = "ticker"
    KLINES = "klines"
    ORDERBOOK = "orderbook"
    BALANCE = "balance"
    ACCOUNT_INFO = "account_info"
    ORDER = "order"
    ORDER_STATUS = "order_status"
    OPEN_ORDERS = "open_orders"
    POSITIONS = "positions"
    TRADES = "trades"


@dataclass
class APIResponseSample:
    """Sample API response with metadata"""
    exchange: str
    response_type: ResponseType
    raw_response: Dict[str, Any]
    timestamp: datetime
    symbol: Optional[str] = None
    error: Optional[str] = None
    response_time_ms: Optional[float] = None


@dataclass
class FieldAnalysis:
    """Analysis of a field across exchanges"""
    field_name: str
    data_types: Set[str]
    present_in_exchanges: Set[str]
    missing_in_exchanges: Set[str]
    value_examples: Dict[str, Any] = field(default_factory=dict)
    is_standardized: bool = False
    standardization_notes: str = ""


@dataclass
class FormatAnalysis:
    """Analysis of response format for a specific API call"""
    response_type: ResponseType
    exchange_formats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    common_fields: Set[str] = field(default_factory=set)
    exchange_specific_fields: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    field_analyses: Dict[str, FieldAnalysis] = field(default_factory=dict)
    standardization_recommendations: List[str] = field(default_factory=list)
    adapter_code_suggestions: List[str] = field(default_factory=list)


class ExchangeAPIFormatAnalyzer:
    """
    Analyzer for exchange API formats and responses.
    
    This class systematically tests exchange APIs, collects response samples,
    analyzes format differences, and generates adapter recommendations.
    """
    
    def __init__(
        self,
        test_symbols: List[str] = None,
        exchanges: List[str] = None,
        mode: str = 'mock',
        save_samples: bool = True,
        output_dir: str = "exchange_format_analysis",
        sequential: bool = False,
        single_exchange: Optional[str] = None
    ):
        """
        Initialize the analyzer.
        
        Args:
            test_symbols: List of symbols to test (default: ['BTCUSDT'])
            exchanges: List of exchanges to test (default: all supported)
            mode: Test mode - 'real' for actual exchange calls, 'mock' for mock data
            save_samples: Whether to save response samples to files
            output_dir: Directory for saving analysis results
            sequential: If True, test exchanges one by one (cleanup after each)
            single_exchange: If provided, test only this exchange
        """
        self.logger = system_logger.getChild('ExchangeAPIFormatAnalyzer')
        self.mode = mode.lower()
        self.save_samples = save_samples
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sequential = sequential
        self.single_exchange = single_exchange.lower() if single_exchange else None
        
        # Test configuration
        self.test_symbols = test_symbols or ['BTCUSDT', 'ETHUSDT']
        
        # Supported exchanges
        supported_exchanges = [e.value for e in ExchangeType]
        if exchanges:
            self.exchanges = [e.lower() for e in exchanges if e.lower() in supported_exchanges]
        else:
            self.exchanges = supported_exchanges
        
        # Filter to single exchange if specified
        if self.single_exchange:
            if self.single_exchange not in self.exchanges:
                raise ValueError(f"Exchange '{self.single_exchange}' not in supported exchanges: {', '.join(self.exchanges)}")
            self.exchanges = [self.single_exchange]
            tprint_info(f"🎯 Single exchange mode: testing only {self.single_exchange.upper()}")
        
        # Response samples storage
        self.response_samples: List[APIResponseSample] = []
        self.format_analyses: Dict[ResponseType, FormatAnalysis] = {}
        
        # Exchange dispatchers
        self.dispatchers: Dict[str, ExchangeDispatcher] = {}
        
        tprint_info(f"✅ Exchange API Format Analyzer initialized")
        tprint_info(f"   Exchanges: {', '.join(self.exchanges)}")
        tprint_info(f"   Symbols: {', '.join(self.test_symbols)}")
        tprint_info(f"   Mode: {self.mode.upper()}")
        if self.sequential:
            tprint_info(f"   Sequential mode: testing exchanges one by one")
        if self.single_exchange:
            tprint_info(f"   Single exchange mode: {self.single_exchange.upper()}")
    
    async def initialize_exchanges(self) -> None:
        """Initialize exchange dispatchers."""
        if self.sequential:
            tprint_info("🔧 Sequential mode: Will initialize exchanges one at a time")
            tprint_info(f"   Total exchanges to process: {len(self.exchanges)}")
        else:
            tprint_info("🔧 Initializing exchange connections...")
            tprint_info(f"   Processing {len(self.exchanges)} exchanges...")
        
        initialized_count = 0
        skipped_count = 0
        failed_count = 0
        
        for idx, exchange_name in enumerate(self.exchanges, 1):
            if self.sequential:
                tprint_info("")
                tprint_info(f"[{idx}/{len(self.exchanges)}] Processing {exchange_name.upper()}...")
            
            tprint_debug(f"   Initializing {exchange_name.upper()}...")
            try:
                # Get API credentials from environment
                api_key = os.getenv(f'{exchange_name.upper()}_API_KEY', '')
                api_secret = os.getenv(f'{exchange_name.upper()}_API_SECRET', '')
                
                if not api_key or not api_secret:
                    if self.mode == 'real':
                        tprint_warning(f"⚠️  Missing credentials for {exchange_name.upper()}, skipping")
                        skipped_count += 1
                        continue
                    # For mock mode, use dummy credentials
                    tprint_debug(f"   Using mock credentials for {exchange_name.upper()}")
                    api_key = 'mock_key'
                    api_secret = 'mock_secret'
                
                # Create exchange config
                exchange_type = ExchangeType[exchange_name.upper()]
                config = ExchangeConfig(
                    exchange_type=exchange_type,
                    api_key=api_key,
                    api_secret=api_secret,
                    use_testnet=(self.mode == 'real'),  # Use testnet for real mode
                    trade_symbol=self.test_symbols[0],
                    mode=TradingMode.PAPER if self.mode == 'mock' else TradingMode.PAPER
                )
                
                # Create dispatcher
                tprint_debug(f"   Creating dispatcher for {exchange_name.upper()}...")
                dispatcher = ExchangeDispatcher(config)
                success = await dispatcher.initialize()
                
                if success:
                    self.dispatchers[exchange_name] = dispatcher
                    initialized_count += 1
                    tprint_success(f"✅ {exchange_name.upper()} initialized successfully")
                else:
                    failed_count += 1
                    tprint_error(f"❌ Failed to initialize {exchange_name.upper()}")
                    
            except Exception as e:
                failed_count += 1
                tprint_error(f"❌ Error initializing {exchange_name.upper()}: {e}")
                self.logger.error(f"Error initializing {exchange_name}: {e}")
        
        tprint_info(f"✅ Exchange initialization complete:")
        tprint_info(f"   Initialized: {initialized_count}")
        if skipped_count > 0:
            tprint_warning(f"   Skipped: {skipped_count}")
        if failed_count > 0:
            tprint_error(f"   Failed: {failed_count}")
    
    async def collect_api_responses(self) -> None:
        """Collect API responses from all exchanges."""
        if self.sequential:
            # Sequential mode: test one exchange at a time
            await self._collect_responses_sequential()
        else:
            # Parallel mode: test all exchanges together
            await self._collect_responses_parallel()
    
    async def _collect_responses_parallel(self) -> None:
        """Collect responses from all exchanges in parallel."""
        tprint_info("📊 Collecting API responses from exchanges (parallel mode)...")
        tprint_info(f"   Testing {len(self.dispatchers)} exchanges with {len(self.test_symbols)} symbols")
        
        # Test different response types
        response_types = [
            (ResponseType.TICKER, self._test_ticker),
            (ResponseType.KLINES, self._test_klines),
            (ResponseType.ORDERBOOK, self._test_orderbook),
            (ResponseType.BALANCE, self._test_balance),
            (ResponseType.ACCOUNT_INFO, self._test_account_info),
        ]
        
        # For mock mode, also test order-related endpoints
        if self.mode == 'mock':
            tprint_debug("   Mock mode: Including order-related endpoints")
            response_types.extend([
                (ResponseType.ORDER_STATUS, self._test_order_status),
                (ResponseType.OPEN_ORDERS, self._test_open_orders),
                (ResponseType.POSITIONS, self._test_positions),
            ])
        
        total_tests = len(response_types) * len(self.dispatchers) * len(self.test_symbols)
        tprint_info(f"   Total API calls to execute: {total_tests}")
        tprint_info("")
        
        successful_samples = 0
        error_samples = 0
        
        for idx, (response_type, test_func) in enumerate(response_types, 1):
            tprint_info(f"📋 [{idx}/{len(response_types)}] Testing {response_type.value}...")
            
            for exchange_name in self.dispatchers.keys():
                tprint_debug(f"   Testing {exchange_name.upper()}...")
                for symbol in self.test_symbols:
                    try:
                        await test_func(exchange_name, symbol, response_type)
                        successful_samples += 1
                    except Exception as e:
                        error_samples += 1
                        tprint_error(f"❌ Error testing {response_type.value} on {exchange_name} ({symbol}): {e}")
                        self.logger.error(f"Error testing {response_type.value} on {exchange_name}: {e}")
        
        tprint_info("")
        tprint_success(f"✅ Response collection complete:")
        tprint_info(f"   Total samples collected: {len(self.response_samples)}")
        tprint_info(f"   Successful: {successful_samples}")
        if error_samples > 0:
            tprint_warning(f"   Errors: {error_samples}")
        
        # Save samples if requested
        if self.save_samples:
            tprint_info("")
            await self._save_response_samples()
    
    async def _collect_responses_sequential(self) -> None:
        """Collect responses from exchanges one by one."""
        tprint_info("📊 Collecting API responses from exchanges (sequential mode)...")
        tprint_info(f"   Testing {len(self.exchanges)} exchanges one by one")
        tprint_info("")
        
        # Test different response types
        response_types = [
            (ResponseType.TICKER, self._test_ticker),
            (ResponseType.KLINES, self._test_klines),
            (ResponseType.ORDERBOOK, self._test_orderbook),
            (ResponseType.BALANCE, self._test_balance),
            (ResponseType.ACCOUNT_INFO, self._test_account_info),
        ]
        
        # For mock mode, also test order-related endpoints
        if self.mode == 'mock':
            tprint_debug("   Mock mode: Including order-related endpoints")
            response_types.extend([
                (ResponseType.ORDER_STATUS, self._test_order_status),
                (ResponseType.OPEN_ORDERS, self._test_open_orders),
                (ResponseType.POSITIONS, self._test_positions),
            ])
        
        total_exchanges = len(self.exchanges)
        total_successful = 0
        total_errors = 0
        
        for exchange_idx, exchange_name in enumerate(self.exchanges, 1):
            tprint_info("=" * 70)
            tprint_info(f"[{exchange_idx}/{total_exchanges}] Testing {exchange_name.upper()}")
            tprint_info("=" * 70)
            
            # Initialize this exchange
            tprint_info(f"   Initializing {exchange_name.upper()}...")
            try:
                api_key = os.getenv(f'{exchange_name.upper()}_API_KEY', '')
                api_secret = os.getenv(f'{exchange_name.upper()}_API_SECRET', '')
                
                if not api_key or not api_secret:
                    if self.mode == 'real':
                        tprint_warning(f"   ⚠️  Missing credentials, skipping {exchange_name.upper()}")
                        continue
                    api_key = 'mock_key'
                    api_secret = 'mock_secret'
                
                exchange_type = ExchangeType[exchange_name.upper()]
                config = ExchangeConfig(
                    exchange_type=exchange_type,
                    api_key=api_key,
                    api_secret=api_secret,
                    use_testnet=(self.mode == 'real'),
                    trade_symbol=self.test_symbols[0],
                    mode=TradingMode.PAPER if self.mode == 'mock' else TradingMode.PAPER
                )
                
                dispatcher = ExchangeDispatcher(config)
                success = await dispatcher.initialize()
                
                if not success:
                    tprint_error(f"   ❌ Failed to initialize {exchange_name.upper()}")
                    continue
                
                # Store dispatcher temporarily for this exchange
                self.dispatchers[exchange_name] = dispatcher
                tprint_success(f"   ✅ {exchange_name.upper()} initialized")
                
            except Exception as e:
                tprint_error(f"   ❌ Error initializing {exchange_name.upper()}: {e}")
                continue
            
            # Test all response types for this exchange
            exchange_samples_before = len(self.response_samples)
            successful_samples = 0
            error_samples = 0
            
            for idx, (response_type, test_func) in enumerate(response_types, 1):
                tprint_info(f"   📋 [{idx}/{len(response_types)}] Testing {response_type.value}...")
                
                for symbol in self.test_symbols:
                    try:
                        await test_func(exchange_name, symbol, response_type)
                        successful_samples += 1
                    except Exception as e:
                        error_samples += 1
                        tprint_error(f"      ❌ Error testing {response_type.value} ({symbol}): {e}")
                        self.logger.error(f"Error testing {response_type.value} on {exchange_name}: {e}")
            
            exchange_samples = len(self.response_samples) - exchange_samples_before
            total_successful += successful_samples
            total_errors += error_samples
            
            tprint_info("")
            tprint_info(f"   {exchange_name.upper()} Summary:")
            tprint_info(f"      Samples collected: {exchange_samples}")
            tprint_info(f"      Successful: {successful_samples}")
            if error_samples > 0:
                tprint_warning(f"      Errors: {error_samples}")
            
            # Save samples for this exchange before cleanup
            if self.save_samples:
                tprint_info(f"   💾 Saving samples for {exchange_name.upper()}...")
                await self._save_response_samples_for_exchange(exchange_name, exchange_samples_before)
            
            # Cleanup this exchange before moving to next
            tprint_info(f"   🧹 Cleaning up {exchange_name.upper()}...")
            try:
                await dispatcher.close()
                tprint_debug(f"      ✅ {exchange_name.upper()} closed")
            except Exception as e:
                tprint_warning(f"      ⚠️  Error closing {exchange_name.upper()}: {e}")
            
            # Remove from dispatchers dict
            if exchange_name in self.dispatchers:
                del self.dispatchers[exchange_name]
            
            tprint_info("")
        
        tprint_info("=" * 70)
        tprint_success("✅ Sequential response collection complete:")
        tprint_info(f"   Total samples collected: {len(self.response_samples)}")
        tprint_info(f"   Total successful: {total_successful}")
        if total_errors > 0:
            tprint_warning(f"   Total errors: {total_errors}")
    
    async def _save_response_samples_for_exchange(self, exchange_name: str, start_index: int) -> None:
        """Save response samples for a specific exchange."""
        samples_dir = self.output_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        # Get samples for this exchange starting from start_index
        exchange_samples = self.response_samples[start_index:]
        
        # Group by response type
        grouped_samples: Dict[str, List[Dict]] = defaultdict(list)
        
        for sample in exchange_samples:
            if sample.exchange == exchange_name:
                grouped_samples[sample.response_type.value].append({
                    'symbol': sample.symbol,
                    'timestamp': sample.timestamp.isoformat(),
                    'response_time_ms': sample.response_time_ms,
                    'error': sample.error,
                    'raw_response': sample.raw_response
                })
        
        # Save grouped samples
        for response_type, samples in grouped_samples.items():
            filename = samples_dir / f"{exchange_name}_{response_type}_samples.json"
            with open(filename, 'w') as f:
                json.dump(samples, f, indent=2, default=str)
            tprint_debug(f"      Saved {filename.name} ({len(samples)} samples)")
    
    async def _test_ticker(self, exchange_name: str, symbol: str, response_type: ResponseType) -> None:
        """Test ticker API call."""
        dispatcher = self.dispatchers[exchange_name]
        start_time = datetime.now()
        
        try:
            ticker = await dispatcher.get_ticker(symbol)
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if ticker:
                sample = APIResponseSample(
                    exchange=exchange_name,
                    response_type=response_type,
                    raw_response=ticker if isinstance(ticker, dict) else {'data': ticker},
                    timestamp=datetime.now(timezone.utc),
                    symbol=symbol,
                    response_time_ms=response_time
                )
                self.response_samples.append(sample)
                tprint_debug(f"      ✓ {exchange_name}/{symbol}: ticker collected ({response_time:.1f}ms)")
            else:
                tprint_warning(f"      ⚠ {exchange_name}/{symbol}: ticker returned empty")
        except Exception as e:
            sample = APIResponseSample(
                exchange=exchange_name,
                response_type=response_type,
                raw_response={},
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                error=str(e)
            )
            self.response_samples.append(sample)
            tprint_debug(f"      ✗ {exchange_name}/{symbol}: ticker failed - {str(e)[:50]}")
    
    async def _test_klines(self, exchange_name: str, symbol: str, response_type: ResponseType) -> None:
        """Test klines API call."""
        dispatcher = self.dispatchers[exchange_name]
        start_time = datetime.now()
        
        try:
            klines = await dispatcher.get_ohlcv(symbol, '1h', limit=10)
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if klines:
                # Convert to dict format for analysis
                if isinstance(klines, list):
                    klines_dict = {
                        'data': [asdict(k) if hasattr(k, '__dict__') else k for k in klines],
                        'count': len(klines)
                    }
                else:
                    klines_dict = klines if isinstance(klines, dict) else {'data': klines}
                
                sample = APIResponseSample(
                    exchange=exchange_name,
                    response_type=response_type,
                    raw_response=klines_dict,
                    timestamp=datetime.now(timezone.utc),
                    symbol=symbol,
                    response_time_ms=response_time
                )
                self.response_samples.append(sample)
                count = len(klines) if isinstance(klines, list) else klines_dict.get('count', 0)
                tprint_debug(f"      ✓ {exchange_name}/{symbol}: klines collected ({count} candles, {response_time:.1f}ms)")
            else:
                tprint_warning(f"      ⚠ {exchange_name}/{symbol}: klines returned empty")
        except Exception as e:
            sample = APIResponseSample(
                exchange=exchange_name,
                response_type=response_type,
                raw_response={},
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                error=str(e)
            )
            self.response_samples.append(sample)
            tprint_debug(f"      ✗ {exchange_name}/{symbol}: klines failed - {str(e)[:50]}")
    
    async def _test_orderbook(self, exchange_name: str, symbol: str, response_type: ResponseType) -> None:
        """Test orderbook API call."""
        dispatcher = self.dispatchers[exchange_name]
        start_time = datetime.now()
        
        try:
            orderbook = await dispatcher.get_order_book(symbol, limit=20)
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if orderbook:
                sample = APIResponseSample(
                    exchange=exchange_name,
                    response_type=response_type,
                    raw_response=orderbook if isinstance(orderbook, dict) else {'data': orderbook},
                    timestamp=datetime.now(timezone.utc),
                    symbol=symbol,
                    response_time_ms=response_time
                )
                self.response_samples.append(sample)
                tprint_debug(f"      ✓ {exchange_name}/{symbol}: orderbook collected ({response_time:.1f}ms)")
            else:
                tprint_warning(f"      ⚠ {exchange_name}/{symbol}: orderbook returned empty")
        except Exception as e:
            sample = APIResponseSample(
                exchange=exchange_name,
                response_type=response_type,
                raw_response={},
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                error=str(e)
            )
            self.response_samples.append(sample)
            tprint_debug(f"      ✗ {exchange_name}/{symbol}: orderbook failed - {str(e)[:50]}")
    
    async def _test_balance(self, exchange_name: str, symbol: str, response_type: ResponseType) -> None:
        """Test balance API call."""
        dispatcher = self.dispatchers[exchange_name]
        start_time = datetime.now()
        
        try:
            balance = await dispatcher.get_balance('USDT')
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            sample = APIResponseSample(
                exchange=exchange_name,
                response_type=response_type,
                raw_response={'balance': balance, 'currency': 'USDT'},
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                response_time_ms=response_time
            )
            self.response_samples.append(sample)
            tprint_debug(f"      ✓ {exchange_name}/{symbol}: balance collected ({balance} USDT, {response_time:.1f}ms)")
        except Exception as e:
            sample = APIResponseSample(
                exchange=exchange_name,
                response_type=response_type,
                raw_response={},
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                error=str(e)
            )
            self.response_samples.append(sample)
            tprint_debug(f"      ✗ {exchange_name}/{symbol}: balance failed - {str(e)[:50]}")
    
    async def _test_account_info(self, exchange_name: str, symbol: str, response_type: ResponseType) -> None:
        """Test account info API call."""
        dispatcher = self.dispatchers[exchange_name]
        start_time = datetime.now()
        
        try:
            account_info = await dispatcher.get_account_info()
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if account_info:
                sample = APIResponseSample(
                    exchange=exchange_name,
                    response_type=response_type,
                    raw_response=account_info if isinstance(account_info, dict) else {'data': account_info},
                    timestamp=datetime.now(timezone.utc),
                    symbol=symbol,
                    response_time_ms=response_time
                )
                self.response_samples.append(sample)
                tprint_debug(f"      ✓ {exchange_name}/{symbol}: account_info collected ({response_time:.1f}ms)")
            else:
                tprint_warning(f"      ⚠ {exchange_name}/{symbol}: account_info returned empty")
        except Exception as e:
            sample = APIResponseSample(
                exchange=exchange_name,
                response_type=response_type,
                raw_response={},
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                error=str(e)
            )
            self.response_samples.append(sample)
            tprint_debug(f"      ✗ {exchange_name}/{symbol}: account_info failed - {str(e)[:50]}")
    
    async def _test_order_status(self, exchange_name: str, symbol: str, response_type: ResponseType) -> None:
        """Test order status API call (mock only)."""
        # This would require creating a test order first
        # For now, skip in real mode
        pass
    
    async def _test_open_orders(self, exchange_name: str, symbol: str, response_type: ResponseType) -> None:
        """Test open orders API call."""
        dispatcher = self.dispatchers[exchange_name]
        start_time = datetime.now()
        
        try:
            open_orders = await dispatcher.get_open_orders(symbol)
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            sample = APIResponseSample(
                exchange=exchange_name,
                response_type=response_type,
                raw_response={'orders': open_orders, 'count': len(open_orders)},
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                response_time_ms=response_time
            )
            self.response_samples.append(sample)
            tprint_debug(f"      ✓ {exchange_name}/{symbol}: open_orders collected ({len(open_orders)} orders, {response_time:.1f}ms)")
        except Exception as e:
            sample = APIResponseSample(
                exchange=exchange_name,
                response_type=response_type,
                raw_response={},
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                error=str(e)
            )
            self.response_samples.append(sample)
            tprint_debug(f"      ✗ {exchange_name}/{symbol}: open_orders failed - {str(e)[:50]}")
    
    async def _test_positions(self, exchange_name: str, symbol: str, response_type: ResponseType) -> None:
        """Test positions API call."""
        dispatcher = self.dispatchers[exchange_name]
        start_time = datetime.now()
        
        try:
            positions = await dispatcher.get_positions()
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            sample = APIResponseSample(
                exchange=exchange_name,
                response_type=response_type,
                raw_response={'positions': positions, 'count': len(positions)},
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                response_time_ms=response_time
            )
            self.response_samples.append(sample)
            tprint_debug(f"      ✓ {exchange_name}/{symbol}: positions collected ({len(positions)} positions, {response_time:.1f}ms)")
        except Exception as e:
            sample = APIResponseSample(
                exchange=exchange_name,
                response_type=response_type,
                raw_response={},
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                error=str(e)
            )
            self.response_samples.append(sample)
            tprint_debug(f"      ✗ {exchange_name}/{symbol}: positions failed - {str(e)[:50]}")
    
    async def _save_response_samples(self) -> None:
        """Save response samples to files."""
        tprint_info("💾 Saving response samples to files...")
        samples_dir = self.output_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        tprint_debug(f"   Output directory: {samples_dir}")
        
        # Group samples by response type and exchange
        grouped_samples: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
        
        for sample in self.response_samples:
            key = (sample.response_type.value, sample.exchange)
            grouped_samples[key].append({
                'symbol': sample.symbol,
                'timestamp': sample.timestamp.isoformat(),
                'response_time_ms': sample.response_time_ms,
                'error': sample.error,
                'raw_response': sample.raw_response
            })
        
        # Save grouped samples
        files_saved = 0
        for (response_type, exchange), samples in grouped_samples.items():
            filename = samples_dir / f"{exchange}_{response_type}_samples.json"
            with open(filename, 'w') as f:
                json.dump(samples, f, indent=2, default=str)
            files_saved += 1
            tprint_debug(f"   Saved {filename.name} ({len(samples)} samples)")
        
        tprint_success(f"✅ Saved {len(self.response_samples)} response samples to {files_saved} files")
    
    def analyze_formats(self) -> None:
        """Analyze response formats and identify differences."""
        tprint_info("🔍 Analyzing response formats...")
        
        # Group samples by response type
        samples_by_type: Dict[ResponseType, List[APIResponseSample]] = defaultdict(list)
        valid_samples = 0
        error_samples = 0
        
        for sample in self.response_samples:
            if not sample.error:
                samples_by_type[sample.response_type].append(sample)
                valid_samples += 1
            else:
                error_samples += 1
        
        tprint_info(f"   Valid samples: {valid_samples}")
        if error_samples > 0:
            tprint_warning(f"   Error samples (skipped): {error_samples}")
        tprint_info(f"   Response types to analyze: {len(samples_by_type)}")
        tprint_info("")
        
        # Analyze each response type
        for idx, (response_type, samples) in enumerate(samples_by_type.items(), 1):
            tprint_info(f"📊 [{idx}/{len(samples_by_type)}] Analyzing {response_type.value} format...")
            tprint_debug(f"   Processing {len(samples)} samples...")
            
            analysis = self._analyze_response_format(response_type, samples)
            self.format_analyses[response_type] = analysis
            
            # Show summary
            common_count = len(analysis.common_fields)
            exchanges_count = len(analysis.exchange_formats)
            tprint_debug(f"   Found {common_count} common fields across {exchanges_count} exchanges")
            if analysis.standardization_recommendations:
                tprint_debug(f"   Generated {len(analysis.standardization_recommendations)} recommendations")
        
        tprint_info("")
        tprint_success(f"✅ Format analysis complete: {len(self.format_analyses)} response types analyzed")
    
    def _analyze_response_format(
        self,
        response_type: ResponseType,
        samples: List[APIResponseSample]
    ) -> FormatAnalysis:
        """Analyze format for a specific response type."""
        analysis = FormatAnalysis(response_type=response_type)
        
        tprint_debug(f"      Extracting fields from {len(samples)} samples...")
        
        # Collect all fields from all exchanges
        all_fields: Set[str] = set()
        exchange_fields: Dict[str, Set[str]] = defaultdict(set)
        field_values: Dict[str, Dict[str, Any]] = defaultdict(dict)
        field_types: Dict[str, Set[str]] = defaultdict(set)
        
        for sample in samples:
            exchange = sample.exchange
            response = sample.raw_response
            
            # Extract fields from response
            fields = self._extract_fields(response)
            all_fields.update(fields)
            exchange_fields[exchange].update(fields)
            
            # Analyze field values and types
            for field in fields:
                value = self._get_nested_value(response, field)
                field_values[field][exchange] = value
                field_types[field].add(type(value).__name__)
        
        tprint_debug(f"      Found {len(all_fields)} unique fields across {len(exchange_fields)} exchanges")
        
        # Identify common and exchange-specific fields
        common_fields = set.intersection(*exchange_fields.values()) if exchange_fields else set()
        analysis.common_fields = common_fields
        
        tprint_debug(f"      Identified {len(common_fields)} common fields")
        
        for exchange, fields in exchange_fields.items():
            exchange_specific = fields - common_fields
            if exchange_specific:
                analysis.exchange_specific_fields[exchange] = exchange_specific
                tprint_debug(f"         {exchange}: {len(exchange_specific)} exchange-specific fields")
        
        # Store exchange formats
        for exchange, fields in exchange_fields.items():
            analysis.exchange_formats[exchange] = {
                'fields': list(fields),
                'field_count': len(fields),
                'common_fields': list(fields & common_fields),
                'specific_fields': list(fields - common_fields)
            }
        
        # Analyze each field
        tprint_debug(f"      Analyzing {len(all_fields)} fields...")
        for field in all_fields:
            field_analysis = FieldAnalysis(
                field_name=field,
                data_types=field_types[field],
                present_in_exchanges=set(field_values[field].keys()),
                missing_in_exchanges=set(self.exchanges) - set(field_values[field].keys()),
                value_examples={ex: val for ex, val in field_values[field].items()}
            )
            analysis.field_analyses[field] = field_analysis
        
        # Generate recommendations
        tprint_debug(f"      Generating recommendations...")
        analysis.standardization_recommendations = self._generate_recommendations(analysis)
        analysis.adapter_code_suggestions = self._generate_adapter_suggestions(analysis)
        
        return analysis
    
    def _extract_fields(self, obj: Any, prefix: str = "") -> Set[str]:
        """Extract all field names from a nested dictionary."""
        fields = set()
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                field_name = f"{prefix}.{key}" if prefix else key
                fields.add(field_name)
                fields.update(self._extract_fields(value, field_name))
        elif isinstance(obj, list):
            # For lists, extract fields from first element if it's a dict
            if obj and isinstance(obj[0], dict):
                fields.update(self._extract_fields(obj[0], prefix))
            elif obj:
                # For non-dict lists, store as array
                fields.add(f"{prefix}[array]" if prefix else "[array]")
        
        return fields
    
    def _get_nested_value(self, obj: Any, field_path: str) -> Any:
        """Get value from nested dictionary using dot notation."""
        parts = field_path.split('.')
        current = obj
        
        for part in parts:
            if '[' in part and ']' in part:
                # Handle array notation
                key, _ = part.split('[')
                if key and key in current:
                    current = current[key]
                    if isinstance(current, list) and current:
                        current = current[0]
            else:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None
        
        return current
    
    def _generate_recommendations(self, analysis: FormatAnalysis) -> List[str]:
        """Generate standardization recommendations."""
        recommendations = []
        
        # Check for missing common fields
        for exchange in analysis.exchange_formats.keys():
            missing_common = analysis.common_fields - set(analysis.exchange_formats[exchange]['fields'])
            if missing_common:
                recommendations.append(
                    f"{exchange.upper()} missing common fields: {', '.join(missing_common)}"
                )
        
        # Check for inconsistent field types
        for field_name, field_analysis in analysis.field_analyses.items():
            if len(field_analysis.data_types) > 1:
                recommendations.append(
                    f"Field '{field_name}' has inconsistent types: {', '.join(field_analysis.data_types)}"
                )
        
        # Check for exchange-specific fields that should be standardized
        for exchange, specific_fields in analysis.exchange_specific_fields.items():
            if len(specific_fields) > 5:  # Arbitrary threshold
                recommendations.append(
                    f"{exchange.upper()} has many exchange-specific fields ({len(specific_fields)}): "
                    f"consider standardizing key fields"
                )
        
        return recommendations
    
    def _generate_adapter_suggestions(self, analysis: FormatAnalysis) -> List[str]:
        """Generate adapter code suggestions."""
        suggestions = []
        
        # Generate adapter function template
        adapter_template = f"""
def adapt_{analysis.response_type.value}_response(raw_response: Dict[str, Any], exchange: str) -> Dict[str, Any]:
    \"\"\"
    Adapt {analysis.response_type.value} response to unified format.
    
    Args:
        raw_response: Raw response from exchange
        exchange: Exchange name
        
    Returns:
        Standardized response dictionary
    \"\"\"
    standardized = {{}}
    
    # Common field mappings
    common_fields = {list(analysis.common_fields)}
    
    # Exchange-specific mappings
    field_mappings = {{
"""
        
        # Add field mappings for each exchange
        for exchange, format_info in analysis.exchange_formats.items():
            suggestions.append(f"    # {exchange.upper()} field mappings")
            # Would need to map exchange-specific fields to common fields
        
        suggestions.append("    return standardized")
        
        return suggestions
    
    def generate_report(self) -> str:
        """Generate comprehensive analysis report."""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("EXCHANGE API FORMAT ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
        report_lines.append(f"Exchanges tested: {', '.join(self.exchanges)}")
        report_lines.append(f"Response samples collected: {len(self.response_samples)}")
        report_lines.append("")
        
        # Summary by response type
        report_lines.append("SUMMARY BY RESPONSE TYPE")
        report_lines.append("-" * 80)
        
        for response_type, analysis in self.format_analyses.items():
            report_lines.append(f"\n{response_type.value.upper()}")
            report_lines.append(f"  Common fields ({len(analysis.common_fields)}): {', '.join(sorted(analysis.common_fields))}")
            
            for exchange, format_info in analysis.exchange_formats.items():
                report_lines.append(f"  {exchange.upper()}:")
                report_lines.append(f"    Total fields: {format_info['field_count']}")
                report_lines.append(f"    Exchange-specific fields: {len(format_info['specific_fields'])}")
                if format_info['specific_fields']:
                    report_lines.append(f"    Specific: {', '.join(format_info['specific_fields'][:10])}")
        
        # Field analysis
        report_lines.append("\n\nFIELD ANALYSIS")
        report_lines.append("-" * 80)
        
        for response_type, analysis in self.format_analyses.items():
            report_lines.append(f"\n{response_type.value.upper()} Fields:")
            for field_name, field_analysis in sorted(analysis.field_analyses.items()):
                report_lines.append(f"  {field_name}:")
                report_lines.append(f"    Types: {', '.join(field_analysis.data_types)}")
                report_lines.append(f"    Present in: {', '.join(sorted(field_analysis.present_in_exchanges))}")
                if field_analysis.missing_in_exchanges:
                    report_lines.append(f"    Missing in: {', '.join(sorted(field_analysis.missing_in_exchanges))}")
        
        # Recommendations
        report_lines.append("\n\nSTANDARDIZATION RECOMMENDATIONS")
        report_lines.append("-" * 80)
        
        for response_type, analysis in self.format_analyses.items():
            if analysis.standardization_recommendations:
                report_lines.append(f"\n{response_type.value.upper()}:")
                for rec in analysis.standardization_recommendations:
                    report_lines.append(f"  - {rec}")
        
        report_lines.append("\n" + "=" * 80)
        
        return "\n".join(report_lines)
    
    async def save_report(self) -> None:
        """Save analysis report to file."""
        tprint_info("💾 Saving analysis reports...")
        
        report = self.generate_report()
        report_file = self.output_dir / "format_analysis_report.txt"
        
        tprint_debug(f"   Generating text report: {report_file}")
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Also save as JSON for programmatic access
        tprint_debug(f"   Generating JSON report...")
        json_report = {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'exchanges_tested': self.exchanges,
            'samples_collected': len(self.response_samples),
            'format_analyses': {
                rt.value: {
                    'common_fields': list(a.common_fields),
                    'exchange_formats': a.exchange_formats,
                    'field_analyses': {
                        field: {
                            'data_types': list(fa.data_types),
                            'present_in_exchanges': list(fa.present_in_exchanges),
                            'missing_in_exchanges': list(fa.missing_in_exchanges)
                        }
                        for field, fa in a.field_analyses.items()
                    },
                    'recommendations': a.standardization_recommendations
                }
                for rt, a in self.format_analyses.items()
            }
        }
        
        json_file = self.output_dir / "format_analysis_report.json"
        with open(json_file, 'w') as f:
            json.dump(json_report, f, indent=2, default=str)
        
        tprint_success(f"✅ Reports saved:")
        tprint_info(f"   Text report: {report_file}")
        tprint_info(f"   JSON report: {json_file}")
        tprint_info(f"   Analyzed {len(self.format_analyses)} response types")
    
    async def cleanup(self) -> None:
        """Cleanup exchange connections."""
        tprint_info("🧹 Cleaning up connections...")
        
        closed_count = 0
        error_count = 0
        
        for exchange_name, dispatcher in self.dispatchers.items():
            try:
                await dispatcher.close()
                closed_count += 1
                tprint_debug(f"   Closed {exchange_name.upper()}")
            except Exception as e:
                error_count += 1
                tprint_warning(f"   Error closing {exchange_name.upper()}: {e}")
                self.logger.warning(f"Error closing {exchange_name}: {e}")
        
        tprint_success(f"✅ Cleanup completed: {closed_count} exchanges closed")
        if error_count > 0:
            tprint_warning(f"   {error_count} errors during cleanup")


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Exchange API Format Analyzer - Test and analyze exchange API formats',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all exchanges in mock mode (parallel)
  python exchange_api_format_analyzer.py --mode mock
  
  # Analyze all exchanges sequentially (one by one)
  python exchange_api_format_analyzer.py --mode mock --sequential
  
  # Analyze only Binance exchange
  python exchange_api_format_analyzer.py --mode mock --single-exchange binance
  
  # Analyze specific exchanges sequentially
  python exchange_api_format_analyzer.py --mode real --exchanges binance okx --sequential
  
  # Analyze single exchange with custom symbols
  python exchange_api_format_analyzer.py --mode mock --single-exchange bingx --symbols BTCUSDT ETHUSDT ADAUSDT
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='mock',
        choices=['real', 'mock'],
        help='Test mode: real (actual exchange calls) or mock (mock data)'
    )
    
    parser.add_argument(
        '--exchanges',
        type=str,
        nargs='+',
        help='Exchanges to test (default: all supported)'
    )
    
    parser.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        default=['BTCUSDT', 'ETHUSDT'],
        help='Trading symbols to test'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='exchange_format_analysis',
        help='Output directory for analysis results'
    )
    
    parser.add_argument(
        '--sequential',
        action='store_true',
        help='Test exchanges one by one (sequential mode) instead of all at once'
    )
    
    parser.add_argument(
        '--single-exchange',
        type=str,
        help='Test only a single exchange (e.g., binance, okx, bingx)'
    )
    
    args = parser.parse_args()
    
    # Validate single exchange if provided
    if args.single_exchange:
        supported = [e.value for e in ExchangeType]
        if args.single_exchange.lower() not in supported:
            tprint_error(f"❌ Invalid exchange: {args.single_exchange}")
            tprint_info(f"   Supported exchanges: {', '.join(supported)}")
            return 1
    
    tprint_info("🚀 Exchange API Format Analyzer")
    tprint_info("=" * 70)
    tprint_info(f"Mode: {args.mode.upper()}")
    if args.single_exchange:
        tprint_info(f"Exchange: {args.single_exchange.upper()} (single exchange mode)")
    else:
        tprint_info(f"Exchanges: {', '.join(args.exchanges) if args.exchanges else 'all supported'}")
    tprint_info(f"Symbols: {', '.join(args.symbols)}")
    tprint_info(f"Output directory: {args.output_dir}")
    if args.sequential:
        tprint_info(f"Sequential mode: ON (testing exchanges one by one)")
    tprint_info("=" * 70)
    tprint_info("")
    
    # Create analyzer
    analyzer = ExchangeAPIFormatAnalyzer(
        test_symbols=args.symbols,
        exchanges=args.exchanges,
        mode=args.mode,
        output_dir=args.output_dir,
        sequential=args.sequential,
        single_exchange=args.single_exchange
    )
    
    try:
        # Initialize exchanges
        tprint_info("")
        await analyzer.initialize_exchanges()
        
        if not analyzer.dispatchers and not analyzer.sequential:
            tprint_error("❌ No exchanges initialized. Exiting.")
            return 1
        
        tprint_info("")
        
        # Collect API responses
        await analyzer.collect_api_responses()
        
        tprint_info("")
        
        # Analyze formats
        analyzer.analyze_formats()
        
        tprint_info("")
        
        # Generate and save report
        await analyzer.save_report()
        
        tprint_info("")
        
        # Print summary
        tprint_info("=" * 70)
        tprint_success("🎉 ANALYSIS COMPLETE")
        tprint_info("=" * 70)
        tprint_info("")
        tprint_info("Summary Report:")
        print(analyzer.generate_report())
        
        return 0
        
    except Exception as e:
        tprint_error(f"💥 Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        tprint_info("")
        await analyzer.cleanup()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
