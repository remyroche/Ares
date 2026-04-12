"""Binance Data Feed for Live Trading.

Fetches OHLCV data from Binance for margin-enabled assets
that pass universe.py filtering rules.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Callable

import pandas as pd
import numpy as np

try:
    from extreme_price_movements.universe import (
        fetch_binance_cross_margin_pairs,
        margin_pairs_to_spot_symbols,
        apply_hardcoded_universe_exclusions,
        deduplicate_symbols_by_base,
    )
    from extreme_price_movements.utils import tprint
except ImportError:
    # Fallback for direct import
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from extreme_price_movements.universe import (
        fetch_binance_cross_margin_pairs,
        margin_pairs_to_spot_symbols,
        apply_hardcoded_universe_exclusions,
        deduplicate_symbols_by_base,
    )
    from extreme_price_movements.utils import tprint


@dataclass
class DataFeedConfig:
    """Configuration for Binance data feed."""
    timeframe: str = "15m"  # Default to 15m bars
    lookback_bars: int = 200  # Number of bars to fetch
    update_interval_seconds: float = 60.0  # Poll interval
    max_concurrent_requests: int = 5  # Rate limiting
    quotes: Tuple[str, ...] = ("USDT",)  # Quote currencies to include


class BinanceDataFeed:
    """Real-time data feed from Binance for margin trading universe.
    
    Features:
    - Fetches margin-enabled symbols from Binance API
    - Applies universe.py filtering rules
    - Maintains OHLCV panel for all tracked symbols
    - Provides real-time updates with configurable intervals
    """
    
    def __init__(
        self,
        api_client: Any,  # Live trading API client
        config: Optional[DataFeedConfig] = None,
        symbol_filter: Optional[Callable[[str], bool]] = None,
    ):
        self.api_client = api_client
        self.config = config or DataFeedConfig()
        self.symbol_filter = symbol_filter  # Optional additional filter
        
        # Universe management
        self.margin_pairs: List[Dict[str, Any]] = []
        self.trading_symbols: List[str] = []
        self._symbol_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Data storage
        self.ohlcv_panel: Dict[str, pd.DataFrame] = {}
        self.last_update_time: Dict[str, datetime] = {}
        
        # Async management
        self._running = False
        self._update_task: Optional[asyncio.Task] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        
        # Callbacks
        self._on_data_update: List[Callable[[str, pd.DataFrame], None]] = []
        self._on_universe_change: List[Callable[[List[str]], None]] = []
    
    async def initialize(self) -> None:
        """Initialize data feed: fetch universe and initial data."""
        tprint("[DataFeed] Initializing Binance data feed...")
        
        # Initialize rate limiting semaphore
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        # Fetch margin-enabled universe
        await self._refresh_universe()
        
        # Fetch initial OHLCV data
        await self._fetch_all_ohlcv()
        
        tprint(f"[DataFeed] Initialized with {len(self.trading_symbols)} symbols")
    
    async def _refresh_universe(self) -> None:
        """Refresh the trading universe from Binance."""
        tprint("[DataFeed] Refreshing margin trading universe...")
        
        try:
            # Fetch margin pairs from Binance
            self.margin_pairs = fetch_binance_cross_margin_pairs()
            
            # Convert to spot symbols
            raw_symbols = margin_pairs_to_spot_symbols(
                self.margin_pairs, 
                quotes=self.config.quotes
            )
            
            # Apply universe.py filtering
            filtered_symbols = apply_hardcoded_universe_exclusions(raw_symbols)
            
            # Deduplicate by base asset
            self.trading_symbols = deduplicate_symbols_by_base(filtered_symbols)
            
            # Apply optional custom filter
            if self.symbol_filter:
                self.trading_symbols = [
                    s for s in self.trading_symbols 
                    if self.symbol_filter(s)
                ]
            
            # Build metadata map
            pair_map = {p["symbol"]: p for p in self.margin_pairs}
            for symbol in self.trading_symbols:
                # Convert spot format (BTC/USDT) to Binance format (BTCUSDT)
                binance_symbol = symbol.replace("/", "")
                self._symbol_metadata[symbol] = pair_map.get(binance_symbol, {})
            
            # Notify listeners
            for callback in self._on_universe_change:
                callback(self.trading_symbols)
            
            tprint(f"[DataFeed] Universe refreshed: {len(self.trading_symbols)} symbols")
            
        except Exception as e:
            tprint(f"[DataFeed] Error refreshing universe: {e}")
            raise
    
    async def _fetch_all_ohlcv(self) -> None:
        """Fetch OHLCV data for all symbols."""
        tprint(f"[DataFeed] Fetching OHLCV for {len(self.trading_symbols)} symbols...")
        
        tasks = [
            self._fetch_symbol_ohlcv(symbol)
            for symbol in self.trading_symbols
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = 0
        for symbol, result in zip(self.trading_symbols, results):
            if isinstance(result, Exception):
                tprint(f"[DataFeed] Error fetching {symbol}: {result}")
            elif result is not None:
                self.ohlcv_panel[symbol] = result
                self.last_update_time[symbol] = datetime.utcnow()
                success_count += 1
        
        tprint(f"[DataFeed] Fetched OHLCV for {success_count}/{len(self.trading_symbols)} symbols")
    
    async def _fetch_symbol_ohlcv(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data for a single symbol."""
        async with self._semaphore:
            try:
                # Convert spot format to Binance format
                binance_symbol = symbol.replace("/", "")
                
                # Calculate time range
                end_time = int(time.time() * 1000)
                # Approximate milliseconds per bar
                timeframe_ms = self._timeframe_to_ms(self.config.timeframe)
                start_time = end_time - (self.config.lookback_bars * timeframe_ms)
                
                # Fetch klines from Binance
                response = await self.api_client.get_klines(
                    symbol=binance_symbol,
                    interval=self.config.timeframe,
                    limit=self.config.lookback_bars,
                    start_time=start_time,
                    end_time=end_time
                )
                
                if not response.success:
                    return None
                
                # Parse klines data
                klines = response.data
                if not klines or not isinstance(klines, list):
                    return None
                
                # Binance kline format: [
                #   open_time, open, high, low, close, volume,
                #   close_time, quote_volume, trades, taker_buy_volume,
                #   taker_buy_quote_volume, ignore
                # ]
                data = []
                for k in klines:
                    if len(k) >= 6:
                        data.append({
                            "timestamp": pd.Timestamp(k[0], unit="ms"),
                            "open": float(k[1]),
                            "high": float(k[2]),
                            "low": float(k[3]),
                            "close": float(k[4]),
                            "volume": float(k[5]),
                        })
                
                if not data:
                    return None
                
                df = pd.DataFrame(data)
                df.set_index("timestamp", inplace=True)
                df.sort_index(inplace=True)
                
                return df
                
            except Exception as e:
                tprint(f"[DataFeed] Error fetching {symbol}: {e}")
                return None
    
    def _timeframe_to_ms(self, timeframe: str) -> int:
        """Convert timeframe string to milliseconds."""
        multipliers = {
            "m": 60 * 1000,
            "h": 60 * 60 * 1000,
            "d": 24 * 60 * 60 * 1000,
        }
        
        unit = timeframe[-1].lower()
        value = int(timeframe[:-1])
        
        return value * multipliers.get(unit, 60 * 1000)
    
    async def start(self) -> None:
        """Start the data feed update loop."""
        if self._running:
            return
        
        self._running = True
        self._update_task = asyncio.create_task(self._update_loop())
        tprint("[DataFeed] Started update loop")
    
    async def stop(self) -> None:
        """Stop the data feed update loop."""
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        tprint("[DataFeed] Stopped update loop")
    
    async def _update_loop(self) -> None:
        """Main update loop for continuous data refresh."""
        while self._running:
            try:
                # Refresh universe periodically (every 24h)
                if self._should_refresh_universe():
                    await self._refresh_universe()
                
                # Update OHLCV data
                await self._update_ohlcv()
                
                # Wait for next update
                await asyncio.sleep(self.config.update_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                tprint(f"[DataFeed] Update loop error: {e}")
                await asyncio.sleep(self.config.update_interval_seconds)
    
    def _should_refresh_universe(self) -> bool:
        """Check if universe should be refreshed (every 24 hours)."""
        # Universe refresh is handled by cache in universe.py
        # This method allows for periodic re-checks if needed
        return False  # Cache handles this
    
    async def _update_ohlcv(self) -> None:
        """Update OHLCV data for all symbols."""
        update_tasks = []
        
        for symbol in self.trading_symbols:
            # Check if update is needed (data is stale)
            last_update = self.last_update_time.get(symbol)
            if last_update is None or \
               (datetime.utcnow() - last_update).total_seconds() > self.config.update_interval_seconds:
                update_tasks.append(self._fetch_symbol_ohlcv(symbol))
        
        if not update_tasks:
            return
        
        results = await asyncio.gather(*update_tasks, return_exceptions=True)
        
        for symbol, result in zip(self.trading_symbols, results):
            if isinstance(result, Exception):
                continue
            if result is not None:
                # Merge with existing data to maintain continuity
                if symbol in self.ohlcv_panel:
                    existing = self.ohlcv_panel[symbol]
                    combined = pd.concat([existing, result])
                    combined = combined[~combined.index.duplicated(keep="last")]
                    combined.sort_index(inplace=True)
                    # Keep only lookback window
                    if len(combined) > self.config.lookback_bars:
                        combined = combined.iloc[-self.config.lookback_bars:]
                    self.ohlcv_panel[symbol] = combined
                else:
                    self.ohlcv_panel[symbol] = result
                
                self.last_update_time[symbol] = datetime.utcnow()
                
                # Notify listeners
                for callback in self._on_data_update:
                    callback(symbol, self.ohlcv_panel[symbol])
    
    def get_panel(self) -> Dict[str, pd.DataFrame]:
        """Get current OHLCV panel for all symbols."""
        return self.ohlcv_panel.copy()
    
    def get_symbol_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get OHLCV data for a specific symbol."""
        return self.ohlcv_panel.get(symbol)
    
    def get_trading_symbols(self) -> List[str]:
        """Get list of trading symbols."""
        return self.trading_symbols.copy()
    
    def register_data_callback(self, callback: Callable[[str, pd.DataFrame], None]) -> None:
        """Register callback for data updates."""
        self._on_data_update.append(callback)
    
    def register_universe_callback(self, callback: Callable[[List[str]], None]) -> None:
        """Register callback for universe changes."""
        self._on_universe_change.append(callback)
    
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Get metadata for a symbol."""
        return self._symbol_metadata.get(symbol, {})


__all__ = ["BinanceDataFeed", "DataFeedConfig"]
