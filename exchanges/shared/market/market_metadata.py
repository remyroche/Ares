"""
Market Metadata Management

Handles market data, instrument specifications, and metadata caching.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

from src.utils.logger import system_logger


class InstrumentType(Enum):
    """Instrument type enumeration"""
    SPOT = "spot"
    FUTURES = "futures"
    PERPETUAL = "perpetual"
    OPTIONS = "options"
    MARGIN = "margin"


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill


@dataclass
class InstrumentSpec:
    """Instrument specification structure"""
    symbol: str
    base_currency: str
    quote_currency: str
    instrument_type: InstrumentType
    status: str  # active, suspended, etc.
    
    # Trading specifications
    tick_size: float
    lot_size: float
    min_notional: float
    
    # Precision
    price_precision: int
    quantity_precision: int
    
    # Optional trading specifications
    max_notional: Optional[float] = None
    
    # Risk specifications
    max_leverage: Optional[float] = None
    margin_ratio: Optional[float] = None
    liquidation_ratio: Optional[float] = None
    
    # Contract specifications
    contract_size: Optional[float] = None
    settlement_currency: Optional[str] = None
    delivery_date: Optional[datetime] = None
    
    # Trading hours
    trading_hours: Dict[str, Any] = field(default_factory=dict)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Cache info
    last_updated: datetime = field(default_factory=datetime.now)
    is_active: bool = True


class MarketMetadataManager:
    """
    Manages market metadata, instrument specifications, and market data.
    """
    
    def __init__(self, exchange_name: str):
        self.exchange_name = exchange_name
        self.logger = system_logger.getChild(f"MarketMetadataManager.{exchange_name}")
        
        # Data storage
        self.instruments: Dict[str, InstrumentSpec] = {}
        self.market_data: Dict[str, Dict[str, Any]] = {}
        self.symbol_cache: Dict[str, str] = {}  # symbol -> exchange_symbol mapping
        
        # Cache settings
        self.cache_ttl = timedelta(minutes=5)
        self.last_refresh: Optional[datetime] = None
        
        # Refresh functions
        self.refresh_functions: Dict[str, callable] = {}
        
    def register_refresh_functions(
        self,
        get_instruments: callable,
        get_ticker: callable,
        get_orderbook: Optional[callable] = None,
        get_funding_rate: Optional[callable] = None
    ) -> None:
        """
        Register exchange-specific refresh functions.
        
        Args:
            get_instruments: Function to get instrument specifications
            get_ticker: Function to get ticker data
            get_orderbook: Optional function to get order book data
            get_funding_rate: Optional function to get funding rate data
        """
        self.refresh_functions = {
            "get_instruments": get_instruments,
            "get_ticker": get_ticker,
            "get_orderbook": get_orderbook,
            "get_funding_rate": get_funding_rate
        }
        
        self.logger.info("Registered market data refresh functions")
    
    async def refresh_instruments(self) -> bool:
        """Refresh instrument specifications from exchange."""
        try:
            if "get_instruments" not in self.refresh_functions:
                self.logger.warning("No instruments refresh function registered")
                return False
                
            instruments_data = await self.refresh_functions["get_instruments"]()
            if not instruments_data:
                self.logger.warning("No instruments data received")
                return False
                
            # Clear existing instruments
            self.instruments.clear()
            
            # Process instruments data
            for instrument_data in instruments_data:
                try:
                    spec = self._parse_instrument_data(instrument_data)
                    if spec:
                        self.instruments[spec.symbol] = spec
                        # Cache symbol mapping
                        self.symbol_cache[spec.symbol.upper()] = spec.symbol
                        
                except Exception as e:
                    self.logger.warning(f"Failed to parse instrument data: {e}")
                    continue
            
            self.last_refresh = datetime.now()
            self.logger.info(f"Refreshed {len(self.instruments)} instruments")
            return True
            
        except Exception as e:
            self.logger.error(f"Error refreshing instruments: {e}")
            return False
    
    def _parse_instrument_data(self, data: Dict[str, Any]) -> Optional[InstrumentSpec]:
        """Parse exchange-specific instrument data into InstrumentSpec."""
        try:
            # Extract basic information
            symbol = data.get("symbol") or data.get("instId", "")
            if not symbol:
                return None
                
            # Parse instrument type
            inst_type_str = data.get("type", "").lower()
            if "spot" in inst_type_str:
                inst_type = InstrumentType.SPOT
            elif "futures" in inst_type_str or "swap" in inst_type_str:
                inst_type = InstrumentType.FUTURES
            elif "perpetual" in inst_type_str:
                inst_type = InstrumentType.PERPETUAL
            elif "options" in inst_type_str:
                inst_type = InstrumentType.OPTIONS
            elif "margin" in inst_type_str:
                inst_type = InstrumentType.MARGIN
            else:
                inst_type = InstrumentType.SPOT  # Default
            
            # Extract currencies
            base_currency = data.get("baseCcy", "") or data.get("base", "")
            quote_currency = data.get("quoteCcy", "") or data.get("quote", "")
            
            # Extract trading specifications
            tick_size = float(data.get("tickSz", 0) or data.get("tickSize", 0))
            lot_size = float(data.get("lotSz", 0) or data.get("lotSize", 0))
            min_notional = float(data.get("minSz", 0) or data.get("minNotional", 0))
            
            # Extract precision
            price_precision = int(data.get("tickSz", "0").count("0") if isinstance(data.get("tickSz"), str) else 8)
            quantity_precision = int(data.get("lotSz", "0").count("0") if isinstance(data.get("lotSz"), str) else 8)
            
            # Extract risk specifications
            max_leverage = data.get("lever", data.get("maxLeverage"))
            if max_leverage:
                max_leverage = float(max_leverage)
                
            # Extract contract specifications
            contract_size = data.get("ctVal", data.get("contractSize"))
            if contract_size:
                contract_size = float(contract_size)
                
            settlement_currency = data.get("settleCcy", data.get("settlementCurrency"))
            
            # Create instrument specification
            spec = InstrumentSpec(
                symbol=symbol,
                base_currency=base_currency,
                quote_currency=quote_currency,
                instrument_type=inst_type,
                status=data.get("state", "active"),
                tick_size=tick_size,
                lot_size=lot_size,
                min_notional=min_notional,
                max_notional=data.get("maxSz", data.get("maxNotional")),
                price_precision=price_precision,
                quantity_precision=quantity_precision,
                max_leverage=max_leverage,
                margin_ratio=data.get("marginRatio"),
                liquidation_ratio=data.get("liquidationRatio"),
                contract_size=contract_size,
                settlement_currency=settlement_currency,
                metadata=data
            )
            
            return spec
            
        except Exception as e:
            self.logger.warning(f"Error parsing instrument data: {e}")
            return None
    
    async def refresh_market_data(self, symbols: Optional[List[str]] = None) -> bool:
        """Refresh market data for specified symbols."""
        try:
            if "get_ticker" not in self.refresh_functions:
                self.logger.warning("No ticker refresh function registered")
                return False
                
            if symbols is None:
                symbols = list(self.instruments.keys())
                
            for symbol in symbols:
                try:
                    ticker_data = await self.refresh_functions["get_ticker"](symbol)
                    if ticker_data:
                        self.market_data[symbol] = {
                            "ticker": ticker_data,
                            "last_updated": datetime.now()
                        }
                        
                except Exception as e:
                    self.logger.warning(f"Failed to refresh market data for {symbol}: {e}")
                    continue
            
            self.logger.info(f"Refreshed market data for {len(symbols)} symbols")
            return True
            
        except Exception as e:
            self.logger.error(f"Error refreshing market data: {e}")
            return False
    
    def get_instrument(self, symbol: str) -> Optional[InstrumentSpec]:
        """Get instrument specification by symbol."""
        # Try direct lookup first
        if symbol in self.instruments:
            return self.instruments[symbol]
            
        # Try case-insensitive lookup
        symbol_upper = symbol.upper()
        if symbol_upper in self.symbol_cache:
            mapped_symbol = self.symbol_cache[symbol_upper]
            return self.instruments.get(mapped_symbol)
            
        return None
    
    def get_instruments_by_type(self, instrument_type: InstrumentType) -> List[InstrumentSpec]:
        """Get instruments by type."""
        return [
            spec for spec in self.instruments.values()
            if spec.instrument_type == instrument_type and spec.is_active
        ]
    
    def get_active_instruments(self) -> List[InstrumentSpec]:
        """Get all active instruments."""
        return [
            spec for spec in self.instruments.values()
            if spec.is_active and spec.status == "active"
        ]
    
    def get_trading_pairs(self, base_currency: str) -> List[InstrumentSpec]:
        """Get trading pairs for a base currency."""
        return [
            spec for spec in self.instruments.values()
            if spec.base_currency.upper() == base_currency.upper() and spec.is_active
        ]
    
    def get_quote_currencies(self) -> Set[str]:
        """Get all quote currencies."""
        return {
            spec.quote_currency for spec in self.instruments.values()
            if spec.is_active
        }
    
    def get_base_currencies(self) -> Set[str]:
        """Get all base currencies."""
        return {
            spec.base_currency for spec in self.instruments.values()
            if spec.is_active
        }
    
    def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached market data for symbol."""
        return self.market_data.get(symbol)
    
    def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get ticker data for symbol."""
        market_data = self.get_market_data(symbol)
        if market_data:
            return market_data.get("ticker")
        return None
    
    def is_symbol_tradable(self, symbol: str) -> bool:
        """Check if symbol is tradable."""
        instrument = self.get_instrument(symbol)
        if not instrument:
            return False
            
        return (
            instrument.is_active and
            instrument.status == "active" and
            instrument.tick_size > 0 and
            instrument.lot_size > 0
        )
    
    def get_minimum_order_size(self, symbol: str) -> Optional[float]:
        """Get minimum order size for symbol."""
        instrument = self.get_instrument(symbol)
        if not instrument:
            return None
            
        return instrument.min_notional
    
    def get_maximum_leverage(self, symbol: str) -> Optional[float]:
        """Get maximum leverage for symbol."""
        instrument = self.get_instrument(symbol)
        if not instrument:
            return None
            
        return instrument.max_leverage
    
    def get_price_precision(self, symbol: str) -> int:
        """Get price precision for symbol."""
        instrument = self.get_instrument(symbol)
        if not instrument:
            return 8  # Default precision
            
        return instrument.price_precision
    
    def get_quantity_precision(self, symbol: str) -> int:
        """Get quantity precision for symbol."""
        instrument = self.get_instrument(symbol)
        if not instrument:
            return 8  # Default precision
            
        return instrument.quantity_precision
    
    def should_refresh(self) -> bool:
        """Check if data should be refreshed."""
        if not self.last_refresh:
            return True
            
        return datetime.now() - self.last_refresh > self.cache_ttl
    
    async def ensure_fresh_data(self) -> bool:
        """Ensure data is fresh, refresh if needed."""
        if self.should_refresh():
            return await self.refresh_instruments()
        return True
    
    def search_instruments(
        self,
        base_currency: Optional[str] = None,
        quote_currency: Optional[str] = None,
        instrument_type: Optional[InstrumentType] = None,
        min_leverage: Optional[float] = None,
        max_leverage: Optional[float] = None
    ) -> List[InstrumentSpec]:
        """Search instruments with filters."""
        results = []
        
        for spec in self.instruments.values():
            if not spec.is_active:
                continue
                
            # Apply filters
            if base_currency and spec.base_currency.upper() != base_currency.upper():
                continue
                
            if quote_currency and spec.quote_currency.upper() != quote_currency.upper():
                continue
                
            if instrument_type and spec.instrument_type != instrument_type:
                continue
                
            if min_leverage and (not spec.max_leverage or spec.max_leverage < min_leverage):
                continue
                
            if max_leverage and spec.max_leverage and spec.max_leverage > max_leverage:
                continue
                
            results.append(spec)
            
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get market metadata statistics."""
        total_instruments = len(self.instruments)
        active_instruments = len(self.get_active_instruments())
        
        type_counts = {}
        for spec in self.instruments.values():
            inst_type = spec.instrument_type.value
            type_counts[inst_type] = type_counts.get(inst_type, 0) + 1
        
        return {
            "total_instruments": total_instruments,
            "active_instruments": active_instruments,
            "inactive_instruments": total_instruments - active_instruments,
            "type_distribution": type_counts,
            "quote_currencies": len(self.get_quote_currencies()),
            "base_currencies": len(self.get_base_currencies()),
            "last_refresh": self.last_refresh.isoformat() if self.last_refresh else None,
            "cache_ttl_minutes": self.cache_ttl.total_seconds() / 60
        }
    
    def cleanup_old_data(self, max_age_hours: int = 24) -> int:
        """Clean up old market data."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        cleaned_count = 0
        
        # Clean up old market data
        symbols_to_remove = []
        for symbol, data in self.market_data.items():
            if data.get("last_updated", datetime.min) < cutoff_time:
                symbols_to_remove.append(symbol)
                
        for symbol in symbols_to_remove:
            del self.market_data[symbol]
            cleaned_count += 1
            
        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} old market data entries")
            
        return cleaned_count