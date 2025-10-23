"""
Instrument Management Utilities

Handles instrument specifications, contract details, and trading parameters.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

from .market_metadata import InstrumentType, InstrumentSpec

from src.utils.logger import system_logger


class ContractStatus(Enum):
    """Contract status enumeration"""
    ACTIVE = "active"
    EXPIRED = "expired"
    SETTLED = "settled"
    SUSPENDED = "suspended"


@dataclass
class ContractSpec:
    """Contract specification structure"""
    symbol: str
    contract_type: str
    delivery_date: Optional[datetime]
    settlement_currency: str
    contract_size: float
    tick_size: float
    lot_size: float
    status: ContractStatus
    underlying_asset: str
    strike_price: Optional[float] = None  # For options
    option_type: Optional[str] = None  # call/put for options


class InstrumentManager:
    """
    Manages instrument specifications and contract details.
    """
    
    def __init__(self, exchange_name: str):
        self.exchange_name = exchange_name
        self.logger = system_logger.getChild(f"InstrumentManager.{exchange_name}")
        self.instruments: Dict[str, InstrumentSpec] = {}
        self.contracts: Dict[str, ContractSpec] = {}
        self.symbol_mappings: Dict[str, str] = {}  # normalized -> exchange symbol
        
    def add_instrument(self, instrument: InstrumentSpec) -> None:
        """Add instrument specification."""
        self.instruments[instrument.symbol] = instrument
        self.symbol_mappings[instrument.symbol.upper()] = instrument.symbol
        self.logger.debug(f"Added instrument {instrument.symbol}")
    
    def get_instrument(self, symbol: str) -> Optional[InstrumentSpec]:
        """Get instrument by symbol."""
        # Try direct lookup
        if symbol in self.instruments:
            return self.instruments[symbol]
        
        # Try case-insensitive lookup
        symbol_upper = symbol.upper()
        if symbol_upper in self.symbol_mappings:
            mapped_symbol = self.symbol_mappings[symbol_upper]
            return self.instruments.get(mapped_symbol)
        
        return None
    
    def get_instruments_by_type(self, instrument_type: InstrumentType) -> List[InstrumentSpec]:
        """Get instruments by type."""
        return [
            inst for inst in self.instruments.values()
            if inst.instrument_type == instrument_type and inst.is_active
        ]
    
    def get_spot_instruments(self) -> List[InstrumentSpec]:
        """Get all spot trading instruments."""
        return self.get_instruments_by_type(InstrumentType.SPOT)
    
    def get_futures_instruments(self) -> List[InstrumentSpec]:
        """Get all futures instruments."""
        return self.get_instruments_by_type(InstrumentType.FUTURES)
    
    def get_perpetual_instruments(self) -> List[InstrumentSpec]:
        """Get all perpetual swap instruments."""
        return self.get_instruments_by_type(InstrumentType.PERPETUAL)
    
    def get_options_instruments(self) -> List[InstrumentSpec]:
        """Get all options instruments."""
        return self.get_instruments_by_type(InstrumentType.OPTIONS)
    
    def get_margin_instruments(self) -> List[InstrumentSpec]:
        """Get all margin trading instruments."""
        return self.get_instruments_by_type(InstrumentType.MARGIN)
    
    def get_instruments_by_base_currency(self, base_currency: str) -> List[InstrumentSpec]:
        """Get instruments by base currency."""
        return [
            inst for inst in self.instruments.values()
            if inst.base_currency.upper() == base_currency.upper() and inst.is_active
        ]
    
    def get_instruments_by_quote_currency(self, quote_currency: str) -> List[InstrumentSpec]:
        """Get instruments by quote currency."""
        return [
            inst for inst in self.instruments.values()
            if inst.quote_currency.upper() == quote_currency.upper() and inst.is_active
        ]
    
    def get_trading_pairs(self, base_currency: str, quote_currency: str) -> List[InstrumentSpec]:
        """Get trading pairs for specific currencies."""
        return [
            inst for inst in self.instruments.values()
            if (inst.base_currency.upper() == base_currency.upper() and
                inst.quote_currency.upper() == quote_currency.upper() and
                inst.is_active)
        ]
    
    def get_available_quote_currencies(self) -> Set[str]:
        """Get all available quote currencies."""
        return {
            inst.quote_currency for inst in self.instruments.values()
            if inst.is_active
        }
    
    def get_available_base_currencies(self) -> Set[str]:
        """Get all available base currencies."""
        return {
            inst.base_currency for inst in self.instruments.values()
            if inst.is_active
        }
    
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
        return instrument.min_notional if instrument else None
    
    def get_maximum_order_size(self, symbol: str) -> Optional[float]:
        """Get maximum order size for symbol."""
        instrument = self.get_instrument(symbol)
        return instrument.max_notional if instrument else None
    
    def get_maximum_leverage(self, symbol: str) -> Optional[float]:
        """Get maximum leverage for symbol."""
        instrument = self.get_instrument(symbol)
        return instrument.max_leverage if instrument else None
    
    def get_tick_size(self, symbol: str) -> Optional[float]:
        """Get tick size for symbol."""
        instrument = self.get_instrument(symbol)
        return instrument.tick_size if instrument else None
    
    def get_lot_size(self, symbol: str) -> Optional[float]:
        """Get lot size for symbol."""
        instrument = self.get_instrument(symbol)
        return instrument.lot_size if instrument else None
    
    def get_price_precision(self, symbol: str) -> int:
        """Get price precision for symbol."""
        instrument = self.get_instrument(symbol)
        return instrument.price_precision if instrument else 8
    
    def get_quantity_precision(self, symbol: str) -> int:
        """Get quantity precision for symbol."""
        instrument = self.get_instrument(symbol)
        return instrument.quantity_precision if instrument else 8
    
    def get_margin_ratio(self, symbol: str) -> Optional[float]:
        """Get margin ratio for symbol."""
        instrument = self.get_instrument(symbol)
        return instrument.margin_ratio if instrument else None
    
    def get_liquidation_ratio(self, symbol: str) -> Optional[float]:
        """Get liquidation ratio for symbol."""
        instrument = self.get_instrument(symbol)
        return instrument.liquidation_ratio if instrument else None
    
    def get_contract_size(self, symbol: str) -> Optional[float]:
        """Get contract size for symbol."""
        instrument = self.get_instrument(symbol)
        return instrument.contract_size if instrument else None
    
    def get_settlement_currency(self, symbol: str) -> Optional[str]:
        """Get settlement currency for symbol."""
        instrument = self.get_instrument(symbol)
        return instrument.settlement_currency if instrument else None
    
    def get_delivery_date(self, symbol: str) -> Optional[datetime]:
        """Get delivery date for symbol."""
        instrument = self.get_instrument(symbol)
        return instrument.delivery_date if instrument else None
    
    def is_futures_contract(self, symbol: str) -> bool:
        """Check if symbol is a futures contract."""
        instrument = self.get_instrument(symbol)
        return instrument and instrument.instrument_type == InstrumentType.FUTURES
    
    def is_perpetual_contract(self, symbol: str) -> bool:
        """Check if symbol is a perpetual contract."""
        instrument = self.get_instrument(symbol)
        return instrument and instrument.instrument_type == InstrumentType.PERPETUAL
    
    def is_options_contract(self, symbol: str) -> bool:
        """Check if symbol is an options contract."""
        instrument = self.get_instrument(symbol)
        return instrument and instrument.instrument_type == InstrumentType.OPTIONS
    
    def is_spot_instrument(self, symbol: str) -> bool:
        """Check if symbol is a spot instrument."""
        instrument = self.get_instrument(symbol)
        return instrument and instrument.instrument_type == InstrumentType.SPOT
    
    def is_margin_instrument(self, symbol: str) -> bool:
        """Check if symbol supports margin trading."""
        instrument = self.get_instrument(symbol)
        return instrument and instrument.instrument_type == InstrumentType.MARGIN
    
    def get_instruments_with_leverage(self, min_leverage: float = 1.0) -> List[InstrumentSpec]:
        """Get instruments that support leverage."""
        return [
            inst for inst in self.instruments.values()
            if (inst.is_active and
                inst.max_leverage and
                inst.max_leverage >= min_leverage)
        ]
    
    def get_instruments_by_leverage_range(self, min_leverage: float, max_leverage: float) -> List[InstrumentSpec]:
        """Get instruments within leverage range."""
        return [
            inst for inst in self.instruments.values()
            if (inst.is_active and
                inst.max_leverage and
                min_leverage <= inst.max_leverage <= max_leverage)
        ]
    
    def search_instruments(
        self,
        base_currency: Optional[str] = None,
        quote_currency: Optional[str] = None,
        instrument_type: Optional[InstrumentType] = None,
        min_leverage: Optional[float] = None,
        max_leverage: Optional[float] = None,
        min_tick_size: Optional[float] = None,
        max_tick_size: Optional[float] = None,
        min_lot_size: Optional[float] = None,
        max_lot_size: Optional[float] = None
    ) -> List[InstrumentSpec]:
        """Search instruments with multiple filters."""
        results = []
        
        for inst in self.instruments.values():
            if not inst.is_active:
                continue
            
            # Apply filters
            if base_currency and inst.base_currency.upper() != base_currency.upper():
                continue
            
            if quote_currency and inst.quote_currency.upper() != quote_currency.upper():
                continue
            
            if instrument_type and inst.instrument_type != instrument_type:
                continue
            
            if min_leverage and (not inst.max_leverage or inst.max_leverage < min_leverage):
                continue
            
            if max_leverage and inst.max_leverage and inst.max_leverage > max_leverage:
                continue
            
            if min_tick_size and inst.tick_size < min_tick_size:
                continue
            
            if max_tick_size and inst.tick_size > max_tick_size:
                continue
            
            if min_lot_size and inst.lot_size < min_lot_size:
                continue
            
            if max_lot_size and inst.lot_size > max_lot_size:
                continue
            
            results.append(inst)
        
        return results
    
    def get_instrument_statistics(self) -> Dict[str, Any]:
        """Get instrument statistics."""
        total_instruments = len(self.instruments)
        active_instruments = len([inst for inst in self.instruments.values() if inst.is_active])
        
        type_counts = {}
        for inst in self.instruments.values():
            inst_type = inst.instrument_type.value
            type_counts[inst_type] = type_counts.get(inst_type, 0) + 1
        
        leverage_instruments = len(self.get_instruments_with_leverage())
        
        return {
            "total_instruments": total_instruments,
            "active_instruments": active_instruments,
            "inactive_instruments": total_instruments - active_instruments,
            "type_distribution": type_counts,
            "leverage_instruments": leverage_instruments,
            "quote_currencies": len(self.get_available_quote_currencies()),
            "base_currencies": len(self.get_available_base_currencies())
        }
    
    def update_instrument_status(self, symbol: str, is_active: bool) -> bool:
        """Update instrument active status."""
        instrument = self.get_instrument(symbol)
        if not instrument:
            return False
        
        instrument.is_active = is_active
        self.logger.info(f"Updated instrument {symbol} status to {'active' if is_active else 'inactive'}")
        return True
    
    def deactivate_instrument(self, symbol: str) -> bool:
        """Deactivate instrument."""
        return self.update_instrument_status(symbol, False)
    
    def activate_instrument(self, symbol: str) -> bool:
        """Activate instrument."""
        return self.update_instrument_status(symbol, True)
    
    def cleanup_inactive_instruments(self) -> int:
        """Remove inactive instruments from memory."""
        inactive_symbols = [
            symbol for symbol, inst in self.instruments.items()
            if not inst.is_active
        ]
        
        for symbol in inactive_symbols:
            del self.instruments[symbol]
            self.symbol_mappings.pop(symbol.upper(), None)
        
        if inactive_symbols:
            self.logger.info(f"Cleaned up {len(inactive_symbols)} inactive instruments")
        
        return len(inactive_symbols)
    
    def get_instrument_summary(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive instrument summary."""
        instrument = self.get_instrument(symbol)
        if not instrument:
            return None
        
        return {
            "symbol": instrument.symbol,
            "base_currency": instrument.base_currency,
            "quote_currency": instrument.quote_currency,
            "instrument_type": instrument.instrument_type.value,
            "status": instrument.status,
            "is_active": instrument.is_active,
            "tick_size": instrument.tick_size,
            "lot_size": instrument.lot_size,
            "min_notional": instrument.min_notional,
            "max_notional": instrument.max_notional,
            "price_precision": instrument.price_precision,
            "quantity_precision": instrument.quantity_precision,
            "max_leverage": instrument.max_leverage,
            "margin_ratio": instrument.margin_ratio,
            "liquidation_ratio": instrument.liquidation_ratio,
            "contract_size": instrument.contract_size,
            "settlement_currency": instrument.settlement_currency,
            "delivery_date": instrument.delivery_date.isoformat() if instrument.delivery_date else None,
            "last_updated": instrument.last_updated.isoformat()
        }