# src/types/data_types.py

"""Data structure type definitions for market data and trading information."""

from typing import Literal, TypedDict

from .base_types import (
OrderId,
PositionId,
Price,
Symbol,
Timestamp,
TradeId,
Volume,
)


class OHLCVData(TypedDict):


    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="ohlcvdata initialization",
    )
    async def initialize(self) -> bool:
        """Initialize OHLCVData."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
    def __init__(self, config: dict[str, Any] | None = None) -> None
    def __init__(self, config: dict[str, Any] | None = Non
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize OHLCVData."""
        self.config = config o
    def __init__(self, config: dict[str, Any] | None = None) -> N
    def __init__(self, config: dict[str, Any] | None = None
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize TickerData."""
        self.config = config or {}
        self.logger = system_lo
    def __init__(self, config: dict[str, Any] | None = None) -> None:
    def __init__(self, config: dict[str, Any] | None = None) ->
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Init
    def __init__(self, config: dict[str, Any] | None = None) -> None
    def __init__(self, config: dict[str, Any] | None = None) -
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize OrderBookData."""
        self.config = config or
    def __init__(self, config: dict[str, Any] | None = None) -> 
    def __init__(self, config: dict[str, Any] | None = Non
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize TradeData."""
        self.config = config or {}
        self.logger = sys
    def __init__(self, config: dict[str, Any] | None = None) -> No
    def __init__(self, config: dict[str, Any] | None = None)
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize AccountInfo."""
        self.config = config or {}
        self.logger = system_logger.getChild("AccountInfo")
        self.is_initialized = False
 -> None:
        """Initialize AccountInfo."""
        self.config = config or {}
        self.logg
    def __init__(self, config: dict[str, Any] | None = None) -> Non
    def __init__(self, config: dict[str, Any] | None = None) 
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize PositionInfo."""
        self.config = config or {}
        self.logger = system_logger.getChild("PositionInfo")
        self.is_initialized = False
-> 
    def __init__(self, config: dict[str, Any] | None = None) -> 
    def __init__(self, config: dict[str, Any] | None = Non
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize OrderInfo."""
        self.config = config or {}
        self.logger = system_logger.getChild("OrderInfo")
        self.is_initialized = False
e) -> None:
        """Initialize OrderInfo."""
        self.config = config or {}
        self.logger = system_logger.getChild("OrderInfo")
        self.is_initialized = False
None:
        """Initialize OrderInfo."""
        self.config = config or {}
        self.logger = system_logger.getChild("OrderInfo")
        self.is_initialized = False
None:
        """Initialize PositionInfo."""
        self.config = config or {}
        self.logger = system_logger.getChild("PositionInfo")
        self.is_initialized = False
e:
        """Initialize PositionInfo."""
        self.config = config or {}
        self.logger = system_logger.getChild("PositionInfo")
        self.is_initialized = False
er = system_logger.getChild("AccountInfo")
        self.is_initialized = False
ne:
        """Initialize AccountInfo."""
        self.config = config or {}
        self.logger = system_logger.getChild("AccountInfo")
        self.is_initialized = False
tem_logger.getChild("TradeData")
        self.is_initialized = False
e) -> None:
        """Initialize TradeData."""
        self.config = config or {}
        self.logger = system_logger.getChild("TradeData")
        self.is_initialized = False
None:
        """Initialize TradeData."""
        self.config = config or {}
        self.logger = system_logger.getChild("TradeData")
        self.is_initialized = False
 {}
        self.logger = system_logger.getChild("OrderBookData")
        self.is_initialized = False
> None:
        """Initialize OrderBookData."""
        self.config = config or {}
        self.logger = system_logger.getChild("OrderBookData")
        self.is_initialized = False
:
        """Initialize OrderBookData."""
        self.config = config or {}
        self.logger = system_logger.getChild("OrderBookData")
        self.is_initialized = False
ialize OrderBookLevel."""
        self.config = config or {}
        self.logger = system_logger.getChild("OrderBookLevel")
        self.is_initialized = False
 None:
        """Initialize OrderBookLevel."""
        self.config = config or {}
        self.logger = system_logger.getChild("OrderBookLevel")
        self.is_initialized = False

        """Initialize OrderBookLevel."""
        self.config = config or {}
        self.logger = system_logger.getChild("OrderBookLevel")
        self.is_initialized = False
gger.getChild("TickerData")
        self.is_initialized = False
) -> None:
        """Initialize TickerData."""
        self.config = config or {}
        self.logger = system_logger.getChild("TickerData")
        self.is_initialized = False
one:
        """Initialize TickerData."""
        self.config = config or {}
        self.logger = system_logger.getChild("Ticke
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="ohlcvdata initialization",
    )
    async def initialize(self) -> bool:
        """Initialize 
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="tickerdata initialization",
    )
    async def initialize(self) -> bool:
        """Initialize TickerData."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
           
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="orderbooklevel initialization",
    )
    async def initialize(self) -> bool:
        """Initialize OrderBookLevel."""

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="orderbookdata initialization",
    )
    async def initialize(self) -> bool:
        """Initialize OrderBookData."""
        try:
            self.logger.info(f"🚀 Initializi
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="tradedata initialization",
    )
    async def initialize(self) -> bool:
        """Initialize TradeData."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="accountinfo initialization",
    )
    async def initialize(self) -> bool:
        """Initialize AccountInfo."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
          
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="positioninfo initialization",
    )
    async def initialize(self) -> bool:
        """Initialize PositionInfo."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_nam
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="orderinfo initialization",
    )
    async def initialize(self) -> bool:
        """Initialize OrderInfo."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
e} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
  self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
        self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
ng {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
 self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
OHLCVData."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
rData")
        self.is_initialized = False
r {}
        self.logger = system_logger.getChild("OHLCVData")
        self.is_initialized = False
e) -> None:
        """Initialize OHLCVData."""
        self.config = config or {}
        self.logger = system_logger.getChild("OHLCVData")
        self.is_initialized = False
:
        """Initialize OHLCVData."""
        self.config = config or {}
        self.logger = system_logger.getChild("OHLCVData")
        self.is_initialized = False
    passpass  # TODO: Add implementation
class OHLCVData(TypedDict):
    pass  # TODO: Add implementation
class OHLCVData(...):
    """..."""
    passtimestamp: Timestamp
open: Price
high: Price
low: Price
close: Price
volume: Volume


class TickerData(TypedDict):
    pass  # TODO: Add implementation
class TickerData(TypedDict):
    pass  # TODO: Add implementation
class TickerData(...):
    """..."""
    passsymbol: Symbol
price: Price
change_24h: float
volume_24h: Volume
high_24h: Price
low_24h: Price
timestamp: Timestamp


class OrderBookLevel(TypedDict):
    pass  # TODO: Add implementation
class OrderBookLevel(TypedDict):
    pass  # TODO: Add implementation
class OrderBookLevel(...):
    """..."""
    passprice: Price
quantity: Volume


class OrderBookData(TypedDict):
    pass  # TODO: Add implementation
class OrderBookData(TypedDict):
    pass  # TODO: Add implementation
class OrderBookData(...):
    """..."""
    passsymbol: Symbol
timestamp: Timestamp
bids: list[OrderBookLevel]
asks: list[OrderBookLevel]


class TradeData(TypedDict):
    pass  # TODO: Add implementation
class TradeData(TypedDict):
    pass  # TODO: Add implementation
class TradeData(...):
    """..."""
    passtrade_id: TradeId
symbol: Symbol
price: Price
quantity: Volume
side: Literal["buy", "sell"]
timestamp: Timestamp


class AccountInfo(TypedDict):
    pass  # TODO: Add implementation
class AccountInfo(TypedDict):
    pass  # TODO: Add implementation
class AccountInfo(...):
    """..."""
    passaccount_id: str
total_balance: float
available_balance: float
margin_balance: float | None
unrealized_pnl: float | None
margin_ratio: float | None
positions: list[dict[str, float]]  # Will be typed more specifically
open_orders: list[dict[str, str]]  # Will be typed more specifically


class PositionInfo(TypedDict):
    pass  # TODO: Add implementation
class PositionInfo(TypedDict):
    pass  # TODO: Add implementation
class PositionInfo(...):
    """..."""
    passposition_id: PositionId
symbol: Symbol
side: Literal["long", "short"]
size: Volume
entry_price: Price
mark_price: Price
unrealized_pnl: float
leverage: float
margin: float
timestamp: Timestamp


class OrderInfo(TypedDict):
    pass  # TODO: Add implementation
class OrderInfo(TypedDict):
    pass  # TODO: Add implementation
class OrderInfo(...):
    """..."""
    passorder_id: OrderId
symbol: Symbol
side: Literal["buy", "sell"]
type: Literal["market", "limit", "stop", "stop_limit"]
quantity: Volume
price: Price | None
stop_price: Price | None
status: Literal["pending", "open", "filled", "cancelled", "rejected"]
filled_quantity: Volume
timestamp: Timestamp


# Aggregate types for convenience
MarketDataDict = dict[Symbol, list[OHLCVData]]
TickerDict = dict[Symbol, TickerData]
OrderBookDict = dict[Symbol, OrderBookData]
