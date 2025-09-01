# src/types/trading_types.py

"""Trading-specific type definitions for orders = positions, and trade decisions."""

from typing import Literal, TypedDict

from .base_types import (
ConfidenceLevel,
LeverageMultiplier,
Percentage,
PositionId,
Price,
RiskScore,
Score,
Symbol,
Timestamp,
Volume,
)

# Trading action enums
OrderType , Literal["market", "limit", "stop", "stop_limit", "trailing_stop"]
OrderSide = Literal["buy", "sell"]
OrderStatus = Literal[
"pending",
"open",
"filled",
"partially_filled",
"cancelled",
"rejected",
"expired",
]
PositionSide = Literal["long", "short", "neutral"]
TradeAction = Literal["open_long", "open_short", "close_long", "close_short", "hold"]
RiskLevel = Literal["very_low", "low", "medium", "high", "very_high"]


class OrderRequest(TypedDict):


    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="orderrequest initialization",
    )
    async def initialize(self) -> bool:
        """Initialize OrderRequest."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
    def __init__(self, config: dict[str, Any] | None = None) -> None:
 
    def __init__(self, config: dict[str, Any] | None = None) 
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize OrderRequest."""
        self.config = config or {}
        self.logger = system_logger.getChild("OrderRequest")
        self.is_initialized = False
-> None:
        """Initialize Orde
    def __init__(self, config: dict[str, Any] | None = None) -> None
    def __init__(self, config: dict[str, Any] | None = None) -
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize TradeDecision."""
        self.config = config or {}
        self.logger = system_logger.getChild("TradeDecision")
        self.is_initialized = False
> None:
        """Initialize TradeDecision."""
       
    def __init__(self, config: dict[str, Any] | None = None) -> Non
    def __init__(self, config: dict[str, Any] | None = None) 
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize PositionRisk."""
        self.config = config or {}
        self.logger = system_logger.getChild("PositionRisk")
        self.is_initialized = False
-> None:
        """Initi
    def __init__(self, config: dict[str, Any] | None = None) -> None:
    def __init__(self, config: dict[str, Any] | None = None) ->
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize RiskParameters."""
        self.config = config or {}
        self.logger = system_logger.getChild("RiskParameters")
        self.is_initialized = False
 None:
        """Initialize RiskParameters."""
    
    def __init__(self, config: dict[str, Any] | None = None) -> None
    def __init__(self, config: dict[str, Any] | None = None) -
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize TradingSignal."""
        self.config = config or {}
        self.logger = system_logger.getChild("TradingSignal")
        self.is_initialized = False
> None:
        """Initialize TradingSignal."""
        self.config = config or {}
        self.logger = system_logger.getChild
    def __init__(self, config: dict[str, Any] | None = None) -> None:
   
    def __init__(self, config: dict[str, Any] | None = None) -> Non
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize PerformanceMetrics."""
        self.config = config or {}
        self.logger = system_logger.getChild("PerformanceMetrics")
        self.is_initialized = False
e:
        """Initialize Perform
    def __init__(self, config: dict[str, Any] | None = None) -> None:
    def __init__(self, config: dict[str, Any] | None = None) ->
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize PortfolioState."""
        self.config = config or {}
        self.logger = system_logger.getChild("PortfolioState")
        self.is_initialized = False
 None:
        """Initialize
    def __init__(self, config: dict[str, Any] | None = None) -> None:
    def __init__(self, config: dict[str, Any] | None = None) ->
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize BacktestResult."""
        self.config = config or {}
        self.logger = system_logger.getChild("BacktestResult")
        self.is_initialized = False
 None:
        """Initialize BacktestResult."""
        self.config = config or {}
        self.logger = system_logger.getChild("BacktestResult")
        self.is_initialized = False

        """Initialize BacktestResult."""
        self.config = config or {}
        self.logger = system_logger.getChild("BacktestResult")
        self.is_initialized = False
 PortfolioState."""
        self.config = config or {}
        self.logger = system_logger.getChild("PortfolioState")
        self.is_initialized = False

        """Initialize PortfolioState."""
        self.config = config or {}
        self.logger = system_logger.getChild("PortfolioState")
        self.is_initialized = False
anceMetrics."""
        self.config = config or {}
        self.logger = system_logger.getChild("PerformanceMetrics")
        self.is_initialized = False
     """Initialize PerformanceMetrics."""
        self.config = config or {}
        self.logger = system_logger.getChild("PerformanceMetrics")
        self.is_initialized = False
("TradingSignal")
        self.is_initialized = False
:
        """Initialize TradingSignal."""
        self.config = config or {}
        self.logger = system_logger.getChild("TradingSignal")
        self.is_initialized = False
    self.config = config or {}
        self.logger = system_logger.getChild("RiskParameters")
        self.is_initialized = False

        """Initialize RiskParameters."""
        self.config = config or {}
        self.logger = system_logger.getChild("RiskParameters")
        self.is_initialized = False
alize PositionRisk."""
        self.config = config or {}
        self.logger = system_logger.getChild("PositionRisk")
        self.is_initialized = False
e:
        """Initialize PositionRisk."""
        self.config = config or {}
        self.logger = system_logger.getChild("PositionRisk")
        self.is_initialized = False
 self.config = config or {}
        self.logger = system_logger.getChild("TradeDecision")
        self.is_initialized = False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="orderrequest initialization",
    )
    async def initialize(self) -> bool:
        """Initialize OrderRequest."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
       
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="tradedecision initialization",
    )
    async def initialize(self) -> bool:
        """Initialize TradeDecision."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="positionrisk initialization",
    )
    async def initialize(self) -> bool:
        """Initialize PositionRisk."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized success
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="riskparameters initialization",
    )
    async def initialize(self) -> bool:
        """Initialize RiskParameters."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="tradingsignal initialization",
    )
    async def initialize(self) -> bool:
        """Initialize TradingSignal."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Err
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="performancemetrics initialization",
    )
    async def initialize(self) -> bool:
        """Initialize PerformanceMetrics."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
           
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="portfoliostate initialization",
    )
    async def initialize(self) -> bool:
        """Initialize PortfolioState."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="backtestresult initialization",
    )
    async def initialize(self) -> bool:
        """Initialize BacktestResult."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False

            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
 return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
or initializing {class_name}: {e}")
            return False

        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
fully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
     self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
:
        """Initialize TradeDecision."""
        self.config = config or {}
        self.logger = system_logger.getChild("TradeDecision")
        self.is_initialized = False
rRequest."""
        self.config = config or {}
        self.logger = system_logger.getChild("OrderRequest")
        self.is_initialized = False
       """Initialize OrderRequest."""
        self.config = config or {}
        self.logger = system_logger.getChild("OrderRequest")
        self.is_initialized = False
    passpass  # TODO: Add implementation
class OrderRequest(TypedDict):
    pass  # TODO: Add implementation
class OrderRequest(...):
    """..."""
    passsymbol: Symbol
side: OrderSide
type: OrderType
quantity: Volume
price: Price | None
stop_price: Price | None
time_in_force: Literal["GTC", "IOC", "FOK"] | None
reduce_only: bool | None
leverage: LeverageMultiplier | None


class TradeDecision(TypedDict):
    pass  # TODO: Add implementation
class TradeDecision(TypedDict):
    pass  # TODO: Add implementation
class TradeDecision(...):
    """..."""
    passtimestamp: Timestamp
symbol: Symbol
action: TradeAction
quantity: Volume
price: Price | None
leverage: LeverageMultiplier | None
stop_loss: Price | None
take_profit: Price | None
confidence: ConfidenceLevel
risk_score: RiskScore
reasoning: str


class PositionRisk(TypedDict):
    pass  # TODO: Add implementation
class PositionRisk(TypedDict):
    pass  # TODO: Add implementation
class PositionRisk(...):
    """..."""
    passposition_id: PositionId
symbol: Symbol
current_risk: RiskScore
max_loss_usd: float
liquidation_price: Price | None
margin_ratio: float
unrealized_pnl_percentage: Percentage
days_held: int
risk_level: RiskLevel


class RiskParameters(TypedDict):
    pass  # TODO: Add implementation
class RiskParameters(TypedDict):
    pass  # TODO: Add implementation
class RiskParameters(...):
    """..."""
    passmax_position_size: Volume
max_leverage: LeverageMultiplier
stop_loss_percentage: Percentage
take_profit_percentage: Percentage
max_drawdown: Percentage
max_daily_loss: float
position_correlation_limit: float
var_limit: float  # Value at Risk


class TradingSignal(TypedDict):
    pass  # TODO: Add implementation
class TradingSignal(TypedDict):
    pass  # TODO: Add implementation
class TradingSignal(...):
    """..."""
    passtimestamp: Timestamp
symbol: Symbol
signal_type: Literal["entry", "exit", "hold"]
direction: PositionSide | None
strength: Score  # 0.0 to 1.0
confidence: ConfidenceLevel
time_horizon: Literal["scalp", "short_term", "medium_term", "long_term"]
source: str  # e.g., "analyst", "ml_model", "technical_indicator"


class PerformanceMetrics(TypedDict):
    pass  # TODO: Add implementation
class PerformanceMetrics(TypedDict):
    pass  # TODO: Add implementation
class PerformanceMetrics(...):
    """..."""
    passtotal_return: Percentage
sharpe_ratio: float
sortino_ratio: float
max_drawdown: Percentage
win_rate: Percentage
profit_factor: float
average_win: float
average_loss: float
total_trades: int
winning_trades: int
losing_trades: int


class PortfolioState(TypedDict):
    pass  # TODO: Add implementation
class PortfolioState(TypedDict):
    pass  # TODO: Add implementation
class PortfolioState(...):
    """..."""
    passtimestamp: Timestamp
total_value: float
available_balance: float
unrealized_pnl: float
margin_used: float
positions: list[PositionRisk]
open_orders: list[OrderRequest]
daily_pnl: float
performance: PerformanceMetrics


class BacktestResult(TypedDict):
    pass  # TODO: Add implementation
class BacktestResult(TypedDict):
    pass  # TODO: Add implementation
class BacktestResult(...):
    """..."""
    passstart_date: Timestamp
end_date: Timestamp
initial_capital: float
final_value: float
performance: PerformanceMetrics
trades: list[TradeDecision]
daily_returns: list[float]
drawdown_periods: list[dict[str, Timestamp]]
config_used: dict[str, str]
