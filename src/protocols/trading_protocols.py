# src/protocols/trading_protocols.py

"""
Enhanced trading system protocols with comprehensive type safety.
"""

from abc import abstractmethod
from typing import (
    Protocol,
    runtime_checkable,
)

from src.custom_types import (
    AccountInfo,
    MarketDataDict,
    ModelInput,
    OrderInfo,
    OrderRequest,
    PerformanceMetrics,
    PositionInfo,
    PredictionResult,
    RegimeClassification,
    RiskParameters,
    Symbol,
    Timestamp,
    TradeDecision,
    TradingSignal,
)


@runtime_checkable
class TradingDataProvider(Protocol):
    """Protocol for trading data providers."""

    @abstractmethod
    async def get_market_data(
        self,
        symbol: Symbol,
        start_time: Timestamp,
        end_time: Timestamp,
    ) -> MarketDataDict:
        """Get market data for the specified time range."""
        ...

    @abstractmethod
    async def get_live_data(self, symbol: Symbol) -> MarketDataDict:
        """Get live market data."""
        ...

    @abstractmethod
    async def get_account_info(self) -> AccountInfo:
        """Get account information."""
        ...

    @abstractmethod
    async def get_positions(self) -> list[PositionInfo]:
        """Get current positions."""
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if provider is connected."""
        ...


@runtime_checkable
class TradingMLPredictor(Protocol):
    """Protocol for ML trading predictors."""

    @abstractmethod
    async def predict_market_direction(
        self,
        input_data: ModelInput,
    ) -> PredictionResult:
        """Predict market direction."""
        ...

    @abstractmethod
    async def classify_regime(self, input_data: ModelInput) -> RegimeClassification:
        """Classify market regime."""
        ...

    @abstractmethod
    async def generate_signals(self, input_data: ModelInput) -> list[TradingSignal]:
        """Generate trading signals."""
        ...

    @abstractmethod
    def get_model_confidence(self) -> float:
        """Get current model confidence."""
        ...

    @abstractmethod
    def is_model_ready(self) -> bool:
        """Check if model is ready for prediction."""
        ...


@runtime_checkable
class TradingRiskManager(Protocol):
    """Protocol for trading risk management."""

    @abstractmethod
    async def validate_trade(self, trade_decision: TradeDecision) -> bool:
        """Validate if trade meets risk criteria."""
        ...

    @abstractmethod
    async def calculate_position_size(
        self,
        symbol: Symbol,
        account_info: AccountInfo,
        risk_parameters: RiskParameters,
    ) -> float:
        """Calculate appropriate position size."""
        ...

    @abstractmethod
    async def assess_portfolio_risk(
        self,
        positions: list[PositionInfo],
    ) -> dict[str, float]:
        """Assess overall portfolio risk."""
        ...

    @abstractmethod
    async def get_stop_loss_price(
        self,
        symbol: Symbol,
        entry_price: float,
        position_side: str,
    ) -> float:
        """Calculate stop loss price."""
        ...

    @abstractmethod
    def get_current_risk_parameters(self) -> RiskParameters:
        """Get current risk parameters."""
        ...


@runtime_checkable
class TradingOrderExecutor(Protocol):
    """Protocol for trading order execution."""

    @abstractmethod
    async def execute_order(self, order: OrderRequest) -> OrderInfo:
        """Execute a trading order."""
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        ...

    @abstractmethod
    async def modify_order(self, order_id: str, updates: dict[str, float]) -> OrderInfo:
        """Modify an existing order."""
        ...

    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderInfo:
        """Get order status."""
        ...

    @abstractmethod
    async def get_open_orders(self, symbol: Symbol | None = None) -> list[OrderInfo]:
        """Get open orders."""
        ...


@runtime_checkable
class TradingPerformanceTracker(Protocol):
    """Protocol for performance tracking."""

    @abstractmethod
    async def record_trade(self, trade_decision: TradeDecision) -> None:
        """Record a completed trade."""
        ...

    @abstractmethod
    async def calculate_performance_metrics(self) -> PerformanceMetrics:
        """Calculate performance metrics."""
        ...

    @abstractmethod
    async def get_trade_history(
        self,
        limit: int | None = None,
    ) -> list[TradeDecision]:
        """Get trade history."""
        ...

    @abstractmethod
    async def generate_performance_report(self) -> str:
        """Generate performance report."""
        ...

    @abstractmethod
    def get_current_pnl(self) -> float:
        """Get current profit/loss."""
        ...


@runtime_checkable
class TradingAnalyst(Protocol):
    """Protocol for market analysis."""

    @abstractmethod
    async def analyze_market_conditions(self, symbol: Symbol) -> dict[str, float]:
        """Analyze current market conditions."""
        ...

    @abstractmethod
    async def generate_features(self, market_data: MarketDataDict) -> ModelInput:
        """Generate ML features from market data."""
        ...

    @abstractmethod
    async def detect_patterns(self, market_data: MarketDataDict) -> list[str]:
        """Detect chart patterns."""
        ...

    @abstractmethod
    async def calculate_support_resistance(self, symbol: Symbol) -> dict[str, float]:
        """Calculate support and resistance levels."""
        ...

    @abstractmethod
    def get_analysis_confidence(self) -> float:
        """Get analysis confidence level."""
        ...


@runtime_checkable
class TradingStrategist(Protocol):
    """Protocol for trading strategy."""

    @abstractmethod
    async def formulate_strategy(
        self,
        market_analysis: dict[str, float],
        regime_classification: RegimeClassification,
    ) -> TradingSignal:
        """Formulate trading strategy."""
        ...

    @abstractmethod
    async def update_strategy_parameters(self, performance: PerformanceMetrics) -> None:
        """Update strategy based on performance."""
        ...

    @abstractmethod
    async def optimize_strategy(self) -> None:
        """Optimize strategy parameters."""
        ...

    @abstractmethod
    def get_strategy_status(self) -> dict[str, str]:
        """Get current strategy status."""
        ...


@runtime_checkable
class TradingTactician(Protocol):
    """Protocol for trade execution tactics."""

    @abstractmethod
    async def execute_strategy(
        self,
        signal: TradingSignal,
        account_info: AccountInfo,
    ) -> TradeDecision | None:
        """Execute trading strategy."""
        ...

    @abstractmethod
    async def manage_positions(
        self,
        positions: list[PositionInfo],
    ) -> list[TradeDecision]:
        """Manage existing positions."""
        ...

    @abstractmethod
    async def optimize_execution(self, order: OrderRequest) -> OrderRequest:
        """Optimize order execution."""
        ...

    @abstractmethod
    def get_execution_status(self) -> dict[str, str]:
        """Get execution status."""
        ...


@runtime_checkable
class TradingSupervisor(Protocol):
    """Protocol for trading system supervision."""

    @abstractmethod
    async def monitor_system_health(self) -> dict[str, str]:
        """Monitor overall system health."""
        ...

    @abstractmethod
    async def coordinate_components(self) -> None:
        """Coordinate all trading components."""
        ...

    @abstractmethod
    async def handle_system_alerts(self, alerts: list[dict[str, str]]) -> None:
        """Handle system alerts."""
        ...

    @abstractmethod
    async def emergency_shutdown(self) -> None:
        """Emergency system shutdown."""
        ...

    @abstractmethod
    def get_system_status(self) -> dict[str, str]:
        """Get overall system status."""
        ...


# Composite protocols for complete trading systems
@runtime_checkable
class CompleteTradingSystem(
    TradingDataProvider,
    TradingMLPredictor,
    TradingRiskManager,
    TradingOrderExecutor,
    TradingPerformanceTracker,
    Protocol,
):
    """Protocol for complete trading systems."""


@runtime_checkable
class TradingComponentManager(Protocol):
    """Protocol for managing trading components."""

    @abstractmethod
    async def start_component(self, component_name: str) -> bool:
        """Start a trading component."""
        ...

    @abstractmethod
    async def stop_component(self, component_name: str) -> bool:
        """Stop a trading component."""
        ...

    @abstractmethod
    async def restart_component(self, component_name: str) -> bool:
        """Restart a trading component."""
        ...

    @abstractmethod
    def get_component_status(self, component_name: str) -> dict[str, str]:
        """Get component status."""
        ...

    @abstractmethod
    def get_all_components_status(self) -> dict[str, dict[str, str]]:
        """Get status of all components."""
        ...
