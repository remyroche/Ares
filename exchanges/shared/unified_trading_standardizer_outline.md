# Unified Trading Standardizer - Structure Outline

## Overview
A comprehensive standardizer for trading-related API responses (orders, positions, balances, account info, trades) that ensures complete equivalency between all exchanges, similar to `UnifiedOHLCVStandardizer` for data collection.

## Purpose
- Standardize trading API responses from different exchanges to a unified format
- Ensure consistency for order execution, position management, and account queries
- Provide exchange-agnostic interface for trading operations
- Integrate with existing trading utilities and validation frameworks

---

## 1. Core Data Structures

### 1.1 StandardizedOrder
```python
@dataclass
class StandardizedOrder:
    """Unified order structure across all exchanges"""
    # Required fields
    order_id: str                    # Exchange-specific order ID
    client_order_id: Optional[str]   # Client-provided order ID
    symbol: str
    exchange: str
    source: ExchangeType
    side: OrderSide                  # BUY/SELL
    order_type: OrderType            # MARKET/LIMIT/STOP/etc
    status: OrderStatus              # NEW/PARTIALLY_FILLED/FILLED/CANCELED/REJECTED
    quantity: float
    price: Optional[float]           # None for market orders
    executed_quantity: float          # Filled quantity
    remaining_quantity: float
    executed_price_avg: Optional[float]  # Average fill price
    timestamp: datetime              # Order creation time
    update_time: datetime            # Last update time
    
    # Optional fields
    stop_price: Optional[float]      # For stop orders
    time_in_force: Optional[str]     # GTC/IOC/FOK
    fee: Optional[float]
    fee_currency: Optional[str]
    
    # Exchange metadata
    raw_order_data: Optional[Dict[str, Any]]  # Original response
    exchange_order_id: str          # Exchange-specific identifier
    
    # Validation
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
```

### 1.2 StandardizedPosition
```python
@dataclass
class StandardizedPosition:
    """Unified position structure across all exchanges"""
    # Required fields
    symbol: str
    exchange: str
    source: ExchangeType
    side: PositionSide              # LONG/SHORT
    size: float                     # Position size
    entry_price: float             # Average entry price
    mark_price: Optional[float]    # Current mark price
    liquidation_price: Optional[float]
    unrealized_pnl: float
    realized_pnl: float
    leverage: Optional[float]      # 1x for spot, >1x for futures
    margin: Optional[float]         # Used margin
    isolated_margin: Optional[float]
    timestamp: datetime
    update_time: datetime
    
    # Optional fields
    position_value: Optional[float]  # Position notional value
    margin_mode: Optional[str]       # ISOLATED/CROSSED
    position_mode: Optional[str]     # HEDGE/ONE_WAY
    
    # Exchange metadata
    exchange_position_id: Optional[str]
    raw_position_data: Optional[Dict[str, Any]]
    
    # Validation
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
```

### 1.3 StandardizedBalance
```python
@dataclass
class StandardizedBalance:
    """Unified balance structure across all exchanges"""
    # Required fields
    currency: str
    exchange: str
    source: ExchangeType
    free: float                     # Available balance
    used: float                     # Locked/in-use balance
    total: float                    # Total balance (free + used)
    timestamp: datetime
    
    # Optional fields
    available_balance: Optional[float]  # Exchange-specific available
    frozen_balance: Optional[float]     # Frozen/locked balance
    account_type: Optional[str]          # SPOT/MARGIN/FUTURES
    
    # Exchange metadata
    raw_balance_data: Optional[Dict[str, Any]]
    
    # Validation
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
```

### 1.4 StandardizedAccountInfo
```python
@dataclass
class StandardizedAccountInfo:
    """Unified account information structure"""
    # Required fields
    exchange: str
    source: ExchangeType
    account_type: str               # SPOT/MARGIN/FUTURES
    can_trade: bool
    can_withdraw: bool
    can_deposit: bool
    timestamp: datetime
    
    # Optional fields
    permissions: List[str]           # Trading permissions
    balances: List[StandardizedBalance]  # Account balances
    total_equity: Optional[float]
    available_margin: Optional[float]
    used_margin: Optional[float]
    margin_ratio: Optional[float]
    
    # Exchange metadata
    raw_account_data: Optional[Dict[str, Any]]
    
    # Validation
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
```

### 1.5 StandardizedTrade
```python
@dataclass
class StandardizedTrade:
    """Unified trade/transaction structure"""
    # Required fields
    trade_id: str                   # Exchange-specific trade ID
    order_id: str                   # Parent order ID
    symbol: str
    exchange: str
    source: ExchangeType
    side: OrderSide                 # BUY/SELL
    price: float
    quantity: float
    timestamp: datetime
    fee: float
    fee_currency: str
    
    # Optional fields
    is_maker: Optional[bool]         # Maker/taker flag
    is_buyer: Optional[bool]         # Buyer/seller flag
    trade_type: Optional[str]        # TRADE/FUNDING_FEE/etc
    
    # Exchange metadata
    raw_trade_data: Optional[Dict[str, Any]]
    
    # Validation
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
```

---

## 2. Main Standardizer Class

### 2.1 UnifiedTradingStandardizer
```python
class UnifiedTradingStandardizer:
    """
    Unified trading data standardizer that ensures complete equivalency 
    across all exchanges for trading operations.
    """
    
    def __init__(self, quality_level: DataQualityLevel = DataQualityLevel.STANDARD):
        """
        Initialize the unified trading standardizer.
        
        Args:
            quality_level: Data quality validation level
        """
        self.quality_level = quality_level
        self.logger = system_logger.getChild("UnifiedTradingStandardizer")
        
        # Exchange-specific field mappings for trading responses
        self.exchange_order_mappings = {...}      # Order field mappings per exchange
        self.exchange_position_mappings = {...}  # Position field mappings
        self.exchange_balance_mappings = {...}   # Balance field mappings
        self.exchange_account_mappings = {...}    # Account info mappings
        self.exchange_trade_mappings = {...}      # Trade field mappings
        
        # Status/enum mappings across exchanges
        self.order_status_mappings = {...}        # Map exchange status to unified status
        self.order_type_mappings = {...}          # Map exchange order types
        self.position_side_mappings = {...}       # Map position sides
```

---

## 3. Core Standardization Methods

### 3.1 Order Standardization
```python
def standardize_order(
    self,
    raw_order: Dict[str, Any],
    exchange: ExchangeType,
    symbol: str
) -> StandardizedOrder:
    """
    Standardize order response from exchange to unified format.
    
    Handles:
    - Field name mapping (orderId -> order_id, etc.)
    - Status conversion (exchange-specific -> unified enum)
    - Type conversion (exchange-specific -> unified enum)
    - Quantity normalization (coin vs contract units)
    - Price precision handling
    - Timestamp conversion (ms/s/us -> datetime)
    """

def standardize_orders(
    self,
    raw_orders: Union[List[Dict], List[List]],
    exchange: ExchangeType,
    symbol: Optional[str] = None
) -> List[StandardizedOrder]:
    """Standardize multiple orders."""

def standardize_orders_to_dataframe(
    self,
    raw_orders: Union[List[Dict], List[List]],
    exchange: ExchangeType,
    symbol: Optional[str] = None
) -> pd.DataFrame:
    """Standardize orders to DataFrame format."""
```

### 3.2 Position Standardization
```python
def standardize_position(
    self,
    raw_position: Dict[str, Any],
    exchange: ExchangeType,
    symbol: str
) -> StandardizedPosition:
    """
    Standardize position response from exchange.
    
    Handles:
    - Position side mapping (long/short, bid/ask, etc.)
    - Quantity normalization
    - PnL calculation standardization
    - Leverage extraction
    - Margin mode detection
    """

def standardize_positions(
    self,
    raw_positions: Union[List[Dict], Dict[str, Dict]],
    exchange: ExchangeType,
    symbol: Optional[str] = None
) -> List[StandardizedPosition]:
    """Standardize multiple positions."""
```

### 3.3 Balance Standardization
```python
def standardize_balance(
    self,
    raw_balance: Dict[str, Any],
    exchange: ExchangeType,
    currency: str
) -> StandardizedBalance:
    """
    Standardize balance response from exchange.
    
    Handles:
    - Balance field mapping (free/available/locked/used)
    - Account type detection (spot/margin/futures)
    - Currency normalization
    """

def standardize_balances(
    self,
    raw_balances: Union[List[Dict], Dict[str, Dict]],
    exchange: ExchangeType
) -> List[StandardizedBalance]:
    """Standardize all balances from account."""
```

### 3.4 Account Info Standardization
```python
def standardize_account_info(
    self,
    raw_account: Dict[str, Any],
    exchange: ExchangeType
) -> StandardizedAccountInfo:
    """
    Standardize account information response.
    
    Handles:
    - Account type detection
    - Permission extraction
    - Balance aggregation
    - Margin information normalization
    """
```

### 3.5 Trade Standardization
```python
def standardize_trade(
    self,
    raw_trade: Dict[str, Any],
    exchange: ExchangeType,
    symbol: str,
    order_id: Optional[str] = None
) -> StandardizedTrade:
    """
    Standardize trade/execution response.
    
    Handles:
    - Trade ID extraction
    - Fee calculation/formatting
    - Maker/taker detection
    - Buyer/seller flag mapping
    """

def standardize_trades(
    self,
    raw_trades: Union[List[Dict], pd.DataFrame],
    exchange: ExchangeType,
    symbol: str
) -> List[StandardizedTrade]:
    """Standardize multiple trades."""
```

---

## 4. Exchange-Specific Mappings

### 4.1 Order Field Mappings
```python
exchange_order_mappings = {
    ExchangeType.BINANCE: {
        'order_id': ['orderId', 'order_id'],
        'client_order_id': ['clientOrderId', 'client_order_id'],
        'symbol': ['symbol'],
        'side': ['side'],  # BUY/SELL
        'type': ['type'],  # MARKET/LIMIT/etc
        'status': ['status'],  # NEW/PARTIALLY_FILLED/FILLED/CANCELED
        'quantity': ['origQty', 'orig_quantity'],
        'executed_quantity': ['executedQty', 'executed_quantity'],
        'price': ['price'],
        'timestamp': ['time', 'timestamp'],
        ...
    },
    ExchangeType.OKX: {
        'order_id': ['ordId', 'order_id'],
        'client_order_id': ['clOrdId', 'client_order_id'],
        'symbol': ['instId', 'symbol'],
        'side': ['side'],  # buy/sell
        'type': ['ordType'],  # market/limit
        'status': ['state'],  # live/filled/canceled
        ...
    },
    # Similar mappings for BINGX, MEXC, GATEIO, PHEMEX
}
```

### 4.2 Status Mappings
```python
order_status_mappings = {
    ExchangeType.BINANCE: {
        'NEW': OrderStatus.PENDING,
        'PARTIALLY_FILLED': OrderStatus.PARTIALLY_FILLED,
        'FILLED': OrderStatus.FILLED,
        'CANCELED': OrderStatus.CANCELED,
        'REJECTED': OrderStatus.REJECTED,
        'EXPIRED': OrderStatus.CANCELED,
    },
    ExchangeType.OKX: {
        'live': OrderStatus.PENDING,
        'partially_filled': OrderStatus.PARTIALLY_FILLED,
        'filled': OrderStatus.FILLED,
        'canceled': OrderStatus.CANCELED,
        ...
    },
    # Similar for other exchanges
}
```

---

## 5. Helper Methods

### 5.1 Field Extraction Helpers
```python
def _extract_order_fields(
    self,
    raw_order: Dict[str, Any],
    exchange: ExchangeType
) -> Dict[str, Any]:
    """Extract order fields using exchange-specific mappings."""

def _normalize_order_status(
    self,
    raw_status: str,
    exchange: ExchangeType
) -> OrderStatus:
    """Convert exchange-specific status to unified OrderStatus enum."""

def _normalize_order_type(
    self,
    raw_type: str,
    exchange: ExchangeType
) -> OrderType:
    """Convert exchange-specific order type to unified OrderType enum."""

def _normalize_quantity(
    self,
    quantity: Union[str, float],
    symbol: str,
    exchange: ExchangeType
) -> float:
    """Normalize quantity (handle different units, precision, etc.)."""

def _normalize_price(
    self,
    price: Union[str, float],
    symbol: str,
    exchange: ExchangeType
) -> float:
    """Normalize price with proper precision handling."""

def _convert_timestamp(
    self,
    timestamp: Union[int, str, float],
    exchange: ExchangeType,
    unit: Optional[str] = None
) -> datetime:
    """Convert timestamp to datetime (handles different units per exchange)."""
```

### 5.2 Validation Helpers
```python
def _validate_order(
    self,
    order: StandardizedOrder
) -> Tuple[bool, List[str]]:
    """Validate standardized order data."""

def _validate_position(
    self,
    position: StandardizedPosition
) -> Tuple[bool, List[str]]:
    """Validate standardized position data."""

def _calculate_position_pnl(
    self,
    position: StandardizedPosition,
    current_price: float
) -> float:
    """Calculate standardized PnL for position."""

def _validate_balance(
    self,
    balance: StandardizedBalance
) -> Tuple[bool, List[str]]:
    """Validate standardized balance data."""
```

---

## 6. Integration Points

### 6.1 Integration with Exchange Dispatcher
```python
def standardize_dispatcher_response(
    self,
    response_type: str,  # 'order', 'position', 'balance', etc.
    raw_response: Any,
    exchange: ExchangeType,
    **kwargs
) -> Union[StandardizedOrder, StandardizedPosition, StandardizedBalance, ...]:
    """
    Convenience method that works with ExchangeDispatcher responses.
    """
```

### 6.2 Integration with Order Manager
```python
def standardize_order_manager_response(
    self,
    raw_order: Dict[str, Any],
    exchange: ExchangeType
) -> StandardizedOrder:
    """
    Standardize order from HighLevelOrderManager.
    """
```

### 6.3 Integration with Balance Manager
```python
def standardize_balance_manager_response(
    self,
    raw_balance: Dict[str, Any],
    exchange: ExchangeType,
    currency: str
) -> StandardizedBalance:
    """
    Standardize balance from HighLevelBalanceManager.
    """
```

---

## 7. Quality and Validation

### 7.1 Quality Processing
```python
def _apply_quality_processing(
    self,
    standardized_data: List[Union[StandardizedOrder, StandardizedPosition, ...]]
) -> List[Union[StandardizedOrder, StandardizedPosition, ...]]:
    """Apply quality validation and filtering."""

def _calculate_quality_score(
    self,
    item: Union[StandardizedOrder, StandardizedPosition, ...]
) -> float:
    """Calculate quality score for standardized data point."""
```

### 7.2 Consistency Validation
```python
def validate_trading_data_consistency(
    self,
    orders: List[StandardizedOrder],
    positions: List[StandardizedPosition],
    balances: List[StandardizedBalance]
) -> Dict[str, Any]:
    """
    Validate consistency across orders, positions, and balances.
    
    Checks:
    - Order positions match actual positions
    - Balance changes match executed orders
    - Position sizes match order fills
    """
```

---

## 8. Data Conversion Utilities

### 8.1 DataFrame Conversion
```python
def orders_to_dataframe(
    self,
    orders: List[StandardizedOrder]
) -> pd.DataFrame:
    """Convert standardized orders to DataFrame."""

def positions_to_dataframe(
    self,
    positions: List[StandardizedPosition]
) -> pd.DataFrame:
    """Convert standardized positions to DataFrame."""

def balances_to_dataframe(
    self,
    balances: List[StandardizedBalance]
) -> pd.DataFrame:
    """Convert standardized balances to DataFrame."""
```

### 8.2 Serialization
```python
def to_dict(self) -> Dict[str, Any]:
    """Convert standardized object to dictionary."""
    
def to_json(self) -> str:
    """Convert standardized object to JSON."""
```

---

## 9. Error Handling and Edge Cases

### 9.1 Missing Fields Handling
- Default values for optional fields
- Fallback field name extraction
- Graceful degradation when fields are missing

### 9.2 Type Conversion
- String to number conversion
- Enum conversion with fallbacks
- Date/time parsing with multiple formats

### 9.3 Precision Handling
- Price precision per symbol
- Quantity precision per symbol
- Exchange-specific rounding rules

---

## 10. Configuration and Extensibility

### 10.1 Custom Mappings
```python
def add_custom_order_mapping(
    self,
    exchange: ExchangeType,
    field_mapping: Dict[str, List[str]]
):
    """Allow custom field mappings for new exchanges or API changes."""

def update_status_mapping(
    self,
    exchange: ExchangeType,
    status_mapping: Dict[str, OrderStatus]
):
    """Update status mappings when exchange API changes."""
```

### 10.2 Schema Versioning
- Support for multiple API versions
- Backward compatibility handling
- Migration utilities for deprecated fields

---

## 11. Testing and Validation

### 11.1 Unit Tests
- Test field mapping for each exchange
- Test status/type conversion
- Test validation logic
- Test edge cases (missing fields, null values, etc.)

### 11.2 Integration Tests
- Test with real exchange API responses
- Test with ExchangeDispatcher
- Test with OrderManager and BalanceManager
- Validate equivalence across exchanges

### 11.3 Validation Against Analyzer
- Use `exchange_api_format_analyzer.py` output as test data
- Validate mappings match analyzed formats
- Generate tests from analyzer findings

---

## 12. Usage Examples

### 12.1 Basic Usage
```python
standardizer = UnifiedTradingStandardizer()

# Standardize order
order = standardizer.standardize_order(
    raw_order=binance_order_response,
    exchange=ExchangeType.BINANCE,
    symbol="BTCUSDT"
)

# Standardize positions
positions = standardizer.standardize_positions(
    raw_positions=okx_positions_response,
    exchange=ExchangeType.OKX
)

# Standardize balances
balances = standardizer.standardize_balances(
    raw_balances=bingx_balance_response,
    exchange=ExchangeType.BINGX
)
```

### 12.2 Integration with Dispatcher
```python
dispatcher = ExchangeDispatcher(config)
orders = await dispatcher.get_open_orders("BTCUSDT")

# Standardize all orders
standardizer = UnifiedTradingStandardizer()
standardized_orders = [
    standardizer.standardize_order(order, ExchangeType.BINANCE, "BTCUSDT")
    for order in orders
]
```

---

## 13. File Structure

```
exchanges/shared/
├── unified_trading_standardizer.py      # Main standardizer class
├── trading_data_structures.py           # Dataclasses (StandardizedOrder, etc.)
├── trading_field_mappings.py            # Exchange-specific mappings
├── trading_status_mappings.py           # Status/type conversion mappings
└── tests/
    ├── test_unified_trading_standardizer.py
    ├── test_order_standardization.py
    ├── test_position_standardization.py
    └── test_balance_standardization.py
```

---

## 14. Key Differences from UnifiedOHLCVStandardizer

1. **Data Type**: Focuses on trading operations (orders, positions, balances) rather than market data (OHLCV)
2. **Real-time Updates**: Handles frequently updating data (order status, position PnL)
3. **State Management**: Maintains relationship between orders → trades → positions
4. **Validation Complexity**: More complex validation (e.g., order-position consistency)
5. **Enum Mappings**: Extensive enum mappings (OrderStatus, OrderType, PositionSide)
6. **Precision Requirements**: Stricter precision requirements for trading operations
7. **Error Handling**: More critical error handling (incorrect order data = financial risk)

---

## 15. Future Enhancements

1. **Caching**: Cache standardized responses for performance
2. **WebSocket Support**: Standardize real-time WebSocket updates
3. **Batch Operations**: Optimize for batch standardization
4. **Metrics**: Track standardization performance and accuracy
5. **Auto-detection**: Auto-detect exchange format changes using analyzer
6. **Versioning**: Support multiple API versions simultaneously
