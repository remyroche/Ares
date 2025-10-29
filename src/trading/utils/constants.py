"""Trading constants for thresholds and configuration."""

# Data quality thresholds
DATA_MISSING_THRESHOLD_CRITICAL = 0.10  # 10% missing values is critical
DATA_MISSING_THRESHOLD_WARNING = 0.05  # 5% missing values triggers warning
EXTREME_PRICE_CHANGE_THRESHOLD = 0.50  # 50% price change is extreme
ZERO_VOLUME_THRESHOLD = 0.10  # 10% zero volume is warning

# Position size thresholds
MIN_POSITION_SIZE = 0.01  # 1% minimum position size
MAX_POSITION_SIZE = 0.25  # 25% maximum position size (default)
LARGE_POSITION_WARNING = 0.50  # 50% position size triggers warning
MIN_TRADE_SIZE_DOLLARS = 10.0  # Minimum $10 trade size

# Confidence thresholds
MIN_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for signals
LOW_CONFIDENCE_WARNING = 0.5  # Below this triggers warning
LOW_REGIME_CONFIDENCE = 0.3  # Low regime confidence threshold

# Technical indicator windows
DEFAULT_ATR_PERIOD = 14
DEFAULT_VOLATILITY_WINDOW = 20
DEFAULT_MOMENTUM_WINDOW = 3
DEFAULT_RSI_PERIOD = 14
DEFAULT_SMA_WINDOW = 20
DEFAULT_EMA_SPAN = 12

# Risk parameters
MAX_PORTFOLIO_RISK = 1.0
MAX_DRAWDOWN_LIMIT = 1.0
MAX_LEVERAGE_MIN = 1.0
MAX_LEVERAGE_MAX = 100.0

# Probability validation
PROBABILITY_SUM_TOLERANCE = 0.1  # 10% tolerance for probability sums

# Retry configuration
DEFAULT_RETRY_MAX_ATTEMPTS = 3
DEFAULT_RETRY_BASE_DELAY = 1.0  # seconds
DEFAULT_RETRY_MAX_DELAY = 60.0  # seconds
DEFAULT_RETRY_EXPONENT = 2.0

# Circuit breaker configuration
DEFAULT_CB_FAILURE_THRESHOLD = 5
DEFAULT_CB_RECOVERY_TIMEOUT = 60  # seconds
DEFAULT_CB_HALF_OPEN_MAX_CALLS = 3

# Rate limiting defaults
DEFAULT_RATE_LIMIT_REQUESTS = 100
DEFAULT_RATE_LIMIT_WINDOW = 60  # seconds

# Timestamp validation
TIMESTAMP_DUPLICATE_THRESHOLD = 0
TIMESTAMP_UNSORTED_THRESHOLD = 0

# Market data validation
MIN_MARKET_DATA_ROWS = 10
OHLC_CONSISTENCY_CHECKS = True

# Exchange validation
VALID_EXCHANGES = ['binance', 'binance_testnet', 'simulated']
VALID_TRADING_MODES = ['paper', 'live', 'backtest', 'simulation']
VALID_ORDER_ACTIONS = ['buy', 'sell', 'hold', 'close']

# Precision defaults
DEFAULT_PRICE_PRECISION = 8
DEFAULT_QUANTITY_PRECISION = 8
