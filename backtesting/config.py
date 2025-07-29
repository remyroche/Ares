# Emails config
from emails_config import EMAIL_CONFIG, COMMAND_EMAIL_CONFIG

# --- General Configuration ---
SYMBOL = 'ETHUSDT'
INTERVAL = '1m'
LOOKBACK_YEARS = 2

# Default configuration for backtesting
BACKTESTING_CONFIG = {
    "strategy": "YourStrategyName",
    "leverage": 5,
    "stop_loss_pct": 2.0,
    "take_profit_pct": 5.0,
    "fee_rate": 0.0005,  # Represents a 0.05% trading fee, a conservative value for many exchanges.
    "data_path": "data/BTC_USDT-1h.csv",
    "start_date": "2022-01-01",
    "end_date": "2023-01-01",
}


# --- Data Caching Configuration ---
KLINES_FILENAME = f"{SYMBOL}_{INTERVAL}_{LOOKBACK_YEARS}y_klines.csv"
AGG_TRADES_FILENAME = f"{SYMBOL}_{LOOKBACK_YEARS}y_aggtrades.csv"
FUTURES_FILENAME = f"{SYMBOL}_futures_{LOOKBACK_YEARS}y_data.csv"
PREPARED_DATA_FILENAME = f"{SYMBOL}_{INTERVAL}_{LOOKBACK_YEARS}y_prepared_data.csv"

# --- Script Names ---
DOWNLOADER_SCRIPT_NAME = "ares_data_downloader.py"
PREPARER_SCRIPT_NAME = "ares_data_preparer.py"
PIPELINE_SCRIPT_NAME = "ares_pipeline.py" # Added for clarity in listener
PIPELINE_PID_FILE = "ares_pipeline.pid" # File to store the pipeline's PID
RESTART_FLAG_FILE = "restart_pipeline.flag" # Flag file to signal pipeline restart

# --- Portfolio & Risk Configuration ---
INITIAL_EQUITY = 10000
RISK_PER_TRADE_PCT = 0.01

# --- Optimal Indicator Parameters ---
# This dictionary now contains the WEIGHTS for the confidence score, which will be found by the optimizer.
BEST_PARAMS = {
    # --- Confidence Score Weights ---
    'weight_trend': 0.4,       # Importance of the trend-following component
    'weight_reversion': 0.3,   # Importance of the mean-reversion component at S/R levels
    'weight_sentiment': 0.3,   # Importance of the futures market sentiment
    
    # --- Trade Execution Parameters ---
    'trade_entry_threshold': 0.6, # Confidence score needed to enter a trade (e.g., 0.6 means 60% confidence)
    'sl_atr_multiplier': 1.5,     # Stop loss distance in multiples of ATR
    'take_profit_rr': 2.0,        # Risk/Reward ratio for setting the take profit
    
    # --- Underlying Indicator Settings (these are now considered more stable) ---
    'adx_period': 20, 
    'trend_threshold': 25,
    'max_strength_threshold': 60, 
    'atr_period': 14, 
    'proximity_multiplier': 0.25,
    'sma_period': 50, 
    'volume_multiplier': 3, 
    'volatility_multiplier': 2,
    'zscore_threshold': 1.5,
    'obv_lookback': 20,
    'bband_length': 20,
    'bband_std': 2.0,
    'bband_squeeze_threshold': 0.01,
    'scaling_factor': 100, 
    'trend_strength_threshold': 25
}
