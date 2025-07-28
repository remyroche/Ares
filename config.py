# config.py

# --- Configuration and Thresholds (Tune these based on your strategy and market conditions) ---
# These values are illustrative and should be adjusted based on backtesting and live market observation.
CONFIG = {
    "bollinger_bands": {
        "window": 20,
        "num_std_dev": 2
    },
    "atr": {
        "window": 14,
        "stop_loss_multiplier": 1.5, # e.g., 1.5x ATR for stop-loss
        "max_risk_per_trade_pct": 0.01 # Max 1% of capital risked per trade
    },
    "order_book": {
        "large_order_threshold_usd": 100000, # USD value to consider an order 'large'
        "spread_narrow_threshold_pct": 0.0005, # e.g., 0.05% spread is narrow
        "spread_wide_threshold_pct": 0.0015,  # e.g., 0.15% spread is wide
        "spoofing_pulling_volume_change_threshold_pct": 0.7 # 70% change in wall size
    },
    "volume": {
        "breakout_follow_through_volume_multiplier": 1.2, # Volume post-breakout should be 20% higher than pre-breakout
        "capitulation_volume_spike_multiplier": 2.0 # Volume spike should be 2x average
    },
    "funding_rate": {
        "high_positive_threshold": 0.0005 # e.g., 0.05% positive funding rate
    },
    "sr_proximity_pct": 0.005, # Price is considered "close" to S/R if within 0.5%
    "confidence_wrong_direction_thresholds": [0.001, 0.005, 0.01, 0.015, 0.02], # 0.1%, 0.5%, 1%, 1.5%, 2%
    # S/R Analyzer specific configurations
    "sr_analyzer": {
        "peak_prominence": 0.005,  # Minimum prominence for peak detection (as % of price)
        "peak_width": 5,           # Minimum width for peaks (in data points)
        "level_tolerance_pct": 0.002, # Tolerance for grouping nearby levels (0.2% of price)
        "min_touches": 2,          # Minimum number of touches for a level to be considered significant
        "volume_lookback_window": 10, # Number of periods to consider for volume at touch
        "max_age_days": 90         # Maximum age for a level to be considered relevant (in days)
    },
    # --- Analyst Specific Configurations ---
    "analyst": {
        "feature_engineering": {
            "wavelet_level": 3, # Decomposition level for wavelet transforms
            "autoencoder_latent_dim": 16, # Latent dimension for autoencoder
            "gbm_feature_threshold": 0.01 # Feature importance threshold for GBM selection
        },
        "market_regime_classifier": {
            "kmeans_n_clusters": 4, # Number of clusters for Wasserstein k-Means (excluding SR_ZONE_ACTION)
            "adx_period": 14,
            "macd_fast_period": 12,
            "macd_slow_period": 26,
            "macd_signal_period": 9,
            "trend_scaling_factor": 100,
            "trend_threshold": 20, # ADX threshold for trend
            "max_strength_threshold": 60 # ADX max strength for scaling
        },
        "regime_predictive_ensembles": {
            # Weights for combining model confidences within ensembles
            "ensemble_weights": {
                "lstm": 0.3,
                "transformer": 0.3,
                "statistical": 0.2,
                "volume": 0.2
            },
            "min_confluence_confidence": 0.7 # Minimum average confidence for a trade signal
        },
        "liquidation_risk_model": {
            "volatility_impact": 0.4,
            "order_book_depth_impact": 0.3,
            "position_impact": 0.3,
            "lookback_periods_volatility": 240, # For historical volatility
            "order_book_depth_threshold": 0.01 # Percentage depth considered significant
        },
        "market_health_analyzer": {
            "atr_weight": 0.3,
            "bollinger_weight": 0.3,
            "ma_cluster_weight": 0.2,
            "momentum_weight": 0.1,
            "obv_weight": 0.1,
            "ma_periods": [20, 50, 100], # Moving average periods for clustering
            "momentum_period": 14 # Period for momentum oscillator (e.g., RSI)
        },
        "high_impact_candle_model": {
            "volume_multiplier": 3.0, # Multiplier for current volume vs. average volume
            "atr_multiplier": 2.0,    # Multiplier for current ATR vs. average ATR
            "volume_sma_period": 20,
            "atr_sma_period": 20
        }
    },
    # --- Tactician Specific Configurations ---
    "tactician": {
        "rl_agent": {
            "model_path": "models/tactician_ppo_model.zip", # Path to save/load RL agent
            "training_steps": 100000, # Number of steps for RL training
            "reward_weights": { # Weights for different components of the reward function
                "pnl": 0.6,
                "drawdown_penalty": -0.2,
                "liquidation_penalty": -1.0,
                "confidence_bonus": 0.1
            },
            "l1_regularization_strength": 0.001, # L1 regularization strength for RL agent's network
            "l2_regularization_strength": 0.001  # L2 regularization strength for RL agent's network
        },
        "laddering": {
            "initial_leverage": 25, # Starting leverage for the first order
            "max_leverage_cap": 100, # Absolute maximum leverage (set by Strategist, but Tactician uses this cap)
            "min_lss_for_ladder": 70, # Minimum LSS to consider adding to a ladder
            "min_confidence_for_ladder": 0.75, # Minimum directional confidence to add to a ladder
            "ladder_step_leverage_increase": 5, # How much leverage increases per ladder step
            "max_ladder_steps": 3 # Maximum number of additional ladder orders
        },
        "risk_management": {
            "risk_per_trade_pct": 0.01 # Max 1% of capital risked per trade (from overall config)
        },
        "order_types": ["MARKET", "LIMIT"] # Supported order types for the Tactician
    }
}

# Emails config
from emails_config import EMAIL_CONFIG, COMMAND_EMAIL_CONFIG

# --- General Configuration ---
SYMBOL = 'ETHUSDT'
INTERVAL = '1m'
LOOKBACK_YEARS = 2

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
