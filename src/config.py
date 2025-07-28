# config.py

from emails_config import EMAIL_CONFIG, COMMAND_EMAIL_CONFIG
from datetime import datetime, timedelta

# --- General Configuration ---
# All configurations are now consolidated into a single CONFIG dictionary.
CONFIG = {
    # --- General System Parameters ---
    "SYMBOL": 'ETHUSDT',
    "INTERVAL": '1m',
    "LOOKBACK_YEARS": 2,

    # --- Script Names & Flags ---
    "DOWNLOADER_SCRIPT_NAME": "ares_data_downloader.py",
    "PREPARER_SCRIPT_NAME": "ares_data_preparer.py",
    "PIPELINE_SCRIPT_NAME": "src/ares_pipeline.py",
    "BACKTESTING_PIPELINE_SCRIPT_NAME": "src/backtesting_pipeline.py",
    "PIPELINE_PID_FILE": "ares_pipeline.pid",
    "RESTART_FLAG_FILE": "restart_pipeline.flag",
    "PROMOTE_CHALLENGER_FLAG_FILE": "promote_challenger.flag", # <-- ADD THIS LINE
        
    # --- Data Caching Configuration ---
    # Filenames are now generated dynamically based on SYMBOL, INTERVAL, LOOKBACK_YEARS
    "KLINES_FILENAME": '', # Will be set dynamically below
    "AGG_TRADES_FILENAME": '', # Will be set dynamically below
    "FUTURES_FILENAME": '', # Will be set dynamically below
    "PREPARED_DATA_FILENAME": '', # Will be set dynamically below

    # --- Portfolio & Risk Configuration ---
    "INITIAL_EQUITY": 10000,
    "RISK_PER_TRADE_PCT": 0.01, # Max 1% of capital risked per trade (used by Tactician)

    # --- Optimal Indicator Parameters (BEST_PARAMS) ---
    # This dictionary now contains the WEIGHTS for the confidence score, which will be found by the optimizer.
    # It is part of the main CONFIG dictionary.
    "BEST_PARAMS": {
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
    },

    # --- General Trading Parameters (moved from root level to a 'general_trading' sub-dict for clarity) ---
    "general_trading": {
        "sr_proximity_pct": 0.005, # Price is considered "close" to S/R if within 0.5%
        "confidence_wrong_direction_thresholds": [0.001, 0.005, 0.01, 0.015, 0.02], # 0.1%, 0.5%, 1%, 1.5%, 2%
    },

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
                "statistical": 0.2, # This key might need to be 'garch' if it's GARCH model
                "volume": 0.2
            },
            "min_confluence_confidence": 0.7, # Minimum average confidence for a trade signal
            "meta_learner_l1_reg": 0.1, # L1 regularization for meta-learner
            "meta_learner_l2_reg": 0.1  # L2 regularization for meta-learner
        },
        "liquidation_risk_model": {
            "volatility_impact": 0.4,
            "order_book_depth_impact": 0.3,
            "position_impact": 0.3,
            "lookback_periods_volatility": 240, # For historical volatility
            "atr_period": 14, # ATR period for LSS volatility calculation
            "atr_to_std_factor": 2.5, # Factor to convert ATR to a proxy for standard deviation
            "ob_depth_range_pct": 0.005, # Percentage range around current price to consider for order book depth
            "liq_buffer_zone_pct": 0.001, # Percentage zone around liquidation price for buffer volume
            "liq_buffer_weight": 2.0, # Weight for liquidation buffer volume in LSS calculation
            "ob_depth_scaling_factor": 10.0, # Scaling factor for order book depth score (tanh function)
            "max_safe_distance_pct": 0.05 # Max percentage distance to liquidation considered "safe" (for health score)
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
        },
        "model_storage_path": "models/analyst/" # Local path for analyst models
    },
    # --- Tactician Specific Configurations ---
    "tactician": {
        "laddering": {
            "initial_leverage": 25, # Starting leverage for the first order
            "max_leverage_cap": 100, # Absolute maximum leverage (set by Strategist, but Tactician uses this cap)
            "min_lss_for_ladder": 70, # Minimum LSS to consider adding to a ladder
            "min_confidence_for_ladder": 0.75, # Minimum directional confidence to add to a ladder
            "ladder_step_leverage_increase": 5, # How much leverage increases per ladder step
            "max_ladder_steps": 3 # Maximum number of additional ladder orders
        },
        "risk_management": {
            "risk_per_trade_pct": 0.01, # Max 1% of capital risked per trade (from overall config)
            "min_lss_for_entry": 60 # Minimum LSS required to open an initial position
        },
        "order_types": ["MARKET", "LIMIT"] # Supported order types for the Tactician
    },
    # --- Strategist Specific Configurations ---
    "strategist": {
        "timeframe": "1D", # Timeframe for macro analysis (e.g., "4H", "1D")
        "ma_periods_for_bias": [50, 200], # Moving average periods for positional bias
        "trading_range_atr_multiplier": 3.0, # Multiplier for ATR to define range from recent price
        "max_leverage_cap_default": 100, # Default max leverage cap
        "sr_relevance_threshold": 5.0, # Minimum strength score for S/R levels to be considered by Strategist
        "avwap_anchor_period": 60 # Lookback period for Anchored VWAP anchor points
    },
    # --- Supervisor Specific Configurations ---
    "supervisor": {
        "meta_learning_frequency_days": 30, # How often to run meta-learning optimization
        "retraining_schedule": {
            "enabled": True,
            "first_retraining_date": (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'), # 30 days after the bot launch
            "retraining_period_days": 30 # Run every 30 days after the first date
        },
        "risk_allocation_lookback_days": 30, # Days to look back for performance for dynamic capital allocation
        "max_capital_allocation_increase_pct": 1.0, # Max +100% of initial allocated capital (i.e., up to 2x)
        "max_capital_allocation_decrease_pct": 0.75, # Max -75% of initial allocated capital (i.e., down to 0.25x)
        "initial_allocated_capital_multiplier": 1.0, # Starting multiplier (1.0 = 100% of INITIAL_EQUITY)
        "daily_summary_log_filename": "reports/daily_summary_log.csv",
        "strategy_performance_log_filename": "reports/strategy_performance_log.csv",
        "optimized_params_csv": "reports/optimized_params_history.csv", # CSV for optimized params
        "model_metadata_csv": "reports/model_metadata_history.csv", # CSV for model metadata
        "bayesian_opt_n_calls": 20, # Number of optimization iterations for Bayesian Optimization
        "bayesian_opt_n_initial_points": 5, # Number of random initial points for Bayesian Optimization
        "bayesian_opt_random_state": 42 # Random state for reproducibility
    },
    # --- Sentinel Specific Configurations ---
    "sentinel": {
        "api_check_interval_seconds": 60, # How often to check API connectivity
        "api_latency_threshold_ms": 500, # Max acceptable API response latency
        "model_output_deviation_threshold": 0.2, # Max acceptable deviation in model output (e.g., confidence score)
        "unusual_trade_volume_multiplier": 5.0, # Multiplier for detecting unusually large trades
        "max_consecutive_errors": 3, # Max errors before triggering shutdown
        "alert_recipient_email": "your_alert_email@example.com" # Email for critical alerts
    },
    # --- Logging Configuration ---
    "logging": {
        "log_level": "INFO", # DEBUG, INFO, WARNING, ERROR, CRITICAL
        "log_file": "logs/ares_system.log",
        "log_to_console": True,
        "max_bytes": 10485760, # 10 MB
        "backup_count": 5 # Keep 5 backup log files
    },
    # --- Pipeline Specific Configurations ---
    "pipeline": {
        "loop_interval_seconds": 10, # How often the main loop runs (e.g., check for new candle)
        "strategist_update_interval_minutes": 1440, # 24 hours (daily update)
        "supervisor_update_interval_minutes": 1440, # 24 hours (daily update)
        "sentinel_check_interval_seconds": 30 # How often Sentinel runs its checks
    },
    # --- Live Trading Specific Configurations ---
    "live_trading": {
        "enabled": False, # Set to True to enable live trading (USE WITH EXTREME CAUTION)
        "testnet": True, # Set to True to use Binance Testnet
        "api_key": "YOUR_BINANCE_API_KEY", # !!! REPLACE WITH YOUR ACTUAL API KEY !!!
        "api_secret": "YOUR_BINANCE_API_SECRET", # !!! REPLACE WITH YOUR ACTUAL API SECRET !!!
        "websocket_streams": {
            "kline": "", # Will be set dynamically below
            "aggTrade": "", # Will be set dynamically below
            "depth": "", # Will be set dynamically below
            "userData": True # Set to True to enable user data stream (requires API Key/Secret)
        }
    },
    # --- Firestore Configuration ---
    "firestore": {
        "enabled": False, # Set to True to enable Firestore integration
        "trade_logs_collection": "trade_logs",
        "optimized_params_collection": "optimized_params",
        "model_metadata_collection": "model_metadata",
        "system_config_collection": "system_config", # For storing overall CONFIG state if needed
        "user_data_collection_path": "users", # Path for user-specific data
        "public_data_collection_path": "public/data" # Path for public/shared data
    },
    # --- Optimization Configuration ---
    # Defines the parameter space for grid search and fine-tuning.
    # Keys are parameter paths (e.g., "analyst.market_regime_classifier.adx_period")
    # Values are lists of values for coarse grid, or ranges for fine-tuning.
    "OPTIMIZATION_CONFIG": {
        "COARSE_GRID_RANGES": {
            # General Trading Parameters
            "atr.stop_loss_multiplier": [1.0, 1.5, 2.0],
            "atr.max_risk_per_trade_pct": [0.005, 0.01, 0.02],
            "general_trading.sr_proximity_pct": [0.003, 0.005, 0.007], # Updated path
            
            # Analyst - Market Regime Classifier
            "analyst.market_regime_classifier.adx_period": [10, 14, 20],
            "analyst.market_regime_classifier.trend_scaling_factor": [50, 100, 150],
            "analyst.market_regime_classifier.trend_threshold": [20, 25, 30],

            # Analyst - Regime Predictive Ensembles
            "analyst.regime_predictive_ensembles.min_confluence_confidence": [0.6, 0.7, 0.8],
            "analyst.regime_predictive_ensembles.meta_learner_l1_reg": [0.01, 0.1, 0.5],
            "analyst.regime_predictive_ensembles.meta_learner_l2_reg": [0.01, 0.1, 0.5],

            # Analyst - Liquidation Risk Model
            "analyst.liquidation_risk_model.volatility_impact": [0.3, 0.4, 0.5],
            "analyst.liquidation_risk_model.order_book_depth_impact": [0.2, 0.3, 0.4],
            "analyst.liquidation_risk_model.atr_to_std_factor": [2.0, 2.5, 3.0],
            "analyst.liquidation_risk_model.ob_depth_range_pct": [0.003, 0.005, 0.007],
            "analyst.liquidation_risk_model.liq_buffer_zone_pct": [0.0005, 0.001, 0.002],
            "analyst.liquidation_risk_model.liq_buffer_weight": [1.0, 2.0, 3.0],
            "analyst.liquidation_risk_model.ob_depth_scaling_factor": [5.0, 10.0, 15.0],
            "analyst.liquidation_risk_model.max_safe_distance_pct": [0.03, 0.05, 0.07],

            # Analyst - Market Health Analyzer (weights)
            "analyst.market_health_analyzer.atr_weight": [0.2, 0.3, 0.4],
            "analyst.market_health_analyzer.bollinger_weight": [0.2, 0.3, 0.4],
            
            # Tactician - Laddering
            "tactician.laddering.initial_leverage": [20, 25, 30],
            "tactician.laddering.min_lss_for_ladder": [60, 70, 80],
            "tactician.laddering.ladder_step_leverage_increase": [3, 5, 7],
            
            # Strategist
            "strategist.trading_range_atr_multiplier": [2.0, 3.0, 4.0],
            "strategist.sr_relevance_threshold": [4.0, 5.0, 6.0],
            "strategist.avwap_anchor_period": [30, 60, 90],

            # Supervisor - Dynamic Risk Allocation
            "supervisor.risk_allocation_lookback_days": [20, 30, 40],
            "supervisor.max_capital_allocation_increase_pct": [0.75, 1.0, 1.25]
        },
        # Multiplier for defining the fine-tuning range around a coarse best parameter.
        # E.g., if best is X, fine-tune range is [X * (1-FINE_TUNE_RANGES_MULTIPLIER), X * (1+FINE_TUNE_RANGES_MULTIPLIER)]
        "FINE_TUNE_RANGES_MULTIPLIER": 0.15, # 15% range around the best coarse value
        "FINE_TUNE_NUM_POINTS": 5, # Number of points to sample in the fine-tuning range

        "INTEGER_PARAMS": [
            "bollinger_bands.window", "bollinger_bands.num_std_dev", "atr.window",
            "analyst.feature_engineering.wavelet_level", "analyst.feature_engineering.autoencoder_latent_dim",
            "analyst.market_regime_classifier.kmeans_n_clusters", "analyst.market_regime_classifier.adx_period",
            "analyst.market_regime_classifier.macd_fast_period", "analyst.market_regime_classifier.macd_slow_period",
            "analyst.market_regime_classifier.macd_signal_period",
            "analyst.liquidation_risk_model.lookback_periods_volatility",
            "analyst.liquidation_risk_model.atr_period",
            "analyst.market_health_analyzer.momentum_period",
            "analyst.high_impact_candle_model.volume_sma_period", "analyst.high_impact_candle_model.atr_sma_period",
            "tactician.laddering.initial_leverage", "tactician.laddering.ladder_step_leverage_increase",
            "tactician.laddering.max_ladder_steps",
            "strategist.ma_periods_for_bias", # Note: This is a list of integers, needs special handling
            "strategist.avwap_anchor_period",
            "supervisor.risk_allocation_lookback_days"
        ],
        # Groups of parameters that represent weights and should sum to 1 (or be normalized)
        # Format: ("path.to.parent_dict", ["weight1_key", "weight2_key", ...])
        "WEIGHT_PARAMS_GROUPS": [
            ("analyst.regime_predictive_ensembles.ensemble_weights", ["lstm", "transformer", "statistical", "volume"]),
            ("analyst.liquidation_risk_model", ["volatility_impact", "order_book_depth_impact", "position_impact"]),
            ("analyst.market_health_analyzer", ["atr_weight", "bollinger_weight", "ma_cluster_weight", "momentum_weight", "obv_weight"]),
        ]
    }
}

# Dynamically set filenames based on general config values
CONFIG["KLINES_FILENAME"] = f"data_cache/{CONFIG['SYMBOL']}_{CONFIG['INTERVAL']}_{CONFIG['LOOKBACK_YEARS']}y_klines.csv"
CONFIG["AGG_TRADES_FILENAME"] = f"data_cache/{CONFIG['SYMBOL']}_{CONFIG['LOOKBACK_YEARS']}y_aggtrades.csv"
CONFIG["FUTURES_FILENAME"] = f"data_cache/{CONFIG['SYMBOL']}_futures_{CONFIG['LOOKBACK_YEARS']}y_data.csv"
CONFIG["PREPARED_DATA_FILENAME"] = f"data_cache/{CONFIG['SYMBOL']}_{CONFIG['INTERVAL']}_{CONFIG['LOOKBACK_YEARS']}y_prepared_data.csv"

# Dynamically set WebSocket stream names
CONFIG["live_trading"]["websocket_streams"]["kline"] = f"{CONFIG['SYMBOL'].lower()}@kline_{CONFIG['INTERVAL']}"
CONFIG["live_trading"]["websocket_streams"]["aggTrade"] = f"{CONFIG['SYMBOL'].lower()}@aggTrade"
CONFIG["live_trading"]["websocket_streams"]["depth"] = f"{CONFIG['SYMBOL'].lower()}@depth5@100ms"

# Expose top-level variables for backward compatibility if needed, but prefer CONFIG['KEY']
SYMBOL = CONFIG['SYMBOL']
INTERVAL = CONFIG['INTERVAL']
LOOKBACK_YEARS = CONFIG['LOOKBACK_YEARS']
KLINES_FILENAME = CONFIG['KLINES_FILENAME']
AGG_TRADES_FILENAME = CONFIG['AGG_TRADES_FILENAME']
FUTURES_FILENAME = CONFIG['FUTURES_FILENAME']
PREPARED_DATA_FILENAME = CONFIG['PREPARED_DATA_FILENAME']
DOWNLOADER_SCRIPT_NAME = CONFIG['DOWNLOADER_SCRIPT_NAME']
PREPARER_SCRIPT_NAME = CONFIG['PREPARER_SCRIPT_NAME']
PIPELINE_SCRIPT_NAME = CONFIG['PIPELINE_SCRIPT_NAME']
BACKTESTING_PIPELINE_SCRIPT_NAME = CONFIG['BACKTESTING_PIPELINE_SCRIPT_NAME']
PIPELINE_PID_FILE = CONFIG['PIPELINE_PID_FILE']
RESTART_FLAG_FILE = CONFIG['RESTART_FLAG_FILE']
INITIAL_EQUITY = CONFIG['INITIAL_EQUITY']
RISK_PER_TRADE_PCT = CONFIG['RISK_PER_TRADE_PCT']
BEST_PARAMS = CONFIG['BEST_PARAMS'] # BEST_PARAMS is now directly part of CONFIG
