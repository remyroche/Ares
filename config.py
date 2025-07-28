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
    }
}
