# src/transition/combined_features_builder.py

from dataclasses import dataclass

REQUIRED_FEATURES = [
"log_returns",
"volatility_20",
"volume_ratio",
"rsi",
"macd",
"macd_signal",
"macd_histogram",
"bb_position",
"bb_width",
"atr",
"volatility_regime",
"volatility_acceleration",
]


@dataclass


