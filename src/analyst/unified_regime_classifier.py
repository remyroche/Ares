# src/analyst/unified_regime_classifier.py
import os
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
from hmmlearn import hmm
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import CONFIG
from src.tactician.sr_breakout_predictor import SRBreakoutPredictor
from src.utils.logger import system_logger
from src.utils.error_handler import (
handle_errors,
)
from src.utils.warning_symbols import (
warning,
)
from src.utils.centralized_decorators_simple import (
comprehensive_data_validation,
validate_data_quality,
with_tracing_span,
)


class UnifiedRegimeClassifier:
    passpass  # TODO: Add implementation
"""
Unified Market Regime Classifier with HMM-based labeling and ensemble prediction.

Approach:
    pass1. HMM-based labeling for basic regimes (BULL, BEAR, SIDEWAYS, VOLATILE)
2. Ensemble prediction with majority voting for basic regimes
3. Location classification (SUPPORT, RESISTANCE, OPEN_RANGE)
"""

def __init__(...):
    passpasspass# Ensure NumPy RNG pickles created under different versions can be loaded
self._enable_numpy_rng_unpickle_compat(system_logger)
self.config = config.get("analyst", {}).get("unified_regime_classifier", {})
self.global_config = config
self.logger = system_logger.getChild("UnifiedRegimeClassifier")
self.exchange = exchange
self.symbol = symbol

# Add print method for compatibility
self.print = self.logger.info

# HMM Configuration - enforce at least 4 states (BULL, BEAR, SIDEWAYS, VOLATILE)
configured_states = int(self.config.get("n_states", 4))
self.n_states = max(4, configured_states)
self.n_iter = self.config.get("n_iter", 100)
self.random_state = self.config.get("random_state", 42)
self.target_timeframe = self.config.get(
"target_timeframe",
"1h",
)  # Strategist works on 1h timeframe
self.volatility_period = self.config.get("volatility_period", 10)

# Thresholds for regime interpretation (configurable)
# Optimized thresholds for better regime balance (reduced from 23 to 18)
self.adx_sideways_threshold = self.config.get("adx_sideways_threshold", 18)  # Lowered for better balance
self.volatility_threshold = self.config.get("volatility_threshold", 0.025)  # Keep same
self.atr_normalized_threshold = self.config.get(
"atr_normalized_threshold",
0.035,  # Keep same
)
self.volatility_percentile_threshold = self.config.get(
"volatility_percentile_threshold",
0.80,  # Keep same
)
# Additional volatility breadth threshold using Bollinger Band width
self.bb_width_volatility_threshold = self.config.get(
"bb_width_volatility_threshold",
0.045,  # Keep same
)

# Log how regime state targets are chosen
self.logger.info(
{
"msg": "UnifiedRegimeClassifier configuration",
"n_states_configured": configured_states,
"n_states_enforced_min": self.n_states,
"min_data_points_default": self.config.get("min_data_points", 1000),
"thresholds": {
"adx_sideways_threshold": self.adx_sideways_threshold,
"volatility_threshold": self.volatility_threshold,
"atr_normalized_threshold": self.atr_normalized_threshold,
"volatility_percentile_threshold": self.volatility_percentile_threshold,
"bb_width_volatility_threshold": self.bb_width_volatility_threshold,
},
}
)

# Detect BLANK mode and adjust minimum data points accordingly
import os

blank_mode = os.environ.get("BLANK_TRAINING_MODE", "0") == "1"
if blank_mode:
    passself.min_data_points = self.config.get(
"min_data_points",
50,
)  # Reduced for BLANK mode
self.logger.info(
"🔧 BLANK MODE DETECTED: Using reduced minimum data points (50)",
)
else:
    passself.min_data_points = self.config.get(
"min_data_points",
1000,
)  # Default for full mode

# Models
self.hmm_model = None
self.scaler = None
self.state_to_regime_map = {}

# Ensemble Models for Basic Regimes
self.basic_ensemble = None

# Location Classifier
self.location_classifier = None
self.location_label_encoder = None

# Enhanced S/R Integration with SRBreakoutPredictor
self.enable_sr_integration = self.config.get("enable_sr_integration", True)
self.sr_predictor = None
self.basic_label_encoder = None

# S/R Configuration for enhanced regime analysis with optimized parameters
self.sr_config = {
"sr_breakout_predictor": {
"enable_sr_breakout_tactics": True,
"sr_proximity_threshold": 0.02,
"breakout_confidence_threshold": 0.6,
"sr_detection_method": "fractal",
"min_sr_strength": 0.3,
"max_sr_levels": 10,
"sr_lookback_periods": 100,
"volume_weight": 0.7,
"price_weight": 0.3,
"atr_multiplier": 1.5,
"breakout_confirmation_periods": 3,
"false_breakout_filter": True,
"use_optimized_params": True,  # Enable optimized parameters

# Enhanced strength calculation configuration
"strength_calculation": {
"enable_enhanced_strength": True,
"touch_count_lookback": 50,
"bounce_rate_threshold": 0.02,
"isolation_distance_threshold": 0.05,
"age_decay_factor": 0.95
},

# DBSCAN clustering configuration
"dbscan_clustering": {
"enable_dbscan_clustering": True,
"eps": 0.01,
"min_samples": 2,
"enable_noise_filtering": True
},

# Feature calculation configuration
"feature_calculation": {
"enable_comprehensive_features": True,
"strength_score_weights": {
"touch_count": 0.3,
"total_volume": 0.2,
"level_age": 0.2,
"bounce_rate": 0.2,
"isolation_score": 0.1
}
}
}
}

# Training Status
self.trained = False
self.last_training_time = None

# Model Paths
# Resolve checkpoints directory to an absolute path anchored at the project root
# Go up from src/analyst/unified_regime_classifier.py → src/analyst → src → project root
project_root = os.path.dirname(
os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
base_checkpoint_dir = CONFIG.get("CHECKPOINT_DIR", "checkpoints")
if not os.path.isabs(base_checkpoint_dir):
    passbase_checkpoint_dir = os.path.join(project_root, base_checkpoint_dir)

self.model_dir = os.path.join(base_checkpoint_dir, "analyst_models")
# Optional hierarchical directory for compatibility
self._hierarchical_model_dir = os.path.join(
self.model_dir,
self.exchange,
self.symbol,
self.target_timeframe,
)
os.makedirs(self.model_dir, exist_ok=True)

self.hmm_model_path = os.path.join(
self.model_dir,
f"unified_hmm_model_{self.exchange}_{self.symbol}_{self.target_timeframe}.joblib",
)
self.ensemble_model_path = os.path.join(
self.model_dir,
f"unified_ensemble_model_{self.exchange}_{self.symbol}_{self.target_timeframe}.joblib",
)
self.location_model_path = os.path.join(
self.model_dir,
f"unified_location_model_{self.exchange}_{self.symbol}_{self.target_timeframe}.joblib",
)

# --- Compatibility shim for NumPy RNG unpickling across versions ---
# Some pickles created with older/newer NumPy versions may store the BitGenerator
# as a class object instead of a simple string (e.g.,
# <class 'numpy.random._mt19937.MT19937'>). Newer NumPy expects a string name.
# We normalize the argument before delegating to NumPy's constructor.
_NUMPY_RNG_UNPICKLE_PATCHED = False

async def initialize_sr_predictor(...) -> ...:
    passpass"""..."""
    passtry:
    pass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
if not self.enable_sr_integration:
    passself.logger.info("S/R integration disabled, skipping SRBreakoutPredictor initialization")
return True

self.logger.info("🚀 Initializing SRBreakoutPredictor for enhanced regime analysis...")

# Initialize SRBreakoutPredictor with enhanced configuration
self.sr_predictor = SRBreakoutPredictor(self.sr_config)
init_success = await self.sr_predictor.initialize()

if not init_success:
    passpasspassself.logger.error("❌ Failed to initialize SRBreakoutPredictor")
return False

self.logger.info("✅ SRBreakoutPredictor initialized successfully for enhanced regime analysis")
return True

except Exception as e:
    passpasspasspasspasspasspasspassself.logger.error(f"❌ Error initializing SRBreakoutPredictor: {e}")
return False

@staticmethod
def _enable_numpy_rng_unpickle_compat(...) -> ...:
    """..."""
    pass# Use an attribute on the function to avoid double patching within this class
if getattr(
UnifiedRegimeClassifier._enable_numpy_rng_unpickle_compat, "_patched", False
):
    passreturn
try:
    pass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
import numpy.random._pickle as np_random_pickle  # type: ignore[attr-defined]

original_ctor = getattr(np_random_pickle, "__bit_generator_ctor", None)
if original_ctor is None:
    passUnifiedRegimeClassifier._enable_numpy_rng_unpickle_compat._patched = (
True
)
return

# Delegate to a module-level picklable ctor defined here to avoid closures
def _normalized_numpy_bitgen_ctor(
bit_generator_name, state=None, *args, **kwargs
):  # type: ignore[override]
name_candidate = bit_generator_name
try:
    pass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
if hasattr(name_candidate, "__name__"):
    passname_candidate = name_candidate.__name__
elif isinstance(name_candidate, str) and name_candidate.startswith(
"<class "
):
    passpassname_candidate = name_candidate.split(".")[-1].split("'>")[0]
except Exception as e:
    passpasspasspasspasspasspassself.logger.debug(f"Error parsing name candidate: {e}")
effective_state = kwargs.get("state", state)
try:
    pass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
return original_ctor(name_candidate, effective_state)
except (TypeError, ValueError):
    passpasstry:
    pass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
return original_ctor(name_candidate)
except Exception as ctor_exc:  # noqa: BLE001
try:
    pass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
import numpy as _np

bitgen_cls = getattr(_np.random, name_candidate, None)
if bitgen_cls is None and name_candidate == "MT19937":
    passtry:
    pass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
import numpy.random._mt19937 as _mt  # type: ignore[attr-defined]

bitgen_cls = getattr(_mt, "MT19937", None)
except Exception:
    passpassbitgen_cls = None
if bitgen_cls is not None:
    passreturn bitgen_cls()
except Exception as e:
    passpasspasspasspasspasspassself.logger.debug(f"Error creating bitgen class: {e}")
raise ctor_exc

np_random_pickle.__bit_generator_ctor = _normalized_numpy_bitgen_ctor  # type: ignore[attr-defined]
UnifiedRegimeClassifier._enable_numpy_rng_unpickle_compat._patched = True
if logger is not None:
    passlogger.info("Applied NumPy RNG unpickle compatibility shim (URC)")
except Exception as _shim_exc:
    passpasspasspasspasspasspassUnifiedRegimeClassifier._enable_numpy_rng_unpickle_compat._patched = True
if logger is not None:
    passlogger.warning(
warning(f"NumPy RNG unpickle shim not applied (URC): {_shim_exc}")
)

@handle_errors(
exceptions=(Exception,),
default_return=False,
context="UnifiedRegimeClassifier.initialize",
)
async def initialize(...) -> ...:
    """..."""
    passself.logger.info(
f"Initializing UnifiedRegimeClassifier for {self.exchange}_{self.symbol}",
)

# Create model directory if it doesn't exist
os.makedirs(self.model_dir, exist_ok=True)

# Try to load existing models
if self.load_models():
    passpassself.logger.info("✅ Loaded existing models successfully")
self.trained = True
else:
    passself.logger.info("ℹ️  No existing models found, will train new models")
self.trained = False

self.logger.info("✅ UnifiedRegimeClassifier initialized successfully")
return True

async def _calculate_features(...) -> ...:
    """..."""
    passif min_data_points is None:
    passmin_data_points = self.min_data_points

if len(klines_df) < min_data_points:
    passself.logger.warning(
f"Insufficient data: {len(klines_df)} < {min_data_points}. Consider reducing min_data_points or collecting more data.",
)
# Try with a lower threshold if possible
if len(klines_df) >= 200:  # Minimum viable amount
self.logger.info(
f"Proceeding with {len(klines_df)} data points (reduced from {min_data_points})",
)
min_data_points = len(klines_df)
else:
    passpassself.logger.error(
f"Data too small: {len(klines_df)} < 200 minimum required",
)
return pd.DataFrame()

self.logger.info(f"🔧 Calculating features for {len(klines_df)} periods...")

# Create features DataFrame
features_df = klines_df.copy()

# Basic price features using price differences
features_df["log_returns"] = np.log(
features_df["close"] / features_df["close"].shift(1),
)
features_df["price_change"] = features_df["close"].pct_change()
features_df["high_low_diff_ratio"] = features_df["high"].diff() / (
features_df["low"].diff() + 1e-8
)
features_df["close_open_diff_ratio"] = features_df["close"].diff() / (
features_df["open"].diff() + 1e-8
)

# Volatility features
features_df["volatility_20"] = features_df["log_returns"].rolling(20).std()
features_df["volatility_10"] = features_df["log_returns"].rolling(10).std()
features_df["volatility_5"] = features_df["log_returns"].rolling(5).std()

# Slightly enhanced volatility features (kept simple)
# EWMA-based volatility provides a smoother, more reactive estimate
features_df["ewma_volatility_20"] = (
features_df["log_returns"].ewm(span=20, adjust=False).std()
)

# Volume features
features_df["volume_ratio"] = (
features_df["volume"] / features_df["volume"].rolling(20).mean()
)
features_df["volume_change"] = features_df["volume"].pct_change()

# Technical indicators
features_df = self._calculate_rsi(features_df)
features_df = self._calculate_macd(features_df)
features_df = self._calculate_bollinger_bands(features_df)
features_df = self._calculate_atr(features_df)
features_df["atr_normalized"] = features_df["atr"] / (
features_df["close"].diff().abs() + 1e-8
)
features_df = self._calculate_adx(features_df)

# Enhanced volatility features for VOLATILE regime detection
features_df["volatility_regime"] = self._calculate_volatility_regime(
features_df,
)
features_df["volatility_acceleration"] = features_df["volatility_20"].diff()
features_df["volatility_momentum"] = features_df["volatility_20"] - features_df[
"volatility_20"
].shift(5)

# Enhanced S/R features for improved regime analysis
if self.sr_predictor and self.enable_sr_integration:
    passpass# Handle async call for enhanced features
try:
    passpass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
features_df = await self._add_enhanced_sr_features(features_df)
except Exception as e:
    passpasspasspasspasspasspassself.logger.warning(f"Enhanced S/R features failed, falling back to basic: {e}")
features_df = self._add_basic_sr_features(features_df)
else:
    pass# Basic S/R features as fallback
features_df = self._add_basic_sr_features(features_df)

# Improved NaN handling: use forward fill for technical indicators
# This preserves more data points while maintaining feature quality
technical_columns = [
"rsi",
"macd",
"macd_signal",
"macd_histogram",
"bb_upper",
"bb_middle",
"bb_lower",
"bb_position",
"bb_width",
"atr",
"atr_normalized",
"adx",
"volatility_regime",
# Enhanced S/R features
"sr_proximity",
"sr_strength",
"sr_zone_width",
"sr_cluster_count",
"sr_fibonacci_proximity",
"sr_elliott_proximity",
"sr_order_flow_imbalance",
"sr_enhanced_strength",
"sr_touch_count",
"sr_bounce_rate",
"sr_isolation_score",
]

for col in technical_columns:
    passpassif col in features_df.columns:
    pass# Forward fill NaN values for technical indicators
features_df[col] = features_df[col].ffill()
# Fill any remaining NaN values with 0
features_df[col] = features_df[col].fillna(0)

# For log_returns and other price-based features, use 0 for NaN
price_features = [
"log_returns",
"price_change",
"volume_change",
"volatility_acceleration",
"volatility_momentum",
]
for col in price_features:
    passpassif col in features_df.columns:
    passfeatures_df[col] = features_df[col].fillna(0)

# For ratio features, use 1 for NaN (neutral ratio)
ratio_features = [
"high_low_diff_ratio",
"close_open_diff_ratio",
"volume_ratio",
]
for col in ratio_features:
    passif col in features_df.columns:
    passfeatures_df[col] = features_df[col].fillna(1)

# For volatility features, use 0 for NaN
vol_features = [
"volatility_20",
"volatility_10",
"volatility_5",
"ewma_volatility_20",
]
for col in vol_features:
    passif col in features_df.columns:
    passfeatures_df[col] = features_df[col].fillna(0)

# Only drop rows that still have NaN values after all the filling
# This should be minimal now
initial_length = len(features_df)
features_df = features_df.dropna()
dropped_rows = initial_length - len(features_df)

self.logger.info(
f"✅ Calculated comprehensive features for {len(features_df)} periods (dropped {dropped_rows} rows due to NaN)",
)

return features_df

def _calculate_volatility_regime(...) -> ...:
    pass"""..."""
    pass# Calculate rolling volatility percentiles (prefer smoothed EWMA if present)
vol_baseline = (
features_df["ewma_volatility_20"]
if "ewma_volatility_20" in features_df.columns
else features_df["volatility_20"]
)
vol_20_percentile = vol_baseline.rolling(100).rank(pct=True)
vol_10_percentile = features_df["volatility_10"].rolling(100).rank(pct=True)

# High volatility regime (configurable top percentile of volatility)
pct = float(getattr(self, "volatility_percentile_threshold", 0.8))
# Use OR to be more permissive (either horizon in the top percentile marks high vol)
high_vol = (vol_20_percentile > pct) | (vol_10_percentile > pct)

# Additional breadth indicators of volatility
atr_norm_high = (
features_df["atr_normalized"]
> float(getattr(self, "atr_normalized_threshold", 0.03))
if "atr_normalized" in features_df.columns
else False
)
bb_width_high = (
features_df["bb_width"]
> float(getattr(self, "bb_width_volatility_threshold", 0.04))
if "bb_width" in features_df.columns
else False
)

# Volatility acceleration (more permissive)
vol_accel = features_df["volatility_20"].diff() > 0

# Combine conditions for VOLATILE regime
volatile_regime = high_vol | atr_norm_high | bb_width_high | vol_accel

return volatile_regime.astype(int)

async def _add_enhanced_sr_features(...) -> ...:
    passpass"""..."""
    passtry:
    pass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
self.logger.info("🔧 Adding enhanced S/R features...")

# Initialize enhanced S/R features
features_df["sr_proximity"] = 0.0
features_df["sr_strength"] = 0.0
features_df["sr_zone_width"] = 0.0
features_df["sr_cluster_count"] = 0
features_df["sr_fibonacci_proximity"] = 0.0
features_df["sr_elliott_proximity"] = 0.0
features_df["sr_order_flow_imbalance"] = 0.0
features_df["sr_enhanced_strength"] = 0.0
features_df["sr_touch_count"] = 0
features_df["sr_bounce_rate"] = 0.0
features_df["sr_isolation_score"] = 0.0

# Calculate enhanced S/R features for each data point
for i in range(50, len(features_df)):  # Start after enough data for S/R calculation
try:
    passpass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
# Get window of data for S/R analysis
window_data = features_df.iloc[max(0, i-100):i+1]
current_price = features_df["close"].iloc[i]

# Get enhanced S/R context
sr_context = await self.sr_predictor.get_sr_context(window_data, current_price)

if sr_context:
    pass# S/R proximity to nearest levels
nearest_support = sr_context.get("nearest_support", current_price * 0.95)
nearest_resistance = sr_context.get("nearest_resistance", current_price * 1.05)

# Calculate proximity (0 = at level, 1 = far from level)
support_proximity = abs(current_price - nearest_support) / current_price
resistance_proximity = abs(current_price - nearest_resistance) / current_price
features_df.loc[features_df.index[i], "sr_proximity"] = min(support_proximity, resistance_proximity)

# S/R strength
support_strength = sr_context.get("support_strength", 0.5)
resistance_strength = sr_context.get("resistance_strength", 0.5)
features_df.loc[features_df.index[i], "sr_strength"] = max(support_strength, resistance_strength)

# Enhanced strength
enhanced_support = sr_context.get("enhanced_strength_support", {})
enhanced_resistance = sr_context.get("enhanced_strength_resistance", {})
max_enhanced_strength = max(
enhanced_support.get("max_strength", 0.5),
enhanced_resistance.get("max_strength", 0.5)
)
features_df.loc[features_df.index[i], "sr_enhanced_strength"] = max_enhanced_strength

# S/R zone width
sr_zone_width = sr_context.get("sr_zone_width", 0.0)
features_df.loc[features_df.index[i], "sr_zone_width"] = sr_zone_width

# Clustering information
clustering_result = sr_context.get("clustering_result", {})
features_df.loc[features_df.index[i], "sr_cluster_count"] = clustering_result.get("n_clusters", 0)

# Fibonacci levels proximity
fibonacci_levels = sr_context.get("fibonacci_levels", {})
if fibonacci_levels:
    passfib_proximities = []
for fib_price in fibonacci_levels.values():
    passif isinstance(fib_price, (int, float)):
    passfib_proximities.append(abs(current_price - fib_price) / current_price)
if fib_proximities:
    passfeatures_df.loc[features_df.index[i], "sr_fibonacci_proximity"] = min(fib_proximities)

# Elliott Wave levels proximity
elliott_levels = sr_context.get("elliott_wave_levels", {})
if elliott_levels:
    passwave_levels = elliott_levels.get("wave_levels", {})
if wave_levels:
    passelliott_proximities = []
for wave_price in wave_levels.values():
    passif isinstance(wave_price, (int, float)):
    passelliott_proximities.append(abs(current_price - wave_price) / current_price)
if elliott_proximities:
    passfeatures_df.loc[features_df.index[i], "sr_elliott_proximity"] = min(elliott_proximities)

# Order flow imbalances
order_flow_analysis = sr_context.get("order_flow_analysis", {})
imbalances = order_flow_analysis.get("imbalances", [])
if imbalances:
    passtotal_imbalance_volume = sum(imb.get("volume", 0.0) for imb in imbalances)
features_df.loc[features_df.index[i], "sr_order_flow_imbalance"] = total_imbalance_volume

# Enhanced metrics from clustered levels
support_levels = sr_context.get("support_levels", [])
resistance_levels = sr_context.get("resistance_levels", [])
all_levels = support_levels + resistance_levels

if all_levels:
    passpass# Calculate average touch count and bounce rate
touch_counts = []
bounce_rates = []
isolation_scores = []

for level in all_levels:
    passif isinstance(level, dict):
    passtouch_counts.append(level.get("touches", 0))
bounce_rates.append(level.get("bounce_rate", 0.0))
isolation_scores.append(level.get("isolation_score", 0.5))

if touch_counts:
    passfeatures_df.loc[features_df.index[i], "sr_touch_count"] = np.mean(touch_counts)
if bounce_rates:
    passfeatures_df.loc[features_df.index[i], "sr_bounce_rate"] = np.mean(bounce_rates)
if isolation_scores:
    passfeatures_df.loc[features_df.index[i], "sr_isolation_score"] = np.mean(isolation_scores)

except Exception as e:
    passpasspasspasspasspasspass# Continue with next data point if there's an error
self.logger.debug(f"Error calculating enhanced S/R features for index {i}: {e}")
continue

self.logger.info("✅ Enhanced S/R features added successfully")
return features_df

except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error adding enhanced S/R features: {e}")
return self._add_basic_sr_features(features_df)

def _add_basic_sr_features(...) -> ...:
    """..."""
    passtry:
    pass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
self.logger.info("🔧 Adding basic S/R features...")

# Initialize basic S/R features
features_df["sr_proximity"] = 0.0
features_df["sr_strength"] = 0.0
features_df["sr_zone_width"] = 0.0
features_df["sr_cluster_count"] = 0
features_df["sr_fibonacci_proximity"] = 0.0
features_df["sr_elliott_proximity"] = 0.0
features_df["sr_order_flow_imbalance"] = 0.0
features_df["sr_enhanced_strength"] = 0.0
features_df["sr_touch_count"] = 0
features_df["sr_bounce_rate"] = 0.0
features_df["sr_isolation_score"] = 0.0

# Calculate basic S/R features using rolling windows
for i in range(20, len(features_df)):
    passtry:
    pass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
# Get window of data for basic S/R analysis
window_data = features_df.iloc[max(0, i-20):i+1]
current_price = features_df["close"].iloc[i]

# Calculate basic pivot levels
high = window_data["high"].max()
low = window_data["low"].min()
close = window_data["close"].iloc[-1]
pivot = (high + low + close) / 3

# Basic support and resistance
r1 = 2 * pivot - low
s1 = 2 * pivot - high

# Calculate basic proximity
support_proximity = abs(current_price - s1) / current_price
resistance_proximity = abs(current_price - r1) / current_price
features_df.loc[features_df.index[i], "sr_proximity"] = min(support_proximity, resistance_proximity)

# Basic strength (based on volume near levels)
tolerance = window_data["close"].std() * 0.1
volume_near_support = window_data[abs(window_data["close"] - s1) <= tolerance]["volume"].sum()
volume_near_resistance = window_data[abs(window_data["close"] - r1) <= tolerance]["volume"].sum()
total_volume = window_data["volume"].sum()

support_strength = volume_near_support / total_volume if total_volume > 0 else 0.5
resistance_strength = volume_near_resistance / total_volume if total_volume > 0 else 0.5
features_df.loc[features_df.index[i], "sr_strength"] = max(support_strength, resistance_strength)

# Basic zone width
zone_width = (r1 - s1) / current_price
features_df.loc[features_df.index[i], "sr_zone_width"] = zone_width

# Set other features to neutral values
features_df.loc[features_df.index[i], "sr_enhanced_strength"] = 0.5
features_df.loc[features_df.index[i], "sr_touch_count"] = 1
features_df.loc[features_df.index[i], "sr_bounce_rate"] = 0.5
features_df.loc[features_df.index[i], "sr_isolation_score"] = 0.5

except Exception as e:
    passpasspasspasspasspasspasspass# Continue with next data point if there's an error
self.logger.debug(f"Error calculating basic S/R features for index {i}: {e}")
continue

self.logger.info("✅ Basic S/R features added successfully")
return features_df

except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error adding basic S/R features: {e}")
return features_df

@handle_errors(
exceptions=(Exception,),
default_return=None,
context="UnifiedRegimeClassifier._calculate_rsi",
)
def _calculate_rsi(...) -> ...:
    """..."""
    pass# Use price differences instead of absolute prices
close_diff = df["close"].diff()
gain = (close_diff.where(close_diff > 0, 0)).rolling(window=period).mean()
loss = (-close_diff.where(close_diff < 0, 0)).rolling(window=period).mean()
rs = gain / loss
df["rsi"] = 100 - (100 / (1 + rs))
return df

@handle_errors(
exceptions=(Exception,),
default_return=None,
context="UnifiedRegimeClassifier._calculate_macd",
)
def _calculate_macd(...) -> ...:
    """..."""
    pass# Use price differences instead of absolute prices
close_diff = df["close"].diff()
exp1 = close_diff.ewm(span=fast).mean()
exp2 = close_diff.ewm(span=slow).mean()
df["macd"] = exp1 - exp2
df["macd_signal"] = df["macd"].ewm(span=signal).mean()
df["macd_histogram"] = df["macd"] - df["macd_signal"]
return df

@handle_errors(
exceptions=(Exception,),
default_return=None,
context="UnifiedRegimeClassifier._calculate_adx",
)
def _calculate_adx(...) -> ...:
    """..."""
    passhigh = df["high"]
low = df["low"]
close = df["close"]

# Calculate True Range (TR)
tr1 = high - low
tr2 = abs(high - close.shift(1))
tr3 = abs(low - close.shift(1))
tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
atr = tr.ewm(alpha=1 / period, adjust=False).mean()

# Calculate Directional Movement (+DM, -DM)
move_up = high.diff()
move_down = low.diff()
plus_dm = ((move_up > move_down) & (move_up > 0)) * move_up
minus_dm = ((move_down > move_up) & (move_down > 0)) * move_down

plus_dm = plus_dm.ewm(alpha=1 / period, adjust=False).mean()
minus_dm = minus_dm.ewm(alpha=1 / period, adjust=False).mean()

# Calculate Directional Index (+DI, -DI)
plus_di = 100 * (plus_dm / atr)
minus_di = 100 * (minus_dm / atr)

# Calculate Directional Movement Index (DX) and ADX
dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
df["adx"] = dx.ewm(alpha=1 / period, adjust=False).mean()
df["adx"] = df["adx"].fillna(25)  # Fill initial NaNs with a neutral value

return df

@handle_errors(
exceptions=(Exception,),
default_return=None,
context="UnifiedRegimeClassifier._calculate_bollinger_bands",
)
def _calculate_bollinger_bands(...) -> ...:
    pass"""..."""
    pass# Use price differences instead of absolute prices
close_diff = df["close"].diff()
sma = close_diff.rolling(window=period).mean()
std = close_diff.rolling(window=period).std()
df["bb_upper"] = sma + (std * std_dev)
df["bb_lower"] = sma - (std * std_dev)
df["bb_position"] = (close_diff - df["bb_lower"]) / (
df["bb_upper"] - df["bb_lower"]
)
df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma
return df

@handle_errors(
exceptions=(Exception,),
default_return=None,
context="UnifiedRegimeClassifier._calculate_atr",
)
def _calculate_atr(...) -> ...:
    """..."""
    pass# Use price differences instead of absolute prices
high_diff = df["high"].diff()
low_diff = df["low"].diff()
close_diff = df["close"].diff()

high_low_diff = high_diff - low_diff
high_close_diff = np.abs(high_diff - close_diff.shift())
low_close_diff = np.abs(low_diff - close_diff.shift())
true_range = np.maximum(
high_low_diff, np.maximum(high_close_diff, low_close_diff)
)
df["atr"] = true_range.rolling(window=period).mean()
return df

# --- Simple fallbacks are no longer required when using basic decorator without recovery strategies ---

def _interpret_hmm_states(...) -> ...:
    """..."""
    passanalysis_df = features_df.copy()
analysis_df["state"] = state_sequence
state_analysis = {}

# Thresholds for regime classification come from instance configuration
adx_sideways_threshold = self.adx_sideways_threshold
volatility_threshold = self.volatility_threshold
atr_norm_threshold = self.atr_normalized_threshold

for state in range(self.n_states):
    passstate_data = analysis_df[analysis_df["state"] == state]

if len(state_data) == 0:
    passcontinue

# Calculate state characteristics
mean_return = state_data["log_returns"].mean()
mean_volatility = state_data["volatility_20"].mean()
mean_adx = state_data["adx"].mean()  # Calculate the mean ADX for the state
mean_atr_norm = state_data["atr_normalized"].mean()

# Optimized regime classification logic for better balance
# Check for sideways movement (moderate directional strength)
is_sideways = mean_adx < adx_sideways_threshold

# First check if it's clearly sideways (low ADX and small returns)
if is_sideways and abs(mean_return) < 0.0003:  # Reduced sideways return threshold
regime = "SIDEWAYS"
# Then check for strong directional movements (reduced ADX requirement)
elif mean_return > 0.0005 and mean_adx > adx_sideways_threshold:  # Reduced return threshold
regime = "BULL"
elif mean_return < -0.0005 and mean_adx > adx_sideways_threshold:  # Reduced return threshold
regime = "BEAR"
# For borderline cases, use a more balanced approach
elif is_sideways:
    passpass# If ADX is low but returns are significant, still classify as directional
if abs(mean_return) > 0.0002:  # Small but meaningful returns
regime = "BULL" if mean_return > 0 else "BEAR"
else:
    passpassregime = "SIDEWAYS"
else:
    pass# Default to directional based on return sign
if mean_return >= 0:
    passregime = "BULL"
else:
    passregime = "BEAR"

state_analysis[state] = {
"regime": regime,
"mean_return": mean_return,
"mean_volatility": mean_volatility,
"mean_adx": mean_adx,  # Store for analysis
"mean_atr_norm": mean_atr_norm,
"count": len(state_data),
}

self.logger.info(
f"State {state}: {regime} "
f"(mean_return={mean_return:.4f}, mean_vol={mean_volatility:.4f}, mean_adx={mean_adx:.2f}, "
f"is_sideways={is_sideways})",
)

# Create state to regime mapping
state_to_regime_map = {
state: data["regime"]
for state, data in state_analysis.items()
if isinstance(state, int)
}

# Persist mapping without post-hoc coverage enforcement
state_analysis["state_to_regime_map"] = state_to_regime_map

# Log summary of how regimes are derived from HMM states
mapped_counts: dict[str, int] = {}
for state, data in state_analysis.items():
    passif state == "state_to_regime_map":
    passcontinue
mapped_counts[data["regime"]] = mapped_counts.get(data["regime"], 0) + int(
data.get("count", 0)
)

self.logger.info(
{
"msg": "HMM state mapping summary",
"n_states": self.n_states,
"unique_mapped_regimes": sorted(
list({r for r in state_to_regime_map.values()})
),
"mapped_counts": mapped_counts,
"thresholds": {
"adx_sideways_threshold": adx_sideways_threshold,
"volatility_threshold": volatility_threshold,
"atr_normalized_threshold": atr_norm_threshold,
},
}
)

return state_analysis

async def _calculate_enhanced_sr_levels(...) -> ...:
    """..."""
    passtry:
    pass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
if not self.sr_predictor or not self.enable_sr_integration:
    pass# Fallback to basic pivot calculation if SRBreakoutPredictor not available
return await self._calculate_basic_pivots(df_window)

if len(df_window) < 5:
    passreturn {
"s1": 0, "s2": 0, "r1": 0, "r2": 0, "pivot": 0,
"enhanced_strengths": {},
"clustering_result": {},
"fibonacci_levels": {},
"elliott_wave_levels": {},
"order_flow_analysis": {},
}

# Get current price for S/R context
current_price = df_window["close"].iloc[-1]

# Get comprehensive S/R context from SRBreakoutPredictor
sr_context = await self.sr_predictor.get_sr_context(df_window, current_price)

if not sr_context:
    passpassself.logger.warning("Failed to get S/R context, falling back to basic calculation")
return await self._calculate_basic_pivots(df_window)

# Extract enhanced S/R levels and metrics
support_levels = sr_context.get("support_levels", [])
resistance_levels = sr_context.get("resistance_levels", [])

# Get nearest levels for traditional pivot format
nearest_support = sr_context.get("nearest_support", current_price * 0.95)
nearest_resistance = sr_context.get("nearest_resistance", current_price * 1.05)

# Calculate traditional pivot levels as fallback
high = df_window["high"].max()
low = df_window["low"].min()
close = df_window["close"].iloc[-1]
pivot = (high + low + close) / 3

# Use enhanced levels if available, otherwise use traditional calculations
s1 = nearest_support if support_levels else (2 * pivot - high)
r1 = nearest_resistance if resistance_levels else (2 * pivot - low)
s2 = s1 * 0.95 if support_levels else (pivot - (high - low))
r2 = r1 * 1.05 if resistance_levels else (pivot + (high - low))

# Enhanced strength metrics
enhanced_strengths = {
"support_strength": sr_context.get("support_strength", 0.5),
"resistance_strength": sr_context.get("resistance_strength", 0.5),
"enhanced_strength_support": sr_context.get("enhanced_strength_support", {}),
"enhanced_strength_resistance": sr_context.get("enhanced_strength_resistance", {}),
}

return {
"s1": s1,
"s2": s2,
"r1": r1,
"r2": r2,
"pivot": pivot,
"enhanced_strengths": enhanced_strengths,
"clustering_result": sr_context.get("clustering_result", {}),
"fibonacci_levels": sr_context.get("fibonacci_levels", {}),
"elliott_wave_levels": sr_context.get("elliott_wave_levels", {}),
"order_flow_analysis": sr_context.get("order_flow_analysis", {}),
"support_levels": support_levels,
"resistance_levels": resistance_levels,
"sr_zone_width": sr_context.get("sr_zone_width", 0.0),
}

except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error calculating enhanced S/R levels: {e}")
return await self._calculate_basic_pivots(df_window)

async def _calculate_basic_pivots(...) -> ...:
    """..."""
    passtry:
    pass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
if len(df_window) < 5:
    passreturn {
"s1": 0, "s2": 0, "r1": 0, "r2": 0, "pivot": 0,
"enhanced_strengths": {},
"clustering_result": {},
"fibonacci_levels": {},
"elliott_wave_levels": {},
"order_flow_analysis": {},
}

# Calculate basic pivot point
high = df_window["high"].max()
low = df_window["low"].min()
close = df_window["close"].iloc[-1]
pivot = (high + low + close) / 3

# Calculate basic support and resistance levels
r1 = 2 * pivot - low
r2 = pivot + (high - low)
s1 = 2 * pivot - high
s2 = pivot - (high - low)

return {
"s1": s1,
"s2": s2,
"r1": r1,
"r2": r2,
"pivot": pivot,
"enhanced_strengths": {
"support_strength": 0.5,
"resistance_strength": 0.5,
},
"clustering_result": {},
"fibonacci_levels": {},
"elliott_wave_levels": {},
"order_flow_analysis": {},
"support_levels": [],
"resistance_levels": [],
"sr_zone_width": 0.0,
}

except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error calculating basic pivots: {e}")
return {
"s1": 0, "s2": 0, "r1": 0, "r2": 0, "pivot": 0,
"enhanced_strengths": {},
"clustering_result": {},
"fibonacci_levels": {},
"elliott_wave_levels": {},
"order_flow_analysis": {},
}

async def _analyze_enhanced_volume_levels(...) -> ...:
    """..."""
    passtry:
    pass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
if not self.sr_predictor or not self.enable_sr_integration:
    pass# Fallback to basic volume analysis
return self._analyze_basic_volume_levels(df_window)

if df_window.empty or len(df_window) < 20:
    passreturn None

# Get current price for S/R context
current_price = df_window["close"].iloc[-1]

# Get comprehensive S/R context including order flow analysis
sr_context = await self.sr_predictor.get_sr_context(df_window, current_price)

if not sr_context:
    passpassself.logger.warning("Failed to get S/R context for volume analysis, falling back to basic")
return self._analyze_basic_volume_levels(df_window)

# Extract order flow analysis
order_flow_analysis = sr_context.get("order_flow_analysis", {})

if not order_flow_analysis:
    passpassreturn self._analyze_basic_volume_levels(df_window)

# Extract enhanced volume levels
volume_profile = order_flow_analysis.get("volume_profile", {})
poc_level = volume_profile.get("poc", {})
value_area = volume_profile.get("value_area", {})
hvns = volume_profile.get("high_volume_nodes", [])

analyzed_levels = {}

# Process POC (Point of Control)
if poc_level:
    passanalyzed_levels["poc"] = {
"price": poc_level.get("price", current_price),
"volume": poc_level.get("volume", 0.0),
"age": poc_level.get("age", 0),
"touches": poc_level.get("touches", 0),
"strength": poc_level.get("strength", 0.5),
"volume_strength": poc_level.get("volume_strength", 0.5),
"touch_strength": poc_level.get("touch_strength", 0.5),
"age_strength": poc_level.get("age_strength", 0.5),
"enhanced_metrics": {
"value_area_high": value_area.get("high", current_price * 1.02),
"value_area_low": value_area.get("low", current_price * 0.98),
"value_area_volume": value_area.get("volume", 0.0),
}
}

# Process top HVNs
for i, hvn in enumerate(hvns[:2]):  # Top 2 HVNs
level_name = "hvn_primary" if i == 0 else "hvn_secondary"
analyzed_levels[level_name] = {
"price": hvn.get("price", current_price),
"volume": hvn.get("volume", 0.0),
"age": hvn.get("age", 0),
"touches": hvn.get("touches", 0),
"strength": hvn.get("strength", 0.5),
"volume_strength": hvn.get("volume_strength", 0.5),
"touch_strength": hvn.get("touch_strength", 0.5),
"age_strength": hvn.get("age_strength", 0.5),
"enhanced_metrics": {
"cluster_id": hvn.get("cluster_id", -1),
"isolation_score": hvn.get("isolation_score", 0.5),
"bounce_rate": hvn.get("bounce_rate", 0.0),
}
}

# Add order flow imbalances if available
imbalances = order_flow_analysis.get("imbalances", [])
if imbalances:
    passanalyzed_levels["order_imbalances"] = {
"count": len(imbalances),
"total_volume": sum(imb.get("volume", 0.0) for imb in imbalances),
"average_size": np.mean([imb.get("size", 0.0) for imb in imbalances]) if imbalances else 0.0,
}

return analyzed_levels if analyzed_levels else None

except Exception as e:
    passpasspasspasspasspasspasspasspassself.logger.error(f"Error in enhanced volume analysis: {e}")
return self._analyze_basic_volume_levels(df_window)

def _analyze_basic_volume_levels(...) -> ...:
    """..."""
    passif df_window.empty or len(df_window) < 20:
    passreturn None

# --- 1. ATR-Dynamic Binning ---
high_low = df_window["high"] - df_window["low"]
high_close = abs(df_window["high"] - df_window["close"].shift())
low_close = abs(df_window["low"] - df_window["close"].shift())
tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
avg_atr = tr.mean()
bin_size = max(avg_atr * 0.25, 1e-6)
min_price = df_window["low"].min()
max_price = df_window["high"].max()
bins = np.arange(min_price, max_price, bin_size)

# --- 2. Find Top 2 HVNs ---
price_bins = pd.cut(df_window["close"], bins=bins, right=False)
volume_by_bin = df_window.groupby(price_bins, observed=False)["volume"].sum()
if volume_by_bin.empty:
    passreturn None

# Get the top 2 HVNs by volume
top_hvns = volume_by_bin.nlargest(2)
if top_hvns.empty:
    passreturn None

# --- 3. Analyze Each HVN ---
analyzed_levels = {}
for i, (level_bin, level_volume) in enumerate(top_hvns.items()):
    passlevel_price = level_bin.mid

# Find first time price entered this bin to determine age
level_indices = df_window.index[
df_window["close"].between(level_bin.left, level_bin.right)
]
if len(level_indices) == 0:
    passcontinue

first_touch_index = level_indices[0]
age = len(df_window) - df_window.index.get_loc(first_touch_index)

# Count touches after formation
touches = 0
data_since_formation = df_window.loc[first_touch_index:]
for k in range(1, len(data_since_formation)):
    passprev_high = data_since_formation["high"].iloc[k - 1]
prev_low = data_since_formation["low"].iloc[k - 1]
curr_high = data_since_formation["high"].iloc[k]
curr_low = data_since_formation["low"].iloc[k]

# A "touch" is when price crosses the level
if (prev_low < level_price < curr_high) or (
prev_high > level_price > curr_low
):
    passtouches += 1

# Calculate additional strength metrics
# Volume strength (normalized)
total_volume = df_window["volume"].sum()
volume_strength = (
min(level_volume / total_volume, 1.0) if total_volume > 0 else 0.0
)

# Touch strength (normalized)
touch_strength = min(touches / 10.0, 1.0)  # Normalize touches

# Age strength (normalized)
age_strength = min(age / len(df_window), 1.0)  # Normalize age

# Calculate overall strength (0.0 to 1.0)
# Factors: volume (50%), touches (30%), age (20%)
overall_strength = (
volume_strength * 0.5 + touch_strength * 0.3 + age_strength * 0.2
)

level_name = "poc" if i == 0 else "hvn_secondary"
analyzed_levels[level_name] = {
"price": level_price,
"volume": level_volume,
"age": age,  # in number of candles
"touches": touches,
"strength": overall_strength,
"volume_strength": volume_strength,
"touch_strength": touch_strength,
"age_strength": age_strength,
}

return analyzed_levels

@validate_data_quality(validation_level="WARNING")
@with_tracing_span("enhanced_location_classification")
async def _classify_enhanced_location(...) -> ...:
    """..."""
    passself.logger.info(
"Classifying location with enhanced S/R analysis using SRBreakoutPredictor...",
)

# --- Configuration for enhanced analysis ---
long_term_period = self.config.get("long_term_hvn_period", 720)  # 30 days on 1h chart
short_term_period = self.config.get("short_term_pivot_period", 24)  # 1 day on 1h chart
tolerance = self.config.get("level_tolerance", 0.01)  # 1% proximity tolerance
min_level_touches = self.config.get("min_level_touches", 1)  # Must have at least 1 re-test
min_strength_threshold = self.config.get("min_strength_threshold", 0.3)  # Minimum S/R strength

# Check if we have enough data for location classification
if len(features_df) < long_term_period:
    passpasspassself.logger.warning(
f"Insufficient data for enhanced location classification. "
f"Need at least {long_term_period} rows, but only have {len(features_df)}. "
f"Returning all OPEN_RANGE labels."
)
return ["OPEN_RANGE"] * len(features_df)

locations = []

# Start loop after the longest period to ensure enough data for all calculations
start_index = long_term_period
for i in range(start_index, len(features_df)):
    passcurrent_price = features_df["close"].iloc[i]
current_price_diff = features_df["close"].diff().iloc[i]

# --- 1. Enhanced S/R Analysis (Short-Term) ---
short_window = features_df.iloc[i - short_term_period : i]
short_sr_levels = await self._calculate_enhanced_sr_levels(short_window)

# --- 2. Enhanced Volume Analysis (Long-Term) ---
long_window = features_df.iloc[i - long_term_period : i]
volume_levels = await self._analyze_enhanced_volume_levels(long_window)

# --- 3. Enhanced Classification Logic ---
loc_sr = None
loc_volume = None
loc_fibonacci = None
loc_elliott = None

# Check S/R level proximity with enhanced strength filtering
support_levels = short_sr_levels.get("support_levels", [])
resistance_levels = short_sr_levels.get("resistance_levels", [])

# Check support levels
for level in support_levels:
    passpassif isinstance(level, dict):
    passlevel_price = level.get("price", level)
level_strength = level.get("enhanced_strength", 0.5)
level_touches = level.get("touches", 0)
else:
    passlevel_price = level
level_strength = 0.5
level_touches = 1

if (level_touches >= min_level_touches and
level_strength >= min_strength_threshold and
abs(current_price - level_price) / current_price <= tolerance):
    passloc_sr = "ENHANCED_SUPPORT"
break

# Check resistance levels
if not loc_sr:
    passfor level in resistance_levels:
    passif isinstance(level, dict):
    passlevel_price = level.get("price", level)
level_strength = level.get("enhanced_strength", 0.5)
level_touches = level.get("touches", 0)
else:
    passlevel_price = level
level_strength = 0.5
level_touches = 1

if (level_touches >= min_level_touches and
level_strength >= min_strength_threshold and
abs(current_price - level_price) / current_price <= tolerance):
    passloc_sr = "ENHANCED_RESISTANCE"
break

# Check volume levels (POC, HVNs)
if volume_levels:
    passfor level_name, level_data in volume_levels.items():
    passif level_name in ["poc", "hvn_primary", "hvn_secondary"]:
    passif (level_data.get("touches", 0) >= min_level_touches and
level_data.get("strength", 0.5) >= min_strength_threshold and
abs(current_price - level_data["price"]) / current_price <= tolerance):
    passlevel_type = "SUPPORT" if current_price > level_data["price"] else "RESISTANCE"
loc_volume = f"ENHANCED_{level_name.upper()}_{level_type}"
break

# Check Fibonacci levels
fibonacci_levels = short_sr_levels.get("fibonacci_levels", {})
if fibonacci_levels:
    passfor fib_type, fib_price in fibonacci_levels.items():
    passif abs(current_price - fib_price) / current_price <= tolerance:
    passfib_direction = "SUPPORT" if current_price > fib_price else "RESISTANCE"
loc_fibonacci = f"FIBONACCI_{fib_type}_{fib_direction}"
break

# Check Elliott Wave levels
elliott_levels = short_sr_levels.get("elliott_wave_levels", {})
if elliott_levels:
    passwave_levels = elliott_levels.get("wave_levels", {})
for wave_type, wave_price in wave_levels.items():
    passif abs(current_price - wave_price) / current_price <= tolerance:
    passwave_direction = "SUPPORT" if current_price > wave_price else "RESISTANCE"
loc_elliott = f"ELLIOTT_{wave_type}_{wave_direction}"
break

# --- 4. Enhanced Final Label Assignment with Priority ---
# Priority: Elliott Wave > Fibonacci > Enhanced S/R > Volume Levels
if loc_elliott:
    passlocations.append(loc_elliott)
elif loc_fibonacci:
    passpasslocations.append(loc_fibonacci)
elif loc_sr and loc_volume:
    passpass# High confluence: Enhanced S/R aligns with volume level
if "SUPPORT" in loc_sr and "SUPPORT" in loc_volume:
    passpasslocations.append("ENHANCED_CONFLUENCE_SUPPORT")
elif "RESISTANCE" in loc_sr and "RESISTANCE" in loc_volume:
    passpasslocations.append("ENHANCED_CONFLUENCE_RESISTANCE")
else:
    passlocations.append(loc_sr)  # Prefer S/R over volume
elif loc_sr:
    passpasslocations.append(loc_sr)
elif loc_volume:
    passpasslocations.append(loc_volume)
else:
    passlocations.append("OPEN_RANGE")

# Pad the beginning of the list for alignment
padding = ["OPEN_RANGE"] * start_index
final_locations = padding + locations

self.logger.info(
f"Finished enhanced location classification. Found: {pd.Series(final_locations).value_counts().to_dict()}",
)
return final_locations

@validate_data_quality(validation_level="WARNING")
@with_tracing_span("location_classification")
def _classify_location(...) -> ...:
    """..."""
    passif self.sr_predictor and self.enable_sr_integration:
    pass# Use enhanced classification if SRBreakoutPredictor is available
import asyncio
try:
    passpass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
# Create event loop if none exists
try:
    passpass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
loop = asyncio.get_event_loop()
except RuntimeError:
    passpassloop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

return loop.run_until_complete(self._classify_enhanced_location(features_df))
except Exception as e:
    passpasspasspasspasspasspassself.logger.warning(f"Enhanced location classification failed, falling back to basic: {e}")
return self._classify_basic_location(features_df)
else:
    passreturn self._classify_basic_location(features_df)

def _classify_basic_location(...) -> ...:
    """..."""
    passself.logger.info("Using basic location classification...")

# --- Configuration for dual-timeframe analysis ---
long_term_hvn_period = self.config.get("long_term_hvn_period", 720)  # 30 days on 1h chart
short_term_pivot_period = self.config.get("short_term_pivot_period", 24)  # 1 day on 1h chart
tolerance = self.config.get("level_tolerance", 0.01)  # 1% proximity tolerance
min_level_touches = self.config.get("min_level_touches", 1)  # Must have at least 1 re-test

# Check if we have enough data for location classification
if len(features_df) < long_term_hvn_period:
    passpassself.logger.warning(
f"Insufficient data for location classification. "
f"Need at least {long_term_hvn_period} rows, but only have {len(features_df)}. "
f"Returning all OPEN_RANGE labels."
)
return ["OPEN_RANGE"] * len(features_df)

locations = []

# Start loop after the longest period to ensure enough data for all calculations
start_index = long_term_hvn_period
for i in range(start_index, len(features_df)):
    passcurrent_price_diff = features_df["close"].diff().iloc[i]

# --- 1. Tactical Pivot Analysis (Short-Term) ---
pivot_window = features_df.iloc[i - short_term_pivot_period : i]
pivots = self._calculate_basic_pivots(pivot_window)
pivot_supports = [pivots["s1"], pivots["s2"]]
pivot_resistances = [pivots["r1"], pivots["r2"]]

# --- 2. Strategic Volume Level Analysis (Long-Term) ---
hvn_window = features_df.iloc[i - long_term_hvn_period : i]
volume_levels = self._analyze_basic_volume_levels(hvn_window)

# --- 3. Classification Logic ---
loc_pivot = None
loc_hvn = None

# Check for Pivot proximity using price differences
for p_sup in pivot_supports:
    passif (
abs(current_price_diff - p_sup) / (abs(current_price_diff) + 1e-8)
<= tolerance
):
    passloc_pivot = "PIVOT_S"
break
if not loc_pivot:
    passfor p_res in pivot_resistances:
    passif (
abs(current_price_diff - p_res)
/ (abs(current_price_diff) + 1e-8)
<= tolerance
):
    passloc_pivot = "PIVOT_R"
break

# Check for HVN proximity using price differences
if volume_levels:
    passpassfor level_data in volume_levels.values():
    pass# Intelligence Rule: Filter out untested levels
if level_data["touches"] < min_level_touches:
    passcontinue

if (
abs(current_price_diff - level_data["price"])
/ (abs(current_price_diff) + 1e-8)
<= tolerance
):
    passhvn_type = (
"SUPPORT"
if current_price_diff > level_data["price"]
else "RESISTANCE"
)
loc_hvn = f"HVN_{hvn_type}"
break  # Stop at the first significant HVN found

# --- 4. Final Label Assignment ---
if loc_pivot and loc_hvn:
    pass# A tactical pivot aligns with a strategic volume level - high confluence
if "S" in loc_pivot and "SUPPORT" in loc_hvn:
    passpasslocations.append("CONFLUENCE_S")
elif "R" in loc_pivot and "RESISTANCE" in loc_hvn:
    passpasslocations.append("CONFLUENCE_R")
else:
    passlocations.append(loc_pivot)
elif loc_pivot:
    passpasslocations.append(loc_pivot)
elif loc_hvn:
    passpasslocations.append(loc_hvn)
else:
    passlocations.append("OPEN_RANGE")

# Pad the beginning of the list for alignment
padding = ["OPEN_RANGE"] * start_index
final_locations = padding + locations

self.logger.info(
f"Finished basic location classification. Found: {pd.Series(final_locations).value_counts().to_dict()}",
)
return final_locations

async def train_hmm_labeler(...) -> ...:
    """..."""
    passtry:
    pass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
self.logger.info("🎓 Training HMM-based Market Regime Classifier with enhanced S/R integration...")

# Initialize SRBreakoutPredictor for enhanced analysis
if self.enable_sr_integration:
    passpasspasssr_init_success = await self.initialize_sr_predictor()
if not sr_init_success:
    passself.logger.warning("Failed to initialize SRBreakoutPredictor, continuing with basic analysis")
self.enable_sr_integration = False

# Calculate features
features_df = await self._calculate_features(historical_klines)
if features_df.empty:
    passpassself.logger.error("No features available for HMM training")
return False

# Prepare features for HMM
hmm_features = features_df[
[
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
"adx",
"volatility_regime",
"volatility_acceleration",
]
].fillna(0)

# Scale features
self.scaler = StandardScaler()
hmm_features_scaled = self.scaler.fit_transform(hmm_features)

# Train HMM model
self.hmm_model = hmm.GaussianHMM(
n_components=self.n_states,
n_iter=self.n_iter,
random_state=self.random_state,
covariance_type="full",
)

self.hmm_model.fit(hmm_features_scaled)

# Get state sequence
state_sequence = self.hmm_model.predict(hmm_features_scaled)

# Interpret states and create regime mapping
state_analysis = self._interpret_hmm_states(features_df, state_sequence)
self.state_to_regime_map = state_analysis["state_to_regime_map"]

self.logger.info("✅ HMM-based regime classifier trained successfully")
return True

except Exception as e:
    passpasspasspasspasspasspasspassself.logger.error(f"❌ Failed to train HMM regime classifier: {e}")
return False

async def train_location_classifier(...) -> ...:
    """..."""
    passtry:
    pass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
self.logger.info("🎓 Training Location Classifier...")

# Calculate features
features_df = await self._calculate_features(historical_klines)
if features_df.empty:
    passself.logger.error(
"No features available for location classifier training",
)
return False

# Check if we have enough data for location classification
long_term_hvn_period = self.config.get("long_term_hvn_period", 720)
if len(features_df) < long_term_hvn_period:
    passpassself.logger.warning(
f"Insufficient data for location classification. "
f"Need at least {long_term_hvn_period} rows, but only have {len(features_df)}. "
f"Skipping location classifier training."
)
return True  # Return True to avoid breaking the pipeline

# Get location labels using the new _classify_location method
location_labels = self._classify_location(features_df)

# Verify that location labels match the features length
if len(location_labels) != len(features_df):
    passpassself.logger.error(
f"Location labels length ({len(location_labels)}) does not match "
f"features length ({len(features_df)}). Skipping location classifier training."
)
return True  # Return True to avoid breaking the pipeline

# Encode location labels
self.location_label_encoder = LabelEncoder()
location_encoded = self.location_label_encoder.fit_transform(
location_labels,
)

# Prepare features for location classification
location_features = features_df[
["close", "volume", "volatility_20", "rsi", "bb_position", "atr"]
].fillna(0)

# Train location classifier
self.location_classifier = LGBMClassifier(
n_estimators=100,
learning_rate=0.1,
max_depth=6,
random_state=42,
verbose=-1,
)

self.location_classifier.fit(location_features, location_encoded)

self.logger.info("✅ Location classifier trained successfully")
return True

except Exception as e:
    passpasspasspasspasspasspasspassself.logger.error(f"❌ Failed to train location classifier: {e}")
return False

async def train_basic_ensemble(...) -> ...:
    """..."""
    passtry:
    pass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
self.logger.info("🎓 Training Basic Regime Ensemble...")

# Calculate features
features_df = await self._calculate_features(historical_klines)
if features_df.empty:
    passself.logger.error("No features available for ensemble training")
return False

# Get HMM-based labels
hmm_features = features_df[
[
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
"adx",
"volatility_regime",
"volatility_acceleration",
]
].fillna(0)
hmm_features_scaled = self.scaler.transform(hmm_features)
state_sequence = self.hmm_model.predict(hmm_features_scaled)

# Map states to regimes
regime_labels = [
self.state_to_regime_map.get(state, "SIDEWAYS")
for state in state_sequence
]

# Encode regime labels
self.basic_label_encoder = LabelEncoder()
regime_encoded = self.basic_label_encoder.fit_transform(regime_labels)

# Prepare features for ensemble
ensemble_features = features_df[
[
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
"adx",
"volatility_regime",
"volatility_acceleration",
]
].fillna(0)

# Train ensemble
self.basic_ensemble = LGBMClassifier(
n_estimators=100,
learning_rate=0.1,
max_depth=6,
random_state=42,
verbose=-1,
)

self.basic_ensemble.fit(ensemble_features, regime_encoded)

self.logger.info("✅ Basic regime ensemble trained successfully")
return True

except Exception as e:
    passpasspasspasspasspasspasspassself.logger.error(f"❌ Failed to train basic ensemble: {e}")
return False

async def train_complete_system(...) -> ...:
    """..."""
    passtry:
    pass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
self.logger.info("🎓 Training Complete Regime Classification System...")

# Initialize SR analyzer
# Legacy S/R/Candle code removed

# Train HMM labeler
if not await self.train_hmm_labeler(historical_klines):
    passreturn False

# Train basic ensemble
if not await self.train_basic_ensemble(historical_klines):
    passreturn False

# Train location classifier
if not await self.train_location_classifier(historical_klines):
    passreturn False

self.trained = True
self.last_training_time = datetime.now()

self.logger.info(
"✅ Complete regime classification system trained successfully",
)
# Persist trained models so subsequent runs can load them
self.save_models()
return True

except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"❌ Failed to train complete system: {e}")
return False

async def predict_regime(...) -> ...:
    """..."""
    passtry:
    pass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
if not self.trained:
    passself.logger.warning("Models not trained, returning default prediction")
return "SIDEWAYS", 0.5, {}

# Calculate features
features_df = await self._calculate_features(current_klines)
if features_df.empty:
    passreturn "SIDEWAYS", 0.5, {}

current_features = features_df.iloc[-1] if len(features_df) > 0 else None
if current_features is None:
    passreturn "SIDEWAYS", 0.5, {}

# Predict regime
regime_features = features_df[
[
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
].fillna(0)

if self.basic_ensemble:
    passregime_proba = self.basic_ensemble.predict_proba(
regime_features.iloc[-1:],
)
regime_pred = self.basic_ensemble.predict(regime_features.iloc[-1:])[0]
regime = self.basic_label_encoder.inverse_transform([regime_pred])[0]
regime_confidence = np.max(regime_proba)
else:
    passregime = "SIDEWAYS"
regime_confidence = 0.5

additional_info = {
"regime_confidence": regime_confidence,
"features_used": list(features_df.columns),
"prediction_time": datetime.now().isoformat(),
}

return regime, regime_confidence, additional_info

except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"❌ Error in regime prediction: {e}")
return "SIDEWAYS", 0.5, {"error": str(e)}

async def predict_regime_and_location(...) -> ...:
    """..."""
    passtry:
    pass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
if not self.trained:
    passself.logger.warning("Models not trained, returning default predictions")
return "SIDEWAYS", "OPEN_RANGE", 0.5, {}

# Calculate features
features_df = await self._calculate_features(current_klines)
if features_df.empty:
    passreturn "SIDEWAYS", "OPEN_RANGE", 0.5, {}

current_features = features_df.iloc[-1] if len(features_df) > 0 else None
if current_features is None:
    passreturn "SIDEWAYS", "OPEN_RANGE", 0.5, {}

# Predict regime
regime_features = features_df[
[
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
].fillna(0)

if self.basic_ensemble:
    passregime_proba = self.basic_ensemble.predict_proba(
regime_features.iloc[-1:],
)
regime_pred = self.basic_ensemble.predict(regime_features.iloc[-1:])[0]
regime = self.basic_label_encoder.inverse_transform([regime_pred])[0]
regime_confidence = np.max(regime_proba)
else:
    passregime = "SIDEWAYS"
regime_confidence = 0.5

# Predict location
location_features = features_df[
["close", "volume", "volatility_20", "rsi", "bb_position", "atr"]
].fillna(0)

if self.location_classifier:
    passlocation_proba = self.location_classifier.predict_proba(
location_features.iloc[-1:],
)
location_pred = self.location_classifier.predict(
location_features.iloc[-1:],
)[0]
location = self.location_label_encoder.inverse_transform(
[location_pred],
)[0]
location_confidence = np.max(location_proba)
else:
    pass# Fallback to rule-based location classification
location_labels = self._classify_location(features_df)
location = location_labels[-1] if location_labels else "OPEN_RANGE"
location_confidence = 0.7

# Calculate overall confidence
overall_confidence = (regime_confidence + location_confidence) / 2

additional_info = {
"regime_confidence": regime_confidence,
"location_confidence": location_confidence,
"features_used": list(features_df.columns),
"prediction_time": datetime.now().isoformat(),
}

return regime, location, overall_confidence, additional_info

except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"❌ Error in regime/location prediction: {e}")
return "SIDEWAYS", "OPEN_RANGE", 0.5, {"error": str(e)}

def save_models(...) -> ...:
    """..."""
    passtry:
    pass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
if self.hmm_model:
    passjoblib.dump(self.hmm_model, self.hmm_model_path)
self.logger.info(f"✅ HMM model saved to {self.hmm_model_path}")

if self.basic_ensemble:
    passjoblib.dump(self.basic_ensemble, self.ensemble_model_path)
self.logger.info(
f"✅ Basic ensemble saved to {self.ensemble_model_path}",
)

if self.location_classifier:
    passjoblib.dump(self.location_classifier, self.location_model_path)
self.logger.info(
f"✅ Location classifier saved to {self.location_model_path}",
)

# Save label encoders
if self.basic_label_encoder:
    passjoblib.dump(
self.basic_label_encoder,
self.ensemble_model_path.replace(".joblib", "_encoder.joblib"),
)

if self.location_label_encoder:
    passjoblib.dump(
self.location_label_encoder,
self.location_model_path.replace(".joblib", "_encoder.joblib"),
)

except Exception:
    passpassself.logger.error(f"❌ Error saving models: {e}")

def load_models(...) -> ...:
    """..."""
    passtry:
    pass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
# Log model directory and candidate paths
self.logger.info(
{
"msg": "UnifiedRegimeClassifier model directories",
"model_dir": self.model_dir,
"hmm_model_path": self.hmm_model_path,
"ensemble_model_path": self.ensemble_model_path,
"location_model_path": self.location_model_path,
"hierarchical_model_dir": getattr(
self, "_hierarchical_model_dir", None
),
}
)

loaded_any = False

# Candidate paths (flat first, then optional hierarchical forms)
hmm_candidates = [
self.hmm_model_path,
os.path.join(
getattr(self, "_hierarchical_model_dir", self.model_dir),
"unified_hmm_model.joblib",
),
]
ensemble_candidates = [
self.ensemble_model_path,
os.path.join(
getattr(self, "_hierarchical_model_dir", self.model_dir),
"unified_ensemble_model.joblib",
),
]
location_candidates = [
self.location_model_path,
os.path.join(
getattr(self, "_hierarchical_model_dir", self.model_dir),
"unified_location_model.joblib",
),
]

def _first_existing(paths: list[str]) -> str | None:
                for p in paths:
    passif os.path.exists(p):
    passreturn p
return None

# Load HMM model
hmm_path = _first_existing(hmm_candidates)
if hmm_path is not None:
    passself.hmm_model = joblib.load(hmm_path)
self.logger.info(f"✅ Loaded HMM model from {hmm_path}")
loaded_any = True

# Load basic ensemble
ensemble_path = _first_existing(ensemble_candidates)
if ensemble_path is not None:
    passself.basic_ensemble = joblib.load(ensemble_path)
self.logger.info(f"✅ Loaded basic ensemble from {ensemble_path}")
loaded_any = True

# Load location classifier
location_path = _first_existing(location_candidates)
if location_path is not None:
    passself.location_classifier = joblib.load(location_path)
self.logger.info(f"✅ Loaded location classifier from {location_path}")
loaded_any = True

# Load label encoders
encoder_candidates = [
self.ensemble_model_path.replace(".joblib", "_encoder.joblib"),
os.path.join(
getattr(self, "_hierarchical_model_dir", self.model_dir),
"unified_ensemble_model_encoder.joblib",
),
]
enc_path = _first_existing(encoder_candidates)
if enc_path is not None:
    passself.basic_label_encoder = joblib.load(enc_path)

location_encoder_candidates = [
self.location_model_path.replace(".joblib", "_encoder.joblib"),
os.path.join(
getattr(self, "_hierarchical_model_dir", self.model_dir),
"unified_location_model_encoder.joblib",
),
]
loc_enc_path = _first_existing(location_encoder_candidates)
if loc_enc_path is not None:
    passself.location_label_encoder = joblib.load(loc_enc_path)

self.trained = loaded_any
return loaded_any

except Exception:
    passpassself.logger.error(f"❌ Error loading models: {e}")
return False

@comprehensive_data_validation
@with_tracing_span("regime_classification")
async def classify_regimes(...) -> ...:
    """..."""
    passtry:
    pass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
if not self.trained:
    passself.logger.info(
"🎓 Models not trained, training complete system now...",
)
training_success = await self.train_complete_system(historical_klines)
if not training_success:
    passself.logger.error("❌ Failed to train regime classification models")
return {"error": "Failed to train regime classification models"}

# Calculate features
features_df = await self._calculate_features(historical_klines)
if features_df.empty:
    passself.logger.error("❌ No features available for classification")
return {"error": "No features available for classification"}

# Get regime predictions
regime_features = features_df[
[
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
"adx",
"volatility_regime",
"volatility_acceleration",
]
].fillna(0)

regimes = []
confidence_scores = []
if self.basic_ensemble and self.basic_label_encoder:
    passpassself.logger.info(
"🔍 Using trained basic ensemble for regime classification",
)
regime_predictions = self.basic_ensemble.predict(regime_features)
regime_probabilities = self.basic_ensemble.predict_proba(
regime_features
)
regimes = self.basic_label_encoder.inverse_transform(
regime_predictions,
).tolist()
# Calculate confidence scores as max probability for each prediction
confidence_scores = [
float(np.max(proba)) for proba in regime_probabilities
]
unique_regimes = list(sorted(set(regimes)))
counts = {
r: int((np.array(regimes) == r).sum()) for r in unique_regimes
}
# Detailed logging on regime prediction composition
self.logger.info(
{
"msg": "Ensemble regime prediction summary",
"unique_regimes": unique_regimes,
"counts": counts,
"expected_min": self.n_states,
"total_records": int(len(regime_features)),
"thresholds": {
"adx_sideways_threshold": getattr(
self, "adx_sideways_threshold", None
),
"volatility_threshold": getattr(
self, "volatility_threshold", None
),
"atr_normalized_threshold": getattr(
self, "atr_normalized_threshold", None
),
"volatility_percentile_threshold": getattr(
self,
"volatility_percentile_threshold",
None,
),
},
}
)
if len(unique_regimes) < self.n_states:
    passself.logger.warning(
warning(
f"Fewer regimes predicted ({len(unique_regimes)}) than expected ({self.n_states}). "
"Consider increasing min_data_points or enhancing volatility features."
)
)
# Fallback to HMM states
elif self.hmm_model and self.scaler and self.state_to_regime_map:
    passpassself.logger.info("🔍 Using HMM model for regime classification")
hmm_features_scaled = self.scaler.transform(regime_features)
state_sequence = self.hmm_model.predict(hmm_features_scaled)
regimes = [
self.state_to_regime_map.get(state, "SIDEWAYS")
for state in state_sequence
]
# For HMM, use a default confidence score since we don't have probabilities
confidence_scores = [0.8] * len(
regimes
)  # Default high confidence for HMM
unique_regimes = list(sorted(set(regimes)))
counts = {
r: int((np.array(regimes) == r).sum()) for r in unique_regimes
}
self.logger.info(
{
"msg": "HMM regime prediction summary",
"unique_regimes": unique_regimes,
"counts": counts,
"expected_min": self.n_states,
"total_records": int(len(regime_features)),
"thresholds": {
"adx_sideways_threshold": getattr(
self, "adx_sideways_threshold", None
),
"volatility_threshold": getattr(
self, "volatility_threshold", None
),
"atr_normalized_threshold": getattr(
self, "atr_normalized_threshold", None
),
"volatility_percentile_threshold": getattr(
self,
"volatility_percentile_threshold",
None,
),
},
}
)
if len(unique_regimes) < self.n_states:
    passself.logger.warning(
warning(
f"Fewer regimes predicted ({len(unique_regimes)}) than expected ({self.n_states}). "
"Consider increasing min_data_points or enhancing volatility features."
)
)
else:
    passself.logger.warning(
"⚠️ No trained models available, attempting to train models now...",
)
# Try to train the complete system
training_success = await self.train_complete_system(historical_klines)
if not training_success:
    passself.logger.error("❌ Failed to train regime classification models")
return {"error": "Failed to train regime classification models"}

# Retry classification after training
self.logger.info(
"🔄 Retrying regime classification with newly trained models...",
)
return await self.classify_regimes(historical_klines)

# Get location predictions
location_labels = self._classify_location(features_df)

regime_distribution = dict(pd.Series(regimes).value_counts())
# Convert numpy types to regular Python types for clean logging
clean_distribution = {k: int(v) for k, v in regime_distribution.items()}
self.logger.info(f"📊 Regime distribution: {clean_distribution}")

return {
"regimes": regimes,
"confidence_scores": confidence_scores,
"locations": location_labels,
"total_records": len(features_df),
"regime_distribution": regime_distribution,
"location_distribution": dict(
pd.Series(location_labels).value_counts(),
),
}

except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"❌ Error in regime classification: {e}")
return {"error": str(e)}

def get_system_status(...) -> ...:
    """..."""
    passreturn {
"trained": self.trained,
"last_training_time": self.last_training_time.isoformat()
if self.last_training_time
else None,
"hmm_model_loaded": self.hmm_model is not None,
"basic_ensemble_loaded": self.basic_ensemble is not None,
"location_classifier_loaded": self.location_classifier is not None,
# Legacy S/R code removed
"n_states": self.n_states,
"target_timeframe": self.target_timeframe,
"state_to_regime_map": self.state_to_regime_map,
}
