# src/training/feature_engineering_optimizer.py

"""
Feature Engineering Optimization Module

This module optimizes feature engineering parameters using:
    pass  # TODO: Add implementation
1. Random Forest + SHAP for correlation analysis
2. Mutual importance matrix for feature parameter selection
3. Regime-specific optimization for each HMM regime
4. Top 3 parameter selection based on correlation, multicollinearity, and mutual information
5. Feature Interaction Engineering for capturing non-linear relationships
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.model_selection import cross_val_score

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors


class FeatureEngineeringOptimizer:
    pass  # TODO: Add implementation
class FeatureEngineeringOptimizer:
class FeatureEngineeringOptimizer:
    """
Optimizes feature engineering parameters using advanced ML techniques.

Features:
    - Random Forest + SHAP for correlation analysis
- Mutual importance matrix for parameter selection
- Regime-specific optimization
- Top 3 parameter selection with correlation/multicollinearity/MI analysis
- Feature Interaction Engineering for non-linear relationships
"""

def __init__(self, config: dict[str, Any]):
    def __init__(self, config: dict[str, Any]):
    def __init__(self, config: dict[str, Any]):
    def __init__(self, config: dict[str, Any]):
        """Initialize the feature engineering optimizer."""
self.config = config
self.logger = system_logger.getChild("FeatureEngineeringOptimizer")

# Feature parameter ranges to optimize
self.feature_params = {
"RSI": {
"lookback_period": [7, 14, 21, 30, 50],
"overbought_threshold": [70, 75, 80, 85],
"oversold_threshold": [15, 20, 25, 30]
},
"MACD": {
"fast_period": [8, 12, 16, 20],
"slow_period": [20, 26, 30, 34],
"signal_period": [7, 9, 11, 13]
},
"Bollinger_Bands": {
"lookback_period": [10, 20, 30, 50],
"std_dev": [1.5, 2.0, 2.5, 3.0],
"squeeze_threshold": [0.1, 0.2, 0.3, 0.4]
},
"SMA": {
"short_period": [5, 10, 15, 20],
"long_period": [20, 30, 50, 100]
},
"EMA": {
"short_period": [5, 10, 15, 20],
"long_period": [20, 30, 50, 100]
},
"ATR": {
"lookback_period": [7, 14, 21, 30]
},
"Stochastic": {
"k_period": [7, 14, 21, 30],
"d_period": [3, 5, 7, 9],
"overbought": [70, 75, 80, 85],
"oversold": [15, 20, 25, 30]
},
"ADX": {
"lookback_period": [7, 14, 21, 30],
"threshold": [20, 25, 30, 35]
},
"CCI": {
"lookback_period": [7, 14, 21, 30],
"constant": [0.015, 0.02, 0.025, 0.03]
}
}

# Feature Interaction Engineering Configuration
self.interaction_config = {
"momentum_volume": {
"enabled": True,
"weight": 1.5,
"features": ["RSI", "MACD", "Stochastic", "Volume_Ratio"]
},
"trend_volatility": {
"enabled": True,
"weight": 1.8,
"features": ["SMA_Ratio", "EMA_Ratio", "BB_Position", "ATR_Normalized"]
},
"oscillator_trend": {
"enabled": True,
"weight": 1.3,
"features": ["RSI", "Williams_R", "CCI", "SMA_Ratio"]
},
"volume_price": {
"enabled": True,
"weight": 1.6,
"features": ["OBV_Normalized", "MFI", "Price_Momentum", "Volume_Ratio"]
},
"volatility_regime": {
"enabled": True,
"weight": 1.4,
"features": ["ATR_Normalized", "BB_Squeeze", "Volatility", "Market_Regime"]
},
"cross_timeframe": {
"enabled": True,
"weight": 1.2,
"features": ["RSI_14", "RSI_30", "MACD_12_26", "MACD_20_40"]
}
}

# Optimization settings
self.optimization_config = config.get("feature_engineering_optimization", {
"n_trials": 100,
"cv_folds": 5,
"random_state": 42,
"correlation_threshold": 0.8,
"mi_threshold": 0.1,
"top_k_parameters": 3,
"interaction_enabled": True,
"max_interactions": 50,
"interaction_selection_threshold": 0.05
})

self.logger.info("🚀 Feature Engineering Optimizer initialized with interaction engineering")

@handle_errors(exceptions=(Exception,), default_return={})
async def optimize_feature_parameters(
self,
data: pd.DataFrame,
target: pd.Series,
regimes: Optional[pd.Series] = None,
symbol: str = "UNKNOWN",
exchange: str = "UNKNOWN",
timeframe: str = "1m"
) -> dict[str, Any]:
        """
Optimize feature engineering parameters for each regime.

Args:
            data: Feature data
target: Target variable
regimes: HMM regime labels (optional)
symbol: Trading symbol
exchange: Exchange name
timeframe: Timeframe

Returns:
            Dictionary with optimized parameters for each regime and feature
"""
self.logger.info(f"🎯 Starting feature parameter optimization for {symbol} on {exchange}")

results = {
"optimization_timestamp": datetime.now().isoformat(),
"symbol": symbol,
"exchange": exchange,
"timeframe": timeframe,
"regime_optimizations": {},
"global_optimizations": {},
"correlation_analysis": {},
"mutual_importance_matrix": {},
"interaction_engineering": {}
}

# 1. Global optimization (all data)
self.logger.info("🌍 Performing global feature parameter optimization...")
global_opt = await self._optimize_global_parameters(data, target)
results["global_optimizations"] = global_opt

# 2. Feature Interaction Engineering
if self.optimization_config.get("interaction_enabled", True):
            self.logger.info("🔗 Performing feature interaction engineering...")
interaction_results = await self._engineer_feature_interactions(data, target)
results["interaction_engineering"] = interaction_results

# 3. Regime-specific optimization
if regimes is not None and len(regimes.unique()) > 1:
            self.logger.info("🎭 Performing regime-specific optimization...")
for regime in regimes.unique():
                regime_mask = regimes == regime
regime_data = data[regime_mask]
regime_target = target[regime_mask]

if len(regime_data) < 100:  # Skip regimes with insufficient data
self.logger.warning(f"⚠️ Regime {regime} has insufficient data ({len(regime_data)} samples), skipping")
continue

self.logger.info(f"🎯 Optimizing parameters for regime {regime} ({len(regime_data)} samples)")
regime_opt = await self._optimize_regime_parameters(
regime_data, regime_target, regime
)
results["regime_optimizations"][f"regime_{regime}"] = regime_opt

# 4. Correlation and mutual importance analysis
self.logger.info("🔍 Performing correlation and mutual importance analysis...")
correlation_analysis = await self._analyze_correlations_and_mi(data, target)
results["correlation_analysis"] = correlation_analysis

# 5. Select top 3 parameters for each feature
self.logger.info("🏆 Selecting top 3 parameters for each feature...")
top_parameters = await self._select_top_parameters(results)
results["top_parameters"] = top_parameters

# 6. Save results
await self._save_optimization_results(results, symbol, exchange, timeframe)

self.logger.info("✅ Feature parameter optimization completed successfully")
return results

async def _optimize_global_parameters(
self,
data: pd.DataFrame,
target: pd.Series
) -> dict[str, Any]:
        """Optimize parameters globally across all data."""

# Create parameter combinations for each feature
param_combinations = {}
for feature_name, params in self.feature_params.items():
            param_combinations[feature_name] = self._generate_param_combinations(params)

# Optimize each feature
optimized_params = {}
for feature_name, combinations in param_combinations.items():
            self.logger.info(f"🔧 Optimizing {feature_name} parameters...")

# Create synthetic features for this parameter combination
feature_scores = []
for params in combinations:
                # Generate synthetic feature based on parameters
synthetic_feature = self._generate_synthetic_feature(
data, feature_name, params
)

if synthetic_feature is not None:
                    # Calculate feature importance using Random Forest + SHAP
importance_score = await self._calculate_feature_importance(
synthetic_feature, target
)
feature_scores.append({
"params": params,
"importance": importance_score,
"feature_values": synthetic_feature
})

# Sort by importance and select top parameters
if feature_scores:
                feature_scores.sort(key=lambda x: x["importance"], reverse=True)
optimized_params[feature_name] = feature_scores[:3]  # Top 3

return optimized_params

async def _optimize_regime_parameters(
self,
data: pd.DataFrame,
target: pd.Series,
regime: int
) -> dict[str, Any]:
        """Optimize parameters for a specific regime."""

# Similar to global optimization but for regime-specific data
param_combinations = {}
for feature_name, params in self.feature_params.items():
            param_combinations[feature_name] = self._generate_param_combinations(params)

optimized_params = {}
for feature_name, combinations in param_combinations.items():
            self.logger.info(f"🎭 Optimizing {feature_name} parameters for regime {regime}...")

feature_scores = []
for params in combinations:
                synthetic_feature = self._generate_synthetic_feature(
data, feature_name, params
)

if synthetic_feature is not None:
                    importance_score = await self._calculate_feature_importance(
synthetic_feature, target
)
feature_scores.append({
"params": params,
"importance": importance_score,
"feature_values": synthetic_feature
})

if feature_scores:
                feature_scores.sort(key=lambda x: x["importance"], reverse=True)
optimized_params[feature_name] = feature_scores[:3]

return optimized_params

async def _analyze_correlations_and_mi(
self,
data: pd.DataFrame,
target: pd.Series
) -> dict[str, Any]:
        """Analyze correlations and mutual information between features."""

# Calculate correlation matrix
correlation_matrix = data.corr()

# Calculate mutual information with target
mi_scores = mutual_info_regression(data, target, random_state=42)
mi_df = pd.DataFrame({
'feature': data.columns,
'mutual_information': mi_scores
}).sort_values('mutual_information', ascending=False)

# Identify highly correlated features
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
if abs(corr_value) > self.optimization_config["correlation_threshold"]:
                    high_corr_pairs.append({
'feature1': correlation_matrix.columns[i],
'feature2': correlation_matrix.columns[j],
'correlation': corr_value
})

return {
"correlation_matrix": correlation_matrix.to_dict(),
"mutual_information": mi_df.to_dict('records'),
"high_correlation_pairs": high_corr_pairs,
"correlation_threshold": self.optimization_config["correlation_threshold"]
}

async def _select_top_parameters(self, optimization_results: dict[str, Any]) -> dict[str, Any]:
        """Select top 3 parameters for each feature considering correlation, MI, etc."""

top_parameters = {}

# Process global optimizations
for feature_name, feature_results in optimization_results["global_optimizations"].items():
            if not feature_results:
                continue

# Get correlation analysis for this feature
correlation_data = optimization_results["correlation_analysis"]

# Score each parameter combination
scored_params = []
for result in feature_results:
                score = await self._calculate_comprehensive_score(
result, correlation_data, feature_name
)
scored_params.append({
"params": result["params"],
"importance": result["importance"],
"comprehensive_score": score
})

# Sort by comprehensive score and select top 3
scored_params.sort(key=lambda x: x["comprehensive_score"], reverse=True)
top_parameters[feature_name] = scored_params[:3]

return top_parameters

async def _calculate_feature_importance(
self,
feature: pd.Series,
target: pd.Series
) -> float:
        """Calculate feature importance using Random Forest + SHAP."""

try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
# Prepare data
X = feature.values.reshape(-1, 1)
y = target.values

# Train Random Forest
rf = RandomForestRegressor(
n_estimators=100,
random_state=42,
n_jobs=-1
)
rf.fit(X, y)

# Calculate SHAP values
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X)

# Calculate importance as mean absolute SHAP value
importance = np.mean(np.abs(shap_values))

return float(importance)

except Exception as e:
            self.logger.warning(f"⚠️ Error calculating feature importance: {e}")
return 0.0

async def _calculate_comprehensive_score(
self,
result: dict[str, Any],
correlation_data: dict[str, Any],
feature_name: str
) -> float:
        """Calculate comprehensive score considering multiple factors."""

base_importance = result["importance"]

# Penalty for high correlation with existing features
correlation_penalty = 0.0
feature_values = result["feature_values"]

# Check correlation with existing features
for pair in correlation_data.get("high_correlation_pairs", []):
            if feature_name in [pair["feature1"], pair["feature2"]]:
                correlation_penalty += abs(pair["correlation"]) * 0.1

# Bonus for high mutual information
mi_bonus = 0.0
for mi_item in correlation_data.get("mutual_information", []):
            if mi_item["feature"] == feature_name:
                mi_bonus = mi_item["mutual_information"] * 0.2
break

# Calculate final score
final_score = base_importance - correlation_penalty + mi_bonus

return max(0.0, final_score)

def _generate_param_combinations(self, params: dict[str, List]) -> List[dict[str, Any]]:
        """Generate all parameter combinations for a feature."""
import itertools

param_names = list(params.keys())
param_values = list(params.values())

combinations = []
for combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combination))
combinations.append(param_dict)

return combinations

def _generate_synthetic_feature(
self,
data: pd.DataFrame,
feature_name: str,
params: dict[str, Any]
) -> Optional[pd.Series]:
        """Generate actual technical indicator feature based on optimized parameters."""

try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
if 'close' not in data.columns:
                self.logger.warning(f"⚠️ No 'close' column found for {feature_name}")
return None

close_prices = data['close']

if feature_name == "RSI":
                lookback = params["lookback_period"]
return self._calculate_rsi(close_prices, lookback)

elif feature_name == "MACD":
                fast = params["fast_period"]
slow = params["slow_period"]
signal = params["signal_period"]
return self._calculate_macd(close_prices, fast, slow, signal)

elif feature_name == "Bollinger_Bands":
                lookback = params["lookback_period"]
std_dev = params["std_dev"]
return self._calculate_bollinger_position(close_prices, lookback, std_dev)

elif feature_name == "SMA":
                short_period = params["short_period"]
long_period = params["long_period"]
return self._calculate_sma_crossover(close_prices, short_period, long_period)

elif feature_name == "EMA":
                short_period = params["short_period"]
long_period = params["long_period"]
return self._calculate_ema_crossover(close_prices, short_period, long_period)

elif feature_name == "ATR":
                lookback = params["lookback_period"]
return self._calculate_atr(data, lookback)

elif feature_name == "Stochastic":
                k_period = params["k_period"]
d_period = params["d_period"]
return self._calculate_stochastic(data, k_period, d_period)

elif feature_name == "ADX":
                lookback = params["lookback_period"]
return self._calculate_adx(data, lookback)

elif feature_name == "CCI":
                lookback = params["lookback_period"]
constant = params["constant"]
return self._calculate_cci(data, lookback, constant)

return None

except Exception as e:
            self.logger.warning(f"⚠️ Error generating feature for {feature_name}: {e}")
return None

def _calculate_rsi(self, prices: pd.Series, lookback: int) -> pd.Series:
        """Calculate RSI with optimized lookback period."""
delta = prices.diff()
gain = (delta.where(delta > 0, 0)).rolling(window=lookback).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=lookback).mean()
rs = gain / loss
rsi = 100 - (100 / (1 + rs))
return rsi

def _calculate_macd(self, prices: pd.Series, fast: int, slow: int, signal: int) -> pd.Series:
        """Calculate MACD with optimized periods."""
ema_fast = prices.ewm(span=fast).mean()
ema_slow = prices.ewm(span=slow).mean()
macd_line = ema_fast - ema_slow
signal_line = macd_line.ewm(span=signal).mean()
histogram = macd_line - signal_line
return histogram

def _calculate_bollinger_position(self, prices: pd.Series, lookback: int, std_dev: float) -> pd.Series:
        """Calculate Bollinger Bands position with optimized parameters."""
sma = prices.rolling(window=lookback).mean()
std = prices.rolling(window=lookback).std()
upper_band = sma + (std * std_dev)
lower_band = sma - (std * std_dev)
# Return position within bands (0 = at lower band, 1 = at upper band)
position = (prices - lower_band) / (upper_band - lower_band)
return position

def _calculate_sma_crossover(self, prices: pd.Series, short_period: int, long_period: int) -> pd.Series:
        """Calculate SMA crossover signal."""
sma_short = prices.rolling(window=short_period).mean()
sma_long = prices.rolling(window=long_period).mean()
crossover = (sma_short - sma_long) / sma_long
return crossover

def _calculate_ema_crossover(self, prices: pd.Series, short_period: int, long_period: int) -> pd.Series:
        """Calculate EMA crossover signal."""
ema_short = prices.ewm(span=short_period).mean()
ema_long = prices.ewm(span=long_period).mean()
crossover = (ema_short - ema_long) / ema_long
return crossover

def _calculate_atr(self, data: pd.DataFrame, lookback: int) -> pd.Series:
        """Calculate ATR with optimized lookback period."""
if not all(col in data.columns for col in ['high', 'low', 'close']):
            return pd.Series(index=data.index)

high = data['high']
low = data['low']
close = data['close']

tr1 = high - low
tr2 = abs(high - close.shift())
tr3 = abs(low - close.shift())
tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
atr = tr.rolling(window=lookback).mean()
return atr

def _calculate_stochastic(self, data: pd.DataFrame, k_period: int, d_period: int) -> pd.Series:
        """Calculate Stochastic oscillator with optimized periods."""
if not all(col in data.columns for col in ['high', 'low', 'close']):
            return pd.Series(index=data.index)

high = data['high']
low = data['low']
close = data['close']

lowest_low = low.rolling(window=k_period).min()
highest_high = high.rolling(window=k_period).max()
k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
d_percent = k_percent.rolling(window=d_period).mean()
return d_percent

def _calculate_adx(self, data: pd.DataFrame, lookback: int) -> pd.Series:
        """Calculate ADX with optimized lookback period."""
if not all(col in data.columns for col in ['high', 'low', 'close']):
            return pd.Series(index=data.index)

high = data['high']
low = data['low']
close = data['close']

# Calculate +DM and -DM
high_diff = high.diff()
low_diff = low.diff()

plus_dm = pd.Series(0, index=high.index)
minus_dm = pd.Series(0, index=high.index)

plus_dm[high_diff > low_diff] = high_diff[high_diff > low_diff]
minus_dm[low_diff > high_diff] = -low_diff[low_diff > high_diff]

# Calculate TR
tr1 = high - low
tr2 = abs(high - close.shift())
tr3 = abs(low - close.shift())
tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

# Calculate smoothed values
tr_smooth = tr.rolling(window=lookback).mean()
plus_dm_smooth = plus_dm.rolling(window=lookback).mean()
minus_dm_smooth = minus_dm.rolling(window=lookback).mean()

# Calculate +DI and -DI
plus_di = 100 * (plus_dm_smooth / tr_smooth)
minus_di = 100 * (minus_dm_smooth / tr_smooth)

# Calculate DX and ADX
dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
adx = dx.rolling(window=lookback).mean()

return adx

def _calculate_cci(self, data: pd.DataFrame, lookback: int, constant: float) -> pd.Series:
        """Calculate CCI with optimized lookback period and constant."""
if not all(col in data.columns for col in ['high', 'low', 'close']):
            return pd.Series(index=data.index)

high = data['high']
low = data['low']
close = data['close']

typical_price = (high + low + close) / 3
sma_tp = typical_price.rolling(window=lookback).mean()
mad = typical_price.rolling(window=lookback).apply(lambda x: np.mean(np.abs(x - x.mean())))
cci = (typical_price - sma_tp) / (constant * mad)

return cci

@handle_errors(exceptions=(Exception,), default_return={})
async def _engineer_feature_interactions(
self,
data: pd.DataFrame,
target: pd.Series
) -> dict[str, Any]:
        """
Engineer feature interactions for capturing non-linear relationships.

Args:
            data: Feature data
target: Target variable

Returns:
            Dictionary with interaction features and their importance
"""
self.logger.info("🔗 Starting feature interaction engineering...")

interaction_results = {
"interaction_features": {},
"interaction_importance": {},
"selected_interactions": {},
"interaction_performance": {}
}

# 1. Create basic pairwise interactions
basic_interactions = self._create_basic_interactions(data)
interaction_results["interaction_features"]["basic"] = basic_interactions

# 2. Create pattern-based interactions
pattern_interactions = self._create_pattern_interactions(data)
interaction_results["interaction_features"]["pattern"] = pattern_interactions

# 3. Create regime-dependent interactions
regime_interactions = self._create_regime_interactions(data)
interaction_results["interaction_features"]["regime"] = regime_interactions

# 4. Combine all interactions
all_interactions = pd.concat([
basic_interactions,
pattern_interactions,
regime_interactions
], axis=1)

# 5. Select optimal interactions
selected_interactions = await self._select_optimal_interactions(all_interactions, target)
interaction_results["selected_interactions"] = selected_interactions

# 6. Calculate interaction importance
importance_scores = await self._calculate_interaction_importance(selected_interactions, target)
interaction_results["interaction_importance"] = importance_scores

# 7. Evaluate interaction performance
performance_metrics = await self._evaluate_interaction_performance(selected_interactions, target)
interaction_results["interaction_performance"] = performance_metrics

self.logger.info(f"✅ Feature interaction engineering completed. Created {len(selected_interactions.columns)} interaction features")

return interaction_results

def _create_basic_interactions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
Create basic pairwise interactions between features.
"""
interactions = []

# Define important feature pairs for interactions
important_pairs = [
("RSI", "MACD"),
("RSI", "Volume_Ratio"),
("MACD", "Volume_Ratio"),
("BB_Position", "ATR_Normalized"),
("SMA_Ratio", "EMA_Ratio"),
("Price_Momentum", "Volume_Ratio"),
("OBV_Normalized", "Price_Momentum"),
("Stochastic", "RSI"),
("Williams_R", "RSI"),
("CCI", "RSI")
]

for feature1, feature2 in important_pairs:
            if feature1 in data.columns and feature2 in data.columns:
                # Create interaction
interaction = data[feature1] * data[feature2]
interactions.append(interaction)

# Create ratio interaction
ratio_interaction = data[feature1] / (data[feature2] + 1e-8)
interactions.append(ratio_interaction)

# Create difference interaction
diff_interaction = data[feature1] - data[feature2]
interactions.append(diff_interaction)

if interactions:
            return pd.concat(interactions, axis=1, keys=[f"interaction_{i}" for i in range(len(interactions))])
else:
            return pd.DataFrame()

def _create_pattern_interactions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
Create pattern-based interactions using predefined patterns.
"""
interactions = []

for pattern_name, pattern_config in self.interaction_config.items():
            if not pattern_config["enabled"]:
                continue

pattern_features = pattern_config["features"]
weight = pattern_config["weight"]

# Find available features for this pattern
available_features = [f for f in pattern_features if f in data.columns]

if len(available_features) >= 2:
                pattern_data = data[available_features]

if pattern_name == "momentum_volume":
                    # Momentum × Volume interactions
momentum_features = [f for f in available_features if f in ["RSI", "MACD", "Stochastic"]]
volume_features = [f for f in available_features if "Volume" in f or "OBV" in f]

if momentum_features and volume_features:
                        momentum_avg = data[momentum_features].mean(axis=1)
volume_avg = data[volume_features].mean(axis=1)

interactions.extend([
momentum_avg * volume_avg * weight,
momentum_avg / (volume_avg + 1e-8) * weight,
momentum_avg.std(axis=1) * volume_avg * weight
])

elif pattern_name == "trend_volatility":
                    # Trend × Volatility interactions
trend_features = [f for f in available_features if "SMA" in f or "EMA" in f]
volatility_features = [f for f in available_features if "ATR" in f or "BB" in f or "Volatility" in f]

if trend_features and volatility_features:
                        trend_avg = data[trend_features].mean(axis=1)
volatility_avg = data[volatility_features].mean(axis=1)

interactions.extend([
trend_avg * volatility_avg * weight,
trend_avg / (volatility_avg + 1e-8) * weight,
np.abs(trend_avg) * volatility_avg * weight
])

elif pattern_name == "oscillator_trend":
                    # Oscillator × Trend interactions
oscillator_features = [f for f in available_features if f in ["RSI", "Williams_R", "CCI", "Stochastic"]]
trend_features = [f for f in available_features if "SMA" in f or "EMA" in f]

if oscillator_features and trend_features:
                        oscillator_avg = data[oscillator_features].mean(axis=1)
trend_avg = data[trend_features].mean(axis=1)

interactions.extend([
oscillator_avg * trend_avg * weight,
oscillator_avg / (trend_avg + 1e-8) * weight,
oscillator_avg.std(axis=1) * trend_avg * weight
])

if interactions:
            return pd.concat(interactions, axis=1, keys=[f"pattern_{i}" for i in range(len(interactions))])
else:
            return pd.DataFrame()

def _create_regime_interactions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
Create regime-dependent interactions.
"""
interactions = []

# Identify market regime based on volatility and trend
volatility = data.get("ATR_Normalized", data.get("Volatility", pd.Series(0.5, index=data.index)))
trend_strength = data.get("SMA_Ratio", pd.Series(1.0, index=data.index))

# Simple regime classification
regime = pd.Series("ranging", index=data.index)
regime[volatility > 0.03] = "volatile"
regime[abs(trend_strength - 1.0) > 0.02] = "trending"

# Create regime-specific interactions
for regime_type in ["trending", "ranging", "volatile"]:
            regime_mask = regime == regime_type

if regime_mask.sum() > 0:
                regime_data = data[regime_mask]

if regime_type == "trending":
                    # Trend-following interactions
trend_features = [f for f in data.columns if "SMA" in f or "EMA" in f or "MACD" in f]
momentum_features = [f for f in data.columns if f in ["RSI", "Stochastic", "CCI"]]

if trend_features and momentum_features:
                        trend_avg = regime_data[trend_features].mean(axis=1)
momentum_avg = regime_data[momentum_features].mean(axis=1)

interaction = trend_avg * momentum_avg * 1.5
interactions.append(interaction)

elif regime_type == "ranging":
                    # Range-trading interactions
oscillator_features = [f for f in data.columns if f in ["RSI", "Stochastic", "Williams_R", "CCI"]]
volume_features = [f for f in data.columns if "Volume" in f or "OBV" in f or "MFI" in f]

if oscillator_features and volume_features:
                        oscillator_avg = regime_data[oscillator_features].mean(axis=1)
volume_avg = regime_data[volume_features].mean(axis=1)

interaction = oscillator_avg * volume_avg * 1.6
interactions.append(interaction)

elif regime_type == "volatile":
                    # Volatility-focused interactions
volatility_features = [f for f in data.columns if "ATR" in f or "BB" in f or "Volatility" in f]
risk_features = [f for f in data.columns if f in ["RSI", "Stochastic", "Williams_R"]]

if volatility_features and risk_features:
                        volatility_avg = regime_data[volatility_features].mean(axis=1)
risk_avg = regime_data[risk_features].mean(axis=1)

interaction = volatility_avg * risk_avg * 1.8
interactions.append(interaction)

if interactions:
            return pd.concat(interactions, axis=1, keys=[f"regime_{i}" for i in range(len(interactions))])
else:
            return pd.DataFrame()

@handle_errors(exceptions=(Exception,), default_return=pd.DataFrame())
async def _select_optimal_interactions(
self,
interactions: pd.DataFrame,
target: pd.Series
) -> pd.DataFrame:
        """
Select optimal interactions based on importance and correlation.
"""
if interactions.empty:
            return pd.DataFrame()

try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
# Calculate mutual information
mi_scores = mutual_info_classif(interactions, target, random_state=42)

# Select interactions based on mutual information threshold
threshold = self.optimization_config.get("interaction_selection_threshold", 0.05)
important_indices = np.where(mi_scores > threshold)[0]

# Limit number of interactions
max_interactions = self.optimization_config.get("max_interactions", 50)
if len(important_indices) > max_interactions:
                # Select top interactions by mutual information
top_indices = np.argsort(mi_scores)[-max_interactions:]
selected_interactions = interactions.iloc[:, top_indices]
else:
                selected_interactions = interactions.iloc[:, important_indices]

return selected_interactions

except Exception as e:
            self.logger.error(f"Interaction selection failed: {e}")
return interactions.iloc[:, :min(50, interactions.shape[1])]  # Return first 50 interactions as fallback

@handle_errors(exceptions=(Exception,), default_return={})
async def _calculate_interaction_importance(
self,
interactions: pd.DataFrame,
target: pd.Series
) -> dict[str, Any]:
        """
Calculate importance of interaction features.
"""
if interactions.empty:
            return {}

try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
# Calculate mutual information for interaction importance
mi_scores = mutual_info_classif(interactions, target, random_state=42)

# Create importance dictionary
importance_dict = {
"mutual_information_scores": mi_scores.tolist(),
"mean_importance": float(np.mean(mi_scores)),
"max_importance": float(np.max(mi_scores)),
"min_importance": float(np.min(mi_scores)),
"std_importance": float(np.std(mi_scores)),
"top_interactions": []
}

# Get top interactions
top_indices = np.argsort(mi_scores)[-10:]  # Top 10
for idx in top_indices:
                importance_dict["top_interactions"].append({
"feature": interactions.columns[idx],
"importance": float(mi_scores[idx])
})

return importance_dict

except Exception as e:
            self.logger.error(f"Interaction importance calculation failed: {e}")
return {}

@handle_errors(exceptions=(Exception,), default_return={})
async def _evaluate_interaction_performance(
self,
interactions: pd.DataFrame,
target: pd.Series
) -> dict[str, Any]:
        """
Evaluate performance of interaction features.
"""
if interactions.empty:
            return {}

try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
# Combine original features with interactions
combined_features = pd.concat([interactions], axis=1)

# Train a simple model to evaluate performance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

model = RandomForestClassifier(n_estimators=50, random_state=42)
scores = cross_val_score(model, combined_features, target, cv=3, scoring='accuracy')

performance_metrics = {
"mean_accuracy": float(np.mean(scores)),
"std_accuracy": float(np.std(scores)),
"min_accuracy": float(np.min(scores)),
"max_accuracy": float(np.max(scores)),
"n_interactions": len(interactions.columns)
}

return performance_metrics

except Exception as e:
            self.logger.error(f"Interaction performance evaluation failed: {e}")
return {}

async def _save_optimization_results(
self,
results: dict[str, Any],
symbol: str,
exchange: str,
timeframe: str
):
        """Save optimization results to file."""

output_dir = Path("data/feature_engineering_optimization")
output_dir.mkdir(parents=True, exist_ok=True)

filename = f"{exchange}_{symbol}_{timeframe}_feature_optimization.json"
filepath = output_dir / filename

with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

self.logger.info(f"💾 Saved optimization results to {filepath}")

def get_optimized_parameters(
self,
symbol: str,
exchange: str,
timeframe: str
) -> dict[str, Any]:
        """Load optimized parameters for use in feature engineering."""

filepath = Path(f"data/feature_engineering_optimization/{exchange}_{symbol}_{timeframe}_feature_optimization.json")

if not filepath.exists():
            self.logger.warning(f"⚠️ No optimization results found for {symbol} on {exchange}")
return {}

try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
with open(filepath, 'r') as f:
                results = json.load(f)

return results.get("top_parameters", {})

except Exception as e:
            self.logger.error(f"❌ Error loading optimization results: {e}")
return {}