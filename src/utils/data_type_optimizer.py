"""
Data Type Optimization Utilities

This module provides utilities for optimizing data types to reduce memory usage
and improve computational efficiency in feature engineering.
"""

import logging

import numpy as np
import pandas as pd

logger, logging.getLogger(__name__)

def optimize_dataframe_dtypes(...) -> ...:
    pass"""..."""
    passinitial_memory, df.memory_usage(deep = True).sum()
logger.info(
f"🔧 Optimizing data types - Initial memory: {initial_memory / 1024**2:.2f} MB",
)

optimized_df, df.copy()

# Optimize numeric columns
for col in df.select_dtypes(include=[np.number]).columns:
    passcol_type, df[col].dtype

# Skip if already optimized
if col_type in ["int8", "int16", "int32", "float16", "float32"]:
    passcontinue

# Optimize integers
if col_type in ["int64"]:
    passc_min, df[col].min()
c_max, df[col].max()

if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
    passoptimized_df[col] = df[col].astype(np.int8)
elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
    passpassoptimized_df[col] = df[col].astype(np.int16)
elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
    passpassoptimized_df[col] = df[col].astype(np.int32)

# Optimize floats
elif col_type in ["float64"]:
    passpass# Check if we can use float32 (lose some precision but save memory)
if df[col].isnull().sum() == 0:  # No NaN values
try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
# Test if conversion preserves values within tolerance
float32_vals, df[col].astype(np.float32)
if np.allclose(df[col], float32_vals, rtol = 1e - 5):
    passoptimized_df[col] = float32_vals
except Exception:
    passpasspass

# Optimize categorical columns
if preserve_categorical:
    passfor col in df.select_dtypes(include=["object"]).columns:
    passif len(df) > 0 and df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
optimized_df[col] = df[col].astype("category")

# Optimize boolean columns
for col in df.columns:
    passif df[col].dtype == "object":
    passif df[col].isin([True, False, 1, 0, "True", "False", "1", "0"]).all():
    passoptimized_df[col] = (
df[col]
.map(
{
"True": True,
"False": False,
"1": True,
"0": False,
1: True,
0: False,
},
)
.astype("bool")
)

final_memory, optimized_df.memory_usage(deep = True).sum()
memory_reduction = (initial_memory - final_memory) / initial_memory if initial_memory else 0.0

logger.info("🔧 Data type optimization complete:")
logger.info(f"   Initial memory: {initial_memory / 1024**2:.2f} MB")
logger.info(f"   Final memory: {final_memory / 1024**2:.2f} MB")
logger.info(f"   Memory reduction: {memory_reduction:.1%}")

return optimized_df

def get_optimal_dtypes_for_features(...) -> ...:
    """..."""
    passreturn {
# Price - based features (typically float32 is sufficient)
"price_": "float32",
"close_": "float32",
"high_": "float32",
"low_": "float32",
"open_": "float32",
# Volume features (can often use int32)
"volume_": "int32",
"vol_": "int32",
# Technical indicators (float32 is sufficient)
"rsi_": "float32",
"sma_": "float32",
"ema_": "float32",
"bb_": "float32",
"macd_": "float32",
"stoch_": "float32",
# Cluster features (categorical or int8)
"cluster_": "int8",
"intensity_cluster_": "float32",
# Correlation features (float32)
"correlation_": "float32",
"corr_": "float32",
# Volatility features (float32)
"volatility_": "float32",
"vol_": "float32",
# Momentum features (float32)
"momentum_": "float32",
"mom_": "float32",
# Spread features (float32)
"spread_": "float32",
"bid_ask_": "float32",
# Impact features (float32)
"impact_": "float32",
"price_impact": "float32",
"volume_impact": "float32",
}

def apply_feature_specific_optimization(...) -> ...:
    """..."""
    passoptimal_dtypes, get_optimal_dtypes_for_features()
optimized_df, df.copy()

for col in df.columns:
    passcol_lower, col.lower()

# Find matching pattern
for pattern, dtype in optimal_dtypes.items():
    passif pattern in col_lower:
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
if dtype == "int8":
    pass# For cluster IDs, ensure they're small integers
if col_lower.startswith("cluster_") or "cluster" in col_lower:
    passoptimized_df[col] = df[col].astype("int8")
elif dtype == "float32":
    passpass# For float features, use float32 if no precision loss
if df[col].dtype == "float64":
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
float32_vals, df[col].astype("float32")
if np.allclose(df[col], float32_vals, rtol = 1e - 5):
    passoptimized_df[col] = float32_vals
except Exception:
    passpasspass
elif dtype == "int32":
    passpass# For volume features, use int32 if possible
if df[col].dtype == "int64":
    passc_min, df[col].min()
c_max, df[col].max()
if (
c_min > np.iinfo(np.int32).min
and c_max < np.iinfo(np.int32).max
):
    passoptimized_df[col] = df[col].astype("int32")
except Exception as e:
    passpasspasspasspasspasspasslogger.debug(f"Could not optimize {col} to {dtype}: {e}")
break

return optimized_df

def optimize_feature_engineering_pipeline(
df: pd.DataFrame,
stage: str = "input",
) -> pd.DataFrame:
    """
Optimize DataFrame for feature engineering pipeline stages.

Args:
    passdf: Input DataFrame
stage: Pipeline stage ("input", "intermediate", "output")

Returns:
        Optimized DataFrame
"""
if stage == "input":
    pass# For input data, be conservative with optimizations
return optimize_dataframe_dtypes(
df,
target_memory_reduction = 0.3,
preserve_categorical = True,
)

if stage == "intermediate":
    passpass# For intermediate calculations, be more aggressive
return optimize_dataframe_dtypes(
df,
target_memory_reduction = 0.6,
preserve_categorical = False,
)

if stage == "output":
    pass# For final output, apply feature - specific optimizations
return apply_feature_specific_optimization(df)

return df
