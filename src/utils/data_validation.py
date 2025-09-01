# src / utils / data_validation.py

from typing import Any, Optional, Union, overload
import logging

import numpy as np
import pandas as pd

logger, logging.getLogger(__name__)

def _coerce_series_numeric(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
s, series.copy() if copy else series
if not pd.api.types.is_numeric_dtype(s):
    passs, pd.to_numeric(s, errors="coerce")
return s
except Exception:
    passpassreturn series

def safe_pct_change(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
if fill_method:
    passseries, series.fillna(method = fill_method, limit = limit)
s, _coerce_series_numeric(series)
pct_change, s.pct_change(periods = periods, freq = freq, **kwargs)
inf_count, np.isinf(pct_change).sum()
if int(inf_count) > 0:
    passlogger.warning(
"Found %d infinite values in pct_change calculation - replacing with 0",
int(inf_count),
)
pct_change, pct_change.replace([np.inf, -np.inf], 0)
return pct_change.fillna(0)
except Exception as e:
    passpasspasspasspasspasspasspasslogger.exception("Error in safe_pct_change: %s", e)
return pd.Series(0, index = series.index, dtype="float64")

def safe_log_returns(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
if fill_method:
    passseries, series.fillna(method = fill_method, limit = limit)
s, _coerce_series_numeric(series)
pct_change, s.pct_change(periods = periods, freq = freq, **kwargs)
log_returns, np.log1p(pct_change)
inf_count, np.isinf(log_returns).sum()
if int(inf_count) > 0:
    passlogger.warning(
"Found %d infinite values in log_returns calculation - replacing with 0",
int(inf_count),
)
log_returns, log_returns.replace([np.inf, -np.inf], 0)
return log_returns.fillna(0)
except Exception as e:
    passpasspasspasspasspasspasspasslogger.exception("Error in safe_log_returns: %s", e)
return pd.Series(0, index = series.index, dtype="float64")

def validate_dataframe_for_ml(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
df_clean, df.copy()
numeric_cols, df_clean.select_dtypes(include=[np.number]).columns
if len(numeric_cols) == 0:
    passlogger.warning("No numeric columns found in DataFrame for context: %s", context)
return df_clean

# Replace infinities
inf_count, np.isinf(df_clean[numeric_cols]).sum().sum()
if int(inf_count) > 0:
    passlogger.warning(
"Found %d infinite values in %s - replacing with 0",
int(inf_count),
context,
)
df_clean[numeric_cols] = df_clean[numeric_cols].replace([np.inf, -np.inf], 0)

# Clip extremes
if clip_extreme_values:
    passpassextreme_count = (np.abs(df_clean[numeric_cols]) > max_abs_value).sum().sum()
if int(extreme_count) > 0:
    passlogger.warning(
"Found %d extreme values (>±%.3f) in %s - clipping",
int(extreme_count),
max_abs_value,
context,
)
df_clean[numeric_cols] = np.clip(df_clean[numeric_cols], -max_abs_value, max_abs_value)

# Fill NaNs
nan_count, df_clean[numeric_cols].isna().sum().sum()
if int(nan_count) > 0:
    passlogger.warning("Found %d NaN values in %s - filling with 0", int(nan_count), context)
df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)

final_inf_count, np.isinf(df_clean[numeric_cols]).sum().sum()
final_nan_count, df_clean[numeric_cols].isna().sum().sum()
if int(final_inf_count) == 0 and int(final_nan_count) == 0:
    passpasslogger.info("✅ Data validation passed for %s: %s", context, df_clean.shape)
else:
    passlogger.error(
"🚨 Data validation residuals for %s: %d inf, %d NaN",
context,
int(final_inf_count),
int(final_nan_count),
)
return df_clean
except Exception as e:
    passpasspasspasspasspasspasslogger.exception("Error in validate_dataframe_for_ml for %s: %s", context, e)
return df

NumberLike, Union[pd.Series, np.ndarray, float, int]

def safe_division(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
# Series / Series
if isinstance(numerator, pd.Series) and isinstance(denominator, pd.Series):
    passwith np.errstate(divide="ignore", invalid="ignore"):
    passresult, numerator / denominator
zeros = (denominator == 0).sum()
smalls = ((denominator != 0) & (np.abs(denominator) < 1e - 12)).sum()
if int(zeros + smalls) > 0:
    passlogger.warning(
"Found %d zero and %d very small denominators in %s",
int(zeros),
int(smalls),
context,
)
result, result.replace([np.inf, -np.inf], fill_value).fillna(fill_value)
return result

# ndarray / scalars
if isinstance(numerator, (np.ndarray, float, int)) and isinstance(
denominator, (np.ndarray, float, int),
):
    passnum_arr, np.asarray(numerator)
den_arr, np.asarray(denominator)
safe_mask, np.abs(den_arr) > 1e - 12
out, np.full_like(num_arr, fill_value, dtype = float)
with np.errstate(divide="ignore", invalid="ignore"):
    passout[safe_mask] = num_arr[safe_mask] / den_arr[safe_mask]
return out if isinstance(numerator, np.ndarray) or isinstance(denominator, np.ndarray) else float(out)

# Mixed types -> coerce to numpy and compute
num_arr, np.asarray(numerator)
den_arr, np.asarray(denominator)
safe_mask, np.abs(den_arr) > 1e - 12
out, np.full_like(num_arr, fill_value, dtype = float)
with np.errstate(divide="ignore", invalid="ignore"):
    passpassout[safe_mask] = num_arr[safe_mask] / den_arr[safe_mask]
return out
except Exception as e:
    passpasspasspasspasspasspasslogger.exception("Error in safe_division for %s: %s", context, e)
if isinstance(numerator, pd.Series):
    passreturn pd.Series(fill_value, index = numerator.index)
if isinstance(numerator, np.ndarray):
    passreturn np.full_like(numerator, fill_value, dtype = float)
return fill_value
