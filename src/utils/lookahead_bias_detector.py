"""
Lookahead Bias Detection System

This module provides comprehensive detection and prevention of lookahead bias
in financial machine learning pipelines.
"""

import re
from typing import Any

import pandas as pd

from src.utils.logger import system_logger
from src.utils.error_handler import handle_data_processing_errors, handle_errors

class LookaheadBiasDetector:

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="lookaheadbiasdetector initialization",
    )
    async def initialize(self) -> bool:
        """Initialize LookaheadBiasDetector."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
    passself.logger.info("Implementation placeholder - needs specific logic")
class LookaheadBiasDetector:
    passself.logger.info("Implementation placeholder - needs specific logic")
class LookaheadBiasDetector:
    pass"""
Comprehensive lookahead bias detection and prevention system.

Detects various types of lookahead bias:
    1. Future information leakage in features
2. Improper temporal alignment
3. Incorrect train / test splits
4. Feature - target correlation issues
"""

def __init__(self, config: dict[str, Any] | None, None) -> None:
        self.config, config or {}
self.logger, system_logger.getChild("LookaheadBiasDetector")
self.detected_issues: list[str] = []
self.critical_issues: list[str] = []

# Configuration for detection strictness
self.strict_mode: bool, self.config.get("strict_mode", False)
self.warning_threshold: int, self.config.get(
"warning_threshold",
50,
)  # Max suspicious features before warning

@handle_data_processing_errors(default_return={}, context="LookaheadBiasDetector.detect_feature_lookahead_bias")
def detect_feature_lookahead_bias(...) -> ...:
    """..."""
    passresults: dict[str, Any] = {
"lookahead_bias_detected": False,
"critical_issues": [],
"warnings": [],
"feature_correlations": {},
"temporal_issues": [],
"recommendations": [],
"implementation_analysis": {},
}

try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
# 1. Check for perfect correlations (indicator of lookahead bias)
self._check_perfect_correlations(features_df, target_series, results)

# 2. Check temporal alignment if timestamps available
if timestamp_col and timestamp_col in features_df.columns:
    passpassself._check_temporal_alignment(
features_df,
target_series,
timestamp_col,
results,
)

# 3. Check for suspicious feature importance patterns
self._check_feature_importance_patterns(features_df, target_series, results)

# 4. Enhanced rolling window analysis
self._check_rolling_window_issues(features_df, results)

# 5. Analyze actual implementation if code provided
if feature_engineering_code:
    passpassself._analyze_implementation(
feature_engineering_code,
features_df,
results,
)

# 6. Generate recommendations
self._generate_recommendations(results)

# Log results
if results["critical_issues"]:
    passself.logger.critical(
f"🚨 LOOKAHEAD BIAS DETECTED: {len(results['critical_issues'])} critical issues",
)
for issue in results["critical_issues"]:
    passself.logger.critical(f"   ❌ {issue}")

if results["warnings"]:
    passself.logger.warning(
f"⚠️ LOOKAHEAD BIAS WARNINGS: {len(results['warnings'])} warnings",
)
for warning_msg in results["warnings"]:
    passself.logger.warning(f"   ⚠️ {warning_msg}")

return results

except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"Error in lookahead bias detection: {e}")
results["error"] = str(e)
return results

def _check_perfect_correlations(...) -> ...:
    """..."""
    pass# Calculate correlations with target
correlations: dict[str, float] = {}
for col in features_df.columns:
    passif col != target_series.name:
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
corr, float(features_df[col].corr(target_series))
if not pd.isna(corr):
    passcorrelations[col] = corr
except Exception:
    passpasscontinue

# Check for suspicious correlations
for feature, corr in correlations.items():
    passabs_corr, abs(corr)

if abs_corr > 0.98:
    passresults["critical_issues"].append(
f"PERFECT CORRELATION: {feature} has {corr:.4f} correlation with target "
f"(indicates lookahead bias)",
)
results["lookahead_bias_detected"] = True

elif abs_corr > 0.9:
    passpasspassresults["warnings"].append(
f"HIGH CORRELATION: {feature} has {corr:.4f} correlation with target "
f"(potential lookahead bias)",
)

elif abs_corr > 0.7:
    passpasspassresults["warnings"].append(
f"MODERATE CORRELATION: {feature} has {corr:.4f} correlation with target "
f"(investigate further)",
)

results["feature_correlations"] = correlations

def _check_temporal_alignment(...) -> ...:
    pass"""..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
# Ensure timestamps are datetime
timestamps, pd.to_datetime(features_df[timestamp_col])

# Check if features and target have same lengths
if len(features_df) != len(target_series):
    passresults["critical_issues"].append(
"TEMPORAL MISMATCH: Features and target have different lengths",
)
results["lookahead_bias_detected"] = True
return

# Check for future information leakage in rolling features
self._check_rolling_feature_timing(features_df, timestamps, results)
except Exception as e:
    passpasspasspasspasspasspasspassresults["warnings"].append(
f"Could not perform temporal alignment check: {e}",
)

def _check_rolling_feature_timing(...) -> ...:
    """..."""
    pass# Look for common rolling feature patterns
rolling_patterns = [
"volatility_",
"momentum_",
"rsi_",
"ma_",
"ema_",
"rolling_",
"std_",
"mean_",
"corr_",
]

suspicious_features: list[str] = []
for col in features_df.columns:
    passfor pattern in rolling_patterns:
    passif pattern in col.lower():
    passsuspicious_features.append(col)
break

if suspicious_features:
    passresults["warnings"].append(
f"ROLLING FEATURES DETECTED: {len(suspicious_features)} features may need lagging. "
f"Check: {suspicious_features[:5]}",
)

def _check_feature_importance_patterns(...) -> ...:
    """..."""
    pass# Calculate feature importance using correlation as proxy
correlations: dict[str, float] = results.get("feature_correlations", {})

if not correlations:
    passreturn

# Sort by absolute correlation
sorted_features, sorted(
correlations.items(),
key = lambda x: abs(x[1]),
reverse = True,
)

# Check for dominance by few features
if len(sorted_features) >= 2:
    passpasstop_corr, abs(sorted_features[0][1])
second_corr, abs(sorted_features[1][1])

# If top 2 features have very high correlations
if top_corr > 0.8 and second_corr > 0.8:
    passresults["critical_issues"].append(
f"FEATURE DOMINANCE: Top 2 features have correlations {top_corr:.4f} and {second_corr:.4f} "
f"({sorted_features[0][0]}, {sorted_features[1][0]}) - likely lookahead bias",
)
results["lookahead_bias_detected"] = True

# If top feature dominates
if top_corr > 0.9:
    passresults["critical_issues"].append(
f"FEATURE DOMINANCE: Top feature {sorted_features[0][0]} has correlation {top_corr:.4f} "
f"- likely lookahead bias",
)
results["lookahead_bias_detected"] = True

def _check_rolling_window_issues(...) -> ...:
    """..."""
    pass# Enhanced patterns for different types of features
rolling_patterns: dict[str, list[str]] = {
"volatility": ["volatility", "std", "atr"],
"momentum": ["momentum", "roc", "rsi", "stoch"],
"moving_averages": ["ma", "ema", "sma"],
"volume": ["volume", "obv", "vwap"],
"depth": ["depth", "spread", "bid", "ask"],
"technical": ["macd", "bb", "cci", "mfi", "williams"],
}

# Features that are inherently lagged by design
inherently_lagged_patterns = [
"slope",
"returns",
"change",
"diff",
"momentum",
"acceleration",
]

# Features that should be investigated but may be legitimate
suspicious_features: list[dict[str, Any]] = []
potentially_legitimate_features: list[dict[str, Any]] = []

# Enhanced legitimate patterns for common technical indicators
enhanced_legitimate_patterns = [
"lag",
"shift",
"prev",
"diff",
"change",
"slope",
"returns",
"pct_change",
"impact",
"correlation",
"spread",
"ratio",
"zscore",
"upper",
"lower",
"momentum",
"acceleration",
"volatility",
"std",
"atr",
"rsi",
"macd",
"bb_",
"sma_",
"ema_",
"vwap",
"obv",
"mfi",
"cci",
"williams",
"stoch",
]

for col in features_df.columns:
    passcol_lower, col.lower()

# Check if feature matches any rolling pattern
matches_rolling_pattern, False
pattern_category: str | None, None

for category, patterns in rolling_patterns.items():
    passif any(pattern in col_lower for pattern in patterns):
    passpassmatches_rolling_pattern, True
pattern_category, category
break

if not matches_rolling_pattern:
    passcontinue

# Check for legitimate lagging indicators (enhanced)
has_legitimate_lagging, any(
lag_pattern in col_lower for lag_pattern in enhanced_legitimate_patterns
)

# Check if feature is inherently lagged
is_inherently_lagged, any(
lag_pattern in col_lower for lag_pattern in inherently_lagged_patterns
)

# Additional checks for common legitimate patterns
is_common_technical_indicator, any(
[
"_" in col_lower
and col_lower.split("_")[0]
in ["sma", "ema", "bb", "rsi", "macd", "atr", "cci", "mfi"],
any(
pattern in col_lower
for pattern in [
"spread",
"ratio",
"zscore",
"impact",
"correlation",
]
),
col_lower.endswith(("_upper", "_lower", "_signal", "_histogram")),
any(
pattern in col_lower
for pattern in ["volatility", "momentum", "returns", "change"]
),
],
)

# Enhanced analysis based on feature type
if (
has_legitimate_lagging
or is_inherently_lagged
or is_common_technical_indicator
):
    passpass# This feature likely has proper lagging - add to potentially legitimate
potentially_legitimate_features.append(
{
"feature": col,
"category": pattern_category,
"lagging_type": self._identify_lagging_type(col_lower),
},
)
else:
    pass# This feature needs investigation
suspicious_features.append(
{
"feature": col,
"category": pattern_category,
"reason": "No obvious lagging pattern detected",
},
)

# Generate detailed analysis
if suspicious_features:
    pass# Only warn if we have too many suspicious features or in strict mode
if len(suspicious_features) > self.warning_threshold or self.strict_mode:
    pass# Group by category for better reporting
by_category: dict[str, list[str]] = {}
for item in suspicious_features:
    passcat, item["category"]
if cat not in by_category:
    passby_category[cat] = []
by_category[cat].append(item["feature"])

warning_msg, f"POTENTIAL LAGGING ISSUES: {len(suspicious_features)} features may need investigation:\n"
for category, features in by_category.items():
    passwarning_msg += f"   • {category.upper()}: {features[:3]}{'...' if len(features) > 3 else ''}\n"

results["warnings"].append(warning_msg.strip())

# Add to results for reference
results["suspicious_features"] = suspicious_features

if potentially_legitimate_features:
    passpass# Log legitimate features for transparency
self.logger.info(
f"✅ Found {len(potentially_legitimate_features)} features with legitimate lagging patterns",
)

# Add to results for reference
results["legitimate_features"] = potentially_legitimate_features

def _identify_lagging_type(...) -> ...:
    passpass"""..."""
    passfeature_lower, feature_name.lower()

if "diff" in feature_lower:
    pass# Try to extract the lag period
diff_match, re.search(r"diff_(\d+)", feature_lower)
if diff_match:
    passlag_period, diff_match.group(1)
return f"difference_lag_{lag_period}"
return "difference_lag_1"  # Default to 1 - period difference

if "lag" in feature_lower:
    passlag_match, re.search(r"lag_(\d+)", feature_lower)
if lag_match:
    passlag_period, lag_match.group(1)
return f"explicit_lag_{lag_period}"
return "explicit_lag_1"

if "shift" in feature_lower:
    passshift_match, re.search(r"shift_(\d+)", feature_lower)
if shift_match:
    passshift_period, shift_match.group(1)
return f"shift_{shift_period}"
return "shift_1"

if "returns" in feature_lower or "pct_change" in feature_lower:
    passreturn "percentage_change"

if "slope" in feature_lower:
    passreturn "slope_calculation"

if "change" in feature_lower:
    passreturn "change_calculation"

if "momentum" in feature_lower:
    passreturn "momentum_calculation"

return "unknown_lagging"

def _generate_recommendations(...) -> ...:
    """..."""
    passrecommendations: list[str] = []

if results["lookahead_bias_detected"]:
    passrecommendations.extend(
[
"🚨 CRITICAL: Stop using current models for live trading",
"🔧 Implement proper temporal alignment in feature engineering",
"📊 Re - train all models with corrected features",
"⏰ Use time - based train / test splits",
"🔍 Add lagging to all rolling window calculations",
],
)

# Enhanced recommendations based on analysis results
if "suspicious_features" in results:
    passpasspasssuspicious_count, len(results["suspicious_features"])
if suspicious_count > 0:
    passrecommendations.append(
f"🔍 Investigate {suspicious_count} features for proper lagging implementation",
)

# Group by category for specific recommendations
by_category: dict[str, list[str]] = {}
for item in results["suspicious_features"]:
    passcat, item["category"]
if cat not in by_category:
    passby_category[cat] = []
by_category[cat].append(item["feature"])

for category in by_category:
    passif category == "moving_averages":
    passrecommendations.append(
f"📈 For {category}: Ensure MA / EMA features use .diff() or .shift() operations",
)
elif category == "volatility":
    passpassrecommendations.append(
f"📊 For {category}: Verify volatility calculations use proper rolling windows",
)
elif category == "momentum":
    passpassrecommendations.append(
f"⚡ For {category}: Check momentum indicators for temporal alignment",
)

if "legitimate_features" in results:
    passpasslegitimate_count, len(results["legitimate_features"])
if legitimate_count > 0:
    passrecommendations.append(
f"✅ {legitimate_count} features have proper lagging patterns - good implementation",
)

# Implementation analysis recommendations
if "implementation_analysis" in results:
    passimpl_analysis, results["implementation_analysis"]

if "properly_lagged_features" in impl_analysis:
    passproper_count, len(impl_analysis["properly_lagged_features"])
if proper_count > 0:
    passrecommendations.append(
f"✅ Implementation analysis confirms {proper_count} features have proper lagging",
)

if "potentially_problematic_features" in impl_analysis:
    passproblematic_count, len(
impl_analysis["potentially_problematic_features"],
)
if problematic_count > 0:
    passrecommendations.append(
f"⚠️ {problematic_count} features may need implementation review",
)

# Correlation - based recommendations
if results["feature_correlations"]:
    passhigh_corr_features = [
feat
for feat, corr in results["feature_correlations"].items()
if abs(corr) > 0.8
]
if high_corr_features:
    passpassrecommendations.append(
f"📊 {len(high_corr_features)} features have high correlation (>0.8) - consider feature selection",
)

# General recommendations if no specific issues
if not results["critical_issues"] and not results["warnings"]:
    passrecommendations.append("✅ No obvious lookahead bias detected")
recommendations.append(
"🔍 Continue monitoring with enhanced detection system",
)
elif results["warnings"] and not results["critical_issues"]:
    passpasspassrecommendations.append(
"⚠️ Minor issues detected - review and address as needed",
)
recommendations.append(
"📈 Consider implementing automated lagging validation",
)

# Add best practices
recommendations.extend(
[
"💡 Best Practice: Use .diff(3) instead of .diff() for better feature independence",
"💡 Best Practice: Implement rolling windows with explicit lagging",
"💡 Best Practice: Validate temporal alignment in feature engineering pipeline",
"💡 Best Practice: Use time - based cross - validation for temporal data",
],
)

results["recommendations"] = recommendations

@handle_errors(default_return = None, context="LookaheadBiasDetector.validate_train_test_split")
def validate_train_test_split(...) -> ...:
    pass"""..."""
    passresults: dict[str, Any] = {"split_valid": True, "issues": [], "recommendations": []}

# Check if split is random (bad) or temporal (good)
if timestamp_col and timestamp_col in X_train.columns and timestamp_col in X_test.columns:
    passtrain_times, pd.to_datetime(X_train[timestamp_col])
test_times, pd.to_datetime(X_test[timestamp_col])

# Check for temporal ordering
if train_times.max() > test_times.min():
    passpassresults["split_valid"] = False
results["issues"].append(
"TEMPORAL LEAKAGE: Training data contains timestamps after test data",
)
results["recommendations"].append(
"Use time - based split: train on earlier data, test on later data",
)

# Check for data overlap
if len(set(X_train.index) & set(X_test.index)) > 0:
    passpassresults["split_valid"] = False
results["issues"].append(
"DATA OVERLAP: Training and test sets share data points",
)
results["recommendations"].append(
"Ensure complete separation between train and test sets",
)

return results

def add_lagging_to_features(...) -> ...:
    """..."""
    passlagged_features, features_df.copy()

# Apply lagging to all features
for col in lagged_features.columns:
    passif col not in ["timestamp", "time", "date"]:  # Skip timestamp columns
lagged_features[col] = lagged_features[col].shift(lag_periods)

# Fill NaN values created by lagging
lagged_features, lagged_features.fillna(method="bfill").fillna(0)

self.logger.info(
f"Applied {lag_periods}-period lagging to {len(features_df.columns)} features",
)

return lagged_features

def _analyze_implementation(...) -> ...:
    """..."""
    passimplementation_analysis: dict[str, Any] = {
"properly_lagged_features": [],
"potentially_problematic_features": [],
"lagging_patterns_found": [],
"recommendations": [],
}

# Common lagging patterns in code
lagging_patterns: dict[str, str] = {
"diff": r"\.diff\((\d+)\)",  # .diff(3)
"shift": r"\.shift\((\d+)\)",  # .shift(1)
"pct_change": r"\.pct_change\((\d+)\)",  # .pct_change(1)
"rolling_diff": r"\.rolling\(.*\)\.diff\((\d+)\)",  # .rolling(20).diff(1)
"ewm_diff": r"\.ewm\(.*\)\.diff\((\d+)\)",  # .ewm(span = 20).diff(1)
}

# Analyze code for lagging patterns
for pattern_name, pattern in lagging_patterns.items():
    passmatches, re.findall(pattern, feature_engineering_code)
if matches:
    passimplementation_analysis["lagging_patterns_found"].append(
{
"pattern": pattern_name,
"matches": matches,
"lag_periods": [
int(m) if str(m).isdigit() else 1 for m in matches
],
},
)

# Cross - reference with actual features
for col in features_df.columns:
    passpasspasscol_lower, col.lower()

# Check if this feature has corresponding lagging in code
feature_has_lagging, self._check_feature_lagging_in_code(
col_lower,
feature_engineering_code,
lagging_patterns,
)

if feature_has_lagging:
    passimplementation_analysis["properly_lagged_features"].append(
{"feature": col, "lagging_type": feature_has_lagging},
)
# Check if it's a base feature that doesn't need lagging
elif not self._is_base_feature(col):
    passpassimplementation_analysis["potentially_problematic_features"].append(
{
"feature": col,
"reason": "No lagging pattern found in implementation",
},
)

# Generate implementation - specific recommendations
if implementation_analysis["properly_lagged_features"]:
    passimplementation_analysis["recommendations"].append(
f"✅ Found {len(implementation_analysis['properly_lagged_features'])} features with proper lagging implementation",
)

if implementation_analysis["potentially_problematic_features"]:
    passpassimplementation_analysis["recommendations"].append(
f"⚠️ {len(implementation_analysis['potentially_problematic_features'])} features may need lagging implementation review",
)

# Update results
results["implementation_analysis"] = implementation_analysis

# Log findings
if implementation_analysis["properly_lagged_features"]:
    passself.logger.info(
f"✅ Implementation analysis: {len(implementation_analysis['properly_lagged_features'])} features have proper lagging",
)

def _check_feature_lagging_in_code(...) -> ...:
    """..."""
    passfeature_lower, feature_name.lower()

# Extract the base feature name (remove suffixes like _diff_1, _change, etc.)
base_feature, self._extract_base_feature_name(feature_lower)

# Look for the feature assignment in code
feature_patterns = [
rf'features\[["\']{re.escape(feature_lower)}["\']\]',
rf'features\[["\']{re.escape(base_feature)}["\']\]',
rf"features\[{re.escape(feature_lower)}\]",
rf"features\[{re.escape(base_feature)}\]",
]

for pattern in feature_patterns:
    passmatches, re.findall(pattern, code)
if matches:
    pass# Found the feature assignment, now check for lagging
for lag_type, lag_pattern in lagging_patterns.items():
    passif re.search(lag_pattern, code):
    passreturn lag_type

return None

def _extract_base_feature_name(...) -> ...:
    """..."""
    pass# Common suffixes to remove
suffixes = [
r"_diff_\d+",
r"_diff",
r"_change",
r"_returns",
r"_slope",
r"_lag_\d+",
r"_lag",
r"_shift_\d+",
r"_shift",
r"_pct_change",
r"_momentum",
r"_acceleration",
]

base_name, feature_name
for suffix in suffixes:
    passbase_name, re.sub(suffix, "", base_name)

return base_name

def _is_base_feature(...) -> ...:
    """..."""
    passbase_features = [
"open",
"high",
"low",
"close",
"volume",
"timestamp",
"ema_",
"sma_",
"rsi",
"macd",
"bb_",
"atr",
"stoch_",
"funding_rate",
"bid_ask_spread",
"market_depth",
]

feature_lower, feature_name.lower()
return any(base in feature_lower for base in base_features)

# Utility functions for easy integration

def detect_lookahead_bias(...) -> ...:
    pass"""..."""
    passdetector, LookaheadBiasDetector()
return detector.detect_feature_lookahead_bias(
features_df,
target_series,
timestamp_col,
)

def validate_temporal_split(...) -> ...:
    """..."""
    passdetector, LookaheadBiasDetector()
return detector.validate_train_test_split(
X_train,
X_test,
y_train,
y_test,
timestamp_col,
)

def apply_feature_lagging(...) -> ...:
    """..."""
    passdetector, LookaheadBiasDetector()
return detector.add_lagging_to_features(features_df, lag_periods)
