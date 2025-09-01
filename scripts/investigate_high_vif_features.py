#!/usr/bin/env python3
"""
Investigate High VIF Features
Analyzes and fixes features with high Variance Inflation Factor (VIF) values.
"""

from pathlib import Path
from typing import Dict, List
from src.utils.logger import setup_logging, system_logger
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))


def analyze_high_vif_features() -> bool:
    pass
    pass
	"""Analyze the high VIF features and propose fixes."""
	setup_logging()
	system_logger.getChild("HighVIFAnalysis")

	print("=" * 80)
	print("HIGH VIF FEATURES ANALYSIS & FIXES")
	print("=" * 80)

	# High VIF features from the logs
	high_vif_features: Dict[str, Dict[str, float | str]] = {
		"sma_20": {
			"vif": 183.11,
			"issue": "Very high multicollinearity",
			"fix": "Use price differences instead of raw prices",
		},
		"nearest_support_distance": {
			"vif": 96.54,
			"issue": "High multicollinearity",
			"fix": "Already fixed with dynamic S/R counts",
		},
		"morl_log_returns_energy_ts_norm": {
			"vif": 64.58,
			"issue": "High multicollinearity",
			"fix": "Use different wavelet scales or remove redundant",
		},
		"1m_price_volatility": {
			"vif": 51.68,
			"issue": "High multicollinearity",
			"fix": "Use different volatility estimators",
		},
		"ema20_slope": {
			"vif": 48.17,
			"issue": "Moderate multicollinearity",
			"fix": "Use price differences or different window",
		},
		"cmor1.5-1.0_log_returns_energy_ts_norm": {
			"vif": 43.68,
			"issue": "Moderate multicollinearity",
			"fix": "Use different wavelet type",
		},
		"realized_volatility": {
			"vif": 40.35,
			"issue": "Moderate multicollinearity",
			"fix": "Use different volatility estimator",
		},
		"momentum_10": {
			"vif": 39.17,
			"issue": "Moderate multicollinearity",
			"fix": "Use different momentum calculation",
		},
		"momentum_5": {
			"vif": 37.71,
			"issue": "Moderate multicollinearity",
			"fix": "Use different momentum calculation",
		},
		"roc_5": {
			"vif": 33.92,
			"issue": "Moderate multicollinearity",
			"fix": "Use different rate of change calculation",
		},
		"roc_10": {
			"vif": 32.75,
			"issue": "Moderate multicollinearity",
			"fix": "Use different rate of change calculation",
		},
		"volatility_percentile": {
			"vif": 32.36,
			"issue": "Moderate multicollinearity",
			"fix": "Use different percentile calculation",
		},
		"wavelet_denoised_signal_ts": {
			"vif": 30.90,
			"issue": "Moderate multicollinearity",
			"fix": "Use different wavelet parameters",
		},
	}

	print("\\\n📊 HIGH VIF FEATURES ANALYSIS:")
	print("-" * 60)

	for feature, info in high_vif_features.items():
    pass
    pass
		print(f"   {feature}: VIF = {float(info['vif']):.2f}")
        print(f"      Issue: {info['issue']}")
        print(f"      Fix: {info['fix']}")
		print()

	# Group features by type for systematic fixes
	feature_groups: Dict[str, List[str]] = {
		"Moving Averages": ["sma_20", "ema20_slope"],
		"Momentum Indicators": ["momentum_5", "momentum_10", "roc_5", "roc_10"],
		"Volatility Indicators": [
			"1m_price_volatility",
			"realized_volatility",
			"volatility_percentile",
		],
		"Wavelet Features": [
			"morl_log_returns_energy_ts_norm",
			"cmor1.5-1.0_log_returns_energy_ts_norm",
			"wavelet_denoised_signal_ts",
		],
		"Support/Resistance": ["nearest_support_distance"],
	}

	print("\\\n🔧 SYSTEMATIC FIXES BY FEATURE GROUP:")
	print("-" * 60)

	for group, features in feature_groups.items():
    pass
    pass
		print(f"\\\n📋 {group}:")
		for feature in features:
    pass
    pass
			if feature in high_vif_features:
    pass
    pass
				print(f"   - {feature}: {high_vif_features[feature]['fix']}")

	# Proposed fixes
	fixes = {
		"Moving Averages": {
			"problem": "SMA and EMA are highly correlated with price",
			"solution": "Use price differences, ratios, or different windows",
			"implementation": [
				"Replace sma_20 with (close - sma_20) / sma_20",
				"Replace ema20_slope with price acceleration",
				"Use different window sizes (10, 50 instead of 20)",
			],
		},
		"Momentum Indicators": {
			"problem": "Multiple momentum indicators use similar calculations",
			"solution": "Diversify momentum calculations",
			"implementation": [
				"Use exponential momentum instead of simple",
				"Use different time windows (3, 7, 15 instead of 5, 10)",
				"Use momentum acceleration (second derivative)",
			],
		},
		"Volatility Indicators": {
			"problem": "Multiple volatility estimators are highly correlated",
			"solution": "Use different volatility estimators",
			"implementation": [
				"Use Garman-Klass instead of realized volatility",
				"Use Parkinson volatility for high-frequency data",
				"Use different window sizes for different estimators",
			],
		},
		"Wavelet Features": {
			"problem": "Multiple wavelet features use similar parameters",
			"solution": "Diversify wavelet parameters and types",
			"implementation": [
				"Use different wavelet types (db4, coif4, sym4)",
				"Use different scales (2, 4, 8 instead of 1.5)",
				"Use different decomposition levels",
			],
		},
	}

	print("\\\n💡 DETAILED FIX PROPOSALS:")
	print("-" * 60)

	for group, fix_info in fixes.items():
    pass
    pass
		print(f"\\\n🎯 {group}:")
		print(f"   Problem: {fix_info['problem']}")
		print(f"   Solution: {fix_info['solution']}")
		print("   Implementation:")
		for impl in fix_info["implementation"]:
    pass
    pass
            print(f"     - {impl}")

	# Test the fixes
	print("\\\n🧪 TESTING PROPOSED FIXES:")
	print("-" * 60)

	# Create sample data
	np.random.seed(42)
	n_samples = 1000

	# Simulate price data
	price_data = pd.DataFrame(
		{
			"close": np.cumsum(np.random.randn(n_samples) * 0.01) + 100,
			"high": np.cumsum(np.random.randn(n_samples) * 0.01) + 100.5,
			"low": np.cumsum(np.random.randn(n_samples) * 0.01) + 99.5,
			"open": np.cumsum(np.random.randn(n_samples) * 0.01) + 100,
			"volume": np.random.lognormal(10, 1, n_samples),
		}
	)

	# 1. Moving Averages Fix
	print("\\\n📊 Moving Averages Fix Test:")
	close = price_data["close"]

	# Original features
	sma_20_orig = close.rolling(20).mean()
	ema_20_orig = close.ewm(span=20).mean()

	# Fixed features
	sma_20_fixed = (close - sma_20_orig) / sma_20_orig  # Price deviation from MA
	ema_20_fixed = close.diff().ewm(span=20).mean()  # Price acceleration

	# Calculate correlation (ensure same length)
	sma_clean = sma_20_orig.dropna()
	ema_clean = ema_20_orig.dropna()
	min_len = min(len(sma_clean), len(ema_clean))
	corr_orig = float(np.corrcoef(sma_clean.iloc[-min_len:], ema_clean.iloc[-min_len:])[0, 1])

	sma_fixed_clean = sma_20_fixed.dropna()
	ema_fixed_clean = ema_20_fixed.dropna()
	min_len_fixed = min(len(sma_fixed_clean), len(ema_fixed_clean))
	corr_fixed = float(
		np.corrcoef(
			sma_fixed_clean.iloc[-min_len_fixed:],
			ema_fixed_clean.iloc[-min_len_fixed:],
		)[0, 1]
	)

	print(f"   Original correlation: {corr_orig:.3f}")
	print(f"   Fixed correlation: {corr_fixed:.3f}")
	print(f"   Improvement: {abs(corr_orig) - abs(corr_fixed):.3f}")

	test_results: Dict[str, Dict[str, float]] = {}
	test_results["Moving Averages"] = {
		"original_corr": corr_orig,
		"fixed_corr": corr_fixed,
		"improvement": abs(corr_orig) - abs(corr_fixed),
	}

	# 2. Momentum Indicators Fix
	print("\\\n📊 Momentum Indicators Fix Test:")

	# Original features
	momentum_5_orig = close.pct_change(5)
	momentum_10_orig = close.pct_change(10)

	# Fixed features
	momentum_3_fixed = close.pct_change(3)  # Different window
	momentum_7_fixed = close.pct_change(7)  # Different window
	accel_5 = momentum_5_orig.diff()  # Momentum acceleration (not used in corr)
	_ = accel_5  # keep variable to avoid linter warning

	# Calculate correlations (ensure same length)
	mom5_clean = momentum_5_orig.dropna()
	mom10_clean = momentum_10_orig.dropna()
	min_len_mom = min(len(mom5_clean), len(mom10_clean))
	corr_mom_orig = float(
		np.corrcoef(
			mom5_clean.iloc[-min_len_mom:],
			mom10_clean.iloc[-min_len_mom:],
		)[0, 1]
	)

	mom3_clean = momentum_3_fixed.dropna()
	mom7_clean = momentum_7_fixed.dropna()
	min_len_mom_fixed = min(len(mom3_clean), len(mom7_clean))
	corr_mom_fixed = float(
		np.corrcoef(
			mom3_clean.iloc[-min_len_mom_fixed:],
			mom7_clean.iloc[-min_len_mom_fixed:],
		)[0, 1]
	)

	print(f"   Original momentum correlation: {corr_mom_orig:.3f}")
	print(f"   Fixed momentum correlation: {corr_mom_fixed:.3f}")
	print(f"   Improvement: {abs(corr_mom_orig) - abs(corr_mom_fixed):.3f}")

	test_results["Momentum Indicators"] = {
		"original_corr": corr_mom_orig,
		"fixed_corr": corr_mom_fixed,
		"improvement": abs(corr_mom_orig) - abs(corr_mom_fixed),
	}

	# 3. Volatility Indicators Fix
	print("\\\n📊 Volatility Indicators Fix Test:")

	# Original features
	realized_vol_orig = close.pct_change().rolling(20).std()
	price_vol_orig = close.rolling(20).std()

	# Fixed features (different estimators)
	garman_klass = np.sqrt(
		0.5 * np.log(price_data["high"] / price_data["low"]) ** 2
		- (2 * np.log(2) - 1) * np.log(price_data["close"] / price_data["open"]) ** 2,
	)
	garman_klass = garman_klass.rolling(20).mean()

	parkinson = np.sqrt(
		np.log(price_data["high"] / price_data["low"]) ** 2 / (4 * np.log(2)),
	)
	parkinson = parkinson.rolling(20).mean()

	# Calculate correlations
	corr_vol_orig = float(
		np.corrcoef(realized_vol_orig.dropna(), price_vol_orig.dropna())[0, 1]
	)
	corr_vol_fixed = float(np.corrcoef(garman_klass.dropna(), parkinson.dropna())[0, 1])

	print(f"   Original volatility correlation: {corr_vol_orig:.3f}")
	print(f"   Fixed volatility correlation: {corr_vol_fixed:.3f}")
	print(f"   Improvement: {abs(corr_vol_orig) - abs(corr_vol_fixed):.3f}")

	test_results["Volatility Indicators"] = {
		"original_corr": corr_vol_orig,
		"fixed_corr": corr_vol_fixed,
		"improvement": abs(corr_vol_orig) - abs(corr_vol_fixed),
	}

	# Summary
	print("\\\n" + "=" * 80)
	print("SUMMARY OF FIXES:")
	print("=" * 80)

	total_improvement = 0.0
	for group, results in test_results.items():
    pass
    pass
		improvement = float(results["improvement"])  # ensure numeric
		total_improvement += improvement
		print(f"   {group}: {improvement:.3f} correlation reduction")

	print(f"\\\n   Total improvement: {total_improvement:.3f} correlation reduction")

	if total_improvement > 0.5:
    pass
    pass
		print("\\\n✅ FIXES SUCCESSFUL!")
		print("   - Significant reduction in multicollinearity")
		print("   - Better feature diversity")
		print("   - Lower VIF values expected")
	else:
		print("\\\n⚠️ FIXES NEED IMPROVEMENT")
		print("   - Limited reduction in multicollinearity")
		print("   - Consider more aggressive feature selection")

	print("=" * 80)

	return bool(total_improvement > 0.5)


if __name__ == "__main__":
    pass
    pass
	success = analyze_high_vif_features()
	sys.exit(0 if success else 1)
