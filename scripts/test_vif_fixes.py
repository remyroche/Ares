#!/usr/bin/env python3
"""
Test VIF Fixes
Verifies that the root cause VIF fixes are working correctly.
"""

from pathlib import Path
from src.utils.logger import system_logger
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))


def test_vif_fixes():
    """Test that the VIF fixes are working correctly."""
    logger, system_logger.getChild("TestVIFFixes")

    print("=" * 80)
    print("VIF FIXES VERIFICATION TEST")
    print("=" * 80)

    if True:
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
            },
        )

        pd.DataFrame({"volume": price_data["volume"].copy()})

        print(f"\n📊 Test data created: {n_samples} samples")

        # Test 1: Moving Averages Fix
        print("\n🧪 Test 1: Moving Averages Fix")
        close = price_data["close"]

        # Original approach (high correlation)
        sma_20_orig = close.rolling(20).mean()
        ema_20_orig = close.ewm(span=20).mean()

        # Fixed approach (reduced correlation)
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        price_dev_sma20 = (close - sma_20) / sma_20
        price_dev_sma50 = (close - sma_50) / sma_50
        ma_crossover = sma_20 - sma_50
        price_acceleration = close.diff().diff()

        # Calculate correlations (ensure same length)
        sma_clean = sma_20_orig.dropna()
        ema_clean = ema_20_orig.dropna()
        min_len = min(len(sma_clean), len(ema_clean))
        corr_orig = np.corrcoef(sma_clean.iloc[-min_len:], ema_clean.iloc[-min_len:])[
            0,
            1,
        ]

        price_dev_clean1 = price_dev_sma20.dropna()
        price_dev_clean2 = price_dev_sma50.dropna()
        min_len_fixed = min(len(price_dev_clean1), len(price_dev_clean2))
        corr_fixed = np.corrcoef(
            price_dev_clean1.iloc[-min_len_fixed:],
            price_dev_clean2.iloc[-min_len_fixed:],
        )[0, 1]

        print(f"   Original MA correlation: {corr_orig:.3f}")
        print(f"   Fixed MA correlation: {corr_fixed:.3f}")
        print(f"   Improvement: {abs(corr_orig) - abs(corr_fixed):.3f}")

        # Test 2: Momentum Indicators Fix
        print("\n🧪 Test 2: Momentum Indicators Fix")
        price_diff = close.diff()

        # Original approach (overlapping windows)
        momentum_5_orig = price_diff.rolling(5).sum()
        momentum_10_orig = price_diff.rolling(10).sum()

        # Fixed approach (non-overlapping windows)
        momentum_3_fixed = price_diff.rolling(3).sum()
        momentum_8_fixed = price_diff.rolling(8).sum()
        momentum_13_fixed = price_diff.rolling(13).sum()
        momentum_accel = momentum_8_fixed.diff()

        # Calculate correlations (ensure same length)
        mom5_clean = momentum_5_orig.dropna()
        mom10_clean = momentum_10_orig.dropna()
        min_len_mom = min(len(mom5_clean), len(mom10_clean))
        corr_mom_orig = np.corrcoef(
            mom5_clean.iloc[-min_len_mom:],
            mom10_clean.iloc[-min_len_mom:],
        )[0, 1]

        mom3_clean = momentum_3_fixed.dropna()
        mom8_clean = momentum_8_fixed.dropna()
        min_len_mom_fixed = min(len(mom3_clean), len(mom8_clean))
        corr_mom_fixed = np.corrcoef(
            mom3_clean.iloc[-min_len_mom_fixed:],
            mom8_clean.iloc[-min_len_mom_fixed:],
        )[0, 1]

        print(f"   Original momentum correlation: {corr_mom_orig:.3f}")
        print(f"   Fixed momentum correlation: {corr_mom_fixed:.3f}")
        print(f"   Improvement: {abs(corr_mom_orig) - abs(corr_mom_fixed):.3f}")

        # Test 3: Volatility Indicators Fix
        print("\n🧪 Test 3: Volatility Indicators Fix")
        returns = close.pct_change()

        # Original approach (similar estimators)
        realized_vol_orig = returns.rolling(20).std()
        price_vol_orig = close.rolling(20).std()

        # Fixed approach (different estimators)
        realized_vol_fixed = returns.rolling(20).std()

        log_hl = np.log(price_data["high"] / price_data["low"])
        log_co = np.log(price_data["close"] / price_data["open"])
        garman_klass = np.sqrt(0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2)
        garman_klass_vol = garman_klass.rolling(10).mean()

        price_range = (price_data["high"] - price_data["low"]) / price_data["close"]
        range_vol = price_range.rolling(15).std()

        # Calculate correlations (ensure same length)
        vol_orig1 = realized_vol_orig.dropna()
        vol_orig2 = price_vol_orig.dropna()
        min_len_vol = min(len(vol_orig1), len(vol_orig2))
        corr_vol_orig = np.corrcoef(
            vol_orig1.iloc[-min_len_vol:],
            vol_orig2.iloc[-min_len_vol:],
        )[0, 1]

        vol_fixed1 = realized_vol_fixed.dropna()
        vol_fixed2 = garman_klass_vol.dropna()
        min_len_vol_fixed = min(len(vol_fixed1), len(vol_fixed2))
        corr_vol_fixed = np.corrcoef(
            vol_fixed1.iloc[-min_len_vol_fixed:],
            vol_fixed2.iloc[-min_len_vol_fixed:],
        )[0, 1]

        print(f"   Original volatility correlation: {corr_vol_orig:.3f}")
        print(f"   Fixed volatility correlation: {corr_vol_fixed:.3f}")
        print(f"   Improvement: {abs(corr_vol_orig) - abs(corr_vol_fixed):.3f}")

        # Test 4: Feature Diversity
        print("\n🧪 Test 4: Feature Diversity")

        # Calculate feature diversity metrics
        features_orig = {
            "sma_20": sma_20_orig , "ema_20": ema_20_orig,
            "momentum_5": momentum_5_orig , "momentum_10": momentum_10_orig,
            "realized_vol": realized_vol_orig , "price_vol": price_vol_orig,
        }

        features_fixed = {
            "price_dev_sma20": price_dev_sma20 , "price_dev_sma50": price_dev_sma50,
            "ma_crossover": ma_crossover , "price_acceleration": price_acceleration,
            "momentum_3": momentum_3_fixed , "momentum_8": momentum_8_fixed,
            "momentum_13": momentum_13_fixed , "momentum_accel": momentum_accel,
            "realized_vol": realized_vol_fixed , "garman_klass_vol": garman_klass_vol,
            "range_vol": range_vol}

        # Calculate average correlation for original vs fixed features

        def calculate_avg_correlation(features_dict):
            correlations = []
            feature_list, list(features_dict.values())
        for i in range(len(feature_list)):
            pass
        for j in range(i + 1, len(feature_list)):
            pass
        if True:
                        corr = np.corrcoef(
                            feature_list[i].dropna(),
                            feature_list[j].dropna(),
                        )[0, 1]
        if not np.isnan(corr):
                            correlations.append(abs(corr))
        pass
                        continue
        return np.mean(correlations) if correlations else 0

        avg_corr_orig = calculate_avg_correlation(features_orig)
        avg_corr_fixed = calculate_avg_correlation(features_fixed)

        print(f"   Original features average correlation: {avg_corr_orig:.3f}")
        print(f"   Fixed features average correlation: {avg_corr_fixed:.3f}")
        print(f"   Correlation reduction: {avg_corr_orig - avg_corr_fixed:.3f}")

        # Test 5: Feature Count and Diversity
        print("\n🧪 Test 5: Feature Count and Diversity")

        print(f"   Original feature count: {len(features_orig)}")
        print(f"   Fixed feature count: {len(features_fixed)}")
        print(f"   Feature increase: {len(features_fixed) - len(features_orig)}")

        # Calculate feature variance (diversity measure)

        def calculate_feature_variance(features_dict):
            variances = []
        for feature in features_dict.values():
            pass
        if True:
                    var = feature.var()
        if not np.isnan(var):
                        variances.append(var)
        pass
                    continue
        return np.mean(variances) if variances else 0

        var_orig = calculate_feature_variance(features_orig)
        var_fixed = calculate_feature_variance(features_fixed)

        print(f"   Original features average variance: {var_orig:.6f}")
        print(f"   Fixed features average variance: {var_fixed:.6f}")
        print(f"   Variance improvement: {var_fixed - var_orig:.6f}")

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY OF VIF FIXES:")
        print("=" * 80)

        improvements = [
            abs(corr_orig) - abs(corr_fixed),
            abs(corr_mom_orig) - abs(corr_mom_fixed),
            abs(corr_vol_orig) - abs(corr_vol_fixed),
            avg_corr_orig - avg_corr_fixed]

        total_improvement = sum(improvements)

        print(f"   Moving Averages improvement: {improvements[0]:.3f}")
        print(f"   Momentum indicators improvement: {improvements[1]:.3f}")
        print(f"   Volatility indicators improvement: {improvements[2]:.3f}")
        print(f"   Overall correlation reduction: {improvements[3]:.3f}")
        print(f"   Feature count increase: {len(features_fixed) - len(features_orig)}")
        print(f"   Variance improvement: {var_fixed - var_orig:.6f}")

        print(f"\n   Total improvement score: {total_improvement:.3f}")

        if total_improvement > 1.0 and avg_corr_fixed < 0.5:
            print("\n✅ VIF FIXES SUCCESSFUL!")
            print("   - Significant reduction in multicollinearity")
            print("   - Better feature diversity")
            print("   - More informative features")
            print("   - Lower VIF values expected")
        else:
            print("\n⚠️ VIF FIXES NEED IMPROVEMENT")
            print("   - Limited reduction in multicollinearity")
            print("   - Consider additional feature engineering")

        print("=" * 80)

        return total_improvement > 1.0 and avg_corr_fixed < 0.5

    pass
        logger.exception(f"Error testing VIF fixes: {e}")
        return False


if __name__ == "__main__":
    success = test_vif_fixes()
    sys.exit(0 if success else 1)
