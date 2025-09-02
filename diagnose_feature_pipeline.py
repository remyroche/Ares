#!/usr/bin/env python3
"""
Diagnostic script to understand the feature pipeline issue.
This script helps identify why only cluster features are reaching the autoencoder.
"""

import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger=logging.getLogger(__name__)


def analyze_feature_pipeline_issue():
    """
    Analyze the feature pipeline issue where only cluster features reach the autoencoder.
    """
    logger.info("🔍 Analyzing feature pipeline issue...")

    print("=" * 80)
    print("FEATURE PIPELINE DIAGNOSTIC REPORT")
    print("=" * 80)

    print("\n📋 ISSUE SUMMARY:")
    print(
        "   ❌ Autoencoder receiving only cluster features (intensity_cluster_*, hmm_composite_cluster_id)",
    )
    print("   ❌ 0 price features found by PriceReturnConverter")
    print("   ❌ 0 volume features found by PriceReturnConverter")
    print("   ❌ Low variance in cluster features (expected behavior)")

    print("\n🔍 ROOT CAUSE ANALYSIS:")
    print(
        "   1. Autoencoder is designed to work with ENGINEERED features=not raw OHLCV data",
    )
    print(
        "   2. Raw OHLCV columns (open=high, low=close, volume) are correctly filtered out",
    )
    print(
        "   3. Expected: Technical indicators=momentum features, volatility features=etc.",
    )
    print("   4. Actual: Only cluster features are being passed to autoencoder")
    print("   5. This indicates a data pipeline issue upstream")

    print("\n📊 EXPECTED FEATURE TYPES:")
    print("   ✅ Technical Indicators: RSI=MACD, Bollinger Bands=ATR, etc.")
    print("   ✅ Momentum Features: Price momentum=volume momentum, etc.")
    print("   ✅ Volatility Features: Realized volatility=volatility regimes, etc.")
    print("   ✅ Order Flow Features: OBV=volume ratios, etc.")
    print("   ✅ Time-based Features: Hour sin/cos=day of week, etc.")
    print(
        "   ❌ Cluster Features: intensity_cluster_*, hmm_composite_cluster_id (should be minimal)",
    )

    print("\n🚨 PROBLEM IDENTIFICATION:")
    print(
        "   The feature engineering pipeline is not generating the expected engineered features.",
    )
    print("   Only cluster features are reaching the autoencoder=which causes:")
    print("   - 0 price features found (no raw OHLCV data)")
    print("   - 0 volume features found (no raw OHLCV data)")
    print("   - Low variance in cluster features (categorical nature)")
    print("   - Autoencoder skipping enhancement (correct behavior)")

    print("\n💡 RECOMMENDATIONS:")
    print("   1. Check the feature engineering pipeline upstream")
    print("   2. Verify that technical indicators are being calculated")
    print("   3. Ensure momentum and volatility features are being generated")
    print("   4. Review the data flow from feature engineering to autoencoder")
    print("   5. Check if feature engineering components are properly initialized")

    print("\n🔧 DEBUGGING STEPS:")
    print("   1. Add logging to feature engineering pipeline")
    print("   2. Check what features are being generated before autoencoder")
    print("   3. Verify feature engineering configuration")
    print("   4. Test feature engineering components individually")
    print("   5. Check if raw OHLCV data is available for feature engineering")

    print("\n📝 NEXT STEPS:")
    print("   1. Run the enhanced autoencoder with new validation logging")
    print("   2. Check logs for detailed feature analysis")
    print("   3. Investigate feature engineering pipeline")
    print("   4. Ensure proper feature engineering before autoencoder step")

    print("\n" + "=" * 80)


def check_expected_features():
    """
    List the expected features that should be available for autoencoder training.
    """
    print("\n📋 EXPECTED FEATURE CATEGORIES:")

    expected_features={
        "Technical Indicators": [
            "RSI_14",
            "MACD_12_26_9",
            "MACDs_12_26_9",
            "MACDh_12_26_9",
            "BBU_20_2.0",
            "BBM_20_2.0",
            "BBL_20_2.0",
            "BBW_20_2.0",
            "BBP_20_2.0",
            "STOCHk_14_3_3",
            "STOCHd_14_3_3",
            "ATR_14",
            "ADX_14",
            "OBV",
            "VWAP",
            "SMA_9",
            "SMA_21",
            "SMA_50",
            "EMA_12",
            "EMA_26",
            "CCI_14",
            "MFI_14",
            "ROC_10",
            "Williams_R_14",
            "Parabolic_SAR",
            "SuperTrend_10_2.0",
            "DCU_20",
            "DCL_20",
            "DCM_20",
            "ATRr_14",
        ],
        "Momentum Features": [
            "Price_Momentum",
            "Price_Acceleration",
            "Volume_Momentum",
            "Volume_Acceleration",
            "Volatility_Momentum",
            "momentum_5",
            "momentum_10",
        ],
        "Volatility Features": [
            "Realized_Volatility",
            "Volatility_Regime_Numeric",
            "volatility",
        ],
        "Order Flow Features": [
            "VROC",
            "OBV_Divergence",
            "Buy_Sell_Pressure_Ratio",
            "Order_Flow_Imbalance",
            "Large_Order_Count",
            "Liquidity_Score",
        ],
        "Time-based Features": [
            "Hour_Sin",
            "Hour_Cos",
            "DayOfWeek_Sin",
            "DayOfWeek_Cos",
        ],
        "Funding Features": [
            "Funding_Momentum",
            "Funding_Divergence",
            "Funding_Extreme",
        ],
        "Divergence Features": ["RSI_MACD_Divergence", "Volume_Price_Divergence"],
        "Support/Resistance Features": [
            "distance_to_sr",
            "sr_strength",
            "sr_type",
            "price_position",
        ],
    }

    for category, features in expected_features.items():
        print(f"   {category}: {len(features)} features")
        if len(features) <= 8:
            print(f"      {', '.join(features)}")
        else:
            print(f"      {', '.join(features[:8])}... (and {len(features)-8} more)")

    total_expected=sum(len(features) for features in expected_features.values())
    print(f"\n   Total expected engineered features: {total_expected}")


def main():
    """
    Main function to run the diagnostic analysis.
    """
    logger.info("🚀 Starting feature pipeline diagnostic...")

    analyze_feature_pipeline_issue()
    check_expected_features()

    logger.info("✅ Feature pipeline diagnostic complete!")
    logger.info("📝 Review the output above to understand the issue and next steps.")


if __name__== "__main__":
    main()
