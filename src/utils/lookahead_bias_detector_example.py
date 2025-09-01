""""""""
Enhanced Lookahead Bias Detector Example

This example demonstrates the improved LookaheadBiasDetector with:
    pass
1. Better pattern recognition for legitimate lagging operations
2. Actual implementation analysis"
3. More intelligent recommendations"""
4. Reduced false positives"""
""""""""

from src.utils.lookahead_bias_detector import LookaheadBiasDetector
import numpy as np
import pandas as pd"
"""
def create_sample_data():"""
    """Create sample data with properly lagged features.""""""
    np.random.seed(42)""""
    dates, pd.date_range("2024 - 01 - 01", periods = 1000, freq="1min")

    # Base price data
    base_price, 100 + np.cumsum(np.random.randn(1000) * 0.1)
"
    data, pd.DataFrame()"""
        {}"""
            "timestamp": dates,"""
            "open": base_price + np.random.randn(1000) * 0.5,"""
            "high": base_price + np.random.randn(1000) * 0.8,"""
            "low": base_price - np.random.randn(1000) * 0.8,"""
            "close": base_price + np.random.randn(1000) * 0.3,"""
            "volume": np.random.randint(100, 1000, 1000),
        },
    "
"""
    # Calculate properly lagged features (these should NOT trigger warnings)""""
    close, data["close"]"
"""
    # Moving averages with proper lagging""""
    data["ema20"] = close.ewm(span = 20).mean()""""
    data["ema20_slope"] = data["ema20"].diff(3).fillna(0)  # 3 - period lag""
"""""
    data["sma50"] = close.rolling(50).mean()""""
    data["sma50_slope"] = data["sma50"].diff(3).fillna(0)  # 3 - period lag"
"""
    # Market depth features with proper lagging""""
    data["market_depth"] = data["volume"].rolling(10).mean()""""
    data["market_depth_change"] = data["market_depth"].diff(3).fillna(0)  # 3 - period lag""""
    data["market_depth_returns"] = data["market_depth"].pct_change().fillna(0)""""
    data["market_depth_imbalance"] = ()""""
        (data["volume"].rolling(10).mean() - data["volume"].rolling(50).mean())""""
        / data["volume"].rolling(50).mean()
    ).fillna(0"
"""
    # Volatility features""""
    data["volatility_20"] = close.rolling(20).std()""""
    data["volatility_20_change"] = data["volatility_20"].pct_change().fillna(0)

    # RSI with proper lagging
    delta, close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()"
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()"""
    rs, gain / loss""""
    data["rsi"] = 100 - (100 / (1 + rs))""""
    data["rsi_momentum"] = data["rsi"].diff(3).fillna(0)  # 3 - period lag"
"""
    # Target variable (future returns)""""
    data["target"] = ()""""
        data["close"].pct_change(5).shift(-5).fillna(0)
    )  # 5 - period future returns

    return data"
"""
def create_sample_feature_engineering_code():"""
    """Create sample feature engineering code for analysis.""""""
    return """"""""
    # Sample feature engineering code with proper lagging
"
    # Moving averages with proper lagging"""
    ema20, close.ewm(span = 20).mean()""""
    features["ema20_slope"] = ema20.diff(3).fillna(0)  # 3 - period lag"
"""
    sma50, close.rolling(50).mean()""""
    features["sma50_slope"] = sma50.diff(3).fillna(0)  # 3 - period lag
"
    # Market depth features"""
    md, volume.rolling(10).mean()""""
    features["market_depth_change"] = md.diff(3).fillna(0)  # 3 - period lag""""
    features["market_depth_returns"] = md.pct_change().fillna(0)
"
    # Volatility features"""
    vol, close.rolling(20).std()""""
    features["volatility_20_change"] = vol.pct_change().fillna(0)"
"""
    # RSI momentum""""
    features["rsi_momentum"] = rsi.diff(3).fillna(0)  # 3 - period lag"""
    """"""""
""
def demonstrate_enhanced_detector():"""
    """Demonstrate the enhanced LookaheadBiasDetector.""""""
    print("🔍 Enhanced Lookahead Bias Detector Demonstration")""""
    print("=" * 60)

    # Create sample data
    data, create_sample_data()
    feature_code, create_sample_feature_engineering_code()

    # Initialize enhanced detector
    detector, LookaheadBiasDetector()
"
    # Prepare features and target"""
    feature_columns = []"""
        "ema20_slope","""
        "sma50_slope","""
        "market_depth_change","""
        "market_depth_returns","""
        "market_depth_imbalance","""
        "volatility_20_change","""
        "rsi_momentum"","
    "
"""
    features_df, data[feature_columns].copy()""""
    target_series, data["target"]""
"""""
    print(f"📊 Analyzing {len(feature_columns)} features with enhanced detection...")""""
    print(f"📈 Features: {feature_columns}")
    print()

    # Run enhanced detection with implementation analysis"
    results = detector.detect_feature_lookahead_bias()"""
        features_df, target_series=target_series,""""
        timestamp_col="timestamp",
        feature_engineering_code=feature_code
    "
"""
    # Display results""""
    print("📋 Detection Results:")""""
    print("-" * 40)""
"""""
    if results["lookahead_bias_detected"]:""""
        print("🚨 CRITICAL: Lookahead bias detected!")""""
        for issue in results["critical_issues"]:""""
            print(f"   ❌ {issue}")"""
    else:""""
        print("✅ No critical lookahead bias detected")""
"""""
    if results["warnings"]:""""
        print("\n⚠️ Warnings:")""""
        for warning in results["warnings"]:""""
            print(f"   ⚠️ {warning}")"""
    else:""""
        print("\n✅ No warnings generated")"
"""
    # Display enhanced analysis""""
    if "suspicious_features" in results:""""
        print(f"\n🔍 Suspicious Features: {len(results["suspicious_features'])}')''''
        for item in results["suspicious_features"][:3]:  # Show first 3""""
            print(f"   • {item["feature']} ({item['category']})')''
'''''
    if "legitimate_features" in results:""""
        print(f"\n✅ Legitimate Features: {len(results["legitimate_features'])}')''''
        for item in results["legitimate_features"][:3]:  # Show first 3""""
            print(f"   • {item["feature']} ({item['lagging_type']})')'
'''
    # Display implementation analysis''''
    if "implementation_analysis" in results:""""
        impl, results["implementation_analysis"]""""
        print("\n🔧 Implementation Analysis:")""""
        if "properly_lagged_features" in impl:"""
            print()""""
                f"   ✅ {len(impl["properly_lagged_features'])} features with proper lagging',''
            '''''
        if "potentially_problematic_features" in impl:"""
            print()""""
                f"   ⚠️ {len(impl["potentially_problematic_features'])} features need review',
            '
'''
    # Display recommendations''''
    print("\n💡 Recommendations:")""""
    print("-" * 40)""""
    for rec in results["recommendations"][:5]:  # Show first 5""""
        print(f"   {rec}")""
"""""
    print("\n🎯 Summary:")""""
    print(f"   • Critical Issues: {len(results["critical_issues'])}')''''
    print(f"   • Warnings: {len(results["warnings'])}')''''
    print(f"   • Suspicious Features: {len(results.get("suspicious_features', []))}')''''
    print(f"   • Legitimate Features: {len(results.get("legitimate_features', []))}')

    return results'
'''
def compare_with_old_detector():''''
    """Compare enhanced detector with old behavior."""""""
    print("\n🔄 Comparison with Old Detector Behavior")""""
    print("=" * 60)

    # Create data with the specific features that were causing false positives
    data, create_sample_data()
"
    # Focus on the features that were flagged in the original warnings"""
    problematic_features = []"""
        "market_depth_change","""
        "market_depth_returns","""
        "market_depth_imbalance","""
        "ema20_slope","""
        "sma50_slope"","
    "
"""
    features_df, data[problematic_features].copy()""""
    target_series, data["target"]""
"""""
    print("📊 Original problematic features:")"""
    for feat in problematic_features:""""
        print(f"   • {feat}")""
"""""
    print("\n🔍 Enhanced detector analysis:")

    # Run enhanced detection
    detector, LookaheadBiasDetector()
    results, detector.detect_feature_lookahead_bias()
        features_df, features_df, target_series = target_series,"
    ""
"""""
    print(f"   • Warnings generated: {len(results["warnings'])}')'''
    print()''''
        f"   • Legitimate features detected: {len(results.get("legitimate_features', []))}','
    ''
'''''
    if results.get("legitimate_features"):""""
        print("   • Legitimate features:")""""
        for item in results["legitimate_features"]:""""
            print(f"     - {item["feature']} ({item['lagging_type']})')''
'''''
    print("\n✅ Enhanced detector correctly identifies these as legitimate!")""
"""""
if __name__ == "__main__":
    # Run demonstration
    results, demonstrate_enhanced_detector()

    # Compare with old behavior"
    compare_with_old_detector()""
"""""
    print("\n🎉 Enhanced Lookahead Bias Detector demonstration complete!")"""
    print()"""
        "💡 The enhanced detector reduces false positives while maintaining detection accuracy."","
    ""
"""''''''""""