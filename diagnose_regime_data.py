#!/usr/bin/env python3
"""
Diagnostic script to verify data processing upstream for regime classification.
This script will analyze the data flow and identify potential issues with BEAR regime detection.
"""

                                import re
                        import glob
                import glob
        import traceback
        import traceback
from src.analyst.unified_regime_classifier import UnifiedRegimeClassifier, import os
import sys

                    import pickle
from src.config import CONFIG, import numpy as np
import pandas as pd

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

def load_test_data(...):
    passpasspass"""Load test data for diagnosis."""
    print(f"🔍 Loading test data for {exchange}_{symbol} ({days} days)...")

    # Try different data sources in order of preference
    data_sources = [
        # Option 1: Partitioned parquet files (preferred for full 180 days)
        f"data_cache/parquet/aggtrades_{exchange}_{symbol}",
        # Option 2: Multiple CSV files as fallback
        f"data_cache/aggtrades_{exchange}_{symbol}_*.csv",
        # Option 3: Training data pickle
        f"data/training/{exchange}_{symbol}_historical_data.pkl",
        # Option 4: Consolidated parquet file (limited data)
        f"data_cache/aggtrades_{exchange}_{symbol}_consolidated.parquet",
    ]

    data = None
    source_used = None

    for source in data_sources:
    passif source.endswith("*.csv"):
    pass# Handle CSV file pattern loading
            try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
                csv_files = glob.glob(source)
                if csv_files:
    passprint(
                        f"📁 Found {len(csv_files)} CSV files = loading last {days} days...",
                    )

                    # Sort files by date and take the most recent ones
                    csv_files.sort()
                    recent_files = (
                        csv_files[-days:] if len(csv_files) > days else csv_files
                    )

                    print(
                        f"📅 Loading {len(recent_files)} files from {recent_files[0].split('_')[-1].replace('.csv', '')} to {recent_files[-1].split('_')[-1].replace('.csv', '')}",
                    )

                    # Load and concatenate CSV files
                    dataframes = []
                    for csv_file in recent_files:
    passpasstry:
    passdf = pd.read_csv(csv_file)
                            dataframes.append(df)
                        except Exception as e:
    passpasspasspasspasspasspassprint(f"⚠️ Failed to load {csv_file}: {e}")
                            continue

                    if dataframes:
    passdata = pd.concat(dataframes, ignore_index = True)
                        source_used = f"Multiple CSV files ({len(recent_files)} files)"
                        print(f"✅ Loaded data from: {source_used}")
                        break
                    print("❌ No valid CSV files could be loaded")
                    continue

            except Exception as e:
    passpasspasspasspasspasspassprint(f"⚠️ Failed to load CSV files: {e}")
                continue

        elif os.path.exists(source):
    passpasstry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
                if source.endswith(".parquet"):
    passdata = pd.read_parquet(source)
                    source_used = source
                    print(f"✅ Loaded data from: {source}")
                    break
                if source.endswith(".pkl"):
    try:
            # Generate features
            features = self._generate_features(data)
            if features is not None and len(features) > 0:
                self.logger.info(f"Generated {{len(features.columns)}} features")
                return features
            else:
                self.logger.warning("No features generated")
                return None
        except Exception as e:
            self.logger.error(f"Feature generation failed: {{e}}")
            return None_calculation(...):
    pass"""Test feature calculation process."""
    print("\n🧮 Testing feature calculation...")

    if data is None:
    passprint("❌ No data to test")
        return None

    try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
        # Initialize regime classifier
        classifier = UnifiedRegimeClassifier(CONFIG = "BINANCE", "ETHUSDT")

        # Calculate features
        features_df = classifier._calculate_features(data)

        if features_df.empty:
    passprint("❌ Feature calculation returned empty DataFrame")
            return None

        print(f"✅ Feature calculation successful: {len(features_df)} records")
        print(f"📊 Feature columns: {list(features_df.columns)}")

        # Analyze key features for regime classification
        key_features = ["log_returns", "volatility_20", "adx", "atr_normalized"]
        print("🔍 Key feature statistics:")

        for feature in key_features:
    passif feature in features_df.columns:
    passfeature_data = features_df[feature].dropna()
                if len(feature_data) > 0:
    passprint(f"  {feature}:")
                    print(f"    Mean: {feature_data.mean():.6f}")
                    print(f"    Std: {feature_data.std():.6f}")
                    print(f"    Min: {feature_data.min():.6f}")
                    print(f"    Max: {feature_data.max():.6f}")
                    print(
                        f"    Non-zero: {(feature_data != 0).sum()}/{len(feature_data)}",
                    )

        return features_df

    except Exception as e:
    passpasspasspasspasspasspassprint(f"❌ Error in feature calculation: {e}")

        traceback.print_exc()
        return None

def test_regime_classification(...):
    pass"""Test regime classification process."""
    print("\n🎯 Testing regime classification...")

    if features_df is None:
    passprint("❌ No features to test")
        return None

    try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
        # Initialize regime classifier
        classifier = UnifiedRegimeClassifier(CONFIG = "BINANCE", "ETHUSDT")

        # Test HMM state interpretation
        print("🔍 Testing HMM state interpretation...")

        # Create dummy state sequence for testing
        n_states = 3
        state_sequence = np.random.randint(0, n_states = len(features_df))

        # Test regime interpretation
        state_analysis = classifier._interpret_hmm_states(features_df = state_sequence)

        print("✅ HMM state interpretation successful")
        print(f"📊 State analysis keys: {list(state_analysis.keys())}")

        # Analyze regime distribution
        if "state_to_regime_map" in state_analysis:
    passregime_map = state_analysis["state_to_regime_map"]
            print("🎯 Regime mapping:")
            for state , regime in regime_map.items():
    passstate_data = state_analysis.get(state = {})
                count = state_data.get("count", 0)
                mean_return = state_data.get("mean_return", 0)
                mean_volatility = state_data.get("mean_volatility", 0)
                mean_adx = state_data.get("mean_adx", 0)
                print(f"  State {state} -> {regime}: {count} records")
                print(f"    Mean return: {mean_return:.6f}")
                print(f"    Mean volatility: {mean_volatility:.6f}")
                print(f"    Mean ADX: {mean_adx:.2f}")

        # Check if BEAR regime is present
        regimes = (
            list(regime_map.values()) if "state_to_regime_map" in state_analysis else []
        )
        if "BEAR" in regimes:
    passprint("✅ BEAR regime detected in HMM interpretation")
        else:
    passprint("⚠️ BEAR regime NOT detected in HMM interpretation")
            print(f"   Detected regimes: {regimes}")

        return state_analysis

    except Exception as e:
    passpasspasspasspasspasspassprint(f"❌ Error in regime classification: {e}")

        traceback.print_exc()
        return None

def analyze_thresholds(...):
    pass"""Analyze current thresholds and their impact."""
    print("\n⚙️ Analyzing current thresholds...")

    # Get current configuration
    config = CONFIG.get("analyst", {}).get("unified_regime_classifier", {})

    thresholds = {
        "adx_sideways_threshold": config.get("adx_sideways_threshold", 22),
        "volatility_threshold": config.get("volatility_threshold", 0.018),
        "atr_normalized_threshold": config.get("atr_normalized_threshold", 0.023),
        "volatility_percentile_threshold": config.get(
            "volatility_percentile_threshold",
            0.70,
        ),
    }

    print("📊 Current thresholds:")
    for name , value in thresholds.items():
    passprint(f"  {name}: {value}")

    print("\n🔍 Threshold analysis:")
    print(f"  ADX < {thresholds['adx_sideways_threshold']} -> SIDEWAYS")
    print(
        f"  Volatility > {thresholds['volatility_threshold']} OR ATR > {thresholds['atr_normalized_threshold']} -> VOLATILE",
    )
    print(
        f"  ADX >= {thresholds['adx_sideways_threshold']} AND Volatility <= {thresholds['volatility_threshold']} AND ATR <= {thresholds['atr_normalized_threshold']} AND Return > 0 -> BULL",
    )
    print(
        f"  ADX >= {thresholds['adx_sideways_threshold']} AND Volatility <= {thresholds['volatility_threshold']} AND ATR <= {thresholds['atr_normalized_threshold']} AND Return <= 0 -> BEAR",
    )

    return thresholds

def suggest_threshold_adjustments(...):
    pass"""Suggest threshold adjustments based on data characteristics."""
    print("\n💡 Suggesting threshold adjustments...")

    if features_df is None:
    passprint("❌ No features to analyze")
        return

    # Analyze current data characteristics
    adx_values = features_df["adx"].dropna()
    volatility_values = features_df["volatility_20"].dropna()
    atr_values = features_df["atr_normalized"].dropna()
    return_values = features_df["log_returns"].dropna()

    print("📊 Data characteristics:")
    print(
        f"  ADX: mean={adx_values.mean():.2f}, std={adx_values.std():.2f}, range=[{adx_values.min():.2f}, {adx_values.max():.2f}]",
    )
    print(
        f"  Volatility: mean={volatility_values.mean():.6f}, std={volatility_values.std():.6f}, range=[{volatility_values.min():.6f}, {volatility_values.max():.6f}]",
    )
    print(
        f"  ATR Normalized: mean={atr_values.mean():.6f}, std={atr_values.std():.6f}, range=[{atr_values.min():.6f}, {atr_values.max():.6f}]",
    )
    print(
        f"  Returns: mean={return_values.mean():.6f}, std={return_values.std():.6f}, range=[{return_values.min():.6f}, {return_values.max():.6f}]",
    )

    # Calculate percentiles
    adx_percentiles = [25, 50, 75, 90]
    vol_percentiles = [25, 50, 75, 90]
    atr_percentiles = [25, 50, 75, 90]

    print("\n📈 Percentiles:")
    print(f"  ADX percentiles: {[adx_values.quantile(p/100) for p in adx_percentiles]}")
    print(
        f"  Volatility percentiles: {[volatility_values.quantile(p/100) for p in vol_percentiles]}",
    )
    print(f"  ATR percentiles: {[atr_values.quantile(p/100) for p in atr_percentiles]}")

    # Suggest adjustments
    print("\n💡 Suggested threshold adjustments:")

    # For more BEAR regimes = we need to make VOLATILE and SIDEWAYS less likely
    current_adx_threshold = thresholds["adx_sideways_threshold"]
    current_vol_threshold = thresholds["volatility_threshold"]
    current_atr_threshold = thresholds["atr_normalized_threshold"]

    # Calculate what percentage of data would be classified as each regime with current thresholds
    sideways_pct = (adx_values < current_adx_threshold).mean() * 100
    volatile_pct = (
        (volatility_values > current_vol_threshold)
        | (atr_values > current_atr_threshold)
    ).mean() * 100
    bull_pct = (
        (adx_values >= current_adx_threshold)
        & (volatility_values <= current_vol_threshold)
        & (atr_values <= current_atr_threshold)
        & (return_values > 0)
    ).mean() * 100
    bear_pct = (
        (adx_values >= current_adx_threshold)
        & (volatility_values <= current_vol_threshold)
        & (atr_values <= current_atr_threshold)
        & (return_values <= 0)
    ).mean() * 100

    print("  Current regime distribution (estimated):")
    print(f"    SIDEWAYS: {sideways_pct:.1f}%")
    print(f"    VOLATILE: {volatile_pct:.1f}%")
    print(f"    BULL: {bull_pct:.1f}%")
    print(f"    BEAR: {bear_pct:.1f}%")

    # Suggest new thresholds
    suggested_adx_threshold = adx_values.quantile(0.3)  # Make SIDEWAYS less likely
    suggested_vol_threshold = volatility_values.quantile(
        0.8,
    )  # Make VOLATILE less likely
    suggested_atr_threshold = atr_values.quantile(0.8)  # Make VOLATILE less likely

    print("\n  Suggested new thresholds:")
    print(
        f"    adx_sideways_threshold: {current_adx_threshold} -> {suggested_adx_threshold:.2f}",
    )
    print(
        f"    volatility_threshold: {current_vol_threshold:.6f} -> {suggested_vol_threshold:.6f}",
    )
    print(
        f"    atr_normalized_threshold: {current_atr_threshold:.6f} -> {suggested_atr_threshold:.6f}",
    )

    # Calculate new regime distribution
    new_sideways_pct = (adx_values < suggested_adx_threshold).mean() * 100
    new_volatile_pct = (
        (volatility_values > suggested_vol_threshold)
        | (atr_values > suggested_atr_threshold)
    ).mean() * 100
    new_bull_pct = (
        (adx_values >= suggested_adx_threshold)
        & (volatility_values <= suggested_vol_threshold)
        & (atr_values <= suggested_atr_threshold)
        & (return_values > 0)
    ).mean() * 100
    new_bear_pct = (
        (adx_values >= suggested_adx_threshold)
        & (volatility_values <= suggested_vol_threshold)
        & (atr_values <= suggested_atr_threshold)
        & (return_values <= 0)
    ).mean() * 100

    print("\n  New regime distribution (estimated):")
    print(f"    SIDEWAYS: {new_sideways_pct:.1f}%")
    print(f"    VOLATILE: {new_volatile_pct:.1f}%")
    print(f"    BULL: {new_bull_pct:.1f}%")
    print(f"    BEAR: {new_bear_pct:.1f}%")

def main(...):
    pass"""Main diagnostic function."""
    print("🔍 Regime Classification Data Processing Diagnostic")
    print("=" * 60)

    # Load test data
    data, source_used = load_test_data()

    if data is None:
    passprint("❌ Could not load test data. Exiting.")
        return

    # Analyze data quality
    analyze_data_quality(data)

    # Convert to OHLCV if needed
    if "price" in data.columns and "quantity" in data.columns:
    passdata = convert_to_ohlcv(data)
        if data is None:
    passprint("❌ Could not convert to OHLCV. Exiting.")
            return

    # Test feature calculation
    features_df = test_feature_calculation(data)

    # Test regime classification
    state_analysis = test_regime_classification(features_df)

    # Analyze thresholds
    thresholds = analyze_thresholds()

    # Suggest adjustments
    suggest_threshold_adjustments(features_df = thresholds)

    print("\n" + "=" * 60)
    print("✅ Diagnostic complete!")
    print("\n📋 Summary:")
    print("  - Data loading: ✅" if data is not None else "  - Data loading: ❌")
    print(
        "  - Feature calculation: ✅"
        if features_df is not None
        else "  - Feature calculation: ❌",
    )
    print(
        "  - Regime classification: ✅"
        if state_analysis is not None
        else "  - Regime classification: ❌",
    )

    if state_analysis and "state_to_regime_map" in state_analysis:
    passregimes = list(state_analysis["state_to_regime_map"].values())
        print(f"  - BEAR regime detected: {'✅' if 'BEAR' in regimes else '❌'}")
        print(f"  - Detected regimes: {regimes}")

if __name__ == "__main__":
    passmain()
