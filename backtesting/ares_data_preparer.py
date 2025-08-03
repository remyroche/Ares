# backtesting/ares_data_preparer.py
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# We assume a config file exists at `src/config.py` that defines a CONFIG dictionary.
# And that data_utils contains the necessary data loading functions.
try:
    from src.analyst.data_utils import (
        load_agg_trades_data,
        load_futures_data,
        load_klines_data,
    )
    from src.config import CONFIG
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure src/config.py and src/analyst/data_utils.py are available.")

# --- DEBUGGING FLAG ---
DEBUG_MODE = True


def validate_data_quality(df: pd.DataFrame, data_type: str) -> tuple[bool, str]:
    """
    Comprehensive data quality validation.
    
    Args:
        df: DataFrame to validate
        data_type: Type of data ('klines', 'agg_trades', 'futures')
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df.empty:
        return False, f"{data_type} DataFrame is empty"
    
    # Check for required columns based on data type
    if data_type == "klines":
        required_cols = ["open", "high", "low", "close", "volume"]
        price_cols = ["open", "high", "low", "close"]
    elif data_type == "agg_trades":
        required_cols = ["price", "quantity"]
        price_cols = ["price"]
    elif data_type == "futures":
        required_cols = ["fundingRate"]
        price_cols = []
    else:
        return False, f"Unknown data type: {data_type}"
    
    # Check for required columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"
    
    # Check for NaN values in critical columns
    nan_counts = df[required_cols].isnull().sum()
    total_nan = nan_counts.sum()
    if total_nan > 0:
        return False, f"Found {total_nan} NaN values in {data_type} data: {nan_counts.to_dict()}"
    
    # Check for infinite values
    inf_counts = np.isinf(df[required_cols].select_dtypes(include=[np.number])).sum()
    total_inf = inf_counts.sum()
    if total_inf > 0:
        return False, f"Found {total_inf} infinite values in {data_type} data: {inf_counts.to_dict()}"
    
    # Check for negative values in price columns
    for col in price_cols:
        if col in df.columns:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                return False, f"Found {negative_count} negative values in {col}"
    
    # Check for zero values in critical columns
    for col in required_cols:
        if col in df.columns:
            zero_count = (df[col] == 0).sum()
            if zero_count > len(df) * 0.5:  # More than 50% zeros
                return False, f"Found {zero_count} zero values in {col} ({zero_count/len(df)*100:.1f}%)"
    
    # Check for reasonable data ranges
    if data_type == "klines":
        # Check that high >= low
        invalid_hl = (df["high"] < df["low"]).sum()
        if invalid_hl > 0:
            return False, f"Found {invalid_hl} rows where high < low"
        
        # Check that open and close are within high-low range
        invalid_oc = ((df["open"] > df["high"]) | (df["open"] < df["low"]) | 
                     (df["close"] > df["high"]) | (df["close"] < df["low"])).sum()
        if invalid_oc > 0:
            return False, f"Found {invalid_oc} rows where open/close outside high-low range"
    
    # Check for duplicate timestamps
    if df.index.duplicated().any():
        return False, f"Found {df.index.duplicated().sum()} duplicate timestamps"
    
    # Check for reasonable data volume
    if len(df) < 100:
        return False, f"Insufficient data: only {len(df)} rows"
    
    return True, "Data quality validation passed"


def clean_data_only(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    """
    Clean data without filling missing values - only remove obvious issues.
    
    Args:
        df: DataFrame to clean
        data_type: Type of data
        
    Returns:
        Cleaned DataFrame
    """
    if df.empty:
        return df
    
    print(f"🧹 Cleaning {data_type} data...")
    initial_shape = df.shape
    
    # Remove duplicate timestamps
    df = df[~df.index.duplicated(keep="first")]
    
    # Replace infinite values with NaN (don't fill, just mark as invalid)
    df = df.replace([np.inf, -np.inf], np.nan)
    
    final_shape = df.shape
    removed_rows = initial_shape[0] - final_shape[0]
    
    if removed_rows > 0:
        print(f"   Removed {removed_rows} rows with duplicate timestamps or infinite values")
    
    print(f"   ✅ {data_type} cleaning completed: {initial_shape} → {final_shape}")
    
    return df


def load_raw_data(symbol: str, exchange: str):
    """Loads raw data from CSV files, running the downloader if they are missing."""
    print(f"--- Step 1: Loading Raw Data for {symbol} on {exchange} ---")

    # Dynamically generate filenames to be exchange and symbol-specific
    lookback_years = CONFIG.get("lookback_years", 2)
    interval = CONFIG.get("trading_interval", "15m")
    
    # Expected filenames (what the system is looking for)
    klines_filename = (
        f"data_cache/{exchange}_{symbol}_{interval}_{lookback_years}y_klines.csv"
    )
    agg_trades_filename = (
        f"data_cache/{exchange}_{symbol}_{lookback_years}y_aggtrades.csv"
    )
    futures_filename = (
        f"data_cache/{exchange}_{symbol}_futures_{lookback_years}y_data.csv"
    )

    # Alternative filename patterns that might exist
    alt_klines_patterns = [
        f"data_cache/klines_{exchange}_{symbol}_{interval}_*.csv",
        f"data_cache/{exchange}_{symbol}_klines_*.csv",
        f"data_cache/klines_{symbol}_{interval}_*.csv"
    ]
    
    alt_agg_trades_patterns = [
        f"data_cache/aggtrades_{exchange}_{symbol}_*.csv",
        f"data_cache/{exchange}_{symbol}_aggtrades_*.csv",
        f"data_cache/aggtrades_{symbol}_*.csv"
    ]
    
    alt_futures_patterns = [
        f"data_cache/futures_{exchange}_{symbol}_*.csv",
        f"data_cache/{exchange}_{symbol}_futures_*.csv",
        f"data_cache/futures_{symbol}_*.csv"
    ]

    def find_alternative_file(patterns):
        """Find the most recent file matching any of the patterns."""
        import glob
        for pattern in patterns:
            files = glob.glob(pattern)
            if files:
                # Sort by modification time and return the most recent
                files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                return files[0]
        return None

    # Check if expected files exist, if not try alternative patterns
    if not os.path.exists(klines_filename):
        alt_klines = find_alternative_file(alt_klines_patterns)
        if alt_klines:
            print(f"Found alternative klines file: {alt_klines}")
            klines_filename = alt_klines
        else:
            print(f"Warning: No klines file found for {symbol} on {exchange}")

    if not os.path.exists(agg_trades_filename):
        alt_agg_trades = find_alternative_file(alt_agg_trades_patterns)
        if alt_agg_trades:
            print(f"Found alternative agg_trades file: {alt_agg_trades}")
            agg_trades_filename = alt_agg_trades
        else:
            print(f"Warning: No agg_trades file found for {symbol} on {exchange}")

    if not os.path.exists(futures_filename):
        alt_futures = find_alternative_file(alt_futures_patterns)
        if alt_futures:
            print(f"Found alternative futures file: {alt_futures}")
            futures_filename = alt_futures
        else:
            print(f"Warning: No futures file found for {symbol} on {exchange}")

    # Check if we have at least some data files
    required_files = [klines_filename, agg_trades_filename, futures_filename]
    existing_files = [f for f in required_files if os.path.exists(f)]
    
    if len(existing_files) == 0:
        print(f"No data files found for {symbol} on {exchange}")
        print("Attempting to download data...")
        
        # The downloader script is now modularized in training steps, but we can keep this for standalone runs.
        downloader_script_name = CONFIG.get(
            "downloader_script_name",
            "src/training/steps/data_downloader.py",
        )

        try:
            # Pass symbol and exchange to the downloader script
            subprocess.run(
                [
                    sys.executable,
                    downloader_script_name,
                    "--symbol",
                    symbol,
                    "--exchange",
                    exchange,
                ],
                check=True,
            )
            
            # After download, try to find files again
            if not os.path.exists(klines_filename):
                alt_klines = find_alternative_file(alt_klines_patterns)
                if alt_klines:
                    klines_filename = alt_klines
                    
            if not os.path.exists(agg_trades_filename):
                alt_agg_trades = find_alternative_file(alt_agg_trades_patterns)
                if alt_agg_trades:
                    agg_trades_filename = alt_agg_trades
                    
            if not os.path.exists(futures_filename):
                alt_futures = find_alternative_file(alt_futures_patterns)
                if alt_futures:
                    futures_filename = alt_futures
                    
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            print(f"ERROR during download: {e}")
            print("Continuing with available data...")

    try:
        # Load data with enhanced error handling and quality validation
        klines_df = pd.DataFrame()
        agg_trades_df = pd.DataFrame()
        futures_df = pd.DataFrame()
        
        if os.path.exists(klines_filename):
            print(f"Loading klines from: {klines_filename}")
            klines_df = load_klines_data(klines_filename)
            if not klines_df.empty:
                # Clean data (remove duplicates, infinite values)
                klines_df = clean_data_only(klines_df, "klines")
                # Validate data quality - FAIL FAST if issues found
                is_valid, error_msg = validate_data_quality(klines_df, "klines")
                if not is_valid:
                    print(f"❌ CRITICAL: Klines data quality issues: {error_msg}")
                    print("Please fix the data quality issues before proceeding.")
                    sys.exit(1)
            else:
                print(f"❌ CRITICAL: Empty klines data from {klines_filename}")
                sys.exit(1)
        else:
            print(f"❌ CRITICAL: Klines file not found: {klines_filename}")
            sys.exit(1)
            
        if os.path.exists(agg_trades_filename):
            print(f"Loading agg_trades from: {agg_trades_filename}")
            agg_trades_df = load_agg_trades_data(agg_trades_filename)
            if not agg_trades_df.empty:
                # Clean data (remove duplicates, infinite values)
                agg_trades_df = clean_data_only(agg_trades_df, "agg_trades")
                # Validate data quality - FAIL FAST if issues found
                is_valid, error_msg = validate_data_quality(agg_trades_df, "agg_trades")
                if not is_valid:
                    print(f"❌ CRITICAL: Agg_trades data quality issues: {error_msg}")
                    print("Please fix the data quality issues before proceeding.")
                    sys.exit(1)
            else:
                print(f"❌ CRITICAL: Empty agg_trades data from {agg_trades_filename}")
                sys.exit(1)
        else:
            print(f"❌ CRITICAL: Agg_trades file not found: {agg_trades_filename}")
            sys.exit(1)
            
        if os.path.exists(futures_filename):
            print(f"Loading futures from: {futures_filename}")
            futures_df = load_futures_data(futures_filename)
            if not futures_df.empty:
                # Clean data (remove duplicates, infinite values)
                futures_df = clean_data_only(futures_df, "futures")
                # Validate data quality - FAIL FAST if issues found
                is_valid, error_msg = validate_data_quality(futures_df, "futures")
                if not is_valid:
                    print(f"❌ CRITICAL: Futures data quality issues: {error_msg}")
                    print("Please fix the data quality issues before proceeding.")
                    sys.exit(1)
            else:
                print(f"⚠️ Warning: Empty futures data from {futures_filename}")
        else:
            print(f"⚠️ Warning: Futures file not found: {futures_filename}")

        # Ensure indices are unique
        if not klines_df.empty:
            klines_df = klines_df[~klines_df.index.duplicated(keep="first")]
        if not agg_trades_df.empty:
            agg_trades_df = agg_trades_df[~agg_trades_df.index.duplicated(keep="first")]

        print(
            f"✅ Successfully loaded high-quality data: {len(klines_df)} k-lines, {len(agg_trades_df)} agg trades, and {len(futures_df)} futures data points.\n",
        )
        return klines_df, agg_trades_df, futures_df
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not load data files: {e}")
        print("Please check that data files exist and are not corrupted.")
        sys.exit(1)


def get_sr_levels(df_daily):
    """Identifies Support/Resistance levels from daily aggregated data."""
    print("--- Step 2: Identifying S/R Levels ---")
    recent_daily = df_daily.tail(365)
    price_bins = pd.cut(recent_daily["close"], bins=100)
    volume_profile = recent_daily.groupby(price_bins, observed=False)["volume"].sum()
    hvn_levels = volume_profile.nlargest(5).index.map(lambda x: x.mid).tolist()
    poc = volume_profile.idxmax().mid
    high_peaks, _ = find_peaks(recent_daily["high"], distance=15)
    low_peaks, _ = find_peaks(-recent_daily["low"], distance=15)
    pivots = (
        recent_daily.iloc[high_peaks]["high"].tolist()
        + recent_daily.iloc[low_peaks]["low"].tolist()
    )
    levels = pd.DataFrame({"price": hvn_levels + [poc] + pivots}).drop_duplicates()
    print("S/R Level identification complete.\n")
    return levels["price"].tolist()


def calculate_features_and_score(
    klines_df,
    agg_trades_df,
    futures_df,
    params,
    sr_levels,
):
    """
    Calculates all feature sub-scores and combines them into a final Confidence Score.
    """
    print("--- Step 3: Calculating Features and Confidence Score ---")

    # Ensure we have valid data before processing
    if klines_df.empty or agg_trades_df.empty:
        print("ERROR: Empty klines or agg_trades data")
        return pd.DataFrame()

    # Clean input data - remove any rows with NaN in critical columns
    klines_df = klines_df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    agg_trades_df = agg_trades_df.dropna(subset=['price', 'quantity'])

    if klines_df.empty or agg_trades_df.empty:
        print("ERROR: No valid data after cleaning")
        return pd.DataFrame()

    klines_df = pd.merge_asof(
        klines_df.sort_index(),
        futures_df.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )
    
    # Fill NaN values in futures data with forward fill, then backward fill
    klines_df.fillna(method="ffill", inplace=True)
    klines_df.fillna(method="bfill", inplace=True)
    
    # For any remaining NaN in funding rate, use 0
    if 'fundingRate' in klines_df.columns:
        klines_df['fundingRate'] = klines_df['fundingRate'].fillna(0)

    # --- 1. Calculate Sub-Scores ---

    # Sentiment Score
    if 'fundingRate' in klines_df.columns and not klines_df['fundingRate'].isna().all():
        fr_mean = klines_df["fundingRate"].mean()
        fr_std = klines_df["fundingRate"].std()
        if fr_std > 0:
            fr_score = (klines_df["fundingRate"] - fr_mean) / fr_std
        else:
            fr_score = pd.Series(0, index=klines_df.index)
        klines_df["sentiment_score"] = np.clip(fr_score, -1, 1).fillna(0)
    else:
        klines_df["sentiment_score"] = 0

    # Trend Score
    try:
        klines_df.ta.adx(
            length=params["adx_period"],
            append=True,
            col_names=("ADX", "DMP", "DMN"),
        )
        # Fill NaN values in ADX columns
        klines_df["ADX"] = klines_df["ADX"].fillna(0)
        klines_df["DMP"] = klines_df["DMP"].fillna(0)
        klines_df["DMN"] = klines_df["DMN"].fillna(0)
    except Exception as e:
        print(f"Warning: Could not calculate ADX: {e}")
        klines_df["ADX"] = 0
        klines_df["DMP"] = 0
        klines_df["DMN"] = 0

    try:
        macd = klines_df.ta.macd(append=False)
        klines_df["MACD_HIST"] = macd["MACDh_12_26_9"].fillna(0)
    except Exception as e:
        print(f"Warning: Could not calculate MACD: {e}")
        klines_df["MACD_HIST"] = 0

    # Calculate normalized histogram with safe division
    close_prices = klines_df["close"].replace(0, np.nan)
    if not close_prices.isna().all():
        normalized_hist = klines_df["MACD_HIST"] / close_prices
        normalized_hist = normalized_hist.fillna(0)
    else:
        normalized_hist = pd.Series(0, index=klines_df.index)

    direction_score = np.clip(normalized_hist * params["scaling_factor"], -1, 1)
    
    # Calculate momentum score with safe division
    trend_threshold = params["trend_threshold"]
    max_strength_threshold = params["max_strength_threshold"]
    threshold_diff = max_strength_threshold - trend_threshold
    
    if threshold_diff > 0:
        momentum_score = np.clip(
            (klines_df["ADX"] - trend_threshold) / threshold_diff,
            0,
            1,
        )
    else:
        momentum_score = pd.Series(0, index=klines_df.index)
    
    klines_df["trend_score"] = (direction_score * momentum_score).fillna(0)

    # Mean Reversion Score
    try:
        klines_df.ta.atr(length=params["atr_period"], append=True, col_names=("ATR"))
        klines_df["ATR"] = klines_df["ATR"].fillna(0)
    except Exception as e:
        print(f"Warning: Could not calculate ATR: {e}")
        klines_df["ATR"] = 0

    proximity_threshold = klines_df["ATR"] * params["proximity_multiplier"]
    is_interacting = pd.Series(False, index=klines_df.index)
    for level in sr_levels:
        is_interacting |= (klines_df["low"] <= level + proximity_threshold) & (
            klines_df["high"] >= level - proximity_threshold
        )
    klines_df["Is_Interacting"] = is_interacting.astype(int)

    resample_interval = CONFIG["INTERVAL"]
    agg_trades_df["delta"] = agg_trades_df["quantity"] * np.where(
        agg_trades_df["is_buyer_maker"],
        -1,
        1,
    )
    
    # Resample with proper NaN handling
    volume_delta = agg_trades_df["delta"].resample(resample_interval).sum()
    klines_df["volume_delta"] = volume_delta.fillna(0)
    
    # Calculate rolling statistics with proper NaN handling
    delta_mean = klines_df["volume_delta"].rolling(window=60, min_periods=1).mean()
    delta_std = klines_df["volume_delta"].rolling(window=60, min_periods=1).std()
    
    # Calculate z-score with safe division
    delta_std_safe = delta_std.replace(0, np.nan)
    if not delta_std_safe.isna().all():
        klines_df["delta_zscore"] = (
            (klines_df["volume_delta"] - delta_mean) / delta_std_safe
        ).fillna(0)
    else:
        klines_df["delta_zscore"] = 0
    
    # Calculate reversion score with safe division
    zscore_threshold = params["zscore_threshold"]
    if zscore_threshold > 0:
        klines_df["reversion_score"] = (
            np.clip(-klines_df["delta_zscore"] / zscore_threshold, -1, 1)
            * klines_df["Is_Interacting"]
        )
    else:
        klines_df["reversion_score"] = 0

    # --- 2. Combine Sub-Scores into Final Score ---
    weight_trend = params.get("weight_trend", 1.0)
    weight_reversion = params.get("weight_reversion", 1.0)
    weight_sentiment = params.get("weight_sentiment", 1.0)
    
    klines_df["Confidence_Score"] = (
        klines_df["trend_score"] * weight_trend
        + klines_df["reversion_score"] * weight_reversion
        + klines_df["sentiment_score"] * weight_sentiment
    )

    total_weight = weight_trend + weight_reversion + weight_sentiment
    if total_weight > 0:
        klines_df["Confidence_Score"] /= total_weight

    # Final cleaning - remove any remaining NaN values
    final_df = klines_df.dropna(subset=["ATR", "Confidence_Score"])
    
    # Additional safety check - fill any remaining NaN with 0
    final_df = final_df.fillna(0)

    if DEBUG_MODE:
        print("\n--- DEBUG INFO (Post-Scoring) ---")
        print(final_df["Confidence_Score"].describe())
        print(f"\nTotal rows sent to backtester: {len(final_df)}")
        print("--- END DEBUG INFO ---\n")

    print("Feature calculation and scoring complete.\n")
    return final_df


def calculate_and_label_regimes(
    klines_df,
    agg_trades_df,
    futures_df,
    params,
    sr_levels,
):
    """Calculates features and labels market regimes."""
    print("--- Step 4: Calculating and Labeling Regimes ---")

    prepared_df = calculate_features_and_score(
        klines_df,
        agg_trades_df,
        futures_df,
        params,
        sr_levels,
    )

    if "ADX" in prepared_df.columns:
        prepared_df["Is_Strong_Trend"] = (
            prepared_df["ADX"] > params.get("trend_strength_threshold", 25)
        ).astype(int)
        print(
            f"Labeled 'Is_Strong_Trend' regime using ADX > {params.get('trend_strength_threshold', 25)}.",
        )
    else:
        prepared_df["Is_Strong_Trend"] = 0

    # Integration with MarketRegimeClassifier
    try:
        from src.analyst.regime_classifier import MarketRegimeClassifier
        from src.analyst.sr_analyzer import SRLevelAnalyzer

        sr_analyzer_instance = SRLevelAnalyzer(CONFIG["sr_analyzer"])
        regime_classifier_instance = MarketRegimeClassifier(
            CONFIG,
            sr_analyzer_instance,
        )

        if not regime_classifier_instance.load_model():
            print("Training a new Market Regime Classifier.")
            regime_classifier_instance.train_classifier(
                prepared_df.copy(),
                prepared_df.copy(),
            )

        predicted_regimes = []
        for idx in prepared_df.index:
            row_df = prepared_df.loc[[idx]]
            regime, _, _ = regime_classifier_instance.predict_regime(
                row_df,
                row_df,
                sr_levels,
            )
            predicted_regimes.append(regime)

        prepared_df["Market_Regime_Label"] = predicted_regimes
        print("Market regime labeling complete.\n")

    except ImportError as e:
        print(f"Could not import analyst modules for regime classification: {e}")
        print("Skipping detailed regime labeling.")
        prepared_df["Market_Regime_Label"] = "UNKNOWN"

    return prepared_df


if __name__ == "__main__":
    from src.analyst.data_utils import create_dummy_data

    klines_filename = CONFIG["KLINES_FILENAME"]
    agg_trades_filename = CONFIG["AGG_TRADES_FILENAME"]
    futures_filename = CONFIG["FUTURES_FILENAME"]
    prepared_data_filename = CONFIG["PREPARED_DATA_FILENAME"]

    # Ensure directories exist
    os.makedirs(os.path.dirname(klines_filename), exist_ok=True)

    create_dummy_data(klines_filename, "klines")
    create_dummy_data(agg_trades_filename, "agg_trades")
    create_dummy_data(futures_filename, "futures")

    klines_df, agg_trades_df, futures_df = load_raw_data()
    if klines_df.empty:
        sys.exit(1)

    daily_df_for_sr = (
        klines_df.resample("D")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            },
        )
        .dropna()
    )
    sr_levels = get_sr_levels(daily_df_for_sr)

    best_params = CONFIG["best_params"]

    prepared_df = calculate_and_label_regimes(
        klines_df,
        agg_trades_df,
        futures_df,
        best_params,
        sr_levels,
    )
    prepared_df.to_csv(prepared_data_filename)
    print(f"--- Final Prepared Data Saved to '{prepared_data_filename}' ---")
