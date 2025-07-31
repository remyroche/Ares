# backtesting/ares_data_preparer.py
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import os
import subprocess
import sys

# We assume a config file exists at `src/config.py` that defines a CONFIG dictionary.
# And that data_utils contains the necessary data loading functions.
try:
    from src.config import CONFIG
    from src.analyst.data_utils import (
        load_klines_data,
        load_agg_trades_data,
        load_futures_data,
    )
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure src/config.py and src/analyst/data_utils.py are available.")

# --- DEBUGGING FLAG ---
DEBUG_MODE = True


def load_raw_data(symbol: str, exchange: str = "BINANCE"):
    """Loads raw data from CSV files, running the downloader if they are missing."""
    print(f"--- Step 1: Loading Raw Data for {symbol} on {exchange} ---")

    # Dynamically generate filenames to be exchange and symbol-specific
    lookback_years = CONFIG.get("lookback_years", 2)
    interval = CONFIG.get("trading_interval", "15m")
    klines_filename = f"data_cache/{exchange}_{symbol}_{interval}_{lookback_years}y_klines.csv"
    agg_trades_filename = f"data_cache/{exchange}_{symbol}_{lookback_years}y_aggtrades.csv"
    futures_filename = f"data_cache/{exchange}_{symbol}_futures_{lookback_years}y_data.csv"

    # The downloader script is now modularized in training steps, but we can keep this for standalone runs.
    downloader_script_name = CONFIG.get("downloader_script_name", "src/training/steps/data_downloader.py")

    required_files = [klines_filename, agg_trades_filename, futures_filename]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print(f"Missing raw data files: {', '.join(missing_files)}")
        print(f"Calling downloader script: '{downloader_script_name}' for {symbol} on {exchange}...")
        try:
            # Pass symbol and exchange to the downloader script
            subprocess.run(
                [sys.executable, downloader_script_name, "--symbol", symbol, "--exchange", exchange], check=True
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            print(f"ERROR during download: {e}")
            sys.exit(1)

    try:
        klines_df = load_klines_data(klines_filename)
        agg_trades_df = load_agg_trades_data(agg_trades_filename)
        futures_df = load_futures_data(futures_filename)

        # Ensure indices are unique
        klines_df = klines_df[~klines_df.index.duplicated(keep="first")]
        agg_trades_df = agg_trades_df[~agg_trades_df.index.duplicated(keep="first")]

        print(
            f"Loaded {len(klines_df)} k-lines, {len(agg_trades_df)} agg trades, and {len(futures_df)} futures data points.\n"
        )
        return klines_df, agg_trades_df, futures_df
    except FileNotFoundError as e:
        print(f"ERROR: Could not find required file: {e}. Exiting.")
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
    klines_df, agg_trades_df, futures_df, params, sr_levels
):
    """
    Calculates all feature sub-scores and combines them into a final Confidence Score.
    """
    print("--- Step 3: Calculating Features and Confidence Score ---")

    klines_df = pd.merge_asof(
        klines_df.sort_index(),
        futures_df.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )
    klines_df.fillna(method="ffill", inplace=True)

    # --- 1. Calculate Sub-Scores ---

    # Sentiment Score
    fr_score = (klines_df["fundingRate"] - klines_df["fundingRate"].mean()) / klines_df[
        "fundingRate"
    ].std()
    klines_df["sentiment_score"] = np.clip(fr_score, -1, 1).fillna(0)

    # Trend Score
    klines_df.ta.adx(
        length=params["adx_period"], append=True, col_names=("ADX", "DMP", "DMN")
    )
    macd = klines_df.ta.macd(append=False)
    klines_df["MACD_HIST"] = macd["MACDh_12_26_9"]
    normalized_hist = klines_df["MACD_HIST"] / klines_df["close"]
    direction_score = np.clip(normalized_hist * params["scaling_factor"], -1, 1)
    momentum_score = np.clip(
        (klines_df["ADX"] - params["trend_threshold"])
        / (params["max_strength_threshold"] - params["trend_threshold"]),
        0,
        1,
    )
    klines_df["trend_score"] = (direction_score * momentum_score).fillna(0)

    # Mean Reversion Score
    klines_df.ta.atr(length=params["atr_period"], append=True, col_names=("ATR"))

    proximity_threshold = klines_df["ATR"] * params["proximity_multiplier"]
    is_interacting = pd.Series(False, index=klines_df.index)
    for level in sr_levels:
        is_interacting |= (klines_df["low"] <= level + proximity_threshold) & (
            klines_df["high"] >= level - proximity_threshold
        )
    klines_df["Is_Interacting"] = is_interacting.astype(int)

    resample_interval = CONFIG["INTERVAL"]
    agg_trades_df["delta"] = agg_trades_df["quantity"] * np.where(
        agg_trades_df["is_buyer_maker"], -1, 1
    )
    klines_df["volume_delta"] = (
        agg_trades_df["delta"].resample(resample_interval).sum().fillna(0)
    )
    delta_mean = klines_df["volume_delta"].rolling(window=60).mean()
    delta_std = klines_df["volume_delta"].rolling(window=60).std()
    klines_df["delta_zscore"] = (
        (klines_df["volume_delta"] - delta_mean) / delta_std.replace(0, np.nan)
    ).fillna(0)
    klines_df["reversion_score"] = (
        np.clip(-klines_df["delta_zscore"] / params["zscore_threshold"], -1, 1)
        * klines_df["Is_Interacting"]
    )

    # --- 2. Combine Sub-Scores into Final Score ---
    klines_df["Confidence_Score"] = (
        klines_df["trend_score"] * params["weight_trend"]
        + klines_df["reversion_score"] * params["weight_reversion"]
        + klines_df["sentiment_score"] * params["weight_sentiment"]
    )

    total_weight = (
        params["weight_trend"] + params["weight_reversion"] + params["weight_sentiment"]
    )
    if total_weight > 0:
        klines_df["Confidence_Score"] /= total_weight

    final_df = klines_df.dropna(subset=["ATR", "Confidence_Score"])

    if DEBUG_MODE:
        print("\n--- DEBUG INFO (Post-Scoring) ---")
        print(final_df["Confidence_Score"].describe())
        print(f"\nTotal rows sent to backtester: {len(final_df)}")
        print("--- END DEBUG INFO ---\n")

    print("Feature calculation and scoring complete.\n")
    return final_df


def calculate_and_label_regimes(
    klines_df, agg_trades_df, futures_df, params, sr_levels
):
    """Calculates features and labels market regimes."""
    print("--- Step 4: Calculating and Labeling Regimes ---")

    prepared_df = calculate_features_and_score(
        klines_df, agg_trades_df, futures_df, params, sr_levels
    )

    if "ADX" in prepared_df.columns:
        prepared_df["Is_Strong_Trend"] = (
            prepared_df["ADX"] > params.get("trend_strength_threshold", 25)
        ).astype(int)
        print(
            f"Labeled 'Is_Strong_Trend' regime using ADX > {params.get('trend_strength_threshold', 25)}."
        )
    else:
        prepared_df["Is_Strong_Trend"] = 0

    # Integration with MarketRegimeClassifier
    try:
        from src.analyst.regime_classifier import MarketRegimeClassifier
        from src.analyst.sr_analyzer import SRLevelAnalyzer

        sr_analyzer_instance = SRLevelAnalyzer(CONFIG["sr_analyzer"])
        regime_classifier_instance = MarketRegimeClassifier(
            CONFIG, sr_analyzer_instance
        )

        if not regime_classifier_instance.load_model():
            print("Training a new Market Regime Classifier.")
            regime_classifier_instance.train_classifier(
                prepared_df.copy(), prepared_df.copy()
            )

        predicted_regimes = []
        for idx in prepared_df.index:
            row_df = prepared_df.loc[[idx]]
            regime, _, _ = regime_classifier_instance.predict_regime(
                row_df, row_df, sr_levels
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
            }
        )
        .dropna()
    )
    sr_levels = get_sr_levels(daily_df_for_sr)

    best_params = CONFIG["best_params"]

    prepared_df = calculate_and_label_regimes(
        klines_df, agg_trades_df, futures_df, best_params, sr_levels
    )
    prepared_df.to_csv(prepared_data_filename)
    print(f"--- Final Prepared Data Saved to '{prepared_data_filename}' ---")
