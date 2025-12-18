"""
Assess Layer Impact Script
--------------------------
This script implements a targeted diagnostic assessment of Layer 0 (Noise Filtering)
and Layer 1 (Sample Weighting) on the downstream performance of the trading pipeline.

Methodology:
1.  **Phase A (Setup):** Load market data and define "Winning Geometries" (TP/SL/Horizon).
2.  **Phase B (Layer 0):** Compare Raw vs. Kalman-Smoothed data to quantify noise reduction
    and its impact on trade outcomes (whipsaw reduction).
3.  **Phase C (Layer 1):** Decompose sample weights into components (Magnitude, Uniqueness,
    Consistency, Quality, Risk, Time) and evaluate their information content (IC) and
    Effective Sample Size (ESS).

Outputs:
- Markdown Report: `layer_impact_report.md`
- CSV Artifacts: `l0_noise_stats.csv`, `l1_component_weights.csv`
"""

import os
import sys
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning
from src.training.steps.labeling.label_based_layer_0 import compute_rolling_vwap
from src.training.steps.labeling.multi_label_voting_utils import compute_kalman_smoothed_price_and_volatility
from src.training.steps.labeling.generate_weights_per_label import (
    generate_weights_per_label,
    compute_uniqueness_weights,
    compute_multi_horizon_consistency,
    compute_label_agreement_consistency
)
from src.training.steps.labeling.feature_generation_meta_labeling_step import (
    compute_realized_returns,
    DEFAULT_TRANSACTION_COST
)

# Mock Geometry dataclass for typing if needed, though we'll use dicts mostly
from dataclasses import dataclass

@dataclass
class MockGeometry:
    family: str
    params: Dict[str, Any]

def load_market_data(data_path: str, symbol: str, timeframe: str) -> pd.DataFrame:
    """Load market data from parquet or CSV."""
    path = Path(data_path)
    if not path.exists():
        # Try constructing path from standard layout if simple path not found
        # standard: data/historical/{exchange}/{symbol}/{timeframe}.parquet
        # but let's assume user provides direct path for now or we search.
        tprint_warning(f"Data path {path} not found. Attempting recursive search...")
        matches = list(Path("data").rglob(f"{symbol}*{timeframe}*.parquet"))
        if matches:
            path = matches[0]
            tprint_info(f"Found match: {path}")
        else:
            raise FileNotFoundError(f"Could not find data for {symbol} {timeframe}")

    if path.suffix == '.parquet':
        df = pd.read_parquet(path)
    elif path.suffix == '.csv':
        df = pd.read_csv(path, parse_dates=True, index_col=0)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    # Basic standardization
    df.columns = [c.lower() for c in df.columns]
    if 'date' in df.columns:
        df = df.set_index('date')
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Ensure sorted unique index
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]

    # Ensure essential columns
    required = ['close', 'high', 'low']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df

def get_winning_geometries() -> List[MockGeometry]:
    """
    Define a set of representative 'Winning Geometries' for the assessment.
    In a real scenario, these could be loaded from a previous optimization artifact.
    Here we define a diverse set covering the main families.
    """
    return [
        # 1. Trend: Wide stops, long horizon
        MockGeometry("Trend Continuation", {"tp_mult": 3.0, "sl_mult": 1.0, "horizon": 24}),
        # 2. Momentum: Tight stops, short horizon
        MockGeometry("Momentum", {"tp_mult": 2.0, "sl_mult": 0.8, "horizon": 12}),
        # 3. Mean Reversion: Tight targets, medium horizon (assumes logic inverts signal)
        MockGeometry("Mean Reversion", {"tp_mult": 1.5, "sl_mult": 1.5, "horizon": 16}),
    ]

def phase_a_setup(args) -> Tuple[pd.DataFrame, List[MockGeometry], pd.Series]:
    """Phase A: Load Data & Setup."""
    tprint_info("--- Phase A: Setup ---")
    df = load_market_data(args.data_path, args.symbol, args.timeframe)
    tprint_success(f"Loaded {len(df)} bars for {args.symbol} {args.timeframe}")

    # Calculate simple volatility for geometry sizing if not present
    if 'volatility_1d' not in df.columns:
        df['volatility_1d'] = df['close'].pct_change().rolling(24).std().bfill()

    geos = get_winning_geometries()

    # Generate simple trend signal for event generation (e.g., RSI or simple MA cross)
    # We need EVENTS to evaluate L1.
    # Let's use a simple volatility breakout logic for events: |ret| > 1.5 * vol
    returns = df['close'].pct_change()
    vol = df['volatility_1d']

    # Use config threshold or default
    threshold = 1.0
    signal_strength = returns.abs() / (vol + 1e-9)
    events_mask = signal_strength > threshold

    # Limit event count for speed if needed, but ensure coverage
    events = df.index[events_mask]
    tprint_info(f"Generated {len(events)} candidate events based on volatility breakout.")

    # Assign random directions for simulation if we don't have a full signal engine
    # In reality, 'Trend' implies following price, 'MR' implies opposing.
    # We will derive direction per geometry later.

    return df, geos, events

def phase_b_layer0_diagnostics(df: pd.DataFrame, geos: List[MockGeometry], events: pd.DatetimeIndex, outcomes_dir: Path) -> pd.DataFrame:
    """Phase B: Layer 0 (Noise Filtering) Diagnostics."""
    tprint_info("--- Phase B: Layer 0 (Noise) Diagnostics ---")

    # 1. Compute Layer 0 features (Kalman)
    # We replicate the logic from label_based_layer_0.py roughly or call it if possible
    # We'll use the imported utility.

    close_series = df['close']
    volume_series = df['volume'] if 'volume' in df.columns else None

    # Standard params often used
    vwap_series = compute_rolling_vwap(close_series, volume_series, lookback=20)

    kalman_price, kalman_vol = compute_kalman_smoothed_price_and_volatility(
        prices=close_series,
        volume=volume_series,
        vwap=vwap_series,
        process_noise=1e-4,
        measurement_noise=0.01,
        vol_window=24
    )

    # 2. Metric: Noise Reduction
    # Compare volatility of returns
    raw_ret = close_series.pct_change()
    kalman_ret = kalman_price.pct_change()

    raw_std = raw_ret.std()
    kalman_std = kalman_ret.std()
    noise_reduction_ratio = raw_std / (kalman_std + 1e-9)

    tprint_info(f"Raw Vol: {raw_std:.5f}, Kalman Vol: {kalman_std:.5f}")
    tprint_info(f"Noise Reduction Ratio (Target > 1.0): {noise_reduction_ratio:.4f}")

    # 3. Metric: Whipsaw Reduction via Barrier Simulation
    # We run compute_realized_returns TWICE for each geometry:
    # Once with Raw Close, Once with Kalman Price.

    results = []

    # Prepare a DataFrame for Simulation
    sim_df = df.copy()
    sim_df['kalman_close'] = kalman_price

    # Signals: Simple Trend Following (+1 if ret > 0 else -1)
    # We stick to one signal logic to isolate Price Data impact
    signals = pd.DataFrame(index=df.index)
    signals['consensus'] = np.sign(close_series.pct_change().rolling(5).mean()).fillna(0)

    for g in geos:
        # RAW Simulation
        # Volatility is usually smoothed in prod, but let's use the respective vol for fairness?
        # Actually, L0 implies we use Kalman for everything. Baseline uses Raw for everything.

        # Baseline (Raw)
        ret_raw, lbl_raw, _, _, _, _, _, _ = compute_realized_returns(
            df=sim_df, # uses 'close' by default
            signals=signals,
            profit_threshold=sim_df['volatility_1d'] * g.params['tp_mult'],
            stop_threshold=sim_df['volatility_1d'] * g.params['sl_mult'],
            horizon=g.params['horizon'],
            transaction_cost=DEFAULT_TRANSACTION_COST
        )

        # Treated (Kalman)
        # We need to hack compute_realized_returns or pass a df where 'close' is kalman
        df_kalman = sim_df.copy()
        df_kalman['close'] = df_kalman['kalman_close']
        # Recalculate vol on kalman? usually vol_regime is separate, but vol_1d is derived.
        # For strict comparison, let's keep vol threshold identical (defined by regime/raw vol)
        # to see if PRICE noise triggers stops.

        ret_kal, lbl_kal, _, _, _, _, _, _ = compute_realized_returns(
            df=df_kalman,
            signals=signals,
            profit_threshold=sim_df['volatility_1d'] * g.params['tp_mult'],
            stop_threshold=sim_df['volatility_1d'] * g.params['sl_mult'],
            horizon=g.params['horizon'],
            transaction_cost=DEFAULT_TRANSACTION_COST
        )

        # Analyze Whipsaws
        # Whipsaw = Raw hit Stop (-1 or loss), but Kalman did not (Profit or Time limit)
        # Align to events

        # Subset to our event set
        common_idx = events.intersection(lbl_raw.index).intersection(lbl_kal.index)

        r_raw = ret_raw.loc[common_idx]
        r_kal = ret_kal.loc[common_idx]

        # Count "Saved" trades: Raw < 0 and Kalman > Raw
        saved_mask = (r_raw < 0) & (r_kal > r_raw)
        n_saved = saved_mask.sum()
        pnl_saved = (r_kal[saved_mask] - r_raw[saved_mask]).sum()

        # Count "Missed" opportunities: Raw > 0 and Kalman < Raw
        missed_mask = (r_raw > 0) & (r_kal < r_raw)
        n_missed = missed_mask.sum()
        pnl_missed = (r_raw[missed_mask] - r_kal[missed_mask]).sum()

        results.append({
            "family": g.family,
            "params": str(g.params),
            "n_events": len(common_idx),
            "n_saved_whipsaws": int(n_saved),
            "pnl_saved": float(pnl_saved),
            "n_missed_opps": int(n_missed),
            "pnl_missed": float(pnl_missed),
            "net_impact": float(pnl_saved - pnl_missed)
        })

    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv(outcomes_dir / "l0_noise_stats.csv", index=False)
    return results_df

def phase_c_layer1_diagnostics(df: pd.DataFrame, events: pd.DatetimeIndex, outcomes_dir: Path) -> pd.DataFrame:
    """Phase C: Layer 1 (Weighting) Component Decomposition."""
    tprint_info("--- Phase C: Layer 1 (Weights) Diagnostics ---")

    # We need realized returns for the events to calculate weights (Magnitudes)
    # Let's assume a generic horizon for the weight calculation assessment (e.g., 24 bars)
    # or use the future return of the asset directly.
    future_ret = df['close'].pct_change(24).shift(-24).reindex(events)

    # Inputs for generate_weights_per_label
    returns_arr = future_ret.fillna(0.0).values
    t_events = events
    close_series = df['close']

    # Pre-compute auxiliary scores
    # Uniqueness
    # We need t1 (end times). Let's assume fixed 24 bar horizon.
    t1 = pd.Series(df.index.searchsorted(events) + 24, index=events)
    # Map back to timestamps, clipping at end
    t1 = t1.apply(lambda x: df.index[min(x, len(df)-1)])

    uniqueness_scores = compute_uniqueness_weights(t1, events, df.index).values

    # Consistency (Multi-horizon)
    cons_s = compute_multi_horizon_consistency(df['close'], horizons=[6, 12, 24])
    consistency_scores = cons_s.reindex(events).fillna(0.5).values

    # Quality (Mock Confident Learning - usually comes from a model)
    # We'll mock it with a noisy version of future returns (oracle-ish but noisy)
    # In prod this is OOF probability.
    label_quality_scores = (np.sign(returns_arr) * 0.5 + 0.5) # Perfect correlation? No, add noise.
    noise = np.random.normal(0, 0.5, size=len(returns_arr))
    label_quality_scores = 1.0 / (1.0 + np.exp(-(returns_arr * 100 + noise))) # Sigmoid

    # Risk (Vol proxy)
    vol_proxy = df['volatility_1d'].reindex(events).fillna(0.0).values

    # Define Schemes
    schemes = {
        "Baseline": {}, # Uniform
        "Magnitude": {"mag_compression": 1.0, "exp_mag": 1.0, "uniq_intensity": 0.0, "exp_learn": 0.0, "exp_cross": 0.0, "quality_intensity": 0.0},
        "Uniqueness": {"mag_compression": 0.0, "exp_mag": 0.0, "uniq_intensity": 1.0, "exp_learn": 0.0, "exp_cross": 0.0, "quality_intensity": 0.0},
        "Consistency": {"mag_compression": 0.0, "exp_mag": 0.0, "uniq_intensity": 0.0, "exp_learn": 0.0, "exp_cross": 1.0, "quality_intensity": 0.0},
        "Quality": {"mag_compression": 0.0, "exp_mag": 0.0, "uniq_intensity": 0.0, "exp_learn": 0.0, "exp_cross": 0.0, "quality_intensity": 1.0},
        "Risk": {"mag_compression": 0.0, "exp_mag": 0.0, "downside_multiplier": 2.0}, # Needs high multiplier to trigger
        "Time": {"mag_compression": 0.0, "exp_mag": 0.0, "exp_learn": 1.0, "learn_slope": 5.0}, # Time sigmoid
        "Full_Composite": {"mag_compression": 0.8, "exp_mag": 1.0, "uniq_intensity": 1.0, "exp_learn": 1.0, "exp_cross": 1.0, "quality_intensity": 1.0, "downside_multiplier": 1.5}
    }

    results = []
    all_weights = pd.DataFrame(index=events)
    all_weights['Future_Return'] = future_ret

    for name, params in schemes.items():
        # Default params to 0.0 influence if not specified in scheme,
        # but generate_weights_per_label defaults are active.
        # We must explicitly zero out others if we want pure isolation.
        # The generate_weights_per_label signature allows overriding.
        # However, it mixes geometrically.
        # E.g. Uniqueness defaults to 1.0?
        # We need to be careful. The function docstring says:
        # W = (Mag^exp * Uniq^exp * Time^exp * Cross * Quality * Risk)
        # So setting exponent to 0.0 makes component 1.0 (neutral).

        # Base neutral config
        call_params = {
            "mag_compression": 0.0, "exp_mag": 0.0, # Neutralizes mag
            "uniq_intensity": 0.0, # Neutralizes uniq (power 0)
            "exp_learn": 0.0, # Neutralizes time
            "exp_cross": 0.0, # Neutralizes consistency (power 0)
            "quality_intensity": 0.0, # Neutralizes quality
            "downside_multiplier": 1.0, # Neutralizes risk
            "time_decay_halflife": None # Neutralizes decay
        }
        # Update with scheme specific
        call_params.update(params)

        w = generate_weights_per_label(
            returns=returns_arr,
            t_events=events,
            uniqueness_scores=uniqueness_scores,
            consistency_scores=consistency_scores,
            label_quality_scores=label_quality_scores,
            vol_proxy=vol_proxy,
            **call_params
        )

        all_weights[name] = w

        # Metrics
        # 1. Effective Sample Size (ESS)
        w_sum = w.sum()
        w_sq_sum = (w*w).sum()
        ess = (w_sum * w_sum) / (w_sq_sum + 1e-9)
        ess_ratio = ess / len(w)

        # 2. Information Coefficient (IC) with Future Return
        # We want weights to correlate with MAGNITUDE of return (absolute)
        # OR direction if Quality is used?
        # Usually L1 weights emphasize "Important" samples.
        # Let's check correlation with Abs(Return) and raw Return.
        ic_abs = np.corrcoef(w, np.abs(returns_arr))[0, 1]

        # 3. Class Separation
        # Assume Target = Sign(Return)
        winners = w[returns_arr > 0]
        losers = w[returns_arr <= 0]
        win_mean = winners.mean() if len(winners) else 0
        lose_mean = losers.mean() if len(losers) else 0
        separation = win_mean - lose_mean

        results.append({
            "Scheme": name,
            "ESS_Ratio": float(ess_ratio),
            "IC_AbsRet": float(ic_abs),
            "Winner_Mean": float(win_mean),
            "Loser_Mean": float(lose_mean),
            "Separation": float(separation)
        })

    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv(outcomes_dir / "l1_component_impact.csv", index=False)
    all_weights.to_csv(outcomes_dir / "l1_component_weights_series.csv")

    return results_df

def generate_report(outcomes_dir: Path, l0_stats: pd.DataFrame, l1_stats: pd.DataFrame):
    """Generate Markdown Report."""
    md_path = outcomes_dir / "layer_impact_report.md"

    with open(md_path, 'w') as f:
        f.write("# Layer Impact Assessment Report\n\n")
        f.write(f"**Date:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 1. Layer 0: Noise Filtering (Kalman/VWAP)\n")
        f.write("Evaluation of Price Smoothing on Trade Outcomes.\n\n")
        f.write(l0_stats.to_markdown(index=False))
        f.write("\n\n")

        total_net = l0_stats['net_impact'].sum()
        f.write(f"**Total Net Impact (PnL Saved - Missed):** {total_net:.4f}\n")
        if total_net > 0:
            f.write("✅ **Conclusion:** Layer 0 successfully reduces whipsaws more than it misses opportunities.\n\n")
        else:
            f.write("⚠️ **Conclusion:** Layer 0 smoothing might be too aggressive (lag causes missed entries).\n\n")

        f.write("## 2. Layer 1: Sample Weighting Decomposition\n")
        f.write("Impact of individual weighting components on sample importance.\n\n")
        f.write(l1_stats.to_markdown(index=False))
        f.write("\n\n")

        best_sep = l1_stats.sort_values('Separation', ascending=False).iloc[0]
        f.write(f"**Best Separator:** {best_sep['Scheme']} (Sep: {best_sep['Separation']:.4f})\n")

        highest_ess = l1_stats.sort_values('ESS_Ratio', ascending=False).iloc[0]
        f.write(f"**Highest Retention:** {highest_ess['Scheme']} (ESS: {highest_ess['ESS_Ratio']:.2%})\n")

    tprint_success(f"Report generated: {md_path}")

def main():
    parser = argparse.ArgumentParser(description="Assess Layer 0 and Layer 1 Impact")
    parser.add_argument("--data-path", type=str, required=True, help="Path to market data parquet/csv")
    parser.add_argument("--symbol", type=str, default="Unknown", help="Symbol name")
    parser.add_argument("--timeframe", type=str, default="15m", help="Timeframe")
    parser.add_argument("--output-dir", type=str, default="outcomes", help="Output directory")

    args = parser.parse_args()

    outcomes_dir = Path(args.output_dir)
    outcomes_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Phase A
        df, geos, events = phase_a_setup(args)

        # Phase B
        l0_stats = phase_b_layer0_diagnostics(df, geos, events, outcomes_dir)

        # Phase C
        l1_stats = phase_c_layer1_diagnostics(df, events, outcomes_dir)

        # Report
        generate_report(outcomes_dir, l0_stats, l1_stats)

    except Exception as e:
        tprint_warning(f"Script failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
