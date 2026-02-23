"""
Diagnostic tool for Trade Quality Decomposition.

This module implements the "Trade Quality Decomposition Plot" diagnostic
to visualize how well the sizing score (and its components) aligns with
realized trade outcomes (Utility, MAE, MFE, Duration).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from extreme_price_movements.utils import tprint

def generate_trade_quality_plot(
    df: pd.DataFrame,
    output_path: str,
    score_col: str = "score",
    u_real_col: str = "u_policy_net",
    mae_hat_log_col: str = "oof_log_mae_q70_hat",
    mfe_hat_log_col: str = "oof_log_mfe_hat",
    dur_hat_log_col: str = "oof_log_dur_hat",
    mae_real_col: str = "mae_ret",
    mfe_real_col: str = "mfe_ret",
    dur_real_col: str = "duration",
    bucket_label: str = "Trade Quality Decomposition",
):
    """
    Generates the Trade Quality Decomposition Plot.

    Step 1 — For OOF trades
    Compute for each trade:
    u_hat      = predicted utility (score_col)
    mae_hat    = predicted MAE
    mfe_hat    = predicted MFE
    dur_hat    = predicted duration
    u_real     = realized log utility

    Step 2 — Rank trades by execution score
    score_exec = u_hat / (log_mae_hat + eps)

    Step 3 — Bucket trades
    Split trades into deciles by score_exec.

    Step 4 — Plot (ONE figure)
    X-axis: score_exec deciles (worst -> best)
    Plot four curves:
    - Realized utility: mean(u_real)
    - Realized MAE: mean(log1p(MAE_real))
    - Realized MFE: mean(log1p(MFE_real))
    - Realized duration: mean(log1p(duration_real))
    """

    tprint(f"Generating Trade Quality Decomposition Plot for {bucket_label}...")

    required_cols = [score_col, u_real_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        tprint(f"  Skipping plot: missing required columns {missing}")
        return

    # Work on a copy
    d = df.copy()

    # Ensure numeric types
    for c in d.columns:
        if pd.api.types.is_numeric_dtype(d[c]):
            d[c] = pd.to_numeric(d[c], errors='coerce').fillna(0.0)

    # 1. Compute score_exec
    # If mae_hat_log_col is available, use it for risk-adjustment
    eps = 1e-6
    if mae_hat_log_col in d.columns:
        # log_mae_hat is expected to be log(1+mae) or similar.
        # Ensure positivity for denominator.
        log_mae = np.maximum(d[mae_hat_log_col].values, 0.0)
        # u_hat (score) / (log_mae + eps)
        # Note: If u_hat can be negative, this interpretation holds (risk-adjusted return).
        d["score_exec"] = d[score_col] / (log_mae + eps)
        tprint(f"  Using risk-adjusted score: {score_col} / {mae_hat_log_col}")
    else:
        # Fallback to raw score if MAE prediction missing
        d["score_exec"] = d[score_col]
        tprint(f"  Using raw score (MAE prediction missing): {score_col}")

    # 2. Bucket trades into deciles
    # Use qcut with rank='first' to handle ties or fewer unique values
    try:
        d["decile"] = pd.qcut(d["score_exec"], 10, labels=False, duplicates='drop')
    except Exception as e:
        tprint(f"  Skipping plot: failed to bucket deciles ({e})")
        return

    # 3. Compute realized metrics per decile
    # Realized columns mapping
    metrics = {
        "Utility": (u_real_col, lambda x: x, "mean(u_real)"),
        "MAE": (mae_real_col, lambda x: np.log1p(np.maximum(x, 0)), "mean(log1p(MAE))"),
        "MFE": (mfe_real_col, lambda x: np.log1p(np.maximum(x, 0)), "mean(log1p(MFE))"),
        "Duration": (dur_real_col, lambda x: np.log1p(np.maximum(x, 0)), "mean(log1p(Dur))"),
    }

    stats = d.groupby("decile")["score_exec"].count().rename("count").to_frame()

    plot_data = {}

    for name, (col, transform, label) in metrics.items():
        if col in d.columns:
            # Apply transform
            vals = transform(d[col].values)
            # Group by decile and compute mean
            # We align by index since 'decile' is in d
            # Create a temp series with same index
            s = pd.Series(vals, index=d.index)
            means = s.groupby(d["decile"]).mean()
            plot_data[name] = means
            stats[name] = means
        else:
            tprint(f"  Missing realized column for {name}: {col}")

    if not plot_data:
        tprint("  No metrics to plot.")
        return

    # 4. Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    colors = {
        "Utility": "blue",
        "MAE": "red",
        "MFE": "green",
        "Duration": "orange"
    }

    markers = {
        "Utility": "o",
        "MAE": "x",
        "MFE": "^",
        "Duration": "s"
    }

    # Utility is primary (usually returns), others on secondary axis?
    # Log1p MAE/MFE/Dur are roughly 0.0-0.1 range for prices, but Dur is log(hours).
    # log1p(24h) ~ 3.2. log1p(0.01) ~ 0.01.
    # Scales might be very different.
    # Let's normalize or use twinx.
    # The prompt says "Plot four curves". Usually implies single Y-axis or dual.
    # "Utility curve UP, MAE curve DOWN, MFE curve UP, Duration curve DOWN".
    # Since scales differ, let's normalize each curve to [0,1] or z-score for shape comparison?
    # Or just plot raw values on twin axes.
    # Utility (log return) ~ 0.005.
    # MAE (log return) ~ 0.005.
    # MFE (log return) ~ 0.01.
    # Duration (log hours) ~ 2.0 - 4.0.
    # Duration is the outlier.

    # Plot Utility, MAE, MFE on left axis (similar scales: log returns).
    # Plot Duration on right axis.

    ax2 = ax1.twinx()

    lines = []
    labels = []

    for name, series in plot_data.items():
        ax = ax2 if name == "Duration" else ax1
        color = colors.get(name, "black")
        marker = markers.get(name, "o")

        l, = ax.plot(series.index, series.values, color=color, marker=marker, label=name, linewidth=2)
        lines.append(l)
        labels.append(name)

    ax1.set_xlabel("Score Decile (0=Worst, 9=Best)")
    ax1.set_ylabel("Log Return (Utility, MAE, MFE)")
    ax2.set_ylabel("Log Duration (Hours)")

    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"Trade Quality Decomposition: {bucket_label}")

    # Combined legend
    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc="upper left")

    # Save
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        tprint(f"  Saved plot to {output_path}")

        # Also save CSV for inspection
        csv_path = output_path.replace(".png", ".csv")
        stats.to_csv(csv_path)
        tprint(f"  Saved data to {csv_path}")

    except Exception as e:
        tprint(f"  Failed to save plot: {e}")
        plt.close(fig)
