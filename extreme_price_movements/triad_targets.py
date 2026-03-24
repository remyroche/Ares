"""
Triad Targets Module for Regime Mining.

This module implements the triad target system for identifying predictive regimes
in price movements. The three targets (EFF, ELA, VAME) are independent continuous
regressands bounded in [0, 1].

Target Definitions:
- target_eff: Trend Efficiency - measures how efficiently price moves toward a final level
- target_ela: Elasticity - measures price stretch relative to ATR
- target_vame: Expansion Sustainability - measures sustainability of price expansion

Critical Constraints:
- No future-on-future persistence
- No full future range as drawdown proxy for VAME
- No simplex-normalization of targets
- Targets remain independent continuous regressands in [0, 1]
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# Constants
TRIAD_TARGET_NAMES: List[str] = ["target_eff", "target_ela", "target_vame"]


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """
    Compute sigmoid function element-wise.

    Parameters
    ----------
    x : np.ndarray or float
        Input values

    Returns
    -------
    np.ndarray or float
        Sigmoid of input: 1 / (1 + exp(-x))
    """
    return 1 / (1 + np.exp(-x))


def harmonic_mean(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
    """
    Compute harmonic mean of two values/arrays.

    Parameters
    ----------
    a : np.ndarray or float
        First value(s)
    b : np.ndarray or float
        Second value(s)

    Returns
    -------
    np.ndarray or float
        Harmonic mean: (2 * a * b) / (a + b + epsilon)
    """
    return (2 * a * b) / (a + b + 1e-9)


def get_bounded_triad(df: pd.DataFrame, n: int = 24) -> pd.DataFrame:
    """
    Compute bounded triad targets for regime mining.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: 'close', 'high', 'low', 'atr'
    n : int
        Horizon in bars (default 24)

    Returns
    -------
    pd.DataFrame
        Original df with added columns: target_eff, target_ela, target_vame

    CRITICAL CONSTRAINTS:
    - Do NOT use future-on-future persistence like: p_eff = s_eff.rolling(n).mean().shift(-n)
    - Do NOT use full future range as a drawdown proxy for VAME.
    - Do NOT simplex-normalize the three targets.
    - Do NOT add a 4th target.
    - Targets should remain independent continuous regressands in [0, 1].
    """
    # Forward-looking price metrics
    fwd_close = df["close"].shift(-n)

    fwd_high_max = df["high"].rolling(n).max().shift(-n)
    fwd_low_min = df["low"].rolling(n).min().shift(-n)

    # Excursion calculations
    up_exc = fwd_high_max - df["close"]
    down_exc = df["close"] - fwd_low_min
    max_excursion = np.maximum(up_exc, down_exc)

    # Final displacement
    final_disp = (fwd_close - df["close"]).abs()

    # 1) Trend Efficiency
    path_traveled = (
        df["close"]
        .diff()
        .abs()
        .rolling(n)
        .sum()
        .shift(-n)
    )

    s_eff = final_disp / (path_traveled + 1e-9)

    # IMPORTANT: use backward persistence, not future-stacked persistence
    p_eff = s_eff.shift(1).rolling(n).mean()

    df["target_eff"] = harmonic_mean(
        s_eff.clip(0, 1),
        p_eff.clip(0, 1)
    )

    # 2) Elasticity
    s_ela = np.tanh(
        max_excursion / (df["atr"] * 1.5 + 1e-9)
    )

    p_ela = 1 - (
        final_disp / (max_excursion + 1e-9)
    )

    df["target_ela"] = harmonic_mean(
        s_ela.clip(0, 1),
        p_ela.clip(0, 1)
    )

    # 3) Expansion Sustainability
    # IMPORTANT: use dominant excursion direction, not final close sign
    direction = np.where(up_exc >= down_exc, 1, -1)

    worst_move_against = np.where(
        direction > 0,
        df["close"] - fwd_low_min,
        fwd_high_max - df["close"]
    )

    vol_ratio = max_excursion / (df["atr"] + 1e-9)

    floor_gate = np.clip(
        (vol_ratio - 1.0) / 0.5,
        0,
        1
    )

    s_vame = sigmoid(vol_ratio - 2) * floor_gate

    p_vame = 1 - (
        worst_move_against / (max_excursion + 1e-9)
    )

    df["target_vame"] = harmonic_mean(
        s_vame.clip(0, 1),
        p_vame.clip(0, 1)
    )

    return df


def compute_triad_targets_for_horizons(
    df: pd.DataFrame,
    horizons: List[int],
    atr_col: str = "atr"
) -> Dict[int, pd.DataFrame]:
    """
    Compute triad targets for multiple horizons.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: 'close', 'high', 'low', and the specified ATR column
    horizons : List[int]
        List of horizon values (in bars) to compute targets for
    atr_col : str
        Name of the ATR column (default 'atr')

    Returns
    -------
    Dict[int, pd.DataFrame]
        Dictionary mapping horizon -> DataFrame with target columns.
        Each DataFrame contains the original data plus target columns
        suffixed with the horizon (e.g., 'target_eff_24', 'target_ela_24', 'target_vame_24')
    """
    results: Dict[int, pd.DataFrame] = {}

    for horizon in horizons:
        # Create a copy to avoid modifying the original
        df_copy = df.copy()

        # Ensure ATR column is correctly named for get_bounded_triad
        if atr_col != "atr" and atr_col in df_copy.columns:
            df_copy["atr"] = df_copy[atr_col]
        elif "atr" not in df_copy.columns:
            raise ValueError(f"ATR column '{atr_col}' not found in DataFrame")

        # Compute targets for this horizon
        df_with_targets = get_bounded_triad(df_copy, n=horizon)

        # Rename target columns with horizon suffix
        rename_map = {
            "target_eff": f"target_eff_{horizon}",
            "target_ela": f"target_ela_{horizon}",
            "target_vame": f"target_vame_{horizon}"
        }
        df_with_targets = df_with_targets.rename(columns=rename_map)

        # Store only the target columns plus original data
        target_cols = list(rename_map.values())
        results[horizon] = df_with_targets[target_cols]

    return results


def compute_target_diagnostics(
    df: pd.DataFrame,
    target_names: List[str] = TRIAD_TARGET_NAMES
) -> Dict:
    """
    Compute target quality diagnostics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the target columns
    target_names : List[str]
        List of target column names to analyze

    Returns
    -------
    Dict
        Dictionary containing:
        - per_target_stats: {target: {mean, std, p05, p25, p50, p75, p95, nonzero_frac}}
        - cross_target_corr: correlation matrix between targets
        - quality_flags: list of any quality issues detected
    """
    per_target_stats: Dict[str, Dict] = {}
    quality_flags: List[str] = []

    for target in target_names:
        if target not in df.columns:
            quality_flags.append(f"Missing target column: {target}")
            continue

        # Drop NaN values for statistics
        target_data = df[target].dropna()

        if len(target_data) == 0:
            quality_flags.append(f"No valid data for target: {target}")
            per_target_stats[target] = {
                "mean": np.nan,
                "std": np.nan,
                "p05": np.nan,
                "p25": np.nan,
                "p50": np.nan,
                "p75": np.nan,
                "p95": np.nan,
                "nonzero_frac": np.nan
            }
            continue

        # Compute statistics
        stats = {
            "mean": float(target_data.mean()),
            "std": float(target_data.std()),
            "p05": float(target_data.quantile(0.05)),
            "p25": float(target_data.quantile(0.25)),
            "p50": float(target_data.quantile(0.50)),
            "p75": float(target_data.quantile(0.75)),
            "p95": float(target_data.quantile(0.95)),
            "nonzero_frac": float((target_data > 0).mean())
        }
        per_target_stats[target] = stats

        # Check for degenerate distributions
        if stats["std"] < 1e-6:
            quality_flags.append(f"Near-zero variance for target: {target}")

        # Check for extreme concentration at boundaries
        near_zero_frac = (target_data < 0.01).mean()
        near_one_frac = (target_data > 0.99).mean()
        if near_zero_frac > 0.9:
            quality_flags.append(f"Target {target} concentrated near 0 ({near_zero_frac:.1%})")
        if near_one_frac > 0.9:
            quality_flags.append(f"Target {target} concentrated near 1 ({near_one_frac:.1%})")

    # Compute cross-target correlation
    available_targets = [t for t in target_names if t in df.columns]
    cross_target_corr: np.ndarray | None = None

    if len(available_targets) > 1:
        target_df = df[available_targets].dropna()
        if len(target_df) > 1:
            cross_target_corr = target_df.corr().values

            # Check for excessive correlation
            if cross_target_corr is not None:
                n_targets = len(available_targets)
                for i in range(n_targets):
                    for j in range(i + 1, n_targets):
                        corr_val = abs(cross_target_corr[i, j])
                        if corr_val > 0.85:
                            quality_flags.append(
                                f"High correlation ({corr_val:.3f}) between "
                                f"{available_targets[i]} and {available_targets[j]}"
                            )

    return {
        "per_target_stats": per_target_stats,
        "cross_target_corr": cross_target_corr,
        "quality_flags": quality_flags
    }


def validate_target_quality(
    diagnostics: Dict,
    variance_threshold: float = 1e-6,
    correlation_threshold: float = 0.85
) -> Tuple[bool, List[str]]:
    """
    Validate target quality based on diagnostics.

    Parameters
    ----------
    diagnostics : Dict
        Output from compute_target_diagnostics()
    variance_threshold : float
        Minimum acceptable variance (default 1e-6)
    correlation_threshold : float
        Maximum acceptable cross-target correlation (default 0.85)

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list_of_issues)
        
    Flags:
    - Variance near zero
    - Excessive cross-target correlation
    - Degenerate distributions
    """
    issues: List[str] = []

    # Check per-target statistics
    per_target_stats = diagnostics.get("per_target_stats", {})
    for target, stats in per_target_stats.items():
        std = stats.get("std", np.nan)
        if np.isnan(std):
            issues.append(f"Cannot compute variance for target: {target}")
        elif std < variance_threshold:
            issues.append(
                f"Variance ({std:.2e}) below threshold ({variance_threshold:.2e}) "
                f"for target: {target}"
            )

        # Check for degenerate distributions (concentrated at boundaries)
        mean = stats.get("mean", np.nan)
        p05 = stats.get("p05", np.nan)
        p95 = stats.get("p95", np.nan)

        if not np.isnan(mean):
            if mean < 0.01:
                issues.append(f"Target {target} mean ({mean:.4f}) near zero boundary")
            elif mean > 0.99:
                issues.append(f"Target {target} mean ({mean:.4f}) near one boundary")

        if not np.isnan(p05) and not np.isnan(p95):
            range_val = p95 - p05
            if range_val < 0.1:
                issues.append(
                    f"Target {target} has narrow interquartile range ({range_val:.4f})"
                )

    # Check cross-target correlation
    cross_target_corr = diagnostics.get("cross_target_corr")
    target_names = list(per_target_stats.keys())

    if cross_target_corr is not None and len(target_names) > 1:
        n_targets = len(target_names)
        for i in range(n_targets):
            for j in range(i + 1, n_targets):
                if i < cross_target_corr.shape[0] and j < cross_target_corr.shape[1]:
                    corr_val = abs(cross_target_corr[i, j])
                    if corr_val > correlation_threshold:
                        issues.append(
                            f"Excessive correlation ({corr_val:.3f}) between "
                            f"{target_names[i]} and {target_names[j]} "
                            f"(threshold: {correlation_threshold})"
                        )

    # Include any pre-computed quality flags
    quality_flags = diagnostics.get("quality_flags", [])
    issues.extend(quality_flags)

    is_valid = len(issues) == 0
    return is_valid, issues
