"""Global residual-state discovery contracts for the frozen meta stream.

The module deliberately separates three concerns:

* realized outcomes define historical unreliability episodes;
* pre-entry market features describe and recognize those episodes;
* July and later rows remain evaluation-only when rolling-origin models are fit.

No function in this module mutates model scores or live policy decisions.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm, wasserstein_distance
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from .path_economic_labels import (
    PATH_ECONOMIC_LABEL_COLUMNS,
    materialize_path_economic_labels,
    path_economic_label_manifest,
)

EVENT_SIGNAL_COLUMNS = (
    "signed_hit_surprise",
    "mean_ev_after_cost",
    "bad_mae_rate",
    "timeout_rate",
)


FEATURE_CONCEPTS: dict[str, dict[str, tuple[str, ...]]] = {
    "oi_drawdown_from_peak_24h": {
        "exact": ("oi_drawdown_from_peak_24h",),
        "proxy": ("mkt_median_oi_drawdown_from_peak_24h",),
    },
    "oi_drawdown_from_peak_72h": {
        "exact": ("oi_drawdown_from_peak_72h",),
        "proxy": ("oi_drawdown_from_peak_168h",),
    },
    "oi_recovery_fraction_24h": {
        "exact": ("oi_recovery_fraction_24h",),
        "proxy": ("mkt_median_oi_recovery_fraction_24h",),
    },
    "oi_drop_acceleration_4h": {
        "exact": ("oi_drop_acceleration_4h_rz",),
        "proxy": ("mkt_oi_flush_breadth_accel_1h",),
    },
    "oi_drop_deceleration_4h": {
        "exact": ("oi_drop_deceleration_4h_rz",),
        "proxy": ("mkt_oi_flush_breadth_recovery_4h",),
    },
    "bars_since_max_oi_drop_24h": {
        "exact": ("bars_since_max_oi_drop_24h_norm",),
        "proxy": ("mkt_median_bars_since_max_oi_drop_24h_norm",),
    },
    "price_down_oi_down_1h": {
        "exact": ("price_down_oi_down_1h_rz",),
        "proxy": ("mkt_pct_price_down_oi_down_1h",),
    },
    "price_down_oi_up_1h": {
        "exact": ("price_down_oi_up_1h_rz",),
        "proxy": ("mkt_pct_price_down_oi_up_1h",),
    },
    "price_up_oi_down_1h": {
        "exact": ("price_up_oi_down_1h_rz",),
        "proxy": ("mkt_pct_price_up_oi_down_1h",),
    },
    "price_up_oi_up_1h": {
        "exact": ("price_up_oi_up_1h_rz",),
        "proxy": ("mkt_pct_price_up_oi_up_1h",),
    },
    "price_down_oi_down_4h": {
        "exact": ("price_down_oi_down_4h_rz",),
        "proxy": ("mkt_pct_price_down_oi_down_4h",),
    },
    "price_up_oi_down_4h": {
        "exact": ("price_up_oi_down_4h_rz",),
        "proxy": ("mkt_pct_price_up_oi_down_4h",),
    },
    "price_minus_oi_recovery_24h": {
        "exact": ("price_minus_oi_recovery_24h",),
        "proxy": ("asset_minus_mkt_oi_recovery_fraction_24h",),
    },
    "funding_sign_persistence": {
        "exact": ("funding_sign_persistence_24h", "funding_sign_persistence_72h"),
        "proxy": ("funding_persistence",),
    },
    "funding_sign_flip_age": {
        "exact": ("hours_since_funding_sign_flip_24h_norm",),
        "proxy": ("funding_flip",),
    },
    "funding_crowding_release": {
        "exact": ("funding_crowding_release_4h",),
        "proxy": ("funding_mean_reversion_after_oi_flush",),
    },
    "funding_change_x_oi_flush": {
        "exact": ("funding_flip_x_oi_flush",),
        "proxy": ("funding_crowding_release_4h",),
    },
    "breadth_change_1h": {
        "exact": ("market_breadth_chg_1h",),
        "proxy": ("breadth_chg_1h",),
    },
    "breadth_acceleration": {
        "exact": ("market_breadth_accel_1h",),
        "proxy": ("breadth_accel_1h",),
    },
    "breadth_recovery_from_6h_low": {
        "exact": ("market_breadth_recovery_from_6h_min",),
        "proxy": ("breadth_recovery_from_6h_min",),
    },
    "breadth_drawdown_from_6h_high": {
        "exact": ("market_breadth_drawdown_from_6h_max",),
        "proxy": ("breadth_min_6h",),
    },
    "pct_assets_recovering_from_24h_low": {
        "exact": ("market_pct_recovering_from_24h_low",),
        "proxy": ("pct_assets_recovering_from_intraday_low",),
    },
    "cross_asset_downside_corr": {
        "exact": ("market_downside_pairwise_corr_24h",),
        "proxy": ("cross_asset_downside_corr_1h", "cross_asset_downside_corr_4h"),
    },
    "correlation_jump": {
        "exact": ("cross_asset_corr_chg_1h",),
        "proxy": ("market_downside_corr_minus_unconditional_corr_24h",),
    },
    "first_pc_variance_share": {
        "exact": ("market_pc1_variance_share_12h", "market_pc1_variance_share_24h"),
        "proxy": ("mkt_first_pc_variance_share_1h", "mkt_first_pc_variance_share_4h"),
    },
    "return_dispersion_change": {
        "exact": ("return_dispersion_chg_1h", "return_dispersion_change"),
        "proxy": ("return_dispersion_1h", "return_dispersion_4h"),
    },
    "pct_assets_volume_expanding": {
        "exact": ("pct_assets_volume_z_gt_2", "pct_assets_climax_volume"),
        "proxy": ("market_volume_participation",),
    },
    "cross_sectional_volume_concentration": {
        "exact": ("cross_sectional_volume_concentration", "mkt_volume_concentration"),
        "proxy": ("mkt_quote_volume_z_24h",),
    },
    "oi_concentration": {
        "exact": ("mkt_oi_concentration_btc_eth",),
        "proxy": ("mkt_oi_dispersion_1h", "mkt_oi_dispersion_4h"),
    },
    "oi_change_participation": {
        "exact": ("mkt_pct_oi_chg_4h_rz_lt_minus1",),
        "proxy": ("pct_assets_oi_down_4h",),
    },
    "breadth_volume_divergence": {
        "exact": ("breadth_volume_divergence",),
        "proxy": ("market_breadth_chg_1h", "pct_assets_volume_z_gt_2"),
    },
    "range_per_unit_volume": {
        "exact": ("range_per_unit_volume",),
        "proxy": ("range_climax_decay_4h", "amihud_z"),
    },
    "asset_minus_market_return": {
        "exact": ("asset_minus_mkt_return_1h_rz", "ret48h_bench_resid"),
        "proxy": ("rv_rel_universe",),
    },
    "asset_minus_market_oi_change": {
        "exact": ("asset_minus_mkt_oi_chg_1h_rz", "asset_minus_mkt_oi_chg_4h_rz"),
        "proxy": ("asset_minus_market_oi_1d",),
    },
    "asset_minus_market_oi_drawdown": {
        "exact": ("asset_minus_mkt_oi_drawdown_24h",),
        "proxy": (),
    },
    "asset_minus_market_price_recovery": {
        "exact": ("asset_minus_mkt_price_recovery_fraction_24h",),
        "proxy": ("asset_mkt_exhaustion_phase_divergence",),
    },
    "asset_minus_market_oi_recovery": {
        "exact": ("asset_minus_mkt_oi_recovery_fraction_24h",),
        "proxy": ("asset_minus_mkt_bars_since_oi_flush_24h",),
    },
}


@dataclass(frozen=True)
class ReliabilityEventConfig:
    timestamp_col: str = "__ts__"
    symbol_col: str = "__symbol__"
    side_col: str = "side_name"
    archetype_col: str = "archetype_policy_key"
    hit_col: str = "clean_exec"
    probability_col: str = "hit_probability"
    ev_col: str = "ev_after_1pct"
    bad_mae_col: str = "full_path_bad_mae_1r"
    timeout_col: str = "timeout"
    significance_z: float = 1.96
    causal_min_days: int = 20
    rolling_days: int = 45
    autocorr_days: int = 30
    join_gap_days: int = 1
    min_event_selected_rows: int = 8
    material_negative_ev: float = -0.0015
    payoff_disagreement_ev: float = -0.0010
    payoff_disagreement_clean_tolerance: float = 0.01
    bootstrap_draws: int = 500
    random_state: int = 20260711


@dataclass(frozen=True)
class ReliabilityEventResult:
    daily_cells: pd.DataFrame
    events: pd.DataFrame
    event_membership: pd.DataFrame
    summary: pd.DataFrame
    sensitivity: pd.DataFrame
    manifest: dict[str, Any]


def _numeric(frame: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=np.float64)
    return pd.to_numeric(frame[name], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def _lag1(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    values = values[valid]
    if values.size < 4 or np.std(values[:-1]) <= 1e-12 or np.std(values[1:]) <= 1e-12:
        return np.nan
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def _causal_z(values: pd.Series, min_days: int) -> pd.Series:
    shifted = pd.to_numeric(values, errors="coerce").shift(1)
    mean = shifted.expanding(min_periods=int(min_days)).mean()
    std = shifted.expanding(min_periods=int(min_days)).std(ddof=0).clip(lower=1e-6)
    return ((pd.to_numeric(values, errors="coerce") - mean) / std).clip(-12.0, 12.0)


def _rolling_lag1(values: pd.Series, window: int) -> pd.Series:
    return (
        pd.to_numeric(values, errors="coerce")
        .rolling(int(window), min_periods=max(8, int(window) // 3))
        .apply(_lag1, raw=True)
    )


def benjamini_hochberg(p_values: pd.Series, alpha: float = 0.05) -> pd.DataFrame:
    values = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=np.float64)
    valid_pos = np.flatnonzero(np.isfinite(values))
    q = np.full(len(values), np.nan, dtype=np.float64)
    rejected = np.zeros(len(values), dtype=bool)
    if valid_pos.size:
        order_local = np.argsort(values[valid_pos], kind="stable")
        ordered_pos = valid_pos[order_local]
        ordered = values[ordered_pos]
        adjusted = ordered * float(valid_pos.size) / np.arange(1, valid_pos.size + 1)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0.0, 1.0)
        q[ordered_pos] = adjusted
        rejected[ordered_pos] = adjusted <= float(alpha)
    return pd.DataFrame({"bh_q": q, "bh_reject": rejected}, index=p_values.index)


def build_daily_reliability_cells(
    selected_rows: pd.DataFrame,
    config: ReliabilityEventConfig | None = None,
) -> pd.DataFrame:
    """Aggregate selected trades into daily side/archetype reliability cells."""
    cfg = config or ReliabilityEventConfig()
    frame = selected_rows.copy(deep=False)
    frame[cfg.timestamp_col] = pd.to_datetime(
        frame[cfg.timestamp_col], utc=True, errors="coerce"
    )
    frame = frame.loc[frame[cfg.timestamp_col].notna()].copy()
    frame["day"] = frame[cfg.timestamp_col].dt.floor("D")
    path_labels = materialize_path_economic_labels(frame)
    for name in PATH_ECONOMIC_LABEL_COLUMNS:
        frame[name] = path_labels[name].to_numpy(dtype=np.float32, copy=False)
    hit = _numeric(frame, cfg.hit_col)
    probability = _numeric(frame, cfg.probability_col).clip(1e-5, 1.0 - 1e-5)
    frame["_residual"] = hit - probability
    frame["_variance"] = probability * (1.0 - probability)
    frame["_loss"] = (-_numeric(frame, cfg.ev_col)).clip(lower=0.0)
    frame["_positive_ev"] = _numeric(frame, cfg.ev_col).clip(lower=0.0)
    frame["_negative_ev"] = (-_numeric(frame, cfg.ev_col)).clip(lower=0.0)
    groups = ["day", cfg.side_col, cfg.archetype_col]
    cells = (
        frame.groupby(groups, observed=True, dropna=False, sort=True)
        .agg(
            selected_rows=(cfg.timestamp_col, "size"),
            distinct_assets=(cfg.symbol_col, "nunique"),
            signed_hit_surprise=("_residual", "mean"),
            residual_sum=("_residual", "sum"),
            residual_variance=("_variance", "sum"),
            expected_clean_rate=(cfg.probability_col, "mean"),
            clean_rate=(cfg.hit_col, "mean"),
            mean_ev_after_cost=(cfg.ev_col, "mean"),
            worst_ev=(cfg.ev_col, "min"),
            loss_rate=(
                cfg.ev_col,
                lambda values: float(
                    (_numeric(pd.DataFrame({"x": values}), "x") < 0.0).mean()
                ),
            ),
            bad_mae_rate=(cfg.bad_mae_col, "mean"),
            timeout_rate=(cfg.timeout_col, "mean"),
            loss_mean=("_loss", "mean"),
            mean_positive_ev=("_positive_ev", "mean"),
            mean_negative_ev=("_negative_ev", "mean"),
            acute_adverse_rate=("path_label_acute_adverse", "mean"),
            slow_timeout_loss_rate=("path_label_slow_timeout_loss", "mean"),
            clean_negative_ev_rate=("path_label_clean_negative_ev", "mean"),
            dirty_negative_ev_rate=("path_label_dirty_negative_ev", "mean"),
            durable_clean_positive_rate=("path_label_durable_clean_positive", "mean"),
        )
        .reset_index()
    )
    cells["surprise_z"] = cells["residual_sum"] / np.sqrt(
        cells["residual_variance"].clip(lower=1e-6)
    )
    cells["surprise_p"] = 2.0 * norm.sf(cells["surprise_z"].abs())
    cells["payoff_asymmetry"] = pd.to_numeric(
        cells["mean_positive_ev"], errors="coerce"
    ) - pd.to_numeric(cells["mean_negative_ev"], errors="coerce")
    output: list[pd.DataFrame] = []
    for _, group in cells.groupby(
        [cfg.side_col, cfg.archetype_col], observed=True, dropna=False, sort=True
    ):
        group = group.sort_values("day", kind="stable").copy()
        z_columns = {
            "mean_ev_after_cost": "ev_z",
            "bad_mae_rate": "bad_mae_z",
            "timeout_rate": "timeout_z",
            "acute_adverse_rate": "acute_adverse_z",
            "slow_timeout_loss_rate": "slow_timeout_loss_z",
            "clean_negative_ev_rate": "clean_negative_ev_z",
        }
        for name, destination in z_columns.items():
            group[destination] = _causal_z(group[name], cfg.causal_min_days)
        group["surprise_ema_3d"] = (
            group["signed_hit_surprise"].ewm(span=3, adjust=False).mean()
        )
        group["surprise_ema_5d"] = (
            group["signed_hit_surprise"].ewm(span=5, adjust=False).mean()
        )
        group["ev_ema_3d"] = (
            group["mean_ev_after_cost"].ewm(span=3, adjust=False).mean()
        )
        group["negative_ev_days_3"] = (
            group["mean_ev_after_cost"].lt(0.0).rolling(3, min_periods=1).sum()
        )
        group["surprise_ema_3d_z"] = _causal_z(
            group["surprise_ema_3d"], cfg.causal_min_days
        )
        group["surprise_ac_rolling"] = _rolling_lag1(
            group["signed_hit_surprise"], cfg.autocorr_days
        )
        group["loss_ac_rolling"] = _rolling_lag1(group["loss_mean"], cfg.autocorr_days)
        output.append(group)
    result = pd.concat(output, ignore_index=True) if output else cells
    bh = benjamini_hochberg(result["surprise_p"])
    result["surprise_bh_q"] = bh["bh_q"].to_numpy()
    result["surprise_bh_reject"] = bh["bh_reject"].to_numpy()
    return result.sort_values(groups, kind="stable").reset_index(drop=True)


def _event_flags(
    cells: pd.DataFrame,
    z_threshold: float,
    config: ReliabilityEventConfig | None = None,
) -> pd.DataFrame:
    cfg = config or ReliabilityEventConfig()
    out = cells.copy()
    persistent_sign = np.sign(out["surprise_ema_3d"]) == np.sign(out["surprise_z"])
    out["event_a_surprise_persistence"] = (
        out["surprise_z"].abs().ge(float(z_threshold))
        & persistent_sign
        & out["surprise_ema_3d_z"].abs().ge(0.5)
    )
    out["event_b_persistent_loss"] = out["mean_ev_after_cost"].lt(0.0) & (
        out["ev_z"].le(-float(z_threshold))
        | (
            out["mean_ev_after_cost"].le(-0.002)
            & out["negative_ev_days_3"].ge(2.0)
            & out["ev_ema_3d"].lt(0.0)
            & out["loss_ac_rolling"].fillna(0.0).gt(0.10)
        )
    )
    out["event_c_bad_path_cluster"] = out["bad_mae_z"].ge(float(z_threshold)) | out[
        "timeout_z"
    ].ge(float(z_threshold))
    out["event_d_calibration_economics_disagreement"] = out["surprise_z"].ge(0.5) & out[
        "mean_ev_after_cost"
    ].le(float(cfg.payoff_disagreement_ev))
    # A daily cell can be clean-rate positive yet economically negative because
    # its losers are materially larger than its wins. Keep this separate from a
    # classifier miss: it is the payoff/path failure the state layer needs to
    # recognize later from observable inputs.
    out["event_e_material_negative_ev"] = (
        out["mean_ev_after_cost"].le(float(cfg.material_negative_ev))
        & out["selected_rows"].ge(int(cfg.min_event_selected_rows))
        & (
            out["ev_z"].le(-0.75)
            | (out["negative_ev_days_3"].ge(2.0) & out["ev_ema_3d"].lt(0.0))
        )
    )
    out["event_f_payoff_asymmetry"] = (
        out["mean_ev_after_cost"].le(float(cfg.payoff_disagreement_ev))
        & out["selected_rows"].ge(int(cfg.min_event_selected_rows))
        & out["clean_rate"].ge(
            out["expected_clean_rate"] - float(cfg.payoff_disagreement_clean_tolerance)
        )
        & out["payoff_asymmetry"].le(float(cfg.payoff_disagreement_ev))
    )
    # These labels are mutually exclusive per realized row. They make a
    # persistent loss interpretable as acute adverse path, slow resolution, or
    # clean-but-negative exit capture rather than one generic failure bucket.
    out["event_g_acute_adverse_path"] = out["acute_adverse_z"].ge(float(z_threshold))
    out["event_h_slow_timeout_loss"] = out["slow_timeout_loss_z"].ge(float(z_threshold))
    out["event_i_clean_negative_ev"] = out["clean_negative_ev_z"].ge(
        float(z_threshold)
    ) & out["mean_ev_after_cost"].lt(0.0)
    flag_cols = [name for name in out.columns if name.startswith("event_")]
    out["active"] = out[flag_cols].any(axis=1)
    out["surprise_sign"] = np.select(
        [out["surprise_z"].gt(0.0), out["surprise_z"].lt(0.0)],
        ["positive", "negative"],
        default="neutral",
    )
    out["evidence_type"] = out.apply(
        lambda row: "|".join(
            name.removeprefix("event_") for name in flag_cols if bool(row[name])
        ),
        axis=1,
    )
    out["primary_event_type"] = np.select(
        [
            out["event_g_acute_adverse_path"],
            out["event_h_slow_timeout_loss"],
            out["event_i_clean_negative_ev"],
            out["event_d_calibration_economics_disagreement"],
            out["event_f_payoff_asymmetry"],
            out["event_e_material_negative_ev"],
            out["event_b_persistent_loss"],
            out["event_c_bad_path_cluster"],
            out["event_a_surprise_persistence"] & out["surprise_z"].lt(0.0),
            out["event_a_surprise_persistence"] & out["surprise_z"].ge(0.0),
        ],
        [
            "acute_adverse_path",
            "slow_timeout_loss",
            "clean_negative_ev",
            "payoff_disagreement",
            "payoff_asymmetry",
            "material_negative_ev",
            "persistent_loss",
            "bad_path_cluster",
            "negative_surprise",
            "positive_surprise",
        ],
        default="inactive",
    )
    return out


def _bootstrap_event(
    group: pd.DataFrame,
    *,
    draws: int,
    random_state: int,
) -> dict[str, float | bool]:
    paired = (
        group[["signed_hit_surprise", "mean_ev_after_cost"]]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
    )
    surprise = paired["signed_hit_surprise"].to_numpy(dtype=float)
    ev = paired["mean_ev_after_cost"].to_numpy(dtype=float)
    if min(len(surprise), len(ev)) < 2:
        return {
            "bootstrap_surprise_ci025": np.nan,
            "bootstrap_surprise_ci975": np.nan,
            "bootstrap_ev_ci025": np.nan,
            "bootstrap_ev_ci975": np.nan,
            "bootstrap_survival": False,
        }
    rng = np.random.default_rng(int(random_state))
    surprise_boot = np.empty(int(draws), dtype=np.float64)
    ev_boot = np.empty(int(draws), dtype=np.float64)
    block = min(3, len(surprise), len(ev))
    for idx in range(int(draws)):
        surprise_starts = rng.integers(
            0, len(surprise), size=math.ceil(len(surprise) / block)
        )
        ev_starts = surprise_starts
        surprise_positions = np.concatenate(
            [
                np.arange(start, start + block) % len(surprise)
                for start in surprise_starts
            ]
        )[: len(surprise)]
        ev_positions = np.concatenate(
            [np.arange(start, start + block) % len(ev) for start in ev_starts]
        )[: len(ev)]
        surprise_boot[idx] = np.mean(surprise[surprise_positions])
        ev_boot[idx] = np.mean(ev[ev_positions])
    s_low, s_high = np.quantile(surprise_boot, [0.025, 0.975])
    e_low, e_high = np.quantile(ev_boot, [0.025, 0.975])
    survives = bool((s_low > 0.0) or (s_high < 0.0) or (e_high < 0.0))
    return {
        "bootstrap_surprise_ci025": float(s_low),
        "bootstrap_surprise_ci975": float(s_high),
        "bootstrap_ev_ci025": float(e_low),
        "bootstrap_ev_ci975": float(e_high),
        "bootstrap_survival": survives,
        "bootstrap_block_days": int(block),
    }


def detect_unreliability_events(
    daily_cells: pd.DataFrame,
    config: ReliabilityEventConfig | None = None,
    *,
    z_threshold: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config or ReliabilityEventConfig()
    threshold = float(cfg.significance_z if z_threshold is None else z_threshold)
    cells = _event_flags(daily_cells, threshold, cfg)
    active = cells.loc[cells["active"]].copy()
    if active.empty:
        return pd.DataFrame(), active
    # Events are intentionally local to one inference population.  Joining an
    # episode merely because two archetypes fail on the same side/day mixes
    # different base error mechanisms and invalidates local matched controls.
    local_days = (
        active.groupby(
            [cfg.side_col, cfg.archetype_col, "day", "primary_event_type"],
            observed=True,
            sort=True,
        )
        .agg(
            evidence_types=(
                "evidence_type",
                lambda values: tuple(
                    sorted(set("|".join(map(str, values)).split("|")))
                ),
            ),
            surprise_sign=(
                "surprise_z",
                lambda values: "positive"
                if float(np.nanmean(values)) >= 0.0
                else "negative",
            ),
        )
        .reset_index()
    )
    event_by_local_day: dict[tuple[str, str, pd.Timestamp, str], int] = {}
    next_id = 0
    for (side, archetype, primary_type), group in local_days.groupby(
        [cfg.side_col, cfg.archetype_col, "primary_event_type"],
        observed=True,
        sort=True,
    ):
        previous_day: pd.Timestamp | None = None
        previous_evidence: set[str] = set()
        previous_sign = ""
        current_id = -1
        for row in group.sort_values("day", kind="stable").itertuples(index=False):
            day = pd.Timestamp(getattr(row, "day"))
            evidence = set(getattr(row, "evidence_types"))
            sign = str(getattr(row, "surprise_sign"))
            gap = (day - previous_day).days if previous_day is not None else 10_000
            compatible = sign == previous_sign or bool(evidence & previous_evidence)
            if gap > int(cfg.join_gap_days) + 1 or not compatible:
                current_id = next_id
                next_id += 1
            event_by_local_day[(str(side), str(archetype), day, str(primary_type))] = (
                current_id
            )
            previous_day = day
            previous_evidence = evidence
            previous_sign = sign
    active["event_id"] = [
        event_by_local_day[
            (str(side), str(archetype), pd.Timestamp(day), str(primary_type))
        ]
        for side, archetype, day, primary_type in zip(
            active[cfg.side_col],
            active[cfg.archetype_col],
            active["day"],
            active["primary_event_type"],
            strict=True,
        )
    ]
    rows: list[dict[str, Any]] = []
    for event_id, group in active.groupby("event_id", sort=True):
        mean_z = float(group["surprise_z"].mean())
        mean_ev = float(group["mean_ev_after_cost"].mean())
        duration = int((group["day"].max() - group["day"].min()).days + 1)
        selected_rows = int(group["selected_rows"].sum())
        local_ac = float(
            pd.to_numeric(group["surprise_ac_rolling"], errors="coerce").abs().max()
        )
        if not np.isfinite(local_ac):
            local_ac = 0.0
        priority = (
            abs(mean_z)
            * math.sqrt(max(duration, 1))
            * math.sqrt(math.log1p(max(selected_rows, 0)))
            * (1.0 + abs(local_ac))
        )
        if mean_ev < 0.0:
            priority *= 1.0 + min(abs(mean_ev) / 0.005, 3.0)
        evidence_values = set("|".join(group["evidence_type"]).split("|"))
        has_bad_path = "c_bad_path_cluster" in evidence_values
        primary_type = str(group["primary_event_type"].iloc[0])
        if primary_type in {
            "clean_negative_ev",
            "payoff_disagreement",
            "payoff_asymmetry",
        }:
            event_class = "payoff_disagreement"
        elif mean_z < 0.0 or mean_ev < 0.0 or has_bad_path:
            event_class = "adverse"
        else:
            event_class = "opportunity"
        adverse_priority = (
            priority if event_class in {"adverse", "payoff_disagreement"} else 0.0
        )
        opportunity_priority = priority if event_class == "opportunity" else 0.0
        bootstrap = _bootstrap_event(
            group,
            draws=cfg.bootstrap_draws,
            random_state=cfg.random_state + int(event_id),
        )
        distinct_arches = sorted(set(map(str, group[cfg.archetype_col])))
        if len(distinct_arches) != 1:
            raise RuntimeError("Local event construction unexpectedly mixed archetypes")
        distinct_assets = int(group["distinct_assets"].sum())
        discovery_eligible = bool(
            duration >= 2
            or mean_ev <= -0.002
            or distinct_assets >= 20
            or selected_rows >= max(2 * int(cfg.min_event_selected_rows), 20)
        )
        rows.append(
            {
                "event_id": f"URE-{int(event_id):04d}",
                "event_start": group["day"].min(),
                "event_end": group["day"].max(),
                "peak_timestamp": group.loc[group["surprise_z"].abs().idxmax(), "day"],
                "side_name": str(group[cfg.side_col].iloc[0]),
                cfg.archetype_col: distinct_arches[0],
                "affected_archetypes": "|".join(distinct_arches),
                "surprise_sign": "positive" if mean_z >= 0.0 else "negative",
                "mean_surprise": float(group["signed_hit_surprise"].mean()),
                "mean_surprise_z": mean_z,
                "mean_ev": mean_ev,
                "worst_ev": float(group["worst_ev"].min()),
                "loss_rate": float(
                    np.average(group["loss_rate"], weights=group["selected_rows"])
                ),
                "bad_mae_rate": float(
                    np.average(group["bad_mae_rate"], weights=group["selected_rows"])
                ),
                "timeout_rate": float(
                    np.average(group["timeout_rate"], weights=group["selected_rows"])
                ),
                "acute_adverse_rate": float(
                    np.average(
                        group["acute_adverse_rate"], weights=group["selected_rows"]
                    )
                ),
                "slow_timeout_loss_rate": float(
                    np.average(
                        group["slow_timeout_loss_rate"], weights=group["selected_rows"]
                    )
                ),
                "clean_negative_ev_rate": float(
                    np.average(
                        group["clean_negative_ev_rate"], weights=group["selected_rows"]
                    )
                ),
                "durable_clean_positive_rate": float(
                    np.average(
                        group["durable_clean_positive_rate"],
                        weights=group["selected_rows"],
                    )
                ),
                "autocorrelation_strength": local_ac,
                "selected_rows": selected_rows,
                "distinct_assets": distinct_assets,
                "event_duration_days": duration,
                "state_failure_mechanism": primary_type,
                "evidence_type": "|".join(
                    sorted(set("|".join(group["evidence_type"]).split("|")))
                ),
                "event_priority": float(priority),
                "event_class": event_class,
                "adverse_priority": float(adverse_priority),
                "opportunity_priority": float(opportunity_priority),
                "bh_surviving_cells": int(group["surprise_bh_reject"].sum()),
                "discovery_eligible": discovery_eligible,
                **bootstrap,
            }
        )
    events = pd.DataFrame(rows).sort_values(
        "event_priority", ascending=False, kind="stable"
    )
    id_map = {int(value.split("-")[-1]): value for value in events["event_id"]}
    active["event_id"] = active["event_id"].map(id_map)
    return events.reset_index(drop=True), active.reset_index(drop=True)


def event_definition_sensitivity(
    daily_cells: pd.DataFrame,
    config: ReliabilityEventConfig | None = None,
    thresholds: Sequence[float] = (1.96, 2.33, 2.58),
) -> pd.DataFrame:
    cfg = config or ReliabilityEventConfig()
    memberships: dict[float, set[str]] = {}
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        flagged = _event_flags(daily_cells, float(threshold), cfg)
        keys = set(
            flagged.loc[flagged["active"]].apply(
                lambda row: f"{row['day']}|{row[cfg.side_col]}|{row[cfg.archetype_col]}",
                axis=1,
            )
        )
        memberships[float(threshold)] = keys
    reference = memberships[float(thresholds[0])]
    for threshold, values in memberships.items():
        union = reference | values
        rows.append(
            {
                "z_threshold": threshold,
                "active_cells": len(values),
                "jaccard_vs_1_96": float(len(reference & values) / max(len(union), 1)),
            }
        )
    return pd.DataFrame(rows)


def summarize_events(
    events: pd.DataFrame,
    membership: pd.DataFrame,
    config: ReliabilityEventConfig | None = None,
) -> pd.DataFrame:
    cfg = config or ReliabilityEventConfig()
    if events.empty:
        return pd.DataFrame()
    exploded = membership[
        ["event_id", cfg.side_col, cfg.archetype_col]
    ].drop_duplicates()
    merged = exploded.merge(
        events,
        on=["event_id", cfg.side_col, cfg.archetype_col],
        how="left",
    )
    return (
        merged.groupby([cfg.side_col, cfg.archetype_col], observed=True, dropna=False)
        .agg(
            event_count=("event_id", "nunique"),
            median_duration_days=("event_duration_days", "median"),
            median_distinct_assets=("distinct_assets", "median"),
            negative_ev_event_fraction=(
                "mean_ev",
                lambda values: float((values < 0.0).mean()),
            ),
            bootstrap_survival_fraction=("bootstrap_survival", "mean"),
            discovery_eligible_fraction=("discovery_eligible", "mean"),
        )
        .reset_index()
    )


def discover_reliability_events(
    selected_rows: pd.DataFrame,
    config: ReliabilityEventConfig | None = None,
) -> ReliabilityEventResult:
    cfg = config or ReliabilityEventConfig()
    cells = build_daily_reliability_cells(selected_rows, cfg)
    events, membership = detect_unreliability_events(cells, cfg)
    sensitivity = event_definition_sensitivity(cells, cfg)
    summary = summarize_events(events, membership, cfg)
    manifest = {
        "schema": "global_residual_unreliability_events_v1",
        "config": asdict(cfg),
        "daily_cells": int(len(cells)),
        "events": int(len(events)),
        "discovery_eligible_events": int(
            events.get("discovery_eligible", pd.Series(dtype=bool)).sum()
        ),
        "leakage_contract": (
            "Events use realized outcomes for retrospective discovery only; no event flag is an inference feature."
        ),
        "path_economic_label_taxonomy": path_economic_label_manifest(),
    }
    return ReliabilityEventResult(
        cells, events, membership, summary, sensitivity, manifest
    )


def audit_feature_concepts(
    available_columns: Iterable[str],
    *,
    configured_columns: Iterable[str] = (),
) -> pd.DataFrame:
    available = set(map(str, available_columns))
    configured = set(map(str, configured_columns))
    rows: list[dict[str, Any]] = []
    for concept, aliases in FEATURE_CONCEPTS.items():
        exact = [name for name in aliases.get("exact", ()) if name in available]
        proxy = [name for name in aliases.get("proxy", ()) if name in available]
        configured_exact = [
            name for name in aliases.get("exact", ()) if name in configured
        ]
        if exact:
            status = "present_exactly"
            matched = exact
        elif proxy:
            status = "present_as_close_proxy"
            matched = proxy
        elif configured_exact:
            status = "unreliable_coverage"
            matched = configured_exact
        else:
            status = "missing"
            matched = []
        rows.append(
            {
                "concept": concept,
                "status": status,
                "matched_features": "|".join(matched),
                "exact_candidates": "|".join(aliases.get("exact", ())),
                "proxy_candidates": "|".join(aliases.get("proxy", ())),
            }
        )
    return pd.DataFrame(rows)


def feature_quality_metrics(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    timestamp_col: str = "__ts__",
    symbol_col: str = "__symbol__",
    july_start: str = "2026-07-01",
) -> pd.DataFrame:
    """Return portable quality diagnostics for a bounded feature sample."""
    ts = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
    rows: list[dict[str, Any]] = []
    for name in feature_columns:
        if name not in frame.columns:
            continue
        values = pd.to_numeric(frame[name], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        # Pandas preserves a native bool dtype through to quantile, where NumPy
        # cannot interpolate booleans. Quality diagnostics treat validity flags
        # as their portable 0/1 representation.
        if pd.api.types.is_bool_dtype(values.dtype):
            values = values.astype(np.float32)
        finite = values.dropna()
        if finite.empty:
            rows.append(
                {"feature": name, "coverage": 0.0, "status": "unreliable_coverage"}
            )
            continue
        q01, q99 = finite.quantile([0.01, 0.99]).tolist()
        zero_rate = float(values.fillna(np.nan).eq(0.0).mean())
        if symbol_col in frame.columns:
            stale = float(
                frame.assign(_value=values)
                .sort_values([symbol_col, timestamp_col], kind="stable")
                .groupby(symbol_col, observed=True)["_value"]
                .apply(lambda series: series.eq(series.shift(1)).mean())
                .mean()
            )
        else:
            ordered = pd.DataFrame({timestamp_col: ts, "_value": values}).sort_values(
                timestamp_col, kind="stable"
            )["_value"]
            stale = float(ordered.eq(ordered.shift(1)).mean())
        train = values.loc[ts.lt(pd.Timestamp(july_start, tz="UTC"))].dropna()
        july = values.loc[ts.ge(pd.Timestamp(july_start, tz="UTC"))].dropna()
        psi = np.nan
        if len(train) >= 100 and len(july) >= 50:
            edges = np.unique(np.nanquantile(train, np.linspace(0.0, 1.0, 11)))
            if len(edges) >= 3:
                train_hist = np.histogram(train, bins=edges)[0].astype(float) + 1e-6
                july_hist = np.histogram(july, bins=edges)[0].astype(float) + 1e-6
                train_hist /= train_hist.sum()
                july_hist /= july_hist.sum()
                psi = float(
                    np.sum((july_hist - train_hist) * np.log(july_hist / train_hist))
                )
        symbol_means = (
            frame.assign(_value=values)
            .groupby(symbol_col, observed=True)["_value"]
            .mean()
            if symbol_col in frame.columns
            else pd.Series(dtype=float)
        )
        between_var = (
            float(np.nanvar(symbol_means.to_numpy(dtype=float)))
            if not symbol_means.empty
            else np.nan
        )
        total_var = float(np.nanvar(finite.to_numpy(dtype=float)))
        asset_variance_share = (
            between_var / max(total_var, 1e-12) if np.isfinite(between_var) else np.nan
        )
        rows.append(
            {
                "feature": name,
                "coverage": float(values.notna().mean()),
                "stale_value_rate": stale,
                "zero_rate": zero_rate,
                "clip_tail_rate": float(((values <= q01) | (values >= q99)).mean()),
                "q01": float(q01),
                "q99": float(q99),
                "train_july_psi": psi,
                "asset_between_variance_share": asset_variance_share,
                "reject_missing_gt_20pct": bool(values.notna().mean() < 0.80),
                "reject_psi_gt_0_25": bool(np.isfinite(psi) and psi > 0.25),
                "reject_asset_scale_concentration": bool(
                    np.isfinite(asset_variance_share) and asset_variance_share > 0.25
                ),
            }
        )
    return pd.DataFrame(rows)


def matched_control_feature_diagnostics(
    observations: pd.DataFrame,
    feature_columns: Sequence[str],
    match_columns: Sequence[str],
    *,
    event_col: str = "is_event",
    neighbors: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match event observations to reliable controls and score candidate features."""
    work = observations.copy()
    event = work[event_col].fillna(False).astype(bool)
    match = work.reindex(columns=match_columns).apply(pd.to_numeric, errors="coerce")
    medians = match.loc[~event].median().fillna(0.0)
    match = match.fillna(medians)
    scale = match.loc[~event].std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    x = ((match - medians) / scale).to_numpy(dtype=np.float32)
    event_pos = np.flatnonzero(event.to_numpy())
    control_pos = np.flatnonzero((~event).to_numpy())
    if not len(event_pos) or not len(control_pos):
        return pd.DataFrame(), pd.DataFrame()
    model = NearestNeighbors(n_neighbors=min(int(neighbors), len(control_pos))).fit(
        x[control_pos]
    )
    _, local = model.kneighbors(x[event_pos])
    matched_control_pos = control_pos[local.reshape(-1)]
    matched = pd.DataFrame(
        {
            "event_row": np.repeat(event_pos, local.shape[1]),
            "control_row": matched_control_pos,
        }
    )
    diagnostics: list[dict[str, Any]] = []
    for name in feature_columns:
        if name not in work.columns:
            continue
        values = pd.to_numeric(work[name], errors="coerce")
        left = values.iloc[matched["event_row"].to_numpy()].to_numpy(dtype=float)
        right = values.iloc[matched["control_row"].to_numpy()].to_numpy(dtype=float)
        valid = np.isfinite(left) & np.isfinite(right)
        if valid.sum() < 20:
            continue
        left = left[valid]
        right = right[valid]
        pooled = math.sqrt(max((np.var(left) + np.var(right)) / 2.0, 1e-12))
        labels = np.concatenate([np.ones(len(left)), np.zeros(len(right))])
        feature_values = np.concatenate([left, right])
        auc = roc_auc_score(labels, feature_values)
        auc = max(float(auc), 1.0 - float(auc))
        mi = mutual_info_classif(
            feature_values.reshape(-1, 1), labels.astype(int), random_state=52
        )[0]
        diagnostics.append(
            {
                "feature": name,
                "matched_pairs": int(len(left)),
                "standardized_mean_difference": float(
                    (np.mean(left) - np.mean(right)) / pooled
                ),
                "wasserstein_distance": float(wasserstein_distance(left, right)),
                "univariate_event_auc": auc,
                "conditional_mutual_information_proxy": float(mi),
                "incremental_matched_control_auc": float(auc - 0.5),
            }
        )
    return pd.DataFrame(diagnostics), matched
