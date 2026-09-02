#!/usr/bin/env python3
"""Causal state-aware EV-map ablation on frozen meta OOS predictions.

This is deliberately a *map* experiment, not a new selector.  Every arm keeps
the same top-10% global-per-timestamp budget.  At each UTC day it fits a small
ridge residual correction using only resolved rows before that day, then ranks
that day's candidates by mapped EV plus the bounded correction.

The source ledger contains the frozen residual-event AE/GMM feature family.
It predates the newer ``meta_resid_arch_*`` naming, so the manifest records
the exact equivalent columns used here.  No realized residual or hit-rate
value from the scored day is ever a feature for that day.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SOURCE = Path(
    "data_perp/reports/meta_v9_recovery_20260717/"
    "residual_state_mda95_hier_newaegmm_frozenpremarch_v2/"
    "canonical_v9_mlp_hierev_ablation_v1/postprocessed_oos_predictions.parquet"
)
DEFAULT_OUT = Path("data_perp/reports/ev_map_state_adjustment_ablation_20260720_v1")
REFIT_CADENCE_HOURS = 6

BASE_EV = "expected_net_ev_after_1pct"
BASE_RANK = "expected_ev_rank_score"
EV = "ev_after_1pct"
KEYS = ("side_name", "archetype_policy_key")

# The frozen V9 ledger uses these names for the residual-archetype state.
# They are one-to-one semantic substitutes for the newer meta_resid_arch_ names.
ALIASES = {
    "meta_resid_arch_support_log1p": "resid_event_aegmm_local_support_log1p",
    "meta_resid_arch_entropy": "resid_event_aegmm_gmm_entropy",
    "meta_resid_arch_expected_hit_surprise": "resid_event_aegmm_expected_market_peer_surprise",
    "expected_hit_surprise": "resid_event_aegmm_expected_ev_timestamp_neutral_surprise",
}

CONTEXT = list(ALIASES)
STATE_BASE = [
    "resid_event_aegmm_posterior_speed",
    "resid_event_aegmm_posterior_acceleration",
    "resid_event_aegmm_dae_reconstruction_error_zscore",
    "rv_ratio_6_24",
    "volume_z_24",
    "ob_spread_bps_z_24h",
    "ob_notional_to_depth_l20_z_24h",
]


@dataclass(frozen=True)
class Arm:
    name: str
    lookback_hours: int
    recon_z_threshold: float
    feature_mode: str
    max_down: float
    max_up: float


def _numeric(frame: pd.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
    if name not in frame:
        return np.full(len(frame), default, dtype=np.float32)
    return (
        pd.to_numeric(frame[name], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(default)
        .to_numpy(dtype=np.float32)
    )


def _materialize_context_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    """Prefer the current contract, otherwise name the frozen V9 equivalent."""
    for canonical, fallback in ALIASES.items():
        if canonical not in frame.columns and fallback in frame.columns:
            frame[canonical] = frame[fallback].astype(np.float32, copy=False)
    return frame


def _previous_window_features(frame: pd.DataFrame, lookback_hours: int) -> pd.DataFrame:
    """Add causal side/archetype recent surprise and expected-state summaries."""
    out = frame.copy()
    out["decision_ts"] = pd.to_datetime(out["__ts__"], utc=True).dt.floor("h")
    group = out[list(KEYS) + ["decision_ts"]].copy()
    group["actual_hr_surprise"] = _numeric(out, "clean_exec") - np.clip(
        _numeric(out, "score_regime_calibrated", 0.5), 0.0, 1.0
    )
    group["expected_hr_surprise"] = _numeric(
        out, "meta_resid_arch_expected_hit_surprise"
    )
    group["entropy"] = _numeric(out, "meta_resid_arch_entropy", 1.0)
    daily = (
        group.groupby(list(KEYS) + ["decision_ts"], observed=True, sort=True)[
            ["actual_hr_surprise", "expected_hr_surprise", "entropy"]
        ]
        .mean()
        .reset_index()
    )
    for name in ("actual_hr_surprise", "expected_hr_surprise", "entropy"):
        col = f"prior_{lookback_hours}h_{name}"
        values = []
        for _, local in daily.groupby(list(KEYS), observed=True, sort=False):
            local = local.sort_values("decision_ts", kind="stable").copy()
            # shift one complete decision timestamp: neither the current row
            # nor another symbol's realized outcome at the same bar is visible.
            local[col] = (
                local.set_index("decision_ts")[name]
                .shift(1)
                .rolling(f"{lookback_hours}h", min_periods=1)
                .mean()
                .to_numpy(dtype=np.float32)
            )
            values.append(local)
        daily = pd.concat(values, ignore_index=True)
    cols = list(KEYS) + ["decision_ts"] + [c for c in daily if c.startswith("prior_")]
    return out.merge(daily[cols], on=list(KEYS) + ["decision_ts"], how="left", sort=False)


def _state_matrix(frame: pd.DataFrame, arm: Arm) -> tuple[np.ndarray, list[str]]:
    """Build one observable state family at a time for attribution."""
    recon = _numeric(frame, "resid_event_aegmm_dae_reconstruction_error_zscore")
    prior = (
        frame.assign(__recon__=recon)
        .sort_values(["__symbol__", "side_name", "__ts__"], kind="stable")
        .groupby(["__symbol__", "side_name"], observed=True)["__recon__"]
        .shift(1).reindex(frame.index).fillna(0.0).to_numpy(dtype=np.float32)
    )
    accel = recon - prior
    # The persistence threshold is causal and local to side x archetype: at
    # timestamp t it is the upper reconstruction-error tail from prior bars,
    # not a fixed global z-score and not a threshold using t's outcomes.
    tail_probability = (
        arm.recon_z_threshold if arm.feature_mode == "ae_recon_persistence" else 0.02
    )
    threshold_state = frame[["side_name", "archetype_policy_key", "decision_ts"]].copy()
    threshold_state["__recon__"] = recon
    threshold_state = (
        threshold_state.groupby(["side_name", "archetype_policy_key", "decision_ts"], observed=True)["__recon__"]
        .median().reset_index().sort_values(["side_name", "archetype_policy_key", "decision_ts"], kind="stable")
    )
    threshold_parts = []
    for _, local in threshold_state.groupby(["side_name", "archetype_policy_key"], observed=True, sort=False):
        local = local.copy()
        local["__threshold__"] = (
            local.set_index("decision_ts")["__recon__"].shift(1)
            .rolling(f"{arm.lookback_hours}h", min_periods=4)
            .quantile(1.0 - tail_probability)
            .to_numpy(dtype=np.float32)
        )
        threshold_parts.append(local)
    threshold_state = pd.concat(threshold_parts, ignore_index=True)
    temp = frame[["__symbol__", "side_name", "archetype_policy_key", "__ts__", "decision_ts"]].copy()
    temp = temp.merge(
        threshold_state[["side_name", "archetype_policy_key", "decision_ts", "__threshold__"]],
        on=["side_name", "archetype_policy_key", "decision_ts"], how="left", sort=False,
    )
    temp["__above__"] = (recon >= temp["__threshold__"].fillna(np.inf).to_numpy(dtype=np.float32)).astype(np.float32)
    temp = temp.sort_values(["__symbol__", "side_name", "__ts__"], kind="stable")
    temp["__persist__"] = temp.groupby(["__symbol__", "side_name"], observed=True)["__above__"].transform(
        lambda s: s.rolling(arm.lookback_hours, min_periods=1).mean()
    )
    persistence = temp["__persist__"].reindex(frame.index).fillna(0.0).to_numpy(dtype=np.float32)
    vol_volume = np.maximum(_numeric(frame, "rv_ratio_6_24") - 1.0, 0.0) * np.maximum(
        _numeric(frame, "volume_z_24"), 0.0
    )
    liquidity_shock = np.maximum(_numeric(frame, "ob_spread_bps_z_24h"), 0.0) + np.maximum(
        _numeric(frame, "ob_notional_to_depth_l20_z_24h"), 0.0
    )
    families = {
        "support": (["meta_resid_arch_support_log1p"], [_numeric(frame, "meta_resid_arch_support_log1p")]),
        "entropy": (["meta_resid_arch_entropy", f"prior_{arm.lookback_hours}h_entropy"], [_numeric(frame, "meta_resid_arch_entropy", 1.0), _numeric(frame, f"prior_{arm.lookback_hours}h_entropy", 1.0)]),
        "hit_surprise_actual": ([f"prior_{arm.lookback_hours}h_actual_hr_surprise"], [_numeric(frame, f"prior_{arm.lookback_hours}h_actual_hr_surprise")]),
        "hit_surprise_expected": (["expected_hit_surprise", "meta_resid_arch_expected_hit_surprise", f"prior_{arm.lookback_hours}h_expected_hr_surprise"], [_numeric(frame, "expected_hit_surprise"), _numeric(frame, "meta_resid_arch_expected_hit_surprise"), _numeric(frame, f"prior_{arm.lookback_hours}h_expected_hr_surprise")]),
        "gmm_posterior_speed": (["resid_event_aegmm_posterior_speed"], [_numeric(frame, "resid_event_aegmm_posterior_speed")]),
        "gmm_posterior_acceleration": (["resid_event_aegmm_posterior_acceleration"], [_numeric(frame, "resid_event_aegmm_posterior_acceleration")]),
        "ae_recon_acceleration": (["ae_reconstruction_error_acceleration"], [accel]),
        "ae_recon_persistence": ([f"ae_reconstruction_time_above_top{arm.recon_z_threshold * 100:g}pct_{arm.lookback_hours}h"], [persistence]),
        "vol_ratio_x_volume_anomaly": (["vol_ratio_x_volume_anomaly"], [vol_volume]),
        "liquidity_shock_proxy": (["liquidity_shock_proxy"], [liquidity_shock]),
    }
    columns, values = families[arm.feature_mode]
    return np.column_stack(values).astype(np.float32, copy=False), columns


def _fit_ridge(train: pd.DataFrame, x: np.ndarray, arm: Arm) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robust standardized ridge residual map, trained on prior candidate rows."""
    # This map controls the deployed top-10% tail.  Top-20 rows regularize the
    # fit, but top-10 rows carry four times the loss weight.  Fitting the full
    # top-30 stream was materially misaligned: it improved global map error
    # while degrading the actually traded tail.
    rank = _numeric(train, BASE_RANK)
    candidate = rank >= 0.80
    # Limit fit population deterministically, retaining temporal spread.
    pos = np.flatnonzero(candidate)
    if pos.size > 60_000:
        pos = np.linspace(0, pos.size - 1, 60_000, dtype=np.int64)
        pos = np.flatnonzero(candidate)[pos]
    xx = x[pos]
    yy = (_numeric(train, EV) - _numeric(train, BASE_EV))[pos]
    weights = (1.0 + 3.0 * (rank[pos] >= 0.90)).astype(np.float32)
    med = np.nanmedian(xx, axis=0).astype(np.float32)
    q25 = np.nanpercentile(xx, 25, axis=0).astype(np.float32)
    q75 = np.nanpercentile(xx, 75, axis=0).astype(np.float32)
    scale = np.maximum(q75 - q25, 1e-4).astype(np.float32)
    z = np.clip((xx - med) / scale, -6.0, 6.0)
    # Mild Huber-like winsorization makes rare path disasters informative
    # without letting a single unresolved geometry dominate a local map.
    yy = np.clip(yy, np.quantile(yy, 0.02), np.quantile(yy, 0.98))
    reg = 25.0 if arm.feature_mode.startswith(("gmm_", "ae_")) else 40.0
    design = np.column_stack([np.ones(len(z), dtype=np.float32), z])
    penalty = np.eye(design.shape[1], dtype=np.float32) * np.float32(reg)
    penalty[0, 0] = 0.0
    root_w = np.sqrt(weights).reshape(-1, 1)
    weighted_design = design * root_w
    beta = np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ (yy * root_w[:, 0]),
    ).astype(np.float32)
    return beta, med, scale


def _predict_correction(x: np.ndarray, params: tuple[np.ndarray, np.ndarray, np.ndarray], arm: Arm) -> np.ndarray:
    beta, med, scale = params
    z = np.clip((x - med) / scale, -6.0, 6.0)
    result = beta[0] + z @ beta[1:]
    return np.clip(result, -arm.max_down, arm.max_up).astype(np.float32)


def _top10_mask(frame: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    # Fixed per-bar global budget: no monthly denominator and no outcome input.
    ordered = frame[["__ts__"]].copy()
    ordered["__score__"] = score
    rank = ordered.groupby("__ts__", sort=False)["__score__"].rank(
        method="first", pct=True, ascending=True
    )
    return rank.to_numpy(dtype=np.float32) > 0.90


def _metrics(frame: pd.DataFrame, selected: np.ndarray) -> dict[str, float]:
    chosen = frame.loc[selected].copy()
    if chosen.empty:
        return {"selected_rows": 0.0}
    chosen["week_start"] = chosen["__ts__"].dt.floor("D").dt.to_period("W-MON").dt.start_time.dt.tz_localize("UTC")
    chosen["month"] = chosen["__ts__"].dt.to_period("M").astype(str)
    weekly = chosen.groupby("week_start", observed=True)[EV].mean()
    monthly = chosen.groupby("month", observed=True)[EV].mean()
    daily = chosen.groupby(chosen["__ts__"].dt.floor("D"), observed=True)[EV].mean()
    errors = _numeric(chosen, EV) - _numeric(chosen, "adjusted_expected_ev")
    return {
        "selected_rows": float(len(chosen)),
        "trades_per_day": float(len(chosen) / max(chosen["__ts__"].dt.floor("D").nunique(), 1)),
        "mean_ev_after_1pct": float(_numeric(chosen, EV).mean()),
        "sum_ev_after_1pct": float(_numeric(chosen, EV).sum()),
        "positive_ev_rate": float((_numeric(chosen, EV) > 0).mean()),
        "clean_exec_precision": float(_numeric(chosen, "clean_exec").mean()),
        "dirty_positive_rate": float(_numeric(chosen, "dirty_positive").mean()),
        "worst_week_ev": float(weekly.min()),
        "worst_month_ev": float(monthly.min()),
        "negative_days_rate": float((daily < 0).mean()),
        "mean_abs_ev_map_error": float(np.abs(errors).mean()),
        "negative_ev_rate": float((_numeric(chosen, EV) < 0).mean()),
    }


def _causal_score(data: pd.DataFrame, arm: Arm) -> pd.DataFrame:
    data = _previous_window_features(data, arm.lookback_hours)
    x, feature_names = _state_matrix(data, arm)
    data["adjustment"] = np.float32(0.0)
    data["adjusted_expected_ev"] = _numeric(data, BASE_EV)
    data["local_correction_applied"] = np.int8(0)
    # Refit every six hours, then apply the frozen correction to the next six
    # completed hourly bars.  This matches a feasible policy-calibration job
    # and avoids silently reusing same-period outcomes between adjacent bars.
    data["calibration_ts"] = data["decision_ts"].dt.floor(f"{REFIT_CADENCE_HOURS}h")
    timestamps = sorted(data["calibration_ts"].dropna().unique())
    # The longest active label path is under 12h.  Preserve a conservative
    # outcome-resolution embargo before any realized row can enter the map.
    embargo = pd.Timedelta(hours=12)
    for current in timestamps:
        current_ts = pd.Timestamp(current)
        current_ts = current_ts.tz_localize("UTC") if current_ts.tzinfo is None else current_ts.tz_convert("UTC")
        test_idx = np.flatnonzero(data["calibration_ts"].eq(current_ts).to_numpy())
        train_end = current_ts - embargo
        start = train_end - pd.Timedelta(hours=arm.lookback_hours)
        train_idx = np.flatnonzero(
            data["decision_ts"].lt(train_end).to_numpy() & data["decision_ts"].ge(start).to_numpy()
        )
        if len(train_idx) < 400:
            continue
        # Strict side x archetype-local fit.  Unsupported cells remain on the
        # parent EV map instead of borrowing another archetype's behavior.
        test = data.iloc[test_idx]
        for (side, archetype), local_positions in test.groupby(list(KEYS), observed=True, sort=False).groups.items():
            local_test = test.index.get_indexer(local_positions)
            mask = (data.iloc[train_idx]["side_name"].astype(str).to_numpy() == str(side)) & (
                data.iloc[train_idx]["archetype_policy_key"].astype(str).to_numpy() == str(archetype)
            )
            local_train = train_idx[mask]
            local_tail_rows = int(
                (_numeric(data.iloc[local_train], BASE_RANK) >= 0.80).sum()
            )
            if len(local_train) < 50 or local_tail_rows < 20:
                continue
            params = _fit_ridge(data.iloc[local_train], x[local_train], arm)
            absolute = test_idx[local_test]
            correction = _predict_correction(x[absolute], params, arm)
            data.iloc[absolute, data.columns.get_loc("adjustment")] = correction
            data.iloc[absolute, data.columns.get_loc("adjusted_expected_ev")] = (
                _numeric(data.iloc[absolute], BASE_EV) + correction
            )
            data.iloc[absolute, data.columns.get_loc("local_correction_applied")] = 1
    data["adjusted_expected_ev"] = data["adjusted_expected_ev"].astype(np.float32)
    data["adjusted_ev_rank"] = data.groupby("__ts__", sort=False)["adjusted_expected_ev"].rank(method="first", pct=True).astype(np.float32)
    data["selected"] = _top10_mask(data, data["adjusted_expected_ev"].to_numpy())
    data["state_feature_count"] = len(feature_names)
    return data


def _error_lift(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame.loc[frame["selected"]].copy()
    if selected.empty:
        return pd.DataFrame()
    selected["map_error"] = _numeric(selected, EV) - _numeric(selected, "adjusted_expected_ev")
    selected["high_error"] = selected["map_error"].le(selected["map_error"].quantile(0.20))
    selected["adjustment_decile"] = pd.qcut(
        selected["adjustment"].rank(method="first"), 10, labels=False, duplicates="drop"
    )
    return (
        selected.groupby("adjustment_decile", observed=True)
        .agg(rows=(EV, "size"), mean_ev=(EV, "mean"), high_error_rate=("high_error", "mean"), mean_adjustment=("adjustment", "mean"))
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--start", default="2026-04-01")
    parser.add_argument("--end", default="2026-06-30 23:59:59+00:00")
    parser.add_argument("--only-arm", action="append", default=[])
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    source = pd.read_parquet(args.source)
    source["__ts__"] = pd.to_datetime(source["__ts__"], utc=True, errors="coerce")
    source = source.loc[source["__ts__"].between(pd.Timestamp(args.start, tz="UTC"), pd.Timestamp(args.end))].copy()
    source = source.dropna(subset=["__ts__", EV, BASE_EV, "side_name", "archetype_policy_key"])
    source = _materialize_context_aliases(source)
    source = source.sort_values("__ts__", kind="stable").reset_index(drop=True)
    arms = [Arm("baseline_parent_ev_map", 0, 1.5, "support", 0.0, 0.0)]
    for lookback in (6, 12, 18):
        for mode in (
            "support",
            "entropy",
            "hit_surprise_actual",
            "hit_surprise_expected",
            "gmm_posterior_speed",
            "gmm_posterior_acceleration",
            "ae_recon_acceleration",
            "vol_ratio_x_volume_anomaly",
            "liquidity_shock_proxy",
        ):
            arms.append(Arm(f"{mode}_{lookback}h", lookback, 1.5, mode, 0.012, 0.004))
        for threshold in (0.01, 0.02, 0.03):
            arms.append(Arm(
                f"ae_recon_persistence_{lookback}h_top{threshold * 100:g}pct",
                lookback,
                threshold,
                "ae_recon_persistence",
                0.012,
                0.004,
            ))
    if args.only_arm:
        requested = set(args.only_arm)
        arms = [arm for arm in arms if arm.name in requested]
        if not arms:
            raise ValueError("--only-arm did not match an available arm")

    records: list[dict[str, float | str]] = []
    baseline_metrics: dict[str, float] | None = None
    for arm in arms:
        print(json.dumps({"event": "arm_start", "arm": arm.name}), flush=True)
        if arm.name == "baseline_parent_ev_map":
            scored = source.copy()
            scored["adjustment"] = np.float32(0.0)
            scored["adjusted_expected_ev"] = _numeric(scored, BASE_EV)
            scored["adjusted_ev_rank"] = _numeric(scored, BASE_RANK)
            scored["selected"] = _top10_mask(scored, scored["adjusted_expected_ev"].to_numpy())
            scored["local_correction_applied"] = np.int8(0)
        else:
            scored = _causal_score(source, arm)
        metrics = _metrics(scored, scored["selected"].to_numpy())
        if baseline_metrics is None:
            baseline_metrics = metrics
        record = {"arm": arm.name, **metrics}
        for key, value in baseline_metrics.items():
            if isinstance(value, (float, int)) and key in metrics:
                record[f"delta_{key}"] = float(metrics[key] - value)
        records.append(record)
        scoped = scored.loc[scored["selected"]].copy()
        scoped["month"] = scoped["__ts__"].dt.to_period("M").astype(str)
        scoped["week_start"] = scoped["__ts__"].dt.to_period("W-MON").dt.start_time.dt.tz_localize("UTC")
        for scope, groupers in {
            "month": ["month"], "week": ["week_start"], "side_archetype": list(KEYS)
        }.items():
            rows = (
                scoped.groupby(groupers, observed=True)
                .agg(rows=(EV, "size"), mean_ev_after_1pct=(EV, "mean"), positive_ev_rate=(EV, lambda s: (pd.to_numeric(s, errors="coerce") > 0).mean()), clean_exec_precision=("clean_exec", "mean"), dirty_positive_rate=("dirty_positive", "mean"))
                .reset_index()
            )
            rows.insert(0, "arm", arm.name)
            rows.to_csv(args.out_dir / f"{arm.name}_{scope}.csv", index=False)
        _error_lift(scored).assign(arm=arm.name).to_csv(args.out_dir / f"{arm.name}_high_error_lift.csv", index=False)
        scored[["__ts__", "__symbol__", "side_name", "archetype_policy_key", EV, BASE_EV, "adjustment", "adjusted_expected_ev", "adjusted_ev_rank", "selected", "local_correction_applied"]].to_parquet(args.out_dir / f"{arm.name}_scored.parquet", index=False, compression="zstd")
        print(json.dumps({"event": "arm_complete", **record}), flush=True)
    summary = pd.DataFrame(records).sort_values("mean_ev_after_1pct", ascending=False, kind="stable")
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    (args.out_dir / "manifest.json").write_text(json.dumps({
        "source": str(args.source), "rows": int(len(source)), "date_start": str(source["__ts__"].min()), "date_end": str(source["__ts__"].max()),
        "selection": "top 10% globally within each timestamp, identical budget per arm", "target": EV,
        "leakage_contract": "For date D every residual/HR feature and ridge correction uses only dates < D; same-day outcomes are excluded.",
        "residual_aliases": ALIASES,
        "state_features": CONTEXT + STATE_BASE + ["ae_reconstruction_error_acceleration", "ae_reconstruction_time_above_threshold", "vol_ratio_x_volume_anomaly", "liquidity_shock_proxy"],
        "arms": [arm.__dict__ for arm in arms],
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
