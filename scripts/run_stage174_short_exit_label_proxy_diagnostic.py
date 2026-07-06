#!/usr/bin/env python3
"""Full-universe short-exit label proxy diagnostic for Stage167.

This is a no-training diagnostic. It replays a small set of executable short
exit policies on the full Stage167 label universe, then asks whether fixed
causal feature proxies trained only on prior rows can recover useful rows in
April, May, and June 2026.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_first_touch_label_training_smoke import _table  # noqa: E402
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    ROUND_TRIP_COST,
    _json_safe,
    _load_feature_store_columns,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
    _spearman,
)
from scripts.run_stage171_fullpath_three_class_proxy_ablation import (  # noqa: E402
    _auc_binary,
    _decile_monotonicity,
    _fit_proxy,
    _score_proxy,
    _selected_top_mask,
)
from scripts.run_stage173_stage167_selected_exit_replay import (  # noqa: E402
    TrailSpec,
    _fetch_paths,
    _fixed_hold_policy,
    _label_policy_frame,
    _safe_numeric,
    _tp_sl_hold_policy,
    _trailing_policy,
)


DEFAULT_LABELS_PATH = Path("data_perp/artifacts/20260703_190000_clean_first_touch_tail_veto_stage167_labels/labels")
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/stage174_short_exit_label_proxy_diagnostic_v1")
DEFAULT_SCORECARD_DIR = Path("data_perp/reports/stage174_short_exit_label_proxy_scorecard_v1")
DEFAULT_MONTHS = ("2026-04", "2026-05", "2026-06")
DEFAULT_TOP_FRACS = (0.30, 0.10, 0.05, 0.03, 0.01)
DEFAULT_MAX_BARRIER = 0.03

DEFAULT_CANDIDATE_POLICIES = (
    "label_first_touch_96",
    "contract_fixed_hold_4",
    "contract_fixed_hold_6",
    "contract_fixed_hold_12",
    "contract_tp_sl_hold_24_tpmax_6",
    "contract_trail_static_act075_gb35_hold24",
    "contract_trail_decay_act075_min040_gb35_hold24",
)

LABEL_OR_FUTURE_PREFIXES = (
    "__y_",
    "__u_",
    "__r_",
    "__mfe",
    "__mae",
    "__tp",
    "__sl",
    "__first_touch",
    "__stage",
)


def _parse_csv(value: str | list[str] | tuple[str, ...], default: tuple[str, ...] = ()) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if str(part).strip()]
    text = str(value).strip()
    if not text:
        return list(default)
    return [part.strip() for part in text.split(",") if part.strip()]


def _parse_float_csv(value: str | list[float] | tuple[float, ...]) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [float(part) for part in value]
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _safe_sum(values: Any) -> float:
    series = _safe_numeric(values).dropna()
    return float(series.sum()) if len(series) else 0.0


def _effective_n(values: Any) -> float:
    counts = pd.Series(values, dtype=object).value_counts(dropna=False)
    if counts.empty:
        return 0.0
    shares = counts.to_numpy(dtype=np.float64) / float(counts.sum())
    denom = float(np.sum(shares * shares))
    return 1.0 / denom if denom > 0.0 else 0.0


def _sigmoid(values: Any) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(values, dtype=np.float64), -60.0, 60.0)))


def _label_parquet_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if files:
            return files
    raise FileNotFoundError(f"No parquet label files found at {path}")


def _load_stage167_labels(path: Path) -> pd.DataFrame:
    requested = [
        "__ts__",
        "__symbol__",
        "__barrier_pct__",
        "__first_touch_effective_tp_abs__",
        "__first_touch_effective_sl_abs__",
        "__first_touch_capture_net__",
        "__first_touch_hit__",
        "__first_touch_stop__",
        "__first_touch_timeout__",
        "__first_touch_bar__",
        "__first_touch_mae_to_sl__",
        "__first_touch_mfe_to_tp__",
        "__first_touch_full_path_mae_to_sl__",
        "__first_touch_full_path_mfe_to_tp__",
        "__u_policy_net__",
        "__r_policy_net__",
        "__y_ret__",
    ]
    parts: list[pd.DataFrame] = []
    for file in _label_parquet_files(path):
        columns = pd.read_parquet(file, columns=None).columns
        keep = [col for col in requested if col in columns]
        missing_keys = sorted({"__ts__", "__symbol__"}.difference(keep))
        if missing_keys:
            raise ValueError(f"{file} is missing label key columns: {missing_keys}")
        parts.append(pd.read_parquet(file, columns=keep))
    frame = pd.concat(parts, ignore_index=True)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], errors="coerce")
    frame["__symbol__"] = frame["__symbol__"].astype(str)
    if frame["__ts__"].isna().any():
        raise ValueError(f"{path} contains non-parseable __ts__ values")
    dupes = int(frame.duplicated(["__ts__", "__symbol__"]).sum())
    if dupes:
        raise ValueError(f"{path} contains duplicate __ts__/__symbol__ keys: {dupes}")
    frame = frame.sort_values(["__ts__", "__symbol__"], kind="mergesort").reset_index(drop=True)
    frame["event_id"] = np.arange(len(frame), dtype=np.int64)
    frame["first_touch_net"] = _safe_numeric(frame.get("__first_touch_capture_net__"), index=frame.index)
    frame["barrier"] = _safe_numeric(frame.get("__barrier_pct__"), index=frame.index)
    frame["side"] = 1.0
    frame["month"] = frame["__ts__"].dt.to_period("M").astype(str)
    frame["week"] = frame["__ts__"].dt.to_period("W-SUN").astype(str)
    return frame


def _filter_feature_names(features: list[str]) -> list[str]:
    out: list[str] = []
    for feature in features:
        name = str(feature)
        if name in {"__ts__", "__symbol__", "event_id", "month", "week"}:
            continue
        if any(name.startswith(prefix) for prefix in LABEL_OR_FUTURE_PREFIXES):
            continue
        out.append(name)
    return list(dict.fromkeys(out))


def _contractize_replay_rows(
    replay_rows: pd.DataFrame,
    *,
    max_barrier: float,
    round_trip_cost: float,
) -> pd.DataFrame:
    out = replay_rows.copy()
    out["policy"] = "contract_" + out["policy"].astype(str)
    out["policy_family"] = "contract_" + out["policy_family"].astype(str)
    ineligible = _safe_numeric(out["barrier_pct"]) > float(max_barrier)
    if bool(ineligible.any()):
        out.loc[ineligible, "net_return"] = -float(round_trip_cost)
        out.loc[ineligible, "gross_return"] = 0.0
        out.loc[ineligible, "exit_bars"] = 0.0
        out.loc[ineligible, "exit_hours"] = 0.0
        out.loc[ineligible, "exit_reason"] = "ineligible_barrier"
        for col in [
            "mae_to_sl_until_exit",
            "mfe_to_tp_until_exit",
            "max_favorable_return_until_exit",
            "max_adverse_return_until_exit",
            "peak_giveback_return",
            "peak_giveback_to_tp",
        ]:
            if col in out.columns:
                out.loc[ineligible, col] = np.nan
    return out


def _build_candidate_policy_rows(
    base: pd.DataFrame,
    paths: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    candidate_policies: list[str],
    round_trip_cost: float,
    max_barrier: float,
) -> pd.DataFrame:
    label = _label_policy_frame(base).copy()
    label["event_id"] = base["event_id"].to_numpy(dtype=np.int64, copy=False)
    if "__first_touch_mae_to_sl__" in base.columns:
        label["mae_to_sl_until_exit"] = _safe_numeric(base["__first_touch_mae_to_sl__"]).to_numpy(dtype=np.float64)
    if "__first_touch_mfe_to_tp__" in base.columns:
        label["mfe_to_tp_until_exit"] = _safe_numeric(base["__first_touch_mfe_to_tp__"]).to_numpy(dtype=np.float64)

    frames: list[pd.DataFrame] = []
    for bars in (4, 6, 12):
        frames.append(_fixed_hold_policy(base, paths, hold_bars=bars, round_trip_cost=round_trip_cost))
    for bars in (24, 96):
        frames.append(
            _tp_sl_hold_policy(
                base,
                paths,
                hold_bars=bars,
                max_tp_bars=6,
                round_trip_cost=round_trip_cost,
            )
        )
    trail_specs = [
        TrailSpec("trail_static_act075_gb35_hold24", 24, 0.75, 0.75, 0.35, 0.0, 0),
        TrailSpec("trail_decay_act075_min040_gb35_hold24", 24, 0.75, 0.40, 0.35, 4.0, 4),
    ]
    for spec in trail_specs:
        frames.append(_trailing_policy(base, paths, spec=spec, round_trip_cost=round_trip_cost))

    replay = pd.concat(frames, ignore_index=True)
    replay["event_id"] = np.tile(base["event_id"].to_numpy(dtype=np.int64, copy=False), len(frames))
    contract = _contractize_replay_rows(
        replay,
        max_barrier=max_barrier,
        round_trip_cost=round_trip_cost,
    )
    policy_rows = pd.concat([label, contract], ignore_index=True)
    keep = set(candidate_policies) | {"contract_tp_sl_hold_96_tpmax_6"}
    policy_rows = policy_rows[policy_rows["policy"].astype(str).isin(keep)].copy()
    policy_rows["month"] = policy_rows["month"].astype(str)
    policy_rows["week"] = policy_rows["week"].astype(str)
    return policy_rows.sort_values(["policy", "__ts__", "__symbol__"], kind="mergesort").reset_index(drop=True)


def _add_target_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    net = _safe_numeric(out["net_return"]).fillna(-0.05)
    mae = _safe_numeric(out.get("mae_to_sl_until_exit")).fillna(10.0)
    exit_bars = _safe_numeric(out.get("exit_bars")).fillna(96.0)
    barrier = _safe_numeric(out.get("barrier_pct")).fillna(0.10)
    out["target_net_clipped"] = net.clip(-0.03, 0.06)
    out["target_net_soft"] = _sigmoid(out["target_net_clipped"] / 0.010)
    out["target_econ_utility"] = (
        out["target_net_clipped"]
        - 0.0040 * (mae - 1.0).clip(lower=0.0)
        - 0.00020 * (exit_bars - 8.0).clip(lower=0.0)
        - 0.35 * (barrier - 0.030).clip(lower=0.0)
    )
    out["target_econ_soft"] = _sigmoid(out["target_econ_utility"] / 0.010)
    return out


def _summary_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    rows = int(len(frame))
    net = _safe_numeric(frame.get("net_return"))
    mae = _safe_numeric(frame.get("mae_to_sl_until_exit"))
    mfe = _safe_numeric(frame.get("mfe_to_tp_until_exit"))
    exit_bars = _safe_numeric(frame.get("exit_bars"))
    barrier = _safe_numeric(frame.get("barrier_pct"))
    reason = frame.get("exit_reason", pd.Series(dtype=object)).astype(str)
    symbols = frame.get("__symbol__", pd.Series(dtype=object)).astype(str)
    timestamps = frame.get("__ts__", pd.Series(dtype=object)).astype(str)
    clean = (net > 0.0) & (mae <= 1.0) & (barrier <= DEFAULT_MAX_BARRIER)
    strict = (net > 0.0) & (mae <= 0.85) & (exit_bars <= 12.0) & (barrier <= DEFAULT_MAX_BARRIER)
    return {
        "rows": rows,
        "finite_rows": int(_safe_numeric(frame.get("finite_path")).ge(0.5).sum()) if rows else 0,
        "sum_net": _safe_sum(net),
        "mean_net": _safe_mean(net),
        "median_net": _safe_quantile(net, 0.50),
        "q10_net": _safe_quantile(net, 0.10),
        "win_rate": _safe_mean(net > 0.0),
        "econ_clean_rate": _safe_mean(clean),
        "strict_clean_rate": _safe_mean(strict),
        "exit_bars_p50": _safe_quantile(exit_bars, 0.50),
        "exit_bars_p90": _safe_quantile(exit_bars, 0.90),
        "mae_to_sl_p90": _safe_quantile(mae, 0.90),
        "mfe_to_tp_p90": _safe_quantile(mfe, 0.90),
        "peak_giveback_to_tp_p90": _safe_quantile(frame.get("peak_giveback_to_tp"), 0.90),
        "sl_rate": _safe_mean(reason.isin(["sl_first_touch", "full_sl"])),
        "tp_rate": _safe_mean(reason.eq("tp_first_touch")),
        "trail_rate": _safe_mean(reason.eq("trailing")),
        "timeout_rate": _safe_mean(reason.str.startswith("timeout")),
        "ineligible_barrier_rate": _safe_mean(reason.eq("ineligible_barrier")),
        "top_symbol_share": float(symbols.value_counts(normalize=True).iloc[0]) if rows else float("nan"),
        "symbol_effective_n": _effective_n(symbols) if rows else 0.0,
        "timestamp_effective_n": _effective_n(timestamps) if rows else 0.0,
    }


def _baseline_rows(policy_rows: pd.DataFrame, *, months: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    for (policy, month), group in policy_rows[policy_rows["month"].isin(months)].groupby(["policy", "month"], sort=True):
        row: dict[str, Any] = {"policy": str(policy), "period": str(month)}
        row.update(_summary_metrics(group))
        monthly_rows.append(row)
    for (policy, week), group in policy_rows[policy_rows["month"].isin(months)].groupby(["policy", "week"], sort=True):
        row = {"policy": str(policy), "period": str(week)}
        row.update(_summary_metrics(group))
        weekly_rows.append(row)
    return pd.DataFrame(monthly_rows), pd.DataFrame(weekly_rows)


def _oracle_rows(policy_rows: pd.DataFrame, *, months: list[str], top_fracs: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (policy, month), group in policy_rows[policy_rows["month"].isin(months)].groupby(["policy", "month"], sort=True):
        reset = group.reset_index(drop=True)
        baseline = _summary_metrics(reset)
        score = _safe_numeric(reset["net_return"])
        for frac in top_fracs:
            mask = _selected_top_mask(score, float(frac))
            selected = reset.loc[mask].copy()
            row: dict[str, Any] = {
                "policy": str(policy),
                "period": str(month),
                "selector": "oracle_net_return",
                "top_frac": float(frac),
            }
            for key, value in baseline.items():
                row[f"baseline_{key}"] = value
            row.update(_summary_metrics(selected))
            row["delta_mean_net_vs_baseline"] = float(row["mean_net"]) - float(row["baseline_mean_net"])
            row["lift_mean_net_vs_baseline"] = (
                float(row["mean_net"]) / float(row["baseline_mean_net"])
                if float(row["baseline_mean_net"] or 0.0) != 0.0
                else float("nan")
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _merge_policy_with_features(policy_frame: pd.DataFrame, feature_frame: pd.DataFrame) -> pd.DataFrame:
    timing_cols = [col for col in ("__ts__", "__symbol__", "month", "week") if col in feature_frame.columns]
    feature_only = feature_frame.drop(columns=timing_cols)
    return policy_frame.merge(feature_only, on="event_id", how="left", validate="many_to_one")


def _proxy_rows(
    *,
    policy_rows: pd.DataFrame,
    feature_frame: pd.DataFrame,
    features: list[str],
    months: list[str],
    top_fracs: list[float],
    proxy_top_k: int,
    min_proxy_rows: int,
    min_abs_ic: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selection_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    target_map = {
        "net_clipped": "target_net_clipped",
        "econ_utility": "target_econ_utility",
    }
    for policy in sorted(policy_rows["policy"].astype(str).unique()):
        model = _merge_policy_with_features(
            _add_target_columns(policy_rows[policy_rows["policy"].astype(str).eq(policy)].reset_index(drop=True)),
            feature_frame,
        )
        month_ser = model["month"].astype(str)
        for month in months:
            train = model.loc[month_ser < str(month)].copy()
            valid = model.loc[month_ser == str(month)].copy()
            if len(train) < int(min_proxy_rows) or valid.empty:
                continue
            baseline = _summary_metrics(valid)
            for target_name, target_col in target_map.items():
                params, _ = _fit_proxy(
                    train,
                    features=features,
                    target_col=target_col,
                    top_k=proxy_top_k,
                    min_rows=min_proxy_rows,
                    min_abs_ic=min_abs_ic,
                )
                for rank, param in enumerate(params, start=1):
                    feature_row = {
                        "policy": str(policy),
                        "period": str(month),
                        "target": str(target_name),
                        "feature_rank": int(rank),
                    }
                    feature_row.update(param)
                    feature_rows.append(feature_row)
                score = _score_proxy(valid, params)
                score_ic_target = _spearman(score, valid[target_col])
                score_ic_net = _spearman(score, valid["net_return"])
                score_auc_win = _auc_binary(score, (_safe_numeric(valid["net_return"]) > 0.0).astype(float))
                monotonicity_target = _decile_monotonicity(score, valid[target_col])
                monotonicity_net = _decile_monotonicity(score, valid["net_return"])
                for frac in top_fracs:
                    mask = _selected_top_mask(score.reset_index(drop=True), float(frac))
                    selected = valid.reset_index(drop=True).loc[mask].copy()
                    row: dict[str, Any] = {
                        "policy": str(policy),
                        "period": str(month),
                        "selector": "prior_month_feature_proxy",
                        "target": str(target_name),
                        "top_frac": float(frac),
                        "train_rows": int(len(train)),
                        "valid_rows": int(len(valid)),
                        "proxy_feature_count": int(len(params)),
                        "proxy_features": ",".join(str(param["feature"]) for param in params),
                        "valid_score_ic_target": score_ic_target,
                        "valid_score_ic_net": score_ic_net,
                        "valid_score_auc_win": score_auc_win,
                        "valid_decile_monotonicity_target": monotonicity_target,
                        "valid_decile_monotonicity_net": monotonicity_net,
                    }
                    for key, value in baseline.items():
                        row[f"baseline_{key}"] = value
                    row.update(_summary_metrics(selected))
                    row["delta_mean_net_vs_baseline"] = float(row["mean_net"]) - float(row["baseline_mean_net"])
                    row["lift_mean_net_vs_baseline"] = (
                        float(row["mean_net"]) / float(row["baseline_mean_net"])
                        if float(row["baseline_mean_net"] or 0.0) != 0.0
                        else float("nan")
                    )
                    row["delta_win_rate_vs_baseline"] = float(row["win_rate"]) - float(row["baseline_win_rate"])
                    row["delta_econ_clean_rate_vs_baseline"] = (
                        float(row["econ_clean_rate"]) - float(row["baseline_econ_clean_rate"])
                    )
                    selection_rows.append(row)
    return pd.DataFrame(selection_rows), pd.DataFrame(feature_rows)


def _alignment_diagnostics(policy_rows: pd.DataFrame) -> pd.DataFrame:
    label = policy_rows[policy_rows["policy"].astype(str).eq("label_first_touch_96")].sort_values("event_id")
    replay = policy_rows[policy_rows["policy"].astype(str).eq("contract_tp_sl_hold_96_tpmax_6")].sort_values("event_id")
    if label.empty or replay.empty:
        return pd.DataFrame()
    joined = label[["event_id", "__ts__", "__symbol__", "month", "net_return", "exit_bars"]].merge(
        replay[["event_id", "net_return", "exit_bars", "exit_reason"]],
        on="event_id",
        how="inner",
        suffixes=("_label", "_replay"),
        validate="one_to_one",
    )
    joined["net_diff"] = _safe_numeric(joined["net_return_replay"]) - _safe_numeric(joined["net_return_label"])
    joined["bar_diff"] = _safe_numeric(joined["exit_bars_replay"]) - _safe_numeric(joined["exit_bars_label"])
    rows = [
        {
            "comparison": "contract_tp_sl_hold_96_tpmax_6_vs_label_first_touch_96",
            "rows": int(len(joined)),
            "mean_abs_net_diff": _safe_mean(joined["net_diff"].abs()),
            "max_abs_net_diff": float(joined["net_diff"].abs().max()) if len(joined) else float("nan"),
            "material_net_diff_rows_1bp": int((joined["net_diff"].abs() > 0.001).sum()),
            "material_net_diff_rate_1bp": _safe_mean(joined["net_diff"].abs() > 0.001),
            "mean_abs_bar_diff": _safe_mean(joined["bar_diff"].abs()),
            "max_abs_bar_diff": float(joined["bar_diff"].abs().max()) if len(joined) else float("nan"),
        }
    ]
    top = joined.reindex(joined["net_diff"].abs().sort_values(ascending=False).index).head(20)
    detail = top.assign(comparison="largest_net_diffs")[
        [
            "comparison",
            "event_id",
            "__ts__",
            "__symbol__",
            "month",
            "net_return_label",
            "net_return_replay",
            "net_diff",
            "exit_bars_label",
            "exit_bars_replay",
            "exit_reason",
        ]
    ]
    return pd.concat([pd.DataFrame(rows), detail], ignore_index=True, sort=False)


def _aggregate_proxy(selection: pd.DataFrame) -> pd.DataFrame:
    if selection.empty:
        return selection
    rows: list[dict[str, Any]] = []
    for keys, group in selection.groupby(["policy", "target", "top_frac"], sort=True):
        policy, target, top_frac = keys
        means = _safe_numeric(group["mean_net"])
        deltas = _safe_numeric(group["delta_mean_net_vs_baseline"])
        ics = _safe_numeric(group["valid_score_ic_net"])
        rows.append(
            {
                "policy": str(policy),
                "target": str(target),
                "top_frac": float(top_frac),
                "months": int(group["period"].nunique()),
                "mean_selected_mean_net": _safe_mean(means),
                "min_selected_mean_net": float(means.min()) if len(means.dropna()) else float("nan"),
                "positive_months_mean_net": int((means > 0.0).sum()),
                "mean_delta_mean_net_vs_baseline": _safe_mean(deltas),
                "min_delta_mean_net_vs_baseline": float(deltas.min()) if len(deltas.dropna()) else float("nan"),
                "mean_win_rate": _safe_mean(group["win_rate"]),
                "mean_econ_clean_rate": _safe_mean(group["econ_clean_rate"]),
                "mean_mae_to_sl_p90": _safe_mean(group["mae_to_sl_p90"]),
                "mean_exit_bars_p90": _safe_mean(group["exit_bars_p90"]),
                "mean_valid_score_ic_net": _safe_mean(ics),
                "min_valid_score_ic_net": float(ics.min()) if len(ics.dropna()) else float("nan"),
                "mean_proxy_feature_count": _safe_mean(group["proxy_feature_count"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["target", "top_frac", "mean_delta_mean_net_vs_baseline", "mean_selected_mean_net"],
        ascending=[True, True, False, False],
    )


def _write_markdown(
    *,
    output_dir: Path,
    baseline: pd.DataFrame,
    oracle: pd.DataFrame,
    proxy: pd.DataFrame,
    proxy_aggregate: pd.DataFrame,
    features: pd.DataFrame,
    alignment: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "stage174_short_exit_label_proxy_diagnostic.md"
    baseline_cols = [
        "policy",
        "period",
        "rows",
        "mean_net",
        "win_rate",
        "econ_clean_rate",
        "exit_bars_p90",
        "mae_to_sl_p90",
        "sl_rate",
        "timeout_rate",
        "trail_rate",
        "ineligible_barrier_rate",
    ]
    proxy_agg_cols = [
        "policy",
        "target",
        "top_frac",
        "months",
        "mean_selected_mean_net",
        "min_selected_mean_net",
        "mean_delta_mean_net_vs_baseline",
        "min_delta_mean_net_vs_baseline",
        "mean_win_rate",
        "mean_econ_clean_rate",
        "mean_valid_score_ic_net",
    ]
    month_focus = proxy[
        proxy["target"].eq("econ_utility") & proxy["top_frac"].isin([0.30, 0.10, 0.05])
    ].copy() if not proxy.empty else proxy
    proxy_cols = [
        "policy",
        "period",
        "target",
        "top_frac",
        "rows",
        "mean_net",
        "baseline_mean_net",
        "delta_mean_net_vs_baseline",
        "win_rate",
        "econ_clean_rate",
        "mae_to_sl_p90",
        "exit_bars_p90",
        "valid_score_ic_net",
        "valid_decile_monotonicity_net",
        "proxy_features",
    ]
    oracle_cols = [
        "policy",
        "period",
        "top_frac",
        "rows",
        "mean_net",
        "baseline_mean_net",
        "delta_mean_net_vs_baseline",
        "win_rate",
        "econ_clean_rate",
        "mae_to_sl_p90",
        "exit_bars_p90",
    ]
    feature_cols = [
        "policy",
        "period",
        "target",
        "feature_rank",
        "feature",
        "train_ic",
        "finite_rows",
    ]
    alignment_cols = [
        "comparison",
        "rows",
        "mean_abs_net_diff",
        "max_abs_net_diff",
        "material_net_diff_rows_1bp",
        "material_net_diff_rate_1bp",
        "event_id",
        "__ts__",
        "__symbol__",
        "net_return_label",
        "net_return_replay",
        "net_diff",
    ]
    lines = [
        "# Stage174 Short-Exit Label Proxy Diagnostic",
        "",
        "Scope: no LightGBM, no Optuna, no portfolio concurrency simulation. Exit outcomes are replayed on the full Stage167 label universe, then feature proxies are selected only on rows before each holdout month.",
        "",
        f"Labels: `{manifest['labels_path']}`",
        f"Feature dir: `{manifest['feature_dir']}`",
        f"Feature count: `{manifest['feature_count']}`",
        f"Rows: `{manifest['rows']}`",
        f"Months evaluated: `{', '.join(manifest['months'])}`",
        f"Round-trip cost: `{manifest['round_trip_cost']:.6f}`",
        f"Contract max barrier: `{manifest['max_barrier']:.4f}`",
        f"Path coverage: `{manifest['path_fetch'].get('finite_path_coverage', float('nan')):.4f}`",
        "",
        "## Monthly Label Baseline",
        "",
        _table(baseline, baseline_cols, limit=120),
        "",
        "## Prior-Month Proxy Aggregate",
        "",
        _table(proxy_aggregate[proxy_aggregate["top_frac"].isin([0.30, 0.10, 0.05])], proxy_agg_cols, limit=120),
        "",
        "## Prior-Month Proxy By Month",
        "",
        _table(month_focus, proxy_cols, limit=160),
        "",
        "## Oracle Upper Bound",
        "",
        _table(oracle[oracle["top_frac"].isin([0.10, 0.01])], oracle_cols, limit=160),
        "",
        "## Replay Alignment",
        "",
        _table(alignment, alignment_cols, limit=40),
        "",
        "## Top Proxy Features",
        "",
        _table(features, feature_cols, limit=120),
        "",
        "## Outputs",
        "",
    ]
    for key, value in manifest["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_scorecard(
    *,
    scorecard_dir: Path,
    manifest: dict[str, Any],
    baseline: pd.DataFrame,
    proxy_aggregate: pd.DataFrame,
    proxy: pd.DataFrame,
    alignment: pd.DataFrame,
) -> Path:
    scorecard_dir.mkdir(parents=True, exist_ok=True)
    path = scorecard_dir / "summary.md"
    focus = proxy_aggregate[
        proxy_aggregate["target"].eq("econ_utility") & proxy_aggregate["top_frac"].isin([0.30, 0.10, 0.05])
    ].copy() if not proxy_aggregate.empty else proxy_aggregate
    focus = focus.sort_values(
        ["top_frac", "mean_delta_mean_net_vs_baseline", "mean_selected_mean_net"],
        ascending=[True, False, False],
    )
    month_focus = proxy[
        proxy["target"].eq("econ_utility") & proxy["top_frac"].isin([0.10, 0.05])
    ].copy() if not proxy.empty else proxy
    base_cols = [
        "policy",
        "period",
        "mean_net",
        "win_rate",
        "econ_clean_rate",
        "exit_bars_p90",
        "mae_to_sl_p90",
    ]
    focus_cols = [
        "policy",
        "target",
        "top_frac",
        "months",
        "mean_selected_mean_net",
        "min_selected_mean_net",
        "mean_delta_mean_net_vs_baseline",
        "min_delta_mean_net_vs_baseline",
        "mean_win_rate",
        "mean_econ_clean_rate",
        "mean_valid_score_ic_net",
    ]
    month_cols = [
        "policy",
        "period",
        "top_frac",
        "mean_net",
        "baseline_mean_net",
        "delta_mean_net_vs_baseline",
        "win_rate",
        "econ_clean_rate",
        "mae_to_sl_p90",
        "valid_score_ic_net",
    ]
    align_cols = [
        "comparison",
        "rows",
        "mean_abs_net_diff",
        "max_abs_net_diff",
        "material_net_diff_rows_1bp",
        "material_net_diff_rate_1bp",
    ]
    lines = [
        "# Stage174 Scorecard - Short-Exit Label Proxy",
        "",
        "Scope: full-universe Stage167 rows, no model training, no Optuna, no portfolio concurrency. Each holdout month uses only prior rows for proxy feature signs and scaling.",
        "",
        f"- Script: `scripts/run_stage174_short_exit_label_proxy_diagnostic.py`",
        f"- Output: `{manifest['output_dir']}`",
        f"- Labels: `{manifest['labels_path']}`",
        f"- Rows: `{manifest['rows']}`",
        f"- Months: `{', '.join(manifest['months'])}`",
        f"- Path coverage: `{manifest['path_fetch'].get('finite_path_coverage', float('nan')):.4f}`",
        "",
        "## Readout",
        "",
        "This scorecard should be read as label learnability evidence, not as a deployable policy result. Positive proxy deltas mean a label is more recoverable by current causal features; they do not prove portfolio PnL after concurrency, ranking, or threshold selection.",
        "",
        "## Monthly Label Baseline",
        "",
        _table(baseline, base_cols, limit=120),
        "",
        "## Prior-Month Proxy Aggregate",
        "",
        _table(focus, focus_cols, limit=120),
        "",
        "## Prior-Month Proxy Month Detail",
        "",
        _table(month_focus, month_cols, limit=160),
        "",
        "## Replay Alignment Check",
        "",
        _table(alignment.head(1), align_cols, limit=10),
        "",
        "## Decision Gate",
        "",
        "Promote no label directly from this diagnostic. A candidate is worth model training only if it shows positive prior-month proxy delta in April, May, and June at top 10% or top 5%, acceptable MAE-until-exit, and no large replay-alignment mismatch that invalidates the execution contract.",
        "",
        "## Outputs",
        "",
    ]
    for key, value in manifest["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_diagnostic(
    *,
    labels_path: Path,
    output_dir: Path,
    scorecard_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_features: int,
    months: list[str],
    candidate_policies: list[str],
    top_fracs: list[float],
    proxy_top_k: int,
    min_proxy_rows: int,
    min_abs_ic: float,
    data_root: Path,
    market_mode: str,
    exchange: str,
    path_len: int,
    apply_delayed_entry: bool,
    round_trip_cost: float,
    max_barrier: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = _load_stage167_labels(labels_path)
    max_month = max(str(month) for month in months)
    labels = labels[labels["month"] <= max_month].reset_index(drop=True)
    labels["event_id"] = np.arange(len(labels), dtype=np.int64)
    rows_exec, paths, path_fetch = _fetch_paths(
        labels,
        labels_path=labels_path,
        data_root=data_root,
        market_mode=market_mode,
        exchange=exchange,
        path_len=path_len,
        apply_delayed_entry=apply_delayed_entry,
    )
    if len(rows_exec) != len(labels):
        raise ValueError(f"Path fetch row count mismatch: labels={len(labels)} paths={len(rows_exec)}")
    policy_rows = _build_candidate_policy_rows(
        labels,
        paths,
        candidate_policies=candidate_policies,
        round_trip_cost=round_trip_cost,
        max_barrier=max_barrier,
    )
    selected_features = _filter_feature_names(_read_feature_list(feature_list_csv, max_features=max_features))
    feature_matrix, feature_manifest = _load_feature_store_columns(
        labels,
        feature_dir=feature_dir,
        selected_features=selected_features,
    )
    features = [feature for feature in selected_features if feature in feature_matrix.columns]
    feature_frame = pd.concat(
        [
            labels[["event_id", "__ts__", "__symbol__", "month", "week"]].reset_index(drop=True),
            feature_matrix.reset_index(drop=True),
        ],
        axis=1,
    )
    monthly, weekly = _baseline_rows(policy_rows[policy_rows["policy"].isin(candidate_policies)], months=months)
    oracle = _oracle_rows(policy_rows[policy_rows["policy"].isin(candidate_policies)], months=months, top_fracs=top_fracs)
    proxy, proxy_features = _proxy_rows(
        policy_rows=policy_rows[policy_rows["policy"].isin(candidate_policies)],
        feature_frame=feature_frame,
        features=features,
        months=months,
        top_fracs=top_fracs,
        proxy_top_k=proxy_top_k,
        min_proxy_rows=min_proxy_rows,
        min_abs_ic=min_abs_ic,
    )
    proxy_aggregate = _aggregate_proxy(proxy)
    alignment = _alignment_diagnostics(policy_rows)

    paths_out = {
        "policy_rows": output_dir / "stage174_policy_rows.csv",
        "monthly_baseline": output_dir / "stage174_label_monthly_baseline.csv",
        "weekly_baseline": output_dir / "stage174_label_weekly_baseline.csv",
        "oracle_selection": output_dir / "stage174_oracle_selection_metrics.csv",
        "proxy_selection": output_dir / "stage174_proxy_selection_metrics.csv",
        "proxy_aggregate": output_dir / "stage174_proxy_aggregate_metrics.csv",
        "proxy_features": output_dir / "stage174_proxy_features.csv",
        "alignment": output_dir / "stage174_alignment_diagnostics.csv",
        "manifest": output_dir / "manifest.json",
    }
    policy_rows.to_csv(paths_out["policy_rows"], index=False)
    monthly.to_csv(paths_out["monthly_baseline"], index=False)
    weekly.to_csv(paths_out["weekly_baseline"], index=False)
    oracle.to_csv(paths_out["oracle_selection"], index=False)
    proxy.to_csv(paths_out["proxy_selection"], index=False)
    proxy_aggregate.to_csv(paths_out["proxy_aggregate"], index=False)
    proxy_features.to_csv(paths_out["proxy_features"], index=False)
    alignment.to_csv(paths_out["alignment"], index=False)
    manifest = {
        "scope": "stage174_short_exit_label_proxy_no_training",
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "scorecard_dir": str(scorecard_dir),
        "rows": int(len(labels)),
        "timestamp_min": labels["__ts__"].min(),
        "timestamp_max": labels["__ts__"].max(),
        "symbols": int(labels["__symbol__"].nunique(dropna=True)),
        "months": list(months),
        "candidate_policies": list(candidate_policies),
        "top_fracs": list(top_fracs),
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "max_features": int(max_features),
        "feature_count": int(len(features)),
        "features": features,
        "feature_store": feature_manifest,
        "data_root": str(data_root),
        "market_mode": str(market_mode),
        "exchange": str(exchange),
        "path_len": int(path_len),
        "apply_delayed_entry": bool(apply_delayed_entry),
        "round_trip_cost": float(round_trip_cost),
        "max_barrier": float(max_barrier),
        "proxy_top_k": int(proxy_top_k),
        "min_proxy_rows": int(min_proxy_rows),
        "min_abs_ic": float(min_abs_ic),
        "path_fetch": path_fetch,
        "outputs": {key: str(value) for key, value in paths_out.items()},
    }
    paths_out["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    markdown = _write_markdown(
        output_dir=output_dir,
        baseline=monthly,
        oracle=oracle,
        proxy=proxy,
        proxy_aggregate=proxy_aggregate,
        features=proxy_features,
        alignment=alignment,
        manifest=manifest,
    )
    manifest["outputs"]["markdown"] = str(markdown)
    scorecard = _write_scorecard(
        scorecard_dir=scorecard_dir,
        manifest=manifest,
        baseline=monthly,
        proxy_aggregate=proxy_aggregate,
        proxy=proxy,
        alignment=alignment,
    )
    manifest["outputs"]["scorecard"] = str(scorecard)
    paths_out["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scorecard-dir", type=Path, default=DEFAULT_SCORECARD_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-features", type=int, default=160)
    parser.add_argument("--months", default=",".join(DEFAULT_MONTHS))
    parser.add_argument("--candidate-policies", default=",".join(DEFAULT_CANDIDATE_POLICIES))
    parser.add_argument("--top-fracs", default=",".join(str(v) for v in DEFAULT_TOP_FRACS))
    parser.add_argument("--proxy-top-k", type=int, default=8)
    parser.add_argument("--min-proxy-rows", type=int, default=300)
    parser.add_argument("--min-abs-ic", type=float, default=0.0)
    parser.add_argument("--data-root", type=Path, default=Path("data_perp"))
    parser.add_argument("--market-mode", default="perps")
    parser.add_argument("--exchange", default="krakenfutures")
    parser.add_argument("--path-len", type=int, default=96)
    parser.add_argument("--no-delayed-entry", action="store_true")
    parser.add_argument("--round-trip-cost", type=float, default=ROUND_TRIP_COST)
    parser.add_argument("--max-barrier", type=float, default=DEFAULT_MAX_BARRIER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_diagnostic(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        scorecard_dir=args.scorecard_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_features=int(args.max_features),
        months=_parse_csv(args.months, DEFAULT_MONTHS),
        candidate_policies=_parse_csv(args.candidate_policies, DEFAULT_CANDIDATE_POLICIES),
        top_fracs=_parse_float_csv(args.top_fracs),
        proxy_top_k=int(args.proxy_top_k),
        min_proxy_rows=int(args.min_proxy_rows),
        min_abs_ic=float(args.min_abs_ic),
        data_root=args.data_root,
        market_mode=str(args.market_mode),
        exchange=str(args.exchange),
        path_len=int(args.path_len),
        apply_delayed_entry=not bool(args.no_delayed_entry),
        round_trip_cost=float(args.round_trip_cost),
        max_barrier=float(args.max_barrier),
    )
    print(json.dumps(_json_safe(manifest), indent=2))


if __name__ == "__main__":
    main()
