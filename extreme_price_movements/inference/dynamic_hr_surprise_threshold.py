"""Apply validated hit-rate-surprise threshold artifacts in live inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd


HEADS = ("long_bars", "long_dist", "short_asset", "short_boll")
T16_POLICY_NAME = "T16_q42_weighted_guard_hr35_last7_11"


@dataclass(frozen=True)
class DynamicHrHeadState:
    head: str
    guarded_y: float
    w_lower: float = 0.0
    w_raise: float = 0.0
    z_eff: float = 0.0
    deployed_threshold: float = np.nan
    dynamic_rejected: bool = False
    fallback_to_deployed: bool = False
    deactivated: bool = False
    reason: str = ""
    as_of: str = ""


@dataclass(frozen=True)
class DynamicHrThresholdResult:
    threshold: float
    applied: bool
    reason: str
    head: str
    z_eff: float = 0.0
    guarded_y: float = np.nan
    w_lower: float = np.nan
    w_raise: float = np.nan
    state_age_days: float = np.nan


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def infer_head(strategy_id: Any) -> str:
    value = str(strategy_id or "")
    for head in HEADS:
        if value.startswith(head):
            return head
    return "unknown"


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _parse_timestamp(value: Any) -> Optional[pd.Timestamp]:
    if value in (None, ""):
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def _first_timestamp(row: Mapping[str, Any], names: tuple[str, ...]) -> Optional[pd.Timestamp]:
    for name in names:
        ts = _parse_timestamp(row.get(name))
        if ts is not None:
            return ts
    return None


def _state_age_days(as_of: str, now: Any | None) -> float:
    ts = _parse_timestamp(as_of)
    if ts is None:
        return float("inf")
    now_ts = _parse_timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    if now_ts is None:
        now_ts = pd.Timestamp.now(tz="UTC")
    return float(max((now_ts - ts).total_seconds(), 0.0) / 86400.0)


def _load_json(path: Path) -> dict[str, DynamicHrHeadState]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return {}
    heads = payload.get("heads", payload)
    if not isinstance(heads, Mapping):
        return {}
    out: dict[str, DynamicHrHeadState] = {}
    for raw_head, item in heads.items():
        if not isinstance(item, Mapping):
            continue
        head = str(item.get("head") or raw_head)
        out[head] = DynamicHrHeadState(
            head=head,
            guarded_y=_finite(item.get("guarded_y", item.get("y")), 1.50),
            w_lower=_finite(item.get("w_lower", item.get("w")), 0.0),
            w_raise=_finite(item.get("w_raise", item.get("w")), 0.0),
            z_eff=_finite(item.get("z_eff"), 0.0),
            deployed_threshold=_finite(item.get("deployed_threshold", item.get("deployed_fixed_threshold")), np.nan),
            dynamic_rejected=_truthy(item.get("dynamic_rejected", False)),
            fallback_to_deployed=_truthy(item.get("fallback_to_deployed", False)),
            deactivated=_truthy(item.get("deactivated", False)),
            reason=str(item.get("deactivation_reason") or item.get("reason") or ""),
            as_of=str(item.get("day_start") or item.get("as_of") or payload.get("as_of") or ""),
        )
    return out


def _load_table(path: Path) -> dict[str, DynamicHrHeadState]:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    if frame.empty or "head" not in frame.columns:
        return {}
    sort_cols = [col for col in ("day_start", "timestamp", "as_of") if col in frame.columns]
    if sort_cols:
        frame = frame.sort_values(sort_cols)
    latest = frame.drop_duplicates("head", keep="last")
    out: dict[str, DynamicHrHeadState] = {}
    for row in latest.to_dict("records"):
        head = str(row.get("head") or "")
        if not head:
            continue
        out[head] = DynamicHrHeadState(
            head=head,
            guarded_y=_finite(row.get("guarded_y", row.get("y")), 1.50),
            w_lower=_finite(row.get("w_lower", row.get("w")), 0.0),
            w_raise=_finite(row.get("w_raise", row.get("w")), 0.0),
            z_eff=_finite(row.get("z_eff"), 0.0),
            deployed_threshold=_finite(row.get("deployed_threshold", row.get("deployed_fixed_threshold")), np.nan),
            dynamic_rejected=_truthy(row.get("dynamic_rejected", False)),
            fallback_to_deployed=_truthy(row.get("fallback_to_deployed", False)),
            deactivated=_truthy(row.get("deactivated", False)),
            reason=str(row.get("deactivation_reason") or row.get("reason") or ""),
            as_of=str(row.get("day_start") or row.get("timestamp") or row.get("as_of") or ""),
        )
    return out


def load_dynamic_hr_surprise_state(path: str | Path | None) -> dict[str, DynamicHrHeadState]:
    if not path:
        return {}
    resolved = Path(path)
    if not resolved.exists():
        return {}
    try:
        if resolved.suffix.lower() == ".json":
            return _load_json(resolved)
        return _load_table(resolved)
    except Exception:
        return {}


def validate_dynamic_hr_replay_gate(
    summary: pd.DataFrame,
    *,
    policy: str = "calendar_dynamic_hr_surprise",
    baseline: str = "fixed_deployed_thresholds",
) -> dict[str, Any]:
    """Return whether a replayed dynamic policy is non-degrading vs baseline."""
    if summary is None or summary.empty or "policy" not in summary.columns:
        return {
            "accepted": False,
            "reason": "missing_policy_summary",
            "policy": policy,
            "baseline": baseline,
        }
    keyed = summary.set_index(summary["policy"].astype(str), drop=False)
    if policy not in keyed.index or baseline not in keyed.index:
        return {
            "accepted": False,
            "reason": "missing_policy_or_baseline_row",
            "policy": policy,
            "baseline": baseline,
        }
    dyn = keyed.loc[policy]
    base = keyed.loc[baseline]
    if isinstance(dyn, pd.DataFrame):
        dyn = dyn.iloc[-1]
    if isinstance(base, pd.DataFrame):
        base = base.iloc[-1]
    checks: dict[str, bool] = {}
    deltas: dict[str, float] = {}
    for col in (
        "total_net_pnl",
        "objective",
        "q05_rolling_week_pnl",
        "q15_rolling_week_pnl",
    ):
        dyn_val = _finite(dyn.get(col), np.nan)
        base_val = _finite(base.get(col), np.nan)
        if np.isfinite(dyn_val) and np.isfinite(base_val):
            deltas[f"{col}_delta"] = float(dyn_val - base_val)
            checks[col] = bool(dyn_val + 1e-12 >= base_val)
    accepted = bool(checks) and all(checks.values())
    return {
        "accepted": accepted,
        "reason": "non_degrading_vs_baseline" if accepted else "degrades_vs_baseline",
        "policy": policy,
        "baseline": baseline,
        "checks": checks,
        "deltas": deltas,
        "policy_metrics": {
            col: _finite(dyn.get(col), np.nan)
            for col in summary.columns
            if col != "policy"
        },
        "baseline_metrics": {
            col: _finite(base.get(col), np.nan)
            for col in summary.columns
            if col != "policy"
        },
    }


def _numeric_series(frame: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce").fillna(default).astype("float64")


def _candidate_spread_net_return(
    frame: pd.DataFrame,
    *,
    return_col: str,
) -> pd.Series:
    base_col = "net_return_before_spread" if "net_return_before_spread" in frame.columns else return_col
    base = _numeric_series(frame, base_col, 0.0)
    full_spread = _numeric_series(frame, "expected_spread_bps", 0.0).clip(lower=0.0)
    entry_half = _numeric_series(frame, "expected_half_spread_bps", np.nan)
    if "spread_cost_bps" in frame.columns:
        entry_half = entry_half.where(np.isfinite(entry_half), _numeric_series(frame, "spread_cost_bps", np.nan))
    exit_half = _numeric_series(frame, "exit_spread_cost_bps", np.nan)
    if "exit_quote_half_spread_bps" in frame.columns:
        exit_half = exit_half.where(np.isfinite(exit_half), _numeric_series(frame, "exit_quote_half_spread_bps", np.nan))
    full_half = full_spread / 2.0
    entry_half = entry_half.where(np.isfinite(entry_half), full_half)
    exit_half = exit_half.where(np.isfinite(exit_half), full_half)
    explicit_cols = [
        col
        for col in (
            "expected_half_spread_bps",
            "spread_cost_bps",
            "exit_spread_cost_bps",
            "exit_quote_half_spread_bps",
        )
        if col in frame.columns
    ]
    if explicit_cols and bool(full_spread.gt(0.0).any()):
        explicit_abs = pd.concat(
            [_numeric_series(frame, col, 0.0).abs() for col in explicit_cols],
            axis=1,
        )
        zero_explicit = explicit_abs.max(axis=1).le(1e-12) & full_spread.gt(0.0)
        if len(zero_explicit) and float(zero_explicit.mean()) > 0.95:
            entry_half = entry_half.where(~zero_explicit, full_half)
            exit_half = exit_half.where(~zero_explicit, full_half)
    return base - (entry_half.fillna(0.0).clip(lower=0.0) + exit_half.fillna(0.0).clip(lower=0.0)) / 10_000.0


def _ewm_shifted(series: pd.Series, halflife_days: float) -> pd.Series:
    series = series.sort_index()
    if series.empty:
        return series
    return (
        series.ewm(
            halflife=pd.Timedelta(days=float(halflife_days)),
            times=series.index,
            adjust=True,
        )
        .mean()
        .shift(1)
    )


def _candidate_surprise_frame(
    candidate_path: str | Path,
    *,
    schema_contract: Mapping[str, Any],
) -> pd.DataFrame:
    path = Path(candidate_path)
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    timestamp_col = str(schema_contract.get("timestamp_col") or "timestamp")
    strategy_col = str(schema_contract.get("strategy_col") or "head")
    rank_col = str(schema_contract.get("rank_col") or "policy_rank_pct")
    p_hit_col = str(schema_contract.get("p_hit_col") or "calibrated_score")
    return_col = str(schema_contract.get("return_col") or "net_return")
    if timestamp_col not in frame.columns or strategy_col not in frame.columns:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce"),
            "strategy_key": frame[strategy_col].astype(str),
        }
    )
    out["head"] = out["strategy_key"].map(infer_head) if strategy_col != "head" else out["strategy_key"].astype(str)
    out["rank"] = _numeric_series(frame, rank_col, np.nan)
    out["p_hit"] = _numeric_series(frame, p_hit_col, np.nan).clip(1e-6, 1.0 - 1e-6)
    if bool(schema_contract.get("spread_adjust_returns", True)):
        out["net_return"] = _candidate_spread_net_return(frame, return_col=return_col)
    else:
        out["net_return"] = _numeric_series(frame, return_col, np.nan)
    weight_col = str(schema_contract.get("surprise_weight_col") or "")
    if weight_col and weight_col != "constant_1" and weight_col in frame.columns:
        out["surprise_weight"] = _numeric_series(frame, weight_col, 1.0).clip(0.0, 100.0)
    else:
        out["surprise_weight"] = 1.0
    out["hit"] = out["net_return"].gt(0.0).astype(float)
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["timestamp", "head", "rank", "p_hit", "net_return"])
    return out.loc[out["head"].isin(HEADS)].sort_values(["timestamp", "head"]).reset_index(drop=True)


def latest_surprise_by_head_from_candidates(
    candidate_path: str | Path,
    params: pd.DataFrame,
    *,
    schema_contract: Mapping[str, Any],
    as_of: Any,
    top_rank_floor: float = 0.70,
    z_clip: float = 5.0,
    count_shrink_n0: float = 20.0,
) -> dict[str, dict[str, float]]:
    """Compute the latest causal standardized HR surprise per head."""
    frame = _candidate_surprise_frame(candidate_path, schema_contract=schema_contract)
    as_of_ts = _parse_timestamp(as_of)
    if frame.empty or as_of_ts is None:
        return {}
    frame = frame.loc[frame["timestamp"].le(as_of_ts)].copy()
    if frame.empty:
        return {}
    x_days = {
        str(row.get("head")): _finite(row.get("x_days"), 7.0)
        for row in params.to_dict("records")
    }
    out: dict[str, dict[str, float]] = {}
    for head, group in frame.groupby("head", sort=True):
        idx = pd.DatetimeIndex(sorted(group["timestamp"].dropna().unique()))
        eligible = group.loc[group["rank"].ge(float(top_rank_floor))].copy()
        if eligible.empty:
            agg = pd.DataFrame(index=idx, data={"num": 0.0, "var": 0.0, "count": 0.0})
        else:
            weight = pd.to_numeric(eligible["surprise_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
            p_hit = eligible["p_hit"].to_numpy(dtype=float)
            hit = eligible["hit"].to_numpy(dtype=float)
            eligible["num_component"] = weight * (hit - p_hit)
            eligible["var_component"] = np.square(weight) * p_hit * (1.0 - p_hit)
            eligible["count_component"] = 1.0
            agg = (
                eligible.groupby("timestamp", sort=True)[
                    ["num_component", "var_component", "count_component"]
                ]
                .sum()
                .rename(
                    columns={
                        "num_component": "num",
                        "var_component": "var",
                        "count_component": "count",
                    }
                )
                .reindex(idx, fill_value=0.0)
            )
        if agg.empty:
            continue
        halflife = float(x_days.get(str(head), 7.0))
        ewma_num = _ewm_shifted(agg["num"], halflife).fillna(0.0)
        ewma_var = _ewm_shifted(agg["var"], halflife).fillna(0.0)
        ewma_count = _ewm_shifted(agg["count"], halflife).fillna(0.0)
        z_raw = ewma_num / np.sqrt(ewma_var + 1e-12)
        count_shrink = ewma_count / (ewma_count + max(float(count_shrink_n0), 0.0))
        z_eff = (z_raw * count_shrink).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-float(z_clip), float(z_clip))
        out[str(head)] = {
            "z_eff": float(z_eff.iloc[-1]),
            "z_raw": float(z_raw.iloc[-1]) if np.isfinite(float(z_raw.iloc[-1])) else 0.0,
            "ewma_num": float(ewma_num.iloc[-1]),
            "ewma_var": float(ewma_var.iloc[-1]),
            "ewma_count": float(ewma_count.iloc[-1]),
            "count_shrink": float(count_shrink.iloc[-1]),
            "surprise_as_of": pd.Timestamp(agg.index[-1]).isoformat(),
        }
    return out


def dynamic_hr_state_payload_from_daily_params(
    daily_params: pd.DataFrame,
    *,
    policy_name: str = T16_POLICY_NAME,
    source_manifest: Mapping[str, Any] | None = None,
    source_replay_dir: str | Path | None = None,
    promotion_gate: Mapping[str, Any] | None = None,
    surprise_by_head: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the live JSON state consumed by ``load_dynamic_hr_surprise_state``."""
    if daily_params is None or daily_params.empty or "head" not in daily_params.columns:
        raise ValueError("daily_params must contain at least one row with a head column")
    frame = daily_params.copy()
    sort_cols = [col for col in ("day_end", "day_start", "timestamp", "as_of") if col in frame.columns]
    if sort_cols:
        for col in sort_cols:
            frame[col] = pd.to_datetime(frame[col], utc=True, errors="coerce")
        frame = frame.sort_values(sort_cols)
    latest = frame.drop_duplicates("head", keep="last")
    heads: dict[str, Any] = {}
    as_of_values: list[pd.Timestamp] = []
    for row in latest.to_dict("records"):
        head = str(row.get("head") or "")
        if head not in HEADS:
            continue
        as_of_ts = _first_timestamp(row, ("day_end", "day_start", "timestamp", "as_of"))
        if as_of_ts is not None:
            as_of_values.append(as_of_ts)
        surprise = dict((surprise_by_head or {}).get(head) or {})
        day_start_ts = _parse_timestamp(row.get("day_start"))
        day_end_ts = _parse_timestamp(row.get("day_end"))
        heads[head] = {
            "head": head,
            "guarded_y": _finite(row.get("guarded_y", row.get("y")), 1.50),
            "y": _finite(row.get("y", row.get("guarded_y")), 1.50),
            "w_lower": _finite(row.get("w_lower", row.get("w")), 0.0),
            "w_raise": _finite(row.get("w_raise", row.get("w")), 0.0),
            "z_eff": _finite(surprise.get("z_eff", row.get("z_eff")), 0.0),
            "z_raw": _finite(surprise.get("z_raw"), 0.0),
            "ewma_num": _finite(surprise.get("ewma_num"), 0.0),
            "ewma_var": _finite(surprise.get("ewma_var"), 0.0),
            "ewma_count": _finite(surprise.get("ewma_count"), 0.0),
            "count_shrink": _finite(surprise.get("count_shrink"), 0.0),
            "surprise_as_of": str(surprise.get("surprise_as_of") or ""),
            "deployed_threshold": _finite(
                row.get("deployed_threshold", row.get("deployed_fixed_threshold")),
                np.nan,
            ),
            "dynamic_rejected": _truthy(row.get("dynamic_rejected", False)),
            "fallback_to_deployed": _truthy(row.get("fallback_to_deployed", False)),
            "deactivated": _truthy(row.get("deactivated", False)),
            "reason": str(row.get("deactivation_reason") or row.get("reason") or ""),
            "as_of": as_of_ts.isoformat() if as_of_ts is not None else "",
            "day_start": day_start_ts.isoformat() if day_start_ts is not None else "",
            "day_end": day_end_ts.isoformat() if day_end_ts is not None else "",
            "x_days": _finite(row.get("x_days"), np.nan),
            "local_band_pnl": _finite(row.get("local_band_pnl"), np.nan),
            "local_band_count": int(_finite(row.get("local_band_count"), 0.0)),
            "recent_validation_guarded": _truthy(row.get("recent_validation_guarded", False)),
            "recent_validation_count": int(_finite(row.get("recent_validation_count"), 0.0)),
            "recent_validation_total_pnl": _finite(row.get("recent_validation_total_pnl"), np.nan),
            "recent_validation_hit_rate": _finite(row.get("recent_validation_hit_rate"), np.nan),
        }
    if not heads:
        raise ValueError("No supported heads found in daily_params")
    as_of = max(as_of_values).isoformat() if as_of_values else pd.Timestamp.now(tz="UTC").isoformat()
    return {
        "schema_version": "dynamic_hr_surprise_state_v1",
        "policy_name": policy_name,
        "as_of": as_of,
        "heads": heads,
        "source_manifest": _json_safe(source_manifest or {}),
        "source_replay_dir": str(source_replay_dir or ""),
        "promotion_gate": _json_safe(promotion_gate or {}),
        "threshold_formula": (
            "clip(Y_h - W_lower_h * max(0, z_eff_h_t) "
            "- W_raise_h * min(0, z_eff_h_t), -0.50, 1.50)"
        ),
        "freshness_contract": "missing or stale state falls back to deployed thresholds",
    }


def write_dynamic_hr_surprise_state_from_replay(
    replay_dir: str | Path,
    output_path: str | Path,
    *,
    policy_name: str = T16_POLICY_NAME,
    policy: str = "calendar_dynamic_hr_surprise",
    baseline: str = "fixed_deployed_thresholds",
) -> dict[str, Any]:
    replay = Path(replay_dir)
    params_path = replay / "calendar_dynamic_hr_surprise_daily_y_params.parquet"
    if not params_path.exists():
        csv_path = replay / "calendar_dynamic_hr_surprise_daily_y_params.csv"
        if not csv_path.exists():
            raise FileNotFoundError(params_path)
        daily_params = pd.read_csv(csv_path)
    else:
        daily_params = pd.read_parquet(params_path)
    summary_path = replay / "calendar_dynamic_hr_surprise_policy_summary.parquet"
    if summary_path.exists():
        summary = pd.read_parquet(summary_path)
    else:
        summary = pd.read_csv(replay / "calendar_dynamic_hr_surprise_policy_summary.csv")
    manifest_path = replay / "calendar_dynamic_hr_surprise_manifest.json"
    source_manifest: dict[str, Any] = {}
    if manifest_path.exists():
        source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dynamic_manifest_path = replay / "dynamic_hr_surprise_manifest.json"
    if dynamic_manifest_path.exists():
        dynamic_manifest = json.loads(dynamic_manifest_path.read_text(encoding="utf-8"))
        if isinstance(dynamic_manifest, Mapping):
            source_manifest = {**dynamic_manifest, **source_manifest}
    promotion_gate = validate_dynamic_hr_replay_gate(summary, policy=policy, baseline=baseline)
    as_of = None
    if "day_end" in daily_params.columns:
        as_of = pd.to_datetime(daily_params["day_end"], utc=True, errors="coerce").max()
    if as_of is None or pd.isna(as_of):
        as_of = source_manifest.get("eval_end") or source_manifest.get("period_end")
    surprise_by_head: dict[str, dict[str, float]] = {}
    candidate_path = source_manifest.get("candidate_path")
    schema_contract = source_manifest.get("schema_contract") or {}
    if candidate_path and isinstance(schema_contract, Mapping):
        try:
            surprise_by_head = latest_surprise_by_head_from_candidates(
                candidate_path,
                daily_params,
                schema_contract=schema_contract,
                as_of=as_of,
                top_rank_floor=float(source_manifest.get("top_rank_floor", 0.70)),
                z_clip=float(source_manifest.get("z_clip", 5.0)),
                count_shrink_n0=float(source_manifest.get("surprise_count_shrink_n0", 20.0)),
            )
        except Exception:
            surprise_by_head = {}
    payload = dynamic_hr_state_payload_from_daily_params(
        daily_params,
        policy_name=policy_name,
        source_manifest=source_manifest,
        source_replay_dir=replay,
        promotion_gate=promotion_gate,
        surprise_by_head=surprise_by_head,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    tmp.replace(out)
    return payload


def patch_portfolio_policy_payload_with_dynamic_hr_surprise(
    payload: Mapping[str, Any],
    *,
    artifact_path: str | Path,
    enabled: bool,
    max_state_age_days: float = 7.0,
    use_deployed_floor: bool = False,
    fallback_to_deployed: bool = False,
    stale_fallback_to_deployed: bool = True,
    lower_bound: float = -0.50,
    upper_bound: float = 1.50,
) -> dict[str, Any]:
    out = dict(payload or {})
    selection = dict(out.get("selection") or {})
    updates = {
        "dynamic_hr_surprise_enabled": bool(enabled),
        "dynamic_hr_surprise_artifact_path": str(artifact_path),
        "dynamic_hr_surprise_use_deployed_floor": bool(use_deployed_floor),
        "dynamic_hr_surprise_fallback_to_deployed": bool(fallback_to_deployed),
        "dynamic_hr_surprise_stale_fallback_to_deployed": bool(stale_fallback_to_deployed),
        "dynamic_hr_surprise_max_state_age_days": float(max_state_age_days),
        "dynamic_hr_surprise_lower_bound": float(lower_bound),
        "dynamic_hr_surprise_upper_bound": float(upper_bound),
    }
    selection.update(updates)
    out["selection"] = selection
    out.update(updates)
    return out


def apply_dynamic_hr_surprise_threshold(
    *,
    strategy_id: Any,
    deployed_threshold: float,
    state: Mapping[str, DynamicHrHeadState] | None,
    enabled: bool,
    use_deployed_floor: bool = True,
    fallback_to_deployed: bool = True,
    stale_fallback_to_deployed: bool = True,
    max_state_age_days: float | None = None,
    now: Any | None = None,
    lower_bound: float = -0.50,
    upper_bound: float = 1.50,
) -> DynamicHrThresholdResult:
    deployed = _finite(deployed_threshold, 1.0)
    head = infer_head(strategy_id)
    if not enabled:
        return DynamicHrThresholdResult(deployed, False, "disabled", head)
    if head == "unknown":
        return DynamicHrThresholdResult(deployed, False, "unknown_head", head)
    item: Optional[DynamicHrHeadState] = (state or {}).get(head)
    if item is None:
        return DynamicHrThresholdResult(deployed, False, "missing_head_state", head)
    age_days = _state_age_days(item.as_of, now)
    if max_state_age_days is not None and age_days > float(max_state_age_days):
        threshold = deployed if stale_fallback_to_deployed else max(deployed, 1.50)
        return DynamicHrThresholdResult(
            float(threshold),
            bool(abs(float(threshold) - deployed) > 1e-12),
            "stale_head_state",
            head,
            z_eff=float(item.z_eff),
            guarded_y=float(item.guarded_y),
            w_lower=float(item.w_lower),
            w_raise=float(item.w_raise),
            state_age_days=float(age_days),
        )
    if item.dynamic_rejected or item.fallback_to_deployed:
        threshold = deployed if fallback_to_deployed else max(deployed, 1.50)
        return DynamicHrThresholdResult(
            float(threshold),
            bool(abs(float(threshold) - deployed) > 1e-12),
            item.reason or "dynamic_rejected_fallback_to_deployed",
            head,
            z_eff=float(item.z_eff),
            guarded_y=float(item.guarded_y),
            w_lower=float(item.w_lower),
            w_raise=float(item.w_raise),
            state_age_days=float(age_days),
        )
    if item.deactivated:
        return DynamicHrThresholdResult(
            float(max(deployed, 1.50)),
            True,
            item.reason or "dynamic_head_deactivated",
            head,
            z_eff=float(item.z_eff),
            guarded_y=float(item.guarded_y),
            w_lower=float(item.w_lower),
            w_raise=float(item.w_raise),
            state_age_days=float(age_days),
        )
    z_eff = _finite(item.z_eff, 0.0)
    threshold = (
        float(item.guarded_y)
        - float(item.w_lower) * max(0.0, z_eff)
        - float(item.w_raise) * min(0.0, z_eff)
    )
    if use_deployed_floor:
        floor = deployed
        if np.isfinite(item.deployed_threshold):
            floor = max(floor, float(item.deployed_threshold))
        threshold = max(floor, threshold)
    threshold = float(np.clip(threshold, float(lower_bound), float(upper_bound)))
    return DynamicHrThresholdResult(
        threshold,
        bool(abs(threshold - deployed) > 1e-12),
        "applied",
        head,
        z_eff=float(z_eff),
        guarded_y=float(item.guarded_y),
        w_lower=float(item.w_lower),
        w_raise=float(item.w_raise),
        state_age_days=float(age_days),
    )
