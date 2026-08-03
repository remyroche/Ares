#!/usr/bin/env python3
"""Backfill a comparable, leakage-safe historical 12h execution-EV panel.

This deliberately does *not* reuse the historical 96-bar first-touch label.
It rebuilds labels over the canonical Kraken Futures hourly OHLCV store with
the current side-parent execution geometry, a one-hour signal-to-decision
delay, a 12-hour resolution timestamp and the policy's explicit round-trip
fee.  Historical score shards define a frozen pre-entry candidate stream;
only their pre-entry score/context fields are model inputs.

The output is an honest comparator, not a claim of full 1-minute fill parity:
the 1-minute archive starts in 2025 and true historical L2 spread is absent.
Those coverage limits are emitted rather than filled or silently bridged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.data_store import PartitionedOHLCVStore
from extreme_price_movements.execution_ev_labels import (
    ExecutionLabelGeometry,
    reason_names,
    simulate_execution_ev_12h,
)


SCHEMA = "historical_comparable_execution_ev_12h_oof_v1"
ID = ("__ts__", "__symbol__", "side_name", "candidate_id")
FEATURES = (
    "score_base",
    "base_margin_to_cutoff",
    "base_margin_to_cutoff_z",
    "base_signal_zscore_within_archetype",
    "base_score_rank_pct_train_prior",
    "support_min_log_count",
    "support_mean_log_count",
    "support_min_frequency",
    "support_mean_frequency",
)
HORIZON_HOURS = 12
DECISION_DELAY_HOURS = 1
MIN_TRAIN_DAYS = 60


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    raw = {str(k): _json_safe(v) for k, v in payload.items() if k != "manifest_sha256"}
    return hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def normalize_historical_symbol(value: str) -> str:
    """Map archived ``BTC/USD:USD`` notation to the canonical raw-store key."""

    value = str(value).strip()
    if not value:
        raise ValueError("blank historical symbol")
    return value.replace("/", "_")


def stable_candidate_id(ts: pd.Timestamp, symbol: str, side: str) -> str:
    payload = f"historical_execution_ev_v1|{pd.Timestamp(ts).isoformat()}|{symbol}|{side}"
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def _selected_shards(root: Path, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    source = root / "data_perp/artifacts/20260713_meta_fullhistory_old55_expandedpool/prediction_shards"
    shards = sorted(source.glob("predictions_*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no historical prediction shards under {source}")
    selected: list[Path] = []
    for path in shards:
        # Filenames are only an optimization; timestamps are filtered below.
        selected.append(path)
    return selected


def load_candidate_stream(root: Path, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, list[Path]]:
    parts: list[pd.DataFrame] = []
    shards = _selected_shards(root, start, end)
    used: list[Path] = []
    use = ["__ts__", "__symbol__", "side_name", *FEATURES]
    for path in shards:
        try:
            frame = pd.read_parquet(path, columns=use)
        except Exception:
            # The archive is heterogeneous; a shard without the fixed causal
            # context is reported by omission rather than relaxed to a label.
            continue
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
        frame = frame.loc[(frame["__ts__"] >= start) & (frame["__ts__"] < end)].copy()
        if frame.empty:
            continue
        used.append(path)
        frame["__symbol__"] = frame["__symbol__"].map(normalize_historical_symbol)
        frame["side_name"] = frame["side_name"].astype(str).str.lower()
        if not frame["side_name"].isin(("long", "short")).all():
            raise ValueError(f"{path}: invalid side")
        parts.append(frame)
    if not parts:
        raise ValueError("no historical candidate rows in requested range")
    out = pd.concat(parts, ignore_index=True)
    out = out.dropna(subset=["__ts__", "__symbol__", "side_name"]).copy()
    out["candidate_id"] = [stable_candidate_id(*row) for row in out[["__ts__", "__symbol__", "side_name"]].itertuples(index=False, name=None)]
    if out.duplicated(list(ID)).any():
        raise ValueError("historical candidate stream has duplicate identities")
    out["execution_decision_utc"] = out["__ts__"] + pd.Timedelta(hours=DECISION_DELAY_HOURS)
    out["execution_label_end_utc"] = out["execution_decision_utc"] + pd.Timedelta(hours=HORIZON_HOURS)
    out["candidate_month"] = out["__ts__"].dt.to_period("M").astype(str)
    return out.sort_values(list(ID), kind="stable").reset_index(drop=True), used


def _current_geometry(policy_path: Path) -> tuple[ExecutionLabelGeometry, ExecutionLabelGeometry, float]:
    raw = json.loads(policy_path.read_text())
    parents = {str(item.get("side")): item for item in raw.get("strategies", []) if item.get("canonical_strategy_id") in {"long__parent", "short__parent"}}
    if set(parents) != {"long", "short"}:
        raise ValueError("policy is missing exact side-parent geometry")

    def convert(item: Mapping[str, Any]) -> ExecutionLabelGeometry:
        return ExecutionLabelGeometry.from_mapping({
            "sl_mult": item["sl_mult"],
            "trailing_activation_mult": item["trailing_activation_mult"],
            "trailing_activation_cap_pct": item.get("trailing_activation_cap_pct", 0.0),
            "trailing_activation_decay_half_life_minutes": item.get("trailing_activation_decay_half_life_bars", 0.0),
            "trailing_activation_decay_start_minutes": item.get("trailing_activation_decay_start_bars", 0.0),
            "trailing_activation_min_mult": item.get("trailing_activation_min_mult", 1.0),
            "trailing_power": item.get("trailing_power", 1.5),
            "trailing_squash_divisor": item.get("trailing_squash_divisor", 2.0),
            "giveback_beta": item.get("giveback_beta", 0.5),
            "adverse_exit_enabled": item.get("adverse_exit_enabled", False),
            "adverse_exit_min_mae_atr": item.get("adverse_exit_min_mae_atr", 1.0),
            "adverse_exit_min_speed_per_15m": item.get("adverse_exit_min_speed", 0.3),
            "adverse_exit_theta": item.get("adverse_exit_theta", 1e9),
            "adverse_exit_fast_minutes": item.get("adverse_exit_fast_bars", 0),
            "adverse_exit_max_mfe_atr": item.get("adverse_exit_max_mfe_atr", 0.25),
        })

    costs = {float(item["cost_pct_per_side"]) for item in parents.values()}
    if len(costs) != 1:
        raise ValueError("side parent policies disagree on fee cost")
    return convert(parents["long"]), convert(parents["short"]), 2.0 * costs.pop()


def _paths_for_symbol(store: PartitionedOHLCVStore, symbol: str, decisions: pd.Series) -> tuple[tuple[np.ndarray, ...], np.ndarray, pd.Series]:
    start = decisions.min() - pd.Timedelta(hours=16)
    end = decisions.max() + pd.Timedelta(hours=HORIZON_HOURS)
    bars = store.load(symbol, columns=["open", "high", "low", "close"], start_ts=start, end_ts=end)
    shape = (len(decisions), HORIZON_HOURS)
    blank = tuple(np.full(shape, np.nan, dtype=np.float32) for _ in range(4))
    atr = pd.Series(np.nan, index=decisions.index, dtype=np.float32)
    if bars.empty:
        return blank, np.zeros(len(decisions), dtype=bool), atr
    bars = bars.loc[~bars.index.duplicated(keep="last")].sort_index()
    bars.index = pd.to_datetime(bars.index, utc=True)
    close = pd.to_numeric(bars["close"], errors="coerce")
    prev = close.shift(1)
    tr = pd.concat([pd.to_numeric(bars["high"], errors="coerce") - pd.to_numeric(bars["low"], errors="coerce"), (pd.to_numeric(bars["high"], errors="coerce") - prev).abs(), (pd.to_numeric(bars["low"], errors="coerce") - prev).abs()], axis=1).max(axis=1)
    atr_frac = tr.rolling(14, min_periods=14).mean() / close
    atr = atr_frac.reindex(pd.DatetimeIndex(decisions.to_numpy())).set_axis(decisions.index).astype(np.float32)
    index_ns = bars.index.astype("int64").to_numpy(np.int64)
    decision_ns = pd.DatetimeIndex(decisions).astype("int64").to_numpy(np.int64)
    starts = np.searchsorted(index_ns, decision_ns)
    offsets = np.arange(HORIZON_HOURS, dtype=np.int64)
    positions = starts[:, None] + offsets[None, :]
    valid = positions[:, -1] < len(index_ns)
    local = np.flatnonzero(valid)
    if len(local):
        expected = decision_ns[local, None] + offsets[None, :] * 3_600_000_000_000
        valid[local] = np.all(index_ns[positions[local]] == expected, axis=1)
    out: list[np.ndarray] = []
    for col in ("open", "high", "low", "close"):
        values = pd.to_numeric(bars[col], errors="coerce").to_numpy(np.float32)
        target = np.full(shape, np.nan, dtype=np.float32)
        local = np.flatnonzero(valid)
        if len(local):
            target[local] = values[positions[local]]
        out.append(target)
    valid &= np.isfinite(atr.to_numpy(np.float64)) & (atr.to_numpy(np.float64) > 0)
    valid &= np.logical_and.reduce([np.isfinite(item).all(axis=1) for item in out])
    return tuple(out), valid, atr


def materialize_labels(candidates: pd.DataFrame, raw_root: Path, policy: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    long_g, short_g, fee = _current_geometry(policy)
    store = PartitionedOHLCVStore(str(raw_root), timeframe="1h")
    result: list[pd.DataFrame] = []
    coverage: list[pd.DataFrame] = []
    for symbol, idx in candidates.groupby("__symbol__", sort=True).groups.items():
        rows = candidates.loc[list(idx)].copy().reset_index(drop=True)
        paths, valid, atr = _paths_for_symbol(store, str(symbol), rows["execution_decision_utc"])
        rows["atr_fraction_14h"] = atr.to_numpy(np.float32)
        rows["complete_hourly_path"] = valid
        coverage.append(rows.loc[:, [*ID, "candidate_month", "complete_hourly_path"]])
        if not valid.any():
            continue
        local = rows.loc[valid].reset_index(drop=True)
        gross, net, reason, exit_bar, mfe, mae = simulate_execution_ev_12h(
            *(item[valid] for item in paths),
            np.where(local["side_name"].eq("long"), 1.0, -1.0).astype(np.float64),
            local["atr_fraction_14h"].to_numpy(np.float64),
            np.full(len(local), fee, dtype=np.float64),
            long_g.vector(), short_g.vector(), 60,
        )
        local["execution_gross_ev_12h"] = gross.astype(np.float32)
        local["execution_net_ev_12h"] = net.astype(np.float32)
        local["execution_cost_return"] = np.float32(fee)
        local["execution_exit_reason"] = reason_names(reason)
        local["execution_exit_hour"] = exit_bar.astype(np.float32)
        local["execution_mfe_return_12h"] = mfe.astype(np.float32)
        local["execution_mae_return_12h"] = mae.astype(np.float32)
        result.append(local)
    if not result:
        raise ValueError("no complete canonical hourly paths")
    labels = pd.concat(result, ignore_index=True).sort_values(list(ID), kind="stable").reset_index(drop=True)
    cov = pd.concat(coverage, ignore_index=True)
    if not np.allclose(labels["execution_gross_ev_12h"] - labels["execution_cost_return"], labels["execution_net_ev_12h"], atol=1e-7, rtol=0):
        raise ValueError("gross-cost reconciliation failed")
    return labels, cov, {"round_trip_fee_return": fee, "long": long_g.vector().tolist(), "short": short_g.vector().tolist()}


def strict_monthly_oof(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    out: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    months = sorted(frame["candidate_month"].unique())
    for side in ("long", "short"):
        side_frame = frame.loc[frame["side_name"].eq(side)].copy()
        for month in months:
            eval_rows = side_frame.loc[side_frame["candidate_month"].eq(month)].copy()
            start = pd.Timestamp(f"{month}-01", tz="UTC")
            cutoff = start - pd.Timedelta(hours=HORIZON_HOURS)
            train = side_frame.loc[(side_frame["execution_label_end_utc"] < start) & (side_frame["__ts__"] < cutoff)].copy()
            enough_age = not train.empty and (start - train["__ts__"].min() >= pd.Timedelta(days=MIN_TRAIN_DAYS))
            audit = {"side": side, "month": month, "train_rows": int(len(train)), "eval_rows": int(len(eval_rows)), "train_cutoff_utc": cutoff, "status": "trained" if enough_age and len(train) >= 1000 and len(eval_rows) else "insufficient_prior_history"}
            if audit["status"] != "trained":
                audits.append(audit); continue
            medians = train.loc[:, list(FEATURES)].apply(pd.to_numeric, errors="coerce").median().fillna(0.0)
            x_train = train.loc[:, list(FEATURES)].apply(pd.to_numeric, errors="coerce").fillna(medians).to_numpy(np.float32)
            x_eval = eval_rows.loc[:, list(FEATURES)].apply(pd.to_numeric, errors="coerce").fillna(medians).to_numpy(np.float32)
            model = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1e-4, random_state=42)
            model.fit(x_train, pd.to_numeric(train["execution_net_ev_12h"], errors="raise").to_numpy(np.float32))
            eval_rows["historical_direct_ev_oof"] = model.predict(x_eval).astype(np.float32)
            eval_rows["oof_train_cutoff_utc"] = cutoff
            eval_rows["oof_fold_month"] = month
            out.append(eval_rows)
            audits.append(audit)
    if not out:
        raise ValueError("no strict OOF folds trained")
    return pd.concat(out, ignore_index=True).sort_values(list(ID), kind="stable").reset_index(drop=True), audits


def _top10_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for month, group in frame.groupby("candidate_month", sort=True):
        n = max(1, int(np.ceil(0.10 * len(group))))
        top = group.nlargest(n, "historical_direct_ev_oof")
        result[str(month)] = {"rows": int(len(group)), "top10_rows": int(len(top)), "top10_net_ev_bps": float(top["execution_net_ev_12h"].mean() * 1e4), "top10_positive_rate": float((top["execution_net_ev_12h"] > 0).mean())}
    return result


def run(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    start, end = pd.Timestamp(args.start, tz="UTC"), pd.Timestamp(args.end, tz="UTC")
    candidates, shards = load_candidate_stream(ROOT, start, end)
    labels, coverage, geometry = materialize_labels(candidates, args.raw_root, args.policy)
    oof, folds = strict_monthly_oof(labels)
    coverage["coverage_month"] = coverage["candidate_month"]
    coverage_table = coverage.groupby("coverage_month", sort=True).agg(candidate_rows=("candidate_id", "size"), complete_hourly_paths=("complete_hourly_path", "sum")).reset_index()
    coverage_table["missing_hourly_paths"] = coverage_table["candidate_rows"] - coverage_table["complete_hourly_paths"]
    coverage_table["hourly_path_coverage"] = coverage_table["complete_hourly_paths"] / coverage_table["candidate_rows"]
    # Pre-archive period remains explicit: no historical OOF candidate source
    # exists before March 2025, and 1m replay is absent in 2024.
    unavailable = pd.DataFrame({"coverage_month": pd.period_range("2024-10", "2025-02", freq="M").astype(str), "candidate_rows": 0, "complete_hourly_paths": 0, "missing_hourly_paths": 0, "hourly_path_coverage": np.nan, "status": "no_archived_causal_candidate_stream"})
    coverage_table["status"] = "available"
    coverage_table = pd.concat([unavailable, coverage_table], ignore_index=True)
    labels.to_parquet(output_dir / "historical_execution_ev_12h_labels.parquet", index=False, compression="zstd")
    oof.to_parquet(output_dir / "historical_direct_ev_strict_oof.parquet", index=False, compression="zstd")
    coverage_table.to_csv(output_dir / "coverage_by_month.csv", index=False)
    _write_json(output_dir / "fold_audit.json", {"folds": folds})
    metrics = _top10_metrics(oof)
    source_manifest = ROOT / "data_perp/artifacts/20260713_meta_fullhistory_old55_expandedpool/manifest.json"
    summary = {"schema": SCHEMA, "period": {"requested_start": start, "requested_end_exclusive": end, "materialized_start": labels["__ts__"].min(), "materialized_end": labels["__ts__"].max()}, "rows": {"candidate": int(len(candidates)), "labels": int(len(labels)), "strict_oof": int(len(oof))}, "features": list(FEATURES), "target": {"name": "execution_net_ev_12h", "horizon_hours": HORIZON_HOURS, "signal_to_decision_hours": DECISION_DELAY_HOURS, "resolution": "decision+12h", "simulator": "simulate_execution_ev_12h", "policy_path": str(args.policy), "policy_sha256": _sha256(args.policy), "cost": "current policy side-parent fee only; historical L2/spread unavailable"}, "geometry": geometry, "oof_contract": {"per_side": True, "walk_forward": "expanding monthly", "purge_hours": HORIZON_HOURS, "embargo_hours": HORIZON_HOURS, "minimum_prior_history_days": MIN_TRAIN_DAYS}, "source_score_contract": {"source_manifest": str(source_manifest), "source_manifest_sha256": _sha256(source_manifest), "source_evidence": "expanding_window_with_oos_age_cap; outcomes joined only for training labels and validation metrics", "allowed_inputs": list(FEATURES), "forbidden": "all archived first-touch/96-bar labels and outcome columns", "representation_selection_exception": {"status": "approved_diagnostic_exception_not_untouched_strict_oos", "detail": "The common 55-feature representation was selected on the July-2026 largest fold and reused backward. Each score_base row is still produced by a prior-row-only fold model, but pre-July feature-family selection is future-informed; use this panel for diagnosis and recurrence research, not strict promotion evidence."}}, "raw_coverage_limits": {"late_2024": "no archived pre-entry candidate stream; not synthesized", "one_minute": "not available in 2024; output is canonical hourly target", "funding": "excluded; authoritative historical feed begins 2026-04-22", "spread_depth": "no trustworthy historical L2; no spread inference substituted"}, "metrics": {"overall_top10_net_ev_bps": float(oof.nlargest(max(1, int(np.ceil(.1 * len(oof)))), "historical_direct_ev_oof")["execution_net_ev_12h"].mean() * 1e4), "by_month": metrics}, "source_shards": [str(p) for p in shards], "artifacts": {name: _sha256(output_dir / name) for name in ["historical_execution_ev_12h_labels.parquet", "historical_direct_ev_strict_oof.parquet", "coverage_by_month.csv", "fold_audit.json"]}}
    summary["manifest_sha256"] = _canonical_hash(summary)
    _write_json(output_dir / "summary.json", summary)
    return {"labels": output_dir / "historical_execution_ev_12h_labels.parquet", "oof": output_dir / "historical_direct_ev_strict_oof.parquet", "summary": output_dir / "summary.json"}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", type=Path, default=ROOT / "data_perp/exchanges/krakenfutures")
    p.add_argument("--policy", type=Path, default=ROOT / "data_perp/reports/simple_policy_1m_joint_trailing_raw_bayesian_champion_20260718_v1/production_staging/best_policy_params.json")
    p.add_argument("--start", default="2024-10-01")
    p.add_argument("--end", default="2026-05-01")
    p.add_argument("--output-dir", type=Path, required=True)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    outputs = run(args)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
