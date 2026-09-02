#!/usr/bin/env python3
"""Learn a causal Meta-HPO objective from strict-OOF downstream evidence.

This is an *offline research* utility.  It never writes a model used by the
live trader.  Its purpose is to replace hand-designed Meta-HPO selection with
two conservative surrogate objectives learned from completed strict-OOF
experiments:

``PriorityProxy``
    Which Meta trials improve timestamp-local candidate ordering at a matched
    budget?

``GateProxy``
    Which trials improve the real dual-MC1 admission population without
    creating an unstable or sparse gate?

The score receipt / outcome boundary is deliberate.  ``collect`` opens only
target-free Base/Meta scores to form every candidate diagnostic.  It opens the
canonical policy labels only afterwards, for OOF evaluation.  The full MC1 and
portfolio labels are independently read from completed downstream receipts.

New feature families are always recorded as *additive overlays* to the frozen
current P8U contract.  Candidate-only feature contracts are rejected by the
trial-bank generator and are never a recommendation target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCHEMA = "strict_r3_p8u_meta_hpo_surrogate_v1"
IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
POLICY_FORBIDDEN = frozenset(
    {
        "policy_path_valid",
        "policy_net_bps",
        "policy_gross_bps",
        "policy_exit_bar_15m",
        "policy_entry_price",
        "policy_exit_price",
        "policy_label_available_ts",
        "policy_exit_reason",
    }
)


def _once(path: Path, payload: object) -> None:
    """Write immutable receipts; never silently replace research evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_months(raw: str) -> tuple[pd.Timestamp, ...]:
    result = tuple(
        pd.Timestamp(f"{value.strip()}-01", tz="UTC")
        for value in raw.split(",")
        if value.strip()
    )
    if not result or tuple(sorted(result)) != result or len(set(result)) != len(result):
        raise ValueError("--months must be unique ascending YYYY-MM values")
    return result


def _safe_float(value: object, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _utc_timestamp(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _rank_desc(frame: pd.DataFrame, score: str) -> pd.Series:
    """Timestamp-local deterministic descending rank in [0, 1]."""
    work = frame.loc[:, ["candidate_id", "__decision_ts__", score]].copy()
    work["__row__"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(
        ["__decision_ts__", score, "candidate_id"],
        ascending=[True, False, True],
        kind="stable",
    )
    ordinal = work.groupby("__decision_ts__", sort=False).cumcount().to_numpy(float) + 1.0
    size = work.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy(float)
    values = np.empty(len(work), dtype=np.float32)
    values[work["__row__"].to_numpy(np.int64)] = (1.0 - (ordinal - 0.5) / size).astype(np.float32)
    return pd.Series(values, index=frame.index, name=f"{score}_rank_ts")


def _topk_mask(frame: pd.DataFrame, score: str, k: int) -> pd.Series:
    if frame.empty:
        return pd.Series(False, index=frame.index)
    work = frame.loc[:, ["candidate_id", "__decision_ts__", score]].copy()
    work["__row__"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(
        ["__decision_ts__", score, "candidate_id"],
        ascending=[True, False, True],
        kind="stable",
    )
    work["__picked__"] = work.groupby("__decision_ts__", sort=False).cumcount().lt(int(k))
    result = np.zeros(len(frame), dtype=bool)
    result[work["__row__"].to_numpy(np.int64)] = work["__picked__"].to_numpy(bool)
    return pd.Series(result, index=frame.index)


def _mean_or_nan(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else np.nan


def _quantile_or_nan(values: pd.Series | np.ndarray, q: float) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.quantile(array, q)) if len(array) else np.nan


def _correlation(left: pd.Series, right: pd.Series) -> float:
    valid = left.notna() & right.notna() & np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 8:
        return np.nan
    value = spearmanr(left.loc[valid], right.loc[valid]).statistic
    return float(value) if np.isfinite(value) else np.nan


def _partial_rank_correlation(meta: pd.Series, base: pd.Series, outcome: pd.Series) -> float:
    """A light, deterministic CMI proxy: partial Spearman rank correlation."""
    valid = meta.notna() & base.notna() & outcome.notna()
    if int(valid.sum()) < 32:
        return np.nan
    matrix = np.column_stack(
        [
            pd.Series(meta.loc[valid]).rank(method="average").to_numpy(float),
            pd.Series(base.loc[valid]).rank(method="average").to_numpy(float),
            pd.Series(outcome.loc[valid]).rank(method="average").to_numpy(float),
        ]
    )
    x = matrix[:, 0]
    b = np.column_stack([np.ones(len(matrix)), matrix[:, 1]])
    xr = x - b @ np.linalg.lstsq(b, x, rcond=None)[0]
    yr = matrix[:, 2] - b @ np.linalg.lstsq(b, matrix[:, 2], rcond=None)[0]
    denom = float(np.linalg.norm(xr) * np.linalg.norm(yr))
    return float(np.dot(xr, yr) / denom) if denom > 0 else np.nan


def _weekly_selected(frame: pd.DataFrame, score: str, k: int) -> pd.DataFrame:
    selected = frame.loc[_topk_mask(frame, score, k), ["__decision_ts__", "policy_net_bps"]].copy()
    if selected.empty:
        return pd.DataFrame(columns=["week", "ev_bps", "trades"])
    selected["week"] = selected["__decision_ts__"].dt.normalize() - pd.to_timedelta(selected["__decision_ts__"].dt.dayofweek, unit="D")
    return selected.groupby("week", as_index=False).agg(ev_bps=("policy_net_bps", "mean"), trades=("policy_net_bps", "size"))


def _aggregate_weekly(weekly: pd.DataFrame, prefix: str) -> dict[str, float]:
    if weekly.empty:
        return {f"{prefix}_{suffix}": np.nan for suffix in ("mean", "q25", "q10", "q05", "positive_fraction", "se")}
    values = weekly.ev_bps.to_numpy(float)
    return {
        f"{prefix}_mean": _mean_or_nan(values),
        f"{prefix}_q25": _quantile_or_nan(values, .25),
        f"{prefix}_q10": _quantile_or_nan(values, .10),
        f"{prefix}_q05": _quantile_or_nan(values, .05),
        f"{prefix}_positive_fraction": float(np.mean(values > 0.0)),
        f"{prefix}_se": float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else np.nan,
    }


def _read_policy(path: Path) -> pd.DataFrame:
    columns = [*IDENTITY, "policy_path_valid", "policy_net_bps", "policy_label_available_ts"]
    result = pd.read_parquet(path, columns=columns)
    result["__decision_ts__"] = pd.to_datetime(result["__decision_ts__"], utc=True, errors="raise")
    result["policy_label_available_ts"] = pd.to_datetime(result["policy_label_available_ts"], utc=True, errors="raise")
    if result.duplicated(list(IDENTITY)).any():
        raise AssertionError(f"{path}: duplicate canonical policy identity")
    return result


def _target_free_scores(root: Path, trial: str, months: Sequence[pd.Timestamp]) -> pd.DataFrame:
    source = root / "target_free_scores" / trial
    pieces: list[pd.DataFrame] = []
    for month in months:
        path = source / f"month={month:%Y-%m}.parquet"
        if not path.exists():
            return pd.DataFrame()
        score = pd.read_parquet(path)
        forbidden = POLICY_FORBIDDEN.intersection(score.columns)
        if forbidden:
            raise AssertionError(f"{path}: target-free Meta receipt leaks {sorted(forbidden)}")
        need = set(IDENTITY).union({"base_rank_ts", "meta_rank_ts"})
        missing = need.difference(score.columns)
        if missing:
            raise AssertionError(f"{path}: missing target-free score fields {sorted(missing)}")
        score = score.loc[:, [*IDENTITY, "base_rank_ts", "meta_rank_ts"]].copy()
        score["__decision_ts__"] = pd.to_datetime(score["__decision_ts__"], utc=True, errors="raise")
        pieces.append(score)
    result = pd.concat(pieces, ignore_index=True)
    if result.duplicated(list(IDENTITY)).any() or not result.side_name.eq("long").all():
        raise AssertionError(f"{root} {trial}: invalid target-free identity")
    return result


def _descriptor_from_scores(scores: pd.DataFrame, policy: pd.DataFrame, *, trial_key: str) -> tuple[dict[str, float], pd.DataFrame]:
    """Build cheap diagnostics after target-free scores have been persisted."""
    frame = scores.merge(policy, on=list(IDENTITY), how="left", validate="one_to_one")
    valid = (
        frame.policy_path_valid.fillna(False).astype(bool)
        & frame.policy_net_bps.notna()
        & frame.policy_label_available_ts.notna()
    )
    frame = frame.loc[valid & frame.base_rank_ts.ge(.70)].copy()
    if frame.empty:
        return {"valid_policy_rows": 0.0}, pd.DataFrame()
    # This coordinate is only a fixed, cheap probe.  It does not change any
    # deployed score; full downstream MC1 remains the expensive label.
    frame["cheap_meta_coordinate"] = .75 * frame.base_rank_ts + .25 * frame.meta_rank_ts
    frame["cheap_meta_rank_ts"] = _rank_desc(frame, "cheap_meta_coordinate")
    frame["base_tail_pct"] = 100.0 * (1.0 - frame.base_rank_ts)
    frame["correction"] = frame.meta_rank_ts - frame.base_rank_ts
    metrics: dict[str, float] = {"valid_policy_rows": float(len(frame)), "timestamps": float(frame.__decision_ts__.nunique())}
    metrics["residual_spearman_ic"] = _correlation(frame.meta_rank_ts, frame.policy_net_bps)
    metrics["conditional_rank_ic_given_base"] = _partial_rank_correlation(frame.meta_rank_ts, frame.base_rank_ts, frame.policy_net_bps)
    metrics["base_meta_rank_spearman"] = _correlation(frame.base_rank_ts, frame.meta_rank_ts)
    metrics["correction_abs_median"] = _quantile_or_nan(np.abs(frame.correction), .50)
    metrics["correction_abs_p90"] = _quantile_or_nan(np.abs(frame.correction), .90)
    metrics["correction_up_fraction"] = float(np.mean(frame.correction > 0.0))
    metrics["correction_down_fraction"] = float(np.mean(frame.correction < 0.0))
    metrics["meta_policy_above_50_fraction"] = float(np.mean(frame.policy_net_bps > 50.0))
    metrics["meta_policy_above_100_fraction"] = float(np.mean(frame.policy_net_bps > 100.0))
    for label, lo, hi in (("0_5", 0, 5), ("5_10", 5, 10), ("10_20", 10, 20), ("20_30", 20, 30)):
        part = frame.loc[frame.base_tail_pct.ge(lo) & frame.base_tail_pct.lt(hi)]
        metrics[f"ic_base_band_{label}"] = _correlation(part.meta_rank_ts, part.policy_net_bps)
    base_top: dict[int, pd.Series] = {}
    meta_top: dict[int, pd.Series] = {}
    for k in (1, 2, 5):
        base_top[k] = _topk_mask(frame, "base_rank_ts", k)
        meta_top[k] = _topk_mask(frame, "cheap_meta_coordinate", k)
        metrics[f"meta_top{k}_policy_net_bps"] = _mean_or_nan(frame.loc[meta_top[k], "policy_net_bps"])
        metrics[f"base_top{k}_policy_net_bps"] = _mean_or_nan(frame.loc[base_top[k], "policy_net_bps"])
        metrics[f"cheap_probe_delta_top{k}_bps"] = metrics[f"meta_top{k}_policy_net_bps"] - metrics[f"base_top{k}_policy_net_bps"]
        metrics[f"top{k}_overlap_fraction"] = float(np.mean(meta_top[k].to_numpy(bool) & base_top[k].to_numpy(bool)))
        added = meta_top[k] & ~base_top[k]
        removed = base_top[k] & ~meta_top[k]
        metrics[f"top{k}_added_policy_net_bps"] = _mean_or_nan(frame.loc[added, "policy_net_bps"])
        metrics[f"top{k}_removed_policy_net_bps"] = _mean_or_nan(frame.loc[removed, "policy_net_bps"])
        metrics[f"top{k}_substitution_delta_bps"] = metrics[f"top{k}_added_policy_net_bps"] - metrics[f"top{k}_removed_policy_net_bps"]
    # Four quadrants expose Base/Meta disagreement rather than hiding it in a
    # blend.  The 0.90 threshold is inside the routed top-30 population.
    for b, m, name in ((True, True, "base_high_meta_high"), (True, False, "base_high_meta_low"), (False, True, "base_low_meta_high"), (False, False, "base_low_meta_low")):
        mask = (frame.base_rank_ts.ge(.90) if b else frame.base_rank_ts.lt(.90)) & (frame.meta_rank_ts.ge(.90) if m else frame.meta_rank_ts.lt(.90))
        metrics[f"{name}_share"] = float(np.mean(mask))
        metrics[f"{name}_policy_net_bps"] = _mean_or_nan(frame.loc[mask, "policy_net_bps"])
    weekly = _weekly_selected(frame, "cheap_meta_coordinate", 2)
    for key, value in _aggregate_weekly(weekly, "cheap_meta_top2_week").items():
        metrics[key] = value
    base_weekly = _weekly_selected(frame, "base_rank_ts", 2).rename(columns={"ev_bps": "base_ev_bps"})
    weekly = weekly.merge(base_weekly.loc[:, ["week", "base_ev_bps"]], on="week", how="left")
    weekly["trial_key"] = trial_key
    weekly["delta_ev_bps"] = weekly.ev_bps - weekly.base_ev_bps
    metrics["cheap_probe_delta_top2_week_q10_bps"] = _quantile_or_nan(weekly.delta_ev_bps, .10)
    metrics["cheap_probe_delta_top2_utility_bps"] = float((frame.loc[meta_top[2], "policy_net_bps"].sum() - frame.loc[base_top[2], "policy_net_bps"].sum()))
    monthly = frame.loc[meta_top[2]].groupby(frame.loc[meta_top[2], "__decision_ts__"].dt.strftime("%Y-%m")).policy_net_bps.mean()
    metrics["cheap_meta_top2_month_median_bps"] = _quantile_or_nan(monthly, .50)
    metrics["cheap_meta_top2_month_q25_bps"] = _quantile_or_nan(monthly, .25)
    metrics["cheap_meta_top2_positive_month_fraction"] = float(np.mean(monthly.to_numpy(float) > 0.0)) if len(monthly) else np.nan
    metrics["cheap_meta_top2_worst_month_bps"] = _quantile_or_nan(monthly, 0.0)
    return metrics, weekly


def _flatten_trial(trial: Mapping[str, Any]) -> dict[str, object]:
    model = trial.get("model", {}) if isinstance(trial.get("model", {}), Mapping) else {}
    result: dict[str, object] = {
        "trial": str(trial.get("name", trial.get("trial", ""))),
        "loss_family": str(model.get("objective", trial.get("objective", "unknown"))),
        "gain_signature": json.dumps(trial.get("gain", []), separators=(",", ":")),
        "truncation": _safe_float(trial.get("truncation")),
        "sigmoid": _safe_float(trial.get("sigmoid")),
        "sample_weight_family": "unweighted" if trial.get("sample_weight") is None else str(trial.get("sample_weight")),
    }
    for name in ("learning_rate", "max_depth", "num_leaves", "min_child_samples", "min_split_gain", "feature_fraction", "bagging_fraction", "lambda_l1", "lambda_l2", "n_estimators"):
        result[f"model_{name}"] = _safe_float(model.get(name))
    gain = np.asarray(trial.get("gain", []), dtype=float)
    result["gain_max"] = float(np.nanmax(gain)) if len(gain) else np.nan
    result["gain_tail_gap"] = float(gain[-1] - gain[-2]) if len(gain) >= 2 else np.nan
    return result


def _contract_mode(manifest: Mapping[str, Any]) -> tuple[str, float]:
    path = manifest.get("meta_feature_contract")
    count = _safe_float(manifest.get("meta_feature_count"))
    if not path:
        # Older LambdaRank receipts predate explicit feature-contract hashes.
        # Their manifest still proves use of the frozen F72 Base and current
        # target-free Meta panel, so they are not replacement contracts.
        if "F72" in str(manifest.get("base_contract", "")) and isinstance(manifest.get("source"), Mapping):
            return "implicit_frozen_current", count if np.isfinite(count) else 72.0
        return "unknown", count
    try:
        payload = json.loads(Path(str(path)).read_text())
    except (OSError, json.JSONDecodeError):
        return "unknown", count
    if payload.get("parent_feature_contract"):
        return "additive_overlay", count
    return "frozen_parent", count


def _collect_descriptors(args: argparse.Namespace) -> Path:
    artifacts = Path(args.artifacts).resolve()
    policy_path = Path(args.policy).resolve()
    out = Path(args.out).resolve()
    months = _as_months(args.months)
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    policy = _read_policy(policy_path)
    registry: list[dict[str, object]] = []
    descriptors: list[dict[str, object]] = []
    weekly_rows: list[pd.DataFrame] = []
    roots = sorted(artifacts.glob(args.pattern))
    for objective_path in roots:
        if objective_path.name != "objective_summary.parquet":
            continue
        receipt_root = objective_path.parent
        score_root = receipt_root / "target_free_scores"
        manifest_path = receipt_root / "run_manifest.json"
        if not score_root.exists() or not manifest_path.exists():
            continue
        try:
            summary = pd.read_parquet(objective_path)
            manifest = json.loads(manifest_path.read_text())
        except Exception as exc:  # receipt scans must be resilient, but auditable
            registry.append({"receipt_root": str(receipt_root), "status": "unreadable", "detail": str(exc)})
            continue
        trial_specs = {str(item.get("name", item.get("trial", ""))): item for item in manifest.get("trials", []) if isinstance(item, Mapping)}
        mode, feature_count = _contract_mode(manifest)
        for _, summary_row in summary.iterrows():
            trial = str(summary_row.get("trial", ""))
            source_key = f"{receipt_root.resolve()}::{trial}"
            key = f"{source_key}::era={args.era}" if args.era else source_key
            scores = _target_free_scores(receipt_root, trial, months)
            if scores.empty:
                registry.append({"trial_key": key, "receipt_root": str(receipt_root.resolve()), "trial": trial, "status": "incomplete_month_coverage"})
                continue
            cheap, weekly = _descriptor_from_scores(scores, policy, trial_key=key)
            spec = _flatten_trial(trial_specs.get(trial, {"name": trial}))
            record: dict[str, object] = {
                "trial_key": key,
                "source_trial_key": source_key,
                "label_era": str(args.era) if args.era else "full_period",
                "receipt_root": str(receipt_root.resolve()),
                "trial": trial,
                "arm": str(summary_row.get("arm", "unknown")),
                "target_family": str(summary_row.get("family", "unknown")),
                "feature_contract_mode": mode,
                "feature_contract_sha256": str(manifest.get("meta_feature_contract_sha256", "unknown")),
                "feature_count": feature_count,
                "receipt_manifest_sha256": _sha(manifest_path),
                **spec,
                **{f"objective_{name}": value for name, value in summary_row.to_dict().items() if name not in {"trial", "arm", "family"}},
                **cheap,
            }
            descriptors.append(record)
            if not weekly.empty:
                weekly_rows.append(weekly)
            registry.append({"trial_key": key, "source_trial_key": source_key, "label_era": str(args.era) if args.era else "full_period", "receipt_root": str(receipt_root.resolve()), "trial": trial, "status": "collected", "score_rows": int(len(scores)), "valid_policy_rows": int(cheap.get("valid_policy_rows", 0.0))})
    if not descriptors:
        raise RuntimeError("no completed Meta objective receipts cover every requested month")
    out.mkdir(parents=True)
    pd.DataFrame(descriptors).to_parquet(out / "trial_descriptors.parquet", index=False, compression="zstd")
    pd.concat(weekly_rows, ignore_index=True).to_parquet(out / "trial_weekly_metrics.parquet", index=False, compression="zstd") if weekly_rows else pd.DataFrame().to_parquet(out / "trial_weekly_metrics.parquet", index=False, compression="zstd")
    pd.DataFrame(registry).to_parquet(out / "trial_registry.parquet", index=False, compression="zstd")
    _once(out / "run_manifest.json", {
        "schema": SCHEMA,
        "mode": "collect",
        "scope": "offline strict-OOF descriptor collection; no live, MC1, admission, portfolio, or exchange mutation",
        "months": [f"{month:%Y-%m}" for month in months], "label_era": str(args.era) if args.era else "full_period",
        "artifacts": str(artifacts), "pattern": args.pattern,
        "policy": str(policy_path), "policy_sha256": _sha(policy_path),
        "additive_feature_rule": "new feature families are evaluated only as overlays to the current frozen Meta contract",
        "target_free_scores_before_outcome_join": True,
    })
    _once(out / "correctness_report.json", {
        "target_free_score_receipts_rejected_when_policy_columns_present": True,
        "canonical_policy_join_occurs_only_after_target_free_score_read": True,
        "held_outcomes_never_enter_score_features": True,
        "features_are_additive_or_frozen_parent_contracts": True,
        "no_live_or_exchange_mutation": True,
    })
    return out


def _downstream_priority_labels(frame: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    valid = frame.policy_path_valid.fillna(False).astype(bool) & frame.policy_net_bps.notna()
    data = frame.loc[valid].copy()
    if data.empty:
        return {}, pd.DataFrame()
    current_weekly: list[pd.DataFrame] = []
    result: dict[str, float] = {}
    for k in (1, 2):
        current = _topk_mask(data, "current_mc1_expected_bps", k)
        bcf = _topk_mask(data, "bcf_mc1_expected_bps", k)
        result[f"priority_top{k}_policy_net_bps"] = _mean_or_nan(data.loc[current, "policy_net_bps"])
        result[f"priority_bcf_top{k}_policy_net_bps"] = _mean_or_nan(data.loc[bcf, "policy_net_bps"])
        result[f"priority_delta_top{k}_bps"] = result[f"priority_top{k}_policy_net_bps"] - result[f"priority_bcf_top{k}_policy_net_bps"]
        result[f"priority_top{k}_substitution_bps"] = _mean_or_nan(data.loc[current & ~bcf, "policy_net_bps"]) - _mean_or_nan(data.loc[bcf & ~current, "policy_net_bps"])
        if k == 2:
            selected = data.loc[current, ["__decision_ts__", "policy_net_bps"]].copy()
            selected["week"] = selected.__decision_ts__.dt.normalize() - pd.to_timedelta(selected.__decision_ts__.dt.dayofweek, unit="D")
            current_weekly.append(selected.groupby("week", as_index=False).agg(ev_bps=("policy_net_bps", "mean"), trades=("policy_net_bps", "size")))
    weekly = current_weekly[0] if current_weekly else pd.DataFrame()
    result["priority_top2_total_utility_bps"] = float(data.loc[_topk_mask(data, "current_mc1_expected_bps", 2), "policy_net_bps"].sum())
    result.update(_aggregate_weekly(weekly, "priority_top2_week"))
    return result, weekly


def _downstream_gate_labels(frame: pd.DataFrame, threshold: float) -> tuple[dict[str, float], pd.DataFrame]:
    valid = frame.policy_path_valid.fillna(False).astype(bool) & frame.policy_net_bps.notna()
    admitted = frame.loc[valid & frame.current_mc1_expected_bps.ge(threshold) & frame.bcf_mc1_expected_bps.ge(threshold)].copy()
    if admitted.empty:
        return {"gate_admitted_rows": 0.0}, pd.DataFrame()
    admitted["week"] = admitted.__decision_ts__.dt.normalize() - pd.to_timedelta(admitted.__decision_ts__.dt.dayofweek, unit="D")
    weekly = admitted.groupby("week", as_index=False).agg(ev_bps=("policy_net_bps", "mean"), trades=("policy_net_bps", "size"))
    result: dict[str, float] = {
        "gate_admitted_rows": float(len(admitted)),
        "gate_admitted_ev_bps": _mean_or_nan(admitted.policy_net_bps),
        "gate_admitted_total_utility_bps": float(admitted.policy_net_bps.sum()),
        "gate_precision_above_50_fraction": float(np.mean(admitted.policy_net_bps > 50.0)),
        "gate_precision_above_100_fraction": float(np.mean(admitted.policy_net_bps > 100.0)),
        "gate_days_with_admission": float(admitted.__decision_ts__.dt.floor("D").nunique()),
    }
    result.update(_aggregate_weekly(weekly, "gate_week"))
    return result, weekly


def _collect_downstream_labels(args: argparse.Namespace) -> Path:
    descriptors_path = Path(args.descriptors).resolve()
    artifacts = Path(args.artifacts).resolve()
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    descriptors = pd.read_parquet(descriptors_path)
    source_to_trial = dict(zip(descriptors.get("source_trial_key", descriptors.trial_key).astype(str), descriptors.trial_key.astype(str)))
    labels: list[dict[str, object]] = []
    weekly_rows: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    for dual_path in sorted(artifacts.glob(args.pattern)):
        if dual_path.name != "dual_predictions.parquet":
            continue
        root = dual_path.parent
        manifest_path = root / "run_manifest.json"
        metrics_path = root / "portfolio_metrics.parquet"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
            metas = manifest.get("metas", [])
            # Multi-head blends are valuable final tests but do not identify a
            # one-trial surrogate label.  They remain portfolio-only evidence.
            if len(metas) != 1:
                audit.append({"downstream_root": str(root), "status": "skip_multi_meta", "meta_count": len(metas)})
                continue
            meta = metas[0]
            source_key = f"{Path(str(meta['root'])).resolve()}::{str(meta['arm'])}"
            key = source_to_trial.get(source_key)
            if key is None:
                audit.append({"downstream_root": str(root), "source_trial_key": source_key, "status": "not_in_descriptor_bank"})
                continue
            data = pd.read_parquet(dual_path)
            data["__decision_ts__"] = pd.to_datetime(data["__decision_ts__"], utc=True, errors="raise")
            if args.start:
                data = data.loc[data.__decision_ts__.ge(_utc_timestamp(args.start))].copy()
            if args.end:
                data = data.loc[data.__decision_ts__.lt(_utc_timestamp(args.end))].copy()
            required = {"current_mc1_expected_bps", "bcf_mc1_expected_bps", "policy_net_bps", "policy_path_valid"}
            if required.difference(data.columns):
                raise AssertionError(f"{dual_path}: missing dual-MC1 label columns")
            priority, priority_week = _downstream_priority_labels(data)
            gate, gate_week = _downstream_gate_labels(data, float(manifest.get("threshold_bps", args.threshold_bps)))
            record: dict[str, object] = {
                "trial_key": key,
                "source_trial_key": source_key,
                "label_era": str(args.era) if args.era else "full_period",
                "downstream_root": str(root.resolve()),
                "threshold_bps": float(manifest.get("threshold_bps", args.threshold_bps)),
                "months": ",".join(str(item) for item in manifest.get("months", [])),
                "dual_rows": int(len(data)),
                **priority, **gate,
            }
            if metrics_path.exists():
                metrics = pd.read_parquet(metrics_path)
                if len(metrics):
                    # Keep portfolio as later confirmation, never as the
                    # fitted cheap proxy target.
                    latest = metrics.iloc[-1].to_dict()
                    for field in ("accepted_rows", "net_ev_bps_per_realised_trade", "net_sum_bps_realised", "worst_month_bps", "worst_week_bps", "max_drawdown"):
                        record[f"portfolio_{field}"] = latest.get(field, np.nan)
            labels.append(record)
            for kind, weekly in (("priority", priority_week), ("gate", gate_week)):
                if not weekly.empty:
                    weekly = weekly.copy(); weekly["trial_key"] = key; weekly["label_kind"] = kind
                    weekly_rows.append(weekly)
            audit.append({"downstream_root": str(root.resolve()), "trial_key": key, "source_trial_key": source_key, "label_era": str(args.era) if args.era else "full_period", "status": "collected"})
        except Exception as exc:
            audit.append({"downstream_root": str(root.resolve()), "status": "unreadable", "detail": str(exc)})
    if not labels:
        raise RuntimeError("no single-Meta completed strict-MC1 receipts match descriptor bank")
    out.mkdir(parents=True)
    pd.DataFrame(labels).drop_duplicates("trial_key", keep="last").to_parquet(out / "downstream_labels.parquet", index=False, compression="zstd")
    pd.concat(weekly_rows, ignore_index=True).to_parquet(out / "downstream_weekly_labels.parquet", index=False, compression="zstd") if weekly_rows else pd.DataFrame().to_parquet(out / "downstream_weekly_labels.parquet", index=False, compression="zstd")
    pd.DataFrame(audit).to_parquet(out / "downstream_label_audit.parquet", index=False, compression="zstd")
    _once(out / "run_manifest.json", {
        "schema": SCHEMA, "mode": "collect-downstream",
        "scope": "offline strict-MC1 / portfolio label collection only; no live mutation",
        "descriptor_source": str(descriptors_path), "descriptor_sha256": _sha(descriptors_path),
        "artifacts": str(artifacts), "threshold_bps_default": float(args.threshold_bps), "label_era": str(args.era) if args.era else "full_period", "start": args.start, "end": args.end,
        "priority_label": "matched timestamp-local MC1 top1/top2 and substitution versus frozen BCF coordinate",
        "gate_label": "real dual-MC1 threshold gate using both BCF and Current mapped EV",
        "portfolio_usage": "confirmation only; excluded from Proxy fit target",
    })
    _once(out / "correctness_report.json", {
        "only_completed_dual_mc1_receipts_used": True,
        "single_meta_only_for_trial_level_labels": True,
        "dual_gate_requires_bcf_and_current_mc1": True,
        "portfolio_pnl_is_not_a_cheap_proxy_training_target": True,
        "no_live_or_exchange_mutation": True,
    })
    return out


def _robust_z(values: pd.Series) -> pd.Series:
    array = values.astype(float)
    median = float(np.nanmedian(array))
    mad = float(np.nanmedian(np.abs(array - median)))
    scale = max(1e-6, 1.4826 * mad)
    return (array - median) / scale


def _target(frame: pd.DataFrame, kind: str) -> tuple[pd.Series, pd.Series]:
    if kind == "priority":
        columns = {
            "priority_top1_policy_net_bps": .20,
            "priority_top2_policy_net_bps": .40,
            "priority_top2_total_utility_bps": .25,
            "priority_top2_week_q10": .15,
        }
        se = frame.get("priority_top2_week_se", pd.Series(np.nan, index=frame.index))
    elif kind == "gate":
        columns = {
            "gate_admitted_ev_bps": .35,
            "gate_admitted_total_utility_bps": .25,
            "gate_precision_above_50_fraction": .15,
            "gate_precision_above_100_fraction": .15,
            "gate_week_q10": .10,
        }
        se = frame.get("gate_week_se", pd.Series(np.nan, index=frame.index))
    else:
        raise ValueError(kind)
    valid_columns = {name: weight for name, weight in columns.items() if name in frame and frame[name].notna().sum() >= 4}
    if len(valid_columns) < 3:
        raise ValueError(f"insufficient {kind} downstream metrics: found {list(valid_columns)}")
    weights = np.asarray(list(valid_columns.values()), dtype=float)
    weights /= weights.sum()
    target = sum(weight * _robust_z(frame[name]) for weight, name in zip(weights, valid_columns))
    # Reliability shrinkage lowers the authority of a noisy weekly label.  It
    # never makes a portfolio metric part of the target.
    finite_se = pd.to_numeric(se, errors="coerce")
    prior_var = float(np.nanvar(target)) if np.isfinite(target).any() else 1.0
    shrink = prior_var / (prior_var + finite_se.fillna(np.nanmedian(finite_se) if finite_se.notna().any() else 1.0) ** 2)
    shrink = pd.Series(np.clip(shrink, .20, 1.0), index=frame.index)
    return target * shrink, 1.0 / np.clip(finite_se.fillna(finite_se.median() if finite_se.notna().any() else 1.0) ** 2, .05, 25.0)


def _numeric_features(frame: pd.DataFrame) -> list[str]:
    blocked = {
        "trial_key", "source_trial_key", "label_era", "receipt_root", "trial", "arm", "target_family", "feature_contract_mode", "feature_contract_sha256", "receipt_manifest_sha256", "loss_family", "gain_signature", "sample_weight_family", "downstream_root", "months",
    }
    columns: list[str] = []
    for name in frame.columns:
        if name in blocked or name.startswith("portfolio_") or name.startswith("priority_") or name.startswith("gate_"):
            continue
        if pd.api.types.is_numeric_dtype(frame[name]) and frame[name].notna().sum() >= 4:
            columns.append(name)
    return columns


@dataclass(frozen=True)
class FitResult:
    name: str
    model: object
    prediction: np.ndarray


@dataclass
class PairwiseProbe:
    """Pickle-safe pairwise ranker projected against its train-set median."""

    imputer: SimpleImputer
    ranker: object
    anchor: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        transformed = self.imputer.transform(x)
        return np.asarray(self.ranker.predict_proba(transformed - self.anchor)[:, 1], dtype=float)


def _pairwise_model(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[object, np.ndarray]:
    pairs: list[np.ndarray] = []
    labels: list[int] = []
    for left in range(len(y)):
        for right in range(left + 1, len(y)):
            if abs(y[left] - y[right]) < .05:
                continue
            pairs.append(x[left] - x[right]); labels.append(int(y[left] > y[right]))
    if len(pairs) < 8 or len(set(labels)) < 2:
        model = Ridge(alpha=8.0).fit(x, y)
        return model, model.predict(x)
    model = Pipeline([("scale", RobustScaler()), ("clf", LogisticRegression(C=.25, max_iter=2000, random_state=seed))])
    model.fit(np.asarray(pairs), labels)
    # Pairwise probability against a fixed median trial gives an ordering-only
    # score while preserving out-of-sample use in cross-validation.
    anchor = np.nanmedian(x, axis=0, keepdims=True)
    return model, model.predict_proba(x - anchor)[:, 1]


def _models(x: np.ndarray, y: np.ndarray, seed: int, sample_weight: np.ndarray | None = None) -> list[FitResult]:
    output: list[FitResult] = []
    ridge = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", RobustScaler()), ("model", Ridge(alpha=12.0))])
    ridge.fit(x, y, **({"model__sample_weight": sample_weight} if sample_weight is not None else {}))
    output.append(FitResult("P0_ridge", ridge, ridge.predict(x)))
    elastic = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", RobustScaler()), ("model", ElasticNet(alpha=.08, l1_ratio=.35, max_iter=8000, random_state=seed))])
    elastic.fit(x, y, **({"model__sample_weight": sample_weight} if sample_weight is not None else {}))
    output.append(FitResult("P1_elastic_net", elastic, elastic.predict(x)))
    tree = Pipeline([("impute", SimpleImputer(strategy="median")), ("model", HistGradientBoostingRegressor(max_depth=2, max_leaf_nodes=7, min_samples_leaf=max(4, len(y) // 10), l2_regularization=5.0, learning_rate=.05, max_iter=120, random_state=seed))])
    tree.fit(x, y, **({"model__sample_weight": sample_weight} if sample_weight is not None else {}))
    output.append(FitResult("P2_depth2_gbdt", tree, tree.predict(x)))
    imp = SimpleImputer(strategy="median").fit(x)
    transformed = imp.transform(x)
    pair, prediction = _pairwise_model(transformed, y, seed)
    output.append(FitResult("P3_pairwise", PairwiseProbe(imp, pair, np.nanmedian(transformed, axis=0, keepdims=True)), prediction))
    return output


def _predict(model: object, x: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(x), dtype=float)


def _leave_one_out_predictions(
    x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, seed: int,
) -> dict[str, np.ndarray]:
    """Strict trial-level LOO predictions used for proxy selection, never fit scores."""
    names = ("P0_ridge", "P1_elastic_net", "P2_depth2_gbdt", "P3_pairwise")
    result = {name: np.full(len(y), np.nan, dtype=float) for name in names}
    for held in range(len(y)):
        train = np.arange(len(y)) != held
        fitted = _models(x[train], y[train], seed + held, sample_weight[train])
        for item in fitted:
            result[item.name][held] = _predict(item.model, x[held: held + 1])[0]
    return result


def _score_rank(y: np.ndarray, pred: np.ndarray, k: int) -> dict[str, float]:
    order_true = np.argsort(-y, kind="stable")
    order_pred = np.argsort(-pred, kind="stable")
    k = min(k, len(y))
    top_true = set(order_true[:k]); top_pred = set(order_pred[:k])
    return {
        "spearman": float(spearmanr(y, pred).statistic) if len(y) > 2 else np.nan,
        f"top{k}_precision": float(len(top_true.intersection(top_pred)) / max(k, 1)),
        f"winner_in_top{k}": float(order_true[0] in top_pred),
        f"regret_top{k}": float(y[order_true[0]] - np.max(y[list(top_pred)])),
    }


def _fit_proxies(args: argparse.Namespace) -> Path:
    descriptors_path = Path(args.descriptors).resolve(); labels_path = Path(args.labels).resolve(); out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    descriptors = pd.read_parquet(descriptors_path)
    labels = pd.read_parquet(labels_path)
    # Per-era receipts deliberately carry their lineage on both sides of the
    # join.  Validate that it agrees, then retain one canonical copy rather
    # than accepting pandas' _x/_y suffixes: grouped temporal validation
    # below must operate on an unambiguous label_era field.
    duplicate_lineage = [
        name for name in ("source_trial_key", "label_era")
        if name in descriptors.columns and name in labels.columns
    ]
    if duplicate_lineage:
        left = descriptors.set_index("trial_key")[duplicate_lineage]
        right = labels.set_index("trial_key")[duplicate_lineage]
        common = left.index.intersection(right.index)
        if not left.loc[common].astype(str).eq(right.loc[common].astype(str)).all().all():
            raise AssertionError("descriptor and downstream label lineage disagree")
        labels = labels.drop(columns=duplicate_lineage)
    frame = descriptors.merge(labels, on="trial_key", how="inner", validate="one_to_one")
    # Repeated source receipts are evidence of reproducibility, not
    # independent training observations.  Collapse exact downstream outcome
    # clones before fitting, preserving the first immutable receipt.
    clone_columns = [
        name for name in (
            "priority_top1_policy_net_bps", "priority_top2_policy_net_bps", "priority_top2_total_utility_bps",
            "gate_admitted_rows", "gate_admitted_ev_bps", "gate_admitted_total_utility_bps",
            "gate_precision_above_50_fraction", "gate_precision_above_100_fraction",
        ) if name in frame
    ]
    frame["downstream_clone_key"] = frame.loc[:, clone_columns].round(8).astype(str).agg("|".join, axis=1) + "::" + frame.get("label_era", pd.Series("full_period", index=frame.index)).astype(str)
    before_deduplication = int(len(frame))
    frame = frame.drop_duplicates("downstream_clone_key", keep="first").reset_index(drop=True)
    if len(frame) < 8:
        raise RuntimeError(f"need at least 8 non-duplicate completed downstream trials; have {len(frame)}")
    features = _numeric_features(frame)
    x = frame.loc[:, features].to_numpy(float)
    out.mkdir(parents=True)
    model_bundle: dict[str, object] = {"schema": SCHEMA, "features": features, "models": {}, "selection": {}}
    predictions = frame.loc[:, ["trial_key", "target_family", "loss_family", "feature_contract_sha256"]].copy()
    validation: list[dict[str, object]] = []
    for kind in ("priority", "gate"):
        y, sample_weight = _target(frame, kind)
        y_values = y.to_numpy(float)
        weight_values = sample_weight.to_numpy(float)
        fitted = _models(x, y_values, int(args.seed), weight_values)
        aggregate = np.column_stack([item.prediction for item in fitted])
        for item in fitted:
            record = {"proxy": kind, "model": item.name, **_score_rank(y_values, item.prediction, min(3, len(y_values)))}
            validation.append(record)
            model_bundle["models"][f"{kind}::{item.name}"] = item.model
        # LOO is the only criterion used to choose P0--P3.  Grouped tests are
        # separately reported as required falsification, never back-filled by
        # the much easier in-sample score.
        loo = _leave_one_out_predictions(x, y_values, weight_values, int(args.seed))
        for name, prediction in loo.items():
            validation.append({"proxy": kind, "model": name, "cv_group": "leave_one_trial_out", "held_group": "all", **_score_rank(y_values, prediction, min(3, len(y_values)))})
        # Leave-one-contract/family/loss validation.  It intentionally uses
        # only cheap descriptors and real downstream labels.
        for group_name in ("target_family", "loss_family", "feature_contract_sha256", "label_era"):
            values = frame[group_name].astype(str)
            for group in sorted(values.unique()):
                test = values.eq(group).to_numpy()
                train = ~test
                if int(test.sum()) < 1 or int(train.sum()) < 6:
                    continue
                cv_models = _models(x[train], y_values[train], int(args.seed), weight_values[train])
                for item in cv_models:
                    pred = _predict(item.model, x[test])
                    validation.append({"proxy": kind, "model": item.name, "cv_group": group_name, "held_group": group, **_score_rank(y_values[test], np.asarray(pred, float), min(3, int(test.sum())))})
        # Prefer LOO rank reliability, then winner containment and low regret.
        candidates = [item for item in validation if item.get("proxy") == kind and item.get("cv_group") == "leave_one_trial_out"]
        ranked = sorted(candidates, key=lambda row: (-np.nan_to_num(row.get("spearman"), nan=-9), -row.get("winner_in_top3", 0.0), row.get("regret_top3", np.inf)))
        chosen = ranked[0]["model"]
        model_bundle["selection"][kind] = chosen
        predictions[f"{kind}_target"] = y_values
        predictions[f"{kind}_prediction_mean"] = aggregate.mean(axis=1)
        predictions[f"{kind}_prediction_std"] = aggregate.std(axis=1)
        predictions[f"{kind}_selected_prediction"] = next(item.prediction for item in fitted if item.name == chosen)
    group_diversity = {
        name: int(frame[name].astype(str).nunique())
        for name in ("target_family", "loss_family", "feature_contract_sha256", "label_era")
    }
    # Receipt diversity is necessary, but not sufficient.  A Meta-HPO proxy
    # must also carry at least a weakly positive relationship into every
    # explicitly held chronological era for *both* responsibilities.  This
    # prevents a high LOO score driven by similar trial receipts from becoming
    # a gate or priority acquisition function.
    cross_era_validation: dict[str, dict[str, object]] = {}
    for kind in ("priority", "gate"):
        chosen_rows = [
            item for item in validation
            if item.get("proxy") == kind
            and item.get("model") == model_bundle["selection"][kind]
            and item.get("cv_group") == "label_era"
        ]
        era_spearman = [float(item["spearman"]) for item in chosen_rows if np.isfinite(item.get("spearman", np.nan))]
        cross_era_validation[kind] = {
            "selected_model": model_bundle["selection"][kind],
            "era_count": int(len(era_spearman)),
            "mean_spearman": float(np.mean(era_spearman)) if era_spearman else float("nan"),
            "worst_spearman": float(np.min(era_spearman)) if era_spearman else float("nan"),
            "passes": bool(
                len(era_spearman) >= 2
                and np.mean(era_spearman) > 0.0
                and np.min(era_spearman) >= -0.05
            ),
        }
    qualified = (
        all(value >= 2 for value in group_diversity.values())
        and len(frame) >= 20
        and all(item["passes"] for item in cross_era_validation.values())
    )
    # A recommendation must pass GateProxy tolerance, be ranking-promising,
    # and expose its uncertainty rather than hiding it in a point estimate.
    predictions["acquisition"] = predictions.priority_selected_prediction + float(args.kappa) * predictions.priority_prediction_std
    predictions.to_parquet(out / "proxy_in_sample_predictions.parquet", index=False, compression="zstd")
    pd.DataFrame(validation).to_parquet(out / "proxy_validation.parquet", index=False, compression="zstd")
    joblib.dump(model_bundle, out / "meta_hpo_proxy_bundle.joblib")
    _once(out / "run_manifest.json", {
        "schema": SCHEMA, "mode": "fit",
        "scope": "offline surrogate only; frozen Base, MC1, dual gate, portfolio contracts untouched",
        "descriptor_source": str(descriptors_path), "labels_source": str(labels_path),
        "descriptor_sha256": _sha(descriptors_path), "labels_sha256": _sha(labels_path),
        "feature_count": len(features), "features": features,
        "matched_receipts_before_clone_deduplication": before_deduplication,
        "non_duplicate_training_trials": int(len(frame)),
        "group_diversity": group_diversity,
        "qualified_for_proxy_guided_hpo": qualified,
        "qualification_rule": "at least 20 non-duplicate labels; at least two target, loss, contract, and explicit label-era groups; and selected Priority/Gate proxies each have positive mean and no worse than -0.05 Spearman in every held label era",
        "cross_era_validation": cross_era_validation,
        "objectives": {"priority": "matched MC1 ranking/utility/stability", "gate": "real dual-MC1 admission quality/volume/stability"},
        "portfolio_pnl_used_for_selection": False,
        "uncertainty": "cross-model prediction standard deviation", "kappa": float(args.kappa),
    })
    return out


def _recommend(args: argparse.Namespace) -> Path:
    descriptors_path = Path(args.descriptors).resolve(); labels_path = Path(args.labels).resolve(); bundle_path = Path(args.bundle).resolve(); out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    descriptors = pd.read_parquet(descriptors_path); labels = pd.read_parquet(labels_path); bundle = joblib.load(bundle_path)
    manifest_path = bundle_path.parent / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not bool(manifest.get("qualified_for_proxy_guided_hpo", False)):
        raise RuntimeError("proxy bank is not qualified for trial selection; expand representative full-MC1 labels first")
    frame = descriptors.loc[~descriptors.trial_key.isin(set(labels.trial_key.astype(str)))].copy()
    if frame.empty:
        raise RuntimeError("no unlabelled strict-OOF trials available for proxy-guided recommendation")
    features = list(bundle["features"])
    for field in features:
        if field not in frame:
            frame[field] = np.nan
    x = frame.loc[:, features].to_numpy(float)
    for kind in ("priority", "gate"):
        values: list[np.ndarray] = []
        for name in ("P0_ridge", "P1_elastic_net", "P2_depth2_gbdt", "P3_pairwise"):
            model = bundle["models"][f"{kind}::{name}"]
            values.append(np.asarray(model.predict(x), dtype=float))
        stack = np.column_stack(values)
        frame[f"{kind}_proxy_mean"] = stack.mean(axis=1)
        frame[f"{kind}_proxy_std"] = stack.std(axis=1)
    floor = float(args.gate_tolerance)
    frame["passes_gate_proxy"] = frame.gate_proxy_mean.ge(floor)
    frame["acquisition"] = frame.priority_proxy_mean + float(args.kappa) * frame.priority_proxy_std
    frame = frame.sort_values(["passes_gate_proxy", "acquisition"], ascending=[False, False], kind="stable")
    out.mkdir(parents=True)
    frame.to_parquet(out / "proxy_guided_trial_recommendations.parquet", index=False, compression="zstd")
    _once(out / "run_manifest.json", {
        "schema": SCHEMA, "mode": "recommend", "scope": "offline HPO acquisition only; no live mutation",
        "descriptors": str(descriptors_path), "labels": str(labels_path), "proxy_bundle": str(bundle_path),
        "selection": "PriorityProxy + kappa*uncertainty subject to GateProxy floor", "gate_tolerance": floor, "kappa": float(args.kappa),
    })
    return out


def _merge_receipts(args: argparse.Namespace) -> Path:
    """Merge immutable per-era receipt tables after exact identity checks."""
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    descriptor_paths = [Path(value).resolve() for value in args.descriptor]
    label_paths = [Path(value).resolve() for value in args.label]
    descriptors = pd.concat([pd.read_parquet(path) for path in descriptor_paths], ignore_index=True, sort=False)
    labels = pd.concat([pd.read_parquet(path) for path in label_paths], ignore_index=True, sort=False)
    if descriptors.trial_key.duplicated().any() or labels.trial_key.duplicated().any():
        raise AssertionError("per-era merge has duplicate trial_key; label_era must be part of identity")
    if not set(labels.trial_key).issubset(set(descriptors.trial_key)):
        raise AssertionError("labels include a trial absent from descriptor receipts")
    out.mkdir(parents=True)
    descriptors.to_parquet(out / "trial_descriptors.parquet", index=False, compression="zstd")
    labels.to_parquet(out / "downstream_labels.parquet", index=False, compression="zstd")
    _once(out / "run_manifest.json", {
        "schema": SCHEMA, "mode": "merge-receipts", "scope": "offline immutable per-era receipt merge; no live mutation",
        "descriptor_sources": [str(path) for path in descriptor_paths], "label_sources": [str(path) for path in label_paths],
        "descriptor_source_sha256": [_sha(path) for path in descriptor_paths], "label_source_sha256": [_sha(path) for path in label_paths],
        "label_eras": sorted(descriptors.label_era.astype(str).unique().tolist()) if "label_era" in descriptors else ["full_period"],
    })
    return out


def _bank(args: argparse.Namespace) -> Path:
    """Emit a diverse *additive* cheap trial bank; execution stays separate."""
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    number = int(args.count)
    if number < 24:
        raise ValueError("--count must be at least 24 for target/loss diversity")
    targets = ("under_bps50", "under_bps100", "over_bps50", "over_bps100", "magnitude", "state")
    target_arm = {
        "under_bps50": "under_bps50__timestamp",
        "under_bps100": "under_bps100__timestamp",
        "over_bps50": "over_bps50__timestamp",
        "over_bps100": "over_bps100__timestamp",
        "magnitude": "magnitude_bps__base_band_block28",
        "state": "state_bps__base_band_block28",
    }
    objectives = ("rank_xendcg", "lambdarank")
    gains = ([0, 1, 2, 4, 7, 11, 16, 24], [0, 1, 2, 5, 10, 18, 28, 40], [0, .25, 1, 3, 7, 14, 24, 36])
    contracts = ("current_frozen", "current_plus_shap_stable", "current_plus_context_conditional")
    trials: list[dict[str, object]] = []
    for index in range(number):
        target = targets[index % len(targets)]
        objective = objectives[(index // len(targets)) % len(objectives)]
        trial = {
            "name": f"proxybank_{index:03d}_{target}_{objective}",
            "target": target,
            "arm_name": target_arm[target],
            "feature_mode": "additive_overlay",
            "parent_contract": str(args.parent_contract),
            "additive_feature_family": contracts[(index // (len(targets) * len(objectives))) % len(contracts)],
            "gain": gains[index % len(gains)],
            "truncation": (6, 8, 12)[index % 3], "sigmoid": (0.75, 1.0, 1.25)[(index // 3) % 3],
            "sample_weight": (
                None,
                {"equal_timestamp": True, "components": []},
                {"equal_timestamp": True, "components": [{"name": "positive_recall", "strength": 0.25, "power": 1.0}]},
            )[(index // 9) % 3],
            "model": {
                "objective": objective, "learning_rate": (.025, .045, .075)[index % 3],
                "max_depth": (2, 3, 4, 5)[(index // 3) % 4], "num_leaves": (7, 15, 31)[(index // 5) % 3],
                "min_child_samples": (200, 350, 500)[(index // 7) % 3], "min_split_gain": (0.0, .001, .005)[(index // 8) % 3], "feature_fraction": (.70, .80, .90)[(index // 11) % 3],
                "bagging_fraction": (.70, .82, .90)[(index // 13) % 3], "lambda_l2": (2.0, 8.0, 20.0)[(index // 17) % 3],
                "lambda_l1": (0.0, .02, .10)[(index // 19) % 3], "n_estimators": 300,
            },
        }
        trials.append(trial)
    out.mkdir(parents=True)
    _once(out / "trial_bank.json", {"schema": SCHEMA, "purpose": "cheap strict-OOF trial bank; all feature families are additive", "trials": trials})
    # The existing strict-OOF screen intentionally consumes a bare list.  Keep
    # the richer human/audit receipt above, but also publish the exact
    # executable payload so a bank cannot silently become documentation-only.
    _once(out / "trials.json", trials)
    for family in contracts:
        family_trials = [trial for trial in trials if trial["additive_feature_family"] == family]
        _once(out / f"trials_{family}.json", family_trials)
        for target in targets:
            routed = [trial for trial in family_trials if trial["target"] == target]
            _once(out / f"trials_{family}_{target}.json", routed)
    _once(out / "run_manifest.json", {"schema": SCHEMA, "mode": "bank", "count": number, "parent_contract": str(args.parent_contract), "feature_rule": "each candidate augments, never replaces, current frozen stack"})
    return out


def _slug(value: str) -> str:
    keep = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    compact = "".join(char if char in keep else "_" for char in value)
    digest = hashlib.sha256(value.encode()).hexdigest()[:10]
    return f"{compact[:72].strip('_')}__{digest}"


def _plan_labels(args: argparse.Namespace) -> Path:
    """Choose a deliberately diverse, additive/current-contract MC1 label bank.

    This planner does not use downstream portfolio outcomes to choose trials.
    It deliberately spans cheap-screen quality tertiles so the surrogate sees
    good, mediocre, and poor Meta configurations rather than only finalists.
    """
    descriptors_path = Path(args.descriptors).resolve(); labels_path = Path(args.labels).resolve() if args.labels else None
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    frame = pd.read_parquet(descriptors_path).copy()
    already = set(pd.read_parquet(labels_path).trial_key.astype(str)) if labels_path else set()
    allowed = {"frozen_parent", "implicit_frozen_current", "additive_overlay"}
    frame = frame.loc[~frame.trial_key.astype(str).isin(already) & frame.feature_contract_mode.isin(allowed)].copy()
    frame = frame.loc[frame.objective_sstable_meta.notna()].copy()
    if len(frame) < int(args.count):
        raise RuntimeError(f"only {len(frame)} eligible current/additive trials for requested {args.count}")
    frame["quality_tertile"] = pd.qcut(frame.objective_sstable_meta.rank(method="first"), 3, labels=("low", "middle", "high")).astype(str)
    frame["trial_slug"] = frame.trial_key.astype(str).map(_slug)
    group_fields = ["target_family", "loss_family", "feature_contract_mode", "quality_tertile"]
    selected: list[int] = []
    # One representative per target/loss/contract/quality cell where support
    # exists.  This makes the initial expensive label set an experiment bank,
    # not a cheap-screen winner list.
    for _, group in frame.groupby(group_fields, dropna=False, sort=True):
        ranked = group.sort_values("objective_sstable_meta", ascending=False, kind="stable")
        selected.append(int(ranked.index[len(ranked) // 2]))
    selected = list(dict.fromkeys(selected))
    numeric = [name for name in ("objective_sstable_meta", "residual_spearman_ic", "conditional_rank_ic_given_base", "model_max_depth", "truncation", "gain_max", "gain_tail_gap", "feature_count") if name in frame]
    scaled = frame.loc[:, numeric].copy()
    for name in numeric:
        median = scaled[name].median(); mad = (scaled[name] - median).abs().median()
        scaled[name] = (scaled[name] - median) / max(1e-6, 1.4826 * mad)
    categorical = ["target_family", "loss_family", "feature_contract_mode", "quality_tertile", "gain_signature", "sample_weight_family"]
    # Fill remaining slots by max-min diversity across model, target, and
    # feature-contract descriptors.  No downstream MC1 or portfolio label
    # enters this acquisition step.
    while len(selected) < int(args.count):
        candidates = [idx for idx in frame.index if int(idx) not in selected]
        if not selected:
            selected.append(int(candidates[0])); continue
        best_idx, best_distance = None, -np.inf
        for index in candidates:
            distances: list[float] = []
            for chosen in selected:
                numerical = float(np.nanmean(np.abs(scaled.loc[index, numeric].to_numpy(float) - scaled.loc[chosen, numeric].to_numpy(float)))) if numeric else 0.0
                categoric = float(sum(frame.at[index, name] != frame.at[chosen, name] for name in categorical)) / max(1, len(categorical))
                distances.append(numerical + categoric)
            distance = min(distances)
            if distance > best_distance:
                best_idx, best_distance = int(index), distance
        assert best_idx is not None
        selected.append(best_idx)
    plan = frame.loc[selected].copy().sort_values(group_fields + ["trial_key"], kind="stable")
    plan["label_plan_era"] = str(args.era)
    plan["label_plan_reason"] = "representative target/loss/contract/quality coverage plus max-min descriptor diversity"
    plan["base_root"] = str(Path(args.base_root).resolve())
    plan["policy"] = str(Path(args.policy).resolve())
    plan["months"] = str(args.months)
    plan["threshold_bps"] = float(args.threshold_bps)
    plan["out_root"] = str(Path(args.out_prefix).resolve())
    out.mkdir(parents=True)
    plan.to_parquet(out / "representative_mc1_label_plan.parquet", index=False, compression="zstd")
    _once(out / "run_manifest.json", {
        "schema": SCHEMA, "mode": "plan-labels", "scope": "offline representative strict-MC1 label plan; no live mutation",
        "descriptors": str(descriptors_path), "labels_already_completed": str(labels_path) if labels_path else None,
        "count": int(args.count), "era": str(args.era), "months": str(args.months),
        "selection": "target/loss/current-or-additive-contract/quality-tertile coverage then max-min descriptor diversity",
        "outcome_or_portfolio_used_for_acquisition": False,
        "feature_rule": "only frozen-current or additive-overlay feature contracts may enter the label plan",
    })
    return out


def _execute_label_task(payload: tuple[str, str, str, str, str, str, float]) -> dict[str, str]:
    """Top-level target for process-safe offline MC1 label execution."""
    import run_strict_r3_p8u_meta_mc1_combination_v1 as mc1

    root_raw, trial, target_raw, base_raw, policy_raw, months_raw, threshold = payload
    result = mc1.run(
        base_root=Path(base_raw), metas=((Path(root_raw), trial, 1.0),), policy_path=Path(policy_raw),
        months=mc1._months(months_raw), out=Path(target_raw), threshold_bps=float(threshold),
    )
    return {"receipt_root": root_raw, "trial": trial, "out": str(result), "status": "completed"}


def _run_label_plan(args: argparse.Namespace) -> Path:
    """Run planned single-Meta downstream labels, serially and immutably.

    ``max-workers`` exists only to use independent CPU capacity during offline
    research.  Every output root is unique and no process mutates shared live
    or model state.  The default of one is deliberately conservative.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import run_strict_r3_p8u_meta_mc1_combination_v1 as mc1

    plan_path = Path(args.plan).resolve(); out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    plan = pd.read_parquet(plan_path)
    if plan.empty:
        raise RuntimeError("empty representative label plan")
    if args.limit is not None:
        if int(args.limit) < 1:
            raise ValueError("--limit must be positive")
        plan = plan.head(int(args.limit)).copy()
    roots = [Path(value).resolve() for value in plan.receipt_root]
    if not all((root / "target_free_scores").exists() for root in roots):
        raise AssertionError("label plan includes a receipt without target-free scores")
    output_prefix = Path(str(plan.out_root.iloc[0])).resolve()
    if output_prefix != Path(args.out_prefix).resolve():
        raise AssertionError("--out-prefix does not match immutable label-plan receipt")
    months = mc1._months(str(plan.months.iloc[0]))
    threshold = float(plan.threshold_bps.iloc[0])
    base_root = Path(str(plan.base_root.iloc[0])).resolve(); policy = Path(str(plan.policy.iloc[0])).resolve()
    if not base_root.exists() or not policy.exists():
        raise FileNotFoundError("label-plan base root or policy source missing")
    tasks: list[tuple[str, str, str, str, str, str, float]] = []
    results: list[dict[str, str]] = []
    for _, row in plan.iterrows():
        root = Path(str(row.receipt_root)).resolve(); trial = str(row.trial)
        target = output_prefix / f"strict_r3_p8u_meta_hpo_label_{str(row.trial_slug)}"
        if target.exists():
            if not bool(args.skip_existing):
                raise FileExistsError(f"planned immutable label root already exists: {target}")
            if not (target / "correctness_report.json").exists() or not (target / "dual_predictions.parquet").exists():
                raise AssertionError(f"existing label root is incomplete and cannot be skipped: {target}")
            results.append({"receipt_root": str(root), "trial": trial, "out": str(target), "status": "already_completed"})
            continue
        tasks.append((str(root), trial, str(target), str(base_root), str(policy), str(plan.months.iloc[0]), threshold))

    workers = int(args.max_workers)
    if workers < 1 or workers > 4:
        raise ValueError("--max-workers must be between 1 and 4")
    if workers == 1:
        for task in tasks:
            results.append(_execute_label_task(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_execute_label_task, task): task for task in tasks}
            for future in as_completed(futures):
                results.append(future.result())
    out.mkdir(parents=True)
    pd.DataFrame(results).to_parquet(out / "completed_label_runs.parquet", index=False, compression="zstd")
    _once(out / "run_manifest.json", {
        "schema": SCHEMA, "mode": "run-label-plan", "scope": "offline strict-prequential MC1 labels only; no live mutation",
        "plan": str(plan_path), "plan_sha256": _sha(plan_path), "runs": len(results), "new_runs": int(sum(row["status"] == "completed" for row in results)), "max_workers": workers,
        "base_root": str(base_root), "policy": str(policy), "months": [f"{month:%Y-%m}" for month in months], "threshold_bps": threshold,
        "contracts_frozen": ["Base", "MC1", "dual gate", "portfolio"],
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True)
    collect = command.add_parser("collect")
    collect.add_argument("--artifacts", type=Path, default=ROOT / "data_perp" / "artifacts")
    collect.add_argument("--policy", type=Path, required=True)
    collect.add_argument("--months", required=True)
    collect.add_argument("--out", type=Path, required=True)
    collect.add_argument("--pattern", default="strict_r3_p8u_*/objective_summary.parquet")
    collect.add_argument("--era", help="explicit temporal block identity; becomes part of trial identity")
    downstream = command.add_parser("collect-downstream")
    downstream.add_argument("--descriptors", type=Path, required=True)
    downstream.add_argument("--artifacts", type=Path, default=ROOT / "data_perp" / "artifacts")
    downstream.add_argument("--out", type=Path, required=True)
    downstream.add_argument("--threshold-bps", type=float, default=50.0)
    downstream.add_argument("--pattern", default="strict_r3_p8u_*/dual_predictions.parquet")
    downstream.add_argument("--era", help="explicit temporal block identity matching the descriptor bank")
    downstream.add_argument("--start", help="inclusive UTC timestamp for a causal output block")
    downstream.add_argument("--end", help="exclusive UTC timestamp for a causal output block")
    fit = command.add_parser("fit")
    fit.add_argument("--descriptors", type=Path, required=True); fit.add_argument("--labels", type=Path, required=True); fit.add_argument("--out", type=Path, required=True)
    fit.add_argument("--seed", type=int, default=1729); fit.add_argument("--kappa", type=float, default=.50)
    recommend = command.add_parser("recommend")
    recommend.add_argument("--descriptors", type=Path, required=True); recommend.add_argument("--labels", type=Path, required=True); recommend.add_argument("--bundle", type=Path, required=True); recommend.add_argument("--out", type=Path, required=True)
    recommend.add_argument("--gate-tolerance", type=float, default=-.25); recommend.add_argument("--kappa", type=float, default=.50)
    merge = command.add_parser("merge-receipts")
    merge.add_argument("--descriptor", type=Path, action="append", required=True)
    merge.add_argument("--label", type=Path, action="append", required=True)
    merge.add_argument("--out", type=Path, required=True)
    bank = command.add_parser("bank")
    bank.add_argument("--out", type=Path, required=True); bank.add_argument("--count", type=int, default=120); bank.add_argument("--parent-contract", required=True)
    label_plan = command.add_parser("plan-labels")
    label_plan.add_argument("--descriptors", type=Path, required=True); label_plan.add_argument("--labels", type=Path)
    label_plan.add_argument("--out", type=Path, required=True); label_plan.add_argument("--count", type=int, default=32)
    label_plan.add_argument("--era", required=True); label_plan.add_argument("--base-root", type=Path, required=True); label_plan.add_argument("--policy", type=Path, required=True)
    label_plan.add_argument("--months", required=True); label_plan.add_argument("--threshold-bps", type=float, default=50.0); label_plan.add_argument("--out-prefix", type=Path, required=True)
    label_run = command.add_parser("run-label-plan")
    label_run.add_argument("--plan", type=Path, required=True); label_run.add_argument("--out", type=Path, required=True); label_run.add_argument("--out-prefix", type=Path, required=True); label_run.add_argument("--max-workers", type=int, default=1)
    label_run.add_argument("--limit", type=int); label_run.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if args.command == "collect": result = _collect_descriptors(args)
    elif args.command == "collect-downstream": result = _collect_downstream_labels(args)
    elif args.command == "fit": result = _fit_proxies(args)
    elif args.command == "recommend": result = _recommend(args)
    elif args.command == "merge-receipts": result = _merge_receipts(args)
    elif args.command == "bank": result = _bank(args)
    elif args.command == "plan-labels": result = _plan_labels(args)
    elif args.command == "run-label-plan": result = _run_label_plan(args)
    else: raise AssertionError(args.command)
    print(result)


if __name__ == "__main__":
    main()
