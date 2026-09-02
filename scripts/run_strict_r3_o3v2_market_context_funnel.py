#!/usr/bin/env python3
"""Strict-OOS screen for *timestamp-level* market-dynamics context heads.

The future market labels in this research family are common to the whole
cross-section at a decision timestamp.  They must therefore not be evaluated
as asset-candidate rankers: any per-candidate variation would be accidental
tie-breaking by unrelated alpha inputs.  This producer instead:

1. compresses only the declared causal market-family fields to one robust
   value per timestamp (cross-sectional median where required);
2. learns the declared future market label with expanding, chronological OOF
   predictions;
3. maps that OOF context signal to the *mean realised policy net* of the
   canonical timestamp-local base-top-30% population, using only already
   resolved training timestamps; and
4. persists target-free held-timestamp receipts before joining held outcomes
   for evaluation.

The output is a temporal calibration/context signal for downstream meta and
MC1 experiments.  It cannot itself reorder assets within the same timestamp.
No live bundle, admission map, policy, or exchange path is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from audit_strict_r3_o3v2_market_dynamics_inputs import FAMILY_CANDIDATES
import run_strict_r3_long_supportive_label_funnel as supportive


SCHEMA = "strict_r3_o3v2_market_context_funnel_v1"
SEED = 1729
BASE_ROUTE = 0.30
EMBARGO = pd.Timedelta(hours=12)
MIN_TRAIN_TIMESTAMPS = 2_000
MIN_OOF_TIMESTAMPS = 1_000
TARGET_CLIP = (0.005, 0.995)
DEFAULT_LEDGER = ROOT / "data_perp/artifacts/strict_r3_schema_v2_prequential_ledger_targetfree_long_2024_2026_raw15m_strictfull_20260812_v1/prequential_stack_ledger.parquet"
DEFAULT_LABELS = ROOT / "data_perp/artifacts/strict_r3_o3v2_market_dynamics_labels_20260825_v2/market_dynamics_labels.parquet"


@dataclass(frozen=True)
class Fold:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    cohort: str


@dataclass(frozen=True)
class Target:
    name: str
    column: str
    direction: float
    scale: float = 1.0


FOLDS: tuple[Fold, ...] = (
    Fold("dev_2025_q2", pd.Timestamp("2025-04-01T00:00:00Z"), pd.Timestamp("2025-07-01T00:00:00Z"), "development"),
    Fold("dev_2025_q3", pd.Timestamp("2025-07-01T00:00:00Z"), pd.Timestamp("2025-10-01T00:00:00Z"), "development"),
    Fold("holdout_2025_q4", pd.Timestamp("2025-10-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z"), "holdout"),
    Fold("oos_2026_q1", pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-04-01T00:00:00Z"), "portability"),
    Fold("oos_2026_q2", pd.Timestamp("2026-04-01T00:00:00Z"), pd.Timestamp("2026-07-01T00:00:00Z"), "portability"),
    Fold("oos_2026_jul", pd.Timestamp("2026-07-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z"), "portability"),
)

GROUPS: dict[str, tuple[Target, ...]] = {
    "trend": (
        Target("market_trend_continuation", "market_trend_continuation_12h", 1.0),
        Target("market_signed_directional_efficiency", "market_signed_directional_efficiency_12h", 1.0),
        Target("market_time_to_trend_break", "market_time_to_trend_break_12h", 1.0),
    ),
    "stretch": (
        Target("market_anchor_reversion_fraction", "market_anchor_reversion_fraction_12h", 1.0),
        Target("market_time_to_anchor_reentry", "market_time_to_anchor_reentry_12h", -1.0),
        Target("market_reversion_overshoot", "market_reversion_overshoot_12h", -1.0),
    ),
    "volatility": (
        Target("market_vol_change", "market_vol_change_12h", 1.0),
        Target("market_vol_acceleration", "market_vol_acceleration_12h", 1.0),
    ),
    "volatility_release": (
        Target("market_vol_of_vol", "market_vol_of_vol_12h", -1.0),
        Target("market_compression_release_ratio", "market_compression_release_ratio_12h", 1.0),
        Target("market_time_to_vol_breakout", "market_time_to_vol_breakout_12h", -1.0),
        Target("market_low_vol_persistence", "market_low_vol_persistence_12h", 1.0),
    ),
    "breadth": (
        Target("market_breadth_change", "market_breadth_change_12h", 1.0),
        Target("market_directional_breadth", "market_directional_breadth_12h", 1.0),
        Target("market_breadth_persistence", "market_breadth_persistence_12h", 1.0),
    ),
    "dispersion": (
        Target("cross_sectional_dispersion_change", "cross_sectional_dispersion_change_12h", 1.0),
        Target("cross_sectional_tail_spread", "cross_sectional_tail_spread_12h", 1.0),
        Target("idiosyncratic_variance_share", "idiosyncratic_variance_share_12h", 1.0),
    ),
    "dependence": (
        Target("market_pairwise_correlation_change", "market_pairwise_correlation_change_12h", -1.0),
        Target("market_pc1_share_change", "market_pc1_share_change_12h", -1.0),
        Target("market_factor_r2_change", "market_factor_r2_change_12h", -1.0),
    ),
    "leadership": (
        Target("leader_continuation", "leader_continuation_12h", 1.0),
        Target("cross_sectional_rank_persistence", "cross_sectional_rank_persistence_12h", 1.0),
        Target("topk_leadership_turnover", "topk_leadership_turnover_12h", -1.0),
    ),
    "flow": (
        Target("market_turnover_change", "market_turnover_change_12h", 1.0),
        Target("market_volume_concentration_change", "market_volume_concentration_change_12h", -1.0),
    ),
    "stress": (
        Target("market_future_max_drawdown", "market_future_max_drawdown_12h", -1.0),
        Target("market_downside_upside_semivol_ratio", "market_downside_upside_semivol_ratio_12h", -1.0),
        Target("market_jump_asymmetry", "market_jump_asymmetry_12h", -1.0),
    ),
    "structural": (
        Target("market_state_shift", "market_state_shift_12h", -1.0),
        Target("market_distribution_break", "market_distribution_break_12h", -1.0),
        Target("market_time_to_change_point", "market_time_to_change_point_12h", 1.0),
    ),
    "leverage": (
        Target("market_open_interest_change", "market_open_interest_change_12h", 1.0),
        # Funding is naturally measured in small per-hour rates.  Scaling the
        # training response preserves its ordering but avoids a no-split
        # numerical degeneracy in the tree objective.
        Target("market_funding_impulse", "market_funding_impulse_12h", 1.0, 100_000.0),
        Target("market_liquidation_imbalance_proxy", "market_liquidation_imbalance_proxy_12h", 1.0),
    ),
}

BLOCK = {
    "trend": "trend_persistence",
    "stretch": "stretch_reversion",
    "volatility": "volatility_regime",
    "volatility_release": "volatility_regime",
    "breadth": "breadth_participation",
    "dispersion": "cross_sectional_dispersion",
    "dependence": "dependence_common_factor",
    "leadership": "leadership_rotation",
    "flow": "volume_flow",
    "stress": "tail_stress",
    "structural": "structural_transition",
    "leverage": "leverage_positioning",
}


def _finite(values: pd.Series | np.ndarray) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(float) if isinstance(values, pd.Series) else np.where(np.isfinite(values), values, np.nan)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_exclusive(path: Path, value: object) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)


def _route_top30(frame: pd.DataFrame) -> pd.Series:
    """Candidate-ID tie-stable top-30% route, before outcome eligibility."""
    working = frame.loc[:, ["__decision_ts__", "candidate_id", "prequential_base_score"]].copy()
    working["_score"] = pd.to_numeric(working.pop("prequential_base_score"), errors="coerce").fillna(-np.inf)
    working["_row"] = np.arange(len(working), dtype=np.int64)
    working = working.sort_values(["__decision_ts__", "_score", "candidate_id", "_row"], ascending=[True, False, True, True], kind="stable")
    size = working.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size")
    selected = working.groupby("__decision_ts__", sort=False).cumcount().lt(np.ceil(size * BASE_ROUTE).astype(int))
    output = pd.Series(False, index=frame.index)
    output.iloc[working["_row"].to_numpy()] = selected.to_numpy(bool)
    return output


def _read_labels(labels: Path, targets: Sequence[Target]) -> pd.DataFrame:
    columns = ["__decision_ts__", "market_label_valid", "market_label_available_ts", *(target.column for target in targets)]
    raw = pd.read_parquet(labels, columns=columns)
    for column in ("__decision_ts__", "market_label_available_ts"):
        raw[column] = pd.to_datetime(raw[column], utc=True, errors="raise")
    # Every row at one timestamp must carry exactly the same market-wide label.
    for target in targets:
        variation = raw.groupby("__decision_ts__", sort=False)[target.column].nunique(dropna=True)
        if (variation.gt(1)).any():
            raise AssertionError(f"market target is not timestamp-global: {target.column}")
    if raw.groupby("__decision_ts__", sort=False)["market_label_valid"].nunique(dropna=False).gt(1).any():
        raise AssertionError("market_label_valid differs within a timestamp")
    return raw.groupby("__decision_ts__", as_index=False, sort=True).first()


def _context_panel(ledger: Path, labels: pd.DataFrame, fields: Sequence[str]) -> pd.DataFrame:
    schema = set(pq.ParquetFile(ledger).schema_arrow.names)
    required = {
        "candidate_id", "__decision_ts__", "prequential_base_score", "base_contract_complete",
        "base_feature_available_fraction", "policy_path_valid", "policy_net_bps", "policy_label_available_ts",
        *fields,
    }
    missing = sorted(required - schema)
    if missing:
        raise AssertionError(f"missing frozen causal context inputs: {missing}")
    raw = pd.read_parquet(ledger, columns=sorted(required))
    raw["__decision_ts__"] = pd.to_datetime(raw["__decision_ts__"], utc=True, errors="raise")
    raw["policy_label_available_ts"] = pd.to_datetime(raw["policy_label_available_ts"], utc=True, errors="coerce")
    raw = raw.loc[raw["__decision_ts__"].lt(labels["__decision_ts__"].max() + pd.Timedelta(hours=1))].copy()
    base_ok = raw["base_contract_complete"].fillna(False).astype(bool) & pd.to_numeric(raw["base_feature_available_fraction"], errors="coerce").ge(0.90)
    route = pd.Series(False, index=raw.index)
    route.loc[base_ok] = _route_top30(raw.loc[base_ok])
    # Medians intentionally collapse asset-varying causal fields to one
    # timestamp representation.  This preserves the market-level semantics of
    # the label and blocks candidate-specific stack outputs from tie-breaking.
    features = raw.groupby("__decision_ts__", sort=True)[list(fields)].median(numeric_only=True).reset_index()
    routed = raw.loc[route].copy()
    policy_ok = routed["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(pd.to_numeric(routed["policy_net_bps"], errors="coerce"))
    policy = routed.loc[policy_ok].groupby("__decision_ts__", sort=True).agg(
        base_top30_policy_mean_bps=("policy_net_bps", "mean"),
        base_top30_policy_rows=("candidate_id", "size"),
        policy_label_available_ts=("policy_label_available_ts", "max"),
    ).reset_index()
    result = features.merge(policy, on="__decision_ts__", how="left", validate="one_to_one")
    result = result.merge(labels, on="__decision_ts__", how="left", validate="one_to_one")
    return result.sort_values("__decision_ts__", kind="stable").reset_index(drop=True)


def _matrix(frame: pd.DataFrame, fields: Sequence[str], medians: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    matrix = frame.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if medians is None:
        medians = np.nanmedian(matrix, axis=0)
        medians[~np.isfinite(medians)] = 0.0
    matrix = np.where(np.isfinite(matrix), matrix, medians)
    return matrix.astype(np.float32), medians.astype(np.float32)


def _fit_target(train: pd.DataFrame, held: pd.DataFrame, *, fields: Sequence[str], target: Target, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    ordered = train.sort_values("__decision_ts__", kind="stable").reset_index(drop=True)
    y = pd.to_numeric(ordered[target.column], errors="coerce").to_numpy(float) * target.scale
    lo, hi = np.nanquantile(y, TARGET_CLIP)
    y = np.clip(y, lo, hi)
    oof = np.full(len(ordered), np.nan, dtype=float)
    # Expanding folds preserve temporal order.  The first tranche seeds the
    # model; every OOF value is made by a model trained strictly before it.
    boundaries = np.linspace(0, len(ordered), 6, dtype=int)
    for fold in range(1, len(boundaries) - 1):
        fit_end, valid_end = int(boundaries[fold]), int(boundaries[fold + 1])
        if fit_end < MIN_TRAIN_TIMESTAMPS // 2 or valid_end <= fit_end:
            continue
        x_fit, medians = _matrix(ordered.iloc[:fit_end], fields)
        x_valid, _ = _matrix(ordered.iloc[fit_end:valid_end], fields, medians)
        model = supportive._model_regressor(seed=seed + fold)
        model.fit(x_fit, y[:fit_end])
        oof[fit_end:valid_end] = target.direction * model.predict(x_valid) / target.scale
    policy = pd.to_numeric(ordered["base_top30_policy_mean_bps"], errors="coerce").to_numpy(float)
    usable = np.isfinite(oof) & np.isfinite(policy)
    if usable.sum() < MIN_OOF_TIMESTAMPS or np.unique(oof[usable]).size < 16:
        return np.full(len(held), np.nan), np.full(len(held), np.nan), {"status": "insufficient_oof", "map_rows": int(usable.sum())}
    corr = spearmanr(oof[usable], policy[usable]).statistic
    orientation = 1.0 if not np.isfinite(corr) or corr >= 0 else -1.0
    mapper = IsotonicRegression(increasing=True, out_of_bounds="clip")
    mapper.fit(orientation * oof[usable], policy[usable])
    x_train, medians = _matrix(ordered, fields)
    x_held, _ = _matrix(held, fields, medians)
    model = supportive._model_regressor(seed=seed + 100)
    model.fit(x_train, y)
    raw = target.direction * model.predict(x_held) / target.scale
    expected = mapper.predict(orientation * raw)
    return raw.astype(np.float32), expected.astype(np.float32), {
        "status": "ok", "map_rows": int(usable.sum()), "target_clip_low": float(lo), "target_clip_high": float(hi),
        "map_oof_policy_spearman": float(corr), "map_orientation": float(orientation),
    }


def _write_score_receipt(out: Path, *, fold: Fold, target: Target, held: pd.DataFrame, raw: np.ndarray, expected: np.ndarray, audit: dict[str, object]) -> None:
    root = out / "target_free_scores" / target.name
    root.mkdir(parents=True, exist_ok=True)
    score_path = root / f"fold={fold.name}.parquet"
    audit_path = out / "audit_parts" / f"{target.name}__{fold.name}.json"
    if score_path.exists() or audit_path.exists():
        raise FileExistsError(f"immutable receipt already exists: {target.name} {fold.name}")
    receipt = pd.DataFrame({
        "__decision_ts__": held["__decision_ts__"].to_numpy(),
        "predicted_context_raw": raw,
        "predicted_context_expected_policy_bps": expected,
    })
    prohibited = [column for column in receipt if column.startswith("market_") or "policy_net" in column]
    if prohibited:
        raise AssertionError(f"target-free context receipt leaked labels/outcomes: {prohibited}")
    receipt.to_parquet(score_path, index=False, compression="zstd")
    _write_exclusive(audit_path, {"target": target.name, "target_column": target.column, "fold": fold.name, **audit})


def _eligible_train(panel: pd.DataFrame, cutoff: pd.Timestamp, target: Target) -> pd.DataFrame:
    valid = (
        panel["market_label_valid"].fillna(False).astype(bool)
        & panel["market_label_available_ts"].lt(cutoff - EMBARGO)
        & panel["policy_label_available_ts"].lt(cutoff - EMBARGO)
        & np.isfinite(pd.to_numeric(panel[target.column], errors="coerce"))
        & np.isfinite(pd.to_numeric(panel["base_top30_policy_mean_bps"], errors="coerce"))
    )
    return panel.loc[valid].copy()


def _evaluate(out: Path, panel: pd.DataFrame, *, targets: Sequence[Target], folds: Sequence[Fold]) -> None:
    rows: list[dict[str, object]] = []
    for target in targets:
        for fold in folds:
            receipt = pd.read_parquet(out / "target_free_scores" / target.name / f"fold={fold.name}.parquet")
            joined = receipt.merge(panel[["__decision_ts__", target.column, "base_top30_policy_mean_bps"]], on="__decision_ts__", how="left", validate="one_to_one")
            score = pd.to_numeric(joined["predicted_context_expected_policy_bps"], errors="coerce").to_numpy(float)
            raw = pd.to_numeric(joined["predicted_context_raw"], errors="coerce").to_numpy(float)
            policy = pd.to_numeric(joined["base_top30_policy_mean_bps"], errors="coerce").to_numpy(float)
            label = pd.to_numeric(joined[target.column], errors="coerce").to_numpy(float)
            valid = np.isfinite(score) & np.isfinite(policy)
            label_valid = valid & np.isfinite(label) & np.isfinite(raw)
            all_ev = float(np.mean(policy[valid])) if valid.any() else np.nan
            policy_ic = float(spearmanr(score[valid], policy[valid]).statistic) if valid.sum() >= 12 else np.nan
            label_ic = float(spearmanr(raw[label_valid], label[label_valid]).statistic) if label_valid.sum() >= 12 else np.nan
            for fraction, band in ((0.20, "top20"), (0.30, "top30"), (1.00, "all")):
                count = max(1, int(np.ceil(valid.sum() * fraction)))
                selected = np.argsort(score[valid], kind="stable")[-count:]
                values = policy[valid][selected]
                rows.append({
                    "target": target.name, "target_column": target.column, "fold": fold.name, "cohort": fold.cohort,
                    "band": band, "timestamps": int(len(values)), "policy_mean_bps": float(values.mean()),
                    "policy_total_bps": float(values.sum()), "delta_vs_all_bps": float(values.mean() - all_ev),
                    "policy_time_rank_ic": policy_ic, "target_rank_ic": label_ic,
                })
    metrics = pd.DataFrame(rows)
    metrics.to_parquet(out / "market_context_metrics.parquet", index=False, compression="zstd")
    summary = metrics.groupby(["target", "target_column", "cohort", "band"], as_index=False).agg(
        mean_policy_bps=("policy_mean_bps", "mean"), median_policy_bps=("policy_mean_bps", "median"),
        worst_policy_bps=("policy_mean_bps", "min"), mean_delta_vs_all_bps=("delta_vs_all_bps", "mean"),
        mean_policy_time_rank_ic=("policy_time_rank_ic", "mean"), mean_target_rank_ic=("target_rank_ic", "mean"), folds=("fold", "nunique"),
    )
    summary.to_parquet(out / "market_context_summary.parquet", index=False, compression="zstd")


def run(*, ledger: Path, labels: Path, out: Path, group: str, folds: Sequence[Fold]) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    targets = GROUPS[group]
    fields = tuple(FAMILY_CANDIDATES[BLOCK[group]])
    labels_frame = _read_labels(labels, targets)
    panel = _context_panel(ledger, labels_frame, fields)
    out.mkdir(parents=True, exist_ok=False)
    (out / "target_free_scores").mkdir()
    (out / "audit_parts").mkdir()
    _write_exclusive(out / "run_contract.json", {
        "schema": SCHEMA, "scope": "offline timestamp-level market-context research only; no live or policy mutation",
        "group": group, "targets": [target.__dict__ for target in targets], "fields": list(fields),
        "ledger": str(ledger.resolve()), "ledger_sha256": _sha256(ledger), "labels": str(labels.resolve()), "labels_sha256": _sha256(labels),
        "base_route": "timestamp-local canonical top 30% prequential base score; used only to aggregate policy outcome mapping",
        "folds": [fold.__dict__ for fold in folds],
    })
    for fold_index, fold in enumerate(folds):
        train = _eligible_train(panel.loc[panel["__decision_ts__"].lt(fold.start)], fold.start, targets[0])
        held = panel.loc[(panel["__decision_ts__"].ge(fold.start)) & (panel["__decision_ts__"].lt(fold.end))].copy()
        if len(held) < 100:
            raise AssertionError(f"{fold.name}: insufficient held timestamps")
        for target_index, target in enumerate(targets):
            target_train = _eligible_train(panel.loc[panel["__decision_ts__"].lt(fold.start)], fold.start, target)
            if len(target_train) < MIN_TRAIN_TIMESTAMPS:
                raise AssertionError(f"{target.name} {fold.name}: insufficient strict training timestamps {len(target_train)}")
            raw, expected, extra = _fit_target(target_train, held, fields=fields, target=target, seed=SEED + fold_index * 100 + target_index * 11)
            _write_score_receipt(out, fold=fold, target=target, held=held, raw=raw, expected=expected, audit={
                "train_timestamps": int(len(target_train)), "held_timestamps": int(len(held)), "fit_cutoff": str(fold.start),
                "embargo_hours": 12, "feature_aggregation": "one cross-sectional median per causal family field and decision timestamp",
                "policy_map_outcome": "mean policy_net_bps among target-free timestamp-local base-top30 candidates",
                **extra,
            })
            print(json.dumps({"event": "scored", "target": target.name, "fold": fold.name, **extra}, sort_keys=True), flush=True)
    _evaluate(out, panel, targets=targets, folds=folds)
    _write_exclusive(out / "run_manifest.json", {
        "schema": SCHEMA, "scope": "offline timestamp-level context screen; does not promote a live feature or alter a model",
        "causality": {
            "features": "frozen target-free causal inputs, aggregated before any outcome join",
            "targets": "future H12 market labels, usable only when label_available_ts is before fold cutoff minus 12h",
            "policy_map": "expanding OOF target scores mapped only to prior resolved base-top30 mean policy outcomes",
            "held_receipts": "written without raw target or held policy outcome before metric join",
            "cross_section": "one context value per timestamp; no accidental asset candidate tie-breaking",
        },
        "rows": int(len(panel)), "label_rows": int(panel["market_label_valid"].fillna(False).sum()),
        "timestamp_feature_coverage": {field: float(pd.to_numeric(panel[field], errors="coerce").notna().mean()) for field in fields},
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=sorted(GROUPS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    args = parser.parse_args()
    print(run(ledger=args.ledger.resolve(), labels=args.labels.resolve(), out=args.out.resolve(), group=args.group, folds=FOLDS))


if __name__ == "__main__":
    main()
