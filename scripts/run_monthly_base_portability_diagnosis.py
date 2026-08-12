#!/usr/bin/env python3
"""Read-only monthly diagnosis of frozen F0/R3 base portability.

The runner separates three questions on a pre-existing strict-OOF base score:

* did the decision-time F0 candidate population shift from the original fit
  population;
* did score-to-R3/exact-net economics change at fixed score;
* would a recent model, trained only on already-resolved *strict-OOF R3*
  support, produce a stable ordering on the same monthly candidates.

It neither tunes, promotes, nor writes models.  The historical TP6 source has
five empty original-label partitions, so reproducing the original full-label
refit is deliberately forbidden; the stability probe uses prior strict-OOF R3
labels only and records that limitation in its immutable manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.base_population_drift import (
    feature_distribution_drift,
    held_out_adversarial_separability,
    population_composition,
)
from extreme_price_movements.base_portability_source_materializer import (
    BasePortabilitySourceContract,
    BasePortabilitySourceError,
    BasePortabilitySourceMaterializer,
    TransportScope,
)
from extreme_price_movements.base_relationship_drift import (
    adjacent_month_fixed_bin_decomposition,
    monthly_relationship_metrics,
)
from extreme_price_movements.tp6_portability_data import SIDES


SCHEMA = "monthly_base_portability_diagnosis_v1"
DEFAULT_LINEAGE = ROOT / "data_perp/artifacts/feature_leaf_reasoning_portability_base_funnel_merged_robustz_20260804_v1/base_feature_arm_lineage.json"
DEFAULT_OOF_ROOT = ROOT / "data_perp/artifacts/feature_leaf_reasoning_strict_oof_f0_20260804_v1"
DEFAULT_PANEL = ROOT / "data_perp/artifacts/full_universe_t2_t4_panel_20260801_v3"
DEFAULT_WINNER = ROOT / "data_perp/artifacts/full_universe_tp6_sl4_h12_sidecar_20260802_v1"
DEFAULT_ROBUST = ROOT / "data_perp/artifacts/tp6_sl4_robust_clear_labels_20260802_v1"
TRANSPORTS = {
    "transport_a_2023q4_to_2024h1": ("2023-04-01", "2024-07-01"),
    "transport_b_2024h1_to_2024h2_to_date": ("2023-04-01", "2024-11-01"),
}
REFIT_PARAMS = dict(
    n_estimators=140, learning_rate=0.05, num_leaves=31, min_child_samples=350,
    subsample=0.80, colsample_bytree=0.80, reg_lambda=8.0, n_jobs=1, verbosity=-1,
    objective="multiclass", num_class=3, random_state=20260805,
)
# These TP6 sidecar partitions are physically empty in the frozen historical
# source.  They are deliberately named, rather than hidden behind a row-count
# condition, because strict OOF still contains these assets and an apparent
# "original-label refit" would silently lose them.
EMPTY_ORIGINAL_TP6_LABEL_PARTITIONS = ("GMT", "KSM", "LRC", "TRX", "XMR")


class MonthlyBasePortabilityError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc(value: object) -> pd.Timestamp:
    out = pd.Timestamp(value)
    return out.tz_localize("UTC") if out.tzinfo is None else out.tz_convert("UTC")


def _strict_oof_path(root: Path, transport: str, side: str, name: str) -> Path:
    path = root / "base_prediction_shards" / transport / side / name
    if not path.is_file():
        raise MonthlyBasePortabilityError(f"strict OOF shard is absent: {path}")
    return path


def validate_direct_strict_oof(frame: pd.DataFrame, *, side: str) -> pd.DataFrame:
    """Validate exact direct R3 strict-OOF score and label identity semantics."""
    required = {
        "candidate_id", "decision_ts", "label_available_ts", "side_name", "r3_class", "net_bps",
        "p_adverse", "p_weak", "p_clear", "base_raw", "base_fit_cutoff_ts",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise MonthlyBasePortabilityError(f"strict OOF shard lacks columns: {missing}")
    work = frame.copy()
    for field in ("decision_ts", "label_available_ts", "base_fit_cutoff_ts"):
        work[field] = pd.to_datetime(work[field], utc=True, errors="coerce")
    if work[["candidate_id", "decision_ts", "label_available_ts", "base_fit_cutoff_ts"]].isna().any().any():
        raise MonthlyBasePortabilityError("strict OOF identity/fit timestamps must be valid UTC")
    if work.candidate_id.duplicated().any() or not work.side_name.astype(str).eq(side).all():
        raise MonthlyBasePortabilityError("strict OOF candidate identity must be unique and same-side")
    if not work.base_fit_cutoff_ts.lt(work.decision_ts).all():
        raise MonthlyBasePortabilityError("strict OOF fit cutoff must precede every decision")
    classes = pd.to_numeric(work.r3_class, errors="coerce")
    if classes.isna().any() or set(classes.astype(int)).difference({0, 1, 2}):
        raise MonthlyBasePortabilityError("strict OOF requires R3 classes 0/1/2")
    probabilities = work[["p_adverse", "p_weak", "p_clear"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1.0, atol=2e-5, rtol=0.0):
        raise MonthlyBasePortabilityError("strict OOF R3 probabilities must be a finite simplex")
    contrast = probabilities[:, 2] - probabilities[:, 0]
    if not np.allclose(pd.to_numeric(work.base_raw, errors="coerce"), contrast, atol=2e-5, rtol=0.0, equal_nan=False):
        raise MonthlyBasePortabilityError("base_raw must be direct P(clear)-P(adverse), without conversion")
    work["r3_clear"] = classes.eq(2).astype(np.int8)
    return work


def prior_resolved_oof_support(frame: pd.DataFrame, *, decision_ts: object) -> pd.DataFrame:
    """Only earlier strict-OOF labels, resolved before the current decision."""
    cutoff = _utc(decision_ts)
    return frame.loc[frame.label_available_ts.lt(cutoff)].copy()


def _read_oof(root: Path, *, transport: str, side: str, name: str) -> tuple[pd.DataFrame, Path]:
    path = _strict_oof_path(root, transport, side, name)
    return validate_direct_strict_oof(pd.read_parquet(path), side=side), path


def _source_for_ids(
    materializer: BasePortabilitySourceMaterializer,
    *, transport: str, side: str, start: pd.Timestamp, end: pd.Timestamp, ids: Iterable[str],
) -> pd.DataFrame:
    """Read the frozen F0 *inputs* for an exact strict-OOF identity set.

    This deliberately does not open the TP6 label sidecars.  Five historical
    sidecar files are physically empty, while the strict-OOF ledger retains
    valid labels for their candidates.  Input-population and recent-OOF-refit
    diagnostics need the decision-time feature contract only; coupling them
    to those broken files would turn a label-storage defect into an apparent
    feature-coverage failure.  Original-label refits remain explicitly
    prohibited in the run manifest.
    """
    needed = pd.Index(list(ids), dtype="object")
    fields = materializer.lineage.selected_features(run=transport, side=side)
    columns = ["candidate_id", "__ts__", "side_name", *fields]
    chunks: list[pd.DataFrame] = []
    for path in sorted((materializer.contract.panel / "parts").glob("*.parquet")):
        part = pd.read_parquet(path, columns=columns)
        part = part.loc[part["candidate_id"].isin(needed) & part["side_name"].astype(str).eq(side)].copy()
        if not part.empty:
            chunks.append(part)
    source = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=columns)
    source["decision_ts"] = pd.to_datetime(source.pop("__ts__"), utc=True, errors="coerce")
    if source["decision_ts"].isna().any():
        raise MonthlyBasePortabilityError("F0 source contains invalid decision timestamps")
    source = source.loc[source.decision_ts.ge(start) & source.decision_ts.lt(end)].copy()
    # R3/TP6 H12 labels resolve exactly 13 hours after decision by the sealed
    # source contract.  It is used only to enforce prior-resolved support.
    source["label_available_ts"] = source["decision_ts"] + pd.Timedelta(hours=13)
    source["label_valid"] = True
    source["asset"] = source["candidate_id"].astype(str).str.split("|", n=1, regex=False).str[0].astype("string")
    if source.candidate_id.duplicated().any() or len(source) != len(needed) or set(source.candidate_id) != set(needed):
        raise MonthlyBasePortabilityError("F0 source and strict OOF candidate identities do not match exactly")
    return source


def _monthly_refit_stability(
    *, materializer: BasePortabilitySourceMaterializer, source: pd.DataFrame,
    transport: str, side: str, strict_oof: pd.DataFrame, outer: pd.DataFrame, train_start: pd.Timestamp,
    min_rows: int, recent_days: int,
) -> pd.DataFrame:
    """Probe score stability with prior resolved strict-OOF R3 labels only."""
    result: list[dict[str, object]] = []
    outer = outer.copy()
    outer["month"] = outer.decision_ts.dt.tz_localize(None).dt.to_period("M").astype(str)
    for month, current in outer.groupby("month", sort=True, observed=True):
        decision_start = current.decision_ts.min()
        support = prior_resolved_oof_support(strict_oof, decision_ts=decision_start)
        recent_start = max(train_start, decision_start - pd.Timedelta(days=recent_days))
        support = support.loc[support.decision_ts.ge(recent_start)].copy()
        if len(support) < min_rows or support.r3_class.nunique() < 3:
            result.append({"transport": transport, "side_name": side, "month": month, "status": "INSUFFICIENT_PRIOR_STRICT_OOF_SUPPORT", "support_rows": int(len(support)), "scored_rows": int(len(current))})
            continue
        fields = materializer.lineage.selected_features(run=transport, side=side)
        raw = source.set_index("candidate_id", verify_integrity=True)
        train = raw.loc[support.candidate_id]
        test = raw.loc[current.candidate_id]
        medians = train.loc[:, fields].apply(pd.to_numeric, errors="coerce").median().fillna(0.0)
        x_train = train.loc[:, fields].apply(pd.to_numeric, errors="coerce").fillna(medians)
        x_test = test.loc[:, fields].apply(pd.to_numeric, errors="coerce").fillna(medians)
        model = lgb.LGBMClassifier(**REFIT_PARAMS).fit(x_train, support.r3_class.astype(int))
        proba = model.predict_proba(x_test)
        recent_score = proba[:, 2] - proba[:, 0]
        frozen_score = current.set_index("candidate_id").loc[test.index, "base_raw"].to_numpy(float)
        # The same candidate rows have resolved R3 labels in ``current``;
        # score both models against them before using similarity alone to
        # diagnose stale-model drift.
        scored = current.set_index("candidate_id").loc[test.index].reset_index()
        scored["recent_refit_score"] = recent_score
        frozen_metrics, _ = monthly_relationship_metrics(
            scored, score_col="base_raw", target_col="r3_clear", winner_col="r3_clear"
        )
        refit_metrics, _ = monthly_relationship_metrics(
            scored, score_col="recent_refit_score", target_col="r3_clear", winner_col="r3_clear"
        )
        frozen_rank_ic = float(frozen_metrics.loc[frozen_metrics.scope.eq("pooled"), "within_query_rank_ic"].iloc[0])
        refit_rank_ic = float(refit_metrics.loc[refit_metrics.scope.eq("pooled"), "within_query_rank_ic"].iloc[0])
        correlation = spearmanr(frozen_score, recent_score, nan_policy="omit").statistic
        top_n = max(1, int(np.ceil(0.05 * len(test))))
        frozen_top = set(test.index[np.argsort(frozen_score, kind="stable")[-top_n:]])
        recent_top = set(test.index[np.argsort(recent_score, kind="stable")[-top_n:]])
        result.append({
            "transport": transport, "side_name": side, "month": month, "status": "OK_PRIOR_STRICT_OOF_R3_ONLY",
            "support_rows": int(len(support)), "scored_rows": int(len(test)), "recent_days": int(recent_days),
            "frozen_rank_ic": frozen_rank_ic, "refit_rank_ic": refit_rank_ic,
            "score_spearman": float(correlation), "top5_jaccard": float(len(frozen_top & recent_top) / len(frozen_top | recent_top)),
        })
    return pd.DataFrame(result)


def _write(frame: pd.DataFrame, path: Path, files: dict[str, str]) -> None:
    frame.to_parquet(path, index=False, compression="zstd")
    files[path.name] = _sha256(path)


def run(args: argparse.Namespace) -> Path:
    destination = Path(args.output)
    if destination.exists():
        raise MonthlyBasePortabilityError(f"output already exists: {destination}")
    source_contract = BasePortabilitySourceContract(panel=Path(args.panel), winner=Path(args.winner), robust=Path(args.robust), lineage=Path(args.lineage))
    materializer = BasePortabilitySourceMaterializer(source_contract)
    oof_root = Path(args.oof_root)
    selected_transports = tuple(args.transport) if args.transport else tuple(TRANSPORTS)
    unknown = sorted(set(selected_transports).difference(TRANSPORTS))
    if unknown:
        raise MonthlyBasePortabilityError(f"unknown transports: {unknown}")
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    files: dict[str, str] = {}
    source_hashes: dict[str, str] = {"lineage": _sha256(Path(args.lineage))}
    try:
        relationship: list[pd.DataFrame] = []
        deciles: list[pd.DataFrame] = []
        population: list[pd.DataFrame] = []
        distribution: list[pd.DataFrame] = []
        adversarial: list[dict[str, object]] = []
        refit: list[pd.DataFrame] = []
        decomposition: list[pd.DataFrame] = []
        for transport in selected_transports:
            train_start, _ = map(_utc, TRANSPORTS[transport])
            transport_outer: list[pd.DataFrame] = []
            for side in SIDES:
                print(f"[portability] {transport}/{side}: reading sealed OOF", flush=True)
                strict, strict_path = _read_oof(oof_root, transport=transport, side=side, name="strict_oof_predictions.parquet")
                outer, outer_path = _read_oof(oof_root, transport=transport, side=side, name="outer_predictions.parquet")
                source_hashes[str(strict_path)] = _sha256(strict_path)
                source_hashes[str(outer_path)] = _sha256(outer_path)
                transport_outer.append(outer)
                # Relationship metrics use only frozen direct OOF predictions and their resolved R3/net labels.
                for target, winner, kind in (("r3_clear", "r3_clear", "r3"), ("net_bps", "r3_clear", "net")):
                    metrics, curve = monthly_relationship_metrics(outer, score_col="base_raw", target_col=target, winner_col=winner)
                    metrics["transport"], metrics["target_kind"] = transport, kind
                    curve["transport"], curve["target_kind"] = transport, kind
                    # Side tables come from their isolated source.  True
                    # global pooled rows are built below after both sides are
                    # present; otherwise a one-side "pooled" row is a
                    # misleading duplicate of the side result.
                    relationship.append(metrics.loc[metrics.scope.eq("side")]); deciles.append(curve.loc[curve.scope.eq("side")])
                fixed = adjacent_month_fixed_bin_decomposition(outer, score_col="base_raw", outcome_col="net_bps")
                fixed["transport"], fixed["side_name"] = transport, side
                decomposition.append(fixed)
                # Population shift uses original fit-window F0 inputs, never realised labels for a feature drift metric.
                fields = materializer.lineage.selected_features(run=transport, side=side)
                evaluation_start = outer.decision_ts.min()
                print(f"[portability] {transport}/{side}: loading F0 source", flush=True)
                all_ids = pd.Index(pd.concat([strict.candidate_id, outer.candidate_id]).unique(), dtype="object")
                source = _source_for_ids(materializer, transport=transport, side=side, start=train_start, end=outer.decision_ts.max() + pd.Timedelta(hours=1), ids=all_ids)
                reference = source.loc[source.decision_ts.lt(evaluation_start)].copy()
                current = source.loc[source.candidate_id.isin(outer.candidate_id)].copy()
                reference = reference.loc[reference.label_available_ts.lt(evaluation_start)].copy()
                current["month"] = current.decision_ts.dt.tz_localize(None).dt.to_period("M").astype(str)
                population.append(population_composition(current, month_column="month", side_column="side_name", asset_column="asset", label_valid_column="label_valid", numeric_columns=fields).assign(transport=transport))
                for month, month_current in current.groupby("month", observed=True, sort=True):
                    # Strictly earlier source reference is required by every drift primitive.
                    current_start = month_current.decision_ts.min()
                    train_ref = reference.loc[reference.decision_ts.lt(current_start)]
                    if len(train_ref) < args.min_drift_rows or len(month_current) < args.min_drift_rows:
                        continue
                    drift = feature_distribution_drift(train_ref, month_current, feature_names=fields, timestamp_column="decision_ts")
                    drift["transport"], drift["side_name"], drift["month"] = transport, side, month
                    distribution.append(drift)
                    sep = held_out_adversarial_separability(train_ref, month_current, feature_names=fields, timestamp_column="decision_ts", max_rows_per_population=args.max_drift_rows)
                    adversarial.append({"transport": transport, "side_name": side, "month": month, "held_out_auc": sep.held_out_auc, "reference_rows_sampled": sep.train_rows_sampled, "current_rows_sampled": sep.current_rows_sampled, "held_out_rows": sep.held_out_rows})
                print(f"[portability] {transport}/{side}: refit stability", flush=True)
                refit.append(_monthly_refit_stability(materializer=materializer, source=source, transport=transport, side=side, strict_oof=strict, outer=outer, train_start=train_start, min_rows=args.min_refit_rows, recent_days=args.recent_refit_days))
                print(f"[portability] {transport}/{side}: complete", flush=True)
            pooled_outer = pd.concat(transport_outer, ignore_index=True)
            for target, winner, kind in (("r3_clear", "r3_clear", "r3"), ("net_bps", "r3_clear", "net")):
                metrics, curve = monthly_relationship_metrics(pooled_outer, score_col="base_raw", target_col=target, winner_col=winner)
                metrics["transport"], metrics["target_kind"] = transport, kind
                curve["transport"], curve["target_kind"] = transport, kind
                relationship.append(metrics.loc[metrics.scope.eq("pooled")]); deciles.append(curve.loc[curve.scope.eq("pooled")])
        _write(pd.concat(relationship, ignore_index=True), staging / "monthly_relationship_metrics.parquet", files)
        _write(pd.concat(deciles, ignore_index=True), staging / "monthly_decile_response.parquet", files)
        _write(pd.concat(decomposition, ignore_index=True), staging / "monthly_score_relationship_decomposition.parquet", files)
        _write(pd.concat(population, ignore_index=True), staging / "monthly_population_composition.parquet", files)
        _write(pd.concat(distribution, ignore_index=True), staging / "monthly_input_distribution_drift.parquet", files)
        _write(pd.DataFrame(adversarial), staging / "monthly_adversarial_population_drift.parquet", files)
        _write(pd.concat(refit, ignore_index=True), staging / "monthly_recent_refit_score_stability.parquet", files)
        manifest = {
            "schema": SCHEMA, "status": "COMPLETE_READ_ONLY_DIAGNOSTIC", "files": files,
            "source_sha256": source_hashes,
            "score_contract": "frozen direct strict-OOF R3 contrast P(clear)-P(adverse)",
            "relationship_targets": ["resolved R3 clear event", "exact TP6/SL4 net bps"],
            "population_reference": "original side/transport strict-OOF fit-era rows strictly before each evaluated month",
            "recent_refit": {
                "enabled": True,
                "labels": "prior resolved strict OOF R3 class only",
                "full_original_label_refit": False,
                "blocked_empty_tp6_label_partitions": list(EMPTY_ORIGINAL_TP6_LABEL_PARTITIONS),
                "reason": "five empty TP6 label partitions prevent a complete original-label refit",
            },
            "no_hpo": True, "no_model_promotion": True, "transports": list(selected_transports),
        }
        manifest_path = staging / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        files[manifest_path.name] = _sha256(manifest_path)
        staging.rename(destination)
        return destination
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--winner", type=Path, default=DEFAULT_WINNER)
    parser.add_argument("--robust", type=Path, default=DEFAULT_ROBUST)
    parser.add_argument("--lineage", type=Path, default=DEFAULT_LINEAGE)
    parser.add_argument("--oof-root", type=Path, default=DEFAULT_OOF_ROOT)
    parser.add_argument("--transport", action="append", choices=tuple(TRANSPORTS))
    parser.add_argument("--min-drift-rows", type=int, default=500)
    parser.add_argument("--max-drift-rows", type=int, default=75_000)
    parser.add_argument("--min-refit-rows", type=int, default=5_000)
    parser.add_argument("--recent-refit-days", type=int, default=180)
    args = parser.parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
