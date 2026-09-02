#!/usr/bin/env python3
"""Strict-OOF first funnel for orthogonal enhanced-base meta supervision.

This research-only runner keeps five physical LambdaRank head slots, but tests
whether policy/path semantics improve a *correction* layer rather than adding a
second base scoring stack.  Semantic outcomes are used only after label
resolution to build targets and bounded training weights.  Persisted score
panels contain no outcomes or semantic fields.

Arms
----
O0_direct_policy
    Direct policy-net ordinal control on the differentiated five slots.
O3_calibrated_residual_semantic
    Fold-local isotonic base calibration; rank the clipped policy residual,
    with bounded archetype/certainty/hard-base-error weights.
O5_base_rank_error_semantic
    Rank the realised-minus-base timestamp rank error, with the same strictly
    training-only semantic weighting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_strict_r3_enhanced_base_live_stack_challenger as parent  # noqa: E402


SCHEMA = "strict_r3_orthogonal_meta_label_funnel_v1"
SEED = 1729
ROUTE = 0.30
TRAIN_MONTHS = 6
RESERVE_DAYS = 28
HEAD_CAP = 120_000
SCORE_MONTHS = tuple(pd.date_range("2025-07-01", "2026-07-01", freq="MS", tz="UTC"))
ARMS = ("O0_direct_policy", "O3_calibrated_residual_semantic", "O5_base_rank_error_semantic")

GEOMETRY_FIELDS = (
    "om_base_rank", "om_efficiency_rank", "om_timing_rank",
    "om_coordinate_min", "om_coordinate_max", "om_coordinate_median",
    "om_coordinate_std", "om_coordinate_range", "e_minus_t", "e_minus_b0",
    "t_minus_b0", "base_component_std",
)
QUERY_FIELDS = (
    "om_query_routed_count", "om_query_base_std", "om_query_base_range",
    "om_query_base_rank", "om_query_top_gap", "om_query_top2_gap",
)
OOD_FIELDS = ("om_score_ood_l1", "om_score_support_proxy")


@dataclass(frozen=True)
class HeadSpec:
    name: str
    fields: tuple[str, ...]
    params: dict[str, object]
    equal_month: bool


@dataclass
class FittedHead:
    spec: HeadSpec
    medians: np.ndarray
    model: LGBMRanker
    reference: np.ndarray

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        raw = self.model.predict(_matrix(frame, self.spec.fields, self.medians)).astype(np.float32)
        rank = _cdf(self.reference, raw).astype(np.float32)
        return raw, rank


def _utc(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _month_end(month: pd.Timestamp) -> pd.Timestamp:
    return month + pd.offsets.MonthBegin(1)


def _cdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    left = np.searchsorted(reference, values, side="left")
    right = np.searchsorted(reference, values, side="right")
    return np.clip((left + right + 1.0) / (2.0 * len(reference)), 0.0, 1.0)


def _matrix(frame: pd.DataFrame, fields: Sequence[str], medians: np.ndarray) -> np.ndarray:
    values = frame.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    return np.where(np.isfinite(values), values, medians[None, :]).astype(np.float32)


def _medians(frame: pd.DataFrame, fields: Sequence[str]) -> np.ndarray:
    return (
        frame.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce")
        .median(axis=0).fillna(0.0).to_numpy(np.float32)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*.parquet")) if path.is_dir() else [path]:
        digest.update(str(child).encode())
        with child.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _source_months(root: Path) -> list[pd.Timestamp]:
    values = []
    for directory in sorted(root.glob("month=*")):
        values.append(pd.Timestamp(f"{directory.name.split('=', 1)[1]}-01", tz="UTC"))
    return values


def _load_window(
    feature_root: Path,
    semantic_root: Path,
    policy: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    columns: Sequence[str],
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for month in pd.period_range(start.to_period("M"), (end - pd.Timedelta(nanoseconds=1)).to_period("M"), freq="M"):
        token = month.strftime("%Y-%m")
        source = feature_root / f"month={token}" / "scores_features.parquet"
        semantic = semantic_root / "parts" / f"month={token}" / "semantics.parquet"
        if not source.exists() or not semantic.exists():
            continue
        raw = pd.read_parquet(source, columns=list(columns))
        raw["__decision_ts__"] = pd.to_datetime(raw["__decision_ts__"], utc=True, errors="raise")
        raw = raw.loc[raw["__decision_ts__"].ge(start) & raw["__decision_ts__"].lt(end)].copy()
        labels = pd.read_parquet(semantic)
        raw = raw.merge(labels, on=["candidate_id", "__decision_ts__"], how="left", validate="one_to_one")
        pieces.append(raw)
    if not pieces:
        return pd.DataFrame(columns=list(columns))
    output = pd.concat(pieces, ignore_index=True)
    output = output.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    return output


def _augment_geometry(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for source, target in (
        ("base_bps", "om_base_rank"),
        ("efficiency_bps", "om_efficiency_rank"),
        ("timing_bps", "om_timing_rank"),
        ("enhanced_base_bps", "om_query_base_rank"),
    ):
        out[target] = out.groupby("__decision_ts__", sort=False)[source].rank(pct=True, method="average")
    coordinate = out.loc[:, ["base_bps", "efficiency_bps", "timing_bps"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    out["om_coordinate_min"] = np.nanmin(coordinate, axis=1)
    out["om_coordinate_max"] = np.nanmax(coordinate, axis=1)
    out["om_coordinate_median"] = np.nanmedian(coordinate, axis=1)
    out["om_coordinate_std"] = np.nanstd(coordinate, axis=1)
    out["om_coordinate_range"] = out["om_coordinate_max"] - out["om_coordinate_min"]
    routed = out["enhanced_base_routed"].fillna(False).astype(bool)
    work = out.loc[routed, ["__decision_ts__", "enhanced_base_bps"]].copy()
    grouped = work.groupby("__decision_ts__", sort=False)["enhanced_base_bps"]
    summary = pd.DataFrame({
        "om_query_routed_count": grouped.size(),
        "om_query_base_std": grouped.std().fillna(0.0),
        "om_query_base_range": grouped.max() - grouped.min(),
    })
    ordered = work.sort_values(["__decision_ts__", "enhanced_base_bps"], ascending=[True, False], kind="stable")
    ordered["__next__"] = ordered.groupby("__decision_ts__", sort=False)["enhanced_base_bps"].shift(-1)
    ordered["__third__"] = ordered.groupby("__decision_ts__", sort=False)["enhanced_base_bps"].shift(-2)
    top = ordered.groupby("__decision_ts__", sort=False).first()
    summary["om_query_top_gap"] = top["enhanced_base_bps"] - top["__next__"]
    summary["om_query_top2_gap"] = top["__next__"] - top["__third__"]
    for field in summary:
        out[field] = out["__decision_ts__"].map(summary[field]).fillna(0.0).astype(np.float32)
    return out


def _fit_ood(train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    values = train.loc[:, ["base_bps", "efficiency_bps", "timing_bps", "e_minus_t", "e_minus_b0", "t_minus_b0"]].apply(pd.to_numeric, errors="coerce")
    median = values.median().fillna(0.0).to_numpy(float)
    mad = (values - pd.Series(median, index=values.columns)).abs().median().to_numpy(float) * 1.4826
    fallback = values.std().replace(0.0, np.nan).fillna(1.0).to_numpy(float)
    scale = np.where(np.isfinite(mad) & (mad > 1e-8), mad, fallback)
    return median, np.maximum(scale, 1e-6)


def _apply_ood(frame: pd.DataFrame, median: np.ndarray, scale: np.ndarray) -> pd.DataFrame:
    out = frame.copy()
    values = out.loc[:, ["base_bps", "efficiency_bps", "timing_bps", "e_minus_t", "e_minus_b0", "t_minus_b0"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    z = np.clip(np.abs((values - median[None, :]) / scale[None, :]), 0.0, 10.0)
    l1 = np.nanmean(z, axis=1)
    out["om_score_ood_l1"] = np.nan_to_num(l1, nan=10.0).astype(np.float32)
    out["om_score_support_proxy"] = np.exp(-out["om_score_ood_l1"].to_numpy(float)).astype(np.float32)
    return out


def _head_specs(base_fields: Sequence[str]) -> tuple[HeadSpec, ...]:
    paths = parent.Paths(
        raw_ledger=Path("unused"), direct_root=Path("unused"), policy_root=Path("unused"),
        current_mc1=Path("unused"), bcf_mc1=Path("unused"), bundle_root=Path("unused"),
    )
    del paths
    # The canonical feature memberships/parameters are loaded through the
    # same frozen selection contract.  Only the slot assignment differs.
    selected = parent._head_specs(tuple(base_fields))
    raw100 = tuple(field for field in selected[0].fields if field in base_fields)
    raw60 = tuple(field for field in selected[-1].fields if field in base_fields)
    state = tuple(field for field in base_fields if any(token in field for token in ("state_", "regime", "spectral", "volatility")))[:32]
    support = tuple(field for field in base_fields if any(token in field for token in ("spread", "depth", "liquidity", "amihud", "oi_", "volume")))[:42]
    slots = (
        ("h1_raw_residual", (*raw100, *GEOMETRY_FIELDS), selected[0]),
        ("h2_base_query_geometry", (*GEOMETRY_FIELDS, *QUERY_FIELDS), selected[1]),
        ("h3_state_transition", (*GEOMETRY_FIELDS, *state), selected[2]),
        ("h4_support_ood_disagreement", (*GEOMETRY_FIELDS, *QUERY_FIELDS, *OOD_FIELDS, *support), selected[3]),
        ("h5_compact_raw_control", (*raw60, *GEOMETRY_FIELDS), selected[4]),
    )
    result = []
    for name, fields, reference in slots:
        fields = tuple(dict.fromkeys(fields))
        result.append(HeadSpec(
            name=name,
            fields=fields,
            params=dict(reference.params),
            equal_month=reference.weight_mode == "equal_month",
        ))
    return tuple(result)


def _sample_queries(frame: pd.DataFrame, *, cap: int, seed: int, equal_month: bool) -> pd.DataFrame:
    out = frame.copy()
    out["__query__"] = out["__decision_ts__"].astype(str) + "|long"
    out["__month__"] = out["__decision_ts__"].dt.to_period("M").astype(str)
    counts = out.groupby("__query__", sort=False)["candidate_id"].transform("size")
    out = out.loc[counts.ge(2)].copy()
    if len(out) <= cap:
        return out.sort_values(["__query__", "candidate_id"], kind="stable")
    meta = out.groupby("__query__", sort=False).agg(rows=("candidate_id", "size"), month=("__month__", "first")).reset_index()
    meta["__hash__"] = meta["__query__"].map(lambda value: int(hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()[:16], 16))
    keep: list[str] = []
    if equal_month:
        allowance = max(2, cap // max(1, meta["month"].nunique()))
        groups: Iterable[tuple[object, pd.DataFrame]] = meta.groupby("month", sort=True)
    else:
        allowance = cap
        groups = [("all", meta)]
    for _, part in groups:
        used = 0
        for query, rows, _month, _hash in part.sort_values(
            ["__hash__", "__query__"], kind="stable"
        ).loc[:, ["__query__", "rows", "month", "__hash__"]].itertuples(index=False, name=None):
            if used + int(rows) <= allowance:
                keep.append(str(query))
                used += int(rows)
    return out.loc[out["__query__"].isin(keep)].sort_values(["__query__", "candidate_id"], kind="stable")


def _semantic_weights(train: pd.DataFrame) -> np.ndarray:
    composite = train["semantic_composite"].astype(str).replace("<NA>", "invalid")
    counts = composite.value_counts()
    archetype = composite.map(lambda value: math.sqrt(len(train) / max(float(counts.loc[value]), 1.0))).to_numpy(float)
    event = train["semantic_tbm_event"].astype(str)
    certainty = np.where(event.eq("ambiguous"), 0.50, np.where(event.eq("vertical"), 0.75, 1.0))
    realised_rank = train.groupby("__decision_ts__", sort=False)["policy_net_bps"].rank(pct=True, method="average")
    hard_error = np.abs(realised_rank.to_numpy(float) - train["base_rank_ts"].to_numpy(float))
    hard = 1.0 + np.where(hard_error >= 0.25, 0.75, np.where(hard_error >= 0.10, 0.25, 0.0))
    weight = archetype * certainty * hard
    weight /= max(float(np.mean(weight)), 1e-12)
    return np.clip(weight, 0.25, 4.0).astype(np.float32)


def _targets(train: pd.DataFrame, arm: str) -> tuple[np.ndarray, dict[str, float]]:
    policy = pd.to_numeric(train["policy_net_bps"], errors="coerce").to_numpy(float)
    if arm == "O0_direct_policy":
        grade = np.digitize(policy, (-100.0, -30.0, 30.0, 90.0)).astype(np.int8)
        return grade, {"target_mean": float(np.mean(policy)), "target_std": float(np.std(policy))}
    if arm == "O3_calibrated_residual_semantic":
        mapping = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(train["base_rank_ts"], policy)
        residual = np.clip(policy - mapping.predict(train["base_rank_ts"]), -500.0, 500.0)
        grade = np.digitize(residual, (-200.0, -75.0, 75.0, 200.0)).astype(np.int8)
        return grade, {"target_mean": float(np.mean(residual)), "target_std": float(np.std(residual))}
    if arm == "O5_base_rank_error_semantic":
        realised = train.groupby("__decision_ts__", sort=False)["policy_net_bps"].rank(pct=True, method="average").to_numpy(float)
        error = realised - train["base_rank_ts"].to_numpy(float)
        grade = np.digitize(error, (-0.25, -0.08, 0.08, 0.25)).astype(np.int8)
        return grade, {"target_mean": float(np.mean(error)), "target_std": float(np.std(error))}
    raise ValueError(arm)


def _fit_head(train: pd.DataFrame, target: np.ndarray, weights: np.ndarray, spec: HeadSpec, seed: int) -> FittedHead:
    sampled = _sample_queries(train, cap=HEAD_CAP, seed=seed, equal_month=spec.equal_month)
    target_by_id = pd.Series(target, index=train["candidate_id"].astype(str))
    weight_by_id = pd.Series(weights, index=train["candidate_id"].astype(str))
    target_sample = target_by_id.loc[sampled["candidate_id"].astype(str)].to_numpy(np.int8)
    weights_sample = weight_by_id.loc[sampled["candidate_id"].astype(str)].to_numpy(np.float32)
    if spec.equal_month:
        counts = sampled["__month__"].value_counts()
        weights_sample *= sampled["__month__"].map(lambda value: 1.0 / counts.loc[value]).to_numpy(float)
        weights_sample *= len(weights_sample) / max(float(np.sum(weights_sample)), 1e-12)
    group = sampled.groupby("__query__", sort=False).size().to_numpy(np.int32)
    medians = _medians(sampled, spec.fields)
    params = dict(spec.params)
    params.update(random_state=seed, n_jobs=1, deterministic=True, force_col_wise=True, verbosity=-1)
    model = LGBMRanker(**params).fit(
        _matrix(sampled, spec.fields, medians), target_sample,
        group=group, sample_weight=weights_sample,
    )
    raw = model.predict(_matrix(sampled, spec.fields, medians))
    return FittedHead(spec, medians, model, np.sort(raw.astype(float), kind="stable"))


def _score_heads(frame: pd.DataFrame, heads: Sequence[FittedHead]) -> pd.DataFrame:
    out = frame.loc[:, ["candidate_id", "__decision_ts__", "side_name", "enhanced_base_routed", "enhanced_base_bps", "base_rank_ts"]].copy()
    ranks = []
    for head in heads:
        raw, rank = head.predict(frame)
        out[f"{head.spec.name}_raw"] = raw
        out[f"{head.spec.name}_rank"] = rank
        ranks.append(rank)
    matrix = np.column_stack(ranks)
    out["orthogonal_consensus_rank"] = np.nanmedian(matrix, axis=1).astype(np.float32)
    out["orthogonal_head_rank_std"] = np.nanstd(matrix, axis=1).astype(np.float32)
    return out


def _metrics(scored: pd.DataFrame, labels: pd.DataFrame, arm: str, month: pd.Timestamp) -> list[dict[str, object]]:
    joined = scored.merge(labels, on="candidate_id", how="left", validate="one_to_one")
    valid = joined["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(pd.to_numeric(joined["policy_net_bps"], errors="coerce"))
    work = joined.loc[valid].copy()
    result: list[dict[str, object]] = []
    score_fields = ["orthogonal_consensus_rank", *[field for field in scored if field.endswith("_rank") and field.startswith("h")]]
    policy = work["policy_net_bps"].to_numpy(float)
    base = work["base_rank_ts"].to_numpy(float)
    for field in score_fields:
        score = pd.to_numeric(work[field], errors="coerce").to_numpy(float)
        good = np.isfinite(score) & np.isfinite(policy)
        rho = float(spearmanr(score[good], policy[good]).statistic) if good.sum() > 10 else np.nan
        corr_base = float(spearmanr(score[good], base[good]).statistic) if good.sum() > 10 else np.nan
        for tail in (0.01, 0.02, 0.05, 0.10):
            threshold = float(np.quantile(score[good], 1.0 - tail, method="higher"))
            selected = policy[good & (score >= threshold)]
            result.append({
                "arm": arm, "month": f"{month:%Y-%m}", "score": field, "tail": tail,
                "rows": int(len(selected)), "net_ev_bps_per_trade": float(np.mean(selected)),
                "net_sum_bps": float(np.sum(selected)), "policy_rank_ic": rho,
                "head_base_rank_corr": corr_base,
            })
    return result


def run(
    *,
    feature_root: Path,
    semantic_root: Path,
    policy_path: Path,
    bundle_root: Path,
    out: Path,
    score_months: Sequence[pd.Timestamp] = SCORE_MONTHS,
) -> None:
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    policy = pd.read_parquet(policy_path, columns=["candidate_id", "policy_path_valid", "policy_net_bps", "policy_label_available_ts"])
    policy["policy_label_available_ts"] = pd.to_datetime(policy["policy_label_available_ts"], utc=True, errors="coerce")
    if policy["candidate_id"].duplicated().any():
        raise AssertionError("canonical policy labels duplicate candidate IDs")
    # Base contract is deliberately read from the current bundle rather than
    # inferred from source columns.
    paths = parent.Paths(Path("unused"), Path("unused"), Path("unused"), Path("unused"), Path("unused"), bundle_root)
    base_fields = parent._base_fields(paths)
    specs = _head_specs(base_fields)
    required = tuple(dict.fromkeys((
        "candidate_id", "__decision_ts__", "side_name", "enhanced_base_routed",
        "enhanced_base_bps", "base_rank_ts", "base_bps", "efficiency_bps", "timing_bps",
        "e_minus_t", "e_minus_b0", "t_minus_b0", "base_component_std", *base_fields,
    )))
    all_metrics: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    for arm_index, arm in enumerate(ARMS):
        arm_root = out / "target_free_scores" / arm
        arm_root.mkdir(parents=True)
        for month in score_months:
            end = _month_end(month)
            reserve_start = month - pd.Timedelta(days=RESERVE_DAYS)
            train_start = month - pd.DateOffset(months=TRAIN_MONTHS)
            train = _load_window(feature_root, semantic_root, policy, start=train_start, end=reserve_start, columns=required)
            held = _load_window(feature_root, semantic_root, policy, start=month, end=end, columns=required)
            train = _augment_geometry(train)
            held = _augment_geometry(held)
            valid_train = (
                train["enhanced_base_routed"].fillna(False).astype(bool)
                & train["semantic_path_valid"].fillna(False).astype(bool)
                & train["policy_path_valid"].fillna(False).astype(bool)
                & train["semantic_label_available_ts"].lt(reserve_start)
                & train["policy_label_available_ts"].lt(reserve_start)
                & np.isfinite(pd.to_numeric(train["policy_net_bps"], errors="coerce"))
            )
            train = train.loc[valid_train].copy()
            held = held.loc[held["enhanced_base_routed"].fillna(False).astype(bool)].copy()
            if len(train) < 5_000 or len(held) < 1_000:
                raise AssertionError(f"{arm} {month:%Y-%m}: insufficient strict OOF support")
            median, scale = _fit_ood(train)
            train = _apply_ood(train, median, scale)
            held = _apply_ood(held, median, scale)
            grade, target_stats = _targets(train, arm)
            weights = np.ones(len(train), dtype=np.float32) if arm == "O0_direct_policy" else _semantic_weights(train)
            heads = tuple(_fit_head(train, grade, weights, spec, SEED + arm_index * 1000 + index) for index, spec in enumerate(specs))
            scored = _score_heads(held, heads)
            score_path = arm_root / f"month={month:%Y-%m}.parquet"
            scored.to_parquet(score_path, index=False, compression="zstd")
            labels = held.loc[:, ["candidate_id", "policy_path_valid", "policy_net_bps"]].copy()
            all_metrics.extend(_metrics(scored, labels, arm, month))
            audit.append({
                "arm": arm, "month": f"{month:%Y-%m}", "train_rows": int(len(train)),
                "held_rows": int(len(held)), "semantic_weighted": arm != "O0_direct_policy",
                "semantic_valid_fraction_train": float(train["semantic_path_valid"].mean()),
                "target_unique_grades": int(np.unique(grade).size),
                **target_stats,
            })
            print(json.dumps({"event": "scored", **audit[-1]}), flush=True)
    metrics = pd.DataFrame(all_metrics)
    metrics.to_parquet(out / "oof_label_funnel_metrics.parquet", index=False, compression="zstd")
    pd.DataFrame(audit).to_parquet(out / "oof_label_funnel_audit.parquet", index=False, compression="zstd")
    # Enforce that every persisted score panel is target-free.  The original
    # labels remain only in local training data and metrics joins.
    prohibited = {"policy_net_bps", "policy_path_valid", "semantic_composite", "semantic_tbm_event"}
    probe = pd.read_parquet(next((out / "target_free_scores" / ARMS[0]).glob("*.parquet")))
    leaked = prohibited.intersection(probe.columns)
    if leaked:
        raise AssertionError(f"target-free score receipt leaked labels: {sorted(leaked)}")
    manifest = {
        "schema": SCHEMA,
        "scope": "offline research only; no live stack mutation",
        "arms": list(ARMS),
        "feature_root": str(feature_root),
        "semantic_root": str(semantic_root),
        "policy_path": str(policy_path),
        "bundle_root": str(bundle_root),
        "source_hashes": {
            "feature_root": _sha256(feature_root),
            "semantic_root": _sha256(semantic_root),
            "policy_path": _sha256(policy_path),
        },
        "causality": {
            "base": "pre-existing strict-OOF enhanced B0/efficiency/timing score",
            "semantics": "post-resolution training target/weight only; absent from target-free scores",
            "purge": "prior 28-day reserve excluded from head fit",
            "labels": "canonical rich policy net and path/policy semantics resolved before reserve start",
            "held_scores": "five head slots predict target-free held rows only",
            "mc1": "not fitted in this first funnel; only a passing candidate may be promoted to experimental MC1",
        },
        "head_slots": [{"name": spec.name, "fields": list(spec.fields), "equal_month": spec.equal_month, "params": spec.params} for spec in specs],
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--semantic-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--months", help="optional YYYY-MM list for an immutable smoke receipt")
    args = parser.parse_args()
    score_months = (
        tuple(pd.Timestamp(f"{token}-01", tz="UTC") for token in args.months.split(","))
        if args.months else SCORE_MONTHS
    )
    run(
        feature_root=args.feature_root,
        semantic_root=args.semantic_root,
        policy_path=args.policy_path,
        bundle_root=args.bundle_root,
        out=args.out,
        score_months=score_months,
    )


if __name__ == "__main__":
    main()
