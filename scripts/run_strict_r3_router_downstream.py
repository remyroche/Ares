#!/usr/bin/env python3
"""Strict-OOF two-head downstream replay for retained economic routers.

Research only.  This adapter makes the downstream contract explicit:

    strict-OOF enhanced base + P3/P4 router score + frozen causal 120 fields
      -> P3/P4 exact timestamp-local top-30 route
      -> retrained T6 (cap80 ordinary) / T9 (cap120 equal-month) heads
      -> Current/BCF score families
      -> independently refitted strict-prequential MC1 maps
      -> dual admission and the existing chronological portfolio mirror.

P3/P4 has routing authority only: it determines which timestamp-local 30% of
candidates reach the consensus layer.  The actual enhanced base score and its
three causal components remain the score coordinates consumed by consensus,
mapping, MC1 and the auction.  Policy outcomes are joined only after the
target-free Current/BCF score panels have been written.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_strict_r3_enhanced_base_live_stack_challenger as parent  # noqa: E402


SCHEMA = "strict_r3_economic_router_downstream_v1"
SEED = 1729
DEFAULT_ROUTE_FRACTION = 0.30
WARMUP_MONTHS = ("2026-01",)
EVALUATION_MONTHS = tuple(pd.date_range("2026-02-01", "2026-07-01", freq="MS", tz="UTC"))
ALL_SCORE_MONTHS = tuple(pd.Timestamp(f"{token}-01", tz="UTC") for token in (*WARMUP_MONTHS, *(f"{m:%Y-%m}" for m in EVALUATION_MONTHS)))
# The first January score bundle trains on the preceding November--December
# router OOF panels.  Those two months are score-source support only: they
# never receive a downstream in-sample score or enter final evaluation.
SOURCE_MONTHS = tuple(pd.date_range("2025-11-01", "2026-07-01", freq="MS", tz="UTC"))
ROUTER_PREFIX = ("candidate_id", "__decision_ts__", "side_name")
SOURCE_PREFIX = (
    "candidate_id", "__decision_ts__", "base_bps", "efficiency_bps", "timing_bps",
    "enhanced_base_bps", "base_rank_ts", "enhanced_base_routed", "e_minus_t",
    "e_minus_b0", "t_minus_b0", "base_component_std", "side_name",
)
POLICY_COLUMNS = (
    "candidate_id", "policy_path_valid", "policy_gross_bps", "policy_net_bps",
    "policy_exit_bar_15m", "policy_entry_price", "policy_exit_price",
    "policy_exit_reason", "policy_label_available_ts", "policy_cost_bps",
)
PROHIBITED_TARGET_FREE = frozenset({
    "policy_path_valid", "policy_gross_bps", "policy_net_bps", "policy_exit_bar_15m",
    "policy_entry_price", "policy_exit_price", "policy_exit_reason",
    "policy_label_available_ts", "policy_cost_bps", "semantic_path_valid",
    "semantic_sequence", "semantic_speed_bin", "semantic_persistence_bin",
    "semantic_pre_adverse_bin", "semantic_policy_conversion_bin", "semantic_exit_reason",
    "semantic_composite", "semantic_tbm_event",
})
RETAINED_HEADS = ("cap80_ordinary", "cap120_equal_month")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    for item in paths:
        digest.update(str(item).encode())
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _progress(out: Path, **payload: object) -> None:
    """Append a low-cost immutable progress receipt for long offline folds."""
    with (out / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _month_end(month: pd.Timestamp) -> pd.Timestamp:
    return month + pd.offsets.MonthBegin(1)


def _month_tokens(root: Path) -> list[str]:
    tokens: set[str] = set()
    for path in root.glob("month=*"):
        if path.is_dir():
            tokens.add(path.name.split("=", 1)[1])
        elif path.is_file() and path.suffix == ".parquet":
            tokens.add(path.stem.split("=", 1)[1])
    return sorted(tokens)


def _source_base_fields(source_path: Path) -> tuple[str, ...]:
    names = tuple(pq.ParquetFile(source_path).schema_arrow.names)
    if names[:len(SOURCE_PREFIX)] != SOURCE_PREFIX:
        raise AssertionError(f"unexpected target-free feature source prefix: {names[:len(SOURCE_PREFIX)]}")
    fields = names[len(SOURCE_PREFIX):]
    if len(fields) != 120 or len(set(fields)) != len(fields):
        raise AssertionError(f"expected ordered 120-field contract, found {len(fields)}")
    # The frozen selected-head receipt validates exactly this JSON-order hash.
    parent.load_conditional_consensus_contract(fields, side="long")
    return fields


def _read_target_free(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    names = set(pq.ParquetFile(path).schema_arrow.names)
    missing = sorted(set(columns) - names)
    if missing:
        raise AssertionError(f"{path}: missing columns {missing[:5]}")
    leaked = sorted(PROHIBITED_TARGET_FREE.intersection(names))
    if leaked:
        raise AssertionError(f"{path}: target-free input has outcome/path fields {leaked}")
    return pd.read_parquet(path, columns=list(columns))


def _timestamp_rank(frame: pd.DataFrame, field: str) -> np.ndarray:
    work = frame.loc[:, ["candidate_id", "__decision_ts__", field]].copy()
    work["__pos__"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(["__decision_ts__", field, "candidate_id"], ascending=[True, True, True], kind="stable")
    ordinal = work.groupby("__decision_ts__", sort=False).cumcount().to_numpy(float)
    count = work.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy(float)
    work["__rank__"] = (ordinal + .5) / count
    return work.sort_values("__pos__", kind="stable")["__rank__"].to_numpy(np.float32)


def _materialize_target_free(
    router_root: Path, source_root: Path, out: Path, route_fraction: float,
) -> tuple[Path, tuple[str, ...], pd.DataFrame]:
    target = out / "target_free_monthly"
    target.mkdir(parents=True)
    source_months = _month_tokens(source_root / "target_free_monthly")
    router_months = _month_tokens(router_root / "target_free_scores")
    required = [f"{month:%Y-%m}" for month in SOURCE_MONTHS]
    if sorted(required) != sorted(set(required)):
        raise AssertionError("duplicate downstream month declaration")
    missing_source = sorted(set(required) - set(source_months))
    missing_router = sorted(set(required) - set(router_months))
    if missing_source or missing_router:
        raise FileNotFoundError(
            f"required target-free months unavailable: source={missing_source}, router={missing_router}"
        )
    exemplar = source_root / "target_free_monthly" / f"month={required[0]}" / "scores_features.parquet"
    base_fields = _source_base_fields(exemplar)
    coverage: list[dict[str, object]] = []
    for token in required:
        source_path = source_root / "target_free_monthly" / f"month={token}" / "scores_features.parquet"
        router_path = router_root / "target_free_scores" / f"month={token}.parquet"
        source = _read_target_free(source_path, (*SOURCE_PREFIX, *base_fields))
        router = _read_target_free(router_path, (*ROUTER_PREFIX, "router_primary_rank"))
        for frame in (source, router):
            frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
        if source["candidate_id"].duplicated().any() or router["candidate_id"].duplicated().any():
            raise AssertionError(f"{token}: duplicate candidate IDs")
        keys = ["candidate_id", "__decision_ts__", "side_name"]
        merged = source.loc[:, list(SOURCE_PREFIX) + list(base_fields)].merge(
            router, on=keys, how="inner", validate="one_to_one",
        )
        if len(merged) != len(source) or len(merged) != len(router):
            raise AssertionError(f"{token}: router/source identity intersection is not exact")
        if not merged["side_name"].astype(str).str.lower().eq("long").all():
            raise AssertionError(f"{token}: non-long row entered long-only downstream source")
        router_score = pd.to_numeric(merged["router_primary_rank"], errors="coerce").to_numpy(float)
        if not np.isfinite(router_score).all():
            raise AssertionError(f"{token}: router primary rank is non-finite")
        # P3/P4 is a router, not a substitute base model.  Retain the actual
        # strict-OOF enhanced base and its component scores from the source
        # panel.  Recompute its timestamp rank deterministically on this
        # exact candidate universe, then overwrite only the *route* flag
        # using the router's own score.
        actual_base = pd.to_numeric(merged["enhanced_base_bps"], errors="coerce").to_numpy(float)
        if not np.isfinite(actual_base).all():
            raise AssertionError(f"{token}: enhanced base score is non-finite")
        merged["base_rank_ts"] = _timestamp_rank(merged, "enhanced_base_bps")
        merged["enhanced_base_routed"] = parent._exact_timestamp_top_fraction(
            merged, "router_primary_rank", route_fraction,
        ).to_numpy(bool)
        components = merged.loc[:, ["base_bps", "efficiency_bps", "timing_bps"]].apply(
            pd.to_numeric, errors="coerce",
        ).to_numpy(float)
        if not np.isfinite(components).all():
            raise AssertionError(f"{token}: enhanced base components are non-finite")
        merged["e_minus_t"] = (components[:, 1] - components[:, 2]).astype(np.float32)
        merged["e_minus_b0"] = (components[:, 1] - components[:, 0]).astype(np.float32)
        merged["t_minus_b0"] = (components[:, 2] - components[:, 0]).astype(np.float32)
        merged["base_component_std"] = np.nanstd(components, axis=1).astype(np.float32)
        # The router score is intentionally *not* persisted into the routed
        # consensus source.  Its only authority ends once it has formed the
        # timestamp-local boolean gate above.  Retaining the numeric rank here
        # would make a later broad numeric-feature selection able to recreate
        # the invalid "router as base" architecture by accident.
        ordered = [
            "candidate_id", "__decision_ts__", "side_name",
            "enhanced_base_bps", "base_rank_ts", "enhanced_base_routed", "base_bps",
            "efficiency_bps", "timing_bps", "e_minus_t", "e_minus_b0", "t_minus_b0",
            "base_component_std", *base_fields,
        ]
        full_rows = len(merged)
        # The exact configured base route is the computational boundary in the
        # deployed design.  Candidates outside it can never reach a
        # consensus, MC1, admission, or auction calculation, so retaining
        # them here only increases memory and creates a non-live-equivalent
        # MC1 population.  The gate was formed without outcomes above.
        result = merged.loc[merged["enhanced_base_routed"], ordered].copy()
        if result.empty:
            raise AssertionError(f"{token}: router top-{route_fraction:.0%} route is empty")
        if "router_primary_rank" in result:
            raise AssertionError(f"{token}: router score escaped its routing-only boundary")
        leaked = PROHIBITED_TARGET_FREE.intersection(result.columns)
        if leaked:
            raise AssertionError(f"{token}: materialized source leaked labels {sorted(leaked)}")
        destination = target / f"month={token}"
        destination.mkdir()
        result.to_parquet(destination / "scores_features.parquet", index=False, compression="zstd")
        coverage.append({
            "month": token, "source_rows": int(full_rows), "rows": int(len(result)),
            "base_feature_complete_fraction": float(result.loc[:, list(base_fields)].notna().all(axis=1).mean()),
            "routed_rows": int(result["enhanced_base_routed"].sum()),
            "router_source_identity_exact": True,
            "router_has_routing_authority_only": True,
            "actual_enhanced_base_preserved": True,
        })
    audit = pd.DataFrame(coverage)
    if audit["base_feature_complete_fraction"].lt(.90).any():
        raise AssertionError("downstream source fails the 90% frozen-feature coverage gate")
    audit.to_parquet(out / "target_free_materialization_audit.parquet", index=False, compression="zstd")
    return target, base_fields, audit


def _load_policy(path: Path) -> pd.DataFrame:
    policy = pd.read_parquet(path, columns=list(POLICY_COLUMNS))
    policy["policy_label_available_ts"] = pd.to_datetime(policy["policy_label_available_ts"], utc=True, errors="coerce")
    if policy["candidate_id"].duplicated().any():
        raise AssertionError("canonical policy label ledger has duplicate candidate IDs")
    return policy


def _restrict_policy_to_source(policy: pd.DataFrame, target_free: Path) -> pd.DataFrame:
    """Retain only router-routed identities before expensive fold work.

    The canonical label ledger is intentionally universe-wide.  Holding it in
    memory during a routed-only replay needlessly triples the resident set and
    has no causal benefit: no non-routed row can enter a downstream fit.
    """
    ids: list[pd.Series] = []
    for path in sorted(target_free.glob("month=*/scores_features.parquet")):
        ids.append(pd.read_parquet(path, columns=["candidate_id"])["candidate_id"].astype(str))
    keep = pd.Index(pd.concat(ids, ignore_index=True).unique())
    result = policy.loc[policy["candidate_id"].astype(str).isin(keep)].copy()
    del ids, keep
    gc.collect()
    if result.empty or result["candidate_id"].duplicated().any():
        raise AssertionError("routed policy restriction failed")
    return result


def _score_router_folds(
    root: Path,
    policy: pd.DataFrame,
    base_fields: tuple[str, ...],
    label_spec: parent.PolicyConversionLabelSpec,
    out: Path,
    n_jobs: int,
) -> pd.DataFrame:
    original_specs = parent._head_specs
    original_fit = parent._fit_heads
    original_train_months = parent.META_TRAIN_MONTHS
    original_reserve_days = parent.RESERVE_DAYS
    try:
        def two_head_specs(fields: tuple[str, ...], feature_contract: str = "current") -> tuple[parent.ConsensusHeadSpec, ...]:
            available = original_specs(fields, feature_contract)
            selected = tuple(item for item in available if item.name in RETAINED_HEADS)
            if tuple(item.name for item in selected) != RETAINED_HEADS:
                raise AssertionError(f"frozen T6/T9 slots missing: {[item.name for item in available]}")
            return selected

        def parallel_fit(*args: object, **kwargs: object):
            kwargs["n_jobs"] = int(n_jobs)
            return original_fit(*args, **kwargs)

        parent._head_specs = two_head_specs
        parent._fit_heads = parallel_fit
        # Preserve the parent’s same-model 28-day CDF reference: it is
        # target-free, but must be scored by the exact held-month head bundle.
        # Four calendar months yields three full months of supervised support
        # plus that predeclared reference reserve.  February is explicitly a
        # cold-start exception because the retained router ledger begins in
        # November; March onward has the full three-month support window.
        parent.META_TRAIN_MONTHS = 4
        parent.RESERVE_DAYS = 28
        audits: list[dict[str, object]] = []
        for month in ALL_SCORE_MONTHS:
            _progress(out, stage="consensus_fold_start", month=f"{month:%Y-%m}")
            audit, _, _ = parent._score_fold(
                root, policy, base_fields, label_spec,
                "base_consensus_correctness", "none",
                parent.BPS_INTEGRATION_SPECS["rank_75_25"], "current",
                month, out, trust_arm="generic_correctness",
            )
            audit["selected_heads"] = list(RETAINED_HEADS)
            audits.append(audit)
            _progress(out, stage="consensus_fold_done", month=f"{month:%Y-%m}", held_rows=int(audit["held_rows"]))
        result = pd.DataFrame(audits)
        result.to_parquet(out / "consensus_fold_audit.parquet", index=False, compression="zstd")
        return result
    finally:
        parent._head_specs = original_specs
        parent._fit_heads = original_fit
        parent.META_TRAIN_MONTHS = original_train_months
        parent.RESERVE_DAYS = original_reserve_days


def _read_scores_for_mc1(score_root: Path, family: str, policy: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "candidate_id", "__decision_ts__", "side_name", "enhanced_base_routed",
        "final_score", "base_rank42", "conditional_consensus_rank",
        "ordinary_shadow_consensus_rank", "correctness_rank", "upstream",
    ]
    pieces = []
    for path in sorted((score_root / "target_free_scores" / family).glob("*.parquet")):
        pieces.append(pd.read_parquet(path, columns=columns))
    if not pieces:
        raise FileNotFoundError(f"no {family} target-free score panels under {score_root}")
    result = pd.concat(pieces, ignore_index=True)
    result["__decision_ts__"] = pd.to_datetime(result["__decision_ts__"], utc=True, errors="raise")
    if result["candidate_id"].duplicated().any():
        raise AssertionError(f"{family}: duplicated target-free candidate IDs")
    return result.merge(policy, on="candidate_id", how="left", validate="one_to_one")


def _score_mc1(
    score_root: Path,
    out: Path,
    policy: pd.DataFrame,
    evaluation_months: tuple[pd.Timestamp, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    original_months = parent.SCORE_MONTHS
    original_train_months = parent.MC1_TRAIN_MONTHS
    try:
        parent.SCORE_MONTHS = ALL_SCORE_MONTHS
        parent.MC1_TRAIN_MONTHS = 3
        _progress(out, stage="mc1_current_start")
        current = _read_scores_for_mc1(score_root, "current", policy)
        current_pred, current_audit = parent._mc1_predictions(current, "current", out)
        current_keep = [
            "candidate_id", "__decision_ts__", "side_name", "enhanced_base_routed",
            "final_score", "mc1_expected_bps", "policy_path_valid", "policy_gross_bps",
            "policy_net_bps", "policy_exit_bar_15m", "policy_entry_price", "policy_exit_price",
            "policy_exit_reason", "policy_label_available_ts", "policy_cost_bps",
        ]
        current_pred = current_pred.loc[:, current_keep].copy()
        del current
        gc.collect()
        _progress(out, stage="mc1_current_done", rows=int(len(current_pred)))
        _progress(out, stage="mc1_bcf_start")
        bcf = _read_scores_for_mc1(score_root, "bcf", policy)
        bcf_pred, bcf_audit = parent._mc1_predictions(bcf, "bcf", out)
        bcf_pred = bcf_pred.loc[:, ["candidate_id", "__decision_ts__", "final_score", "mc1_expected_bps"]].copy()
        del bcf
        gc.collect()
        _progress(out, stage="mc1_bcf_done", rows=int(len(bcf_pred)))
    finally:
        parent.SCORE_MONTHS = original_months
        parent.MC1_TRAIN_MONTHS = original_train_months
    start, end = min(evaluation_months), _month_end(max(evaluation_months))
    current_pred = current_pred.loc[current_pred["__decision_ts__"].ge(start) & current_pred["__decision_ts__"].lt(end)].copy()
    bcf_pred = bcf_pred.loc[bcf_pred["__decision_ts__"].ge(start) & bcf_pred["__decision_ts__"].lt(end)].copy()
    combined = parent._combined_challenger(current_pred, bcf_pred)
    pd.concat([current_audit, bcf_audit], ignore_index=True).to_parquet(out / "mc1_fit_audit.parquet", index=False, compression="zstd")
    combined.to_parquet(out / "dual_mc1_predictions.parquet", index=False, compression="zstd")
    return combined, current_audit, bcf_audit


def _timestamp_score_metrics(frame: pd.DataFrame, score_fields: Sequence[str], *, scope: str) -> pd.DataFrame:
    valid = frame.loc[
        frame["enhanced_base_routed"].fillna(False).astype(bool)
        & frame["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce")),
    ].copy()
    records: list[dict[str, object]] = []
    for field in score_fields:
        for fraction in (.01, .02, .03, .05, .10):
            values: list[float] = []
            counts: list[int] = []
            for _, part in valid.groupby("__decision_ts__", sort=False):
                n = max(1, int(np.ceil(len(part) * fraction)))
                selected = part.nlargest(n, field, keep="first")
                values.append(float(pd.to_numeric(selected["policy_net_bps"], errors="coerce").mean()))
                counts.append(len(selected))
            records.append({
                "scope": scope, "score_field": field, "fraction": fraction,
                "timestamp_mean_net_ev_bps": float(np.mean(values)) if values else np.nan,
                "timestamp_worst_net_ev_bps": float(np.min(values)) if values else np.nan,
                "timestamps": int(len(values)), "selected_rows": int(sum(counts)),
            })
    return pd.DataFrame(records)


def _per_head_metrics(score_root: Path, out: Path, policy: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for family in ("current", "bcf"):
        panel = parent._read_score_panels(score_root, family, policy)
        fields = ["final_score", "conditional_consensus_rank", "ordinary_shadow_consensus_rank"]
        fields.extend(sorted(field for field in panel.columns if field.startswith("head__") and field.endswith("__rank")))
        frames.append(_timestamp_score_metrics(panel, fields, scope=family))
    result = pd.concat(frames, ignore_index=True)
    result.to_parquet(out / "per_head_timestamp_metrics.parquet", index=False, compression="zstd")
    return result


def _portfolio_metrics(frame: pd.DataFrame, out: Path, thresholds: Sequence[float]) -> pd.DataFrame:
    results: list[dict[str, object]] = []
    original_threshold = parent.MC1_THRESHOLD_BPS
    try:
        for threshold in thresholds:
            parent.MC1_THRESHOLD_BPS = float(threshold)
            label = f"router_dual_{int(threshold)}"
            result = parent._portfolio_metrics(frame, label, "2026_febjul", out)
            result["threshold_bps"] = float(threshold)
            results.append(result)
    finally:
        parent.MC1_THRESHOLD_BPS = original_threshold
    metrics = pd.DataFrame(results)
    metrics.to_parquet(out / "portfolio_metrics.parquet", index=False, compression="zstd")
    return metrics


def _audit(out: Path, target_free: Path, score_root: Path, folds: pd.DataFrame, combined: pd.DataFrame) -> dict[str, object]:
    score_checks = []
    for family in ("current", "bcf"):
        for path in sorted((score_root / "target_free_scores" / family).glob("*.parquet")):
            names = set(pq.ParquetFile(path).schema_arrow.names)
            leaked = sorted(PROHIBITED_TARGET_FREE.intersection(names))
            score_checks.append({"family": family, "path": str(path), "leaked": leaked, "rows": int(pq.ParquetFile(path).metadata.num_rows)})
            if leaked:
                raise AssertionError(f"{path}: outcome leaked into target-free score output")
    if not folds["selected_heads"].map(lambda value: tuple(value) == RETAINED_HEADS).all():
        raise AssertionError("downstream fold did not use exactly T6/T9")
    if combined["candidate_id"].duplicated().any():
        raise AssertionError("dual MC1 combined output has duplicate candidate identities")
    if combined[["current_final_score", "bcf_final_score", "current_mc1_expected_bps", "bcf_mc1_expected_bps"]].isna().any().any():
        raise AssertionError("Current/BCF MC1 combination has a missing score or expected-EV coordinate")
    result = {
        "schema": SCHEMA,
        "target_free_source": str(target_free),
        "target_free_scores_have_no_outcomes": True,
        "consensus_head_contract": list(RETAINED_HEADS),
        "consensus_training": "three prior calendar months plus a same-model target-free 28-day CDF reserve; February is documented cold start",
        "mc1_training": "up to three prior scored calendar months, each policy label available before held month",
        "dual_identity_exact": True,
        "score_checks": score_checks,
    }
    (out / "correctness_report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run(
    *, router_root: Path, source_root: Path, policy_path: Path, out: Path,
    policy_label: str, thresholds: Sequence[float], n_jobs: int, route_fraction: float,
    resume_score_root: Path | None = None,
    resume_mc1_root: Path | None = None,
) -> None:
    if out.exists():
        raise FileExistsError(out)
    if policy_label not in parent.POLICY_CONVERSION_LABEL_SPECS:
        raise ValueError(f"unknown policy conversion label: {policy_label}")
    out.mkdir(parents=True)
    if resume_score_root is None:
        _progress(out, stage="materialization_start")
        target_free, base_fields, materialization = _materialize_target_free(
            router_root, source_root, out, route_fraction,
        )
        _progress(out, stage="materialization_done", rows=int(materialization["rows"].sum()))
    else:
        score_root = resume_score_root.resolve()
        target_free = score_root / "target_free_monthly"
        materialization = pd.read_parquet(score_root / "target_free_materialization_audit.parquet")
        # No consensus fit occurs on a resumed score root, so the original
        # source-field prefix validation is already sealed by that root’s
        # materialisation receipt.  Avoid reinterpreting its richer adapter
        # prefix as the raw upstream source schema.
        base_fields = ()
        _progress(out, stage="reuse_target_free_scores", score_root=str(score_root))
    policy = _load_policy(policy_path)
    policy = _restrict_policy_to_source(policy, target_free)
    if resume_score_root is None:
        folds = _score_router_folds(
            target_free, policy, base_fields,
            parent.POLICY_CONVERSION_LABEL_SPECS[policy_label], out, n_jobs,
        )
        score_root = out
    else:
        folds = pd.read_parquet(score_root / "consensus_fold_audit.parquet")
        _progress(out, stage="reuse_consensus_scores", folds=int(len(folds)))
    if resume_mc1_root is None:
        combined, current_audit, bcf_audit = _score_mc1(score_root, out, policy, EVALUATION_MONTHS)
    else:
        mc1_root = resume_mc1_root.resolve()
        combined = pd.read_parquet(mc1_root / "dual_mc1_predictions.parquet")
        combined["__decision_ts__"] = pd.to_datetime(combined["__decision_ts__"], utc=True, errors="raise")
        start, end = min(EVALUATION_MONTHS), _month_end(max(EVALUATION_MONTHS))
        combined = combined.loc[combined["__decision_ts__"].ge(start) & combined["__decision_ts__"].lt(end)].copy()
        current_audit = bcf_audit = pd.DataFrame()
        _progress(out, stage="reuse_mc1_predictions", mc1_root=str(mc1_root), rows=int(len(combined)))
    _per_head_metrics(score_root, out, policy)
    _timestamp_score_metrics(combined, ("current_final_score", "bcf_final_score"), scope="dual").to_parquet(
        out / "score_timestamp_metrics.parquet", index=False, compression="zstd",
    )
    portfolio = _portfolio_metrics(combined, out, thresholds)
    _progress(out, stage="portfolio_done", rows=int(len(portfolio)))
    audit = _audit(out, target_free, score_root, folds, combined)
    manifest = {
        "schema": SCHEMA,
        "scope": "offline research only; no live configuration, order, or bundle changed",
        "router_root": str(router_root), "source_root": str(source_root), "resume_score_root": str(resume_score_root) if resume_score_root else None,
        "resume_mc1_root": str(resume_mc1_root) if resume_mc1_root else None,
        "policy_path": str(policy_path), "policy_label": policy_label,
        "route": f"router-primary exact timestamp-local top {route_fraction:.0%}",
        "route_fraction": float(route_fraction),
        "base_router_separation": "P3/P4 controls only enhanced_base_routed; enhanced_base_bps, base_bps, efficiency_bps and timing_bps are preserved from the strict-OOF enhanced-base source",
        "consensus_heads": list(RETAINED_HEADS), "consensus_train_months": 3,
        "consensus_reference_reserve_days": 28,
        "mc1_train_months_max": 3, "evaluation_months": [f"{m:%Y-%m}" for m in EVALUATION_MONTHS],
        "thresholds_bps": list(map(float, thresholds)), "n_jobs_per_head": int(n_jobs),
        "source_hashes": {"router": _sha256(router_root), "source": _sha256(source_root), "policy": _sha256(policy_path)},
        "rows": {"materialized": int(materialization["rows"].sum()), "portfolio_metric_rows": int(len(portfolio))},
        "correctness": audit,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy-label", default="direct_policy_economic_200_0_50_150", choices=tuple(parent.POLICY_CONVERSION_LABEL_SPECS))
    parser.add_argument("--thresholds", default="30,50", help="comma-separated dual MC1 expected-EV thresholds")
    parser.add_argument("--route-fraction", type=float, default=DEFAULT_ROUTE_FRACTION)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--resume-score-root", type=Path, default=None, help="completed router target-free/consensus root to reuse for MC1/reporting only")
    parser.add_argument("--resume-mc1-root", type=Path, default=None, help="completed MC1 root to reuse for reporting/portfolio only")
    args = parser.parse_args()
    if args.n_jobs < 1:
        parser.error("--n-jobs must be positive")
    if not 0 < args.route_fraction <= 1:
        parser.error("--route-fraction must lie in (0, 1]")
    thresholds = tuple(float(value) for value in args.thresholds.split(",") if value)
    if not thresholds:
        parser.error("at least one threshold is required")
    run(
        router_root=args.router_root, source_root=args.source_root, policy_path=args.policy_path,
        out=args.out, policy_label=args.policy_label, thresholds=thresholds, n_jobs=args.n_jobs,
        route_fraction=args.route_fraction,
        resume_score_root=args.resume_score_root, resume_mc1_root=args.resume_mc1_root,
    )


if __name__ == "__main__":
    main()
