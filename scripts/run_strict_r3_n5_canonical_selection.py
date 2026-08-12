#!/usr/bin/env python3
"""Select and confirm the canonical strict-R3 N5 sizing overlay.

The command is a sequential funnel:

1. 2025-only chronological, month-balanced conditional MDA;
2. 2025-only joint mean/risk target and forest HPO with median pruning;
3. full 2025 development replay of the frozen winner;
4. one untouched 2026 confirmation replay;
5. matched causal-admission portfolio and risk-adjusted wallet diagnostics.

Ranking and causal EV admission remain frozen throughout.  N5 only changes
relative position size after admission.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.n5_forest_support_sizing import (  # noqa: E402
    BASELINE_N5_PARAMS,
    N5ForestParams,
    fit_n5_forest,
    n5_hpo_candidates,
)
from extreme_price_movements.trust_sizing_ablation import (  # noqa: E402
    ParentExpectation,
    discover_cmi_edges,
    trust_feature_family,
)
from extreme_price_movements.stage_i_causal_admission import (  # noqa: E402
    Causal21dAdmissionSpec,
    apply_causal_21d_side_admission,
)
from scripts.replay_strict_r3_forward_portfolio import _auction_candidates  # noqa: E402
from scripts.replay_strict_r3_policy_portfolio_2025_2026 import _run  # noqa: E402
from scripts.run_strict_r3_c3_window_cadence_ablation import _causal_reliability_context  # noqa: E402
from scripts.run_strict_r3_trust_sizing_ablation import (  # noqa: E402
    INPUTS,
    PERIODS,
    _blocks,
    _load,
    _period_tail_metrics,
    _portfolio,
    _sample_equal_month,
    _stability,
)


SCHEMA = "strict_r3_n5_portable_selection_hpo_v2"
SEED = 20260810
TAILS = (0.01, 0.02, 0.05)
FEATURE_GROUP_CONFIG = ROOT / "config/strict_r3_n5_feature_groups_v2.json"
CANONICAL_N5_CONTRACT = ROOT / "config/strict_r3_ldf_support_v3.json"
OLD_COMPACT_12 = (
    "consensus_rank", "correctness_raw", "correctness_rank",
    "reliability_recent_3d_mean_residual_bps",
    "reliability_recent_3d_adverse100_rate",
    "reliability_base_consensus_gap",
    "reliability_recent_3d_positive_rate",
    "reliability_base_consensus_mean", "reliability_upstream_rank",
    "k9_entropy", "k9_model_ood_marginal", "reliability_recent_7d_support",
)

# The historical trust-sizing loader retains several execution diagnostics that
# are not consumed by the LDF MDA/HPO funnel.  Keeping the lean selection
# population avoids a second multi-gigabyte dataframe when the stronger
# 101-field sidecar is joined.
SELECTION_PRIMARY_COLUMNS = (
    "candidate_id", "__decision_ts__", "__symbol__", "side_name",
    "base_score", "base_rank", "base_anchor_bps", "consensus_rank", "final_score",
    "correctness_raw", "correctness_rank",
    "k9_entropy", "k9_top2_margin", "k9_ood_distance",
    "k9_path_support_effective_28d", "k9_path_support_adequate_fraction",
    "k9_model_ood_marginal", "k9_model_drift_psi",
    "leaf_support_effective", "leaf_support_p05", "leaf_support_p50",
    "leaf_support_p95", "leaf_ood_marginal", "leaf_ood_joint",
    "geometry_bundle_sha256", "policy_path_valid", "policy_gross_bps",
    "policy_net_bps", "policy_exit_reason", "policy_label_available_ts",
)


def _feature_group_config() -> dict[str, Any]:
    payload = json.loads(FEATURE_GROUP_CONFIG.read_text())
    if payload.get("schema") not in {
        "strict_r3_n5_semantic_feature_groups_v2",
        "strict_r3_n5_semantic_feature_groups_v3_active_rule",
        "strict_r3_n5_semantic_feature_groups_v3_k9_weighted",
        "strict_r3_n5_semantic_feature_groups_schema_v2_additive_v1",
    }:
        raise ValueError("unexpected N5 semantic feature-group schema")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_primary_for_selection(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    available = set(pq.ParquetFile(path).schema.names)
    columns = [field for field in SELECTION_PRIMARY_COLUMNS if field in available]
    missing = sorted(set(SELECTION_PRIMARY_COLUMNS).difference(columns))
    if missing:
        raise ValueError(f"strict-R3 primary ledger lacks selection columns: {missing}")
    frame = pd.read_parquet(path, columns=columns)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
    frame["policy_label_available_ts"] = pd.to_datetime(
        frame["policy_label_available_ts"], utc=True, errors="coerce",
    )
    frame = frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if frame["candidate_id"].duplicated().any():
        raise ValueError("strict-R3 primary selection ledger has duplicate candidate IDs")
    admitted, admission_audit = apply_causal_21d_side_admission(
        frame,
        score_column="final_score", net_column="policy_net_bps",
        decision_column="__decision_ts__", label_available_column="policy_label_available_ts",
        identity_column="candidate_id",
        spec=Causal21dAdmissionSpec(mode="hierarchical_tail_side_shrinkage_v2"),
    )
    mapped = pd.to_numeric(admitted["causal_21d_side_expected_net_bps"], errors="coerce")
    admitted["raw_expected_bps"] = mapped
    admitted["mapped_ev_available"] = mapped.notna()
    context, _groups = _causal_reliability_context(admitted)
    context.index = admitted.index
    admitted = pd.concat([admitted, context], axis=1)
    return admitted, {
        "source_rows": len(frame),
        "mapped_ev_available_rows": int(admitted["mapped_ev_available"].sum()),
        "admission_rows": len(admission_audit),
        "primary_columns": list(columns),
        "raw_k9_memberships_used": False,
    }


def _load_selection_input(
    path: Path,
    *,
    feature_sidecar: Path | None,
    feature_contract: Path | None,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    if feature_sidecar is None:
        frame, default_fields, audit = _load(path)
        return frame, default_fields, audit
    frame, audit = _load_primary_for_selection(path)
    default_fields: list[str] = []
    if feature_contract is None:
        raise ValueError("a feature sidecar requires --feature-contract")
    contract = json.loads(feature_contract.read_text())
    fields = list(dict.fromkeys(map(str, contract["features"])))
    sidecar_schema = set(pq.ParquetFile(feature_sidecar).schema.names)
    missing = sorted(set(fields).difference(sidecar_schema))
    if missing:
        raise ValueError(f"N5 feature sidecar lacks contract fields: {missing}")
    # Read only new model fields plus the small immutable score-lineage guard.
    # Full 100-field sidecars otherwise create a needless duplicate 1m-row
    # dataframe before the chronological folds subsample it.
    lineage_fields = [
        field for field in (
            "base_score", "base_rank", "base_anchor_bps", "consensus_rank",
            "final_score", "correctness_raw", "correctness_rank",
        ) if field in fields and field in frame.columns
    ]
    enrich = [field for field in fields if field not in frame.columns]
    sidecar = pd.read_parquet(
        feature_sidecar, columns=["candidate_id", *lineage_fields, *enrich],
    )
    if sidecar["candidate_id"].duplicated().any():
        raise ValueError("N5 feature sidecar contains duplicate candidate IDs")
    if len(sidecar) != len(frame) or not np.array_equal(
        sidecar["candidate_id"].to_numpy(), frame["candidate_id"].to_numpy(),
    ):
        raise ValueError(
            "feature sidecar is not in identical candidate-id order to the strict-R3 "
            "primary ledger; refuse an order-dependent enrichment join"
        )
    # The primary strict-R3 ledger owns all upstream scores.  A sidecar may
    # enrich it with target-free context, never replace its score lineage.
    # Check any overlapping values explicitly, then merge only absent fields.
    if lineage_fields:
        for field in lineage_fields:
            primary = pd.to_numeric(frame[field], errors="coerce").to_numpy(float)
            candidate = pd.to_numeric(sidecar[field], errors="coerce").to_numpy(float)
            if not np.allclose(primary, candidate, rtol=0.0, atol=1e-8, equal_nan=True):
                raise ValueError(
                    f"feature sidecar attempts to replace strict-R3 source field: {field}"
                )
    if enrich:
        for field in enrich:
            frame[field] = sidecar[field].to_numpy(copy=False)
    coverage = frame.loc[:, fields].apply(pd.to_numeric, errors="coerce").notna().mean()
    variance = frame.loc[:, fields].apply(pd.to_numeric, errors="coerce").var()
    failed = [
        field for field in fields
        if coverage[field] < 0.90 or not np.isfinite(variance[field]) or variance[field] <= 1e-12
    ]
    if failed:
        raise ValueError(f"challenger feature coverage/variance gate failed: {failed}")
    return frame, fields, {
        **audit,
        "feature_sidecar": str(feature_sidecar),
        "feature_sidecar_sha256": _sha256(feature_sidecar),
        "feature_contract": str(feature_contract),
        "feature_contract_sha256": _sha256(feature_contract),
        "eligible_features": len(fields),
        "eligible_feature_names": fields,
    }


def _sampling_frame(frame: pd.DataFrame, *, cap: int, seed: int) -> pd.DataFrame:
    """Equal-month subsample preserving the score surface and chronology."""

    if len(frame) <= cap:
        return frame.copy()
    work = frame.copy()
    month = work["__decision_ts__"].dt.to_period("M").astype(str)
    months = sorted(month.unique())
    quota = max(1, cap // max(len(months), 1))
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for token in months:
        index = np.flatnonzero(month.eq(token).to_numpy())
        # Stratify by the frozen score decile so every tail remains represented.
        score = pd.to_numeric(work.iloc[index]["final_score"], errors="coerce")
        decile = pd.qcut(score.rank(method="first"), 10, labels=False, duplicates="drop")
        local: list[np.ndarray] = []
        per_decile = max(1, quota // max(int(decile.nunique()), 1))
        for value in sorted(decile.dropna().unique()):
            positions = index[np.flatnonzero(decile.eq(value).to_numpy())]
            if len(positions) > per_decile:
                positions = np.sort(rng.choice(positions, per_decile, replace=False))
            local.append(positions)
        selected.append(np.concatenate(local))
    index = np.concatenate(selected)
    if len(index) > cap:
        index = np.sort(rng.choice(index, cap, replace=False))
    return work.iloc[index].sort_values(["__decision_ts__", "candidate_id"], kind="stable")


def _iter_fold_data(
    frame: pd.DataFrame,
    *,
    year: int,
    train_cap: int,
    held_cap: int | None,
    block_months: int = 3,
) -> Iterable[dict[str, Any]]:
    """Yield one chronological fold at a time.

    Full-universe replay must not retain every three-month held block and its
    full feature matrix simultaneously.  Selection callers can materialise the
    small list explicitly through ``_fold_data``; replay callers stream it.
    """
    if block_months == 3:
        blocks = _blocks(year)
    elif block_months == 1:
        start, end = PERIODS[year]
        blocks = []
        cutoff = start
        while cutoff < end:
            held_end = min(cutoff + pd.DateOffset(months=1), end)
            blocks.append((cutoff, held_end))
            cutoff = held_end
    else:
        raise ValueError("MDA blocks must be one or three months")
    for fold_index, (cutoff, held_end) in enumerate(blocks):
        train_start = cutoff - pd.DateOffset(months=3)
        train_all = frame.loc[
            frame["__decision_ts__"].ge(train_start)
            & frame["__decision_ts__"].lt(cutoff)
            & frame["policy_label_available_ts"].lt(cutoff)
            & frame["policy_path_valid"].fillna(False).astype(bool)
            & frame["mapped_ev_available"].astype(bool)
            & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
        ].copy()
        held = frame.loc[
            frame["__decision_ts__"].ge(cutoff) & frame["__decision_ts__"].lt(held_end)
        ].copy()
        parent = ParentExpectation.fit(train_all["final_score"], train_all["policy_net_bps"])
        train_all["parent_expected_bps"] = parent.predict(train_all["final_score"])
        held["parent_expected_bps"] = parent.predict(held["final_score"])
        train_floor = float(pd.to_numeric(train_all["final_score"], errors="coerce").quantile(0.70))
        train = train_all.loc[pd.to_numeric(train_all["final_score"], errors="coerce").ge(train_floor)].copy()
        train = _sample_equal_month(train, int(train_cap))
        held["trust_gate_active"] = (
            held["mapped_ev_available"].astype(bool)
            & pd.to_numeric(held["final_score"], errors="coerce").ge(train_floor)
        )
        if held_cap is not None:
            held = _sampling_frame(held, cap=int(held_cap), seed=SEED + 100 + fold_index)
        yield {
            "fold": fold_index,
            "train_start": train_start,
            "cutoff": cutoff,
            "held_end": held_end,
            "train": train,
            "held": held,
            "train_floor": train_floor,
        }


def _fold_data(
    frame: pd.DataFrame,
    *,
    year: int,
    train_cap: int,
    held_cap: int | None,
    block_months: int = 3,
) -> list[dict[str, Any]]:
    """Materialise folds for MDA/HPO, whose sampled frames are bounded."""

    return list(
        _iter_fold_data(
            frame,
            year=year,
            train_cap=train_cap,
            held_cap=held_cap,
            block_months=block_months,
        )
    )


def _edges(train: pd.DataFrame, fields: Sequence[str]) -> list[Any]:
    source = train.loc[
        pd.to_numeric(train["final_score"], errors="coerce").ge(
            pd.to_numeric(train["final_score"], errors="coerce").quantile(0.80)
        )
    ].copy()
    edges, _ = discover_cmi_edges(
        source,
        fields,
        mode="rank_loss",
        stable=True,
        max_edges=8,
        sample_cap=30_000,
    )
    return edges


def _output(
    held: pd.DataFrame,
    prediction: Any,
    multiplier: np.ndarray,
    *,
    arm: str,
) -> pd.DataFrame:
    result = held.loc[
        :,
        [
            "candidate_id", "__decision_ts__", "__symbol__", "final_score",
            "policy_path_valid", "policy_gross_bps", "policy_net_bps",
            "policy_exit_reason", "geometry_bundle_sha256", "raw_expected_bps",
            "parent_expected_bps", "trust_gate_active",
        ],
    ].copy()
    pred = prediction.as_frame()
    pred.index = result.index
    result = pd.concat([result, pred], axis=1)
    result["trust_size_multiplier"] = np.where(
        result["trust_gate_active"].to_numpy(bool), multiplier, 1.0,
    ).astype(np.float32)
    result["arm"] = arm
    return result.reset_index(drop=True)


def _objective(output: pd.DataFrame, *, arm: str) -> tuple[float, dict[str, Any]]:
    global_metrics = _period_tail_metrics(output, arm=arm, period_kind="global")
    monthly = _period_tail_metrics(output, arm=arm, period_kind="month")
    stability = _stability(monthly)
    g = global_metrics.set_index("tail")["exposure_weighted_net_bps"]
    tail_score = float(g.get(0.01, np.nan) + 0.5 * g.get(0.02, np.nan) + 0.2 * g.get(0.05, np.nan))
    s = stability.loc[stability["tail"].isin(TAILS)]
    portability = float(s["portability"].mean())
    worst = float(s["worst_month_bps"].min())
    score = tail_score + 0.25 * portability - max(0.0, -worst)
    return score, {
        "selection_score": score,
        "weighted_tail_score": tail_score,
        "mean_portability_top1_2_5": portability,
        "worst_month_top1_2_5": worst,
        "top1_net_bps": float(g.get(0.01, np.nan)),
        "top2_net_bps": float(g.get(0.02, np.nan)),
        "top5_net_bps": float(g.get(0.05, np.nan)),
    }


def _mda_group(field: str, config: dict[str, Any] | None = None) -> str:
    """Resolve one field to exactly one checked-in semantic group."""

    payload = config or _feature_group_config()
    text = str(field)
    matches: list[str] = []
    for spec in payload["groups"]:
        if (
            text in set(map(str, spec.get("exact", ())))
            or any(text.startswith(str(prefix)) for prefix in spec.get("prefixes", ()))
            or any(str(token) in text for token in spec.get("contains", ()))
        ):
            matches.append(str(spec["id"]))
    if len(matches) != 1:
        raise ValueError(f"feature {field!r} maps to {matches}, expected exactly one group")
    return matches[0]


def _conditional_strata(
    held: pd.DataFrame,
    *,
    group: str,
    config: dict[str, Any],
) -> np.ndarray:
    rules = config["conditional_permutation"]
    month = held["__decision_ts__"].dt.to_period("M").astype(str)
    consensus = pd.to_numeric(held["consensus_rank"], errors="coerce").fillna(0.5)
    consensus_decile = pd.qcut(
        consensus.rank(method="first"), 10, labels=False, duplicates="drop",
    ).fillna(-1).astype(int).astype(str)
    pieces = [month, consensus_decile]
    if group in set(rules["bundle_groups"]):
        pieces.append(held["geometry_bundle_sha256"].fillna("missing").astype(str))
    if group in set(rules["support_groups"]):
        support_field = next(
            (
                field for field in (
                    "leaf_support_effective",
                    "k9_cluster_timestamp_support_weighted",
                    "k9_cluster_weighted_fit_support",
                ) if field in held.columns
            ),
            None,
        )
        if support_field is None:
            raise KeyError("support-conditioned MDA needs a declared causal support field")
        support = pd.to_numeric(held[support_field], errors="coerce")
        support_bin = pd.qcut(
            support.rank(method="first"), 5, labels=False, duplicates="drop",
        ).fillna(-1).astype(int).astype(str)
        pieces.append(support_bin)
    if group in set(rules["market_state_groups"]):
        state_field = next(
            (
                field for field in (
                    "k9_entropy",
                    "k9_cluster_weighted_ood",
                    "k9_cluster_timestamp_ood_weighted",
                ) if field in held.columns
            ),
            None,
        )
        if state_field is None:
            raise KeyError("market-state-conditioned MDA needs a declared causal state field")
        entropy = pd.to_numeric(held[state_field], errors="coerce")
        state_bin = pd.qcut(
            entropy.rank(method="first"), 4, labels=False, duplicates="drop",
        ).fillna(-1).astype(int).astype(str)
        pieces.append(state_bin)
    return np.asarray(
        ["|".join(values) for values in zip(*(piece.to_numpy() for piece in pieces))],
        dtype=object,
    )


def _permute_feature_frame(
    held: pd.DataFrame,
    fields: Sequence[str],
    strata: np.ndarray,
    rng: np.random.Generator,
    *,
    joint: bool,
) -> pd.DataFrame:
    permuted = held.copy()
    if joint:
        source = permuted.loc[:, list(fields)].to_numpy(copy=True)
        shuffled = source.copy()
        for stratum in np.unique(strata):
            index = np.flatnonzero(strata == stratum)
            shuffled[index, :] = source[rng.permutation(index), :]
        permuted.loc[:, list(fields)] = shuffled
    else:
        for field in fields:
            values = permuted[field].to_numpy(copy=True)
            shuffled = values.copy()
            for stratum in np.unique(strata):
                index = np.flatnonzero(strata == stratum)
                shuffled[index] = values[rng.permutation(index)]
            permuted[field] = shuffled
    return permuted


def _permuted_output(
    held: pd.DataFrame,
    bundle: Any,
    fields: Sequence[str],
    strata: np.ndarray,
    rng: np.random.Generator,
    *,
    joint: bool,
) -> pd.DataFrame:
    permuted = _permute_feature_frame(
        held, fields, strata, rng, joint=joint,
    )
    prediction, multiplier = bundle.size_multiplier(permuted)
    return _output(held, prediction, multiplier, arm="permuted")


def _portable_summary(
    detail: pd.DataFrame,
    keys: Sequence[str],
    folds: int,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for identity, block in detail.groupby(list(keys), sort=True):
        identity = identity if isinstance(identity, tuple) else (identity,)
        fold_values = block.groupby("fold")["mda_loss"].mean()
        median = float(np.median(fold_values))
        mad = float(np.median(np.abs(fold_values - median)))
        row = dict(zip(keys, identity, strict=True))
        portability = (config or _feature_group_config())["portability"]
        worst = float(fold_values.min())
        row.update(
            mda_median=median,
            mda_mad=mad,
            mda_worst_fold=worst,
            positive_fold_recurrence=float((fold_values > 0.0).mean()),
            portable_mda_score=(
                median
                - float(portability["lambda_mad"]) * mad
                - float(portability["gamma_negative_worst"]) * max(0.0, -worst)
            ),
            fold_count=int(folds),
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["portable_mda_score", "positive_fold_recurrence", "mda_median"],
        ascending=False, kind="stable",
    )


def _mda(
    folds: Sequence[dict[str, Any]],
    fields: Sequence[str],
    *,
    params: N5ForestParams,
    repeats: int,
    checkpoint_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Hierarchical MDA: semantic-family knockout, then conditional members."""

    rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    config = _feature_group_config()
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def checkpoint_path(stage: str, fold_index: int, ordinal: int) -> Path | None:
        if checkpoint_dir is None:
            return None
        return checkpoint_dir / f"{stage}_fold{fold_index:02d}_{ordinal:03d}.parquet"

    def load_or_write(path: Path | None, generated: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if path is not None:
            pd.DataFrame(generated).to_parquet(path, index=False)
        return generated

    fold_cache: list[dict[str, Any]] = []
    groups: dict[str, list[str]] = {}
    for field in fields:
        groups.setdefault(_mda_group(field, config), []).append(field)
    for fold in folds:
        train, held = fold["train"], fold["held"]
        edges = _edges(train, fields)
        bundle, _train_prediction = fit_n5_forest(train, fields, edges, params=params)
        prediction, multiplier = bundle.size_multiplier(held)
        baseline = _output(held, prediction, multiplier, arm="baseline")
        baseline_score, baseline_metrics = _objective(baseline, arm="baseline")
        fold_cache.append(
            {
                "fold": fold, "held": held, "bundle": bundle,
                "baseline_score": baseline_score,
                "baseline_metrics": baseline_metrics,
            }
        )
        for group_ordinal, (group, group_fields) in enumerate(groups.items()):
            path = checkpoint_path("group", int(fold["fold"]), group_ordinal)
            if path is not None and path.exists():
                cached = pd.read_parquet(path).to_dict("records")
                if not cached or {str(row["group"]) for row in cached} != {group}:
                    raise ValueError(f"invalid MDA group checkpoint: {path}")
                group_rows.extend(cached)
                print(json.dumps({"event": "mda_group_checkpoint_reused", "fold": fold["fold"], "group": group}), flush=True)
                continue
            local_rows: list[dict[str, Any]] = []
            strata = _conditional_strata(held, group=group, config=config)
            for repeat in range(int(repeats)):
                rng = np.random.default_rng(
                    SEED + 10_000 * int(fold["fold"]) + 100 * group_ordinal + repeat
                )
                candidate = _permuted_output(
                    held, bundle, group_fields, strata, rng, joint=True,
                )
                candidate_score, candidate_metrics = _objective(candidate, arm="permuted")
                local_rows.append(
                    {
                        "fold": fold["fold"], "cutoff": fold["cutoff"],
                        "environment_kind": "fold",
                        "environment_id": str(fold["cutoff"]),
                        "group": group, "group_field_count": len(group_fields),
                        "repeat": repeat, "baseline_score": baseline_score,
                        "permuted_score": candidate_score,
                        "mda_loss": baseline_score - candidate_score,
                        **{f"baseline_{key}": value for key, value in baseline_metrics.items()},
                        **{f"permuted_{key}": value for key, value in candidate_metrics.items()},
                    }
                )
                for bundle_id, base_bundle in baseline.groupby(
                    "geometry_bundle_sha256", sort=False,
                ):
                    if len(base_bundle) < 200:
                        continue
                    candidate_bundle = candidate.loc[
                        candidate["geometry_bundle_sha256"].astype(str).eq(str(bundle_id))
                    ]
                    if len(candidate_bundle) != len(base_bundle):
                        raise AssertionError("group MDA changed Geometry/K9 bundle support")
                    base_bundle_score, _ = _objective(base_bundle, arm="baseline")
                    candidate_bundle_score, _ = _objective(
                        candidate_bundle, arm="permuted",
                    )
                    local_rows.append(
                        {
                            "fold": fold["fold"], "cutoff": fold["cutoff"],
                            "environment_kind": "geometry_bundle",
                            "environment_id": str(bundle_id),
                            "group": group, "group_field_count": len(group_fields),
                            "repeat": repeat, "baseline_score": base_bundle_score,
                            "permuted_score": candidate_bundle_score,
                            "mda_loss": base_bundle_score - candidate_bundle_score,
                        }
                    )
            group_rows.extend(load_or_write(path, local_rows))
            print(json.dumps({"event": "mda_group_checkpoint_written", "fold": fold["fold"], "group": group, "rows": len(local_rows)}), flush=True)
    group_detail = pd.DataFrame(group_rows)
    group_summary = _portable_summary(
        group_detail.loc[group_detail["environment_kind"].eq("fold")],
        ("group",), len(folds), config,
    )
    minimum_recurrence = float(
        config["portability"]["minimum_positive_environment_fraction"]
    )
    passing_groups = set(
        group_summary.loc[
            group_summary["mda_median"].gt(0.0)
            & group_summary["positive_fold_recurrence"].ge(minimum_recurrence),
            "group",
        ].astype(str)
    )
    if not passing_groups:
        passing_groups = set(group_summary.head(2)["group"].astype(str))
    stage2_fields = [
        field for field in fields if _mda_group(field, config) in passing_groups
    ]
    for cached in fold_cache:
        fold = cached["fold"]
        held = cached["held"]
        bundle = cached["bundle"]
        baseline_score = cached["baseline_score"]
        baseline_metrics = cached["baseline_metrics"]
        for field_ordinal, field in enumerate(stage2_fields):
            group = _mda_group(field, config)
            path = checkpoint_path("field", int(fold["fold"]), field_ordinal)
            if path is not None and path.exists():
                cached = pd.read_parquet(path).to_dict("records")
                if not cached or {str(row["field"]) for row in cached} != {field}:
                    raise ValueError(f"invalid MDA field checkpoint: {path}")
                rows.extend(cached)
                print(json.dumps({"event": "mda_field_checkpoint_reused", "fold": fold["fold"], "field": field}), flush=True)
                continue
            local_rows = []
            strata = _conditional_strata(held, group=group, config=config)
            for repeat in range(int(repeats)):
                rng = np.random.default_rng(
                    SEED + 1_000_000 + 10_000 * int(fold["fold"]) + 100 * field_ordinal + repeat
                )
                candidate = _permuted_output(
                    held, bundle, [field], strata, rng, joint=False,
                )
                candidate_score, candidate_metrics = _objective(candidate, arm="permuted")
                local_rows.append(
                    {
                        "fold": fold["fold"], "cutoff": fold["cutoff"],
                        "environment_kind": "fold", "environment_id": str(fold["cutoff"]),
                        "field": field, "family": trust_feature_family(field),
                        "group": group, "repeat": repeat,
                        "baseline_score": baseline_score,
                        "permuted_score": candidate_score,
                        "mda_loss": baseline_score - candidate_score,
                        **{f"baseline_{key}": value for key, value in baseline_metrics.items()},
                        **{f"permuted_{key}": value for key, value in candidate_metrics.items()},
                    }
                )
            rows.extend(load_or_write(path, local_rows))
            print(json.dumps({"event": "mda_field_checkpoint_written", "fold": fold["fold"], "field": field, "rows": len(local_rows)}), flush=True)
    detail = pd.DataFrame(rows)
    summary = _portable_summary(
        detail, ("field", "family", "group"), len(folds), config,
    )
    selected: list[str] = []
    for group in sorted(passing_groups):
        block = summary.loc[
            summary["group"].eq(group)
            & summary["positive_fold_recurrence"].ge(minimum_recurrence)
            & summary["mda_median"].ge(0.0)
        ].head(5)
        if block.empty:
            block = summary.loc[summary["group"].eq(group)].head(1)
        selected.extend(block["field"].astype(str).tolist())
    selected = list(dict.fromkeys(selected))[:40]
    if len(selected) < 12:
        selected = summary.head(min(20, len(summary)))["field"].astype(str).tolist()
    return (
        detail.merge(summary, on=["field", "family", "group"], how="left"),
        group_detail.merge(group_summary, on="group", how="left"),
        selected,
    )


def _evaluate_contract(
    folds: Sequence[dict[str, Any]],
    fields: Sequence[str],
    *,
    params: N5ForestParams,
    arm: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    parts: list[pd.DataFrame] = []
    for fold in folds:
        edges = _edges(fold["train"], fields)
        bundle, _ = fit_n5_forest(fold["train"], fields, edges, params=params)
        prediction, multiplier = bundle.size_multiplier(fold["held"])
        parts.append(_output(fold["held"], prediction, multiplier, arm=arm))
    output = pd.concat(parts, ignore_index=True)
    _, metrics = _objective(output, arm=arm)
    return output, metrics


def _indistinguishable_from_full(
    candidate: dict[str, Any],
    full: dict[str, Any],
    tolerances: dict[str, Any],
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    checks = (
        ("selection_score", "selection_score_tolerance"),
        ("top1_net_bps", "top1_tolerance_bps"),
        ("top2_net_bps", "top2_tolerance_bps"),
        ("top5_net_bps", "top5_tolerance_bps"),
        ("worst_month_top1_2_5", "worst_month_tolerance_bps"),
    )
    for metric, tolerance in checks:
        if float(candidate[metric]) < float(full[metric]) - float(tolerances[tolerance]):
            failures.append(metric)
    return not failures, failures


def _backward_grouped_elimination(
    folds: Sequence[dict[str, Any]],
    fields: Sequence[str],
    *,
    params: N5ForestParams,
    group_mda: pd.DataFrame,
    feature_mda: pd.DataFrame,
    checkpoint_path: Path | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """Retrained full-down elimination with bounded within-group pruning."""

    config = _feature_group_config()
    tolerances = config["backward_elimination"]
    active = list(fields)
    _full_output, full_metrics = _evaluate_contract(
        folds, active, params=params, arm="backward_full_challenger",
    )
    rows: list[dict[str, Any]] = [
        {
            "stage": "full", "candidate": "full_challenger", "removed": [],
            "field_count": len(active), "accepted": True, "failed_metrics": [],
            **full_metrics,
        }
    ]

    def checkpoint() -> None:
        """Persist completed retrained removals without changing selection."""
        if checkpoint_path is not None:
            pd.DataFrame(rows).to_parquet(checkpoint_path, index=False)

    checkpoint()
    group_order = (
        group_mda.drop_duplicates("group")
        .sort_values(
            ["portable_mda_score", "positive_fold_recurrence"],
            ascending=[True, True], kind="stable",
        )["group"]
        .astype(str)
        .tolist()
    )
    for group in group_order:
        removed = [
            field for field in active
            if _mda_group(field, config) == group
        ]
        candidate_fields = [field for field in active if field not in set(removed)]
        if not removed or len(candidate_fields) < 12:
            continue
        _output_frame, metrics = _evaluate_contract(
            folds, candidate_fields, params=params, arm=f"without_{group}",
        )
        accepted, failures = _indistinguishable_from_full(
            metrics, full_metrics, tolerances,
        )
        rows.append(
            {
                "stage": "group", "candidate": f"without_{group}",
                "removed": removed, "field_count": len(candidate_fields),
                "accepted": accepted, "failed_metrics": failures, **metrics,
            }
        )
        checkpoint()
        if accepted:
            active = candidate_fields

    # Conditional MDA is only a proposal order. Every removal is retrained and
    # must remain indistinguishable from the original full challenger.
    order = (
        feature_mda.drop_duplicates("field")
        .loc[lambda value: value["field"].isin(active)]
        .sort_values(
            ["portable_mda_score", "positive_fold_recurrence"],
            ascending=[True, True], kind="stable",
        )["field"]
        .astype(str)
        .tolist()
    )[: int(tolerances["maximum_feature_trials"])]
    for field in order:
        if field not in active or len(active) <= 12:
            continue
        candidate_fields = [value for value in active if value != field]
        _output_frame, metrics = _evaluate_contract(
            folds, candidate_fields, params=params, arm=f"without_{field}",
        )
        accepted, failures = _indistinguishable_from_full(
            metrics, full_metrics, tolerances,
        )
        rows.append(
            {
                "stage": "feature", "candidate": f"without_{field}",
                "removed": [field], "field_count": len(candidate_fields),
                "accepted": accepted, "failed_metrics": failures, **metrics,
            }
        )
        checkpoint()
        if accepted:
            active = candidate_fields
    return active, pd.DataFrame(rows)


def _required_ablation_contracts(
    expanded_fields: Sequence[str],
    selected_fields: Sequence[str],
) -> dict[str, list[str]]:
    config = _feature_group_config()
    expanded = list(expanded_fields)

    def without(group: str) -> list[str]:
        return [field for field in expanded if _mda_group(field, config) != group]

    contribution = {
        "leaf_support_contribution_weighted",
        "leaf_support_contribution_weighted_log",
        "leaf_support_high_contribution_min",
        "leaf_contributor_effective_n",
    }
    canonical = list(json.loads(CANONICAL_N5_CONTRACT.read_text())["features"])
    groups = {field: _mda_group(field, config) for field in expanded}
    return {
        "canonical45_frozen_params": canonical,
        "canonical45_matched_params": canonical,
        "expanded_full": expanded,
        "expanded_without_structural_covcorr": without("F_structural_covariance_breaks"),
        "expanded_without_cluster_path_correctness": without("E_cluster_path_correctness"),
        "expanded_without_ten_head_state": without("G_ten_head_committee_state"),
        "expanded_without_contribution_support": [
            field for field in expanded if field not in contribution
        ],
        "grouped_backward_selected": list(selected_fields),
        "old_compact12": list(OLD_COMPACT_12),
        "expanded_without_ood": without("C_structural_ood"),
        "expanded_without_all_recent_correctness": [
            field for field in expanded
            if groups[field] not in {
                "D_recent_global_correctness", "E_cluster_path_correctness",
            }
        ],
        "core_plus_committee": [
            field for field in expanded
            if groups[field] in {"A_core_score_model_state", "G_ten_head_committee_state"}
        ],
        "core_plus_structural_trust": [
            field for field in expanded
            if groups[field] in {
                "A_core_score_model_state", "B_activated_support", "C_structural_ood",
                "F_structural_covariance_breaks", "H_geometry_k9_ambiguity_state",
            }
        ],
    }


def _trial(
    folds: Sequence[dict[str, Any]],
    fields: Sequence[str],
    params: N5ForestParams,
    *,
    trial: int,
    pruning_reference: Sequence[Sequence[float]],
    prune_after_folds: int,
    median_pruner_min_trials: int,
) -> tuple[pd.DataFrame, dict[str, Any], bool]:
    parts: list[pd.DataFrame] = []
    interim: list[float] = []
    pruned = False
    for step, fold in enumerate(folds):
        edges = _edges(fold["train"], fields)
        bundle, _ = fit_n5_forest(fold["train"], fields, edges, params=params)
        prediction, multiplier = bundle.size_multiplier(fold["held"])
        output = _output(fold["held"], prediction, multiplier, arm=f"trial_{trial:03d}")
        parts.append(output)
        score, _ = _objective(pd.concat(parts, ignore_index=True), arm=f"trial_{trial:03d}")
        interim.append(score)
        if step + 1 >= int(prune_after_folds):
            completed_at_step = [value[step] for value in pruning_reference if len(value) > step]
            if (
                len(completed_at_step) >= int(median_pruner_min_trials)
                and score < float(np.median(completed_at_step))
            ):
                pruned = True
                break
    combined = pd.concat(parts, ignore_index=True)
    score, metrics = _objective(combined, arm=f"trial_{trial:03d}")
    metrics.update(
        {
            "trial": trial,
            "pruned": pruned,
            "completed_folds": len(parts),
            "interim_scores": interim,
            "params": asdict(params),
        }
    )
    return combined, metrics, pruned


def _risk_metrics(equity: pd.DataFrame, decisions: pd.DataFrame, replay: dict[str, Any]) -> dict[str, Any]:
    accepted = decisions.loc[decisions["accepted"].fillna(False).astype(bool)].copy()
    if equity.empty:
        return {}
    eq = equity.copy()
    eq["timestamp"] = pd.to_datetime(eq["timestamp"], utc=True, errors="coerce")
    column = "mtm_equity" if "mtm_equity" in eq else "wallet"
    daily = (
        eq.dropna(subset=["timestamp"])
        .set_index("timestamp")[column]
        .resample("1D")
        .last()
        .ffill()
        .dropna()
    )
    returns = daily.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
    downside = returns.loc[returns < 0.0]
    daily_mean = float(returns.mean()) if len(returns) else 0.0
    daily_sd = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    downside_rms = float(np.sqrt(np.mean(downside**2))) if len(downside) else 0.0
    annualized_log = float(np.log(max(float(daily.iloc[-1]) / max(float(daily.iloc[0]), 1e-12), 1e-12)) * 365.0 / max(len(daily) - 1, 1))
    max_dd = abs(float(replay.get("max_drawdown", np.nan)))
    daily_drawdown = daily / daily.cummax() - 1.0
    ulcer = float(np.sqrt(np.mean((100.0 * daily_drawdown) ** 2)))
    pnl = pd.to_numeric(accepted.get("position_size"), errors="coerce").fillna(0.0) * pd.to_numeric(accepted.get("position_net_return"), errors="coerce").fillna(0.0)
    gains = float(pnl.loc[pnl > 0.0].sum())
    losses = float(-pnl.loc[pnl < 0.0].sum())
    compounded = float(replay.get("compounded_return", np.nan))
    return {
        "daily_sharpe_annualized": daily_mean / daily_sd * math.sqrt(365.0) if daily_sd > 0.0 else np.nan,
        "daily_sortino_annualized": daily_mean / downside_rms * math.sqrt(365.0) if downside_rms > 0.0 else np.nan,
        "daily_omega_zero": float(returns.clip(lower=0.0).sum() / max(-returns.clip(upper=0.0).sum(), 1e-12)),
        "annualized_log_return": annualized_log,
        "log_calmar": annualized_log / max(max_dd, 1e-12),
        "compounded_return_over_maxdd": compounded / max(max_dd, 1e-12),
        "ulcer_index_pct": ulcer,
        "profit_factor": gains / max(losses, 1e-12),
        "positive_days": int((returns > 0.0).sum()),
        "negative_days": int((returns < 0.0).sum()),
        "daily_observations": int(len(returns)),
    }


def _portfolio_full(
    source: pd.DataFrame,
    output: pd.DataFrame,
    *,
    arm: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    size = output.loc[:, ["candidate_id", "trust_size_multiplier"]]
    # The source frame carries the whole 101-field MDA contract.  The auction
    # needs only execution/admission columns, so copying all feature columns
    # here made a full-universe replay retain several multi-GB frames at once.
    # Keep the portfolio replay semantically identical while projecting first.
    required = (
        "candidate_id", "__decision_ts__", "__symbol__", "side_name",
        "causal_21d_side_admitted_ge_50bps",
        "causal_21d_side_expected_net_bps", "policy_path_valid",
        "policy_net_bps", "policy_gross_bps", "policy_exit_bar_15m",
        "policy_entry_price", "policy_exit_price", "policy_exit_reason",
        "policy_outcome_source",
    )
    available = [column for column in required if column in source.columns]
    missing = sorted(set(required).difference(available))
    if missing:
        raise ValueError(f"portfolio source lacks required fields: {missing}")
    mask = source["__decision_ts__"].ge(start) & source["__decision_ts__"].lt(end)
    evaluation = source.loc[mask, available].copy()
    evaluation = evaluation.merge(size, on="candidate_id", how="left", validate="one_to_one")
    evaluation["trust_size_multiplier"] = evaluation["trust_size_multiplier"].fillna(1.0)
    candidates = _auction_candidates(evaluation, strategy_prefix="strict_r3_n5_canonical")
    candidates = candidates.merge(size, on="candidate_id", how="left", validate="one_to_one")
    candidates["portfolio_size_multiplier"] = candidates["trust_size_multiplier"].fillna(1.0)
    decisions, equity, monthly, summary = _run(
        candidates,
        0.0,
        arm,
        initial_wallet=1_000.0,
        perp_leverage=7.0,
        margin_slot_wallet_fraction=0.10,
    )
    replay = summary.get("replay_metric_summary", {})
    if isinstance(replay, str):
        replay = json.loads(replay)
    risk = {**replay, **_risk_metrics(equity, decisions, replay)}
    risk.update(
        {
            "arm": arm,
            "accepted_trades": int(summary["accepted_trades"]),
            "net_bps_per_trade": float(summary["net_bps_per_trade"]),
            "gross_bps_per_trade": float(summary["gross_bps_per_trade"]),
            "positive_rate": float(summary["positive_rate"]),
        }
    )
    risk = {
        key: (json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value)
        for key, value in risk.items()
    }
    return decisions, equity, monthly, risk


def _full_replay(
    frame: pd.DataFrame,
    fields: Sequence[str],
    params: N5ForestParams,
    *,
    year: int,
    train_cap: int,
    arm: str = "N5_challenger",
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[Any]]:
    parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    bundles: list[Any] = []
    for fold in _iter_fold_data(frame, year=year, train_cap=train_cap, held_cap=None):
        edges = _edges(fold["train"], fields)
        bundle, _ = fit_n5_forest(fold["train"], fields, edges, params=params)
        prediction, multiplier = bundle.size_multiplier(fold["held"])
        parts.append(_output(fold["held"], prediction, multiplier, arm=arm))
        audits.append(
            {
                "year": year,
                "arm": arm,
                "fold": fold["fold"],
                "train_start": fold["train_start"],
                "cutoff": fold["cutoff"],
                "held_end": fold["held_end"],
                "train_rows": len(fold["train"]),
                "held_rows": len(fold["held"]),
                "field_count": len(fields),
                "edge_count": len(edges),
                **bundle.target_audit,
            }
        )
        bundles.append(bundle)
        # Explicitly drop the full feature train/held frames before creating
        # the next block.  ``parts`` retains only the narrow _output schema.
        del fold
        gc.collect()
    return pd.concat(parts, ignore_index=True), audits, bundles


def main() -> None:
    global FEATURE_GROUP_CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-cap", type=int, default=40_000)
    parser.add_argument("--mda-held-cap", type=int, default=24_000)
    parser.add_argument("--hpo-held-cap", type=int, default=45_000)
    parser.add_argument("--mda-repeats", type=int, default=5)
    parser.add_argument("--max-trials", type=int, default=200)
    parser.add_argument(
        "--hpo-patience",
        type=int,
        default=30,
        help="Stop HPO after this many consecutive trials fail to improve the completed-trial winner.",
    )
    parser.add_argument(
        "--median-pruner-min-trials",
        type=int,
        default=5,
        help="Completed reference trials required before aggressive chronological median pruning.",
    )
    parser.add_argument(
        "--median-pruner-after-folds",
        type=int,
        default=2,
        help="Start median pruning after this many chronological held folds.",
    )
    parser.add_argument("--feature-sidecar-2025", type=Path)
    parser.add_argument("--feature-sidecar-2026", type=Path)
    parser.add_argument("--feature-contract", type=Path)
    parser.add_argument(
        "--feature-group-config", type=Path,
        help="Semantic group configuration paired with --feature-contract.",
    )
    parser.add_argument(
        "--reuse-mda-dir",
        type=Path,
        help="Reuse an immutable completed MDA checkpoint instead of recomputing permutations.",
    )
    parser.add_argument(
        "--hpo-feature-mode",
        choices=("compact", "full"),
        default="compact",
        help="Tune the portable compact contract or its mandatory unreduced control.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a partial run from deterministic MDA checkpoints in --out-dir.",
    )
    args = parser.parse_args()
    if args.feature_group_config is not None:
        FEATURE_GROUP_CONFIG = args.feature_group_config
    if args.out_dir.exists() and not args.resume:
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=args.resume)

    development, all_fields, dev_audit = _load_selection_input(
        INPUTS[2025],
        feature_sidecar=args.feature_sidecar_2025,
        feature_contract=args.feature_contract,
    )
    family_config = _feature_group_config()
    family_mapping = pd.DataFrame(
        {
            "feature": all_fields,
            "group": [_mda_group(field, family_config) for field in all_fields],
        }
    )
    schema_payload = {
        "schema": SCHEMA,
        "features": all_fields,
        "feature_count": len(all_fields),
        "feature_group_config": str(FEATURE_GROUP_CONFIG),
        "feature_group_config_sha256": _sha256(FEATURE_GROUP_CONFIG),
        "challenger_contract": (
            None if args.feature_contract is None else str(args.feature_contract)
        ),
    }
    schema_path = args.out_dir / "full_feature_schema.json"
    if args.resume and schema_path.exists():
        previous = json.loads(schema_path.read_text())
        if previous != schema_payload:
            raise ValueError("cannot resume MDA with a different feature/schema contract")
    else:
        family_mapping.to_parquet(args.out_dir / "feature_family_mapping.parquet", index=False)
        schema_path.write_text(json.dumps(schema_payload, indent=2) + "\n")
    if args.reuse_mda_dir is not None:
        mda_detail = pd.read_parquet(args.reuse_mda_dir / "portable_mda_detail.parquet")
        mda_summary = pd.read_parquet(args.reuse_mda_dir / "portable_mda_summary.parquet")
        group_mda_detail_path = args.reuse_mda_dir / "portable_group_mda_detail.parquet"
        group_mda_detail = (
            pd.read_parquet(group_mda_detail_path)
            if group_mda_detail_path.exists() else pd.DataFrame()
        )
        compact_fields = list(json.loads((args.reuse_mda_dir / "selected_feature_contract.json").read_text())["fields"])
        backward_path = args.reuse_mda_dir / "backward_elimination_path.parquet"
        backward_path_frame = (
            pd.read_parquet(backward_path) if backward_path.exists() else pd.DataFrame()
        )
        mda_proposed_fields = compact_fields
    else:
        dev_folds_mda = _fold_data(
            development,
            year=2025,
            train_cap=args.train_cap,
            held_cap=args.mda_held_cap,
            block_months=1,
        )
        mda_detail, group_mda_detail, mda_proposed_fields = _mda(
            dev_folds_mda,
            all_fields,
            params=BASELINE_N5_PARAMS,
            repeats=args.mda_repeats,
            checkpoint_dir=args.out_dir / "mda_checkpoints",
        )
        mda_summary = mda_detail.drop_duplicates("field").loc[
            :,
            [
                "field", "family", "group", "mda_median", "mda_mad", "mda_worst_fold",
                "positive_fold_recurrence", "portable_mda_score",
            ],
        ].sort_values("portable_mda_score", ascending=False, kind="stable")
        compact_fields, backward_path_frame = _backward_grouped_elimination(
            dev_folds_mda,
            all_fields,
            params=BASELINE_N5_PARAMS,
            group_mda=group_mda_detail,
            feature_mda=mda_detail,
        )
    selected_fields = list(all_fields if args.hpo_feature_mode == "full" else compact_fields)
    mda_detail.to_parquet(args.out_dir / "portable_mda_detail.parquet", index=False)
    mda_summary.to_parquet(args.out_dir / "portable_mda_summary.parquet", index=False)
    group_mda_detail.to_parquet(
        args.out_dir / "portable_group_mda_detail.parquet", index=False,
    )
    backward_path_frame.to_parquet(
        args.out_dir / "backward_elimination_path.parquet", index=False,
    )
    selected_set = set(compact_fields)
    removal_reason: dict[str, str] = {}
    if not backward_path_frame.empty:
        for row in backward_path_frame.loc[
            backward_path_frame["accepted"].fillna(False).astype(bool)
        ].to_dict("records"):
            if row.get("stage") == "full":
                continue
            removed = row.get("removed", ())
            if isinstance(removed, np.ndarray):
                removed = removed.tolist()
            for field in removed:
                removal_reason[str(field)] = (
                    f"accepted {row['stage']} removal: matched development metrics "
                    "remained within the full-challenger tolerance gate"
                )
    feature_decisions = pd.DataFrame(
        {
            "feature": all_fields,
            "group": [_mda_group(field, family_config) for field in all_fields],
            "selected": [field in selected_set for field in all_fields],
            "decision": [
                "retained by backward elimination"
                if field in selected_set
                else removal_reason.get(field, "not retained; see elimination path")
                for field in all_fields
            ],
        }
    ).merge(
        mda_summary.rename(columns={"field": "feature"}),
        on=["feature", "group"], how="left", suffixes=("", "_mda"),
    )
    feature_decisions.to_parquet(args.out_dir / "feature_selection_decisions.parquet", index=False)
    (args.out_dir / "selected_feature_contract.json").write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "selection_era": "2025 development only",
                "compatible_history": "2024-10 through 2025-07",
                "fields": selected_fields,
                "field_count": len(selected_fields),
                "compact_fields": compact_fields,
                "compact_field_count": len(compact_fields),
                "conditional_mda_proposed_fields": mda_proposed_fields,
                "conditional_mda_proposed_field_count": len(mda_proposed_fields),
                "feature_mode": args.hpo_feature_mode,
                "selection": (
                    "full challenger -> grouped knockout MDA -> context-conditional MDA -> "
                    "portability scoring -> retrained backward group elimination -> "
                    "bounded retrained within-group elimination; no fixed target field count"
                ),
                "unreduced_control_rule": "full contract remains eligible and wins whenever compact MDA does not improve full development performance",
                "raw_k9_memberships_used": False,
            },
            indent=2,
        )
        + "\n"
    )

    hpo_folds = _fold_data(
        development,
        year=2025,
        train_cap=args.train_cap,
        held_cap=args.hpo_held_cap,
    )
    if args.hpo_patience < 1:
        raise ValueError("--hpo-patience must be positive")
    if args.median_pruner_min_trials < 1:
        raise ValueError("--median-pruner-min-trials must be positive")
    if args.median_pruner_after_folds < 1:
        raise ValueError("--median-pruner-after-folds must be positive")
    trials = n5_hpo_candidates(max_trials=int(args.max_trials))
    trial_rows: list[dict[str, Any]] = []
    interim_reference: list[list[float]] = []
    best_completed_score = -np.inf
    trials_without_improvement = 0
    hpo_checkpoint_dir = args.out_dir / "hpo_checkpoints"
    hpo_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for index, params in enumerate(trials):
        checkpoint = hpo_checkpoint_dir / f"trial_{index:03d}.parquet"
        if checkpoint.exists():
            flat = pd.read_parquet(checkpoint).iloc[0].to_dict()
            expected_params = json.dumps(asdict(params), sort_keys=True)
            if str(flat.pop("params_json")) != expected_params:
                raise ValueError(f"HPO checkpoint {checkpoint} belongs to another candidate surface")
            interim = json.loads(str(flat.pop("interim_scores_json")))
            print(json.dumps({"event": "n5_hpo_trial_reused", "trial": index}), flush=True)
        else:
            print(json.dumps({"event": "n5_hpo_trial_start", "trial": index, "params": asdict(params)}), flush=True)
            _trial_output, metrics, pruned = _trial(
                hpo_folds,
                selected_fields,
                params,
                trial=index,
                pruning_reference=interim_reference,
                prune_after_folds=int(args.median_pruner_after_folds),
                median_pruner_min_trials=int(args.median_pruner_min_trials),
            )
            interim = list(metrics.pop("interim_scores"))
            flat = {key: value for key, value in metrics.items() if key != "params"}
            flat.update({f"param__{key}": value for key, value in asdict(params).items()})
            checkpoint_row = dict(flat)
            checkpoint_row["interim_scores_json"] = json.dumps(interim)
            checkpoint_row["params_json"] = json.dumps(asdict(params), sort_keys=True)
            pd.DataFrame([checkpoint_row]).to_parquet(checkpoint, index=False)
            print(
                json.dumps(
                    {
                        "event": "n5_hpo_trial_complete",
                        "trial": index,
                        "score": flat["selection_score"],
                        "pruned": bool(flat["pruned"]),
                    }
                ),
                flush=True,
            )
        interim_reference.append(list(interim))
        trial_rows.append(flat)
        pruned = bool(flat["pruned"])
        improved = False
        if not pruned and int(flat["completed_folds"]) == len(hpo_folds):
            score = float(flat["selection_score"])
            if score > best_completed_score:
                best_completed_score = score
                improved = True
        # A median-pruned trial has already demonstrated that it cannot beat
        # the contemporaneous reference at an earlier chronological fold.  It
        # is therefore a non-improving trial for HPO-patience purposes; not
        # counting it could needlessly exhaust all 200 candidates.
        if improved:
            trials_without_improvement = 0
        else:
            trials_without_improvement += 1
        if trials_without_improvement >= int(args.hpo_patience):
            print(
                json.dumps(
                    {
                        "event": "n5_hpo_early_stop",
                        "trial": index,
                        "patience": int(args.hpo_patience),
                        "best_completed_score": best_completed_score,
                    }
                ),
                flush=True,
            )
            break
    hpo = pd.DataFrame(trial_rows)
    eligible = hpo.loc[~hpo["pruned"] & hpo["completed_folds"].eq(len(hpo_folds))].copy()
    if eligible.empty:
        raise RuntimeError("all N5 HPO trials were pruned")
    eligible = eligible.sort_values(
        ["selection_score", "mean_portability_top1_2_5", "worst_month_top1_2_5", "top1_net_bps"],
        ascending=False,
        kind="stable",
    )
    winner_trial = int(eligible.iloc[0]["trial"])
    winner = trials[winner_trial]
    hpo.to_parquet(args.out_dir / "hpo_trials.parquet", index=False)
    (args.out_dir / "winner.json").write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "winner_trial": winner_trial,
                "params": asdict(winner),
                "selected_fields": selected_fields,
                "selection_metrics": eligible.iloc[0].to_dict(),
                "2026_used_for_selection": False,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )

    ablation_rows: list[dict[str, Any]] = []
    ablation_contracts = _required_ablation_contracts(all_fields, compact_fields)
    for arm, arm_fields in ablation_contracts.items():
        missing = sorted(set(arm_fields).difference(development.columns))
        if missing:
            raise ValueError(f"required development ablation {arm} lacks {missing}")
        arm_params = BASELINE_N5_PARAMS if arm == "canonical45_frozen_params" else winner
        _arm_output, arm_metrics = _evaluate_contract(
            hpo_folds, arm_fields, params=arm_params, arm=arm,
        )
        ablation_rows.append(
            {
                "arm": arm, "field_count": len(arm_fields),
                "params": json.dumps(asdict(arm_params), sort_keys=True),
                "selection_data": "2025 development only",
                **arm_metrics,
            }
        )
    pd.DataFrame(ablation_rows).sort_values(
        ["selection_score", "worst_month_top1_2_5", "top1_net_bps"],
        ascending=False, kind="stable",
    ).to_parquet(args.out_dir / "development_required_ablation_metrics.parquet", index=False)
    (args.out_dir / "development_ablation_contracts.json").write_text(
        json.dumps(ablation_contracts, indent=2) + "\n"
    )

    all_global: list[pd.DataFrame] = []
    all_monthly: list[pd.DataFrame] = []
    all_weekly: list[pd.DataFrame] = []
    all_stability: list[pd.DataFrame] = []
    all_portfolio_risk: list[dict[str, Any]] = []
    canonical_fields = list(json.loads(CANONICAL_N5_CONTRACT.read_text())["features"])
    for year in (2025, 2026):
        # Do not retain both full-universe years in memory.  The 2026 frame is
        # loaded only after 2025 has been fully persisted and released.
        frame = development if year == 2025 else _load_selection_input(
            INPUTS[2026],
            feature_sidecar=args.feature_sidecar_2026,
            feature_contract=args.feature_contract,
        )[0]
        start, end = PERIODS[year]
        # Process the fixed canonical control first and release it before
        # generating the challenger.  Retaining both full held populations
        # concurrently is unnecessary and was the source of the prior OOM.
        canonical_output, canonical_audit, _canonical_bundles = _full_replay(
            frame,
            canonical_fields,
            BASELINE_N5_PARAMS,
            year=year,
            train_cap=args.train_cap,
            arm="N5_canonical45_control",
        )
        for output, arm in ((canonical_output, "N5_canonical45_control"),):
            global_metrics = _period_tail_metrics(output, arm=arm, period_kind="global").assign(year=year)
            monthly = _period_tail_metrics(output, arm=arm, period_kind="month").assign(year=year)
            weekly = _period_tail_metrics(output, arm=arm, period_kind="week").assign(year=year)
            stability = _stability(monthly.drop(columns="year")).assign(year=year)
            all_global.append(global_metrics)
            all_monthly.append(monthly)
            all_weekly.append(weekly)
            all_stability.append(stability)
            decisions, equity, portfolio_monthly, risk = _portfolio_full(
                frame,
                output,
                arm=arm,
                start=start,
                end=end,
            )
            risk["year"] = year
            all_portfolio_risk.append(risk)
            decisions.to_parquet(args.out_dir / f"portfolio_decisions_{year}_{arm}.parquet", index=False)
            equity.to_parquet(args.out_dir / f"portfolio_equity_{year}_{arm}.parquet", index=False)
            portfolio_monthly.to_parquet(args.out_dir / f"portfolio_monthly_{year}_{arm}.parquet", index=False)
            del decisions, equity, portfolio_monthly
            gc.collect()
        del canonical_output, _canonical_bundles
        gc.collect()

        winner_output, audit, bundles = _full_replay(
            frame,
            selected_fields,
            winner,
            year=year,
            train_cap=args.train_cap,
            arm="N5_grouped_challenger",
        )
        # The equal-size control differs only in the multiplier.  Flip it
        # in-place, capture its metrics, then restore the challenger values;
        # do not make a second full-universe DataFrame copy.
        saved_multiplier = winner_output["trust_size_multiplier"].to_numpy(copy=True)
        for arm, equal_size in (("equal_control", True), ("N5_grouped_challenger", False)):
            if equal_size:
                winner_output.loc[:, "trust_size_multiplier"] = 1.0
            else:
                winner_output.loc[:, "trust_size_multiplier"] = saved_multiplier
            output = winner_output
            global_metrics = _period_tail_metrics(output, arm=arm, period_kind="global").assign(year=year)
            monthly = _period_tail_metrics(output, arm=arm, period_kind="month").assign(year=year)
            weekly = _period_tail_metrics(output, arm=arm, period_kind="week").assign(year=year)
            stability = _stability(monthly.drop(columns="year")).assign(year=year)
            all_global.append(global_metrics)
            all_monthly.append(monthly)
            all_weekly.append(weekly)
            all_stability.append(stability)
            decisions, equity, portfolio_monthly, risk = _portfolio_full(
                frame, output, arm=arm, start=start, end=end,
            )
            risk["year"] = year
            all_portfolio_risk.append(risk)
            decisions.to_parquet(args.out_dir / f"portfolio_decisions_{year}_{arm}.parquet", index=False)
            equity.to_parquet(args.out_dir / f"portfolio_equity_{year}_{arm}.parquet", index=False)
            portfolio_monthly.to_parquet(args.out_dir / f"portfolio_monthly_{year}_{arm}.parquet", index=False)
            del decisions, equity, portfolio_monthly
            gc.collect()
        winner_output.loc[:, "trust_size_multiplier"] = saved_multiplier
        del saved_multiplier
        winner_output.assign(year=year).to_parquet(
            args.out_dir / f"oof_predictions_{year}.parquet", index=False, compression="zstd",
        )
        pd.concat(
            [pd.DataFrame(canonical_audit), pd.DataFrame(audit)], ignore_index=True,
        ).to_parquet(args.out_dir / f"fold_audit_{year}.parquet", index=False)
        for index, bundle in enumerate(bundles):
            bundle_dir = args.out_dir / f"bundle_{year}_fold{index}"
            bundle_dir.mkdir()
            joblib.dump(bundle, bundle_dir / "n5_bundle.joblib", compress=3)
            (bundle_dir / "run_manifest.json").write_text(json.dumps(bundle.manifest(), indent=2, default=str) + "\n")
        del winner_output, bundles, audit, canonical_audit
        if year == 2025:
            # Feature-selection and HPO are finished; the full 2025 source is
            # no longer required before loading the untouched 2026 frame.
            del development
        else:
            del frame
        gc.collect()

    pd.concat(all_global, ignore_index=True).to_parquet(args.out_dir / "metrics_global.parquet", index=False)
    pd.concat(all_monthly, ignore_index=True).to_parquet(args.out_dir / "metrics_monthly.parquet", index=False)
    pd.concat(all_weekly, ignore_index=True).to_parquet(args.out_dir / "metrics_weekly.parquet", index=False)
    pd.concat(all_stability, ignore_index=True).to_parquet(args.out_dir / "stability.parquet", index=False)
    pd.DataFrame(all_portfolio_risk).to_parquet(args.out_dir / "portfolio_risk_metrics.parquet", index=False)
    manifest = {
        "schema": SCHEMA,
        "status": "N5_CANONICAL_RESEARCH_SIZING_SELECTED",
        "selection_era": "2025 development OOF only",
        "confirmation_era": "2026 frozen once; not used for selection",
        "selection_hpo_2026_rows_read": 0,
        "compatible_history_span": "2024-10-01 through 2025-07-31",
        "mda": {
            "style": (
                "hierarchical semantic-family knockout then conditional feature MDA, "
                "within month x frozen-score decile"
            ),
            "folds": "seven one-month 2025 folds",
            "semantic_group_config": str(FEATURE_GROUP_CONFIG),
            "semantic_group_config_sha256": _sha256(FEATURE_GROUP_CONFIG),
            "train_cap": args.train_cap,
            "held_cap": args.mda_held_cap,
            "repeats": args.mda_repeats,
            "selected_fields": selected_fields,
            "compact_fields": compact_fields,
            "feature_mode": args.hpo_feature_mode,
            "reuse_mda_dir": str(args.reuse_mda_dir) if args.reuse_mda_dir else None,
            "backward_elimination": _feature_group_config()["backward_elimination"],
        },
        "hpo": {
            "max_trials": args.max_trials,
            "held_cap": args.hpo_held_cap,
            "hpo_patience": args.hpo_patience,
            "median_pruner_min_trials": args.median_pruner_min_trials,
            "median_pruner_after_folds": args.median_pruner_after_folds,
            "winner_trial": winner_trial,
            "winner_params": asdict(winner),
            "targets": {
                "mean": ["policy_net", "parent_residual", "winsorized_net"],
                "risk": ["oob_squared", "oob_downside", "oob_absolute"],
            },
        },
        "ranking": "unchanged pooled-global strict-R3 final_score",
        "admission": "unchanged causal 21/42/84-day expected net >= +50 bps",
        "integration": "post-admission bounded relative size multiplier only",
        "portfolio": "8 concurrent, 2 entries/bar, 1/asset, 80% margin, 7x leverage",
        "cost": "selected SimplePolicyOptimiser policy net with 100 bps exactly once",
        "raw_k9_memberships_used": False,
        "source_2025": str(INPUTS[2025]),
        "source_2025_sha256": _sha256(INPUTS[2025]),
        "source_2026": str(INPUTS[2026]),
        "source_2026_sha256": _sha256(INPUTS[2026]),
        "development_source_audit": dev_audit,
        "seed": SEED,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), "winner_trial": winner_trial, "field_count": len(selected_fields)}), flush=True)


if __name__ == "__main__":
    main()
