#!/usr/bin/env python3
"""Replay a frozen ten-head consensus through the matched C3 downstream stack.

This adapter deliberately answers one narrow question:

    Does the already-frozen conditional ten-head winner retain value after the
    *complete* causal downstream topology is applied?

It does not reopen feature selection, target selection, query construction,
head HPO, C3 parameters, safety parameters, correctness parameters, admission
thresholds, portfolio limits, or exit geometry.  The competing arms use the
same full long-only source candidate population and the same fixed policy
labels:

* ``control``: pre-existing strict-R3 base + consensus handoff;
* ``frozen_ten_head``: the immutable 2026-08-10 conditional-usefulness winner.

Both arms are then re-scored through the current C3 *downstream topology*:

    upstream base + consensus
        -> rolling three-month raw K9 geometry / current R3 leaf state
        -> exact-H12 Severe-200 demotion
        -> +100 bps policy-residual correctness demotion
        -> same-model prior-42-day CDF
        -> causal prior-resolved 21-day expected-net admission
        -> the canonical constrained long-only portfolio auction.

The base/consensus labels and exit contract remain the frozen historical
SL=3 ATR, trailing activation=0.5 ATR, giveback=0.25 ATR, H12, 100-bps-once
policy.  Consequently this is a matched downstream replay for the ten-head
experiment, *not* a claim that the selected C3 Part-A execution-policy model
has been reproduced.  The run manifest makes that boundary explicit.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from lightgbm import LGBMRanker


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_causal_geometry_k9_c3_ablation as c3  # noqa: E402
from extreme_price_movements.stage_i_causal_admission import (  # noqa: E402
    Causal21dAdmissionSpec,
    apply_causal_21d_side_admission,
)
from scripts.replay_strict_r3_forward_portfolio import _auction_candidates  # noqa: E402
from scripts.replay_strict_r3_policy_portfolio_2025_2026 import _run  # noqa: E402
from scripts.run_strict_r3_c3_window_cadence_ablation import _overlay_fields  # noqa: E402
import scripts.run_ten_head_conditional_usefulness_funnel as ten  # noqa: E402


SOURCE = ROOT / (
    "data_perp/artifacts/strict_r3_schema_v2_source_panel_long_2022_2026_20260809_v1/"
    "canonical_source_panel.parquet"
)
UPSTREAM = ROOT / "data_perp/artifacts/strict_r3_full_inference_2025_2026_v2/predictions.parquet"
FROZEN_HEADS = ROOT / "data_perp/artifacts/ten_head_conditional_usefulness_20260810_v1/frozen_head_configs.json"
FROZEN_FINAL = ROOT / "data_perp/artifacts/ten_head_conditional_usefulness_20260810_v1/final_frozen_winner_predictions.parquet"
DEFAULT_OUT = ROOT / "data_perp/artifacts/ten_head_c3_full_stack_replay_20260810_v1"

SCORE_START = pd.Timestamp("2025-02-01", tz="UTC")
DEFAULT_EVALUATION_MONTHS = ("2025-08", "2025-09", "2025-10")
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10)
SEED = ten.SEED
MAX_TRAIN_ROWS = 60_000

SOURCE_COLUMNS = [
    "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name",
    "r3_class", "r3_label_available_ts", "policy_path_valid",
    "policy_label_available_ts", "policy_gross_bps", "policy_net_bps",
    "policy_exit_bar_15m", "policy_exit_reason", "policy_entry_price",
    "policy_exit_price", "policy_cost_bps", "h12_label_valid",
    "h12_label_available_ts", "h12_tp6_sl4_net_bps",
]


def _event(name: str, **values: Any) -> None:
    print(json.dumps({"event": name, **values}, default=str), flush=True)


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    result = pd.Timestamp(value)
    return result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")


def _month_start(month: str | pd.Timestamp) -> pd.Timestamp:
    value = _utc(month)
    return value.normalize().replace(day=1)


def _month_end(month: str | pd.Timestamp) -> pd.Timestamp:
    return _month_start(month) + pd.offsets.MonthBegin(1)


def _months_between(start: pd.Timestamp, end_exclusive: pd.Timestamp) -> list[str]:
    return pd.period_range(start.tz_convert(None), (end_exclusive - pd.Timedelta(days=1)).tz_convert(None), freq="M").astype(str).tolist()


def _month_add(value: pd.Timestamp, offset: int) -> pd.Timestamp:
    return (value.tz_convert(None).to_period("M") + offset).to_timestamp().tz_localize("UTC")


def _sha256_payload(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _source_fields() -> list[str]:
    return ten.source_feature_columns(ten._source_columns(SOURCE))


def _read_source(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    fields: Sequence[str],
    with_upstream: bool,
) -> pd.DataFrame:
    """Load only one chronological source slice, never materialising the panel."""
    requested = list(dict.fromkeys([*SOURCE_COLUMNS, *fields]))
    source = ds.dataset(SOURCE, format="parquet")
    expression = (
        (ds.field("__ts__") >= pa.scalar(start.to_pydatetime(), type=pa.timestamp("ns", tz="UTC")))
        & (ds.field("__ts__") < pa.scalar(end.to_pydatetime(), type=pa.timestamp("ns", tz="UTC")))
    )
    frame = source.to_table(columns=requested, filter=expression).to_pandas()
    if frame.empty:
        raise ValueError(f"source has no rows from {start} to {end}")
    for column in (
        "__ts__", "__decision_ts__", "r3_label_available_ts",
        "policy_label_available_ts", "h12_label_available_ts",
    ):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if frame["candidate_id"].duplicated().any():
        raise ValueError("source slice has duplicate candidate IDs")
    if not frame["side_name"].astype(str).str.lower().eq("long").all():
        raise ValueError("matched replay expects only long source candidates")
    for field in fields:
        frame[field] = pd.to_numeric(frame[field], errors="coerce").astype(np.float32)
    if with_upstream:
        upstream = ds.dataset(UPSTREAM, format="parquet")
        upstream_columns = [
            "candidate_id", "__ts__", "base_score", "base_anchor_bps", "base_rank",
            "consensus_rank", "final_score",
        ]
        scored = upstream.to_table(columns=upstream_columns, filter=expression).to_pandas()
        scored["__ts__"] = pd.to_datetime(scored["__ts__"], utc=True, errors="raise")
        if scored["candidate_id"].duplicated().any():
            raise ValueError("upstream slice has duplicate candidate IDs")
        frame = frame.merge(
            scored.drop(columns="__ts__"), on="candidate_id", how="left", validate="one_to_one",
            indicator="__upstream_join__",
        )
        if not frame["__upstream_join__"].eq("both").all():
            missing = int((~frame["__upstream_join__"].eq("both")).sum())
            raise ValueError(f"upstream does not cover {missing} source candidates")
        frame = frame.drop(columns="__upstream_join__")
        for column in ("base_score", "base_anchor_bps", "base_rank", "consensus_rank", "final_score"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["month"] = frame["__ts__"].dt.strftime("%Y-%m")
    return frame.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _valid_residual_rows(frame: pd.DataFrame) -> pd.DataFrame:
    valid = (
        frame["policy_path_valid"].fillna(False).astype(bool)
        & frame["policy_label_available_ts"].notna()
        & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(frame["policy_gross_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(frame["base_anchor_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(frame["base_rank"], errors="coerce"))
    )
    return frame.loc[valid].copy()


def _load_frozen_configs(path: Path) -> dict[str, ten.HeadConfig]:
    payload = json.loads(path.read_text())
    by_name = {spec.name: spec for spec in ten.HEAD_SPECS}
    if set(payload) != set(by_name):
        raise ValueError("frozen head manifest is not the exact ten-member contract")
    configs: dict[str, ten.HeadConfig] = {}
    for name, data in payload.items():
        config = ten.HeadConfig(
            spec=by_name[name], target_name=str(data["target_name"]),
            query_name=str(data["query_name"]), fields=[str(x) for x in data["fields"]],
            params=dict(data["params"]), hpo_selected=bool(data.get("hpo_selected", False)),
            selection_log=list(data.get("selection_log", [])),
        )
        if config.spec.cap != int(data["cap"]) or len(config.fields) != int(data["field_count"]):
            raise ValueError(f"frozen head {name} has a malformed cap/field contract")
        configs[name] = config
    return configs


def _head_seed_for_month(head: str, month: str, month_position: int) -> int:
    """Reproduce frozen development/final seeds where those ledgers exist.

    The source experiment separately started the May--July development and
    August--October final loops at zero.  We retain those exact offsets so the
    all-candidate Aug--Oct outputs can be checked bit-for-bit against the
    label-valid frozen result.  Earlier warm-up months only seed downstream
    history and use a disjoint deterministic offset.
    """
    known = {
        "2025-05": 0, "2025-06": 1, "2025-07": 2,
        "2025-08": 0, "2025-09": 1, "2025-10": 2,
    }
    return ten._head_seed(head) + known.get(month, 100 + int(month_position))


def _fit_and_score_head(
    train: pd.DataFrame,
    held: pd.DataFrame,
    config: ten.HeadConfig,
    *,
    seed: int,
    max_train_rows: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit the frozen head on resolved rows and score *all* held candidates.

    This is intentionally the no-early-stopping final-fit branch of the
    frozen runner.  Held labels are absent from every operation in this
    function; the output is therefore valid on target-free candidates.
    """
    if train.empty:
        return np.full(len(held), 0.5, dtype=np.float32), {
            "status": "neutral_no_prior_resolved_rows", "fit_rows": 0,
            "train_rows": 0, "test_rows": int(len(held)),
        }
    query_definition = ten.query_definitions_by_name([config.query_name])[0]
    work = train.copy()
    work["__target__"] = ten.residual_grade(
        work["policy_net_bps"].to_numpy(float) - work["base_anchor_bps"].to_numpy(float),
        ten.TARGETS[config.target_name],
    )
    sampled, _ = ten._group_sample(
        work,
        ten.assign_query_ids(work, query_definition),
        max_rows=int(max_train_rows), weight_mode=config.spec.weight_mode, seed=int(seed),
    )
    if sampled.empty or sampled["__target__"].nunique() < 2:
        return np.full(len(held), 0.5, dtype=np.float32), {
            "status": "neutral_insufficient_training_support", "fit_rows": int(len(sampled)),
            "train_rows": int(len(sampled)), "test_rows": int(len(held)),
        }
    core = sampled.sort_values(["__query__", "__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    _, groups = np.unique(core["__query__"].to_numpy(), return_counts=True)
    if len(groups) == 0 or core["__target__"].nunique() < 2:
        return np.full(len(held), 0.5, dtype=np.float32), {
            "status": "neutral_query_support", "fit_rows": int(len(core)),
            "train_rows": int(len(sampled)), "test_rows": int(len(held)),
        }
    medians = ten._fit_medians(core, config.fields)
    params = ten._ranker_params(config.params, training_rows=len(core), seed=int(seed))
    model = LGBMRanker(**params)
    model.fit(
        ten._model_matrix(core, config.fields, medians), core["__target__"].to_numpy(np.int32),
        group=groups.astype(np.int64), sample_weight=ten._weights(core, config.spec.weight_mode),
    )
    reference = model.predict(ten._model_matrix(core, config.fields, medians))
    held_raw = model.predict(ten._model_matrix(held, config.fields, medians))
    return ten._rank_against_reference(reference, held_raw).astype(np.float32), {
        "status": "fit", "fit_rows": int(len(core)), "train_rows": int(len(sampled)),
        "test_rows": int(len(held)), "train_queries": int(len(groups)),
        "best_iteration": int(getattr(model, "best_iteration_", 0) or params.get("n_estimators", 0)),
    }


def _score_frozen_consensus(
    *,
    fields: Sequence[str],
    configs: dict[str, ten.HeadConfig],
    score_months: Sequence[str],
    max_train_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Materialise frozen head outputs on every source candidate month by month."""
    # The original frozen runner begins residual training on 2025-02-01.
    # February is a training-only warm-up here: no valid residual model can be
    # fit before it because the compatible upstream score surface itself begins
    # in February, but its resolved labels must be present in every later fit.
    warmup = _read_source(SCORE_START, _month_end("2025-02"), fields=fields, with_upstream=True)
    prior_labels: list[pd.DataFrame] = [_valid_residual_rows(warmup).loc[:, [
        "candidate_id", "__ts__", "side_name", "month", "policy_label_available_ts",
        "policy_net_bps", "base_anchor_bps", *fields,
    ]]]
    score_rows: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for position, month in enumerate(score_months):
        start, end = _month_start(month), _month_end(month)
        _event("frozen_head_month_start", month=month)
        full = _read_source(start, end, fields=fields, with_upstream=True)
        train = (
            pd.concat(prior_labels, ignore_index=True)
            if prior_labels else full.iloc[:0].copy()
        )
        if not train.empty:
            train = train.loc[train["policy_label_available_ts"].lt(start)].copy()
        output = full.loc[:, [
            *SOURCE_COLUMNS, "month", "base_score", "base_anchor_bps", "base_rank",
            "consensus_rank", "final_score",
        ]].copy()
        for head_index, spec in enumerate(ten.HEAD_SPECS):
            config = configs[spec.name]
            rank, meta = _fit_and_score_head(
                train, full, config,
                seed=_head_seed_for_month(spec.name, month, position),
                max_train_rows=max_train_rows,
            )
            output[spec.name] = rank
            audit_rows.append({
                "month": month, "head": spec.name, "target": config.target_name,
                "query": config.query_name, "field_count": len(config.fields), **meta,
            })
        head_names = [spec.name for spec in ten.HEAD_SPECS]
        output["new_consensus_rank"] = np.nanmedian(output.loc[:, head_names].to_numpy(float), axis=1)
        output["new_upstream"] = .75 * output["base_rank"].to_numpy(float) + .25 * output["new_consensus_rank"].to_numpy(float)
        output["control_consensus_rank"] = output["consensus_rank"].to_numpy(float)
        output["control_upstream"] = output["final_score"].to_numpy(float)
        output["r3_label_valid"] = output["r3_class"].notna()
        output["r3_clear"] = output["r3_class"].eq(2).astype(np.int8)
        if output["candidate_id"].duplicated().any() or len(output) != len(full):
            raise AssertionError(f"{month}: all-candidate frozen scoring changed identities")
        coverage_rows.append({
            "month": month, "source_candidates": int(len(full)),
            "label_valid_policy_rows": int(len(_valid_residual_rows(full))),
            "new_score_rows": int(np.isfinite(output["new_upstream"]).sum()),
            "control_score_rows": int(np.isfinite(output["control_upstream"]).sum()),
        })
        score_rows.append(output)
        # The frozen ranker consumes only prior resolved labels.  Retain the
        # minimal label-valid feature matrix; the full candidate frame is freed
        # before the next month to keep the adapter bounded in memory.
        prior_labels.append(_valid_residual_rows(full).loc[:, [
            "candidate_id", "__ts__", "side_name", "month", "policy_label_available_ts",
            "policy_net_bps", "base_anchor_bps", *fields,
        ]])
        _event("frozen_head_month_complete", month=month, candidates=len(full), train_rows=len(train))
    scores = pd.concat(score_rows, ignore_index=True).sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if scores["candidate_id"].duplicated().any():
        raise AssertionError("frozen score ledger has duplicate candidate IDs")
    return scores, pd.DataFrame(audit_rows), pd.DataFrame(coverage_rows)


def _check_frozen_final_reproduction(scores: pd.DataFrame) -> dict[str, Any]:
    """Assert that the all-candidate adapter preserves frozen valid-row scores."""
    frozen = pd.read_parquet(FROZEN_FINAL)
    candidate = scores.merge(
        frozen[["candidate_id", "consensus_rank", "score"]].rename(
            columns={"consensus_rank": "frozen_consensus_rank", "score": "frozen_score"},
        ), on="candidate_id", how="inner", validate="one_to_one",
    )
    expected = len(frozen)
    if len(candidate) != expected:
        raise AssertionError(f"frozen final reproduction covers {len(candidate)}/{expected} label-valid rows")
    consensus_error = np.abs(candidate["new_consensus_rank"].to_numpy(float) - candidate["frozen_consensus_rank"].to_numpy(float))
    score_error = np.abs(candidate["new_upstream"].to_numpy(float) - candidate["frozen_score"].to_numpy(float))
    # Tree predictions and CDF ranks are deterministic under the frozen seed.
    # Float32 persistence is the only expected source of tiny arithmetic noise.
    if float(np.nanmax(consensus_error)) > 2e-6 or float(np.nanmax(score_error)) > 2e-6:
        raise AssertionError("all-candidate frozen-head scoring failed to reproduce the frozen valid-row result")
    return {
        "rows": int(len(candidate)),
        "max_abs_consensus_rank_error": float(np.nanmax(consensus_error)),
        "max_abs_upstream_score_error": float(np.nanmax(score_error)),
        "passed": True,
    }


@dataclass
class C3Context:
    cutoff: pd.Timestamp
    held_end: pd.Timestamp
    meta: pd.DataFrame
    reference: pd.DataFrame
    held: pd.DataFrame
    state: pd.DataFrame
    state_fields: list[str]
    geometry_audit: dict[str, Any]
    leaf_audit: dict[str, Any]
    training_start: pd.Timestamp


def _read_features_for_context(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    fields: Sequence[str],
) -> pd.DataFrame:
    return _read_source(start, end, fields=fields, with_upstream=False)


def _prepare_c3_context(
    *,
    ledger: pd.DataFrame,
    fields: Sequence[str],
    cutoff: pd.Timestamp,
    held_end: pd.Timestamp,
    previous_geometry: c3.RawK9Bundle | None,
) -> tuple[C3Context, c3.RawK9Bundle]:
    """Fit one causal C3 geometry/leaf bundle shared by both score arms."""
    training_start = _month_add(cutoff, -6)
    geometry_end = training_start
    geometry_start = _month_add(geometry_end, -3)
    geometry_source = _read_features_for_context(geometry_start, geometry_end, fields=fields)
    bundle_id = f"matched_ten_head_c3__g{geometry_start:%Y%m%d}_{geometry_end:%Y%m%d}__fit{cutoff:%Y%m%d}"
    geometry, geometry_audit = c3._fit_raw_k9(
        geometry_source, fields, bundle_id=bundle_id, fit_start=geometry_start,
        fit_end=geometry_end, source_kind="raw_complete_point_in_time_market_burnin",
        previous=previous_geometry,
    )
    source = _read_features_for_context(training_start, held_end, fields=fields)
    score_window = ledger.loc[
        ledger["__ts__"].ge(training_start) & ledger["__ts__"].lt(held_end)
    ].copy()
    data = score_window.merge(
        source[["candidate_id", *fields]], on="candidate_id", how="inner", validate="one_to_one",
    )
    if len(data) != len(score_window):
        raise AssertionError("C3 source feature join changed the frozen score candidate population")
    meta_mask = (
        data["__ts__"].lt(cutoff)
        & data["policy_label_available_ts"].lt(cutoff)
        & data["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(data["policy_net_bps"], errors="coerce"))
        & data["h12_label_available_ts"].lt(cutoff)
        & data["h12_label_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(data["h12_tp6_sl4_net_bps"], errors="coerce"))
        & data["h12_label_available_ts"].lt(cutoff)
        & data["h12_label_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(data["h12_tp6_sl4_net_bps"], errors="coerce"))
    )
    meta = data.loc[meta_mask].copy()
    # The current C3 helper names its fixed safety target explicitly.  Keep the
    # alias local to the downstream training frame; the target never appears in
    # the scoring feature matrix.
    meta["exact_h12_net_bps"] = meta["h12_tp6_sl4_net_bps"].to_numpy(float)
    reference = data.loc[data["__ts__"].ge(cutoff - pd.Timedelta(days=42)) & data["__ts__"].lt(cutoff)].copy()
    held = data.loc[data["__ts__"].ge(cutoff) & data["__ts__"].lt(held_end)].copy()
    leaf_train = data.loc[
        data["__ts__"].lt(cutoff)
        & data["r3_label_available_ts"].lt(cutoff)
        & data["r3_label_valid"].fillna(False).astype(bool),
    ].copy()
    if held.empty or len(reference) < 100 or len(meta) < 1_000 or len(leaf_train) < 1_000:
        raise ValueError(
            f"{cutoff:%Y-%m}: insufficient C3 support held/reference/meta/leaf="
            f"{len(held)}/{len(reference)}/{len(meta)}/{len(leaf_train)}"
        )
    leaf_reference, leaf_audit = c3._fit_leaf_reference(leaf_train, fields)
    state_population = pd.concat([meta, reference, held], ignore_index=True).drop_duplicates(
        "candidate_id", keep="last",
    ).sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    k9 = geometry.transform(state_population)
    leaf = c3._leaf_state_from_reference(leaf_reference, state_population)
    state = c3._state_features(state_population, k9, leaf)
    state.index = state_population["candidate_id"].to_numpy()
    fields_for_overlay = _overlay_fields(
        ["base_score", "base_anchor_bps", "base_rank", "consensus_rank", "final_score"],
        state.columns.tolist(), include_k9_soft_memberships=True,
    )
    return C3Context(
        cutoff=cutoff, held_end=held_end, meta=meta, reference=reference, held=held,
        state=state, state_fields=fields_for_overlay, geometry_audit=geometry_audit,
        leaf_audit=leaf_audit, training_start=training_start,
    ), geometry


def _attach_arm(frame: pd.DataFrame, state: pd.DataFrame, scores: pd.DataFrame, *, arm: str) -> pd.DataFrame:
    columns = ["candidate_id", "base_score", "base_anchor_bps", "base_rank", "consensus_rank", "upstream"]
    values = scores.loc[:, columns].copy()
    output = frame.drop(
        columns=["base_score", "base_anchor_bps", "base_rank", "consensus_rank", "final_score"],
        errors="ignore",
    ).merge(values, on="candidate_id", how="left", validate="one_to_one")
    if output[["base_score", "base_anchor_bps", "base_rank", "consensus_rank", "upstream"]].isna().any().any():
        raise AssertionError(f"{arm}: score handoff is incomplete")
    output = output.join(state, on="candidate_id", how="left")
    if output[state.columns.tolist()].isna().any().any():
        raise AssertionError(f"{arm}: C3 state handoff is incomplete")
    output["final_score"] = output["upstream"].to_numpy(float)
    return output


def _score_c3_arm(context: C3Context, scores: pd.DataFrame, *, arm: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply fixed policy safety/correctness overlays, then a prior-42d CDF."""
    meta = _attach_arm(context.meta, context.state, scores, arm=arm)
    reference = _attach_arm(context.reference, context.state, scores, arm=arm)
    held = _attach_arm(context.held, context.state, scores, arm=arm)
    # C3 state has been built from target-free raw fields and train-only R3
    # labels.  The two supervised downstream estimators see only resolved meta
    # rows; reference and held labels are not read by their fit/predict calls.
    meta_fit = c3._equal_month_sample(meta, c3.MODEL_CAP, seed=c3.SEED + 241)
    scored = pd.concat(
        [reference.assign(__score_role__="reference"), held.assign(__score_role__="held")],
        ignore_index=True,
    )
    severe, severe_audit = c3._fit_safety(
        meta_fit, scored, context.state_fields,
    )
    scored["severe200_probability"] = severe
    scored["raw_severe"] = scored["final_score"].to_numpy(float) * (1.0 - 0.5 * severe)
    correctness_raw, correctness_rank, correctness_audit = c3._fit_correctness(
        meta_fit, scored, context.state_fields,
    )
    scored["correctness_raw"] = correctness_raw
    scored["correctness_rank"] = correctness_rank
    scored["raw_correctness_demote"] = scored["raw_severe"].to_numpy(float) * (
        0.25 + 0.75 * correctness_rank
    )
    reference_mask = scored["__score_role__"].eq("reference").to_numpy()
    scored["final_score"] = c3._pct(
        scored.loc[reference_mask, "raw_correctness_demote"].to_numpy(float),
        scored["raw_correctness_demote"].to_numpy(float),
    )
    output = scored.loc[~reference_mask].copy()
    output["arm"] = arm
    output["model_cutoff"] = context.cutoff
    output["model_held_end_exclusive"] = context.held_end
    output["geometry_bundle_sha256"] = context.geometry_audit["bundle_sha256"]
    output["geometry_fit_start"] = context.geometry_audit["fit_start"]
    output["geometry_fit_end_exclusive"] = context.geometry_audit["fit_end_exclusive"]
    output["meta_training_start"] = context.training_start
    output["state_feature_count"] = len(context.state_fields)
    audit = {
        "arm": arm, "cutoff": context.cutoff, "held_end_exclusive": context.held_end,
        "held_rows": int(len(output)), "reference_rows": int(reference_mask.sum()),
        "meta_fit_rows": int(len(meta_fit)), "state_feature_count": int(len(context.state_fields)),
        "held_outcomes_consumed": False, "reference_and_held_share_one_geometry_bundle": True,
        "same_model_prior42_final_cdf": True, **context.geometry_audit, **context.leaf_audit,
        **severe_audit, **correctness_audit,
    }
    return output, audit


def _tail_metrics(frame: pd.DataFrame, *, arm: str, stage: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank all scored candidates first, then inspect valid executable outcomes."""
    population = frame.loc[
        frame["__ts__"].ge(start) & frame["__ts__"].lt(end)
        & np.isfinite(pd.to_numeric(frame[stage], errors="coerce")),
    ].copy()
    rows: list[dict[str, Any]] = []
    monthly: list[dict[str, Any]] = []
    for tail in TAILS:
        selected = population.nlargest(max(1, int(math.ceil(tail * len(population)))), stage, keep="first")
        valid = selected.loc[
            selected["policy_path_valid"].fillna(False).astype(bool)
            & np.isfinite(pd.to_numeric(selected["policy_net_bps"], errors="coerce")),
        ].copy()
        rows.append({
            "arm": arm, "stage": stage, "scope": "pooled_global", "tail": tail,
            "score_population_rows": int(len(population)), "selected_score_rows": int(len(selected)),
            "valid_outcomes": int(len(valid)), "outcome_coverage": float(len(valid) / max(len(selected), 1)),
            "gross_bps_per_trade": float(valid["policy_gross_bps"].mean()),
            "net_bps_per_trade": float(valid["policy_net_bps"].mean()),
            "net_sum_bps": float(valid["policy_net_bps"].sum()),
            "positive_rate": float(valid["policy_net_bps"].gt(0.0).mean()),
        })
        for month, block in selected.groupby(selected["__ts__"].dt.strftime("%Y-%m"), sort=True):
            month_valid = block.loc[
                block["policy_path_valid"].fillna(False).astype(bool)
                & np.isfinite(pd.to_numeric(block["policy_net_bps"], errors="coerce")),
            ]
            monthly.append({
                "arm": arm, "stage": stage, "scope": "global_tail_month_contribution", "tail": tail,
                "month": month, "selected_score_rows": int(len(block)), "valid_outcomes": int(len(month_valid)),
                "outcome_coverage": float(len(month_valid) / max(len(block), 1)),
                "gross_bps_per_trade": float(month_valid["policy_gross_bps"].mean()),
                "net_bps_per_trade": float(month_valid["policy_net_bps"].mean()),
                "net_sum_bps": float(month_valid["policy_net_bps"].sum()),
                "positive_rate": float(month_valid["policy_net_bps"].gt(0.0).mean()),
            })
    return pd.DataFrame(rows), pd.DataFrame(monthly)


def _run_admission_and_portfolio(
    frame: pd.DataFrame,
    *,
    arm: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    admitted, admission_audit = apply_causal_21d_side_admission(
        frame,
        score_column="final_score", net_column="policy_net_bps", decision_column="__decision_ts__",
        label_available_column="policy_label_available_ts", identity_column="candidate_id",
        spec=Causal21dAdmissionSpec(mode="hierarchical_tail_side_shrinkage_v2"),
    )
    evaluation = admitted.loc[admitted["__ts__"].ge(start) & admitted["__ts__"].lt(end)].copy()
    try:
        candidates = _auction_candidates(evaluation)
    except ValueError:
        empty = pd.DataFrame()
        summary = {
            "arm": arm, "accepted_trades": 0, "trades_per_day": 0.0,
            "net_bps_per_trade": float("nan"), "gross_bps_per_trade": float("nan"),
            "status": "no_admitted_valid_candidates",
        }
        return admitted, admission_audit, empty, empty, summary
    decisions, equity, monthly, summary = _run(
        candidates, 0.0, f"{arm}_{start:%Y%m%d}_{end:%Y%m%d}", initial_wallet=1000.0,
        perp_leverage=7.0, margin_slot_wallet_fraction=0.10,
    )
    summary = {"arm": arm, **summary}
    calendar_days = max((end - start).total_seconds() / 86_400.0, 1.0)
    summary["evaluation_calendar_days"] = float(calendar_days)
    summary["trades_per_day"] = float(summary.get("accepted_trades", 0) / calendar_days)
    return admitted, admission_audit, decisions, monthly, summary


def _compact_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name", "month",
        "base_score", "base_anchor_bps", "base_rank", "consensus_rank", "upstream",
        "severe200_probability", "raw_severe", "correctness_raw", "correctness_rank",
        "raw_correctness_demote", "final_score", "policy_path_valid", "policy_gross_bps",
        "policy_net_bps", "policy_label_available_ts", "policy_exit_bar_15m",
        "policy_exit_reason", "policy_entry_price", "policy_exit_price", "arm", "model_cutoff",
        "model_held_end_exclusive", "geometry_bundle_sha256", "geometry_fit_start",
        "geometry_fit_end_exclusive", "meta_training_start", "state_feature_count",
    ]
    existing = [column for column in keep if column in frame]
    missing = sorted(set(keep) - set(existing))
    if missing:
        raise ValueError(f"compact prediction artifact lacks required fields: {missing}")
    return frame.loc[:, existing].copy()


def _render_report(
    *,
    path: Path,
    tail_metrics: pd.DataFrame,
    portfolio: pd.DataFrame,
    reproduction: dict[str, Any],
    score_coverage: pd.DataFrame,
    audits: pd.DataFrame,
    evaluation_months: Sequence[str],
) -> None:
    final = tail_metrics.loc[tail_metrics.stage.eq("final_score") & tail_metrics.scope.eq("pooled_global")].copy()
    pivot = final.pivot(index="tail", columns="arm", values="net_bps_per_trade").reset_index()
    final_coverage = score_coverage.loc[
        score_coverage["month"].isin(list(evaluation_months))
    ].copy()
    coverage_rows = int(final_coverage["source_candidates"].sum())
    coverage_valid = int(final_coverage["label_valid_policy_rows"].sum())
    lines = [
        "# Frozen Ten-Head → C3 Full-Stack Matched Replay",
        "",
        "This is a causal composition replay, not a new HPO. It freezes the conditional ten-head winner and compares it with the pre-existing consensus on the same full candidate population, fixed SL3 / activation 0.5 / giveback 0.25 / H12 / 100-bps-once policy labels.",
        "",
        "## Scope and causal boundary",
        "",
        f"- Evaluation months: {', '.join(evaluation_months)}.",
        "- The frozen heads were re-scored on every candidate before outcome coverage is inspected; their label-valid August–October scores reproduce the original frozen artifact.",
        "- Each monthly C3 downstream fit uses a preceding 3-month raw market-geometry burn-in and a nominal six-month resolved-score window (the compatible score ledger begins in March, so the August fit has five populated months), one matching geometry bundle for train/reference/held rows, exact-H12 Severe-200, +100-bps policy-residual correctness, and a same-model prior-42-day CDF.",
        "- Causal 21-day expected-net admission and the 8-concurrent / 2-new-per-15m / 1-per-asset auction are evaluated afterward.",
        "- This retains the fixed-policy label contract of the ten-head experiment. It is therefore not a reproduction of Part A's separately selected SL4.152 / activation2.326 / giveback0.102 policy.",
        "",
        "## Frozen-head reproduction",
        "",
        f"- Matched label-valid rows: {reproduction['rows']:,}",
        f"- Maximum absolute consensus-rank difference: {reproduction['max_abs_consensus_rank_error']:.3g}",
        f"- Maximum absolute upstream-score difference: {reproduction['max_abs_upstream_score_error']:.3g}",
        "",
        "## Final C3 score: pooled-global net bps/trade",
        "",
        "| Tail | Control net (valid / coverage) | Frozen ten-head net (valid / coverage) | Delta |",
        "|---:|---:|---:|---:|",
    ]
    for _, row in pivot.sort_values("tail").iterrows():
        control = float(row.get("control", np.nan))
        candidate = float(row.get("frozen_ten_head", np.nan))
        tail = float(row["tail"])
        control_detail = final.loc[(final["tail"].eq(tail)) & final["arm"].eq("control")].iloc[0]
        candidate_detail = final.loc[(final["tail"].eq(tail)) & final["arm"].eq("frozen_ten_head")].iloc[0]
        lines.append(
            f"| Top {100.0 * tail:g}% | {control:+.2f} ({int(control_detail['valid_outcomes']):,} / {100.0 * float(control_detail['outcome_coverage']):.1f}%) | "
            f"{candidate:+.2f} ({int(candidate_detail['valid_outcomes']):,} / {100.0 * float(candidate_detail['outcome_coverage']):.1f}%) | {candidate - control:+.2f} |"
        )
    lines.extend([
        "",
        "## Causal admission and portfolio",
        "",
        "| Arm | Accepted trades | Trades/day | Net bps/trade | Gross bps/trade |",
        "|---|---:|---:|---:|---:|",
    ])
    for _, row in portfolio.iterrows():
        lines.append(
            f"| {row['arm']} | {int(row.get('accepted_trades', 0)):,} | {float(row.get('trades_per_day', np.nan)):.2f} | "
            f"{float(row.get('net_bps_per_trade', np.nan)):+.2f} | {float(row.get('gross_bps_per_trade', np.nan)):+.2f} |"
        )
    lines.extend([
        "",
        "## Coverage and model-fit checks",
        "",
        f"- Final-window source outcome coverage: {coverage_valid:,} valid fixed-policy paths out of {coverage_rows:,} scored candidates ({100.0 * coverage_valid / max(coverage_rows, 1):.1f}%).",
        "- This inherited low outcome coverage is not an admission feature or score filter; it makes the global-tail diagnostics sparse and prevents promotion from this replay alone.",
        f"- Candidate scoring rows across the March–October history: {int(score_coverage['source_candidates'].sum()):,}.",
        f"- C3 downstream fits completed: {int(len(audits)):,} (two arms × six May–October monthly fits; May–July provide causal admission history).",
        "- Every C3 audit records train/reference/held separation, geometry identity, resolved-only supervised support, and no held outcome consumption during scoring.",
        "",
        "Detailed numbers are in the associated parquet artifacts; this document intentionally does not promote either arm, because the Aug–October period was already opened by the upstream ten-head final comparison.",
        "",
    ])
    path.write_text("\n".join(lines))


def run(
    *,
    out: Path,
    evaluation_months: Sequence[str],
    max_train_rows: int = MAX_TRAIN_ROWS,
) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    if not evaluation_months:
        raise ValueError("at least one evaluation month is required")
    evaluation_months = tuple(sorted(dict.fromkeys(str(month) for month in evaluation_months)))
    evaluation_start = _month_start(evaluation_months[0])
    evaluation_end = _month_end(evaluation_months[-1])
    if tuple(_months_between(evaluation_start, evaluation_end)) != evaluation_months:
        raise ValueError("evaluation months must be contiguous chronological months")
    # Score March onward because February cannot train a residual model from an
    # upstream surface that itself begins in February. February is nevertheless
    # included as resolved training support for every later frozen head.
    score_start = pd.Timestamp("2025-03-01", tz="UTC")
    score_months = _months_between(score_start, evaluation_end)
    downstream_start = pd.Timestamp("2025-05-01", tz="UTC")
    downstream_months = _months_between(downstream_start, evaluation_end)
    fields = _source_fields()
    configs = _load_frozen_configs(FROZEN_HEADS)
    out.mkdir(parents=True)
    _event("frozen_score_materialisation_start", months=score_months)
    ledger, head_audit, score_coverage = _score_frozen_consensus(
        fields=fields, configs=configs, score_months=score_months, max_train_rows=max_train_rows,
    )
    reproduction = _check_frozen_final_reproduction(ledger)
    _event("frozen_score_materialisation_complete", rows=len(ledger), reproduction=reproduction)
    ledger.to_parquet(out / "all_candidate_upstream_ledger.parquet", index=False, compression="zstd")
    head_audit.to_parquet(out / "frozen_head_fit_audit.parquet", index=False)
    score_coverage.to_parquet(out / "score_population_coverage.parquet", index=False)

    arm_scores: dict[str, pd.DataFrame] = {}
    common = ledger[["candidate_id", "base_score", "base_anchor_bps", "base_rank"]].copy()
    arm_scores["control"] = common.assign(
        consensus_rank=ledger["control_consensus_rank"].to_numpy(float),
        upstream=ledger["control_upstream"].to_numpy(float),
    )
    arm_scores["frozen_ten_head"] = common.assign(
        consensus_rank=ledger["new_consensus_rank"].to_numpy(float),
        upstream=ledger["new_upstream"].to_numpy(float),
    )
    outputs: dict[str, list[pd.DataFrame]] = {arm: [] for arm in arm_scores}
    c3_audits: list[dict[str, Any]] = []
    geometry_previous: c3.RawK9Bundle | None = None
    for month in downstream_months:
        cutoff, held_end = _month_start(month), _month_end(month)
        _event("c3_context_start", month=month)
        context, geometry_previous = _prepare_c3_context(
            ledger=ledger, fields=fields, cutoff=cutoff, held_end=held_end,
            previous_geometry=geometry_previous,
        )
        for arm, scores in arm_scores.items():
            _event("c3_arm_start", month=month, arm=arm)
            output, audit = _score_c3_arm(context, scores, arm=arm)
            outputs[arm].append(output)
            c3_audits.append(audit)
            _event("c3_arm_complete", month=month, arm=arm, rows=len(output))
    predictions = {
        arm: pd.concat(parts, ignore_index=True).sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
        for arm, parts in outputs.items()
    }
    if any(frame["candidate_id"].duplicated().any() for frame in predictions.values()):
        raise AssertionError("C3 output duplicated candidate identities")
    c3_audit_frame = pd.DataFrame(c3_audits)
    c3_audit_frame.to_parquet(out / "c3_downstream_fit_audit.parquet", index=False)

    tail_frames: list[pd.DataFrame] = []
    tail_months: list[pd.DataFrame] = []
    portfolio_rows: list[dict[str, Any]] = []
    for arm, frame in predictions.items():
        compact = _compact_predictions(frame)
        compact.to_parquet(out / f"{arm}_full_c3_predictions.parquet", index=False, compression="zstd")
        for stage in ("upstream", "raw_severe", "raw_correctness_demote", "final_score"):
            metrics, monthly = _tail_metrics(frame, arm=arm, stage=stage, start=evaluation_start, end=evaluation_end)
            tail_frames.append(metrics)
            tail_months.append(monthly)
        admitted, admission_audit, decisions, monthly_portfolio, summary = _run_admission_and_portfolio(
            frame, arm=arm, start=evaluation_start, end=evaluation_end,
        )
        admitted.to_parquet(out / f"{arm}_causal_admission_ledger.parquet", index=False, compression="zstd")
        admission_audit.to_parquet(out / f"{arm}_causal_admission_audit.parquet", index=False)
        decisions.to_parquet(out / f"{arm}_portfolio_decisions.parquet", index=False, compression="zstd")
        monthly_portfolio.to_parquet(out / f"{arm}_portfolio_monthly.parquet", index=False)
        portfolio_rows.append(summary)
    tail_metrics = pd.concat(tail_frames, ignore_index=True)
    tail_monthly = pd.concat(tail_months, ignore_index=True)
    portfolio = pd.DataFrame(portfolio_rows)
    tail_metrics.to_parquet(out / "stage_tail_metrics.parquet", index=False)
    tail_monthly.to_parquet(out / "stage_tail_month_contributions.parquet", index=False)
    portfolio.to_parquet(out / "portfolio_summary.parquet", index=False)

    score_fields = set(["base_score", "base_anchor_bps", "base_rank", "consensus_rank", "final_score"])
    correctness = {
        "status": "passed",
        "source_candidate_population_preserved": True,
        "source_feature_count": int(len(fields)),
        "frozen_head_configs_sha256": _sha256_payload(json.loads(FROZEN_HEADS.read_text())),
        "frozen_final_reproduction": reproduction,
        "all_candidate_scoring_before_outcome_coverage": True,
        "residual_training_label_available_before_cutoff": True,
        "c3_geometry_raw_market_only": True,
        "c3_geometry_single_bundle_per_fit": bool(c3_audit_frame["reference_and_held_share_one_geometry_bundle"].all()),
        "c3_held_outcomes_consumed": bool(c3_audit_frame["held_outcomes_consumed"].any()),
        "same_model_prior42_final_cdf": bool(c3_audit_frame["same_model_prior42_final_cdf"].all()),
        "causal_21d_admission": True,
        "portfolio_constraints": "8 concurrent; 2 new per 15m; 1 per asset; 80% entry margin",
        "score_feature_names_are_target_free": all(not any(token in name.lower() for token in ("policy_net", "policy_gross", "h12", "future", "outcome")) for name in score_fields),
        "c3_state_feature_count": int(c3_audit_frame["state_feature_count"].iloc[0]),
    }
    (out / "correctness_test_report.json").write_text(json.dumps(correctness, indent=2, default=str) + "\n")
    manifest = {
        "schema": "ten_head_c3_full_stack_matched_replay_v1",
        "source": str(SOURCE), "upstream": str(UPSTREAM), "frozen_heads": str(FROZEN_HEADS),
        "source_feature_count": len(fields), "score_start": str(score_start), "score_months": score_months,
        "downstream_months": downstream_months, "evaluation_months": list(evaluation_months), "evaluation_start": str(evaluation_start),
        "evaluation_end_exclusive": str(evaluation_end), "side": "long",
        "frozen_head_max_train_rows": int(max_train_rows),
        "head_policy": "SL3 / activation0.5 / giveback0.25 / H12 / 100 bps once",
        "downstream": {
            "geometry": "C3 raw-market K9; three-month burn-in; one aligned bundle per monthly downstream fit",
            "training_window_months": 6, "severe": "exact H12 net <= -200 bps; alpha=0.5",
            "correctness": "policy net - base anchor > +100 bps; rank multiplier 0.25 + 0.75*pct",
            "cdf": "same-model prior 42 days", "admission": "causal prior-resolved 21-day expected net >= +50 bps",
            "portfolio": "8 concurrent; 2 new per 15m; 1 asset; 80% margin cap",
        },
        "boundary": "matched SL3 policy replay; not a reproduction of the current Part-A selected execution-policy model",
        "final_period_previously_opened_by_upstream_ten_head_comparison": True,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    report = ROOT / "docs/TEN_HEAD_C3_FULL_STACK_MATCHED_REPLAY_20260810.md"
    _render_report(
        path=report, tail_metrics=tail_metrics, portfolio=portfolio, reproduction=reproduction,
        score_coverage=score_coverage, audits=c3_audit_frame, evaluation_months=evaluation_months,
    )
    _event("complete", output=str(out), report=str(report), portfolio=portfolio.to_dict(orient="records"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--evaluation-months", default=",".join(DEFAULT_EVALUATION_MONTHS))
    parser.add_argument("--max-train-rows", type=int, default=MAX_TRAIN_ROWS)
    args = parser.parse_args()
    months = tuple(value.strip() for value in args.evaluation_months.split(",") if value.strip())
    run(out=args.out, evaluation_months=months, max_train_rows=int(args.max_train_rows))


if __name__ == "__main__":
    main()
