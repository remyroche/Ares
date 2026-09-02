#!/usr/bin/env python3
"""Rebuild the Stage-3 C2 conversion coordinate under the frozen :00 stack.

This is the first executable part of Stage 4.  It starts with the retained
canonical C0 control and the one development-selected C2 consensus challenger
(``rho=.50``, ``temperature=.05``), then rebuilds only C2's correctness model
and same-model prior-28-day CDF.  The D2 base, causal anchor, ten residual
heads, 120-field order, and frozen Oct--Dec Geometry/K9 parent are reused
unchanged.  The resulting panel remains target-free; policy labels are used
only to fit C2's conversion bundles and are never persisted with scores.

It is offline, long-only and :00-only.  It does not read a live bundle or
write any inference, admission, portfolio, exchange, or exit artifact.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_canonical_current import (  # noqa: E402
    FrozenGeometryK9View,
    K9_CLUSTERS,
    MODEL_CAP,
    SEED,
    _aggregate_state_fields,
    _canonical_ldf_geometry_aliases,
    _equal_month_sample,
    _fit_correctness,
    score_four_week_conversion_bundle,
    score_monthly_upstream_bundle,
)


CONTROL_ROOT = ROOT / (
    "data_perp/artifacts/strict_r3_score_family_current_v5_canonical_policy_"
    "reconstruction_2025_2026_20260816_v4"
)
SOURCE_LEDGER = ROOT / (
    "data_perp/artifacts/strict_r3_schema_v2_prequential_ledger_targetfree_long_"
    "2024_2026_20260809_v1/prequential_stack_ledger.parquet"
)
STAGE3_ROOT = ROOT / "data_perp/artifacts/strict_r3_stage3_consensus_screen_20260823_v1"
LABELS = ROOT / (
    "data_perp/artifacts/strict_r3_long_base_recall_funnel_2025dev_holdout_"
    "2026oos_20260822_v1/outcome_joined_recall_ledger.parquet"
)
DEFAULT_OUT = ROOT / "data_perp/artifacts/strict_r3_stage4_c2_conversion_20260823_v1"

RESERVE_DAYS = 28
ROUTE_FRACTION = 0.30
C2_COLUMN = "consensus__c2_r50_t05"
C2_UPSTREAM = "upstream__c2_r50_t05"
IDENTITY = ("candidate_id", "__decision_ts__", "side_name")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _block_token(path: Path) -> str:
    name = path.parents[1].name.removeprefix("block=")
    return name


def _read_stage3_weights(path: Path) -> dict[str, dict[str, float]]:
    audit = pd.read_parquet(path)
    candidate = audit.loc[audit["arm"].eq(C2_COLUMN)].copy()
    if candidate.empty:
        raise ValueError(f"Stage 3 audit has no {C2_COLUMN} arm")
    if not candidate["effective_head_count_passed"].fillna(False).astype(bool).all():
        raise ValueError("Stage 3 C2 has a sub-three effective-head block")
    result: dict[str, dict[str, float]] = {}
    for row in candidate.itertuples(index=False):
        token = str(row.control_block).removeprefix("block=")
        result[token] = {key: float(value) for key, value in json.loads(row.weights).items()}
    return result


def _head_columns(frame: pd.DataFrame) -> list[str]:
    columns = sorted(
        column for column in frame
        if column.startswith("conditional_head__") and column.endswith("__rank")
    )
    if len(columns) != 10:
        raise ValueError(f"expected ten frozen residual-head rank fields, found {len(columns)}")
    return columns


def _apply_c2(frame: pd.DataFrame, weights: dict[str, float], *, route_required: bool) -> pd.DataFrame:
    """Apply the block's already strict-prequential C2 weights target-free."""

    result = frame.copy()
    heads = _head_columns(result)
    if set(heads) != set(weights):
        raise ValueError("C2 head-weight identity differs from the frozen residual contract")
    route = result["base_route_timestamp_top30"].fillna(False).astype(bool).to_numpy()
    matrix = result.loc[:, heads].to_numpy(dtype=float, copy=False)
    vector = np.asarray([weights[head] for head in heads], dtype=float)
    consensus = matrix @ vector
    consensus[~route] = np.nan
    result["conditional_consensus_rank"] = consensus
    result["upstream"] = np.where(
        route,
        .75 * pd.to_numeric(result["base_rank42"], errors="coerce").to_numpy(float) + .25 * consensus,
        np.nan,
    )
    if route_required and not route.any():
        raise ValueError("held scorer produced no timestamp-local routed rows")
    return result


def _load_stage3_lookup(path: Path) -> pd.DataFrame:
    columns = [
        "candidate_id", "__decision_ts__", "base_score", "base_rank42", "base_anchor_bps",
        C2_COLUMN, C2_UPSTREAM,
    ]
    frame = pd.read_parquet(path, columns=columns)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    if frame["candidate_id"].duplicated().any():
        raise ValueError("Stage 3 candidate identities are not unique")
    return frame


def _load_labels(path: Path) -> pd.DataFrame:
    columns = [
        "candidate_id", "policy_path_valid", "policy_net_bps", "policy_label_available_ts",
    ]
    frame = pd.read_parquet(path, columns=columns)
    frame["policy_label_available_ts"] = pd.to_datetime(
        frame["policy_label_available_ts"], utc=True, errors="coerce",
    )
    if frame["candidate_id"].duplicated().any():
        raise ValueError("canonical policy labels are not one-to-one by candidate")
    return frame


def _training_metadata(
    *,
    cutoff: pd.Timestamp,
    base_fields: tuple[str, ...],
    stage3: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Construct strict-prequential conversion training inputs for one block.

    October--December 2024 has no frozen individual-head reconstruction, so
    it keeps the inherited prequential C0 coordinate.  From January 2025,
    every replacement coordinate is the published strict-prequential C2 held
    score.  This is a disclosed warm-up fallback, never a non-OOF substitute.
    """

    reserve_start = cutoff - pd.Timedelta(days=RESERVE_DAYS)
    train_start = cutoff - pd.DateOffset(months=6)
    needed = [
        "candidate_id", "__decision_ts__", "side_name", "r3_class", "r3_label_available_ts",
        "prequential_base_score", "prequential_base_rank42", "prequential_base_anchor_bps",
        "prequential_consensus_rank", "prequential_upstream", "stack_is_prequential", *base_fields,
    ]
    # The raw 120-field panel is intentionally *not* loaded here.  This pass
    # is only for strict label chronology and deterministic equal-month
    # sampling; fields are streamed later only for the selected identities.
    needed = [field for field in needed if field not in set(base_fields)]
    raw = pd.read_parquet(
        SOURCE_LEDGER, columns=needed,
        filters=[("__decision_ts__", ">=", train_start), ("__decision_ts__", "<", reserve_start)],
    )
    raw["__decision_ts__"] = pd.to_datetime(raw["__decision_ts__"], utc=True, errors="raise")
    raw["r3_label_available_ts"] = pd.to_datetime(raw["r3_label_available_ts"], utc=True, errors="raise")
    raw = raw.merge(labels, on="candidate_id", how="left", validate="one_to_one")
    raw["policy_path_valid"] = raw["policy_path_valid"].fillna(False).astype(bool)
    raw = raw.merge(
        stage3,
        on=["candidate_id", "__decision_ts__"], how="left", validate="one_to_one", suffixes=("", "__stage3"),
    )
    raw["base_score"] = raw["base_score"].where(raw["base_score"].notna(), raw["prequential_base_score"])
    raw["base_rank42"] = raw["base_rank42"].where(raw["base_rank42"].notna(), raw["prequential_base_rank42"])
    raw["base_anchor_bps"] = raw["base_anchor_bps"].where(raw["base_anchor_bps"].notna(), raw["prequential_base_anchor_bps"])
    raw["conditional_consensus_rank"] = raw[C2_COLUMN].where(
        raw[C2_COLUMN].notna(), raw["prequential_consensus_rank"],
    )
    raw["upstream"] = raw[C2_UPSTREAM].where(raw[C2_UPSTREAM].notna(), raw["prequential_upstream"])
    raw = raw.drop(columns=[
        "prequential_base_score", "prequential_base_rank42", "prequential_base_anchor_bps",
        "prequential_consensus_rank", "prequential_upstream", "base_score__stage3",
        "base_rank42__stage3", "base_anchor_bps__stage3", C2_COLUMN, C2_UPSTREAM,
    ], errors="ignore")
    if raw["candidate_id"].duplicated().any() or not raw["stack_is_prequential"].fillna(False).astype(bool).all():
        raise ValueError("C2 conversion training input lost one-to-one strict-prequential lineage")
    return raw


def _source_chunks(
    *, start: pd.Timestamp, end: pd.Timestamp, base_fields: tuple[str, ...], days: int = 7,
) -> Iterable[pd.DataFrame]:
    """Yield complete target-free source slices without a wide nine-month load."""

    cursor = start
    columns = [*IDENTITY, *base_fields]
    while cursor < end:
        stop = min(cursor + pd.Timedelta(days=days), end)
        piece = pd.read_parquet(
            SOURCE_LEDGER,
            columns=columns,
            filters=[("__decision_ts__", ">=", cursor), ("__decision_ts__", "<", stop)],
        )
        piece["__decision_ts__"] = pd.to_datetime(
            piece["__decision_ts__"], utc=True, errors="raise",
        )
        if piece["candidate_id"].duplicated().any():
            raise AssertionError("target-free source slice repeated candidate identities")
        yield piece.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
        cursor = stop


def _stream_state(
    *,
    conversion: Any,
    start: pd.Timestamp,
    end: pd.Timestamp,
    base_fields: tuple[str, ...],
    retain_ids: set[str],
) -> pd.DataFrame:
    """Materialise only required aggregate state while advancing K9 causally.

    Geometry/K9 membership meanings are immutable: this constructs a local
    chronological runtime history from the frozen parent, then discards each
    target-free source slice.  Leaf trust is the unchanged C0 artifact because
    it depends only on strict-R3/R3 inputs, not on the C2 coordinate.
    """

    parent = copy.copy(conversion.geometry.parent)
    history = parent.state_history.copy()
    history["__decision_ts__"] = pd.to_datetime(history["__decision_ts__"], utc=True, errors="raise")
    parent.state_history = history.loc[history["__decision_ts__"].lt(start)].copy().reset_index(drop=True)
    view = FrozenGeometryK9View(parent=parent, temperature_scale=conversion.geometry.temperature_scale)
    membership_columns = [f"k09__cluster_{index:02d}__membership" for index in range(K9_CLUSTERS)]
    retained: list[pd.DataFrame] = []
    for piece in _source_chunks(start=start, end=end, base_fields=base_fields):
        state = _canonical_ldf_geometry_aliases(
            pd.concat(
                [view.transform(piece), conversion.leaf_trust.transform(piece)],
                axis=1,
            ),
        )
        aggregate = _aggregate_state_fields(state)
        selected = piece["candidate_id"].isin(retain_ids).to_numpy()
        if selected.any():
            retained.append(pd.concat(
                [
                    piece.loc[selected, list(IDENTITY)].reset_index(drop=True),
                    state.loc[selected, list(aggregate)].reset_index(drop=True),
                ],
                axis=1,
            ))
        # Advance only after the whole slice has been transformed, so every
        # timestamp sees strictly preceding completed target-free state.
        mass = state.loc[:, membership_columns].copy()
        mass.columns = [f"k{index}" for index in range(K9_CLUSTERS)]
        mass["__decision_ts__"] = piece["__decision_ts__"].to_numpy()
        event = mass.groupby("__decision_ts__", sort=True).sum().reset_index()
        parent.state_history = pd.concat([parent.state_history, event], ignore_index=True).sort_values(
            "__decision_ts__", kind="stable",
        ).drop_duplicates("__decision_ts__", keep="last").reset_index(drop=True)
    output = pd.concat(retained, ignore_index=True)
    if output["candidate_id"].duplicated().any():
        raise AssertionError("streamed conversion state repeated a candidate identity")
    return output


def _score_source_window(
    *,
    cutoff: pd.Timestamp,
    end: pd.Timestamp,
    base_fields: tuple[str, ...],
    upstream: Any,
    weights: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reserve_start = cutoff - pd.Timedelta(days=RESERVE_DAYS)
    source = pd.read_parquet(
        SOURCE_LEDGER,
        columns=[*IDENTITY, *base_fields],
        filters=[("__decision_ts__", ">=", reserve_start), ("__decision_ts__", "<", end)],
    )
    source["__decision_ts__"] = pd.to_datetime(source["__decision_ts__"], utc=True, errors="raise")
    reference = source.loc[source["__decision_ts__"].lt(cutoff)].copy()
    held = source.loc[source["__decision_ts__"].ge(cutoff)].copy()
    if reference.empty or held.empty:
        raise ValueError("C2 same-model reference or held population is empty")
    reference_upstream = score_monthly_upstream_bundle(
        upstream, reference, allow_prior_reference=True, prior_reference_start=reserve_start,
        route_top_fraction=None,
    )
    held_upstream = score_monthly_upstream_bundle(
        upstream, held, allow_prior_reference=True, prior_reference_start=reserve_start,
        route_top_fraction=ROUTE_FRACTION,
    )
    scored_reference = _apply_c2(
        reference.merge(reference_upstream, on=list(IDENTITY), how="left", validate="one_to_one"),
        weights, route_required=False,
    )
    scored_held = _apply_c2(
        held.merge(held_upstream, on=list(IDENTITY), how="left", validate="one_to_one"),
        weights, route_required=True,
    )
    return scored_reference, scored_held


def _block_paths() -> list[Path]:
    paths = sorted(CONTROL_ROOT.glob("bundles/block=*/upstream/monthly_upstream_bundle.joblib"))
    if not paths:
        raise FileNotFoundError("no frozen current upstream bundles")
    return paths


def _run_block(
    bundle_path: Path,
    *,
    stage3: pd.DataFrame,
    labels: pd.DataFrame,
    weights_by_block: dict[str, dict[str, float]],
    output_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    token = _block_token(bundle_path)
    weight_token = token.removesuffix("_finalcoverage")
    if weight_token not in weights_by_block:
        raise KeyError(f"Stage 3 audit has no C2 weights for {token}")
    upstream = joblib.load(bundle_path)
    cutoff = _utc(upstream.cutoff)
    end = _utc(upstream.end_exclusive)
    base_fields = tuple(upstream.base_fields)
    original_conversion_path = bundle_path.parents[1] / "conversion/four_week_conversion_bundle.joblib"
    original_conversion = joblib.load(original_conversion_path)
    training = _training_metadata(
        cutoff=cutoff, base_fields=base_fields, stage3=stage3, labels=labels,
    )
    reference, held = _score_source_window(
        cutoff=cutoff, end=end, base_fields=base_fields, upstream=upstream,
        weights=weights_by_block[weight_token],
    )
    reserve_start = cutoff - pd.Timedelta(days=RESERVE_DAYS)
    train_start = cutoff - pd.DateOffset(months=6)
    # This is exactly the supervised conversion population used by the
    # canonical fitter, except that it stays narrow until the deterministic
    # equal-month sample is known.  The preceding 28-day reserve is excluded.
    meta = training.loc[
        training["__decision_ts__"].ge(train_start)
        & training["__decision_ts__"].lt(reserve_start)
        & training["policy_label_available_ts"].lt(reserve_start)
        & np.isfinite(pd.to_numeric(training["policy_net_bps"], errors="coerce"))
    ].copy()
    if meta.empty:
        raise ValueError("C2 conversion has no strict-resolved policy training population")
    meta_fit = _equal_month_sample(meta, MODEL_CAP, seed=SEED + 5001)
    retain_ids = set(meta_fit["candidate_id"].astype(str))
    retain_ids.update(reference["candidate_id"].astype(str))
    retain_ids.update(held["candidate_id"].astype(str))
    state = _stream_state(
        conversion=original_conversion,
        start=train_start,
        end=end,
        base_fields=base_fields,
        retain_ids=retain_ids,
    )
    meta_state = meta_fit.merge(state, on=list(IDENTITY), how="left", validate="one_to_one")
    state_fields = tuple(column for column in state.columns if column not in IDENTITY)
    if meta_state.loc[:, list(state_fields)].isna().all(axis=1).any():
        raise ValueError("streamed Geometry/K9 state does not cover every sampled conversion row")
    correctness_fields = (
        "base_score", "base_anchor_bps", "base_rank42",
        "conditional_consensus_rank", "upstream", *state_fields,
    )
    correctness = _fit_correctness(meta_state, correctness_fields)
    # Reuse C0 leaf-trust, frozen Geometry/K9 and shadow Severe untouched.
    # All three are C2-independent.  Only the supervision consuming C2's
    # coordinate is re-fit, eliminating the prior nine-month wide-frame peak.
    conversion = copy.copy(original_conversion)
    conversion.correctness = correctness
    conversion.manifest = dict(original_conversion.manifest)
    conversion.manifest.update({
        "stage3_arm": C2_COLUMN,
        "stage3_c2_weights": weights_by_block[weight_token],
        "stage3_weight_block": weight_token,
        "pre2025_c2_history": "inherited strict-prequential C0 warm-up only",
        "c2_conversion_rebuild": "streamed_geometry_state; C0 leaf-trust/severe unchanged because C2-independent",
        "calibration_reserve_days": RESERVE_DAYS,
        "calibration_reserve_start": reserve_start.isoformat(),
        "source_hashes": {
            "stage3_target_free": _sha(STAGE3_ROOT / "stage3_target_free_scores.parquet"),
            "stage3_prequential_audit": _sha(STAGE3_ROOT / "strict_prequential_audit.parquet"),
            "canonical_policy_labels": _sha(LABELS),
            "frozen_upstream_bundle": _sha(bundle_path),
        },
    })
    score_ids = pd.concat(
        [reference["candidate_id"], held["candidate_id"]], ignore_index=True,
    )
    score_state = state.loc[state["candidate_id"].isin(score_ids)].copy()
    converted, audit = score_four_week_conversion_bundle(
        conversion, reference=reference, held=held, precomputed_state=score_state,
    )
    held_converted = converted.loc[converted["__score_role__"].eq("held")].copy()
    heads = _head_columns(held)
    auxiliary = held.loc[:, [
        "candidate_id", "base_route_timestamp_top30", *heads,
    ]].copy()
    held_converted = held_converted.merge(auxiliary, on="candidate_id", how="left", validate="one_to_one")
    matrix = held_converted.loc[:, heads].to_numpy(float, copy=False)
    held_converted["stage3_c2_head_iqr"] = np.nanpercentile(matrix, 75, axis=1) - np.nanpercentile(matrix, 25, axis=1)
    held_converted["stage3_c2_head_mad"] = np.nanmedian(
        np.abs(matrix - np.nanmedian(matrix, axis=1, keepdims=True)), axis=1,
    )
    keep = [
        "candidate_id", "__decision_ts__", "side_name", "base_score", "base_rank42", "base_anchor_bps",
        "conditional_consensus_rank", "upstream", "ordinary_shadow_consensus_rank",
        "ordinary_shadow_upstream", "correctness_rank", "raw_correctness_demote", "final_score",
        "base_route_timestamp_top30", "stage3_c2_head_iqr", "stage3_c2_head_mad", *heads,
        "geometry_bundle_sha256", "conversion_bundle_sha256",
    ]
    result = held_converted.loc[:, keep].copy()
    forbidden = {"policy_net_bps", "policy_path_valid", "policy_label_available_ts"}
    if forbidden & set(result.columns):
        raise AssertionError("target-free C2 conversion output contains a policy label")
    block_dir = output_root / "c2_conversion_bundles" / f"block={token}"
    block_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(conversion, block_dir / "four_week_conversion_bundle.joblib")
    (block_dir / "run_manifest.json").write_text(json.dumps(conversion.manifest, indent=2, sort_keys=True) + "\n")
    result.to_parquet(block_dir / "held_target_free_scores.parquet", index=False, compression="zstd")
    return result, {
        "control_block": token,
        "cutoff": cutoff.isoformat(), "end_exclusive": end.isoformat(),
        "training_rows": int(len(meta_fit)), "held_rows": int(len(result)),
        "reference_rows": int(len(reference)), "c2_effective_head_count": float(1.0 / sum(value * value for value in weights_by_block[weight_token].values())),
        "conversion_audit": audit.to_dict(orient="records"),
        "geometry_bundle_sha256": conversion.geometry.bundle_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--start", default="2025-01-01T00:00:00Z")
    parser.add_argument("--end", default="2026-08-01T00:00:00Z")
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    start, end = _utc(args.start), _utc(args.end)
    weights = _read_stage3_weights(STAGE3_ROOT / "strict_prequential_audit.parquet")
    stage3 = _load_stage3_lookup(STAGE3_ROOT / "stage3_target_free_scores.parquet")
    labels = _load_labels(LABELS)
    args.out_dir.mkdir(parents=True)
    outputs: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for path in _block_paths():
        token = _block_token(path)
        cutoff = _utc(joblib.load(path).cutoff)
        if cutoff < start or cutoff >= end:
            continue
        print(json.dumps({"event": "c2_conversion_block_start", "block": token}), flush=True)
        try:
            scored, audit = _run_block(
                path, stage3=stage3, labels=labels, weights_by_block=weights, output_root=args.out_dir,
            )
        except ValueError as error:
            # The first eligible block can legitimately lack six months of
            # resolved canonical policy labels.  It cannot receive an
            # invented conversion model, so preserve the absence explicitly
            # and begin only when strict supervised support exists.
            if "no strict-resolved policy training population" not in str(error):
                raise
            audits.append({
                "control_block": token,
                "cutoff": cutoff.isoformat(),
                "status": "skipped_insufficient_strict_prequential_policy_support",
            })
            print(json.dumps({"event": "c2_conversion_block_skipped", "block": token, "reason": "insufficient_prequential_policy_support"}), flush=True)
            continue
        outputs.append(scored)
        audits.append(audit)
        print(json.dumps({"event": "c2_conversion_block_complete", "block": token, "rows": len(scored)}), flush=True)
    if not outputs:
        raise ValueError("no Stage 4 C2 block had sufficient strict prequential policy support")
    target_free = pd.concat(outputs, ignore_index=True).sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable",
    ).reset_index(drop=True)
    if target_free["candidate_id"].duplicated().any() or target_free["__decision_ts__"].dt.minute.ne(0).any():
        raise AssertionError("Stage 4 target-free C2 output is not a unique :00 population")
    target_free.to_parquet(args.out_dir / "current_c2_target_free_scores.parquet", index=False, compression="zstd")
    pd.DataFrame(audits).to_parquet(args.out_dir / "c2_conversion_audit.parquet", index=False)
    manifest = {
        "schema": "strict_r3_stage4_c2_conversion_rebuild_v1",
        "scope": "offline long-only :00-only; no live/canonical/admission/portfolio/exit artifact modified",
        "stage3_arm": C2_COLUMN,
        "score_coordinate": "frozen D2/base-anchor/ten heads; C2 reliability-weighted consensus; rebuilt correctness and same-model prior-28-day CDF",
        "policy": "canonical source-aligned frozen rich parent policy net; invalid paths used only for training exclusion",
        "reserve_days": RESERVE_DAYS,
        "geometry": "frozen Oct-Dec 2024 parent reused; never refit",
        "inputs": {
            "source_ledger": {"path": str(SOURCE_LEDGER), "sha256": _sha(SOURCE_LEDGER)},
            "stage3_scores": {"path": str(STAGE3_ROOT / 'stage3_target_free_scores.parquet'), "sha256": _sha(STAGE3_ROOT / 'stage3_target_free_scores.parquet')},
            "stage3_audit": {"path": str(STAGE3_ROOT / 'strict_prequential_audit.parquet'), "sha256": _sha(STAGE3_ROOT / 'strict_prequential_audit.parquet')},
            "policy_labels": {"path": str(LABELS), "sha256": _sha(LABELS)},
        },
        "blocks": audits,
        "rows": int(len(target_free)),
        "next_required": "fit current-native and separately BCF-native Stage-4 MC1 maps, then run unchanged dual admission and constrained portfolio",
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"event": "stage4_c2_conversion_complete", "rows": len(target_free)}), flush=True)


if __name__ == "__main__":
    main()
