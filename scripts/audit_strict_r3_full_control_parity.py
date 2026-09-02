#!/usr/bin/env python3
"""Reproduce the frozen long-only :00 strict-R3 control before research.

This is deliberately a *control gate*, not a challenger.  It re-scores the
immutable current-v5 upstream and conversion bundles from the target-free
prequential source.  Geometry/K9's dynamic 28-day state is advanced over a
target-free warm-up span before the same-model reserve, so the reserve itself
does not accidentally begin with a cold geometry state.

No outcome column is read by the scorer.  The stored frozen control is used
only after target-free scores are complete, as the comparison oracle.  A
failure writes an immutable receipt and exits non-zero; downstream research
must not use a failed control receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_canonical_current import (  # noqa: E402
    _aggregate_state_fields,
    _canonical_ldf_geometry_aliases,
    score_four_week_conversion_bundle,
    score_monthly_upstream_bundle,
)


DEFAULT_SOURCE = ROOT / (
    "data_perp/artifacts/strict_r3_schema_v2_prequential_ledger_targetfree_long_"
    "2024_2026_20260809_v1/prequential_stack_ledger.parquet"
)
DEFAULT_CONTROL = ROOT / (
    "data_perp/artifacts/strict_r3_score_family_current_v5_canonical_policy_"
    "reconstruction_2025_2026_20260816_v4"
)
DEFAULT_B0 = ROOT / (
    "data_perp/artifacts/strict_r3_long_base_recall_funnel_2025dev_holdout_"
    "2026oos_20260822_v1"
)
DEFAULT_OUT = ROOT / (
    "data_perp/artifacts/strict_r3_base_recall_residual2_consensus_research_"
    "20260822_v4_control_parity"
)

RESERVE_DAYS = 28
ROUTE_FRACTION = 0.30
NUMERIC_RTOL = 1e-4
NUMERIC_ATOL = 1e-8

CORE_FIELDS = (
    "p_adverse",
    "p_weak",
    "p_clear",
    "base_score",
    "base_rank42",
    "base_anchor_bps",
    "conditional_consensus_rank",
    "upstream",
    "ordinary_shadow_consensus_rank",
    "ordinary_shadow_upstream",
    "correctness_raw",
    "correctness_rank",
    "raw_correctness_demote",
    "final_score",
    "severe200_probability_shadow",
)
BOOLEAN_FIELDS = (
    "base_route_timestamp_top30",
    "correctness_gate_active",
    "severe_affects_final_score",
)
IDENTITY_FIELDS = ("candidate_id", "__decision_ts__", "side_name")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _block_name(path: Path) -> str:
    # ``.../bundles/block=<cutoff>/upstream/monthly_upstream_bundle.joblib``
    # places the immutable block directory two levels above the payload.
    match = re.fullmatch(r"block=(\d{8}T\d{6}Z)(?:_finalcoverage)?", path.parents[1].name)
    if match is None:
        raise ValueError(f"invalid frozen block path: {path}")
    return match.group(1)


def _assert_exact_hour(frame: pd.DataFrame, *, context: str) -> None:
    decision = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    invalid = (decision.dt.minute != 0) | (decision.dt.second != 0) | (decision.dt.microsecond != 0)
    if invalid.any():
        raise AssertionError(
            f"{context} has non-:00 decision timestamps: "
            f"{decision.loc[invalid].head(5).astype(str).tolist()}"
        )


def _score_path(control: Path, block: str) -> Path:
    direct = control / "scores" / f"block={block}.parquet"
    if direct.exists():
        return direct
    terminal = control / "scores" / f"block={block}_finalcoverage.parquet"
    if terminal.exists():
        return terminal
    raise FileNotFoundError(f"no stored frozen score block for {block}")


def _prior_upstream_bundle_path(control: Path, *, cutoff: pd.Timestamp) -> Path | None:
    """Find the frozen upstream producer that ended exactly at ``cutoff``.

    A four-week conversion block is calibrated against the immediately prior
    28-day score domain.  That reserve was historically produced by the
    preceding upstream fit, not retroactively rescored by the held fit.
    """
    for candidate in sorted(control.glob("bundles/block=*/upstream/monthly_upstream_bundle.joblib")):
        predecessor = joblib.load(candidate)
        if _utc(predecessor.end_exclusive) == cutoff:
            return candidate
    return None


def _hour_chunks(frame: pd.DataFrame, hours: int) -> Iterable[pd.DataFrame]:
    timestamps = pd.Index(pd.to_datetime(frame["__decision_ts__"], utc=True).unique()).sort_values()
    for offset in range(0, len(timestamps), hours):
        selected = timestamps[offset:offset + hours]
        yield frame.loc[frame["__decision_ts__"].isin(selected)].copy()


def _causal_conversion_state(
    bundle: Any,
    warmup: pd.DataFrame,
    score_input: pd.DataFrame,
    *,
    chunk_hours: int,
) -> pd.DataFrame:
    """Materialise dynamic Geometry/K9 state with target-free warm-up only.

    ``FrozenGeometryK9.transform`` consults its historic K9 mass to form
    rolling support/OOD fields.  The frozen Oct--Dec history ends before the
    2025 control.  Extending that history with completed target-free chunks is
    required to give the prior-28 reserve its genuine prior state, while
    retaining the unchanged frozen encoder/K9 definition.
    """

    parent = bundle.geometry.parent
    original_history = parent.state_history
    parts: list[pd.DataFrame] = []
    membership_columns = [f"k09__cluster_{index:02d}__membership" for index in range(9)]
    try:
        # Never materialise warm-up and score windows together.  Warm-up is
        # needed solely to advance target-free dynamic K9 history and is then
        # discarded; retaining it roughly doubles peak RAM on 170 symbols.
        for source, retain in ((warmup, False), (score_input, True)):
            ordered = source.sort_values(
                ["__decision_ts__", "candidate_id"], kind="stable",
            ).reset_index(drop=True)
            for piece in _hour_chunks(ordered, chunk_hours):
                geometry_state = _canonical_ldf_geometry_aliases(
                    pd.concat(
                        [bundle.geometry.transform(piece), bundle.leaf_trust.transform(piece)],
                        axis=1,
                    ),
                )
                aggregate = _aggregate_state_fields(geometry_state)
                if retain:
                    parts.append(pd.concat(
                        [
                            piece.loc[:, list(IDENTITY_FIELDS)].reset_index(drop=True),
                            geometry_state.loc[:, list(aggregate)].reset_index(drop=True),
                        ],
                        axis=1,
                    ))
                mass = geometry_state.loc[:, membership_columns].copy()
                mass.columns = [f"k{index}" for index in range(9)]
                mass["__decision_ts__"] = pd.to_datetime(
                    piece["__decision_ts__"], utc=True,
                ).to_numpy()
                timestamp_mass = mass.groupby("__decision_ts__", sort=True).sum().reset_index()
                parent.state_history = pd.concat(
                    [parent.state_history, timestamp_mass], ignore_index=True,
                ).sort_values("__decision_ts__", kind="stable").drop_duplicates(
                    "__decision_ts__", keep="last",
                ).reset_index(drop=True)
        state = pd.concat(parts, ignore_index=True)
    finally:
        parent.state_history = original_history
    if state["candidate_id"].duplicated().any():
        raise AssertionError("causal conversion state has duplicate candidate identities")
    return state


def _numeric_audit(left: pd.Series, right: pd.Series) -> dict[str, Any]:
    a = pd.to_numeric(left, errors="coerce").to_numpy(float)
    b = pd.to_numeric(right, errors="coerce").to_numpy(float)
    same_finite = np.isfinite(a) == np.isfinite(b)
    delta = np.abs(a - b)
    denominator = np.maximum(np.maximum(np.abs(a), np.abs(b)), NUMERIC_ATOL)
    relative = delta / denominator
    finite = np.isfinite(delta)
    passed = bool(
        same_finite.all()
        and np.allclose(a, b, rtol=NUMERIC_RTOL, atol=NUMERIC_ATOL, equal_nan=True)
    )
    return {
        "passed": passed,
        "finite_pair_rows": int(finite.sum()),
        "max_abs_delta": float(delta[finite].max()) if finite.any() else 0.0,
        "max_relative_delta": float(relative[finite].max()) if finite.any() else 0.0,
    }


def _boolean_audit(left: pd.Series, right: pd.Series) -> dict[str, Any]:
    a = left.fillna(False).astype(bool).to_numpy()
    b = right.fillna(False).astype(bool).to_numpy()
    return {"passed": bool(np.array_equal(a, b)), "different_rows": int((a != b).sum())}


def _head_audit(rescored: pd.DataFrame) -> dict[str, Any]:
    heads = sorted(
        field for field in rescored.columns
        if field.startswith("conditional_head__") and field.endswith("__rank")
    )
    if len(heads) != 10:
        raise AssertionError(f"expected ten conditional heads, found {len(heads)}")
    route = rescored["base_route_timestamp_top30"].fillna(False).to_numpy(bool)
    reconstructed = np.full(len(rescored), np.nan, dtype=float)
    if route.any():
        # Preserve the score reducer's float32 median semantics.
        reconstructed[route] = np.nanmedian(
            rescored.loc[route, heads].to_numpy(dtype=np.float32, copy=False), axis=1,
        )
    aggregate = rescored["conditional_consensus_rank"].to_numpy(float)
    return {
        "head_count": len(heads),
        "head_names": heads,
        "median_matches_reconstructed_aggregate": bool(np.allclose(
            reconstructed, aggregate, rtol=0.0, atol=0.0, equal_nan=True,
        )),
    }


def _rescore_block(
    source: Path,
    control: Path,
    b0_root: Path,
    bundle_path: Path,
    *,
    chunk_hours: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    block = _block_name(bundle_path)
    b0_control_block = bundle_path.parents[1].name
    upstream = joblib.load(bundle_path)
    conversion_path = bundle_path.parents[1] / "conversion" / "four_week_conversion_bundle.joblib"
    conversion = joblib.load(conversion_path)
    cutoff = _utc(upstream.cutoff)
    bundle_end = _utc(upstream.end_exclusive)
    # ``*_finalcoverage`` is an intentionally shortened immutable historical
    # score artifact.  Its bundle can nominally extend beyond the archived
    # score coverage, so the parity contract is its persisted decision span,
    # not an unreported tail of the fitted bundle.
    stored = pd.read_parquet(_score_path(control, block))
    stored["__decision_ts__"] = pd.to_datetime(
        stored["__decision_ts__"], utc=True, errors="raise",
    )
    stored_end = stored["__decision_ts__"].max() + pd.Timedelta(hours=1)
    end = min(bundle_end, stored_end)
    stored_start = stored["__decision_ts__"].min()
    if stored_start < cutoff:
        raise AssertionError(f"{block} stored score coverage predates the bundle cutoff")
    reserve_start = cutoff - pd.Timedelta(days=RESERVE_DAYS)
    fields = tuple(upstream.base_fields)
    columns = [*IDENTITY_FIELDS, *fields]
    score_input = pd.read_parquet(
        source,
        columns=columns,
        filters=[("__decision_ts__", ">=", reserve_start), ("__decision_ts__", "<", end)],
    )
    score_input["__decision_ts__"] = pd.to_datetime(
        score_input["__decision_ts__"], utc=True, errors="raise",
    )
    _assert_exact_hour(score_input, context=f"{block} target-free score input")
    if score_input["candidate_id"].duplicated().any():
        raise AssertionError(f"{block} target-free source duplicated candidate identities")
    reference = score_input.loc[score_input["__decision_ts__"].lt(cutoff)].copy()
    held = score_input.loc[score_input["__decision_ts__"].ge(cutoff)].copy()
    if reference.empty or held.empty:
        raise AssertionError(f"{block} has empty same-model reserve or held population")
    # The frozen final-score coordinate uses a complete prior-28 reference.
    # Only held decisions are stopped below their timestamp-local top-30%
    # route.  This mirrors the persisted monthly-prequential control scorer.
    upstream_reference = score_monthly_upstream_bundle(
        upstream,
        reference,
        allow_prior_reference=True,
        prior_reference_start=reserve_start,
        route_top_fraction=None,
    )
    upstream_held = score_monthly_upstream_bundle(
        upstream,
        held,
        allow_prior_reference=True,
        prior_reference_start=reserve_start,
        route_top_fraction=ROUTE_FRACTION,
    )
    # Legacy frozen bundles did not persist the file digest in their in-memory
    # manifest.  The historical score receipt used the immutable payload
    # digests, so restore those exact values before the conversion handoff.
    upstream_hash = _sha256(bundle_path)
    upstream_reference["upstream_bundle_sha256"] = upstream_hash
    upstream_held["upstream_bundle_sha256"] = upstream_hash
    scored_reference = reference.merge(
        upstream_reference,
        on=list(IDENTITY_FIELDS), how="left", validate="one_to_one",
    )
    scored_held = held.merge(
        upstream_held,
        on=list(IDENTITY_FIELDS), how="left", validate="one_to_one",
    )
    # Use the original frozen monthly-prequential scorer directly.  Its audit
    # identifies this topology explicitly; the later lockstep implementation
    # is a successor contract and must not be substituted during control
    # reproduction.
    converted, conversion_audit = score_four_week_conversion_bundle(
        conversion,
        reference=scored_reference,
        held=scored_held,
    )
    converted["conversion_bundle_sha256"] = _sha256(conversion_path)
    held_rescored = converted.loc[converted["__score_role__"].eq("held")].copy()
    # The conversion scorer deliberately persists only the aggregate upstream
    # quantities it consumes.  The base probabilities, timestamp route, and
    # individual head ranks are nevertheless part of the Stage-0 contract.
    # Carry them from the *same* target-free upstream pass; never reconstruct
    # them from a held outcome ledger or infer them from the aggregate C1.
    upstream_auxiliary = [
        "candidate_id",
        "p_adverse", "p_weak", "p_clear",
        "base_route_timestamp_top30",
        *[
            field for field in scored_held.columns
            if field.startswith("conditional_head__")
            and field.endswith(("__raw", "__rank"))
        ],
    ]
    held_rescored = held_rescored.merge(
        scored_held.loc[:, upstream_auxiliary],
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    if held_rescored.loc[:, upstream_auxiliary[1:]].isna().all(axis=1).any():
        raise AssertionError(f"{block} conversion output lost upstream auxiliary coverage")
    # Final-coverage artifacts can retain only a late slice of a live bundle.
    # Earlier held hours were still scored target-free above so their causal
    # Geometry/K9 state reaches the retained slice, but they are not a stored
    # candidate population and therefore not part of the equality comparison.
    held_rescored = held_rescored.loc[
        held_rescored["candidate_id"].isin(stored["candidate_id"])
    ].copy()
    _assert_exact_hour(stored, context=f"{block} stored control")
    b0 = pd.read_parquet(
        b0_root / "b0_target_free_reconstruction.parquet",
        columns=["candidate_id", "control_block", "p_adverse", "p_weak", "p_clear", "base_route_timestamp_top30"],
        # The shortened final July block carries ``_finalcoverage`` in its
        # immutable B0 identity.  Use the real directory token rather than a
        # normalised cutoff string so terminal coverage stays in parity.
        filters=[("control_block", "=", b0_control_block)],
    )
    b0 = b0.drop(columns="control_block")
    if b0["candidate_id"].duplicated().any():
        raise AssertionError(f"{block} frozen B0 route artifact duplicates candidate IDs")
    expected = stored.merge(b0, on="candidate_id", how="outer", indicator="__b0_merge__", validate="one_to_one")
    if not expected["__b0_merge__"].eq("both").all():
        raise AssertionError(
            f"{block} stored/B0 identity mismatch: {expected['__b0_merge__'].value_counts().to_dict()}"
        )
    expected = expected.drop(columns="__b0_merge__")
    merged = expected.merge(
        held_rescored,
        on="candidate_id",
        how="outer",
        indicator=True,
        suffixes=("__stored", "__rescored"),
        validate="one_to_one",
    )
    if not merged["_merge"].eq("both").all():
        raise AssertionError(
            f"{block} candidate identity mismatch: "
            f"{merged['_merge'].value_counts().to_dict()}"
        )
    identity = {
        field: bool((merged[f"{field}__stored"] == merged[f"{field}__rescored"]).all())
        for field in ("__decision_ts__", "side_name")
    }
    if not all(identity.values()):
        raise AssertionError(f"{block} identity field mismatch: {identity}")
    numeric = {
        field: _numeric_audit(merged[f"{field}__stored"], merged[f"{field}__rescored"])
        for field in CORE_FIELDS
    }
    boolean = {
        field: _boolean_audit(merged[f"{field}__stored"], merged[f"{field}__rescored"])
        for field in BOOLEAN_FIELDS
    }
    head = _head_audit(held_rescored)
    bundle_identity = {
        "upstream_bundle": bool((
            merged["upstream_bundle_sha256__stored"]
            == merged["upstream_bundle_sha256__rescored"]
        ).all()),
        "conversion_bundle": bool((
            merged["conversion_bundle_sha256__stored"]
            == merged["conversion_bundle_sha256__rescored"]
        ).all()),
        "geometry_bundle": bool((
            merged["geometry_bundle_sha256__stored"]
            == merged["geometry_bundle_sha256__rescored"]
        ).all()),
    }
    passed = (
        all(n["passed"] for n in numeric.values())
        and all(b["passed"] for b in boolean.values())
        and all(bundle_identity.values())
        and head["median_matches_reconstructed_aggregate"]
    )
    audit = {
        "block": block,
        "cutoff": cutoff.isoformat(),
        "reserve_start": reserve_start.isoformat(),
        "bundle_end_exclusive": bundle_end.isoformat(),
        "stored_coverage_start": stored_start.isoformat(),
        "stored_coverage_end_exclusive": end.isoformat(),
        "final_cdf_reference_contract": "complete same-upstream prior-28 reference; held-only timestamp-local top-30 route",
        "geometry_state_contract": "exact persisted monthly-prequential scorer; frozen geometry/K9 is never refit",
        "source_rows_reserve_plus_held": int(len(score_input)),
        "reserve_rows": int(len(reference)),
        "held_rows": int(len(held_rescored)),
        "stored_rows": int(len(expected)),
        "identity": {"candidate_ids": True, **identity},
        "numeric": numeric,
        "boolean": boolean,
        "head": head,
        "bundle_identity": bundle_identity,
        "conversion_audit": conversion_audit.to_dict(orient="records"),
        "passed": bool(passed),
    }
    compact = held_rescored.loc[:, [
        *IDENTITY_FIELDS,
        *CORE_FIELDS,
        *BOOLEAN_FIELDS,
        "conversion_bundle_sha256", "geometry_bundle_sha256", "upstream_bundle_sha256",
        *[field for field in held_rescored.columns if field.startswith("conditional_head__") and field.endswith("__rank")],
    ]].copy()
    compact["control_block"] = block
    return compact, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--control-root", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--b0-root", type=Path, default=DEFAULT_B0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--blocks", nargs="*", help="optional YYYYMMDDTHHMMSSZ control cutoffs")
    parser.add_argument("--chunk-hours", type=int, default=24)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    if args.chunk_hours < 1:
        raise ValueError("chunk-hours must be positive")
    if not args.source.is_file() or not args.control_root.is_dir() or not args.b0_root.is_dir():
        raise FileNotFoundError("target-free source, B0 route artifact, or frozen control root is unavailable")
    bundle_paths = sorted(args.control_root.glob("bundles/block=*/upstream/monthly_upstream_bundle.joblib"))
    if args.blocks:
        requested = set(args.blocks)
        bundle_paths = [path for path in bundle_paths if _block_name(path) in requested]
        missing = requested - {_block_name(path) for path in bundle_paths}
        if missing:
            raise ValueError(f"unknown frozen control block(s): {sorted(missing)}")
    if not bundle_paths:
        raise FileNotFoundError("no frozen upstream blocks selected")
    args.out_dir.mkdir(parents=True)
    rows: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    failure: Exception | None = None
    try:
        for bundle_path in bundle_paths:
            rescored, audit = _rescore_block(
                args.source, args.control_root, args.b0_root, bundle_path, chunk_hours=args.chunk_hours,
            )
            rows.append(rescored)
            audits.append(audit)
            print(json.dumps({
                "event": "control_block_scored", "block": audit["block"],
                "held_rows": audit["held_rows"], "passed": audit["passed"],
            }), flush=True)
            if not audit["passed"]:
                raise AssertionError(f"frozen control parity failed in {audit['block']}")
    except Exception as error:  # preserve the receipt before propagating the failure
        failure = error
    pd.DataFrame(audits).to_json(
        args.out_dir / "control_parity_blocks.json", orient="records", indent=2, date_format="iso",
    )
    if rows:
        pd.concat(rows, ignore_index=True).to_parquet(
            args.out_dir / "control_rescored_target_free.parquet", index=False, compression="zstd",
        )
    status = "passed" if failure is None and len(audits) == len(bundle_paths) and all(a["passed"] for a in audits) else "failed"
    manifest = {
        "schema": "strict_r3_full_control_parity_v1",
        "status": status,
        "scope": "offline long-only :00 frozen-control audit; no live/canonical/order-capable artifact modified",
        "source": {"path": str(args.source), "sha256": _sha256(args.source)},
        "control_root": str(args.control_root),
        "b0_route_artifact": {
            "path": str(args.b0_root / "b0_target_free_reconstruction.parquet"),
            "sha256": _sha256(args.b0_root / "b0_target_free_reconstruction.parquet"),
        },
        "block_count_requested": len(bundle_paths),
        "block_count_completed": len(audits),
        "base_route_fraction": ROUTE_FRACTION,
        "same_model_reserve_days": RESERVE_DAYS,
        "geometry_state": "frozen encoder/K9; exact persisted monthly-prequential scorer; never refit",
        "numeric_tolerance": {"relative": NUMERIC_RTOL, "absolute": NUMERIC_ATOL},
        "failure": None if failure is None else repr(failure),
    }
    (args.out_dir / "control_parity.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if failure is not None:
        raise failure
    print(json.dumps({"event": "full_control_parity_complete", **manifest}, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
