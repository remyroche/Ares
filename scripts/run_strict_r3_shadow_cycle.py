#!/usr/bin/env python3
"""Run one immutable strict-R3 shadow scoring/admission cycle.

This is deliberately not an exchange process.  It composes the canonical
target-free scorer and Cell-day admission CLI, verifies their manifests, and
emits hypothetical long candidates only.  The module has no exchange client,
credentials, order, cancel, or position mutation path.  A separate, future
promotion must consume these decisions after hourly replay parity passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_shadow_portfolio import (  # noqa: E402
    ShadowPortfolioPolicy,
    ShadowPortfolioState,
    add_shadow_entries,
    apply_adaptive_exit_v1,
    advance_shadow_state,
    auction_admitted_snapshot,
    causal_flat_fill_omitted_15m,
)
from extreme_price_movements.adaptive_exit_v1 import (  # noqa: E402
    AdaptiveExitV1Bundle,
    ENTRY_CONTEXT as ADAPTIVE_EXIT_ENTRY_CONTEXT,
    SCORE_SOURCE_TO_FEATURE as ADAPTIVE_EXIT_SCORE_FIELDS,
)
from extreme_price_movements.strict_r3_inference_bundle import (  # noqa: E402
    SCHEMA_V6,
    StrictR3InferenceBundle,
)
from extreme_price_movements.strict_r3_mc1_mapper import MC1D2Bundle  # noqa: E402
from extreme_price_movements.inference.canonical_stack_reporting import (  # noqa: E402
    CANONICAL_STACK_REPORTING_KEYS,
)

CANONICAL_ADMISSION_MODE = "strict_r3_robust21_plus_mc1_d2_authority_v1"


def _load_state_bars(
    bar_root: Path,
    symbols: set[str],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    for symbol in sorted(symbols):
        stem = symbol.lower().replace("/", "").replace("_", "")
        path = bar_root / f"{stem}_15m.parquet"
        if not path.exists():
            continue
        bars = pd.read_parquet(path, columns=["open", "high", "low", "close"])
        bars.index = pd.to_datetime(bars.index, utc=True, errors="raise")
        output[symbol] = causal_flat_fill_omitted_15m(
            bars, start=start, end=end,
        )
    return output


def _load_entry_bars(
    bar_root: Path,
    symbols: set[str],
    *,
    end: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    """Load full causal history only for the at-most-two accepted entries."""
    output: dict[str, pd.DataFrame] = {}
    for symbol in sorted(symbols):
        stem = symbol.lower().replace("/", "").replace("_", "")
        path = bar_root / f"{stem}_15m.parquet"
        if not path.exists():
            continue
        bars = pd.read_parquet(path, columns=["open", "high", "low", "close"])
        bars.index = pd.to_datetime(bars.index, utc=True, errors="raise")
        # Fill only completed intervals before the exact decision boundary.
        # The decision cell itself may be absent because Kraken has not yet
        # published a 15-minute trade candle.  In that case the candidate's
        # hash-bound official 1-hour open is the declared fallback and must not
        # be compared with a synthetic flat candle.  If an observed 15-minute
        # decision cell exists, retain it so add_shadow_entries enforces exact
        # equality as before.
        prior_end = end - pd.Timedelta(minutes=15)
        filled = causal_flat_fill_omitted_15m(bars, end=prior_end)
        observed = bars.loc[bars.index == end, ["open", "high", "low", "close"]]
        output[symbol] = pd.concat([filled, observed]).loc[
            lambda frame: ~frame.index.duplicated(keep="last")
        ].sort_index()
    return output


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str], *, log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            text=True, check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"shadow stage failed rc={completed.returncode}; see {log_path}",
        )


def _run_receipted(
    command: list[str], *, log_path: Path, receipt_path: Path,
) -> None:
    """Run a deterministic no-order stage and require its immutable receipt.

    A child that exits zero but emits neither an output directory nor its
    receipt is not a valid admission result.  Retry exactly once only when no
    output exists, which is safe because the target remains new/immutable.
    Any partial output fails closed rather than being overwritten.
    """
    _run(command, log_path=log_path)
    if receipt_path.is_file():
        return
    if receipt_path.parent.exists():
        raise RuntimeError(
            f"shadow stage exited without receipt and left partial output: {receipt_path}"
        )
    retry_log = log_path.with_name(log_path.stem + "_retry.log")
    _run(command, log_path=retry_log)
    if not receipt_path.is_file():
        raise RuntimeError(
            f"shadow stage exited without required receipt after retry: {receipt_path}"
        )


def _start(command: list[str], *, log_path: Path) -> tuple[subprocess.Popen[str], object]:
    """Start an isolated no-exchange scorer for overlap with another scorer."""
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True,
    )
    return process, handle


def _wait(process: subprocess.Popen[str], handle: object, *, log_path: Path) -> None:
    returncode = process.wait()
    handle.close()
    if returncode:
        raise RuntimeError(f"shadow stage failed rc={returncode}; see {log_path}")


def _restore_immutable_prediction_prefix(
    *,
    prefix_path: Path,
    current_path: Path,
    manifest_path: Path,
    allow_missing_current_prefix_rows: bool = False,
    validate_overlap_base_outputs: bool = True,
    strip_policy_label_columns: bool = False,
) -> dict[str, object]:
    """Retain sealed downstream outputs and append only newly scored rows.

    The live top-20 compute route deliberately stops below-route rows after the
    base. Re-scoring a historical prefix would consequently replace previously
    materialized downstream values with nulls. Historical rows are immutable:
    validate every base output that was recomputed, then restore the complete
    old row and append only new candidate identities.
    """
    prefix = pd.read_parquet(prefix_path)
    current = pd.read_parquet(current_path)
    stripped_columns: list[str] = []
    if strip_policy_label_columns:
        stripped_columns = [
            field for field in prefix.columns
            if field.startswith("policy_")
        ]
        prefix = prefix.drop(columns=stripped_columns)
    for frame, name in ((prefix, "prediction prefix"), (current, "current predictions")):
        frame["candidate_id"] = frame["candidate_id"].astype(str)
        if frame["candidate_id"].isna().any() or frame["candidate_id"].duplicated().any():
            raise ValueError(f"{name} contains invalid candidate identities")
    current_indexed = current.set_index("candidate_id", drop=False)
    prefix_indexed = prefix.set_index("candidate_id", drop=False)
    missing = prefix_indexed.index.difference(current_indexed.index)
    if len(missing) and not allow_missing_current_prefix_rows:
        raise ValueError(f"current score omitted {len(missing)} immutable prefix rows")
    overlap_ids = prefix_indexed.index.intersection(current_indexed.index)
    overlap = current_indexed.loc[overlap_ids]
    prefix_overlap = prefix_indexed.loc[overlap_ids]
    if validate_overlap_base_outputs:
        for field in (
            "base_score", "base_rank42", "base_anchor_bps",
            "upstream_bundle_sha256", "conversion_bundle_sha256",
            "geometry_bundle_sha256",
        ):
            if field not in prefix_overlap or field not in overlap:
                raise ValueError(f"immutable prediction audit lacks {field}")
            left = prefix_overlap[field]
            right = overlap[field]
            if pd.api.types.is_numeric_dtype(left) or pd.api.types.is_numeric_dtype(right):
                if not np.allclose(
                    pd.to_numeric(left, errors="coerce").to_numpy(float),
                    pd.to_numeric(right, errors="coerce").to_numpy(float),
                    atol=1e-9, rtol=0.0, equal_nan=True,
                ):
                    raise ValueError(f"immutable prediction prefix conflicts on {field}")
            elif not left.astype(str).eq(right.astype(str)).all():
                raise ValueError(f"immutable prediction prefix conflicts on {field}")
    new = current.loc[~current["candidate_id"].isin(set(prefix["candidate_id"]))].copy()
    columns = list(prefix.columns) + [
        field for field in current.columns if field not in prefix.columns
    ]
    if "base_route_timestamp_top20" in columns and "base_route_timestamp_top20" not in prefix:
        prefix["base_route_timestamp_top20"] = pd.to_numeric(
            prefix.get("final_score"), errors="coerce"
        ).notna()
    if "base_route_fraction" in columns and "base_route_fraction" not in prefix:
        prefix["base_route_fraction"] = 0.20
    if "base_route_status" in columns and "base_route_status" not in prefix:
        prefix["base_route_status"] = np.where(
            prefix["base_route_timestamp_top20"],
            "immutable_historical_routed",
            "immutable_historical_below_route",
        )
    output = pd.concat([
        prefix.reindex(columns=columns), new.reindex(columns=columns),
    ], ignore_index=True)
    output = output.sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable",
    ).reset_index(drop=True)
    output.to_parquet(current_path, index=False, compression="zstd")
    manifest = json.loads(manifest_path.read_text())
    audit = {
        "prefix_rows": int(len(prefix)),
        "prefix_overlap_rows": int(len(overlap_ids)),
        "prefix_rows_absent_from_current_replay": int(len(missing)),
        "missing_current_prefix_rows_allowed": bool(
            allow_missing_current_prefix_rows
        ),
        "new_rows": int(len(new)),
        "output_rows": int(len(output)),
        "base_fields_exact": bool(validate_overlap_base_outputs),
        "overlap_validation": (
            "exact_base_and_bundle_fields"
            if validate_overlap_base_outputs
            else "skipped_only_for_hash_sealed_bootstrap_prefix"
        ),
        "stripped_non_prediction_columns": stripped_columns,
        "historical_downstream_rows_restored": int(len(prefix)),
        "prefix_sha256": _sha(prefix_path),
        "output_sha256": _sha(current_path),
    }
    manifest["held_rows"] = int(len(output))
    manifest["immutable_prediction_prefix"] = audit
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return audit


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-bundle", type=Path, required=True)
    parser.add_argument("--held-candidates", type=Path, required=True)
    parser.add_argument(
        "--policy-label-candidates", type=Path,
        help=(
            "Immutable historical candidate prefix used only for the UTC-day "
            "resolved-label refresh. Defaults to --held-candidates."
        ),
    )
    parser.add_argument("--held-features", type=Path, required=True)
    parser.add_argument(
        "--portfolio-policy-override", type=Path,
        help="Sealed challenger portfolio policy; defaults to the inference bundle policy.",
    )
    parser.add_argument("--bcf-monthly-bundle-dir", type=Path)
    parser.add_argument("--bcf-reference-ledger", type=Path)
    parser.add_argument("--bcf-mc1-ledger", type=Path)
    parser.add_argument("--bcf-mc1-bundle-dir", type=Path)
    parser.add_argument("--portfolio-state-json", type=Path, required=True)
    parser.add_argument("--decision-ts", required=True)
    parser.add_argument(
        "--immutable-prediction-prefix", type=Path,
        help="Previous successful hourly predictions retained byte-for-byte by identity.",
    )
    parser.add_argument(
        "--allow-missing-current-prefix-rows", action="store_true",
        help=(
            "First-successor bootstrap only: retain sealed prefix identities "
            "that are absent from the reconstructed current population. Every "
            "overlapping base output must still match exactly."
        ),
    )
    parser.add_argument(
        "--sealed-bootstrap-prediction-prefix", action="store_true",
        help=(
            "First-successor bootstrap only. Requires the immutable prefix to "
            "be the exact resolved-score ledger sealed by the inference bundle. "
            "Historical replay values are then retained without comparing them "
            "to a changed successor runtime."
        ),
    )
    parser.add_argument(
        "--intraday-frozen-resolved-ledger", type=Path,
        help=(
            "Prior successful hourly runtime ledger whose exact bytes define "
            "calibration for the remainder of the UTC day."
        ),
    )
    parser.add_argument(
        "--reset-calibration-to-sealed-base", action="store_true",
        help=(
            "Start a new same-day calibration carry from the sealed parent-policy "
            "ledger. This is only valid for an explicitly resealed policy-identity "
            "transition and never appends current outcomes."
        ),
    )
    parser.add_argument(
        "--lockstep-geometry-k9-state-in", type=Path,
        help=(
            "Immutable Geometry/K9 state emitted by the immediately preceding "
            "successful hourly score. Its exact producer hashes are validated "
            "by the forward scorer before the current hour is scored."
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("shadow-only",), default="shadow-only",
        help="No order-capable mode exists in this runner.",
    )
    return parser.parse_args()


def main() -> None:
    args = _args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable shadow cycle exists: {args.out_dir}")
    decision_ts = pd.Timestamp(args.decision_ts)
    decision_ts = (
        decision_ts.tz_localize("UTC")
        if decision_ts.tzinfo is None else decision_ts.tz_convert("UTC")
    )
    bundle = StrictR3InferenceBundle.load(args.inference_bundle, root=ROOT)
    bundle_audit = bundle.validate(decision_ts=decision_ts)
    if str(bundle.payload["schema"]) != SCHEMA_V6:
        raise ValueError("canonical shadow inference requires Robust-21 + MC1 schema-v6")
    conversion_bundle_dir = bundle.path("conversion_bundle_dir")
    upstream_bundle_dir = bundle.path("upstream_bundle_dir")
    reference_candidates = bundle.path("same_model_reference_candidates")
    reference_features = bundle.path("same_model_reference_features")
    resolved_score_label_ledger = bundle.path("resolved_score_label_ledger")
    # Two policy roles are intentionally distinct.  MC1/calibration is tied
    # to the immutable source-aligned parent-policy label ledger it was fit
    # against; the rich execution policy governs only live/replay exits.  A
    # recovery must never use the latter to assemble a former's label state.
    execution_policy_json = bundle.path("exit_policy")
    calibration_policy_json = (
        bundle.path("calibration_policy")
        if "calibration_policy" in (bundle.payload.get("paths") or {})
        else execution_policy_json
    )
    portfolio_policy_json = (
        args.portfolio_policy_override
        if args.portfolio_policy_override is not None
        else bundle.path("portfolio_policy")
    )
    adaptive_exit_bundle_dir = bundle.path("adaptive_exit_v1_bundle_dir")
    mc1_bundle_dir = bundle.path("mc1_d2_bundle_dir")
    r5_bundle_dir = bundle.path("cell_day_trust_bundle_dir")
    a5_bundle_dir = bundle.path("a5_bundle_dir")
    dual_mode = any(value is not None for value in (
        args.bcf_monthly_bundle_dir, args.bcf_reference_ledger,
        args.bcf_mc1_ledger, args.bcf_mc1_bundle_dir,
    ))
    if dual_mode and not all(value is not None for value in (
        args.bcf_monthly_bundle_dir, args.bcf_reference_ledger,
        args.bcf_mc1_ledger, args.bcf_mc1_bundle_dir,
    )):
        raise ValueError("dual BCF/current mode requires all four BCF artifact arguments")
    args.out_dir.mkdir(parents=True)
    score_dir = args.out_dir / "score"
    policy_label_dir = args.out_dir / "resolved_policy_labels"
    runtime_resolved_dir = args.out_dir / "runtime_resolved_state"
    admission_dir = args.out_dir / "admission"
    bcf_score_dir: Path | None = None
    bcf_command: list[str] | None = None
    if dual_mode:
        bcf_score_dir = args.out_dir / "bcf_score"
        bcf_command = [
            sys.executable, str(ROOT / "scripts" / "score_strict_r3_bcf_forward.py"),
            "--monthly-bundle-dir", str(args.bcf_monthly_bundle_dir),
            "--reference-ledger", str(args.bcf_reference_ledger),
            "--held-features", str(args.held_features),
            "--decision-ts", decision_ts.isoformat(),
            "--reference-cache-dir", str(
                ROOT / "data_perp/artifacts/strict_r3_bcf_same_model_reference_cache_v2"
            ),
            "--out-dir", str(bcf_score_dir),
        ]
    _run([
        sys.executable, str(ROOT / "scripts" / "score_strict_r3_forward.py"),
        "--schema", "current-v5",
        "--bundle-dir", str(conversion_bundle_dir),
        "--upstream-bundle-dir", str(upstream_bundle_dir),
        "--reference-candidates", str(reference_candidates),
        "--reference-features", str(reference_features),
        "--held-candidates", str(args.held_candidates),
        "--held-features", str(args.held_features),
        "--out-dir", str(score_dir),
        # Frozen with the canonical batch producer.  A different score-piece
        # width changes rolling K9 support/OOD inputs even when model weights
        # and cluster semantics are unchanged.
        "--lockstep-score-chunk-hours", "72",
        *(
            [
                "--lockstep-geometry-k9-state-in",
                str(args.lockstep_geometry_k9_state_in),
            ]
            if args.lockstep_geometry_k9_state_in is not None else []
        ),
    ], log_path=args.out_dir / "score.log")
    # The two independent scorers were previously overlapped to reduce wall
    # time.  On the live host, their simultaneous model/geometry allocations
    # exceed available memory and can terminate the current scorer without a
    # useful receipt.  Run them sequentially: the score functions, frozen
    # references and dual-admission semantics are unchanged.
    if dual_mode:
        if bcf_command is None or bcf_score_dir is None:
            raise AssertionError("dual BCF scorer was not configured")
        # BCF is an independent scorer with an immutable output contract.  A
        # zero exit code alone is insufficient: a prior interrupted process
        # could otherwise reach dual admission without its BCF predictions.
        # Apply the same receipt discipline as the MC1 admission stages.
        _run_receipted(
            bcf_command,
            log_path=args.out_dir / "bcf_score.log",
            receipt_path=bcf_score_dir / "run_manifest.json",
        )
        if not (bcf_score_dir / "predictions.parquet").is_file():
            raise RuntimeError("BCF scorer receipt lacks predictions.parquet")
    immutable_prediction_prefix_audit = None
    if args.immutable_prediction_prefix is not None:
        if args.sealed_bootstrap_prediction_prefix:
            sealed_prefix = bundle.path("resolved_score_label_ledger").resolve()
            if args.immutable_prediction_prefix.resolve() != sealed_prefix:
                raise ValueError(
                    "bootstrap prediction prefix is not the hash-sealed resolved ledger"
                )
        immutable_prediction_prefix_audit = _restore_immutable_prediction_prefix(
            prefix_path=args.immutable_prediction_prefix,
            current_path=score_dir / "predictions.parquet",
            manifest_path=score_dir / "run_manifest.json",
            allow_missing_current_prefix_rows=bool(
                args.allow_missing_current_prefix_rows
            ),
            validate_overlap_base_outputs=not bool(
                args.sealed_bootstrap_prediction_prefix
            ),
            strip_policy_label_columns=bool(
                args.sealed_bootstrap_prediction_prefix
            ),
        )
    # Calibration is frozen within each UTC day.  Carry its exact bytes during
    # the day; only the first decision of a new UTC day materializes newly
    # resolved labels from the immutable historical candidate prefix.
    if args.reset_calibration_to_sealed_base and args.intraday_frozen_resolved_ledger is not None:
        raise ValueError("calibration base reset cannot carry an intraday ledger")
    if args.intraday_frozen_resolved_ledger is None and not args.reset_calibration_to_sealed_base:
        _run([
            sys.executable,
            str(ROOT / "scripts" / "materialize_strict_r3_frozen_policy_labels_v2.py"),
            "--candidates", str(
                args.policy_label_candidates or args.held_candidates
            ),
            "--bar-root", str(ROOT / str(bundle.payload["runtime"]["policy_bar_root"])),
            "--policy-json", str(calibration_policy_json),
            "--label-available-before", decision_ts.normalize().isoformat(),
            "--out-dir", str(policy_label_dir),
        ], log_path=args.out_dir / "resolved_policy_labels.log")
    assembly_command = [
        sys.executable,
        str(ROOT / "scripts" / "assemble_strict_r3_runtime_resolved_ledger.py"),
        "--base-resolved-ledger", str(resolved_score_label_ledger),
        "--current-predictions", str(score_dir / "predictions.parquet"),
        "--policy-json", str(calibration_policy_json),
        "--decision-ts", decision_ts.isoformat(),
        "--out-dir", str(runtime_resolved_dir),
    ]
    if args.reset_calibration_to_sealed_base:
        assembly_command.append("--reset-to-sealed-base")
    elif args.intraday_frozen_resolved_ledger is None:
        assembly_command.extend([
            "--current-policy-labels",
            str(policy_label_dir / "frozen_policy_labels.parquet"),
        ])
    if args.intraday_frozen_resolved_ledger is not None:
        assembly_command.extend([
            "--intraday-frozen-ledger",
            str(args.intraday_frozen_resolved_ledger),
        ])
    _run(assembly_command, log_path=args.out_dir / "runtime_resolved_state.log")
    _run_receipted([
        sys.executable, str(ROOT / "scripts" / "admit_strict_r3_mc1_forward.py"),
        "--resolved-score-label-ledger", str(
            runtime_resolved_dir / "walkforward_scored_label_ledger.parquet"
        ),
        "--current-predictions", str(score_dir / "predictions.parquet"),
        "--mc1-bundle-dir", str(mc1_bundle_dir),
        "--r5-bundle-dir", str(r5_bundle_dir),
        "--a5-bundle-dir", str(a5_bundle_dir),
        "--decision-ts", decision_ts.isoformat(),
        "--out-dir", str(admission_dir),
    ], log_path=args.out_dir / "admission.log",
       receipt_path=admission_dir / "run_manifest.json")

    score_manifest = json.loads((score_dir / "run_manifest.json").read_text())
    runtime_resolved_manifest = json.loads(
        (runtime_resolved_dir / "run_manifest.json").read_text()
    )
    active_admission_dir = admission_dir
    if dual_mode:
        if bcf_score_dir is None:
            raise AssertionError("BCF scorer was not configured")
        dual_admission_dir = args.out_dir / "dual_admission"
        _run_receipted([
            sys.executable, str(ROOT / "scripts" / "admit_strict_r3_bcf_current_dual_forward.py"),
            "--current-admission", str(admission_dir / "admitted_predictions.parquet"),
            "--bcf-predictions", str(bcf_score_dir / "predictions.parquet"),
            "--bcf-resolved-ledger", str(args.bcf_mc1_ledger),
            "--bcf-mc1-bundle-dir", str(args.bcf_mc1_bundle_dir),
            "--decision-ts", decision_ts.isoformat(),
            "--out-dir", str(dual_admission_dir),
        ], log_path=args.out_dir / "dual_admission.log",
           receipt_path=dual_admission_dir / "run_manifest.json")
        active_admission_dir = dual_admission_dir
    admission_manifest = json.loads((active_admission_dir / "run_manifest.json").read_text())
    checks = {
        "shadow_only": args.mode == "shadow-only",
        "order_submission_disabled": True,
        "exchange_calls_zero": True,
        "target_free_scoring": score_manifest.get("outcome_columns_consumed") == [],
        "no_held_percentiles": score_manifest.get("held_percentile_operations") == 0,
        "exact_lockstep_producer": score_manifest.get("producer_topology") == "exact_lockstep_shared_cutoff",
        "same_bundle_reference_and_held": bool(score_manifest.get("same_bundle_for_reference_and_held")),
        "same_upstream_reference_and_held": bool(score_manifest.get("same_upstream_bundle_for_reference_and_held_per_producer")),
        "frozen_geometry": score_manifest.get("geometry_contract") == "one_frozen_oct_dec_2024_geometry_K9_view_temperature_0.25",
        "geometry_k9_state_checkpoint": (
            (score_dir / "geometry_k9_state" / "run_manifest.json").is_file()
        ),
        "canonical_mc1_admission": (
            dual_mode or bundle.payload.get("admission_contract") == CANONICAL_ADMISSION_MODE
        ),
        "current_outcomes_absent": admission_manifest.get("current_outcomes_consumed") == [],
        "runtime_labels_strictly_prior_day": bool(
            runtime_resolved_manifest.get("strictly_prior_to_utc_day")
        ),
        "current_decision_labels_excluded": (
            int(runtime_resolved_manifest.get("current_decision_ids_appended", -1)) == 0
        ),
        "policy_cost_once": True,
        # In dual mode BCF intentionally scores the whole target-free routed
        # frame, while current-v5 MC1 remains unavailable for rows that fail
        # its frozen complete-base contract.  The latter is a row-local
        # fail-closed condition, not an incomplete BCF map.  Require an exact
        # identity join here; the normal current admission receipt separately
        # proves current eligible_rows == mapped_rows.
        "all_eligible_rows_mapped": (
            int(admission_manifest.get("joined_rows", -1))
            == int(admission_manifest.get("current_rows", -2))
            if dual_mode else
            int(admission_manifest.get("mapped_rows", -1))
            == int(admission_manifest.get("eligible_rows", -2))
        ),
        "feature_complete_fraction_meets_bundle_gate": (
            float(score_manifest.get("held_complete_base_contract_fraction", 0.0))
            >= float(bundle.payload["feature_parity"]["minimum_cycle_complete_fraction"])
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"strict-R3 shadow contract failed: {checks}")

    admitted = pd.read_parquet(active_admission_dir / "admitted_predictions.parquet")
    if "policy_net_bps" in admitted and admitted["policy_net_bps"].notna().any():
        raise AssertionError("shadow decisions unexpectedly contain current outcomes")
    # Execution must consume the same exact decision-open value that made the
    # point-in-time candidate executable.  It is intentionally absent from
    # model features/predictions, so join it back by immutable candidate ID
    # only after scoring and admission have completed.
    held_entry = pd.read_parquet(args.held_candidates)
    required_entry = {"candidate_id", "decision_open", "signal_atr"}
    missing_entry = sorted(required_entry.difference(held_entry.columns))
    if missing_entry:
        raise ValueError(f"held candidates lack execution lineage {missing_entry}")
    held_entry = held_entry[["candidate_id", "decision_open", "signal_atr"]].copy()
    if held_entry["candidate_id"].duplicated().any():
        raise ValueError("held candidate execution lineage is not identity-unique")
    admitted = admitted.merge(
        held_entry,
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    mc1_controller = MC1D2Bundle.load(mc1_bundle_dir)
    adaptive_controller = AdaptiveExitV1Bundle.load(adaptive_exit_bundle_dir)
    for field in ADAPTIVE_EXIT_ENTRY_CONTEXT:
        if field not in admitted:
            admitted[field] = float("nan")
        admitted[f"__adaptive_entry__{field}"] = admitted[field]
    score_history = pd.read_parquet(
        runtime_resolved_dir / "walkforward_scored_label_ledger.parquet",
        columns=["__decision_ts__", "__symbol__", *ADAPTIVE_EXIT_SCORE_FIELDS],
    )
    score_history["__decision_ts__"] = pd.to_datetime(
        score_history["__decision_ts__"], utc=True, errors="raise"
    )
    score_history = score_history[score_history["__decision_ts__"].le(decision_ts)]
    histories: dict[str, tuple[dict[str, object], ...]] = {}
    for symbol, history in score_history.sort_values("__decision_ts__").groupby(
        "__symbol__", sort=False
    ):
        histories[str(symbol)] = tuple(
            {
                "available_at": row["__decision_ts__"].isoformat(),
                **{
                    field: float(pd.to_numeric(row[field], errors="coerce"))
                    for field in ADAPTIVE_EXIT_SCORE_FIELDS
                },
            }
            for _, row in history.tail(3).iterrows()
        )
    admitted["__adaptive_score_history"] = admitted["__symbol__"].astype(str).map(
        histories
    ).map(lambda value: tuple() if not isinstance(value, tuple) else value)
    portfolio_policy = ShadowPortfolioPolicy.from_payload(
        json.loads(portfolio_policy_json.read_text()),
    )
    input_portfolio_state = ShadowPortfolioState.from_payload(
        json.loads(args.portfolio_state_json.read_text()),
        expected_as_of_ts=decision_ts,
    )
    policy_payload = json.loads(execution_policy_json.read_text())
    policy_values = policy_payload.get("winner", policy_payload)
    open_symbols = {position.symbol for position in input_portfolio_state.open_positions}
    bar_root = ROOT / str(bundle.payload["runtime"]["policy_bar_root"])
    state_bars = _load_state_bars(
        bar_root,
        open_symbols,
        start=min(
            [input_portfolio_state.as_of_ts, *[
                position.entry_ts for position in input_portfolio_state.open_positions
                if position.entry_ts is not None
            ]]
        ),
        end=decision_ts,
    )
    # The preceding cycle stamps its output for this decision while retaining
    # each position's next unprocessed 15-minute bar. Advance those completed
    # bars before admission/auction so freed slots and realised wallet P&L are
    # contemporaneous with the new decision.
    portfolio_state, realized_exits = advance_shadow_state(
        input_portfolio_state,
        decision_ts=decision_ts,
        bars_by_symbol=state_bars,
        stop_loss_atr=float(policy_values["sl_mult"]),
        trailing_activation_atr=float(policy_values["trailing_activation_mult"]),
        trailing_giveback_atr=float(policy_values["fixed_trailing_gap_mult"]),
        cost_bps=100.0,
        defer_incomplete_paths=True,
    )
    deferred_exit_symbols = sorted(
        position.symbol for position in portfolio_state.open_positions
        if position.next_bar_ts is not None and position.next_bar_ts < decision_ts
    )
    portfolio_state, adaptive_exit_decisions = apply_adaptive_exit_v1(
        portfolio_state,
        current_candidates=admitted,
        bars_by_symbol=state_bars,
        controller=adaptive_controller,
        baseline_params=policy_values,
    )
    if adaptive_exit_decisions.empty:
        adaptive_exit_decisions = pd.DataFrame(columns=[
            "candidate_id", "adaptive_exit_decision_ts",
            "adaptive_exit_effective_from", "adaptive_exit_f1_prediction",
            "adaptive_exit_f4_prediction", "adaptive_exit_disagreement",
            "adaptive_exit_disagreement_p80", "adaptive_exit_core_activation_atr",
            "adaptive_exit_selected_activation_atr", "adaptive_exit_abstained",
            "adaptive_exit_fallback_to_base", "adaptive_exit_bundle_id",
            "adaptive_exit_fallback_reason",
        ])
    adaptive_exit_decisions.to_parquet(
        args.out_dir / "adaptive_exit_decisions.parquet", index=False, compression="zstd"
    )
    auction = auction_admitted_snapshot(
        admitted, state=portfolio_state, policy=portfolio_policy,
    )
    accepted_symbols = set(
        auction.loc[
            auction["portfolio_accepted"].fillna(False).astype(bool), "__symbol__",
        ].astype(str)
    )
    entry_bars = _load_entry_bars(
        bar_root, accepted_symbols, end=decision_ts,
    )
    next_portfolio_state = add_shadow_entries(
        portfolio_state, auction,
        bars_by_symbol=entry_bars,
        timeout_hours=int(policy_payload.get("timeout_hours", 12)),
        base_trailing_activation_atr=float(policy_values["trailing_activation_mult"]),
    )
    next_portfolio_state = ShadowPortfolioState(
        decision_ts + pd.Timedelta(hours=1),
        next_portfolio_state.wallet,
        next_portfolio_state.open_positions,
    )
    next_state_path = args.out_dir / "next_portfolio_state.json"
    next_state_path.write_text(json.dumps(next_portfolio_state.to_payload(), indent=2) + "\n")
    if realized_exits.empty:
        realized_exits = pd.DataFrame(columns=[
            "candidate_id", "symbol", "side", "entry_ts", "exit_ts",
            "entry_price", "exit_price", "gross_bps", "cost_bps", "net_bps",
            "gross_notional", "wallet_pnl", "exit_reason",
        ])
    realized_exits.to_parquet(args.out_dir / "shadow_exits.parquet", index=False)
    selected = (
        admitted["dual_bcf_current_admitted_ge_30bps"].fillna(False).astype(bool)
        if dual_mode else admitted["mc1_d2_admitted_ge_50bps"].fillna(False).astype(bool)
    )
    decision_columns = [
        "candidate_id", "__decision_ts__", "__symbol__", "side_name",
        "final_score", "causal_21d_side_expected_net_bps",
        "causal_21d_side_admitted_ge_50bps", "frozen_base_contract_complete",
        "trust_posterior_expected_bps", "trust_residual_q25_bps",
        "trust_p_map_overestimate_100bps", "trust_effective_support",
        "trust_risk_corroborated", "trust_authority",
        "trust_corrected_expected_net_bps", "auction_rank_adjustment_bps",
        "trust_posterior_admitted_ge_50bps",
        "a4_raw_expected_bps", "a4_raw_predictive_sd_bps",
        "a4_effective_support", "a4_p_ev_positive_raw",
        "a5_calibrated_expected_bps", "a5_calibrated_p_positive",
        "a5_bounded10_expected_bps", "a5_timestamp_top15",
        "a5_bounded10_available", "a5_bounded10_admitted",
        "robust21_expected_net_bps", "robust21_support_days",
        "robust21_admitted_ge_50bps", "mc1_d2_expected_net_bps",
        "mc1_d2_recent_global_shift_bps", "mc1_d2_available",
        "mc1_d2_admitted_ge_50bps", "mc1_d2_bundle_id",
        "bcf_mc1_expected_net_bps", "bcf_mc1_recent_global_shift_bps",
        "bcf_mc1_available", "bcf_mc1_admitted_ge_30bps", "bcf_mc1_bundle_id",
        "current_mc1_admitted_ge_30bps", "dual_bcf_current_admitted_ge_30bps",
        "dual_auction_priority_bps", "dual_admission_rejection_reason",
        "ev_mapping_vintage_mode", "geometry_bundle_sha256",
        "conversion_bundle_sha256", "upstream_bundle_sha256",
        "base_route_fraction", "base_route_timestamp",
        "base_route_timestamp_top30", "base_route_timestamp_top20",
        "base_route_status",
        # Execution-only lineage is persisted for every MC1-admitted row, not
        # only the shadow-auction winners.  The live executor can therefore
        # preflight all admitted candidates before applying its two-entry cap.
        "decision_open", "signal_atr",
        "portfolio_accepted", "portfolio_rejection_reason",
        "portfolio_priority_rank", "portfolio_initial_margin",
        "portfolio_gross_notional", "portfolio_wallet",
        "portfolio_open_positions_before", "portfolio_committed_margin_before",
        "portfolio_margin_cap", "portfolio_policy_schema", "portfolio_state_schema",
        *CANONICAL_STACK_REPORTING_KEYS,
    ]
    # Historical trust/A5 columns remain optional shadow diagnostics.  They
    # are not inputs to MC1 authority and must not block a canonical cycle.
    decision_columns = list(dict.fromkeys(decision_columns))
    decisions = auction.loc[:, [column for column in decision_columns if column in auction]].copy()
    decisions["policy_parent_name"] = "SimplePolicyOptimiser"
    decisions["policy_sl_atr"] = float(policy_values["sl_mult"])
    decisions["policy_trailing_activation_atr"] = float(
        policy_values["trailing_activation_mult"]
    )
    decisions["policy_trailing_giveback_atr"] = float(
        policy_values["fixed_trailing_gap_mult"]
    )
    decisions["policy_timeout_hours"] = int(policy_payload.get("timeout_hours", 12))
    decisions["policy_cost_bps_once"] = 100.0
    decisions["shadow_action"] = "reject"
    decisions.loc[
        decisions["portfolio_accepted"].fillna(False).astype(bool), "shadow_action",
    ] = "hypothetical_entry"
    decisions["order_submission_enabled"] = False
    decisions["exchange_calls"] = 0
    decisions.to_parquet(
        args.out_dir / "shadow_decisions.parquet", index=False, compression="zstd",
    )
    manifest = {
        "schema": (
            "strict_r3_bcf_current_dual_mc1_rich_exit_shadow_cycle_v1"
            if dual_mode else "strict_r3_robust21_mc1_d2_adaptive_exit_v1_shadow_cycle_v1"
        ),
        "mode": "shadow-only",
        "decision_ts": decision_ts.isoformat(),
        "checks": checks,
        "inference_bundle_audit": bundle_audit,
        "rows": int(len(decisions)),
        "feature_complete_rows": int(decisions["frozen_base_contract_complete"].fillna(False).sum()),
        "mapped_rows": int(decisions["causal_21d_side_expected_net_bps"].notna().sum()),
        "admitted_rows": int(selected.sum()),
        "dual_bcf_current_mode": bool(dual_mode),
        "active_admission_dir": str(active_admission_dir),
        "portfolio_accepted_rows": int(decisions["portfolio_accepted"].sum()),
        "portfolio_open_positions_before": int(len(portfolio_state.open_positions)),
        "portfolio_wallet": float(portfolio_state.wallet),
        "next_portfolio_state_sha256": _sha(next_state_path),
        "next_portfolio_state_as_of_ts": next_portfolio_state.as_of_ts.isoformat(),
        "next_portfolio_open_positions": int(len(next_portfolio_state.open_positions)),
        "realized_exit_rows": int(len(realized_exits)),
        "deferred_exit_symbols_missing_15m": deferred_exit_symbols,
        "adaptive_exit_open_positions_scored": int(len(adaptive_exit_decisions)),
        "adaptive_exit_fallback_rows": int(
            adaptive_exit_decisions.get(
                "adaptive_exit_fallback_to_base", pd.Series(dtype=bool)
            ).fillna(False).sum()
        ),
        "adaptive_exit_bundle_id": adaptive_controller.manifest["bundle_id"],
        "adaptive_exit_bundle_manifest_sha256": _sha(
            adaptive_exit_bundle_dir / "run_manifest.json"
        ),
        "mc1_d2_bundle_id": mc1_controller.manifest["bundle_id"],
        "mc1_d2_bundle_manifest_sha256": _sha(mc1_bundle_dir / "run_manifest.json"),
        "mc1_d2_admitted_rows": int(selected.sum()),
        "robust21_control_admitted_rows": int(
            admitted["robust21_admitted_ge_50bps"].fillna(False).sum()
        ),
        "order_submission_enabled": False,
        "exchange_calls": 0,
        "score_manifest_sha256": _sha(score_dir / "run_manifest.json"),
        "admission_manifest_sha256": _sha(active_admission_dir / "run_manifest.json"),
        "inference_bundle_sha256": _sha(args.inference_bundle),
        "execution_policy_json": str(execution_policy_json),
        "execution_policy_json_sha256": _sha(execution_policy_json),
        "calibration_policy_json": str(calibration_policy_json),
        "calibration_policy_json_sha256": _sha(calibration_policy_json),
        "portfolio_policy_json_sha256": _sha(portfolio_policy_json),
        "portfolio_state_json_sha256": _sha(args.portfolio_state_json),
        "runtime_resolved_state_sha256": _sha(
            runtime_resolved_dir / "walkforward_scored_label_ledger.parquet"
        ),
        "immutable_prediction_prefix_audit": immutable_prediction_prefix_audit,
        "geometry_k9_state_input": (
            str(args.lockstep_geometry_k9_state_in)
            if args.lockstep_geometry_k9_state_in is not None else None
        ),
        "geometry_k9_state_output": str(
            score_dir / "geometry_k9_state" / "run_manifest.json"
        ),
        "geometry_k9_state_mode": score_manifest.get("geometry_k9_state", {}).get("mode"),
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
    )
    print(json.dumps({"event": "complete", **manifest}))


if __name__ == "__main__":
    main()
