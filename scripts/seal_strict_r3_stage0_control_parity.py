#!/usr/bin/env python3
"""Seal the aggregate Stage-0 parity receipt for :00 strict-R3 research.

This is intentionally an offline audit.  It verifies the frozen current-v5
control at the interfaces where a later base, residual, consensus, or mapper
challenger could otherwise drift silently: the ordered 120-field contract,
target-free score replay, native current/BCF MC1 outputs, target-free dual
admission, and the common constrained rich-parent portfolio/exits.

Adaptive Exit V1 has only partial historical OOF controller-state coverage.
The parent-policy receipt is therefore allowed to seal on its own, but the
result deliberately remains ``adaptive_exit_pending`` until a candidate-bound
Adaptive replay is supplied.  A challenger cannot be promoted as a complete
stack on the basis of this parent-only receipt.
"""

from __future__ import annotations

import argparse
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


RTOL = 1e-4
ATOL = 1e-8
ARMS = (
    "parent_proxy_15m_decision",
    "simple_policy_1m_control_decision",
    "frozen_rich_15m_aggregated_decision",
    "exact_1m_rich_v1_decision",
    "exact_1m_rich_v1_plus5",
)

DEFAULT_ROOT = ROOT / "data_perp/artifacts"
DEFAULT_RESEARCH_ROOT = DEFAULT_ROOT / "strict_r3_base_recall_residual2_consensus_research_20260822_v4"
DEFAULT_CONTROL = DEFAULT_ROOT / "strict_r3_base_recall_residual2_consensus_research_20260822_v4_control_parity_full_v2_20260822"
DEFAULT_CURRENT = DEFAULT_ROOT / "strict_r3_base_recall_residual2_consensus_research_20260822_v4_current_mc1_control_parity_20260822"
DEFAULT_BCF = DEFAULT_ROOT / "strict_r3_base_recall_residual2_consensus_research_20260822_v4_mc1_native_control_parity_20260822"
DEFAULT_DUAL = DEFAULT_ROOT / "strict_r3_base_recall_residual2_consensus_research_20260822_v4_dual30_bcf_priority_control_parity_20260822"
DEFAULT_RICH = DEFAULT_ROOT / "strict_r3_base_recall_residual2_consensus_research_20260822_v4_rich_exit_control_parity_20260822"
DEFAULT_CURRENT_REFERENCE = DEFAULT_ROOT / "strict_r3_score_family_matched_mc1_canonical_policy_20260817_v7_current"
DEFAULT_BCF_REFERENCE = DEFAULT_ROOT / "strict_r3_score_family_matched_mc1_canonical_policy_20260817_v7_bcf"
DEFAULT_DUAL_REFERENCE = DEFAULT_ROOT / "strict_r3_exact_1m_dual30_bcf_priority_candidates_decision_2025_2026_20260817_v2"
DEFAULT_RICH_REFERENCE = DEFAULT_ROOT / "strict_r3_exact_1m_rich_matched_attribution_2025_2026_20260817_v8"
DEFAULT_SCORE_CONTROL = DEFAULT_ROOT / "strict_r3_score_family_current_v5_canonical_policy_reconstruction_2025_2026_20260816_v4"
DEFAULT_FEATURE_CONTRACT = ROOT / "config/strict_r3_canonical_v2_feature_contract.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _normalised_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="raise").astype("int64")


def frame_parity(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    sort_by: Iterable[str],
    rtol: float = RTOL,
    atol: float = ATOL,
) -> dict[str, Any]:
    """Compare full saved frames, keeping categorical states exact.

    This is deliberately stricter than comparing score aggregates.  Numeric
    fields use the research contract's 0.01% tolerance; all identities,
    categories, timestamps, null locations, and column order must match.
    """

    if list(actual.columns) != list(expected.columns):
        raise AssertionError("column order or names differ")
    keys = list(sort_by)
    if not keys:
        raise ValueError("a deterministic sort key is required")
    if actual.duplicated(keys).any() or expected.duplicated(keys).any():
        raise AssertionError(f"duplicate parity identities for {keys}")
    left = actual.sort_values(keys, kind="stable").reset_index(drop=True)
    right = expected.sort_values(keys, kind="stable").reset_index(drop=True)
    if len(left) != len(right):
        raise AssertionError(f"row count differs: {len(left)} != {len(right)}")

    categorical: list[str] = []
    numeric: dict[str, dict[str, float | bool]] = {}
    for column in left.columns:
        lhs, rhs = left[column], right[column]
        if pd.api.types.is_datetime64_any_dtype(lhs) or isinstance(lhs.dtype, pd.DatetimeTZDtype):
            same = bool(np.array_equal(_normalised_timestamp(lhs), _normalised_timestamp(rhs)))
            categorical.append(column)
            if not same:
                raise AssertionError(f"timestamp differs for {column}")
        elif pd.api.types.is_bool_dtype(lhs) or not pd.api.types.is_numeric_dtype(lhs):
            same = lhs.equals(rhs)
            categorical.append(column)
            if not same:
                raise AssertionError(f"categorical value differs for {column}")
        else:
            a = pd.to_numeric(lhs, errors="coerce").to_numpy(float)
            b = pd.to_numeric(rhs, errors="coerce").to_numpy(float)
            if not np.array_equal(np.isfinite(a), np.isfinite(b)):
                raise AssertionError(f"finite/null positions differ for {column}")
            finite = np.isfinite(a)
            delta = np.abs(a[finite] - b[finite])
            denominator = np.maximum(np.maximum(np.abs(a[finite]), np.abs(b[finite])), atol)
            relative = delta / denominator
            passed = bool(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True))
            numeric[column] = {
                "passed": passed,
                "max_abs_delta": float(delta.max()) if delta.size else 0.0,
                "max_relative_delta": float(relative.max()) if relative.size else 0.0,
            }
            if not passed:
                raise AssertionError(f"numeric parity differs for {column}: {numeric[column]}")
    return {
        "rows": int(len(left)),
        "identity_keys": keys,
        "categorical_exact_columns": categorical,
        "numeric": numeric,
    }


def _audit_feature_contract(feature_contract: Path, score_control: Path) -> dict[str, Any]:
    expected = tuple(_read_json(feature_contract)["base_fields_by_side"]["long"])
    if len(expected) != 120:
        raise AssertionError(f"declared long feature contract is not 120 fields: {len(expected)}")
    upstream_paths = sorted(score_control.glob("bundles/block=*/upstream/monthly_upstream_bundle.joblib"))
    conversion_paths = sorted(score_control.glob("bundles/block=*/conversion/four_week_conversion_bundle.joblib"))
    if not upstream_paths or len(upstream_paths) != len(conversion_paths):
        raise AssertionError("incomplete upstream/conversion bundle collection")
    for path in [*upstream_paths, *conversion_paths]:
        bundle = joblib.load(path)
        if tuple(bundle.base_fields) != expected:
            raise AssertionError(f"feature order mismatch in {path}")
        if hasattr(bundle, "base_medians") and len(bundle.base_medians) != len(expected):
            raise AssertionError(f"base median width mismatch in {path}")
    return {
        "status": "passed",
        "declared_fields": len(expected),
        "ordered_field_sha256": hashlib.sha256("\n".join(expected).encode()).hexdigest(),
        "upstream_bundle_count": len(upstream_paths),
        "conversion_bundle_count": len(conversion_paths),
    }


def _audit_target_free_score_control(control: Path) -> dict[str, Any]:
    receipt = _read_json(control / "control_parity.json")
    blocks = json.loads((control / "control_parity_blocks.json").read_text())
    if receipt.get("status") != "passed":
        raise AssertionError("target-free score control receipt did not pass")
    if receipt.get("block_count_completed") != receipt.get("block_count_requested"):
        raise AssertionError("not every control block completed")
    if len(blocks) != receipt["block_count_completed"]:
        raise AssertionError("control block receipt count mismatch")
    for block in blocks:
        if not all(block.get("identity", {}).values()):
            raise AssertionError(f"identity parity failed in {block['block']}")
        if not all(value.get("passed", False) for value in block.get("numeric", {}).values()):
            raise AssertionError(f"numeric parity failed in {block['block']}")
        if not all(value.get("passed", False) for value in block.get("boolean", {}).values()):
            raise AssertionError(f"categorical parity failed in {block['block']}")
        if block.get("head", {}).get("head_count") != 10 or not block["head"].get("median_matches_reconstructed_aggregate"):
            raise AssertionError(f"head consensus parity failed in {block['block']}")
        if not all(block.get("bundle_identity", {}).values()):
            raise AssertionError(f"bundle identity parity failed in {block['block']}")
        for conversion in block.get("conversion_audit", []):
            required = {
                "same_conversion_model_reference_and_held": True,
                "upstream_scores_are_prequential_monthly": True,
                "geometry_refit_cadence": "never",
                "held_percentile_operations": 0,
            }
            for key, value in required.items():
                if conversion.get(key) != value:
                    raise AssertionError(f"conversion invariant {key} failed in {block['block']}")
    return {
        "status": "passed",
        "blocks": int(len(blocks)),
        "route_fraction": receipt["base_route_fraction"],
        "same_model_reserve_days": receipt["same_model_reserve_days"],
        "source_sha256": receipt["source"]["sha256"],
    }


def _audit_mc1(actual: Path, reference: Path, *, name: str) -> dict[str, Any]:
    frame = pd.read_parquet(actual)
    expected = pd.read_parquet(reference)
    result = frame_parity(frame, expected, sort_by=("candidate_id",))
    families = sorted(frame["family"].dropna().unique().tolist())
    expected_family = "bcf" if name == "bcf" else "current_v5"
    if families != [expected_family]:
        raise AssertionError(f"{name} native mapper identity is invalid: {families}")
    result["family"] = expected_family
    result["actual_sha256"] = _sha256(actual)
    result["reference_sha256"] = _sha256(reference)
    return result


def _audit_dual(actual: Path, reference: Path) -> dict[str, Any]:
    result = frame_parity(
        pd.read_parquet(actual), pd.read_parquet(reference), sort_by=("candidate_id",),
    )
    result["actual_sha256"] = _sha256(actual)
    result["reference_sha256"] = _sha256(reference)
    return result


def _audit_rich(actual_root: Path, reference_root: Path) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for arm in ARMS:
        name = f"{arm}_accepted_trades.parquet"
        actual, reference = actual_root / name, reference_root / name
        if not actual.is_file() or not reference.is_file():
            raise FileNotFoundError(f"missing rich replay output for {arm}")
        result = frame_parity(
            pd.read_parquet(actual), pd.read_parquet(reference),
            sort_by=("candidate_id", "decision_timestamp", "arm"),
        )
        result["actual_sha256"] = _sha256(actual)
        result["reference_sha256"] = _sha256(reference)
        arms[arm] = result
    return {"status": "passed", "arms": arms}


def _file_manifest(paths: dict[str, Path]) -> dict[str, dict[str, str]]:
    return {name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESEARCH_ROOT / "stage0_control_parity_receipt_20260822_v1")
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--bcf", type=Path, default=DEFAULT_BCF)
    parser.add_argument("--dual", type=Path, default=DEFAULT_DUAL)
    parser.add_argument("--rich", type=Path, default=DEFAULT_RICH)
    parser.add_argument("--current-reference", type=Path, default=DEFAULT_CURRENT_REFERENCE)
    parser.add_argument("--bcf-reference", type=Path, default=DEFAULT_BCF_REFERENCE)
    parser.add_argument("--dual-reference", type=Path, default=DEFAULT_DUAL_REFERENCE)
    parser.add_argument("--rich-reference", type=Path, default=DEFAULT_RICH_REFERENCE)
    parser.add_argument("--score-control", type=Path, default=DEFAULT_SCORE_CONTROL)
    parser.add_argument("--feature-contract", type=Path, default=DEFAULT_FEATURE_CONTRACT)
    args = parser.parse_args()

    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable audit directory: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    try:
        feature = _audit_feature_contract(args.feature_contract, args.score_control)
        score = _audit_target_free_score_control(args.control)
        mc1 = {
            "current_native": _audit_mc1(
                args.current / "predictions_current_v5_mc1_d2.parquet",
                args.current_reference / "predictions_current_v5_mc1_d2.parquet",
                name="current",
            ),
            "bcf_native": _audit_mc1(
                args.bcf / "predictions_bcf_mc1_d2.parquet",
                args.bcf_reference / "predictions_bcf_mc1_d2.parquet",
                name="bcf",
            ),
        }
        dual = _audit_dual(args.dual / "candidates.parquet", args.dual_reference / "candidates.parquet")
        rich = _audit_rich(args.rich, args.rich_reference)
        manifest = {
            "schema": "strict_r3_stage0_aggregate_control_parity_v1",
            "status": "parent_policy_control_passed_adaptive_exit_pending",
            "scope": "offline long-only :00 research audit; no live/canonical/order-capable artifact modified",
            "numeric_tolerance": {"relative": RTOL, "absolute": ATOL},
            "feature_contract": feature,
            "target_free_current_score": score,
            "native_mc1": mc1,
            "dual_admission_and_auction": dual,
            "common_portfolio_and_parent_exits": rich,
            "adaptive_exit_v1": {
                "status": "pending_candidate_bound_exact_oof_replay",
                "reason": "The existing rich-parent parity artifact has no candidate-bound Adaptive V1 output. Parent fallback is exactly reproduced; Adaptive must be replayed only where its exact OOF state exists.",
                "promotion_block": True,
            },
            "result": "No challenger may be accepted until this receipt and the later Adaptive subset replay are both passing.",
        }
        (args.out_dir / "stage0_control_parity.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"event": "stage0_parent_control_parity_passed", "out_dir": str(args.out_dir)}, sort_keys=True))
    except Exception as exc:
        failure = {
            "schema": "strict_r3_stage0_aggregate_control_parity_v1",
            "status": "failed",
            "scope": "offline long-only :00 research audit",
            "failure": f"{type(exc).__name__}: {exc}",
        }
        (args.out_dir / "stage0_control_parity.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        raise


if __name__ == "__main__":
    main()
