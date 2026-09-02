#!/usr/bin/env python3
"""Materialise target-free candidates for the BCF/current-v5 dual-MC1 route.

The two frozen v7 score panels are deliberately separate candidate routes.
This producer keeps only their exact identity intersection and applies the
predeclared *score-only* contract:

    BCF MC1 expected net >= 30 bps
    AND current-v5 MC1 expected net >= 30 bps
    → portfolio priority = BCF MC1 expected net

Policy outcomes are read solely to prove that the independently attached
canonical policy contract agrees across the two panels.  They are never used
to decide candidate membership and never persisted in the request/candidate
panels.  Future exact-one-minute paths are materialised only after this
immutable target-free request is sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.exact_1m_policy_contract import (  # noqa: E402
    Exact1mExecutionContract,
)


DEFAULT_BCF = ROOT / (
    "data_perp/artifacts/strict_r3_score_family_matched_mc1_canonical_policy_"
    "20260817_v7_bcf/predictions_bcf_mc1_d2.parquet"
)
DEFAULT_CURRENT = ROOT / (
    "data_perp/artifacts/strict_r3_score_family_matched_mc1_canonical_policy_"
    "20260817_v7_current/predictions_current_v5_mc1_d2.parquet"
)
DEFAULT_OUT = ROOT / (
    "data_perp/artifacts/strict_r3_exact_1m_dual30_bcf_priority_candidates_"
    "2025_2026_20260817_v1"
)

POLICY_FIELDS = (
    "policy_path_valid",
    "policy_gross_bps",
    "policy_net_bps",
    "policy_exit_bar_15m",
    "policy_entry_price",
    "policy_exit_price",
    "policy_exit_reason",
    "policy_label_available_ts",
    "policy_outcome_source",
    "policy_cost_bps",
)
IDENTITY_FIELDS = ("__decision_ts__", "__symbol__", "side_name")
BCF_FEATURE_FIELDS = (
    "final_score",
    "base_rank42",
    "conditional_consensus_rank",
    "upstream",
    "ordinary_shadow_consensus_rank",
    "correctness_rank",
)
THRESHOLD_BPS = 30.0
CONTRACT_VERSION = "strict_r3_dual30_bcf_priority_targetfree_candidates_v1"
REQUEST_SCHEMA_V1 = "strict_r3_exact_1m_dual_admission_download_request_v1"
REQUEST_SCHEMA_V2 = "strict_r3_exact_1m_dual_admission_download_request_v2"
REQUIRED_DOWNLOADER_WARMUP_MINUTES = 100 * 60


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _read(path: Path, *, family: str) -> pd.DataFrame:
    columns = [
        "candidate_id", *IDENTITY_FIELDS, "mc1_expected_bps", *BCF_FEATURE_FIELDS,
        *POLICY_FIELDS,
    ]
    if family != "bcf":
        columns = [column for column in columns if column not in BCF_FEATURE_FIELDS]
    frame = pd.read_parquet(path, columns=columns)
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["__decision_ts__"] = pd.to_datetime(
        frame["__decision_ts__"], utc=True, errors="raise"
    )
    frame["policy_label_available_ts"] = pd.to_datetime(
        frame["policy_label_available_ts"], utc=True, errors="raise"
    )
    if frame["candidate_id"].duplicated().any():
        raise AssertionError(f"{family} panel has duplicate candidate IDs")
    if frame.loc[:, ["candidate_id", *IDENTITY_FIELDS]].isna().any().any():
        raise AssertionError(f"{family} panel has incomplete candidate identity")
    if not np.isfinite(pd.to_numeric(frame["mc1_expected_bps"], errors="coerce")).all():
        raise AssertionError(f"{family} panel has non-finite MC1 expected EV")
    return frame


def _series_equal(left: pd.Series, right: pd.Series, field: str) -> bool:
    if pd.api.types.is_numeric_dtype(left):
        return bool(np.isclose(
            pd.to_numeric(left, errors="coerce").to_numpy(float),
            pd.to_numeric(right, errors="coerce").to_numpy(float),
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        ).all())
    if pd.api.types.is_datetime64_any_dtype(left) or pd.api.types.is_datetime64_any_dtype(right):
        lhs = pd.to_datetime(left, utc=True, errors="raise")
        rhs = pd.to_datetime(right, utc=True, errors="raise")
        return lhs.equals(rhs)
    return left.fillna("__null__").astype(str).equals(right.fillna("__null__").astype(str))


def _match_and_audit(bcf: pd.DataFrame, current: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    merged = bcf.merge(
        current,
        on="candidate_id",
        how="inner",
        suffixes=("_bcf", "_current"),
        validate="one_to_one",
    )
    if merged.empty:
        raise RuntimeError("BCF/current-v5 panels have no matched candidate IDs")
    audited = [*IDENTITY_FIELDS, *POLICY_FIELDS]
    mismatches: list[str] = []
    for field in audited:
        if not _series_equal(merged[f"{field}_bcf"], merged[f"{field}_current"], field):
            mismatches.append(field)
    if mismatches:
        raise AssertionError(
            "BCF/current-v5 candidate or canonical-policy contract mismatch: "
            f"{mismatches}"
        )
    audit = {
        "bcf_rows": int(len(bcf)),
        "current_v5_rows": int(len(current)),
        "matched_rows": int(len(merged)),
        "bcf_only_rows": int(len(bcf) - len(merged)),
        "current_v5_only_rows": int(len(current) - len(merged)),
        "identity_and_policy_fields_asserted_equal": audited,
        "equivalence_status": "pass",
    }
    return merged, audit


def _target_free_candidates(merged: pd.DataFrame, contract: Exact1mExecutionContract) -> pd.DataFrame:
    """Apply only frozen MC1 values; do not inspect path/outcome columns here."""
    bcf_ev = pd.to_numeric(merged["mc1_expected_bps_bcf"], errors="raise")
    current_ev = pd.to_numeric(merged["mc1_expected_bps_current"], errors="raise")
    selected = merged.loc[
        bcf_ev.ge(THRESHOLD_BPS) & current_ev.ge(THRESHOLD_BPS)
    ].copy()
    if selected.empty:
        raise RuntimeError("dual +30 bps target-free contract selected no candidates")
    result = pd.DataFrame({
        "candidate_id": selected["candidate_id"].astype(str),
        "timestamp": selected["__decision_ts___bcf"],
        "symbol": selected["__symbol___bcf"].astype(str),
        "side_name": selected["side_name_bcf"].astype(str),
        "entry_ts": selected["__decision_ts___bcf"] + pd.Timedelta(
            minutes=int(contract.entry_delay_minutes)
        ),
        "bcf_mc1_expected_bps": bcf_ev.loc[selected.index].to_numpy(float),
        "current_v5_mc1_expected_bps": current_ev.loc[selected.index].to_numpy(float),
        "priority_bps": bcf_ev.loc[selected.index].to_numpy(float),
        "mapped_expected_net_bps": bcf_ev.loc[selected.index].to_numpy(float),
        "bcf_final_score": pd.to_numeric(selected["final_score"], errors="raise"),
        "bcf_base_rank42": pd.to_numeric(selected["base_rank42"], errors="raise"),
        "bcf_conditional_consensus_rank": pd.to_numeric(
            selected["conditional_consensus_rank"], errors="raise"
        ),
        "bcf_upstream": pd.to_numeric(selected["upstream"], errors="raise"),
        "bcf_ordinary_shadow_consensus_rank": pd.to_numeric(
            selected["ordinary_shadow_consensus_rank"], errors="raise"
        ),
        "bcf_correctness_rank": pd.to_numeric(selected["correctness_rank"], errors="raise"),
        "auction_priority_source": "bcf_mc1_expected_bps",
        "admission_contract": "bcf_mc1_expected_bps>=30 AND current_v5_mc1_expected_bps>=30",
        "contract_version": CONTRACT_VERSION,
    })
    if result["candidate_id"].duplicated().any():
        raise AssertionError("target-free dual candidate panel has duplicate candidate IDs")
    if not result["side_name"].eq("long").all():
        raise AssertionError("dual candidate panel unexpectedly contains a non-long row")
    return result.sort_values(["timestamp", "priority_bps", "candidate_id"], ascending=[True, False, True], kind="stable").reset_index(drop=True)


def _monthly_coverage(candidates: pd.DataFrame) -> pd.DataFrame:
    output = candidates.copy()
    output["month"] = output["timestamp"].dt.strftime("%Y-%m")
    return output.groupby("month", sort=True).agg(
        candidates=("candidate_id", "size"),
        decisions=("timestamp", "nunique"),
        symbols=("symbol", "nunique"),
        bcf_mc1_mean_bps=("bcf_mc1_expected_bps", "mean"),
        current_v5_mc1_mean_bps=("current_v5_mc1_expected_bps", "mean"),
    ).reset_index()


def _entry_timestamp_semantics(entry_delay_minutes: int) -> str:
    if int(entry_delay_minutes) == 0:
        return "UTC decision timestamp; entry_ts equals decision timestamp (uniform zero-minute delay)"
    return (
        "UTC decision timestamp; entry_ts is uniformly decision timestamp + "
        f"{int(entry_delay_minutes)} minutes"
    )


def materialize(args: argparse.Namespace) -> Path:
    output = Path(args.out_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite immutable output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    contract = Exact1mExecutionContract(entry_delay_minutes=int(args.entry_delay_minutes))
    contract.validate()
    bcf_path, current_path = Path(args.bcf).resolve(), Path(args.current_v5).resolve()
    bcf, current = _read(bcf_path, family="bcf"), _read(current_path, family="current_v5")
    merged, equivalence = _match_and_audit(bcf, current)
    candidates = _target_free_candidates(merged, contract)
    candidates_path = output / "candidates.parquet"
    candidates.to_parquet(candidates_path, index=False, compression="zstd")
    request_columns = [
        "candidate_id", "timestamp", "symbol", "side_name", "entry_ts", "priority_bps",
    ]
    request_path = output / "candidate_download_request.parquet"
    candidates.loc[:, request_columns].to_parquet(request_path, index=False, compression="zstd")
    monthly = _monthly_coverage(candidates)
    monthly.to_parquet(output / "monthly_coverage.parquet", index=False, compression="zstd")
    request_revision = int(args.request_revision)
    if request_revision not in (1, 2):
        raise ValueError("request-revision must be 1 or 2")
    request_manifest = {
        "schema": REQUEST_SCHEMA_V2 if request_revision == 2 else REQUEST_SCHEMA_V1,
        "request_revision": request_revision,
        "target_free": True,
        "selection_inputs": ["bcf_mc1_expected_bps", "current_v5_mc1_expected_bps"],
        "selection_predicate": "bcf_mc1_expected_bps >= 30 AND current_v5_mc1_expected_bps >= 30",
        "forbidden_selection_inputs": ["policy_path_valid", "policy_net_bps", "outcome", "label"],
        "auction_priority": "priority_bps = bcf_mc1_expected_bps",
        "rows": int(len(candidates)),
        "schema_columns": request_columns,
        "candidate_sha256": _sha256(request_path),
        "contract_hash": contract.hash,
        "entry_delay_minutes": int(contract.entry_delay_minutes),
        "horizon_minutes": int(contract.horizon_minutes),
        "required_downloader_warmup_minutes": REQUIRED_DOWNLOADER_WARMUP_MINUTES,
        "required_downloader_horizon_minutes": int(
            contract.entry_delay_minutes + contract.horizon_minutes
        ),
        "downloader_window": "[timestamp-warmup, timestamp+horizon), timestamp is the UTC decision time",
        "timestamp_semantics": _entry_timestamp_semantics(int(contract.entry_delay_minutes)),
    }
    (output / "candidate_download_request.json").write_text(
        json.dumps(_json_safe(request_manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": CONTRACT_VERSION,
        "status": "complete_target_free_research_candidate_panel",
        "purpose": "exact-1m rich-policy research source; not a live admission or execution artifact",
        "sources": {
            "bcf_predictions": {"path": str(bcf_path.relative_to(ROOT)), "sha256": _sha256(bcf_path)},
            "current_v5_predictions": {"path": str(current_path.relative_to(ROOT)), "sha256": _sha256(current_path)},
        },
        "identity_contract": "inner candidate_id join; UTC decision timestamp, symbol, side and canonical policy fields asserted equal",
        "policy_equivalence_audit": equivalence,
        "selection": {
            "target_free": True,
            "inputs": request_manifest["selection_inputs"],
            "predicate": request_manifest["selection_predicate"],
            "priority": request_manifest["auction_priority"],
            "explicitly_not_used": request_manifest["forbidden_selection_inputs"],
        },
        "exact_1m_execution_contract": {**contract.to_dict(), "hash": contract.hash},
        "artifacts": {
            "candidates": {"path": "candidates.parquet", "sha256": _sha256(candidates_path), "rows": int(len(candidates))},
            "download_request": {"path": "candidate_download_request.parquet", "sha256": _sha256(request_path)},
            "download_request_manifest": {"path": "candidate_download_request.json", "sha256": _sha256(output / "candidate_download_request.json")},
            "monthly_coverage": {"path": "monthly_coverage.parquet", "sha256": _sha256(output / "monthly_coverage.parquet")},
        },
        "coverage": {
            "start": candidates["timestamp"].min(),
            "end": candidates["timestamp"].max(),
            "rows": int(len(candidates)),
            "symbols": int(candidates["symbol"].nunique()),
            "decision_timestamps": int(candidates["timestamp"].nunique()),
            "side": "long",
        },
        "code": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": _sha256(Path(__file__).resolve())},
    }
    (output / "run_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bcf", type=Path, default=DEFAULT_BCF)
    parser.add_argument("--current-v5", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--entry-delay-minutes", type=int, default=5)
    parser.add_argument("--request-revision", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps({"out_dir": str(materialize(args))}, sort_keys=True))


if __name__ == "__main__":
    main()
