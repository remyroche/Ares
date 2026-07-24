#!/usr/bin/env python3
"""Promote a causal pre-MLP recent-EV threshold policy into a frozen bundle.

The policy is intentionally a separate admission layer.  It preserves the
frozen V9 residual overlay and the frozen MLP/hierarchical-EV calibration, then
uses only completed historical OOS rows to modulate final rank admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
POLICY_ID = "ev_target_side_archetype_global_top10_before_mlp_28d_flat_v1"
POLICY_NAME = "s52_v9_tail95_mlp_hierev_evtarget28d_prempl_top10_v1"
FAMILY = "ev_target_side_archetype_multiplier_before_mlp"
DEFAULT_BUNDLE = ROOT / "data_perp/artifacts/s59_s52_finalfit_sharedchampion_v9tail95_mlp_hierev_20260713"
DEFAULT_HISTORY = ROOT / "data_perp/reports/meta_complete_monthly_v9_mlp_hierev_20260713/walkforward_rank_history.parquet"
DEFAULT_ABLATION = ROOT / "data_perp/reports/meta_recent_ev_target_mapping_ablation_20260713"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reference_rows(history_path: Path) -> pd.DataFrame:
    columns = [
        "__ts__",
        "__symbol__",
        "side_name",
        "archetype_policy_key",
        "policy_parent_rank",
        "rank_mlp_direct",
        "ev_after_1pct",
        "__fold__",
    ]
    rows = pd.read_parquet(history_path, columns=columns).copy()
    rows = rows.rename(columns={"__ts__": "timestamp", "__symbol__": "symbol"})
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True, errors="coerce")
    rows["policy_archetype"] = rows["archetype_policy_key"].astype(str)
    rows = rows.dropna(
        subset=["timestamp", "side_name", "policy_archetype", "policy_parent_rank", "rank_mlp_direct", "ev_after_1pct"]
    ).sort_values("timestamp", kind="stable")
    if rows.empty:
        raise ValueError("no usable causal reference rows")
    return rows.loc[
        :,
        [
            "timestamp", "symbol", "side_name", "policy_archetype",
            "policy_parent_rank", "rank_mlp_direct", "ev_after_1pct", "__fold__",
        ],
    ]


def _policy_payload(
    reference_path: Path,
    rows: pd.DataFrame,
    ablation_dir: Path,
    history_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "threshold_basis_policy_v2",
        "policy_id": POLICY_ID,
        "policy_name": POLICY_NAME,
        "arm": "before_mlp_evtarget_28d_flat",
        "enabled": True,
        "live_compatible_selection": True,
        "family": FAMILY,
        "notes": (
            "Canonical causal admission layer. It learns a global top-10% EV target "
            "from completed OOS history, estimates side x archetype thresholds from "
            "the preceding 28 complete days using policy_parent_rank, then applies the "
            "support-shrunk multiplier to the final MLP expected-EV rank."
        ),
        "window_days": 28,
        "smoothing": "flat",
        "top_fraction": 0.10,
        "min_reference_rows": 40,
        "local_support_target": 160.0,
        "multiplier_min": 0.50,
        "multiplier_max": 1.50,
        "calibration_reference_score_col": "policy_parent_rank",
        "apply_reference_score_col": "rank_mlp_direct",
        "live_score_col": "expected_ev_rank_score",
        "return_col": "ev_after_1pct",
        "selection_group": "timestamp",
        "reference_candidates_path": reference_path.name,
        "reference_columns": list(rows.columns),
        "reference_rows": int(len(rows)),
        "reference_timestamp_min": rows["timestamp"].min().isoformat(),
        "reference_timestamp_max": rows["timestamp"].max().isoformat(),
        "cost_contract": "ev_after_1pct contains the sole 1% round-trip cost; no extra fee is subtracted.",
        "causal_contract": (
            "At decision timestamp t, only reference rows with timestamp < t may be used. "
            "The local calibration window is [t-28d, t)."
        ),
        "score_contract": (
            "Threshold quality is measured from parent rank; the support-shrunk multiplier "
            "is applied to the frozen final MLP expected-EV rank."
        ),
        "source_ablation_dir": str(ablation_dir.relative_to(ROOT)),
        "source_rank_history": str(history_path.relative_to(ROOT)),
    }


def _patch_policy_block(payload: dict[str, Any], policy_path: str) -> None:
    payload.update(
        {
            "policy_name": POLICY_NAME,
            "threshold_basis_policy_enabled": True,
            "threshold_basis_policy_id": POLICY_ID,
            "threshold_basis_policy_path": policy_path,
            "threshold_basis_family": FAMILY,
            "threshold_basis_window_days": 28,
            "source_threshold_basis_policy": policy_path,
        }
    )
    selection = payload.get("selection")
    if isinstance(selection, dict):
        _patch_policy_block(selection, policy_path)


def promote(bundle: Path, history: Path, ablation_dir: Path) -> Path:
    policy_dir = bundle / "policy_params"
    config_path = policy_dir / "optimized_portfolio_policy_config.json"
    manifest_path = policy_dir / "promoted_policy_manifest.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not (ablation_dir / "manifest.json").exists():
        raise FileNotFoundError(ablation_dir / "manifest.json")
    rows = _reference_rows(history)
    reference_path = policy_dir / "threshold_basis_reference_candidates_evtarget28d_prempl.parquet"
    rows.to_parquet(reference_path, index=False, compression="zstd")
    policy_path = policy_dir / "threshold_basis_policy_evtarget28d_prempl.json"
    _write(policy_path, _policy_payload(reference_path, rows, ablation_dir, history))

    relative_policy = str(policy_path.relative_to(ROOT))
    config = _load(config_path)
    _patch_policy_block(config, relative_policy)
    _write(config_path, config)

    manifest = _load(manifest_path) if manifest_path.exists() else {}
    manifest.update(
        {
            "policy_name": POLICY_NAME,
            "canonical_admission_policy_id": POLICY_ID,
            "canonical_admission_policy_path": relative_policy,
            "threshold_basis_policy_enabled": True,
            "threshold_basis_policy_id": POLICY_ID,
            "threshold_basis_policy_path": relative_policy,
            "threshold_basis_family": FAMILY,
            "threshold_basis_window_days": 28,
            "rolling_8d_modulator_enabled": False,
            "admission_contract": "causal side x archetype EV-target mapping + final MLP expected-EV rank >= historical top10 cutoff",
        }
    )
    manifest["file_sha256"] = {
        str(path.relative_to(bundle)): _sha256(path)
        for path in sorted(policy_dir.rglob("*"))
        if path.is_file() and path != manifest_path
    }
    _write(manifest_path, manifest)
    return policy_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--ablation-dir", type=Path, default=DEFAULT_ABLATION)
    args = parser.parse_args()
    policy_path = promote(args.bundle.resolve(), args.history.resolve(), args.ablation_dir.resolve())
    print(policy_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
