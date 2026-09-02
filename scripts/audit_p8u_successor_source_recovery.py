#!/usr/bin/env python3
"""Audit recoverability of the P8U C0/C1 successor source contract.

This is deliberately metadata-only: it reads no candidate rows, outcomes,
models, or exchange endpoints.  It distinguishes exact historical artefacts
from retained raw/preprocessing inputs that could support a separately named
successor retrain.  It never creates a model, substitutes a feature contract,
or grants inference/exchange authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


EXACT_HISTORICAL = {
    "router_contract": "data_perp/artifacts/strict_r3_p8u_router_oof_apr25_jul26_successorlabels_20260828_v1/run_contract.json",
    "base_hpo": "data_perp/artifacts/strict_r3_p8u_precision_preservation_hpo_raw_cat_20260827_v2/run_manifest.json",
    "under_f120_contract": "data_perp/artifacts/strict_r3_p8u_meta_under_fullfeatures_selection_20260828_v2/contracts/under_f120.json",
    "router_feature_panels": "data_perp/artifacts/strict_r3_f72_router_feature_source_contiguous_20260826_v1",
    "base_meta_feature_panels_early": "data_perp/artifacts/strict_r3_incumbent_meta_causal_features_20260827_preaug_v1",
    "base_meta_feature_panels_late": "data_perp/artifacts/strict_r3_incumbent_meta_causal_features_20260827_v1",
    "policy_label_ledger": "data_perp/artifacts/strict_r3_p8u_router_policy_label_successor_20260828_v1/canonical_reconciled_policy_labels.parquet",
    "base_target_free_ledger": "data_perp/artifacts/strict_r3_p8u_tail125_base_history_aug25_jul26_successorlabels_20260828_v1/scheme=tail_linear_125/target_free_scores",
    "dual_mc1_packages": "data_perp/artifacts/strict_r3_p8u_f72_underf120_dual_mc1_sixmonth_aug25_aug26_20260828_v4",
}

RETAINED_RECOVERY_INPUTS = {
    "source_anchored_160_symbol_hourly_panel": "data_perp/artifacts/strict_r3_p8u_canonical_source_state_20260828_v1/source_panel_state.joblib",
    "f72_feature_contract": "config/strict_r3_p8u_f72_feature_contract_20260828_v1.json",
    "under_model_geometry": "config/strict_r3_p8u_meta_under_fullfeature_xendcg_20260828_v1.json",
    "raw_hourly_ohlcv": "data_perp/ohlcv",
    "raw_15m_ohlcv": "data_perp/15m_ohlcv_perp/ohlcv",
    "raw_execution_1m_ohlcv": "data_perp/exchanges/krakenfutures/execution_1m/ohlcv",
    "supportive_path_labels": "data_perp/artifacts/strict_r3_long_supportive_path_labels_2024_2026_20260823_v6_observed_entry/parts",
    "f72_base_selection": "data_perp/artifacts/strict_r3_b0_family_addback_20260826_v1_policy_ordinal_base_g3/selection.json",
    "c1_source_bundle": "data_perp/artifacts/causal_sr_c1_lva_inference_bundle_20260901_v2_current/bundle_manifest.json",
    "c1_state_advance": "data_perp/artifacts/causal_sr_c1_state_advance_20260901_v2_patched_through_1915/run_manifest.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _entry(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    result: dict[str, Any] = {
        "path": relative,
        "exists": path.exists(),
        "kind": "missing" if not path.exists() else ("file" if path.is_file() else "directory"),
    }
    if path.is_file():
        result["sha256"] = _sha256(path)
        result["bytes"] = path.stat().st_size
    elif path.is_dir():
        entries = list(path.iterdir())
        result["direct_entry_count"] = len(entries)
        result["parquet_files"] = sum(1 for item in path.rglob("*.parquet"))
    return result


def _source_panel_summary() -> dict[str, Any]:
    path = ROOT / RETAINED_RECOVERY_INPUTS["source_anchored_160_symbol_hourly_panel"]
    if not path.is_file():
        return {"available": False}
    payload = joblib.load(path)
    panel = payload.get("panel") if isinstance(payload, dict) else None
    close = panel.get("close") if isinstance(panel, dict) else None
    if not isinstance(close, pd.DataFrame):
        return {"available": False, "reason": "missing close panel"}
    index = pd.DatetimeIndex(close.index)
    return {
        "available": True,
        "symbols": int(len(payload.get("symbols") or ())),
        "rows": int(len(index)),
        "start": index.min().isoformat(),
        "end_inclusive": index.max().isoformat(),
        "hourly_contiguous": bool(index.equals(pd.date_range(index.min(), index.max(), freq="h", tz="UTC"))),
        "sha256": _sha256(path),
    }


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"immutable preflight already exists: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    exact = {name: _entry(path) for name, path in EXACT_HISTORICAL.items()}
    retained = {name: _entry(path) for name, path in RETAINED_RECOVERY_INPUTS.items()}
    missing_exact = sorted(name for name, item in exact.items() if not item["exists"])
    available_recovery = sorted(name for name, item in retained.items() if item["exists"])
    payload = {
        "schema": "p8u-c0-c1-successor-source-recovery-preflight-v1",
        "scope": "metadata-only source recovery audit; no row reads except retained source-panel coverage; no outcomes/models/inference/exchange authority",
        "status": "exact_historical_reconstruction_blocked" if missing_exact else "exact_historical_inputs_present",
        "exact_historical": exact,
        "retained_recovery_inputs": retained,
        "source_panel_summary": _source_panel_summary(),
        "missing_exact_contract_components": missing_exact,
        "available_successor_rebuild_inputs": available_recovery,
        "allowed_next_action": (
            "restore every exact historical component from an external backup, or create a separately named source-aligned successor and validate it independently"
        ),
        "prohibited": [
            "substituting a similarly named Under-F120 contract",
            "claiming historical bit parity from a successor retrain",
            "enabling exchange or order authority",
        ],
    }
    _write_exclusive(args.out, payload)
    print(json.dumps({
        "status": payload["status"],
        "missing_exact": len(missing_exact),
        "retained_recovery_inputs": len(available_recovery),
        "out": str(args.out),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
