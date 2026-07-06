#!/usr/bin/env python3
"""Extract AE/GMM live-predictable archetype diagnostics from LGBM model sidecars."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _load_sidecar(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("rb") as handle:
            obj = pickle.load(handle)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _model_name_from_sidecar(path: Path, run_root: Path) -> str:
    try:
        rel = path.relative_to(run_root)
    except Exception:
        rel = path
    parts = list(rel.parts)
    if "lgbm_reference" in parts:
        idx = parts.index("lgbm_reference")
        tail = parts[idx + 1 :]
        if len(tail) >= 2:
            return str(tail[1])
    return path.parent.parent.name


def _selected_config_rows(
    *,
    model_name: str,
    layer: str,
    sidecar_path: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    config = dict(state.get("selected_config", {}) or {})
    return {
        "model_name": model_name,
        "layer": layer,
        "sidecar_path": str(sidecar_path),
        "enabled": bool(state.get("enabled", False)),
        "reason": str(state.get("reason", "")),
        "input_feature_count": len(state.get("feature_columns", []) or []),
        "generated_feature_count": len(state.get("latent_columns", []) or []) + int(state.get("max_components", 0) or 0),
        "gmm_n_components": _as_int(state.get("gmm_n_components")),
        "gmm_reg_covar": _as_float(state.get("gmm_reg_covar")),
        "smooth_lambda": _as_float(state.get("smooth_lambda")),
        "reconstruction_error_mean": _as_float(state.get("reconstruction_error_mean")),
        "reconstruction_error_std": _as_float(state.get("reconstruction_error_std")),
        "hpo_report_count": _as_int(state.get("hpo_report_count")),
        "selected_final_score": _as_float(config.get("final_score")),
        "selected_economic_regime_separation": _as_float(config.get("economic_regime_separation")),
        "selected_target_signature_score": _as_float(config.get("target_signature_score")),
        "selected_target_signature_stability": _as_float(config.get("target_signature_stability")),
        "selected_target_signature_contrast": _as_float(config.get("target_signature_contrast")),
        "selected_temporal_stability_score": _as_float(config.get("temporal_stability_score")),
        "selected_switch_rate": _as_float(config.get("switch_rate")),
        "selected_avg_duration": _as_float(config.get("avg_duration")),
        "selected_side_balance_score": _as_float(config.get("side_balance_score")),
        "selected_min_cluster_long_share": _as_float(config.get("min_cluster_long_share")),
        "selected_min_cluster_short_share": _as_float(config.get("min_cluster_short_share")),
        "selected_min_occupancy": _as_float(config.get("min_occupancy")),
        "selected_max_occupancy": _as_float(config.get("max_occupancy")),
        "selected_occupancy_balance_score": _as_float(config.get("occupancy_balance_score")),
        "selected_validation_log_likelihood": _as_float(config.get("validation_log_likelihood")),
        "selected_converged": bool(config.get("converged", False)),
    }


def _hpo_rows(model_name: str, layer: str, state: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, report in enumerate(state.get("hpo_reports", []) or [], start=1):
        if not isinstance(report, dict):
            continue
        rows.append(
            {
                "model_name": model_name,
                "layer": layer,
                "rank": rank,
                "n_components": _as_int(report.get("n_components")),
                "reg_covar": _as_float(report.get("reg_covar")),
                "smooth_lambda": _as_float(report.get("smooth_lambda")),
                "final_score": _as_float(report.get("final_score")),
                "economic_regime_separation": _as_float(report.get("economic_regime_separation")),
                "target_signature_score": _as_float(report.get("target_signature_score")),
                "target_signature_stability": _as_float(report.get("target_signature_stability")),
                "target_signature_contrast": _as_float(report.get("target_signature_contrast")),
                "temporal_stability_score": _as_float(report.get("temporal_stability_score")),
                "side_balance_score": _as_float(report.get("side_balance_score")),
                "min_occupancy": _as_float(report.get("min_occupancy")),
                "max_occupancy": _as_float(report.get("max_occupancy")),
                "occupancy_balance_score": _as_float(report.get("occupancy_balance_score")),
                "validation_log_likelihood": _as_float(report.get("validation_log_likelihood")),
                "occupancy_ok": bool(report.get("occupancy_ok", False)),
                "side_coverage_ok": bool(report.get("side_coverage_ok", False)),
                "converged": bool(report.get("converged", False)),
                "error": str(report.get("error", "")),
            }
        )
    return rows


def _side_rows(model_name: str, layer: str, state: dict[str, Any]) -> list[dict[str, Any]]:
    config = dict(state.get("selected_config", {}) or {})
    rows: list[dict[str, Any]] = []
    for item in config.get("cluster_side_counts", []) or []:
        if not isinstance(item, dict):
            continue
        out = dict(item)
        out["model_name"] = model_name
        out["layer"] = layer
        rows.append(out)
    return rows


def _signature_rows(model_name: str, layer: str, state: dict[str, Any]) -> list[dict[str, Any]]:
    config = dict(state.get("selected_config", {}) or {})
    rows: list[dict[str, Any]] = []
    for item in config.get("cluster_target_signatures", []) or []:
        if not isinstance(item, dict):
            continue
        out = dict(item)
        out["model_name"] = model_name
        out["layer"] = layer
        rows.append(out)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--layer", default="base", choices=["base", "meta", "all"])
    args = parser.parse_args()

    run_root = Path(args.data_root) / "artifacts" / args.run_id
    layers = ["base", "meta"] if args.layer == "all" else [args.layer]
    model_rows: list[dict[str, Any]] = []
    hpo_rows: list[dict[str, Any]] = []
    side_rows: list[dict[str, Any]] = []
    signature_rows: list[dict[str, Any]] = []
    for layer in layers:
        pattern = run_root / "lgbm_reference" / layer
        for sidecar_path in pattern.glob("**/final_model_checkpoint/checkpoint_sidecar.pkl"):
            sidecar = _load_sidecar(sidecar_path)
            if not sidecar:
                continue
            state = dict(sidecar.get("ae_gmm_state", {}) or {})
            model_name = _model_name_from_sidecar(sidecar_path, run_root)
            model_rows.append(
                _selected_config_rows(
                    model_name=model_name,
                    layer=layer,
                    sidecar_path=sidecar_path,
                    state=state,
                )
            )
            hpo_rows.extend(_hpo_rows(model_name, layer, state))
            side_rows.extend(_side_rows(model_name, layer, state))
            signature_rows.extend(_signature_rows(model_name, layer, state))
    out_dir = run_root / "diagnostics" / "ae_gmm_archetypes"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(model_rows).to_csv(out_dir / "ae_gmm_model_summary.csv", index=False)
    pd.DataFrame(hpo_rows).to_csv(out_dir / "ae_gmm_hpo_configs.csv", index=False)
    pd.DataFrame(side_rows).to_csv(out_dir / "ae_gmm_cluster_side_counts.csv", index=False)
    pd.DataFrame(signature_rows).to_csv(out_dir / "ae_gmm_target_signatures.csv", index=False)
    summary = {
        "run_id": args.run_id,
        "layers": layers,
        "model_count": len(model_rows),
        "enabled_model_count": sum(1 for row in model_rows if row.get("enabled")),
        "hpo_config_rows": len(hpo_rows),
        "cluster_side_rows": len(side_rows),
        "target_signature_rows": len(signature_rows),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote {out_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
