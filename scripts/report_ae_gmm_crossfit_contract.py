#!/usr/bin/env python3
"""Audit the leakage-safe contract for path-aware AE/GMM smoke features."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    if pd.isna(value):
        return None
    return value


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_csv_if_available(path: Path | None, **kwargs: Any) -> pd.DataFrame:
    if path is None or not path.exists() or path.stat().st_size <= 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except EmptyDataError:
        return pd.DataFrame()


def _read_ledger_availability(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists() or path.stat().st_size <= 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(
            path,
            usecols=lambda col: "oof_available" in str(col) or str(col) == "side",
        )
    except (EmptyDataError, ValueError):
        return pd.DataFrame()


def _parse_side_report(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [dict(v) for v in value if isinstance(v, dict)]
    if pd.isna(value):
        return []
    try:
        parsed = json.loads(str(value))
    except Exception:
        return []
    return [dict(v) for v in parsed] if isinstance(parsed, list) else []


def _bool_col(frame: pd.DataFrame, column: str) -> bool:
    return column in frame.columns


def _positive_rows(frame: pd.DataFrame, *columns: str) -> int:
    for column in columns:
        if column in frame.columns:
            return int(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).gt(0.5).sum())
    return 0


def build_report(
    *,
    manifest_path: Path,
    diagnostics_path: Path | None,
    candidate_ledger_path: Path | None,
    output_dir: Path,
    min_global_coverage: float,
    min_side_coverage: float,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    ae_cfg = dict(manifest.get("ae_gmm_state_features", {}) or {})
    if diagnostics_path is None:
        diag_candidate = manifest.get("outputs", {}).get("diagnostics")
        diagnostics_path = Path(diag_candidate) if diag_candidate else None
    if candidate_ledger_path is None:
        ledger_candidate = manifest.get("outputs", {}).get("candidate_ledger")
        candidate_ledger_path = Path(ledger_candidate) if ledger_candidate else None
    diagnostics = _read_csv_if_available(diagnostics_path)
    ledger = _read_ledger_availability(candidate_ledger_path)

    feature_names = [str(v) for v in ae_cfg.get("feature_names", [])]
    hard_cluster_features = [name for name in feature_names if name.endswith("gmm_cluster_id") or name.endswith("cluster_t")]
    soft_features = [
        name
        for name in feature_names
        if any(token in name for token in ("gmm_prob_", "posterior", "dist_center", "mahal", "entropy", "speed", "accel"))
    ]
    availability_features = [name for name in feature_names if name.endswith("ae_gmm_oof_available")]

    rows: list[dict[str, Any]] = []
    side_rows: list[dict[str, Any]] = []
    if not diagnostics.empty:
        for _, row in diagnostics.iterrows():
            side_reports = _parse_side_report(row.get("ae_gmm_side_context_report"))
            status = "pass"
            reasons: list[str] = []
            enabled = bool(row.get("ae_gmm_state_features_enabled", False))
            train_scope = str(row.get("ae_gmm_state_train_feature_scope", ""))
            valid_scope = str(row.get("ae_gmm_state_validation_feature_scope", ""))
            crossfit_enabled = bool(row.get("ae_gmm_state_crossfit_enabled", False))
            train_scope_is_oof = train_scope == "inner_chronological_oof"
            train_scope_is_in_sample = train_scope == "outer_train_in_sample"
            coverage = float(pd.to_numeric(pd.Series([row.get("ae_gmm_state_crossfit_coverage")]), errors="coerce").iloc[0])
            if not enabled:
                status = "fail"
                reasons.append("ae_gmm_disabled")
            if not (train_scope_is_oof or train_scope_is_in_sample):
                status = "fail"
                reasons.append(f"train_scope={train_scope}")
            if valid_scope != "frozen_outer_train_artifact":
                status = "fail"
                reasons.append(f"validation_scope={valid_scope}")
            if train_scope_is_oof and not crossfit_enabled:
                status = "fail"
                reasons.append("crossfit_disabled")
            if train_scope_is_oof and math.isfinite(coverage) and coverage < float(min_global_coverage):
                status = "warn" if status == "pass" else status
                reasons.append(f"global_coverage<{min_global_coverage:g}")
            if hard_cluster_features:
                status = "fail"
                reasons.append("hard_cluster_feature_present")
            if not soft_features:
                status = "fail"
                reasons.append("no_soft_archetype_features")
            if not availability_features:
                status = "warn" if status == "pass" else status
                reasons.append("missing_oof_availability_feature")
            rows.append(
                {
                    "period": row.get("period"),
                    "status": status,
                    "reasons": ",".join(reasons),
                    "train_scope": train_scope,
                    "validation_scope": valid_scope,
                    "train_contract": "in_sample_outer_train" if train_scope_is_in_sample else "inner_oof_crossfit",
                    "crossfit_enabled": crossfit_enabled,
                    "crossfit_split_count": row.get("ae_gmm_state_crossfit_split_count"),
                    "crossfit_fitted_folds": row.get("ae_gmm_state_crossfit_fitted_folds"),
                    "crossfit_coverage": coverage,
                    "path_aware_hpo": row.get("ae_gmm_state_path_aware_hpo"),
                    "path_cleanliness_score": row.get("ae_gmm_state_path_cleanliness_score"),
                    "temporal_concentration_hpo": row.get("ae_gmm_state_temporal_concentration_hpo"),
                    "max_cluster_time_bucket_share": row.get("ae_gmm_state_max_cluster_time_bucket_share"),
                    "feature_count": row.get("ae_gmm_state_feature_count"),
                    "side_context_feature_count": row.get("ae_gmm_side_context_feature_count"),
                }
            )
            for side in side_reports:
                side_status = "pass"
                side_reasons: list[str] = []
                side_coverage = float(side.get("train_crossfit_coverage", float("nan")))
                if side.get("status") != "ok":
                    side_status = "fail"
                    side_reasons.append(str(side.get("status")))
                if train_scope_is_oof and math.isfinite(side_coverage) and side_coverage < float(min_side_coverage):
                    side_status = "warn" if side_status == "pass" else side_status
                    side_reasons.append(f"side_coverage<{min_side_coverage:g}")
                side_rows.append(
                    {
                        "period": row.get("period"),
                        "side": side.get("side"),
                        "status": side_status,
                        "reasons": ",".join(side_reasons),
                        "train_rows": side.get("train_rows"),
                        "valid_rows": side.get("valid_rows"),
                        "feature_count": side.get("feature_count"),
                        "train_crossfit_rows": side.get("train_crossfit_rows"),
                        "train_crossfit_uncovered_rows": side.get("train_crossfit_uncovered_rows"),
                        "train_crossfit_coverage": side_coverage,
                        "train_crossfit_folds": side.get("train_crossfit_folds"),
                        "path_cleanliness_score": side.get("path_cleanliness_score"),
                        "temporal_concentration_score": side.get("temporal_concentration_score"),
                    }
                )

    summary = {
        "enabled": bool(ae_cfg.get("enabled", False)),
        "feature_policy": ae_cfg.get("feature_policy"),
        "side_context_mode": ae_cfg.get("side_context_mode"),
        "train_feature_scope": ae_cfg.get("train_feature_scope"),
        "validation_feature_scope": ae_cfg.get("validation_feature_scope"),
        "train_contract": (
            "in_sample_outer_train"
            if ae_cfg.get("train_feature_scope") == "outer_train_in_sample"
            else "inner_oof_crossfit"
            if ae_cfg.get("train_feature_scope") == "inner_chronological_oof"
            else "unknown"
        ),
        "crossfit_train_features": bool(ae_cfg.get("crossfit_train_features", False)),
        "generated_feature_count": int(ae_cfg.get("generated_feature_count", 0) or 0),
        "hard_cluster_feature_count": int(len(hard_cluster_features)),
        "soft_feature_count": int(len(soft_features)),
        "availability_feature_count": int(len(availability_features)),
        "ledger_has_global_oof_available": _bool_col(ledger, "ctx_ae_gmm_oof_available") or _bool_col(ledger, "ae_gmm_oof_available"),
        "ledger_has_long_oof_available": _bool_col(ledger, "ctx_long_ae_gmm_oof_available") or _bool_col(ledger, "long_ae_gmm_oof_available"),
        "ledger_has_short_oof_available": _bool_col(ledger, "ctx_short_ae_gmm_oof_available") or _bool_col(ledger, "short_ae_gmm_oof_available"),
        "ledger_global_oof_available_positive_rows": _positive_rows(ledger, "ctx_ae_gmm_oof_available", "ae_gmm_oof_available"),
        "ledger_long_oof_available_positive_rows": _positive_rows(ledger, "ctx_long_ae_gmm_oof_available", "long_ae_gmm_oof_available"),
        "ledger_short_oof_available_positive_rows": _positive_rows(ledger, "ctx_short_ae_gmm_oof_available", "short_ae_gmm_oof_available"),
    }
    if not bool(ae_cfg.get("enabled", False)):
        period_df = pd.DataFrame(rows)
        side_df = pd.DataFrame(side_rows)
        overall_status = "not_applicable"
        summary["overall_status"] = overall_status
        output_dir.mkdir(parents=True, exist_ok=True)
        period_path = output_dir / "ae_gmm_crossfit_contract_periods.csv"
        side_path = output_dir / "ae_gmm_crossfit_contract_sides.csv"
        json_path = output_dir / "ae_gmm_crossfit_contract.json"
        md_path = output_dir / "ae_gmm_crossfit_contract.md"
        period_df.to_csv(period_path, index=False)
        side_df.to_csv(side_path, index=False)
        result = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "manifest_path": str(manifest_path),
            "diagnostics_path": str(diagnostics_path) if diagnostics_path else None,
            "candidate_ledger_path": str(candidate_ledger_path) if candidate_ledger_path else None,
            "summary": summary,
            "outputs": {"periods_csv": str(period_path), "sides_csv": str(side_path), "json": str(json_path), "markdown": str(md_path)},
        }
        json_path.write_text(json.dumps(_json_safe(result), indent=2), encoding="utf-8")
        md_path.write_text(
            "\n".join(
                [
                    "# AE/GMM Crossfit Contract",
                    "",
                    f"- Manifest: `{manifest_path}`",
                    "- Overall status: `not_applicable`",
                    "- Reason: AE/GMM state features are disabled for this arm.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return result

    period_df = pd.DataFrame(rows)
    side_df = pd.DataFrame(side_rows)
    if period_df.empty:
        overall_status = "fail"
    elif (period_df["status"] == "fail").any() or (not side_df.empty and (side_df["status"] == "fail").any()):
        overall_status = "fail"
    elif (period_df["status"] == "warn").any() or (not side_df.empty and (side_df["status"] == "warn").any()):
        overall_status = "warn"
    else:
        overall_status = "pass"
    summary["overall_status"] = overall_status

    output_dir.mkdir(parents=True, exist_ok=True)
    period_path = output_dir / "ae_gmm_crossfit_contract_periods.csv"
    side_path = output_dir / "ae_gmm_crossfit_contract_sides.csv"
    json_path = output_dir / "ae_gmm_crossfit_contract.json"
    md_path = output_dir / "ae_gmm_crossfit_contract.md"
    period_df.to_csv(period_path, index=False)
    side_df.to_csv(side_path, index=False)
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "diagnostics_path": str(diagnostics_path) if diagnostics_path else None,
        "candidate_ledger_path": str(candidate_ledger_path) if candidate_ledger_path else None,
        "summary": summary,
        "outputs": {"periods_csv": str(period_path), "sides_csv": str(side_path), "json": str(json_path), "markdown": str(md_path)},
    }
    json_path.write_text(json.dumps(_json_safe(result), indent=2), encoding="utf-8")
    lines = [
        "# AE/GMM Crossfit Contract",
        "",
        f"- Manifest: `{manifest_path}`",
        f"- Overall status: `{overall_status}`",
        f"- Hard cluster features: `{len(hard_cluster_features)}`",
        f"- Soft features: `{len(soft_features)}`",
        f"- Availability features: `{len(availability_features)}`",
        "",
    ]
    if not period_df.empty:
        lines.append("## Periods")
        lines.append(period_df.to_markdown(index=False))
        lines.append("")
    if not side_df.empty:
        lines.append("## Sides")
        lines.append(side_df.to_markdown(index=False))
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, default=None)
    parser.add_argument("--candidate-ledger", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-global-coverage", type=float, default=0.60)
    parser.add_argument("--min-side-coverage", type=float, default=0.50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_report(
        manifest_path=args.manifest,
        diagnostics_path=args.diagnostics,
        candidate_ledger_path=args.candidate_ledger,
        output_dir=args.output_dir,
        min_global_coverage=float(args.min_global_coverage),
        min_side_coverage=float(args.min_side_coverage),
    )
    print(json.dumps(_json_safe(result), indent=2))
    return 0 if result["summary"]["overall_status"] in {"pass", "warn", "not_applicable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
