#!/usr/bin/env python3
"""Build a compact promotion decision pack for AE/GMM archetype validation."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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


def _read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists() or path.stat().st_size <= 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists() or path.stat().st_size <= 0:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _metric(row: pd.Series | dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, float("nan")))
    except Exception:
        return float("nan")


def _status_from_row(row: pd.Series, *, baseline: pd.Series | None, legacy: pd.Series | None) -> tuple[str, str]:
    failures: list[str] = []
    warnings: list[str] = []
    contract = str(row.get("crossfit_status", "") or "")
    hard_ids = _metric(row, "hard_cluster_feature_count")
    meta_mean = _metric(row, "meta_mean_u")
    meta_worst = _metric(row, "meta_worst_month_mean_u")
    meta_bad = _metric(row, "meta_bad_mae_1r_rate")
    meta_timeout = _metric(row, "meta_timeout_rate")
    effect_rows = _metric(row, "effect_rows")
    effect_stable = _metric(row, "stable_effect_rows")

    if contract == "fail":
        failures.append("crossfit_contract_fail")
    elif contract == "warn":
        warnings.append("crossfit_contract_warn")
    if hard_ids > 0:
        warnings.append("hard_cluster_ids_present")
    if not math.isfinite(meta_mean) or meta_mean <= 0.0:
        failures.append("meta_mean_u_not_positive")
    if not math.isfinite(meta_worst) or meta_worst <= 0.0:
        failures.append("meta_worst_month_not_positive")
    if math.isfinite(meta_bad) and meta_bad > 0.50:
        warnings.append("meta_bad_mae_above_final_bar")
    if math.isfinite(meta_timeout) and meta_timeout > 0.12:
        warnings.append("meta_timeout_above_final_bar")
    if effect_rows <= 0 and str(row.get("arm")) != "G0":
        warnings.append("no_effect_rows")
    if effect_rows > 0 and effect_stable <= 0:
        warnings.append("no_stable_effect_rows")

    if baseline is not None and str(row.get("arm")) != "G0":
        if _metric(row, "meta_mean_u") <= _metric(baseline, "meta_mean_u"):
            warnings.append("does_not_beat_g0_meta_mean")
        if _metric(row, "meta_worst_month_mean_u") < _metric(baseline, "meta_worst_month_mean_u"):
            warnings.append("worse_than_g0_meta_worst")
    if legacy is not None and str(row.get("arm")) not in {"G0", "G1"}:
        if _metric(row, "meta_mean_u") <= _metric(legacy, "meta_mean_u"):
            warnings.append("does_not_beat_g1_meta_mean")
        if _metric(row, "meta_bad_mae_1r_rate") > _metric(legacy, "meta_bad_mae_1r_rate"):
            warnings.append("worse_than_g1_meta_bad_mae")

    if failures:
        return "reject_or_rework", ",".join(failures + warnings)
    if str(row.get("arm")) == "G0":
        return "baseline_only", ",".join(warnings)
    if str(row.get("arm")) in {"G1", "G2", "G3"} and hard_ids > 0:
        return "benchmark_only", ",".join(warnings)
    if warnings:
        return "diagnostic_only", ",".join(warnings)
    return "candidate_soft_layer", ""


def _effect_summary(effect_path: Path | None) -> dict[str, Any]:
    frame = _read_csv(effect_path)
    if frame.empty:
        return {
            "effect_rows": 0,
            "stable_effect_rows": 0,
            "size_lift_rows": 0,
            "penalty_shadow_rows": 0,
            "best_residual_utility": float("nan"),
            "rank_band_count": 0,
        }
    stable = frame[
        pd.to_numeric(frame.get("month_stability", 0.0), errors="coerce").fillna(0.0).ge(0.60)
        & pd.to_numeric(frame.get("fold_stability", 0.0), errors="coerce").fillna(0.0).ge(0.60)
    ]
    actions = frame.get("action", pd.Series("", index=frame.index)).astype(str)
    residual = pd.to_numeric(frame.get("residual_utility_mean", pd.Series(dtype=float)), errors="coerce")
    return {
        "effect_rows": int(len(frame)),
        "stable_effect_rows": int(len(stable)),
        "size_lift_rows": int(actions.str.contains("size_lift", na=False).sum()),
        "penalty_shadow_rows": int(actions.str.contains("penalty", na=False).sum()),
        "best_residual_utility": float(residual.max(skipna=True)) if not residual.dropna().empty else float("nan"),
        "rank_band_count": int(frame.get("rank_band", pd.Series(dtype=str)).nunique(dropna=True)),
    }


def _contract_summary_from_arm(arm: dict[str, Any]) -> dict[str, Any]:
    embedded = arm.get("crossfit_contract_audit") or {}
    output_path_raw = ((embedded.get("outputs") or {}).get("json") if isinstance(embedded, dict) else None)
    if output_path_raw:
        current = _read_json(Path(output_path_raw))
        if current:
            return dict(current.get("summary") or {})
    arm_dir = arm.get("output_dir")
    if arm_dir:
        current = _read_json(Path(str(arm_dir)) / "crossfit_contract_audit" / "ae_gmm_crossfit_contract.json")
        if current:
            return dict(current.get("summary") or {})
    return dict(embedded.get("summary") or {}) if isinstance(embedded, dict) else {}


def build_pack(
    *,
    manifest_path: Path,
    comparison_path: Path | None,
    schema_path: Path | None,
    output_dir: Path,
    effect_override: dict[str, Path] | None = None,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    comparison = _read_csv(
        comparison_path
        or manifest_path.parent / "comparison" / "ae_gmm_archetype_ablation_comparison.csv"
    )
    schema = _read_csv(schema_path or manifest_path.parent / "schema_audit" / "ae_gmm_schema_audit.csv")
    schema_by_arm = {str(row["arm"]): row for _idx, row in schema.iterrows()} if not schema.empty and "arm" in schema.columns else {}
    rows: list[dict[str, Any]] = []
    for arm in manifest.get("arms", []):
        arm_name = str(arm.get("arm"))
        comp_row = comparison[comparison["arm"].astype(str).eq(arm_name)].iloc[0] if not comparison.empty and "arm" in comparison.columns and comparison["arm"].astype(str).eq(arm_name).any() else pd.Series(dtype=object)
        contract_summary = _contract_summary_from_arm(arm)
        schema_row = schema_by_arm.get(arm_name, pd.Series(dtype=object))
        effect_path = None
        if effect_override and arm_name in effect_override:
            effect_path = effect_override[arm_name]
        else:
            effect_manifest = arm.get("side_effect_matrix") or {}
            effect_path_raw = (effect_manifest.get("outputs") or {}).get("effect_matrix")
            effect_path = Path(effect_path_raw) if effect_path_raw else None
        eff = _effect_summary(effect_path)
        row = {
            "arm": arm_name,
            "feature_policy": arm.get("feature_policy"),
            "include_cluster_id": bool(arm.get("include_cluster_id", False)),
            "crossfit_status": contract_summary.get("overall_status"),
            "hard_cluster_feature_count": contract_summary.get("hard_cluster_feature_count", schema_row.get("n_cluster_id_features", float("nan"))),
            "availability_feature_count": contract_summary.get("availability_feature_count", float("nan")),
            "ledger_has_global_oof_available": contract_summary.get("ledger_has_global_oof_available"),
            "ledger_has_long_oof_available": contract_summary.get("ledger_has_long_oof_available"),
            "ledger_has_short_oof_available": contract_summary.get("ledger_has_short_oof_available"),
            "base_mean_u": comp_row.get("base_mean_u", float("nan")),
            "base_worst_month_mean_u": comp_row.get("base_worst_month_mean_u", float("nan")),
            "base_bad_mae_1r_rate": comp_row.get("base_bad_mae_1r_rate", float("nan")),
            "base_timeout_rate": comp_row.get("base_timeout_rate", float("nan")),
            "base_final_oracle_recall": comp_row.get("base_final_oracle_recall", float("nan")),
            "meta_mean_u": comp_row.get("meta_mean_u", float("nan")),
            "meta_worst_month_mean_u": comp_row.get("meta_worst_month_mean_u", float("nan")),
            "meta_bad_mae_1r_rate": comp_row.get("meta_bad_mae_1r_rate", float("nan")),
            "meta_timeout_rate": comp_row.get("meta_timeout_rate", float("nan")),
            "meta_final_oracle_recall": comp_row.get("meta_final_oracle_recall", float("nan")),
            "schema_pass": schema_row.get("schema_pass", np.nan),
            **eff,
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    baseline = out[out["arm"].eq("G0")].iloc[0] if "G0" in set(out.get("arm", [])) else None
    legacy = out[out["arm"].eq("G1")].iloc[0] if "G1" in set(out.get("arm", [])) else None
    decisions = [_status_from_row(row, baseline=baseline, legacy=legacy) for _idx, row in out.iterrows()]
    out["promotion_status"] = [d[0] for d in decisions]
    out["promotion_reasons"] = [d[1] for d in decisions]

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ae_gmm_promotion_decision_pack.csv"
    json_path = output_dir / "ae_gmm_promotion_decision_pack.json"
    md_path = output_dir / "ae_gmm_promotion_decision_pack.md"
    out.to_csv(csv_path, index=False)
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "rows": int(len(out)),
        "status_counts": out["promotion_status"].value_counts(dropna=False).to_dict() if "promotion_status" in out.columns else {},
        "outputs": {"csv": str(csv_path), "json": str(json_path), "markdown": str(md_path)},
    }
    json_path.write_text(json.dumps(_json_safe(result), indent=2), encoding="utf-8")
    cols = [
        "arm",
        "promotion_status",
        "promotion_reasons",
        "crossfit_status",
        "hard_cluster_feature_count",
        "availability_feature_count",
        "meta_mean_u",
        "meta_worst_month_mean_u",
        "meta_bad_mae_1r_rate",
        "meta_timeout_rate",
        "meta_final_oracle_recall",
        "stable_effect_rows",
        "rank_band_count",
    ]
    md_path.write_text(
        "\n".join(
            [
                "# AE/GMM Promotion Decision Pack",
                "",
                f"- Manifest: `{manifest_path}`",
                "",
                out[[col for col in cols if col in out.columns]].to_markdown(index=False),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, default=None)
    parser.add_argument("--schema", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--effect-override",
        action="append",
        default=[],
        help="Optional ARM=path override for a regenerated effect matrix.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    overrides: dict[str, Path] = {}
    for item in args.effect_override:
        if "=" not in str(item):
            raise ValueError("--effect-override must be ARM=path")
        arm, path = str(item).split("=", 1)
        overrides[arm.strip()] = Path(path.strip())
    result = build_pack(
        manifest_path=args.manifest,
        comparison_path=args.comparison,
        schema_path=args.schema,
        output_dir=args.output_dir,
        effect_override=overrides,
    )
    print(json.dumps(_json_safe(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
