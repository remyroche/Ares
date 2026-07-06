#!/usr/bin/env python3
"""Summarize AE/GMM archetype ablation arms across base, meta, and effect layers."""

from __future__ import annotations

import argparse
import json
import math
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


def _read_csv(path_value: Any) -> pd.DataFrame:
    if path_value is None:
        return pd.DataFrame()
    path = Path(str(path_value))
    if not path.exists() or path.stat().st_size <= 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _best_row(frame: pd.DataFrame, sort_cols: tuple[str, ...]) -> dict[str, Any]:
    if frame.empty:
        return {}
    work = frame.copy()
    for col in sort_cols:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    available = [col for col in sort_cols if col in work.columns]
    if not available:
        return work.iloc[0].to_dict()
    ascending = [False if col in {"mean_u", "worst_month_mean_u", "final_oracle_recall"} else True for col in available]
    return work.sort_values(available, ascending=ascending, na_position="last").iloc[0].to_dict()


def _num(row: dict[str, Any], key: str) -> float:
    try:
        value = float(row.get(key, np.nan))
    except Exception:
        value = float("nan")
    return value


def _metric(row: dict[str, Any], key: str) -> Any:
    return row.get(key, np.nan)


def build_comparison(manifest_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for arm in manifest.get("arms", []):
        arm_name = str(arm.get("arm", ""))
        output_dir = Path(str(arm.get("output_dir", "")))
        summary = dict(arm.get("summary") or {})
        base_aggregate = _read_csv(summary.get("aggregate_path") or output_dir / "label_feature_store_model_smoke_aggregate.csv")
        meta_info = dict(arm.get("meta_path_filter_smoke") or {})
        meta_aggregate = _read_csv(meta_info.get("aggregate_path"))
        effect_info = dict(arm.get("side_effect_matrix") or {})
        base_best = _best_row(
            base_aggregate,
            ("mean_u", "worst_month_mean_u", "bad_mae_1r_rate", "timeout_rate"),
        )
        meta_best = _best_row(
            meta_aggregate,
            ("mean_u", "worst_month_mean_u", "bad_mae_1r_rate", "timeout_rate"),
        )
        rows.append(
            {
                "arm": arm_name,
                "description": arm.get("description"),
                "feature_policy": arm.get("feature_policy"),
                "smoke_feature_policy": arm.get("smoke_feature_policy"),
                "side_context_mode": arm.get("side_context_mode", "off"),
                "include_cluster_id": bool(arm.get("include_cluster_id", False)),
                "path_aware_hpo": bool(arm.get("path_aware_hpo", False)),
                "temporal_concentration_hpo": bool(arm.get("temporal_concentration_hpo", False)),
                "base_rows": int(len(base_aggregate)),
                "base_best_selector": base_best.get("selector_variant"),
                "base_mean_u": _num(base_best, "mean_u"),
                "base_worst_month_mean_u": _num(base_best, "worst_month_mean_u"),
                "base_bad_mae_1r_rate": _num(base_best, "bad_mae_1r_rate"),
                "base_timeout_rate": _num(base_best, "timeout_rate"),
                "base_final_oracle_recall": _num(base_best, "final_oracle_recall"),
                "base_positive_months": _num(base_best, "positive_months"),
                "meta_status": meta_info.get("status"),
                "meta_rows": int(len(meta_aggregate)),
                "meta_best_selector": meta_best.get("selector_variant"),
                "meta_best_variant": meta_best.get("meta_variant"),
                "meta_keep_frac": _metric(meta_best, "keep_frac"),
                "meta_mean_u": _num(meta_best, "mean_u"),
                "meta_worst_month_mean_u": _num(meta_best, "worst_month_mean_u"),
                "meta_bad_mae_1r_rate": _num(meta_best, "bad_mae_1r_rate"),
                "meta_timeout_rate": _num(meta_best, "timeout_rate"),
                "meta_final_oracle_recall": _num(meta_best, "final_oracle_recall"),
                "meta_positive_months": _num(meta_best, "positive_months"),
                "effect_rows": int(effect_info.get("effect_rows", 0) or 0),
                "effect_feature_count": int(effect_info.get("archetype_feature_count", 0) or 0),
                "effect_input_rows": int(effect_info.get("rows", 0) or 0),
            }
        )
    out = pd.DataFrame(rows)
    for baseline_arm in ("G0", "G1"):
        base = out.loc[out["arm"].eq(baseline_arm)]
        if base.empty:
            continue
        base_row = base.iloc[0]
        for col in (
            "base_mean_u",
            "base_worst_month_mean_u",
            "base_bad_mae_1r_rate",
            "base_timeout_rate",
            "base_final_oracle_recall",
            "meta_mean_u",
            "meta_worst_month_mean_u",
            "meta_bad_mae_1r_rate",
            "meta_timeout_rate",
            "meta_final_oracle_recall",
        ):
            if col in out.columns:
                out[f"{col}_delta_vs_{baseline_arm.lower()}"] = pd.to_numeric(
                    out[col],
                    errors="coerce",
                ) - float(base_row.get(col, np.nan))
    payload = {
        "manifest_path": str(manifest_path),
        "arms": [str(v) for v in out.get("arm", pd.Series(dtype=str)).tolist()],
        "rows": int(len(out)),
        "execute": bool(manifest.get("execute", False)),
        "candidate_selectors": manifest.get("candidate_selectors"),
        "run_meta_smoke": bool(manifest.get("run_meta_smoke", False)),
    }
    return out, payload


def write_report(comparison: pd.DataFrame, payload: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ae_gmm_archetype_ablation_comparison.csv"
    json_path = output_dir / "ae_gmm_archetype_ablation_comparison.json"
    md_path = output_dir / "ae_gmm_archetype_ablation_comparison.md"
    comparison.to_csv(csv_path, index=False)
    payload = {**payload, "outputs": {"csv": str(csv_path), "json": str(json_path), "markdown": str(md_path)}}
    json_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    cols = [
        "arm",
        "feature_policy",
        "base_mean_u",
        "base_bad_mae_1r_rate",
        "base_timeout_rate",
        "meta_status",
        "meta_mean_u",
        "meta_bad_mae_1r_rate",
        "meta_timeout_rate",
        "effect_rows",
    ]
    present = [col for col in cols if col in comparison.columns]
    lines = [
        "# AE/GMM Archetype Ablation Comparison",
        "",
        f"- Source manifest: `{payload['manifest_path']}`",
        f"- Candidate selectors: `{payload.get('candidate_selectors')}`",
        f"- Meta smoke: `{payload.get('run_meta_smoke')}`",
        "",
    ]
    if present and not comparison.empty:
        lines.append(comparison[present].to_markdown(index=False))
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    comparison, payload = build_comparison(args.manifest)
    manifest = write_report(comparison, payload, args.output_dir)
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
