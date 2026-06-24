"""Consolidate J4/J5 contextual meta freeze decisions across heads.

The J4/J5 capacity script writes one artifact directory per run.  This
consolidator turns those per-run decisions into one explicit freeze manifest:
baseline artifact, selected contextual arm, selected capacity/distillation
configuration, directional thresholds, and current fresh-OOS status.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_INPUT_DIRS = (
    Path("data_perp/reports/j4_contextual_meta_capacity_ablation_short_asset_full_no_j5_20260623"),
    Path("data_perp/reports/j4_j5_contextual_meta_capacity_ablation_long_bars_full_20260623"),
    Path("data_perp/reports/j4_j5_contextual_meta_capacity_ablation_diagnostic_heads_full_20260623"),
)
EXPECTED_HEADS = ("long_bars", "long_dist", "short_asset", "short_boll")
HEAD_META_OOF_PREFIX = {
    "long_bars": "meta_oof_long_bars",
    "long_dist": "meta_oof_long_dist",
    "short_asset": "meta_oof_short_asset",
    "short_boll": "meta_oof_short_boll",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if not np.isfinite(val) else val
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _timestamp_window(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"development_window_start": "", "development_window_end": "", "timestamp_count": 0}
    df = pd.read_csv(path, usecols=["timestamp"])
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dropna()
    if ts.empty:
        return {"development_window_start": "", "development_window_end": "", "timestamp_count": 0}
    return {
        "development_window_start": ts.min().isoformat(),
        "development_window_end": ts.max().isoformat(),
        "timestamp_count": int(ts.nunique()),
    }


def _timestamp_window_from_frame(path: Path, column: str = "timestamp") -> dict[str, Any]:
    if not path.exists():
        return {"start": "", "end": "", "timestamp_count": 0}
    try:
        df = pd.read_parquet(path, columns=[column])
    except Exception:
        return {"start": "", "end": "", "timestamp_count": 0}
    ts = pd.to_datetime(df[column], utc=True, errors="coerce").dropna()
    if ts.empty:
        return {"start": "", "end": "", "timestamp_count": 0}
    return {"start": ts.min().isoformat(), "end": ts.max().isoformat(), "timestamp_count": int(ts.nunique())}


def _baseline_meta_oof_window(baseline_artifact_dir: Path, head: str) -> dict[str, Any]:
    prefix = HEAD_META_OOF_PREFIX.get(head, f"meta_oof_{head}")
    meta_oof_dir = baseline_artifact_dir / "meta_oof"
    matches = sorted(meta_oof_dir.glob(f"{prefix}*.parquet")) if meta_oof_dir.exists() else []
    if not matches:
        return {"baseline_meta_oof_start": "", "baseline_meta_oof_end": "", "baseline_meta_oof_timestamp_count": 0}
    window = _timestamp_window_from_frame(matches[0], "timestamp")
    return {
        "baseline_meta_oof_start": window["start"],
        "baseline_meta_oof_end": window["end"],
        "baseline_meta_oof_timestamp_count": window["timestamp_count"],
    }


def _max_iso_timestamp(*values: Any) -> str:
    parsed = pd.to_datetime([v for v in values if isinstance(v, str) and v], utc=True, errors="coerce")
    parsed = parsed.dropna()
    if len(parsed) == 0:
        return ""
    return parsed.max().isoformat()


def _normalise_selected_capacity(value: Any, promoted: bool) -> str:
    text = "" if pd.isna(value) else str(value)
    if text and text.lower() != "nan":
        return text
    return "none_retain_context_arm" if not promoted else ""


def _best_config_row(configs: pd.DataFrame, head: str, config_id: Any) -> dict[str, Any]:
    if configs.empty or "head" not in configs.columns or "config_id" not in configs.columns:
        return {}
    config_text = "" if pd.isna(config_id) else str(config_id)
    if not config_text or config_text.lower() == "nan":
        return {}
    rows = configs.loc[configs["head"].astype(str).eq(head) & configs["config_id"].astype(str).eq(config_text)]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def build_manifest(input_dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest_rows: list[dict[str, Any]] = []
    top_config_rows: list[dict[str, Any]] = []
    audit_items: list[dict[str, Any]] = []
    missing: list[str] = []
    input_audit_statuses: dict[str, str] = {}
    j5_without_promoted: list[str] = []

    required = (
        "j4_j5_contextual_meta_freeze_decisions.csv",
        "j4_contextual_meta_config_summary.csv",
        "j4_j5_contextual_meta_feature_arm_freeze.csv",
        "j4_j5_contextual_meta_requirement_audit.json",
        "run_config.json",
    )

    for input_dir in input_dirs:
        for name in required:
            if not (input_dir / name).exists():
                missing.append(str(input_dir / name))
        if missing:
            continue

        freeze = _read_csv(input_dir / "j4_j5_contextual_meta_freeze_decisions.csv")
        configs = _read_csv(input_dir / "j4_contextual_meta_config_summary.csv")
        feature_freeze = _read_csv(input_dir / "j4_j5_contextual_meta_feature_arm_freeze.csv")
        j5 = _read_csv(input_dir / "j5_contextual_meta_distillation_summary.csv")
        run_config = _read_json(input_dir / "run_config.json")
        requirement_audit = _read_json(input_dir / "j4_j5_contextual_meta_requirement_audit.json")
        input_audit_statuses[str(input_dir)] = str(requirement_audit.get("status", "missing"))
        window = _timestamp_window(input_dir / "j4_j5_contextual_meta_directional_timestamp_metrics.csv")

        promoted_configs = configs.loc[configs.get("config_promoted", pd.Series(False, index=configs.index)).astype(bool)]
        if promoted_configs.empty and not j5.empty:
            j5_without_promoted.append(str(input_dir))

        if not configs.empty:
            ranked = configs.sort_values(
                [
                    "head",
                    "config_promoted",
                    "median_episode_delta_hr30",
                    "q25_episode_delta_hr30",
                    "median_delta_hr30",
                    "median_delta_ndcg",
                ],
                ascending=[True, False, False, False, False, False],
            )
            top_config_rows.extend(ranked.groupby("head", group_keys=False).head(3).assign(artifact_dir=str(input_dir)).to_dict(orient="records"))

        feature_by_head = (
            feature_freeze.set_index("head", drop=False)
            if not feature_freeze.empty and "head" in feature_freeze.columns
            else pd.DataFrame()
        )
        for _, row in freeze.iterrows():
            head = str(row["head"])
            baseline_dir = Path(str(run_config.get("baseline_artifact_dir", "")))
            baseline_window = _baseline_meta_oof_window(baseline_dir, head)
            promoted = str(row.get("promotion_status", "")).startswith("development_promoted")
            best = _best_config_row(configs, head, row.get("j4_best_config", ""))
            feature_row = feature_by_head.loc[head].to_dict() if head in feature_by_head.index else {}
            selected_capacity = _normalise_selected_capacity(row.get("selected_capacity_config", ""), promoted)
            manifest_rows.append(
                {
                    "head": head,
                    "artifact_dir": str(input_dir),
                    "baseline_artifact_dir": str(run_config.get("baseline_artifact_dir", "")),
                    "model_contract": "single_existing_meta_head_unchanged_y_bin_binary_logloss_one_probability",
                    "training_objective": "binary_log_loss",
                    "sample_weight_contract": "ordinary_bce_no_top30_reweighting",
                    "selected_contextual_feature_arm": str(row.get("selected_contextual_feature_arm", "")),
                    "selected_feature_arm_source": str(feature_row.get("selection_source", "")),
                    "selected_capacity_config": selected_capacity,
                    "selected_distillation_variant": str(row.get("selected_distillation_variant", "")),
                    "decision": str(row.get("decision", "")),
                    "promotion_status": str(row.get("promotion_status", "")),
                    "fresh_oos_status": str(row.get("fresh_oos_status", "")),
                    "rank_threshold": float(run_config.get("rank_threshold", 0.70)),
                    "top_fraction": float(1.0 - float(run_config.get("rank_threshold", 0.70))),
                    "selection_order": "median_episode_delta_hr30,q25_episode_delta_hr30,timestamp_balanced_delta_hr30,delta_ndcg_top30,net_correct_trades_gained",
                    "hr10_min_delta": -float(run_config.get("directional_hr_tolerance", 0.001)),
                    "hr20_min_delta": -float(run_config.get("directional_hr_tolerance", 0.001)),
                    "normal_period_hr30_min_delta": -float(run_config.get("directional_hr_tolerance", 0.001)),
                    "ndcg30_min_delta": 0.0,
                    "min_seed_pass_rate": float(run_config.get("min_seed_pass_rate", np.nan)),
                    "j4_seeds": ",".join(str(x) for x in run_config.get("j4_seeds", [])),
                    "outer_folds": int(run_config.get("outer_folds", 0) or 0),
                    "max_j4_configs": int(run_config.get("max_j4_configs", 0) or 0),
                    "max_train_rows": int(run_config.get("max_train_rows", 0) or 0),
                    "j4_best_config": str(row.get("j4_best_config", "")),
                    "j4_best_seed_pass_rate": row.get("j4_best_seed_pass_rate", np.nan),
                    "j4_best_config_promoted": bool(best.get("config_promoted", False)) if best else False,
                    "j4_best_median_episode_delta_hr30": row.get("j4_best_median_episode_delta_hr30", np.nan),
                    "j4_best_q25_episode_delta_hr30": best.get("q25_episode_delta_hr30", np.nan),
                    "j4_best_median_delta_hr30": best.get("median_delta_hr30", np.nan),
                    "j4_best_median_delta_ndcg": best.get("median_delta_ndcg", np.nan),
                    "j4_best_median_delta_hr10": best.get("median_delta_hr10", np.nan),
                    "j4_best_median_delta_hr20": best.get("median_delta_hr20", np.nan),
                    "j4_best_median_net_correct": best.get("median_net_correct", np.nan),
                    "j4_best_min_leaf_count": best.get("min_leaf_count_min", np.nan),
                    "j4_best_context_split_share": best.get("context_split_share_mean", np.nan),
                    "j4_best_context_gain_share": best.get("context_gain_share_mean", np.nan),
                    "j5_rows": int(len(j5.loc[j5.get("head", pd.Series(dtype=str)).astype(str).eq(head)])) if not j5.empty else 0,
                    "input_audit_status": str(requirement_audit.get("status", "")),
                    **window,
                    **baseline_window,
                    "effective_fresh_oos_after": _max_iso_timestamp(
                        window.get("development_window_end", ""),
                        baseline_window.get("baseline_meta_oof_end", ""),
                    ),
                }
            )

    manifest = pd.DataFrame(manifest_rows)
    top_configs = pd.DataFrame(top_config_rows)
    heads = set(manifest.get("head", pd.Series(dtype=str)).astype(str))
    expected_heads = set(EXPECTED_HEADS)
    contract_cols = {
        "baseline_artifact_dir",
        "model_contract",
        "training_objective",
        "sample_weight_contract",
        "selected_contextual_feature_arm",
        "selected_capacity_config",
        "selected_distillation_variant",
        "rank_threshold",
        "selection_order",
        "hr10_min_delta",
        "hr20_min_delta",
        "normal_period_hr30_min_delta",
        "ndcg30_min_delta",
        "fresh_oos_status",
    }

    audit_items.append(
        {
            "requirement": "input_artifacts_present",
            "status": "passed" if not missing else "failed",
            "metrics": {"missing": missing},
        }
    )
    audit_items.append(
        {
            "requirement": "input_audits_pass",
            "status": "passed" if input_audit_statuses and all(v == "passed" for v in input_audit_statuses.values()) else "failed",
            "metrics": input_audit_statuses,
        }
    )
    audit_items.append(
        {
            "requirement": "all_expected_heads_present",
            "status": "passed" if heads == expected_heads else "failed",
            "metrics": {"expected": sorted(expected_heads), "found": sorted(heads)},
        }
    )
    audit_items.append(
        {
            "requirement": "freeze_contract_explicit",
            "status": "passed" if contract_cols <= set(manifest.columns) and not manifest.empty else "failed",
            "metrics": {"required_columns": sorted(contract_cols), "rows": int(len(manifest))},
        }
    )
    audit_items.append(
        {
            "requirement": "no_j5_without_promoted_j4",
            "status": "passed" if not j5_without_promoted else "failed",
            "metrics": {"offending_artifact_dirs": j5_without_promoted},
        }
    )
    audit_items.append(
        {
            "requirement": "fresh_oos_not_consumed",
            "status": "passed"
            if not manifest.empty and manifest["fresh_oos_status"].astype(str).eq("pending_later_labelled_interval").all()
            else "failed",
            "metrics": {"statuses": sorted(set(manifest.get("fresh_oos_status", pd.Series(dtype=str)).astype(str)))},
        }
    )

    audit = {"status": "passed" if all(x["status"] == "passed" for x in audit_items) else "failed", "items": audit_items}
    return manifest, top_configs, audit


def _write_report(out_dir: Path, manifest: pd.DataFrame, top_configs: pd.DataFrame, audit: dict[str, Any]) -> None:
    lines = [
        "# J4/J5 All-Head Freeze Manifest",
        "",
        "This manifest consolidates the completed J4 capacity/regularization ablations and records the frozen development decision per head.",
        "Fresh chronological OOS remains pending and is not consumed by these artifacts.",
        "",
        "## Audit",
        "",
        pd.DataFrame(audit.get("items", [])).to_markdown(index=False),
        "",
    ]
    if not manifest.empty:
        cols = [
            "head",
            "selected_contextual_feature_arm",
            "selected_capacity_config",
            "selected_distillation_variant",
            "promotion_status",
            "j4_best_config",
            "j4_best_median_episode_delta_hr30",
            "j4_best_seed_pass_rate",
            "fresh_oos_status",
            "development_window_end",
            "baseline_meta_oof_end",
            "effective_fresh_oos_after",
        ]
        lines.extend(["## Freeze Decisions", "", manifest[[c for c in cols if c in manifest.columns]].to_markdown(index=False, floatfmt=".6f"), ""])
    if not top_configs.empty:
        cols = [
            "head",
            "config_id",
            "config_promoted",
            "seed_pass_rate",
            "median_episode_delta_hr30",
            "q25_episode_delta_hr30",
            "median_delta_hr30",
            "median_delta_ndcg",
            "median_delta_hr10",
            "median_delta_hr20",
        ]
        lines.extend(["## Top J4 Configs", "", top_configs[[c for c in cols if c in top_configs.columns]].to_markdown(index=False, floatfmt=".6f"), ""])
    (out_dir / "j4_j5_contextual_meta_all_head_freeze_report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_perp/reports/j4_j5_contextual_meta_all_head_freeze_20260623"),
    )
    args = parser.parse_args()

    input_dirs = list(args.input_dir or DEFAULT_INPUT_DIRS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest, top_configs, audit = build_manifest(input_dirs)
    manifest.to_csv(args.output_dir / "j4_j5_contextual_meta_all_head_freeze_manifest.csv", index=False)
    top_configs.to_csv(args.output_dir / "j4_j5_contextual_meta_all_head_top_configs.csv", index=False)
    (args.output_dir / "j4_j5_contextual_meta_all_head_freeze_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, default=_json_default)
    )
    (args.output_dir / "j4_j5_contextual_meta_all_head_freeze_inputs.json").write_text(
        json.dumps({"input_dirs": [str(x) for x in input_dirs]}, indent=2, sort_keys=True)
    )
    _write_report(args.output_dir, manifest, top_configs, audit)
    print(f"[j4_j5_freeze] wrote manifest to {args.output_dir}", flush=True)
    if audit["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
