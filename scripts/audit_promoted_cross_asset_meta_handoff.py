#!/usr/bin/env python3
"""Audit promoted cross-asset representation features in the S52 meta handoff.

This audit compares a baseline train-meta smoke with a promoted-feature smoke.
It answers a narrow handoff question:

* were the promoted features materialized under the OOF/prior-fold contract?
* were they actually consumed by the meta smoke?
* did the promoted smoke improve the top-k selector metrics that matter?
* is anything still blocking policy/frozen replay promotion?
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(
    "data_perp/reports/s52_trailing_profit_best_pointwise_scored_ledger_20260705_v1/"
    "s52_trailing_regime_meta_handoff_xmarket_v1"
)
DEFAULT_BASELINE_SMOKE_DIR = DEFAULT_ROOT / "train_meta_smoke_baseline_for_promoted_compare_v2"
DEFAULT_PROMOTED_HANDOFF_DIR = DEFAULT_ROOT / "train_meta_handoff_promoted_cross_asset_v1"
DEFAULT_PROMOTED_SMOKE_DIR = DEFAULT_PROMOTED_HANDOFF_DIR / "train_meta_smoke_v2"
DEFAULT_PROMOTION_JSON = (
    DEFAULT_ROOT
    / "cross_asset_representation_meta_ablation_v2_conditional_control_v2"
    / "cross_asset_representation_meta_ablation_v2_promotion.json"
)
DEFAULT_OUT_DIR = DEFAULT_PROMOTED_HANDOFF_DIR / "promoted_cross_asset_meta_handoff_audit_v1"

CONTRACT_NAME = "train_meta_regime_handoff_contract.json"
MANIFEST_NAME = "manifest.json"
FEATURE_IMPORTANCE_NAME = "s52_train_meta_regime_handoff_smoke_feature_importance.csv"
FORBIDDEN_FEATURES = {
    "exec_margin",
    "ev_after_1pct",
    "ret_net",
    "u_policy_net",
    "clean_exec",
    "dirty_positive",
    "full_path_bad_mae_1r",
    "timeout",
    "long_bad_path_label",
    "long_path_clean_exec_label",
    "long_path_dirty_positive_label",
    "long_path_quality_soft",
}
CORE_SELECTOR_METRICS = (
    "mean_keep010_exec_margin",
    "mean_keep010_clean_exec_precision",
    "mean_keep010_full_path_bad_mae",
    "mean_keep010_timeout",
    "mean_keep010_oracle_recall",
    "mean_keep030_exec_margin",
    "mean_keep030_clean_exec_precision",
    "mean_keep030_full_path_bad_mae",
    "mean_keep030_timeout",
    "mean_keep030_oracle_recall",
    "mean_auc_clean_exec",
    "mean_ap_clean_exec",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _metric(payload: dict[str, Any], metric: str) -> float:
    value = (payload.get("best_selector") or {}).get(metric)
    try:
        return float(value)
    except Exception:
        return float("nan")


def _delta(promoted: float, baseline: float) -> float:
    if not math.isfinite(promoted) or not math.isfinite(baseline):
        return float("nan")
    return promoted - baseline


def _threshold_status(manifest: dict[str, Any]) -> str:
    return str((manifest.get("best_threshold_policy") or {}).get("threshold_policy_status") or "missing")


def _feature_importance(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["test_month", "model", "feature", "importance"])
    return pd.read_csv(path)


def _feature_checks(
    *,
    baseline_fi: pd.DataFrame,
    promoted_fi: pd.DataFrame,
    promoted_columns: list[str],
) -> dict[str, Any]:
    def used_features(frame: pd.DataFrame) -> set[str]:
        if frame.empty or "feature" not in frame.columns:
            return set()
        imp = pd.to_numeric(frame.get("importance"), errors="coerce").fillna(0.0)
        return set(frame.loc[imp > 0.0, "feature"].astype(str))

    baseline_used = used_features(baseline_fi)
    promoted_used = used_features(promoted_fi)
    leaked = sorted((baseline_used | promoted_used) & FORBIDDEN_FEATURES)
    promoted_used_cols = sorted(col for col in promoted_columns if col in promoted_used)
    cross_used = sorted(feature for feature in promoted_used if feature.startswith("cross_lgbm_"))
    return {
        "forbidden_features_used": leaked,
        "forbidden_features_absent": not leaked,
        "promoted_columns_used": promoted_used_cols,
        "promoted_column_use_count": int(len(promoted_used_cols)),
        "all_promoted_columns_used": set(promoted_columns).issubset(promoted_used),
        "cross_lgbm_features_used": cross_used,
        "cross_lgbm_feature_use_count": int(len(cross_used)),
    }


def _metric_rows(baseline: dict[str, Any], promoted: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in CORE_SELECTOR_METRICS:
        b = _metric(baseline, metric)
        p = _metric(promoted, metric)
        rows.append(
            {
                "metric": metric,
                "baseline": b,
                "promoted": p,
                "delta": _delta(p, b),
            }
        )
    return pd.DataFrame(rows)


def _gate_decision(
    *,
    contract_block: dict[str, Any],
    feature_checks: dict[str, Any],
    metrics: pd.DataFrame,
    promoted_manifest: dict[str, Any],
) -> dict[str, Any]:
    lookup = metrics.set_index("metric")["delta"].to_dict()
    checks = {
        "handoff_no_in_sample_backfill": bool(contract_block.get("no_in_sample_backfill") is True),
        "handoff_has_promoted_rows": float(contract_block.get("coverage_all_promoted_columns") or 0.0) > 0.0,
        "feature_no_forbidden_outcomes": bool(feature_checks.get("forbidden_features_absent")),
        "feature_promoted_columns_consumed": int(feature_checks.get("promoted_column_use_count") or 0) > 0,
        "top10_exec_improves": float(lookup.get("mean_keep010_exec_margin", float("nan"))) > 0.0,
        "top10_clean_precision_improves": float(lookup.get("mean_keep010_clean_exec_precision", float("nan"))) > 0.0,
        "top10_bad_mae_improves": float(lookup.get("mean_keep010_full_path_bad_mae", float("nan"))) < 0.0,
        "top10_oracle_recall_stable": float(lookup.get("mean_keep010_oracle_recall", float("nan"))) >= -0.01,
        "top30_exec_improves": float(lookup.get("mean_keep030_exec_margin", float("nan"))) > 0.0,
        "top30_oracle_recall_improves": float(lookup.get("mean_keep030_oracle_recall", float("nan"))) > 0.0,
        "ap_improves": float(lookup.get("mean_ap_clean_exec", float("nan"))) > 0.0,
        "auc_improves": float(lookup.get("mean_auc_clean_exec", float("nan"))) > 0.0,
    }
    warnings = {
        "top30_bad_mae_nonworse": float(lookup.get("mean_keep030_full_path_bad_mae", float("nan"))) <= 0.0,
        "top30_timeout_nonworse": float(lookup.get("mean_keep030_timeout", float("nan"))) <= 0.0,
        "threshold_policy_passes": _threshold_status(promoted_manifest) == "pass",
    }
    failed_checks = [key for key, value in checks.items() if not value]
    warning_flags = [key for key, value in warnings.items() if not value]
    handoff_status = "pass" if not failed_checks else "fail"
    meta_feature_status = "conditional_pass_for_deeper_meta_eval" if not failed_checks else "blocked"
    policy_status = "blocked" if warning_flags or _threshold_status(promoted_manifest) != "pass" else "candidate"
    return {
        "handoff_status": handoff_status,
        "meta_feature_status": meta_feature_status,
        "simple_policy_status": policy_status,
        "frozen_replay_status": "blocked",
        "failed_checks": failed_checks,
        "warning_flags": warning_flags,
        "checks": checks,
        "warnings": warnings,
        "read": (
            "Promoted cross-asset features are acceptable for deeper meta evaluation"
            if not failed_checks
            else "Promoted cross-asset features are not ready for deeper meta evaluation"
        ),
    }


def _write_markdown(out_dir: Path, payload: dict[str, Any], metrics: pd.DataFrame) -> Path:
    gate = payload["gate_decision"]
    feature = payload["feature_checks"]
    contract = payload["handoff_contract"]
    threshold = payload["threshold_policy_comparison"]
    display = metrics.copy()
    for col in ("baseline", "promoted", "delta"):
        display[col] = display[col].map(lambda x: "nan" if not math.isfinite(float(x)) else f"{float(x):.6f}")
    lines = [
        "# Promoted Cross-Asset Meta Handoff Audit",
        "",
        "## Verdict",
        "",
        f"- meta feature status: `{gate['meta_feature_status']}`",
        f"- simple policy status: `{gate['simple_policy_status']}`",
        f"- frozen replay status: `{gate['frozen_replay_status']}`",
        f"- failed checks: `{', '.join(gate['failed_checks']) or 'none'}`",
        f"- warning flags: `{', '.join(gate['warning_flags']) or 'none'}`",
        "",
        "## Handoff Contract",
        "",
        f"- preferred variant: `{contract.get('preferred_variant')}`",
        f"- promoted columns: `{', '.join(contract.get('promoted_columns') or [])}`",
        f"- rows with promoted columns: `{contract.get('rows_with_all_promoted_columns')}`",
        f"- promoted coverage: `{float(contract.get('coverage_all_promoted_columns') or 0.0):.4%}`",
        f"- no in-sample backfill: `{contract.get('no_in_sample_backfill')}`",
        "",
        "## Feature Consumption",
        "",
        f"- promoted columns used: `{', '.join(feature.get('promoted_columns_used') or [])}`",
        f"- forbidden outcome/path features used: `{', '.join(feature.get('forbidden_features_used') or []) or 'none'}`",
        "",
        "## Selector Metrics",
        "",
        display.to_markdown(index=False),
        "",
        "## Threshold Policy",
        "",
        f"- baseline: `{threshold['baseline_status']}`",
        f"- promoted: `{threshold['promoted_status']}`",
        "- read: threshold templates are still diagnostic unless status is `pass`.",
        "",
        "## Next Gate",
        "",
        "Use these promoted columns in the next deeper train_meta evaluation, but do not advance frozen replay from this artifact.",
    ]
    path = out_dir / "promoted_cross_asset_meta_handoff_audit.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def run_audit(
    *,
    baseline_smoke_dir: Path,
    promoted_smoke_dir: Path,
    promoted_handoff_dir: Path,
    promotion_json: Path,
    out_dir: Path,
) -> dict[str, Any]:
    baseline_manifest = _read_json(baseline_smoke_dir / MANIFEST_NAME)
    promoted_manifest = _read_json(promoted_smoke_dir / MANIFEST_NAME)
    contract = _read_json(promoted_handoff_dir / CONTRACT_NAME)
    promotion = _read_json(promotion_json)
    contract_block = contract.get("promoted_cross_asset_representation") or {}
    promoted_columns = [str(col) for col in contract_block.get("promoted_columns") or []]
    baseline_fi = _feature_importance(baseline_smoke_dir / FEATURE_IMPORTANCE_NAME)
    promoted_fi = _feature_importance(promoted_smoke_dir / FEATURE_IMPORTANCE_NAME)
    feature_checks = _feature_checks(
        baseline_fi=baseline_fi,
        promoted_fi=promoted_fi,
        promoted_columns=promoted_columns,
    )
    metrics = _metric_rows(baseline_manifest, promoted_manifest)
    gate = _gate_decision(
        contract_block=contract_block,
        feature_checks=feature_checks,
        metrics=metrics,
        promoted_manifest=promoted_manifest,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_path = out_dir / "promoted_cross_asset_meta_handoff_metric_deltas.csv"
    metrics.to_csv(metric_path, index=False)
    payload = {
        "generated_by": "audit_promoted_cross_asset_meta_handoff",
        "baseline_smoke_dir": str(baseline_smoke_dir),
        "promoted_smoke_dir": str(promoted_smoke_dir),
        "promoted_handoff_dir": str(promoted_handoff_dir),
        "promotion_json": str(promotion_json),
        "baseline_best_selector": baseline_manifest.get("best_selector"),
        "promoted_best_selector": promoted_manifest.get("best_selector"),
        "threshold_policy_comparison": {
            "baseline_status": _threshold_status(baseline_manifest),
            "promoted_status": _threshold_status(promoted_manifest),
            "baseline": baseline_manifest.get("best_threshold_policy"),
            "promoted": promoted_manifest.get("best_threshold_policy"),
        },
        "handoff_contract": contract_block,
        "promotion_artifact_status": promotion.get("status"),
        "promotion_promoted_variants": [item.get("variant") for item in promotion.get("promote_to_deeper_meta_eval") or []],
        "feature_checks": feature_checks,
        "gate_decision": gate,
        "outputs": {
            "json": str(out_dir / "promoted_cross_asset_meta_handoff_audit.json"),
            "markdown": str(out_dir / "promoted_cross_asset_meta_handoff_audit.md"),
            "metric_deltas": str(metric_path),
        },
    }
    markdown_path = _write_markdown(out_dir, payload, metrics)
    payload["outputs"]["markdown"] = str(markdown_path)
    (out_dir / "promoted_cross_asset_meta_handoff_audit.json").write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True)
    )
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-smoke-dir", type=Path, default=DEFAULT_BASELINE_SMOKE_DIR)
    parser.add_argument("--promoted-smoke-dir", type=Path, default=DEFAULT_PROMOTED_SMOKE_DIR)
    parser.add_argument("--promoted-handoff-dir", type=Path, default=DEFAULT_PROMOTED_HANDOFF_DIR)
    parser.add_argument("--promotion-json", type=Path, default=DEFAULT_PROMOTION_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_audit(
        baseline_smoke_dir=args.baseline_smoke_dir,
        promoted_smoke_dir=args.promoted_smoke_dir,
        promoted_handoff_dir=args.promoted_handoff_dir,
        promotion_json=args.promotion_json,
        out_dir=args.out_dir,
    )
    print(json.dumps(_json_safe({"event": "promoted_cross_asset_meta_handoff_audit_done", **payload}), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
