#!/usr/bin/env python3
"""Audit AE/GMM archetype feature coverage in ablation candidate ledgers."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


AE_GMM_TOKENS = (
    "gmm",
    "cluster",
    "archetype",
    "posterior",
    "mahalanobis",
    "reconstruction",
    "latent",
)
HARD_CLUSTER_TOKENS = ("cluster_id", "cluster_t")
SOFT_PROB_TOKENS = ("gmm_prob_", "posterior_", "cluster_posterior_")
DISTANCE_TOKENS = ("dist_center", "mahal", "density", "nll", "likelihood")
TRANSITION_TOKENS = ("delta_", "accel", "speed", "time_since", "stability", "flip_count")


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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size <= 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_columns(path: Path) -> list[str]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    if path.suffix.lower() in {".parquet", ".pq"}:
        return list(pd.read_parquet(path, columns=[]).columns)
    return list(pd.read_csv(path, nrows=0).columns)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _is_ae_gmm(name: str) -> bool:
    lower = name.lower()
    return any(token in lower for token in AE_GMM_TOKENS)


def _is_hard_cluster_id(name: str) -> bool:
    lower = name.lower()
    return any(token in lower for token in HARD_CLUSTER_TOKENS)


def _count_matching(columns: list[str], predicate) -> int:
    return int(sum(1 for col in columns if predicate(str(col))))


def _bucket_counts(columns: list[str]) -> dict[str, Any]:
    ctx_cols = [str(col) for col in columns if str(col).startswith("ctx_")]
    ctx_long = [col for col in ctx_cols if col.startswith("ctx_long_")]
    ctx_short = [col for col in ctx_cols if col.startswith("ctx_short_")]
    ctx_global_ae = [
        col
        for col in ctx_cols
        if not col.startswith(("ctx_long_", "ctx_short_")) and _is_ae_gmm(col)
    ]
    ctx_market = [
        col
        for col in ctx_cols
        if not col.startswith(("ctx_long_", "ctx_short_")) and not _is_ae_gmm(col)
    ]
    hard_ids = [col for col in columns if _is_hard_cluster_id(str(col))]
    soft_prob = [
        col
        for col in columns
        if any(token in str(col).lower() for token in SOFT_PROB_TOKENS)
    ]
    distance = [
        col
        for col in columns
        if any(token in str(col).lower() for token in DISTANCE_TOKENS)
    ]
    transition = [
        col
        for col in columns
        if any(token in str(col).lower() for token in TRANSITION_TOKENS)
    ]
    entropy = [col for col in columns if "entropy" in str(col).lower()]
    reconstruction = [col for col in columns if "reconstruction" in str(col).lower()]
    return {
        "n_candidate_ledger_columns": int(len(columns)),
        "n_ctx_total": int(len(ctx_cols)),
        "n_ctx_market_state": int(len(ctx_market)),
        "n_ctx_global_ae_gmm": int(len(ctx_global_ae)),
        "n_ctx_long_ae_gmm": int(len(ctx_long)),
        "n_ctx_short_ae_gmm": int(len(ctx_short)),
        "n_cluster_id_features": int(len(hard_ids)),
        "n_soft_prob_features": int(len(soft_prob)),
        "n_distance_features": int(len(distance)),
        "n_transition_features": int(len(transition)),
        "n_entropy_features": int(len(entropy)),
        "n_reconstruction_features": int(len(reconstruction)),
        "hard_cluster_columns": ",".join(hard_ids[:20]),
        "ctx_long_examples": ",".join(ctx_long[:10]),
        "ctx_short_examples": ",".join(ctx_short[:10]),
    }


def _best_row(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or "mean_u" not in frame.columns:
        return {}
    work = frame.copy()
    for col in ("mean_u", "worst_month_mean_u", "bad_mae_1r_rate", "timeout_rate", "final_oracle_recall"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    sort_cols = [
        col
        for col in ("mean_u", "worst_month_mean_u", "final_oracle_recall", "bad_mae_1r_rate", "timeout_rate")
        if col in work.columns
    ]
    ascending = [
        False if col in {"mean_u", "worst_month_mean_u", "final_oracle_recall"} else True
        for col in sort_cols
    ]
    return work.sort_values(sort_cols, ascending=ascending, na_position="last").iloc[0].to_dict()


def _metric(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, float("nan")))
    except Exception:
        return float("nan")


def _effect_counts(effect_path: Path) -> dict[str, Any]:
    effect = _read_csv(effect_path)
    if effect.empty or "archetype_feature" not in effect.columns:
        return {
            "effect_rows": int(len(effect)),
            "effect_feature_count": 0,
            "effect_ctx_long_rows": 0,
            "effect_ctx_short_rows": 0,
        }
    feature = effect["archetype_feature"].astype(str)
    return {
        "effect_rows": int(len(effect)),
        "effect_feature_count": int(feature.nunique(dropna=True)),
        "effect_ctx_long_rows": int(feature.str.startswith("ctx_long_").sum()),
        "effect_ctx_short_rows": int(feature.str.startswith("ctx_short_").sum()),
    }


def build_schema_audit(manifest_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = _read_json(manifest_path)
    rows: list[dict[str, Any]] = []
    for arm in manifest.get("arms", []):
        arm_name = str(arm.get("arm", ""))
        output_dir = Path(str(arm.get("output_dir", "")))
        candidate_path = output_dir / "label_feature_store_model_smoke_candidate_ledger.csv"
        columns = _read_columns(candidate_path)
        base_best = _best_row(_read_csv(output_dir / "label_feature_store_model_smoke_aggregate.csv"))
        meta_best = _best_row(_read_csv(output_dir / "meta_path_filter_smoke" / "gmm_train_meta_path_filter_smoke_aggregate.csv"))
        effect_manifest = _read_json(output_dir / "side_archetype_effect_matrix" / "side_archetype_effect_matrix_manifest.json")
        row = {
            "arm": arm_name,
            "feature_policy": arm.get("feature_policy"),
            "side_context_mode": arm.get("side_context_mode", "off"),
            "include_cluster_id": bool(arm.get("include_cluster_id", False)),
            "candidate_ledger_path": str(candidate_path) if candidate_path.exists() else None,
            **_bucket_counts(columns),
            "effect_side_domain": ",".join(effect_manifest.get("side_domain", []) or []),
            "effect_grouping": ",".join(effect_manifest.get("grouping", []) or []),
            **_effect_counts(output_dir / "side_archetype_effect_matrix" / "side_archetype_effect_matrix.csv"),
            "base_mean_u": _metric(base_best, "mean_u"),
            "base_worst_month_mean_u": _metric(base_best, "worst_month_mean_u"),
            "base_bad_mae_1r_rate": _metric(base_best, "bad_mae_1r_rate"),
            "base_timeout_rate": _metric(base_best, "timeout_rate"),
            "base_final_oracle_recall": _metric(base_best, "final_oracle_recall"),
            "meta_mean_u": _metric(meta_best, "mean_u"),
            "meta_worst_month_mean_u": _metric(meta_best, "worst_month_mean_u"),
            "meta_bad_mae_1r_rate": _metric(meta_best, "bad_mae_1r_rate"),
            "meta_timeout_rate": _metric(meta_best, "timeout_rate"),
            "meta_final_oracle_recall": _metric(meta_best, "final_oracle_recall"),
        }
        should_have_side_ctx = str(row["side_context_mode"]) == "long_short"
        should_exclude_hard_id = not bool(row["include_cluster_id"])
        has_candidate_ledger = candidate_path.exists()
        expected_market_min = 20 if has_candidate_ledger else 0
        row["schema_pass"] = bool(
            has_candidate_ledger
            and row["n_ctx_market_state"] >= expected_market_min
            and (not should_have_side_ctx or (row["n_ctx_long_ae_gmm"] > 0 and row["n_ctx_short_ae_gmm"] > 0))
            and (not should_exclude_hard_id or row["n_cluster_id_features"] == 0)
            and row["effect_side_domain"] == "long,short"
            and row["effect_grouping"] == "side,rank_band,archetype_feature"
            and (not should_have_side_ctx or (row["effect_ctx_long_rows"] > 0 and row["effect_ctx_short_rows"] > 0))
        )
        failures: list[str] = []
        if not candidate_path.exists():
            failures.append("missing_candidate_ledger")
        if row["n_ctx_market_state"] < expected_market_min:
            failures.append("low_ctx_market_state")
        if should_have_side_ctx and row["n_ctx_long_ae_gmm"] <= 0:
            failures.append("missing_ctx_long_ae_gmm")
        if should_have_side_ctx and row["n_ctx_short_ae_gmm"] <= 0:
            failures.append("missing_ctx_short_ae_gmm")
        if should_exclude_hard_id and row["n_cluster_id_features"] > 0:
            failures.append("hard_cluster_id_present")
        if row["effect_side_domain"] != "long,short":
            failures.append("bad_effect_side_domain")
        if row["effect_grouping"] != "side,rank_band,archetype_feature":
            failures.append("bad_effect_grouping")
        if should_have_side_ctx and row["effect_ctx_long_rows"] <= 0:
            failures.append("effect_missing_ctx_long")
        if should_have_side_ctx and row["effect_ctx_short_rows"] <= 0:
            failures.append("effect_missing_ctx_short")
        row["schema_failures"] = ",".join(failures)
        rows.append(row)
    out = pd.DataFrame(rows)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "rows": int(len(out)),
        "schema_pass_arms": int(out["schema_pass"].sum()) if "schema_pass" in out.columns else 0,
        "schema_failed_arms": int((~out["schema_pass"]).sum()) if "schema_pass" in out.columns else 0,
    }
    return out, payload


def write_outputs(frame: pd.DataFrame, payload: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ae_gmm_schema_audit.csv"
    json_path = output_dir / "ae_gmm_schema_audit.json"
    md_path = output_dir / "ae_gmm_schema_audit.md"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(_json_safe({"manifest": payload, "rows": frame.to_dict("records")}), indent=2), encoding="utf-8")
    cols = [
        "arm",
        "feature_policy",
        "side_context_mode",
        "schema_pass",
        "schema_failures",
        "n_ctx_market_state",
        "n_ctx_global_ae_gmm",
        "n_ctx_long_ae_gmm",
        "n_ctx_short_ae_gmm",
        "n_cluster_id_features",
        "n_soft_prob_features",
        "n_distance_features",
        "n_transition_features",
        "effect_feature_count",
        "effect_ctx_long_rows",
        "effect_ctx_short_rows",
        "meta_mean_u",
        "meta_bad_mae_1r_rate",
        "meta_timeout_rate",
        "meta_final_oracle_recall",
    ]
    present = [col for col in cols if col in frame.columns]
    lines = ["# AE/GMM Schema Audit", "", f"- Source manifest: `{payload['manifest_path']}`", ""]
    if present:
        lines.append(frame[present].to_markdown(index=False))
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"csv": str(csv_path), "json": str(json_path), "markdown": str(md_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame, payload = build_schema_audit(args.manifest)
    outputs = write_outputs(frame, payload, args.output_dir)
    print(json.dumps(_json_safe({**payload, "outputs": outputs}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
