#!/usr/bin/env python3
"""Materialize side/archetype interaction context features for train_meta.

This is the concrete handoff after the direct-context interaction smoke.  It
adds live-predictable side/archetype identity and AE/GMM/cross-asset context
interactions to the train_meta input table while keeping interaction evidence
as manifest/report metadata only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_direct_cross_asset_train_meta_feature_set import (  # noqa: E402
    KEY_COLUMNS,
    OUTCOME_COLUMNS,
    SAFE_METADATA_COLUMNS,
)
from scripts.run_direct_context_interaction_meta_smoke import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_INTERACTION_SMOKE_DIR,
    _json_safe,
    _load_feature_columns,
    _variant_frame_and_features,
)
from scripts.run_direct_context_risk_aware_train_meta_smoke import (  # noqa: E402
    DEFAULT_FEATURE_MANIFEST,
    DEFAULT_FEATURE_SET_DIR,
    DEFAULT_HANDOFF,
)


DEFAULT_OUT_DIR = DEFAULT_FEATURE_SET_DIR / "train_meta_interaction_context_feature_set_v1"
DEFAULT_SELECTOR = "s12_ev_clean_strong_risk"
DEFAULT_TOP_FRAC = 0.10


def _downcast_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
        elif pd.api.types.is_integer_dtype(out[col]) and col not in {"month"}:
            out[col] = pd.to_numeric(out[col], errors="coerce", downcast="integer")
    return out


def _risk_cross_features(interaction_features: list[str], one_hot_features: list[str]) -> list[str]:
    one_hot_suffixes = set(one_hot_features)
    out = []
    for col in interaction_features:
        if not col.startswith("intx_xctx_ev_score_oof__"):
            continue
        suffix = col.split("__", 1)[1] if "__" in col else ""
        if suffix not in one_hot_suffixes:
            out.append(col)
    return out


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _variant_evidence(
    *,
    interaction_smoke_dir: Path,
    selector: str,
    top_frac: float,
) -> dict[str, Any]:
    aggregate = _read_csv_if_exists(interaction_smoke_dir / "interaction_train_meta_aggregate.csv")
    aggregate_delta = _read_csv_if_exists(interaction_smoke_dir / "interaction_train_meta_aggregate_delta.csv")
    cell_summary = _read_csv_if_exists(interaction_smoke_dir / "interaction_train_meta_cell_delta_summary.csv")

    evidence_rows: list[dict[str, Any]] = []
    if not aggregate.empty:
        top = aggregate[
            aggregate["selector"].astype(str).eq(selector)
            & np.isclose(pd.to_numeric(aggregate["top_frac"], errors="coerce"), float(top_frac))
        ].copy()
        if not top.empty:
            grouped = top.groupby("variant", as_index=False).agg(
                months=("month", "nunique"),
                precision_positive_ev=("precision_positive_ev", "mean"),
                ev_weighted_precision=("ev_weighted_precision", "mean"),
                mean_ev_after_1pct=("mean_ev_after_1pct", "mean"),
                full_sl_rate=("full_sl_rate", "mean"),
                timeout_rate=("timeout_rate", "mean"),
                clean_exec_proxy_rate=("clean_exec_proxy_rate", "mean"),
            )
            evidence_rows.extend(grouped.to_dict("records"))

    delta_rows: list[dict[str, Any]] = []
    if not aggregate_delta.empty:
        top_delta = aggregate_delta[
            aggregate_delta["selector"].astype(str).eq(selector)
            & np.isclose(pd.to_numeric(aggregate_delta["top_frac"], errors="coerce"), float(top_frac))
        ].copy()
        if not top_delta.empty:
            grouped_delta = top_delta.groupby("variant", as_index=False).agg(
                mean_delta_ev=("delta_mean_ev_after_1pct", "mean"),
                mean_delta_precision=("delta_precision_positive_ev", "mean"),
                mean_delta_full_sl=("delta_full_sl_rate", "mean"),
                mean_delta_timeout=("delta_timeout_rate", "mean"),
                mean_delta_clean=("delta_clean_exec_proxy_rate", "mean"),
            )
            delta_rows.extend(grouped_delta.to_dict("records"))

    cell_rows: list[dict[str, Any]] = []
    if not cell_summary.empty:
        selected = cell_summary[cell_summary["selector"].astype(str).eq(selector)].copy()
        if not selected.empty:
            cell_rows.extend(selected.to_dict("records"))

    return {
        "interaction_smoke_dir": str(interaction_smoke_dir),
        "selector": selector,
        "top_frac": float(top_frac),
        "aggregate": evidence_rows,
        "aggregate_delta": delta_rows,
        "cell_delta_summary": cell_rows,
        "note": "Evidence metadata only; not joined as model input columns.",
    }


def _feature_group_summary(groups: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for name, cols in groups.items():
        rows.append({"feature_group": name, "feature_count": int(len(cols))})
    return pd.DataFrame(rows)


def _feature_availability(frame: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not features or "month" not in frame.columns:
        empty_month = pd.DataFrame(columns=["month", "feature", "non_null_share"])
        empty_summary = pd.DataFrame(columns=["feature", "min_monthly_non_null_share", "mean_monthly_non_null_share"])
        return empty_month, empty_summary
    months = frame["month"].astype(str)
    availability = frame[features].notna().groupby(months, sort=True).mean()
    by_month = availability.reset_index(names="month").melt(
        id_vars="month",
        var_name="feature",
        value_name="non_null_share",
    )
    summary = by_month.groupby("feature", as_index=False).agg(
        min_monthly_non_null_share=("non_null_share", "min"),
        mean_monthly_non_null_share=("non_null_share", "mean"),
        fully_missing_months=("non_null_share", lambda s: int((pd.to_numeric(s, errors="coerce") <= 0.0).sum())),
        usable_months=("non_null_share", lambda s: int((pd.to_numeric(s, errors="coerce") > 0.0).sum())),
    )
    return by_month, summary.sort_values(
        ["fully_missing_months", "mean_monthly_non_null_share", "feature"],
        ascending=[False, True, True],
    )


def _write_report(
    path: Path,
    *,
    manifest: dict[str, Any],
    feature_groups: dict[str, list[str]],
    evidence: dict[str, Any],
    availability_summary: pd.DataFrame,
) -> None:
    aggregate = pd.DataFrame(evidence.get("aggregate", []))
    aggregate_delta = pd.DataFrame(evidence.get("aggregate_delta", []))
    cell_summary = pd.DataFrame(evidence.get("cell_delta_summary", []))
    sparse = availability_summary.head(25) if not availability_summary.empty else pd.DataFrame()
    lines = [
        "# Train Meta Interaction Context Feature Set",
        "",
        "## Status",
        "",
        f"- Rows: `{manifest['rows']}`",
        f"- Feature columns: `{manifest['feature_count']}`",
        f"- Base direct-context features: `{len(feature_groups.get('base_direct_context', []))}`",
        f"- Side/archetype identity features: `{len(feature_groups.get('side_archetype_identity', []))}`",
        f"- Context cell interaction features: `{len(feature_groups.get('context_cell_interactions', []))}`",
        f"- Risk cross interaction features: `{len(feature_groups.get('risk_cross_interactions', []))}`",
        "- Evidence from the interaction smoke is stored only in manifests/reports.",
        "- No accepted-cell flags, future outcomes, or stability-prior features are model inputs.",
        "",
        "## Top10 Interaction Evidence",
        "",
        aggregate.to_markdown(index=False) if not aggregate.empty else "Interaction evidence not found.",
        "",
        "## Top10 Delta vs EV-Only",
        "",
        aggregate_delta.to_markdown(index=False) if not aggregate_delta.empty else "Delta evidence not found.",
        "",
        "## Cell Delta Coverage",
        "",
        cell_summary.to_markdown(index=False) if not cell_summary.empty else "Cell evidence not found.",
        "",
        "## Sparsest Feature Availability",
        "",
        sparse.to_markdown(index=False) if not sparse.empty else "Availability summary not generated.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    *,
    handoff_path: Path,
    feature_manifest_path: Path,
    interaction_smoke_dir: Path,
    output_dir: Path,
    include_risk_interactions: bool,
    selector: str = DEFAULT_SELECTOR,
    top_frac: float = DEFAULT_TOP_FRAC,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(handoff_path)
    base_features = [col for col in _load_feature_columns(feature_manifest_path) if col in frame.columns]
    variant = "i3_context_risk_cell_interactions" if include_risk_interactions else "i2_xctx_cell_interactions"
    interaction_frame, all_features, metadata = _variant_frame_and_features(frame, base_features, variant=variant)

    one_hot_features = [col for col in metadata.get("one_hot_features", []) if col in interaction_frame.columns]
    interaction_features = [col for col in metadata.get("interaction_features", []) if col in interaction_frame.columns]
    risk_features = _risk_cross_features(interaction_features, one_hot_features)
    context_cell_features = [col for col in interaction_features if col not in set(risk_features)]
    feature_groups = {
        "base_direct_context": [col for col in base_features if col in interaction_frame.columns],
        "side_archetype_identity": one_hot_features,
        "context_cell_interactions": context_cell_features,
        "risk_cross_interactions": risk_features,
    }
    if not include_risk_interactions:
        feature_groups["risk_cross_interactions"] = []
    selected_features = []
    seen: set[str] = set()
    for group in (
        "base_direct_context",
        "side_archetype_identity",
        "context_cell_interactions",
        "risk_cross_interactions",
    ):
        for col in feature_groups[group]:
            if col in interaction_frame.columns and col not in seen:
                seen.add(col)
                selected_features.append(col)

    keep_cols: list[str] = []
    for col in list(KEY_COLUMNS) + list(SAFE_METADATA_COLUMNS) + list(OUTCOME_COLUMNS) + selected_features:
        if col in interaction_frame.columns and col not in keep_cols:
            keep_cols.append(col)
    out = _downcast_frame(interaction_frame[keep_cols])

    evidence = _variant_evidence(
        interaction_smoke_dir=interaction_smoke_dir,
        selector=selector,
        top_frac=top_frac,
    )
    availability_by_month, availability_summary = _feature_availability(out, selected_features)
    outputs = {
        "handoff": output_dir / "train_meta_interaction_context_handoff.parquet",
        "feature_manifest": output_dir / "train_meta_interaction_context_feature_manifest.json",
        "feature_group_summary": output_dir / "train_meta_interaction_context_feature_groups.csv",
        "feature_availability_by_month": output_dir / "train_meta_interaction_context_feature_availability_by_month.csv",
        "feature_availability_summary": output_dir / "train_meta_interaction_context_feature_availability_summary.csv",
        "interaction_evidence": output_dir / "train_meta_interaction_context_evidence.json",
        "report": output_dir / "train_meta_interaction_context_feature_set.md",
        "manifest": output_dir / "manifest.json",
    }
    out.to_parquet(outputs["handoff"], index=False)
    _feature_group_summary(feature_groups).to_csv(outputs["feature_group_summary"], index=False)
    availability_by_month.to_csv(outputs["feature_availability_by_month"], index=False)
    availability_summary.to_csv(outputs["feature_availability_summary"], index=False)
    outputs["interaction_evidence"].write_text(json.dumps(_json_safe(evidence), indent=2), encoding="utf-8")
    feature_manifest = {
        "scope": "train_meta_interaction_context_feature_set",
        "source_handoff_path": str(handoff_path),
        "source_feature_manifest_path": str(feature_manifest_path),
        "interaction_smoke_dir": str(interaction_smoke_dir),
        "feature_columns": selected_features,
        "feature_count": len(selected_features),
        "feature_groups": feature_groups,
        "feature_group_counts": {key: int(len(value)) for key, value in feature_groups.items()},
        "materialized_variant": variant,
        "include_risk_interactions": bool(include_risk_interactions),
        "outcome_columns": [col for col in OUTCOME_COLUMNS if col in out.columns],
        "key_columns": [col for col in KEY_COLUMNS if col in out.columns],
        "recommended_usage": {
            "primary_context": "base_direct_context + side_archetype_identity + context_cell_interactions",
            "risk_cross_interactions": "included as train_meta context when enabled; evaluate by side x archetype before promotion",
        },
        "no_leakage_contract": {
            "feature_columns": (
                "live-predictable direct context, side/archetype identity, and products of live-predictable "
                "context with live-predictable side/archetype indicators"
            ),
            "interaction_evidence": "report/manifest metadata only; not joined as model input flags",
            "future_outcomes": "kept only in outcome columns for train/eval, never listed in feature_columns",
            "stability_features": "excluded",
        },
    }
    outputs["feature_manifest"].write_text(json.dumps(_json_safe(feature_manifest), indent=2), encoding="utf-8")
    manifest = {
        "scope": "train_meta_interaction_context_feature_set",
        "output_dir": str(output_dir),
        "rows": int(len(out)),
        "columns": int(len(out.columns)),
        "feature_count": int(len(selected_features)),
        "feature_group_counts": feature_manifest["feature_group_counts"],
        "fully_missing_feature_month_pairs": int(
            (pd.to_numeric(availability_by_month.get("non_null_share", pd.Series(dtype=float)), errors="coerce") <= 0.0).sum()
        ),
        "materialized_variant": variant,
        "include_risk_interactions": bool(include_risk_interactions),
        "outputs": {key: str(value) for key, value in outputs.items()},
        "no_leakage_contract": feature_manifest["no_leakage_contract"],
    }
    outputs["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    _write_report(
        outputs["report"],
        manifest=manifest,
        feature_groups=feature_groups,
        evidence=evidence,
        availability_summary=availability_summary,
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-path", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--feature-manifest-path", type=Path, default=DEFAULT_FEATURE_MANIFEST)
    parser.add_argument("--interaction-smoke-dir", type=Path, default=DEFAULT_INTERACTION_SMOKE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--exclude-risk-interactions", action="store_true")
    parser.add_argument("--selector", default=DEFAULT_SELECTOR)
    parser.add_argument("--top-frac", type=float, default=DEFAULT_TOP_FRAC)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run(
        handoff_path=args.handoff_path,
        feature_manifest_path=args.feature_manifest_path,
        interaction_smoke_dir=args.interaction_smoke_dir,
        output_dir=args.output_dir,
        include_risk_interactions=not bool(args.exclude_risk_interactions),
        selector=str(args.selector),
        top_frac=float(args.top_frac),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
