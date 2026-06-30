#!/usr/bin/env python3
"""Materialize deployable native reliability component models only.

This is the fast path after the research OOF reliability blend has already
selected component features, rank references, and blend coefficients.  It fits
only full-fit live-contract component models for each head and reuses prior OOF
component score distributions for deployable rank references.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from scripts import run_reliability_blend_optuna as rb


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict("records")


def _prefer_new_period_soft(records: list[dict[str, Any]], fallback: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one deployable B3/new-period soft-qfail config per head when available."""
    by_head: dict[str, dict[str, Any]] = {}
    for row in records:
        if not isinstance(row, dict) or not row.get("head"):
            continue
        if str(row.get("variant")) == rb.BLEND_NEW_SOFT:
            by_head.setdefault(str(row["head"]), row)
    for row in fallback:
        if isinstance(row, dict) and row.get("head"):
            by_head.setdefault(str(row["head"]), row)
    return [by_head[h] for h in sorted(by_head)]


def _rank_references_from_scores(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"component score reference file does not exist: {path}")
    df = pd.read_parquet(path)
    required = {"head", "anchor_score", "period_new_score", "qfail_soft_score"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise RuntimeError(f"component score reference file missing required columns: {missing}")
    out: dict[str, dict[str, Any]] = {}
    for head, group in df.groupby(df["head"].astype(str), sort=True):
        refs: dict[str, Any] = {}
        for col in (
            "anchor_score",
            "period_old_score",
            "period_new_score",
            "qfail_hard_score",
            "qfail_soft_score",
        ):
            if col in group.columns:
                refs[col] = rb._score_reference_payload(
                    pd.to_numeric(group[col], errors="coerce").to_numpy(dtype="float32")
                )
        out[str(head)] = refs
    return out


def _bundle_manifest(bundle: dict[str, Any], *, bundle_path: Path, output_dir: Path, diag_path: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": bundle["schema_version"],
        "status": bundle["status"],
        "native_component_scoring": bundle["native_component_scoring"],
        "distilled_student_status": bundle["distilled_student_status"],
        "component_model_bundle_path": str(bundle_path),
        "component_diagnostics_path": str(diag_path),
        "heads": {},
    }
    for head, head_bundle in dict(bundle.get("heads", {}) or {}).items():
        model_rows = [rb._component_model_summary(a) for a in list(head_bundle.get("models", []) or [])]
        full_fit_rows = [
            row
            for row in model_rows
            if str(row.get("fold", "")).lower() == "full_fit"
            or str(row.get("model_scope", "")).lower() == "full_fit"
        ]
        manifest["heads"][str(head)] = {
            "score_columns": dict(head_bundle.get("score_columns", {}) or {}),
            "model_count": int(len(model_rows)),
            "full_fit_model_count": int(len(full_fit_rows)),
            "full_fit_components": sorted(
                str(row.get("component"))
                for row in full_fit_rows
                if row.get("component") is not None
            ),
            "models": model_rows,
            "component_rank_references": {
                str(name): rb._score_reference_summary(payload)
                for name, payload in dict(head_bundle.get("component_rank_references", {}) or {}).items()
                if isinstance(payload, dict)
            },
        }
    return manifest


def _write_outputs(
    *,
    output_dir: Path,
    component_dir: Path,
    bundle: dict[str, Any],
    feature_manifest: dict[str, Any],
    diag_rows: list[dict[str, Any]],
) -> Path:
    diag_path = output_dir / "native_fullfit_component_diagnostics.csv"
    pd.DataFrame(diag_rows).to_csv(diag_path, index=False)
    bundle_path = component_dir / "reliability_blend_native_component_models.joblib"
    joblib.dump(bundle, bundle_path, compress=3)
    manifest = _bundle_manifest(bundle, bundle_path=bundle_path, output_dir=output_dir, diag_path=diag_path)
    (component_dir / "reliability_blend_native_component_model_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=rb._json_default) + "\n"
    )
    (output_dir / "reliability_blend_feature_target_manifest.json").write_text(
        json.dumps(feature_manifest, indent=2, default=rb._json_default) + "\n"
    )
    return bundle_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-artifact-dir", default="data_perp/artifacts/meta_featureselect_recentguard_20260622_0119")
    parser.add_argument("--baseline-artifact-dir", default="data_perp/artifacts/20260617_090000_no_mkt4_labelhpo_final_fit")
    parser.add_argument("--report-dir", default="data_perp/reports/meta_featureselect_recentguard_20260622_0119")
    parser.add_argument("--feature-dir", default="data_perp/features/20260605_070000")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--only-head", nargs="*", default=list(rb.HEADS))
    parser.add_argument("--transform-cache", default="")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--reference-component-scores", type=Path, default=Path("data_perp/reports/reliability_blend_optuna_20260623_native_lgbm_only_50k/reliability_blend_component_scores.parquet"))
    parser.add_argument("--blend-winners", type=Path, default=Path("data_perp/reports/reliability_blend_optuna_20260623_native_lgbm_only_50k/reliability_blend_optuna_winners.csv"))
    parser.add_argument("--default-soft-qfail-config", type=Path, default=Path("data_perp/reports/reliability_blend_optuna_20260623_native_lgbm_only_50k/reliability_blend_soft_qfail_default_by_head.csv"))
    parser.add_argument("--aux-native-lgbm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--aux-native-hpo-trials", type=int, default=12)
    parser.add_argument("--aux-native-hpo-patience", type=int, default=4)
    parser.add_argument("--aux-native-max-depth", type=int, default=5)
    parser.add_argument("--aux-native-min-child-pct-min", type=float, default=0.02)
    parser.add_argument("--aux-native-min-child-pct-max", type=float, default=0.07)
    parser.add_argument("--aux-native-reuse-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--aux-native-reuse-feature-source",
        default=(
            "data_perp/reports/reliability_blend_optuna_fullfit_smoke_v2_20260624,"
            "data_perp/reports/reliability_blend_optuna_20260623_native_lgbm_only_50k"
        ),
    )
    parser.add_argument("--aux-native-reuse-min-features", type=int, default=8)
    parser.add_argument("--aux-native-reuse-min-fraction", type=float, default=0.25)
    parser.add_argument("--max-control-path-features", type=int, default=64)
    parser.add_argument("--qfail-meta-output-max-features", type=int, default=80)
    parser.add_argument("--qfail-meta-output-max-derived-features", type=int, default=48)
    parser.add_argument("--qfail-interaction-max-context-features", type=int, default=24)
    parser.add_argument("--qfail-soft-rank-threshold", type=float, default=0.50)
    parser.add_argument("--qfail-soft-min-train-rows", type=int, default=500)
    parser.add_argument("--qfail-soft-max-depth", type=int, default=3)
    parser.add_argument("--qfail-soft-n-estimators", type=int, default=180)
    parser.add_argument("--qfail-soft-min-child-fraction", type=float, default=0.025)
    parser.add_argument("--max-timestamp-features", type=int, default=220)
    parser.add_argument("--period-soft-rank-threshold", type=float, default=0.50)
    parser.add_argument("--period-soft-horizon-hours", type=int, default=24)
    parser.add_argument("--period-soft-halflife-hours", type=float, default=12.0)
    parser.add_argument("--period-soft-inner-folds", type=int, default=3)
    parser.add_argument("--period-soft-min-train-timestamps", type=int, default=200)
    parser.add_argument("--period-soft-tail-ramp-share", type=float, default=0.20)
    parser.add_argument("--period-soft-tail-ramp-power", type=float, default=1.5)
    parser.add_argument("--period-soft-badness-base-weight", type=float, default=1.0)
    parser.add_argument("--period-soft-selection-target", choices=["tail_severity", "soft_percentile"], default="tail_severity")
    parser.add_argument("--period-soft-hpo-trials", type=int, default=24)
    parser.add_argument("--period-soft-hpo-max-features", type=int, default=120)
    parser.add_argument("--period-soft-min-child-fraction", type=float, default=0.08)
    parser.add_argument("--period-soft-max-depth", type=int, default=3)
    parser.add_argument("--period-soft-n-estimators", type=int, default=220)
    parser.add_argument("--seed", type=int, default=37)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = rb._ensure_dir(args.output_dir)
    component_dir = rb._ensure_dir(output_dir / "reliability_blend_component_models")
    reuse_cache = rb._load_native_aux_feature_reuse_cache(args.aux_native_reuse_feature_source)
    setattr(args, "_native_aux_feature_reuse_cache", reuse_cache)
    heads, meta_models, base_bundle, symbol_columns = rb._load_heads(args)
    transform_cache = Path(args.transform_cache) if str(args.transform_cache).strip() else None
    rank_refs = _rank_references_from_scores(args.reference_component_scores)
    blend_winners = _load_records(args.blend_winners)
    soft_defaults = _load_records(args.default_soft_qfail_config)
    deployable_defaults = _prefer_new_period_soft(blend_winners, soft_defaults)

    bundle: dict[str, Any] = {
        "schema_version": "reliability_blend_native_component_models_v1",
        "status": "full_fit_live_contract_only",
        "native_component_scoring": "deployable full_fit live_contract_safe_v1 component models with reused OOF rank references",
        "distilled_student_status": "audit_fallback_only",
        "full_fit_component_models_enabled": True,
        "full_fit_live_feature_contract": "live_contract_safe_v1",
        "heads": {},
        "blend_winners": blend_winners,
        "default_soft_qfail_config_by_head": deployable_defaults,
        "default_deployable_config_by_head": deployable_defaults,
        "params": rb._public_arg_dict(args),
    }
    feature_manifest: dict[str, Any] = {
        "generated_by": "materialize_native_reliability_component_bundle_fullfit",
        "period_new_target": "future 24h EWMA of timestamp mean abs(anchor_score - y_bin) for rows with anchor_rank>=0.5, percentile-normalized on each training fold",
        "period_new_tail_labels": "fold-local labels period_bad_05/10/15 and period_tail_severity are derived from period_new_target for period-learner weighting and HPO diagnostics only",
        "period_new_hpo_objective": "0.45*APLift@5 + 0.25*APLift@10 + 0.15*Recall@10 + 0.10*NDCG@10 + 0.05*difficulty_decile_spread",
        "period_new_sample_weights": "asymmetric smooth ramp: base weight increases only for above-median difficulty; selected bad-share boost is applied gradually over the configured tail ramp",
        "qfail_soft_target": "(1-y_bin) * anchor_score inside anchor top50 rank>=0.50, no timestamp smoothing",
        "native_aux_feature_reuse": {
            "enabled": bool(args.aux_native_reuse_features),
            "source": str(args.aux_native_reuse_feature_source),
            "cache_entries": int(len(reuse_cache)),
            "min_features": int(args.aux_native_reuse_min_features),
            "min_fraction": float(args.aux_native_reuse_min_fraction),
        },
        "reference_component_scores": str(args.reference_component_scores),
        "blend_winners": str(args.blend_winners),
        "default_soft_qfail_config": str(args.default_soft_qfail_config),
        "features": {},
    }
    diag_rows: list[dict[str, Any]] = []
    for head in heads:
        print(f"[native_fullfit_bundle] head={head.head}", flush=True)
        panel = rb._downcast_numeric(rb._normalise_keys(pd.read_parquet(head.meta_oof_path)), exclude=["timestamp", "symbol"])
        panel = panel.sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)
        if int(args.max_rows) > 0 and len(panel) > int(args.max_rows):
            keep = pd.Index(range(len(panel))).to_numpy()
            keep = keep[:: max(1, int(len(panel) / int(args.max_rows)))]
            panel = panel.iloc[keep[: int(args.max_rows)]].sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)
        race = meta_models[head.meta_key]
        _current_x, raw = rb.ctx._assemble_head_context(
            head=head,
            panel=panel,
            race=race,
            base_bundle=base_bundle,
            feature_dir=Path(args.feature_dir),
            transform_cache=transform_cache,
            symbol_columns=symbol_columns,
            regime_context=None,
            max_regime_columns=0,
        )
        y = rb.ctx._meta_target(panel)
        anchor_score = rb.ctx._current_meta_score(panel)
        rank0 = rb.fixed._rank0(panel, anchor_score)
        bundle["heads"].setdefault(
            head.head,
            {
                "models": [],
                "score_columns": {
                    "anchor_score": "anchor_score",
                    "period_new": "period_new_score",
                    "qfail_soft": "qfail_soft_score",
                    "qfail_hard": "qfail_hard_score",
                },
                "component_rank_references": dict(rank_refs.get(head.head, {})),
            },
        )
        feature_manifest["features"].setdefault(head.head, {})
        artifacts, diag = rb._fit_full_fit_live_contract_components(
            head=head.head,
            panel=panel,
            raw=raw,
            y=y,
            anchor_score=anchor_score,
            rank0=rank0,
            args=args,
            feature_manifest=feature_manifest,
        )
        bundle["heads"][head.head]["models"].extend(artifacts)
        diag_rows.append({"head": head.head, "rows": int(len(panel)), "full_fit_model_count": int(len(artifacts)), **diag})
        print(f"[native_fullfit_bundle] head={head.head} full_fit_models={len(artifacts)}", flush=True)
        _write_outputs(
            output_dir=output_dir,
            component_dir=component_dir,
            bundle=bundle,
            feature_manifest=feature_manifest,
            diag_rows=diag_rows,
        )

    bundle_path = _write_outputs(
        output_dir=output_dir,
        component_dir=component_dir,
        bundle=bundle,
        feature_manifest=feature_manifest,
        diag_rows=diag_rows,
    )
    print(json.dumps({"bundle_path": str(bundle_path), "heads": sorted(bundle["heads"].keys())}, indent=2))


if __name__ == "__main__":
    main()
