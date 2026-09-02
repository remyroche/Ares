#!/usr/bin/env python3
"""Provenance-only admissibility audit for the short conditional-payoff repair.

No model is fitted.  The audit accepts only strict March--April exact-policy
rows and explicitly whitelists score/support/context inputs available at the
decision.  Realised path labels and timing/target-price/wait action fields are
never admitted as model features.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data_perp/artifacts/short_conditional_payoff_readiness_20260730_v1"
SCORES = ROOT / "data_perp/artifacts/marapr2025_all_score_ic_ev_waterfall_20260730_v1/all_score_waterfall.parquet"
PEAK = ROOT / "data_perp/artifacts/febapr2025_historical_auxiliary_oof_20260729_v2/oof_predictions.parquet"
SLOPE = ROOT / "data_perp/artifacts/febapr2025_historical_future_slope_fixed_geometry_oof_20260730_v1/oof_predictions.parquet"
MAE = ROOT / "data_perp/artifacts/febapr2025_historical_auxiliary_oof_20260729_v2_timing_mae/oof_predictions.parquet"
PANEL = ROOT / "data_perp/artifacts/canonical_opportunity_payoff_trust_panel_20260729_v2/panel.parquet"
IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
SCORE_TRIPLET = ("score_base_alpha", "score_residual_expected_ev", "direct_q25_return")
PEAK_SUPPORT = ("pred_peak_mfe_12h_atr__p_hit", "pred_peak_mfe_12h_atr__conditional_mean")
SLOPE_SUPPORT = ("pred_future_slope_atr_per_hour__diagnostic",)
MAE_SUPPORT = ("pred_mae_before_meaningful_mfe_atr__p_hit", "pred_mae_before_meaningful_mfe_atr__if_hit", "pred_mae_before_meaningful_mfe_atr__if_no_hit")
COMPACT_CONTEXT = (
    "base_score_z_timestamp_side", "base_margin_to_top40_cutoff_z",
    "range_24h_pct", "__meta_raw__volatility_zscore", "trend_r2_24", "jump_intensity", "__meta_raw__chop_score",
    "preentry_transition__range_24h_pct__delta_3h",
    "preentry_transition__meta_raw__volatility_zscore__delta_3h",
    "preentry_transition__trend_r2_24__delta_3h",
    "preentry_transition__jump_intensity__delta_3h",
    "preentry_transition__meta_raw__chop_score__delta_3h",
    "preentry_transition__regime_source_shock_impulse_score__delta_3h",
    "preentry_transition__regime_source_compression_score__delta_3h",
    "preentry_transition__regime_source_dirty_shock_avoid_score__delta_3h",
)
FORBIDDEN_TOKENS = ("execution_", "__meaningful", "__peak", "time_to", "timing", "target_price", "wait", "mapped_", "opportunity_")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Mapping): return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [safe(v) for v in value]
    if value is pd.NaT or (not isinstance(value, (str, bytes, bool)) and pd.isna(value)): return None
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(safe(dict(value)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def assert_admissible(features: Sequence[str]) -> None:
    forbidden = [feature for feature in features if any(token in feature.lower() for token in FORBIDDEN_TOKENS)]
    if forbidden:
        raise ValueError(f"forbidden future/action/outcome feature(s): {forbidden}")


def expected_mae_support(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["pred_mae_before_meaningful_mfe_atr__p_hit"] * frame["pred_mae_before_meaningful_mfe_atr__if_hit"]
        + (1.0 - frame["pred_mae_before_meaningful_mfe_atr__p_hit"]) * frame["pred_mae_before_meaningful_mfe_atr__if_no_hit"]
    )


def recommended_feature_sets() -> dict[str, list[str]]:
    """Smallest role-specific sets; compact context is a separately gated ablation."""
    result = {
        "short_p_net_positive": list(SCORE_TRIPLET),
        "short_conditional_favorable_magnitude": [*SCORE_TRIPLET, "peak_expected_mfe_atr_oof", *SLOPE_SUPPORT],
        "short_conditional_adverse_severity": [*SCORE_TRIPLET, "mae_expected_before_meaningful_mfe_atr_oof"],
    }
    for features in result.values(): assert_admissible(features)
    return result


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=columns)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    if frame.duplicated(list(IDENTITY)).any(): raise ValueError(f"duplicate four-field identity in {path}")
    return frame


def _merge_exact(left: pd.DataFrame, right: pd.DataFrame, name: str) -> pd.DataFrame:
    joined = left.merge(right, on=list(IDENTITY), how="outer", validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError(f"{name} does not exact-join: {joined['_merge'].value_counts().to_dict()}")
    return joined.drop(columns="_merge")


def _enrich_from_superset(left: pd.DataFrame, right: pd.DataFrame, name: str) -> pd.DataFrame:
    """Attach canonical provenance from a larger panel without changing the strict population."""
    joined = left.merge(right, on=list(IDENTITY), how="left", validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError(f"{name} lacks strict-row provenance: {joined['_merge'].value_counts().to_dict()}")
    return joined.drop(columns="_merge")


def _inventory(frame: pd.DataFrame, fields: Sequence[str], family: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (month, side), local in frame.groupby(["candidate_month", "side_name"], observed=True, sort=True):
        for field in fields:
            values = pd.to_numeric(local[field], errors="coerce")
            records.append({"family": family, "field": field, "candidate_month": month, "side_name": side, "rows": int(len(local)), "finite_rows": int(np.isfinite(values).sum()), "finite_fraction": float(np.isfinite(values).mean()), "admissible_preentry_or_oof_prediction": True})
    return pd.DataFrame(records)


def run(output_dir: Path = OUTPUT) -> dict[str, Any]:
    if output_dir.exists(): raise FileExistsError(f"refusing to overwrite immutable output {output_dir}")
    required = (SCORES, PEAK, SLOPE, MAE, PANEL)
    if not all(path.is_file() for path in required): raise FileNotFoundError("required exact-row source is absent")
    score = _read(SCORES, [*IDENTITY, "execution_net_ev_12h", "execution_label_end_utc", "candidate_month", "residual_fold", *SCORE_TRIPLET])
    peak = _read(PEAK, [*IDENTITY, "__decision_ts__", "__label_end_ts__", *PEAK_SUPPORT])
    slope = _read(SLOPE, [*IDENTITY, "__decision_ts__", "__label_end_ts__", *SLOPE_SUPPORT])
    mae = _read(MAE, [*IDENTITY, "__decision_ts__", "__label_end_ts__", *MAE_SUPPORT])
    panel = _read(PANEL, [*IDENTITY, "__decision_ts__", "execution_label_end_utc", "execution_net_ev_12h", "fold_id", "fold_validation_start_utc", "fold_validation_end_utc", "effective_label_resolution_utc", *COMPACT_CONTEXT])
    frame = _merge_exact(score, peak.drop(columns=["__decision_ts__", "__label_end_ts__"]), "score/peak")
    frame = _merge_exact(frame, slope.drop(columns=["__decision_ts__", "__label_end_ts__"]), "score/slope")
    frame = _merge_exact(frame, mae.drop(columns=["__decision_ts__", "__label_end_ts__"]), "score/mae")
    panel = panel.loc[panel["__ts__"].ge(pd.Timestamp("2025-03-01T00:00:00Z")) & panel["__ts__"].lt(pd.Timestamp("2025-05-01T00:00:00Z"))].copy()
    frame = _enrich_from_superset(frame, panel, "strict score/support/panel")
    if len(frame) != 140_682 or set(frame.side_name) != {"long", "short"}: raise ValueError("not the complete strict March-April population")
    for field in ("execution_label_end_utc_x", "execution_label_end_utc_y", "effective_label_resolution_utc", "fold_validation_start_utc", "fold_validation_end_utc"):
        frame[field] = pd.to_datetime(frame[field], utc=True, errors="raise")
    if not np.allclose(frame["execution_net_ev_12h_x"], frame["execution_net_ev_12h_y"], rtol=0, atol=1e-12): raise ValueError("exact net labels disagree across sources")
    if not frame["execution_label_end_utc_x"].equals(frame["execution_label_end_utc_y"]): raise ValueError("exact label ends disagree across sources")
    frame["peak_expected_mfe_atr_oof"] = frame[PEAK_SUPPORT[0]] * frame[PEAK_SUPPORT[1]]
    frame["mae_expected_before_meaningful_mfe_atr_oof"] = expected_mae_support(frame)
    derived = ("peak_expected_mfe_atr_oof", "mae_expected_before_meaningful_mfe_atr_oof")
    all_features = [*SCORE_TRIPLET, *PEAK_SUPPORT, *SLOPE_SUPPORT, *MAE_SUPPORT, *derived, *COMPACT_CONTEXT]
    inventory = pd.concat([
        _inventory(frame, SCORE_TRIPLET, "score_triplet"), _inventory(frame, [*PEAK_SUPPORT, "peak_expected_mfe_atr_oof"], "peak_oof"),
        _inventory(frame, SLOPE_SUPPORT, "slope_oof"), _inventory(frame, [*MAE_SUPPORT, "mae_expected_before_meaningful_mfe_atr_oof"], "mae_oof"),
        _inventory(frame, COMPACT_CONTEXT, "causal_context"),
    ], ignore_index=True)
    required_features = [*SCORE_TRIPLET, *PEAK_SUPPORT, *SLOPE_SUPPORT, *MAE_SUPPORT, *derived]
    required_inventory = inventory.loc[inventory.field.isin(required_features)]
    if not required_inventory.finite_fraction.eq(1.0).all(): raise ValueError("required score/support feature has missing/nonfinite strict-row values")
    full_context = sorted(inventory.loc[(inventory.family.eq("causal_context")) & inventory.finite_fraction.eq(1.0), "field"].unique())
    partial_context = sorted(inventory.loc[(inventory.family.eq("causal_context")) & inventory.finite_fraction.lt(1.0), "field"].unique())
    fold = frame.groupby(["candidate_month", "side_name", "residual_fold", "fold_id", "fold_validation_start_utc", "fold_validation_end_utc"], observed=True).agg(rows=("candidate_id", "size"), label_end_min=("execution_label_end_utc_x", "min"), label_end_max=("execution_label_end_utc_x", "max"), effective_resolution_min=("effective_label_resolution_utc", "min"), effective_resolution_max=("effective_label_resolution_utc", "max")).reset_index()
    fold["all_labels_resolve_after_decision"] = fold.label_end_min.ge(fold.fold_validation_start_utc)
    forbidden = pd.DataFrame([
        {"field_or_family": "execution_net_ev_12h / positive_net_12h", "reason": "realised exact H12 labels; targets only"},
        {"field_or_family": "__meaningful_mfe_reached_12h__ / __peak_mfe_atr_12h__", "reason": "realised future path labels; predicted OOF supports only"},
        {"field_or_family": "pred_time_to_first_meaningful_mfe*", "reason": "timing action-layer support; excluded from conversion ranker"},
        {"field_or_family": "target-price / wait / timing / MAE actions", "reason": "separate action layer; excluded"},
        {"field_or_family": "mapped_* / causal mapping outcomes", "reason": "post-score mapping/calibration values; excluded"},
    ])
    targets = pd.DataFrame([
        {"head": "short_p_net_positive", "exact_target": "1[execution_net_ev_12h > 0]", "conditioning": "all short strict rows", "label_availability": "execution_label_end_utc"},
        {"head": "short_conditional_favorable_magnitude", "exact_target": "execution_net_ev_12h", "conditioning": "execution_net_ev_12h > 0", "label_availability": "execution_label_end_utc"},
        {"head": "short_conditional_adverse_severity", "exact_target": "-execution_net_ev_12h", "conditioning": "execution_net_ev_12h <= 0", "label_availability": "execution_label_end_utc"},
    ])
    features = pd.DataFrame([{"head": head, "feature_order": index, "feature": feature} for head, names in recommended_feature_sets().items() for index, feature in enumerate(names)])
    stage = Path(tempfile.mkdtemp(dir=output_dir.parent, prefix=f".{output_dir.name}."))
    try:
        tables = {"feature_inventory.csv": inventory, "fold_provenance_coverage.csv": fold, "forbidden_fields.csv": forbidden, "recommended_feature_sets.csv": features, "target_contracts.csv": targets}
        for name, table in tables.items(): table.to_csv(stage / name, index=False)
        manifest = {
            "schema": "short_conditional_payoff_readiness_v1", "status": "READY_TO_TRAIN_STRICT_SHORT_ONLY_WITH_DECLARED_WHITELIST", "promotion_eligible": False,
            "scope": {"rows": int(len(frame)), "months": ["2025-03", "2025-04"], "short_rows": int(frame.side_name.eq("short").sum()), "long_rows_audit_only": int(frame.side_name.eq("long").sum())},
            "contracts": {"join": "complete one-to-one four-field identity across score triplet, peak/slope/MAE OOF streams and canonical fold panel", "score_triplet": list(SCORE_TRIPLET), "auxiliary_provenance": "strict side-local March-April OOF; pre-entry feature scan passed and label resolution is decision+12h", "labels": "exact current-spread frozen-policy H12 net; conditional targets are hard-class subsets only", "forbidden": forbidden.field_or_family.tolist(), "contexts": {"full_coverage_joinable": full_context, "partial_coverage_exclude_without_predeclared_missingness_rule": partial_context, "use": "all contexts are excluded from smallest baseline sets; test only as a separately predeclared ablation"}, "actions": "timing, target-price, wait and MAE actions remain separate"},
            "sources": {str(path): sha256(path) for path in required}, "outputs_sha256": {name: sha256(stage / name) for name in tables}, "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        }
        write_json(stage / "manifest.json", manifest); (stage / "manifest.sha256").write_text(f"{sha256(stage / 'manifest.json')}  manifest.json\n", encoding="utf-8"); os.replace(stage, output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True); raise
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    print(json.dumps(safe(run(parser.parse_args().output_dir)), sort_keys=True))
