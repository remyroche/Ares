#!/usr/bin/env python3
"""V2 meta ablation for cross-asset representation outputs.

This script tests whether the V1 representation outputs improve the existing
train_meta-style filter.  It is deliberately stricter than simply appending the
columns to the S52 handoff:

* representation inputs must be OOF/prior-fold on train rows;
* baseline and augmented variants use the same train/validation rows;
* month-forward validation is retained;
* metrics are top-k/path-quality metrics, not AUC-first.

With the current three-month S52 artifact, V1 representations exist for May and
June.  Therefore the strict V2 ablation can train on May and validate on June.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_s52_train_meta_regime_handoff_smoke import (  # noqa: E402
    KEY_COLUMNS,
    _breakdown_rows,
    _candidate_column,
    _feature_columns,
    _feature_importance,
    _fit_classifier,
    _fit_regressor,
    _fit_side_classifier,
    _json_safe,
    _load_joined_frame,
    _make_xy,
    _num,
    _predict,
    _selector_metrics,
    _summarize,
    _summarize_threshold_policies,
    _threshold_policy_rows,
)
from scripts.run_cross_asset_archetype_representation_v1 import AE_OUTPUT_COLUMNS  # noqa: E402


DEFAULT_HANDOFF_DIR = Path(
    "data_perp/reports/s52_trailing_profit_best_pointwise_scored_ledger_20260705_v1"
    "/s52_trailing_regime_meta_handoff_longsplit_v2"
)
DEFAULT_REPRESENTATION_DIR = DEFAULT_HANDOFF_DIR / "cross_asset_archetype_representation_v1"
DEFAULT_REPRESENTATION_PREDICTIONS = DEFAULT_REPRESENTATION_DIR / "cross_asset_representation_v1_predictions.parquet"
DEFAULT_OUT_DIR = DEFAULT_HANDOFF_DIR / "cross_asset_representation_meta_ablation_v2"

CROSS_LGBM_REPRESENTATION_COLUMNS = (
    "cross_lgbm_exec_margin_score",
    "cross_lgbm_bad_mae_score",
    "cross_lgbm_timeout_score",
    "cross_lgbm_dirty_positive_score",
    "cross_lgbm_clean_risk_composite",
)
AE_REPRESENTATION_COLUMNS = tuple(AE_OUTPUT_COLUMNS)
REPRESENTATION_COLUMNS = CROSS_LGBM_REPRESENTATION_COLUMNS + AE_REPRESENTATION_COLUMNS
REQUIRED_REPRESENTATION_COLUMNS = CROSS_LGBM_REPRESENTATION_COLUMNS + (
    "market_z_0",
    "market_z_1",
    "market_z_2",
    "market_z_3",
    "market_ae_recon_error",
    "market_ae_recon_error_pct",
    "market_ae_mahalanobis_diag",
)
CONDITIONAL_REPRESENTATION_SPECS = {
    "cond_cross_exec_margin_score": ("cross_lgbm_exec_margin_score", 1.0),
    "cond_cross_low_bad_mae_score": ("cross_lgbm_bad_mae_score", -1.0),
    "cond_cross_low_timeout_score": ("cross_lgbm_timeout_score", -1.0),
    "cond_cross_low_dirty_positive_score": ("cross_lgbm_dirty_positive_score", -1.0),
    "cond_cross_clean_risk_composite": ("cross_lgbm_clean_risk_composite", 1.0),
    "cond_market_ae_low_recon_error": ("market_ae_recon_error", -1.0),
    "cond_market_ae_low_mahalanobis": ("market_ae_mahalanobis_diag", -1.0),
}
CONDITIONAL_REPRESENTATION_COLUMNS = tuple(CONDITIONAL_REPRESENTATION_SPECS.keys())
CONDITIONAL_REPRESENTATION_MASK_COLUMNS = tuple(f"{col}_accepted_cell" for col in CONDITIONAL_REPRESENTATION_COLUMNS)
CONDITIONAL_CROSS_COLUMNS = tuple(
    col for col in CONDITIONAL_REPRESENTATION_COLUMNS if col.startswith("cond_cross_")
) + tuple(
    col for col in CONDITIONAL_REPRESENTATION_MASK_COLUMNS if col.startswith("cond_cross_")
)
CONDITIONAL_AE_COLUMNS = tuple(
    col for col in CONDITIONAL_REPRESENTATION_COLUMNS if col.startswith("cond_market_ae_")
) + tuple(
    col for col in CONDITIONAL_REPRESENTATION_MASK_COLUMNS if col.startswith("cond_market_ae_")
)
META_VARIANTS = {
    "m0_baseline_meta": (),
    "m1_cross_lgbm_meta": CROSS_LGBM_REPRESENTATION_COLUMNS,
    "m1b_cross_lgbm_risk_only_meta": (
        "cross_lgbm_bad_mae_score",
        "cross_lgbm_timeout_score",
        "cross_lgbm_dirty_positive_score",
        "cross_lgbm_clean_risk_composite",
    ),
    "m1c_cross_lgbm_badmae_only_meta": ("cross_lgbm_bad_mae_score",),
    "m2_market_ae_ood_meta": AE_REPRESENTATION_COLUMNS,
    "m3_cross_lgbm_plus_ae_meta": CROSS_LGBM_REPRESENTATION_COLUMNS + AE_REPRESENTATION_COLUMNS,
    "m4_conditional_cross_meta": CONDITIONAL_CROSS_COLUMNS,
    "m5_conditional_ae_meta": CONDITIONAL_AE_COLUMNS,
    "m6_conditional_cross_plus_ae_meta": CONDITIONAL_CROSS_COLUMNS + CONDITIONAL_AE_COLUMNS,
    "m7_cross_plus_conditional_cross_meta": CROSS_LGBM_REPRESENTATION_COLUMNS + CONDITIONAL_CROSS_COLUMNS,
}
V2_NEVER_FEATURE_COLUMNS = {
    "has_cross_lgbm_representation",
    "gmm_cluster_id",
    "aegmm_cluster",
    "side_aegmm_cluster",
    "long_bad_path_label",
    "long_path_clean_exec_label",
    "long_path_dirty_positive_label",
    "long_path_quality_soft",
    "long_path_slow_profit",
    "long_path_post_mfe_bad_drawdown",
    "long_path_full_bad_mae_1r",
    "long_path_time_to_profit_bars",
    "long_path_post_mfe_drawdown_norm",
}


def _load_frame_with_representations(
    *,
    handoff_dir: Path,
    ledger_path: Path | None,
    representation_predictions: Path,
    frontier: str,
) -> pd.DataFrame:
    handoff_path = handoff_dir / "train_meta_regime_handoff.parquet"
    if ledger_path is None:
        ledger_path = handoff_dir / "s52_trailing_regime_scored_ledger.parquet"
    frame = _load_joined_frame(handoff_path, ledger_path, frontier)
    pred_cols = list(KEY_COLUMNS) + ["month"] + list(REPRESENTATION_COLUMNS)
    try:
        import pyarrow.parquet as pq

        available = set(pq.read_schema(representation_predictions).names)
        use_cols = [col for col in pred_cols if col in available]
    except Exception:
        use_cols = pred_cols
    reps = pd.read_parquet(representation_predictions, columns=use_cols)
    missing = [col for col in REPRESENTATION_COLUMNS if col not in reps.columns]
    if missing:
        raise ValueError(f"Representation prediction file is missing columns: {missing}")
    reps = reps.drop_duplicates(list(KEY_COLUMNS), keep="last")
    merged = frame.merge(
        reps[list(KEY_COLUMNS) + list(REPRESENTATION_COLUMNS)],
        on=list(KEY_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    merged["has_cross_lgbm_representation"] = merged.loc[:, REQUIRED_REPRESENTATION_COLUMNS].notna().all(axis=1)
    return merged


def _variant_feature_columns(data: pd.DataFrame, variant: str) -> tuple[list[str], list[str]]:
    rep_cols = set(REPRESENTATION_COLUMNS) | set(CONDITIONAL_REPRESENTATION_COLUMNS) | set(CONDITIONAL_REPRESENTATION_MASK_COLUMNS)
    if variant == "m0_baseline_meta":
        base = data.drop(columns=[col for col in rep_cols if col in data.columns])
        numeric, categorical = _feature_columns(base)
        return _filter_v2_feature_columns(numeric, categorical)
    if variant in META_VARIANTS:
        allowed = set(META_VARIANTS[variant])
        drop_cols = [col for col in rep_cols if col in data.columns and col not in allowed]
        variant_data = data.drop(columns=drop_cols)
        numeric, categorical = _feature_columns(variant_data)
        return _filter_v2_feature_columns(numeric, categorical)
    raise ValueError(f"Unknown meta variant: {variant}")


def _top_fraction_metrics(frame: pd.DataFrame, score: pd.Series, frac: float) -> dict[str, float]:
    valid = frame.copy()
    valid["_score__tmp"] = pd.to_numeric(score, errors="coerce")
    valid = valid[valid["_score__tmp"].notna()]
    if valid.empty:
        return {
            "rows": 0.0,
            "exec_margin": float("nan"),
            "clean_exec": float("nan"),
            "bad_mae": float("nan"),
            "timeout": float("nan"),
        }
    n = max(1, int(math.ceil(len(valid) * float(frac))))
    top = valid.sort_values("_score__tmp", ascending=False).head(n)
    return {
        "rows": float(len(top)),
        "exec_margin": float(_num(top.get("exec_margin"), index=top.index).mean()),
        "clean_exec": float(_num(top.get("clean_exec_label"), index=top.index, default=0.0).mean()),
        "bad_mae": float(_num(top.get("full_path_bad_mae_1r"), index=top.index, default=0.0).mean()),
        "timeout": float(_num(top.get("timeout"), index=top.index, default=0.0).mean()),
        "mfe_before_mae": float(_num(top.get("mfe_before_mae_1r"), index=top.index, default=0.0).mean()),
        "mae_before_mfe": float(_num(top.get("mae_before_mfe_1r"), index=top.index, default=0.0).mean()),
    }


def _control_scores(score: pd.Series, frame: pd.DataFrame, *, seed: int) -> list[pd.Series]:
    rng = np.random.default_rng(int(seed))
    base = _num(score).replace([np.inf, -np.inf], np.nan)
    valid_values = base.dropna().to_numpy(dtype=np.float64)
    if len(valid_values) == 0:
        valid_values = np.array([0.0], dtype=np.float64)
    perm_values = valid_values.copy()
    rng.shuffle(perm_values)
    perm = pd.Series(np.resize(perm_values, len(base)), index=base.index, dtype=np.float32)
    block = pd.Series(np.nan, index=base.index, dtype=np.float32)
    if "month" in frame.columns:
        for _, idx in frame.groupby("month", dropna=False).groups.items():
            vals = base.loc[idx].dropna().to_numpy(dtype=np.float64)
            if len(vals) == 0:
                vals = valid_values
            vals = vals.copy()
            rng.shuffle(vals)
            block.loc[idx] = np.resize(vals, len(idx)).astype(np.float32)
    else:
        block = perm.copy()
    noise = rng.normal(size=len(base)).astype(np.float64)
    for idx in range(1, len(noise)):
        noise[idx] = 0.75 * noise[idx - 1] + noise[idx]
    std = float(np.nanstd(valid_values)) if len(valid_values) else 1.0
    mean = float(np.nanmean(valid_values)) if len(valid_values) else 0.0
    noise = (noise - float(np.nanmean(noise))) / max(float(np.nanstd(noise)), 1e-9)
    noise_ar1 = pd.Series(mean + std * noise, index=base.index, dtype=np.float32)
    return [perm, block, noise_ar1]


def _accepted_cells_from_train(
    train: pd.DataFrame,
    *,
    min_rows: int,
    min_clean_rows: int,
    top_frac: float,
) -> tuple[dict[str, set[tuple[str, str]]], list[dict[str, Any]]]:
    accepted: dict[str, set[tuple[str, str]]] = {col: set() for col in CONDITIONAL_REPRESENTATION_COLUMNS}
    diagnostics: list[dict[str, Any]] = []
    required = {"side_name", "source_semantic_family", "score"}
    if not required.issubset(train.columns):
        return accepted, diagnostics
    group_cols = ["side_name", "source_semantic_family"]
    for cell_idx, ((side, family), cell) in enumerate(train.groupby(group_cols, dropna=False)):
        side_key = str(side)
        family_key = str(family)
        if len(cell) < int(min_rows):
            continue
        clean_rows = int(_num(cell.get("clean_exec_label"), index=cell.index, default=0.0).fillna(0.0).gt(0.5).sum())
        if clean_rows < int(min_clean_rows):
            continue
        base_metrics = _top_fraction_metrics(cell, _num(cell.get("score"), index=cell.index), top_frac)
        if not np.isfinite(base_metrics["exec_margin"]):
            continue
        for out_col, (source_col, direction) in CONDITIONAL_REPRESENTATION_SPECS.items():
            if source_col not in cell.columns:
                continue
            candidate_score = direction * _num(cell.get(source_col), index=cell.index)
            metrics = _top_fraction_metrics(cell, candidate_score, top_frac)
            if not np.isfinite(metrics["exec_margin"]):
                continue
            control_exec_values = [
                _top_fraction_metrics(cell, control_score, top_frac)["exec_margin"]
                for control_score in _control_scores(candidate_score, cell, seed=20260705 + cell_idx * 100 + len(diagnostics))
            ]
            control_median = float(np.nanmedian(control_exec_values)) if control_exec_values else float("nan")
            control_std = float(np.nanstd(control_exec_values)) if control_exec_values else float("nan")
            control_adjusted_exec = metrics["exec_margin"] - control_median - 0.5 * control_std
            delta_exec = metrics["exec_margin"] - base_metrics["exec_margin"]
            delta_clean = metrics["clean_exec"] - base_metrics["clean_exec"]
            delta_bad = metrics["bad_mae"] - base_metrics["bad_mae"]
            delta_timeout = metrics["timeout"] - base_metrics["timeout"]
            delta_mfe_first = metrics["mfe_before_mae"] - base_metrics["mfe_before_mae"]
            delta_mae_first = metrics["mae_before_mfe"] - base_metrics["mae_before_mfe"]
            utility_accept = (
                delta_exec > 0.0
                and control_adjusted_exec > 0.0
                and (delta_bad <= 0.0 or delta_clean >= 0.0 or delta_mfe_first >= 0.0)
                and delta_timeout <= 0.02
            )
            risk_accept = (
                delta_bad <= -0.05
                and delta_exec >= -0.0015
                and control_adjusted_exec >= -0.0005
                and delta_timeout <= 0.02
            )
            path_accept = (
                (delta_mfe_first >= 0.05 or delta_mae_first <= -0.05)
                and delta_exec >= -0.0015
                and control_adjusted_exec >= -0.0005
                and delta_bad <= 0.05
            )
            accepted_flag = bool(utility_accept or risk_accept or path_accept)
            diagnostics.append(
                {
                    "side_name": side_key,
                    "source_semantic_family": family_key,
                    "conditional_feature": out_col,
                    "source_col": source_col,
                    "top_frac": float(top_frac),
                    "cell_rows": int(len(cell)),
                    "clean_rows": int(clean_rows),
                    "top_rows": int(metrics["rows"]),
                    "base_exec_margin": float(base_metrics["exec_margin"]),
                    "candidate_exec_margin": float(metrics["exec_margin"]),
                    "delta_exec_margin": float(delta_exec),
                    "delta_clean_exec": float(delta_clean),
                    "delta_bad_mae": float(delta_bad),
                    "delta_timeout": float(delta_timeout),
                    "delta_mfe_before_mae": float(delta_mfe_first),
                    "delta_mae_before_mfe": float(delta_mae_first),
                    "control_exec_margin_median": control_median,
                    "control_exec_margin_std": control_std,
                    "control_adjusted_exec_margin": float(control_adjusted_exec),
                    "utility_accept": bool(utility_accept),
                    "risk_accept": bool(risk_accept),
                    "path_accept": bool(path_accept),
                    "accepted": accepted_flag,
                }
            )
            if accepted_flag:
                accepted[out_col].add((side_key, family_key))
    return accepted, diagnostics


def _with_conditional_representation_features(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    min_rows: int,
    min_clean_rows: int,
    top_frac: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[tuple[str, str]]], list[dict[str, Any]]]:
    accepted, diagnostics = _accepted_cells_from_train(
        train,
        min_rows=min_rows,
        min_clean_rows=min_clean_rows,
        top_frac=top_frac,
    )
    out_train = train.copy()
    out_valid = valid.copy()
    for out_col, (source_col, direction) in CONDITIONAL_REPRESENTATION_SPECS.items():
        cells = accepted.get(out_col, set())
        for frame in (out_train, out_valid):
            if source_col not in frame.columns:
                frame[out_col] = 0.0
                frame[f"{out_col}_accepted_cell"] = 0.0
                continue
            cell_keys = list(zip(frame["side_name"].astype(str), frame["source_semantic_family"].astype(str)))
            active = pd.Series([key in cells for key in cell_keys], index=frame.index)
            score = direction * _num(frame.get(source_col), index=frame.index).replace([np.inf, -np.inf], np.nan)
            frame[out_col] = score.where(active, 0.0).fillna(0.0).astype(np.float32)
            frame[f"{out_col}_accepted_cell"] = active.astype(np.float32)
    manifest = {col: sorted(cells) for col, cells in accepted.items()}
    return out_train, out_valid, manifest, diagnostics


def _filter_v2_feature_columns(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[list[str], list[str]]:
    numeric = [col for col in numeric_cols if col not in V2_NEVER_FEATURE_COLUMNS]
    categorical = [col for col in categorical_cols if col not in V2_NEVER_FEATURE_COLUMNS]
    return numeric, categorical


def _score_meta_variant(
    *,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    variant: str,
    seed: int,
    fold_idx: int,
) -> tuple[pd.DataFrame, list[pd.DataFrame], list[str]]:
    x_train, x_valid, feature_names = _make_xy(
        train,
        valid,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )
    models = {
        "clean_exec": _fit_classifier(x_train, train["clean_exec_label"], train, seed + fold_idx),
        "positive_margin": _fit_classifier(x_train, train["positive_exec_margin"], train, seed + 11 + fold_idx),
        "bad_path": _fit_classifier(x_train, train["bad_path_label"], train, seed + 23 + fold_idx),
        "timeout": _fit_classifier(x_train, train["timeout"], train, seed + 31 + fold_idx),
        "exec_margin": _fit_regressor(x_train, train["exec_margin"], train, seed + 43 + fold_idx),
        "long_clean_exec": _fit_side_classifier(
            x_train,
            train["long_path_clean_exec_label"],
            train,
            side="long",
            seed=seed + 53 + fold_idx,
        ),
        "long_bad_path": _fit_side_classifier(
            x_train,
            train["long_bad_path_label"],
            train,
            side="long",
            seed=seed + 61 + fold_idx,
        ),
    }
    prefix = f"score_{variant}"
    scored = valid.copy()
    scored[f"{prefix}_clean_exec"] = _predict(models["clean_exec"], x_valid, classifier=True)
    scored[f"{prefix}_positive_margin"] = _predict(models["positive_margin"], x_valid, classifier=True)
    scored[f"{prefix}_bad_path"] = _predict(models["bad_path"], x_valid, classifier=True)
    scored[f"{prefix}_timeout"] = _predict(models["timeout"], x_valid, classifier=True)
    scored[f"{prefix}_exec_margin"] = _predict(models["exec_margin"], x_valid, classifier=False)
    long_valid = scored["side_name"].astype(str).str.lower().eq("long")
    scored[f"{prefix}_long_clean_exec"] = _predict(models["long_clean_exec"], x_valid, classifier=True)
    scored.loc[~long_valid, f"{prefix}_long_clean_exec"] = np.nan
    scored[f"{prefix}_long_bad_path"] = _predict(models["long_bad_path"], x_valid, classifier=True)
    scored.loc[~long_valid, f"{prefix}_long_bad_path"] = np.nan
    scored[f"{prefix}_clean_minus_risk"] = (
        scored[f"{prefix}_clean_exec"].fillna(0.0)
        + 0.60 * scored[f"{prefix}_positive_margin"].fillna(0.0)
        - 0.70 * scored[f"{prefix}_bad_path"].fillna(0.0)
        - 0.30 * scored[f"{prefix}_timeout"].fillna(0.0)
    )
    scored[f"{prefix}_exec_margin_risk_blend"] = (
        scored[f"{prefix}_exec_margin"].fillna(0.0)
        + 0.0030 * scored[f"{prefix}_clean_exec"].fillna(0.0)
        + 0.0020 * scored[f"{prefix}_positive_margin"].fillna(0.0)
        - 0.0040 * scored[f"{prefix}_bad_path"].fillna(0.0)
        - 0.0020 * scored[f"{prefix}_timeout"].fillna(0.0)
    )
    long_clean = scored[f"{prefix}_long_clean_exec"].where(
        scored[f"{prefix}_long_clean_exec"].notna(),
        scored[f"{prefix}_clean_exec"],
    )
    long_bad = scored[f"{prefix}_long_bad_path"].where(
        scored[f"{prefix}_long_bad_path"].notna(),
        scored[f"{prefix}_bad_path"],
    )
    scored[f"{prefix}_long_aware_clean_minus_risk"] = scored[f"{prefix}_clean_minus_risk"]
    scored.loc[long_valid, f"{prefix}_long_aware_clean_minus_risk"] = (
        long_clean.loc[long_valid].fillna(0.0)
        + 0.55 * scored.loc[long_valid, f"{prefix}_positive_margin"].fillna(0.0)
        - 0.80 * long_bad.loc[long_valid].fillna(1.0)
        - 0.25 * scored.loc[long_valid, f"{prefix}_timeout"].fillna(1.0)
    )
    importances = [
        _feature_importance(model, feature_names, f"{variant}_{label}", str(valid["month"].iloc[0]))
        for label, model in models.items()
    ]
    return scored, importances, feature_names


def _threshold_rows_for_variant(scored: pd.DataFrame, variant: str, test_month: str) -> list[dict[str, Any]]:
    prefix = f"score_{variant}"
    tmp = scored.copy()
    tmp["score_meta_clean_exec"] = tmp[f"{prefix}_clean_exec"]
    tmp["score_meta_positive_margin"] = tmp[f"{prefix}_positive_margin"]
    tmp["score_meta_bad_path"] = tmp[f"{prefix}_bad_path"]
    tmp["score_meta_timeout"] = tmp[f"{prefix}_timeout"]
    tmp["score_meta_exec_margin"] = tmp[f"{prefix}_exec_margin"]
    tmp["score_meta_long_clean_exec"] = tmp[f"{prefix}_long_clean_exec"]
    tmp["score_meta_long_bad_path"] = tmp[f"{prefix}_long_bad_path"]
    selector_cols = {
        f"{variant}_clean_exec": f"{prefix}_clean_exec",
        f"{variant}_positive_margin": f"{prefix}_positive_margin",
        f"{variant}_exec_margin": f"{prefix}_exec_margin",
        f"{variant}_clean_minus_risk": f"{prefix}_clean_minus_risk",
        f"{variant}_exec_margin_risk_blend": f"{prefix}_exec_margin_risk_blend",
        f"{variant}_long_aware_clean_minus_risk": f"{prefix}_long_aware_clean_minus_risk",
    }
    return _threshold_policy_rows(tmp, selector_cols, test_month)


def _summarize_ablation(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    out = summary.copy()
    out["variant"] = out["selector"].astype(str).str.extract(r"^(m[0-9]+_[^_]+(?:_[^_]+)?)", expand=False)
    # Keep explicit robust parsing for current names.
    for variant in META_VARIANTS:
        out.loc[out["selector"].astype(str).str.startswith(variant), "variant"] = variant
    comparable = out[out["selector"].astype(str).str.endswith("long_aware_clean_minus_risk")].copy()
    if comparable.empty:
        comparable = out[out["selector"].astype(str).str.endswith("clean_minus_risk")].copy()
    return comparable.sort_values(
        ["mean_keep010_exec_margin", "mean_keep010_full_path_bad_mae", "mean_keep010_timeout"],
        ascending=[False, True, True],
    )


def _promotion_recommendations(ablation_summary: pd.DataFrame) -> dict[str, Any]:
    if ablation_summary.empty or "variant" not in ablation_summary.columns:
        return {
            "status": "no_valid_evidence",
            "promote_to_deeper_meta_eval": [],
            "shadow_only": list(META_VARIANTS),
            "reason": "No comparable ablation summary rows were produced.",
        }
    rows = ablation_summary.drop_duplicates("variant", keep="first").set_index("variant", drop=False)
    baseline = rows.loc["m0_baseline_meta"] if "m0_baseline_meta" in rows.index else None
    promote: list[dict[str, Any]] = []
    shadow: list[dict[str, Any]] = []

    def metric(row: pd.Series, col: str, default: float = float("nan")) -> float:
        try:
            return float(row.get(col, default))
        except Exception:
            return default

    for variant in sorted(META_VARIANTS):
        if variant == "m0_baseline_meta":
            continue
        if variant not in rows.index or baseline is None:
            shadow.append({"variant": variant, "reason": "missing_comparable_row"})
            continue
        row = rows.loc[variant]
        exec_delta = metric(row, "mean_keep030_exec_margin") - metric(baseline, "mean_keep030_exec_margin")
        bad_delta = metric(row, "mean_keep030_full_path_bad_mae") - metric(baseline, "mean_keep030_full_path_bad_mae")
        timeout_delta = metric(row, "mean_keep030_timeout") - metric(baseline, "mean_keep030_timeout")
        recall_delta = metric(row, "mean_keep030_oracle_recall") - metric(baseline, "mean_keep030_oracle_recall")
        top10_exec_delta = metric(row, "mean_keep010_exec_margin") - metric(baseline, "mean_keep010_exec_margin")
        top10_bad_delta = metric(row, "mean_keep010_full_path_bad_mae") - metric(baseline, "mean_keep010_full_path_bad_mae")
        passes = (
            np.isfinite(exec_delta)
            and exec_delta > 0.00025
            and bad_delta <= 0.0
            and timeout_delta <= 0.005
            and recall_delta >= -0.005
            and top10_exec_delta >= -0.00025
            and top10_bad_delta <= 0.010
        )
        candidate = {
            "variant": variant,
            "feature_columns": list(META_VARIANTS[variant]),
            "mean_keep030_exec_margin": metric(row, "mean_keep030_exec_margin"),
            "mean_keep030_full_path_bad_mae": metric(row, "mean_keep030_full_path_bad_mae"),
            "mean_keep030_timeout": metric(row, "mean_keep030_timeout"),
            "mean_keep030_oracle_recall": metric(row, "mean_keep030_oracle_recall"),
            "delta_vs_baseline": {
                "mean_keep030_exec_margin": exec_delta,
                "mean_keep030_full_path_bad_mae": bad_delta,
                "mean_keep030_timeout": timeout_delta,
                "mean_keep030_oracle_recall": recall_delta,
                "mean_keep010_exec_margin": top10_exec_delta,
                "mean_keep010_full_path_bad_mae": top10_bad_delta,
            },
        }
        if passes:
            candidate["reason"] = "top30_exec_improves_with_nonworse_bad_mae_and_stable_timeout_recall"
            promote.append(candidate)
        else:
            candidate["reason"] = "does_not_clear_conservative_top30_promotion_rule"
            shadow.append(candidate)
    return {
        "status": "candidate_features_available" if promote else "shadow_only",
        "promotion_rule": {
            "exec_delta_gt": 0.00025,
            "bad_mae_delta_lte": 0.0,
            "timeout_delta_lte": 0.005,
            "oracle_recall_delta_gte": -0.005,
            "top10_exec_delta_gte": -0.00025,
            "top10_bad_mae_delta_lte": 0.010,
            "basis": "strict same-row June-only V2 top30 comparison versus m0_baseline_meta",
        },
        "promote_to_deeper_meta_eval": promote,
        "shadow_only": shadow,
        "baseline_variant": "m0_baseline_meta" if baseline is not None else None,
    }


def run_ablation(
    *,
    handoff_dir: Path,
    ledger_path: Path | None,
    representation_predictions: Path,
    out_dir: Path,
    frontier: str,
    train_scope: str,
    seed: int,
    min_train_rows: int,
    min_valid_rows: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = _load_frame_with_representations(
        handoff_dir=handoff_dir,
        ledger_path=ledger_path,
        representation_predictions=representation_predictions,
        frontier=frontier,
    )
    selected_col = _candidate_column(frontier)
    if train_scope == "selected":
        data = data[data[selected_col]].copy()
    elif train_scope != "all":
        raise ValueError("--train-scope must be selected or all")
    months = sorted(str(m) for m in data["month"].dropna().unique())
    fold_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    breakdown_rows: list[dict[str, Any]] = []
    importances: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    feature_manifest: dict[str, list[str]] = {}
    conditional_acceptance_manifest: dict[str, Any] = {}
    conditional_diagnostic_rows: list[dict[str, Any]] = []
    scored_months: list[str] = []
    for fold_idx, test_month in enumerate(months[1:], start=1):
        train_all = data[data["month"].astype(str).lt(str(test_month))].copy()
        valid = data[
            data["month"].astype(str).eq(str(test_month))
            & data["has_cross_lgbm_representation"].astype(bool)
        ].copy()
        train = train_all[train_all["has_cross_lgbm_representation"].astype(bool)].copy()
        if len(train) < int(min_train_rows) or len(valid) < int(min_valid_rows):
            continue
        scored_months.append(str(test_month))
        print(
            json.dumps(
                {
                    "event": "cross_asset_meta_ablation_fold_start",
                    "test_month": test_month,
                    "train_rows": int(len(train)),
                    "valid_rows": int(len(valid)),
                    "train_months": sorted(train["month"].astype(str).unique().tolist()),
                    "valid_months": sorted(valid["month"].astype(str).unique().tolist()),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        combined = valid.copy()
        combined["score_base"] = _num(combined.get("score"), index=combined.index)
        train_cond, valid_cond, accepted_cells, acceptance_diagnostics = _with_conditional_representation_features(
            train,
            valid,
            min_rows=max(50, int(min_train_rows)),
            min_clean_rows=5,
            top_frac=0.10,
        )
        for row in acceptance_diagnostics:
            row["test_month"] = str(test_month)
        conditional_diagnostic_rows.extend(acceptance_diagnostics)
        conditional_acceptance_manifest[str(test_month)] = {
            col: [
                {"side_name": side, "source_semantic_family": family}
                for side, family in cells
            ]
            for col, cells in accepted_cells.items()
        }
        for variant in META_VARIANTS:
            if variant.startswith("m4_") or variant.startswith("m5_") or variant.startswith("m6_") or variant.startswith("m7_"):
                train_variant = train_cond
                valid_variant = valid_cond
                feature_source = pd.concat([train_cond, valid_cond], ignore_index=True, sort=False)
            else:
                train_variant = train
                valid_variant = valid
                feature_source = data
            numeric_cols, categorical_cols = _variant_feature_columns(feature_source, variant)
            feature_manifest[variant] = numeric_cols + categorical_cols
            scored, imps, _ = _score_meta_variant(
                train=train_variant,
                valid=valid_variant,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                variant=variant,
                seed=seed,
                fold_idx=fold_idx * 100 + len(feature_manifest),
            )
            importances.extend(imps)
            prefix = f"score_{variant}"
            selector_cols = {
                f"{variant}_clean_exec": f"{prefix}_clean_exec",
                f"{variant}_positive_margin": f"{prefix}_positive_margin",
                f"{variant}_exec_margin": f"{prefix}_exec_margin",
                f"{variant}_clean_minus_risk": f"{prefix}_clean_minus_risk",
                f"{variant}_exec_margin_risk_blend": f"{prefix}_exec_margin_risk_blend",
                f"{variant}_long_aware_clean_minus_risk": f"{prefix}_long_aware_clean_minus_risk",
            }
            for selector, score_col in selector_cols.items():
                fold_rows.append(_selector_metrics(scored, score_col, selector, test_month))
                for keep_frac in (0.30, 0.20, 0.10):
                    breakdown_rows.extend(_breakdown_rows(scored, score_col, selector, test_month, keep_frac))
            threshold_rows.extend(_threshold_rows_for_variant(scored, variant, test_month))
            score_cols = [col for col in scored.columns if col.startswith(prefix)]
            combined = combined.merge(
                scored[list(KEY_COLUMNS) + score_cols],
                on=list(KEY_COLUMNS),
                how="left",
                validate="one_to_one",
            )
        prediction_frames.append(combined)
    folds = pd.DataFrame(fold_rows)
    summary = _summarize(folds)
    ablation_summary = _summarize_ablation(summary)
    promotion = _promotion_recommendations(ablation_summary)
    threshold_policy = pd.DataFrame(threshold_rows)
    threshold_summary = _summarize_threshold_policies(threshold_policy)
    breakdown = pd.DataFrame(breakdown_rows)
    conditional_diagnostics = pd.DataFrame(conditional_diagnostic_rows)
    importance = (
        pd.concat([part for part in importances if not part.empty], ignore_index=True)
        if any(not part.empty for part in importances)
        else pd.DataFrame(columns=["test_month", "model", "feature", "importance"])
    )
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    folds.to_csv(out_dir / "cross_asset_representation_meta_ablation_v2_folds.csv", index=False)
    summary.to_csv(out_dir / "cross_asset_representation_meta_ablation_v2_summary.csv", index=False)
    ablation_summary.to_csv(out_dir / "cross_asset_representation_meta_ablation_v2_comparison.csv", index=False)
    (out_dir / "cross_asset_representation_meta_ablation_v2_promotion.json").write_text(
        json.dumps(_json_safe(promotion), indent=2, sort_keys=True)
    )
    threshold_policy.to_csv(out_dir / "cross_asset_representation_meta_ablation_v2_threshold_policy_folds.csv", index=False)
    threshold_summary.to_csv(out_dir / "cross_asset_representation_meta_ablation_v2_threshold_policy_summary.csv", index=False)
    breakdown.to_csv(out_dir / "cross_asset_representation_meta_ablation_v2_breakdown.csv", index=False)
    conditional_diagnostics.to_csv(out_dir / "cross_asset_representation_meta_ablation_v2_conditional_acceptance.csv", index=False)
    importance.to_csv(out_dir / "cross_asset_representation_meta_ablation_v2_feature_importance.csv", index=False)
    if not predictions.empty:
        predictions.to_parquet(out_dir / "cross_asset_representation_meta_ablation_v2_predictions.parquet", index=False)
    manifest = {
        "generated_by": "run_cross_asset_representation_meta_ablation_v2",
        "handoff_dir": str(handoff_dir),
        "ledger_path": str(ledger_path) if ledger_path is not None else str(handoff_dir / "s52_trailing_regime_scored_ledger.parquet"),
        "representation_predictions": str(representation_predictions),
        "out_dir": str(out_dir),
        "frontier": str(frontier),
        "train_scope": str(train_scope),
        "months": months,
        "scored_months": scored_months,
        "rows": int(len(data)),
        "rows_with_oof_representation": int(data["has_cross_lgbm_representation"].sum()),
        "prediction_rows": int(len(predictions)),
        "variants": sorted(META_VARIANTS),
        "feature_columns_by_variant": feature_manifest,
        "conditional_acceptance_by_scored_month": conditional_acceptance_manifest,
        "conditional_acceptance_rows": int(len(conditional_diagnostics)),
        "conditional_accepted_rows": int(conditional_diagnostics["accepted"].sum()) if "accepted" in conditional_diagnostics.columns else 0,
        "promotion_recommendation": promotion,
        "leakage_contract": {
            "representation_inputs": "OOF/prior-fold V1 predictions only",
            "conditional_acceptance": "accepted side x source_semantic_family cells are learned from train rows only, then applied to the validation month",
            "fold_rule": "train rows require representation outputs and are strictly earlier than validation month",
            "fair_comparison": "baseline and augmented variants use identical train/validation rows",
            "current_artifact_note": "With Apr/May/Jun data and V1 predictions for May/Jun, strict V2 scores June only.",
            "targets": "outcomes joined only for model labels and validation diagnostics",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))
    _write_report(out_dir, manifest, ablation_summary, threshold_summary, promotion)
    return manifest


def _write_report(
    out_dir: Path,
    manifest: dict[str, Any],
    ablation_summary: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    promotion: dict[str, Any],
) -> None:
    lines = [
        "# Cross-Asset Representation Meta Ablation V2",
        "",
        "## Scope",
        "",
        "Strict month-forward train_meta-style ablation comparing baseline meta features with baseline + V1 cross-LGBM representation scores.",
        "",
        "## Contract",
        "",
        f"- rows: `{manifest.get('rows')}`",
        f"- rows with OOF representation: `{manifest.get('rows_with_oof_representation')}`",
        f"- prediction rows: `{manifest.get('prediction_rows')}`",
        f"- scored months: `{', '.join(manifest.get('scored_months', []))}`",
        f"- variants: `{', '.join(manifest.get('variants', []))}`",
        "",
        "## Main Comparison",
        "",
    ]
    if ablation_summary.empty:
        lines.append("No valid ablation folds were produced.")
    else:
        cols = [
            "selector",
            "meta_smoke_status",
            "mean_keep030_exec_margin",
            "mean_keep030_clean_exec_precision",
            "mean_keep030_full_path_bad_mae",
            "mean_keep030_timeout",
            "mean_keep020_exec_margin",
            "mean_keep010_exec_margin",
            "mean_keep010_full_path_bad_mae",
            "mean_keep010_timeout",
        ]
        lines.append(ablation_summary[[col for col in cols if col in ablation_summary.columns]].to_markdown(index=False))
    lines.extend(["", "## Threshold Policy Summary", ""])
    if threshold_summary.empty:
        lines.append("No threshold-policy rows were produced.")
    else:
        cols = [
            "selector",
            "policy_id",
            "budget_frac",
            "threshold_policy_status",
            "mean_exec_margin",
            "worst_exec_margin",
            "mean_full_path_bad_mae",
            "mean_timeout",
            "mean_selected_rows",
            "mean_long_share",
        ]
        lines.append(threshold_summary[[col for col in cols if col in threshold_summary.columns]].head(30).to_markdown(index=False))
    lines.extend(["", "## Promotion Recommendation", ""])
    lines.append(f"- status: `{promotion.get('status')}`")
    promoted = promotion.get("promote_to_deeper_meta_eval", [])
    if promoted:
        for item in promoted:
            lines.append(
                f"- promote `{item.get('variant')}`: top30 exec `{_json_safe(item.get('mean_keep030_exec_margin'))}`, "
                f"bad-MAE `{_json_safe(item.get('mean_keep030_full_path_bad_mae'))}`, "
                f"timeout `{_json_safe(item.get('mean_keep030_timeout'))}`"
            )
    else:
        lines.append("- No representation variant clears the conservative promotion rule.")
    lines.extend(
        [
            "",
            "## Leakage Notes",
            "",
            "- Representation inputs are V1 OOF/prior-fold predictions.",
            "- Baseline and augmented variants use the same rows.",
            "- No in-sample April representation is backfilled.",
            "- This is a meta ablation, not frozen replay evidence.",
            "",
        ]
    )
    (out_dir / "cross_asset_representation_meta_ablation_v2_report.md").write_text("\n".join(lines))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-dir", type=Path, default=DEFAULT_HANDOFF_DIR)
    parser.add_argument("--ledger-path", type=Path, default=None)
    parser.add_argument("--representation-predictions", type=Path, default=DEFAULT_REPRESENTATION_PREDICTIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--frontier", default="10")
    parser.add_argument("--train-scope", choices=("selected", "all"), default="selected")
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--min-train-rows", type=int, default=100)
    parser.add_argument("--min-valid-rows", type=int, default=30)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_ablation(
        handoff_dir=args.handoff_dir,
        ledger_path=args.ledger_path,
        representation_predictions=args.representation_predictions,
        out_dir=args.out_dir,
        frontier=args.frontier,
        train_scope=args.train_scope,
        seed=args.seed,
        min_train_rows=args.min_train_rows,
        min_valid_rows=args.min_valid_rows,
    )
    print(json.dumps(_json_safe({"event": "cross_asset_meta_ablation_done", **manifest}), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
