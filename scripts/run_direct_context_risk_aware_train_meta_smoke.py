#!/usr/bin/env python3
"""Risk-aware train_meta smoke for the direct cross-asset context handoff.

This consumes ``train_meta_direct_context_handoff.parquet`` and its feature
manifest.  It fits month-forward heads for:

* EV after 1% round-trip cost;
* full stop-loss proxy;
* timeout;
* clean executable proxy.

The heads are combined into simple risk-adjusted scores and evaluated with
top-k precision/EV/path-quality metrics.  This is a train_meta candidate smoke,
not a frozen replay.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_FEATURE_SET_DIR = Path(
    "data_perp/reports/contextual_tp_sl_ablation_workflow_v14_runtime_health_20260701/"
    "direct_cross_asset_meta_context_v1/train_meta_direct_context_feature_set_v1"
)
DEFAULT_HANDOFF = DEFAULT_FEATURE_SET_DIR / "train_meta_direct_context_handoff.parquet"
DEFAULT_FEATURE_MANIFEST = DEFAULT_FEATURE_SET_DIR / "train_meta_direct_context_feature_manifest.json"
DEFAULT_OUT_DIR = DEFAULT_FEATURE_SET_DIR / "risk_aware_train_meta_smoke_v1"

TOP_FRACS = (0.30, 0.20, 0.10)
TARGETS = {
    "ev": "exec_ev_after_1pct_cost",
    "full_sl": "full_sl",
    "timeout": "timeout",
    "clean": "clean_exec_proxy",
}
SELECTOR_SPECS = {
    "s0_ev_only": {
        "weights": {"ev": 1.0, "full_sl": 0.0, "timeout": 0.0, "clean": 0.0},
        "guards": {},
    },
    "s1_ev_minus_fullsl": {
        "weights": {"ev": 1.0, "full_sl": -0.015, "timeout": 0.0, "clean": 0.0},
        "guards": {},
    },
    "s2_ev_minus_timeout": {
        "weights": {"ev": 1.0, "full_sl": 0.0, "timeout": -0.015, "clean": 0.0},
        "guards": {},
    },
    "s3_ev_minus_fullsl_timeout": {
        "weights": {"ev": 1.0, "full_sl": -0.015, "timeout": -0.015, "clean": 0.0},
        "guards": {},
    },
    "s4_ev_clean_minus_risk": {
        "weights": {"ev": 1.0, "full_sl": -0.012, "timeout": -0.012, "clean": 0.006},
        "guards": {},
    },
    "s5_clean_first": {
        "weights": {"ev": 0.5, "full_sl": -0.015, "timeout": -0.015, "clean": 0.012},
        "guards": {},
    },
    "s6_ev_strong_fullsl": {
        "weights": {"ev": 1.0, "full_sl": -0.040, "timeout": -0.005, "clean": 0.004},
        "guards": {},
    },
    "s7_clean_strong_fullsl": {
        "weights": {"ev": 0.65, "full_sl": -0.040, "timeout": -0.010, "clean": 0.012},
        "guards": {},
    },
    "s8_ev_fullsl_guard_q65": {
        "weights": {"ev": 1.0, "full_sl": 0.0, "timeout": 0.0, "clean": 0.0},
        "guards": {"pred_full_sl": 0.65},
    },
    "s9_clean_fullsl_guard_q65": {
        "weights": {"ev": 0.5, "full_sl": -0.015, "timeout": -0.015, "clean": 0.012},
        "guards": {"pred_full_sl": 0.65},
    },
    "s10_clean_fullsl_guard_q50": {
        "weights": {"ev": 0.5, "full_sl": -0.015, "timeout": -0.015, "clean": 0.012},
        "guards": {"pred_full_sl": 0.50},
    },
    "s11_clean_dual_guard_q70": {
        "weights": {"ev": 0.65, "full_sl": -0.030, "timeout": -0.012, "clean": 0.012},
        "guards": {"pred_full_sl": 0.70, "pred_timeout": 0.70},
    },
    "s12_ev_clean_strong_risk": {
        "weights": {"ev": 1.0, "full_sl": -0.060, "timeout": -0.030, "clean": 0.018},
        "guards": {},
    },
    "s13_ev_clean_fullsl_neutral_timeout": {
        "weights": {"ev": 1.0, "full_sl": -0.060, "timeout": -0.025, "clean": 0.012},
        "guards": {},
    },
}
CELL_AWARE_SELECTOR_SPECS = {
    "s14_cell_prior_fullsl_s12": {
        "base_selector": "s12_ev_clean_strong_risk",
        "full_sl_scale": 0.04,
        "timeout_scale": 0.04,
        "clean_scale": 0.0,
        "top_frac": 0.10,
    },
    "s15_cell_prior_fullsl_timeout_s12": {
        "base_selector": "s12_ev_clean_strong_risk",
        "full_sl_scale": 0.02,
        "timeout_scale": 0.03,
        "clean_scale": 0.03,
        "top_frac": 0.10,
    },
    "s16_cell_prior_clean_risk_s12": {
        "base_selector": "s12_ev_clean_strong_risk",
        "full_sl_scale": 0.06,
        "timeout_scale": 0.04,
        "clean_scale": 0.005,
        "top_frac": 0.10,
    },
    "s17_cell_prior_ev_fullsl_s12": {
        "base_selector": "s12_ev_clean_strong_risk",
        "full_sl_scale": 0.035,
        "timeout_scale": 0.015,
        "clean_scale": 0.010,
        "ev_shortfall_scale": 1.25,
        "ev_premium_scale": 0.35,
        "top_frac": 0.10,
    },
    "s18_long_cell_prior_ev_fullsl_s12": {
        "base_selector": "s12_ev_clean_strong_risk",
        "full_sl_scale": 0.045,
        "timeout_scale": 0.010,
        "clean_scale": 0.010,
        "ev_shortfall_scale": 1.75,
        "ev_premium_scale": 0.20,
        "side_focus": "long",
        "top_frac": 0.10,
    },
}
COMPOSITE_SELECTOR_SPECS = {
    "s19_long_s16_short_s12": {
        "long_selector": "s16_cell_prior_clean_risk_s12",
        "fallback_selector": "s12_ev_clean_strong_risk",
        "side_focus": "long",
    },
    "s20_long_s14_short_s12": {
        "long_selector": "s14_cell_prior_fullsl_s12",
        "fallback_selector": "s12_ev_clean_strong_risk",
        "side_focus": "long",
    },
    "s21_long_s18_short_s12": {
        "long_selector": "s18_long_cell_prior_ev_fullsl_s12",
        "fallback_selector": "s12_ev_clean_strong_risk",
        "side_focus": "long",
    },
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _load_feature_columns(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    cols = payload.get("feature_columns", [])
    if not isinstance(cols, list):
        raise ValueError("feature manifest does not contain list feature_columns")
    return [str(c) for c in cols]


def _usable_fit_features(frame: pd.DataFrame, idx: pd.Index, features: list[str]) -> tuple[list[str], int, int]:
    if not features or len(idx) == 0:
        return [], 0, 0
    values = frame.loc[idx, features]
    observed = values.notna().any(axis=0)
    # HistGradientBoosting can technically consume constant columns, but they
    # add fit cost and no split information.  Dropping them also makes fold
    # feature availability explicit in the fit-events artifact.
    varying = values.nunique(dropna=True) > 1
    usable = [str(col) for col in features if bool(observed.get(col, False)) and bool(varying.get(col, False))]
    all_null = int((~observed).sum())
    constant = int((observed & ~varying).sum())
    return usable, all_null, constant


def _fit_month_forward_heads(
    frame: pd.DataFrame,
    features: list[str],
    *,
    max_fit_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    preds = pd.DataFrame(index=frame.index)
    for name in TARGETS:
        preds[f"pred_{name}"] = np.nan
    months = sorted(frame["month"].dropna().astype(str).unique().tolist())
    rng = np.random.default_rng(seed)
    events: list[dict[str, Any]] = []
    for month in months[1:]:
        train_idx = frame.index[frame["month"].astype(str) < month]
        val_idx = frame.index[frame["month"].astype(str).eq(month)]
        if len(val_idx) == 0:
            continue
        if len(train_idx) > max_fit_rows:
            train_idx = pd.Index(rng.choice(train_idx.to_numpy(), size=max_fit_rows, replace=False))
        for target_name, target_col in TARGETS.items():
            y = pd.to_numeric(frame.loc[train_idx, target_col], errors="coerce")
            valid_idx = train_idx[y.notna().to_numpy()]
            y_valid = pd.to_numeric(frame.loc[valid_idx, target_col], errors="coerce")
            fit_features, all_null_features, constant_features = _usable_fit_features(frame, valid_idx, features)
            if len(valid_idx) < 1000 or y_valid.nunique(dropna=True) < 2 or not fit_features:
                events.append(
                    {
                        "month": month,
                        "target": target_name,
                        "status": "skipped",
                        "train_rows": int(len(valid_idx)),
                        "validation_rows": int(len(val_idx)),
                        "requested_feature_count": int(len(features)),
                        "feature_count": int(len(fit_features)),
                        "all_null_feature_count": int(all_null_features),
                        "constant_feature_count": int(constant_features),
                    }
                )
                continue
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                HistGradientBoostingRegressor(
                    max_iter=128,
                    learning_rate=0.035,
                    max_leaf_nodes=15,
                    l2_regularization=2.0,
                    min_samples_leaf=120,
                    random_state=seed + len(events),
                ),
            )
            model.fit(frame.loc[valid_idx, fit_features], y_valid.astype(float))
            pred = model.predict(frame.loc[val_idx, fit_features])
            if target_name != "ev":
                pred = np.clip(pred, 0.0, 1.0)
            preds.loc[val_idx, f"pred_{target_name}"] = pred.astype("float32")
            events.append(
                {
                    "month": month,
                    "target": target_name,
                    "status": "fit",
                    "train_rows": int(len(valid_idx)),
                    "validation_rows": int(len(val_idx)),
                    "requested_feature_count": int(len(features)),
                    "feature_count": int(len(fit_features)),
                    "all_null_feature_count": int(all_null_features),
                    "constant_feature_count": int(constant_features),
                }
            )
    return preds, events


def _add_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
    for score_name, spec in SELECTOR_SPECS.items():
        weights = spec["weights"]
        score = pd.Series(0.0, index=out.index, dtype="float64")
        for head, weight in weights.items():
            score = score + float(weight) * pd.to_numeric(out[f"pred_{head}"], errors="coerce")
        out[f"score_{score_name}"] = score.astype("float32")
    return out


def _prior_topk_cell_metrics(
    frame: pd.DataFrame,
    *,
    score_col: str,
    month: str,
    top_frac: float,
    min_rows: int = 50,
) -> tuple[pd.DataFrame, dict[str, float]]:
    prior = frame[frame["month"].astype(str) < str(month)].copy()
    prior = prior[pd.to_numeric(prior[score_col], errors="coerce").notna()]
    empty_cols = [
        "side_name",
        "source_archetype",
        "prior_cell_rows",
        "prior_cell_selected_rows",
        "prior_cell_full_sl_rate",
        "prior_cell_timeout_rate",
        "prior_cell_clean_rate",
        "prior_cell_mean_ev",
    ]
    if prior.empty:
        return pd.DataFrame(columns=empty_cols), {
            "prior_global_full_sl_rate": float("nan"),
            "prior_global_timeout_rate": float("nan"),
            "prior_global_clean_rate": float("nan"),
            "prior_global_mean_ev": float("nan"),
        }
    selections: list[pd.DataFrame] = []
    for _, grp in prior.groupby(["month", "side_name", "source_archetype"], dropna=False):
        if len(grp) < int(min_rows):
            continue
        n = max(1, int(math.ceil(len(grp) * float(top_frac))))
        selected = grp.assign(_score=pd.to_numeric(grp[score_col], errors="coerce")).sort_values(
            "_score", ascending=False
        ).head(n)
        selections.append(selected)
    if not selections:
        return pd.DataFrame(columns=empty_cols), {
            "prior_global_full_sl_rate": float("nan"),
            "prior_global_timeout_rate": float("nan"),
            "prior_global_clean_rate": float("nan"),
            "prior_global_mean_ev": float("nan"),
        }
    selected_prior = pd.concat(selections, ignore_index=False)
    global_metrics = {
        "prior_global_full_sl_rate": float(pd.to_numeric(selected_prior["full_sl"], errors="coerce").mean()),
        "prior_global_timeout_rate": float(pd.to_numeric(selected_prior["timeout"], errors="coerce").mean()),
        "prior_global_clean_rate": float(pd.to_numeric(selected_prior["clean_exec_proxy"], errors="coerce").mean()),
        "prior_global_mean_ev": float(
            pd.to_numeric(selected_prior["exec_ev_after_1pct_cost"], errors="coerce").mean()
        ),
    }
    cell = (
        selected_prior.groupby(["side_name", "source_archetype"], dropna=False)
        .agg(
            prior_cell_rows=("month", "size"),
            prior_cell_selected_rows=("month", "size"),
            prior_cell_full_sl_rate=("full_sl", "mean"),
            prior_cell_timeout_rate=("timeout", "mean"),
            prior_cell_clean_rate=("clean_exec_proxy", "mean"),
            prior_cell_mean_ev=("exec_ev_after_1pct_cost", "mean"),
        )
        .reset_index()
    )
    return cell, global_metrics


def _add_cell_aware_scores(predictions: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    out = predictions.copy()
    months = sorted(out["month"].dropna().astype(str).unique().tolist())
    events: list[dict[str, Any]] = []
    for selector_name in CELL_AWARE_SELECTOR_SPECS:
        out[f"score_{selector_name}"] = np.nan
    prior_cols = [
        "prior_cell_full_sl_rate",
        "prior_cell_timeout_rate",
        "prior_cell_clean_rate",
        "prior_cell_mean_ev",
        "prior_global_full_sl_rate",
        "prior_global_timeout_rate",
        "prior_global_clean_rate",
        "prior_global_mean_ev",
        "prior_cell_excess_full_sl",
        "prior_cell_excess_timeout",
        "prior_cell_clean_shortfall",
        "prior_cell_ev_shortfall",
        "prior_cell_ev_premium",
    ]
    for col in prior_cols:
        out[col] = np.nan
    for selector_name, spec in CELL_AWARE_SELECTOR_SPECS.items():
        base_score_col = f"score_{spec['base_selector']}"
        if base_score_col not in out.columns:
            events.append({"selector": selector_name, "status": "skipped_missing_base", "base": spec["base_selector"]})
            continue
        for month in months[1:]:
            val_idx = out.index[out["month"].astype(str).eq(month)]
            if len(val_idx) == 0:
                continue
            cell_metrics, global_metrics = _prior_topk_cell_metrics(
                out,
                score_col=base_score_col,
                month=month,
                top_frac=float(spec.get("top_frac", 0.10)),
            )
            if cell_metrics.empty or not np.isfinite(global_metrics["prior_global_full_sl_rate"]):
                events.append(
                    {
                        "selector": selector_name,
                        "month": month,
                        "status": "skipped_no_prior_cell_metrics",
                        "validation_rows": int(len(val_idx)),
                    }
                )
                continue
            current = out.loc[val_idx, ["side_name", "source_archetype"]].reset_index()
            current = current.merge(cell_metrics, on=["side_name", "source_archetype"], how="left")
            for metric, global_value in global_metrics.items():
                current[metric] = float(global_value)
            current["prior_cell_full_sl_rate"] = current["prior_cell_full_sl_rate"].fillna(
                global_metrics["prior_global_full_sl_rate"]
            )
            current["prior_cell_timeout_rate"] = current["prior_cell_timeout_rate"].fillna(
                global_metrics["prior_global_timeout_rate"]
            )
            current["prior_cell_clean_rate"] = current["prior_cell_clean_rate"].fillna(
                global_metrics["prior_global_clean_rate"]
            )
            current["prior_cell_mean_ev"] = current["prior_cell_mean_ev"].fillna(
                global_metrics["prior_global_mean_ev"]
            )
            current["prior_cell_excess_full_sl"] = (
                current["prior_cell_full_sl_rate"] - global_metrics["prior_global_full_sl_rate"]
            ).clip(lower=0.0)
            current["prior_cell_excess_timeout"] = (
                current["prior_cell_timeout_rate"] - global_metrics["prior_global_timeout_rate"]
            ).clip(lower=0.0)
            current["prior_cell_clean_shortfall"] = (
                global_metrics["prior_global_clean_rate"] - current["prior_cell_clean_rate"]
            ).clip(lower=0.0)
            current["prior_cell_ev_shortfall"] = (
                global_metrics["prior_global_mean_ev"] - current["prior_cell_mean_ev"]
            ).clip(lower=0.0)
            current["prior_cell_ev_premium"] = (
                current["prior_cell_mean_ev"] - global_metrics["prior_global_mean_ev"]
            ).clip(lower=0.0)
            current = current.set_index("index")
            # These prior-cell columns are selector-independent for a given base selector;
            # repeated assignment is intentional and deterministic.
            out.loc[current.index, prior_cols] = current[prior_cols].to_numpy(dtype="float64")
            base_score = pd.to_numeric(out.loc[current.index, base_score_col], errors="coerce")
            score = base_score.copy()
            if spec.get("side_focus"):
                active_adjustment = current["side_name"].astype(str).eq(str(spec["side_focus"])).astype("float64")
            else:
                active_adjustment = pd.Series(1.0, index=current.index, dtype="float64")
            score = score - float(spec["full_sl_scale"]) * current["prior_cell_excess_full_sl"] * pd.to_numeric(
                out.loc[current.index, "pred_full_sl"], errors="coerce"
            ) * active_adjustment
            score = score - float(spec["timeout_scale"]) * current["prior_cell_excess_timeout"] * pd.to_numeric(
                out.loc[current.index, "pred_timeout"], errors="coerce"
            ) * active_adjustment
            score = score + float(spec["clean_scale"]) * current["prior_cell_clean_shortfall"] * pd.to_numeric(
                out.loc[current.index, "pred_clean"], errors="coerce"
            ) * active_adjustment
            score = score - float(spec.get("ev_shortfall_scale", 0.0)) * current["prior_cell_ev_shortfall"] * (
                1.0 + pd.to_numeric(out.loc[current.index, "pred_full_sl"], errors="coerce").fillna(0.0)
            ) * active_adjustment
            score = score + float(spec.get("ev_premium_scale", 0.0)) * current["prior_cell_ev_premium"] * (
                1.0 + pd.to_numeric(out.loc[current.index, "pred_clean"], errors="coerce").fillna(0.0)
            ) * active_adjustment
            out.loc[current.index, f"score_{selector_name}"] = score.astype("float32")
            events.append(
                {
                    "selector": selector_name,
                    "month": month,
                    "status": "fit_prior_cell_adjustment",
                    "validation_rows": int(len(val_idx)),
                    "prior_cells": int(len(cell_metrics)),
                    "side_focus": spec.get("side_focus", ""),
                    "ev_shortfall_scale": float(spec.get("ev_shortfall_scale", 0.0)),
                    "ev_premium_scale": float(spec.get("ev_premium_scale", 0.0)),
                    **global_metrics,
                }
            )
    return out, events


def _add_composite_scores(predictions: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    out = predictions.copy()
    events: list[dict[str, Any]] = []
    side_values = out["side_name"].astype(str) if "side_name" in out.columns else pd.Series("", index=out.index)
    for selector_name, spec in COMPOSITE_SELECTOR_SPECS.items():
        fallback_col = f"score_{spec['fallback_selector']}"
        long_col = f"score_{spec['long_selector']}"
        if fallback_col not in out.columns or long_col not in out.columns:
            out[f"score_{selector_name}"] = np.nan
            events.append(
                {
                    "selector": selector_name,
                    "status": "skipped_missing_source_score",
                    "fallback_selector": spec["fallback_selector"],
                    "long_selector": spec["long_selector"],
                }
            )
            continue
        fallback = pd.to_numeric(out[fallback_col], errors="coerce")
        focused = pd.to_numeric(out[long_col], errors="coerce")
        use_focused = side_values.eq(str(spec.get("side_focus", "long"))) & focused.notna()
        score = fallback.where(~use_focused, focused)
        out[f"score_{selector_name}"] = score.astype("float32")
        events.append(
            {
                "selector": selector_name,
                "status": "fit_composite_score",
                "fallback_selector": spec["fallback_selector"],
                "long_selector": spec["long_selector"],
                "side_focus": spec.get("side_focus", "long"),
                "focused_rows": int(use_focused.sum()),
                "fallback_rows": int((~use_focused & fallback.notna()).sum()),
            }
        )
    return out, events


def _apply_prediction_guards(grp: pd.DataFrame, guards: dict[str, float]) -> pd.DataFrame:
    if not guards:
        return grp
    mask = pd.Series(True, index=grp.index)
    for pred_col, quantile in guards.items():
        values = pd.to_numeric(grp[pred_col], errors="coerce")
        threshold = values.quantile(float(quantile))
        if not np.isfinite(float(threshold)):
            mask &= False
        else:
            mask &= values <= float(threshold)
    return grp[mask]


def _empty_topk_record(
    *,
    group_cols: list[str],
    keys: tuple[Any, ...],
    selector: str,
    frac: float,
    rows: int,
    eligible_rows: int,
) -> dict[str, Any]:
    rec = {col: key for col, key in zip(group_cols, keys)}
    rec.update(
        {
            "selector": selector,
            "top_frac": float(frac),
            "rows": int(rows),
            "eligible_rows": int(eligible_rows),
            "guard_pass_rate": float(eligible_rows / rows) if rows > 0 else float("nan"),
            "selected_rows": 0,
            "selected_share": 0.0,
            "precision_positive_ev": float("nan"),
            "ev_weighted_precision": float("nan"),
            "mean_ev_after_1pct": float("nan"),
            "sum_ev_after_1pct": 0.0,
            "full_sl_rate": float("nan"),
            "timeout_rate": float("nan"),
            "clean_exec_proxy_rate": float("nan"),
            "pred_ev_mean": float("nan"),
            "pred_full_sl_mean": float("nan"),
            "pred_timeout_mean": float("nan"),
            "pred_clean_mean": float("nan"),
        }
    )
    return rec


def _topk_metrics(
    frame: pd.DataFrame,
    *,
    selector_name: str,
    score_col: str,
    guards: dict[str, float],
    group_cols: list[str],
    min_group_rows: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    valid = frame[pd.to_numeric(frame[score_col], errors="coerce").notna()].copy()
    valid[score_col] = pd.to_numeric(valid[score_col], errors="coerce")
    for keys, grp in valid.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        if len(grp) < int(min_group_rows):
            continue
        eligible = _apply_prediction_guards(grp, guards)
        ordered = eligible.sort_values(score_col, ascending=False)
        for frac in TOP_FRACS:
            n = max(1, int(math.ceil(len(ordered) * frac)))
            target_n = max(1, int(math.ceil(len(grp) * frac)))
            if ordered.empty:
                rows.append(
                    _empty_topk_record(
                        group_cols=group_cols,
                        keys=keys,
                        selector=selector_name,
                        frac=frac,
                        rows=len(grp),
                        eligible_rows=len(eligible),
                    )
                )
                continue
            sel = ordered.head(min(target_n, len(ordered)))
            ev = pd.to_numeric(sel["exec_ev_after_1pct_cost"], errors="coerce")
            abs_ev = float(ev.abs().sum())
            rec = {col: key for col, key in zip(group_cols, keys)}
            rec.update(
                {
                    "selector": selector_name,
                    "top_frac": float(frac),
                    "rows": int(len(grp)),
                    "eligible_rows": int(len(eligible)),
                    "guard_pass_rate": float(len(eligible) / len(grp)) if len(grp) > 0 else float("nan"),
                    "selected_rows": int(len(sel)),
                    "selected_share": float(len(sel) / len(grp)) if len(grp) > 0 else float("nan"),
                    "precision_positive_ev": float((ev > 0).mean()),
                    "ev_weighted_precision": float(ev.clip(lower=0).sum() / abs_ev) if abs_ev > 0 else float("nan"),
                    "mean_ev_after_1pct": float(ev.mean()),
                    "sum_ev_after_1pct": float(ev.sum()),
                    "full_sl_rate": float(pd.to_numeric(sel["full_sl"], errors="coerce").mean()),
                    "timeout_rate": float(pd.to_numeric(sel["timeout"], errors="coerce").mean()),
                    "clean_exec_proxy_rate": float(pd.to_numeric(sel["clean_exec_proxy"], errors="coerce").mean()),
                    "pred_ev_mean": float(pd.to_numeric(sel["pred_ev"], errors="coerce").mean()),
                    "pred_full_sl_mean": float(pd.to_numeric(sel["pred_full_sl"], errors="coerce").mean()),
                    "pred_timeout_mean": float(pd.to_numeric(sel["pred_timeout"], errors="coerce").mean()),
                    "pred_clean_mean": float(pd.to_numeric(sel["pred_clean"], errors="coerce").mean()),
                }
            )
            rows.append(rec)
    return pd.DataFrame(rows)


def _delta_vs_ev_only(metrics: pd.DataFrame, *, key_cols: list[str]) -> pd.DataFrame:
    base = metrics[metrics["selector"].eq("s0_ev_only")]
    rows: list[dict[str, Any]] = []
    for _, cur in metrics[~metrics["selector"].eq("s0_ev_only")].iterrows():
        mask = pd.Series(True, index=base.index)
        for col in key_cols:
            mask &= base[col].eq(cur[col])
        if not mask.any():
            continue
        ref = base[mask].iloc[0]
        rec = {col: cur[col] for col in key_cols}
        rec["selector"] = cur["selector"]
        rec["rows"] = int(cur["rows"])
        rec["selected_rows"] = int(cur["selected_rows"])
        for metric in (
            "precision_positive_ev",
            "ev_weighted_precision",
            "mean_ev_after_1pct",
            "full_sl_rate",
            "timeout_rate",
            "clean_exec_proxy_rate",
        ):
            rec[metric] = float(cur[metric])
            rec[f"ev_only_{metric}"] = float(ref[metric])
            rec[f"delta_{metric}"] = float(cur[metric] - ref[metric])
        rows.append(rec)
    return pd.DataFrame(rows)


def _summarize_cell_deltas(cell_delta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cell_delta.empty:
        return pd.DataFrame(), pd.DataFrame()
    top10 = cell_delta[cell_delta["top_frac"].eq(0.10)].copy()
    if top10.empty:
        return pd.DataFrame(), pd.DataFrame()
    summary = top10.groupby("selector", as_index=False).agg(
        cells=("selector", "size"),
        better_ev=("delta_mean_ev_after_1pct", lambda s: int((s > 0).sum())),
        better_precision=("delta_precision_positive_ev", lambda s: int((s > 0).sum())),
        lower_full_sl=("delta_full_sl_rate", lambda s: int((s < 0).sum())),
        lower_timeout=("delta_timeout_rate", lambda s: int((s < 0).sum())),
        mean_delta_ev=("delta_mean_ev_after_1pct", "mean"),
        mean_delta_precision=("delta_precision_positive_ev", "mean"),
        mean_delta_full_sl=("delta_full_sl_rate", "mean"),
        mean_delta_timeout=("delta_timeout_rate", "mean"),
        mean_delta_clean=("delta_clean_exec_proxy_rate", "mean"),
    )
    worst_cols = [
        "selector",
        "month",
        "side_name",
        "source_archetype",
        "rows",
        "selected_rows",
        "delta_mean_ev_after_1pct",
        "delta_precision_positive_ev",
        "delta_full_sl_rate",
        "delta_timeout_rate",
        "mean_ev_after_1pct",
        "full_sl_rate",
        "timeout_rate",
        "clean_exec_proxy_rate",
    ]
    worst = top10.sort_values(["delta_full_sl_rate", "delta_mean_ev_after_1pct"], ascending=[False, True])
    worst = worst[[col for col in worst_cols if col in worst.columns]].head(80).reset_index(drop=True)
    return summary, worst


def _write_report(
    path: Path,
    manifest: dict[str, Any],
    aggregate: pd.DataFrame,
    deltas: pd.DataFrame,
    cell_delta_summary: pd.DataFrame,
) -> None:
    summary = aggregate.groupby(["selector", "top_frac"], as_index=False).agg(
        months=("month", "nunique"),
        selected_share=("selected_share", "mean"),
        guard_pass_rate=("guard_pass_rate", "mean"),
        precision_positive_ev=("precision_positive_ev", "mean"),
        ev_weighted_precision=("ev_weighted_precision", "mean"),
        mean_ev_after_1pct=("mean_ev_after_1pct", "mean"),
        full_sl_rate=("full_sl_rate", "mean"),
        timeout_rate=("timeout_rate", "mean"),
        clean_exec_proxy_rate=("clean_exec_proxy_rate", "mean"),
    )
    top10 = deltas[deltas["top_frac"].eq(0.10)].copy() if not deltas.empty else pd.DataFrame()
    if not top10.empty:
        delta_summary = top10.groupby("selector", as_index=False).agg(
            months=("month", "nunique"),
            mean_delta_ev=("delta_mean_ev_after_1pct", "mean"),
            mean_delta_precision=("delta_precision_positive_ev", "mean"),
            mean_delta_full_sl=("delta_full_sl_rate", "mean"),
            mean_delta_timeout=("delta_timeout_rate", "mean"),
            mean_delta_clean=("delta_clean_exec_proxy_rate", "mean"),
        )
    else:
        delta_summary = pd.DataFrame()
    lines = [
        "# Direct Context Risk-Aware Train Meta Smoke",
        "",
        "## Status",
        "",
        f"- Rows: `{manifest['rows']}`",
        f"- Feature count: `{manifest['feature_count']}`",
        f"- Selectors: `{', '.join(manifest['selectors'])}`",
        "- Heads and selectors are fit month-forward on strictly earlier months.",
        "- Metrics are top-k precision/EV/path-quality; AUC is intentionally not used.",
        "",
        "## Aggregate Top-k",
        "",
        summary.to_markdown(index=False) if not summary.empty else "No aggregate metrics.",
        "",
        "## Top10 Delta vs EV-Only",
        "",
        delta_summary.to_markdown(index=False) if not delta_summary.empty else "No delta rows.",
        "",
        "## Top10 Side x Archetype Cell Delta Coverage",
        "",
        cell_delta_summary.to_markdown(index=False) if not cell_delta_summary.empty else "No cell delta rows.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    *,
    handoff_path: Path,
    feature_manifest_path: Path,
    output_dir: Path,
    max_fit_rows: int,
    min_group_rows: int,
    seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(handoff_path)
    feature_columns = [col for col in _load_feature_columns(feature_manifest_path) if col in frame.columns]
    missing_targets = sorted(set(TARGETS.values()).difference(frame.columns))
    if missing_targets:
        raise ValueError(f"handoff missing target columns: {missing_targets}")
    preds, events = _fit_month_forward_heads(frame, feature_columns, max_fit_rows=max_fit_rows, seed=seed)
    keep_cols = [
        "__ts__",
        "__symbol__",
        "month",
        "side_name",
        "source_archetype",
        "exec_ev_after_1pct_cost",
        "full_sl",
        "timeout",
        "clean_exec_proxy",
    ]
    predictions = pd.concat([frame[[c for c in keep_cols if c in frame.columns]].copy(), preds], axis=1)
    predictions = _add_scores(predictions)
    predictions, cell_score_events = _add_cell_aware_scores(predictions)
    predictions, composite_score_events = _add_composite_scores(predictions)
    selector_specs = {
        **SELECTOR_SPECS,
        **{name: {"weights": {}, "guards": {}, **spec} for name, spec in CELL_AWARE_SELECTOR_SPECS.items()},
        **{name: {"weights": {}, "guards": {}, **spec} for name, spec in COMPOSITE_SELECTOR_SPECS.items()},
    }
    aggregate = pd.concat(
        [
            _topk_metrics(
                predictions,
                selector_name=selector_name,
                score_col=f"score_{selector_name}",
                guards=dict(spec.get("guards", {})),
                group_cols=["month"],
                min_group_rows=min_group_rows,
            )
            for selector_name, spec in selector_specs.items()
        ],
        ignore_index=True,
    )
    cell_metrics = pd.concat(
        [
            _topk_metrics(
                predictions,
                selector_name=selector_name,
                score_col=f"score_{selector_name}",
                guards=dict(spec.get("guards", {})),
                group_cols=["month", "side_name", "source_archetype"],
                min_group_rows=min_group_rows,
            )
            for selector_name, spec in selector_specs.items()
        ],
        ignore_index=True,
    )
    aggregate_delta = _delta_vs_ev_only(aggregate, key_cols=["month", "top_frac"])
    cell_delta = _delta_vs_ev_only(cell_metrics, key_cols=["month", "side_name", "source_archetype", "top_frac"])
    cell_delta_summary, worst_cell_tradeoffs = _summarize_cell_deltas(cell_delta)
    outputs = {
        "predictions": output_dir / "risk_aware_train_meta_predictions.parquet",
        "aggregate": output_dir / "risk_aware_train_meta_aggregate.csv",
        "cell_metrics": output_dir / "risk_aware_train_meta_by_cell.csv",
        "aggregate_delta": output_dir / "risk_aware_train_meta_aggregate_delta.csv",
        "cell_delta": output_dir / "risk_aware_train_meta_cell_delta.csv",
        "cell_delta_summary": output_dir / "risk_aware_train_meta_cell_delta_summary.csv",
        "worst_cell_tradeoffs": output_dir / "risk_aware_train_meta_worst_cell_tradeoffs.csv",
        "fit_events": output_dir / "risk_aware_train_meta_fit_events.csv",
        "cell_score_events": output_dir / "risk_aware_train_meta_cell_score_events.csv",
        "manifest": output_dir / "manifest.json",
        "report": output_dir / "risk_aware_train_meta_smoke.md",
    }
    predictions.to_parquet(outputs["predictions"], index=False)
    aggregate.to_csv(outputs["aggregate"], index=False)
    cell_metrics.to_csv(outputs["cell_metrics"], index=False)
    aggregate_delta.to_csv(outputs["aggregate_delta"], index=False)
    cell_delta.to_csv(outputs["cell_delta"], index=False)
    cell_delta_summary.to_csv(outputs["cell_delta_summary"], index=False)
    worst_cell_tradeoffs.to_csv(outputs["worst_cell_tradeoffs"], index=False)
    pd.DataFrame(events).to_csv(outputs["fit_events"], index=False)
    pd.DataFrame(cell_score_events + composite_score_events).to_csv(outputs["cell_score_events"], index=False)
    manifest = {
        "scope": "direct_context_risk_aware_train_meta_smoke",
        "handoff_path": str(handoff_path),
        "feature_manifest_path": str(feature_manifest_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "feature_count": int(len(feature_columns)),
        "targets": TARGETS,
        "selectors": list(selector_specs.keys()),
        "selector_specs": selector_specs,
        "cell_aware_selector_specs": CELL_AWARE_SELECTOR_SPECS,
        "composite_selector_specs": COMPOSITE_SELECTOR_SPECS,
        "leakage_contract": (
            "all heads fit month-forward on strictly earlier months; cell-aware selectors use only "
            "prior-month side x archetype selected-row outcomes; composite selectors switch between live "
            "meta scores by side using no future outcomes; accepted-cell metadata is not used as input"
        ),
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    outputs["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    _write_report(outputs["report"], manifest, aggregate, aggregate_delta, cell_delta_summary)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-path", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--feature-manifest-path", type=Path, default=DEFAULT_FEATURE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-fit-rows", type=int, default=80_000)
    parser.add_argument("--min-group-rows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=101)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run(
        handoff_path=args.handoff_path,
        feature_manifest_path=args.feature_manifest_path,
        output_dir=args.output_dir,
        max_fit_rows=int(args.max_fit_rows),
        min_group_rows=int(args.min_group_rows),
        seed=int(args.seed),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
