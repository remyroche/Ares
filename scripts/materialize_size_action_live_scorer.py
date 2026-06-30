#!/usr/bin/env python3
"""Materialize a fail-closed live scorer bundle for size-action components."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_exact_state_size_action_learning import _json_safe, _lgbm_deterministic_kwargs  # noqa: E402


DEFAULT_ARM = "C3fo_bagged_safety_c3ed_or_calibrated_group_hurdle_positive_value_acceptance_union_gate"
SCORER_COMPONENT = "calibrated_hurdle_full_arm"
HEAD_PREFIXES = ("long_bars", "long_dist", "short_asset", "short_boll")
HEAD_ALIASES = {"short_bollinger": "short_boll"}
DECISION_THRESHOLD_KEYS = {
    "p_intervene_min",
    "action_selector_min",
    "positive_value_min",
    "pred_delta_J_min",
}
LEAKAGE_COLUMNS = {
    "best_multiplier",
    "best_gain",
    "best_margin",
    "best_gain_per_notional",
    "best_margin_per_notional",
    "group_affected_notional",
    "best_immediate_gain",
    "best_capacity_gain",
    "best_immediate_gain_per_notional",
    "best_capacity_gain_per_notional",
    "best_nonbaseline_gain",
    "worst_nonbaseline_gain",
    "best_nonbaseline_multiplier",
    "group_can_bind",
    "y_intervene",
    "group_best_projected_notional_removed_to_remaining_capital",
    "group_best_projected_removed_trade_share_timestamp",
    "group_best_projected_removed_trade_share_strategy",
    "group_best_projected_notional_removed_to_open_notional",
    "delta_full_J",
    "delta_immediate_J",
    "delta_full_net_pnl",
    "delta_full_cost_pnl",
    "delta_full_turnover",
    "delta_full_J_per_notional",
    "delta_immediate_J_per_notional",
    "zero_cut_target",
    "zero_cut_trainable",
    "action_positive",
    "action_economic_positive",
    "rank_relevance",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _resolve_run_dir(freeze_manifest: dict[str, Any], freeze_manifest_path: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    raw = freeze_manifest.get("run_dir")
    if not raw:
        raise ValueError("freeze manifest does not contain run_dir; pass --run-dir")
    path = Path(str(raw))
    if path.is_absolute():
        return path
    cwd_relative = Path.cwd() / path
    if cwd_relative.exists():
        return cwd_relative
    return (freeze_manifest_path.parent / path).resolve()


def _selected_features_by_model(run_dir: Path, panel_columns: set[str], max_features: int) -> dict[str, list[str]]:
    path = run_dir / "size_action_selected_features.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if not {"model", "feature"}.issubset(frame.columns):
        return {}
    out: dict[str, list[str]] = {}
    for model_name, group in frame.sort_values(["model", "rank"]).groupby("model"):
        features: list[str] = []
        for raw in group["feature"].dropna().astype(str):
            if raw in LEAKAGE_COLUMNS:
                continue
            if raw == "strategy_code" or raw in panel_columns:
                if raw not in features:
                    features.append(raw)
            if len(features) >= int(max_features):
                break
        out[str(model_name)] = features
    return out


def _fallback_feature_columns(frame: pd.DataFrame, max_features: int) -> list[str]:
    features: list[str] = []
    for col in frame.columns:
        if col in LEAKAGE_COLUMNS or col in {"timestamp", "strategy_id", "split", "fold_id"}:
            continue
        if col == "strategy_code" or pd.api.types.is_numeric_dtype(frame[col]):
            features.append(str(col))
        if len(features) >= int(max_features):
            break
    return features


def _prepare_rows(frame: pd.DataFrame, strategy_map: dict[str, int]) -> pd.DataFrame:
    out = frame.copy()
    out["strategy_code"] = out.get("strategy_id", "").astype(str).map(strategy_map).fillna(-1).astype(float)
    return out


def _strategy_head(strategy_id: Any) -> str:
    raw = str(strategy_id)
    for alias, head in HEAD_ALIASES.items():
        if raw == alias or raw.startswith(f"{alias}_"):
            return head
    for prefix in HEAD_PREFIXES:
        if raw == prefix or raw.startswith(f"{prefix}_"):
            return prefix
    return "unknown"


def _required_inputs_for_features(feature_cols: list[str]) -> list[str]:
    return sorted(set(feature_cols).difference({"strategy_code"}) | {"timestamp", "strategy_id", "multiplier"})


def _parse_head_decision_thresholds(raw: str | None) -> dict[str, dict[str, float]]:
    """Parse per-head live decision thresholds.

    Accepted forms:

    * JSON object, e.g. ``{"short_asset": {"p_intervene_min": 0.8}}``.
    * Compact CLI form:
      ``short_asset:p_intervene_min=0.8,pred_delta_J_min=320;short_boll:p_intervene_min=0.2``.
    """
    text = str(raw or "").strip()
    if not text:
        return {}
    if text.startswith("{"):
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("--head-decision-thresholds JSON must be an object")
        parsed: dict[str, dict[str, float]] = {}
        for head, values in payload.items():
            if str(head) not in HEAD_PREFIXES:
                raise ValueError(f"Unknown head in decision thresholds: {head}")
            if not isinstance(values, dict):
                raise ValueError(f"Decision thresholds for {head} must be an object")
            parsed[str(head)] = {}
            for key, value in values.items():
                if str(key) not in DECISION_THRESHOLD_KEYS:
                    raise ValueError(f"Unknown decision threshold key for {head}: {key}")
                parsed[str(head)][str(key)] = float(value)
        return parsed
    parsed: dict[str, dict[str, float]] = {}
    for head_block in text.split(";"):
        block = head_block.strip()
        if not block:
            continue
        if ":" not in block:
            raise ValueError(f"Expected head:key=value block, got: {block}")
        head, assignments = block.split(":", 1)
        head = head.strip()
        if head not in HEAD_PREFIXES:
            raise ValueError(f"Unknown head in decision thresholds: {head}")
        parsed.setdefault(head, {})
        for assignment in assignments.split(","):
            token = assignment.strip()
            if not token:
                continue
            if "=" not in token:
                raise ValueError(f"Expected key=value threshold token, got: {token}")
            key, value = token.split("=", 1)
            key = key.strip()
            if key not in DECISION_THRESHOLD_KEYS:
                raise ValueError(f"Unknown decision threshold key for {head}: {key}")
            parsed[head][key] = float(value)
    return parsed


def _serialise_medians(medians: dict[str, pd.Series]) -> dict[str, dict[str, float]]:
    return {
        str(component): {str(k): float(v) for k, v in median_series.items()}
        for component, median_series in medians.items()
    }


def _matrix(frame: pd.DataFrame, feature_cols: list[str], medians: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.DataFrame(index=frame.index)
    for col in feature_cols:
        if col in frame.columns:
            data[col] = pd.to_numeric(frame[col], errors="coerce")
        else:
            data[col] = np.nan
    if medians is None:
        medians = data.replace([np.inf, -np.inf], np.nan).median(axis=0).fillna(0.0)
    data = data.replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)
    return data.astype(float), medians.astype(float)


def _validate_required_inputs(frame: pd.DataFrame, required_columns: list[str]) -> tuple[bool, list[str]]:
    missing = [col for col in required_columns if col not in frame.columns]
    if missing:
        return False, [f"missing:{col}" for col in missing]
    nonfinite: list[str] = []
    for col in required_columns:
        if col in {"timestamp", "strategy_id"}:
            if frame[col].isna().any() or frame[col].astype(str).str.len().eq(0).any():
                nonfinite.append(col)
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.isna().any() or np.isinf(values.to_numpy(dtype=float, copy=False)).any():
            nonfinite.append(col)
    if nonfinite:
        return False, [f"nonfinite:{col}" for col in nonfinite]
    return True, []


def _predict_component(component: dict[str, Any], frame: pd.DataFrame, medians: dict[str, float], *, proba: bool) -> np.ndarray:
    model = component.get("model")
    if isinstance(model, dict) and "constant" in model:
        return np.full(len(frame), float(model.get("constant", 0.0)), dtype=float)
    features = list(component.get("feature_columns") or [])
    x, _ = _matrix(frame, features, pd.Series(medians, dtype=float))
    if proba and hasattr(model, "predict_proba"):
        pred = np.asarray(model.predict_proba(x), dtype=float)
        if pred.ndim == 2 and pred.shape[1] > 1:
            return pred[:, 1]
        return pred.reshape(-1)
    if hasattr(model, "predict"):
        return np.asarray(model.predict(x), dtype=float).reshape(-1)
    return np.zeros(len(frame), dtype=float)


def _constant_component(kind: str, value: Any, reason: str, rows: int) -> dict[str, Any]:
    return {"kind": kind, "constant": _json_safe(value), "reason": reason, "fit_rows": int(rows)}


def _fit_binary_classifier(
    frame: pd.DataFrame,
    feature_cols: list[str],
    y: pd.Series,
    *,
    seed: int,
    weights: np.ndarray | None = None,
) -> tuple[Any, pd.Series, dict[str, Any]]:
    y_arr = pd.to_numeric(y, errors="coerce").fillna(0).astype(int).to_numpy()
    if len(frame) < 30 or len(np.unique(y_arr)) < 2 or not feature_cols:
        rate = float(np.mean(y_arr)) if len(y_arr) else 0.0
        return _constant_component("binary_classifier", rate, "insufficient_rows_or_single_class", len(frame)), pd.Series(dtype=float), {
            "fit_rows": int(len(frame)),
            "positive_rows": int(y_arr.sum()) if len(y_arr) else 0,
            "constant": True,
            "positive_rate": rate,
        }
    from lightgbm import LGBMClassifier

    x, medians = _matrix(frame, feature_cols)
    pos = max(int(y_arr.sum()), 1)
    neg = max(int(len(y_arr) - y_arr.sum()), 1)
    model = LGBMClassifier(
        objective="binary",
        n_estimators=180,
        learning_rate=0.035,
        num_leaves=15,
        max_depth=4,
        min_child_samples=max(20, int(0.03 * len(frame))),
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=2.0,
        scale_pos_weight=float(max(neg / pos, 1.0)),
        **_lgbm_deterministic_kwargs(seed),
        verbose=-1,
    )
    model.fit(x, y_arr, sample_weight=weights)
    return model, medians, {
        "fit_rows": int(len(frame)),
        "positive_rows": int(y_arr.sum()),
        "constant": False,
        "positive_rate": float(np.mean(y_arr)),
    }


def _fit_regressor(
    frame: pd.DataFrame,
    feature_cols: list[str],
    y: pd.Series,
    *,
    seed: int,
    weights: np.ndarray | None = None,
) -> tuple[Any, pd.Series, dict[str, Any]]:
    y_arr = pd.to_numeric(y, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if len(frame) < 30 or not feature_cols:
        value = float(np.nanmean(y_arr)) if len(y_arr) else 0.0
        return _constant_component("regressor", value, "insufficient_rows", len(frame)), pd.Series(dtype=float), {
            "fit_rows": int(len(frame)),
            "constant": True,
            "target_mean": value,
        }
    from lightgbm import LGBMRegressor

    x, medians = _matrix(frame, feature_cols)
    model = LGBMRegressor(
        objective="regression",
        n_estimators=220,
        learning_rate=0.035,
        num_leaves=15,
        max_depth=4,
        min_child_samples=max(20, int(0.03 * len(frame))),
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=2.0,
        **_lgbm_deterministic_kwargs(seed),
        verbose=-1,
    )
    model.fit(x, y_arr, sample_weight=weights)
    return model, medians, {
        "fit_rows": int(len(frame)),
        "constant": False,
        "target_mean": float(np.nanmean(y_arr)) if len(y_arr) else 0.0,
    }


def _component_threshold_summary(run_dir: Path, arm: str) -> dict[str, Any]:
    path = run_dir / "size_action_gate_thresholds.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if "arm" in frame.columns:
        frame = frame.loc[frame["arm"].astype(str).eq(str(arm))].copy()
    if frame.empty:
        return {}
    keep_cols = [
        c
        for c in frame.columns
        if c in {
            "fold_id",
            "arm",
            "score_source",
            "union_primary_arm",
            "union_secondary_arm",
            "calibrated_group_hurdle_recall_p_intervene_min",
            "calibrated_group_hurdle_recall_stage1_cal_positive_rate_min",
            "calibrated_group_hurdle_recall_stage1_cal_mean_gain_min",
            "calibrated_group_hurdle_recall_p_action_value_positive_min",
            "calibrated_group_hurdle_recall_cal_positive_rate_min",
            "calibrated_group_hurdle_recall_cal_mean_delta_J_min",
            "calibrated_group_hurdle_recall_pred_delta_J_min",
            "calibrated_group_hurdle_recall_top_fraction",
            "secondary_positive_value_floor_cal_mean_delta_J_min",
            "secondary_positive_value_floor_pred_delta_J_min",
            "calibrated_group_hurdle_action_acceptance_thresholds",
        }
    ]
    return {
        "source": str(path),
        "rows": int(len(frame)),
        "columns": keep_cols,
        "records": _json_safe(frame[keep_cols].to_dict(orient="records")),
    }


def _fit_full_arm_scorer(
    panel: pd.DataFrame,
    *,
    run_dir: Path,
    arm: str,
    max_features: int,
    seed: int,
    fit_split: str,
) -> tuple[dict[str, Any], list[str], dict[str, pd.Series], dict[str, int], dict[str, Any]]:
    rows = panel.copy()
    rows["split"] = rows.get("split", "train").astype(str)
    if fit_split != "all":
        rows = rows.loc[rows["split"].eq(fit_split)].copy()
    strategy_categories = sorted(rows.get("strategy_id", pd.Series(dtype=str)).dropna().astype(str).unique())
    strategy_map = {strategy: idx for idx, strategy in enumerate(strategy_categories)}
    rows = _prepare_rows(rows, strategy_map)

    selected = _selected_features_by_model(run_dir, set(rows.columns), int(max_features))
    fallback = _fallback_feature_columns(rows, int(max_features))
    stage1_features = selected.get("stage1_intervention_classifier") or fallback
    action_features = selected.get("action_value_selector") or selected.get("action_ranker_selector") or fallback
    positive_features = selected.get("action_positive_selector") or action_features
    all_features = sorted(set(stage1_features) | set(action_features) | set(positive_features))

    group_cols = ["timestamp", "strategy_id"]
    group_rows = rows.sort_values(group_cols + ["multiplier"]).drop_duplicates(group_cols).copy()
    stage1_y = pd.to_numeric(group_rows.get("y_intervene", 0.0), errors="coerce").fillna(0.0).gt(0.0).astype(int)
    group_gain = pd.to_numeric(group_rows.get("best_gain", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    stage1_weights = np.ones(len(group_rows), dtype=float)
    if len(group_gain):
        scale = max(float(np.nanmedian(np.abs(group_gain[group_gain != 0]))) if np.any(group_gain != 0) else 1.0, 1.0)
        stage1_weights *= 1.0 + 0.5 * np.clip(np.maximum(group_gain, 0.0) / scale, 0.0, 10.0)
        stage1_weights *= 1.0 + 0.75 * np.clip(np.abs(np.minimum(group_gain, 0.0)) / scale, 0.0, 10.0)
    stage1_model, stage1_medians, stage1_diag = _fit_binary_classifier(
        group_rows,
        stage1_features,
        stage1_y,
        seed=int(seed),
        weights=stage1_weights,
    )

    action_rows = rows.loc[
        pd.to_numeric(rows.get("multiplier", 1.0), errors="coerce").fillna(1.0).lt(1.0)
        & pd.to_numeric(rows.get("action_binds", 0.0), errors="coerce").fillna(0.0).gt(0.0)
    ].copy()
    action_choice = (
        pd.to_numeric(action_rows.get("multiplier", 1.0), errors="coerce").round(6)
        .eq(pd.to_numeric(action_rows.get("best_multiplier", 1.0), errors="coerce").round(6))
        & pd.to_numeric(action_rows.get("best_gain", 0.0), errors="coerce").fillna(0.0).gt(0.0)
    ).astype(int)
    action_delta = pd.to_numeric(action_rows.get("delta_full_J", 0.0), errors="coerce").fillna(0.0)
    action_scale = max(float(np.nanmedian(np.abs(action_delta[action_delta != 0]))) if len(action_delta) and np.any(action_delta != 0) else 1.0, 1.0)
    action_weights = 1.0 + np.clip(np.abs(action_delta.to_numpy(dtype=float)) / action_scale, 0.0, 10.0)

    action_selector, action_medians, action_diag = _fit_binary_classifier(
        action_rows,
        action_features,
        action_choice,
        seed=int(seed) + 101,
        weights=action_weights if len(action_rows) else None,
    )
    full_value, full_medians, full_diag = _fit_regressor(
        action_rows,
        action_features,
        action_delta,
        seed=int(seed) + 202,
        weights=action_weights if len(action_rows) else None,
    )
    immediate_target = pd.to_numeric(action_rows.get("delta_immediate_J", 0.0), errors="coerce").fillna(0.0)
    immediate_value, immediate_medians, immediate_diag = _fit_regressor(
        action_rows,
        action_features,
        immediate_target,
        seed=int(seed) + 303,
        weights=action_weights if len(action_rows) else None,
    )
    capacity_value, capacity_medians, capacity_diag = _fit_regressor(
        action_rows,
        action_features,
        action_delta - immediate_target,
        seed=int(seed) + 404,
        weights=action_weights if len(action_rows) else None,
    )
    positive_y = action_delta.gt(0.0).astype(int)
    positive_model, positive_medians, positive_diag = _fit_binary_classifier(
        action_rows,
        positive_features,
        positive_y,
        seed=int(seed) + 505,
        weights=action_weights if len(action_rows) else None,
    )

    components = {
        "stage1_intervention_classifier": {
            "model": stage1_model,
            "feature_columns": stage1_features,
            "median_key": "stage1_intervention_classifier",
        },
        "action_selector": {
            "model": action_selector,
            "feature_columns": action_features,
            "median_key": "action_selector",
        },
        "full_value_regressor": {
            "model": full_value,
            "feature_columns": action_features,
            "median_key": "full_value_regressor",
        },
        "immediate_value_regressor": {
            "model": immediate_value,
            "feature_columns": action_features,
            "median_key": "immediate_value_regressor",
        },
        "capacity_value_regressor": {
            "model": capacity_value,
            "feature_columns": action_features,
            "median_key": "capacity_value_regressor",
        },
        "positive_value_classifier": {
            "model": positive_model,
            "feature_columns": positive_features,
            "median_key": "positive_value_classifier",
        },
    }
    medians = {
        "stage1_intervention_classifier": stage1_medians,
        "action_selector": action_medians,
        "full_value_regressor": full_medians,
        "immediate_value_regressor": immediate_medians,
        "capacity_value_regressor": capacity_medians,
        "positive_value_classifier": positive_medians,
    }
    diagnostics = {
        "component": SCORER_COMPONENT,
        "fit_split": fit_split,
        "fit_rows": int(len(rows)),
        "group_rows": int(len(group_rows)),
        "action_rows": int(len(action_rows)),
        "stage1": stage1_diag,
        "action_selector": action_diag,
        "full_value": full_diag,
        "immediate_value": immediate_diag,
        "capacity_value": capacity_diag,
        "positive_value": positive_diag,
        "feature_counts": {
            "stage1": int(len(stage1_features)),
            "action": int(len(action_features)),
            "positive": int(len(positive_features)),
            "union": int(len(all_features)),
        },
        "thresholds": _component_threshold_summary(run_dir, arm),
    }
    return components, all_features, medians, strategy_map, diagnostics


def _fit_head_specific_scorers(
    panel: pd.DataFrame,
    *,
    run_dir: Path,
    arm: str,
    max_features: int,
    seed: int,
    fit_split: str,
    min_group_rows: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Fit independent size-action component stacks per strategy head.

    C3el decisions are materially head-dependent: a global action-value model can
    learn a broadly plausible risk state while applying it to the wrong head.
    These scoped stacks keep feature reuse and model form unchanged, but isolate
    training targets, medians, strategy-code maps, and diagnostics by head.
    """
    if panel.empty or "strategy_id" not in panel.columns:
        return {}, {"enabled": False, "reason": "empty_panel_or_missing_strategy_id"}

    scoped: dict[str, dict[str, Any]] = {}
    skipped: dict[str, dict[str, Any]] = {}
    head_series = panel["strategy_id"].map(_strategy_head)
    for head in sorted(str(x) for x in head_series.dropna().unique() if str(x) != "unknown"):
        head_panel = panel.loc[head_series.eq(head)].copy()
        fit_rows = head_panel.copy()
        fit_rows["split"] = fit_rows.get("split", "train").astype(str)
        if fit_split != "all":
            fit_rows = fit_rows.loc[fit_rows["split"].eq(fit_split)].copy()
        group_rows = (
            fit_rows[["timestamp", "strategy_id"]]
            .dropna()
            .drop_duplicates()
            .shape[0]
            if {"timestamp", "strategy_id"}.issubset(fit_rows.columns)
            else 0
        )
        if group_rows < int(min_group_rows):
            skipped[head] = {"fit_rows": int(len(fit_rows)), "group_rows": int(group_rows), "reason": "insufficient_group_rows"}
            continue
        components, feature_cols, medians, strategy_map, diagnostics = _fit_full_arm_scorer(
            head_panel,
            run_dir=run_dir,
            arm=arm,
            max_features=int(max_features),
            seed=int(seed) + 1009 * (1 + HEAD_PREFIXES.index(head) if head in HEAD_PREFIXES else 9),
            fit_split=str(fit_split),
        )
        diagnostics = dict(diagnostics)
        diagnostics["scope"] = "head"
        diagnostics["head"] = head
        scoped[head] = {
            "scope": "head",
            "head": head,
            "components": components,
            "feature_columns": list(feature_cols),
            "required_input_columns": _required_inputs_for_features(list(feature_cols)),
            "medians": _serialise_medians(medians),
            "strategy_map": {str(k): int(v) for k, v in strategy_map.items()},
            "diagnostics": _json_safe(diagnostics),
        }

    return scoped, {
        "enabled": True,
        "required": True,
        "min_group_rows": int(min_group_rows),
        "heads_fitted": sorted(scoped),
        "heads_skipped": skipped,
        "unknown_strategy_rows": int(head_series.eq("unknown").sum()),
    }


def materialize_bundle(
    *,
    freeze_manifest_path: Path,
    run_dir: Path | None,
    out_dir: Path,
    arm: str,
    material_gain: float,
    top_fraction: float,
    max_features: int,
    seed: int,
    fit_split: str,
    head_specific: bool = True,
    min_head_group_rows: int = 30,
    head_decision_thresholds: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    freeze_manifest_path = freeze_manifest_path.resolve()
    freeze_manifest = _read_json(freeze_manifest_path)
    resolved_run_dir = _resolve_run_dir(freeze_manifest, freeze_manifest_path, run_dir).resolve()
    panel_path = resolved_run_dir / "size_action_exact_panel.csv"
    if not panel_path.exists():
        raise FileNotFoundError(f"missing exact-state panel: {panel_path}")
    panel = pd.read_csv(panel_path)
    components, feature_cols, medians, strategy_map, diagnostics = _fit_full_arm_scorer(
        panel,
        run_dir=resolved_run_dir,
        arm=arm,
        max_features=int(max_features),
        seed=int(seed),
        fit_split=str(fit_split),
    )
    head_scoped_components: dict[str, dict[str, Any]] = {}
    head_specific_diag: dict[str, Any] = {"enabled": False, "required": False}
    if bool(head_specific):
        head_scoped_components, head_specific_diag = _fit_head_specific_scorers(
            panel,
            run_dir=resolved_run_dir,
            arm=arm,
            max_features=int(max_features),
            seed=int(seed),
            fit_split=str(fit_split),
            min_group_rows=int(min_head_group_rows),
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "size_action_live_scorer.joblib"
    try:
        import joblib
    except Exception as exc:  # pragma: no cover - dependency issue is explicit at runtime.
        raise RuntimeError("joblib is required to materialize the live scorer bundle") from exc
    joblib.dump(
        {
            "component": SCORER_COMPONENT,
            "arm": arm,
            "coverage": "full_arm",
            "components": components,
            "feature_columns": feature_cols,
            "medians": _serialise_medians(medians),
            "strategy_map": strategy_map,
            "head_specific": head_scoped_components,
            "head_specific_required": bool(head_specific),
            "thresholds": diagnostics.get("thresholds", {}),
            "top_fraction": float(top_fraction),
            "fit_split": fit_split,
            "fail_closed": True,
        },
        model_path,
    )

    feature_contract = {
        "component": SCORER_COMPONENT,
        "coverage": "full_arm",
        "required_columns": feature_cols,
        "derived_columns": {"strategy_code": "mapped from strategy_id using training strategy_map"},
        "required_input_columns": _required_inputs_for_features(feature_cols),
        "required_input_columns_by_head": {
            str(head): list(payload.get("required_input_columns") or [])
            for head, payload in head_scoped_components.items()
        },
        "head_feature_columns": {
            str(head): list(payload.get("feature_columns") or [])
            for head, payload in head_scoped_components.items()
        },
        "head_specific_required": bool(head_specific),
        "nonfinite_policy": "reject_or_impute_training_median_then_fail_closed_if_required_input_missing",
        "column_order": feature_cols,
    }
    imputation_contract = {
        "component": SCORER_COMPONENT,
        "medians": _serialise_medians(medians),
        "strategy_map": strategy_map,
        "head_specific_medians": {
            str(head): payload.get("medians", {})
            for head, payload in head_scoped_components.items()
        },
        "head_specific_strategy_maps": {
            str(head): payload.get("strategy_map", {})
            for head, payload in head_scoped_components.items()
        },
        "unknown_strategy_code": -1.0,
    }
    policy_contract = {
        "component": SCORER_COMPONENT,
        "arm": arm,
        "coverage": "full_arm",
        "missing_component_blocker": None,
        "score_mode": "stage1 intervention probability plus non-baseline action selector and action-value safety",
        "selection_rule": (
            "fail closed unless required features are present; apply deterministic binding filter, "
            "score strategy-timestamp intervention probability, select non-baseline multiplier with "
            "action selector/value safety, otherwise return multiplier=1.0"
        ),
        "top_fraction": float(top_fraction),
        "material_gain": float(material_gain),
        "thresholds": diagnostics.get("thresholds", {}),
        "head_specific_enabled": bool(head_specific),
        "head_specific_required": bool(head_specific),
        "head_specific_heads": sorted(head_scoped_components),
        "head_specific_min_group_rows": int(min_head_group_rows),
        "decision_thresholds": {
            "p_intervene_min": 0.50,
            "action_selector_min": 0.50,
            "positive_value_min": 0.50,
            "pred_delta_J_min": 0.0,
        },
        "head_decision_thresholds": {
            str(head): {str(key): float(value) for key, value in thresholds.items()}
            for head, thresholds in (head_decision_thresholds or {}).items()
        },
        "fail_closed": True,
    }
    _write_json(out_dir / "size_action_live_feature_contract.json", feature_contract)
    _write_json(out_dir / "size_action_live_imputation.json", imputation_contract)
    _write_json(out_dir / "size_action_live_policy_contract.json", policy_contract)

    run_manifest = _read_json(resolved_run_dir / "manifest.json")
    scorer_manifest = {
        "generated_by": "materialize_size_action_live_scorer",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "live",
        "coverage": "full_arm",
        "component": SCORER_COMPONENT,
        "missing_components": [],
        "arm": arm,
        "run_dir": str(resolved_run_dir),
        "freeze_manifest": str(freeze_manifest_path),
        "policy_variant": (freeze_manifest.get("source_manifest") or run_manifest).get("policy_variant"),
        "feature_columns": feature_cols,
        "model_artifacts": ["size_action_live_scorer.joblib"],
        "imputation_policy": "training_median_with_fail_closed_required_input_contract",
        "fail_closed": True,
        "score_contract": {
            "unit": "strategy_timestamp_action_group_with_multiplier_rows",
            "default_multiplier": 1.0,
            "missing_required_feature": "reject",
            "nonfinite_score": "reject",
            "output_score": "selected_multiplier_and_component_scores",
        },
        "head_decision_thresholds": {
            str(head): {str(key): float(value) for key, value in thresholds.items()}
            for head, thresholds in (head_decision_thresholds or {}).items()
        },
        "diagnostics": _json_safe(diagnostics),
        "head_specific_diagnostics": _json_safe(head_specific_diag),
        "hashes": {
            "model_bundle": _sha256(model_path),
            "feature_contract": _sha256(out_dir / "size_action_live_feature_contract.json"),
            "imputation_contract": _sha256(out_dir / "size_action_live_imputation.json"),
            "policy_contract": _sha256(out_dir / "size_action_live_policy_contract.json"),
            "source_panel": _sha256(panel_path),
        },
    }
    _write_json(out_dir / "size_action_live_scorer_manifest.json", scorer_manifest)
    scorer_manifest["hashes"]["scorer_manifest"] = _sha256(out_dir / "size_action_live_scorer_manifest.json")
    _write_json(out_dir / "size_action_live_scorer_manifest.json", scorer_manifest)
    return scorer_manifest


def load_live_scorer_bundle(bundle_dir: Path) -> dict[str, Any]:
    """Load a materialized size-action scorer bundle."""
    try:
        import joblib
    except Exception as exc:  # pragma: no cover - dependency issue is explicit at runtime.
        raise RuntimeError("joblib is required to load the live scorer bundle") from exc
    bundle_dir = Path(bundle_dir)
    manifest = _read_json(bundle_dir / "size_action_live_scorer_manifest.json")
    feature_contract = _read_json(bundle_dir / "size_action_live_feature_contract.json")
    policy_contract = _read_json(bundle_dir / "size_action_live_policy_contract.json")
    imputation_contract = _read_json(bundle_dir / "size_action_live_imputation.json")
    model_path = bundle_dir / "size_action_live_scorer.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"missing live scorer model bundle: {model_path}")
    model_bundle = joblib.load(model_path)
    return {
        "manifest": manifest,
        "feature_contract": feature_contract,
        "policy_contract": policy_contract,
        "imputation_contract": imputation_contract,
        "model_bundle": model_bundle,
    }


def score_size_action_frame(bundle_dir: Path, frame: pd.DataFrame) -> pd.DataFrame:
    """Score strategy/timestamp action groups with fail-closed semantics.

    The scorer expects one or more rows per `(timestamp, strategy_id)`, with
    candidate `multiplier` rows when non-baseline actions can bind. If the live
    feature contract is incomplete for a group, the returned action is a no-op
    multiplier of `1.0` with `accepted=False`.
    """
    loaded = load_live_scorer_bundle(bundle_dir)
    model_bundle = loaded["model_bundle"]
    feature_contract = loaded["feature_contract"]
    policy_contract = loaded["policy_contract"]
    if model_bundle.get("coverage") != "full_arm" or loaded["manifest"].get("coverage") != "full_arm":
        raise ValueError("size-action scorer bundle is not full_arm coverage")
    if model_bundle.get("fail_closed") is not True or loaded["manifest"].get("fail_closed") is not True:
        raise ValueError("size-action scorer bundle is not fail-closed")

    required_inputs = list(feature_contract.get("required_input_columns") or [])
    strategy_map = {str(k): int(v) for k, v in (model_bundle.get("strategy_map") or {}).items()}
    global_medians_by_component = model_bundle.get("medians") or {}
    global_components = model_bundle.get("components") or {}
    global_thresholds = policy_contract.get("decision_thresholds") or {}
    head_specific = model_bundle.get("head_specific") or {}
    head_required_inputs = feature_contract.get("required_input_columns_by_head") or {}
    head_specific_required = bool(policy_contract.get("head_specific_required", model_bundle.get("head_specific_required", False)))

    if frame.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "strategy_id",
                "selected_multiplier",
                "accepted",
                "reject_reason",
                "p_intervene",
                "action_selector_score",
                "positive_value_score",
                "pred_delta_J",
                "pred_immediate_J",
                "pred_capacity_J",
                "head",
                "component_scope",
                "head_specific_component",
            ]
        )
    work = frame.copy()
    if "multiplier" not in work.columns:
        work["multiplier"] = 1.0

    results: list[dict[str, Any]] = []
    for (timestamp, strategy_id), group in work.groupby(["timestamp", "strategy_id"], sort=False):
        head = _strategy_head(strategy_id)
        head_bundle = head_specific.get(head) or {}
        if head_bundle:
            components = head_bundle.get("components") or {}
            medians_by_component = head_bundle.get("medians") or {}
            active_strategy_map = {str(k): int(v) for k, v in (head_bundle.get("strategy_map") or {}).items()}
            active_feature_cols = list(head_bundle.get("feature_columns") or [])
            active_required_inputs = list(head_required_inputs.get(head) or head_bundle.get("required_input_columns") or _required_inputs_for_features(active_feature_cols))
            component_scope = f"head:{head}"
            using_head_specific = True
            thresholds = dict(global_thresholds)
            thresholds.update((policy_contract.get("head_decision_thresholds") or {}).get(head, {}))
        elif head_specific_required:
            components = {}
            medians_by_component = {}
            active_strategy_map = strategy_map
            active_required_inputs = []
            component_scope = f"missing_head:{head}"
            using_head_specific = False
            thresholds = global_thresholds
        else:
            components = global_components
            medians_by_component = global_medians_by_component
            active_strategy_map = strategy_map
            active_required_inputs = required_inputs
            component_scope = "global"
            using_head_specific = False
            thresholds = global_thresholds

        p_min = float(thresholds.get("p_intervene_min", 0.50))
        action_min = float(thresholds.get("action_selector_min", 0.50))
        positive_min = float(thresholds.get("positive_value_min", 0.50))
        value_min = float(thresholds.get("pred_delta_J_min", 0.0))

        base_result = {
            "timestamp": timestamp,
            "strategy_id": strategy_id,
            "selected_multiplier": 1.0,
            "accepted": False,
            "reject_reason": "no_action_selected",
            "p_intervene": 0.0,
            "action_selector_score": 0.0,
            "positive_value_score": 0.0,
            "pred_delta_J": 0.0,
            "pred_immediate_J": 0.0,
            "pred_capacity_J": 0.0,
            "head": head,
            "component_scope": component_scope,
            "head_specific_component": bool(using_head_specific),
        }
        if head_specific_required and not head_bundle:
            base_result["reject_reason"] = "missing_head_specific_component"
            results.append(base_result)
            continue

        group = _prepare_rows(group, active_strategy_map)
        valid, reasons = _validate_required_inputs(group, active_required_inputs)
        if not valid:
            base_result["reject_reason"] = ";".join(reasons) if reasons else "invalid_required_inputs"
            results.append(base_result)
            continue

        group_first = group.iloc[[0]].copy()
        stage1_component = components.get("stage1_intervention_classifier") or {}
        p_intervene = float(
            _predict_component(
                stage1_component,
                group_first,
                medians_by_component.get("stage1_intervention_classifier", {}),
                proba=True,
            )[0]
        )
        base_result["p_intervene"] = p_intervene
        if not np.isfinite(p_intervene) or p_intervene < p_min:
            base_result["reject_reason"] = "stage1_below_threshold"
            results.append(base_result)
            continue

        multiplier = pd.to_numeric(group["multiplier"], errors="coerce").fillna(1.0)
        action_rows = group.loc[multiplier.lt(1.0)].copy()
        if "action_binds" in action_rows.columns:
            action_rows = action_rows.loc[pd.to_numeric(action_rows["action_binds"], errors="coerce").fillna(0.0).gt(0.0)].copy()
        if action_rows.empty:
            base_result["reject_reason"] = "no_binding_nonbaseline_action"
            results.append(base_result)
            continue

        action_score = _predict_component(
            components.get("action_selector") or {},
            action_rows,
            medians_by_component.get("action_selector", {}),
            proba=True,
        )
        positive_score = _predict_component(
            components.get("positive_value_classifier") or {},
            action_rows,
            medians_by_component.get("positive_value_classifier", {}),
            proba=True,
        )
        full_value = _predict_component(
            components.get("full_value_regressor") or {},
            action_rows,
            medians_by_component.get("full_value_regressor", {}),
            proba=False,
        )
        immediate_value = _predict_component(
            components.get("immediate_value_regressor") or {},
            action_rows,
            medians_by_component.get("immediate_value_regressor", {}),
            proba=False,
        )
        capacity_value = _predict_component(
            components.get("capacity_value_regressor") or {},
            action_rows,
            medians_by_component.get("capacity_value_regressor", {}),
            proba=False,
        )
        composite = action_score + positive_score + np.maximum(full_value, value_min) / max(abs(value_min), 1.0)
        order = np.argsort(-composite)
        accepted_idx: int | None = None
        for idx in order:
            if (
                np.isfinite(action_score[idx])
                and np.isfinite(positive_score[idx])
                and np.isfinite(full_value[idx])
                and action_score[idx] >= action_min
                and positive_score[idx] >= positive_min
                and full_value[idx] >= value_min
            ):
                accepted_idx = int(idx)
                break
        if accepted_idx is None:
            base_result["reject_reason"] = "action_value_safety_below_threshold"
            base_result["action_selector_score"] = float(np.nanmax(action_score)) if len(action_score) else 0.0
            base_result["positive_value_score"] = float(np.nanmax(positive_score)) if len(positive_score) else 0.0
            base_result["pred_delta_J"] = float(np.nanmax(full_value)) if len(full_value) else 0.0
            results.append(base_result)
            continue

        selected = action_rows.iloc[accepted_idx]
        results.append(
            {
                **base_result,
                "selected_multiplier": float(pd.to_numeric(pd.Series([selected["multiplier"]]), errors="coerce").fillna(1.0).iloc[0]),
                "accepted": True,
                "reject_reason": "",
                "action_selector_score": float(action_score[accepted_idx]),
                "positive_value_score": float(positive_score[accepted_idx]),
                "pred_delta_J": float(full_value[accepted_idx]),
                "pred_immediate_J": float(immediate_value[accepted_idx]),
                "pred_capacity_J": float(capacity_value[accepted_idx]),
            }
        )
    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--arm", default=DEFAULT_ARM)
    parser.add_argument("--material-gain", type=float, default=50.0)
    parser.add_argument("--top-fraction", type=float, default=0.075)
    parser.add_argument("--max-features", type=int, default=96)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--fit-split", choices=["train", "eval", "all"], default="train")
    parser.add_argument("--disable-head-specific", action="store_true")
    parser.add_argument("--min-head-group-rows", type=int, default=30)
    parser.add_argument(
        "--head-decision-thresholds",
        default="",
        help=(
            "Optional per-head live thresholds, e.g. "
            "short_asset:p_intervene_min=0.8,pred_delta_J_min=320;"
            "short_boll:p_intervene_min=0.2,pred_delta_J_min=200"
        ),
    )
    args = parser.parse_args()

    payload = materialize_bundle(
        freeze_manifest_path=args.freeze_manifest,
        run_dir=args.run_dir,
        out_dir=args.out_dir,
        arm=args.arm,
        material_gain=args.material_gain,
        top_fraction=args.top_fraction,
        max_features=args.max_features,
        seed=args.seed,
        fit_split=args.fit_split,
        head_specific=not bool(args.disable_head_specific),
        min_head_group_rows=int(args.min_head_group_rows),
        head_decision_thresholds=_parse_head_decision_thresholds(args.head_decision_thresholds),
    )
    print(
        {
            "out_dir": str(args.out_dir),
            "arm": payload["arm"],
            "component": payload["component"],
            "coverage": payload["coverage"],
            "feature_count": len(payload["feature_columns"]),
            "missing_components": payload["missing_components"],
            "head_specific_heads": payload.get("head_specific_diagnostics", {}).get("heads_fitted", []),
        }
    )


if __name__ == "__main__":
    main()
