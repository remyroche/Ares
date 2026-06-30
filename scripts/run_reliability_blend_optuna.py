#!/usr/bin/env python3
"""Optuna blends for anchor, difficult-period, and q_fail learners.

The experiment keeps the trading contract unchanged: one meta label, one score.
It only creates OOF component scores and searches a bounded nonlinear blend:

    score = ranked_anchor + alpha * shaped(ranked_period) + beta * shaped(ranked_q_fail)

where alpha and beta are constrained to [-0.5, 0.5].  Higher q_fail means higher
failure risk, so Optuna should choose a negative beta when the learner is useful.
The shaped components allow high-tail or low-tail emphasis with powers in [0.5, 2].
The exported score is a ranking score, not a calibrated probability.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import warnings
import hashlib
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from extreme_price_movements.performance_regimes.spectral_position import (
    MarketSpectralPositionConfig,
    fit_market_spectral_position_encoder,
    market_spectral_position_feature_names,
    transform_market_spectral_position,
)
from scripts import run_anchored_reliability_meta_correction as anchored
from scripts import run_canonical_context_retrain_experiment as canon
from scripts import run_contextual_meta_stack_trials as stack
from scripts import run_fixed_band_qfail_ablation as fixed
from scripts import run_one_head_contextual_meta_ablation as ctx
from scripts.diagnose_meta_recent_failures import (
    _base_models_for_head,
    _discover_heads,
    _downcast_numeric,
    _feature_store_union,
    _normalise_keys,
    _prepare_model_matrix,
    lgb,
)
from scripts.quantify_bad_regime_archetype_usefulness import _pick_realized_return


HEADS = anchored.HEADS
optuna.logging.set_verbosity(optuna.logging.WARNING)
SPECTRAL_POSITION_STATE_COLUMNS = market_spectral_position_feature_names("state_spectral_")

BLEND_OLD_HARD = "B1_old_period_hard_qfail"
BLEND_NEW_HARD = "B2_new_period_hard_qfail"
BLEND_NEW_SOFT = "B3_new_period_soft_qfail"
BLEND_OLD_SOFT = "B4_old_period_soft_qfail"
BLEND_VARIANTS = (BLEND_OLD_HARD, BLEND_NEW_HARD, BLEND_NEW_SOFT, BLEND_OLD_SOFT)
SOFT_QFAIL_BLEND_VARIANTS = (BLEND_NEW_SOFT, BLEND_OLD_SOFT)
NONLINEAR_POWER_GRID = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
NONLINEAR_SIDE_GRID = ("high", "low")
DEFAULT_BLEND_CONFIG_PATH = Path("config/reliability_blend_default_configs.json")
META_OUTPUT_TOKENS = (
    "oof_abs_raw_score",
    "oof_lgbm_raw_score",
    "oof_lgbm_prob",
    "oof_prob_",
    "oof_prob_std",
    "oof_prob_range",
    "oof_raw_score_",
    "oof_raw_score_std",
    "oof_raw_score_range",
    "oof_margin_from_neutral",
    "oof_score_margin",
    "oof_rank_margin",
    "oof_prob_uncertainty",
    "oof_entropy",
    "oof_variance_proxy",
    "oof_model_count",
    "oof_tree_count",
    "oof_contrib",
    "oof_top_",
    "oof_positive_contrib",
    "oof_negative_contrib",
    "oof_num_material_contrib_features",
    "oof_mean_tree_contribution",
    "oof_max_tree_contribution",
    "oof_top_tree_contribution_share",
    "oof_score_path_std",
    "oof_score_path_",
    "oof_score_path_volatility",
    "oof_score_path_drawdown",
    "oof_score_path_min",
    "oof_score_path_max",
    "oof_score_reversal_count",
    "oof_rank_path_std",
    "oof_rank_100_minus_50",
    "oof_score_100_minus_50",
    "oof_score_100_minus_75",
    "oof_score_final",
    "oof_score_early",
    "oof_feature_drift",
    "oof_latent_mahalanobis_drift",
    "oof_regime_centroid_similarity",
    "oof_leaf_count",
    "oof_rare_leaf",
    "oof_leaf_weight",
    "oof_leaf_depth",
    "oof_leaf_value_abs",
    "oof_large_leaf_value",
    "oof_leaf_train_freq",
    "oof_leaf_surprisal",
    "oof_leaf_low_freq_fraction",
    "oof_leaf_proximity",
    "oof_leaf_model_space_distance",
    "oof_support_gap",
    "oof_leaf_pred",
    "oof_dae_reconstruction_error",
    "oof_dae_b16_",
    "oof_cluster_entropy",
    "oof_gmm_prob",
    "oof_gmm_dist",
    "oof_gmm_mahal",
    "oof_min_mahalanobis",
    "oof_expected_mahalanobis",
    "oof_cluster_t",
    "oof_time_since_cluster_change",
    "oof_rolling_cluster_stability",
    "oof_cluster_flip_count",
    "oof_model_count",
    "oof_tree_count",
    "diag_mean_pred",
    "diag_std_pred",
)
META_OUTPUT_UNCERTAINTY_TOKENS = (
    "uncert",
    "entropy",
    "variance",
    "std",
    "range",
    "margin",
    "prob_",
    "raw_score",
    "abs_raw",
    "contrib",
    "tree_count",
    "model_count",
    "diag_std",
)
META_OUTPUT_DRIFT_TOKENS = (
    "drift",
    "path",
    "reversal",
    "score_100",
    "rank_100",
    "score_early",
    "regime",
    "centroid",
    "dae",
    "gmm",
    "cluster",
    "mahal",
    "rolling_cluster",
    "time_since_cluster",
)
META_OUTPUT_SUPPORT_TOKENS = (
    "leaf_count",
    "rare_leaf",
    "leaf_weight",
    "leaf_depth",
    "leaf_value",
    "leaf_train_freq",
    "leaf_surprisal",
    "leaf_low_freq",
    "leaf_proximity",
    "leaf_model_space",
    "support_gap",
    "leaf_pred",
)
META_OUTPUT_FORBIDDEN_TOKENS = (
    "target",
    "hit_rate",
    "leaf_error",
    "rank_bin",
    "net_ret",
    "win_rate",
    "lift",
    "realized",
    "barrier",
    "future",
    "pnl",
)
LIVE_CONTRACT_OOF_FEATURES = {
    "oof_lgbm_prob",
    "oof_meta_clf",
    "oof_base_clf",
    "oof_p_move",
    "oof_rank_pct",
    "oof_prob_uncertainty",
    "oof_entropy",
    "oof_rare_leaf_fraction",
    "oof_leaf_count_p10",
    "oof_leaf_count_min",
    "oof_leaf_weight_p10",
    "oof_leaf_depth_mean",
    "oof_contrib_top1_abs_share",
    "oof_contrib_top3_abs_share",
    "oof_contrib_entropy",
    "oof_contrib_balance",
    "oof_num_material_contrib_features",
    "oof_feature_drift_psi_core",
    "oof_feature_drift_ks_core",
    "oof_feature_drift_cov_shift",
    "oof_regime_centroid_similarity_train",
}


def _json_default(value: Any) -> Any:
    return ctx._json_default(value)


def _stable_seed_offset(*parts: Any) -> int:
    h = hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16) % 100000


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _public_arg_dict(args: argparse.Namespace) -> dict[str, Any]:
    return {str(k): v for k, v in vars(args).items() if not str(k).startswith("_")}


def _feature_contract_hash(columns: list[str]) -> str:
    payload = json.dumps([str(c) for c in columns], separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _component_label_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"period", "period_new", "new_period", "q_period", "q_period_soft"}:
        return "new_period"
    if text in {"qfail", "q_fail", "qfail_soft", "soft_qfail", "high_confidence_failure"}:
        return "qfail_soft"
    return text


def _scope_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text == "full_fit":
        return "full_fit"
    return "oof"


def _feature_reuse_key(head: str, label: str, scope: str) -> tuple[str, str, str]:
    return (str(head), _component_label_name(label), _scope_name(scope))


def _empty_feature_reuse_cache() -> dict[tuple[str, str, str], dict[str, Any]]:
    return {}


def _add_feature_reuse_entry(
    cache: dict[tuple[str, str, str], dict[str, Any]],
    *,
    head: str,
    label: str,
    scope: str,
    features: list[str],
    params: dict[str, Any] | None = None,
    source: str,
    priority: int,
    allow_partial_match: bool = False,
) -> None:
    clean_features = [str(c) for c in dict.fromkeys(features or []) if str(c).strip()]
    if not clean_features:
        return
    key = _feature_reuse_key(head, label, scope)
    prior = cache.get(key)
    if prior is not None and int(prior.get("priority", -1)) > int(priority):
        return
    cache[key] = {
        "head": str(head),
        "label": _component_label_name(label),
        "scope": _scope_name(scope),
        "features": clean_features,
        "params": dict(params or {}),
        "source": str(source),
        "priority": int(priority),
        "feature_count": int(len(clean_features)),
        "feature_sha256": _feature_contract_hash(clean_features),
        "has_params": bool(params),
        "allow_partial_match": bool(allow_partial_match),
    }


def _load_feature_reuse_from_manifest(
    cache: dict[tuple[str, str, str], dict[str, Any]],
    path: Path,
) -> None:
    obj = json.loads(path.read_text())
    features_by_head = obj.get("features", {}) if isinstance(obj, dict) else {}
    if not isinstance(features_by_head, dict):
        return
    for head, feature_blocks in features_by_head.items():
        if not isinstance(feature_blocks, dict):
            continue
        for name, features in feature_blocks.items():
            if not isinstance(features, list):
                continue
            lower = str(name).lower()
            if "period_timestamp" in lower:
                label = "new_period"
            elif "qfail_row" in lower:
                label = "qfail_soft"
            else:
                continue
            scope = "full_fit" if lower.startswith("full_fit") else "oof"
            _add_feature_reuse_entry(
                cache,
                head=str(head),
                label=label,
                scope=scope,
                features=[str(c) for c in features],
                params=None,
                source=f"{path}::{name}",
                priority=10,
                allow_partial_match=True,
            )


def _iter_component_artifacts(obj: Any) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    if not isinstance(obj, dict):
        return artifacts
    for head, head_bundle in dict(obj.get("heads", {}) or {}).items():
        if not isinstance(head_bundle, dict):
            continue
        for artifact in list(head_bundle.get("models", []) or []):
            if isinstance(artifact, dict):
                artifact = dict(artifact)
                artifact.setdefault("head", str(head))
                artifacts.append(artifact)
    return artifacts


def _load_feature_reuse_from_component_bundle(
    cache: dict[tuple[str, str, str], dict[str, Any]],
    path: Path,
) -> None:
    obj = joblib.load(path)
    for artifact in _iter_component_artifacts(obj):
        head = str(artifact.get("head", ""))
        label = _component_label_name(artifact.get("component") or artifact.get("target_label"))
        if label not in {"new_period", "qfail_soft"}:
            continue
        scope = _scope_name(artifact.get("model_scope") or artifact.get("fold"))
        selected = [str(c) for c in (artifact.get("selected_features") or []) if str(c).strip()]
        inputs = [str(c) for c in (artifact.get("input_feature_columns") or []) if str(c).strip()]
        params = dict(artifact.get("hpo_params") or {})
        if selected:
            _add_feature_reuse_entry(
                cache,
                head=head,
                label=label,
                scope=scope,
                features=selected,
                params=params,
                source=f"{path}::{head}:{scope}:{label}:selected_features",
                priority=50 if scope == "full_fit" else 40,
            )
        elif inputs:
            _add_feature_reuse_entry(
                cache,
                head=head,
                label=label,
                scope=scope,
                features=inputs,
                params=params,
                source=f"{path}::{head}:{scope}:{label}:input_feature_columns",
                priority=20 if scope == "full_fit" else 15,
            )


def _load_native_aux_feature_reuse_cache(paths_text: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    cache = _empty_feature_reuse_cache()
    for raw_path in str(paths_text or "").split(","):
        if not raw_path.strip():
            continue
        path = Path(raw_path.strip())
        if path.is_dir():
            candidates = [
                path / "reliability_blend_component_models" / "reliability_blend_native_component_models.joblib",
                path / "reliability_blend_component_models" / "reliability_blend_native_component_model_manifest.json",
                path / "reliability_blend_feature_target_manifest.json",
                path / "reliability_blend_native_component_models.joblib",
                path / "reliability_blend_native_component_model_manifest.json",
            ]
        else:
            candidates = [path]
        for candidate in candidates:
            if not candidate.exists():
                continue
            name = candidate.name
            try:
                if name.endswith(".joblib"):
                    _load_feature_reuse_from_component_bundle(cache, candidate)
                elif name == "reliability_blend_feature_target_manifest.json":
                    _load_feature_reuse_from_manifest(cache, candidate)
            except Exception as exc:
                print(f"[reliability_blend] WARNING: failed to load feature reuse source {candidate}: {exc}", flush=True)
    return cache


def _resolve_native_aux_feature_reuse(
    args: argparse.Namespace,
    *,
    head: str | None,
    label: str,
    scope: str,
    available_columns: list[str],
) -> tuple[list[str] | None, dict[str, Any] | None, dict[str, Any]]:
    cache = getattr(args, "_native_aux_feature_reuse_cache", None) or {}
    if not bool(getattr(args, "aux_native_reuse_features", True)):
        return None, None, {"enabled": False, "reason": "disabled"}
    if not cache:
        return None, None, {"enabled": True, "reason": "empty_cache"}
    heads = [str(head)] if head else []
    labels = [_component_label_name(label)]
    scopes = [_scope_name(scope)]
    if "full_fit" in scopes:
        scopes.append("oof")
    else:
        scopes.append("full_fit")
    available = set(map(str, available_columns))
    best: dict[str, Any] | None = None
    best_present: list[str] = []
    for candidate_head in heads:
        for candidate_scope in scopes:
            entry = cache.get(_feature_reuse_key(candidate_head, labels[0], candidate_scope))
            if not entry:
                continue
            present = [c for c in entry.get("features", []) if c in available]
            min_features = int(getattr(args, "aux_native_reuse_min_features", 8))
            min_fraction = float(getattr(args, "aux_native_reuse_min_fraction", 0.25))
            if bool(entry.get("allow_partial_match", False)):
                required = min_features
            else:
                required = max(min_features, int(math.ceil(min_fraction * max(1, len(entry.get("features", []))))))
            if len(present) < required:
                continue
            score = (int(entry.get("priority", 0)), len(present), -len(entry.get("features", [])))
            if best is None or score > best.get("_score", (-1, -1, -10**9)):
                best = dict(entry)
                best["_score"] = score
                best_present = present
    if best is None:
        return None, None, {
            "enabled": True,
            "reason": "no_compatible_reuse_entry",
            "head": str(head or ""),
            "label": _component_label_name(label),
            "scope": _scope_name(scope),
            "cache_entries": int(len(cache)),
        }
    params = dict(best.get("params") or {})
    return best_present, (params or None), {
        "enabled": True,
        "reason": "reused",
        "head": str(head or ""),
        "label": _component_label_name(label),
        "scope": _scope_name(scope),
        "source": str(best.get("source", "")),
        "source_scope": str(best.get("scope", "")),
        "source_feature_count": int(best.get("feature_count", 0) or 0),
        "matched_feature_count": int(len(best_present)),
        "matched_fraction": float(len(best_present) / max(int(best.get("feature_count", 0) or 0), 1)),
        "feature_sha256": _feature_contract_hash(best_present),
        "reused_hpo_params": bool(params),
    }


def _timestamp_bounds(values: pd.Series | np.ndarray | list[Any]) -> dict[str, Any]:
    ts = pd.to_datetime(pd.Series(values), utc=True, errors="coerce")
    if ts.notna().any():
        return {
            "min": ts.min().isoformat(),
            "max": ts.max().isoformat(),
            "rows": int(len(ts)),
        }
    return {"min": None, "max": None, "rows": int(len(ts))}


def _component_model_summary(artifact: dict[str, Any] | None) -> dict[str, Any]:
    if not artifact:
        return {}
    out: dict[str, Any] = {}
    for key, value in artifact.items():
        if key == "model":
            out["model_class"] = type(value).__name__
        elif key in {"input_feature_columns", "selected_features", "timestamp_feature_columns"}:
            vals = [str(v) for v in (value or [])]
            out[key] = vals[:100]
            out[f"{key}_count"] = int(len(vals))
            out[f"{key}_sha256"] = _feature_contract_hash(vals)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        elif isinstance(value, dict):
            out[key] = _json_default(value)
        elif isinstance(value, (list, tuple)):
            out[key] = _json_default(list(value[:100]))
            out[f"{key}_count"] = int(len(value))
        else:
            out[key] = type(value).__name__
    return out


def _score_reference_payload(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    arr.sort()
    return {
        "scores": arr.tolist(),
        "n_rows": int(arr.size),
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        "min": float(arr[0]) if arr.size else np.nan,
        "max": float(arr[-1]) if arr.size else np.nan,
    }


def _score_reference_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_rows": int(payload.get("n_rows", 0) or 0),
        "sha256": payload.get("sha256"),
        "min": payload.get("min"),
        "max": payload.get("max"),
    }


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    y_arr = np.asarray(y)
    s = np.asarray(score, dtype=np.float64)
    mask = np.isfinite(s) & np.isfinite(y_arr)
    if int(mask.sum()) < 30 or len(np.unique(y_arr[mask])) < 2:
        return np.nan
    return float(roc_auc_score(y_arr[mask], s[mask]))


def _safe_logloss(y: np.ndarray, score: np.ndarray) -> float:
    y_arr = np.asarray(y)
    s = np.clip(np.asarray(score, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    mask = np.isfinite(s) & np.isfinite(y_arr)
    if int(mask.sum()) < 30 or len(np.unique(y_arr[mask])) < 2:
        return np.nan
    return float(log_loss(y_arr[mask].astype(int), s[mask], labels=[0, 1]))


def _rank01(values: np.ndarray) -> np.ndarray:
    vals = pd.to_numeric(pd.Series(values), errors="coerce")
    finite = vals.notna()
    out = np.full(len(vals), 0.5, dtype=np.float32)
    if int(finite.sum()) <= 1:
        return out
    out[finite.to_numpy()] = vals[finite].rank(method="average", pct=True).to_numpy(dtype=np.float32)
    return np.clip(out, 0.0, 1.0)


def _rank01_by_timestamp(timestamps: pd.Series, values: np.ndarray) -> np.ndarray:
    return stack._rank_pct_by_timestamp(pd.to_datetime(timestamps, utc=True, errors="coerce"), np.asarray(values, dtype=np.float32))


def _top_hr_parts(y: np.ndarray, score: np.ndarray, pct: float) -> tuple[float, int]:
    mask = np.isfinite(score) & (np.asarray(y) >= 0)
    if int(mask.sum()) < 10:
        return np.nan, 0
    yy = np.asarray(y, dtype=np.float32)[mask]
    ss = np.asarray(score, dtype=np.float64)[mask]
    k = max(1, int(math.ceil(float(pct) * len(yy))))
    if k >= len(yy):
        top = np.arange(len(yy), dtype=np.int64)
    else:
        top = np.argpartition(ss, -k)[-k:]
    return float(np.mean(yy[top])), int(k)


def _tophr(y: np.ndarray, score: np.ndarray) -> float:
    hr10, _ = _top_hr_parts(y, score, 0.10)
    hr20, _ = _top_hr_parts(y, score, 0.20)
    hr30, _ = _top_hr_parts(y, score, 0.30)
    vals = [hr10, hr20, hr30]
    if not all(np.isfinite(vals)):
        return np.nan
    return float(hr10 + 0.33 * hr20 + 0.25 * hr30)


def _week_tophr_table(timestamps: pd.Series, y: np.ndarray, score: np.ndarray, *, min_week_rows: int) -> pd.DataFrame:
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce")
    frame = pd.DataFrame({"timestamp": ts, "y": y, "score": score})
    frame = frame[np.isfinite(frame["score"]) & (frame["y"] >= 0)].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["week"] = frame["timestamp"].dt.tz_convert(None).dt.to_period("W").dt.start_time.astype(str)
    rows: list[dict[str, Any]] = []
    for week, group in frame.groupby("week", sort=True):
        if len(group) < int(min_week_rows):
            continue
        yy = group["y"].to_numpy(dtype=np.float32)
        ss = group["score"].to_numpy(dtype=np.float64)
        hr10, k10 = _top_hr_parts(yy, ss, 0.10)
        hr20, k20 = _top_hr_parts(yy, ss, 0.20)
        hr30, k30 = _top_hr_parts(yy, ss, 0.30)
        rows.append(
            {
                "week": week,
                "rows": int(len(group)),
                "hr10": hr10,
                "hr20": hr20,
                "hr30": hr30,
                "k10": k10,
                "k20": k20,
                "k30": k30,
                "tophr": float(hr10 + 0.33 * hr20 + 0.25 * hr30) if np.isfinite(hr10) and np.isfinite(hr20) and np.isfinite(hr30) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _blend_metrics(timestamps: pd.Series, y: np.ndarray, score: np.ndarray, *, min_week_rows: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "auc": _safe_auc(y, score),
        "logloss": _safe_logloss(y, score),
    }
    for pct in (10, 20, 30):
        hr, count = _top_hr_parts(y, score, pct / 100.0)
        out[f"global_hr{pct}"] = hr
        out[f"global_k{pct}"] = count
    out["global_tophr"] = _tophr(y, score)
    weekly = _week_tophr_table(timestamps, y, score, min_week_rows=min_week_rows)
    out["week_count"] = int(len(weekly))
    if weekly.empty:
        # Small smoke/subsample runs can span too thinly across weeks.  Keep the
        # Optuna objective finite there; full runs use the requested weekly tails.
        out.update({"q05_tophr": out["global_tophr"], "q25_tophr": out["global_tophr"], "objective": out["global_tophr"]})
        return out
    out["q05_tophr"] = float(weekly["tophr"].quantile(0.05))
    out["q25_tophr"] = float(weekly["tophr"].quantile(0.25))
    for pct in (10, 20, 30):
        out[f"q05_hr{pct}"] = float(weekly[f"hr{pct}"].quantile(0.05))
        out[f"q25_hr{pct}"] = float(weekly[f"hr{pct}"].quantile(0.25))
    out["objective"] = float(out["global_tophr"] + 0.33 * out["q25_tophr"] + 0.20 * out["q05_tophr"])
    return out


def _metric_context(timestamps: pd.Series, y: np.ndarray, *, min_week_rows: int) -> dict[str, Any]:
    y_arr = np.asarray(y, dtype=np.float32)
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").reset_index(drop=True)
    mask = (y_arr >= 0) & ts.notna().to_numpy()
    y_eval = y_arr[mask]
    ts_eval = ts[mask]
    week_labels = ts_eval.dt.tz_convert(None).dt.to_period("W").dt.start_time.astype(str)
    week_groups: list[np.ndarray] = []
    for _week, ids in pd.Series(np.arange(len(y_eval), dtype=np.int64)).groupby(week_labels, sort=True):
        arr = ids.to_numpy(dtype=np.int64)
        if len(arr) >= int(min_week_rows):
            week_groups.append(arr)
    return {"mask": mask, "y": y_eval, "week_groups": week_groups}


def _top_hr_arrays(y: np.ndarray, score: np.ndarray, pct: float) -> tuple[float, int]:
    if len(y) < 10:
        return np.nan, 0
    k = max(1, int(math.ceil(float(pct) * len(y))))
    if k >= len(y):
        top = np.arange(len(y), dtype=np.int64)
    else:
        top = np.argpartition(np.asarray(score, dtype=np.float64), -k)[-k:]
    return float(np.mean(np.asarray(y, dtype=np.float32)[top])), int(k)


def _tophr_arrays(y: np.ndarray, score: np.ndarray) -> tuple[float, dict[str, Any]]:
    hr10, k10 = _top_hr_arrays(y, score, 0.10)
    hr20, k20 = _top_hr_arrays(y, score, 0.20)
    hr30, k30 = _top_hr_arrays(y, score, 0.30)
    if not np.isfinite(hr10) or not np.isfinite(hr20) or not np.isfinite(hr30):
        return np.nan, {"global_hr10": hr10, "global_hr20": hr20, "global_hr30": hr30, "global_k10": k10, "global_k20": k20, "global_k30": k30}
    return float(hr10 + 0.33 * hr20 + 0.25 * hr30), {
        "global_hr10": hr10,
        "global_hr20": hr20,
        "global_hr30": hr30,
        "global_k10": k10,
        "global_k20": k20,
        "global_k30": k30,
    }


def _blend_metrics_fast(context: dict[str, Any], score: np.ndarray) -> dict[str, Any]:
    y = np.asarray(context["y"], dtype=np.float32)
    s = np.asarray(score, dtype=np.float32)
    global_tophr, parts = _tophr_arrays(y, s)
    out: dict[str, Any] = {
        **parts,
        "global_tophr": global_tophr,
        "week_count": int(len(context["week_groups"])),
    }
    week_vals: list[float] = []
    week_hr10: list[float] = []
    week_hr20: list[float] = []
    week_hr30: list[float] = []
    for ids in context["week_groups"]:
        top, subparts = _tophr_arrays(y[ids], s[ids])
        if np.isfinite(top):
            week_vals.append(top)
            week_hr10.append(float(subparts["global_hr10"]))
            week_hr20.append(float(subparts["global_hr20"]))
            week_hr30.append(float(subparts["global_hr30"]))
    if not week_vals:
        out.update({"q05_tophr": global_tophr, "q25_tophr": global_tophr})
        out["objective"] = global_tophr
        return out
    vals = np.asarray(week_vals, dtype=np.float64)
    out["q05_tophr"] = float(np.nanquantile(vals, 0.05))
    out["q25_tophr"] = float(np.nanquantile(vals, 0.25))
    for pct, arr in ((10, week_hr10), (20, week_hr20), (30, week_hr30)):
        values = np.asarray(arr, dtype=np.float64)
        out[f"q05_hr{pct}"] = float(np.nanquantile(values, 0.05))
        out[f"q25_hr{pct}"] = float(np.nanquantile(values, 0.25))
    out["objective"] = float(global_tophr + 0.33 * out["q25_tophr"] + 0.20 * out["q05_tophr"]) if np.isfinite(global_tophr) else np.nan
    return out


def _shape_rank(values: np.ndarray, *, power: float, side: str) -> np.ndarray:
    x = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    p = float(power)
    if not np.isfinite(p):
        p = 1.0
    p = min(2.0, max(0.5, p))
    if str(side) == "low":
        return np.power(1.0 - x, p).astype(np.float32, copy=False)
    return np.power(x, p).astype(np.float32, copy=False)


def _blend_score(
    anchor_rank: np.ndarray,
    period_rank: np.ndarray,
    qfail_rank: np.ndarray,
    alpha: float,
    beta: float,
    *,
    period_power: float = 1.0,
    period_side: str = "high",
    qfail_power: float = 1.0,
    qfail_side: str = "high",
) -> np.ndarray:
    return (
        np.asarray(anchor_rank, dtype=np.float32)
        + float(alpha) * _shape_rank(period_rank, power=period_power, side=period_side)
        + float(beta) * _shape_rank(qfail_rank, power=qfail_power, side=qfail_side)
    ).astype(np.float32, copy=False)


def _clean_blend_default_config(config: dict[str, Any] | None, *, variant: str | None = None) -> dict[str, Any] | None:
    if not config:
        return None
    if variant is not None and str(config.get("variant", variant)) != str(variant):
        return None
    try:
        alpha = float(config["alpha"])
        beta = float(config["beta"])
        period_power = float(config.get("period_power", 1.0))
        qfail_power = float(config.get("qfail_power", 1.0))
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(alpha) or not np.isfinite(beta):
        return None
    alpha = min(0.5, max(-0.5, alpha))
    beta = min(0.5, max(-0.5, beta))
    period_power = min(2.0, max(0.5, period_power if np.isfinite(period_power) else 1.0))
    qfail_power = min(2.0, max(0.5, qfail_power if np.isfinite(qfail_power) else 1.0))
    period_side = str(config.get("period_side", "high"))
    qfail_side = str(config.get("qfail_side", "high"))
    if period_side not in NONLINEAR_SIDE_GRID:
        period_side = "high"
    if qfail_side not in NONLINEAR_SIDE_GRID:
        qfail_side = "high"
    return {
        "alpha": alpha,
        "beta": beta,
        "period_power": period_power,
        "period_side": period_side,
        "qfail_power": qfail_power,
        "qfail_side": qfail_side,
    }


def _load_blend_default_configs(path: str | Path | None) -> dict[str, dict[str, dict[str, Any]]]:
    if not path:
        return {}
    fp = Path(path)
    if not fp.exists():
        return {}
    try:
        raw = json.loads(fp.read_text())
    except Exception as exc:
        warnings.warn(f"Could not read reliability blend default config {fp}: {exc}")
        return {}
    configs = raw.get("configs", raw) if isinstance(raw, dict) else {}
    out: dict[str, dict[str, dict[str, Any]]] = {}
    if not isinstance(configs, dict):
        return out
    for head, value in configs.items():
        if not isinstance(value, dict):
            continue
        if "variant" in value:
            variant = str(value.get("variant"))
            cleaned = _clean_blend_default_config(value, variant=variant)
            if cleaned:
                out.setdefault(str(head), {})[variant] = cleaned
            continue
        for variant, config in value.items():
            if isinstance(config, dict):
                cleaned = _clean_blend_default_config(config, variant=str(variant))
                if cleaned:
                    out.setdefault(str(head), {})[str(variant)] = cleaned
    return out


def _optimise_blend(
    *,
    timestamps: pd.Series,
    y: np.ndarray,
    anchor_rank: np.ndarray,
    period_rank: np.ndarray,
    qfail_rank: np.ndarray,
    variant: str,
    seed: int,
    n_trials: int,
    min_week_rows: int,
    default_config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    context = _metric_context(timestamps, y, min_week_rows=min_week_rows)

    def objective(trial: optuna.Trial) -> float:
        alpha = trial.suggest_float("alpha", -0.5, 0.5)
        beta = trial.suggest_float("beta", -0.5, 0.5)
        period_power = trial.suggest_categorical("period_power", list(NONLINEAR_POWER_GRID))
        qfail_power = trial.suggest_categorical("qfail_power", list(NONLINEAR_POWER_GRID))
        period_side = trial.suggest_categorical("period_side", list(NONLINEAR_SIDE_GRID))
        qfail_side = trial.suggest_categorical("qfail_side", list(NONLINEAR_SIDE_GRID))
        score = _blend_score(
            anchor_rank,
            period_rank,
            qfail_rank,
            alpha,
            beta,
            period_power=float(period_power),
            period_side=str(period_side),
            qfail_power=float(qfail_power),
            qfail_side=str(qfail_side),
        )
        metric = _blend_metrics_fast(context, score).get("objective", np.nan)
        return float(metric) if np.isfinite(metric) else -1e9

    sampler = optuna.samplers.TPESampler(seed=int(seed), n_startup_trials=min(20, max(5, int(n_trials // 4))))
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    default_shape = {"period_power": 1.0, "qfail_power": 1.0, "period_side": "high", "qfail_side": "high"}
    cleaned_default = _clean_blend_default_config(default_config, variant=variant)
    if cleaned_default is not None:
        study.enqueue_trial(cleaned_default)
    study.enqueue_trial({"alpha": 0.0, "beta": 0.0, **default_shape})
    study.enqueue_trial({"alpha": -0.1, "beta": -0.1, **default_shape})
    study.enqueue_trial({"alpha": -0.2, "beta": -0.2, **default_shape})
    if int(n_trials) > 0:
        study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        alpha = float(trial.params.get("alpha", np.nan))
        beta = float(trial.params.get("beta", np.nan))
        period_power = float(trial.params.get("period_power", 1.0))
        qfail_power = float(trial.params.get("qfail_power", 1.0))
        period_side = str(trial.params.get("period_side", "high"))
        qfail_side = str(trial.params.get("qfail_side", "high"))
        score = _blend_score(
            anchor_rank,
            period_rank,
            qfail_rank,
            alpha,
            beta,
            period_power=period_power,
            period_side=period_side,
            qfail_power=qfail_power,
            qfail_side=qfail_side,
        )
        rows.append(
            {
                "variant": variant,
                "trial": int(trial.number),
                "alpha": alpha,
                "beta": beta,
                "period_power": period_power,
                "period_side": period_side,
                "qfail_power": qfail_power,
                "qfail_side": qfail_side,
                **_blend_metrics_fast(context, score),
            }
        )
    if not rows and cleaned_default is not None:
        score = _blend_score(
            anchor_rank,
            period_rank,
            qfail_rank,
            float(cleaned_default["alpha"]),
            float(cleaned_default["beta"]),
            period_power=float(cleaned_default.get("period_power", 1.0)),
            period_side=str(cleaned_default.get("period_side", "high")),
            qfail_power=float(cleaned_default.get("qfail_power", 1.0)),
            qfail_side=str(cleaned_default.get("qfail_side", "high")),
        )
        rows.append(
            {
                "variant": variant,
                "trial": -1,
                **cleaned_default,
                **_blend_metrics_fast(context, score),
            }
        )
    best = max(rows, key=lambda r: float(r.get("objective", -1e9))) if rows else {"variant": variant, "alpha": 0.0, "beta": 0.0}
    best_score = _blend_score(
        anchor_rank,
        period_rank,
        qfail_rank,
        float(best.get("alpha", 0.0)),
        float(best.get("beta", 0.0)),
        period_power=float(best.get("period_power", 1.0)),
        period_side=str(best.get("period_side", "high")),
        qfail_power=float(best.get("qfail_power", 1.0)),
        qfail_side=str(best.get("qfail_side", "high")),
    )
    best["auc"] = _safe_auc(y, best_score)
    best["logloss"] = _safe_logloss(y, best_score)
    return best, pd.DataFrame(rows)


def _timestamp_mean_error(
    timestamps: pd.Series,
    y: np.ndarray,
    anchor_score: np.ndarray,
    rank0: np.ndarray,
    *,
    rank_threshold: float,
) -> pd.Series:
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").reset_index(drop=True)
    y_arr = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(anchor_score, dtype=np.float64), 0.0, 1.0)
    r = np.asarray(rank0, dtype=np.float64)
    mask = (y_arr >= 0) & np.isfinite(p) & np.isfinite(r) & (r >= float(rank_threshold))
    if not mask.any():
        return pd.Series(dtype="float32")
    err = np.abs(p[mask] - y_arr[mask])
    raw = pd.DataFrame({"timestamp": ts.iloc[np.flatnonzero(mask)].to_numpy(), "error": err}).groupby("timestamp", sort=True)["error"].mean()
    return raw.astype("float32")


def _future_ewma(raw: pd.Series, *, horizon_hours: int, halflife_hours: float) -> pd.Series:
    if raw.empty:
        return raw
    raw = raw.sort_index().astype(float)
    idx = pd.to_datetime(raw.index, utc=True, errors="coerce")
    times = idx.view("int64")
    vals = raw.to_numpy(dtype=np.float64)
    out = np.full(len(vals), np.nan, dtype=np.float64)
    horizon_ns = int(pd.Timedelta(hours=int(horizon_hours)).value)
    half_ns = float(pd.Timedelta(hours=float(halflife_hours)).value)
    for i, t in enumerate(times):
        end = int(np.searchsorted(times, t + horizon_ns, side="right"))
        start = i + 1
        if end <= start:
            continue
        sub_vals = vals[start:end]
        sub_times = times[start:end]
        finite = np.isfinite(sub_vals)
        if not finite.any():
            continue
        age = (sub_times[finite] - t).astype(np.float64)
        weights = np.exp(-np.log(2.0) * age / max(half_ns, 1.0))
        out[i] = float(np.sum(weights * sub_vals[finite]) / np.sum(weights))
    return pd.Series(out, index=raw.index, dtype="float32")


def _percentile_from_train(train_values: np.ndarray, values: np.ndarray, *, nonfinite_fill: float | None = 0.5) -> np.ndarray:
    train = np.asarray(train_values, dtype=np.float64)
    base = np.sort(train[np.isfinite(train)])
    fill = np.nan if nonfinite_fill is None else float(nonfinite_fill)
    out = np.full(len(values), fill, dtype=np.float32)
    if len(base) < 10:
        return out
    vals = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(vals)
    out[finite] = (np.searchsorted(base, vals[finite], side="right") / float(len(base))).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def _soft_difficulty_target(
    timestamps: pd.Series,
    y: np.ndarray,
    anchor_score: np.ndarray,
    rank0: np.ndarray,
    *,
    rank_threshold: float,
    horizon_hours: int,
    halflife_hours: float,
) -> pd.Series:
    raw = _timestamp_mean_error(timestamps, y, anchor_score, rank0, rank_threshold=rank_threshold)
    return _future_ewma(raw, horizon_hours=horizon_hours, halflife_hours=halflife_hours)


def _period_detection_core(target: np.ndarray, pred: np.ndarray, *, bad_share: float) -> dict[str, Any]:
    y = np.asarray(target, dtype=np.float64)
    s = np.asarray(pred, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(s)
    if int(mask.sum()) < 20:
        return {
            "ap": np.nan,
            "ap_lift": np.nan,
            "recall": np.nan,
            "ndcg": np.nan,
            "spearman": np.nan,
            "difficulty_decile_spread": np.nan,
            "bad_threshold": np.nan,
            "bad_count": 0,
        }
    yy = y[mask]
    ss = s[mask]
    threshold = float(np.nanquantile(yy, 1.0 - float(bad_share)))
    bad = (yy >= threshold).astype(np.int8)
    if len(np.unique(bad)) < 2:
        ap = np.nan
    else:
        ap = float(average_precision_score(bad, ss))
    k = max(1, int(math.ceil(float(bad_share) * len(yy))))
    order = np.argsort(ss, kind="mergesort")[::-1]
    top = order[:k]
    recall = float(np.sum(bad[top]) / max(float(np.sum(bad)), 1.0))
    gains = bad[top].astype(float)
    denom = np.log2(np.arange(2, len(gains) + 2, dtype=np.float64))
    dcg = float(np.sum(gains / denom))
    ideal = np.sort(bad)[::-1][:k].astype(float)
    ideal_dcg = float(np.sum(ideal / denom)) if len(ideal) else 0.0
    ndcg = float(dcg / ideal_dcg) if ideal_dcg > 0 else np.nan
    spear = spearmanr(yy, ss, nan_policy="omit").correlation
    top_decile = yy[order[: max(1, int(math.ceil(0.10 * len(yy))))]]
    bottom_decile = yy[order[-max(1, int(math.ceil(0.10 * len(yy)))) :]]
    return {
        "ap": ap,
        "ap_lift": float((ap - float(bad_share)) / (1.0 - float(bad_share))) if np.isfinite(ap) else np.nan,
        "recall": recall,
        "ndcg": ndcg,
        "spearman": float(spear) if np.isfinite(spear) else np.nan,
        "difficulty_decile_spread": float(np.nanmean(top_decile) - np.nanmean(bottom_decile)),
        "bad_threshold": threshold,
        "bad_count": int(np.sum(bad)),
    }


def _period_detection_metrics(target: np.ndarray, pred: np.ndarray, *, bad_share: float = 0.10) -> dict[str, Any]:
    core = _period_detection_core(target, pred, bad_share=bad_share)
    return {
        "ap10": core["ap"],
        "ap_lift10": core["ap_lift"],
        "recall10": core["recall"],
        "ndcg10": core["ndcg"],
        "spearman": core["spearman"],
        "difficulty_decile_spread": core["difficulty_decile_spread"],
    }


def _period_tail_detection_metrics(target: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for share, suffix in ((0.05, "05"), (0.10, "10"), (0.15, "15")):
        core = _period_detection_core(target, pred, bad_share=share)
        out[f"ap{suffix}"] = core["ap"]
        out[f"ap_lift{suffix}"] = core["ap_lift"]
        out[f"recall{suffix}"] = core["recall"]
        out[f"ndcg{suffix}"] = core["ndcg"]
        out[f"bad_threshold{suffix}"] = core["bad_threshold"]
        out[f"bad_count{suffix}"] = core["bad_count"]
        if suffix == "10":
            out["spearman"] = core["spearman"]
            out["difficulty_decile_spread"] = core["difficulty_decile_spread"]
    return out


def _period_tail_labels(target: np.ndarray) -> dict[str, np.ndarray]:
    y = np.asarray(target, dtype=np.float64)
    finite = np.isfinite(y)
    labels: dict[str, np.ndarray] = {}
    for share, suffix in ((0.05, "05"), (0.10, "10"), (0.15, "15")):
        arr = np.full(len(y), np.nan, dtype=np.float32)
        if finite.any():
            threshold = float(np.nanquantile(y[finite], 1.0 - share))
            arr[finite] = 0.0
            arr[finite & (y >= threshold)] = 1.0
        labels[f"period_bad_{suffix}"] = arr
    severity = np.full(len(y), np.nan, dtype=np.float32)
    if finite.any():
        q85 = float(np.nanquantile(y[finite], 0.85))
        q95 = float(np.nanquantile(y[finite], 0.95))
        denom = max(q95 - q85, 1e-6)
        severity[finite] = np.clip((y[finite] - q85) / denom, 0.0, 1.0).astype(np.float32)
    labels["period_tail_severity"] = severity
    return labels


def _fit_timestamp_regressor(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    *,
    sample_weight: np.ndarray | None,
    seed: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, Any | None, dict[str, Any], dict[str, Any] | None]:
    mask = np.isfinite(y_train)
    ids = np.flatnonzero(mask)
    if len(ids) < int(args.period_soft_min_train_timestamps):
        fill = float(np.nanmean(y_train[ids])) if len(ids) else 0.5
        diag = {"reason": "insufficient_period_rows", "train_rows": int(len(ids))}
        artifact = {
            "backend": "constant",
            "reason": diag["reason"],
            "fill_value": float(fill),
            "target_label": "new_period",
            "seed": int(seed),
        }
        return np.full(len(x_valid), fill, dtype=np.float32), None, diag, artifact
    keep_cols = [c for c in x_train.columns if pd.to_numeric(x_train[c], errors="coerce").notna().mean() > 0.02]
    if not keep_cols:
        fill = float(np.nanmean(y_train[ids]))
        diag = {"reason": "empty_period_matrix", "train_rows": int(len(ids))}
        artifact = {
            "backend": "constant",
            "reason": diag["reason"],
            "fill_value": float(fill),
            "target_label": "new_period",
            "seed": int(seed),
        }
        return np.full(len(x_valid), fill, dtype=np.float32), None, diag, artifact
    x_all = pd.concat([x_train, x_valid], axis=0, ignore_index=True, copy=False).replace([np.inf, -np.inf], np.nan)
    prepared = _prepare_model_matrix(x_all.loc[:, keep_cols])
    x_tr = prepared.iloc[: len(x_train)]
    x_va = prepared.iloc[len(x_train) :]
    min_child = max(8, int(math.ceil(float(args.period_soft_min_child_fraction) * len(ids))))
    reg = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=int(args.period_soft_n_estimators),
        learning_rate=0.035,
        max_depth=int(args.period_soft_max_depth),
        num_leaves=max(4, min(16, 2 ** int(args.period_soft_max_depth))),
        min_child_samples=int(min_child),
        subsample=0.90,
        colsample_bytree=0.90,
        reg_alpha=0.1,
        reg_lambda=2.0,
        random_state=int(seed),
        n_jobs=max(1, min(6, os.cpu_count() or 2)),
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )
    weights = sample_weight[ids] if sample_weight is not None else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg.fit(x_tr.iloc[ids], y_train[ids].astype(np.float32), sample_weight=weights)
    pred = np.clip(reg.predict(x_va), 0.0, 1.0).astype(np.float32, copy=False)
    diag = {
        "reason": "",
        "train_rows": int(len(ids)),
        "feature_count": int(len(keep_cols)),
        "min_child_samples": int(min_child),
    }
    artifact = {
        "backend": "lightweight_lgbm_fallback",
        "model": reg,
        "target_label": "new_period",
        "input_feature_columns": [str(c) for c in keep_cols],
        "feature_contract_sha256": _feature_contract_hash([str(c) for c in keep_cols]),
        "matrix_preparation": "diagnose_meta_recent_failures._prepare_model_matrix on persisted input_feature_columns; LightGBM handles missing values",
        "seed": int(seed),
        "params": reg.get_params(),
        "diagnostics": dict(diag),
    }
    return pred, reg, diag, artifact


def _fit_native_aux_regressor(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    *,
    sample_weight: np.ndarray | None,
    timestamps_train: pd.Series,
    assets_train: pd.Series | np.ndarray | None,
    seed: int,
    min_train_rows: int,
    label: str,
    head: str | None,
    scope: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, dict[str, Any], dict[str, Any] | None]:
    """Fit an auxiliary learner through the native train-meta LGBM pipeline.

    The production meta head is not retrained here. This wrapper only reuses the
    same LGBM stability feature-selection and HPO machinery for q_fail/new_period
    auxiliary scores, with their own targets and sample weights.
    """
    y = np.asarray(y_train, dtype=np.float32)
    mask = np.isfinite(y)
    if sample_weight is not None:
        sw_all = np.asarray(sample_weight, dtype=np.float32)
        mask &= np.isfinite(sw_all) & (sw_all > 0.0)
    else:
        sw_all = np.ones(len(y), dtype=np.float32)
    ids = np.flatnonzero(mask)
    if len(ids) < int(min_train_rows):
        return None, {
            "native_backend": "lgbm_stability_pipeline",
            "native_reason": "insufficient_native_aux_rows",
            "native_train_rows": int(len(ids)),
        }, None
    try:
        x_tr, x_va, keep_cols = fixed._prepare_train_pred_matrix(x_train, x_valid)
        if not keep_cols:
            return None, {
                "native_backend": "lgbm_stability_pipeline",
                "native_reason": "empty_native_aux_matrix",
                "native_train_rows": int(len(ids)),
            }, None
        preset_features, preset_params, preset_diag = _resolve_native_aux_feature_reuse(
            args,
            head=head,
            label=label,
            scope=scope,
            available_columns=[str(c) for c in x_tr.columns],
        )
        from extreme_price_movements.lgbm_pipeline import train_lgbm_stability_pipeline

        ts = pd.to_datetime(timestamps_train, utc=True, errors="coerce").reset_index(drop=True)
        assets = None
        if assets_train is not None:
            assets = pd.Series(assets_train).reset_index(drop=True).astype(str)
        model = train_lgbm_stability_pipeline(
            x_tr.iloc[ids].reset_index(drop=True),
            y[ids].astype(np.float32),
            sample_weight=sw_all[ids].astype(np.float32),
            random_state=int(seed),
            mode="regressor",
            timestamps=ts.iloc[ids].reset_index(drop=True),
            assets=(assets.iloc[ids].reset_index(drop=True) if assets is not None and len(assets) == len(y) else None),
            returns=y[ids].astype(np.float32),
            hard_labels=None,
            hpo_trials_override=int(args.aux_native_hpo_trials),
            hpo_patience_override=int(args.aux_native_hpo_patience),
            hpo_objective_mode="train_meta",
            preset_feature_names=preset_features,
            preset_best_params=preset_params,
            preset_source=str(preset_diag.get("source", "")) if preset_diag else None,
            reference_artifact_dir=None,
            cfg={
                "auxiliary_lgbm_label": str(label),
                "lgbm_regime_specialist_feature_engineering_lgbm_enabled": True,
                "lgbm_regime_specialist_feature_engineering_elasticnet_enabled": False,
                "lgbm_hpo_overrides": {
                    "max_depth_max": int(args.aux_native_max_depth),
                    "min_child_samples_pct_min": float(args.aux_native_min_child_pct_min),
                    "min_child_samples_pct_max": float(args.aux_native_min_child_pct_max),
                },
            },
        )
        if model is None:
            return None, {
                "native_backend": "lgbm_stability_pipeline",
                "native_reason": "native_pipeline_returned_none",
                "native_train_rows": int(len(ids)),
                "native_input_feature_count": int(len(keep_cols)),
            }, None
        pred = np.clip(model.predict(x_va), 0.0, 1.0).astype(np.float32, copy=False)
        metrics = dict(getattr(model, "metrics", {}) or {})
        selected = [str(c) for c in (getattr(model, "selected_features", []) or [])]
        diag = {
            "native_backend": "lgbm_stability_pipeline",
            "native_reason": "",
            "native_train_rows": int(len(ids)),
            "native_input_feature_count": int(len(keep_cols)),
            "native_selected_feature_count": int(len(selected)),
            "native_hpo_available": bool(metrics.get("hpo_available", False)),
            "native_hpo_completed_trials": int(metrics.get("hpo_completed_trials", 0) or 0),
            "native_hpo_best_value": float(metrics.get("hpo_best_value", np.nan)),
            "native_J_final": float(metrics.get("J_final", np.nan)),
            "native_auc_or_spearman": float(metrics.get("auc", np.nan)),
            "native_prune_rounds": int(metrics.get("feature_pruning_rounds_completed", 0) or 0),
            "native_aux_feature_selector": "lgbm_only",
            **{f"native_reuse_{k}": v for k, v in dict(preset_diag or {}).items()},
        }
        artifact = {
            "backend": "lgbm_stability_pipeline",
            "model": model,
            "target_label": str(label),
            "input_feature_columns": [str(c) for c in keep_cols],
            "selected_features": selected,
            "feature_contract_sha256": _feature_contract_hash([str(c) for c in keep_cols]),
            "selected_features_sha256": _feature_contract_hash(selected),
            "matrix_preparation": "scripts.run_fixed_band_qfail_ablation._prepare_train_pred_matrix persisted input_feature_columns; model._frame selects selected_features",
            "seed": int(seed),
            "hpo_objective_mode": "train_meta",
            "hpo_params": dict(getattr(model, "best_params", {}) or {}),
            "native_reuse": dict(preset_diag or {}),
            "metrics": metrics,
            "diagnostics": dict(diag),
        }
        return pred, diag, artifact
    except Exception as exc:
        return None, {
            "native_backend": "lgbm_stability_pipeline",
            "native_reason": f"native_pipeline_failed: {type(exc).__name__}: {exc}",
            "native_train_rows": int(len(ids)),
        }, None


def _period_sample_weights(
    target: np.ndarray,
    *,
    bottom_share: float,
    boost: float,
    ramp_share: float,
    ramp_power: float,
    badness_base_weight: float,
) -> np.ndarray:
    y = np.asarray(target, dtype=np.float64)
    w = np.ones(len(y), dtype=np.float64)
    finite = np.isfinite(y)
    if finite.any():
        badness = np.clip((y[finite] - 0.5) / 0.5, 0.0, 1.0)
        w[finite] *= 1.0 + float(badness_base_weight) * badness
        tail_pct = _percentile_from_train(y[finite], y).astype(np.float64)
        ramp_tail_share = max(float(bottom_share), float(ramp_share), 1e-6)
        ramp_start = max(0.0, 1.0 - ramp_tail_share)
        ramp = np.clip((tail_pct - ramp_start) / max(1.0 - ramp_start, 1e-6), 0.0, 1.0)
        ramp = np.power(ramp, max(float(ramp_power), 0.10))
        w *= 1.0 + (float(boost) - 1.0) * ramp
    w[~finite] = 0.0
    return np.clip(w, 0.0, 100.0).astype(np.float32)


def _choose_period_soft_hpo(
    z_train: pd.DataFrame,
    target_train: pd.Series,
    *,
    seed: int,
    args: argparse.Namespace,
) -> tuple[float, float, pd.DataFrame]:
    target = target_train.to_numpy(dtype=np.float32)
    rows: list[dict[str, Any]] = []
    order = np.arange(len(target))
    folds = np.array_split(order, max(2, int(args.period_soft_inner_folds)))
    for share in (0.05, 0.075, 0.10, 0.15):
        for boost in (2.0, 3.0, 4.0, 5.0):
            pred = np.full(len(target), np.nan, dtype=np.float32)
            for i, va in enumerate(folds):
                tr = np.setdiff1d(order, va, assume_unique=False)
                if len(tr) < int(args.period_soft_min_train_timestamps):
                    continue
                weights = _period_sample_weights(
                    target[tr],
                    bottom_share=share,
                    boost=boost,
                    ramp_share=float(args.period_soft_tail_ramp_share),
                    ramp_power=float(args.period_soft_tail_ramp_power),
                    badness_base_weight=float(args.period_soft_badness_base_weight),
                )
                p, _model, _diag, _artifact = _fit_timestamp_regressor(
                    z_train.iloc[tr].reset_index(drop=True),
                    target[tr],
                    z_train.iloc[va].reset_index(drop=True),
                    sample_weight=weights,
                    seed=int(seed + 17 * i + int(1000 * share) + int(10 * boost)),
                    args=args,
                )
                pred[va] = p
            fill = float(np.nanmean(target[np.isfinite(target)])) if np.isfinite(target).any() else 0.5
            pred[~np.isfinite(pred)] = fill
            metrics = _period_tail_detection_metrics(target, pred)
            objective = float(
                0.45 * metrics["ap_lift05"]
                + 0.25 * metrics["ap_lift10"]
                + 0.15 * metrics["recall10"]
                + 0.10 * metrics["ndcg10"]
                + 0.05 * metrics["difficulty_decile_spread"]
            )
            labels = _period_tail_labels(target)
            rows.append(
                {
                    "bottom_share": share,
                    "boost": boost,
                    "period_hpo_objective": objective,
                    "period_bad05_rate": float(np.nanmean(labels["period_bad_05"])),
                    "period_bad10_rate": float(np.nanmean(labels["period_bad_10"])),
                    "period_bad15_rate": float(np.nanmean(labels["period_bad_15"])),
                    "period_tail_severity_mean": float(np.nanmean(labels["period_tail_severity"])),
                    **metrics,
                }
            )
    table = pd.DataFrame(rows).sort_values("period_hpo_objective", ascending=False)
    if table.empty:
        return 0.10, 3.0, table
    best = table.iloc[0]
    return float(best["bottom_share"]), float(best["boost"]), table


def _fit_soft_qfail_regressor(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    anchor_train: np.ndarray,
    rank_train: np.ndarray,
    x_valid: pd.DataFrame,
    *,
    timestamps_train: pd.Series,
    assets_train: pd.Series | np.ndarray | None,
    seed: int,
    head: str | None,
    scope: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any] | None]:
    y = np.asarray(y_train, dtype=np.float64)
    p = np.clip(np.asarray(anchor_train, dtype=np.float64), 0.0, 1.0)
    r = np.asarray(rank_train, dtype=np.float64)
    mask = (y >= 0) & np.isfinite(p) & np.isfinite(r) & (r >= float(args.qfail_soft_rank_threshold))
    target = np.full(len(y), np.nan, dtype=np.float32)
    target[mask] = ((1.0 - y[mask]) * p[mask]).astype(np.float32)
    if int(mask.sum()) < int(args.qfail_soft_min_train_rows):
        fill = float(np.nanmean(target[mask])) if mask.any() else 0.0
        diag = {"reason": "insufficient_soft_qfail_rows", "train_rows": int(mask.sum())}
        artifact = {
            "backend": "constant",
            "reason": diag["reason"],
            "fill_value": float(fill),
            "target_label": "qfail_soft",
            "seed": int(seed),
        }
        return np.full(len(x_valid), fill, dtype=np.float32), diag, artifact
    x_tr, x_va, keep_cols = fixed._prepare_train_pred_matrix(x_train, x_valid)
    if not keep_cols:
        fill = float(np.nanmean(target[mask]))
        diag = {"reason": "empty_soft_qfail_matrix", "train_rows": int(mask.sum())}
        artifact = {
            "backend": "constant",
            "reason": diag["reason"],
            "fill_value": float(fill),
            "target_label": "qfail_soft",
            "seed": int(seed),
        }
        return np.full(len(x_valid), fill, dtype=np.float32), diag, artifact
    weights = fixed._equal_timestamp_weights(timestamps_train, mask)
    weights *= (1.0 + 1.0 * ((r >= 0.70) & mask).astype(np.float32) + 0.5 * ((r >= 0.50) & (r < 0.70) & mask).astype(np.float32))
    if bool(args.aux_native_lgbm):
        pred_native, native_diag, native_artifact = _fit_native_aux_regressor(
            x_train,
            target,
            x_valid,
            sample_weight=np.where(weights > 0.0, weights, 0.0).astype(np.float32),
            timestamps_train=timestamps_train,
            assets_train=assets_train,
            seed=int(seed),
            min_train_rows=int(args.qfail_soft_min_train_rows),
            label="qfail_soft",
            head=head,
            scope=scope,
            args=args,
        )
        if pred_native is not None:
            native_diag["reason"] = ""
            native_diag["train_rows"] = int(mask.sum())
            native_diag["target_mean"] = float(np.nanmean(target[mask]))
            native_diag["selection_objective"] = "native_train_meta_stability_hpo_on_soft_failure_magnitude"
            if native_artifact is not None:
                native_artifact["selection_objective"] = native_diag["selection_objective"]
                native_artifact["target_definition"] = "(1-y_bin) * anchor_score inside anchor top50 rank>=0.50"
                native_artifact["diagnostics"] = dict(native_diag)
            return pred_native, native_diag, native_artifact
    ids = np.flatnonzero(mask)
    min_child = max(50, int(math.ceil(float(args.qfail_soft_min_child_fraction) * len(ids))))
    reg = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=int(args.qfail_soft_n_estimators),
        learning_rate=0.035,
        max_depth=int(args.qfail_soft_max_depth),
        num_leaves=max(4, min(16, 2 ** int(args.qfail_soft_max_depth))),
        min_child_samples=int(min_child),
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=2.0,
        random_state=int(seed),
        n_jobs=max(1, min(6, os.cpu_count() or 2)),
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg.fit(x_tr.iloc[ids], target[ids], sample_weight=np.where(weights[ids] > 0.0, weights[ids], 1.0))
    pred = np.clip(reg.predict(x_va), 0.0, 1.0).astype(np.float32, copy=False)
    diag = {
        "reason": "",
        "train_rows": int(len(ids)),
        "feature_count": int(len(keep_cols)),
        "min_child_samples": int(min_child),
        "target_mean": float(np.nanmean(target[ids])),
    }
    artifact = {
        "backend": "lightweight_lgbm_fallback",
        "model": reg,
        "target_label": "qfail_soft",
        "target_definition": "(1-y_bin) * anchor_score inside anchor top50 rank>=0.50",
        "input_feature_columns": [str(c) for c in keep_cols],
        "feature_contract_sha256": _feature_contract_hash([str(c) for c in keep_cols]),
        "matrix_preparation": "scripts.run_fixed_band_qfail_ablation._prepare_train_pred_matrix persisted input_feature_columns; LightGBM handles missing values",
        "seed": int(seed),
        "params": reg.get_params(),
        "diagnostics": dict(diag),
    }
    return pred, diag, artifact


def _timestamp_anchor_state(timestamps: pd.Series, anchor_score: np.ndarray, rank0: np.ndarray) -> pd.DataFrame:
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").reset_index(drop=True)
    p = np.clip(np.asarray(anchor_score, dtype=np.float32), 1e-6, 1.0 - 1e-6)
    r = np.clip(np.asarray(rank0, dtype=np.float32), 0.0, 1.0)
    entropy = -(p * np.log(p) + (1.0 - p) * np.log1p(-p))
    frame = pd.DataFrame({"timestamp": ts, "p": p, "rank": r, "entropy": entropy.astype(np.float32)})
    if frame.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
    frame["is_top50"] = (frame["rank"] >= 0.50).astype(np.float32)
    frame["is_top70"] = (frame["rank"] >= 0.70).astype(np.float32)
    frame["is_top80"] = (frame["rank"] >= 0.80).astype(np.float32)
    frame["is_top90"] = (frame["rank"] >= 0.90).astype(np.float32)
    frame["uncertainty"] = (frame["p"] * (1.0 - frame["p"])).astype(np.float32)
    grouped = frame.groupby("timestamp", sort=True, dropna=False)
    state = grouped.agg(
        anchor_ts_rows=("p", "size"),
        anchor_ts_score_mean=("p", "mean"),
        anchor_ts_score_std=("p", "std"),
        anchor_ts_score_q10=("p", lambda s: float(np.nanquantile(s, 0.10))),
        anchor_ts_score_q50=("p", "median"),
        anchor_ts_score_q90=("p", lambda s: float(np.nanquantile(s, 0.90))),
        anchor_ts_rank_mean=("rank", "mean"),
        anchor_ts_rank_std=("rank", "std"),
        anchor_ts_rank_q10=("rank", lambda s: float(np.nanquantile(s, 0.10))),
        anchor_ts_rank_q50=("rank", "median"),
        anchor_ts_rank_q90=("rank", lambda s: float(np.nanquantile(s, 0.90))),
        anchor_ts_share_rank_ge50=("is_top50", "mean"),
        anchor_ts_share_rank_ge70=("is_top70", "mean"),
        anchor_ts_share_rank_ge80=("is_top80", "mean"),
        anchor_ts_share_rank_ge90=("is_top90", "mean"),
        anchor_ts_uncertainty_mean=("uncertainty", "mean"),
        anchor_ts_uncertainty_q90=("uncertainty", lambda s: float(np.nanquantile(s, 0.90))),
        anchor_ts_entropy_mean=("entropy", "mean"),
        anchor_ts_entropy_q90=("entropy", lambda s: float(np.nanquantile(s, 0.90))),
    )
    top30 = frame[frame["rank"] >= 0.70].groupby("timestamp", sort=True)
    if len(top30):
        top_state = top30.agg(
            anchor_ts_top30_score_mean=("p", "mean"),
            anchor_ts_top30_score_std=("p", "std"),
            anchor_ts_top30_rank_mean=("rank", "mean"),
        )
        state = state.join(top_state, how="left")
    state["anchor_ts_log_rows"] = np.log1p(state["anchor_ts_rows"].astype(np.float32))
    state["anchor_ts_score_q90_q10"] = state["anchor_ts_score_q90"] - state["anchor_ts_score_q10"]
    state["anchor_ts_rank_q90_q10"] = state["anchor_ts_rank_q90"] - state["anchor_ts_rank_q10"]
    state = state.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return _downcast_numeric(state)


_SPECTRAL_SOURCE_EXCLUDE_TOKENS = (
    "target",
    "label",
    "future",
    "realized",
    "pnl",
    "barrier",
    "candidate",
    "qfail",
    "fail",
    "period_bad",
    "period_tail",
)


def _append_fold_spectral_position_features(
    z_train_base: pd.DataFrame,
    z_valid_base: pd.DataFrame,
    *,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not bool(getattr(args, "period_spectral_features", True)):
        return z_train_base, z_valid_base.reindex(columns=z_train_base.columns), {
            "spectral_enabled": False,
            "spectral_feature_count": 0,
            "spectral_source_feature_count": 0,
        }
    train = z_train_base.copy()
    valid = z_valid_base.copy()
    train.index = pd.to_datetime(train.index, utc=True, errors="coerce")
    valid.index = pd.to_datetime(valid.index, utc=True, errors="coerce")
    source_cols: list[str] = []
    for col in train.columns:
        low = str(col).lower()
        if any(tok in low for tok in _SPECTRAL_SOURCE_EXCLUDE_TOKENS):
            continue
        if str(col).startswith(("anchor_ts_", "q_period_", "q_fail_", "rank_", "score_")):
            continue
        vals = pd.to_numeric(train[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if float(vals.notna().mean()) < 0.05:
            continue
        var = float(vals.var(ddof=0)) if vals.notna().sum() > 1 else 0.0
        if np.isfinite(var) and var > 1e-12:
            source_cols.append(str(col))
    if len(source_cols) < 2:
        aligned_valid = valid.reindex(columns=train.columns)
        return train, aligned_valid, {
            "spectral_enabled": True,
            "spectral_reason": "insufficient_source_columns",
            "spectral_feature_count": 0,
            "spectral_source_feature_count": int(len(source_cols)),
        }
    max_features = int(getattr(args, "period_spectral_max_features", 64))
    source_cols = source_cols[: max(2, max_features)]

    def _to_spectral_frame(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.loc[:, [c for c in source_cols if c in frame.columns]].copy()
        out.insert(0, "timestamp", pd.to_datetime(frame.index, utc=True, errors="coerce"))
        return out.reset_index(drop=True)

    cfg = MarketSpectralPositionConfig(
        lookback=int(getattr(args, "period_spectral_lookback", 48)),
        min_periods=int(getattr(args, "period_spectral_min_periods", 24)),
        top_k=int(getattr(args, "period_spectral_top_k", 3)),
        max_features=max_features,
        shrinkage=float(getattr(args, "period_spectral_shrinkage", 0.10)),
        prefix="state_spectral_",
    )
    encoder = fit_market_spectral_position_encoder(
        _to_spectral_frame(train),
        timestamp_col="timestamp",
        feature_columns=source_cols,
        config=cfg,
    )
    spectral_train = transform_market_spectral_position(_to_spectral_frame(train), encoder).set_index("timestamp")
    spectral_valid = transform_market_spectral_position(_to_spectral_frame(valid), encoder).set_index("timestamp")
    spectral_train = spectral_train.reindex(train.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    spectral_valid = spectral_valid.reindex(valid.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    train_out = _downcast_numeric(train.join(spectral_train, how="left").replace([np.inf, -np.inf], np.nan).fillna(0.0))
    valid_out = _downcast_numeric(
        valid.join(spectral_valid, how="left")
        .reindex(columns=train_out.columns)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    spectral_cols = [c for c in train_out.columns if str(c).startswith("state_spectral_")]
    return train_out, valid_out, {
        "spectral_enabled": True,
        "spectral_reason": "",
        "spectral_feature_count": int(len(spectral_cols)),
        "spectral_source_feature_count": int(len(encoder.get("feature_columns") or [])),
        "spectral_lookback": int(cfg.lookback),
        "spectral_min_periods": int(cfg.min_periods),
        "spectral_output_columns": list(spectral_cols),
    }


def _lagged_by_symbol(
    timestamps: pd.Series,
    symbols: pd.Series | np.ndarray,
    values: np.ndarray,
    *,
    lags: tuple[int, ...],
    prefix: str,
) -> pd.DataFrame:
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").reset_index(drop=True)
    sym = pd.Series(symbols).reset_index(drop=True).astype(str)
    vals = pd.to_numeric(pd.Series(values), errors="coerce").astype("float32")
    frame = pd.DataFrame({"timestamp": ts, "symbol": sym, "value": vals})
    order = frame.sort_values(["symbol", "timestamp"], kind="mergesort").index.to_numpy(dtype=np.int64)
    sorted_frame = frame.loc[order].reset_index()
    grouped = sorted_frame.groupby("symbol", sort=False)["value"]
    out_sorted = pd.DataFrame({"index": sorted_frame["index"].to_numpy(dtype=np.int64)})
    for lag in lags:
        shifted = grouped.shift(int(lag))
        out_sorted[f"{prefix}_diff_{lag}obs_by_symbol"] = (sorted_frame["value"] - shifted).to_numpy(dtype=np.float32)
    prev = grouped.shift(1)
    trailing_mean = prev.groupby(sorted_frame["symbol"], sort=False).rolling(24, min_periods=3).mean().reset_index(level=0, drop=True)
    trailing_std = prev.groupby(sorted_frame["symbol"], sort=False).rolling(24, min_periods=3).std(ddof=0).reset_index(level=0, drop=True)
    out_sorted[f"{prefix}_minus_prev24_mean_by_symbol"] = (sorted_frame["value"] - trailing_mean).to_numpy(dtype=np.float32)
    out_sorted[f"{prefix}_prev24_std_by_symbol"] = trailing_std.to_numpy(dtype=np.float32)
    out = out_sorted.set_index("index").reindex(np.arange(len(frame))).reset_index(drop=True)
    return _downcast_numeric(out.replace([np.inf, -np.inf], np.nan).fillna(0.0))


def _anchor_meta_drift_features(
    timestamps: pd.Series,
    symbols: pd.Series | np.ndarray,
    anchor_score: np.ndarray,
    rank0: np.ndarray,
) -> pd.DataFrame:
    score = _lagged_by_symbol(timestamps, symbols, anchor_score, lags=(1, 4, 24), prefix="metaout_anchor_score")
    rank = _lagged_by_symbol(timestamps, symbols, rank0, lags=(1, 4, 24), prefix="metaout_anchor_rank")
    return stack._combine_features(score, rank)


def _select_meta_output_columns(raw: pd.DataFrame, *, max_cols: int) -> list[str]:
    if raw.empty or int(max_cols) <= 0:
        return []
    scored_by_col: dict[str, float] = {}
    for col in raw.columns:
        low = str(col).lower()
        if any(tok in low for tok in META_OUTPUT_FORBIDDEN_TOKENS):
            continue
        if not any(tok in low for tok in META_OUTPUT_TOKENS):
            continue
        vals = pd.to_numeric(raw[col], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(vals)
        coverage = float(finite.mean()) if len(vals) else 0.0
        if coverage < 0.05:
            continue
        var = float(np.nanvar(vals[finite])) if finite.any() else 0.0
        scored_by_col[str(col)] = coverage * math.log1p(max(var, 0.0))
    if not scored_by_col:
        return []

    def _ranked_matching(tokens: tuple[str, ...]) -> list[str]:
        cols = [c for c in scored_by_col if any(tok in c.lower() for tok in tokens)]
        return sorted(cols, key=lambda c: (scored_by_col[c], c), reverse=True)

    cap = int(max_cols)
    per_bucket = max(1, cap // 4)
    selected: list[str] = []
    for bucket in (
        META_OUTPUT_UNCERTAINTY_TOKENS,
        META_OUTPUT_DRIFT_TOKENS,
        META_OUTPUT_SUPPORT_TOKENS,
    ):
        for col in _ranked_matching(bucket)[:per_bucket]:
            if col not in selected:
                selected.append(col)
    for col in sorted(scored_by_col, key=lambda c: (scored_by_col[c], c), reverse=True):
        if col not in selected:
            selected.append(col)
        if len(selected) >= cap:
            break
    return selected[:cap]


def _meta_output_extra_features(
    raw: pd.DataFrame,
    timestamps: pd.Series,
    symbols: pd.Series | np.ndarray,
    *,
    selected_cols: list[str],
    max_derived_cols: int,
) -> pd.DataFrame:
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").reset_index(drop=True)
    parts: list[pd.DataFrame] = []
    base_values: dict[str, np.ndarray] = {}
    for col in selected_cols:
        if col not in raw.columns:
            continue
        vals = pd.to_numeric(raw[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        safe_name = str(col).replace("__", "_")
        base_values[f"metaout__{safe_name}"] = vals
    if base_values:
        parts.append(pd.DataFrame(base_values))
    derived_cols = selected_cols[: int(max_derived_cols)]
    for col in derived_cols:
        if col not in raw.columns:
            continue
        vals = pd.to_numeric(raw[col], errors="coerce").astype("float32")
        safe_name = str(col).replace("__", "_")
        frame = pd.DataFrame({"timestamp": ts, "value": vals})
        grouped = frame.groupby("timestamp", sort=False)["value"]
        mean = grouped.transform("mean").fillna(0.0).to_numpy(dtype=np.float32)
        std = grouped.transform("std").fillna(0.0).to_numpy(dtype=np.float32)
        parts.append(
            pd.DataFrame(
                {
                    f"metaout__{safe_name}__minus_ts_mean": (vals.to_numpy(dtype=np.float32) - mean).astype(np.float32),
                    f"metaout__{safe_name}__z_ts": ((vals.to_numpy(dtype=np.float32) - mean) / np.maximum(std, 1e-4)).astype(np.float32),
                    f"metaout__{safe_name}__rank_ts": stack._rank_pct_by_timestamp(ts, vals.to_numpy(dtype=np.float32)),
                }
            )
        )
        if any(
            tok in str(col).lower()
            for tok in (
                "drift",
                "uncert",
                "entropy",
                "path",
                "dae",
                "mahal",
                "cluster",
                "gmm",
                "variance",
                "std",
                "range",
                "margin",
                "contrib",
                "score_100",
                "rank_100",
                "reversal",
                "support",
                "leaf",
            )
        ):
            parts.append(_lagged_by_symbol(timestamps, symbols, vals.to_numpy(dtype=np.float32), lags=(1, 4, 24), prefix=f"metaout__{safe_name}"))
    if not parts:
        return pd.DataFrame(index=raw.index)
    return _downcast_numeric(pd.concat([p.reset_index(drop=True) for p in parts], axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0))


def _anchor_extra_features(timestamps: pd.Series, anchor_score: np.ndarray, rank0: np.ndarray) -> pd.DataFrame:
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").reset_index(drop=True)
    p = np.clip(np.asarray(anchor_score, dtype=np.float32), 1e-6, 1.0 - 1e-6)
    r = np.clip(np.asarray(rank0, dtype=np.float32), 0.0, 1.0)
    entropy = -(p * np.log(p) + (1.0 - p) * np.log1p(-p))
    frame = pd.DataFrame({"timestamp": ts, "p": p, "rank": r})
    grp_p = frame.groupby("timestamp", sort=False)["p"]
    grp_r = frame.groupby("timestamp", sort=False)["rank"]
    mean = grp_p.transform("mean").to_numpy(dtype=np.float32)
    std = grp_p.transform("std").fillna(0.0).to_numpy(dtype=np.float32)
    q10 = grp_p.transform(lambda s: float(np.nanquantile(s, 0.10))).to_numpy(dtype=np.float32)
    q90 = grp_p.transform(lambda s: float(np.nanquantile(s, 0.90))).to_numpy(dtype=np.float32)
    rank_mean = grp_r.transform("mean").to_numpy(dtype=np.float32)
    rank_std = grp_r.transform("std").fillna(0.0).to_numpy(dtype=np.float32)
    count = grp_p.transform("size").to_numpy(dtype=np.float32)
    return _downcast_numeric(
        pd.DataFrame(
            {
                "anchor_uncertainty_p1mp": (p * (1.0 - p)).astype(np.float32),
                "anchor_entropy": entropy.astype(np.float32),
                "anchor_abs_distance_from_half": np.abs(p - 0.5).astype(np.float32),
                "anchor_rank0": r.astype(np.float32),
                "anchor_rank0_sq": np.square(r).astype(np.float32),
                "anchor_rank0_sqrt": np.sqrt(r).astype(np.float32),
                "anchor_rank_gap_to_70": (r - 0.70).astype(np.float32),
                "anchor_rank_gap_to_90": (r - 0.90).astype(np.float32),
                "anchor_is_top10": (r >= 0.90).astype(np.float32),
                "anchor_is_top20": (r >= 0.80).astype(np.float32),
                "anchor_is_top30": (r >= 0.70).astype(np.float32),
                "anchor_is_candidate_30_50": ((r >= 0.50) & (r < 0.70)).astype(np.float32),
                "anchor_score_minus_timestamp_mean": (p - mean).astype(np.float32),
                "anchor_score_z_timestamp": ((p - mean) / np.maximum(std, 1e-4)).astype(np.float32),
                "anchor_score_distance_to_ts_q90": (p - q90).astype(np.float32),
                "anchor_score_distance_to_ts_q10": (p - q10).astype(np.float32),
                "anchor_rank_minus_timestamp_mean": (r - rank_mean).astype(np.float32),
                "anchor_rank_z_timestamp": ((r - rank_mean) / np.maximum(rank_std, 1e-4)).astype(np.float32),
                "anchor_timestamp_score_std": std.astype(np.float32),
                "anchor_timestamp_rank_std": rank_std.astype(np.float32),
                "anchor_timestamp_log_row_count": np.log1p(count).astype(np.float32),
            }
        )
    )


def _select_qfail_context_columns(structural: pd.DataFrame, *, max_context_cols: int) -> list[str]:
    if structural.empty or int(max_context_cols) <= 0:
        return []
    context_markers = (
        "prediction_support_quality",
        "prediction_reconstruction_anomaly",
        "prediction_path_instability",
        "regime_similarity_or_novelty",
        "leverage_funding_crowding",
        "liquidity_participation_stress",
        "tail_volatility_stress",
        "relative_value_dislocation",
        "breadth_market_state",
        "network_concentration",
        "leaf_support",
        "leaf_occupancy_novelty",
        "leaf_path_rarity",
        "leaf_depth",
        "leaf_structural_uncertainty",
        "state_spectral_",
    )
    candidates = [
        c
        for c in structural.columns
        if any(marker in str(c) for marker in context_markers)
        and pd.to_numeric(structural[c], errors="coerce").notna().mean() > 0.02
    ]
    if not candidates:
        return []
    variances = pd.to_numeric(structural[candidates].var(axis=0, numeric_only=True), errors="coerce").fillna(0.0)
    return variances.sort_values(ascending=False).head(int(max_context_cols)).index.tolist()


def _qfail_context_interactions(
    structural: pd.DataFrame,
    extra: pd.DataFrame,
    *,
    context_cols: list[str],
) -> pd.DataFrame:
    if structural.empty or extra.empty or not context_cols:
        return pd.DataFrame(index=extra.index)
    anchor_cols = [
        c
        for c in (
            "anchor_rank0",
            "anchor_rank_gap_to_70",
            "anchor_is_top30",
            "anchor_is_candidate_30_50",
            "anchor_score_z_timestamp",
            "anchor_uncertainty_p1mp",
        )
        if c in extra.columns
    ]
    if not anchor_cols:
        return pd.DataFrame(index=extra.index)
    out: dict[str, np.ndarray] = {}
    for c in context_cols:
        if c not in structural.columns:
            continue
        base = pd.to_numeric(structural[c], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        for a in anchor_cols:
            av = pd.to_numeric(extra[a], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            out[f"qfail_ix__{a}__x__{c}"] = (av * base).astype(np.float32, copy=False)
    return _downcast_numeric(pd.DataFrame(out, index=extra.index))


def _normalised_live_contract_oof_name(name: str) -> str:
    raw = str(name)
    for prefix in ("control__", "metaout__", "export__", "metaout__export_", "control__export__"):
        if raw.startswith(prefix):
            raw = raw[len(prefix) :]
    if raw.startswith("export_oof_"):
        raw = "oof_" + raw[len("export_oof_") :]
    if raw.startswith("export__"):
        raw = raw[len("export__") :]
    if not raw.startswith("oof_") and raw in {"lgbm_prob", "meta_clf", "base_clf", "p_move", "rank_pct"}:
        raw = f"oof_{raw}"
    return raw


def _is_live_contract_oof_feature(name: str) -> bool:
    return _normalised_live_contract_oof_name(name) in LIVE_CONTRACT_OOF_FEATURES


def _live_contract_component_source_features(
    *,
    raw: pd.DataFrame,
    timestamps: pd.Series,
    symbols: pd.Series | np.ndarray,
    anchor_score: np.ndarray,
    rank0: np.ndarray,
    max_control_path_features: int,
    max_meta_output_features: int,
    max_meta_output_derived_features: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build native component inputs that the live feature ledger can resolve."""
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").reset_index(drop=True)
    sym = pd.Series(symbols).reset_index(drop=True).astype(str)
    deployable_path_cols = [
        c
        for c in raw.columns
        if _is_live_contract_oof_feature(c)
    ][: int(max_control_path_features)]
    anchor_ctrl, path_cols = anchored._anchor_control_features(
        raw,
        ts,
        anchor_score,
        max_path_cols=int(max_control_path_features),
        selected_cols=deployable_path_cols,
    )
    extra = _anchor_extra_features(ts, anchor_score, rank0)
    drift = _anchor_meta_drift_features(ts, sym, anchor_score, rank0)
    metaout_cols = [
        c
        for c in _select_meta_output_columns(raw, max_cols=max(int(max_meta_output_features) * 4, int(max_meta_output_features)))
        if _is_live_contract_oof_feature(c)
    ][: int(max_meta_output_features)]
    metaout = _meta_output_extra_features(
        raw,
        ts,
        sym,
        selected_cols=metaout_cols,
        max_derived_cols=int(max_meta_output_derived_features),
    )
    out = stack._combine_features(anchor_ctrl, extra, drift, metaout)
    return out, {
        "live_contract_path_control_cols": int(len(path_cols)),
        "live_contract_metaout_cols": int(len(metaout_cols)),
        "live_contract_feature_count": int(out.shape[1]),
    }


def _fit_full_fit_live_contract_components(
    *,
    head: str,
    panel: pd.DataFrame,
    raw: pd.DataFrame,
    y: np.ndarray,
    anchor_score: np.ndarray,
    rank0: np.ndarray,
    args: argparse.Namespace,
    feature_manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ts = pd.to_datetime(panel["timestamp"], utc=True, errors="coerce").reset_index(drop=True)
    symbols = panel["symbol"].reset_index(drop=True).astype(str) if "symbol" in panel.columns else pd.Series([""] * len(panel))
    source, source_diag = _live_contract_component_source_features(
        raw=raw.reset_index(drop=True),
        timestamps=ts,
        symbols=symbols,
        anchor_score=anchor_score,
        rank0=rank0,
        max_control_path_features=int(args.max_control_path_features),
        max_meta_output_features=int(args.qfail_meta_output_max_features),
        max_meta_output_derived_features=int(args.qfail_meta_output_max_derived_features),
    )
    qfail_context_cols = _select_qfail_context_columns(
        source,
        max_context_cols=int(args.qfail_interaction_max_context_features),
    )
    extra_cols = [
        c
        for c in (
            "anchor_rank0",
            "anchor_rank_gap_to_70",
            "anchor_is_top30",
            "anchor_is_candidate_30_50",
            "anchor_score_z_timestamp",
            "anchor_uncertainty_p1mp",
        )
        if c in source.columns
    ]
    qfail_ix = _qfail_context_interactions(
        source,
        source.loc[:, extra_cols] if extra_cols else source,
        context_cols=qfail_context_cols,
    )
    qfail_x = stack._combine_features(source, qfail_ix)
    feature_manifest["features"].setdefault(head, {})["full_fit_live_contract_qfail_row"] = sorted(map(str, qfail_x.columns))

    artifacts: list[dict[str, Any]] = []
    diag: dict[str, Any] = {
        **source_diag,
        "live_contract_qfail_context_cols": int(len(qfail_context_cols)),
        "live_contract_qfail_feature_count": int(qfail_x.shape[1]),
    }

    qfail_valid = qfail_x.iloc[: max(1, min(16, len(qfail_x)))].reset_index(drop=True)
    soft_pred, soft_diag, soft_artifact = _fit_soft_qfail_regressor(
        qfail_x.reset_index(drop=True),
        y,
        anchor_score,
        rank0,
        qfail_valid,
        timestamps_train=ts,
        assets_train=symbols,
        seed=int(args.seed + _stable_seed_offset(head, "full_fit_qfail_soft")),
        head=head,
        scope="full_fit",
        args=args,
    )
    diag.update({f"full_fit_qfail_soft_{k}": v for k, v in soft_diag.items()})
    diag["full_fit_qfail_soft_preview_rows"] = int(len(soft_pred))
    if soft_artifact is not None:
        soft_artifact.update(
            {
                "head": head,
                "fold": "full_fit",
                "model_scope": "full_fit",
                "component": "qfail_soft",
                "live_feature_contract": "live_contract_safe_v1",
                "train_timestamps": _timestamp_bounds(ts),
                "target_definition": "(1-y_bin) * anchor_score inside anchor top50 rank>=0.50",
                "diagnostics": dict(soft_diag),
            }
        )
        artifacts.append(soft_artifact)

    timestamp_base = stack._timestamp_feature_table(
        source,
        ts,
        max_columns=int(args.max_timestamp_features),
    )
    timestamp_market, _timestamp_preview, spectral_diag = _append_fold_spectral_position_features(
        timestamp_base,
        timestamp_base.iloc[: max(1, min(16, len(timestamp_base)))],
        args=args,
    )
    timestamp_source = timestamp_market.join(_timestamp_anchor_state(ts, anchor_score, rank0), how="left")
    timestamp_source = _downcast_numeric(timestamp_source.replace([np.inf, -np.inf], np.nan).fillna(0.0))
    diag["live_contract_period_spectral_feature_count"] = int(spectral_diag.get("spectral_feature_count", 0))
    diag["live_contract_period_spectral_source_feature_count"] = int(
        spectral_diag.get("spectral_source_feature_count", 0)
    )
    feature_manifest["features"].setdefault(head, {})["full_fit_live_contract_period_timestamp"] = sorted(
        map(str, timestamp_source.columns)
    )
    target_raw = _soft_difficulty_target(
        ts,
        y,
        anchor_score,
        rank0,
        rank_threshold=float(args.period_soft_rank_threshold),
        horizon_hours=int(args.period_soft_horizon_hours),
        halflife_hours=float(args.period_soft_halflife_hours),
    ).reindex(timestamp_source.index)
    target = pd.Series(
        _percentile_from_train(
            target_raw.to_numpy(dtype=np.float32),
            target_raw.to_numpy(dtype=np.float32),
            nonfinite_fill=None,
        ),
        index=target_raw.index,
        dtype="float32",
    )
    share, boost, hpo_table = _choose_period_soft_hpo(
        timestamp_source,
        target,
        seed=int(args.seed + _stable_seed_offset(head, "full_fit_period_hpo")),
        args=args,
    )
    weights = _period_sample_weights(
        target.to_numpy(dtype=np.float32),
        bottom_share=float(share),
        boost=float(boost),
        ramp_share=float(args.period_soft_tail_ramp_share),
        ramp_power=float(args.period_soft_tail_ramp_power),
        badness_base_weight=float(args.period_soft_badness_base_weight),
    )
    period_target = target.to_numpy(dtype=np.float32)
    period_transform = "soft_future_error_percentile"
    if str(args.period_soft_selection_target).strip().lower() == "tail_severity":
        period_target = _period_tail_labels(period_target)["period_tail_severity"].astype(np.float32)
        period_transform = "tail_severity_from_soft_future_error"
    period_valid = timestamp_source.iloc[: max(1, min(16, len(timestamp_source)))].reset_index(drop=True)
    period_pred = None
    period_diag: dict[str, Any]
    period_artifact: dict[str, Any] | None = None
    if bool(args.aux_native_lgbm):
        period_pred, period_diag, period_artifact = _fit_native_aux_regressor(
            timestamp_source.reset_index(drop=True),
            period_target,
            period_valid,
            sample_weight=weights,
            timestamps_train=pd.Series(timestamp_source.index),
            assets_train=None,
            seed=int(args.seed + _stable_seed_offset(head, "full_fit_new_period")),
            min_train_rows=int(args.period_soft_min_train_timestamps),
            label="new_period",
            head=head,
            scope="full_fit",
            args=args,
        )
    if period_pred is None:
        period_pred, _period_model, period_diag, period_artifact = _fit_timestamp_regressor(
            timestamp_source.reset_index(drop=True),
            period_target,
            period_valid,
            sample_weight=weights,
            seed=int(args.seed + _stable_seed_offset(head, "full_fit_new_period_fallback")),
            args=args,
        )
        period_diag["native_backend"] = "lightweight_lgbm_fallback"
    period_diag["selection_objective"] = f"full_fit_live_contract_on_{period_transform}"
    diag.update({f"full_fit_new_period_{k}": v for k, v in period_diag.items()})
    diag["full_fit_new_period_preview_rows"] = int(len(period_pred) if period_pred is not None else 0)
    diag["full_fit_period_hpo_rows"] = int(len(hpo_table))
    diag["full_fit_period_selected_bottom_share"] = float(share)
    diag["full_fit_period_selected_boost"] = float(boost)
    if period_artifact is not None:
        period_artifact.update(
            {
                "head": head,
                "fold": "full_fit",
                "model_scope": "full_fit",
                "component": "new_period",
                "live_feature_contract": "live_contract_safe_v1",
                "train_timestamps": _timestamp_bounds(pd.Series(timestamp_source.index)),
                "period_soft_selected_bottom_share": float(share),
                "period_soft_selected_boost": float(boost),
                "target_definition": feature_manifest["period_new_target"],
                "target_transform": period_transform,
                "diagnostics": dict(period_diag),
            }
        )
        artifacts.append(period_artifact)
    return artifacts, diag


def _load_heads(args: argparse.Namespace) -> tuple[list[Any], Any, Any, list[str]]:
    meta_artifact_dir = Path(args.meta_artifact_dir)
    baseline_artifact_dir = Path(args.baseline_artifact_dir)
    report_dir = Path(args.report_dir)
    meta_state = joblib.load(meta_artifact_dir / "models" / "model_state_meta.pkl")
    meta_models = meta_state["bundle"]["meta_models"]
    heads = _discover_heads(meta_artifact_dir, report_dir, meta_models)
    wanted = set(str(x) for x in (args.only_head or HEADS))
    heads = [h for h in heads if h.head in wanted]
    with (baseline_artifact_dir / "base_models_intermediate.pkl").open("rb") as fh:
        base_bundle = pickle.load(fh)
    return heads, meta_models, base_bundle, _feature_store_union(Path(args.feature_dir))


def run(args: argparse.Namespace) -> Path:
    out_dir = _ensure_dir(Path(args.output_dir))
    canonical_defs = canon._load_canonical_definitions(Path(args.canonical_reduction))
    if not canonical_defs:
        raise RuntimeError("No canonical definitions could be loaded")
    heads, meta_models, base_bundle, symbol_columns = _load_heads(args)
    transform_cache = Path(args.transform_cache) if str(args.transform_cache).strip() else None
    blend_default_configs = _load_blend_default_configs(args.blend_default_config) if bool(args.enqueue_blend_defaults) else {}
    native_aux_feature_reuse_cache = _load_native_aux_feature_reuse_cache(args.aux_native_reuse_feature_source)
    setattr(args, "_native_aux_feature_reuse_cache", native_aux_feature_reuse_cache)
    if bool(args.aux_native_reuse_features):
        print(
            "[reliability_blend] native aux feature reuse cache: "
            f"entries={len(native_aux_feature_reuse_cache)}, source={args.aux_native_reuse_feature_source or 'none'}",
            flush=True,
        )

    all_score_frames: list[pd.DataFrame] = []
    period_hpo_rows: list[pd.DataFrame] = []
    component_diag_rows: list[dict[str, Any]] = []
    component_model_bundle: dict[str, Any] = {
        "schema_version": "reliability_blend_native_component_models_v1",
        "status": "oof_component_models_and_full_fit_live_contract",
        "native_component_scoring": "default_required_for_new_runs; deployable scoring uses full_fit live_contract_safe_v1 component models",
        "distilled_student_status": "audit_fallback_only",
        "full_fit_component_models_enabled": bool(getattr(args, "persist_full_fit_component_models", True)),
        "full_fit_live_feature_contract": "live_contract_safe_v1",
        "heads": {},
    }
    feature_manifest: dict[str, Any] = {
        "period_old_target": "binary difficult period from rolling HR@30 surprise using anchored._period_increment_features",
        "period_new_target": "future 24h EWMA of timestamp mean abs(anchor_score - y_bin) for rows with anchor_rank>=0.5, percentile-normalized on each training fold",
        "period_new_spectral_inputs": "causal state_spectral_* features from rolling prior market-feature covariance/projection geometry; fitted inside each outer fold and scored on held-out timestamps",
        "period_new_tail_labels": "fold-local labels period_bad_05/10/15 and period_tail_severity are derived from period_new_target for period-learner weighting and HPO diagnostics only",
        "period_new_hpo_objective": "0.45*APLift@5 + 0.25*APLift@10 + 0.15*Recall@10 + 0.10*NDCG@10 + 0.05*difficulty_decile_spread",
        "period_new_sample_weights": "asymmetric smooth ramp: base weight increases only for above-median difficulty; selected bad-share boost is applied gradually over the configured tail ramp",
        "auxiliary_lgbm_backend": "q_fail_soft and new_period use the native train-meta LGBM stability feature-selection/HPO pipeline when --aux-native-lgbm is enabled; production train_meta is not retrained",
        "auxiliary_lgbm_selection_objectives": {
            "new_period": "native LGBM train-meta stability/HPO mechanics on fold-local tail_severity by default",
            "qfail_soft": "native LGBM train-meta stability/HPO mechanics on soft failure magnitude inside anchor top50",
        },
        "qfail_hard_target": "1[y_bin=0] inside anchor top30 rank>=0.70",
        "qfail_soft_target": "(1-y_bin) * anchor_score inside anchor top50 rank>=0.50, no timestamp smoothing",
        "calibration": "disabled: reliability blend exports ranking scores only; no calibrated probability layer is fitted or consumed",
        "native_component_model_persistence": "enabled by default; fitted qfail_soft/new_period models, feature contracts, HPO params, diagnostics, and blend coefficients are persisted for future live-equivalent scoring",
        "native_aux_feature_reuse": {
            "enabled": bool(args.aux_native_reuse_features),
            "source": str(args.aux_native_reuse_feature_source or ""),
            "cache_entries": int(len(native_aux_feature_reuse_cache)),
            "min_features": int(args.aux_native_reuse_min_features),
            "min_fraction": float(args.aux_native_reuse_min_fraction),
            "contract": "reuse same-head qfail_soft/new_period selected features and HPO params when available; otherwise reuse prior same-head component feature contract; fall back to fresh LGBM selection when incompatible",
        },
        "qfail_added_feature_blocks": [
            "anchor boundary/location features: rank gaps to 70/90, top10/top20/top30 flags, 30-50 replacement band flag",
            "timestamp-relative anchor features: score/rank z-score, distance to timestamp q10/q90, timestamp dispersion",
            "meta-output diagnostics: uncertainty, path instability, model drift, reconstruction anomaly, cluster/mahalanobis state, and per-symbol score/rank drift",
            "bounded support/context interactions selected on each training fold and reused on validation",
        ],
        "period_added_feature_blocks": [
            "timestamp anchor state: score/rank mean/std/q10/q50/q90, top-rank shares, top30 score stats",
            "timestamp uncertainty state: entropy and p*(1-p) means/tails",
            "canonical/current timestamp aggregates from existing meta feature matrix",
        ],
        "blend_search": {
            "alpha_beta_bounds": [-0.5, 0.5],
            "nonlinear_component_powers": list(NONLINEAR_POWER_GRID),
            "component_sides": list(NONLINEAR_SIDE_GRID),
            "objective": "global_tophr + 0.33*q25_tophr + 0.20*q05_tophr, where tophr=HR10+0.33*HR20+0.25*HR30",
            "default_config_path": str(args.blend_default_config),
            "default_configs_loaded": {head: sorted(configs.keys()) for head, configs in blend_default_configs.items()},
        },
        "features": {},
        "params": _public_arg_dict(args),
    }

    for head in heads:
        print(f"[reliability_blend] head={head.head}", flush=True)
        panel = _downcast_numeric(_normalise_keys(pd.read_parquet(head.meta_oof_path)), exclude=["timestamp", "symbol"])
        panel = panel.sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)
        if int(args.max_rows) > 0 and len(panel) > int(args.max_rows):
            keep = np.linspace(0, len(panel) - 1, int(args.max_rows)).round().astype(int)
            panel = panel.iloc[np.unique(keep)].sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)
        race = meta_models[head.meta_key]
        current_x, raw = ctx._assemble_head_context(
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
        base_x = stack._assemble_base_selected_matrix(
            head=head,
            panel=panel,
            base_bundle=base_bundle,
            feature_dir=Path(args.feature_dir),
            transform_cache=transform_cache,
            symbol_columns=symbol_columns,
        )
        y = ctx._meta_target(panel)
        anchor_score = ctx._current_meta_score(panel)
        rank0 = fixed._rank0(panel, anchor_score)
        global_soft_target = _soft_difficulty_target(
            panel["timestamp"],
            y,
            anchor_score,
            rank0,
            rank_threshold=float(args.period_soft_rank_threshold),
            horizon_hours=int(args.period_soft_horizon_hours),
            halflife_hours=float(args.period_soft_halflife_hours),
        )
        old_period = np.full(len(panel), np.nan, dtype=np.float32)
        new_period = np.full(len(panel), np.nan, dtype=np.float32)
        qfail_hard = np.full(len(panel), np.nan, dtype=np.float32)
        qfail_soft = np.full(len(panel), np.nan, dtype=np.float32)
        fold_valid = np.zeros(len(panel), dtype=bool)
        path_cols: list[str] | None = None
        feature_manifest["features"].setdefault(head.head, {"period_timestamp": set(), "qfail_row": set()})
        component_model_bundle["heads"].setdefault(
            head.head,
            {
                "models": [],
                "score_columns": {
                    "anchor_score": "anchor_score",
                    "period_new": "period_new_score",
                    "qfail_soft": "qfail_soft_score",
                    "qfail_hard": "qfail_hard_score",
                },
            },
        )

        folds = canon._make_chrono_folds(panel["timestamp"], int(args.outer_folds), embargo_hours=int(args.embargo_hours))
        for fold in folds:
            tr = np.asarray(fold.train_idx, dtype=np.int64)
            va = np.asarray(fold.valid_idx, dtype=np.int64)
            fold_valid[va] = True
            ts_train = panel["timestamp"].iloc[tr].reset_index(drop=True)
            ts_valid = panel["timestamp"].iloc[va].reset_index(drop=True)
            sym_train = panel["symbol"].iloc[tr].reset_index(drop=True) if "symbol" in panel.columns else pd.Series([""] * len(tr))
            sym_valid = panel["symbol"].iloc[va].reset_index(drop=True) if "symbol" in panel.columns else pd.Series([""] * len(va))
            raw_train = raw.iloc[tr].reset_index(drop=True)
            raw_valid = raw.iloc[va].reset_index(drop=True)
            anchor_ctrl_train, path_cols = anchored._anchor_control_features(
                raw_train,
                ts_train,
                anchor_score[tr],
                max_path_cols=int(args.max_control_path_features),
                selected_cols=path_cols,
            )
            anchor_ctrl_valid, _ = anchored._anchor_control_features(
                raw_valid,
                ts_valid,
                anchor_score[va],
                max_path_cols=int(args.max_control_path_features),
                selected_cols=path_cols,
            )
            canonical_train, canonical_valid, canonical_diag = stack._canonical_fold_frames(
                raw,
                fold,
                canonical_defs,
                trailing_window=int(args.trailing_window),
                min_periods=int(args.min_periods),
                min_resolved_features=int(args.min_resolved_features),
            )
            meta_models_list = list(getattr(getattr(race, "best_model", None), "models", []) or [])
            meta_leaf_train, meta_leaf_valid, meta_leaf_diag = stack._leaf_structural_fold_features(
                models=meta_models_list,
                x_train=current_x.iloc[tr].reset_index(drop=True),
                x_valid=current_x.iloc[va].reset_index(drop=True),
                prefix="anchor_meta_leaf",
                max_models=int(args.leaf_max_models),
                tree_stride=int(args.leaf_tree_stride),
                max_trees=int(args.leaf_max_trees),
            )
            base_models, _base_features = _base_models_for_head(base_bundle, head)
            base_leaf_train, base_leaf_valid, base_leaf_diag = stack._leaf_structural_fold_features(
                models=base_models,
                x_train=base_x.iloc[tr].reset_index(drop=True),
                x_valid=base_x.iloc[va].reset_index(drop=True),
                prefix="base_leaf",
                max_models=int(args.leaf_max_models),
                tree_stride=int(args.leaf_tree_stride),
                max_trees=int(args.leaf_max_trees),
            )
            leaf_train = stack._combine_features(base_leaf_train, meta_leaf_train)
            leaf_valid = stack._combine_features(base_leaf_valid, meta_leaf_valid)
            extra_train = _anchor_extra_features(ts_train, anchor_score[tr], rank0[tr])
            extra_valid = _anchor_extra_features(ts_valid, anchor_score[va], rank0[va])
            anchor_drift_train = _anchor_meta_drift_features(ts_train, sym_train, anchor_score[tr], rank0[tr])
            anchor_drift_valid = _anchor_meta_drift_features(ts_valid, sym_valid, anchor_score[va], rank0[va])
            meta_source_train = stack._combine_features(panel.iloc[tr].reset_index(drop=True), raw_train)
            meta_source_valid = stack._combine_features(panel.iloc[va].reset_index(drop=True), raw_valid)
            metaout_cols = _select_meta_output_columns(meta_source_train, max_cols=int(args.qfail_meta_output_max_features))
            metaout_train = _meta_output_extra_features(
                meta_source_train,
                ts_train,
                sym_train,
                selected_cols=metaout_cols,
                max_derived_cols=int(args.qfail_meta_output_max_derived_features),
            )
            metaout_valid = _meta_output_extra_features(
                meta_source_valid,
                ts_valid,
                sym_valid,
                selected_cols=metaout_cols,
                max_derived_cols=int(args.qfail_meta_output_max_derived_features),
            )
            extra_train = stack._combine_features(extra_train, anchor_drift_train, metaout_train)
            extra_valid = stack._combine_features(extra_valid, anchor_drift_valid, metaout_valid)
            structural_train = stack._combine_features(canonical_train, leaf_train, extra_train)
            structural_valid = stack._combine_features(canonical_valid, leaf_valid, extra_valid)
            qfail_context_cols = _select_qfail_context_columns(
                structural_train,
                max_context_cols=int(args.qfail_interaction_max_context_features),
            )
            qfail_ix_train = _qfail_context_interactions(structural_train, extra_train, context_cols=qfail_context_cols)
            qfail_ix_valid = _qfail_context_interactions(structural_valid, extra_valid, context_cols=qfail_context_cols)
            full_train = stack._combine_features(anchor_ctrl_train, structural_train, qfail_ix_train)
            full_valid = stack._combine_features(anchor_ctrl_valid, structural_valid, qfail_ix_valid)
            feature_manifest["features"][head.head]["qfail_row"].update(map(str, full_train.columns))

            # Old difficult-period learner.
            z_source_train = stack._combine_features(canonical_train, current_x.iloc[tr].reset_index(drop=True))
            z_source_valid = stack._combine_features(canonical_valid, current_x.iloc[va].reset_index(drop=True))
            anchor_state_train = _timestamp_anchor_state(ts_train, anchor_score[tr], rank0[tr])
            anchor_state_valid = _timestamp_anchor_state(ts_valid, anchor_score[va], rank0[va])
            z_train_base = stack._timestamp_feature_table(
                z_source_train,
                ts_train,
                max_columns=int(args.max_timestamp_features),
            )
            z_valid_base = stack._timestamp_feature_table(
                z_source_valid,
                ts_valid,
                max_columns=int(args.max_timestamp_features),
            )
            z_train_market, z_valid_market, spectral_diag = _append_fold_spectral_position_features(
                z_train_base,
                z_valid_base,
                args=args,
            )
            z_train = z_train_market.join(anchor_state_train, how="left")
            z_valid = z_valid_market.join(anchor_state_valid, how="left").reindex(columns=z_train.columns)
            z_train = _downcast_numeric(z_train.replace([np.inf, -np.inf], np.nan).fillna(0.0))
            z_valid = _downcast_numeric(z_valid.replace([np.inf, -np.inf], np.nan).fillna(0.0))
            nuisance_train = anchored._timestamp_nuisance_table(ts_train, panel["symbol"].iloc[tr] if "symbol" in panel.columns else None)
            nuisance_valid = anchored._timestamp_nuisance_table(ts_valid, panel["symbol"].iloc[va] if "symbol" in panel.columns else None)
            period_old_train, period_old_valid, old_diag = anchored._period_increment_features(
                z_train=z_train,
                z_valid=z_valid,
                nuisance_train=nuisance_train,
                nuisance_valid=nuisance_valid,
                train_timestamps=ts_train,
                valid_timestamps=ts_valid,
                y_train=y[tr],
                anchor_train=anchor_score[tr],
                seed=int(args.seed + 101 * fold.fold_id),
                args=args,
            )
            old_period[va] = pd.to_numeric(period_old_valid.get("q_period_inc", pd.Series(0.0, index=period_old_valid.index)), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            feature_manifest["features"][head.head]["period_timestamp"].update(map(str, z_train.columns))

            # New soft difficult-period learner.
            target_train_raw = _soft_difficulty_target(
                ts_train,
                y[tr],
                anchor_score[tr],
                rank0[tr],
                rank_threshold=float(args.period_soft_rank_threshold),
                horizon_hours=int(args.period_soft_horizon_hours),
                halflife_hours=float(args.period_soft_halflife_hours),
            ).reindex(z_train.index)
            target_train = pd.Series(
                _percentile_from_train(
                    target_train_raw.to_numpy(dtype=np.float32),
                    target_train_raw.to_numpy(dtype=np.float32),
                    nonfinite_fill=None,
                ),
                index=target_train_raw.index,
                dtype="float32",
            )
            share, boost, hpo_table = _choose_period_soft_hpo(z_train, target_train, seed=int(args.seed + 211 * fold.fold_id), args=args)
            hpo_table.insert(0, "fold", int(fold.fold_id))
            hpo_table.insert(0, "head", head.head)
            period_hpo_rows.append(hpo_table)
            weights = _period_sample_weights(
                target_train.to_numpy(dtype=np.float32),
                bottom_share=share,
                boost=boost,
                ramp_share=float(args.period_soft_tail_ramp_share),
                ramp_power=float(args.period_soft_tail_ramp_power),
                badness_base_weight=float(args.period_soft_badness_base_weight),
            )
            period_fit_target = target_train.to_numpy(dtype=np.float32)
            period_selection_target = period_fit_target
            period_selection_objective = "soft_future_error_percentile"
            if str(args.period_soft_selection_target).strip().lower() == "tail_severity":
                period_selection_target = _period_tail_labels(period_fit_target)["period_tail_severity"].astype(np.float32)
                period_selection_objective = "tail_severity_from_soft_future_error"
            period_pred = None
            new_period_diag: dict[str, Any]
            new_period_artifact: dict[str, Any] | None = None
            if bool(args.aux_native_lgbm):
                period_pred, new_period_diag, new_period_artifact = _fit_native_aux_regressor(
                    z_train.reset_index(drop=True),
                    period_selection_target,
                    z_valid.reset_index(drop=True),
                    sample_weight=weights,
                    timestamps_train=pd.Series(z_train.index),
                    assets_train=None,
                    seed=int(args.seed + 307 * fold.fold_id),
                    min_train_rows=int(args.period_soft_min_train_timestamps),
                    label="new_period",
                    head=head.head,
                    scope="oof",
                    args=args,
                )
                if period_pred is not None:
                    new_period_diag["selection_objective"] = f"native_train_meta_stability_hpo_on_{period_selection_objective}"
                    if new_period_artifact is not None:
                        new_period_artifact["selection_objective"] = new_period_diag["selection_objective"]
                        new_period_artifact["target_definition"] = feature_manifest["period_new_target"]
                        new_period_artifact["target_transform"] = period_selection_objective
                        new_period_artifact["diagnostics"] = dict(new_period_diag)
            if period_pred is None:
                period_pred, _period_model, new_period_diag, new_period_artifact = _fit_timestamp_regressor(
                    z_train.reset_index(drop=True),
                    period_selection_target,
                    z_valid.reset_index(drop=True),
                    sample_weight=weights,
                    seed=int(args.seed + 307 * fold.fold_id),
                    args=args,
                )
                new_period_diag["native_backend"] = "lightweight_lgbm_fallback"
                new_period_diag["selection_objective"] = f"lightweight_lgbm_on_{period_selection_objective}"
                if new_period_artifact is not None:
                    new_period_artifact["selection_objective"] = new_period_diag["selection_objective"]
                    new_period_artifact["target_definition"] = feature_manifest["period_new_target"]
                    new_period_artifact["target_transform"] = period_selection_objective
                    new_period_artifact["diagnostics"] = dict(new_period_diag)
            if bool(args.persist_component_models) and new_period_artifact is not None:
                new_period_artifact.update(
                    {
                        "head": head.head,
                        "fold": int(fold.fold_id),
                        "component": "new_period",
                        "train_timestamps": _timestamp_bounds(ts_train),
                        "valid_timestamps": _timestamp_bounds(ts_valid),
                        "period_soft_selected_bottom_share": float(share),
                        "period_soft_selected_boost": float(boost),
                    }
                )
                component_model_bundle["heads"][head.head]["models"].append(new_period_artifact)
            pred_series = pd.Series(period_pred, index=z_valid.index)
            new_period[va] = stack._align_timestamp_features(ts_valid, pd.DataFrame({"q_period_soft": pred_series}))["q_period_soft"].to_numpy(dtype=np.float32)

            # Hard and soft q_fail learners.
            fail_mask = (rank0[tr] >= 0.70) & (y[tr] >= 0)
            fail_target = np.full(len(tr), np.nan, dtype=np.float32)
            fail_target[fail_mask] = (y[tr][fail_mask] == 0).astype(np.float32)
            fail_pred, fail_diag = fixed._crossfit_two_models(
                control_train=anchor_ctrl_train,
                full_train=full_train,
                control_valid=anchor_ctrl_valid,
                full_valid=full_valid,
                target_train=fail_target,
                mask_train=fail_mask,
                timestamps_train=ts_train,
                seed=int(args.seed + 401 * fold.fold_id),
                args=args,
            )
            qfail_hard[va] = fail_pred["full_valid"]
            soft_pred, soft_diag, soft_artifact = _fit_soft_qfail_regressor(
                full_train,
                y[tr],
                anchor_score[tr],
                rank0[tr],
                full_valid,
                timestamps_train=ts_train,
                assets_train=sym_train,
                seed=int(args.seed + 503 * fold.fold_id),
                head=head.head,
                scope="oof",
                args=args,
            )
            qfail_soft[va] = soft_pred
            if bool(args.persist_component_models) and soft_artifact is not None:
                soft_artifact.update(
                    {
                        "head": head.head,
                        "fold": int(fold.fold_id),
                        "component": "qfail_soft",
                        "train_timestamps": _timestamp_bounds(ts_train),
                        "valid_timestamps": _timestamp_bounds(ts_valid),
                    }
                )
                component_model_bundle["heads"][head.head]["models"].append(soft_artifact)

            valid_period_target_raw = global_soft_target.reindex(pd.to_datetime(ts_valid, utc=True, errors="coerce").drop_duplicates()).dropna()
            component_diag_rows.append(
                {
                    "head": head.head,
                    "fold": int(fold.fold_id),
                    "train_rows": int(len(tr)),
                    "valid_rows": int(len(va)),
                    "period_soft_selected_bottom_share": share,
                    "period_soft_selected_boost": boost,
                    "period_timestamp_feature_count": int(z_train.shape[1]),
                    "period_spectral_feature_count": int(spectral_diag.get("spectral_feature_count", 0)),
                    "period_spectral_source_feature_count": int(spectral_diag.get("spectral_source_feature_count", 0)),
                    "qfail_meta_output_selected_cols": int(len(metaout_cols)),
                    "qfail_meta_output_feature_count": int(metaout_train.shape[1] + anchor_drift_train.shape[1]),
                    "qfail_interaction_context_cols": int(len(qfail_context_cols)),
                    "qfail_interaction_feature_count": int(qfail_ix_train.shape[1]),
                    **{f"old_period_{k}": v for k, v in old_diag.items()},
                    **{f"new_period_{k}": v for k, v in new_period_diag.items()},
                    **{f"qfail_hard_{k}": v for k, v in fail_diag.items()},
                    **{f"qfail_soft_{k}": v for k, v in soft_diag.items()},
                    **canonical_diag,
                    **meta_leaf_diag,
                    **base_leaf_diag,
                    "valid_period_target_timestamps": int(len(valid_period_target_raw)),
                }
            )
            print(f"[reliability_blend] head={head.head} fold={fold.fold_id}/{len(folds)} train={len(tr)} valid={len(va)}", flush=True)

        if bool(args.persist_component_models) and bool(getattr(args, "persist_full_fit_component_models", True)):
            full_fit_artifacts, full_fit_diag = _fit_full_fit_live_contract_components(
                head=head.head,
                panel=panel,
                raw=raw,
                y=y,
                anchor_score=anchor_score,
                rank0=rank0,
                args=args,
                feature_manifest=feature_manifest,
            )
            component_model_bundle["heads"][head.head]["models"].extend(full_fit_artifacts)
            component_diag_rows.append(
                {
                    "head": head.head,
                    "fold": "full_fit",
                    "train_rows": int(len(panel)),
                    "valid_rows": 0,
                    "full_fit_model_count": int(len(full_fit_artifacts)),
                    **full_fit_diag,
                }
            )
            print(
                "[reliability_blend] "
                f"head={head.head} full_fit_live_contract_models={len(full_fit_artifacts)}",
                flush=True,
            )

        component_frame = pd.DataFrame(
            {
                "head": head.head,
                "row_id": np.arange(len(panel), dtype=np.int64),
                "timestamp": pd.to_datetime(panel["timestamp"], utc=True, errors="coerce"),
                "symbol": panel["symbol"].astype(str) if "symbol" in panel.columns else "",
                "y_bin": y,
                "anchor_score": np.where(fold_valid, anchor_score, np.nan),
                "anchor_rank_timestamp": np.where(fold_valid, rank0, np.nan),
                "period_old_score": old_period,
                "period_new_score": new_period,
                "qfail_hard_score": qfail_hard,
                "qfail_soft_score": qfail_soft,
            }
        )
        all_score_frames.append(component_frame)

    scores = pd.concat(all_score_frames, axis=0, ignore_index=True) if all_score_frames else pd.DataFrame()
    if scores.empty:
        raise RuntimeError("No component scores were generated")
    scores.to_parquet(out_dir / "reliability_blend_component_scores_preoptuna.parquet", index=False)

    blend_rows: list[dict[str, Any]] = []
    trial_frames: list[pd.DataFrame] = []
    period_eval_rows: list[dict[str, Any]] = []
    for head, group in scores.groupby("head", sort=True):
        eval_mask = np.isfinite(pd.to_numeric(group["anchor_score"], errors="coerce").to_numpy(dtype=np.float32)) & (group["y_bin"].to_numpy(dtype=np.float32) >= 0)
        eval_group = group.loc[eval_mask].copy()
        y = eval_group["y_bin"].to_numpy(dtype=np.float32)
        ts = eval_group["timestamp"]
        anchor_rank = _rank01(eval_group["anchor_score"].to_numpy(dtype=np.float32))
        old_period_rank = _rank01(eval_group["period_old_score"].to_numpy(dtype=np.float32))
        new_period_rank = _rank01(eval_group["period_new_score"].to_numpy(dtype=np.float32))
        hard_qfail_rank = _rank01(eval_group["qfail_hard_score"].to_numpy(dtype=np.float32))
        soft_qfail_rank = _rank01(eval_group["qfail_soft_score"].to_numpy(dtype=np.float32))
        if head in component_model_bundle.get("heads", {}):
            component_model_bundle["heads"][head]["component_rank_references"] = {
                "anchor_score": _score_reference_payload(eval_group["anchor_score"].to_numpy(dtype=np.float32)),
                "period_old_score": _score_reference_payload(eval_group["period_old_score"].to_numpy(dtype=np.float32)),
                "period_new_score": _score_reference_payload(eval_group["period_new_score"].to_numpy(dtype=np.float32)),
                "qfail_hard_score": _score_reference_payload(eval_group["qfail_hard_score"].to_numpy(dtype=np.float32)),
                "qfail_soft_score": _score_reference_payload(eval_group["qfail_soft_score"].to_numpy(dtype=np.float32)),
            }
        full_target = _soft_difficulty_target(
            ts,
            y,
            eval_group["anchor_score"].to_numpy(dtype=np.float32),
            eval_group["anchor_rank_timestamp"].to_numpy(dtype=np.float32),
            rank_threshold=float(args.period_soft_rank_threshold),
            horizon_hours=int(args.period_soft_horizon_hours),
            halflife_hours=float(args.period_soft_halflife_hours),
        )
        ts_pred = eval_group.groupby("timestamp", sort=True)[["period_old_score", "period_new_score"]].mean()
        target_eval = pd.Series(
            _percentile_from_train(
                full_target.to_numpy(dtype=np.float32),
                full_target.to_numpy(dtype=np.float32),
                nonfinite_fill=None,
            ),
            index=full_target.index,
        )
        period_eval_rows.append({"head": head, "period_model": "old_binary_surprise", **_period_tail_detection_metrics(target_eval.reindex(ts_pred.index).to_numpy(dtype=np.float32), ts_pred["period_old_score"].to_numpy(dtype=np.float32))})
        period_eval_rows.append({"head": head, "period_model": "new_soft_future_error_tail_weighted", **_period_tail_detection_metrics(target_eval.reindex(ts_pred.index).to_numpy(dtype=np.float32), ts_pred["period_new_score"].to_numpy(dtype=np.float32))})
        variants = {
            BLEND_OLD_HARD: (old_period_rank, hard_qfail_rank),
            BLEND_NEW_HARD: (new_period_rank, hard_qfail_rank),
            BLEND_NEW_SOFT: (new_period_rank, soft_qfail_rank),
            BLEND_OLD_SOFT: (old_period_rank, soft_qfail_rank),
        }
        baseline_context = _metric_context(ts, y, min_week_rows=int(args.min_week_rows))
        baseline_metrics = _blend_metrics_fast(baseline_context, anchor_rank)
        baseline_metrics["auc"] = _safe_auc(y, anchor_rank)
        baseline_metrics["logloss"] = _safe_logloss(y, anchor_rank)
        blend_rows.append({"head": head, "variant": "A0_anchor_rank_only", "alpha": 0.0, "beta": 0.0, **baseline_metrics})
        for variant, (period_rank, qfail_rank) in variants.items():
            best, trials = _optimise_blend(
                timestamps=ts,
                y=y,
                anchor_rank=anchor_rank,
                period_rank=period_rank,
                qfail_rank=qfail_rank,
                variant=variant,
                seed=int(args.seed + _stable_seed_offset(head, variant)),
                n_trials=int(args.optuna_trials),
                min_week_rows=int(args.min_week_rows),
                default_config=blend_default_configs.get(str(head), {}).get(str(variant)),
            )
            best["head"] = head
            blend_rows.append(best)
            trials.insert(0, "head", head)
            trial_frames.append(trials)
            score = _blend_score(
                anchor_rank,
                period_rank,
                qfail_rank,
                float(best["alpha"]),
                float(best["beta"]),
                period_power=float(best.get("period_power", 1.0)),
                period_side=str(best.get("period_side", "high")),
                qfail_power=float(best.get("qfail_power", 1.0)),
                qfail_side=str(best.get("qfail_side", "high")),
            )
            scores.loc[eval_group.index, f"blend_{variant}_score"] = score
            scores.loc[eval_group.index, f"blend_{variant}_rank"] = _rank01(score)

    # Convert feature set manifests from sets to sorted lists.
    for head, blocks in feature_manifest["features"].items():
        for block, cols in list(blocks.items()):
            feature_manifest["features"][head][block] = sorted(cols)

    blend_summary = pd.DataFrame(blend_rows)
    default_soft_rows: list[pd.Series] = []
    if not blend_summary.empty:
        soft_candidates = blend_summary[blend_summary["variant"].isin(SOFT_QFAIL_BLEND_VARIANTS)].copy()
        for _head, group in soft_candidates.groupby("head", sort=True):
            default_soft_rows.append(group.sort_values(["objective", "global_tophr", "q25_tophr"], ascending=False).iloc[0])
    default_soft_summary = pd.DataFrame(default_soft_rows)
    deployable_rows: list[pd.Series] = []
    if not blend_summary.empty:
        deployable_candidates = blend_summary[
            blend_summary["variant"].isin((BLEND_NEW_SOFT, BLEND_NEW_HARD))
        ].copy()
        for _head, group in deployable_candidates.groupby("head", sort=True):
            preferred = group.loc[group["variant"].astype(str).eq(BLEND_NEW_SOFT)]
            chooser = preferred if not preferred.empty else group
            deployable_rows.append(
                chooser.sort_values(["objective", "global_tophr", "q25_tophr"], ascending=False).iloc[0]
            )
    deployable_default_summary = pd.DataFrame(deployable_rows)
    optuna_trials = pd.concat(trial_frames, axis=0, ignore_index=True) if trial_frames else pd.DataFrame()
    component_diag = pd.DataFrame(component_diag_rows)
    period_hpo = pd.concat(period_hpo_rows, axis=0, ignore_index=True) if period_hpo_rows else pd.DataFrame()
    period_eval = pd.DataFrame(period_eval_rows)

    scores.to_parquet(out_dir / "reliability_blend_component_scores.parquet", index=False)
    blend_summary.to_csv(out_dir / "reliability_blend_optuna_winners.csv", index=False)
    default_soft_summary.to_csv(out_dir / "reliability_blend_soft_qfail_default_by_head.csv", index=False)
    deployable_default_summary.to_csv(out_dir / "reliability_blend_deployable_default_by_head.csv", index=False)
    optuna_trials.to_csv(out_dir / "reliability_blend_optuna_trials.csv", index=False)
    component_diag.to_csv(out_dir / "reliability_blend_component_diagnostics.csv", index=False)
    period_hpo.to_csv(out_dir / "reliability_blend_period_soft_hpo_trials.csv", index=False)
    period_eval.to_csv(out_dir / "reliability_blend_period_detection_metrics.csv", index=False)
    component_model_manifest_path = None
    component_model_bundle_path = None
    if bool(args.persist_component_models):
        component_model_dir = Path(args.component_model_dir) if str(args.component_model_dir).strip() else out_dir / "reliability_blend_component_models"
        component_model_dir = _ensure_dir(component_model_dir)
        component_model_bundle["blend_winners"] = blend_summary.to_dict("records")
        component_model_bundle["default_soft_qfail_config_by_head"] = default_soft_summary.to_dict("records")
        component_model_bundle["default_deployable_config_by_head"] = deployable_default_summary.to_dict("records")
        component_model_bundle["optuna_trials_path"] = str(out_dir / "reliability_blend_optuna_trials.csv")
        component_model_bundle["component_scores_path"] = str(out_dir / "reliability_blend_component_scores.parquet")
        component_model_bundle["feature_target_manifest_path"] = str(out_dir / "reliability_blend_feature_target_manifest.json")
        component_model_bundle["params"] = _public_arg_dict(args)
        component_model_bundle_path = component_model_dir / "reliability_blend_native_component_models.joblib"
        joblib.dump(component_model_bundle, component_model_bundle_path, compress=3)
        manifest = {
            "schema_version": component_model_bundle["schema_version"],
            "status": component_model_bundle["status"],
            "native_component_scoring": component_model_bundle["native_component_scoring"],
            "distilled_student_status": component_model_bundle["distilled_student_status"],
            "component_model_bundle_path": str(component_model_bundle_path),
            "component_scores_path": str(out_dir / "reliability_blend_component_scores.parquet"),
            "blend_winners_path": str(out_dir / "reliability_blend_optuna_winners.csv"),
            "default_soft_qfail_config_path": str(out_dir / "reliability_blend_soft_qfail_default_by_head.csv"),
            "default_deployable_config_path": str(out_dir / "reliability_blend_deployable_default_by_head.csv"),
            "heads": {},
        }
        for model_head, bundle in component_model_bundle.get("heads", {}).items():
            model_rows = [
                _component_model_summary(artifact)
                for artifact in list(bundle.get("models", []) or [])
            ]
            full_fit_rows = [
                row
                for row in model_rows
                if str(row.get("fold", "")).lower() == "full_fit"
                or str(row.get("model_scope", "")).lower() == "full_fit"
            ]
            manifest["heads"][str(model_head)] = {
                "score_columns": dict(bundle.get("score_columns", {}) or {}),
                "model_count": int(len(model_rows)),
                "full_fit_model_count": int(len(full_fit_rows)),
                "full_fit_components": sorted(
                    str(row.get("component"))
                    for row in full_fit_rows
                    if row.get("component") is not None
                ),
                "models": model_rows,
                "component_rank_references": {
                    str(name): _score_reference_summary(payload)
                    for name, payload in dict(bundle.get("component_rank_references", {}) or {}).items()
                    if isinstance(payload, dict)
                },
            }
        component_model_manifest_path = component_model_dir / "reliability_blend_native_component_model_manifest.json"
        component_model_manifest_path.write_text(json.dumps(manifest, indent=2, default=_json_default))
    (out_dir / "reliability_blend_feature_target_manifest.json").write_text(json.dumps(feature_manifest, indent=2, default=_json_default))
    train_meta_wiring = {
        "status": "handoff_plan",
        "calibration": "disabled; consume blend ranks/scores as ranking features, not probabilities",
        "meta_contract": {
            "head_count": "one meta head per strategy head",
            "label": "unchanged y_bin",
            "output": "one probability/ranking score from train_meta",
        },
        "feature_blocks_to_append": [
            "qfail_soft_score and timestamp/global ranks",
            "period_new_score and timestamp/global ranks",
            "best nonlinear blend score/rank per frozen variant",
            "fold-fitted state_spectral_* market spectral-position context for difficult-period learning and optional meta joins",
            "fold-fitted canonical model-state context",
            "fold-fitted canonical market-state context",
            "leaf structural support/novelty summaries",
        ],
        "feature_selection_pipeline": {
            "reuse": "train_meta selected-feature/HPO flow",
            "scope": "auxiliary learners only; the production train_meta model is not physically retrained with these features",
            "change": "reuse native train-meta LGBM stability-selection/HPO mechanics while changing auxiliary targets/objectives: period tail severity and soft q_fail magnitude",
            "leakage_guard": "all qfail/period/context features must be generated in the outer training fold and scored on held-out rows before feature selection",
        },
        "native_component_artifacts": {
            "status": "persisted" if bool(args.persist_component_models) else "disabled",
            "bundle": str(component_model_bundle_path) if component_model_bundle_path is not None else None,
            "manifest": str(component_model_manifest_path) if component_model_manifest_path is not None else None,
            "live_path": "native component models and blend coefficients are the default for future reliability scoring; distilled student is audit/fallback only",
        },
        "policy_objective": {
            "name": "top_tail_reliability_objective",
            "formula": "global_tophr + 0.33*q25_tophr + 0.20*q05_tophr",
            "tophr": "HR@10 + 0.33*HR@20 + 0.25*HR@30",
            "period_hpo_formula": "0.45*APLift@5 + 0.25*APLift@10 + 0.15*Recall@10 + 0.10*NDCG@10 + 0.05*difficulty_decile_spread",
            "guards": [
                "no material HR@10 deterioration",
                "minimum evaluable rows per week/timestamp",
                "report Q5/Q10/Q25/Q50/Q75 weekly HR tails",
                "compare entrant HR versus removed-row HR when baseline ranks are available",
            ],
        },
        "hpo_search_space_additions": {
            "alpha": [-0.5, 0.5],
            "beta": [-0.5, 0.5],
            "period_power": list(NONLINEAR_POWER_GRID),
            "qfail_power": list(NONLINEAR_POWER_GRID),
            "period_side": list(NONLINEAR_SIDE_GRID),
            "qfail_side": list(NONLINEAR_SIDE_GRID),
            "period_soft_bottom_share": [0.05, 0.075, 0.10, 0.15],
            "period_soft_boost": [2.0, 3.0, 4.0, 5.0],
            "period_soft_tail_ramp_share": float(args.period_soft_tail_ramp_share),
            "period_soft_tail_ramp_power": float(args.period_soft_tail_ramp_power),
            "period_soft_selection_target": str(args.period_soft_selection_target),
            "period_spectral_features": bool(args.period_spectral_features),
            "period_spectral_lookback": int(args.period_spectral_lookback),
            "period_spectral_min_periods": int(args.period_spectral_min_periods),
            "period_spectral_top_k": int(args.period_spectral_top_k),
            "period_spectral_max_features": int(args.period_spectral_max_features),
            "period_spectral_shrinkage": float(args.period_spectral_shrinkage),
            "aux_native_lgbm": bool(args.aux_native_lgbm),
            "aux_native_hpo_trials": int(args.aux_native_hpo_trials),
            "aux_native_hpo_patience": int(args.aux_native_hpo_patience),
        },
    }
    (out_dir / "reliability_blend_train_meta_wiring_plan.json").write_text(json.dumps(train_meta_wiring, indent=2, default=_json_default))

    cols = [
        "head",
        "variant",
        "alpha",
        "beta",
        "period_power",
        "period_side",
        "qfail_power",
        "qfail_side",
        "objective",
        "auc",
        "logloss",
        "global_hr10",
        "global_hr20",
        "global_hr30",
        "global_tophr",
        "q25_tophr",
        "q05_tophr",
        "q25_hr10",
        "q25_hr20",
        "q25_hr30",
        "q05_hr10",
        "q05_hr20",
        "q05_hr30",
    ]
    lines = ["# Reliability Blend Optuna Summary", "", "## Winning Blend Configs", "", blend_summary[[c for c in cols if c in blend_summary.columns]].to_markdown(index=False, floatfmt=".6f"), ""]
    if not default_soft_summary.empty:
        lines.extend(["## Default Soft q_fail Configs", "", default_soft_summary[[c for c in cols if c in default_soft_summary.columns]].to_markdown(index=False, floatfmt=".6f"), ""])
    if not period_eval.empty:
        lines.extend(["## Difficult Period Learner Metrics", "", period_eval.to_markdown(index=False, floatfmt=".6f"), ""])
    lines.extend(
        [
            "## Calibration",
            "",
            "Calibration is disabled for this pipeline. The blend output is optimized and consumed as a ranking score only.",
            "",
            "## Added Feature Blocks",
            "",
            "- soft q_fail: anchor boundary/rank-band features, timestamp-relative anchor score/rank features, meta-output drift/uncertainty/path diagnostics, and capped fold-selected support/context interactions.",
            "- difficult period: timestamp anchor-state aggregates, uncertainty/entropy summaries, existing canonical/current timestamp aggregates, and causal `state_spectral_*` market spectral-position features.",
            "",
            "## train_meta Wiring",
            "",
            "- Reuse the existing train_meta feature-selection/HPO flow by appending the fold-fitted reliability feature blocks.",
            "- Select reliability arms with `global_tophr + 0.33*q25_tophr + 0.20*q05_tophr`, where `tophr = HR@10 + 0.33*HR@20 + 0.25*HR@30`.",
            "- Keep leakage controls: generate q_fail, period, context, and leaf references inside the outer training fold before held-out scoring.",
            "",
            "## Targets",
            "",
            "- old difficult period: binary rolling HR@30 surprise labels from the existing anchored period learner.",
            "- new difficult period: future 24h EWMA of timestamp mean absolute anchor error on rows with anchor rank >= 0.5, percentile-normalized inside each training fold.",
            "- new difficult-period tail labels: fold-local `period_bad_05`, `period_bad_10`, `period_bad_15`, and `period_tail_severity` derived from the soft target for weighting and HPO diagnostics.",
            "- new difficult-period HPO: selects bottom-share/boost by emphasizing APLift@5, APLift@10, Recall@10, NDCG@10, and difficulty-decile spread; the fold learner then uses native LGBM stability selection/HPO on tail severity when enabled.",
            "- hard q_fail: hard failure label inside anchor top 30%.",
            "- soft q_fail: `(1-y_bin) * anchor_score` inside anchor top 50%, with no timestamp smoothing; the fold learner uses native LGBM stability selection/HPO on this soft failure magnitude when enabled.",
            "",
        ]
    )
    (out_dir / "reliability_blend_optuna_summary.md").write_text("\n".join(lines))
    print(f"[reliability_blend] wrote {out_dir}", flush=True)
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-artifact-dir", default="data_perp/artifacts/meta_featureselect_recentguard_20260622_0119")
    parser.add_argument("--baseline-artifact-dir", default="data_perp/artifacts/20260617_090000_no_mkt4_labelhpo_final_fit")
    parser.add_argument("--report-dir", default="data_perp/reports/meta_featureselect_recentguard_20260622_0119")
    parser.add_argument("--feature-dir", default="data_perp/features/20260605_070000")
    parser.add_argument("--canonical-reduction", default="data_perp/reports/meta_recent_failure_diagnostics_20260622_archetype_usefulness_multitarget_clean_contract_v1/canonical_archetype_reduction.csv")
    parser.add_argument("--transform-cache", default="")
    parser.add_argument("--output-dir", default="data_perp/reports/reliability_blend_optuna_20260623")
    parser.add_argument("--only-head", nargs="*", default=list(HEADS))
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--embargo-hours", type=int, default=24)
    parser.add_argument("--inner-embargo-hours", type=int, default=12)
    parser.add_argument("--trailing-window", type=int, default=24 * 28)
    parser.add_argument("--min-periods", type=int, default=24 * 7)
    parser.add_argument("--min-resolved-features", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-control-path-features", type=int, default=40)
    parser.add_argument("--max-timestamp-features", type=int, default=80)
    parser.add_argument("--leaf-max-models", type=int, default=1)
    parser.add_argument("--leaf-tree-stride", type=int, default=3)
    parser.add_argument("--leaf-max-trees", type=int, default=80)
    parser.add_argument("--min-train-rows", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=160)
    parser.add_argument("--min-child-fraction", type=float, default=0.025)
    parser.add_argument("--period-short-window", type=int, default=72)
    parser.add_argument("--period-long-window", type=int, default=120)
    parser.add_argument("--period-difficult-quantile", type=float, default=0.25)
    parser.add_argument("--period-min-train-timestamps", type=int, default=80)
    parser.add_argument("--period-max-depth", type=int, default=3)
    parser.add_argument("--period-n-estimators", type=int, default=140)
    parser.add_argument("--period-min-child-fraction", type=float, default=0.035)
    parser.add_argument("--period-soft-rank-threshold", type=float, default=0.50)
    parser.add_argument("--period-soft-horizon-hours", type=int, default=24)
    parser.add_argument("--period-soft-halflife-hours", type=float, default=12.0)
    parser.add_argument("--period-soft-inner-folds", type=int, default=3)
    parser.add_argument("--period-soft-min-train-timestamps", type=int, default=80)
    parser.add_argument("--period-soft-max-depth", type=int, default=3)
    parser.add_argument("--period-soft-n-estimators", type=int, default=120)
    parser.add_argument("--period-soft-min-child-fraction", type=float, default=0.035)
    parser.add_argument("--period-soft-tail-ramp-share", type=float, default=0.15)
    parser.add_argument("--period-soft-tail-ramp-power", type=float, default=1.5)
    parser.add_argument("--period-soft-badness-base-weight", type=float, default=1.0)
    parser.add_argument("--period-soft-selection-target", choices=["tail_severity", "soft_percentile"], default="tail_severity")
    parser.add_argument("--period-spectral-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--period-spectral-lookback", type=int, default=48)
    parser.add_argument("--period-spectral-min-periods", type=int, default=24)
    parser.add_argument("--period-spectral-top-k", type=int, default=3)
    parser.add_argument("--period-spectral-max-features", type=int, default=64)
    parser.add_argument("--period-spectral-shrinkage", type=float, default=0.10)
    parser.add_argument("--qfail-soft-rank-threshold", type=float, default=0.50)
    parser.add_argument("--qfail-soft-min-train-rows", type=int, default=500)
    parser.add_argument("--qfail-soft-max-depth", type=int, default=3)
    parser.add_argument("--qfail-soft-n-estimators", type=int, default=180)
    parser.add_argument("--qfail-soft-min-child-fraction", type=float, default=0.025)
    parser.add_argument("--qfail-meta-output-max-features", type=int, default=80)
    parser.add_argument("--qfail-meta-output-max-derived-features", type=int, default=48)
    parser.add_argument("--qfail-interaction-max-context-features", type=int, default=24)
    parser.add_argument("--aux-native-lgbm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--aux-native-hpo-trials", type=int, default=24)
    parser.add_argument("--aux-native-hpo-patience", type=int, default=8)
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
        help=(
            "Comma-separated component bundle/report directories or feature-target manifests "
            "used to reuse same-head native q_fail/new_period feature contracts and HPO params."
        ),
    )
    parser.add_argument("--aux-native-reuse-min-features", type=int, default=8)
    parser.add_argument("--aux-native-reuse-min-fraction", type=float, default=0.25)
    parser.add_argument("--blend-default-config", default=str(DEFAULT_BLEND_CONFIG_PATH))
    parser.add_argument("--enqueue-blend-defaults", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persist-component-models", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--persist-full-fit-component-models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After OOF component scoring, fit deployable full_fit qfail/new_period "
            "native components on live-contract-safe inputs. Disable only for cheap diagnostics."
        ),
    )
    parser.add_argument(
        "--component-model-dir",
        default="",
        help="Directory for fitted native q_fail/new_period component bundles. Defaults to output-dir/reliability_blend_component_models.",
    )
    parser.add_argument("--optuna-trials", type=int, default=120)
    parser.add_argument("--min-week-rows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=37)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
