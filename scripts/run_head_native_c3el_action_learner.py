#!/usr/bin/env python3
"""Train and replay a head-native C3el size-action learner.

This is intentionally different from the previous C3el head overlays:

* the training unit is a ``head x timestamp x strategy`` action group;
* each head fits its own feature selection, classifier, value model, and
  threshold;
* the model sees all candidate multipliers when labels are available, not only
  rows where a frozen shared scorer already chose an intervention;
* the default action is always multiplier=1.0 unless the head-specific gate
  clears a conservative threshold.

The script consumes a full exact-state action panel for training labels and a
full live/replay action-feature panel for scoring. It then replays the combined
head-native schedule through the existing portfolio policy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if sys.version_info < (3, 10):
    raise SystemExit("run_head_native_c3el_action_learner.py requires Python 3.10+; use the repo Python 3.11 runtime.")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import fit_hierarchical_ev_curves, normalise_candidate_table
from scripts.run_exact_state_size_action_learning import (  # noqa: E402
    GROUP_LABEL_COLUMNS,
    _add_stage1_context_interaction_features,
    _group_action_table,
    _lgbm_deterministic_kwargs,
    _select_lgbm_features,
)
from scripts.run_global_portfolio_period_multiplier import _load_policy_params
from scripts.run_size_action_live_scorer_replay import _head_from_strategy, _load_candidates, _replay, _summarise


HEADS = ("long_bars", "long_dist", "short_asset", "short_boll")
SCORE_COLUMNS = {
    "timestamp",
    "strategy_id",
    "head",
    "multiplier",
    "delta_full_J",
    "delta_immediate_J",
    "delta_full_net_pnl",
    "delta_full_cost_pnl",
    "delta_full_turnover",
    "delta_full_J_per_notional",
    "delta_immediate_J_per_notional",
    "fold_id",
    "split",
    *GROUP_LABEL_COLUMNS,
}
ACTION_LABEL_COLUMNS = {
    "action_binds",
    "action_positive",
    "action_economic_positive",
    "action_positive_label",
}
PRESETS: dict[str, dict[str, Any]] = {
    "short_asset_default": {
        "active_heads": "short_asset",
        "min_train_groups": 60,
        "min_positive_groups": 5,
        "fallback_thresholds": "short_asset=0.8",
        "fallback_max_eval_keep_share_by_head": "short_asset=0.25",
        "fallback_min_pred_delta_by_head": "short_asset=320",
        "guard_low_strategy_candidate_count_max_by_head": "short_asset=24",
        "guard_min_removed_trade_share_timestamp_by_head": "short_asset=0.55",
    },
    "short_asset_plus_shortboll_guard04": {
        "active_heads": "short_asset,short_boll",
        "min_train_groups": 60,
        "min_positive_groups": 5,
        "fallback_thresholds": "short_asset=0.8,short_boll=0.0",
        "fallback_max_eval_keep_share_by_head": "short_asset=0.25,short_boll=0.05",
        "fallback_min_pred_delta_by_head": "short_asset=320,short_boll=50",
        "guard_low_strategy_candidate_count_max_by_head": "short_asset=24",
        "guard_min_removed_trade_share_timestamp_by_head": "short_asset=0.55",
        "action_model_objective_by_head": "short_boll=quantile",
        "action_quantile_alpha_by_head": "short_boll=0.2",
        "action_feature_min_by_head": "short_boll:projected_removed_trade_share_strategy=0.4",
    },
}


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _apply_preset(args: argparse.Namespace) -> None:
    if args.preset == "custom":
        return
    preset = PRESETS.get(str(args.preset))
    if preset is None:
        raise ValueError(f"Unknown preset: {args.preset}")
    for key, value in preset.items():
        setattr(args, key, value)


def _resolve_active_heads(raw: str | None) -> tuple[set[str], set[str]]:
    """Return requested and effective active heads.

    An empty ``--active-heads`` means "all heads" for the runner.  Keep that
    explicit so downstream promotion/status reports do not confuse an all-head
    head-native run with a no-op overlay run.
    """
    requested = {x.strip() for x in str(raw or "").split(",") if x.strip()}
    unknown = sorted(requested.difference(HEADS))
    if unknown:
        raise ValueError(f"Unknown active head(s): {unknown}; expected one of {list(HEADS)}")
    effective = set(requested) if requested else set(HEADS)
    return requested, effective


def _resolve_selected_heads(raw: str | None, *, active_heads: set[str]) -> tuple[set[str], set[str]]:
    """Return requested and effective heads whose scored actions are applied.

    ``--active-heads`` controls which heads are trained/scored.  ``--selected-heads``
    controls which of those scored heads are allowed to affect the replay
    schedule.  Keeping these separate avoids treating a head-native C3el run as
    a single all-head overlay.
    """
    text = str(raw or "").strip()
    if not text:
        return set(), set(active_heads)
    if text.lower() in {"none", "noop", "baseline"}:
        return set(), set()
    requested = {x.strip() for x in text.split(",") if x.strip()}
    unknown = sorted(requested.difference(HEADS))
    if unknown:
        raise ValueError(f"Unknown selected head(s): {unknown}; expected one of {list(HEADS)}")
    inactive = sorted(requested.difference(active_heads))
    if inactive:
        raise ValueError(
            f"Selected head(s) were not active/scored: {inactive}; "
            f"active heads are {sorted(active_heads)}"
        )
    return requested, set(requested)


def _parse_head_float_map(raw: str | None) -> dict[str, float]:
    if raw is None or not str(raw).strip():
        return {}
    out: dict[str, float] = {}
    for part in str(raw).replace(";", ",").split(","):
        token = part.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Expected head=value token, got: {token}")
        head, value = token.split("=", 1)
        out[str(head).strip()] = float(value)
    return out


def _parse_float_grid(raw: str | None) -> list[float]:
    if raw is None or not str(raw).strip():
        return []
    vals: list[float] = []
    for part in str(raw).split(","):
        token = part.strip()
        if token:
            vals.append(float(token))
    return vals


def _parse_head_int_map(raw: str | None) -> dict[str, int]:
    return {head: int(round(value)) for head, value in _parse_head_float_map(raw).items()}


def _parse_head_float_list_map(raw: str | None) -> dict[str, list[float]]:
    """Parse per-head grids, using ``;`` between heads and ``|`` inside grids.

    Example: ``short_asset=0.35|0.55|0.75;short_boll=0.10|0.25``.
    """
    if raw is None or not str(raw).strip():
        return {}
    out: dict[str, list[float]] = {}
    for part in str(raw).split(";"):
        token = part.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Expected head=v1|v2 token, got: {token}")
        head, values = token.split("=", 1)
        head = str(head).strip()
        vals = [float(v.strip()) for v in str(values).split("|") if v.strip()]
        if not head or not vals:
            raise ValueError(f"Expected non-empty head and values in token: {token}")
        out[head] = vals
    return out


def _parse_head_str_map(raw: str | None) -> dict[str, str]:
    if raw is None or not str(raw).strip():
        return {}
    out: dict[str, str] = {}
    for part in str(raw).replace(";", ",").split(","):
        token = part.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Expected head=value token, got: {token}")
        head, value = token.split("=", 1)
        out[str(head).strip()] = str(value).strip()
    return out


def _parse_head_feature_min_map(raw: str | None) -> dict[str, dict[str, float]]:
    """Parse rules like ``short_boll:feature=0.2,short_asset:other=1``."""
    if raw is None or not str(raw).strip():
        return {}
    out: dict[str, dict[str, float]] = {}
    for part in str(raw).replace(";", ",").split(","):
        token = part.strip()
        if not token:
            continue
        if ":" not in token or "=" not in token:
            raise ValueError(f"Expected head:feature=value token, got: {token}")
        head_feature, value = token.split("=", 1)
        head, feature = head_feature.split(":", 1)
        head = str(head).strip()
        feature = str(feature).strip()
        if not head or not feature:
            raise ValueError(f"Expected non-empty head and feature in token: {token}")
        out.setdefault(head, {})[feature] = float(value)
    return out


def _parse_head_feature_max_map(raw: str | None) -> dict[str, dict[str, float]]:
    """Parse rules like ``short_asset:cooldown_count=38.5``."""
    return _parse_head_feature_min_map(raw)


def _week_start(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, utc=True, errors="coerce").dt.to_period("W-SUN").dt.start_time.dt.tz_localize("UTC")


def _normalise_action_panel(frame: pd.DataFrame, *, require_labels: bool) -> pd.DataFrame:
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.loc[out["timestamp"].notna()].copy()
    out["strategy_id"] = out["strategy_id"].astype(str)
    out["head"] = out["strategy_id"].map(_head_from_strategy)
    out["multiplier"] = pd.to_numeric(out.get("multiplier"), errors="coerce").fillna(1.0).clip(0.0, 1.0)
    if "action_binds" not in out.columns:
        removed = pd.to_numeric(out.get("projected_removed_trade_count"), errors="coerce").fillna(0.0)
        out["action_binds"] = np.where(out["multiplier"].lt(1.0) & removed.gt(0.0), 1.0, 0.0)
    else:
        out["action_binds"] = pd.to_numeric(out["action_binds"], errors="coerce").fillna(0.0)
    if require_labels:
        for col in ("delta_full_J", "delta_immediate_J"):
            if col not in out.columns:
                raise ValueError(f"training action panel is missing required label column: {col}")
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    else:
        for col in ("delta_full_J", "delta_immediate_J"):
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
            else:
                out[col] = 0.0
    return out


def _load_training_panels(paths: list[Path]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in paths:
        frame = _normalise_action_panel(_read_frame(path), require_labels=True)
        parts.append(frame)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True, sort=False)
    out = out.drop_duplicates(subset=["timestamp", "strategy_id", "multiplier"], keep="last")
    return out.sort_values(["timestamp", "strategy_id", "multiplier"]).reset_index(drop=True)


def _load_eval_actions(path: Path, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frame = _normalise_action_panel(_read_frame(path), require_labels=False)
    frame = frame.loc[frame["timestamp"].ge(start) & frame["timestamp"].lt(end)].copy()
    return frame.sort_values(["timestamp", "strategy_id", "multiplier"]).reset_index(drop=True)


def _pre_start_deployable_candidates(deployable: pd.DataFrame, *, start: pd.Timestamp) -> pd.DataFrame:
    """Return EV-curve training candidates, failing closed if none predate eval."""
    out = deployable.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    train = out.loc[out["timestamp"].lt(start)].copy()
    if train.empty:
        raise ValueError(
            "No deployable candidates before the evaluation start are available "
            "to fit the hierarchical EV curve. Refusing to fit EV curves on "
            "evaluation-period rows."
        )
    return train


def _group_features_from_panel(
    panel: pd.DataFrame,
    *,
    epsilon_gain: float,
    epsilon_margin: float,
    epsilon_gain_per_notional: float,
    epsilon_margin_per_notional: float,
) -> pd.DataFrame:
    groups = _group_action_table(
        panel,
        epsilon_gain=float(epsilon_gain),
        epsilon_margin=float(epsilon_margin),
        epsilon_gain_per_notional=float(epsilon_gain_per_notional),
        epsilon_margin_per_notional=float(epsilon_margin_per_notional),
    )
    if groups.empty:
        return groups
    groups["timestamp"] = pd.to_datetime(groups["timestamp"], utc=True, errors="coerce")
    groups["strategy_id"] = groups["strategy_id"].astype(str)
    groups["head"] = groups["strategy_id"].map(_head_from_strategy)
    groups = _relabel_group_features(
        groups,
        epsilon_gain=epsilon_gain,
        epsilon_margin=epsilon_margin,
        epsilon_gain_per_notional=epsilon_gain_per_notional,
        epsilon_margin_per_notional=epsilon_margin_per_notional,
    )
    groups = _add_stage1_context_interaction_features(groups)
    return groups


def _relabel_group_features(
    groups: pd.DataFrame,
    *,
    epsilon_gain: float,
    epsilon_margin: float,
    epsilon_gain_per_notional: float,
    epsilon_margin_per_notional: float,
) -> pd.DataFrame:
    """Recompute strict intervention labels on a cached group table."""
    if groups.empty:
        return groups.copy()
    out = groups.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["strategy_id"] = out["strategy_id"].astype(str)
    if "head" not in out.columns:
        out["head"] = out["strategy_id"].map(_head_from_strategy)
    can_bind = pd.to_numeric(out.get("group_can_bind"), errors="coerce").fillna(0.0).gt(0.0)
    best_multiplier = pd.to_numeric(out.get("best_multiplier"), errors="coerce").fillna(1.0)
    best_gain = pd.to_numeric(out.get("best_gain"), errors="coerce").fillna(0.0)
    best_margin = pd.to_numeric(out.get("best_margin"), errors="coerce").fillna(0.0)
    best_gain_norm = pd.to_numeric(out.get("best_gain_per_notional"), errors="coerce").fillna(0.0)
    best_margin_norm = pd.to_numeric(out.get("best_margin_per_notional"), errors="coerce").fillna(0.0)
    out["y_intervene"] = (
        can_bind
        & best_multiplier.lt(1.0)
        & best_gain.gt(float(epsilon_gain))
        & best_margin.gt(float(epsilon_margin))
        & best_gain_norm.gt(float(epsilon_gain_per_notional))
        & best_margin_norm.gt(float(epsilon_margin_per_notional))
    ).astype(float)
    return out


def _slice_group_cache(groups: pd.DataFrame, mask: pd.Series | np.ndarray) -> pd.DataFrame:
    if groups.empty:
        return groups.copy()
    return groups.loc[mask].copy()


def _feature_columns(frame: pd.DataFrame, *, exclude: set[str], max_features: int) -> list[str]:
    cols: list[str] = []
    for col in frame.columns:
        if col in exclude:
            continue
        if not pd.api.types.is_numeric_dtype(frame[col]):
            continue
        vals = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if vals.notna().sum() > 0 and vals.nunique(dropna=True) > 1:
            cols.append(str(col))
    return cols[: int(max_features)]


def _prepare_matrix(frame: pd.DataFrame, features: list[str], medians: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    x = pd.DataFrame(index=frame.index)
    for col in features:
        x[col] = pd.to_numeric(frame[col], errors="coerce") if col in frame.columns else np.nan
    x = x.replace([np.inf, -np.inf], np.nan)
    if medians is None:
        medians = x.median(axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x = x.fillna(medians).fillna(0.0).astype(np.float32)
    return x, medians.astype(float)


def _make_group_target(groups: pd.DataFrame, *, mode: str, epsilon_gain: float, epsilon_gain_per_notional: float) -> pd.Series:
    can_bind = pd.to_numeric(groups.get("group_can_bind"), errors="coerce").fillna(0.0).gt(0.0)
    best_nonbase = pd.to_numeric(groups.get("best_nonbaseline_gain"), errors="coerce").fillna(0.0)
    best_gain = pd.to_numeric(groups.get("best_gain"), errors="coerce").fillna(0.0)
    best_gain_notional = pd.to_numeric(groups.get("best_gain_per_notional"), errors="coerce").fillna(0.0)
    best_multiplier = pd.to_numeric(groups.get("best_multiplier"), errors="coerce").fillna(1.0)
    strict = pd.to_numeric(groups.get("y_intervene"), errors="coerce").fillna(0.0).gt(0.0)
    if mode == "strict":
        return strict.astype(int)
    if mode == "positive_nonbaseline":
        target = can_bind & best_nonbase.gt(float(epsilon_gain)) & best_multiplier.lt(1.0)
        return target.astype(int)
    if mode == "positive_nonbaseline_notional":
        target = (
            can_bind
            & best_nonbase.gt(float(epsilon_gain))
            & best_gain.gt(float(epsilon_gain))
            & best_gain_notional.gt(float(epsilon_gain_per_notional))
            & best_multiplier.lt(1.0)
        )
        return target.astype(int)
    raise ValueError(f"Unknown group target mode: {mode}")


def _make_action_target(action_rows: pd.DataFrame, *, epsilon_gain: float) -> pd.Series:
    binds = pd.to_numeric(action_rows.get("action_binds"), errors="coerce").fillna(0.0).gt(0.0)
    delta = pd.to_numeric(action_rows.get("delta_full_J"), errors="coerce").fillna(0.0)
    return (binds & delta.gt(float(epsilon_gain))).astype(int)


def _fit_classifier(train: pd.DataFrame, features: list[str], y: pd.Series, *, weights: np.ndarray, seed: int) -> tuple[Any, pd.Series, dict[str, Any]]:
    y_arr = pd.to_numeric(y, errors="coerce").fillna(0).astype(int).to_numpy()
    if len(train) < 30 or len(np.unique(y_arr)) < 2 or not features:
        rate = float(np.mean(y_arr)) if len(y_arr) else 0.0
        return {"constant": rate}, pd.Series(dtype=float), {
            "constant": True,
            "fit_rows": int(len(train)),
            "positive_rows": int(y_arr.sum()) if len(y_arr) else 0,
            "positive_rate": rate,
            "feature_count": int(len(features)),
        }
    from lightgbm import LGBMClassifier

    x, medians = _prepare_matrix(train, features)
    pos = max(int(y_arr.sum()), 1)
    neg = max(int(len(y_arr) - y_arr.sum()), 1)
    model = LGBMClassifier(
        objective="binary",
        n_estimators=160,
        learning_rate=0.04,
        num_leaves=15,
        max_depth=4,
        min_child_samples=max(8, int(0.05 * len(train))),
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=3.0,
        scale_pos_weight=float(max(neg / pos, 1.0)),
        **_lgbm_deterministic_kwargs(seed),
        verbose=-1,
    )
    model.fit(x, y_arr, sample_weight=weights)
    return model, medians, {
        "constant": False,
        "fit_rows": int(len(train)),
        "positive_rows": int(y_arr.sum()),
        "positive_rate": float(np.mean(y_arr)),
        "feature_count": int(len(features)),
    }


def _fit_regressor(
    train: pd.DataFrame,
    features: list[str],
    target: pd.Series,
    *,
    weights: np.ndarray,
    seed: int,
    objective: str = "regression",
    quantile_alpha: float = 0.20,
) -> tuple[Any, pd.Series, dict[str, Any]]:
    y = pd.to_numeric(target, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    if len(train) < 30 or np.nanstd(y) <= 1e-12 or not features:
        value = float(np.nanmean(y)) if len(y) else 0.0
        return {"constant": value}, pd.Series(dtype=float), {
            "constant": True,
            "fit_rows": int(len(train)),
            "target_mean": value,
            "feature_count": int(len(features)),
            "objective": str(objective),
            "quantile_alpha": float(quantile_alpha),
        }
    from lightgbm import LGBMRegressor

    x, medians = _prepare_matrix(train, features)
    objective = str(objective)
    if objective not in {"regression", "quantile"}:
        raise ValueError(f"Unknown action regressor objective: {objective}")
    model_kwargs: dict[str, Any] = {"objective": objective}
    if objective == "quantile":
        model_kwargs["alpha"] = float(np.clip(float(quantile_alpha), 0.01, 0.99))
    model = LGBMRegressor(
        n_estimators=180,
        learning_rate=0.035,
        num_leaves=15,
        max_depth=4,
        min_child_samples=max(8, int(0.04 * len(train))),
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=3.0,
        **_lgbm_deterministic_kwargs(seed),
        **model_kwargs,
        verbose=-1,
    )
    model.fit(x, y, sample_weight=weights)
    return model, medians, {
        "constant": False,
        "fit_rows": int(len(train)),
        "target_mean": float(np.nanmean(y)),
        "feature_count": int(len(features)),
        "objective": objective,
        "quantile_alpha": float(quantile_alpha),
    }


def _predict(model: Any, frame: pd.DataFrame, features: list[str], medians: pd.Series, *, proba: bool) -> np.ndarray:
    if isinstance(model, dict) and "constant" in model:
        return np.full(len(frame), float(model["constant"]), dtype=float)
    x, _ = _prepare_matrix(frame, features, medians)
    if proba and hasattr(model, "predict_proba"):
        raw = np.asarray(model.predict_proba(x), dtype=float)
        if raw.ndim == 2 and raw.shape[1] > 1:
            return raw[:, 1]
        return raw.reshape(-1)
    return np.asarray(model.predict(x), dtype=float).reshape(-1)


def _action_training_weights(action_rows: pd.DataFrame, *, positive_weight: float, negative_weight: float) -> np.ndarray:
    delta = pd.to_numeric(action_rows.get("delta_full_J"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    scale = max(float(np.nanmedian(np.abs(delta[delta != 0.0]))) if np.any(delta != 0.0) else 1.0, 1.0)
    weights = np.ones(len(action_rows), dtype=float)
    weights[delta > 0.0] += float(positive_weight) * np.clip(delta[delta > 0.0] / scale, 0.0, 5.0)
    weights[delta < 0.0] += float(negative_weight) * np.clip(np.abs(delta[delta < 0.0]) / scale, 0.0, 5.0)
    return np.clip(weights, 0.05, 20.0)


def _group_training_weights(groups: pd.DataFrame, y: pd.Series, *, positive_weight: float, harm_weight: float) -> np.ndarray:
    gain = pd.to_numeric(groups.get("best_nonbaseline_gain"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    harm = pd.to_numeric(groups.get("worst_nonbaseline_gain"), errors="coerce").fillna(0.0).clip(upper=0.0).abs().to_numpy(dtype=float)
    scale = max(float(np.nanmedian(np.abs(gain[gain != 0.0]))) if np.any(gain != 0.0) else 1.0, 1.0)
    y_arr = pd.to_numeric(y, errors="coerce").fillna(0).astype(int).to_numpy()
    weights = np.ones(len(groups), dtype=float)
    weights[y_arr > 0] += float(positive_weight) * np.clip(np.maximum(gain[y_arr > 0], 0.0) / scale, 0.0, 5.0)
    weights[y_arr <= 0] += float(harm_weight) * np.clip(harm[y_arr <= 0] / scale, 0.0, 5.0)
    return np.clip(weights, 0.05, 20.0)


def _score_actions(
    action_model: Any,
    action_features: list[str],
    action_medians: pd.Series,
    action_positive_model: Any | None,
    action_positive_features: list[str] | None,
    action_positive_medians: pd.Series | None,
    actions: pd.DataFrame,
    *,
    action_score_mode: str = "value",
) -> pd.DataFrame:
    out = actions.copy()
    raw_value = _predict(action_model, out, action_features, action_medians, proba=False)
    if action_positive_model is None or action_positive_features is None or action_positive_medians is None:
        positive_prob = np.ones(len(out), dtype=float)
    else:
        positive_prob = _predict(
            action_positive_model,
            out,
            list(action_positive_features),
            action_positive_medians,
            proba=True,
        )
    out["pred_action_value_raw"] = raw_value
    out["pred_action_positive_prob"] = np.clip(positive_prob, 0.0, 1.0)
    mode = str(action_score_mode)
    if mode == "value":
        score = raw_value
    elif mode == "positive_probability":
        score = out["pred_action_positive_prob"].to_numpy(dtype=float)
    elif mode == "prob_x_value":
        score = out["pred_action_positive_prob"].to_numpy(dtype=float) * np.maximum(raw_value, 0.0)
    else:
        raise ValueError(f"Unknown action score mode: {action_score_mode}")
    out["pred_action_delta_J"] = np.asarray(score, dtype=float)
    return out


def _select_group_actions(
    groups: pd.DataFrame,
    action_scores: pd.DataFrame,
    *,
    p_intervene: np.ndarray,
    threshold: float,
    min_pred_delta: float,
    allowed_multipliers: set[float],
) -> pd.DataFrame:
    out = groups[["timestamp", "strategy_id", "head"]].copy()
    out["p_intervene"] = np.asarray(p_intervene, dtype=float)
    nonbase = action_scores.loc[action_scores["multiplier"].lt(1.0)].copy()
    if allowed_multipliers:
        allowed = {round(float(v), 6) for v in allowed_multipliers}
        nonbase = nonbase.loc[nonbase["multiplier"].round(6).isin(allowed)].copy()
    if nonbase.empty:
        out["selected_multiplier"] = 1.0
        out["pred_action_delta_J"] = 0.0
        out["gate_keep"] = False
        return out
    nonbase["action_binds"] = pd.to_numeric(nonbase.get("action_binds"), errors="coerce").fillna(0.0)
    nonbase = nonbase.loc[nonbase["action_binds"].gt(0.0)].copy()
    if nonbase.empty:
        out["selected_multiplier"] = 1.0
        out["pred_action_delta_J"] = 0.0
        out["gate_keep"] = False
        return out
    ranked = nonbase.sort_values(["timestamp", "strategy_id", "pred_action_delta_J", "multiplier"], ascending=[True, True, False, True])
    best = ranked.groupby(["timestamp", "strategy_id"], as_index=False).head(1)[
        ["timestamp", "strategy_id", "multiplier", "pred_action_delta_J"]
    ]
    out = out.merge(best, on=["timestamp", "strategy_id"], how="left")
    out["selected_multiplier"] = pd.to_numeric(out["multiplier"], errors="coerce").fillna(1.0).clip(0.0, 1.0)
    out["pred_action_delta_J"] = pd.to_numeric(out["pred_action_delta_J"], errors="coerce").fillna(0.0)
    out["gate_keep"] = out["p_intervene"].ge(float(threshold)) & out["pred_action_delta_J"].gt(float(min_pred_delta)) & out[
        "selected_multiplier"
    ].lt(1.0)
    out.loc[~out["gate_keep"], "selected_multiplier"] = 1.0
    return out.drop(columns=["multiplier"], errors="ignore")


def _cap_selected_actions(selection: pd.DataFrame, *, max_keep: int) -> pd.DataFrame:
    """Keep only the strongest selected actions, reverting the rest to no-op."""
    out = selection.copy()
    keep = out["gate_keep"].astype(bool)
    n_keep = int(keep.sum())
    cap = int(max_keep)
    if cap < 0 or n_keep <= cap:
        return out
    if cap <= 0:
        out["gate_keep"] = False
        out["selected_multiplier"] = 1.0
        return out
    strength = (
        pd.to_numeric(out.get("p_intervene"), errors="coerce").fillna(0.0)
        * pd.to_numeric(out.get("pred_action_delta_J"), errors="coerce").fillna(0.0).clip(lower=0.0)
        * (1.0 - pd.to_numeric(out.get("selected_multiplier"), errors="coerce").fillna(1.0).clip(0.0, 1.0))
    )
    selected_idx = out.loc[keep].assign(_strength=strength.loc[keep]).sort_values(
        ["_strength", "pred_action_delta_J", "p_intervene"], ascending=[False, False, False]
    ).head(cap).index
    new_keep = pd.Series(False, index=out.index)
    new_keep.loc[selected_idx] = True
    out["gate_keep"] = new_keep.to_numpy(dtype=bool)
    out.loc[~out["gate_keep"], "selected_multiplier"] = 1.0
    return out


def _apply_low_breadth_share_guard(
    selection: pd.DataFrame,
    action_rows: pd.DataFrame,
    *,
    max_strategy_candidate_count: float | None,
    min_removed_trade_share_timestamp: float | None,
) -> tuple[pd.DataFrame, int]:
    """Revert cuts when a narrow strategy pool dominates timestamp removals."""
    if selection.empty:
        return selection, 0
    if max_strategy_candidate_count is None or min_removed_trade_share_timestamp is None:
        return selection, 0
    if not np.isfinite(float(max_strategy_candidate_count)) or not np.isfinite(float(min_removed_trade_share_timestamp)):
        return selection, 0
    out = selection.copy()
    if "guard_low_breadth_share" not in out.columns:
        out["guard_low_breadth_share"] = False
    keep = selection.get("gate_keep", pd.Series(False, index=selection.index)).astype(bool)
    if not keep.any():
        return out, 0
    selected = selection.loc[keep, ["timestamp", "strategy_id", "selected_multiplier"]].copy()
    selected["selected_multiplier_round"] = pd.to_numeric(selected["selected_multiplier"], errors="coerce").fillna(1.0).round(6)
    guard_cols = [
        "timestamp",
        "strategy_id",
        "multiplier",
        "strategy_candidate_count",
        "projected_removed_trade_share_timestamp",
    ]
    missing = [col for col in guard_cols if col not in action_rows.columns]
    if missing:
        return out, 0
    actions = action_rows[guard_cols].copy()
    actions["multiplier_round"] = pd.to_numeric(actions["multiplier"], errors="coerce").fillna(1.0).round(6)
    merged = selected.merge(
        actions,
        left_on=["timestamp", "strategy_id", "selected_multiplier_round"],
        right_on=["timestamp", "strategy_id", "multiplier_round"],
        how="left",
    )
    guard_mask = (
        pd.to_numeric(merged["strategy_candidate_count"], errors="coerce").le(float(max_strategy_candidate_count))
        & pd.to_numeric(merged["projected_removed_trade_share_timestamp"], errors="coerce").ge(
            float(min_removed_trade_share_timestamp)
        )
    )
    if not guard_mask.any():
        return out, 0
    guarded_idx = pd.MultiIndex.from_frame(merged.loc[guard_mask, ["timestamp", "strategy_id"]])
    out_idx = pd.MultiIndex.from_frame(out[["timestamp", "strategy_id"]])
    revert = keep & out_idx.isin(guarded_idx)
    out.loc[revert, "gate_keep"] = False
    out.loc[revert, "selected_multiplier"] = 1.0
    out.loc[revert, "guard_low_breadth_share"] = True
    return out, int(revert.sum())


def _apply_action_feature_min_guard(
    selection: pd.DataFrame,
    action_rows: pd.DataFrame,
    *,
    min_rules: dict[str, float],
) -> tuple[pd.DataFrame, int]:
    """Revert selected cuts unless their selected action row satisfies all min rules."""
    if selection.empty or not min_rules:
        return selection, 0
    out = selection.copy()
    if "guard_action_feature_min" not in out.columns:
        out["guard_action_feature_min"] = False
    keep = selection.get("gate_keep", pd.Series(False, index=selection.index)).astype(bool)
    if not keep.any():
        return out, 0
    selected = selection.loc[keep, ["timestamp", "strategy_id", "selected_multiplier"]].copy()
    selected["selected_multiplier_round"] = pd.to_numeric(selected["selected_multiplier"], errors="coerce").fillna(1.0).round(6)
    guard_cols = ["timestamp", "strategy_id", "multiplier", *min_rules.keys()]
    missing = [col for col in guard_cols if col not in action_rows.columns]
    if missing:
        # Fail closed for selected rows when a required guard feature is absent.
        out.loc[keep, "gate_keep"] = False
        out.loc[keep, "selected_multiplier"] = 1.0
        out.loc[keep, "guard_action_feature_min"] = True
        return out, int(keep.sum())
    actions = action_rows[guard_cols].copy()
    actions["multiplier_round"] = pd.to_numeric(actions["multiplier"], errors="coerce").fillna(1.0).round(6)
    merged = selected.merge(
        actions,
        left_on=["timestamp", "strategy_id", "selected_multiplier_round"],
        right_on=["timestamp", "strategy_id", "multiplier_round"],
        how="left",
    )
    guard_mask = pd.Series(False, index=merged.index)
    for feature, threshold in min_rules.items():
        vals = pd.to_numeric(merged[feature], errors="coerce")
        guard_mask |= vals.lt(float(threshold)) | vals.isna()
    if not guard_mask.any():
        return out, 0
    guarded_idx = pd.MultiIndex.from_frame(merged.loc[guard_mask, ["timestamp", "strategy_id"]])
    out_idx = pd.MultiIndex.from_frame(out[["timestamp", "strategy_id"]])
    revert = keep & out_idx.isin(guarded_idx)
    out.loc[revert, "gate_keep"] = False
    out.loc[revert, "selected_multiplier"] = 1.0
    out.loc[revert, "guard_action_feature_min"] = True
    return out, int(revert.sum())


def _apply_action_feature_max_guard(
    selection: pd.DataFrame,
    action_rows: pd.DataFrame,
    *,
    max_rules: dict[str, float],
) -> tuple[pd.DataFrame, int]:
    """Revert selected cuts unless their selected action row is below all max rules."""
    if selection.empty or not max_rules:
        return selection, 0
    out = selection.copy()
    if "guard_action_feature_max" not in out.columns:
        out["guard_action_feature_max"] = False
    keep = selection.get("gate_keep", pd.Series(False, index=selection.index)).astype(bool)
    if not keep.any():
        return out, 0
    selected = selection.loc[keep, ["timestamp", "strategy_id", "selected_multiplier"]].copy()
    selected["selected_multiplier_round"] = pd.to_numeric(selected["selected_multiplier"], errors="coerce").fillna(1.0).round(6)
    guard_cols = ["timestamp", "strategy_id", "multiplier", *max_rules.keys()]
    missing = [col for col in guard_cols if col not in action_rows.columns]
    if missing:
        out.loc[keep, "gate_keep"] = False
        out.loc[keep, "selected_multiplier"] = 1.0
        out.loc[keep, "guard_action_feature_max"] = True
        return out, int(keep.sum())
    actions = action_rows[guard_cols].copy()
    actions["multiplier_round"] = pd.to_numeric(actions["multiplier"], errors="coerce").fillna(1.0).round(6)
    merged = selected.merge(
        actions,
        left_on=["timestamp", "strategy_id", "selected_multiplier_round"],
        right_on=["timestamp", "strategy_id", "multiplier_round"],
        how="left",
    )
    guard_mask = pd.Series(False, index=merged.index)
    for feature, threshold in max_rules.items():
        vals = pd.to_numeric(merged[feature], errors="coerce")
        guard_mask |= vals.gt(float(threshold)) | vals.isna()
    if not guard_mask.any():
        return out, 0
    guarded_idx = pd.MultiIndex.from_frame(merged.loc[guard_mask, ["timestamp", "strategy_id"]])
    out_idx = pd.MultiIndex.from_frame(out[["timestamp", "strategy_id"]])
    revert = keep & out_idx.isin(guarded_idx)
    out.loc[revert, "gate_keep"] = False
    out.loc[revert, "selected_multiplier"] = 1.0
    out.loc[revert, "guard_action_feature_max"] = True
    return out, int(revert.sum())


def _realized_value_for_selection(selection: pd.DataFrame, action_rows: pd.DataFrame) -> float:
    if selection.empty:
        return 0.0
    realized = selection.loc[selection["gate_keep"], ["timestamp", "strategy_id", "selected_multiplier"]].copy()
    if realized.empty:
        return 0.0
    realized["selected_multiplier_round"] = pd.to_numeric(realized["selected_multiplier"], errors="coerce").fillna(1.0).round(6)
    actions = action_rows[["timestamp", "strategy_id", "multiplier", "delta_full_J"]].copy()
    actions["selected_multiplier_round"] = pd.to_numeric(actions["multiplier"], errors="coerce").fillna(1.0).round(6)
    merged = realized.merge(actions, on=["timestamp", "strategy_id", "selected_multiplier_round"], how="left")
    return float(pd.to_numeric(merged["delta_full_J"], errors="coerce").fillna(0.0).sum())


def _choose_threshold(
    groups: pd.DataFrame,
    action_rows: pd.DataFrame,
    p: np.ndarray,
    action_scores: pd.DataFrame,
    *,
    grid: list[float],
    min_keep: int,
    min_pred_delta_grid: list[float],
    allowed_multipliers: set[float],
    feature_min_rules: dict[str, float] | None = None,
    feature_max_rules: dict[str, float] | None = None,
) -> tuple[float, dict[str, Any]]:
    best_threshold = 1.01
    best_min_pred_delta = float(min_pred_delta_grid[0]) if min_pred_delta_grid else 0.0
    best_value = 0.0
    best_keep = 0
    threshold_trials: list[dict[str, Any]] = []
    delta_grid = [float(x) for x in (min_pred_delta_grid or [0.0])]
    for threshold in grid:
        for min_pred_delta in delta_grid:
            selected = _select_group_actions(
                groups,
                action_scores,
                p_intervene=p,
                threshold=float(threshold),
                min_pred_delta=float(min_pred_delta),
                allowed_multipliers=allowed_multipliers,
            )
            selected, feature_min_guarded = _apply_action_feature_min_guard(
                selected,
                action_rows,
                min_rules=feature_min_rules or {},
            )
            selected, feature_max_guarded = _apply_action_feature_max_guard(
                selected,
                action_rows,
                max_rules=feature_max_rules or {},
            )
            keep = int(selected["gate_keep"].sum())
            value = _realized_value_for_selection(selected, action_rows) if keep >= int(min_keep) else 0.0
            threshold_trials.append(
                {
                    "threshold": float(threshold),
                    "min_pred_delta": float(min_pred_delta),
                    "keep": keep,
                    "value": float(value),
                    "eligible": bool(keep >= int(min_keep)),
                    "feature_min_guarded": int(feature_min_guarded),
                    "feature_max_guarded": int(feature_max_guarded),
                }
            )
            if keep < int(min_keep):
                continue
            if value > best_value or (
                np.isclose(value, best_value)
                and value > 0.0
                and (keep < best_keep or (keep == best_keep and float(min_pred_delta) > best_min_pred_delta))
            ):
                best_threshold = float(threshold)
                best_min_pred_delta = float(min_pred_delta)
                best_value = value
                best_keep = keep
    return best_threshold, {
        "threshold": best_threshold,
        "min_pred_delta": best_min_pred_delta,
        "threshold_value": best_value,
        "threshold_keep": best_keep,
        "threshold_has_positive_value": bool(best_keep > 0 and best_value > 0.0),
        "threshold_trials": threshold_trials,
    }


def _resolve_head_config(
    *,
    head: str,
    args: argparse.Namespace,
    threshold_grid: list[float],
    base_min_pred_delta_grid: list[float],
    allowed_multipliers: set[float],
    group_target_mode_by_head: dict[str, str],
    epsilon_gain_by_head: dict[str, float],
    epsilon_margin_by_head: dict[str, float],
    epsilon_gain_per_notional_by_head: dict[str, float],
    epsilon_margin_per_notional_by_head: dict[str, float],
    threshold_grid_by_head: dict[str, list[float]],
    min_pred_delta_grid_by_head: dict[str, list[float]],
    allowed_multipliers_by_head: dict[str, list[float]],
    min_train_groups_by_head: dict[str, int],
    min_positive_groups_by_head: dict[str, int],
    min_threshold_keep_by_head: dict[str, int],
    threshold_holdout_frac_by_head: dict[str, float],
    max_group_features_by_head: dict[str, int],
    max_action_features_by_head: dict[str, int],
    eval_keep_multiplier_by_head: dict[str, float],
    max_eval_keep_share_by_head: dict[str, float],
    action_model_objective_by_head: dict[str, str],
    action_quantile_alpha_by_head: dict[str, float],
    action_score_mode_by_head: dict[str, str],
    action_positive_epsilon_by_head: dict[str, float],
    fallback_thresholds: dict[str, float],
    fallback_min_delta_by_head: dict[str, float],
) -> dict[str, Any]:
    """Return the full effective training/scoring contract for one head."""
    effective_threshold_grid = list(threshold_grid_by_head.get(head, threshold_grid))
    if head in fallback_thresholds:
        effective_threshold_grid = sorted({*effective_threshold_grid, float(fallback_thresholds[head])})
    effective_min_pred_delta_grid = list(min_pred_delta_grid_by_head.get(head, base_min_pred_delta_grid))
    if head in fallback_min_delta_by_head:
        effective_min_pred_delta_grid = sorted({*effective_min_pred_delta_grid, float(fallback_min_delta_by_head[head])})
    effective_allowed_multipliers = {
        float(v) for v in allowed_multipliers_by_head.get(head, sorted(allowed_multipliers))
    }
    config = {
        "head": str(head),
        "group_target_mode": group_target_mode_by_head.get(head, str(args.group_target_mode)),
        "epsilon_gain": float(epsilon_gain_by_head.get(head, float(args.epsilon_gain))),
        "epsilon_margin": float(epsilon_margin_by_head.get(head, float(args.epsilon_margin))),
        "epsilon_gain_per_notional": float(
            epsilon_gain_per_notional_by_head.get(head, float(args.epsilon_gain_per_notional))
        ),
        "epsilon_margin_per_notional": float(
            epsilon_margin_per_notional_by_head.get(head, float(args.epsilon_margin_per_notional))
        ),
        "threshold_grid": [float(x) for x in effective_threshold_grid],
        "min_pred_delta_grid": [float(x) for x in effective_min_pred_delta_grid],
        "allowed_multipliers": sorted(effective_allowed_multipliers),
        "min_train_groups": int(min_train_groups_by_head.get(head, int(args.min_train_groups))),
        "min_positive_groups": int(min_positive_groups_by_head.get(head, int(args.min_positive_groups))),
        "min_threshold_keep": int(min_threshold_keep_by_head.get(head, int(args.min_threshold_keep))),
        "threshold_holdout_frac": float(
            threshold_holdout_frac_by_head.get(head, float(args.threshold_holdout_frac))
        ),
        "max_group_features": int(max_group_features_by_head.get(head, int(args.max_group_features))),
        "max_action_features": int(max_action_features_by_head.get(head, int(args.max_action_features))),
        "eval_keep_multiplier": float(eval_keep_multiplier_by_head.get(head, float(args.eval_keep_multiplier))),
        "max_eval_keep_share": float(max_eval_keep_share_by_head.get(head, float(args.max_eval_keep_share))),
        "action_model_objective": action_model_objective_by_head.get(head, str(args.action_model_objective)),
        "action_quantile_alpha": float(action_quantile_alpha_by_head.get(head, float(args.action_quantile_alpha))),
        "action_score_mode": action_score_mode_by_head.get(head, str(args.action_score_mode)),
        "action_positive_epsilon": float(
            action_positive_epsilon_by_head.get(head, float(args.action_positive_epsilon))
        ),
    }
    if config["group_target_mode"] not in {"strict", "positive_nonbaseline", "positive_nonbaseline_notional"}:
        raise ValueError(f"Unknown group target mode for {head}: {config['group_target_mode']}")
    if config["action_model_objective"] not in {"regression", "quantile"}:
        raise ValueError(f"Unknown action objective for {head}: {config['action_model_objective']}")
    if config["action_score_mode"] not in {"value", "positive_probability", "prob_x_value"}:
        raise ValueError(f"Unknown action score mode for {head}: {config['action_score_mode']}")
    if not config["threshold_grid"]:
        raise ValueError(f"Effective threshold grid for {head} cannot be empty")
    if not config["min_pred_delta_grid"]:
        config["min_pred_delta_grid"] = [0.0]
    if not config["allowed_multipliers"]:
        raise ValueError(f"Effective allowed multipliers for {head} cannot be empty")
    return config


def _fit_head_models(
    train_panel: pd.DataFrame,
    *,
    head: str,
    seed: int,
    max_group_features: int,
    max_action_features: int,
    group_target_mode: str,
    epsilon_gain: float,
    epsilon_margin: float,
    epsilon_gain_per_notional: float,
    epsilon_margin_per_notional: float,
    action_model_objective: str,
    action_quantile_alpha: float,
    action_positive_epsilon: float,
    precomputed_groups: pd.DataFrame | None = None,
) -> tuple[dict[str, Any] | None, pd.DataFrame, pd.DataFrame]:
    head_panel = train_panel.loc[train_panel["head"].astype(str).eq(str(head))].copy()
    if head_panel.empty:
        return None, pd.DataFrame(), pd.DataFrame()
    if precomputed_groups is None:
        groups = _group_features_from_panel(
            head_panel,
            epsilon_gain=epsilon_gain,
            epsilon_margin=epsilon_margin,
            epsilon_gain_per_notional=epsilon_gain_per_notional,
            epsilon_margin_per_notional=epsilon_margin_per_notional,
        )
    else:
        groups = _relabel_group_features(
            precomputed_groups,
            epsilon_gain=epsilon_gain,
            epsilon_margin=epsilon_margin,
            epsilon_gain_per_notional=epsilon_gain_per_notional,
            epsilon_margin_per_notional=epsilon_margin_per_notional,
        )
        groups = _add_stage1_context_interaction_features(groups)
    if groups.empty:
        return None, pd.DataFrame(), head_panel
    y = _make_group_target(groups, mode=group_target_mode, epsilon_gain=epsilon_gain, epsilon_gain_per_notional=epsilon_gain_per_notional)
    group_candidates = _feature_columns(groups, exclude=SCORE_COLUMNS, max_features=512)
    group_weights = _group_training_weights(groups, y, positive_weight=2.0, harm_weight=0.75)
    group_features = _select_lgbm_features(
        groups,
        group_candidates,
        y,
        task="classification",
        max_features=max_group_features,
        sample_weight=group_weights,
        seed=seed,
    )
    group_model, group_medians, group_diag = _fit_classifier(groups, group_features, y, weights=group_weights, seed=seed)

    action_rows = head_panel.loc[head_panel["multiplier"].lt(1.0)].copy()
    if action_rows.empty:
        action_features: list[str] = []
        action_model: Any = {"constant": 0.0}
        action_medians = pd.Series(dtype=float)
        action_diag = {"constant": True, "fit_rows": 0, "target_mean": 0.0, "feature_count": 0}
        action_positive_features: list[str] = []
        action_positive_model: Any = {"constant": 0.0}
        action_positive_medians = pd.Series(dtype=float)
        action_positive_diag = {"constant": True, "fit_rows": 0, "positive_rows": 0, "positive_rate": 0.0, "feature_count": 0}
    else:
        action_candidates = _feature_columns(action_rows, exclude=SCORE_COLUMNS | ACTION_LABEL_COLUMNS, max_features=512)
        action_weights = _action_training_weights(action_rows, positive_weight=2.0, negative_weight=0.5)
        action_features = _select_lgbm_features(
            action_rows,
            action_candidates,
            pd.to_numeric(action_rows["delta_full_J"], errors="coerce").fillna(0.0),
            task="regression",
            max_features=max_action_features,
            sample_weight=action_weights,
            seed=seed + 17,
        )
        action_model, action_medians, action_diag = _fit_regressor(
            action_rows,
            action_features,
            pd.to_numeric(action_rows["delta_full_J"], errors="coerce").fillna(0.0),
            weights=action_weights,
            seed=seed + 17,
            objective=action_model_objective,
            quantile_alpha=action_quantile_alpha,
        )
        action_positive_y = _make_action_target(action_rows, epsilon_gain=action_positive_epsilon)
        action_positive_weights = _action_training_weights(action_rows, positive_weight=3.0, negative_weight=0.5)
        if action_positive_y.nunique(dropna=True) < 2:
            action_positive_features = []
        else:
            action_positive_features = _select_lgbm_features(
                action_rows,
                action_candidates,
                action_positive_y,
                task="classification",
                max_features=max_action_features,
                sample_weight=action_positive_weights,
                seed=seed + 31,
            )
        action_positive_model, action_positive_medians, action_positive_diag = _fit_classifier(
            action_rows,
            action_positive_features,
            action_positive_y,
            weights=action_positive_weights,
            seed=seed + 31,
        )
    bundle = {
        "head": head,
        "group_features": group_features,
        "group_model": group_model,
        "group_medians": group_medians,
        "group_diag": group_diag,
        "action_features": action_features,
        "action_model": action_model,
        "action_medians": action_medians,
        "action_diag": action_diag,
        "action_positive_features": action_positive_features,
        "action_positive_model": action_positive_model,
        "action_positive_medians": action_positive_medians,
        "action_positive_diag": action_positive_diag,
        "train_groups": int(len(groups)),
        "train_actions": int(len(action_rows)),
        "train_positive_groups": int(y.sum()),
        "train_positive_group_rate": float(y.mean()) if len(y) else 0.0,
    }
    return bundle, groups, head_panel


def _score_head(
    bundle: dict[str, Any],
    eval_panel: pd.DataFrame,
    *,
    threshold: float,
    min_pred_delta: float,
    allowed_multipliers: set[float],
    epsilon_gain: float,
    epsilon_margin: float,
    epsilon_gain_per_notional: float,
    epsilon_margin_per_notional: float,
    action_score_mode: str,
    threshold_feature_min_rules: dict[str, float] | None = None,
    threshold_feature_max_rules: dict[str, float] | None = None,
    precomputed_groups: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if precomputed_groups is None:
        groups = _group_features_from_panel(
            eval_panel,
            epsilon_gain=epsilon_gain,
            epsilon_margin=epsilon_margin,
            epsilon_gain_per_notional=epsilon_gain_per_notional,
            epsilon_margin_per_notional=epsilon_margin_per_notional,
        )
    else:
        groups = _relabel_group_features(
            precomputed_groups,
            epsilon_gain=epsilon_gain,
            epsilon_margin=epsilon_margin,
            epsilon_gain_per_notional=epsilon_gain_per_notional,
            epsilon_margin_per_notional=epsilon_margin_per_notional,
        )
        groups = _add_stage1_context_interaction_features(groups)
    if groups.empty:
        return pd.DataFrame()
    p = _predict(bundle["group_model"], groups, bundle["group_features"], bundle["group_medians"], proba=True)
    action_scores = _score_actions(
        bundle["action_model"],
        bundle["action_features"],
        bundle["action_medians"],
        bundle.get("action_positive_model"),
        bundle.get("action_positive_features"),
        bundle.get("action_positive_medians"),
        eval_panel,
        action_score_mode=action_score_mode,
    )
    selected = _select_group_actions(
        groups,
        action_scores,
        p_intervene=p,
        threshold=float(threshold),
        min_pred_delta=float(min_pred_delta),
        allowed_multipliers=allowed_multipliers,
    )
    selected, _min_guarded = _apply_action_feature_min_guard(
        selected,
        eval_panel,
        min_rules=threshold_feature_min_rules or {},
    )
    selected, _max_guarded = _apply_action_feature_max_guard(
        selected,
        eval_panel,
        max_rules=threshold_feature_max_rules or {},
    )
    return selected


def _schedule_for_heads(schedule: pd.DataFrame, heads: set[str]) -> pd.DataFrame:
    """Return a copy where only the requested heads can apply non-1 multipliers."""
    out = schedule.copy()
    if out.empty:
        return out
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["strategy_id"] = out["strategy_id"].astype(str)
    if "head" not in out.columns:
        out["head"] = out["strategy_id"].map(_head_from_strategy)
    out["head"] = out["head"].astype(str)
    out["multiplier"] = pd.to_numeric(out.get("multiplier"), errors="coerce").fillna(1.0).clip(0.0, 1.0)
    selected = set(str(head) for head in heads)
    if not selected:
        out["multiplier"] = 1.0
        return out
    out.loc[~out["head"].isin(selected), "multiplier"] = 1.0
    return out


def _schedule_action_summary(schedule: pd.DataFrame, *, label: str) -> pd.DataFrame:
    if schedule.empty:
        return pd.DataFrame()
    work = schedule.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work["strategy_id"] = work["strategy_id"].astype(str)
    if "head" not in work.columns:
        work["head"] = work["strategy_id"].map(_head_from_strategy)
    work["multiplier"] = pd.to_numeric(work.get("multiplier"), errors="coerce").fillna(1.0).clip(0.0, 1.0)
    rows: list[dict[str, Any]] = []
    for head, group in work.groupby("head", dropna=False):
        nonbase = group["multiplier"].lt(1.0)
        rows.append(
            {
                "schedule": str(label),
                "head": str(head),
                "groups": int(len(group)),
                "intervention_groups": int(nonbase.sum()),
                "intervention_share": float(nonbase.mean()) if len(group) else 0.0,
                "mean_multiplier": float(group["multiplier"].mean()) if len(group) else 1.0,
                "min_multiplier": float(group["multiplier"].min()) if len(group) else 1.0,
                "timestamp_count": int(group["timestamp"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-action-panels", type=Path, nargs="+", required=True)
    parser.add_argument("--eval-action-features", type=Path, required=True)
    parser.add_argument("--broad-candidates", type=Path, required=True)
    parser.add_argument("--deployable-candidates", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--policy-variant", default="refit_bar4_strategy_bar2")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--preset",
        choices=["custom", *sorted(PRESETS)],
        default="custom",
        help="Apply a validated C3el configuration. Use custom to control every option manually.",
    )
    parser.add_argument("--start", default="2026-05-29T00:00:00+00:00")
    parser.add_argument("--end", default="2026-06-26T00:00:00+00:00")
    parser.add_argument("--market-mode", default="perps")
    parser.add_argument("--group-target-mode", choices=["strict", "positive_nonbaseline", "positive_nonbaseline_notional"], default="positive_nonbaseline")
    parser.add_argument(
        "--group-target-mode-by-head",
        default="",
        help="Optional head-specific target modes, e.g. short_boll=strict,short_asset=positive_nonbaseline.",
    )
    parser.add_argument("--epsilon-gain", type=float, default=50.0)
    parser.add_argument("--epsilon-gain-by-head", default="")
    parser.add_argument("--epsilon-margin", type=float, default=25.0)
    parser.add_argument("--epsilon-margin-by-head", default="")
    parser.add_argument("--epsilon-gain-per-notional", type=float, default=0.001)
    parser.add_argument("--epsilon-gain-per-notional-by-head", default="")
    parser.add_argument("--epsilon-margin-per-notional", type=float, default=0.0005)
    parser.add_argument("--epsilon-margin-per-notional-by-head", default="")
    parser.add_argument("--threshold-grid", default="0.35,0.45,0.55,0.65,0.75,0.85")
    parser.add_argument(
        "--threshold-grid-by-head",
        default="",
        help="Optional per-head grids, using ';' between heads and '|' inside grids, e.g. short_asset=0.5|0.7;short_boll=0.1|0.3.",
    )
    parser.add_argument("--min-train-groups", type=int, default=80)
    parser.add_argument("--min-train-groups-by-head", default="")
    parser.add_argument("--min-positive-groups", type=int, default=8)
    parser.add_argument("--min-positive-groups-by-head", default="")
    parser.add_argument("--min-threshold-keep", type=int, default=2)
    parser.add_argument("--min-threshold-keep-by-head", default="")
    parser.add_argument("--threshold-holdout-frac", type=float, default=0.30)
    parser.add_argument("--threshold-holdout-frac-by-head", default="")
    parser.add_argument("--min-pred-delta", type=float, default=0.0)
    parser.add_argument(
        "--min-pred-delta-grid",
        default="",
        help="Optional comma-separated predicted-delta gates to include in holdout threshold selection.",
    )
    parser.add_argument(
        "--min-pred-delta-grid-by-head",
        default="",
        help="Optional per-head predicted-delta grids, using ';' between heads and '|' inside grids.",
    )
    parser.add_argument("--eval-keep-multiplier", type=float, default=4.0)
    parser.add_argument("--eval-keep-multiplier-by-head", default="")
    parser.add_argument("--max-eval-keep-share", type=float, default=0.10)
    parser.add_argument("--max-eval-keep-share-by-head", default="")
    parser.add_argument("--fallback-thresholds", default="")
    parser.add_argument("--fallback-max-eval-keep-share", type=float, default=0.10)
    parser.add_argument("--fallback-max-eval-keep-share-by-head", default="")
    parser.add_argument("--fallback-min-pred-delta", type=float, default=0.0)
    parser.add_argument("--fallback-min-pred-delta-by-head", default="")
    parser.add_argument("--guard-low-strategy-candidate-count-max", type=float, default=float("nan"))
    parser.add_argument("--guard-low-strategy-candidate-count-max-by-head", default="")
    parser.add_argument("--guard-min-removed-trade-share-timestamp", type=float, default=float("nan"))
    parser.add_argument("--guard-min-removed-trade-share-timestamp-by-head", default="")
    parser.add_argument("--allowed-multipliers", default="0,0.5,0.75")
    parser.add_argument(
        "--allowed-multipliers-by-head",
        default="",
        help="Optional per-head multiplier grids, using ';' between heads and '|' inside grids.",
    )
    parser.add_argument("--active-heads", default="")
    parser.add_argument(
        "--selected-heads",
        default="",
        help=(
            "Subset of active/scored heads whose C3el multipliers are applied in "
            "the final replay. Empty means all active heads; use 'none' for a "
            "no-op applied schedule while still writing scored diagnostics."
        ),
    )
    parser.add_argument(
        "--head-isolated-replays",
        action="store_true",
        help="Replay one additional arm per active head with only that head's C3el schedule applied.",
    )
    parser.add_argument("--max-group-features", type=int, default=48)
    parser.add_argument("--max-group-features-by-head", default="")
    parser.add_argument("--max-action-features", type=int, default=64)
    parser.add_argument("--max-action-features-by-head", default="")
    parser.add_argument("--action-model-objective", choices=["regression", "quantile"], default="regression")
    parser.add_argument("--action-model-objective-by-head", default="")
    parser.add_argument("--action-quantile-alpha", type=float, default=0.20)
    parser.add_argument("--action-quantile-alpha-by-head", default="")
    parser.add_argument(
        "--action-score-mode",
        choices=["value", "positive_probability", "prob_x_value"],
        default="value",
        help="Score candidate size actions by value, direct positive-action probability, or probability times positive value.",
    )
    parser.add_argument("--action-score-mode-by-head", default="")
    parser.add_argument(
        "--action-positive-epsilon",
        type=float,
        default=50.0,
        help="Minimum exact-state delta_full_J for a binding action row to be positive in the direct action classifier.",
    )
    parser.add_argument("--action-positive-epsilon-by-head", default="")
    parser.add_argument("--threshold-action-feature-min-by-head", default="")
    parser.add_argument("--threshold-action-feature-max-by-head", default="")
    parser.add_argument("--action-feature-min-by-head", default="")
    parser.add_argument("--action-feature-max-by-head", default="")
    parser.add_argument("--seed", type=int, default=20260628)
    args = parser.parse_args()
    _apply_preset(args)

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    threshold_grid = _parse_float_grid(args.threshold_grid)
    if not threshold_grid:
        raise ValueError("--threshold-grid cannot be empty")
    base_min_pred_delta_grid = sorted({float(args.min_pred_delta), *_parse_float_grid(args.min_pred_delta_grid)})
    allowed_multipliers = {float(x.strip()) for x in str(args.allowed_multipliers).split(",") if x.strip()}
    group_target_mode_by_head = _parse_head_str_map(args.group_target_mode_by_head)
    epsilon_gain_by_head = _parse_head_float_map(args.epsilon_gain_by_head)
    epsilon_margin_by_head = _parse_head_float_map(args.epsilon_margin_by_head)
    epsilon_gain_per_notional_by_head = _parse_head_float_map(args.epsilon_gain_per_notional_by_head)
    epsilon_margin_per_notional_by_head = _parse_head_float_map(args.epsilon_margin_per_notional_by_head)
    threshold_grid_by_head = _parse_head_float_list_map(args.threshold_grid_by_head)
    min_pred_delta_grid_by_head = _parse_head_float_list_map(args.min_pred_delta_grid_by_head)
    allowed_multipliers_by_head = _parse_head_float_list_map(args.allowed_multipliers_by_head)
    min_train_groups_by_head = _parse_head_int_map(args.min_train_groups_by_head)
    min_positive_groups_by_head = _parse_head_int_map(args.min_positive_groups_by_head)
    min_threshold_keep_by_head = _parse_head_int_map(args.min_threshold_keep_by_head)
    threshold_holdout_frac_by_head = _parse_head_float_map(args.threshold_holdout_frac_by_head)
    max_group_features_by_head = _parse_head_int_map(args.max_group_features_by_head)
    max_action_features_by_head = _parse_head_int_map(args.max_action_features_by_head)
    eval_keep_multiplier_by_head = _parse_head_float_map(args.eval_keep_multiplier_by_head)
    max_eval_keep_share_by_head = _parse_head_float_map(args.max_eval_keep_share_by_head)
    fallback_thresholds = _parse_head_float_map(args.fallback_thresholds)
    fallback_max_keep_share_by_head = _parse_head_float_map(args.fallback_max_eval_keep_share_by_head)
    fallback_min_delta_by_head = _parse_head_float_map(args.fallback_min_pred_delta_by_head)
    guard_candidate_count_by_head = _parse_head_float_map(args.guard_low_strategy_candidate_count_max_by_head)
    guard_removed_share_by_head = _parse_head_float_map(args.guard_min_removed_trade_share_timestamp_by_head)
    action_model_objective_by_head = _parse_head_str_map(args.action_model_objective_by_head)
    action_quantile_alpha_by_head = _parse_head_float_map(args.action_quantile_alpha_by_head)
    action_score_mode_by_head = _parse_head_str_map(args.action_score_mode_by_head)
    action_positive_epsilon_by_head = _parse_head_float_map(args.action_positive_epsilon_by_head)
    threshold_action_feature_min_by_head = _parse_head_feature_min_map(args.threshold_action_feature_min_by_head)
    threshold_action_feature_max_by_head = _parse_head_feature_max_map(args.threshold_action_feature_max_by_head)
    action_feature_min_by_head = _parse_head_feature_min_map(args.action_feature_min_by_head)
    action_feature_max_by_head = _parse_head_feature_max_map(args.action_feature_max_by_head)
    requested_active_heads, active_heads = _resolve_active_heads(args.active_heads)
    requested_selected_heads, selected_heads = _resolve_selected_heads(args.selected_heads, active_heads=active_heads)
    head_configs = {
        head: _resolve_head_config(
            head=head,
            args=args,
            threshold_grid=threshold_grid,
            base_min_pred_delta_grid=base_min_pred_delta_grid,
            allowed_multipliers=allowed_multipliers,
            group_target_mode_by_head=group_target_mode_by_head,
            epsilon_gain_by_head=epsilon_gain_by_head,
            epsilon_margin_by_head=epsilon_margin_by_head,
            epsilon_gain_per_notional_by_head=epsilon_gain_per_notional_by_head,
            epsilon_margin_per_notional_by_head=epsilon_margin_per_notional_by_head,
            threshold_grid_by_head=threshold_grid_by_head,
            min_pred_delta_grid_by_head=min_pred_delta_grid_by_head,
            allowed_multipliers_by_head=allowed_multipliers_by_head,
            min_train_groups_by_head=min_train_groups_by_head,
            min_positive_groups_by_head=min_positive_groups_by_head,
            min_threshold_keep_by_head=min_threshold_keep_by_head,
            threshold_holdout_frac_by_head=threshold_holdout_frac_by_head,
            max_group_features_by_head=max_group_features_by_head,
            max_action_features_by_head=max_action_features_by_head,
            eval_keep_multiplier_by_head=eval_keep_multiplier_by_head,
            max_eval_keep_share_by_head=max_eval_keep_share_by_head,
            action_model_objective_by_head=action_model_objective_by_head,
            action_quantile_alpha_by_head=action_quantile_alpha_by_head,
            action_score_mode_by_head=action_score_mode_by_head,
            action_positive_epsilon_by_head=action_positive_epsilon_by_head,
            fallback_thresholds=fallback_thresholds,
            fallback_min_delta_by_head=fallback_min_delta_by_head,
        )
        for head in HEADS
    }

    train_panel = _load_training_panels(list(args.train_action_panels))
    eval_panel = _load_eval_actions(args.eval_action_features, start=start, end=end)
    train_panel = train_panel.loc[train_panel["timestamp"].lt(end)].copy()
    train_group_cache: dict[str, pd.DataFrame] = {}
    eval_group_cache: dict[str, pd.DataFrame] = {}
    for head in HEADS:
        if active_heads and head not in active_heads:
            continue
        head_config = head_configs[head]
        train_head = train_panel.loc[train_panel["head"].eq(head)].copy()
        eval_head = eval_panel.loc[eval_panel["head"].eq(head)].copy()
        print(
            f"[c3el] precompute groups head={head} train_rows={len(train_head)} eval_rows={len(eval_head)}",
            flush=True,
        )
        train_group_cache[head] = _group_features_from_panel(
            train_head,
            epsilon_gain=head_config["epsilon_gain"],
            epsilon_margin=head_config["epsilon_margin"],
            epsilon_gain_per_notional=head_config["epsilon_gain_per_notional"],
            epsilon_margin_per_notional=head_config["epsilon_margin_per_notional"],
        )
        eval_groups = _group_features_from_panel(
            eval_head,
            epsilon_gain=head_config["epsilon_gain"],
            epsilon_margin=head_config["epsilon_margin"],
            epsilon_gain_per_notional=head_config["epsilon_gain_per_notional"],
            epsilon_margin_per_notional=head_config["epsilon_margin_per_notional"],
        )
        if not eval_groups.empty:
            eval_groups["fold_week_start"] = _week_start(eval_groups["timestamp"])
        eval_group_cache[head] = eval_groups
    params, _payload = _load_policy_params(args.policy_manifest, args.policy_variant)
    candidates = _load_candidates(args.broad_candidates, start=start, end=end)
    deployable = normalise_candidate_table(pd.read_parquet(args.deployable_candidates))
    deployable_train = _pre_start_deployable_candidates(deployable, start=start)
    ev_curve = fit_hierarchical_ev_curves(deployable_train)

    schedule = eval_panel[["timestamp", "strategy_id", "head"]].drop_duplicates().copy()
    schedule["multiplier"] = 1.0
    fold_rows: list[dict[str, Any]] = []
    score_parts: list[pd.DataFrame] = []
    feature_rows: list[dict[str, Any]] = []
    threshold_trial_rows: list[dict[str, Any]] = []

    eval_panel["fold_week_start"] = _week_start(eval_panel["timestamp"])
    weeks = sorted(eval_panel["fold_week_start"].dropna().unique())
    for fold_idx, week in enumerate(weeks):
        week_ts = pd.Timestamp(week)
        week_ts = week_ts.tz_localize("UTC") if week_ts.tzinfo is None else week_ts.tz_convert("UTC")
        cutoff = max(week_ts, start)
        train_before_week = train_panel.loc[train_panel["timestamp"].lt(cutoff)].copy()
        eval_week = eval_panel.loc[eval_panel["fold_week_start"].eq(week)].copy()
        if eval_week.empty:
            continue
        for head in HEADS:
            if active_heads and head not in active_heads:
                continue
            head_eval = eval_week.loc[eval_week["head"].eq(head)].copy()
            if head_eval.empty:
                continue
            head_config = head_configs[head]
            print(f"[c3el] fold={fold_idx} week={week} head={head}", flush=True)
            effective_allowed_multipliers = {float(v) for v in head_config["allowed_multipliers"]}
            head_train = train_before_week.loc[train_before_week["head"].eq(head)].copy()
            head_train_groups_all = train_group_cache.get(head, pd.DataFrame())
            head_train_groups = (
                _slice_group_cache(head_train_groups_all, head_train_groups_all["timestamp"].lt(cutoff))
                if not head_train_groups_all.empty
                else pd.DataFrame()
            )
            if head_train_groups.empty:
                fold_rows.append(
                    {
                        "week_start": str(week),
                        "head": head,
                        "used_model": False,
                        "reason": "empty_head_train_groups",
                        "train_groups": 0,
                        "eval_groups": int(head_eval[["timestamp", "strategy_id"]].drop_duplicates().shape[0]),
                    }
                )
                continue
            head_y = _make_group_target(
                head_train_groups,
                mode=head_config["group_target_mode"],
                epsilon_gain=head_config["epsilon_gain"],
                epsilon_gain_per_notional=head_config["epsilon_gain_per_notional"],
            )
            if len(head_train_groups) < int(head_config["min_train_groups"]) or int(head_y.sum()) < int(
                head_config["min_positive_groups"]
            ):
                fold_rows.append(
                    {
                        "week_start": str(week),
                        "head": head,
                        "used_model": False,
                        "reason": "insufficient_head_labels",
                        "train_groups": int(len(head_train_groups)),
                        "train_positive_groups": int(head_y.sum()),
                        "eval_groups": int(head_eval[["timestamp", "strategy_id"]].drop_duplicates().shape[0]),
                    }
                )
                continue

            effective_action_model_objective = str(head_config["action_model_objective"])
            effective_action_quantile_alpha = float(head_config["action_quantile_alpha"])
            effective_action_score_mode = str(head_config["action_score_mode"])
            effective_threshold_grid = list(head_config["threshold_grid"])
            effective_min_pred_delta_grid = list(head_config["min_pred_delta_grid"])

            unique_ts = pd.Index(pd.to_datetime(head_train_groups["timestamp"], utc=True).dropna().drop_duplicates().sort_values())
            holdout_frac = min(max(float(head_config["threshold_holdout_frac"]), 0.0), 0.8)
            if len(unique_ts) >= 8 and holdout_frac > 0.0:
                holdout_n = max(1, int(np.ceil(len(unique_ts) * holdout_frac)))
                holdout_start = pd.Timestamp(unique_ts[-holdout_n])
                fit_panel = head_train.loc[head_train["timestamp"].lt(holdout_start)].copy()
                threshold_panel = head_train.loc[head_train["timestamp"].ge(holdout_start)].copy()
                fit_groups = _slice_group_cache(head_train_groups, head_train_groups["timestamp"].lt(holdout_start))
                threshold_eval_groups = _slice_group_cache(
                    head_train_groups,
                    head_train_groups["timestamp"].ge(holdout_start),
                )
            else:
                holdout_start = pd.NaT
                fit_panel = head_train.copy()
                threshold_panel = head_train.copy()
                fit_groups = head_train_groups.copy()
                threshold_eval_groups = head_train_groups.copy()
            bundle_for_threshold, threshold_groups, threshold_train_panel = _fit_head_models(
                fit_panel if not fit_panel.empty else head_train,
                head=head,
                seed=int(args.seed) + 101 * fold_idx,
                max_group_features=int(head_config["max_group_features"]),
                max_action_features=int(head_config["max_action_features"]),
                group_target_mode=head_config["group_target_mode"],
                epsilon_gain=head_config["epsilon_gain"],
                epsilon_margin=head_config["epsilon_margin"],
                epsilon_gain_per_notional=head_config["epsilon_gain_per_notional"],
                epsilon_margin_per_notional=head_config["epsilon_margin_per_notional"],
                action_model_objective=effective_action_model_objective,
                action_quantile_alpha=effective_action_quantile_alpha,
                action_positive_epsilon=float(head_config["action_positive_epsilon"]),
                precomputed_groups=fit_groups if not fit_groups.empty else head_train_groups,
            )
            if bundle_for_threshold is None:
                fold_rows.append(
                    {
                        "week_start": str(week),
                        "head": head,
                        "used_model": False,
                        "reason": "fit_failed",
                        "train_groups": int(len(head_train_groups)),
                        "eval_groups": int(head_eval[["timestamp", "strategy_id"]].drop_duplicates().shape[0]),
                    }
                )
                continue
            if threshold_eval_groups.empty:
                threshold_eval_groups = threshold_groups
                threshold_panel_for_value = threshold_train_panel
            else:
                threshold_panel_for_value = threshold_panel
            p_holdout = _predict(
                bundle_for_threshold["group_model"],
                threshold_eval_groups,
                bundle_for_threshold["group_features"],
                bundle_for_threshold["group_medians"],
                proba=True,
            )
            action_scores_holdout = _score_actions(
                bundle_for_threshold["action_model"],
                bundle_for_threshold["action_features"],
                bundle_for_threshold["action_medians"],
                bundle_for_threshold.get("action_positive_model"),
                bundle_for_threshold.get("action_positive_features"),
                bundle_for_threshold.get("action_positive_medians"),
                threshold_panel_for_value,
                action_score_mode=effective_action_score_mode,
            )
            threshold, threshold_diag = _choose_threshold(
                threshold_eval_groups,
                threshold_panel_for_value,
                p_holdout,
                action_scores_holdout,
                grid=effective_threshold_grid,
                min_keep=int(head_config["min_threshold_keep"]),
                min_pred_delta_grid=effective_min_pred_delta_grid,
                allowed_multipliers=effective_allowed_multipliers,
                feature_min_rules=threshold_action_feature_min_by_head.get(head, {}),
                feature_max_rules=threshold_action_feature_max_by_head.get(head, {}),
            )
            for trial in threshold_diag.get("threshold_trials", []):
                threshold_trial_rows.append(
                    {
                        "week_start": str(week),
                        "head": head,
                        "holdout_start": str(holdout_start) if pd.notna(holdout_start) else "",
                        "threshold_holdout_groups": int(len(threshold_eval_groups)),
                        **trial,
                    }
                )

            effective_min_pred_delta = float(threshold_diag.get("min_pred_delta", float(args.min_pred_delta)))
            bundle, groups_full, _panel_full = _fit_head_models(
                head_train,
                head=head,
                seed=int(args.seed) + 101 * fold_idx + 7,
                max_group_features=int(head_config["max_group_features"]),
                max_action_features=int(head_config["max_action_features"]),
                group_target_mode=head_config["group_target_mode"],
                epsilon_gain=head_config["epsilon_gain"],
                epsilon_margin=head_config["epsilon_margin"],
                epsilon_gain_per_notional=head_config["epsilon_gain_per_notional"],
                epsilon_margin_per_notional=head_config["epsilon_margin_per_notional"],
                action_model_objective=effective_action_model_objective,
                action_quantile_alpha=effective_action_quantile_alpha,
                action_positive_epsilon=float(head_config["action_positive_epsilon"]),
                precomputed_groups=head_train_groups,
            )
            if bundle is None:
                continue
            head_eval_groups_all = eval_group_cache.get(head, pd.DataFrame())
            head_eval_groups = (
                _slice_group_cache(head_eval_groups_all, head_eval_groups_all["fold_week_start"].eq(week))
                if not head_eval_groups_all.empty and "fold_week_start" in head_eval_groups_all.columns
                else pd.DataFrame()
            )
            selected = _score_head(
                bundle,
                head_eval,
                threshold=float(threshold),
                min_pred_delta=effective_min_pred_delta,
                allowed_multipliers=effective_allowed_multipliers,
                epsilon_gain=head_config["epsilon_gain"],
                epsilon_margin=head_config["epsilon_margin"],
                epsilon_gain_per_notional=head_config["epsilon_gain_per_notional"],
                epsilon_margin_per_notional=head_config["epsilon_margin_per_notional"],
                action_score_mode=effective_action_score_mode,
                threshold_feature_min_rules=threshold_action_feature_min_by_head.get(head, {}),
                threshold_feature_max_rules=threshold_action_feature_max_by_head.get(head, {}),
                precomputed_groups=head_eval_groups,
            )
            if selected.empty:
                continue
            holdout_groups_n = max(int(len(threshold_eval_groups)), 1)
            raw_threshold_keep = int(threshold_diag["threshold_keep"])
            threshold_has_positive_value = bool(threshold_diag.get("threshold_has_positive_value", False))
            fallback_used = False
            fallback_suppressed = bool(raw_threshold_keep <= 0 and head in fallback_thresholds)
            effective_fallback_max_keep_share = float(
                fallback_max_keep_share_by_head.get(head, float(args.fallback_max_eval_keep_share))
            )
            if raw_threshold_keep <= 0:
                max_eval_keep = 0
            else:
                scaled_keep = np.ceil(
                    raw_threshold_keep
                    * (float(len(selected)) / float(holdout_groups_n))
                    * max(float(head_config["eval_keep_multiplier"]), 0.0)
                )
                share_keep = np.floor(max(float(head_config["max_eval_keep_share"]), 0.0) * float(len(selected)))
                max_eval_keep = int(max(1, min(scaled_keep, share_keep if share_keep > 0 else scaled_keep)))
            selected = _cap_selected_actions(selected, max_keep=max_eval_keep)
            effective_guard_candidate_count = guard_candidate_count_by_head.get(
                head, float(args.guard_low_strategy_candidate_count_max)
            )
            effective_guard_removed_share = guard_removed_share_by_head.get(
                head, float(args.guard_min_removed_trade_share_timestamp)
            )
            selected, guarded_count = _apply_low_breadth_share_guard(
                selected,
                head_eval,
                max_strategy_candidate_count=effective_guard_candidate_count,
                min_removed_trade_share_timestamp=effective_guard_removed_share,
            )
            action_feature_min_rules = action_feature_min_by_head.get(head, {})
            selected, action_feature_min_guarded_count = _apply_action_feature_min_guard(
                selected,
                head_eval,
                min_rules=action_feature_min_rules,
            )
            action_feature_max_rules = action_feature_max_by_head.get(head, {})
            selected, action_feature_max_guarded_count = _apply_action_feature_max_guard(
                selected,
                head_eval,
                max_rules=action_feature_max_rules,
            )
            selected["week_start"] = week
            score_parts.append(selected)
            keep = selected.loc[selected["gate_keep"], ["timestamp", "strategy_id", "selected_multiplier"]].copy()
            if not keep.empty:
                keep["key_multiplier"] = pd.to_numeric(keep["selected_multiplier"], errors="coerce").fillna(1.0).clip(0.0, 1.0)
                sched_idx = pd.MultiIndex.from_frame(schedule[["timestamp", "strategy_id"]])
                keep_idx = pd.MultiIndex.from_frame(keep[["timestamp", "strategy_id"]])
                mask = sched_idx.isin(keep_idx)
                mult_map = keep.set_index(["timestamp", "strategy_id"])["key_multiplier"]
                schedule.loc[mask, "multiplier"] = [
                    float(mult_map.loc[(ts, sid)]) for ts, sid in schedule.loc[mask, ["timestamp", "strategy_id"]].itertuples(index=False, name=None)
                ]
            fold_rows.append(
                {
                    "week_start": str(week),
                    "head": head,
                    "used_model": True,
                    "holdout_start": str(holdout_start) if pd.notna(holdout_start) else "",
                    "train_groups": int(len(groups_full)),
                    "train_positive_groups": int(bundle["train_positive_groups"]),
                    "train_positive_group_rate": float(bundle["train_positive_group_rate"]),
                    "train_actions": int(bundle["train_actions"]),
                    "eval_groups": int(len(selected)),
                    "kept_eval_groups": int(selected["gate_keep"].sum()),
                    "threshold": float(threshold),
                    "fallback_used": bool(fallback_used),
                    "fallback_suppressed": bool(fallback_suppressed),
                    "effective_min_pred_delta": float(effective_min_pred_delta),
                    "effective_fallback_max_keep_share": float(effective_fallback_max_keep_share),
                    "threshold_value": float(threshold_diag["threshold_value"]),
                    "threshold_keep": int(threshold_diag["threshold_keep"]),
                    "threshold_has_positive_value": bool(threshold_has_positive_value),
                    "threshold_min_pred_delta": float(threshold_diag.get("min_pred_delta", effective_min_pred_delta)),
                    "threshold_trial_count": int(len(threshold_diag.get("threshold_trials", []))),
                    "threshold_holdout_groups": int(len(threshold_eval_groups)),
                    "max_eval_keep": int(max_eval_keep),
                    "guarded_eval_groups": int(guarded_count),
                    "action_feature_min_guarded_eval_groups": int(action_feature_min_guarded_count),
                    "action_feature_max_guarded_eval_groups": int(action_feature_max_guarded_count),
                    "effective_guard_low_strategy_candidate_count_max": (
                        None
                        if not np.isfinite(float(effective_guard_candidate_count))
                        else float(effective_guard_candidate_count)
                    ),
                    "effective_guard_min_removed_trade_share_timestamp": (
                        None if not np.isfinite(float(effective_guard_removed_share)) else float(effective_guard_removed_share)
                    ),
                    "group_feature_count": int(len(bundle["group_features"])),
                    "action_feature_count": int(len(bundle["action_features"])),
                    "effective_group_target_mode": str(head_config["group_target_mode"]),
                    "effective_epsilon_gain": float(head_config["epsilon_gain"]),
                    "effective_epsilon_margin": float(head_config["epsilon_margin"]),
                    "effective_epsilon_gain_per_notional": float(head_config["epsilon_gain_per_notional"]),
                    "effective_epsilon_margin_per_notional": float(head_config["epsilon_margin_per_notional"]),
                    "effective_threshold_grid": json.dumps(head_config["threshold_grid"]),
                    "effective_min_pred_delta_grid": json.dumps(head_config["min_pred_delta_grid"]),
                    "effective_allowed_multipliers": json.dumps(head_config["allowed_multipliers"]),
                    "effective_min_train_groups": int(head_config["min_train_groups"]),
                    "effective_min_positive_groups": int(head_config["min_positive_groups"]),
                    "effective_min_threshold_keep": int(head_config["min_threshold_keep"]),
                    "effective_threshold_holdout_frac": float(head_config["threshold_holdout_frac"]),
                    "effective_max_group_features": int(head_config["max_group_features"]),
                    "effective_max_action_features": int(head_config["max_action_features"]),
                    "effective_eval_keep_multiplier": float(head_config["eval_keep_multiplier"]),
                    "effective_max_eval_keep_share": float(head_config["max_eval_keep_share"]),
                    "effective_action_model_objective": str(effective_action_model_objective),
                    "effective_action_quantile_alpha": float(effective_action_quantile_alpha),
                    "effective_action_score_mode": str(effective_action_score_mode),
                    "effective_action_positive_epsilon": float(head_config["action_positive_epsilon"]),
                    "action_positive_feature_count": int(len(bundle["action_positive_features"])),
                    "action_positive_rows": int(bundle["action_positive_diag"].get("positive_rows", 0)),
                    "action_positive_rate": float(bundle["action_positive_diag"].get("positive_rate", 0.0)),
                    "action_feature_min_rules": json.dumps(action_feature_min_rules, sort_keys=True),
                    "action_feature_max_rules": json.dumps(action_feature_max_rules, sort_keys=True),
                    "threshold_action_feature_min_rules": json.dumps(
                        threshold_action_feature_min_by_head.get(head, {}),
                        sort_keys=True,
                    ),
                    "threshold_action_feature_max_rules": json.dumps(
                        threshold_action_feature_max_by_head.get(head, {}),
                        sort_keys=True,
                    ),
                    "group_model_constant": bool(bundle["group_diag"].get("constant", False)),
                    "action_model_constant": bool(bundle["action_diag"].get("constant", False)),
                }
            )
            feature_rows.append(
                {
                    "week_start": str(week),
                    "head": head,
                    "group_features": json.dumps(bundle["group_features"]),
                    "action_features": json.dumps(bundle["action_features"]),
                    "action_positive_features": json.dumps(bundle["action_positive_features"]),
                }
            )

    baseline, _baseline_metrics = _replay(candidates, params, ev_curve, market_mode=args.market_mode, arm="C0_baseline")
    raw_schedule = schedule.copy()
    applied_schedule = _schedule_for_heads(raw_schedule, selected_heads)
    schedule_summaries = [
        _schedule_action_summary(raw_schedule, label="raw_scored"),
        _schedule_action_summary(applied_schedule, label="applied"),
    ]

    head_native, _native_metrics = _replay(
        candidates,
        params,
        ev_curve,
        market_mode=args.market_mode,
        arm="C3el_head_native",
        schedule=applied_schedule[["timestamp", "strategy_id", "multiplier"]],
    )
    selected_accepted_frames = [baseline, head_native]
    isolated_replay_metrics: dict[str, Any] = {}
    if selected_heads != active_heads:
        all_scored, all_scored_metrics = _replay(
            candidates,
            params,
            ev_curve,
            market_mode=args.market_mode,
            arm="C3el_head_native_all_scored",
            schedule=raw_schedule[["timestamp", "strategy_id", "multiplier"]],
        )
        selected_accepted_frames.append(all_scored)
        isolated_replay_metrics["all_scored"] = all_scored_metrics
    if bool(args.head_isolated_replays):
        for head in sorted(active_heads):
            head_schedule = _schedule_for_heads(raw_schedule, {head})
            schedule_summaries.append(_schedule_action_summary(head_schedule, label=f"only_{head}"))
            head_accepted, head_metrics = _replay(
                candidates,
                params,
                ev_curve,
                market_mode=args.market_mode,
                arm=f"C3el_head_native_only_{head}",
                schedule=head_schedule[["timestamp", "strategy_id", "multiplier"]],
            )
            selected_accepted_frames.append(head_accepted)
            isolated_replay_metrics[f"only_{head}"] = head_metrics
    accepted_all = pd.concat(selected_accepted_frames, ignore_index=True)
    accepted_all.to_csv(args.out_dir / "accepted_trades.csv", index=False)
    raw_schedule.to_csv(args.out_dir / "head_native_raw_size_schedule.csv", index=False)
    applied_schedule.to_csv(args.out_dir / "head_native_size_schedule.csv", index=False)
    pd.concat([frame for frame in schedule_summaries if not frame.empty], ignore_index=True).to_csv(
        args.out_dir / "head_native_schedule_summary.csv",
        index=False,
    )
    if score_parts:
        pd.concat(score_parts, ignore_index=True).to_csv(args.out_dir / "head_native_group_scores.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.out_dir / "head_native_folds.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(args.out_dir / "head_native_selected_features.csv", index=False)
    if threshold_trial_rows:
        pd.DataFrame(threshold_trial_rows).to_csv(args.out_dir / "head_native_threshold_trials.csv", index=False)
    for keys, name in [
        (["arm"], "overall"),
        (["arm", "head"], "by_head"),
        (["arm", "week_start"], "weekly"),
        (["arm", "week_start", "head"], "weekly_by_head"),
        (["arm", "month"], "monthly"),
        (["arm", "month", "head"], "monthly_by_head"),
    ]:
        _summarise(accepted_all, keys).to_csv(args.out_dir / f"{name}.csv", index=False)
    manifest = {
        "generated_by": "run_head_native_c3el_action_learner",
        "preset": str(args.preset),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "train_action_panels": [str(p) for p in args.train_action_panels],
        "eval_action_features": str(args.eval_action_features),
        "broad_candidates": str(args.broad_candidates),
        "deployable_candidates": str(args.deployable_candidates),
        "policy_manifest": str(args.policy_manifest),
        "policy_variant": str(args.policy_variant),
        "market_mode": str(args.market_mode),
        "train_rows": int(len(train_panel)),
        "eval_action_rows": int(len(eval_panel)),
        "ev_curve_train_rows": int(len(deployable_train)),
        "schedule_groups": int(len(schedule)),
        "raw_scored_interventions": int(pd.to_numeric(raw_schedule["multiplier"], errors="coerce").fillna(1.0).lt(1.0).sum()),
        "interventions": int(pd.to_numeric(applied_schedule["multiplier"], errors="coerce").fillna(1.0).lt(1.0).sum()),
        "c3el_contract": "head_native",
        "head_configs": head_configs,
        "active_head_configs": {
            head: head_configs[head]
            for head in sorted(active_heads if active_heads else set(HEADS))
            if head in head_configs
        },
        "group_target_mode": str(args.group_target_mode),
        "group_target_mode_by_head": group_target_mode_by_head,
        "epsilon_gain": float(args.epsilon_gain),
        "epsilon_gain_by_head": epsilon_gain_by_head,
        "epsilon_margin": float(args.epsilon_margin),
        "epsilon_margin_by_head": epsilon_margin_by_head,
        "epsilon_gain_per_notional": float(args.epsilon_gain_per_notional),
        "epsilon_gain_per_notional_by_head": epsilon_gain_per_notional_by_head,
        "epsilon_margin_per_notional": float(args.epsilon_margin_per_notional),
        "epsilon_margin_per_notional_by_head": epsilon_margin_per_notional_by_head,
        "threshold_grid": threshold_grid,
        "threshold_grid_by_head": threshold_grid_by_head,
        "min_pred_delta": float(args.min_pred_delta),
        "min_pred_delta_grid": base_min_pred_delta_grid,
        "min_pred_delta_grid_by_head": min_pred_delta_grid_by_head,
        "eval_keep_multiplier": float(args.eval_keep_multiplier),
        "eval_keep_multiplier_by_head": eval_keep_multiplier_by_head,
        "max_eval_keep_share": float(args.max_eval_keep_share),
        "max_eval_keep_share_by_head": max_eval_keep_share_by_head,
        "fallback_thresholds": fallback_thresholds,
        "fallback_max_eval_keep_share": float(args.fallback_max_eval_keep_share),
        "fallback_max_eval_keep_share_by_head": fallback_max_keep_share_by_head,
        "fallback_min_pred_delta": float(args.fallback_min_pred_delta),
        "fallback_min_pred_delta_by_head": fallback_min_delta_by_head,
        "guard_low_strategy_candidate_count_max": (
            None
            if not np.isfinite(float(args.guard_low_strategy_candidate_count_max))
            else float(args.guard_low_strategy_candidate_count_max)
        ),
        "guard_low_strategy_candidate_count_max_by_head": guard_candidate_count_by_head,
        "guard_min_removed_trade_share_timestamp": (
            None
            if not np.isfinite(float(args.guard_min_removed_trade_share_timestamp))
            else float(args.guard_min_removed_trade_share_timestamp)
        ),
        "guard_min_removed_trade_share_timestamp_by_head": guard_removed_share_by_head,
        "allowed_multipliers": sorted(allowed_multipliers),
        "allowed_multipliers_by_head": allowed_multipliers_by_head,
        "active_heads": sorted(active_heads),
        "requested_active_heads": sorted(requested_active_heads),
        "selected_heads": sorted(selected_heads),
        "requested_selected_heads": sorted(requested_selected_heads),
        "head_isolated_replays": bool(args.head_isolated_replays),
        "isolated_replay_metrics": isolated_replay_metrics,
        "min_train_groups": int(args.min_train_groups),
        "min_train_groups_by_head": min_train_groups_by_head,
        "min_positive_groups": int(args.min_positive_groups),
        "min_positive_groups_by_head": min_positive_groups_by_head,
        "min_threshold_keep": int(args.min_threshold_keep),
        "min_threshold_keep_by_head": min_threshold_keep_by_head,
        "threshold_holdout_frac": float(args.threshold_holdout_frac),
        "threshold_holdout_frac_by_head": threshold_holdout_frac_by_head,
        "max_group_features": int(args.max_group_features),
        "max_group_features_by_head": max_group_features_by_head,
        "max_action_features": int(args.max_action_features),
        "max_action_features_by_head": max_action_features_by_head,
        "action_model_objective": str(args.action_model_objective),
        "action_model_objective_by_head": action_model_objective_by_head,
        "action_quantile_alpha": float(args.action_quantile_alpha),
        "action_quantile_alpha_by_head": action_quantile_alpha_by_head,
        "action_score_mode": str(args.action_score_mode),
        "action_score_mode_by_head": action_score_mode_by_head,
        "action_positive_epsilon": float(args.action_positive_epsilon),
        "action_positive_epsilon_by_head": action_positive_epsilon_by_head,
        "threshold_action_feature_min_by_head": threshold_action_feature_min_by_head,
        "threshold_action_feature_max_by_head": threshold_action_feature_max_by_head,
        "action_feature_min_by_head": action_feature_min_by_head,
        "action_feature_max_by_head": action_feature_max_by_head,
    }
    _write_json(args.out_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    print(_summarise(accepted_all, ["arm"]).to_string(index=False))


if __name__ == "__main__":
    main()
