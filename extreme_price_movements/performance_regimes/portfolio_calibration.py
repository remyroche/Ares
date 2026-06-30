"""Portfolio-level threshold/rank/weight/gate calibration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CrossStrategyArchetypeFeatures:
    X: pd.DataFrame
    metadata: pd.DataFrame


@dataclass(frozen=True)
class ThresholdedArchetypeModulationInputs:
    p_active: pd.DataFrame
    activity_scores: pd.DataFrame
    modulation_scores: pd.DataFrame
    diagnostics: pd.DataFrame


@dataclass(frozen=True)
class PortfolioActionTargetBundle:
    by_strategy: dict[str, pd.DataFrame]
    diagnostics: pd.DataFrame


@dataclass(frozen=True)
class PortfolioCalibratorConfig:
    backend: Literal["optuna", "ebm_gam", "linear"] = "linear"
    activation_cutoff: float = 0.0
    allow_cash: bool = True
    renormalize_nonzero: bool = True
    archetype_score_threshold: float = 0.0
    archetype_score_ramp_power: float = 1.0
    archetype_score_ramp_gain: float = 1.0
    archetype_base_p_active_floor: float = 0.0
    optuna_tune_archetype_score_threshold: bool = True
    optuna_archetype_score_threshold_range: tuple[float, float] = (0.0, 0.8)
    optuna_tune_archetype_score_ramp: bool = True
    optuna_archetype_score_ramp_power_range: tuple[float, float] = (0.5, 3.0)
    optuna_archetype_score_ramp_gain_range: tuple[float, float] = (0.25, 4.0)
    optuna_archetype_nonzero_penalty: float = 0.0
    optuna_trials: int = 20
    optuna_objective: Literal["mse", "hybrid", "portfolio"] = "hybrid"
    optuna_mse_weight: float = 0.25
    optuna_ev_weight: float = 1.0
    optuna_hit_rate_weight: float = 1.0
    optuna_loss_streak_weight: float = 2.0
    optuna_loss_streak_hours: float = 72.0
    optuna_downside_weight: float = 1.0
    optuna_turnover_weight: float = 0.05
    optuna_cash_share_target: float = 1.0
    optuna_cash_share_weight: float = 0.0
    optuna_cash_share_excess_power: float = 1.0
    optuna_unjustified_deactivation_weight: float = 0.0
    optuna_unjustified_deactivation_gate_margin: float = 0.0
    optuna_active_utility_lcb_weight: float = 0.0
    optuna_active_utility_lcb_z: float = 1.0
    optuna_loss_density_weight: float = 0.0
    optuna_loss_density_window_hours: float = 72.0
    optuna_loss_density_target: float = 0.50
    threshold_delta_clip: tuple[float, float] = (-0.50, 0.50)
    rank_delta_clip: tuple[float, float] = (-1.0, 1.0)
    weight_log_delta_clip: tuple[float, float] = (-8.0, 4.0)
    activation_gate_clip: tuple[float, float] = (-8.0, 8.0)
    random_state: int = 42


@dataclass(frozen=True)
class PortfolioCalibrator:
    strategies: tuple[str, ...]
    feature_columns: tuple[str, ...]
    config: PortfolioCalibratorConfig
    coefficients: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    intercepts: dict[str, dict[str, float]] = field(default_factory=dict)
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    effective_backend: str = "linear"
    diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)


def _is_archetype_score_column(column: str) -> bool:
    text = str(column)
    return text.startswith("strategy_") and (
        "_archetype_" in text or "_compressed_archetype_" in text
    )


def _threshold_archetype_score_frame(
    X: pd.DataFrame,
    *,
    threshold: float,
    ramp_power: float = 1.0,
    ramp_gain: float = 1.0,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    archetype_cols = tuple(col for col in X.columns if _is_archetype_score_column(str(col)))
    if not archetype_cols:
        return X, archetype_cols
    out = X.copy()
    threshold = float(np.clip(threshold, -1.0, 1.0))
    ramp_power = max(float(ramp_power), 1e-6)
    ramp_gain = float(np.clip(ramp_gain, 0.0, 16.0))
    values = out.loc[:, list(archetype_cols)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    normalized_exceedance = ((values - threshold) / max(1.0 - threshold, 1e-6)).clip(0.0, 1.0)
    out.loc[:, list(archetype_cols)] = ramp_gain * np.power(normalized_exceedance, ramp_power)
    return out, archetype_cols


def _action_feature_frame(
    X: pd.DataFrame,
    config: PortfolioCalibratorConfig,
    model_payload: Any | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...], float, float, float]:
    threshold = float(config.archetype_score_threshold)
    ramp_power = float(config.archetype_score_ramp_power)
    ramp_gain = float(config.archetype_score_ramp_gain)
    if isinstance(model_payload, dict) and "archetype_score_threshold" in model_payload:
        threshold = float(model_payload.get("archetype_score_threshold", threshold))
        ramp_power = float(model_payload.get("archetype_score_ramp_power", ramp_power))
        ramp_gain = float(model_payload.get("archetype_score_ramp_gain", ramp_gain))
    transformed, archetype_cols = _threshold_archetype_score_frame(
        X,
        threshold=threshold,
        ramp_power=ramp_power,
        ramp_gain=ramp_gain,
    )
    return transformed, archetype_cols, threshold, ramp_power, ramp_gain


def threshold_archetype_scores_for_modulation(
    p_active: pd.DataFrame,
    *,
    min_p_active: float = 0.50,
    min_active_share: float = 0.0,
    relax_floor_to_min_active_share: bool = True,
) -> ThresholdedArchetypeModulationInputs:
    """Convert p-active scores into neutral-below-threshold modulation inputs.

    Below-threshold archetypes are treated as absent rather than as strongly
    inactive.  The centered activity score therefore becomes zero below the
    p-active floor, while above-threshold values keep the 2*p-1 interpretation.
    ``modulation_scores`` additionally expose normalized exceedance above the
    effective p-active threshold, which is the portfolio-calibrator input.
    """

    p = (
        p_active.replace([np.inf, -np.inf], np.nan)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .clip(0.0, 1.0)
        .astype(np.float32, copy=False)
    )
    requested_threshold = float(np.clip(min_p_active, 0.0, 1.0))
    min_active_share = float(np.clip(min_active_share, 0.0, 1.0))
    effective_thresholds: dict[str, float] = {}
    relaxed_columns: dict[str, bool] = {}
    requested_active_shares: dict[str, float] = {}
    active_columns: dict[str, pd.Series] = {}
    modulation_columns: dict[str, pd.Series] = {}
    for column in p.columns:
        series = p[column]
        threshold = requested_threshold
        requested_active = series >= threshold
        requested_active_share = float(requested_active.mean())
        relaxed = False
        if (
            relax_floor_to_min_active_share
            and min_active_share > 0.0
            and requested_active_share < min_active_share
            and series.nunique(dropna=True) > 1
        ):
            original_threshold = threshold
            threshold = min(threshold, float(series.quantile(1.0 - min_active_share)))
            relaxed = bool(threshold < original_threshold - 1e-12)
        threshold = float(np.clip(threshold, 0.0, 1.0))
        active = series >= threshold
        effective_thresholds[str(column)] = threshold
        relaxed_columns[str(column)] = relaxed
        requested_active_shares[str(column)] = requested_active_share
        active_columns[str(column)] = active
        modulation_columns[str(column)] = ((series - threshold) / max(1.0 - threshold, 1e-6)).clip(0.0, 1.0)
    active = pd.DataFrame(active_columns, index=p.index).reindex(columns=p.columns).fillna(False)
    modulation_scores = (
        pd.DataFrame(modulation_columns, index=p.index)
        .reindex(columns=p.columns)
        .fillna(0.0)
        .astype(np.float32, copy=False)
    )
    thresholded_p = p.where(active, 0.0).astype(np.float32, copy=False)
    activity_scores = ((2.0 * p - 1.0).where(active, 0.0)).clip(-1.0, 1.0).astype(np.float32, copy=False)
    rows = []
    for column in p.columns:
        mask = active[column]
        effective_threshold = float(effective_thresholds.get(str(column), requested_threshold))
        rows.append(
            {
                "archetype_id": str(column),
                "requested_min_p_active": requested_threshold,
                "effective_min_p_active": effective_threshold,
                "min_active_share": min_active_share,
                "relax_floor_to_min_active_share": bool(relax_floor_to_min_active_share),
                "floor_relaxed_to_min_active_share": bool(relaxed_columns.get(str(column), False)),
                "requested_active_share": float(requested_active_shares.get(str(column), np.nan)),
                "p_active_mean_before_threshold": float(p[column].mean()),
                "p_active_mean_after_threshold": float(thresholded_p[column].mean()),
                "active_share_after_threshold": float(mask.mean()),
                "suppressed_share": float((~mask).mean()),
                "activity_score_nonzero_share": float(activity_scores[column].ne(0.0).mean()),
                "modulation_score_nonzero_share": float(modulation_scores[column].ne(0.0).mean()),
                "modulation_score_max": float(modulation_scores[column].max()),
            }
        )
    return ThresholdedArchetypeModulationInputs(
        p_active=thresholded_p,
        activity_scores=activity_scores,
        modulation_scores=modulation_scores,
        diagnostics=pd.DataFrame(rows),
    )


def build_portfolio_action_targets_from_labels(
    labels,
    timestamps: pd.Index,
    *,
    strategies: Sequence[str],
    threshold_delta_scale: float = 0.25,
    rank_delta_scale: float = 0.50,
    weight_log_delta_scale: float = 2.0,
    activation_gate_scale: float = 4.0,
    activation_gate_quality_threshold: float = 0.0,
    bad_regime_threshold_penalty_scale: float = 0.0,
    bad_regime_rank_penalty_scale: float = 0.0,
    bad_regime_weight_penalty_scale: float = 0.0,
    bad_regime_activation_penalty_scale: float = 0.0,
    bad_regime_pressure_column: str = "composite_bad_pressure",
) -> PortfolioActionTargetBundle:
    """Convert fold-local performance labels into compact action targets.

    Positive centered quality lowers thresholds, improves rank/weight, and
    opens the activation gate. Negative centered quality does the opposite and
    explicitly teaches full deactivation through negative gate targets.
    """

    index = pd.Index(timestamps)
    by_strategy: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []
    for strategy in [str(s) for s in strategies]:
        label_set = labels.by_strategy.get(strategy)
        if label_set is None:
            quality = pd.Series(0.0, index=index)
            bad_pressure = pd.Series(0.0, index=index)
        else:
            quality = (
                2.0
                * pd.to_numeric(label_set.good_label, errors="coerce")
                .reindex(index)
                .fillna(0.5)
                - 1.0
            ).clip(-1.0, 1.0)
            pressure_source = getattr(label_set, str(bad_regime_pressure_column), None)
            if pressure_source is None:
                pressure_source = getattr(label_set, "composite_bad_pressure", None)
            if pressure_source is None:
                bad_pressure = pd.Series(0.0, index=index)
            else:
                bad_pressure = (
                    pd.to_numeric(pressure_source, errors="coerce")
                    .reindex(index)
                    .fillna(0.0)
                    .clip(0.0, 1.0)
                )
        target = pd.DataFrame(index=index)
        target["threshold_delta"] = (
            -float(threshold_delta_scale) * quality
            + float(bad_regime_threshold_penalty_scale) * bad_pressure
        ).clip(
            -abs(float(threshold_delta_scale)),
            abs(float(threshold_delta_scale)) + abs(float(bad_regime_threshold_penalty_scale)),
        )
        target["rank_delta"] = (
            float(rank_delta_scale) * quality
            - float(bad_regime_rank_penalty_scale) * bad_pressure
        ).clip(
            -abs(float(rank_delta_scale)) - abs(float(bad_regime_rank_penalty_scale)),
            abs(float(rank_delta_scale)),
        )
        target["weight_log_delta"] = (
            float(weight_log_delta_scale) * quality
            - float(bad_regime_weight_penalty_scale) * bad_pressure
        ).clip(
            -abs(float(weight_log_delta_scale)) - abs(float(bad_regime_weight_penalty_scale)),
            abs(float(weight_log_delta_scale)),
        )
        activation_centered_quality = (
            quality
            - float(activation_gate_quality_threshold)
            - float(bad_regime_activation_penalty_scale) * bad_pressure
        )
        target["activation_gate"] = float(activation_gate_scale) * activation_centered_quality
        by_strategy[strategy] = target.astype(np.float32, copy=False)
        rows.append(
            {
                "strategy": strategy,
                "quality_mean": float(quality.mean()),
                "quality_std": float(quality.std(ddof=0)),
                "bad_regime_pressure_column": str(bad_regime_pressure_column),
                "bad_regime_pressure_mean": float(bad_pressure.mean()),
                "bad_regime_pressure_active_share": float(bad_pressure.gt(0.0).mean()),
                "activation_gate_quality_threshold": float(activation_gate_quality_threshold),
                "bad_regime_threshold_penalty_scale": float(bad_regime_threshold_penalty_scale),
                "bad_regime_rank_penalty_scale": float(bad_regime_rank_penalty_scale),
                "bad_regime_weight_penalty_scale": float(bad_regime_weight_penalty_scale),
                "bad_regime_activation_penalty_scale": float(bad_regime_activation_penalty_scale),
                "threshold_delta_std": float(target["threshold_delta"].std(ddof=0)),
                "rank_delta_std": float(target["rank_delta"].std(ddof=0)),
                "weight_log_delta_std": float(target["weight_log_delta"].std(ddof=0)),
                "activation_gate_std": float(target["activation_gate"].std(ddof=0)),
                "activation_target_deactivation_share": float((target["activation_gate"] < 0.0).mean()),
            }
        )
    return PortfolioActionTargetBundle(by_strategy, pd.DataFrame(rows))


def normalize_nonzero(weights: pd.DataFrame) -> pd.DataFrame:
    out = weights.copy().astype(np.float32)
    row_sum = out.sum(axis=1)
    mask = row_sum.abs() > 1e-12
    out.loc[mask] = out.loc[mask].div(row_sum.loc[mask], axis=0)
    return out


def _median_step_hours(index: pd.Index) -> float:
    if isinstance(index, pd.DatetimeIndex) and len(index) > 1:
        diffs = index.sort_values().to_series().diff().dropna().dt.total_seconds().to_numpy(dtype=float)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        if len(diffs):
            return max(float(np.nanmedian(diffs) / 3600.0), 1e-9)
    return 1.0


def _max_consecutive_loss_hours(returns: pd.Series, *, step_hours: float | None = None) -> float:
    values = pd.to_numeric(returns, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    bar_hours = float(step_hours if step_hours is not None else _median_step_hours(returns.index))
    max_run = 0
    current = 0
    for value in values:
        if float(value) < 0.0:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return float(max_run * bar_hours)


def _target_action_series(
    target_frame: pd.DataFrame,
    action: str,
    *,
    default: float,
    index: pd.Index,
) -> pd.Series:
    if action in target_frame.columns:
        return pd.to_numeric(target_frame[action], errors="coerce").reindex(index).fillna(default)
    return pd.Series(float(default), index=index)


def _replay_weights_from_action_prediction(
    current_prediction: np.ndarray,
    *,
    current_strategy: str,
    current_action: str,
    strategies: Sequence[str],
    action_targets: Mapping[str, pd.DataFrame],
    index: pd.Index,
    config: PortfolioCalibratorConfig,
) -> pd.DataFrame:
    raw = pd.DataFrame(0.0, index=index, columns=[str(s) for s in strategies], dtype=np.float32)
    for strategy in [str(s) for s in strategies]:
        target_frame = dict(action_targets or {}).get(strategy, pd.DataFrame(index=index))
        gate = _target_action_series(target_frame, "activation_gate", default=1.0, index=index)
        weight_log_delta = _target_action_series(target_frame, "weight_log_delta", default=0.0, index=index)
        if strategy == current_strategy and current_action == "activation_gate":
            gate = pd.Series(current_prediction, index=index).astype(float)
        if strategy == current_strategy and current_action == "weight_log_delta":
            weight_log_delta = pd.Series(current_prediction, index=index).astype(float)
        active = gate >= float(config.activation_cutoff)
        raw[strategy] = np.where(
            active,
            np.exp(np.clip(weight_log_delta.to_numpy(dtype=float), -8.0, 4.0)),
            0.0,
        )
    if bool(config.renormalize_nonzero):
        raw = normalize_nonzero(raw)
    if not bool(config.allow_cash):
        empty = raw.sum(axis=1).abs() <= 1e-12
        if bool(empty.any()) and len(raw.columns):
            raw.loc[empty, :] = 1.0 / float(len(raw.columns))
    return raw.astype(np.float32, copy=False)


def _portfolio_replay_metrics(
    weights: pd.DataFrame,
    strategy_returns: pd.DataFrame,
    *,
    active_utility_lcb_z: float = 1.0,
    loss_density_window_hours: float = 72.0,
    loss_density_target: float = 0.50,
) -> dict[str, float]:
    returns = (
        strategy_returns.reindex(index=weights.index, columns=weights.columns)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32, copy=False)
    )
    portfolio_returns = (weights * returns).sum(axis=1)
    active_weight = weights.sum(axis=1)
    negative = portfolio_returns.clip(upper=0.0)
    step_hours = _median_step_hours(portfolio_returns.index)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    active_returns = portfolio_returns.loc[active_weight.abs().gt(1e-12)]
    if len(active_returns):
        active_mean = float(active_returns.mean())
        active_std = float(active_returns.std(ddof=0))
        active_lcb = active_mean - max(float(active_utility_lcb_z), 0.0) * active_std / np.sqrt(len(active_returns))
    else:
        active_lcb = 0.0
    bars = max(1, int(np.ceil(float(loss_density_window_hours) / step_hours)))
    loss_density = portfolio_returns.lt(0.0).astype(float).rolling(bars, min_periods=1).mean()
    loss_density_excess = (
        loss_density - float(np.clip(loss_density_target, 0.0, 1.0))
    ).clip(lower=0.0)
    return {
        "portfolio_ev": float(portfolio_returns.mean()),
        "portfolio_hit_rate": float((portfolio_returns > 0.0).mean()),
        "portfolio_loss_rate": float((portfolio_returns < 0.0).mean()),
        "portfolio_downside_mean": float((-negative).mean()),
        "portfolio_max_loss_streak_hours": _max_consecutive_loss_hours(
            portfolio_returns,
            step_hours=step_hours,
        ),
        "portfolio_turnover_mean": float(turnover.mean()),
        "portfolio_cash_share": float(active_weight.abs().le(1e-12).mean()),
        "portfolio_mean_active_strategies": float(weights.gt(0.0).sum(axis=1).mean()),
        "portfolio_active_utility_lcb": float(active_lcb),
        "portfolio_active_utility_lcb_shortfall": float(max(0.0, -active_lcb)),
        "portfolio_loss_density_mean": float(loss_density.mean()),
        "portfolio_loss_density_excess_mean": float(loss_density_excess.mean()),
    }


def _portfolio_action_objective_loss(
    pred: np.ndarray,
    *,
    mse_loss: float,
    current_strategy: str | None,
    current_action: str | None,
    strategies: Sequence[str] | None,
    action_targets: Mapping[str, pd.DataFrame] | None,
    strategy_returns: pd.DataFrame | None,
    index: pd.Index,
    config: PortfolioCalibratorConfig,
) -> tuple[float, dict[str, float]]:
    if (
        strategy_returns is None
        or not strategies
        or current_strategy is None
        or current_action not in {"activation_gate", "weight_log_delta"}
        or str(config.optuna_objective) == "mse"
    ):
        return float(mse_loss), {}
    weights = _replay_weights_from_action_prediction(
        pred,
        current_strategy=str(current_strategy),
        current_action=str(current_action),
        strategies=strategies,
        action_targets=action_targets or {},
        index=index,
        config=config,
    )
    metrics = _portfolio_replay_metrics(
        weights,
        strategy_returns,
        active_utility_lcb_z=float(config.optuna_active_utility_lcb_z),
        loss_density_window_hours=float(config.optuna_loss_density_window_hours),
        loss_density_target=float(config.optuna_loss_density_target),
    )
    streak_threshold = max(float(config.optuna_loss_streak_hours), 1e-9)
    streak_excess = max(0.0, metrics["portfolio_max_loss_streak_hours"] - streak_threshold) / streak_threshold
    metrics["portfolio_loss_streak_excess"] = float(streak_excess)
    cash_target = float(np.clip(config.optuna_cash_share_target, 0.0, 1.0))
    cash_excess = max(0.0, metrics["portfolio_cash_share"] - cash_target) / max(1.0 - cash_target, 1e-9)
    cash_excess_power = max(float(config.optuna_cash_share_excess_power), 1e-6)
    cash_penalty = float(cash_excess**cash_excess_power)
    metrics["portfolio_cash_share_excess"] = float(cash_excess)
    metrics["portfolio_cash_share_penalty"] = cash_penalty
    target_frame = dict(action_targets or {}).get(str(current_strategy), pd.DataFrame(index=index))
    target_gate = _target_action_series(target_frame, "activation_gate", default=1.0, index=index)
    target_active_cutoff = float(config.activation_cutoff) + float(
        config.optuna_unjustified_deactivation_gate_margin
    )
    target_should_be_active = target_gate >= target_active_cutoff
    current_active = (
        weights[str(current_strategy)].abs() > 1e-12
        if str(current_strategy) in weights.columns
        else pd.Series(False, index=index)
    )
    unjustified_deactivation_share = float((target_should_be_active & ~current_active).mean())
    metrics["portfolio_current_strategy_target_active_share"] = float(target_should_be_active.mean())
    metrics["portfolio_unjustified_deactivation_share"] = unjustified_deactivation_share
    loss = 0.0
    if str(config.optuna_objective) == "hybrid":
        loss += max(float(config.optuna_mse_weight), 0.0) * float(mse_loss)
    loss -= float(config.optuna_ev_weight) * metrics["portfolio_ev"]
    loss -= float(config.optuna_hit_rate_weight) * metrics["portfolio_hit_rate"]
    loss += max(float(config.optuna_loss_streak_weight), 0.0) * streak_excess
    loss += max(float(config.optuna_downside_weight), 0.0) * metrics["portfolio_downside_mean"]
    loss += max(float(config.optuna_turnover_weight), 0.0) * metrics["portfolio_turnover_mean"]
    loss += max(float(config.optuna_cash_share_weight), 0.0) * cash_penalty
    loss += max(float(config.optuna_unjustified_deactivation_weight), 0.0) * unjustified_deactivation_share
    loss += max(float(config.optuna_active_utility_lcb_weight), 0.0) * metrics[
        "portfolio_active_utility_lcb_shortfall"
    ]
    loss += max(float(config.optuna_loss_density_weight), 0.0) * metrics[
        "portfolio_loss_density_excess_mean"
    ]
    return float(loss), metrics


def _fit_linear_action(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[np.ndarray, float]:
    X_arr = X.to_numpy(dtype=np.float32, copy=False)
    y_arr = pd.to_numeric(y, errors="coerce").reindex(X.index).fillna(0.0).to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(X_arr)), X_arr])
    try:
        coef, *_ = np.linalg.lstsq(design, y_arr, rcond=None)
    except Exception:
        coef = np.zeros(design.shape[1], dtype=float)
    return coef[1:].astype(float), float(coef[0])


def _fit_backend_action(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    config: PortfolioCalibratorConfig,
    strategy: str | None = None,
    action: str | None = None,
    strategies: Sequence[str] | None = None,
    action_targets: Mapping[str, pd.DataFrame] | None = None,
    strategy_returns: pd.DataFrame | None = None,
) -> tuple[np.ndarray, float, Any | None, str]:
    backend = str(config.backend)
    X_base, archetype_cols, base_threshold, base_ramp_power, base_ramp_gain = _action_feature_frame(X, config)
    coef, intercept = _fit_linear_action(X_base, y)
    if backend == "ebm_gam":
        try:
            from interpret.glassbox import ExplainableBoostingRegressor

            model = ExplainableBoostingRegressor(
                interactions=5,
                random_state=int(config.random_state),
            )
            model.fit(X_base, y)
            return coef, intercept, model, "ebm_gam"
        except Exception:
            return coef, intercept, None, "linear_fallback_for_ebm_gam"
    if backend == "optuna":
        try:
            import optuna

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            y_arr = pd.to_numeric(y, errors="coerce").reindex(X.index).fillna(0.0).to_numpy(dtype=np.float32)
            low_raw, high_raw = config.optuna_archetype_score_threshold_range
            low = float(max(min(low_raw, high_raw), -1.0))
            high = float(min(max(low_raw, high_raw), 1.0))
            if not archetype_cols or not bool(config.optuna_tune_archetype_score_threshold):
                low = high = base_threshold
            power_low_raw, power_high_raw = config.optuna_archetype_score_ramp_power_range
            power_low = max(1e-6, float(min(power_low_raw, power_high_raw)))
            power_high = max(power_low, float(max(power_low_raw, power_high_raw)))
            gain_low_raw, gain_high_raw = config.optuna_archetype_score_ramp_gain_range
            gain_low = max(0.0, float(min(gain_low_raw, gain_high_raw)))
            gain_high = max(gain_low, float(max(gain_low_raw, gain_high_raw)))
            if not archetype_cols or not bool(config.optuna_tune_archetype_score_ramp):
                power_low = power_high = base_ramp_power
                gain_low = gain_high = base_ramp_gain

            def objective(trial):
                shrink = trial.suggest_float("coefficient_shrinkage", 0.0, 2.0)
                bias = trial.suggest_float("intercept_shift", -1.0, 1.0)
                threshold = (
                    trial.suggest_float("archetype_score_threshold", low, high)
                    if high > low
                    else float(low)
                )
                ramp_power = (
                    trial.suggest_float("archetype_score_ramp_power", power_low, power_high)
                    if power_high > power_low
                    else float(power_low)
                )
                ramp_gain = (
                    trial.suggest_float("archetype_score_ramp_gain", gain_low, gain_high)
                    if gain_high > gain_low
                    else float(gain_low)
                )
                X_trial, _cols = _threshold_archetype_score_frame(
                    X,
                    threshold=threshold,
                    ramp_power=ramp_power,
                    ramp_gain=ramp_gain,
                )
                trial_coef, trial_intercept = _fit_linear_action(X_trial, y)
                pred = (trial_intercept + bias) + X_trial.to_numpy(dtype=np.float32, copy=False) @ (
                    trial_coef * shrink
                )
                mse_loss = float(np.nanmean((y_arr - pred) ** 2))
                loss, _metrics = _portfolio_action_objective_loss(
                    pred,
                    mse_loss=mse_loss,
                    current_strategy=strategy,
                    current_action=action,
                    strategies=strategies,
                    action_targets=action_targets,
                    strategy_returns=strategy_returns,
                    index=X.index,
                    config=config,
                )
                nonzero_penalty = max(float(config.optuna_archetype_nonzero_penalty), 0.0)
                if nonzero_penalty > 0.0 and archetype_cols:
                    nonzero_share = float(X_trial.loc[:, list(archetype_cols)].ne(0.0).mean().mean())
                    loss += nonzero_penalty * nonzero_share
                if not np.isfinite(loss):
                    return float("inf")
                return float(loss)

            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=int(config.random_state)),
            )
            study.optimize(
                objective,
                n_trials=max(1, int(config.optuna_trials)),
                show_progress_bar=False,
            )
            shrink = float(study.best_params.get("coefficient_shrinkage", 1.0))
            bias = float(study.best_params.get("intercept_shift", 0.0))
            threshold = float(study.best_params.get("archetype_score_threshold", base_threshold))
            ramp_power = float(study.best_params.get("archetype_score_ramp_power", base_ramp_power))
            ramp_gain = float(study.best_params.get("archetype_score_ramp_gain", base_ramp_gain))
            X_best, _cols = _threshold_archetype_score_frame(
                X,
                threshold=threshold,
                ramp_power=ramp_power,
                ramp_gain=ramp_gain,
            )
            best_coef, best_intercept = _fit_linear_action(X_best, y)
            best_pred = (best_intercept + bias) + X_best.to_numpy(dtype=np.float32, copy=False) @ (
                best_coef * shrink
            )
            best_mse = float(np.nanmean((y_arr - best_pred) ** 2))
            best_objective_loss, best_portfolio_metrics = _portfolio_action_objective_loss(
                best_pred,
                mse_loss=best_mse,
                current_strategy=strategy,
                current_action=action,
                strategies=strategies,
                action_targets=action_targets,
                strategy_returns=strategy_returns,
                index=X.index,
                config=config,
            )
            return (
                best_coef * shrink,
                best_intercept + bias,
                {
                    "best_params": dict(study.best_params),
                    "optuna_objective": str(config.optuna_objective),
                    "optuna_best_value": float(study.best_value),
                    "optuna_refit_objective_loss": float(best_objective_loss),
                    "optuna_refit_mse": float(best_mse),
                    "archetype_score_threshold": threshold,
                    "archetype_score_ramp_power": ramp_power,
                    "archetype_score_ramp_gain": ramp_gain,
                    "archetype_base_p_active_floor": float(config.archetype_base_p_active_floor),
                    "archetype_effective_p_active_threshold": float(
                        np.clip(
                            config.archetype_base_p_active_floor
                            + threshold * (1.0 - config.archetype_base_p_active_floor),
                            0.0,
                            1.0,
                        )
                    ),
                    "optuna_archetype_nonzero_penalty": float(config.optuna_archetype_nonzero_penalty),
                    "optuna_cash_share_target": float(config.optuna_cash_share_target),
                    "optuna_cash_share_weight": float(config.optuna_cash_share_weight),
                    "optuna_cash_share_excess_power": float(config.optuna_cash_share_excess_power),
                    "optuna_unjustified_deactivation_weight": float(
                        config.optuna_unjustified_deactivation_weight
                    ),
                    "optuna_unjustified_deactivation_gate_margin": float(
                        config.optuna_unjustified_deactivation_gate_margin
                    ),
                    "optuna_active_utility_lcb_weight": float(config.optuna_active_utility_lcb_weight),
                    "optuna_active_utility_lcb_z": float(config.optuna_active_utility_lcb_z),
                    "optuna_loss_density_weight": float(config.optuna_loss_density_weight),
                    "optuna_loss_density_window_hours": float(config.optuna_loss_density_window_hours),
                    "optuna_loss_density_target": float(config.optuna_loss_density_target),
                    **best_portfolio_metrics,
                    "archetype_feature_count": int(len(archetype_cols)),
                },
                "optuna_compact_linear_thresholded_ramped_archetypes",
            )
        except Exception:
            return coef, intercept, None, "linear_fallback_for_optuna"
    return coef, intercept, None, "linear"


def train_portfolio_calibrator(
    X: pd.DataFrame,
    *,
    strategies: Sequence[str],
    action_targets: Mapping[str, pd.DataFrame] | None = None,
    strategy_returns: pd.DataFrame | None = None,
    config: PortfolioCalibratorConfig = PortfolioCalibratorConfig(),
) -> PortfolioCalibrator:
    """Train compact per-strategy action functions.

    Missing action targets default to neutral actions except activation gates,
    which default to active.  This keeps the action space available without
    inventing arbitrary per-timestamp labels.
    """

    features = tuple(str(c) for c in X.columns)
    X_fit = (
        X.reindex(columns=features)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32, copy=False)
    )
    strategy_names = tuple(str(s) for s in strategies)
    returns_fit = None
    if strategy_returns is not None:
        returns_fit = (
            strategy_returns.reindex(index=X_fit.index, columns=strategy_names)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(np.float32, copy=False)
        )
    coefficients: dict[str, dict[str, np.ndarray]] = {}
    intercepts: dict[str, dict[str, float]] = {}
    models: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, object]] = []
    effective_backends: set[str] = set()
    for strategy in strategy_names:
        target_frame = dict(action_targets or {}).get(strategy, pd.DataFrame(index=X_fit.index))
        coefficients[strategy] = {}
        intercepts[strategy] = {}
        models[strategy] = {}
        for action, default in [
            ("threshold_delta", 0.0),
            ("rank_delta", 0.0),
            ("weight_log_delta", 0.0),
            ("activation_gate", 1.0),
        ]:
            y = (
                pd.to_numeric(target_frame[action], errors="coerce").reindex(X_fit.index)
                if action in target_frame.columns
                else pd.Series(default, index=X_fit.index)
            ).fillna(default)
            coef, intercept, model, effective_backend = _fit_backend_action(
                X_fit,
                y,
                config=config,
                strategy=strategy,
                action=action,
                strategies=strategy_names,
                action_targets=action_targets,
                strategy_returns=returns_fit,
            )
            coefficients[strategy][action] = coef
            intercepts[strategy][action] = intercept
            if model is not None:
                models[strategy][action] = model
            effective_backends.add(effective_backend)
            X_action, archetype_cols, threshold, ramp_power, ramp_gain = _action_feature_frame(X_fit, config, model)
            pred = intercept + X_action.to_numpy(dtype=np.float32, copy=False) @ coef.astype(np.float32, copy=False)
            base_p_active_floor = float(np.clip(config.archetype_base_p_active_floor, 0.0, 1.0))
            if isinstance(model, dict):
                base_p_active_floor = float(
                    np.clip(model.get("archetype_base_p_active_floor", base_p_active_floor), 0.0, 1.0)
                )
            effective_p_active_threshold = float(
                np.clip(base_p_active_floor + threshold * (1.0 - base_p_active_floor), 0.0, 1.0)
            )
            archetype_nonzero_share = (
                float(X_action.loc[:, list(archetype_cols)].ne(0.0).mean().mean())
                if archetype_cols
                else 0.0
            )
            rows.append(
                {
                    "strategy": strategy,
                    "action": action,
                    "requested_backend": str(config.backend),
                    "effective_backend": effective_backend,
                    "optuna_objective": str(config.optuna_objective),
                    "target_mean": float(y.mean()),
                    "prediction_std": float(np.nanstd(pred)),
                    "nonzero_coefficients": int(np.sum(np.abs(coef) > 1e-12)),
                    "optuna_best_value": float(model.get("optuna_best_value", np.nan))
                    if isinstance(model, dict)
                    else np.nan,
                    "optuna_refit_objective_loss": float(model.get("optuna_refit_objective_loss", np.nan))
                    if isinstance(model, dict)
                    else np.nan,
                    "optuna_refit_mse": float(model.get("optuna_refit_mse", np.nan))
                    if isinstance(model, dict)
                    else np.nan,
                    "optuna_portfolio_ev": float(model.get("portfolio_ev", np.nan))
                    if isinstance(model, dict)
                    else np.nan,
                    "optuna_portfolio_hit_rate": float(model.get("portfolio_hit_rate", np.nan))
                    if isinstance(model, dict)
                    else np.nan,
                    "optuna_portfolio_loss_rate": float(model.get("portfolio_loss_rate", np.nan))
                    if isinstance(model, dict)
                    else np.nan,
                    "optuna_portfolio_downside_mean": float(model.get("portfolio_downside_mean", np.nan))
                    if isinstance(model, dict)
                    else np.nan,
                    "optuna_portfolio_max_loss_streak_hours": float(
                        model.get("portfolio_max_loss_streak_hours", np.nan)
                    )
                    if isinstance(model, dict)
                    else np.nan,
                    "optuna_portfolio_loss_streak_excess": float(
                        model.get("portfolio_loss_streak_excess", np.nan)
                    )
                    if isinstance(model, dict)
                    else np.nan,
                    "optuna_portfolio_turnover_mean": float(
                        model.get("portfolio_turnover_mean", np.nan)
                    )
                    if isinstance(model, dict)
                    else np.nan,
                    "optuna_portfolio_cash_share": float(model.get("portfolio_cash_share", np.nan))
                    if isinstance(model, dict)
                    else np.nan,
                    "optuna_portfolio_cash_share_excess": float(
                        model.get("portfolio_cash_share_excess", np.nan)
                    )
                    if isinstance(model, dict)
                    else np.nan,
                    "optuna_portfolio_cash_share_penalty": float(
                        model.get("portfolio_cash_share_penalty", np.nan)
                    )
                    if isinstance(model, dict)
                    else np.nan,
                    "optuna_portfolio_current_strategy_target_active_share": float(
                        model.get("portfolio_current_strategy_target_active_share", np.nan)
                    )
                    if isinstance(model, dict)
                    else np.nan,
                    "optuna_portfolio_unjustified_deactivation_share": float(
                        model.get("portfolio_unjustified_deactivation_share", np.nan)
                    )
                    if isinstance(model, dict)
                    else np.nan,
                    "optuna_portfolio_mean_active_strategies": float(
                        model.get("portfolio_mean_active_strategies", np.nan)
                    )
                    if isinstance(model, dict)
                    else np.nan,
                    "optuna_portfolio_active_utility_lcb": float(
                        model.get("portfolio_active_utility_lcb", np.nan)
                    )
                    if isinstance(model, dict)
                    else np.nan,
                    "optuna_portfolio_active_utility_lcb_shortfall": float(
                        model.get("portfolio_active_utility_lcb_shortfall", np.nan)
                    )
                    if isinstance(model, dict)
                    else np.nan,
                    "optuna_portfolio_loss_density_mean": float(
                        model.get("portfolio_loss_density_mean", np.nan)
                    )
                    if isinstance(model, dict)
                    else np.nan,
                    "optuna_portfolio_loss_density_excess_mean": float(
                        model.get("portfolio_loss_density_excess_mean", np.nan)
                    )
                    if isinstance(model, dict)
                    else np.nan,
                    "archetype_score_threshold": float(threshold),
                    "archetype_score_ramp_power": float(ramp_power),
                    "archetype_score_ramp_gain": float(ramp_gain),
                    "archetype_base_p_active_floor": base_p_active_floor,
                    "archetype_effective_p_active_threshold": effective_p_active_threshold,
                    "optuna_archetype_nonzero_penalty": float(config.optuna_archetype_nonzero_penalty),
                    "optuna_cash_share_target": float(config.optuna_cash_share_target),
                    "optuna_cash_share_weight": float(config.optuna_cash_share_weight),
                    "optuna_cash_share_excess_power": float(config.optuna_cash_share_excess_power),
                    "optuna_unjustified_deactivation_weight": float(
                        config.optuna_unjustified_deactivation_weight
                    ),
                    "optuna_unjustified_deactivation_gate_margin": float(
                        config.optuna_unjustified_deactivation_gate_margin
                    ),
                    "optuna_active_utility_lcb_weight": float(config.optuna_active_utility_lcb_weight),
                    "optuna_active_utility_lcb_z": float(config.optuna_active_utility_lcb_z),
                    "optuna_loss_density_weight": float(config.optuna_loss_density_weight),
                    "optuna_loss_density_window_hours": float(config.optuna_loss_density_window_hours),
                    "optuna_loss_density_target": float(config.optuna_loss_density_target),
                    "archetype_feature_count": int(len(archetype_cols)),
                    "archetype_feature_nonzero_share": archetype_nonzero_share,
                }
            )
    return PortfolioCalibrator(
        strategies=tuple(str(s) for s in strategies),
        feature_columns=features,
        config=config,
        coefficients=coefficients,
        intercepts=intercepts,
        models=models,
        effective_backend=",".join(sorted(effective_backends)) if effective_backends else str(config.backend),
        diagnostics=pd.DataFrame(rows),
    )


def score_frozen_portfolio_calibrator(
    X: pd.DataFrame,
    calibrator: PortfolioCalibrator,
) -> pd.DataFrame:
    X_score = (
        X.reindex(columns=calibrator.feature_columns)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32, copy=False)
    )
    cols: dict[str, np.ndarray] = {}
    cfg = calibrator.config
    clips = {
        "threshold_delta": cfg.threshold_delta_clip,
        "rank_delta": cfg.rank_delta_clip,
        "weight_log_delta": cfg.weight_log_delta_clip,
        "activation_gate": cfg.activation_gate_clip,
    }
    for strategy in calibrator.strategies:
        for action in ["threshold_delta", "rank_delta", "weight_log_delta", "activation_gate"]:
            coef = calibrator.coefficients.get(strategy, {}).get(action)
            intercept = calibrator.intercepts.get(strategy, {}).get(action, 0.0 if action != "activation_gate" else 1.0)
            model = calibrator.models.get(strategy, {}).get(action)
            X_action, _archetype_cols, _threshold, _ramp_power, _ramp_gain = _action_feature_frame(X_score, cfg, model)
            action_arr = X_action.to_numpy(dtype=np.float32, copy=False)
            if model is not None and not isinstance(model, dict):
                try:
                    values = np.asarray(model.predict(X_action), dtype=float)
                except Exception:
                    values = intercept + action_arr @ np.asarray(coef, dtype=np.float32) if coef is not None else np.full(len(X_score), intercept, dtype=np.float32)
            elif coef is None:
                values = np.full(len(X_score), intercept, dtype=np.float32)
            else:
                values = intercept + action_arr @ np.asarray(coef, dtype=np.float32)
            lo, hi = clips[action]
            cols[f"{strategy}__{action}"] = np.clip(values, float(lo), float(hi))
    return pd.DataFrame(cols, index=X_score.index)


def apply_portfolio_actions(
    strategy_scores: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    base_thresholds: Mapping[str, float],
    base_weights: Mapping[str, float],
    strategy_ranks: pd.DataFrame | None = None,
    base_rank_thresholds: Mapping[str, float] | None = None,
    activation_cutoffs: Mapping[str, float] | None = None,
    allow_cash: bool = True,
    renormalize: bool = True,
) -> pd.DataFrame:
    """Apply threshold/rank/weight/gate actions without all-active clamping."""

    strategies = [str(c) for c in strategy_scores.columns]
    final = pd.DataFrame(np.float32(0.0), index=strategy_scores.index, columns=strategies)
    for strategy in strategies:
        score = pd.to_numeric(strategy_scores[strategy], errors="coerce").fillna(-np.inf)
        threshold_delta = pd.to_numeric(
            actions.get(f"{strategy}__threshold_delta", pd.Series(0.0, index=final.index)),
            errors="coerce",
        ).reindex(final.index).fillna(0.0)
        weight_log_delta = pd.to_numeric(
            actions.get(f"{strategy}__weight_log_delta", pd.Series(0.0, index=final.index)),
            errors="coerce",
        ).reindex(final.index).fillna(0.0)
        rank_delta = pd.to_numeric(
            actions.get(f"{strategy}__rank_delta", pd.Series(0.0, index=final.index)),
            errors="coerce",
        ).reindex(final.index).fillna(0.0)
        gate = pd.to_numeric(
            actions.get(f"{strategy}__activation_gate", pd.Series(1.0, index=final.index)),
            errors="coerce",
        ).reindex(final.index).fillna(1.0)
        threshold = float(base_thresholds.get(strategy, 0.0)) + threshold_delta
        if strategy_ranks is not None and strategy in strategy_ranks.columns:
            rank_score = pd.to_numeric(strategy_ranks[strategy], errors="coerce").reindex(final.index).fillna(-np.inf)
            rank_threshold = float((base_rank_thresholds or {}).get(strategy, -np.inf))
            rank_active = rank_score + rank_delta >= rank_threshold
        else:
            rank_active = pd.Series(True, index=final.index)
        cutoff = float((activation_cutoffs or {}).get(strategy, 0.0))
        raw_weight = float(base_weights.get(strategy, 1.0)) * np.exp(np.clip(weight_log_delta, -8.0, 4.0))
        active = (score >= threshold) & rank_active & (gate >= cutoff)
        final[strategy] = np.where(active, raw_weight, 0.0)
    if renormalize:
        final = normalize_nonzero(final)
    if not allow_cash:
        empty = final.sum(axis=1).abs() <= 1e-12
        if bool(empty.any()):
            # Explicit fallback only when the policy forbids cash/no-trade.
            best = strategy_scores.loc[empty].idxmax(axis=1)
            for idx, strategy in best.items():
                final.loc[idx, strategy] = 1.0
    return final


def score_frozen_actions_to_weights(
    X: pd.DataFrame,
    strategy_scores: pd.DataFrame,
    calibrator: PortfolioCalibrator,
    *,
    base_thresholds: Mapping[str, float],
    base_weights: Mapping[str, float],
) -> pd.DataFrame:
    actions = score_frozen_portfolio_calibrator(X, calibrator)
    return apply_portfolio_actions(
        strategy_scores,
        actions,
        base_thresholds=base_thresholds,
        base_weights=base_weights,
        activation_cutoffs={s: calibrator.config.activation_cutoff for s in calibrator.strategies},
        allow_cash=bool(calibrator.config.allow_cash),
        renormalize=bool(calibrator.config.renormalize_nonzero),
    )
