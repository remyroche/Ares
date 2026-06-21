"""Training/inference-parity regime context features.

These helpers do not fit regimes. They consume row-level regime outputs from a
pooled regime model and deterministically create:

* per-asset regime features for base models,
* cross-sectional residual/z-score regime features,
* global market-regime aggregate features for meta models.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from extreme_price_movements.unsupervised_regime_learning.regime_models import (
    AdvancedRegimeLearningArtifact,
)


@dataclass(frozen=True)
class RegimeContextFeatureConfig:
    timestamp_col: str = "timestamp"
    symbol_col: str = "symbol"
    per_asset_prefix: str = "url_asset__"
    residual_prefix: str = "url_xs_z__"
    market_prefix: str = "url_market__"
    max_residual_features: int = 128
    max_market_probability_features: int = 128
    max_market_context_features: int = 96
    include_latent_context_composites: bool = True
    include_context_portfolios: bool = True
    interaction_prefix: str = "url_sigreg__"
    max_signal_interaction_signal_features: int = 24
    max_signal_interaction_regime_features: int = 64
    max_signal_regime_interaction_features: int = 256
    min_cross_section_assets: int = 2
    allow_positional_row_alignment: bool = True
    eps: float = 1e-8


def _safe_feature_name(name: str) -> str:
    return (
        str(name)
        .replace("/", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "_")
    )


def _numeric_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    cols = [str(col) for col in dict.fromkeys(columns) if str(col) in frame.columns]
    if not cols:
        return pd.DataFrame(index=frame.index)
    out = frame.reindex(columns=cols).apply(pd.to_numeric, errors="coerce")
    return out.astype(np.float32, copy=False)


_STATE_PROBABILITY_RE = re.compile(r"(?:^|_)regime_prob_\d+$")
_TRANSITION_CONTEXT_RE = re.compile(
    r"^(?:url_)?(?P<method>.+?)_"
    r"(?P<metric>regime_prob_entropy|regime_prob_max|regime_prob_change_\d+h|"
    r"regime_transition_hazard|time_since_regime_change|expected_regime_duration)$"
)
_STATE_PROBABILITY_METHOD_RE = re.compile(r"^(?P<method>.+?)_regime_prob_\d+$")


def _is_state_probability_column(name: str) -> bool:
    """Return True only for per-state probability columns.

    Transition/context columns such as ``*_regime_prob_entropy`` and
    ``*_regime_prob_change_1h`` are regime outputs, but they are not probability
    simplex coordinates. Treating them as states corrupts market entropy and
    concentration aggregates.
    """

    return bool(_STATE_PROBABILITY_RE.search(str(name)))


def _clip01(values: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32, copy=False)


def _row_mean(arrays: Sequence[np.ndarray], n_rows: int) -> np.ndarray:
    valid = [np.asarray(values, dtype=np.float32).reshape(-1) for values in arrays if len(values) == n_rows]
    if not valid:
        return np.zeros(n_rows, dtype=np.float32)
    return np.nan_to_num(np.vstack(valid).mean(axis=0), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _row_max(arrays: Sequence[np.ndarray], n_rows: int) -> np.ndarray:
    valid = [np.asarray(values, dtype=np.float32).reshape(-1) for values in arrays if len(values) == n_rows]
    if not valid:
        return np.zeros(n_rows, dtype=np.float32)
    return np.nan_to_num(np.vstack(valid).max(axis=0), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _row_std(arrays: Sequence[np.ndarray], n_rows: int) -> np.ndarray:
    valid = [np.asarray(values, dtype=np.float32).reshape(-1) for values in arrays if len(values) == n_rows]
    if len(valid) < 2:
        return np.zeros(n_rows, dtype=np.float32)
    return np.nan_to_num(np.vstack(valid).std(axis=0), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _method_context_columns(columns: Sequence[str]) -> dict[str, dict[str, list[str]]]:
    by_method: dict[str, dict[str, list[str]]] = {}
    for col in columns:
        name = str(col)
        match = _TRANSITION_CONTEXT_RE.match(name)
        if match:
            method = str(match.group("method"))
            metric = str(match.group("metric"))
            by_method.setdefault(method, {}).setdefault(metric, []).append(name)
            continue
        prob_match = _STATE_PROBABILITY_METHOD_RE.match(name)
        if prob_match and _is_state_probability_column(name):
            method = str(prob_match.group("method"))
            by_method.setdefault(method, {}).setdefault("state_prob", []).append(name)
    return by_method


def _latent_regime_context_composites(
    frame: pd.DataFrame,
    regime_outputs: pd.DataFrame,
    *,
    config: RegimeContextFeatureConfig,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Build latent-context composites for defensive and conditional gating.

    These features deliberately avoid return labels and direct alpha. They
    summarize regime certainty, transition pressure, maturity, and cross-method
    disagreement so downstream models can learn when latent context is stable
    enough to trust signals and when regime churn argues for reduced exposure.
    """

    if regime_outputs.empty:
        return pd.DataFrame(index=frame.index), {}
    numeric = _numeric_frame(regime_outputs, list(regime_outputs.columns)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if numeric.empty:
        return pd.DataFrame(index=frame.index), {}
    n = len(numeric)
    by_method = _method_context_columns(list(numeric.columns))
    asset_data: dict[str, np.ndarray] = {}
    uncertainty_parts: list[np.ndarray] = []
    confidence_parts: list[np.ndarray] = []
    transition_parts: list[np.ndarray] = []
    maturity_parts: list[np.ndarray] = []
    defensive_parts: list[np.ndarray] = []
    stability_parts: list[np.ndarray] = []
    portfolio_budget_parts: list[np.ndarray] = []
    portfolio_cut_parts: list[np.ndarray] = []
    portfolio_quality_parts: list[np.ndarray] = []

    for method, metrics in sorted(by_method.items()):
        entropy = None
        if metrics.get("regime_prob_entropy"):
            entropy = _clip01(_row_mean([numeric[col].to_numpy(dtype=np.float32, copy=False) for col in metrics["regime_prob_entropy"]], n))
        max_prob = None
        if metrics.get("regime_prob_max"):
            max_prob = _clip01(_row_mean([numeric[col].to_numpy(dtype=np.float32, copy=False) for col in metrics["regime_prob_max"]], n))
        state_cols = [col for col in metrics.get("state_prob", []) if col in numeric.columns]
        if (entropy is None or max_prob is None) and state_cols:
            probs = np.nan_to_num(numeric[state_cols].to_numpy(dtype=np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
            row_sum = np.sum(probs, axis=1, keepdims=True)
            norm = np.zeros_like(probs, dtype=np.float32)
            np.divide(
                probs,
                np.maximum(row_sum, float(config.eps)),
                out=norm,
                where=row_sum > 0.0,
            )
            if entropy is None:
                ent = -np.sum(norm * np.log(np.maximum(norm, 1e-12)), axis=1)
                entropy = _clip01(ent / np.log(float(max(2, norm.shape[1]))))
            if max_prob is None:
                max_prob = _clip01(np.max(norm, axis=1))
        if entropy is None:
            entropy = np.zeros(n, dtype=np.float32)
        if max_prob is None:
            max_prob = np.ones(n, dtype=np.float32)

        hazard_cols = [col for col in metrics.get("regime_transition_hazard", []) if col in numeric.columns]
        hazard = _clip01(_row_mean([numeric[col].to_numpy(dtype=np.float32, copy=False) for col in hazard_cols], n))
        change_cols = [
            col
            for metric, cols in metrics.items()
            if str(metric).startswith("regime_prob_change_")
            for col in cols
            if col in numeric.columns
        ]
        change = _clip01(_row_mean([np.abs(numeric[col].to_numpy(dtype=np.float32, copy=False)) for col in change_cols], n))
        time_cols = [col for col in metrics.get("time_since_regime_change", []) if col in numeric.columns]
        duration_cols = [col for col in metrics.get("expected_regime_duration", []) if col in numeric.columns]
        time_since = _row_mean([numeric[col].to_numpy(dtype=np.float32, copy=False) for col in time_cols], n)
        expected_duration = _row_mean([numeric[col].to_numpy(dtype=np.float32, copy=False) for col in duration_cols], n)
        maturity = _clip01(time_since / np.maximum(time_since + expected_duration + 1.0, float(config.eps)))

        uncertainty = _clip01(0.50 * entropy + 0.50 * (1.0 - max_prob))
        transition = _clip01(0.60 * hazard + 0.40 * change)
        stability = _clip01(0.40 * (1.0 - uncertainty) + 0.35 * (1.0 - transition) + 0.25 * maturity)
        defensive = _clip01(0.45 * uncertainty + 0.35 * transition + 0.20 * (1.0 - maturity))
        confidence = _clip01(0.35 * max_prob + 0.25 * (1.0 - entropy) + 0.25 * (1.0 - transition) + 0.15 * maturity)
        risk_budget = _clip01((0.50 * confidence + 0.35 * stability + 0.15 * maturity) * (1.0 - defensive))
        risk_cut = _clip01(1.0 - risk_budget)
        confidence_stability = _clip01(np.sqrt(np.maximum(confidence * stability, 0.0)) * (1.0 - 0.50 * defensive))

        safe_method = _safe_feature_name(method)
        asset_data[f"{config.per_asset_prefix}latent_{safe_method}_uncertainty"] = uncertainty
        asset_data[f"{config.per_asset_prefix}latent_{safe_method}_transition_pressure"] = transition
        asset_data[f"{config.per_asset_prefix}latent_{safe_method}_maturity"] = maturity
        asset_data[f"{config.per_asset_prefix}latent_{safe_method}_conditional_confidence"] = confidence
        if bool(config.include_context_portfolios):
            asset_data[f"{config.per_asset_prefix}ctx_portfolio_{safe_method}_risk_budget"] = risk_budget
            asset_data[f"{config.per_asset_prefix}ctx_portfolio_{safe_method}_risk_cut"] = risk_cut
            asset_data[f"{config.per_asset_prefix}ctx_portfolio_{safe_method}_confidence_stability"] = confidence_stability
        uncertainty_parts.append(uncertainty)
        confidence_parts.append(confidence)
        transition_parts.append(transition)
        maturity_parts.append(maturity)
        defensive_parts.append(defensive)
        stability_parts.append(stability)
        portfolio_budget_parts.append(risk_budget)
        portfolio_cut_parts.append(risk_cut)
        portfolio_quality_parts.append(confidence_stability)

    if not asset_data:
        return pd.DataFrame(index=frame.index), {}

    uncertainty_mean = _row_mean(uncertainty_parts, n)
    uncertainty_max = _row_max(uncertainty_parts, n)
    transition_mean = _row_mean(transition_parts, n)
    transition_max = _row_max(transition_parts, n)
    maturity_mean = _row_mean(maturity_parts, n)
    confidence_mean = _row_mean(confidence_parts, n)
    confidence_disagreement = _clip01(_row_std(confidence_parts + uncertainty_parts, n))
    defensive_score = _clip01(0.45 * _row_mean(defensive_parts, n) + 0.25 * uncertainty_max + 0.20 * transition_max + 0.10 * confidence_disagreement)
    stability_score = _clip01(_row_mean(stability_parts, n) * (1.0 - 0.50 * confidence_disagreement))
    conditional_score = _clip01(confidence_mean * (1.0 - 0.60 * defensive_score) * (1.0 - 0.35 * confidence_disagreement))
    portfolio_budget_mean = _row_mean(portfolio_budget_parts, n)
    portfolio_budget_min = (
        _clip01(1.0 - _row_max(portfolio_cut_parts, n))
        if portfolio_cut_parts
        else np.zeros(n, dtype=np.float32)
    )
    portfolio_cut_max = _row_max(portfolio_cut_parts, n)
    portfolio_quality_mean = _row_mean(portfolio_quality_parts, n)
    portfolio_disagreement = _clip01(_row_std(portfolio_budget_parts + portfolio_cut_parts, n))

    asset_data[f"{config.per_asset_prefix}latent_uncertainty_mean"] = uncertainty_mean
    asset_data[f"{config.per_asset_prefix}latent_uncertainty_max"] = uncertainty_max
    asset_data[f"{config.per_asset_prefix}latent_transition_pressure_mean"] = transition_mean
    asset_data[f"{config.per_asset_prefix}latent_transition_pressure_max"] = transition_max
    asset_data[f"{config.per_asset_prefix}latent_regime_maturity_mean"] = maturity_mean
    asset_data[f"{config.per_asset_prefix}latent_method_disagreement_score"] = confidence_disagreement
    asset_data[f"{config.per_asset_prefix}latent_context_stability_score"] = stability_score
    asset_data[f"{config.per_asset_prefix}latent_defensive_no_trade_score"] = defensive_score
    asset_data[f"{config.per_asset_prefix}latent_conditional_confidence_score"] = conditional_score
    if bool(config.include_context_portfolios):
        asset_data[f"{config.per_asset_prefix}ctx_portfolio_risk_budget_mean"] = portfolio_budget_mean
        asset_data[f"{config.per_asset_prefix}ctx_portfolio_risk_budget_min"] = portfolio_budget_min
        asset_data[f"{config.per_asset_prefix}ctx_portfolio_risk_cut_max"] = portfolio_cut_max
        asset_data[f"{config.per_asset_prefix}ctx_portfolio_confidence_stability_mean"] = portfolio_quality_mean
        asset_data[f"{config.per_asset_prefix}ctx_portfolio_family_disagreement"] = portfolio_disagreement

    asset_frame = pd.DataFrame(asset_data, index=frame.index).astype(np.float32, copy=False)
    parts = [asset_frame]
    groups: dict[str, list[str]] = {}
    latent_asset_cols = [col for col in asset_frame.columns if "latent_" in str(col)]
    portfolio_asset_cols = [col for col in asset_frame.columns if "ctx_portfolio_" in str(col)]
    if latent_asset_cols:
        groups["latent_asset_context"] = latent_asset_cols
    if portfolio_asset_cols:
        groups["context_portfolio_asset"] = portfolio_asset_cols
    if str(config.timestamp_col) in frame.columns:
        ts = pd.to_datetime(frame[str(config.timestamp_col)], utc=True, errors="coerce")
        group_key = pd.Series(ts.to_numpy(), index=frame.index)
        asset = parts[0]
        focus_cols = [
            f"{config.per_asset_prefix}latent_uncertainty_mean",
            f"{config.per_asset_prefix}latent_transition_pressure_mean",
            f"{config.per_asset_prefix}latent_method_disagreement_score",
            f"{config.per_asset_prefix}latent_context_stability_score",
            f"{config.per_asset_prefix}latent_defensive_no_trade_score",
            f"{config.per_asset_prefix}latent_conditional_confidence_score",
            f"{config.per_asset_prefix}ctx_portfolio_risk_budget_mean",
            f"{config.per_asset_prefix}ctx_portfolio_risk_budget_min",
            f"{config.per_asset_prefix}ctx_portfolio_risk_cut_max",
            f"{config.per_asset_prefix}ctx_portfolio_confidence_stability_mean",
            f"{config.per_asset_prefix}ctx_portfolio_family_disagreement",
        ]
        focus = asset.reindex(columns=[col for col in focus_cols if col in asset.columns])
        if not focus.empty:
            grouped = focus.groupby(group_key, sort=False)
            mean = grouped.transform("mean").fillna(0.0)
            std = grouped.transform("std").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            market_data: dict[str, np.ndarray] = {}
            for col in focus.columns:
                base = str(col).replace(str(config.per_asset_prefix), "")
                market_data[f"{config.market_prefix}mean__{base}"] = mean[col].to_numpy(dtype=np.float32, copy=False)
                market_data[f"{config.market_prefix}dispersion__{base}"] = std[col].to_numpy(dtype=np.float32, copy=False)
            market_defensive = _clip01(
                0.45 * mean[f"{config.per_asset_prefix}latent_defensive_no_trade_score"].to_numpy(dtype=np.float32, copy=False)
                + 0.25 * mean[f"{config.per_asset_prefix}latent_transition_pressure_mean"].to_numpy(dtype=np.float32, copy=False)
                + 0.20 * mean[f"{config.per_asset_prefix}latent_uncertainty_mean"].to_numpy(dtype=np.float32, copy=False)
                + 0.10 * std[f"{config.per_asset_prefix}latent_conditional_confidence_score"].to_numpy(dtype=np.float32, copy=False)
            )
            market_conditional = _clip01(
                mean[f"{config.per_asset_prefix}latent_conditional_confidence_score"].to_numpy(dtype=np.float32, copy=False)
                * (1.0 - 0.50 * market_defensive)
                * (
                    1.0
                    - 0.25
                    * _clip01(std[f"{config.per_asset_prefix}latent_method_disagreement_score"].to_numpy(dtype=np.float32, copy=False))
                )
            )
            market_data[f"{config.market_prefix}latent_defensive_no_trade_score"] = market_defensive
            market_data[f"{config.market_prefix}latent_conditional_confidence_score"] = market_conditional
            market_frame = pd.DataFrame(market_data, index=frame.index).astype(np.float32, copy=False)
            parts.append(market_frame)
            latent_market_cols = [col for col in market_frame.columns if "latent_" in str(col)]
            portfolio_market_cols = [col for col in market_frame.columns if "ctx_portfolio_" in str(col)]
            if latent_market_cols:
                groups["latent_market_context"] = latent_market_cols
            if portfolio_market_cols:
                groups["context_portfolio_market"] = portfolio_market_cols

            residual_focus = focus.copy()
            grouped_focus = residual_focus.groupby(group_key, sort=False)
            z_mean = grouped_focus.transform("mean")
            z_std = grouped_focus.transform("std").replace(0.0, np.nan)
            count = grouped_focus.transform("count")
            z = (residual_focus - z_mean) / z_std
            valid_ts = ts.notna().to_numpy(dtype=bool)
            valid = count.ge(int(config.min_cross_section_assets))
            valid.iloc[~valid_ts, :] = False
            z = z.where(valid, 0.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
            z.columns = [
                f"{config.residual_prefix}{str(col).replace(str(config.per_asset_prefix), '')}"
                for col in z.columns
            ]
            z = z.astype(np.float32, copy=False)
            parts.append(z)
            latent_residual_cols = [col for col in z.columns if "latent_" in str(col)]
            portfolio_residual_cols = [col for col in z.columns if "ctx_portfolio_" in str(col)]
            if latent_residual_cols:
                groups["latent_cross_sectional_context"] = latent_residual_cols
            if portfolio_residual_cols:
                groups["context_portfolio_cross_sectional"] = portfolio_residual_cols

    out = pd.concat(parts, axis=1).loc[:, lambda x: ~x.columns.duplicated()].astype(np.float32, copy=False)
    return out, groups


def _aligned_regime_outputs(
    frame: pd.DataFrame,
    regime_outputs: pd.DataFrame,
    *,
    config: RegimeContextFeatureConfig,
) -> pd.DataFrame:
    if not isinstance(regime_outputs, pd.DataFrame) or regime_outputs.empty:
        return pd.DataFrame(index=frame.index)
    if len(regime_outputs) != len(frame):
        raise ValueError(
            "regime_outputs must have the same row count as frame. "
            f"Got frame_rows={len(frame)} and regime_output_rows={len(regime_outputs)}. "
            "Generate row-level regime outputs for this frame before building context features."
        )
    if regime_outputs.index.equals(frame.index):
        return regime_outputs
    if not bool(config.allow_positional_row_alignment):
        raise ValueError(
            "regime_outputs index does not match frame index and positional alignment is disabled."
        )
    out = regime_outputs.copy(deep=False)
    out.index = frame.index
    return out


def _rank_columns_by_variance(values: pd.DataFrame, max_cols: int) -> list[str]:
    cols = [str(col) for col in values.columns]
    cap = int(max_cols or 0)
    if cap <= 0 or len(cols) <= cap:
        return cols
    arr = values.to_numpy(dtype=np.float32, copy=False)
    var = np.nanvar(arr, axis=0)
    var = np.nan_to_num(var, nan=0.0, posinf=0.0, neginf=0.0)
    idx = np.argsort(var, kind="mergesort")[-cap:]
    idx = np.sort(idx)
    return [cols[int(i)] for i in idx]


def cross_sectional_regime_residuals(
    frame: pd.DataFrame,
    regime_outputs: pd.DataFrame,
    *,
    config: RegimeContextFeatureConfig = RegimeContextFeatureConfig(),
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute per-timestamp cross-sectional z-scores of regime outputs."""

    if str(config.timestamp_col) not in frame.columns:
        return pd.DataFrame(index=frame.index)
    regime_outputs = _aligned_regime_outputs(frame, regime_outputs, config=config)
    if regime_outputs.empty:
        return pd.DataFrame(index=frame.index)
    candidate = list(columns or regime_outputs.columns)
    numeric = _numeric_frame(regime_outputs, candidate)
    if numeric.empty:
        return pd.DataFrame(index=frame.index)
    selected = _rank_columns_by_variance(numeric, int(config.max_residual_features))
    numeric = numeric.reindex(columns=selected)
    ts = pd.to_datetime(frame[str(config.timestamp_col)], utc=True, errors="coerce")
    group_key = pd.Series(ts.to_numpy(), index=numeric.index)
    valid_ts = ts.notna().to_numpy(dtype=bool)
    grouped = numeric.groupby(group_key, sort=False)
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0.0, np.nan)
    count = grouped.transform("count")
    z = (numeric - mean) / std
    valid = count.ge(int(config.min_cross_section_assets))
    valid.iloc[~valid_ts, :] = False
    z = z.where(valid, 0.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    z.columns = [f"{config.residual_prefix}{_safe_feature_name(col)}" for col in z.columns]
    return z.astype(np.float32, copy=False)


def market_regime_aggregate_features(
    frame: pd.DataFrame,
    regime_outputs: pd.DataFrame,
    *,
    config: RegimeContextFeatureConfig = RegimeContextFeatureConfig(),
    probability_columns: Sequence[str] | None = None,
    label_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Aggregate per-asset regime outputs into market-wide regime context."""

    if str(config.timestamp_col) not in frame.columns:
        return pd.DataFrame(index=frame.index)
    regime_outputs = _aligned_regime_outputs(frame, regime_outputs, config=config)
    if regime_outputs.empty:
        return pd.DataFrame(index=frame.index)
    ts = pd.to_datetime(frame[str(config.timestamp_col)], utc=True, errors="coerce")
    group_key = pd.Series(ts.to_numpy(), index=frame.index)
    parts: list[pd.DataFrame] = []
    prob_cols = [
        str(col)
        for col in (
            probability_columns
            if probability_columns is not None
            else [c for c in regime_outputs.columns if _is_state_probability_column(str(c))]
        )
        if str(col) in regime_outputs.columns and _is_state_probability_column(str(col))
    ]
    if prob_cols:
        prob_values = _numeric_frame(regime_outputs, prob_cols)
        prob_cols = _rank_columns_by_variance(
            prob_values,
            int(config.max_market_probability_features),
        )
        prob_values = prob_values.reindex(columns=prob_cols)
        grouped = prob_values.groupby(group_key, sort=False)
        means = grouped.transform("mean")
        stds = grouped.transform("std").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        mean_cols = {col: f"{config.market_prefix}mean__{_safe_feature_name(col)}" for col in prob_cols}
        std_cols = {col: f"{config.market_prefix}dispersion__{_safe_feature_name(col)}" for col in prob_cols}
        parts.append(means.rename(columns=mean_cols).fillna(0.0).astype(np.float32, copy=False))
        parts.append(stds.rename(columns=std_cols).fillna(0.0).astype(np.float32, copy=False))
        row_sum = means.sum(axis=1).replace(0.0, np.nan)
        norm = means.div(row_sum, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        n_states = max(1, norm.shape[1])
        entropy = -np.sum(
            norm.to_numpy(dtype=np.float64) * np.log(np.maximum(norm.to_numpy(dtype=np.float64), 1e-12)),
            axis=1,
        )
        entropy = entropy / np.log(float(n_states)) if n_states > 1 else np.zeros(len(norm), dtype=np.float64)
        concentration = np.sum(norm.to_numpy(dtype=np.float64) ** 2, axis=1)
        parts.append(
            pd.DataFrame(
                {
                    f"{config.market_prefix}prob_entropy": np.nan_to_num(entropy, nan=0.0).astype(np.float32),
                    f"{config.market_prefix}prob_concentration": np.nan_to_num(concentration, nan=0.0).astype(np.float32),
                    f"{config.market_prefix}prob_max_share": norm.max(axis=1).fillna(0.0).to_numpy(dtype=np.float32),
                },
                index=frame.index,
            )
        )
    context_cols = [
        str(col)
        for col in regime_outputs.columns
        if str(col) not in set(prob_cols)
        and not str(col).endswith("_regime")
        and (
            "regime_prob_entropy" in str(col)
            or "regime_prob_max" in str(col)
            or "regime_prob_change" in str(col)
            or "transition_hazard" in str(col)
            or "time_since_regime_change" in str(col)
            or "expected_regime_duration" in str(col)
        )
    ]
    if context_cols:
        context_values = _numeric_frame(regime_outputs, context_cols)
        context_cols = _rank_columns_by_variance(
            context_values,
            int(config.max_market_context_features),
        )
        context_values = context_values.reindex(columns=context_cols)
        grouped_context = context_values.groupby(group_key, sort=False)
        context_mean = grouped_context.transform("mean")
        context_std = grouped_context.transform("std").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        mean_cols = {col: f"{config.market_prefix}mean__{_safe_feature_name(col)}" for col in context_cols}
        std_cols = {col: f"{config.market_prefix}dispersion__{_safe_feature_name(col)}" for col in context_cols}
        parts.append(context_mean.rename(columns=mean_cols).fillna(0.0).astype(np.float32, copy=False))
        parts.append(context_std.rename(columns=std_cols).fillna(0.0).astype(np.float32, copy=False))
    label_cols = [
        str(col)
        for col in (
            label_columns
            if label_columns is not None
            else [c for c in regime_outputs.columns if str(c).endswith("_regime")]
        )
        if str(col) in regime_outputs.columns
    ]
    for col in label_cols:
        labels = pd.to_numeric(regime_outputs[col], errors="coerce")
        if labels.notna().sum() == 0:
            continue
        safe = _safe_feature_name(col)
        states = sorted(int(v) for v in pd.unique(labels.dropna()) if int(v) >= 0)
        if not states:
            continue
        shares = []
        share_cols: dict[str, np.ndarray] = {}
        for state in states:
            indicator = labels.eq(state).astype(np.float32)
            share = indicator.groupby(group_key, sort=False).transform("mean").fillna(0.0)
            share_cols[f"{config.market_prefix}{safe}_share_state_{state}"] = share.to_numpy(dtype=np.float32)
            shares.append(share.to_numpy(dtype=np.float64, copy=False))
        share_arr = np.vstack(shares).T if shares else np.zeros((len(frame), 0), dtype=np.float64)
        if share_arr.size:
            entropy = -np.sum(share_arr * np.log(np.maximum(share_arr, 1e-12)), axis=1)
            entropy = entropy / np.log(float(share_arr.shape[1])) if share_arr.shape[1] > 1 else np.zeros(len(frame))
            concentration = np.sum(share_arr * share_arr, axis=1)
            share_cols[f"{config.market_prefix}{safe}_entropy"] = np.nan_to_num(entropy, nan=0.0).astype(np.float32)
            share_cols[f"{config.market_prefix}{safe}_concentration"] = np.nan_to_num(concentration, nan=0.0).astype(np.float32)
            share_cols[f"{config.market_prefix}{safe}_max_share"] = np.max(share_arr, axis=1).astype(np.float32)
        if share_cols:
            parts.append(pd.DataFrame(share_cols, index=frame.index))
    if not parts:
        return pd.DataFrame(index=frame.index)
    return pd.concat(parts, axis=1).astype(np.float32, copy=False)


def build_regime_context_feature_frame(
    frame: pd.DataFrame,
    regime_outputs: pd.DataFrame,
    *,
    config: RegimeContextFeatureConfig = RegimeContextFeatureConfig(),
    include_per_asset: bool = True,
    include_cross_sectional_residuals: bool = True,
    include_market_aggregates: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build base/meta regime context features from row-level regime outputs."""

    parts: list[pd.DataFrame] = []
    groups: dict[str, list[str]] = {}
    aligned_outputs = (
        _aligned_regime_outputs(frame, regime_outputs, config=config)
        if isinstance(regime_outputs, pd.DataFrame)
        else pd.DataFrame(index=frame.index)
    )
    numeric = _numeric_frame(aligned_outputs, list(aligned_outputs.columns)) if not aligned_outputs.empty else pd.DataFrame(index=frame.index)
    if include_per_asset and not numeric.empty:
        per_asset = numeric.add_prefix(str(config.per_asset_prefix)).astype(np.float32, copy=False)
        groups["per_asset_regime_outputs"] = list(per_asset.columns)
        parts.append(per_asset)
    if include_cross_sectional_residuals and not numeric.empty:
        residuals = cross_sectional_regime_residuals(frame, numeric, config=config)
        if not residuals.empty:
            groups["cross_sectional_regime_residuals"] = list(residuals.columns)
            parts.append(residuals)
    if include_market_aggregates and not aligned_outputs.empty:
        market = market_regime_aggregate_features(frame, aligned_outputs, config=config)
        if not market.empty:
            groups["market_regime_aggregates"] = list(market.columns)
            parts.append(market)
    if bool(config.include_latent_context_composites) and not aligned_outputs.empty:
        latent, latent_groups = _latent_regime_context_composites(
            frame,
            aligned_outputs,
            config=config,
        )
        if not latent.empty:
            for key, value in latent_groups.items():
                groups[key] = list(value)
            parts.append(latent)
    out = pd.concat(parts if parts else [pd.DataFrame(index=frame.index)], axis=1)
    if not out.empty:
        out = out.loc[:, ~out.columns.duplicated()].astype(np.float32, copy=False)
    diagnostics = {
        "input_rows": int(len(frame)),
        "input_regime_output_columns": int(aligned_outputs.shape[1]),
        "output_feature_count": int(out.shape[1]),
        "groups": {key: int(len(value)) for key, value in groups.items()},
        "feature_groups": groups,
        "row_alignment": (
            "index"
            if isinstance(regime_outputs, pd.DataFrame) and regime_outputs.index.equals(frame.index)
            else "positional"
            if isinstance(regime_outputs, pd.DataFrame) and len(regime_outputs) == len(frame)
            else "empty"
        ),
        "train_inference_parity_surface": "deterministic_row_level_regime_output_transform",
    }
    return out, diagnostics


def generate_signal_regime_interaction_features(
    frame: pd.DataFrame,
    signal_features: Sequence[str],
    regime_features: Sequence[str],
    *,
    config: RegimeContextFeatureConfig = RegimeContextFeatureConfig(),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate bounded signal x regime interaction candidates.

    The transform intentionally does not fit scalers or quantile thresholds. It
    multiplies existing signal columns by row-level regime/context columns so
    training and inference can recreate exactly the same values after the base
    signal and regime-output surfaces are available.
    """

    signal_cols = [
        str(col)
        for col in dict.fromkeys(signal_features)
        if str(col) in frame.columns
    ]
    regime_cols = [
        str(col)
        for col in dict.fromkeys(regime_features)
        if str(col) in frame.columns and str(col) not in set(signal_cols)
    ]
    if not signal_cols or not regime_cols:
        return pd.DataFrame(index=frame.index), {
            "status": "empty_input",
            "input_signal_features": int(len(signal_cols)),
            "input_regime_features": int(len(regime_cols)),
            "output_feature_count": 0,
        }

    signals = _numeric_frame(frame, signal_cols).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    regimes = _numeric_frame(frame, regime_cols).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    selected_signals = _rank_columns_by_variance(
        signals,
        int(config.max_signal_interaction_signal_features),
    )
    selected_regimes = _rank_columns_by_variance(
        regimes,
        int(config.max_signal_interaction_regime_features),
    )
    signals = signals.reindex(columns=selected_signals)
    regimes = regimes.reindex(columns=selected_regimes)
    if signals.empty or regimes.empty:
        return pd.DataFrame(index=frame.index), {
            "status": "no_numeric_inputs",
            "input_signal_features": int(len(signal_cols)),
            "input_regime_features": int(len(regime_cols)),
            "selected_signal_features": int(len(selected_signals)),
            "selected_regime_features": int(len(selected_regimes)),
            "output_feature_count": 0,
        }

    s_arr = np.nan_to_num(signals.to_numpy(dtype=np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    r_arr = np.nan_to_num(regimes.to_numpy(dtype=np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    pair_scores: list[tuple[float, int, int]] = []
    for i in range(s_arr.shape[1]):
        prod = s_arr[:, [i]] * r_arr
        var = np.nanvar(prod, axis=0)
        var = np.nan_to_num(var, nan=0.0, posinf=0.0, neginf=0.0)
        pair_scores.extend((float(score), int(i), int(j)) for j, score in enumerate(var))
    if not pair_scores:
        return pd.DataFrame(index=frame.index), {
            "status": "no_pairs",
            "selected_signal_features": int(len(selected_signals)),
            "selected_regime_features": int(len(selected_regimes)),
            "output_feature_count": 0,
        }

    cap = int(config.max_signal_regime_interaction_features or 0)
    if cap <= 0:
        cap = len(pair_scores)
    pair_scores.sort(key=lambda item: item[0], reverse=True)
    selected_pairs = pair_scores[: min(cap, len(pair_scores))]
    data: dict[str, np.ndarray] = {}
    used_names: set[str] = set()
    for _score, i, j in selected_pairs:
        signal = selected_signals[int(i)]
        regime = selected_regimes[int(j)]
        name = (
            f"{config.interaction_prefix}"
            f"{_safe_feature_name(signal)}__x__{_safe_feature_name(regime)}"
        )
        if name in used_names:
            continue
        used_names.add(name)
        values = s_arr[:, int(i)] * r_arr[:, int(j)]
        data[name] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    out = pd.DataFrame(data, index=frame.index).astype(np.float32, copy=False) if data else pd.DataFrame(index=frame.index)
    diagnostics = {
        "status": "completed",
        "input_signal_features": int(len(signal_cols)),
        "input_regime_features": int(len(regime_cols)),
        "selected_signal_features": int(len(selected_signals)),
        "selected_regime_features": int(len(selected_regimes)),
        "candidate_pair_count": int(len(pair_scores)),
        "output_feature_count": int(out.shape[1]),
        "feature_groups": {"signal_regime_interactions": list(out.columns)},
        "train_inference_parity_surface": "deterministic_signal_times_regime_output_transform",
    }
    return out, diagnostics


def regime_outputs_from_artifact(
    artifact: AdvancedRegimeLearningArtifact,
    *,
    include_model_features: bool = True,
    include_probabilities: bool = True,
    include_labels: bool = True,
    include_transitions: bool = True,
    only_model_or_top_methods: bool = True,
    max_top_methods: int = 5,
) -> pd.DataFrame:
    """Collect row-level regime outputs from an artifact for context features."""

    methods = _artifact_context_methods(
        artifact,
        only_model_or_top_methods=only_model_or_top_methods,
        max_top_methods=max_top_methods,
    )
    parts: list[pd.DataFrame] = []
    if include_model_features:
        part = getattr(artifact, "model_regime_features", pd.DataFrame())
        if isinstance(part, pd.DataFrame) and not part.empty:
            parts.append(part)
    if include_probabilities:
        part = getattr(artifact, "regime_probabilities", pd.DataFrame())
        if isinstance(part, pd.DataFrame) and not part.empty:
            cols = _filter_method_columns(
                part.columns,
                methods,
                probability=True,
            )
            parts.append(part.reindex(columns=cols) if cols else part.iloc[:, 0:0])
    if include_labels:
        part = getattr(artifact, "regime_labels", pd.DataFrame())
        if isinstance(part, pd.DataFrame) and not part.empty:
            cols = _filter_method_columns(part.columns, methods)
            labels = part.reindex(columns=cols).apply(lambda s: pd.to_numeric(s, errors="coerce"))
            parts.append(labels)
    if include_transitions:
        part = getattr(artifact, "regime_transition_features", pd.DataFrame())
        if isinstance(part, pd.DataFrame) and not part.empty:
            cols = _filter_method_columns(part.columns, methods, transition=True)
            parts.append(part.reindex(columns=cols) if cols else part.iloc[:, 0:0])
    out = pd.concat(parts if parts else [pd.DataFrame(index=getattr(artifact, "row_keys", pd.DataFrame()).index)], axis=1)
    return out.loc[:, ~out.columns.duplicated()] if not out.empty else out


def _artifact_context_methods(
    artifact: AdvancedRegimeLearningArtifact,
    *,
    only_model_or_top_methods: bool,
    max_top_methods: int,
) -> list[str]:
    if not bool(only_model_or_top_methods):
        return []
    diagnostics = getattr(artifact, "diagnostics", {}) or {}
    methods: list[str] = []
    if isinstance(diagnostics, Mapping):
        for key in ("kept_methods", "model_regime_methods"):
            value = diagnostics.get(key, [])
            if isinstance(value, (list, tuple, set)):
                methods.extend(str(item) for item in value if str(item))
    diag = getattr(artifact, "regime_diagnostics", pd.DataFrame())
    cap = max(1, int(max_top_methods or 1))
    if isinstance(diag, pd.DataFrame) and not diag.empty and "method" in diag.columns:
        ordered = diag
        score_col = "UsefulRegimeScore" if "UsefulRegimeScore" in ordered.columns else "TotalScore"
        if score_col in ordered.columns:
            ordered = ordered.sort_values(score_col, ascending=False, kind="mergesort")
        methods.extend(ordered["method"].astype(str).head(cap).tolist())
    out: list[str] = []
    seen: set[str] = set()
    for method in methods:
        key = str(method)
        if key and key not in seen:
            seen.add(key)
            out.append(key)
        if len(out) >= cap:
            break
    return out


def _filter_method_columns(
    columns: Sequence[Any],
    methods: Sequence[str],
    *,
    probability: bool = False,
    transition: bool = False,
) -> list[str]:
    method_list = [str(method) for method in methods if str(method)]
    if not method_list:
        return [str(col) for col in columns]
    out: list[str] = []
    for col in columns:
        name = str(col)
        for method in method_list:
            if transition:
                keep = name.startswith(f"url_{method}_")
            elif probability:
                keep = name.startswith(f"{method}_regime_prob_")
            else:
                keep = name.startswith(f"{method}_")
            if keep:
                out.append(name)
                break
    return out


def build_regime_context_features_from_artifact(
    frame: pd.DataFrame,
    artifact: AdvancedRegimeLearningArtifact,
    *,
    config: RegimeContextFeatureConfig = RegimeContextFeatureConfig(),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    outputs = regime_outputs_from_artifact(artifact)
    return build_regime_context_feature_frame(frame, outputs, config=config)
