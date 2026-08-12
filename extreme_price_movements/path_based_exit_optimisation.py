"""Strict-prequential path-based adaptive exit optimisation.

This module is the second-stage wrapper around the frozen winner produced by
``simple_policy_optimiser``.  It never changes candidate admission, entry,
position size, cost accounting, or the incumbent exit when the selected
action is the zero action.  Candidate actions may only tighten the stop or
alter the still-live trailing geometry.  The implementation deliberately
anchors every counterfactual to the incumbent replay on the identical path.

The research contract is intentionally conservative:

* five levels for each of stop, activation, trailing power, and giveback;
* all 625 joint actions are replayed in vectorised trade batches;
* structure discovery, binning, robust fitting, and calibration are train-only;
* chronological folds are purged by the complete outcome horizon;
* a trade, including every one of its decision states, belongs to one fold;
* the learned surface is factorised and shrunk rather than treating actions as
  625 unrelated classes;
* uncertainty and action distance must be cleared before intervention;
* deployment remains on the incumbent unless predeclared portability gates
  pass on strict OOF sequential replay.

The public functions are usable independently by tests and offline runners.
``run_after_global_policy_optimisation`` is the integration point used by the
simple policy optimiser.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


SCHEMA = "path_based_exit_optimisation_v1"
ACTION_COMPONENTS = ("stop", "activation", "power", "giveback")
BASELINE_LEVELS = (0, 2, 2, 2)
EPS = 1.0e-12


class AdaptiveExitContractError(RuntimeError):
    """Raised when a causality, identity, or replay contract is violated."""


@dataclass(frozen=True)
class AdaptiveExitConfig:
    """Portable adaptive-exit research and deployment contract."""

    action_levels: int = 5
    stop_min_mult: float = 0.50
    activation_min_ratio: float = 0.50
    activation_max_ratio: float = 1.50
    power_min_ratio: float = 0.50
    power_max_ratio: float = 1.50
    giveback_min_ratio: float = 0.50
    giveback_max_ratio: float = 1.50
    decision_every_bars: int = 1
    max_hourly_age_minutes: float = 90.0
    stale_hourly_mode: str = "path_only"
    n_bins: int = 5
    max_state_features: int = 18
    max_state_pairs: int = 8
    include_action_pair_synergies: bool = True
    cmi_permutations: int = 24
    cmi_min_folds: int = 2
    support_prior: float = 300.0
    ridge_prior: float = 25.0
    huber_delta_bps: float = 150.0
    robust_iterations: int = 5
    uncertainty_lambda: float = 0.75
    action_distance_lambda_bps: float = 2.0
    min_action_edge_bps: float = 5.0
    max_actions_per_trade: int = 8
    purge_hours: float = 12.0
    min_train_trades: int = 500
    min_validation_trades: int = 100
    max_counterfactual_trades: int = 20_000
    action_batch_size: int = 25
    trade_batch_size: int = 4_096
    random_seed: int = 20260812
    required_positive_month_fraction: float = 0.60
    required_worst_month_delta_bps: float = -10.0
    required_min_uplift_bps: float = 5.0
    required_relative_stability_ratio: float = -0.25

    def validate(self) -> None:
        if self.action_levels != 5:
            raise ValueError("the canonical adaptive-exit grid requires five levels")
        if self.stop_min_mult <= 0.0 or self.stop_min_mult > 1.0:
            raise ValueError("stop_min_mult must be in (0, 1]")
        for low, high, name in (
            (self.activation_min_ratio, self.activation_max_ratio, "activation"),
            (self.power_min_ratio, self.power_max_ratio, "power"),
            (self.giveback_min_ratio, self.giveback_max_ratio, "giveback"),
        ):
            if not 0.0 < low <= 1.0 <= high:
                raise ValueError(f"{name} ratios must straddle one")
        if self.stale_hourly_mode not in {"fail_closed", "path_only"}:
            raise ValueError("stale_hourly_mode must be fail_closed or path_only")
        if self.n_bins < 3:
            raise ValueError("n_bins must be at least three")
        if self.purge_hours < 0.0:
            raise ValueError("purge_hours cannot be negative")


@dataclass(frozen=True)
class ExitAction:
    """One point on the factorised five-by-five-by-five-by-five grid."""

    stop_level: int
    activation_level: int
    power_level: int
    giveback_level: int

    @property
    def levels(self) -> tuple[int, int, int, int]:
        return (
            self.stop_level,
            self.activation_level,
            self.power_level,
            self.giveback_level,
        )

    @property
    def action_id(self) -> str:
        return "s%d_a%d_p%d_g%d" % self.levels

    @property
    def is_baseline(self) -> bool:
        return self.levels == BASELINE_LEVELS

    def normalized_distance(self) -> float:
        baseline = np.asarray(BASELINE_LEVELS, dtype=float)
        levels = np.asarray(self.levels, dtype=float)
        return float(np.mean(np.abs(levels - baseline) / 2.0))

    def intensity(self) -> str:
        distance = self.normalized_distance()
        if distance <= 0.25:
            return "mild"
        if distance <= 0.625:
            return "medium"
        return "aggressive"


def build_action_grid(config: AdaptiveExitConfig | None = None) -> tuple[ExitAction, ...]:
    """Return the deterministic 625-action grid with the baseline first."""

    cfg = config or AdaptiveExitConfig()
    cfg.validate()
    actions = [
        ExitAction(stop, activation, power, giveback)
        for stop in range(cfg.action_levels)
        for activation in range(cfg.action_levels)
        for power in range(cfg.action_levels)
        for giveback in range(cfg.action_levels)
    ]
    actions.sort(key=lambda action: (not action.is_baseline, action.levels))
    if len(actions) != 625 or sum(action.is_baseline for action in actions) != 1:
        raise AssertionError("adaptive exit action-grid identity failure")
    return tuple(actions)


def _piecewise_ratio(level: int, low: float, high: float) -> float:
    if level not in range(5):
        raise ValueError("action level must be in [0, 4]")
    if level <= 2:
        return float(low + (1.0 - low) * (level / 2.0))
    return float(1.0 + (high - 1.0) * ((level - 2.0) / 2.0))


def materialize_action_params(
    baseline: Mapping[str, Any],
    action: ExitAction,
    config: AdaptiveExitConfig | None = None,
) -> dict[str, Any]:
    """Map a discrete action to incumbent parameters without hidden changes."""

    cfg = config or AdaptiveExitConfig()
    cfg.validate()
    result = dict(baseline)
    baseline_sl = float(baseline["sl_mult"])
    stop_intensity = float(action.stop_level) / 4.0
    result["sl_mult"] = baseline_sl * (
        1.0 - stop_intensity * (1.0 - cfg.stop_min_mult)
    )
    result["trailing_activation_mult"] = float(
        baseline["trailing_activation_mult"]
    ) * _piecewise_ratio(
        action.activation_level,
        cfg.activation_min_ratio,
        cfg.activation_max_ratio,
    )
    baseline_power = float(baseline.get("trailing_power", 1.5))
    result["trailing_power"] = baseline_power * _piecewise_ratio(
        action.power_level,
        cfg.power_min_ratio,
        cfg.power_max_ratio,
    )
    giveback_ratio = _piecewise_ratio(
        action.giveback_level, cfg.giveback_min_ratio, cfg.giveback_max_ratio
    )
    fixed_gap = float(baseline.get("fixed_trailing_gap_mult", 0.0) or 0.0)
    if fixed_gap > 0.0:
        # The incumbent 15m contract uses a fixed trailing gap.  Preserve its
        # exact zero action and map the giveback component to that gap.  Power
        # is behaviourally inactive and therefore deduplicated after replay.
        result["fixed_trailing_gap_mult"] = fixed_gap * giveback_ratio
        result["giveback_beta"] = float(baseline.get("giveback_beta", 0.5))
    else:
        result["giveback_beta"] = float(baseline.get("giveback_beta", 0.5)) * giveback_ratio
    result["adaptive_exit_action_id"] = action.action_id
    result["adaptive_exit_action_distance"] = action.normalized_distance()
    if action.is_baseline:
        for field_name in (
            "sl_mult",
            "trailing_activation_mult",
            "trailing_power",
            "giveback_beta",
        ):
            expected = float(baseline.get(field_name, result[field_name]))
            if not math.isclose(float(result[field_name]), expected, abs_tol=1e-12):
                raise AssertionError("baseline action changed incumbent geometry")
        if fixed_gap > 0.0 and not math.isclose(
            float(result["fixed_trailing_gap_mult"]), fixed_gap, abs_tol=1e-12
        ):
            raise AssertionError("baseline action changed incumbent fixed trailing gap")
    if float(result["sl_mult"]) > baseline_sl + 1e-12:
        raise AssertionError("adaptive stop may never widen")
    return result


def causal_hourly_asof_join(
    states: pd.DataFrame,
    hourly: pd.DataFrame,
    *,
    decision_col: str = "decision_ts",
    available_col: str = "available_at",
    by: Sequence[str] = ("symbol",),
    config: AdaptiveExitConfig | None = None,
) -> pd.DataFrame:
    """Backward as-of join with explicit age/staleness and no interpolation."""

    cfg = config or AdaptiveExitConfig()
    cfg.validate()
    left = states.copy()
    right = hourly.copy()
    left[decision_col] = pd.to_datetime(left[decision_col], utc=True, errors="raise")
    right[available_col] = pd.to_datetime(right[available_col], utc=True, errors="raise")
    keys = list(by)
    missing = [column for column in (*keys, decision_col) if column not in left]
    missing += [column for column in (*keys, available_col) if column not in right]
    if missing:
        raise AdaptiveExitContractError(f"hourly as-of join lacks columns: {missing}")
    left["__adaptive_order"] = np.arange(len(left), dtype=np.int64)
    left = left.sort_values([*keys, decision_col], kind="stable")
    right = right.sort_values([*keys, available_col], kind="stable")
    joined = pd.merge_asof(
        left,
        right,
        left_on=decision_col,
        right_on=available_col,
        by=keys or None,
        direction="backward",
        allow_exact_matches=True,
    )
    joined["hourly_feature_age_minutes"] = (
        joined[decision_col] - joined[available_col]
    ).dt.total_seconds() / 60.0
    joined["hourly_feature_stale"] = (
        joined[available_col].isna()
        | (joined["hourly_feature_age_minutes"] < 0.0)
        | (joined["hourly_feature_age_minutes"] > cfg.max_hourly_age_minutes)
    )
    if (joined[available_col] > joined[decision_col]).fillna(False).any():
        raise AdaptiveExitContractError("future hourly feature joined to a path state")
    hourly_fields = [
        column
        for column in right.columns
        if column not in {*keys, available_col}
    ]
    if cfg.stale_hourly_mode == "path_only":
        joined.loc[joined["hourly_feature_stale"], hourly_fields] = np.nan
    else:
        joined["adaptive_exit_eligible"] = ~joined["hourly_feature_stale"]
    return joined.sort_values("__adaptive_order").drop(columns="__adaptive_order")


def build_decision_states(
    rows: pd.DataFrame,
    paths: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    entry_prices: np.ndarray,
    baseline_exit_bars: np.ndarray,
    bar_minutes: int,
    hourly: pd.DataFrame | None = None,
    entry_feature_columns: Sequence[str] = (),
    hourly_feature_columns: Sequence[str] = (),
    config: AdaptiveExitConfig | None = None,
) -> pd.DataFrame:
    """Materialize causal path states while the frozen incumbent is alive."""

    cfg = config or AdaptiveExitConfig()
    cfg.validate()
    opens, highs, lows, closes = (np.asarray(value) for value in paths)
    count, bars = closes.shape
    if any(value.shape != (count, bars) for value in (opens, highs, lows)):
        raise AdaptiveExitContractError("OHLC path shapes differ")
    if len(rows) != count or len(entry_prices) != count or len(baseline_exit_bars) != count:
        raise AdaptiveExitContractError("row/path/baseline identity length mismatch")
    required = {"candidate_id", "timestamp", "side", "barrier_pct"}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise AdaptiveExitContractError(f"decision-state rows lack: {missing}")
    side = np.asarray(rows["side"], dtype=float)
    if not np.isin(side, (-1.0, 1.0)).all():
        raise AdaptiveExitContractError("side must be -1 or +1")
    entry = np.asarray(entry_prices, dtype=float)
    atr = np.asarray(rows["barrier_pct"], dtype=float)
    if np.nanmedian(atr) > 0.5:
        atr = atr / 100.0
    decision_every = max(1, int(cfg.decision_every_bars))
    records: list[pd.DataFrame] = []
    running_mfe = np.zeros(count, dtype=float)
    running_mae = np.zeros(count, dtype=float)
    previous_pnl = np.zeros(count, dtype=float)
    previous_velocity = np.zeros(count, dtype=float)
    last_mfe_bar = np.zeros(count, dtype=int)
    entry_ts = pd.to_datetime(rows["timestamp"], utc=True, errors="raise")
    for bar in range(0, bars, decision_every):
        alive = bar <= np.asarray(baseline_exit_bars, dtype=int)
        finite = np.isfinite(closes[:, bar]) & np.isfinite(entry) & (entry > 0.0)
        keep = alive & finite
        if not np.any(keep):
            continue
        favorable = np.where(
            side > 0.0,
            highs[:, bar] / entry - 1.0,
            1.0 - lows[:, bar] / entry,
        )
        adverse = np.where(
            side > 0.0,
            1.0 - lows[:, bar] / entry,
            highs[:, bar] / entry - 1.0,
        )
        new_mfe = favorable > running_mfe + EPS
        running_mfe = np.maximum(running_mfe, np.nan_to_num(favorable, nan=0.0))
        running_mae = np.maximum(running_mae, np.nan_to_num(adverse, nan=0.0))
        last_mfe_bar = np.where(new_mfe, bar, last_mfe_bar)
        pnl = side * (closes[:, bar] / entry - 1.0)
        hours_step = max(decision_every * bar_minutes / 60.0, EPS)
        velocity = (pnl - previous_pnl) / hours_step
        acceleration = (velocity - previous_velocity) / hours_step
        local = rows.loc[keep, ["candidate_id", "symbol", "side"]].copy()
        local["entry_ts"] = entry_ts.loc[keep].to_numpy()
        local["decision_ts"] = (
            entry_ts.loc[keep] + pd.to_timedelta(bar * bar_minutes, unit="m")
        ).to_numpy()
        local["path_bar"] = int(bar)
        local["trade_age_hours"] = float(bar * bar_minutes / 60.0)
        local["pnl_bps"] = pnl[keep] * 10_000.0
        local["pnl_atr"] = pnl[keep] / np.maximum(atr[keep], EPS)
        local["mfe_bps"] = running_mfe[keep] * 10_000.0
        local["mfe_atr"] = running_mfe[keep] / np.maximum(atr[keep], EPS)
        local["mae_bps"] = running_mae[keep] * 10_000.0
        local["mae_atr"] = running_mae[keep] / np.maximum(atr[keep], EPS)
        local["drawdown_from_mfe_bps"] = (running_mfe[keep] - pnl[keep]) * 10_000.0
        local["fraction_given_back"] = np.clip(
            (running_mfe[keep] - pnl[keep]) / np.maximum(running_mfe[keep], EPS),
            0.0,
            5.0,
        )
        local["time_since_mfe_hours"] = (
            (bar - last_mfe_bar[keep]) * bar_minutes / 60.0
        )
        local["new_mfe"] = new_mfe[keep]
        local["pnl_velocity_bps_per_hour"] = velocity[keep] * 10_000.0
        local["pnl_acceleration_bps_per_hour2"] = acceleration[keep] * 10_000.0
        local["atr_frac"] = atr[keep]
        for column in entry_feature_columns:
            if column in rows:
                local[f"entry__{column}"] = rows.loc[keep, column].to_numpy()
        records.append(local)
        previous_pnl = pnl
        previous_velocity = velocity
    states = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    if states.empty or hourly is None:
        return states
    available = hourly.loc[:, ["symbol", "available_at", *hourly_feature_columns]].copy()
    joined = causal_hourly_asof_join(states, available, config=cfg)
    for column in hourly_feature_columns:
        if column in joined:
            entry_value = joined.groupby("candidate_id", sort=False)[column].transform("first")
            joined[f"hourly_delta__{column}"] = pd.to_numeric(
                joined[column], errors="coerce"
            ) - pd.to_numeric(entry_value, errors="coerce")
    return joined


@dataclass(frozen=True)
class BinContract:
    edges: Mapping[str, tuple[float, ...]]

    @classmethod
    def fit(
        cls, frame: pd.DataFrame, columns: Sequence[str], n_bins: int = 5
    ) -> "BinContract":
        result: dict[str, tuple[float, ...]] = {}
        quantiles = np.linspace(0.0, 1.0, int(n_bins) + 1)[1:-1]
        for column in columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
            finite = values[np.isfinite(values)]
            if len(finite) < max(20, n_bins * 4):
                continue
            edges = np.unique(np.quantile(finite, quantiles))
            if len(edges) >= 2:
                result[str(column)] = tuple(map(float, edges))
        return cls(edges=result)

    def transform(self, frame: pd.DataFrame, column: str) -> np.ndarray:
        edges = np.asarray(self.edges[column], dtype=float)
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        missing_bin = len(edges) + 1
        return np.where(np.isfinite(values), np.digitize(values, edges), missing_bin).astype(
            np.int16
        )


def _mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 20:
        return 0.0
    x_codes, x_inverse = np.unique(x[valid], return_inverse=True)
    y_codes, y_inverse = np.unique(y[valid], return_inverse=True)
    joint = np.zeros((len(x_codes), len(y_codes)), dtype=float)
    np.add.at(joint, (x_inverse, y_inverse), 1.0)
    joint /= max(joint.sum(), EPS)
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    mask = joint > 0.0
    return float(np.sum(joint[mask] * np.log(joint[mask] / (px @ py)[mask])))


def discover_portable_cmi_features(
    states: pd.DataFrame,
    target: np.ndarray,
    *,
    candidate_features: Sequence[str],
    environment_col: str = "environment",
    config: AdaptiveExitConfig | None = None,
) -> tuple[tuple[str, ...], pd.DataFrame, BinContract]:
    """Train-only permutation-adjusted CMI with era stability penalties."""

    cfg = config or AdaptiveExitConfig()
    cfg.validate()
    frame = states.copy()
    frame["__target_bin"] = pd.qcut(
        pd.Series(np.asarray(target, dtype=float)).rank(method="first"),
        q=cfg.n_bins,
        labels=False,
        duplicates="drop",
    ).to_numpy()
    contract = BinContract.fit(frame, candidate_features, cfg.n_bins)
    rng = np.random.default_rng(cfg.random_seed)
    environments = (
        frame[environment_col].astype(str).to_numpy()
        if environment_col in frame
        else pd.to_datetime(frame["decision_ts"], utc=True).dt.to_period("M").astype(str).to_numpy()
    )
    rows: list[dict[str, Any]] = []
    for feature in candidate_features:
        if feature not in contract.edges:
            continue
        x = contract.transform(frame, feature)
        y = frame["__target_bin"].to_numpy()
        env_values: list[float] = []
        for environment in np.unique(environments):
            mask = environments == environment
            if mask.sum() < 50:
                continue
            observed = _mutual_information(x[mask], y[mask])
            nulls = []
            local_indices = np.flatnonzero(mask)
            for _ in range(cfg.cmi_permutations):
                permuted = y.copy()
                permuted[local_indices] = rng.permutation(y[local_indices])
                nulls.append(_mutual_information(x[mask], permuted[mask]))
            env_values.append(observed - float(np.mean(nulls)))
        if not env_values:
            continue
        median = float(np.median(env_values))
        mad = float(np.median(np.abs(np.asarray(env_values) - median)))
        positive_fraction = float(np.mean(np.asarray(env_values) > 0.0))
        portability = median - 0.50 * mad - max(0.0, -min(env_values))
        rows.append(
            {
                "feature": feature,
                "median_cmi_adjusted": median,
                "mad_cmi_adjusted": mad,
                "worst_environment_cmi_adjusted": float(min(env_values)),
                "positive_environment_fraction": positive_fraction,
                "portability_score": portability,
                "environments": int(len(env_values)),
            }
        )
    audit = pd.DataFrame(rows).sort_values(
        ["portability_score", "positive_environment_fraction"], ascending=False
    ) if rows else pd.DataFrame(columns=["feature", "portability_score"])
    selected = tuple(audit.head(cfg.max_state_features)["feature"].astype(str))
    return selected, audit, contract


def effective_support(
    trade_ids: Sequence[Any],
    timestamps: Sequence[Any],
    assets: Sequence[Any],
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    """Support diagnostics that do not reward duplicated path-state rows."""

    frame = pd.DataFrame(
        {
            "trade": pd.Series(trade_ids, dtype=str),
            "timestamp": pd.to_datetime(timestamps, utc=True),
            "asset": pd.Series(assets, dtype=str),
        }
    )
    weight = (
        np.ones(len(frame), dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )
    if len(weight) != len(frame):
        raise ValueError("support weight length mismatch")
    trade_weight = pd.Series(weight).groupby(frame["trade"]).sum().to_numpy(float)
    kish = float(trade_weight.sum() ** 2 / max(np.square(trade_weight).sum(), EPS))
    block = frame["timestamp"].dt.floor("12h").nunique()
    assets_n = frame["asset"].nunique()
    months = frame["timestamp"].dt.to_period("M").nunique()
    normalized = trade_weight / max(trade_weight.sum(), EPS)
    hhi = float(np.square(normalized).sum())
    return {
        "unique_trades": float(frame["trade"].nunique()),
        "kish_ess": kish,
        "blocks_12h": float(block),
        "assets": float(assets_n),
        "months": float(months),
        "trade_weight_hhi": hhi,
        "effective_support": float(
            min(kish, frame["trade"].nunique(), block * 4.0, assets_n * 100.0)
            / max(1.0 + 4.0 * hhi, 1.0)
        ),
    }


@dataclass
class BayesianActionSurface:
    """Robust factorised Gaussian action surface anchored at zero action."""

    feature_names: tuple[str, ...]
    center: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    coefficient_sd: np.ndarray
    noise_sd_bps: float
    support: dict[str, float]
    config: AdaptiveExitConfig

    @staticmethod
    def _action_basis(
        actions: Sequence[ExitAction], *, include_pairs: bool = True
    ) -> np.ndarray:
        raw = np.asarray([action.levels for action in actions], dtype=float)
        baseline = np.asarray(BASELINE_LEVELS, dtype=float)
        delta = (raw - baseline) / 2.0
        delta[:, 0] = raw[:, 0] / 4.0
        pairs = np.column_stack(
            [delta[:, left] * delta[:, right] for left in range(4) for right in range(left + 1, 4)]
        )
        return np.column_stack([delta, pairs]) if include_pairs else delta

    @classmethod
    def fit(
        cls,
        states: pd.DataFrame,
        actions: Sequence[ExitAction],
        delta_q_bps: np.ndarray,
        *,
        feature_names: Sequence[str],
        sample_weight: np.ndarray | None = None,
        config: AdaptiveExitConfig | None = None,
    ) -> "BayesianActionSurface":
        cfg = config or AdaptiveExitConfig()
        cfg.validate()
        state_raw = states.loc[:, list(feature_names)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if state_raw.shape[1]:
            center = np.nanmedian(state_raw, axis=0)
            q25 = np.nanquantile(state_raw, 0.25, axis=0)
            q75 = np.nanquantile(state_raw, 0.75, axis=0)
            scale = np.maximum(q75 - q25, 1.0e-6)
        else:
            center = np.empty(0, dtype=float)
            scale = np.empty(0, dtype=float)
        state = np.nan_to_num((state_raw - center) / scale, nan=0.0, posinf=5.0, neginf=-5.0)
        state = np.clip(state, -5.0, 5.0)
        action_basis = cls._action_basis(
            actions, include_pairs=cfg.include_action_pair_synergies
        )
        if delta_q_bps.shape != (len(states), len(actions)):
            raise ValueError("delta-Q matrix does not match states/actions")
        base_index = next(index for index, action in enumerate(actions) if action.is_baseline)
        if not np.allclose(delta_q_bps[:, base_index], 0.0, atol=1.0e-8):
            raise AdaptiveExitContractError("baseline delta-Q must be exactly zero")
        # Rows are state/action pairs.  Accumulate the normal equations one
        # action at a time: a 20k x 625 x feature design would otherwise use
        # several GB even though the factorised parameter count is small.
        target_matrix = np.asarray(delta_q_bps, dtype=float)
        state_weight = (
            np.ones(len(states), dtype=float)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float)
        )
        design_width = action_basis.shape[1] + state.shape[1] * 4
        coefficients = np.zeros(design_width, dtype=float)
        prior = float(cfg.ridge_prior)
        for _ in range(cfg.robust_iterations):
            precision = prior * np.eye(design_width)
            rhs = np.zeros(design_width, dtype=float)
            squared_error = 0.0
            error_weight = 0.0
            for action_index, basis in enumerate(action_basis):
                if action_index == base_index:
                    continue
                design = np.hstack(
                    [
                        np.broadcast_to(basis, (len(state), len(basis))),
                        (state[:, :, None] * basis[None, None, :4]).reshape(
                            len(state), -1
                        ),
                    ]
                )
                target = target_matrix[:, action_index]
                residual = target - design @ coefficients
                robust = np.minimum(
                    1.0,
                    cfg.huber_delta_bps / np.maximum(np.abs(residual), EPS),
                )
                weight = state_weight * robust
                xtw = design.T * weight
                precision += xtw @ design
                rhs += xtw @ target
                squared_error += float(np.sum(weight * np.square(residual)))
                error_weight += float(np.sum(weight))
            coefficients = np.linalg.solve(precision, rhs)
        noise = float(np.sqrt(squared_error / max(error_weight, EPS)))
        covariance = np.linalg.pinv(precision) * max(noise**2, 1.0)
        coefficient_sd = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        support = effective_support(
            states["candidate_id"],
            states["decision_ts"],
            states.get("symbol", pd.Series("", index=states.index)),
            sample_weight,
        )
        shrink = support["effective_support"] / (
            support["effective_support"] + cfg.support_prior
        )
        coefficients *= shrink
        coefficient_sd /= max(math.sqrt(shrink), 1.0e-3)
        return cls(
            tuple(feature_names), center, scale, coefficients, coefficient_sd,
            max(noise, 1.0), support, cfg,
        )

    def predict(
        self, states: pd.DataFrame, actions: Sequence[ExitAction]
    ) -> tuple[np.ndarray, np.ndarray]:
        raw = states.loc[:, list(self.feature_names)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        state = np.nan_to_num((raw - self.center) / self.scale, nan=0.0, posinf=5.0, neginf=-5.0)
        state = np.clip(state, -5.0, 5.0)
        action_basis = self._action_basis(
            actions,
            include_pairs=self.config.include_action_pair_synergies,
        )
        mean = np.empty((len(states), len(actions)), dtype=np.float32)
        sd = np.empty_like(mean)
        for action_index, basis in enumerate(action_basis):
            design = np.hstack(
                [
                    np.broadcast_to(basis, (len(state), len(basis))),
                    (state[:, :, None] * basis[None, None, :4]).reshape(
                        len(state), -1
                    ),
                ]
            )
            mean[:, action_index] = design @ self.coefficients
            variance = (
                np.square(design) @ np.square(self.coefficient_sd)
            ) + self.noise_sd_bps**2
            sd[:, action_index] = np.sqrt(np.maximum(variance, 0.0))
        baseline = np.asarray([action.is_baseline for action in actions])
        mean[:, baseline] = 0.0
        sd[:, baseline] = 0.0
        return mean, sd

    def choose_actions(
        self, states: pd.DataFrame, actions: Sequence[ExitAction]
    ) -> pd.DataFrame:
        mean, sd = self.predict(states, actions)
        distance = np.asarray([action.normalized_distance() for action in actions])
        score = (
            mean
            - self.config.uncertainty_lambda * sd
            - self.config.action_distance_lambda_bps * distance[None, :]
        )
        baseline_index = next(index for index, action in enumerate(actions) if action.is_baseline)
        selected = np.argmax(score, axis=1)
        best_score = score[np.arange(len(states)), selected]
        selected = np.where(best_score >= self.config.min_action_edge_bps, selected, baseline_index)
        return pd.DataFrame(
            {
                "selected_action_index": selected.astype(np.int16),
                "selected_action_id": [actions[index].action_id for index in selected],
                "predicted_delta_q_bps": mean[np.arange(len(states)), selected],
                "predicted_delta_q_sd_bps": sd[np.arange(len(states)), selected],
                "conservative_delta_q_bps": score[np.arange(len(states)), selected],
                "action_distance": distance[selected],
                "action_intensity": [actions[index].intensity() for index in selected],
                "intervened": selected != baseline_index,
            }
        )


def replay_static_action_grid(
    rows: pd.DataFrame,
    paths: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    baseline_params: Mapping[str, Any],
    *,
    cost_pct: float,
    size_power: float,
    actions: Sequence[ExitAction] | None = None,
    simulator: Callable[..., Mapping[str, Any]] | None = None,
    config: AdaptiveExitConfig | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], pd.DataFrame]:
    """Replay the 625 actions on identical paths in bounded action batches."""

    cfg = config or AdaptiveExitConfig()
    cfg.validate()
    grid = tuple(actions or build_action_grid(cfg))
    if simulator is None:
        from .simple_policy_optimiser import simulate_and_score as simulator
    count = len(rows)
    values = np.empty((count, len(grid)), dtype=np.float32)
    exit_bars = np.empty((count, len(grid)), dtype=np.int16)
    exit_reasons = np.empty((count, len(grid)), dtype=object)
    audit_rows: list[dict[str, Any]] = []
    for start in range(0, len(grid), cfg.action_batch_size):
        for action_index in range(start, min(start + cfg.action_batch_size, len(grid))):
            action = grid[action_index]
            params = materialize_action_params(baseline_params, action, cfg)
            metrics = simulator(
                rows.copy(),
                *paths,
                cost_pct=cost_pct,
                size_power=size_power,
                max_concurrent_trades=1_000_000_000,
                max_concurrent_per_asset=1_000_000_000,
                max_new_entries_per_bar=1_000_000_000,
                **params,
            )
            selected = np.asarray(metrics["selected_mask"], dtype=bool)
            if selected.shape != (count,) or not selected.all():
                raise AdaptiveExitContractError(
                    f"action replay dropped candidates for {action.action_id}"
                )
            values[:, action_index] = np.asarray(metrics["net_returns"], dtype=float) * 10_000.0
            exit_bars[:, action_index] = np.asarray(metrics["exit_bars"], dtype=np.int16)
            exit_reasons[:, action_index] = np.asarray(metrics["exit_reason"], dtype=object)
            audit_rows.append(
                {
                    "action_index": action_index,
                    "action_id": action.action_id,
                    "is_baseline": action.is_baseline,
                    "distance": action.normalized_distance(),
                    "intensity": action.intensity(),
                    "mean_net_bps": float(np.mean(values[:, action_index])),
                    "trades": count,
                }
            )
    baseline_index = next(index for index, action in enumerate(grid) if action.is_baseline)
    delta = values - values[:, [baseline_index]]
    delta[:, baseline_index] = 0.0
    representatives: dict[str, str] = {}
    for action_index, row in enumerate(audit_rows):
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(values[:, action_index]).tobytes())
        digest.update(np.ascontiguousarray(exit_bars[:, action_index]).tobytes())
        digest.update("\x1f".join(map(str, exit_reasons[:, action_index])).encode())
        behavior_id = digest.hexdigest()
        representative = representatives.setdefault(behavior_id, str(row["action_id"]))
        row["behavior_id"] = behavior_id
        row["behavior_representative_action_id"] = representative
        row["behaviorally_duplicate"] = representative != row["action_id"]
    return delta, {"net_bps": values, "exit_bars": exit_bars, "exit_reason": exit_reasons}, pd.DataFrame(audit_rows)


def chronological_purged_folds(
    rows: pd.DataFrame,
    *,
    n_folds: int = 4,
    timestamp_col: str = "timestamp",
    purge_hours: float = 12.0,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Expanding chronological folds with whole-trade identity and purge."""

    timestamps = pd.to_datetime(rows[timestamp_col], utc=True, errors="raise")
    unique_months = np.asarray(sorted(timestamps.dt.to_period("M").astype(str).unique()))
    if len(unique_months) < 3:
        return ()
    validation_months = np.array_split(unique_months[1:], min(n_folds, len(unique_months) - 1))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    purge = pd.Timedelta(hours=float(purge_hours))
    for months in validation_months:
        if not len(months):
            continue
        val_mask = timestamps.dt.to_period("M").astype(str).isin(months).to_numpy()
        val_start = timestamps[val_mask].min()
        train_mask = (timestamps < val_start - purge).to_numpy()
        train = np.flatnonzero(train_mask)
        validation = np.flatnonzero(val_mask)
        if len(train) and len(validation):
            folds.append((train, validation))
    return tuple(folds)


def run_strict_oof_static_ablation(
    trade_states: pd.DataFrame,
    actions: Sequence[ExitAction],
    delta_q_bps: np.ndarray,
    action_net_bps: np.ndarray,
    *,
    candidate_features: Sequence[str],
    output_dir: str | Path | None = None,
    config: AdaptiveExitConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run the strict-OOF entry-action ladder before sequential promotion.

    This is the scalable diagnostic ladder from the shared specification.  It
    deliberately remains labelled ``static``: passing it is necessary but not
    sufficient.  Only the later sequential path replay may promote an overlay.
    All binning, CMI discovery, robust fitting and shrinkage occur inside each
    chronological training fold.
    """

    cfg = config or AdaptiveExitConfig()
    cfg.validate()
    frame = trade_states.reset_index(drop=True).copy()
    if delta_q_bps.shape != action_net_bps.shape:
        raise ValueError("delta-Q and action-value matrices differ")
    if delta_q_bps.shape != (len(frame), len(actions)):
        raise ValueError("OOF action matrices do not align to trade states")
    folds = chronological_purged_folds(
        frame,
        n_folds=4,
        timestamp_col="decision_ts",
        purge_hours=cfg.purge_hours,
    )
    if not folds:
        raise AdaptiveExitContractError("not enough chronological months for OOF")
    baseline_index = next(index for index, action in enumerate(actions) if action.is_baseline)
    decision_frames: list[pd.DataFrame] = []
    feature_audits: list[pd.DataFrame] = []
    model_manifests: list[dict[str, Any]] = []
    arms = (
        "A_frozen_baseline",
        "B_train_best_static_joint",
        "C_raw_unshrunk_surface",
        "D_bayes_action_main",
        "E_bayes_state_main",
        "G_action_pair_synergies",
        "H_sparse_cmi_bayes_support",
    )
    for fold_index, (train_index, validation_index) in enumerate(folds, start=1):
        if len(train_index) < cfg.min_train_trades or len(validation_index) < cfg.min_validation_trades:
            continue
        train = frame.iloc[train_index].copy()
        validation = frame.iloc[validation_index].copy()
        oracle_train = np.nanmax(delta_q_bps[train_index], axis=1)
        selected, audit, _bins = discover_portable_cmi_features(
            train,
            oracle_train,
            candidate_features=[feature for feature in candidate_features if feature in train],
            config=cfg,
        )
        audit["fold"] = fold_index
        feature_audits.append(audit)
        best_static_index = int(np.nanargmax(np.nanmean(delta_q_bps[train_index], axis=0)))
        fitted: dict[str, BayesianActionSurface] = {}
        fitted["C_raw_unshrunk_surface"] = BayesianActionSurface.fit(
            train,
            actions,
            delta_q_bps[train_index],
            feature_names=selected,
            config=replace(
                cfg,
                support_prior=1.0e-6,
                ridge_prior=1.0e-4,
                uncertainty_lambda=0.0,
                min_action_edge_bps=0.0,
            ),
        )
        fitted["D_bayes_action_main"] = BayesianActionSurface.fit(
            train,
            actions,
            delta_q_bps[train_index],
            feature_names=(),
            config=replace(cfg, include_action_pair_synergies=False),
        )
        fitted["E_bayes_state_main"] = BayesianActionSurface.fit(
            train,
            actions,
            delta_q_bps[train_index],
            feature_names=selected,
            config=replace(cfg, include_action_pair_synergies=False),
        )
        fitted["G_action_pair_synergies"] = BayesianActionSurface.fit(
            train,
            actions,
            delta_q_bps[train_index],
            feature_names=selected,
            config=replace(cfg, support_prior=1.0e-6),
        )
        fitted["H_sparse_cmi_bayes_support"] = BayesianActionSurface.fit(
            train,
            actions,
            delta_q_bps[train_index],
            feature_names=selected,
            config=cfg,
        )
        base_decision = pd.DataFrame(
            {
                "selected_action_index": np.repeat(baseline_index, len(validation)),
                "selected_action_id": np.repeat(actions[baseline_index].action_id, len(validation)),
                "predicted_delta_q_bps": np.zeros(len(validation)),
                "predicted_delta_q_sd_bps": np.zeros(len(validation)),
                "conservative_delta_q_bps": np.zeros(len(validation)),
                "action_distance": np.zeros(len(validation)),
                "action_intensity": np.repeat("mild", len(validation)),
                "intervened": np.repeat(False, len(validation)),
            }
        )
        arm_decisions: dict[str, pd.DataFrame] = {
            "A_frozen_baseline": base_decision,
            "B_train_best_static_joint": pd.DataFrame(
                {
                    "selected_action_index": np.repeat(best_static_index, len(validation)),
                    "selected_action_id": np.repeat(actions[best_static_index].action_id, len(validation)),
                    "predicted_delta_q_bps": np.repeat(
                        float(np.nanmean(delta_q_bps[train_index, best_static_index])), len(validation)
                    ),
                    "predicted_delta_q_sd_bps": np.repeat(
                        float(np.nanstd(delta_q_bps[train_index, best_static_index])), len(validation)
                    ),
                    "conservative_delta_q_bps": np.repeat(
                        float(np.nanmean(delta_q_bps[train_index, best_static_index])), len(validation)
                    ),
                    "action_distance": np.repeat(actions[best_static_index].normalized_distance(), len(validation)),
                    "action_intensity": np.repeat(actions[best_static_index].intensity(), len(validation)),
                    "intervened": np.repeat(best_static_index != baseline_index, len(validation)),
                }
            ),
        }
        for arm, surface in fitted.items():
            arm_decisions[arm] = surface.choose_actions(validation, actions)
            model_manifests.append(
                {
                    "fold": fold_index,
                    "arm": arm,
                    "train_rows": int(len(train)),
                    "validation_rows": int(len(validation)),
                    "train_end": pd.to_datetime(train["decision_ts"], utc=True).max(),
                    "validation_start": pd.to_datetime(validation["decision_ts"], utc=True).min(),
                    "features": list(surface.feature_names),
                    "support": surface.support,
                    "config": asdict(surface.config),
                }
            )
        for arm in arms:
            local = arm_decisions[arm].reset_index(drop=True)
            selected_index = local["selected_action_index"].to_numpy(int)
            local.insert(0, "fold", fold_index)
            local.insert(1, "arm", arm)
            local.insert(2, "candidate_id", validation["candidate_id"].astype(str).to_numpy())
            local["timestamp"] = pd.to_datetime(validation["decision_ts"], utc=True).to_numpy()
            local["baseline_net_bps"] = action_net_bps[validation_index, baseline_index]
            local["adaptive_net_bps"] = action_net_bps[validation_index, selected_index]
            local["oracle_adaptive_net_bps"] = np.nanmax(action_net_bps[validation_index], axis=1)
            if "mfe_bps" in validation:
                local["mfe_bps"] = validation["mfe_bps"].to_numpy(float)
            decision_frames.append(local)
    if not decision_frames:
        raise AdaptiveExitContractError("OOF folds failed minimum support")
    decisions = pd.concat(decision_frames, ignore_index=True)
    metrics_rows: list[dict[str, Any]] = []
    monthly_frames: list[pd.DataFrame] = []
    weekly_frames: list[pd.DataFrame] = []
    for arm, part in decisions.groupby("arm", sort=False):
        metrics, monthly, weekly = evaluate_adaptive_exit(part)
        metrics_rows.append({"arm": arm, **metrics})
        monthly.insert(0, "arm", arm)
        weekly.insert(0, "arm", arm)
        monthly_frames.append(monthly)
        weekly_frames.append(weekly)
    metrics_table = pd.DataFrame(metrics_rows).sort_values(
        ["delta_ev_bps", "relative_month_stability_ratio"], ascending=False
    )
    winner = metrics_table.iloc[0].to_dict()
    gate = promotion_gate(winner, cfg)
    # Static OOF can nominate a sequential challenger but can never promote by itself.
    gate["static_oof_only"] = True
    gate["decision"] = "RUN_SEQUENTIAL_OOF_REPLAY" if gate["passes"] else "KEEP_FROZEN_BASELINE"
    gate["passes"] = False
    summary = {
        "schema": SCHEMA,
        "stage": "strict_oof_static_action_ablation",
        "folds": int(decisions["fold"].nunique()),
        "winner": winner,
        "promotion_gate": gate,
        "models": model_manifests,
    }
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        decisions.to_parquet(out / "adaptive_exit_oof_decisions.parquet", index=False)
        metrics_table.to_csv(out / "adaptive_exit_ablation_results.csv", index=False)
        pd.concat(monthly_frames, ignore_index=True).to_parquet(
            out / "adaptive_exit_monthly_metrics.parquet", index=False
        )
        pd.concat(weekly_frames, ignore_index=True).to_parquet(
            out / "adaptive_exit_weekly_metrics.parquet", index=False
        )
        if feature_audits:
            pd.concat(feature_audits, ignore_index=True).to_parquet(
                out / "adaptive_exit_cmi_audit.parquet", index=False
            )
        (out / "adaptive_exit_static_oof_summary.json").write_text(
            json.dumps(_safe_json(summary), indent=2, sort_keys=True) + "\n"
        )
    return decisions, metrics_table, summary


def _max_drawdown(values: np.ndarray) -> float:
    equity = np.cumsum(np.asarray(values, dtype=float))
    if not len(equity):
        return 0.0
    return float(np.min(equity - np.maximum.accumulate(np.r_[0.0, equity])[-len(equity):]))


def _sortino(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    downside = values[values < 0.0]
    if not len(values) or not len(downside):
        return float("nan")
    return float(np.mean(values) / max(np.sqrt(np.mean(np.square(downside))), EPS))


def _cvar(values: np.ndarray, fraction: float = 0.05) -> float:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return float("nan")
    count = max(1, int(math.ceil(len(values) * fraction)))
    return float(np.mean(np.partition(values, count - 1)[:count]))


def evaluate_adaptive_exit(
    trades: pd.DataFrame,
    *,
    baseline_col: str = "baseline_net_bps",
    adaptive_col: str = "adaptive_net_bps",
    mfe_col: str = "mfe_bps",
    prediction_col: str = "conservative_delta_q_bps",
    action_distance_col: str = "action_distance",
    intervention_col: str = "intervened",
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Compute economic, calibration, intervention, and relative stability metrics."""

    frame = trades.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    baseline = pd.to_numeric(frame[baseline_col], errors="raise").to_numpy(float)
    adaptive = pd.to_numeric(frame[adaptive_col], errors="raise").to_numpy(float)
    delta = adaptive - baseline
    frame["delta_bps"] = delta
    frame["month"] = frame["timestamp"].dt.to_period("M").astype(str)
    frame["week"] = frame["timestamp"].dt.to_period("W-SUN").astype(str)
    monthly = frame.groupby("month", sort=True).agg(
        trades=(adaptive_col, "size"),
        baseline_ev_bps=(baseline_col, "mean"),
        adaptive_ev_bps=(adaptive_col, "mean"),
        delta_ev_bps=("delta_bps", "mean"),
        baseline_pnl_bps=(baseline_col, "sum"),
        adaptive_pnl_bps=(adaptive_col, "sum"),
    ).reset_index()
    weekly = frame.groupby("week", sort=True).agg(
        trades=(adaptive_col, "size"),
        baseline_ev_bps=(baseline_col, "mean"),
        adaptive_ev_bps=(adaptive_col, "mean"),
        delta_ev_bps=("delta_bps", "mean"),
    ).reset_index()
    month_delta = monthly["delta_ev_bps"].to_numpy(float)
    week_delta = weekly["delta_ev_bps"].to_numpy(float)
    delta_mean = float(np.mean(delta)) if len(delta) else 0.0
    month_mad = float(np.median(np.abs(month_delta - np.median(month_delta)))) if len(month_delta) else float("nan")
    week_mad = float(np.median(np.abs(week_delta - np.median(week_delta)))) if len(week_delta) else float("nan")
    baseline_dd = _max_drawdown(baseline)
    adaptive_dd = _max_drawdown(adaptive)
    delta_dd_abs = abs(adaptive_dd) - abs(baseline_dd)
    winners = baseline > 0.0
    losers = ~winners
    mfe_source = frame[mfe_col] if mfe_col in frame else pd.Series(np.nan, index=frame.index)
    mfe = pd.to_numeric(mfe_source, errors="coerce").to_numpy(float)
    distance_source = (
        frame[action_distance_col]
        if action_distance_col in frame
        else pd.Series(0.0, index=frame.index)
    )
    distance = pd.to_numeric(distance_source, errors="coerce").fillna(0.0).to_numpy(float)
    interventions = frame.get(intervention_col, pd.Series(False, index=frame.index)).astype(bool).to_numpy()
    baseline_winner_pnl = float(baseline[winners].sum())
    metrics: dict[str, Any] = {
        "trades": int(len(frame)),
        "baseline_ev_bps": float(np.mean(baseline)) if len(frame) else 0.0,
        "adaptive_ev_bps": float(np.mean(adaptive)) if len(frame) else 0.0,
        "delta_ev_bps": delta_mean,
        "baseline_pnl_bps": float(np.sum(baseline)),
        "adaptive_pnl_bps": float(np.sum(adaptive)),
        "delta_pnl_bps": float(np.sum(delta)),
        "baseline_max_drawdown_bps": baseline_dd,
        "adaptive_max_drawdown_bps": adaptive_dd,
        "delta_abs_max_drawdown_bps": delta_dd_abs,
        "delta_pnl_over_abs_delta_maxdd": float(np.sum(delta) / max(abs(delta_dd_abs), EPS)),
        "baseline_sortino": _sortino(baseline),
        "adaptive_sortino": _sortino(adaptive),
        "delta_sortino": _sortino(adaptive) - _sortino(baseline),
        "baseline_trade_cvar_5_bps": _cvar(baseline),
        "adaptive_trade_cvar_5_bps": _cvar(adaptive),
        "delta_trade_cvar_5_bps": _cvar(adaptive) - _cvar(baseline),
        "winner_retention": float(adaptive[winners].sum() / max(baseline_winner_pnl, EPS)),
        "adaptive_mfe_capture": float(np.nansum(adaptive) / max(np.nansum(mfe), EPS)),
        "baseline_mfe_capture": float(np.nansum(baseline) / max(np.nansum(mfe), EPS)),
        "delta_mfe_capture": float(np.nansum(adaptive - baseline) / max(np.nansum(mfe), EPS)),
        "delta_given_baseline_winner_bps": float(np.mean(delta[winners])) if winners.any() else float("nan"),
        "delta_winners_bps": float(np.mean(delta[winners])) if winners.any() else float("nan"),
        "delta_losers_bps": float(np.mean(delta[losers])) if losers.any() else float("nan"),
        "intervention_trade_rate": float(np.mean(interventions)) if len(frame) else 0.0,
        "mean_action_distance": float(np.mean(distance)) if len(frame) else 0.0,
        "action_efficiency_bps": float(np.sum(delta) / max(np.sum(distance), EPS)),
        "positive_month_fraction": float(np.mean(month_delta > 0.0)) if len(month_delta) else float("nan"),
        "positive_week_fraction": float(np.mean(week_delta > 0.0)) if len(week_delta) else float("nan"),
        "worst_month_delta_ev_bps": float(np.min(month_delta)) if len(month_delta) else float("nan"),
        "worst_week_delta_ev_bps": float(np.min(week_delta)) if len(week_delta) else float("nan"),
        "month_delta_ev_mad_bps": month_mad,
        "week_delta_ev_mad_bps": week_mad,
        # Relative stability is reported next to absolute dispersion.  The
        # signed form penalises instability when mean uplift is weak/negative.
        "relative_month_stability_ratio": float(delta_mean / max(month_mad, EPS)),
        "relative_week_stability_ratio": float(delta_mean / max(week_mad, EPS)),
        "month_cv_abs": float(np.std(month_delta) / max(abs(np.mean(month_delta)), EPS)) if len(month_delta) else float("nan"),
        "week_cv_abs": float(np.std(week_delta) / max(abs(np.mean(week_delta)), EPS)) if len(week_delta) else float("nan"),
        "worst_month_relative_to_baseline": float(
            monthly.loc[monthly["delta_ev_bps"].idxmin(), "adaptive_ev_bps"]
            / max(abs(monthly.loc[monthly["delta_ev_bps"].idxmin(), "baseline_ev_bps"]), EPS)
        ) if len(monthly) else float("nan"),
    }
    daily = frame.groupby(frame["timestamp"].dt.floor("D"), sort=True).agg(
        baseline=(baseline_col, "sum"), adaptive=(adaptive_col, "sum")
    )
    metrics.update(
        {
            "baseline_daily_cvar_5_bps": _cvar(daily["baseline"].to_numpy(float)),
            "adaptive_daily_cvar_5_bps": _cvar(daily["adaptive"].to_numpy(float)),
            "delta_daily_cvar_5_bps": _cvar(daily["adaptive"].to_numpy(float)) - _cvar(daily["baseline"].to_numpy(float)),
        }
    )
    if prediction_col in frame:
        prediction = pd.to_numeric(frame[prediction_col], errors="coerce")
        valid = prediction.notna()
        calibration = frame.loc[valid, [prediction_col, "delta_bps"]].copy()
        calibration["bucket"] = pd.qcut(
            calibration[prediction_col].rank(method="first"),
            q=min(10, max(2, len(calibration) // 20)),
            labels=False,
            duplicates="drop",
        )
        calibration = calibration.groupby("bucket", sort=True).agg(
            rows=("delta_bps", "size"),
            predicted_conservative_delta_bps=(prediction_col, "mean"),
            realised_delta_bps=("delta_bps", "mean"),
        ).reset_index()
        if len(calibration) >= 2:
            left_rank = calibration["predicted_conservative_delta_bps"].rank()
            right_rank = calibration["realised_delta_bps"].rank()
            metrics["posterior_calibration_spearman"] = float(
                np.corrcoef(left_rank, right_rank)[0, 1]
            )
            metrics["posterior_calibration_monotonic_violations"] = int(
                np.sum(np.diff(calibration["realised_delta_bps"]) < 0.0)
            )
    else:
        calibration = pd.DataFrame()
    for intensity in ("mild", "medium", "aggressive"):
        if "action_intensity" in frame:
            mask = frame["action_intensity"].eq(intensity).to_numpy()
            metrics[f"{intensity}_action_rate"] = float(np.mean(mask))
            metrics[f"{intensity}_delta_ev_bps"] = float(np.mean(delta[mask])) if mask.any() else float("nan")
    if "oracle_adaptive_net_bps" in frame:
        oracle_delta = pd.to_numeric(frame["oracle_adaptive_net_bps"], errors="coerce").to_numpy(float) - baseline
        metrics["portable_oracle_capture_ratio"] = float(np.sum(delta) / max(np.sum(np.maximum(oracle_delta, 0.0)), EPS))
    return metrics, monthly, weekly


def promotion_gate(metrics: Mapping[str, Any], config: AdaptiveExitConfig | None = None) -> dict[str, Any]:
    cfg = config or AdaptiveExitConfig()
    checks = {
        "minimum_uplift": float(metrics.get("delta_ev_bps", -np.inf)) >= cfg.required_min_uplift_bps,
        "positive_month_coverage": float(metrics.get("positive_month_fraction", -np.inf)) >= cfg.required_positive_month_fraction,
        "worst_month": float(metrics.get("worst_month_delta_ev_bps", -np.inf)) >= cfg.required_worst_month_delta_bps,
        "relative_month_stability": float(metrics.get("relative_month_stability_ratio", -np.inf)) >= cfg.required_relative_stability_ratio,
        "winner_retention": float(metrics.get("winner_retention", 0.0)) >= 0.85,
        "no_trade_cvar_harm": float(metrics.get("delta_trade_cvar_5_bps", -np.inf)) >= 0.0,
    }
    return {
        "checks": checks,
        "passes": bool(all(checks.values())),
        "decision": "PROMOTE_ADAPTIVE_EXIT" if all(checks.values()) else "KEEP_FROZEN_BASELINE",
    }


def _safe_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_after_global_policy_optimisation(
    *,
    rows: pd.DataFrame,
    paths: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    baseline_params: Mapping[str, Any],
    cost_pct: float,
    size_power: float,
    output_dir: str | Path,
    simulator: Callable[..., Mapping[str, Any]] | None = None,
    config: AdaptiveExitConfig | None = None,
    entry_feature_columns: Sequence[str] = (),
    hourly_features: pd.DataFrame | None = None,
    hourly_feature_columns: Sequence[str] = (),
) -> dict[str, Any]:
    """Run the bounded post-global adaptive-exit research stage.

    This runner materialises the full counterfactual grid and decision-state
    ledger.  Model fitting is skipped, explicitly and fail-closed, when there
    is insufficient chronological support.  A later versioned runner can
    consume these immutable artifacts without rerunning paths.
    """

    cfg = config or AdaptiveExitConfig()
    cfg.validate()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    actions = build_action_grid(cfg)
    ordered = rows.copy().reset_index(drop=True)
    if "candidate_id" not in ordered:
        ordered["candidate_id"] = [f"adaptive_exit_{index}" for index in range(len(ordered))]
    if ordered["candidate_id"].duplicated().any():
        raise AdaptiveExitContractError("candidate_id must be unique")
    if len(ordered) > cfg.max_counterfactual_trades:
        # Deterministic equal-month subsampling avoids recent-regime dominance.
        month = pd.to_datetime(ordered["timestamp"], utc=True).dt.to_period("M")
        groups = list(ordered.groupby(month, sort=True))
        quota = max(1, cfg.max_counterfactual_trades // max(len(groups), 1))
        selected = pd.concat(
            [part.sort_values("candidate_id", kind="stable").head(quota) for _, part in groups]
        ).head(cfg.max_counterfactual_trades)
        indices = selected.index.to_numpy(int)
        ordered = selected.reset_index(drop=True)
        paths = tuple(np.asarray(array)[indices] for array in paths)  # type: ignore[assignment]
    delta_q, replay, action_audit = replay_static_action_grid(
        ordered,
        paths,
        baseline_params,
        cost_pct=cost_pct,
        size_power=size_power,
        actions=actions,
        simulator=simulator,
        config=cfg,
    )
    baseline_index = next(index for index, action in enumerate(actions) if action.is_baseline)
    baseline_metrics = (simulator or __import__(
        "extreme_price_movements.simple_policy_optimiser", fromlist=["simulate_and_score"]
    ).simulate_and_score)(
        ordered.copy(), *paths, cost_pct=cost_pct, size_power=size_power,
        max_concurrent_trades=1_000_000_000,
        max_concurrent_per_asset=1_000_000_000,
        max_new_entries_per_bar=1_000_000_000,
        **baseline_params,
    )
    if not np.allclose(
        replay["net_bps"][:, baseline_index],
        np.asarray(baseline_metrics["net_returns"], dtype=float) * 10_000.0,
        atol=1.0e-6,
    ):
        raise AdaptiveExitContractError("zero-action replay failed incumbent parity")
    states = build_decision_states(
        ordered,
        paths,
        entry_prices=np.asarray(baseline_metrics["entry_prices"], dtype=float),
        baseline_exit_bars=np.asarray(baseline_metrics["exit_bars"], dtype=int),
        bar_minutes=max(1, int(round(12.0 * 60.0 / paths[3].shape[1]))),
        hourly=hourly_features,
        entry_feature_columns=entry_feature_columns,
        hourly_feature_columns=hourly_feature_columns,
        config=cfg,
    )
    decision_path = out / "adaptive_exit_decision_states.parquet"
    action_path = out / "counterfactual_action_values.npz"
    audit_path = out / "action_grid_audit.parquet"
    states.to_parquet(decision_path, index=False)
    np.savez_compressed(
        action_path,
        delta_q_bps=delta_q,
        net_bps=replay["net_bps"],
        exit_bars=replay["exit_bars"],
        candidate_id=ordered["candidate_id"].astype(str).to_numpy(),
        action_id=np.asarray([action.action_id for action in actions]),
    )
    action_audit.to_parquet(audit_path, index=False)
    manifest = {
        "schema": SCHEMA,
        "status": "COUNTERFACTUAL_LEDGER_COMPLETE_MODEL_NOT_PROMOTED",
        "config": asdict(cfg),
        "baseline_params": dict(baseline_params),
        "cost_pct_per_side": float(cost_pct),
        "size_power": float(size_power),
        "candidate_rows": int(len(ordered)),
        "decision_states": int(len(states)),
        "actions": int(len(actions)),
        "baseline_action_id": actions[baseline_index].action_id,
        "baseline_parity_max_abs_bps": float(
            np.max(np.abs(
                replay["net_bps"][:, baseline_index]
                - np.asarray(baseline_metrics["net_returns"], dtype=float) * 10_000.0
            ))
        ),
        "causality": {
            "entry": "inherited frozen next-executable-event policy",
            "hourly_join": "backward_asof_available_at_lte_decision_ts",
            "downstream_fitting": "strict chronological OOF with 12h purge required",
            "promotion": "fail_closed_until_oof_sequential_replay_gate",
        },
        "outputs_sha256": {
            decision_path.name: _sha(decision_path),
            action_path.name: _sha(action_path),
            audit_path.name: _sha(audit_path),
        },
    }
    manifest_path = out / "run_manifest.json"
    manifest_path.write_text(json.dumps(_safe_json(manifest), indent=2, sort_keys=True) + "\n")
    return manifest


__all__ = [
    "ACTION_COMPONENTS",
    "AdaptiveExitConfig",
    "AdaptiveExitContractError",
    "BayesianActionSurface",
    "BinContract",
    "ExitAction",
    "build_action_grid",
    "build_decision_states",
    "causal_hourly_asof_join",
    "chronological_purged_folds",
    "discover_portable_cmi_features",
    "effective_support",
    "evaluate_adaptive_exit",
    "materialize_action_params",
    "promotion_gate",
    "replay_static_action_grid",
    "run_after_global_policy_optimisation",
]
