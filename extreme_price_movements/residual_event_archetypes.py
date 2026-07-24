"""Causal residual-event archetypes for side x base-archetype state discovery.

This module is deliberately a research/state-generation layer, not a policy
overlay.  It answers a narrower question than calibration: *which observable
pre-entry states precede unusually large, persistent local residuals?*

The contract is intentionally strict:

* global tail EV and side x archetype score thresholds are fitted on train rows;
* realized outcomes are used only to define train/OOS assessment labels and
  train-side AE/GMM economic priors;
* AE/GMM transforms exposed to meta/base consume pre-entry columns only;
* recent hit-rate / EV performance is never an input.  A causal 8-day smoother
  is provided solely as an assessment overlay.

The resulting states are local to ``side_name x archetype_policy_key``.  A
side-level model is used only when a local stream lacks enough support.  The
shared market layer is intentionally separate and optional: local failure modes
are the primary discovery target.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional in a minimal test environment.
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

from .features_gmm_ae import (
    ae_gmm_feature_columns,
    fit_ae_gmm_state,
    transform_ae_gmm_features,
)
from .features_negative_residuals import add_residual_state_target_composites
from .local_economic_aegmm import BASE_DIRECTIONAL_STATE_FEATURES

RESIDUAL_EVENT_PREFIX = "resid_event_aegmm_"
RESIDUAL_EVENT_MARKET_PREFIX = "resid_event_market_aegmm_"
RESIDUAL_EVENT_TARGET_PREFIX = "resid_event_target_"
RESIDUAL_EVENT_TEMPORAL_SUFFIXES = (
    "posterior_speed",
    "posterior_acceleration",
    "posterior_switch_pressure",
    "posterior_entropy_delta",
    "ood_recent_max_24h",
    "ood_recent_max_48h",
    "ood_recent_max_96h",
    "reconstruction_recent_max_24h",
    "reconstruction_recent_max_48h",
    "reconstruction_recent_max_96h",
    "hours_since_ood_spike_96h_norm",
)
# These are causal trajectory summaries, not static market-state levels.  The
# source values are observable pre-entry market/OI/funding/breadth context; we
# collapse them by side x archetype x timestamp before taking differences so a
# symbol's row order cannot create a spurious transition.
RESIDUAL_EVENT_TRAJECTORY_SUFFIXES = (
    "oi_flush_impulse_1h",
    "oi_recovery_impulse_4h",
    "funding_release_impulse_4h",
    "breadth_recovery_impulse_4h",
    "short_covering_impulse_1h",
    "liquidity_stress_impulse_1h",
    "deleveraging_to_recovery_rotation_4h",
)
EVENT_CLASSES: tuple[str, ...] = (
    "normal",
    "negative_residual_event",
    "adverse_path_event",
    "positive_residual_event",
    "favorable_near_miss_event",
    "high_variance_event",
)

# These targets are deliberately narrower than the broad residual-event
# classes above.  They distinguish failure mechanisms inside the operational
# top tail, so the meta head can learn that a clean-looking direction may still
# be non-actionable because of cost, adverse path, or slow resolution.  They
# are used only to estimate train-side AE/GMM posterior priors; no realized
# field is accepted by the OOS transform.
EXECUTABLE_FAILURE_TARGETS: tuple[str, ...] = (
    "top_tail_false_positive",
    "top_tail_clean_cost_fragile",
    "top_tail_adverse_loss",
    "top_tail_adverse_false_positive",
    "top_tail_timeout_failure",
    "top_tail_timeout_loss",
    "top_tail_dirty_positive",
    "top_tail_dirty_loss",
    "top_tail_clean_executable",
    "near_tail_clean_executable",
    "near_tail_clean_cost_fragile",
    "top_tail_residual_false_positive",
    "top_tail_residual_adverse_loss",
    "top_tail_residual_timeout_loss",
    "near_tail_positive_residual_clean_executable",
    # Leaf-audit mechanisms are explicitly side x archetype scoped.  They
    # remain output labels during AE/GMM fitting and become only frozen priors
    # at OOS/inference; raw OOD/posterior columns are never used to define
    # their outcomes.
    "long_mixed_latent_misfire",
    "short_mixed_off_manifold",
    "short_default_latent_uncertainty",
    # These separate a directionally plausible call that later becomes
    # non-executable from an immediately wrong-direction call.  The leaf
    # audit shows that latent transitions, liquidity dislocation and shocks
    # explain these pathways differently, so they must not be collapsed into
    # the generic false-positive target.
    "top_tail_reversal_after_initial_success",
    "long_mixed_reversal_after_initial_success",
    "short_mixed_reversal_after_initial_success",
    "long_breakout_overconfident_path_loss",
    "short_breakout_overconfident_path_loss",
)

# Explicitly reject outcomes, decision artifacts, and previous-performance
# overlays.  Asset-relative return/OI residuals remain valid: they are known at
# entry and are not realized model residuals.
OUTCOME_COLUMNS = frozenset(
    {
        "ev_after_1pct",
        "exec_margin",
        "clean_exec",
        "dirty_positive",
        "first_touch_bad_mae_1r",
        "full_path_bad_mae_1r",
        "timeout",
        "full_stop_loss",
        "stop_or_adverse",
        "ret_net",
        "u_policy_net",
        "mfe_before_mae_1r",
        "mae_before_mfe_1r",
        "target_soft",
        "__target_soft__",
        "__first_touch_target_soft__",
        "__first_touch_policy_soft__",
    }
)
FORBIDDEN_FEATURE_TOKENS = (
    "hit_surprise",
    "ev_surprise",
    "recent_hit",
    "recent_ev",
    "archetype_hit_surprise",
    "threshold_basis",
    "historical_rank",
    "selected_for_monitor",
    "outcomes_available",
    "meta_resid_arch_",
    "meta_resid_signed_",
    "meta_resid_negative_",
    "meta_resid_positive_",
    "resid_event_",
    "resid_target_",
)


@dataclass(frozen=True)
class ResidualEventArchetypeConfig:
    timestamp_col: str = "__ts__"
    symbol_col: str = "__symbol__"
    side_col: str = "side_name"
    archetype_col: str = "archetype_policy_key"
    score_col: str = "score_meta_base_soft_label"
    probability_col: str = "hit_probability"
    hit_col: str = "clean_exec"
    ev_col: str = "ev_after_1pct"
    dirty_col: str = "dirty_positive"
    bad_mae_col: str = "full_path_bad_mae_1r"
    timeout_col: str = "timeout"
    stop_col: str = "stop_or_adverse"
    top10_fraction: float = 0.10
    top20_fraction: float = 0.20
    min_global_threshold_rows: int = 2_000
    min_local_threshold_rows: int = 600
    threshold_grid_size: int = 72
    # Local thresholds target the global top-k EV, but discovery is deliberately
    # limited to the operational score tail.  Without this floor, a locally
    # weak score scale can turn an EV-equivalent "top10" population into half
    # of an archetype's candidates.  The floor is fitted once on train scores
    # and frozen for OOS assignment; it is not a monthly/OOS percentile rule.
    local_tail_max_fraction: float = 0.20
    calibration_bins: int = 20
    calibration_shrinkage_rows: float = 180.0
    timestamp_min_peers: int = 8
    min_event_rows_per_day: int = 8
    event_z_threshold: float = 1.75
    extreme_event_z_threshold: float = 2.75
    persistence_z_threshold: float = 0.70
    min_local_state_rows: int = 1_500
    min_side_state_rows: int = 3_000
    allow_side_fallback: bool = False
    min_event_class_rows: int = 30
    min_feature_coverage: float = 0.35
    feature_scope: str = "meta_full"  # meta_full | base_directional
    # Applied only after coverage/type validation.  The default is deliberately
    # wider than the current feature store so column ordering cannot silently
    # exclude a useful family before local MI is measured.
    max_feature_candidates: int = 640
    max_features_after_mi: int = 72
    max_features_after_lgbm: int = 48
    mi_sample_rows: int = 45_000
    mi_bins: int = 8
    lgbm_enabled: bool = True
    lgbm_min_rows: int = 1_200
    lgbm_num_boost_round: int = 140
    # The encoder needs enough temporal coverage to see rare market states,
    # while the mixture can cheaply use a much larger transformed population.
    ae_gmm_max_train_rows: int = 15_000
    gmm_max_train_rows: int = 100_000
    ae_gmm_max_iter: int = 160
    ae_gmm_clusters: tuple[int, ...] = (3, 4, 5, 6)
    ae_gmm_reg_covars: tuple[float, ...] = (1e-4, 1e-3, 3e-3)
    ae_gmm_covariance_types: tuple[str, ...] = ("diag", "tied", "full")
    ae_gmm_smooth_lambdas: tuple[float, ...] = (0.0,)
    prior_shrinkage_rows: float = 120.0
    # Local side x archetype states are the primary discovery object. The
    # timestamp-level market block is intentionally opt-in and evaluated as a
    # separate ablation after local-state evidence is established.
    enable_market_secondary: bool = False
    # These two blocks are evaluated as explicit ablations.  They never alter
    # the base/meta label and only emit frozen train-side state priors at OOS.
    enable_executable_quality_targets: bool = True
    enable_transition_trajectory_features: bool = True
    market_max_features: int = 48
    market_min_rows: int = 1_000
    random_state: int = 20260712


def _numeric(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float32")
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _binary(frame: pd.DataFrame, col: str) -> pd.Series:
    return _numeric(frame, col, 0.0).fillna(0.0).gt(0.5)


def _local_key(side: object, archetype: object) -> str:
    return f"{str(side).strip().lower()}|{str(archetype).strip() or 'missing'}"


def _side_arch(
    frame: pd.DataFrame, config: ResidualEventArchetypeConfig
) -> tuple[pd.Series, pd.Series, pd.Series]:
    side = (
        frame.get(config.side_col, pd.Series("missing", index=frame.index))
        .astype(str)
        .str.lower()
    )
    archetype = frame.get(
        config.archetype_col, pd.Series("missing", index=frame.index)
    ).astype(str)
    archetype = archetype.replace({"": "missing", "nan": "missing", "None": "missing"})
    key = pd.Series(
        [_local_key(s, a) for s, a in zip(side, archetype, strict=True)],
        index=frame.index,
        dtype="object",
    )
    return side, archetype, key


def _time_spread_indices(n: int, cap: int) -> np.ndarray:
    if n <= int(cap):
        return np.arange(n, dtype=np.int64)
    thirds = ((0, n // 3), (n // 3, (2 * n) // 3), ((2 * n) // 3, n))
    per = max(1, int(cap) // 3)
    chunks = [
        np.linspace(start, stop - 1, min(per, stop - start), dtype=np.int64)
        for start, stop in thirds
        if stop > start
    ]
    return np.unique(np.concatenate(chunks))[: int(cap)].astype(np.int64, copy=False)


def _index_positions(index: pd.Index, labels: Iterable[object]) -> np.ndarray:
    """Return positional offsets for group labels without assuming RangeIndex."""

    positions = index.get_indexer(pd.Index(list(labels)))
    if (positions < 0).any():
        raise KeyError("group labels are not present in the source frame index")
    return positions.astype(np.int64, copy=False)


def _safe_quantiles(values: np.ndarray, q: np.ndarray) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.zeros(len(q), dtype=np.float32)
    edges = np.quantile(finite, q)
    # Strictly increasing bins make searchsorted stable even for flat scores.
    return np.maximum.accumulate(
        edges + np.arange(len(edges), dtype=np.float64) * 1e-8
    ).astype(np.float32)


@dataclass
class GlobalEVThresholdState:
    """Train-fitted global EV target with local score cutoffs.

    The target is the global score top-k mean EV.  Each side x archetype gets
    the least restrictive score threshold that reaches that global target in
    train data.  It therefore does not create easier monthly/local percentile
    tails merely because a stream is sparse.
    """

    config: ResidualEventArchetypeConfig
    global_thresholds: dict[str, float] = field(default_factory=dict)
    global_targets: dict[str, float] = field(default_factory=dict)
    local_thresholds: dict[str, dict[str, float]] = field(default_factory=dict)
    score_reference: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )
    train_rows: int = 0

    def _target_name(self, fraction: float) -> str:
        return f"top{int(round(100.0 * float(fraction)))}"

    @staticmethod
    def _threshold_for_target(
        score: np.ndarray,
        ev: np.ndarray,
        *,
        target_ev: float,
        min_rows: int,
        grid_size: int,
        local_tail_max_fraction: float,
    ) -> float:
        valid = np.isfinite(score) & np.isfinite(ev)
        if int(valid.sum()) < int(min_rows) or not np.isfinite(target_ev):
            return float("nan")
        s = score[valid].astype(np.float64, copy=False)
        y = ev[valid].astype(np.float64, copy=False)
        floor_quantile = float(
            np.clip(1.0 - float(local_tail_max_fraction), 0.0, 0.995)
        )
        grid = np.unique(
            np.quantile(s, np.linspace(floor_quantile, 0.995, int(grid_size)))
        )
        best_threshold = float("nan")
        best_gap = float("inf")
        for threshold in grid:
            chosen = y[s >= float(threshold)]
            if chosen.size < int(min_rows):
                continue
            mean_ev = float(np.mean(chosen))
            gap = abs(mean_ev - float(target_ev))
            if mean_ev >= float(target_ev) and gap < best_gap:
                best_gap = gap
                best_threshold = float(threshold)
        return best_threshold

    def fit(self, train: pd.DataFrame) -> "GlobalEVThresholdState":
        score = _numeric(train, self.config.score_col).to_numpy(dtype=np.float32)
        ev = _numeric(train, self.config.ev_col).to_numpy(dtype=np.float32)
        valid = np.isfinite(score) & np.isfinite(ev)
        if int(valid.sum()) < int(self.config.min_global_threshold_rows):
            raise ValueError("insufficient finite train rows for global EV thresholds")
        valid_score = score[valid]
        valid_ev = ev[valid]
        self.train_rows = int(valid.sum())
        self.score_reference = valid_score[
            _time_spread_indices(len(valid_score), 200_000)
        ].astype(np.float32, copy=True)
        side, arch, key = _side_arch(train, self.config)
        self.global_thresholds = {}
        self.global_targets = {}
        self.local_thresholds = {}
        for fraction in (self.config.top10_fraction, self.config.top20_fraction):
            name = self._target_name(fraction)
            global_threshold = float(np.quantile(valid_score, 1.0 - float(fraction)))
            global_target = float(np.mean(valid_ev[valid_score >= global_threshold]))
            self.global_thresholds[name] = global_threshold
            self.global_targets[name] = global_target
        work = pd.DataFrame(
            {
                "key": key.to_numpy(copy=False),
                "score": score,
                "ev": ev,
            },
            index=train.index,
        )
        for local, group in work.groupby("key", observed=True, sort=True):
            g_score = group["score"].to_numpy(dtype=np.float32)
            g_ev = group["ev"].to_numpy(dtype=np.float32)
            valid_local = np.isfinite(g_score) & np.isfinite(g_ev)
            support = int(valid_local.sum())
            payload: dict[str, float] = {"support": float(support)}
            if support:
                # This train-derived score floor enforces the requested
                # top10-to-top20 discovery scope even when the global EV
                # target would otherwise choose a much looser local cutoff.
                local_floor = float(
                    np.quantile(
                        g_score[valid_local],
                        1.0 - float(self.config.local_tail_max_fraction),
                    )
                )
            else:
                local_floor = float("nan")
            for fraction in (self.config.top10_fraction, self.config.top20_fraction):
                name = self._target_name(fraction)
                fallback = float(self.global_thresholds[name])
                threshold = self._threshold_for_target(
                    g_score,
                    g_ev,
                    target_ev=float(self.global_targets[name]),
                    min_rows=max(30, int(self.config.min_local_threshold_rows)),
                    grid_size=int(self.config.threshold_grid_size),
                    local_tail_max_fraction=float(self.config.local_tail_max_fraction),
                )
                raw_threshold = float(threshold) if np.isfinite(threshold) else fallback
                payload[name] = float(max(raw_threshold, local_floor))
                payload[f"{name}_local_score_floor"] = float(local_floor)
                payload[f"{name}_source"] = 1.0 if np.isfinite(threshold) else 0.0
            # The broader band must contain the local top10 population.
            payload["top20"] = min(float(payload["top20"]), float(payload["top10"]))
            top20_support = int(
                np.sum(valid_local & (g_score >= float(payload["top20"])))
            )
            # A high EV-equivalent threshold can starve a locally important
            # archetype even when it has ample observations in its own top20.
            # Fall back only to that train-derived local top20 floor; never
            # broaden residual-state discovery beyond the requested band.
            if (
                np.isfinite(local_floor)
                and top20_support < int(self.config.min_local_state_rows)
                and support >= int(self.config.min_local_state_rows)
            ):
                payload["top20"] = min(float(local_floor), float(payload["top10"]))
                payload["top20_source"] = 2.0
                top20_support = int(
                    np.sum(valid_local & (g_score >= float(payload["top20"])))
                )
            payload["top20_support"] = float(top20_support)
            self.local_thresholds[str(local)] = payload
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.global_thresholds:
            raise RuntimeError("GlobalEVThresholdState is not fitted")
        score = _numeric(frame, self.config.score_col).to_numpy(dtype=np.float32)
        _side, _arch, key = _side_arch(frame, self.config)
        top10 = np.full(
            len(frame), float(self.global_thresholds["top10"]), dtype=np.float32
        )
        top20 = np.full(
            len(frame), float(self.global_thresholds["top20"]), dtype=np.float32
        )
        source = np.zeros(len(frame), dtype=np.int8)
        for local, positions in key.groupby(key, sort=False).groups.items():
            payload = self.local_thresholds.get(str(local))
            if payload is None:
                continue
            pos = _index_positions(frame.index, positions)
            top10[pos] = np.float32(payload["top10"])
            top20[pos] = np.float32(payload["top20"])
            source[pos] = np.int8(payload.get("top10_source", 0.0) > 0.5)
        ref = np.sort(self.score_reference)
        if len(ref):
            rank = np.searchsorted(ref, score, side="right") / float(len(ref))
        else:
            rank = np.full(len(frame), np.nan, dtype=np.float32)
        out = pd.DataFrame(index=frame.index)
        out["resid_event_global_score_rank_pct"] = np.asarray(rank, dtype=np.float32)
        out["resid_event_top10_threshold"] = top10
        out["resid_event_top20_threshold"] = top20
        out["resid_event_top10_population"] = (score >= top10).astype(np.int8)
        out["resid_event_top20_population"] = (score >= top20).astype(np.int8)
        out["resid_event_near_miss_population"] = (
            (score >= top20) & (score < top10)
        ).astype(np.int8)
        out["resid_event_local_threshold_available"] = source
        return out

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "residual_event_global_ev_threshold_v1",
            "score_col": self.config.score_col,
            "global_thresholds": self.global_thresholds,
            "global_targets": self.global_targets,
            "local_threshold_count": int(len(self.local_thresholds)),
            "train_rows": int(self.train_rows),
            "contract": (
                "global score top-k EV target; side x archetype score threshold "
                "fitted on train only and floored at its train top20 score cutoff"
            ),
        }


@dataclass
class ScoreExpectationState:
    """Frozen hierarchical score expectation for one realized outcome."""

    config: ResidualEventArchetypeConfig
    target_col: str | None = None
    direct_col: str | None = None
    edges: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    global_rates: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )
    local_rates: dict[str, np.ndarray] = field(default_factory=dict)
    local_counts: dict[str, np.ndarray] = field(default_factory=dict)

    def fit(self, train: pd.DataFrame) -> "ScoreExpectationState":
        score = _numeric(train, self.config.score_col).to_numpy(dtype=np.float32)
        target_col = self.target_col or self.config.hit_col
        value = _numeric(train, target_col).to_numpy(dtype=np.float32)
        valid = np.isfinite(score) & np.isfinite(value)
        if int(valid.sum()) < int(self.config.min_global_threshold_rows):
            raise ValueError("insufficient train rows for residual expectation state")
        bins = max(4, int(self.config.calibration_bins))
        self.edges = _safe_quantiles(score[valid], np.linspace(0.0, 1.0, bins + 1))
        bucket = np.clip(
            np.searchsorted(self.edges[1:-1], score, side="right"), 0, bins - 1
        )
        global_sum = np.bincount(
            bucket[valid], weights=value[valid], minlength=bins
        ).astype(np.float64)
        global_count = np.bincount(bucket[valid], minlength=bins).astype(np.float64)
        global_mean = float(np.mean(value[valid]))
        self.global_rates = (
            (global_sum + 20.0 * global_mean) / np.maximum(global_count + 20.0, 1.0)
        ).astype(np.float32)
        self.local_rates = {}
        self.local_counts = {}
        _side, _arch, key = _side_arch(train, self.config)
        for local, idx in key.groupby(key, sort=True).groups.items():
            pos = _index_positions(train.index, idx)
            mask = valid[pos]
            if int(mask.sum()) < max(
                60, int(self.config.min_local_threshold_rows) // 4
            ):
                continue
            local_bucket = bucket[pos][mask]
            local_value = value[pos][mask]
            sums = np.bincount(local_bucket, weights=local_value, minlength=bins).astype(
                np.float64
            )
            counts = np.bincount(local_bucket, minlength=bins).astype(np.float64)
            local_mean = float(np.mean(local_value))
            rates = (sums + 12.0 * local_mean) / np.maximum(counts + 12.0, 1.0)
            self.local_rates[str(local)] = rates.astype(np.float32)
            self.local_counts[str(local)] = counts.astype(np.float32)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.Series:
        if self.edges.size == 0 or self.global_rates.size == 0:
            raise RuntimeError("ScoreExpectationState is not fitted")
        # The hit expectation may use the frozen model's own probability. EV
        # expectation has no direct realized-value proxy, so it always uses the
        # train-fitted score calibration.
        direct_name = self.direct_col
        if direct_name is None and (self.target_col or self.config.hit_col) == self.config.hit_col:
            direct_name = self.config.probability_col
        direct = _numeric(frame, direct_name) if direct_name else pd.Series(
            np.nan, index=frame.index, dtype=np.float32
        )
        score = _numeric(frame, self.config.score_col).to_numpy(dtype=np.float32)
        bins = len(self.global_rates)
        bucket = np.clip(
            np.searchsorted(self.edges[1:-1], score, side="right"), 0, bins - 1
        )
        expected = self.global_rates[bucket].astype(np.float32, copy=True)
        _side, _arch, key = _side_arch(frame, self.config)
        shrink = float(self.config.calibration_shrinkage_rows)
        for local, idx in key.groupby(key, sort=False).groups.items():
            rates = self.local_rates.get(str(local))
            counts = self.local_counts.get(str(local))
            if rates is None or counts is None:
                continue
            pos = _index_positions(frame.index, idx)
            b = bucket[pos]
            w = counts[b] / np.maximum(counts[b] + shrink, 1e-6)
            expected[pos] = (w * rates[b] + (1.0 - w) * self.global_rates[b]).astype(
                np.float32
            )
        expected_series = pd.Series(expected, index=frame.index, dtype="float32")
        finite_direct = direct.notna()
        expected_series.loc[finite_direct] = direct.loc[finite_direct].astype(
            np.float32
        )
        return expected_series


@dataclass
class ResidualEventBaselineState:
    """Train-only daily residual baseline for large persistent local events."""

    config: ResidualEventArchetypeConfig
    stats: dict[str, dict[str, float]] = field(default_factory=dict)
    global_stats: dict[str, float] = field(default_factory=dict)
    ev_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    global_ev_stats: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def _aggregate_daily(
        frame: pd.DataFrame, config: ResidualEventArchetypeConfig
    ) -> pd.DataFrame:
        work = frame.copy(deep=False)
        ts = pd.to_datetime(work[config.timestamp_col], utc=True, errors="coerce")
        work["__resid_event_day__"] = ts.dt.floor("D")
        _side, _arch, key = _side_arch(work, config)
        work["__resid_event_key__"] = key.to_numpy(copy=False)
        for col in (
            "resid_event_timestamp_neutral_surprise",
            "resid_event_ev_timestamp_neutral_surprise",
            "resid_event_global_surprise",
        ):
            if col not in work.columns:
                work[col] = np.float32(0.0)
        top10 = _numeric(work, "resid_event_top10_population", 0.0).fillna(0.0).gt(0.5)
        near = (
            _numeric(work, "resid_event_near_miss_population", 0.0).fillna(0.0).gt(0.5)
        )
        frames: list[pd.DataFrame] = []
        for zone, mask in (("top10", top10), ("near_miss", near)):
            part = work.loc[mask & work["__resid_event_day__"].notna()]
            if part.empty:
                continue
            summary = (
                part.groupby(
                    ["__resid_event_key__", "__resid_event_day__"],
                    observed=True,
                    sort=True,
                )
                .agg(
                    rows=("__resid_event_key__", "size"),
                    mean_neutral=("resid_event_timestamp_neutral_surprise", "mean"),
                    mean_ev_neutral=(
                        "resid_event_ev_timestamp_neutral_surprise",
                        "mean",
                    ),
                    mean_global=("resid_event_global_surprise", "mean"),
                    mean_ev=(config.ev_col, "mean"),
                    bad_mae=(config.bad_mae_col, "mean"),
                    timeout=(config.timeout_col, "mean"),
                    dirty=(config.dirty_col, "mean"),
                )
                .reset_index()
            )
            support = summary["rows"].to_numpy(dtype=np.float32)
            support_weight = support / np.maximum(
                support + float(config.min_event_rows_per_day), 1.0
            )
            # Shrink sparse daily surprises rather than dropping rare local
            # streams.  A very large event can still qualify, but one or two
            # noisy rows cannot dominate the state target.
            summary["mean_neutral"] = (
                pd.to_numeric(summary["mean_neutral"], errors="coerce").to_numpy(
                    dtype=np.float32
                )
                * support_weight
            )
            summary["mean_ev_neutral"] = (
                pd.to_numeric(summary["mean_ev_neutral"], errors="coerce").to_numpy(
                    dtype=np.float32
                )
                * support_weight
            )
            summary["support_weight"] = support_weight.astype(np.float32)
            summary["zone"] = zone
            frames.append(summary)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def fit(self, labelled_train: pd.DataFrame) -> "ResidualEventBaselineState":
        daily = self._aggregate_daily(labelled_train, self.config)
        if daily.empty:
            self.stats = {}
            self.global_stats = {"mean": 0.0, "std": 1.0}
            self.ev_stats = {}
            self.global_ev_stats = {"mean": 0.0, "std": 1.0}
            return self

        def _fit_metric(column: str) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
            local: dict[str, dict[str, float]] = {}
            for (key, zone), group in daily.groupby(
                ["__resid_event_key__", "zone"], observed=True, sort=True
            ):
                values = pd.to_numeric(group[column], errors="coerce").to_numpy(
                    dtype=np.float64
                )
                finite = values[np.isfinite(values)]
                if finite.size < 5:
                    continue
                median = float(np.median(finite))
                q25, q75 = np.quantile(finite, [0.25, 0.75])
                mad = float(np.median(np.abs(finite - median)))
                local[f"{key}|{zone}"] = {
                    "mean": median,
                    "std": max(
                        float((q75 - q25) / 1.349),
                        float(1.4826 * mad),
                        float(np.std(finite) * 0.25),
                        1e-4,
                    ),
                    "rows": float(finite.size),
                }
            all_values = pd.to_numeric(daily[column], errors="coerce").to_numpy(
                dtype=np.float64
            )
            all_values = all_values[np.isfinite(all_values)]
            if not all_values.size:
                return local, {"mean": 0.0, "std": 1.0}
            median = float(np.median(all_values))
            q25, q75 = np.quantile(all_values, [0.25, 0.75])
            mad = float(np.median(np.abs(all_values - median)))
            return local, {
                "mean": median,
                "std": max(
                    float((q75 - q25) / 1.349),
                    float(1.4826 * mad),
                    float(np.std(all_values) * 0.25),
                    1e-4,
                ),
            }

        self.stats, self.global_stats = _fit_metric("mean_neutral")
        self.ev_stats, self.global_ev_stats = _fit_metric("mean_ev_neutral")
        return self

    def annotate(self, labelled: pd.DataFrame) -> pd.DataFrame:
        out = labelled.copy(deep=False)
        daily = self._aggregate_daily(out, self.config)
        if daily.empty:
            for col in (
                "resid_event_daily_neutral_z",
                "resid_event_daily_ev_neutral_z",
                "resid_event_daily_mean_ev",
                "resid_event_negative_large",
                "resid_event_positive_large",
                "resid_event_persistent",
                "resid_event_persistence_strength",
                "resid_event_large_event_strength",
            ):
                out[col] = 0.0
            out["resid_event_class"] = "normal"
            return out
        z = np.zeros(len(daily), dtype=np.float32)
        ev_z = np.zeros(len(daily), dtype=np.float32)
        for idx, (_, row) in enumerate(daily.iterrows()):
            key = f"{row['__resid_event_key__']}|{row['zone']}"
            stat = self.stats.get(key, self.global_stats)
            ev_stat = self.ev_stats.get(key, self.global_ev_stats)
            z[idx] = np.float32(
                (float(row["mean_neutral"]) - float(stat["mean"]))
                / max(float(stat["std"]), 1e-4)
            )
            ev_z[idx] = np.float32(
                (float(row["mean_ev_neutral"]) - float(ev_stat["mean"]))
                / max(float(ev_stat["std"]), 1e-4)
            )
        daily["resid_event_daily_neutral_z"] = z
        daily["resid_event_daily_ev_neutral_z"] = ev_z
        daily = daily.sort_values(
            ["__resid_event_key__", "zone", "__resid_event_day__"], kind="stable"
        )
        # Prefer the component with the larger standardized surprise for
        # persistence. This catches path/cost failures where clean hit-rate is
        # positive but executable EV is persistently negative.
        daily["__event_z__"] = np.where(
            daily["resid_event_daily_ev_neutral_z"].abs()
            > daily["resid_event_daily_neutral_z"].abs(),
            daily["resid_event_daily_ev_neutral_z"],
            daily["resid_event_daily_neutral_z"],
        ).astype(np.float32)
        daily["__previous_z__"] = daily.groupby(
            ["__resid_event_key__", "zone"], observed=True
        )["__event_z__"].shift(1)
        daily["__previous_day__"] = daily.groupby(
            ["__resid_event_key__", "zone"], observed=True
        )["__resid_event_day__"].shift(1)
        contiguous = (
            daily["__resid_event_day__"] - daily["__previous_day__"]
        ).dt.days.eq(1)
        neg = (
            daily["resid_event_daily_neutral_z"].le(
                -float(self.config.event_z_threshold)
            )
            | daily["resid_event_daily_ev_neutral_z"].le(
                -float(self.config.event_z_threshold)
            )
        )
        pos = (
            daily["resid_event_daily_neutral_z"].ge(
                float(self.config.event_z_threshold)
            )
            & daily["resid_event_daily_ev_neutral_z"].ge(0.0)
        ) | (
            daily["resid_event_daily_ev_neutral_z"].ge(
                float(self.config.event_z_threshold)
            )
            & daily["resid_event_daily_neutral_z"].ge(0.0)
        )
        persistent = contiguous & (
            (
                neg
                & daily["__previous_z__"].le(
                    -float(self.config.persistence_z_threshold)
                )
            )
            | (
                pos
                & daily["__previous_z__"].ge(float(self.config.persistence_z_threshold))
            )
        )
        # Two aligned days are enough when the current surprise is material.
        # A single exceptionally large day is retained as an acute event so
        # rare failure states are not erased merely because no similar day was
        # observed immediately before it.
        extreme = daily["__event_z__"].abs().ge(
            float(self.config.extreme_event_z_threshold)
        )
        persistent = persistent | extreme
        previous = pd.to_numeric(daily["__previous_z__"], errors="coerce").fillna(0.0)
        current = pd.to_numeric(daily["__event_z__"], errors="coerce").fillna(0.0)
        same_sign = np.sign(current).eq(np.sign(previous)) & contiguous
        daily["resid_event_persistence_strength"] = np.where(
            same_sign,
            np.abs(current * previous),
            0.0,
        ).astype(np.float32)
        daily["resid_event_large_event_strength"] = np.maximum(
            np.abs(current) - float(self.config.event_z_threshold), 0.0
        ).astype(np.float32)
        daily["resid_event_negative_large"] = neg.astype(np.int8)
        daily["resid_event_positive_large"] = pos.astype(np.int8)
        daily["resid_event_persistent"] = persistent.astype(np.int8)
        ts = pd.to_datetime(out[self.config.timestamp_col], utc=True, errors="coerce")
        _side, _arch, key = _side_arch(out, self.config)
        out = out.copy(deep=False)
        out["__resid_event_key__"] = key.to_numpy(copy=False)
        out["__resid_event_day__"] = ts.dt.floor("D")
        zone = np.where(
            _numeric(out, "resid_event_top10_population", 0.0).fillna(0.0).to_numpy()
            > 0.5,
            "top10",
            np.where(
                _numeric(out, "resid_event_near_miss_population", 0.0)
                .fillna(0.0)
                .to_numpy()
                > 0.5,
                "near_miss",
                "outside",
            ),
        )
        out["__resid_event_zone__"] = zone
        merge_cols = ["__resid_event_key__", "__resid_event_day__", "zone"]
        attach = daily.loc[
            :,
            [
                *merge_cols,
                "resid_event_daily_neutral_z",
                "resid_event_daily_ev_neutral_z",
                "mean_ev",
                "resid_event_negative_large",
                "resid_event_positive_large",
                "resid_event_persistent",
                "resid_event_persistence_strength",
                "resid_event_large_event_strength",
            ],
        ]
        out = out.merge(
            attach,
            left_on=[
                "__resid_event_key__",
                "__resid_event_day__",
                "__resid_event_zone__",
            ],
            right_on=merge_cols,
            how="left",
            sort=False,
        ).set_axis(labelled.index, axis=0, copy=False)
        for name in (
            "resid_event_daily_neutral_z",
            "resid_event_daily_ev_neutral_z",
            "resid_event_negative_large",
            "resid_event_positive_large",
            "resid_event_persistent",
            "resid_event_persistence_strength",
            "resid_event_large_event_strength",
        ):
            out[name] = _numeric(out, name, 0.0).fillna(0.0).astype(np.float32)
        out["resid_event_daily_mean_ev"] = _numeric(
            out, "mean_ev", 0.0
        ).fillna(0.0).astype(np.float32)
        top10 = out["__resid_event_zone__"].eq("top10")
        near = out["__resid_event_zone__"].eq("near_miss")
        negative = out["resid_event_negative_large"].gt(0.5) & out[
            "resid_event_persistent"
        ].gt(0.5)
        positive = out["resid_event_positive_large"].gt(0.5) & out[
            "resid_event_persistent"
        ].gt(0.5) & out["resid_event_daily_mean_ev"].gt(0.0)
        adverse_path = (
            top10
            & negative
            & (
                _binary(out, self.config.bad_mae_col)
                | _binary(out, self.config.timeout_col)
                | _binary(out, self.config.stop_col)
                | _binary(out, self.config.dirty_col)
            )
        )
        cls = np.full(len(out), "normal", dtype=object)
        cls[top10 & negative] = "negative_residual_event"
        cls[adverse_path.to_numpy()] = "adverse_path_event"
        cls[top10 & positive] = "positive_residual_event"
        cls[near & positive] = "favorable_near_miss_event"
        high_var = (
            (top10 | near)
            & (
                out["resid_event_daily_neutral_z"]
                .abs()
                .ge(float(self.config.event_z_threshold))
                | out["resid_event_daily_ev_neutral_z"]
                .abs()
                .ge(float(self.config.event_z_threshold))
            )
            & ~negative
            & ~positive
        )
        cls[high_var.to_numpy()] = "high_variance_event"
        out["resid_event_class"] = pd.Categorical(cls, categories=EVENT_CLASSES)
        return out.drop(columns=["zone", "mean_ev"], errors="ignore")


def add_residual_event_targets(
    frame: pd.DataFrame,
    *,
    threshold_state: GlobalEVThresholdState,
    expectation_state: ScoreExpectationState,
    ev_expectation_state: ScoreExpectationState | None = None,
    baseline_state: ResidualEventBaselineState | None = None,
) -> pd.DataFrame:
    """Attach outcome-only discovery labels; never call this at inference."""

    config = threshold_state.config
    missing = [
        name for name in (config.hit_col, config.ev_col) if name not in frame.columns
    ]
    if missing:
        raise ValueError(f"residual-event labels require realized columns: {missing}")
    out = frame.copy(deep=False)
    threshold_features = threshold_state.transform(out)
    for name in threshold_features.columns:
        out[name] = threshold_features[name].to_numpy(copy=False)
    expected = expectation_state.transform(out)
    hit = _numeric(out, config.hit_col, 0.0).fillna(0.0).clip(0.0, 1.0)
    global_surprise = (hit - expected).astype(np.float32)
    expected_ev = (
        ev_expectation_state.transform(out)
        if ev_expectation_state is not None
        else pd.Series(0.0, index=out.index, dtype=np.float32)
    )
    actual_ev = _numeric(out, config.ev_col, 0.0).fillna(0.0)
    ev_global_surprise = (actual_ev - expected_ev).astype(np.float32)
    ts = pd.to_datetime(out[config.timestamp_col], utc=True, errors="coerce")
    valid_ts = ts.notna()
    tmp = pd.DataFrame(
        {"ts": ts, "residual": global_surprise, "valid": valid_ts}, index=out.index
    )
    sums = tmp.loc[tmp["valid"]].groupby("ts", sort=False)["residual"].transform("sum")
    counts = (
        tmp.loc[tmp["valid"]].groupby("ts", sort=False)["residual"].transform("count")
    )
    peer = pd.Series(0.0, index=out.index, dtype="float32")
    valid_peer = counts.gt(1)
    peer.loc[valid_peer.index[valid_peer]] = (
        (sums.loc[valid_peer] - global_surprise.loc[valid_peer])
        / (counts.loc[valid_peer] - 1.0)
    ).astype(np.float32)
    neutral = (global_surprise - peer).astype(np.float32)
    # Tiny timestamp batches cannot define market-neutral surprise. Keep the
    # global residual in that case rather than inventing a peer correction.
    small = pd.Series(True, index=out.index)
    small.loc[counts.index] = counts.lt(int(config.timestamp_min_peers))
    neutral.loc[small] = global_surprise.loc[small]
    ev_tmp = pd.DataFrame(
        {"ts": ts, "residual": ev_global_surprise, "valid": valid_ts},
        index=out.index,
    )
    ev_sums = ev_tmp.loc[ev_tmp["valid"]].groupby("ts", sort=False)[
        "residual"
    ].transform("sum")
    ev_counts = ev_tmp.loc[ev_tmp["valid"]].groupby("ts", sort=False)[
        "residual"
    ].transform("count")
    ev_peer = pd.Series(0.0, index=out.index, dtype="float32")
    valid_ev_peer = ev_counts.gt(1)
    ev_peer.loc[valid_ev_peer.index[valid_ev_peer]] = (
        (ev_sums.loc[valid_ev_peer] - ev_global_surprise.loc[valid_ev_peer])
        / (ev_counts.loc[valid_ev_peer] - 1.0)
    ).astype(np.float32)
    ev_neutral = (ev_global_surprise - ev_peer).astype(np.float32)
    ev_small = pd.Series(True, index=out.index)
    ev_small.loc[ev_counts.index] = ev_counts.lt(int(config.timestamp_min_peers))
    ev_neutral.loc[ev_small] = ev_global_surprise.loc[ev_small]
    out["resid_event_expected_hit"] = expected
    out["resid_event_expected_ev"] = expected_ev.astype(np.float32)
    out["resid_event_global_surprise"] = global_surprise
    out["resid_event_ev_global_surprise"] = ev_global_surprise
    # The peer component is the broader market/model-stream problem at the
    # same timestamp.  Local discovery uses the neutral residual first; this
    # market component is retained as a distinct secondary training target so
    # sequential archetype failures can share a common explanation without
    # contaminating the local label.
    out["resid_event_market_peer_surprise"] = peer
    out["resid_event_timestamp_neutral_surprise"] = neutral
    out["resid_event_ev_market_peer_surprise"] = ev_peer
    out["resid_event_ev_timestamp_neutral_surprise"] = ev_neutral
    out["resid_event_timestamp_peer_count"] = (
        counts.reindex(out.index).fillna(1).astype(np.int16)
    )
    if baseline_state is None:
        return out
    return baseline_state.annotate(out)


def causal_eight_day_hit_rate_overlay(
    frame: pd.DataFrame,
    *,
    config: ResidualEventArchetypeConfig,
    selected_col: str = "resid_event_top10_population",
    half_life_days: float = 8.0,
    embargo_hours: float = 12.0,
) -> pd.DataFrame:
    """Assessment-only causal hit-rate smoother, excluding current outcomes.

    It is intentionally not used by state fitting, feature screening, or OOS
    transform.  It mirrors the *type* of information in the promoted 8-day
    policy without importing its threshold/calibration decisions.
    """

    required = {config.timestamp_col, config.hit_col}
    if not required.issubset(frame.columns):
        return pd.DataFrame(index=frame.index)
    work = frame.copy(deep=False)
    ts = pd.to_datetime(work[config.timestamp_col], utc=True, errors="coerce")
    side, arch, key = _side_arch(work, config)
    work = pd.DataFrame(
        {
            "ts": ts,
            "key": key,
            "selected": _numeric(frame, selected_col, 0.0).fillna(0.0).gt(0.5),
            "hit": _numeric(frame, config.hit_col, np.nan),
        },
        index=frame.index,
    ).sort_values(["ts", "key"], kind="stable")
    result = pd.DataFrame(
        index=work.index,
        data={"assessment_hr8_surprise": np.nan, "assessment_hr8_effective_n": 0.0},
    )
    decay = math.log(2.0) / max(float(half_life_days), 1e-6)
    for local, group in work.groupby("key", sort=False):
        history_ts: list[pd.Timestamp] = []
        history_hit: list[float] = []
        for day, current in group.groupby(group["ts"].dt.floor("D"), sort=True):
            cutoff = pd.Timestamp(day)
            if cutoff.tzinfo is None:
                cutoff = cutoff.tz_localize("UTC")
            else:
                cutoff = cutoff.tz_convert("UTC")
            cutoff -= pd.Timedelta(hours=float(embargo_hours))
            while history_ts and history_ts[0] < cutoff - pd.Timedelta(days=60):
                history_ts.pop(0)
                history_hit.pop(0)
            if history_ts:
                ages = np.array(
                    [
                        (cutoff - value).total_seconds() / 86_400.0
                        for value in history_ts
                    ],
                    dtype=np.float64,
                )
                weights = np.exp(-decay * np.maximum(ages, 0.0))
                expected = float(
                    np.dot(weights, np.asarray(history_hit)) / max(weights.sum(), 1e-8)
                )
                effective = float(
                    weights.sum() ** 2 / max(np.dot(weights, weights), 1e-8)
                )
                result.loc[current.index, "assessment_hr8_surprise"] = (
                    _numeric(frame.loc[current.index], config.hit_col, np.nan)
                    - expected
                ).to_numpy(dtype=np.float32)
                result.loc[current.index, "assessment_hr8_effective_n"] = np.float32(
                    effective
                )
            resolved = current.loc[current["selected"] & current["hit"].notna()]
            history_ts.extend(resolved["ts"].tolist())
            history_hit.extend(resolved["hit"].astype(float).tolist())
    return result.reindex(frame.index).astype(np.float32)


def _is_usable_feature(
    name: str, frame: pd.DataFrame, config: ResidualEventArchetypeConfig
) -> bool:
    if name not in frame.columns or name in OUTCOME_COLUMNS:
        return False
    lower = str(name).lower()
    if lower.startswith("__") or any(
        token in lower for token in FORBIDDEN_FEATURE_TOKENS
    ):
        return False
    if not (
        pd.api.types.is_numeric_dtype(frame[name])
        or pd.api.types.is_bool_dtype(frame[name])
    ):
        return False
    if (
        str(config.feature_scope) == "base_directional"
        and str(name) not in BASE_DIRECTIONAL_STATE_FEATURES
    ):
        return False
    values = _numeric(frame, str(name))
    if float(values.notna().mean()) < float(config.min_feature_coverage):
        return False
    return int(values.nunique(dropna=True)) >= 8


def _is_market_feature(name: str) -> bool:
    lower = str(name).lower()
    if (
        lower.startswith(("asset_", "symbol_"))
        or (
            lower.startswith("xs_")
            and not lower.startswith(("xs_mean__", "xs_median__", "xs_std__"))
        )
        or any(
            token in lower
            for token in (
                "asset_minus_mkt",
                "symbol_minus_mkt",
                "peer_resid",
                "ts_resid",
            )
        )
    ):
        return False
    return lower.startswith(
        (
            "mkt_",
            "market_",
            "pct_assets_",
            "cross_asset_",
            "crossasset_",
            "state_spectral_",
            "xs_mean__",
            "xs_median__",
            "xs_std__",
        )
    ) or any(
        token in lower
        for token in (
            "breadth_",
            "cross_section",
            "downside_corr",
            "pc1_variance",
            "return_dispersion",
            "liquidation",
        )
    )


def inference_feature_basket(
    frame: pd.DataFrame,
    candidates: Iterable[str] | None,
    config: ResidualEventArchetypeConfig,
) -> list[str]:
    source = list(candidates) if candidates is not None else list(frame.columns)
    usable = [
        str(name)
        for name in dict.fromkeys(source)
        if _is_usable_feature(str(name), frame, config)
    ]
    cap = int(config.max_feature_candidates)
    return usable if cap <= 0 else usable[:cap]


def _binned_mi(values: np.ndarray, target: np.ndarray, bins: int) -> float:
    valid = np.isfinite(values) & np.isfinite(target)
    if int(valid.sum()) < 80 or np.unique(target[valid]).size < 2:
        return 0.0
    x = values[valid].astype(np.float64, copy=False)
    y = target[valid].astype(np.int8, copy=False)
    edges = _safe_quantiles(x, np.linspace(0.0, 1.0, max(3, int(bins)) + 1))
    bucket = np.clip(np.searchsorted(edges[1:-1], x, side="right"), 0, len(edges) - 2)
    joint = np.zeros((int(bucket.max()) + 1, 2), dtype=np.float64)
    np.add.at(joint, (bucket, y), 1.0)
    pxy = joint / max(float(joint.sum()), 1.0)
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    nonzero = pxy > 0.0
    return float(
        np.sum(
            pxy[nonzero] * np.log(pxy[nonzero] / np.maximum((px * py)[nonzero], 1e-12))
        )
    )


def _matrix(
    frame: pd.DataFrame, features: Sequence[str], medians: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    x = (
        frame.reindex(columns=list(features))
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32, copy=False)
    )
    x[~np.isfinite(x)] = np.nan
    if medians is None:
        medians = np.nanmedian(x, axis=0).astype(np.float32)
        medians = np.nan_to_num(medians, nan=0.0)
    missing = ~np.isfinite(x)
    if missing.any():
        x = x.copy()
        x[missing] = np.take(medians, np.nonzero(missing)[1])
    low = np.nanpercentile(x, 0.5, axis=0).astype(np.float32)
    high = np.nanpercentile(x, 99.5, axis=0).astype(np.float32)
    np.clip(x, low, high, out=x)
    return x, medians


def _lgbm_screen(
    x: np.ndarray,
    labels: np.ndarray,
    features: Sequence[str],
    config: ResidualEventArchetypeConfig,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    if (
        not bool(config.lgbm_enabled)
        or lgb is None
        or len(x) < int(config.lgbm_min_rows)
    ):
        return np.zeros(x.shape[1], dtype=np.float32), {"available": 0.0}
    split = max(300, int(round(0.70 * len(x))))
    if len(x) - split < 150:
        return np.zeros(x.shape[1], dtype=np.float32), {"available": 0.0}
    importance = np.zeros(x.shape[1], dtype=np.float64)
    tail_lifts: list[float] = []
    average_precision_lifts: list[float] = []
    for target_name in (
        "negative_residual_event",
        "adverse_path_event",
        "positive_residual_event",
        "favorable_near_miss_event",
    ):
        y = (labels == EVENT_CLASSES.index(target_name)).astype(np.int8)
        if (
            y[:split].sum() < int(config.min_event_class_rows)
            or y[split:].sum() == 0
            or y[split:].sum() == len(y[split:])
        ):
            continue
        params = {
            "objective": "binary",
            "learning_rate": 0.04,
            "max_depth": 3,
            "num_leaves": 7,
            "min_data_in_leaf": 70,
            "min_gain_to_split": 0.01,
            "feature_fraction": 0.80,
            "bagging_fraction": 0.80,
            "bagging_freq": 1,
            "lambda_l1": 0.15,
            "lambda_l2": 6.0,
            "seed": int(seed + EVENT_CLASSES.index(target_name) * 13),
            "num_threads": 2,
            "verbosity": -1,
            "force_col_wise": True,
        }
        booster = lgb.train(
            params,
            lgb.Dataset(x[:split], label=y[:split], free_raw_data=True),
            num_boost_round=int(config.lgbm_num_boost_round),
        )
        pred = np.asarray(booster.predict(x[split:]), dtype=np.float64)
        valid_y = y[split:]
        base_rate = float(np.mean(valid_y))
        if not 0.0 < base_rate < 1.0:
            continue
        top_count = max(1, int(math.ceil(0.15 * len(pred))))
        top = np.argpartition(pred, len(pred) - top_count)[-top_count:]
        top15_lift = float(np.mean(valid_y[top]) / max(base_rate, 1e-8))
        try:
            from sklearn.metrics import average_precision_score

            ap_lift = float(
                average_precision_score(valid_y, pred) / max(base_rate, 1e-8)
            )
        except ValueError:
            ap_lift = 1.0
        # Gain attribution is controlled by OOS rare-event precision rather
        # than AUC. This matches the actual use: recognize the most adverse or
        # favorable residual states within the globally defined top20 stream.
        weight = max(0.0, 0.70 * (top15_lift - 1.0) + 0.30 * (ap_lift - 1.0))
        importance += weight * np.asarray(
            booster.feature_importance(importance_type="gain"), dtype=np.float64
        )
        tail_lifts.append(top15_lift)
        average_precision_lifts.append(ap_lift)
    if not tail_lifts or float(importance.max()) <= 0.0:
        return np.zeros(x.shape[1], dtype=np.float32), {
            "available": 1.0,
            "mean_top15_precision_lift": float(np.mean(tail_lifts))
            if tail_lifts
            else np.nan,
            "mean_average_precision_lift": float(np.mean(average_precision_lifts))
            if average_precision_lifts
            else np.nan,
        }
    return (importance / float(importance.max())).astype(np.float32), {
        "available": 1.0,
        "mean_top15_precision_lift": float(np.mean(tail_lifts)),
        "mean_average_precision_lift": float(np.mean(average_precision_lifts)),
    }


def screen_local_residual_features(
    frame: pd.DataFrame,
    labels: np.ndarray,
    candidates: Sequence[str],
    *,
    config: ResidualEventArchetypeConfig,
    seed: int,
) -> tuple[list[str], pd.DataFrame, dict[str, float]]:
    """Binned local MI plus chronological shallow-LGBM interaction screening."""

    population = (
        _numeric(frame, "resid_event_top20_population", 0.0)
        .fillna(0.0)
        .gt(0.5)
        .to_numpy()
    )
    if int(population.sum()) < max(250, int(config.min_event_class_rows) * 4):
        return [], pd.DataFrame(), {"population_rows": float(population.sum())}
    sub = frame.loc[population]
    y = np.asarray(labels, dtype=np.int32)[population]
    sample_idx = _time_spread_indices(
        len(sub), min(len(sub), int(config.mi_sample_rows))
    )
    sample = sub.iloc[sample_idx]
    y_sample = y[sample_idx]
    rows: list[dict[str, float | str]] = []
    score_values = _numeric(sample, config.score_col).to_numpy(dtype=np.float32)
    score_edges = _safe_quantiles(
        score_values, np.linspace(0.0, 1.0, 6, dtype=np.float64)
    )
    score_bin = np.searchsorted(score_edges[1:-1], score_values, side="right")
    time_bands = np.array_split(np.arange(len(sample), dtype=np.int64), 3)
    for feature in candidates:
        values = _numeric(sample, feature).to_numpy(dtype=np.float32)
        mi_negative = _binned_mi(
            values,
            np.isin(
                y_sample,
                [
                    EVENT_CLASSES.index("negative_residual_event"),
                    EVENT_CLASSES.index("adverse_path_event"),
                ],
            ).astype(np.int8),
            config.mi_bins,
        )
        mi_positive = _binned_mi(
            values,
            np.isin(
                y_sample,
                [
                    EVENT_CLASSES.index("positive_residual_event"),
                    EVENT_CLASSES.index("favorable_near_miss_event"),
                ],
            ).astype(np.int8),
            config.mi_bins,
        )
        conditional_negative: list[float] = []
        conditional_positive: list[float] = []
        for band in range(5):
            local = score_bin == band
            if int(local.sum()) < 100:
                continue
            conditional_negative.append(
                _binned_mi(
                    values[local],
                    np.isin(
                        y_sample[local],
                        [
                            EVENT_CLASSES.index("negative_residual_event"),
                            EVENT_CLASSES.index("adverse_path_event"),
                        ],
                    ).astype(np.int8),
                    config.mi_bins,
                )
            )
            conditional_positive.append(
                _binned_mi(
                    values[local],
                    np.isin(
                        y_sample[local],
                        [
                            EVENT_CLASSES.index("positive_residual_event"),
                            EVENT_CLASSES.index("favorable_near_miss_event"),
                        ],
                    ).astype(np.int8),
                    config.mi_bins,
                )
            )
        temporal_mi: list[float] = []
        for band in time_bands:
            if len(band) < 100:
                continue
            adverse_band = np.isin(
                y_sample[band],
                [
                    EVENT_CLASSES.index("negative_residual_event"),
                    EVENT_CLASSES.index("adverse_path_event"),
                ],
            ).astype(np.int8)
            favorable_band = np.isin(
                y_sample[band],
                [
                    EVENT_CLASSES.index("positive_residual_event"),
                    EVENT_CLASSES.index("favorable_near_miss_event"),
                ],
            ).astype(np.int8)
            temporal_mi.append(
                max(
                    _binned_mi(values[band], adverse_band, config.mi_bins),
                    _binned_mi(values[band], favorable_band, config.mi_bins),
                )
            )
        conditional_mi = max(
            float(np.mean(conditional_negative)) if conditional_negative else 0.0,
            float(np.mean(conditional_positive)) if conditional_positive else 0.0,
        )
        temporal_mean = float(np.mean(temporal_mi)) if temporal_mi else 0.0
        temporal_worst = float(np.min(temporal_mi)) if temporal_mi else 0.0
        temporal_stability = temporal_worst / max(temporal_mean, 1e-8)
        rows.append(
            {
                "feature": str(feature),
                "mi_negative": mi_negative,
                "mi_positive": mi_positive,
                "mi_max": max(mi_negative, mi_positive),
                "conditional_mi": conditional_mi,
                "temporal_mi_mean": temporal_mean,
                "temporal_mi_worst": temporal_worst,
                "temporal_stability": float(np.clip(temporal_stability, 0.0, 1.0)),
            }
        )
    metrics = (
        pd.DataFrame(rows).sort_values("mi_max", ascending=False, kind="stable")
        if rows
        else pd.DataFrame()
    )
    if metrics.empty:
        return [], metrics, {"population_rows": float(population.sum())}
    mi_kept = (
        metrics.head(int(config.max_features_after_mi))["feature"].astype(str).tolist()
    )
    x, _medians = _matrix(sub, mi_kept)
    if bool(config.lgbm_enabled):
        lgbm_score, lgbm_meta = _lgbm_screen(x, y, mi_kept, config, seed)
    else:
        lgbm_score = np.zeros(len(mi_kept), dtype=np.float32)
        lgbm_meta = {"available": 0.0, "disabled": 1.0}
    metrics["lgbm_validation_gain"] = (
        metrics["feature"].map(dict(zip(mi_kept, lgbm_score, strict=True))).fillna(0.0)
    )
    metrics["combined_score"] = (
        0.45 * metrics["mi_max"]
        + 0.25 * metrics["conditional_mi"]
        + 0.15 * metrics["temporal_mi_worst"]
        + 0.15 * metrics["lgbm_validation_gain"]
    )
    adaptive_feature_limit = min(
        int(config.max_features_after_lgbm),
        max(8, int(np.sqrt(max(int(population.sum()), 1)))),
    )
    ordered = (
        metrics.sort_values("combined_score", ascending=False, kind="stable")
        .head(max(adaptive_feature_limit * 3, adaptive_feature_limit))
    )
    # Greedy correlation pruning prevents repeated volatility/OI variants from
    # consuming the bottleneck while retaining the strongest local economic
    # representative of each redundant family.
    selected: list[str] = []
    corr_sample = sample.reindex(columns=ordered["feature"].astype(str).tolist()).apply(
        pd.to_numeric, errors="coerce"
    )
    for feature in ordered["feature"].astype(str):
        if len(selected) >= adaptive_feature_limit:
            break
        if selected:
            corr = corr_sample[selected].corrwith(corr_sample[feature]).abs()
            if bool(corr.gt(0.94).any()):
                continue
        selected.append(feature)
    diagnostics = {
        "population_rows": float(population.sum()),
        "candidate_rows": float(len(candidates)),
        "selected_rows": float(len(selected)),
        "adaptive_feature_limit": float(adaptive_feature_limit),
        **lgbm_meta,
    }
    return selected, metrics, diagnostics


def _state_static_feature_names(prefix: str = RESIDUAL_EVENT_PREFIX) -> list[str]:
    raw = ae_gmm_feature_columns(prefix)
    keep = (
        "dae_b16_",
        "gmm_prob_",
        "gmm_cluster_posterior_",
        "gmm_cluster_id",
        "gmm_posterior_max",
        "gmm_posterior_margin",
        "gmm_entropy",
        "gmm_unknown_probability",
        "gmm_ood_score",
        "mahalanobis_distance",
        "min_mahalanobis",
        "expected_mahalanobis",
        "AE_reconstruction_error",
        "dae_reconstruction_error",
    )
    return [
        name
        for name in raw
        if any(token in name for token in keep)
        and not any(
            token in name
            for token in (
                "delta",
                "accel",
                "speed",
                "cluster_t",
                "flip",
                "stability",
                "time_since",
            )
        )
    ]


def residual_event_feature_names() -> list[str]:
    names = _state_static_feature_names()
    names += [
        f"{RESIDUAL_EVENT_PREFIX}{suffix}"
        for suffix in RESIDUAL_EVENT_TEMPORAL_SUFFIXES
    ]
    names += [
        f"{RESIDUAL_EVENT_PREFIX}expected_negative_residual_event",
        f"{RESIDUAL_EVENT_PREFIX}expected_adverse_path_event",
        f"{RESIDUAL_EVENT_PREFIX}expected_positive_residual_event",
        f"{RESIDUAL_EVENT_PREFIX}expected_favorable_near_miss_event",
        f"{RESIDUAL_EVENT_PREFIX}expected_ev_after_1pct",
        f"{RESIDUAL_EVENT_PREFIX}expected_market_peer_surprise",
        f"{RESIDUAL_EVENT_PREFIX}expected_ev_timestamp_neutral_surprise",
        f"{RESIDUAL_EVENT_PREFIX}expected_persistence_strength",
        f"{RESIDUAL_EVENT_PREFIX}expected_directional_ev_divergence",
        f"{RESIDUAL_EVENT_PREFIX}expected_bullish_tape_adverse_ev",
        f"{RESIDUAL_EVENT_PREFIX}expected_timestamp_ev_sign_disagreement",
        f"{RESIDUAL_EVENT_PREFIX}expected_persistent_subthreshold_damage",
        f"{RESIDUAL_EVENT_PREFIX}expected_persistent_material_nontail",
        f"{RESIDUAL_EVENT_PREFIX}local_support_log1p",
        f"{RESIDUAL_EVENT_PREFIX}local_model",
    ]
    names += [
        f"{RESIDUAL_EVENT_PREFIX}{suffix}"
        for suffix in RESIDUAL_EVENT_TRAJECTORY_SUFFIXES
    ]
    names += [
        f"{RESIDUAL_EVENT_PREFIX}expected_correct_direction",
        f"{RESIDUAL_EVENT_PREFIX}expected_negative_executable_ev",
        f"{RESIDUAL_EVENT_PREFIX}expected_adverse_path_damage",
        f"{RESIDUAL_EVENT_PREFIX}expected_executable_adverse_path_event",
        f"{RESIDUAL_EVENT_PREFIX}expected_correct_direction_bad_trade",
        f"{RESIDUAL_EVENT_PREFIX}expected_correct_direction_adverse_path_event",
        f"{RESIDUAL_EVENT_PREFIX}expected_executable_quality_gap",
    ]
    names += [
        f"{RESIDUAL_EVENT_PREFIX}expected_{target}"
        for target in EXECUTABLE_FAILURE_TARGETS
    ]
    return list(dict.fromkeys(names))


def residual_event_market_feature_names() -> list[str]:
    names = _state_static_feature_names(RESIDUAL_EVENT_MARKET_PREFIX)
    names += [
        f"{RESIDUAL_EVENT_MARKET_PREFIX}support_log1p",
        f"{RESIDUAL_EVENT_MARKET_PREFIX}enabled",
    ]
    return list(dict.fromkeys(names))


def residual_event_distilled_feature_names(*, include_market: bool = True) -> list[str]:
    """Revision-stable residual-state features for historical model ablations.

    Raw latent coordinates, cluster IDs, and posterior slots can change meaning
    when a later nested state revision is fitted.  These semantic priors and
    uncertainty/distance summaries retain the same interpretation across
    revisions and are therefore the safe default for a combined OOS archive.
    """

    local_suffixes = (
        "gmm_posterior_max",
        "gmm_posterior_margin",
        "gmm_entropy",
        "gmm_unknown_probability",
        "gmm_ood_score",
        "mahalanobis_distance",
        "min_mahalanobis",
        "expected_mahalanobis",
        "AE_reconstruction_error",
        "dae_reconstruction_error",
        "dae_reconstruction_error_zscore",
        *RESIDUAL_EVENT_TEMPORAL_SUFFIXES,
        "expected_negative_residual_event",
        "expected_adverse_path_event",
        "expected_positive_residual_event",
        "expected_favorable_near_miss_event",
        "expected_ev_after_1pct",
        "expected_market_peer_surprise",
        "expected_ev_timestamp_neutral_surprise",
        "expected_persistence_strength",
        "expected_directional_ev_divergence",
        "expected_bullish_tape_adverse_ev",
        "expected_timestamp_ev_sign_disagreement",
        "expected_persistent_subthreshold_damage",
        "expected_persistent_material_nontail",
        "expected_correct_direction",
        "expected_negative_executable_ev",
        "expected_adverse_path_damage",
        "expected_executable_adverse_path_event",
        "expected_correct_direction_bad_trade",
        "expected_correct_direction_adverse_path_event",
        "expected_executable_quality_gap",
        *[f"expected_{target}" for target in EXECUTABLE_FAILURE_TARGETS],
        "local_support_log1p",
        "local_model",
    )
    names = [f"{RESIDUAL_EVENT_PREFIX}{suffix}" for suffix in local_suffixes]
    if include_market:
        market_suffixes = (
            "gmm_posterior_max",
            "gmm_posterior_margin",
            "gmm_entropy",
            "gmm_unknown_probability",
            "gmm_ood_score",
            "mahalanobis_distance",
            "min_mahalanobis",
            "expected_mahalanobis",
            "AE_reconstruction_error",
            "dae_reconstruction_error",
            "dae_reconstruction_error_zscore",
            "support_log1p",
            "enabled",
        )
        names.extend(
            f"{RESIDUAL_EVENT_MARKET_PREFIX}{suffix}" for suffix in market_suffixes
        )
    return names


def residual_event_quality_probability_feature_names() -> list[str]:
    """Return only semantic executable-quality probabilities.

    This is deliberately narrower than :func:`residual_event_distilled_feature_names`.
    It is for conditional meta-head ablations where raw uncertainty, distance,
    and temporal summaries would otherwise add dozens of correlated features.
    Every name represents one frozen, train-derived economic probability.
    """

    return [
        f"{RESIDUAL_EVENT_PREFIX}expected_{target}"
        for target in EXECUTABLE_FAILURE_TARGETS
    ]


def add_residual_event_temporal_context(
    generated: pd.DataFrame,
    observable: pd.DataFrame,
    config: ResidualEventArchetypeConfig,
) -> pd.DataFrame:
    """Add causal local-state transitions from frozen AE/GMM outputs.

    State is collapsed by timestamp within side x archetype before temporal
    differences are taken. This prevents symbol row ordering from generating
    artificial speed. A single-timestamp inference caller should provide
    observable history through ``transform_oos_with_history``.
    """

    out = generated.copy(deep=False)
    timestamp = pd.to_datetime(
        observable[config.timestamp_col], utc=True, errors="coerce"
    )
    _side, _arch, key = _side_arch(observable, config)
    if "state_revision_cutoff" in observable.columns:
        revision = observable["state_revision_cutoff"].astype(str).fillna("missing")
        key = key.astype(str) + "|revision=" + revision
    posterior_columns = [
        f"{RESIDUAL_EVENT_PREFIX}gmm_cluster_posterior_{idx}" for idx in range(7)
    ]
    trajectory_sources = (
        _trajectory_source_columns(observable)
        if bool(config.enable_transition_trajectory_features)
        else {}
    )
    for _, labels in key.groupby(key, sort=False).groups.items():
        labels = pd.Index(list(labels))
        local_ts = timestamp.loc[labels]
        valid = local_ts.notna()
        if not bool(valid.any()):
            continue
        valid_labels = labels[valid.to_numpy()]
        panel_source = out.loc[
            valid_labels,
            [
                *posterior_columns,
                f"{RESIDUAL_EVENT_PREFIX}gmm_ood_score",
                f"{RESIDUAL_EVENT_PREFIX}dae_reconstruction_error_zscore",
            ],
        ].copy()
        # Pre-entry lifecycle values are aggregated over the local side x
        # archetype stream before temporal differences are calculated.  This
        # keeps the feature causal and independent of asset iteration order.
        for mechanism, source_name in trajectory_sources.items():
            panel_source[f"__trajectory__{mechanism}"] = _numeric(
                observable.loc[valid_labels], source_name, 0.0
            ).fillna(0.0).to_numpy(dtype=np.float32)
        panel_source["__ts__"] = local_ts.loc[valid_labels].to_numpy()
        aggregations = {name: "mean" for name in posterior_columns}
        aggregations[f"{RESIDUAL_EVENT_PREFIX}gmm_ood_score"] = "max"
        aggregations[f"{RESIDUAL_EVENT_PREFIX}dae_reconstruction_error_zscore"] = (
            "max"
        )
        aggregations.update(
            {f"__trajectory__{mechanism}": "mean" for mechanism in trajectory_sources}
        )
        panel = (
            panel_source.groupby("__ts__", observed=True, sort=True)
            .agg(aggregations)
            .sort_index()
        )
        posterior = panel[posterior_columns].to_numpy(dtype=np.float32, copy=False)
        posterior_delta = np.vstack(
            [
                np.zeros((1, posterior.shape[1]), dtype=np.float32),
                np.diff(posterior, axis=0),
            ]
        )
        speed = np.linalg.norm(posterior_delta, axis=1).astype(np.float32)
        panel[f"{RESIDUAL_EVENT_PREFIX}posterior_speed"] = speed
        panel[f"{RESIDUAL_EVENT_PREFIX}posterior_acceleration"] = np.diff(
            speed, prepend=speed[0]
        ).astype(np.float32)
        previous_posterior = np.vstack(
            [posterior[:1], posterior[:-1]]
        ).astype(np.float32, copy=False)
        posterior_overlap = np.sum(posterior * previous_posterior, axis=1)
        panel[f"{RESIDUAL_EVENT_PREFIX}posterior_switch_pressure"] = np.clip(
            1.0 - posterior_overlap, 0.0, 1.0
        ).astype(np.float32)
        posterior_entropy = -np.sum(
            np.where(posterior > 1e-8, posterior * np.log(np.maximum(posterior, 1e-8)), 0.0),
            axis=1,
        )
        panel[f"{RESIDUAL_EVENT_PREFIX}posterior_entropy_delta"] = np.diff(
            posterior_entropy, prepend=posterior_entropy[0]
        ).astype(np.float32)
        ood_name = f"{RESIDUAL_EVENT_PREFIX}gmm_ood_score"
        reconstruction_name = (
            f"{RESIDUAL_EVENT_PREFIX}dae_reconstruction_error_zscore"
        )
        for hours in (24, 48, 96):
            panel[f"{RESIDUAL_EVENT_PREFIX}ood_recent_max_{hours}h"] = (
                panel[ood_name].rolling(f"{hours}h", min_periods=1).max()
            )
            panel[
                f"{RESIDUAL_EVENT_PREFIX}reconstruction_recent_max_{hours}h"
            ] = panel[reconstruction_name].rolling(
                f"{hours}h", min_periods=1
            ).max()
        spike = panel[ood_name].ge(1.0) | panel[reconstruction_name].ge(2.0)
        last_spike: pd.Timestamp | None = None
        hours_since = np.full(len(panel), 96.0, dtype=np.float32)
        for position, (ts_value, active) in enumerate(
            zip(panel.index, spike.to_numpy(bool), strict=False)
        ):
            current = pd.Timestamp(ts_value)
            if active:
                last_spike = current
            if last_spike is not None:
                hours_since[position] = np.float32(
                    min(
                        96.0,
                        max(
                            0.0,
                            (current - last_spike).total_seconds() / 3600.0,
                        ),
                    )
                )
        panel[f"{RESIDUAL_EVENT_PREFIX}hours_since_ood_spike_96h_norm"] = (
            hours_since / np.float32(96.0)
        )
        if trajectory_sources:
            def _lag_delta(series: pd.Series, hours: int) -> pd.Series:
                # ``reindex`` yields NaN over gaps, so a missing timestamp
                # cannot be mistaken for a compressed multi-hour transition.
                prior = series.reindex(panel.index - pd.Timedelta(hours=hours))
                prior.index = panel.index
                return (series - prior).where(prior.notna(), 0.0).astype(np.float32)

            mechanism_values = {
                mechanism: panel[f"__trajectory__{mechanism}"].astype(np.float32)
                for mechanism in trajectory_sources
            }
            if "oi_flush" in mechanism_values:
                panel[f"{RESIDUAL_EVENT_PREFIX}oi_flush_impulse_1h"] = _lag_delta(
                    mechanism_values["oi_flush"], 1
                )
            if "oi_recovery" in mechanism_values:
                panel[f"{RESIDUAL_EVENT_PREFIX}oi_recovery_impulse_4h"] = _lag_delta(
                    mechanism_values["oi_recovery"], 4
                )
            if "funding_release" in mechanism_values:
                panel[f"{RESIDUAL_EVENT_PREFIX}funding_release_impulse_4h"] = _lag_delta(
                    mechanism_values["funding_release"], 4
                )
            if "breadth_recovery" in mechanism_values:
                panel[f"{RESIDUAL_EVENT_PREFIX}breadth_recovery_impulse_4h"] = _lag_delta(
                    mechanism_values["breadth_recovery"], 4
                )
            if "short_covering" in mechanism_values:
                panel[f"{RESIDUAL_EVENT_PREFIX}short_covering_impulse_1h"] = _lag_delta(
                    mechanism_values["short_covering"], 1
                )
            if "liquidity_stress" in mechanism_values:
                stress = mechanism_values["liquidity_stress"].abs()
                panel[f"{RESIDUAL_EVENT_PREFIX}liquidity_stress_impulse_1h"] = _lag_delta(
                    stress, 1
                )
            if "oi_flush" in mechanism_values and "breadth_recovery" in mechanism_values:
                panel[
                    f"{RESIDUAL_EVENT_PREFIX}deleveraging_to_recovery_rotation_4h"
                ] = (
                    _lag_delta(mechanism_values["breadth_recovery"], 4)
                    - _lag_delta(mechanism_values["oi_flush"], 4)
                ).astype(np.float32)
        for suffix in RESIDUAL_EVENT_TEMPORAL_SUFFIXES:
            name = f"{RESIDUAL_EVENT_PREFIX}{suffix}"
            out.loc[valid_labels, name] = (
                local_ts.loc[valid_labels]
                .map(panel[name])
                .fillna(0.0)
                .to_numpy(dtype=np.float32)
            )
        for suffix in RESIDUAL_EVENT_TRAJECTORY_SUFFIXES:
            name = f"{RESIDUAL_EVENT_PREFIX}{suffix}"
            if name not in panel:
                continue
            out.loc[valid_labels, name] = (
                local_ts.loc[valid_labels]
                .map(panel[name])
                .fillna(0.0)
                .to_numpy(dtype=np.float32)
            )
    return out


@dataclass
class _LocalResidualState:
    key: str
    feature_columns: list[str]
    ae_gmm_state: dict[str, Any]
    priors: dict[str, np.ndarray]
    support_rows: int
    screening: pd.DataFrame = field(default_factory=pd.DataFrame)
    screening_meta: dict[str, float] = field(default_factory=dict)


@dataclass
class _MarketResidualState:
    feature_columns: list[str]
    ae_gmm_state: dict[str, Any]
    support_rows: int


def _posterior_priors(
    transformed: pd.DataFrame,
    labelled: pd.DataFrame,
    config: ResidualEventArchetypeConfig,
) -> dict[str, np.ndarray]:
    posterior_cols = [
        f"{RESIDUAL_EVENT_PREFIX}gmm_cluster_posterior_{idx}"
        for idx in range(7)
        if f"{RESIDUAL_EVENT_PREFIX}gmm_cluster_posterior_{idx}" in transformed
    ]
    if not posterior_cols:
        return {}
    posterior = (
        transformed[posterior_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    denominator = posterior.sum(axis=0).astype(np.float64)
    priors: dict[str, np.ndarray] = {}
    classes = labelled["resid_event_class"].astype(str)
    targets = {
        "negative_residual_event": classes.isin(
            ["negative_residual_event", "adverse_path_event"]
        ).to_numpy(dtype=np.float32),
        "adverse_path_event": classes.eq("adverse_path_event").to_numpy(
            dtype=np.float32
        ),
        "positive_residual_event": classes.eq("positive_residual_event").to_numpy(
            dtype=np.float32
        ),
        "favorable_near_miss_event": classes.eq("favorable_near_miss_event").to_numpy(
            dtype=np.float32
        ),
        "ev_after_1pct": _numeric(labelled, config.ev_col, 0.0)
        .fillna(0.0)
        .to_numpy(dtype=np.float32),
        "market_peer_surprise": _numeric(
            labelled, "resid_event_market_peer_surprise", 0.0
        )
        .fillna(0.0)
        .to_numpy(dtype=np.float32),
        "ev_timestamp_neutral_surprise": _numeric(
            labelled, "resid_event_ev_timestamp_neutral_surprise", 0.0
        )
        .fillna(0.0)
        .to_numpy(dtype=np.float32),
        "persistence_strength": _numeric(
            labelled, "resid_event_persistence_strength", 0.0
        )
        .fillna(0.0)
        .to_numpy(dtype=np.float32),
        "directional_ev_divergence": _numeric(
            labelled,
            "resid_target_side_archetype_directional_ev_divergence_24h",
            0.0,
        )
        .fillna(0.0)
        .to_numpy(dtype=np.float32),
        "bullish_tape_adverse_ev": _numeric(
            labelled,
            "resid_target_side_archetype_bullish_tape_adverse_ev_24h",
            0.0,
        )
        .fillna(0.0)
        .to_numpy(dtype=np.float32),
        "timestamp_ev_sign_disagreement": _numeric(
            labelled,
            "resid_target_side_archetype_timestamp_ev_sign_disagreement_24h",
            0.0,
        )
        .fillna(0.0)
        .to_numpy(dtype=np.float32),
        "persistent_subthreshold_damage": _numeric(
            labelled,
            "resid_target_side_archetype_persistent_subthreshold_damage_24h",
            0.0,
        )
        .fillna(0.0)
        .to_numpy(dtype=np.float32),
        "persistent_material_nontail": _numeric(
            labelled,
            "resid_target_side_archetype_persistent_material_nontail_24h",
            0.0,
        )
        .fillna(0.0)
        .to_numpy(dtype=np.float32),
    }
    targets.update(_executable_quality_targets(labelled, config))
    for name, target in targets.items():
        global_mean = float(np.mean(target)) if len(target) else 0.0
        local = (posterior.T @ target.astype(np.float64)) / np.maximum(
            denominator, 1e-8
        )
        strength = denominator / (denominator + float(config.prior_shrinkage_rows))
        priors[name] = (strength * local + (1.0 - strength) * global_mean).astype(
            np.float32
        )
    return priors


def _executable_quality_targets(
    labelled: pd.DataFrame,
    config: ResidualEventArchetypeConfig,
) -> dict[str, np.ndarray]:
    """Return outcome-only labels separating direction from executable quality.

    ``clean_exec`` is the directional/first-touch success signal.  A direction
    can still be unsuitable to trade when the realized return is negative after
    costs or the path reaches adverse/timeout territory.  Keeping these as
    separate targets prevents the state model from treating every correct
    direction as a deployable trade.
    """

    n = len(labelled)
    if not bool(config.enable_executable_quality_targets):
        return {}
    correct = _numeric(labelled, config.hit_col, 0.0).fillna(0.0).clip(0.0, 1.0)
    ev = _numeric(labelled, config.ev_col, 0.0).fillna(0.0)
    top10 = _numeric(labelled, "resid_event_top10_population", 0.0).fillna(0.0)
    top20 = _numeric(labelled, "resid_event_top20_population", 0.0).fillna(0.0)
    first_touch_bad_mae = _numeric(
        labelled, "first_touch_bad_mae_1r", 0.0
    ).fillna(0.0).clip(0.0, 1.0)
    full_path_bad_mae = _numeric(
        labelled, config.bad_mae_col, 0.0
    ).fillna(0.0).clip(0.0, 1.0)
    timeout = _numeric(labelled, config.timeout_col, 0.0).fillna(0.0).clip(0.0, 1.0)
    dirty = _numeric(labelled, config.dirty_col, 0.0).fillna(0.0).clip(0.0, 1.0)
    stop = _numeric(labelled, config.stop_col, 0.0).fillna(0.0).clip(0.0, 1.0)

    # A bounded soft target provides a meaningful ordering between a timeout,
    # a full-path MAE breach and a hard stop while remaining robust to missing
    # optional stop columns in historical candidate shards.
    damage = np.clip(
        0.35 * first_touch_bad_mae.to_numpy(dtype=np.float32)
        + 0.50 * full_path_bad_mae.to_numpy(dtype=np.float32)
        + 0.20 * timeout.to_numpy(dtype=np.float32)
        + 0.25 * stop.to_numpy(dtype=np.float32)
        + 0.15 * dirty.to_numpy(dtype=np.float32),
        0.0,
        1.0,
    ).astype(np.float32)
    correct_arr = correct.to_numpy(dtype=np.float32)
    negative_exec = (
        top10.gt(0.5).to_numpy() & ev.le(0.0).to_numpy()
    ).astype(np.float32)
    correct_but_bad = (
        (correct_arr > 0.5) & ((damage >= 0.35) | (ev.to_numpy() <= 0.0))
    ).astype(np.float32)
    adverse_event = (
        (full_path_bad_mae.to_numpy() > 0.5)
        | (timeout.to_numpy() > 0.5)
        | (stop.to_numpy() > 0.5)
    ).astype(np.float32)
    top_tail = top10.gt(0.5).to_numpy()
    near_tail = top20.gt(0.5).to_numpy() & ~top_tail
    event_class = labelled.get(
        "resid_event_class", pd.Series("normal", index=labelled.index)
    ).astype(str)
    negative_residual_event = event_class.isin(
        ("negative_residual_event", "adverse_path_event")
    ).to_numpy()
    positive_residual_event = event_class.isin(
        ("positive_residual_event", "favorable_near_miss_event")
    ).to_numpy()
    side, archetype, _ = _side_arch(labelled, config)
    side_arr = side.to_numpy(dtype=object)
    archetype_arr = archetype.astype(str).str.lower().to_numpy(dtype=object)
    ev_arr = ev.to_numpy(dtype=np.float32)
    adverse_arr = adverse_event.astype(bool)
    clean_path = np.asarray(
        (damage < 0.35)
        & (timeout.to_numpy(dtype=np.float32) <= 0.5)
        & (dirty.to_numpy(dtype=np.float32) <= 0.5)
    , dtype=bool)
    # A first-touch/directional success can still turn into a non-actionable
    # trade when the remaining path is adverse or dirty and net return is
    # negative.  This is the path-reversal ambiguity that generic hit-rate
    # residuals cannot express on their own.
    reversal_after_initial_success = (
        top_tail
        & negative_residual_event
        & (correct_arr > 0.5)
        & (ev_arr <= 0.0)
        & (adverse_arr | (dirty.to_numpy(dtype=np.float32) > 0.5))
    )
    targets = {
        "correct_direction": correct_arr,
        "negative_executable_ev": negative_exec,
        "adverse_path_damage": damage,
        "executable_adverse_path_event": adverse_event,
        "correct_direction_bad_trade": correct_but_bad,
        "correct_direction_adverse_path_event": (
            (correct_arr > 0.5) & (adverse_event > 0.5)
        ).astype(np.float32),
        "executable_quality_gap": (correct_arr - damage).astype(np.float32),
    }
    # Keep the mechanisms disjoint where possible.  That gives downstream
    # feature selection something actionable to choose between instead of six
    # almost-identical variants of negative EV.
    targets.update(
        {
            # High-ranked call that got the direction wrong and lost money.
            "top_tail_false_positive": (
                top_tail & (correct_arr <= 0.5) & (ev_arr <= 0.0)
            ).astype(np.float32),
            # Correct direction, no material adverse path, but no net edge
            # after the canonical cost contract.  This is a calibration/cost
            # failure rather than a stop or timeout state.
            "top_tail_clean_cost_fragile": (
                top_tail
                & (correct_arr > 0.5)
                & (ev_arr <= 0.0)
                & clean_path
            ).astype(np.float32),
            # A path-damage failure that also loses after costs.  It is kept
            # separate from an ordinary false positive because it should be
            # explained by lifecycle/volatility state rather than direction.
            "top_tail_adverse_loss": (
                top_tail & adverse_arr & (ev_arr <= 0.0)
            ).astype(np.float32),
            # The especially harmful overlap: a high-ranked call was wrong
            # *and* experienced a damaging path.  It is more actionable than
            # a generic false-positive for a residual-state context model.
            "top_tail_adverse_false_positive": (
                top_tail & (correct_arr <= 0.5) & adverse_arr & (ev_arr <= 0.0)
            ).astype(np.float32),
            # Timeouts are a distinct slow-resolution mechanism; they remain
            # useful even where the timeout exit is mildly positive.
            "top_tail_timeout_failure": (
                top_tail & (timeout.to_numpy(dtype=np.float32) > 0.5)
            ).astype(np.float32),
            # Timeout itself can be a tolerable positive exit.  This stricter
            # label captures only the slow-resolution cases that lost after
            # the canonical fee/cost contract.
            "top_tail_timeout_loss": (
                top_tail & (timeout.to_numpy(dtype=np.float32) > 0.5) & (ev_arr <= 0.0)
            ).astype(np.float32),
            # Positive utility does not make a dirty path deployable.  This
            # isolates the difficult "winner with ugly path" population.
            "top_tail_dirty_positive": (
                top_tail & (dirty.to_numpy(dtype=np.float32) > 0.5) & (ev_arr > 0.0)
            ).astype(np.float32),
            "top_tail_dirty_loss": (
                top_tail & (dirty.to_numpy(dtype=np.float32) > 0.5) & (ev_arr <= 0.0)
            ).astype(np.float32),
            # Positive counterpart for the meta head: a genuinely executable
            # outcome, distinct from mere first-touch directional correctness.
            "top_tail_clean_executable": (
                top_tail
                & (correct_arr > 0.5)
                & (ev_arr > 0.0)
                & clean_path
            ).astype(np.float32),
            # The policy trades a top tail, but a state model can also help
            # rank clean, executable opportunities just below it.  These
            # labels never create a local percentile admission rule: they are
            # regular meta inputs evaluated under the same global top-tail
            # policy after ranking.
            "near_tail_clean_executable": (
                near_tail
                & (correct_arr > 0.5)
                & (ev_arr > 0.0)
                & clean_path
            ).astype(np.float32),
            "near_tail_clean_cost_fragile": (
                near_tail
                & (correct_arr > 0.5)
                & (ev_arr <= 0.0)
                & clean_path
            ).astype(np.float32),
            # These are residualized mechanisms, not raw outcome-frequency
            # labels.  They isolate the rows where a score/archetype-local
            # expectation failed, which is the information a downstream meta
            # head does not already receive from its base anchors.
            "top_tail_residual_false_positive": (
                top_tail
                & negative_residual_event
                & (correct_arr <= 0.5)
                & (ev_arr <= 0.0)
            ).astype(np.float32),
            "top_tail_residual_adverse_loss": (
                top_tail & negative_residual_event & adverse_arr & (ev_arr <= 0.0)
            ).astype(np.float32),
            "top_tail_residual_timeout_loss": (
                top_tail
                & negative_residual_event
                & (timeout.to_numpy(dtype=np.float32) > 0.5)
                & (ev_arr <= 0.0)
            ).astype(np.float32),
            "near_tail_positive_residual_clean_executable": (
                near_tail
                & positive_residual_event
                & (correct_arr > 0.5)
                & (ev_arr > 0.0)
                & clean_path
            ).astype(np.float32),
            # These three labels encode the persistent side x archetype
            # failure patterns found in the negative-hit-residual leaf audit.
            # They intentionally use realized outcomes and residual labels,
            # not OOD/AE/GMM values, so the state model must discover the
            # observable precursor rather than receive it by construction.
            "long_mixed_latent_misfire": (
                (side_arr == "long")
                & (np.char.find(archetype_arr.astype(str), "long_mixed") >= 0)
                & top_tail
                & negative_residual_event
                & (ev_arr <= 0.0)
            ).astype(np.float32),
            "short_mixed_off_manifold": (
                (side_arr == "short")
                & (np.char.find(archetype_arr.astype(str), "short_mixed") >= 0)
                & top_tail
                & negative_residual_event
                & (ev_arr <= 0.0)
                & ((correct_arr <= 0.5) | adverse_arr)
            ).astype(np.float32),
            "short_default_latent_uncertainty": (
                (side_arr == "short")
                & (np.char.find(archetype_arr.astype(str), "short_default") >= 0)
                & top_tail
                & negative_residual_event
                & (ev_arr <= 0.0)
                & (adverse_arr | (timeout.to_numpy(dtype=np.float32) > 0.5))
            ).astype(np.float32),
            # Broader path-reversal state, then its two important local
            # variants.  Side/archetype identity is part of the label scope;
            # AE/GMM/OOD, shock, dislocation and liquidity remain strictly
            # observable candidate inputs rather than outcome-label criteria.
            "top_tail_reversal_after_initial_success": reversal_after_initial_success.astype(
                np.float32
            ),
            "long_mixed_reversal_after_initial_success": (
                reversal_after_initial_success
                & (side_arr == "long")
                & (np.char.find(archetype_arr.astype(str), "long_mixed") >= 0)
            ).astype(np.float32),
            "short_mixed_reversal_after_initial_success": (
                reversal_after_initial_success
                & (side_arr == "short")
                & (np.char.find(archetype_arr.astype(str), "short_mixed") >= 0)
            ).astype(np.float32),
            "long_breakout_overconfident_path_loss": (
                (side_arr == "long")
                & (np.char.find(archetype_arr.astype(str), "breakout") >= 0)
                & top_tail
                & negative_residual_event
                & (ev_arr <= 0.0)
                & ((correct_arr <= 0.5) | adverse_arr)
            ).astype(np.float32),
            "short_breakout_overconfident_path_loss": (
                (side_arr == "short")
                & (np.char.find(archetype_arr.astype(str), "breakout") >= 0)
                & top_tail
                & negative_residual_event
                & (ev_arr <= 0.0)
                & ((correct_arr <= 0.5) | adverse_arr)
            ).astype(np.float32),
        }
    )
    return targets


# Priority-ordered aliases make the trajectory block portable across older and
# newer feature-store schemas.  Every candidate is observable at decision time.
_TRAJECTORY_SOURCE_ALIASES: dict[str, tuple[str, ...]] = {
    "oi_flush": (
        "mkt_systemic_deleveraging_score",
        "liquidation_onset_score",
        "mkt_oi_flush_breadth_accel_1h",
    ),
    "oi_recovery": (
        "mkt_median_oi_recovery_fraction_24h",
        "mkt_oi_flush_breadth_recovery_4h",
        "post_flush_leverage_rebuild",
    ),
    "funding_release": (
        "funding_crowding_release_4h",
        "funding_confirmed_short_covering",
        "funding_mean_reversion_after_oi_flush",
    ),
    "breadth_recovery": (
        "market_breadth_recovery_from_24h_min",
        "market_breadth_recovery_from_6h_min",
        "breadth_recovery_from_6h_min",
    ),
    "short_covering": (
        "short_covering_score_market",
        "mkt_median_short_cover_intensity_1h",
        "asset_short_covering_score",
    ),
    "liquidity_stress": (
        "mark_perp_dislocation",
        "q_tail_width__ob_spread_z_x_rv_24h",
        "q_tail_width__ob_trade_size_to_l1_depth_z_24h",
    ),
}


def _trajectory_source_columns(frame: pd.DataFrame) -> dict[str, str]:
    """Resolve one stable observable source per transition mechanism."""

    result: dict[str, str] = {}
    for mechanism, aliases in _TRAJECTORY_SOURCE_ALIASES.items():
        for name in aliases:
            if name not in frame.columns:
                continue
            values = _numeric(frame, name)
            # Live/history bridge calls may carry only a few bars.  Three
            # observations are enough to establish a causal direction; the
            # derived value remains zero until its requested lag exists.
            if int(values.notna().sum()) >= max(3, int(len(frame) * 0.05)):
                result[mechanism] = name
                break
    return result


@dataclass
class ResidualEventArchetypeState:
    """Frozen side x archetype AE/GMM residual-state bundle."""

    config: ResidualEventArchetypeConfig
    threshold_state: GlobalEVThresholdState | None = None
    expectation_state: ScoreExpectationState | None = None
    ev_expectation_state: ScoreExpectationState | None = None
    baseline_state: ResidualEventBaselineState | None = None
    side_models: dict[str, _LocalResidualState] = field(default_factory=dict)
    local_models: dict[str, _LocalResidualState] = field(default_factory=dict)
    market_model: _MarketResidualState | None = None
    feature_metrics_: pd.DataFrame = field(default_factory=pd.DataFrame)
    event_catalog_: pd.DataFrame = field(default_factory=pd.DataFrame)
    train_start_: str | None = None
    train_end_: str | None = None

    def _fit_local(
        self,
        labelled: pd.DataFrame,
        candidates: Sequence[str],
        *,
        key: str,
        seed: int,
    ) -> _LocalResidualState | None:
        population = labelled.loc[
            _numeric(labelled, "resid_event_top20_population", 0.0).fillna(0.0).gt(0.5)
        ].copy()
        if len(population) < int(self.config.min_local_state_rows):
            return None
        labels = (
            population["resid_event_class"].cat.codes.to_numpy(dtype=np.int32)
            if pd.api.types.is_categorical_dtype(population["resid_event_class"])
            else pd.Categorical(
                population["resid_event_class"], categories=EVENT_CLASSES
            ).codes.astype(np.int32)
        )
        selected, screen, screening_meta = screen_local_residual_features(
            population, labels, candidates, config=self.config, seed=seed
        )
        if len(selected) < 2:
            return None
        targets = {
            "negative_residual_event": np.isin(
                labels,
                [
                    EVENT_CLASSES.index("negative_residual_event"),
                    EVENT_CLASSES.index("adverse_path_event"),
                ],
            ).astype(np.float32),
            "adverse_path_event": (
                labels == EVENT_CLASSES.index("adverse_path_event")
            ).astype(np.float32),
            "positive_residual_event": (
                labels == EVENT_CLASSES.index("positive_residual_event")
            ).astype(np.float32),
            "favorable_near_miss_event": (
                labels == EVENT_CLASSES.index("favorable_near_miss_event")
            ).astype(np.float32),
            "timestamp_neutral_surprise": _numeric(
                population, "resid_event_timestamp_neutral_surprise", 0.0
            ).to_numpy(dtype=np.float32),
            "ev_timestamp_neutral_surprise": _numeric(
                population, "resid_event_ev_timestamp_neutral_surprise", 0.0
            ).to_numpy(dtype=np.float32),
            "market_peer_surprise": _numeric(
                population, "resid_event_market_peer_surprise", 0.0
            ).to_numpy(dtype=np.float32),
            "persistence_strength": _numeric(
                population, "resid_event_persistence_strength", 0.0
            ).to_numpy(dtype=np.float32),
            "directional_ev_divergence": _numeric(
                population,
                "resid_target_side_archetype_directional_ev_divergence_24h",
                0.0,
            ).to_numpy(dtype=np.float32),
            "bullish_tape_adverse_ev": _numeric(
                population,
                "resid_target_side_archetype_bullish_tape_adverse_ev_24h",
                0.0,
            ).to_numpy(dtype=np.float32),
            "timestamp_ev_sign_disagreement": _numeric(
                population,
                "resid_target_side_archetype_timestamp_ev_sign_disagreement_24h",
                0.0,
            ).to_numpy(dtype=np.float32),
            "persistent_subthreshold_damage": _numeric(
                population,
                "resid_target_side_archetype_persistent_subthreshold_damage_24h",
                0.0,
            ).to_numpy(dtype=np.float32),
            "persistent_material_nontail": _numeric(
                population,
                "resid_target_side_archetype_persistent_material_nontail_24h",
                0.0,
            ).to_numpy(dtype=np.float32),
            "ev_after_1pct": _numeric(population, self.config.ev_col, 0.0).to_numpy(
                dtype=np.float32
            ),
            "bad_mae": _numeric(population, self.config.bad_mae_col, 0.0).to_numpy(
                dtype=np.float32
            ),
            "timeout": _numeric(population, self.config.timeout_col, 0.0).to_numpy(
                dtype=np.float32
            ),
        }
        targets.update(_executable_quality_targets(population, self.config))
        state = fit_ae_gmm_state(
            population.reindex(columns=selected),
            economic_targets=targets,
            random_state=int(seed),
            max_train_rows=int(self.config.ae_gmm_max_train_rows),
            gmm_max_train_rows=int(self.config.gmm_max_train_rows),
            ae_max_iter=int(self.config.ae_gmm_max_iter),
            cluster_candidates=self.config.ae_gmm_clusters,
            reg_covar_candidates=self.config.ae_gmm_reg_covars,
            covariance_type_candidates=self.config.ae_gmm_covariance_types,
            smooth_lambda_candidates=self.config.ae_gmm_smooth_lambdas,
            path_aware_hpo=True,
            temporal_concentration_hpo=True,
            temporal_stability_hpo=True,
            final_refit_all_rows=False,
        )
        if not bool(state.get("enabled", False)):
            return None
        transformed = transform_ae_gmm_features(
            population.reindex(columns=selected),
            state,
            index=population.index,
            prefix=RESIDUAL_EVENT_PREFIX,
        )
        priors = _posterior_priors(transformed, population, self.config)
        screen = screen.copy()
        screen["model_key"] = key
        return _LocalResidualState(
            key=key,
            feature_columns=list(selected),
            ae_gmm_state=state,
            priors=priors,
            support_rows=int(len(population)),
            screening=screen,
            screening_meta=dict(screening_meta),
        )

    def _fit_market_secondary(
        self,
        labelled: pd.DataFrame,
        candidates: Sequence[str],
    ) -> _MarketResidualState | None:
        """Fit the secondary timestamp-level market state after local labels exist."""

        if not bool(self.config.enable_market_secondary):
            return None
        features = [name for name in candidates if _is_market_feature(name)]
        features = features[: int(self.config.market_max_features)]
        if len(features) < 2:
            return None
        ts = pd.to_datetime(
            labelled[self.config.timestamp_col], utc=True, errors="coerce"
        )
        base = labelled.loc[ts.notna(), [self.config.timestamp_col, *features]].copy()
        if base.empty:
            return None
        base[self.config.timestamp_col] = pd.to_datetime(
            base[self.config.timestamp_col], utc=True, errors="coerce"
        )
        panel = (
            base.groupby(self.config.timestamp_col, observed=True, sort=True)[features]
            .mean()
            .reset_index()
        )
        if len(panel) < int(self.config.market_min_rows):
            return None
        event = labelled.loc[ts.notna()].copy()
        event[self.config.timestamp_col] = pd.to_datetime(
            event[self.config.timestamp_col], utc=True, errors="coerce"
        )
        top10 = _numeric(event, "resid_event_top10_population", 0.0).fillna(0.0).gt(0.5)
        event["__negative_event__"] = (
            event["resid_event_class"]
            .astype(str)
            .isin(["negative_residual_event", "adverse_path_event"])
            .astype(np.float32)
        )
        event["__positive_event__"] = (
            event["resid_event_class"]
            .astype(str)
            .isin(["positive_residual_event", "favorable_near_miss_event"])
            .astype(np.float32)
        )
        event["__top10__"] = top10.astype(np.float32)
        target_frame = (
            event.groupby(self.config.timestamp_col, observed=True, sort=True)
            .agg(
                negative_event_pressure=("__negative_event__", "mean"),
                positive_event_pressure=("__positive_event__", "mean"),
                top10_participation=("__top10__", "mean"),
                market_peer_surprise=("resid_event_market_peer_surprise", "mean"),
                ev_after_1pct=(self.config.ev_col, "mean"),
            )
            .reset_index()
        )
        panel = panel.merge(
            target_frame,
            on=self.config.timestamp_col,
            how="inner",
            validate="one_to_one",
        )
        if len(panel) < int(self.config.market_min_rows):
            return None
        state = fit_ae_gmm_state(
            panel.reindex(columns=features),
            economic_targets={
                "negative_event_pressure": _numeric(
                    panel, "negative_event_pressure", 0.0
                ).to_numpy(dtype=np.float32),
                "positive_event_pressure": _numeric(
                    panel, "positive_event_pressure", 0.0
                ).to_numpy(dtype=np.float32),
                "market_peer_surprise": _numeric(
                    panel, "market_peer_surprise", 0.0
                ).to_numpy(dtype=np.float32),
                "ev_after_1pct": _numeric(panel, self.config.ev_col, 0.0).to_numpy(
                    dtype=np.float32
                ),
            },
            random_state=int(self.config.random_state + 97_001),
            max_train_rows=int(self.config.ae_gmm_max_train_rows),
            gmm_max_train_rows=int(self.config.gmm_max_train_rows),
            ae_max_iter=int(self.config.ae_gmm_max_iter),
            cluster_candidates=self.config.ae_gmm_clusters,
            reg_covar_candidates=self.config.ae_gmm_reg_covars,
            covariance_type_candidates=self.config.ae_gmm_covariance_types,
            smooth_lambda_candidates=self.config.ae_gmm_smooth_lambdas,
            path_aware_hpo=True,
            temporal_concentration_hpo=True,
            temporal_stability_hpo=True,
            final_refit_all_rows=False,
        )
        if not bool(state.get("enabled", False)):
            return None
        return _MarketResidualState(
            feature_columns=features,
            ae_gmm_state=state,
            support_rows=int(len(panel)),
        )

    def fit(
        self,
        train: pd.DataFrame,
        *,
        candidate_features: Iterable[str] | None = None,
        market_candidate_features: Iterable[str] | None = None,
    ) -> "ResidualEventArchetypeState":
        self.threshold_state = GlobalEVThresholdState(self.config).fit(train)
        self.expectation_state = ScoreExpectationState(self.config).fit(train)
        self.ev_expectation_state = ScoreExpectationState(
            self.config, target_col=self.config.ev_col, direct_col=""
        ).fit(train)
        raw = add_residual_event_targets(
            train,
            threshold_state=self.threshold_state,
            expectation_state=self.expectation_state,
            ev_expectation_state=self.ev_expectation_state,
        )
        self.baseline_state = ResidualEventBaselineState(self.config).fit(raw)
        labelled = add_residual_event_targets(
            train,
            threshold_state=self.threshold_state,
            expectation_state=self.expectation_state,
            ev_expectation_state=self.ev_expectation_state,
            baseline_state=self.baseline_state,
        )
        labelled = add_residual_state_target_composites(
            labelled,
            timestamp_col=self.config.timestamp_col,
            side_col=self.config.side_col,
            archetype_col=self.config.archetype_col,
        )
        ts = pd.to_datetime(
            labelled[self.config.timestamp_col], utc=True, errors="coerce"
        )
        self.train_start_ = str(ts.min())
        self.train_end_ = str(ts.max())
        candidates = inference_feature_basket(labelled, candidate_features, self.config)
        market_candidates = inference_feature_basket(
            labelled,
            market_candidate_features,
            self.config,
        )
        side, arch, key = _side_arch(labelled, self.config)
        self.side_models = {}
        self.local_models = {}
        feature_rows: list[pd.DataFrame] = []
        catalog_rows: list[dict[str, Any]] = []
        if bool(self.config.allow_side_fallback):
            for offset, (side_name, idx) in enumerate(
                side.groupby(side, sort=True).groups.items()
            ):
                group = labelled.loc[idx]
                if len(group) < int(self.config.min_side_state_rows):
                    continue
                model = self._fit_local(
                    group,
                    candidates,
                    key=f"side::{side_name}",
                    seed=self.config.random_state + offset * 101,
                )
                if model is not None:
                    self.side_models[str(side_name)] = model
                    feature_rows.append(model.screening)
                    catalog_rows.append(
                        {
                            "model_key": model.key,
                            "support_rows": model.support_rows,
                            "feature_count": len(model.feature_columns),
                            "fallback": True,
                            "screening_meta": model.screening_meta,
                            **{
                                f"screen_{name}": value
                                for name, value in model.screening_meta.items()
                            },
                            "state_manifest": model.ae_gmm_state.get("manifest", {}),
                        }
                    )
        for offset, (local, idx) in enumerate(
            key.groupby(key, sort=True).groups.items()
        ):
            group = labelled.loc[idx]
            model = self._fit_local(
                group,
                candidates,
                key=f"local::{local}",
                seed=self.config.random_state + 10_000 + offset * 101,
            )
            if model is not None:
                self.local_models[str(local)] = model
                feature_rows.append(model.screening)
                catalog_rows.append(
                    {
                        "model_key": model.key,
                        "support_rows": model.support_rows,
                        "feature_count": len(model.feature_columns),
                        "fallback": False,
                        "screening_meta": model.screening_meta,
                        **{
                            f"screen_{name}": value
                            for name, value in model.screening_meta.items()
                        },
                        "state_manifest": model.ae_gmm_state.get("manifest", {}),
                    }
                )
        self.feature_metrics_ = (
            pd.concat(feature_rows, ignore_index=True)
            if feature_rows
            else pd.DataFrame()
        )
        self.event_catalog_ = pd.DataFrame(catalog_rows)
        self.market_model = self._fit_market_secondary(labelled, market_candidates)
        return self

    def transform_oos(self, frame: pd.DataFrame) -> pd.DataFrame:
        forbidden = [name for name in OUTCOME_COLUMNS if name in frame.columns]
        if forbidden:
            raise ValueError(
                f"OOS residual-event transform received outcome columns: {sorted(forbidden)}"
            )
        if (
            self.threshold_state is None
            or self.expectation_state is None
            or self.ev_expectation_state is None
        ):
            raise RuntimeError("ResidualEventArchetypeState is not fitted")
        out = pd.DataFrame(
            0.0,
            index=frame.index,
            columns=[
                *residual_event_feature_names(),
                *residual_event_market_feature_names(),
            ],
            dtype=np.float32,
        )
        _side, _arch, key = _side_arch(frame, self.config)
        for local, idx in key.groupby(key, sort=False).groups.items():
            labels = pd.Index(list(idx))
            side_name = (
                str(frame.loc[labels, self.config.side_col].iloc[0]).lower()
                if self.config.side_col in frame
                else "missing"
            )
            model = self.local_models.get(str(local))
            if model is None and bool(self.config.allow_side_fallback):
                model = self.side_models.get(side_name)
            if model is None:
                continue
            part = frame.loc[labels]
            transformed = transform_ae_gmm_features(
                part.reindex(columns=model.feature_columns),
                model.ae_gmm_state,
                index=part.index,
                prefix=RESIDUAL_EVENT_PREFIX,
            )
            for name in _state_static_feature_names():
                if name in transformed:
                    out.loc[labels, name] = (
                        _numeric(transformed, name, 0.0)
                        .fillna(0.0)
                        .to_numpy(dtype=np.float32)
                    )
            posterior_cols = [
                f"{RESIDUAL_EVENT_PREFIX}gmm_cluster_posterior_{i}" for i in range(7)
            ]
            posterior = out.loc[labels, posterior_cols].to_numpy(
                dtype=np.float32, copy=False
            )
            for target, priors in model.priors.items():
                suffix = f"expected_{target}"
                name = f"{RESIDUAL_EVENT_PREFIX}{suffix}"
                if name in out:
                    padded = np.zeros(7, dtype=np.float32)
                    padded[: min(7, len(priors))] = priors[: min(7, len(priors))]
                    out.loc[labels, name] = posterior @ padded
            out.loc[labels, f"{RESIDUAL_EVENT_PREFIX}local_support_log1p"] = np.float32(
                np.log1p(model.support_rows)
            )
            out.loc[labels, f"{RESIDUAL_EVENT_PREFIX}local_model"] = np.float32(
                str(local) in self.local_models
            )
        out = add_residual_event_temporal_context(out, frame, self.config)
        if self.market_model is not None:
            ts = pd.to_datetime(
                frame[self.config.timestamp_col], utc=True, errors="coerce"
            )
            source = frame.loc[
                ts.notna(),
                [self.config.timestamp_col, *self.market_model.feature_columns],
            ].copy()
            source[self.config.timestamp_col] = pd.to_datetime(
                source[self.config.timestamp_col], utc=True, errors="coerce"
            )
            panel = source.groupby(self.config.timestamp_col, observed=True, sort=True)[
                self.market_model.feature_columns
            ].mean()
            if not panel.empty:
                transformed = transform_ae_gmm_features(
                    panel.reindex(columns=self.market_model.feature_columns),
                    self.market_model.ae_gmm_state,
                    index=panel.index,
                    prefix=RESIDUAL_EVENT_MARKET_PREFIX,
                )
                static = transformed.reindex(
                    columns=_state_static_feature_names(RESIDUAL_EVENT_MARKET_PREFIX),
                    fill_value=0.0,
                )
                lookup = pd.DataFrame(
                    {self.config.timestamp_col: ts}, index=frame.index
                ).join(static, on=self.config.timestamp_col, how="left")
                for name in static.columns:
                    out[name] = (
                        _numeric(lookup, name, 0.0)
                        .fillna(0.0)
                        .to_numpy(dtype=np.float32)
                    )
                out[f"{RESIDUAL_EVENT_MARKET_PREFIX}support_log1p"] = np.float32(
                    np.log1p(self.market_model.support_rows)
                )
                out[f"{RESIDUAL_EVENT_MARKET_PREFIX}enabled"] = 1.0
        return out.astype(np.float32, copy=False)

    def transform_oos_with_history(
        self,
        history: pd.DataFrame,
        oos: pd.DataFrame,
    ) -> pd.DataFrame:
        """Transform OOS rows with pre-entry history for temporal-state parity."""

        if history.empty:
            return self.transform_oos(oos)
        for name, source in (("history", history), ("oos", oos)):
            forbidden = [column for column in OUTCOME_COLUMNS if column in source]
            if forbidden:
                raise ValueError(
                    f"Residual-event {name} received outcome columns: {sorted(forbidden)}"
                )
        combined = pd.concat([history, oos], ignore_index=True, copy=False)
        transformed = self.transform_oos(combined)
        result = transformed.iloc[len(history) :].copy()
        result.index = oos.index
        return result.astype(np.float32, copy=False)

    def annotate_outcomes_for_assessment(self, frame: pd.DataFrame) -> pd.DataFrame:
        if (
            self.threshold_state is None
            or self.expectation_state is None
            or self.ev_expectation_state is None
            or self.baseline_state is None
        ):
            raise RuntimeError("ResidualEventArchetypeState is not fitted")
        labelled = add_residual_event_targets(
            frame,
            threshold_state=self.threshold_state,
            expectation_state=self.expectation_state,
            ev_expectation_state=self.ev_expectation_state,
            baseline_state=self.baseline_state,
        )
        labelled = add_residual_state_target_composites(
            labelled,
            timestamp_col=self.config.timestamp_col,
            side_col=self.config.side_col,
            archetype_col=self.config.archetype_col,
        )
        # Assessment-only realized labels let the discovery runner verify that
        # each frozen posterior prior separates its own mechanism OOS.  These
        # columns are never generated by ``transform_oos`` and are explicitly
        # forbidden from inference/meta feature inputs.
        for target, values in _executable_quality_targets(
            labelled, self.config
        ).items():
            labelled[f"{RESIDUAL_EVENT_TARGET_PREFIX}{target}"] = values
        return labelled

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "residual_event_archetype_aegmm_v1",
            "train_start": self.train_start_,
            "train_end": self.train_end_,
            "config": asdict(self.config),
            "threshold_state": self.threshold_state.manifest()
            if self.threshold_state
            else {},
            "local_model_count": int(len(self.local_models)),
            "side_fallback_model_count": int(len(self.side_models)),
            "market_secondary": {
                "enabled": self.market_model is not None,
                "feature_count": len(self.market_model.feature_columns)
                if self.market_model
                else 0,
                "support_rows": self.market_model.support_rows
                if self.market_model
                else 0,
            },
            "generated_features": [
                *residual_event_feature_names(),
                *residual_event_market_feature_names(),
            ],
            "executable_failure_targets": list(EXECUTABLE_FAILURE_TARGETS),
            "target_roles": {
                "top_tail_false_positive": "negative direction and negative net EV",
                "top_tail_clean_cost_fragile": "correct clean direction but negative net EV after costs",
                "top_tail_adverse_loss": "adverse or stop-like path with negative net EV",
                "top_tail_adverse_false_positive": "wrong high-tail call with both adverse path and negative net EV",
                "top_tail_timeout_failure": "slow-resolution execution context; not an automatic reject",
                "top_tail_timeout_loss": "slow-resolution timeout that lost after costs",
                "top_tail_dirty_positive": "positive EV with path ugliness; execution/sizing context",
                "top_tail_dirty_loss": "dirty high-tail path that lost after costs",
                "top_tail_clean_executable": "positive clean executable counterpart",
                "near_tail_clean_executable": "clean positive opportunity in the train-defined top10-to-top20 score band",
                "near_tail_clean_cost_fragile": "correct clean near-tail opportunity that still loses after costs",
                "top_tail_residual_false_positive": "negative-residual high-tail false positive",
                "top_tail_residual_adverse_loss": "negative-residual high-tail adverse loss",
                "top_tail_residual_timeout_loss": "negative-residual high-tail timeout loss",
                "near_tail_positive_residual_clean_executable": "positive-residual clean executable opportunity below the top10 score band",
                "long_mixed_latent_misfire": "negative-residual long mixed high-tail failure, defined without latent inputs",
                "short_mixed_off_manifold": "negative-residual short mixed adverse or wrong high-tail failure, defined without OOD inputs",
                "short_default_latent_uncertainty": "negative-residual short default adverse/timeout high-tail failure, defined without uncertainty inputs",
                "top_tail_reversal_after_initial_success": "initial directional success that later becomes an adverse/dirty negative-EV path",
                "long_mixed_reversal_after_initial_success": "long mixed instance of the initial-success path-reversal failure",
                "short_mixed_reversal_after_initial_success": "short mixed instance of the initial-success path-reversal failure",
                "long_breakout_overconfident_path_loss": "long breakout negative-residual wrong/adverse high-tail loss",
                "short_breakout_overconfident_path_loss": "short breakout negative-residual wrong/adverse high-tail loss",
            },
            "leakage_contract": {
                "event_labels": "realized outcomes only; used for train-side state selection and OOS assessment",
                "thresholds": "global top10/top20 EV targets and local score cutoffs fitted on train only",
                "inference": "frozen local AE/GMM/scalers and train-derived component priors consume pre-entry features only",
                "recent_performance": "excluded from state inputs; 8-day hit-rate smoother is assessment-only",
                "market_drift": "timestamp-neutral labels subtract leave-one-out same-timestamp residual before local daily event detection",
                "secondary_market_layer": (
                    "leave-one-out same-timestamp peer surprise is retained as a separate "
                    "economic target/prior after the primary local neutral-residual target"
                ),
                "partition_routing": (
                    "strict side x archetype by default; side fallback is disabled unless "
                    "the experiment explicitly enables allow_side_fallback"
                ),
            },
        }
