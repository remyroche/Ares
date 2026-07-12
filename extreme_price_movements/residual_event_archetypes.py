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
from .local_economic_aegmm import BASE_DIRECTIONAL_STATE_FEATURES

RESIDUAL_EVENT_PREFIX = "resid_event_aegmm_"
RESIDUAL_EVENT_MARKET_PREFIX = "resid_event_market_aegmm_"
EVENT_CLASSES: tuple[str, ...] = (
    "normal",
    "negative_residual_event",
    "adverse_path_event",
    "positive_residual_event",
    "favorable_near_miss_event",
    "high_variance_event",
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
    ae_gmm_max_train_rows: int = 4_500
    ae_gmm_max_iter: int = 96
    ae_gmm_clusters: tuple[int, ...] = (3, 4, 5, 6)
    ae_gmm_reg_covars: tuple[float, ...] = (1e-4, 1e-3, 3e-3)
    ae_gmm_smooth_lambdas: tuple[float, ...] = (0.0,)
    prior_shrinkage_rows: float = 120.0
    enable_market_secondary: bool = True
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
    ) -> float:
        valid = np.isfinite(score) & np.isfinite(ev)
        if int(valid.sum()) < int(min_rows) or not np.isfinite(target_ev):
            return float("nan")
        s = score[valid].astype(np.float64, copy=False)
        y = ev[valid].astype(np.float64, copy=False)
        grid = np.unique(np.quantile(s, np.linspace(0.50, 0.995, int(grid_size))))
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
            support = int((np.isfinite(g_score) & np.isfinite(g_ev)).sum())
            payload: dict[str, float] = {"support": float(support)}
            for fraction in (self.config.top10_fraction, self.config.top20_fraction):
                name = self._target_name(fraction)
                fallback = float(self.global_thresholds[name])
                threshold = self._threshold_for_target(
                    g_score,
                    g_ev,
                    target_ev=float(self.global_targets[name]),
                    min_rows=max(30, int(self.config.min_local_threshold_rows)),
                    grid_size=int(self.config.threshold_grid_size),
                )
                payload[name] = float(threshold) if np.isfinite(threshold) else fallback
                payload[f"{name}_source"] = 1.0 if np.isfinite(threshold) else 0.0
            # The broader band must contain the local top10 population.
            payload["top20"] = min(float(payload["top20"]), float(payload["top10"]))
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
            "contract": "global score top-k EV target; side x archetype score threshold fitted on train only",
        }


@dataclass
class ScoreExpectationState:
    """Frozen hierarchical calibration of hit probability from the score only."""

    config: ResidualEventArchetypeConfig
    edges: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    global_rates: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )
    local_rates: dict[str, np.ndarray] = field(default_factory=dict)
    local_counts: dict[str, np.ndarray] = field(default_factory=dict)

    def fit(self, train: pd.DataFrame) -> "ScoreExpectationState":
        score = _numeric(train, self.config.score_col).to_numpy(dtype=np.float32)
        hit = _numeric(train, self.config.hit_col).to_numpy(dtype=np.float32)
        valid = np.isfinite(score) & np.isfinite(hit)
        if int(valid.sum()) < int(self.config.min_global_threshold_rows):
            raise ValueError("insufficient train rows for residual expectation state")
        bins = max(4, int(self.config.calibration_bins))
        self.edges = _safe_quantiles(score[valid], np.linspace(0.0, 1.0, bins + 1))
        bucket = np.clip(
            np.searchsorted(self.edges[1:-1], score, side="right"), 0, bins - 1
        )
        global_sum = np.bincount(
            bucket[valid], weights=hit[valid], minlength=bins
        ).astype(np.float64)
        global_count = np.bincount(bucket[valid], minlength=bins).astype(np.float64)
        global_mean = float(np.mean(hit[valid]))
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
            local_hit = hit[pos][mask]
            sums = np.bincount(local_bucket, weights=local_hit, minlength=bins).astype(
                np.float64
            )
            counts = np.bincount(local_bucket, minlength=bins).astype(np.float64)
            local_mean = float(np.mean(local_hit))
            rates = (sums + 12.0 * local_mean) / np.maximum(counts + 12.0, 1.0)
            self.local_rates[str(local)] = rates.astype(np.float32)
            self.local_counts[str(local)] = counts.astype(np.float32)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.Series:
        if self.edges.size == 0 or self.global_rates.size == 0:
            raise RuntimeError("ScoreExpectationState is not fitted")
        # Prefer the frozen model's own probability when it is available.  It
        # is the contemporaneous expectation whose error we want to explain;
        # the train-fitted score calibration below is a compatibility fallback
        # for historical shards that only contain the raw score.
        direct = _numeric(frame, self.config.probability_col)
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
        expected_series.loc[finite_direct] = (
            direct.loc[finite_direct].clip(0.0, 1.0).astype(np.float32)
        )
        return expected_series


@dataclass
class ResidualEventBaselineState:
    """Train-only daily residual baseline for large persistent local events."""

    config: ResidualEventArchetypeConfig
    stats: dict[str, dict[str, float]] = field(default_factory=dict)
    global_stats: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def _aggregate_daily(
        frame: pd.DataFrame, config: ResidualEventArchetypeConfig
    ) -> pd.DataFrame:
        work = frame.copy(deep=False)
        ts = pd.to_datetime(work[config.timestamp_col], utc=True, errors="coerce")
        work["__resid_event_day__"] = ts.dt.floor("D")
        _side, _arch, key = _side_arch(work, config)
        work["__resid_event_key__"] = key.to_numpy(copy=False)
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
            summary["support_weight"] = support_weight.astype(np.float32)
            summary["zone"] = zone
            frames.append(summary)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def fit(self, labelled_train: pd.DataFrame) -> "ResidualEventBaselineState":
        daily = self._aggregate_daily(labelled_train, self.config)
        if daily.empty:
            self.stats = {}
            self.global_stats = {"mean": 0.0, "std": 1.0}
            return self
        self.stats = {}
        for (key, zone), group in daily.groupby(
            ["__resid_event_key__", "zone"], observed=True, sort=True
        ):
            values = pd.to_numeric(group["mean_neutral"], errors="coerce").to_numpy(
                dtype=np.float64
            )
            finite = values[np.isfinite(values)]
            if finite.size < 5:
                continue
            median = float(np.median(finite))
            q25, q75 = np.quantile(finite, [0.25, 0.75])
            mad = float(np.median(np.abs(finite - median)))
            robust_scale = max(
                float((q75 - q25) / 1.349),
                float(1.4826 * mad),
                float(np.std(finite) * 0.25),
                1e-4,
            )
            self.stats[f"{key}|{zone}"] = {
                "mean": median,
                "std": robust_scale,
                "rows": float(finite.size),
            }
        all_values = pd.to_numeric(daily["mean_neutral"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        all_values = all_values[np.isfinite(all_values)]
        if all_values.size:
            global_median = float(np.median(all_values))
            q25, q75 = np.quantile(all_values, [0.25, 0.75])
            global_mad = float(np.median(np.abs(all_values - global_median)))
            global_scale = max(
                float((q75 - q25) / 1.349),
                float(1.4826 * global_mad),
                float(np.std(all_values) * 0.25),
                1e-4,
            )
        else:
            global_median, global_scale = 0.0, 1.0
        self.global_stats = {"mean": global_median, "std": global_scale}
        return self

    def annotate(self, labelled: pd.DataFrame) -> pd.DataFrame:
        out = labelled.copy(deep=False)
        daily = self._aggregate_daily(out, self.config)
        if daily.empty:
            for col in (
                "resid_event_daily_neutral_z",
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
        for idx, (_, row) in enumerate(daily.iterrows()):
            stat = self.stats.get(
                f"{row['__resid_event_key__']}|{row['zone']}", self.global_stats
            )
            z[idx] = np.float32(
                (float(row["mean_neutral"]) - float(stat["mean"]))
                / max(float(stat["std"]), 1e-4)
            )
        daily["resid_event_daily_neutral_z"] = z
        daily = daily.sort_values(
            ["__resid_event_key__", "zone", "__resid_event_day__"], kind="stable"
        )
        daily["__previous_z__"] = daily.groupby(
            ["__resid_event_key__", "zone"], observed=True
        )["resid_event_daily_neutral_z"].shift(1)
        daily["__previous_day__"] = daily.groupby(
            ["__resid_event_key__", "zone"], observed=True
        )["__resid_event_day__"].shift(1)
        contiguous = (
            daily["__resid_event_day__"] - daily["__previous_day__"]
        ).dt.days.eq(1)
        neg = daily["resid_event_daily_neutral_z"].le(
            -float(self.config.event_z_threshold)
        )
        pos = daily["resid_event_daily_neutral_z"].ge(
            float(self.config.event_z_threshold)
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
        extreme = (
            daily["resid_event_daily_neutral_z"]
            .abs()
            .ge(float(self.config.extreme_event_z_threshold))
        )
        persistent = persistent | extreme
        previous = pd.to_numeric(daily["__previous_z__"], errors="coerce").fillna(0.0)
        current = pd.to_numeric(
            daily["resid_event_daily_neutral_z"], errors="coerce"
        ).fillna(0.0)
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
            "resid_event_negative_large",
            "resid_event_positive_large",
            "resid_event_persistent",
            "resid_event_persistence_strength",
            "resid_event_large_event_strength",
        ):
            out[name] = _numeric(out, name, 0.0).fillna(0.0).astype(np.float32)
        top10 = out["__resid_event_zone__"].eq("top10")
        near = out["__resid_event_zone__"].eq("near_miss")
        negative = out["resid_event_negative_large"].gt(0.5) & out[
            "resid_event_persistent"
        ].gt(0.5)
        positive = out["resid_event_positive_large"].gt(0.5) & out[
            "resid_event_persistent"
        ].gt(0.5)
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
            & out["resid_event_daily_neutral_z"]
            .abs()
            .ge(float(self.config.event_z_threshold))
            & ~negative
            & ~positive
        )
        cls[high_var.to_numpy()] = "high_variance_event"
        out["resid_event_class"] = pd.Categorical(cls, categories=EVENT_CLASSES)
        return out.drop(columns=["zone"], errors="ignore")


def add_residual_event_targets(
    frame: pd.DataFrame,
    *,
    threshold_state: GlobalEVThresholdState,
    expectation_state: ScoreExpectationState,
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
    out["resid_event_expected_hit"] = expected
    out["resid_event_global_surprise"] = global_surprise
    # The peer component is the broader market/model-stream problem at the
    # same timestamp.  Local discovery uses the neutral residual first; this
    # market component is retained as a distinct secondary training target so
    # sequential archetype failures can share a common explanation without
    # contaminating the local label.
    out["resid_event_market_peer_surprise"] = peer
    out["resid_event_timestamp_neutral_surprise"] = neutral
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
        rows.append(
            {
                "feature": str(feature),
                "mi_negative": mi_negative,
                "mi_positive": mi_positive,
                "mi_max": max(mi_negative, mi_positive),
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
        metrics["mi_max"] + 0.15 * metrics["lgbm_validation_gain"]
    )
    selected = (
        metrics.sort_values("combined_score", ascending=False, kind="stable")
        .head(int(config.max_features_after_lgbm))["feature"]
        .astype(str)
        .tolist()
    )
    diagnostics = {
        "population_rows": float(population.sum()),
        "candidate_rows": float(len(candidates)),
        "selected_rows": float(len(selected)),
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
        f"{RESIDUAL_EVENT_PREFIX}expected_negative_residual_event",
        f"{RESIDUAL_EVENT_PREFIX}expected_adverse_path_event",
        f"{RESIDUAL_EVENT_PREFIX}expected_positive_residual_event",
        f"{RESIDUAL_EVENT_PREFIX}expected_favorable_near_miss_event",
        f"{RESIDUAL_EVENT_PREFIX}expected_ev_after_1pct",
        f"{RESIDUAL_EVENT_PREFIX}expected_market_peer_surprise",
        f"{RESIDUAL_EVENT_PREFIX}expected_persistence_strength",
        f"{RESIDUAL_EVENT_PREFIX}local_support_log1p",
        f"{RESIDUAL_EVENT_PREFIX}local_model",
    ]
    return list(dict.fromkeys(names))


def residual_event_market_feature_names() -> list[str]:
    names = _state_static_feature_names(RESIDUAL_EVENT_MARKET_PREFIX)
    names += [
        f"{RESIDUAL_EVENT_MARKET_PREFIX}support_log1p",
        f"{RESIDUAL_EVENT_MARKET_PREFIX}enabled",
    ]
    return list(dict.fromkeys(names))


@dataclass
class _LocalResidualState:
    key: str
    feature_columns: list[str]
    ae_gmm_state: dict[str, Any]
    priors: dict[str, np.ndarray]
    support_rows: int
    screening: pd.DataFrame = field(default_factory=pd.DataFrame)


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
        "persistence_strength": _numeric(
            labelled, "resid_event_persistence_strength", 0.0
        )
        .fillna(0.0)
        .to_numpy(dtype=np.float32),
    }
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


@dataclass
class ResidualEventArchetypeState:
    """Frozen side x archetype AE/GMM residual-state bundle."""

    config: ResidualEventArchetypeConfig
    threshold_state: GlobalEVThresholdState | None = None
    expectation_state: ScoreExpectationState | None = None
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
        selected, screen, _meta = screen_local_residual_features(
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
            "market_peer_surprise": _numeric(
                population, "resid_event_market_peer_surprise", 0.0
            ).to_numpy(dtype=np.float32),
            "persistence_strength": _numeric(
                population, "resid_event_persistence_strength", 0.0
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
        state = fit_ae_gmm_state(
            population.reindex(columns=selected),
            economic_targets=targets,
            random_state=int(seed),
            max_train_rows=int(self.config.ae_gmm_max_train_rows),
            gmm_max_train_rows=int(self.config.ae_gmm_max_train_rows),
            ae_max_iter=int(self.config.ae_gmm_max_iter),
            cluster_candidates=self.config.ae_gmm_clusters,
            reg_covar_candidates=self.config.ae_gmm_reg_covars,
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
            gmm_max_train_rows=int(self.config.ae_gmm_max_train_rows),
            ae_max_iter=int(self.config.ae_gmm_max_iter),
            cluster_candidates=self.config.ae_gmm_clusters,
            reg_covar_candidates=self.config.ae_gmm_reg_covars,
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
        self, train: pd.DataFrame, *, candidate_features: Iterable[str] | None = None
    ) -> "ResidualEventArchetypeState":
        self.threshold_state = GlobalEVThresholdState(self.config).fit(train)
        self.expectation_state = ScoreExpectationState(self.config).fit(train)
        raw = add_residual_event_targets(
            train,
            threshold_state=self.threshold_state,
            expectation_state=self.expectation_state,
        )
        self.baseline_state = ResidualEventBaselineState(self.config).fit(raw)
        labelled = add_residual_event_targets(
            train,
            threshold_state=self.threshold_state,
            expectation_state=self.expectation_state,
            baseline_state=self.baseline_state,
        )
        ts = pd.to_datetime(
            labelled[self.config.timestamp_col], utc=True, errors="coerce"
        )
        self.train_start_ = str(ts.min())
        self.train_end_ = str(ts.max())
        candidates = inference_feature_basket(labelled, candidate_features, self.config)
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
                        "state_manifest": model.ae_gmm_state.get("manifest", {}),
                    }
                )
        self.feature_metrics_ = (
            pd.concat(feature_rows, ignore_index=True)
            if feature_rows
            else pd.DataFrame()
        )
        self.event_catalog_ = pd.DataFrame(catalog_rows)
        self.market_model = self._fit_market_secondary(labelled, candidates)
        return self

    def transform_oos(self, frame: pd.DataFrame) -> pd.DataFrame:
        forbidden = [name for name in OUTCOME_COLUMNS if name in frame.columns]
        if forbidden:
            raise ValueError(
                f"OOS residual-event transform received outcome columns: {sorted(forbidden)}"
            )
        if self.threshold_state is None or self.expectation_state is None:
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

    def annotate_outcomes_for_assessment(self, frame: pd.DataFrame) -> pd.DataFrame:
        if (
            self.threshold_state is None
            or self.expectation_state is None
            or self.baseline_state is None
        ):
            raise RuntimeError("ResidualEventArchetypeState is not fitted")
        return add_residual_event_targets(
            frame,
            threshold_state=self.threshold_state,
            expectation_state=self.expectation_state,
            baseline_state=self.baseline_state,
        )

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
