"""Local, economically validated AE/GMM states for base and meta models.

The latent state itself is learned only from pre-entry features. Realized
outcomes are used on training rows to select a stable GMM configuration, name
the resulting states, and estimate shrinkage-stabilized state priors. OOS rows
are transformed by the frozen AE/GMM state and never require outcome columns.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .features_gmm_ae import (
    AE_GMM_MAX_COMPONENTS,
    fit_ae_gmm_state,
    transform_ae_gmm_features,
)
from .meta_cross_sectional_geometry import (
    DEFAULT_RELATIVE_FEATURES,
    geometry_feature_names,
)
from .path_economic_labels import materialize_path_economic_labels

LOCAL_ECONOMIC_AEGMM_PREFIX = "local_econ_aegmm_"

META_MARKET_STATE_FEATURES: tuple[str, ...] = (
    "mkt_median_oi_chg_1h_rz",
    "mkt_median_oi_chg_4h_rz",
    "mkt_pct_oi_chg_1h_rz_lt_minus1",
    "mkt_pct_oi_chg_4h_rz_lt_minus1",
    "mkt_pct_oi_chg_4h_rz_lt_minus2",
    "mkt_pct_oi_drawdown_24h_lt_minus5pct",
    "mkt_median_oi_drawdown_from_peak_24h",
    "mkt_median_oi_recovery_fraction_24h",
    "mkt_median_bars_since_max_oi_drop_24h_norm",
    "mkt_oi_flush_breadth_accel_1h",
    "mkt_oi_flush_breadth_recovery_4h",
    "mkt_pct_price_down_oi_down_1h",
    "mkt_pct_price_down_oi_up_1h",
    "mkt_pct_price_up_oi_down_1h",
    "mkt_pct_price_up_oi_up_1h",
    "mkt_pct_price_down_oi_down_4h",
    "mkt_pct_price_down_oi_up_4h",
    "mkt_pct_price_up_oi_down_4h",
    "mkt_pct_price_up_oi_up_4h",
    "mkt_median_long_flush_intensity_4h",
    "mkt_median_short_build_intensity_4h",
    "mkt_median_short_cover_intensity_1h",
    "market_breadth_chg_1h",
    "market_breadth_accel_1h",
    "market_breadth_recovery_from_6h_min",
    "market_breadth_recovery_from_24h_min",
    "market_breadth_drawdown_from_6h_max",
    "market_pct_recovering_from_24h_low",
    "market_pc1_variance_share_12h",
    "market_pc1_variance_share_24h",
    "market_pc1_variance_share_chg_4h",
    "market_downside_pairwise_corr_24h",
    "market_downside_corr_minus_unconditional_corr_24h",
    "mkt_systemic_deleveraging_score",
    "mkt_flush_exhaustion_score",
    "mkt_leverage_rebuild_score",
)

META_CROSS_SECTIONAL_GEOMETRY_FEATURES: tuple[str, ...] = tuple(
    dict.fromkeys((*DEFAULT_RELATIVE_FEATURES, *geometry_feature_names()))
)

BASE_DIRECTIONAL_STATE_FEATURES: tuple[str, ...] = (
    "loc_range_pos_24",
    "loc_range_pos_48",
    "loc_swing_range_pos_24",
    "trend_slope_48h",
    "range_expansion_ratio",
    "efficiency_ratio_20",
    "oi_drawdown_from_peak_24h",
    "oi_drawdown_from_peak_72h",
    "oi_drawdown_from_peak_168h",
    "oi_recovery_fraction_24h",
    "oi_recovery_fraction_72h",
    "bars_since_oi_low_24h_norm",
    "bars_since_oi_low_72h_norm",
    "bars_since_max_oi_drop_24h_norm",
    "bars_since_max_oi_drop_72h_norm",
    "oi_drop_acceleration_4h_rz",
    "oi_drop_deceleration_4h_rz",
    "price_down_oi_down_1h_rz",
    "price_down_oi_up_1h_rz",
    "price_up_oi_down_1h_rz",
    "price_up_oi_up_1h_rz",
    "price_down_oi_down_4h_rz",
    "price_down_oi_up_4h_rz",
    "price_up_oi_down_4h_rz",
    "price_up_oi_up_4h_rz",
    "price_recovery_fraction_24h",
    "price_recovery_fraction_72h",
    "price_minus_oi_recovery_24h",
    "price_minus_oi_recovery_72h",
    "price_recovery_oi_still_falling_1h",
    "price_recovery_oi_still_falling_4h",
    "funding_positive_to_negative_intensity",
    "funding_negative_to_positive_intensity",
    "funding_crowding_release_4h",
    "downside_deceleration_4h_rz",
    "downside_deceleration_8h_rz",
    "price_recovery_from_low_24h_atr",
    "price_recovery_from_low_72h_atr",
    "bars_since_price_low_24h_norm",
    "bars_since_price_low_72h_norm",
    "volume_climax_decay_4h",
    "range_climax_decay_4h",
    "wick_recovery_intensity",
    "asset_liquidation_phase_score",
    "asset_flush_exhaustion_score",
    "asset_short_covering_score",
)

OUTCOME_OR_DERIVED_COLUMNS = frozenset(
    {
        # This column is computed once on the complete candidate batch during
        # fit.  It defines the true global top-tail labels used only to score
        # train-time state semantics; it is never an AE/GMM input or an OOS
        # requirement.
        "__local_econ_aegmm_global_rank_pct__",
        "__first_touch_target_soft__",
        "__first_touch_policy_soft__",
        "__target_soft__",
        "target_soft",
        "exec_margin",
        "ev_after_1pct",
        "first_touch_gross",
        "first_touch_bad_mae_1r",
        "full_path_bad_mae_1r",
        "timeout",
        "clean_exec",
        "dirty_positive",
        "full_stop_loss",
        "stop_or_adverse",
        "mfe_before_mae_1r",
        "mae_before_mfe_1r",
        "ret_net",
        "u_policy_net",
        "reference_rank_pct",
        "reference_rank_band",
        "hit_surprise",
        "negative_hit_surprise",
        "positive_hit_surprise",
        "negative_tail_label",
        "positive_tail_label",
        "ev_surprise",
    }
)

_GLOBAL_REFERENCE_RANK_COLUMN = "__local_econ_aegmm_global_rank_pct__"

ECONOMIC_PRIOR_NAMES: tuple[str, ...] = (
    "hit_surprise",
    "ev",
    "clean_positive",
    "dirty_positive",
    "bad_mae",
    "timeout",
    "negative_tail",
    "positive_tail",
    "acute_adverse",
    "slow_timeout_loss",
    "clean_negative_ev",
    # State priors must also describe the population that the meta model is
    # ultimately asked to order.  These values are fit only from the global
    # top-10% base-confidence tail at each decision timestamp; non-tail rows
    # are missing rather than treated as failures.
    "top10_hit_surprise",
    "top10_ev",
    "top10_clean_positive",
    "top10_dirty_positive",
    "top10_bad_mae",
    "top10_timeout",
    "top10_negative_tail",
    "top10_acute_adverse",
    "top10_slow_timeout_loss",
    "top10_clean_negative_ev",
)

ECONOMIC_STATE_NAMES: tuple[str, ...] = (
    "clean_high_confidence",
    "dirty_high_confidence",
    "slow_timeout_positive",
    "acute_adverse_false_positive",
    "payoff_mismatch",
    "bad_mae_false_positive",
    "missed_clean_opportunity",
    "high_variance_uncertain",
    "low_edge_noise",
)

_DIRECT_AEGMM_SUFFIXES: tuple[str, ...] = (
    *(f"dae_b16_{idx:02d}" for idx in range(16)),
    *(f"gmm_cluster_posterior_{idx}" for idx in range(AE_GMM_MAX_COMPONENTS)),
    "gmm_cluster_id",
    "gmm_posterior_max",
    "gmm_posterior_margin",
    "gmm_posterior_delta_1",
    "gmm_posterior_accel_1",
    "gmm_entropy",
    "cluster_entropy_norm",
    "cluster_entropy_delta_1",
    "cluster_entropy_accel_1",
    "mahalanobis_distance",
    "min_mahalanobis",
    "min_mahalanobis_delta_1",
    "expected_mahalanobis",
    "expected_mahalanobis_delta_1",
    "expected_mahalanobis_accel_1",
    "cluster_speed",
    "cluster_acceleration",
    "time_since_cluster_change",
    "rolling_cluster_stability",
    "cluster_flip_count_20",
    "AE_reconstruction_error",
    "dae_reconstruction_error",
    "dae_reconstruction_error_zscore",
    "dae_reconstruction_error_delta_1",
    "dae_reconstruction_error_accel_1",
    "latent_mahalanobis_drift",
    "latent_speed",
    "latent_acceleration",
)


def _safe_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9]+", "_", str(value)).strip("_").lower()
    return token or "state"


def _num(frame: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(float(default), index=frame.index, dtype=np.float32)
    return pd.to_numeric(frame[name], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def _canonical_archetype(frame: pd.DataFrame, preferred: str) -> pd.Series:
    for name in (
        preferred,
        "__archetype_policy_key__",
        "archetype_label_family",
        "__archetype_label_family__",
        "policy_archetype",
        "local_side_archetype",
        "source_archetype",
    ):
        if name not in frame.columns:
            continue
        values = frame[name].astype(str).replace({"nan": "", "None": ""})
        if values.str.len().gt(0).any():
            return values.where(values.str.len().gt(0), "missing")
    return pd.Series("missing", index=frame.index, dtype="object")


@dataclass(frozen=True)
class EconomicAEGMMBlock:
    """One observable latent space.

    Timestamp-level blocks describe a market/cross-sectional state. Their fit
    matrix is reduced to one row per timestamp within each side/archetype.
    Asset-level blocks retain individual rows and are suitable for base-layer
    directional state inputs.
    """

    name: str
    features: tuple[str, ...]
    timestamp_level: bool = True


def default_meta_economic_aegmm_blocks() -> tuple[EconomicAEGMMBlock, ...]:
    """Three complementary meta state spaces offered to LightGBM selection."""

    joint = tuple(
        dict.fromkeys(
            (*META_MARKET_STATE_FEATURES, *META_CROSS_SECTIONAL_GEOMETRY_FEATURES)
        )
    )
    return (
        EconomicAEGMMBlock(
            name="market_state",
            features=META_MARKET_STATE_FEATURES,
            timestamp_level=True,
        ),
        EconomicAEGMMBlock(
            name="cross_sectional_geometry",
            features=META_CROSS_SECTIONAL_GEOMETRY_FEATURES,
            timestamp_level=True,
        ),
        EconomicAEGMMBlock(
            name="joint_market_geometry",
            features=joint,
            timestamp_level=True,
        ),
    )


def default_base_economic_aegmm_blocks() -> tuple[EconomicAEGMMBlock, ...]:
    """Asset-local directional state only; no market/trust/context inputs."""

    return (
        EconomicAEGMMBlock(
            name="base_directional_state",
            features=BASE_DIRECTIONAL_STATE_FEATURES,
            timestamp_level=False,
        ),
    )


@dataclass(frozen=True)
class LocalEconomicAEGMMConfig:
    semantic_version: str = "path_first_touch_top10_v4"
    # The state layer is an input to the meta model.  Its train-time economic
    # descriptors must therefore use the frozen base score, never a reference
    # meta prediction that would not exist when the meta model is asked to
    # score a new candidate batch.
    score_col: str = "score_base"
    side_col: str = "side_name"
    archetype_col: str = "archetype_policy_key"
    timestamp_col: str = "__ts__"
    min_side_rows: int = 1_200
    min_local_rows: int = 600
    min_fit_rows: int = 200
    ae_max_train_rows: int = 15_000
    gmm_max_train_rows: int = 100_000
    # Validation keeps a bounded, time-spread fit by default. Once a state
    # design is selected, the final pre-inference refit can set this flag to
    # include every resolved row before the frozen cutoff, preserving rare
    # regimes rather than allowing the cap to discard them.
    full_train_fit: bool = False
    ae_max_iter: int = 80
    cluster_candidates: tuple[int, ...] = (3, 4, 5, 6, 7)
    reg_covar_candidates: tuple[float, ...] = (1e-4, 1e-3, 3e-3)
    # Smoothing across a candidate-table row order is invalid. State dynamics
    # must be materialized separately on a coherent timestamp sequence.
    smooth_lambda_candidates: tuple[float, ...] = (0.0,)
    component_complexity_penalty: float = 0.06
    prior_strength: float = 50.0
    fit_side_fallbacks: bool = True
    fit_local_models: bool = True
    random_state: int = 20260711


@dataclass(frozen=True)
class HierarchicalEconomicAEGMMConfig:
    """Configuration for a shared state geometry with local economic effects.

    Market and cross-sectional states are common observations.  Fitting a
    separate AE/GMM geometry for every side/archetype spends most of the model
    capacity rediscovering the same market event on thin samples.  This
    configuration therefore fits one frozen observable state sequence per
    block, while outcome mappings remain strictly side x archetype specific.

    ``top10_*`` descriptors remain train-only response targets.  OOS rows use
    only the frozen posterior and the identity available at decision time.
    """

    semantic_version: str = "hierarchical_shared_state_top10_v1"
    score_col: str = "score_base"
    side_col: str = "side_name"
    archetype_col: str = "archetype_policy_key"
    timestamp_col: str = "__ts__"
    min_fit_rows: int = 400
    min_response_side_rows: int = 1_200
    min_response_local_rows: int = 600
    min_response_side_tail_rows: int = 80
    min_response_local_tail_rows: int = 40
    ae_max_train_rows: int = 15_000
    gmm_max_train_rows: int = 100_000
    full_train_fit: bool = False
    ae_max_iter: int = 80
    # Shared market states must earn their extra topology.  The old local
    # search frequently selected six or seven sparse components.
    cluster_candidates: tuple[int, ...] = (3, 4, 5)
    reg_covar_candidates: tuple[float, ...] = (1e-4, 1e-3, 3e-3)
    smooth_lambda_candidates: tuple[float, ...] = (0.0,)
    component_complexity_penalty: float = 0.12
    prior_strength: float = 80.0
    random_state: int = 20260712


@dataclass
class _FrozenLocalAEGMM:
    model_key: str
    block_name: str
    input_features: list[str]
    state: dict[str, Any]
    prior_matrix: np.ndarray
    # Persist names with the matrix. Older serialized bundles do not have this
    # attribute; transform falls back to the historical prefix of the registry.
    prior_names: tuple[str, ...]
    semantic_by_cluster: dict[int, str]
    cluster_support: np.ndarray
    support_rows: int
    fit_rows: int
    train_start: str | None
    train_end: str | None
    catalog: list[dict[str, Any]] = field(default_factory=list)


def local_economic_aegmm_feature_names(block_names: Sequence[str]) -> list[str]:
    names: list[str] = []
    for raw_block in block_names:
        prefix = f"{LOCAL_ECONOMIC_AEGMM_PREFIX}{_safe_token(raw_block)}_"
        names.extend(f"{prefix}{suffix}" for suffix in _DIRECT_AEGMM_SUFFIXES)
        names.extend(f"{prefix}prob__{name}" for name in ECONOMIC_STATE_NAMES)
        names.extend(f"{prefix}expected_{name}" for name in ECONOMIC_PRIOR_NAMES)
        names.extend(
            (
                f"{prefix}support_log1p",
                f"{prefix}local_model",
                f"{prefix}enabled",
            )
        )
    return names


def _reference_descriptors(
    frame: pd.DataFrame, config: LocalEconomicAEGMMConfig
) -> pd.DataFrame:
    score = _num(frame, config.score_col, 0.5).fillna(0.5).clip(0.0, 1.0)
    clean = _num(frame, "clean_exec", 0.0).fillna(0.0).clip(0.0, 1.0)
    ev = _num(frame, "ev_after_1pct", 0.0).fillna(0.0).clip(-0.20, 0.20)
    dirty = _num(frame, "dirty_positive", 0.0).fillna(0.0).clip(0.0, 1.0)
    # A high full-path MAE can occur after an otherwise profitable path.  The
    # local state semantics need the first-touch adverse label to distinguish
    # actual stop-like false positives from late path roughness.
    bad_mae = _num(frame, "first_touch_bad_mae_1r", np.nan)
    if bad_mae.isna().all():
        bad_mae = _num(frame, "full_path_bad_mae_1r", 0.0)
    bad_mae = bad_mae.fillna(0.0).clip(0.0, 1.0)
    timeout = _num(frame, "timeout", 0.0).fillna(0.0).clip(0.0, 1.0)
    timestamp = pd.to_datetime(
        frame.get(config.timestamp_col), utc=True, errors="coerce"
    )
    # State models are fitted per side x archetype, but "top 10%" must retain
    # the production candidate-book denominator.  Re-ranking inside a local
    # archetype makes every singleton/tiny local slice appear top-tail and
    # corrupts the train-only cluster semantics.  ``fit`` materializes this
    # column once over the complete candidate batch before partitioning.
    rank = _num(frame, _GLOBAL_REFERENCE_RANK_COLUMN, np.nan)
    if rank.notna().mean() < 0.99:
        fallback = score.groupby(timestamp, sort=False).rank(method="average", pct=True)
        rank = rank.where(rank.notna(), fallback)
    rank = rank.fillna(0.0).clip(0.0, 1.0).astype(np.float32)
    hit_surprise = (clean - score).astype(np.float32)
    top10 = rank.ge(0.90)
    top10_20 = rank.ge(0.80) & ~top10
    path = materialize_path_economic_labels(frame)
    descriptors = pd.DataFrame(
        {
            "score": score,
            "rank_pct": rank,
            "hit_surprise": hit_surprise,
            "negative_hit_surprise": (-hit_surprise).clip(lower=0.0),
            "positive_hit_surprise": hit_surprise.clip(lower=0.0),
            "ev": ev,
            "returns": ev,
            "clean_positive": clean,
            "dirty_positive": dirty,
            "bad_mae": bad_mae,
            "timeout": timeout,
            "negative_tail": (top10 & ((clean < 0.5) | (ev <= 0.0))).astype(np.float32),
            "positive_tail": (top10_20 & (clean >= 0.5) & (ev > 0.0)).astype(
                np.float32
            ),
            "acute_adverse": path["path_label_acute_adverse"].to_numpy(
                dtype=np.float32
            ),
            "slow_timeout_loss": path["path_label_slow_timeout_loss"].to_numpy(
                dtype=np.float32
            ),
            "clean_negative_ev": path["path_label_clean_negative_ev"].to_numpy(
                dtype=np.float32
            ),
        },
        index=frame.index,
    )
    # A timestamp-level state is shared by the candidate book, but its
    # usefulness is decided by the global top-10% base-confidence tail.  Keep
    # tail outcomes as NaN outside that band so timestamp aggregation computes
    # a tail rate/EV, not a diluted all-candidate average.
    for source in (
        "hit_surprise",
        "ev",
        "clean_positive",
        "dirty_positive",
        "bad_mae",
        "timeout",
        "negative_tail",
        "acute_adverse",
        "slow_timeout_loss",
        "clean_negative_ev",
    ):
        descriptors[f"top10_{source}"] = descriptors[source].where(top10, np.nan)
    return descriptors.astype(np.float32)


def _usable_features(frame: pd.DataFrame, requested: Sequence[str]) -> list[str]:
    columns: list[str] = []
    for raw_name in requested:
        name = str(raw_name)
        if name in OUTCOME_OR_DERIVED_COLUMNS or name not in frame.columns:
            continue
        if not (
            pd.api.types.is_numeric_dtype(frame[name]) or frame[name].dtype == bool
        ):
            continue
        values = pd.to_numeric(frame[name], errors="coerce")
        if (
            int(values.notna().sum()) < 20
            or float(values.std(skipna=True) or 0.0) <= 1e-8
        ):
            continue
        columns.append(name)
    return list(dict.fromkeys(columns))


def _timestamp_fit_frame(
    frame: pd.DataFrame,
    features: Sequence[str],
    descriptors: pd.DataFrame,
    timestamp_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    timestamp = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
    valid = timestamp.notna()
    feature_frame = frame.loc[valid, list(features)].apply(
        pd.to_numeric, errors="coerce"
    )
    feature_frame = feature_frame.assign(__fit_ts__=timestamp.loc[valid].to_numpy())
    descriptor_frame = descriptors.loc[valid].assign(
        __fit_ts__=timestamp.loc[valid].to_numpy()
    )
    x_fit = feature_frame.groupby("__fit_ts__", sort=True, observed=True).median(
        numeric_only=True
    )
    y_fit = descriptor_frame.groupby("__fit_ts__", sort=True, observed=True).mean(
        numeric_only=True
    )
    common = x_fit.index.intersection(y_fit.index)
    return x_fit.loc[common], y_fit.loc[common]


def _semantic_for_cluster(metrics: Mapping[str, float]) -> str:
    def tail_or_all(name: str) -> float:
        tail = metrics.get(f"top10_{name}", float("nan"))
        if math.isfinite(float(tail)):
            return float(tail)
        return float(metrics.get(name, 0.0))

    acute_adverse = tail_or_all("acute_adverse")
    bad_mae = tail_or_all("bad_mae")
    timeout = tail_or_all("timeout")
    clean_negative_ev = tail_or_all("clean_negative_ev")
    clean_positive = tail_or_all("clean_positive")
    dirty_positive = tail_or_all("dirty_positive")
    negative_tail = tail_or_all("negative_tail")
    ev = tail_or_all("ev")
    hit_surprise = tail_or_all("hit_surprise")
    if acute_adverse >= 0.18 and bad_mae >= 0.18:
        return "acute_adverse_false_positive"
    if tail_or_all("slow_timeout_loss") >= 0.015 and timeout >= 0.015:
        return "slow_timeout_positive"
    # A state with some clean-but-negative exits is not a payoff-mismatch
    # regime when its aggregate top-tail EV remains positive.  That case is a
    # profitable state with normal intraday noise, and collapsing it into the
    # same semantic bucket as persistent clean losses destroys state diversity.
    if clean_negative_ev >= 0.10 and clean_positive >= 0.50 and ev <= 0.0:
        return "payoff_mismatch"
    if metrics.get("positive_tail", 0.0) >= 0.20 and ev > 0.0:
        return "missed_clean_opportunity"
    if negative_tail >= 0.30:
        if bad_mae >= 0.55:
            return "bad_mae_false_positive"
        if timeout >= 0.12:
            return "slow_timeout_positive"
        if dirty_positive >= 0.45:
            return "dirty_high_confidence"
    if (
        clean_positive >= 0.65
        and ev > 0.0
        and hit_surprise >= 0.0
        and negative_tail <= 0.20
    ):
        return "clean_high_confidence"
    if metrics.get("hit_surprise_std", 0.0) >= 0.35:
        return "high_variance_uncertain"
    return "low_edge_noise"


def _posterior_priors(
    posterior: np.ndarray,
    descriptors: pd.DataFrame,
    *,
    prior_strength: float,
    model_key: str,
) -> tuple[np.ndarray, np.ndarray, dict[int, str], list[dict[str, Any]]]:
    k = int(posterior.shape[1]) if posterior.ndim == 2 else 0
    matrix = np.zeros(
        (AE_GMM_MAX_COMPONENTS, len(ECONOMIC_PRIOR_NAMES)), dtype=np.float32
    )
    support = np.zeros(AE_GMM_MAX_COMPONENTS, dtype=np.float32)
    semantics: dict[int, str] = {}
    catalog: list[dict[str, Any]] = []
    values = descriptors.reindex(columns=list(ECONOMIC_PRIOR_NAMES)).to_numpy(
        dtype=np.float32
    )
    finite_values = np.isfinite(values)
    global_mean = np.divide(
        np.nansum(values, axis=0),
        np.maximum(np.sum(finite_values, axis=0), 1),
    ).astype(np.float32)
    global_mean = np.nan_to_num(global_mean, nan=0.0)
    surprise = descriptors["hit_surprise"].to_numpy(dtype=np.float32)
    for cluster in range(k):
        weight = posterior[:, cluster].astype(np.float32)
        finite = np.isfinite(values)
        denominator = np.sum(weight[:, None] * finite, axis=0)
        numerator = np.nansum(weight[:, None] * values, axis=0)
        raw = np.divide(
            numerator,
            np.maximum(denominator, 1e-6),
            out=global_mean.copy(),
            where=denominator > 0.0,
        ).astype(np.float32)
        mass = float(np.sum(weight))
        shrunk = (
            np.float32(mass) * raw + np.float32(prior_strength) * global_mean
        ) / np.float32(max(mass + prior_strength, 1e-6))
        matrix[cluster] = shrunk
        support[cluster] = np.float32(mass)
        metrics = {
            name: float(shrunk[idx]) for idx, name in enumerate(ECONOMIC_PRIOR_NAMES)
        }
        surprise_mean = float(np.sum(weight * surprise) / max(mass, 1e-6))
        surprise_var = float(
            np.sum(weight * np.square(surprise - surprise_mean)) / max(mass, 1e-6)
        )
        metrics["hit_surprise_std"] = math.sqrt(max(surprise_var, 0.0))
        semantic = _semantic_for_cluster(metrics)
        semantics[cluster] = semantic
        catalog.append(
            {
                "model_key": model_key,
                "cluster": cluster,
                "semantic": semantic,
                "posterior_support": mass,
                **metrics,
            }
        )
    return matrix, support, semantics, catalog


def _fit_one_model(
    frame: pd.DataFrame,
    block: EconomicAEGMMBlock,
    config: LocalEconomicAEGMMConfig,
    *,
    model_key: str,
    seed: int,
) -> _FrozenLocalAEGMM | None:
    features = _usable_features(frame, block.features)
    if len(features) < 2:
        return None
    descriptors = _reference_descriptors(frame, config)
    if block.timestamp_level:
        x_fit, y_fit = _timestamp_fit_frame(
            frame, features, descriptors, config.timestamp_col
        )
    else:
        order = (
            pd.to_datetime(frame[config.timestamp_col], utc=True, errors="coerce")
            .sort_values(kind="stable")
            .index
        )
        x_fit = frame.loc[order, features].apply(pd.to_numeric, errors="coerce")
        y_fit = descriptors.loc[order]
    if len(x_fit) < int(config.min_fit_rows):
        return None
    economic_targets = {
        name: y_fit[name].to_numpy(dtype=np.float32)
        for name in y_fit.columns
        if name != "score" and name != "rank_pct"
    }
    # Temporal concentration is a configuration-quality constraint, not an
    # economic target or posterior prior. Keep it private so GMM HPO can
    # reject one-era clusters without treating calendar time as state meaning.
    if block.timestamp_level:
        fit_timestamp = pd.DatetimeIndex(
            pd.to_datetime(x_fit.index, utc=True, errors="coerce")
        )
    else:
        fit_timestamp = pd.DatetimeIndex(
            pd.to_datetime(
                frame.loc[order, config.timestamp_col], utc=True, errors="coerce"
            )
        )
    time_bucket = np.full(len(fit_timestamp), np.nan, dtype=np.float32)
    valid_timestamp = ~fit_timestamp.isna()
    if bool(valid_timestamp.any()):
        time_bucket[valid_timestamp] = (
            fit_timestamp.asi8[valid_timestamp]
            // np.int64(7 * 24 * 60 * 60 * 1_000_000_000)
        ).astype(np.float32, copy=False)
    economic_targets["_time_bucket"] = time_bucket
    state = fit_ae_gmm_state(
        x_fit,
        economic_targets=economic_targets,
        random_state=int(seed),
        # HPO stays on an evenly time-spread sample.  ``full_train_fit``
        # performs one selected-configuration refit on every resolved row;
        # it must not expand every HPO candidate to the full data set.
        max_train_rows=min(int(config.ae_max_train_rows), len(x_fit)),
        gmm_max_train_rows=min(int(config.gmm_max_train_rows), len(x_fit)),
        ae_max_iter=int(config.ae_max_iter),
        cluster_candidates=config.cluster_candidates,
        reg_covar_candidates=config.reg_covar_candidates,
        smooth_lambda_candidates=config.smooth_lambda_candidates,
        require_both_sides=False,
        path_aware_hpo=True,
        temporal_concentration_hpo=True,
        temporal_stability_hpo=bool(block.timestamp_level),
        component_complexity_penalty=float(config.component_complexity_penalty),
        final_refit_all_rows=bool(config.full_train_fit),
    )
    if not state.get("enabled", False):
        return None
    generated = transform_ae_gmm_features(x_fit, state, index=x_fit.index)
    posterior = generated[
        [
            f"gmm_cluster_posterior_{idx}"
            for idx in range(int(state["gmm_n_components"]))
        ]
    ].to_numpy(dtype=np.float32, copy=False)
    prior_matrix, cluster_support, semantics, catalog = _posterior_priors(
        posterior,
        y_fit,
        prior_strength=float(config.prior_strength),
        model_key=model_key,
    )
    for row in catalog:
        row["block"] = block.name
    timestamp = pd.to_datetime(frame[config.timestamp_col], utc=True, errors="coerce")
    return _FrozenLocalAEGMM(
        model_key=model_key,
        block_name=block.name,
        input_features=features,
        state=state,
        prior_matrix=prior_matrix,
        prior_names=ECONOMIC_PRIOR_NAMES,
        semantic_by_cluster=semantics,
        cluster_support=cluster_support,
        support_rows=int(len(frame)),
        fit_rows=int(len(x_fit)),
        train_start=str(timestamp.min()),
        train_end=str(timestamp.max()),
        catalog=catalog,
    )


def _posterior_response_matrix(
    posterior: np.ndarray,
    descriptors: pd.DataFrame,
    *,
    fallback: np.ndarray,
    prior_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit posterior-weighted response means with a parent-state fallback.

    The calculation is deliberately vectorized because it runs once for every
    side/archetype response table.  Missing tail descriptors do not count as a
    negative observation; their posterior mass is excluded per target.
    """

    values = descriptors.reindex(columns=list(ECONOMIC_PRIOR_NAMES)).to_numpy(
        dtype=np.float32, copy=False
    )
    weight = np.asarray(posterior, dtype=np.float32)
    if weight.ndim != 2:
        raise ValueError("Posterior response fitting requires a 2D posterior matrix")
    if values.shape[0] != weight.shape[0]:
        raise ValueError("Posterior/descriptors row mismatch")
    fallback_arr = np.asarray(fallback, dtype=np.float32)
    if fallback_arr.shape != (weight.shape[1], values.shape[1]):
        raise ValueError("Parent response matrix shape does not match posterior")
    finite = np.isfinite(values)
    cleaned = np.where(finite, values, 0.0).astype(np.float32, copy=False)
    mass = weight.T @ finite.astype(np.float32, copy=False)
    numerator = weight.T @ cleaned
    raw = np.divide(
        numerator,
        np.maximum(mass, 1e-6),
        out=fallback_arr.copy(),
        where=mass > 0.0,
    )
    strength = np.float32(max(float(prior_strength), 0.0))
    shrunk = (mass * raw + strength * fallback_arr) / np.maximum(mass + strength, 1e-6)
    support = weight.sum(axis=0, dtype=np.float32)
    return shrunk.astype(np.float32, copy=False), support.astype(np.float32, copy=False)


def _global_response_fallback(
    descriptors: pd.DataFrame, component_count: int
) -> np.ndarray:
    values = descriptors.reindex(columns=list(ECONOMIC_PRIOR_NAMES)).to_numpy(
        dtype=np.float32, copy=False
    )
    finite = np.isfinite(values)
    mean = np.divide(
        np.nansum(values, axis=0),
        np.maximum(finite.sum(axis=0), 1),
    )
    mean = np.nan_to_num(mean, nan=0.0).astype(np.float32, copy=False)
    return np.repeat(mean[None, :], int(component_count), axis=0).astype(
        np.float32, copy=False
    )


def _response_catalog_rows(
    *,
    block_name: str,
    scope: str,
    side: str,
    archetype: str,
    matrix: np.ndarray,
    posterior_support: np.ndarray,
    response_rows: int,
    tail_rows: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cluster in range(int(matrix.shape[0])):
        metrics = {
            name: float(matrix[cluster, index])
            for index, name in enumerate(ECONOMIC_PRIOR_NAMES)
        }
        rows.append(
            {
                "model_key": f"shared_response::{scope}::{side}::{archetype}::{block_name}",
                "block": block_name,
                "scope": scope,
                "side_name": side,
                "archetype_policy_key": archetype,
                "cluster": int(cluster),
                "semantic": _semantic_for_cluster(metrics),
                "posterior_support": float(posterior_support[cluster]),
                "response_rows": int(response_rows),
                "response_tail_rows": int(tail_rows),
                **metrics,
            }
        )
    return rows


@dataclass
class _HierarchicalStateResponses:
    """Train-only local outcome maps for one shared AE/GMM state block."""

    block_name: str
    global_matrix: np.ndarray
    global_support: np.ndarray
    side_matrices: dict[str, np.ndarray]
    side_support: dict[str, float]
    side_semantics: dict[str, dict[int, str]]
    local_matrices: dict[tuple[str, str], np.ndarray]
    local_support: dict[tuple[str, str], float]
    local_semantics: dict[tuple[str, str], dict[int, str]]
    catalog: list[dict[str, Any]]

    @staticmethod
    def _semantics(matrix: np.ndarray) -> dict[int, str]:
        return {
            int(cluster): _semantic_for_cluster(
                {
                    name: float(matrix[cluster, index])
                    for index, name in enumerate(ECONOMIC_PRIOR_NAMES)
                }
            )
            for cluster in range(int(matrix.shape[0]))
        }

    @classmethod
    def fit(
        cls,
        *,
        block_name: str,
        frame: pd.DataFrame,
        posterior: np.ndarray,
        descriptors: pd.DataFrame,
        config: HierarchicalEconomicAEGMMConfig,
    ) -> "_HierarchicalStateResponses":
        component_count = int(posterior.shape[1])
        fallback = _global_response_fallback(descriptors, component_count)
        global_matrix, global_support = _posterior_response_matrix(
            posterior,
            descriptors,
            fallback=fallback,
            prior_strength=float(config.prior_strength),
        )
        global_tail_rows = int(np.isfinite(descriptors["top10_ev"]).sum())
        catalog = _response_catalog_rows(
            block_name=block_name,
            scope="global",
            side="all",
            archetype="all",
            matrix=global_matrix,
            posterior_support=global_support,
            response_rows=len(frame),
            tail_rows=global_tail_rows,
        )
        side_values = frame[config.side_col].astype(str).str.lower().to_numpy()
        archetype_values = (
            _canonical_archetype(frame, config.archetype_col).astype(str).to_numpy()
        )
        groups = (
            pd.DataFrame({"side": side_values, "archetype": archetype_values})
            .groupby(["side", "archetype"], sort=True, observed=True)
            .indices
        )
        side_matrices: dict[str, np.ndarray] = {}
        side_support: dict[str, float] = {}
        side_semantics: dict[str, dict[int, str]] = {}
        local_matrices: dict[tuple[str, str], np.ndarray] = {}
        local_support: dict[tuple[str, str], float] = {}
        local_semantics: dict[tuple[str, str], dict[int, str]] = {}

        for side in sorted(pd.unique(side_values)):
            positions = np.flatnonzero(side_values == side)
            side_desc = descriptors.iloc[positions]
            tail_rows = int(np.isfinite(side_desc["top10_ev"]).sum())
            if len(positions) < int(config.min_response_side_rows) or tail_rows < int(
                config.min_response_side_tail_rows
            ):
                continue
            matrix, support = _posterior_response_matrix(
                posterior[positions],
                side_desc,
                fallback=global_matrix,
                prior_strength=float(config.prior_strength),
            )
            side_key = str(side)
            side_matrices[side_key] = matrix
            side_support[side_key] = float(len(positions))
            side_semantics[side_key] = cls._semantics(matrix)
            catalog.extend(
                _response_catalog_rows(
                    block_name=block_name,
                    scope="side",
                    side=side_key,
                    archetype="all",
                    matrix=matrix,
                    posterior_support=support,
                    response_rows=len(positions),
                    tail_rows=tail_rows,
                )
            )

        for (side, archetype), raw_positions in groups.items():
            positions = np.asarray(raw_positions, dtype=np.int64)
            local_desc = descriptors.iloc[positions]
            tail_rows = int(np.isfinite(local_desc["top10_ev"]).sum())
            if len(positions) < int(config.min_response_local_rows) or tail_rows < int(
                config.min_response_local_tail_rows
            ):
                continue
            side_key = str(side)
            archetype_key = str(archetype)
            parent = side_matrices.get(side_key, global_matrix)
            matrix, support = _posterior_response_matrix(
                posterior[positions],
                local_desc,
                fallback=parent,
                prior_strength=float(config.prior_strength),
            )
            key = (side_key, archetype_key)
            local_matrices[key] = matrix
            local_support[key] = float(len(positions))
            local_semantics[key] = cls._semantics(matrix)
            catalog.extend(
                _response_catalog_rows(
                    block_name=block_name,
                    scope="local",
                    side=side_key,
                    archetype=archetype_key,
                    matrix=matrix,
                    posterior_support=support,
                    response_rows=len(positions),
                    tail_rows=tail_rows,
                )
            )
        return cls(
            block_name=block_name,
            global_matrix=global_matrix,
            global_support=global_support,
            side_matrices=side_matrices,
            side_support=side_support,
            side_semantics=side_semantics,
            local_matrices=local_matrices,
            local_support=local_support,
            local_semantics=local_semantics,
            catalog=catalog,
        )

    def apply(
        self,
        *,
        frame: pd.DataFrame,
        generated: pd.DataFrame,
        config: HierarchicalEconomicAEGMMConfig,
    ) -> pd.DataFrame:
        """Replace shared priors with frozen local-response priors on OOS rows."""

        prefix = f"{LOCAL_ECONOMIC_AEGMM_PREFIX}{_safe_token(self.block_name)}_"
        posterior_columns = [
            f"{prefix}gmm_cluster_posterior_{index}"
            for index in range(AE_GMM_MAX_COMPONENTS)
        ]
        posterior = generated.reindex(
            columns=posterior_columns, fill_value=0.0
        ).to_numpy(dtype=np.float32, copy=False)
        row_sum = posterior.sum(axis=1, keepdims=True)
        posterior = np.divide(
            posterior,
            np.maximum(row_sum, 1e-8),
            out=np.full_like(posterior, 1.0 / max(posterior.shape[1], 1)),
            where=row_sum > 1e-8,
        )
        side_values = frame[config.side_col].astype(str).str.lower().to_numpy()
        archetype_values = (
            _canonical_archetype(frame, config.archetype_col).astype(str).to_numpy()
        )
        groups = (
            pd.DataFrame({"side": side_values, "archetype": archetype_values})
            .groupby(["side", "archetype"], sort=False, observed=True)
            .indices
        )
        local_flag = np.zeros(len(frame), dtype=np.float32)
        support = np.zeros(len(frame), dtype=np.float32)
        semantic_probability = np.zeros(
            (len(frame), len(ECONOMIC_STATE_NAMES)), dtype=np.float32
        )
        expected = np.zeros((len(frame), len(ECONOMIC_PRIOR_NAMES)), dtype=np.float32)
        for (side, archetype), raw_positions in groups.items():
            positions = np.asarray(raw_positions, dtype=np.int64)
            key = (str(side), str(archetype))
            matrix = self.local_matrices.get(key)
            semantics = self.local_semantics.get(key)
            if matrix is not None:
                local_flag[positions] = 1.0
                support[positions] = np.float32(self.local_support.get(key, 0.0))
            else:
                matrix = self.side_matrices.get(str(side), self.global_matrix)
                semantics = self.side_semantics.get(str(side))
                support[positions] = np.float32(
                    self.side_support.get(str(side), float(self.global_support.sum()))
                )
            expected[positions] = posterior[positions] @ matrix
            if semantics is None:
                semantics = self._semantics(self.global_matrix)
            for cluster, semantic in semantics.items():
                if int(cluster) >= posterior.shape[1]:
                    continue
                semantic_probability[
                    positions, ECONOMIC_STATE_NAMES.index(semantic)
                ] += posterior[positions, int(cluster)]
        for index, name in enumerate(ECONOMIC_PRIOR_NAMES):
            generated[f"{prefix}expected_{name}"] = expected[:, index]
        for index, name in enumerate(ECONOMIC_STATE_NAMES):
            generated[f"{prefix}prob__{name}"] = semantic_probability[:, index]
        generated[f"{prefix}support_log1p"] = np.log1p(support).astype(np.float32)
        generated[f"{prefix}local_model"] = local_flag
        generated[f"{prefix}enabled"] = np.float32(1.0)
        return generated


@dataclass
class HierarchicalEconomicAEGMM:
    """Shared observable state geometry with side x archetype response maps.

    The state representation is learned once per block from all eligible train
    timestamps.  Outcome effects are then fitted independently per
    side/archetype and shrink through side to global parents.  This keeps state
    identity stable while preserving the fact that one market state can be
    favorable for one archetype and adverse for another.
    """

    config: HierarchicalEconomicAEGMMConfig
    blocks: tuple[EconomicAEGMMBlock, ...]
    shared_models: dict[str, _FrozenLocalAEGMM] = field(default_factory=dict)
    responses: dict[str, _HierarchicalStateResponses] = field(default_factory=dict)
    catalog_: pd.DataFrame = field(default_factory=pd.DataFrame)
    train_start_: str | None = None
    train_end_: str | None = None

    def fit(self, train: pd.DataFrame) -> "HierarchicalEconomicAEGMM":
        missing = [
            name
            for name in (self.config.timestamp_col, self.config.side_col)
            if name not in train.columns
        ]
        if missing:
            raise ValueError(
                f"Hierarchical AE/GMM training frame is missing columns: {missing}"
            )
        timestamp = pd.to_datetime(
            train[self.config.timestamp_col], utc=True, errors="coerce"
        )
        self.train_start_ = str(timestamp.min())
        self.train_end_ = str(timestamp.max())
        score = _num(train, self.config.score_col, 0.5).fillna(0.5).clip(0.0, 1.0)
        global_rank = (
            score.groupby(timestamp, sort=False)
            .rank(method="average", pct=True)
            .fillna(0.0)
            .astype(np.float32)
        )
        fit_frame = train.copy(deep=False)
        fit_frame[_GLOBAL_REFERENCE_RANK_COLUMN] = global_rank.to_numpy(
            dtype=np.float32, copy=False
        )
        self.shared_models = {}
        self.responses = {}
        catalog: list[dict[str, Any]] = []
        for block_index, block in enumerate(self.blocks):
            model = _fit_one_model(
                fit_frame,
                block,
                self.config,  # type: ignore[arg-type]
                model_key=f"shared::{block.name}",
                seed=int(self.config.random_state + block_index * 10_003),
            )
            if model is None:
                continue
            direct = LocalEconomicAEGMM._transform_timestamp_level_model(
                fit_frame,
                model,
                block_name=block.name,
                local=False,
                timestamp_col=self.config.timestamp_col,
            ).reindex(fit_frame.index, fill_value=0.0)
            prefix = f"{LOCAL_ECONOMIC_AEGMM_PREFIX}{_safe_token(block.name)}_"
            posterior = direct.reindex(
                columns=[
                    f"{prefix}gmm_cluster_posterior_{index}"
                    for index in range(AE_GMM_MAX_COMPONENTS)
                ],
                fill_value=0.0,
            ).to_numpy(dtype=np.float32, copy=False)
            descriptors = _reference_descriptors(fit_frame, self.config)  # type: ignore[arg-type]
            response = _HierarchicalStateResponses.fit(
                block_name=block.name,
                frame=fit_frame,
                posterior=posterior,
                descriptors=descriptors,
                config=self.config,
            )
            self.shared_models[block.name] = model
            self.responses[block.name] = response
            catalog.extend(model.catalog)
            catalog.extend(response.catalog)
        self.catalog_ = pd.DataFrame(catalog)
        return self

    def transform_oos(self, oos: pd.DataFrame) -> pd.DataFrame:
        forbidden = sorted(OUTCOME_OR_DERIVED_COLUMNS.intersection(oos.columns))
        if forbidden:
            raise ValueError(
                f"OOS hierarchical AE/GMM transform received outcomes: {forbidden[:12]}"
            )
        output = pd.DataFrame(
            0.0,
            index=oos.index,
            columns=local_economic_aegmm_feature_names(
                [block.name for block in self.blocks]
            ),
            dtype=np.float32,
        )
        for block in self.blocks:
            model = self.shared_models.get(block.name)
            response = self.responses.get(block.name)
            if model is None or response is None:
                continue
            if not block.timestamp_level:
                direct = LocalEconomicAEGMM._transform_model(
                    oos, model, block_name=block.name, local=False
                )
            else:
                direct = LocalEconomicAEGMM._transform_timestamp_level_model(
                    oos,
                    model,
                    block_name=block.name,
                    local=False,
                    timestamp_col=self.config.timestamp_col,
                )
            if direct.empty:
                continue
            full = direct.reindex(oos.index, fill_value=0.0).astype(
                np.float32, copy=False
            )
            response.apply(frame=oos, generated=full, config=self.config)
            output.loc[:, full.columns] = full.to_numpy(dtype=np.float32, copy=False)
        return output.astype(np.float32, copy=False)

    def transform_train(self, train: pd.DataFrame) -> pd.DataFrame:
        safe = train.drop(
            columns=[
                name for name in OUTCOME_OR_DERIVED_COLUMNS if name in train.columns
            ],
            errors="ignore",
        )
        return self.transform_oos(safe)

    def transform_oos_with_history(
        self,
        history: pd.DataFrame,
        oos: pd.DataFrame,
    ) -> pd.DataFrame:
        """Transform OOS rows after observable context for dynamic state parity.

        AE/GMM posterior deltas, speed, and acceleration are sequence features.
        The prior bars are usable at a new decision timestamp even when their
        trade outcomes are embargoed from state-response fitting.  This helper
        accepts only pre-entry history, applies the frozen transform across the
        concatenated sequence, and returns the target OOS suffix unchanged.
        """

        if history.empty:
            return self.transform_oos(oos)
        for name, frame in (("history", history), ("oos", oos)):
            forbidden = sorted(OUTCOME_OR_DERIVED_COLUMNS.intersection(frame.columns))
            if forbidden:
                raise ValueError(
                    f"Hierarchical AE/GMM {name} received outcomes: {forbidden[:12]}"
                )
        required = [self.config.timestamp_col, self.config.side_col]
        for name in required:
            if name not in history.columns or name not in oos.columns:
                raise ValueError(
                    f"Hierarchical AE/GMM history transform requires {name!r}"
                )
        combined = pd.concat([history, oos], ignore_index=True, copy=False)
        timestamp = pd.to_datetime(
            combined[self.config.timestamp_col], utc=True, errors="coerce"
        )
        # Preserve a deterministic order inside a timestamp while ensuring no
        # target row is evaluated before its observable history.
        combined = combined.assign(
            __hierarchical_order__=np.arange(len(combined), dtype=np.int64)
        )
        combined = combined.assign(
            __hierarchical_ts__=timestamp.to_numpy()
        ).sort_values(["__hierarchical_ts__", "__hierarchical_order__"], kind="stable")
        transformed = self.transform_oos(
            combined.drop(columns=["__hierarchical_order__", "__hierarchical_ts__"])
        )
        transformed["__hierarchical_order__"] = combined[
            "__hierarchical_order__"
        ].to_numpy()
        target = transformed.loc[
            transformed["__hierarchical_order__"].ge(len(history))
        ].sort_values("__hierarchical_order__", kind="stable")
        target = target.drop(columns="__hierarchical_order__")
        target.index = oos.index
        return target.astype(np.float32, copy=False)

    def required_input_features(self) -> list[str]:
        required = [
            self.config.side_col,
            self.config.archetype_col,
            self.config.timestamp_col,
        ]
        for model in self.shared_models.values():
            required.extend(model.input_features)
        return list(dict.fromkeys(required))

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "hierarchical_economic_aegmm_v1",
            "semantic_version": self.config.semantic_version,
            "train_start": self.train_start_,
            "train_end": self.train_end_,
            "blocks": [block.name for block in self.blocks],
            "shared_model_count": int(len(self.shared_models)),
            "local_response_group_count": int(
                sum(
                    len(response.local_matrices) for response in self.responses.values()
                )
            ),
            "side_response_group_count": int(
                sum(len(response.side_matrices) for response in self.responses.values())
            ),
            "cluster_candidates": list(self.config.cluster_candidates),
            "full_train_fit": bool(self.config.full_train_fit),
            "fit_sample_contract": (
                "AE/GMM HPO uses an evenly time-spread train-only sample; the selected "
                "shared state may be refit on all resolved rows before the cutoff. "
                "Side/archetype response priors use train outcomes only and OOS transforms "
                "use frozen posteriors plus side/archetype identity only."
            ),
            "models": [
                {
                    "block": name,
                    "model_key": model.model_key,
                    "support_rows": int(model.support_rows),
                    "fit_rows": int(model.fit_rows),
                    "components": int(model.state.get("gmm_n_components", 0)),
                    "response_local_groups": int(
                        len(self.responses[name].local_matrices)
                    ),
                    "response_side_groups": int(
                        len(self.responses[name].side_matrices)
                    ),
                }
                for name, model in self.shared_models.items()
            ],
            "leakage_contract": (
                "State inputs are observable pre-entry features. AE/GMM, response priors, "
                "and support estimates are fitted only on train rows. OOS transformation "
                "rejects outcome columns and never refits state geometry."
            ),
        }


@dataclass
class LocalEconomicAEGMM:
    """Frozen side/archetype-local AE/GMM features for direct model use."""

    config: LocalEconomicAEGMMConfig
    blocks: tuple[EconomicAEGMMBlock, ...]
    side_models: dict[tuple[str, str], _FrozenLocalAEGMM] = field(default_factory=dict)
    local_models: dict[tuple[str, str, str], _FrozenLocalAEGMM] = field(
        default_factory=dict
    )
    catalog_: pd.DataFrame = field(default_factory=pd.DataFrame)
    train_start_: str | None = None
    train_end_: str | None = None

    def fit(self, train: pd.DataFrame) -> "LocalEconomicAEGMM":
        missing = [
            name
            for name in (self.config.timestamp_col, self.config.side_col)
            if name not in train.columns
        ]
        if missing:
            raise ValueError(
                f"Local AE/GMM training frame is missing columns: {missing}"
            )
        timestamp = pd.to_datetime(
            train[self.config.timestamp_col], utc=True, errors="coerce"
        )
        self.train_start_ = str(timestamp.min())
        self.train_end_ = str(timestamp.max())
        score = _num(train, self.config.score_col, 0.5).fillna(0.5).clip(0.0, 1.0)
        # Compute the rank before any side/archetype partitioning. This is the
        # global candidate-book rank at a decision timestamp, matching the
        # top-10% decision context used by the meta/policy stack.
        global_rank = (
            score.groupby(timestamp, sort=False)
            .rank(method="average", pct=True)
            .fillna(0.0)
            .astype(np.float32)
        )
        fit_frame = train.copy(deep=False)
        fit_frame[_GLOBAL_REFERENCE_RANK_COLUMN] = global_rank.to_numpy(
            dtype=np.float32, copy=False
        )
        side = fit_frame[self.config.side_col].astype(str).str.lower()
        archetype = _canonical_archetype(fit_frame, self.config.archetype_col).astype(
            str
        )
        catalog: list[dict[str, Any]] = []
        self.side_models = {}
        self.local_models = {}
        for block_index, block in enumerate(self.blocks):
            if self.config.fit_side_fallbacks:
                for side_key, index in train.groupby(side, sort=True).groups.items():
                    group = fit_frame.loc[index]
                    if len(group) < int(self.config.min_side_rows):
                        continue
                    model = _fit_one_model(
                        group,
                        block,
                        self.config,
                        model_key=f"side::{side_key}::{block.name}",
                        seed=self.config.random_state
                        + block_index * 10_000
                        + len(self.side_models) * 101,
                    )
                    if model is not None:
                        self.side_models[(block.name, str(side_key))] = model
                        catalog.extend(model.catalog)
            if self.config.fit_local_models:
                keys = pd.DataFrame(
                    {"side": side, "archetype": archetype}, index=train.index
                )
                groups = keys.groupby(
                    ["side", "archetype"], observed=True, sort=True
                ).groups
                for (side_key, archetype_key), index in groups.items():
                    group = fit_frame.loc[index]
                    if len(group) < int(self.config.min_local_rows):
                        continue
                    model = _fit_one_model(
                        group,
                        block,
                        self.config,
                        model_key=f"local::{side_key}::{archetype_key}::{block.name}",
                        seed=(
                            self.config.random_state
                            + block_index * 10_000
                            + len(self.local_models) * 137
                            + 17
                        ),
                    )
                    if model is not None:
                        self.local_models[
                            (block.name, str(side_key), str(archetype_key))
                        ] = model
                        catalog.extend(model.catalog)
        self.catalog_ = pd.DataFrame(catalog)
        return self

    @staticmethod
    def _transform_model(
        frame: pd.DataFrame,
        model: _FrozenLocalAEGMM,
        *,
        block_name: str,
        local: bool,
    ) -> pd.DataFrame:
        prefix = f"{LOCAL_ECONOMIC_AEGMM_PREFIX}{_safe_token(block_name)}_"
        missing = [name for name in model.input_features if name not in frame.columns]
        if missing:
            raise ValueError(
                f"Frozen local AE/GMM input parity failure for {model.model_key}: "
                f"missing={missing[:12]}"
            )
        x = frame.reindex(columns=model.input_features).apply(
            pd.to_numeric, errors="coerce"
        )
        generated = transform_ae_gmm_features(x, model.state, index=frame.index)
        posterior_cols = [
            f"gmm_cluster_posterior_{idx}" for idx in range(AE_GMM_MAX_COMPONENTS)
        ]
        posterior = generated[posterior_cols].to_numpy(dtype=np.float32, copy=False)
        output = pd.DataFrame(index=frame.index)
        for suffix in _DIRECT_AEGMM_SUFFIXES:
            output[f"{prefix}{suffix}"] = (
                pd.to_numeric(generated[suffix], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=np.float32)
            )
        semantic_probability = np.zeros(
            (len(frame), len(ECONOMIC_STATE_NAMES)), dtype=np.float32
        )
        for cluster, semantic in model.semantic_by_cluster.items():
            semantic_probability[:, ECONOMIC_STATE_NAMES.index(semantic)] += posterior[
                :, cluster
            ]
        semantic_sum = semantic_probability.sum(axis=1, keepdims=True)
        semantic_probability = np.divide(
            semantic_probability,
            np.where(semantic_sum <= 0.0, 1.0, semantic_sum),
        ).astype(np.float32)
        for idx, semantic in enumerate(ECONOMIC_STATE_NAMES):
            output[f"{prefix}prob__{semantic}"] = semantic_probability[:, idx]
        expected = posterior @ model.prior_matrix
        # State bundles are persisted for replay/live use. Prior-schema changes
        # must not invalidate older frozen bundles: retain their named columns
        # and expose new tail-conditioned fields as neutral zeros.
        stored_names = tuple(
            getattr(model, "prior_names", ())
            or ECONOMIC_PRIOR_NAMES[: expected.shape[1]]
        )
        expected_by_name = {
            name: expected[:, idx].astype(np.float32, copy=False)
            for idx, name in enumerate(stored_names[: expected.shape[1]])
        }
        zeros = np.zeros(len(frame), dtype=np.float32)
        for name in ECONOMIC_PRIOR_NAMES:
            output[f"{prefix}expected_{name}"] = expected_by_name.get(name, zeros)
        output[f"{prefix}support_log1p"] = np.float32(np.log1p(model.support_rows))
        output[f"{prefix}local_model"] = np.float32(1.0 if local else 0.0)
        output[f"{prefix}enabled"] = np.float32(1.0)
        return output

    @classmethod
    def _transform_timestamp_level_model(
        cls,
        frame: pd.DataFrame,
        model: _FrozenLocalAEGMM,
        *,
        block_name: str,
        local: bool,
        timestamp_col: str,
    ) -> pd.DataFrame:
        """Transform one causal state sequence, then broadcast it to candidates.

        A timestamp-level market state is common to every candidate within a
        side x archetype x timestamp.  Running AE/GMM directly on candidate
        rows makes temporal deltas depend on arbitrary asset row ordering.
        Aggregate only observable inputs at each timestamp, transform the
        chronological sequence once, and join the resulting state back to the
        candidates.  The generated dynamics therefore use only the current
        and prior timestamps.
        """

        if timestamp_col not in frame.columns:
            raise ValueError(
                f"Timestamp-level local AE/GMM model {model.model_key} requires "
                f"{timestamp_col!r} for causal state dynamics"
            )
        missing = [name for name in model.input_features if name not in frame.columns]
        if missing:
            raise ValueError(
                f"Frozen local AE/GMM input parity failure for {model.model_key}: "
                f"missing={missing[:12]}"
            )
        timestamp = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
        valid = timestamp.notna()
        if not bool(valid.any()):
            return pd.DataFrame(index=frame.index)
        valid_index = frame.index[valid.to_numpy()]
        # ``median`` is causal because every constituent is observed at the
        # same decision timestamp; sort before AE/GMM computes deltas.
        state_input = frame.loc[valid_index, model.input_features].apply(
            pd.to_numeric, errors="coerce"
        )
        state_input["__state_ts__"] = timestamp.loc[valid_index].to_numpy()
        by_timestamp = (
            state_input.groupby("__state_ts__", sort=True, observed=True)
            .median(numeric_only=True)
            .sort_index()
        )
        transformed_timestamp = cls._transform_model(
            by_timestamp,
            model,
            block_name=block_name,
            local=local,
        )
        broadcast = transformed_timestamp.reindex(
            pd.DatetimeIndex(timestamp.loc[valid_index].to_numpy())
        )
        return pd.DataFrame(
            broadcast.to_numpy(dtype=np.float32, copy=False),
            index=valid_index,
            columns=transformed_timestamp.columns,
            dtype=np.float32,
        )

    def transform_oos(self, oos: pd.DataFrame) -> pd.DataFrame:
        forbidden = sorted(OUTCOME_OR_DERIVED_COLUMNS.intersection(oos.columns))
        if forbidden:
            raise ValueError(
                f"OOS local AE/GMM transform received outcomes: {forbidden[:12]}"
            )
        output = pd.DataFrame(
            0.0,
            index=oos.index,
            columns=local_economic_aegmm_feature_names(
                [block.name for block in self.blocks]
            ),
            dtype=np.float32,
        )
        side = (
            oos.get(self.config.side_col, pd.Series("missing", index=oos.index))
            .astype(str)
            .str.lower()
        )
        archetype = _canonical_archetype(oos, self.config.archetype_col).astype(str)
        keys = pd.DataFrame({"side": side, "archetype": archetype}, index=oos.index)
        for block in self.blocks:
            for (side_key, archetype_key), index in keys.groupby(
                ["side", "archetype"], observed=True, sort=False
            ).groups.items():
                model = self.local_models.get(
                    (block.name, str(side_key), str(archetype_key))
                )
                local = model is not None
                if model is None:
                    model = self.side_models.get((block.name, str(side_key)))
                if model is None:
                    continue
                group = oos.loc[index]
                if block.timestamp_level:
                    transformed = self._transform_timestamp_level_model(
                        group,
                        model,
                        block_name=block.name,
                        local=local,
                        timestamp_col=self.config.timestamp_col,
                    )
                else:
                    transformed = self._transform_model(
                        group, model, block_name=block.name, local=local
                    )
                if transformed.empty:
                    continue
                output.loc[transformed.index, transformed.columns] = (
                    transformed.to_numpy(dtype=np.float32, copy=False)
                )
        return output.astype(np.float32, copy=False)

    def transform_train(self, train: pd.DataFrame) -> pd.DataFrame:
        """Transform fit rows with outcomes stripped from the inference matrix."""
        safe = train.drop(
            columns=[
                name for name in OUTCOME_OR_DERIVED_COLUMNS if name in train.columns
            ],
            errors="ignore",
        )
        return self.transform_oos(safe)

    def required_input_features(self) -> list[str]:
        required = [self.config.side_col, self.config.archetype_col]
        if any(block.timestamp_level for block in self.blocks):
            required.append(self.config.timestamp_col)
        for model in list(self.side_models.values()) + list(self.local_models.values()):
            required.extend(model.input_features)
        return list(dict.fromkeys(required))

    def manifest(self) -> dict[str, Any]:
        model_rows = []
        for scope, models in (("side", self.side_models), ("local", self.local_models)):
            for key, model in models.items():
                model_rows.append(
                    {
                        "scope": scope,
                        "lookup_key": [str(value) for value in key],
                        "model_key": model.model_key,
                        "block": model.block_name,
                        "support_rows": int(model.support_rows),
                        "fit_rows": int(model.fit_rows),
                        "train_start": model.train_start,
                        "train_end": model.train_end,
                        "input_features": list(model.input_features),
                        "gmm_n_components": int(
                            model.state.get("gmm_n_components", 0) or 0
                        ),
                        "selected_config": dict(
                            model.state.get("selected_config", {}) or {}
                        ),
                        "sample_manifest": model.state.get("sample_manifest"),
                    }
                )
        return {
            "schema": "local_economic_aegmm_v1",
            "semantic_version": str(self.config.semantic_version),
            "train_start": self.train_start_,
            "train_end": self.train_end_,
            "blocks": [
                {
                    "name": block.name,
                    "timestamp_level": bool(block.timestamp_level),
                    "requested_features": list(block.features),
                }
                for block in self.blocks
            ],
            "side_model_count": int(len(self.side_models)),
            "local_model_count": int(len(self.local_models)),
            "models": model_rows,
            "cluster_candidates": list(self.config.cluster_candidates),
            "ae_max_train_rows": int(self.config.ae_max_train_rows),
            "gmm_max_train_rows": int(self.config.gmm_max_train_rows),
            "full_train_fit": bool(self.config.full_train_fit),
            "fit_sample_contract": (
                "time-spread HPO sample followed by selected-configuration refit on all resolved rows before fit cutoff"
                if self.config.full_train_fit
                else "bounded train-only time-spread sample"
            ),
            "component_complexity_penalty": float(
                self.config.component_complexity_penalty
            ),
            "generated_features": local_economic_aegmm_feature_names(
                [block.name for block in self.blocks]
            ),
            "required_input_features": self.required_input_features(),
            "temporal_dynamics": (
                "timestamp-level blocks aggregate observable inputs per side x archetype "
                "x timestamp, transform the chronologically sorted state sequence, and "
                "broadcast causal speed/acceleration outputs back to candidates"
            ),
            "leakage_contract": {
                "state_inputs": "pre-entry numeric features only",
                "economic_hpo": "training outcomes only",
                "top_tail_basis": (
                    "Global candidate-book rank per decision timestamp is computed before "
                    "side x archetype partitioning; local models never redefine top 10%."
                ),
                "state_priors": "posterior-weighted, train-only, shrinkage-stabilized",
                "oos_assignment": "frozen scaler, AE, GMM, semantics, and priors",
                "outcomes_at_inference": False,
                "recent_realized_performance_features": False,
            },
        }


def _apply_frozen_ood_state(
    frame: pd.DataFrame, state: Mapping[str, Any]
) -> pd.DataFrame:
    columns = [str(name) for name in state.get("columns", [])]
    if not columns:
        return frame
    values = frame.reindex(columns=columns).to_numpy(dtype=np.float32, copy=True)
    finite = np.isfinite(values)
    mean = np.asarray(state["mean"], dtype=np.float32)
    std = np.asarray(state["std"], dtype=np.float32)
    q25 = np.asarray(state["q25"], dtype=np.float32)
    q75 = np.asarray(state["q75"], dtype=np.float32)
    iqr = np.maximum(q75 - q25, 1e-6)
    filled = np.where(finite, values, mean)
    z = (filled - mean) / np.maximum(std, 1e-6)
    abs_z = np.abs(z)
    exceed = ((filled < q25 - 1.5 * iqr) | (filled > q75 + 1.5 * iqr)) & finite
    out = frame.copy(deep=False)
    additions = pd.DataFrame(
        {
            "meta_sel_ood_abs_z_mean": np.mean(abs_z, axis=1),
            "meta_sel_ood_abs_z_max": np.max(abs_z, axis=1),
            "meta_sel_ood_abs_z_p95": np.quantile(abs_z, 0.95, axis=1),
            "meta_sel_ood_iqr_exceed_frac": np.mean(exceed, axis=1),
            "meta_sel_ood_missing_frac": np.mean(~finite, axis=1),
            "meta_sel_ood_centroid_l2": np.sqrt(np.mean(z * z, axis=1)),
        },
        index=frame.index,
        dtype=np.float32,
    )
    return pd.concat(
        [
            out.drop(
                columns=[name for name in additions.columns if name in out.columns]
            ),
            additions,
        ],
        axis=1,
        copy=False,
    )


@dataclass
class LocalEconomicAEGMMModelBundle:
    """Frozen primary LightGBM model with direct local AE/GMM inputs."""

    model: Any
    local_aegmm: LocalEconomicAEGMM
    selected_features: list[str]
    raw_selected_features: list[str]
    feature_medians: dict[str, float]
    ood_state: dict[str, Any]
    score_col: str = "score_meta_base_soft_label"
    fit_through: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def required_input_features(self) -> list[str]:
        generated = set(
            local_economic_aegmm_feature_names(
                [block.name for block in self.local_aegmm.blocks]
            )
        )
        raw = [name for name in self.raw_selected_features if name not in generated]
        raw.extend(self.local_aegmm.required_input_features())
        return list(dict.fromkeys(raw))

    def _append_geometry_if_required(self, frame: pd.DataFrame) -> pd.DataFrame:
        required_geometry = [
            name
            for name in self.local_aegmm.required_input_features()
            if name.startswith("meta_xsgeom_")
        ]
        if not required_geometry or all(
            name in frame.columns for name in required_geometry
        ):
            return frame
        if self.score_col not in frame.columns:
            raise ValueError(
                "Cross-sectional AE/GMM inference requires the frozen base/meta score "
                f"column {self.score_col!r} on the complete candidate batch"
            )
        from .meta_cross_sectional_geometry import materialize_cross_sectional_geometry

        generated = materialize_cross_sectional_geometry(
            frame, score_col=self.score_col
        )
        additions = generated.reindex(
            columns=[name for name in generated.columns if name not in frame.columns]
        )
        return pd.concat([frame, additions], axis=1, copy=False)

    def predict(self, pre_entry_frame: pd.DataFrame) -> np.ndarray:
        forbidden = sorted(
            OUTCOME_OR_DERIVED_COLUMNS.intersection(pre_entry_frame.columns)
        )
        if forbidden:
            raise ValueError(
                f"Frozen local AE/GMM model received outcomes: {forbidden[:12]}"
            )
        safe = self._append_geometry_if_required(pre_entry_frame)
        generated = self.local_aegmm.transform_oos(safe)
        augmented = pd.concat([safe, generated], axis=1, copy=False)
        matrix = augmented.reindex(columns=self.raw_selected_features).apply(
            pd.to_numeric, errors="coerce"
        )
        for name in self.raw_selected_features:
            matrix[name] = (
                matrix[name]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(float(self.feature_medians.get(name, 0.0)))
            )
        matrix = _apply_frozen_ood_state(matrix.astype(np.float32), self.ood_state)
        matrix = matrix.reindex(columns=self.selected_features, fill_value=0.0).astype(
            np.float32, copy=False
        )
        return np.asarray(self.model.predict(matrix), dtype=np.float32).reshape(-1)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "local_economic_aegmm_model_bundle_v1",
            "fit_through": self.fit_through,
            "selected_feature_count": int(len(self.selected_features)),
            "raw_selected_feature_count": int(len(self.raw_selected_features)),
            "required_input_features": self.required_input_features(),
            "local_aegmm": self.local_aegmm.manifest(),
            "metadata": self.metadata,
            "inference_contract": (
                "Materialize causal cross-sectional geometry on the complete candidate batch, "
                "apply the frozen local AE/GMM states, then feed the selected continuous state "
                "outputs directly to the frozen primary LightGBM model."
            ),
        }


def assert_local_aegmm_oos_safe_features(features: Sequence[str]) -> None:
    forbidden = sorted(
        set(str(name) for name in features) & set(OUTCOME_OR_DERIVED_COLUMNS)
    )
    if forbidden:
        raise ValueError(
            f"Local AE/GMM input feature contract contains outcomes: {forbidden}"
        )


__all__ = [
    "ECONOMIC_PRIOR_NAMES",
    "ECONOMIC_STATE_NAMES",
    "EconomicAEGMMBlock",
    "HierarchicalEconomicAEGMM",
    "HierarchicalEconomicAEGMMConfig",
    "BASE_DIRECTIONAL_STATE_FEATURES",
    "LOCAL_ECONOMIC_AEGMM_PREFIX",
    "META_CROSS_SECTIONAL_GEOMETRY_FEATURES",
    "META_MARKET_STATE_FEATURES",
    "LocalEconomicAEGMM",
    "LocalEconomicAEGMMConfig",
    "LocalEconomicAEGMMModelBundle",
    "assert_local_aegmm_oos_safe_features",
    "default_base_economic_aegmm_blocks",
    "default_meta_economic_aegmm_blocks",
    "local_economic_aegmm_feature_names",
]
