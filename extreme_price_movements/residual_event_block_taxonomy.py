"""Block-level taxonomy for residual-adverse market episodes.

This module is intentionally descriptive.  It starts with already-materialized
daily adverse calendar cells, builds a causal observable trajectory around each
contiguous block, and groups *blocks* rather than individual trades.  The
result is research evidence for whether a local overlay should be one detector
or several mechanism-specific detectors; it is not itself an inference policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from extreme_price_movements.unsupervised_regime_learning.failure_episodes import (
    validate_inference_feature_columns,
)
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

KEY_COLUMNS = ("day", "side_name", "archetype_policy_key")

# Each family intentionally uses only observable price, OI, funding, and
# cross-sectional state.  Missing primitives are simply omitted, allowing the
# taxonomy to compare historical stores with different feature revisions.
MECHANISM_FAMILIES: dict[str, tuple[str, ...]] = {
    "liquidation_pressure": (
        "asset_liquidation_phase_score",
        "mkt_median_oi_chg_4h_rz",
        "mkt_oi_flush_breadth_accel_1h",
        "mkt_pct_price_down_oi_down_4h",
        "price_down_oi_down_4h_rz",
        "shock_12h",
        "market_pc1_variance_share_12h",
    ),
    "recovery_short_covering": (
        "asset_short_covering_score",
        "mkt_oi_flush_breadth_recovery_4h",
        "mkt_pct_price_up_oi_down_1h",
        "price_up_oi_down_1h_rz",
        "market_breadth_recovery_from_24h_min",
        "mkt_leverage_rebuild_score",
        "price_recovery_fraction_24h",
    ),
    "funding_transition": (
        "funding_crowding_release_4h",
        "funding_positive_to_negative_intensity",
        "funding_negative_to_positive_intensity",
        "funding_sign_persistence_24h",
        "funding_sign_persistence_72h",
    ),
    "correlation_fragmentation": (
        "market_downside_pairwise_corr_24h",
        "market_downside_corr_minus_unconditional_corr_24h",
        "market_pc1_variance_share_24h",
        "market_pc1_variance_share_chg_4h",
        "xs_dispersion__ob_depth_to_qv_z_x_rvol_z",
        "eth_ret_24h",
    ),
    "volatility_compression_transition": (
        "volatility_ratio_short_long",
        "vol_of_vol_cp_logstd_8_32",
        "return_autocorr_cp_logstd_8_32",
        "range_climax_decay_4h",
        "volume_climax_decay_4h",
        "downside_deceleration_4h_rz",
        "downside_deceleration_8h_rz",
    ),
    "asset_market_divergence": (
        "asset_mkt_liquidation_phase_divergence",
        "asset_mkt_exhaustion_phase_divergence",
        "asset_minus_mkt_oi_chg_1h_rz",
        "asset_minus_mkt_oi_chg_4h_rz",
        "asset_minus_mkt_oi_recovery_fraction_24h",
        "asset_minus_mkt_price_recovery_fraction_24h",
        "asset_minus_mkt_short_cover_intensity_1h",
    ),
}


@dataclass(frozen=True)
class BlockTaxonomyConfig:
    """Trajectory and clustering settings for a descriptive taxonomy."""

    pre_days: int = 2
    post_days: int = 1
    min_reference_days: int = 30
    controls_per_block: int = 3
    max_clusters: int = 5
    min_cluster_blocks: int = 3
    max_event_days: int | None = None


def canonical_calendar(calendar: pd.DataFrame) -> pd.DataFrame:
    """Return one daily side x archetype calendar row with a binary event flag."""

    missing = set(KEY_COLUMNS).difference(calendar.columns)
    if missing:
        raise KeyError(f"Calendar missing keys: {sorted(missing)}")
    result = calendar.copy()
    result["day"] = pd.to_datetime(result["day"], utc=True).dt.floor("D")
    for name in ("side_name", "archetype_policy_key"):
        result[name] = result[name].astype(str)
    event_column = (
        "adverse_calendar_cell"
        if "adverse_calendar_cell" in result.columns
        else "adverse_event_rows"
    )
    if event_column not in result.columns:
        raise KeyError("Calendar needs adverse_calendar_cell or adverse_event_rows")
    result["adverse_event"] = (
        pd.to_numeric(result[event_column], errors="coerce").fillna(0).gt(0)
    )
    aggregations: dict[str, str] = {"adverse_event": "max"}
    for name in (
        "selected_rows",
        "mean_ev_after_1pct",
        "clean_exec_rate",
        "clean_exec_precision",
        "signed_surprise",
        "persistence_strength",
        "large_event_strength",
    ):
        if name in result.columns:
            aggregations[name] = "mean"
    for name in result.columns:
        if name.startswith("expost__"):
            aggregations[name] = "mean"
    return (
        result.groupby(list(KEY_COLUMNS), observed=True, as_index=False)
        .agg(aggregations)
        .sort_values(list(KEY_COLUMNS), kind="stable")
    )


def attach_event_blocks(
    calendar: pd.DataFrame,
    *,
    max_event_days: int | None = None,
) -> pd.DataFrame:
    """Assign contiguous adverse-day blocks independently per side x archetype."""

    if max_event_days is not None and int(max_event_days) < 1:
        raise ValueError("max_event_days must be positive when provided")

    result = canonical_calendar(calendar)
    result["event_block"] = "normal"
    for _, index in result.groupby(
        ["side_name", "archetype_policy_key"], observed=True, sort=False
    ).groups.items():
        local = result.loc[index].sort_values("day", kind="stable")
        current = 0
        previous: pd.Timestamp | None = None
        active = False
        active_days = 0
        for row_index, row in local.iterrows():
            if not bool(row["adverse_event"]):
                previous = row["day"]
                active = False
                active_days = 0
                continue
            contiguous = (
                active
                and previous is not None
                and (row["day"] - previous == pd.Timedelta(days=1))
                and (
                    max_event_days is None
                    or active_days < int(max_event_days)
                )
            )
            if not contiguous:
                current += 1
                active_days = 0
            result.at[row_index, "event_block"] = f"event_{current:03d}"
            previous = row["day"]
            active = True
            active_days += 1
    return result


def daily_observable_state(
    states: pd.DataFrame,
    *,
    features: list[str],
    selected_only: bool = True,
) -> pd.DataFrame:
    """Collapse rows to a daily-open observable state without outcomes.

    The first available timestamp per day is selected before any
    cross-sectional aggregation.  This prevents end-of-day market state from
    describing an adverse event that occurred earlier in that same day.
    Medians then prevent a large symbol count from giving an individual asset
    or a sparse snapshot undue influence.  Outcome columns are never used.
    """

    required = {"__ts__", "side_name", "archetype_policy_key"}
    missing = required.difference(states.columns)
    if missing:
        raise KeyError(f"State rows missing keys: {sorted(missing)}")
    available = [name for name in features if name in states.columns]
    if not available:
        raise ValueError("No requested observable taxonomy features are available")
    # This is the earliest shared boundary between descriptive episode work and
    # a prospective detector.  The caller may load a wide ledger, but outcome
    # fields must never survive into the day-open state panel.
    validate_inference_feature_columns(available)
    columns = ["__ts__", "side_name", "archetype_policy_key", *available]
    if selected_only and "selected_top30" in states.columns:
        columns.append("selected_top30")
    result = states.loc[:, list(dict.fromkeys(columns))].copy()
    result["__ts__"] = pd.to_datetime(result["__ts__"], utc=True)
    result["day"] = result["__ts__"].dt.floor("D")
    result["side_name"] = result["side_name"].astype(str)
    result["archetype_policy_key"] = result["archetype_policy_key"].astype(str)
    # The timestamp must be chosen from the raw observable universe.  Selecting
    # top-30 rows first could move the snapshot later in the day and create a
    # subtle same-day look-ahead for an episode detector.
    first_timestamp = result.groupby(
        ["day", "side_name", "archetype_policy_key"], observed=True
    )["__ts__"].transform("min")
    result = result.loc[result["__ts__"].eq(first_timestamp)].copy()
    if selected_only and "selected_top30" in result:
        result = result.loc[result["selected_top30"].fillna(False).astype(bool)].copy()
    # Convert the full feature block once. Repeated column assignment fragments
    # wide frames and becomes material when this is called for every monthly
    # candidate shard.
    numeric = result.loc[:, available].apply(pd.to_numeric, errors="coerce")
    result = pd.concat(
        [
            result.loc[:, [*KEY_COLUMNS]],
            numeric.astype(np.float32, copy=False),
        ],
        axis=1,
        copy=False,
    )
    return (
        result.groupby(list(KEY_COLUMNS), observed=True, as_index=False)[available]
        .median()
        .sort_values(list(KEY_COLUMNS), kind="stable")
    )


def _robust_z(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    output = np.full(values.shape, np.nan, dtype=np.float64)
    available = np.isfinite(reference).any(axis=0)
    if not bool(available.any()):
        return output
    usable_reference = reference[:, available]
    median = np.nanmedian(usable_reference, axis=0)
    q25 = np.nanquantile(usable_reference, 0.25, axis=0)
    q75 = np.nanquantile(usable_reference, 0.75, axis=0)
    scale = np.maximum(q75 - q25, 1e-4)
    output[:, available] = np.clip(
        (values[:, available] - median) / scale,
        -8.0,
        8.0,
    )
    return output


def _nanmean_axis0(values: np.ndarray) -> np.ndarray:
    """Mean without warnings for an absent historical feature revision."""

    finite = np.isfinite(values)
    count = finite.sum(axis=0)
    total = np.where(finite, values, 0.0).sum(axis=0)
    return np.divide(
        total, count, out=np.full(values.shape[1], np.nan), where=count > 0
    )


def _nanmax_abs_axis0(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    filled = np.where(finite, np.abs(values), -np.inf)
    maximum = filled.max(axis=0)
    maximum[~finite.any(axis=0)] = np.nan
    return maximum


def _nanmean_scalar(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if len(finite) else np.nan


def _trajectory_summary(
    local: pd.DataFrame,
    event_days: pd.DatetimeIndex,
    features: list[str],
    config: BlockTaxonomyConfig,
) -> tuple[dict[str, float], np.ndarray, list[str]]:
    """Summarise a block using only observable daily state around its dates."""

    start, end = event_days.min(), event_days.max()
    pre_start = start - pd.Timedelta(days=config.pre_days)
    post_end = end + pd.Timedelta(days=config.post_days)
    history = local.loc[local["day"].lt(start), features]
    if len(history) < config.min_reference_days:
        history = local.loc[local["day"].lt(start), features]
    if len(history) == 0:
        history = local.loc[:, features]
    # Trajectory positions are stable semantics, rather than post-hoc cluster
    # labels: preceding state, onset change, adverse peak, and early recovery.
    pre = local.loc[
        local["day"].between(pre_start, start - pd.Timedelta(days=1)), features
    ]
    active = local.loc[local["day"].isin(event_days), features]
    post = local.loc[
        local["day"].between(end + pd.Timedelta(days=1), post_end), features
    ]
    prior = (
        pre.median(axis=0).to_numpy(np.float64)
        if len(pre)
        else np.nanmedian(history.to_numpy(np.float64), axis=0)
    )
    active_values = active.to_numpy(np.float64)
    active_mean = _nanmean_axis0(active_values) if len(active_values) else prior
    peak = _nanmax_abs_axis0(active_values) if len(active_values) else np.abs(prior)
    post_mean = post.median(axis=0).to_numpy(np.float64) if len(post) else active_mean
    values = np.concatenate([prior, active_mean, peak, post_mean]).reshape(4, -1)
    z = _robust_z(values, history.to_numpy(np.float64))
    names: list[str] = []
    vector: list[float] = []
    output: dict[str, float] = {
        "event_start": start,
        "event_end": end,
        "event_days": float(len(event_days)),
    }
    for position, row in zip(("pre", "active", "peak_abs", "post"), z, strict=True):
        for feature, value in zip(features, row, strict=True):
            name = f"{position}__{feature}"
            output[name] = float(value)
            names.append(name)
            vector.append(float(value))
    # Explicit transition terms are needed because equal levels can have very
    # different meaning at onset and recovery.
    for feature_index, feature in enumerate(features):
        onset = z[1, feature_index] - z[0, feature_index]
        recovery = z[3, feature_index] - z[1, feature_index]
        for name, value in (
            (f"onset_delta__{feature}", onset),
            (f"recovery_delta__{feature}", recovery),
        ):
            output[name] = float(value)
            names.append(name)
            vector.append(float(value))
    # Raw per-feature trajectories remain in the artifact for diagnostics.  A
    # block taxonomy, however, has far fewer independent episodes than rows;
    # clustering the raw matrix would make distance mostly noise.  Reduce only
    # the clustering representation to six economic mechanism intensities and
    # transition amplitudes, while retaining signs as a separate family mean.
    family_names: list[str] = []
    family_vector: list[float] = []
    mechanism_families = dict(MECHANISM_FAMILIES)
    attribution_features = tuple(
        feature for feature in features if feature.startswith("base_attr_")
    )
    if attribution_features:
        mechanism_families["model_attribution_shift"] = attribution_features
    error_shape_features = tuple(
        feature for feature in features if feature.startswith("expost__")
    )
    if error_shape_features:
        mechanism_families["error_shape"] = error_shape_features
    for family, candidates in mechanism_families.items():
        positions = [features.index(name) for name in candidates if name in features]
        if not positions:
            continue
        active_values = z[1, positions]
        peak_values = z[2, positions]
        onset_values = z[1, positions] - z[0, positions]
        recovery_values = z[3, positions] - z[1, positions]
        summaries = {
            "active_mean_z": _nanmean_scalar(active_values),
            "peak_abs_z": _nanmean_scalar(np.abs(peak_values)),
            "onset_abs_delta": _nanmean_scalar(np.abs(onset_values)),
            "recovery_abs_delta": _nanmean_scalar(np.abs(recovery_values)),
        }
        for suffix, value in summaries.items():
            name = f"family__{family}__{suffix}"
            output[name] = value
            family_names.append(name)
            family_vector.append(value)
    # Preserve the residual-vector geometry instead of reducing every model
    # failure coordinate to one family mean. These remain strictly ex-post
    # descriptive fields and are never eligible for the prospective detector.
    for feature in error_shape_features:
        position = features.index(feature)
        suffix = feature.removeprefix("expost__")
        for phase, value in (
            ("active", z[1, position]),
            ("onset_delta", z[1, position] - z[0, position]),
        ):
            name = f"family__error_vector__{phase}__{suffix}"
            output[name] = float(value)
            family_names.append(name)
            family_vector.append(float(value))
    for feature in attribution_features:
        position = features.index(feature)
        suffix = feature.removeprefix("base_attr_")
        active_value = float(z[1, position])
        onset_value = float(z[1, position] - z[0, position])
        for phase, value in (("active", active_value), ("onset_delta", onset_value)):
            name = f"family__attribution_vector__{phase}__{suffix}"
            output[name] = value
            family_names.append(name)
            family_vector.append(value)
        if feature.startswith("base_attr_signed__"):
            sign_flip = float(
                np.isfinite(z[0, position])
                and np.isfinite(z[1, position])
                and z[0, position] * z[1, position] < 0.0
            )
            name = f"family__attribution_vector__sign_flip__{suffix}"
            output[name] = sign_flip
            family_names.append(name)
            family_vector.append(sign_flip)
    if family_vector:
        return output, np.asarray(family_vector, dtype=np.float32), family_names
    return output, np.asarray(vector, dtype=np.float32), names


def _choose_cluster_count(
    matrix: np.ndarray, maximum: int, minimum_cluster_blocks: int
) -> int:
    """Choose a small descriptive clustering granularity by silhouette score."""

    count = len(matrix)
    if count < 6:
        return 0
    upper = min(int(maximum), count - 1)
    best_count, best_score = 0, -np.inf
    for clusters in range(2, upper + 1):
        labels = AgglomerativeClustering(
            n_clusters=clusters, linkage="ward"
        ).fit_predict(matrix)
        _, support = np.unique(labels, return_counts=True)
        if len(support) < 2 or int(support.min()) < int(minimum_cluster_blocks):
            continue
        score = silhouette_score(matrix, labels)
        if score > best_score:
            best_count, best_score = clusters, float(score)
    return best_count


def build_block_taxonomy(
    calendar: pd.DataFrame,
    daily_state: pd.DataFrame,
    *,
    config: BlockTaxonomyConfig = BlockTaxonomyConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return block taxonomy plus a wide causal trajectory feature table.

    Clustering is local to each side x archetype.  A family label is therefore
    only comparable within that local failure population.  This avoids a global
    cluster spending all of its capacity separating long from short or broad
    archetype identities.
    """

    events = attach_event_blocks(
        calendar,
        max_event_days=config.max_event_days,
    )
    observable_features = [
        name for name in daily_state.columns if name not in KEY_COLUMNS
    ]
    error_features = [name for name in events.columns if name.startswith("expost__")]
    features = [*observable_features, *error_features]
    merged = events.merge(
        daily_state, on=list(KEY_COLUMNS), how="left", validate="one_to_one"
    )
    block_rows: list[dict[str, object]] = []
    vectors: dict[tuple[str, str], list[np.ndarray]] = {}
    vector_names: dict[tuple[str, str], list[str]] = {}
    local_indices: dict[tuple[str, str], list[int]] = {}
    for (side, archetype), local in merged.groupby(
        ["side_name", "archetype_policy_key"], observed=True, sort=False
    ):
        local = local.sort_values("day", kind="stable")
        usable = [name for name in features if local[name].notna().any()]
        if not usable:
            continue
        for block, rows in local.loc[local["event_block"].ne("normal")].groupby(
            "event_block", observed=True, sort=False
        ):
            summary, vector, names = _trajectory_summary(
                local, pd.DatetimeIndex(rows["day"]), usable, config
            )
            summary.update(
                {
                    "side_name": str(side),
                    "archetype_policy_key": str(archetype),
                    "event_block": str(block),
                    "calendar_selected_rows": float(
                        rows.get("selected_rows", pd.Series(dtype=float)).sum()
                    ),
                    "calendar_mean_ev": float(
                        rows.get("mean_ev_after_1pct", pd.Series(dtype=float)).mean()
                    ),
                    "calendar_mean_signed_surprise": float(
                        rows.get("signed_surprise", pd.Series(dtype=float)).mean()
                    ),
                    "calendar_persistence_strength": float(
                        rows.get("persistence_strength", pd.Series(dtype=float)).max()
                    ),
                    "calendar_large_event_strength": float(
                        rows.get("large_event_strength", pd.Series(dtype=float)).max()
                    ),
                }
            )
            # Retain raw error-shape levels for semantic interpretation. The
            # clustered representation still uses robust trajectory vectors;
            # these fields only explain the resulting ex-post mode.
            for name in error_features:
                values = pd.to_numeric(rows.get(name), errors="coerce")
                summary[f"calendar_error__{name.removeprefix('expost__')}"] = (
                    float(values.mean()) if values.notna().any() else np.nan
                )
            key = (str(side), str(archetype))
            local_indices.setdefault(key, []).append(len(block_rows))
            vectors.setdefault(key, []).append(vector)
            vector_names[key] = names
            block_rows.append(summary)
    if not block_rows:
        return pd.DataFrame(), pd.DataFrame()
    blocks = pd.DataFrame(block_rows)
    blocks["block_family"] = "insufficient_local_blocks"
    blocks["block_family_id"] = -1
    blocks["cluster_silhouette"] = np.nan
    for key, indices in local_indices.items():
        matrix = np.vstack(vectors[key]).astype(np.float32, copy=False)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=8.0, neginf=-8.0)
        clusters = _choose_cluster_count(
            matrix, config.max_clusters, config.min_cluster_blocks
        )
        if clusters == 0:
            continue
        labels = AgglomerativeClustering(
            n_clusters=clusters, linkage="ward"
        ).fit_predict(matrix)
        silhouette = silhouette_score(matrix, labels)
        for position, block_index in enumerate(indices):
            label = int(labels[position])
            blocks.loc[block_index, "block_family_id"] = label
            blocks.loc[block_index, "block_family"] = f"local_family_{label:02d}"
            blocks.loc[block_index, "cluster_silhouette"] = silhouette
    return blocks, blocks.copy()


def block_family_profiles(
    trajectories: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise local families with their largest signed trajectory deltas."""

    if trajectories.empty:
        return pd.DataFrame()
    family_columns = [
        name for name in trajectories.columns if name.startswith("family__")
    ]
    trajectory_columns = family_columns or [
        name
        for name in trajectories.columns
        if name.startswith(
            ("active__", "onset_delta__", "recovery_delta__", "peak_abs__")
        )
    ]
    rows: list[dict[str, object]] = []
    for keys, group in trajectories.groupby(
        ["side_name", "archetype_policy_key", "block_family"], observed=True
    ):
        mean = group[trajectory_columns].mean(numeric_only=True)
        top = mean.abs().sort_values(ascending=False).head(8)
        details = "|".join(f"{name}={mean[name]:+.2f}" for name in top.index)
        rows.append(
            {
                "side_name": keys[0],
                "archetype_policy_key": keys[1],
                "block_family": keys[2],
                "blocks": len(group),
                "mean_event_days": float(group["event_days"].mean()),
                "mean_calendar_ev": float(group["calendar_mean_ev"].mean()),
                "mean_signed_surprise": float(
                    group["calendar_mean_signed_surprise"].mean()
                ),
                "mean_silhouette": float(group["cluster_silhouette"].mean()),
                "detector_assessable_blocks": int(group["detector_assessable"].sum())
                if "detector_assessable" in group.columns
                else 0,
                "detector_recognized_blocks": int(group["detector_recognized"].sum())
                if "detector_recognized" in group.columns
                else 0,
                "detector_block_recall": float(
                    group.loc[
                        group["detector_assessable"], "detector_recognized"
                    ].mean()
                )
                if "detector_assessable" in group.columns
                and group["detector_assessable"].any()
                else np.nan,
                "detector_mean_max_risk": float(group["detector_max_risk"].mean())
                if "detector_max_risk" in group.columns
                else np.nan,
                "dominant_trajectory_features": details,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["side_name", "archetype_policy_key", "blocks"], ascending=[True, True, False]
    )


def annotate_onset_mechanism_profiles(blocks: pd.DataFrame) -> pd.DataFrame:
    """Annotate every adverse block with an observable onset mechanism profile.

    Local clustering deliberately leaves sparse populations unclustered.  That
    is the correct statistical choice, but it should not hide the fact that an
    unclustered block still has an interpretable *market transition*.  This
    inventory uses only the state at block onset and its change from the prior
    two days.  It is therefore safe to use as a research catalogue for later
    frozen detectors, unlike full-block recovery summaries.

    The annotations are descriptive, not hard labels or an active inference
    feature.  ``onset_primary_mechanism`` means "largest observable mechanism
    intensity" rather than a claim that the mechanism caused the loss.
    """

    if blocks.empty:
        return blocks.copy()
    result = blocks.copy()
    scores: dict[str, np.ndarray] = {}
    available: list[str] = []
    for family in MECHANISM_FAMILIES:
        active_name = f"family__{family}__active_mean_z"
        onset_name = f"family__{family}__onset_abs_delta"
        is_available = active_name in result.columns or onset_name in result.columns
        if is_available:
            available.append(family)
        active_source = (
            result[active_name]
            if active_name in result.columns
            else pd.Series(np.nan, index=result.index, dtype=np.float64)
        )
        onset_source = (
            result[onset_name]
            if onset_name in result.columns
            else pd.Series(np.nan, index=result.index, dtype=np.float64)
        )
        active = pd.to_numeric(active_source, errors="coerce").to_numpy(np.float64)
        onset = pd.to_numeric(onset_source, errors="coerce").to_numpy(np.float64)
        # A material transition and a material state are both needed.  Taking
        # absolute magnitude is intentional here: the detailed signed
        # trajectory columns remain in the artifact for interpretation.
        if not is_available:
            continue
        scores[family] = np.nan_to_num(
            0.55 * np.abs(onset) + 0.45 * np.abs(active), nan=0.0
        )
        result[f"onset_mechanism_score__{family}"] = scores[family].astype(np.float32)
    names = list(scores)
    if not names:
        result["onset_primary_mechanism"] = "unavailable"
        result["onset_primary_mechanism_score"] = np.nan
        result["onset_mechanism_margin"] = np.nan
        result["onset_mechanism_confident"] = False
        result["onset_mechanism_available_count"] = 0
        return result
    matrix = np.column_stack([scores[name] for name in names])
    order = np.argsort(matrix, axis=1)
    winner = order[:, -1]
    runner_up = order[:, -2] if len(names) > 1 else winner
    top = matrix[np.arange(len(result)), winner]
    second = matrix[np.arange(len(result)), runner_up]
    result["onset_primary_mechanism"] = np.asarray(names, dtype=object)[winner]
    result["onset_primary_mechanism_score"] = top.astype(np.float32)
    result["onset_mechanism_margin"] = (top - second).astype(np.float32)
    result["onset_mechanism_confident"] = (top >= 0.75) & ((top - second) >= 0.20)
    result["onset_mechanism_available_count"] = int(len(available))
    return result


def attach_detector_block_coverage(
    blocks: pd.DataFrame,
    detector_daily: pd.DataFrame,
    *,
    risk_column: str,
    threshold: float,
) -> pd.DataFrame:
    """Attach a frozen detector's event-block coverage for taxonomy review.

    The detector output is read-only evaluation evidence.  It neither changes
    taxonomy families nor participates in their clustering.
    """

    if blocks.empty:
        return blocks.copy()
    required = {"day", "side_name", "archetype_policy_key", risk_column}
    missing = required.difference(detector_daily.columns)
    if missing:
        raise KeyError(f"Detector daily output missing: {sorted(missing)}")
    daily = detector_daily.loc[:, list(required)].copy()
    daily["day"] = pd.to_datetime(daily["day"], utc=True).dt.floor("D")
    daily["detector_risk"] = pd.to_numeric(daily[risk_column], errors="coerce")
    daily = daily.groupby(list(KEY_COLUMNS), observed=True, as_index=False)[
        "detector_risk"
    ].max()
    result = blocks.copy()
    maximum: list[float] = []
    for block in result.itertuples(index=False):
        mask = (
            daily["side_name"].eq(block.side_name)
            & daily["archetype_policy_key"].eq(block.archetype_policy_key)
            & daily["day"].between(
                pd.Timestamp(block.event_start), pd.Timestamp(block.event_end)
            )
        )
        values = daily.loc[mask, "detector_risk"]
        maximum.append(float(values.max()) if len(values) else np.nan)
    result["detector_max_risk"] = maximum
    result["detector_assessable"] = result["detector_max_risk"].notna()
    result["detector_recognized"] = result["detector_assessable"] & result[
        "detector_max_risk"
    ].ge(float(threshold))
    return result


def detector_recognized_missed_contrasts(blocks: pd.DataFrame) -> pd.DataFrame:
    """Compare recognized and missed adverse blocks within local populations.

    This is deliberately a block-level descriptive contrast, not another
    fitting target.  It identifies which *observable trajectory* distinguishes
    the narrow mechanism already caught by a high-precision detector from the
    remaining adverse blocks it does not claim to explain.
    """

    required = {
        "side_name",
        "archetype_policy_key",
        "detector_assessable",
        "detector_recognized",
    }
    if not required.issubset(blocks.columns):
        return pd.DataFrame()
    values = [name for name in blocks.columns if name.startswith("family__")]
    rows: list[dict[str, object]] = []
    for (side, archetype), local in blocks.groupby(
        ["side_name", "archetype_policy_key"], observed=True
    ):
        local = local.loc[local["detector_assessable"]].copy()
        hit = local.loc[local["detector_recognized"]]
        missed = local.loc[~local["detector_recognized"]]
        if hit.empty or missed.empty:
            continue
        numeric = local.loc[:, values].apply(pd.to_numeric, errors="coerce")
        hit_numeric = numeric.loc[hit.index]
        missed_numeric = numeric.loc[missed.index]
        comparable = [
            feature
            for feature in values
            if hit_numeric[feature].notna().any()
            and missed_numeric[feature].notna().any()
        ]
        if not comparable:
            continue
        reference = numeric.loc[:, comparable]
        scale = (
            reference.quantile(0.75, numeric_only=True)
            - reference.quantile(0.25, numeric_only=True)
        ).clip(lower=1e-4)
        hit_median = hit_numeric.loc[:, comparable].median(numeric_only=True)
        missed_median = missed_numeric.loc[:, comparable].median(numeric_only=True)
        delta = hit_median - missed_median
        for feature, value in delta.items():
            rows.append(
                {
                    "side_name": side,
                    "archetype_policy_key": archetype,
                    "feature": feature,
                    "recognized_blocks": int(len(hit)),
                    "missed_blocks": int(len(missed)),
                    "recognized_median": float(hit_median[feature]),
                    "missed_median": float(missed_median[feature]),
                    "median_difference": float(value),
                    "robust_standardized_difference": float(
                        np.clip(value / scale[feature], -8.0, 8.0)
                    ),
                }
            )
    return (
        pd.DataFrame(rows).sort_values(
            ["side_name", "archetype_policy_key", "robust_standardized_difference"],
            key=lambda column: column.abs()
            if column.name == "robust_standardized_difference"
            else column,
            ascending=[True, True, False],
            kind="stable",
        )
        if rows
        else pd.DataFrame()
    )


def matched_benign_block_controls(
    calendar: pd.DataFrame,
    daily_state: pd.DataFrame,
    blocks: pd.DataFrame,
    *,
    config: BlockTaxonomyConfig = BlockTaxonomyConfig(),
) -> pd.DataFrame:
    """Find whole benign windows with the closest observable pre-state.

    Candidate controls must be non-event windows of equal duration, and their
    pre-window cannot overlap an adverse event.  This produces the correct
    contrast set for asking what differentiates bad instances of a broad state
    from harmless lookalikes.
    """

    if config.controls_per_block <= 0:
        return pd.DataFrame()
    events = attach_event_blocks(
        calendar,
        max_event_days=config.max_event_days,
    )
    features = [name for name in daily_state.columns if name not in KEY_COLUMNS]
    merged = events.merge(
        daily_state, on=list(KEY_COLUMNS), how="left", validate="one_to_one"
    )
    rows: list[dict[str, object]] = []
    for block in blocks.itertuples(index=False):
        local = merged.loc[
            merged["side_name"].eq(block.side_name)
            & merged["archetype_policy_key"].eq(block.archetype_policy_key)
        ].sort_values("day", kind="stable")
        usable = [name for name in features if local[name].notna().any()]
        if len(local) < config.min_reference_days or not usable:
            continue
        start, end = pd.Timestamp(block.event_start), pd.Timestamp(block.event_end)
        duration = int((end - start).days) + 1
        pre_days = pd.date_range(
            start - pd.Timedelta(days=config.pre_days),
            start - pd.Timedelta(days=1),
            freq="D",
            tz="UTC",
        )
        event_pre = (
            local.set_index("day")
            .reindex(pre_days)[usable]
            .median(axis=0)
            .to_numpy(np.float64)
        )
        prior = local.loc[local["day"].lt(start), usable].to_numpy(np.float64)
        if len(prior) < config.min_reference_days:
            continue
        available = np.isfinite(prior).any(axis=0)
        median = np.zeros(prior.shape[1], dtype=np.float64)
        scale = np.ones(prior.shape[1], dtype=np.float64)
        if available.any():
            usable_prior = prior[:, available]
            median[available] = np.nanmedian(usable_prior, axis=0)
            scale[available] = np.maximum(
                np.nanquantile(usable_prior, 0.75, axis=0)
                - np.nanquantile(usable_prior, 0.25, axis=0),
                1e-4,
            )
        event_pre = np.nan_to_num((event_pre - median) / scale, nan=0.0)
        candidate_starts = local.loc[
            local["day"].between(
                local["day"].min() + pd.Timedelta(days=config.pre_days), end
            ),
            "day",
        ].drop_duplicates()
        candidates: list[tuple[pd.Timestamp, float]] = []
        event_days = set(local.loc[local["adverse_event"], "day"])
        for candidate_start in candidate_starts:
            candidate_window = pd.date_range(
                candidate_start,
                candidate_start + pd.Timedelta(days=duration - 1),
                freq="D",
                tz="UTC",
            )
            candidate_pre = pd.date_range(
                candidate_start - pd.Timedelta(days=config.pre_days),
                candidate_start - pd.Timedelta(days=1),
                freq="D",
                tz="UTC",
            )
            if candidate_start >= start or any(
                day in event_days for day in candidate_window.union(candidate_pre)
            ):
                continue
            control_pre = (
                local.set_index("day")
                .reindex(candidate_pre)[usable]
                .median(axis=0)
                .to_numpy(np.float64)
            )
            if not np.isfinite(control_pre).any():
                continue
            control_z = np.nan_to_num((control_pre - median) / scale, nan=0.0)
            distance = float(np.sqrt(np.mean((event_pre - control_z) ** 2)))
            candidates.append((candidate_start, distance))
        for rank, (control_start, distance) in enumerate(
            sorted(candidates, key=lambda item: item[1])[: config.controls_per_block],
            start=1,
        ):
            rows.append(
                {
                    "side_name": block.side_name,
                    "archetype_policy_key": block.archetype_policy_key,
                    "event_block": block.event_block,
                    "event_start": start,
                    "event_end": end,
                    "control_rank": rank,
                    "control_start": control_start,
                    "control_end": control_start + pd.Timedelta(days=duration - 1),
                    "pre_state_distance": distance,
                }
            )
    return pd.DataFrame(rows)
