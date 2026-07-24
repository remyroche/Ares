#!/usr/bin/env python3
"""Compare local residual-state overlays on the V9 champion.

RuleFit, contrastive subgroup, Bayesian Rule List, recursive partitioning,
shallow LGBM, and compact MLP arms share identical chronological folds,
features, labels, controls, and overlay search.  LGBM/MLP also model onset and
persistent episode phases. July prototypes are diagnostic-only matched controls
and never enter model fitting.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.mixture import GaussianMixture

from extreme_price_movements.residual_rule_models import (
    RobustMatrixTransform,
    build_rule_arm,
    matched_benign_controls,
    matched_benign_period_controls,
)
from extreme_price_movements.unsupervised_regime_learning.economic_relevance import (
    EconomicRegimeRelevanceConfig,
    materialize_composite_features,
    run_economic_regime_relevance,
)
from scripts import run_meta_residual_event_balanced_error_overlay as base


ARMS = (
    "rulefit",
    "contrastive_subgroup",
    "brl",
    "model_based_recursive_partition",
    "episode_lgbm",
    "episode_lgbm_contrastive",
    "episode_lgbm_adverse_subtypes",
    "episode_lgbm_subtype_moe",
    "episode_lgbm_pooled_reliability",
    "episode_mlp",
)
TOP10_PERIOD_EVENT_TARGET = "top10_adverse_period_target"
TOP10_MARKET_PERIOD_EVENT_TARGET = "top10_market_adverse_period_target"
GLOBAL_MARKET_EPISODE_RISK = "global_market_episode_risk"
GLOBAL_MARKET_EPISODE_RISK_PCT = "global_market_episode_risk_percentile"
SIDE_MARKET_EPISODE_RISK = "side_market_episode_risk"
SIDE_MARKET_EPISODE_RISK_PCT = "side_market_episode_risk_percentile"
POOLED_MECHANISM_COLUMN = "pooled_mechanism_bin"
POOLED_RELIABILITY_FEATURES = (
    "mechanism_reliability_risk",
    "mechanism_reliability_local_support",
    "mechanism_reliability_side_support",
    "mechanism_reliability_global_support",
)


def _utc_ns(values: pd.Series | pd.Index | np.ndarray) -> pd.Series:
    """Normalize Arrow/Pandas timestamp units before causal as-of joins.

    Parquet reads may preserve ``datetime64[us, UTC]`` while internally built
    daily opens are nanosecond timestamps.  ``merge_asof`` requires identical
    units, even though both represent the same UTC instants.
    """

    return pd.to_datetime(values, utc=True).astype("datetime64[ns, UTC]")
OVERLAY_CANDIDATE_COLUMNS = (
    "side_name",
    "archetype_policy_key",
    "model_arm",
    "model_target",
    "risk_variant",
    "threshold",
    "mode",
    "alpha",
    "flagged_parent_rows",
    "activity_ratio",
    "overall_ev_delta",
    "event_ev_delta",
    "normal_ev_delta",
    "positive_ev_rate_delta",
    "objective",
    "promotable",
    "target_prevalence",
    "risk_precision",
    "risk_lift",
    "risk_fpr",
    "event_blocks",
    "recognized_event_blocks",
    "event_block_recall",
    "adjusted_selected_rows",
    "adjusted_mean_ev",
    "adjusted_positive_ev_rate",
    "adjusted_clean_precision",
    "adjusted_event_mean_ev",
    "adjusted_normal_mean_ev",
    "adjusted_mean_month_ev",
    "adjusted_std_month_ev",
    "adjusted_worst_month_ev",
    "oof_event_cells",
    "oof_event_cells_intervened",
    "oof_event_cells_improved",
    "oof_event_intervention_recall",
    "oof_event_improvement_recall",
    "promotable_pre_episode_intervention",
)
# A residual overlay is an adverse *period* model, not an individual-trade
# loss model.  It scores the currently observable local candidate stream, then
# applies that one state score to the already-admitted parent top-10 rows.  The
# broader top-20 context retains cross-sectional support and dispersion without
# allowing below-top-10 outcomes to define the target.
PERIOD_CONTEXT_FLOOR = 0.80
PERIOD_STATE_FEATURES = (
    "period_context_rows",
    "period_parent_rank_q90",
    "period_parent_rank_iqr",
    "period_meta_score_q90",
    "period_meta_score_iqr",
    "period_hit_probability_q90",
    "period_hit_probability_iqr",
)
# Episode overlays should distinguish a mature adverse state from an onset or
# recovery transition.  These sources are observable market-state features;
# they deliberately exclude target, residual, and recent realised-performance
# fields.  The local side x archetype model decides which trajectories matter.
EPISODE_TRAJECTORY_SOURCE_FEATURES = (
    "negative_breadth_pct",
    "extreme_negative_breadth_pct",
    "breadth_dispersion",
    "median_alt_minus_btc",
    "btc_resilience_alt_weakness",
    "short_covering_score_market",
    "flush_recovery_state",
    "funding_deleveraging_divergence",
    "post_flush_leverage_rebuild",
    "mkt_regime_change__oi_contraction__cumulative_change_2d",
    "mkt_regime_change__eth_correlation__cumulative_change_2d",
    "mkt_regime_change__btc_alt_relative_strength__cumulative_change_2d",
)
# A difficult period is defined by how the observable market state evolves,
# not by one candidate row.  Keep this basis deliberately compact: 6/24h
# captures an abrupt transition, while 48h separates a persistent regime from
# a one-day shock.  Every lookup remains strictly as-of the decision clock.
EPISODE_TRAJECTORY_HOURS = (6, 24, 48)
EPISODE_TRAJECTORY_MAX_ASOF_LAG = pd.Timedelta(minutes=90)
MARKET_PANEL_FEATURE_BATCH = 12
ADVERSE_SUBTYPE_PREFIX = "episode_adverse_subtype"
MARKET_ADVERSE_SUBTYPE_PREFIX = "market_adverse_subtype"
EPISODE_TRAJECTORY_FEATURES = tuple(
    name
    for source in EPISODE_TRAJECTORY_SOURCE_FEATURES
    for name in (
        *(
            f"episode_traj__{source}__delta_{hours}h"
            for hours in EPISODE_TRAJECTORY_HOURS
        ),
        f"episode_traj__{source}__velocity_accel_6h_vs_24h",
        f"episode_traj__{source}__velocity_accel_24h_vs_48h",
        f"episode_traj__{source}__trend_agreement_48h",
        f"episode_traj__{source}__trend_intensity_48h",
        f"episode_traj__{source}__state_variability_48h",
    )
)
# A pooled market detector is deliberately narrow: it can use only observable
# market/transition families. Local side x archetype arms receive its frozen
# score as context and decide whether that broad state is relevant. This avoids
# making every local arm rediscover the same rare systemic episode from a small
# number of daily observations.
MARKET_EPISODE_FEATURE_TOKENS = (
    "mkt_",
    "market_",
    "breadth",
    "corr_",
    "correlation",
    "dispersion",
    "synchronization",
    "btc_",
    "eth_",
    "funding",
    "oi_",
    "liquidation",
    "deleverag",
    "flush",
    "short_cover",
    "rv_",
    "volat",
    "volume",
    "regime_change",
)
DEFAULT_GROUPS = (
    ("long", "long_volcompression_wideslow_candidate"),
    ("short", "short_default_clean_path"),
)
MECHANISM_TOKENS = {
    "liquidation_deleveraging": (
        "systemic_deleveraging", "oi_flush", "long_flush", "price_down_oi_down",
        "deleveraging", "oi_drawdown",
    ),
    "short_covering_rebound": (
        "short_covering", "flush_recovery", "price_up_oi_down", "breadth_recovery",
        "price_recovery", "failed_downside",
    ),
    "breadth_fragmentation": (
        "negative_breadth", "dispersion", "new_low", "fragmentation",
    ),
    "correlation_breakdown": (
        "corr_eth", "corr_btc", "correlation", "pc1_variance", "synchronization",
    ),
    "funding_dislocation": ("funding",),
    "predicted_tape_ev_divergence": (
        "expected_directional_ev_divergence", "expected_bullish_tape_adverse_ev",
        "expected_timestamp_ev_sign_disagreement",
    ),
    "predicted_persistent_damage": (
        "expected_persistent_subthreshold_damage", "expected_persistent_material_nontail",
    ),
}


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_safe_json(payload), indent=2, sort_keys=True) + "\n")


def _trajectory_source_dependencies(features: list[str]) -> list[str]:
    """Return primitive state fields needed by selected trajectory columns."""

    requested = set(features)
    return [
        source
        for source in EPISODE_TRAJECTORY_SOURCE_FEATURES
        if any(name.startswith(f"episode_traj__{source}__") for name in requested)
    ]


def _market_episode_candidates(columns: list[str]) -> list[str]:
    """Return observable market-state candidates for the pooled daily model."""

    return [
        name
        for name in dict.fromkeys([
            *EPISODE_TRAJECTORY_SOURCE_FEATURES,
            *EPISODE_TRAJECTORY_FEATURES,
            *columns,
    ])
        if name not in {GLOBAL_MARKET_EPISODE_RISK, GLOBAL_MARKET_EPISODE_RISK_PCT}
        and any(token in name.lower() for token in MARKET_EPISODE_FEATURE_TOKENS)
    ][:160]


def _retain_episode_columns(
    frame: pd.DataFrame,
    observable_features: list[str],
) -> pd.DataFrame:
    """Release unrelated joined-ledger columns before daily state aggregation."""

    required = set(base.KEYS) | {
        "day",
        "parent_rank_v9",
        "ev_after_1pct",
        "clean_exec",
        "dirty_positive",
        "full_path_bad_mae_1r",
        "timeout",
        base.EVENT,
        base.TARGET,
        base.SIDE_EVENT,
        TOP10_PERIOD_EVENT_TARGET,
        TOP10_MARKET_PERIOD_EVENT_TARGET,
        "market_adverse_period",
        "market_adverse_cell_count",
        "episode_phase",
        "episode_block",
        "episode_phase_offset_days",
        "market_episode_phase",
        "market_episode_block",
    }
    required.update(name for name in frame.columns if name.startswith("episode_"))
    required.update(observable_features)
    keep = [name for name in frame.columns if name in required]
    drop = [name for name in frame.columns if name not in required]
    if drop:
        frame = frame.drop(columns=drop)
    return frame.loc[:, keep]


def _trajectory_reference_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the shared observable market clock used by all local overlays.

    A side x archetype candidate stream may be intermittent.  Its membership
    must not determine what a six- or 24-hour market change means.  This panel
    is collapsed from the full candidate universe by timestamp, before any
    side/archetype split or target fit, and carries only observable inputs.
    """

    return _observable_market_state_panel(frame, list(EPISODE_TRAJECTORY_SOURCE_FEATURES))


def _observable_market_state_panel(
    frame: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Collapse observable candidate fields to one full-universe market state."""

    sources = [name for name in dict.fromkeys(features) if name in frame]
    if not sources:
        return pd.DataFrame(columns=["__ts__"])
    # The joined OOF ledger can have nearly a million rows. Projecting every
    # market candidate at once makes a transient wide copy larger than the
    # final state panel. Aggregate small column batches instead; the exact
    # timestamp median is unchanged while peak memory is bounded.
    timestamps = pd.to_datetime(frame["__ts__"], utc=True)
    result = (
        pd.DataFrame({"__ts__": timestamps})
        .drop_duplicates("__ts__", keep="last")
        .sort_values("__ts__", kind="stable")
        .reset_index(drop=True)
    )
    for offset in range(0, len(sources), MARKET_PANEL_FEATURE_BATCH):
        chunk = sources[offset : offset + MARKET_PANEL_FEATURE_BATCH]
        values = frame.loc[:, ["__ts__", *chunk]].copy()
        values["__ts__"] = timestamps.to_numpy()
        for name in chunk:
            values[name] = pd.to_numeric(values[name], errors="coerce").astype(
                np.float32
            )
        aggregate = (
            values.groupby("__ts__", observed=True, sort=True)[chunk]
            .median()
            .reset_index()
        )
        result = result.merge(
            aggregate, on="__ts__", how="left", validate="one_to_one"
        )
        del values, aggregate
    return result


def _attach_trajectory_reference(
    state: pd.DataFrame,
    reference: pd.DataFrame | None,
) -> pd.DataFrame:
    """Replace local source aggregates with the common observable market state."""

    if reference is None or reference.empty or state.empty:
        return state
    sources = [
        name for name in EPISODE_TRAJECTORY_SOURCE_FEATURES if name in reference.columns
    ]
    if not sources:
        return state
    values = reference.loc[:, ["__ts__", *sources]].copy()
    values["__ts__"] = pd.to_datetime(values["__ts__"], utc=True)
    merged = state.merge(
        values,
        on="__ts__",
        how="left",
        validate="one_to_one",
        suffixes=("", "__market_clock"),
    )
    for name in sources:
        replacement = f"{name}__market_clock"
        if replacement not in merged:
            continue
        merged[name] = pd.to_numeric(merged[replacement], errors="coerce").astype(np.float32)
        merged.drop(columns=replacement, inplace=True)
    return merged


def _add_episode_trajectory_features(
    state: pd.DataFrame,
    *,
    history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add causal transition and persistence features to daily state frames.

    The lookup for a timestamp ``t`` is strictly as-of ``t - horizon``.  A
    caller may pass earlier fit states while scoring a later fold; even then,
    the as-of cutoff prevents values after the required historical point from
    entering the feature.  Missing history remains missing rather than being
    silently backfilled from a future timestamp.
    """

    # Candidate unions can contain an observable source through both the base
    # registry and a trajectory dependency. Keep the right-most, explicitly
    # materialized value before constructing an as-of matrix.
    output = state.loc[:, ~state.columns.duplicated(keep="last")].copy()
    # Daily-state builders may reserve trajectory columns before calling this
    # helper.  Recreate them as one contiguous block below rather than leaving
    # duplicate placeholders beside the materialized values.
    output = output.drop(
        columns=[name for name in EPISODE_TRAJECTORY_FEATURES if name in output],
        errors="ignore",
    )
    if output.empty:
        return pd.concat(
            [
                output,
                pd.DataFrame(
                    {
                        name: np.full(len(output), np.nan, dtype=np.float32)
                        for name in EPISODE_TRAJECTORY_FEATURES
                    },
                    index=output.index,
                ),
            ],
            axis=1,
            copy=False,
        )
    output["__ts__"] = pd.to_datetime(output["__ts__"], utc=True)
    source_columns = [
        source for source in EPISODE_TRAJECTORY_SOURCE_FEATURES
        if source in output.columns
    ]
    trajectory_values = {
        name: np.full(len(output), np.nan, dtype=np.float32)
        for name in EPISODE_TRAJECTORY_FEATURES
    }
    if not source_columns:
        return pd.concat(
            [output, pd.DataFrame(trajectory_values, index=output.index)],
            axis=1,
            copy=False,
        )

    source_history = output.loc[:, ["__ts__", *source_columns]].copy()
    if history is not None and not history.empty:
        history = history.loc[:, ~history.columns.duplicated(keep="last")]
        history_columns = [name for name in source_columns if name in history.columns]
        if history_columns:
            prior = history.loc[:, ["__ts__", *history_columns]].copy()
            prior["__ts__"] = pd.to_datetime(prior["__ts__"], utc=True)
            for name in source_columns:
                if name not in prior:
                    prior[name] = np.float32(np.nan)
            source_history = pd.concat(
                [prior.loc[:, ["__ts__", *source_columns]], source_history],
                ignore_index=True,
            )
    source_history = (
        source_history.sort_values("__ts__", kind="stable")
        .drop_duplicates("__ts__", keep="last")
    )
    query_base = output.loc[:, ["__ts__"]].sort_values("__ts__", kind="stable")
    for source in source_columns:
        observed = source_history.loc[:, ["__ts__", source]].copy()
        observed[source] = pd.to_numeric(observed[source], errors="coerce").astype(np.float32)
        observed = observed.loc[observed[source].notna()].sort_values("__ts__", kind="stable")
        if observed.empty:
            continue
        current = pd.to_numeric(output[source], errors="coerce").to_numpy(np.float32)
        deltas: dict[int, np.ndarray] = {}
        for hours in EPISODE_TRAJECTORY_HOURS:
            query = query_base.copy()
            query["__lookup_ts__"] = query["__ts__"] - pd.Timedelta(hours=hours)
            prior = pd.merge_asof(
                query.sort_values("__lookup_ts__", kind="stable"),
                observed.rename(columns={"__ts__": "__observed_ts__", source: "__value__"}),
                left_on="__lookup_ts__",
                right_on="__observed_ts__",
                direction="backward",
                allow_exact_matches=True,
                tolerance=EPISODE_TRAJECTORY_MAX_ASOF_LAG,
            ).set_index("__ts__")["__value__"]
            previous = output["__ts__"].map(prior).to_numpy(np.float32)
            delta = current - previous
            delta[~np.isfinite(current) | ~np.isfinite(previous)] = np.nan
            deltas[hours] = delta
            trajectory_values[f"episode_traj__{source}__delta_{hours}h"] = delta.astype(np.float32)
        velocity_6 = deltas[6] / np.float32(6.0)
        velocity_24 = deltas[24] / np.float32(24.0)
        velocity_48 = deltas[48] / np.float32(48.0)
        trajectory_values[f"episode_traj__{source}__velocity_accel_6h_vs_24h"] = (
            velocity_6 - velocity_24
        ).astype(np.float32)
        trajectory_values[f"episode_traj__{source}__velocity_accel_24h_vs_48h"] = (
            velocity_24 - velocity_48
        ).astype(np.float32)

        # Agreement is signed and bounded in [-1, 1].  It distinguishes a
        # persistent directional state from a choppy transition without
        # creating the unbounded products that previously made composite
        # magnitudes unstable.  Intensity retains the scale of the underlying
        # observable state, so LGBM can distinguish a weak agreement from a
        # large, coherent transition.
        velocities = np.column_stack((velocity_6, velocity_24, velocity_48))
        finite = np.isfinite(velocities)
        finite_count = finite.sum(axis=1)
        finite_velocity = np.where(finite, velocities, 0.0)
        signed_mean = finite_velocity.sum(axis=1) / np.maximum(finite_count, 1)
        abs_mean = np.abs(finite_velocity).sum(axis=1) / np.maximum(finite_count, 1)
        agreement = signed_mean / (abs_mean + np.float32(1e-6))
        agreement[finite_count < 2] = np.nan
        intensity = abs_mean
        intensity[finite_count < 2] = np.nan
        trajectory_values[f"episode_traj__{source}__trend_agreement_48h"] = agreement.astype(np.float32)
        trajectory_values[f"episode_traj__{source}__trend_intensity_48h"] = intensity.astype(np.float32)

        # Use only observations at or before the current state timestamp.
        # Rolling time variance is a direct observable measure of transition
        # instability; it is not a realized-performance feature.
        temporal = observed.set_index("__ts__")[source].astype(np.float32)
        variability = temporal.rolling("48h", min_periods=3).std()
        variability_at_state = output["__ts__"].map(variability).to_numpy(np.float32)
        trajectory_values[f"episode_traj__{source}__state_variability_48h"] = variability_at_state
    return pd.concat(
        [output, pd.DataFrame(trajectory_values, index=output.index)],
        axis=1,
        copy=False,
    )


def _attach_train_only_adverse_subtype_features(
    fit: pd.DataFrame,
    score: pd.DataFrame,
    features: list[str],
    *,
    target_column: str,
    seed: int,
    feature_prefix: str = ADVERSE_SUBTYPE_PREFIX,
    include_component_posteriors: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any], dict[str, Any] | None]:
    """Encode heterogeneous adverse daily episodes with a frozen local GMM.

    A single adverse-period label can contain liquidation, recovery, and slow
    deterioration episodes.  The mixture is deliberately fitted *only* on
    pre-entry state vectors of train-side adverse days.  Its posterior and
    density therefore tell the LGBM which observable adverse-state family a
    new day resembles; realised outcomes never enter the OOS transform.

    BIC selects one to three components within the fold.  Components with less
    than three train adverse days are rejected, preventing a one-off calendar
    episode from becoming a synthetic state feature.
    """

    report: dict[str, Any] = {
        "enabled": True,
        "fit_status": "unavailable",
        "selected_components": 0,
        "train_adverse_days": 0,
        "component_support": [],
        "features": list(features),
    }
    if not features or fit.empty:
        return fit, score, [], report, None
    positive = fit[target_column].to_numpy(np.int8) > 0
    n_positive = int(positive.sum())
    report["train_adverse_days"] = n_positive
    if n_positive < 6:
        report["fit_status"] = "insufficient_adverse_days"
        return fit, score, [], report, None

    x_fit = _matrix(fit, features)
    robust = RobustMatrixTransform().fit(x_fit)
    z_fit = robust.transform(x_fit)
    z_positive = z_fit[positive]
    max_components = min(3, max(1, n_positive // 3))
    candidates: list[tuple[float, GaussianMixture, np.ndarray]] = []
    for components in range(1, max_components + 1):
        try:
            gmm = GaussianMixture(
                n_components=components,
                covariance_type="diag",
                reg_covar=1e-3,
                n_init=4,
                max_iter=128,
                random_state=seed + components,
            ).fit(z_positive)
            labels = gmm.predict(z_positive)
            support = np.bincount(labels, minlength=components)
            if components > 1 and int(support.min()) < 3:
                continue
            candidates.append((float(gmm.bic(z_positive)), gmm, support))
        except (ValueError, np.linalg.LinAlgError):
            continue
    if not candidates:
        report["fit_status"] = "no_stable_component_solution"
        return fit, score, [], report, None
    _, model, support = min(candidates, key=lambda item: item[0])
    encoder = {
        "features": list(features),
        "robust": robust,
        "model": model,
        "feature_prefix": str(feature_prefix),
        "include_component_posteriors": bool(include_component_posteriors),
        "feature_schema": "daily_observable_state",
        "transform_contract": (
            "robust-transform the stored observable daily-state feature order, then "
            "emit frozen GMM posteriors, entropy, and negative log-density"
        ),
    }
    fit_features = _materialize_adverse_subtype_features(fit, encoder)
    score_features = _materialize_adverse_subtype_features(score, encoder)
    fit = fit.copy()
    score = score.copy()
    fit = pd.concat([fit, fit_features], axis=1, copy=False)
    score = pd.concat([score, score_features], axis=1, copy=False)
    output_features = list(fit_features.columns)
    signatures: list[str] = []
    for component, centre in enumerate(model.means_):
        order = np.argsort(np.abs(centre))[::-1][: min(5, len(features))]
        signatures.append(
            ",".join(
                f"{features[index]}={float(centre[index]):+.2f}"
                for index in order
            )
        )
    report.update({
        "fit_status": "ok",
        "selected_components": int(model.n_components),
        "component_support": [int(value) for value in support.tolist()],
        "component_signatures": signatures,
        "bic": float(model.bic(z_positive)),
        "features": list(features),
    })
    return fit, score, output_features, report, encoder


def _materialize_adverse_subtype_features(
    state: pd.DataFrame,
    encoder: dict[str, Any],
) -> pd.DataFrame:
    """Transform an inference-time daily state with a frozen subtype encoder.

    This function deliberately accepts only the serialized observable feature
    order.  It is shared by model fitting and future forward materialization,
    so a saved subtype arm cannot silently use a different AE/GMM-style state
    transform at inference.
    """

    features = [str(name) for name in encoder["features"]]
    feature_prefix = str(encoder.get("feature_prefix", ADVERSE_SUBTYPE_PREFIX))
    include_component_posteriors = bool(
        encoder.get("include_component_posteriors", True)
    )
    robust = encoder["robust"]
    model = encoder["model"]
    z = robust.transform(_matrix(state, features))
    posterior = model.predict_proba(z).astype(np.float32)
    eps = np.float32(1e-6)
    values: dict[str, np.ndarray] = {}
    if include_component_posteriors:
        # Component ids are safe only inside a model fitted with this exact
        # encoder. Cross-fold consumers must use invariant summaries below.
        values.update({
            f"{feature_prefix}_posterior_{component}": posterior[:, component]
            if component < int(model.n_components)
            else np.zeros(len(state), dtype=np.float32)
            for component in range(3)
        })
    values[f"{feature_prefix}_max_posterior"] = posterior.max(axis=1).astype(np.float32)
    entropy_scale = max(float(np.log(max(int(model.n_components), 2))), 1.0)
    values[f"{feature_prefix}_entropy"] = (
        -np.sum(posterior * np.log(np.maximum(posterior, eps)), axis=1)
    ).astype(np.float32) / np.float32(entropy_scale)
    values[f"{feature_prefix}_neg_log_density"] = (
        -model.score_samples(z)
    ).astype(np.float32)
    return pd.DataFrame(values, index=state.index, dtype=np.float32)


def _fit_same_fold_subtype_moe(
    fit_state: pd.DataFrame,
    score_state: pd.DataFrame,
    features: list[str],
    *,
    target_column: str,
    period_control_mode: str,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Fit one local period head per train-only adverse-state subtype.

    Component IDs are used only inside this exact fit/score pair.  The LGBM
    heads receive the observable state primitives, not component IDs, so the
    resulting risk is a learnable mechanism signature rather than a GMM label
    lookup.  The maximum head risk is conservative: any recognised adverse
    mechanism can demote a parent candidate, subject to normal OOF gates.
    """

    fit_aug, score_aug, _, report, encoder = _attach_train_only_adverse_subtype_features(
        fit_state, score_state, features, target_column=target_column, seed=seed,
    )
    if encoder is None or int(encoder["model"].n_components) < 2:
        raise ValueError("subtype_moe_requires_at_least_two_train_only_components")
    z = encoder["robust"].transform(_matrix(fit_aug, features))
    component = encoder["model"].predict(z)
    fit_scores: list[np.ndarray] = []
    score_scores: list[np.ndarray] = []
    bundles: list[dict[str, Any]] = []
    rules: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    support: list[int] = []
    for subtype in range(int(encoder["model"].n_components)):
        local_target = f"__same_fold_subtype_{subtype}_target"
        fit_aug[local_target] = (
            (fit_aug[target_column].to_numpy(np.int8) > 0) & (component == subtype)
        ).astype(np.int8)
        score_aug[local_target] = np.int8(0)
        n_positive = int(fit_aug[local_target].sum())
        if n_positive < 4:
            continue
        bundle, fit_score, score_score, _, local_rules, local_controls = _fit_arm(
            "episode_lgbm", fit_aug, score_aug, features, seed + subtype,
            target_column=local_target, period_control_mode=period_control_mode,
        )
        support.append(n_positive)
        bundles.append({"subtype": subtype, "bundle": bundle, "target": local_target})
        fit_scores.append(np.asarray(fit_score, dtype=np.float32))
        score_scores.append(np.asarray(score_score, dtype=np.float32))
        rules.extend({"subtype": subtype, **row} for row in local_rules)
        controls.extend({"subtype": subtype, **row} for row in local_controls)
    if not bundles:
        raise ValueError("subtype_moe_has_no_supported_mechanism_head")
    combined_fit = np.max(np.vstack(fit_scores), axis=0).astype(np.float32)
    combined_score = np.max(np.vstack(score_scores), axis=0).astype(np.float32)
    reference = np.sort(combined_fit[np.isfinite(combined_fit)])
    report.update({"mechanism_head_support": support, "mechanism_heads": len(bundles)})
    return ({"subtype_moe": bundles, "adverse_subtype_encoder": encoder}, combined_fit, combined_score, reference, rules, controls, report)


def _shrunk_mechanism_reliability(
    train: pd.DataFrame,
    score: pd.DataFrame,
    *,
    mechanism_column: str,
    target_column: str,
    local_group_columns: tuple[str, str] = ("side_name", "archetype_policy_key"),
    side_column: str = "side_name",
    shrinkage_k: float = 20.0,
) -> pd.DataFrame:
    """Return train-only global/side/local empirical-Bayes mechanism risk.

    Small local side x archetype cells are shrunk first to their side's rate
    for the same mechanism, then to the global rate.  No score outcome enters
    this calculation.  The support field makes any later policy use auditable.
    """

    train_required = list(dict.fromkeys([
        mechanism_column, target_column, *local_group_columns, side_column,
    ]))
    score_required = list(dict.fromkeys([
        mechanism_column, *local_group_columns, side_column,
    ]))
    missing = [name for name in train_required if name not in train]
    missing += [name for name in score_required if name not in score and name not in missing]
    if missing:
        raise KeyError(f"Mechanism reliability requires columns: {missing}")
    work = train.loc[:, train_required].copy()
    work[target_column] = pd.to_numeric(work[target_column], errors="coerce").fillna(0.0)
    global_stats = work.groupby(mechanism_column, observed=True)[target_column].agg(["mean", "count"])
    side_stats = work.groupby([side_column, mechanism_column], observed=True)[target_column].agg(["mean", "count"])
    local_stats = work.groupby([*local_group_columns, mechanism_column], observed=True)[target_column].agg(["mean", "count"])
    output = score.loc[:, [*local_group_columns, mechanism_column]].copy()
    global_mean = output[mechanism_column].map(global_stats["mean"]).fillna(float(work[target_column].mean()))
    global_count = output[mechanism_column].map(global_stats["count"]).fillna(0.0)
    side_index = pd.MultiIndex.from_frame(output.loc[:, [side_column, mechanism_column]])
    side_mean = pd.Series(side_index.map(side_stats["mean"]), index=output.index).fillna(global_mean)
    side_count = pd.Series(side_index.map(side_stats["count"]), index=output.index).fillna(0.0)
    side_weight = side_count / (side_count + float(shrinkage_k))
    side_shrunk = side_weight * side_mean + (1.0 - side_weight) * global_mean
    local_index = pd.MultiIndex.from_frame(output.loc[:, [*local_group_columns, mechanism_column]])
    local_mean = pd.Series(local_index.map(local_stats["mean"]), index=output.index).fillna(side_shrunk)
    local_count = pd.Series(local_index.map(local_stats["count"]), index=output.index).fillna(0.0)
    local_weight = local_count / (local_count + float(shrinkage_k))
    output["mechanism_reliability_risk"] = (
        local_weight * local_mean + (1.0 - local_weight) * side_shrunk
    ).astype(np.float32)
    output["mechanism_reliability_local_support"] = local_count.astype(np.float32)
    output["mechanism_reliability_side_support"] = side_count.astype(np.float32)
    output["mechanism_reliability_global_support"] = global_count.astype(np.float32)
    return output


def _fit_mechanism_density_bins(
    train: pd.DataFrame,
    score: pd.DataFrame,
    *,
    density_column: str = f"{MARKET_ADVERSE_SUBTYPE_PREFIX}_neg_log_density",
    entropy_column: str = f"{MARKET_ADVERSE_SUBTYPE_PREFIX}_entropy",
    bins: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Assign stable, train-derived pooled mechanism bins.

    The bin is based on density and uncertainty quantiles, not a GMM component
    label.  It is therefore comparable across chronological encoder refits and
    remains entirely observable at scoring time.
    """

    columns = [density_column, entropy_column]
    if any(name not in train or name not in score for name in columns):
        raise KeyError(f"Missing pooled mechanism columns: {columns}")
    fit = train.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    edges: dict[str, list[float]] = {}
    for name in columns:
        values = fit[name].dropna().to_numpy(np.float64)
        if len(values) < max(12, bins * 3):
            edges[name] = []
            continue
        quantiles = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bins + 1)))
        edges[name] = quantiles.tolist() if len(quantiles) >= 3 else []

    def _transform(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        density = pd.to_numeric(out[density_column], errors="coerce")
        entropy = pd.to_numeric(out[entropy_column], errors="coerce")
        if not edges[density_column] or not edges[entropy_column]:
            out["pooled_mechanism_bin"] = np.int16(-1)
            return out
        d = np.searchsorted(np.asarray(edges[density_column][1:-1]), density.fillna(-np.inf), side="right")
        e = np.searchsorted(np.asarray(edges[entropy_column][1:-1]), entropy.fillna(-np.inf), side="right")
        missing = density.isna() | entropy.isna()
        out["pooled_mechanism_bin"] = np.where(missing, -1, d * bins + e).astype(np.int16)
        return out

    contract = {"density_edges": edges[density_column], "entropy_edges": edges[entropy_column], "bins": int(bins)}
    return _transform(train), _transform(score), contract


def _market_values_at_daily_open(
    market: pd.DataFrame,
    days: pd.Series,
    *,
    columns: tuple[str, str] = (
        f"{MARKET_ADVERSE_SUBTYPE_PREFIX}_neg_log_density",
        f"{MARKET_ADVERSE_SUBTYPE_PREFIX}_entropy",
    ),
) -> pd.DataFrame:
    """Return causal market values at each UTC day open.

    The pooled reliability layer is a daily state model.  Sampling at day open
    prevents a later market observation from explaining a loss realised during
    the same day.  ``merge_asof`` also makes missing market bars explicit
    rather than silently carrying a future value backwards.
    """

    result = pd.DataFrame({"day": pd.Series(_utc_ns(days)).dt.floor("D")})
    result["__ts__"] = result["day"]
    available = [name for name in columns if name in market.columns]
    if not available:
        for name in columns:
            result[name] = np.float32(np.nan)
        return result.drop(columns="__ts__")
    reference = (
        market.loc[:, ["__ts__", *available]]
        .assign(__ts__=lambda frame: _utc_ns(frame["__ts__"]))
        .drop_duplicates("__ts__", keep="last")
        .sort_values("__ts__", kind="stable")
        .rename(columns={"__ts__": "__market_ts__"})
    )
    result = pd.merge_asof(
        result.sort_values("__ts__", kind="stable"),
        reference,
        left_on="__ts__",
        right_on="__market_ts__",
        direction="backward",
        tolerance=EPISODE_TRAJECTORY_MAX_ASOF_LAG,
    ).drop(columns=["__ts__", "__market_ts__"])
    for name in columns:
        if name not in result:
            result[name] = np.float32(np.nan)
        result[name] = pd.to_numeric(result[name], errors="coerce").astype(np.float32)
    return result


def _pooled_daily_target_frame(
    rows: pd.DataFrame,
    *,
    target_column: str,
    top10_floor: float,
) -> pd.DataFrame:
    """Collapse the all-archetype selected stream to one causal daily label.

    The target is retained solely for train-side empirical-Bayes priors.  It is
    never read from a score day.  This representation avoids treating several
    intraday rows from one difficult day as independent evidence.
    """

    required = ["day", "side_name", "archetype_policy_key", "parent_rank_v9", target_column]
    missing = [name for name in required if name not in rows]
    if missing:
        raise KeyError(
            "Pooled reliability requires prepared all-archetype daily labels: "
            f"{missing}"
        )
    selected = rows.loc[
        pd.to_numeric(rows["parent_rank_v9"], errors="coerce").ge(float(top10_floor)),
        required,
    ].copy()
    if selected.empty:
        return pd.DataFrame(columns=["day", "side_name", "archetype_policy_key", target_column])
    selected["day"] = pd.to_datetime(selected["day"], utc=True).dt.floor("D")
    selected[target_column] = pd.to_numeric(
        selected[target_column], errors="coerce"
    ).fillna(0.0).astype(np.float32)
    return (
        selected.groupby(
            ["day", "side_name", "archetype_policy_key"], observed=True, sort=True
        )[target_column]
        .max()
        .reset_index()
    )


def _causal_pooled_reliability(
    train: pd.DataFrame,
    fit_query: pd.DataFrame,
    score: pd.DataFrame,
    *,
    mechanism_column: str,
    target_column: str,
    shrinkage_k: float,
) -> pd.DataFrame:
    """Compute same-day-excluded reliability priors for train and score days.

    ``train`` values are expanding and exclude every outcome from the current
    UTC day.  ``score`` values use the frozen train aggregate.  This is the
    critical distinction between a valid train-derived prior and a daily-label
    lookup that would leak the local target into the local episode classifier.
    """

    keys = ["day", "side_name", "archetype_policy_key", mechanism_column]
    work = train.loc[:, [*keys, target_column]].copy()
    work["day"] = pd.to_datetime(work["day"], utc=True).dt.floor("D")
    work[target_column] = pd.to_numeric(work[target_column], errors="coerce").fillna(0.0)

    def _historical_stats(group_columns: list[str], prefix: str) -> pd.DataFrame:
        daily = (
            work.groupby(["day", *group_columns], observed=True, sort=True)[target_column]
            .agg(["sum", "count"])
            .reset_index()
        )
        ordered = daily.sort_values([*group_columns, "day"], kind="stable")
        grouped = ordered.groupby(group_columns, observed=True, sort=False)
        ordered[f"{prefix}_sum"] = grouped["sum"].cumsum() - ordered["sum"]
        ordered[f"{prefix}_count"] = grouped["count"].cumsum() - ordered["count"]
        return ordered.loc[:, ["day", *group_columns, f"{prefix}_sum", f"{prefix}_count"]]

    global_stats = _historical_stats([mechanism_column], "global")
    side_stats = _historical_stats(["side_name", mechanism_column], "side")
    local_stats = _historical_stats(
        ["side_name", "archetype_policy_key", mechanism_column], "local"
    )
    train_out = fit_query.loc[:, keys].copy()
    train_out["day"] = pd.to_datetime(train_out["day"], utc=True).dt.floor("D")
    train_out = train_out.merge(
        global_stats, on=["day", mechanism_column], how="left", validate="many_to_one"
    ).merge(
        side_stats, on=["day", "side_name", mechanism_column], how="left", validate="many_to_one"
    ).merge(
        local_stats,
        on=["day", "side_name", "archetype_policy_key", mechanism_column],
        how="left",
        validate="many_to_one",
    )

    frozen = _shrunk_mechanism_reliability(
        work,
        score,
        mechanism_column=mechanism_column,
        target_column=target_column,
        shrinkage_k=shrinkage_k,
    )
    global_count = pd.to_numeric(train_out["global_count"], errors="coerce").fillna(0.0)
    global_mean = pd.to_numeric(train_out["global_sum"], errors="coerce") / global_count.replace(0.0, np.nan)
    # A neutral prior is used only until the first resolved observation for the
    # mechanism.  It carries no current-day outcome information.
    global_mean = global_mean.fillna(0.5)
    side_count = pd.to_numeric(train_out["side_count"], errors="coerce").fillna(0.0)
    side_mean = pd.to_numeric(train_out["side_sum"], errors="coerce") / side_count.replace(0.0, np.nan)
    side_weight = side_count / (side_count + float(shrinkage_k))
    side_shrunk = side_weight * side_mean.fillna(global_mean) + (1.0 - side_weight) * global_mean
    local_count = pd.to_numeric(train_out["local_count"], errors="coerce").fillna(0.0)
    local_mean = pd.to_numeric(train_out["local_sum"], errors="coerce") / local_count.replace(0.0, np.nan)
    local_weight = local_count / (local_count + float(shrinkage_k))
    train_result = pd.DataFrame(index=train_out.index)
    train_result["mechanism_reliability_risk"] = (
        local_weight * local_mean.fillna(side_shrunk) + (1.0 - local_weight) * side_shrunk
    ).astype(np.float32)
    train_result["mechanism_reliability_local_support"] = local_count.astype(np.float32)
    train_result["mechanism_reliability_side_support"] = side_count.astype(np.float32)
    train_result["mechanism_reliability_global_support"] = global_count.astype(np.float32)
    return pd.concat(
        [train_result.reset_index(drop=True), frozen.loc[:, list(POOLED_RELIABILITY_FEATURES)].reset_index(drop=True)],
        keys=["train", "score"],
    )


def _attach_fold_local_pooled_reliability(
    fit_state: pd.DataFrame,
    score_state: pd.DataFrame,
    pooled_rows: pd.DataFrame,
    market_reference: pd.DataFrame,
    *,
    side: str,
    archetype: str,
    target_column: str,
    top10_floor: float,
    shrinkage_k: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Attach fold-local, partially pooled reliability features to a state pair."""

    history = _pooled_daily_target_frame(
        pooled_rows, target_column=target_column, top10_floor=top10_floor
    )
    cutoff = pd.to_datetime(fit_state["day"], utc=True).max() + pd.Timedelta(days=1)
    history = history.loc[history["day"].lt(cutoff)].copy()
    fit_market = _market_values_at_daily_open(
        market_reference, history["day"].drop_duplicates().reset_index(drop=True)
    )
    score_market = _market_values_at_daily_open(market_reference, score_state["day"])
    fit_market, score_market, bin_contract = _fit_mechanism_density_bins(
        fit_market, score_market
    )
    history = history.merge(
        fit_market.loc[:, ["day", POOLED_MECHANISM_COLUMN]],
        on="day", how="left", validate="many_to_one",
    )
    fit_lookup = _market_values_at_daily_open(market_reference, fit_state["day"])
    _, fit_lookup, _ = _fit_mechanism_density_bins(fit_market, fit_lookup)
    fit_lookup = fit_lookup.loc[:, ["day", POOLED_MECHANISM_COLUMN]]
    fit_local = fit_state.loc[:, ["day"]].copy()
    fit_local["side_name"] = str(side)
    fit_local["archetype_policy_key"] = str(archetype)
    fit_local = fit_local.merge(fit_lookup, on="day", how="left", validate="one_to_one")
    score_local = score_state.loc[:, ["day"]].copy()
    score_local["side_name"] = str(side)
    score_local["archetype_policy_key"] = str(archetype)
    score_local = score_local.merge(
        score_market.loc[:, ["day", POOLED_MECHANISM_COLUMN]],
        on="day", how="left", validate="one_to_one",
    )
    combined = _causal_pooled_reliability(
        history,
        fit_local,
        score_local,
        mechanism_column=POOLED_MECHANISM_COLUMN,
        target_column=target_column,
        shrinkage_k=shrinkage_k,
    )
    train_features = combined.xs("train").reset_index(drop=True)
    score_features = combined.xs("score").reset_index(drop=True)
    fit = pd.concat([fit_state.reset_index(drop=True), train_features], axis=1, copy=False)
    score = pd.concat([score_state.reset_index(drop=True), score_features], axis=1, copy=False)
    fit[POOLED_MECHANISM_COLUMN] = fit_local[POOLED_MECHANISM_COLUMN].to_numpy(np.int16)
    score[POOLED_MECHANISM_COLUMN] = score_local[POOLED_MECHANISM_COLUMN].to_numpy(np.int16)
    report = {
        "status": "ok",
        "historical_daily_cells": int(len(history)),
        "mechanism_contract": bin_contract,
        "mean_train_reliability": float(train_features["mechanism_reliability_risk"].mean()),
        "mean_score_reliability": float(score_features["mechanism_reliability_risk"].mean()),
    }
    return fit, score, report


def _groups(
    text: str,
    train: pd.DataFrame | None = None,
    *,
    min_rows: int = 1_200,
    max_groups: int = 0,
) -> list[tuple[str, str]]:
    if not text or str(text).strip().lower() == "all":
        if train is None:
            return list(DEFAULT_GROUPS)
        counts = (
            train.groupby(["side_name", "archetype_policy_key"], observed=True)
            .size()
            .sort_values(ascending=False)
        )
        groups = [
            (str(side), str(archetype))
            for (side, archetype), rows in counts.items()
            if int(rows) >= int(min_rows)
        ]
        return groups[: int(max_groups)] if int(max_groups) > 0 else groups
    if str(text).strip().lower() == "default":
        return list(DEFAULT_GROUPS)
    result = []
    for item in text.split(","):
        side, archetype = item.strip().split("::", 1)
        result.append((side, archetype))
    return result


def _episode_phase_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach outcome-only episode phases for temporal diagnosis.

    They never enter an inference matrix.  Their purpose is to test whether a
    mechanism separates onset, persistent stress, and the immediate recovery
    period rather than appearing useful only after an episode is complete.
    """

    output = frame.copy()
    output["episode_phase"] = "normal"
    output["episode_block"] = np.int32(-1)
    output["episode_phase_offset_days"] = np.int16(-1)
    output["episode_onset_target"] = np.int8(0)
    output["episode_persistent_target"] = np.int8(0)
    output["episode_recovery_target"] = np.int8(0)
    group_columns = ["side_name", "archetype_policy_key"]
    for _, index in output.groupby(group_columns, observed=True).groups.items():
        local = output.loc[index]
        active_days = np.sort(local.loc[local[base.EVENT].gt(0), "day"].dropna().unique())
        if not len(active_days):
            continue
        block = -1
        previous: pd.Timestamp | None = None
        for day in active_days:
            stamp = pd.Timestamp(day)
            if previous is None or (stamp - previous) > pd.Timedelta(days=1):
                block += 1
                offset = 0
            else:
                offset += 1
            active = local.index[local["day"].eq(stamp) & local[base.EVENT].gt(0)]
            output.loc[active, "episode_block"] = np.int32(block)
            output.loc[active, "episode_phase_offset_days"] = np.int16(offset)
            output.loc[active, "episode_phase"] = "onset" if offset == 0 else "persistent"
            output.loc[active, "episode_onset_target"] = np.int8(offset == 0)
            output.loc[active, "episode_persistent_target"] = np.int8(offset > 0)
            recovery_day = stamp + pd.Timedelta(days=1)
            recovery = local.index[
                local["day"].eq(recovery_day) & local[base.EVENT].eq(0)
            ]
            if len(recovery):
                output.loc[recovery, "episode_block"] = np.int32(block)
                output.loc[recovery, "episode_phase_offset_days"] = np.int16(offset + 1)
                output.loc[recovery, "episode_phase"] = "recovery"
                output.loc[recovery, "episode_recovery_target"] = np.int8(1)
            previous = stamp
    return output


def _attach_top10_period_targets(frame: pd.DataFrame, *, top10_floor: float) -> pd.DataFrame:
    """Attach outcome-only side x archetype period labels to a score frame.

    ``base.EVENT`` is computed from the aggregate daily top-10 stream.  The
    helper repeats that period outcome over its member rows only for training;
    it is never an inference feature.
    """

    high_rank = pd.to_numeric(frame["parent_rank_v9"], errors="coerce").ge(
        top10_floor
    )
    frame[TOP10_PERIOD_EVENT_TARGET] = (
        high_rank & frame[base.EVENT].eq(1)
    ).astype(np.int8)
    frame[f"episode_onset_{TOP10_PERIOD_EVENT_TARGET}"] = (
        frame["episode_phase"].eq("onset")
        & frame[TOP10_PERIOD_EVENT_TARGET].eq(1)
    ).astype(np.int8)
    frame[f"episode_persistent_{TOP10_PERIOD_EVENT_TARGET}"] = (
        frame["episode_phase"].eq("persistent")
        & frame[TOP10_PERIOD_EVENT_TARGET].eq(1)
    ).astype(np.int8)
    return frame


def _attach_market_period_targets(
    frame: pd.DataFrame, *, top10_floor: float, min_adverse_cells: int = 2
) -> pd.DataFrame:
    """Attach a secondary broad-market episode label without using row losses.

    A market episode is present only when at least ``min_adverse_cells``
    distinct side x archetype daily residual streams are adverse together.  It
    is an outcome-only *training label*; inference sees only observable market
    and local-state features.  The local period target remains primary.
    """

    cells = (
        frame.groupby(["day", "side_name", "archetype_policy_key"], observed=True)[
            base.EVENT
        ]
        .max()
        .reset_index()
    )
    daily = (
        cells.groupby("day", observed=True)[base.EVENT]
        .sum()
        .rename("market_adverse_cell_count")
        .reset_index()
    )
    daily["market_adverse_period"] = daily["market_adverse_cell_count"].ge(
        min_adverse_cells
    ).astype(np.int8)
    # Assign field arrays explicitly so callers retain their original frame
    # object and no target column can accidentally enter a feature projection
    # through an implicit merge result.
    lookup = daily.set_index("day")
    frame["market_adverse_cell_count"] = frame["day"].map(
        lookup["market_adverse_cell_count"]
    ).fillna(0).astype(np.int8)
    frame["market_adverse_period"] = frame["day"].map(
        lookup["market_adverse_period"]
    ).fillna(0).astype(np.int8)
    high_rank = pd.to_numeric(frame["parent_rank_v9"], errors="coerce").ge(
        top10_floor
    )
    frame[TOP10_MARKET_PERIOD_EVENT_TARGET] = (
        high_rank & frame["market_adverse_period"].eq(1)
    ).astype(np.int8)
    return frame


def _attach_market_episode_phases(frame: pd.DataFrame) -> pd.DataFrame:
    """Add global onset/persistence labels for the secondary market target."""

    daily = (
        frame.loc[:, ["day", "market_adverse_period"]]
        .drop_duplicates("day")
        .sort_values("day", kind="stable")
        .copy()
    )
    daily["market_episode_phase"] = "normal"
    daily["market_episode_block"] = np.int32(-1)
    previous: pd.Timestamp | None = None
    block = -1
    for index, row in daily.loc[daily["market_adverse_period"].gt(0)].iterrows():
        day = pd.Timestamp(row["day"])
        onset = previous is None or day - previous > pd.Timedelta(days=1)
        if onset:
            block += 1
        daily.loc[index, "market_episode_phase"] = "onset" if onset else "persistent"
        daily.loc[index, "market_episode_block"] = np.int32(block)
        previous = day
    lookup = daily.set_index("day")
    frame["market_episode_phase"] = frame["day"].map(
        lookup["market_episode_phase"]
    ).fillna("normal")
    frame["market_episode_block"] = frame["day"].map(
        lookup["market_episode_block"]
    ).fillna(-1).astype(np.int32)
    frame[f"episode_onset_{TOP10_MARKET_PERIOD_EVENT_TARGET}"] = (
        frame["market_episode_phase"].eq("onset")
        & frame[TOP10_MARKET_PERIOD_EVENT_TARGET].eq(1)
    ).astype(np.int8)
    frame[f"episode_persistent_{TOP10_MARKET_PERIOD_EVENT_TARGET}"] = (
        frame["market_episode_phase"].eq("persistent")
        & frame[TOP10_MARKET_PERIOD_EVENT_TARGET].eq(1)
    ).astype(np.int8)
    return frame


def _period_state_frame(
    context_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
    features: list[str],
    *,
    target_column: str,
    event_column: str,
    history_state: pd.DataFrame | None = None,
    trajectory_reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one causal local state per timestamp for a period overlay.

    ``context_rows`` is the same side x archetype parent top-20 stream.  It
    supplies only pre-entry feature distributions.  ``decision_rows`` is the
    parent top-10 stream and supplies the outcome-only training label and the
    rows to which the resulting state score is broadcast.  Keeping these two
    populations distinct prevents an individual winning or losing trade from
    defining a market-period feature while preserving the actual top-10
    deployment decision universe.
    """

    derived = set(PERIOD_STATE_FEATURES) | set(EPISODE_TRAJECTORY_FEATURES)
    raw_features = [
        name for name in features
        if name not in derived and name in context_rows.columns
    ]
    # Feature selection may legitimately retain only timestamp-distribution
    # summaries.  Build the state from a harmless observable anchor in that
    # case; the anchor never reaches the fitted matrix unless it was selected.
    if not raw_features:
        raw_features = ["parent_rank_v9"]
    state = base._timestamp_training_frame(
        context_rows,
        raw_features,
        target_column=target_column,
        event_column=event_column,
    ).drop(columns=[target_column, event_column, "day", "ev_after_1pct", "clean_exec"])

    context_grouped = context_rows.groupby("__ts__", observed=True, sort=True)
    support = context_grouped.size().rename("period_context_rows")
    state = state.merge(support.reset_index(), on="__ts__", how="left", validate="one_to_one")
    for source, q90_name, iqr_name in (
        ("parent_rank_v9", "period_parent_rank_q90", "period_parent_rank_iqr"),
        ("score_meta_base_soft_label", "period_meta_score_q90", "period_meta_score_iqr"),
        ("hit_probability", "period_hit_probability_q90", "period_hit_probability_iqr"),
    ):
        if source not in context_rows.columns:
            state[q90_name] = np.float32(np.nan)
            state[iqr_name] = np.float32(np.nan)
            continue
        values = pd.to_numeric(context_rows[source], errors="coerce")
        grouped = values.groupby(context_rows["__ts__"], observed=True, sort=True)
        q90 = grouped.quantile(0.90).rename(q90_name)
        iqr = (grouped.quantile(0.75) - grouped.quantile(0.25)).rename(iqr_name)
        state = state.merge(q90.reset_index(), on="__ts__", how="left", validate="one_to_one")
        state = state.merge(iqr.reset_index(), on="__ts__", how="left", validate="one_to_one")

    # This transform happens before outcome labels are merged.  The caller can
    # supply the earlier fold state while scoring later OOS timestamps, but the
    # as-of lookup itself is strictly bounded at t - horizon.
    state = _attach_trajectory_reference(state, trajectory_reference)
    state = _add_episode_trajectory_features(
        state,
        history=trajectory_reference if trajectory_reference is not None else history_state,
    )

    # Labels remain defined only from parent-top-10 rows.  This is deliberately
    # separate from the context aggregation above: no realized label/output is
    # ever used to construct an inference-time state feature.
    decision_grouped = decision_rows.groupby("__ts__", observed=True, sort=True)
    labels = decision_grouped.agg(
        day=("day", "first"),
        ev_after_1pct=("ev_after_1pct", "mean"),
        clean_exec=("clean_exec", "mean"),
    )
    labels[target_column] = decision_grouped[target_column].max().astype(np.int8)
    if event_column != target_column:
        labels[event_column] = decision_grouped[event_column].max().astype(np.int8)
    state = state.merge(labels.reset_index(), on="__ts__", how="inner", validate="one_to_one")
    return state


def _daily_episode_state_frame(
    decision_rows: pd.DataFrame,
    features: list[str],
    *,
    target_column: str,
    event_column: str,
    market_reference: pd.DataFrame,
) -> pd.DataFrame:
    """Build one pre-open market signature per local side x archetype day.

    The target remains that side/archetype's aggregate adverse-day outcome, but
    every predictor is taken at the UTC day open from the shared observable
    market clock. This removes repeated intraday rows from the learning unit
    and prevents a later state within a bad day from explaining its earlier
    losses. Scores are subsequently broadcast to that day's parent top-10
    decisions only for policy evaluation.
    """

    decision_rows = decision_rows.copy()
    decision_rows["__ts__"] = _utc_ns(decision_rows["__ts__"])
    if "day" not in decision_rows.columns:
        decision_rows["day"] = decision_rows["__ts__"].dt.floor("D")
    else:
        decision_rows["day"] = _utc_ns(decision_rows["day"]).dt.floor("D")
    market_reference = market_reference.copy()
    market_reference["__ts__"] = _utc_ns(market_reference["__ts__"])
    daily = (
        decision_rows.groupby("day", observed=True, sort=True)
        .agg(
            ev_after_1pct=("ev_after_1pct", "mean"),
            clean_exec=("clean_exec", "mean"),
        )
        .reset_index()
    )
    target = decision_rows.groupby("day", observed=True, sort=True)[target_column].max()
    daily[target_column] = daily["day"].map(target).fillna(0).astype(np.int8)
    if event_column != target_column:
        event = decision_rows.groupby("day", observed=True, sort=True)[event_column].max()
        daily[event_column] = daily["day"].map(event).fillna(0).astype(np.int8)
    daily["__ts__"] = _utc_ns(daily["day"])

    raw_features = [
        name for name in features
        if name not in set(PERIOD_STATE_FEATURES)
        and name not in set(EPISODE_TRAJECTORY_FEATURES)
        and name in market_reference.columns
    ]
    # Most inputs are deliberately pooled from the full observable market
    # clock.  A small number of context fields, such as the frozen side-level
    # episode score, are nevertheless side-specific by construction.  Those
    # fields are already one pre-entry value per side/day and can therefore be
    # merged from this local decision population without turning row outcomes
    # into features.
    local_features = [
        name for name in features
        if name not in set(PERIOD_STATE_FEATURES)
        and name not in set(EPISODE_TRAJECTORY_FEATURES)
        and name not in raw_features
        and name in decision_rows.columns
    ]
    state = daily.loc[:, ["__ts__", "day", "ev_after_1pct", "clean_exec", target_column]].copy()
    if event_column != target_column:
        state[event_column] = daily[event_column].to_numpy(np.int8)
    if raw_features:
        reference = market_reference.loc[:, ["__ts__", *raw_features]].copy()
        reference = reference.rename(columns={"__ts__": "__market_ts__"}).sort_values("__market_ts__", kind="stable")
        state = pd.merge_asof(
            state.sort_values("__ts__", kind="stable"),
            reference,
            left_on="__ts__",
            right_on="__market_ts__",
            direction="backward",
            tolerance=EPISODE_TRAJECTORY_MAX_ASOF_LAG,
        ).drop(columns="__market_ts__")
    if local_features:
        local_daily = (
            decision_rows.loc[:, ["day", *local_features]]
            .groupby("day", observed=True, sort=True)[local_features]
            .median()
            .reset_index()
        )
        state = state.merge(local_daily, on="day", how="left", validate="one_to_one")
    missing_features = [name for name in features if name not in state]
    if missing_features:
        state = pd.concat(
            [
                state,
                pd.DataFrame(
                    {
                        name: np.full(len(state), np.nan, dtype=np.float32)
                        for name in missing_features
                    },
                    index=state.index,
                ),
            ],
            axis=1,
            copy=False,
        )
    state = _attach_trajectory_reference(state, market_reference)
    state = _add_episode_trajectory_features(state, history=market_reference)
    return state


def _build_unsupervised_relevance_frame(
    train: pd.DataFrame,
    candidates: list[str],
    *,
    max_rows: int,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Create the bounded train-only input used for composite discovery."""
    # The regime learner is deliberately restricted to observable candidates;
    # its outcome labels are used only while scoring train-side relevance.
    primitive = candidates[: min(len(candidates), 180)]
    # ``run_economic_regime_relevance`` constructs several temporary copies
    # while scoring nonlinear states.  Passing the full joined ledger (hundreds
    # of unused columns) needlessly multiplies memory use.  Keep every train
    # row and every required field, but project to its exact contract first.
    relevance_columns = list(
        dict.fromkeys(
            [
                "__ts__",
                "side_name",
                "archetype_policy_key",
                "parent_rank_v9",
                "ev_after_1pct",
                "clean_exec",
                "dirty_positive",
                "full_path_bad_mae_1r",
                "timeout",
                "stop_or_adverse",
                *primitive,
            ]
        )
    )
    # Both economic tasks are defined entirely inside the frozen parent top-20:
    # demote bad rows already admitted by top-10, and promote clean rows from
    # top-20 excluding top-10.  Retaining the original percentile avoids an
    # accidental re-rank of this filtered population while cutting the costly
    # relevance scan to the only rows that can influence an overlay.
    relevance_rank = pd.to_numeric(train["parent_rank_v9"], errors="coerce")

    def _time_spread_indices(frame: pd.DataFrame, budget: int) -> np.ndarray:
        if budget <= 0 or frame.empty:
            return np.empty(0, dtype=np.int64)
        source = frame.sort_values("__ts__", kind="stable")
        if len(source) <= budget:
            return source.index.to_numpy(np.int64, copy=False)
        parts = np.array_split(np.arange(len(source)), 3)
        counts = np.full(3, budget // 3, dtype=np.int64)
        counts[: budget % 3] += 1
        chosen = [
            part[np.linspace(0, len(part) - 1, num=min(len(part), int(count)), dtype=np.int64)]
            for part, count in zip(parts, counts, strict=True)
            if len(part) and count
        ]
        return source.index.to_numpy(np.int64, copy=False)[np.concatenate(chosen)]

    # Select from a narrow metadata view before materializing the feature
    # matrix.  This keeps discovery bounded even when the joined model ledger
    # contains hundreds of context columns.
    candidate_meta = train.loc[
        relevance_rank.ge(0.80),
        ["__ts__", "ev_after_1pct", "dirty_positive"],
    ].copy()
    if max_rows > 0 and len(candidate_meta) > max_rows:
        adverse = (
            pd.to_numeric(candidate_meta["ev_after_1pct"], errors="coerce").lt(0.0)
            | pd.to_numeric(candidate_meta["dirty_positive"], errors="coerce").fillna(0).gt(0.0)
        )
        # Keep a material benign pool so promotion states are assessed along
        # with failure states.  The allocation is deterministic and time-spread
        # within each class.
        normal_budget = min(int(max_rows) // 5, int((~adverse).sum()))
        adverse_budget = int(max_rows) - normal_budget
        sampled_indices = np.concatenate(
            [
                _time_spread_indices(candidate_meta.loc[adverse], adverse_budget),
                _time_spread_indices(candidate_meta.loc[~adverse], normal_budget),
            ]
        )
        relevance_frame = train.reindex(
            index=sampled_indices, columns=relevance_columns
        ).copy()
    else:
        relevance_frame = train.reindex(
            index=candidate_meta.index, columns=relevance_columns
        ).copy()
    return relevance_frame, primitive, relevance_columns


def _discover_unsupervised_episode_composites(
    relevance_frame: pd.DataFrame,
    primitive: list[str],
    relevance_columns: list[str],
    *,
    output: Path,
    max_rows: int,
) -> tuple[dict[str, list[str]], dict[str, list[dict[str, Any]]]]:
    """Discover local nonlinear composites from an isolated train-only sample."""

    relevance = run_economic_regime_relevance(
        relevance_frame,
        primitive,
        config=EconomicRegimeRelevanceConfig(
            # The overlay operates on one global V9 candidate stream.  Do not
            # let monthly rank normalization manufacture different top-k
            # populations when discovery is meant to explain parent top-10
            # failures across episodes.
            score_col="parent_rank_v9",
            score_is_percentile=True,
            month_col="",
            max_features_per_group=48,
            max_features_for_composites=8,
            max_composites_per_group_task=24,
            lgbm_max_features=48,
            lgbm_n_estimators=120,
            lgbm_max_depth=3,
            min_group_rows=500,
            lgbm_min_rows=350,
            random_state=20260714,
        ),
    )
    feature_map = {
        str(key): [str(name) for name in values]
        for key, values in (relevance.ebm_candidate_manifest.get("feature_map") or {}).items()
    }
    wanted = {
        name.removesuffix("__intensity")
        for values in feature_map.values()
        for name in values
    }
    definitions = [
        definition for definition in relevance.composite_definitions
        if str(definition.get("name")) in wanted
    ]
    # Definitions are local by construction.  Do not materialize every local
    # definition across the full 900k-row ledger: it creates a wide global
    # frame even though a definition can only be consumed by its own side x
    # archetype arm.  Each group materializes its frozen definitions just
    # before fitting, which retains exact train/OOS parity at far lower memory.
    definitions_by_group: dict[str, list[dict[str, Any]]] = {}
    for definition in definitions:
        key = f"{definition.get('side_name')}|{definition.get('archetype_policy_key')}"
        definitions_by_group.setdefault(key, []).append(definition)
    relevance.feature_metrics.to_csv(output / "unsupervised_feature_relevance.csv", index=False)
    relevance.composite_metrics.to_csv(output / "unsupervised_composite_relevance.csv", index=False)
    relevance.lgbm_feature_metrics.to_csv(output / "unsupervised_local_lgbm_features.csv", index=False)
    relevance.lgbm_model_metrics.to_csv(output / "unsupervised_local_lgbm_models.csv", index=False)
    _write_json(output / "unsupervised_composite_contract.json", {
        "feature_map": feature_map,
        "composite_definitions": definitions,
        "training_only": True,
        "oos_transform": "same frozen composite definitions; no OOS relevance fitting",
        "relevance_input_columns": relevance_columns,
        "relevance_input_rows": int(len(relevance_frame)),
        "max_rows": int(max_rows),
    })
    del relevance
    return feature_map, definitions_by_group


def _materialize_local_composites(
    frame: pd.DataFrame,
    definitions: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[str]]:
    """Attach a local group's already-frozen composite definitions only."""

    if not definitions or frame.empty:
        return frame, []
    composites = materialize_composite_features(
        frame, definitions, include_intensity=True
    )
    return pd.concat([frame, composites], axis=1, copy=False), list(composites.columns)


def _matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return frame.reindex(columns=features).to_numpy(dtype=np.float32, copy=True)


def _boost_matched_controls(
    weights: np.ndarray, controls: np.ndarray, *, multiplier: float = 2.0
) -> np.ndarray:
    out = np.asarray(weights, dtype=np.float32).copy()
    out[controls] *= np.float32(multiplier)
    return (out / max(float(out.mean()), 1e-8)).astype(np.float32)


def _fit_arm(
    arm_name: str,
    fit: pd.DataFrame,
    score: pd.DataFrame,
    features: list[str],
    seed: int,
    target_column: str = base.TARGET,
    period_control_mode: str = "timestamp",
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    x_fit = _matrix(fit, features)
    x_score = _matrix(score, features)
    robust = RobustMatrixTransform().fit(x_fit)
    z_fit = robust.transform(x_fit)
    z_score = robust.transform(x_score)
    weights = base._sample_weights(
        fit, target_column=target_column, event_column=target_column
    )
    blocks = base._event_block_ids(fit, event_column=target_column)
    contrastive = arm_name == "episode_lgbm_contrastive"
    positive = fit[target_column].to_numpy(np.int8) > 0
    block_count = max(int(np.unique(blocks[positive & (blocks >= 0)]).size), 1)
    controls_per_event = (
        max(4, min(128, int(np.ceil(float(positive.sum()) / float(block_count)))))
        if contrastive
        else 4
    )
    # Every supported research target denotes a daily adverse episode.  Rows
    # inside that day are only a deployment population to which the daily state
    # score is broadcast; they must never change the control definition.
    period_target = (
        "period" in target_column
        or target_column in {
            "episode_onset_target",
            "episode_persistent_target",
        }
    )
    if period_target and period_control_mode == "episode_windows":
        controls, control_report = matched_benign_period_controls(
            z_fit,
            fit[target_column],
            blocks,
            fit["day"].to_numpy(),
            controls_per_event=controls_per_event,
        )
    else:
        controls, control_report = matched_benign_controls(
            z_fit,
            fit[target_column],
            blocks,
            controls_per_event=controls_per_event,
        )
    weights = _boost_matched_controls(weights, controls)
    model = build_rule_arm(arm_name, seed=seed)
    train_mask = positive | controls if contrastive else np.ones(len(fit), dtype=bool)
    if train_mask.sum() < 40 or np.unique(positive[train_mask]).size < 2:
        raise ValueError("Insufficient matched episode/control rows for contrastive fit")
    model.fit(
        z_fit[train_mask],
        fit[target_column].to_numpy(np.int8)[train_mask],
        weights[train_mask],
        features,
    )
    fit_score = model.predict_proba(z_fit)
    score_value = model.predict_proba(z_score)
    reference = np.sort(fit_score[np.isfinite(fit_score)])
    bundle = {"robust": robust, "model": model, "features": features, "reference": reference}
    return bundle, fit_score, score_value, reference, model.describe(), control_report


def _prepare(train: pd.DataFrame, valid: pd.DataFrame, args: argparse.Namespace, config: base.Config):
    events = base._load_event_cells(args.event_calendar, args.extension_calendar)
    train = base._attach_event_target(train, events)
    valid = base._attach_event_target(valid, events)
    expected = base._fit_expected_clean_baseline(train, top10_floor=config.top10_floor)
    train, train_calendar = base._v9_residual_calendar(
        train, top10_floor=config.top10_floor, expected_clean_baseline=expected
    )
    valid, valid_calendar = base._v9_residual_calendar(
        valid, top10_floor=config.top10_floor, expected_clean_baseline=expected
    )
    for frame in (train, valid):
        frame[base.SIDE_EVENT] = (
            frame.groupby(["day", "side_name"], observed=True)[base.EVENT]
            .transform("max").fillna(0).astype(np.int8)
        )
    train = _episode_phase_labels(train)
    valid = _episode_phase_labels(valid)
    for frame in (train, valid):
        # The primary regime target is deliberately period-level. A daily
        # side x archetype event is derived from aggregate selected-trade
        # surprise and EV by ``_v9_residual_calendar``; every top-10 row in
        # that period receives the same outcome label. Individual losses are
        # not used to define this target.
        _attach_top10_period_targets(frame, top10_floor=config.top10_floor)
        _attach_market_period_targets(frame, top10_floor=config.top10_floor)
        _attach_market_episode_phases(frame)
    return train, valid, train_calendar, valid_calendar, expected


def _load_pooled_daily_context_decisions(
    args: argparse.Namespace,
    config: base.Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a narrow all-archetype decision stream for daily context labels.

    The local overlay needs a full-universe market/side context, but loading a
    300-column residual state table for every candidate row is unnecessary and
    can exhaust memory.  This loader keeps only causal parent rank, realised
    fields needed to *define daily train labels*, and the side/archetype keys.
    Observable market features are loaded separately as one row per timestamp.
    """

    if not bool(getattr(args, "direct_parent_rank", False)):
        raise ValueError(
            "--pooled-daily-context currently requires --direct-parent-rank so "
            "the compact source has an explicit causal parent rank"
        )
    source = Path(args.champion_ledger)
    schema = set(pq.read_schema(source).names)
    required = [
        *base.KEYS,
        "historical_rank",
        "ev_after_1pct",
        "clean_exec",
        "dirty_positive",
        "full_path_bad_mae_1r",
        "timeout",
    ]
    missing = [name for name in required if name not in schema]
    if missing:
        raise KeyError(
            "Compact pooled context source is missing required parent fields: "
            f"{missing}"
        )
    frame = pd.read_parquet(source, columns=required)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
    start = pd.Timestamp(config.train_start, tz="UTC")
    train_end = pd.Timestamp(config.train_end, tz="UTC")
    eval_end = pd.Timestamp(config.eval_end, tz="UTC")
    frame = frame.loc[frame["__ts__"].ge(start) & frame["__ts__"].lt(eval_end)]
    frame["parent_rank_v9"] = pd.to_numeric(
        frame["historical_rank"], errors="coerce"
    ).astype(np.float32)
    train = frame.loc[frame["__ts__"].lt(train_end)].copy()
    valid = frame.loc[frame["__ts__"].ge(train_end)].copy()
    train, valid, _, _, _ = _prepare(train, valid, args, config)
    return train, valid


def _load_pooled_market_clock(
    path: Path,
    config: base.Config,
    candidates: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load one compact observable market clock for pooled daily context.

    ``features_negative_residuals`` is already timestamp-level and causal.  It
    therefore provides a sufficient shared market representation without
    duplicating it onto every side/archetype candidate row.
    """

    if path is None or not path.exists():
        raise FileNotFoundError(
            "--pooled-daily-context requires the causal negative-residual "
            "feature parquet used by the local overlay"
        )
    schema = set(pq.read_schema(path).names)
    requested = [
        name for name in candidates
        if name in set(base.NEGATIVE_RESIDUAL_META_FEATURE_KEYS) and name in schema
    ]
    # ``_market_episode_candidates`` deliberately expands primitive fields into
    # derived trajectory names for model selection.  This loader reads the raw
    # causal parquet, where trajectories do not exist yet; retain only physical
    # columns and derive transitions later on the shared timestamp clock.
    requested = [
        name for name in _market_episode_candidates(requested) if name in schema
    ]
    if not requested:
        raise ValueError("No observable market features available for pooled daily context")
    market = pd.read_parquet(path, columns=requested)
    market.index = pd.to_datetime(market.index, utc=True, errors="coerce")
    market = market.loc[market.index.notna() & ~market.index.duplicated(keep="last")]
    market.index.name = "__ts__"
    market = market.reset_index().sort_values("__ts__", kind="stable")
    start = pd.Timestamp(config.train_start, tz="UTC")
    train_end = pd.Timestamp(config.train_end, tz="UTC")
    eval_end = pd.Timestamp(config.eval_end, tz="UTC")
    market = market.loc[market["__ts__"].ge(start) & market["__ts__"].lt(eval_end)]
    for name in requested:
        market[name] = pd.to_numeric(market[name], errors="coerce").astype(np.float32)
    return (
        market.loc[market["__ts__"].lt(train_end)].reset_index(drop=True),
        market.loc[market["__ts__"].ge(train_end)].reset_index(drop=True),
        requested,
    )


def _attach_pooled_market_context(
    local_market: pd.DataFrame,
    pooled_market: pd.DataFrame,
) -> pd.DataFrame:
    """Attach frozen pooled global daily context to a local market clock."""

    columns = [
        name for name in (GLOBAL_MARKET_EPISODE_RISK, GLOBAL_MARKET_EPISODE_RISK_PCT)
        if name in pooled_market.columns
    ]
    if not columns:
        return local_market
    context = pooled_market.loc[:, ["__ts__", *columns]].drop_duplicates(
        "__ts__", keep="last"
    )
    return local_market.merge(context, on="__ts__", how="left", validate="one_to_one")


def _fit_group_daily_episode_arm(
    train_rows: pd.DataFrame,
    valid_rows: pd.DataFrame,
    candidates: list[str],
    arm_name: str,
    config: base.Config,
    seed: int,
    *,
    target_column: str,
    period_control_mode: str,
    market_train: pd.DataFrame,
    market_valid: pd.DataFrame,
    pooled_train: pd.DataFrame | None = None,
    pooled_reliability_shrinkage_k: float = 20.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit one overlay arm on causal daily episode signatures, not trade rows."""

    local = train_rows.loc[train_rows["parent_rank_v9"].ge(config.top10_floor)].sort_values("__ts__")
    local_valid = valid_rows.loc[valid_rows["parent_rank_v9"].ge(config.top10_floor)].sort_values("__ts__")
    market_full = (
        pd.concat([market_train, market_valid], ignore_index=True)
        .sort_values("__ts__", kind="stable")
        .drop_duplicates("__ts__", keep="last")
    )
    state_candidates = list(dict.fromkeys([
        *candidates,
        *[name for name in EPISODE_TRAJECTORY_SOURCE_FEATURES if name in market_train.columns],
        *EPISODE_TRAJECTORY_FEATURES,
    ]))
    oof_parts: list[pd.DataFrame] = []
    model_rows: list[dict[str, Any]] = []
    rules: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    side_values = train_rows["side_name"].dropna().astype(str).unique()
    archetype_values = train_rows["archetype_policy_key"].dropna().astype(str).unique()
    if len(side_values) != 1 or len(archetype_values) != 1:
        raise ValueError("Daily local overlay requires exactly one side x archetype group")
    side = str(side_values[0])
    archetype = str(archetype_values[0])

    def _daily_screen_config(state: pd.DataFrame) -> base.Config:
        blocks = base._event_block_ids(state, event_column=target_column)
        positive_blocks = np.unique(blocks[(state[target_column].to_numpy(np.int8) > 0) & (blocks >= 0)])
        # Daily event samples have far fewer independent blocks than timestamp
        # rows. Cap model capacity by blocks rather than carrying the normal
        # 24-feature intraday budget into a rare-episode classifier.
        feature_budget = min(
            int(config.max_features),
            max(4, min(10, 2 * int(len(positive_blocks)))),
        )
        return replace(config, max_features=feature_budget)

    def _part(score_rows: pd.DataFrame, state: pd.DataFrame, scores: np.ndarray, reference: np.ndarray, fold_start: pd.Timestamp) -> pd.DataFrame:
        score_map = pd.Series(scores, index=state["day"])
        values = score_rows["day"].map(score_map).to_numpy(np.float32)
        columns = list(dict.fromkeys([
            *base.KEYS, "day", "parent_rank_v9", "ev_after_1pct", "clean_exec",
            base.EVENT, base.TARGET, target_column,
            *[name for name in ("episode_phase", "episode_block", "episode_phase_offset_days") if name in score_rows],
        ]))
        result = score_rows.loc[:, columns].copy()
        result["model_arm"] = arm_name
        result["model_target"] = target_column
        result[base.RISK_SCORE] = values
        result[base.RISK_PCT] = base._midrank(values, reference)
        diagnostic_columns = [
            name for name in (POOLED_MECHANISM_COLUMN, *POOLED_RELIABILITY_FEATURES)
            if name in state.columns
        ]
        if diagnostic_columns:
            diagnostics = state.loc[:, ["day", *diagnostic_columns]].drop_duplicates(
                "day", keep="last"
            )
            result = result.merge(diagnostics, on="day", how="left", validate="many_to_one")
        result["fold_start"] = fold_start
        return result

    for fold_index, fold_start in enumerate(base.FOLD_STARTS):
        fold_end = base.FOLD_STARTS[fold_index + 1] if fold_index + 1 < len(base.FOLD_STARTS) else pd.Timestamp(config.train_end, tz="UTC")
        fit_rows = local.loc[local["__ts__"].lt(fold_start - pd.Timedelta(days=2))]
        score_rows = local.loc[local["__ts__"].ge(fold_start) & local["__ts__"].lt(fold_end)]
        if score_rows.empty:
            continue
        fit_state = _daily_episode_state_frame(
            fit_rows, state_candidates, target_column=target_column,
            event_column=target_column, market_reference=market_train,
        )
        if len(fit_state) < 60 or int(fit_state[target_column].sum()) < 6:
            continue
        screen_config = _daily_screen_config(fit_state)
        selected, _ = base._screen_features(
            fit_state, state_candidates, screen_config, target_column=target_column,
            min_finite_rows=60,
        )
        if not selected:
            continue
        state_inputs = list(dict.fromkeys([
            *selected, *_trajectory_source_dependencies(selected),
        ]))
        fit_state = _daily_episode_state_frame(
            fit_rows, state_inputs, target_column=target_column,
            event_column=target_column, market_reference=market_train,
        )
        score_state = _daily_episode_state_frame(
            score_rows, state_inputs, target_column=target_column,
            event_column=target_column, market_reference=market_train,
        )
        model_features = selected
        fit_arm_name = arm_name
        pooled_reliability_report: dict[str, Any] = {"status": "not_requested"}
        if arm_name == "episode_lgbm_pooled_reliability":
            if pooled_train is None:
                model_rows.append({
                    "model_arm": arm_name, "stage": "oof_failed", "fold_start": fold_start,
                    "train_rows": len(fit_rows), "train_state_rows": len(fit_state),
                    "score_rows": len(score_rows), "features": len(selected),
                    "selected_features": "|".join(selected), "matched_control_rows": 0,
                    "model_target": target_column, "state_granularity": "daily_open",
                    "failure": "pooled_reliability_requires_pooled_daily_context",
                })
                continue
            fit_state, score_state, pooled_reliability_report = _attach_fold_local_pooled_reliability(
                fit_state, score_state, pooled_train, market_train,
                side=side, archetype=archetype, target_column=target_column,
                top10_floor=config.top10_floor,
                shrinkage_k=pooled_reliability_shrinkage_k,
            )
            model_features = list(dict.fromkeys([*selected, *POOLED_RELIABILITY_FEATURES]))
            fit_arm_name = "episode_lgbm"
        subtype_report: dict[str, Any] = {"fit_status": "not_requested", "selected_components": 0}
        subtype_encoder: dict[str, Any] | None = None
        if arm_name == "episode_lgbm_subtype_moe":
            try:
                bundle, _, score_values, reference, local_rules, local_controls, subtype_report = (
                    _fit_same_fold_subtype_moe(
                        fit_state, score_state, selected,
                        target_column=target_column,
                        period_control_mode=period_control_mode,
                        seed=seed + fold_index,
                    )
                )
                subtype_encoder = bundle.get("adverse_subtype_encoder")
            except Exception as exc:
                model_rows.append({
                    "model_arm": arm_name, "stage": "oof_failed", "fold_start": fold_start,
                    "train_rows": len(fit_rows), "train_state_rows": len(fit_state),
                    "score_rows": len(score_rows), "features": len(selected),
                    "selected_features": "|".join(selected), "matched_control_rows": 0,
                    "model_target": target_column, "state_granularity": "daily_open",
                    "failure": f"{type(exc).__name__}: {exc}",
                })
                continue
        elif arm_name == "episode_lgbm_adverse_subtypes":
            fit_state, score_state, subtype_features, subtype_report, subtype_encoder = (
                _attach_train_only_adverse_subtype_features(
                    fit_state,
                    score_state,
                    selected,
                    target_column=target_column,
                    seed=seed + fold_index,
                )
            )
            model_features = list(dict.fromkeys([*selected, *subtype_features]))
            # The new arm changes the observable representation, not the
            # underlying period classifier.  Keep the same constrained LGBM
            # architecture as the existing episode arm for an apples-to-apples
            # ablation.
            fit_arm_name = "episode_lgbm"
        try:
            if arm_name == "episode_lgbm_subtype_moe":
                raise StopIteration
            bundle, _, score_values, reference, local_rules, local_controls = _fit_arm(
                fit_arm_name, fit_state, score_state, model_features, seed + fold_index,
                target_column=target_column, period_control_mode=period_control_mode,
            )
            if subtype_encoder is not None:
                bundle["adverse_subtype_encoder"] = subtype_encoder
        except StopIteration:
            pass
        except Exception as exc:
            model_rows.append({
                "model_arm": arm_name, "stage": "oof_failed", "fold_start": fold_start,
                "train_rows": len(fit_rows), "train_state_rows": len(fit_state),
                "score_rows": len(score_rows), "features": len(model_features),
                "selected_features": "|".join(model_features), "matched_control_rows": 0,
                "model_target": target_column, "state_granularity": "daily_open",
                "adverse_subtype_status": subtype_report.get("fit_status"),
                "adverse_subtype_components": subtype_report.get("selected_components", 0),
                "adverse_subtype_support": "|".join(map(str, subtype_report.get("component_support", []))),
                "adverse_subtype_signatures": " || ".join(subtype_report.get("component_signatures", [])),
                "failure": f"{type(exc).__name__}: {exc}",
            })
            continue
        oof_parts.append(_part(score_rows, score_state, score_values, reference, fold_start))
        model_rows.append({
            "model_arm": arm_name, "stage": "oof", "fold_start": fold_start,
            "train_rows": len(fit_rows), "train_state_rows": len(fit_state),
            "score_rows": len(score_rows), "features": len(model_features),
            "selected_features": "|".join(model_features),
            "matched_control_rows": int(sum(row["control_rows"] for row in local_controls)),
            "model_target": target_column, "state_granularity": "daily_open",
            "adverse_subtype_status": subtype_report.get("fit_status"),
            "adverse_subtype_components": subtype_report.get("selected_components", 0),
            "adverse_subtype_support": "|".join(map(str, subtype_report.get("component_support", []))),
            "adverse_subtype_signatures": " || ".join(subtype_report.get("component_signatures", [])),
            "pooled_reliability_status": pooled_reliability_report.get("status"),
            "pooled_reliability_daily_cells": pooled_reliability_report.get("historical_daily_cells", 0),
        })
        rules.extend({"model_arm": arm_name, "stage": "oof", "fold_start": fold_start, **row} for row in local_rules)
        controls.extend({"model_arm": arm_name, "stage": "oof", "fold_start": fold_start, **row} for row in local_controls)

    if not oof_parts:
        return pd.DataFrame(), pd.DataFrame(model_rows), None, rules, controls

    full_state = _daily_episode_state_frame(
        local, state_candidates, target_column=target_column,
        event_column=target_column, market_reference=market_full,
    )
    selected, _ = base._screen_features(
        full_state, state_candidates, _daily_screen_config(full_state), target_column=target_column,
        min_finite_rows=60,
    )
    final = None
    if selected and not local_valid.empty:
        state_inputs = list(dict.fromkeys([
            *selected, *_trajectory_source_dependencies(selected),
        ]))
        full_state = _daily_episode_state_frame(
            local, state_inputs, target_column=target_column,
            event_column=target_column, market_reference=market_full,
        )
        valid_state = _daily_episode_state_frame(
            local_valid, state_inputs, target_column=target_column,
            event_column=target_column, market_reference=market_full,
        )
        model_features = selected
        fit_arm_name = arm_name
        pooled_reliability_report = {"status": "not_requested"}
        if arm_name == "episode_lgbm_pooled_reliability":
            if pooled_train is None:
                model_rows.append({
                    "model_arm": arm_name, "stage": "final_failed", "fold_start": config.train_end,
                    "train_rows": len(local), "train_state_rows": len(full_state),
                    "score_rows": len(local_valid), "features": len(selected),
                    "selected_features": "|".join(selected), "matched_control_rows": 0,
                    "model_target": target_column, "state_granularity": "daily_open",
                    "failure": "pooled_reliability_requires_pooled_daily_context",
                })
                return pd.concat(oof_parts, ignore_index=True), pd.DataFrame(model_rows), None, rules, controls
            full_state, valid_state, pooled_reliability_report = _attach_fold_local_pooled_reliability(
                full_state, valid_state, pooled_train, market_full,
                side=side, archetype=archetype, target_column=target_column,
                top10_floor=config.top10_floor,
                shrinkage_k=pooled_reliability_shrinkage_k,
            )
            model_features = list(dict.fromkeys([*selected, *POOLED_RELIABILITY_FEATURES]))
            fit_arm_name = "episode_lgbm"
        subtype_report = {"fit_status": "not_requested", "selected_components": 0}
        subtype_encoder = None
        if arm_name == "episode_lgbm_subtype_moe":
            try:
                bundle, _, score_values, reference, local_rules, local_controls, subtype_report = (
                    _fit_same_fold_subtype_moe(
                        full_state, valid_state, selected,
                        target_column=target_column,
                        period_control_mode=period_control_mode,
                        seed=seed + 10_000,
                    )
                )
                subtype_encoder = bundle.get("adverse_subtype_encoder")
            except Exception as exc:
                model_rows.append({
                    "model_arm": arm_name, "stage": "final_failed", "fold_start": config.train_end,
                    "train_rows": len(local), "train_state_rows": len(full_state),
                    "score_rows": len(local_valid), "features": len(selected),
                    "selected_features": "|".join(selected), "matched_control_rows": 0,
                    "model_target": target_column, "state_granularity": "daily_open",
                    "failure": f"{type(exc).__name__}: {exc}",
                })
                return pd.concat(oof_parts, ignore_index=True), pd.DataFrame(model_rows), None, rules, controls
        elif arm_name == "episode_lgbm_adverse_subtypes":
            full_state, valid_state, subtype_features, subtype_report, subtype_encoder = (
                _attach_train_only_adverse_subtype_features(
                    full_state,
                    valid_state,
                    selected,
                    target_column=target_column,
                    seed=seed + 10_000,
                )
            )
            model_features = list(dict.fromkeys([*selected, *subtype_features]))
            fit_arm_name = "episode_lgbm"
        try:
            if arm_name != "episode_lgbm_subtype_moe":
                bundle, _, score_values, reference, local_rules, local_controls = _fit_arm(
                    fit_arm_name, full_state, valid_state, model_features, seed + 10_000,
                    target_column=target_column, period_control_mode=period_control_mode,
                )
            if subtype_encoder is not None:
                bundle["adverse_subtype_encoder"] = subtype_encoder
            day_score = pd.Series(score_values, index=valid_state["day"])
            final = {
                "bundle": bundle,
                "index": local_valid.index.to_numpy(),
                "score": local_valid["day"].map(day_score).to_numpy(np.float32),
                "reference": reference,
            }
            model_rows.append({
                "model_arm": arm_name, "stage": "final", "fold_start": config.train_end,
                "train_rows": len(local), "train_state_rows": len(full_state),
                "score_rows": len(local_valid), "features": len(model_features),
                "selected_features": "|".join(model_features),
                "matched_control_rows": int(sum(row["control_rows"] for row in local_controls)),
                "model_target": target_column, "state_granularity": "daily_open",
                "adverse_subtype_status": subtype_report.get("fit_status"),
                "adverse_subtype_components": subtype_report.get("selected_components", 0),
                "adverse_subtype_support": "|".join(map(str, subtype_report.get("component_support", []))),
                "adverse_subtype_signatures": " || ".join(subtype_report.get("component_signatures", [])),
                "pooled_reliability_status": pooled_reliability_report.get("status"),
                "pooled_reliability_daily_cells": pooled_reliability_report.get("historical_daily_cells", 0),
            })
            rules.extend({"model_arm": arm_name, "stage": "final", "fold_start": config.train_end, **row} for row in local_rules)
            controls.extend({"model_arm": arm_name, "stage": "final", "fold_start": config.train_end, **row} for row in local_controls)
        except Exception as exc:
            model_rows.append({
                "model_arm": arm_name, "stage": "final_failed", "fold_start": config.train_end,
                "train_rows": len(local), "train_state_rows": len(full_state),
                "score_rows": len(local_valid), "features": len(model_features),
                "selected_features": "|".join(model_features), "matched_control_rows": 0,
                "model_target": target_column, "state_granularity": "daily_open",
                "adverse_subtype_status": subtype_report.get("fit_status"),
                "adverse_subtype_components": subtype_report.get("selected_components", 0),
                "adverse_subtype_support": "|".join(map(str, subtype_report.get("component_support", []))),
                "adverse_subtype_signatures": " || ".join(subtype_report.get("component_signatures", [])),
                "failure": f"{type(exc).__name__}: {exc}",
            })
    return pd.concat(oof_parts, ignore_index=True), pd.DataFrame(model_rows), final, rules, controls


def _daily_market_context_config(state: pd.DataFrame, config: base.Config) -> base.Config:
    """Bound global detector capacity by independent adverse market blocks."""

    return _daily_episode_context_config(state, config, "market_adverse_period")


def _daily_episode_context_config(
    state: pd.DataFrame,
    config: base.Config,
    target_column: str,
) -> base.Config:
    """Bound a daily context model by independent target-event blocks."""

    blocks = base._event_block_ids(state, event_column=target_column)
    positive = state[target_column].to_numpy(np.int8) > 0
    positive_blocks = np.unique(blocks[positive & (blocks >= 0)])
    feature_budget = min(
        int(config.max_features),
        max(4, min(10, 2 * int(len(positive_blocks)))),
    )
    return replace(config, max_features=feature_budget)


def _attach_daily_market_context(
    market: pd.DataFrame,
    daily_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Broadcast a causal daily pooled-state score onto the market clock."""

    result = market.copy()
    result["day"] = pd.to_datetime(result["__ts__"], utc=True).dt.floor("D")
    if daily_scores.empty:
        result[GLOBAL_MARKET_EPISODE_RISK] = np.float32(np.nan)
        result[GLOBAL_MARKET_EPISODE_RISK_PCT] = np.float32(np.nan)
        return result.drop(columns="day")
    lookup = daily_scores.drop_duplicates("day", keep="last").set_index("day")
    result[GLOBAL_MARKET_EPISODE_RISK] = result["day"].map(
        lookup[GLOBAL_MARKET_EPISODE_RISK]
    ).astype(np.float32)
    result[GLOBAL_MARKET_EPISODE_RISK_PCT] = result["day"].map(
        lookup[GLOBAL_MARKET_EPISODE_RISK_PCT]
    ).astype(np.float32)
    return result.drop(columns="day")


def _attach_daily_market_features(
    market: pd.DataFrame,
    daily_features: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Broadcast a frozen daily observable-state representation to the clock.

    The daily frame is indexed by UTC day and contains only train-fitted
    transforms of the observable market state.  This helper is deliberately
    generic so a local overlay can consume an episode-family representation
    without ever seeing the outcome labels that selected the training days.
    """

    result = market.copy()
    result["day"] = pd.to_datetime(result["__ts__"], utc=True).dt.floor("D")
    if "day" not in daily_features.columns:
        if "__ts__" not in daily_features.columns:
            raise KeyError("Daily market features require day or __ts__")
        daily_features = daily_features.copy()
        daily_features["day"] = pd.to_datetime(
            daily_features["__ts__"], utc=True
        ).dt.floor("D")
    available = [name for name in feature_columns if name in daily_features.columns]
    if not available:
        for name in feature_columns:
            result[name] = np.float32(np.nan)
        return result.drop(columns="day")
    lookup = daily_features.loc[:, ["day", *available]].drop_duplicates(
        "day", keep="last"
    )
    result = result.merge(lookup, on="day", how="left", validate="many_to_one")
    for name in feature_columns:
        if name not in result:
            result[name] = np.float32(np.nan)
    return result.drop(columns="day")


def _attach_daily_side_context(
    frame: pd.DataFrame,
    daily_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Broadcast frozen side-specific daily context to matching-side rows only."""

    result = frame.copy()
    result["day"] = pd.to_datetime(result["__ts__"], utc=True).dt.floor("D")
    if daily_scores.empty:
        result[SIDE_MARKET_EPISODE_RISK] = np.float32(np.nan)
        result[SIDE_MARKET_EPISODE_RISK_PCT] = np.float32(np.nan)
        return result
    lookup = daily_scores.drop_duplicates(["day", "side_name"], keep="last")
    result = result.merge(
        lookup.loc[:, [
            "day", "side_name", SIDE_MARKET_EPISODE_RISK,
            SIDE_MARKET_EPISODE_RISK_PCT,
        ]],
        on=["day", "side_name"],
        how="left",
        validate="many_to_one",
    )
    return result


def _fit_daily_episode_context(
    train_decisions: pd.DataFrame,
    valid_decisions: pd.DataFrame,
    candidates: list[str],
    config: base.Config,
    *,
    market_train: pd.DataFrame,
    market_valid: pd.DataFrame,
    target_column: str,
    risk_column: str,
    risk_pct_column: str,
    seed: int,
    context_label: str,
) -> tuple[pd.DataFrame, dict[str, Any] | None, pd.DataFrame, pd.DataFrame]:
    """Fit one chronological daily context detector on a preselected population.

    The population may be the whole selected market or one selected side.  It
    is still one state and one realized target per UTC day: individual rows
    merely supply the deployment population to which the score is broadcast.
    """

    market_full = (
        pd.concat([market_train, market_valid], ignore_index=True, copy=False)
        .sort_values("__ts__", kind="stable")
        .drop_duplicates("__ts__", keep="last")
    )
    state_candidates = _market_episode_candidates(candidates)
    oof_parts: list[pd.DataFrame] = []
    report_rows: list[dict[str, Any]] = []

    for fold_index, fold_start in enumerate(base.FOLD_STARTS):
        fold_end = (
            base.FOLD_STARTS[fold_index + 1]
            if fold_index + 1 < len(base.FOLD_STARTS)
            else pd.Timestamp(config.train_end, tz="UTC")
        )
        fit_rows = train_decisions.loc[
            train_decisions["__ts__"].lt(fold_start - pd.Timedelta(days=2))
        ]
        score_rows = train_decisions.loc[
            train_decisions["__ts__"].ge(fold_start)
            & train_decisions["__ts__"].lt(fold_end)
        ]
        if score_rows.empty:
            continue
        fit_state = _daily_episode_state_frame(
            fit_rows, state_candidates, target_column=target_column,
            event_column=target_column, market_reference=market_train,
        )
        positives = int(fit_state[target_column].sum())
        if len(fit_state) < 60 or positives < 6:
            report_rows.append({
                "context": context_label, "stage": "oof_skipped", "fold_start": fold_start,
                "daily_train_rows": len(fit_state), "adverse_days": positives,
                "reason": "insufficient_daily_episode_support",
            })
            continue
        local_config = _daily_episode_context_config(fit_state, config, target_column)
        selected, _ = base._screen_features(
            fit_state, state_candidates, local_config,
            target_column=target_column, min_finite_rows=60,
        )
        if not selected:
            report_rows.append({
                "context": context_label, "stage": "oof_skipped", "fold_start": fold_start,
                "daily_train_rows": len(fit_state), "adverse_days": positives,
                "reason": "no_observable_market_features",
            })
            continue
        state_inputs = list(dict.fromkeys([
            *selected, *_trajectory_source_dependencies(selected),
        ]))
        fit_state = _daily_episode_state_frame(
            fit_rows, state_inputs, target_column=target_column,
            event_column=target_column, market_reference=market_train,
        )
        score_state = _daily_episode_state_frame(
            score_rows, state_inputs, target_column=target_column,
            event_column=target_column, market_reference=market_train,
        )
        try:
            _, _, score_values, reference, _, _ = _fit_arm(
                "episode_lgbm", fit_state, score_state, selected, seed + fold_index,
                target_column=target_column, period_control_mode="episode_windows",
            )
        except Exception as exc:
            report_rows.append({
                "context": context_label, "stage": "oof_failed", "fold_start": fold_start,
                "daily_train_rows": len(fit_state), "adverse_days": positives,
                "features": "|".join(selected), "reason": f"{type(exc).__name__}: {exc}",
            })
            continue
        part = score_state.loc[:, ["day", target_column]].copy()
        part[risk_column] = score_values.astype(np.float32)
        part[risk_pct_column] = base._midrank(score_values, reference)
        part["fold_start"] = fold_start
        oof_parts.append(part)
        report_rows.append({
            "context": context_label, "stage": "oof", "fold_start": fold_start,
            "daily_train_rows": len(fit_state), "daily_score_rows": len(score_state),
            "adverse_days": positives, "features": "|".join(selected),
        })

    oof = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame(
        columns=["day", target_column, risk_column, risk_pct_column]
    )
    bundle: dict[str, Any] | None = None
    final_scores = pd.DataFrame(columns=["day", risk_column, risk_pct_column])
    full_state = _daily_episode_state_frame(
        train_decisions, state_candidates, target_column=target_column,
        event_column=target_column, market_reference=market_full,
    )
    positives = int(full_state[target_column].sum()) if target_column in full_state else 0
    if len(full_state) >= 60 and positives >= 6 and not valid_decisions.empty:
        local_config = _daily_episode_context_config(full_state, config, target_column)
        selected, _ = base._screen_features(
            full_state, state_candidates, local_config,
            target_column=target_column, min_finite_rows=60,
        )
        if selected:
            state_inputs = list(dict.fromkeys([
                *selected, *_trajectory_source_dependencies(selected),
            ]))
            full_state = _daily_episode_state_frame(
                train_decisions, state_inputs, target_column=target_column,
                event_column=target_column, market_reference=market_full,
            )
            valid_state = _daily_episode_state_frame(
                valid_decisions, state_inputs, target_column=target_column,
                event_column=target_column, market_reference=market_full,
            )
            try:
                bundle, _, score_values, reference, _, _ = _fit_arm(
                    "episode_lgbm", full_state, valid_state, selected, seed + 10_000,
                    target_column=target_column, period_control_mode="episode_windows",
                )
                final_scores = valid_state.loc[:, ["day"]].copy()
                final_scores[risk_column] = score_values.astype(np.float32)
                final_scores[risk_pct_column] = base._midrank(score_values, reference)
                report_rows.append({
                    "context": context_label, "stage": "final", "fold_start": config.train_end,
                    "daily_train_rows": len(full_state), "daily_score_rows": len(valid_state),
                    "adverse_days": positives, "features": "|".join(selected),
                })
            except Exception as exc:
                report_rows.append({
                    "context": context_label, "stage": "final_failed", "fold_start": config.train_end,
                    "daily_train_rows": len(full_state), "adverse_days": positives,
                    "features": "|".join(selected), "reason": f"{type(exc).__name__}: {exc}",
                })

    report = pd.DataFrame(report_rows)
    if not oof.empty:
        blocks = base._event_block_ids(oof.rename(columns={target_column: base.EVENT}), event_column=base.EVENT)
        risk = base._risk_metrics(
            oof.rename(columns={target_column: base.TARGET}),
            oof[risk_pct_column].to_numpy(np.float32), 0.90,
            target=oof[target_column].to_numpy(bool), blocks=blocks,
        )
        report = pd.concat([report, pd.DataFrame([{
            "context": context_label, "stage": "oof_summary", **risk,
        }])], ignore_index=True)
    return oof, bundle, final_scores, report


def _fit_global_market_episode_context(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    candidates: list[str],
    config: base.Config,
    *,
    market_train: pd.DataFrame,
    market_valid: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any] | None, pd.DataFrame]:
    """Fit a pooled adverse-market-day detector under chronological folds.

    This is intentionally a daily market-state model, not a row error model.
    Its label is a day with multiple simultaneous adverse local cells. The
    resulting frozen risk score is only a candidate context feature for later
    local models; it does not alter parent ranks by itself.
    """

    target = "market_adverse_period"
    train_decisions = train.loc[
        train["parent_rank_v9"].ge(config.top10_floor)
    ].sort_values("__ts__", kind="stable")
    valid_decisions = valid.loc[
        valid["parent_rank_v9"].ge(config.top10_floor)
    ].sort_values("__ts__", kind="stable")
    oof, bundle, final_scores, report = _fit_daily_episode_context(
        train_decisions, valid_decisions, candidates, config,
        market_train=market_train, market_valid=market_valid,
        target_column=target, risk_column=GLOBAL_MARKET_EPISODE_RISK,
        risk_pct_column=GLOBAL_MARKET_EPISODE_RISK_PCT, seed=seed,
        context_label="global_market",
    )
    return (
        _attach_daily_market_context(market_train, oof),
        _attach_daily_market_context(market_valid, final_scores),
        oof,
        bundle,
        report,
    )


def _fit_global_market_adverse_subtypes(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    candidates: list[str],
    config: base.Config,
    *,
    market_train: pd.DataFrame,
    market_valid: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any] | None, pd.DataFrame, list[str]]:
    """Discover broad adverse market-state families before local attribution.

    The training population is all top-10 rows across archetypes, reduced to
    one observable market state per day.  The GMM is fit only on train-side
    days labelled as broad market-adverse.  OOS rows receive frozen posterior
    features; the local side x archetype model decides whether any family is
    relevant.  This is intentionally a representation, not an overlay gate.
    """

    target = "market_adverse_period"
    prefix = MARKET_ADVERSE_SUBTYPE_PREFIX
    output_columns = [
        f"{prefix}_max_posterior",
        f"{prefix}_entropy",
        f"{prefix}_neg_log_density",
    ]
    train_selected = train.loc[train["parent_rank_v9"].ge(config.top10_floor)].sort_values(
        "__ts__", kind="stable"
    )
    valid_selected = valid.loc[valid["parent_rank_v9"].ge(config.top10_floor)].sort_values(
        "__ts__", kind="stable"
    )
    state_candidates = _market_episode_candidates(candidates)
    market_full = (
        pd.concat([market_train, market_valid], ignore_index=True, copy=False)
        .sort_values("__ts__", kind="stable")
        .drop_duplicates("__ts__", keep="last")
    )
    oof_parts: list[pd.DataFrame] = []
    report_rows: list[dict[str, Any]] = []

    for fold_index, fold_start in enumerate(base.FOLD_STARTS):
        fold_end = (
            base.FOLD_STARTS[fold_index + 1]
            if fold_index + 1 < len(base.FOLD_STARTS)
            else pd.Timestamp(config.train_end, tz="UTC")
        )
        fit_rows = train_selected.loc[
            train_selected["__ts__"].lt(fold_start - pd.Timedelta(days=2))
        ]
        score_rows = train_selected.loc[
            train_selected["__ts__"].ge(fold_start)
            & train_selected["__ts__"].lt(fold_end)
        ]
        if score_rows.empty:
            continue
        fit_state = _daily_episode_state_frame(
            fit_rows, state_candidates, target_column=target,
            event_column=target, market_reference=market_train,
        )
        if len(fit_state) < 60 or int(fit_state[target].sum()) < 6:
            report_rows.append({
                "stage": "oof_skipped", "fold_start": fold_start,
                "daily_train_rows": len(fit_state),
                "adverse_days": int(fit_state[target].sum()),
                "reason": "insufficient_market_adverse_support",
            })
            continue
        local_config = _daily_market_context_config(fit_state, config)
        selected, _ = base._screen_features(
            fit_state, state_candidates, local_config,
            target_column=target, min_finite_rows=60,
        )
        state_inputs = list(dict.fromkeys([
            *selected, *_trajectory_source_dependencies(selected),
        ]))
        if not selected:
            continue
        fit_state = _daily_episode_state_frame(
            fit_rows, state_inputs, target_column=target,
            event_column=target, market_reference=market_train,
        )
        score_state = _daily_episode_state_frame(
            score_rows, state_inputs, target_column=target,
            event_column=target, market_reference=market_train,
        )
        _, score_out, _, subtype_report, _ = _attach_train_only_adverse_subtype_features(
            fit_state, score_state, selected, target_column=target,
            seed=seed + fold_index, feature_prefix=prefix,
            include_component_posteriors=False,
        )
        for name in output_columns:
            if name not in score_out:
                score_out[name] = np.float32(np.nan)
        part = score_out.loc[:, ["day", *output_columns]].copy()
        part["fold_start"] = fold_start
        oof_parts.append(part)
        report_rows.append({
            "stage": "oof", "fold_start": fold_start,
            "daily_train_rows": len(fit_state), "daily_score_rows": len(score_state),
            "adverse_days": int(fit_state[target].sum()),
            "features": "|".join(selected), **subtype_report,
        })

    oof = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame(
        columns=["day", *output_columns]
    )
    final = pd.DataFrame(columns=["day", *output_columns])
    encoder: dict[str, Any] | None = None
    full_state = _daily_episode_state_frame(
        train_selected, state_candidates, target_column=target,
        event_column=target, market_reference=market_full,
    )
    if len(full_state) >= 60 and int(full_state[target].sum()) >= 6 and not valid_selected.empty:
        local_config = _daily_market_context_config(full_state, config)
        selected, _ = base._screen_features(
            full_state, state_candidates, local_config,
            target_column=target, min_finite_rows=60,
        )
        if selected:
            state_inputs = list(dict.fromkeys([
                *selected, *_trajectory_source_dependencies(selected),
            ]))
            full_state = _daily_episode_state_frame(
                train_selected, state_inputs, target_column=target,
                event_column=target, market_reference=market_full,
            )
            valid_state = _daily_episode_state_frame(
                valid_selected, state_inputs, target_column=target,
                event_column=target, market_reference=market_full,
            )
            _, final_out, _, subtype_report, encoder = _attach_train_only_adverse_subtype_features(
                full_state, valid_state, selected, target_column=target,
                seed=seed + 10_000, feature_prefix=prefix,
                include_component_posteriors=False,
            )
            for name in output_columns:
                if name not in final_out:
                    final_out[name] = np.float32(np.nan)
            final = final_out.loc[:, ["day", *output_columns]].copy()
            report_rows.append({
                "stage": "final", "fold_start": config.train_end,
                "daily_train_rows": len(full_state), "daily_score_rows": len(valid_state),
                "adverse_days": int(full_state[target].sum()),
                "features": "|".join(selected), **subtype_report,
            })
    report = pd.DataFrame(report_rows)
    return (
        _attach_daily_market_features(market_train, oof, output_columns),
        _attach_daily_market_features(market_valid, final, output_columns),
        encoder,
        report,
        output_columns,
    )


def _fit_side_market_episode_context(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    candidates: list[str],
    config: base.Config,
    *,
    market_train: pd.DataFrame,
    market_valid: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]], pd.DataFrame]:
    """Fit one daily adverse-period context detector per side.

    This is a hierarchical context layer for sparse local archetype models. It
    pools only same-side adverse days, never combines long and short outcomes,
    and preserves the local side x archetype target as the sole promotion
    target.
    """

    train_selected = train.loc[train["parent_rank_v9"].ge(config.top10_floor)]
    valid_selected = valid.loc[valid["parent_rank_v9"].ge(config.top10_floor)]
    all_oof: list[pd.DataFrame] = []
    all_final: list[pd.DataFrame] = []
    reports: list[pd.DataFrame] = []
    bundles: dict[str, dict[str, Any]] = {}
    for side_index, side in enumerate(("long", "short")):
        train_side = train_selected.loc[train_selected["side_name"].astype(str).eq(side)].sort_values("__ts__", kind="stable")
        valid_side = valid_selected.loc[valid_selected["side_name"].astype(str).eq(side)].sort_values("__ts__", kind="stable")
        if train_side.empty:
            continue
        oof, bundle, final, report = _fit_daily_episode_context(
            train_side, valid_side, candidates, config,
            market_train=market_train, market_valid=market_valid,
            target_column=base.SIDE_EVENT, risk_column=SIDE_MARKET_EPISODE_RISK,
            risk_pct_column=SIDE_MARKET_EPISODE_RISK_PCT,
            seed=seed + 1_000 * side_index, context_label=f"side:{side}",
        )
        if not oof.empty:
            oof["side_name"] = side
            all_oof.append(oof)
        if not final.empty:
            final["side_name"] = side
            all_final.append(final)
        if not report.empty:
            report["side_name"] = side
            reports.append(report)
        if bundle is not None:
            bundles[side] = bundle
    oof = pd.concat(all_oof, ignore_index=True) if all_oof else pd.DataFrame(
        columns=["day", "side_name", base.SIDE_EVENT, SIDE_MARKET_EPISODE_RISK, SIDE_MARKET_EPISODE_RISK_PCT]
    )
    final = pd.concat(all_final, ignore_index=True) if all_final else pd.DataFrame(
        columns=["day", "side_name", SIDE_MARKET_EPISODE_RISK, SIDE_MARKET_EPISODE_RISK_PCT]
    )
    report = pd.concat(reports, ignore_index=True) if reports else pd.DataFrame()
    return (
        _attach_daily_side_context(train, oof),
        _attach_daily_side_context(valid, final),
        oof,
        bundles,
        report,
    )


def _fit_group_arm(
    train_rows: pd.DataFrame,
    valid_rows: pd.DataFrame,
    candidates: list[str],
    arm_name: str,
    config: base.Config,
    seed: int,
    target_column: str = base.TARGET,
    period_control_mode: str = "timestamp",
    trajectory_train: pd.DataFrame | None = None,
    trajectory_valid: pd.DataFrame | None = None,
    state_granularity: str = "timestamp",
    market_train: pd.DataFrame | None = None,
    market_valid: pd.DataFrame | None = None,
    pooled_train: pd.DataFrame | None = None,
    pooled_reliability_shrinkage_k: float = 20.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    if state_granularity == "daily_open":
        if market_train is None or market_valid is None:
            raise ValueError("daily_open episode signatures require full-universe market panels")
        return _fit_group_daily_episode_arm(
            train_rows, valid_rows, candidates, arm_name, config, seed,
            target_column=target_column,
            period_control_mode=period_control_mode,
            market_train=market_train,
            market_valid=market_valid,
            pooled_train=pooled_train,
            pooled_reliability_shrinkage_k=pooled_reliability_shrinkage_k,
        )
    if state_granularity != "timestamp":
        raise ValueError(f"Unknown episode state granularity: {state_granularity}")
    # Kept solely to load older experiment bundles. New research must use a
    # single pre-open state per day. A repeated intraday candidate state turns
    # one difficult period into hundreds of pseudo-independent observations.
    raise ValueError(
        "timestamp episode states are legacy-only; use daily_open so difficult "
        "periods, not individual candidate rows, are the learning unit"
    )
    # Phase labels retain the same top-10 decision universe as their parent
    # period target.  State features are summarized from the local top-20
    # context, so a contemporaneous regime is represented by the candidate
    # population rather than a single future winning/losing row.
    local_floor = config.top10_floor if "top10_" in target_column else config.top20_floor
    local = train_rows.loc[train_rows["parent_rank_v9"].ge(local_floor)].sort_values("__ts__")
    context_floor = min(float(local_floor), PERIOD_CONTEXT_FLOOR)
    context = train_rows.loc[
        train_rows["parent_rank_v9"].ge(context_floor)
    ].sort_values("__ts__")
    available_trajectory_sources = [
        name for name in EPISODE_TRAJECTORY_SOURCE_FEATURES
        if name in train_rows.columns
    ]
    trajectory_oof = trajectory_train
    trajectory_final = (
        pd.concat([trajectory_train, trajectory_valid], ignore_index=True)
        .sort_values("__ts__", kind="stable")
        .drop_duplicates("__ts__", keep="last")
        if trajectory_train is not None and trajectory_valid is not None
        else trajectory_train
    )
    state_candidates = list(dict.fromkeys([
        *candidates,
        *available_trajectory_sources,
        *PERIOD_STATE_FEATURES,
        *EPISODE_TRAJECTORY_FEATURES,
    ]))
    oof_parts: list[pd.DataFrame] = []
    model_rows: list[dict[str, Any]] = []
    rules: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    for fold_index, fold_start in enumerate(base.FOLD_STARTS):
        fold_end = base.FOLD_STARTS[fold_index + 1] if fold_index + 1 < len(base.FOLD_STARTS) else pd.Timestamp(config.train_end, tz="UTC")
        fit_rows = local.loc[local["__ts__"].lt(fold_start - pd.Timedelta(days=2))]
        score_rows = local.loc[local["__ts__"].ge(fold_start) & local["__ts__"].lt(fold_end)]
        fit_context = context.loc[context["__ts__"].lt(fold_start - pd.Timedelta(days=2))]
        score_context = context.loc[
            context["__ts__"].ge(fold_start) & context["__ts__"].lt(fold_end)
        ]
        required_positive_rows = (
            config.min_positive_rows
            if target_column == base.TARGET
            else max(20, config.min_positive_rows // 2)
        )
        if len(fit_rows) < config.min_train_rows or int(fit_rows[target_column].sum()) < required_positive_rows or score_rows.empty:
            continue
        fit_state = _period_state_frame(
            fit_context,
            fit_rows,
            state_candidates,
            target_column=target_column,
            event_column=target_column,
            trajectory_reference=trajectory_oof,
        )
        selected, _ = base._screen_features(
            fit_state, state_candidates, config, target_column=target_column
        )
        if not selected:
            continue
        # Keep the raw dependencies even when a trajectory is the selected
        # predictor. They are needed to recreate its causal lookup at scoring.
        state_input_features = list(dict.fromkeys([
            *selected,
            *_trajectory_source_dependencies(selected),
        ]))
        fit_state = _period_state_frame(
            fit_context,
            fit_rows,
            state_input_features,
            target_column=target_column,
            event_column=target_column,
            trajectory_reference=trajectory_oof,
        )
        score_state = _period_state_frame(
            score_context,
            score_rows,
            state_input_features,
            target_column=target_column,
            event_column=target_column,
            history_state=fit_state,
            trajectory_reference=trajectory_oof,
        )
        try:
            bundle, _, state_score, reference, local_rules, local_controls = _fit_arm(
                arm_name, fit_state, score_state, selected, seed + fold_index,
                target_column=target_column,
                period_control_mode=period_control_mode,
            )
        except Exception as exc:
            model_rows.append({
                "model_arm": arm_name, "stage": "oof_failed", "fold_start": fold_start,
                "train_rows": len(fit_rows), "train_state_rows": len(fit_state),
                "score_rows": len(score_rows), "features": len(selected),
                "selected_features": "|".join(selected), "matched_control_rows": 0,
                "model_target": target_column,
                "failure": f"{type(exc).__name__}: {exc}",
            })
            continue
        score_map = pd.Series(state_score, index=score_state["__ts__"])
        row_score = score_rows["__ts__"].map(score_map).to_numpy(np.float32)
        phase_columns = [
            name for name in ("episode_phase", "episode_block", "episode_phase_offset_days")
            if name in score_rows.columns
        ]
        part_columns = list(dict.fromkeys([
            *base.KEYS, "day", "parent_rank_v9", "ev_after_1pct", "clean_exec",
            base.EVENT, base.TARGET, target_column, *phase_columns,
        ]))
        part = score_rows.loc[:, part_columns].copy()
        part["model_arm"] = arm_name
        part["model_target"] = target_column
        part[base.RISK_SCORE] = row_score
        part[base.RISK_PCT] = base._midrank(row_score, reference)
        part["fold_start"] = fold_start
        oof_parts.append(part)
        model_rows.append({
            "model_arm": arm_name, "stage": "oof", "fold_start": fold_start,
            "train_rows": len(fit_rows), "train_state_rows": len(fit_state),
            "score_rows": len(score_rows), "features": len(selected),
            "selected_features": "|".join(selected), "matched_control_rows": int(sum(row["control_rows"] for row in local_controls)),
            "model_target": target_column,
        })
        for row in local_rules:
            rules.append({"model_arm": arm_name, "stage": "oof", "fold_start": fold_start, **row})
        for row in local_controls:
            controls.append({"model_arm": arm_name, "stage": "oof", "fold_start": fold_start, **row})
    if not oof_parts:
        return pd.DataFrame(), pd.DataFrame(model_rows), None, rules, controls

    full_state = _period_state_frame(
        context,
        local,
        state_candidates,
        target_column=target_column,
        event_column=target_column,
        trajectory_reference=trajectory_final,
    )
    selected, _ = base._screen_features(
        full_state, state_candidates, config, target_column=target_column
    )
    local_valid = valid_rows.loc[valid_rows["parent_rank_v9"].ge(local_floor)].sort_values("__ts__")
    valid_context = valid_rows.loc[
        valid_rows["parent_rank_v9"].ge(context_floor)
    ].sort_values("__ts__")
    final = None
    if selected and not local_valid.empty:
        state_input_features = list(dict.fromkeys([
            *selected,
            *_trajectory_source_dependencies(selected),
        ]))
        full_state = _period_state_frame(
            context,
            local,
            state_input_features,
            target_column=target_column,
            event_column=target_column,
            trajectory_reference=trajectory_final,
        )
        valid_state = _period_state_frame(
            valid_context,
            local_valid,
            state_input_features,
            target_column=target_column,
            event_column=target_column,
            history_state=full_state,
            trajectory_reference=trajectory_final,
        )
        try:
            bundle, _, state_score, reference, local_rules, local_controls = _fit_arm(
                arm_name, full_state, valid_state, selected, seed + 10_000,
                target_column=target_column,
                period_control_mode=period_control_mode,
            )
        except Exception as exc:
            model_rows.append({
                "model_arm": arm_name, "stage": "final_failed", "fold_start": config.train_end,
                "train_rows": len(local), "train_state_rows": len(full_state),
                "score_rows": len(local_valid), "features": len(selected),
                "selected_features": "|".join(selected), "matched_control_rows": 0,
                "model_target": target_column,
                "failure": f"{type(exc).__name__}: {exc}",
            })
            return pd.concat(oof_parts, ignore_index=True), pd.DataFrame(model_rows), None, rules, controls
        score_map = pd.Series(state_score, index=valid_state["__ts__"])
        row_score = local_valid["__ts__"].map(score_map).to_numpy(np.float32)
        final = {"bundle": bundle, "index": local_valid.index.to_numpy(), "score": row_score, "reference": reference}
        model_rows.append({
            "model_arm": arm_name, "stage": "final", "fold_start": config.train_end,
            "train_rows": len(local), "train_state_rows": len(full_state),
            "score_rows": len(local_valid), "features": len(selected),
            "selected_features": "|".join(selected), "matched_control_rows": int(sum(row["control_rows"] for row in local_controls)),
            "model_target": target_column,
        })
        for row in local_rules:
            rules.append({"model_arm": arm_name, "stage": "final", "fold_start": config.train_end, **row})
        for row in local_controls:
            controls.append({"model_arm": arm_name, "stage": "final", "fold_start": config.train_end, **row})
    return pd.concat(oof_parts, ignore_index=True), pd.DataFrame(model_rows), final, rules, controls


def _mechanism_features(columns: list[str]) -> dict[str, list[str]]:
    return {
        mechanism: [name for name in columns if any(token in name.lower() for token in tokens)]
        for mechanism, tokens in MECHANISM_TOKENS.items()
    }


def _classify_mechanisms(frame: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    families = _mechanism_features(list(frame.columns))
    scores = np.zeros((len(frame), len(families)), dtype=np.float32)
    names = list(families)
    source_features: list[str] = []
    for family_index, name in enumerate(names):
        features = families[name]
        source_features.append("|".join(features))
        if not features:
            continue
        train = reference.reindex(columns=features).apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        values = frame.reindex(columns=features).apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        median = np.nanmedian(train, axis=0)
        scale = np.maximum(np.nanquantile(train, 0.75, axis=0) - np.nanquantile(train, 0.25, axis=0), 1e-4)
        z = np.abs((values - median) / scale)
        scores[:, family_index] = np.nanmax(np.nan_to_num(z, nan=0.0), axis=1)
    dominant = np.argmax(scores, axis=1)
    total = scores.sum(axis=1)
    result = pd.DataFrame(index=frame.index)
    result["mechanism_class"] = np.asarray(names, dtype=object)[dominant]
    result["mechanism_confidence"] = scores[np.arange(len(frame)), dominant] / np.maximum(total, 1e-6)
    result["mechanism_strength"] = scores[np.arange(len(frame)), dominant]
    result["mechanism_source_features"] = np.asarray(source_features, dtype=object)[dominant]
    result.loc[result["mechanism_strength"].lt(1.0), "mechanism_class"] = "unclassified_low_intensity"
    return result


def _calendar(frame: pd.DataFrame, reference: pd.DataFrame, selector: str) -> pd.DataFrame:
    # Calendar reporting is only defined for admitted rows.  Classifying every
    # source row repeatedly creates several wide 900k-row matrices without
    # changing a single calendar cell.  Use the candidate-stream reference for
    # robust normalization and classify just the selected rows.
    selected = frame.loc[frame[selector].ge(0.90)].copy()
    if selected.empty:
        return pd.DataFrame()
    if "day" not in selected.columns:
        selected["day"] = pd.to_datetime(selected["__ts__"], utc=True).dt.floor("D")
    reference_rows = reference.loc[
        reference["parent_rank_v9"].ge(0.80)
    ] if "parent_rank_v9" in reference.columns else reference
    mechanism = _classify_mechanisms(selected, reference_rows)
    selected = pd.concat(
        [selected.reset_index(drop=True), mechanism.reset_index(drop=True)],
        axis=1,
        copy=False,
    )
    return (
        selected.groupby(
            ["day", "side_name", "archetype_policy_key", "mechanism_class"],
            observed=True,
            dropna=False,
        )
        .agg(
            selected_rows=("ev_after_1pct", "size"),
            mean_ev_after_1pct=("ev_after_1pct", "mean"),
            clean_exec_precision=("clean_exec", "mean"),
            adverse_calendar_cell=(base.EVENT, "max"),
            mechanism_confidence=("mechanism_confidence", "mean"),
            mechanism_strength=("mechanism_strength", "mean"),
            mechanism_source_features=("mechanism_source_features", "first"),
        )
        .reset_index()
    )


def _event_intervention_report(
    frame: pd.DataFrame,
    parent_rank: np.ndarray,
    adjusted_rank: np.ndarray,
    flagged: np.ndarray,
    *,
    top10_floor: float,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Measure whether an overlay actually changes difficult OOS periods.

    Aggregate EV can improve by reallocating normal rows.  The overlay's stated
    job is narrower: intervene inside a local adverse period.  This report is
    therefore an evaluation diagnostic, never a training feature or selector.
    """

    payload = frame.loc[:, ["day", "side_name", "archetype_policy_key", base.EVENT,
                             "ev_after_1pct", "clean_exec"]].copy()
    payload["parent_selected"] = (
        np.asarray(parent_rank, dtype=np.float32) >= top10_floor
    )
    payload["overlay_selected"] = (
        np.asarray(adjusted_rank, dtype=np.float32) >= top10_floor
    )
    payload["overlay_flagged"] = np.asarray(flagged, dtype=bool)
    payload["selection_changed"] = (
        payload["parent_selected"] != payload["overlay_selected"]
    )
    selected_parent = payload.loc[payload["parent_selected"]]
    selected_overlay = payload.loc[payload["overlay_selected"]]
    keys = ["day", "side_name", "archetype_policy_key"]
    parent = selected_parent.groupby(keys, observed=True).agg(
        parent_rows=("ev_after_1pct", "size"),
        parent_mean_ev=("ev_after_1pct", "mean"),
        parent_clean_precision=("clean_exec", "mean"),
    )
    overlay = selected_overlay.groupby(keys, observed=True).agg(
        overlay_rows=("ev_after_1pct", "size"),
        overlay_mean_ev=("ev_after_1pct", "mean"),
        overlay_clean_precision=("clean_exec", "mean"),
    )
    state = payload.groupby(keys, observed=True).agg(
        adverse_calendar_cell=(base.EVENT, "max"),
        flagged_parent_rows=("overlay_flagged", "sum"),
        selection_changed_rows=("selection_changed", "sum"),
    )
    report = (
        state.join(parent, how="left")
        .join(overlay, how="left")
        .reset_index()
    )
    for name in (
        "parent_rows", "overlay_rows", "parent_mean_ev", "overlay_mean_ev",
        "parent_clean_precision", "overlay_clean_precision",
    ):
        report[name] = report[name].fillna(0.0)
    report["ev_delta"] = report["overlay_mean_ev"] - report["parent_mean_ev"]
    report["clean_precision_delta"] = (
        report["overlay_clean_precision"] - report["parent_clean_precision"]
    )
    report["intervened"] = report["selection_changed_rows"].gt(0)
    events = report.loc[report["adverse_calendar_cell"].gt(0)].copy()
    summary: dict[str, float | int] = {
        "event_cells": int(len(events)),
        "event_cells_flagged": int(events["flagged_parent_rows"].gt(0).sum()),
        "event_cells_intervened": int(events["intervened"].sum()),
        "event_cells_improved": int(
            (events["intervened"] & events["ev_delta"].gt(0)).sum()
        ),
        "event_cells_worsened": int(
            (events["intervened"] & events["ev_delta"].lt(0)).sum()
        ),
        "event_intervention_rate": float(events["intervened"].mean())
        if len(events) else np.nan,
        "event_flag_rate": float(events["flagged_parent_rows"].gt(0).mean())
        if len(events) else np.nan,
    }
    return report, summary


def _validate_oos_candidates(
    candidates: pd.DataFrame,
    event_interventions: pd.DataFrame,
    *,
    minimum_event_cells: int,
    minimum_intervention_recall: float,
    minimum_improved_cells: int,
    minimum_activity_ratio: float = 0.90,
) -> pd.DataFrame:
    """Apply the frozen, post-selection validation contract per local overlay.

    The OOF search is allowed to nominate a candidate.  It is not allowed to
    silently turn that candidate into an active policy: the untouched period
    must contain enough adverse cells and the frozen action must improve them.
    This is deliberately evaluated at the daily episode-cell level, never on
    isolated trade rows.
    """

    fields = [
        "side_name", "archetype_policy_key", "model_arm", "model_target",
        "oos_event_cells", "oos_event_cells_intervened", "oos_event_cells_improved",
        "oos_event_intervention_recall", "oos_mean_event_ev_delta",
        "oos_mean_event_clean_precision_delta", "oos_activity_ratio",
        "oos_validation_status", "oos_validated",
    ]
    if candidates.empty:
        return pd.DataFrame(columns=fields)
    rows: list[dict[str, Any]] = []
    for candidate in candidates.to_dict("records"):
        mask = (
            event_interventions["side_name"].astype(str).eq(str(candidate["side_name"]))
            & event_interventions["archetype_policy_key"].astype(str).eq(
                str(candidate["archetype_policy_key"])
            )
        )
        group = event_interventions.loc[mask]
        events = group.loc[group["adverse_calendar_cell"].gt(0)]
        parent_rows = float(group["parent_rows"].sum())
        overlay_rows = float(group["overlay_rows"].sum())
        event_cells = int(len(events))
        intervened = int(events["intervened"].sum()) if event_cells else 0
        improved = int(
            (events["intervened"] & events["ev_delta"].gt(0)).sum()
        ) if event_cells else 0
        recall = float(intervened / event_cells) if event_cells else 0.0
        event_ev_delta = float(events["ev_delta"].mean()) if event_cells else np.nan
        event_clean_delta = (
            float(events["clean_precision_delta"].mean()) if event_cells else np.nan
        )
        activity_ratio = overlay_rows / parent_rows if parent_rows > 0 else 0.0
        if event_cells < minimum_event_cells:
            status = "not_evaluable_insufficient_untouched_event_cells"
        elif recall < minimum_intervention_recall:
            status = "fail_insufficient_untouched_intervention"
        elif improved < minimum_improved_cells:
            status = "fail_insufficient_untouched_improved_cells"
        elif not np.isfinite(event_ev_delta) or event_ev_delta <= 0.0:
            status = "fail_nonpositive_untouched_event_ev_delta"
        elif not np.isfinite(event_clean_delta) or event_clean_delta < 0.0:
            status = "fail_negative_untouched_event_clean_precision_delta"
        elif activity_ratio < minimum_activity_ratio:
            status = "fail_excessive_untouched_activity_loss"
        else:
            status = "pass"
        rows.append({
            "side_name": candidate["side_name"],
            "archetype_policy_key": candidate["archetype_policy_key"],
            "model_arm": candidate["model_arm"],
            "model_target": candidate["model_target"],
            "oos_event_cells": event_cells,
            "oos_event_cells_intervened": intervened,
            "oos_event_cells_improved": improved,
            "oos_event_intervention_recall": recall,
            "oos_mean_event_ev_delta": event_ev_delta,
            "oos_mean_event_clean_precision_delta": event_clean_delta,
            "oos_activity_ratio": activity_ratio,
            "oos_validation_status": status,
            "oos_validated": bool(status == "pass"),
        })
    return pd.DataFrame(rows, columns=fields)


def _episode_intervention_diagnostics(
    frame: pd.DataFrame,
    adjusted_rank: np.ndarray,
    *,
    top10_floor: float,
) -> dict[str, float | int]:
    """Measure whether an OOF policy actually changes adverse episode cells.

    The input is one side x archetype OOF stream. Promotion requires action in
    the daily episode cells themselves; aggregate normal-period reallocation
    cannot satisfy this diagnostic.
    """

    parent = pd.to_numeric(frame["parent_rank_v9"], errors="coerce").to_numpy(
        np.float32
    )
    adjusted = np.asarray(adjusted_rank, dtype=np.float32)
    payload = pd.DataFrame(
        {
            "day": pd.to_datetime(frame["day"], utc=True),
            "event": frame[base.EVENT].to_numpy(bool),
            "parent_selected": parent >= top10_floor,
            "overlay_selected": adjusted >= top10_floor,
            "ev": pd.to_numeric(frame["ev_after_1pct"], errors="coerce").to_numpy(
                np.float32
            ),
        }
    )
    payload["changed"] = payload["parent_selected"] != payload["overlay_selected"]
    cells: list[dict[str, bool]] = []
    for _, local in payload.loc[payload["event"]].groupby("day", observed=True):
        parent_ev = local.loc[local["parent_selected"], "ev"].mean()
        overlay_ev = local.loc[local["overlay_selected"], "ev"].mean()
        intervened = bool(local["changed"].any())
        improved = bool(
            intervened
            and np.isfinite(parent_ev)
            and np.isfinite(overlay_ev)
            and overlay_ev > parent_ev
        )
        cells.append({"intervened": intervened, "improved": improved})
    if not cells:
        return {
            "oof_event_cells": 0,
            "oof_event_cells_intervened": 0,
            "oof_event_cells_improved": 0,
            "oof_event_intervention_recall": np.nan,
            "oof_event_improvement_recall": np.nan,
        }
    result = pd.DataFrame(cells)
    total = len(result)
    intervened = int(result["intervened"].sum())
    improved = int(result["improved"].sum())
    return {
        "oof_event_cells": int(total),
        "oof_event_cells_intervened": intervened,
        "oof_event_cells_improved": improved,
        "oof_event_intervention_recall": float(intervened / total),
        "oof_event_improvement_recall": float(improved / total),
    }


def _search_episode_overlay(
    frame: pd.DataFrame,
    config: base.Config,
    *,
    minimum_intervention_recall: float,
    minimum_improved_cells: int,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Run the standard OOF search with explicit difficult-period gates."""

    search, _ = base._search_local_overlay(
        frame, config, risk_column=base.RISK_PCT
    )
    parent = frame["parent_rank_v9"].to_numpy(np.float32)
    risk = frame[base.RISK_PCT].to_numpy(np.float32)
    diagnostics: list[dict[str, float | int]] = []
    for row in search.to_dict("records"):
        adjusted, _ = base._apply_rank(
            parent,
            risk,
            float(row["threshold"]),
            float(row["alpha"]),
            str(row["mode"]) == "hard_block",
            config.top10_floor,
        )
        diagnostics.append(
            _episode_intervention_diagnostics(
                frame, adjusted, top10_floor=config.top10_floor
            )
        )
    search = pd.concat(
        [search, pd.DataFrame(diagnostics, index=search.index)], axis=1
    )
    original = search["promotable"].astype(bool)
    intervention_ok = (
        pd.to_numeric(search["oof_event_intervention_recall"], errors="coerce")
        .fillna(0.0)
        .ge(float(minimum_intervention_recall))
    )
    improvement_ok = (
        pd.to_numeric(search["oof_event_cells_improved"], errors="coerce")
        .fillna(0)
        .ge(int(minimum_improved_cells))
    )
    search["promotable_pre_episode_intervention"] = original
    search["promotable"] = original & intervention_ok & improvement_ok
    search = search.sort_values(
        ["promotable", "objective", "oof_event_cells_improved", "activity_ratio"],
        ascending=[False, False, False, False],
        kind="stable",
    )
    accepted = search.loc[search["promotable"]]
    return search, (accepted.iloc[0].to_dict() if not accepted.empty else None)


def _july_matched_controls(
    july_path: Path,
    state_path: Path,
    train: pd.DataFrame,
    candidates: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not july_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    july = pd.read_parquet(july_path)
    july["__ts__"] = pd.to_datetime(july["__ts__"], utc=True)
    state_columns = [*base.KEYS, *[name for name in candidates if name not in july.columns]]
    available = set(pq.read_schema(state_path).names)
    state_columns = [name for name in state_columns if name in available]
    state = pd.read_parquet(state_path, columns=state_columns)
    state["__ts__"] = pd.to_datetime(state["__ts__"], utc=True)
    july = july.merge(state.drop_duplicates(base.KEYS), on=base.KEYS, how="left", validate="one_to_one")
    july["day"] = july["__ts__"].dt.floor("D")
    target_days = pd.to_datetime(["2026-07-06", "2026-07-09", "2026-07-10"], utc=True)
    benign = train.loc[train[base.EVENT].eq(0)].copy()
    rows: list[dict[str, Any]] = []
    usable = [name for name in candidates if name in july.columns and name in benign.columns][:24]
    for day in target_days:
        day_rows = july.loc[july["day"].eq(day)]
        for (side, archetype), episode in day_rows.groupby(["side_name", "archetype_policy_key"], observed=True):
            pool = benign.loc[benign["side_name"].astype(str).eq(str(side)) & benign["archetype_policy_key"].astype(str).eq(str(archetype))]
            if len(pool) < 20 or not usable:
                continue
            pool_daily = pool.groupby("day", observed=True)[usable].median()
            proto = episode[usable].apply(pd.to_numeric, errors="coerce").median().to_numpy(np.float32)
            train_values = pool_daily.to_numpy(np.float32)
            robust = RobustMatrixTransform().fit(train_values)
            z_pool = robust.transform(train_values)
            z_proto = robust.transform(proto.reshape(1, -1))[0]
            distance = np.mean((z_pool - z_proto) ** 2, axis=1)
            for order in np.argsort(distance, kind="stable")[:10]:
                rows.append({
                    "prototype_day": day, "side_name": side, "archetype_policy_key": archetype,
                    "matched_benign_day": pool_daily.index[order], "distance": float(distance[order]),
                    "diagnostic_only_july_informed": True, "used_for_model_fit": False,
                })
    return pd.DataFrame(rows), july


def _write_dossier(
    output: Path,
    accepted: pd.DataFrame,
    rules: pd.DataFrame,
    model_report: pd.DataFrame,
    baseline_overlay: Path | None = None,
) -> None:
    focus = {
        "long_volcompression_wideslow_candidate": "Long vol-compression",
        "short_default_clean_path": "Short default",
    }
    lines = ["# Residual Rules and Composites", "", "All metrics are chronological train-OOF selection metrics unless explicitly marked final-fit.", ""]
    baseline_accept = pd.DataFrame()
    baseline_models = pd.DataFrame()
    baseline_screen = pd.DataFrame()
    if baseline_overlay is not None and baseline_overlay.exists():
        baseline_accept = pd.read_csv(baseline_overlay / "accepted_local_overlays.csv")
        baseline_models = pd.read_csv(baseline_overlay / "model_report.csv")
        baseline_screen = pd.read_csv(baseline_overlay / "feature_screening.csv")
    for archetype, title in focus.items():
        lines += [f"## {title}", ""]
        prior = baseline_accept.loc[
            baseline_accept.get("archetype_policy_key", pd.Series(dtype=str)).eq(archetype)
        ]
        if not prior.empty:
            row = prior.iloc[0]
            lines += [
                "### Established event-balanced LGBM evidence",
                "",
                f"- Risk lift: `{row['risk_lift']:.3f}x`; FPR: `{row['risk_fpr']:.2%}`; recognized blocks: `{int(row['recognized_event_blocks'])}/{int(row['event_blocks'])}`",
                f"- OOF EV delta: `{row['overall_ev_delta']:+.4%}`; event EV delta: `{row['event_ev_delta']:+.4%}`; activity: `{row['activity_ratio']:.2%}`",
            ]
            prior_model = baseline_models.loc[
                baseline_models.get("archetype_policy_key", pd.Series(dtype=str)).eq(archetype)
                & baseline_models.get("stage", pd.Series(dtype=str)).eq("final")
            ]
            if not prior_model.empty:
                lines.append(f"- Selected features: `{prior_model.iloc[0]['selected_features']}`")
            prior_screen = baseline_screen.loc[
                baseline_screen.get("archetype_policy_key", pd.Series(dtype=str)).eq(archetype)
                & baseline_screen.get("fold_start", pd.Series(dtype=str)).astype(str).eq("final")
                & baseline_screen.get("selected", pd.Series(dtype=bool)).astype(bool)
            ].sort_values("screen_score", ascending=False).head(12)
            if not prior_screen.empty:
                lines += ["", "Highest-ranked established composites/features:"]
                for feature in prior_screen.to_dict("records"):
                    lines.append(
                        f"- `{feature['feature']}`: MI `{feature['binned_mi']:.5f}`, "
                        f"tail lift `{feature['tail_lift']:.3f}x`, direction `{int(feature['tail_direction']):+d}`"
                    )
            lines.append("")
        local_accept = (
            accepted.loc[accepted["archetype_policy_key"].eq(archetype)]
            if "archetype_policy_key" in accepted
            else pd.DataFrame()
        )
        if local_accept.empty:
            lines += ["No rule-model overlay passed the frozen promotion contract.", ""]
        else:
            for row in local_accept.to_dict("records"):
                lines += [
                    f"- Selected arm: `{row['model_arm']}`",
                    f"- Risk lift: `{row.get('risk_lift', float('nan')):.3f}x`; FPR: `{row.get('risk_fpr', float('nan')):.2%}`; recognized blocks: `{int(row.get('recognized_event_blocks', 0))}`",
                    f"- Overlay: `{row.get('mode')}` at percentile `{row.get('threshold')}` with alpha `{row.get('alpha')}`",
                ]
        local_models = model_report.loc[model_report["archetype_policy_key"].eq(archetype) & model_report["stage"].eq("final")]
        for row in local_models.to_dict("records"):
            lines += [f"### {row['model_arm']}", "", f"Selected observable features: `{row['selected_features']}`", ""]
            local_rules = (
                rules.loc[rules["archetype_policy_key"].eq(archetype) & rules["model_arm"].eq(row["model_arm"]) & rules["stage"].eq("final")].head(12)
                if {"archetype_policy_key", "model_arm", "stage"}.issubset(rules.columns)
                else pd.DataFrame()
            )
            for rule in local_rules.to_dict("records"):
                lines.append(f"- `{rule.get('rule', rule.get('rule_list', ''))}` (weight/risk `{rule.get('weight', float('nan')):.4f}`)")
            lines.append("")
    (output / "rules_composites_long_volcompression_short_default.md").write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = base.Config(
        train_start=args.train_start, train_end=args.train_end, eval_end=args.eval_end,
        max_features=args.max_features, seed=args.seed,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    train, valid, coverage = base._load_frames(args, config)
    print(
        f"loaded train_rows={len(train):,} valid_rows={len(valid):,}",
        flush=True,
    )
    train, valid, train_calendar, valid_calendar, expected = _prepare(train, valid, args, config)
    candidates = base._candidate_features(train.columns)
    if args.use_unsupervised_composites:
        del valid
        gc.collect()
        print("building bounded unsupervised relevance frame", flush=True)
        relevance_frame, primitive, relevance_columns = _build_unsupervised_relevance_frame(
            train, candidates, max_rows=int(args.max_unsupervised_rows)
        )
        print(
            f"running unsupervised relevance rows={len(relevance_frame):,} features={len(primitive):,}",
            flush=True,
        )
        # The relevance engine materializes dense nonlinear candidate tables.
        # Releasing the joined frames first prevents its short-lived work set
        # from competing with the later full-train local model fits.
        del train
        gc.collect()
        unsupervised_feature_map, unsupervised_definitions = (
            _discover_unsupervised_episode_composites(
                relevance_frame,
                primitive,
                relevance_columns,
                output=args.output,
                max_rows=int(args.max_unsupervised_rows),
            )
        )
        print("unsupervised relevance completed", flush=True)
        del relevance_frame
        gc.collect()
        train, valid, coverage = base._load_frames(args, config)
        print(
            f"reloaded model frames train_rows={len(train):,} valid_rows={len(valid):,}",
            flush=True,
        )
        train, valid, train_calendar, valid_calendar, expected = _prepare(
            train, valid, args, config
        )
        candidates = base._candidate_features(train.columns)
    else:
        unsupervised_feature_map, unsupervised_definitions = {}, {}
    train_calendar.to_csv(args.output / "train_residual_calendar.csv", index=False)
    valid_calendar.to_csv(args.output / "eval_residual_calendar.csv", index=False)
    expected.to_csv(args.output / "train_expected_clean_baseline.csv", index=False)
    print(
        f"unsupervised_composites groups={len(unsupervised_feature_map):,} "
        f"feature_refs={sum(len(values) for values in unsupervised_feature_map.values()):,}",
        flush=True,
    )
    composite_source_features = list(dict.fromkeys(
        str(name)
        for definitions in unsupervised_definitions.values()
        for definition in definitions
        for name in (definition.get("feature"), definition.get("feature_b"))
        if name
    ))
    # The episode overlay is a market-transition model. Release unrelated
    # joined-ledger columns before the daily aggregation so we do not retain a
    # wide row-level matrix while creating one market state per timestamp.
    market_state_candidates = _market_episode_candidates(candidates)
    episode_observable_features = list(dict.fromkeys([
        *market_state_candidates,
        *composite_source_features,
    ]))
    train = _retain_episode_columns(train, episode_observable_features)
    valid = _retain_episode_columns(valid, episode_observable_features)
    gc.collect()
    market_train = _observable_market_state_panel(train, market_state_candidates)
    market_valid = _observable_market_state_panel(valid, market_state_candidates)
    market_adverse_subtype_encoder: dict[str, Any] | None = None
    market_adverse_subtype_report = pd.DataFrame()
    market_adverse_subtype_features: list[str] = []
    pooled_daily_reliability = pd.DataFrame()
    pooled_reliability_train: pd.DataFrame | None = None
    if bool(getattr(args, "pooled_daily_context", False)):
        # Build broad daily context from a narrow all-archetype decision stream
        # and a timestamp-level market panel.  The local candidate data remains
        # filtered, which bounds memory while retaining cross-archetype support.
        pooled_train, pooled_valid = _load_pooled_daily_context_decisions(args, config)
        pooled_market_train, pooled_market_valid, pooled_candidates = _load_pooled_market_clock(
            args.negative_residual_features, config, market_state_candidates
        )
        (
            pooled_market_train,
            pooled_market_valid,
            global_market_oof,
            global_market_bundle,
            global_market_report,
        ) = _fit_global_market_episode_context(
            pooled_train,
            pooled_valid,
            pooled_candidates,
            config,
            market_train=pooled_market_train,
            market_valid=pooled_market_valid,
            seed=args.seed + 70_000,
        )
        (
            pooled_market_train,
            pooled_market_valid,
            market_adverse_subtype_encoder,
            market_adverse_subtype_report,
            market_adverse_subtype_features,
        ) = _fit_global_market_adverse_subtypes(
            pooled_train,
            pooled_valid,
            pooled_candidates,
            config,
            market_train=pooled_market_train,
            market_valid=pooled_market_valid,
            seed=args.seed + 75_000,
        )
        market_train = _attach_pooled_market_context(market_train, pooled_market_train)
        market_valid = _attach_pooled_market_context(market_valid, pooled_market_valid)
        market_train = _attach_daily_market_features(
            market_train, pooled_market_train, market_adverse_subtype_features
        )
        market_valid = _attach_daily_market_features(
            market_valid, pooled_market_valid, market_adverse_subtype_features
        )
        market_state_candidates = list(dict.fromkeys([
            *market_state_candidates, *market_adverse_subtype_features,
        ]))
        # Compact causal handoff for future partial pooling.  It deliberately
        # contains daily labels only on train rows; the context values are
        # frozen observable market transforms keyed by the same UTC day.
        pooled_daily_reliability = (
            pooled_train.loc[pooled_train["parent_rank_v9"].ge(config.top10_floor)]
            .groupby(["day", "side_name", "archetype_policy_key"], observed=True, sort=True)
            .agg(
                market_adverse_period=("market_adverse_period", "max"),
                side_adverse_period=(base.SIDE_EVENT, "max"),
                local_adverse_period=(base.EVENT, "max"),
                selected_rows=("parent_rank_v9", "size"),
            )
            .reset_index()
        )
        subtype_daily = pooled_market_train.loc[:, ["__ts__", *market_adverse_subtype_features]].copy()
        subtype_daily["day"] = pd.to_datetime(subtype_daily["__ts__"], utc=True).dt.floor("D")
        subtype_daily = subtype_daily.groupby("day", observed=True, sort=True)[market_adverse_subtype_features].first().reset_index()
        pooled_daily_reliability = pooled_daily_reliability.merge(
            subtype_daily, on="day", how="left", validate="many_to_one"
        )
        pooled_target_columns = [
            name for name in (
                TOP10_PERIOD_EVENT_TARGET,
                TOP10_MARKET_PERIOD_EVENT_TARGET,
                f"episode_onset_{TOP10_PERIOD_EVENT_TARGET}",
                f"episode_persistent_{TOP10_PERIOD_EVENT_TARGET}",
                f"episode_onset_{TOP10_MARKET_PERIOD_EVENT_TARGET}",
                f"episode_persistent_{TOP10_MARKET_PERIOD_EVENT_TARGET}",
            )
            if name in pooled_train.columns
        ]
        pooled_reliability_train = pooled_train.loc[:, list(dict.fromkeys([
            "day", "side_name", "archetype_policy_key", "parent_rank_v9", *pooled_target_columns,
        ]))].copy()
        (
            pooled_train,
            pooled_valid,
            side_market_oof,
            side_market_bundles,
            side_market_report,
        ) = _fit_side_market_episode_context(
            pooled_train,
            pooled_valid,
            pooled_candidates,
            config,
            market_train=pooled_market_train,
            market_valid=pooled_market_valid,
            seed=args.seed + 80_000,
        )
        side_scores = pd.concat(
            [
                pooled_train.loc[:, ["day", "side_name", SIDE_MARKET_EPISODE_RISK, SIDE_MARKET_EPISODE_RISK_PCT]],
                pooled_valid.loc[:, ["day", "side_name", SIDE_MARKET_EPISODE_RISK, SIDE_MARKET_EPISODE_RISK_PCT]],
            ],
            ignore_index=True,
            copy=False,
        ).drop_duplicates(["day", "side_name"], keep="last")
        train = _attach_daily_side_context(train, side_scores)
        valid = _attach_daily_side_context(valid, side_scores)
        coverage["pooled_daily_context"] = {
            "enabled": True,
            "decision_train_rows": int(len(pooled_train)),
            "decision_valid_rows": int(len(pooled_valid)),
            "market_train_rows": int(len(pooled_market_train)),
            "market_valid_rows": int(len(pooled_market_valid)),
            "market_features": list(pooled_candidates),
            "market_adverse_subtype_features": list(market_adverse_subtype_features),
        }
        del pooled_valid, pooled_market_train, pooled_market_valid
        gc.collect()
    else:
        (
            market_train,
            market_valid,
            global_market_oof,
            global_market_bundle,
            global_market_report,
        ) = _fit_global_market_episode_context(
            train,
            valid,
            market_state_candidates,
            config,
            market_train=market_train,
            market_valid=market_valid,
            seed=args.seed + 70_000,
        )
        (
            train,
            valid,
            side_market_oof,
            side_market_bundles,
            side_market_report,
        ) = _fit_side_market_episode_context(
            train,
            valid,
            market_state_candidates,
            config,
            market_train=market_train,
            market_valid=market_valid,
            seed=args.seed + 80_000,
        )
    global_market_context_features = [
        name
        for name in (GLOBAL_MARKET_EPISODE_RISK, GLOBAL_MARKET_EPISODE_RISK_PCT)
        if name in market_train.columns
        and pd.to_numeric(market_train[name], errors="coerce").notna().any()
    ]
    market_adverse_subtype_features = [
        name
        for name in market_adverse_subtype_features
        if name in market_train.columns
        and pd.to_numeric(market_train[name], errors="coerce").notna().any()
    ]
    print(
        "global market episode context "
        f"oof_days={len(global_market_oof):,} "
        f"features={','.join(global_market_context_features) or 'unavailable'}",
        flush=True,
    )
    print(
        "market adverse subtype context "
        f"features={','.join(market_adverse_subtype_features) or 'unavailable'}",
        flush=True,
    )
    side_market_context_features = [
        name
        for name in (SIDE_MARKET_EPISODE_RISK, SIDE_MARKET_EPISODE_RISK_PCT)
        if name in train.columns
        and pd.to_numeric(train[name], errors="coerce").notna().any()
    ]
    print(
        "side market episode context "
        f"oof_days={len(side_market_oof):,} "
        f"bundles={','.join(sorted(side_market_bundles)) or 'unavailable'} "
        f"features={','.join(side_market_context_features) or 'unavailable'}",
        flush=True,
    )
    target_groups = _groups(
        args.groups,
        train,
        min_rows=args.min_group_rows,
        max_groups=args.max_groups,
    )

    all_oof: list[pd.DataFrame] = []
    all_model_rows: list[pd.DataFrame] = []
    all_rules: list[dict[str, Any]] = []
    all_controls: list[dict[str, Any]] = []
    final_states: dict[tuple[str, str, str], dict[str, Any]] = {}
    accepted_rows: list[dict[str, Any]] = []
    search_rows: list[pd.DataFrame] = []
    selected_arms = [
        name.strip() for name in str(args.arms).split(",") if name.strip()
    ]
    unknown_arms = sorted(set(selected_arms) - set(ARMS))
    if unknown_arms:
        raise ValueError(f"Unknown model arms: {unknown_arms}; expected one of {list(ARMS)}")
    if not selected_arms:
        raise ValueError("At least one model arm is required")
    print(
        f"evaluating groups={len(target_groups):,} arms={','.join(selected_arms)}",
        flush=True,
    )
    for group_index, (side, archetype) in enumerate(target_groups):
        train_rows = train.loc[train["side_name"].astype(str).eq(side) & train["archetype_policy_key"].astype(str).eq(archetype)]
        valid_rows = valid.loc[valid["side_name"].astype(str).eq(side) & valid["archetype_policy_key"].astype(str).eq(archetype)]
        local_definitions = unsupervised_definitions.get(f"{side}|{archetype}", [])
        train_rows, local_composites = _materialize_local_composites(
            train_rows, local_definitions
        )
        valid_rows, _ = _materialize_local_composites(valid_rows, local_definitions)
        local_candidates = list(dict.fromkeys([
            *local_composites,
            *side_market_context_features,
            *global_market_context_features,
            *market_state_candidates,
        ]))
        print(
            f"group={group_index + 1}/{len(target_groups)} side={side} archetype={archetype} "
            f"train_rows={len(train_rows):,} local_composites={len(local_composites):,}",
            flush=True,
        )
        for arm_index, arm in enumerate(selected_arms):
            if args.overlay_target_mode == "top10_period_event":
                arm_targets = [TOP10_PERIOD_EVENT_TARGET]
            elif args.overlay_target_mode == "top10_market_period_event":
                arm_targets = [TOP10_MARKET_PERIOD_EVENT_TARGET]
            else:
                raise ValueError(
                    "Only daily period targets are supported: "
                    "top10_period_event or top10_market_period_event"
                )
            # These two arms can model evolving state phases without making the
            # rule-list search combinatorially expensive.  Each phase still has
            # to pass the same original adverse-event OOF overlay contract.
            if arm in {
                "episode_lgbm",
                "episode_lgbm_contrastive",
                "episode_lgbm_adverse_subtypes",
                "episode_lgbm_subtype_moe",
                "episode_lgbm_pooled_reliability",
                "episode_mlp",
            }:
                if args.overlay_target_mode == "top10_period_event":
                    arm_targets += [
                        f"episode_onset_{TOP10_PERIOD_EVENT_TARGET}",
                        f"episode_persistent_{TOP10_PERIOD_EVENT_TARGET}",
                    ]
                elif args.overlay_target_mode == "top10_market_period_event":
                    arm_targets += [
                        f"episode_onset_{TOP10_MARKET_PERIOD_EVENT_TARGET}",
                        f"episode_persistent_{TOP10_MARKET_PERIOD_EVENT_TARGET}",
                    ]
            for target_index, target_column in enumerate(arm_targets):
                arm_label = arm if target_column == base.TARGET else f"{arm}__{target_column.removesuffix('_target')}"
                print(
                    f"  arm={arm_label} target={target_column}",
                    flush=True,
                )
                oof, model_report, final, rules, controls = _fit_group_arm(
                    train_rows, valid_rows, local_candidates, arm, config,
                    args.seed + 100_000 * group_index + 1_000 * arm_index + 97 * target_index,
                    target_column=target_column,
                    period_control_mode=args.period_control_mode,
                    state_granularity=args.episode_state_granularity,
                    market_train=market_train,
                    market_valid=market_valid,
                    pooled_train=pooled_reliability_train,
                    pooled_reliability_shrinkage_k=args.pooled_reliability_shrinkage_k,
                )
                if oof.empty:
                    if not model_report.empty:
                        model_report["model_arm"] = arm_label
                        model_report.insert(0, "archetype_policy_key", archetype)
                        model_report.insert(0, "side_name", side)
                        all_model_rows.append(model_report)
                    continue
                oof["side_name"] = side
                oof["archetype_policy_key"] = archetype
                oof["model_arm"] = arm_label
                all_oof.append(oof)
                model_report["model_arm"] = arm_label
                model_report.insert(0, "archetype_policy_key", archetype)
                model_report.insert(0, "side_name", side)
                all_model_rows.append(model_report)
                for row in rules:
                    all_rules.append({"side_name": side, "archetype_policy_key": archetype, **row, "model_arm": arm_label})
                for row in controls:
                    all_controls.append({"side_name": side, "archetype_policy_key": archetype, **row, "model_arm": arm_label})
                search, best = _search_episode_overlay(
                    oof,
                    config,
                    minimum_intervention_recall=args.minimum_oof_event_intervention_recall,
                    minimum_improved_cells=args.minimum_oof_improved_event_cells,
                )
                search.insert(0, "model_target", target_column)
                search.insert(0, "model_arm", arm_label)
                search.insert(0, "archetype_policy_key", archetype)
                search.insert(0, "side_name", side)
                search_rows.append(search)
                if best is not None:
                    accepted_rows.append({"side_name": side, "archetype_policy_key": archetype, "model_arm": arm_label, "model_target": target_column, **best})
                if final is not None:
                    final_states[(side, archetype, arm_label)] = final

    if not all_oof:
        raise RuntimeError("No interpretable model arm produced chronological OOF predictions")
    oof_frame = pd.concat(all_oof, ignore_index=True)
    model_frame = pd.concat(all_model_rows, ignore_index=True)
    rules_frame = pd.DataFrame(all_rules)
    controls_frame = pd.DataFrame(all_controls)
    search_frame = pd.concat(search_rows, ignore_index=True)
    oof_candidates = pd.DataFrame(accepted_rows)
    if oof_candidates.empty:
        # Empty research runs are expected. Preserve a readable schema so
        # inference/replay can safely consume a negative-result artifact.
        oof_candidates = pd.DataFrame(columns=OVERLAY_CANDIDATE_COLUMNS)
    if not oof_candidates.empty:
        oof_candidates = oof_candidates.sort_values("objective", ascending=False).drop_duplicates(
            ["side_name", "archetype_policy_key"], keep="first"
        )

    valid["interpretable_rule_risk_score"] = np.float32(np.nan)
    valid["interpretable_rule_risk_percentile"] = np.float32(0.5)
    selected_params: dict[tuple[str, str], dict[str, Any]] = {}
    for row in oof_candidates.to_dict("records"):
        key = (str(row["side_name"]), str(row["archetype_policy_key"]), str(row["model_arm"]))
        state = final_states.get(key)
        if state is None:
            continue
        idx = state["index"]
        valid.loc[idx, "interpretable_rule_risk_score"] = state["score"]
        valid.loc[idx, "interpretable_rule_risk_percentile"] = base._midrank(state["score"], state["reference"])
        selected_params[(key[0], key[1])] = {**row, "risk_variant": "interpretable_rule_risk_percentile"}
        joblib.dump(state["bundle"], args.output / f"model__{key[2]}__{key[0]}__{key[1]}.joblib", compress=3)
    parent = valid["parent_rank_v9"].to_numpy(np.float32)
    adjusted, flagged = base._apply_selected_overlays(valid, selected_params, "parent_rank_v9")
    valid["parent_rank_v9_interpretable_rule_overlay"] = adjusted
    valid["interpretable_rule_overlay_flagged"] = flagged

    train_mechanisms = _calendar(train, train, "parent_rank_v9")
    eval_mechanisms = _calendar(valid.assign(parent_rank_v9_interpretable_rule_overlay=adjusted), train, "parent_rank_v9_interpretable_rule_overlay")
    mechanism_calendar = pd.concat([
        train_mechanisms.assign(calendar_partition="train"),
        eval_mechanisms.assign(calendar_partition="untouched_eval"),
    ], ignore_index=True)
    if not oof_frame.empty and "episode_phase" in oof_frame.columns:
        phase_metrics = (
            oof_frame.groupby(["side_name", "archetype_policy_key", "model_arm", "episode_phase"], observed=True)
            .agg(
                rows=(base.TARGET, "size"),
                adverse_rate=(base.TARGET, "mean"),
                mean_risk_percentile=(base.RISK_PCT, "mean"),
                mean_ev=("ev_after_1pct", "mean"),
                clean_precision=("clean_exec", "mean"),
            )
            .reset_index()
        )
    else:
        phase_metrics = pd.DataFrame()
    if not phase_metrics.empty:
        phase_separation = (
            phase_metrics.pivot_table(
                index=["side_name", "archetype_policy_key", "model_arm"],
                columns="episode_phase",
                values="mean_risk_percentile",
                aggfunc="mean",
            )
            .reset_index()
        )
        for phase in ("onset", "persistent", "recovery"):
            if phase in phase_separation.columns and "normal" in phase_separation.columns:
                phase_separation[f"{phase}_minus_normal_risk_pct"] = (
                    phase_separation[phase] - phase_separation["normal"]
                )
    else:
        phase_separation = pd.DataFrame()
    if not model_frame.empty:
        cross_archetype_summary = (
            model_frame.loc[model_frame["stage"].eq("oof")]
            .groupby(
                ["side_name", "archetype_policy_key", "model_arm", "model_target"],
                observed=True,
            )
            .agg(
                chronological_folds=("fold_start", "nunique"),
                mean_train_rows=("train_rows", "mean"),
                mean_state_rows=("train_state_rows", "mean"),
                mean_selected_features=("features", "mean"),
                selected_features=("selected_features", "last"),
            )
            .reset_index()
        )
        if not oof_candidates.empty:
            accepted_local = oof_candidates.loc[:, [
                "side_name", "archetype_policy_key", "model_arm", "model_target",
                "risk_lift", "risk_fpr", "recognized_event_blocks", "objective",
            ]].copy()
            cross_archetype_summary = cross_archetype_summary.merge(
                accepted_local,
                on=["side_name", "archetype_policy_key", "model_arm", "model_target"],
                how="left",
                validate="one_to_one",
            )
    else:
        cross_archetype_summary = pd.DataFrame()
    july_controls, july_context = _july_matched_controls(
        args.july_predictions, args.state_artifact, train, candidates
    )
    july_mechanisms = pd.DataFrame()
    if not july_context.empty:
        july_context["day"] = july_context["__ts__"].dt.floor("D")
        july_context[base.EVENT] = np.int8(0)
        july_mechanisms = _calendar(july_context, train, "parent_rank_v9")
        july_mechanisms["calendar_partition"] = "july_forward_diagnostic"
        mechanism_calendar = pd.concat(
            [mechanism_calendar, july_mechanisms], ignore_index=True, copy=False
        )

    parent_metrics = base._selection_metrics(valid, parent, config.top10_floor)
    adjusted_metrics = base._selection_metrics(valid, adjusted, config.top10_floor)
    summary = pd.DataFrame([
        {"selector": "v9_parent", **parent_metrics},
        {"selector": "v9_interpretable_rule_overlay", **adjusted_metrics},
    ])
    for metric in ("mean_ev", "positive_ev_rate", "clean_precision", "event_mean_ev", "normal_mean_ev"):
        summary[f"delta_{metric}_vs_parent"] = summary[metric] - parent_metrics[metric]
    event_interventions, event_intervention_summary = _event_intervention_report(
        valid,
        parent,
        adjusted,
        flagged,
        top10_floor=config.top10_floor,
    )
    oos_validation = _validate_oos_candidates(
        oof_candidates,
        event_interventions,
        minimum_event_cells=args.minimum_oos_event_cells,
        minimum_intervention_recall=args.minimum_oof_event_intervention_recall,
        minimum_improved_cells=args.minimum_oof_improved_event_cells,
        minimum_activity_ratio=args.minimum_oos_activity_ratio,
    )
    validated_keys = set(
        zip(
            oos_validation.loc[oos_validation["oos_validated"], "side_name"].astype(str),
            oos_validation.loc[oos_validation["oos_validated"], "archetype_policy_key"].astype(str),
        )
    )
    accepted = oof_candidates.loc[
        [
            (str(row.side_name), str(row.archetype_policy_key)) in validated_keys
            for row in oof_candidates.itertuples(index=False)
        ]
    ].copy()

    valid.to_parquet(args.output / "oos_predictions.parquet", index=False, compression="zstd")
    oof_frame.to_parquet(args.output / "train_oof_predictions.parquet", index=False, compression="zstd")
    global_market_oof.to_parquet(
        args.output / "global_market_episode_context_oof.parquet",
        index=False,
        compression="zstd",
    )
    global_market_report.to_csv(
        args.output / "global_market_episode_context_report.csv", index=False
    )
    if global_market_bundle is not None:
        joblib.dump(
            global_market_bundle,
            args.output / "global_market_episode_context.joblib",
            compress=3,
        )
    market_adverse_subtype_report.to_csv(
        args.output / "market_adverse_subtype_context_report.csv", index=False
    )
    if not pooled_daily_reliability.empty:
        pooled_daily_reliability.to_parquet(
            args.output / "pooled_daily_mechanism_reliability_train.parquet",
            index=False,
            compression="zstd",
        )
    if market_adverse_subtype_encoder is not None:
        joblib.dump(
            market_adverse_subtype_encoder,
            args.output / "market_adverse_subtype_context.joblib",
            compress=3,
        )
    side_market_oof.to_parquet(
        args.output / "side_market_episode_context_oof.parquet",
        index=False,
        compression="zstd",
    )
    side_market_report.to_csv(
        args.output / "side_market_episode_context_report.csv", index=False
    )
    for side, bundle in side_market_bundles.items():
        joblib.dump(
            bundle,
            args.output / f"side_market_episode_context__{side}.joblib",
            compress=3,
        )
    model_frame.to_csv(args.output / "model_report.csv", index=False)
    rules_frame.to_csv(args.output / "extracted_rules.csv", index=False)
    controls_frame.to_csv(args.output / "train_matched_benign_controls.csv", index=False)
    july_controls.to_csv(args.output / "july_06_09_10_matched_benign_controls.csv", index=False)
    july_mechanisms.to_csv(args.output / "july_mechanism_calendar.csv", index=False)
    search_frame.to_csv(args.output / "overlay_search.csv", index=False)
    oof_candidates.to_csv(args.output / "oof_accepted_candidates.csv", index=False)
    oos_validation.to_csv(args.output / "oos_candidate_validation.csv", index=False)
    accepted.to_csv(args.output / "accepted_overlays.csv", index=False)
    mechanism_calendar.to_csv(args.output / "residual_calendar_with_mechanisms.csv", index=False)
    phase_metrics.to_csv(args.output / "episode_phase_oof_metrics.csv", index=False)
    phase_separation.to_csv(args.output / "episode_phase_separation.csv", index=False)
    cross_archetype_summary.to_csv(args.output / "cross_archetype_pattern_summary.csv", index=False)
    event_interventions.to_csv(args.output / "oos_event_intervention_report.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    pd.concat([
        base._breakdown(valid, "v9_parent", parent),
        base._breakdown(valid, "v9_interpretable_rule_overlay", adjusted),
    ], ignore_index=True).to_csv(args.output / "breakdowns.csv", index=False)
    _write_dossier(
        args.output, oof_candidates, rules_frame, model_frame, args.baseline_overlay
    )

    manifest = {
        "schema": "meta_residual_interpretable_rule_overlay_v2_period_state",
        "config": asdict(config), "coverage": coverage, "model_arms": selected_arms,
        "episode_state_granularity": args.episode_state_granularity,
        "target_groups": [list(group) for group in target_groups],
        "unsupervised_feature_map": unsupervised_feature_map,
        "episode_phase_contract": (
            "Onset/persistent/recovery phases are outcome-only labels derived from local calendar "
            "blocks. Phase-specific LGBM/MLP arms use onset or persistent membership only as their "
            "training target; all inputs remain pre-entry observable features. Recovery is diagnostic "
            "only. No phase column is ever passed as an inference feature."
        ),
        "overlay_target_contract": (
            "all parent-top-10 rows within a side x archetype adverse period; "
            "the day-level event is derived from aggregate clean-rate surprise "
            "and after-cost EV, with onset and persistence labels"
            if args.overlay_target_mode == "top10_period_event"
            else "parent-top-10 rows during a multi-archetype broad market adverse period; "
            "the day-level event requires two or more distinct adverse side x archetype cells"
        ),
        "period_state_contract": {
            "decision_population": "parent top-10 rows by side x archetype",
            "context_population": "same side x archetype parent top-20 rows at the decision timestamp",
            "context_floor": PERIOD_CONTEXT_FLOOR,
            "derived_observable_features": list(PERIOD_STATE_FEATURES),
            "trajectory_source_features": list(EPISODE_TRAJECTORY_SOURCE_FEATURES),
            "trajectory_features": list(EPISODE_TRAJECTORY_FEATURES),
            "trajectory_contract": (
                "For each daily-open state, delta_6h, delta_24h and delta_48h use only "
                "the latest observable value at or before t-horizon. Acceleration compares "
                "6h/24h and 24h/48h velocities; trend agreement/intensity summarize their "
                "signed coherence; state_variability_48h uses observations at or before t. "
                "Missing historical support remains missing."
            ),
            "trajectory_clock": (
                "Full candidate-universe timestamp median for observable market sources; "
                "not local archetype candidate availability. As-of lookup tolerance is 90 minutes."
            ),
            "label_source": (
                "aggregate daily selected-trade clean-rate surprise and after-cost EV; "
                "repeated only across the parent top-10 decision rows in an adverse period"
            ),
            "forbidden_inference_inputs": [
                "ev_after_1pct", "clean_exec", "adverse_calendar_cell",
                "episode_phase", "individual trade path/outcome fields",
            ],
        },
        "global_market_episode_context": {
            "enabled": bool(global_market_bundle is not None),
            "features": global_market_context_features,
            "learning_unit": "one UTC day with a pre-open full-market state",
            "target": (
                "market_adverse_period: two or more distinct adverse "
                "side x archetype cells on the same day"
            ),
            "usage": (
                "fold-frozen daily risk and percentile are candidate inputs to "
                "local side x archetype overlays; they never adjust parent rank directly"
            ),
            "oof_days": int(len(global_market_oof)),
        },
        "market_adverse_subtype_context": {
            "enabled": bool(market_adverse_subtype_encoder is not None),
            "features": market_adverse_subtype_features,
            "learning_unit": "one UTC day with a pre-open pooled market state",
            "fit_population": (
                "train-side broad market-adverse days only; transform receives "
                "only frozen observable-state GMM posteriors, entropy, and density"
            ),
            "usage": (
                "candidate context for local overlays only; no direct parent-rank "
                "adjustment and no realized outcome field at inference"
            ),
        },
        "pooled_mechanism_reliability": {
            "enabled": "episode_lgbm_pooled_reliability" in selected_arms,
            "requires_pooled_daily_context": True,
            "mechanism": (
                "train-fold quantile bins over invariant market adverse-subtype "
                "negative-log-density and entropy; never a cross-fold GMM component id"
            ),
            "features": list(POOLED_RELIABILITY_FEATURES),
            "shrinkage_k": float(args.pooled_reliability_shrinkage_k),
            "fit_contract": (
                "Local reliability is an expanding prior excluding the current UTC day; "
                "it is shrunk local side x archetype -> same side -> global for each "
                "mechanism bin. Scored days receive only frozen pre-score aggregates."
            ),
            "pooled_train_daily_cells": int(len(pooled_daily_reliability)),
        },
        "side_market_episode_context": {
            "enabled": bool(side_market_bundles),
            "features": side_market_context_features,
            "sides": sorted(side_market_bundles),
            "learning_unit": "one UTC day per side with a pre-open full-market state",
            "target": "side_adverse_calendar_cell: any adverse local archetype cell on the same side/day",
            "usage": (
                "fold-frozen same-side daily risk and percentile are candidate inputs "
                "to local side x archetype overlays; they never adjust parent rank directly"
            ),
            "oof_days": int(len(side_market_oof)),
        },
        "period_control_contract": {
            "mode": args.period_control_mode,
            "timestamp": "nearest benign state timestamps in robust observable space",
            "episode_windows": "same-duration benign calendar windows in robust observable space",
        },
        "episode_intervention_promotion_contract": {
            "minimum_oof_event_intervention_recall": float(
                args.minimum_oof_event_intervention_recall
            ),
            "minimum_oof_improved_event_cells": int(
                args.minimum_oof_improved_event_cells
            ),
            "contract": (
                "An arm must change and improve difficult OOF episode cells; "
                "normal-period reallocation alone cannot make it promotable."
            ),
        },
        "untouched_oos_validation_contract": {
            "minimum_event_cells": int(args.minimum_oos_event_cells),
            "minimum_intervention_recall": float(args.minimum_oof_event_intervention_recall),
            "minimum_improved_cells": int(args.minimum_oof_improved_event_cells),
            "minimum_activity_ratio": float(args.minimum_oos_activity_ratio),
            "contract": (
                "OOF-promotable candidates remain research-only until their frozen final "
                "state both intervenes in and improves enough untouched daily episode cells."
            ),
        },
        "candidate_features": candidates,
        "oof_accepted_candidates": oof_candidates.to_dict("records"),
        "oos_candidate_validation": oos_validation.to_dict("records"),
        "accepted_overlays": accepted.to_dict("records"),
        "oos_event_intervention": event_intervention_summary,
        "july_matched_controls": int(len(july_controls)),
        "model_based_recursive_partitioning": "executed as shallow event-balanced Bernoulli log-likelihood recursive partitioning",
        "leakage_contract": (
            "Every model arm, feature screen, robust transform, rule threshold, percentile reference, "
            "matched training control, and overlay parameter is fitted on chronological train folds only. "
            "April-June 2026 is untouched evaluation. July 6/9/10 prototypes generate diagnostic matched "
            "controls only and are explicitly excluded from model fitting and promotion."
        ),
    }
    _write_json(args.output / "manifest.json", manifest)
    print(summary.to_string(index=False), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Keep paths identical to the established event-balanced runner.
    parser.add_argument("--champion-ledger", type=Path, default=Path("data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/champion_frozen_single_source_202501_20260710/frozen_champion_single_source_ledger.parquet"))
    parser.add_argument("--train-oof-predictions-dir", type=Path, default=Path("data_perp/reports/s59_h5_2025start_monthly_v4_base_configfull_mdafs120_hpo150_largestfold_oos15_ae3000_nocrossfit_k34567_payload300k_20260706/train_meta_regime_handoff_singlehead_base_soft_lgbmpipeline_auto_hpo150_oos15_top30_hpo45k_20260706_v5/best_full_oos_fixedfs_streamed_v1/prediction_shards"))
    parser.add_argument("--train-oof-rank-cache", type=Path, default=Path("data_perp/reports/residual_event_archetype_true_base_oof_compactlocal_market_20260712_v3/meta_oof_global_rank_202504_202603.parquet"))
    parser.add_argument("--state-artifact", type=Path, required=True)
    parser.add_argument(
        "--additional-state-artifact",
        type=Path,
        action="append",
        default=[],
        help="Additional disjoint OOS residual-state artifact for extended history.",
    )
    parser.add_argument(
        "--direct-parent-rank",
        action="store_true",
        help=(
            "Research-only: use a supplied causal candidate parent rank directly, "
            "without applying the V9 strict local rank adjustment."
        ),
    )
    parser.add_argument(
        "--state-group-filter",
        default="",
        help="Optional side::archetype parquet-level state filter for isolated local studies.",
    )
    parser.add_argument(
        "--pooled-daily-context",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fit global and side daily context from a compact all-archetype parent "
            "stream plus timestamp-level market features, while retaining the local "
            "state filter for the overlay itself. Requires --direct-parent-rank."
        ),
    )
    parser.add_argument(
        "--pooled-reliability-shrinkage-k",
        type=float,
        default=20.0,
        help=(
            "Empirical-Bayes support shrinkage for the pooled mechanism reliability arm. "
            "Higher values retain more same-side/global prior influence in sparse local cells."
        ),
    )
    parser.add_argument("--parent-eval-predictions", type=Path, default=Path("data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/lifecycle_residual_aware_ae_gmm_overlay_pca8_clip8_baseline_globaloverlay_sparse_shock_composite/oos_predictions_historical_rank.parquet"))
    parser.add_argument("--v9-predictions", type=Path, default=Path("data_perp/reports/meta_residual_extreme_local_champion_overlay_ooftrain_tieaware_downonly_20260712_v9/oos_predictions.parquet"))
    parser.add_argument("--v9-manifest", type=Path, default=Path("data_perp/reports/meta_residual_extreme_local_champion_overlay_ooftrain_tieaware_downonly_20260712_v9/manifest.json"))
    parser.add_argument("--v9-selected-features", type=Path, default=Path("data_perp/reports/meta_residual_extreme_local_champion_overlay_ooftrain_tieaware_downonly_20260712_v9/selected_local_features_strict.csv"))
    parser.add_argument("--negative-residual-features", type=Path, default=Path("data_perp/features/20260712_185800/symbol=BTC_USD:USD.parquet"))
    parser.add_argument("--temporal-state-features", type=Path, default=Path("data_perp/reports/residual_event_target_transitions_july_oos_20260713_v2_support_fallback/oos_temporal_state_context_apr2025_july2026.parquet"))
    parser.add_argument("--event-calendar", type=Path, default=Path("data_perp/reports/residual_episode_recognition_calendar_20260712_v1/calendar_recognized_vs_ignored.csv"))
    parser.add_argument("--extension-calendar", type=Path, default=Path("data_perp/reports/residual_event_target_transitions_july_oos_20260713_v2_support_fallback/residual_event_calendar.csv"))
    parser.add_argument("--july-predictions", type=Path, default=Path("data_perp/reports/meta_residual_event_balanced_error_overlay_20260713_v10_frozen_candidate/july_forward/july_predictions.parquet"))
    parser.add_argument("--baseline-overlay", type=Path, default=Path("data_perp/reports/meta_residual_event_balanced_error_overlay_20260713_v10_frozen_candidate"))
    parser.add_argument("--output", type=Path, default=Path("data_perp/reports/meta_residual_interpretable_rule_overlay_20260713_v1"))
    parser.add_argument(
        "--groups",
        default="all",
        help="Comma-separated side::archetype pairs, 'all' for every supported group, or 'default'.",
    )
    parser.add_argument(
        "--min-group-rows",
        type=int,
        default=1_200,
        help="Minimum train rows for an automatically selected side x archetype group.",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=0,
        help="Optional cap for smoke runs; 0 evaluates every supported group.",
    )
    parser.add_argument(
        "--use-unsupervised-composites",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Discover local train-only nonlinear composites before fitting arms.",
    )
    parser.add_argument(
        "--max-unsupervised-rows",
        type=int,
        default=90_000,
        help=(
            "Maximum top-20 training rows used only for local composite discovery; "
            "all local overlay fits retain their full eligible train rows."
        ),
    )
    parser.add_argument(
        "--arms",
        default=",".join(ARMS),
        help="Comma-separated model arms. Defaults to every supported arm.",
    )
    parser.add_argument(
        "--overlay-target-mode",
        choices=(
            "top10_period_event",
            "top10_market_period_event",
        ),
        default="top10_period_event",
        help=(
            "Use the local side x archetype adverse-period target or the secondary "
            "multi-archetype market-episode target. Both are daily labels."
        ),
    )
    parser.add_argument(
        "--period-control-mode",
        choices=("timestamp", "episode_windows"),
        default="timestamp",
        help=(
            "Train-side control sampler for period targets. Timestamp controls are "
            "the current default; equal-duration episode windows remain a diagnostic arm."
        ),
    )
    parser.add_argument(
        "--episode-state-granularity",
        choices=("daily_open",),
        default="daily_open",
        help=(
            "Learning unit for period overlays. daily_open uses one pre-open, "
            "full-market signature per side x archetype day. Intraday candidate "
            "rows are used only to apply and evaluate that daily state score."
        ),
    )
    parser.add_argument(
        "--minimum-oof-event-intervention-recall",
        type=float,
        default=0.20,
        help=(
            "Minimum share of local OOF adverse episode cells whose parent top-10 "
            "admission changes before an overlay can be promoted."
        ),
    )
    parser.add_argument(
        "--minimum-oof-improved-event-cells",
        type=int,
        default=2,
        help=(
            "Minimum number of local OOF adverse episode cells with improved mean "
            "EV after the frozen overlay action."
        ),
    )
    parser.add_argument(
        "--minimum-oos-event-cells",
        type=int,
        default=2,
        help=(
            "Minimum untouched daily adverse cells required before an OOF candidate "
            "can be called validated or written to accepted_overlays.csv."
        ),
    )
    parser.add_argument(
        "--minimum-oos-activity-ratio",
        type=float,
        default=0.90,
        help=(
            "Minimum retained parent top-10 activity for an untouched OOS-validated "
            "episode overlay."
        ),
    )
    parser.add_argument("--train-start", default="2025-04-01")
    parser.add_argument("--train-end", default="2026-04-01")
    parser.add_argument("--eval-end", default="2026-07-01")
    parser.add_argument("--max-features", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()
    print(json.dumps(_safe_json(run(args)), indent=2))


if __name__ == "__main__":
    main()
