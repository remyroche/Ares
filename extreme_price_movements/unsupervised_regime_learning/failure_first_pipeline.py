"""End-to-end, fail-closed primitives for failure-first regime research.

The failure calendar and taxonomy are outcome-derived research artifacts.  The
hourly state table is deliberately separate and contains decision-time fields
only.  A supervised detector may be fitted only after the explicit sufficiency
gate passes; a caller cannot silently train on a handful of failure episodes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import re
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

from extreme_price_movements.unsupervised_regime_learning.failure_first_hourly import (
    DEFAULT_WINDOW_OFFSETS_HOURS,
    build_hourly_state_transition_labels,
    validate_inference_feature_columns,
)


DEFAULT_MARKET_FEATURES = (
    "mkt_state__volatility_of_volatility_48__h0",
    "mkt_state__atr_compression_ratio__h0",
    "mkt_state__atr_slope__h0",
    "mkt_state__atr_pct_change__h0",
    "mkt_state__mkt_atr_expansion_4h__h0",
    "mkt_state__efficiency_ratio_20__h0",
    "mkt_state__path_efficiency_24__h0",
    "mkt_state__trend_r2_24__h0",
    "mkt_state__range_expansion_ratio__h0",
    "mkt_state__breakout_efficiency_4h__h0",
    "mkt_state__market_breadth_4h__h0",
    "mkt_state__negative_breadth_pct__h0",
    "mkt_state__breadth_dispersion__h0",
    "mkt_state__market_breadth_recovery_from_24h_min__h0",
    "mkt_state__avg_pair_corr_24h__h0",
    "mkt_state__corr_concentration_24h__h0",
    "mkt_state__market_downside_pairwise_corr_24h__h0",
    "mkt_state__mkt_funding_mean_z_30d__h0",
    "mkt_state__mkt_funding_dispersion_z_30d__h0",
    "mkt_state__mkt_funding_chg_4h__h0",
)

DEFAULT_MODEL_HEALTH_FEATURES = (
    "causal_recent_side_isotonic_ev",
    "catboost__residual__without_hpo__all_features",
    "existing_alpha_ev",
    "alpha_prediction_uncertainty",
    "alpha_leaf_support",
    "pred_peak_MFE_12h_ATR",
    "base_oof_score",
    "base_margin_to_cutoff_z",
    "oof_clean_favorable_probability",
    "catboost_entropy",
    "catboost_adverse_probability_mass",
    "catboost_favorable_probability_mass",
)

IDENTITY_COLUMNS = (
    "candidate_id",
    "__ts__",
    "__symbol__",
    "side_name",
    "execution_decision_utc",
)


@dataclass(frozen=True)
class FailureFirstSufficiencyConfig:
    """Predeclared support required before taxonomy or detector fitting."""

    minimum_failure_episodes: int = 40
    minimum_complete_window_episodes: int = 40
    minimum_failure_bins: int = 40
    minimum_span_days: int = 180
    minimum_observed_days: int = 180
    maximum_calendar_gap_days: int = 21
    minimum_profile_features: int = 10
    minimum_detector_rows: int = 1_000
    minimum_transitions: int = 50
    maximum_detector_features: int = 40


@dataclass
class FrozenFailureTaxonomyBundle:
    """Persistable failure-only taxonomy frozen before detector evaluation."""

    method: str
    profile_columns: list[str]
    imputer: SimpleImputer
    scaler: RobustScaler
    model: Any
    label_map: dict[int, str]
    selected_clusters: int
    fit_cutoff_utc: str
    train_episode_ids: list[str]
    robust_clip: float

    def score(self, profiles: pd.DataFrame) -> pd.DataFrame:
        _require(
            profiles,
            ("episode_id", *self.profile_columns),
            "taxonomy scoring profiles",
        )
        raw = (
            profiles.loc[:, self.profile_columns]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(np.float64)
        )
        values = np.clip(
            self.scaler.transform(self.imputer.transform(raw)),
            -float(self.robust_clip),
            float(self.robust_clip),
        )
        labels = np.asarray(self.model.predict(values), dtype=np.int16)
        if self.method == "gmm":
            probability = np.asarray(
                self.model.predict_proba(values), dtype=np.float64
            ).max(axis=1)
        else:
            distance = np.asarray(self.model.transform(values), dtype=np.float64)
            inverse = 1.0 / np.maximum(distance, 1e-6)
            probability = inverse.max(axis=1) / inverse.sum(axis=1)
        output = profiles[["episode_id"]].copy()
        output["expost__failure_cluster"] = labels
        output["expost__failure_cluster_probability"] = probability.astype(
            np.float32
        )
        output["expost__failure_taxonomy_label"] = [
            self.label_map[int(value)] for value in labels
        ]
        return output


def _utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce")


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _require(frame: pd.DataFrame, columns: Sequence[str], source: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise KeyError(f"{source} missing required columns: {missing}")


def _unique_candidate(frame: pd.DataFrame, source: str) -> None:
    if frame["candidate_id"].isna().any():
        raise ValueError(f"{source} contains null candidate_id")
    if frame["candidate_id"].duplicated().any():
        raise ValueError(f"{source} candidate_id is not unique")


def prepare_failure_first_sources(
    ledger: pd.DataFrame,
    state_source: pd.DataFrame,
    *,
    rich_context: pd.DataFrame | None = None,
    requested_market_features: Sequence[str] = DEFAULT_MARKET_FEATURES,
    requested_health_features: Sequence[str] = DEFAULT_MODEL_HEALTH_FEATURES,
    score_valid_flag: str = "causal_recent_side_isotonic_ev__is_oof",
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Join explicit strict model-OOS outcomes to observable decision context."""

    _require(
        ledger,
        (
            *IDENTITY_COLUMNS,
            "execution_label_end_utc",
            "execution_gross_ev_12h",
            "execution_net_ev_12h",
            "causal_recent_side_isotonic_ev",
            score_valid_flag,
        ),
        "ledger",
    )
    _require(state_source, IDENTITY_COLUMNS, "state source")
    ledger = ledger.copy()
    state_source = state_source.copy()
    _unique_candidate(ledger, "ledger")
    _unique_candidate(state_source, "state source")
    ledger["execution_decision_utc"] = _utc(ledger["execution_decision_utc"])
    ledger["execution_label_end_utc"] = _utc(ledger["execution_label_end_utc"])
    strict = ledger.loc[
        ledger[score_valid_flag]
        .fillna(False)
        .astype(bool)
    ].copy()
    if strict.empty:
        raise ValueError(
            f"ledger contains no rows under strict score flag {score_valid_flag}"
        )
    if strict["execution_decision_utc"].isna().any():
        raise ValueError("strict ledger has invalid execution_decision_utc")

    requested = list(
        dict.fromkeys([*requested_market_features, *requested_health_features])
    )
    joined = strict.copy()
    source_frames = [("state", state_source)]
    if rich_context is not None:
        rich_context = rich_context.copy()
        _require(rich_context, IDENTITY_COLUMNS, "rich context")
        _unique_candidate(rich_context, "rich context")
        source_frames.append(("rich", rich_context))
    feature_sources: dict[str, str] = {}
    for source_name, source in source_frames:
        additions = [
            name
            for name in requested
            if name in source.columns and name not in joined.columns
        ]
        availability = [
            name
            for name in (
                "raw_state_source_utc_h0",
                "alpha_available_at",
                "peak_mfe_available_at",
                "catboost_available_at",
            )
            if name in source.columns and name not in joined.columns
        ]
        if additions or availability:
            joined = joined.merge(
                source.loc[:, ["candidate_id", *additions, *availability]],
                on="candidate_id",
                how="left",
                validate="one_to_one",
            )
            feature_sources.update({name: source_name for name in additions})
    for name in requested:
        if name in strict.columns:
            feature_sources[name] = "ledger"

    feature_columns = [name for name in requested if name in joined.columns]
    validate_inference_feature_columns(feature_columns)
    if not feature_columns:
        raise ValueError("no requested observable failure-first features are available")
    for name in (
        "raw_state_source_utc_h0",
        "alpha_available_at",
        "peak_mfe_available_at",
        "catboost_available_at",
    ):
        if name not in joined:
            continue
        available = _utc(joined[name])
        invalid = available.notna() & available.gt(joined["execution_decision_utc"])
        if invalid.any():
            raise ValueError(
                f"{name} is after decision time for {int(invalid.sum())} rows"
            )
    joined["observable_feature_coverage"] = (
        joined.loc[:, feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .notna()
        .mean(axis=1)
    )
    audit = {
        "strict_oof_rows": int(len(strict)),
        "strict_score_valid_flag": str(score_valid_flag),
        "joined_rows": int(len(joined)),
        "state_rows_matched": int(
            joined["candidate_id"].isin(state_source["candidate_id"]).sum()
        ),
        "rich_rows_matched": int(
            joined["candidate_id"].isin(rich_context["candidate_id"]).sum()
        )
        if rich_context is not None
        else 0,
        "selected_observable_features": feature_columns,
        "feature_sources": feature_sources,
        "missing_requested_features": [
            name for name in requested if name not in joined.columns
        ],
        "mean_observable_feature_coverage": float(
            joined["observable_feature_coverage"].mean()
        ),
    }
    return joined, feature_columns, audit


def build_hourly_observable_state(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    timestamp_col: str = "execution_decision_utc",
) -> tuple[pd.DataFrame, list[str]]:
    """Aggregate candidate context into one global decision-time hourly state."""

    validate_inference_feature_columns(feature_columns)
    _require(
        frame,
        (timestamp_col, "__symbol__", "side_name", *feature_columns),
        "observable source",
    )
    work = frame.copy()
    work[timestamp_col] = _utc(work[timestamp_col]).dt.floor("h")
    work = work.loc[work[timestamp_col].notna()].copy()
    for name in feature_columns:
        work[name] = pd.to_numeric(work[name], errors="coerce")
    rows: list[dict[str, Any]] = []
    for stamp, group in work.groupby(timestamp_col, sort=True, observed=True):
        counts = group["__symbol__"].astype(str).value_counts(normalize=True)
        row: dict[str, Any] = {
            timestamp_col: stamp,
            "side_name": "global",
            "state_asof_utc": stamp,
            "candidate_rows": int(len(group)),
            "distinct_assets": int(group["__symbol__"].nunique()),
            "side_long_share": float(
                group["side_name"].astype(str).str.casefold().eq("long").mean()
            ),
            "asset_hhi": float(np.square(counts.to_numpy(np.float64)).sum()),
        }
        for name in feature_columns:
            row[f"state__{_slug(name)}__mean"] = float(group[name].mean())
        rows.append(row)
    state = pd.DataFrame.from_records(rows)
    state_features = [
        name for name in state.columns if name.startswith("state__")
    ]
    validate_inference_feature_columns(state_features)
    return state, state_features


def extract_failure_episode_windows(
    hourly_state: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    state_feature_columns: Sequence[str],
    offsets_hours: Sequence[int] = DEFAULT_WINDOW_OFFSETS_HOURS,
    timestamp_col: str = "execution_decision_utc",
) -> pd.DataFrame:
    """Extract the fixed event-study grid around each global failure onset."""

    validate_inference_feature_columns(state_feature_columns)
    _require(
        hourly_state,
        (timestamp_col, "state_asof_utc", *state_feature_columns),
        "hourly state",
    )
    _require(
        episodes,
        (
            "episode_id",
            "episode_onset_decision_utc",
            "episode_onset_available_utc",
        ),
        "episodes",
    )
    if hourly_state[timestamp_col].duplicated().any():
        raise ValueError("hourly state must have one global row per hour")
    targets: list[dict[str, Any]] = []
    for episode in episodes.itertuples(index=False):
        values = episode._asdict()
        anchor = pd.Timestamp(values["episode_onset_decision_utc"])
        anchor = (
            anchor.tz_localize("UTC")
            if anchor.tzinfo is None
            else anchor.tz_convert("UTC")
        )
        for offset in offsets_hours:
            target = anchor + pd.Timedelta(hours=int(offset))
            targets.append(
                {
                    "episode_id": values["episode_id"],
                    "anchor_decision_utc": anchor,
                    "anchor_available_utc": values[
                        "episode_onset_available_utc"
                    ],
                    "offset_hours": int(offset),
                    "window_start_utc": target - pd.Timedelta(hours=1)
                    if int(offset) != 0
                    else target,
                    "window_end_utc": target
                    if int(offset) != 0
                    else target + pd.Timedelta(hours=1),
                    timestamp_col: target,
                    "relative_phase": "pre_onset"
                    if int(offset) < 0
                    else ("onset" if int(offset) == 0 else "post_onset"),
                }
            )
    target_frame = pd.DataFrame.from_records(targets)
    if target_frame.empty:
        return pd.DataFrame(
            columns=[
                "episode_id",
                "anchor_decision_utc",
                "anchor_available_utc",
                "offset_hours",
                "window_start_utc",
                "window_end_utc",
                timestamp_col,
                "relative_phase",
                "availability_check_pass",
                "feature_coverage",
                "window_complete",
            ]
        )
    state_columns = [
        timestamp_col,
        "state_asof_utc",
        "candidate_rows",
        "distinct_assets",
        "side_long_share",
        "asset_hhi",
        *state_feature_columns,
    ]
    output = target_frame.merge(
        hourly_state.loc[:, state_columns],
        on=timestamp_col,
        how="left",
        validate="many_to_one",
    )
    output["availability_check_pass"] = (
        _utc(output["state_asof_utc"]).le(_utc(output[timestamp_col]))
        & output["state_asof_utc"].notna()
    )
    output["feature_coverage"] = (
        output.loc[:, state_feature_columns].notna().mean(axis=1)
    )
    output["window_complete"] = (
        output["availability_check_pass"] & output["feature_coverage"].ge(0.80)
    )
    return output.sort_values(
        ["episode_id", "offset_hours"], kind="stable"
    ).reset_index(drop=True)


def extract_failure_episode_outcomes(
    ledger: pd.DataFrame,
    episode_windows: pd.DataFrame,
    *,
    timestamp_col: str = "execution_decision_utc",
) -> pd.DataFrame:
    """Build a separate ex-post outcome table keyed to the event-study grid."""

    _require(
        ledger,
        (
            timestamp_col,
            "execution_label_end_utc",
            "execution_gross_ev_12h",
            "execution_net_ev_12h",
        ),
        "outcome ledger",
    )
    keys = episode_windows.loc[
        :,
        [
            "episode_id",
            "offset_hours",
            "window_start_utc",
            "window_end_utc",
            timestamp_col,
        ],
    ].copy()
    work = ledger.copy()
    work[timestamp_col] = _utc(work[timestamp_col]).dt.floor("h")
    work["execution_label_end_utc"] = _utc(work["execution_label_end_utc"])
    work["execution_gross_ev_12h"] = pd.to_numeric(
        work["execution_gross_ev_12h"], errors="coerce"
    )
    work["execution_net_ev_12h"] = pd.to_numeric(
        work["execution_net_ev_12h"], errors="coerce"
    )
    work["__cost__"] = (
        work["execution_gross_ev_12h"] - work["execution_net_ev_12h"]
    )
    summary = (
        work.groupby(timestamp_col, sort=True, observed=True)
        .agg(
            expost__candidate_rows=("candidate_id", "size"),
            expost__gross_ev_mean=("execution_gross_ev_12h", "mean"),
            expost__cost_mean=("__cost__", "mean"),
            expost__net_ev_mean=("execution_net_ev_12h", "mean"),
            expost__positive_net_rate=(
                "execution_net_ev_12h",
                lambda values: float(pd.Series(values).gt(0.0).mean()),
            ),
            outcome_available_max_utc=("execution_label_end_utc", "max"),
        )
        .reset_index()
    )
    return keys.merge(
        summary, on=timestamp_col, how="left", validate="many_to_one"
    ).sort_values(["episode_id", "offset_hours"], kind="stable")


def build_failure_episode_profiles(
    windows: pd.DataFrame,
    *,
    state_feature_columns: Sequence[str],
    maximum_profile_features: int = 40,
) -> tuple[pd.DataFrame, list[str]]:
    """Build a compact, explicitly ex-post failure trajectory profile.

    Every observable state enters at onset.  Remaining capacity is filled by
    predeclared -12h-to-onset deltas for economically distinct market/model
    families.  This keeps clustering near the plan's 30--50 descriptor range
    instead of feeding hundreds of highly collinear window cells into a small
    failure sample.
    """

    _require(windows, ("episode_id", "offset_hours"), "episode windows")
    output = windows[["episode_id"]].drop_duplicates().set_index("episode_id")
    profile_columns: list[str] = []
    if int(maximum_profile_features) < 1:
        raise ValueError("maximum_profile_features must be positive")
    ordered = list(dict.fromkeys(state_feature_columns))
    onset = windows.loc[windows["offset_hours"].eq(0)].set_index("episode_id")
    for feature in ordered[: int(maximum_profile_features)]:
        name = (
            f"expost__profile__{feature.removeprefix('state__')}__onset"
        )
        output[name] = pd.to_numeric(onset[feature], errors="coerce")
        profile_columns.append(name)
    remaining = int(maximum_profile_features) - len(profile_columns)
    if remaining > 0:
        family_tokens = (
            "volatility_of_volatility",
            "atr_slope",
            "market_breadth_4h",
            "avg_pair_corr",
            "funding_chg_4h",
            "causal_recent_side_isotonic_ev",
            "base_margin_to_cutoff_z",
            "catboost_entropy",
        )
        transition_features: list[str] = []
        for token in family_tokens:
            match = next(
                (name for name in ordered if token in name),
                None,
            )
            if match is not None and match not in transition_features:
                transition_features.append(match)
        pre = windows.loc[windows["offset_hours"].eq(-12)].set_index(
            "episode_id"
        )
        for feature in transition_features[:remaining]:
            name = (
                f"expost__profile__{feature.removeprefix('state__')}"
                "__delta_h-12_to_onset"
            )
            output[name] = pd.to_numeric(
                onset[feature], errors="coerce"
            ) - pd.to_numeric(pre[feature], errors="coerce")
            profile_columns.append(name)
    return output.reset_index(), profile_columns


def _taxonomy_family(column: str) -> str:
    value = column.casefold()
    if "fund" in value:
        return "funding_transition"
    if any(token in value for token in ("oi_", "open_interest", "leverage")):
        return "leverage_repricing"
    if any(token in value for token in ("liquid", "spread", "depth", "volume")):
        return "liquidity_dislocation"
    if any(token in value for token in ("breadth", "corr", "dispersion")):
        return "correlation_fragmentation"
    if any(token in value for token in ("vol", "atr", "range", "wick")):
        return "volatility_expansion"
    if any(token in value for token in ("ret", "price", "trend", "momentum")):
        return "directional_transition"
    return "mixed_observable_state"


def fit_frozen_failure_taxonomy(
    profiles: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    profile_columns: Sequence[str],
    fit_cutoff_utc: pd.Timestamp,
    method: str = "gmm",
    min_clusters: int = 5,
    max_clusters: int = 8,
    minimum_cluster_episodes: int = 5,
    robust_clip: float = 2.0,
    random_state: int = 20260726,
) -> tuple[
    FrozenFailureTaxonomyBundle,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Fit one failure-only taxonomy before the detector evaluation origin."""

    if method not in {"gmm", "kmeans"}:
        raise ValueError("taxonomy method must be gmm or kmeans")
    if not 2 <= int(min_clusters) <= int(max_clusters) <= 8:
        raise ValueError("taxonomy cluster range must satisfy 2 <= min <= max <= 8")
    columns = list(dict.fromkeys(profile_columns))
    if not columns or any(not name.startswith("expost__") for name in columns):
        raise ValueError("taxonomy requires explicit ex-post profile columns")
    _require(
        episodes,
        ("episode_id", "episode_end_available_utc"),
        "taxonomy episodes",
    )
    boundary = pd.Timestamp(fit_cutoff_utc)
    if boundary.tzinfo is None:
        raise ValueError("taxonomy fit cutoff must be timezone-aware")
    boundary = boundary.tz_convert("UTC")
    available = episodes.loc[
        _utc(episodes["episode_end_available_utc"]).lt(boundary),
        ["episode_id"],
    ]
    train = profiles.merge(
        available, on="episode_id", how="inner", validate="one_to_one"
    )
    if len(train) < int(min_clusters) * int(minimum_cluster_episodes):
        raise ValueError(
            "insufficient pre-cutoff failure episodes for stable taxonomy: "
            f"{len(train)}"
        )
    raw = (
        train.loc[:, columns]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(np.float64)
    )
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    if not np.isfinite(robust_clip) or float(robust_clip) <= 0:
        raise ValueError("taxonomy robust_clip must be finite and positive")
    values = np.clip(
        scaler.fit_transform(imputer.fit_transform(raw)),
        -float(robust_clip),
        float(robust_clip),
    )
    candidates: list[dict[str, Any]] = []
    fitted: dict[int, Any] = {}
    labels_by_k: dict[int, np.ndarray] = {}
    upper = min(int(max_clusters), len(train) // int(minimum_cluster_episodes))
    for clusters in range(int(min_clusters), upper + 1):
        if method == "gmm":
            model: Any = GaussianMixture(
                n_components=clusters,
                covariance_type="diag",
                reg_covar=1e-3,
                n_init=5,
                max_iter=300,
                random_state=int(random_state),
            ).fit(values)
            labels = np.asarray(model.predict(values), dtype=np.int16)
            objective = -float(model.bic(values))
            criterion = "negative_bic"
        else:
            model = KMeans(
                n_clusters=clusters,
                n_init=30,
                max_iter=400,
                random_state=int(random_state),
            ).fit(values)
            labels = np.asarray(model.labels_, dtype=np.int16)
            objective = float(silhouette_score(values, labels))
            criterion = "silhouette"
        counts = np.bincount(labels, minlength=clusters)
        support_pass = bool(counts.min() >= int(minimum_cluster_episodes))
        candidates.append(
            {
                "expost__cluster_count": clusters,
                "expost__selection_objective": objective,
                "expost__selection_criterion": criterion,
                "expost__minimum_cluster_episodes": int(counts.min()),
                "expost__cluster_support_pass": support_pass,
            }
        )
        fitted[clusters] = model
        labels_by_k[clusters] = labels
    selection = pd.DataFrame.from_records(candidates)
    supported = selection.loc[selection["expost__cluster_support_pass"]]
    if supported.empty:
        raise ValueError(
            "no taxonomy cluster count satisfies the minimum per-cluster support"
        )
    selected = int(
        supported.sort_values(
            ["expost__selection_objective", "expost__cluster_count"],
            ascending=[False, True],
            kind="stable",
        ).iloc[0]["expost__cluster_count"]
    )
    model = fitted[selected]
    labels = labels_by_k[selected]
    centers = np.vstack(
        [values[labels == cluster].mean(axis=0) for cluster in range(selected)]
    )
    label_map: dict[int, str] = {}
    summaries: list[dict[str, Any]] = []
    for cluster, center in enumerate(centers):
        dominant_index = int(np.nanargmax(np.abs(center)))
        dominant = columns[dominant_index]
        direction = "elevated" if center[dominant_index] >= 0 else "depressed"
        semantic = (
            f"{_taxonomy_family(dominant)}__{direction}__c{cluster:02d}"
        )
        label_map[cluster] = semantic
        summaries.append(
            {
                "expost__failure_cluster": cluster,
                "expost__failure_taxonomy_label": semantic,
                "expost__cluster_episode_count": int((labels == cluster).sum()),
                "expost__dominant_profile_feature": dominant,
                "expost__dominant_standardized_effect": float(
                    center[dominant_index]
                ),
            }
        )
    bundle = FrozenFailureTaxonomyBundle(
        method=method,
        profile_columns=columns,
        imputer=imputer,
        scaler=scaler,
        model=model,
        label_map=label_map,
        selected_clusters=selected,
        fit_cutoff_utc=boundary.isoformat(),
        train_episode_ids=train["episode_id"].astype(str).tolist(),
        robust_clip=float(robust_clip),
    )
    assignments = bundle.score(profiles)
    assignments = assignments.merge(
        episodes.loc[
            :, ["episode_id", "episode_end_available_utc"]
        ],
        on="episode_id",
        how="left",
        validate="one_to_one",
    )
    assignments["taxonomy_label_available_utc"] = pd.concat(
        [
            _utc(assignments["episode_end_available_utc"]),
            pd.Series(boundary, index=assignments.index),
        ],
        axis=1,
    ).max(axis=1)
    return (
        bundle,
        assignments,
        selection,
        pd.DataFrame.from_records(summaries),
    )


def evaluate_taxonomy_bootstrap_stability(
    bundle: FrozenFailureTaxonomyBundle,
    profiles: pd.DataFrame,
    *,
    repetitions: int = 100,
    minimum_median_ari: float = 0.80,
    minimum_q10_ari: float = 0.50,
    random_state: int = 20260726,
) -> dict[str, Any]:
    """Measure label-invariant taxonomy reproducibility under resampling."""

    if int(repetitions) < 10:
        raise ValueError("taxonomy stability requires at least 10 repetitions")
    train = profiles.loc[
        profiles["episode_id"].astype(str).isin(bundle.train_episode_ids)
    ].copy()
    if len(train) != len(bundle.train_episode_ids):
        raise ValueError("taxonomy stability is missing frozen train episodes")
    raw = (
        train.loc[:, bundle.profile_columns]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(np.float64)
    )
    values = np.clip(
        bundle.scaler.transform(bundle.imputer.transform(raw)),
        -float(bundle.robust_clip),
        float(bundle.robust_clip),
    )
    reference = np.asarray(bundle.model.predict(values), dtype=np.int16)
    generator = np.random.default_rng(int(random_state))
    rows: list[dict[str, Any]] = []
    for repetition in range(int(repetitions)):
        sampled = generator.integers(0, len(values), size=len(values))
        seed = int(random_state) + repetition + 1
        if bundle.method == "gmm":
            model: Any = GaussianMixture(
                n_components=int(bundle.selected_clusters),
                covariance_type="diag",
                reg_covar=1e-3,
                n_init=5,
                max_iter=300,
                random_state=seed,
            ).fit(values[sampled])
        elif bundle.method == "kmeans":
            model = KMeans(
                n_clusters=int(bundle.selected_clusters),
                n_init=30,
                max_iter=400,
                random_state=seed,
            ).fit(values[sampled])
        else:
            raise ValueError("unsupported frozen taxonomy method")
        predicted = np.asarray(model.predict(values), dtype=np.int16)
        counts = np.bincount(
            predicted, minlength=int(bundle.selected_clusters)
        )
        rows.append(
            {
                "repetition": repetition,
                "adjusted_rand_index": float(
                    adjusted_rand_score(reference, predicted)
                ),
                "minimum_cluster_episodes": int(counts.min()),
            }
        )
    result = pd.DataFrame.from_records(rows)
    median = float(result["adjusted_rand_index"].median())
    q10 = float(result["adjusted_rand_index"].quantile(0.10))
    passed = bool(
        median >= float(minimum_median_ari)
        and q10 >= float(minimum_q10_ari)
    )
    return {
        "status": "PASS" if passed else "UNSTABLE_TAXONOMY",
        "pass": passed,
        "repetitions": int(repetitions),
        "train_episodes": int(len(train)),
        "selected_clusters": int(bundle.selected_clusters),
        "median_adjusted_rand_index": median,
        "q10_adjusted_rand_index": q10,
        "minimum_adjusted_rand_index": float(
            result["adjusted_rand_index"].min()
        ),
        "median_minimum_cluster_episodes": float(
            result["minimum_cluster_episodes"].median()
        ),
        "minimum_median_ari_required": float(minimum_median_ari),
        "minimum_q10_ari_required": float(minimum_q10_ari),
    }


def choose_taxonomy_fit_cutoff(
    episodes: pd.DataFrame,
    *,
    minimum_failure_episodes: int,
) -> pd.Timestamp:
    """Choose the earliest cutoff with the predeclared resolved episode count."""

    _require(episodes, ("episode_end_available_utc",), "episodes")
    values = _utc(episodes["episode_end_available_utc"]).dropna().sort_values()
    if len(values) < int(minimum_failure_episodes):
        raise ValueError("insufficient resolved failure episodes for taxonomy cutoff")
    return pd.Timestamp(values.iloc[int(minimum_failure_episodes) - 1]) + pd.Timedelta(
        "1ns"
    )


def build_hourly_failure_state_targets(
    health: pd.DataFrame,
    episode_membership: pd.DataFrame,
    taxonomy_assignments: pd.DataFrame,
    *,
    health_bin_hours: int = 6,
    horizon_hours: int = 3,
) -> pd.DataFrame:
    """Expand resolved health bins and create availability-explicit targets."""

    _require(
        health,
        (
            "decision_bin_start_utc",
            "bin_available_utc",
            "evaluation_origin",
            "model_failure_bin",
        ),
        "health",
    )
    _require(
        episode_membership,
        ("episode_id", "decision_bin_start_utc", "evaluation_origin"),
        "episode membership",
    )
    _require(
        taxonomy_assignments,
        (
            "episode_id",
            "expost__failure_taxonomy_label",
            "taxonomy_label_available_utc",
        ),
        "taxonomy assignments",
    )
    bin_episode = episode_membership.loc[
        :, ["episode_id", "decision_bin_start_utc", "evaluation_origin"]
    ].drop_duplicates()
    bin_episode = bin_episode.merge(
        taxonomy_assignments.loc[
            :,
            [
                "episode_id",
                "expost__failure_taxonomy_label",
                "taxonomy_label_available_utc",
            ],
        ],
        on="episode_id",
        how="left",
        validate="many_to_one",
    )
    bins = health.merge(
        bin_episode,
        on=["decision_bin_start_utc", "evaluation_origin"],
        how="left",
        validate="one_to_one",
    )
    bins["failure_state"] = np.where(
        bins["model_failure_bin"].fillna(False).astype(bool),
        bins["expost__failure_taxonomy_label"],
        "stable",
    )
    unresolved_failure = (
        bins["model_failure_bin"].fillna(False).astype(bool)
        & bins["failure_state"].isna()
    )
    if unresolved_failure.any():
        raise ValueError("failure bins lack a frozen taxonomy assignment")
    bins["state_available_utc"] = pd.concat(
        [
            _utc(bins["bin_available_utc"]),
            _utc(bins["taxonomy_label_available_utc"]),
        ],
        axis=1,
    ).max(axis=1)
    bins.loc[
        ~bins["model_failure_bin"].fillna(False).astype(bool),
        "state_available_utc",
    ] = _utc(
        bins.loc[
            ~bins["model_failure_bin"].fillna(False).astype(bool),
            "bin_available_utc",
        ]
    )
    rows: list[dict[str, Any]] = []
    for row in bins.itertuples(index=False):
        values = row._asdict()
        for offset in range(int(health_bin_hours)):
            rows.append(
                {
                    "execution_decision_utc": pd.Timestamp(
                        values["decision_bin_start_utc"]
                    )
                    + pd.Timedelta(hours=offset),
                    "side_name": "global",
                    "evaluation_origin": values["evaluation_origin"],
                    "failure_state": values["failure_state"],
                    "state_available_utc": values["state_available_utc"],
                }
            )
    hourly = pd.DataFrame.from_records(rows)
    observed = _utc(hourly["state_available_utc"]).max()
    labels = build_hourly_state_transition_labels(
        hourly,
        state_col="failure_state",
        state_available_col="state_available_utc",
        timestamp_col="execution_decision_utc",
        side_col="side_name",
        boundary_columns=("evaluation_origin",),
        horizon_hours=int(horizon_hours),
        observed_through=observed,
    )
    labels = labels.rename(
        columns={
            "target__current_state": "target__current_failure_state",
            f"target__destination_state_{int(horizon_hours)}h": (
                f"target__destination_state_{int(horizon_hours)}h"
            ),
        }
    )
    availability_columns = [
        "target__current_state_label_resolution_utc",
        "target__active_transition_label_resolution_utc",
        f"target__future_label_resolution_utc",
    ]
    labels["transition_label_available_at"] = pd.concat(
        [_utc(labels[name]) for name in availability_columns], axis=1
    ).max(axis=1)
    return labels


def evaluate_failure_detector_label_sufficiency(
    targets: pd.DataFrame,
    *,
    config: FailureFirstSufficiencyConfig | None = None,
    horizon_hours: int = 3,
) -> dict[str, Any]:
    """Gate all four supervised heads before any chronological fitting."""

    cfg = config or FailureFirstSufficiencyConfig()
    columns = [
        f"target__transition_within_{int(horizon_hours)}h",
        "target__active_transition",
        "target__current_failure_state",
        f"target__destination_state_{int(horizon_hours)}h",
        "transition_label_available_at",
    ]
    _require(targets, columns, "detector targets")
    complete = targets.loc[targets[columns].notna().all(axis=1)].copy()
    transition_rows = int(
        pd.to_numeric(
            complete[f"target__transition_within_{int(horizon_hours)}h"],
            errors="coerce",
        )
        .eq(1.0)
        .sum()
    )
    active_rows = int(
        pd.to_numeric(
            complete["target__active_transition"], errors="coerce"
        )
        .eq(1.0)
        .sum()
    )
    current_counts = (
        complete["target__current_failure_state"]
        .astype(str)
        .value_counts()
        .to_dict()
    )
    destination_counts = (
        complete[f"target__destination_state_{int(horizon_hours)}h"]
        .astype(str)
        .value_counts()
        .to_dict()
    )
    minimum_state = min(current_counts.values()) if current_counts else 0
    minimum_destination = (
        min(destination_counts.values()) if destination_counts else 0
    )
    criteria = {
        "complete_detector_rows": {
            "observed": int(len(complete)),
            "required": int(cfg.minimum_detector_rows),
            "pass": len(complete) >= int(cfg.minimum_detector_rows),
        },
        "transition_positive_rows": {
            "observed": transition_rows,
            "required": int(cfg.minimum_transitions),
            "pass": transition_rows >= int(cfg.minimum_transitions),
        },
        "active_transition_positive_rows": {
            "observed": active_rows,
            "required": int(cfg.minimum_transitions),
            "pass": active_rows >= int(cfg.minimum_transitions),
        },
        "minimum_current_state_rows": {
            "observed": int(minimum_state),
            "required": int(cfg.minimum_transitions),
            "pass": minimum_state >= int(cfg.minimum_transitions),
        },
        "minimum_destination_state_rows": {
            "observed": int(minimum_destination),
            "required": int(cfg.minimum_transitions),
            "pass": minimum_destination >= int(cfg.minimum_transitions),
        },
    }
    passed = all(bool(item["pass"]) for item in criteria.values())
    return {
        "detector_training_allowed": passed,
        "status": "PASS" if passed else "INSUFFICIENT_CLASS_SUPPORT",
        "criteria": criteria,
        "current_state_counts": current_counts,
        "destination_state_counts": destination_counts,
    }


def evaluate_failure_first_sufficiency(
    health: pd.DataFrame,
    episodes: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    profile_feature_count: int,
    config: FailureFirstSufficiencyConfig | None = None,
) -> dict[str, Any]:
    """Return a machine-readable gate; no individual criterion is advisory."""

    cfg = config or FailureFirstSufficiencyConfig()
    complete = (
        windows.groupby("episode_id", observed=True)["window_complete"].all()
        if len(windows)
        else pd.Series(dtype=bool)
    )
    failure_bins = int(health["model_failure_bin"].fillna(False).sum())
    episode_count = int(len(episodes))
    complete_episodes = int(complete.sum())
    if len(health):
        observed_bins = (
            _utc(health["decision_bin_start_utc"])
            .dropna()
            .drop_duplicates()
            .sort_values()
        )
        span_days = float(
            (observed_bins.max() - observed_bins.min())
            / pd.Timedelta(days=1)
        )
        observed_days = int(observed_bins.dt.normalize().nunique())
        bin_gaps = observed_bins.diff().dropna()
        maximum_gap_days = (
            float(bin_gaps.max() / pd.Timedelta(days=1))
            if len(bin_gaps)
            else 0.0
        )
    else:
        span_days = 0.0
        observed_days = 0
        maximum_gap_days = 0.0
    criteria = {
        "failure_episodes": {
            "observed": episode_count,
            "required": int(cfg.minimum_failure_episodes),
            "pass": episode_count >= int(cfg.minimum_failure_episodes),
        },
        "complete_window_episodes": {
            "observed": complete_episodes,
            "required": int(cfg.minimum_complete_window_episodes),
            "pass": complete_episodes
            >= int(cfg.minimum_complete_window_episodes),
        },
        "failure_bins": {
            "observed": failure_bins,
            "required": int(cfg.minimum_failure_bins),
            "pass": failure_bins >= int(cfg.minimum_failure_bins),
        },
        "calendar_span_days": {
            "observed": span_days,
            "required": int(cfg.minimum_span_days),
            "pass": span_days >= int(cfg.minimum_span_days),
        },
        "observed_calendar_days": {
            "observed": observed_days,
            "required": int(cfg.minimum_observed_days),
            "pass": observed_days >= int(cfg.minimum_observed_days),
        },
        "maximum_calendar_gap_days": {
            "observed": maximum_gap_days,
            "required_maximum": int(cfg.maximum_calendar_gap_days),
            "pass": maximum_gap_days
            <= int(cfg.maximum_calendar_gap_days),
        },
        "profile_features": {
            "observed": int(profile_feature_count),
            "required": int(cfg.minimum_profile_features),
            "pass": int(profile_feature_count)
            >= int(cfg.minimum_profile_features),
        },
    }
    passed = all(bool(item["pass"]) for item in criteria.values())
    return {
        "taxonomy_training_allowed": passed,
        "detector_training_allowed": False,
        "status": "PASS" if passed else "INSUFFICIENT_SUPPORT",
        "criteria": criteria,
        "config": asdict(cfg),
        "policy": (
            "Descriptive artifacts are valid. Taxonomy and detector fitting "
            "must be skipped unless every criterion passes."
        ),
    }


def attach_episode_window_coverage(
    episodes: pd.DataFrame,
    windows: pd.DataFrame,
) -> pd.DataFrame:
    """Populate the episode-level complete-window audit field."""

    output = episodes.copy()
    if output.empty:
        return output
    complete = (
        windows.groupby("episode_id", observed=True)["window_complete"]
        .all()
        .rename("complete_window_coverage")
    )
    output = output.drop(columns=["complete_window_coverage"], errors="ignore")
    return output.merge(
        complete, on="episode_id", how="left", validate="one_to_one"
    ).fillna({"complete_window_coverage": False})


def frame_fingerprint(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """Stable content fingerprint for manifests and sealed audit artifacts."""

    usable = [name for name in columns if name in frame.columns]
    if not usable:
        return hashlib.sha256(b"empty").hexdigest()
    values = frame.loc[:, usable].copy()
    for name in usable:
        if pd.api.types.is_datetime64_any_dtype(values[name]):
            values[name] = _utc(values[name]).astype(str)
    hashed = pd.util.hash_pandas_object(values, index=False).to_numpy(np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


__all__ = [
    "DEFAULT_MARKET_FEATURES",
    "DEFAULT_MODEL_HEALTH_FEATURES",
    "FrozenFailureTaxonomyBundle",
    "FailureFirstSufficiencyConfig",
    "attach_episode_window_coverage",
    "build_failure_episode_profiles",
    "build_hourly_failure_state_targets",
    "build_hourly_observable_state",
    "choose_taxonomy_fit_cutoff",
    "evaluate_failure_detector_label_sufficiency",
    "evaluate_failure_first_sufficiency",
    "extract_failure_episode_outcomes",
    "extract_failure_episode_windows",
    "fit_frozen_failure_taxonomy",
    "evaluate_taxonomy_bootstrap_stability",
    "frame_fingerprint",
    "prepare_failure_first_sources",
]
