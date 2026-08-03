"""Leakage-explicit primitives for failure-first hourly regime research.

The functions in this module intentionally stop before model training or
artifact persistence.  They turn a score-ranked candidate ledger into an
*ex-post* failure calendar and episode taxonomy, while keeping the only
potentially reusable state labels in a clearly separate ``target__`` block.
Nothing returned here is implicitly an inference feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler


EXPOST_PREFIX = "expost__"
TARGET_PREFIX = "target__"
FUTURE_PREFIX = "future__"
FORBIDDEN_INFERENCE_PREFIXES = (EXPOST_PREFIX, TARGET_PREFIX, FUTURE_PREFIX)
DEFAULT_WINDOW_OFFSETS_HOURS = (-48, -24, -12, -6, -3, 0, 3, 6, 12)


@dataclass(frozen=True)
class FailureFirstHourlyConfig:
    """Column names and fixed conventions for the hourly research panel."""

    timestamp_col: str = "__ts__"
    candidate_id_col: str = "candidate_id"
    side_col: str = "side_name"
    score_col: str = "score"
    exact_net_ev_col: str = "execution_net_ev_12h"
    label_resolution_col: str = "execution_label_end_utc"
    top_k_fraction: float = 0.10
    failure_threshold: float = 0.0
    episode_gap_hours: int = 0


@dataclass(frozen=True)
class FailureTaxonomyResult:
    """Deterministic, descriptive taxonomy output for failure episodes only."""

    assignments: pd.DataFrame
    cluster_summary: pd.DataFrame
    selection: pd.DataFrame
    method: str
    selected_clusters: int


def _utc(values: pd.Series | Iterable[object]) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"frame lacks required columns: {missing}")


def _candidate_sort_key(values: pd.Series) -> pd.Series:
    """A stable cross-type candidate key; never rely on input row order for ties."""

    if values.isna().any():
        raise ValueError("candidate_id must be populated for deterministic global ties")
    return values.astype(str)


def validate_inference_feature_columns(columns: Iterable[str]) -> tuple[str, ...]:
    """Reject labels, ex-post profiles and future fields from an inference contract."""

    names = tuple(str(column) for column in columns)
    rejected = [
        name
        for name in names
        if name.casefold().startswith(tuple(prefix.casefold() for prefix in FORBIDDEN_INFERENCE_PREFIXES))
    ]
    if rejected:
        raise ValueError(
            "Failure-first inference features cannot contain target/expost/future "
            f"columns: {sorted(rejected)}"
        )
    return names


def select_pooled_global_top_k(
    frame: pd.DataFrame,
    *,
    score_col: str,
    fraction: float = 0.10,
    candidate_id_col: str = "candidate_id",
) -> pd.Series:
    """Return one global top-k mask, pooled across timestamps and sides.

    Selection is deliberately *not* per timestamp, side, asset, or archetype.
    Non-finite scores are ineligible.  Equal scores are broken by ascending
    stable candidate ID, giving reproducible selection regardless of input row
    order.
    """

    _require_columns(frame, (score_col, candidate_id_col))
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    score = pd.to_numeric(frame[score_col], errors="coerce")
    eligible = score.notna() & np.isfinite(score)
    output = pd.Series(False, index=frame.index, dtype=bool)
    if not eligible.any():
        return output
    ranked = pd.DataFrame(
        {
            "__score__": score.loc[eligible],
            "__candidate__": _candidate_sort_key(frame.loc[eligible, candidate_id_col]),
            "__position__": np.flatnonzero(eligible.to_numpy()),
        }
    ).sort_values(["__score__", "__candidate__"], ascending=[False, True], kind="mergesort")
    count = int(np.ceil(float(fraction) * len(ranked)))
    output.iloc[ranked.head(count)["__position__"].to_numpy(dtype=int)] = True
    return output


def build_hourly_failure_calendar(
    frame: pd.DataFrame,
    selected: pd.Series | np.ndarray | None = None,
    *,
    config: FailureFirstHourlyConfig = FailureFirstHourlyConfig(),
) -> pd.DataFrame:
    """Aggregate selected exact-policy EV into side-aware hourly failure labels.

    A calendar label is populated only when *every* selected candidate in the
    hour has a finite exact-policy EV and a finite label-resolution timestamp.
    Its availability timestamp is the maximum constituent resolution.  Thus a
    caller can safely purge or filter labels at any historical boundary.
    """

    required = (
        config.timestamp_col,
        config.side_col,
        config.exact_net_ev_col,
        config.label_resolution_col,
    )
    _require_columns(frame, required)
    work = frame.copy()
    work[config.timestamp_col] = _utc(work[config.timestamp_col])
    work[config.label_resolution_col] = _utc(work[config.label_resolution_col])
    if work[config.timestamp_col].isna().any():
        raise ValueError("hourly failure calendar requires valid UTC decision timestamps")
    if selected is None:
        selected = select_pooled_global_top_k(
            work,
            score_col=config.score_col,
            fraction=config.top_k_fraction,
            candidate_id_col=config.candidate_id_col,
        )
    selected_series = pd.Series(selected, index=work.index).fillna(False).astype(bool)
    work = work.loc[selected_series].copy()
    columns = [
        config.timestamp_col,
        config.side_col,
        "expost__selected_count",
        "expost__resolved_selected_count",
        "expost__selected_net_ev_mean",
        "expost__selected_net_ev_sum",
        "target__failure",
        "target__failure_label_resolution_utc",
    ]
    if work.empty:
        return pd.DataFrame(columns=columns)
    work["hour_utc"] = work[config.timestamp_col].dt.floor("h")
    work["__net_ev__"] = pd.to_numeric(work[config.exact_net_ev_col], errors="coerce")
    work["__resolved__"] = work["__net_ev__"].notna() & work[config.label_resolution_col].notna()
    rows: list[dict[str, object]] = []
    for (hour, side), local in work.groupby(["hour_utc", config.side_col], sort=True, observed=True):
        resolved = local.loc[local["__resolved__"]]
        complete = len(resolved) == len(local)
        resolution = local[config.label_resolution_col].max() if complete else pd.NaT
        mean_ev = float(resolved["__net_ev__"].mean()) if complete else np.nan
        rows.append(
            {
                config.timestamp_col: hour,
                config.side_col: side,
                "expost__selected_count": int(len(local)),
                "expost__resolved_selected_count": int(len(resolved)),
                "expost__selected_net_ev_mean": mean_ev,
                "expost__selected_net_ev_sum": float(resolved["__net_ev__"].sum()) if complete else np.nan,
                "target__failure": float(mean_ev <= float(config.failure_threshold)) if complete else np.nan,
                "target__failure_label_resolution_utc": resolution,
            }
        )
    return pd.DataFrame(rows).sort_values(
        [config.timestamp_col, config.side_col], kind="mergesort"
    ).reset_index(drop=True)


def build_contiguous_failure_episodes(
    calendar: pd.DataFrame,
    *,
    timestamp_col: str = "__ts__",
    side_col: str = "side_name",
    gap_hours: int = 0,
    min_hours: int = 1,
) -> pd.DataFrame:
    """Build side-local contiguous episodes from resolved hourly failure labels."""

    _require_columns(calendar, (timestamp_col, side_col, "target__failure"))
    if gap_hours < 0 or min_hours < 1:
        raise ValueError("gap_hours must be >=0 and min_hours must be >=1")
    work = calendar.copy()
    work[timestamp_col] = _utc(work[timestamp_col])
    active = work.loc[work["target__failure"].eq(1.0) & work[timestamp_col].notna()].copy()
    rows: list[dict[str, object]] = []
    for side, local in active.groupby(side_col, sort=True, observed=True):
        local = local.sort_values(timestamp_col, kind="mergesort")
        starts = local[timestamp_col].diff().gt(pd.Timedelta(hours=int(gap_hours) + 1)).fillna(True)
        local["__episode__"] = starts.cumsum().astype(int)
        for ordinal, block in local.groupby("__episode__", sort=True):
            if len(block) < min_hours:
                continue
            start, end = block[timestamp_col].min(), block[timestamp_col].max()
            net_ev = (
                pd.to_numeric(block["expost__selected_net_ev_mean"], errors="coerce")
                if "expost__selected_net_ev_mean" in block
                else pd.Series(np.nan, index=block.index, dtype=float)
            )
            resolution = (
                _utc(block["target__failure_label_resolution_utc"]).max()
                if "target__failure_label_resolution_utc" in block
                else pd.NaT
            )
            rows.append(
                {
                    side_col: side,
                    "expost__episode_id": f"{side}::{start.isoformat()}::{int(ordinal):04d}",
                    "expost__episode_start_utc": start,
                    "expost__episode_end_utc": end,
                    "expost__episode_hours": int(len(block)),
                    "expost__episode_net_ev_mean": float(net_ev.mean()),
                    "expost__episode_label_resolution_utc": resolution,
                }
            )
    return pd.DataFrame(rows).sort_values(
        [side_col, "expost__episode_start_utc"], kind="mergesort"
    ).reset_index(drop=True) if rows else pd.DataFrame(
        columns=[side_col, "expost__episode_id", "expost__episode_start_utc", "expost__episode_end_utc", "expost__episode_hours", "expost__episode_net_ev_mean", "expost__episode_label_resolution_utc"]
    )


def extract_episode_windows(
    hourly_state: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    offsets_hours: Sequence[int] = DEFAULT_WINDOW_OFFSETS_HOURS,
    timestamp_col: str = "__ts__",
    side_col: str = "side_name",
) -> pd.DataFrame:
    """Extract exact hourly state windows, clearly segregated as ex-post data."""

    validate_inference_feature_columns(feature_columns)
    _require_columns(hourly_state, (timestamp_col, side_col, *feature_columns))
    _require_columns(episodes, (side_col, "expost__episode_id", "expost__episode_start_utc"))
    offsets = tuple(int(offset) for offset in offsets_hours)
    if not offsets:
        raise ValueError("offsets_hours must be non-empty")
    state = hourly_state.loc[:, [timestamp_col, side_col, *feature_columns]].copy()
    state[timestamp_col] = _utc(state[timestamp_col]).dt.floor("h")
    if state.duplicated([timestamp_col, side_col]).any():
        raise ValueError("hourly_state must have one row per side and hour")
    renamed = state.rename(columns={name: f"expost__window__{name}" for name in feature_columns})
    rows: list[dict[str, object]] = []
    for episode in episodes.itertuples(index=False):
        values = episode._asdict()
        start = pd.Timestamp(values["expost__episode_start_utc"])
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        else:
            start = start.tz_convert("UTC")
        for offset in offsets:
            rows.append(
                {
                    side_col: values[side_col],
                    "expost__episode_id": values["expost__episode_id"],
                    "expost__window_offset_hours": offset,
                    "expost__window_target_utc": start + pd.Timedelta(hours=offset),
                }
            )
    targets = pd.DataFrame(rows)
    result = targets.merge(
        renamed,
        left_on=["expost__window_target_utc", side_col],
        right_on=[timestamp_col, side_col],
        how="left",
        validate="many_to_one",
    ).drop(columns=[timestamp_col])
    return result.sort_values(
        ["expost__episode_id", "expost__window_offset_hours"], kind="mergesort"
    ).reset_index(drop=True)


def build_failure_episode_profiles(
    windows: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Pivot only failure-episode windows into a deterministic ex-post profile."""

    prefixed = tuple(f"expost__window__{name}" for name in feature_columns)
    _require_columns(windows, ("expost__episode_id", "expost__window_offset_hours", *prefixed))
    if windows["expost__episode_id"].isna().any():
        raise ValueError("profiles require failure episode IDs")
    output = windows[["expost__episode_id"]].drop_duplicates().set_index("expost__episode_id")
    for column in prefixed:
        pivot = windows.pivot(index="expost__episode_id", columns="expost__window_offset_hours", values=column)
        pivot = pivot.rename(columns=lambda offset: f"expost__profile__{column.removeprefix('expost__window__')}__h{int(offset):+d}")
        output = output.join(pivot, how="left")
    return output.reset_index().sort_values("expost__episode_id", kind="mergesort").reset_index(drop=True)


def _semantic_family(column: str) -> str:
    value = column.casefold()
    if any(token in value for token in ("fund", "funding")):
        return "funding_transition"
    if any(token in value for token in ("oi", "open_interest", "leverage")):
        return "leverage_repricing"
    if any(token in value for token in ("liquid", "spread", "depth", "volume")):
        return "liquidity_dislocation"
    if any(token in value for token in ("breadth", "corr", "correlation", "dispersion")):
        return "correlation_fragmentation"
    if any(token in value for token in ("vol", "atr", "range", "wick")):
        return "volatility_expansion"
    if any(token in value for token in ("ret", "price", "momentum", "trend")):
        return "directional_transition"
    return "mixed_observable_state"


def fit_failure_episode_taxonomy(
    profiles: pd.DataFrame,
    *,
    profile_columns: Sequence[str] | None = None,
    method: Literal["gmm", "kmeans"] = "gmm",
    min_clusters: int = 2,
    max_clusters: int = 8,
    random_state: int = 20260726,
) -> FailureTaxonomyResult:
    """Fit a deterministic 2--8 cluster taxonomy on failure profiles only.

    Cluster choice is BIC for GMM and silhouette for KMeans.  Outputs remain
    ``expost__`` descriptors: they are useful for diagnosis and target design,
    never direct inference columns.
    """

    _require_columns(profiles, ("expost__episode_id",))
    if method not in {"gmm", "kmeans"}:
        raise ValueError("method must be 'gmm' or 'kmeans'")
    if min_clusters < 2 or max_clusters < min_clusters or max_clusters > 8:
        raise ValueError("cluster range must satisfy 2 <= min <= max <= 8")
    columns = tuple(profile_columns) if profile_columns is not None else tuple(
        name for name in profiles.columns if name.startswith("expost__profile__")
    )
    if not columns:
        raise ValueError("failure taxonomy needs ex-post profile columns")
    if any(not name.startswith(EXPOST_PREFIX) for name in columns):
        raise ValueError("failure taxonomy profiles must be explicitly ex-post")
    if len(profiles) < min_clusters:
        raise ValueError("insufficient failure episodes for requested cluster range")
    raw = profiles.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    imputer = SimpleImputer(strategy="median")
    values = RobustScaler(quantile_range=(10.0, 90.0)).fit_transform(imputer.fit_transform(raw))
    candidates = list(range(min_clusters, min(max_clusters, len(values)) + 1))
    if not candidates:
        raise ValueError("no valid cluster count")
    selection_rows: list[dict[str, object]] = []
    fitted: dict[int, object] = {}
    for clusters in candidates:
        if method == "gmm":
            model: object = GaussianMixture(
                n_components=clusters, covariance_type="diag", reg_covar=1e-4,
                n_init=3, max_iter=250, random_state=random_state,
            ).fit(values)
            objective = -float(model.bic(values))
            criterion = "negative_bic"
        else:
            model = KMeans(n_clusters=clusters, n_init=20, max_iter=300, random_state=random_state).fit(values)
            labels = model.labels_
            objective = float(silhouette_score(values, labels)) if clusters < len(values) else -np.inf
            criterion = "silhouette"
        fitted[clusters] = model
        selection_rows.append({"expost__cluster_count": clusters, "expost__selection_objective": objective, "expost__selection_criterion": criterion})
    selection = pd.DataFrame(selection_rows).sort_values(["expost__selection_objective", "expost__cluster_count"], ascending=[False, True], kind="mergesort")
    selected_clusters = int(selection.iloc[0]["expost__cluster_count"])
    model = fitted[selected_clusters]
    if method == "gmm":
        labels = model.predict(values)  # type: ignore[union-attr]
        probability = model.predict_proba(values).max(axis=1)  # type: ignore[union-attr]
    else:
        labels = model.labels_  # type: ignore[union-attr]
        probability = np.ones(len(values), dtype=float)
    standardized_centres = np.vstack([values[labels == cluster].mean(axis=0) for cluster in range(selected_clusters)])
    cluster_rows: list[dict[str, object]] = []
    label_map: dict[int, str] = {}
    for cluster, centre in enumerate(standardized_centres):
        dominant_index = int(np.nanargmax(np.abs(centre)))
        dominant = columns[dominant_index]
        direction = "elevated" if centre[dominant_index] >= 0 else "depressed"
        semantic = f"{_semantic_family(dominant)}__{direction}"
        label_map[cluster] = semantic
        cluster_rows.append(
            {
                "expost__failure_cluster": cluster,
                "expost__failure_taxonomy_label": semantic,
                "expost__cluster_episode_count": int((labels == cluster).sum()),
                "expost__dominant_profile_feature": dominant,
                "expost__dominant_standardized_effect": float(centre[dominant_index]),
            }
        )
    assignments = profiles[["expost__episode_id"]].copy()
    assignments["expost__failure_cluster"] = labels.astype(np.int16)
    assignments["expost__failure_cluster_probability"] = probability.astype(np.float32)
    assignments["expost__failure_taxonomy_label"] = [label_map[int(label)] for label in labels]
    return FailureTaxonomyResult(
        assignments=assignments,
        cluster_summary=pd.DataFrame(cluster_rows).sort_values("expost__failure_cluster").reset_index(drop=True),
        selection=selection.reset_index(drop=True),
        method=method,
        selected_clusters=selected_clusters,
    )


def build_hourly_state_transition_labels(
    hourly_state: pd.DataFrame,
    *,
    state_col: str,
    state_available_col: str | None = None,
    timestamp_col: str = "__ts__",
    side_col: str = "side_name",
    boundary_columns: Sequence[str] = (),
    horizon_hours: int = 3,
    observed_through: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Create side-local hourly state/transition labels with explicit resolution.

    ``target__destination_state_3h`` is the first different observed state in
    the fully observed next three hourly bars; if no change occurs it equals the
    current state.  Missing any intervening bar makes future labels unknown,
    rather than assuming persistence.
    """

    required = [timestamp_col, side_col, state_col, *boundary_columns]
    if state_available_col is not None:
        required.append(state_available_col)
    _require_columns(hourly_state, required)
    if horizon_hours < 1:
        raise ValueError("horizon_hours must be positive")
    columns = [timestamp_col, side_col, *boundary_columns, state_col]
    if state_available_col is not None:
        columns.append(state_available_col)
    work = hourly_state[columns].copy()
    work[timestamp_col] = _utc(work[timestamp_col]).dt.floor("h")
    if state_available_col is not None:
        work[state_available_col] = _utc(work[state_available_col])
    if work[timestamp_col].isna().any() or work[state_col].isna().any():
        raise ValueError("state labels require non-null UTC timestamps and current states")
    group_columns = [side_col, *boundary_columns]
    if work.duplicated([timestamp_col, *group_columns]).any():
        raise ValueError(
            "state labels require one state per scope, boundary, and hour"
        )
    observed = pd.Timestamp(observed_through) if observed_through is not None else work[timestamp_col].max()
    observed = observed.tz_localize("UTC") if observed.tzinfo is None else observed.tz_convert("UTC")
    rows: list[dict[str, object]] = []
    group_key: str | list[str] = group_columns
    if len(group_columns) == 1:
        group_key = group_columns[0]
    for group_values, local in work.groupby(
        group_key, sort=True, observed=True
    ):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group_identity = dict(zip(group_columns, group_values, strict=True))
        local = local.sort_values(timestamp_col, kind="mergesort")
        lookup = dict(zip(local[timestamp_col], local[state_col], strict=True))
        availability_lookup = (
            dict(
                zip(
                    local[timestamp_col],
                    local[state_available_col],
                    strict=True,
                )
            )
            if state_available_col is not None
            else {stamp: stamp for stamp in local[timestamp_col]}
        )
        for stamp, current in local[[timestamp_col, state_col]].itertuples(index=False):
            previous = lookup.get(stamp - pd.Timedelta(hours=1))
            previous_available = availability_lookup.get(
                stamp - pd.Timedelta(hours=1)
            )
            current_available = availability_lookup.get(stamp)
            future_stamps = [stamp + pd.Timedelta(hours=step) for step in range(1, horizon_hours + 1)]
            timestamps_observed = (
                stamp + pd.Timedelta(hours=horizon_hours) <= observed
                and all(item in lookup for item in future_stamps)
            )
            future_available = (
                [availability_lookup[item] for item in future_stamps]
                if timestamps_observed
                else []
            )
            fully_observed = bool(
                timestamps_observed
                and pd.notna(current_available)
                and pd.Timestamp(current_available) <= observed
                and all(
                    pd.notna(value) and pd.Timestamp(value) <= observed
                    for value in future_available
                )
            )
            future_states = (
                [lookup[item] for item in future_stamps]
                if fully_observed
                else []
            )
            active_available = (
                max(current_available, previous_available)
                if previous is not None
                and pd.notna(current_available)
                and pd.notna(previous_available)
                else pd.NaT
            )
            future_resolution = (
                max([current_available, *future_available])
                if fully_observed
                and pd.notna(current_available)
                and all(pd.notna(value) for value in future_available)
                else pd.NaT
            )
            transition = any(value != current for value in future_states) if fully_observed else np.nan
            destination = next((value for value in future_states if value != current), current) if fully_observed else np.nan
            rows.append(
                {
                    timestamp_col: stamp,
                    **group_identity,
                    "target__current_state": current
                    if pd.notna(current_available)
                    and pd.Timestamp(current_available) <= observed
                    else np.nan,
                    "target__current_state_label_resolution_utc": current_available
                    if pd.notna(current_available)
                    and pd.Timestamp(current_available) <= observed
                    else pd.NaT,
                    "target__active_transition": float(current != previous)
                    if previous is not None
                    and pd.notna(active_available)
                    and pd.Timestamp(active_available) <= observed
                    else np.nan,
                    "target__active_transition_label_resolution_utc": active_available,
                    f"target__transition_within_{horizon_hours}h": float(transition) if fully_observed else np.nan,
                    f"target__destination_state_{horizon_hours}h": destination,
                    f"target__future_label_resolution_utc": future_resolution,
                }
            )
    return pd.DataFrame(rows).sort_values(
        [timestamp_col, *group_columns], kind="mergesort"
    ).reset_index(drop=True)
