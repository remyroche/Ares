"""Pooled historical regime-transition research with symmetric labels.

This module intentionally does not reuse the legacy failure-first hourly
targets.  Those targets copy six-hour economic bins onto hourly rows and
define a destination as the first changed hour.  Here, a transition has a
stable origin in ``[-12h, -3h)`` and a settled destination in ``[+6h,+12h)``.

The fitted state geometry and all future windows are research-label
construction only.  Model inputs are causal values available at the stated
decision time.  Pooled fitting is permitted by the research contract, but the
result is not walk-forward promotion evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler


EXACT_EVENT_OFFSETS = (-48, -24, -12, -6, -3, 0, 3, 6, 12)
EXISTING_TRANSITION_PREFIX = "mkt_regime_change__"
TARGET_PREFIXES = ("target__", "expost__")


@dataclass(frozen=True)
class TransitionResearchConfig:
    """Label geometry for the pooled historical research dataset."""

    minimum_feature_coverage: float = 0.80
    minimum_state_share: float = 0.01
    minimum_state_clusters: int = 5
    maximum_state_clusters: int = 8
    origin_start_hours: int = -12
    origin_end_hours: int = -3
    destination_start_hours: int = 6
    destination_end_hours: int = 12
    minimum_origin_dominance: float = 2.0 / 3.0
    minimum_destination_dominance: float = 2.0 / 3.0
    event_separation_hours: int = 12
    active_cap_hours: int = 6
    random_state: int = 1729


@dataclass
class StateGeometry:
    """Outcome-free pooled market-state geometry used to create targets."""

    feature_columns: tuple[str, ...]
    imputer: SimpleImputer
    scaler: RobustScaler
    cluster: MiniBatchKMeans
    selection: pd.DataFrame


def _utc(values: pd.Series | pd.Index) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True, errors="coerce"))


def _stable_id(prefix: str, *values: object) -> str:
    payload = "|".join(str(value) for value in values)
    return f"{prefix}_{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def _numeric_columns(frame: pd.DataFrame, excluded: Iterable[str]) -> list[str]:
    blocked = set(excluded)
    return [
        name
        for name in frame.columns
        if name not in blocked
        and not name.startswith(TARGET_PREFIXES)
        and pd.api.types.is_numeric_dtype(frame[name])
    ]


def load_compact_market_panel(
    path: Path,
    *,
    start: str | pd.Timestamp | None = "2023-01-01T00:00:00Z",
    end: str | pd.Timestamp | None = None,
    minimum_feature_coverage: float = 0.80,
) -> pd.DataFrame:
    """Load the canonical one-row-per-hour market transition store.

    The compact parquet stores ``ts`` as its index.  A bar timestamp ``ts`` is
    observable for a decision at ``ts + 1h``.  Internal gaps start a new
    segment; no lag or label is allowed to bridge them.
    """

    panel = pd.read_parquet(path)
    if panel.index.name != "ts":
        if "ts" not in panel:
            raise ValueError("compact transition store requires a ts index/column")
        panel = panel.set_index("ts")
    panel.index = _utc(panel.index)
    panel = panel.loc[panel.index.notna()].sort_index(kind="stable")
    if panel.index.duplicated().any():
        raise ValueError("compact transition store requires one row per hour")
    if start is not None:
        panel = panel.loc[panel.index >= pd.Timestamp(start)]
    if end is not None:
        panel = panel.loc[panel.index < pd.Timestamp(end)]
    panel = panel.drop(columns=["__symbol__"], errors="ignore")
    numeric = panel.apply(pd.to_numeric, errors="coerce")
    coverage = numeric.notna().mean()
    retained = coverage.loc[coverage >= float(minimum_feature_coverage)].index
    panel = numeric.loc[:, retained].astype(np.float32)
    panel.insert(0, "source_utc", panel.index)
    panel.insert(1, "execution_decision_utc", panel.index + pd.Timedelta(hours=1))
    gap = panel["source_utc"].diff().ne(pd.Timedelta(hours=1))
    panel.insert(2, "segment_id", gap.cumsum().astype(np.int32).to_numpy())
    panel = panel.reset_index(drop=True)
    return panel


def add_causal_transition_features(
    panel: pd.DataFrame,
    *,
    stems: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Add exact-gap causal 3/6/12/24h velocity and shift features."""

    result = panel.copy()
    excluded = {"source_utc", "execution_decision_utc", "segment_id"}
    candidates = _numeric_columns(result, excluded)
    if stems is None:
        preferred_tokens = (
            "breadth",
            "correlation",
            "funding",
            "oi_",
            "compression",
            "breakout",
            "short_cover",
            "flush",
            "btc_resilience",
            "deleverag",
            "recovery",
        )
        stems = [
            name
            for name in candidates
            if not name.startswith(EXISTING_TRANSITION_PREFIX)
            and any(token in name for token in preferred_tokens)
        ][:24]
    generated: list[str] = []
    grouped = result.groupby("segment_id", observed=True, sort=False)
    for name in stems:
        values = pd.to_numeric(result[name], errors="coerce")
        for hours in (3, 6, 12, 24):
            lagged = grouped[name].shift(hours)
            column = f"transition_new__{name}__delta_{hours}h"
            result[column] = (values - lagged).astype(np.float32)
            generated.append(column)
        short = grouped[name].rolling(3, min_periods=3).mean().reset_index(
            level=0, drop=True
        )
        long = grouped[name].rolling(12, min_periods=12).mean().reset_index(
            level=0, drop=True
        )
        scale = grouped[name].rolling(24, min_periods=12).std().reset_index(
            level=0, drop=True
        )
        column = f"transition_new__{name}__short_long_shift_z"
        result[column] = ((short - long) / (scale + 1e-6)).clip(-8, 8).astype(
            np.float32
        )
        generated.append(column)
    existing = [
        name for name in candidates if name.startswith(EXISTING_TRANSITION_PREFIX)
    ]
    return result, tuple(existing + generated)


def fit_pooled_state_geometry(
    panel: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    config: TransitionResearchConfig = TransitionResearchConfig(),
) -> StateGeometry:
    """Fit an outcome-free pooled state geometry and choose 5–8 states."""

    columns = tuple(feature_columns)
    raw = panel.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    imputer = SimpleImputer(strategy="median")
    values = imputer.fit_transform(raw)
    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    values = scaler.fit_transform(values)
    rng = np.random.default_rng(config.random_state)
    sample_index = np.arange(len(values))
    if len(values) > 12_000:
        sample_index = np.sort(rng.choice(len(values), 12_000, replace=False))
    rows: list[dict[str, object]] = []
    fitted: dict[int, MiniBatchKMeans] = {}
    for clusters in range(
        int(config.minimum_state_clusters),
        int(config.maximum_state_clusters) + 1,
    ):
        model = MiniBatchKMeans(
            n_clusters=clusters,
            random_state=config.random_state,
            batch_size=min(4096, len(values)),
            n_init=10,
            max_iter=300,
        ).fit(values)
        labels = model.labels_
        shares = np.bincount(labels, minlength=clusters) / len(labels)
        score = float(silhouette_score(values[sample_index], labels[sample_index]))
        supported = bool(shares.min() >= float(config.minimum_state_share))
        rows.append(
            {
                "clusters": clusters,
                "silhouette": score,
                "minimum_state_share": float(shares.min()),
                "support_pass": supported,
            }
        )
        fitted[clusters] = model
    selection = pd.DataFrame(rows)
    eligible = selection.loc[selection["support_pass"]]
    chosen_row = (
        eligible.sort_values(["silhouette", "clusters"], ascending=[False, True]).iloc[0]
        if len(eligible)
        else selection.sort_values(
            ["minimum_state_share", "silhouette"], ascending=[False, False]
        ).iloc[0]
    )
    chosen = int(chosen_row["clusters"])
    return StateGeometry(
        feature_columns=columns,
        imputer=imputer,
        scaler=scaler,
        cluster=fitted[chosen],
        selection=selection.assign(selected=lambda x: x["clusters"].eq(chosen)),
    )


def attach_pooled_states(
    panel: pd.DataFrame,
    geometry: StateGeometry,
) -> pd.DataFrame:
    """Attach target-construction state and causal state-context features."""

    result = panel.copy()
    raw = result.loc[:, geometry.feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    values = geometry.scaler.transform(geometry.imputer.transform(raw))
    # ``SimpleImputer``/``RobustScaler`` may promote a float32 compact panel
    # to float64 while a frozen MiniBatchKMeans was fitted on float32.  Recent
    # sklearn releases reject that mixed dtype in ``predict``.  Cast only to
    # the already-fitted model's native representation; no transform is refit.
    values = values.astype(geometry.cluster.cluster_centers_.dtype, copy=False)
    distance = geometry.cluster.transform(values)
    state = geometry.cluster.predict(values).astype(np.int16)
    result["target__pooled_state"] = state
    # The current outcome-free pooled state is decision-time observable under
    # this research geometry.  Keep a separately named causal copy so model
    # ablations need not consume a target-prefixed field.
    result["state_context__current_state"] = state.astype(np.float32)
    result["state_context__nearest_distance"] = distance.min(axis=1).astype(
        np.float32
    )
    result["state_context__top2_margin"] = (
        np.partition(distance, kth=1, axis=1)[:, 1]
        - np.partition(distance, kth=1, axis=1)[:, 0]
    ).astype(np.float32)
    grouped = result.groupby("segment_id", observed=True, sort=False)
    prior = grouped["target__pooled_state"].shift(1)
    switch = result["target__pooled_state"].ne(prior).astype(np.float32)
    switch.loc[prior.isna()] = np.nan
    result["state_context__switch_1h"] = switch
    result["state_context__switch_count_6h"] = (
        switch.groupby(result["segment_id"], observed=True)
        .rolling(6, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
        .astype(np.float32)
    )
    run = (
        result["segment_id"].ne(result["segment_id"].shift())
        | result["target__pooled_state"].ne(
            result["target__pooled_state"].shift()
        )
    ).cumsum()
    result["state_context__state_age_hours"] = (
        result.groupby(run, observed=True).cumcount().clip(upper=168).astype(np.float32)
    )
    return result


def _mode_and_share(values: np.ndarray) -> tuple[int, float]:
    labels, counts = np.unique(values.astype(np.int16), return_counts=True)
    position = int(np.argmax(counts))
    return int(labels[position]), float(counts[position] / counts.sum())


def discover_stabilized_transition_events(
    panel: pd.DataFrame,
    *,
    config: TransitionResearchConfig = TransitionResearchConfig(),
) -> pd.DataFrame:
    """Discover true origin→destination events from symmetric state windows."""

    required = {
        "source_utc",
        "execution_decision_utc",
        "segment_id",
        "target__pooled_state",
    }
    missing = required.difference(panel.columns)
    if missing:
        raise KeyError(f"transition panel missing {sorted(missing)}")
    feature_columns = [
        name
        for name in panel.columns
        if not name.startswith(TARGET_PREFIXES)
        and name
        not in {
            "source_utc",
            "execution_decision_utc",
            "segment_id",
        }
        and pd.api.types.is_numeric_dtype(panel[name])
    ]
    scale = (
        panel[feature_columns].quantile(0.75)
        - panel[feature_columns].quantile(0.25)
    ).replace(0.0, np.nan)
    records: list[dict[str, object]] = []
    for segment, local in panel.groupby("segment_id", sort=True, observed=True):
        local = local.sort_values("source_utc", kind="stable").reset_index()
        states = local["target__pooled_state"].to_numpy(np.int16)
        last_anchor = -10_000
        for index in range(12, len(local) - 12):
            origin_values = states[index - 12 : index - 3]
            destination_values = states[index + 6 : index + 12]
            origin, origin_share = _mode_and_share(origin_values)
            destination, destination_share = _mode_and_share(destination_values)
            if (
                origin == destination
                or origin_share < config.minimum_origin_dominance
                or destination_share < config.minimum_destination_dominance
                or index - last_anchor < int(config.event_separation_hours)
            ):
                continue
            # The anchor must be a genuine departure from the stable origin,
            # not an arbitrary midpoint selected by the future window.
            if states[index - 1] != origin or states[index] == origin:
                continue
            if destination not in states[index : index + 6]:
                continue
            transition_end = min(index + int(config.active_cap_hours), len(local))
            for candidate in range(index, transition_end - 2):
                if np.all(states[candidate : candidate + 3] == destination):
                    transition_end = candidate + 3
                    break
            origin_mean = local.loc[index - 12 : index - 4, feature_columns].mean()
            destination_mean = local.loc[index + 6 : index + 11, feature_columns].mean()
            shift = ((destination_mean - origin_mean).abs() / (scale + 1e-6))
            shift_score = float(shift.replace([np.inf, -np.inf], np.nan).median())
            anchor = pd.Timestamp(local.loc[index, "source_utc"])
            event_id = _stable_id("transition", segment, anchor, origin, destination)
            records.append(
                {
                    "event_id": event_id,
                    "segment_id": int(segment),
                    "anchor_source_utc": anchor,
                    "anchor_decision_utc": anchor + pd.Timedelta(hours=1),
                    "transition_start_utc": anchor,
                    "transition_end_utc": pd.Timestamp(
                        local.loc[transition_end - 1, "source_utc"]
                    )
                    + pd.Timedelta(hours=1),
                    "target_available_utc": anchor + pd.Timedelta(hours=13),
                    "source_state": origin,
                    "destination_state": destination,
                    "transition_archetype": f"state_{origin}_to_state_{destination}",
                    "origin_dominance": origin_share,
                    "destination_dominance": destination_share,
                    "robust_pre_post_shift": shift_score,
                    "label_contract": (
                        "origin[-12h,-3h); destination[+6h,+12h); "
                        "one-hour source-to-decision delay"
                    ),
                }
            )
            last_anchor = index
    return pd.DataFrame.from_records(records)


def materialize_transition_labels(
    panel: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Attach mutually explicit lead, active, phase and destination targets."""

    result = panel.copy()
    timestamp = pd.DatetimeIndex(result["source_utc"])
    n = len(result)
    result["target__event_id"] = pd.Series([None] * n, dtype="object")
    result["target__phase"] = "stable"
    for horizon in (1, 3, 6, 12):
        result[f"target__onset_within_{horizon}h"] = np.int8(0)
    result["target__transition_active"] = np.int8(0)
    result["target__destination_state"] = pd.Series(
        np.full(n, np.nan), dtype="float32"
    )
    result["target__transition_archetype"] = pd.Series([None] * n, dtype="object")
    result["target__time_to_onset_hours"] = np.nan
    result["target__available_utc"] = pd.Series(
        pd.NaT, index=result.index, dtype="datetime64[ns, UTC]"
    )
    for event in events.itertuples(index=False):
        anchor = pd.Timestamp(event.anchor_source_utc)
        end = pd.Timestamp(event.transition_end_utc)
        segment = result["segment_id"].eq(int(event.segment_id)).to_numpy()
        relative = (timestamp - anchor) / pd.Timedelta(hours=1)
        owned = segment & (relative >= -12) & (relative < 12)
        # Events are separated by at least 12h.  In the rare overlap, retain
        # the closest anchor so all correlated rows share one event group.
        existing = result["target__event_id"].notna().to_numpy()
        replace = owned & (
            ~existing
            | (
                np.abs(relative)
                < np.abs(
                    pd.to_numeric(
                        result["target__time_to_onset_hours"], errors="coerce"
                    ).fillna(np.inf)
                )
            )
        )
        result.loc[replace, "target__event_id"] = event.event_id
        result.loc[replace, "target__time_to_onset_hours"] = relative[replace]
        result.loc[replace, "target__available_utc"] = event.target_available_utc
        for horizon in (1, 3, 6, 12):
            lead_horizon = segment & (relative >= -horizon) & (relative < 0)
            result.loc[
                lead_horizon, f"target__onset_within_{horizon}h"
            ] = 1
        lead = segment & (relative >= -3) & (relative < 0)
        active = segment & (timestamp >= anchor) & (timestamp < end)
        destination_rows = lead | active
        result.loc[active, "target__transition_active"] = 1
        result.loc[destination_rows, "target__destination_state"] = float(
            event.destination_state
        )
        result.loc[destination_rows, "target__transition_archetype"] = (
            event.transition_archetype
        )
        phase_masks = (
            ("approach", segment & (relative >= -12) & (relative < -6)),
            ("acceleration", segment & (relative >= -6) & (relative < -3)),
            ("immediate_lead", lead),
            ("transition", active),
            (
                "early_destination",
                segment & (timestamp >= end) & (relative < 6),
            ),
            ("settled_destination", segment & (relative >= 6) & (relative < 12)),
        )
        for phase, mask in phase_masks:
            result.loc[mask, "target__phase"] = phase
    for horizon in (1, 3, 6, 12):
        column = f"target__onset_within_{horizon}h"
        result[column] = result[column].astype(np.int8)
    result["target__transition_active"] = result[
        "target__transition_active"
    ].astype(np.int8)
    return result


def materialize_event_snapshots(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    *,
    offsets: Sequence[int] = EXACT_EVENT_OFFSETS,
) -> pd.DataFrame:
    """Materialize the requested exact before/after event-study grid."""

    lookup = panel.set_index(["segment_id", "source_utc"])
    rows: list[pd.DataFrame] = []
    for event in events.itertuples(index=False):
        stamps = [
            pd.Timestamp(event.anchor_source_utc) + pd.Timedelta(hours=int(offset))
            for offset in offsets
        ]
        keys = pd.MultiIndex.from_arrays(
            [
                np.full(len(stamps), int(event.segment_id)),
                pd.DatetimeIndex(stamps),
            ],
            names=["segment_id", "source_utc"],
        )
        local = lookup.reindex(keys).reset_index()
        local.insert(0, "event_id", event.event_id)
        local.insert(1, "anchor_source_utc", event.anchor_source_utc)
        local.insert(2, "offset_hours", list(offsets))
        rows.append(local)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
