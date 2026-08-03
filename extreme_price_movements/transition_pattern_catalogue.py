"""Causal-input / ex-post-label contracts for transition-pattern research.

This module deliberately keeps the two sides of a regime-transition catalogue
separate:

* ``causal_predictor_columns`` and ``summarize_event_preonset_sequences`` use
  values known strictly before an event anchor; and
* ``materialize_adaptive_transition_phases`` creates *labels* whose
  availability explicitly reflects the future state observations needed to
  establish a transition, confirmation, failure, or reversal.

The labels are for training and evaluation only.  In particular,
``target__pattern_phase`` is never an eligible causal predictor.  Downstream
OOF models must fit any embedding, GMM, rule list, or classifier only on the
training fold's pre-onset summaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


HOUR = pd.Timedelta(hours=1)

PHASES: tuple[str, ...] = (
    "stable_origin",
    "precondition",
    "approach",
    "acceleration",
    "trigger",
    "active_dislocation",
    "confirmation",
    "settled",
    "failed_transition",
    "reversal",
    "stable_destination",
)
TRANSITION_PHASES = frozenset(
    {
        "precondition",
        "approach",
        "acceleration",
        "trigger",
        "active_dislocation",
        "confirmation",
        "failed_transition",
        "reversal",
    }
)
_EVENT_REQUIRED = {
    "event_id",
    "segment_id",
    "anchor_source_utc",
    "transition_end_utc",
    "source_state",
    "destination_state",
}
_PANEL_REQUIRED = {
    "source_utc",
    "execution_decision_utc",
    "segment_id",
    "target__pooled_state",
}
_FORBIDDEN_PREDICTOR_PREFIXES = ("target__", "expost__", "label__")
_FORBIDDEN_PREDICTOR_TOKENS = (
    "phase",
    "event_id",
    "available_utc",
    "future",
    "outcome",
    "realized",
    "mfe",
    "mae",
    "timeout",
    "exit_",
)
_IDENTITY_COLUMNS = {
    "source_utc",
    "execution_decision_utc",
    "segment_id",
    "calendar_segment_id",
    "source_segment_id",
}


@dataclass(frozen=True)
class TransitionPatternConfig:
    """Fixed label and pre-onset sequence windows for research catalogue v1."""

    precondition_hours: int = 168
    approach_hours: int = 24
    acceleration_hours: int = 6
    trigger_hours: int = 3
    confirmation_hours: int = 6
    settled_hours: int = 24
    reversal_search_hours: int = 72
    reversal_label_hours: int = 6
    stable_persistence_hours: int = 12
    sequence_horizons_hours: tuple[int, ...] = (1, 3, 6, 12, 24, 72, 168)

    def __post_init__(self) -> None:
        if not (
            self.precondition_hours > self.approach_hours > self.acceleration_hours > self.trigger_hours > 0
        ):
            raise ValueError("phase horizons must satisfy precondition > approach > acceleration > trigger > 0")
        if min(self.confirmation_hours, self.settled_hours, self.reversal_search_hours, self.reversal_label_hours, self.stable_persistence_hours) <= 0:
            raise ValueError("all pattern horizons must be positive")
        if self.settled_hours <= self.confirmation_hours:
            raise ValueError("settled_hours must exceed confirmation_hours")
        if not self.sequence_horizons_hours or min(self.sequence_horizons_hours) <= 0:
            raise ValueError("sequence horizons must be non-empty positive integers")


def _utc_series(values: pd.Series, name: str) -> pd.Series:
    result = pd.to_datetime(values, utc=True, errors="coerce")
    if result.isna().any():
        raise ValueError(f"{name} has invalid UTC values")
    return result


def _prepare_panel(panel: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(_PANEL_REQUIRED.difference(panel.columns))
    if missing:
        raise KeyError(f"transition pattern panel lacks {missing}")
    result = panel.copy()
    result["source_utc"] = _utc_series(result["source_utc"], "source_utc")
    result["execution_decision_utc"] = _utc_series(
        result["execution_decision_utc"], "execution_decision_utc"
    )
    if not result["execution_decision_utc"].eq(result["source_utc"] + HOUR).all():
        raise ValueError("execution_decision_utc must equal source_utc + one hour")
    result["segment_id"] = pd.to_numeric(result["segment_id"], errors="raise").astype("int64")
    result = result.sort_values(["segment_id", "source_utc"], kind="stable").reset_index(drop=True)
    if result.duplicated(["segment_id", "source_utc"]).any():
        raise ValueError("transition pattern panel has duplicate segment/timestamp rows")
    return result


def _prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(_EVENT_REQUIRED.difference(events.columns))
    if missing:
        raise KeyError(f"transition pattern events lack {missing}")
    result = events.copy()
    result["segment_id"] = pd.to_numeric(result["segment_id"], errors="raise").astype("int64")
    for name in ("anchor_source_utc", "transition_end_utc"):
        result[name] = _utc_series(result[name], name)
    if "target_available_utc" in result:
        result["target_available_utc"] = _utc_series(result["target_available_utc"], "target_available_utc")
    else:
        # The original symmetric event definition needs a settled future state
        # through +12h, observed one hour later.
        result["target_available_utc"] = result["anchor_source_utc"] + pd.Timedelta(hours=13)
    if result["event_id"].duplicated().any():
        raise ValueError("transition pattern events have duplicate event_id values")
    if result["transition_end_utc"].le(result["anchor_source_utc"]).any():
        raise ValueError("transition event end must be after anchor")
    return result.sort_values(["segment_id", "anchor_source_utc", "event_id"], kind="stable").reset_index(drop=True)


def causal_predictor_columns(frame: pd.DataFrame) -> list[str]:
    """Return numeric decision-time columns safe for a pattern model.

    The check is intentionally conservative: all target/ex-post/label fields,
    event/phase identity, availability fields, and common forward-path outcome
    names are rejected.  A caller that passes a manual feature list receives
    the same fail-closed validation via ``validate_causal_predictor_columns``.
    """

    eligible: list[str] = []
    for name in frame.columns:
        lowered = str(name).lower()
        if name in _IDENTITY_COLUMNS or str(name).startswith(_FORBIDDEN_PREDICTOR_PREFIXES):
            continue
        if any(token in lowered for token in _FORBIDDEN_PREDICTOR_TOKENS):
            continue
        if pd.api.types.is_numeric_dtype(frame[name]):
            eligible.append(str(name))
    return eligible


def validate_causal_predictor_columns(frame: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    """Fail closed unless every requested field is eligible causal input."""

    requested = [str(name) for name in columns]
    unknown = sorted(set(requested).difference(frame.columns))
    if unknown:
        raise KeyError(f"unknown transition pattern feature(s): {unknown}")
    eligible = set(causal_predictor_columns(frame))
    forbidden = [name for name in requested if name not in eligible]
    if forbidden:
        raise ValueError(f"non-causal transition pattern feature(s): {forbidden}")
    return requested


def _continuous_state_age(panel: pd.DataFrame) -> pd.Series:
    """State persistence in hours, reset at either a segment or timestamp gap."""

    source = panel["source_utc"]
    state = pd.to_numeric(panel["target__pooled_state"], errors="coerce")
    new_run = (
        panel["segment_id"].ne(panel["segment_id"].shift())
        | source.diff().ne(HOUR)
        | state.ne(state.shift())
        | state.isna()
    )
    run = new_run.cumsum()
    age = panel.groupby(run, observed=True).cumcount().astype("int32") + 1
    return age.where(state.notna(), 0).astype("int32")


def _event_phase_plan(
    panel: pd.DataFrame,
    event: pd.Series,
    config: TransitionPatternConfig,
) -> tuple[list[tuple[str, pd.Timestamp, pd.Timestamp, pd.Timestamp]], dict[str, object]]:
    """Return event-local phase intervals and non-input diagnostic metadata."""

    segment = int(event["segment_id"])
    local = panel.loc[panel["segment_id"].eq(segment), ["source_utc", "target__pooled_state"]].copy()
    anchor = pd.Timestamp(event["anchor_source_utc"])
    end = pd.Timestamp(event["transition_end_utc"])
    target_available = pd.Timestamp(event["target_available_utc"])
    destination = int(event["destination_state"])
    source_state = int(event["source_state"])

    def exact_states(start: pd.Timestamp, stop: pd.Timestamp) -> pd.Series | None:
        expected = pd.date_range(start, stop - HOUR, freq="h", tz="UTC")
        if not len(expected):
            return pd.Series(dtype=float)
        indexed = local.set_index("source_utc")["target__pooled_state"]
        values = indexed.reindex(expected)
        return None if values.isna().any() else values

    confirmation_end = end + pd.Timedelta(hours=config.confirmation_hours)
    confirmation_state = exact_states(end, confirmation_end)
    destination_confirmed = bool(
        confirmation_state is not None
        and pd.to_numeric(confirmation_state, errors="coerce").eq(destination).all()
    )
    plan: list[tuple[str, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []

    def add(phase: str, start: pd.Timestamp, stop: pd.Timestamp, resolved: pd.Timestamp) -> None:
        if stop > start:
            # A phase target is unavailable until the event and the phase's
            # required future evidence are both known.  This is intentionally
            # conservative for all pre-onset rows.
            plan.append((phase, start, stop, max(target_available, resolved + HOUR)))

    add("precondition", anchor - pd.Timedelta(hours=config.precondition_hours), anchor - pd.Timedelta(hours=config.approach_hours), anchor)
    add("approach", anchor - pd.Timedelta(hours=config.approach_hours), anchor - pd.Timedelta(hours=config.acceleration_hours), anchor)
    add("acceleration", anchor - pd.Timedelta(hours=config.acceleration_hours), anchor - pd.Timedelta(hours=config.trigger_hours), anchor)
    add("trigger", anchor - pd.Timedelta(hours=config.trigger_hours), anchor, anchor)
    add("active_dislocation", anchor, end, end)

    metadata: dict[str, object] = {
        "event_id": str(event["event_id"]),
        "destination_confirmed": destination_confirmed,
        "reversal_detected": False,
        "reversal_source_utc": pd.NaT,
    }
    if not destination_confirmed:
        add("failed_transition", end, confirmation_end, confirmation_end)
        return plan, metadata

    add("confirmation", end, confirmation_end, confirmation_end)
    settled_end = end + pd.Timedelta(hours=config.settled_hours)
    add("settled", confirmation_end, settled_end, settled_end)

    reversal_stop = end + pd.Timedelta(hours=config.reversal_search_hours)
    post = exact_states(confirmation_end, reversal_stop)
    if post is not None:
        reversal = post.loc[pd.to_numeric(post, errors="coerce").eq(source_state)]
        if len(reversal):
            reversal_start = pd.Timestamp(reversal.index[0])
            reversal_end = min(
                reversal_start + pd.Timedelta(hours=config.reversal_label_hours),
                reversal_stop,
            )
            add("reversal", reversal_start, reversal_end, reversal_end)
            metadata["reversal_detected"] = True
            metadata["reversal_source_utc"] = reversal_start
    return plan, metadata


def materialize_adaptive_transition_phases(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    *,
    config: TransitionPatternConfig = TransitionPatternConfig(),
) -> pd.DataFrame:
    """Attach adaptive ex-post phases with explicit label availability.

    ``target__pattern_phase`` is mutually exclusive.  Overlapping event
    windows are resolved deterministically to the closest anchor (then event
    id); an output row is therefore never silently assigned two phase labels.
    Rows in a state run shorter than ``stable_persistence_hours`` are left
    unavailable rather than falsely called stable.
    """

    work = _prepare_panel(panel)
    event_frame = _prepare_events(events)
    key_to_row = {
        (int(row.segment_id), pd.Timestamp(row.source_utc)): int(index)
        for index, row in work[["segment_id", "source_utc"]].iterrows()
    }
    phase = pd.Series(pd.NA, index=work.index, dtype="string")
    phase_event = pd.Series(pd.NA, index=work.index, dtype="string")
    available = pd.Series(pd.NaT, index=work.index, dtype="datetime64[ns, UTC]")
    rank: dict[int, tuple[float, int, str]] = {}
    # Reversal is an observed late-path override of the otherwise generic
    # settled interval.  The other intervals do not overlap within one event.
    phase_precedence = {"reversal": 0, "failed_transition": 0}
    event_metadata: list[dict[str, object]] = []

    for _, event in event_frame.iterrows():
        plan, metadata = _event_phase_plan(work, event, config)
        event_metadata.append(metadata)
        anchor = pd.Timestamp(event["anchor_source_utc"])
        segment = int(event["segment_id"])
        for phase_name, start, stop, phase_available in plan:
            timestamps = pd.date_range(start, stop - HOUR, freq="h", tz="UTC")
            for stamp in timestamps:
                row = key_to_row.get((segment, stamp))
                if row is None:
                    continue  # never bridge a source gap or a different segment
                candidate_rank = (
                    abs((stamp - anchor) / HOUR),
                    phase_precedence.get(phase_name, 1),
                    str(event["event_id"]),
                )
                if row in rank and rank[row] <= candidate_rank:
                    continue
                rank[row] = candidate_rank
                phase.iloc[row] = phase_name
                phase_event.iloc[row] = str(event["event_id"])
                available.iloc[row] = phase_available

    state = pd.to_numeric(work["target__pooled_state"], errors="coerce")
    state_age = _continuous_state_age(work)
    stable = phase.isna() & state.notna() & state_age.ge(config.stable_persistence_hours)

    # A stable destination is an already-settled successful destination state.
    # All other long-enough quiet state runs are stable origins/controls.  The
    # distinction is a label, never a causal input field.
    destination_stable = pd.Series(False, index=work.index)
    for event, metadata in zip(event_frame.itertuples(index=False), event_metadata):
        if not bool(metadata["destination_confirmed"]):
            continue
        floor = pd.Timestamp(event.transition_end_utc) + pd.Timedelta(hours=config.settled_hours)
        mask = (
            work["segment_id"].eq(int(event.segment_id))
            & work["source_utc"].ge(floor)
            & state.eq(int(event.destination_state))
            & state_age.ge(config.stable_persistence_hours)
        )
        destination_stable |= mask
    phase.loc[stable & destination_stable] = "stable_destination"
    phase.loc[stable & ~destination_stable] = "stable_origin"
    available.loc[stable] = work.loc[stable, "execution_decision_utc"]

    result = work.copy()
    result["target__pattern_phase"] = phase
    result["target__pattern_phase_available_utc"] = available
    result["target__pattern_event_id"] = phase_event
    result["target__pattern_transition_context"] = phase.isin(TRANSITION_PHASES).astype("float32").where(phase.notna())
    result["target__pattern_transition_context_available_utc"] = available.where(phase.notna())
    result["target__pattern_stable_eligible"] = stable.astype("int8")
    result["state_context__pattern_state_age_hours"] = state_age.astype("int32")
    # Keep attrs serializable.  A DataFrame attr breaks ordinary pandas concat
    # because attrs equality then becomes element-wise rather than scalar.
    result.attrs["transition_pattern_event_metadata"] = event_metadata
    return result


def summarize_event_preonset_sequences(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    *,
    feature_columns: Sequence[str] | None = None,
    config: TransitionPatternConfig = TransitionPatternConfig(),
) -> pd.DataFrame:
    """Summarize exact continuous pre-onset paths for later train-only clustering.

    Each sequence is ``[anchor-H, anchor)``.  Missing timestamps, including a
    gap inside an existing segment, fail that feature/horizon closed to NaN and
    set ``sequence__complete_<H>h`` to zero.  No +offset/post-entry field is
    read by this function.
    """

    work = _prepare_panel(panel)
    event_frame = _prepare_events(events)
    features = (
        causal_predictor_columns(work)
        if feature_columns is None
        else validate_causal_predictor_columns(work, feature_columns)
    )
    records: list[dict[str, object]] = []
    for event in event_frame.itertuples(index=False):
        anchor = pd.Timestamp(event.anchor_source_utc)
        segment = int(event.segment_id)
        local = work.loc[work["segment_id"].eq(segment), ["source_utc", *features]].set_index("source_utc")
        record: dict[str, object] = {
            "event_id": str(event.event_id),
            "segment_id": segment,
            "anchor_source_utc": anchor,
            "sequence_available_utc": anchor,
            "source_state": int(event.source_state),
            "destination_state": int(event.destination_state),
        }
        for horizon in config.sequence_horizons_hours:
            expected = pd.date_range(anchor - pd.Timedelta(hours=int(horizon)), anchor - HOUR, freq="h", tz="UTC")
            present = bool(expected.isin(local.index).all())
            sequence = local.reindex(expected)
            complete = bool(present and len(sequence) == int(horizon) and not sequence.index.has_duplicates)
            record[f"sequence__complete_{horizon}h"] = np.int8(complete)
            for feature in features:
                values = pd.to_numeric(sequence[feature], errors="coerce") if complete else pd.Series(dtype=float)
                if not complete or values.isna().any():
                    record[f"sequence__{feature}__mean_{horizon}h"] = np.nan
                    record[f"sequence__{feature}__std_{horizon}h"] = np.nan
                    record[f"sequence__{feature}__slope_per_hour_{horizon}h"] = np.nan
                    record[f"sequence__{feature}__delta_{horizon}h"] = np.nan
                    continue
                array = values.to_numpy(dtype=float)
                x = np.arange(len(array), dtype=float)
                record[f"sequence__{feature}__mean_{horizon}h"] = float(np.mean(array))
                record[f"sequence__{feature}__std_{horizon}h"] = float(np.std(array, ddof=0))
                record[f"sequence__{feature}__slope_per_hour_{horizon}h"] = float(np.polyfit(x, array, deg=1)[0]) if len(array) > 1 else 0.0
                record[f"sequence__{feature}__delta_{horizon}h"] = float(array[-1] - array[0]) if len(array) > 1 else 0.0
        records.append(record)
    return pd.DataFrame.from_records(records)


def sample_stable_vs_transition(
    labeled: pd.DataFrame,
    *,
    stable_to_transition_ratio: float = 1.0,
    random_state: int = 1729,
) -> pd.DataFrame:
    """Return a balanced, event-grouped stable-vs-transition research sample.

    The sample is a label construction utility, not a causal feature maker.
    Positive rows keep their transition event group.  Stable controls are
    deterministically subsampled and grouped by segment plus UTC week so a
    downstream OOF splitter can keep correlated controls together.
    """

    required = {
        "source_utc",
        "segment_id",
        "target__pattern_phase",
        "target__pattern_event_id",
        "target__pattern_transition_context",
        "target__pattern_transition_context_available_utc",
        "target__pattern_stable_eligible",
    }
    missing = sorted(required.difference(labeled.columns))
    if missing:
        raise KeyError(f"pattern-labelled frame lacks {missing}")
    if stable_to_transition_ratio <= 0:
        raise ValueError("stable_to_transition_ratio must be positive")
    work = labeled.copy()
    work["source_utc"] = _utc_series(work["source_utc"], "source_utc")
    work["target__pattern_transition_context_available_utc"] = pd.to_datetime(
        work["target__pattern_transition_context_available_utc"], utc=True, errors="coerce"
    )
    positive = work.loc[
        work["target__pattern_transition_context"].eq(1.0)
        & work["target__pattern_transition_context_available_utc"].notna()
    ].copy()
    stable = work.loc[
        work["target__pattern_stable_eligible"].eq(1)
        & work["target__pattern_transition_context_available_utc"].notna()
    ].copy()
    requested = int(np.ceil(len(positive) * float(stable_to_transition_ratio)))
    stable = stable.sample(n=min(requested, len(stable)), random_state=random_state) if requested else stable.iloc[0:0]
    positive["target__stable_vs_transition"] = np.int8(1)
    stable["target__stable_vs_transition"] = np.int8(0)
    positive["transition_cv_group_id"] = "event::" + positive["target__pattern_event_id"].astype(str)
    stable["transition_cv_group_id"] = (
        "stable::"
        + stable["segment_id"].astype(str)
        + "::"
        + stable["source_utc"].dt.tz_convert("UTC").dt.strftime("%G-W%V")
    )
    result = pd.concat([positive, stable], ignore_index=True)
    return result.sort_values("source_utc", kind="stable").reset_index(drop=True)
