"""Bounded, timestamp-safe readers for strict contribution-event streams.

The event sidecar is deliberately partitioned by a paired source part.  A
consumer must never decode a whole ``(contract, side, head)`` scope: this
module opens at most a small, caller-bounded number of Parquet files, keeps a
single Arrow batch and one unfinished candidate group per file, and merges
their candidate groups deterministically.  The timestamp-block API is the
safe production API for prequential health: all feature events at ``T`` are
scored before any label-resolution events at ``T`` are applied.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
from pathlib import Path
from typing import Iterable, Iterator, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd

try:  # pragma: no cover - production requires Arrow
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pa = pq = None

from .strict_contribution_event_stream import EVENT_COLUMNS, StrictContributionEventStreams
from .strict_event_store import StrictEventStoreError


_GROUP_IDENTITY = ("candidate_id", "fold_id", "transport", "meta_partition")
_DEFAULT_BATCH_ROWS = 131_072
_DEFAULT_MAX_OPEN_PARTS = 8


def _require_arrow() -> None:
    if pa is None or pq is None:
        raise StrictEventStoreError("pyarrow is required for contribution-event reading")


def _dictionary_values(table: "pa.Table", column: str) -> tuple[np.ndarray, tuple[str, ...]]:
    array = table.column(column).combine_chunks()
    if not isinstance(array, pa.DictionaryArray):
        raise StrictEventStoreError(f"event-stream {column} must retain dictionary physical encoding")
    return array.indices.to_numpy(zero_copy_only=False).astype(np.int32, copy=False), tuple(map(str, array.dictionary.to_pylist()))


def _timestamp_ns(table: "pa.Table", column: str) -> np.ndarray:
    array = table.column(column).combine_chunks()
    return array.cast(pa.int64()).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)


def _float32(table: "pa.Table", column: str) -> np.ndarray:
    return table.column(column).combine_chunks().to_numpy(zero_copy_only=False).astype(np.float32, copy=False)


def _family_codes(
    rule_codes: np.ndarray, rule_dictionary: Sequence[str], direction_codes: np.ndarray,
    direction_dictionary: Sequence[str], family_lookup: MutableMapping[tuple[str, str], int],
) -> tuple[np.ndarray, np.ndarray]:
    """Map a batch-local dictionary pair to global compact family codes."""

    direction_map = np.empty(len(direction_dictionary), dtype=np.int8)
    for index, value in enumerate(direction_dictionary):
        if value == "negative":
            direction_map[index] = 0
        elif value == "positive":
            direction_map[index] = 1
        else:
            raise StrictEventStoreError(f"unknown contribution direction in event stream: {value}")
    result = np.empty(len(rule_codes), dtype=np.int32)
    # Allocate only observed rule/direction pairs.  Dictionary layouts vary by
    # batch and may contain combinations which have no physical row.
    seen: dict[int, int] = {}
    width = len(direction_dictionary)
    for index in range(len(rule_codes)):
        pair = int(rule_codes[index]) * width + int(direction_codes[index])
        code = seen.get(pair)
        if code is None:
            key = (str(rule_dictionary[int(rule_codes[index])]), str(direction_dictionary[int(direction_codes[index])]))
            code = family_lookup.get(key)
            if code is None:
                code = len(family_lookup)
                family_lookup[key] = code
            seen[pair] = code
        result[index] = code
    return result, direction_map[direction_codes]


@dataclass(frozen=True)
class ContributionEventGroup:
    timestamp_ns: int
    candidate_id: str
    fold_id: str
    transport: str
    meta_partition: str
    decision_timestamp_ns: int
    feature_generation_timestamp_ns: int
    label_available_timestamp_ns: int
    semantic_label: float
    head_prediction: float
    net_bps: float
    base_expected_bps: float
    asset: str
    family_codes: np.ndarray
    directions: np.ndarray
    contributions: np.ndarray


@dataclass(frozen=True)
class ContributionEventTimestampBlock:
    """All complete candidate groups which share one strict event timestamp."""

    timestamp_ns: int
    groups: tuple[ContributionEventGroup, ...]


def _same_candidate(left: ContributionEventGroup, right: ContributionEventGroup) -> bool:
    return (
        left.timestamp_ns == right.timestamp_ns
        and left.candidate_id == right.candidate_id
        and left.fold_id == right.fold_id
        and left.transport == right.transport
        and left.meta_partition == right.meta_partition
    )


def _merge_candidate(left: ContributionEventGroup, right: ContributionEventGroup) -> ContributionEventGroup:
    if not _same_candidate(left, right):
        raise StrictEventStoreError("attempted to merge distinct contribution-event candidates")
    scalar = (
        "decision_timestamp_ns", "feature_generation_timestamp_ns", "label_available_timestamp_ns", "semantic_label", "head_prediction", "net_bps", "base_expected_bps", "asset",
    )
    if any(getattr(left, field) != getattr(right, field) for field in scalar):
        raise StrictEventStoreError("candidate contribution group is inconsistent across Parquet batches")
    return ContributionEventGroup(
        timestamp_ns=left.timestamp_ns, candidate_id=left.candidate_id, fold_id=left.fold_id,
        transport=left.transport, meta_partition=left.meta_partition,
        decision_timestamp_ns=left.decision_timestamp_ns,
        feature_generation_timestamp_ns=left.feature_generation_timestamp_ns,
        label_available_timestamp_ns=left.label_available_timestamp_ns,
        semantic_label=left.semantic_label,
        head_prediction=left.head_prediction, net_bps=left.net_bps,
        base_expected_bps=left.base_expected_bps, asset=left.asset,
        family_codes=np.concatenate((left.family_codes, right.family_codes)),
        directions=np.concatenate((left.directions, right.directions)),
        contributions=np.concatenate((left.contributions, right.contributions)),
    )


class _PartGroups:
    """Lazy candidate-group cursor over one sorted physical Parquet part.

    At most one Arrow batch plus a queue of views into that batch is retained.
    A candidate whose contiguous family rows span a batch boundary is merged
    before it can be emitted.
    """

    def __init__(
        self, path: Path, *, timestamp_column: str, family_lookup: MutableMapping[tuple[str, str], int],
        include_after_ns: int | None = None, include_before_ns: int | None = None,
        batch_rows: int = _DEFAULT_BATCH_ROWS,
    ) -> None:
        _require_arrow()
        if int(batch_rows) <= 0:
            raise StrictEventStoreError("contribution event reader batch_rows must be positive")
        self.path = path
        self._timestamp_column = timestamp_column
        self._lookup = family_lookup
        self._after = None if include_after_ns is None else int(include_after_ns)
        self._before = None if include_before_ns is None else int(include_before_ns)
        self._file = pq.ParquetFile(path)
        names = set(self._file.schema_arrow.names)
        if set(EVENT_COLUMNS).difference(names):
            raise StrictEventStoreError(f"event stream part has incomplete schema: {path}")
        self._batches = self._file.iter_batches(batch_size=int(batch_rows), columns=list(EVENT_COLUMNS))
        self._ready: deque[ContributionEventGroup] = deque()
        self._carry: ContributionEventGroup | None = None
        self._done = False
        self._last_row_timestamp: int | None = None

    def _decode_batch(self, batch: "pa.RecordBatch") -> list[ContributionEventGroup]:
        table = pa.Table.from_batches([batch])
        if not len(table):
            return []
        time = _timestamp_ns(table, self._timestamp_column)
        if np.any(time[1:] < time[:-1]):
            raise StrictEventStoreError(f"event stream part is not ordered by {self._timestamp_column}: {self.path}")
        if self._last_row_timestamp is not None and int(time[0]) < self._last_row_timestamp:
            raise StrictEventStoreError(f"event stream part regresses across Parquet batches: {self.path}")
        self._last_row_timestamp = int(time[-1])
        identity_codes: list[np.ndarray] = []
        identity_values: list[tuple[str, ...]] = []
        for column in _GROUP_IDENTITY:
            codes, values = _dictionary_values(table, column)
            identity_codes.append(codes)
            identity_values.append(values)
        changed = np.zeros(len(table), dtype=bool)
        changed[0] = True
        for codes in identity_codes:
            changed[1:] |= codes[1:] != codes[:-1]
        starts = np.flatnonzero(changed).astype(np.int64, copy=False)
        ends = np.empty(len(starts), dtype=np.int64)
        ends[:-1] = starts[1:]
        ends[-1] = len(table)
        rule_codes, rules = _dictionary_values(table, "rule_signature")
        direction_codes, directions = _dictionary_values(table, "contribution_direction")
        family_codes, direction_values = _family_codes(rule_codes, rules, direction_codes, directions, self._lookup)
        decision = _timestamp_ns(table, "decision_ts")
        feature_generation = _timestamp_ns(table, "feature_generation_ts")
        label_available = _timestamp_ns(table, "label_available_ts")
        semantic = _float32(table, "semantic_label")
        prediction = _float32(table, "head_prediction")
        net = _float32(table, "net_bps")
        expected = _float32(table, "base_expected_bps")
        contributions = _float32(table, "family_ensemble_tree_contribution")
        asset_codes, assets = _dictionary_values(table, "asset")
        result: list[ContributionEventGroup] = []
        for start, end in zip(starts, ends, strict=True):
            start_i, end_i = int(start), int(end)
            timestamp = int(time[start_i])
            # The range is defined at candidate level: a physical candidate
            # group is never cut merely because a source part overlaps a bound.
            if (self._after is not None and timestamp < self._after) or (self._before is not None and timestamp >= self._before):
                continue
            if np.any(time[start_i:end_i] != timestamp):
                raise StrictEventStoreError("event-stream candidate group crosses its timestamp boundary")
            if not np.all(decision[start_i:end_i] == decision[start_i]):
                raise StrictEventStoreError("candidate event group has inconsistent decision timestamps")
            if not np.all(feature_generation[start_i:end_i] == feature_generation[start_i]):
                raise StrictEventStoreError("candidate event group has inconsistent feature-generation timestamps")
            if not np.all(label_available[start_i:end_i] == label_available[start_i]):
                raise StrictEventStoreError("candidate event group has inconsistent label-availability timestamps")
            if (
                not np.all(semantic[start_i:end_i] == semantic[start_i])
                or not np.all(prediction[start_i:end_i] == prediction[start_i])
                or not np.all(net[start_i:end_i] == net[start_i])
                or not np.all(expected[start_i:end_i] == expected[start_i])
                or not np.all(asset_codes[start_i:end_i] == asset_codes[start_i])
            ):
                raise StrictEventStoreError("candidate event group has inconsistent outcome fields")
            result.append(ContributionEventGroup(
                timestamp_ns=timestamp,
                candidate_id=identity_values[0][int(identity_codes[0][start_i])],
                fold_id=identity_values[1][int(identity_codes[1][start_i])],
                transport=identity_values[2][int(identity_codes[2][start_i])],
                meta_partition=identity_values[3][int(identity_codes[3][start_i])],
                decision_timestamp_ns=int(decision[start_i]),
                feature_generation_timestamp_ns=int(feature_generation[start_i]),
                label_available_timestamp_ns=int(label_available[start_i]), semantic_label=float(semantic[start_i]),
                head_prediction=float(prediction[start_i]), net_bps=float(net[start_i]),
                base_expected_bps=float(expected[start_i]), asset=assets[int(asset_codes[start_i])],
                family_codes=family_codes[start_i:end_i], directions=direction_values[start_i:end_i],
                contributions=contributions[start_i:end_i],
            ))
        return result

    def _fill_ready(self) -> None:
        while not self._ready and not self._done:
            try:
                batch = next(self._batches)
            except StopIteration:
                self._done = True
                if self._carry is not None:
                    self._ready.append(self._carry)
                    self._carry = None
                return
            for group in self._decode_batch(batch):
                if self._carry is None:
                    self._carry = group
                elif _same_candidate(self._carry, group):
                    self._carry = _merge_candidate(self._carry, group)
                else:
                    self._ready.append(self._carry)
                    self._carry = group

    def has_group(self) -> bool:
        self._fill_ready()
        return bool(self._ready)

    def timestamp(self) -> int:
        if not self.has_group():
            raise StrictEventStoreError("attempted to read exhausted contribution-event part")
        return int(self._ready[0].timestamp_ns)

    def pop(self) -> ContributionEventGroup:
        if not self.has_group():
            raise StrictEventStoreError("attempted to pop exhausted contribution-event part")
        return self._ready.popleft()


def _timestamp_bound_ns(value: object) -> int:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")
    return int(parsed.value)


def _select_parts(
    streams: StrictContributionEventStreams, *, dataset: str, contract: str, side: str, head: str,
    months: Iterable[str] | None, include_after_ns: int | None, include_before_ns: int | None,
    max_open_parts: int,
) -> pd.DataFrame:
    selected = streams.part_index.loc[
        streams.part_index["dataset"].eq(dataset)
        & streams.part_index["contract"].astype(str).eq(str(contract))
        & streams.part_index["side"].astype(str).eq(str(side))
        & streams.part_index["head"].astype(str).eq(str(head))
    ].copy()
    if months is not None:
        selected = selected.loc[selected["month"].astype(str).isin(set(map(str, months)))].copy()
    if selected.empty:
        return selected
    # Reject physical files which cannot contain the requested time interval
    # before opening Parquet metadata or decoding a row.
    if include_after_ns is not None:
        selected = selected.loc[
            selected["max_timestamp"].map(_timestamp_bound_ns).ge(int(include_after_ns))
        ].copy()
    if include_before_ns is not None:
        selected = selected.loc[
            selected["min_timestamp"].map(_timestamp_bound_ns).lt(int(include_before_ns))
        ].copy()
    if len(selected) > int(max_open_parts):
        raise StrictEventStoreError(
            "unbounded contribution-event request: select a narrow month/time window "
            f"(selected_parts={len(selected)}, max_open_parts={int(max_open_parts)})"
        )
    return selected.sort_values(["month", "path"], kind="stable")


def iter_contribution_event_groups(
    streams: StrictContributionEventStreams, *, dataset: str, contract: str, side: str, head: str,
    months: Iterable[str] | None = None, include_after_ns: int | None = None, include_before_ns: int | None = None,
    family_lookup: MutableMapping[tuple[str, str], int] | None = None,
    batch_rows: int = _DEFAULT_BATCH_ROWS, max_open_parts: int = _DEFAULT_MAX_OPEN_PARTS,
) -> Iterator[ContributionEventGroup]:
    """K-way merge a bounded set of already-sorted event pieces.

    ``months`` is mandatory in production callers.  A defensive open-file cap
    rejects broad requests even when a caller accidentally omits it.
    """

    if dataset not in {"feature_event_order", "resolution_event_order"}:
        raise StrictEventStoreError("unknown contribution event-stream dataset")
    if int(max_open_parts) <= 0:
        raise StrictEventStoreError("contribution event reader max_open_parts must be positive")
    selected = _select_parts(
        streams, dataset=dataset, contract=contract, side=side, head=head, months=months,
        include_after_ns=include_after_ns, include_before_ns=include_before_ns,
        max_open_parts=int(max_open_parts),
    )
    if selected.empty:
        return
    timestamp_column = str(selected["timestamp_column"].iloc[0])
    if not selected["timestamp_column"].eq(timestamp_column).all():
        raise StrictEventStoreError("event-stream scope has inconsistent ordering columns")
    lookup: MutableMapping[tuple[str, str], int] = family_lookup if family_lookup is not None else {}
    parts: list[_PartGroups] = []
    for item in selected.itertuples(index=False):
        part = _PartGroups(
            streams.root / str(item.path), timestamp_column=timestamp_column, family_lookup=lookup,
            include_after_ns=include_after_ns, include_before_ns=include_before_ns, batch_rows=int(batch_rows),
        )
        if part.has_group():
            parts.append(part)
    heap: list[tuple[int, int]] = [(part.timestamp(), index) for index, part in enumerate(parts)]
    heapq.heapify(heap)
    while heap:
        _, index = heapq.heappop(heap)
        part = parts[index]
        yield part.pop()
        if part.has_group():
            heapq.heappush(heap, (part.timestamp(), index))


def iter_contribution_event_timestamp_blocks(
    streams: StrictContributionEventStreams, *, dataset: str, contract: str, side: str, head: str,
    months: Iterable[str] | None = None, include_after_ns: int | None = None, include_before_ns: int | None = None,
    family_lookup: MutableMapping[tuple[str, str], int] | None = None,
    batch_rows: int = _DEFAULT_BATCH_ROWS, max_open_parts: int = _DEFAULT_MAX_OPEN_PARTS,
) -> Iterator[ContributionEventTimestampBlock]:
    """Yield complete, deterministically ordered candidate sets for each timestamp."""

    current_time: int | None = None
    current: list[ContributionEventGroup] = []
    for group in iter_contribution_event_groups(
        streams, dataset=dataset, contract=contract, side=side, head=head, months=months,
        include_after_ns=include_after_ns, include_before_ns=include_before_ns, family_lookup=family_lookup,
        batch_rows=batch_rows, max_open_parts=max_open_parts,
    ):
        if current_time is None:
            current_time = group.timestamp_ns
        if group.timestamp_ns != current_time:
            current.sort(key=lambda item: (item.candidate_id, item.fold_id, item.transport, item.meta_partition))
            yield ContributionEventTimestampBlock(timestamp_ns=current_time, groups=tuple(current))
            current_time = group.timestamp_ns
            current = []
        current.append(group)
    if current_time is not None:
        current.sort(key=lambda item: (item.candidate_id, item.fold_id, item.transport, item.meta_partition))
        yield ContributionEventTimestampBlock(timestamp_ns=current_time, groups=tuple(current))


__all__ = [
    "ContributionEventGroup", "ContributionEventTimestampBlock",
    "iter_contribution_event_groups", "iter_contribution_event_timestamp_blocks",
]
