"""Read-only TP6/R3 source boundary for base portability diagnosis.

This module deliberately does *not* fit a base model or score a row.  It
loads exactly the F0 fields declared by a frozen
``base_feature_arm_lineage.json`` alongside the TP6/SL4 H12 label substrate so
that downstream diagnostics can distinguish model, input-population and
economic-relationship drift without quietly changing the base contract.

The physical candidate panel is read part-by-part and column-pruned to the
union of the selected F0 fields.  Returned scopes retain that union (useful for
cross-transport diagnostics) and also declare the precise side/run feature
contract a later frozen-model diagnostic must use.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .tp6_portability_data import LABEL_RESOLUTION_HOURS, SIDES, TP6_SL4_COST_BPS


F0_ARM = "F0_current_frozen"
R3_CLEAR_BUFFER_BPS = 25
R3_LABEL_COLUMNS = (
    "label_valid",
    "lower_touch_minute",
    "robust_clear_event_b25",
)


class BasePortabilitySourceError(ValueError):
    """Raised when frozen source/lineage identity or target semantics drift."""


@dataclass(frozen=True)
class BasePortabilitySourceContract:
    """Immutable roots needed for the read-only base-diagnosis source."""

    panel: Path
    winner: Path
    robust: Path
    lineage: Path
    cost_bps: float = TP6_SL4_COST_BPS

    def __post_init__(self) -> None:
        for name in ("panel", "winner", "robust", "lineage"):
            path = Path(getattr(self, name))
            if not path.exists():
                raise BasePortabilitySourceError(f"required source path is absent: {path}")
            object.__setattr__(self, name, path)
        if not np.isfinite(float(self.cost_bps)) or float(self.cost_bps) < 0:
            raise BasePortabilitySourceError("cost_bps must be finite and non-negative")


@dataclass(frozen=True)
class TransportScope:
    """End-exclusive decision-time scope used by a diagnostic transport."""

    name: str
    start: object
    end: object
    lineage_run: str | None = None

    def timestamps(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        start, end = (_utc(self.start, name="start"), _utc(self.end, name="end"))
        if end <= start:
            raise BasePortabilitySourceError("transport end must be after start")
        return start, end


@dataclass(frozen=True)
class F0Lineage:
    """Exact F0 selected fields, keyed by lineage run then canonical side."""

    features_by_run_side: Mapping[str, Mapping[str, tuple[str, ...]]]

    @property
    def union_features(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(
            feature
            for run in self.features_by_run_side.values()
            for fields in run.values()
            for feature in fields
        ))

    def selected_features(self, *, run: str, side: str) -> tuple[str, ...]:
        if side not in SIDES:
            raise BasePortabilitySourceError(f"unknown canonical side: {side!r}")
        try:
            fields = self.features_by_run_side[run][side]
        except KeyError as exc:
            raise BasePortabilitySourceError(
                f"F0 lineage lacks side={side!r} for run={run!r}"
            ) from exc
        return fields


@dataclass(frozen=True)
class BasePortabilitySourcePanel:
    """A bounded side/transport diagnostic panel and its frozen F0 contract."""

    frame: pd.DataFrame
    side: str
    transport: str
    lineage_run: str
    selected_features: tuple[str, ...]
    union_features: tuple[str, ...]


def _utc(value: object, *, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    if pd.isna(timestamp):
        raise BasePortabilitySourceError(f"{name} must be a finite UTC timestamp")
    return timestamp


def parse_f0_feature_lineage(path: Path) -> F0Lineage:
    """Parse and validate side-local F0 feature contracts from frozen lineage."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BasePortabilitySourceError(f"cannot read F0 feature lineage: {path}") from exc
    if not isinstance(payload, list):
        raise BasePortabilitySourceError("base feature lineage must be a JSON list")
    output: dict[str, dict[str, tuple[str, ...]]] = {}
    for item in payload:
        if not isinstance(item, Mapping) or str(item.get("arm", "")) != F0_ARM:
            continue
        run, side = str(item.get("run", "")), str(item.get("side", "")).lower()
        fields = tuple(map(str, item.get("features", ())))
        if not run or side not in SIDES or not fields or len(fields) != len(set(fields)):
            raise BasePortabilitySourceError("F0 lineage needs non-empty unique fields, run, and canonical side")
        target = str(item.get("target", ""))
        if "R3" not in target or "cost +25bps" not in target:
            raise BasePortabilitySourceError("F0 lineage does not declare the canonical R3 b25 target")
        old = output.setdefault(run, {}).get(side)
        if old is not None and old != fields:
            raise BasePortabilitySourceError(f"ambiguous F0 feature contract for {run}/{side}")
        output[run][side] = fields
    if not output:
        raise BasePortabilitySourceError("lineage contains no F0_current_frozen contracts")
    incomplete = {run: sorted(set(SIDES).difference(sides)) for run, sides in output.items() if set(sides) != set(SIDES)}
    if incomplete:
        raise BasePortabilitySourceError(f"F0 lineage lacks canonical side contracts: {incomplete}")
    return F0Lineage(features_by_run_side={run: dict(sides) for run, sides in output.items()})


class BasePortabilitySourceMaterializer:
    """Column-pruned, identity-strict TP6/R3 reader for portability diagnostics."""

    def __init__(self, contract: BasePortabilitySourceContract):
        self.contract = contract
        self.lineage = parse_f0_feature_lineage(contract.lineage)

    def load(self, *, scope: TransportScope, side: str) -> BasePortabilitySourcePanel:
        """Return valid R3 rows for one side and end-exclusive transport period.

        The returned frame contains only decision-time F0 union fields and
        declared label/economic columns.  Invalid/incomplete future paths are
        rejected rather than turned into a weak R3 outcome.
        """
        if side not in SIDES:
            raise BasePortabilitySourceError(f"unknown canonical side: {side!r}")
        start, end = scope.timestamps()
        lineage_run = scope.lineage_run or scope.name
        selected = self.lineage.selected_features(run=lineage_run, side=side)
        union = self.lineage.union_features
        names = _part_names(self.contract)
        parts = [
            _read_part(self.contract, name=name, union_features=union, side=side, start=start, end=end)
            for name in names
        ]
        parts = [part for part in parts if not part.empty]
        if not parts:
            raise BasePortabilitySourceError(f"no valid source rows for {scope.name}/{side}")
        frame = pd.concat(parts, ignore_index=True)
        if frame["candidate_id"].isna().any() or frame["candidate_id"].duplicated().any():
            raise BasePortabilitySourceError("candidate identity must be non-null and unique across source parts")
        frame = frame.sort_values(["decision_ts", "candidate_id"], kind="stable").reset_index(drop=True)
        return BasePortabilitySourcePanel(
            frame=frame, side=side, transport=scope.name, lineage_run=lineage_run,
            selected_features=selected, union_features=union,
        )


def _part_names(contract: BasePortabilitySourceContract) -> list[str]:
    names = sorted(path.name for path in (contract.panel / "parts").glob("*.parquet"))
    if not names:
        raise BasePortabilitySourceError("candidate panel contains no parquet parts")
    missing = [
        name for name in names
        if not (contract.winner / "parts" / name).is_file() or not (contract.robust / "parts" / name).is_file()
    ]
    if missing:
        raise BasePortabilitySourceError(f"source parts are incomplete: {missing[:8]}")
    return names


def _require_parquet_columns(path: Path, columns: Sequence[str]) -> None:
    available = set(pq.ParquetFile(path).schema_arrow.names)
    missing = sorted(set(columns).difference(available))
    if missing:
        raise BasePortabilitySourceError(f"{path.name} lacks required columns: {missing[:10]}")


def _read_part(
    contract: BasePortabilitySourceContract,
    *, name: str, union_features: Sequence[str], side: str, start: pd.Timestamp, end: pd.Timestamp,
) -> pd.DataFrame:
    panel_path = contract.panel / "parts" / name
    panel_columns = list(dict.fromkeys(["candidate_id", "__ts__", "side_name", *union_features]))
    _require_parquet_columns(panel_path, panel_columns)
    panel = pd.read_parquet(panel_path, columns=panel_columns)
    panel["decision_ts"] = pd.to_datetime(panel.pop("__ts__"), utc=True, errors="coerce")
    if panel["decision_ts"].isna().any():
        raise BasePortabilitySourceError(f"{name} has invalid decision timestamps")
    panel = panel.loc[
        panel["side_name"].astype(str).eq(side)
        & panel["decision_ts"].ge(start)
        & panel["decision_ts"].lt(end)
    ].copy()
    if panel.empty:
        return panel

    winner_path = contract.winner / "parts" / name
    robust_path = contract.robust / "parts" / name
    winner_columns = ("candidate_id", "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps", "__label_available_at__")
    robust_columns = ("candidate_id", *R3_LABEL_COLUMNS)
    _require_parquet_columns(winner_path, winner_columns)
    _require_parquet_columns(robust_path, robust_columns)
    winner = pd.read_parquet(winner_path, columns=list(winner_columns)).rename(columns={
        "t4_tp6_sl4_gross_bps": "gross_bps", "t4_tp6_sl4_net_bps": "net_bps",
        "__label_available_at__": "label_available_ts",
    })
    robust = pd.read_parquet(robust_path, columns=list(robust_columns))
    if winner["candidate_id"].duplicated().any() or robust["candidate_id"].duplicated().any():
        raise BasePortabilitySourceError(f"{name} has non-unique label candidate identities")
    # A candidate can legitimately have no complete future path in the
    # sidecar (for example after a listing interruption).  That is an invalid
    # supervision row, never an ordinary weak outcome.  Retain the join flags
    # just long enough to turn such rows into explicit invalid rows, then
    # remove them through the single validity gate below.  Failing here would
    # force callers either to discard whole symbols or to silently encode the
    # missing labels as economic failures.
    merged = panel.merge(winner, on="candidate_id", how="left", validate="one_to_one", indicator="_winner_join")
    merged = merged.drop(columns="_winner_join").merge(
        robust, on="candidate_id", how="left", validate="one_to_one", indicator="_robust_join"
    )
    joined = merged["_robust_join"].eq("both") & merged["label_available_ts"].notna()
    merged = merged.drop(columns="_robust_join")
    merged.loc[~joined, "label_valid"] = False
    merged["label_available_ts"] = pd.to_datetime(merged["label_available_ts"], utc=True, errors="coerce")
    if merged.loc[joined, "label_available_ts"].isna().any():
        raise BasePortabilitySourceError(f"{name} has invalid label availability timestamps on joined rows")
    resolution_hours = (merged["label_available_ts"] - merged["decision_ts"]).dt.total_seconds() / 3600.0
    if not np.allclose(resolution_hours.loc[joined].to_numpy(float), LABEL_RESOLUTION_HOURS, atol=1e-6, rtol=0.0):
        raise BasePortabilitySourceError("TP6 labels must resolve exactly 13h after the decision bar")
    valid = merged["label_valid"].fillna(False).astype(bool)
    merged = merged.loc[valid].copy()
    if merged.empty:
        return merged
    gross = pd.to_numeric(merged["gross_bps"], errors="coerce")
    net = pd.to_numeric(merged["net_bps"], errors="coerce")
    if not np.isfinite(gross.to_numpy(float)).all() or not np.isfinite(net.to_numpy(float)).all():
        raise BasePortabilitySourceError("valid TP6 rows must have finite gross and net bps")
    if not np.allclose(gross.to_numpy(float) - net.to_numpy(float), float(contract.cost_bps), atol=0.02, rtol=0.0):
        raise BasePortabilitySourceError("TP6 source must charge the declared cost exactly once")
    clear = pd.to_numeric(merged["robust_clear_event_b25"], errors="coerce")
    lower = pd.to_numeric(merged["lower_touch_minute"], errors="coerce")
    if clear.isna().any() or lower.isna().any() or not clear.isin((0, 1)).all():
        raise BasePortabilitySourceError("valid rows require finite canonical R3 b25/lower path inputs")
    merged["r3_class"] = np.select([clear.eq(1), lower.ge(0)], [2, 0], default=1).astype(np.int8)
    merged["asset"] = merged["candidate_id"].astype(str).str.split("|", n=1, regex=False).str[0].astype("string")
    for field in [*union_features, "gross_bps", "net_bps"]:
        merged[field] = pd.to_numeric(merged[field], errors="coerce").astype(np.float32)
    return merged


__all__ = [
    "BasePortabilitySourceContract", "BasePortabilitySourceError", "BasePortabilitySourceMaterializer",
    "BasePortabilitySourcePanel", "F0Lineage", "F0_ARM", "R3_CLEAR_BUFFER_BPS", "TransportScope",
    "parse_f0_feature_lineage",
]
