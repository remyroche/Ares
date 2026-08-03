"""Memory-bounded Stage-I reference-layout data adapter.

The adapter reads narrow monthly identity/label ledgers first, selects a
deterministic time×side×economics stratified selector cohort, and only then
loads declared PIT fields through an exact ``(__symbol__, __ts__)`` loader.
It never as-of joins, fills missing values, or treats historical/reference
populations as interchangeable.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd


IDENTITY_LEDGER_COLUMNS = (
    "candidate_id", "__ts__", "__symbol__", "side_name", "label_valid",
)
NET_COLUMN_ALIASES = ("exact_net_bps", "t4_tp6_sl4_net_bps")
GROSS_COLUMN_ALIASES = ("exact_gross_bps", "t4_tp6_sl4_gross_bps")
LABEL_AVAILABLE_ALIASES = ("label_available_ts", "__label_available_at__")
R3_LABEL_COLUMNS = (
    "t2_tp6_sl4_event",
    "robust_clear_event_b25",
    "robust_clear_soft_b25_t50",
)
POPULATIONS = ("historical_2022_2023", "surface_2024", "common30_2025_2026")
MIN_FEATURE_COVERAGE = 0.90


class StageIProductionDataError(ValueError):
    pass


def _canonical_symbol(value: Any) -> str:
    """Normalize the reference-surface filename alias to the store payload id."""
    symbol = str(value).strip()
    if "/" not in symbol and "_USD:USD" in symbol:
        symbol = symbol.replace("_USD:USD", "/USD:USD", 1)
    return symbol


@dataclass(frozen=True)
class MonthlyReferencePartition:
    path: str | Path
    source_month: str
    population: str


@dataclass(frozen=True)
class StageIDataChunk:
    ledger: pd.DataFrame
    features: pd.DataFrame
    population_summary: pd.DataFrame


PointFeatureLoader = Callable[[pd.DataFrame, Sequence[str]], pd.DataFrame]


def make_static_pit_feature_loader(
    *,
    feature_store_dir: str | Path,
    feature_contract: Any,
    max_rows_per_batch: int = 8_000,
    max_columns_per_read: int = 64,
    verify_frozen_schema: bool = True,
) -> PointFeatureLoader:
    """Adapt the canonical bounded static store iterator to this adapter.

    The frozen contract is deliberately supplied by the caller.  This adapter
    never discovers a replacement universe or broadens the selected fields.
    """
    from extreme_price_movements.packb_static_point_feature_loader import (
        iter_point_in_time_feature_batches,
    )

    frozen_columns = tuple(getattr(feature_contract, "feature_columns", ()))

    def load(identity: pd.DataFrame, fields: Sequence[str]) -> pd.DataFrame:
        requested = tuple(map(str, fields))
        if requested != frozen_columns:
            raise StageIProductionDataError(
                "PIT loader fields must equal the supplied frozen feature contract; "
                "do not discover or widen fields during Stage I"
            )
        request = identity[["candidate_id", "__ts__", "__symbol__"]].copy()
        request["__request_row__"] = np.arange(len(request), dtype=np.int64)
        # Long/short candidates deliberately share the same entry-time market
        # point.  Read that exact store key once, then expand it back to each
        # side-specific candidate identity after the verified PIT lookup.
        unique = request.drop_duplicates(["__symbol__", "__ts__"], keep="first").copy()
        batches: list[pd.DataFrame] = []
        for batch in iter_point_in_time_feature_batches(
            unique[["candidate_id", "__ts__", "__symbol__"]],
            feature_store_dir=feature_store_dir,
            feature_contract=feature_contract,
            max_rows_per_batch=max_rows_per_batch,
            max_columns_per_read=max_columns_per_read,
            coverage_discovery=False,
            verify_frozen_schema=verify_frozen_schema,
        ):
            if not np.asarray(batch.matched_exact_keys, dtype=bool).all():
                raise StageIProductionDataError("canonical PIT store missed an exact requested key")
            part = batch.identity.copy()
            part.loc[:, list(requested)] = batch.features.loc[:, list(requested)].to_numpy()
            batches.append(part)
        point = pd.concat(batches, ignore_index=True).drop(columns=["candidate_id"])
        if point.duplicated(["__symbol__", "__ts__"]).any():
            raise StageIProductionDataError("canonical PIT loader duplicated an exact market key")
        point["__pit_matched__"] = True
        result = request.merge(
            point, on=["__symbol__", "__ts__"], how="left", validate="many_to_one", sort=False
        ).sort_values("__request_row__", kind="stable")
        if not result["__pit_matched__"].fillna(False).all():
            raise StageIProductionDataError("canonical PIT store missed an expanded side candidate")
        return result.drop(columns=["__request_row__", "__pit_matched__"]).reset_index(drop=True)

    return load


def _read_partition(partition: MonthlyReferencePartition) -> pd.DataFrame:
    if partition.population not in POPULATIONS:
        raise StageIProductionDataError(f"unknown reference population: {partition.population}")
    path = Path(partition.path)
    files = [path] if path.is_file() else sorted(path.glob("*.parquet")) if path.is_dir() else []
    if not files:
        raise StageIProductionDataError(f"monthly identity/label partition is missing: {path}")
    pieces: list[pd.DataFrame] = []
    for file in files:
        import pyarrow.parquet as pq

        available_columns = set(pq.ParquetFile(file).schema_arrow.names)
        missing = set(IDENTITY_LEDGER_COLUMNS).difference(available_columns)
        if missing:
            raise StageIProductionDataError(
                f"{file} lacks required identity/validity columns: {sorted(missing)}"
            )
        net_column = next((name for name in NET_COLUMN_ALIASES if name in available_columns), None)
        gross_column = next((name for name in GROSS_COLUMN_ALIASES if name in available_columns), None)
        available_column = next(
            (name for name in LABEL_AVAILABLE_ALIASES if name in available_columns), None
        )
        if net_column is None or gross_column is None or available_column is None:
            raise StageIProductionDataError(
                f"{file} lacks an exact gross/net or label-availability contract column"
            )
        missing_r3 = set(R3_LABEL_COLUMNS).difference(available_columns)
        if missing_r3:
            raise StageIProductionDataError(
                f"{file} lacks the frozen R3 economic-simplex labels: {sorted(missing_r3)}"
            )
        source_columns = [
            *IDENTITY_LEDGER_COLUMNS, gross_column, net_column, available_column,
            *R3_LABEL_COLUMNS,
        ]
        piece = pd.read_parquet(file, columns=list(dict.fromkeys(source_columns)))
        piece = piece.rename(
            columns={
                gross_column: "exact_gross_bps", net_column: "exact_net_bps",
                available_column: "label_available_ts",
            }
        )
        pieces.append(piece)
    frame = pd.concat(pieces, ignore_index=True)
    frame = frame.copy()
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    frame["label_available_ts"] = pd.to_datetime(frame["label_available_ts"], utc=True, errors="coerce")
    if frame[["__ts__", "label_available_ts"]].isna().any(axis=None):
        raise StageIProductionDataError(f"{path} has invalid decision/label timestamps")
    lag = frame["label_available_ts"] - frame["__ts__"]
    if not lag.eq(pd.Timedelta(hours=13)).all():
        raise StageIProductionDataError(
            f"{path} violates exact signal-close +13h label availability"
        )
    frame["side_name"] = frame["side_name"].astype(str).str.lower()
    frame["__symbol__"] = frame["__symbol__"].map(_canonical_symbol)
    if not frame["side_name"].isin(["long", "short"]).all():
        raise StageIProductionDataError(f"{path} has invalid side_name")
    frame["source_month"] = str(partition.source_month)
    frame["population_segment"] = str(partition.population)
    frame = frame.loc[frame["label_valid"].astype(bool)].copy()
    for column in ("exact_gross_bps", "exact_net_bps"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(frame[["exact_gross_bps", "exact_net_bps"]].to_numpy(dtype=float)).all():
        raise StageIProductionDataError(f"{path} has non-finite gross/net economics on label-valid rows")
    if not np.allclose(
        frame["exact_gross_bps"].to_numpy(float) - 100.0,
        frame["exact_net_bps"].to_numpy(float), rtol=0.0, atol=2e-3,
    ):
        raise StageIProductionDataError(f"{path} violates gross - 100bps = net")
    for column in R3_LABEL_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.loc[:, list(R3_LABEL_COLUMNS)].isna().any(axis=None):
        raise StageIProductionDataError(f"{path} has missing R3 labels on label-valid rows")
    return frame


def load_reference_ledgers(
    partitions: Sequence[MonthlyReferencePartition],
    *,
    frozen_candidate_universe: Sequence[Any] | None = None,
) -> pd.DataFrame:
    if not partitions:
        raise StageIProductionDataError("at least one monthly reference partition is required")
    frames = [_read_partition(partition) for partition in partitions]
    ledger = pd.concat(frames, ignore_index=True)
    if frozen_candidate_universe is not None:
        allowed = pd.Index(np.asarray(frozen_candidate_universe, dtype=object))
        ledger = ledger.loc[ledger["candidate_id"].isin(allowed)].copy()
    if ledger.empty:
        raise StageIProductionDataError("no label-valid rows remain after frozen-universe filter")
    if ledger["candidate_id"].duplicated().any():
        raise StageIProductionDataError("candidate_id must be unique across reference partitions")
    return ledger


def stratified_selector_sample(
    ledger: pd.DataFrame,
    *,
    max_rows: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Deterministic month×side×economic-bin sample before any wide PIT load."""
    if int(max_rows) < 1:
        raise StageIProductionDataError("selector max_rows must be positive")
    work = ledger.copy()
    work["selector_month"] = work["__ts__"].dt.strftime("%Y-%m")
    net = pd.to_numeric(work["exact_net_bps"], errors="coerce")
    if net.isna().any():
        raise StageIProductionDataError("exact_net_bps must be finite for economic stratification")
    # Global fixed bps cuts avoid fitting a future-dependent quantile surface.
    work["selector_economic_bin"] = pd.cut(
        net, bins=[-np.inf, -200.0, 0.0, 50.0, np.inf], labels=False
    ).astype(int)
    groups = list(work.groupby(["selector_month", "side_name", "selector_economic_bin"], observed=True, sort=True))
    quota = max(1, int(max_rows) // max(len(groups), 1))
    pieces: list[pd.DataFrame] = []
    for _, group in groups:
        if len(group) <= quota:
            pieces.append(group)
        else:
            pieces.append(group.sample(n=quota, random_state=int(random_state), replace=False))
    sample = pd.concat(pieces, ignore_index=True)
    if len(sample) > int(max_rows):
        sample = sample.sample(n=int(max_rows), random_state=int(random_state), replace=False)
    return sample.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _validate_exact_feature_frame(
    ledger: pd.DataFrame, features: pd.DataFrame, declared_features: Sequence[str]
) -> pd.DataFrame:
    required = {"candidate_id", "__ts__", "__symbol__", *map(str, declared_features)}
    missing = required.difference(features.columns)
    if missing:
        raise StageIProductionDataError(f"PIT loader did not return declared exact fields: {sorted(missing)}")
    out = features.loc[:, ["candidate_id", "__ts__", "__symbol__", *map(str, declared_features)]].copy()
    out["__ts__"] = pd.to_datetime(out["__ts__"], utc=True, errors="coerce")
    if out["__ts__"].isna().any() or out.duplicated(["candidate_id"]).any():
        raise StageIProductionDataError("PIT loader returned invalid/duplicate candidate identities")
    left = pd.MultiIndex.from_frame(ledger[["candidate_id", "__ts__", "__symbol__"]])
    right = pd.MultiIndex.from_frame(out[["candidate_id", "__ts__", "__symbol__"]])
    if not left.equals(right):
        raise StageIProductionDataError("PIT loader must preserve exact candidate_id/symbol/signal-close timestamp order")
    values = out.loc[:, list(map(str, declared_features))]
    coverage = values.notna().mean(axis=0)
    low = coverage[coverage < MIN_FEATURE_COVERAGE]
    if not low.empty:
        raise StageIProductionDataError(f"declared feature coverage below 90%: {low.to_dict()}")
    constant = [column for column in values if values[column].dropna().nunique() <= 1]
    if constant:
        raise StageIProductionDataError(f"declared features are constant/non-incremental: {constant}")
    # Deliberately retain NaNs for model-contract handling; never zero-fill.
    return out


def load_selector_sample(
    partitions: Sequence[MonthlyReferencePartition],
    *,
    declared_features: Sequence[str],
    pit_feature_loader: PointFeatureLoader,
    selector_max_rows: int,
    frozen_candidate_universe: Sequence[Any] | None = None,
    random_state: int = 42,
) -> StageIDataChunk:
    ledger = load_reference_ledgers(partitions, frozen_candidate_universe=frozen_candidate_universe)
    sample = stratified_selector_sample(ledger, max_rows=selector_max_rows, random_state=random_state)
    features = _validate_exact_feature_frame(sample, pit_feature_loader(sample, declared_features), declared_features)
    summary = sample.groupby(["population_segment", "source_month", "side_name"], observed=True).size().rename("rows").reset_index()
    return StageIDataChunk(ledger=sample, features=features, population_summary=summary)


def iter_frozen_oof_chunks(
    partitions: Sequence[MonthlyReferencePartition],
    *,
    selected_features: Sequence[str],
    pit_feature_loader: PointFeatureLoader,
    batch_rows: int = 10_000,
    frozen_candidate_universe: Sequence[Any] | None = None,
) -> Iterator[StageIDataChunk]:
    """Stream only frozen selected fields for full strict-OOF generation."""
    if int(batch_rows) < 1:
        raise StageIProductionDataError("batch_rows must be positive")
    ledger = load_reference_ledgers(partitions, frozen_candidate_universe=frozen_candidate_universe)
    ledger = ledger.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    for start in range(0, len(ledger), int(batch_rows)):
        chunk = ledger.iloc[start:start + int(batch_rows)].copy()
        features = _validate_exact_feature_frame(chunk, pit_feature_loader(chunk, selected_features), selected_features)
        summary = chunk.groupby(["population_segment", "source_month", "side_name"], observed=True).size().rename("rows").reset_index()
        yield StageIDataChunk(ledger=chunk, features=features, population_summary=summary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage-I production-data adapter dry run")
    parser.add_argument("--dry-run", action="store_true", help="print the adapter contract; never reads features or trains")
    args = parser.parse_args(argv)
    if not args.dry_run:
        parser.error("only --dry-run is supported; callers supply reference partitions and PIT loader in code")
    print("stage_i_production_data_adapter: label_valid -> frozen-universe -> stratified sample -> exact PIT join -> coverage gate")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
