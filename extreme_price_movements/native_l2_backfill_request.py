"""Build a label-free request manifest for missing native-L2 history.

The request is deliberately narrower than a training materializer.  It reads
only candidate product identity and candidate-availability timestamps, plus
the already sealed native sidecar's product/timestamp columns.  Labels,
scores, costs, portfolio fields, and model outputs are never loaded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import pyarrow.parquet as pq


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _read_candidate_pairs(
    panel: Mapping[str, Any],
    *,
    root: Path,
) -> pd.DataFrame:
    """Read only ``symbol`` and the declared candidate time column."""
    path = _resolve_path(root, str(panel["path"]))
    time_column = str(panel["time_column"])
    available = set(pq.ParquetFile(path).schema.names)
    symbol_column = next(
        (column for column in ("__symbol__", "symbol", "product", "instrument") if column in available),
        None,
    )
    if symbol_column is None:
        raise ValueError(f"candidate panel has no product identity column: {path}")
    if time_column not in available:
        raise ValueError(f"declared candidate time column is missing: {path}:{time_column}")
    frame = pd.read_parquet(path, engine="pyarrow", columns=[symbol_column, time_column])
    frame = frame.rename(columns={symbol_column: "symbol", time_column: "candidate_ts"})
    frame["symbol"] = frame["symbol"].astype("string")
    frame["candidate_ts"] = pd.to_datetime(frame["candidate_ts"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["symbol", "candidate_ts"])
    frame = frame.loc[frame["symbol"].ne("")].copy()
    frame["utc_day"] = frame["candidate_ts"].dt.strftime("%Y-%m-%d")
    frame["panel_id"] = str(panel["panel_id"])
    return frame[["panel_id", "symbol", "candidate_ts", "utc_day"]]


def _candidate_day_requirements(
    panels: Iterable[Mapping[str, Any]],
    *,
    root: Path,
) -> pd.DataFrame:
    pieces = [_read_candidate_pairs(panel, root=root) for panel in panels]
    if not pieces:
        return pd.DataFrame(
            columns=["symbol", "utc_day", "candidate_rows", "panel_count", "panel_ids", "min_candidate_ts", "max_candidate_ts"]
        )
    rows = pd.concat(pieces, ignore_index=True)
    grouped = (
        rows.groupby(["symbol", "utc_day"], sort=True, observed=True)
        .agg(
            candidate_rows=("candidate_ts", "size"),
            panel_count=("panel_id", "nunique"),
            panel_ids=("panel_id", lambda values: ",".join(sorted(set(map(str, values))))),
            min_candidate_ts=("candidate_ts", "min"),
            max_candidate_ts=("candidate_ts", "max"),
        )
        .reset_index()
    )
    return grouped


def _native_day_counts(
    native_sidecar: Path | None,
    *,
    root: Path,
) -> pd.DataFrame:
    columns = ["symbol", "snapshot_ts"]
    if native_sidecar is None:
        return pd.DataFrame(columns=["symbol", "utc_day", "native_snapshots"])
    path = _resolve_path(root, native_sidecar)
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "utc_day", "native_snapshots"])
    frame = pd.read_parquet(path, engine="pyarrow", columns=columns)
    frame["symbol"] = frame["symbol"].astype("string")
    frame["snapshot_ts"] = pd.to_datetime(frame["snapshot_ts"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["symbol", "snapshot_ts"])
    frame["utc_day"] = frame["snapshot_ts"].dt.strftime("%Y-%m-%d")
    return (
        frame.groupby(["symbol", "utc_day"], sort=True, observed=True)
        .size()
        .rename("native_snapshots")
        .reset_index()
    )


def build_backfill_request(
    panels: Iterable[Mapping[str, Any]],
    *,
    root: Path,
    native_sidecar: Path | None = None,
    source_allowlist: tuple[str, ...] = ("kraken_futures_l2_snapshot",),
    max_staleness_hours: float = 2.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return symbol/day requirements and a fail-closed request summary."""
    requirements = _candidate_day_requirements(panels, root=root)
    native = _native_day_counts(native_sidecar, root=root)
    if native.empty:
        requirements["native_snapshots"] = 0
    else:
        requirements = requirements.merge(native, on=["symbol", "utc_day"], how="left")
        requirements["native_snapshots"] = requirements["native_snapshots"].fillna(0).astype(int)
    requirements["native_coverage"] = requirements["native_snapshots"].gt(0)
    required_days = sorted(requirements["utc_day"].unique().tolist())
    required_symbols = sorted(requirements["symbol"].unique().tolist())
    covered_pairs = int(requirements["native_coverage"].sum())
    summary = {
        "schema": "native_l2_backfill_request_v1",
        "status": "RESEARCH_ONLY_BACKFILL_REQUEST_NO_MODEL",
        "source_allowlist": list(source_allowlist),
        "proxy_sources_excluded": ["local_ohlcv_summary"],
        "max_staleness_hours": float(max_staleness_hours),
        "candidate_rows_represented": int(requirements["candidate_rows"].sum()) if not requirements.empty else 0,
        "candidate_symbol_count": len(required_symbols),
        "candidate_day_count": len(required_days),
        "candidate_symbol_day_pairs": int(len(requirements)),
        "currently_covered_symbol_day_pairs": covered_pairs,
        "missing_symbol_day_pairs": int(len(requirements) - covered_pairs),
        "required_candidate_min_day": required_days[0] if required_days else None,
        "required_candidate_max_day": required_days[-1] if required_days else None,
        "native_sidecar": str(native_sidecar) if native_sidecar is not None else None,
        "labels_used": False,
        "scores_used": False,
        "costs_used": False,
        "model_fitted": False,
        "promotion_eligible": False,
    }
    return requirements.sort_values(["utc_day", "symbol"], kind="stable").reset_index(drop=True), summary
