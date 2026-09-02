#!/usr/bin/env python3
"""Materialize decision-time context for causal global-book transition labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from scripts.materialize_canonical_economic_conversion_transition_context import (
        COMPACT_REGIME_COLUMNS,
        CORE_MARKET_COLUMNS,
        TRANSITION_COLUMNS,
        sha256,
    )
    from scripts.materialize_canonical_global_book_conversion_transition_labels import (
        GLOBAL_EV_BANDS,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from materialize_canonical_economic_conversion_transition_context import (
        COMPACT_REGIME_COLUMNS,
        CORE_MARKET_COLUMNS,
        TRANSITION_COLUMNS,
        sha256,
    )
    from materialize_canonical_global_book_conversion_transition_labels import (
        GLOBAL_EV_BANDS,
    )


ROOT = Path(__file__).resolve().parents[1]
PANEL_SOURCE = (
    ROOT / "data_perp/artifacts/canonical_opportunity_payoff_trust_panel_20260729_v2"
)
LABEL_SOURCE = (
    ROOT
    / "data_perp/artifacts/"
    "canonical_global_book_conversion_transition_labels_20260729_v1"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data_perp/artifacts/canonical_global_book_conversion_context_20260729_v1"
)
SCHEMA = "canonical_global_book_conversion_context_v1"

BASE_SCORE_CONTEXT = (
    "base_oof_score",
    "base_rank_pct_timestamp_global",
    "base_score_z_timestamp_global",
    "base_group_rows_timestamp_global",
    "base_margin_to_top40_cutoff",
)
PANEL_CONTEXT = (
    *BASE_SCORE_CONTEXT,
    *CORE_MARKET_COLUMNS,
    *TRANSITION_COLUMNS,
    *COMPACT_REGIME_COLUMNS,
)
COORDINATE_COLUMNS = (
    "mapped_direct_net",
    "map_reference_rows",
    "map_side_reference_rows",
    "map_cell_reference_rows",
    "causal_global_mapped_ev_percentile",
    "causal_global_mapped_ev_reference_rows",
    "causal_global_mapped_ev_cutoff_p90",
    "causal_global_mapped_ev_margin_to_p90",
)
TRAILING_HOURS = (3, 12)
PROHIBITED_TOKENS = (
    "target",
    "label",
    "outcome",
    "opportunity_",
    "exit",
    "mfe",
    "mae",
    "realized",
    "execution_net",
    "execution_gross",
    "execution_cost",
    "execution_return",
    "execution_policy",
    "wait_action",
    "target_price",
)


def _safe(value: Any) -> Any:
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _manifest(root: Path, schema: str) -> tuple[dict[str, Any], dict[str, str]]:
    manifest = root / "manifest.json"
    sidecar = root / "manifest.sha256"
    if not manifest.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"immutable context source is incomplete: {root}")
    if sidecar.read_text(encoding="utf-8").split()[0] != sha256(manifest):
        raise ValueError(f"context source manifest checksum mismatch: {root}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("schema") != schema:
        raise ValueError(f"unexpected source schema at {root}: {payload.get('schema')}")
    return payload, {
        str(manifest): sha256(manifest),
        str(sidecar): sha256(sidecar),
    }


def _source_hashes(
    panel_source: Path, label_source: Path
) -> tuple[dict[str, Any], dict[str, str]]:
    panel_manifest, panel_hashes = _manifest(
        panel_source, "canonical_opportunity_payoff_trust_panel_v2"
    )
    label_manifest, label_hashes = _manifest(
        label_source, "canonical_global_book_conversion_transition_labels_v1"
    )
    paths = (
        panel_source / "panel.parquet",
        label_source / "global_book_transition_labels.parquet",
        label_source / "global_ev_band_transition_labels.parquet",
        label_source / "candidate_global_mapped_ev_coordinates.parquet",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"global-book context source lacks files: {missing}")
    expected = label_manifest.get("outputs_sha256", {})
    for path in paths[1:]:
        if expected.get(path.name) != sha256(path):
            raise ValueError(f"global-book label output hash mismatch: {path.name}")
    return (
        {"panel": panel_manifest, "labels": label_manifest},
        {
            **panel_hashes,
            **label_hashes,
            **{str(path): sha256(path) for path in paths},
        },
    )


def _context_name(column: str, scope: str = "current") -> str:
    return f"context__{scope}__{column.strip('_')}__mean"


def global_book_feature_columns() -> tuple[str, ...]:
    current = (
        "context__current_population_support",
        "context__current_coordinate_available_support",
        "context__current_coordinate_available_share",
        "context__current_mapped_score_std",
        "context__current_above_causal_p90_share",
        "context__current_long_share",
        "context__current_unique_assets",
        "context__current_largest_asset_share",
        *(
            f"context__current__band_{band}_share"
            for band in GLOBAL_EV_BANDS
        ),
        *(_context_name(column) for column in (*PANEL_CONTEXT, *COORDINATE_COLUMNS)),
    )
    trailing: list[str] = []
    for hours in TRAILING_HOURS:
        scope = f"trailing_{hours}h"
        trailing.extend(
            (
                f"context__{scope}__population_support",
                f"context__{scope}__coordinate_available_support",
                f"context__{scope}__coordinate_available_share",
                f"context__{scope}__mapped_score_mean",
                f"context__{scope}__mapped_score_std",
                f"context__{scope}__long_share",
                f"context__{scope}__above_causal_p90_share",
                *(
                    f"context__{scope}__band_{band}_share"
                    for band in GLOBAL_EV_BANDS
                ),
            )
        )
    return (*current, *trailing)


def band_feature_columns() -> tuple[str, ...]:
    current = (
        "context__global_common_ev_band_ordinal",
        "context__current_band_support",
        "context__current_band_long_share",
        "context__current_band_unique_assets",
        *(_context_name(column, "current_band") for column in (*PANEL_CONTEXT, *COORDINATE_COLUMNS)),
    )
    trailing: list[str] = []
    for hours in TRAILING_HOURS:
        scope = f"trailing_{hours}h_band"
        trailing.extend(
            (
                f"context__{scope}__support",
                f"context__{scope}__mapped_score_mean",
                f"context__{scope}__mapped_score_std",
                f"context__{scope}__long_share",
            )
        )
    return (*current, *trailing)


def _validate_feature_surface(columns: Iterable[str]) -> None:
    columns = tuple(columns)
    bad = [
        column
        for column in columns
        if any(token in column.lower() for token in PROHIBITED_TOKENS)
    ]
    if bad:
        raise ValueError(f"noncausal field entered global-book context: {bad}")
    if len(columns) != len(set(columns)):
        raise ValueError("global-book context feature contract has duplicates")


def _normalise_candidates(
    panel: pd.DataFrame, coordinates: pd.DataFrame
) -> pd.DataFrame:
    required = {
        "candidate_id",
        "side_name",
        "__symbol__",
        "__ts__",
        "execution_decision_utc",
        *PANEL_CONTEXT,
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"canonical panel lacks global-book context: {missing}")
    panel = panel.loc[:, list(required)].copy()
    panel["candidate_id"] = panel["candidate_id"].astype(str)
    panel["side_name"] = panel["side_name"].astype(str).str.lower()
    panel["__symbol__"] = panel["__symbol__"].astype(str)
    panel["__ts__"] = pd.to_datetime(
        panel["__ts__"], utc=True, errors="raise"
    )
    panel["execution_decision_utc"] = pd.to_datetime(
        panel["execution_decision_utc"], utc=True, errors="raise"
    )
    if not panel["execution_decision_utc"].eq(
        panel["__ts__"] + pd.Timedelta(hours=1)
    ).all():
        raise ValueError(
            "canonical panel violates signal-time plus one-hour decision identity"
        )
    coordinates = coordinates.copy()
    coordinates["candidate_id"] = coordinates["candidate_id"].astype(str)
    coordinates["execution_decision_utc"] = pd.to_datetime(
        coordinates["execution_decision_utc"], utc=True, errors="raise"
    )
    if panel["candidate_id"].duplicated().any() or coordinates["candidate_id"].duplicated().any():
        raise ValueError("global-book candidate context identity is not unique")
    joined = panel.merge(
        coordinates,
        on=["candidate_id", "execution_decision_utc"],
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(coordinates):
        raise ValueError("canonical panel does not cover every mapped coordinate")
    for column in (*PANEL_CONTEXT, *COORDINATE_COLUMNS):
        joined[column] = pd.to_numeric(joined[column], errors="coerce")
        if np.isinf(joined[column].to_numpy(float, na_value=np.nan)).any():
            raise ValueError(f"global-book context contains infinity: {column}")
    joined["_coordinate_available"] = (
        joined["causal_global_mapped_ev_percentile"].notna()
        & joined["causal_global_mapped_ev_band"].notna()
        & joined["causal_global_mapped_ev_cutoff_p90"].notna()
    )
    return joined.sort_values(
        ["execution_decision_utc", "candidate_id"], kind="stable"
    ).reset_index(drop=True)


def _geometry(
    population: pd.DataFrame,
    *,
    include_band_shares: bool = True,
) -> dict[str, float]:
    """Causal population geometry; never performs timestamp-local top-k."""

    if population.empty:
        empty = {
            "population_support": 0,
            "coordinate_available_support": 0,
            "coordinate_available_share": np.nan,
            "mapped_score_mean": np.nan,
            "mapped_score_std": np.nan,
            "long_share": np.nan,
            "above_causal_p90_share": np.nan,
        }
        if include_band_shares:
            empty.update(
                {f"band_{band}_share": np.nan for band in GLOBAL_EV_BANDS}
            )
        return empty
    available = population.loc[population["_coordinate_available"]]
    result = {
        "population_support": int(len(population)),
        "coordinate_available_support": int(len(available)),
        "coordinate_available_share": float(
            population["_coordinate_available"].mean()
        ),
        "mapped_score_mean": float(population["mapped_direct_net"].mean()),
        "mapped_score_std": float(population["mapped_direct_net"].std(ddof=0)),
        "long_share": float(population["side_name"].eq("long").mean()),
        "above_causal_p90_share": float(
            (
                available["mapped_direct_net"]
                >= available["causal_global_mapped_ev_cutoff_p90"]
            ).mean()
        )
        if len(available)
        else np.nan,
    }
    if include_band_shares:
        denominator = len(available)
        for band in GLOBAL_EV_BANDS:
            result[f"band_{band}_share"] = (
                float(
                    available["causal_global_mapped_ev_band"].eq(band).sum()
                    / denominator
                )
                if denominator
                else np.nan
            )
    return result


def _mean_fields(
    population: pd.DataFrame, columns: Iterable[str], scope: str
) -> dict[str, float]:
    return {
        _context_name(column, scope): float(
            pd.to_numeric(population[column], errors="coerce").mean()
        )
        if len(population)
        else np.nan
        for column in columns
    }


def _time_slice(
    candidates: pd.DataFrame,
    stamp_ns: np.ndarray,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Return the half-open decision-time slice without a full-frame mask."""

    left = int(np.searchsorted(stamp_ns, start.value, side="left"))
    right = int(np.searchsorted(stamp_ns, end.value, side="left"))
    return candidates.iloc[left:right]


def _time_point(
    candidates: pd.DataFrame,
    stamp_ns: np.ndarray,
    *,
    anchor: pd.Timestamp,
) -> pd.DataFrame:
    """Return all candidates at one exact decision timestamp."""

    left = int(np.searchsorted(stamp_ns, anchor.value, side="left"))
    right = int(np.searchsorted(stamp_ns, anchor.value, side="right"))
    return candidates.iloc[left:right]


def materialize_global_book_context(
    candidates: pd.DataFrame, label_keys: pd.DataFrame
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    stamp_ns = candidates["execution_decision_utc"].array.asi8
    cache: dict[int, dict[str, Any]] = {}
    for key in label_keys.itertuples(index=False):
        anchor = pd.Timestamp(key.cohort_anchor_utc)
        fraction = float(key.book_fraction)
        cache_key = anchor.value
        cached = cache.get(cache_key)
        if cached is None:
            current_population = _time_point(
                candidates, stamp_ns, anchor=anchor
            )
            geometry = _geometry(current_population)
            cached = {
                "context__current_population_support": geometry[
                    "population_support"
                ],
                "context__current_coordinate_available_support": geometry[
                    "coordinate_available_support"
                ],
                "context__current_coordinate_available_share": geometry[
                    "coordinate_available_share"
                ],
                "context__current_mapped_score_std": geometry[
                    "mapped_score_std"
                ],
                "context__current_above_causal_p90_share": geometry[
                    "above_causal_p90_share"
                ],
                "context__current_long_share": geometry["long_share"],
                "context__current_unique_assets": int(
                    current_population["__symbol__"].nunique()
                ),
                "context__current_largest_asset_share": float(
                    current_population["__symbol__"]
                    .value_counts(normalize=True)
                    .max()
                )
                if len(current_population)
                else np.nan,
                **{
                    f"context__current__band_{band}_share": geometry[
                        f"band_{band}_share"
                    ]
                    for band in GLOBAL_EV_BANDS
                },
                **_mean_fields(
                    current_population,
                    (*PANEL_CONTEXT, *COORDINATE_COLUMNS),
                    "current",
                ),
            }
            for hours in TRAILING_HOURS:
                scope = f"trailing_{hours}h"
                trailing_population = _time_slice(
                    candidates,
                    stamp_ns,
                    start=anchor - pd.Timedelta(hours=hours),
                    end=anchor,
                )
                trailing = _geometry(trailing_population)
                for name in (
                    "population_support",
                    "coordinate_available_support",
                    "coordinate_available_share",
                    "mapped_score_mean",
                    "mapped_score_std",
                    "long_share",
                    "above_causal_p90_share",
                    *(
                        f"band_{band}_share"
                        for band in GLOBAL_EV_BANDS
                    ),
                ):
                    cached[f"context__{scope}__{name}"] = trailing[name]
            cache[cache_key] = cached
        record: dict[str, Any] = {
            "cohort_anchor_utc": anchor,
            "horizon_hours": int(key.horizon_hours),
            "book_fraction": fraction,
            "label_audit__before_global_hour_complete_flag": bool(
                getattr(key, "before_global_hour_complete_flag", False)
            ),
            "label_audit__after_global_hour_complete_flag": bool(
                getattr(key, "after_global_hour_complete_flag", False)
            ),
            "label_audit__before_target_available_utc": getattr(
                key, "before_target_available_utc", pd.NaT
            ),
            "label_audit__after_target_available_utc": getattr(
                key, "after_target_available_utc", pd.NaT
            ),
            **cached,
        }
        records.append(record)
    output = pd.DataFrame.from_records(records)
    expected = global_book_feature_columns()
    if tuple(column for column in output if column.startswith("context__")) != expected:
        raise ValueError("global-book context feature order drifted")
    return output


def materialize_band_context(
    candidates: pd.DataFrame, label_keys: pd.DataFrame
) -> pd.DataFrame:
    ordinal = {band: index for index, band in enumerate(GLOBAL_EV_BANDS)}
    records: list[dict[str, Any]] = []
    stamp_ns = candidates["execution_decision_utc"].array.asi8
    cache: dict[tuple[int, str], dict[str, Any]] = {}
    for key in label_keys.itertuples(index=False):
        anchor = pd.Timestamp(key.cohort_anchor_utc)
        band = str(key.global_common_ev_band)
        cache_key = (anchor.value, band)
        cached = cache.get(cache_key)
        if cached is None:
            current_population = _time_point(
                candidates, stamp_ns, anchor=anchor
            )
            current = current_population.loc[
                current_population[
                    "causal_global_mapped_ev_band"
                ].eq(band)
            ]
            if len(current) and not current["_coordinate_available"].all():
                raise ValueError(
                    "global-band current context includes unavailable coordinates"
                )
            cached = {
                "context__current_band_support": int(len(current)),
                "context__current_band_long_share": float(
                    current["side_name"].eq("long").mean()
                )
                if len(current)
                else np.nan,
                "context__current_band_unique_assets": int(
                    current["__symbol__"].nunique()
                ),
                **_mean_fields(
                    current,
                    (*PANEL_CONTEXT, *COORDINATE_COLUMNS),
                    "current_band",
                ),
            }
            for hours in TRAILING_HOURS:
                scope = f"trailing_{hours}h_band"
                trailing_population = _time_slice(
                    candidates,
                    stamp_ns,
                    start=anchor - pd.Timedelta(hours=hours),
                    end=anchor,
                )
                trailing = trailing_population.loc[
                    trailing_population[
                        "causal_global_mapped_ev_band"
                    ].eq(band)
                ]
                if len(trailing) and not trailing["_coordinate_available"].all():
                    raise ValueError(
                        "global-band trailing context includes unavailable coordinates"
                    )
                cached[f"context__{scope}__support"] = int(len(trailing))
                cached[f"context__{scope}__mapped_score_mean"] = float(
                    trailing["mapped_direct_net"].mean()
                )
                cached[f"context__{scope}__mapped_score_std"] = float(
                    trailing["mapped_direct_net"].std(ddof=0)
                )
                cached[f"context__{scope}__long_share"] = float(
                    trailing["side_name"].eq("long").mean()
                )
            cache[cache_key] = cached
        record: dict[str, Any] = {
            "cohort_anchor_utc": anchor,
            "horizon_hours": int(key.horizon_hours),
            "global_common_ev_band": band,
            "label_audit__before_global_hour_complete_flag": bool(
                getattr(key, "before_global_hour_complete_flag", False)
            ),
            "label_audit__after_global_hour_complete_flag": bool(
                getattr(key, "after_global_hour_complete_flag", False)
            ),
            "label_audit__before_target_available_utc": getattr(
                key, "before_target_available_utc", pd.NaT
            ),
            "label_audit__after_target_available_utc": getattr(
                key, "after_target_available_utc", pd.NaT
            ),
            "context__global_common_ev_band_ordinal": float(ordinal[band]),
            **cached,
        }
        records.append(record)
    output = pd.DataFrame.from_records(records)
    expected = band_feature_columns()
    if tuple(column for column in output if column.startswith("context__")) != expected:
        raise ValueError("global-band context feature order drifted")
    return output


def plan(panel_source: Path, label_source: Path, output: Path) -> dict[str, Any]:
    manifests, hashes = _source_hashes(panel_source, label_source)
    return {
        "action": "PLAN_ONLY_NO_MATERIALIZATION",
        "schema": SCHEMA,
        "panel_source": str(panel_source),
        "label_source": str(label_source),
        "output": str(output),
        "source_sha256": hashes,
        "source_panel_identity_sha256": manifests["panel"].get("identity_sha256"),
        "global_book_features": list(global_book_feature_columns()),
        "global_band_features": list(band_feature_columns()),
        "contracts": {
            "anchor": "execution-decision UTC",
            "population_geometry": "current and trailing causal mapped-EV population summaries only; no timestamp-local top-k selection",
            "trailing": "past candidates in [anchor-3h,anchor) and [anchor-12h,anchor); no outcomes",
            "forbidden": "all outcomes, labels, exit/MFE/MAE, target-price and wait-action fields",
            "task_isolation": "book_fraction and horizon_hours are immutable keys, never model features; train one declared task at a time",
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    panel_source = Path(args.panel_source)
    label_source = Path(args.label_source)
    output = Path(args.output_dir)
    if args.plan_only:
        return plan(panel_source, label_source, output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output}")
    manifests, hashes = _source_hashes(panel_source, label_source)
    _validate_feature_surface(global_book_feature_columns())
    _validate_feature_surface(band_feature_columns())
    panel = pd.read_parquet(
        panel_source / "panel.parquet",
        columns=[
            "candidate_id",
            "side_name",
            "__symbol__",
            "__ts__",
            "execution_decision_utc",
            *PANEL_CONTEXT,
        ],
    )
    coordinates = pd.read_parquet(
        label_source / "candidate_global_mapped_ev_coordinates.parquet"
    )
    candidates = _normalise_candidates(panel, coordinates)
    book_labels = pd.read_parquet(
        label_source / "global_book_transition_labels.parquet",
        columns=[
            "cohort_anchor_utc",
            "horizon_hours",
            "book_fraction",
            "before_global_hour_complete_flag",
            "after_global_hour_complete_flag",
            "before_target_available_utc",
            "after_target_available_utc",
        ],
    ).drop_duplicates()
    band_labels = pd.read_parquet(
        label_source / "global_ev_band_transition_labels.parquet",
        columns=[
            "cohort_anchor_utc",
            "horizon_hours",
            "global_common_ev_band",
            "before_global_hour_complete_flag",
            "after_global_hour_complete_flag",
            "before_target_available_utc",
            "after_target_available_utc",
        ],
    ).drop_duplicates()
    for frame in (book_labels, band_labels):
        frame["cohort_anchor_utc"] = pd.to_datetime(
            frame["cohort_anchor_utc"], utc=True, errors="raise"
        )
    book_context = materialize_global_book_context(candidates, book_labels)
    band_context = materialize_band_context(candidates, band_labels)
    if book_context.duplicated(
        ["cohort_anchor_utc", "horizon_hours", "book_fraction"]
    ).any():
        raise ValueError("global-book context identity is not one-to-one")
    if band_context.duplicated(
        ["cohort_anchor_utc", "horizon_hours", "global_common_ev_band"]
    ).any():
        raise ValueError("global-band context identity is not one-to-one")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    book_context.to_parquet(
        temporary / "global_book_context.parquet",
        index=False,
        compression="zstd",
    )
    band_context.to_parquet(
        temporary / "global_ev_band_context.parquet",
        index=False,
        compression="zstd",
    )
    manifest = {
        "schema": SCHEMA,
        "status": "IMMUTABLE_DECISION_TIME_GLOBAL_BOOK_CONTEXT",
        "source_artifacts_sha256": hashes,
        "source_panel_identity_sha256": manifests["panel"].get("identity_sha256"),
        "global_book_feature_columns": list(global_book_feature_columns()),
        "global_band_feature_columns": list(band_feature_columns()),
        "rows": {
            "global_book": int(len(book_context)),
            "global_band": int(len(band_context)),
            "mapped_candidates": int(len(candidates)),
        },
        "contracts": {
            "feature_surface": "mapping coordinates, score geometry and pre-entry market/regime context only",
            "population_geometry": "current/trailing mapped-score distribution and causal p90/band mass only; no timestamp-local or side-local top-k",
            "task_isolation": "book_fraction and horizon_hours are output keys only; downstream training must filter one declared task rather than use them as features",
            "training_admission": "require both complete-hour flags and actual after_target_available_utc strictly before the fold validation boundary",
            "coordinate_missingness": "book context exposes causal-coordinate availability support/share; band context fails closed to rows with an assigned available coordinate",
            "no_outcomes": "no execution outcome, transition label, exit, MFE, MAE, wait or target-price field",
        },
        "outputs_sha256": {
            path.name: sha256(path) for path in sorted(temporary.glob("*.parquet"))
        },
        "checksum_convention": "manifest.json is verified by detached manifest.sha256",
    }
    (temporary / "manifest.json").write_text(
        json.dumps(_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (temporary / "manifest.sha256").write_text(
        f"{sha256(temporary / 'manifest.json')}  manifest.json\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return {
        "output": str(output),
        "global_book_rows": int(len(book_context)),
        "global_band_rows": int(len(band_context)),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--panel-source", type=Path, default=PANEL_SOURCE)
    result.add_argument("--label-source", type=Path, default=LABEL_SOURCE)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--plan-only", action="store_true")
    return result


def main() -> None:
    print(json.dumps(_safe(run(parser().parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
