#!/usr/bin/env python3
"""Materialize a genuine-OOS base+meta source for failure-taxonomy sensitivity.

This is deliberately separate from the three-year frozen diagnostic backcast.
It converts the saved top-30 meta OOS ledger into the taxonomy source contract
without manufacturing a meta probability from future outcomes.  The final meta
EV rank is converted to a clean-execution probability with a side x archetype,
rank-bin map that uses only earlier UTC days; early rows fall back to the base
probability.  The resulting source can therefore support a strict, limited
base+meta sensitivity analysis without changing the descriptive taxonomy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


DEFAULT_INPUT = Path(
    "data_perp/reports/meta_v9_recovery_20260717/"
    "residual_state_mda95_hier_newaegmm_hpo150_v1/oos_predictions.parquet"
)
DEFAULT_OUTPUT = Path(
    "data_perp/reports/failure_taxonomy_strict_oos_source_20260719_v1"
)

KEYS = ("__ts__", "__symbol__", "side_name", "archetype_policy_key")
OUTCOME_TOKENS = (
    "clean_exec",
    "ev_after",
    "dirty_positive",
    "bad_mae",
    "timeout",
    "stop",
    "outcome",
    "target",
    "realized",
    "return",
    "utility",
)
REQUIRED = {
    *KEYS,
    "score_base",
    "score_base_residual_ev_rank_train_reference",
    "clean_exec",
    "ev_after_1pct",
    "dirty_positive",
    "full_path_bad_mae_1r",
    "timeout",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _observable_columns(columns: list[str], input_path: Path) -> list[str]:
    """Retain point-in-time numeric context and model outputs only."""

    contract_path = input_path.parent / "manifest.json"
    contract_features: set[str] = set()
    if contract_path.exists():
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
        for features in (payload.get("feature_contract") or {}).values():
            contract_features.update(str(name) for name in features)
    # Preserve core model-state fields even if an older manifest does not name
    # them explicitly. The saved contract remains the primary memory bound.
    contract_features.update(
        {
            "score",
            "score_base_ev_mapped",
            "score_base_ev_residual_expert",
            "score_base_ev_residual_expert_hier_mapped",
            "meta_residual_expert_delta_ev",
            "score_base_ev_rank_train_reference",
            "gmm_ood_score",
            "AE_reconstruction_error",
        }
    )
    retained: list[str] = []
    for name in columns:
        normalized = name.casefold()
        if name in KEYS or name in REQUIRED:
            continue
        if name not in contract_features:
            continue
        if any(token in normalized for token in OUTCOME_TOKENS):
            continue
        retained.append(name)
    return retained


def _causal_clean_probability(
    frame: pd.DataFrame,
    *,
    rank_column: str,
    base_column: str,
    bins: int,
    shrinkage: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return side/archetype probabilities using only prior UTC-day outcomes.

    All rows of a day receive predictions before that day contributes counts.
    A rank-bin global rate supplies a stable parent prior; local support then
    shrinks toward it.  The base score is only a cold-start fallback.
    """

    work = frame.loc[:, ["day", "side_name", "archetype_policy_key", rank_column, base_column, "clean_exec"]].copy()
    work = work.rename(columns={rank_column: "rank_value", base_column: "base_value"})
    rank = pd.to_numeric(work["rank_value"], errors="coerce").clip(0.0, 1.0)
    work["rank_bin"] = np.minimum((rank * bins).fillna(0.0).astype(int), bins - 1)
    work["clean_exec"] = pd.to_numeric(work["clean_exec"], errors="coerce").clip(0.0, 1.0)
    work["row_index"] = np.arange(len(work), dtype=np.int64)
    probabilities = np.full(len(work), np.nan, dtype=np.float32)
    supports = np.zeros(len(work), dtype=np.float32)
    local_counts: dict[tuple[str, str, int], float] = {}
    local_hits: dict[tuple[str, str, int], float] = {}
    global_counts: dict[int, float] = {}
    global_hits: dict[int, float] = {}
    global_total_count = 0.0
    global_total_hits = 0.0

    for _, day_rows in work.groupby("day", sort=True, observed=True):
        # Score every row before incorporating this day's resolved outcomes.
        for row in day_rows.itertuples(index=False):
            key = (str(row.side_name), str(row.archetype_policy_key), int(row.rank_bin))
            local_count = local_counts.get(key, 0.0)
            local_hit = local_hits.get(key, 0.0)
            bucket_count = global_counts.get(int(row.rank_bin), 0.0)
            bucket_hit = global_hits.get(int(row.rank_bin), 0.0)
            fallback = float(np.clip(row.base_value, 0.0, 1.0))
            if bucket_count >= 10.0:
                parent = bucket_hit / bucket_count
            elif global_total_count >= 10.0:
                parent = global_total_hits / global_total_count
            else:
                parent = fallback
            probability = (local_hit + shrinkage * parent) / (local_count + shrinkage)
            probabilities[int(row.row_index)] = np.float32(np.clip(probability, 0.0, 1.0))
            supports[int(row.row_index)] = np.float32(local_count)
        for row in day_rows.itertuples(index=False):
            outcome = float(row.clean_exec)
            if not np.isfinite(outcome):
                continue
            key = (str(row.side_name), str(row.archetype_policy_key), int(row.rank_bin))
            local_counts[key] = local_counts.get(key, 0.0) + 1.0
            local_hits[key] = local_hits.get(key, 0.0) + outcome
            rank_bin = int(row.rank_bin)
            global_counts[rank_bin] = global_counts.get(rank_bin, 0.0) + 1.0
            global_hits[rank_bin] = global_hits.get(rank_bin, 0.0) + outcome
            global_total_count += 1.0
            global_total_hits += outcome
    return probabilities, supports


def _top_fraction_by_timestamp(
    frame: pd.DataFrame,
    score_column: str,
    fraction: float,
) -> pd.Series:
    score = pd.to_numeric(frame[score_column], errors="coerce")
    # Stable average ranks make ties explicit and avoid arbitrary row order.
    percentile = score.groupby(frame["__ts__"], observed=True).rank(
        method="average", pct=True
    )
    return percentile.gt(1.0 - float(fraction)) & score.notna()


def materialize(
    input_path: Path,
    output: Path,
    *,
    monitor_fraction: float = 0.10,
    rank_bins: int = 20,
    local_shrinkage: float = 25.0,
) -> dict[str, Any]:
    schema = pq.ParquetFile(input_path).schema_arrow
    available = set(schema.names)
    missing = sorted(REQUIRED.difference(available))
    if missing:
        raise KeyError(f"Strict OOS meta ledger is missing required columns: {missing}")
    observable = _observable_columns(schema.names, input_path)
    frame = pd.read_parquet(input_path, columns=sorted(REQUIRED))
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    frame = frame.loc[frame["__ts__"].notna()].copy()
    frame["day"] = frame["__ts__"].dt.floor("D")
    frame = frame.sort_values(["__ts__", "__symbol__", "side_name"], kind="stable").reset_index(drop=True)
    final_rank = "score_base_residual_ev_rank_train_reference"
    frame["historical_rank"] = pd.to_numeric(frame[final_rank], errors="coerce")
    frame["base_score"] = pd.to_numeric(frame["score_base"], errors="coerce")
    probability, support = _causal_clean_probability(
        frame,
        rank_column=final_rank,
        base_column="score_base",
        bins=int(rank_bins),
        shrinkage=float(local_shrinkage),
    )
    # The residual taxonomy consumes ``hit_probability`` as its expected clean
    # outcome.  This source represents the base+meta contract, so it must use
    # the causal probability inferred from the final meta rank.  ``base_score``
    # remains a separate diagnostic and only supplies the documented cold-start
    # prior inside _causal_clean_probability.
    frame["score_meta_base_soft_label"] = probability
    frame["meta_probability_local_support"] = support
    frame["hit_probability"] = probability
    frame["exec_margin"] = pd.to_numeric(frame["ev_after_1pct"], errors="coerce")
    frame["first_touch_bad_mae_1r"] = np.nan
    frame["stop_or_adverse"] = pd.to_numeric(
        frame["full_path_bad_mae_1r"], errors="coerce"
    )
    frame["outcomes_available"] = True
    frame["evidence_scope"] = "genuine_base_meta_oos_limited_sensitivity"
    frame["selected_top30"] = True
    frame["selected_for_monitor"] = _top_fraction_by_timestamp(
        frame, final_rank, monitor_fraction
    )

    output.mkdir(parents=True, exist_ok=True)
    candidate_root = output / "candidate_shards"
    candidate_root.mkdir(exist_ok=True)
    source_columns = [
        *KEYS,
        "hit_probability",
        "clean_exec",
        "ev_after_1pct",
        "exec_margin",
        "dirty_positive",
        "base_score",
        "score_meta_base_soft_label",
        "historical_rank",
        "full_path_bad_mae_1r",
        "first_touch_bad_mae_1r",
        "timeout",
        "stop_or_adverse",
        "selected_for_monitor",
        "outcomes_available",
        "evidence_scope",
        "selected_top30",
        "meta_probability_local_support",
        *observable,
    ]
    source_columns = list(dict.fromkeys(source_columns))
    for period, group in frame.groupby(frame["__ts__"].dt.strftime("%Y%m"), sort=True):
        start = group["__ts__"].min()
        end = group["__ts__"].max() + pd.Timedelta(hours=1)
        feature_part = pd.read_parquet(
            input_path,
            columns=[*KEYS, *observable],
            filters=[("__ts__", ">=", start), ("__ts__", "<", end)],
        )
        joined = group.merge(
            feature_part,
            on=list(KEYS),
            how="left",
            validate="one_to_one",
            sort=False,
        )
        missing_columns = sorted(set(source_columns).difference(joined.columns))
        if missing_columns:
            raise KeyError(f"Materialized strict OOS source lacks columns: {missing_columns}")
        joined.loc[:, source_columns].to_parquet(
            candidate_root / f"candidates_{period}.parquet",
            index=False,
            compression="zstd",
        )
    manifest = {
        "schema": "strict_oos_base_meta_failure_taxonomy_source_v1",
        "evidence_scope": "genuine_base_meta_oos_limited_sensitivity",
        "input": str(input_path.resolve()),
        "start": frame["__ts__"].min(),
        "end": frame["__ts__"].max(),
        "rows": int(len(frame)),
        "days": int(frame["day"].nunique()),
        "symbols": int(frame["__symbol__"].nunique()),
        "selected_for_monitor_rows": int(frame["selected_for_monitor"].sum()),
        "monitor_fraction": float(monitor_fraction),
        "monitor_rank_column": final_rank,
        "candidate_stream": "fixed top-30 base candidate stream represented by meta OOS ledger",
        "meta_probability_contract": {
            "source": "score_base_residual_ev_rank_train_reference",
            "method": "prior-day side_x_archetype rank-bin empirical clean mapping",
            "output_column": "hit_probability",
            "base_score_role": "separate diagnostic and cold-start prior only",
            "rank_bins": int(rank_bins),
            "local_shrinkage": float(local_shrinkage),
            "cold_start": "base_score",
            "same_day_outcomes_used": False,
        },
        "first_touch_bad_mae_contract": "unavailable in meta OOS ledger; intentionally NaN",
        "observable_columns": [name for name in source_columns if name not in REQUIRED],
        "three_year_claim": False,
        "known_limitations": [
            "This is a limited April-July 2026 OOS sensitivity, not a three-year OOS replacement.",
            "Meta clean probability is causally calibrated from prior OOS days because the saved meta output is an EV rank, not a probability.",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--monitor-fraction", type=float, default=0.10)
    parser.add_argument("--rank-bins", type=int, default=20)
    parser.add_argument("--local-shrinkage", type=float, default=25.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            materialize(
                args.input,
                args.output,
                monitor_fraction=args.monitor_fraction,
                rank_bins=args.rank_bins,
                local_shrinkage=args.local_shrinkage,
            ),
            default=_json_default,
        )
    )
