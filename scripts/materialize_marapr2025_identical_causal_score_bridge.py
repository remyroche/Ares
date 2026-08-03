#!/usr/bin/env python3
"""Build the identical-row base-to-execution-EV causal mapping bridge.

The five score layers use the same exact-policy rows and the same daily
prior-resolved mapping support.  This runner fits only two predeclared pooled
isotonic diagnostic maps (raw base and raw direct q25); it does not fit or
select an alpha/residual/direct model, threshold, action, or policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from scripts.materialize_source_separated_ic_ev_waterfall import (
    cutoff_ties,
    full_ic,
    response_20bin,
    safe,
    score_compression,
    sha256,
    stable_top,
    tail_metrics,
)
from scripts.run_cross_era_tail_payoff_challenger import chronological_folds


ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
SOURCE_FAMILY = "marapr2025_identical_causal_score_bridge"
EXPECTED_RAW_ROWS = 140_682
EXPECTED_COMMON_ROWS = 136_074
MAP_WINDOW_DAYS = 21
MAP_MINIMUM_ROWS = 2_000
TOP_FRACTIONS = (0.01, 0.05, 0.10, 0.20)
BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_SEED = 20260730

DEFAULT_RAW = ROOT / (
    "data_perp/artifacts/marapr2025_all_score_ic_ev_waterfall_20260730_v1/"
    "all_score_waterfall.parquet"
)
DEFAULT_RAW_MANIFEST = DEFAULT_RAW.with_name("manifest.json")
DEFAULT_BASE_OOF = ROOT / (
    "data_perp/artifacts/febapr2025_canonical_base_oof_20260727_v1/"
    "oof_predictions.parquet"
)
DEFAULT_BASE_MANIFEST = DEFAULT_BASE_OOF.with_name("manifest.json")
DEFAULT_RESIDUAL_OOF = ROOT / (
    "data_perp/artifacts/febapr2025_canonical_residual_oof_20260727_v1/"
    "oof_predictions.parquet"
)
DEFAULT_RESIDUAL_MANIFEST = DEFAULT_RESIDUAL_OOF.with_name("manifest.json")
DEFAULT_DIRECT = ROOT / (
    "data_perp/artifacts/cross_era_direct_net_quantile_challenger_20260730_v1/"
    "historical_oof_winner.parquet"
)
DEFAULT_DIRECT_MANIFEST = DEFAULT_DIRECT.with_name("manifest.json")
DEFAULT_DIRECT_FROZEN = DEFAULT_DIRECT.with_name("frozen_before_current_evaluation.json")
DEFAULT_DIRECT_DATASET = ROOT / (
    "data_perp/artifacts/cross_era_tail_payoff_dataset_20260730_v3/"
    "cross_era_tail_payoff_dataset.parquet"
)
DEFAULT_DIRECT_DATASET_MANIFEST = DEFAULT_DIRECT_DATASET.with_name("manifest.json")
DEFAULT_OUTPUT = ROOT / (
    "data_perp/artifacts/"
    "marapr2025_identical_causal_score_bridge_20260730_v1"
)

LAYERS: Mapping[str, str] = {
    "raw_base_alpha": "score_raw_base_alpha",
    "causal_mapped_base_ev": "score_causal_mapped_base_ev",
    "residual_expected_ev": "score_residual_expected_ev",
    "raw_direct_q25_ev": "score_raw_direct_q25_ev",
    "causal_mapped_direct_q25_ev": "score_causal_mapped_direct_q25_ev",
}
FLOW = (
    ("raw_base_alpha", "causal_mapped_base_ev"),
    ("causal_mapped_base_ev", "residual_expected_ev"),
    ("residual_expected_ev", "raw_direct_q25_ev"),
    ("raw_direct_q25_ev", "causal_mapped_direct_q25_ev"),
)


class BridgeError(RuntimeError):
    """Raised when an identical-row or temporal contract is not proven."""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalise(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = sorted(set(IDENTITY).difference(frame.columns))
    if missing:
        raise BridgeError(f"{name} missing identity fields: {missing}")
    result = frame.copy()
    result["candidate_id"] = result["candidate_id"].astype(str)
    result["side_name"] = result["side_name"].astype(str).str.lower()
    result["__symbol__"] = result["__symbol__"].astype(str)
    result["__ts__"] = pd.to_datetime(result["__ts__"], utc=True, errors="raise")
    if not result["side_name"].isin(("long", "short")).all():
        raise BridgeError(f"{name} has invalid sides")
    if result[list(IDENTITY)].isna().any().any():
        raise BridgeError(f"{name} has null identity")
    if result.duplicated(list(IDENTITY)).any():
        raise BridgeError(f"{name} has duplicate identity")
    return result


def _output_hash(
    manifest: Mapping[str, Any], name: str, path: Path
) -> None:
    record = manifest.get("outputs", {}).get(name)
    expected = record.get("sha256") if isinstance(record, Mapping) else record
    if str(expected) != sha256(path):
        raise BridgeError(f"manifest does not bind {path}")


def _identity_hash(frame: pd.DataFrame) -> str:
    values = frame.loc[:, list(IDENTITY)].copy()
    values["__ts__"] = pd.to_datetime(values["__ts__"], utc=True).astype(str)
    values = values.astype(str).sort_values(list(IDENTITY), kind="stable")
    return hashlib.sha256(values.to_csv(index=False).encode()).hexdigest()


def _validate_raw(raw: pd.DataFrame, expected_rows: int) -> pd.DataFrame:
    required = {
        *IDENTITY,
        "candidate_month",
        "execution_decision_utc",
        "execution_label_end_utc",
        "execution_gross_ev_12h",
        "execution_cost_return",
        "execution_net_ev_12h",
        "execution_mfe_return_12h",
        "execution_mae_return_12h",
        "execution_exit_class",
        "opportunity_gross_above_cost_0bps",
        "__first_touch_target_soft__",
        "score_base_alpha",
        "score_residual_expected_ev",
        "direct_q25_return",
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise BridgeError(f"raw waterfall missing: {missing}")
    result = _normalise(raw, "raw waterfall")
    if len(result) != int(expected_rows):
        raise BridgeError(f"raw rows {len(result)} != {expected_rows}")
    if (
        int(expected_rows) == EXPECTED_RAW_ROWS
        and set(result["candidate_month"].astype(str)) != {"2025-03", "2025-04"}
    ):
        raise BridgeError("raw waterfall is not exactly March-April 2025")
    for column in ("execution_decision_utc", "execution_label_end_utc"):
        result[column] = pd.to_datetime(result[column], utc=True, errors="raise")
    if not result["execution_decision_utc"].eq(
        result["__ts__"] + pd.Timedelta(hours=1)
    ).all():
        raise BridgeError("decision timestamp is not signal plus one hour")
    if not result["execution_label_end_utc"].eq(
        result["execution_decision_utc"] + pd.Timedelta(hours=12)
    ).all():
        raise BridgeError("execution labels are not exact 12h")
    if not np.allclose(
        result["execution_gross_ev_12h"] - result["execution_cost_return"],
        result["execution_net_ev_12h"],
        rtol=0.0,
        atol=1e-7,
    ):
        raise BridgeError("gross-cost-net accounting fails")
    return result


def direct_fold_membership(dataset: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the exact old-March/old-April chronological OOF rows."""

    source = _normalise(dataset, "direct training dataset")
    source["label_resolution_utc"] = pd.to_datetime(
        source["label_resolution_utc"], utc=True, errors="raise"
    )
    parts: list[pd.DataFrame] = []
    for fold in chronological_folds(source):
        if fold.start >= pd.Timestamp("2025-03-01T00:00:00Z") and fold.end <= pd.Timestamp(
            "2025-05-01T00:00:00Z"
        ):
            valid = source.iloc[fold.valid].loc[
                :, [*IDENTITY, "label_resolution_utc", "era"]
            ].copy()
            valid["direct_oof_fold"] = fold.name
            valid["direct_fit_cutoff_utc"] = fold.start
            parts.append(valid)
    if len(parts) != 2:
        raise BridgeError("did not reconstruct exactly March and April direct folds")
    result = pd.concat(parts, ignore_index=True)
    if len(result) != EXPECTED_RAW_ROWS:
        raise BridgeError(
            f"direct fold membership rows {len(result)} != {EXPECTED_RAW_ROWS}"
        )
    return result


def causal_pooled_maps(
    frame: pd.DataFrame,
    *,
    score_columns: Sequence[str],
    minimum_rows: int = MAP_MINIMUM_ROWS,
    window_days: int = MAP_WINDOW_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit predeclared daily pooled maps on identical prior-resolved rows."""

    result = frame.copy()
    result["map_available"] = False
    result["map_reference_rows"] = 0
    result["map_snapshot_utc"] = pd.Series(
        pd.NaT, index=result.index, dtype="datetime64[ns, UTC]"
    )
    for score in score_columns:
        result[f"causal_map__{score}"] = np.nan
    ts = pd.to_datetime(result["__ts__"], utc=True, errors="raise")
    resolved = pd.to_datetime(
        result["execution_label_end_utc"], utc=True, errors="raise"
    )
    net = pd.to_numeric(result["execution_net_ev_12h"], errors="raise")
    audits: list[dict[str, Any]] = []
    for day, indices in result.groupby(ts.dt.floor("D"), sort=True).groups.items():
        day = pd.Timestamp(day)
        reference = (
            ts.lt(day)
            & ts.ge(day - pd.Timedelta(days=int(window_days)))
            & resolved.lt(day)
            & np.isfinite(net)
        )
        for score in score_columns:
            reference &= np.isfinite(
                pd.to_numeric(result[score], errors="coerce")
            )
        reference_rows = int(reference.sum())
        available = reference_rows >= int(minimum_rows)
        positions = list(indices)
        result.loc[positions, "map_available"] = available
        result.loc[positions, "map_reference_rows"] = reference_rows
        result.loc[positions, "map_snapshot_utc"] = day
        reference_frame = result.loc[reference, [*IDENTITY]].copy()
        audit = {
            "snapshot_utc": day,
            "candidate_rows": int(len(positions)),
            "map_available": bool(available),
            "reference_rows": reference_rows,
            "minimum_reference_rows": int(minimum_rows),
            "window_days": int(window_days),
            "reference_identity_sha256": (
                _identity_hash(reference_frame) if reference_rows else None
            ),
            "reference_label_end_max_utc": (
                resolved.loc[reference].max() if reference_rows else pd.NaT
            ),
        }
        if available:
            for score in score_columns:
                model = IsotonicRegression(out_of_bounds="clip")
                model.fit(result.loc[reference, score], net.loc[reference])
                mapped = model.predict(result.loc[positions, score])
                result.loc[positions, f"causal_map__{score}"] = mapped
                audit[f"{score}__reference_unique_scores"] = int(
                    result.loc[reference, score].nunique()
                )
                audit[f"{score}__mapped_unique_scores"] = int(
                    np.unique(mapped).size
                )
        audits.append(audit)
    return result, pd.DataFrame.from_records(audits)


def build_bridge(
    raw: pd.DataFrame,
    base_oof: pd.DataFrame,
    residual_oof: pd.DataFrame,
    direct: pd.DataFrame,
    direct_membership: pd.DataFrame,
    *,
    expected_raw_rows: int = EXPECTED_RAW_ROWS,
    expected_common_rows: int | None = EXPECTED_COMMON_ROWS,
    minimum_rows: int = MAP_MINIMUM_ROWS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = _validate_raw(raw, expected_raw_rows)
    base = _normalise(base_oof, "base OOF")
    residual = _normalise(residual_oof, "residual OOF")
    direct = _normalise(direct, "direct OOF")
    membership = _normalise(direct_membership, "direct fold membership")

    base_required = {"base_oof_score", "__decision_ts__"}
    residual_required = {
        "base_oof_score",
        "residual_expected_ev",
        "residual_is_oof",
        "__decision_ts__",
    }
    direct_required = {
        "q25_net_bps",
        "execution_net_ev_12h",
        "label_resolution_utc",
        "era",
    }
    for name, source, required in (
        ("base", base, base_required),
        ("residual", residual, residual_required),
        ("direct", direct, direct_required),
    ):
        missing = sorted(required.difference(source.columns))
        if missing:
            raise BridgeError(f"{name} source missing: {missing}")

    base["__decision_ts__"] = pd.to_datetime(
        base["__decision_ts__"], utc=True, errors="raise"
    )
    residual["__decision_ts__"] = pd.to_datetime(
        residual["__decision_ts__"], utc=True, errors="raise"
    )
    direct["label_resolution_utc"] = pd.to_datetime(
        direct["label_resolution_utc"], utc=True, errors="raise"
    )
    membership["label_resolution_utc"] = pd.to_datetime(
        membership["label_resolution_utc"], utc=True, errors="raise"
    )
    residual = residual.loc[residual["residual_is_oof"].astype(bool)].copy()
    start, end = pd.Timestamp("2025-03-01T00:00:00Z"), pd.Timestamp(
        "2025-05-01T00:00:00Z"
    )
    base = base.loc[base["__ts__"].ge(start) & base["__ts__"].lt(end)]
    residual = residual.loc[
        residual["__ts__"].ge(start) & residual["__ts__"].lt(end)
    ]
    direct = direct.loc[direct["__ts__"].ge(start) & direct["__ts__"].lt(end)]

    joined = raw.merge(
        base.loc[:, [*IDENTITY, "base_oof_score", "__decision_ts__"]].rename(
            columns={
                "base_oof_score": "source_base_oof_score",
                "__decision_ts__": "base_decision_ts",
            }
        ),
        on=list(IDENTITY),
        how="left",
        validate="one_to_one",
    )
    joined = joined.merge(
        residual.loc[
            :,
            [
                *IDENTITY,
                "base_oof_score",
                "residual_expected_ev",
                "__decision_ts__",
            ],
        ].rename(
            columns={
                "base_oof_score": "residual_source_base_score",
                "residual_expected_ev": "source_residual_expected_ev",
                "__decision_ts__": "residual_decision_ts",
            }
        ),
        on=list(IDENTITY),
        how="left",
        validate="one_to_one",
    )
    joined = joined.merge(
        direct.loc[
            :,
            [
                *IDENTITY,
                "q25_net_bps",
                "execution_net_ev_12h",
                "label_resolution_utc",
                "era",
            ],
        ].rename(
            columns={
                "execution_net_ev_12h": "direct_execution_net",
                "label_resolution_utc": "direct_label_end",
                "era": "direct_era",
            }
        ),
        on=list(IDENTITY),
        how="left",
        validate="one_to_one",
    )
    joined = joined.merge(
        membership.loc[
            :,
            [
                *IDENTITY,
                "label_resolution_utc",
                "era",
                "direct_oof_fold",
                "direct_fit_cutoff_utc",
            ],
        ].rename(
            columns={
                "label_resolution_utc": "membership_label_end",
                "era": "membership_era",
            }
        ),
        on=list(IDENTITY),
        how="left",
        validate="one_to_one",
    )
    required_join = (
        "source_base_oof_score",
        "source_residual_expected_ev",
        "q25_net_bps",
        "direct_oof_fold",
    )
    if joined[list(required_join)].isna().any().any():
        raise BridgeError("OOF sources do not completely cover the raw bridge")
    exact_pairs = (
        ("score_base_alpha", "source_base_oof_score"),
        ("score_base_alpha", "residual_source_base_score"),
        ("score_residual_expected_ev", "source_residual_expected_ev"),
    )
    for left, right in exact_pairs:
        if not np.array_equal(
            joined[left].to_numpy(float), joined[right].to_numpy(float)
        ):
            raise BridgeError(f"score lineage mismatch: {left} vs {right}")
    if not np.allclose(
        joined["direct_q25_return"].to_numpy(float) * 1e4,
        joined["q25_net_bps"].to_numpy(float),
        rtol=0.0,
        atol=1e-10,
    ):
        raise BridgeError("direct raw-score lineage mismatch")
    if not joined["base_decision_ts"].eq(joined["execution_decision_utc"]).all():
        raise BridgeError("base score is not decision-time aligned")
    if not joined["residual_decision_ts"].eq(
        joined["execution_decision_utc"]
    ).all():
        raise BridgeError("residual score is not decision-time aligned")
    if not np.array_equal(
        joined["execution_net_ev_12h"].to_numpy(float),
        joined["direct_execution_net"].to_numpy(float),
    ):
        raise BridgeError("direct and canonical net labels differ")
    label_end = pd.to_datetime(
        joined["execution_label_end_utc"], utc=True, errors="raise"
    )
    if not label_end.equals(
        pd.to_datetime(joined["direct_label_end"], utc=True, errors="raise")
    ) or not label_end.equals(
        pd.to_datetime(joined["membership_label_end"], utc=True, errors="raise")
    ):
        raise BridgeError("direct and canonical label horizons differ")
    if not joined["direct_era"].astype(str).eq(
        joined["membership_era"].astype(str)
    ).all():
        raise BridgeError("direct OOF era differs from reconstructed membership")
    fit_cutoff = pd.to_datetime(
        joined["direct_fit_cutoff_utc"], utc=True, errors="raise"
    )
    if not joined["__ts__"].ge(fit_cutoff).all():
        raise BridgeError("direct OOF score precedes its validation fold")

    joined["score_raw_base_alpha"] = joined["score_base_alpha"].astype(float)
    joined["score_raw_direct_q25_ev"] = joined["q25_net_bps"].astype(float) / 1e4
    mapped, audit = causal_pooled_maps(
        joined,
        score_columns=("score_raw_base_alpha", "score_raw_direct_q25_ev"),
        minimum_rows=minimum_rows,
    )
    bridge = mapped.loc[mapped["map_available"]].copy()
    if expected_common_rows is not None and len(bridge) != int(expected_common_rows):
        raise BridgeError(
            f"joint mapped rows {len(bridge)} != {expected_common_rows}"
        )
    bridge["score_causal_mapped_base_ev"] = bridge[
        "causal_map__score_raw_base_alpha"
    ].astype(float)
    bridge["score_residual_expected_ev"] = bridge[
        "score_residual_expected_ev"
    ].astype(float)
    bridge["score_causal_mapped_direct_q25_ev"] = bridge[
        "causal_map__score_raw_direct_q25_ev"
    ].astype(float)
    if not np.isfinite(bridge.loc[:, list(LAYERS.values())].to_numpy(float)).all():
        raise BridgeError("non-finite bridge score")
    bridge["candidate_month"] = bridge["__ts__"].dt.strftime("%Y-%m")
    bridge["source_family"] = SOURCE_FAMILY
    keep = [
        *IDENTITY,
        "candidate_month",
        "execution_decision_utc",
        "execution_label_end_utc",
        "execution_gross_ev_12h",
        "execution_cost_return",
        "execution_net_ev_12h",
        "execution_mfe_return_12h",
        "execution_mae_return_12h",
        "execution_exit_class",
        "opportunity_gross_above_cost_0bps",
        "__first_touch_target_soft__",
        "direct_oof_fold",
        "direct_fit_cutoff_utc",
        "map_snapshot_utc",
        "map_reference_rows",
        "source_family",
        *LAYERS.values(),
    ]
    return (
        bridge.loc[:, keep]
        .sort_values(list(IDENTITY), kind="stable")
        .reset_index(drop=True),
        audit,
    )


def diagnostics(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    parts: dict[str, list[pd.DataFrame]] = {
        key: []
        for key in (
            "full_ic",
            "tails",
            "compression",
            "response_cells",
            "response_summary",
            "cutoff_ties",
        )
    }
    for score in LAYERS.values():
        parts["full_ic"].append(full_ic(frame, source_family=SOURCE_FAMILY, score=score))
        parts["tails"].append(
            tail_metrics(frame, source_family=SOURCE_FAMILY, score=score)
        )
        parts["compression"].append(
            score_compression(frame, source_family=SOURCE_FAMILY, score=score)
        )
        cells, summary = response_20bin(
            frame, source_family=SOURCE_FAMILY, score=score
        )
        parts["response_cells"].append(cells)
        parts["response_summary"].append(summary)
        parts["cutoff_ties"].append(
            cutoff_ties(frame, source_family=SOURCE_FAMILY, score=score)
        )
    return {
        name: pd.concat(values, ignore_index=True) for name, values in parts.items()
    }


def selection_books(frame: pd.DataFrame) -> pd.DataFrame:
    books: list[pd.DataFrame] = []
    for month, local in frame.groupby("candidate_month", sort=True, observed=True):
        for layer, score in LAYERS.items():
            for fraction in TOP_FRACTIONS:
                selected = stable_top(local, score, fraction)
                book = selected.loc[
                    :,
                    [
                        *IDENTITY,
                        "candidate_month",
                        "execution_gross_ev_12h",
                        "execution_cost_return",
                        "execution_net_ev_12h",
                    ],
                ].copy()
                book["layer"] = layer
                book["score_column"] = score
                book["fraction"] = float(fraction)
                book["selected_score"] = selected[score].to_numpy(float)
                books.append(book)
    return pd.concat(books, ignore_index=True)


def selection_flow(books: pd.DataFrame) -> pd.DataFrame:
    top = books.loc[np.isclose(books["fraction"], 0.10)]
    rows: list[dict[str, Any]] = []
    for month, local in top.groupby("candidate_month", sort=True, observed=True):
        by_layer = {
            layer: part.set_index(list(IDENTITY))
            for layer, part in local.groupby("layer", sort=False, observed=True)
        }
        for source, destination in FLOW:
            left, right = by_layer[source], by_layer[destination]
            common = left.index.intersection(right.index)
            union = left.index.union(right.index)
            rows.append(
                {
                    "candidate_month": str(month),
                    "source_layer": source,
                    "destination_layer": destination,
                    "source_rows": int(len(left)),
                    "destination_rows": int(len(right)),
                    "shared_rows": int(len(common)),
                    "source_retention": float(len(common) / len(left)),
                    "jaccard": float(len(common) / len(union)),
                    "source_mean_net_bps": float(
                        left["execution_net_ev_12h"].mean() * 1e4
                    ),
                    "destination_mean_net_bps": float(
                        right["execution_net_ev_12h"].mean() * 1e4
                    ),
                    "destination_minus_source_net_bps": float(
                        (
                            right["execution_net_ev_12h"].mean()
                            - left["execution_net_ev_12h"].mean()
                        )
                        * 1e4
                    ),
                }
            )
    return pd.DataFrame.from_records(rows)


def bootstrap_top10(books: pd.DataFrame) -> pd.DataFrame:
    top = books.loc[np.isclose(books["fraction"], 0.10)].copy()
    top["day"] = pd.to_datetime(top["__ts__"], utc=True).dt.floor("D")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    for month, local in top.groupby("candidate_month", sort=True, observed=True):
        days = np.array(sorted(local["day"].unique()))
        stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for layer, part in local.groupby("layer", sort=False, observed=True):
            grouped = (
                part.groupby("day", observed=True)["execution_net_ev_12h"]
                .agg(["sum", "count"])
                .reindex(days, fill_value=0)
            )
            stats[str(layer)] = (
                grouped["sum"].to_numpy(float),
                grouped["count"].to_numpy(float),
            )
        draws = {
            layer: np.empty(BOOTSTRAP_DRAWS, dtype=float) for layer in stats
        }
        for draw in range(BOOTSTRAP_DRAWS):
            sample = rng.integers(0, len(days), size=len(days))
            for layer, (sums, counts) in stats.items():
                draws[layer][draw] = sums[sample].sum() / counts[sample].sum() * 1e4
        baseline = draws["raw_base_alpha"]
        for layer, values in draws.items():
            delta = values - baseline
            observed = local.loc[
                local["layer"].eq(layer), "execution_net_ev_12h"
            ].mean() * 1e4
            rows.append(
                {
                    "candidate_month": str(month),
                    "layer": layer,
                    "bootstrap_unit": "UTC signal day",
                    "draws": BOOTSTRAP_DRAWS,
                    "mean_net_bps": float(observed),
                    "mean_net_ci_low_bps": float(np.quantile(values, 0.025)),
                    "mean_net_ci_high_bps": float(np.quantile(values, 0.975)),
                    "delta_vs_raw_base_bps": float(np.mean(delta)),
                    "delta_ci_low_bps": float(np.quantile(delta, 0.025)),
                    "delta_ci_high_bps": float(np.quantile(delta, 0.975)),
                }
            )
    return pd.DataFrame.from_records(rows)


def run(
    *,
    raw_path: Path = DEFAULT_RAW,
    raw_manifest_path: Path = DEFAULT_RAW_MANIFEST,
    base_oof_path: Path = DEFAULT_BASE_OOF,
    base_manifest_path: Path = DEFAULT_BASE_MANIFEST,
    residual_oof_path: Path = DEFAULT_RESIDUAL_OOF,
    residual_manifest_path: Path = DEFAULT_RESIDUAL_MANIFEST,
    direct_path: Path = DEFAULT_DIRECT,
    direct_manifest_path: Path = DEFAULT_DIRECT_MANIFEST,
    direct_frozen_path: Path = DEFAULT_DIRECT_FROZEN,
    direct_dataset_path: Path = DEFAULT_DIRECT_DATASET,
    direct_dataset_manifest_path: Path = DEFAULT_DIRECT_DATASET_MANIFEST,
    output_dir: Path = DEFAULT_OUTPUT,
    expected_raw_rows: int = EXPECTED_RAW_ROWS,
    expected_common_rows: int = EXPECTED_COMMON_ROWS,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite sealed artifact: {output_dir}")

    raw_manifest = _read_json(raw_manifest_path)
    if raw_manifest.get("schema") != "marapr2025_all_score_ic_ev_waterfall_v1":
        raise BridgeError("unexpected raw waterfall manifest")
    _output_hash(raw_manifest, "all_score_waterfall", raw_path)
    base_manifest = _read_json(base_manifest_path)
    if base_manifest.get("schema") != "febapr2025_canonical_base_oof_v1":
        raise BridgeError("unexpected base OOF manifest")
    _output_hash(base_manifest, "oof_predictions.parquet", base_oof_path)
    residual_manifest = _read_json(residual_manifest_path)
    if residual_manifest.get("schema") != "febapr2025_canonical_residual_oof_v1":
        raise BridgeError("unexpected residual OOF manifest")
    if sha256(residual_oof_path) != str(
        residual_manifest.get("output_sha256")
        or residual_manifest.get("oof_sha256")
        or residual_manifest.get("outputs", {}).get("oof_predictions.parquet")
    ):
        raise BridgeError("residual manifest does not bind OOF predictions")
    direct_manifest = _read_json(direct_manifest_path)
    if direct_manifest.get("schema") != "cross_era_direct_net_quantile_challenger_v1":
        raise BridgeError("unexpected direct OOF manifest")
    _output_hash(direct_manifest, "historical_oof_winner", direct_path)
    if sha256(direct_frozen_path) != str(direct_manifest.get("frozen_state_sha256")):
        raise BridgeError("direct frozen-state hash mismatch")
    frozen = _read_json(direct_frozen_path)
    if (
        frozen.get("winner", {}).get("score_column") != "q25_net_bps"
        or frozen.get("winner", {}).get("mapped_column") != "mapped_q25_bps"
    ):
        raise BridgeError("direct winner is not the true q25 head")
    dataset_manifest = _read_json(direct_dataset_manifest_path)
    if dataset_manifest.get("schema") != "cross_era_tail_payoff_dataset_v3":
        raise BridgeError("unexpected direct training dataset manifest")
    _output_hash(dataset_manifest, "dataset", direct_dataset_path)
    if frozen.get("dataset", {}).get("sha256") != sha256(direct_dataset_path):
        raise BridgeError("direct frozen state does not bind training dataset")

    membership = direct_fold_membership(pd.read_parquet(direct_dataset_path))
    bridge, map_audit = build_bridge(
        pd.read_parquet(raw_path),
        pd.read_parquet(base_oof_path),
        pd.read_parquet(residual_oof_path),
        pd.read_parquet(direct_path),
        membership,
        expected_raw_rows=int(expected_raw_rows),
        expected_common_rows=int(expected_common_rows),
    )
    results = diagnostics(bridge)
    books = selection_books(bridge)
    results["selection_books"] = books
    results["selection_flow_top10"] = selection_flow(books)
    results["daily_bootstrap_top10"] = bootstrap_top10(books)
    results["causal_map_audit"] = map_audit
    results["identical_score_bridge"] = bridge

    stage = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    try:
        stage.mkdir(parents=True, exist_ok=False)
        outputs: dict[str, dict[str, Any]] = {}
        for name, frame in results.items():
            path = stage / f"{name}.parquet"
            frame.to_parquet(path, index=False, compression="zstd")
            outputs[name] = {
                "path": path.name,
                "rows": int(len(frame)),
                "sha256": sha256(path),
            }
        month_rows = (
            bridge["candidate_month"].value_counts().sort_index().astype(int).to_dict()
        )
        sources = {}
        for name, path, manifest_path in (
            ("raw_all_score", raw_path, raw_manifest_path),
            ("base_oof", base_oof_path, base_manifest_path),
            ("residual_oof", residual_oof_path, residual_manifest_path),
            ("direct_q25_oof", direct_path, direct_manifest_path),
            ("direct_training_dataset", direct_dataset_path, direct_dataset_manifest_path),
        ):
            sources[name] = {
                "path": str(path),
                "sha256": sha256(path),
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256(manifest_path),
            }
        sources["direct_q25_oof"]["frozen_state_path"] = str(direct_frozen_path)
        sources["direct_q25_oof"]["frozen_state_sha256"] = sha256(
            direct_frozen_path
        )
        manifest = {
            "schema": "marapr2025_identical_causal_score_bridge_v1",
            "status": "SEALED_REUSED_MONTH_DIAGNOSTIC_NO_PROMOTION",
            "promotion_eligible": False,
            "portfolio_replay_authorized": False,
            "population": {
                "raw_rows": int(expected_raw_rows),
                "joint_map_available_rows": int(len(bridge)),
                "rows_by_month": month_rows,
                "excluded_rows": int(expected_raw_rows - len(bridge)),
                "excluded_reason": (
                    "March 1-2 (4,608 rows) have fewer than 2000 "
                    "prior-resolved rows; "
                    "every score layer uses the identical remaining population"
                ),
                "february_status": (
                    "not comparable: strict residual and true direct-q25 OOF "
                    "identical-row lineages begin in March"
                ),
            },
            "contracts": {
                "identity": list(IDENTITY),
                "layers": dict(LAYERS),
                "mapping": (
                    "two predeclared pooled isotonic maps fit independently "
                    "from raw base and raw direct q25 to exact net"
                ),
                "mapping_reference": (
                    "same candidate population and exact reference identities; "
                    "21d window; execution_label_end_utc < UTC-day snapshot; "
                    "minimum 2000 rows; no side map"
                ),
                "selection": (
                    "one month-level pooled-global top 1/5/10/20 across "
                    "timestamps and sides; candidate-id ascending ties; no quotas"
                ),
                "economics": (
                    "same exact current-spread 1m deployed-policy 12h gross, "
                    "one explicit cost, and net labels"
                ),
                "model_fit_in_runner": False,
                "calibration_fit_in_runner": True,
                "calibration_selection_or_hpo": False,
                "evidence": (
                    "diagnostic on reused research months; direct-head HPO "
                    "selection means this is not an untouched final test"
                ),
            },
            "sources": sources,
            "outputs": outputs,
            "outputs_sha256": {
                record["path"]: record["sha256"] for record in outputs.values()
            },
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
                "shared_metrics_path": str(
                    ROOT / "scripts/materialize_source_separated_ic_ev_waterfall.py"
                ),
                "shared_metrics_sha256": sha256(
                    ROOT / "scripts/materialize_source_separated_ic_ev_waterfall.py"
                ),
            },
        }
        manifest_path = stage / "manifest.json"
        manifest_path.write_text(
            json.dumps(safe(manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (stage / "manifest.sha256").write_text(
            f"{sha256(manifest_path)}  manifest.json\n", encoding="utf-8"
        )
        os.replace(stage, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    value.add_argument("--raw-manifest", type=Path, default=DEFAULT_RAW_MANIFEST)
    value.add_argument("--base-oof", type=Path, default=DEFAULT_BASE_OOF)
    value.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    value.add_argument("--residual-oof", type=Path, default=DEFAULT_RESIDUAL_OOF)
    value.add_argument(
        "--residual-manifest", type=Path, default=DEFAULT_RESIDUAL_MANIFEST
    )
    value.add_argument("--direct", type=Path, default=DEFAULT_DIRECT)
    value.add_argument("--direct-manifest", type=Path, default=DEFAULT_DIRECT_MANIFEST)
    value.add_argument("--direct-frozen", type=Path, default=DEFAULT_DIRECT_FROZEN)
    value.add_argument("--direct-dataset", type=Path, default=DEFAULT_DIRECT_DATASET)
    value.add_argument(
        "--direct-dataset-manifest",
        type=Path,
        default=DEFAULT_DIRECT_DATASET_MANIFEST,
    )
    value.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--expected-raw-rows", type=int, default=EXPECTED_RAW_ROWS)
    value.add_argument(
        "--expected-common-rows", type=int, default=EXPECTED_COMMON_ROWS
    )
    return value


if __name__ == "__main__":
    args = parser().parse_args()
    result = run(
        raw_path=args.raw,
        raw_manifest_path=args.raw_manifest,
        base_oof_path=args.base_oof,
        base_manifest_path=args.base_manifest,
        residual_oof_path=args.residual_oof,
        residual_manifest_path=args.residual_manifest,
        direct_path=args.direct,
        direct_manifest_path=args.direct_manifest,
        direct_frozen_path=args.direct_frozen,
        direct_dataset_path=args.direct_dataset,
        direct_dataset_manifest_path=args.direct_dataset_manifest,
        output_dir=args.output_dir,
        expected_raw_rows=args.expected_raw_rows,
        expected_common_rows=args.expected_common_rows,
    )
    print(json.dumps(safe(result), indent=2, sort_keys=True))
