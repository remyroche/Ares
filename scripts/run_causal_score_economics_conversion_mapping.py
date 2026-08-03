#!/usr/bin/env python3
"""Map frozen scores causally into decomposed 12-hour execution economics.

Every UTC-day snapshot uses only outcomes whose 12-hour labels resolved before
the start of that day.  Score ranks are side-local, while every trading metric
uses one pooled global top-k with deterministic candidate-ID tie-breaking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
EXIT_CLASSES = ("trailing", "timeout", "full_stop", "adverse_exit")
OUTCOMES = (
    "execution_gross_ev_12h",
    "execution_cost_return",
    "execution_net_ev_12h",
    "execution_mfe_return_12h",
    "execution_mae_return_12h",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    valid = np.asarray(reference, dtype=float)
    valid = np.sort(valid[np.isfinite(valid)])
    current = np.asarray(values, dtype=float)
    if not len(valid):
        return np.full(len(current), np.nan, dtype=float)
    return np.searchsorted(valid, current, side="right") / float(len(valid))


def _decile(percentile: np.ndarray) -> np.ndarray:
    values = np.asarray(percentile, dtype=float)
    result = np.full(len(values), -1, dtype=np.int8)
    finite = np.isfinite(values)
    result[finite] = np.minimum(
        np.floor(np.clip(values[finite], 0.0, 1.0) * 10.0).astype(np.int8),
        9,
    )
    return result


def _mean(values: pd.Series | np.ndarray, default: float = 0.0) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else float(default)


def _std(values: pd.Series | np.ndarray, default: float = 0.0) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(array.std(ddof=0)) if len(array) else float(default)


def _shrunk(local: float, support: int, prior: float, shrinkage: float) -> float:
    weight = float(support) / (float(support) + max(float(shrinkage), 0.0))
    return float(weight * float(local) + (1.0 - weight) * float(prior))


def _hierarchical_mean(
    global_values: pd.Series | np.ndarray,
    side_values: pd.Series | np.ndarray,
    cell_values: pd.Series | np.ndarray,
    *,
    side_shrinkage: float,
    cell_shrinkage: float,
    default: float = 0.0,
) -> float:
    global_mean = _mean(global_values, default)
    side_array = np.asarray(side_values, dtype=float)
    side_array = side_array[np.isfinite(side_array)]
    side_mean = _shrunk(
        _mean(side_array, global_mean),
        len(side_array),
        global_mean,
        side_shrinkage,
    )
    cell_array = np.asarray(cell_values, dtype=float)
    cell_array = cell_array[np.isfinite(cell_array)]
    return _shrunk(
        _mean(cell_array, side_mean),
        len(cell_array),
        side_mean,
        cell_shrinkage,
    )


def _quantile_with_fallback(
    global_values: pd.Series | np.ndarray,
    side_values: pd.Series | np.ndarray,
    cell_values: pd.Series | np.ndarray,
    *,
    quantile: float,
    minimum_rows: int,
) -> tuple[float, str, int]:
    sources = (
        ("side_decile", np.asarray(cell_values, dtype=float)),
        ("side", np.asarray(side_values, dtype=float)),
        ("global", np.asarray(global_values, dtype=float)),
    )
    fallback: tuple[str, np.ndarray] | None = None
    for level, array in sources:
        array = array[np.isfinite(array)]
        if fallback is None and len(array):
            fallback = (level, array)
        if len(array) >= int(minimum_rows):
            return float(np.quantile(array, quantile)), level, int(len(array))
    if fallback is None:
        return 0.0, "zero", 0
    level, array = fallback
    return float(np.quantile(array, quantile)), f"{level}_under_supported", int(
        len(array)
    )


def _normalise_ledger(frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    required = {
        *IDENTITY,
        "execution_decision_utc",
        "execution_label_end_utc",
        "candidate_month",
        "source_family",
        "evidence_tier",
        "path_frequency",
        "cost_contract",
        "promotion_eligible",
        "diagnostic_only",
        "exact_policy_parity",
        "historical_observed_spread",
        "execution_exit_class",
        "opportunity_gross_above_cost_0bps",
        "opportunity_gross_above_cost_25bps",
        *OUTCOMES,
        score_column,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"conversion input missing fields: {missing}")
    out = frame.copy()
    if out.empty or out.duplicated(list(IDENTITY), keep=False).any():
        raise ValueError("conversion input identities must be nonempty and unique")
    if out["candidate_id"].astype(str).duplicated().any():
        raise ValueError("candidate_id must be unique for deterministic global rank")
    for column in (
        "source_family",
        "evidence_tier",
        "path_frequency",
        "cost_contract",
        "promotion_eligible",
        "diagnostic_only",
        "exact_policy_parity",
        "historical_observed_spread",
    ):
        if out[column].nunique(dropna=False) != 1:
            raise ValueError(f"conversion input mixes {column}")
    if not out["diagnostic_only"].astype(bool).eq(
        ~out["promotion_eligible"].astype(bool)
    ).all():
        raise ValueError("diagnostic_only must invert promotion_eligible")
    if out["historical_observed_spread"].astype(bool).any():
        raise ValueError("historical observed spread is unavailable")
    if out["promotion_eligible"].astype(bool).any():
        allowlist = {
            "canonical_base_exact1m_current_spread_cf",
            "canonical_residual_exact1m_current_spread_cf",
        }
        if str(out["source_family"].iloc[0]) not in allowlist:
            raise ValueError("noncanonical source cannot be promotion eligible")
        if not out["path_frequency"].eq("exact_1m").all():
            raise ValueError("promotion-eligible source must use exact 1m paths")
        if not out["exact_policy_parity"].astype(bool).all():
            raise ValueError("promotion-eligible source lacks exact-policy parity")
    for column in ("__ts__", "execution_decision_utc", "execution_label_end_utc"):
        out[column] = pd.to_datetime(out[column], utc=True, errors="raise")
    if not out["execution_decision_utc"].equals(
        out["__ts__"] + pd.Timedelta(hours=1)
    ):
        raise ValueError("decision must equal signal + one hour")
    if not out["execution_label_end_utc"].equals(
        out["execution_decision_utc"] + pd.Timedelta(hours=12)
    ):
        raise ValueError("resolution must equal decision + twelve hours")
    numeric = out.loc[:, [*OUTCOMES, score_column]].apply(
        pd.to_numeric, errors="raise"
    )
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("score/economics contain non-finite values")
    if not np.allclose(
        numeric["execution_gross_ev_12h"].to_numpy(float)
        - numeric["execution_cost_return"].to_numpy(float),
        numeric["execution_net_ev_12h"].to_numpy(float),
        rtol=0.0,
        atol=1e-7,
    ):
        raise ValueError("gross-cost-net reconciliation failed")
    out["score_raw"] = numeric[score_column].to_numpy(float)
    out["side_name"] = out["side_name"].astype(str).str.lower()
    out["candidate_id"] = out["candidate_id"].astype(str)
    if not out["side_name"].isin(("long", "short")).all():
        raise ValueError("unknown side in conversion input")
    allowed_exit_classes = {"trailing", "timeout", "full_stop", "adverse_exit"}
    if not out["execution_exit_class"].astype(str).isin(allowed_exit_classes).all():
        raise ValueError("conversion input contains an unknown canonical exit class")
    gross = numeric["execution_gross_ev_12h"].to_numpy(float)
    cost = numeric["execution_cost_return"].to_numpy(float)
    if not np.array_equal(
        out["opportunity_gross_above_cost_0bps"].astype(bool).to_numpy(),
        gross > cost,
    ):
        raise ValueError("0bps opportunity flag does not match gross > cost")
    if not np.array_equal(
        out["opportunity_gross_above_cost_25bps"].astype(bool).to_numpy(),
        gross > cost + 0.0025,
    ):
        raise ValueError("25bps opportunity flag does not match gross > cost + 25bps")
    return out.sort_values(
        ["execution_decision_utc", "candidate_id"], kind="stable"
    ).reset_index(drop=True)


def _component_map(
    reference: pd.DataFrame,
    *,
    side: str,
    decile: int,
    side_shrinkage: float,
    cell_shrinkage: float,
    minimum_quantile_rows: int,
) -> dict[str, Any]:
    side_reference = reference.loc[reference["side_name"].eq(side)]
    cell = side_reference.loc[
        side_reference["causal_score_decile"].eq(int(decile))
    ]
    output: dict[str, Any] = {
        "map_reference_rows": int(len(reference)),
        "map_side_reference_rows": int(len(side_reference)),
        "map_cell_reference_rows": int(len(cell)),
    }
    continuous = {
        "mapped_direct_net": "execution_net_ev_12h",
        "mapped_expected_gross": "execution_gross_ev_12h",
        "mapped_expected_cost": "execution_cost_return",
        "mapped_expected_mfe": "execution_mfe_return_12h",
        "mapped_expected_mae": "execution_mae_return_12h",
        "mapped_opportunity_probability_0bps": "opportunity_gross_above_cost_0bps",
        "mapped_opportunity_probability_25bps": "opportunity_gross_above_cost_25bps",
    }
    for output_column, source_column in continuous.items():
        output[output_column] = _hierarchical_mean(
            reference[source_column],
            side_reference[source_column],
            cell[source_column],
            side_shrinkage=side_shrinkage,
            cell_shrinkage=cell_shrinkage,
        )
    output["mapped_cost_std"] = _std(cell["execution_cost_return"])
    opportunity_global = reference.loc[
        reference["opportunity_gross_above_cost_0bps"],
        "execution_gross_ev_12h",
    ]
    opportunity_side = side_reference.loc[
        side_reference["opportunity_gross_above_cost_0bps"],
        "execution_gross_ev_12h",
    ]
    opportunity_cell = cell.loc[
        cell["opportunity_gross_above_cost_0bps"],
        "execution_gross_ev_12h",
    ]
    for quantile, label in ((0.50, "q50"), (0.80, "q80")):
        value, level, support = _quantile_with_fallback(
            opportunity_global,
            opportunity_side,
            opportunity_cell,
            quantile=quantile,
            minimum_rows=minimum_quantile_rows,
        )
        output[f"mapped_opportunity_gross_{label}"] = value
        output[f"mapped_opportunity_gross_{label}_fallback"] = level
        output[f"mapped_opportunity_gross_{label}_support"] = support
    output["mapped_opportunity_q50_net_diagnostic"] = (
        output["mapped_opportunity_probability_0bps"]
        * output["mapped_opportunity_gross_q50"]
        - output["mapped_expected_cost"]
    )
    output["mapped_opportunity_q80_net_diagnostic"] = (
        output["mapped_opportunity_probability_0bps"]
        * output["mapped_opportunity_gross_q80"]
        - output["mapped_expected_cost"]
    )
    exit_mixture = 0.0
    for exit_class in EXIT_CLASSES:
        global_indicator = reference["execution_exit_class"].eq(exit_class)
        side_indicator = side_reference["execution_exit_class"].eq(exit_class)
        cell_indicator = cell["execution_exit_class"].eq(exit_class)
        probability = _hierarchical_mean(
            global_indicator.astype(float),
            side_indicator.astype(float),
            cell_indicator.astype(float),
            side_shrinkage=side_shrinkage,
            cell_shrinkage=cell_shrinkage,
        )
        conditional_net = _hierarchical_mean(
            reference.loc[global_indicator, "execution_net_ev_12h"],
            side_reference.loc[side_indicator, "execution_net_ev_12h"],
            cell.loc[cell_indicator, "execution_net_ev_12h"],
            side_shrinkage=side_shrinkage,
            cell_shrinkage=cell_shrinkage,
            default=_mean(reference["execution_net_ev_12h"]),
        )
        output[f"mapped_exit_probability_{exit_class}"] = probability
        output[f"mapped_exit_conditional_net_{exit_class}"] = conditional_net
        exit_mixture += probability * conditional_net
    probability_sum = sum(
        output[f"mapped_exit_probability_{exit_class}"]
        for exit_class in EXIT_CLASSES
    )
    if probability_sum > 0.0:
        for exit_class in EXIT_CLASSES:
            key = f"mapped_exit_probability_{exit_class}"
            output[key] /= probability_sum
    output["mapped_adverse_probability"] = (
        output["mapped_exit_probability_full_stop"]
        + output["mapped_exit_probability_adverse_exit"]
    )
    output["mapped_timeout_probability"] = output["mapped_exit_probability_timeout"]
    output["mapped_exit_mixture_net_diagnostic"] = float(exit_mixture)
    return output


def causal_component_mapping(
    frame: pd.DataFrame,
    *,
    score_column: str,
    window_days: int,
    minimum_reference_rows: int,
    minimum_side_rows: int,
    minimum_quantile_rows: int,
    side_shrinkage: float,
    cell_shrinkage: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply one resolved-before-day causal component map to a source family."""

    out = _normalise_ledger(frame, score_column)
    out["causal_score_percentile"] = np.nan
    out["causal_score_decile"] = np.int8(-1)
    mapped_defaults: dict[str, Any] = {
        "mapped_eligible": False,
        "map_reference_rows": 0,
        "map_side_reference_rows": 0,
        "map_cell_reference_rows": 0,
        "mapped_direct_net": np.nan,
        "mapped_expected_gross": np.nan,
        "mapped_expected_cost": np.nan,
        "mapped_expected_mfe": np.nan,
        "mapped_expected_mae": np.nan,
        "mapped_cost_std": np.nan,
        "mapped_opportunity_probability_0bps": np.nan,
        "mapped_opportunity_probability_25bps": np.nan,
        "mapped_opportunity_gross_q50": np.nan,
        "mapped_opportunity_gross_q80": np.nan,
        "mapped_opportunity_gross_q50_fallback": "unavailable",
        "mapped_opportunity_gross_q80_fallback": "unavailable",
        "mapped_opportunity_gross_q50_support": 0,
        "mapped_opportunity_gross_q80_support": 0,
        "mapped_opportunity_q50_net_diagnostic": np.nan,
        "mapped_opportunity_q80_net_diagnostic": np.nan,
        "mapped_adverse_probability": np.nan,
        "mapped_timeout_probability": np.nan,
        "mapped_exit_mixture_net_diagnostic": np.nan,
    }
    for exit_class in EXIT_CLASSES:
        mapped_defaults[f"mapped_exit_probability_{exit_class}"] = np.nan
        mapped_defaults[f"mapped_exit_conditional_net_{exit_class}"] = np.nan
    for column, value in mapped_defaults.items():
        out[column] = value
    decision_day = out["execution_decision_utc"].dt.floor("D")
    resolution = out["execution_label_end_utc"]
    audits: list[dict[str, Any]] = []
    for snapshot, indices in out.groupby(decision_day, sort=True).groups.items():
        snapshot = pd.Timestamp(snapshot)
        positions = out.index.get_indexer(indices)
        lower = snapshot - pd.Timedelta(days=int(window_days))
        reference_mask = resolution.lt(snapshot)
        if int(window_days) > 0:
            reference_mask &= resolution.ge(lower)
        reference = out.loc[
            reference_mask & out["causal_score_decile"].ge(0)
        ].copy()
        current = out.iloc[positions]
        percentile = np.full(len(current), np.nan, dtype=float)
        basis: dict[str, str] = {}
        for side in ("long", "short"):
            current_mask = current["side_name"].eq(side).to_numpy()
            if not current_mask.any():
                continue
            side_reference = reference.loc[reference["side_name"].eq(side)]
            if len(side_reference):
                percentile[current_mask] = _empirical_percentile(
                    side_reference["score_raw"].to_numpy(float),
                    current.loc[current_mask, "score_raw"].to_numpy(float),
                )
                basis[side] = "prior_resolved_window"
            else:
                for _, timestamp_rows in current.loc[current_mask].groupby(
                    "execution_decision_utc", sort=True
                ):
                    local_positions = current.index.get_indexer(timestamp_rows.index)
                    percentile[local_positions] = _empirical_percentile(
                        timestamp_rows["score_raw"].to_numpy(float),
                        timestamp_rows["score_raw"].to_numpy(float),
                    )
                basis[side] = "same_timestamp_warmup_only"
        deciles = _decile(percentile)
        out.iloc[
            positions, out.columns.get_loc("causal_score_percentile")
        ] = percentile
        out.iloc[positions, out.columns.get_loc("causal_score_decile")] = deciles
        reference_rows = int(len(reference))
        side_support: dict[str, int] = {}
        for side in ("long", "short"):
            side_positions = positions[
                current["side_name"].eq(side).to_numpy()
            ]
            side_reference_rows = int(reference["side_name"].eq(side).sum())
            side_support[side] = side_reference_rows
            eligible = (
                reference_rows >= int(minimum_reference_rows)
                and side_reference_rows >= int(minimum_side_rows)
            )
            if not eligible or not len(side_positions):
                continue
            for decile_value in np.unique(
                out.iloc[side_positions]["causal_score_decile"].to_numpy(int)
            ):
                if decile_value < 0:
                    continue
                local_positions = side_positions[
                    out.iloc[side_positions]["causal_score_decile"]
                    .eq(int(decile_value))
                    .to_numpy()
                ]
                mapped = _component_map(
                    reference,
                    side=side,
                    decile=int(decile_value),
                    side_shrinkage=side_shrinkage,
                    cell_shrinkage=cell_shrinkage,
                    minimum_quantile_rows=minimum_quantile_rows,
                )
                out.iloc[
                    local_positions, out.columns.get_loc("mapped_eligible")
                ] = True
                for column, value in mapped.items():
                    out.iloc[
                        local_positions, out.columns.get_loc(column)
                    ] = value
        reference_max = (
            reference["execution_label_end_utc"].max()
            if len(reference)
            else pd.NaT
        )
        if pd.notna(reference_max) and not pd.Timestamp(reference_max) < snapshot:
            raise ValueError("causal reference includes an unresolved outcome")
        audits.append(
            {
                "snapshot_utc": snapshot,
                "window_start_utc": lower if int(window_days) > 0 else None,
                "reference_rows": reference_rows,
                "long_reference_rows": side_support.get("long", 0),
                "short_reference_rows": side_support.get("short", 0),
                "reference_label_end_max_utc": reference_max,
                "current_rows": int(len(positions)),
                "mapped_rows": int(out.iloc[positions]["mapped_eligible"].sum()),
                "long_percentile_basis": basis.get("long"),
                "short_percentile_basis": basis.get("short"),
            }
        )
    if not out["causal_score_decile"].between(0, 9).all():
        raise ValueError("causal score decile assignment is incomplete")
    mapped = out.loc[out["mapped_eligible"]]
    if len(mapped):
        probabilities = mapped.loc[
            :,
            [
                "mapped_opportunity_probability_0bps",
                "mapped_opportunity_probability_25bps",
                "mapped_adverse_probability",
                "mapped_timeout_probability",
                *[
                    f"mapped_exit_probability_{exit_class}"
                    for exit_class in EXIT_CLASSES
                ],
            ],
        ].to_numpy(float)
        if not np.isfinite(probabilities).all():
            raise ValueError("mapped probabilities are non-finite")
        if ((probabilities < -1e-12) | (probabilities > 1.0 + 1e-12)).any():
            raise ValueError("mapped probability is outside [0,1]")
        exit_sum = mapped.loc[
            :,
            [
                f"mapped_exit_probability_{exit_class}"
                for exit_class in EXIT_CLASSES
            ],
        ].sum(axis=1)
        if not np.allclose(exit_sum.to_numpy(float), 1.0, atol=1e-8, rtol=0.0):
            raise ValueError("mapped exit probabilities do not sum to one")
    return out, pd.DataFrame(audits)


def _stable_select(
    frame: pd.DataFrame,
    *,
    score_column: str,
    fraction: float,
) -> pd.DataFrame:
    valid = frame.loc[
        np.isfinite(pd.to_numeric(frame[score_column], errors="coerce"))
    ].copy()
    if valid.empty:
        return valid
    count = max(1, int(math.ceil(float(fraction) * len(valid))))
    order = np.lexsort(
        (
            valid["candidate_id"].astype(str).to_numpy(),
            -pd.to_numeric(valid[score_column], errors="raise").to_numpy(float),
        )
    )
    return valid.iloc[order[:count]].copy()


def _oracle_recall(
    population: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    target_column: str,
    fraction: float,
) -> float:
    oracle = _stable_select(
        population.assign(__oracle__=pd.to_numeric(population[target_column])),
        score_column="__oracle__",
        fraction=fraction,
    )
    if oracle.empty:
        return float("nan")
    return float(
        len(set(oracle["candidate_id"]).intersection(selected["candidate_id"]))
        / len(oracle)
    )


def _economic_row(
    population: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    mapping: str,
    scope: str,
    fraction: float,
    selection_basis: str,
) -> dict[str, Any]:
    net = pd.to_numeric(selected["execution_net_ev_12h"], errors="raise")
    gross = pd.to_numeric(selected["execution_gross_ev_12h"], errors="raise")
    cost = pd.to_numeric(selected["execution_cost_return"], errors="raise")
    exit_reason = selected["execution_exit_class"].astype(str)
    return {
        "mapping": mapping,
        "scope": scope,
        "selection_basis": selection_basis,
        "top_k_fraction": float(fraction),
        "eligible_rows": int(len(population)),
        "selected_rows": int(len(selected)),
        "mean_gross_bps": float(10_000.0 * gross.mean()),
        "mean_cost_bps": float(10_000.0 * cost.mean()),
        "mean_net_bps": float(10_000.0 * net.mean()),
        "sum_net_return": float(net.sum()),
        "positive_net_rate": float((net > 0.0).mean()),
        "opportunity_0bps_rate": float(
            selected["opportunity_gross_above_cost_0bps"].mean()
        ),
        "opportunity_25bps_rate": float(
            selected["opportunity_gross_above_cost_25bps"].mean()
        ),
        "trailing_rate": float(exit_reason.eq("trailing").mean()),
        "timeout_rate": float(exit_reason.eq("timeout").mean()),
        "full_stop_rate": float(exit_reason.eq("full_stop").mean()),
        "adverse_exit_rate": float(exit_reason.eq("adverse_exit").mean()),
        "long_rows": int(selected["side_name"].eq("long").sum()),
        "short_rows": int(selected["side_name"].eq("short").sum()),
        "unique_assets": int(selected["__symbol__"].nunique()),
        "largest_asset_share": float(
            selected["__symbol__"].value_counts(normalize=True).max()
        ),
        "oracle_gross_recall": _oracle_recall(
            population,
            selected,
            target_column="execution_gross_ev_12h",
            fraction=fraction,
        ),
        "oracle_net_recall": _oracle_recall(
            population,
            selected,
            target_column="execution_net_ev_12h",
            fraction=fraction,
        ),
    }


def evaluate_global_tail(
    frame: pd.DataFrame,
    *,
    score_columns: Sequence[str],
    top_k_fractions: Sequence[float],
) -> pd.DataFrame:
    eligible = frame.loc[frame["mapped_eligible"]].copy()
    rows: list[dict[str, Any]] = []
    for score_column in score_columns:
        score_valid = eligible.loc[
            np.isfinite(pd.to_numeric(eligible[score_column], errors="coerce"))
        ].copy()
        if score_valid.empty:
            continue
        scopes: list[tuple[str, str, pd.DataFrame]] = [
            ("pooled", "one_pooled_global_book", score_valid)
        ]
        scopes.extend(
            (
                f"month_{month}",
                "month_local_global_diagnostic",
                local,
            )
            for month, local in score_valid.groupby("candidate_month", sort=True)
        )
        latest_start = score_valid["execution_decision_utc"].max().floor("D") - pd.Timedelta(
            days=6
        )
        scopes.append(
            (
                "latest_7d",
                "latest_week_global_diagnostic",
                score_valid.loc[
                    score_valid["execution_decision_utc"].ge(latest_start)
                ],
            )
        )
        for scope, selection_basis, population in scopes:
            if population.empty:
                continue
            for fraction in top_k_fractions:
                selected = _stable_select(
                    population, score_column=score_column, fraction=fraction
                )
                rows.append(
                    _economic_row(
                        population,
                        selected,
                        mapping=score_column,
                        scope=scope,
                        fraction=fraction,
                        selection_basis=selection_basis,
                    )
                )
    return pd.DataFrame(rows)


def _safe_auc(target: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(target)) < 2:
        return float("nan")
    return float(roc_auc_score(target, score))


def calibration_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    eligible = frame.loc[frame["mapped_eligible"]].copy()
    if eligible.empty:
        return pd.DataFrame()
    scopes: list[tuple[str, pd.DataFrame]] = [("pooled", eligible)]
    scopes.extend(
        (f"month_{month}", local)
        for month, local in eligible.groupby("candidate_month", sort=True)
    )
    rows: list[dict[str, Any]] = []
    for scope, local in scopes:
        opportunity = local["opportunity_gross_above_cost_0bps"].astype(int).to_numpy()
        opportunity_score = local[
            "mapped_opportunity_probability_0bps"
        ].to_numpy(float)
        direct = local["mapped_direct_net"].to_numpy(float)
        net = local["execution_net_ev_12h"].to_numpy(float)
        gross = local["execution_gross_ev_12h"].to_numpy(float)
        exit_actual = local["execution_exit_class"].astype(str)
        exit_brier = 0.0
        for exit_class in EXIT_CLASSES:
            actual = exit_actual.eq(exit_class).astype(float).to_numpy()
            predicted = local[f"mapped_exit_probability_{exit_class}"].to_numpy(
                float
            )
            exit_brier += float(np.mean(np.square(actual - predicted)))
        event = opportunity.astype(bool)
        q50 = local["mapped_opportunity_gross_q50"].to_numpy(float)
        q80 = local["mapped_opportunity_gross_q80"].to_numpy(float)
        q50_error = gross[event] - q50[event]
        q80_error = gross[event] - q80[event]
        rows.append(
            {
                "scope": scope,
                "rows": int(len(local)),
                "opportunity_prevalence": float(opportunity.mean()),
                "opportunity_auc": _safe_auc(opportunity, opportunity_score),
                "opportunity_average_precision": float(
                    average_precision_score(opportunity, opportunity_score)
                ),
                "opportunity_brier": float(
                    brier_score_loss(opportunity, opportunity_score)
                ),
                "exit_multiclass_brier_sum": exit_brier,
                "direct_net_spearman": float(
                    spearmanr(direct, net, nan_policy="omit").statistic
                ),
                "direct_net_mae_bps": float(10_000.0 * np.mean(np.abs(direct - net))),
                "direct_net_rmse_bps": float(
                    10_000.0 * np.sqrt(np.mean(np.square(direct - net)))
                ),
                "opportunity_q50_pinball_bps": float(
                    10_000.0
                    * np.mean(
                        np.maximum(0.50 * q50_error, (0.50 - 1.0) * q50_error)
                    )
                ),
                "opportunity_q80_pinball_bps": float(
                    10_000.0
                    * np.mean(
                        np.maximum(0.80 * q80_error, (0.80 - 1.0) * q80_error)
                    )
                ),
                "opportunity_q50_coverage": float(
                    np.mean(gross[event] <= q50[event])
                ),
                "opportunity_q80_coverage": float(
                    np.mean(gross[event] <= q80[event])
                ),
                "opportunity_rows": int(event.sum()),
            }
        )
    return pd.DataFrame(rows)


def component_decile_report(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (month, side, decile), local in frame.groupby(
        ["candidate_month", "side_name", "causal_score_decile"], sort=True
    ):
        net = pd.to_numeric(local["execution_net_ev_12h"], errors="raise")
        gross = pd.to_numeric(local["execution_gross_ev_12h"], errors="raise")
        cost = pd.to_numeric(local["execution_cost_return"], errors="raise")
        exit_reason = local["execution_exit_class"].astype(str)
        opportunity = local["opportunity_gross_above_cost_0bps"].astype(bool)
        rows.append(
            {
                "candidate_month": month,
                "side_name": side,
                "causal_score_decile": int(decile),
                "rows": int(len(local)),
                "score_mean": float(local["score_raw"].mean()),
                "opportunity_rate": float(opportunity.mean()),
                "conditional_opportunity_gross_bps": float(
                    10_000.0 * gross.loc[opportunity].mean()
                ),
                "mean_gross_bps": float(10_000.0 * gross.mean()),
                "mean_cost_bps": float(10_000.0 * cost.mean()),
                "mean_net_bps": float(10_000.0 * net.mean()),
                "positive_net_rate": float((net > 0.0).mean()),
                "trailing_rate": float(exit_reason.eq("trailing").mean()),
                "timeout_rate": float(exit_reason.eq("timeout").mean()),
                "full_stop_rate": float(exit_reason.eq("full_stop").mean()),
                "adverse_exit_rate": float(exit_reason.eq("adverse_exit").mean()),
                "trailing_conditional_net_bps": float(
                    10_000.0 * net.loc[exit_reason.eq("trailing")].mean()
                ),
                "timeout_conditional_net_bps": float(
                    10_000.0 * net.loc[exit_reason.eq("timeout")].mean()
                ),
                "full_stop_conditional_net_bps": float(
                    10_000.0 * net.loc[exit_reason.eq("full_stop")].mean()
                ),
                "adverse_exit_conditional_net_bps": float(
                    10_000.0 * net.loc[exit_reason.eq("adverse_exit")].mean()
                ),
                "mean_mfe_bps": float(
                    10_000.0 * local["execution_mfe_return_12h"].mean()
                ),
                "mean_mae_bps": float(
                    10_000.0 * local["execution_mae_return_12h"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Path]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    raw = pd.read_parquet(args.ledger)
    mapped, causal_audit = causal_component_mapping(
        raw,
        score_column=args.score_column,
        window_days=args.window_days,
        minimum_reference_rows=args.minimum_reference_rows,
        minimum_side_rows=args.minimum_side_rows,
        minimum_quantile_rows=args.minimum_quantile_rows,
        side_shrinkage=args.side_shrinkage,
        cell_shrinkage=args.cell_shrinkage,
    )
    score_columns = [
        "score_raw",
        "mapped_direct_net",
        "mapped_opportunity_q50_net_diagnostic",
        "mapped_opportunity_q80_net_diagnostic",
        "mapped_exit_mixture_net_diagnostic",
    ]
    economics = evaluate_global_tail(
        mapped,
        score_columns=score_columns,
        top_k_fractions=args.top_k_fractions,
    )
    calibration = calibration_metrics(mapped)
    deciles = component_decile_report(mapped)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    mapped_path = args.output_dir / "causal_mapped_candidates.parquet"
    audit_path = args.output_dir / "causal_snapshot_audit.parquet"
    economics_path = args.output_dir / "global_tail_economics.csv"
    calibration_path = args.output_dir / "component_calibration.csv"
    decile_path = args.output_dir / "month_side_decile_components.csv"
    mapped.to_parquet(mapped_path, index=False, compression="zstd")
    causal_audit.to_parquet(audit_path, index=False, compression="zstd")
    economics.to_csv(economics_path, index=False)
    calibration.to_csv(calibration_path, index=False)
    deciles.to_csv(decile_path, index=False)
    mapped_eligible = mapped.loc[mapped["mapped_eligible"]]
    manifest = {
        "schema": "causal_score_economics_conversion_mapping_v1",
        "status": "CAUSAL_COMPONENT_MAPPING_COMPLETE",
        "source": {"path": str(args.ledger), "sha256": _sha256(args.ledger)},
        "source_family": str(mapped["source_family"].iloc[0]),
        "evidence_tier": str(mapped["evidence_tier"].iloc[0]),
        "score_column": args.score_column,
        "causal_contract": {
            "snapshot": "UTC day start",
            "reference_rule": "execution_label_end_utc < snapshot",
            "window_days": int(args.window_days),
            "percentile": (
                "side-local prior-resolved-window empirical percentile; "
                "same-timestamp score peers only for unmapped warmup coordinates"
            ),
            "minimum_reference_rows": int(args.minimum_reference_rows),
            "minimum_side_rows": int(args.minimum_side_rows),
            "side_shrinkage": float(args.side_shrinkage),
            "cell_shrinkage": float(args.cell_shrinkage),
            "minimum_quantile_rows": int(args.minimum_quantile_rows),
        },
        "selection_contract": {
            "primary": "one pooled global top-k",
            "tie_break": "candidate_id ascending",
            "not_per_timestamp": True,
            "month_and_latest_week": "diagnostic slices",
            "top_k_fractions": [float(value) for value in args.top_k_fractions],
        },
        "score_roles": {
            "mapped_direct_net": "admission challenger",
            "mapped_opportunity_q50_net_diagnostic": "diagnostic only",
            "mapped_opportunity_q80_net_diagnostic": "optimistic diagnostic only",
            "mapped_exit_mixture_net_diagnostic": "explanatory diagnostic only",
        },
        "rows": {
            "input": int(len(mapped)),
            "mapped_eligible": int(len(mapped_eligible)),
            "warmup_unmapped": int((~mapped["mapped_eligible"]).sum()),
        },
        "promotion_boundary": {
            "source_promotion_eligible": bool(mapped["promotion_eligible"].iloc[0]),
            "exact_policy_parity": bool(mapped["exact_policy_parity"].iloc[0]),
            "historical_observed_spread": False,
        },
        "outputs": {},
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
    }
    outputs = {
        "mapped": mapped_path,
        "audit": audit_path,
        "economics": economics_path,
        "calibration": calibration_path,
        "deciles": decile_path,
    }
    manifest["outputs"] = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in outputs.items()
    }
    manifest_path = args.output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    outputs["manifest"] = manifest_path
    return outputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--score-column", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-days", type=int, default=21)
    parser.add_argument("--minimum-reference-rows", type=int, default=1_000)
    parser.add_argument("--minimum-side-rows", type=int, default=500)
    parser.add_argument("--minimum-quantile-rows", type=int, default=200)
    parser.add_argument("--side-shrinkage", type=float, default=500.0)
    parser.add_argument("--cell-shrinkage", type=float, default=250.0)
    parser.add_argument(
        "--top-k-fractions",
        type=float,
        nargs="+",
        default=(0.01, 0.05, 0.10, 0.20),
    )
    return parser


def main() -> None:
    outputs = run(_parser().parse_args())
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
