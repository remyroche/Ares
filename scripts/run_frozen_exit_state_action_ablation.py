#!/usr/bin/env python3
"""Frozen, identical-book stateful exit-action diagnostics.

This runner does not alter candidate selection, scores, ranks, weights, or
portfolio composition.  It replays three pre-declared state-machine changes on
the exact residual-selected March/April books:

* T4: deployed stops/trailing with a forced close after 240 observed 1m bars;
* D2: trailing activation decay starts at 120 bars, half-life 120, floor 50%;
* W75: deployed trailing gap tightened to 75% of its original width.

The deployed row-level canonical cost is held fixed across every arm.  This is
an attribution diagnostic, not a deployable policy search.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_execution_ev_policy_labels import (  # noqa: E402
    _load_candidates,
    _policy_contract,
    _resolved_geometry,
    _simulate_batch,
    _simulation_kwargs,
)
from extreme_price_movements.simple_policy_optimiser import (  # noqa: E402
    simulate_and_score,
)

FIXED_ROOT = (
    ROOT / "data_perp/artifacts/fixed_horizon_action_ablation_20260730_v2"
)
SELECTED_ROOT = (
    ROOT
    / "data_perp/artifacts/residual_selected_exit_opportunity_counterfactual_20260730_v3"
)
LABEL_ROOT = (
    ROOT
    / "data_perp/artifacts/febapr2025_execution_ev_current_spread_12h_labels_20260727_v1"
)
INPUT_ROOT = (
    ROOT
    / "data_perp/artifacts/febapr2025_execution_ev_deployed_policy_inputs_20260727_v1"
)
PATH_ROOT = (
    ROOT / "data_perp/artifacts/febapr2025_top40_exact1m_paths_20260727_v1"
)
POLICY_PATH = (
    ROOT
    / "data_perp/reports/simple_policy_1m_joint_trailing_raw_bayesian_champion_20260718_v1"
    / "production_staging/best_policy_params.json"
)
OUT = ROOT / "data_perp/artifacts/frozen_exit_state_action_ablation_20260730_v4"

IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
TOPS = (0.01, 0.05, 0.10, 0.20)
FIXED_ARMS = ("fixed_1h", "fixed_2h", "fixed_4h", "fixed_8h", "fixed_12h")
STATE_ARMS = ("T4", "D2", "W75", "P50")
ARMS = ("deployed", *FIXED_ARMS, *STATE_ARMS)
SIM_FIELDS = (
    "execution_gross_ev_12h",
    "execution_net_ev_12h",
    "execution_exit_reason",
    "execution_exit_hour",
    "execution_mfe_return_12h",
    "execution_mae_return_12h",
    "execution_entry_price",
    "execution_exit_price",
    "execution_expected_spread_bps",
    "execution_entry_half_spread_bps",
    "execution_exit_half_spread_bps",
)


class ContractError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(safe(payload), indent=2, sort_keys=True) + "\n")


def verify_seal(root: Path, schema: str) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    seal_path = root / "manifest.sha256"
    if not manifest_path.is_file() or not seal_path.is_file():
        raise ContractError(f"missing seal: {root}")
    if sha256(manifest_path) != seal_path.read_text().split()[0]:
        raise ContractError(f"manifest seal mismatch: {root}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != schema:
        raise ContractError(f"schema mismatch: {root}")
    for name, expected in manifest.get("outputs_sha256", {}).items():
        if sha256(root / name) != expected:
            raise ContractError(f"sealed output mismatch: {root / name}")
    return manifest


def variant_strategy(strategy: Mapping[str, Any], arm: str) -> dict[str, Any]:
    """Return one and only one frozen state-machine intervention."""
    result = dict(strategy)
    if arm == "T4":
        return result
    if arm == "D2":
        result["trailing_activation_decay_start_bars"] = 120
        result["trailing_activation_decay_half_life_bars"] = 120.0
        result["trailing_activation_min_mult"] = 0.50
        return result
    if arm == "W75":
        fixed_gap = float(result.get("fixed_trailing_gap_mult", 0.0) or 0.0)
        if fixed_gap > 0.0:
            result["fixed_trailing_gap_mult"] = 0.75 * fixed_gap
        else:
            beta = float(result.get("giveback_beta", 0.5) or 0.5)
            result["giveback_beta"] = 0.75 * beta
        return result
    if arm in {"C0", "P50"}:
        return result
    raise ContractError(f"unknown state arm: {arm}")


def _parse_paths(payloads: pd.Series) -> tuple[np.ndarray, ...]:
    count = len(payloads)
    arrays = tuple(np.empty((count, 720), dtype=np.float32) for _ in range(4))
    for row_index, payload in enumerate(payloads.astype(str)):
        parsed = json.loads(payload)
        timestamp = np.asarray(parsed["timestamp"], dtype=np.int64)
        values = tuple(
            np.asarray(parsed[name], dtype=np.float32)
            for name in ("open", "high", "low", "close")
        )
        if (
            timestamp.shape != (720,)
            or any(value.shape != (720,) for value in values)
            or any(not np.isfinite(value).all() for value in values)
            or not np.all(np.diff(timestamp) == 60_000_000_000)
        ):
            raise ContractError("every replay path must be contiguous finite 720x1m")
        for target, value in zip(arrays, values):
            target[row_index] = value
    return arrays


def _load_selected_paths(selected: pd.DataFrame) -> pd.DataFrame:
    if selected["candidate_id"].duplicated().any():
        raise ContractError("selected candidate_id must be unique")
    wanted = set(selected["candidate_id"].astype(str))
    pieces: list[pd.DataFrame] = []
    parquet = pq.ParquetFile(PATH_ROOT / "paths.parquet")
    columns = [*IDENTITY, "execution_future_path"]
    for batch in parquet.iter_batches(batch_size=512, columns=columns):
        local = batch.to_pandas()
        local = local.loc[local["candidate_id"].astype(str).isin(wanted)]
        if not local.empty:
            pieces.append(local)
    if not pieces:
        raise ContractError("no selected exact paths found")
    paths = pd.concat(pieces, ignore_index=True)
    if paths["candidate_id"].duplicated().any():
        raise ContractError("selected exact paths contain duplicate candidate_id")
    result = selected.loc[:, list(IDENTITY)].merge(
        paths,
        on="candidate_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "__source"),
    )
    if result["execution_future_path"].isna().any() or len(result) != len(selected):
        raise ContractError("selected exact path coverage is incomplete")
    if (
        not result["side_name__source"].astype(str).eq(
            result["side_name"].astype(str)
        ).all()
        or not pd.to_datetime(result["__ts____source"], utc=True).eq(
            pd.to_datetime(result["__ts__"], utc=True)
        ).all()
        or not result["__symbol____source"]
        .astype(str)
        .str.replace("/", "_", regex=False)
        .eq(result["__symbol__"].astype(str))
        .all()
    ):
        raise ContractError("selected exact path normalized identity mismatch")
    return result


def _strategy_lookup(policy: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    lookup = {
        str(raw.get("canonical_strategy_id")): dict(raw)
        for raw in policy["strategies"]
        if isinstance(raw, Mapping)
        and raw.get("selected", True)
        and raw.get("canonical_strategy_id")
    }
    if not lookup:
        raise ContractError("policy has no selected canonical strategies")
    return lookup


def _simulate_p50_batch(
    rows: pd.DataFrame,
    arrays: tuple[np.ndarray, ...],
    strategy: Mapping[str, Any],
) -> pd.DataFrame:
    """Replay the fixed 50% partial action and unchanged remainder state."""
    cost_pct, size_power, params = _simulation_kwargs(strategy)
    simulator_rows = pd.DataFrame(
        {
            "timestamp": rows["__decision_ts__"].to_numpy(),
            "symbol": rows["__symbol__"].astype(str).to_numpy(),
            "side_name": rows["side_name"].astype(str).to_numpy(),
            "side": np.where(rows["side_name"].eq("long"), 1.0, -1.0),
            "rank_pct": np.ones(len(rows), dtype=np.float32),
            "barrier_pct": rows["__barrier_pct__"].to_numpy(dtype=np.float32),
            "policy_archetype": rows["policy_archetype"].astype(str).to_numpy(),
        }
    )
    metrics = simulate_and_score(
        simulator_rows,
        *arrays,
        cost_pct=cost_pct,
        size_power=size_power,
        max_concurrent_trades=1_000_000_000,
        max_concurrent_per_asset=1_000_000_000,
        max_new_entries_per_bar=1_000_000_000,
        partial_exit_on_first_trailing_activation_fraction=0.5,
        **params,
    )
    selected = np.asarray(metrics["selected_mask"], dtype=bool)
    if selected.shape != (len(rows),) or not selected.all():
        raise ContractError("P50 replay unexpectedly dropped valid rows")
    entry = np.asarray(metrics["entry_prices"], dtype=np.float64)
    side = np.where(rows["side_name"].eq("long"), 1.0, -1.0)
    highs = arrays[1].astype(np.float64, copy=False)
    lows = arrays[2].astype(np.float64, copy=False)
    mfe = np.where(
        side > 0,
        np.max(highs / entry[:, None] - 1.0, axis=1),
        np.max(1.0 - lows / entry[:, None], axis=1),
    )
    mae = np.where(
        side > 0,
        np.max(1.0 - lows / entry[:, None], axis=1),
        np.max(highs / entry[:, None] - 1.0, axis=1),
    )
    partial_mask = np.asarray(metrics["partial_exit_mask"], dtype=bool)
    return pd.DataFrame(
        {
            "execution_gross_ev_12h": np.asarray(
                metrics["p50_gross_returns"], dtype=np.float64
            ),
            "execution_net_ev_12h": np.asarray(
                metrics["p50_net_returns"], dtype=np.float64
            ),
            "execution_exit_reason": list(metrics["exit_reason"]),
            "execution_exit_hour": np.asarray(
                metrics["exit_bars"], dtype=np.float64
            )
            / 60.0,
            "execution_mfe_return_12h": mfe,
            "execution_mae_return_12h": mae,
            "execution_entry_price": entry,
            "execution_exit_price": np.asarray(
                metrics["exit_prices"], dtype=np.float64
            ),
            "execution_expected_spread_bps": np.asarray(
                metrics["expected_spread_bps"], dtype=np.float64
            ),
            "execution_entry_half_spread_bps": np.asarray(
                metrics["entry_half_spread_bps"], dtype=np.float64
            ),
            "execution_exit_half_spread_bps": np.asarray(
                metrics["exit_spread_cost_bps"], dtype=np.float64
            ),
            "partial_exit_mask": partial_mask.astype(np.int8),
            "partial_exit_hour": np.where(
                partial_mask,
                np.asarray(metrics["partial_exit_bars"], dtype=np.float64)
                / 60.0,
                -1.0,
            ),
            "partial_exit_return": np.asarray(
                metrics["partial_exit_returns"], dtype=np.float64
            ),
            "multi_exit_fee_return": np.asarray(
                metrics["p50_fee_returns"], dtype=np.float64
            ),
        }
    )


def _simulate_arms(
    rows: pd.DataFrame,
    arrays: tuple[np.ndarray, ...],
    strategies: Mapping[str, Mapping[str, Any]],
) -> dict[str, pd.DataFrame]:
    outputs = {
        arm: pd.DataFrame(index=np.arange(len(rows)), columns=list(SIM_FIELDS))
        for arm in ("C0", "T4", "D2", "W75")
    }
    outputs["P50"] = pd.DataFrame(
        index=np.arange(len(rows)),
        columns=[
            *SIM_FIELDS,
            "partial_exit_mask",
            "partial_exit_hour",
            "partial_exit_return",
            "multi_exit_fee_return",
        ],
    )
    for geometry_key, index_values in rows.groupby(
        "execution_geometry_key", sort=True
    ).groups.items():
        if str(geometry_key) not in strategies:
            raise ContractError(f"unresolved strategy: {geometry_key}")
        positions = np.asarray(list(index_values), dtype=np.int64)
        local_rows = rows.iloc[positions]
        local_arrays = tuple(array[positions] for array in arrays)
        for arm in ("C0", "T4", "D2", "W75"):
            strategy = variant_strategy(strategies[str(geometry_key)], arm)
            arm_arrays = (
                tuple(array[:, :240] for array in local_arrays)
                if arm == "T4"
                else local_arrays
            )
            simulated = _simulate_batch(local_rows, arm_arrays, strategy)
            outputs[arm].iloc[positions] = simulated.loc[:, list(SIM_FIELDS)].to_numpy()
        p50 = _simulate_p50_batch(
            local_rows,
            local_arrays,
            variant_strategy(strategies[str(geometry_key)], "P50"),
        )
        outputs["P50"].iloc[positions] = p50.loc[
            :, outputs["P50"].columns
        ].to_numpy()
    for arm, frame in outputs.items():
        for column in frame.columns:
            if column == "execution_exit_reason":
                frame[column] = frame[column].astype(str)
            else:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        numeric = frame.drop(columns=["execution_exit_reason"])
        if numeric.isna().any().any():
            raise ContractError(f"{arm} produced non-finite simulator outputs")
    return outputs


def weighted_mean(frame: pd.DataFrame, value: str, weight: str) -> float:
    denominator = float(frame[weight].sum())
    if denominator <= 0.0:
        raise ContractError("empty weighted scope")
    return float((frame[weight] * frame[value]).sum() / denominator)


def metric_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for month, month_rows in frame.groupby("candidate_month", sort=True):
        for fraction in TOPS:
            weight = f"weight_top_{int(fraction * 100):02d}"
            active = month_rows.loc[month_rows[weight].gt(0)]
            total = float(active[weight].sum())
            scopes = [("global", active)] + [
                (f"side_{side}", local)
                for side, local in active.groupby("side_name", sort=True)
            ]
            for scope, local in scopes:
                deployed = weighted_mean(local, "net__deployed", weight)
                fixed12 = weighted_mean(local, "net__fixed_12h", weight)
                for arm in ARMS:
                    net = weighted_mean(local, f"net__{arm}", weight)
                    output.append(
                        {
                            "candidate_month": month,
                            "top_fraction": fraction,
                            "scope": scope,
                            "arm": arm,
                            "expected_selected_rows": float(local[weight].sum()),
                            "global_expected_selected_rows": total,
                            "net_bps": net * 10_000.0,
                            "gross_bps": weighted_mean(
                                local, f"gross__{arm}", weight
                            )
                            * 10_000.0,
                            "cost_bps": weighted_mean(
                                local, f"cost__{arm}", weight
                            )
                            * 10_000.0,
                            "paired_delta_vs_deployed_bps": (
                                net - deployed
                            )
                            * 10_000.0,
                            "paired_delta_vs_fixed_12h_bps": (
                                net - fixed12
                            )
                            * 10_000.0,
                            "positive_rate": weighted_mean(
                                local, f"positive__{arm}", weight
                            ),
                        }
                    )
    return output


def bootstrap_rows(
    frame: pd.DataFrame, *, draws: int = 2_000
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for month, month_rows in frame.groupby("candidate_month", sort=True):
        for fraction in TOPS:
            weight = f"weight_top_{int(fraction * 100):02d}"
            active = month_rows.loc[month_rows[weight].gt(0)].copy()
            scopes = [("global", active)] + [
                (f"side_{side}", local.copy())
                for side, local in active.groupby("side_name", sort=True)
            ]
            for scope, local in scopes:
                local["day"] = pd.to_datetime(
                    local.execution_decision_utc, utc=True
                ).dt.floor("D")
                days = sorted(local.day.unique())
                rng = np.random.default_rng(
                    20260730
                    + int(fraction * 100)
                    + (0 if scope == "global" else sum(map(ord, scope)))
                )
                index = rng.integers(0, len(days), size=(draws, len(days)))
                for arm in ARMS:
                    local["_absolute"] = local[weight] * local[f"net__{arm}"]
                    local["_deployed"] = local[weight] * (
                        local[f"net__{arm}"] - local["net__deployed"]
                    )
                    local["_fixed12"] = local[weight] * (
                        local[f"net__{arm}"] - local["net__fixed_12h"]
                    )
                    local["_den"] = local[weight]
                    daily = (
                        local.groupby("day", sort=True)[
                            ["_absolute", "_deployed", "_fixed12", "_den"]
                        ]
                        .sum()
                        .reindex(days, fill_value=0.0)
                    )
                    denominator = daily._den.to_numpy()[index].sum(axis=1)
                    samples = {
                        "net": daily._absolute.to_numpy()[index].sum(axis=1)
                        / denominator,
                        "deployed": daily._deployed.to_numpy()[index].sum(axis=1)
                        / denominator,
                        "fixed12": daily._fixed12.to_numpy()[index].sum(axis=1)
                        / denominator,
                    }
                    output.append(
                        {
                            "candidate_month": month,
                            "top_fraction": fraction,
                            "scope": scope,
                            "arm": arm,
                            "days": len(days),
                            "draws": draws,
                            "net_bps": weighted_mean(
                                local, f"net__{arm}", weight
                            )
                            * 10_000.0,
                            "net_ci_low_bps": float(
                                np.quantile(samples["net"], 0.025) * 10_000.0
                            ),
                            "net_ci_high_bps": float(
                                np.quantile(samples["net"], 0.975) * 10_000.0
                            ),
                            "paired_delta_vs_deployed_bps": weighted_mean(
                                local.assign(
                                    _delta=local[f"net__{arm}"]
                                    - local["net__deployed"]
                                ),
                                "_delta",
                                weight,
                            )
                            * 10_000.0,
                            "paired_delta_vs_deployed_ci_low_bps": float(
                                np.quantile(samples["deployed"], 0.025)
                                * 10_000.0
                            ),
                            "paired_delta_vs_deployed_ci_high_bps": float(
                                np.quantile(samples["deployed"], 0.975)
                                * 10_000.0
                            ),
                            "paired_delta_vs_fixed_12h_bps": weighted_mean(
                                local.assign(
                                    _delta=local[f"net__{arm}"]
                                    - local["net__fixed_12h"]
                                ),
                                "_delta",
                                weight,
                            )
                            * 10_000.0,
                            "paired_delta_vs_fixed_12h_ci_low_bps": float(
                                np.quantile(samples["fixed12"], 0.025)
                                * 10_000.0
                            ),
                            "paired_delta_vs_fixed_12h_ci_high_bps": float(
                                np.quantile(samples["fixed12"], 0.975)
                                * 10_000.0
                            ),
                        }
                    )
    return output


def _parity_rows(
    replay: pd.DataFrame, labels: pd.DataFrame, rows: pd.DataFrame
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    numeric_fields = {
        "execution_gross_ev_12h": "execution_gross_ev_12h",
        "execution_net_ev_12h": "execution_net_ev_12h",
        "execution_exit_hour": "execution_exit_hour",
        "execution_entry_price": "execution_entry_price",
        "execution_exit_price": "execution_exit_price",
        "execution_expected_spread_bps": "execution_expected_spread_bps",
        "execution_entry_half_spread_bps": "execution_entry_half_spread_bps",
        "execution_exit_half_spread_bps": "execution_exit_half_spread_bps",
    }
    for field, label_field in numeric_fields.items():
        delta = np.abs(
            replay[field].to_numpy(dtype=np.float64)
            - labels[label_field].to_numpy(dtype=np.float64)
        )
        output.append(
            {
                "field": field,
                "max_abs_delta": float(np.max(delta)),
                "mismatch_rows": int(np.sum(delta > 1e-10)),
                "passed": bool(np.max(delta) <= 1e-10),
            }
        )
    reason_match = replay.execution_exit_reason.astype(str).to_numpy() == (
        labels.execution_exit_reason.astype(str).to_numpy()
    )
    output.append(
        {
            "field": "execution_exit_reason",
            "max_abs_delta": 0.0 if reason_match.all() else None,
            "mismatch_rows": int((~reason_match).sum()),
            "passed": bool(reason_match.all()),
        }
    )
    geometry_match = rows.execution_geometry_key.astype(str).to_numpy() == (
        labels.execution_geometry_key.astype(str).to_numpy()
    )
    output.append(
        {
            "field": "execution_geometry_key",
            "max_abs_delta": 0.0 if geometry_match.all() else None,
            "mismatch_rows": int((~geometry_match).sum()),
            "passed": bool(geometry_match.all()),
        }
    )
    if not all(row["passed"] for row in output):
        raise ContractError("deployed simulator control parity failed")
    return output


def _promotion_gate(
    metrics: pd.DataFrame, bootstrap: pd.DataFrame
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    scopes = ("global", "side_long", "side_short")
    for arm in STATE_ARMS:
        local_metrics = metrics.loc[
            metrics.top_fraction.eq(0.10)
            & metrics.scope.isin(scopes)
            & metrics.arm.eq(arm)
        ]
        local_bootstrap = bootstrap.loc[
            bootstrap.top_fraction.eq(0.10)
            & bootstrap.scope.isin(scopes)
            & bootstrap.arm.eq(arm)
        ]
        conditions = {
            "positive_net_every_month_and_side": bool(
                len(local_metrics) == 6 and local_metrics.net_bps.gt(0).all()
            ),
            "delta_vs_deployed_ci_positive_every_month_and_side": bool(
                len(local_bootstrap) == 6
                and local_bootstrap.paired_delta_vs_deployed_ci_low_bps.gt(0).all()
            ),
            "delta_vs_fixed12_ci_positive_every_month_and_side": bool(
                len(local_bootstrap) == 6
                and local_bootstrap.paired_delta_vs_fixed_12h_ci_low_bps.gt(0).all()
            ),
        }
        records.append(
            {
                "arm": arm,
                **conditions,
                "passes_all_retrospective_diagnostic_gates": all(
                    conditions.values()
                ),
                "promotion_eligible": False,
            }
        )
    return pd.DataFrame(records)


def _p50_robustness_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for month, month_rows in frame.groupby("candidate_month", sort=True):
        for fraction in TOPS:
            weight = f"weight_top_{int(fraction * 100):02d}"
            active = month_rows.loc[month_rows[weight].gt(0)]
            scopes = [("global", active)] + [
                (f"side_{side}", local)
                for side, local in active.groupby("side_name", sort=True)
            ]
            for scope, local in scopes:
                canonical = weighted_mean(local, "net__P50", weight)
                exact = weighted_mean(local, "multi_exit_net__P50", weight)
                records.append(
                    {
                        "candidate_month": month,
                        "top_fraction": fraction,
                        "scope": scope,
                        "expected_selected_rows": float(local[weight].sum()),
                        "partial_exit_rate": weighted_mean(
                            local, "partial_exit__P50", weight
                        ),
                        "canonical_cost_bps": weighted_mean(
                            local, "cost__P50", weight
                        )
                        * 10_000.0,
                        "exact_multi_exit_fee_bps": weighted_mean(
                            local, "multi_exit_cost__P50", weight
                        )
                        * 10_000.0,
                        "canonical_cost_net_bps": canonical * 10_000.0,
                        "exact_multi_exit_net_bps": exact * 10_000.0,
                        "exact_minus_canonical_net_bps": (
                            exact - canonical
                        )
                        * 10_000.0,
                    }
                )
    return records


def run(output: Path = OUT, *, draws: int = 2_000) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    fixed_manifest = verify_seal(
        FIXED_ROOT, "fixed_horizon_action_ablation_v2"
    )
    selected_manifest = verify_seal(
        SELECTED_ROOT, "residual_selected_exit_opportunity_counterfactual_v3"
    )
    frame = pd.read_parquet(FIXED_ROOT / "paired_candidates.parquet")
    if frame.duplicated(list(IDENTITY), keep=False).any():
        raise ContractError("fixed controls contain duplicate identities")
    if frame["candidate_id"].duplicated().any():
        raise ContractError("fixed controls require unique candidate_id")
    frame = frame.reset_index(drop=True)
    frame["__order__"] = np.arange(len(frame))

    policy, policy_contract = _policy_contract(
        POLICY_PATH, horizon_minutes_override=720
    )
    candidates = _load_candidates(
        INPUT_ROOT / "candidates.parquet",
        INPUT_ROOT / "context.parquet",
        INPUT_ROOT / "path_targets.parquet",
        decision_delay_minutes=60,
        allow_subset=True,
    )
    candidates = frame.loc[:, list(IDENTITY) + ["__order__"]].merge(
        candidates,
        on="candidate_id",
        how="left",
        validate="one_to_one",
        suffixes=("__selected", ""),
    )
    if candidates["__barrier_pct__"].isna().any():
        raise ContractError("selected policy input coverage is incomplete")
    if (
        not candidates["side_name"].astype(str).eq(
            candidates["side_name__selected"].astype(str)
        ).all()
        or not pd.to_datetime(candidates["__ts__"], utc=True).eq(
            pd.to_datetime(candidates["__ts____selected"], utc=True)
        ).all()
        or not candidates["__symbol__"]
        .astype(str)
        .str.replace("/", "_", regex=False)
        .eq(candidates["__symbol____selected"].astype(str))
        .all()
    ):
        raise ContractError("selected policy input normalized identity mismatch")
    candidates, geometry = _resolved_geometry(candidates, policy)
    candidates = candidates.sort_values("__order__").reset_index(drop=True)
    if not candidates["__decision_ts__"].equals(
        pd.to_datetime(frame.execution_decision_utc, utc=True)
    ):
        raise ContractError("decision timestamps changed")

    path_rows = _load_selected_paths(frame)
    arrays = _parse_paths(path_rows.execution_future_path.reset_index(drop=True))
    simulations = _simulate_arms(
        candidates, arrays, _strategy_lookup(policy)
    )

    label_columns = [
        *IDENTITY,
        "execution_geometry_key",
        *SIM_FIELDS,
    ]
    labels = pd.read_parquet(
        LABEL_ROOT / "labels.parquet", columns=label_columns
    )
    if labels["candidate_id"].duplicated().any():
        raise ContractError("deployed labels contain duplicate candidate_id")
    labels = frame.loc[:, list(IDENTITY) + ["__order__"]].merge(
        labels,
        on="candidate_id",
        how="left",
        validate="one_to_one",
        suffixes=("__selected", ""),
    ).sort_values("__order__").reset_index(drop=True)
    if labels["execution_geometry_key"].isna().any():
        raise ContractError("deployed-label parity coverage is incomplete")
    if (
        not labels["side_name"].astype(str).eq(
            labels["side_name__selected"].astype(str)
        ).all()
        or not pd.to_datetime(labels["__ts__"], utc=True).eq(
            pd.to_datetime(labels["__ts____selected"], utc=True)
        ).all()
        or not labels["__symbol__"]
        .astype(str)
        .str.replace("/", "_", regex=False)
        .eq(labels["__symbol____selected"].astype(str))
        .all()
    ):
        raise ContractError("deployed-label normalized identity mismatch")
    parity = pd.DataFrame(_parity_rows(simulations["C0"], labels, candidates))

    # Fixed controls are already sealed in the parent paired artifact.
    for arm in ("deployed", *FIXED_ARMS):
        for prefix in ("gross", "net", "cost", "positive"):
            column = f"{prefix}__{arm}"
            if column not in frame:
                raise ContractError(f"fixed parent missing {column}")
    canonical_cost = frame["cost__deployed"].to_numpy(dtype=np.float64)
    if not np.allclose(canonical_cost, frame["cost"], atol=1e-12):
        raise ContractError("canonical row cost changed")
    for arm in STATE_ARMS:
        gross = simulations[arm]["execution_gross_ev_12h"].to_numpy(
            dtype=np.float64
        )
        frame[f"gross__{arm}"] = gross
        frame[f"cost__{arm}"] = canonical_cost
        frame[f"net__{arm}"] = gross - canonical_cost
        frame[f"positive__{arm}"] = frame[f"net__{arm}"].gt(0).astype(int)
        frame[f"exit_hour__{arm}"] = simulations[arm][
            "execution_exit_hour"
        ].to_numpy(dtype=np.float64)
        frame[f"exit_reason__{arm}"] = simulations[arm][
            "execution_exit_reason"
        ].astype(str).to_numpy()
        if arm == "P50":
            frame["partial_exit__P50"] = simulations["P50"][
                "partial_exit_mask"
            ].to_numpy(dtype=np.int8)
            frame["partial_exit_hour__P50"] = simulations["P50"][
                "partial_exit_hour"
            ].to_numpy(dtype=np.float64)
            frame["partial_exit_return__P50"] = simulations["P50"][
                "partial_exit_return"
            ].to_numpy(dtype=np.float64)
            frame["multi_exit_cost__P50"] = simulations["P50"][
                "multi_exit_fee_return"
            ].to_numpy(dtype=np.float64)
            frame["multi_exit_net__P50"] = (
                frame["gross__P50"] - frame["multi_exit_cost__P50"]
            )
        if not np.allclose(
            frame[f"gross__{arm}"] - frame[f"cost__{arm}"],
            frame[f"net__{arm}"],
            atol=1e-12,
        ):
            raise ContractError(f"{arm} fixed-cost accounting failed")

    c0 = simulations["C0"]
    effect_rows: list[dict[str, Any]] = []
    for month, local_index in frame.groupby("candidate_month", sort=True).groups.items():
        pos = np.asarray(list(local_index), dtype=np.int64)
        for arm in STATE_ARMS:
            gross_delta = np.abs(
                simulations[arm].execution_gross_ev_12h.to_numpy(
                    dtype=np.float64
                )[pos]
                - c0.execution_gross_ev_12h.to_numpy(dtype=np.float64)[pos]
            )
            hour_delta = np.abs(
                simulations[arm].execution_exit_hour.to_numpy(
                    dtype=np.float64
                )[pos]
                - c0.execution_exit_hour.to_numpy(dtype=np.float64)[pos]
            )
            reason_changed = (
                simulations[arm].execution_exit_reason.astype(str).to_numpy()[pos]
                != c0.execution_exit_reason.astype(str).to_numpy()[pos]
            )
            record = {
                "candidate_month": month,
                "arm": arm,
                "rows": len(pos),
                "gross_outcome_change_rate": float(np.mean(gross_delta > 1e-12)),
                "exit_hour_change_rate": float(np.mean(hour_delta > 1e-12)),
                "exit_reason_change_rate": float(np.mean(reason_changed)),
            }
            if arm == "D2":
                record["rows_reaching_changed_decay_threshold_rate"] = float(
                    np.mean(
                        c0.execution_exit_hour.to_numpy(dtype=np.float64)[pos]
                        > (121.0 / 60.0)
                    )
                )
            if arm == "P50":
                record["partial_exit_rate"] = float(
                    simulations["P50"].partial_exit_mask.to_numpy(
                        dtype=np.float64
                    )[pos].mean()
                )
                record["mean_partial_exit_hour_when_active"] = float(
                    simulations["P50"]
                    .partial_exit_hour.iloc[pos]
                    .loc[
                        simulations["P50"].partial_exit_mask.iloc[pos].astype(
                            bool
                        ).to_numpy()
                    ]
                    .mean()
                )
            effect_rows.append(record)
    effects = pd.DataFrame(effect_rows)

    metrics = pd.DataFrame(metric_rows(frame))
    bootstrap = pd.DataFrame(bootstrap_rows(frame, draws=draws))
    promotion = _promotion_gate(metrics, bootstrap)
    p50_robustness = pd.DataFrame(_p50_robustness_rows(frame))
    primary = metrics.loc[
        metrics.top_fraction.eq(0.10)
        & metrics.scope.isin(("global", "side_long", "side_short"))
        & metrics.arm.isin(STATE_ARMS)
    ].copy()
    diagnosis = {
        "schema": "frozen_exit_state_action_diagnosis_v4",
        "status": "DIAGNOSTIC_ONLY_NO_ARM_SELECTION_NO_PROMOTION",
        "primary_top10_metrics": primary.to_dict(orient="records"),
        "promotion_gate": promotion.to_dict(orient="records"),
        "partial_profit_arm_status": "P50_COMPLETED_CAUSAL_NEXT_BAR_OPEN",
    }

    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        output_values: dict[str, Any] = {
            "paired_candidates.parquet": frame.drop(columns=["__order__"]),
            "metrics.csv": metrics,
            "daily_bootstrap_ci.csv": bootstrap,
            "control_parity.csv": parity,
            "action_effects.csv": effects,
            "promotion_gate.csv": promotion,
            "p50_multi_exit_robustness.csv": p50_robustness,
            "diagnosis.json": diagnosis,
        }
        for name, value in output_values.items():
            if name.endswith(".parquet"):
                value.to_parquet(stage / name, index=False, compression="zstd")
            elif name.endswith(".csv"):
                value.to_csv(stage / name, index=False)
            else:
                write_json(stage / name, value)
        manifest = {
            "schema": "frozen_exit_state_action_ablation_v4",
            "status": "SEALED_DIAGNOSTIC_ONLY_UNCHANGED_BOOKS_NO_PROMOTION",
            "promotion_eligible": False,
            "arms": list(ARMS),
            "state_arm_contract": {
                "T4": (
                    "deployed geometry on first 240 observed 1m bars; normal "
                    "stops/trailing retain priority, then remaining rows use "
                    "the simulator timeout close-fill proxy"
                ),
                "D2": (
                    "deployed geometry except activation decay begins at bar "
                    "120, half-life 120 bars, asymptotic floor 50%; sticky "
                    "arming, caps, exit pressure and max-arm window unchanged"
                ),
                "W75": (
                    "deployed geometry except active trailing-width parameter "
                    "is multiplied by 0.75; fixed gap if positive, otherwise "
                    "giveback_beta; the simulator minimum gap floor remains"
                ),
                "P50": (
                    "at first causally known trailing activation, exit 50% "
                    "at the next-bar executable open using the deployed "
                    "spread/gap proxy; leave the remaining 50% in the exact "
                    "unchanged state machine"
                ),
            },
            "contract": {
                "selection": (
                    "exact sealed v3 candidate IDs and fractional pooled-global "
                    "monthly top-1/5/10/20 weights; no reranking"
                ),
                "cost": (
                    "sealed deployed canonical row cost reused exactly once "
                    "for every arm; simulator-return-dependent variant fees "
                    "are intentionally not used"
                ),
                "uncertainty": (
                    f"{draws} paired UTC-day clustered draws after freezing "
                    "IDs and fractional weights"
                ),
                "control_parity": (
                    "exact selected-row replay checks gross/net, exit bar and "
                    "reason, executable entry/exit prices, spread fields and "
                    "geometry key before any variant is accepted"
                ),
                "selection_of_arm": (
                    "forbidden on reused March/April diagnostics; all results "
                    "are retrospective mechanism evidence only"
                ),
            },
            "policy_contract": policy_contract,
            "geometry": geometry,
            "input_provenance": {
                "fixed_manifest_sha256": sha256(FIXED_ROOT / "manifest.json"),
                "fixed_paired_sha256": fixed_manifest["outputs_sha256"][
                    "paired_candidates.parquet"
                ],
                "selected_manifest_sha256": sha256(
                    SELECTED_ROOT / "manifest.json"
                ),
                "selected_paired_sha256": selected_manifest["outputs_sha256"][
                    "selected_counterfactuals.parquet"
                ],
                "labels_manifest_sha256": sha256(LABEL_ROOT / "manifest.json"),
                "labels_sha256": sha256(LABEL_ROOT / "labels.parquet"),
                "input_candidates_sha256": sha256(INPUT_ROOT / "candidates.parquet"),
                "input_context_sha256": sha256(INPUT_ROOT / "context.parquet"),
                "input_path_targets_sha256": sha256(
                    INPUT_ROOT / "path_targets.parquet"
                ),
                "path_manifest_sha256": sha256(PATH_ROOT / "manifest.json"),
                "paths_sha256": sha256(PATH_ROOT / "paths.parquet"),
                "policy_sha256": sha256(POLICY_PATH),
                "simulator_sha256": sha256(
                    ROOT / "extreme_price_movements/simple_policy_optimiser.py"
                ),
            },
            "outputs_sha256": {
                name: sha256(stage / name) for name in output_values
            },
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "limitations": [
                "March and April are reused diagnostics and cannot select a deployable action.",
                "No portfolio concurrency, exposure, asset-limit, or simple-policy replay is claimed.",
                "D2 threshold exposure is reported from control survival past bar 121; internal cap/pressure binding is not exposed by the simulator.",
                "W75 can be inert where the simulator minimum trailing-gap floor binds.",
                "P50 primary paired attribution reuses canonical row cost once; the output also records the exact two-exit fee return for later robustness analysis.",
            ],
        }
        write_json(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(
            f"{sha256(stage / 'manifest.json')}  manifest.json\n"
        )
        os.replace(stage, output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--bootstrap-draws", type=int, default=2_000)
    args = parser.parse_args()
    print(json.dumps(safe(run(args.output, draws=args.bootstrap_draws)), indent=2))


if __name__ == "__main__":
    main()
