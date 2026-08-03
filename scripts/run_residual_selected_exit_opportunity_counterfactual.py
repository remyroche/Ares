#!/usr/bin/env python3
"""Same-ID opportunity and exit-capture diagnostic for the residual control.

The runner never reranks candidates.  It freezes the mapped, pooled-global,
fractional-tie books from the sealed H0 residual control, joins exact 12-hour
path outcomes by candidate identity, and separates available opportunity from
deployed exit capture.  Every hindsight quantity is diagnostic only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCORE_ROOT = ROOT / "data_perp/artifacts/canonical_execution_reliability_exit_hurdle_ablation_20260730_v1"
PANEL_ROOT = ROOT / "data_perp/artifacts/canonical_execution_reliability_input_20260730_v4"
PATH_ROOT = ROOT / "data_perp/artifacts/febapr2025_top40_exact1m_paths_20260727_v1"
OUT = ROOT / "data_perp/artifacts/residual_selected_exit_opportunity_counterfactual_20260730_v3"
CONTROL = "H0__A0__score_residual_expected_ev"
IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
TOPS = (0.01, 0.05, 0.10, 0.20)
TIME = "execution_decision_utc"
NET = "execution_net_ev_12h"
GROSS = "execution_gross_ev_12h"
COST = "execution_cost_return"
BOOTSTRAP_FIELDS = (
    "deployed_net",
    "oracle_mfe_net",
    "fixed_12h_net",
    "oracle_regret",
    "opportunity_0bps",
    "opportunity_25bps",
    "opportunity_50bps",
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
        raise ContractError(f"missing manifest seal: {root}")
    if sha256(manifest_path) != seal_path.read_text().split()[0]:
        raise ContractError(f"manifest seal mismatch: {root}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != schema:
        raise ContractError(f"schema mismatch: {root}")
    for name, expected in manifest.get("outputs_sha256", {}).items():
        path = root / name
        if not path.is_file() or sha256(path) != expected:
            raise ContractError(f"sealed output mismatch: {path}")
    return manifest


def global_book_weights(
    frame: pd.DataFrame, score: str, fraction: float
) -> tuple[pd.Series, dict[str, float]]:
    """Fractional expected membership under a random boundary-tie draw."""
    if frame.empty:
        raise ContractError("cannot select an empty global book")
    count = max(1, int(math.ceil(len(frame) * fraction)))
    ordered = frame.sort_values(
        [score, "candidate_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    cutoff = float(ordered[score].iloc[count - 1])
    above = frame[score].gt(cutoff)
    tied = frame[score].eq(cutoff)
    need = int(count - above.sum())
    population = int(tied.sum())
    if need < 0 or population <= 0 or need > population:
        raise ContractError("invalid boundary-tie accounting")
    weights = pd.Series(0.0, index=frame.index)
    weights.loc[above] = 1.0
    weights.loc[tied] = need / population
    return weights, {
        "candidate_rows": int(len(frame)),
        "selected_rows": int(count),
        "cutoff": cutoff,
        "boundary_tie_population": population,
        "cutoff_tie_selected_share": float(need / count),
    }


def decode_fixed_12h(
    payload: str, decision_price: float, side: str
) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    parsed = json.loads(payload) if isinstance(payload, str) else payload
    close = np.asarray(parsed["close"], dtype=np.float64)
    timestamp = pd.to_datetime(
        np.asarray(parsed["timestamp"], dtype=np.int64), unit="ns", utc=True
    )
    if (
        close.shape != (720,)
        or timestamp.shape != (720,)
        or not np.isfinite(close).all()
        or not np.all(np.diff(timestamp.astype("int64")) == 60_000_000_000)
    ):
        raise ContractError("execution path must contain 720 contiguous finite minutes")
    if not np.isfinite(decision_price) or decision_price <= 0:
        raise ContractError("invalid native decision price")
    side_name = str(side).lower()
    if side_name not in {"long", "short"}:
        raise ContractError(f"unknown side: {side}")
    sign = 1.0 if side_name == "long" else -1.0
    return (
        float(sign * (close[719] / decision_price - 1.0)),
        pd.Timestamp(timestamp[0]),
        pd.Timestamp(timestamp[-1]),
    )


def load_fixed_paths(
    root: Path, selected_ids: set[str]
) -> tuple[pd.DataFrame, dict[str, str]]:
    path = root / "paths.parquet"
    source = pd.read_parquet(
        path,
        columns=[
            "candidate_id",
            "side_name",
            "__symbol__",
            "__ts__",
            "execution_future_path",
            "decision_price",
        ],
    )
    local = source.loc[source.candidate_id.astype(str).isin(selected_ids)]
    rows = []
    path_columns = [
        "candidate_id",
        "side_name",
        "__symbol__",
        "__ts__",
        "execution_future_path",
        "decision_price",
    ]
    for candidate_id, side_name, symbol, signal_utc, payload, decision_price in (
        local.loc[:, path_columns].itertuples(index=False, name=None)
    ):
        gross, start, end = decode_fixed_12h(
            payload,
            float(decision_price),
            str(side_name),
        )
        rows.append(
            {
                "candidate_id": candidate_id,
                "path_side_name": side_name,
                "path_symbol": symbol,
                "path_signal_utc": signal_utc,
                "path_start_utc": start,
                "path_last_minute_utc": end,
                "fixed_12h_gross": gross,
            }
        )
    result = pd.DataFrame(rows)
    used_hashes = {"paths.parquet": sha256(path)}
    if (
        len(result) != len(selected_ids)
        or result.empty
        or result.candidate_id.astype(str).duplicated().any()
        or set(result.candidate_id.astype(str)) != selected_ids
    ):
        raise ContractError("exact native path coverage failed for selected IDs")
    return result, used_hashes


def weighted_mean(frame: pd.DataFrame, column: str, weight: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce")
    weights = pd.to_numeric(frame[weight], errors="raise")
    valid = values.notna() & weights.gt(0)
    denominator = float(weights.loc[valid].sum())
    if denominator <= 0:
        return np.nan
    return float(np.dot(values.loc[valid], weights.loc[valid]) / denominator)


def weighted_support(frame: pd.DataFrame, column: str, weight: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce")
    weights = pd.to_numeric(frame[weight], errors="raise")
    return float(weights.loc[values.notna() & weights.gt(0)].sum())


def metric_row(
    frame: pd.DataFrame,
    *,
    month: str,
    fraction: float,
    scope: str,
    weight: str,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    denominator = float(frame[weight].sum())
    if denominator <= 0:
        raise ContractError("scope has no expected selected weight")
    result: dict[str, Any] = {
        "candidate_month": month,
        "top_fraction": fraction,
        "scope": scope,
        "candidate_rows": int(len(frame)),
        "expected_selected_rows": denominator,
        **selection,
    }
    means = {
        field: weighted_mean(frame, field, weight)
        for field in (
            "deployed_gross",
            "deployed_net",
            "cost",
            "oracle_mfe_gross",
            "oracle_mfe_net",
            "pre_exit_mfe_gross",
            "pre_exit_mfe_net",
            "fixed_12h_gross",
            "fixed_12h_net",
            "oracle_regret",
            "fixed_12h_delta_vs_deployed",
            "pre_exit_uncaptured_net_opportunity",
        )
    }
    for field, value in means.items():
        result[f"{field}_bps"] = value * 10_000.0
    result["pre_exit_mfe_expected_support"] = weighted_support(
        frame, "pre_exit_mfe_gross", weight
    )
    for field in (
        "opportunity_0bps",
        "opportunity_25bps",
        "opportunity_50bps",
        "deployed_positive",
        "fixed_12h_positive",
        "full_stop",
        "timeout",
    ):
        result[f"{field}_rate"] = weighted_mean(frame, field, weight)
    for field in ("capture_ratio", "economic_capture_ratio"):
        result[field] = weighted_mean(frame, field, weight)
        result[f"{field}_expected_support"] = weighted_support(frame, field, weight)
    result["gross_capture_ratio_of_means"] = (
        means["deployed_gross"] / means["oracle_mfe_gross"]
        if means["oracle_mfe_gross"] > 0
        else np.nan
    )
    for exit_name in ("full_stop", "timeout"):
        indicator = pd.to_numeric(frame[exit_name], errors="raise")
        result[f"{exit_name}_oracle_regret_contribution_bps"] = float(
            (frame[weight] * indicator * frame["oracle_regret"]).sum()
            / denominator
            * 10_000.0
        )
    return result


def exit_rows(
    frame: pd.DataFrame,
    *,
    month: str,
    fraction: float,
    weight: str,
) -> list[dict[str, Any]]:
    total = float(frame[weight].sum())
    rows = []
    for exit_class, local in frame.groupby("execution_exit_class", sort=True):
        local_weight = float(local[weight].sum())
        if local_weight <= 0:
            continue
        rows.append(
            {
                "candidate_month": month,
                "top_fraction": fraction,
                "execution_exit_class": str(exit_class),
                "expected_selected_rows": local_weight,
                "selected_book_share": local_weight / total,
                "deployed_net_conditional_bps": weighted_mean(
                    local, "deployed_net", weight
                )
                * 10_000.0,
                "oracle_mfe_net_conditional_bps": weighted_mean(
                    local, "oracle_mfe_net", weight
                )
                * 10_000.0,
                "oracle_regret_conditional_bps": weighted_mean(
                    local, "oracle_regret", weight
                )
                * 10_000.0,
                "oracle_regret_book_contribution_bps": float(
                    (local[weight] * local.oracle_regret).sum() / total * 10_000.0
                ),
                "fixed_12h_net_conditional_bps": weighted_mean(
                    local, "fixed_12h_net", weight
                )
                * 10_000.0,
            }
        )
    return rows


def bootstrap_rows(
    frame: pd.DataFrame,
    *,
    month: str,
    fraction: float,
    scope: str,
    weight: str,
    draws: int,
    seed: int,
) -> list[dict[str, Any]]:
    local = frame.loc[frame[weight].gt(0)].copy()
    local["day"] = pd.to_datetime(local[TIME], utc=True).dt.floor("D")
    days = sorted(local.day.unique())
    if len(days) < 2:
        return []
    rng = np.random.default_rng(seed)
    rows = []
    for field in BOOTSTRAP_FIELDS:
        valid = pd.to_numeric(local[field], errors="coerce").notna()
        daily = (
            local.loc[valid]
            .assign(
                _num=lambda x: x[weight] * pd.to_numeric(x[field], errors="raise"),
                _den=lambda x: x[weight],
            )
            .groupby("day", sort=True)[["_num", "_den"]]
            .sum()
            .reindex(days, fill_value=0.0)
        )
        index = rng.integers(0, len(days), size=(draws, len(days)))
        numerator = daily._num.to_numpy()[index].sum(axis=1)
        denominator = daily._den.to_numpy()[index].sum(axis=1)
        estimates = numerator / denominator
        scale = 10_000.0 if field not in {
            "opportunity_0bps",
            "opportunity_25bps",
            "opportunity_50bps",
        } else 1.0
        rows.append(
            {
                "candidate_month": month,
                "top_fraction": fraction,
                "scope": scope,
                "metric": field,
                "days": len(days),
                "draws": draws,
                "estimate": weighted_mean(local, field, weight) * scale,
                "ci_low": float(np.quantile(estimates, 0.025) * scale),
                "ci_high": float(np.quantile(estimates, 0.975) * scale),
                "unit": "bps" if scale == 10_000.0 else "rate",
            }
        )
    return rows


def load_inputs() -> tuple[pd.DataFrame, dict[str, Any]]:
    score_manifest = verify_seal(
        SCORE_ROOT, "canonical_execution_reliability_exit_hurdle_ablation_v1"
    )
    panel_manifest = verify_seal(
        PANEL_ROOT, "canonical_execution_reliability_input_v4"
    )
    scores = pd.read_parquet(SCORE_ROOT / "scores.parquet")
    scores = scores.loc[
        scores.config.eq(CONTROL) & scores.mapped_eligible.astype(bool),
        [*IDENTITY, TIME, "candidate_month", "mapped_score", "mapped_eligible"],
    ].copy()
    if scores.empty or scores.duplicated(list(IDENTITY)).any():
        raise ContractError("residual-control mapped identities are invalid")
    panel_columns = [
        *IDENTITY,
        TIME,
        GROSS,
        NET,
        COST,
        "execution_exit_class",
        "execution_exit_reason",
        "execution_mfe_return_12h",
        "pre_exit_mfe_return",
        "pre_exit_path_policy_parity",
        "target_pre_exit_capture_ratio",
        "target_pre_exit_economic_capture_ratio",
        "target_pre_exit_uncaptured_net_opportunity_return",
        "target_pre_exit_opportunity_0bps",
        "target_pre_exit_opportunity_25bps",
        "target_pre_exit_opportunity_50bps",
    ]
    # The three threshold labels live in the separately sealed target pack.
    threshold_columns = [column for column in panel_columns if column.startswith("target_pre_exit_opportunity_")]
    panel_columns = [column for column in panel_columns if column not in threshold_columns]
    panel = pd.read_parquet(PANEL_ROOT / "panel.parquet", columns=panel_columns)
    target_root = ROOT / "data_perp/artifacts/canonical_execution_reliability_target_pack_20260730_v1"
    target_manifest = verify_seal(
        target_root, "canonical_execution_reliability_target_pack_v1"
    )
    targets = pd.read_parquet(
        target_root / "labels.parquet",
        columns=[*IDENTITY, *threshold_columns],
    )
    panel = panel.merge(targets, on=list(IDENTITY), how="inner", validate="one_to_one")
    joined = scores.merge(panel, on=[*IDENTITY, TIME], how="inner", validate="one_to_one")
    if len(joined) != len(scores):
        raise ContractError("authoritative panel does not cover every mapped score")
    if not np.allclose(joined[GROSS] - joined[COST], joined[NET], atol=1e-10):
        raise ContractError("gross - cost != net")
    for column in threshold_columns:
        if not joined[column].isin([0, 1]).all():
            raise ContractError(f"non-binary opportunity label: {column}")
    provenance = {
        "score_manifest_sha256": sha256(SCORE_ROOT / "manifest.json"),
        "score_file_sha256": score_manifest["outputs_sha256"]["scores.parquet"],
        "panel_manifest_sha256": sha256(PANEL_ROOT / "manifest.json"),
        "panel_file_sha256": panel_manifest["outputs_sha256"]["panel.parquet"],
        "target_manifest_sha256": sha256(target_root / "manifest.json"),
        "target_file_sha256": target_manifest["outputs_sha256"]["labels.parquet"],
    }
    return joined, provenance


def run(output: Path = OUT, *, draws: int = 2_000) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    frame, provenance = load_inputs()
    path_manifest = json.loads((PATH_ROOT / "manifest.json").read_text())
    if (
        path_manifest.get("cost_accounting")
        != "fee_once_entry_spread_once_exit_spread_once"
        or path_manifest.get("coverage", {}).get("by_month", {}).get(
            "2025-03", {}
        ).get("coverage")
        != 1.0
        or path_manifest.get("coverage", {}).get("by_month", {}).get(
            "2025-04", {}
        ).get("coverage")
        != 1.0
    ):
        raise ContractError("exact execution-path contract failed")

    selection_meta: dict[tuple[str, float], dict[str, Any]] = {}
    for month, local in frame.groupby("candidate_month", sort=True):
        for fraction in TOPS:
            weight, meta = global_book_weights(local, "mapped_score", fraction)
            column = f"weight_top_{int(fraction * 100):02d}"
            frame.loc[local.index, column] = weight
            selection_meta[(str(month), fraction)] = meta
    weight_columns = [f"weight_top_{int(fraction * 100):02d}" for fraction in TOPS]
    selected = frame.loc[frame[weight_columns].max(axis=1).gt(0)].copy()
    paths, used_path_hashes = load_fixed_paths(
        PATH_ROOT, set(selected.candidate_id.astype(str))
    )
    selected = selected.merge(paths, on="candidate_id", how="inner", validate="one_to_one")
    selected["path_signal_utc"] = pd.to_datetime(selected.path_signal_utc, utc=True)
    selected["path_start_utc"] = pd.to_datetime(selected.path_start_utc, utc=True)
    selected["path_last_minute_utc"] = pd.to_datetime(
        selected.path_last_minute_utc, utc=True
    )
    selected["__ts__"] = pd.to_datetime(selected["__ts__"], utc=True)
    selected[TIME] = pd.to_datetime(selected[TIME], utc=True)
    symbol_normalized = (
        selected.path_symbol.astype(str).str.replace("/", "_", regex=False)
    )
    if (
        not selected.path_side_name.astype(str).eq(selected.side_name.astype(str)).all()
        or not selected.path_signal_utc.eq(selected["__ts__"]).all()
        or not selected.path_start_utc.eq(selected[TIME]).all()
        or not selected.path_last_minute_utc.eq(
            selected[TIME] + pd.Timedelta(minutes=719)
        ).all()
        or not symbol_normalized.eq(selected["__symbol__"].astype(str)).all()
    ):
        raise ContractError("execution-path identity/timestamp parity failed")
    selected["deployed_gross"] = selected[GROSS]
    selected["deployed_net"] = selected[NET]
    selected["cost"] = selected[COST]
    selected["oracle_mfe_gross"] = selected.execution_mfe_return_12h
    selected["oracle_mfe_net"] = selected.oracle_mfe_gross - selected.cost
    selected["pre_exit_mfe_gross"] = selected.pre_exit_mfe_return.where(
        selected.pre_exit_path_policy_parity.astype(bool)
    )
    selected["pre_exit_mfe_net"] = selected.pre_exit_mfe_gross - selected.cost
    selected["fixed_12h_net"] = selected.fixed_12h_gross - selected.cost
    selected["oracle_regret"] = selected.oracle_mfe_net - selected.deployed_net
    selected["fixed_12h_delta_vs_deployed"] = (
        selected.fixed_12h_net - selected.deployed_net
    )
    selected["pre_exit_uncaptured_net_opportunity"] = selected[
        "target_pre_exit_uncaptured_net_opportunity_return"
    ]
    selected["opportunity_0bps"] = selected.target_pre_exit_opportunity_0bps
    selected["opportunity_25bps"] = selected.target_pre_exit_opportunity_25bps
    selected["opportunity_50bps"] = selected.target_pre_exit_opportunity_50bps
    selected["deployed_positive"] = selected.deployed_net.gt(0).astype(int)
    selected["fixed_12h_positive"] = selected.fixed_12h_net.gt(0).astype(int)
    selected["full_stop"] = selected.execution_exit_class.eq("full_stop").astype(int)
    selected["timeout"] = selected.execution_exit_class.eq("timeout").astype(int)
    selected["capture_ratio"] = selected.target_pre_exit_capture_ratio
    selected["economic_capture_ratio"] = (
        selected.target_pre_exit_economic_capture_ratio
    )
    if (selected.oracle_regret < -1e-8).any():
        raise ContractError("MFE oracle is below deployed gross on at least one row")

    metrics: list[dict[str, Any]] = []
    exits: list[dict[str, Any]] = []
    bootstrap: list[dict[str, Any]] = []
    for month, month_rows in selected.groupby("candidate_month", sort=True):
        for fraction in TOPS:
            weight = f"weight_top_{int(fraction * 100):02d}"
            active = month_rows.loc[month_rows[weight].gt(0)].copy()
            meta = selection_meta[(str(month), fraction)]
            metrics.append(
                metric_row(
                    active,
                    month=str(month),
                    fraction=fraction,
                    scope="global",
                    weight=weight,
                    selection=meta,
                )
            )
            exits.extend(
                exit_rows(
                    active,
                    month=str(month),
                    fraction=fraction,
                    weight=weight,
                )
            )
            bootstrap.extend(
                bootstrap_rows(
                    active,
                    month=str(month),
                    fraction=fraction,
                    scope="global",
                    weight=weight,
                    draws=draws,
                    seed=20260730 + int(fraction * 100),
                )
            )
            for side, side_rows in active.groupby("side_name", sort=True):
                metrics.append(
                    metric_row(
                        side_rows,
                        month=str(month),
                        fraction=fraction,
                        scope=f"side_{side}",
                        weight=weight,
                        selection={
                            "selected_rows": meta["selected_rows"],
                            "cutoff": meta["cutoff"],
                            "boundary_tie_population": meta[
                                "boundary_tie_population"
                            ],
                            "cutoff_tie_selected_share": meta[
                                "cutoff_tie_selected_share"
                            ],
                        },
                    )
                )
                bootstrap.extend(
                    bootstrap_rows(
                        side_rows,
                        month=str(month),
                        fraction=fraction,
                        scope=f"side_{side}",
                        weight=weight,
                        draws=draws,
                        seed=20261730 + int(fraction * 100),
                    )
                )
    metrics_frame = pd.DataFrame(metrics)
    # Exact parity with the sealed control economics is required.
    economics = pd.read_csv(SCORE_ROOT / "economics.csv")
    parity_rows = []
    for month, stage in (("2025-03", "march_oof"), ("2025-04", "april_frozen_diagnostic")):
        for fraction in TOPS:
            expected = economics.loc[
                economics.config.eq(CONTROL)
                & economics.stage.eq(stage)
                & economics.window.eq("aggregate")
                & economics.score_kind.eq("mapped")
                & economics.top_fraction.eq(fraction)
            ]
            actual = metrics_frame.loc[
                metrics_frame.candidate_month.eq(month)
                & metrics_frame.top_fraction.eq(fraction)
                & metrics_frame.scope.eq("global")
            ]
            if len(expected) != 1 or len(actual) != 1:
                raise ContractError("sealed economics parity row missing")
            expected_row, actual_row = expected.iloc[0], actual.iloc[0]
            deltas = {
                "net_bps_delta": float(
                    actual_row.deployed_net_bps
                    - expected_row.random_tie_expected_net_bps
                ),
                "gross_bps_delta": float(
                    actual_row.deployed_gross_bps
                    - expected_row.random_tie_expected_gross_bps
                ),
                "cost_bps_delta": float(
                    actual_row.cost_bps
                    - expected_row.random_tie_expected_cost_bps
                ),
            }
            if max(abs(value) for value in deltas.values()) > 1e-9:
                raise ContractError("selected-book economics parity failed")
            parity_rows.append(
                {
                    "candidate_month": month,
                    "top_fraction": fraction,
                    **deltas,
                    "passed": True,
                }
            )

    decisions = []
    for _, row in metrics_frame.loc[metrics_frame.scope.eq("global")].iterrows():
        if row.oracle_mfe_net_bps <= 0:
            diagnosis = "insufficient_gross_opportunity_after_cost"
        elif row.deployed_net_bps <= 0:
            diagnosis = "positive_oracle_opportunity_but_exit_capture_failure"
        else:
            diagnosis = "positive_opportunity_and_positive_deployed_conversion"
        decisions.append(
            {
                "candidate_month": row.candidate_month,
                "top_fraction": row.top_fraction,
                "diagnosis": diagnosis,
                "oracle_mfe_net_bps": row.oracle_mfe_net_bps,
                "deployed_net_bps": row.deployed_net_bps,
                "fixed_12h_net_bps": row.fixed_12h_net_bps,
                "opportunity_25bps_rate": row.opportunity_25bps_rate,
            }
        )

    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        outputs: dict[str, Any] = {
            "selected_counterfactuals.parquet": selected,
            "metrics.csv": metrics_frame,
            "exit_attribution.csv": pd.DataFrame(exits),
            "daily_bootstrap_ci.csv": pd.DataFrame(bootstrap),
            "control_parity.csv": pd.DataFrame(parity_rows),
            "diagnosis.json": {
                "schema": "residual_selected_exit_opportunity_diagnosis_v3",
                "status": "DIAGNOSTIC_ONLY_NO_RERANKING_NO_PROMOTION",
                "rows": decisions,
                "decision_rule": {
                    "oracle_nonpositive": "repair upstream target/horizon/universe/cost design",
                    "oracle_positive_deployed_nonpositive": "repair separate timing/exit action layer",
                    "both_positive": "selector has economically convertible opportunity",
                },
            },
        }
        for name, value in outputs.items():
            path = stage / name
            if name.endswith(".parquet"):
                value.to_parquet(path, index=False, compression="zstd")
            elif name.endswith(".csv"):
                value.to_csv(path, index=False)
            else:
                write_json(path, value)
        manifest = {
            "schema": "residual_selected_exit_opportunity_counterfactual_v3",
            "status": "SEALED_DIAGNOSTIC_ONLY_IDENTICAL_ID_NO_RERANKING_NO_PROMOTION",
            "promotion_eligible": False,
            "control_config": CONTROL,
            "selection_contract": {
                "score": "causal mapped_score from sealed H0 residual control",
                "scope": "one pooled-global book within candidate month; never per timestamp or per side",
                "top_fractions": list(TOPS),
                "ties": "fractional expected random-boundary membership",
                "same_ids": "all counterfactual outcomes use the unchanged selected candidate IDs and weights",
            },
            "counterfactual_contract": {
                "oracle_mfe": "12h MFE minus the deployed row cost; hindsight ceiling only",
                "fixed_12h": "side-signed exact execution-path minute-720 close versus decision price, minus the canonical deployed row cost exactly once; diagnostic only",
                "pre_exit": "MFE through and including the labeled deployed exit minute, reported only on rows where exact policy-path parity holds",
                "regret": "oracle MFE net minus deployed net; equals MFE gross minus deployed gross",
                "uncertainty": f"{draws} UTC-day cluster bootstrap draws after freezing book weights",
            },
            "input_provenance": {
                **provenance,
                "path_manifest_sha256": sha256(PATH_ROOT / "manifest.json"),
                "execution_path_files_sha256": used_path_hashes,
            },
            "rows": {
                "mapped_candidate_rows": int(len(frame)),
                "selected_union_rows": int(len(selected)),
                "months": {
                    str(key): int(value)
                    for key, value in selected.candidate_month.value_counts().items()
                },
            },
            "outputs_sha256": {name: sha256(stage / name) for name in outputs},
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "limitations": [
                "MFE and fixed-time returns are hindsight diagnostics, not causal policy evidence.",
                "The fixed-time arm uses exact execution 1-minute candidate paths and subtracts the canonical cost once; it is not a claim that a market order at that close is executable without additional slippage.",
                "April is reused frozen diagnostic evidence and is not a new promotion holdout.",
            ],
        }
        write_json(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(
            f"{sha256(stage / 'manifest.json')}  manifest.json\n"
        )
        os.replace(stage, output)
    except Exception:
        import shutil

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
