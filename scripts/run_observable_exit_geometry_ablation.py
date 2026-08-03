#!/usr/bin/env python3
"""Nested OOF one-axis exit geometry on the observable global-top-k stream."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_execution_ev_policy_labels import (  # noqa: E402
    IDENTITY,
    PATH_COLUMNS,
    _load_candidates,
    _load_symbol_bars,
    _resolved_geometry,
    _simulate_batch,
    _strategy_maps,
)
from scripts.run_current_decile_exit_geometry_ablation import (  # noqa: E402
    DEFAULT_SCORE,
    _safe,
    _sha256,
    _write_json,
)

ARM_FACTORS: tuple[tuple[str, str | None, float], ...] = (
    ("parent", None, 1.0),
    ("stop_0p90", "sl_mult", 0.90),
    ("stop_1p10", "sl_mult", 1.10),
    ("activation_0p90", "trailing_activation_mult", 0.90),
    ("activation_1p10", "trailing_activation_mult", 1.10),
    ("giveback_0p90", "giveback_beta", 0.90),
    ("giveback_1p10", "giveback_beta", 1.10),
)
ARM_NAMES = tuple(item[0] for item in ARM_FACTORS)
PARENT_INDEX = ARM_NAMES.index("parent")
ECONOMIC_ARM_FACTORS: tuple[tuple[str, str | None, float], ...] = (
    ("parent", None, 1.0),
    ("hard_tp_1p25", "hard_tp_abs_pct", 0.0125),
    ("hard_tp_1p50", "hard_tp_abs_pct", 0.0150),
    ("hard_tp_2p00", "hard_tp_abs_pct", 0.0200),
    ("activation_0p50", "trailing_activation_mult", 0.50),
    ("giveback_0p50", "giveback_beta", 0.50),
    ("activation_cap_1p50", "trailing_activation_cap_pct", 0.0150),
)
BRACKET_ARM_FACTORS: tuple[tuple[str, str | None, float], ...] = (
    ("parent", None, 1.0),
    ("bracket_tp1p50_sl1p00", "__override__", 0.0),
    ("bracket_tp2p00_sl1p00", "__override__", 0.0),
    ("bracket_tp2p00_sl1p50", "__override__", 0.0),
    ("bracket_tp2p50_sl1p50", "__override__", 0.0),
    ("bracket_tp3p00_sl1p50", "__override__", 0.0),
    ("bracket_tp3p00_sl2p00", "__override__", 0.0),
)
BRACKET_OVERRIDES: dict[str, dict[str, float]] = {
    "bracket_tp1p50_sl1p00": {
        "hard_tp_abs_pct": 0.015,
        "sl_abs_cap_pct": 0.010,
    },
    "bracket_tp2p00_sl1p00": {
        "hard_tp_abs_pct": 0.020,
        "sl_abs_cap_pct": 0.010,
    },
    "bracket_tp2p00_sl1p50": {
        "hard_tp_abs_pct": 0.020,
        "sl_abs_cap_pct": 0.015,
    },
    "bracket_tp2p50_sl1p50": {
        "hard_tp_abs_pct": 0.025,
        "sl_abs_cap_pct": 0.015,
    },
    "bracket_tp3p00_sl1p50": {
        "hard_tp_abs_pct": 0.030,
        "sl_abs_cap_pct": 0.015,
    },
    "bracket_tp3p00_sl2p00": {
        "hard_tp_abs_pct": 0.030,
        "sl_abs_cap_pct": 0.020,
    },
}


def _variant(
    strategy: Mapping[str, Any],
    parameter: str | None,
    value: float,
    *,
    absolute: bool = False,
) -> dict[str, Any]:
    result = dict(strategy)
    if parameter is not None:
        result[parameter] = (
            float(value)
            if absolute
            else float(result[parameter]) * float(value)
        )
    return result


def _fold_local_top_k(
    ledger: pd.DataFrame,
    *,
    fold_col: str,
    score_col: str,
    top_fraction: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    selected_parts: list[pd.DataFrame] = []
    audit: list[dict[str, Any]] = []
    identity = list(IDENTITY)
    for fold, group in ledger.groupby(fold_col, sort=True):
        ranked = group.sort_values(
            [score_col, *identity],
            ascending=[False, *([True] * len(identity))],
            kind="stable",
        )
        count = max(1, int(np.ceil(float(top_fraction) * len(ranked))))
        chosen = ranked.iloc[:count].copy()
        chosen["fold_global_rank"] = np.arange(1, count + 1, dtype=np.int64)
        chosen["fold_global_rank_pct"] = chosen["fold_global_rank"] / len(ranked)
        selected_parts.append(chosen)
        digest = hashlib.sha256(
            "\n".join(
                "|".join(str(row[column]) for column in identity)
                for _, row in chosen.loc[:, identity].iterrows()
            ).encode()
        ).hexdigest()
        audit.append(
            {
                "fold": int(fold),
                "eligible_rows": int(len(ranked)),
                "selected_rows": int(count),
                "selected_fraction": float(count / len(ranked)),
                "selected_identity_sha256": digest,
            }
        )
    return pd.concat(selected_parts, ignore_index=True), audit


def _load_selected(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = [
        *IDENTITY,
        args.fold_col,
        args.score_col,
        "execution_net_ev_12h",
        "execution_gross_ev_12h",
    ]
    ledger = pd.read_parquet(args.oof_ledger, columns=columns)
    ledger["__ts__"] = pd.to_datetime(ledger["__ts__"], utc=True, errors="raise")
    ledger["side_name"] = ledger["side_name"].astype(str).str.lower()
    valid = ledger[args.fold_col].notna() & np.isfinite(
        pd.to_numeric(ledger[args.score_col], errors="coerce")
    )
    ledger = ledger.loc[valid].copy()
    ledger[args.fold_col] = pd.to_numeric(
        ledger[args.fold_col], errors="raise"
    ).astype(int)
    selected, admission_audit = _fold_local_top_k(
        ledger,
        fold_col=args.fold_col,
        score_col=args.score_col,
        top_fraction=args.top_fraction,
    )

    candidates = _load_candidates(
        args.candidates,
        args.context,
        args.path_targets,
        decision_delay_minutes=args.decision_delay_minutes,
    )
    policy = json.loads(args.policy_json.read_text(encoding="utf-8"))
    candidates, geometry = _resolved_geometry(candidates, policy)
    keep = [
        *IDENTITY,
        "__decision_ts__",
        "__barrier_pct__",
        "policy_archetype",
    ]
    rows = selected.merge(
        candidates.loc[:, keep],
        on=list(IDENTITY),
        how="inner",
        validate="one_to_one",
    )
    if len(rows) != len(selected):
        raise ValueError(
            "selected candidate/context coverage is incomplete: "
            f"{len(rows)} != {len(selected)}"
        )
    rows = rows.sort_values(
        [args.fold_col, "__ts__", "candidate_id"], kind="stable"
    ).reset_index(drop=True)
    rows["geometry_group"] = (
        rows["side_name"].astype(str)
        + "__"
        + rows["policy_archetype"].astype(str)
    )
    return rows, {
        "policy": policy,
        "geometry": geometry,
        "scored_rows": int(len(ledger)),
        "admission_audit": admission_audit,
    }


def _empty_result(n_rows: int, arm_names: Sequence[str] = ARM_NAMES) -> dict[str, np.ndarray]:
    shape = (n_rows, len(arm_names))
    return {
        "net": np.full(shape, np.nan, dtype=np.float64),
        "gross": np.full(shape, np.nan, dtype=np.float64),
        "cost": np.full(shape, np.nan, dtype=np.float64),
        "exit_hour": np.full(shape, np.nan, dtype=np.float64),
        "mfe": np.full(shape, np.nan, dtype=np.float64),
        "mae": np.full(shape, np.nan, dtype=np.float64),
        "exit_reason": np.full(shape, "", dtype=object),
    }


def _simulate_arm_grid(
    rows: pd.DataFrame,
    policy: Mapping[str, Any],
    *,
    data_root: Path,
    horizon_minutes: int,
    batch_rows: int,
    arm_factors: Sequence[tuple[str, str | None, float]] = ARM_FACTORS,
) -> dict[str, np.ndarray]:
    arm_names = tuple(item[0] for item in arm_factors)
    parents, _ = _strategy_maps(policy)
    output = _empty_result(len(rows), arm_names)
    for number, (symbol, indices) in enumerate(
        rows.groupby("__symbol__", sort=True).groups.items(), start=1
    ):
        positions = np.asarray(list(indices), dtype=np.int64)
        local_rows = rows.loc[positions].copy().reset_index(drop=True)
        start = local_rows["__decision_ts__"].min()
        end = local_rows["__decision_ts__"].max() + pd.Timedelta(
            minutes=horizon_minutes
        )
        bars = _load_symbol_bars(data_root, str(symbol), start, end)
        grid = pd.date_range(start, end, freq="min", inclusive="left", tz="UTC")
        values = bars.reindex(grid).loc[:, list(PATH_COLUMNS)].to_numpy(dtype=np.float32)
        offsets = (
            ((local_rows["__decision_ts__"] - start) / pd.Timedelta(minutes=1))
            .astype(np.int64)
            .to_numpy()
        )
        for begin in range(0, len(local_rows), int(batch_rows)):
            stop = min(begin + int(batch_rows), len(local_rows))
            batch = local_rows.iloc[begin:stop].copy().reset_index(drop=True)
            batch_offsets = offsets[begin:stop]
            matrices = tuple(
                np.stack(
                    [
                        values[offset : offset + horizon_minutes, column]
                        for offset in batch_offsets
                    ]
                )
                for column in range(len(PATH_COLUMNS))
            )
            if not all(np.isfinite(matrix).all() for matrix in matrices):
                raise ValueError(
                    f"incomplete selected 1m candidate path for {symbol}"
                )
            target = positions[begin:stop]
            for side in ("long", "short"):
                local = np.flatnonzero(
                    batch["side_name"].astype(str).eq(side).to_numpy()
                )
                if not len(local):
                    continue
                for arm_index, (arm, parameter, value) in enumerate(arm_factors):
                    absolute = (
                        arm.startswith("hard_tp_")
                        or arm.startswith("activation_cap_")
                    )
                    strategy = dict(parents[side])
                    if arm in BRACKET_OVERRIDES:
                        strategy.update(BRACKET_OVERRIDES[arm])
                    else:
                        strategy = _variant(
                            strategy,
                            parameter,
                            value,
                            absolute=absolute,
                        )
                    result = _simulate_batch(
                        batch.iloc[local].reset_index(drop=True),
                        tuple(matrix[local] for matrix in matrices),
                        strategy,
                    )
                    destination = (target[local], arm_index)
                    output["net"][destination] = result[
                        "execution_net_ev_12h"
                    ].to_numpy(dtype=np.float64)
                    output["gross"][destination] = result[
                        "execution_gross_ev_12h"
                    ].to_numpy(dtype=np.float64)
                    output["cost"][destination] = result[
                        "execution_cost_return"
                    ].to_numpy(dtype=np.float64)
                    output["exit_hour"][destination] = result[
                        "execution_exit_hour"
                    ].to_numpy(dtype=np.float64)
                    output["mfe"][destination] = result[
                        "execution_mfe_return_12h"
                    ].to_numpy(dtype=np.float64)
                    output["mae"][destination] = result[
                        "execution_mae_return_12h"
                    ].to_numpy(dtype=np.float64)
                    output["exit_reason"][destination] = result[
                        "execution_exit_reason"
                    ].astype(str).to_numpy()
        if number == 1 or number % 25 == 0:
            scored = np.isfinite(output["net"]).all(axis=1).sum()
            print(
                f"[observable-exit-geometry] symbols={number} rows_scored={scored}",
                flush=True,
            )
    for name in ("net", "gross", "cost", "exit_hour", "mfe", "mae"):
        if not np.isfinite(output[name]).all():
            raise ValueError(f"geometry sweep did not score every {name} value")
    if (output["exit_reason"] == "").any():
        raise ValueError("geometry sweep did not emit every exit reason")
    if not np.allclose(
        output["gross"] - output["cost"], output["net"], rtol=0.0, atol=1e-10
    ):
        raise ValueError("gross - cost != net in geometry sweep")
    return output


def _one_se_arm(
    net: np.ndarray,
    positions: np.ndarray,
    *,
    min_support: int,
    arm_names: Sequence[str] = ARM_NAMES,
) -> dict[str, Any]:
    support = int(len(positions))
    if support < int(min_support):
        return {
            "arm": "parent",
            "support": support,
            "status": "insufficient_support",
        }
    values = net[positions]
    means = values.mean(axis=0)
    best = int(np.argmax(means))
    if best == PARENT_INDEX:
        chosen = PARENT_INDEX
    else:
        delta = values[:, best] - values[:, PARENT_INDEX]
        delta_se = float(delta.std(ddof=1) / np.sqrt(max(support, 1)))
        chosen = best if float(delta.mean()) > delta_se else PARENT_INDEX
    return {
        "arm": arm_names[chosen],
        "support": support,
        "status": "selected",
        "parent_mean_net_return": float(means[PARENT_INDEX]),
        "selected_mean_net_return": float(means[chosen]),
        "grid": [
            {"arm": arm, "mean_net_return": float(means[index])}
            for index, arm in enumerate(arm_names)
        ],
    }


def _daily_block_ci(
    rows: pd.DataFrame,
    values: np.ndarray,
    *,
    seed: int = 20260727,
    draws: int = 2_000,
) -> tuple[float, float]:
    daily = (
        pd.DataFrame(
            {
                "day": pd.to_datetime(rows["__ts__"], utc=True).dt.floor("D"),
                "value": values,
            }
        )
        .groupby("day", sort=True)["value"]
        .mean()
        .to_numpy(dtype=np.float64)
    )
    if len(daily) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    sampled = daily[rng.integers(0, len(daily), size=(int(draws), len(daily)))]
    means = sampled.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _metric_record(
    rows: pd.DataFrame,
    result: Mapping[str, np.ndarray],
    positions: np.ndarray,
    arm_indices: np.ndarray,
    *,
    fold: int,
    policy_arm: str,
    scope: str,
) -> dict[str, Any]:
    chosen = arm_indices[positions]
    net = result["net"][positions, chosen]
    gross = result["gross"][positions, chosen]
    cost = result["cost"][positions, chosen]
    exit_hour = result["exit_hour"][positions, chosen]
    ci_low, ci_high = _daily_block_ci(rows.iloc[positions], net, seed=20260727 + fold)
    reasons = pd.Series(result["exit_reason"][positions, chosen]).value_counts()
    return {
        "fold": int(fold),
        "policy_arm": policy_arm,
        "scope": scope,
        "rows": int(len(positions)),
        "mean_net_return": float(net.mean()),
        "mean_net_bps": float(net.mean() * 10_000.0),
        "mean_gross_return": float(gross.mean()),
        "mean_cost_return": float(cost.mean()),
        "positive_rate": float((net > 0.0).mean()),
        "mean_exit_hour": float(exit_hour.mean()),
        "daily_block_ci_low_bps": float(ci_low * 10_000.0),
        "daily_block_ci_high_bps": float(ci_high * 10_000.0),
        "exit_reason_counts": json.dumps(
            {str(key): int(value) for key, value in reasons.items()},
            sort_keys=True,
        ),
    }


def _nested_evaluate(
    rows: pd.DataFrame,
    result: Mapping[str, np.ndarray],
    *,
    fold_col: str,
    min_support: int,
    arm_names: Sequence[str] = ARM_NAMES,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    n_rows = len(rows)
    parent = np.full(n_rows, PARENT_INDEX, dtype=np.int64)
    side_choice = parent.copy()
    contextual_choice = parent.copy()
    selections: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    fold_values = rows[fold_col].astype(int).to_numpy()
    folds = sorted(np.unique(fold_values))
    for fold in folds:
        validation = np.flatnonzero(fold_values == fold)
        training = np.flatnonzero(fold_values < fold)
        detail: dict[str, Any] = {"sides": {}, "groups": {}}
        for side in ("long", "short"):
            fit = training[
                rows.iloc[training]["side_name"].astype(str).to_numpy() == side
            ]
            selected = _one_se_arm(
                result["net"], fit, min_support=min_support, arm_names=arm_names
            )
            detail["sides"][side] = selected
            apply = validation[
                rows.iloc[validation]["side_name"].astype(str).to_numpy() == side
            ]
            side_choice[apply] = arm_names.index(selected["arm"])
        for group in sorted(rows["geometry_group"].astype(str).unique()):
            fit = training[
                rows.iloc[training]["geometry_group"].astype(str).to_numpy() == group
            ]
            selected = _one_se_arm(
                result["net"], fit, min_support=min_support, arm_names=arm_names
            )
            detail["groups"][group] = selected
            apply = validation[
                rows.iloc[validation]["geometry_group"].astype(str).to_numpy()
                == group
            ]
            contextual_choice[apply] = arm_names.index(selected["arm"])
        selections[str(fold)] = detail
        for policy_arm, choice in (
            ("side_parent", parent),
            ("side_only_nested", side_choice),
            ("side_x_decile_nested", contextual_choice),
        ):
            records.append(
                _metric_record(
                    rows,
                    result,
                    validation,
                    choice,
                    fold=fold,
                    policy_arm=policy_arm,
                    scope="global",
                )
            )
            for side in ("long", "short"):
                scoped = validation[
                    rows.iloc[validation]["side_name"].astype(str).to_numpy() == side
                ]
                if len(scoped):
                    records.append(
                        _metric_record(
                            rows,
                            result,
                            scoped,
                            choice,
                            fold=fold,
                            policy_arm=policy_arm,
                            scope=side,
                        )
                    )
    return (
        pd.DataFrame(records),
        selections,
        {
            "side_parent": parent,
            "side_only_nested": side_choice,
            "side_x_decile_nested": contextual_choice,
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-ledger", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--path-targets", type=Path, required=True)
    parser.add_argument("--policy-json", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data_perp"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-col", default=DEFAULT_SCORE)
    parser.add_argument(
        "--fold-col", default="execution_ev_model_ablation_oof_fold"
    )
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--decision-delay-minutes", type=int, default=60)
    parser.add_argument("--horizon-minutes", type=int, default=1440)
    parser.add_argument("--min-support", type=int, default=100)
    parser.add_argument("--batch-rows", type=int, default=250)
    parser.add_argument(
        "--family",
        choices=("local", "economic", "bracket"),
        default="local",
        help=(
            "local is the +/-10%% one-axis perturbation; economic tests hard "
            "take-profit and materially earlier/tighter capture policies; "
            "bracket couples a fixed profit target with a bounded stop"
        ),
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if not 0.0 < float(args.top_fraction) <= 1.0:
        raise ValueError("top-fraction must be in (0,1]")
    args.output_dir.mkdir(parents=True)
    arm_factors = {
        "local": ARM_FACTORS,
        "economic": ECONOMIC_ARM_FACTORS,
        "bracket": BRACKET_ARM_FACTORS,
    }[args.family]
    arm_names = tuple(item[0] for item in arm_factors)
    rows, context = _load_selected(args)
    result = _simulate_arm_grid(
        rows,
        context["policy"],
        data_root=args.data_root,
        horizon_minutes=args.horizon_minutes,
        batch_rows=args.batch_rows,
        arm_factors=arm_factors,
    )
    parity_delta = result["net"][:, PARENT_INDEX] - rows[
        "execution_net_ev_12h"
    ].to_numpy(dtype=np.float64)
    parity = {
        "max_abs_net_return_delta": float(np.max(np.abs(parity_delta))),
        "mean_abs_net_return_delta": float(np.mean(np.abs(parity_delta))),
    }
    if parity["max_abs_net_return_delta"] > 5e-7:
        raise ValueError(f"parent replay parity failed: {parity}")

    metrics, selections, choices = _nested_evaluate(
        rows,
        result,
        fold_col=args.fold_col,
        min_support=args.min_support,
        arm_names=arm_names,
    )
    metrics.to_csv(args.output_dir / "fold_scope_metrics.csv", index=False)
    candidate = rows.loc[
        :,
        [
            *IDENTITY,
            args.fold_col,
            args.score_col,
            "policy_archetype",
            "geometry_group",
            "fold_global_rank",
            "fold_global_rank_pct",
        ],
    ].copy()
    for arm_index, arm in enumerate(arm_names):
        for metric in ("net", "gross", "cost", "exit_hour", "mfe", "mae"):
            candidate[f"{metric}__{arm}"] = result[metric][:, arm_index]
        candidate[f"exit_reason__{arm}"] = result["exit_reason"][:, arm_index]
    for policy_arm, choice in choices.items():
        candidate[f"chosen_geometry_arm__{policy_arm}"] = np.asarray(arm_names)[
            choice
        ]
    candidate.to_parquet(
        args.output_dir / "candidate_geometry_replay.parquet",
        index=False,
        compression="zstd",
    )

    later = metrics[
        (metrics["fold"] > metrics["fold"].min()) & metrics["scope"].eq("global")
    ]
    aggregate = (
        later.groupby("policy_arm", sort=False)
        .apply(
            lambda group: pd.Series(
                {
                    "rows": int(group["rows"].sum()),
                    "mean_net_bps": float(
                        np.average(group["mean_net_bps"], weights=group["rows"])
                    ),
                    "mean_gross_bps": float(
                        np.average(
                            group["mean_gross_return"] * 10_000.0,
                            weights=group["rows"],
                        )
                    ),
                    "positive_rate": float(
                        np.average(group["positive_rate"], weights=group["rows"])
                    ),
                    "worst_fold_net_bps": float(group["mean_net_bps"].min()),
                    "latest_fold_net_bps": float(
                        group.loc[group["fold"].idxmax(), "mean_net_bps"]
                    ),
                    "latest_fold_rows": int(
                        group.loc[group["fold"].idxmax(), "rows"]
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    aggregate.to_csv(args.output_dir / "aggregate_metrics.csv", index=False)
    parent = aggregate.loc[aggregate["policy_arm"].eq("side_parent")].iloc[0]
    challengers = aggregate.loc[~aggregate["policy_arm"].eq("side_parent")].copy()
    challengers["delta_vs_parent_bps"] = (
        challengers["mean_net_bps"] - parent["mean_net_bps"]
    )
    promotable = challengers.loc[
        (challengers["mean_net_bps"] > 0.0)
        & (challengers["latest_fold_net_bps"] > 0.0)
        & (challengers["worst_fold_net_bps"] >= parent["worst_fold_net_bps"])
        & (challengers["delta_vs_parent_bps"] > 0.0)
    ]
    status = (
        "promotable_challenger_found"
        if len(promotable)
        else "nested_outer_oof_not_promoted"
    )
    summary = {
        "schema": "observable_exit_geometry_ablation_v1",
        "status": status,
        "selection_contract": (
            "one pooled global top-k independently inside each outer OOF fold; "
            "geometry for a fold uses only selected outcomes from earlier folds; "
            "one-standard-error fallback to side parent"
        ),
        "geometry_contract": {
            "one_axis_at_a_time": True,
            "arms": [
                {
                    "arm": arm,
                    "parameter": parameter,
                    "value": value,
                    "mode": (
                        "multi_parameter_absolute_override"
                        if arm in BRACKET_OVERRIDES
                        else (
                            "absolute"
                            if arm.startswith("hard_tp_")
                            or arm.startswith("activation_cap_")
                            else "factor"
                        )
                    ),
                    "overrides": BRACKET_OVERRIDES.get(arm),
                }
                for arm, parameter, value in arm_factors
            ],
            "family": args.family,
            "horizon_minutes": int(args.horizon_minutes),
            "simulator": (
                "extreme_price_movements.simple_policy_optimiser.simulate_and_score"
            ),
        },
        "score": args.score_col,
        "top_fraction": float(args.top_fraction),
        "scored_rows": int(context["scored_rows"]),
        "selected_rows": int(len(rows)),
        "admission_audit": context["admission_audit"],
        "parent_replay_parity": parity,
        "source_geometry_audit": context["geometry"],
        "aggregate_later_folds": aggregate.to_dict(orient="records"),
        "challengers": challengers.to_dict(orient="records"),
        "selections": selections,
        "sources": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in {
                "oof_ledger": args.oof_ledger,
                "candidates": args.candidates,
                "context": args.context,
                "path_targets": args.path_targets,
                "policy": args.policy_json,
            }.items()
        },
    }
    _write_json(args.output_dir / "summary.json", summary)
    return summary


def main() -> None:
    summary = run(_parser().parse_args())
    print(json.dumps(_safe(summary["aggregate_later_folds"]), indent=2))


if __name__ == "__main__":
    main()
