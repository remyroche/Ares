#!/usr/bin/env python3
"""Nested OOF exit-geometry scaling for current observable rank deciles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.simple_policy_1m_contextual import (  # noqa: E402
    geometry_scaled_params,
)
from scripts.materialize_execution_ev_policy_labels import (  # noqa: E402
    IDENTITY,
    PATH_COLUMNS,
    _load_candidates,
    _load_symbol_bars,
    _resolved_geometry,
    _simulate_batch,
    _strategy_maps,
)

SCALE_GRID = (0.80, 0.90, 1.00, 1.10, 1.20)
DEFAULT_SCORE = (
    "catboost__direct__without_hpo__mda_1se"
    "__recent_ev_catboost_predicted_archetype"
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
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
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


def _load_selected(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    ledger_columns = [
        *IDENTITY,
        args.fold_col,
        args.score_col,
        "execution_net_ev_12h",
        "execution_gross_ev_12h",
    ]
    ledger = pd.read_parquet(args.oof_ledger, columns=ledger_columns)
    ledger["__ts__"] = pd.to_datetime(ledger["__ts__"], utc=True, errors="raise")
    scored = ledger[args.fold_col].notna() & np.isfinite(
        pd.to_numeric(ledger[args.score_col], errors="coerce")
    )
    ledger = ledger.loc[scored].copy()
    if args.side_filter != "all":
        ledger = ledger.loc[
            ledger["side_name"].astype(str).str.lower().eq(args.side_filter)
        ].copy()
    take = int(np.ceil(float(args.top_fraction) * len(ledger)))
    selected = ledger.nlargest(take, args.score_col).copy()
    selected["global_score_rank"] = selected[args.score_col].rank(
        method="first", ascending=False
    )
    selected["global_score_rank_pct"] = selected["global_score_rank"] / len(ledger)

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
    selected = selected.merge(
        candidates.loc[:, keep],
        on=list(IDENTITY),
        how="inner",
        validate="one_to_one",
    )
    if len(selected) != take:
        raise ValueError(
            f"selected candidate/context coverage is incomplete: {len(selected)} != {take}"
        )
    selected = selected.sort_values(list(IDENTITY), kind="stable").reset_index(drop=True)
    selected["geometry_group"] = (
        selected["side_name"].astype(str)
        + "__"
        + selected["policy_archetype"].astype(str)
    )
    return selected, {"policy": policy, "geometry": geometry, "scored_rows": len(ledger)}


def _simulate_scale_grid(
    rows: pd.DataFrame,
    policy: Mapping[str, Any],
    *,
    data_root: Path,
    horizon_minutes: int,
    batch_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parents, _ = _strategy_maps(policy)
    n_rows = len(rows)
    net = np.full((n_rows, len(SCALE_GRID)), np.nan, dtype=np.float64)
    gross = np.full_like(net, np.nan)
    exit_hour = np.full_like(net, np.nan)
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
            target = positions[begin:stop]
            for side in ("long", "short"):
                local = np.flatnonzero(batch["side_name"].astype(str).eq(side).to_numpy())
                if not len(local):
                    continue
                for scale_index, scale in enumerate(SCALE_GRID):
                    strategy = geometry_scaled_params(parents[side], scale)
                    result = _simulate_batch(
                        batch.iloc[local].reset_index(drop=True),
                        tuple(matrix[local] for matrix in matrices),
                        strategy,
                    )
                    net[target[local], scale_index] = result[
                        "execution_net_ev_12h"
                    ].to_numpy(dtype=np.float64)
                    gross[target[local], scale_index] = result[
                        "execution_gross_ev_12h"
                    ].to_numpy(dtype=np.float64)
                    exit_hour[target[local], scale_index] = result[
                        "execution_exit_hour"
                    ].to_numpy(dtype=np.float64)
        if number == 1 or number % 25 == 0:
            print(
                f"[current-decile-geometry] symbols={number} rows_scored={np.isfinite(net).all(axis=1).sum()}",
                flush=True,
            )
    if not np.isfinite(net).all() or not np.isfinite(gross).all():
        raise ValueError("geometry sweep did not score every selected path")
    return net, gross, exit_hour


def _one_se_scale(
    net: np.ndarray,
    positions: np.ndarray,
    *,
    min_support: int,
) -> dict[str, Any]:
    support = int(len(positions))
    parent_index = SCALE_GRID.index(1.0)
    if support < int(min_support):
        return {
            "scale": 1.0,
            "support": support,
            "status": "insufficient_support",
        }
    values = net[positions]
    means = values.mean(axis=0)
    best = int(np.argmax(means))
    if best == parent_index:
        chosen = parent_index
    else:
        delta = values[:, best] - values[:, parent_index]
        se_delta = float(delta.std(ddof=1) / np.sqrt(max(support, 1)))
        if float(delta.mean()) <= se_delta:
            chosen = parent_index
        else:
            best_se = float(values[:, best].std(ddof=1) / np.sqrt(support))
            eligible = np.flatnonzero(means >= means[best] - best_se)
            chosen = min(eligible, key=lambda index: abs(SCALE_GRID[index] - 1.0))
    return {
        "scale": float(SCALE_GRID[chosen]),
        "support": support,
        "status": "selected",
        "parent_mean_net_return": float(means[parent_index]),
        "selected_mean_net_return": float(means[chosen]),
        "grid": [
            {"scale": float(scale), "mean_net_return": float(means[index])}
            for index, scale in enumerate(SCALE_GRID)
        ],
    }


def _metrics(
    net: np.ndarray,
    gross: np.ndarray,
    positions: np.ndarray,
    scale_indices: np.ndarray,
) -> dict[str, Any]:
    chosen_net = net[positions, scale_indices[positions]]
    chosen_gross = gross[positions, scale_indices[positions]]
    return {
        "rows": int(len(positions)),
        "mean_net_return": float(chosen_net.mean()),
        "mean_gross_return": float(chosen_gross.mean()),
        "sum_net_return": float(chosen_net.sum()),
        "positive_rate": float((chosen_net > 0.0).mean()),
    }


def _nested_evaluate(
    rows: pd.DataFrame,
    net: np.ndarray,
    gross: np.ndarray,
    *,
    fold_col: str,
    min_support: int,
) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray]:
    parent_index = SCALE_GRID.index(1.0)
    contextual_indices = np.full(len(rows), parent_index, dtype=np.int64)
    side_indices = np.full(len(rows), parent_index, dtype=np.int64)
    selections: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    folds = sorted(pd.to_numeric(rows[fold_col], errors="raise").astype(int).unique())
    for fold in folds:
        validation = np.flatnonzero(rows[fold_col].astype(int).to_numpy() == fold)
        training = np.flatnonzero(rows[fold_col].astype(int).to_numpy() < fold)
        fold_selection: dict[str, Any] = {"groups": {}, "sides": {}}
        if len(training):
            for group in sorted(rows["geometry_group"].unique()):
                fit = training[
                    rows.iloc[training]["geometry_group"].astype(str).to_numpy()
                    == str(group)
                ]
                detail = _one_se_scale(net, fit, min_support=min_support)
                fold_selection["groups"][str(group)] = detail
                apply = validation[
                    rows.iloc[validation]["geometry_group"].astype(str).to_numpy()
                    == str(group)
                ]
                contextual_indices[apply] = SCALE_GRID.index(detail["scale"])
            for side in ("long", "short"):
                fit = training[
                    rows.iloc[training]["side_name"].astype(str).to_numpy() == side
                ]
                detail = _one_se_scale(net, fit, min_support=min_support)
                fold_selection["sides"][side] = detail
                apply = validation[
                    rows.iloc[validation]["side_name"].astype(str).to_numpy() == side
                ]
                side_indices[apply] = SCALE_GRID.index(detail["scale"])
        selections[str(fold)] = fold_selection
        for arm, indices in (
            ("side_parent", np.full(len(rows), parent_index, dtype=np.int64)),
            ("side_only_nested", side_indices),
            ("side_x_decile_nested", contextual_indices),
        ):
            records.append(
                {
                    "fold": int(fold),
                    "arm": arm,
                    **_metrics(net, gross, validation, indices),
                }
            )
    return pd.DataFrame(records), selections, contextual_indices


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
    parser.add_argument(
        "--side-filter",
        choices=("all", "long", "short"),
        default="all",
        help="Eligibility ablation before the one global top-k auction.",
    )
    parser.add_argument("--decision-delay-minutes", type=int, default=60)
    parser.add_argument("--horizon-minutes", type=int, default=1440)
    parser.add_argument("--min-support", type=int, default=100)
    parser.add_argument("--batch-rows", type=int, default=250)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if not 0.0 < float(args.top_fraction) <= 1.0:
        raise ValueError("top-fraction must be in (0,1]")
    args.output_dir.mkdir(parents=True)
    rows, context = _load_selected(args)
    net, gross, exit_hour = _simulate_scale_grid(
        rows,
        context["policy"],
        data_root=args.data_root,
        horizon_minutes=args.horizon_minutes,
        batch_rows=args.batch_rows,
    )
    parent_index = SCALE_GRID.index(1.0)
    parity_delta = net[:, parent_index] - rows["execution_net_ev_12h"].to_numpy(
        dtype=np.float64
    )
    parity = {
        "max_abs_net_return_delta": float(np.max(np.abs(parity_delta))),
        "mean_abs_net_return_delta": float(np.mean(np.abs(parity_delta))),
    }
    if parity["max_abs_net_return_delta"] > 5e-7:
        raise ValueError(f"scale=1 parent replay parity failed: {parity}")
    fold_metrics, selections, contextual_indices = _nested_evaluate(
        rows,
        net,
        gross,
        fold_col=args.fold_col,
        min_support=args.min_support,
    )
    fold_metrics.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    scale_frame = rows.loc[
        :,
        [
            *IDENTITY,
            args.fold_col,
            args.score_col,
            "policy_archetype",
            "geometry_group",
        ],
    ].copy()
    for index, scale in enumerate(SCALE_GRID):
        scale_frame[f"net_return__scale_{scale:.2f}"] = net[:, index]
        scale_frame[f"gross_return__scale_{scale:.2f}"] = gross[:, index]
        scale_frame[f"exit_hour__scale_{scale:.2f}"] = exit_hour[:, index]
    scale_frame["nested_contextual_scale"] = np.asarray(SCALE_GRID)[
        contextual_indices
    ]
    scale_frame.to_parquet(
        args.output_dir / "selected_geometry_sweep.parquet",
        index=False,
        compression="zstd",
    )
    later = fold_metrics[fold_metrics["fold"] > fold_metrics["fold"].min()]
    aggregate_rows: list[dict[str, Any]] = []
    for arm, group in later.groupby("arm", sort=False):
        aggregate_rows.append(
            {
                "arm": str(arm),
                "rows": int(group["rows"].sum()),
                "mean_net_return": float(
                    np.average(group["mean_net_return"], weights=group["rows"])
                ),
                "mean_gross_return": float(
                    np.average(group["mean_gross_return"], weights=group["rows"])
                ),
                "sum_net_return": float(group["sum_net_return"].sum()),
                "positive_rate": float(
                    np.average(group["positive_rate"], weights=group["rows"])
                ),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(args.output_dir / "aggregate_metrics.csv", index=False)
    summary = {
        "schema": "current_decile_exit_geometry_ablation_v1",
        "status": "nested_outer_oof_not_promoted",
        "selection_contract": (
            "canonical recent-EV mapped score; one global pooled top-k across "
            "outer-OOF rows; fold geometry uses selected outcomes from earlier "
            "folds only; one-standard-error fallback to side parent"
        ),
        "score": args.score_col,
        "top_fraction": float(args.top_fraction),
        "scored_rows": int(context["scored_rows"]),
        "selected_rows": int(len(rows)),
        "scale_grid": list(SCALE_GRID),
        "parent_replay_parity": parity,
        "source_geometry_audit": context["geometry"],
        "fold_metrics": fold_metrics.to_dict(orient="records"),
        "aggregate_later_folds": aggregate.to_dict(orient="records"),
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
