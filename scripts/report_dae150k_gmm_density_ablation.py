#!/usr/bin/env python3
"""Report a controlled DAE150k/GMM density ablation.

Every arm is evaluated with the same label, cost, OOS month, global-top-k, and
frozen-parameter contracts.  The report deliberately ranks candidates by
top-10% meta EV first and worst-week top-10% EV second; it does not use a
density proxy to choose an economic winner.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.ae_gmm_economic_ablation import economic_metrics, split_months
from scripts.run_alternative_representation_search import _score_column_for_metrics


def _parse_ids(value: str) -> list[str]:
    values = [item.strip() for item in str(value).split(",") if item.strip()]
    if not values:
        raise ValueError("--density-ids must name at least one density arm")
    if len(values) != len(set(values)):
        raise ValueError("--density-ids contains duplicate identifiers")
    return values


def _overall_top10(metrics: pd.DataFrame, *, layer: str) -> dict[str, Any]:
    rows = metrics.loc[
        metrics["scope"].eq("overall") & metrics["top_frac"].eq(0.10)
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one overall top10 row for {layer}, found {len(rows)}")
    row = rows.iloc[0]
    return {
        f"{layer}_top10_ev_after_1pct": float(row["mean_ev_after_1pct"]),
        f"{layer}_top10_worst_week_ev": float(row["worst_week_ev"]),
        f"{layer}_top10_worst_month_ev": float(row["worst_month_ev"]),
        f"{layer}_top10_clean_exec_precision": float(row["clean_exec_precision"]),
        f"{layer}_top10_stop_or_adverse_rate": float(row["stop_or_adverse_rate"]),
        f"{layer}_top10_timeout_rate": float(row["timeout_rate"]),
        f"{layer}_top10_selected_rows": int(row["selected_rows"]),
        f"{layer}_top10_trades_per_day": float(row["trades_per_day"]),
    }


def _load_density_metadata(root: Path) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    for stage in (1, 2):
        path = root / f"density_stage{stage}_summary.csv"
        if path.exists():
            table = pd.read_csv(path)
            table["density_stage"] = stage
            tables.append(table)
    refinement = root / "overlap_refinement_summary.csv"
    if refinement.exists():
        table = pd.read_csv(refinement)
        if not table.empty:
            table["source_density_id"] = table["density_id"].astype(str)
            table["density_id"] = (
                table["source_density_id"]
                + "__"
                + table["metric"].astype(str)
                + "__lambda"
                + table["overlap_lambda"].astype(float).map(lambda value: f"{value:g}")
            )
            table["density_stage"] = "refined"
            tables.append(table)
    if not tables:
        return pd.DataFrame(columns=["density_id"])
    return pd.concat(tables, ignore_index=True, sort=False).drop_duplicates("density_id")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _adjacent_regularization_values(values: list[float], winner: float) -> list[float]:
    """Return a small log-spaced continuation only when `winner` is on a grid edge."""
    grid = sorted({float(value) for value in values})
    if len(grid) < 2 or float(winner) not in grid:
        return [float(winner)]
    index = grid.index(float(winner))
    if 0 < index < len(grid) - 1:
        return [float(winner)]
    if index == 0:
        ratio = grid[1] / grid[0]
        adjacent = grid[0] / ratio
        inward = grid[1]
    else:
        ratio = grid[-1] / grid[-2]
        adjacent = grid[-1] * ratio
        inward = grid[-2]
    # This is a regularization value, so preserve a positive finite value even
    # for unusual user-provided grids.
    if not math.isfinite(adjacent) or adjacent <= 0:
        return [float(winner)]
    return sorted({float(inward), float(winner), float(adjacent)})


def _adjacent_component_values(values: list[int], winner: int) -> list[int]:
    """Continue a component grid at an edge without expanding a settled winner."""
    grid = sorted({int(value) for value in values})
    if len(grid) < 2 or int(winner) not in grid:
        return [int(winner)]
    index = grid.index(int(winner))
    if 0 < index < len(grid) - 1:
        return [int(winner)]
    step = grid[1] - grid[0] if index == 0 else grid[-1] - grid[-2]
    adjacent = max(2, int(winner) - step) if index == 0 else int(winner) + step
    inward = grid[1] if index == 0 else grid[-2]
    return sorted({int(inward), int(winner), adjacent})


def _next_stage_recommendation(
    *,
    winner: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Make any edge follow-up explicit and reproducible.

    The initial panel is deliberately small. An economic winner at a panel edge
    needs a local continuation before covariance or overlap conclusions are
    drawn. Interior winners retain their selected value only.
    """
    stage1 = config["gmm_search"]["stage1"]
    winner_components = int(winner["components"])
    winner_reg = float(winner["reg_covar"])
    components = _adjacent_component_values(stage1["components"], winner_components)
    reg_covars = _adjacent_regularization_values(stage1["reg_covar"], winner_reg)
    component_expanded = len(components) > 1
    regularization_expanded = len(reg_covars) > 1
    return {
        "schema": "dae150k_gmm_density_boundary_expansion_v1",
        "source_density_id": str(winner["density_id"]),
        "source_economic_rank": int(winner["economic_rank"]),
        "reason": (
            "economic winner is on at least one initial-grid boundary"
            if component_expanded or regularization_expanded
            else "economic winner is interior to every initial-grid dimension"
        ),
        "boundary_expansion_required": bool(component_expanded or regularization_expanded),
        "components": components,
        "reg_covar": reg_covars,
        "latent_preprocessing": [str(winner["latent_preprocessing"])],
        "covariance_types": ["diag", "tied"],
        "notes": [
            "Evaluate these local continuation arms with the same frozen base/meta contract.",
            "Only proceed to Bhattacharyya refinement after this continuation and covariance comparison select one density contract.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--density-ids", required=True)
    parser.add_argument("--stage-label", required=True)
    parser.add_argument(
        "--search-config",
        type=Path,
        default=ROOT / "extreme_price_movements/config/dae150k_gmm_density_ablation_v1.json",
        help="GMM grid contract used to determine whether the economic winner is at a boundary.",
    )
    parser.add_argument("--oos-months", default="2026-02,2026-03,2026-04,2026-05,2026-06")
    parser.add_argument(
        "--baseline-density-id",
        default="",
        help="Optional arm used only for delta columns; winner selection is independent.",
    )
    args = parser.parse_args()
    density_ids = _parse_ids(args.density_ids)
    periods = split_months([value.strip() for value in args.oos_months.split(",") if value.strip()])
    report_dir = args.output_root / "gmm_economic_reports" / args.stage_label
    report_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    for density_id in density_ids:
        base_path = args.output_root / "base" / density_id / "best_oos_scored_ledger.parquet"
        meta_path = args.output_root / "meta" / density_id / "oos_predictions.parquet"
        if not base_path.exists():
            raise FileNotFoundError(f"Missing base ledger for {density_id}: {base_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing meta predictions for {density_id}: {meta_path}")
        base = pd.read_parquet(base_path)
        meta = pd.read_parquet(meta_path)
        meta_score = _score_column_for_metrics(meta)
        if meta_score is None:
            raise ValueError(f"Cannot identify meta score column for {density_id}")
        base_metrics = economic_metrics(
            base, arm=density_id, months=periods["base_oos"]
        )
        meta_metrics = economic_metrics(
            meta, arm=density_id, score_col=meta_score, months=periods["meta_oos"]
        )
        base_metrics.to_csv(report_dir / f"{density_id}__base_metrics.csv", index=False)
        meta_metrics.to_csv(report_dir / f"{density_id}__meta_metrics.csv", index=False)
        summary_rows.append(
            {
                "density_id": density_id,
                "meta_score_col": meta_score,
                **_overall_top10(base_metrics, layer="base"),
                **_overall_top10(meta_metrics, layer="meta"),
            }
        )
    summary = pd.DataFrame(summary_rows)
    metadata = _load_density_metadata(args.output_root)
    if not metadata.empty:
        summary = summary.merge(metadata, on="density_id", how="left", validate="one_to_one")
    baseline = str(args.baseline_density_id)
    if baseline:
        if baseline not in set(summary["density_id"]):
            raise ValueError(f"Baseline density arm is not in this report: {baseline}")
        reference = summary.loc[summary["density_id"].eq(baseline)].iloc[0]
        for column in (
            "base_top10_ev_after_1pct",
            "base_top10_worst_week_ev",
            "meta_top10_ev_after_1pct",
            "meta_top10_worst_week_ev",
            "meta_top10_clean_exec_precision",
            "meta_top10_stop_or_adverse_rate",
        ):
            summary[f"delta_vs_{baseline}__{column}"] = summary[column] - float(reference[column])
    summary = summary.sort_values(
        ["meta_top10_ev_after_1pct", "meta_top10_worst_week_ev", "base_top10_ev_after_1pct"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    summary.insert(0, "economic_rank", range(1, len(summary) + 1))
    summary.to_csv(report_dir / "base_meta_top10_stability_summary.csv", index=False)
    winner = summary.iloc[0].to_dict()
    config = json.loads(args.search_config.read_text(encoding="utf-8"))
    _write_json(
        report_dir / "winner.json",
        {
            "schema": "dae150k_gmm_density_economic_winner_v1",
            "stage_label": str(args.stage_label),
            "selection_rule": "highest meta global-top10 mean EV after 1% cost; worst-week top10 EV breaks ties",
            "base_oos_months": periods["base_oos"],
            "meta_oos_months": periods["meta_oos"],
            "winner": winner,
        },
    )
    _write_json(
        report_dir / "next_stage_recommendation.json",
        _next_stage_recommendation(winner=winner, config=config),
    )


if __name__ == "__main__":
    main()
