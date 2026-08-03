#!/usr/bin/env python3
"""Bind independent meaningful-MFE events to the exact May--July score ledger.

This is a diagnostic-only materializer.  It neither fits a model nor changes
the frozen global selection policy.  All tails are pooled global first; side
views are attribution only.  The hourly triple-barrier labels describe path
opportunity, not executable exit-policy PnL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.materialize_source_separated_ic_ev_waterfall import (
    IDENTITY_COLUMNS,
    TOP_FRACTIONS,
    rank_ic,
    safe,
    score_columns,
    sha256,
    stable_top,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALLSCORE = ROOT / (
    "data_perp/artifacts/mayjul2026_exact_allscore_ic_ev_waterfall_20260730_v1/"
    "allscore_waterfall.parquet"
)
DEFAULT_ALLSCORE_MANIFEST = DEFAULT_ALLSCORE.with_name("manifest.json")
DEFAULT_GRID = ROOT / (
    "data_perp/artifacts/meaningful_mfe_exact_policy_label_grid_20260727_v1/"
    "meaningful_mfe_label_grid.parquet"
)
DEFAULT_GRID_MANIFEST = DEFAULT_GRID.with_name("manifest.json")
DEFAULT_OUTPUT = ROOT / (
    "data_perp/artifacts/meaningful_mfe_tail_recall_20260730_v1"
)
GRID_NAMES = (
    "h12_u1p5atr",
    "h12_u2p0atr",
    "h24_u1p5atr",
    "h24_u2p0atr",
)
PRIMARY_GRID = "h12_u1p5atr"
EXPECTED_ROWS = 127_777
SOURCE_FAMILY = "mayjul2026_exact12h_meaningful_mfe_tail_recall"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _declared_output(
    manifest_path: Path,
    path: Path,
    *,
    schema: str,
    output_key: str,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != schema:
        raise ValueError(f"unexpected manifest schema at {manifest_path}")
    record = manifest.get("outputs", {}).get(output_key, {})
    declared = Path(str(record.get("path")))
    if not declared.is_absolute():
        declared = ROOT / declared
    if declared.resolve() != path.resolve():
        raise ValueError(f"manifest path mismatch for {path}")
    if str(record.get("sha256")) != sha256(path):
        raise ValueError(f"manifest hash mismatch for {path}")
    return manifest


def _identity(
    frame: pd.DataFrame,
    source: str,
    *,
    uniqueness: Sequence[str] = IDENTITY_COLUMNS,
) -> pd.DataFrame:
    missing = sorted(set(IDENTITY_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"{source} missing identity columns: {missing}")
    result = frame.copy()
    result["candidate_id"] = result["candidate_id"].astype(str)
    result["side_name"] = result["side_name"].astype(str)
    result["__symbol__"] = result["__symbol__"].astype(str)
    result["__ts__"] = pd.to_datetime(result["__ts__"], utc=True, errors="raise")
    missing_uniqueness = sorted(set(uniqueness).difference(result.columns))
    if missing_uniqueness:
        raise ValueError(
            f"{source} missing uniqueness columns: {missing_uniqueness}"
        )
    if result.duplicated(list(uniqueness)).any():
        raise ValueError(f"{source} has duplicate identities")
    return result


def bind_opportunity_labels(
    allscore: pd.DataFrame,
    grid: pd.DataFrame,
    *,
    grid_names: Sequence[str] = GRID_NAMES,
    expected_rows: int = EXPECTED_ROWS,
) -> pd.DataFrame:
    """Join every declared grid cell without changing score-ledger identities."""

    anchor = _identity(allscore, "all-score ledger")
    if len(anchor) != expected_rows:
        raise ValueError(f"all-score rows {len(anchor)} != expected {expected_rows}")
    if anchor["candidate_id"].duplicated().any():
        raise ValueError("all-score ledger has duplicate candidate IDs")
    decision = pd.to_datetime(
        anchor["execution_decision_utc"], utc=True, errors="raise"
    )
    endpoint = pd.to_datetime(
        anchor["execution_label_end_utc"], utc=True, errors="raise"
    )
    if not endpoint.eq(decision + pd.Timedelta(hours=12)).all():
        raise ValueError("all-score ledger is not exact decision+12h")

    labels = _identity(
        grid,
        "meaningful-MFE grid",
        uniqueness=(*IDENTITY_COLUMNS, "grid_name"),
    )
    required = {
        "grid_name",
        "horizon_hours",
        "label_valid",
        "label_resolution_utc",
        "execution_decision_utc",
        "execution_net_ev_12h",
        "soft_label",
        "favorable_first",
        "adverse_first",
        "timeout",
        "oof_entry_atr_fraction",
        "upper_return",
        "peak_mfe_atr",
        "upper_atr",
    }
    missing = sorted(required.difference(labels.columns))
    if missing:
        raise ValueError(f"meaningful-MFE grid missing columns: {missing}")

    joined_cells: list[pd.DataFrame] = []
    for grid_name in grid_names:
        cell = labels.loc[
            labels["grid_name"].eq(grid_name) & labels["label_valid"].astype(bool)
        ].copy()
        if len(cell) < expected_rows:
            raise ValueError(f"{grid_name} has insufficient valid rows")
        if cell.duplicated(list(IDENTITY_COLUMNS)).any():
            raise ValueError(f"{grid_name} has duplicate identities")
        columns = [
            *IDENTITY_COLUMNS,
            "grid_name",
            "horizon_hours",
            "label_resolution_utc",
            "execution_decision_utc",
            "execution_net_ev_12h",
            "soft_label",
            "favorable_first",
            "adverse_first",
            "timeout",
            "oof_entry_atr_fraction",
            "upper_return",
            "peak_mfe_atr",
            "upper_atr",
        ]
        joined = anchor.merge(
            cell.loc[:, columns],
            on=list(IDENTITY_COLUMNS),
            how="left",
            validate="one_to_one",
            suffixes=("", "_grid"),
            indicator=True,
        )
        if not joined["_merge"].eq("both").all():
            raise ValueError(f"{grid_name} does not cover every all-score identity")
        joined = joined.drop(columns="_merge")
        grid_decision = pd.to_datetime(
            joined["execution_decision_utc_grid"], utc=True, errors="raise"
        )
        if not grid_decision.eq(decision).all():
            raise ValueError(f"{grid_name} decision timestamps disagree")
        resolution = pd.to_datetime(
            joined["label_resolution_utc"], utc=True, errors="raise"
        )
        horizon = pd.to_numeric(joined["horizon_hours"], errors="raise")
        expected_resolution = grid_decision + pd.to_timedelta(horizon, unit="h")
        if not resolution.eq(expected_resolution).all():
            raise ValueError(f"{grid_name} label resolution disagrees with horizon")
        net_delta = np.abs(
            pd.to_numeric(joined["execution_net_ev_12h"], errors="raise")
            - pd.to_numeric(joined["execution_net_ev_12h_grid"], errors="raise")
        )
        if float(net_delta.max()) > 1e-12:
            raise ValueError(f"{grid_name} exact-policy net does not match anchor")
        simplex = (
            joined["favorable_first"].astype(int)
            + joined["adverse_first"].astype(int)
            + joined["timeout"].astype(int)
        )
        if not simplex.eq(1).all():
            raise ValueError(f"{grid_name} path outcomes are not exhaustive")

        # "Any touch" uses the actual max(ATR multiple, return floor) upper
        # barrier and includes rows that touched it only after an adverse
        # barrier.  "Favorable first" is the conservative clean event after
        # accounting for that competing risk.  Neither is executable PnL.
        peak_return = (
            pd.to_numeric(joined["peak_mfe_atr"], errors="raise")
            * pd.to_numeric(joined["oof_entry_atr_fraction"], errors="raise")
        )
        upper_return = pd.to_numeric(joined["upper_return"], errors="raise")
        joined["meaningful_mfe_any_touch"] = (
            peak_return.ge(upper_return - 1e-12).astype(np.int8)
        )
        joined["meaningful_mfe_clean_first"] = (
            joined["favorable_first"].astype(np.int8)
        )
        joined["path_opportunity_above_exact_cost"] = (
            pd.to_numeric(joined["execution_mfe_return_12h"], errors="raise")
            > pd.to_numeric(joined["execution_cost_return"], errors="raise")
        ).astype(np.int8)
        joined["exact_net_positive"] = (
            pd.to_numeric(joined["execution_net_ev_12h"], errors="raise") > 0.0
        ).astype(np.int8)
        joined["mfe_to_gross_gap"] = (
            pd.to_numeric(joined["execution_mfe_return_12h"], errors="raise")
            - pd.to_numeric(joined["execution_gross_ev_12h"], errors="raise")
        )
        joined_cells.append(joined)

    result = pd.concat(joined_cells, ignore_index=True)
    if len(result) != expected_rows * len(grid_names):
        raise ValueError("joined opportunity grid has unexpected row count")
    return result


def _scopes(frame: pd.DataFrame):
    yield "pooled_global", frame
    for side, local in frame.groupby("side_name", sort=True, observed=True):
        yield f"side_{side}", local


def _binary_rank_metrics(score: pd.Series, target: pd.Series) -> tuple[float, float]:
    values = pd.to_numeric(score, errors="raise").to_numpy(float)
    labels = pd.to_numeric(target, errors="raise").to_numpy(int)
    if len(np.unique(labels)) < 2:
        return np.nan, np.nan
    return (
        float(roc_auc_score(labels, values)),
        float(average_precision_score(labels, values)),
    )


def event_rank_metrics(frame: pd.DataFrame, score: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    events = (
        "meaningful_mfe_any_touch",
        "meaningful_mfe_clean_first",
        "path_opportunity_above_exact_cost",
        "exact_net_positive",
    )
    for grid_name, grid_rows in frame.groupby("grid_name", sort=True, observed=True):
        for month, month_rows in grid_rows.groupby(
            "candidate_month", sort=True, observed=True
        ):
            for scope, local in _scopes(month_rows):
                for event in events:
                    auc, ap = _binary_rank_metrics(local[score], local[event])
                    rows.append(
                        {
                            "source_family": SOURCE_FAMILY,
                            "grid_name": str(grid_name),
                            "is_primary_grid": bool(grid_name == PRIMARY_GRID),
                            "candidate_month": str(month),
                            "scope": scope,
                            "score": score,
                            "event": event,
                            "rows": int(len(local)),
                            "event_rows": int(local[event].sum()),
                            "event_rate": float(local[event].mean()),
                            "rank_ic": rank_ic(local[score], local[event]),
                            "roc_auc": auc,
                            "average_precision": ap,
                        }
                    )
    return pd.DataFrame(rows)


def _conditional_mean(frame: pd.DataFrame, mask: pd.Series, column: str) -> float:
    values = pd.to_numeric(frame.loc[mask, column], errors="coerce").dropna()
    return float(values.mean()) if len(values) else np.nan


def tail_event_metrics(frame: pd.DataFrame, score: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for grid_name, grid_rows in frame.groupby("grid_name", sort=True, observed=True):
        for month, month_rows in grid_rows.groupby(
            "candidate_month", sort=True, observed=True
        ):
            for scope, local in _scopes(month_rows):
                for fraction in TOP_FRACTIONS:
                    selected = stable_top(local, score, fraction)
                    row: dict[str, Any] = {
                        "source_family": SOURCE_FAMILY,
                        "grid_name": str(grid_name),
                        "is_primary_grid": bool(grid_name == PRIMARY_GRID),
                        "candidate_month": str(month),
                        "scope": scope,
                        "score": score,
                        "fraction": float(fraction),
                        "candidate_rows": int(len(local)),
                        "selected_rows": int(len(selected)),
                    }
                    for event in (
                        "meaningful_mfe_any_touch",
                        "meaningful_mfe_clean_first",
                        "path_opportunity_above_exact_cost",
                        "exact_net_positive",
                    ):
                        population_events = int(local[event].sum())
                        selected_events = int(selected[event].sum())
                        selected_fraction = len(selected) / len(local)
                        recall = (
                            selected_events / population_events
                            if population_events
                            else np.nan
                        )
                        row[f"{event}_population_rate"] = float(local[event].mean())
                        row[f"{event}_selected_rate"] = float(selected[event].mean())
                        row[f"{event}_selected_events"] = selected_events
                        row[f"{event}_recall"] = recall
                        row[f"{event}_precision_lift"] = (
                            float(selected[event].mean() / local[event].mean())
                            if local[event].mean() > 0.0
                            else np.nan
                        )
                        row[f"{event}_recall_lift_vs_random"] = (
                            float(recall / selected_fraction)
                            if np.isfinite(recall)
                            else np.nan
                        )
                    incidence = selected["meaningful_mfe_any_touch"].astype(bool)
                    clean = selected["meaningful_mfe_clean_first"].astype(bool)
                    exact_opportunity = selected[
                        "path_opportunity_above_exact_cost"
                    ].astype(bool)
                    population_incidence = local[
                        "meaningful_mfe_any_touch"
                    ].astype(bool)
                    population_clean = local[
                        "meaningful_mfe_clean_first"
                    ].astype(bool)
                    population_exact_opportunity = local[
                        "path_opportunity_above_exact_cost"
                    ].astype(bool)
                    selected_given_touch = _conditional_mean(
                        selected, incidence, "exact_net_positive"
                    )
                    selected_given_clean = _conditional_mean(
                        selected, clean, "exact_net_positive"
                    )
                    selected_given_exact_opportunity = _conditional_mean(
                        selected, exact_opportunity, "exact_net_positive"
                    )
                    population_given_touch = _conditional_mean(
                        local, population_incidence, "exact_net_positive"
                    )
                    population_given_clean = _conditional_mean(
                        local, population_clean, "exact_net_positive"
                    )
                    population_given_exact_opportunity = _conditional_mean(
                        local,
                        population_exact_opportunity,
                        "exact_net_positive",
                    )
                    row.update(
                        {
                            "selected_soft_label_mean": float(
                                pd.to_numeric(
                                    selected["soft_label"], errors="raise"
                                ).mean()
                            ),
                            "selected_adverse_first_rate": float(
                                selected["adverse_first"].mean()
                            ),
                            "selected_timeout_rate": float(
                                selected["timeout"].mean()
                            ),
                            "population_net_positive_given_any_touch": population_given_touch,
                            "selected_net_positive_given_any_touch": selected_given_touch,
                            "net_positive_given_any_touch_lift": (
                                selected_given_touch / population_given_touch
                                if population_given_touch > 0.0
                                else np.nan
                            ),
                            "population_net_positive_given_clean_first": population_given_clean,
                            "selected_net_positive_given_clean_first": selected_given_clean,
                            "net_positive_given_clean_first_lift": (
                                selected_given_clean / population_given_clean
                                if population_given_clean > 0.0
                                else np.nan
                            ),
                            "population_net_positive_given_exact_cost_opportunity": population_given_exact_opportunity,
                            "selected_net_positive_given_exact_cost_opportunity": selected_given_exact_opportunity,
                            "net_positive_given_exact_cost_opportunity_lift": (
                                selected_given_exact_opportunity
                                / population_given_exact_opportunity
                                if population_given_exact_opportunity > 0.0
                                else np.nan
                            ),
                            "mean_mfe_bps": float(
                                selected["execution_mfe_return_12h"].mean() * 1e4
                            ),
                            "mean_gross_bps": float(
                                selected["execution_gross_ev_12h"].mean() * 1e4
                            ),
                            "mean_cost_bps": float(
                                selected["execution_cost_return"].mean() * 1e4
                            ),
                            "mean_net_bps": float(
                                selected["execution_net_ev_12h"].mean() * 1e4
                            ),
                            "mean_mfe_to_gross_gap_bps": float(
                                selected["mfe_to_gross_gap"].mean() * 1e4
                            ),
                            "mean_net_bps_given_any_touch": (
                                _conditional_mean(
                                    selected, incidence, "execution_net_ev_12h"
                                )
                                * 1e4
                            ),
                            "mean_net_bps_given_clean_first": (
                                _conditional_mean(
                                    selected, clean, "execution_net_ev_12h"
                                )
                                * 1e4
                            ),
                        }
                    )
                    rows.append(row)
    return pd.DataFrame(rows)


def run(
    *,
    allscore_path: Path = DEFAULT_ALLSCORE,
    allscore_manifest_path: Path = DEFAULT_ALLSCORE_MANIFEST,
    grid_path: Path = DEFAULT_GRID,
    grid_manifest_path: Path = DEFAULT_GRID_MANIFEST,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    allscore_manifest = _declared_output(
        allscore_manifest_path,
        allscore_path,
        schema="mayjul2026_exact_allscore_ic_ev_waterfall_v1",
        output_key="allscore_waterfall",
    )
    grid_manifest = _declared_output(
        grid_manifest_path,
        grid_path,
        schema="materialize_meaningful_mfe_label_grid_v1",
        output_key="labels",
    )
    allscore = pd.read_parquet(allscore_path)
    scores = score_columns(allscore)
    bound = bind_opportunity_labels(
        allscore, pd.read_parquet(grid_path), expected_rows=int(allscore_manifest["rows"])
    )
    rank = pd.concat(
        [event_rank_metrics(bound, score) for score in scores], ignore_index=True
    )
    tails = pd.concat(
        [tail_event_metrics(bound, score) for score in scores], ignore_index=True
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Any] = {}
    for name, table in (("event_rank_metrics", rank), ("tail_event_metrics", tails)):
        path = output_dir / f"{name}.parquet"
        table.to_parquet(path, index=False)
        outputs[name] = {
            "path": str(path),
            "rows": int(len(table)),
            "sha256": sha256(path),
        }
    manifest = {
        "schema": "meaningful_mfe_tail_recall_v1",
        "status": "DIAGNOSTIC_ONLY_NO_FIT_NO_MAPPING_NO_PROMOTION",
        "promotion_eligible": False,
        "rows_per_grid": int(allscore_manifest["rows"]),
        "grid_names": list(GRID_NAMES),
        "primary_grid": PRIMARY_GRID,
        "inputs": {
            "allscore": {
                "path": str(allscore_path),
                "sha256": sha256(allscore_path),
                "manifest_path": str(allscore_manifest_path),
                "manifest_sha256": sha256(allscore_manifest_path),
            },
            "meaningful_mfe_grid": {
                "path": str(grid_path),
                "sha256": sha256(grid_path),
                "manifest_path": str(grid_manifest_path),
                "manifest_sha256": sha256(grid_manifest_path),
                "source_sha256": grid_manifest.get("input", {}).get("sha256"),
            },
        },
        "contracts": {
            "selection": (
                "month-level pooled global top 1/5/10/20 is primary; side views "
                "are attribution only; score descending and candidate-ID tie break"
            ),
            "primary_event": (
                "h12_u1p5atr favorable barrier touched before adverse 1ATR barrier; "
                "same-hour conflict is adverse"
            ),
            "incidence_event": (
                "peak MFE return reaches the actual max(ATR multiple, return-floor) "
                "upper barrier, irrespective of whether adverse touched first"
            ),
            "economic_opportunity": (
                "exact 12h path MFE exceeds row-specific exact policy cost; distinct "
                "from net-positive capture"
            ),
            "horizon": (
                "12h grids match the executable policy horizon; 24h grids are "
                "sensitivity-only and observe 12 additional hours"
            ),
            "label_boundary": (
                "triple-barrier labels use hourly OHLC and do not replay the frozen "
                "exit policy; outcomes are targets/support labels only"
            ),
            "mapping": "no score mapping is read, fit or emitted",
        },
        "outputs": outputs,
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    (output_dir / "manifest.sha256").write_text(
        f"{sha256(manifest_path)}  manifest.json\n", encoding="utf-8"
    )
    return safe(manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allscore", type=Path, default=DEFAULT_ALLSCORE)
    parser.add_argument(
        "--allscore-manifest", type=Path, default=DEFAULT_ALLSCORE_MANIFEST
    )
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--grid-manifest", type=Path, default=DEFAULT_GRID_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                allscore_path=args.allscore,
                allscore_manifest_path=args.allscore_manifest,
                grid_path=args.grid,
                grid_manifest_path=args.grid_manifest,
                output_dir=args.output_dir,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
