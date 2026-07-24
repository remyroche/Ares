#!/usr/bin/env python3
"""Compare two base OOS ledgers on identical rows and label economics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd


TOP_K = (10, 20, 30)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _quoted(path: Path) -> str:
    return str(path).replace("'", "''")


def _metric_query(*, dimensions: list[str]) -> str:
    group_cols = ", ".join(dimensions)
    prefix = f"{group_cols}, " if group_cols else ""
    group_by = f"GROUP BY {group_cols}, model, top_k" if group_cols else "GROUP BY model, top_k"
    return f"""
        SELECT
            {prefix}model,
            top_k,
            count(*)::BIGINT AS selected_rows,
            count(DISTINCT symbol)::BIGINT AS symbols,
            count(DISTINCT CAST(ts AS DATE))::BIGINT AS days,
            count(*) / greatest(count(DISTINCT CAST(ts AS DATE)), 1) AS trades_per_day,
            avg(net_ev) AS net_ev_per_trade,
            sum(net_ev) AS net_ev_sum,
            avg(hit) AS hit_rate,
            avg(stop) AS stop_rate,
            avg(timeout) AS timeout_rate,
            avg(bad_mae) AS bad_mae_rate
        FROM selected_long
        {group_by}
        ORDER BY {prefix}model, top_k
    """


def _stability(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, top_k), group in selected.groupby(["model", "top_k"], sort=True):
        week = pd.to_numeric(group["net_ev_per_trade"], errors="coerce").dropna()
        month = pd.to_numeric(group["month_ev_per_trade"], errors="coerce").dropna()
        rows.append(
            {
                "model": model,
                "top_k": int(top_k),
                "worst_week_ev": float(week.min()) if len(week) else np.nan,
                "q10_week_ev": float(week.quantile(0.10)) if len(week) else np.nan,
                "q25_week_ev": float(week.quantile(0.25)) if len(week) else np.nan,
                "median_week_ev": float(week.median()) if len(week) else np.nan,
                "worst_month_ev": float(month.min()) if len(month) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _delta_table(table: pd.DataFrame, *, keys: list[str]) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame()
    new = table.loc[table["model"].eq("new")].drop(columns="model")
    old = table.loc[table["model"].eq("incumbent")].drop(columns="model")
    out = new.merge(old, on=keys, how="inner", suffixes=("_new", "_incumbent"))
    metric_bases = [
        col[: -len("_new")]
        for col in out.columns
        if col.endswith("_new")
        and pd.api.types.is_numeric_dtype(out[col])
    ]
    for metric in metric_bases:
        out[f"{metric}_delta"] = out[f"{metric}_new"] - out[f"{metric}_incumbent"]
    return out


def run(*, new_ledger: Path, incumbent_ledger: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA memory_limit='8GB'")
    # Artifact generations historically contain both UTC-aware and UTC-naive
    # timestamps.  Compare the represented UTC instant explicitly; DuckDB's
    # implicit TIMESTAMP/TIMESTAMPTZ coercion otherwise depends on the host
    # timezone and can shift joins across CET/CEST boundaries.
    con.execute("SET TimeZone='UTC'")
    con.execute(
        f"""
        CREATE TEMP VIEW new_raw AS
        SELECT
            epoch_ns("__ts__") AS ts_ns,
            make_timestamp_ns(epoch_ns("__ts__")) AS ts,
            "__symbol__" AS symbol,
            CAST(side AS TINYINT) AS side,
            side_name,
            "__archetype_label_family__" AS archetype,
            oos_fold,
            score,
            selected_top10,
            selected_top20,
            selected_top30,
            "__first_touch_capture_net__" AS net_ev,
            "__first_touch_hit__" AS hit,
            "__first_touch_stop__" AS stop,
            "__first_touch_timeout__" AS timeout,
            "__first_touch_mae_to_sl__" AS mae_to_sl
        FROM read_parquet('{_quoted(new_ledger)}')
        """
    )
    con.execute(
        f"""
        CREATE TEMP VIEW incumbent_raw AS
        SELECT
            epoch_ns("__ts__") AS ts_ns,
            make_timestamp_ns(epoch_ns("__ts__")) AS ts,
            "__symbol__" AS symbol,
            CAST(side AS TINYINT) AS side,
            score,
            selected_top10,
            selected_top20,
            selected_top30,
            "__first_touch_capture_net__" AS net_ev,
            "__first_touch_hit__" AS hit,
            "__first_touch_stop__" AS stop,
            "__first_touch_timeout__" AS timeout,
            "__first_touch_mae_to_sl__" AS mae_to_sl
        FROM read_parquet('{_quoted(incumbent_ledger)}')
        """
    )
    counts = con.execute(
        """
        SELECT
          (SELECT count(*) FROM new_raw) AS new_rows,
          (SELECT count(*) FROM incumbent_raw) AS incumbent_rows,
          (SELECT count(*) FROM new_raw n INNER JOIN incumbent_raw o USING(ts_ns, symbol, side)) AS overlap_rows,
          (SELECT count(*) - count(DISTINCT (ts_ns, symbol, side)) FROM new_raw) AS new_duplicate_keys,
          (SELECT count(*) - count(DISTINCT (ts_ns, symbol, side)) FROM incumbent_raw) AS incumbent_duplicate_keys
        """
    ).df().iloc[0].to_dict()
    con.execute(
        """
        CREATE TEMP TABLE overlap AS
        SELECT
            n.ts_ns,
            n.ts,
            n.symbol,
            n.side,
            n.side_name,
            coalesce(nullif(n.archetype, ''), 'missing') AS archetype,
            n.oos_fold,
            n.score AS new_score,
            o.score AS incumbent_score,
            n.selected_top10 AS new_top10,
            n.selected_top20 AS new_top20,
            n.selected_top30 AS new_top30,
            o.selected_top10 AS incumbent_top10,
            o.selected_top20 AS incumbent_top20,
            o.selected_top30 AS incumbent_top30,
            n.net_ev,
            n.hit,
            n.stop,
            n.timeout,
            CAST(n.mae_to_sl >= 1.0 AS DOUBLE) AS bad_mae,
            abs(n.net_ev - o.net_ev) AS label_net_abs_diff,
            CAST(
                coalesce(n.hit, -1) != coalesce(o.hit, -1)
                OR coalesce(n.stop, -1) != coalesce(o.stop, -1)
                OR coalesce(n.timeout, -1) != coalesce(o.timeout, -1)
                AS INTEGER
            ) AS label_event_mismatch
        FROM new_raw n
        INNER JOIN incumbent_raw o USING(ts_ns, symbol, side)
        """
    )
    parity = con.execute(
        """
        SELECT
            max(label_net_abs_diff) AS max_label_net_abs_diff,
            sum(label_event_mismatch)::BIGINT AS label_event_mismatch_rows,
            corr(new_score, incumbent_score) AS score_pearson_corr,
            avg(abs(new_score - incumbent_score)) AS score_mae
        FROM overlap
        """
    ).df().iloc[0].to_dict()
    con.execute(
        """
        CREATE TEMP VIEW selected_long AS
        SELECT ts, symbol, side_name, archetype,
               strftime(ts, '%Y-%m') AS month,
               CAST(date_trunc('week', ts) AS DATE) AS week_start,
               net_ev, hit, stop, timeout, bad_mae,
               'new' AS model, 10 AS top_k FROM overlap WHERE new_top10
        UNION ALL SELECT ts, symbol, side_name, archetype, strftime(ts, '%Y-%m'), CAST(date_trunc('week', ts) AS DATE), net_ev, hit, stop, timeout, bad_mae,
               'new', 20 FROM overlap WHERE new_top20
        UNION ALL SELECT ts, symbol, side_name, archetype, strftime(ts, '%Y-%m'), CAST(date_trunc('week', ts) AS DATE), net_ev, hit, stop, timeout, bad_mae,
               'new', 30 FROM overlap WHERE new_top30
        UNION ALL SELECT ts, symbol, side_name, archetype, strftime(ts, '%Y-%m'), CAST(date_trunc('week', ts) AS DATE), net_ev, hit, stop, timeout, bad_mae,
               'incumbent', 10 FROM overlap WHERE incumbent_top10
        UNION ALL SELECT ts, symbol, side_name, archetype, strftime(ts, '%Y-%m'), CAST(date_trunc('week', ts) AS DATE), net_ev, hit, stop, timeout, bad_mae,
               'incumbent', 20 FROM overlap WHERE incumbent_top20
        UNION ALL SELECT ts, symbol, side_name, archetype, strftime(ts, '%Y-%m'), CAST(date_trunc('week', ts) AS DATE), net_ev, hit, stop, timeout, bad_mae,
               'incumbent', 30 FROM overlap WHERE incumbent_top30
        """
    )
    tables = {
        "overall": con.execute(_metric_query(dimensions=[])).df(),
        "monthly": con.execute(_metric_query(dimensions=["month"])).df(),
        "weekly": con.execute(_metric_query(dimensions=["week_start"])).df(),
        "side": con.execute(_metric_query(dimensions=["side_name"])).df(),
        "side_archetype": con.execute(
            _metric_query(dimensions=["side_name", "archetype"])
        ).df(),
    }
    tables.update(
        {
            "overall_delta": _delta_table(tables["overall"], keys=["top_k"]),
            "monthly_delta": _delta_table(tables["monthly"], keys=["month", "top_k"]),
            "weekly_delta": _delta_table(tables["weekly"], keys=["week_start", "top_k"]),
            "side_delta": _delta_table(tables["side"], keys=["side_name", "top_k"]),
            "side_archetype_delta": _delta_table(
                tables["side_archetype"], keys=["side_name", "archetype", "top_k"]
            ),
        }
    )
    selection_overlap = con.execute(
        """
        SELECT top_k,
               both_rows,
               either_rows,
               both_rows / greatest(either_rows, 1) AS jaccard,
               both_rows / greatest(new_rows, 1) AS new_recall_of_incumbent,
               both_rows / greatest(incumbent_rows, 1) AS incumbent_recall_of_new
        FROM (
          SELECT 10 AS top_k,
                 sum(CAST(new_top10 AND incumbent_top10 AS BIGINT)) AS both_rows,
                 sum(CAST(new_top10 OR incumbent_top10 AS BIGINT)) AS either_rows,
                 sum(CAST(new_top10 AS BIGINT)) AS new_rows,
                 sum(CAST(incumbent_top10 AS BIGINT)) AS incumbent_rows FROM overlap
          UNION ALL SELECT 20,
                 sum(CAST(new_top20 AND incumbent_top20 AS BIGINT)),
                 sum(CAST(new_top20 OR incumbent_top20 AS BIGINT)),
                 sum(CAST(new_top20 AS BIGINT)), sum(CAST(incumbent_top20 AS BIGINT)) FROM overlap
          UNION ALL SELECT 30,
                 sum(CAST(new_top30 AND incumbent_top30 AS BIGINT)),
                 sum(CAST(new_top30 OR incumbent_top30 AS BIGINT)),
                 sum(CAST(new_top30 AS BIGINT)), sum(CAST(incumbent_top30 AS BIGINT)) FROM overlap
        )
        ORDER BY top_k
        """
    ).df()
    weekly_monthly = con.execute(
        """
        SELECT model, top_k, date_trunc('week', ts) AS period,
               avg(net_ev) AS net_ev_per_trade, NULL::DOUBLE AS month_ev_per_trade
        FROM selected_long GROUP BY model, top_k, period
        UNION ALL
        SELECT model, top_k, date_trunc('month', ts) AS period,
               NULL::DOUBLE, avg(net_ev)
        FROM selected_long GROUP BY model, top_k, period
        """
    ).df()
    stability = _stability(weekly_monthly)
    tables["selection_overlap"] = selection_overlap
    tables["stability"] = stability
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    manifest = {
        "schema": "exact_base_oos_retrain_comparison_v2",
        "new_ledger": str(new_ledger),
        "incumbent_ledger": str(incumbent_ledger),
        "row_contract": counts,
        "label_and_score_parity": parity,
        "selection_basis": "each model's stored fold-local selected_top10/20/30 flags on exact overlapping rows",
        "timestamp_contract": "join_on_utc_epoch_ns; UTC-naive timestamps are interpreted as UTC; calendar reporting is UTC",
        "cost_contract": "__first_touch_capture_net__ from the shared materialized label; label equality is audited",
        "outputs": {name: str(output_dir / f"{name}.csv") for name in tables},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-ledger", type=Path, required=True)
    parser.add_argument("--incumbent-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(json.dumps(_json_safe(run(**vars(args))), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
