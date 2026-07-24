#!/usr/bin/env python3
"""Detailed low-memory report for the market-state encoder ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import joblib
import pandas as pd


ARMS = ("parent_95", "ae_gmm", "mlp_gmm", "mlp_direct", "ae_mlp_gmm")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--previous-evcentric", type=Path, default=Path(
        "data_perp/reports/meta_market_state_threshold_calibration_parent95_20260712_v2/oos_predictions.parquet"
    ))
    args = p.parse_args()
    source = args.artifact / "oos_predictions.parquet"
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA memory_limit='2GB'")
    con.execute(f"CREATE VIEW src AS SELECT * FROM read_parquet('{source}')")
    budget = con.execute("SELECT count(*) FROM src WHERE policy_parent_rank >= 0.90").fetchone()[0]
    score_columns = {
        "parent_95": "policy_parent_rank",
        "ae_gmm": "rank_ae_gmm",
        "mlp_gmm": "rank_mlp_gmm",
        "mlp_direct": "rank_mlp_direct",
        "ae_mlp_gmm": "rank_ae_mlp_gmm",
    }
    if args.previous_evcentric.exists():
        con.execute(f"CREATE VIEW prior AS SELECT * FROM read_parquet('{args.previous_evcentric}')")
        con.execute(
            """CREATE VIEW src_prior AS
            SELECT s.*, p.rank_market_state AS rank_previous_evcentric
            FROM src s LEFT JOIN prior p
            USING (__ts__, __symbol__, side_name, archetype_policy_key)"""
        )
        table = "src_prior"
        score_columns["previous_evcentric"] = "rank_previous_evcentric"
    else:
        table = "src"
    details: list[pd.DataFrame] = []
    selected_views: list[str] = []
    for index, (arm, score) in enumerate(score_columns.items()):
        view = f"selected_{index}"
        con.execute(
            f"""CREATE TEMP VIEW {view} AS
            SELECT *, '{arm}' AS arm
            FROM {table}
            QUALIFY row_number() OVER (ORDER BY {score} DESC NULLS LAST) <= {budget}"""
        )
        selected_views.append(view)
    con.execute("CREATE TEMP VIEW selected AS " + " UNION ALL ".join(f"SELECT * FROM {v}" for v in selected_views))
    grouping_specs: dict[str, list[tuple[str, str]]] = {
        "overall": [],
        "month": [("strftime(__ts__, '%Y-%m')", "month")],
        "week": [("date_trunc('week', __ts__)", "week_start")],
        "side": [("side_name", "side_name")],
        "archetype": [("archetype_policy_key", "archetype_policy_key")],
        "side_archetype": [("side_name", "side_name"), ("archetype_policy_key", "archetype_policy_key")],
        "month_side_archetype": [("strftime(__ts__, '%Y-%m')", "month"), ("side_name", "side_name"), ("archetype_policy_key", "archetype_policy_key")],
    }
    for level, expressions in grouping_specs.items():
        select_group = "".join(
            f", {expression} AS {alias}" if expression != alias else f", {expression}"
            for expression, alias in expressions
        )
        group_names = [alias for _, alias in expressions]
        group_by = ", " + ", ".join(group_names) if group_names else ""
        query = f"""SELECT '{level}' AS level, arm{select_group}, count(*) AS rows,
            avg(ev_after_1pct) AS mean_net_ev_top10,
            sum(ev_after_1pct) AS sum_net_ev_top10,
            avg(clean_exec) AS clean_rate
            FROM selected GROUP BY arm{group_by}"""
        details.append(con.execute(query).df())
    pd.concat(details, ignore_index=True).to_csv(args.artifact / "detailed_breakdowns.csv", index=False)
    frontier: list[dict[str, float | str]] = []
    for arm in ("mlp_gmm", "mlp_direct"):
        raw = score_columns[arm]
        for shrink in (0.0, 0.10, 0.20, 0.25, 0.33, 0.50, 0.67, 0.75, 1.0):
            score = f"policy_parent_rank + {shrink} * ({raw} - policy_parent_rank)"
            q = f"""WITH picked AS (
                SELECT *, row_number() OVER (ORDER BY {score} DESC NULLS LAST) AS rn
                FROM {table}), weekly AS (
                SELECT date_trunc('week', __ts__) week_start, avg(ev_after_1pct) ev
                FROM picked WHERE rn <= {budget} GROUP BY 1), monthly AS (
                SELECT date_trunc('month', __ts__) month_start, avg(ev_after_1pct) ev
                FROM picked WHERE rn <= {budget} GROUP BY 1)
                SELECT (SELECT avg(ev_after_1pct) FROM picked WHERE rn <= {budget}),
                       (SELECT min(ev) FROM weekly),
                       (SELECT quantile_cont(ev, .10) FROM weekly),
                       (SELECT quantile_cont(ev, .20) FROM weekly),
                       (SELECT quantile_cont(ev, .30) FROM weekly),
                       (SELECT min(ev) FROM monthly)"""
            values = con.execute(q).fetchone()
            frontier.append(dict(
                arm=arm, shrink=shrink, mean_net_ev_top10=values[0],
                worst_week_net_ev_top10=values[1], q10_week_net_ev_top10=values[2],
                q20_week_net_ev_top10=values[3], q30_week_net_ev_top10=values[4],
                worst_month_net_ev_top10=values[5],
            ))
    pd.DataFrame(frontier).to_csv(args.artifact / "shrinkage_frontier_diagnostic.csv", index=False)
    payload = joblib.load(args.artifact / "encoder_calibrators.joblib")
    feature_rows = []
    model_rows = []
    for arm, arm_payload in payload.items():
        for model in arm_payload["models"]:
            model_rows.append({
                "arm": arm, "side": model.side, "archetype": model.archetype,
                "rows": model.rows,
                "selected_feature_count": len(model.features),
                "gmm_components": model.gmm.n_components if model.gmm is not None else None,
            })
            feature_rows.extend(
                {"arm": arm, "side": model.side, "archetype": model.archetype, "feature": feature}
                for feature in model.features
            )
    pd.DataFrame(model_rows).to_csv(args.artifact / "local_model_manifest.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(args.artifact / "selected_features.csv", index=False)
    (args.artifact / "report_manifest.json").write_text(json.dumps({
        "top10_budget": budget,
        "arms": list(score_columns),
        "previous_evcentric": str(args.previous_evcentric),
        "note": "Shrinkage frontier is OOS diagnostic only and must not be used for promotion selection.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
