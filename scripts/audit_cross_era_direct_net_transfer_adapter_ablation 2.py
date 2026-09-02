#!/usr/bin/env python3
"""Post-label audit for a frozen raw-score direct-net transfer experiment.

This module never fits, selects, calibrates or maps a score.  It verifies the
label-free scoring artifact before joining exact 12-hour labels and reports
the raw global-book economics and the per-side raw-IC mapping prohibition.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_july_exact_preentry_heads import IDENTITY, sha256
from scripts.run_cross_era_direct_net_transfer_adapter_ablation import (
    SCHEMA,
    SIDES,
    _assert_identity,
    _hash,
    _month,
    _top_economics,
    _write_json,
    raw_tail_metrics,
)


def assert_complete_coverage(predictions: pd.DataFrame, labels: pd.DataFrame) -> dict[str, Any]:
    _assert_identity(predictions, "predictions")
    _assert_identity(labels, "labels")
    left = predictions.loc[:, list(IDENTITY)].sort_values(list(IDENTITY)).reset_index(drop=True)
    right = labels.loc[:, list(IDENTITY)].sort_values(list(IDENTITY)).reset_index(drop=True)
    if len(left) != len(right) or not left.equals(right):
        raise ValueError(f"current label identity coverage mismatch: predictions={len(left)} labels={len(right)}")
    return {"prediction_rows": int(len(left)), "label_rows": int(len(right)), "identity_complete_one_to_one": True}


def raw_ic_gate(frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    """Per-side/month raw IC and the explicit prohibition on mapping rescue."""

    work = frame.copy()
    work["month"] = _month(work)
    rows: list[dict[str, Any]] = []
    for side in SIDES:
        local_side = work.loc[work["side_name"].astype(str).eq(side)]
        for period, local in [("all", local_side), *((str(month), part) for month, part in local_side.groupby("month", sort=True))]:
            ic = float(local[score_column].corr(local["execution_net_ev_12h"], method="spearman")) if len(local) >= 3 else float("nan")
            rows.append({"side_name": side, "period": period, "rows": int(len(local)), "raw_rank_ic": ic, "mapping_eligible": bool(np.isfinite(ic) and ic >= 0.0), "mapping_prohibition": None if np.isfinite(ic) and ic >= 0.0 else "negative_or_undefined_raw_within_side_ic"})
    return pd.DataFrame(rows)


def probability_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["month"] = _month(work)
    net = pd.to_numeric(work["execution_net_ev_12h"], errors="raise").to_numpy(float) * 1e4
    rows: list[dict[str, Any]] = []
    for side in SIDES:
        side_mask = work["side_name"].astype(str).eq(side).to_numpy()
        for period in ("all", *sorted(work.loc[side_mask, "month"].astype(str).unique())):
            mask = side_mask if period == "all" else side_mask & work["month"].eq(period).to_numpy()
            for threshold, column in ((100.0, "p_loss_le_100"), (200.0, "p_loss_le_200")):
                y = (net[mask] <= -threshold).astype(int)
                p = np.clip(work.loc[mask, column].to_numpy(float), 1e-6, 1.0 - 1e-6)
                bins = np.minimum((p * 10).astype(int), 9)
                ece = sum((bins == index).mean() * abs(p[bins == index].mean() - y[bins == index].mean()) for index in range(10) if (bins == index).any())
                rows.append({"side_name": side, "period": period, "head": column, "rows": int(mask.sum()), "prevalence": float(y.mean()), "predicted_mean": float(p.mean()), "brier": float(brier_score_loss(y, p)), "ece10": float(ece), "roc_auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else np.nan, "pr_auc": float(average_precision_score(y, p)) if np.unique(y).size == 2 else np.nan})
    return pd.DataFrame(rows)


def promotion_gates(
    historical: pd.DataFrame,
    current: pd.DataFrame,
    score_column: str,
    old_to_recent: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    historical_economics, historical_ic = raw_tail_metrics(historical, score_column)
    current_economics, current_ic = raw_tail_metrics(current, score_column)
    history_monthly = _top_economics(historical, score_column)[1]
    current_side_local = _side_local_top10(current, score_column)
    gates = {
        "historical_global_top10_positive": historical_economics["global_top10_net_ev_bps"] > 0.0,
        "historical_all_months_positive": bool((history_monthly["net_ev_bps"] > 0.0).all()),
        "historical_latest_month_positive": historical_economics["latest_month_top10_net_ev_bps"] > 0.0,
        "historical_raw_ic_gate": bool(historical_ic["mapping_eligible"].all()),
        "current_global_top10_positive": current_economics["global_top10_net_ev_bps"] > 0.0,
        "current_long_local_top10_positive": current_side_local["long"] > 0.0,
        "current_short_local_top10_positive": current_side_local["short"] > 0.0,
        "current_raw_ic_gate": bool(current_ic["mapping_eligible"].all()),
        "old_to_recent_global_top10_positive": bool(
            old_to_recent is not None
            and float(old_to_recent["global_top10_net_ev_bps"]) > 0.0
        ),
        "old_to_recent_raw_ic_gate": bool(
            old_to_recent is not None
            and bool(old_to_recent["raw_ic_gate_passed"])
        ),
    }
    gates["mapping_authorized"] = False
    gates["portfolio_replay_authorized"] = bool(all(value for key, value in gates.items() if key not in {"mapping_authorized", "portfolio_replay_authorized"}))
    return gates


def _side_local_top10(frame: pd.DataFrame, score_column: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for side in SIDES:
        local = frame.loc[frame["side_name"].astype(str).eq(side)].sort_values([score_column, "candidate_id"], ascending=[False, True], kind="stable")
        chosen = local.iloc[: max(1, int(math.ceil(.10 * len(local))))]
        result[side] = float(chosen["execution_net_ev_12h"].mean() * 1e4)
    return result


def _verify_frozen(source_dir: Path) -> dict[str, Any]:
    frozen_path = source_dir / "frozen_before_current_evaluation.json"
    frozen = json.loads(frozen_path.read_text())
    if frozen.get("schema") != SCHEMA:
        raise ValueError("unexpected frozen source schema")
    for name, record in frozen.get("outputs", {}).items():
        path = Path(record["path"])
        if path.exists() and sha256(path) != record["sha256"]:
            raise ValueError(f"frozen output hash mismatch: {name}")
    return frozen


def _verify_scoring(scoring_dir: Path) -> dict[str, Any]:
    report_path = scoring_dir / "report.json"
    report = json.loads(report_path.read_text())
    if report.get("schema") != SCHEMA or report.get("status") != "scored_label_free_before_current_outcomes":
        raise ValueError("scoring report does not prove a label-free frozen score")
    prediction = report["outputs"]["predictions"]
    if sha256(Path(prediction["path"])) != prediction["sha256"]:
        raise ValueError("scored prediction hash mismatch")
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    frozen = _verify_frozen(args.source_dir)
    scored_report = _verify_scoring(args.scoring_dir)
    frozen_path = args.source_dir / "frozen_before_current_evaluation.json"
    if scored_report["frozen_source"]["sha256"] != sha256(frozen_path):
        raise ValueError("scoring artifact is bound to a different frozen selection")
    predictions = pd.read_parquet(scored_report["outputs"]["predictions"]["path"])
    labels = pd.read_parquet(args.current_labels)
    coverage = assert_complete_coverage(predictions, labels)
    if "execution_net_ev_12h" not in labels:
        raise ValueError("current labels lack exact execution_net_ev_12h")
    if any(column.startswith("mapped_") for column in predictions.columns):
        raise AssertionError("raw-score transfer audit refuses mapped score columns")
    current = predictions.merge(labels.loc[:, [*IDENTITY, "execution_net_ev_12h"]], on=list(IDENTITY), how="inner", validate="one_to_one")
    winner_column = str(frozen["winner"]["score_column"])
    history = pd.read_parquet(frozen["outputs"]["historical_oof_all_arms"]["path"])
    historical = history.loc[:, [column for column in history.columns if column != "execution_net_ev_12h" or True]].copy()
    # The historical table contains all arms.  Its selected score already has
    # the final raw column; no rescore/reselection occurs here.
    historical_score = winner_column
    if historical_score not in historical or historical_score not in current:
        raise ValueError(f"frozen winner score unavailable in both frames: {historical_score}")
    transfer = pd.read_csv(frozen["outputs"]["old_to_recent_transfer"]["path"])
    old_to_recent_rows = transfer.loc[
        transfer["transfer"].eq("old_to_recent_forward")
        & transfer["arm"].eq(str(frozen["winner"]["arm"]))
    ]
    if len(old_to_recent_rows) != 1:
        raise ValueError(
            "frozen winner lacks exactly one old-to-recent transfer record"
        )
    old_to_recent = old_to_recent_rows.iloc[0].to_dict()
    gates = promotion_gates(
        historical,
        current,
        winner_column,
        old_to_recent=old_to_recent,
    )
    tables = {
        "current_scored_exact": current,
        "current_raw_ic": raw_ic_gate(current, winner_column),
        "historical_raw_ic": raw_ic_gate(historical, historical_score),
        "current_probability_metrics": probability_metrics(current),
        "current_global_economics": _top_economics(current, winner_column)[1],
    }
    args.output_dir.mkdir(parents=True)
    outputs: dict[str, Any] = {}
    for name, table in tables.items():
        suffix = ".parquet" if name == "current_scored_exact" else ".csv"
        path = args.output_dir / f"{name}{suffix}"
        table.to_parquet(path, index=False) if suffix == ".parquet" else table.to_csv(path, index=False)
        outputs[name] = {**_hash(path), "rows": int(len(table))}
    report = {"schema": SCHEMA, "status": "completed_post_label_research_only", "promotion_eligible": False, "portfolio_replay_authorized": bool(gates["portfolio_replay_authorized"]), "mapping_authorized": False, "coverage": coverage, "winner": frozen["winner"], "old_to_recent_winner_transfer": old_to_recent, "gates": gates, "contract": "raw scores only; negative/undefined raw within-side IC forbids mapping rescue", "sources": {"frozen": _hash(args.source_dir / "frozen_before_current_evaluation.json"), "scoring": _hash(args.scoring_dir / "report.json"), "labels": _hash(args.current_labels)}, "outputs": outputs}
    report_path = args.output_dir / "report.json"
    _write_json(report_path, report)
    _write_json(args.output_dir / "manifest.json", {"schema": SCHEMA, "status": report["status"], "report": _hash(report_path), "outputs": outputs})
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source-dir", type=Path, required=True)
    value.add_argument("--scoring-dir", type=Path, required=True)
    value.add_argument("--current-labels", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    return value


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2, sort_keys=True, default=str))
