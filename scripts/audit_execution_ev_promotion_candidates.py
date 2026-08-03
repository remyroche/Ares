#!/usr/bin/env python3
"""Create one comparable promotion audit for execution-EV score candidates.

Each candidate is evaluated independently on its complete strict-OOF panel.
Ranking is one pooled global top-k across sides; month, fold and week views are
diagnostics and never quotas or timestamp-local selection rules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


TARGET = "execution_net_ev_12h"
SIDE = "side_name"
FOLD = "oof_fold"
DECISION = "execution_decision_utc"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _global_topk(frame: pd.DataFrame, score: str, mask: np.ndarray, fraction: float) -> pd.DataFrame:
    eligible = frame.loc[np.asarray(mask, dtype=bool) & frame[score].notna() & frame[TARGET].notna()]
    if eligible.empty:
        return eligible
    rows = max(1, int(np.ceil(float(fraction) * len(eligible))))
    return eligible.nlargest(rows, score, keep="first")


def _metrics(frame: pd.DataFrame, selected: pd.DataFrame, score: str, scope: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "scope": scope,
        "candidate_rows": int(len(frame)),
        "selected_rows": int(len(selected)),
        "score_coverage": float(frame[score].notna().mean()) if len(frame) else float("nan"),
        "selected_mean_predicted_ev_bps": float(selected[score].mean() * 1e4) if len(selected) else float("nan"),
        "selected_mean_realized_ev_bps": float(selected[TARGET].mean() * 1e4) if len(selected) else float("nan"),
        "selected_sum_realized_ev": float(selected[TARGET].sum()) if len(selected) else 0.0,
        "selected_positive_rate": float((selected[TARGET] > 0).mean()) if len(selected) else float("nan"),
    }
    paired = frame.loc[frame[score].notna() & frame[TARGET].notna(), [score, TARGET]]
    result["spearman"] = float(paired[score].corr(paired[TARGET], method="spearman")) if len(paired) > 1 else float("nan")
    binary = (paired[TARGET] > 0).astype(np.int8)
    result["positive_ev_auc"] = (
        float(roc_auc_score(binary, paired[score]))
        if len(paired) > 1 and binary.nunique() == 2
        else float("nan")
    )
    for side in ("long", "short"):
        rows = selected.loc[selected[SIDE].astype(str).str.lower().eq(side)]
        result[f"selected_{side}_rows"] = int(len(rows))
        result[f"selected_{side}_mean_realized_ev_bps"] = float(rows[TARGET].mean() * 1e4) if len(rows) else float("nan")
    return result


def _slices(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    decision = pd.to_datetime(frame[DECISION], utc=True, errors="raise")
    fold = pd.to_numeric(frame[FOLD], errors="coerce")
    latest_fold = int(fold.max())
    latest_month = decision.dt.to_period("M").max()
    latest_end = decision.max() + pd.Timedelta(nanoseconds=1)
    latest_start = latest_end - pd.Timedelta(days=7)
    return {
        "all_oof": np.ones(len(frame), dtype=bool),
        "latest_fold": fold.eq(latest_fold).to_numpy(),
        "latest_month": decision.dt.to_period("M").eq(latest_month).to_numpy(),
        "latest_7d": decision.ge(latest_start).to_numpy() & decision.lt(latest_end).to_numpy(),
    }


def _deciles(frame: pd.DataFrame, score: str, candidate: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    work = frame.loc[frame[score].notna() & frame[TARGET].notna()].copy()
    for (fold, side), group in work.groupby([FOLD, SIDE], sort=True):
        ranked = group[score].rank(method="first", pct=True)
        decile = np.minimum(9, np.floor(ranked.to_numpy(float) * 10.0).astype(int))
        for bucket in range(10):
            selected = group.loc[decile == bucket]
            if selected.empty:
                continue
            rows.append({
                "candidate": candidate,
                "fold": int(fold),
                "side": str(side),
                "score_decile": int(bucket),
                "rows": int(len(selected)),
                "mean_predicted_ev_bps": float(selected[score].mean() * 1e4),
                "mean_realized_ev_bps": float(selected[TARGET].mean() * 1e4),
                "positive_rate": float((selected[TARGET] > 0).mean()),
            })
    return rows


def audit_candidate(
    name: str,
    path: Path,
    score: str,
    *,
    top_fraction: float,
    min_latest_month_rows: int,
    min_latest_7d_rows: int,
    min_side_rows: int,
    min_fold_coverage: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    frame = pd.read_parquet(path)
    required = {TARGET, SIDE, FOLD, DECISION, score}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name}: missing required columns: {', '.join(missing)}")
    frame = frame.copy()
    frame[DECISION] = pd.to_datetime(frame[DECISION], utc=True, errors="raise")
    frame[SIDE] = frame[SIDE].astype(str).str.lower()
    if set(frame[SIDE]) - {"long", "short"}:
        raise ValueError(f"{name}: side must be long/short")
    frame[score] = pd.to_numeric(frame[score], errors="coerce")
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    slices = _slices(frame)
    slice_rows: list[dict[str, Any]] = []
    selected_by_scope: dict[str, pd.DataFrame] = {}
    for scope, mask in slices.items():
        population = frame.loc[mask]
        selected = _global_topk(frame, score, mask, top_fraction)
        selected_by_scope[scope] = selected
        slice_rows.append({"candidate": name, **_metrics(population, selected, score, scope)})
    lookup = {row["scope"]: row for row in slice_rows}
    fold_coverage = (
        frame.groupby(FOLD, sort=True)[score].apply(lambda values: float(values.notna().mean()))
    )
    all_selected = selected_by_scope["all_oof"]
    admitted_recent: dict[str, dict[str, Any]] = {}
    for scope in ("latest_fold", "latest_month", "latest_7d"):
        scope_mask = pd.Series(slices[scope], index=frame.index)
        rows = all_selected.loc[scope_mask.reindex(all_selected.index).fillna(False).to_numpy(bool)]
        admitted_recent[scope] = {
            "rows": int(len(rows)),
            "mean_realized_ev_bps": float(rows[TARGET].mean() * 1e4) if len(rows) else float("nan"),
        }
    latest_month = admitted_recent["latest_month"]
    latest_7d = admitted_recent["latest_7d"]
    gates = {
        "aggregate_positive": lookup["all_oof"]["selected_mean_realized_ev_bps"] > 0,
        "latest_fold_positive": admitted_recent["latest_fold"]["mean_realized_ev_bps"] > 0,
        "latest_month_positive": latest_month["mean_realized_ev_bps"] > 0,
        "latest_7d_positive": latest_7d["mean_realized_ev_bps"] > 0,
        "latest_month_coverage": latest_month["rows"] >= int(min_latest_month_rows),
        "latest_7d_coverage": latest_7d["rows"] >= int(min_latest_7d_rows),
        "both_sides_covered": all(
            lookup["all_oof"][f"selected_{side}_rows"] >= int(min_side_rows)
            for side in ("long", "short")
        ),
        "both_sides_positive": all(
            lookup["all_oof"][f"selected_{side}_mean_realized_ev_bps"] > 0
            for side in ("long", "short")
        ),
        "fold_score_coverage": bool((fold_coverage >= float(min_fold_coverage)).all()),
    }
    promotion = {
        "candidate": name,
        "source_path": str(path),
        "score_column": score,
        **{f"gate__{key}": bool(value) for key, value in gates.items()},
        "promotion_eligible": bool(all(gates.values())),
        "aggregate_ev_bps": lookup["all_oof"]["selected_mean_realized_ev_bps"],
        "latest_fold_ev_bps": admitted_recent["latest_fold"]["mean_realized_ev_bps"],
        "latest_month_ev_bps": latest_month["mean_realized_ev_bps"],
        "latest_7d_ev_bps": latest_7d["mean_realized_ev_bps"],
        "aggregate_selected_rows": int(len(all_selected)),
        "latest_fold_selected_rows": admitted_recent["latest_fold"]["rows"],
        "latest_month_selected_rows": latest_month["rows"],
        "latest_7d_selected_rows": latest_7d["rows"],
        "latest_fold_diagnostic_topk_ev_bps": lookup["latest_fold"]["selected_mean_realized_ev_bps"],
        "latest_month_diagnostic_topk_ev_bps": lookup["latest_month"]["selected_mean_realized_ev_bps"],
        "latest_7d_diagnostic_topk_ev_bps": lookup["latest_7d"]["selected_mean_realized_ev_bps"],
        "long_ev_bps": lookup["all_oof"]["selected_long_mean_realized_ev_bps"],
        "short_ev_bps": lookup["all_oof"]["selected_short_mean_realized_ev_bps"],
        "minimum_fold_score_coverage": float(fold_coverage.min()),
        "decision": "eligible_for_policy_replay" if all(gates.values()) else "reject_before_policy_replay",
    }
    composition: list[dict[str, Any]] = []
    for dimension, values in (
        ("fold", all_selected[FOLD].astype(str)),
        ("month", all_selected[DECISION].dt.to_period("M").astype(str)),
        ("side", all_selected[SIDE].astype(str)),
    ):
        for value, count in values.value_counts(dropna=False, sort=False).items():
            composition.append({
                "candidate": name,
                "scope": "all_oof_global_topk",
                "dimension": dimension,
                "value": str(value),
                "selected_rows": int(count),
                "selected_share": float(count / len(all_selected)) if len(all_selected) else float("nan"),
            })
    return promotion, slice_rows, composition, _deciles(frame, score, name)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        nargs=3,
        action="append",
        metavar=("NAME", "PARQUET", "SCORE_COLUMN"),
        required=True,
        help="Repeat for each independently materialized strict-OOF score.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--min-latest-month-rows", type=int, default=100)
    parser.add_argument("--min-latest-7d-rows", type=int, default=20)
    parser.add_argument("--min-side-rows", type=int, default=50)
    parser.add_argument("--min-fold-coverage", type=float, default=0.99)
    return parser


def run(args: argparse.Namespace) -> pd.DataFrame:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    promotions: list[dict[str, Any]] = []
    slices: list[dict[str, Any]] = []
    composition: list[dict[str, Any]] = []
    deciles: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for name, raw_path, score in args.candidate:
        path = Path(raw_path)
        promotion, local_slices, local_composition, local_deciles = audit_candidate(
            str(name), path, str(score),
            top_fraction=float(args.top_fraction),
            min_latest_month_rows=int(args.min_latest_month_rows),
            min_latest_7d_rows=int(args.min_latest_7d_rows),
            min_side_rows=int(args.min_side_rows),
            min_fold_coverage=float(args.min_fold_coverage),
        )
        promotions.append(promotion)
        slices.extend(local_slices)
        composition.extend(local_composition)
        deciles.extend(local_deciles)
        sources.append({"name": name, "path": str(path), "score": score, "sha256": _sha256(path)})
    table = pd.DataFrame(promotions).sort_values(
        ["promotion_eligible", "latest_7d_ev_bps", "aggregate_ev_bps"],
        ascending=[False, False, False],
        kind="stable",
    )
    table.to_csv(args.output_dir / "promotion_table.csv", index=False)
    pd.DataFrame(slices).to_csv(args.output_dir / "slice_metrics.csv", index=False)
    pd.DataFrame(composition).to_csv(args.output_dir / "selected_composition.csv", index=False)
    pd.DataFrame(deciles).to_csv(args.output_dir / "calibration_deciles.csv", index=False)
    manifest = {
        "schema": "execution_ev_common_promotion_audit_v1",
        "ranking": "one pooled global top-k across sides; slice rankings are diagnostics only",
        "selection_quotas": "none; coverage thresholds are promotion gates, not ranking quotas",
        "target": TARGET,
        "top_fraction": float(args.top_fraction),
        "thresholds": {
            "min_latest_month_rows": int(args.min_latest_month_rows),
            "min_latest_7d_rows": int(args.min_latest_7d_rows),
            "min_side_rows": int(args.min_side_rows),
            "min_fold_score_coverage": float(args.min_fold_coverage),
        },
        "sources": sources,
        "promotion_eligible": table.loc[table["promotion_eligible"], "candidate"].tolist(),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(_safe(manifest), indent=2, sort_keys=True) + "\n")
    return table


if __name__ == "__main__":
    print(run(_parser().parse_args()).to_string(index=False))
