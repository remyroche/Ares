#!/usr/bin/env python3
"""Report strict target-free T6/T9 scores on the new B/E/T upstream.

This reporter is deliberately outcome-free until every score identity has
already been assembled and checked.  It then joins canonical rich-policy
outcomes solely to calculate diagnostics.  The primary metrics are fixed-K
within-timestamp realised net EV and the distribution of weekly/monthly
top-two-per-timestamp EV; global percentile tails remain a secondary view.

Research only.  It never writes a model, mapper, admission, portfolio, or
live-execution artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SCHEMA = "strict_r3_meta_newbase_metrics_v1"
IDENTITY = ["candidate_id", "__decision_ts__", "side_name"]
PROHIBITED = {
    "policy_path_valid", "policy_net_bps", "policy_gross_bps",
    "policy_label_available_ts", "semantic_policy_net_bps",
    "semantic_path_valid", "semantic_label_available_ts",
}
# Top-3 is the primary selection/HPO boundary; 1/2/5/10 are the requested
# published diagnostics.
K_VALUES = (1, 2, 3, 5, 10)
TAIL_VALUES = (.01, .02, .05, .10)


def _exclusive_json(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    sources = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    for source in sources:
        digest.update(str(source).encode())
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _months(raw: str) -> tuple[pd.Timestamp, ...]:
    result = tuple(pd.Timestamp(f"{value.strip()}-01", tz="UTC") for value in raw.split(",") if value.strip())
    if not result:
        raise ValueError("at least one YYYY-MM month is required")
    return result


def _rank_desc(frame: pd.DataFrame, field: str) -> pd.Series:
    work = frame.loc[:, ["__decision_ts__", "candidate_id", field]].copy()
    work["__row__"] = np.arange(len(work), dtype=np.int64)
    work["__score__"] = pd.to_numeric(work[field], errors="coerce").fillna(-np.inf)
    work = work.sort_values(["__decision_ts__", "__score__", "candidate_id"], ascending=[True, False, True], kind="stable")
    order = work.groupby("__decision_ts__", sort=False).cumcount().to_numpy(float) + 1.0
    count = work.groupby("__decision_ts__", sort=False).candidate_id.transform("size").to_numpy(float)
    result = np.empty(len(frame), dtype=np.float32)
    result[work["__row__"].to_numpy(np.int64)] = 1.0 - (order - .5) / count
    return pd.Series(result, index=frame.index, name=f"{field}__rank")


def _read_panel(feature_root: Path, score_root: Path, month: pd.Timestamp) -> pd.DataFrame:
    base_path = feature_root / f"month={month:%Y-%m}" / "scores_features.parquet"
    t6_path = score_root / "target_free_scores" / "T6_rank_error_ordinal" / f"month={month:%Y-%m}.parquet"
    t9_path = score_root / "target_free_scores" / "T9_exit5_ordinal" / f"month={month:%Y-%m}.parquet"
    for source in (base_path, t6_path, t9_path):
        if not source.exists():
            raise FileNotFoundError(source)
    base = pd.read_parquet(base_path)
    leaked = sorted(PROHIBITED.intersection(base.columns))
    if leaked:
        raise AssertionError(f"{month:%Y-%m}: target-free base source leaks {leaked}")
    required = set(IDENTITY) | {"enhanced_base_bps"}
    missing = sorted(required - set(base.columns))
    if missing:
        raise KeyError(f"{month:%Y-%m}: base source misses {missing}")
    base = base.loc[:, [*IDENTITY, "enhanced_base_bps"]].copy()
    base["__decision_ts__"] = pd.to_datetime(base["__decision_ts__"], utc=True, errors="raise")
    base["base_rank"] = _rank_desc(base, "enhanced_base_bps")
    # The route is recomputed from the current source rather than inherited
    # from a legacy persisted flag: exact ceil(30%) is the production
    # coordinate supplied to the meta heads.
    ranked = base.sort_values(["__decision_ts__", "enhanced_base_bps", "candidate_id"], ascending=[True, False, True], kind="stable").copy()
    ranked["__ordinal__"] = ranked.groupby("__decision_ts__", sort=False).cumcount() + 1
    total = ranked.groupby("__decision_ts__", sort=False).candidate_id.transform("size").to_numpy(float)
    routed = np.zeros(len(base), dtype=bool)
    routed[ranked.index.to_numpy(np.int64)] = ranked["__ordinal__"].to_numpy() <= np.ceil(total * .30)
    base["routed"] = routed
    # T6/T9 are intentionally scored only after this same deterministic
    # top-30 upstream route.  Merge their receipts against this routed
    # population, not against the wider upstream source panel.
    base = base.loc[base.routed].copy()
    heads: list[pd.DataFrame] = []
    for name, source in (("t6", t6_path), ("t9", t9_path)):
        frame = pd.read_parquet(source)
        leaked = sorted(PROHIBITED.intersection(frame.columns))
        if leaked:
            raise AssertionError(f"{month:%Y-%m}/{name}: target-free score source leaks {leaked}")
        rank_fields = [field for field in frame if field.startswith("head__") and field.endswith("__rank")]
        if len(rank_fields) != 1:
            raise AssertionError(f"{month:%Y-%m}/{name}: expected one frozen physical-head rank, found {rank_fields}")
        heads.append(frame.loc[:, [*IDENTITY, rank_fields[0]]].rename(columns={rank_fields[0]: f"{name}_rank"}))
    output = base
    for head in heads:
        output = output.merge(head, on=IDENTITY, how="inner", validate="one_to_one")
    if len(output) != len(base) or output.duplicated(IDENTITY).any():
        raise AssertionError(f"{month:%Y-%m}: base/T6/T9 identity mismatch")
    output["s11_rank"] = .75 * output["base_rank"] + .20 * output["t6_rank"] + .05 * output["t9_rank"]
    return output


def _score_order(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    result = frame.loc[:, ["candidate_id", "__decision_ts__", "policy_net_bps", field]].copy()
    result["__score__"] = pd.to_numeric(result[field], errors="coerce").fillna(-np.inf)
    result = result.sort_values(["__decision_ts__", "__score__", "candidate_id"], ascending=[True, False, True], kind="stable")
    result["rank"] = result.groupby("__decision_ts__", sort=False).cumcount() + 1
    result["query_n"] = result.groupby("__decision_ts__", sort=False).candidate_id.transform("size")
    return result


def _timestamp_rows(frame: pd.DataFrame, score: str) -> tuple[list[dict[str, object]], pd.DataFrame]:
    ordered = _score_order(frame, score)
    rows: list[dict[str, object]] = []
    top2 = ordered.loc[ordered["rank"].le(2)].groupby("__decision_ts__", sort=False)["policy_net_bps"].mean().rename("top2_timestamp_ev").reset_index()
    top2["week"] = top2["__decision_ts__"].dt.to_period("W-SUN").astype(str)
    top2["month"] = top2["__decision_ts__"].dt.strftime("%Y-%m")
    for k in K_VALUES:
        selected = ordered.loc[ordered["rank"].le(k)]
        grouped = selected.groupby("__decision_ts__", sort=False)["policy_net_bps"].mean()
        rows.append({
            "metric_scope": "timestamp_fixed_k", "score": score, "top_k": k,
            "timestamps": int(grouped.size), "trades": int(selected.shape[0]),
            "net_ev_bps_per_trade": float(grouped.mean()),
            "net_sum_bps": float(selected["policy_net_bps"].sum()),
        })
    return rows, top2


def _global_rows(frame: pd.DataFrame, score: str) -> list[dict[str, object]]:
    ordered = frame.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable")
    rows: list[dict[str, object]] = []
    for tail in TAIL_VALUES:
        count = max(1, int(np.ceil(len(ordered) * tail)))
        selected = ordered.iloc[:count]
        rows.append({
            "metric_scope": "global_percentile_tail", "score": score, "top_fraction": tail,
            "trades": int(len(selected)), "net_ev_bps_per_trade": float(selected.policy_net_bps.mean()),
            "net_sum_bps": float(selected.policy_net_bps.sum()),
        })
    return rows


def _stability_rows(top2: pd.DataFrame, score: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    stability: list[dict[str, object]] = []
    periods: list[dict[str, object]] = []
    for resolution in ("week", "month"):
        values = top2.groupby(resolution, sort=True).top2_timestamp_ev.mean()
        for token, value in values.items():
            periods.append({"score": score, "resolution": resolution, "period": str(token), "top2_timestamp_ev_bps": float(value)})
        for label, q in (("q01", .01), ("q05", .05), ("q25", .25), ("q50", .50)):
            stability.append({
                "score": score, "resolution": resolution, "quantile": label,
                "periods": int(values.size), "top2_timestamp_ev_bps": float(values.quantile(q)),
            })
    return stability, periods


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--months", required=True, help="comma-separated YYYY-MM held months")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    months = _months(args.months)
    policy = pd.read_parquet(args.policy_path, columns=["candidate_id", "policy_path_valid", "policy_net_bps", "policy_label_available_ts"])
    policy["policy_path_valid"] = policy["policy_path_valid"].fillna(False).astype(bool)
    policy["policy_net_bps"] = pd.to_numeric(policy["policy_net_bps"], errors="coerce")
    policy["policy_label_available_ts"] = pd.to_datetime(policy["policy_label_available_ts"], utc=True, errors="coerce")
    if policy.candidate_id.duplicated().any():
        raise AssertionError("canonical policy labels have duplicate candidate IDs")
    panels = [_read_panel(args.feature_root, args.score_root, month) for month in months]
    target_free = pd.concat(panels, ignore_index=True)
    if target_free.duplicated(IDENTITY).any():
        raise AssertionError("duplicate target-free score identity across months")
    args.out.mkdir(parents=True)
    target_free.to_parquet(args.out / "target_free_scores.parquet", index=False, compression="zstd")
    joined = target_free.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    valid = joined.policy_path_valid & np.isfinite(joined.policy_net_bps)
    measured = joined.loc[valid].copy()
    scores = ("base_rank", "t6_rank", "t9_rank", "s11_rank")
    global_rows: list[dict[str, object]] = []
    timestamp_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    period_rows: list[dict[str, object]] = []
    for score in scores:
        global_rows.extend(_global_rows(measured, score))
        rows, top2 = _timestamp_rows(measured, score)
        timestamp_rows.extend(rows)
        stability, periods = _stability_rows(top2, score)
        stability_rows.extend(stability)
        period_rows.extend(periods)
    pd.DataFrame(global_rows).to_parquet(args.out / "global_tail_metrics.parquet", index=False, compression="zstd")
    pd.DataFrame(timestamp_rows).to_parquet(args.out / "timestamp_topk_metrics.parquet", index=False, compression="zstd")
    pd.DataFrame(stability_rows).to_parquet(args.out / "top2_stability_quantiles.parquet", index=False, compression="zstd")
    pd.DataFrame(period_rows).to_parquet(args.out / "top2_period_metrics.parquet", index=False, compression="zstd")
    correctness = {
        "schema": SCHEMA,
        "target_free_rows": int(len(target_free)),
        "measured_valid_rows": int(len(measured)),
        "target_free_score_columns": list(target_free.columns),
        "prohibited_score_columns": sorted(PROHIBITED),
        "no_prohibited_score_columns": not bool(PROHIBITED.intersection(target_free.columns)),
        "unique_identity": not bool(target_free.duplicated(IDENTITY).any()),
        "policy_join_only_after_target_free_receipt": True,
        "months": [f"{month:%Y-%m}" for month in months],
    }
    _exclusive_json(args.out / "correctness_report.json", correctness)
    _exclusive_json(args.out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline research-only T6/T9 metric reporting; no model, mapper, admission, portfolio, or live mutation",
        "feature_root": str(args.feature_root), "score_root": str(args.score_root), "policy_path": str(args.policy_path),
        "source_sha256": {"feature_root": _sha(args.feature_root), "score_root": _sha(args.score_root), "policy_path": _sha(args.policy_path)},
        "months": [f"{month:%Y-%m}" for month in months],
        "score_definitions": {"base_rank": "new B/E/T timestamp-local upstream rank", "t6_rank": "frozen cap80 rank-error correction", "t9_rank": "frozen cap120 exit-quality correction", "s11_rank": "0.75 base + 0.20 T6 + 0.05 T9"},
        "primary_metric": "timestamp-local fixed top-3/2/1 realised policy net, with weekly/monthly top2 distribution diagnostics",
        "causality": "target-free scores sealed before canonical policy outcomes are joined for metrics",
    })


if __name__ == "__main__":
    main()
