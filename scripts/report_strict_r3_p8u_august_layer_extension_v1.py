#!/usr/bin/env python3
"""Report target-free Router/Base/Under metrics for the August-27 extension.

The utility is evaluation-only: score identities already exist.  It reads
target-free score receipts, joins policy outcomes afterwards, and explicitly
excludes decisions at or after 2026-08-28T00:00:00Z.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ROUTER = ROOT / "data_perp/artifacts/strict_r3_p8u_august01_27_router_score_overlay_20260828_v1/target_free_scores/month=2026-08.parquet"
BASE = ROOT / "data_perp/artifacts/strict_r3_p8u_august01_27_f72_base_scores_20260828_v1/scheme=tail_linear_125/target_free_scores/month=2026-08.parquet"
UNDER = ROOT / "data_perp/artifacts/strict_r3_p8u_august01_27_under_f120_targetfree_scores_20260828_v1/target_free_scores/xendcg_selected_under_bps100/month=2026-08.parquet"
POLICY = ROOT / "data_perp/artifacts/strict_r3_p8u_router_policy_label_successor_fullprehistory_aug27_20260828_v1/canonical_reconciled_policy_labels.parquet"
CUTOFF = pd.Timestamp("2026-08-28T00:00:00Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def _write_once(path: Path, text: str) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)


def _top(frame: pd.DataFrame, score: str, fraction: float) -> pd.DataFrame:
    work = frame.sort_values(["__decision_ts__", score, "candidate_id"], ascending=[True, False, True], kind="stable").copy()
    work["ordinal"] = work.groupby("__decision_ts__", sort=False).cumcount()
    count = work.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size")
    return work.loc[work.ordinal.lt(np.maximum(1, np.ceil(count * fraction)).astype(int))].copy()


def _tail(frame: pd.DataFrame, score: str, label: str) -> pd.DataFrame:
    rows = []
    for fraction in (.01, .02, .05, .10, .15):
        selected = _top(frame, score, fraction)
        ts = selected.groupby("__decision_ts__", sort=True).policy_net_bps.agg(
            timestamp_net_ev_bps="mean",
            timestamp_positive_hit_rate=lambda values: float(values.gt(0).mean()),
            timestamp_gt50_hit_rate=lambda values: float(values.gt(50).mean()),
        )
        rows.append({
            "label": label, "fraction": fraction, "timestamps": int(len(ts)), "selected_rows": int(len(selected)),
            "timestamp_net_ev_bps": float(ts.timestamp_net_ev_bps.mean()),
            "timestamp_positive_hit_rate": float(ts.timestamp_positive_hit_rate.mean()),
            "timestamp_gt50_hit_rate": float(ts.timestamp_gt50_hit_rate.mean()),
            "timestamp_worst_net_ev_bps": float(ts.timestamp_net_ev_bps.min()),
        })
    return pd.DataFrame(rows)


def _router_recall(frame: pd.DataFrame) -> pd.DataFrame:
    selected = _top(frame, "router_primary_rank", .50)
    rows = []
    for hurdle in (50.0, 100.0, 150.0, 200.0):
        all_hits = frame.assign(hit=frame.policy_net_bps.gt(hurdle)).groupby("__decision_ts__", sort=True).hit.sum()
        chosen = selected.assign(hit=selected.policy_net_bps.gt(hurdle)).groupby("__decision_ts__", sort=True).hit.sum()
        report = pd.DataFrame({"oracle_hits": all_hits, "selected_hits": chosen}).fillna(0.0)
        report = report.loc[report.oracle_hits.gt(0)].copy()
        rows.append({
            "policy_net_hurdle_bps": hurdle, "timestamps_with_oracle": int(len(report)),
            "top50_recall": float((report.selected_hits / report.oracle_hits).mean()),
            "selected_hits": int(report.selected_hits.sum()), "oracle_hits": int(report.oracle_hits.sum()),
        })
    return pd.DataFrame(rows)


def _rank(frame: pd.DataFrame, source: str, target: str) -> pd.DataFrame:
    work = frame.loc[:, ["candidate_id", "__decision_ts__", source]].copy()
    work["__row__"] = np.arange(len(work), dtype=np.int64)
    ordered = work.sort_values(["__decision_ts__", source, "candidate_id"], ascending=[True, False, True], kind="stable")
    ordinal = ordered.groupby("__decision_ts__", sort=False).cumcount().to_numpy(float) + 1.0
    count = ordered.groupby("__decision_ts__", sort=False).candidate_id.transform("size").to_numpy(float)
    ranks = np.empty(len(ordered), dtype=np.float32)
    ranks[ordered.__row__.to_numpy(np.int64)] = 1.0 - (ordinal - .5) / count
    frame[target] = ranks
    return frame


def _cmi(frame: pd.DataFrame) -> float:
    work = frame.loc[:, ["base_rank_ts", "meta_rank_ts", "policy_net_bps"]].copy()
    work["base_bin"] = np.minimum(9, np.floor(work.base_rank_ts.to_numpy(float) * 10.0).astype(int))
    work["meta_bin"] = np.minimum(9, np.floor(work.meta_rank_ts.to_numpy(float) * 10.0).astype(int))
    work["target_bin"] = np.digitize(work.policy_net_bps.to_numpy(float), [-200, -50, 0, 50, 100, 200, 400], right=True)
    total, value = len(work), 0.0
    for _, cell in work.groupby("base_bin", sort=True):
        if len(cell) < 20:
            continue
        joint = cell.groupby(["meta_bin", "target_bin"], sort=False).size() / len(cell)
        pm = cell.groupby("meta_bin", sort=False).size() / len(cell)
        py = cell.groupby("target_bin", sort=False).size() / len(cell)
        inner = sum(float(prob) * np.log(max(float(prob) / max(float(pm.loc[m]) * float(py.loc[y]), 1e-12), 1e-12)) for (m, y), prob in joint.items())
        value += len(cell) / total * inner
    return float(value)


def _markdown(router: pd.DataFrame, tail: pd.DataFrame, cmi: float) -> str:
    def table(frame: pd.DataFrame) -> list[str]:
        lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
        for row in frame.itertuples(index=False, name=None):
            cells = [f"{float(value):.4f}" if isinstance(value, (float, np.floating)) else str(value) for value in row]
            lines.append("| " + " | ".join(cells) + " |")
        return lines
    return "\n".join([
        "# August 1–27 target-free layer extension", "",
        "Scores were persisted before policy labels were joined. Metrics are timestamp-local, then averaged over the 648 decision timestamps. Rich policy net includes the 100-bps cost once. This is retrospective reconciliation, not promotion evidence.", "",
        "## Router50 recall", "", *table(router), "",
        "## Base and Current-family timestamp tails", "", *table(tail), "",
        f"Under F120 conditional MI given F72 Base: **{cmi:.6f} nats**.", "",
    ])


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    out.mkdir(parents=True)
    policy = pd.read_parquet(POLICY, columns=["candidate_id", "policy_path_valid", "policy_net_bps"])
    policy["candidate_id"] = policy.candidate_id.astype(str)
    policy["policy_net_bps"] = pd.to_numeric(policy.policy_net_bps, errors="coerce")
    router = pd.read_parquet(ROUTER, columns=["candidate_id", "__decision_ts__", "router_primary_rank"])
    base = pd.read_parquet(BASE, columns=["candidate_id", "__decision_ts__", "side_name", "base_score", "base_rank_ts"])
    under = pd.read_parquet(UNDER, columns=["candidate_id", "__decision_ts__", "meta_rank_ts"])
    for frame in (router, base, under):
        frame["candidate_id"] = frame.candidate_id.astype(str)
        frame["__decision_ts__"] = pd.to_datetime(frame.__decision_ts__, utc=True, errors="raise")
        frame.drop(frame.index[frame.__decision_ts__.ge(CUTOFF)], inplace=True)
    router = router.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    router = router.loc[router.policy_path_valid.fillna(False).astype(bool) & np.isfinite(router.policy_net_bps)].copy()
    base = base.merge(under, on=["candidate_id", "__decision_ts__"], how="inner", validate="one_to_one").merge(policy, on="candidate_id", how="left", validate="one_to_one")
    base = base.loc[base.policy_path_valid.fillna(False).astype(bool) & np.isfinite(base.policy_net_bps)].copy()
    base["current_blend"] = .75 * pd.to_numeric(base.base_rank_ts) + .25 * pd.to_numeric(base.meta_rank_ts)
    base = _rank(base, "current_blend", "current_rank_ts")
    router_report = _router_recall(router)
    tails = pd.concat([_tail(base, "base_score", "F72 Base"), _tail(base, "current_rank_ts", "Current 75/25 Base/Under")], ignore_index=True)
    cmi = _cmi(base)
    router_report.to_parquet(out / "router_recall.parquet", index=False, compression="zstd")
    tails.to_parquet(out / "base_current_timestamp_tails.parquet", index=False, compression="zstd")
    _write_once(out / "AUGUST_LAYER_EXTENSION_RECEIPT.md", _markdown(router_report, tails, cmi))
    _write_once(out / "run_manifest.json", json.dumps({
        "schema": "strict_r3_p8u_august_layer_extension_v1", "scope": "offline target-free score evaluation only",
        "decision_scope": "timestamp < 2026-08-28T00:00:00Z", "router_rows": int(len(router)), "base_under_rows": int(len(base)),
        "sources": {str(path.relative_to(ROOT)): _sha256(path) for path in (ROUTER, BASE, UNDER, POLICY)},
        "cost": "policy net bps carries the 100-bps round-trip cost once",
    }, indent=2, sort_keys=True) + "\n")
    print(out)


if __name__ == "__main__":
    main()
