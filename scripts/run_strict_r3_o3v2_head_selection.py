#!/usr/bin/env python3
"""Quality/complementarity selection for frozen O3-v2 specialist outputs.

The input is a collection of already strict-OOF, target-free specialist score
receipts.  This stage never refits a head.  It selects a compact ensemble on a
declared development block and writes aggregate target-free receipts suitable
for the unchanged MC1 adapter.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


SCHEMA = "strict_r3_o3v2_head_selection_v1"
TAILS = (.01, .02, .05)
SLOTS = ("cap100_ordinary", "cap80_ordinary", "cap120_equal_month", "cap40_equal_month", "cap60_equal_month")


def _months(raw: str) -> tuple[str, ...]:
    return tuple(value.strip() for value in raw.split(",") if value.strip())


def _write_exclusive(path: Path, value: object) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def _residual(value: pd.Series, base: pd.Series) -> pd.Series:
    bucket = np.minimum(9, np.maximum(0, np.floor(pd.to_numeric(base, errors="coerce").fillna(.5) * 10))).astype(int)
    return pd.to_numeric(value, errors="coerce") - pd.to_numeric(value, errors="coerce").groupby(bucket, sort=False).transform("mean")


def _utility(frame: pd.DataFrame, rank: np.ndarray) -> dict[str, float]:
    outcome = pd.to_numeric(frame["policy_net_bps"], errors="coerce").to_numpy(float)
    finite = np.isfinite(rank) & np.isfinite(outcome)
    result = {"rank_ic": float(spearmanr(rank[finite], outcome[finite]).statistic) if finite.sum() >= 12 else np.nan}
    for tail, token in ((.01, "top1"), (.02, "top2"), (.05, "top5")):
        cut = np.quantile(rank[finite], 1. - tail, method="higher")
        result[token] = float(np.mean(outcome[finite & (rank >= cut)]))
    result["utility"] = float(.4 * result["top1"] + .35 * result["top2"] + .25 * result["top5"] + 25. * result["rank_ic"])
    return result


def _head_stats(panel: pd.DataFrame, heads: list[str]) -> pd.DataFrame:
    policy_rank = panel.groupby("__decision_ts__", sort=False)["policy_net_bps"].rank(pct=True, method="average")
    base = pd.to_numeric(panel["f1_base_rank_ts"], errors="coerce")
    rows = []
    for head in heads:
        rank = pd.to_numeric(panel[head], errors="coerce")
        residual_rank = _residual(rank, base)
        residual_outcome = _residual(panel["policy_net_bps"], base)
        high = rank.ge(.80)
        bad = policy_rank.le(.40)
        stats = _utility(panel, rank.to_numpy(float))
        rows.append({
            "head": head, "standalone_top1": stats["top1"], "standalone_top2": stats["top2"], "standalone_top5": stats["top5"],
            "policy_rank_ic": stats["rank_ic"], "utility": stats["utility"],
            "corr_to_base": float(rank.corr(base, method="spearman")),
            "conditional_residual_ic": float(residual_rank.corr(residual_outcome, method="spearman")),
            "double_fault_rate": float((high & bad).mean()),
        })
    return pd.DataFrame(rows).sort_values("utility", ascending=False, kind="stable").reset_index(drop=True)


def _ensemble_metrics(panel: pd.DataFrame, heads: list[str]) -> tuple[float, pd.DataFrame]:
    rows = []
    for month, part in panel.groupby(panel["__decision_ts__"].dt.strftime("%Y-%m"), sort=True):
        consensus = np.nanmedian(part.loc[:, heads].to_numpy(float), axis=1)
        mix = .75 * pd.to_numeric(part["f1_base_rank_ts"], errors="coerce").to_numpy(float) + .25 * consensus
        stat = _utility(part, mix)
        rows.append({"month": month, **stat})
    result = pd.DataFrame(rows)
    selection_score = float(result["utility"].mean() - .25 * result["utility"].std(ddof=0) - max(0., -float(result["utility"].min())))
    return selection_score, result


def run(*, score_root: Path, target_name: str, policy_path: Path, out: Path, months: tuple[str, ...]) -> None:
    if out.exists():
        raise FileExistsError(out)
    pieces = []
    for month in months:
        source = score_root / "target_free_scores" / target_name / f"month={month}" / "scores.parquet"
        if not source.exists():
            raise FileNotFoundError(source)
        part = pd.read_parquet(source)
        prohibited = [field for field in part if field.startswith(("policy_", "semantic_"))]
        if prohibited:
            raise AssertionError(f"{source}: outcome field in held score receipt: {prohibited}")
        pieces.append(part)
    raw = pd.concat(pieces, ignore_index=True)
    head_columns = [field for field in raw if field.endswith("__rank") and field != "specialist_ensemble_rank"]
    if not head_columns:
        raise AssertionError("no specialist head ranks")
    policy = pd.read_parquet(policy_path, columns=("candidate_id", "policy_path_valid", "policy_net_bps"))
    panel = raw.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    panel = panel.loc[panel["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(pd.to_numeric(panel["policy_net_bps"], errors="coerce"))].copy()
    if panel.empty:
        raise AssertionError("no valid policy outcomes for development head selection")
    panel["__decision_ts__"] = pd.to_datetime(panel["__decision_ts__"], utc=True, errors="raise")
    stats = _head_stats(panel, head_columns)
    chosen: list[str] = []
    trace: list[dict[str, object]] = []
    current_score = -np.inf
    remaining = head_columns.copy()
    while remaining and len(chosen) < 5:
        trials = []
        for head in remaining:
            score, metrics = _ensemble_metrics(panel, [*chosen, head])
            # Correlation/double-fault penalties prevent a high-scoring clone
            # from joining a compact correction ensemble.
            if chosen:
                conditional_corr = np.nanmax([
                    abs(float(_residual(panel[head], panel["f1_base_rank_ts"]).corr(_residual(panel[other], panel["f1_base_rank_ts"]), method="spearman")))
                    for other in chosen
                ])
                score -= 10.0 * max(0., conditional_corr - .85)
            trials.append((head, score, metrics))
        head, score, metrics = max(trials, key=lambda item: (item[1], item[0]))
        incremental = score - current_score if np.isfinite(current_score) else np.inf
        accepted = bool(not chosen or incremental > 0.0)
        trace.extend(metrics.assign(step=len(chosen) + 1, chosen_head=head, accepted=accepted, selection_score=score, incremental_score=incremental).to_dict("records"))
        if not accepted:
            break
        chosen.append(head)
        remaining.remove(head)
        current_score = score
    if not chosen:
        raise AssertionError("no head selected")
    out.mkdir(parents=True)
    stats.to_parquet(out / "head_complementarity.parquet", index=False, compression="zstd")
    pd.DataFrame(trace).to_parquet(out / "head_selection_trace.parquet", index=False, compression="zstd")
    # Write best-1...best-N receipts without any policy/semantic fields.  The
    # standardized five slots allow the existing MC1 adapter to compare them
    # in aggregate mode without changing its inference feature semantics.
    for count in range(1, len(chosen) + 1):
        selected = chosen[:count]
        for month, part in raw.groupby(pd.to_datetime(raw["__decision_ts__"], utc=True).dt.strftime("%Y-%m"), sort=True):
            ranks = part.loc[:, selected].to_numpy(float)
            consensus = np.nanmedian(ranks, axis=1).astype(np.float32)
            receipt = part.loc[:, ["candidate_id", "__decision_ts__", "side_name", "f1_base_rank_ts"]].copy()
            receipt = receipt.rename(columns={"f1_base_rank_ts": "base_rank_ts"})
            receipt["conditional_consensus_rank"] = consensus
            receipt["ordinary_shadow_consensus_rank"] = consensus
            receipt["head_agreement_std"] = np.nanstd(ranks, axis=1).astype(np.float32)
            receipt["o3v2_rank_75_25"] = .75 * pd.to_numeric(receipt["base_rank_ts"], errors="coerce") + .25 * consensus
            for slot_index, slot in enumerate(SLOTS):
                receipt[f"head__{slot}__rank"] = ranks[:, min(slot_index, count - 1)].astype(np.float32)
            dest = out / "target_free_scores" / f"best{count}" / f"month={month}.parquet"
            dest.parent.mkdir(parents=True, exist_ok=True)
            receipt.to_parquet(dest, index=False, compression="zstd")
    _write_exclusive(out / "selected_head_subsets.json", {
        "schema": SCHEMA, "target": target_name, "development_months": list(months),
        "selection": "greedy strict-OOF ensemble utility with conditional-correlation penalty; no MC1 outcome used",
        "ordered_heads": chosen, "subsets": {f"best{i}": chosen[:i] for i in range(1, len(chosen) + 1)},
    })
    _write_exclusive(out / "run_manifest.json", {
        "schema": SCHEMA, "scope": "offline O3-v2 head subset selection; no refit, MC1, portfolio, or live mutation",
        "source_score_root": str(score_root), "target": target_name, "development_months": list(months),
        "causality": "input head outputs were target-free strict OOF receipts; outcomes enter only after receipt loading for development selection",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--months", required=True)
    args = parser.parse_args()
    run(score_root=args.score_root, target_name=args.target, policy_path=args.policy_path, out=args.out, months=_months(args.months))


if __name__ == "__main__":
    main()
