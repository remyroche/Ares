#!/usr/bin/env python3
"""Strict-OOS per-head audit for the enhanced-base consensus contract.

This is deliberately a diagnostic-only companion to the full challenger.  It
re-fits each monthly five-head bundle exactly as the challenger does, persists
only target-free ranks, and joins policy outcomes afterwards.  It never fits
MC1, runs an admission map, or touches live execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import run_strict_r3_enhanced_base_live_stack_challenger as stack


AUDIT_MONTHS = tuple(pd.date_range("2025-10-01", "2026-07-01", freq="MS", tz="UTC"))
TAILS = (0.01, 0.02, 0.05)


def _sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _tail_mean(frame: pd.DataFrame, score: str, fraction: float) -> tuple[int, float]:
    n = max(1, int(np.ceil(fraction * len(frame))))
    selected = frame.nlargest(n, score, keep="all").head(n)
    return int(len(selected)), float(selected["policy_net_bps"].mean())


def _spearman(left: pd.Series, right: pd.Series) -> float:
    return float(left.corr(right, method="spearman"))


def _metrics_for_score(frame: pd.DataFrame, score: str) -> dict[str, float | int]:
    output: dict[str, float | int] = {
        "rows": int(len(frame)),
        "rank_ic_policy_net": _spearman(frame[score], frame["policy_net_bps"]),
        "rank_corr_to_consensus": _spearman(frame[score], frame["conditional_consensus_rank"]),
    }
    for fraction in TAILS:
        rows, bps = _tail_mean(frame, score, fraction)
        token = f"top{int(fraction * 100)}"
        output[f"{token}_rows"] = rows
        output[f"{token}_net_bps"] = bps
    return output


def _per_head_metrics(panel: pd.DataFrame, head_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    whole: list[dict[str, object]] = []
    monthly: list[dict[str, object]] = []
    for head in head_columns:
        name = head.removeprefix("head__").removesuffix("__rank")
        record: dict[str, object] = {"head": name, **_metrics_for_score(panel, head)}
        others = [column for column in head_columns if column != head]
        loo = panel.loc[:, others].median(axis=1)
        record["loo_top1_net_bps"] = _tail_mean(panel.assign(__loo__=loo), "__loo__", .01)[1]
        record["loo_top2_net_bps"] = _tail_mean(panel.assign(__loo__=loo), "__loo__", .02)[1]
        record["loo_top5_net_bps"] = _tail_mean(panel.assign(__loo__=loo), "__loo__", .05)[1]
        record["full_consensus_top1_net_bps"] = _tail_mean(panel, "conditional_consensus_rank", .01)[1]
        record["full_consensus_top2_net_bps"] = _tail_mean(panel, "conditional_consensus_rank", .02)[1]
        record["full_consensus_top5_net_bps"] = _tail_mean(panel, "conditional_consensus_rank", .05)[1]
        record["leave_one_out_delta_top1_bps"] = record["full_consensus_top1_net_bps"] - record["loo_top1_net_bps"]
        record["leave_one_out_delta_top2_bps"] = record["full_consensus_top2_net_bps"] - record["loo_top2_net_bps"]
        record["leave_one_out_delta_top5_bps"] = record["full_consensus_top5_net_bps"] - record["loo_top5_net_bps"]
        whole.append(record)
        for month, part in panel.groupby(panel["__decision_ts__"].dt.strftime("%Y-%m"), sort=True):
            monthly.append({"head": name, "month": month, **_metrics_for_score(part, head)})
    return pd.DataFrame(whole).sort_values("top5_net_bps", ascending=False), pd.DataFrame(monthly).sort_values(["head", "month"])


def _correlations(panel: pd.DataFrame, heads: list[str]) -> pd.DataFrame:
    corr = panel.loc[:, heads].corr(method="spearman")
    output: list[dict[str, object]] = []
    for left in heads:
        for right in heads:
            if left < right:
                output.append({
                    "head_left": left.removeprefix("head__").removesuffix("__rank"),
                    "head_right": right.removeprefix("head__").removesuffix("__rank"),
                    "spearman_rank_correlation": float(corr.loc[left, right]),
                })
    return pd.DataFrame(output).sort_values("spearman_rank_correlation")


def _layer_metrics(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the aggregate meta coordinates alongside individual heads."""

    layers = {
        "median_selected_five_head_consensus": "conditional_consensus_rank",
        "ordinary_head_shadow_consensus": "ordinary_shadow_consensus_rank",
        "75_25_base_consensus_upstream": "upstream",
        "residual_correctness_demotion": "correctness_rank",
        "current_pre_mc1_final_score": "final_score",
    }
    whole: list[dict[str, object]] = []
    monthly: list[dict[str, object]] = []
    for name, column in layers.items():
        whole.append({"layer": name, "score_column": column, **_metrics_for_score(panel, column)})
        for month, part in panel.groupby(panel["__decision_ts__"].dt.strftime("%Y-%m"), sort=True):
            monthly.append({"layer": name, "score_column": column, "month": month, **_metrics_for_score(part, column)})
    return pd.DataFrame(whole), pd.DataFrame(monthly).sort_values(["layer", "month"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw-ledger", type=Path, required=True)
    parser.add_argument("--direct-root", type=Path, required=True)
    parser.add_argument(
        "--target-free-feature-root", type=Path, required=True,
        help="Completed challenger target-free monthly feature ledger; never a compact score-only source.",
    )
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--current-mc1", type=Path, required=True)
    parser.add_argument("--bcf-mc1", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    paths = stack.Paths(
        raw_ledger=args.raw_ledger.resolve(), direct_root=args.direct_root.resolve(),
        policy_root=args.policy_root.resolve(), current_mc1=args.current_mc1.resolve(),
        bcf_mc1=args.bcf_mc1.resolve(), bundle_root=args.bundle_root.resolve(),
    )
    fields = stack._base_fields(paths)
    policy = stack._load_policy(paths)
    audits: list[dict[str, object]] = []
    for month in AUDIT_MONTHS:
        print(json.dumps({"event": "head_audit_month_begin", "month": f"{month:%Y-%m}"}), flush=True)
        audit, current_path, _ = stack._score_fold(
            args.target_free_feature_root.resolve(), policy, fields, month, out,
        )
        audits.append(audit)
        print(json.dumps({"event": "head_audit_month_complete", **audit}), flush=True)
    raw = pd.concat([pd.read_parquet(path) for path in sorted((out / "target_free_scores" / "current").glob("*.parquet"))], ignore_index=True)
    forbidden = {"policy_net_bps", "policy_path_valid", "policy_label_available_ts", "policy_gross_bps"}
    if forbidden.intersection(raw.columns):
        raise AssertionError("target-free audit receipt contains an outcome field")
    panel = raw.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    panel["__decision_ts__"] = pd.to_datetime(panel["__decision_ts__"], utc=True)
    panel = panel.loc[
        panel["enhanced_base_routed"].fillna(False).astype(bool)
        & panel["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(panel["policy_net_bps"], errors="coerce"))
    ].copy()
    heads = sorted(column for column in panel.columns if column.startswith("head__") and column.endswith("__rank"))
    if len(heads) != stack.EXPECTED_RESEARCH_HEADS:
        raise AssertionError(f"expected {stack.EXPECTED_RESEARCH_HEADS} persisted head ranks, found {len(heads)}")
    all_metrics, monthly_metrics = _per_head_metrics(panel, heads)
    layer_metrics, layer_monthly_metrics = _layer_metrics(panel)
    correlation = _correlations(panel, heads)
    specs = stack._head_specs(fields)
    definition = pd.DataFrame([{
        "head": spec.name, "feature_count": len(spec.fields), "query": spec.query,
        "weight_mode": spec.weight_mode, "cap": spec.cap,
        "target": "policy_net_residual_ordinal_150_50",
        "target_edges_bps": json.dumps(list(spec.target_edges_bps)),
        "ranker_params": json.dumps(spec.params, sort_keys=True),
    } for spec in specs])
    pd.DataFrame(audits).to_parquet(out / "head_fit_audit.parquet", index=False, compression="zstd")
    all_metrics.to_parquet(out / "head_oos_metrics.parquet", index=False, compression="zstd")
    monthly_metrics.to_parquet(out / "head_oos_monthly_metrics.parquet", index=False, compression="zstd")
    layer_metrics.to_parquet(out / "meta_layer_oos_metrics.parquet", index=False, compression="zstd")
    layer_monthly_metrics.to_parquet(out / "meta_layer_oos_monthly_metrics.parquet", index=False, compression="zstd")
    correlation.to_parquet(out / "head_rank_correlation.parquet", index=False, compression="zstd")
    definition.to_parquet(out / "head_definition.parquet", index=False, compression="zstd")
    receipt = {
        "scope": "offline strict-OOS diagnostic; no MC1/admission/portfolio or live effects",
        "months": [f"{month:%Y-%m}" for month in AUDIT_MONTHS],
        "population": "enhanced-base routed rows with canonical valid policy outcomes joined only after target-free OOS scores",
        "head_target": "policy net residual ordinal bins [-150,-50,50,150] bps",
        "leave_one_out": "median of the other nine ranks; diagnostic rank contribution, not a refit portfolio counterfactual",
        "source_hashes": {
            "runner": _sha256([Path(stack.__file__)]),
            "contract": _sha256([stack.CONSENSUS_CONTRACT]),
            "policy": _sha256([paths.policy_root]),
        },
    }
    (out / "run_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out": str(out), "rows": int(len(panel)), "heads": len(heads)}), flush=True)


if __name__ == "__main__":
    main()
