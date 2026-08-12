#!/usr/bin/env python3
"""Evaluate a small predeclared causal trust/OOD gate for path corrections.

The gate never uses outcomes or fitted thresholds.  It only modulates the
already-produced cluster correction using path representation/margin and
strict OOF regime support fields.  This is a diagnostic reliability overlay,
not a new cluster or residual fit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLAY = ROOT / "data_perp/artifacts/tp6_sl4_conditional_cluster_residual_20260811_extended_regime_v1"
DEFAULT_META = ROOT / "data_perp/artifacts/tp6_sl4_extended_cluster_meta_pool_regime_20260811_v1.parquet"
DEFAULT_OUT = ROOT / "data_perp/artifacts/tp6_sl4_cluster_reliability_gate_20260811_v1"
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
LAMBDAS = (0.25, 0.50, 0.75, 1.00)
GATE_FIELDS = [
    "cluster_path_represented_mass", "cluster_path_entropy", "cluster_path_top2_margin",
    "regime_state_ood_score", "regime_state_margin", "state_switch_probability",
    "transition_state_ood_score", "transition_state_margin", "transition_state_entropy",
]


def _gate_features(frame: pd.DataFrame) -> pd.DataFrame:
    # All fields are decision-time path/regime descriptors.  No target-derived
    # assignment-quality field is used here because the current family matrix
    # emits zero for that optional diagnostic when no confidence calibration is
    # available; represented mass and path margin remain valid.
    required = GATE_FIELDS
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"reliability gate missing causal fields: {missing}")
    return frame.loc[:, required].apply(pd.to_numeric, errors="coerce")


def _metrics(frame: pd.DataFrame, score: str, gate: str, lam: float, gate_weight: pd.Series) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    ranked = frame.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable")
    for tail in TAILS:
        n = max(1, int(np.ceil(len(ranked) * tail)))
        selected = ranked.head(n)
        out.append({
            "gate": gate, "lambda": float(lam), "tail": float(tail), "period": "all",
            "trades": int(len(selected)),
            "gross_bps_per_trade": float(selected.gross_bps.mean()),
            "net_bps_per_trade": float(selected.net_bps.mean()),
            "rank_ic": float(frame[score].corr(frame.net_bps, method="spearman")),
            "admission_rate": float(frame[score].notna().mean()),
            "gate_positive_rate": float((gate_weight > 0).mean()),
            "gate_mean_weight": float(gate_weight.mean()),
        })
    for month, group in frame.groupby("month_key", sort=True):
        ranked_month = group.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable")
        for tail in (0.01, 0.05, 0.10):
            n = max(1, int(np.ceil(len(ranked_month) * tail)))
            selected = ranked_month.head(n)
            out.append({
                "gate": gate, "lambda": float(lam), "tail": float(tail), "period": str(month),
                "trades": int(len(selected)),
                "gross_bps_per_trade": float(selected.gross_bps.mean()),
                "net_bps_per_trade": float(selected.net_bps.mean()),
                "rank_ic": float(group[score].corr(group.net_bps, method="spearman")) if group[score].nunique() > 1 else np.nan,
                "admission_rate": float(frame[score].notna().mean()),
                "gate_positive_rate": float((gate_weight.loc[group.index] > 0).mean()),
                "gate_mean_weight": float(gate_weight.loc[group.index].mean()),
            })
    return out


def run(*, replay_dir: Path = DEFAULT_REPLAY, meta_path: Path = DEFAULT_META, out_dir: Path = DEFAULT_OUT) -> Path:
    replay_dir, meta_path, out_dir = map(Path, (replay_dir, meta_path, out_dir))
    predictions = pd.read_parquet(replay_dir / "conditional_cluster_oof_predictions.parquet")
    features = pd.read_parquet(replay_dir / "conditional_cluster_features_oof.parquet")
    meta = pd.read_parquet(meta_path, columns=[
        "candidate_id", "regime_state_ood_score", "regime_state_margin", "state_switch_probability",
        "transition_state_ood_score", "transition_state_margin", "transition_state_entropy",
    ])
    frame = predictions.merge(features, on=["candidate_id", "decision_ts", "month_key", "fold"], validate="one_to_one")
    frame = frame.merge(meta, on="candidate_id", validate="one_to_one")
    trust = _gate_features(frame)
    represented = trust["cluster_path_represented_mass"].clip(0.0, 1.0)
    margin = trust["cluster_path_top2_margin"].clip(0.0, 1.0)
    ood = trust["regime_state_ood_score"].clip(0.0, 1.0)
    transition_margin = trust["transition_state_margin"].clip(0.0, 1.0)
    # Every gate is declared before inspecting outcomes.  The soft gate is
    # bounded and monotone: weak path representation or weak state support
    # shrinks, but never reverses, the signed correction.
    gates: dict[str, pd.Series] = {
        "all": pd.Series(1.0, index=frame.index),
        "path_repr_025": (represented >= 0.25).astype(float),
        "path_repr_050": (represented >= 0.50).astype(float),
        "path_repr_margin": ((represented >= 0.50) & (margin >= 0.25)).astype(float),
        "path_regime_support": ((represented >= 0.50) & (margin >= 0.25) & (ood >= 0.20) & (transition_margin >= 0.20)).astype(float),
        "soft_path_regime_support": (
            (represented / 0.50).clip(0.0, 1.0)
            * (margin / 0.25).clip(0.0, 1.0)
            * (ood / 0.20).clip(0.0, 1.0)
            * (transition_margin / 0.20).clip(0.0, 1.0)
        ).astype(float),
    }
    outputs: list[dict[str, object]] = []
    score_frame = frame.loc[:, ["candidate_id", "month_key", "fold", "gross_bps", "net_bps", "base_score", "cluster_only_score", "cluster_context_score"]].copy()
    score_frame["cluster_only_correction"] = score_frame.cluster_only_score - score_frame.base_score
    score_frame["cluster_context_correction"] = score_frame.cluster_context_score - score_frame.base_score
    for gate_name, weight in gates.items():
        for lam in LAMBDAS:
            for source in ("cluster_only", "cluster_context"):
                score_frame["_score"] = score_frame.base_score + float(lam) * score_frame[f"{source}_correction"] * weight.to_numpy(float)
                outputs.extend(_metrics(score_frame, "_score", f"{source}__{gate_name}", float(lam), weight))
    metrics = pd.DataFrame(outputs)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(out_dir / "reliability_gate_metrics.parquet", index=False, compression="zstd")
    coverage = pd.DataFrame([
        {
            "gate": name,
            "rows": int(len(weight)),
            "positive_weight_rows": int((weight > 0).sum()),
            "positive_weight_rate": float((weight > 0).mean()),
            "mean_weight": float(weight.mean()),
            "p10_weight": float(weight.quantile(.10)),
            "median_weight": float(weight.quantile(.50)),
            "p90_weight": float(weight.quantile(.90)),
        }
        for name, weight in gates.items()
    ])
    coverage.to_parquet(out_dir / "reliability_gate_coverage.parquet", index=False, compression="zstd")
    monthly = metrics.loc[(metrics["period"] != "all") & (metrics["tail"] == 0.05)].copy()
    stability = monthly.groupby(["gate", "lambda"], as_index=False).agg(
        months=("period", "nunique"), mean_top5_net_bps=("net_bps_per_trade", "mean"),
        median_top5_net_bps=("net_bps_per_trade", "median"), worst_month_top5_net_bps=("net_bps_per_trade", "min"),
        best_month_top5_net_bps=("net_bps_per_trade", "max"), positive_months=("net_bps_per_trade", lambda x: int((x > 0).sum())),
    )
    stability.to_parquet(out_dir / "reliability_gate_stability.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "tp6_sl4_cluster_reliability_gate_v1",
        "replay": str(replay_dir), "meta_pool": str(meta_path), "rows": int(len(frame)),
        "gates": list(gates), "lambdas": list(LAMBDAS), "tails": list(TAILS),
        "features": GATE_FIELDS,
        "thresholds_predeclared": {"represented_mass": [0.25, 0.50], "path_margin": 0.25, "regime_ood": 0.20, "transition_margin": 0.20},
        "outcome_free_gate_inputs": True,
        "fit": "none; all thresholds declared before evaluation",
        "global_ranking": True,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = ["# TP6/SL4 conditional path reliability/OOD gate", "", "All thresholds are predeclared causal path/regime conditions; no gate is fitted on held outcomes.", "", "The overlay shrinks the cluster correction; it does not filter rows. The score coverage is therefore 100% for every arm, while gate coverage and mean weight are recorded separately in `reliability_gate_coverage.parquet`.", "", "## Pooled top-5 net bps/trade", "", "| arm | lambda | net | mean monthly | worst month | positive months |", "|---|---:|---:|---:|---:|---:|"]
    pooled = metrics.loc[(metrics["period"] == "all") & (metrics["tail"] == 0.05)].merge(stability, on=["gate", "lambda"], how="left")
    for _, row in pooled.sort_values("net_bps_per_trade", ascending=False).iterrows():
        lines.append(
            f"| {row['gate']} | {float(row['lambda']):.2f} | "
            f"{float(row['net_bps_per_trade']):.2f} | "
            f"{float(row['mean_top5_net_bps']):.2f} | "
            f"{float(row['worst_month_top5_net_bps']):.2f} | "
            f"{int(row['positive_months'])}/{int(row['months'])} |"
        )
    (out_dir / "RELIABILITY_GATE_REPORT.md").write_text("\n".join(lines) + "\n")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(run(replay_dir=args.replay_dir, meta_path=args.meta, out_dir=args.out))
