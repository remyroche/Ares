#!/usr/bin/env python3
"""Causal locality gates for the incremental frozen-K9 history LDF arm.

This is deliberately a post-model *sizing* ablation.  It blends matched core
and core-plus-K9 LDF multipliers, never changes final-score ranking or
admission, and only trusts the K9 increment where the candidate has a
concentrated frozen-cluster membership and enough prior-resolved local support.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_strict_r3_n5_canonical_selection as selection  # noqa: E402


SEED = 20260811
GATE_COLUMNS = ("k9_top2_margin", "cluster_recent_7d_support")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--core-dir", type=Path, required=True)
    parser.add_argument("--history-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _quantiles(train: pd.DataFrame) -> dict[str, float]:
    margin = pd.to_numeric(train["k9_top2_margin"], errors="coerce")
    support = np.log1p(pd.to_numeric(train["cluster_recent_7d_support"], errors="coerce"))
    return {
        "margin_q75": float(margin.quantile(0.75)),
        "margin_q95": float(margin.quantile(0.95)),
        "support_q50": float(support.quantile(0.50)),
        "support_q75": float(support.quantile(0.75)),
        "support_q90": float(support.quantile(0.90)),
    }


def _gate(frame: pd.DataFrame, q: dict[str, float], arm: str) -> np.ndarray:
    margin = pd.to_numeric(frame["k9_top2_margin"], errors="coerce").fillna(-np.inf).to_numpy(float)
    support = np.log1p(pd.to_numeric(frame["cluster_recent_7d_support"], errors="coerce").fillna(0.0).to_numpy(float))
    if arm == "history_ungated":
        return np.ones(len(frame), dtype=np.float32)
    if arm == "history_hard_m75_s50":
        return ((margin >= q["margin_q75"]) & (support >= q["support_q50"])).astype(np.float32)
    if arm == "history_hard_m90_s75":
        return ((margin >= q["margin_q95"]) & (support >= q["support_q75"])).astype(np.float32)
    if arm == "history_soft_m75_95_x_s50_90":
        m_scale = max(q["margin_q95"] - q["margin_q75"], 1e-9)
        s_scale = max(q["support_q90"] - q["support_q50"], 1e-9)
        m = np.clip((margin - q["margin_q75"]) / m_scale, 0.0, 1.0)
        s = np.clip((support - q["support_q50"]) / s_scale, 0.0, 1.0)
        return (m * s).astype(np.float32)
    raise ValueError(f"unknown gate arm: {arm}")


def _load_outputs(path: Path) -> pd.DataFrame:
    parts = sorted(path.glob("compact_additive_fold*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no compact fold outputs under {path}")
    return pd.concat([pd.read_parquet(part) for part in parts], ignore_index=True)


def main() -> None:
    args = _args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    core = _load_outputs(args.core_dir)
    history = _load_outputs(args.history_dir).loc[:, ["candidate_id", "trust_size_multiplier"]].rename(
        columns={"trust_size_multiplier": "history_multiplier"}
    )
    output = core.merge(history, on="candidate_id", validate="one_to_one")
    surface = pd.read_parquet(args.surface, columns=["candidate_id", "__decision_ts__", *GATE_COLUMNS])
    surface["__decision_ts__"] = pd.to_datetime(surface["__decision_ts__"], utc=True)
    output = output.merge(
        surface.loc[:, ["candidate_id", *GATE_COLUMNS]], on="candidate_id", validate="one_to_one"
    )
    output["__decision_ts__"] = pd.to_datetime(output["__decision_ts__"], utc=True)
    arms = ("core_control", "history_ungated", "history_hard_m75_s50", "history_hard_m90_s75", "history_soft_m75_95_x_s50_90")
    pieces: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    for fold, cutoff in enumerate(sorted(output["__decision_ts__"].dt.to_period("M").astype(str).unique())):
        held = output.loc[output["__decision_ts__"].dt.to_period("M").astype(str).eq(cutoff)].copy()
        cutoff_ts = pd.Timestamp(f"{cutoff}-01", tz="UTC")
        train = surface.loc[
            surface["__decision_ts__"].ge(cutoff_ts - pd.DateOffset(months=3))
            & surface["__decision_ts__"].lt(cutoff_ts)
        ]
        if train.empty:
            raise ValueError(f"no causal gate reference rows before {cutoff}")
        q = _quantiles(train)
        core_multiplier = pd.to_numeric(held["trust_size_multiplier"], errors="coerce").fillna(1.0).to_numpy(float)
        history_multiplier = pd.to_numeric(held["history_multiplier"], errors="coerce").fillna(1.0).to_numpy(float)
        for arm in arms:
            part = held.copy()
            if arm == "core_control":
                gate = np.zeros(len(part), dtype=np.float32)
            else:
                gate = _gate(part, q, arm)
            part["k9_locality_gate"] = gate
            part["trust_size_multiplier"] = core_multiplier + gate * (history_multiplier - core_multiplier)
            part["arm"] = arm
            part["fold"] = fold
            pieces.append(part)
            audit.append({"fold": fold, "cutoff": cutoff, "arm": arm, **q, "mean_gate": float(gate.mean()), "active_gate_rate": float((gate > 0).mean())})
    scored = pd.concat(pieces, ignore_index=True)
    args.out_dir.mkdir(parents=True)
    scored.to_parquet(args.out_dir / "oof_predictions.parquet", index=False, compression="zstd")
    metrics = []
    for arm, block in scored.groupby("arm", sort=True):
        metrics.extend([
            selection._period_tail_metrics(block, arm=str(arm), period_kind="global").assign(metric_kind="global"),
            selection._period_tail_metrics(block, arm=str(arm), period_kind="month").assign(metric_kind="month"),
        ])
    pd.concat(metrics, ignore_index=True).to_parquet(args.out_dir / "metrics.parquet", index=False)
    pd.DataFrame(audit).to_parquet(args.out_dir / "gate_audit.parquet", index=False)
    (args.out_dir / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_frozen_k9_locality_gate_ablation_v1",
        "core_dir": str(args.core_dir), "history_dir": str(args.history_dir),
        "surface": str(args.surface), "raw_k9_memberships_used": False,
        "semantics": "candidate-specific frozen-K9 history is mixed only into relative LDF sizing; ranking and admission are unchanged",
        "gate_reference": "preceding three calendar months of decision-time feature values; no outcomes used",
        "arms": list(arms),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
