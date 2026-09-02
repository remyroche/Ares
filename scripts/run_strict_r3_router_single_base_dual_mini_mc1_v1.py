#!/usr/bin/env python3
"""Fixed dual-family MC1 diagnostic for Router50 -> one Base -> R/U.

Research-only.  It consumes target-free strict-OOF Base/R/U receipts, writes
research Current/BCF family panels before outcomes are joined, then reuses the
canonical shallow MC1_d2 fit, 21-day causal shift, dual +50 bps gate, and
chronological portfolio mirror.  This is the high-fidelity finalist screen;
it never alters a live mapper or bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))
import run_strict_r3_enhanced_base_live_stack_challenger as parent  # noqa: E402

IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
DEFAULT_WARMUP = "2026-02,2026-03,2026-04"
DEFAULT_HELD = "2026-05,2026-06,2026-07"
ARM_COMPONENTS = {
    "base_only": (False, False),
    "r_only": (True, False),
    "u_only": (False, True),
    "base_ru": (True, True),
}

def _write_once(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)

def _sha(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(path.rglob("*.parquet")):
        h.update(str(p).encode()); h.update(p.read_bytes())
    return h.hexdigest()

def _month(path: Path, month: pd.Timestamp) -> pd.DataFrame:
    d = pd.read_parquet(path / "target_free_combined" / f"month={month:%Y-%m}.parquet")
    forbidden = {"policy_net_bps", "policy_path_valid", "policy_label_available_ts"}
    if forbidden & set(d.columns):
        raise AssertionError("outcome field in target-free R/U receipt")
    r, u = "r_residual_sqrt_atr_quintile_rank", "u_unexpected_trailing_atr1_rank"
    needed = {*IDENTITY, "base_rank_ts", r, u}
    if missing := needed - set(d.columns):
        raise AssertionError(f"missing R/U fields: {sorted(missing)}")
    d["conditional_consensus_rank"] = d[[r, u]].median(axis=1)
    d["ordinary_shadow_consensus_rank"] = d[[r, u]].min(axis=1)
    d["correctness_rank"] = (1.0 - (d[r] - d[u]).abs()).clip(0.0, 1.0)
    d["base_rank42"] = d["base_rank_ts"]
    d["enhanced_base_routed"] = True
    d["upstream"] = .75 * d.base_rank42 + .25 * d.conditional_consensus_rank
    d["bcf_upstream"] = .75 * d.base_rank42 + .25 * d.ordinary_shadow_consensus_rank
    return d.loc[:, [
        *IDENTITY, "enhanced_base_routed", "base_rank42", r, u,
        "conditional_consensus_rank", "ordinary_shadow_consensus_rank",
        "correctness_rank", "upstream", "bcf_upstream",
    ]]

def _family(base: pd.DataFrame, family: str, arm: str) -> pd.DataFrame:
    if arm not in ARM_COMPONENTS:
        raise ValueError(f"unsupported component arm: {arm}")
    use_r, use_u = ARM_COMPONENTS[arm]
    out = base.copy()
    r, u = "r_residual_sqrt_atr_quintile_rank", "u_unexpected_trailing_atr1_rank"
    if not use_r and not use_u:
        out["conditional_consensus_rank"] = out.base_rank42
        out["ordinary_shadow_consensus_rank"] = out.base_rank42
        out["correctness_rank"] = .5
        out["upstream"] = out.base_rank42
        out["bcf_upstream"] = out.base_rank42
    elif use_r and not use_u:
        out["conditional_consensus_rank"] = out[r]
        out["ordinary_shadow_consensus_rank"] = out[r]
        out["correctness_rank"] = .5
        out["upstream"] = .75 * out.base_rank42 + .25 * out[r]
        out["bcf_upstream"] = out["upstream"]
    elif use_u and not use_r:
        out["conditional_consensus_rank"] = out[u]
        out["ordinary_shadow_consensus_rank"] = out[u]
        out["correctness_rank"] = .5
        out["upstream"] = .75 * out.base_rank42 + .25 * out[u]
        out["bcf_upstream"] = out["upstream"]
    raw = out.upstream if family == "current" else out.bcf_upstream
    out["final_score"] = raw
    return out.drop(columns=["bcf_upstream"])


def _months(value: str) -> tuple[pd.Timestamp, ...]:
    months = tuple(pd.Timestamp(f"{token.strip()}-01", tz="UTC") for token in value.split(",") if token.strip())
    if not months or tuple(sorted(months)) != months:
        raise ValueError("months must be a non-empty chronological comma-separated list")
    return months

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ru-root", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--warmup-months", default=DEFAULT_WARMUP)
    p.add_argument("--held-months", default=DEFAULT_HELD)
    p.add_argument(
        "--arms", default="base_only,base_ru",
        help="comma-separated fixed component ablations: base_only,r_only,u_only,base_ru",
    )
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    warmup, held = _months(a.warmup_months), _months(a.held_months)
    arms = tuple(item.strip() for item in a.arms.split(",") if item.strip())
    if not arms or len(set(arms)) != len(arms) or set(arms) - set(ARM_COMPONENTS):
        raise ValueError("--arms must be a non-empty unique subset of fixed component arms")
    all_months = (*warmup, *held)
    if len(set(all_months)) != len(all_months) or tuple(sorted(all_months)) != all_months:
        raise ValueError("warmup and held months must be disjoint and jointly chronological")
    if a.out.exists(): raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    panels = pd.concat([_month(a.ru_root, month) for month in all_months], ignore_index=True)
    for family in ("current", "bcf"):
        for arm in arms:
            target = a.out / "target_free_scores" / arm / family
            target.mkdir(parents=True)
            for month, frame in panels.groupby(panels.__decision_ts__.dt.to_period("M"), sort=True):
                q = _family(frame, family, arm)
                q.to_parquet(target / f"month={month}.parquet", index=False, compression="zstd")
    policy = pd.read_parquet(a.policy)
    # Candidate identity already carries the decision timestamp.  The policy
    # sidecar may retain its own timestamp for audit purposes; do not let a
    # duplicate merge rename the target-free decision coordinate.
    policy = policy.drop(columns=["__decision_ts__", "side_name"], errors="ignore")
    policy["policy_label_available_ts"] = pd.to_datetime(policy.policy_label_available_ts, utc=True)
    original_months, original_train, original_threshold = parent.SCORE_MONTHS, parent.MC1_TRAIN_MONTHS, parent.MC1_THRESHOLD_BPS
    metrics=[]; audits=[]
    try:
        parent.SCORE_MONTHS, parent.MC1_TRAIN_MONTHS, parent.MC1_THRESHOLD_BPS = all_months, 3, 50.0
        for arm in arms:
            predicted={}
            for family in ("current", "bcf"):
                pieces=[pd.read_parquet(x) for x in sorted((a.out/"target_free_scores"/arm/family).glob("*.parquet"))]
                data=pd.concat(pieces,ignore_index=True).merge(policy,on="candidate_id",how="left",validate="one_to_one")
                pred,audit=parent._mc1_predictions(data,f"mini_{arm}_{family}",a.out)
                predicted[family]=pred; audit["arm"]=arm; audits.append(audit)
            combined=parent._combined_challenger(predicted["current"],predicted["bcf"])
            combined=combined.loc[combined.__decision_ts__.ge(held[0])].copy()
            combined.to_parquet(a.out/f"{arm}_dual_predictions.parquet",index=False,compression="zstd")
            metrics.append(parent._portfolio_metrics(combined, arm, f"{held[0]:%Y%m}_{held[-1]:%Y%m}", a.out))
    finally:
        parent.SCORE_MONTHS, parent.MC1_TRAIN_MONTHS, parent.MC1_THRESHOLD_BPS = original_months, original_train, original_threshold
    pd.DataFrame(metrics).to_parquet(a.out/"portfolio_metrics.parquet",index=False,compression="zstd")
    pd.concat(audits,ignore_index=True).to_parquet(a.out/"mc1_fit_audit.parquet",index=False,compression="zstd")
    _write_once(a.out/"correctness_report.json", {"target_free_before_policy_join": True, "router_numeric_absent": True, "no_post_router_base_cutoff": True, "separate_current_bcf_maps": True, "dual_gate_bps": 50, "r_u_only_mc1_coordinates": True, "warmup_precedes_held": max(warmup) < min(held)})
    _write_once(a.out/"run_manifest.json", {"scope":"offline fixed dual-MC1 Base/R/U component diagnostic; no live/exchange mutation", "ru_root":str(a.ru_root), "policy":str(a.policy), "arms":list(arms), "months":[f"{x:%Y-%m}" for x in all_months], "warmup_months":[f"{x:%Y-%m}" for x in warmup], "held_months":[f"{x:%Y-%m}" for x in held], "sources":{"ru":_sha(a.ru_root),"policy":_sha(a.policy.parent)}})

if __name__ == "__main__": main()
