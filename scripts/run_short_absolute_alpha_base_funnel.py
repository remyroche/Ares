#!/usr/bin/env python3
"""Worker runner for the short absolute-alpha LambdaRank funnel.

This is a sequential *base-layer* research contract.  It preserves target-free
features and exact completed one-minute parent-policy labels, but selects arms
on global absolute policy-net tails rather than within-hour uplift alone.
Workers are intentionally one arm × one chronological fold so native LightGBM
memory is released between candidates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_short_policy_conversion_funnel import PolicySpec, run  # noqa: E402


SEED = 17
P0_CONFIG = ROOT / "config/strict_r3_short_p0_f90_base_v1.json"
POLICIES = ROOT / "data_perp/artifacts/strict_r3_short_policy_conversion_labels_12m_20260820_v1"
FEATURES = ROOT / "data_perp/artifacts/strict_r3_short_features_full2024_20260820_v1/canonical120_features.parquet"
CANDIDATES = ROOT / "data_perp/artifacts/strict_r3_short_base_3m_train_3m_oos_2024_20260820_v1/short_target_free_candidate_population.parquet"

FOLDS: dict[str, tuple[str, str, str]] = {
    "D1": ("2023-10-01", "2024-05-01", "2024-07-01"),
    "D2": ("2023-10-01", "2024-07-01", "2024-09-01"),
    "F3": ("2023-10-01", "2024-09-01", "2024-10-01"),
}

# The order is intentional: Q is a single-factor breadth screen; labels and
# weight authority are only evaluated after a breadth choice is frozen.
ARMS: dict[str, PolicySpec] = {
    "Q1_policy": PolicySpec("Q1_policy", "Current one-hour policy-net relevance control.", "policy_bps", truncation=32, gain_family="linear", query_hours=1),
    "Q2_policy": PolicySpec("Q2_policy", "Two-hour policy-net training queries.", "policy_bps", truncation=32, gain_family="linear", query_hours=2),
    "Q4_policy": PolicySpec("Q4_policy", "Four-hour policy-net training queries.", "policy_bps", truncation=32, gain_family="linear", query_hours=4),
    "Q8_policy": PolicySpec("Q8_policy", "Eight-hour policy-net training queries.", "policy_bps", truncation=32, gain_family="linear", query_hours=8),
    "Q24_policy": PolicySpec("Q24_policy", "UTC-day policy-net training queries.", "policy_bps", truncation=32, gain_family="linear", query_hours=24),
    "L1_cost_deadzone": PolicySpec("L1_cost_deadzone", "Compress losses; resolve policy net around the cost-clearing dead zone.", "policy_edges", truncation=32, gain_family="cost_deadzone", absolute_edges_bps=(-100.0, 25.0, 100.0, 200.0, 400.0)),
    "L2_cost_margin": PolicySpec("L2_cost_margin", "Strong cost-margin relevance: <=0, +50, +100, +200, +350, +600 bps.", "policy_edges", truncation=32, gain_family="cost_margin", absolute_edges_bps=(0.0, 50.0, 100.0, 200.0, 350.0, 600.0)),
    "N25": PolicySpec("N25", "Policy net quantized at 25 bps.", "policy_quantized", truncation=32, gain_family="linear", quantization_bps=25.0),
    "N50": PolicySpec("N50", "Policy net quantized at 50 bps.", "policy_quantized", truncation=32, gain_family="linear", quantization_bps=50.0),
    "N75": PolicySpec("N75", "Policy net quantized at 75 bps.", "policy_quantized", truncation=32, gain_family="linear", quantization_bps=75.0),
    "N100": PolicySpec("N100", "Policy net quantized at 100 bps.", "policy_quantized", truncation=32, gain_family="linear", quantization_bps=100.0),
    "N150": PolicySpec("N150", "Policy net quantized at 150 bps.", "policy_quantized", truncation=32, gain_family="linear", quantization_bps=150.0),
    "H25": PolicySpec("H25", "25% relative rank plus 75% absolute cost-margin relevance.", "hybrid_absolute_relative", truncation=32, gain_family="cost_margin", absolute_edges_bps=(0.0, 50.0, 100.0, 200.0, 350.0, 600.0), hybrid_relative_weight=.25),
    "H50": PolicySpec("H50", "50% relative rank plus 50% absolute cost-margin relevance.", "hybrid_absolute_relative", truncation=32, gain_family="cost_margin", absolute_edges_bps=(0.0, 50.0, 100.0, 200.0, 350.0, 600.0), hybrid_relative_weight=.50),
    "H75": PolicySpec("H75", "75% relative rank plus 25% absolute cost-margin relevance.", "hybrid_absolute_relative", truncation=32, gain_family="cost_margin", absolute_edges_bps=(0.0, 50.0, 100.0, 200.0, 350.0, 600.0), hybrid_relative_weight=.75),
    "W_opportunity": PolicySpec("W_opportunity", "Policy-net relevance with opportunity-spread query weighting.", "policy_bps", truncation=32, gain_family="linear", weight_kind="opportunity_spread"),
    "W_tail": PolicySpec("W_tail", "Policy-net relevance with bounded economic-tail row weighting.", "policy_bps", truncation=32, gain_family="linear", row_weight_kind="economic_tail"),
    "W_opportunity_tail": PolicySpec("W_opportunity_tail", "Opportunity-spread query weight times bounded economic-tail row weight.", "policy_bps", truncation=32, gain_family="linear", weight_kind="opportunity_spread", row_weight_kind="economic_tail"),
}
for _k in (8, 12, 16, 24, 32, 40, 48):
    ARMS[f"K{_k}"] = PolicySpec(f"K{_k}", f"Policy-net control with LambdaRank truncation K={_k}.", "policy_bps", truncation=_k, gain_family="linear")


def _utc(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _f90_fields() -> list[str]:
    config = json.loads(P0_CONFIG.read_text())
    selection = ROOT / str(config["feature_contract"]["selection_artifact"])
    fields = [str(value) for value in json.loads(selection.read_text())["feature_sets"]["90"]]
    if len(fields) != 90 or len(set(fields)) != 90:
        raise ValueError("frozen short F90 contract is invalid")
    return fields


def run_worker(*, out: Path, arm_name: str, fold_name: str, query_hours: int | None = None) -> Path:
    if out.exists():
        raise FileExistsError(out)
    if arm_name not in ARMS or fold_name not in FOLDS:
        raise KeyError(f"unknown arm/fold: {arm_name}/{fold_name}")
    spec = ARMS[arm_name]
    if query_hours is not None:
        if query_hours not in {1, 2, 4, 8, 24}:
            raise ValueError("query-hours must be one of 1, 2, 4, 8, 24")
        spec = replace(spec, query_hours=int(query_hours), name=f"{spec.name}_q{query_hours}h")
    train_start, oos_start, oos_end = (_utc(value) for value in FOLDS[fold_name])
    fields = _f90_fields()
    run(
        out=out, policies=POLICIES, features_path=FEATURES, candidates_path=CANDIDATES,
        fields=fields, train_start=train_start, oos_start=oos_start, oos_end=oos_end, specs=(spec,),
    )
    receipt: dict[str, Any] = {
        "schema": "strict_r3_short_absolute_alpha_worker_v1", "side": "short", "seed": SEED,
        "arm": arm_name, "spec": asdict(spec), "fold": fold_name,
        "frozen_f90_config": str(P0_CONFIG), "frozen_f90_config_sha256": _sha256(P0_CONFIG),
        "features": str(FEATURES), "features_sha256": _sha256(FEATURES),
        "candidates": str(CANDIDATES), "candidates_sha256": _sha256(CANDIDATES),
        "policy_labels": str(POLICIES), "policy_manifest_sha256": _sha256(POLICIES / "run_manifest.json"),
        "advancement_contract": {
            "primary": "global absolute policy-net tail economics at 0.25%, 0.5%, 1%, and 2% after the 100-bps cost",
            "guardrail": "within-hour policy IC and month-level absolute-tail stability",
            "hard_gate_for_promotion": "top 0.5% > +50 bps; top 1% > +25 bps; no development month below zero at top 0.5%",
            "strictness": "training labels resolve before held OOS decision timestamps; OOS candidate scores stay target-free until evaluation",
        },
    }
    (out / "absolute_alpha_worker_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-out", type=Path)
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--fold", choices=tuple(FOLDS))
    parser.add_argument("--query-hours", type=int, default=None)
    parser.add_argument("--list-arms", action="store_true")
    args = parser.parse_args()
    if args.list_arms:
        print(json.dumps({name: asdict(spec) for name, spec in ARMS.items()}, indent=2))
        return
    if args.worker_out is None or args.arm is None or args.fold is None:
        parser.error("--worker-out, --arm, and --fold are required")
    print(run_worker(out=args.worker_out.resolve(), arm_name=args.arm, fold_name=args.fold, query_hours=args.query_hours))


if __name__ == "__main__":
    main()
