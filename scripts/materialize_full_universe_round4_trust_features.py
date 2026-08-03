#!/usr/bin/env python3
"""Materialise the strictly causal Round-4 trust-feature contract.

The resulting ledger intentionally contains only decision-time fields.  It
joins the frozen three-state base simplex to the existing daily-prequential B2
payoff mapping, whose construction is already frozen in the Round-B artifact.
It does *not* manufacture seed, geometry, teacher, or meta-ensemble stability
from incompatible experiments.  Those fields are recorded as unavailable in
the manifest so a later runner cannot silently mistake a proxy for a valid
same-contract disagreement signal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "data_perp/artifacts/full_universe_base_hpo_20260802_v1"
DEFAULT_DEV = ROOT / "data_perp/artifacts/full_universe_round_b_residual_dev_20260803_v1/predictions.parquet"
DEFAULT_OOS = ROOT / "data_perp/artifacts/full_universe_round_b_residual_20260803_v1/predictions.parquet"
DEFAULT_OUT = ROOT / "data_perp/artifacts/full_universe_round4_trust_features_20260803_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--dev-mapping", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--oos-mapping", type=Path, default=DEFAULT_OOS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _base_predictions(root: Path) -> pd.DataFrame:
    parts = []
    cols = ["candidate_id", "__ts__", "side_name", "p_upper", "p_lower", "p_timeout"]
    for side in ("long", "short"):
        frame = pd.read_parquet(root / side / "target_screen_predictions.parquet", columns=cols)
        if set(frame["side_name"].unique()) != {side}:
            raise ValueError(f"{side} base artifact has unexpected side rows")
        parts.append(frame)
    out = pd.concat(parts, ignore_index=True)
    out["__ts__"] = pd.to_datetime(out["__ts__"], utc=True)
    if out["candidate_id"].duplicated().any():
        raise ValueError("frozen base candidate IDs must be unique")
    return out


def _mapping(path: Path) -> pd.DataFrame:
    cols = [
        "candidate_id", "__ts__", "side_name", "base_expected_net_bps",
        "base_expected_gross_bps", "base_payoff_mixture_sd_bps",
    ]
    frame = pd.read_parquet(path, columns=cols)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
    if frame["candidate_id"].duplicated().any():
        raise ValueError(f"mapping contains duplicate candidates: {path}")
    return frame


def _trust_features(base: pd.DataFrame, mapped: pd.DataFrame) -> pd.DataFrame:
    out = mapped.merge(base, on=["candidate_id", "__ts__", "side_name"], validate="one_to_one")
    if len(out) != len(mapped):
        raise ValueError("every B2-mapped row must have one frozen base simplex")
    probability = out[["p_upper", "p_lower", "p_timeout"]].to_numpy(dtype=float)
    if not np.allclose(probability.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5):
        raise ValueError("base probability simplex does not sum to one")
    safe = np.clip(probability, 1e-12, 1.0)
    ordered = np.sort(probability, axis=1)
    out["trust_base_entropy_normalised"] = -(safe * np.log(safe)).sum(axis=1) / np.log(3.0)
    out["trust_base_probability_hhi"] = (probability * probability).sum(axis=1)
    out["trust_base_top_probability"] = ordered[:, 2]
    out["trust_base_top2_probability_margin"] = ordered[:, 2] - ordered[:, 1]
    out["trust_base_upper_lower_margin"] = probability[:, 0] - probability[:, 1]
    out["trust_base_upper_competitor_margin"] = probability[:, 0] - np.maximum(probability[:, 1], probability[:, 2])
    # These are the B2 values constructed from prior-resolved conditional
    # gross payoffs, with the declared fixed 100 bps cost applied once.
    out["trust_mapped_payoff_mixture_sd_bps"] = out["base_payoff_mixture_sd_bps"]
    out["trust_mapped_value_margin_over_cost_bps"] = out["base_expected_net_bps"]
    keep = [
        "candidate_id", "__ts__", "side_name", "p_upper", "p_lower", "p_timeout",
        "base_expected_gross_bps", "base_expected_net_bps", "base_payoff_mixture_sd_bps",
        "trust_base_entropy_normalised", "trust_base_probability_hhi", "trust_base_top_probability",
        "trust_base_top2_probability_margin", "trust_base_upper_lower_margin",
        "trust_base_upper_competitor_margin", "trust_mapped_payoff_mixture_sd_bps",
        "trust_mapped_value_margin_over_cost_bps",
    ]
    return out[keep].sort_values(["__ts__", "candidate_id"], kind="mergesort").reset_index(drop=True)


def _availability() -> dict[str, object]:
    return {
        "materialised_exactly": {
            "base_entropy": "trust_base_entropy_normalised: Shannon entropy of frozen p_upper/p_lower/p_timeout",
            "base_margins": [
                "trust_base_top2_probability_margin",
                "trust_base_upper_lower_margin",
                "trust_base_upper_competitor_margin",
            ],
            "base_concentration": ["trust_base_probability_hhi", "trust_base_top_probability"],
            "mapped_uncertainty": "trust_mapped_payoff_mixture_sd_bps: B2 predicted-event mixture dispersion under the contemporaneous prior-resolved side-shrunk payoff map",
            "mapped_value_margin": "trust_mapped_value_margin_over_cost_bps: B2 expected gross minus exactly one fixed 100 bps cost",
        },
        "unavailable_not_synthesised": {
            "base_seed_stability": "No multi-seed predictions for the identical selected TP3/SL2, tau=.25, 200-tree base fit and score map.",
            "nearby_geometry_stability": "Only Apr--Jul older 80-tree sibling geometries exist; there are no matched selected-base sibling predictions for Aug--Nov.",
            "teacher_student_disagreement": "The stored distillation artifact contains a causal student prediction, not the future-path teacher prediction. Its model is also fitted through 2024-08-01, unlike the selected base fitted through 2024-04-01; it is not an interchangeable feature of the frozen selected stack.",
            "residual_ensemble_dispersion": "One selected residual model prediction is stored; no same-target ensemble ledger exists.",
            "reliability_ensemble_dispersion": "One selected cost-clear model prediction is stored; no same-target ensemble ledger exists.",
        },
        "permitted_round4_feature_contract": [
            "trust_base_entropy_normalised", "trust_base_probability_hhi", "trust_base_top_probability",
            "trust_base_top2_probability_margin", "trust_base_upper_lower_margin",
            "trust_base_upper_competitor_margin", "trust_mapped_payoff_mixture_sd_bps",
            "trust_mapped_value_margin_over_cost_bps",
        ],
        "leakage_statement": "All materialised fields are algebraic functions of frozen base probabilities and the existing B2 expected-payoff ledger. The materialiser reads no realised gross/net outcome, path, or label-availability field.",
    }


def main() -> None:
    args = parse_args()
    base = _base_predictions(args.base)
    dev = _trust_features(base, _mapping(args.dev_mapping))
    oos = _trust_features(base, _mapping(args.oos_mapping))
    if set(dev["candidate_id"]) & set(oos["candidate_id"]):
        raise ValueError("development and OOS trust ledgers overlap")
    args.out.mkdir(parents=True, exist_ok=True)
    dev.to_parquet(args.out / "development.parquet", index=False)
    oos.to_parquet(args.out / "oos.parquet", index=False)
    manifest = {
        "schema": "full_universe_round4_trust_features_v1",
        "base_artifact": str(args.base),
        "mapping_sources": {"development": str(args.dev_mapping), "oos": str(args.oos_mapping)},
        "development_rows": len(dev), "oos_rows": len(oos),
        "development_window": [str(dev["__ts__"].min()), str(dev["__ts__"].max())],
        "oos_window": [str(oos["__ts__"].min()), str(oos["__ts__"].max())],
        "coverage": {
            "development_all_finite": float(np.isfinite(dev.drop(columns=["candidate_id", "__ts__", "side_name"]).to_numpy(dtype=float)).all(axis=1).mean()),
            "oos_all_finite": float(np.isfinite(oos.drop(columns=["candidate_id", "__ts__", "side_name"]).to_numpy(dtype=float)).all(axis=1).mean()),
        },
        **_availability(),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"development_rows": len(dev), "oos_rows": len(oos), "coverage": manifest["coverage"]}))


if __name__ == "__main__":
    main()
