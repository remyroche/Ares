#!/usr/bin/env python3
"""Verify the completed 2025 frozen T2 feature-contract materialisation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATERIALISATION = ROOT / "data_perp/artifacts/t2_fresh_frozen_feature_contract_20260801_v1"
DEFAULT_CONTRACT = ROOT / "data_perp/artifacts/controlled_target_supportive_prepared_ledger_20260801_v5/frozen_raw_causal_features.json"
DEFAULT_POPULATION = ROOT / "data_perp/artifacts/febapr2025_canonical_exact_policy_base_population_20260727_v1/population.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialisation", type=Path, default=DEFAULT_MATERIALISATION)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--population", type=Path, default=DEFAULT_POPULATION)
    args = parser.parse_args()
    fields = json.loads(args.contract.read_text(encoding="utf-8"))["raw_feature_columns"]
    manifest = json.loads((args.materialisation / "manifest.json").read_text(encoding="utf-8"))
    parts = sorted((args.materialisation / "parts").glob("*.parquet"))
    expected = pd.read_parquet(args.population, columns=["candidate_id"])
    finite_counts = np.zeros(len(fields), dtype=np.int64)
    all_complete = 0
    observed_ids: list[pd.Series] = []
    rows = 0
    for part in parts:
        frame = pd.read_parquet(part, columns=["candidate_id", *fields])
        values = frame.loc[:, fields].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        finite = np.isfinite(values)
        finite_counts += finite.sum(axis=0)
        all_complete += int(finite.all(axis=1).sum())
        rows += len(frame)
        observed_ids.append(frame["candidate_id"])
    ids = pd.concat(observed_ids, ignore_index=True)
    exact_ids = len(ids) == len(expected) and not ids.duplicated().any() and set(ids) == set(expected["candidate_id"])
    coverage = pd.DataFrame({"feature": fields, "finite_rows": finite_counts, "finite_rate": finite_counts / max(rows, 1)})
    coverage = coverage.sort_values(["finite_rate", "feature"], kind="stable")
    result = {
        "schema": "t2_fresh_frozen_feature_contract_verification_v1",
        "materialisation": str(args.materialisation.resolve()),
        "rows": int(rows),
        "parts": len(parts),
        "feature_count": len(fields),
        "identity_exact": bool(exact_ids),
        "features_at_least_90pct_finite": int((coverage["finite_rate"] >= 0.90).sum()),
        "features_below_90pct_finite": int((coverage["finite_rate"] < 0.90).sum()),
        "all_361_finite_rows": int(all_complete),
        "all_361_finite_rate": float(all_complete / max(rows, 1)),
        "contains_realised_cost_input": "execution_cost_return" in fields,
        "approved_latent_state_use": manifest["state_use_approval"],
        "status": "READY_FOR_FROZEN_REPLAY_WITH_NATIVE_MISSING_VALUE_ROUTING" if exact_ids else "BLOCKED_IDENTITY_MISMATCH",
        "caveat": "34 sparse fields remain missing on some rows, as in training; this is not a 361-fields-complete-case model.",
    }
    coverage.to_csv(args.materialisation / "per_feature_coverage.csv", index=False)
    (args.materialisation / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
