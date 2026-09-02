#!/usr/bin/env python3
"""Aggregate isolated strict-OOF Router full-universe prescreen folds.

The prescreen deliberately permits one held fold per worker so large wide
matrices cannot retain memory across folds.  This utility only reads those
completed immutable part receipts; it never refits a model or reads outcomes.
It validates the common causal contract, combines feature evidence, applies
the deterministic redundancy representative veto, and seals the serious
feature pool for the next randomized-subspace stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import screen_strict_r3_router_full_universe_v1 as screen  # noqa: E402


SCHEMA = "strict_r3_router_full_universe_prescreen_aggregate_v1"


def _write_once(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _hash_lines(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _parts(value: str) -> list[Path]:
    paths = [Path(token.strip()).resolve() for token in value.split(",") if token.strip()]
    if len(paths) < 3 or len(paths) != len(set(paths)):
        raise ValueError("provide at least three unique completed part paths")
    return paths


def run(parts: list[Path], out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"immutable artifact already exists: {out}")
    contracts: list[dict[str, object]] = []
    manifests: list[dict[str, object]] = []
    for part in parts:
        contract_path = part / "run_contract.json"
        manifest_path = part / "run_manifest.json"
        required = [
            part / "fold_gain_split.parquet", part / "fold_shap.parquet", part / "fold_univariate.parquet",
            part / "fold_metrics.parquet", part / "redundancy_sample.parquet", contract_path, manifest_path,
        ]
        if any(not path.exists() for path in required):
            raise AssertionError(f"incomplete prescreen part: {part}")
        contract = json.loads(contract_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        if contract.get("schema") != screen.SCHEMA or manifest.get("status") != "complete":
            raise AssertionError(f"invalid completed prescreen part: {part}")
        if len(contract.get("held_months", [])) != 1:
            raise AssertionError(f"part does not contain exactly one held fold: {part}")
        contracts.append(contract)
        manifests.append(manifest)
    common_keys = (
        "feature_roots", "hygiene_contract", "hygiene_feature_sha256", "feature_count", "policy",
        "primary_target", "row_weight_scheme", "train_months", "reserve_days", "train_cap", "held_cap",
        "shap_rows", "tail_shap_rows", "univariate_rows", "serious_fields",
    )
    for key in common_keys:
        values = {json.dumps(contract.get(key), sort_keys=True) for contract in contracts}
        if len(values) != 1:
            raise AssertionError(f"part contract mismatch for {key}")
    held_months = [contract["held_months"][0] for contract in contracts]
    if len(set(held_months)) != len(held_months):
        raise AssertionError("duplicated held month across parts")
    order = sorted(range(len(parts)), key=lambda index: held_months[index])
    parts = [parts[index] for index in order]
    contracts = [contracts[index] for index in order]
    held_months = sorted(held_months)

    importance = pd.concat([pd.read_parquet(part / "fold_gain_split.parquet") for part in parts], ignore_index=True)
    shap = pd.concat([pd.read_parquet(part / "fold_shap.parquet") for part in parts], ignore_index=True)
    univariate = pd.concat([pd.read_parquet(part / "fold_univariate.parquet") for part in parts], ignore_index=True)
    fold_metrics = pd.concat([pd.read_parquet(part / "fold_metrics.parquet") for part in parts], ignore_index=True)
    samples = pd.concat([pd.read_parquet(part / "redundancy_sample.parquet") for part in parts], ignore_index=True)
    expected_fields = set(importance["feature"])
    if set(shap["feature"]) != expected_fields or set(univariate["feature"]) != expected_fields:
        raise AssertionError("incomplete per-feature evidence across isolated parts")
    summary = importance.groupby("feature", sort=False).agg(
        gain_median=("gain", "median"), gain_mean=("gain", "mean"),
        split_median=("split", "median"), split_presence=("split", lambda x: float(pd.Series(x).gt(0).mean())),
    ).reset_index().merge(
        shap.groupby("feature", sort=False).agg(
            shap_median=("mean_abs_shap", "median"), tail_shap_median=("tail_mean_abs_shap", "median"),
        ).reset_index(), on="feature", how="inner", validate="one_to_one",
    ).merge(
        univariate.groupby("feature", sort=False).agg(
            univariate_spearman_median=("univariate_spearman", "median"),
            univariate_spearman_abs=("univariate_spearman", lambda x: float(x.abs().median())),
        ).reset_index(), on="feature", how="inner", validate="one_to_one",
    )
    summary["screen_score"] = (
        .34 * screen._normalised_rank(summary["gain_median"])
        + .14 * screen._normalised_rank(summary["split_presence"])
        + .20 * screen._normalised_rank(summary["shap_median"])
        + .20 * screen._normalised_rank(summary["tail_shap_median"])
        + .12 * screen._normalised_rank(summary["univariate_spearman_abs"])
    )
    serious_fields = int(contracts[0]["serious_fields"])
    general = set(summary.nlargest(max(serious_fields * 2, 650), "screen_score")["feature"])
    rescue = set(summary.nlargest(max(serious_fields // 3, 100), "univariate_spearman_abs")["feature"])
    candidate = summary.loc[summary["feature"].isin(general | rescue)].copy()
    selected = screen._redundancy_veto(samples, candidate, serious_fields)
    summary["general_screen"] = summary["feature"].isin(general)
    summary["univariate_rescue"] = summary["feature"].isin(rescue)
    summary["serious_feature"] = summary["feature"].isin(selected)
    out.mkdir(parents=True)
    importance.to_parquet(out / "fold_gain_split.parquet", index=False, compression="zstd")
    shap.to_parquet(out / "fold_shap.parquet", index=False, compression="zstd")
    univariate.to_parquet(out / "fold_univariate.parquet", index=False, compression="zstd")
    fold_metrics.sort_values("held_month").to_parquet(out / "fold_metrics.parquet", index=False, compression="zstd")
    summary.sort_values(["screen_score", "feature"], ascending=[False, True], kind="stable").to_parquet(out / "feature_screen.parquet", index=False, compression="zstd")
    _write_once(out / "serious_feature_contract.json", {
        "schema": SCHEMA,
        "scope": "research-only Router serious feature pool; requires random-subspace selection and compression ladder",
        "feature_contract": selected, "feature_contract_sha256": _hash_lines(selected),
        "feature_count": len(selected), "pre_veto_candidate_count": len(candidate),
        "selection": "three isolated strict-OOF full-model folds; gain/split + global/tail SHAP + train-only univariate rescue + .97 Spearman representative veto",
        "source_parts": [str(part) for part in parts], "held_months": held_months,
        "primary_target": contracts[0]["primary_target"], "row_weight_scheme": contracts[0]["row_weight_scheme"],
    })
    _write_once(out / "run_manifest.json", {
        "schema": SCHEMA, "status": "complete",
        "scope": "offline aggregation of completed strict-OOF Router prescreen parts; no refit/live/exchange mutation",
        "source_parts": [str(part) for part in parts], "held_months": held_months,
        "hygiene_feature_count": int(contracts[0]["feature_count"]), "serious_features": len(selected),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", required=True, help="comma-separated completed isolated prescreen parts")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(_parts(args.parts), args.out.resolve())


if __name__ == "__main__":
    main()
