#!/usr/bin/env python3
"""Audit incremental causal information in a candidate H4 feature block.

Given strict-prior predictions from the unchanged 91-field control and a
target-free feature panel, this computes, per new feature:

* residual Spearman against the OOF action advantage after the control model;
* discretised conditional mutual information given the control score decile;
* directional top-1/2/5% action advantage and lift versus the control tail.

It is descriptive Stage-1 evidence only.  It does not train a policy, choose
an authority rule, access exchange data, or mutate a live/research contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score


KEY = ("candidate_id", "state_decision_ts")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _bins(values: pd.Series, count: int = 10) -> pd.Series:
    return pd.qcut(values.rank(method="first"), q=count, labels=False, duplicates="drop")


def _conditional_mi(feature: pd.Series, residual: pd.Series, control: pd.Series) -> float:
    frame = pd.DataFrame({"feature": feature, "residual": residual, "control": control}).dropna()
    if len(frame) < 250:
        return np.nan
    frame["feature_bin"] = _bins(frame["feature"])
    frame["residual_bin"] = _bins(frame["residual"])
    frame["control_bin"] = _bins(frame["control"])
    total = float(len(frame))
    return float(sum(
        len(group) / total * mutual_info_score(group["feature_bin"], group["residual_bin"])
        for _, group in frame.groupby("control_bin", observed=True)
        if len(group) >= 25
    ))


def _tail_mean(frame: pd.DataFrame, score: str, pct: int) -> float:
    count = max(1, int(np.ceil(len(frame) * pct / 100.0)))
    return float(frame.nlargest(count, score)["advantage_bps"].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--extra-feature-panel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    screen_root = args.screen_root.resolve()
    extra_path = args.extra_feature_panel.resolve()
    pred = pd.read_parquet(screen_root / "2025_strict_prior_exit_sensitivity_predictions.parquet")
    pred = pred.loc[
        pred["model_variant"].eq("control_91"),
        [*KEY, "held_month", "advantage_bps", "direct_advantage_score"],
    ].copy()
    if pred.duplicated(list(KEY)).any():
        raise AssertionError("control OOF prediction duplicates state identity")
    extra = pd.read_parquet(extra_path).copy()
    extra["candidate_id"] = extra["candidate_id"].astype(str)
    extra["state_decision_ts"] = pd.to_datetime(extra["state_decision_ts"], utc=True, errors="raise")
    numeric = tuple(str(column) for column in extra.columns if column not in KEY and pd.api.types.is_numeric_dtype(extra[column]))
    if not numeric or extra.duplicated(list(KEY)).any():
        raise AssertionError("extra panel must provide unique numeric state fields")
    panel = pred.merge(extra.loc[:, [*KEY, *numeric]], on=list(KEY), how="inner", validate="one_to_one")
    if len(panel) != len(pred):
        raise AssertionError("extra panel does not cover every strict-prior OOF state")
    panel["control_residual_bps"] = panel["advantage_bps"] - panel["direct_advantage_score"]
    records: list[dict[str, object]] = []
    for feature in numeric:
        source = panel.loc[:, ["advantage_bps", "direct_advantage_score", "control_residual_bps", feature]].dropna().copy()
        if len(source) < 250 or source[feature].nunique() < 3:
            continue
        residual_spearman = float(source[feature].corr(source["control_residual_bps"], method="spearman"))
        # Direction follows the residual relationship, then uses the same
        # OOF state rows to report a transparent tail diagnostic.
        source["directional_feature_score"] = source[feature] * (1.0 if residual_spearman >= 0.0 else -1.0)
        row: dict[str, object] = {
            "feature": feature,
            "rows": int(len(source)),
            "residual_spearman": residual_spearman,
            "conditional_mutual_information_nats": _conditional_mi(source[feature], source["control_residual_bps"], source["direct_advantage_score"]),
        }
        for pct in (1, 2, 5):
            feature_mean = _tail_mean(source, "directional_feature_score", pct)
            control_mean = _tail_mean(source, "direct_advantage_score", pct)
            row[f"top{pct}_feature_advantage_bps"] = feature_mean
            row[f"top{pct}_control_advantage_bps"] = control_mean
            row[f"top{pct}_feature_minus_control_bps"] = feature_mean - control_mean
        records.append(row)
    result = pd.DataFrame(records).sort_values(
        ["conditional_mutual_information_nats", "residual_spearman"], ascending=[False, False], kind="stable"
    )
    out.mkdir(parents=True, exist_ok=False)
    result.to_parquet(out / "incremental_information_by_feature.parquet", index=False, compression="zstd")
    panel.loc[:, [*KEY, "held_month", "advantage_bps", "direct_advantage_score", "control_residual_bps", *numeric]].to_parquet(
        out / "strict_prior_joined_information_panel.parquet", index=False, compression="zstd",
    )
    manifest = {
        "schema": "causal-sr-h4-incremental-information-v1",
        "scope": "offline descriptive Stage-1 audit only; no policy, model, admission, portfolio, MC1, C1 S/R, Geometry/K9, live, or exchange mutation",
        "strict_prior_source": str(screen_root),
        "strict_prior_predictions_sha256": _sha256(screen_root / "2025_strict_prior_exit_sensitivity_predictions.parquet"),
        "extra_feature_panel": str(extra_path),
        "extra_feature_panel_sha256": _sha256(extra_path),
        "residual": "realised temporary-action advantage minus strict-prior 91-field direct OOF score",
        "conditional_mi": "discretised mutual information(feature, residual | 91-field direct-score decile)",
        "tail_diagnostic": "feature direction follows residual Spearman; results are evidence only, not a selected authority rule",
        "no_exchange_calls": True,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
