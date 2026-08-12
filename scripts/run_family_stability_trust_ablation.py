#!/usr/bin/env python3
"""Post-fit trust ablation using cross-partition family IC stability.

The family stability statistics are computed from meta-train and pre-test
calibration labels only. They are joined to frozen outer-test predictions and
used only to attenuate the already-fit residual correction; no outer outcome is
used to choose a threshold.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


TAILS = (0.005, 0.01, 0.02, 0.05, 0.10)
STABILITY_THRESHOLDS = (0.25, 0.50, 0.75)


def _metrics(frame: pd.DataFrame, score: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def one(block: pd.DataFrame, period: str) -> None:
        ordered = block.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable")
        for tail in TAILS:
            n = max(1, int(math.ceil(len(ordered) * tail)))
            chosen = ordered.head(n)
            net = chosen["policy_net_bps"].to_numpy(float)
            rows.append({
                "score": score, "period": period, "tail": tail, "trades": int(n),
                "gross_bps_per_trade": float(np.nanmean(chosen["policy_gross_bps"])),
                "net_bps_per_trade": float(np.nanmean(net)),
                "win_rate_net": float(np.mean(net > 0.0)),
                "median_net_bps": float(np.nanmedian(net)),
                "p10_net_bps": float(np.nanpercentile(net, 10)),
            })

    one(frame, "pooled")
    month = pd.to_datetime(frame["__ts__"], utc=True).dt.strftime("%Y-%m")
    for name, block in frame.assign(_period=month).groupby("_period", observed=True):
        one(block, str(name))
    return rows


def _load_stability(root: Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for path in sorted(root.glob("fold_audit_*.json")):
        audit = json.loads(path.read_text())
        rows = audit.get("authority_selection_audit") or []
        if not rows:
            raise ValueError(f"missing authority_selection_audit in {path}")
        result[str(audit["fold"])] = {
            str(row["family"]): float(row.get("authority_score", np.nan))
            for row in rows
        }
    if not result:
        raise ValueError(f"no fold audits found below {root}")
    return result


def run(args: argparse.Namespace) -> Path:
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f"refusing to overwrite populated output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.predictions)
    test = frame.loc[frame["split"].eq("test")].copy().reset_index(drop=True)
    required = {
        "fold", "candidate_id", "__ts__", "policy_net_bps", "policy_gross_bps",
        "cap120_policy_correction", "mlp_residual_delta", "family_selected_mass",
        "family_assignment_quality",
    }
    missing = sorted(required.difference(test.columns))
    if missing:
        raise ValueError(f"prediction artifact missing required fields: {missing}")
    family_fields = sorted(
        c for c in test.columns
        if c.startswith("base_structural_family__sf_") and not c.startswith("family_")
    )
    if not family_fields:
        raise ValueError("no frozen family contribution fields found")
    fold_scores = _load_stability(Path(args.stability_audit_root))
    stable_scores = np.zeros((len(test), len(family_fields)), dtype=np.float32)
    for fold, idx in test.groupby("fold", observed=True).groups.items():
        score_map = fold_scores.get(str(fold))
        if score_map is None:
            raise ValueError(f"no stability audit for fold {fold}")
        for j, family in enumerate(family_fields):
            stable_scores[idx, j] = float(score_map.get(family, np.nan))
    shares = test[[f"family_abs_share__{f}" for f in family_fields]].to_numpy(float)
    represented = test["family_selected_mass"].to_numpy(float).clip(0.0, 1.0)
    positive_stable = np.maximum(stable_scores, 0.0)
    stable_positive_mass = (shares * (stable_scores > 0.0)).sum(axis=1).clip(0.0, 1.0)
    stable_score_mass = (shares * positive_stable).sum(axis=1)
    stable_share = (stable_positive_mass / np.maximum(represented, 1e-8)).clip(0.0, 1.0)
    stable_score = (stable_score_mass / np.maximum(represented, 1e-8)).clip(0.0, 1.0)
    base = test["cap120_policy_correction"].to_numpy(float)
    delta = np.clip(0.50 * test["mlp_residual_delta"].to_numpy(float), -75.0, 75.0)
    quality = test["family_assignment_quality"].to_numpy(float).clip(0.0, 1.0)
    mass_scale = np.minimum(represented / 0.80, 1.0)
    scores: dict[str, np.ndarray] = {"arm_A_cap120": base}
    # Constants are predeclared and are not selected from outer outcomes.
    scores["stability_weighted"] = base + delta * stable_share
    scores["stability_mass_weighted"] = base + delta * stable_share * mass_scale
    scores["stability_quality_blended"] = base + delta * stable_share * np.minimum(quality / 0.45, 1.0)
    for threshold in STABILITY_THRESHOLDS:
        scores[f"stability_gate_{threshold:.2f}"] = base + np.where(stable_share >= threshold, delta, 0.0)
    test["stable_positive_mass"] = stable_positive_mass.astype("float32")
    test["stable_family_share"] = stable_share.astype("float32")
    test["stable_family_score"] = stable_score.astype("float32")
    metrics_rows: list[dict[str, object]] = []
    for name, value in scores.items():
        test[name] = value.astype("float32")
        metrics_rows.extend(_metrics(test, name))
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_parquet(out / "stability_ablation_metrics.parquet", index=False, compression="zstd")
    monthly = metrics.loc[
        metrics["tail"].eq(0.05) & metrics["period"].astype(str).str.fullmatch(r"\d{4}-\d{2}")
    ].copy()
    summary = []
    for score, block in monthly.groupby("score", observed=True):
        worst = block.loc[block["net_bps_per_trade"].idxmin()]
        summary.append({
            "score": score, "months": int(len(block)),
            "mean_net_bps_per_trade": float(block["net_bps_per_trade"].mean()),
            "median_net_bps_per_trade": float(block["net_bps_per_trade"].median()),
            "worst_net_bps_per_trade": float(worst["net_bps_per_trade"]),
            "worst_month": str(worst["period"]),
            "positive_months": int((block["net_bps_per_trade"] > 0.0).sum()),
        })
    pd.DataFrame(summary).sort_values("mean_net_bps_per_trade", ascending=False).to_parquet(
        out / "stability_ablation_monthly_top5.parquet", index=False, compression="zstd"
    )
    test.to_parquet(out / "stability_ablation_predictions.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "family_stability_trust_ablation_v1", "status": "complete",
        "predictions": str(args.predictions), "stability_audit_root": str(args.stability_audit_root),
        "outer_test_rows": int(len(test)), "family_count": len(family_fields),
        "thresholds": list(STABILITY_THRESHOLDS), "rules_predeclared": True,
        "oos_outcomes_used_for_selection": False,
        "stability_definition": "positive lower(train_rank_ic, calibration_rank_ic) family mass divided by represented mass",
        "policy": "global pooled tails; exact 15m trailing policy labels already frozen upstream",
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--stability-audit-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
    print(run(ap.parse_args()))


if __name__ == "__main__":
    main()
