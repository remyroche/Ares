#!/usr/bin/env python3
"""Exact 2**4 component-combination funnel for the canonical residual meta layer.

The four requested blocks are enumerated without post-test feature selection:

* model support/OOD;
* archetype signed exposure;
* uncertainty;
* compact structural.

Every arm uses the same long-only 2025 rows, frozen canonical Base+Consensus
anchor, train-only isotonic expected-net map, ordinal residual target, and
4-hour x side LambdaRank fit.  The script is intentionally a separate runner
so the 16-arm contract and its selection rule remain auditable.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_tp6_sl4_downstream_retrain_2025 import MONTHS, _load  # noqa: E402
from scripts.run_tp6_sl4_canonical_residual_meta_granular_ablation import (  # noqa: E402
    DEFAULT_CONTRACT,
    DEFAULT_HEADS,
    DEFAULT_SELECTION,
    DEFAULT_STRUCTURAL,
    SEED,
    TAILS,
    _feature_frame,
    _fit_meta,
    _load_structural,
    _map_canonical,
    _metric,
)
from scripts.run_tp6_sl4_canonical_residual_meta_block_ablation import BLOCKS  # noqa: E402

DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/tp6_sl4_component_combo_funnel_long_20260808_v1"


def _safe(frame: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    return frame.reindex(columns=list(cols)).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _arms() -> list[dict[str, object]]:
    names = ("model_support_ood", "archetype_signed_exposure", "uncertainty", "compact_structural")
    out: list[dict[str, object]] = []
    for bits in itertools.product((0, 1), repeat=4):
        chosen = [name for name, bit in zip(names, bits) if bit]
        label = "control" if not chosen else "+".join(chosen)
        out.append({"arm": label, "bits": dict(zip(names, bits)), "components": chosen})
    return out


def _names_for(arm: dict[str, object], selected: dict[str, list[str]], structural_groups: dict[str, list[str]]) -> list[str]:
    bits = arm["bits"]
    parts: list[str] = []
    if bits["model_support_ood"]:
        parts.extend(selected.get("support_ood", BLOCKS["support_ood"]))
    if bits["uncertainty"]:
        parts.extend(selected.get("uncertainty", BLOCKS["uncertainty"]))
    if bits["compact_structural"]:
        parts.extend(structural_groups["structural_compact"])
    if bits["archetype_signed_exposure"]:
        # Signed exposure is deliberately kept distinct from the compact
        # structural summary.  It includes only signed archetype fields and
        # signed aggregate descriptors, never abs/active exposure fields.
        signed = [c for c in structural_groups["structural_archetype"] if "__signed_contribution" in c]
        signed += ["archetype_signed_total", "archetype_signed_max"]
        parts.extend(signed)
    parts = list(dict.fromkeys(c for c in parts if c not in {"canonical_expected_net_bps", "base_plus_consensus25"}))
    return ["canonical_expected_net_bps", "base_plus_consensus25", *parts]


def _metric_rows(pred: pd.DataFrame) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metrics: list[dict[str, object]] = []
    stability: list[dict[str, object]] = []
    for arm, block in pred.groupby("arm", sort=True):
        for tail in TAILS:
            row = _metric(block, "meta_score", tail)
            row.update({"arm": arm, "scope": "global_2025_long"})
            metrics.append(row)
        month_vals: list[float] = []
        for month, month_block in block.groupby("month", sort=True):
            row = _metric(month_block, "meta_score", 0.05)
            row.update({"arm": arm, "month": str(month), "scope": "monthly_2025_long"})
            metrics.append(row)
            month_vals.append(float(row["net_bps_per_trade"]))
        vals = np.asarray(month_vals, dtype=float)
        med = float(np.nanmedian(vals))
        stability.append({
            "arm": arm,
            "months": int(len(vals)),
            "mean_top5_net_bps": float(np.nanmean(vals)),
            "median_top5_net_bps": med,
            "mad_top5_net_bps": float(np.nanmedian(np.abs(vals - med))),
            "worst_month_top5_net_bps": float(np.nanmin(vals)),
            "positive_months_top5": int(np.sum(vals > 0.0)),
        })
    return metrics, stability


def run(*, output_dir: Path = DEFAULT_OUTPUT, heads_path: Path = DEFAULT_HEADS,
        structural_path: Path = DEFAULT_STRUCTURAL, contract_path: Path = DEFAULT_CONTRACT,
        selection_path: Path = DEFAULT_SELECTION) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    panel, context, context_hash = _load()
    heads = pd.read_parquet(heads_path)
    panel = panel.merge(heads, on="candidate_id", how="inner", validate="one_to_one", suffixes=("", "_head"))
    panel = panel.loc[panel.side_name.eq("long")].copy()
    structural, structural_groups, structural_audit = _load_structural(structural_path, contract_path)
    panel = panel.merge(structural, on="candidate_id", how="left", validate="one_to_one", suffixes=("", "_struct"))
    structural_cols = list(dict.fromkeys(c for values in structural_groups.values() for c in values))
    panel[structural_cols] = panel[structural_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    selected_doc = json.loads(selection_path.read_text())
    selected = {str(k): list(v) for k, v in selected_doc.get("selected_features", {}).items()}

    arms = _arms()
    parts: list[pd.DataFrame] = []
    fit_audit: list[dict[str, object]] = []
    feature_audit: list[dict[str, object]] = []
    dev_months = sorted(set(panel.month.astype(str)) & set(MONTHS))
    for month in dev_months:
        held = panel.loc[panel.month.astype(str).eq(month)].copy()
        cutoff = pd.Timestamp(month, tz="UTC")
        train = panel.loc[(panel["__ts__"] < cutoff) & (panel["label_available_ts"] < cutoff) & panel.month.astype(str).ne(month)].copy()
        if len(train) < 500 or held.empty:
            continue
        tr_expected, te_expected = _map_canonical(train, held)
        train["canonical_expected_net_bps"] = tr_expected
        held["canonical_expected_net_bps"] = te_expected
        train["meta_residual_bps"] = train.exact_net_bps.to_numpy(float) - tr_expected
        held["meta_residual_bps"] = held.exact_net_bps.to_numpy(float) - te_expected
        tr_model, te_model, feat_audit = _feature_frame(train, held, context)
        tr_struct = train.reindex(columns=structural_cols).fillna(0.0)
        te_struct = held.reindex(columns=structural_cols).fillna(0.0)
        grade = np.digitize(train.meta_residual_bps.to_numpy(float), [-150.0, -50.0, 50.0, 150.0]).astype(np.int32)
        for arm in arms:
            label = str(arm["arm"])
            if label == "control":
                rank = np.full(len(held), 0.5, dtype=np.float32)
                raw = np.zeros(len(held), dtype=np.float32)
                feature_names: list[str] = []
            else:
                feature_names = _names_for(arm, selected, structural_groups)
                xtr = pd.concat([tr_model, tr_struct], axis=1)
                xte = pd.concat([te_model, te_struct], axis=1)
                xtr["canonical_expected_net_bps"] = train.canonical_expected_net_bps.to_numpy(float)
                xte["canonical_expected_net_bps"] = held.canonical_expected_net_bps.to_numpy(float)
                xtr["base_plus_consensus25"] = train.base_plus_consensus25.to_numpy(float)
                xte["base_plus_consensus25"] = held.base_plus_consensus25.to_numpy(float)
                xtr = xtr.reindex(columns=feature_names).fillna(0.0)
                xte = xte.reindex(columns=feature_names).fillna(0.0)
                _, raw, rank, _ = _fit_meta(train, held, xtr, xte, grade, seed=SEED + int(month[-2:]) * 100 + len(feature_names))
            out = held[["candidate_id", "__ts__", "month", "side_name", "exact_net_bps", "exact_gross_bps", "base_plus_consensus25", "canonical_expected_net_bps", "meta_residual_bps"]].copy()
            out["arm"] = label
            out["meta_residual_rank"] = rank
            out["meta_score"] = 0.75 * out.base_plus_consensus25.to_numpy(float) + 0.25 * rank
            out["raw_meta_residual"] = raw
            out["feature_count"] = len(feature_names)
            parts.append(out)
            feature_audit.append({"month": month, "arm": label, "components": arm["components"], "feature_count": len(feature_names), "features": feature_names})
        fit_audit.append({"month": month, "train_rows": int(len(train)), "held_rows": int(len(held)), "recent_train_month": feat_audit["recent_train_month"], "structural_train_coverage": float(train[structural_cols].notna().any(axis=1).mean()), "structural_held_coverage": float(held[structural_cols].notna().any(axis=1).mean())})

    pred = pd.concat(parts, ignore_index=True)
    metrics, stability = _metric_rows(pred)
    metrics_frame = pd.DataFrame(metrics)
    stability_frame = pd.DataFrame(stability)
    global_top5 = metrics_frame.loc[metrics_frame.scope.eq("global_2025_long") & metrics_frame["tail"].eq(0.05)].copy()
    ranked = global_top5.merge(stability_frame, on="arm", how="left")
    ranked = ranked.sort_values(["net_bps_per_trade", "mean_top5_net_bps", "worst_month_top5_net_bps", "positive_months_top5"], ascending=[False, False, False, False], kind="stable")
    ranked["selection_rank"] = np.arange(1, len(ranked) + 1)
    top3 = ranked.head(3).copy()

    output_dir.mkdir(parents=True)
    pred.to_parquet(output_dir / "predictions.parquet", index=False, compression="zstd")
    metrics_frame.to_parquet(output_dir / "metrics.parquet", index=False)
    stability_frame.to_parquet(output_dir / "stability.parquet", index=False)
    ranked.to_parquet(output_dir / "selection_ranking.parquet", index=False)
    top3.to_parquet(output_dir / "top3_configs.parquet", index=False)
    pd.DataFrame(fit_audit).to_parquet(output_dir / "fit_audit.parquet", index=False)
    pd.DataFrame(feature_audit).to_parquet(output_dir / "feature_audit.parquet", index=False)
    manifest = {
        "schema": "tp6_sl4_component_combo_funnel_v1",
        "status": "COMPLETE",
        "side": "long",
        "months": list(MONTHS),
        "components": ["model_support_ood", "archetype_signed_exposure", "uncertainty", "compact_structural"],
        "arm_count": len(arms),
        "arms": arms,
        "selection_rule": "pooled global top-5 net bps/trade; stability mean, worst month, positive-month count as tie-breaks",
        "base_contract": "canonical TP6/SL4 Base+Consensus 75/25",
        "target": "exact_net_bps - train-only isotonic(canonical score)",
        "query": "4-hour UTC x side",
        "residual_grades": [-150.0, -50.0, 50.0, 150.0],
        "blend": "0.75 canonical score + 0.25 residual rank",
        "selected_features": selected,
        "structural_groups": structural_groups,
        "structural_audit": structural_audit,
        "canonical_context_sha256": context_hash,
        "top3": top3[["selection_rank", "arm", "net_bps_per_trade", "mean_top5_net_bps", "worst_month_top5_net_bps", "positive_months_top5"]].to_dict(orient="records"),
        "artifact_names": ["predictions.parquet", "metrics.parquet", "stability.parquet", "selection_ranking.parquet", "top3_configs.parquet", "fit_audit.parquet", "feature_audit.parquet", "run_manifest.json"],
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    report = ["# TP6/SL4 exact 16-way component-combination funnel", "", "The 16 arms enumerate every subset of model support/OOD, archetype signed exposure, uncertainty and compact structural inputs. All arms share the same rows, target, canonical anchor, query grouping and blend.", "", "## Selection ranking", "", ranked.round(3).to_string(index=False), "", "## Monthly stability", "", stability_frame.round(3).to_string(index=False), "", "## Top three frozen configs", "", top3.round(3).to_string(index=False), "", "## Contract", "", json.dumps(manifest, indent=2, default=str)]
    (output_dir / "TP6_SL4_COMPONENT_COMBO_FUNNEL_REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"output": str(output_dir), "rows": int(len(pred)), "arms": len(arms), "top3": manifest["top3"]}, indent=2, default=str))
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--heads", type=Path, default=DEFAULT_HEADS)
    parser.add_argument("--structural", type=Path, default=DEFAULT_STRUCTURAL)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    args = parser.parse_args()
    run(output_dir=args.output_dir, heads_path=args.heads, structural_path=args.structural, contract_path=args.contract, selection_path=args.selection)
