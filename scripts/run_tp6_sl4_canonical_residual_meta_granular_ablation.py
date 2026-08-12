#!/usr/bin/env python3
"""Granular long-only residual-meta ablation with structural exposure inputs.

This runner is deliberately separate from the promoted canonical residual
runner.  It keeps the frozen TP6/SL4 Base+Consensus contract, the train-only
isotonic expected-net anchor, ordinal residual target and 4h x side queries,
then compares model-layer groups with a frozen long-side structural
archetype/cluster exposure contract.

Structural inputs are path/exposure descriptors only.  They contain no
outcomes, GAM corrections, or realised labels.  Archetype IDs are read from
the frozen recurrent contract and cluster memberships are deterministic sums
of archetype exposures.  The full structural feature matrix is retained for
audit, while compact structural summaries are the primary ablation input.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_tp6_sl4_downstream_retrain_2025 import MONTHS, _load, _pct  # noqa: E402
from scripts.run_tp6_sl4_canonical_residual_meta_block_ablation import (  # noqa: E402
    BLOCKS as MODEL_BLOCKS,
    DEFAULT_HEADS,
    _feature_frame,
    _fit_meta,
    _map_canonical,
)

DEFAULT_HEADS = ROOT / "data_perp/artifacts/tp6_sl4_canonical_head_health_2025_v1/canonical_head_health_2025.parquet"
DEFAULT_STRUCTURAL = ROOT / "data_perp/artifacts/tp6_sl4_archetype_cluster_vcgam_oof_20260814_v3/archetype_row_exposures.parquet"
DEFAULT_CONTRACT = ROOT / "data_perp/artifacts/tp6_sl4_archetype_cluster_vcgam_oof_20260814_v3/cofiring_cluster_contract.json"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/tp6_sl4_canonical_residual_meta_granular_long_20260808_v1"
DEFAULT_SELECTION = ROOT / "data_perp/artifacts/tp6_sl4_canonical_residual_feature_selection_20260808_v1/selected_features.json"
SEED = 20260808
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)


MODEL_GROUPS = {
    "model_uncertainty": ("uncertainty",),
    "model_support_ood": ("support_ood",),
    "model_drift": ("drift",),
    "model_market_state": ("market_state",),
    "model_uncertainty_support": ("uncertainty", "support_ood"),
    "model_all": ("uncertainty", "support_ood", "drift", "market_state"),
}

# Finer, predeclared splits used in the second granular pass.  These are
# subsets of the selected model-layer fields, not post-test feature picks.
MODEL_SUBGROUPS = {
    "model_head_dispersion": ("uncertainty",),
    "model_probability": ("uncertainty",),
    "model_market_liquidity": ("market_state",),
    "model_market_regime": ("market_state",),
}


def _safe(frame: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = frame.reindex(columns=list(cols)).apply(pd.to_numeric, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def _entropy(values: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(values, dtype=float), 0.0, None)
    den = x.sum(axis=1, keepdims=True)
    p = np.divide(x, den, out=np.zeros_like(x), where=den > 1e-12)
    return -np.sum(np.where(p > 1e-12, p * np.log(np.maximum(p, 1e-12)), 0.0), axis=1)


def _top2_margin(values: np.ndarray) -> np.ndarray:
    x = np.sort(np.asarray(values, dtype=float), axis=1)
    if x.shape[1] < 2:
        return np.zeros(x.shape[0], dtype=float)
    return x[:, -1] - x[:, -2]


def _load_structural(path: Path, contract_path: Path) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, object]]:
    raw = pd.read_parquet(path)
    raw = raw.drop_duplicates("candidate_id", keep="last")
    arch_abs = sorted(c for c in raw.columns if c.startswith("archetype__") and c.endswith("__abs_contribution"))
    arch_signed = sorted(c for c in raw.columns if c.startswith("archetype__") and c.endswith("__signed_contribution"))
    arch_active = sorted(c for c in raw.columns if c.startswith("archetype__") and c.endswith("__active"))
    if not arch_abs or not arch_signed:
        raise RuntimeError("structural exposure artifact has no archetype exposures")
    contract = json.loads(contract_path.read_text())
    clusters = contract.get("clusters", [])
    cluster_map: dict[str, list[str]] = {}
    for cluster in clusters:
        cid = str(cluster["cluster_id"])
        cluster_map[cid] = [str(v) for v in cluster.get("family_fields", [])]

    # Stable, compact row-level structural descriptors.
    out = raw[["candidate_id", "__ts__"]].copy()
    for c in ["archetype_matched_mass", "archetype_unmatched_mass", "archetype_entropy", "archetype_top2_margin"]:
        out[c] = pd.to_numeric(raw[c], errors="coerce") if c in raw else 0.0
    abs_v = _safe(raw, arch_abs).fillna(0.0).to_numpy(float)
    signed_v = _safe(raw, arch_signed).fillna(0.0).to_numpy(float)
    active_v = _safe(raw, arch_active).fillna(0.0).to_numpy(float)
    out["archetype_abs_total"] = abs_v.sum(axis=1)
    out["archetype_signed_total"] = signed_v.sum(axis=1)
    out["archetype_abs_max"] = abs_v.max(axis=1)
    out["archetype_signed_max"] = signed_v.max(axis=1)
    out["archetype_active_count"] = active_v.sum(axis=1)
    out["archetype_active_mass"] = abs_v[active_v > 0].sum(axis=1) if False else (abs_v * active_v).sum(axis=1)
    out["archetype_abs_entropy"] = _entropy(abs_v)
    out["archetype_abs_top2_margin"] = _top2_margin(abs_v)

    # Preserve the individual archetype fields for the full structural arm.
    arch_features: list[str] = []
    for c in arch_abs + arch_signed + arch_active:
        name = "structural__" + c.replace("archetype__", "")
        out[name] = _safe(raw, [c]).fillna(0.0).iloc[:, 0].to_numpy(float)
        arch_features.append(name)

    cluster_abs: list[str] = []
    cluster_signed: list[str] = []
    cluster_active: list[str] = []
    for cid, families in cluster_map.items():
        idx = []
        for family in families:
            token = f"archetype__{family}__"
            idx.extend(i for i, c in enumerate(arch_abs) if token in c)
        # Family names are archetype_000N and the exposure ordering is stable.
        sidx = []
        for family in families:
            token = f"archetype__{family}__"
            sidx.extend(i for i, c in enumerate(arch_signed) if token in c)
        aidx = []
        for family in families:
            token = f"archetype__{family}__"
            aidx.extend(i for i, c in enumerate(arch_active) if token in c)
        ca = f"structural__cluster__{cid}__abs_exposure"
        cs = f"structural__cluster__{cid}__signed_exposure"
        ct = f"structural__cluster__{cid}__active_mass"
        out[ca] = abs_v[:, idx].sum(axis=1) if idx else 0.0
        out[cs] = signed_v[:, sidx].sum(axis=1) if sidx else 0.0
        out[ct] = (abs_v[:, idx] * active_v[:, aidx]).sum(axis=1) if idx and len(aidx) == len(idx) else 0.0
        cluster_abs.append(ca); cluster_signed.append(cs); cluster_active.append(ct)
    cluster_values = _safe(out, cluster_abs).fillna(0.0).to_numpy(float)
    out["structural__cluster__active_count"] = (cluster_values > 0.0).sum(axis=1)
    out["structural__cluster__abs_entropy"] = _entropy(cluster_values)
    out["structural__cluster__abs_top2_margin"] = _top2_margin(cluster_values)
    out["structural__cluster__abs_total"] = cluster_values.sum(axis=1)
    out["structural__cluster__abs_max"] = cluster_values.max(axis=1)
    out["structural__cluster__signed_total"] = _safe(out, cluster_signed).fillna(0.0).sum(axis=1).to_numpy(float)

    groups = {
        "structural_transport": ["archetype_matched_mass", "archetype_unmatched_mass", "archetype_entropy", "archetype_top2_margin", "archetype_abs_entropy", "archetype_abs_top2_margin", "structural__cluster__abs_entropy", "structural__cluster__abs_top2_margin"],
        "structural_archetype": ["archetype_matched_mass", "archetype_unmatched_mass", "archetype_entropy", "archetype_top2_margin", "archetype_abs_total", "archetype_signed_total", "archetype_abs_max", "archetype_signed_max", "archetype_active_count", "archetype_active_mass", "archetype_abs_entropy", "archetype_abs_top2_margin", *arch_features],
        "structural_cluster": [*cluster_abs, *cluster_signed, *cluster_active, "structural__cluster__active_count", "structural__cluster__abs_entropy", "structural__cluster__abs_top2_margin", "structural__cluster__abs_total", "structural__cluster__abs_max", "structural__cluster__signed_total"],
        "structural_compact": ["archetype_matched_mass", "archetype_unmatched_mass", "archetype_entropy", "archetype_top2_margin", "archetype_abs_total", "archetype_signed_total", "archetype_abs_max", "archetype_signed_max", "archetype_active_count", "archetype_active_mass", "archetype_abs_entropy", "archetype_abs_top2_margin", "structural__cluster__active_count", "structural__cluster__abs_entropy", "structural__cluster__abs_top2_margin", "structural__cluster__abs_total", "structural__cluster__abs_max", "structural__cluster__signed_total"],
        "structural_full": [*arch_features, *cluster_abs, *cluster_signed, *cluster_active, "archetype_matched_mass", "archetype_unmatched_mass", "archetype_entropy", "archetype_top2_margin", "archetype_abs_total", "archetype_signed_total", "archetype_abs_max", "archetype_signed_max", "archetype_active_count", "archetype_active_mass", "archetype_abs_entropy", "archetype_abs_top2_margin", "structural__cluster__active_count", "structural__cluster__abs_entropy", "structural__cluster__abs_top2_margin", "structural__cluster__abs_total", "structural__cluster__abs_max", "structural__cluster__signed_total"],
    }
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    audit = {"rows": int(len(out)), "archetype_abs_fields": len(arch_abs), "archetype_signed_fields": len(arch_signed), "archetype_active_fields": len(arch_active), "clusters": len(cluster_map), "cluster_map": cluster_map}
    return out, groups, audit


def _metric(frame: pd.DataFrame, score: str, tail: float) -> dict[str, object]:
    n = max(1, int(math.ceil(len(frame) * tail)))
    top = frame.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable").head(n)
    return {"tail": float(tail), "trades": int(len(top)), "gross_bps_per_trade": float(top.exact_gross_bps.mean()), "net_bps_per_trade": float(top.exact_net_bps.mean()), "rank_ic": float(frame[[score, "exact_net_bps"]].corr(method="spearman").iloc[0, 1])}


def _feature_names(arm: str, selected: dict[str, list[str]], structural_groups: dict[str, list[str]]) -> list[str]:
    parts: list[str] = []
    if arm in MODEL_GROUPS:
        for b in MODEL_GROUPS[arm]:
            parts.extend(selected.get(b, MODEL_BLOCKS[b]))
    elif arm == "model_all_plus_structural":
        for b in MODEL_GROUPS["model_all"]:
            parts.extend(selected.get(b, MODEL_BLOCKS[b]))
        parts.extend(structural_groups["structural_compact"])
    elif arm == "model_uncertainty_plus_structural":
        parts.extend(selected.get("uncertainty", MODEL_BLOCKS["uncertainty"]))
        parts.extend(structural_groups["structural_compact"])
    elif arm == "structural_transport":
        parts.extend(structural_groups["structural_transport"])
    elif arm == "structural_archetype":
        parts.extend(structural_groups["structural_archetype"])
    elif arm == "structural_cluster":
        parts.extend(structural_groups["structural_cluster"])
    elif arm == "structural_compact":
        parts.extend(structural_groups["structural_compact"])
    elif arm == "structural_full":
        parts.extend(structural_groups["structural_full"])
    elif arm == "model_all_plus_structural_full":
        for b in MODEL_GROUPS["model_all"]:
            parts.extend(selected.get(b, MODEL_BLOCKS[b]))
        parts.extend(structural_groups["structural_full"])
    elif arm == "model_head_dispersion":
        values = selected.get("uncertainty", MODEL_BLOCKS["uncertainty"])
        parts.extend([c for c in values if c.startswith("consensus_head_") or c == "base_consensus_disagreement"])
    elif arm == "model_probability":
        values = selected.get("uncertainty", MODEL_BLOCKS["uncertainty"])
        parts.extend([c for c in values if c.startswith("r3_meta_") or c in {"base_conviction", "base_probability_entropy", "base_probability_top2_margin"}])
    elif arm == "model_market_liquidity":
        values = selected.get("market_state", MODEL_BLOCKS["market_state"])
        tokens = ("spread", "depth", "amihud", "liquidity", "volume", "oiw")
        parts.extend([c for c in values if any(t in c for t in tokens)])
    elif arm == "model_market_regime":
        values = selected.get("market_state", MODEL_BLOCKS["market_state"])
        tokens = ("spread", "depth", "amihud", "liquidity", "volume", "oiw")
        parts.extend([c for c in values if not any(t in c for t in tokens)])
    elif arm == "structural_archetype_abs":
        parts.extend([c for c in structural_groups["structural_archetype"] if "__signed_contribution" not in c and "__active" not in c])
    elif arm == "structural_archetype_signed":
        parts.extend([c for c in structural_groups["structural_archetype"] if "__abs_contribution" not in c and "__active" not in c])
    elif arm == "structural_archetype_active":
        parts.extend([c for c in structural_groups["structural_archetype"] if "__active" in c or c in {"archetype_matched_mass", "archetype_unmatched_mass", "archetype_entropy", "archetype_top2_margin"}])
    elif arm == "structural_cluster_abs":
        parts.extend([c for c in structural_groups["structural_cluster"] if "__abs_" in c or "__abs_exposure" in c or "__active_count" in c or "__entropy" in c or "__top2_margin" in c])
    elif arm == "structural_cluster_signed":
        parts.extend([c for c in structural_groups["structural_cluster"] if "__signed_" in c])
    elif arm == "structural_cluster_active":
        parts.extend([c for c in structural_groups["structural_cluster"] if "__active_" in c or c == "structural__cluster__active_count"])
    # Anchor features are appended once for every non-control arm.
    dedup = list(dict.fromkeys(c for c in parts if c not in {"canonical_expected_net_bps", "base_plus_consensus25"}))
    return ["canonical_expected_net_bps", "base_plus_consensus25", *dedup]


ARMS = [
    "A_control",
    "model_uncertainty", "model_support_ood", "model_drift", "model_market_state", "model_uncertainty_support", "model_all",
    "structural_transport", "structural_archetype", "structural_cluster", "structural_compact", "structural_full",
    "model_uncertainty_plus_structural", "model_all_plus_structural", "model_all_plus_structural_full",
    "model_head_dispersion", "model_probability", "model_market_liquidity", "model_market_regime",
    "structural_archetype_abs", "structural_archetype_signed", "structural_archetype_active",
    "structural_cluster_abs", "structural_cluster_signed", "structural_cluster_active",
]


def run(*, output_dir: Path = DEFAULT_OUTPUT, heads_path: Path = DEFAULT_HEADS, structural_path: Path = DEFAULT_STRUCTURAL, contract_path: Path = DEFAULT_CONTRACT, selection_path: Path = DEFAULT_SELECTION) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    panel, context, context_hash = _load()
    heads = pd.read_parquet(heads_path)
    panel = panel.merge(heads, on="candidate_id", how="inner", validate="one_to_one", suffixes=("", "_head"))
    # Retain the full historical long-side substrate for chronological fits;
    # only the held-month selection below is restricted to MONTHS.
    panel = panel.loc[panel.side_name.eq("long")].copy()
    structural, structural_groups, structural_audit = _load_structural(structural_path, contract_path)
    panel = panel.merge(structural, on="candidate_id", how="left", validate="one_to_one", suffixes=("", "_struct"))
    structural_cols = [c for values in structural_groups.values() for c in values]
    structural_cols = list(dict.fromkeys(structural_cols))
    coverage = float(panel[structural_cols].notna().any(axis=1).mean())
    panel[structural_cols] = panel[structural_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    selected_doc = json.loads(selection_path.read_text())
    selected = {str(k): list(v) for k, v in selected_doc.get("selected_features", {}).items()}
    parts: list[pd.DataFrame] = []
    fit_audit: list[dict[str, object]] = []
    feature_audit: list[dict[str, object]] = []
    dev_months = sorted(set(panel.month.astype(str)) & set(MONTHS))
    for month in dev_months:
        held = panel.loc[panel.month.astype(str).eq(month)].copy()
        train = panel.loc[(panel.__ts__ < pd.Timestamp(month, tz="UTC")) & (panel.label_available_ts < pd.Timestamp(month, tz="UTC")) & panel.month.astype(str).ne(month)].copy()
        if len(train) < 500 or held.empty:
            continue
        tr_expected, te_expected = _map_canonical(train, held)
        train["canonical_expected_net_bps"] = tr_expected
        held["canonical_expected_net_bps"] = te_expected
        train["meta_residual_bps"] = train.exact_net_bps.to_numpy(float) - tr_expected
        held["meta_residual_bps"] = held.exact_net_bps.to_numpy(float) - te_expected
        tr_model, te_model, feat_audit = _feature_frame(train, held, context)
        # Align structural features without allowing missing path rows to be
        # silently confused with an archetype.  The coverage itself is audited.
        tr_struct = train.reindex(columns=structural_cols).fillna(0.0)
        te_struct = held.reindex(columns=structural_cols).fillna(0.0)
        for arm in ARMS:
            if arm == "A_control":
                meta_rank = np.full(len(held), 0.5, dtype=np.float32)
                raw = np.zeros(len(held), dtype=np.float32)
                feature_names: list[str] = []
            else:
                feature_names = _feature_names(arm, selected, structural_groups)
                xtr = pd.concat([tr_model, tr_struct], axis=1)
                xte = pd.concat([te_model, te_struct], axis=1)
                xtr["canonical_expected_net_bps"] = train.canonical_expected_net_bps.to_numpy(float)
                xte["canonical_expected_net_bps"] = held.canonical_expected_net_bps.to_numpy(float)
                xtr["base_plus_consensus25"] = train.base_plus_consensus25.to_numpy(float)
                xte["base_plus_consensus25"] = held.base_plus_consensus25.to_numpy(float)
                xtr = xtr.reindex(columns=feature_names).fillna(0.0)
                xte = xte.reindex(columns=feature_names).fillna(0.0)
                grade = np.digitize(train.meta_residual_bps.to_numpy(float), [-150.0, -50.0, 50.0, 150.0]).astype(np.int32)
                _, raw, meta_rank, _ = _fit_meta(train, held, xtr, xte, grade, seed=SEED + int(month[-2:]) * 100 + len(feature_names))
            out = held[["candidate_id", "__ts__", "month", "side_name", "exact_net_bps", "exact_gross_bps", "base_plus_consensus25", "canonical_expected_net_bps", "meta_residual_bps"]].copy()
            out["arm"] = arm
            out["meta_residual_rank"] = meta_rank
            out["meta_score"] = 0.75 * out.base_plus_consensus25.to_numpy(float) + 0.25 * meta_rank
            out["raw_meta_residual"] = raw
            out["feature_count"] = len(feature_names)
            parts.append(out)
            feature_audit.append({"month": month, "arm": arm, "feature_count": len(feature_names), "features": feature_names})
        fit_audit.append({"month": month, "train_rows": int(len(train)), "held_rows": int(len(held)), "recent_train_month": feat_audit["recent_train_month"], "structural_train_coverage": float(train[structural_cols].notna().any(axis=1).mean()), "structural_held_coverage": float(held[structural_cols].notna().any(axis=1).mean())})
    pred = pd.concat(parts, ignore_index=True)
    metrics: list[dict[str, object]] = []
    stability: list[dict[str, object]] = []
    for arm, block in pred.groupby("arm", sort=True):
        for tail in TAILS:
            row = _metric(block, "meta_score", tail); row.update({"arm": arm, "scope": "global_2025_long"}); metrics.append(row)
        vals = np.asarray([_metric(m, "meta_score", 0.05)["net_bps_per_trade"] for _, m in block.groupby("month", sort=True)], dtype=float)
        med = float(np.nanmedian(vals))
        stability.append({"arm": arm, "months": int(len(vals)), "mean_top5_net_bps": float(np.nanmean(vals)), "median_top5_net_bps": med, "mad_top5_net_bps": float(np.nanmedian(np.abs(vals-med))), "worst_month_top5_net_bps": float(np.nanmin(vals)), "positive_months_top5": int(np.sum(vals > 0.0))})
        for month, m in block.groupby("month", sort=True):
            row = _metric(m, "meta_score", 0.05); row.update({"arm": arm, "month": month, "scope": "monthly_2025_long"}); metrics.append(row)
    output_dir.mkdir(parents=True)
    pred.to_parquet(output_dir / "predictions.parquet", index=False, compression="zstd")
    pd.DataFrame(metrics).to_parquet(output_dir / "metrics.parquet", index=False)
    pd.DataFrame(stability).to_parquet(output_dir / "stability.parquet", index=False)
    pd.DataFrame(fit_audit).to_parquet(output_dir / "fit_audit.parquet", index=False)
    pd.DataFrame(feature_audit).to_parquet(output_dir / "feature_audit.parquet", index=False)
    contract = {"schema": "tp6_sl4_canonical_residual_meta_granular_ablation_v1", "status": "COMPLETE", "side": "long", "months": list(MONTHS), "base_contract": "canonical TP6/SL4 Base+Consensus 75/25", "target": "exact_net_bps - train-only isotonic(CanonicalScore)", "query": "4-hour UTC x side", "residual_grades": [-150.0, -50.0, 50.0, 150.0], "blend": "0.75 canonical score + 0.25 residual rank", "model_blocks": MODEL_BLOCKS, "model_selected_features": selected, "structural_groups": structural_groups, "structural_audit": structural_audit, "structural_coverage_2025_long": coverage, "arms": ARMS, "structural_source": str(structural_path), "structural_contract": str(contract_path), "selection_source": str(selection_path), "canonical_context_sha256": context_hash, "structural_is_outcome_free": True, "artifacts": ["predictions.parquet", "metrics.parquet", "stability.parquet", "fit_audit.parquet", "feature_audit.parquet", "run_manifest.json"]}
    (output_dir / "run_manifest.json").write_text(json.dumps(contract, indent=2, default=str) + "\n")
    lines = ["# TP6/SL4 canonical residual meta granular long-only ablation", "", "All arms use the frozen Base+Consensus control, train-only isotonic expected-net anchor, ordinal residual grades and 4h x side LambdaRank. Structural fields are frozen path/archetype/cluster exposures without outcomes.", "", "## Global and monthly metrics", "", pd.DataFrame(metrics).round(3).to_string(index=False), "", "## Stability", "", pd.DataFrame(stability).round(3).to_string(index=False), "", "## Contract", "", json.dumps(contract, indent=2, default=str)]
    (output_dir / "TP6_SL4_CANONICAL_RESIDUAL_META_GRANULAR_LONG_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(output_dir), "rows": int(len(pred)), "arms": ARMS, "structural_coverage": coverage}, indent=2))
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
