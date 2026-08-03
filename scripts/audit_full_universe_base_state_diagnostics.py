#!/usr/bin/env python3
"""Audit decision-time base-state diagnostics from frozen full-universe artifacts.

This is deliberately an audit/materialisation utility, not a training runner.  It
never reads realised PnL to construct an inference feature.  The only fitted
objects are robust feature-location/scale statistics taken from the frozen base
training interval.  It therefore separates what is genuinely available to a
future trust meta head from diagnostics that still need new prediction artifacts
(seed stability, raw-vs-calibrated score, and true payoff quantiles).

Default inputs are the side-local 200-tree TP3/SL2 base artifacts.  The output
covers their prediction rows (Apr--Nov 2024); callers can select the untouched
Aug--Nov interval with --start/--end.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = ROOT / "data_perp/artifacts/full_universe_t2_t4_panel_20260801_v3"
DEFAULT_BASE = ROOT / "data_perp/artifacts/full_universe_base_hpo_20260802_v1"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/full_universe_base_state_diagnostics_20260803_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    p.add_argument("--base", type=Path, default=DEFAULT_BASE)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--start", default=None, help="inclusive UTC timestamp")
    p.add_argument("--end", default=None, help="exclusive UTC timestamp")
    return p.parse_args()


def read_manifest(base: Path, side: str) -> dict:
    with (base / side / "target_family_manifest.json").open() as f:
        return json.load(f)


def selected_features(manifest: dict, side: str) -> list[str]:
    contract = manifest["feature_contract"]
    values = [v for k, v in contract.items() if k.endswith(f"|{side}")]
    if len(values) != 1:
        raise ValueError(f"expected one {side} feature contract, got {list(contract)}")
    return values[0]


def prediction_frame(base: Path) -> pd.DataFrame:
    frames = []
    for side in ("long", "short"):
        df = pd.read_parquet(base / side / "target_screen_predictions.parquet")
        df["__ts__"] = pd.to_datetime(df["__ts__"], utc=True)
        if set(df["side_name"].unique()) != {side}:
            raise ValueError(f"unexpected side rows in {side} prediction artifact")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    if out["candidate_id"].duplicated().any():
        raise ValueError("base prediction candidate IDs are not unique")
    return out


def panel_feature_frame(panel: Path, columns: list[str]) -> pd.DataFrame:
    files = sorted((panel / "parts").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no panel parts below {panel}")
    frames = []
    for path in files:
        frames.append(pd.read_parquet(path, columns=columns))
    out = pd.concat(frames, ignore_index=True)
    out["__ts__"] = pd.to_datetime(out["__ts__"], utc=True)
    if out["candidate_id"].duplicated().any():
        raise ValueError("panel candidate IDs are not unique")
    return out


def robust_reference(panel: pd.DataFrame, features: list[str], train_end: pd.Timestamp) -> tuple[pd.Series, pd.Series]:
    train = panel.loc[panel["__ts__"] < train_end, features]
    med = train.median(numeric_only=True)
    mad = (train - med).abs().median(numeric_only=True)
    # 1.4826*MAD is a normal-consistent robust scale.  A zero/absent scale is
    # deliberately left NaN, rather than creating a false OOD signal.
    scale = 1.4826 * mad.replace(0.0, np.nan)
    return med, scale


def expected_columns(manifest_long: dict, manifest_short: dict) -> dict[str, np.ndarray]:
    def means(manifest: dict) -> np.ndarray:
        vals = manifest["arms"][0]["details"]["conditional_net_means_bps"]
        if len(vals) != 3:
            raise ValueError("expected conditional upper/lower/timeout means")
        return np.asarray(vals, dtype=float)
    return {"long": means(manifest_long), "short": means(manifest_short)}


def diagnostics(df: pd.DataFrame, means_by_side: dict[str, np.ndarray], feature_sets: dict[str, list[str]], medians: dict[str, pd.Series], scales: dict[str, pd.Series]) -> pd.DataFrame:
    result = df[["candidate_id", "__ts__", "side_name", "gross_bps", "net_bps", "score_bps", "p_upper", "p_lower", "p_timeout"]].copy()
    probs = result[["p_upper", "p_lower", "p_timeout"]].to_numpy(dtype=float)
    if not np.allclose(probs.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5):
        raise ValueError("base probability simplex does not sum to one")
    clipped = np.clip(probs, 1e-12, 1.0)
    result["base_entropy"] = -(clipped * np.log(clipped)).sum(axis=1)
    result["base_entropy_normalised"] = result["base_entropy"] / np.log(3.0)
    result["base_simplex_hhi"] = (probs * probs).sum(axis=1)
    ordered = np.sort(probs, axis=1)
    result["base_top_probability"] = ordered[:, 2]
    result["base_top2_probability_margin"] = ordered[:, 2] - ordered[:, 1]
    result["base_upper_lower_margin"] = probs[:, 0] - probs[:, 1]
    result["base_upper_competitor_margin"] = probs[:, 0] - np.maximum(probs[:, 1], probs[:, 2])

    rebuilt = np.empty(len(result), dtype=float)
    mixture_sd = np.empty(len(result), dtype=float)
    missing_count = np.empty(len(result), dtype=np.int16)
    robust_ood_l1 = np.empty(len(result), dtype=float)
    robust_ood_max = np.empty(len(result), dtype=float)
    for side in ("long", "short"):
        mask = result["side_name"].eq(side).to_numpy()
        p = probs[mask]
        mus = means_by_side[side]
        mean = p @ mus
        rebuilt[mask] = mean
        mixture_sd[mask] = np.sqrt((p * (mus[None, :] - mean[:, None]) ** 2).sum(axis=1))
        features = feature_sets[side]
        x = df.loc[mask, features]
        finite = np.isfinite(x.to_numpy(dtype=float))
        missing_count[mask] = (~finite).sum(axis=1)
        z = (x - medians[side]) / scales[side]
        # A diagonal robust distance is intentionally transparent.  It is not
        # represented as a model-leaf OOD score, because base tree leaves and
        # training embeddings were not saved in the frozen artifact.
        abs_z = np.abs(z.to_numpy(dtype=float))
        robust_ood_l1[mask] = np.nanmean(np.minimum(abs_z, 20.0), axis=1)
        robust_ood_max[mask] = np.nanmax(np.minimum(abs_z, 20.0), axis=1)

    result["base_expected_net_bps_rebuilt"] = rebuilt
    result["base_expected_gross_bps_rebuilt"] = rebuilt + 100.0
    result["base_cost_margin_bps"] = rebuilt
    result["base_score_rebuild_difference_bps"] = result["score_bps"] - rebuilt
    result["base_conditional_payoff_mixture_sd_bps"] = mixture_sd
    result["base_feature_missing_count"] = missing_count
    result["base_feature_missing_fraction"] = missing_count / 36.0
    result["base_feature_ood_robust_l1"] = robust_ood_l1
    result["base_feature_ood_robust_max"] = robust_ood_max
    return result


def availability(manifest_long: dict, manifest_short: dict, out: pd.DataFrame) -> dict:
    return {
        "source_contract": {
            "base_artifact": "side-local 200-tree TP3/SL2 T2 tau=0.25 predictions",
            "entry_exit": manifest_long["entry"] + "; " + manifest_long["exit"],
            "base_train_window": manifest_long["train_window"],
            "conditional_payoffs": {
                "long_net_bps": manifest_long["arms"][0]["details"]["conditional_net_means_bps"],
                "short_net_bps": manifest_short["arms"][0]["details"]["conditional_net_means_bps"],
            },
            "cost_bps": 100.0,
        },
        "materialised_now": {
            "entropy_margin_concentration": ["base_entropy_normalised", "base_simplex_hhi", "base_top2_probability_margin", "base_upper_lower_margin", "base_upper_competitor_margin"],
            "fixed_payoff_expected_value_and_cost_margin": ["base_expected_gross_bps_rebuilt", "base_expected_net_bps_rebuilt", "base_cost_margin_bps"],
            "conditional_payoff_uncertainty_proxy": "base_conditional_payoff_mixture_sd_bps; dispersion of fixed training-only event payoff means under the predicted simplex, not conditional payoff quantiles",
            "missingness": ["base_feature_missing_count", "base_feature_missing_fraction"],
            "ood": "base_feature_ood_robust_l1/max: train-window robust diagonal distance over the exact 36 frozen side-specific base features",
            "score_mapping_integrity": "base_score_rebuild_difference_bps; verifies the stored score is exactly the fixed event-payoff conversion",
        },
        "not_materialisable_from_current_frozen_artifacts": {
            "seed_prediction_stability": "No same-contract multi-seed base predictions were saved.",
            "raw_vs_calibrated_difference": "Only the already event-payoff-mapped score_bps is stored. No distinct raw-score or post-hoc calibrator output exists.",
            "true_context_conditional_payoff_uncertainty": "No per-row conditional payoff quantiles/distribution model was fitted or saved; mixture SD is only a fixed-mean proxy.",
            "tree_leaf_ood_or_training_embedding_distance": "Frozen serialized tree leaves / training embeddings were not retained. Robust feature OOD is a causal substitute, not an equivalent diagnostic.",
            "teacher_student_disagreement": "The distillation experiment output is not the frozen base teacher state and must not be silently joined as one.",
            "prequential_score_percentile": "The base artifact begins after its training window; a score-history seed or historical OOF predictions are required for a fully prequential initial percentile.",
        },
        "geometry_stability": {
            "status": "partially available, not materialised into this production-contract ledger",
            "available_source": "full_universe_t2_t4_target_screen_20260801_v1 stores all four geometry predictions for Apr--Jul 2024, trained Apr 2023--Apr 2024 at the older 80-tree capacity.",
            "limitation": "Those sibling geometry predictions do not match the selected 200-tree base capacity and do not cover the Aug--Nov untouched OOS interval. They are suitable only as a development diagnostic. A valid meta feature requires frozen sibling TP/SL predictions from the same fit/capacity on the deployed period.",
        },
        "rows": int(len(out)),
        "coverage": {
            "feature_missing_fraction_le_10pct": float((out["base_feature_missing_fraction"] <= 0.10).mean()),
            "exact_score_rebuild_fraction": float((out["base_score_rebuild_difference_bps"].abs() <= 1e-4).mean()),
        },
    }


def report_text(audit: dict, out: pd.DataFrame) -> str:
    missing = out["base_feature_missing_fraction"]
    lines = [
        "# Full-universe base-state diagnostic availability audit",
        "",
        "This audit is a decision-time feature lineage check for the frozen TP3/SL2, tau=0.25 base. It does not use realised PnL to produce any diagnostic.",
        "",
        "## Availability matrix",
        "",
        "| Requested diagnostic | Status | Exact interpretation |",
        "|---|---|---|",
        "| Entropy, margins, concentration | available | Exact functions of the stored three-state simplex. |",
        "| Cost margin / expected gross-net | available | Fixed training-only side-local event-payoff mapping, with declared 100 bps cost. |",
        "| Conditional-payoff uncertainty | proxy only | Mixture standard deviation of the three fixed event means; not a conditional payoff interval. |",
        "| Feature missingness | available | Missing/non-finite fraction of the frozen 36 selected side-local base features. |",
        "| OOD | available proxy | Robust diagonal feature distance fit on the frozen training interval; not leaf/embedding OOD. |",
        "| Raw-versus-calibrated difference | unavailable | Only the mapped `score_bps` is saved; no separate raw/calibrated pair exists. |",
        "| Seed stability | unavailable | No identical-fit multi-seed prediction ledger exists. |",
        "| Geometry stability | development-only | 80-tree sibling-geometry predictions exist through Jul-2024, not matching 200-tree OOS predictions. |",
        "",
        "## Materialised now",
        "",
        "- Probability simplex entropy, HHI concentration, upper/lower margin, upper-versus-best-competitor margin, and top-two margin are exact functions of the stored `p_upper`, `p_lower`, `p_timeout`.",
        "- Rebuilt expected gross/net and cost margin use the side-local training-only conditional event means retained in each manifest. `base_score_rebuild_difference_bps` is an integrity check, not a new calibration.",
        "- `base_conditional_payoff_mixture_sd_bps` is an economically useful but limited uncertainty proxy: it measures uncertainty arising from the predicted event mix under fixed payoff means. It is not q75 minus q25 of conditional realised payoff.",
        "- Missingness is measured over the exact 36 side-local selected base features. Robust OOD uses only those decision-time values and robust location/scale fit on the frozen Apr-2023--Apr-2024 train interval.",
        "",
        "## Coverage",
        "",
        f"- Rows materialised: {len(out):,}.",
        f"- Rows with at most 10% missing selected base features: {(missing <= .10).mean():.2%}.",
        f"- Median selected-feature missing fraction: {missing.median():.2%}.",
        f"- Exact fixed-payoff score rebuild: {audit['coverage']['exact_score_rebuild_fraction']:.2%} of rows within 1e-4 bps.",
        "",
        "## Explicit gaps",
        "",
        "- There is no multi-seed prediction ledger, so seed stability cannot be computed honestly.",
        "- There is no separately stored raw score/calibrated score pair. The stored score is the fixed event-payoff mapping; a zero difference from its rebuild proves identity, not calibration quality.",
        "- There are no conditional payoff quantile models or saved leaf/embedding artifacts. Do not label the mixture SD as a full payoff-uncertainty or leaf-OOD measure.",
        "- Four older geometry predictions exist only for Apr--Jul and 80-tree models. They must not be used as an OOS feature for the 200-tree selected base without matching sibling predictions.",
        "",
        "## Causal next materialisation",
        "",
        "1. Save three-to-five seed predictions for the identical base fit and compute per-row standard deviation/rank agreement.",
        "2. For each 200-tree base fold, save raw probability-to-score mapping inputs, the prequential calibration prediction, and the final score separately.",
        "3. Fit payoff q25/q50/q75 or a two-part failure-severity model using only prior-resolved rows, then add its prediction interval rather than using the fixed-mean proxy.",
        "4. Retrain/snapshot all four TP/SL sibling geometries with the identical feature contract and fit window; calculate rank correlation and score spread only from simultaneously available predictions.",
        "5. If tree-leaf OOD is required, persist train-leaf occupancy/counts with the fitted model. The robust feature OOD in this ledger remains valid in the meantime.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    manifests = {side: read_manifest(args.base, side) for side in ("long", "short")}
    feature_sets = {side: selected_features(manifests[side], side) for side in ("long", "short")}
    if any(len(v) != 36 for v in feature_sets.values()):
        raise ValueError(f"expected 36 selected features per side, got { {k: len(v) for k, v in feature_sets.items()} }")
    preds = prediction_frame(args.base)
    if args.start:
        preds = preds.loc[preds["__ts__"] >= pd.Timestamp(args.start, tz="UTC")]
    if args.end:
        preds = preds.loc[preds["__ts__"] < pd.Timestamp(args.end, tz="UTC")]
    all_features = sorted(set(feature_sets["long"]) | set(feature_sets["short"]))
    panel_cols = ["candidate_id", "__ts__"] + all_features
    panel = panel_feature_frame(args.panel, panel_cols)
    merged = preds.merge(panel, on=["candidate_id", "__ts__"], how="left", validate="one_to_one", indicator=True)
    if not (merged["_merge"] == "both").all():
        raise ValueError(f"unmatched predictions: {(merged['_merge'] != 'both').sum()}")
    merged = merged.drop(columns="_merge")
    train_end = pd.Timestamp(manifests["long"]["train_window"][1], tz="UTC")
    medians, scales = {}, {}
    for side in ("long", "short"):
        medians[side], scales[side] = robust_reference(panel, feature_sets[side], train_end)
    out = diagnostics(merged, expected_columns(manifests["long"], manifests["short"]), feature_sets, medians, scales)
    audit = availability(manifests["long"], manifests["short"], out)
    args.output.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output / "base_state_diagnostics.parquet", index=False)
    (args.output / "availability_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (args.output / "BASE_STATE_DIAGNOSTIC_AUDIT.md").write_text(report_text(audit, out))
    print(json.dumps({"output": str(args.output), "rows": len(out), "coverage": audit["coverage"]}, indent=2))


if __name__ == "__main__":
    main()
