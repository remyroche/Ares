#!/usr/bin/env python3
"""Synthesize bad-regime archetype candidates from recent-failure diagnostics.

This is a lightweight bridge between the diagnostic tools and deployable regime
features.  It does not train production models.  It groups evidence from:

* high-confidence failure classifiers,
* adversarial bad-week classifiers,
* leaf x archetype/residual interaction diagnostics,

into recurring mechanism channels and proposes continuous, causal feature names
that can later be generated cross-fitted for the meta models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FORBIDDEN_TOKENS = (
    "barrier_pct",
    "leaf_target",
    "rank_bin_",
    "rank_bin_win_rate",
    "rank_bin_lift",
    "rank_bin_net_ret",
    "rank_bin_se",
    "net_ret_oof",
    "policy_result",
    "post_trade",
    "diag_",
    "future",
    "target",
)


MECHANISM_SPECS: dict[str, dict[str, Any]] = {
    "model_path_fragility": {
        "tokens": (
            "score_path",
            "score_early",
            "rank_100_minus_50",
            "rank_path",
            "support_gap",
            "leaf_count",
            "dae",
            "gmm",
            "cluster_entropy",
            "reconstruction",
            "mahal",
            "regime_centroid",
            "centroid_similarity",
        ),
        "deployable_features": (
            "model_path_fragility_score",
            "model_support_gap_score",
            "model_reconstruction_anomaly_score",
        ),
        "model_layer": "meta",
    },
    "oi_funding_crowding": {
        "tokens": (
            "oi_",
            "_oi",
            "open_interest",
            "fund",
            "funding",
            "leverage",
            "oi_to_volume",
            "price_x_oi",
        ),
        "deployable_features": (
            "oi_funding_crowding_score",
            "oi_to_volume_pressure_score",
            "funding_crowding_tail_score",
        ),
        "model_layer": "meta",
    },
    "liquidity_spread_stress": {
        "tokens": (
            "spread",
            "liquid",
            "liquidity",
            "amihud",
            "depth",
            "volume",
            "rvol",
            "turnover",
        ),
        "deployable_features": (
            "liquidity_stress_score",
            "spread_to_move_stress_score",
            "cross_asset_liquidity_tail_score",
        ),
        "model_layer": "meta",
    },
    "volatility_tail_stress": {
        "tokens": (
            "vol",
            "rv",
            "atr",
            "range",
            "tail",
            "q_",
            "coexceed",
            "extreme",
        ),
        "deployable_features": (
            "tail_stress_score",
            "volatility_tail_width_score",
            "tail_co_movement_score",
        ),
        "model_layer": "meta",
    },
    "trend_range_breakout_state": {
        "tokens": (
            "trend",
            "slope",
            "range",
            "donchian",
            "dist_from_high",
            "dist_from_low",
            "dist_rolling",
            "compression",
            "efficiency",
            "breakout",
            "vwap",
            "adx",
            "rsi",
        ),
        "deployable_features": (
            "trend_range_breakout_stress_score",
            "range_compression_release_score",
            "breakout_distance_state_score",
        ),
        "model_layer": "base_asset_local_or_meta_context",
    },
    "breadth_market_return_state": {
        "tokens": (
            "mkt_ret",
            "btc_ret",
            "eth_ret",
            "breadth",
            "pct_assets",
            "market_breadth",
            "cs_ret",
        ),
        "deployable_features": (
            "market_breadth_fragility_score",
            "market_return_state_score",
            "cross_sectional_return_dispersion_score",
        ),
        "model_layer": "meta",
    },
    "relative_value_path_state": {
        "tokens": (
            "carry_adj",
            "peer_resid",
            "ret_self",
            "ret4h_peer",
            "ker_",
            "impulse_ratio",
            "dir_path",
            "zscore_price",
            "draw_sym",
            "retest",
            "wick",
        ),
        "deployable_features": (
            "relative_value_path_stress_score",
            "carry_path_asymmetry_score",
            "peer_residual_break_score",
        ),
        "model_layer": "base_asset_local_or_meta_context",
    },
    "covariance_network_concentration": {
        "tokens": (
            "eig_",
            "pc1",
            "effective_rank",
            "participation_ratio",
            "mean_abs_corr",
            "cov",
            "corr",
            "precision",
            "partial_corr",
            "network",
        ),
        "deployable_features": (
            "covariance_distance_from_trailing_baseline",
            "largest_eigenvalue_share",
            "effective_rank_change",
            "precision_edge_turnover",
            "network_rewiring_score",
        ),
        "model_layer": "meta",
    },
}


def _is_forbidden(feature: str) -> bool:
    lower = str(feature).lower()
    return any(token in lower for token in FORBIDDEN_TOKENS)


def _mechanism_for_feature(feature: str) -> str:
    lower = str(feature).lower()
    best = "other_structural"
    best_hits = 0
    for channel, spec in MECHANISM_SPECS.items():
        hits = sum(1 for token in spec["tokens"] if token in lower)
        if hits > best_hits:
            best = channel
            best_hits = hits
    return best


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _normalise_gain(frame: pd.DataFrame, *, source: str, weight: float) -> pd.DataFrame:
    if frame.empty or "feature" not in frame.columns:
        return pd.DataFrame()
    out = frame.copy()
    out["feature"] = out["feature"].astype(str)
    out = out.loc[~out["feature"].map(_is_forbidden)].copy()
    if out.empty:
        return out
    gain = pd.to_numeric(out.get("gain_mean", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    head = out["head"].astype(str) if "head" in out.columns else pd.Series("unknown", index=out.index)
    max_gain = gain.groupby(head).transform("max").replace(0.0, np.nan)
    out["evidence_source"] = source
    out["mechanism_channel"] = out["feature"].map(_mechanism_for_feature)
    out["evidence_strength"] = (gain / max_gain).fillna(0.0).clip(0.0, 1.0) * float(weight)
    out["fold_count"] = pd.to_numeric(out.get("fold_count", 0), errors="coerce").fillna(0).astype(int)
    out["split_mean"] = pd.to_numeric(out.get("split_mean", 0.0), errors="coerce").fillna(0.0)
    return out[["head", "feature", "mechanism_channel", "evidence_source", "evidence_strength", "fold_count", "split_mean"]]


def _normalise_leaf_interactions(frame: pd.DataFrame, *, source: str, weight: float) -> pd.DataFrame:
    if frame.empty or "archetype_feature" not in frame.columns:
        return pd.DataFrame()
    out = frame.copy()
    out["feature"] = out["archetype_feature"].astype(str)
    out = out.loc[~out["feature"].map(_is_forbidden)].copy()
    if out.empty:
        return out
    score = pd.to_numeric(out.get("interaction_score", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    head = out["head"].astype(str) if "head" in out.columns else pd.Series("unknown", index=out.index)
    max_score = score.groupby(head).transform("max").replace(0.0, np.nan)
    out["evidence_source"] = source
    out["mechanism_channel"] = out["feature"].map(_mechanism_for_feature)
    out["evidence_strength"] = (score / max_score).fillna(0.0).clip(0.0, 1.0) * float(weight)
    out["fold_count"] = 0
    out["split_mean"] = pd.to_numeric(out.get("n_leaf", 0.0), errors="coerce").fillna(0.0)
    return out[["head", "feature", "mechanism_channel", "evidence_source", "evidence_strength", "fold_count", "split_mean"]]


def _collect_evidence(classifier_dir: Path, leaf_dir: Path) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    pieces.append(
        _normalise_gain(
            _read_csv(classifier_dir / "high_conf_failure_importance_all.csv"),
            source="high_conf_failure",
            weight=1.00,
        )
    )
    adv = _read_csv(classifier_dir / "adversarial_importance_all.csv")
    if not adv.empty and "diagnostic" in adv.columns:
        adv = adv.loc[adv["diagnostic"].astype(str).eq("adversarial_global_bad_weeks")].copy()
    pieces.append(_normalise_gain(adv, source="adversarial_bad_week_shift", weight=0.35))
    leaf_frames: list[pd.DataFrame] = []
    for path in sorted(leaf_dir.glob("*_leaf_archetype_interactions.csv")):
        leaf_frames.append(_read_csv(path))
    if leaf_frames:
        pieces.append(
            _normalise_leaf_interactions(
                pd.concat(leaf_frames, ignore_index=True),
                source="leaf_x_archetype_residual",
                weight=0.70,
            )
        )
    pieces = [p for p in pieces if isinstance(p, pd.DataFrame) and not p.empty]
    if not pieces:
        return pd.DataFrame(columns=["head", "feature", "mechanism_channel", "evidence_source", "evidence_strength"])
    evidence = pd.concat(pieces, ignore_index=True)
    evidence["head"] = evidence["head"].astype(str)
    return evidence


def _collapse_evidence(evidence: pd.DataFrame) -> pd.DataFrame:
    if evidence.empty:
        return evidence
    collapsed = (
        evidence.groupby(["head", "feature", "mechanism_channel", "evidence_source"], as_index=False)
        .agg(
            evidence_strength=("evidence_strength", "max"),
            evidence_strength_mean=("evidence_strength", "mean"),
            fold_count=("fold_count", "max"),
            split_mean=("split_mean", "sum"),
            repeated_support_rows=("feature", "size"),
        )
    )
    # Repeated leaf interactions are useful support, but should not dominate
    # linearly just because many leaves were inspected.
    collapsed["support_factor"] = np.log1p(collapsed["repeated_support_rows"]) / np.log(1 + 25)
    collapsed["support_factor"] = collapsed["support_factor"].clip(0.50, 1.25)
    collapsed["evidence_strength"] = collapsed["evidence_strength"] * collapsed["support_factor"]
    return collapsed


def _summarise_features(evidence: pd.DataFrame) -> pd.DataFrame:
    if evidence.empty:
        return pd.DataFrame()
    grouped = (
        evidence.groupby(["feature", "mechanism_channel"], as_index=False)
        .agg(
            evidence_strength_sum=("evidence_strength", "sum"),
            evidence_strength_mean=("evidence_strength", "mean"),
            head_count=("head", "nunique"),
            source_count=("evidence_source", "nunique"),
            fold_count_max=("fold_count", "max"),
            split_or_support_sum=("split_mean", "sum"),
            evidence_rows=("feature", "size"),
            repeated_support_rows=("repeated_support_rows", "sum"),
        )
        .sort_values(["evidence_strength_sum", "head_count", "source_count"], ascending=False)
    )
    recurrence = np.log1p(grouped["head_count"].clip(lower=0)) / np.log(1 + max(grouped["head_count"].max(), 1))
    source_factor = np.log1p(grouped["source_count"].clip(lower=0)) / np.log(1 + max(grouped["source_count"].max(), 1))
    grouped["recurrence_score"] = recurrence.fillna(0.0)
    grouped["source_diversity_score"] = source_factor.fillna(0.0)
    grouped["feature_score"] = (
        grouped["evidence_strength_sum"]
        * (0.60 + 0.25 * grouped["recurrence_score"] + 0.15 * grouped["source_diversity_score"])
    )
    return grouped.sort_values("feature_score", ascending=False).reset_index(drop=True)


def _summarise_channels(feature_summary: pd.DataFrame) -> pd.DataFrame:
    if feature_summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for channel, group in feature_summary.groupby("mechanism_channel", sort=False):
        spec = MECHANISM_SPECS.get(channel, {})
        top = group.sort_values("feature_score", ascending=False).head(12)
        rows.append(
            {
                "mechanism_channel": channel,
                "channel_score": float(group["feature_score"].sum()),
                "feature_count": int(group["feature"].nunique()),
                "head_count": int(group["head_count"].max()),
                "source_count": int(group["source_count"].max()),
                "top_features": ", ".join(top["feature"].astype(str).tolist()),
                "deployable_feature_candidates": ", ".join(spec.get("deployable_features", ())),
                "recommended_layer": spec.get("model_layer", "meta"),
            }
        )
    return pd.DataFrame(rows).sort_values("channel_score", ascending=False).reset_index(drop=True)


def _archetype_definitions(channel_summary: pd.DataFrame) -> dict[str, Any]:
    definitions: dict[str, Any] = {}
    name_map = {
        "model_path_fragility": "model_path_fragility_archetype",
        "oi_funding_crowding": "leverage_crowding_archetype",
        "liquidity_spread_stress": "liquidity_stress_archetype",
        "volatility_tail_stress": "tail_volatility_stress_archetype",
        "trend_range_breakout_state": "trend_range_breakout_archetype",
        "breadth_market_return_state": "market_breadth_archetype",
        "relative_value_path_state": "relative_value_path_archetype",
        "covariance_network_concentration": "network_concentration_archetype",
    }
    for _, row in channel_summary.iterrows():
        channel = str(row["mechanism_channel"])
        if channel == "other_structural":
            continue
        definitions[name_map.get(channel, f"{channel}_archetype")] = {
            "mechanism_channel": channel,
            "evidence_score": float(row["channel_score"]),
            "top_features": [s.strip() for s in str(row["top_features"]).split(",") if s.strip()][:12],
            "deployable_features": [
                s.strip() for s in str(row["deployable_feature_candidates"]).split(",") if s.strip()
            ],
            "recommended_layer": str(row["recommended_layer"]),
            "status": "candidate_from_diagnostics_not_yet_cross_fitted",
        }
    return definitions


def _write_report(
    out_dir: Path,
    feature_summary: pd.DataFrame,
    channel_summary: pd.DataFrame,
    definitions: dict[str, Any],
) -> None:
    lines: list[str] = []
    lines.append("# Bad-Regime Archetype Synthesis")
    lines.append("")
    lines.append(
        "This report groups clean failure-classifier, adversarial-shift, and leaf residual-interaction "
        "evidence into deployable soft-regime candidates.  It is a synthesis artifact, not a trained "
        "production feature generator."
    )
    lines.append("")
    lines.append("## Mechanism Channels")
    for _, row in channel_summary.head(12).iterrows():
        lines.append(
            f"- `{row['mechanism_channel']}` score={row['channel_score']:.3f}, "
            f"features={int(row['feature_count'])}, top={row['top_features']}"
        )
        if row.get("deployable_feature_candidates"):
            lines.append(f"  deployable candidates: {row['deployable_feature_candidates']}")
    lines.append("")
    lines.append("## Top Feature Evidence")
    for _, row in feature_summary.head(30).iterrows():
        lines.append(
            f"- `{row['feature']}` -> `{row['mechanism_channel']}` "
            f"score={row['feature_score']:.3f}, heads={int(row['head_count'])}, "
            f"sources={int(row['source_count'])}"
        )
    lines.append("")
    lines.append("## Proposed Soft Archetypes")
    for name, payload in definitions.items():
        deployable = ", ".join(payload.get("deployable_features", []))
        lines.append(f"- `{name}`: {deployable}")
    lines.append("")
    lines.append("## Remaining Work")
    lines.append("- Generate these deployable scores causally by timestamp, with trailing baselines only.")
    lines.append("- Cross-fit the scores into train_base/train_meta before using them as model features.")
    lines.append("- Validate feature lift with leave-one-episode-out and recent-window OOF metrics.")
    lines.append("- Residualize adversarial evidence against time/universe nuisance variables before treating it as causal.")
    (out_dir / "bad_regime_archetype_synthesis.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classifier-dir",
        default="data_perp/reports/meta_recent_failure_diagnostics_20260622_clean_classifiers",
    )
    parser.add_argument(
        "--leaf-dir",
        default="data_perp/reports/meta_recent_failure_diagnostics_20260622_leaf_archetype_check",
    )
    parser.add_argument(
        "--output-dir",
        default="data_perp/reports/meta_recent_failure_diagnostics_20260622_archetype_synthesis_clean_contract_v1",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    classifier_dir = Path(args.classifier_dir)
    leaf_dir = Path(args.leaf_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    evidence = _collect_evidence(classifier_dir, leaf_dir)
    evidence.to_csv(out_dir / "archetype_evidence_long.csv", index=False)
    collapsed = _collapse_evidence(evidence)
    collapsed.to_csv(out_dir / "archetype_evidence_collapsed.csv", index=False)
    feature_summary = _summarise_features(collapsed)
    feature_summary.to_csv(out_dir / "archetype_feature_candidates.csv", index=False)
    channel_summary = _summarise_channels(feature_summary)
    channel_summary.to_csv(out_dir / "archetype_mechanism_candidates.csv", index=False)
    definitions = _archetype_definitions(channel_summary)
    (out_dir / "soft_archetype_definitions.json").write_text(json.dumps(definitions, indent=2, sort_keys=True))
    _write_report(out_dir, feature_summary, channel_summary, definitions)
    print(f"Wrote archetype synthesis to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
