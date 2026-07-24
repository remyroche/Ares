#!/usr/bin/env python3
"""Greedily add sparse archetype failure overlays to the residual champion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import scripts.run_meta_residual_extreme_local_champion_overlay as champion


CLUSTER = champion.CLUSTER_COLUMN
THRESHOLDS = (0.90, 0.925, 0.95, 0.975, 0.99)
ALPHAS = (0.005, 0.01, 0.02, 0.03, 0.05)

MECHANISMS: dict[str, dict[str, Any]] = {
    "long_breakout_fragmentation": {
        "side": "long",
        "archetype": "long_breakout_diagnostic_candidate",
        "features": (
            "breadth_dispersion",
            "fragmented_new_low_breadth",
            "btc_resilience_alt_weakness",
            "correlation_breakdown_dispersion",
        ),
    },
    "long_breakout_failed_broadening": {
        "side": "long",
        "archetype": "long_breakout_diagnostic_candidate",
        "features": (
            "unconfirmed_long_breakout",
            "false_clean_short",
            "peer_decoupling_acceleration",
        ),
    },
    "long_volcompression_false_compression": {
        "side": "long",
        "archetype": "long_volcompression_wideslow_candidate",
        "features": (
            "compressed_index_fragmented_assets",
            "thin_compression",
            "flush_recovery_state",
        ),
    },
    "short_breakout_liquidation_exhaustion": {
        "side": "short",
        "archetype": "short_breakout_precision",
        "features": (
            "short_breakout_exhaustion",
            "range_climax_reversal",
            "deleveraging_without_followthrough",
            "flush_recovery_state",
        ),
    },
    "short_default_short_cover_transition": {
        "side": "short",
        "archetype": "short_default_clean_path",
        "features": (
            "short_covering_score_market",
            "funding_confirmed_long_flush",
            "funding_confirmed_short_covering",
        ),
    },
    "short_mixed_washout_recovery": {
        "side": "short",
        "archetype": "short_mixed_clean_path",
        "features": (
            "broad_washout_recovery",
            "downside_breadth_intensity",
            "fragmented_flush_recovery",
            "btc_decoupling_dispersion",
        ),
    },
    "long_mixed_recovery_conflict": {
        "side": "long",
        "archetype": "long_mixed_wideslow_tentative",
        "features": (
            "fragile_leverage_rebuild",
            "short_signal_recovery_conflict",
            "extreme_negative_breadth_pct",
        ),
    },
    "short_breakout_transition_exhaustion": {
        "side": "short",
        "archetype": "short_breakout_precision",
        "features": (
            "mkt_regime_change__short_covering__delta_1h",
            "mkt_regime_change__short_covering__acceleration_1h",
            "mkt_regime_change__flush_recovery__delta_1h",
            "mkt_regime_change__flush_recovery__cumulative_change_2d",
            "mkt_regime_change__oi_contraction__acceleration_1h",
            "mkt_regime_change__eth_correlation__cumulative_change_2d",
        ),
        "directions": {
            "mkt_regime_change__oi_contraction__acceleration_1h": -1.0,
            "mkt_regime_change__eth_correlation__cumulative_change_2d": -1.0,
        },
    },
    "short_default_transition_covering": {
        "side": "short",
        "archetype": "short_default_clean_path",
        "features": (
            "mkt_regime_change__short_covering__delta_1h",
            "mkt_regime_change__flush_recovery__delta_1h",
            "mkt_regime_change__oi_contraction__acceleration_1h",
            "mkt_regime_change__funding__cumulative_change_2d",
        ),
        "directions": {
            "mkt_regime_change__oi_contraction__acceleration_1h": -1.0,
            "mkt_regime_change__funding__cumulative_change_2d": -1.0,
        },
    },
    "long_breakout_transition_fragmentation": {
        "side": "long",
        "archetype": "long_breakout_diagnostic_candidate",
        "features": (
            "mkt_regime_change__negative_breadth__cumulative_change_2d",
            "mkt_regime_change__eth_correlation__cumulative_change_2d",
            "mkt_regime_change__btc_alt_relative_strength__cumulative_change_2d",
            "mkt_regime_change__funding__acceleration_1h",
        ),
        "directions": {
            "mkt_regime_change__eth_correlation__cumulative_change_2d": -1.0,
            "mkt_regime_change__btc_alt_relative_strength__cumulative_change_2d": -1.0,
        },
    },
    "long_dirtyavoid_transition_stress": {
        "side": "long",
        "archetype": "long_dirtyavoid_sparse_questionable",
        "features": (
            "mkt_regime_change__negative_breadth__cumulative_change_2d",
            "mkt_regime_change__btc_alt_relative_strength__cumulative_change_2d",
            "mkt_regime_change__funding__cumulative_change_2d",
            "mkt_regime_change__oi_contraction__cumulative_change_2d",
        ),
        "directions": {
            "mkt_regime_change__btc_alt_relative_strength__cumulative_change_2d": -1.0,
        },
    },
    "short_breakout_recent_latent_dislocation": {
        "side": "short",
        "archetype": "short_breakout_precision",
        "features": (
            "resid_event_aegmm_reconstruction_recent_max_24h",
            "resid_event_aegmm_reconstruction_recent_max_48h",
            "resid_event_aegmm_reconstruction_recent_max_96h",
            "resid_event_aegmm_posterior_speed",
            "resid_event_aegmm_posterior_acceleration",
            "resid_event_aegmm_hours_since_ood_spike_96h_norm",
        ),
        "directions": {
            "resid_event_aegmm_hours_since_ood_spike_96h_norm": -1.0,
        },
    },
    "long_dirtyavoid_recent_latent_dislocation": {
        "side": "long",
        "archetype": "long_dirtyavoid_sparse_questionable",
        "features": (
            "resid_event_aegmm_reconstruction_recent_max_24h",
            "resid_event_aegmm_reconstruction_recent_max_48h",
            "resid_event_aegmm_reconstruction_recent_max_96h",
            "resid_event_aegmm_posterior_speed",
            "resid_event_aegmm_posterior_acceleration",
            "resid_event_aegmm_hours_since_ood_spike_96h_norm",
        ),
        "directions": {
            "resid_event_aegmm_hours_since_ood_spike_96h_norm": -1.0,
        },
    },
}


def _midrank(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    output = np.full(len(values), 0.5, dtype=np.float32)
    finite = np.isfinite(values)
    left = np.searchsorted(reference, values[finite], side="left")
    right = np.searchsorted(reference, values[finite], side="right")
    output[finite] = (left + right) / (2.0 * max(len(reference), 1))
    return output


def _fit_mechanism(
    train: pd.DataFrame, spec: dict[str, Any]
) -> dict[str, Any]:
    local = train.loc[
        train["side_name"].astype(str).eq(spec["side"])
        & train["archetype_policy_key"].astype(str).eq(spec["archetype"])
    ]
    archetype_refs: dict[str, np.ndarray] = {}
    cluster_refs: dict[int, dict[str, np.ndarray]] = {}
    cluster_support: dict[int, int] = {}
    for feature in spec["features"]:
        values = pd.to_numeric(local[feature], errors="coerce").to_numpy(np.float32)
        reference = np.sort(values[np.isfinite(values)])
        if len(reference) >= 500:
            archetype_refs[feature] = reference
    for cluster_id, group in local.groupby(CLUSTER, observed=True):
        if not np.isfinite(cluster_id):
            continue
        cluster_id = int(cluster_id)
        cluster_support[cluster_id] = int(len(group))
        refs = {}
        for feature in spec["features"]:
            values = pd.to_numeric(group[feature], errors="coerce").to_numpy(np.float32)
            reference = np.sort(values[np.isfinite(values)])
            if len(reference) >= 200:
                refs[feature] = reference
        cluster_refs[cluster_id] = refs
    return {
        "side": spec["side"],
        "archetype": spec["archetype"],
        "features": tuple(archetype_refs),
        "archetype_refs": archetype_refs,
        "cluster_refs": cluster_refs,
        "cluster_support": cluster_support,
        "directions": {
            str(feature): float(direction)
            for feature, direction in spec.get("directions", {}).items()
        },
    }


def _mechanism_score(frame: pd.DataFrame, state: dict[str, Any]) -> np.ndarray:
    output = np.full(len(frame), 0.5, dtype=np.float32)
    local_mask = (
        frame["side_name"].astype(str).eq(state["side"])
        & frame["archetype_policy_key"].astype(str).eq(state["archetype"])
    ).to_numpy()
    for cluster_id in np.unique(
        pd.to_numeric(frame.loc[local_mask, CLUSTER], errors="coerce").dropna().to_numpy()
    ):
        cluster_id = int(cluster_id)
        idx = np.flatnonzero(
            local_mask
            & pd.to_numeric(frame[CLUSTER], errors="coerce").eq(cluster_id).to_numpy()
        )
        if not len(idx):
            continue
        archetype_components = []
        cluster_components = []
        for feature in state["features"]:
            values = pd.to_numeric(frame.iloc[idx][feature], errors="coerce").to_numpy(
                np.float32
            )
            archetype_components.append(
                (
                    _midrank(values, state["archetype_refs"][feature])
                    if float(state["directions"].get(feature, 1.0)) >= 0.0
                    else 1.0 - _midrank(values, state["archetype_refs"][feature])
                )
            )
            reference = state["cluster_refs"].get(cluster_id, {}).get(feature)
            if reference is not None:
                local_rank = _midrank(values, reference)
                if float(state["directions"].get(feature, 1.0)) < 0.0:
                    local_rank = 1.0 - local_rank
                cluster_components.append(local_rank)
        if not archetype_components:
            continue
        parent = np.mean(np.column_stack(archetype_components), axis=1)
        if cluster_components:
            child = np.mean(np.column_stack(cluster_components), axis=1)
            support = float(state["cluster_support"].get(cluster_id, 0))
            confidence = min(1.0, support / 1500.0)
            output[idx] = confidence * child + (1.0 - confidence) * parent
        else:
            output[idx] = parent
    return output.astype(np.float32)


def _apply(
    rank: np.ndarray, score: np.ndarray, threshold: float, alpha: float
) -> np.ndarray:
    adjusted = rank.copy()
    intensity = np.clip(
        (score - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0
    )
    selected = rank >= 0.90
    adjusted[selected] -= np.float32(alpha) * intensity[selected]
    return np.clip(adjusted, 0.0, 1.0)


def _objective(
    rank: np.ndarray,
    base_count: int,
    month_code: np.ndarray,
    month_count: int,
    ev: np.ndarray,
    clean: np.ndarray,
) -> dict[str, float]:
    selected = rank >= 0.90
    finite_ev = selected & np.isfinite(ev)
    finite_clean = selected & np.isfinite(clean)
    ev_count = np.bincount(month_code, weights=finite_ev, minlength=month_count)
    clean_count = np.bincount(
        month_code, weights=finite_clean, minlength=month_count
    )
    ev_sum = np.bincount(
        month_code, weights=np.where(finite_ev, ev, 0.0), minlength=month_count
    )
    clean_sum = np.bincount(
        month_code,
        weights=np.where(finite_clean, clean, 0.0),
        minlength=month_count,
    )
    valid = (ev_count > 0) & (clean_count > 0)
    monthly_ev = ev_sum[valid] / ev_count[valid]
    monthly_clean = clean_sum[valid] / clean_count[valid]
    activity = float(selected.sum()) / max(base_count, 1)
    value = (
        float(monthly_ev.mean())
        - 0.5 * float(monthly_ev.std())
        + 0.25 * float(monthly_ev.min())
        + 0.002 * float(monthly_clean.mean())
        - 0.01 * abs(np.log(max(activity, 1e-6)))
    )
    return {
        "objective": value,
        "activity_ratio": activity,
        "mean_month_ev": float(monthly_ev.mean()),
        "worst_month_ev": float(monthly_ev.min()),
        "clean_precision": float(monthly_clean.mean()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_start = pd.Timestamp(args.train_start, tz="UTC")
    train_end = pd.Timestamp(args.train_end, tz="UTC")
    eval_end = pd.Timestamp(args.eval_end, tz="UTC")
    train, valid, coverage = champion._load_joined(
        champion_path=args.champion_ledger,
        parent_eval_path=args.parent_eval_predictions,
        state_path=args.state_artifact,
        train_oof_predictions_dir=args.train_oof_predictions_dir,
        train_oof_rank_cache=args.train_oof_rank_cache,
        train_start=train_start,
        train_end=train_end,
        eval_end=eval_end,
        negative_residual_features=args.negative_residual_features,
    )
    states = {name: _fit_mechanism(train, spec) for name, spec in MECHANISMS.items()}
    train_scores = {name: _mechanism_score(train, state) for name, state in states.items()}
    valid_scores = {name: _mechanism_score(valid, state) for name, state in states.items()}
    parent_train = pd.to_numeric(train["historical_rank"], errors="coerce").to_numpy(np.float32)
    parent_valid = pd.to_numeric(valid["historical_rank"], errors="coerce").to_numpy(np.float32)
    base_count = int((parent_train >= 0.90).sum())
    month_code, month_labels = pd.factorize(
        train["__ts__"].dt.strftime("%Y-%m"), sort=True
    )
    month_code = month_code.astype(np.int16, copy=False)
    month_count = int(len(month_labels))
    train_ev = pd.to_numeric(train["ev_after_1pct"], errors="coerce").to_numpy(
        np.float32
    )
    train_clean = pd.to_numeric(train["clean_exec"], errors="coerce").to_numpy(
        np.float32
    )
    current_train = parent_train.copy()
    current_valid = parent_valid.copy()
    current_objective = _objective(
        current_train,
        base_count,
        month_code,
        month_count,
        train_ev,
        train_clean,
    )["objective"]
    remaining = set(MECHANISMS)
    greedy_rows = []
    accepted = []
    while remaining:
        candidates = []
        for name in sorted(remaining):
            for threshold in THRESHOLDS:
                for alpha in ALPHAS:
                    proposed = _apply(current_train, train_scores[name], threshold, alpha)
                    metrics = _objective(
                        proposed,
                        base_count,
                        month_code,
                        month_count,
                        train_ev,
                        train_clean,
                    )
                    candidates.append(
                        {"mechanism": name, "threshold": threshold, "alpha": alpha, **metrics}
                    )
        best = max(candidates, key=lambda row: row["objective"])
        gain = float(best["objective"] - current_objective)
        best["incremental_objective"] = gain
        best["step"] = len(accepted) + 1
        greedy_rows.extend(candidates)
        if gain <= args.minimum_objective_gain or best["activity_ratio"] < 0.95:
            break
        name = str(best["mechanism"])
        current_train = _apply(
            current_train, train_scores[name], float(best["threshold"]), float(best["alpha"])
        )
        current_valid = _apply(
            current_valid, valid_scores[name], float(best["threshold"]), float(best["alpha"])
        )
        current_objective = float(best["objective"])
        accepted.append(best)
        remaining.remove(name)

    parent_mask = parent_valid >= 0.90
    final_mask = current_valid >= 0.90
    parent_name = str(args.parent_name)
    final_name = f"{parent_name}_greedy_sparse_mechanisms"
    summary = pd.DataFrame(
        [
            champion._metric_row(valid, parent_mask, parent_name),
            champion._metric_row(valid, final_mask, final_name),
        ]
    )
    _, autocorrelation = champion._autocorrelation_report(
        valid, {parent_name: parent_mask, final_name: final_mask}
    )
    mean_ac = (
        autocorrelation.groupby("arm", observed=True)["signed_surprise_autocorr_lag1"]
        .apply(lambda values: float(values.abs().mean()))
        .rename("mean_abs_signed_surprise_autocorr_lag1")
    )
    summary = summary.merge(mean_ac, left_on="selector", right_index=True, how="left")
    valid["historical_rank_greedy_sparse_mechanisms"] = current_valid
    valid["selected_parent"] = parent_mask
    valid["selected_greedy_sparse_mechanisms"] = final_mask
    for name, score in valid_scores.items():
        valid[f"mechanism__{name}"] = score
    valid.to_parquet(args.output_dir / "oos_predictions.parquet", index=False, compression="zstd")
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    pd.DataFrame(accepted).to_csv(args.output_dir / "accepted_mechanisms.csv", index=False)
    pd.DataFrame(greedy_rows).to_csv(args.output_dir / "greedy_search.csv", index=False)
    pd.concat(
        [
            champion._breakdown(valid, parent_mask, parent_name),
            champion._breakdown(valid, final_mask, final_name),
        ],
        ignore_index=True,
    ).to_csv(args.output_dir / "breakdowns.csv", index=False)
    autocorrelation.to_csv(args.output_dir / "hit_surprise_autocorrelation.csv", index=False)
    manifest = {
        "schema": "meta_residual_sparse_mechanism_overlay_v1",
        "parent": parent_name,
        "coverage": coverage,
        "mechanisms": MECHANISMS,
        "accepted_mechanisms": accepted,
        "selection_period": [args.train_start, args.train_end],
        "untouched_evaluation_period": [args.train_end, args.eval_end],
        "leakage_contract": "Mechanism formulas are fixed. Cluster/archetype references and greedy parameters use train OOF rows through March; April-June is evaluated once.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion-ledger", type=Path, default=Path("data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/champion_frozen_single_source_202501_20260710/frozen_champion_single_source_ledger.parquet"))
    parser.add_argument("--train-oof-predictions-dir", type=Path, default=Path("data_perp/reports/s59_h5_2025start_monthly_v4_base_configfull_mdafs120_hpo150_largestfold_oos15_ae3000_nocrossfit_k34567_payload300k_20260706/train_meta_regime_handoff_singlehead_base_soft_lgbmpipeline_auto_hpo150_oos15_top30_hpo45k_20260706_v5/best_full_oos_fixedfs_streamed_v1/prediction_shards"))
    parser.add_argument("--train-oof-rank-cache", type=Path, default=Path("data_perp/reports/residual_event_archetype_true_base_oof_compactlocal_market_20260712_v3/meta_oof_global_rank_202504_202603.parquet"))
    parser.add_argument("--state-artifact", type=Path, default=Path("data_perp/reports/residual_event_archetype_true_base_oof_compactlocal_market_20260712_v3/oos_residual_event_states.parquet"))
    parser.add_argument("--parent-eval-predictions", type=Path, default=Path("data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/lifecycle_residual_aware_ae_gmm_overlay_pca8_clip8_baseline_globaloverlay_sparse_shock_composite/oos_predictions_historical_rank.parquet"))
    parser.add_argument("--negative-residual-features", type=Path, default=Path("data_perp/features/20260712_185800/symbol=BTC_USD:USD.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_perp/reports/meta_residual_sparse_mechanism_overlay_20260712_v1"))
    parser.add_argument("--train-start", default="2025-04-01")
    parser.add_argument("--train-end", default="2026-04-01")
    parser.add_argument("--eval-end", default="2026-07-01")
    parser.add_argument("--minimum-objective-gain", type=float, default=1e-6)
    parser.add_argument("--parent-name", default=champion.PARENT)
    args = parser.parse_args()
    manifest = run(args)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
