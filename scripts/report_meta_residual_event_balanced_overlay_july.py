#!/usr/bin/env python3
"""Apply the frozen event-balanced V9 overlay to a July single-source ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts import run_meta_residual_extreme_local_champion_overlay as champion
from scripts.run_meta_residual_event_balanced_error_overlay import (
    Config,
    KEYS,
    RISK_PCT,
    RISK_SCORE,
    SIDE_RISK_PCT,
    SIDE_RISK_SCORE,
    _add_risk_variants,
    _apply_selected_overlays,
    _load_frames,
    _merge_temporal,
    _midrank,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        default=Path(
            "data_perp/reports/meta_residual_event_balanced_error_overlay_"
            "20260713_v10_frozen_candidate"
        ),
    )
    parser.add_argument(
        "--july-ledger",
        type=Path,
        default=Path(
            "data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/"
            "july_complete_01_12_v9_mlp_strict_consistent_20260713/"
            "july_08_10_complete_predictions.parquet"
        ),
    )
    parser.add_argument(
        "--state-artifact",
        type=Path,
        default=Path(
            "data_perp/reports/residual_event_target_transitions_july_oos_20260713_"
            "v2_support_fallback/oos_residual_event_states.parquet"
        ),
    )
    parser.add_argument(
        "--temporal-state-features",
        type=Path,
        default=Path(
            "data_perp/reports/residual_event_target_transitions_july_oos_20260713_"
            "v2_support_fallback/oos_temporal_state_context_apr2025_july2026.parquet"
        ),
    )
    parser.add_argument(
        "--market-features",
        type=Path,
        default=Path("data_perp/features/20260712_185800/symbol=BTC_USD:USD.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data_perp/reports/meta_residual_event_balanced_error_overlay_"
            "20260713_v10_frozen_candidate/july_forward"
        ),
    )
    return parser.parse_args()


def _training_args() -> SimpleNamespace:
    return SimpleNamespace(
        champion_ledger=Path(
            "data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/"
            "champion_frozen_single_source_202501_20260710/"
            "frozen_champion_single_source_ledger.parquet"
        ),
        parent_eval_predictions=Path(
            "data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/"
            "lifecycle_residual_aware_ae_gmm_overlay_pca8_clip8_baseline_"
            "globaloverlay_sparse_shock_composite/oos_predictions_historical_rank.parquet"
        ),
        state_artifact=Path(
            "data_perp/reports/residual_event_archetype_true_base_oof_compactlocal_"
            "market_20260712_v3/oos_residual_event_states.parquet"
        ),
        train_oof_predictions_dir=Path(
            "data_perp/reports/s59_h5_2025start_monthly_v4_base_configfull_"
            "mdafs120_hpo150_largestfold_oos15_ae3000_nocrossfit_k34567_"
            "payload300k_20260706/train_meta_regime_handoff_singlehead_base_soft_"
            "lgbmpipeline_auto_hpo150_oos15_top30_hpo45k_20260706_v5/"
            "best_full_oos_fixedfs_streamed_v1/prediction_shards"
        ),
        train_oof_rank_cache=Path(
            "data_perp/reports/residual_event_archetype_true_base_oof_compactlocal_"
            "market_20260712_v3/meta_oof_global_rank_202504_202603.parquet"
        ),
        negative_residual_features=Path(
            "data_perp/features/20260712_185800/symbol=BTC_USD:USD.parquet"
        ),
        temporal_state_features=Path(
            "data_perp/reports/residual_event_target_transitions_july_oos_20260713_"
            "v2_support_fallback/oos_temporal_state_context_apr2025_july2026.parquet"
        ),
        v9_manifest=Path(
            "data_perp/reports/meta_residual_extreme_local_champion_overlay_"
            "ooftrain_tieaware_downonly_20260712_v9/manifest.json"
        ),
        v9_selected_features=Path(
            "data_perp/reports/meta_residual_extreme_local_champion_overlay_"
            "ooftrain_tieaware_downonly_20260712_v9/selected_local_features_strict.csv"
        ),
        v9_predictions=Path(
            "data_perp/reports/meta_residual_extreme_local_champion_overlay_"
            "ooftrain_tieaware_downonly_20260712_v9/oos_predictions.parquet"
        ),
    )


def _merge_context(
    july: pd.DataFrame,
    *,
    state_path: Path,
    temporal_path: Path,
    market_path: Path,
    required: set[str],
) -> pd.DataFrame:
    state_names = set(pq.read_schema(state_path).names)
    state_cols = [name for name in required if name in state_names and name not in july]
    if state_cols:
        state = pd.read_parquet(state_path, columns=KEYS + sorted(state_cols))
        state["__ts__"] = pd.to_datetime(state["__ts__"], utc=True)
        july = july.merge(
            state.drop_duplicates(KEYS, keep="last"),
            on=KEYS,
            how="left",
            validate="one_to_one",
        )
    july = _merge_temporal(july, temporal_path)
    market_names = set(pq.read_schema(market_path).names)
    market_cols = [name for name in required if name in market_names and name not in july]
    if market_cols:
        market = pd.read_parquet(market_path, columns=sorted(market_cols))
        market.index = pd.to_datetime(market.index, utc=True, errors="coerce")
        market = market.loc[~market.index.duplicated(keep="last")]
        market.index.name = "__ts__"
        july = july.merge(
            market.reset_index(), on="__ts__", how="left", validate="many_to_one"
        )
    return july


def _score_frozen_state(
    rows: pd.DataFrame,
    model_path: Path,
    state_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    state = np.load(state_path, allow_pickle=False)
    features = state["features"].astype(str).tolist()
    medians = state["medians"].astype(np.float32)
    reference = state["reference"].astype(np.float32)
    values = rows.loc[:, ["__ts__", *features]].copy()
    for feature in features:
        values[feature] = pd.to_numeric(values[feature], errors="coerce").astype(
            np.float32
        )
    timestamp = values.groupby("__ts__", observed=True, sort=True)[features].median()
    matrix = timestamp.to_numpy(np.float32, copy=True)
    missing = ~np.isfinite(matrix)
    if missing.any():
        matrix[missing] = np.take(medians, np.nonzero(missing)[1])
    model = lgb.Booster(model_file=str(model_path))
    timestamp_score = np.asarray(model.predict(matrix), dtype=np.float32)
    score_map = pd.Series(timestamp_score, index=timestamp.index)
    row_score = rows["__ts__"].map(score_map).to_numpy(np.float32)
    return row_score, _midrank(row_score, reference)


def _daily_metrics(frame: pd.DataFrame, rank_col: str, selector: str) -> pd.DataFrame:
    selected = frame.loc[frame[rank_col].ge(0.90)].copy()
    selected["day"] = selected["__ts__"].dt.strftime("%Y-%m-%d")
    report = (
        selected.groupby("day", observed=True, sort=True)
        .agg(
            selected_rows=("ev_after_1pct", "size"),
            mean_ev_after_1pct=("ev_after_1pct", "mean"),
            positive_ev_rate=("ev_after_1pct", lambda values: float((values > 0).mean())),
            clean_exec_precision=("clean_exec", "mean"),
        )
        .reset_index()
    )
    report.insert(1, "selector", selector)
    return report


def main() -> None:
    args = _args()
    artifact = json.loads((args.overlay_dir / "manifest.json").read_text())
    accepted = {
        (str(row["side_name"]), str(row["archetype_policy_key"])): row
        for row in artifact["accepted_local_overlays"]
    }
    required: set[str] = set()
    for state_path in args.overlay_dir.glob("state__*.npz"):
        with np.load(state_path, allow_pickle=False) as state:
            required.update(state["features"].astype(str).tolist())

    train_args = _training_args()
    train, _, _ = _load_frames(train_args, Config())
    july = pd.read_parquet(args.july_ledger)
    july["__ts__"] = pd.to_datetime(july["__ts__"], utc=True)
    july = july.sort_values(KEYS, kind="stable").drop_duplicates(KEYS, keep="last")
    july = _merge_context(
        july,
        state_path=args.state_artifact,
        temporal_path=args.temporal_state_features,
        market_path=args.market_features,
        required=required | set(champion.STATE_NATIVE_FEATURES),
    )
    missing = sorted(name for name in required if name not in july)
    if missing:
        raise RuntimeError("Missing frozen overlay inputs: " + ", ".join(missing))

    v9_manifest = json.loads(train_args.v9_manifest.read_text())
    catalog = pd.read_csv(train_args.v9_selected_features)
    july["parent_rank_v9"], _, _ = champion._rank_for_params(
        train, july, catalog, v9_manifest["strict_best"]
    )
    july[RISK_SCORE] = np.float32(np.nan)
    july[RISK_PCT] = np.float32(0.5)
    for side, archetype in accepted:
        mask = (
            july["side_name"].astype(str).eq(side)
            & july["archetype_policy_key"].astype(str).eq(archetype)
            & july["parent_rank_v9"].ge(0.80)
        )
        if not mask.any():
            continue
        score, percentile = _score_frozen_state(
            july.loc[mask],
            args.overlay_dir / f"model__{side}__{archetype}.txt",
            args.overlay_dir / f"state__{side}__{archetype}.npz",
        )
        july.loc[mask, RISK_SCORE] = score
        july.loc[mask, RISK_PCT] = percentile
    july[SIDE_RISK_SCORE] = np.float32(np.nan)
    july[SIDE_RISK_PCT] = np.float32(0.5)
    for side in sorted({key[0] for key in accepted}):
        mask = july["side_name"].astype(str).eq(side) & july["parent_rank_v9"].ge(0.80)
        score, percentile = _score_frozen_state(
            july.loc[mask],
            args.overlay_dir / f"model__side_parent__{side}.txt",
            args.overlay_dir / f"state__side_parent__{side}.npz",
        )
        july.loc[mask, SIDE_RISK_SCORE] = score
        july.loc[mask, SIDE_RISK_PCT] = percentile
    july = _add_risk_variants(july)
    adjusted, flagged = _apply_selected_overlays(july, accepted, "parent_rank_v9")
    july["parent_rank_v9_event_balanced_overlay"] = adjusted
    july["event_balanced_overlay_flagged"] = flagged

    args.output.mkdir(parents=True, exist_ok=True)
    july.to_parquet(args.output / "july_predictions.parquet", index=False, compression="zstd")
    daily = pd.concat(
        [
            _daily_metrics(july, "parent_rank_v9", "v9_parent"),
            _daily_metrics(
                july,
                "parent_rank_v9_event_balanced_overlay",
                "v9_event_balanced_overlay",
            ),
        ],
        ignore_index=True,
    )
    daily.to_csv(args.output / "daily_metrics.csv", index=False)
    overall = (
        daily.groupby("selector", observed=True)
        .apply(
            lambda group: pd.Series(
                {
                    "selected_rows": int(group["selected_rows"].sum()),
                    "mean_ev_after_1pct": float(
                        np.average(
                            group["mean_ev_after_1pct"],
                            weights=group["selected_rows"],
                        )
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    overall.to_csv(args.output / "overall_metrics.csv", index=False)
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
