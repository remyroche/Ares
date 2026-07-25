#!/usr/bin/env python3
"""Causally extend the frozen label-HPO winner and build a policy handoff.

The base model is deterministically refit because the original ablation did
not persist it.  Apr-Jun predictions must reproduce the frozen artifact before
any July score is accepted.  The residual model, EV map, and 21-day admission
calibrator are loaded unchanged from the frozen artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.base_residual_label_ablation import (  # noqa: E402
    FixedWindowCalendar,
    build_soft_label,
    default_label_recipes,
    label_components,
    rank_mask,
)
from extreme_price_movements.training_resource_guard import (  # noqa: E402
    TrainingResourceGuard,
    TrainingResourceLimits,
)
from scripts.run_base_residual_label_ablation import (  # noqa: E402
    DEFAULT_AE_ROOT,
    DEFAULT_BASE_CONTRACT,
    DEFAULT_FEATURE_STORE,
    DEFAULT_LABELS,
    DEFAULT_PATH_LABELS,
    ECONOMIC,
    WEIGHT,
    _base_rank,
    _contract,
    _fit_base,
    _load_feature_matrix,
    _load_side_labels,
    _residual_predict,
    _side_loader,
    _time_spread_indices,
)

DEFAULT_FROZEN = ROOT / "data_perp/artifacts/base_residual_label_ablation_20260725_v2"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/label_hpo_policy_replay_20260725_v1"
WINNER = {"long": "hpo_01", "short": "baseline_24h"}
TRIAL_INDEX = {"baseline_24h": 0, "timeout_12h": 1, "time_aware_12h": 2, "hpo_01": 4}
SIDE_OFFSET = {"long": 0, "short": 1000}


def _rank_high_is_best(frame: pd.DataFrame, score: str) -> pd.Series:
    return frame.groupby(["__ts__", "side_name"], sort=False)[score].rank(
        method="first", pct=True, ascending=True
    )


def _policy_contract(labels_root: Path, side: str) -> pd.DataFrame:
    import duckdb

    files = sorted(labels_root.glob(f"train_global_{side}_5_*.parquet"))
    con = duckdb.connect(database=":memory:")
    try:
        out = con.execute(
            """
            SELECT candidate_id, __barrier_pct__, __archetype_policy_key__
            FROM read_parquet(?, union_by_name=true)
            """,
            [list(map(str, files))],
        ).fetchdf()
    finally:
        con.close()
    return out.drop_duplicates("candidate_id")


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    guard = TrainingResourceGuard(
        limits=TrainingResourceLimits(
            min_free_ram_bytes=2 * 1024**3,
            max_process_rss_bytes=12 * 1024**3,
            min_free_disk_bytes=5 * 1024**3,
            check_interval_seconds=1.0,
        ),
        disk_path=args.output,
        telemetry_path=args.output / "resource_telemetry.jsonl",
    )
    calendar = FixedWindowCalendar(
        base_train_start=pd.Timestamp("2025-09-01", tz="UTC"),
        base_train_end=pd.Timestamp("2026-01-01", tz="UTC"),
        base_oos_end=pd.Timestamp("2026-07-21", tz="UTC"),
        meta_train_end=pd.Timestamp("2026-04-01", tz="UTC"),
    )
    recipes = {r.recipe_id: r for r in default_label_recipes(args.seed)}
    all_scored: list[pd.DataFrame] = []
    evidence: dict[str, object] = {
        "requested_end_inclusive": "2026-07-23",
        "effective_cutoff_is_data_bounded": True,
        "sides": {},
    }
    for side in ("long", "short"):
        guard.checkpoint(f"extend:{side}:load")
        frame, label_evidence = _load_side_labels(
            labels_root=args.labels,
            path_labels_root=args.path_labels,
            side=side,
            calendar=calendar,
        )
        features, params = _contract(args.base_contract, side)
        loader, candidates, loader_evidence = _side_loader(
            side=side,
            ae_root=args.ae_root,
            feature_store=args.feature_store,
            guard=guard,
        )
        missing = sorted(set(features) - set(candidates))
        if missing:
            raise RuntimeError(f"{side}: missing features {missing}")
        matrix = _load_feature_matrix(loader, frame, features, guard=guard, side=side)
        complete = np.isfinite(matrix.to_numpy(np.float32)).all(axis=1)
        frame = frame.loc[complete].reset_index(drop=True)
        matrix = matrix.loc[complete].reset_index(drop=True)
        masks = calendar.masks(frame["__ts__"])
        train = np.flatnonzero(masks["base_train"])
        oos = np.flatnonzero(masks["base_oos"])
        recipe_id = WINNER[side]
        target, _ = build_soft_label(label_components(frame), recipes[recipe_id])
        seed = args.seed + SIDE_OFFSET[side] + TRIAL_INDEX[recipe_id] * 20
        model = _fit_base(
            matrix,
            target,
            frame[WEIGHT].to_numpy(np.float64),
            _time_spread_indices(
                frame, np.isin(np.arange(len(frame)), train), args.base_train_rows
            ),
            params,
            seed=seed,
        )
        base_prediction = np.full(len(frame), np.nan)
        base_prediction[oos] = model.predict(matrix.iloc[oos])
        oos_frame = frame.iloc[oos].reset_index(drop=True)
        oos_matrix = matrix.iloc[oos].reset_index(drop=True)
        base_oos = base_prediction[oos]
        rank_pct = _base_rank(oos_frame, base_oos)
        top40 = rank_mask(oos_frame, base_oos, fraction=0.40, scope="timestamp_side")
        ev_map = joblib.load(args.frozen / side / "base_ev_map.joblib")
        calibrator = joblib.load(args.frozen / side / "admission_calibrator.joblib")
        residual = lgb.Booster(
            model_file=str(args.frozen / side / "residual_model.txt")
        )
        indices = np.flatnonzero(top40)
        score = _residual_predict(
            ev_map, residual, oos_matrix, base_oos, rank_pct, indices
        )
        scored = oos_frame.iloc[indices][
            ["candidate_id", "side_name", "__ts__", "__symbol__", ECONOMIC, WEIGHT]
        ].copy()
        scored["base_prediction"] = base_oos[indices]
        scored["base_rank_pct_timestamp_side"] = rank_pct[indices]
        scored["residual_score"] = score
        scored["calibrated_ev"] = calibrator.predict(score)
        scored["admitted_after_21d_calibrator"] = scored["calibrated_ev"] > 0.0
        scored["label_recipe"] = recipe_id

        frozen = pd.read_parquet(args.frozen / side / "meta_oos_predictions.parquet")
        check = frozen.merge(
            scored,
            on="candidate_id",
            how="left",
            suffixes=("_frozen", "_extended"),
            validate="one_to_one",
        )
        if check["residual_score_extended"].isna().any():
            raise RuntimeError(f"{side}: extended scoring missed frozen rows")
        parity = {
            name: float(
                np.max(
                    np.abs(
                        check[f"{name}_frozen"].to_numpy(float)
                        - check[f"{name}_extended"].to_numpy(float)
                    )
                )
            )
            for name in ("base_prediction", "residual_score", "calibrated_ev")
        }
        if parity["base_prediction"] > 1e-10 or parity["residual_score"] > 1e-10:
            raise RuntimeError(f"{side}: frozen parity failed: {parity}")
        scored.to_parquet(args.output / f"{side}_extended_scores.parquet", index=False)
        all_scored.append(scored)
        evidence["sides"][side] = {
            "winner_recipe": recipe_id,
            "base_refit_seed": seed,
            "rows": len(scored),
            "maximum_signal_timestamp": scored["__ts__"].max().isoformat(),
            "frozen_apr_jun_parity_max_abs": parity,
            "label_evidence": label_evidence,
            "loader_evidence": loader_evidence,
        }

    combined = pd.concat(all_scored, ignore_index=True)
    combined.to_parquet(args.output / "extended_scores.parquet", index=False)
    admitted = combined.loc[combined["admitted_after_21d_calibrator"]].copy()
    admitted["rank_pct"] = _rank_high_is_best(admitted, "residual_score")
    admitted = admitted.merge(
        pd.concat(
            [_policy_contract(args.labels, side) for side in ("long", "short")],
            ignore_index=True,
        ).drop_duplicates("candidate_id"),
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    handoff = pd.DataFrame(
        {
            "candidate_id": admitted["candidate_id"],
            "timestamp": admitted["__ts__"],
            "symbol": admitted["__symbol__"],
            "side": np.where(admitted["side_name"].eq("short"), -1.0, 1.0),
            "side_name": admitted["side_name"],
            "strategy_id": admitted["side_name"].map(
                lambda value: f"{value}_label_hpo_winner"
            ),
            "rank_pct": admitted["rank_pct"],
            "calibrated_score": admitted["calibrated_ev"],
            "barrier_pct": admitted["__barrier_pct__"],
            "archetype_policy_key": admitted["__archetype_policy_key__"],
            "policy_archetype": admitted["__archetype_policy_key__"],
            "residual_score": admitted["residual_score"],
        }
    )
    if handoff[["barrier_pct", "archetype_policy_key"]].isna().any().any():
        raise RuntimeError("policy contract join is incomplete")
    handoff.to_parquet(args.output / "policy_handoff_admitted.parquet", index=False)
    evidence["effective_maximum_signal_timestamp"] = (
        combined["__ts__"].max().isoformat()
    )
    evidence["policy_handoff_rows"] = len(handoff)
    evidence["policy_handoff_top10_rows"] = int((handoff["rank_pct"] >= 0.90).sum())
    (args.output / "manifest.json").write_text(
        json.dumps(evidence, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--path-labels", type=Path, default=DEFAULT_PATH_LABELS)
    parser.add_argument("--feature-store", type=Path, default=DEFAULT_FEATURE_STORE)
    parser.add_argument("--ae-root", type=Path, default=DEFAULT_AE_ROOT)
    parser.add_argument("--base-contract", type=Path, default=DEFAULT_BASE_CONTRACT)
    parser.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-train-rows", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, default=str))
