#!/usr/bin/env python3
"""Build and frozen-score strict post-OOF inputs for the promoted EV winner.

Only the winner's actually selected raw inputs are carried: Peak-MFE, the
seven-class path-CatBoost probabilities/entropy, alpha/context, and the clean
probability.  Timing/MAE/turn/slope are deliberately outside this EV action
layer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.execution_ev_model_ablation import (  # noqa: E402
    predict_execution_ev_model_ablation_bundle,
)

ID = ["__ts__", "__symbol__", "side_name", "candidate_id"]
WINNER = "catboost__residual__without_hpo__all_features"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/execution_ev_context_head_clean_forward_july19_20260726_v2"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=columns)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    frame["side_name"] = frame["side_name"].astype(str).str.lower()
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    if frame.duplicated(ID).any():
        raise ValueError(f"duplicate identity in {path}")
    return frame


def _exact(base: pd.DataFrame, addition: pd.DataFrame, *, name: str) -> pd.DataFrame:
    out = base.merge(addition, on=ID, how="inner", validate="one_to_one")
    if len(out) != len(base):
        raise ValueError(f"{name} does not cover the strict base identity")
    return out


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alpha", type=Path, default=ROOT / "data_perp/artifacts/execution_ev_alpha_oof_july20_20260726_v1/alpha_oof.parquet")
    p.add_argument("--context", type=Path, default=ROOT / "data_perp/artifacts/packb_downstream_representation_july20_20260726_v1_31_8/context.parquet")
    p.add_argument("--peak", type=Path, default=ROOT / "data_perp/artifacts/path_head_peak_forward_oos_july19_20260726_v2/peak_mfe_forward_oos_predictions.parquet")
    p.add_argument("--path-catboost", type=Path, default=ROOT / "data_perp/artifacts/path_head_catboost_forward_oos_july19_20260726_v1/catboost_archetype_forward_oos_predictions.parquet")
    p.add_argument("--clean", type=Path, default=ROOT / "data_perp/artifacts/meaningful_mfe_catboost_v2_ablation_july20_20260726_v1/exact_policy_paired.parquet")
    p.add_argument("--bundle", type=Path, default=ROOT / "data_perp/artifacts/execution_ev_context_head_clean_20260726_v1/execution_ev_model_ablation_bundle.joblib")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return p


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    bundle = joblib.load(args.bundle)
    raw = list(bundle.raw_feature_columns)
    alpha_columns = list(dict.fromkeys([*ID, "available_at", *[c for c in raw if c.startswith("existing_alpha") or c.startswith("alpha_") or c.startswith("base_archetype_label")]]))
    alpha = _read(args.alpha, alpha_columns).rename(columns={"available_at": "alpha_available_at"})
    context = _read(args.context, [*ID, "base_oof_score", "base_margin_to_cutoff", "base_margin_to_cutoff_z"])
    alpha = _exact(alpha, context, name="alpha/context base score and cutoff fields")
    # The alpha stream is the population contract.  It supplies base score and
    # cutoff context only from the same pre-entry selected-candidate stream.
    peak = _read(args.peak, [*ID, "pred_expected_peak_mfe_atr", "is_forward_oos", "peak_feature_complete", "prediction_available_at"])
    peak = peak.loc[peak["is_forward_oos"].astype(bool) & peak["peak_feature_complete"].astype(bool)].copy()
    peak = peak.rename(columns={"pred_expected_peak_mfe_atr": "pred_peak_MFE_12h_ATR", "prediction_available_at": "peak_mfe_available_at"}).drop(columns=["is_forward_oos", "peak_feature_complete"])
    cat_cols = [
        "probability__immediate_adverse_path", "probability__early_mfe_full_reversal", "probability__fast_realization_winner",
        "probability__late_breakout", "probability__slow_grinder", "probability__noisy_timeout_usable_mfe", "probability__dead_timeout",
    ]
    cat = _read(args.path_catboost, [*ID, *cat_cols, "probability_entropy", "predicted_path_archetype", "is_forward_oos", "catboost_feature_complete", "prediction_available_at"])
    cat = cat.loc[cat["is_forward_oos"].astype(bool) & cat["catboost_feature_complete"].astype(bool)].copy()
    cat = cat.rename(columns={
        **{name: f"catboost_p_{i}" for i, name in enumerate(cat_cols)},
        "probability_entropy": "catboost_entropy", "predicted_path_archetype": "catboost_archetype",
        "prediction_available_at": "catboost_available_at",
    }).drop(columns=["is_forward_oos", "catboost_feature_complete"])
    clean = _read(args.clean, [*ID, "catboost_hard_ensemble_platt", "execution_decision_utc", "execution_net_ev_12h", "execution_gross_ev_12h", "execution_label_end_utc", "execution_exit_reason", "execution_exit_hour", "execution_mfe_return_12h", "execution_mae_return_12h", "execution_cost_return"])
    clean = clean.rename(columns={"catboost_hard_ensemble_platt": "oof_clean_favorable_probability"})

    # Path CatBoost has the latest frozen-head availability boundary, so it
    # defines the strict action-layer intersection.  Alpha and clean are
    # expanding/checkpoint OOF inputs and must additionally be at decision.
    # The strict action table is the explicit intersection of independently
    # valid frozen inputs.  Record the reduction rather than silently filling
    # unavailable features or loosening a head's information boundary.
    input_rows = {"catboost_forward_complete": int(len(cat))}
    table = cat.merge(peak, on=ID, how="inner", validate="one_to_one")
    input_rows["after_peak_forward_complete"] = int(len(table))
    table = table.merge(alpha, on=ID, how="inner", validate="one_to_one")
    input_rows["after_alpha_context"] = int(len(table))
    table = table.merge(clean, on=ID, how="inner", validate="one_to_one")
    input_rows["after_clean_and_execution_labels"] = int(len(table))
    table["execution_decision_utc"] = pd.to_datetime(table["execution_decision_utc"], utc=True, errors="raise")
    for col in ("alpha_available_at", "peak_mfe_available_at", "catboost_available_at"):
        table[col] = pd.to_datetime(table[col], utc=True, errors="raise")
    table["joint_prediction_available_at"] = table[["alpha_available_at", "peak_mfe_available_at", "catboost_available_at"]].max(axis=1)
    table["action_layer_eligible"] = table["joint_prediction_available_at"].le(table["execution_decision_utc"])
    table = table.loc[table["action_layer_eligible"]].copy().reset_index(drop=True)
    input_rows["after_joint_available_at"] = int(len(table))
    # The clean extension is OOF/checkpoint data without a separate persisted
    # availability field; it is bound to its pre-entry decision timestamp.
    table["clean_probability_available_at"] = table["execution_decision_utc"]
    table["ev_input_origin"] = "frozen_final_refit_forward_intersection"
    table["is_oof"] = False
    table["promotion_eligible"] = False
    for column in raw:
        if column not in table:
            raise ValueError(f"winner raw input missing after exact joins: {column}")
        table[column] = pd.to_numeric(table[column], errors="raise")
    if table.loc[:, raw].isna().any().any():
        raise ValueError("winner raw inputs contain nulls")
    score = predict_execution_ev_model_ablation_bundle(
        bundle, table, algorithms=("catboost",), target_modes=("residual",),
        hpo_arms=("without_hpo",), arms=("all_features",),
    )
    if list(score.columns) != [WINNER]:
        raise AssertionError(f"unexpected frozen winner score columns: {score.columns.tolist()}")
    table["frozen_winner_raw_ev"] = score[WINNER].to_numpy(dtype=float)
    # The saved bundle's 21-day admission calibrator is OOF-time-dependent.
    # It is intentionally not refit/replayed here; raw final-fit score is
    # action-layer input, not a claim of post-calibrator promotion performance.
    table["admission_calibrator_status"] = "requires_causal_21d_history_not_refit_in_forward_builder"
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "strict_forward_winner_inputs_and_raw_scores.parquet"
    table.to_parquet(output, index=False, compression="zstd")
    payload = {
        "schema": "execution_ev_frozen_winner_forward_inputs_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "winner": WINNER,
        "raw_feature_columns": raw,
        "rows": int(len(table)),
        "input_intersection_rows": input_rows,
        "timestamp_bounds": {"min": str(table["__ts__"].min()), "max": str(table["__ts__"].max())},
        "strict_action_layer": {"all_heads_forward_only": True, "joint_available_at_lte_decision": True, "admission_calibrator": "not_refit; requires separate causal 21d forward state"},
        "sources": {name: {"path": str(path), "sha256": _sha256(path)} for name, path in {"alpha": args.alpha, "context": args.context, "peak": args.peak, "path_catboost": args.path_catboost, "clean": args.clean, "bundle": args.bundle}.items()},
        "output": {"path": str(output), "sha256": _sha256(output)},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    result = run(_parser().parse_args())
    print(json.dumps({"rows": result["rows"], "timestamp_bounds": result["timestamp_bounds"]}, indent=2))
