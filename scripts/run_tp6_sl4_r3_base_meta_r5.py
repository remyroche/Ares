#!/usr/bin/env python3
"""Strict chronological R3 base-opportunity plus residual-meta R5 screen.

The base learns the R3 clear/adverse/weak event.  The meta is fit only on
chronological OOF base probabilities and predicts exact-net residual around a
causal, prior-OOF score-to-bps map.  It is therefore not allowed to learn from
an in-sample base score or from the final evaluation outcomes.
"""
from __future__ import annotations

import argparse, gc, json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SIDES = ("long", "short")
BASE_PARAMS = dict(n_estimators=140, learning_rate=.05, num_leaves=31, min_child_samples=350,
                   subsample=.8, colsample_bytree=.8, reg_lambda=8., n_jobs=1, verbosity=-1)
META_PARAMS = dict(n_estimators=180, learning_rate=.035, num_leaves=15, min_child_samples=500,
                   subsample=.8, colsample_bytree=.8, reg_lambda=15., n_jobs=1, verbosity=-1)

# These are deliberately configured meta-only causal market/context fields,
# picked across volatility, breadth, OI, funding, correlation, transition and
# crash/recovery families.  A later MDA pass may reduce this list, but base
# fields are never silently reused here.
META_CONTEXT = [
    "mkt_ret_eq_24h", "regime_liquidity_score", "mkt_rv_ratio_1h_24h", "mkt_oi_chg_z_24h",
    "mkt_funding_dispersion", "cross_asset_corr_4h", "mkt_systemic_deleveraging_score",
    "mkt_flush_exhaustion_score", "post_liquidation_rebound_score", "negative_breadth_pct",
    "btc_resilience_alt_weakness", "short_covering_score_market", "deleveraging_without_followthrough",
    "short_signal_recovery_conflict", "market_state_transition_entropy_5d", "breakout_retention_4h",
]
OOF_FOLDS = (("2023-07-01", "2023-09-01"), ("2023-09-01", "2023-11-01"),
             ("2023-11-01", "2024-01-01"), ("2024-01-01", "2024-03-01"))


def _base_features(root: Path, side: str) -> list[str]:
    x = json.loads((root / side / "target_family_manifest.json").read_text())
    features = x["feature_contract"][f"T2_soft_barrier|tp3_sl2|{side}"]
    if not 30 <= len(features) <= 40:
        raise ValueError("unexpected frozen base feature contract")
    return features


def _matrix(frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return frame[cols].replace([np.inf, -np.inf], np.nan).fillna(0.).to_numpy(np.float32)


def _classes(frame: pd.DataFrame) -> np.ndarray:
    return np.select(
        [frame.robust_clear_event_b25.eq(1.), frame.lower_touch_minute.ge(0)], [2, 0], default=1,
    ).astype(np.int8)


def _weights(frame: pd.DataFrame, classes: np.ndarray) -> np.ndarray:
    agreement = frame[["robust_clear_event_b0", "robust_clear_event_b25", "robust_clear_event_b50"]].nunique(axis=1).eq(1).to_numpy(float)
    certainty = .5 + .5 * agreement
    counts = np.bincount(classes, minlength=3).astype(float)
    class_weight = np.sqrt(len(frame) / np.maximum(counts, 1.))[classes]
    class_weight /= class_weight.mean()
    weight = np.clip(certainty * class_weight, .25, 4.)
    return weight / weight.mean()


def _map_fit(score: np.ndarray, net: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.unique(np.quantile(score, np.linspace(0, 1, 11)))
    if len(edges) < 3:
        return np.array([-np.inf, np.inf]), np.array([float(np.mean(net))])
    bins = np.clip(np.digitize(score, edges[1:-1], right=True), 0, 9)
    means = np.array([net[bins == i].mean() if (bins == i).any() else net.mean() for i in range(10)])
    return edges, means


def _map_apply(score: np.ndarray, mapping: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    edges, means = mapping
    return means[np.clip(np.digitize(score, edges[1:-1], right=True), 0, len(means)-1)]


def _read_side(panel: Path, winner: Path, robust: Path, side: str, base: list[str], cutoff: pd.Timestamp | None = None, start: pd.Timestamp | None = None) -> pd.DataFrame:
    identity = ["candidate_id", "__ts__", "side_name", *base, *META_CONTEXT]
    winner_cols = ["candidate_id", "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps", "__label_available_at__"]
    robust_cols = ["candidate_id", "label_valid", "lower_touch_minute", "robust_clear_event_b0", "robust_clear_event_b25", "robust_clear_event_b50"]
    pieces = []
    for part in sorted((panel / "parts").glob("*.parquet")):
        x = pd.read_parquet(part, columns=identity)
        x = x.loc[x.side_name.eq(side)]
        if cutoff is not None:
            x = x.loc[pd.to_datetime(x.__ts__, utc=True).lt(cutoff)]
        if start is not None:
            x = x.loc[pd.to_datetime(x.__ts__, utc=True).ge(start)]
        if x.empty:
            continue
        w = pd.read_parquet(winner / "parts" / part.name, columns=winner_cols)
        r = pd.read_parquet(robust / "parts" / part.name, columns=robust_cols)
        x = x.merge(w, on="candidate_id", how="inner", validate="one_to_one").merge(r, on="candidate_id", how="left", validate="one_to_one")
        pieces.append(x.loc[x.label_valid.eq(True)])
    out = pd.concat(pieces, ignore_index=True)
    del pieces
    numeric = [*base, *META_CONTEXT]
    out[numeric] = out[numeric].astype(np.float32)
    out["__ts__"] = pd.to_datetime(out.__ts__, utc=True)
    out["__label_available_at__"] = pd.to_datetime(out.__label_available_at__, utc=True)
    if not np.allclose(out.t4_tp6_sl4_gross_bps - 100., out.t4_tp6_sl4_net_bps, atol=2e-3):
        raise ValueError("cost contract mismatch")
    return out


def _base_probability(model: lgb.LGBMClassifier, frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    p = model.predict_proba(_matrix(frame, cols))
    return p


def _context_for_ids(panel: Path, ids: set[str]) -> pd.DataFrame:
    pieces = []
    for part in sorted((panel / "parts").glob("*.parquet")):
        x = pd.read_parquet(part, columns=["candidate_id", *META_CONTEXT])
        x = x.loc[x.candidate_id.isin(ids)]
        if not x.empty:
            pieces.append(x)
    return pd.concat(pieces, ignore_index=True)


def _final_from_oof(panel: Path, winner: Path, robust: Path, side: str, base_cols: list[str], oof_paths: list[Path], evaluation_start: pd.Timestamp, evaluation_end: pd.Timestamp | None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Fit R5 from separately materialised chronological base-OOF folds."""
    oof = pd.concat([pd.read_parquet(path) for path in oof_paths], ignore_index=True)
    oof = oof.sort_values(["fold", "candidate_id"], kind="mergesort").reset_index(drop=True)
    mapped = []
    for number, chunk in oof.groupby("fold", observed=True):
        history = oof.loc[oof.fold.lt(number)]
        if history.empty:
            continue
        part = chunk.copy()
        part["base_expected_bps"] = _map_apply(part.base_raw.to_numpy(float), _map_fit(history.base_raw.to_numpy(float), history.net_bps.to_numpy(float)))
        mapped.append(part)
    meta_train = pd.concat(mapped, ignore_index=True)
    context = _context_for_ids(panel, set(meta_train.candidate_id))
    meta_train = meta_train.merge(context, on="candidate_id", how="inner", validate="one_to_one")
    meta_train["residual_target"] = meta_train.net_bps - meta_train.base_expected_bps
    meta_features = ["prob_adverse", "prob_weak", "prob_clear", "base_expected_bps", *META_CONTEXT]
    meta_model = lgb.LGBMRegressor(objective="huber", alpha=.9, random_state=20260901, **META_PARAMS)
    meta_model.fit(_matrix(meta_train, meta_features), meta_train.residual_target.to_numpy(float))
    del context, meta_train
    gc.collect()
    boundary = pd.Timestamp("2024-03-01", tz="UTC")
    base_train = _read_side(panel, winner, robust, side, base_cols, cutoff=boundary)
    labels = base_train[["robust_clear_event_b25", "lower_touch_minute"]]
    y = _classes(labels)
    weight_input = base_train[["robust_clear_event_b0", "robust_clear_event_b25", "robust_clear_event_b50"]]
    base_model = lgb.LGBMClassifier(objective="multiclass", num_class=3, random_state=20260900, **BASE_PARAMS)
    base_model.fit(_matrix(base_train, base_cols), y, sample_weight=_weights(weight_input, y))
    del base_train, labels, weight_input, y
    gc.collect()
    final = _read_side(panel, winner, robust, side, base_cols, start=evaluation_start, cutoff=evaluation_end)
    p = _base_probability(base_model, final, base_cols)
    result = final[["candidate_id", "__ts__", "side_name", "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps", *META_CONTEXT]].copy()
    result.columns = ["candidate_id", "__ts__", "side_name", "gross_bps", "net_bps", *META_CONTEXT]
    result["prob_adverse"] = p[:, 0]; result["prob_weak"] = p[:, 1]; result["prob_clear"] = p[:, 2]; result["base_raw"] = p[:, 2] - p[:, 0]
    result["base_expected_bps"] = _map_apply(result.base_raw.to_numpy(float), _map_fit(oof.base_raw.to_numpy(float), oof.net_bps.to_numpy(float)))
    result["meta_residual_bps"] = meta_model.predict(_matrix(result, meta_features))
    result["score_base_bps"] = result.base_expected_bps
    result["score_base_meta_bps"] = result.base_expected_bps + result.meta_residual_bps
    lineage = {"side": side, "base_features": base_cols, "meta_context_features": META_CONTEXT, "meta_features": meta_features,
               "oof_folds": OOF_FOLDS, "oof_rows": len(oof), "meta_train_rows": int(len(oof.loc[oof.fold.gt(0)])),
               "final_base_train_end": "2024-03-01", "evaluation_start": str(evaluation_start), "evaluation_end": None if evaluation_end is None else str(evaluation_end),
               "base_output_is_same_side_raw_probabilities": True,
               "meta_target": "exact net minus prior-OOF causal expected-bps map"}
    return oof, result, lineage


def _screen_side(frame: pd.DataFrame, base_cols: list[str], side: str, stop_after_oof: bool = False, only_fold: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, object]]:
    oof = []
    for number, (start_text, end_text) in enumerate(OOF_FOLDS):
        if only_fold is not None and number != only_fold:
            continue
        start, end = pd.Timestamp(start_text, tz="UTC"), pd.Timestamp(end_text, tz="UTC")
        train_mask = frame.__label_available_at__.lt(start)
        held_mask = frame.__ts__.ge(start) & frame.__ts__.lt(end)
        if int(train_mask.sum()) < 50_000 or int(held_mask.sum()) < 10_000:
            raise ValueError(f"insufficient OOF support {side} fold {number}")
        train_label = frame.loc[train_mask, ["robust_clear_event_b25", "lower_touch_minute"]]
        y = _classes(train_label)
        train_weight_input = frame.loc[train_mask, ["robust_clear_event_b0", "robust_clear_event_b25", "robust_clear_event_b50"]]
        model = lgb.LGBMClassifier(objective="multiclass", num_class=3, random_state=20260810 + number, **BASE_PARAMS)
        model.fit(_matrix(frame.loc[train_mask, base_cols], base_cols), y, sample_weight=_weights(train_weight_input, y))
        p = _base_probability(model, frame.loc[held_mask, base_cols], base_cols)
        result = frame.loc[held_mask, ["candidate_id", "__ts__", "side_name", "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps"]].copy()
        result.columns = ["candidate_id", "__ts__", "side_name", "gross_bps", "net_bps"]
        result["fold"] = number; result["prob_adverse"] = p[:, 0]; result["prob_weak"] = p[:, 1]; result["prob_clear"] = p[:, 2]
        result["base_raw"] = p[:, 2] - p[:, 0]
        oof.append(result)
        del train_label, train_weight_input, model, p, result
        gc.collect()
    oof = pd.concat(oof, ignore_index=True)
    if only_fold is not None:
        return oof, None, {"side": side, "base_features": base_cols, "meta_context_features": META_CONTEXT,
                           "oof_folds": [OOF_FOLDS[only_fold]], "oof_rows": len(oof), "meta_train_rows": 0,
                           "staged": "single chronological base OOF fold only"}
    # Each OOF fold's expected bps map only observes earlier resolved OOF rows.
    mapped = []
    for number, chunk in oof.groupby("fold", observed=True):
        history = oof.loc[oof.fold.lt(number)]
        if history.empty:
            continue
        part = chunk.copy(); part["base_expected_bps"] = _map_apply(part.base_raw.to_numpy(float), _map_fit(history.base_raw.to_numpy(float), history.net_bps.to_numpy(float)))
        mapped.append(part)
    meta_train = pd.concat(mapped, ignore_index=True)
    meta_train = meta_train.merge(frame[["candidate_id", *META_CONTEXT]], on="candidate_id", how="inner", validate="one_to_one")
    meta_train["residual_target"] = meta_train.net_bps - meta_train.base_expected_bps
    if stop_after_oof:
        return oof, None, {"side": side, "base_features": base_cols, "meta_context_features": META_CONTEXT,
                           "oof_folds": OOF_FOLDS, "oof_rows": len(oof), "meta_train_rows": len(meta_train),
                           "staged": "base OOF only; no evaluation model or score was fit"}
    # Final base has no evaluation outcomes in its training set.
    boundary = pd.Timestamp("2024-03-01", tz="UTC")
    base_train_mask = frame.__label_available_at__.lt(boundary)
    evaluation_mask = frame.__ts__.ge(boundary)
    base_train_label = frame.loc[base_train_mask, ["robust_clear_event_b25", "lower_touch_minute"]]
    y = _classes(base_train_label)
    base_train_weights = frame.loc[base_train_mask, ["robust_clear_event_b0", "robust_clear_event_b25", "robust_clear_event_b50"]]
    base_model = lgb.LGBMClassifier(objective="multiclass", num_class=3, random_state=20260900, **BASE_PARAMS)
    base_model.fit(_matrix(frame.loc[base_train_mask, base_cols], base_cols), y, sample_weight=_weights(base_train_weights, y))
    p = _base_probability(base_model, frame.loc[evaluation_mask, base_cols], base_cols)
    final = frame.loc[evaluation_mask, ["candidate_id", "__ts__", "side_name", "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps", *META_CONTEXT]].copy()
    final.columns = ["candidate_id", "__ts__", "side_name", "gross_bps", "net_bps", *META_CONTEXT]
    final["prob_adverse"] = p[:, 0]; final["prob_weak"] = p[:, 1]; final["prob_clear"] = p[:, 2]; final["base_raw"] = p[:, 2] - p[:, 0]
    base_map = _map_fit(oof.base_raw.to_numpy(float), oof.net_bps.to_numpy(float))
    final["base_expected_bps"] = _map_apply(final.base_raw.to_numpy(float), base_map)
    meta_features = ["prob_adverse", "prob_weak", "prob_clear", "base_expected_bps", *META_CONTEXT]
    # Per-row residual training; the base outputs are strict OOF for every row.
    meta_model = lgb.LGBMRegressor(objective="huber", alpha=.9, random_state=20260901, **META_PARAMS)
    meta_model.fit(_matrix(meta_train, meta_features), meta_train.residual_target.to_numpy(float))
    final["meta_residual_bps"] = meta_model.predict(_matrix(final, meta_features))
    final["score_base_bps"] = final.base_expected_bps
    final["score_base_meta_bps"] = final.base_expected_bps + final.meta_residual_bps
    lineage = {"side": side, "base_features": base_cols, "meta_context_features": META_CONTEXT,
               "meta_features": meta_features, "oof_folds": OOF_FOLDS, "oof_rows": len(oof),
               "meta_train_rows": len(meta_train), "final_base_train_end": "2024-03-01",
               "evaluation_start": "2024-03-01", "base_output_is_same_side_raw_probabilities": True,
               "meta_target": "realised exact-net bps minus prior-OOF causal base expected-bps map"}
    return oof, final, lineage


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, default=ROOT / "data_perp/artifacts/full_universe_t2_t4_panel_20260801_v3")
    p.add_argument("--winner", type=Path, default=ROOT / "data_perp/artifacts/full_universe_tp6_sl4_h12_sidecar_20260802_v1")
    p.add_argument("--robust", type=Path, default=ROOT / "data_perp/artifacts/tp6_sl4_robust_clear_labels_20260802_v1")
    p.add_argument("--features", type=Path, default=ROOT / "data_perp/artifacts/full_universe_base_hpo_20260802_v1")
    p.add_argument("--side", choices=SIDES, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stop-after-oof", action="store_true", help="materialise chronological base OOF only")
    p.add_argument("--only-fold", type=int, choices=range(len(OOF_FOLDS)), help="run one OOF fold in an isolated process")
    p.add_argument("--final-from-oof", type=Path, nargs="+", help="run final R5 from isolated base-OOF prediction files")
    p.add_argument("--evaluation-end", help="exclusive evaluation end; use monthly shards to cap memory")
    p.add_argument("--evaluation-start", default="2024-03-01", help="inclusive evaluation start for a streamed shard")
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    base = _base_features(args.features, args.side)
    if args.final_from_oof:
        if args.stop_after_oof or args.only_fold is not None:
            raise ValueError("--final-from-oof cannot be combined with OOF-only options")
        evaluation_start = pd.Timestamp(args.evaluation_start, tz="UTC")
        evaluation_end = None if args.evaluation_end is None else pd.Timestamp(args.evaluation_end, tz="UTC")
        oof, final, lineage = _final_from_oof(args.panel, args.winner, args.robust, args.side, base, args.final_from_oof, evaluation_start, evaluation_end)
    else:
        cutoff = pd.Timestamp(OOF_FOLDS[args.only_fold][1] if args.only_fold is not None else "2024-03-01", tz="UTC") if args.stop_after_oof else None
        frame = _read_side(args.panel, args.winner, args.robust, args.side, base, cutoff=cutoff)
        oof, final, lineage = _screen_side(frame, base, args.side, stop_after_oof=args.stop_after_oof, only_fold=args.only_fold)
    args.out.mkdir(parents=True)
    oof.to_parquet(args.out / "base_oof_predictions.parquet", index=False)
    if final is not None:
        final.to_parquet(args.out / "base_meta_oos_predictions.parquet", index=False)
    manifest = {"schema": "tp6_sl4_r3_base_meta_r5_v1", "status": "COMPLETED", "contract": {
        "geometry": "selected TP=+6 ATR / SL=-4 ATR / H12", "cost_bps": 100,
        "base_target": "R3 robust-clear b25 / adverse-first / weak-unresolved",
        "meta": "strict OOF base outputs, same-side raw probabilities plus meta-only context, Huber exact-net residual",
        "no_final_evaluation_labels_in_training": True}, "lineage": lineage}
    (args.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=lambda v: v.item() if hasattr(v, "item") else str(v)) + "\n")
    print(json.dumps({"side": args.side, "oof_rows": len(oof), "evaluation_rows": 0 if final is None else len(final), "meta_train_rows": lineage["meta_train_rows"]}))


if __name__ == "__main__":
    main()
