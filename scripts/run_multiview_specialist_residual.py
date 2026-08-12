#!/usr/bin/env python3
"""Train strict-OOF diverse multi-view specialists and a bounded residual meta.

The specialist router is fitted from calibration-period, opportunity-conditioned
co-activation and joint synergy.  It never selects views from unconditional
specialist-score Spearman correlation.  Every specialist prediction supplied
to the residual meta model was produced by a model fitted before that row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import sys

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.multiview_specialists import (  # noqa: E402
    SynergyConfig, apply_synergy_features, discover_opportunity_views,
    is_permitted_feature, opportunity_conditioned_synergy,
)
from scripts.run_market_spine_covariance_meta import LONG_HISTORY_FOLDS, ContinuousFold, _utc  # noqa: E402


LEDGER = ROOT / "data_perp/artifacts/tp6_m6_shared_regime_residual_features_20260809_v3/shared_regime_residual_ledger.parquet"
STORE = ROOT / "data_perp/artifacts/full_universe_t2_t4_panel_20260801_v3/parts/*.parquet"
DEFAULT_OUT = ROOT / "data_perp/artifacts/multiview_specialist_residual_20260810_v1"
LABEL_DELAY = pd.Timedelta(hours=13)
TAILS = (.01, .05, .10)
SEED = 20260810
# Bounded proxy prevents a full 780-column parquet materialisation from
# multiplying memory during every chronological fold.  View discovery remains
# data-driven inside this broad, deterministic observable sample.
MAX_DISCOVERY_PROXY_FEATURES = 80
MAX_SPECIALIST_TRAIN_ROWS = 150_000
MAX_META_CORRECTION_BPS = 50.0


def _quoted(column: str) -> str:
    return '"' + str(column).replace('"', '""') + '"'


def _feature_schema() -> list[str]:
    con = duckdb.connect()
    try:
        rows = con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(STORE)]).fetchall()
        return [str(row[0]) for row in rows]
    finally:
        con.close()


def _discovery_candidates(schema: list[str], excluded: set[str]) -> list[str]:
    """Broad deterministic observable proxy; view membership is learned later."""
    fields = [field for field in schema if field not in excluded and is_permitted_feature(field)]
    fields.sort(key=lambda field: hashlib.blake2b(field.encode(), digest_size=8, person=b"mv-disc").hexdigest())
    return fields[:MAX_DISCOVERY_PROXY_FEATURES]


def load_panel() -> tuple[pd.DataFrame, list[str]]:
    """Join a bounded, diverse observable feature pool by candidate identity."""
    base_columns = [
        "candidate_id", "__ts__", "side_name", "event", "net_bps", "gross_bps",
        "p_clear", "p_adverse", "p_weak", "prequential_base_expected_net_bps",
        "shared_regime_contract_complete",
    ]
    chosen = _discovery_candidates(_feature_schema(), set(base_columns) | {"__decision_ts__", "__symbol__"})
    con = duckdb.connect()
    try:
        select_store = ", ".join(f"s.{_quoted(column)}" for column in chosen)
        select_ledger = ", ".join(f"l.{_quoted(column)}" for column in base_columns)
        query = f"""
            SELECT {select_ledger}, {select_store}
            FROM read_parquet(?) l
            INNER JOIN read_parquet(?) s USING (candidate_id)
            WHERE l.shared_regime_contract_complete
        """
        frame = con.execute(query, [str(LEDGER), str(STORE)]).fetchdf()
    finally:
        con.close()
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    frame["label_available_ts"] = frame["__ts__"] + LABEL_DELAY
    numeric = ["event", "net_bps", "gross_bps", "p_clear", "p_adverse", "p_weak", "prequential_base_expected_net_bps", *chosen]
    # DuckDB returns doubles by default.  Store features are only specialist
    # inputs, so float32 halves the persistent panel footprint.
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    frame["base_score"] = frame["p_clear"] - .5 * frame["p_adverse"]
    required = ["event", "net_bps", "gross_bps", "p_clear", "p_adverse", "p_weak", "prequential_base_expected_net_bps", "base_score"]
    frame = frame.loc[np.isfinite(frame.loc[:, required].to_numpy(float)).all(axis=1)].copy()
    if not np.allclose(frame["gross_bps"] - frame["net_bps"], 100., atol=.02):
        raise ValueError("the TP6/SL4 fixed 100-bps cost contract failed")
    return frame.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True), chosen


def _specialist_model() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=180, learning_rate=.04, num_leaves=20,
        min_child_samples=500, colsample_bytree=.75, subsample=.80, subsample_freq=1,
        reg_lambda=20., random_state=SEED, n_jobs=1, verbosity=-1,
    )


def _meta_model() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="huber", alpha=.90, n_estimators=100, learning_rate=.035,
        num_leaves=12, min_child_samples=600, colsample_bytree=.80,
        reg_lambda=35., random_state=SEED, n_jobs=1, verbosity=-1,
    )


def _metrics(frame: pd.DataFrame, score: np.ndarray, fold: ContinuousFold, arm: str) -> list[dict[str, object]]:
    data = frame.copy(); data["score"] = score
    rows: list[dict[str, object]] = []
    for side, piece in [("pooled", data), *((side, q) for side, q in data.groupby("side_name", observed=True, sort=True))]:
        for tail in TAILS:
            n = max(1, int(np.ceil(len(piece) * tail)))
            selected = piece.sort_values(["score", "candidate_id"], ascending=[False, True], kind="stable").head(n)
            rows.append({"fold": fold.name, "fold_family": fold.family, "side": side, "arm": arm, "tail": tail, "rows": len(piece), "tail_rows": n, "net_bps": float(selected.net_bps.mean()), "gross_bps": float(selected.gross_bps.mean()), "rank_ic": float(piece.score.rank().corr(piece.net_bps.rank()))})
    return rows


def _fit_fold(panel: pd.DataFrame, candidates: list[str], fold: ContinuousFold) -> tuple[list[dict[str, object]], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_start, cal_start, test_start, test_end = map(_utc, (fold.train_start, fold.calibration_start, fold.test_start, fold.test_end))
    train = panel.loc[panel.__ts__.between(train_start, cal_start, inclusive="left") & panel.label_available_ts.lt(cal_start)].copy()
    calibration = panel.loc[panel.__ts__.between(cal_start, test_start, inclusive="left") & panel.label_available_ts.lt(test_start)].copy()
    test = panel.loc[panel.__ts__.between(test_start, test_end, inclusive="left")].copy()
    if min(map(len, (train, calibration, test))) == 0:
        raise ValueError(f"empty strict split: {fold.name}")
    all_metrics = _metrics(test, test.base_score.to_numpy(float), fold, "B0_r3_base_score")
    out = test.loc[:, ["candidate_id", "__ts__", "side_name", "net_bps", "gross_bps", "base_score", "prequential_base_expected_net_bps"]].copy()
    out["fold"] = fold.name
    view_audit: list[pd.DataFrame] = []
    synergy_audit: list[pd.DataFrame] = []
    importance: list[dict[str, object]] = []
    specialist_test = pd.DataFrame(index=test.index)
    specialist_cal = pd.DataFrame(index=calibration.index)
    for side in ("long", "short"):
        tr = train.loc[train.side_name.eq(side)].copy(); ca = calibration.loc[calibration.side_name.eq(side)].copy(); te = test.loc[test.side_name.eq(side)].copy()
        if min(map(len, (tr, ca, te))) < 2_000 or tr.event.nunique() < 2:
            raise ValueError(f"insufficient side support for {fold.name}/{side}")
        if len(tr) > MAX_SPECIALIST_TRAIN_ROWS:
            fit_rows = tr.sample(MAX_SPECIALIST_TRAIN_ROWS, random_state=SEED, replace=False).sort_values(["__ts__", "candidate_id"], kind="stable")
        else:
            fit_rows = tr
        view_features, discovery, discovery_edges = discover_opportunity_views(
            fit_rows, candidates, base_score_column="base_score", label_column="event",
        )
        discovery["fold"], discovery["side"], discovery["audit_kind"] = fold.name, side, "feature_activation"
        discovery_edges["fold"], discovery_edges["side"], discovery_edges["audit_kind"] = fold.name, side, "feature_joint_synergy"
        view_audit.extend((discovery, discovery_edges))
        cal_scores: dict[str, str] = {}; test_scores: dict[str, str] = {}
        for view, fields in view_features.items():
            # Median fill values are fit solely on the specialist's train rows.
            medians = fit_rows.loc[:, fields].median(numeric_only=True)
            x_train = fit_rows.loc[:, fields].fillna(medians); x_cal = ca.loc[:, fields].fillna(medians); x_test = te.loc[:, fields].fillna(medians)
            model = _specialist_model().fit(x_train, fit_rows.event.to_numpy(int))
            name = f"mv_score__{view}"
            ca.loc[:, name] = model.predict_proba(x_cal)[:, 1]
            te.loc[:, name] = model.predict_proba(x_test)[:, 1]
            cal_scores[view] = name; test_scores[view] = name
            for feature, gain in zip(fields, model.booster_.feature_importance(importance_type="gain"), strict=True):
                importance.append({"fold": fold.name, "side": side, "view": view, "feature": feature, "gain": float(gain)})
        if len(cal_scores) < 2:
            raise ValueError(f"fewer than two usable specialists for {fold.name}/{side}")
        synergies, cal_pairs = opportunity_conditioned_synergy(ca, cal_scores, base_score_column="base_score", label_column="event", config=SynergyConfig())
        synergies["fold"], synergies["side"], synergies["audit_kind"] = fold.name, side, "specialist_score_joint_synergy"
        synergy_audit.append(synergies)
        test_pairs = apply_synergy_features(te, test_scores, synergies, base_score_column="base_score")
        meta_features = ["p_clear", "p_adverse", "p_weak", "base_score", "prequential_base_expected_net_bps", *cal_scores.values(), *cal_pairs.columns]
        # Pair columns are structurally identical because test application uses
        # only train/calibration-frozen selected rows.  Absent pairs are zeros.
        for field in cal_pairs.columns:
            ca.loc[:, field] = cal_pairs[field]
        for field in cal_pairs.columns:
            te.loc[:, field] = test_pairs[field].to_numpy(float) if field in test_pairs else 0.
        meta_train = ca.loc[:, meta_features].fillna(0.)
        target = (ca.net_bps - ca.prequential_base_expected_net_bps).clip(-300., 300.)
        meta = _meta_model().fit(meta_train, target.to_numpy(float))
        correction = np.clip(meta.predict(te.loc[:, meta_features].fillna(0.)), -MAX_META_CORRECTION_BPS, MAX_META_CORRECTION_BPS)
        te.loc[:, "mv_residual_correction_bps"] = correction
        test.loc[te.index, "mv_residual_correction_bps"] = correction
        for name in test_scores:
            specialist_test.loc[te.index, test_scores[name]] = te.loc[:, test_scores[name]]
        for name in cal_scores:
            specialist_cal.loc[ca.index, cal_scores[name]] = ca.loc[:, cal_scores[name]]
    adjusted = test.prequential_base_expected_net_bps.to_numpy(float) + test.mv_residual_correction_bps.fillna(0.).to_numpy(float)
    all_metrics.extend(_metrics(test, adjusted, fold, "MV_specialists_bounded_huber_residual"))
    out["mv_residual_correction_bps"] = test.mv_residual_correction_bps.fillna(0.).to_numpy(float)
    out["mv_expected_net_bps"] = adjusted
    for column in specialist_test:
        out[column] = specialist_test[column].to_numpy(float)
    return all_metrics, out, pd.concat(view_audit, ignore_index=True), pd.concat(synergy_audit, ignore_index=True), pd.DataFrame(importance)


def run(out: Path = DEFAULT_OUT, *, folds: tuple[ContinuousFold, ...] = LONG_HISTORY_FOLDS[3:]) -> Path:
    panel, candidates = load_panel()
    out.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, object]] = []; predictions: list[pd.DataFrame] = []; selections: list[pd.DataFrame] = []; synergies: list[pd.DataFrame] = []; importance: list[pd.DataFrame] = []
    for fold in folds:
        fold_metrics, fold_predictions, fold_selection, fold_synergy, fold_importance = _fit_fold(panel, candidates, fold)
        metrics.extend(fold_metrics); predictions.append(fold_predictions); selections.append(fold_selection); synergies.append(fold_synergy); importance.append(fold_importance)
    pd.DataFrame(metrics).to_parquet(out / "metrics.parquet", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(out / "predictions.parquet", index=False)
    pd.concat(selections, ignore_index=True).to_parquet(out / "view_feature_selection.parquet", index=False)
    pd.concat(synergies, ignore_index=True).to_parquet(out / "opportunity_synergy_routing.parquet", index=False)
    pd.concat(importance, ignore_index=True).to_parquet(out / "specialist_feature_importance.parquet", index=False)
    manifest = {"schema": "multiview_specialist_residual_v2_data_discovered", "status": "COMPLETED_DIAGNOSTIC_NO_PROMOTION", "input_ledger": str(LEDGER), "feature_store": str(STORE), "geometry": "TP6/SL4/H12", "cost_bps": 100., "folds": [asdict(fold) for fold in folds], "views": "train-fold data-discovered from opportunity-conditioned feature co-activation and joint synergy; semantic names are not router inputs", "feature_diversity": "disjoint raw fields; within-view Spearman is only a secondary redundancy veto", "routing": "calibration-only opportunity-conditioned co-activation and joint synergy; unconditional Spearman is not a routing criterion", "meta": "all available strict-OOF specialist scores plus selected synergy features feed a side-local bounded Huber residual", "correction_cap_bps": MAX_META_CORRECTION_BPS}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--all-folds", action="store_true", help="also run the three earlier 2023/24 folds")
    args = parser.parse_args()
    print(run(args.out, folds=LONG_HISTORY_FOLDS if args.all_folds else LONG_HISTORY_FOLDS[3:]))


if __name__ == "__main__":
    main()
