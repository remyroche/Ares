#!/usr/bin/env python3
"""Full-meta-feature three-class base-error transport ablation.

For each side and chronological held-out environment, the residual of the
same-side frozen base expected-net score is split into *training* terciles:
base overestimate / approximately correct / base underestimate.  The model is
then allowed the full configured causal meta universe, screened inside that
training fold only.  It corrects the base in common bps; final selection is
one global book after that reconstruction.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from extreme_price_movements.base_error_tercile_meta import (  # noqa: E402
    expected_base_error_bps, fit_base_error_tercile_map, labels_from_base_error,
)
from extreme_price_movements.config import CFG  # noqa: E402
from extreme_price_movements.transport_supervised_archetypes import (  # noqa: E402
    configured_available_meta_features, training_univariate_screen,
)

LEDGER = ROOT / "data_perp/artifacts/tp6_m6_shared_regime_residual_features_20260809_v3/shared_regime_residual_ledger.parquet"
PANEL = ROOT / "data_perp/artifacts/full_universe_t2_t4_panel_20260801_v3/parts/*.parquet"
OUT = ROOT / "data_perp/artifacts/transport_base_error_tercile_20260803_v1"
REQUIRED = (
    "p_adverse", "p_weak", "p_clear", "prequential_base_expected_net_bps",
    "base_raw", "cost_to_atr", "base_entropy", "base_top2_margin",
)


def _folds(timestamp: pd.Series) -> pd.Series:
    """Five chronological environments that never split a decision bar."""
    values = pd.Index(timestamp.drop_duplicates().sort_values())
    lookup = {
        item: min(4, int(5 * position / max(len(values), 1)))
        for position, item in enumerate(values)
    }
    return timestamp.map(lookup).astype(np.int8)


def _matrix(train: pd.DataFrame, score: pd.DataFrame, fields: list[str]) -> tuple[np.ndarray, np.ndarray]:
    median = train.loc[:, fields].replace([np.inf, -np.inf], np.nan).median().fillna(0.)
    x_train = train.loc[:, fields].replace([np.inf, -np.inf], np.nan).fillna(median).to_numpy(np.float32)
    x_score = score.loc[:, fields].replace([np.inf, -np.inf], np.nan).fillna(median).to_numpy(np.float32)
    return x_train, x_score


def _tail_rows(frame: pd.DataFrame, score: str, *, fold: int, arm: str) -> list[dict[str, object]]:
    result = []
    ranked = frame.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable")
    for fraction in (.01, .05, .10):
        selection = ranked.head(max(1, int(np.ceil(len(ranked) * fraction))))
        for side, part in (("global", selection), ("long", selection[selection.side_name.eq("long")]), ("short", selection[selection.side_name.eq("short")])):
            if len(part):
                result.append({"fold": fold, "arm": arm, "scope": side, "top_fraction": fraction, "rows": len(part), "net_bps": float(part.net_bps.mean()), "gross_bps": float(part.gross_bps.mean()), "total_net_bps": float(part.net_bps.sum()), "long_share": float(selection.side_name.eq("long").mean())})
    return result


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(config={"threads": "2", "memory_limit": "512MB", "temp_directory": "/tmp"})
    panel_columns = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{PANEL.as_posix()}') LIMIT 1").fetchdf().column_name.tolist()
    available = configured_available_meta_features(CFG, panel_columns)
    meta_select = ", ".join(f'p."{name}"' for name in available)
    query = f'''SELECT l.candidate_id,l.__ts__,l.side_name,l.net_bps,l.gross_bps,l.base_raw,l.p_adverse,l.p_weak,l.p_clear,l.prequential_base_expected_net_bps,
    p."atr_1h",p."decision_price",p."assumed_round_trip_cost_bps",{meta_select}
    FROM read_parquet('{LEDGER.as_posix()}') l JOIN read_parquet('{PANEL.as_posix()}') p USING(candidate_id)
    WHERE l.shared_regime_contract_complete AND l.prequential_base_expected_net_bps IS NOT NULL AND abs(hash(l.candidate_id)) % 5 = 0'''
    frame = con.execute(query).fetchdf()
    con.close()
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
    frame = frame.sort_values("__ts__", kind="stable").reset_index(drop=True)
    atr_bps = np.abs(frame.atr_1h.to_numpy(float)) / np.maximum(np.abs(frame.decision_price.to_numpy(float)), 1e-12) * 1e4
    frame["cost_to_atr"] = np.clip(frame.assumed_round_trip_cost_bps.to_numpy(float) / np.maximum(atr_bps, 1e-6), 0., 100.)
    probability = frame[["p_adverse", "p_weak", "p_clear"]].to_numpy(float)
    probability /= np.maximum(probability.sum(axis=1, keepdims=True), 1e-12)
    frame[["p_adverse", "p_weak", "p_clear"]] = probability
    frame["base_entropy"] = -(probability * np.log(np.maximum(probability, 1e-12))).sum(axis=1)
    ordered_probability = np.sort(probability, axis=1)
    frame["base_top2_margin"] = ordered_probability[:, -1] - ordered_probability[:, -2]
    frame["fold"] = _folds(frame["__ts__"])
    coverage = 1. - frame.loc[:, available].isna().mean()
    usable = coverage[coverage.ge(.90)].index.tolist()
    pd.DataFrame({"feature": available, "coverage": coverage.reindex(available), "usable": pd.Index(available).isin(usable)}).to_parquet(OUT / "base_error_meta_feature_coverage.parquet", index=False)
    rows, metrics, selections = [], [], []
    for fold in (2, 3, 4):
        test = frame.loc[frame.fold.eq(fold)].copy()
        test_start = test.__ts__.min()
        train = frame.loc[frame.__ts__.lt(test_start - pd.Timedelta(hours=13))].copy()
        for side in ("long", "short"):
            local_train = train.loc[train.side_name.eq(side)].copy()
            local_test = test.loc[test.side_name.eq(side)].copy()
            # Fit both side-local mappings from the same earlier-only training
            # population.  The current side consumes only its own boundaries
            # and shrunk class means; passing the pooled frame simply makes
            # the explicit map contract complete.
            mapping = fit_base_error_tercile_map(train, shrinkage_support=1_000.)
            label = labels_from_base_error(local_train, mapping)
            selected = training_univariate_screen(local_train, usable, label.astype(float), maximum=64)
            fields = list(dict.fromkeys([*REQUIRED, *selected]))
            x_train, x_test = _matrix(local_train, local_test, fields)
            count = np.bincount(label, minlength=3).astype(float)
            weight = np.sqrt(len(label) / np.maximum(3. * count[label], 1.))
            weight = np.clip(weight / weight.mean(), .5, 2.)
            model = lgb.LGBMClassifier(
                objective="multiclass", num_class=3, n_estimators=180,
                learning_rate=.035, num_leaves=24, min_child_samples=750,
                colsample_bytree=.8, reg_lambda=30., random_state=20260803 + fold,
                n_jobs=1, verbosity=-1,
            ).fit(x_train, label, sample_weight=weight)
            p = np.clip(model.predict_proba(x_test), 1e-6, 1.)
            p /= p.sum(axis=1, keepdims=True)
            observed = labels_from_base_error(local_test, mapping)
            correction = expected_base_error_bps(p, local_test.side_name, mapping)
            local_test["base_error_overestimate_probability"] = p[:, 0]
            local_test["base_error_approximately_correct_probability"] = p[:, 1]
            local_test["base_error_underestimate_probability"] = p[:, 2]
            local_test["tercile_meta_correction_bps"] = correction
            local_test["tercile_meta_score_bps"] = local_test.prequential_base_expected_net_bps.to_numpy(float) + correction
            rows.append(local_test)
            metrics.append({"fold": fold, "side_name": side, "train_rows": len(local_train), "test_rows": len(local_test), "selected_feature_count": len(fields), "selected_context_features": selected, "lower_tercile_bps": mapping.edges_by_side[side][0], "upper_tercile_bps": mapping.edges_by_side[side][1], "test_log_loss": float(log_loss(observed, p, labels=[0, 1, 2])), "test_accuracy": float(accuracy_score(observed, p.argmax(axis=1)))})
    prediction = pd.concat(rows, ignore_index=True)
    for fold, part in prediction.groupby("fold", observed=True):
        selections.extend(_tail_rows(part, "prequential_base_expected_net_bps", fold=int(fold), arm="frozen_base"))
        selections.extend(_tail_rows(part, "tercile_meta_score_bps", fold=int(fold), arm="full_meta_tercile"))
    prediction.to_parquet(OUT / "base_error_tercile_oof_predictions.parquet", index=False)
    pd.DataFrame(metrics).to_parquet(OUT / "base_error_tercile_classifier_metrics.parquet", index=False)
    pd.DataFrame(selections).to_parquet(OUT / "base_error_tercile_economics.parquet", index=False)
    (OUT / "run_manifest.json").write_text(json.dumps({"schema": "transport_base_error_tercile_v1", "row_proxy": "deterministic 20% candidate sample", "all_configured_meta_features": len(available), "usable_meta_features": len(usable), "feature_selection": "training-fold-only univariate screen; 64 context candidates plus required setup/base fields", "target": "side-local training residual terciles: base overestimate / approximately correct / base underestimate", "base_contract": "same-side frozen prequential_base_expected_net_bps; correction reconstructed in common bps", "evaluation": "chronological environments, 13-hour label-availability embargo, global top-k after common-bps mapping", "status": "COMPLETED_DIAGNOSTIC_NO_PROMOTION"}, indent=2) + "\n")


if __name__ == "__main__":
    run()
