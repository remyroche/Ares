#!/usr/bin/env python3
"""Strict expanding-month conversion/reliability learner ablation.

The frozen residual score is not refit.  For each transport month and side, a
small conversion learner is fit only on earlier rows whose 13-hour outcomes
were resolved before the month start.  It tests a Huber residual target against
a three-class under/accurate/overconfidence target and applies fixed correction
strengths to the frozen score.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PREDICTION_SOURCES = (
    ROOT / "data_perp/artifacts/frozen_specialist_primary_oos_20260810_v1/predictions.parquet",
    ROOT / "data_perp/artifacts/frozen_specialist_query_residual_impact_20260810_v1/predictions_q4h_side.parquet",
)
LEDGER = ROOT / "data_perp/artifacts/tp6_m6_shared_regime_residual_features_20260809_v3/shared_regime_residual_ledger.parquet"
OUT = ROOT / "data_perp/artifacts/frozen_conversion_reliability_learner_ablation_20260810_v2"
LABEL_DELAY = pd.Timedelta(hours=13)
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10)
CORRECTION_STRENGTHS = (0.25, 0.50, 1.00)
RESIDUAL_BOUNDARY_BPS = 75.0
CLASS_CENTERS = np.asarray([-100.0, 0.0, 100.0], dtype=float)

BASE_FIELDS = ["p_clear", "p_adverse", "p_weak", "prequential_base_expected_net_bps"]
SOFT_FIELDS = ["regime_p_calm", "regime_p_trend", "regime_p_stress", "regime_p_transition"]
CONTEXT_FIELDS = [
    "regime_entropy", "regime_transition_onset_proxy", "regime_state_duration_hours",
    "mkt_ret_eq_24h", "regime_liquidity_score", "mkt_rv_ratio_1h_24h", "mkt_oi_chg_z_24h",
    "mkt_funding_dispersion", "cross_asset_corr_4h", "mkt_systemic_deleveraging_score",
    "mkt_flush_exhaustion_score", "post_liquidation_rebound_score", "negative_breadth_pct",
    "btc_resilience_alt_weakness", "short_covering_score_market", "deleveraging_without_followthrough",
    "short_signal_recovery_conflict",
]
FEATURES = BASE_FIELDS + SOFT_FIELDS + CONTEXT_FIELDS + ["score"]


def _join() -> pd.DataFrame:
    con = duckdb.connect()
    con.execute("PRAGMA threads=2")
    con.execute("PRAGMA memory_limit='6GB'")
    fields = ", ".join(f'l."{f}"' for f in BASE_FIELDS + SOFT_FIELDS + CONTEXT_FIELDS)
    source_sql = "[" + ",".join("?" for _ in PREDICTION_SOURCES) + "]"
    frame = con.execute(
        f'''SELECT p.*, {fields}
            FROM read_parquet({source_sql}, union_by_name=true) p
            INNER JOIN read_parquet(?) l USING (candidate_id)''',
        [*(str(path) for path in PREDICTION_SOURCES), str(LEDGER)],
    ).fetchdf()
    con.close()
    expected = sum(len(pd.read_parquet(path, columns=["candidate_id"])) for path in PREDICTION_SOURCES)
    if len(frame) != expected or frame.candidate_id.duplicated().any():
        raise ValueError("prediction/ledger join is not one-to-one")
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    frame["label_available_ts"] = frame["__ts__"] + LABEL_DELAY
    numeric = list(dict.fromkeys(FEATURES + ["net_bps", "gross_bps"]))
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame = frame.loc[np.isfinite(frame[numeric].to_numpy(float)).all(axis=1)].copy()
    if not np.allclose(frame[SOFT_FIELDS].sum(axis=1).to_numpy(float), 1.0, atol=1e-5):
        raise ValueError("causal soft regime probabilities do not sum to one")
    frame["residual_target"] = frame["net_bps"] - frame["score"]
    frame["reliability_class"] = np.select(
        (frame.residual_target <= -RESIDUAL_BOUNDARY_BPS, frame.residual_target >= RESIDUAL_BOUNDARY_BPS),
        (0, 2), default=1,
    ).astype(np.int32)
    frame["month"] = frame["__ts__"].dt.to_period("M").astype(str)
    return frame.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if len(train) < 500:
        return np.zeros(len(test), dtype=float), np.zeros(len(test), dtype=float), {"status": "identity_insufficient_prior", "train_rows": int(len(train))}
    fields = FEATURES
    med = train[fields].median(numeric_only=True)
    x_train = train[fields].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0).astype(np.float32)
    x_test = test[fields].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0).astype(np.float32)
    common = dict(
        n_estimators=160, learning_rate=0.03, max_depth=4, num_leaves=15,
        min_child_samples=500, feature_fraction=0.80, bagging_fraction=0.80,
        bagging_freq=1, reg_lambda=20.0, reg_alpha=0.0, random_state=20260810,
        n_jobs=1, verbosity=-1,
    )
    reg = lgb.LGBMRegressor(objective="huber", alpha=0.85, **common)
    reg.fit(x_train, train.residual_target.to_numpy(float))
    reg_pred = np.clip(reg.predict(x_test), -200.0, 200.0)
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=3, **common)
    clf.fit(x_train, train.reliability_class.to_numpy(np.int32))
    proba = clf.predict_proba(x_test)
    # LightGBM preserves class order for present classes; enforce a stable
    # three-column simplex even if an early side has no extreme class.
    full = np.zeros((len(test), 3), dtype=float)
    for col, cls in enumerate(clf.classes_.astype(int)):
        full[:, cls] = proba[:, col]
    cls_pred = full @ CLASS_CENTERS
    return reg_pred, np.clip(cls_pred, -200.0, 200.0), {
        "status": "trained", "train_rows": int(len(train)),
        "train_class_counts": {str(k): int(v) for k, v in zip(*np.unique(train.reliability_class, return_counts=True))},
    }


def _metrics(frame: pd.DataFrame, score_column: str, arm: str, scope: str = "pooled") -> list[dict[str, object]]:
    groups = [("pooled", frame)] if scope == "pooled" else list(frame.groupby("side_name", sort=True))
    rows: list[dict[str, object]] = []
    for side, group in groups:
        for frac in TAILS:
            n = max(1, int(np.ceil(len(group) * frac)))
            selected = group.sort_values([score_column, "candidate_id"], ascending=[False, True], kind="stable").head(n)
            rows.append({
                "arm": arm, "scope": side, "tail": frac, "rows": int(n),
                "gross_bps": float(selected.gross_bps.mean()), "net_bps": float(selected.net_bps.mean()),
                "rank_ic": float(group[score_column].rank().corr(group.net_bps.rank())),
                "long_rows": int(selected.side_name.eq("long").sum()), "short_rows": int(selected.side_name.eq("short").sum()),
            })
    return rows


def run(out: Path = OUT) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    frame = _join()
    frame["regression_correction"] = 0.0
    frame["ordinal_correction"] = 0.0
    train_audit: list[dict[str, object]] = []
    for month in sorted(frame.month.unique()):
        month_start = pd.Timestamp(month, tz="UTC")
        test_mask = frame.month.eq(month)
        # Explicitly require outcomes resolved before the test month.  No row
        # from the current month can train its own conversion learner.
        prior_mask = frame.label_available_ts.lt(month_start) & frame.__ts__.lt(month_start)
        for side in ("long", "short"):
            train = frame.loc[prior_mask & frame.side_name.eq(side)].copy()
            test_idx = frame.index[test_mask & frame.side_name.eq(side)].to_numpy()
            if len(test_idx) == 0:
                continue
            reg_pred, ord_pred, audit = _fit_predict(train, frame.loc[test_idx])
            frame.loc[test_idx, "regression_correction"] = reg_pred
            frame.loc[test_idx, "ordinal_correction"] = ord_pred
            train_audit.append({"month": month, "side": side, **audit})
    base = frame["prequential_base_expected_net_bps"].to_numpy(float)
    raw = frame["score"].to_numpy(float)
    for strength in CORRECTION_STRENGTHS:
        frame[f"score_regression_a{strength:g}"] = raw + strength * frame.regression_correction.to_numpy(float)
        frame[f"score_ordinal_a{strength:g}"] = raw + strength * frame.ordinal_correction.to_numpy(float)
    arms = {"raw_control": "score"}
    arms.update({f"regression_a{a:g}": f"score_regression_a{a:g}" for a in CORRECTION_STRENGTHS})
    arms.update({f"ordinal_a{a:g}": f"score_ordinal_a{a:g}" for a in CORRECTION_STRENGTHS})
    metrics: list[dict[str, object]] = []
    for name, column in arms.items():
        metrics.extend(_metrics(frame, column, name, "pooled"))
        metrics.extend(_metrics(frame, column, name, "side"))
    frame.to_parquet(out / "predictions.parquet", index=False, compression="zstd")
    pd.DataFrame(metrics).to_parquet(out / "metrics_all.parquet", index=False, compression="zstd")
    transport = frame[frame["fold"].astype(str).str.startswith("transport")].copy()
    transport_metrics: list[dict[str, object]] = []
    for name, column in arms.items():
        transport_metrics.extend(_metrics(transport, column, name, "pooled"))
        transport_metrics.extend(_metrics(transport, column, name, "side"))
    pd.DataFrame(transport_metrics).to_parquet(out / "metrics_transport.parquet", index=False, compression="zstd")
    pd.DataFrame(train_audit).to_parquet(out / "training_audit.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "frozen_conversion_reliability_learner_ablation_v2_historical_support",
        "prediction_sources": [str(path) for path in PREDICTION_SOURCES], "ledger": str(LEDGER),
        "feature_contract": FEATURES,
        "target_regression": "net_bps - frozen_score, Huber",
        "target_ordinal": "three classes: residual <= -75, |residual| < 75, residual >= 75 bps",
        "class_centers_bps": CLASS_CENTERS.tolist(),
        "expanding_month_contract": "train only prior months with label_available_ts < month start, separate long/short models",
        "correction_strengths": CORRECTION_STRENGTHS,
        "rows": int(len(frame)), "arms": arms,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


if __name__ == "__main__":
    print(run())
