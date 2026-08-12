#!/usr/bin/env python3
"""Long-only causal global-vs-regime conversion mapping.

The frozen residual score is not refit.  A prior-resolved calibration map is
applied to the score using either a global correction, a (degenerate here)
side correction, or a strongly shrunk expectation over the causal soft regime
probabilities.  The full primary + transport strict-OOS chronology is used;
short rows never enter the frame or calibration.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.shared_regime_calibration import prequential_shared_bps_calibration

PREDICTIONS = ROOT / "data_perp/artifacts/long_only_reliability_boundary_ablation_20260810_v1/all_long_only_predictions.parquet"
LEDGER = ROOT / "data_perp/artifacts/tp6_m6_shared_regime_residual_features_20260809_v3/shared_regime_residual_ledger.parquet"
OUT = ROOT / "data_perp/artifacts/long_only_regime_conversion_map_20260810_v1"
SOFT = ["regime_p_calm", "regime_p_trend", "regime_p_stress", "regime_p_transition"]
MODES = ("C0_global", "C1_side", "C2_side_soft_regime", "C3_hierarchical_affine_soft_regime")
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10)
LABEL_DELAY = pd.Timedelta(hours=13)


def _load() -> pd.DataFrame:
    con = duckdb.connect()
    con.execute("PRAGMA threads=2")
    con.execute("PRAGMA memory_limit='6GB'")
    fields = ", ".join(f'l."{field}"' for field in SOFT + [
        "regime_entropy", "regime_transition_onset_proxy", "regime_state_duration_hours",
    ])
    frame = con.execute(
        f'''SELECT p.*, {fields}
            FROM read_parquet(?) p
            INNER JOIN read_parquet(?) l USING (candidate_id)''',
        [str(PREDICTIONS), str(LEDGER)],
    ).fetchdf()
    con.close()
    frame = frame.loc[frame.side_name.astype(str).str.lower().eq("long")].copy()
    if frame.candidate_id.duplicated().any():
        raise ValueError("candidate IDs must be unique after long-only join")
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    frame[SOFT] = frame[SOFT].apply(pd.to_numeric, errors="coerce")
    frame = frame.loc[np.isfinite(frame[SOFT].to_numpy(float)).all(axis=1)].copy()
    if not np.allclose(frame[SOFT].sum(axis=1).to_numpy(float), 1.0, atol=1e-5):
        raise ValueError("causal soft regime probabilities do not sum to one")
    frame["outcome_resolved_at"] = frame["__ts__"] + LABEL_DELAY
    frame["month"] = frame["__ts__"].dt.to_period("M").astype(str)
    frame = frame.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    return frame


def _metrics(frame: pd.DataFrame, column: str, arm: str, period: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for frac in TAILS:
        n = max(1, int(np.ceil(len(frame) * frac)))
        chosen = frame.sort_values([column, "candidate_id"], ascending=[False, True], kind="stable").head(n)
        rows.append({
            "arm": arm, "period": period, "tail": frac, "population_rows": int(len(frame)),
            "selected_rows": int(n), "gross_bps": float(chosen.gross_bps.mean()),
            "net_bps": float(chosen.net_bps.mean()),
            "rank_ic": float(frame[column].rank().corr(frame.net_bps.rank())),
        })
    return rows


def run(out: Path = OUT) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    frame = _load()
    raw = frame.score.to_numpy(float)
    target = frame.net_bps.to_numpy(float)
    frame["outcome_resolved_at"] = pd.to_datetime(frame["outcome_resolved_at"], utc=True)
    predictions = frame[["candidate_id", "__ts__", "month", "side_name", "net_bps", "gross_bps", "score", "fold"]].copy()
    metrics: list[dict[str, object]] = []
    audits: dict[str, object] = {}
    metrics.extend(_metrics(frame, "score", "raw_control", "all_eras"))
    for mode in MODES:
        calibrated, audit = prequential_shared_bps_calibration(
            frame,
            raw,
            target,
            mode=mode,
            decision_timestamp_column="__ts__",
            resolution_column="outcome_resolved_at",
            side_column="side_name",
            soft_regime_columns=SOFT,
            anchor="day",
            min_global_rows=500,
            global_shrink_rows=5_000.0,
            side_shrink_rows=1_500.0,
            regime_shrink_rows=3_000.0,
            regime_weight_cap=0.50,
        )
        column = "mapped__" + mode
        predictions[column] = calibrated
        scored = frame.copy()
        scored[column] = calibrated
        metrics.extend(_metrics(scored, column, mode, "all_eras"))
        periods = {
            "sep_oct_2023": scored[scored.month.isin(["2023-09", "2023-10"])],
            "nov_dec_2023": scored[scored.month.isin(["2023-11", "2023-12"])],
            "jan_feb_2024": scored[scored.month.isin(["2024-01", "2024-02"])],
            "jul_oct_2024": scored[scored.month.isin(["2024-07", "2024-08", "2024-09", "2024-10"])],
            "nov_2024": scored[scored.month.eq("2024-11")],
        }
        for period, group in periods.items():
            if len(group):
                metrics.extend(_metrics(group, column, mode, period))
        for month, group in scored.groupby("month", sort=True):
            metrics.extend(_metrics(group, column, mode, month))
        audits[mode] = {
            "anchors": int(len(audit)),
            "identity_anchors": int((audit.status == "identity_no_prior_resolved_support").sum()),
            "first_anchor": audit.anchor_utc.min().isoformat() if len(audit) else None,
            "last_anchor": audit.anchor_utc.max().isoformat() if len(audit) else None,
        }
        audit.to_parquet(out / f"{mode.lower()}_audit.parquet", index=False, compression="zstd")
    # Add raw-control period/month rows after all mapped arms are materialized.
    for period, group in {
        "sep_oct_2023": frame[frame.month.isin(["2023-09", "2023-10"])],
        "nov_dec_2023": frame[frame.month.isin(["2023-11", "2023-12"])],
        "jan_feb_2024": frame[frame.month.isin(["2024-01", "2024-02"])],
        "jul_oct_2024": frame[frame.month.isin(["2024-07", "2024-08", "2024-09", "2024-10"])],
        "nov_2024": frame[frame.month.eq("2024-11")],
    }.items():
        if len(group):
            metrics.extend(_metrics(group, "score", "raw_control", period))
    for month, group in frame.groupby("month", sort=True):
        metrics.extend(_metrics(group, "score", "raw_control", month))
    predictions.to_parquet(out / "predictions.parquet", index=False, compression="zstd")
    pd.DataFrame(metrics).to_parquet(out / "metrics.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "long_only_regime_conversion_map_v1",
        "prediction_source": str(PREDICTIONS), "causal_ledger": str(LEDGER),
        "side": "long_only", "short_rows_used": 0,
        "modes": list(MODES), "soft_regime_fields": SOFT,
        "label_delay": str(LABEL_DELAY), "anchor": "day",
        "fit_contract": "outcome_resolved_at < each daily anchor; prior rows only",
        "rows": int(len(frame)), "audit_summary": audits,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


if __name__ == "__main__":
    print(run())
