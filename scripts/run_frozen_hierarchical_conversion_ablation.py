#!/usr/bin/env python3
"""Prequential side/regime conversion calibration for the frozen residual stack.

This is a matched calibration-only ablation.  It does not refit the base,
specialist, or residual model.  Each day's calibration uses only rows whose
13-hour outcome was resolved before that day's first decision timestamp.  C0
and C1 provide global and side corrections; C2/C3 add a strongly shrunk
expectation over the causal five-state soft regime probabilities.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.funnel_selection import global_tail_metrics
from extreme_price_movements.shared_regime_calibration import prequential_shared_bps_calibration

PREDICTIONS = ROOT / "data_perp/artifacts/frozen_specialist_query_residual_impact_20260810_v1/predictions_q4h_side.parquet"
LEDGER = ROOT / "data_perp/artifacts/tp6_m6_shared_regime_residual_features_20260809_v3/shared_regime_residual_ledger.parquet"
OUT = ROOT / "data_perp/artifacts/frozen_hierarchical_conversion_ablation_20260810_v1"
SOFT = ["regime_p_calm", "regime_p_trend", "regime_p_stress", "regime_p_transition"]
CONTEXT = SOFT + ["regime_entropy", "regime_transition_onset_proxy", "regime_state_duration_hours"]
MODES = ("C0_global", "C1_side", "C2_side_soft_regime", "C3_hierarchical_affine_soft_regime")
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10)
LABEL_DELAY = pd.Timedelta(hours=13)


def _metric(frame: pd.DataFrame, score_column: str, label: str, scope: str = "pooled") -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    groups = [("pooled", frame)] if scope == "pooled" else list(frame.groupby("side_name", sort=True))
    for side, group in groups:
        for frac in TAILS:
            n = max(1, int(np.ceil(len(group) * frac)))
            selected = group.sort_values([score_column, "candidate_id"], ascending=[False, True], kind="stable").head(n)
            rows.append({
                "arm": label, "scope": side, "tail": frac, "rows": int(n),
                "gross_bps": float(selected.gross_bps.mean()),
                "net_bps": float(selected.net_bps.mean()),
                "rank_ic": float(group[score_column].rank().corr(group.net_bps.rank())),
                "long_rows": int(selected.side_name.eq("long").sum()),
                "short_rows": int(selected.side_name.eq("short").sum()),
            })
    return rows


def run(out: Path = OUT) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    # Restrict the ledger scan to the prediction IDs in DuckDB.  The source
    # ledger is ~1.3m rows; materialising it before the join needlessly spikes
    # memory and was the dominant cost of this diagnostic.
    con = duckdb.connect()
    con.execute("PRAGMA threads=2")
    con.execute("PRAGMA memory_limit='6GB'")
    select_context = ", ".join(f'l."{field}"' for field in CONTEXT)
    frame = con.execute(
        f'''SELECT p.*, {select_context}
            FROM read_parquet(?) p
            INNER JOIN read_parquet(?) l USING (candidate_id)''',
        [str(PREDICTIONS), str(LEDGER)],
    ).fetchdf()
    con.close()
    if frame.candidate_id.duplicated().any():
        raise ValueError("candidate IDs must be unique after prediction/ledger join")
    if len(frame) != len(pd.read_parquet(PREDICTIONS, columns=["candidate_id"])):
        raise ValueError("not every frozen prediction joined to the causal regime ledger")
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    frame[CONTEXT] = frame[CONTEXT].apply(pd.to_numeric, errors="coerce")
    frame = frame.loc[np.isfinite(frame[CONTEXT].to_numpy(float)).all(axis=1)].copy()
    soft_sum = frame[SOFT].sum(axis=1)
    if not np.allclose(soft_sum.to_numpy(float), 1.0, atol=1e-5):
        raise ValueError("causal soft regime probabilities do not sum to one")
    frame["outcome_resolved_at"] = frame["__ts__"] + LABEL_DELAY
    frame = frame.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    raw = frame.score.to_numpy(float)
    target = frame.net_bps.to_numpy(float)
    predictions = frame[["candidate_id", "__ts__", "side_name", "net_bps", "gross_bps", "score", "fold"]].copy()
    metrics: list[dict[str, object]] = []
    # Frozen control: the original score is ranked exactly as produced.
    metrics.extend(_metric(frame, "score", "raw_control", "pooled"))
    metrics.extend(_metric(frame, "score", "raw_control", "side"))
    audits: dict[str, object] = {}
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
        column = "calibrated__" + mode
        predictions[column] = calibrated
        scored = frame.copy()
        scored[column] = calibrated
        metrics.extend(_metric(scored, column, mode, "pooled"))
        metrics.extend(_metric(scored, column, mode, "side"))
        audits[mode] = {
            "anchors": int(len(audit)),
            "identity_anchors": int((audit.status == "identity_no_prior_resolved_support").sum()),
            "first_anchor": audit.anchor_utc.min().isoformat() if len(audit) else None,
            "last_anchor": audit.anchor_utc.max().isoformat() if len(audit) else None,
        }
        audit.to_parquet(out / f"{mode.lower()}_audit.parquet", index=False, compression="zstd")
    predictions.to_parquet(out / "predictions.parquet", index=False, compression="zstd")
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_parquet(out / "metrics.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "frozen_hierarchical_conversion_ablation_v1",
        "input_predictions": str(PREDICTIONS),
        "causal_ledger": str(LEDGER),
        "modes": list(MODES),
        "soft_regime_fields": SOFT,
        "context_fields": CONTEXT,
        "label_delay": str(LABEL_DELAY),
        "fit_contract": "prior rows only with outcome_resolved_at < each day's first decision timestamp",
        "anchor": "day",
        "selection": "raw frozen score control; no model or threshold tuning",
        "rows": int(len(frame)),
        "audit_summary": audits,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


if __name__ == "__main__":
    print(run())
