#!/usr/bin/env python3
"""Causal trust/shrinkage overlay for the frozen residual stack.

The overlay never changes the frozen model.  It uses only resolved rows in the
trailing 21 calendar days to estimate side/regime support, soft-state OOD, and
recent side-local score rank IC.  It shrinks the residual correction
``score - prequential_base_expected_net_bps`` toward zero when trust is weak.
All formulas and constants are fixed before evaluating transport rows.
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

from extreme_price_movements.funnel_selection import global_tail_metrics

PREDICTIONS = ROOT / "data_perp/artifacts/frozen_specialist_query_residual_impact_20260810_v1/predictions_q4h_side.parquet"
LEDGER = ROOT / "data_perp/artifacts/tp6_m6_shared_regime_residual_features_20260809_v3/shared_regime_residual_ledger.parquet"
OUT = ROOT / "data_perp/artifacts/frozen_residual_trust_overlay_20260810_v1"
SOFT = ["regime_p_calm", "regime_p_trend", "regime_p_stress", "regime_p_transition"]
CONTEXT = SOFT + ["regime_entropy", "regime_transition_onset_proxy", "regime_state_duration_hours"]
LABEL_DELAY = pd.Timedelta(hours=13)
WINDOW = pd.Timedelta(days=21)
SUPPORT_SHRINK_ROWS = 2_000.0
IC_SCALE = 0.05
OOD_SCALE = 4.0
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10)


def _rank_ic(score: np.ndarray, net: np.ndarray) -> float:
    if len(score) < 32 or np.unique(score).size < 2 or np.unique(net).size < 2:
        return 0.0
    value = pd.Series(score).rank(method="average").corr(pd.Series(net).rank(method="average"))
    return 0.0 if not np.isfinite(value) else float(value)


def _js(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0)
    q = np.clip(np.asarray(q, dtype=float), 1e-12, 1.0)
    m = 0.5 * (p + q[None, :])
    return 0.5 * np.sum(p * np.log(p / m), axis=1) + 0.5 * np.sum(q[None, :] * np.log(q[None, :] / m), axis=1)


def _metrics(frame: pd.DataFrame, score_column: str, arm: str, scope: str = "pooled") -> list[dict[str, object]]:
    groups = [("pooled", frame)] if scope == "pooled" else list(frame.groupby("side_name", sort=True))
    rows: list[dict[str, object]] = []
    for side, group in groups:
        for frac in TAILS:
            n = max(1, int(np.ceil(len(group) * frac)))
            selected = group.sort_values([score_column, "candidate_id"], ascending=[False, True], kind="stable").head(n)
            rows.append({
                "arm": arm, "scope": side, "tail": frac, "rows": int(n),
                "gross_bps": float(selected.gross_bps.mean()),
                "net_bps": float(selected.net_bps.mean()),
                "rank_ic": _rank_ic(group[score_column].to_numpy(float), group.net_bps.to_numpy(float)),
                "long_rows": int(selected.side_name.eq("long").sum()),
                "short_rows": int(selected.side_name.eq("short").sum()),
            })
    return rows


def _joined_frame() -> pd.DataFrame:
    con = duckdb.connect()
    con.execute("PRAGMA threads=2")
    con.execute("PRAGMA memory_limit='6GB'")
    fields = ", ".join(f'l."{field}"' for field in CONTEXT)
    frame = con.execute(
        f'''SELECT p.*, {fields}
            FROM read_parquet(?) p
            INNER JOIN read_parquet(?) l USING (candidate_id)''',
        [str(PREDICTIONS), str(LEDGER)],
    ).fetchdf()
    con.close()
    expected = len(pd.read_parquet(PREDICTIONS, columns=["candidate_id"]))
    if len(frame) != expected or frame.candidate_id.duplicated().any():
        raise ValueError("prediction/ledger join is not one-to-one")
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    frame["label_available_ts"] = frame["__ts__"] + LABEL_DELAY
    frame[CONTEXT] = frame[CONTEXT].apply(pd.to_numeric, errors="coerce")
    frame = frame.loc[np.isfinite(frame[CONTEXT].to_numpy(float)).all(axis=1)].copy()
    if not np.allclose(frame[SOFT].sum(axis=1).to_numpy(float), 1.0, atol=1e-5):
        raise ValueError("soft regime probabilities do not sum to one")
    return frame.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _trust_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["day"] = out["__ts__"].dt.floor("D")
    out["regime_support"] = 0.0
    out["regime_ood_js"] = np.nan
    out["support_trust"] = 0.0
    out["ood_trust"] = 0.0
    out["recent_side_rank_ic"] = 0.0
    out["ic_trust"] = 0.0
    for day, day_rows in out.groupby("day", sort=True):
        cutoff = pd.Timestamp(day)
        prior_mask = out["label_available_ts"].lt(cutoff) & out["__ts__"].ge(cutoff - WINDOW) & out["__ts__"].lt(cutoff)
        current_idx = day_rows.index.to_numpy()
        if not prior_mask.any():
            continue
        for side in ("long", "short"):
            cur_idx = day_rows.index[day_rows.side_name.eq(side)].to_numpy()
            prior_idx = out.index[prior_mask & out.side_name.eq(side)].to_numpy()
            if len(cur_idx) == 0 or len(prior_idx) == 0:
                continue
            prior_p = out.loc[prior_idx, SOFT].to_numpy(float)
            prior_support = prior_p.sum(axis=0)
            prior_dist = prior_support / max(float(prior_support.sum()), 1.0)
            cur_p = out.loc[cur_idx, SOFT].to_numpy(float)
            effective_support = cur_p @ prior_support
            js = _js(cur_p, prior_dist)
            ic = _rank_ic(out.loc[prior_idx, "score"].to_numpy(float), out.loc[prior_idx, "net_bps"].to_numpy(float))
            support_trust = effective_support / (effective_support + SUPPORT_SHRINK_ROWS)
            ood_trust = np.exp(-OOD_SCALE * js)
            ic_trust = np.clip(ic / IC_SCALE, 0.0, 1.0)
            out.loc[cur_idx, "regime_support"] = effective_support
            out.loc[cur_idx, "regime_ood_js"] = js
            out.loc[cur_idx, "support_trust"] = support_trust
            out.loc[cur_idx, "ood_trust"] = ood_trust
            out.loc[cur_idx, "recent_side_rank_ic"] = ic
            out.loc[cur_idx, "ic_trust"] = ic_trust
    out["regime_ood_js"] = out["regime_ood_js"].fillna(1.0)
    out["trust_support"] = out["support_trust"]
    out["trust_ood"] = out["ood_trust"]
    out["trust_ic"] = out["ic_trust"]
    out["trust_combined"] = out["support_trust"] * out["ood_trust"] * (0.25 + 0.75 * out["ic_trust"])
    residual = out["score"].to_numpy(float) - out["prequential_base_expected_net_bps"].to_numpy(float)
    base = out["prequential_base_expected_net_bps"].to_numpy(float)
    for name, trust in (("support", out.trust_support), ("ood", out.trust_ood), ("ic", out.trust_ic), ("combined", out.trust_combined)):
        out["score_trust_" + name] = base + trust.to_numpy(float) * residual
    return out


def run(out: Path = OUT) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    frame = _trust_features(_joined_frame())
    metrics: list[dict[str, object]] = []
    arms = {"raw_control": "score", "support": "score_trust_support", "ood": "score_trust_ood", "ic": "score_trust_ic", "combined": "score_trust_combined"}
    for name, column in arms.items():
        metrics.extend(_metrics(frame, column, name, "pooled"))
        metrics.extend(_metrics(frame, column, name, "side"))
    frame.to_parquet(out / "predictions_with_trust.parquet", index=False, compression="zstd")
    pd.DataFrame(metrics).to_parquet(out / "metrics.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "frozen_residual_trust_overlay_ablation_v1",
        "prediction_artifact": str(PREDICTIONS),
        "causal_ledger": str(LEDGER),
        "window": str(WINDOW),
        "label_delay": str(LABEL_DELAY),
        "support_shrink_rows": SUPPORT_SHRINK_ROWS,
        "ood_scale": OOD_SCALE,
        "ic_scale": IC_SCALE,
        "trust_formula": "support * exp(-4*JS) * (0.25 + 0.75*clip(recent_side_rank_ic/0.05,0,1))",
        "arms": arms,
        "rows": int(len(frame)),
        "selection": "pooled global top-k; monthly and side diagnostics",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


if __name__ == "__main__":
    print(run())
