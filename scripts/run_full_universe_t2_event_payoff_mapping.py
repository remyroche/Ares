#!/usr/bin/env python3
"""Round-A B0/B1/B2 causal event-payoff mapping evaluation.

This is deliberately independent from the base and meta runners.  It takes
already-produced full-universe T2 TP3/SL2 base predictions and asks whether
the raw base score can be improved merely by a *causal* conversion of the
three predicted event probabilities into bps.

B0: the frozen base ``score_bps``.
B1: one global event -> realised-net payoff vector fitted on resolved rows
    before the OOS boundary.
B2: separate long/short event -> realised-net payoff vectors, likewise fitted
    only before the OOS boundary.

Rows only enter a map when their complete 12 hour path is known before the
boundary.  Neither B1 nor B2 can inspect an OOS label while building its map.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


EVENT_NAMES = {0: "upper_first", 1: "lower_first", 2: "timeout"}
TOP_FRACTIONS = (0.01, 0.05, 0.10, 0.20)
PROBABILITY_COLUMNS = ("p_upper", "p_lower", "p_timeout")


def _read_predictions(root: Path) -> pd.DataFrame:
    paths = [root / side / "target_screen_predictions.parquet" for side in ("long", "short")]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing side prediction artifact(s): " + ", ".join(missing))
    columns = ["candidate_id", "__ts__", "side_name", "gross_bps", "net_bps", "score_bps", *PROBABILITY_COLUMNS]
    frame = pd.concat([pd.read_parquet(path, columns=columns) for path in paths], ignore_index=True)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
    if frame.candidate_id.duplicated().any():
        raise ValueError("Base prediction candidate_id is not unique across sides")
    probabilities = frame.loc[:, PROBABILITY_COLUMNS].to_numpy(float)
    if not np.isfinite(probabilities).all() or (probabilities < -1e-8).any():
        raise ValueError("Base probabilities must be finite and non-negative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-4, atol=1e-5):
        raise ValueError("Base p_upper/p_lower/p_timeout must sum to one")
    return frame


def _read_event_labels(panel: Path) -> pd.DataFrame:
    parts = sorted((panel / "parts").glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No panel parts at {panel}")
    columns = ["candidate_id", "t2_tp3_sl2_event", "t2_tp3_sl2_exit_minute"]
    labels = pd.concat([pd.read_parquet(part, columns=columns) for part in parts], ignore_index=True)
    if labels.candidate_id.duplicated().any():
        raise ValueError("Panel candidate_id is not unique")
    labels = labels.rename(columns={"t2_tp3_sl2_event": "event", "t2_tp3_sl2_exit_minute": "exit_minute"})
    if not labels.event.isin(EVENT_NAMES).all():
        raise ValueError("Unexpected TP3/SL2 barrier event code")
    return labels


def _fit_event_map(calibration: pd.DataFrame, *, side_local: bool) -> tuple[dict[str, list[float]], np.ndarray]:
    """Fit conditional realised payoffs.  A missing state fails closed to global.

    Every state occurs abundantly in the full panel.  The global fallback makes
    the mapping reproducible even for a deliberately shortened calibration
    range, while the manifest exposes every fallback explicitly.
    """
    global_means = calibration.groupby("event", observed=True).net_bps.mean().reindex(range(3))
    if global_means.isna().any():
        raise ValueError("Calibration interval has no examples of every barrier event")
    details: dict[str, list[float]] = {"global": [float(value) for value in global_means]}
    if not side_local:
        return details, np.tile(global_means.to_numpy(float), (len(calibration), 1))

    by_side = calibration.groupby(["side_name", "event"], observed=True).net_bps.mean().unstack("event").reindex(columns=range(3))
    for side in ("long", "short"):
        values = by_side.loc[side] if side in by_side.index else global_means
        filled = values.fillna(global_means)
        details[side] = [float(value) for value in filled]
        details[f"{side}_fallback_to_global"] = [bool(pd.isna(value)) for value in values]
    # This matrix is for the calibration diagnostics only; OOS gets an
    # equivalent map below.  Returning it avoids a hidden inference branch.
    matrix = np.vstack([np.asarray(details[row.side_name], dtype=float) for _, row in calibration.iterrows()])
    return details, matrix


def _apply_event_map(frame: pd.DataFrame, details: dict[str, list[float]], *, side_local: bool) -> np.ndarray:
    probs = frame.loc[:, PROBABILITY_COLUMNS].to_numpy(float)
    if not side_local:
        return probs @ np.asarray(details["global"], dtype=float)
    payoff = np.vstack([np.asarray(details[side], dtype=float) for side in frame.side_name])
    return np.einsum("ij,ij->i", probs, payoff)


def _metrics(frame: pd.DataFrame, score_column: str) -> list[dict]:
    ordered = frame.sort_values([score_column, "candidate_id"], ascending=[False, True], kind="mergesort")
    result = []
    for fraction in TOP_FRACTIONS:
        selected = ordered.head(int(np.ceil(len(ordered) * fraction)))
        result.append({
            "top_fraction": fraction,
            "n": int(len(selected)),
            "gross_bps": float(selected.gross_bps.mean()),
            "net_bps": float(selected.net_bps.mean()),
            "long_n": int(selected.side_name.eq("long").sum()),
            "short_n": int(selected.side_name.eq("short").sum()),
        })
    return result


def _score_diagnostics(frame: pd.DataFrame, score_column: str) -> dict:
    score = frame[score_column].to_numpy(float)
    net = frame.net_bps.to_numpy(float)
    gross = frame.gross_bps.to_numpy(float)
    return {
        "score_mean_bps": float(np.mean(score)),
        "score_std_bps": float(np.std(score)),
        "net_spearman_ic": float(spearmanr(score, net).statistic),
        "gross_spearman_ic": float(spearmanr(score, gross).statistic),
    }


def _event_calibration(frame: pd.DataFrame) -> dict:
    probabilities = frame.loc[:, PROBABILITY_COLUMNS].to_numpy(float)
    labels = frame.event.to_numpy(int)
    one_hot = np.eye(3)[labels]
    brier = np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
    log_loss = -np.log(np.maximum(probabilities[np.arange(len(frame)), labels], 1e-12)).mean()
    return {
        "multiclass_brier": float(brier),
        "multiclass_log_loss": float(log_loss),
        "predicted_event_frequency": {EVENT_NAMES[i]: float(probabilities[:, i].mean()) for i in range(3)},
        "realised_event_frequency": {EVENT_NAMES[i]: float((labels == i).mean()) for i in range(3)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True, help="Root containing long/ and short/ base prediction artifacts")
    parser.add_argument("--panel", type=Path, required=True, help="Full-universe v3 panel containing TP3/SL2 event labels")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--map-start", default="2024-04-01", help="First timestamp allowed into event-payoff calibration")
    parser.add_argument("--oos-start", default="2024-08-01", help="Untouched OOS start; maps are frozen before this time")
    parser.add_argument("--oos-end", default="2024-12-01", help="Exclusive OOS end")
    parser.add_argument("--horizon-minutes", type=int, default=720)
    args = parser.parse_args()
    map_start = pd.Timestamp(args.map_start, tz="UTC")
    oos_start = pd.Timestamp(args.oos_start, tz="UTC")
    oos_end = pd.Timestamp(args.oos_end, tz="UTC")
    if not map_start < oos_start < oos_end:
        raise ValueError("Require map-start < oos-start < oos-end")

    pred = _read_predictions(args.base_root)
    labels = _read_event_labels(args.panel)
    data = pred.merge(labels, on="candidate_id", how="left", validate="one_to_one")
    if data.event.isna().any():
        raise ValueError("Panel is missing TP3/SL2 labels for base predictions")
    data["event"] = data.event.astype(int)
    # A conservative resolution guard uses the full horizon rather than the
    # individual early-exit time.  This makes the leakage proof independent of
    # exit-recording conventions.
    resolved = data["__ts__"] + pd.Timedelta(minutes=args.horizon_minutes)
    calibration = data[(data["__ts__"] >= map_start) & (resolved < oos_start)].copy()
    oos = data[(data["__ts__"] >= oos_start) & (data["__ts__"] < oos_end)].copy()
    if calibration.empty or oos.empty:
        raise ValueError("Calibration or OOS interval is empty")

    global_details, _ = _fit_event_map(calibration, side_local=False)
    side_details, _ = _fit_event_map(calibration, side_local=True)
    oos["b0_raw_score_bps"] = oos.score_bps.astype(float)
    oos["b1_global_event_payoff_bps"] = _apply_event_map(oos, global_details, side_local=False)
    oos["b2_side_event_payoff_bps"] = _apply_event_map(oos, side_details, side_local=True)

    variants = {
        "B0_raw_base_score": "b0_raw_score_bps",
        "B1_global_event_payoff": "b1_global_event_payoff_bps",
        "B2_side_local_event_payoff": "b2_side_event_payoff_bps",
    }
    rows = []
    diagnostics = {}
    for name, column in variants.items():
        diagnostics[name] = _score_diagnostics(oos, column)
        for metric in _metrics(oos, column):
            rows.append({"variant": name, **metric})
    metrics = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    oos.to_parquet(args.out / "oos_scored_predictions.parquet", index=False)
    metrics.to_parquet(args.out / "global_oos_metrics.parquet", index=False)
    calibration_counts = calibration.groupby(["side_name", "event"], observed=True).agg(n=("candidate_id", "size"), mean_net_bps=("net_bps", "mean"))
    manifest = {
        "schema": "round_a_b1_b2_event_payoff_mapping_v1",
        "base_prediction_root": str(args.base_root),
        "panel": str(args.panel),
        "contract": {
            "geometry": "T2 TP3/SL2",
            "entry": "inherited exactly from frozen base prediction artifact",
            "exit": "first TP/SL touch, else H12 timeout",
            "return": "stored realised gross/net barrier-exit bps",
            "global_selection": "one pooled long/short/timestamp book, deterministic candidate_id tie-break",
        },
        "causality": {
            "map_start": str(map_start),
            "oos_start": str(oos_start),
            "oos_end_exclusive": str(oos_end),
            "resolution_rule": f"decision timestamp + {args.horizon_minutes} minutes < oos_start",
            "calibration_rows": int(len(calibration)),
            "oos_rows": int(len(oos)),
            "assertion": "B1/B2 maps contain no OOS outcome labels",
        },
        "maps": {"B1_global": global_details, "B2_side_local": side_details},
        "calibration_event_counts_and_payoffs": calibration_counts.reset_index().assign(event=lambda x: x.event.map(EVENT_NAMES)).to_dict("records"),
        "oos_event_probability_calibration": _event_calibration(oos),
        "score_diagnostics": diagnostics,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
