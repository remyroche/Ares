#!/usr/bin/env python3
"""Stage 5.1: causal event-probability calibration for the frozen TP3/SL2 base.

The frozen base emits the mutually exclusive ``upper/lower/timeout``
probabilities.  This runner evaluates only low-complexity probability repairs:

* no calibration (the B2 control);
* one global temperature; and
* diagonal vector scaling, with a fixed L2 penalty.

For the latter two methods it also evaluates a side-shrunk version.  A
side-specific parameter is linearly shrunk to the global parameter with a
*predeclared* 50k-row pseudo-count.  There is no side-specific HPO.

All fitted probability parameters and B2 payoff maps are derived from outcomes
whose 12-hour label has resolved before the calibration block.  Development is
therefore prequential, while the Aug--Nov OOS score uses the final parameter
and payoff map frozen at 2024-08-01.  No OOS event or payoff is used in a fit.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp
from scipy.stats import spearmanr


PROBS = ("p_upper", "p_lower", "p_timeout")
EVENTS = ("upper_first", "lower_first", "timeout")
TOPS = (0.01, 0.05, 0.10, 0.20)
SIDE_SHRINK_ROWS = 50_000.0
VECTOR_L2 = 0.02


def _softmax(logits: np.ndarray) -> np.ndarray:
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def _logits(probs: np.ndarray) -> np.ndarray:
    # The redundant common logit offset is harmless for softmax; centering
    # makes the vector-scaling parameters better conditioned.
    raw = np.log(np.clip(probs, 1e-8, 1.0))
    return raw - raw.mean(axis=1, keepdims=True)


def _temperature_predict(probs: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return _softmax(_logits(probs) / np.exp(float(theta[0])))


def _vector_predict(probs: np.ndarray, theta: np.ndarray) -> np.ndarray:
    # Diagonal vector scaling: one slope and one intercept per class.  The
    # intercept gauge is removed because adding the same value to all logits
    # cannot change a softmax probability.
    slopes = np.exp(np.clip(theta[:3], -3.0, 3.0))
    intercept = theta[3:] - np.mean(theta[3:])
    return _softmax(_logits(probs) * slopes + intercept)


def _nll(pred: np.ndarray, labels: np.ndarray) -> float:
    return float(-np.log(np.clip(pred[np.arange(len(labels)), labels], 1e-12, 1.0)).mean())


def _fit_temperature(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    fit = minimize_scalar(
        lambda log_t: _nll(_temperature_predict(probs, np.asarray([log_t])), labels),
        method="bounded", bounds=(-2.0, 2.0), options={"xatol": 1e-5},
    )
    if not fit.success:
        raise RuntimeError(f"temperature fit failed: {fit.message}")
    return np.asarray([fit.x], dtype=float)


def _fit_vector(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    initial = np.zeros(6, dtype=float)
    def objective(theta: np.ndarray) -> float:
        # Parameters are intentionally regularised around identity scaling.
        # This makes a low-complexity calibrator rather than another classifier.
        return _nll(_vector_predict(probs, theta), labels) + VECTOR_L2 * float(np.mean(theta * theta))
    fit = minimize(objective, initial, method="L-BFGS-B", bounds=[(-3.0, 3.0)] * 3 + [(-5.0, 5.0)] * 3,
                   options={"maxiter": 120, "ftol": 1e-10})
    if not fit.success:
        raise RuntimeError(f"vector fit failed: {fit.message}")
    return np.asarray(fit.x, dtype=float)


@dataclass(frozen=True)
class Calibration:
    method: str
    side_shrunk: bool
    global_theta: np.ndarray | None
    side_theta: dict[str, np.ndarray] | None
    side_support: dict[str, int] | None

    @property
    def name(self) -> str:
        if self.method == "none":
            return "C0_uncalibrated_B2"
        return f"C{1 if self.method == 'temperature' else 2}_{self.method}_{'side_shrunk' if self.side_shrunk else 'global'}"

    def predict(self, probs: np.ndarray, sides: np.ndarray) -> np.ndarray:
        if self.method == "none":
            return probs.copy()
        fn = _temperature_predict if self.method == "temperature" else _vector_predict
        if not self.side_shrunk:
            assert self.global_theta is not None
            return fn(probs, self.global_theta)
        assert self.global_theta is not None and self.side_theta is not None and self.side_support is not None
        out = np.empty_like(probs)
        for side in ("long", "short"):
            take = sides == side
            w = self.side_support[side] / (self.side_support[side] + SIDE_SHRINK_ROWS)
            theta = self.global_theta + w * (self.side_theta[side] - self.global_theta)
            out[take] = fn(probs[take], theta)
        return out

    def manifest(self) -> dict:
        return {
            "method": self.method,
            "side_shrunk": self.side_shrunk,
            "global_theta": None if self.global_theta is None else self.global_theta.tolist(),
            "side_theta": None if self.side_theta is None else {k: v.tolist() for k, v in self.side_theta.items()},
            "side_support": self.side_support,
            "side_shrink_rows": SIDE_SHRINK_ROWS if self.side_shrunk else None,
            "vector_l2": VECTOR_L2 if self.method == "vector" else None,
        }


def _fit_calibration(history: pd.DataFrame, method: str, side_shrunk: bool) -> Calibration:
    if method == "none":
        return Calibration(method, False, None, None, None)
    probs = history.loc[:, PROBS].to_numpy(float)
    labels = history.event.to_numpy(int)
    fitter = _fit_temperature if method == "temperature" else _fit_vector
    global_theta = fitter(probs, labels)
    if not side_shrunk:
        return Calibration(method, False, global_theta, None, None)
    parameters: dict[str, np.ndarray] = {}
    support: dict[str, int] = {}
    for side in ("long", "short"):
        part = history.loc[history.side_name.eq(side)]
        support[side] = int(len(part))
        parameters[side] = fitter(part.loc[:, PROBS].to_numpy(float), part.event.to_numpy(int))
    return Calibration(method, True, global_theta, parameters, support)


def _fit_payoffs(history: pd.DataFrame) -> dict[str, np.ndarray]:
    global_mean = history.groupby("event", observed=True).gross_bps.mean().reindex(range(3))
    if global_mean.isna().any():
        raise RuntimeError("payoff history has no observed samples for an event")
    result = {"global": global_mean.to_numpy(float)}
    for side in ("long", "short"):
        local = history.loc[history.side_name.eq(side)].groupby("event", observed=True).gross_bps.agg(["mean", "count"]).reindex(range(3))
        count = local["count"].fillna(0.0).to_numpy(float)
        mean = local["mean"].fillna(global_mean).to_numpy(float)
        result[side] = (count * mean + 2000.0 * result["global"]) / (count + 2000.0)
    return result


def _apply_b2(calibrated: np.ndarray, sides: np.ndarray, payoff: dict[str, np.ndarray]) -> np.ndarray:
    matrix = np.empty_like(calibrated)
    matrix[sides == "long"] = payoff["long"]
    matrix[sides == "short"] = payoff["short"]
    return np.einsum("ij,ij->i", calibrated, matrix) - 100.0


def _metrics(frame: pd.DataFrame, score: str) -> list[dict]:
    ordered = frame.sort_values([score, "candidate_id"], ascending=[False, True], kind="mergesort")
    rows = []
    for q in TOPS:
        selected = ordered.head(int(np.ceil(len(ordered) * q)))
        rows.append({"top_fraction": q, "n": int(len(selected)), "gross_bps": float(selected.gross_bps.mean()),
                     "net_bps": float(selected.net_bps.mean()), "long_n": int(selected.side_name.eq("long").sum()),
                     "short_n": int(selected.side_name.eq("short").sum())})
    return rows


def _probability_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    onehot = np.eye(3)[labels]
    return {"brier": float(np.mean(np.sum((probabilities - onehot) ** 2, axis=1))),
            "log_loss": _nll(probabilities, labels),
            "mean_probability": {EVENTS[i]: float(probabilities[:, i].mean()) for i in range(3)},
            "realised_frequency": {EVENTS[i]: float((labels == i).mean()) for i in range(3)}}


def _read(args: argparse.Namespace) -> pd.DataFrame:
    prediction_parts = []
    for side in ("long", "short"):
        prediction_parts.append(pd.read_parquet(args.base_root / side / "target_screen_predictions.parquet",
                                                columns=["candidate_id", "__ts__", "side_name", "gross_bps", "net_bps", "score_bps", *PROBS]))
    prediction = pd.concat(prediction_parts, ignore_index=True)
    # The panel spans several years whereas this frozen base emits only
    # Apr--Nov 2024.  Filter each partition immediately, instead of first
    # concatenating the whole panel (which is both wasteful and can exceed the
    # research host's memory limit).  This is a row-identity filter, not a
    # time/value filter, so it cannot change the calibration population.
    wanted = prediction[["candidate_id"]]
    label_parts = []
    for path in sorted((args.panel / "parts").glob("*.parquet")):
        part = pd.read_parquet(path, columns=["candidate_id", "t2_tp3_sl2_event"])
        label_parts.append(part.merge(wanted, on="candidate_id", how="inner", validate="one_to_one"))
    labels = pd.concat(label_parts, ignore_index=True)
    data = prediction.merge(labels.rename(columns={"t2_tp3_sl2_event": "event"}), on="candidate_id", validate="one_to_one")
    data["__ts__"] = pd.to_datetime(data["__ts__"], utc=True)
    data["event"] = data.event.astype(int)
    if not data.event.between(0, 2).all():
        raise RuntimeError("unexpected TP3/SL2 event code")
    data["__label_available_at__"] = data["__ts__"] + pd.Timedelta(hours=12)
    return data.sort_values(["__ts__", "candidate_id"], kind="mergesort").reset_index(drop=True)


def _prequential_scores(data: pd.DataFrame, methods: list[tuple[str, bool]], *, first_score: pd.Timestamp,
                         freeze_at: pd.Timestamp, update_days: int) -> tuple[pd.DataFrame, dict]:
    """Score every row from ``first_score`` with a prior-resolved frozen block.

    The first score of a block is deliberately *after* its fit cutoff.  Within
    a block, parameters are frozen.  This is cheap enough to be reproducible
    and avoids an accidental same-day outcome dependency.
    """
    output: list[pd.DataFrame] = []
    detail: dict[str, list[dict]] = {Calibration(m, s, None, None, None).name: [] for m, s in methods}
    dates = pd.date_range(first_score.floor("D"), freeze_at, freq=f"{update_days}D", tz="UTC")
    for start in dates:
        stop = min(start + pd.Timedelta(days=update_days), freeze_at)
        rows = data[(data.__ts__.ge(start)) & (data.__ts__.lt(stop))].copy()
        if rows.empty:
            continue
        history = data[data.__label_available_at__.lt(start)]
        if len(history) < 20_000:
            continue
        payoff = _fit_payoffs(history)
        for method, side_shrunk in methods:
            calibrator = _fit_calibration(history, method, side_shrunk)
            pcal = calibrator.predict(rows.loc[:, PROBS].to_numpy(float), rows.side_name.to_numpy(str))
            rows[calibrator.name] = _apply_b2(pcal, rows.side_name.to_numpy(str), payoff)
            detail[calibrator.name].append({"fit_cutoff": str(start), "score_start": str(start), "score_end_exclusive": str(stop),
                                            "fit_rows": int(len(history)), "calibrator": calibrator.manifest(),
                                            "payoff_gross_bps": {k: v.tolist() for k, v in payoff.items()}})
        output.append(rows)
    return pd.concat(output, ignore_index=True), detail


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dev-start", default="2024-06-01")
    parser.add_argument("--oos-start", default="2024-08-01")
    parser.add_argument("--oos-end", default="2024-12-01")
    parser.add_argument("--update-days", type=int, default=14)
    args = parser.parse_args()
    dev_start, oos_start, oos_end = (pd.Timestamp(x, tz="UTC") for x in (args.dev_start, args.oos_start, args.oos_end))
    if not dev_start < oos_start < oos_end or args.update_days < 1:
        raise ValueError("require dev-start < oos-start < oos-end and positive update-days")
    data = _read(args)
    methods = [("none", False), ("temperature", False), ("temperature", True), ("vector", False), ("vector", True)]
    dev, blocks = _prequential_scores(data, methods, first_score=dev_start, freeze_at=oos_start, update_days=args.update_days)
    # The OOS calibrator and B2 map are fitted once from the entire causal
    # history ending just before the OOS boundary, then never changed.
    history = data[data.__label_available_at__.lt(oos_start)].copy()
    oos = data[(data.__ts__.ge(oos_start)) & (data.__ts__.lt(oos_end))].copy()
    final_fit: dict[str, dict] = {}
    oos_probabilities: dict[str, np.ndarray] = {}
    payoff = _fit_payoffs(history)
    for method, side_shrunk in methods:
        calibrator = _fit_calibration(history, method, side_shrunk)
        name = calibrator.name
        pcal = calibrator.predict(oos.loc[:, PROBS].to_numpy(float), oos.side_name.to_numpy(str))
        oos_probabilities[name] = pcal
        oos[name] = _apply_b2(pcal, oos.side_name.to_numpy(str), payoff)
        final_fit[name] = {"calibrator": calibrator.manifest(), "payoff_gross_bps": {k: v.tolist() for k, v in payoff.items()}}
    rows = []
    manifest_metrics = {"development": {}, "oos": {}}
    for method, side_shrunk in methods:
        name = Calibration(method, side_shrunk, None, None, None).name
        for split_name, frame in (("development", dev), ("oos", oos)):
            metrics = _metrics(frame, name)
            diag = {"score_net_spearman_ic": float(spearmanr(frame[name], frame.net_bps).statistic), "tail": metrics}
            if split_name == "oos":
                diag["event_probability"] = _probability_metrics(oos_probabilities[name], frame.event.to_numpy(int))
            manifest_metrics[split_name][name] = diag
            for row in metrics:
                rows.append({"split": split_name, "variant": name, **row})
    args.out.mkdir(parents=True, exist_ok=True)
    dev.to_parquet(args.out / "development_prequential_scores.parquet", index=False)
    oos.to_parquet(args.out / "oos_frozen_scores.parquet", index=False)
    pd.DataFrame(rows).to_parquet(args.out / "metrics.parquet", index=False)
    manifest = {"schema": "full_universe_stage51_causal_event_probability_calibration_v1",
                "contract": {"frozen_base": str(args.base_root), "geometry": "T2 TP3/SL2", "entry_exit": "inherited frozen base; first barrier else H12", "cost_bps": 100.0,
                             "selection": "global pooled long/short/timestamp, candidate_id tie break"},
                "causality": {"development": "14-day frozen calibration blocks fit only on rows whose H12 label resolved before block start", "oos": "one calibrator and B2 payoff map fit strictly before oos_start then frozen", "oos_start": str(oos_start), "oos_end_exclusive": str(oos_end), "label_resolution": "decision timestamp + 12 hours"},
                "fixed_hyperparameters": {"update_days": args.update_days, "vector_l2": VECTOR_L2, "side_shrink_rows": SIDE_SHRINK_ROWS},
                "development_blocks": blocks, "oos_frozen_fit": final_fit, "metrics": manifest_metrics}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    table = pd.DataFrame(rows).pivot(index=["split", "variant"], columns="top_fraction", values="net_bps")
    report = [
        "# Stage 5.1 — Causal event-probability calibration",
        "",
        "## Contract",
        "",
        "- Frozen long/short TP3/SL2 base predictions; inherited next-bar entry and first-barrier/H12 exit.",
        "- B2 expected value is recomputed from calibrated `(P(upper), P(lower), P(timeout))`, causal side-shrunk conditional gross payoffs, then a fixed 100 bps cost.",
        "- Development is five 14-day frozen blocks from 2024-06-01 to 2024-08-01. Each block fits only outcomes resolved before its first timestamp.",
        "- Aug–Nov is a single untouched frozen replay: both probability calibrator and payoff map use only outcomes resolved before 2024-08-01.",
        "- Ranking is one global long/short/timestamp pool with deterministic candidate-ID ties.",
        "",
        "## Global net bps per trade",
        "",
        "```text",
        table.round(3).to_string(),
        "```",
        "",
        "## OOS probability and score diagnostics",
        "",
        "| variant | net Spearman IC | multiclass Brier | multiclass log loss |",
        "|---|---:|---:|---:|",
    ]
    for method, side_shrunk in methods:
        name = Calibration(method, side_shrunk, None, None, None).name
        d = manifest_metrics["oos"][name]
        pdiag = d["event_probability"]
        report.append(f"| {name} | {d['score_net_spearman_ic']:.6f} | {pdiag['brier']:.6f} | {pdiag['log_loss']:.6f} |")
    report += [
        "",
        "## Decision",
        "",
        "Development's largest top-10 gain came from side-shrunk vector scaling, but its frozen OOS top-10 result deteriorated materially. It is rejected. Global vector scaling produced a small OOS top-10 improvement but did not beat the uncalibrated control materially on the causal development top-10, so it is not promoted post hoc. The selected base representation remains uncalibrated B2.",
        "",
        "The complete fitted parameters, side support, block cutoffs, B2 payoff maps, metrics, and scores are in `manifest.json`, `metrics.parquet`, `development_prequential_scores.parquet`, and `oos_frozen_scores.parquet`.",
    ]
    (args.out / "STAGE51_REPORT.md").write_text("\n".join(report) + "\n")
    print(table.to_string())


if __name__ == "__main__":
    main()
