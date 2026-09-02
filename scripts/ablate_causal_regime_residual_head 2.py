#!/usr/bin/env python3
"""One strict forward residual-head ablation: raw inputs vs raw + causal state."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.causal_execution_regimes import CausalRegimeStateModel  # noqa: E402
from scripts.diagnose_causal_execution_ev_regimes import (  # noqa: E402
    FORWARD, HEAD_AVAILABILITY, ID, OLD, STATE_FEATURES, _read, _state_availability_mask,
)

OUTPUT = ROOT / "data_perp/artifacts/causal_regime_residual_head_ablation_july13_20260726_v1"
TRAIN_CUTOFF = pd.Timestamp("2026-07-01T00:00:00Z")
EVAL_START = pd.Timestamp("2026-07-13T00:00:00Z")


def _metrics(frame: pd.DataFrame, score: str) -> dict[str, float | int]:
    net = frame["execution_net_ev_12h"]
    n = max(1, int(np.ceil(len(frame) * 0.10)))
    top = frame.nlargest(n, score)
    return {
        "rows": int(len(frame)), "mean_net_ev": float(net.mean()),
        "rank_spearman_score_vs_net_ev": float(frame[score].corr(net, method="spearman")),
        "top10_rows": int(len(top)), "top10_net_ev": float(top["execution_net_ev_12h"].mean()),
        "top10_lift_vs_unconditional": float(top["execution_net_ev_12h"].mean() - net.mean()),
        "top10_positive_rate": float((top["execution_net_ev_12h"] > 0).mean()),
    }


def _ridge_predict(train: pd.DataFrame, evaluation: pd.DataFrame, features: list[str], target: pd.Series, ridge: float = 10.0) -> np.ndarray:
    """Bounded deterministic residual head used only for this input ablation."""
    x_train = train[features].to_numpy(dtype=float); x_eval = evaluation[features].to_numpy(dtype=float)
    median = np.nanmedian(x_train, axis=0); median = np.where(np.isfinite(median), median, 0.0)
    x_train = np.where(np.isfinite(x_train), x_train, median); x_eval = np.where(np.isfinite(x_eval), x_eval, median)
    mean = x_train.mean(axis=0); scale = np.maximum(x_train.std(axis=0), 1e-6)
    x_train = (x_train - mean) / scale; x_eval = (x_eval - mean) / scale
    x_train = np.c_[np.ones(len(x_train)), x_train]; x_eval = np.c_[np.ones(len(x_eval)), x_eval]
    penalty = np.eye(x_train.shape[1]) * ridge; penalty[0, 0] = 0.0
    coef = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ target.to_numpy(dtype=float))
    return x_eval @ coef


def run(output_dir: Path = OUTPUT) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    columns = list(dict.fromkeys([*ID, "execution_decision_utc", "execution_label_end_utc", "execution_net_ev_12h", *STATE_FEATURES, *HEAD_AVAILABILITY.values()]))
    data = pd.concat([_read(OLD, columns), _read(FORWARD, columns)], ignore_index=True).drop_duplicates(ID, keep="last")
    data = data.loc[_state_availability_mask(data)].copy()
    # Both the unsupervised state and supervised head see only resolved labels
    # whose decision and 12h execution label end before the July-1 cutoff.
    train = data.loc[(data["execution_decision_utc"] < TRAIN_CUTOFF) & (data["execution_label_end_utc"] < TRAIN_CUTOFF)].copy()
    evaluation = data.loc[data["execution_decision_utc"] >= EVAL_START].copy()
    reports: dict[str, object] = {}; rows: list[pd.DataFrame] = []
    for side in ("long", "short"):
        tr = train.loc[train.side_name.eq(side)].copy(); ev = evaluation.loc[evaluation.side_name.eq(side)].copy()
        state = CausalRegimeStateModel.fit(tr, STATE_FEATURES)
        state_train = state.transform(tr); state_eval = state.transform(ev)
        extra = list(state.predictor_feature_columns)
        tr = pd.concat([tr.reset_index(drop=True), state_train.reset_index(drop=True)], axis=1)
        ev = pd.concat([ev.reset_index(drop=True), state_eval.reset_index(drop=True)], axis=1)
        target = tr["execution_net_ev_12h"] - tr["existing_alpha_ev"]
        side_report: dict[str, object] = {"train_rows": int(len(tr)), "eval_rows": int(len(ev)), "state_k": state.selected_k, "models": {}}
        for name, features in (("baseline_raw", STATE_FEATURES), ("plus_causal_regime_inputs", [*STATE_FEATURES, *extra])):
            score_col = f"{name}_score"
            ev[score_col] = ev["existing_alpha_ev"] + _ridge_predict(tr, ev, features, target)
            side_report["models"][name] = _metrics(ev, score_col)
        reports[side] = side_report
        ev["side_name"] = side; rows.append(ev.loc[:, [*ID, "execution_net_ev_12h", "baseline_raw_score", "plus_causal_regime_inputs_score", *extra]])
    scored = pd.concat(rows, ignore_index=True)
    global_report = {name: _metrics(scored, f"{name}_score") for name in ("baseline_raw", "plus_causal_regime_inputs")}
    output_dir.mkdir(parents=True); scored.to_parquet(output_dir / "strict_forward_predictions.parquet", index=False)
    payload = {
        "contract": "Exploratory strict single-cutoff ablation only: state geometry and residual models train per-side on May-Jun decisions with execution labels resolved before Jul1; evaluation begins Jul13. No outcome/calendar/weights are used in state fitting. State ID and transition labels are excluded; posterior/geometry only are added.",
        "reports": reports, "global": global_report,
        "output": str(output_dir / "strict_forward_predictions.parquet"),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
