#!/usr/bin/env python3
"""Stage-E E6 complete-candidate overlay using already-frozen v9 OOF/OOS decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data_perp/artifacts"
ALIGNMENT = ART / "historical_exact_h12_alignment_sidecar_research_only_20260731_v1/alignment_sidecar.parquet"
EVENTS = ART / "historical_exact_h12_postcost_events_20260731_v1/postcost_events.parquet"
REPLAY = ART / "stage_d_compact_action_model_20260731_v9/stage_d_action_policy_replay.parquet"
V9_MANIFEST = ART / "stage_d_compact_action_model_20260731_v9/run_manifest.json"
DEFAULT_OUTPUT = ART / "stage_e_full_candidate_overlay_20260731_v1"
START = pd.Timestamp("2024-04-01T00:00:00Z")
# Exclude the final entry day so every 12h candidate that clears has its frozen
# action decision inside the already-scored v9 period. This boundary is fixed
# by prediction availability, never by economics.
END = pd.Timestamp("2024-11-30T00:00:00Z")
SEED = 20260731
BOOTSTRAPS = 1000


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def bootstrap(frame: pd.DataFrame) -> dict[str, Any]:
    day = frame.assign(day=frame.decision_ts.dt.floor("D")).groupby("day").agg(delta=("incremental_net_bps", "sum"), rows=("candidate_id", "size"))
    rng = np.random.default_rng(SEED)
    values = np.empty(BOOTSTRAPS, dtype=float)
    sums = day.delta.to_numpy(); counts = day.rows.to_numpy()
    for i in range(BOOTSTRAPS):
        ix = rng.integers(0, len(day), len(day))
        values[i] = sums[ix].sum() / counts[ix].sum()
    return {
        "seed": SEED, "reps": BOOTSTRAPS, "utc_day_blocks": len(day),
        "mean_bps": float(values.mean()), "ci_95_bps": [float(np.quantile(values, .025)), float(np.quantile(values, .975))],
        "probability_positive": float(np.mean(values > 0)),
    }


def payoff_metrics(values: pd.Series) -> dict[str, float]:
    pos = values[values > 0]; neg = values[values < 0]
    return {
        "win_rate": float((values > 0).mean()),
        "payoff_ratio": float(pos.mean() / abs(neg.mean())) if len(pos) and len(neg) else float("nan"),
        "profit_factor": float(pos.sum() / abs(neg.sum())) if len(pos) and len(neg) else float("nan"),
    }


def run(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    alignment = pd.read_parquet(ALIGNMENT, columns=[
        "candidate_id", "symbol", "side", "decision_ts", "label_end_ts", "exact_h12_gross_bps", "row_cost_bps", "exact_h12_net_bps", "exit_reason",
    ])
    events = pd.read_parquet(EVENTS, columns=["candidate_id", "postcost_h0_event", "postcost_h0_favorable_minute"])
    replay = pd.read_parquet(REPLAY)
    replay = replay.loc[(replay.arm == "compact_readmitted") & (replay.action_threshold_bps == 0.0)].copy()
    if replay.candidate_id.duplicated().any():
        raise ValueError("frozen selected replay must be one row per clear candidate")
    population = alignment.loc[alignment.decision_ts.ge(START) & alignment.decision_ts.lt(END)].merge(events, on="candidate_id", validate="one_to_one")
    if population.candidate_id.duplicated().any():
        raise ValueError("entry population is not unique")
    clear_eligible = population.postcost_h0_event.eq("clear_cost_first") & population.postcost_h0_favorable_minute.lt(718)
    expected = set(population.loc[clear_eligible, "candidate_id"].astype(str))
    available = set(replay.candidate_id.astype(str))
    if not expected.issubset(available):
        raise ValueError(f"frozen replay coverage missing {len(expected - available)} clear candidates")
    action = replay[[
        "candidate_id", "action_decision_ts", "action", "predicted_delta_continue_bps", "predicted_continue_probability",
        "policy_gross_bps", "policy_cost_bps", "policy_net_bps", "loss_avoided_correct_exit_bps", "false_exit_opportunity_cost_bps",
    ]]
    frame = population.merge(action, on="candidate_id", how="left", validate="one_to_one")
    has_action = frame.candidate_id.astype(str).isin(expected)
    if frame.loc[has_action, "action"].isna().any() or frame.loc[~has_action, "action"].notna().any():
        raise ValueError("only exact first-clear eligible rows may receive overlay actions")
    frame["p0_gross_bps"] = frame.exact_h12_gross_bps
    frame["p0_cost_bps"] = frame.row_cost_bps
    frame["p0_net_bps"] = frame.exact_h12_net_bps
    frame["p1_gross_bps"] = np.where(has_action, frame.policy_gross_bps, frame.p0_gross_bps)
    frame["p1_cost_bps"] = np.where(has_action, frame.policy_cost_bps, frame.p0_cost_bps)
    frame["p1_net_bps"] = np.where(has_action, frame.policy_net_bps, frame.p0_net_bps)
    frame["incremental_net_bps"] = frame.p1_net_bps - frame.p0_net_bps
    frame["overlay_action_eligible"] = has_action
    frame["overlay_changed_to_exit"] = has_action & frame.action.eq("EXIT_NOW")
    if not np.allclose(frame.p0_gross_bps - frame.p0_cost_bps, frame.p0_net_bps, atol=1e-6):
        raise ValueError("P0 cost arithmetic drift")
    if not np.allclose(frame.p1_gross_bps - frame.p1_cost_bps, frame.p1_net_bps, atol=1e-6):
        raise ValueError("P1 cost arithmetic drift")
    if not np.allclose(frame.loc[~has_action, "p1_net_bps"], frame.loc[~has_action, "p0_net_bps"], atol=0):
        raise ValueError("non-clear candidates changed")
    frame["month"] = frame.decision_ts.dt.strftime("%Y-%m")
    boot = bootstrap(frame)
    overall = {
        "candidate_rows": len(frame), "symbols": int(frame.symbol.nunique()), "months": int(frame.month.nunique()),
        "clear_event_rate": float(frame.postcost_h0_event.eq("clear_cost_first").mean()),
        "action_eligible_rate": float(has_action.mean()), "overlay_exit_rate_all_candidates": float(frame.overlay_changed_to_exit.mean()),
        "p0_gross_bps": float(frame.p0_gross_bps.mean()), "p0_cost_bps": float(frame.p0_cost_bps.mean()), "p0_net_bps": float(frame.p0_net_bps.mean()),
        "p1_gross_bps": float(frame.p1_gross_bps.mean()), "p1_cost_bps": float(frame.p1_cost_bps.mean()), "p1_net_bps": float(frame.p1_net_bps.mean()),
        "incremental_net_bps": float(frame.incremental_net_bps.mean()),
        "event_shares": {str(k): float(v) for k, v in frame.postcost_h0_event.value_counts(normalize=True).items()},
        "exit_reason_shares": {str(k): float(v) for k, v in frame.exit_reason.value_counts(normalize=True, dropna=False).items()},
        "p0_trade_metrics": payoff_metrics(frame.p0_net_bps), "p1_trade_metrics": payoff_metrics(frame.p1_net_bps),
        "bootstrap": boot,
    }
    slices = []
    for dimension, groups in (("side", frame.groupby("side")), ("month", frame.groupby("month"))):
        for value, part in groups:
            slices.append({"dimension": dimension, "value": str(value), "rows": len(part), "p0_net_bps": float(part.p0_net_bps.mean()), "p1_net_bps": float(part.p1_net_bps.mean()), "incremental_net_bps": float(part.incremental_net_bps.mean())})
    latest = max(frame.month)
    latest_part = frame.loc[frame.month.eq(latest)]
    slices.append({"dimension": "latest_period", "value": latest, "rows": len(latest_part), "p0_net_bps": float(latest_part.p0_net_bps.mean()), "p1_net_bps": float(latest_part.p1_net_bps.mean()), "incremental_net_bps": float(latest_part.incremental_net_bps.mean())})
    avoided = float(frame.loss_avoided_correct_exit_bps.fillna(0).mean())
    sacrificed = float(frame.false_exit_opportunity_cost_bps.fillna(0).mean())
    waterfall = pd.DataFrame([
        {"order": 0, "component": "P0 net", "bps_per_candidate": overall["p0_net_bps"]},
        {"order": 1, "component": "+ avoided giveback", "bps_per_candidate": avoided},
        {"order": 2, "component": "- sacrificed retained upside", "bps_per_candidate": -sacrificed},
        {"order": 3, "component": "- additional latency/slippage", "bps_per_candidate": 0.0},
        {"order": 4, "component": "= P1 net", "bps_per_candidate": overall["p1_net_bps"]},
    ])
    if not np.isclose(overall["p0_net_bps"] + avoided - sacrificed, overall["p1_net_bps"], atol=1e-9):
        raise ValueError("overlay waterfall does not reconcile")
    keep = [
        "candidate_id", "symbol", "side", "decision_ts", "label_end_ts", "month", "postcost_h0_event", "postcost_h0_favorable_minute",
        "overlay_action_eligible", "overlay_changed_to_exit", "action_decision_ts", "action", "predicted_delta_continue_bps",
        "p0_gross_bps", "p0_cost_bps", "p0_net_bps", "p1_gross_bps", "p1_cost_bps", "p1_net_bps", "incremental_net_bps",
    ]
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        frame[keep].to_parquet(stage / "stage_e_full_candidate_overlay.parquet", index=False, compression="zstd")
        waterfall.to_parquet(stage / "stage_e_full_candidate_waterfall.parquet", index=False, compression="zstd")
        pd.DataFrame(slices).to_parquet(stage / "stage_e_full_candidate_slices.parquet", index=False, compression="zstd")
        dump(stage / "stage_e_full_candidate_overlay_summary.json", overall)
        outputs = {p.name: sha(p) for p in stage.iterdir()}
        dump(stage / "run_manifest.json", {
            "schema": "stage_e_full_candidate_overlay_v1", "status": "COMPLETE_RESEARCH_ONLY_NO_ENTRY_OR_PORTFOLIO_CHANGE",
            "period": {"entry_start": str(START), "entry_end_exclusive": str(END)},
            "entry_population_sha256": hashlib.sha256("\n".join(frame.candidate_id.astype(str)).encode()).hexdigest(),
            "p0_p1_entry_population_identical": True, "non_clear_policy_unchanged": True, "only_first_clear_action_changed": True,
            "inputs": {str(p): sha(p) for p in (ALIGNMENT, EVENTS, REPLAY, V9_MANIFEST)},
            "code_sha256": {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}, "outputs_sha256": outputs,
            "summary": overall, "limitations": ["candidate-level evaluation only", "no portfolio or sizing logic", "uses frozen historical v9 predictions; not second OOS"],
        })
        (stage / "manifest.sha256").write_text(f"{sha(stage / 'run_manifest.json')}  run_manifest.json\n")
        os.replace(stage, output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return overall


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output.resolve()), indent=2, default=str))
