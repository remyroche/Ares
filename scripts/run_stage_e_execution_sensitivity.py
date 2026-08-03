#!/usr/bin/env python3
"""Stage-E4 frozen-decision execution sensitivity.

This runner is deliberately an evaluator, not a trainer.  It binds the
canonical Stage-D v9 final-OOS decisions and replays only the EXIT_NOW arm
under predeclared fill perturbations.  The continuation counterfactual,
candidate population, action assignments, model predictions, and margin are
never changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data_perp/artifacts"
V9 = ART / "stage_d_compact_action_model_20260731_v9"
D0 = ART / "stage_d_action_counterfactuals_20260731_v2"
V9_REPLAY = V9 / "stage_d_action_policy_replay.parquet"
V9_MANIFEST = V9 / "run_manifest.json"
COUNTERFACTUALS = D0 / "stage_d_action_counterfactuals.parquet"
POSTCOST_EVENTS = ART / "historical_exact_h12_postcost_events_20260731_v1/postcost_events.parquet"
PATHS = (
    ART / "failure_2022_2023_pf_exact1m_paths_20260730_v1/paths.parquet",
    ART / "failure_2024_exact1m_paths_20260730_v2/paths.parquet",
)
DEFAULT_OUTPUT = ART / "stage_e_execution_sensitivity_20260731_v4"

LATENCIES = (0, 1, 2, 5, 10)
SLIPPAGES = (0.0, 10.0, 25.0, 50.0)
ESTIMATOR_STRESSES = (-25.0, 0.0, 25.0)
MATERIAL_NEXT_FILL_BPS = 25.0
CLOSE_ADVERSE_BARRIER_BPS = 25.0
LARGE_JUMP_FRACTION = 0.50
HORIZON_MINUTES = 720
SCHEMA = "stage_e4_frozen_decision_execution_sensitivity_v1"


class ContractError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def id_sha(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, values))).encode()).hexdigest()


def dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def adverse_exit_fill_proxy(side: str, raw_open: np.ndarray, half_spread_bps: np.ndarray) -> np.ndarray:
    """Vectorised exact frozen close-trigger adverse fill proxy."""
    raw = np.asarray(raw_open, dtype=np.float64)
    half = np.maximum(np.nan_to_num(np.asarray(half_spread_bps, dtype=np.float64), nan=0.0), 0.0) / 10_000.0
    is_long = np.asarray(side) == "long"
    spread_px = np.where(is_long, raw * (1.0 - half), raw * (1.0 + half))
    gap = np.minimum(spread_px * 15.0 / 10_000.0, spread_px * 75.0 / 10_000.0)
    # Match the canonical helper's final float32 conversion.
    return np.where(is_long, spread_px - gap, spread_px + gap).astype(np.float32).astype(np.float64)


def load_frozen_decisions() -> pd.DataFrame:
    manifest = json.loads(V9_MANIFEST.read_text())
    sealed = manifest["outputs_sha256"].get(V9_REPLAY.name)
    if sealed != sha256(V9_REPLAY):
        raise ContractError("canonical v9 replay seal mismatch")
    if float(manifest["development_selected_margin_bps"]) != 0.0:
        raise ContractError("Stage-E4 requires the frozen zero-bps v9 margin")
    replay = pd.read_parquet(V9_REPLAY)
    frozen = replay.loc[
        replay.split.eq("final_oos") & replay.selected_margin_from_development.astype(bool)
    ].copy()
    if frozen.candidate_id.duplicated().any() or frozen.empty:
        raise ContractError("v9 final-OOS frozen decisions are not unique")
    if not frozen.action_threshold_bps.eq(0.0).all():
        raise ContractError("non-zero action margin in selected v9 replay")
    expected = np.where(
        frozen.predicted_delta_continue_bps.gt(0.0), "CONTINUE_FROZEN_POLICY", "EXIT_NOW"
    )
    if not np.array_equal(expected, frozen.action.to_numpy()):
        raise ContractError("sealed actions do not match the frozen v9 decision rule")
    return frozen


def _path_rows(candidate_ids: set[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    chunks = []
    for path in PATHS:
        schema = pq.ParquetFile(path).schema_arrow
        timestamp_type = schema.field("__ts__").type
        lo: Any = start.to_pydatetime() if "timestamp" in str(timestamp_type) else start.value
        hi: Any = end.to_pydatetime() if "timestamp" in str(timestamp_type) else end.value
        table = pq.read_table(
            path,
            columns=["candidate_id", "execution_future_path"],
            filters=[("__ts__", ">=", lo), ("__ts__", "<", hi)],
        )
        part = table.to_pandas()
        part = part.loc[part.candidate_id.astype(str).isin(candidate_ids)]
        if not part.empty:
            chunks.append(part)
    if not chunks:
        raise ContractError("no immutable exact-1m paths matched frozen v9 rows")
    result = pd.concat(chunks, ignore_index=True)
    if result.candidate_id.duplicated().any():
        raise ContractError("duplicate path payloads for frozen candidates")
    missing = candidate_ids - set(result.candidate_id.astype(str))
    if missing:
        raise ContractError(f"missing {len(missing)} immutable paths")
    return result


def decode_needed_path_state(raw: str, clear_i: int) -> dict[str, float]:
    payload = json.loads(raw)
    arrays = {k: np.asarray(payload[k], dtype=np.float64) for k in ("open", "high", "low", "close")}
    if any(v.shape != (HORIZON_MINUTES,) for v in arrays.values()):
        raise ContractError("path payload is not exact 720x1m OHLC")
    i = int(clear_i)
    if not 0 <= i <= 717:
        raise ContractError("invalid canonical actionable clear index")
    result = {f"raw_open_latency_{latency}": float(arrays["open"][i + 2 + latency]) if i + 2 + latency < HORIZON_MINUTES else np.nan for latency in LATENCIES}
    result.update({
        "decision_time_price": float(arrays["close"][i]),
        "clear_bar_open": float(arrays["open"][i]),
        "clear_bar_high": float(arrays["high"][i]),
        "clear_bar_low": float(arrays["low"][i]),
    })
    return result


def materialize_replay_frame() -> tuple[pd.DataFrame, dict[str, Any]]:
    frozen = load_frozen_decisions()
    cf = pd.read_parquet(COUNTERFACTUALS)
    event_geometry = pd.read_parquet(
        POSTCOST_EVENTS,
        columns=["candidate_id", "fixed_cost_bps", "adverse_barrier_pct", "postcost_h0_event"],
    )
    event_geometry = event_geometry.loc[event_geometry.postcost_h0_event.eq("clear_cost_first")]
    cols = [
        "candidate_id", "first_clear_bar_index", "entry_executable_price",
        "exit_half_spread_bps", "known_row_cost_bps", "action_exit_raw_open",
        "action_exit_executable_price", "action_decision_ts", "action_execution_ts",
    ]
    frame = frozen.merge(cf[cols], on="candidate_id", how="left", validate="one_to_one", suffixes=("", "_d0"))
    frame = frame.merge(event_geometry.drop(columns="postcost_h0_event"), on="candidate_id", how="left", validate="one_to_one")
    if frame.first_clear_bar_index.isna().any():
        raise ContractError("D0 counterfactual join is incomplete")
    start = pd.Timestamp(frame.action_decision_ts.min()).floor("D") - pd.Timedelta(days=1)
    end = pd.Timestamp(frame.action_decision_ts.max()).ceil("D") + pd.Timedelta(days=1)
    paths = _path_rows(set(frame.candidate_id.astype(str)), start, end)
    decoded = []
    clear_by_id = frame.set_index("candidate_id").first_clear_bar_index
    for row in paths.itertuples(index=False):
        state = decode_needed_path_state(row.execution_future_path, int(clear_by_id.loc[row.candidate_id]))
        state["candidate_id"] = str(row.candidate_id)
        decoded.append(state)
    frame = frame.merge(pd.DataFrame(decoded), on="candidate_id", validate="one_to_one")
    # Canonical fill parity is a hard prerequisite.
    canonical_fill = adverse_exit_fill_proxy(
        frame.side.to_numpy(), frame.raw_open_latency_0.to_numpy(), frame.exit_half_spread_bps.to_numpy()
    )
    if not np.array_equal(canonical_fill.astype(np.float32), frame.action_exit_executable_price.to_numpy().astype(np.float32)):
        raise ContractError("independent latency-0 fill does not match sealed D0 fill")

    side_sign = np.where(frame.side.eq("long"), 1.0, -1.0)
    favorable_clear_extreme = np.where(frame.side.eq("long"), frame.clear_bar_high, frame.clear_bar_low)
    adverse_clear_extreme = np.where(frame.side.eq("long"), frame.clear_bar_low, frame.clear_bar_high)
    favorable_move_bps = side_sign * (favorable_clear_extreme / frame.entry_executable_price - 1.0) * 10_000.0
    adverse_move_bps = side_sign * (adverse_clear_extreme / frame.entry_executable_price - 1.0) * 10_000.0
    clear_bar_open_move_bps = side_sign * (frame.clear_bar_open / frame.entry_executable_price - 1.0) * 10_000.0
    if frame[["fixed_cost_bps", "adverse_barrier_pct"]].isna().any().any():
        raise ContractError("frozen post-cost event geometry join is incomplete")
    # H0 clear is the exact fixed-cost hurdle (100 bps in the frozen pack),
    # while the adverse geometry comes from the row's frozen barrier.
    frame["clear_jump_fraction"] = np.maximum(favorable_move_bps - np.maximum(clear_bar_open_move_bps, 0.0), 0.0) / frame.fixed_cost_bps
    adverse_barrier_bps = frame.adverse_barrier_pct * 10_000.0
    frame["distance_to_adverse_barrier_bps"] = np.abs(adverse_move_bps + adverse_barrier_bps)
    frame["next_fill_divergence_bps"] = np.abs(frame.action_exit_raw_open / frame.decision_time_price - 1.0) * 10_000.0
    frame["slice_large_clear_jump"] = frame.clear_jump_fraction.ge(LARGE_JUMP_FRACTION)
    frame["slice_clear_adverse_geometry_close"] = frame.distance_to_adverse_barrier_bps.le(CLOSE_ADVERSE_BARRIER_BPS)
    frame["slice_next_fill_materially_differs"] = frame.next_fill_divergence_bps.ge(MATERIAL_NEXT_FILL_BPS)
    frame["max_latency_available"] = frame.first_clear_bar_index.le(HORIZON_MINUTES - 3 - max(LATENCIES))
    audit = {
        "v9_final_oos_rows": int(len(frame)),
        "v9_candidate_id_sha256": id_sha(frame.candidate_id),
        "v9_action_sha256": hashlib.sha256("\n".join(frame.action.astype(str)).encode()).hexdigest(),
        "common_support_rows": int(frame.max_latency_available.sum()),
        "tail_rows_without_10m_fill": int((~frame.max_latency_available).sum()),
        "canonical_fill_float32_exact": True,
    }
    return frame, audit


def replay_scenario(frame: pd.DataFrame, latency: int, slippage: float, estimator_stress: float) -> pd.DataFrame:
    raw = frame[f"raw_open_latency_{latency}"].to_numpy()
    if not np.isfinite(raw).all():
        raise ContractError("scenario contains unavailable latency fill")
    fill = adverse_exit_fill_proxy(frame.side.to_numpy(), raw, frame.exit_half_spread_bps.to_numpy())
    sign = np.where(frame.side.eq("long"), 1.0, -1.0)
    exit_gross_before_stress = sign * (fill / frame.entry_executable_price.to_numpy() - 1.0) * 10_000.0
    exit_gross = exit_gross_before_stress + float(estimator_stress)
    exit_cost = frame.known_row_cost_bps.to_numpy() + float(slippage)
    exit_net = exit_gross - exit_cost
    is_continue = frame.action.eq("CONTINUE_FROZEN_POLICY").to_numpy()
    z = frame[["candidate_id", "source_symbol", "side", "action_decision_ts", "action", "net_continue_bps"]].copy()
    z["latency_minutes"] = int(latency)
    z["added_exit_slippage_bps"] = float(slippage)
    z["exit_estimator_stress_bps"] = float(estimator_stress)
    z["replayed_exit_gross_bps"] = exit_gross
    z["replayed_exit_cost_bps"] = exit_cost
    z["replayed_exit_net_bps"] = exit_net
    z["policy_net_bps"] = np.where(is_continue, frame.net_continue_bps, exit_net)
    z["incremental_vs_always_continue_bps"] = z.policy_net_bps - z.net_continue_bps
    z["incremental_vs_always_exit_bps"] = z.policy_net_bps - exit_net
    return z


def aggregate(z: pd.DataFrame, population: str, slice_name: str = "all", slice_value: str = "ALL") -> dict[str, Any]:
    return {
        "population": population,
        "slice": slice_name,
        "slice_value": slice_value,
        "latency_minutes": int(z.latency_minutes.iloc[0]),
        "added_exit_slippage_bps": float(z.added_exit_slippage_bps.iloc[0]),
        "exit_estimator_stress_bps": float(z.exit_estimator_stress_bps.iloc[0]),
        "rows": int(len(z)),
        "candidate_id_sha256": id_sha(z.candidate_id),
        "continue_rate": float(z.action.eq("CONTINUE_FROZEN_POLICY").mean()),
        "exit_rate": float(z.action.eq("EXIT_NOW").mean()),
        "policy_net_bps": float(z.policy_net_bps.mean()),
        "always_continue_net_bps": float(z.net_continue_bps.mean()),
        "always_exit_net_bps": float(z.replayed_exit_net_bps.mean()),
        "uplift_vs_always_continue_bps": float(z.incremental_vs_always_continue_bps.mean()),
        "uplift_vs_always_exit_bps": float(z.incremental_vs_always_exit_bps.mean()),
    }


def run(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    frame, audit = materialize_replay_frame()
    common = frame.loc[frame.max_latency_available].copy()
    rows: list[dict[str, Any]] = []
    row_audit = []
    slice_columns = {
        "large_clear_jump": "slice_large_clear_jump",
        "clear_adverse_geometry_close": "slice_clear_adverse_geometry_close",
        "next_fill_materially_differs": "slice_next_fill_materially_differs",
    }
    for latency in LATENCIES:
        for slippage in SLIPPAGES:
            for stress in ESTIMATOR_STRESSES:
                replay = replay_scenario(common, latency, slippage, stress)
                rows.append(aggregate(replay, "fixed_common_support"))
                for name, col in slice_columns.items():
                    mask = common[col].to_numpy()
                    for value, selection in (("TRUE", mask), ("FALSE", ~mask)):
                        if selection.any():
                            rows.append(aggregate(replay.loc[selection], "fixed_common_support", name, value))
                if latency == 0 and slippage == 0.0 and stress == 0.0:
                    row_audit = replay.assign(population="fixed_common_support")

    # Full canonical population parity is reported separately; it must match
    # the sealed v9 policy and is not mixed into latency comparisons.
    full0 = replay_scenario(frame, 0, 0.0, 0.0)
    rows.append(aggregate(full0, "canonical_full_population"))
    expected = frame.policy_net_bps.to_numpy()
    if not np.allclose(full0.policy_net_bps, expected, rtol=0.0, atol=1e-5):
        raise ContractError("unperturbed full replay does not reproduce v9 policy net")
    audit["canonical_policy_parity_max_abs_bps"] = float(np.max(np.abs(full0.policy_net_bps - expected)))
    audit["frozen_actions_all_scenarios"] = True
    audit["known_row_cost_applied_once"] = True
    audit["incremental_slippage_applied_once"] = True

    results = pd.DataFrame(rows)
    base_grid = results.loc[
        results.population.eq("fixed_common_support") & results.slice.eq("all") &
        results.exit_estimator_stress_bps.eq(0.0)
    ].copy()
    positive = base_grid.loc[base_grid.uplift_vs_always_continue_bps.gt(0)].sort_values(
        ["latency_minutes", "added_exit_slippage_bps"], ascending=[False, False]
    )
    max_combo = None if positive.empty else {
        "latency_minutes": int(positive.iloc[0].latency_minutes),
        "added_exit_slippage_bps": float(positive.iloc[0].added_exit_slippage_bps),
        "uplift_vs_always_continue_bps": float(positive.iloc[0].uplift_vs_always_continue_bps),
        "selection_rule": "lexicographic maximum latency then slippage among positive fixed-common-support cells at zero estimator stress",
    }
    baseline_slices = results.loc[
        results.population.eq("fixed_common_support") & results.latency_minutes.eq(0) &
        results.added_exit_slippage_bps.eq(0.0) & results.exit_estimator_stress_bps.eq(0.0)
    ]
    grid_lines = ["| Latency | Slippage | Net policy | Uplift vs continue |", "|---:|---:|---:|---:|"]
    for r in base_grid.sort_values(["latency_minutes", "added_exit_slippage_bps"]).itertuples():
        grid_lines.append(f"| {r.latency_minutes}m | {r.added_exit_slippage_bps:.0f} bps | {r.policy_net_bps:.3f} | {r.uplift_vs_always_continue_bps:.3f} |")
    slice_lines = ["| Slice | Value | Rows | Uplift vs continue |", "|---|---:|---:|---:|"]
    for r in baseline_slices.loc[baseline_slices.slice.ne("all")].itertuples():
        slice_lines.append(f"| {r.slice} | {r.slice_value} | {r.rows:,} | {r.uplift_vs_always_continue_bps:.3f} |")
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        results.to_parquet(stage / "stage_e_execution_sensitivity.parquet", index=False, compression="zstd")
        pd.DataFrame(row_audit).to_parquet(stage / "stage_e_execution_sensitivity_row_audit.parquet", index=False, compression="zstd")
        summary = (
            "# Stage E4 — frozen-decision execution sensitivity\n\n"
            f"The canonical v9 decisions, predictions, selected features, calibrators, and zero-bps margin were never refit or changed. "
            f"The fixed latency-comparison population contains {len(common):,} of {len(frame):,} final-OOS rows; "
            f"{len(frame)-len(common):,} immutable paths lack a 10-minute-delayed open and are disclosed rather than silently changing support.\n\n"
            f"Maximum positive latency/slippage combination at zero estimator stress: `{max_combo}`.\n\n"
            "## Zero-estimator-stress grid\n\n" + "\n".join(grid_lines) + "\n\n"
            "## Barrier and fill-ambiguity slices at the canonical execution point\n\n" + "\n".join(slice_lines) + "\n\n"
            "Slippage is added once to the exit cost. Estimator stress is a signed perturbation to exit gross value. "
            "The three ambiguity slices are predeclared as: clear-bar incremental favorable move >=50% of the frozen H0 cost-clear hurdle; "
            "clear-bar adverse extreme within 25 bps of the symmetric adverse barrier; and canonical raw next-fill open differing by >=25 bps from the decision-time close.\n"
        )
        (stage / "stage_e_execution_sensitivity_summary.md").write_text(summary)
        dump(stage / "stage_e_execution_sensitivity_audit.json", audit | {"maximum_positive_combination": max_combo})
        outputs = {p.name: sha256(p) for p in stage.iterdir()}
        manifest = {
            "schema": SCHEMA,
            "status": "RESEARCH_ONLY_FROZEN_V9_DECISIONS_NO_REFIT",
            "inputs": {str(p): sha256(p) for p in (V9_REPLAY, V9_MANIFEST, COUNTERFACTUALS, POSTCOST_EVENTS, *PATHS)},
            "predeclared_grid": {"latency_minutes": LATENCIES, "added_exit_slippage_bps": SLIPPAGES, "exit_estimator_stress_bps": ESTIMATOR_STRESSES},
            "slice_thresholds": {"large_jump_fraction": LARGE_JUMP_FRACTION, "close_adverse_barrier_bps": CLOSE_ADVERSE_BARRIER_BPS, "material_next_fill_bps": MATERIAL_NEXT_FILL_BPS},
            "decision_contract": {"source": "canonical Stage-D v9 final_oos selected-margin rows", "margin_bps": 0.0, "model_refit": False, "predictions_recomputed": False, "actions_recomputed": False},
            "population_audit": audit,
            "maximum_positive_combination": max_combo,
            "outputs_sha256": outputs,
            "runner_sha256": sha256(Path(__file__)),
            "tests_sha256": sha256(ROOT / "tests/test_stage_e_execution_sensitivity.py"),
        }
        dump(stage / "run_manifest.json", manifest)
        (stage / "manifest.sha256").write_text(f"{sha256(stage / 'run_manifest.json')}  run_manifest.json\n")
        os.replace(stage, output)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, default=str))
