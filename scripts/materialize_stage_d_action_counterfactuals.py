#!/usr/bin/env python3
"""Materialise Stage-D clear-event EXIT_NOW versus frozen-CONTINUE labels.

This is a target-only D0 materialisation.  It keeps the frozen candidate,
policy and cost contracts intact and intentionally contains no model features
or action-policy selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_historical_exact_h12_alignment_sidecar import COST_MODEL_ID, EXECUTION_POLICY_ID
from scripts.materialize_historical_exact_h12_postcost_events import TARGET_ID as POSTCOST_TARGET_ID

ART = ROOT / "data_perp/artifacts"
ALIGNMENT = ART / "historical_exact_h12_alignment_sidecar_research_only_20260731_v1/alignment_sidecar.parquet"
EVENTS = ART / "historical_exact_h12_postcost_events_20260731_v1/postcost_events.parquet"
PATHS = (
    ART / "failure_2022_2023_pf_exact1m_paths_20260730_v1/paths.parquet",
    ART / "failure_2024_exact1m_paths_20260730_v2/paths.parquet",
)
OUT = ART / "stage_d_action_counterfactuals_20260731_v2"
SCHEMA = "stage_d_clear_event_action_counterfactuals_v2"
PATH_SOURCE_ID = "historical_exact_h12_720x1m_paths_2022_2024_v1"
ACTION_FILL_CONVENTION_ID = "completed_1m_clear_then_strictly_later_1m_open_adverse_exit_proxy_v2"
HORIZON_MINUTES = 720


class ContractError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def id_digest(values: pd.Series) -> str:
    """Stable identity seal for a candidate population."""
    payload = "\n".join(sorted(values.astype(str).tolist())).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(safe(payload), indent=2, sort_keys=True) + "\n")


def _utc(value: Any, name: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        result = result.tz_localize("UTC")
    else:
        result = result.tz_convert("UTC")
    if pd.isna(result):
        raise ContractError(f"invalid UTC {name}")
    return result


def _decode_path(raw: Any, decision_ts: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(raw) if isinstance(raw, str) else raw
    timestamps = np.asarray(payload["timestamp"], dtype=np.int64)
    open_ = np.asarray(payload["open"], dtype=np.float64)
    if timestamps.shape != (HORIZON_MINUTES,) or open_.shape != (HORIZON_MINUTES,):
        raise ContractError("path must contain exactly 720 one-minute timestamps and opens")
    expected = decision_ts.value + np.arange(HORIZON_MINUTES, dtype=np.int64) * pd.Timedelta(minutes=1).value
    if not np.array_equal(timestamps, expected):
        raise ContractError("path is not exact decision-aligned one-minute data")
    if not np.isfinite(open_).all() or (open_ <= 0.0).any():
        raise ContractError("path opens are invalid")
    return timestamps, open_


def adverse_exit_fill_proxy(
    *, side: float, exit_price: float, quote_half_spread_bps: float,
) -> float:
    """Exact close-trigger branch of the frozen optimiser exit-fill proxy.

    This is intentionally local rather than importing the optimiser module:
    that module imports optional Numba-heavy model configuration at import
    time.  The formula matches ``_adverse_exit_fill_proxy_array(...,
    trigger='close')``: side-adverse half spread plus the same base gap (with
    zero through-candle component for a close trigger).
    """
    if not np.isfinite([side, exit_price, quote_half_spread_bps]).all() or exit_price <= 0.0 or quote_half_spread_bps < 0.0:
        raise ContractError("invalid action exit-fill inputs")
    # Keep the repository function's array dtypes and final float32 cast.
    # The test suite extracts the implementation directly from
    # simple_policy_optimiser.py and checks bitwise numerical parity.
    side_arr = np.asarray([side], dtype=np.float32)
    px = np.asarray([exit_price], dtype=np.float64)
    is_long = side_arr >= 0.0
    quote_half_spread = np.asarray([quote_half_spread_bps], dtype=np.float64)
    quote_half_spread = np.maximum(np.nan_to_num(quote_half_spread, nan=0.0), 0.0) / 10000.0
    px = np.where(is_long, px * (1.0 - quote_half_spread), px * (1.0 + quote_half_spread))
    finite = np.isfinite(px) & (px > 0.0)
    through = np.zeros_like(px, dtype=np.float64)  # close trigger
    gap = np.minimum(px * 15.0 / 10000.0, px * 75.0 / 10000.0)
    filled = np.where(is_long, px - gap, px + gap)
    result = np.where(finite & np.isfinite(filled) & (filled > 0.0), filled, np.nan).astype(np.float32, copy=False)[0]
    if not np.isfinite(result) or result <= 0.0:
        raise ContractError("causal exit fill is invalid")
    return float(result)


def build_counterfactual(
    *,
    candidate: Mapping[str, Any],
    favorable_minute: int,
    raw_path: Any,
    path_source_file: str,
) -> dict[str, Any]:
    """Build one eligible clear-first paired counterfactual.

    One-minute path timestamps are bar *open* timestamps.  Therefore a clear
    using the high/low of event bar i first becomes observable at its close,
    i+1.  The strict action contract requires execution after that decision,
    at the next immutable bar open i+2.  Hence i>=718 is not actionable from
    the immutable 720-open source and is excluded by the caller.
    """
    side_name = str(candidate["side"]).lower()
    if side_name not in {"long", "short"}:
        raise ContractError("unsupported side")
    if not 0 <= int(favorable_minute) < HORIZON_MINUTES - 2:
        raise ContractError("completed first clear does not have an immutable strictly later execution open")
    decision_ts = _utc(candidate["decision_ts"], "decision_ts")
    entry_ts = _utc(candidate["entry_ts"], "entry_ts")
    horizon_end_ts = _utc(candidate["label_end_ts"], "label_end_ts")
    label_available_ts = _utc(candidate["label_available_ts"], "label_available_ts")
    if entry_ts != decision_ts or horizon_end_ts != decision_ts + pd.Timedelta(hours=12) or label_available_ts != horizon_end_ts:
        raise ContractError("frozen H12 timing contract drift")
    _, open_ = _decode_path(raw_path, decision_ts)
    i = int(favorable_minute)
    clear_event_bar_open_ts = decision_ts + pd.Timedelta(minutes=i)
    first_clear_ts = clear_event_bar_open_ts + pd.Timedelta(minutes=1)
    action_decision_ts = first_clear_ts
    action_execution_ts = decision_ts + pd.Timedelta(minutes=i + 2)
    if not action_decision_ts < action_execution_ts <= horizon_end_ts:
        raise ContractError("causal action timing failed")
    side = 1.0 if side_name == "long" else -1.0
    entry = float(candidate["execution_entry_price"])
    cost = float(candidate["row_cost_bps"])
    exit_half_spread = float(candidate["exit_half_spread_bps"])
    continue_gross = float(candidate["exact_h12_gross_bps"])
    continue_net = float(candidate["exact_h12_net_bps"])
    values = np.asarray([entry, cost, exit_half_spread, continue_gross, continue_net], dtype=float)
    if not np.isfinite(values).all() or entry <= 0.0 or cost < 0.0 or exit_half_spread < 0.0:
        raise ContractError("invalid frozen action economics")
    if not np.isclose(continue_gross - cost, continue_net, rtol=0.0, atol=1e-6):
        raise ContractError("frozen continue economics are not cost-once")
    fill = adverse_exit_fill_proxy(
        side=side, exit_price=float(open_[i + 2]),
        quote_half_spread_bps=exit_half_spread,
    )
    exit_gross = side * (fill / entry - 1.0) * 10_000.0
    exit_net = exit_gross - cost
    delta = continue_net - exit_net
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "side": side_name,
        "entry_ts": entry_ts,
        "clear_event_bar_open_ts": clear_event_bar_open_ts,
        "first_clear_ts": first_clear_ts,
        "first_clear_bar_index": i,
        "action_decision_ts": action_decision_ts,
        "action_execution_ts": action_execution_ts,
        "horizon_end_ts": horizon_end_ts,
        "label_available_ts": label_available_ts,
        "execution_policy_id": str(candidate["execution_policy_id"]),
        "cost_model_id": str(candidate["cost_model_id"]),
        "path_source_id": PATH_SOURCE_ID,
        "path_source_file": path_source_file,
        "postcost_target_id": POSTCOST_TARGET_ID,
        "action_fill_convention_id": ACTION_FILL_CONVENTION_ID,
        "entry_executable_price": entry,
        "action_exit_raw_open": float(open_[i + 2]),
        "action_exit_executable_price": fill,
        "exit_half_spread_bps": exit_half_spread,
        "adverse_exit_base_gap_bps": 15.0,
        "known_row_cost_bps": cost,
        "net_exit_now_gross_bps": exit_gross,
        "net_exit_now_cost_bps": cost,
        "net_exit_now_bps": exit_net,
        "net_continue_gross_bps": continue_gross,
        "net_continue_cost_bps": cost,
        "net_continue_bps": continue_net,
        "delta_continue_bps": delta,
        "continue_better": int(delta > 0.0),
    }


def _load_candidates(alignment_path: Path, events_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    alignment = pd.read_parquet(alignment_path, columns=[
        "candidate_id", "side", "decision_ts", "entry_ts", "label_end_ts", "label_available_ts",
        "execution_policy_id", "cost_model_id", "execution_entry_price", "row_cost_bps",
        "exact_h12_gross_bps", "exact_h12_net_bps", "exit_half_spread_bps",
    ])
    events = pd.read_parquet(events_path, columns=[
        "candidate_id", "side", "postcost_target_id", "postcost_h0_event", "postcost_h0_favorable_minute",
        "execution_policy_id", "cost_model_id",
    ])
    if alignment.candidate_id.duplicated().any() or events.candidate_id.duplicated().any():
        raise ContractError("frozen IDs must be unique")
    frame = events.merge(alignment, on="candidate_id", how="inner", validate="one_to_one", suffixes=("_event", ""))
    if len(frame) != len(events):
        raise ContractError("event/alignment identity coverage mismatch")
    for column in ("side", "execution_policy_id", "cost_model_id"):
        if not frame[f"{column}_event"].astype(str).eq(frame[column].astype(str)).all():
            raise ContractError(f"event/alignment {column} contract mismatch")
    if not frame.postcost_target_id.eq(POSTCOST_TARGET_ID).all() or not frame.execution_policy_id.eq(EXECUTION_POLICY_ID).all() or not frame.cost_model_id.eq(COST_MODEL_ID).all():
        raise ContractError("frozen event/policy/cost identifiers differ")
    clear = frame.loc[frame.postcost_h0_event.eq("clear_cost_first")].copy()
    clear["postcost_h0_favorable_minute"] = pd.to_numeric(clear.postcost_h0_favorable_minute, errors="raise").astype(int)
    exclusions = clear.loc[clear.postcost_h0_favorable_minute.ge(HORIZON_MINUTES - 2), ["candidate_id", "side", "postcost_h0_favorable_minute"]].copy()
    exclusions["exclusion_reason"] = np.where(
        exclusions.postcost_h0_favorable_minute.eq(HORIZON_MINUTES - 2),
        "no_immutable_strictly_later_open_after_completed_clear",
        "clear_bar_completion_and_strictly_later_open_not_immutable",
    )
    return clear.loc[clear.postcost_h0_favorable_minute.lt(HORIZON_MINUTES - 2)].copy(), exclusions


def run(*, alignment_path: Path, events_path: Path, path_files: tuple[Path, ...], output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    candidates, exclusions = _load_candidates(alignment_path, events_path)
    wanted = set(candidates.candidate_id.astype(str))
    seen: set[str] = set()
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    writer: pq.ParquetWriter | None = None
    written = 0
    try:
        counterfactual_path = stage / "stage_d_action_counterfactuals.parquet"
        for path in path_files:
            source_id = path.name
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=256, columns=["candidate_id", "execution_future_path"]):
                paths = batch.to_pandas()
                paths = paths.loc[paths.candidate_id.astype(str).isin(wanted)]
                if paths.empty:
                    continue
                joined = paths.merge(candidates, on="candidate_id", how="inner", validate="one_to_one")
                rows = [build_counterfactual(candidate=row._asdict(), favorable_minute=int(row.postcost_h0_favorable_minute), raw_path=row.execution_future_path, path_source_file=source_id) for row in joined.itertuples(index=False)]
                ids = {row["candidate_id"] for row in rows}
                if len(ids) != len(rows) or seen.intersection(ids):
                    raise ContractError("duplicate exact action path")
                seen.update(ids)
                table = pa.Table.from_pylist(rows)
                if writer is None:
                    writer = pq.ParquetWriter(counterfactual_path, table.schema, compression="zstd")
                writer.write_table(table if writer.schema == table.schema else table.cast(writer.schema))
                written += len(rows)
                if written % 5_000 < len(rows):
                    print(f"[stage-d-d0] materialized {written:,} action rows", flush=True)
            print(f"[stage-d-d0] scanned {path.name}; materialized {written:,} action rows", flush=True)
        if writer is None:
            raise ContractError("no eligible exact paths")
        writer.close(); writer = None
        if seen != wanted:
            raise ContractError(f"exact action path coverage incomplete: missing {len(wanted - seen)}")
        rows = pd.read_parquet(counterfactual_path).sort_values(["action_decision_ts", "candidate_id"], kind="stable").reset_index(drop=True)
        if rows.candidate_id.duplicated().any() or not rows.action_decision_ts.lt(rows.action_execution_ts).all():
            raise ContractError("action identity/timing invariant failed")
        for arm in ("net_exit_now", "net_continue"):
            if not np.allclose(rows[f"{arm}_gross_bps"] - rows[f"{arm}_cost_bps"], rows[f"{arm}_bps"], atol=1e-6):
                raise ContractError(f"{arm} cost-once invariant failed")
        if not np.allclose(rows.net_continue_bps - rows.net_exit_now_bps, rows.delta_continue_bps, atol=1e-6):
            raise ContractError("delta invariant failed")
        rows.to_parquet(counterfactual_path, index=False, compression="zstd")
        ledger = rows.loc[:, ["candidate_id", "side", "entry_ts", "clear_event_bar_open_ts", "first_clear_ts", "action_decision_ts", "action_execution_ts", "horizon_end_ts", "label_available_ts", "execution_policy_id", "cost_model_id", "path_source_id"]]
        ledger.to_parquet(stage / "stage_d_action_identity_ledger.parquet", index=False, compression="zstd")
        exclusions.to_parquet(stage / "stage_d_action_exclusions.parquet", index=False, compression="zstd")
        samples: list[pd.Series] = []
        for side in ("long", "short"):
            side_rows = rows.loc[rows.side.eq(side)].sort_values(["first_clear_bar_index", "candidate_id"], kind="stable")
            if not side_rows.empty:
                samples.extend([side_rows.iloc[0], side_rows.iloc[-1]])
        audit = ["# Stage-D action counterfactual audit", "", f"- Exact H0 clear-first rows in frozen event pack: **{len(rows) + len(exclusions):,}**.", f"- Eligible actionable rows: **{len(rows):,}**.", f"- Explicitly excluded tail clear rows (`i >= 718`): **{len(exclusions):,}**.", "- Path timestamps are one-minute bar **opens**. A clear using bar `i` high/low is observable only at its close: `first_clear_ts = action_decision_ts = decision_ts + (i + 1)m`.", "- Strictly subsequent action execution uses immutable path open `i + 2`; this deliberately makes `action_decision_ts < action_execution_ts`.", "- CONTINUE is the unchanged frozen-policy exact H12 outcome. Both arms are entry-to-outcome net returns and deduct the frozen row cost once.", "- No model feature, model, threshold, entry rule, sizing, or portfolio policy is materialized.", "", "## Representative raw-path arithmetic", ""]
        for sample in samples:
            side_sign = -1.0 if sample.side == "long" else 1.0
            quoted = sample.action_exit_raw_open * (1.0 + side_sign * sample.exit_half_spread_bps / 10_000.0)
            gap = min(quoted * sample.adverse_exit_base_gap_bps / 10_000.0, quoted * 75.0 / 10_000.0)
            audit.extend([
                f"- `{sample.side}` candidate `{sample.candidate_id}`, clear bar open index `{int(sample.first_clear_bar_index)}`: event bar open `{sample.clear_event_bar_open_ts}`, decision/clear completion `{sample.action_decision_ts}`, executable open `{sample.action_execution_ts}`.",
                f"  raw open `{sample.action_exit_raw_open:.10g}`; quoted `{quoted:.10g}` with `{sample.exit_half_spread_bps:.6g}` bps half spread; gap `{gap:.10g}`; executable fill `{sample.action_exit_executable_price:.10g}`; stored EXIT_NOW `{sample.net_exit_now_bps:.6f}` bps; frozen CONTINUE `{sample.net_continue_bps:.6f}` bps; delta `{sample.delta_continue_bps:.6f}` bps.",
            ])
        audit.append("")
        (stage / "stage_d_action_counterfactual_audit.md").write_text("\n".join(audit))
        population = {"schema": "stage_d_action_population_manifest_v2", "source_event": POSTCOST_TARGET_ID, "population": "exact H0 clear_cost_first", "clear_first_rows": int(len(rows) + len(exclusions)), "eligible_action_rows": int(len(rows)), "excluded_tail_clear_rows": int(len(exclusions)), "eligible_candidate_id_sha256": id_digest(rows.candidate_id), "excluded_candidate_id_sha256": id_digest(exclusions.candidate_id), "action_timing": "path bar i open -> completed clear/action decision i+1 -> strictly later executable open i+2", "execution_policy_id": EXECUTION_POLICY_ID, "cost_model_id": COST_MODEL_ID, "path_source_id": PATH_SOURCE_ID, "action_fill_convention_id": ACTION_FILL_CONVENTION_ID}
        write_json(stage / "stage_d_action_population_manifest.json", population)
        correctness = {
            "schema": "stage_d_action_correctness_report_v2",
            "rows": int(len(rows)),
            "exclusions": int(len(exclusions)),
            "identity": {"eligible_candidate_id_sha256": id_digest(rows.candidate_id), "excluded_candidate_id_sha256": id_digest(exclusions.candidate_id)},
            "invariants": {
                "candidate_id_unique": not bool(rows.candidate_id.duplicated().any()),
                "clear_bar_open_before_observable_clear": bool(rows.clear_event_bar_open_ts.lt(rows.first_clear_ts).all()),
                "strict_action_decision_before_execution": bool(rows.action_decision_ts.lt(rows.action_execution_ts).all()),
                "execution_at_strictly_later_path_open": bool((rows.action_execution_ts - rows.clear_event_bar_open_ts).eq(pd.Timedelta(minutes=2)).all()),
                "cost_once_each_arm": True,
                "paired_delta_exact": True,
                "continue_is_frozen_h12": bool(np.allclose(rows.net_continue_bps, rows.net_continue_gross_bps - rows.net_continue_cost_bps, atol=1e-6)),
                "tail_exclusions_only_i_ge_718": bool(exclusions.postcost_h0_favorable_minute.ge(HORIZON_MINUTES - 2).all()),
            },
        }
        write_json(stage / "correctness_test_report.json", correctness)
        outputs = {name: sha256(stage / name) for name in ("stage_d_action_counterfactuals.parquet", "stage_d_action_identity_ledger.parquet", "stage_d_action_exclusions.parquet", "stage_d_action_counterfactual_audit.md", "stage_d_action_population_manifest.json", "correctness_test_report.json")}
        manifest = {"schema": SCHEMA, "status": "MATERIALIZED_D0_TARGETS_ONLY_NO_MODEL_OR_POLICY_CHANGE", "rows": int(len(rows)), "contract": population, "inputs": {str(path): sha256(path) for path in (alignment_path, events_path, *path_files)}, "outputs_sha256": outputs, "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))}}
        write_json(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(f"{sha256(stage / 'manifest.json')}  manifest.json\n")
        os.replace(stage, output)
        return manifest
    except Exception:
        if writer is not None: writer.close()
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment", type=Path, default=ALIGNMENT)
    parser.add_argument("--events", type=Path, default=EVENTS)
    parser.add_argument("--paths", type=Path, nargs="+", default=list(PATHS))
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    print(json.dumps(safe(run(alignment_path=args.alignment, events_path=args.events, path_files=tuple(args.paths), output=args.output)), indent=2))


if __name__ == "__main__":
    main()
