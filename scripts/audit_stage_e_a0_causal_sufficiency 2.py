#!/usr/bin/env python3
"""Stage-E1 causal sufficiency and target-proximity audit for frozen Stage-D A0.

The independent reconstruction deliberately accepts only entry-known fields and
the completed one-minute prefix through ``action_decision_ts``.  Realised exit
fills are loaded only by the separate diagnostic path after reconstruction.
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
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ART = ROOT / "data_perp/artifacts"
COUNTER_ROOT = ART / "stage_d_action_counterfactuals_20260731_v2"
FEATURE_ROOT = ART / "stage_d_action_features_20260731_v5"
MODEL_ROOT = ART / "stage_d_compact_action_model_20260731_v9"
MODEL_REPRO_ROOT = ART / "stage_d_compact_action_model_20260731_v10"
COUNTER = COUNTER_ROOT / "stage_d_action_counterfactuals.parquet"
SEALED_FEATURES = FEATURE_ROOT / "stage_d_action_features.parquet"
GROUPS = FEATURE_ROOT / "stage_d_action_feature_groups.json"
DICTIONARY = FEATURE_ROOT / "stage_d_action_feature_dictionary.json"
FEATURE_MANIFEST = FEATURE_ROOT / "run_manifest.json"
COMPACT_FEATURE_MANIFEST = MODEL_ROOT / "stage_d_compact_feature_manifest.json"
RAW_PANEL = ART / "long_exact_h12_raw_base_panel_20260730_v2/raw_base_panel.parquet"
PATHS = (
    ART / "failure_2022_2023_pf_exact1m_paths_20260730_v1/paths.parquet",
    ART / "failure_2024_exact1m_paths_20260730_v2/paths.parquet",
)
DEFAULT_OUTPUT = ART / "stage_e_a0_causal_sufficiency_20260731_v3"
SCHEMA = "stage_e_a0_causal_sufficiency_v3"
LATENCIES = (0, 1, 2, 5, 10)
PREFIX_AUDIT_ROWS = 1000
TARGET_OR_FUTURE_COLUMNS = {
    "action_exit_raw_open", "action_exit_executable_price", "net_exit_now_gross_bps",
    "net_exit_now_cost_bps", "net_exit_now_bps", "net_continue_gross_bps",
    "net_continue_cost_bps", "net_continue_bps", "delta_continue_bps", "continue_better",
    "horizon_end_ts", "label_available_ts",
}
RECON_COUNTER_COLUMNS = [
    "candidate_id", "side", "entry_ts", "first_clear_bar_index", "action_decision_ts",
    "entry_executable_price",
]
CAUSALLY_UNRECONSTRUCTABLE = {
    "known_row_cost_bps": "source is realised execution_cost_return resolved with the future exit path; prior canonical target-purity contract explicitly forbids it as an input",
    "estimated_net_if_exit_now_bps": "depends on causally unavailable realised row_cost_bps in the sealed Stage-D formula",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def id_hash(values: Iterable[Any]) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def require_canonical_inputs() -> dict[str, str]:
    required = [
        COUNTER, SEALED_FEATURES, GROUPS, DICTIONARY, FEATURE_MANIFEST,
        COMPACT_FEATURE_MANIFEST, RAW_PANEL, *PATHS,
        COUNTER_ROOT / "manifest.json", MODEL_ROOT / "run_manifest.json",
        MODEL_REPRO_ROOT / "run_manifest.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing canonical Stage-D inputs: {missing}")
    v9 = json.loads((MODEL_ROOT / "run_manifest.json").read_text())
    v10 = json.loads((MODEL_REPRO_ROOT / "run_manifest.json").read_text())
    for name in (
        "stage_d_action_policy_replay.parquet", "stage_d_compact_feature_manifest.json",
        "stage_d_compact_model_results.parquet", "stage_d_leave_group_out_results.parquet",
    ):
        if v9.get("outputs_sha256", {}).get(name) != v10.get("outputs_sha256", {}).get(name):
            raise ValueError(f"Stage-D v9/v10 reproducibility mismatch: {name}")
    return {str(p.relative_to(ROOT)): sha256(p) for p in required}


def _category(name: str) -> str:
    if name in {"time_to_clear_minutes", "gross_return_at_action_bps", "estimated_net_if_exit_now_bps"}:
        return "action_state"
    if name in {"known_row_cost_bps", "estimated_spread_bps", "entry_half_spread_bps", "exit_half_spread_bps"}:
        return "cost"
    if name in {"barrier_pct"}:
        return "policy_geometry"
    if name == "side_long":
        return "symbol_or_side_identity"
    if name in {"entry_price_log"}:
        return "entry_static"
    # The remainder are frozen entry-time controls, not action-path state.
    return "entry_static"


def _relationship(name: str) -> str:
    if name == "estimated_net_if_exit_now_bps":
        return "target-adjacent: estimated decision-time EXIT_NOW net; target subtracts realised later EXIT_NOW net"
    if name == "gross_return_at_action_bps":
        return "target-adjacent: decision-time mark-to-market component of estimated exit value"
    if name == "known_row_cost_bps":
        return "CAUSAL DEFECT: renamed realised execution_cost_return; future exit-dependent, though it cancels algebraically between target arms"
    if name in {"estimated_spread_bps", "entry_half_spread_bps", "exit_half_spread_bps"}:
        return "known execution-cost proxy; correlated with realised exit fill but not a future fill"
    if name == "time_to_clear_minutes":
        return "action-event timing state; no realised suffix"
    return "causal covariate; no direct realised target arithmetic"


def build_inventory() -> pd.DataFrame:
    groups = json.loads(GROUPS.read_text())
    fields = groups["A0_minimal_action_state_control"]
    if len(fields) != 61 or len(set(fields)) != 61:
        raise ValueError("canonical A0 contract is not exactly 61 unique fields")
    dictionary = json.loads(DICTIONARY.read_text())
    compact = json.loads(COMPACT_FEATURE_MANIFEST.read_text())
    frozen = {x["side"]: set(x["selected_features"]) for x in compact["frozen_before_final_oos"] if x["arm"] == "compact_readmitted"}
    states = [x for x in compact["training_only_preprocessing"] if x["arm"] == "leave_out__A1_path_geometry_to_clear"]
    rows = []
    for name in fields:
        meta = dictionary[name]
        fold_names = {side: sorted(x["fold"] for x in states if x["side"] == side and name in x["preprocessing"]["selected"]) for side in ("long", "short")}
        selected_sides = [s for s in ("long", "short") if name in frozen.get(s, set())]
        rows.append({
            "feature_name": name,
            "selected_long": name in frozen.get("long", set()),
            "selected_short": name in frozen.get("short", set()),
            "selected_fold_count": sum(len(v) for v in fold_names.values()),
            "selected_long_folds": json.dumps(fold_names["long"]),
            "selected_short_folds": json.dumps(fold_names["short"]),
            "category": _category(name),
            "availability_ts": meta["feature_available_ts"],
            "raw_source": meta["source"],
            "formula": meta["formula"],
            "live_computation": ("UNAVAILABLE: " + CAUSALLY_UNRECONSTRUCTABLE[name]) if name in CAUSALLY_UNRECONSTRUCTABLE else ("prefix recomputation" if name in {"time_to_clear_minutes", "gross_return_at_action_bps"} else "frozen entry row / frozen geometry"),
            "target_relationship": _relationship(name),
            "contains_current_gross_return": name == "gross_return_at_action_bps",
            "contains_current_exit_value": name == "estimated_net_if_exit_now_bps",
            "contains_time_to_clear": name == "time_to_clear_minutes",
            "contains_cost": name in {"known_row_cost_bps", "estimated_spread_bps", "entry_half_spread_bps", "exit_half_spread_bps", "estimated_net_if_exit_now_bps"},
            "contains_barrier_geometry": name == "barrier_pct",
            "contains_entry_price": name == "entry_price_log",
            "contains_policy_state": name in {"barrier_pct", "known_row_cost_bps"},
            "point_in_time_safe_declared": bool(meta["point_in_time_safe"]),
            "live_reproducible_declared": bool(meta["live_reproducible"]),
            "stage_e_verified_point_in_time_safe": name not in CAUSALLY_UNRECONSTRUCTABLE,
            "stage_e_causal_defect": CAUSALLY_UNRECONSTRUCTABLE.get(name, ""),
        })
    return pd.DataFrame(rows)


def reconstruct_prefix_a0(
    *, side: str, stop_index: int, entry_price: float,
    prefix_timestamp: np.ndarray, prefix_close: np.ndarray,
) -> dict[str, float | int]:
    """Recompute causal action-state fields without any future/target input."""
    n = int(stop_index) + 1
    if side not in {"long", "short"} or n <= 0 or len(prefix_close) != n or len(prefix_timestamp) != n:
        raise ValueError("invalid completed prefix")
    if not np.isfinite(entry_price) or entry_price <= 0:
        raise ValueError("invalid entry-known geometry")
    sign = 1.0 if side == "long" else -1.0
    gross = sign * (float(prefix_close[-1]) / float(entry_price) - 1.0) * 10_000.0
    return {
        "side_long": float(side == "long"),
        "time_to_clear_minutes": float(n),
        "gross_return_at_action_bps": float(gross),
        "entry_price_log": float(np.log(entry_price)),
        "path_observed_through_bar_open_ns": int(prefix_timestamp[-1]),
    }


def _decode_prefix(raw: str | dict[str, Any], stop: int) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(raw) if isinstance(raw, str) else raw
    end = int(stop) + 1
    # Slice before conversion so the independent reconstruction never exposes
    # the future suffix to its computation function.
    ts = np.asarray(payload["timestamp"][:end], dtype=np.int64)
    close = np.asarray(payload["close"][:end], dtype=np.float64)
    if len(ts) != end or len(close) != end or not np.isfinite(close).all():
        raise ValueError("malformed completed prefix")
    return ts, close


def _exit_fill(side: str, raw_open: float, half_spread_bps: float) -> float:
    # Exact frozen Stage-D action fill arithmetic, including final float32 cast.
    sign = np.asarray([1.0 if side == "long" else -1.0], dtype=np.float32)
    px = np.asarray([raw_open], dtype=np.float64)
    is_long = sign >= 0.0
    spread = np.maximum(np.nan_to_num(np.asarray([half_spread_bps], dtype=np.float64), nan=0.0), 0.0) / 10_000.0
    px = np.where(is_long, px * (1.0 - spread), px * (1.0 + spread))
    gap = np.minimum(px * 15.0 / 10_000.0, px * 75.0 / 10_000.0)
    return float(np.where(is_long, px - gap, px + gap).astype(np.float32)[0])


def _stream_path_reconstruction(base: pd.DataFrame, wanted: set[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    lookup = base.set_index("candidate_id", verify_integrity=True)
    recon_cols: dict[str, list[Any]] = {name: [] for name in (
        "candidate_id", "side_long", "time_to_clear_minutes", "gross_return_at_action_bps",
        "entry_price_log", "path_observed_through_bar_open_ns",
    )}
    latency_cols: dict[str, list[Any]] = {name: [] for name in (
        "candidate_id", "latency_minutes_beyond_canonical", "realised_net_exit_bps",
    )}
    seen: set[str] = set()
    max_batch = 0
    for source in PATHS:
        pf = pq.ParquetFile(source)
        # A 720-minute JSON payload is large; keep batches deliberately small
        # so decoded strings plus numerical prefixes remain memory bounded.
        for batch in pf.iter_batches(columns=["candidate_id", "execution_future_path"], batch_size=64):
            q = batch.to_pandas()
            q["candidate_id"] = q.candidate_id.astype(str)
            q = q[q.candidate_id.isin(wanted - seen)]
            if q.empty:
                continue
            max_batch = max(max_batch, len(q))
            for candidate_id, raw in q[["candidate_id", "execution_future_path"]].itertuples(index=False, name=None):
                row = lookup.loc[candidate_id]
                stop = int(row.first_clear_bar_index)
                payload = json.loads(raw) if isinstance(raw, str) else raw
                ts, close = _decode_prefix(payload, stop)
                values = reconstruct_prefix_a0(
                    side=str(row.side), stop_index=stop, entry_price=float(row.entry_executable_price),
                    prefix_timestamp=ts, prefix_close=close,
                )
                values["candidate_id"] = candidate_id
                for name in recon_cols:
                    recon_cols[name].append(values[name])
                opens = payload["open"]
                for latency in LATENCIES:
                    idx = stop + 2 + latency
                    if idx >= len(opens):
                        continue
                    # The separate outcome diagnostic is populated later after
                    # joining future-resolved cost/fill fields.  Retain raw
                    # opens here without exposing them to reconstruction.
                    latency_cols["candidate_id"].append(candidate_id)
                    latency_cols["latency_minutes_beyond_canonical"].append(latency)
                    latency_cols["realised_net_exit_bps"].append(float(opens[idx]))
                seen.add(candidate_id)
                if len(seen) % 10000 == 0:
                    print(f"[stage-e1] reconstructed {len(seen):,}/{len(wanted):,}", flush=True)
    if seen != wanted:
        raise ValueError(f"path coverage incomplete: {len(wanted - seen)} missing")
    return pd.DataFrame(recon_cols), pd.DataFrame(latency_cols), {"max_path_batch_rows": max_batch, "path_rows": len(seen), "bounded_batch_size": 64}


def _compare_features(recomputed: pd.DataFrame, sealed: pd.DataFrame, fields: list[str], selected: set[str]) -> dict[str, Any]:
    available = [name for name in fields if name in recomputed.columns]
    joined = sealed[["candidate_id", *fields]].merge(recomputed[["candidate_id", *available]], on="candidate_id", suffixes=("_sealed", "_recomputed"), validate="one_to_one")
    rows = []
    for name in fields:
        if name in CAUSALLY_UNRECONSTRUCTABLE:
            rows.append({"feature_name": name, "selected": name in selected, "exact_equal": False, "tolerance": None, "max_abs_error": None, "rows_compared": 0, "passed": False, "failure_reason": CAUSALLY_UNRECONSTRUCTABLE[name]})
            continue
        a = pd.to_numeric(joined[f"{name}_sealed"], errors="coerce").to_numpy(float)
        b = pd.to_numeric(joined[f"{name}_recomputed"], errors="coerce").to_numpy(float)
        finite = np.isfinite(a) & np.isfinite(b)
        same_missing = np.array_equal(np.isnan(a), np.isnan(b))
        abs_err = np.abs(a[finite] - b[finite])
        max_abs = float(abs_err.max()) if len(abs_err) else 0.0
        exact = bool(same_missing and np.array_equal(a[finite], b[finite]))
        tolerance = 1e-5
        passed = bool(same_missing and (not len(abs_err) or max_abs <= tolerance))
        rows.append({"feature_name": name, "selected": name in selected, "exact_equal": exact, "tolerance": tolerance, "max_abs_error": max_abs, "rows_compared": int(finite.sum()), "passed": passed, "failure_reason": "" if passed else "sealed/recomputed mismatch"})
    failures = [r["feature_name"] for r in rows if r["selected"] and not r["passed"]]
    return {"rows": len(joined), "feature_results": rows, "selected_failures": failures, "passed": not failures}


def _exit_value_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dimensions: list[tuple[str, str, pd.DataFrame]] = [("overall", "ALL", frame)]
    dimensions += [("side", str(k), q) for k, q in frame.groupby("side")]
    dimensions += [("month", str(k), q) for k, q in frame.assign(month=frame.action_decision_ts.dt.strftime("%Y-%m")).groupby("month")]
    for latency, z in frame.groupby("latency_minutes_beyond_canonical"):
        parts = [("overall", "ALL", z), *[("side", str(k), q) for k, q in z.groupby("side")], *[("month", str(k), q) for k, q in z.assign(month=z.action_decision_ts.dt.strftime("%Y-%m")).groupby("month")]]
        for dimension, value, q in parts:
            err = q.estimated_net_if_exit_now_bps - q.realised_net_exit_bps
            quantiles = err.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
            rows.append({
                "latency_minutes_beyond_canonical": int(latency), "dimension": dimension, "value": value,
                "rows": len(q), "bias_estimate_minus_realised_bps": float(err.mean()), "mae_bps": float(err.abs().mean()),
                "error_q01_bps": float(quantiles.loc[0.01]), "error_q05_bps": float(quantiles.loc[0.05]),
                "error_q25_bps": float(quantiles.loc[0.25]), "error_q50_bps": float(quantiles.loc[0.5]),
                "error_q75_bps": float(quantiles.loc[0.75]), "error_q95_bps": float(quantiles.loc[0.95]),
                "error_q99_bps": float(quantiles.loc[0.99]),
                "pearson_correlation": float(q.estimated_net_if_exit_now_bps.corr(q.realised_net_exit_bps)),
                "exact_equal_fraction": float(np.isclose(q.estimated_net_if_exit_now_bps, q.realised_net_exit_bps, rtol=0.0, atol=1e-12).mean()),
            })
    return pd.DataFrame(rows)


def run(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    inputs = require_canonical_inputs()
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        inventory = build_inventory()
        a0 = inventory.feature_name.tolist()
        compact = json.loads(COMPACT_FEATURE_MANIFEST.read_text())
        selected = set().union(*(set(x["selected_features"]) for x in compact["frozen_before_final_oos"] if x["arm"] == "compact_readmitted"))

        # This explicit allow-list is part of the causal boundary.  No realised
        # arm, continuation, delta, label, or future timestamp is read here.
        if TARGET_OR_FUTURE_COLUMNS.intersection(RECON_COUNTER_COLUMNS):
            raise AssertionError("target/future column entered reconstruction allow-list")
        base = pd.read_parquet(COUNTER, columns=RECON_COUNTER_COLUMNS)
        for c in ("entry_ts", "action_decision_ts"):
            base[c] = pd.to_datetime(base[c], utc=True)
        base["candidate_id"] = base.candidate_id.astype(str)
        if base.candidate_id.duplicated().any():
            raise ValueError("counterfactual identity is not unique")

        # The selected outcome-derived cost already forces a fail-closed E1
        # result.  Independently exercise prefix reconstruction on a fixed,
        # deterministic bounded sample; do not spend unbounded compute claiming
        # completeness after the causal contract has already failed.
        prefix_base = base.sort_values(["action_decision_ts", "candidate_id"], kind="stable").head(PREFIX_AUDIT_ROWS).copy()
        action_recon, latency, bounded = _stream_path_reconstruction(prefix_base, set(prefix_base.candidate_id))
        raw_available = set(pq.read_schema(RAW_PANEL).names)
        entry_fields = [x for x in a0 if x not in action_recon.columns and x not in {"known_row_cost_bps", "exit_half_spread_bps"} and x in raw_available]
        panel = pd.read_parquet(RAW_PANEL, columns=["candidate_id", *entry_fields]).drop_duplicates("candidate_id")
        panel["candidate_id"] = panel.candidate_id.astype(str)
        recomputed = prefix_base[["candidate_id"]].merge(action_recon, on="candidate_id", validate="one_to_one").merge(panel, on="candidate_id", how="left", validate="one_to_one")
        # estimated_spread and entry_half_spread/barrier are frozen geometry and
        # are not all in raw panel.  Read them from the sealed alignment source
        # used by Stage-D, never from the outcome artifact.
        missing = [x for x in a0 if x not in recomputed.columns and x not in CAUSALLY_UNRECONSTRUCTABLE]
        if missing:
            alignment = ART / "historical_exact_h12_alignment_sidecar_research_only_20260731_v1/alignment_sidecar.parquet"
            available = set(pq.read_schema(alignment).names)
            cols = [x for x in missing if x in available]
            if set(cols) != set(missing):
                raise ValueError(f"A0 fields unavailable from raw/prefix/geometry sources: {sorted(set(missing)-set(cols))}")
            geo = pd.read_parquet(alignment, columns=["candidate_id", *cols]).drop_duplicates("candidate_id")
            geo["candidate_id"] = geo.candidate_id.astype(str)
            recomputed = recomputed.merge(geo, on="candidate_id", how="left", validate="one_to_one")
        recomputed = recomputed[["candidate_id", *[x for x in a0 if x in recomputed.columns]]]

        sealed = pd.read_parquet(SEALED_FEATURES, columns=["candidate_id", "side", "action_decision_ts", *a0])
        sealed["candidate_id"] = sealed.candidate_id.astype(str)
        sealed_population_hash = id_hash(sealed.candidate_id)
        sealed = sealed[sealed.candidate_id.isin(set(prefix_base.candidate_id))].copy()
        comparison = _compare_features(recomputed, sealed, a0, selected)
        comparison.update({
            "schema": "stage_e_independent_feature_recomputation_v1",
            "status": "FAIL_CLOSED_SELECTED_FIELD_CAUSAL_DEFECT",
            "population_rows": len(base),
            "prefix_audit_rows": len(prefix_base),
            "scope_note": "bounded deterministic prefix sample; full recomputation stopped after selected outcome-derived cost proved causal failure",
            "reconstruction_counter_columns": RECON_COUNTER_COLUMNS,
            "target_or_future_columns_prohibited": sorted(TARGET_OR_FUTURE_COLUMNS),
            "population_candidate_id_sha256": sealed_population_hash,
            "prefix_candidate_id_sha256": id_hash(sealed.candidate_id),
            "bounded_compute": bounded,
            "independent_inputs": [str(RAW_PANEL.relative_to(ROOT)), *(str(p.relative_to(ROOT)) for p in PATHS)],
        })

        # Future-resolved values enter only this separate diagnostic, after the
        # independent matrix has already been materialised and compared.
        audit_outcomes = pd.read_parquet(COUNTER, columns=[
            "candidate_id", "side", "entry_executable_price", "exit_half_spread_bps",
            "known_row_cost_bps", "net_exit_now_bps", "net_continue_bps", "delta_continue_bps",
        ])
        audit_outcomes["candidate_id"] = audit_outcomes.candidate_id.astype(str)
        latency = latency.rename(columns={"realised_net_exit_bps": "future_raw_open"}).merge(audit_outcomes, on="candidate_id", validate="many_to_one")
        signs = np.where(latency.side.eq("long"), 1.0, -1.0)
        fills = np.asarray([_exit_fill(side, raw, spread) for side, raw, spread in latency[["side", "future_raw_open", "exit_half_spread_bps"]].itertuples(index=False, name=None)])
        latency["realised_net_exit_bps"] = signs * (fills / latency.entry_executable_price.to_numpy(float) - 1.0) * 10_000.0 - latency.known_row_cost_bps.to_numpy(float)
        latency = latency.merge(sealed[["candidate_id", "action_decision_ts", "estimated_net_if_exit_now_bps"]], on="candidate_id", validate="many_to_one")
        exit_metrics = _exit_value_metrics(latency)
        canonical = exit_metrics.query("latency_minutes_beyond_canonical == 0 and dimension == 'overall'").iloc[0]
        alignment_path = ART / "historical_exact_h12_alignment_sidecar_research_only_20260731_v1/alignment_sidecar.parquet"
        outcome_lineage = pd.read_parquet(alignment_path, columns=["candidate_id", "row_cost_bps", "exit_reason", "exit_hour", "estimated_spread_bps"])
        outcome_lineage["candidate_id"] = outcome_lineage.candidate_id.astype(str)
        cost_audit = audit_outcomes.merge(outcome_lineage, on="candidate_id", validate="one_to_one")
        by_exit = cost_audit.groupby("exit_reason").row_cost_bps.agg(["count", "mean", "std", "min", "max"]).reset_index().to_dict("records")
        cost_proximity = {
            "source_field": "alignment_sidecar.row_cost_bps <- label_execution_cost_return",
            "prior_contract_evidence": "scripts/run_exact_h12_target_purity_ablation.py states realised row cost and exit-time spread are outcome-bound and forbidden as inputs",
            "spearman_with_delta_continue": float(cost_audit.row_cost_bps.corr(cost_audit.delta_continue_bps, method="spearman")),
            "pearson_with_future_exit_hour": float(cost_audit.row_cost_bps.corr(cost_audit.exit_hour)),
            "cost_by_future_exit_reason": by_exit,
            "target_cost_cancels_exactly": bool(np.allclose(cost_audit.net_continue_bps - cost_audit.net_exit_now_bps, cost_audit.delta_continue_bps, rtol=0.0, atol=1e-9)),
            "interpretation": "cost is not algebraically embedded in delta, but is a future-outcome proxy unavailable at action decision",
        }
        comparison["causal_defects"] = CAUSALLY_UNRECONSTRUCTABLE
        comparison["realised_cost_target_proximity"] = cost_proximity
        causal_checks = {
            "decision_time_estimate_reconstructed_from_completed_prefix": False,
            "reconstruction_does_not_read_target_or_future_columns": not TARGET_OR_FUTURE_COLUMNS.intersection(RECON_COUNTER_COLUMNS),
            "completed_prefix_ends_at_action_decision": bool((pd.to_datetime(action_recon.path_observed_through_bar_open_ns, utc=True) + pd.Timedelta(minutes=1)).reset_index(drop=True).eq(action_recon[["candidate_id"]].merge(prefix_base[["candidate_id", "action_decision_ts"]], on="candidate_id", validate="one_to_one").action_decision_ts.reset_index(drop=True)).all()),
            "estimate_not_identical_to_realised_next_fill": float(canonical.exact_equal_fraction) < 1.0,
            "all_selected_fields_reconstructable": comparison["passed"],
            "realised_row_cost_is_decision_time_known": False,
        }
        decision = "PASS" if all(causal_checks.values()) else "STAGE_D_PASS_REVOKED_TARGET_PROXY_OR_CAUSAL_DEFECT"

        inventory.to_parquet(stage / "stage_e_a0_feature_inventory.parquet", index=False, compression="zstd")
        exit_metrics.to_parquet(stage / "stage_e_a0_exit_value_diagnostics.parquet", index=False, compression="zstd")
        dump_json(stage / "stage_e_independent_feature_recomputation.json", comparison)
        audit = [
            "# Stage-E1 A0 causal sufficiency and target-proximity audit", "",
            f"Decision: **{decision}**.", "",
            "## What the estimate is", "",
            "Stage D defined `estimated_net_if_exit_now_bps` as the completed clear-bar close mark-to-market less a field renamed `known_row_cost_bps`. The clear-bar close is available at `action_decision_ts`; the cost field is not. The realised `EXIT_NOW` counterfactual separately uses the strictly later executable open plus the frozen adverse-fill convention.", "",
            "## Independent reconstruction", "",
            f"- Population: **{len(base):,}** rows; bounded prefix audit: **{comparison['rows']:,}** rows; A0 fields: **{len(a0)}**; selected-field failures: **{comparison['selected_failures']}**.",
            f"- Exact deterministic matches: **{sum(x['exact_equal'] for x in comparison['feature_results'])}/{len(a0)}**; all selected fields reconstruct causally: **{comparison['passed']}**.",
            f"- Maximum path batch: **{bounded['max_path_batch_rows']:,}** rows. The reconstruction allow-list contains no realised arm, continuation, delta, label, or future timestamp column.", "",
            "## Exit-value proximity", "",
            f"At canonical latency, estimate-minus-realised bias is **{canonical.bias_estimate_minus_realised_bps:.4f} bps**, MAE **{canonical.mae_bps:.4f} bps**, correlation **{canonical.pearson_correlation:.6f}**, and exact-equality fraction **{canonical.exact_equal_fraction:.6f}**.",
            "Side, month, error quantile, and +0/+1/+2/+5/+10-minute latency results are in `stage_e_a0_exit_value_diagnostics.parquet`.", "",
            "## Causal defect: realised row cost", "",
            f"`known_row_cost_bps` has Spearman **{cost_proximity['spearman_with_delta_continue']:.6f}** with the action target and Pearson **{cost_proximity['pearson_with_future_exit_hour']:.6f}** with future exit hour. Mean cost by future exit reason: `{json.dumps(by_exit, sort_keys=True)}`.", "",
            "## Causal interpretation", "",
            "The gross clear-bar mark and time-to-clear reconstruct causally. The sealed net estimate does not: it subtracts `known_row_cost_bps`, which is actually the future-resolved `execution_cost_return`. The repository's earlier canonical target-purity report explicitly classifies realised row cost as outcome-bound and forbidden as an input. The value also varies materially by future exit reason, so renaming it did not make it observable. Because this unavailable field was selected on both sides and drives the E2 result, the Stage-D pass is revoked.", "",
            f"Checks: `{json.dumps(causal_checks, sort_keys=True)}`", "",
        ]
        (stage / "stage_e_a0_target_proximity_audit.md").write_text("\n".join(audit))

        outputs = {p.name: sha256(p) for p in stage.iterdir()}
        manifest = {
            "schema": SCHEMA, "stage": "E1", "status": decision,
            "canonical_inputs_sha256": inputs, "rows": len(base), "a0_fields": len(a0),
            "selected_fields_union": sorted(selected), "causal_checks": causal_checks,
            "realised_cost_target_proximity": cost_proximity,
            "bounded_compute": bounded, "outputs_sha256": outputs,
            "runner_sha256": sha256(Path(__file__)),
            "prohibited_scope": "no model fitting, no E2+ ablations, no policy or portfolio changes",
        }
        dump_json(stage / "run_manifest.json", manifest)
        (stage / "manifest.sha256").write_text(f"{sha256(stage / 'run_manifest.json')}  run_manifest.json\n")
        os.replace(stage, output)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
