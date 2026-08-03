#!/usr/bin/env python3
"""Materialise the immutable Stage-0 root-cause diagnostic substrate.

The input alignment's ``exact_h12_gross_bps`` is an *execution-adjusted,
pre-fee* return: the frozen simulator has already applied its spread-aware
entry/fill convention.  It is therefore retained verbatim, but never relabelled
as a raw pre-spread market return.  Historical spread and slippage components
are not separately reconstructable from this source and are emitted as missing
with an explicit status instead of being guessed or double-counted.

This is a diagnostic materialisation only.  It neither fits a model nor changes
candidate admission, ranking, policy geometry, sizing, or portfolio behaviour.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data_perp/artifacts"
ALIGNMENT = ART / "historical_exact_h12_alignment_sidecar_research_only_20260731_v1/alignment_sidecar.parquet"
EVENTS = ART / "historical_exact_h12_postcost_events_20260731_v1/postcost_events.parquet"
PERSISTENCE = ART / "historical_exact_h12_postcost_persistence_labels_20260731_v1/postcost_persistence_labels.parquet"
COUNTERFACTUALS = ART / "stage_d_action_counterfactuals_20260731_v2/stage_d_action_counterfactuals.parquet"
OOF_SCORES = ART / "reconstructed_base_residual_stack_2022_2024_20260730_v3/oof_scores.parquet"
RAW_FEATURE_CONTRACT = ART / "long_exact_h12_raw_base_panel_20260730_v2/raw_feature_contract.json"
ACTION_FEATURES = ART / "stage_d_action_features_20260731_v5/stage_d_action_features.parquet"
ACTION_FEATURE_LINEAGE = ART / "stage_d_action_features_20260731_v5/stage_d_action_feature_lineage.parquet"
POLICY = ART / "s59_s52_base66_residualstate_meta_v9tail95_mlp_hierev_ev70_geometry_20260717_v2/simple_policy_optimiser/deployment/best_policy_params_perps.json"
DEFAULT_OUTPUT = ART / "root_cause_diagnostic_substrate_20260731_v4"

SCHEMA = "root_cause_diagnostic_substrate_v4"
IDENTITY = ("candidate_id", "side", "decision_ts")
SCORE_COLUMNS = (
    "score_base_alpha",
    "score_residual_alpha",
    # These are the frozen OOF expected-value maps used by the economic
    # residual audit.  Keep their lineage alongside the unitless alpha
    # scores; Stage 2 must never reconstruct them from outcome rows.
    "score_base_expected_ev",
    "score_residual_expected_ev",
    "score_residual_delta_alpha",
)
FUTURE_TOKENS = (
    "future", "mfe", "mae", "giveback", "exit_reason", "exit_hour",
    "first_event", "postcost", "retained", "timeout", "adverse", "label",
    "target", "outcome", "action_exit", "net_continue", "delta_continue",
)
REALIZED_COST_TOKENS = (
    "row_cost", "known_row_cost", "execution_cost", "fee_return", "total_cost",
)
TARGET_COLUMNS = {
    "gross_h12_bps", "execution_adjusted_gross_h12_bps", "fee_bps", "spread_bps",
    "slippage_bps", "total_cost_bps", "net_h12_bps", "action_delta_clean_bps",
    "action_continue_clean_value_bps", "action_exit_clean_value_bps",
}


class ContractError(RuntimeError):
    """Raised when an immutable diagnostic contract cannot be proven."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
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
    path.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def id_digest(values: Iterable[Any]) -> str:
    payload = "\n".join(sorted(str(value) for value in values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def display_path(path: Path) -> str:
    """Keep manifests readable without making unit-test inputs root-bound."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _utc(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")


def _read_policy_fixed_fee_bps(policy_path: Path) -> tuple[float, dict[str, Any]]:
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    selected = [dict(row) for row in payload.get("strategies", []) if bool(row.get("selected", True))]
    costs = sorted({float(row["cost_pct_per_side"]) for row in selected if row.get("cost_pct_per_side") is not None})
    if not selected or len(costs) != 1 or not np.isfinite(costs[0]) or costs[0] < 0.0:
        raise ContractError("frozen policy does not provide one finite selected per-side cost")
    per_side = costs[0]
    return 2.0 * per_side * 10_000.0, {
        "policy_sha256": sha256(policy_path),
        "selected_strategy_count": len(selected),
        "selected_per_side_cost_pct_values": costs,
        "fixed_round_trip_fee_bps": 2.0 * per_side * 10_000.0,
        "source": display_path(policy_path),
    }


def _load_oof(path: Path, wanted: set[str]) -> pd.DataFrame:
    columns = ["candidate_id", "__ts__", "side_name", *SCORE_COLUMNS, "stack_lineage", "residual_fold", "residual_is_oof"]
    scores = pd.read_parquet(path, columns=columns)
    scores["candidate_id"] = scores.candidate_id.astype(str)
    scores = scores.loc[scores.candidate_id.isin(wanted)].copy()
    if scores.candidate_id.duplicated().any() or set(scores.candidate_id) != wanted:
        raise ContractError("base/residual OOF artifact does not cover the exact diagnostic population one-to-one")
    _utc(scores, ("__ts__",))
    if not scores.residual_is_oof.astype(bool).all() or not scores.stack_lineage.astype(str).eq("frozen_pf_2022aug_2024").all():
        raise ContractError("base/residual rows lack the declared exact-ID frozen OOF provenance")
    scores = scores.rename(columns={"side_name": "score_side", "__ts__": "score_ts"})
    return scores


def _classify_feature(name: str) -> tuple[str, str]:
    lowered = str(name).lower()
    if lowered in TARGET_COLUMNS or lowered in {"exact_h12_gross_bps", "exact_h12_net_bps", "delta_continue_bps", "continue_better"}:
        return "REJECT_DIRECT_TARGET", "direct target/label column"
    if any(token in lowered for token in REALIZED_COST_TOKENS):
        return "REJECT_REALIZED_COST", "realised or future-resolved cost component"
    if lowered in {"estimated_net_if_exit_now_bps", "gross_return_at_action_bps"}:
        return "REVIEW_TARGET_ADJACENT", "action-value/target arithmetic-adjacent field"
    if any(token in lowered for token in FUTURE_TOKENS):
        return "REJECT_FUTURE_OR_OUTCOME", "future path, outcome, or target-derived semantic token"
    if "predicted" in lowered or "expected_ev" in lowered or "mapped" in lowered:
        return "REVIEW_MODEL_DERIVED", "requires explicit OOF/prequential lineage"
    if "exit_price" in lowered or "fill" in lowered:
        return "REJECT_FUTURE_FILL", "realised or future fill semantic token"
    return "NO_NAME_BASED_OVERLAP", "name does not establish causal admissibility"


def build_feature_target_proximity_report(
    *, raw_feature_contract: Path, action_features: Path, action_lineage: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    raw = json.loads(raw_feature_contract.read_text(encoding="utf-8"))
    for name in raw.get("raw_feature_columns", []):
        disposition, reason = _classify_feature(str(name))
        rows.append({
            "feature_source": str(raw_feature_contract.relative_to(ROOT)), "feature_name": str(name),
            "declared_point_in_time_safe": np.nan, "declared_live_reproducible": np.nan,
            "target_proximity_disposition": disposition, "target_proximity_reason": reason,
        })
    if action_features.is_file():
        lineage = pd.read_parquet(action_lineage) if action_lineage.is_file() else pd.DataFrame()
        lineage = lineage.drop_duplicates("feature_name") if "feature_name" in lineage else lineage
        lineage_map = lineage.set_index("feature_name") if not lineage.empty else pd.DataFrame()
        identity = {
            "candidate_id", "source_symbol", "side", "entry_ts", "first_clear_ts", "action_decision_ts",
            "action_execution_ts", "horizon_end_ts", "label_available_ts", "execution_policy_id", "cost_model_id",
            "path_source_id", "path_observed_through_bar_open_ts", "market_source_bar_open_ts",
            "market_feature_available_ts", "market_entry_source_bar_open_ts", "feature_available_ts",
            "eligible_universe_membership_sha256",
        }
        for name in pq.read_schema(action_features).names:
            if name in identity:
                continue
            disposition, reason = _classify_feature(name)
            declared_safe = np.nan
            declared_live = np.nan
            if not lineage_map.empty and name in lineage_map.index:
                declared_safe = lineage_map.loc[name].get("point_in_time_safe", np.nan)
                declared_live = lineage_map.loc[name].get("live_reproducible", np.nan)
            rows.append({
                "feature_source": str(action_features.relative_to(ROOT)), "feature_name": name,
                "declared_point_in_time_safe": declared_safe, "declared_live_reproducible": declared_live,
                "target_proximity_disposition": disposition, "target_proximity_reason": reason,
            })
    return pd.DataFrame(rows).sort_values(["feature_source", "feature_name"], kind="stable").reset_index(drop=True)


def _source_bindings() -> dict[str, dict[str, str]]:
    return {
        "reference_ideal_entry_gross": {
            "status": "NOT_AVAILABLE", "reason": "no immutable raw-pre-spread frozen-policy replay is materialised; current alignment gross already embeds spread-aware fills",
        },
        "executable_entry_gross": {
            "status": "AVAILABLE_EXECUTION_ADJUSTED_PRE_FEE", "source": str(ALIGNMENT.relative_to(ROOT)),
            "column": "exact_h12_gross_bps", "reason": "relative to executable entry and frozen spread-aware exit fill; not a raw pre-spread return",
        },
        "delayed_entry_1m_gross": {
            "status": "NOT_AVAILABLE", "reason": "sealed exact paths contain only the original 720-minute horizon; a delayed 12-hour label requires an additional immutable tail",
        },
        "delayed_entry_5m_gross": {
            "status": "NOT_AVAILABLE", "reason": "sealed exact paths contain only the original 720-minute horizon; a delayed 12-hour label requires an additional immutable tail",
        },
        "delayed_entry_10m_gross": {
            "status": "NOT_AVAILABLE", "reason": "sealed exact paths contain only the original 720-minute horizon; a delayed 12-hour label requires an additional immutable tail",
        },
        "frozen_policy_gross": {
            "status": "AVAILABLE_EXECUTION_ADJUSTED_PRE_FEE", "source": str(ALIGNMENT.relative_to(ROOT)), "column": "exact_h12_gross_bps",
        },
        "fee": {
            "status": "AVAILABLE_FUTURE_RESOLVED_OUTCOME_ONLY", "source": str(ALIGNMENT.relative_to(ROOT)), "column": "row_cost_bps",
        },
        "spread": {
            "status": "NOT_SEPARATELY_RECONSTRUCTABLE", "reason": "spread drag is embedded in the frozen execution-adjusted gross outcome; estimated spread is an ex-ante proxy, not realised separate spread cost",
        },
        "slippage": {
            "status": "NOT_SEPARATELY_RECONSTRUCTABLE", "reason": "the frozen fill convention embeds adverse-fill/gap effects in gross and stores no separate realised slippage component",
        },
    }


def build_substrate(
    *, alignment_path: Path, events_path: Path, persistence_path: Path, counterfactual_path: Path,
    oof_path: Path, policy_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    alignment_columns = [
        "candidate_id", "symbol", "side", "decision_ts", "feature_cutoff_ts", "entry_ts", "label_end_ts",
        "label_available_ts", "execution_policy_id", "cost_model_id", "policy_archetype", "execution_geometry_key",
        "execution_geometry_source", "exact_h12_gross_bps", "row_cost_bps", "exact_h12_net_bps",
        "estimated_spread_bps", "entry_half_spread_bps", "exit_half_spread_bps", "exit_reason", "exit_hour",
    ]
    ledger = pd.read_parquet(alignment_path, columns=alignment_columns)
    ledger["candidate_id"] = ledger.candidate_id.astype(str)
    if ledger.candidate_id.duplicated().any() or not ledger.side.isin(("long", "short")).all():
        raise ContractError("alignment identity or side contract failed")
    _utc(ledger, ("decision_ts", "feature_cutoff_ts", "entry_ts", "label_end_ts", "label_available_ts"))
    # The frozen main population is explicitly the later PF, USD-linear
    # perpetual universe.  Its CCXT symbols have ``/USD:USD`` settlement;
    # inverse PI contracts use e.g. ``/USD:BTC`` and are deliberately a
    # separate research population.  Fail closed if either slips into this
    # ledger instead of silently pooling incompatible return contracts.
    pf_symbol_pattern = r"[A-Z0-9]+/USD:USD"
    if not ledger.symbol.astype(str).str.fullmatch(pf_symbol_pattern).all():
        invalid = sorted(ledger.loc[~ledger.symbol.astype(str).str.fullmatch(pf_symbol_pattern), "symbol"].astype(str).unique())[:10]
        raise ContractError(f"mixed or non-PF USD-linear contract symbols in diagnostic population: {invalid}")
    if not ledger.feature_cutoff_ts.le(ledger.decision_ts).all() or not ledger.entry_ts.eq(ledger.decision_ts).all():
        raise ContractError("alignment feature/entry timing is non-causal")
    if not ledger.label_end_ts.eq(ledger.decision_ts + pd.Timedelta(hours=12)).all() or not ledger.label_available_ts.eq(ledger.label_end_ts).all():
        raise ContractError("alignment H12 label availability drift")
    values = ledger[["exact_h12_gross_bps", "row_cost_bps", "exact_h12_net_bps"]].to_numpy(float)
    if not np.isfinite(values).all() or not np.allclose(values[:, 0] - values[:, 1], values[:, 2], rtol=0.0, atol=1e-6):
        raise ContractError("frozen fee-only gross/net reconciliation failed")

    wanted = set(ledger.candidate_id)
    event_columns = [
        "candidate_id", "postcost_h0_event", "postcost_h0_favorable_minute", "postcost_h0_adverse_minute", "postcost_h0_resolved_minute",
        "postcost_h25_event", "postcost_h25_favorable_minute", "postcost_h25_adverse_minute", "postcost_h25_resolved_minute", "fixed_cost_bps",
    ]
    events = pd.read_parquet(events_path, columns=event_columns)
    events["candidate_id"] = events.candidate_id.astype(str)
    events = events.loc[events.candidate_id.isin(wanted)]
    if events.candidate_id.duplicated().any() or set(events.candidate_id) != wanted:
        raise ContractError("post-cost event source does not cover diagnostic IDs one-to-one")
    persistence = pd.read_parquet(persistence_path, columns=["candidate_id", "postcost_h0_four_state", "postcost_h0_retained_net", "postcost_h0_giveback_after_clear"])
    persistence["candidate_id"] = persistence.candidate_id.astype(str)
    persistence = persistence.loc[persistence.candidate_id.isin(wanted)]
    if persistence.candidate_id.duplicated().any() or set(persistence.candidate_id) != wanted:
        raise ContractError("persistence source does not cover diagnostic IDs one-to-one")
    counter = pd.read_parquet(counterfactual_path, columns=[
        "candidate_id", "action_decision_ts", "action_execution_ts", "net_continue_gross_bps", "net_exit_now_gross_bps", "delta_continue_bps",
    ])
    counter["candidate_id"] = counter.candidate_id.astype(str)
    if counter.candidate_id.duplicated().any() or not set(counter.candidate_id).issubset(wanted):
        raise ContractError("counterfactual ID contract failed")
    _utc(counter, ("action_decision_ts", "action_execution_ts"))
    if not counter.action_decision_ts.lt(counter.action_execution_ts).all():
        raise ContractError("counterfactual action timing failed")
    clean_delta = counter.net_continue_gross_bps.to_numpy(float) - counter.net_exit_now_gross_bps.to_numpy(float)
    if not np.allclose(clean_delta, counter.delta_continue_bps.to_numpy(float), rtol=0.0, atol=1e-6):
        raise ContractError("counterfactual gross-arm delta does not match sealed paired delta")

    fixed_fee_bps, policy_info = _read_policy_fixed_fee_bps(policy_path)
    counter["action_delta_clean_bps"] = clean_delta
    counter["action_continue_clean_value_bps"] = counter.net_continue_gross_bps.to_numpy(float) - fixed_fee_bps
    counter["action_exit_clean_value_bps"] = counter.net_exit_now_gross_bps.to_numpy(float) - fixed_fee_bps
    if not np.allclose(counter.action_continue_clean_value_bps - counter.action_exit_clean_value_bps, counter.action_delta_clean_bps, rtol=0.0, atol=1e-6):
        raise ContractError("fixed-ex-ante action target arithmetic failed")
    counter = counter.drop(columns=["delta_continue_bps"]).rename(columns={
        "net_continue_gross_bps": "action_continue_execution_adjusted_gross_bps",
        "net_exit_now_gross_bps": "action_exit_execution_adjusted_gross_bps",
    })

    scores = _load_oof(oof_path, wanted)
    ledger = ledger.merge(events, on="candidate_id", validate="one_to_one").merge(persistence, on="candidate_id", validate="one_to_one").merge(counter, on="candidate_id", how="left", validate="one_to_one").merge(scores, on="candidate_id", validate="one_to_one")
    # The OOF scorer is stamped at the feature cutoff, one bar before the
    # decision/entry timestamp.  Treating it as a decision-time score would
    # both reject valid causal lineage and obscure the one-bar availability
    # convention in the canonical alignment sidecar.
    if (
        not ledger.score_side.astype(str).eq(ledger.side.astype(str)).all()
        or not ledger.score_ts.eq(ledger.feature_cutoff_ts).all()
        or not ledger.score_ts.le(ledger.decision_ts).all()
    ):
        raise ContractError("exact-ID OOF score identity/cutoff lineage does not match the diagnostic row")
    if not np.isfinite(ledger.loc[:, SCORE_COLUMNS].to_numpy(float)).all():
        raise ContractError("exact-ID OOF scores are incomplete")

    ledger = ledger.rename(columns={
        "exact_h12_gross_bps": "execution_adjusted_gross_h12_bps",
        "exact_h12_net_bps": "net_h12_bps",
        "row_cost_bps": "fee_bps",
    })
    ledger["product"] = "perpetual"
    ledger["contract_family"] = "PF_USD_LINEAR_PERPETUAL"
    ledger["settlement_currency"] = "USD"
    ledger["contract_population_validation"] = "all symbols match CCXT /USD:USD; inverse PI /USD:<base> symbols rejected"
    ledger["gross_h12_bps"] = ledger.execution_adjusted_gross_h12_bps
    ledger["raw_pre_spread_gross_h12_bps"] = np.nan
    ledger["spread_bps"] = np.nan
    ledger["slippage_bps"] = np.nan
    ledger["total_cost_bps"] = ledger.fee_bps
    ledger["fixed_ex_ante_fee_bps"] = fixed_fee_bps
    ledger["gross_h12_semantics"] = "EXECUTION_ADJUSTED_PRE_FEE_SPREAD_EMBEDDED"
    ledger["cost_decomposition_status"] = "FEE_ONLY_EXACT_SPREAD_AND_SLIPPAGE_EMBEDDED_NOT_SEPARATELY_RECONSTRUCTABLE"
    ledger["fee_is_future_resolved"] = True
    ledger["net_reconciliation_bps"] = ledger.execution_adjusted_gross_h12_bps - ledger.total_cost_bps - ledger.net_h12_bps
    ledger["action_target_status"] = np.where(
        ledger.action_delta_clean_bps.notna(),
        "AVAILABLE_GROSS_ARMS_FIXED_EX_ANTE_FEE_TARGET_ONLY",
        "NOT_ELIGIBLE_NO_ACTIONABLE_FIRST_CLEAR",
    )
    if not np.allclose(ledger.net_reconciliation_bps.to_numpy(float), 0.0, rtol=0.0, atol=1e-6):
        raise ContractError("diagnostic net reconciliation drift")
    ledger = ledger.sort_values(["decision_ts", "candidate_id"], kind="stable").reset_index(drop=True)

    grouping = [("all", ledger), ("side", ledger.groupby("side", sort=True)), ("month", ledger.assign(month=ledger.decision_ts.dt.strftime("%Y-%m")).groupby("month", sort=True))]
    recon_rows: list[dict[str, Any]] = []
    for scope, source in grouping:
        iterator = [("ALL", source)] if scope == "all" else source
        for value, part in iterator:
            recon_rows.append({
                "scope": scope, "value": str(value), "rows": int(len(part)),
                "mean_execution_adjusted_gross_h12_bps": float(part.execution_adjusted_gross_h12_bps.mean()),
                "mean_fee_bps": float(part.fee_bps.mean()), "mean_net_h12_bps": float(part.net_h12_bps.mean()),
                "max_abs_fee_reconciliation_bps": float(np.abs(part.net_reconciliation_bps).max()),
                "spread_bps_status": "EMBEDDED_NOT_SEPARATELY_RECONSTRUCTABLE",
                "slippage_bps_status": "EMBEDDED_NOT_SEPARATELY_RECONSTRUCTABLE",
                "status": "FEE_ONLY_RECONCILED_NO_SPREAD_OR_SLIPPAGE_DOUBLE_COUNT",
            })
    reconciliation = pd.DataFrame(recon_rows)
    return ledger, reconciliation, {"policy_fixed_cost": policy_info, "source_bindings": _source_bindings()}


def run(
    *, output: Path = DEFAULT_OUTPUT, alignment_path: Path = ALIGNMENT, events_path: Path = EVENTS,
    persistence_path: Path = PERSISTENCE, counterfactual_path: Path = COUNTERFACTUALS,
    oof_path: Path = OOF_SCORES, policy_path: Path = POLICY, raw_feature_contract: Path = RAW_FEATURE_CONTRACT,
    action_features: Path = ACTION_FEATURES, action_feature_lineage: Path = ACTION_FEATURE_LINEAGE,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    ledger, reconciliation, details = build_substrate(
        alignment_path=alignment_path, events_path=events_path, persistence_path=persistence_path,
        counterfactual_path=counterfactual_path, oof_path=oof_path, policy_path=policy_path,
    )
    proximity = build_feature_target_proximity_report(
        raw_feature_contract=raw_feature_contract, action_features=action_features, action_lineage=action_feature_lineage,
    )
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        ledger.to_parquet(stage / "diagnostic_row_ledger.parquet", index=False, compression="zstd")
        reconciliation.to_parquet(stage / "target_cost_reconciliation.parquet", index=False, compression="zstd")
        proximity.to_parquet(stage / "feature_target_proximity_report.parquet", index=False, compression="zstd")
        manifest = {
            "schema": SCHEMA,
            "status": "MATERIALIZED_DIAGNOSTIC_ONLY_PARTIAL_COST_DECOMPOSITION_EXPLICIT",
            "promotion_eligible": False,
            "rows": int(len(ledger)),
            "ordered_candidate_id_sha256": id_digest(ledger.candidate_id),
            "row_identity": list(IDENTITY),
            "columns": list(ledger.columns),
            "timing_contract": {
                "feature_cutoff": "feature_cutoff_ts <= decision_ts",
                "entry": "entry_ts == decision_ts",
                "label": "label_end_ts == decision_ts + 12h; label_available_ts == label_end_ts",
                "action": "action_decision_ts < action_execution_ts where an actionable clear event exists",
            },
            "gross_cost_contract": {
                "gross_h12_bps": "exact frozen execution-adjusted pre-fee return, with spread-aware fill effects embedded",
                "raw_pre_spread_gross_h12_bps": "NOT_AVAILABLE; deliberately never inferred from execution-adjusted gross",
                "fee_bps": "future-resolved outcome fee; diagnostic/reconciliation only and forbidden model input",
                "spread_bps": "NOT_SEPARATELY_RECONSTRUCTABLE from this historical source; left null",
                "slippage_bps": "NOT_SEPARATELY_RECONSTRUCTABLE from this historical source; left null",
                "net": "execution_adjusted_gross_h12_bps - fee_bps exactly",
            },
            "product_population_contract": {
                "contract_family": "PF_USD_LINEAR_PERPETUAL",
                "settlement_currency": "USD",
                "source_lineage": "data_perp later frozen PF main population",
                "symbol_validation": "all rows must match ^[A-Z0-9]+/USD:USD$; mixed/inverse PI forms fail materialisation",
                "inverse_population_handling": "excluded; inverse PI uses /USD:<base> symbols and must remain a separately labelled research population",
            },
            "clean_action_target": {
                "formula": "action_continue_execution_adjusted_gross_bps - action_exit_execution_adjusted_gross_bps",
                "fixed_ex_ante_fee_bps": details["policy_fixed_cost"]["fixed_round_trip_fee_bps"],
                "prohibition": "known_row_cost_bps and all realised cost fields are absent from action target outputs and must never enter action features",
            },
            "score_lineage": {
                "artifact": str(oof_path.relative_to(ROOT)), "columns": list(SCORE_COLUMNS),
                "requirements_verified": "exact candidate ID, side and feature_cutoff timestamp (one bar before decision), stack_lineage=frozen_pf_2022aug_2024, residual_is_oof=true",
            },
            **details,
            "inputs": {
                str(path.relative_to(ROOT)): sha256(path)
                for path in (alignment_path, events_path, persistence_path, counterfactual_path, oof_path, policy_path, raw_feature_contract)
            },
            "outputs_sha256": {
                name: sha256(stage / name)
                for name in ("diagnostic_row_ledger.parquet", "target_cost_reconciliation.parquet", "feature_target_proximity_report.parquet")
            },
            "runner": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": sha256(Path(__file__))},
        }
        write_json(stage / "diagnostic_population_manifest.json", manifest)
        write_json(stage / "run_manifest.json", manifest)
        (stage / "manifest.sha256").write_text(f"{sha256(stage / 'run_manifest.json')}  run_manifest.json\n", encoding="utf-8")
        os.replace(stage, output)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(_safe(run(output=args.output)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
