#!/usr/bin/env python3
"""Matched, research-only attribution of the frozen Strict-R3 rich exit policy.

This runner answers one deliberately narrow question without touching the
live trader: holding the *target-free* BCF/current-v5 dual-MC1 route and its
BCF-MC1 auction priority fixed, how does the source-aligned 15-minute parent
proxy compare with the exact one-minute rich-policy state machine?

Arms
----
``parent_proxy_15m_decision``
    The canonical source-aligned parent-policy outcome, entered at the
    decision timestamp where that outcome is available.
``frozen_rich_15m_aggregated_decision``
    The *same frozen rich parameters* replayed over 48 completed 15-minute
    OHLC bars deterministically aggregated from the exact one-minute path.
    This is the clean bar-resolution control, not the legacy parent policy.
``exact_1m_rich_v1_decision``
    The frozen rich policy on complete post-decision one-minute paths.
``exact_1m_rich_v1_plus5``
    The same frozen rich policy on complete paths beginning five minutes
    after the decision.

Candidate routing is always performed *before* any outcome/path columns are
read: both frozen MC1 maps have already cleared +30 bps and ``priority_bps``
is the predeclared BCF-MC1 auction priority.  For the research headline,
invalid/incomplete outcomes are excluded only *after* that frozen route.  They
do not become zero-return pseudo-trades and do not reserve portfolio capacity.
Their count and reason are persisted, so this cannot be mistaken for a live
candidate-eligibility rule.

The portfolio comparison is chronological and uses one identical canonical
auction contract per arm (7x leverage, 80% margin budget, 10% margin slots,
two entries per decision).  Priority is *only* BCF MC1 expected bps.  The
canonical fixed 10%-margin-slot contract has no score-dependent sizing, so
this runner intentionally does not introduce a timestamp-local rank for
sizing or selection.

No live config, executor, exchange API, or runtime state is imported or
modified by this research producer.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.exact_1m_rich_policy_contract import (  # noqa: E402
    Exact1mRichExecutionContract,
    Exact1mRichV2ExecutionContract,
    RichExitExtensions,
    replay_exact_1m_rich_policy,
    replay_exact_1m_rich_policy_v2,
)
from extreme_price_movements.strict_r3_rich_policy_15m_control import (  # noqa: E402
    FrozenRich15mAggregationContract,
    replay_frozen_rich_policy_15m_aggregate,
)
from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    compute_replay_metrics,
    normalise_candidate_table,
    replay_candidates,
)
from extreme_price_movements.strict_r3_rich_policy import RichPolicyParams  # noqa: E402
from scripts.report_strict_r3_mc1_d2_controlled_portfolio import (  # noqa: E402
    CAUSAL_AUCTION_CURVE,
    _params as canonical_portfolio_params,
)


DEFAULT_CANDIDATE_DIR = ROOT / (
    "data_perp/artifacts/strict_r3_exact_1m_dual30_bcf_priority_candidates_"
    "decision_2025_2026_20260817_v2"
)
DEFAULT_SOURCE_BCF = ROOT / (
    "data_perp/artifacts/strict_r3_score_family_matched_mc1_canonical_policy_"
    "20260817_v7_bcf/predictions_bcf_mc1_d2.parquet"
)
DEFAULT_FROZEN_POLICY = ROOT / (
    "data_perp/artifacts/strict_r3_rich_policy_hpo_long_20260817_v1/"
    "frozen_challenger.json"
)
DEFAULT_SIMPLE_POLICY_CONTROL = ROOT / "config/strict_r3_frozen_15m_policy.json"
DEFAULT_OUT = ROOT / (
    "data_perp/artifacts/strict_r3_exact_1m_rich_matched_attribution_"
    "2025_2026_20260817_v1"
)

POLICY_COLUMNS = (
    "candidate_id",
    "__decision_ts__",
    "__symbol__",
    "side_name",
    "policy_path_valid",
    "policy_gross_bps",
    "policy_net_bps",
    "policy_exit_bar_15m",
    "policy_entry_price",
    "policy_exit_price",
    "policy_exit_reason",
    "policy_cost_bps",
)
RESEARCH_SCHEMA = "strict_r3_exact_1m_rich_matched_attribution_v2"
INVALID_OUTCOME_REASON = "OUTCOME_UNAVAILABLE_EXCLUDED_FROM_EVALUATION"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _utc(values: object) -> object:
    if isinstance(values, pd.Series):
        return pd.to_datetime(values, utc=True, errors="raise")
    return pd.Timestamp(values, tz="UTC") if pd.Timestamp(values).tzinfo is None else pd.Timestamp(values).tz_convert("UTC")


def _candidate_panel(candidate_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidate_dir = Path(candidate_dir).resolve()
    parquet = candidate_dir / "candidates.parquet"
    manifest_path = candidate_dir / "run_manifest.json"
    request_path = candidate_dir / "candidate_download_request.parquet"
    request_manifest_path = candidate_dir / "candidate_download_request.json"
    for path in (parquet, manifest_path, request_path, request_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing immutable target-free input: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    request_manifest = json.loads(request_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("selection", {}).get("target_free") is not True:
        raise AssertionError("candidate panel does not declare target-free routing")
    predicate = str(manifest.get("selection", {}).get("predicate", ""))
    if "bcf_mc1_expected_bps >= 30" not in predicate or "current_v5_mc1_expected_bps >= 30" not in predicate:
        raise AssertionError("candidate panel is not the frozen dual-MC1 >=30 route")
    if str(manifest.get("selection", {}).get("priority")) != "priority_bps = bcf_mc1_expected_bps":
        raise AssertionError("candidate panel does not bind BCF MC1 priority")
    columns = [
        "candidate_id", "timestamp", "symbol", "side_name", "entry_ts", "priority_bps",
        "bcf_mc1_expected_bps", "current_v5_mc1_expected_bps",
    ]
    frame = pd.read_parquet(parquet, columns=columns).copy()
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["timestamp"] = _utc(frame["timestamp"])
    frame["entry_ts"] = _utc(frame["entry_ts"])
    frame["symbol"] = frame["symbol"].astype(str)
    if frame["candidate_id"].duplicated().any():
        raise AssertionError("target-free candidate panel has duplicate candidate identities")
    if not frame["side_name"].astype(str).str.lower().eq("long").all():
        raise AssertionError("matched attribution is long-only")
    for column in ("priority_bps", "bcf_mc1_expected_bps", "current_v5_mc1_expected_bps"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(frame[column]).all():
            raise AssertionError(f"candidate panel has non-finite {column}")
    if not np.isclose(frame["priority_bps"], frame["bcf_mc1_expected_bps"], rtol=0.0, atol=1e-12).all():
        raise AssertionError("target-free BCF priority differs from BCF MC1 expected EV")
    if not (frame["bcf_mc1_expected_bps"].ge(30.0) & frame["current_v5_mc1_expected_bps"].ge(30.0)).all():
        raise AssertionError("candidate panel includes a row below the frozen dual admission threshold")
    audit = {
        "candidate_dir": str(candidate_dir),
        "candidate_sha256": _sha256(parquet),
        "candidate_manifest_sha256": _sha256(manifest_path),
        "request_sha256": _sha256(request_path),
        "request_manifest_sha256": _sha256(request_manifest_path),
        "request_schema": request_manifest.get("schema"),
        "request_contract_hash": request_manifest.get("contract_hash"),
        "rows": int(len(frame)),
        "target_free": True,
        "selection_predicate": predicate,
        "priority": manifest["selection"]["priority"],
    }
    return frame.sort_values(["timestamp", "candidate_id"], kind="stable").reset_index(drop=True), audit


def _read_source_policy(path: Path, candidate_ids: pd.Series) -> pd.DataFrame:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    source = pd.read_parquet(path, columns=list(POLICY_COLUMNS)).copy()
    source["candidate_id"] = source["candidate_id"].astype(str)
    if source["candidate_id"].duplicated().any():
        raise AssertionError("BCF policy source has duplicate candidate IDs")
    source = source.loc[source["candidate_id"].isin(set(candidate_ids.astype(str)))].copy()
    source["__decision_ts__"] = _utc(source["__decision_ts__"])
    if len(source) != int(candidate_ids.nunique()):
        missing = int(candidate_ids.nunique() - source["candidate_id"].nunique())
        raise AssertionError(f"BCF parent-policy source misses {missing} routed candidate IDs")
    return source


def _load_frozen_policy(path: Path) -> tuple[RichPolicyParams, float, dict[str, Any]]:
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = RichPolicyParams.from_mapping(dict(payload.get("params") or {}))
    median = float(payload.get("median_atr_fraction_fitted_on_complete_2024_development"))
    if not np.isfinite(median) or median <= 0.0:
        raise AssertionError("frozen rich-policy artifact has invalid development median ATR fraction")
    if not np.isclose(float(payload.get("cost_bps")), 100.0):
        raise AssertionError("frozen rich-policy artifact does not bind cost exactly once")
    return params, median, {"path": str(path), "sha256": _sha256(path), "schema": payload.get("schema")}


def _load_simple_policy_control(path: Path) -> tuple[RichPolicyParams, dict[str, Any]]:
    """Map the predeclared simple policy onto the exact-1m replay engine.

    The control keeps only the historical SL3 / activation-0.5 / giveback-0.25
    geometry.  Every rich-only transform is deliberately disabled, so the
    resulting paired arm measures what the already-live rich optimiser adds.
    """
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    winner = dict(payload.get("winner") or {})
    required = {"sl_mult", "trailing_activation_mult", "fixed_trailing_gap_mult"}
    missing = sorted(required.difference(winner))
    if missing:
        raise ValueError(f"simple-policy control misses parameters: {missing}")
    params = RichPolicyParams.from_mapping({
        **winner,
        "sl_abs_floor_pct": 0.0,
        "sl_abs_cap_pct": 0.0,
        "trailing_activation_min_pct": 0.0,
        "trailing_activation_cap_pct": 0.0,
        "trailing_activation_decay_half_life_bars": 0.0,
        "trailing_activation_decay_start_bars": 0,
        "trailing_activation_min_mult": 1.0,
        "atr_power": 1.0,
        "atr_multiplier": 1.0,
        "sl_atr_power": None,
        "sl_atr_multiplier": None,
        "tp_atr_power": None,
        "tp_atr_multiplier": None,
        "capital_protect_mfe_mult": 0.0,
        "capital_protect_regression_frac": 0.45,
        "capital_protect_lock_frac": None,
        "capital_protect_min_lock_bps": 0.0,
        "adverse_exit_enabled": False,
        "adverse_exit_theta": None,
    })
    return params, {
        "path": str(path),
        "sha256": _sha256(path),
        "schema": payload.get("schema"),
        "selection": payload.get("selection"),
        "control_params": params.to_dict(),
        "disabled_rich_controls": [
            "separate_sl_tp_atr_transform", "sl_absolute_floor_cap",
            "activation_absolute_floor_cap_decay", "capital_protection",
            "fast_adverse_exit",
        ],
    }


def _valid_parent_policy(source: pd.DataFrame) -> pd.Series:
    return (
        source["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(source["policy_net_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(source["policy_gross_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(source["policy_exit_bar_15m"], errors="coerce"))
        & np.isfinite(pd.to_numeric(source["policy_entry_price"], errors="coerce"))
        & np.isfinite(pd.to_numeric(source["policy_exit_price"], errors="coerce"))
    )


def _parent_proxy_outcomes(candidates: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    cols = ["candidate_id", *[column for column in POLICY_COLUMNS if column != "candidate_id"]]
    merged = candidates.merge(source.loc[:, cols], on="candidate_id", how="left", validate="one_to_one")
    if not _utc(merged["timestamp"]).equals(_utc(merged["__decision_ts__"])):
        raise AssertionError("source policy decision times differ from the frozen target-free candidate panel")
    if not merged["symbol"].astype(str).eq(merged["__symbol__"].astype(str)).all():
        raise AssertionError("source policy symbols differ from frozen target-free candidate panel")
    valid = _valid_parent_policy(merged)
    exit_bar = pd.to_numeric(merged["policy_exit_bar_15m"], errors="coerce")
    decision = _utc(merged["timestamp"])
    return pd.DataFrame({
        "candidate_id": merged["candidate_id"].astype(str),
        "decision_timestamp": decision,
        "entry_timestamp": decision,
        "entry_price": pd.to_numeric(merged["policy_entry_price"], errors="coerce"),
        "exit_timestamp": decision + pd.to_timedelta((exit_bar + 1.0).fillna(48.0) * 15.0, unit="min"),
        "exit_price": pd.to_numeric(merged["policy_exit_price"], errors="coerce"),
        "gross_bps": pd.to_numeric(merged["policy_gross_bps"], errors="coerce"),
        "net_bps": pd.to_numeric(merged["policy_net_bps"], errors="coerce"),
        "exit_reason": merged["policy_exit_reason"].fillna(INVALID_OUTCOME_REASON).astype(str),
        "outcome_available": valid.to_numpy(bool),
        "outcome_invalid_reason": np.where(valid, "", INVALID_OUTCOME_REASON),
        "outcome_source": "source_aligned_15m_parent_policy",
    })


@dataclass(frozen=True)
class ExactDataset:
    outcomes: pd.DataFrame
    audit: dict[str, Any]
    training_rows: pd.DataFrame
    entry: np.ndarray
    atr: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray


def _assert_v2_matches_live_oracle_sample(
    *,
    entries: np.ndarray,
    atr: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    entry_timestamps: pd.Series,
    candidate_ids: pd.Series,
    params: RichPolicyParams,
    median_atr_fraction: float,
    entry_delay_minutes: int,
    sample_size: int | None = 256,
    verify_live_state_machine: bool = False,
) -> dict[str, Any]:
    """Prove the accelerated default path matches the scalar live oracle.

    The v2 engine is used only as an accelerator with *default* extensions.
    Its semantics must be byte-for-byte equivalent on a deterministic sample
    before it is allowed to evaluate the full path population.
    """
    if sample_size is None:
        # An exhaustive audit is deliberately opt-in: it is suitable for the
        # offline, immutable label materialiser, while the large historical
        # attribution jobs retain a small deterministic sentinel for speed.
        take = np.arange(len(entries), dtype=np.int64)
        audited_rows = len(take)
        audit_scope = "all_valid_rows"
    else:
        if int(sample_size) <= 0:
            raise ValueError("sample_size must be positive or None for an exhaustive audit")
        audited_rows = min(int(sample_size), len(entries))
        hashes = pd.util.hash_pandas_object(candidate_ids.astype(str), index=False).to_numpy(np.uint64)
        take = np.argsort(hashes, kind="stable")[:audited_rows]
        audit_scope = "deterministic_candidate_id_sample"
    positions = pd.DataFrame({
        "entry_price": entries[take],
        "atr": atr[take],
        "entry_ts": entry_timestamps.iloc[take].reset_index(drop=True),
    })
    scalar = replay_exact_1m_rich_policy(
        positions=positions,
        highs=highs[take], lows=lows[take], closes=closes[take], params=params,
        median_atr_fraction=float(median_atr_fraction),
        contract=Exact1mRichExecutionContract(entry_delay_minutes=int(entry_delay_minutes)),
    )
    vector = replay_exact_1m_rich_policy_v2(
        entry=entries[take], atr=atr[take], highs=highs[take], lows=lows[take], closes=closes[take],
        entry_timestamps=entry_timestamps.iloc[take], params=params,
        median_atr_fraction=float(median_atr_fraction), extensions=RichExitExtensions(),
        contract=Exact1mRichV2ExecutionContract(entry_delay_minutes=int(entry_delay_minutes)),
    )
    for key in ("path_valid", "exit_minute", "exit_reason"):
        if not np.array_equal(np.asarray(scalar[key]), np.asarray(vector[key])):
            raise AssertionError(f"vectorized rich replay differs from scalar live oracle: {key}")
    for key in ("gross_bps", "net_bps", "exit_price"):
        if not np.allclose(np.asarray(scalar[key], dtype=float), np.asarray(vector[key], dtype=float), rtol=0.0, atol=1e-12, equal_nan=True):
            raise AssertionError(f"vectorized rich replay differs from scalar live oracle: {key}")
    scalar_ts = pd.to_datetime(scalar["exit_timestamp"], utc=True)
    vector_ts = pd.to_datetime(vector["exit_timestamp"], utc=True)
    if not scalar_ts.equals(vector_ts):
        raise AssertionError("vectorized rich replay differs from scalar live oracle: exit_timestamp")
    live_state_machine_rows = 0
    if verify_live_state_machine:
        # This calls the actual completed-1m live policy state machine without
        # any exchange client or order path.  It intentionally holds the
        # executable-book hard-stop option off because the historical label
        # contract uses the frozen threshold-fill proxy; fill-quality is a
        # separate execution audit.
        from extreme_price_movements.inference.strict_r3_live_execution import (
            _advance_rich_policy_position,
        )

        for local_idx, source_idx in enumerate(take):
            entry_ts = pd.Timestamp(entry_timestamps.iloc[int(source_idx)])
            entry_ts = entry_ts.tz_localize("UTC") if entry_ts.tzinfo is None else entry_ts.tz_convert("UTC")
            bars = pd.DataFrame({
                "high": highs[int(source_idx)],
                "low": lows[int(source_idx)],
                "close": closes[int(source_idx)],
            }, index=pd.date_range(entry_ts, periods=highs.shape[1], freq="min", tz="UTC"))
            live = _advance_rich_policy_position(
                position={
                    "entry_price": float(entries[int(source_idx)]),
                    "atr": float(atr[int(source_idx)]),
                    "entry_signal_atr": float(atr[int(source_idx)]),
                    "side": "long",
                    "entry_ts": entry_ts.isoformat(),
                    "timeout_ts": (entry_ts + pd.Timedelta(hours=12)).isoformat(),
                    "next_bar_ts": entry_ts.isoformat(),
                    "maximum_favourable": 0.0,
                    "maximum_adverse": 0.0,
                    "trailing_armed": False,
                    "capital_protect_armed": False,
                    "smooth_armed": False,
                    "rich_adaptive_activation_multiplier": 1.0,
                },
                bars=bars,
                params=params,
                median_atr_fraction=float(median_atr_fraction),
                executable_vwap_hard_stop=False,
                close_based_hard_stop=False,
            )
            live_exit = live.get("exit")
            if not isinstance(live_exit, dict):
                raise AssertionError("live minute state machine failed to resolve a complete H12 path")
            expected_ts = pd.Timestamp(scalar_ts[local_idx])
            if (
                str(live_exit.get("exit_reason")) != str(scalar["exit_reason"][local_idx])
                or pd.Timestamp(live_exit.get("exit_ts")).tz_convert("UTC") != expected_ts
                or not np.isclose(float(live_exit.get("exit_price")), float(scalar["exit_price"][local_idx]), rtol=0.0, atol=1e-12)
                or not np.isclose(float(live_exit.get("gross_bps")), float(scalar["gross_bps"][local_idx]), rtol=0.0, atol=1e-12)
                or not np.isclose(float(live_exit.get("net_bps")), float(scalar["net_bps"][local_idx]), rtol=0.0, atol=1e-12)
            ):
                raise AssertionError(
                    "scalar exact replay differs from the live completed-minute policy state machine"
                )
            live_state_machine_rows += 1
    receipt = {
        "engine": "vectorized_v2_default_extensions",
        "oracle_equivalence_rows": int(audited_rows),
        "oracle_equivalence_scope": audit_scope,
        "oracle_equivalence": "exact scalar-v1 equality for validity/reason/minute/timestamp and 1e-12 numeric outcomes",
    }
    if verify_live_state_machine:
        receipt["live_state_machine_equivalence_rows"] = int(live_state_machine_rows)
        receipt["live_state_machine_equivalence"] = (
            "exact equality against the completed-one-minute live policy state machine; "
            "historical threshold-fill proxy only, not exchange-fill quality"
        )
    return receipt


def _exact_1m_outcomes(
    candidates: pd.DataFrame,
    dataset_dir: Path,
    *,
    params: RichPolicyParams,
    median_atr_fraction: float,
    expected_entry_delay_minutes: int,
) -> ExactDataset:
    dataset_dir = Path(dataset_dir).resolve()
    manifest_path = dataset_dir / "dataset_manifest.json"
    audit_path = dataset_dir / "candidate_path_audit.parquet"
    rows_path = dataset_dir / "training_rows.parquet"
    paths_path = dataset_dir / "exact_paths.npz"
    for path in (manifest_path, audit_path, rows_path, paths_path):
        if not path.is_file():
            raise FileNotFoundError(f"exact-1m dataset is not yet materialised: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract_payload = dict(manifest.get("contract") or {})
    if int(contract_payload.get("entry_delay_minutes", -1)) != int(expected_entry_delay_minutes):
        raise AssertionError("exact-1m dataset does not have the requested entry-delay contract")
    exact_contract = Exact1mRichExecutionContract(
        entry_delay_minutes=int(expected_entry_delay_minutes)
    )
    exact_contract.validate()
    # The materializer records a source-path contract, whereas this runner
    # applies the rich exit-state contract to those already sealed paths.  The
    # schemas deliberately differ, so require their common executable
    # semantics instead of falsely requiring equal hashes.
    required_source_semantics = {
        "bar_minutes": 1,
        "horizon_minutes": 12 * 60,
        "policy_cost_bps_once": 100.0,
        "same_bar_activation_allowed": False,
    }
    for key, expected in required_source_semantics.items():
        value = contract_payload.get(key)
        if isinstance(expected, float):
            if not np.isclose(float(value), expected):
                raise AssertionError(f"exact-1m source contract has incompatible {key}")
        elif value != expected:
            raise AssertionError(f"exact-1m source contract has incompatible {key}")
    audit = pd.read_parquet(audit_path).copy()
    audit["candidate_id"] = audit["candidate_id"].astype(str)
    audit["timestamp"] = _utc(audit["timestamp"])
    audit["entry_ts"] = _utc(audit["entry_ts"])
    if audit["candidate_id"].duplicated().any():
        raise AssertionError("exact-1m path audit has duplicate candidate IDs")
    if set(audit["candidate_id"]) != set(candidates["candidate_id"]):
        raise AssertionError("exact-1m audit candidate identity set differs from target-free route")
    lookup = candidates.loc[:, ["candidate_id", "timestamp", "symbol"]].merge(
        audit.loc[:, ["candidate_id", "timestamp", "symbol", "entry_ts", "path_valid", "path_invalid_reason"]],
        on="candidate_id", how="left", suffixes=("_candidate", "_exact"), validate="one_to_one",
    )
    if not _utc(lookup["timestamp_candidate"]).equals(_utc(lookup["timestamp_exact"])):
        raise AssertionError("exact-1m audit decision timestamp differs from target-free route")
    if not lookup["symbol_candidate"].astype(str).eq(lookup["symbol_exact"].astype(str)).all():
        raise AssertionError("exact-1m audit symbol differs from target-free route")
    expected_entry = _utc(lookup["timestamp_candidate"]) + pd.Timedelta(minutes=int(expected_entry_delay_minutes))
    if not _utc(lookup["entry_ts"]).equals(expected_entry):
        raise AssertionError("exact-1m audit entries differ from declared decision-delay contract")

    training = pd.read_parquet(rows_path).copy()
    training["candidate_id"] = training["candidate_id"].astype(str)
    training["entry_ts"] = _utc(training["entry_ts"])
    arrays = np.load(paths_path, allow_pickle=False)
    for key in ("entry", "atr", "high", "low", "close", "candidate_id"):
        if key not in arrays:
            raise AssertionError(f"exact-1m path archive lacks {key}")
    path_ids = arrays["candidate_id"].astype(str)
    if len(training) != len(path_ids) or not training["candidate_id"].astype(str).equals(pd.Series(path_ids)):
        raise AssertionError("exact-1m training rows are not aligned to the path archive")
    if training["candidate_id"].duplicated().any():
        raise AssertionError("exact-1m training rows have duplicate candidate IDs")
    entries = np.asarray(arrays["entry"], dtype=float)
    atr = np.asarray(arrays["atr"], dtype=float)
    highs = np.asarray(arrays["high"], dtype=float)
    lows = np.asarray(arrays["low"], dtype=float)
    closes = np.asarray(arrays["close"], dtype=float)
    acceleration_audit = _assert_v2_matches_live_oracle_sample(
        entries=entries, atr=atr, highs=highs, lows=lows, closes=closes,
        entry_timestamps=training["entry_ts"], candidate_ids=training["candidate_id"],
        params=params, median_atr_fraction=float(median_atr_fraction),
        entry_delay_minutes=int(expected_entry_delay_minutes),
    )
    result = replay_exact_1m_rich_policy_v2(
        entry=entries, atr=atr, highs=highs, lows=lows, closes=closes,
        entry_timestamps=training["entry_ts"], params=params,
        median_atr_fraction=float(median_atr_fraction), extensions=RichExitExtensions(),
        contract=Exact1mRichV2ExecutionContract(
            entry_delay_minutes=int(expected_entry_delay_minutes),
        ),
    )
    if not result["path_valid"].all():
        raise AssertionError("materialised complete exact-1m paths failed frozen rich replay")
    realised = pd.DataFrame({
        "candidate_id": training["candidate_id"].astype(str),
        "entry_price": entries,
        "exit_timestamp": pd.to_datetime(result["exit_timestamp"], utc=True),
        "exit_price": np.asarray(result["exit_price"], dtype=float),
        "gross_bps": np.asarray(result["gross_bps"], dtype=float),
        "net_bps": np.asarray(result["net_bps"], dtype=float),
        "exit_reason": np.asarray(result["exit_reason"], dtype=object),
        "exit_minute": np.asarray(result["exit_minute"], dtype=np.int16),
    })
    base = pd.DataFrame({
        "candidate_id": lookup["candidate_id"].astype(str),
        "decision_timestamp": _utc(lookup["timestamp_candidate"]),
        "entry_timestamp": _utc(lookup["entry_ts"]),
        "outcome_available": lookup["path_valid"].fillna(False).astype(bool),
        "outcome_invalid_reason": lookup["path_invalid_reason"].fillna(INVALID_OUTCOME_REASON).astype(str),
    })
    outcome = base.merge(realised, on="candidate_id", how="left", validate="one_to_one")
    valid = outcome["outcome_available"].to_numpy(bool)
    required = outcome.loc[valid, ["entry_price", "exit_timestamp", "exit_price", "gross_bps", "net_bps"]]
    if required.isna().any().any():
        raise AssertionError("a path-valid exact-1m candidate lacks a frozen rich-policy outcome")
    outcome["exit_reason"] = outcome["exit_reason"].fillna(INVALID_OUTCOME_REASON).astype(str)
    outcome["outcome_source"] = "exact_1m_rich_v1_frozen"
    return ExactDataset(
        outcomes=outcome,
        audit={
            "dataset_dir": str(dataset_dir), "manifest_sha256": _sha256(manifest_path),
            "training_rows_sha256": _sha256(rows_path), "path_archive_sha256": _sha256(paths_path),
            "path_audit_sha256": _sha256(audit_path),
            "source_path_contract": contract_payload,
            "source_path_contract_hash": manifest.get("contract_hash"),
            "rich_replay_contract": exact_contract.to_dict(),
            "rich_replay_contract_hash": exact_contract.hash,
            "source_to_rich_semantic_compatibility": sorted(required_source_semantics),
            "replay_acceleration": acceleration_audit,
            "entry_delay_minutes": int(expected_entry_delay_minutes),
            "routed_rows": int(len(base)), "path_valid_rows": int(valid.sum()),
            "path_invalid_rows": int((~valid).sum()),
        },
        training_rows=training.loc[:, ["candidate_id", "entry_ts"]].copy(),
        entry=entries,
        atr=atr,
        highs=highs,
        lows=lows,
        closes=closes,
    )


def _frozen_rich_15m_aggregated_outcomes(
    exact: ExactDataset,
    *,
    params: RichPolicyParams,
    median_atr_fraction: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Project one exact decision-entry dataset onto the frozen 15m control.

    This intentionally shares the already sealed target-free route and
    source paths with the exact decision arm.  Therefore any difference is
    due to aggregate-bar resolution rather than entry, parameters, ATR,
    cost, candidate identity, policy geometry, or coverage treatment.
    """
    contract = FrozenRich15mAggregationContract()
    contract.validate()
    rows = exact.training_rows.copy()
    result = replay_frozen_rich_policy_15m_aggregate(
        entry=exact.entry,
        atr=exact.atr,
        highs=exact.highs,
        lows=exact.lows,
        closes=exact.closes,
        entry_timestamps=rows["entry_ts"],
        params=params,
        median_atr_fraction=float(median_atr_fraction),
        contract=contract,
    )
    if not result["path_valid"].all():
        raise AssertionError("complete exact one-minute paths failed the aggregated 15m control")
    realised = pd.DataFrame({
        "candidate_id": rows["candidate_id"].astype(str),
        "entry_price": np.asarray(exact.entry, dtype=float),
        "exit_timestamp": pd.to_datetime(result["exit_timestamp"], utc=True),
        "exit_price": np.asarray(result["exit_price"], dtype=float),
        "gross_bps": np.asarray(result["gross_bps"], dtype=float),
        "net_bps": np.asarray(result["net_bps"], dtype=float),
        "exit_reason": np.asarray(result["exit_reason"], dtype=object),
        "exit_bar_15m": np.asarray(result["exit_bar_15m"], dtype=np.int16),
    })
    base = exact.outcomes.loc[:, [
        "candidate_id", "decision_timestamp", "entry_timestamp",
        "outcome_available", "outcome_invalid_reason",
    ]].copy()
    outcome = base.merge(realised, on="candidate_id", how="left", validate="one_to_one")
    valid = outcome["outcome_available"].fillna(False).astype(bool)
    required = outcome.loc[valid, ["entry_price", "exit_timestamp", "exit_price", "gross_bps", "net_bps"]]
    if required.isna().any().any():
        raise AssertionError("a path-valid exact decision candidate lacks a frozen 15m aggregate outcome")
    outcome["exit_reason"] = outcome["exit_reason"].fillna(INVALID_OUTCOME_REASON).astype(str)
    outcome["outcome_source"] = "frozen_rich_15m_aggregated_from_exact_1m"
    return outcome, {
        "control_contract": contract.to_dict(),
        "control_contract_hash": contract.hash,
        "input_exact_dataset": exact.audit,
        "aggregation": {
            "source": "complete exact one-minute OHLC paths",
            "bars": 48,
            "timestamp": "completed_15m_bar_end",
            "convention": "aggregate low-cross with live stop/capital/trailing/fast-adverse priority; prior-aggregate-bar trailing arm",
        },
        "path_valid_rows": int(valid.sum()),
        "path_invalid_rows": int((~valid).sum()),
    }


def _portfolio_candidates(
    routed: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    arm: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Form an evaluation-only complete-outcome table *after* frozen routing.

    The second return is the complete routed/population audit.  No outcome
    field appears in the target-free candidate identity/priority contract.
    """
    merged = routed.merge(outcomes, on="candidate_id", how="left", validate="one_to_one")
    if merged["outcome_available"].isna().any():
        raise AssertionError("outcome producer did not account for every routed candidate")
    population = merged.loc[:, [
        "candidate_id", "timestamp", "symbol", "priority_bps", "bcf_mc1_expected_bps",
        "current_v5_mc1_expected_bps", "decision_timestamp", "entry_timestamp",
        "outcome_available", "outcome_invalid_reason", "outcome_source",
    ]].copy()
    valid = merged["outcome_available"].fillna(False).astype(bool)
    # This exclusion happens only here, after a sealed target-free route has
    # been joined to outcomes.  It is a supervised/replay coverage rule, not a
    # historical candidate eligibility rule and it does not reserve capacity.
    selected = merged.loc[valid].copy()
    if selected.empty:
        raise RuntimeError(f"{arm}: no label-complete outcome rows after frozen route")
    exit_ts = _utc(selected["exit_timestamp"])
    entry_ts = _utc(selected["entry_timestamp"])
    holding_bars = np.maximum(
        1,
        np.ceil((exit_ts - entry_ts).dt.total_seconds().to_numpy(float) / (15.0 * 60.0)),
    ).astype(int)
    candidate = pd.DataFrame({
        # A +5m exact arm is scheduled at its actual entry time.  The auction
        # score/rank is nevertheless derived from its frozen decision cohort.
        "timestamp": entry_ts,
        "decision_timestamp": _utc(selected["decision_timestamp"]),
        "candidate_id": selected["candidate_id"].astype(str),
        "symbol": selected["symbol"].astype(str), "side": "long",
        "strategy_id": "strict_r3_exact_1m_rich_matched_long",
        "policy_archetype": "strict_r3_exact_1m_rich_matched_long",
        # The canonical margin-slot portfolio has fixed 10%-wallet slots and
        # rank multipliers fixed at one.  A constant pass-through score keeps
        # the generic portfolio API satisfied without creating an additional
        # timestamp-local rank rule.  Selection/order is solely priority_bps.
        "normalized_rank_score": 1.0,
        "strategy_rank_pct": 1.0,
        "base_strategy_threshold": 0.0,
        "calibrated_score": 1.0,
        # ``priority_bps`` is authoritative for ordering.  The normalised
        # rank is retained only for the predeclared sizing formula.
        "portfolio_priority_adjustment": pd.to_numeric(selected["priority_bps"], errors="raise"),
        "entry_price": pd.to_numeric(selected["entry_price"], errors="raise"),
        "exit_timestamp": exit_ts,
        "exit_price": pd.to_numeric(selected["exit_price"], errors="raise"),
        "net_return": pd.to_numeric(selected["net_bps"], errors="raise") / 10_000.0,
        "gross_return": pd.to_numeric(selected["gross_bps"], errors="raise") / 10_000.0,
        "holding_bars": holding_bars,
        "simple_policy_exit_reason": selected["exit_reason"].astype(str),
        # Every outcome engine has already included the same 100-bps policy
        # cost once in net_bps.  Auction friction must not debit it again.
        "fees_bps": 100.0, "expected_friction_bps": 0.0, "price_gap_bps": 0.0,
        "liquidity_capacity_weight": 1.0, "policy_outcome_available": True,
        "outcome_source": selected["outcome_source"].astype(str),
        "arm": arm,
    })
    return normalise_candidate_table(candidate), population


def _attach_candidate_ids(decisions: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    result = decisions.copy()
    source = candidates.reset_index(drop=True)
    index = pd.to_numeric(result.get("candidate_index"), errors="raise").astype(int)
    if (index < 0).any() or (index >= len(source)).any():
        raise AssertionError("portfolio decision candidate index is out of range")
    for column in ("candidate_id", "decision_timestamp", "outcome_source", "arm"):
        result[column] = source.iloc[index.to_numpy()][column].to_numpy()
    return result


def _metrics_by_month(accepted: pd.DataFrame, arm: str) -> pd.DataFrame:
    data = accepted.copy()
    if data.empty:
        return pd.DataFrame(columns=[
            "arm", "month", "accepted_trades", "net_ev_bps_per_trade", "gross_ev_bps_per_trade",
            "net_sum_bps", "gross_sum_bps", "portfolio_net_pnl_quote", "exit_reason_count",
        ])
    data["month"] = _utc(data["decision_timestamp"]).dt.strftime("%Y-%m")
    data["net_bps"] = pd.to_numeric(data["position_net_return"], errors="coerce") * 10_000.0
    data["gross_bps"] = pd.to_numeric(data["position_gross_return"], errors="coerce") * 10_000.0
    data["portfolio_net_pnl_quote"] = (
        pd.to_numeric(data["position_size"], errors="coerce")
        * pd.to_numeric(data["position_net_return"], errors="coerce")
    )
    out = data.groupby("month", sort=True).agg(
        accepted_trades=("candidate_id", "size"),
        net_ev_bps_per_trade=("net_bps", "mean"),
        gross_ev_bps_per_trade=("gross_bps", "mean"),
        net_sum_bps=("net_bps", "sum"), gross_sum_bps=("gross_bps", "sum"),
        portfolio_net_pnl_quote=("portfolio_net_pnl_quote", "sum"),
    ).reset_index()
    out.insert(0, "arm", arm)
    return out


def _exit_reason_metrics(accepted: pd.DataFrame, arm: str) -> pd.DataFrame:
    data = accepted.copy()
    if data.empty:
        return pd.DataFrame(columns=["arm", "exit_reason", "trades", "share", "net_ev_bps_per_trade", "net_sum_bps"])
    data["net_bps"] = pd.to_numeric(data["position_net_return"], errors="coerce") * 10_000.0
    out = data.groupby("position_exit_reason", sort=True).agg(
        trades=("candidate_id", "size"), net_ev_bps_per_trade=("net_bps", "mean"),
        net_sum_bps=("net_bps", "sum"),
    ).reset_index().rename(columns={"position_exit_reason": "exit_reason"})
    out["share"] = out["trades"] / max(int(out["trades"].sum()), 1)
    out.insert(0, "arm", arm)
    return out


def _headline_metrics(
    *,
    arm: str,
    routed_population: pd.DataFrame,
    candidates: pd.DataFrame,
    decisions: pd.DataFrame,
    equity: pd.DataFrame,
) -> dict[str, Any]:
    accepted = decisions.loc[decisions["accepted"].fillna(False).astype(bool)].copy()
    raw = compute_replay_metrics(candidates, decisions, equity, params=canonical_portfolio_params())
    net_bps = pd.to_numeric(accepted.get("position_net_return"), errors="coerce") * 10_000.0
    return {
        "arm": arm,
        "routed_candidates_target_free": int(len(routed_population)),
        "label_complete_candidates_after_route": int(len(candidates)),
        "excluded_invalid_or_incomplete_after_route": int(len(routed_population) - len(candidates)),
        "outcome_coverage_after_route": float(len(candidates) / max(len(routed_population), 1)),
        "portfolio_accepted_trades": int(len(accepted)),
        "net_ev_bps_per_trade": float(net_bps.mean()) if len(net_bps) else float("nan"),
        "net_sum_bps": float(net_bps.sum()) if len(net_bps) else 0.0,
        "portfolio_net_pnl_quote": float(raw.get("net_pnl", np.nan)),
        "portfolio_final_wallet": float(raw.get("final_wallet", np.nan)),
        "portfolio_max_drawdown": float(raw.get("max_drawdown", np.nan)),
        "portfolio_sortino": float(raw.get("sortino", np.nan)),
        "portfolio_worst_week_return": float(raw.get("worst_week", np.nan)),
        "auction_priority": "BCF MC1 priority_bps only; fixed margin slots, no timestamp-local rank",
        "evaluation_rule": "label-complete outcomes are excluded after target-free routing; no pseudo-trade or capacity reservation",
    }


def _write_arm(
    output: Path,
    arm: str,
    candidates: pd.DataFrame,
    population: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    decisions, equity, _ = replay_candidates(
        candidates, canonical_portfolio_params(), mode="global_auction", ev_curve=CAUSAL_AUCTION_CURVE,
        market_mode="perp",
    )
    decisions = _attach_candidate_ids(decisions, candidates)
    accepted = decisions.loc[decisions["accepted"].fillna(False).astype(bool)].copy()
    metrics = _headline_metrics(
        arm=arm, routed_population=population, candidates=candidates, decisions=decisions, equity=equity,
    )
    (output / f"{arm}_portfolio_candidates.parquet").parent.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(output / f"{arm}_portfolio_candidates.parquet", index=False, compression="zstd")
    population.to_parquet(output / f"{arm}_routed_outcome_coverage.parquet", index=False, compression="zstd")
    decisions.to_parquet(output / f"{arm}_portfolio_decisions.parquet", index=False, compression="zstd")
    accepted.to_parquet(output / f"{arm}_accepted_trades.parquet", index=False, compression="zstd")
    equity.to_parquet(output / f"{arm}_portfolio_equity.parquet", index=False, compression="zstd")
    return metrics, _metrics_by_month(accepted, arm), _exit_reason_metrics(accepted, arm), accepted


def _comparison_deltas(summary: pd.DataFrame, monthly: pd.DataFrame, exits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Keep legacy policy, bar-resolution, and entry-delay effects separate."""
    pairs = (
        ("legacy_policy_and_resolution_delta_exact1m_decision_minus_parent15m", "exact_1m_rich_v1_decision", "parent_proxy_15m_decision"),
        ("rich_parameter_uplift_exact1m_decision_minus_simple_policy_control", "exact_1m_rich_v1_decision", "simple_policy_1m_control_decision"),
        ("clean_resolution_delta_exact1m_decision_minus_frozen_rich_15m_aggregated_decision", "exact_1m_rich_v1_decision", "frozen_rich_15m_aggregated_decision"),
        ("entry_delay_delta_exact1m_plus5_minus_decision", "exact_1m_rich_v1_plus5", "exact_1m_rich_v1_decision"),
    )
    numeric_summary = (
        "label_complete_candidates_after_route", "portfolio_accepted_trades",
        "net_ev_bps_per_trade", "net_sum_bps", "portfolio_net_pnl_quote",
        "portfolio_final_wallet", "portfolio_max_drawdown", "portfolio_sortino",
        "portfolio_worst_week_return",
    )
    overall_rows: list[dict[str, Any]] = []
    monthly_rows: list[pd.DataFrame] = []
    exit_rows: list[pd.DataFrame] = []
    for comparison, treatment, control in pairs:
        by_arm = summary.set_index("arm")
        if treatment in by_arm.index and control in by_arm.index:
            row: dict[str, Any] = {"comparison": comparison, "treatment_arm": treatment, "control_arm": control}
            for column in numeric_summary:
                row[f"delta_{column}"] = float(by_arm.loc[treatment, column] - by_arm.loc[control, column])
            overall_rows.append(row)
        if not monthly.empty:
            left = monthly.loc[monthly["arm"].eq(treatment)].drop(columns="arm")
            right = monthly.loc[monthly["arm"].eq(control)].drop(columns="arm")
            if not left.empty and not right.empty:
                merged = left.merge(right, on="month", how="outer", suffixes=("_treatment", "_control"))
                merged.insert(0, "comparison", comparison)
                for field in (
                    "accepted_trades", "net_ev_bps_per_trade", "gross_ev_bps_per_trade",
                    "net_sum_bps", "gross_sum_bps", "portfolio_net_pnl_quote",
                ):
                    merged[f"delta_{field}"] = (
                        pd.to_numeric(merged[f"{field}_treatment"], errors="coerce")
                        - pd.to_numeric(merged[f"{field}_control"], errors="coerce")
                    )
                monthly_rows.append(merged)
        if not exits.empty:
            left = exits.loc[exits["arm"].eq(treatment)].drop(columns="arm")
            right = exits.loc[exits["arm"].eq(control)].drop(columns="arm")
            if not left.empty and not right.empty:
                merged = left.merge(right, on="exit_reason", how="outer", suffixes=("_treatment", "_control"))
                merged.insert(0, "comparison", comparison)
                for field in ("trades", "share", "net_ev_bps_per_trade", "net_sum_bps"):
                    merged[f"delta_{field}"] = (
                        pd.to_numeric(merged[f"{field}_treatment"], errors="coerce")
                        - pd.to_numeric(merged[f"{field}_control"], errors="coerce")
                    )
                exit_rows.append(merged)
    return (
        pd.DataFrame(overall_rows),
        pd.concat(monthly_rows, ignore_index=True) if monthly_rows else pd.DataFrame(),
        pd.concat(exit_rows, ignore_index=True) if exit_rows else pd.DataFrame(),
    )


def _assert_empty_or_new(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite immutable output: {output}")
    output.mkdir(parents=True, exist_ok=False)


def _run(args: argparse.Namespace) -> Path:
    output = Path(args.out_dir).resolve()
    _assert_empty_or_new(output)
    routed, route_audit = _candidate_panel(Path(args.candidate_dir))
    source = _read_source_policy(Path(args.source_bcf), routed["candidate_id"])
    params, median, frozen_audit = _load_frozen_policy(Path(args.frozen_policy))
    simple_params, simple_audit = _load_simple_policy_control(Path(args.simple_policy_control))

    parent_outcomes = _parent_proxy_outcomes(routed, source)
    arms: list[tuple[str, pd.DataFrame, dict[str, Any]]] = [
        ("parent_proxy_15m_decision", parent_outcomes, {"source_bcf": str(Path(args.source_bcf).resolve()), "source_bcf_sha256": _sha256(Path(args.source_bcf))}),
    ]
    if args.exact_decision_dir is not None:
        simple = _exact_1m_outcomes(
            routed, Path(args.exact_decision_dir), params=simple_params, median_atr_fraction=median,
            expected_entry_delay_minutes=0,
        )
        arms.append(("simple_policy_1m_control_decision", simple.outcomes, simple.audit | {"simple_policy_control": simple_audit}))
        # ``ExactDataset`` owns three dense 720-minute path matrices.  The
        # outcome frame and receipt above are all later stages need; retaining
        # the matrices while replaying the frozen-rich decision and +5m arms
        # needlessly triples peak memory and can make this otherwise
        # deterministic research comparison die part-way through an immutable
        # output.  Release only the local accelerator buffers, never results.
        del simple
        gc.collect()
        decision = _exact_1m_outcomes(
            routed, Path(args.exact_decision_dir), params=params, median_atr_fraction=median,
            expected_entry_delay_minutes=0,
        )
        aggregate_outcomes, aggregate_audit = _frozen_rich_15m_aggregated_outcomes(
            decision, params=params, median_atr_fraction=median,
        )
        arms.append(("frozen_rich_15m_aggregated_decision", aggregate_outcomes, aggregate_audit))
        arms.append(("exact_1m_rich_v1_decision", decision.outcomes, decision.audit))
        del aggregate_outcomes, decision
        gc.collect()
    if args.exact_plus5_dir is not None:
        plus5 = _exact_1m_outcomes(
            routed, Path(args.exact_plus5_dir), params=params, median_atr_fraction=median,
            expected_entry_delay_minutes=5,
        )
        arms.append(("exact_1m_rich_v1_plus5", plus5.outcomes, plus5.audit))
        del plus5
        gc.collect()
    if bool(args.require_all_arms) and len(arms) != 5:
        raise ValueError("--require-all-arms requires both an exact decision and exact +5m dataset")

    all_metrics: list[dict[str, Any]] = []
    monthly: list[pd.DataFrame] = []
    exits: list[pd.DataFrame] = []
    arm_receipts: dict[str, Any] = {}
    for arm, outcome, audit in arms:
        candidate, population = _portfolio_candidates(routed, outcome, arm=arm)
        metrics, month, reason, _ = _write_arm(output, arm, candidate, population)
        all_metrics.append(metrics)
        monthly.append(month)
        exits.append(reason)
        arm_receipts[arm] = audit
    summary = pd.DataFrame(all_metrics)
    summary.to_parquet(output / "attribution_summary.parquet", index=False, compression="zstd")
    summary.to_csv(output / "attribution_summary.csv", index=False)
    monthly_out = pd.concat(monthly, ignore_index=True) if monthly else pd.DataFrame()
    monthly_out.to_parquet(output / "monthly_portfolio_metrics.parquet", index=False, compression="zstd")
    exit_out = pd.concat(exits, ignore_index=True) if exits else pd.DataFrame()
    exit_out.to_parquet(output / "exit_reason_metrics.parquet", index=False, compression="zstd")
    overall_delta, monthly_delta, exit_delta = _comparison_deltas(summary, monthly_out, exit_out)
    overall_delta.to_parquet(output / "attribution_deltas.parquet", index=False, compression="zstd")
    monthly_delta.to_parquet(output / "monthly_attribution_deltas.parquet", index=False, compression="zstd")
    exit_delta.to_parquet(output / "exit_reason_attribution_deltas.parquet", index=False, compression="zstd")
    if not exit_out.empty:
        pivot = exit_out.pivot(index="exit_reason", columns="arm", values="net_ev_bps_per_trade").reset_index()
        pivot.to_parquet(output / "exit_reason_net_ev_delta_pivot.parquet", index=False, compression="zstd")
    route_audit.update({
        "frozen_policy": frozen_audit,
        "simple_policy_control": simple_audit,
        "arms": arm_receipts,
        "portfolio_contract": {
            "source": "scripts.report_strict_r3_mc1_d2_controlled_portfolio._params",
            "7x_leverage": True, "margin_budget_pct": 0.80,
            "margin_slot_pct": 0.10, "max_new_entries_per_decision": 2,
            "priority": "BCF MC1 priority_bps only; no timestamp-local rank",
        },
        "outcome_handling": {
            "routing": "target-free before outcomes are read",
            "evaluation": "invalid/incomplete outcomes excluded after route; no capacity reservation",
            "not_live_eligibility": True,
        },
        "output_schema": RESEARCH_SCHEMA,
        "code_sha256": _sha256(Path(__file__).resolve()),
    })
    (output / "run_manifest.json").write_text(
        json.dumps(_json_safe(route_audit), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--source-bcf", type=Path, default=DEFAULT_SOURCE_BCF)
    parser.add_argument("--frozen-policy", type=Path, default=DEFAULT_FROZEN_POLICY)
    parser.add_argument("--simple-policy-control", type=Path, default=DEFAULT_SIMPLE_POLICY_CONTROL)
    parser.add_argument("--exact-decision-dir", type=Path, default=None)
    parser.add_argument("--exact-plus5-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--require-all-arms", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(_run(parse_args()))


if __name__ == "__main__":
    main()
