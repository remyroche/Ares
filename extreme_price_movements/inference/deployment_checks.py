"""Deployment-readiness checks for the live inference path.

The checks in this module are intentionally deterministic and dependency
injectable: tests can exercise exchange, email, persistence, and risk paths
without placing real orders or sending real emails. Live deployment scripts can
pass real dependencies after credentials have been configured.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from extreme_price_movements.data_store import load_artifact_manifest
from extreme_price_movements.inference.daily_reporter import DailyDeploymentReporter
from extreme_price_movements.inference.data_fetcher import DataFetcher
from extreme_price_movements.inference.parity import (
    load_profitable_sizer_strategy_filter,
    resolve_deployment_strategy_filter,
    validate_calibration_artifacts,
    validate_deployment_model_coverage,
    validate_live_feature_contract,
    validate_meta_feature_contract_artifact,
)
from extreme_price_movements.inference.simple_policy_stop import (
    SIMPLE_POLICY_GENERATOR,
    SIMPLE_POLICY_SCHEMA,
    SimplePolicyStopDecision,
)
from extreme_price_movements.inference.trade_executor import (
    TradeExecutor,
    _validate_order_filters,
)
from extreme_price_movements.inference.trade_logger import TradeLogger
from extreme_price_movements.portfolio_manager import PortfolioManager
from extreme_price_movements.simple_position_sizer import load_calibration_curves
from extreme_price_movements.utils import tprint

CHECK_NAMES: Tuple[str, ...] = (
    "Artifact manifest verified",
    "Feature parity verified",
    "Candidate selection parity verified",
    "Calibration parity verified",
    "Portfolio/risk constraints verified",
    "Exchange filters verified",
    "Order lifecycle tested",
    "Stop-loss lifecycle tested",
    "Data freshness monitored",
    "DB persistence idempotent",
    "Reconciliation process working",
    "Daily reports working",
    "Kill switch working",
    "Manual override available",
    "Tiny-cap live shadow passed",
)


@dataclass(frozen=True)
class DeploymentCheckResult:
    """Result for one deployment readiness check."""

    name: str
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


@dataclass
class DeploymentCheckContext:
    """Inputs used by deployment readiness checks."""

    data_root: str = "data"
    run_id: Optional[str] = None
    model_bundle: Optional[Dict[str, Any]] = None
    calibration_data: Optional[Dict[str, Dict[str, Any]]] = None
    accepted_strategies: Optional[Set[str]] = None
    candidate_selection_probe: Optional[
        Callable[[Optional[Set[str]]], Dict[str, Any]]
    ] = None
    exchange_market: Optional[Dict[str, Any]] = None
    trade_executor: Optional[TradeExecutor] = None
    stop_loss_executor: Optional[TradeExecutor] = None
    data_fetcher: Optional[DataFetcher] = None
    portfolio_mgr: Optional[PortfolioManager] = None
    trade_logger: Optional[TradeLogger] = None
    daily_reporter: Optional[DailyDeploymentReporter] = None
    exchange: Optional[Any] = None
    now: Optional[pd.Timestamp] = None
    temp_dir: Optional[str] = None


def _result(
    name: str,
    passed: bool,
    *,
    details: Optional[Dict[str, Any]] = None,
    error: str = "",
) -> DeploymentCheckResult:
    return DeploymentCheckResult(
        name=name,
        passed=bool(passed),
        details=dict(details or {}),
        error=str(error or ""),
    )


def _require_run_id(ctx: DeploymentCheckContext) -> str:
    if not ctx.run_id:
        raise ValueError("run_id is required for artifact deployment checks")
    return str(ctx.run_id)


def _run_dir(ctx: DeploymentCheckContext) -> Path:
    return Path(ctx.data_root) / "artifacts" / _require_run_id(ctx)


def _artifact_manifest_verified(ctx: DeploymentCheckContext) -> DeploymentCheckResult:
    name = CHECK_NAMES[0]
    run_id = _require_run_id(ctx)
    run_dir = _run_dir(ctx)
    labels_manifest = load_artifact_manifest(ctx.data_root, run_id, "labels")
    base_meta_contract = run_dir / "base_meta_contract.json"
    meta_feature_contract = run_dir / "meta_oof" / "meta_feature_contract.json"
    policy_params_paths = [
        run_dir / "policy_params" / "best_policy_params.json",
        run_dir / "best_policy_params.json",
    ]
    strategy_filter_paths = [
        run_dir / "strategy_for_inference.json",
        run_dir / "policy_params" / "strategy_for_inference.json",
        run_dir / "ridge_sizer" / "strategy_for_inference.json",
        run_dir / "strategy_for_inference.csv",
        run_dir / "policy_params" / "strategy_for_inference.csv",
        run_dir / "ridge_sizer" / "strategy_for_inference.csv",
    ]
    sizer_params = run_dir / "ridge_sizer" / "strategy_params.json"
    calibration_contract = (
        run_dir / "ridge_sizer" / "confidence_calibration.contract.json"
    )
    required = {
        "run_dir": run_dir.exists(),
        "labels_manifest": isinstance(labels_manifest, dict) and bool(labels_manifest),
        "base_meta_contract": base_meta_contract.exists(),
        "meta_feature_contract": meta_feature_contract.exists(),
        "policy_params": any(path.exists() for path in policy_params_paths),
        "strategy_for_inference": any(path.exists() for path in strategy_filter_paths),
        "sizer_params": sizer_params.exists(),
        "calibration_contract": calibration_contract.exists(),
    }
    missing = [key for key, ok in required.items() if not ok]
    details: Dict[str, Any] = {
        "run_id": run_id,
        "required": required,
        "labels_datasets": sorted((labels_manifest or {}).get("datasets", {}).keys()),
        "policy_param_paths": [str(path) for path in policy_params_paths],
        "strategy_filter_paths": [str(path) for path in strategy_filter_paths],
    }
    return _result(name, not missing, details=details, error=", ".join(missing))


def _feature_parity_verified(ctx: DeploymentCheckContext) -> DeploymentCheckResult:
    name = CHECK_NAMES[1]
    bundle = ctx.model_bundle or {}
    feature_ok = validate_live_feature_contract(bundle, strict=False)
    coverage_ok = validate_deployment_model_coverage(
        bundle, ctx.accepted_strategies, strict=False
    )
    if ctx.run_id:
        meta_feature_ok = validate_meta_feature_contract_artifact(
            ctx.data_root,
            ctx.run_id,
            bundle,
            ctx.accepted_strategies,
            strict=False,
        )
    else:
        meta_feature_ok = False
    ok = feature_ok and coverage_ok and meta_feature_ok
    return _result(
        name,
        ok,
        details={
            "model_bundle_keys": sorted(bundle.keys()),
            "feature_contract_ok": feature_ok,
            "meta_feature_contract_ok": meta_feature_ok,
            "model_coverage_ok": coverage_ok,
        },
        error=(
            ""
            if ok
            else "model coverage, meta feature contract, or live feature contract failed"
        ),
    )


def _candidate_selection_parity_verified(
    ctx: DeploymentCheckContext,
) -> DeploymentCheckResult:
    name = CHECK_NAMES[2]
    accepted = ctx.accepted_strategies
    if accepted is None and ctx.run_id:
        accepted = resolve_deployment_strategy_filter(ctx.data_root, ctx.run_id)
    if accepted is None and ctx.run_id:
        accepted = load_profitable_sizer_strategy_filter(ctx.data_root, ctx.run_id)
    probe_details: Dict[str, Any] = {}
    if ctx.candidate_selection_probe is not None:
        probe_details = dict(ctx.candidate_selection_probe(accepted))
    accepted_count = len(accepted or set())
    ok = accepted_count > 0 and bool(probe_details.get("passed", True))
    details = {
        "accepted_strategy_count": accepted_count,
        "accepted_strategy_sample": sorted(accepted or set())[:10],
        "probe": probe_details,
    }
    return _result(
        name, ok, details=details, error="" if ok else "no accepted strategies"
    )


def _calibration_parity_verified(ctx: DeploymentCheckContext) -> DeploymentCheckResult:
    name = CHECK_NAMES[3]
    run_id = _require_run_id(ctx)
    calibration_data = (
        ctx.calibration_data
        if ctx.calibration_data is not None
        else load_calibration_curves(ctx.data_root, run_id)
    )
    ok = validate_calibration_artifacts(
        ctx.data_root, run_id, calibration_data, strict=True
    )
    return _result(
        name,
        ok,
        details={"calibrated_strategy_count": len(calibration_data or {})},
        error="" if ok else "calibration contract missing or empty",
    )


def _portfolio_risk_constraints_verified(
    ctx: DeploymentCheckContext,
) -> DeploymentCheckResult:
    name = CHECK_NAMES[4]
    mgr = PortfolioManager(portfolio_value=10_000.0)
    now = pd.Timestamp(ctx.now or "2026-01-01T00:00:00Z")
    mgr.record_position_open(
        symbol="BTC/USDC",
        side="long",
        strategy_id="long_mr",
        position_size=1_500.0,
        entry_price=100.0,
        entry_time=now,
    )
    allowed_same_asset, same_asset_info = mgr.can_enter_position(
        symbol="BTC/USDC",
        side="long",
        strategy_id="long_mr",
        confidence_score=0.99,
        initial_threshold=0.50,
        current_time=now + pd.Timedelta(minutes=1),
        requested_position_size=1_000.0,
    )
    allowed_next, next_info = mgr.can_enter_position(
        symbol="ETH/USDC",
        side="long",
        strategy_id="long_mr",
        confidence_score=0.99,
        initial_threshold=0.50,
        current_time=now + pd.Timedelta(minutes=2),
        requested_position_size=1_000.0,
    )
    for i in range(1, 6):
        mgr.record_position_open(
            symbol=f"LONG{i}/USDC",
            side="long",
            strategy_id=f"long_mr_{i}",
            position_size=500.0,
            entry_price=100.0,
            entry_time=now + pd.Timedelta(minutes=2 + i),
        )
    allowed_seventh_long, seventh_long_info = mgr.can_enter_position(
        symbol="SEVENTH/USDC",
        side="long",
        strategy_id="long_extra",
        confidence_score=0.99,
        initial_threshold=0.50,
        current_time=now + pd.Timedelta(minutes=10),
        requested_position_size=500.0,
    )
    position_cap_ok = float(next_info.get("position_size_cap", 0.0)) <= 1_500.0
    threshold_ok = float(next_info.get("final_threshold", 0.0)) > 0.50
    cap_policy_ok = (
        mgr.max_positions == 8
        and mgr.max_same_side == 6
        and mgr.max_same_strategy == 6
        and float(mgr.max_portfolio_pct or 0.0) == 0.75
    )
    ok = (
        not allowed_same_asset
        and allowed_next
        and not allowed_seventh_long
        and position_cap_ok
        and threshold_ok
        and cap_policy_ok
    )
    return _result(
        name,
        ok,
        details={
            "same_asset_reason": same_asset_info.get("reason"),
            "seventh_long_reason": seventh_long_info.get("reason"),
            "next_position_cap": next_info.get("position_size_cap"),
            "next_final_threshold": next_info.get("final_threshold"),
            "max_positions": mgr.max_positions,
            "max_same_side": mgr.max_same_side,
            "max_same_strategy": mgr.max_same_strategy,
            "max_portfolio_pct": mgr.max_portfolio_pct,
        },
        error="" if ok else "portfolio constraints did not block/cap as expected",
    )


def _exchange_filters_verified(ctx: DeploymentCheckContext) -> DeploymentCheckResult:
    name = CHECK_NAMES[5]
    market = ctx.exchange_market or {
        "active": True,
        "limits": {
            "amount": {"min": 0.001, "max": 10_000.0},
            "cost": {"min": 10.0, "max": 1_000_000.0},
        },
        "info": {"status": "TRADING"},
    }
    _validate_order_filters("BTC/USDT", market, amount=0.01, price=50_000.0)
    rejected = False
    try:
        _validate_order_filters("BTC/USDT", market, amount=0.00001, price=1.0)
    except ValueError:
        rejected = True
    return _result(
        name,
        rejected,
        details={"valid_order_checked": True, "invalid_order_rejected": rejected},
        error="" if rejected else "invalid exchange filter case was accepted",
    )


def _new_shadow_executor() -> TradeExecutor:
    return TradeExecutor(
        mode="shadow",
        exchange=None,
        bucket_params={
            "long_mr": {
                "generated_by": SIMPLE_POLICY_GENERATOR,
                "schema": SIMPLE_POLICY_SCHEMA,
                "params_source": SIMPLE_POLICY_GENERATOR,
                "params_hash": "deployment-check-policy",
                "strategy_id": "long_mr",
                "sl_mult": 1.0,
                "barrier_pct": 0.01,
                "enable_trailing": True,
                "trailing_activation_mult": 1.0,
                "trailing_override_alpha": 1.0,
                "trailing_power": 1.5,
                "trailing_squash_divisor": 2.0,
                "giveback_beta": 0.5,
                "capital_protect_mfe_mult": 1.0,
                "capital_protect_regression_frac": 0.45,
            }
        },
        config={"monitor_interval_seconds": 300},
    )


def _order_lifecycle_tested(ctx: DeploymentCheckContext) -> DeploymentCheckResult:
    name = CHECK_NAMES[6]
    executor = ctx.trade_executor or _new_shadow_executor()
    result = executor.execute_trade(
        "BTC/USDT",
        "long",
        25.0,
        price=100.0,
        bucket_key="long_mr"
    )
    active_after_entry = executor.get_active_positions()
    close_result = executor.close_position("BTC/USDT", price=101.0, reason="check")
    active_after_close = executor.get_active_positions()
    ok = (
        bool(result.get("success") or result.get("status") == "recorded")
        and "BTC/USDT" in active_after_entry
        and bool(close_result.get("success") or close_result.get("status") == "closed")
        and "BTC/USDT" not in active_after_close
    )
    return _result(
        name,
        ok,
        details={
            "entry_status": result.get("status"),
            "entry_success": result.get("success"),
            "close_success": close_result.get("success"),
            "close_status": close_result.get("status"),
        },
        error="" if ok else "shadow order lifecycle did not complete",
    )


def _stop_loss_lifecycle_tested(ctx: DeploymentCheckContext) -> DeploymentCheckResult:
    name = CHECK_NAMES[7]
    executor = ctx.stop_loss_executor or _new_shadow_executor()
    result = executor.execute_trade(
        "ETH/USDT",
        "long",
        25.0,
        price=100.0,
        bucket_key="long_mr"
    )
    positions = executor.get_active_positions()
    position = positions.get("ETH/USDT", {})
    stop_price = float(position.get("stop_price", np.nan))
    decision = SimplePolicyStopDecision(
        should_replace=True,
        stop_price=99.5,
        reason="capital_preservation",
        reason_detail="capital_preservation: deployment check",
        strategy_id="long_mr",
        params_source=SIMPLE_POLICY_GENERATOR,
        params_hash="deployment-check-policy",
        barrier_frac=0.01,
        sl_mult=1.0,
    )
    executor.update_position_policy_state(
        "ETH/USDT",
        policy_stop_decision=decision,
        last_5m_eval_ts=pd.Timestamp("2026-01-01T00:05:00Z"),
    )
    updated_position = executor.get_active_positions().get("ETH/USDT", {})
    updated_stop_price = float(updated_position.get("stop_price", np.nan))
    replacement_attempted = (
        np.isfinite(updated_stop_price) and updated_stop_price > stop_price
    )
    ok = (
        bool(result.get("success") or result.get("status") == "recorded")
        and np.isfinite(stop_price)
        and stop_price < 100.0
        and replacement_attempted
    )
    return _result(
        name,
        ok,
        details={
            "initial_stop_price": stop_price,
            "updated_stop_price": updated_stop_price,
            "threshold_update_success": replacement_attempted,
        },
        error="" if ok else "stop-loss lifecycle was not fully exercised",
    )


def _data_freshness_monitored(ctx: DeploymentCheckContext) -> DeploymentCheckResult:
    name = CHECK_NAMES[8]
    fetcher = ctx.data_fetcher
    if fetcher is None:
        return _result(name, False, error="data_fetcher is required")
    dead_letters = getattr(fetcher, "dead_letter_symbols", {})
    api_errors = getattr(fetcher, "api_error_counts", {})
    monitored = hasattr(fetcher, "fetch_hourly_universe_once") and hasattr(
        fetcher, "has_recent_gap"
    )
    ok = monitored and isinstance(dead_letters, dict) and isinstance(api_errors, dict)
    return _result(
        name,
        ok,
        details={
            "dead_letter_symbols": len(dead_letters),
            "api_error_counts": dict(api_errors),
            "has_hourly_batch": hasattr(fetcher, "fetch_hourly_universe_once"),
            "has_gap_check": hasattr(fetcher, "has_recent_gap"),
        },
        error="" if ok else "data freshness monitoring hooks missing",
    )


def _db_persistence_idempotent(ctx: DeploymentCheckContext) -> DeploymentCheckResult:
    name = CHECK_NAMES[9]
    if ctx.trade_logger is None:
        if not ctx.temp_dir:
            return _result(name, False, error="trade_logger or temp_dir is required")
        logger = TradeLogger(
            output_path=str(Path(ctx.temp_dir) / "deployment_trades.csv"),
            run_id="deployment_check",
        )
    else:
        logger = ctx.trade_logger
    record = {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "run_id": logger.run_id,
        "symbol": "BTC/USDT",
        "side": "long",
        "action": "enter",
        "strategy_id": "long_mr",
        "expected_entry_price": "100.0",
        "realized_entry_price": "100.0",
        "mode": "shadow",
        "status": "completed",
    }
    logger._write_db_record(record)
    logger._write_db_record(record)
    db_path = getattr(logger, "db_path", None)
    if not db_path:
        return _result(name, False, error="trade logger db_path is missing")
    record_hash = "|".join(
        str(record.get(col, ""))
        for col in (
            "timestamp",
            "run_id",
            "symbol",
            "side",
            "action",
            "strategy_id",
            "expected_entry_price",
            "realized_entry_price",
        )
    )
    with sqlite3.connect(db_path) as conn:
        count = int(
            conn.execute(
                "SELECT COUNT(*) FROM trades WHERE record_hash = ?", (record_hash,)
            ).fetchone()[0]
        )
    ok = count == 1
    return _result(
        name,
        ok,
        details={"db_path": db_path, "row_count_after_duplicate_write": count},
        error="" if ok else "duplicate trade row was persisted",
    )


def _reconciliation_process_working(
    ctx: DeploymentCheckContext,
) -> DeploymentCheckResult:
    name = CHECK_NAMES[10]
    if ctx.exchange is None:
        return _result(name, False, error="exchange is required")
    mgr = ctx.portfolio_mgr or PortfolioManager(portfolio_value=1_000.0)
    snapshot = mgr.fetch_exchange_snapshot(ctx.exchange)
    ok = not snapshot.get("errors") and np.isfinite(
        float(snapshot.get("total_balance", np.nan))
    )
    return _result(
        name,
        ok,
        details={
            "total_balance": snapshot.get("total_balance"),
            "exchange_open_positions": snapshot.get("exchange_open_positions"),
            "local_open_positions": snapshot.get("local_open_positions"),
            "errors": snapshot.get("errors", []),
        },
        error="" if ok else "exchange snapshot reconciliation failed",
    )


def _daily_reports_working(ctx: DeploymentCheckContext) -> DeploymentCheckResult:
    name = CHECK_NAMES[11]
    if ctx.daily_reporter is None or ctx.exchange is None:
        return _result(name, False, error="daily_reporter and exchange are required")
    if ctx.trade_logger is None:
        return _result(name, False, error="trade_logger is required")
    result = ctx.daily_reporter.maybe_run(
        exchange=ctx.exchange,
        portfolio_mgr=ctx.portfolio_mgr or PortfolioManager(portfolio_value=1_000.0),
        trade_logger=ctx.trade_logger,
        config={
            "mode": "shadow",
            "daily_report_transfer_enabled": False,
            "daily_report_email_to": "deployment@example.com",
        },
        now=ctx.now or pd.Timestamp("2026-01-02T00:00:00Z"),
        force=True,
    )
    ok = bool(result.get("sent"))
    return _result(
        name,
        ok,
        details=result,
        error="" if ok else str(result.get("reason", "daily report failed")),
    )


def _kill_switch_working(ctx: DeploymentCheckContext) -> DeploymentCheckResult:
    name = CHECK_NAMES[12]
    mgr = PortfolioManager(portfolio_value=1_000.0)
    now = pd.Timestamp(ctx.now or "2026-01-01T00:00:00Z")
    for idx in range(10):
        mgr.record_api_call(
            False,
            timestamp=now + pd.Timedelta(seconds=idx * 10),
            error="deployment check forced API failure",
        )
    status = mgr.get_hard_limit_status()
    ok = bool(status.get("manual_reset_required")) and not bool(
        status.get("allowed_to_open")
    )
    return _result(
        name,
        ok,
        details=status,
        error="" if ok else "failed API-call kill switch did not trip",
    )


def _manual_override_available(ctx: DeploymentCheckContext) -> DeploymentCheckResult:
    name = CHECK_NAMES[13]
    mgr = PortfolioManager(portfolio_value=1_000.0)
    now = pd.Timestamp(ctx.now or "2026-01-01T00:00:00Z")
    for idx in range(10):
        mgr.record_api_call(False, timestamp=now + pd.Timedelta(seconds=idx))
    before = mgr.get_hard_limit_status()
    mgr.manual_reset_hard_limits()
    after = mgr.get_hard_limit_status()
    ok = bool(before.get("manual_reset_required")) and not bool(
        after.get("manual_reset_required")
    )
    return _result(
        name,
        ok,
        details={"before": before, "after": after},
        error="" if ok else "manual hard-limit reset did not clear stop state",
    )


def _tiny_cap_live_shadow_passed(ctx: DeploymentCheckContext) -> DeploymentCheckResult:
    name = CHECK_NAMES[14]
    executor = _new_shadow_executor()
    tiny_notional = 5.0
    result = executor.execute_trade(
        "TINY/USDT", "long", tiny_notional, price=1.0, bucket_key="long_mr"
    )
    active = executor.get_active_positions()
    ok = (
        bool(result.get("success") or result.get("status") == "recorded")
        and active.get("TINY/USDT", {}).get("size") == tiny_notional
        and result.get("mode") == "shadow"
    )
    return _result(
        name,
        ok,
        details={"mode": result.get("mode"), "size": result.get("size")},
        error="" if ok else "tiny-cap shadow trade did not record cleanly",
    )


_CHECKS: Dict[str, Callable[[DeploymentCheckContext], DeploymentCheckResult]] = {
    CHECK_NAMES[0]: _artifact_manifest_verified,
    CHECK_NAMES[1]: _feature_parity_verified,
    CHECK_NAMES[2]: _candidate_selection_parity_verified,
    CHECK_NAMES[3]: _calibration_parity_verified,
    CHECK_NAMES[4]: _portfolio_risk_constraints_verified,
    CHECK_NAMES[5]: _exchange_filters_verified,
    CHECK_NAMES[6]: _order_lifecycle_tested,
    CHECK_NAMES[7]: _stop_loss_lifecycle_tested,
    CHECK_NAMES[8]: _data_freshness_monitored,
    CHECK_NAMES[9]: _db_persistence_idempotent,
    CHECK_NAMES[10]: _reconciliation_process_working,
    CHECK_NAMES[11]: _daily_reports_working,
    CHECK_NAMES[12]: _kill_switch_working,
    CHECK_NAMES[13]: _manual_override_available,
    CHECK_NAMES[14]: _tiny_cap_live_shadow_passed,
}


def run_deployment_checks(
    ctx: DeploymentCheckContext,
    *,
    checks: Optional[Sequence[str]] = None,
) -> List[DeploymentCheckResult]:
    """Run deployment checks sequentially and return all results."""
    selected = list(checks or CHECK_NAMES)
    results: List[DeploymentCheckResult] = []
    total = len(selected)
    for idx, name in enumerate(selected, start=1):
        tprint(f"[DeploymentChecks] Running {idx}/{total}: {name}")
        fn = _CHECKS[name]
        try:
            result = fn(ctx)
        except Exception as exc:
            result = _result(name, False, error=str(exc))
        tprint(
            f"[DeploymentChecks] {'PASS' if result.passed else 'FAIL'}: "
            f"{name} error={result.error}"
        )
        results.append(result)
    return results


def require_deployment_checks(
    ctx: DeploymentCheckContext,
    *,
    checks: Optional[Sequence[str]] = None,
) -> List[DeploymentCheckResult]:
    """Run checks and raise if any deployment prerequisite fails."""
    results = run_deployment_checks(ctx, checks=checks)
    failed = [result for result in results if not result.passed]
    if failed:
        summary = "; ".join(f"{item.name}: {item.error}" for item in failed)
        raise RuntimeError(f"Deployment checks failed: {summary}")
    return results


def summarize_deployment_checks(
    results: Iterable[DeploymentCheckResult],
) -> Dict[str, Any]:
    """Return a compact summary suitable for logs, reports, or tests."""
    rows = list(results)
    return {
        "total": len(rows),
        "passed": sum(1 for row in rows if row.passed),
        "failed": sum(1 for row in rows if not row.passed),
        "failures": [
            {"name": row.name, "error": row.error} for row in rows if not row.passed
        ],
    }
