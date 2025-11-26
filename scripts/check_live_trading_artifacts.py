#!/usr/bin/env python3
"""Live Trading Artifact Contract Checker.

This CLI validates that all required artifacts for live trading are
present and loadable for a given (symbol, exchange, timeframe, direction).

It checks that:
- Regime base and ensemble models can be loaded via UnifiedModelLoader.
- Analyst base and ensemble models can be loaded.
- Tactician base and ensemble models can be loaded.
- Optimized parameters from final_parameters_optimization are available.
- Specialist regime outputs are available via load_live_regime_outputs.

Exit code is non-zero if any *required* component is missing.

Usage (from project root):

  python scripts/check_live_trading_artifacts.py \
      --symbol ETHUSDT --exchange binance --timeframe 15m --direction long

You can also use this in CI to gate live deployments.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.tprint import (
    tprint,
    tprint_info,
    tprint_warning,
    tprint_error,
    tprint_success,
)
from src.utils.logger import system_logger
from src.trading.integration.unified_model_loader import get_unified_model_loader
from src.trading.integration.optimized_parameters_integration import (
    get_optimized_params_integration,
)
from src.trading.integration.live_regime_outputs import load_live_regime_outputs


logger = system_logger.getChild("LiveArtifactContract")


async def check_live_artifacts(
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str,
    strict: bool = True,
) -> Dict[str, Any]:
    """Run all artifact checks for the given trading context.

    Returns a structured dict with per-check booleans and messages.
    """

    loader = get_unified_model_loader()
    opt_integration = get_optimized_params_integration()

    results: Dict[str, Any] = {"context": {
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": timeframe,
        "direction": direction,
    }}

    # 1. Regime models
    regime_base_ok = False
    regime_ensemble_ok = False
    try:
        base_models = await loader.load_regime_base_models(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            direction=direction,
        )
        regime_base_ok = bool(base_models)
        if regime_base_ok:
            tprint_success(f"✅ Regime base models loaded ({len(base_models)} models)")
        else:
            tprint_warning("⚠️ No regime base models found")
    except Exception as exc:  # pragma: no cover - defensive
        tprint_error(f"❌ Failed to load regime base models: {exc}")
        results.setdefault("errors", {})["regime_base"] = str(exc)

    try:
        ensemble_model = await loader.load_regime_ensemble_model(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            direction=direction,
        )
        regime_ensemble_ok = ensemble_model is not None
        if regime_ensemble_ok:
            tprint_success("✅ Regime ensemble model loaded")
        else:
            tprint_warning("⚠️ No regime ensemble model found")
    except Exception as exc:  # pragma: no cover - defensive
        tprint_error(f"❌ Failed to load regime ensemble model: {exc}")
        results.setdefault("errors", {})["regime_ensemble"] = str(exc)

    results["regime_models"] = {
        "base_ok": regime_base_ok,
        "ensemble_ok": regime_ensemble_ok,
    }

    # 2. Analyst models
    analyst_base_ok = False
    analyst_ensemble_ok = False
    try:
        analyst_base = await loader.load_analyst_base_models(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            direction=direction,
        )
        analyst_base_ok = bool(analyst_base)
        if analyst_base_ok:
            tprint_success(f"✅ Analyst base models loaded ({len(analyst_base)} models)")
        else:
            tprint_warning("⚠️ No analyst base models found")
    except Exception as exc:
        tprint_error(f"❌ Failed to load analyst base models: {exc}")
        results.setdefault("errors", {})["analyst_base"] = str(exc)

    try:
        analyst_ensemble = await loader.load_analyst_ensemble_model(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            direction=direction,
        )
        analyst_ensemble_ok = analyst_ensemble is not None
        if analyst_ensemble_ok:
            tprint_success("✅ Analyst ensemble model loaded")
        else:
            tprint_warning("⚠️ No analyst ensemble model found")
    except Exception as exc:
        tprint_error(f"❌ Failed to load analyst ensemble model: {exc}")
        results.setdefault("errors", {})["analyst_ensemble"] = str(exc)

    results["analyst_models"] = {
        "base_ok": analyst_base_ok,
        "ensemble_ok": analyst_ensemble_ok,
    }

    # 3. Tactician models
    tactician_base_ok = False
    tactician_ensemble_ok = False
    try:
        tactician_base = await loader.load_tactician_base_models(
            symbol=symbol,
            exchange=exchange,
            timeframe="5m",  # Tactician is typically 5m
            direction=direction,
        )
        tactician_base_ok = bool(tactician_base)
        if tactician_base_ok:
            tprint_success(f"✅ Tactician base models loaded ({len(tactician_base)} models)")
        else:
            tprint_warning("⚠️ No tactician base models found")
    except Exception as exc:
        tprint_error(f"❌ Failed to load tactician base models: {exc}")
        results.setdefault("errors", {})["tactician_base"] = str(exc)

    try:
        tactician_ensemble = await loader.load_tactician_ensemble_model(
            symbol=symbol,
            exchange=exchange,
            timeframe="5m",
            direction=direction,
        )
        tactician_ensemble_ok = tactician_ensemble is not None
        if tactician_ensemble_ok:
            tprint_success("✅ Tactician ensemble model loaded")
        else:
            tprint_warning("⚠️ No tactician ensemble model found")
    except Exception as exc:
        tprint_error(f"❌ Failed to load tactician ensemble model: {exc}")
        results.setdefault("errors", {})["tactician_ensemble"] = str(exc)

    results["tactician_models"] = {
        "base_ok": tactician_base_ok,
        "ensemble_ok": tactician_ensemble_ok,
    }

    # 4. Optimized parameters (final_parameters_optimization)
    optimized_params_ok = False
    try:
        opt_params = await opt_integration.get_optimized_parameters(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            direction=direction,
        )
        optimized_params_ok = bool(opt_params)
        if optimized_params_ok:
            tprint_success(
                f"✅ Optimized parameters loaded ({len(opt_params)} keys) for {symbol} {timeframe} {direction}"
            )
        else:
            tprint_warning("⚠️ No optimized parameters found; live trading will use defaults")
    except Exception as exc:
        tprint_error(f"❌ Failed to load optimized parameters: {exc}")
        results.setdefault("errors", {})["optimized_parameters"] = str(exc)

    results["optimized_parameters"] = {
        "ok": optimized_params_ok,
    }

    # 5. Specialist regime outputs (live regime stack)
    regime_outputs_ok = False
    try:
        # Build a small synthetic index on the requested timeframe to probe
        # the live regime outputs contract without needing real data.
        now = pd.Timestamp.utcnow().floor("T")
        # Use 50 bars back to ensure enough history for most artifacts
        periods = 50
        if timeframe.endswith("m"):
            freq = f"{timeframe[:-1]}T"
        else:
            freq = timeframe
        idx = pd.date_range(end=now, periods=periods, freq=freq)

        regime_df = load_live_regime_outputs(
            symbol=symbol,
            exchange=exchange,
            direction=direction,
            base_timeframe=timeframe,
            regime_timeframe=None,
            target_index=idx,
            config_overrides=None,
            artifact_router=None,
            strict=False,
        )
        if regime_df is not None and not regime_df.empty and regime_df.shape[1] > 0:
            regime_outputs_ok = True
            tprint_success(
                f"✅ Live specialist regime outputs available ({regime_df.shape[1]} columns)"
            )
        else:
            tprint_warning("⚠️ No live specialist regime outputs available for this context")
    except Exception as exc:
        tprint_error(f"❌ Failed to load live regime outputs: {exc}")
        results.setdefault("errors", {})["live_regime_outputs"] = str(exc)

    results["live_regime_outputs"] = {
        "ok": regime_outputs_ok,
    }

    # Overall contract status
    required_checks = {
        "regime_base": regime_base_ok,
        "regime_ensemble": regime_ensemble_ok,
        "analyst_base": analyst_base_ok,
        "analyst_ensemble": analyst_ensemble_ok,
        "tactician_base": tactician_base_ok,
        "tactician_ensemble": tactician_ensemble_ok,
        "optimized_parameters": optimized_params_ok,
        "live_regime_outputs": regime_outputs_ok,
    }

    results["required_checks"] = required_checks
    results["all_required_ok"] = all(required_checks.values())

    if strict and not results["all_required_ok"]:
        missing = [k for k, ok in required_checks.items() if not ok]
        tprint_error(
            "❌ Live trading artifact contract FAILED: missing/invalid components: "
            + ", ".join(missing)
        )
    elif results["all_required_ok"]:
        tprint_success("✅ Live trading artifact contract satisfied for this context")
    else:
        tprint_warning("⚠️ Live trading artifact contract partially satisfied (non-strict mode)")

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check live trading artifact contract")
    parser.add_argument("--symbol", required=True, help="Trading symbol, e.g. ETHUSDT")
    parser.add_argument("--exchange", default="binance", help="Exchange name")
    parser.add_argument("--timeframe", default="15m", help="Base timeframe (e.g. 15m, 1h)")
    parser.add_argument("--direction", default="long", help="Trading direction (long/short)")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat any missing component as a hard failure (exit code != 0)",
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Print full JSON result payload to stdout",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    tprint_info(
        f"🔍 Checking live artifact contract for {args.symbol} on {args.exchange} "
        f"[{args.timeframe}] {args.direction} (strict={args.strict})"
    )

    results = asyncio.run(
        check_live_artifacts(
            symbol=args.symbol,
            exchange=args.exchange,
            timeframe=args.timeframe,
            direction=args.direction,
            strict=args.strict,
        )
    )

    if args.json_output:
        print(json.dumps(results, indent=2, default=str))

    if args.strict and not results.get("all_required_ok", False):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
