#!/usr/bin/env python3
"""
Bridge script: Convert weighted_meta_labeling_step outputs to meta_gated_backtest artifacts.

This script reads the outputs from weighted_meta_labeling_step and generates:
1. labeled_data artifact (with meta_probability and realized_return)
2. meta_gating_config.json
3. Isotonic regressor artifact

Usage:
    python scripts/build_weighted_meta_backtest_artifacts.py \\
        --symbol ETHUSDT --exchange binance --timeframe 15m --direction long \\
        [--dry-run]
"""

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error
from src.utils.pipeline_standards import PipelineStandards
from src.training.steps.base_step import BaseStep


class _ArtifactSaver(BaseStep):
    """Minimal BaseStep subclass to use _save_artifact."""

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover - helper only
        raise NotImplementedError("Utility helper; execute not implemented.")


def find_latest_weighted_labeled_data(
    symbol: str,
    timeframe: str,
    outcomes_dir: Path = Path("outcomes"),
) -> Optional[Path]:
    """Find the latest weighted labeled data CSV."""
    pattern = f"weighted_labeled_data_{symbol}_{timeframe}_*.csv"
    candidates = sorted(outcomes_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_weighted_hpo_best_params(
    symbol: str,
    timeframe: str,
    direction: str,
) -> Optional[Dict[str, Any]]:
    """Load weighted HPO best params from outcomes or standardized reports."""
    candidate_dirs = []
    try:
        base_dir = PipelineStandards.build_path('reports', exchange='binance', asset=symbol)
        candidate_dirs.append(Path(base_dir) / "post_hpo_evaluation")
    except Exception:
        pass
    candidate_dirs.append(Path("outcomes"))
    
    patterns = [
        f"meta_labeling_hpo_best_params_{symbol}_{timeframe}_{direction}_*.json",
        f"meta_labeling_hpo_best_params_{symbol}_{timeframe}_*.json",
    ]
    
    json_files = []
    for d in candidate_dirs:
        if not d.exists():
            continue
        for pat in patterns:
            json_files.extend(d.glob(pat))
    
    json_files = sorted(json_files, key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not json_files:
        stage_report_pattern = f"hpo_stage_report_layer3_model_{symbol}_*_{timeframe}_{direction}_*.json"
        stage_reports = sorted(Path("outcomes").glob(stage_report_pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if not stage_reports:
            return None

        latest_report = stage_reports[0]
        try:
            with open(latest_report, "r") as f:
                layer3_report = json.load(f)
        except Exception as e:
            tprint_warning(f"⚠️ Failed to load Layer 3 stage report from {latest_report}: {e}")
            return None

        run_timestamp = str(layer3_report.get("run_timestamp", ""))
        symbol_r = str(layer3_report.get("symbol", symbol))
        exchange_r = str(layer3_report.get("exchange", "binance"))
        timeframe_r = str(layer3_report.get("timeframe", timeframe))
        direction_r = str(layer3_report.get("direction", direction))

        layer2_report = None
        try:
            l2_pat = f"hpo_stage_report_layer2_trading_{symbol_r}_{exchange_r}_{timeframe_r}_{direction_r}_{run_timestamp}.json"
            l2_candidates = sorted(Path("outcomes").glob(l2_pat), key=lambda p: p.stat().st_mtime, reverse=True)
            if l2_candidates:
                with open(l2_candidates[0], "r") as f:
                    layer2_report = json.load(f)
        except Exception:
            layer2_report = None

        merged_best: Dict[str, Any] = {}
        try:
            if isinstance(layer2_report, dict):
                l2_best = layer2_report.get("best_params", {})
                if isinstance(l2_best, dict):
                    merged_best.update(l2_best)
        except Exception:
            pass

        try:
            l3_best = layer3_report.get("best_params", {})
            if isinstance(l3_best, dict):
                merged_best.update(l3_best)
        except Exception:
            pass

        payload = {
            "best_params": merged_best,
            "layer2_best_params": layer2_report.get("best_params", {}) if isinstance(layer2_report, dict) else {},
            "layer3_best_params": layer3_report.get("best_params", {}) if isinstance(layer3_report, dict) else {},
            "run_timestamp": run_timestamp,
            "source": {
                "layer3_report": str(latest_report),
            },
        }
        try:
            payload["source"]["layer2_report"] = str(l2_candidates[0]) if 'l2_candidates' in locals() and l2_candidates else None
        except Exception:
            pass

        return payload
    
    latest_json = json_files[0]
    try:
        with open(latest_json, "r") as f:
            return json.load(f)
    except Exception as e:
        tprint_warning(f"⚠️ Failed to load HPO params from {latest_json}: {e}")
        return None


def fit_isotonic_regressor(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> IsotonicRegression:
    """Fit isotonic regressor from probabilities and binary labels."""
    # Filter to valid pairs
    valid_mask = np.isfinite(probabilities) & np.isfinite(labels) & (labels >= 0) & (labels <= 1)
    
    if valid_mask.sum() < 50:
        raise ValueError(f"Insufficient valid samples for isotonic regression: {valid_mask.sum()}")
    
    p_clean = probabilities[valid_mask]
    y_clean = labels[valid_mask].astype(float)
    
    iso_regressor = IsotonicRegression(out_of_bounds='clip')
    iso_regressor.fit(p_clean, y_clean)
    
    return iso_regressor


def build_meta_gating_config(
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str,
    iso_regressor_path: str,
    hpo_params: Optional[Dict[str, Any]] = None,
    transaction_cost: float = 0.003,
    profit_threshold: Optional[float] = None,
    stop_threshold: Optional[float] = None,
    horizon: Optional[int] = None,
    selection_mode: str = "threshold",
    top_quantile: Optional[float] = None,
) -> Dict[str, Any]:
    """Build meta_gating_config.json structure."""
    # Extract thresholds from HPO params if available
    prob_threshold = 0.6
    er_threshold = 0.0
    use_expected_return = False
    
    if hpo_params:
        best_params = hpo_params.get("best_params", {})
        if "prob_threshold" in best_params:
            prob_threshold = float(best_params["prob_threshold"])
        if "er_threshold" in best_params:
            er_threshold = float(best_params["er_threshold"])
            use_expected_return = er_threshold > 0
        
        # Also check Layer 2 params
        layer2_params = hpo_params.get("layer2_best_params", {})
        if "prob_threshold" in layer2_params:
            prob_threshold = float(layer2_params["prob_threshold"])
        if "er_threshold" in layer2_params:
            er_threshold = float(layer2_params["er_threshold"])
            use_expected_return = er_threshold > 0
    
    config = {
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": timeframe,
        "direction": direction,
        "model_family": "analyst_meta",
        "meta_gating": {
            "version": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
            "transaction_cost": float(transaction_cost),
            "entry": {
                "selection_mode": str(selection_mode),
                "top_quantile": float(top_quantile) if top_quantile is not None else None,
                "prob_threshold": float(prob_threshold),
                "use_expected_return": bool(use_expected_return),
                "expected_return_threshold": float(er_threshold),
                "expected_return_unit": "fraction",
                "min_trades": 0,  # Will be updated after backtest
            },
            "calibration": {
                "iso_regressor_artifact": iso_regressor_path,
                "fitted_on": "weighted_oof",
            },
        },
    }

    try:
        if hpo_params:
            config["weighted_hpo"] = {
                "run_timestamp": hpo_params.get("run_timestamp"),
                "best_params": hpo_params.get("best_params", {}),
                "layer2_best_params": hpo_params.get("layer2_best_params", {}),
                "layer3_best_params": hpo_params.get("layer3_best_params", {}),
                "source": hpo_params.get("source", {}),
            }
    except Exception:
        pass
    
    # Add triple_barrier config if available
    if profit_threshold is not None or stop_threshold is not None or horizon is not None:
        config["meta_gating"]["triple_barrier"] = {}
        if profit_threshold is not None:
            config["meta_gating"]["triple_barrier"]["profit_threshold"] = float(profit_threshold)
        if stop_threshold is not None:
            config["meta_gating"]["triple_barrier"]["stop_threshold"] = float(stop_threshold)
        if horizon is not None:
            config["meta_gating"]["triple_barrier"]["horizon_bars"] = int(horizon)
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Convert weighted_meta_labeling outputs to meta_gated_backtest artifacts"
    )
    parser.add_argument("--symbol", type=str, required=True, help="Trading symbol (e.g., ETHUSDT)")
    parser.add_argument("--exchange", type=str, default="binance", help="Exchange name")
    parser.add_argument("--timeframe", type=str, required=True, help="Timeframe (e.g., 15m)")
    parser.add_argument("--direction", type=str, default="long", choices=["long", "short"], help="Trading direction")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode (don't write files)")
    parser.add_argument("--outcomes-dir", type=str, default="outcomes", help="Outcomes directory")
    parser.add_argument("--transaction-cost", type=float, default=0.003, help="Transaction cost")
    
    args = parser.parse_args()
    
    tprint_info("=" * 70)
    tprint_info("🔧 Building Weighted Meta-Labeling Backtest Artifacts")
    tprint_info("=" * 70)
    tprint_info(f"   Symbol: {args.symbol}")
    tprint_info(f"   Exchange: {args.exchange}")
    tprint_info(f"   Timeframe: {args.timeframe}")
    tprint_info(f"   Direction: {args.direction}")
    tprint_info(f"   Dry-run: {args.dry_run}")
    
    outcomes_dir = Path(args.outcomes_dir)
    
    # ------------------------------------------------------------------
    # 1. Load weighted labeled data CSV
    # ------------------------------------------------------------------
    tprint_info("\n[1/4] Loading weighted labeled data...")
    csv_path = find_latest_weighted_labeled_data(args.symbol, args.timeframe, outcomes_dir)
    
    if csv_path is None:
        tprint_error(f"❌ No weighted labeled data found for {args.symbol} {args.timeframe}")
        tprint_info(f"   Searched pattern: weighted_labeled_data_{args.symbol}_{args.timeframe}_*.csv")
        return 1
    
    tprint_success(f"   ✅ Found: {csv_path.name}")
    
    try:
        labeled_data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        tprint_info(f"   Loaded {len(labeled_data)} rows")
    except Exception as e:
        tprint_error(f"❌ Failed to load CSV: {e}")
        return 1
    
    # Validate required columns
    required_cols = ["meta_probability", "realized_return", "binary_label"]
    missing_cols = [col for col in required_cols if col not in labeled_data.columns]
    if missing_cols:
        tprint_error(f"❌ Missing required columns: {missing_cols}")
        return 1
    
    # ------------------------------------------------------------------
    # 2. Save labeled_data artifact
    # ------------------------------------------------------------------
    tprint_info("\n[2/4] Saving labeled_data artifact...")
    
    artifact_name = f"labeled_data_{args.symbol}_{args.exchange}_{args.timeframe}_{args.direction}"
    
    if args.dry_run:
        tprint_info(f"   [DRY-RUN] Would save artifact: {artifact_name}")
    else:
        try:
            # Create a minimal step instance to use artifact saving
            step = _ArtifactSaver(step_name="build_weighted_meta_backtest_artifacts")
            step.set_context(
                symbol=args.symbol,
                exchange=args.exchange,
                timeframe=args.timeframe,
                direction=args.direction,
                model="analyst",
                execution_mode="full",
            )
            
            artifact_path = step._save_artifact(
                data=labeled_data,
                artifact_name=artifact_name,
                artifact_type="data",
                compression="auto",
                data_category="features",
                metadata={
                    'symbol': args.symbol,
                    'exchange': args.exchange,
                    'timeframe': args.timeframe,
                    'direction': args.direction,
                    'source': 'weighted_meta_labeling',
                    'created_at': datetime.utcnow().isoformat(),
                }
            )
            tprint_success(f"   ✅ Saved artifact: {artifact_name}")
            tprint_info(f"   Path: {artifact_path}")
        except Exception as e:
            tprint_error(f"❌ Failed to save artifact: {e}")
            import traceback
            tprint_error(traceback.format_exc())
            return 1
    
    # ------------------------------------------------------------------
    # 3. Fit and save isotonic regressor
    # ------------------------------------------------------------------
    tprint_info("\n[3/4] Fitting isotonic regressor...")
    
    # Extract valid pairs
    valid_mask = (
        labeled_data["meta_probability"].notna() &
        labeled_data["binary_label"].notna()
    )
    
    if valid_mask.sum() < 50:
        tprint_error(f"❌ Insufficient valid samples: {valid_mask.sum()}")
        return 1
    
    probabilities = labeled_data.loc[valid_mask, "meta_probability"].values
    labels = labeled_data.loc[valid_mask, "binary_label"].values
    
    try:
        iso_regressor = fit_isotonic_regressor(probabilities, labels)
        tprint_success(f"   ✅ Fitted isotonic regressor on {valid_mask.sum()} samples")
    except Exception as e:
        tprint_error(f"❌ Failed to fit isotonic regressor: {e}")
        return 1
    
    # Save isotonic regressor
    va_dir = Path("versioned_artifacts") / f"{args.symbol}_{args.exchange}_{args.timeframe}_{args.direction}_analyst"
    va_dir.mkdir(parents=True, exist_ok=True)
    
    artifacts_dir = va_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    iso_filename = "iso_regressor_analyst_meta.pkl"
    iso_path = artifacts_dir / iso_filename
    iso_rel_path = f"artifacts/{iso_filename}"
    
    if args.dry_run:
        tprint_info(f"   [DRY-RUN] Would save isotonic regressor to: {iso_path}")
    else:
        try:
            with open(iso_path, "wb") as f:
                pickle.dump(iso_regressor, f)
            tprint_success(f"   ✅ Saved isotonic regressor: {iso_rel_path}")
        except Exception as e:
            tprint_error(f"❌ Failed to save isotonic regressor: {e}")
            return 1
    
    # ------------------------------------------------------------------
    # 4. Load HPO params and build gating config
    # ------------------------------------------------------------------
    tprint_info("\n[4/4] Building meta_gating_config.json...")
    
    hpo_params = load_weighted_hpo_best_params(args.symbol, args.timeframe, args.direction)
    if hpo_params:
        tprint_success(f"   ✅ Loaded HPO best params")
    else:
        tprint_warning("   ⚠️ No HPO params found, using defaults")
    
    # Extract triple-barrier params from labeled data metadata if available
    profit_threshold = None
    stop_threshold = None
    horizon = None
    
    # Try to infer from config or use defaults
    if "profit_threshold" in labeled_data.attrs:
        profit_threshold = labeled_data.attrs["profit_threshold"]
    if "stop_threshold" in labeled_data.attrs:
        stop_threshold = labeled_data.attrs["stop_threshold"]
    if "horizon" in labeled_data.attrs:
        horizon = labeled_data.attrs["horizon"]
    
    gating_config = build_meta_gating_config(
        symbol=args.symbol,
        exchange=args.exchange,
        timeframe=args.timeframe,
        direction=args.direction,
        iso_regressor_path=iso_rel_path,
        hpo_params=hpo_params,
        transaction_cost=args.transaction_cost,
        profit_threshold=profit_threshold,
        stop_threshold=stop_threshold,
        horizon=horizon,
        selection_mode="top_quantile",
        top_quantile=None,
    )

    # Compute an initial top-quantile equivalent to the old prob_threshold gate coverage.
    # This makes gating scale-free (rank-based) while keeping a comparable trade frequency.
    try:
        prob_thr_ref = float(gating_config.get("meta_gating", {}).get("entry", {}).get("prob_threshold", 0.6))
        p = labeled_data["meta_probability"].astype(float)
        m = np.isfinite(p.values)
        if int(np.sum(m)) > 0:
            top_q = float(np.mean(p.values[m] >= float(prob_thr_ref)))
            top_q = float(np.clip(top_q, 0.0, 0.999999))
            gating_config["meta_gating"]["entry"]["top_quantile"] = float(top_q)
            tprint_info(
                f"   ↪ Rank-gating: set top_quantile={top_q:.4f} (equivalent coverage to prob_threshold={prob_thr_ref:.3f})"
            )
        else:
            tprint_warning("   ⚠️ Rank-gating: no finite meta_probability values; leaving top_quantile unset")
    except Exception as e:
        tprint_warning(f"   ⚠️ Rank-gating: failed to compute top_quantile from labeled_data: {e}")
    
    gating_path = va_dir / "meta_gating_config.json"

    try:
        if not args.dry_run and hpo_params:
            run_ts = str(hpo_params.get("run_timestamp") or datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
            compat_name = f"meta_labeling_hpo_best_params_{args.symbol}_{args.exchange}_{args.timeframe}_{args.direction}_{run_ts}.json"
            compat_path = Path("outcomes") / compat_name
            compat_payload = {
                "best_params": hpo_params.get("best_params", {}),
                "best_score": hpo_params.get("best_score", hpo_params.get("best_value", 0.0)),
                "best_edge": hpo_params.get("best_edge", 0.0),
                "layer2_best_params": hpo_params.get("layer2_best_params", {}),
                "layer3_best_params": hpo_params.get("layer3_best_params", {}),
                "run_timestamp": run_ts,
                "source": hpo_params.get("source", {}),
            }
            compat_path.parent.mkdir(parents=True, exist_ok=True)
            with open(compat_path, "w") as f:
                json.dump(compat_payload, f, indent=2, default=str)
            tprint_success(f"   ✅ Saved HPO params compatibility file")
            tprint_info(f"   Path: {compat_path}")
    except Exception as e:
        tprint_warning(f"   ⚠️ Failed to write HPO compatibility file: {e}")
    
    if args.dry_run:
        tprint_info(f"   [DRY-RUN] Would save gating config to: {gating_path}")
        tprint_info(f"   Config preview:")
        tprint_info(json.dumps(gating_config, indent=2))
    else:
        try:
            with open(gating_path, "w") as f:
                json.dump(gating_config, f, indent=2)
            tprint_success(f"   ✅ Saved meta_gating_config.json")
            tprint_info(f"   Path: {gating_path}")
        except Exception as e:
            tprint_error(f"❌ Failed to save gating config: {e}")
            return 1
    
    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    tprint_info("\n" + "=" * 70)
    tprint_success("✅ Conversion complete!")
    tprint_info("=" * 70)
    tprint_info(f"   Artifact: {artifact_name}")
    tprint_info(f"   Isotonic regressor: {iso_rel_path}")
    tprint_info(f"   Gating config: {gating_path}")
    tprint_info("\n   Next step: Run meta_gated_backtest")
    tprint_info("   python3 ares_launcher.py --step meta_gated_backtest \\")
    tprint_info(f"       --symbol {args.symbol} --exchange {args.exchange} \\")
    tprint_info(f"       --timeframe {args.timeframe} --direction {args.direction} \\")
    tprint_info("       --execution-mode full")
    
    return 0


if __name__ == "__main__":
    exit(main())








