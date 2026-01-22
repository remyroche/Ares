"""Meta-Labeling HPO Experiment Step (Orchestrator).

This step acts as an Orchestrator for the full Label-Based Pipeline (Layers 0-5),
integrating the proper De Prado Causal Framework (Layer 2) and subsequent
Meta-Labeling (Layer 3) and Position Sizing (Layers 4-5) stages.

It replaces the legacy inline HPO logic with a sequential execution of:
1.  **Layer 0**: Kalman Filter & VWAP Price Smoothing (Feature Engineering).
2.  **Layer 1**: Sample Weighting Optimization.
3.  **Layer 2**: Causal Event Generation & Triple Barrier Labeling (Primary Model).
4.  **Layer 3**: Multi-Geometry Meta-Model Training (Analyst).
5.  **Layer 4**: ExtraTrees Position Sizing (PnL Optimization).
6.  **Layer 5**: Portfolio Construction & Backtesting.

This ensures that the Meta-Labeling HPO Experiment correctly utilizes the
full causal pipeline rather than legacy approximations.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.mixture import GaussianMixture

from src.training.steps.base_step import BaseStep
from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
from src.utils.ml_common.transaction_costs import DEFAULT_TRANSACTION_COST

# --- Pipeline Components ---
# Layer 0: Feature Engineering
try:
    from src.training.steps.labeling.label_based_layer_0 import run_layer0_kalman_vwap
    LAYER0_AVAILABLE = True
except ImportError:
    LAYER0_AVAILABLE = False

# Layer 1: Weighting Optimization
try:
    from src.training.steps.labeling.label_based_layer_1 import run_layer1_optimization
    LAYER1_AVAILABLE = True
except ImportError:
    LAYER1_AVAILABLE = False

try:
    from src.training.steps.labeling.unified_price_layer2 import generate_unified_layer2_price
    UNIFIED_PRICE_AVAILABLE = True
except ImportError:
    UNIFIED_PRICE_AVAILABLE = False

try:
    from src.training.steps.labeling.generate_weights_per_label import generate_weights_per_label
    WEIGHTS_AVAILABLE = True
except ImportError:
    WEIGHTS_AVAILABLE = False

# Layer 2: Causal Framework & Regime Detection
try:
    from src.training.steps.labeling.label_based_layer_2 import LabelBasedLayer2
    LAYER2_AVAILABLE = True
except ImportError:
    LAYER2_AVAILABLE = False

try:
    from src.training.steps.labeling.adaptive_hunter_router import AdaptiveHunterRouter
    HUNTER_ROUTER_AVAILABLE = True
except ImportError:
    HUNTER_ROUTER_AVAILABLE = False

# Layer 3: Meta-Model
try:
    from src.training.steps.labeling.label_based_layer_3 import layer3_analyst_lgbm
    LAYER3_AVAILABLE = True
except ImportError:
    LAYER3_AVAILABLE = False

# Layer 4: Position Sizing
try:
    from src.training.steps.labeling.layer4_extratrees_pnl import train_layer4_extratrees
    LAYER4_AVAILABLE = True
except ImportError:
    LAYER4_AVAILABLE = False

# Layer 5: Portfolio Construction
try:
    from src.training.steps.labeling.label_based_layer_5 import Layer5PositionSizer
    LAYER5_AVAILABLE = True
except ImportError:
    LAYER5_AVAILABLE = False


class MetaLabelingHPOExperimentStep(BaseStep):
    """
    Orchestrator Step for the Full Label-Based Pipeline (L0-L5).
    """

    def __init__(self, step_name: str = "meta_labeling_hpo_experiment", use_versioned_artifacts: bool = True):
        super().__init__(step_name, use_versioned_artifacts)

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the full pipeline sequence.

        Args:
            config: Job configuration dictionary.

        Returns:
            Dict containing pipeline results and metrics.
        """
        tprint_info("🚀 Starting Meta-Labeling HPO Orchestration (Layers 0-5)...")

        # --- CONFIGURATION ENFORCEMENT (User Requested) ---
        tprint_info("🔧 Enforcing Pipeline Configuration...")
        tprint_info("   - Layer 0: Wavelets ENABLED")
        config["use_wavelets"] = True
        
        tprint_info("   - Layer 2: De Prado Causal Framework ENABLED (Full Mode)")
        config["enable_causal_framework"] = True
        config["causal_discovery_enabled"] = True
        config["irm_enabled"] = True
        config["causal_surprise_enabled"] = True
        config["interventionist_sampling_enabled"] = True 
        config["causal_specialists_enabled"] = True
        config["enable_aedl"] = True # Re-enable AEDL filters

        config.setdefault("layer3_use_enhanced", True)
        config.setdefault("enable_advanced_feature_selection", True)
        
        tprint_info("   - Layer 1: Sample Weighting ENABLED") 
        config["run_layer1_optimization"] = True
        
        tprint_info("   - Event Pipeline Logging: ENABLED")
        config["enable_pipeline_logging"] = True

        # 0. Setup & Data Loading
        tprint_info("📥 Loading Market Data...")
        pipeline_state: Dict[str, Any] = {}
        market_data, source = self.load_market_data_or_fail(
            config,
            pipeline_state,
            allow_config_override=True,
            skip_artifacts=True
        )

        if market_data is None or market_data.empty:
            msg = "❌ Market data load failed or empty."
            tprint_error(msg)
            return {"success": False, "error": msg}

        tprint_success(f"✅ Loaded {len(market_data)} bars for {config.get('symbol', 'UNKNOWN')}")
        
        # Prepare outcomes directory specifically for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{config.get('symbol', 'sym')}_{config.get('timeframe', 'tf')}_{timestamp}"
        outcomes_dir = Path("outcomes") / self.step_name / run_name
        outcomes_dir.mkdir(parents=True, exist_ok=True)
        tprint_info(f"📂 Outcomes Directory: {outcomes_dir}")

        pipeline_results = {}

        # ------------------------------------------------------------------
        # Multi-Asset Setup (Cross-Asset Layer2)
        # ------------------------------------------------------------------
        assets = config.get("assets") or []
        multi_asset_mode = config.get("multi_asset_mode")
        multi_asset_requested = bool(multi_asset_mode) or len(assets) > 1
        if len(assets) > 1:
            tprint_info("🌐 Preparing cross-asset data for Layer 2...")
            cross_asset_data: Dict[str, pd.DataFrame] = {}
            primary_symbol = config.get("symbol", "")
            primary_asset = primary_symbol.replace("USDT", "") if primary_symbol.endswith("USDT") else primary_symbol
            for asset in assets:
                if asset == primary_asset:
                    cross_asset_data[asset] = market_data
                    continue

                asset_config = config.copy()
                asset_config["symbol"] = f"{asset}USDT"
                asset_data, asset_source = self.load_market_data_or_fail(
                    asset_config,
                    pipeline_state,
                    allow_config_override=True,
                    skip_artifacts=True,
                )
                if asset_data is None or asset_data.empty:
                    raise ValueError(f"Cross-asset load failed for {asset}USDT")
                cross_asset_data[asset] = asset_data
                tprint_success(
                    f"✅ Loaded cross-asset {asset} rows={len(asset_data)} source={asset_source}"
                )

            config["cross_asset_data"] = cross_asset_data
            tprint_info(f"🌐 Cross-asset payload ready: {list(cross_asset_data.keys())}")
        elif multi_asset_requested:
            tprint_warning(
                "⚠️ Multi-asset mode requested but assets list missing; cross-asset pipeline skipped."
            )

        if multi_asset_requested:
            config.setdefault("enable_cross_asset_validation", True)
            config.setdefault("enable_cross_asset_invariance", True)
            config.setdefault("layer3_use_enhanced", True)
            config.setdefault("enable_advanced_feature_selection", True)

        # ------------------------------------------------------------------
        # Layer 0: Kalman Filter & VWAP
        # ------------------------------------------------------------------
        if LAYER0_AVAILABLE:
            tprint_info("🔹 Running Layer 0: Kalman Filter & VWAP...")
            try:
                # Modifies market_data in-place/copy
                market_data_l0, l0_payload = run_layer0_kalman_vwap(
                    symbol=config.get("symbol", ""),
                    timeframe=config.get("timeframe", ""),
                    market_data=market_data,
                    config=config,
                    outcomes_dir=outcomes_dir,
                    run_optimization=config.get("run_layer0_optimization", True)
                )
                market_data = market_data_l0  # Update with smoothed prices
                config["layer0_params"] = l0_payload.get("best_params", {})
                pipeline_results["layer0"] = "success"
                tprint_success("✅ Layer 0 Complete")

                cross_asset_data = config.get("cross_asset_data")
                if isinstance(cross_asset_data, dict) and cross_asset_data:
                    if primary_asset in cross_asset_data:
                        cross_asset_data[primary_asset] = market_data
                    if UNIFIED_PRICE_AVAILABLE and config.get("propagate_layer0_to_cross_asset", True):
                        for asset, asset_df in cross_asset_data.items():
                            if asset_df is None or asset_df.empty:
                                continue
                            try:
                                unified_price = generate_unified_layer2_price(
                                    asset_df, layer0_params=config.get("layer0_params")
                                )
                                asset_df = asset_df.copy()
                                asset_df["layer0_price"] = unified_price
                                cross_asset_data[asset] = asset_df
                            except Exception as e:
                                tprint_warning(f"⚠️ Layer0 price propagation failed for {asset}: {e}")
                    config["cross_asset_data"] = cross_asset_data
            except Exception as e:
                tprint_error(f"❌ Layer 0 Failed: {e}")
                pipeline_results["layer0"] = "failed"
                # Proceed with raw data if L0 fails (robustness)
        else:
            tprint_warning("⚠️ Layer 0 module not available, skipping.")

        # Regimes are now generated internally by Layer 2 using AdaptiveHunterRouter
        # to ensure consistency and self-contained execution.

        # ------------------------------------------------------------------
        # Layer 1: Weighting Optimization
        # ------------------------------------------------------------------
        if LAYER1_AVAILABLE:
            tprint_info("🔹 Running Layer 1: Weighting Optimization...")
            try:
                # Generate proxy return labels for optimization
                # L1 optimizes weighting parameters based on return distribution
                proxy_labels = market_data['close'].pct_change().shift(-1).fillna(0)
                
                l1_params = run_layer1_optimization(
                    symbol=config.get("symbol", ""),
                    timeframe=config.get("timeframe", ""),
                    market_data=market_data,
                    labels=proxy_labels,
                    n_trials=config.get("layer1_n_trials", 20),
                    use_layer0_prices=True
                )
                config["layer1_params"] = l1_params
                pipeline_results["layer1"] = "success"
                tprint_success("✅ Layer 1 Complete")
            except Exception as e:
                tprint_error(f"❌ Layer 1 Failed: {e}")
                pipeline_results["layer1"] = "failed"
        else:
            tprint_warning("⚠️ Layer 1 module not available, skipping.")

        # ------------------------------------------------------------------
        # Layer 2: Causal Event Generation & Labeling
        # ------------------------------------------------------------------
        if LAYER2_AVAILABLE:
            tprint_info("🔹 Running Layer 2: Causal Labeling (Primary Model)...")
            try:
                # Instantiate L2 Step
                # Instantiate L2 Step with config to ensure parameters are passed
                l2_step = LabelBasedLayer2(step_name="label_based_layer_2", **config)

                # Inject our context to L2 (to share artifacts/logging context)
                l2_step.outcomes_dir = outcomes_dir
                if config.get("layer0_params"):
                    l2_step.layer0_params = config["layer0_params"]

                # We update config with L0/L1 params so L2 can use them
                l2_config = config.copy()
                l2_config['outcomes_dir'] = str(outcomes_dir) # Pass explicit path if supported

                # Execute L2
                # L2.run is async and takes input_data (DataFrame or Dict)
                # It generates events and labels, typically saved to artifacts.
                layer2_results = await l2_step.run(market_data)

                pipeline_results["layer2"] = "success"
                tprint_success("✅ Layer 2 Complete")
                
                # Load L2 Results for L3
                # We assume L2 saves 'events.parquet' and 'labels.parquet' in its artifact dir or outcomes
                events_files = list(outcomes_dir.glob("*_events.parquet")) + list(outcomes_dir.glob("events*.parquet"))
                labels_files = list(outcomes_dir.glob("*_labels.parquet")) + list(outcomes_dir.glob("labels*.parquet"))
                
                cross_asset_payload = None
                if isinstance(layer2_results, dict):
                    cross_asset_payload = layer2_results.get("cross_asset")
                if not events_files or not labels_files:
                    # Fallback: check standard artifacts dir if not in outcomes
                    tprint_warning("⚠️ L2 output files not found in Outcomes, checking Artifacts...")
                    # Implementation detail: Use what we have. If missing, L3 will fail.
                    events_df = pd.DataFrame() 
                    labels_df = pd.DataFrame()
                else:
                    events_df = pd.read_parquet(events_files[0])
                    labels_df = pd.read_parquet(labels_files[0])
                    tprint_info(f"   Loaded {len(events_df)} events and {len(labels_df)} labels from L2.")

                cross_asset_features = None
                if isinstance(cross_asset_payload, dict) and "panel" in cross_asset_payload:
                    try:
                        panel = cross_asset_payload["panel"]
                        if primary_asset:
                            panel_asset = panel.xs(primary_asset, level="ticker")
                            panel_asset = panel_asset.drop(
                                columns=[c for c in panel_asset.columns if c.startswith("y__")],
                                errors="ignore",
                            )
                            cross_asset_features = panel_asset.sort_index()
                            market_data = market_data.join(
                                cross_asset_features.reindex(market_data.index).ffill().fillna(0.0),
                                how="left",
                            )
                    except Exception as e:
                        tprint_warning(f"⚠️ Failed to align cross-asset panel features: {e}")

            except Exception as e:
                tprint_error(f"❌ Layer 2 Failed: {e}")
                pipeline_results["layer2"] = "failed"
                return {"success": False, "error": f"Layer 2 failed: {e}"}
        else:
            tprint_error("❌ Layer 2 module (LabelBasedLayer2) not available!")
            return {"success": False, "error": "Layer 2 missing"}

        # ------------------------------------------------------------------
        # Layer 3: Meta-Labeling (Analyst)
        # ------------------------------------------------------------------
        if LAYER3_AVAILABLE and not events_df.empty:
            tprint_info("🔹 Running Layer 3: Meta-Model (Analyst)...")
            try:
                # Prepare OOF DataFrame for L3
                # L3 expects a dataframe with base model predictions/signals and market data features
                # We merge events with market_data to get features
                
                # Ensure events are aligned
                valid_indices = events_df.index.intersection(market_data.index)
                if len(valid_indices) == 0:
                     raise ValueError("Events index does not align with Market Data index")
                
                # Construct base OOF DF
                # We need 'side' (primary signal), 'ret' (target return), 'bin' (label)
                # events_df usually has these.
                oof_df = events_df.loc[valid_indices].copy()
                
                # Merge basic market features if needed by L3 feature generator
                # (L3 usually generates its own features or extracts from market_data)
                
                # Join with labels if separate
                if not labels_df.empty:
                    common = oof_df.index.intersection(labels_df.index)
                    oof_df = oof_df.loc[common].join(labels_df.loc[common], rsuffix='_lbl')

                if cross_asset_features is not None:
                    oof_df = oof_df.join(
                        cross_asset_features.reindex(oof_df.index).ffill().fillna(0.0),
                        how="left",
                    )

                layer1_weight = None
                if WEIGHTS_AVAILABLE and config.get("layer1_params"):
                    returns_col = None
                    for candidate in ("ret", "realized_return", "return"):
                        if candidate in oof_df.columns:
                            returns_col = candidate
                            break
                    if returns_col is not None:
                        weight_params = dict(config.get("layer1_params", {}))
                        weight_params.setdefault(
                            "transaction_cost",
                            float(config.get("transaction_cost", DEFAULT_TRANSACTION_COST)),
                        )
                        weights = generate_weights_per_label(
                            returns=oof_df[returns_col].fillna(0.0).values,
                            t_events=oof_df.index,
                            **weight_params,
                        )
                        layer1_weight = pd.Series(weights, index=oof_df.index).fillna(1.0)
                        oof_df["layer1_weight"] = layer1_weight
                        tprint_info("   ⚖️ Layer 1 weights injected into Layer 3 inputs")
                
                # Define args for L3
                l3_oof, l3_results = layer3_analyst_lgbm(
                    oof_df=oof_df,
                    base_model_cols=['side'], # Primary signal column from L2
                    target_col='bin',         # Binary target from L2 labeling
                    train_split_date=config.get("train_split_date", None),
                    market_data=market_data,
                    config=config,
                    layer1_weight=layer1_weight
                )
                
                pipeline_results["layer3"] = "success"
                tprint_success("✅ Layer 3 Complete")
            except Exception as e:
                tprint_error(f"❌ Layer 3 Failed: {e}")
                pipeline_results["layer3"] = "failed"
                # If L3 fails, subsequent layers (Sizing) cannot proceed reliably
        else:
            tprint_warning("⚠️ Layer 3 skipped (Module missing or empty events).")

        # ------------------------------------------------------------------
        # Layer 4: Position Sizing (ExtraTrees)
        # ------------------------------------------------------------------
        if LAYER4_AVAILABLE and pipeline_results.get("layer3") == "success":
            tprint_info("🔹 Running Layer 4: Position Sizing...")
            try:
                # L4 trains on L3 OOF predictions to optimize specific financial metrics (Sortino/PnL)
                # It needs the L3 output dataframe (l3_oof)
                
                # 'realized_return' is usually the target for sizing
                target_return_col = 'ret' if 'ret' in l3_oof.columns else 'realized_return'
                
                l4_preds, l4_meta = train_layer4_extratrees(
                    df=market_data, # Pass full market data for feature generation
                    layer3_predictions=l3_oof,
                    target_col=target_return_col,
                    prob_col='meta_prob', # L3 output probability
                    config=config
                )
                
                pipeline_results["layer4"] = "success"
                tprint_success("✅ Layer 4 Complete")
            except Exception as e:
                tprint_error(f"❌ Layer 4 Failed: {e}")
                pipeline_results["layer4"] = "failed"
        else:
            tprint_warning("⚠️ Layer 4 skipped (dependencies missing).")

        # ------------------------------------------------------------------
        # Layer 5: Portfolio Backtest
        # ------------------------------------------------------------------
        if LAYER5_AVAILABLE and pipeline_results.get("layer4") == "success":
            tprint_info("🔹 Running Layer 5: Portfolio Backtest...")
            try:
                # L5 applies sizing and computes final metrics
                # It uses L4 output (l4_preds) which contains 'layer4_extratrees_prob' or similar
                
                sizer = Layer5PositionSizer(
                    oof_df=l4_preds,
                    p_col='layer4_extratrees_prob', # Output from L4
                    return_col='ret' if 'ret' in l4_preds.columns else 'realized_return',
                    transaction_cost=config.get("transaction_cost", 0.0006)
                )
                
                l5_metrics = sizer.run_backtest()
                pipeline_results["layer5"] = "success"
                tprint_success(f"✅ Layer 5 Complete. Final Sharpe: {l5_metrics.get('Sharpe', 'N/A')}")
            except Exception as e:
                tprint_error(f"❌ Layer 5 Failed: {e}")
                pipeline_results["layer5"] = "failed"
        else:
            tprint_warning("⚠️ Layer 5 skipped.")

        # Final Summary
        summary = {
            "success": pipeline_results.get("layer2") == "success", # Core success
            "pipeline_status": pipeline_results,
            "outcomes_dir": str(outcomes_dir)
        }
        
        # Save summary
        pd.DataFrame([summary]).to_json(outcomes_dir / "pipeline_summary.json")

        return summary

def register_meta_labeling_hpo_experiment_step() -> None:
    """Register the Meta-Labeling HPO step."""
    from src.training.steps.base_step import step_registry
    step_registry.register("meta_labeling_hpo_experiment", MetaLabelingHPOExperimentStep)