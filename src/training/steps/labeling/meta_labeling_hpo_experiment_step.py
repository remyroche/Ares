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

# REF: MULTI_ASSET_LOOP
Modified to iterate over all configured assets, running the full pipeline
for each as the primary target while using others as context.
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

# --- Pipeline Components with Lazy Loading ---
class LazyImports:
    """Lazy import wrapper to defer heavy module initialization until needed."""
    
    def __init__(self):
        self._imports = {}
        self._availability = {}
        self._import_times = {}
        self._load_order = []
    
    def _lazy_import(self, module_path, item_name, availability_key):
        """Lazy import with caching and performance tracking."""
        if availability_key not in self._availability:
            import_start = datetime.now()
            try:
                if self.verbose:
                    tprint_info(f"   🔄 Lazy loading {availability_key}: {module_path}.{item_name}")
                
                module = __import__(module_path, fromlist=[item_name])
                self._imports[availability_key] = getattr(module, item_name)
                self._availability[availability_key] = True
                
                import_duration = (datetime.now() - import_start).total_seconds()
                self._import_times[availability_key] = import_duration
                self._load_order.append(availability_key)
                
                if self.verbose:
                    tprint_success(f"   ✅ Loaded {availability_key} in {import_duration:.2f}s")
                
            except ImportError as e:
                self._imports[availability_key] = None
                self._availability[availability_key] = False
                self._import_times[availability_key] = 0.0
                
                if self.verbose:
                    tprint_error(f"   ❌ Failed to load {availability_key}: {e}")
        
        return self._imports[availability_key]
    
    def get_import_stats(self):
        """Get statistics about lazy loading performance."""
        loaded_count = sum(1 for status in self._availability.values() if status)
        total_import_time = sum(self._import_times.values())
        
        stats = {
            "total_modules": len(self._availability),
            "loaded_modules": loaded_count,
            "failed_modules": len(self._availability) - loaded_count,
            "total_import_time": total_import_time,
            "load_order": self._load_order.copy(),
            "import_times": self._import_times.copy()
        }
        return stats
    
    def log_import_summary(self):
        """Log a summary of all lazy imports."""
        stats = self.get_import_stats()
        
        tprint_info("📊 LAZY IMPORT SUMMARY:")
        tprint_info(f"   📦 Total modules: {stats['total_modules']}")
        tprint_info(f"   ✅ Loaded: {stats['loaded_modules']}")
        tprint_info(f"   ❌ Failed: {stats['failed_modules']}")
        tprint_info(f"   ⏱️ Total import time: {stats['total_import_time']:.2f}s")
        
        if stats['load_order']:
            tprint_info("   🔄 Load order:")
            for i, module in enumerate(stats['load_order'], 1):
                duration = stats['import_times'][module]
                status = "✅" if self._availability[module] else "❌"
                tprint_info(f"      {i:2d}. {status} {module} ({duration:.2f}s)")
    
    @property
    def verbose(self):
        """Get verbosity setting from global context if available."""
        # Try to get verbosity from a global config or default to True
        return True  # Default to verbose for now
    
    # Layer 0: Feature Engineering
    @property
    def run_layer0_kalman_vwap(self):
        return self._lazy_import('src.training.steps.labeling.label_based_layer_0', 
                               'run_layer0_kalman_vwap', 'layer0')
    
    @property
    def layer0_available(self):
        if 'layer0' not in self._availability:
            _ = self.run_layer0_kalman_vwap  # Trigger import check
        return self._availability.get('layer0', False)
    
    # Layer 1: Weighting Optimization
    @property
    def run_layer1_optimization(self):
        return self._lazy_import('src.training.steps.labeling.label_based_layer_1',
                               'run_layer1_optimization', 'layer1')
    
    @property
    def layer1_available(self):
        if 'layer1' not in self._availability:
            _ = self.run_layer1_optimization
        return self._availability.get('layer1', False)
    
    # Unified Price Layer 2
    @property
    def generate_unified_layer2_price(self):
        return self._lazy_import('src.training.steps.labeling.unified_price_layer2',
                               'generate_unified_layer2_price', 'unified_price')
    
    @property
    def unified_price_available(self):
        if 'unified_price' not in self._availability:
            _ = self.generate_unified_layer2_price
        return self._availability.get('unified_price', False)
    
    # Weights Generation
    @property
    def generate_weights_per_label(self):
        return self._lazy_import('src.training.steps.labeling.generate_weights_per_label',
                               'generate_weights_per_label', 'weights')
    
    @property
    def weights_available(self):
        if 'weights' not in self._availability:
            _ = self.generate_weights_per_label
        return self._availability.get('weights', False)
    
    # Layer 2: Causal Framework & Regime Detection
    @property
    def LabelBasedLayer2(self):
        return self._lazy_import('src.training.steps.labeling.label_based_layer_2',
                               'LabelBasedLayer2', 'layer2')
    
    @property
    def layer2_available(self):
        if 'layer2' not in self._availability:
            _ = self.LabelBasedLayer2
        return self._availability.get('layer2', False)
    
    @property
    def AdaptiveHunterRouter(self):
        return self._lazy_import('src.training.steps.labeling.adaptive_hunter_router',
                               'AdaptiveHunterRouter', 'hunter_router')
    
    @property
    def hunter_router_available(self):
        if 'hunter_router' not in self._availability:
            _ = self.AdaptiveHunterRouter
        return self._availability.get('hunter_router', False)
    
    # Layer 3: Meta-Model
    @property
    def layer3_analyst_lgbm(self):
        return self._lazy_import('src.training.steps.labeling.label_based_layer_3',
                               'layer3_analyst_lgbm', 'layer3')
    
    @property
    def layer3_available(self):
        if 'layer3' not in self._availability:
            _ = self.layer3_analyst_lgbm
        return self._availability.get('layer3', False)
    
    # Layer 4: Position Sizing
    @property
    def train_layer4_extratrees(self):
        return self._lazy_import('src.training.steps.labeling.layer4_extratrees_pnl',
                               'train_layer4_extratrees', 'layer4')
    
    @property
    def layer4_available(self):
        if 'layer4' not in self._availability:
            _ = self.train_layer4_extratrees
        return self._availability.get('layer4', False)
    
    # Layer 5: Portfolio Construction
    @property
    def Layer5PositionSizer(self):
        return self._lazy_import('src.training.steps.labeling.label_based_layer_5',
                               'Layer5PositionSizer', 'layer5')
    
    @property
    def layer5_available(self):
        if 'layer5' not in self._availability:
            _ = self.Layer5PositionSizer
        return self._availability.get('layer5', False)

# Global lazy imports instance
_lazy_imports = LazyImports()


class MetaLabelingHPOExperimentStep(BaseStep):
    """
    Orchestrator Step for the Full Label-Based Pipeline (L0-L5).
    """

    def __init__(self, step_name: str = "meta_labeling_hpo_experiment", use_versioned_artifacts: bool = True):
        super().__init__(step_name, use_versioned_artifacts)
        self._l2_step = None  # Will be initialized lazily
        self._l2_cache_key: Optional[str] = None
        # Reference to global lazy imports for easy access
        self.lazy_imports = _lazy_imports
        
        # Log lazy import initialization
        tprint_info("🔄 Lazy loading system initialized")
        tprint_info(f"   📦 {len(self.lazy_imports._availability)} modules available for deferred loading")

    def _build_l2_cache_key(self, market_data: pd.DataFrame, config: Dict[str, Any]) -> str:
        data_key = f"{market_data.shape}_{market_data.index[0] if len(market_data) else ''}_{market_data.index[-1] if len(market_data) else ''}"
        config_key = str(sorted((config or {}).items()))
        return f"{data_key}::{hash(config_key)}"

    def _is_global_multi_asset(self, config: Dict[str, Any]) -> bool:
        exec_mode = str(config.get("execution_mode", "")).lower()
        return bool(
            config.get("multi_asset_mode")
            or exec_mode in ("small_multi_asset", "full_multi_asset")
        )

    def _compute_global_sample_weights(
        self,
        oof_df: pd.DataFrame,
        config: Dict[str, Any],
    ) -> Optional[pd.Series]:
        """Compute inverse-volatility * uniqueness weights per asset for global mode."""
        try:
            if "asset_id" not in oof_df.columns:
                return None

            from src.training.steps.labeling.generate_weights_per_label import (
                compute_uniqueness_weights,
                finalize_sample_weights,
            )

            asset_stats = config.get("asset_stats") or {}
            cross_asset_data = config.get("cross_asset_data") or {}
            base_horizon = int(config.get("base_horizon_bars") or config.get("horizon") or 48)

            weights = pd.Series(1.0, index=oof_df.index, dtype=float)
            for asset in sorted(oof_df["asset_id"].dropna().unique()):
                asset_events = oof_df[oof_df["asset_id"] == asset]
                if asset_events.empty:
                    continue

                asset_md = cross_asset_data.get(asset)
                if asset_md is None or asset_md.empty:
                    continue

                asset_index = asset_md.index
                event_times = asset_events.index
                if isinstance(event_times, pd.MultiIndex):
                    event_times = event_times.get_level_values(0)

                pos = asset_index.get_indexer(event_times)
                valid_mask = pos >= 0
                if not valid_mask.any():
                    continue

                pos = pos[valid_mask]
                event_times = event_times[valid_mask]
                end_pos = np.clip(pos + base_horizon, 0, len(asset_index) - 1)
                t1 = pd.Series(asset_index[end_pos], index=event_times)
                uniq = compute_uniqueness_weights(t1, event_times, asset_index)
                uniq = uniq.reindex(event_times).fillna(1.0)

                stats = asset_stats.get(asset, {})
                asset_vol = stats.get("returns_std") or stats.get("vol_std")
                if asset_vol is None or not np.isfinite(asset_vol):
                    asset_vol = float(
                        asset_md.get("raw_returns", asset_md.get("close")).pct_change().std()
                        if "raw_returns" not in asset_md.columns
                        else asset_md["raw_returns"].std()
                    )
                asset_vol = float(asset_vol) if asset_vol and np.isfinite(asset_vol) else 0.0
                inv_vol = 1.0 / (asset_vol + 1e-9)

                weight_vals = uniq.values * inv_vol
                weights.loc[asset_events.index[valid_mask]] = weight_vals

            final_weights = finalize_sample_weights(weights.values)
            return pd.Series(final_weights, index=oof_df.index).fillna(1.0)
        except Exception as exc:
            tprint_warning(f"   ⚠️ Global sample weight computation failed: {exc}")
            return None

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the full pipeline sequence.
        Refactored to support multi-asset iteration.

        Args:
            config: Job configuration dictionary.

        Returns:
            Dict containing pipeline results and metrics (for the last processed asset or aggregated).
        """
        pipeline_start = datetime.now()
        tprint_info("🚀 Starting Meta-Labeling HPO Orchestration (Layers 0-5)...")
        tprint_info(f"⚡ Lazy Loading Enabled: {len(self.lazy_imports._availability)} modules deferred")
        tprint_info(f"📊 Pipeline Start Time: {pipeline_start.strftime('%H:%M:%S')}")

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

        exec_mode = str(config.get("execution_mode", "light")).lower()
        tprint_info(f"   - Execution Mode: {exec_mode.upper()} (config-driven)")
        if config.get("lookback_days"):
            tprint_info(
                f"   - Lookback Days: {config['lookback_days']} (config override)"
            )

        if (
            config.get("multi_asset_mode")
            and config.get("market_data") is not None
            and config.get("pooled_market_data_ready", False)
        ):
            tprint_info("🌍 Pooled multi-asset data detected; running single pooled pipeline.")
            return await self._process_asset(
                target_asset=config.get("primary_asset", "MULTI"),
                market_data=config["market_data"],
                cross_asset_dict=config.get("cross_asset_data", {}),
                base_config=config,
                pipeline_start_global=pipeline_start,
            )

        if self._is_global_multi_asset(config):
            return await self._execute_global_multi_asset(config, pipeline_start)

        # 1. Identify Assets to Trade
        primary_symbol = config.get("symbol", "")
        primary_asset = primary_symbol.replace("USDT", "") if primary_symbol.endswith("USDT") else primary_symbol
        
        raw_assets = config.get("assets")
        if isinstance(raw_assets, str):
            asset_list = [a.strip() for a in raw_assets.split(',')]
        elif isinstance(raw_assets, list):
            asset_list = raw_assets
        else:
            asset_list = []
            
        # Ensure distinct list: [primary_asset] + others
        target_assets = []
        if primary_asset:
            target_assets.append(primary_asset)
        
        for a in asset_list:
            if a not in target_assets:
                target_assets.append(a)
        
        if not target_assets:
            tprint_error("❌ No assets found in configuration.")
            return {"success": False, "error": "No assets configured"}

        tprint_info(f"🎯 Target Assets for Execution: {target_assets}")
        
        # 2. Pre-Load Data for All Assets
        tprint_info("📥 Pre-Loading Market Data for all assets...")
        all_market_data: Dict[str, pd.DataFrame] = {}
        pipeline_state: Dict[str, Any] = {}
        
        for asset in target_assets:
            symbol = f"{asset}USDT"
            asset_config = config.copy()
            asset_config["symbol"] = symbol
            
            tprint_info(f"   📥 Loading {symbol}...")
            # We skip artifacts here to avoid side effects during loading
            data, source = self.load_market_data_or_fail(
                asset_config,
                pipeline_state,
                allow_config_override=True,
                skip_artifacts=True
            )
            
            if data is not None and not data.empty:
                all_market_data[asset] = data
                tprint_success(f"   ✅ Loaded {asset}: {len(data)} bars ({source})")
            else:
                tprint_error(f"   ❌ Failed to load {asset}")

        if not all_market_data:
             tprint_error("❌ No market data loaded successfully.")
             return {"success": False, "error": "All data loads failed"}

        # 3. Iterate and Process Each Asset
        global_results = {}
        processed_count = 0
        
        for i, target_asset in enumerate(target_assets):
            if target_asset not in all_market_data:
                tprint_warning(f"⏩ Skipping {target_asset}: No data loaded.")
                continue
                
            tprint_info(f"\n{'='*60}")
            tprint_info(f"🚀 Processing Asset {i+1}/{len(target_assets)}: {target_asset}")
            tprint_info(f"{'='*60}")
            
            # Prepare context
            target_data = all_market_data[target_asset].copy()
            cross_data = {a: d.copy() for a, d in all_market_data.items() if a != target_asset}
            
            # Run Pipeline for this asset
            try:
                asset_result = await self._process_asset(
                    target_asset=target_asset,
                    market_data=target_data,
                    cross_asset_dict=cross_data,
                    base_config=config,
                    pipeline_start_global=pipeline_start
                )
                global_results[target_asset] = asset_result
                processed_count += 1
            except Exception as e:
                tprint_error(f"❌ Failed processing {target_asset}: {e}")
                global_results[target_asset] = {"success": False, "error": str(e)}

        # Final Summary
        total_duration = (datetime.now() - pipeline_start).total_seconds()
        
        tprint_success(f"\n🏁 ALL ASSETS PROCESSED: {processed_count}/{len(target_assets)}")
        tprint_info(f"⏱️ Total Execution Time: {total_duration:.1f}s")
        
        # Reuse last result's format for compatibility, or synthesize a summary
        final_summary = {
            "success": any(res.get("success", False) for res in global_results.values()),
            "global_results": global_results,
            "metrics": {
                "total_duration": total_duration,
                "processed_assets": processed_count
            }
        }
        
        return final_summary

    async def _execute_global_multi_asset(
        self,
        config: Dict[str, Any],
        pipeline_start_global: datetime,
    ) -> Dict[str, Any]:
        """Run a single pooled multi-asset pipeline execution."""
        tprint_info("🌍 Global multi-asset mode detected: pooling assets for a single run")

        primary_symbol = config.get("symbol", "")
        primary_asset = primary_symbol.replace("USDT", "") if primary_symbol.endswith("USDT") else primary_symbol

        raw_assets = config.get("assets")
        if isinstance(raw_assets, str):
            asset_list = [a.strip() for a in raw_assets.split(',')]
        elif isinstance(raw_assets, list):
            asset_list = raw_assets
        else:
            asset_list = []

        target_assets = []
        if primary_asset:
            target_assets.append(primary_asset)
        for asset in asset_list:
            if asset not in target_assets:
                target_assets.append(asset)

        if not target_assets:
            tprint_error("❌ No assets found in configuration.")
            return {"success": False, "error": "No assets configured"}

        tprint_info(f"🎯 Global Assets: {target_assets}")

        tprint_info("📥 Loading market data for global pool...")
        all_market_data: Dict[str, pd.DataFrame] = {}
        pipeline_state: Dict[str, Any] = {}

        for asset in target_assets:
            symbol = f"{asset}USDT"
            asset_config = config.copy()
            asset_config["symbol"] = symbol
            tprint_info(f"   📥 Loading {symbol}...")
            data, source = self.load_market_data_or_fail(
                asset_config,
                pipeline_state,
                allow_config_override=True,
                skip_artifacts=True,
            )
            if data is not None and not data.empty:
                all_market_data[asset] = data
                tprint_success(f"   ✅ Loaded {asset}: {len(data)} bars ({source})")
            else:
                tprint_error(f"   ❌ Failed to load {asset}")

        if not all_market_data:
            tprint_error("❌ No market data loaded successfully.")
            return {"success": False, "error": "All data loads failed"}

        if self.lazy_imports.layer0_available and config.get("global_layer0_per_asset", True):
            tprint_info("   🌊 Running Layer 0 per asset before pooling...")
            for asset, asset_df in list(all_market_data.items()):
                try:
                    run_opt = bool(config.get("run_layer0_optimization", True)) and asset == primary_asset
                    md_l0, l0_payload = self.lazy_imports.run_layer0_kalman_vwap(
                        symbol=f"{asset}USDT",
                        timeframe=config.get("timeframe", ""),
                        market_data=asset_df,
                        config=config,
                        outcomes_dir=Path(config.get("outcomes_dir", "outcomes")),
                        run_optimization=run_opt,
                    )
                    all_market_data[asset] = md_l0
                    if run_opt:
                        config["layer0_params"] = l0_payload.get("best_params", {})
                except Exception as e:
                    tprint_warning(f"⚠️ Layer 0 per-asset processing failed for {asset}: {e}")

        from src.training.steps.labeling.global_meta_labeling_hpo_sample_weighted import (
            GlobalMetaLabelingHPOSampleWeightedStep,
        )

        global_step = GlobalMetaLabelingHPOSampleWeightedStep("global_meta_labeling_hpo_sample_weighted")
        combined_data = global_step._combine_asset_data(
            all_market_data,
            target_assets,
            config,
            timeframe=config.get("timeframe"),
        )

        pooled_config = config.copy()
        pooled_config.update(
            {
                "multi_asset_mode": True,
                "assets": target_assets,
                "cross_asset_data": all_market_data,
                "market_data": combined_data,
                "primary_asset": primary_asset or target_assets[0],
                "asset_stats": global_step.asset_stats,
                "pooled_market_data_ready": True,
                "skip_layer0_in_global": True,
                "skip_layer1_in_global": True,
                "label_return_column": "residual_return",
                "use_market_residual_labels": True,
            }
        )

        return await self._process_asset(
            target_asset=pooled_config["primary_asset"],
            market_data=combined_data,
            cross_asset_dict=all_market_data,
            base_config=pooled_config,
            pipeline_start_global=pipeline_start_global,
        )

    async def _process_asset(self, 
                           target_asset: str, 
                           market_data: pd.DataFrame, 
                           cross_asset_dict: Dict[str, pd.DataFrame], 
                           base_config: Dict[str, Any],
                           pipeline_start_global: datetime) -> Dict[str, Any]:
        """
        Run the Layers 0-5 pipeline for a single asset.
        """
        # Clone config and update symbol
        config = base_config.copy()
        symbol = f"{target_asset}USDT"
        config["symbol"] = symbol
        config["primary_asset"] = target_asset
        
        # Prepare outcomes directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if config.get("multi_asset_mode") and config.get("pooled_market_data_ready", False):
            assets = config.get("assets") or []
            asset_tag = "_".join(assets) if assets else "multi_asset"
            run_name = f"multi_asset_{asset_tag}_{config.get('timeframe', 'tf')}_{timestamp}"
            outcomes_dir = Path("outcomes") / "multi_asset" / run_name
        else:
            run_name = f"{symbol}_{config.get('timeframe', 'tf')}_{timestamp}"
            outcomes_dir = Path("outcomes") / self.step_name / run_name
        outcomes_dir.mkdir(parents=True, exist_ok=True)
        tprint_info(f"📂 Outcomes Directory ({target_asset}): {outcomes_dir}")

        pipeline_results = {}
        
        # Inject cross-asset data into config for internal steps
        cross_asset_payload = dict(cross_asset_dict or {})
        if target_asset not in cross_asset_payload:
            cross_asset_payload[target_asset] = market_data
        config["cross_asset_data"] = cross_asset_payload
        config["assets"] = sorted(list(cross_asset_payload.keys()))
        
        if cross_asset_payload:
             tprint_info(f"🌐 Context Assets: {list(cross_asset_payload.keys())}")
             config.setdefault("enable_cross_asset_validation", True)
             config.setdefault("enable_cross_asset_invariance", True)

        # ------------------------------------------------------------------
        # Layer 0: Kalman Filter & VWAP
        # ------------------------------------------------------------------
        skip_layer0 = bool(config.get("skip_layer0_in_global"))
        if skip_layer0:
            tprint_info("ℹ️ Layer 0 skipped by configuration (skip_layer0_in_global=True).")
        elif self.lazy_imports.layer0_available:
            tprint_info("🔹 Running Layer 0: Kalman Filter & VWAP...")
            try:
                # Modifies market_data in-place/copy
                market_data_l0, l0_payload = self.lazy_imports.run_layer0_kalman_vwap(
                    symbol=symbol,
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

                # Propagate Layer 0 Smoothing to Context Assets
                # We do this locally for this run context
                if self.lazy_imports.unified_price_available and config.get("propagate_layer0_to_cross_asset", True) and cross_asset_payload:
                    tprint_info("   🌐 Propagating Layer 0 smoothing to context assets...")
                    updated_context = {}
                    for asset, asset_df in cross_asset_payload.items():
                         if asset_df is None or asset_df.empty: continue
                         try:
                            # Apply THIS asset's optimised params (or should we use default?)
                            # Using optimized params creates consistency in filtering logic
                            unified_price = self.lazy_imports.generate_unified_layer2_price(
                                asset_df, layer0_params=config.get("layer0_params")
                            )
                            asset_df = asset_df.copy()
                            asset_df["layer0_price"] = unified_price
                            updated_context[asset] = asset_df
                         except Exception as e:
                            tprint_warning(f"⚠️ Layer0 propagation failed for {asset}: {e}")
                            updated_context[asset] = asset_df
                    if target_asset not in updated_context:
                        updated_context[target_asset] = market_data
                    config["cross_asset_data"] = updated_context
                    config["assets"] = sorted(list(updated_context.keys()))

            except Exception as e:
                tprint_error(f"❌ Layer 0 Failed: {e}")
                pipeline_results["layer0"] = "failed"
                # Proceed with raw data if L0 fails (robustness)
        else:
            tprint_warning("⚠️ Layer 0 module not available, skipping.")

        # ------------------------------------------------------------------
        # Layer 1: Weighting Optimization
        # ------------------------------------------------------------------
        skip_layer1 = bool(config.get("skip_layer1_in_global"))
        if skip_layer1:
            tprint_info("ℹ️ Layer 1 skipped by configuration (skip_layer1_in_global=True).")
        elif self.lazy_imports.layer1_available:
            tprint_info("🔹 Running Layer 1: Weighting Optimization...")
            try:
                # Generate proxy return labels for optimization
                # L1 optimizes weighting parameters based on return distribution
                label_col = config.get("label_return_column", "close")
                if label_col in market_data.columns:
                    proxy_labels = market_data[label_col].pct_change().shift(-1).fillna(0)
                else:
                    proxy_labels = market_data["close"].pct_change().shift(-1).fillna(0)
                
                l1_params = self.lazy_imports.run_layer1_optimization(
                    symbol=symbol,
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
        events_df = pd.DataFrame()
        labels_df = pd.DataFrame()
        cross_asset_features = None
        
        if self.lazy_imports.layer2_available:
            tprint_info("🔹 Running Layer 2: Causal Labeling (Primary Model)...")
            layer2_start_time = datetime.now()
            try:
                l2_cache_key = self._build_l2_cache_key(market_data, config)
                
                # Check if we need to re-init L2 step for new symbol
                # or if we can reuse the instance. L2 step instance might hold symbol-specific state.
                # Safer to re-instantiate or clear state thoroughly.
                # Given 'symbol' is passed in init often, we re-instantiate to be safe.
                
                tprint_info("   🏗️ Initializing Layer 2 framework...")
                l2_step = self.lazy_imports.LabelBasedLayer2(step_name="label_based_layer_2", **config)
                self._l2_step = l2_step # Update cached ref
                self._l2_cache_key = l2_cache_key
                tprint_success("   ✅ Layer 2 framework initialized")

                # Inject our context to L2 (to share artifacts/logging context)
                l2_step.outcomes_dir = outcomes_dir
                if config.get("layer0_params"):
                    l2_step.layer0_params = config["layer0_params"]
                    tprint_info(f"   📊 Layer 0 params injected: {len(l2_step.layer0_params)} parameters")

                # Execute L2 with progress tracking
                tprint_info("   🎯 Executing Layer 2 causal framework...")
                l2_results = await l2_step.execute(market_data, config)
                
                if not isinstance(l2_results, dict):
                    raise RuntimeError(f"Layer 2 returned invalid payload type: {type(l2_results)}")
                
                if 'error' in l2_results:
                    raise RuntimeError(f"Layer 2 execution failed: {l2_results['error']}")

                # Extract and log key metrics
                events_files = list(outcomes_dir.glob("*_events.parquet")) + list(outcomes_dir.glob("events*.parquet"))
                labels_files = list(outcomes_dir.glob("*_labels.parquet")) + list(outcomes_dir.glob("labels*.parquet"))
                
                tprint_success(f"   📈 Layer 2 generated {len(events_files)} event files and {len(labels_files)} label files")
                
                # Load events and labels with metrics
                if events_files:
                    events_df = pd.read_parquet(events_files[0])
                    event_rate = len(events_df) / len(market_data) * 100
                    tprint_info(f"   📊 Loaded {len(events_df)} events ({event_rate:.2f}% of bars)")
                else:
                    tprint_warning("⚠️ L2 events parquet not found in Outcomes. Checking Artifacts...")

                if labels_files:
                    labels_df = pd.read_parquet(labels_files[0])
                    label_balance = labels_df['bin'].mean() * 100 if 'bin' in labels_df.columns else 0
                    tprint_info(f"   📊 Loaded {len(labels_df)} labels (balance: {label_balance:.1f}% positive)")
                elif not events_df.empty:
                    label_cols = ['bin', 'label', 'side', 'ret', 'realized_return']
                    available_label_cols = [c for c in label_cols if c in events_df.columns]
                    if available_label_cols:
                        labels_df = events_df[available_label_cols].copy()
                        tprint_info(
                            f"   📊 Derived {len(labels_df)} labels from events_df columns: {available_label_cols}"
                        )
                    else:
                        tprint_warning("⚠️ L2 labels parquet missing and no label columns found in events_df.")

                # Process cross-asset features with metrics
                cross_asset_payload = None
                if isinstance(l2_results, dict):
                    cross_asset_payload = l2_results.get("cross_asset")
                
                if isinstance(cross_asset_payload, dict) and "panel" in cross_asset_payload:
                    try:
                        panel = cross_asset_payload["panel"]
                        if target_asset:
                            panel_asset = panel.xs(target_asset, level="ticker")
                            panel_asset = panel_asset.drop(
                                columns=[c for c in panel_asset.columns if c.startswith("y__")],
                                errors="ignore",
                            )
                            cross_asset_features = panel_asset.sort_index()
                            ca_features_count = len([c for c in cross_asset_features.columns if c.startswith(("ca__", "ms__"))])
                            tprint_info(f"   🌐 Processing {ca_features_count} cross-asset features")
                            
                            market_data = market_data.join(
                                cross_asset_features.reindex(market_data.index).ffill().fillna(0.0),
                                how="left",
                            )
                            tprint_success(f"   ✅ Cross-asset features aligned (shape: {cross_asset_features.shape})")
                    except Exception as e:
                        tprint_warning(f"⚠️ Failed to align cross-asset panel features: {e}")

                # Log Layer 2 execution metrics
                layer2_duration = (datetime.now() - layer2_start_time).total_seconds()
                tprint_success(f"✅ Layer 2 Complete in {layer2_duration:.1f}s")
                tprint_info(f"   📊 Final metrics: {len(events_df)} events, {len(market_data)} bars, {len(cross_asset_features) if cross_asset_features is not None else 0} CA features")
                pipeline_results["layer2"] = "success"

            except Exception as e:
                layer2_duration = (datetime.now() - layer2_start_time).total_seconds()
                tprint_error(f"❌ Layer 2 Failed after {layer2_duration:.1f}s: {e}")
                pipeline_results["layer2"] = "failed"
                # return {"success": False, "error": f"Layer 2 failed: {e}"} 
                # Don't return, allow catching in loop
                raise RuntimeError(f"Layer 2 failed: {e}")
        else:
            tprint_error("❌ Layer 2 module (LabelBasedLayer2) not available!")
            raise RuntimeError("Layer 2 missing")

        # ------------------------------------------------------------------
        # Layer 3: Meta-Labeling (Analyst)
        # ------------------------------------------------------------------
        l3_oof = pd.DataFrame()
        if self.lazy_imports.layer3_available and not events_df.empty:
            tprint_info("🔹 Running Layer 3: Meta-Model (Analyst)...")
            layer3_start_time = datetime.now()
            try:
                # Prepare OOF DataFrame for L3
                
                # Ensure events are aligned
                valid_indices = events_df.index.intersection(market_data.index)
                if len(valid_indices) == 0:
                     raise ValueError("Events index does not align with Market Data index")
                
                alignment_rate = len(valid_indices) / len(events_df) * 100
                tprint_info(f"   📊 Event-market alignment: {len(valid_indices)}/{len(events_df)} ({alignment_rate:.1f}%)")
                
                # Construct base OOF DF
                oof_df = events_df.loc[valid_indices].copy()
                tprint_info(f"   📊 Base OOF DataFrame: {len(oof_df)} samples, {len(oof_df.columns)} features")
                
                # Join with labels if separate
                if not labels_df.empty:
                    common = oof_df.index.intersection(labels_df.index)
                    oof_df = oof_df.loc[common].join(labels_df.loc[common], rsuffix='_lbl')
                    tprint_info(f"   📊 Joined labels: {len(common)} aligned samples")

                if cross_asset_features is not None:
                    ca_features = cross_asset_features.reindex(oof_df.index).ffill().fillna(0.0)
                    overlap = oof_df.columns.intersection(ca_features.columns)
                    if len(overlap) > 0:
                        tprint_info(
                            f"   📊 Dropping {len(overlap)} overlapping cross-asset columns before join"
                        )
                        ca_features = ca_features.drop(columns=overlap, errors="ignore")
                    if not ca_features.empty:
                        oof_df = oof_df.join(ca_features, how="left")
                        tprint_info(f"   🌐 Added {len(ca_features.columns)} cross-asset features")

                layer1_weight = None
                if self.lazy_imports.weights_available and config.get("layer1_params"):
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
                        tprint_info(f"   ⚖️ Computing Layer 1 weights for {returns_col} returns...")
                        weights = self.lazy_imports.generate_weights_per_label(
                            returns=oof_df[returns_col].fillna(0.0).values,
                            t_events=oof_df.index,
                            **weight_params,
                        )
                        layer1_weight = pd.Series(weights, index=oof_df.index).fillna(1.0)
                        oof_df["layer1_weight"] = layer1_weight
                        weight_stats = {
                            'mean': layer1_weight.mean(),
                            'std': layer1_weight.std(),
                            'min': layer1_weight.min(),
                            'max': layer1_weight.max()
                        }
                        tprint_info(f"   ⚖️ Layer 1 weights injected: mean={weight_stats['mean']:.3f}, range=[{weight_stats['min']:.3f}, {weight_stats['max']:.3f}]")

                if layer1_weight is None and self._is_global_multi_asset(config):
                    global_weights = self._compute_global_sample_weights(oof_df, config)
                    if global_weights is not None:
                        oof_df["layer1_weight"] = global_weights
                        layer1_weight = global_weights
                        weight_stats = {
                            "mean": layer1_weight.mean(),
                            "std": layer1_weight.std(),
                            "min": layer1_weight.min(),
                            "max": layer1_weight.max(),
                        }
                        tprint_info(
                            "   ⚖️ Global sample weights injected: "
                            f"mean={weight_stats['mean']:.3f}, range=[{weight_stats['min']:.3f}, {weight_stats['max']:.3f}]"
                        )
                
                # Define args for L3
                tprint_info("   🧠 Training Layer 3 meta-model...")
                l3_oof, l3_results = self.lazy_imports.layer3_analyst_lgbm(
                    oof_df=oof_df,
                    base_model_cols=['side'], # Primary signal column from L2
                    target_col='bin',         # Binary target from L2 labeling
                    train_split_date=config.get("train_split_date", None),
                    market_data=market_data,
                    config=config,
                    layer1_weight=layer1_weight
                )
                
                # Log Layer 3 results and metrics
                if isinstance(l3_results, dict):
                    if 'model_performance' in l3_results:
                        perf = l3_results['model_performance']
                        tprint_success(f"   📈 L3 Model Performance:")
                        if 'auc' in perf:
                            tprint_info(f"      - AUC: {perf['auc']:.4f}")
                        if 'accuracy' in perf:
                            tprint_info(f"      - Accuracy: {perf['accuracy']:.4f}")
                        if 'feature_count' in perf:
                            tprint_info(f"      - Features: {perf['feature_count']}")
                    
                    if 'selected_features' in l3_results:
                        feature_count = len(l3_results['selected_features'])
                        tprint_info(f"   🎯 Selected {feature_count} features for meta-model")
                
                tprint_success(f"   ✅ Layer 3 OOF predictions: {len(l3_oof)} samples")
                pipeline_results["layer3"] = "success"
                
                layer3_duration = (datetime.now() - layer3_start_time).total_seconds()
                tprint_success(f"✅ Layer 3 Complete in {layer3_duration:.1f}s")
                
            except Exception as e:
                layer3_duration = (datetime.now() - layer3_start_time).total_seconds()
                tprint_error(f"❌ Layer 3 Failed after {layer3_duration:.1f}s: {e}")
                pipeline_results["layer3"] = "failed"
        else:
            if not self.lazy_imports.layer3_available:
                tprint_warning("⚠️ Layer 3 module not available, skipping.")
            elif events_df.empty:
                tprint_warning("⚠️ Layer 3 skipped: no events from Layer 2.")
            else:
                tprint_warning("⚠️ Layer 3 skipped: unknown reason.")

        # ------------------------------------------------------------------
        # Layer 4: Position Sizing (ExtraTrees)
        # ------------------------------------------------------------------
        l4_preds = pd.DataFrame()
        if self.lazy_imports.layer4_available and pipeline_results.get("layer3") == "success":
            tprint_info("🔹 Running Layer 4: Position Sizing...")
            layer4_start_time = datetime.now()
            try:
                # L4 trains on L3 OOF predictions
                target_return_col = 'ret' if 'ret' in l3_oof.columns else 'realized_return'
                if target_return_col not in l3_oof.columns:
                    raise ValueError(f"Target return column '{target_return_col}' not found in L3 OOF")
                
                tprint_info(f"   📊 Training position sizer on {len(l3_oof)} samples")
                tprint_info(f"   🎯 Target column: {target_return_col}")
                
                # Check for required probability column
                prob_col = 'meta_prob'
                if prob_col not in l3_oof.columns:
                    # Try alternative probability column names
                    for alt_col in ['prob', 'probability', 'meta_probability']:
                        if alt_col in l3_oof.columns:
                            prob_col = alt_col
                            break
                    else:
                        raise ValueError(f"No probability column found in L3 OOF (checked: meta_prob, prob, probability, meta_probability)")
                
                tprint_info(f"   📊 Using probability column: {prob_col}")
                
                # Train Layer 4 model
                tprint_info("   🧠 Training ExtraTrees position sizer...")
                l4_preds, l4_meta = self.lazy_imports.train_layer4_extratrees(
                    df=market_data, # Pass full market data for feature generation
                    layer3_predictions=l3_oof,
                    target_col=target_return_col,
                    prob_col=prob_col, # L3 output probability
                    config=config
                )
                
                # Log Layer 4 metrics
                if isinstance(l4_meta, dict):
                    if 'model_performance' in l4_meta:
                        perf = l4_meta['model_performance']
                        tprint_success(f"   📈 L4 Model Performance:")
                        if 'sortino' in perf:
                            tprint_info(f"      - Sortino: {perf['sortino']:.4f}")
                        if 'sharpe' in perf:
                            tprint_info(f"      - Sharpe: {perf['sharpe']:.4f}")
                        if 'max_drawdown' in perf:
                            tprint_info(f"      - Max DD: {perf['max_drawdown']:.4f}")
                    
                    if 'feature_importance' in l4_meta:
                        top_features = list(l4_meta['feature_importance'].keys())[:5]
                        tprint_info(f"   🎯 Top features: {', '.join(top_features)}")
                
                tprint_success(f"   ✅ Layer 4 predictions: {len(l4_preds)} samples")
                pipeline_results["layer4"] = "success"
                
                layer4_duration = (datetime.now() - layer4_start_time).total_seconds()
                tprint_success(f"✅ Layer 4 Complete in {layer4_duration:.1f}s")
                
            except Exception as e:
                layer4_duration = (datetime.now() - layer4_start_time).total_seconds()
                tprint_error(f"❌ Layer 4 Failed after {layer4_duration:.1f}s: {e}")
                pipeline_results["layer4"] = "failed"
        else:
            if not self.lazy_imports.layer4_available:
                tprint_warning("⚠️ Layer 4 module not available, skipping.")
            elif pipeline_results.get("layer3") != "success":
                tprint_warning("⚠️ Layer 4 skipped: Layer 3 failed or incomplete.")
            else:
                tprint_warning("⚠️ Layer 4 skipped: unknown reason.")

        # ------------------------------------------------------------------
        # Layer 5: Portfolio Backtest
        # ------------------------------------------------------------------
        if self.lazy_imports.layer5_available and pipeline_results.get("layer4") == "success":
            tprint_info("🔹 Running Layer 5: Portfolio Backtest...")
            layer5_start_time = datetime.now()
            try:
                # L5 applies sizing and computers final metrics
                
                # Check for required probability column in L4 predictions
                prob_col = 'layer4_extratrees_prob'
                if prob_col not in l4_preds.columns:
                    # Try alternative column names
                    for alt_col in ['prob', 'probability', 'sizing_prob']:
                        if alt_col in l4_preds.columns:
                            prob_col = alt_col
                            break
                    else:
                        raise ValueError(f"No sizing probability column found in L4 predictions")
                
                # Check for return column
                return_col = 'ret' if 'ret' in l4_preds.columns else 'realized_return'
                if return_col not in l4_preds.columns:
                    raise ValueError(f"Return column not found in L4 predictions")
                
                tprint_info(f"   📊 Running backtest on {len(l4_preds)} predictions")
                tprint_info(f"   🎯 Using probability: {prob_col}, returns: {return_col}")
                
                transaction_cost = config.get("transaction_cost", 0.0006)
                tprint_info(f"   💰 Transaction cost: {transaction_cost:.4f}")
                
                # Initialize and run position sizer
                tprint_info("   🧠 Initializing portfolio backtest...")
                sizer = self.lazy_imports.Layer5PositionSizer(
                    oof_df=l4_preds,
                    p_col=prob_col, # Output from L4
                    return_col=return_col,
                    transaction_cost=transaction_cost
                )
                
                tprint_info("   📈 Running portfolio backtest...")
                l5_metrics = sizer.run_backtest()
                
                # Log Layer 5 results
                if isinstance(l5_metrics, dict):
                    tprint_success(f"   📊 Portfolio Performance:")
                    key_metrics = ['sharpe', 'sortino', 'max_drawdown', 'total_return', 'win_rate']
                    for metric in key_metrics:
                        if metric in l5_metrics:
                            value = l5_metrics[metric]
                            if isinstance(value, (int, float)):
                                tprint_info(f"      - {metric.title()}: {value:.4f}")
                            else:
                                tprint_info(f"      - {metric.title()}: {value}")
                
                pipeline_results["layer5"] = "success"
                
                layer5_duration = (datetime.now() - layer5_start_time).total_seconds()
                final_sharpe = l5_metrics.get('sharpe', 'N/A')
                tprint_success(f"✅ Layer 5 Complete in {layer5_duration:.1f}s | Final Sharpe: {final_sharpe}")
                
            except Exception as e:
                layer5_duration = (datetime.now() - layer5_start_time).total_seconds()
                tprint_error(f"❌ Layer 5 Failed after {layer5_duration:.1f}s: {e}")
                pipeline_results["layer5"] = "failed"
        else:
            if not self.lazy_imports.layer5_available:
                tprint_warning("⚠️ Layer 5 module not available, skipping.")
            elif pipeline_results.get("layer4") != "success":
                tprint_warning("⚠️ Layer 5 skipped: Layer 4 failed or incomplete.")
            else:
                tprint_warning("⚠️ Layer 5 skipped: unknown reason.")

        # Asset Summary
        total_duration = (datetime.now() - pipeline_start_global).total_seconds() # Approximate
        
        # Calculate success rates
        completed_layers = sum(1 for status in pipeline_results.values() if status == "success")
        total_layers = len(pipeline_results)
        success_rate = completed_layers / total_layers * 100 if total_layers > 0 else 0
        
        # Log lazy import performance summary
        tprint_info(f"🔄 LAZY LOADING PERFORMANCE ({target_asset}):")
        lazy_stats = self.lazy_imports.get_import_stats()
        tprint_info(f"   📦 Modules loaded: {lazy_stats['loaded_modules']}/{lazy_stats['total_modules']}")
        if lazy_stats['total_import_time'] > 0:
            tprint_info(f"   🚀 Startup savings: Deferred {lazy_stats['loaded_modules']} heavy imports")
        
        summary = {
            "success": pipeline_results.get("layer2") == "success", # Core success
            "pipeline_status": pipeline_results,
            "outcomes_dir": str(outcomes_dir),
            "metrics": {
                "completed_layers": completed_layers,
                "total_layers": total_layers,
                "success_rate": success_rate,
                "events_generated": len(events_df) if not events_df.empty else 0,
                "market_bars": len(market_data),
                "cross_asset_features": len(cross_asset_features) if cross_asset_features is not None else 0,
                "lazy_loading": lazy_stats
            }
        }
        
        # Save summary
        pd.DataFrame([summary]).to_json(outcomes_dir / "pipeline_summary.json")
        
        # Final pipeline summary with metrics
        tprint_success(f"🎯 ASSET PIPELINE COMPLETE: {target_asset}")
        tprint_info(f"   ✅ Success Rate: {success_rate:.1f}% ({completed_layers}/{total_layers} layers)")
        tprint_info(f"   📈 Events Generated: {len(events_df) if not events_df.empty else 0}")
        tprint_info(f"   📁 Results: {outcomes_dir}")
        
        # Layer-by-layer status
        tprint_info(f"   🔍 LAYER STATUS:")
        for layer, status in pipeline_results.items():
            status_icon = "✅" if status == "success" else "❌" if status == "failed" else "⚠️"
            tprint_info(f"      {status_icon} {layer.upper()}: {status}")
        
        return summary

def register_meta_labeling_hpo_experiment_step() -> None:
    """Register the Meta-Labeling HPO step."""
    from src.training.steps.base_step import step_registry
    step_registry.register("meta_labeling_hpo_experiment", MetaLabelingHPOExperimentStep)