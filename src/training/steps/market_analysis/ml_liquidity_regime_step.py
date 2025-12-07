"""
ML Liquidity Regime Step

This step constructs liquidity-based regimes from 15m OHLCV data, focused on
Effort vs Result of price moves and participation (volume context).

Primary goals:
- Detect Valid Trend, Absorption, Ghost/Drift, Steamroller, and Apathy regimes.
- Assign regimes via a hierarchical rule-based decision tree optimized for
  winsorized CoV (WCV) separation across core liquidity axes.
- Optionally refine regimes using centroid-based clustering in a discriminative
  liquidity feature space.
- Compute distance-based soft probabilities for each regime and expose them as
  standardized downstream features.
- Save 15m training artifacts (feature frame, regime labels, tree thresholds,
  and quality metrics).
- Save native 15m regime probabilities as a dedicated artifact for downstream
  consumers.
"""

import logging
import time
from typing import Any, Dict, Optional, Tuple, List
from dataclasses import asdict, is_dataclass
from datetime import datetime
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# Hardware and vectorization optimizations
hardware_optimization_enabled = False
try:
    from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager, WorkloadType, OptimizationLevel
    from src.utils.matrix_operations.hardware_integration import get_hardware_manager
    from src.feature_selection.optimizations.vectorized_operations import VectorizedOperations, VectorizationConfig
    hardware_optimization_enabled = True
except ImportError as e:
    logging.warning(f"Hardware optimization tools not available: {e}")
    UnifiedHardwareManager = None
    get_hardware_manager = None
    VectorizedOperations = None
    VectorizationConfig = None

from src.training.steps.base_step import BaseStep
from src.utils.tprint import (
    tprint,
    tprint_info,
    tprint_warning,
    tprint_error,
)
from src.features_common.transforms.scaling_normalization import (
    ScalingNormalizer,
    winsorized_zscore_normalize,
    rolling_adaptive_normalize,
)
from src.utils.feature_common.volume_transforms import log1p_zscore_normalize
from src.feature_generation.categories.liquidity_regime_features import (
    generate_liquidity_regime_features,
)
from src.utils.versioned_artifacts.temporal_splits import (
    create_temporal_split_config_for_pipeline,
    TemporalSplitConfig,
)
from src.training.steps.market_analysis.clusters.cluster_quality_assessor import (
    ClusterQualityAssessor,
    ClusterQualityMetrics,
)
from src.training.steps.market_analysis.clusters.liquidity_cluster_quality_assessor import (
    LiquidityClusterQualityAssessor,
    LiquidityClusterQualityMetrics,
)
from src.utils.ml_common.feature_engineering.feature_smoothing import apply_ewm_smoothing
from src.utils.ml_common.evaluation.hsic import calculate_hsic

logger = logging.getLogger(__name__)


class MLLiquidityRegimeStep(BaseStep):
    """Pipeline step to construct liquidity-based regimes from 15m OHLCV."""

    def __init__(self, step_name: str = "ml_liquidity_regime_step"):
        super().__init__(step_name, use_versioned_artifacts=True)
        self.logger = logger.getChild("MLLiquidityRegimeStep") if hasattr(logger, "getChild") else logger
        
        # Initialize hardware optimization
        self.hardware_manager = None
        self.vectorized_ops = None
        if hardware_optimization_enabled:
            try:
                self.hardware_manager = get_hardware_manager()
                if self.hardware_manager is not None:
                    self.hardware_manager.optimize_for_workload(
                        WorkloadType.FEATURE_ENGINEERING,
                        OptimizationLevel.BALANCED
                    )
                
                vector_config = VectorizationConfig(
                    enable_vectorization=True,
                    enable_hardware_acceleration=True,
                    use_optimized_algorithms=True,
                    chunk_size=5000,  # Optimized for M1
                    memory_limit_mb=1024
                )
                self.vectorized_ops = VectorizedOperations(vector_config)
                tprint("✅ Hardware optimization enabled", "SUCCESS")
            except Exception as e:
                tprint_warning(f"Hardware optimization failed, falling back to CPU: {e}")
        
        tprint(f"✅ Initialized {step_name} step", "SUCCESS")
    
    def generate_config_variations(self, base_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate multiple configuration variations for automated testing."""

        # Define parameter ranges for testing
        config_variations = {
            "liquidity_n_regimes": ["auto", 4, 5, 6, 7, 8],
            # Prefer WCoV-based regime detection; keep others for experimentation
            "liquidity_regime_detection_method": ["wcv", "silhouette", "elbow", "gap_statistic"],
            "liquidity_min_regime_size": [0.03, 0.05, 0.08],
            # Center stability search around the empirically best 0.6
            "liquidity_regime_stability_threshold": [0.6, 0.55, 0.65],
            "liquidity_enable_centroid_refinement": [True, False],
            "liquidity_centroid_iterations": [2, 3, 5],
            "liquidity_use_ewm_features": [True, False],
            "liquidity_ewm_periods": [[2, 6, 10], [3, 8, 12], [2, 4, 8]],
            "liquidity_enable_prob_calibration": [True, False],
            # Center temporal context around the currently best config (12/96, 24)
            # and probe shorter relative-volume and range windows for burst sensitivity
            "liquidity_rvol_lookback_24": [48, 24, 16],
            "liquidity_rvol_lookback_168": [384, 192, 672],
            "liquidity_range_std_lookback": [96, 48, 32],
            "liquidity_winsor_lower": [0.005, 0.01, 0.025],
            "liquidity_winsor_upper": [0.975, 0.99, 0.995],
            "liquidity_prune_strategy": ["none", "light", "moderate", "aggressive"],
        }

        # Generate combinations (limit to reasonable number)
        max_combinations = int(base_config.get("liquidity_max_config_combinations", 50))

        # Create systematic variations
        configs = []

        # 1. Base config (default)
        base_copy = base_config.copy()
        for key, values in config_variations.items():
            if key in base_copy:
                base_copy[key] = values[0]  # Use first value as default
        configs.append(base_copy)

        # 2. Single parameter variations
        for param_name, param_values in config_variations.items():
            if len(configs) >= max_combinations:
                break
            for value in param_values[1:]:  # Skip first value (already used in base)
                if len(configs) >= max_combinations:
                    break
                config = base_config.copy()
                config[param_name] = value
                configs.append(config)

        # 3. Key interaction combinations (most important parameters)
        key_params = {
            "liquidity_n_regimes": ["auto", 5, 7],
            "liquidity_regime_detection_method": ["silhouette", "elbow"],
            "liquidity_enable_centroid_refinement": [True, False],
            "liquidity_use_ewm_features": [True, False],
        }

        for param_combo in itertools.product(*[key_params[k] for k in key_params]):
            if len(configs) >= max_combinations:
                break
            config = base_config.copy()
            for i, param_name in enumerate(key_params):
                config[param_name] = param_combo[i]
            configs.append(config)

        tprint_info(f"🔧 Generated {len(configs)} configuration variations for testing")
        return configs[:max_combinations]
    
    async def run_config_batch(self, configs: List[Dict[str, Any]], symbol: str, exchange: str) -> List[Dict[str, Any]]:
        """Run a batch of configurations and collect results."""
        
        results = []
        total_configs = len(configs)
        
        for i, base_config in enumerate(configs):
            # For HPO-style runs, skip the generic ClusterQualityAssessor which
            # can be very expensive on long histories. Liquidity-specific
            # quality via LiquidityClusterQualityAssessor remains enabled.
            config = dict(base_config)
            config.setdefault("liquidity_quality_skip_generic_cluster_assessor", True)
            # Also skip probability calibration during HPO sweeps so that we
            # only pay the calibration cost once for the final winning
            # configuration when execute is called directly.
            config.setdefault("liquidity_skip_calibration_for_hpo", True)
            # Mark this config as part of an HPO sweep so downstream logic can
            # apply HPO-specific optimizations.
            config.setdefault("liquidity_enable_hpo", True)
            # Disable rule-based teacher during HPO to avoid expensive
            # per-feature, per-sample soft-label derivation. The final best
            # config, when executed directly via the launcher, will not have
            # this override and will therefore run the full teacher.
            config.setdefault("liquidity_enable_teacher", False)
            # Disable centroid refinement during HPO; refinement and distance-
            # based regime confidence are reserved for the final model run.
            config.setdefault("liquidity_enable_centroid_refinement", False)
            # Skip probability timeframe mapping during HPO runs; probabilities
            # are mapped only once for the final winning configuration.
            config.setdefault("liquidity_skip_15m_mapping_for_hpo", True)
            # When auto-pruning is enabled for a config, run the pruned
            # retrain on a subsample of the data during HPO to reduce
            # training time while still capturing relative feature effects.
            config.setdefault("liquidity_prune_subsample_for_hpo", True)
            config.setdefault("liquidity_prune_subsample_fraction", 0.5)

            tprint_info(f"🚀 Running config {i+1}/{total_configs}: {self.get_config_signature(config)}")
            
            try:
                # Run the step with this configuration
                start_time = time.time()
                result = await self.execute(config)
                execution_time = time.time() - start_time
                
                # Extract key metrics
                metrics = result.get("metrics", {})
                quality_metrics = {
                    "config_signature": self.get_config_signature(config),
                    "config_id": i + 1,
                    "execution_time": execution_time,
                    "success": result.get("success", False),
                    "overall_quality_score": metrics.get("liquidity_overall_quality_score", 0.0),
                    "effort_result_cov_separation_score": metrics.get("liquidity_effort_result_cov_separation_score", 0.0),
                    "returns_cov_separation_score": metrics.get("liquidity_returns_cov_separation_score", 0.0),
                    "class_balance_score": metrics.get("class_balance_score", 0.0),
                    # Forward 1h return diagnostics (teacher regimes)
                    "forward_return_mean_1h_weighted": metrics.get("forward_return_mean_1h_weighted", None),
                    "forward_return_mean_1h_best_regime": metrics.get("forward_return_mean_1h_best_regime", None),
                    # Teacher-based diagnostics (if available)
                    "teacher_label_agreement_rate": metrics.get("teacher_label_agreement_rate", None),
                    "teacher_mean_confidence": metrics.get("teacher_mean_confidence", None),
                    "n_regimes": metrics.get("n_regimes", 0),
                    "n_samples": metrics.get("n_samples", 0),
                    "error": result.get("error", ""),
                }
                
                # Add configuration details
                quality_metrics.update({
                    f"config_{k}": v for k, v in config.items() 
                    if k.startswith("liquidity_") and not callable(v)
                })
                
                # Detailed per-trial HPO logging for diagnostics
                try:
                    overall_score = float(quality_metrics.get("overall_quality_score", 0.0) or 0.0)
                    effort_cov = float(quality_metrics.get("effort_result_cov_separation_score", 0.0) or 0.0)
                    returns_cov = float(quality_metrics.get("returns_cov_separation_score", 0.0) or 0.0)
                    class_balance = float(quality_metrics.get("class_balance_score", 0.0) or 0.0)
                    n_regimes_val = quality_metrics.get("n_regimes", 0)
                    n_samples_val = quality_metrics.get("n_samples", 0)

                    try:
                        n_regimes = int(n_regimes_val or 0)
                    except Exception:
                        n_regimes = 0

                    try:
                        n_samples = int(n_samples_val or 0)
                    except Exception:
                        n_samples = 0

                    parts = [
                        f"trial={i+1}/{total_configs}",
                        f"score={overall_score:.3f}",
                        f"effort_cov={effort_cov:.3f}",
                        f"returns_cov={returns_cov:.3f}",
                        f"class_balance={class_balance:.3f}",
                        f"n_regimes={n_regimes}",
                        f"n_samples={n_samples}",
                    ]

                    tprint_info("📊 HPO trial summary: " + ", ".join(parts))
                except Exception as log_exc:
                    tprint_warning(f"Failed to log HPO trial summary for config {i+1}: {log_exc}")

                results.append(quality_metrics)
                
                if result.get("success", False):
                    tprint_info(f"✅ Config {i+1} completed: quality_score={quality_metrics['overall_quality_score']:.3f}")
                else:
                    tprint_warning(f"⚠️ Config {i+1} failed: {quality_metrics['error']}")
                    
            except Exception as e:
                tprint_error(f"❌ Config {i+1} crashed: {e}")
                results.append({
                    "config_signature": self.get_config_signature(config),
                    "config_id": i + 1,
                    "execution_time": 0,
                    "success": False,
                    "error": str(e),
                    "overall_quality_score": 0.0,
                })
        
        return results
    
    def get_config_signature(self, config: Dict[str, Any]) -> str:
        """Generate a compact signature for configuration identification."""
        key_params = [
            "liquidity_n_regimes",
            "liquidity_regime_detection_method", 
            "liquidity_enable_centroid_refinement",
            "liquidity_use_ewm_features",
            "liquidity_min_regime_size",
            "liquidity_regime_stability_threshold"
        ]
        
        parts = []
        for param in key_params:
            value = config.get(param, "default")
            if isinstance(value, list):
                value = "_".join(str(x) for x in value[:2])  # Limit list length
            parts.append(f"{param[:8]}={value}")
        
        return "|".join(parts)
    
    def analyze_and_rank_results(self, results: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Analyze results and rank configurations by quality."""
        
        if not results:
            return pd.DataFrame(), {}
        
        df = pd.DataFrame(results)
        
        # Filter successful runs
        successful = df[df["success"] == True].copy()
        failed = df[df["success"] == False].copy()
        
        tprint_info(f"📊 Analysis: {len(successful)} successful, {len(failed)} failed runs")
        
        if len(successful) == 0:
            tprint_warning("⚠️ No successful configurations to analyze")
            return df, {"best_config": None, "analysis": "no_successful_runs"}
        
        # Compute composite score based purely on overall liquidity regime quality.
        successful["composite_score"] = successful["overall_quality_score"].astype(float)
        
        # Sort by composite score
        successful = successful.sort_values("composite_score", ascending=False)
        
        # Get best configuration
        best_config = successful.iloc[0].to_dict()
        
        # Analysis summary
        analysis = {
            "best_config": best_config,
            "total_runs": len(results),
            "successful_runs": len(successful),
            "failed_runs": len(failed),
            "best_composite_score": best_config["composite_score"],
            "best_quality_score": best_config["overall_quality_score"],
            "top_5_configs": successful.head(5).to_dict("records"),
            "parameter_importance": self.analyze_parameter_importance(successful),
        }
        
        # Display results
        self.display_results_summary(successful, failed, analysis)
        
        return pd.concat([successful, failed], ignore_index=True), analysis
    
    def analyze_parameter_importance(self, successful: pd.DataFrame) -> Dict[str, Any]:
        """Analyze which parameters correlate with better results."""
        
        importance = {}
        
        # Analyze key parameters
        key_params = [
            "config_liquidity_n_regimes",
            "config_liquidity_regime_detection_method",
            "config_liquidity_enable_centroid_refinement", 
            "config_liquidity_use_ewm_features"
        ]
        
        for param in key_params:
            if param in successful.columns:
                # Group by parameter value and compute mean scores
                param_analysis = successful.groupby(param)["composite_score"].agg([
                    "count", "mean", "std", "min", "max"
                ]).round(4)
                importance[param] = param_analysis.to_dict()
        
        return importance
    
    def display_results_summary(self, successful: pd.DataFrame, failed: pd.DataFrame, analysis: Dict[str, Any]) -> None:
        """Display comprehensive results summary."""
        
        print("\n" + "="*80)
        print("🏆 LIQUIDITY REGIME CONFIGURATION OPTIMIZATION RESULTS")
        print("="*80)
        
        print(f"\n📊 SUMMARY:")
        print(f"   Total configurations tested: {analysis['total_runs']}")
        print(f"   Successful runs: {analysis['successful_runs']}")
        print(f"   Failed runs: {analysis['failed_runs']}")
        print(f"   Success rate: {analysis['successful_runs']/analysis['total_runs']*100:.1f}%")
        
        if analysis['best_config']:
            print(f"\n🥇 BEST CONFIGURATION:")
            print(f"   Signature: {analysis['best_config']['config_signature']}")
            print(f"   Composite Score: {analysis['best_composite_score']:.4f}")
            print(f"   Quality Score: {analysis['best_quality_score']:.4f}")
            print(f"   Execution Time: {analysis['best_config']['execution_time']:.1f}s")
            print(f"   N Regimes: {analysis['best_config']['n_regimes']}")
        
        print(f"\n🏅 TOP 5 CONFIGURATIONS:")
        cols = [
            "config_id", "config_signature", "composite_score", 
            "overall_quality_score", "execution_time", "n_regimes"
        ]
        print(successful[cols].head(5).to_string(index=False))
        
        # Parameter importance analysis
        if analysis["parameter_importance"]:
            print(f"\n🔍 PARAMETER IMPORTANCE:")
            for param, stats in analysis["parameter_importance"].items():
                param_name = param.replace("config_", "")
                print(f"\n   {param_name}:")
                for value, metrics in stats.items():
                    if isinstance(metrics, dict) and "count" in metrics:
                        print(f"      {value}: score={metrics['mean']:.3f} (count={metrics['count']})")
        
        if len(failed) > 0:
            print(f"\n❌ COMMON FAILURE MODES:")
            error_counts = failed["error"].value_counts().head(5)
            for error, count in error_counts.items():
                print(f"   {count}x: {error[:100]}...")
        
        print("\n" + "="*80)
    
    async def run_automated_config_optimization(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run automated configuration optimization.
        
        This method:
        1. Generates multiple configuration variations
        2. Runs each configuration
        3. Analyzes and ranks results
        4. Returns the best configuration and analysis
        """
        
        tprint_info("🔬 Starting automated liquidity regime configuration optimization...")
        
        # Generate configuration variations
        configs = self.generate_config_variations(base_config)
        
        # Run configuration batch
        symbol = base_config.get("symbol", "ETHUSDT")
        exchange = base_config.get("exchange", "binance")
        
        results = await self.run_config_batch(configs, symbol, exchange)
        
        # Analyze and rank results
        results_df, analysis = self.analyze_and_rank_results(results)
        
        # Save results
        self.save_optimization_results(results_df, analysis, symbol)
        
        tprint_info("🎯 Automated configuration optimization completed!")
        
        return {
            "best_config": analysis.get("best_config"),
            "analysis": analysis,
            "all_results": results_df.to_dict("records"),
            "optimization_summary": {
                "total_configs_tested": len(configs),
                "successful_runs": analysis.get("successful_runs", 0),
                "best_composite_score": analysis.get("best_composite_score", 0),
            }
        }
    
    def save_optimization_results(self, results_df: pd.DataFrame, analysis: Dict[str, Any], symbol: str) -> None:
        """Save optimization results to files."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        results_file = outcomes_dir / f"liquidity_config_optimization_{symbol}_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        tprint_info(f"💾 Saved detailed results to: {results_file}")
        
        # Save analysis summary
        analysis_file = outcomes_dir / f"liquidity_config_optimization_{symbol}_{timestamp}_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        tprint_info(f"💾 Saved analysis summary to: {analysis_file}")
        
        # Save best config as YAML for easy reuse
        if analysis.get("best_config"):
            best_config_file = outcomes_dir / f"liquidity_best_config_{symbol}_{timestamp}.yaml"
            best_config = analysis["best_config"]
            
            # Extract only config parameters (not metrics)
            config_params = {}
            for key, value in best_config.items():
                if key.startswith("config_"):
                    param_name = key.replace("config_", "")
                    config_params[param_name] = value
            
            with open(best_config_file, 'w') as f:
                import yaml
                yaml.dump(config_params, f, default_flow_style=False, indent=2)
            tprint_info(f"Saved best config to: {best_config_file}")

    def _maybe_apply_latest_best_config(self, config: Dict[str, Any], symbol: str) -> None:
        """Optionally apply latest liquidity best-config overrides.

        This helper looks for the most recent
        outcomes/liquidity_best_config_{symbol}_*.yaml produced by
        automated_liquidity_optimizer.py and applies its parameters as
        config overrides.

        Guardrails:
        - If config["liquidity_use_best_config"] is explicitly False,
          this helper is a no-op.
        - If the caller already supplied any concrete liquidity_*
          parameters (beyond meta toggles), we assume they are
          intentionally overriding and skip auto-loading.
        - Any errors while loading/parsing simply fall back to existing
          config + defaults.
        """

        try:
            # Allow explicit opt-out for callers such as HPO workflows.
            if not bool(config.get("liquidity_use_best_config", True)):
                return

            # Detect explicit liquidity_* overrides supplied by caller.
            liquidity_keys = [k for k in config.keys() if k.startswith("liquidity_")]
            meta_keys = {"liquidity_use_best_config", "liquidity_best_config_dir"}
            explicit_keys = [k for k in liquidity_keys if k not in meta_keys]
            if explicit_keys:
                # Respect explicit liquidity_* configuration provided by caller.
                return

            outcomes_base = config.get("liquidity_best_config_dir", "outcomes")
            outcomes_dir = Path(outcomes_base)
            if not outcomes_dir.exists():
                return

            pattern = f"liquidity_best_config_{symbol}_*.yaml"
            candidates = sorted(outcomes_dir.glob(pattern))
            if not candidates:
                return

            best_path = candidates[-1]

            try:
                import yaml  # Local import to avoid hard dependency at module import time
            except Exception as import_exc:  # pragma: no cover - defensive
                tprint_warning(
                    f"Failed to import yaml for liquidity best-config loading; "
                    f"continuing with defaults: {import_exc}"
                )
                return

            with best_path.open("r") as f:
                loaded = yaml.safe_load(f) or {}

            if not isinstance(loaded, dict):
                return

            overrides: Dict[str, Any] = {
                str(k): v for k, v in loaded.items() if str(k).startswith("liquidity_")
            }

            if not overrides:
                return

            config.update(overrides)
            tprint_info(
                f"Using latest liquidity best-config overrides from {best_path.name} "
                f"(dir={outcomes_dir})"
            )

        except Exception as exc:  # pragma: no cover - defensive guardrail
            tprint_warning(
                f"Failed to apply latest liquidity best-config overrides; "
                f"continuing with defaults: {exc}"
            )

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the liquidity regime construction and model training.

        Expected config keys (minimum):
            - symbol: Trading symbol (e.g., 'ETHUSDT')
            - exchange: Exchange name (e.g., 'binance')
            - regime_timeframe: Timeframe used for liquidity regimes (default: '15m')
            - direction: Trading direction (default: 'long')
            - execution_mode: 'full', 'light', 'blank', etc.
        """
        start_time = time.time()

        try:
            symbol = str(config.get("symbol", "ETHUSDT"))
            exchange = str(config.get("exchange", "binance"))
            base_timeframe = str(config.get("timeframe", "15m"))
            regime_timeframe = str(config.get("regime_timeframe", base_timeframe))
            direction = str(config.get("direction", "long"))
            execution_mode = str(config.get("execution_mode", "light")).lower()

            # Enforce 15m-native liquidity regimes when the base timeframe is
            # 15m, overriding any conflicting regime_timeframe (such as 1h)
            # so that regimes and probabilities are generated directly on the
            # 15m grid rather than a coarser 1h grid.
            if base_timeframe == "15m" and regime_timeframe != base_timeframe:
                tprint_warning(
                    f"Overriding regime_timeframe={regime_timeframe} to base timeframe={base_timeframe} "
                    "for ml_liquidity_regime_step to ensure 15m-native regimes and probabilities."
                )
                regime_timeframe = base_timeframe
                config["regime_timeframe"] = regime_timeframe

            if not symbol or not exchange:
                raise ValueError("Config must include 'symbol' and 'exchange'")

            # Optionally apply latest winning liquidity configuration if
            # available and no explicit liquidity_* overrides are present.
            # This wires the ml_liquidity_regime_step launcher path to
            # automatically consume the best settings discovered by
            # scripts/automated_liquidity_optimizer.py while leaving HPO
            # workflows free to explore their own config space.
            self._maybe_apply_latest_best_config(config, symbol)

            # ------------------------------------------------------------------
            # 0) Default liquidity-specific configuration with data-driven regimes
            # ------------------------------------------------------------------
            liquidity_defaults: Dict[str, Any] = {
                "liquidity_output_timeframe": "15m",
                "liquidity_prob_interpolation_mode": "step",  # 'step' or 'linear'
                # Use fast mode for cluster quality assessment by default to
                # avoid O(n^2) silhouette calculations in large blank-mode
                # runs. Comprehensive assessments can override this to False.
                "liquidity_quality_fast_mode": True,
                "liquidity_min_samples": 200,
                "liquidity_train_fraction": 0.8,
                "liquidity_use_ewm_features": True,
                "liquidity_enable_prob_calibration": True,
                "liquidity_enable_hpo": False,
                # Canonical best-performing configuration (see outcomes/liquidity_best_config_*)
                "liquidity_n_regimes": 5,
                "liquidity_regime_detection_method": "wcv",  # wcv, silhouette, elbow, gap_statistic
                "liquidity_min_regime_size": 0.03,
                "liquidity_max_regimes": 8,  # Upper bound for auto-detection
                "liquidity_regime_stability_threshold": 0.6,
                # Centroid refinement settings
                "liquidity_enable_centroid_refinement": True,
                "liquidity_centroid_iterations": 2,
                "liquidity_centroid_min_improvement": 0.05,
                # Temporal context and winsorization tuned around best config
                "liquidity_rvol_lookback_24": 96,
                "liquidity_rvol_lookback_168": 672,
                "liquidity_range_std_lookback": 192,
                "liquidity_winsor_lower": 0.005,
                "liquidity_winsor_upper": 0.975,
                # Temporal smoothing of liquidity regimes to reduce flicker
                "liquidity_enable_smoothing": True,
                "liquidity_smoothing_window": 5,
                "liquidity_smoothing_min_run": 2,
                "liquidity_use_smoothed_for_training": True,
            }
            for k, v in liquidity_defaults.items():
                config.setdefault(k, v)

            # Map high-level pruning strategy to concrete auto-prune settings.
            # Users can still override low-level thresholds directly if needed.
            prune_strategy = str(config.get("liquidity_prune_strategy", "none")).lower()
            if prune_strategy == "none":
                if "liquidity_auto_prune_features" not in config:
                    config["liquidity_auto_prune_features"] = False
            else:
                # Enable auto-pruning and apply strategy-specific defaults
                config.setdefault("liquidity_auto_prune_features", True)

                if prune_strategy == "light":
                    config.setdefault("liquidity_prune_min_gain_normalized", 0.03)
                    config.setdefault("liquidity_prune_neg_corr_threshold", -0.15)
                    config.setdefault("liquidity_prune_max_features", 5)
                elif prune_strategy == "moderate":
                    config.setdefault("liquidity_prune_min_gain_normalized", 0.02)
                    config.setdefault("liquidity_prune_neg_corr_threshold", -0.10)
                    config.setdefault("liquidity_prune_max_features", 10)
                elif prune_strategy == "aggressive":
                    config.setdefault("liquidity_prune_min_gain_normalized", 0.01)
                    config.setdefault("liquidity_prune_neg_corr_threshold", -0.05)
                    config.setdefault("liquidity_prune_max_features", 15)

            tprint_info(
                f"🚀 Starting {self.step_name} for {symbol} on {exchange} "
                f"(regime_timeframe={regime_timeframe})"
            )

            # ------------------------------------------------------------------
            # 1) Load OHLCV market data for regime_timeframe
            # ------------------------------------------------------------------
            self.set_context(
                symbol=symbol,
                exchange=exchange,
                timeframe=regime_timeframe,
                direction=direction,
                model="liquidity_regime",
                execution_mode=execution_mode,
            )

            market_data_1h, market_source = self.load_market_data_or_fail(
                {**config, "timeframe": regime_timeframe},
                pipeline_state={},
                allow_config_override=True,
            )

            if not isinstance(market_data_1h, pd.DataFrame) or market_data_1h.empty:
                raise ValueError("Loaded market data is empty or not a DataFrame")

            if not isinstance(market_data_1h.index, pd.DatetimeIndex):
                try:
                    market_data_1h = market_data_1h.copy()
                    # Handle tz-aware objects by converting to UTC then dropping tz
                    try:
                        market_data_1h.index = pd.to_datetime(market_data_1h.index)
                    except (TypeError, ValueError):
                        market_data_1h.index = pd.to_datetime(market_data_1h.index, utc=True)
                        market_data_1h.index = market_data_1h.index.tz_convert(None)
                except Exception as exc:
                    raise ValueError("Market data index could not be converted to DatetimeIndex") from exc

            tprint_info(
                f"✅ Loaded {regime_timeframe} market data from {market_source}: {market_data_1h.shape} "
                f"({market_data_1h.index.min()} → {market_data_1h.index.max()})"
            )

            # Create temporal split config with 6-month burn-in for indicator stabilization
            split_config = create_temporal_split_config_for_pipeline(
                symbol=symbol,
                exchange=exchange,
                timeframe=regime_timeframe,
                data_start=market_data_1h.index.min(),
                data_end=market_data_1h.index.max(),
                enable_burnin=True,
                # Use default burnin_pct=1/12 (3 months)
            )
            tprint_info(
                f"📅 Temporal split config: "
                f"burn-in={split_config.burnin.start if split_config.burnin else 'None'} → "
                f"{split_config.burnin.effective_end if split_config.burnin else 'None'}, "
                f"train={split_config.training.start} → {split_config.training.effective_end}, "
                f"val={split_config.validation.start} → {split_config.validation.effective_end}, "
                f"test={split_config.test.start} → {split_config.test.effective_end}"
            )

            # ------------------------------------------------------------------
            # 2) Generate liquidity features on regime_timeframe grid
            #    Use the full feature generator so all core scaled features
            #    required by hierarchical regime assignment are present.
            #    For HPO-style workflows, cache the feature matrix per
            #    symbol/timeframe within this run so that multiple configs
            #    reuse the same features instead of recomputing them.
            # ------------------------------------------------------------------
            cache_key = (symbol, exchange, regime_timeframe)
            liquidity_df = getattr(self, "_liquidity_feature_cache", {}).get(cache_key)
            if liquidity_df is None:
                liquidity_df = self._generate_liquidity_features(market_data_1h, config)
                if not hasattr(self, "_liquidity_feature_cache"):
                    self._liquidity_feature_cache = {}
                self._liquidity_feature_cache[cache_key] = liquidity_df

            if "dollar_volume" in liquidity_df.columns:
                liquidity_df = liquidity_df.drop(columns=["dollar_volume"])

            # ------------------------------------------------------------------
            # 3) Construct semantic liquidity regimes using data-driven detection
            #    Cache hierarchical regimes per (symbol, timeframe, config
            #    regime method / n_regimes) so HPO runs with different model
            #    hyperparameters can reuse the same teacher labels.
            # ------------------------------------------------------------------
            n_regimes_config = config.get("liquidity_n_regimes", "auto")
            
            if n_regimes_config == "auto":
                # Data-driven regime detection
                optimal_n_regimes = self._detect_optimal_regime_count(liquidity_df, config)
                config["liquidity_n_regimes"] = optimal_n_regimes
                tprint_info(f"🎯 Data-driven regime detection: optimal_n_regimes={optimal_n_regimes}")
            else:
                optimal_n_regimes = int(n_regimes_config)
                tprint_info(f"📊 Using fixed regime count: {optimal_n_regimes}")
            
            regime_cache_key = (
                symbol,
                exchange,
                regime_timeframe,
                str(config.get("liquidity_regime_detection_method", "silhouette")),
                str(config.get("liquidity_n_regimes", n_regimes_config)),
            )
            cached_regimes = getattr(self, "_liquidity_regime_cache", {}).get(regime_cache_key)
            if cached_regimes is not None and len(cached_regimes) == len(liquidity_df):
                liquidity_df["liquidity_regime"] = cached_regimes
            else:
                liquidity_df = self._hierarchical_regime_assignment(liquidity_df, config)
                if not hasattr(self, "_liquidity_regime_cache"):
                    self._liquidity_regime_cache = {}
                self._liquidity_regime_cache[regime_cache_key] = liquidity_df["liquidity_regime"].copy()

            # ------------------------------------------------------------------
            # 3b) Refine regimes with centroid-based clustering
            # ------------------------------------------------------------------
            refine_enabled = bool(config.get("liquidity_enable_centroid_refinement", True))
            if refine_enabled:
                tprint_info("🔄 Refining regime assignments using centroid clustering...")

                # Top discriminative features for distance calculation
                refinement_features = [
                    # Core dimensions
                    "rvol_24_scaled", "rvol_168_scaled", "vol_z_24", "rvol_20",
                    "delta_regime_signal_scaled", "volume_direction_conviction",
                    "amihud_spike_ratio_scaled", "volume_efficiency_ratio",
                    # Trend persistence
                    "consecutive_direction_ratio_6h", "trend_confirmation_6h",
                    # Reversal patterns
                    "whipsaw_count", "reversal_intensity",
                    # Interaction features
                    "volume_range_interaction", "trend_strength", "trap_indicator",
                    # Tier 1 features
                    "rolls_spread", "breakout_failure_rate", "cumulative_delta_divergence",
                    "order_flow_persistence", "volume_depth_ratio",
                    # Tier 2 features
                    "parkinsons_volatility", "vwap_distance", "kyles_lambda_enhanced",
                    "trap_score", "vol_of_vol",
                ]

                refined_labels = self._refine_regimes_with_centroids(
                    df=liquidity_df,
                    regime_labels=liquidity_df["liquidity_regime"],
                    feature_cols=refinement_features,
                    n_iterations=int(config.get("liquidity_centroid_iterations", 3)),
                    min_distance_improvement=float(config.get("liquidity_centroid_min_improvement", 0.05)),
                )

                liquidity_df["liquidity_regime"] = refined_labels

                # Print refined regime distribution
                refined_counts = liquidity_df["liquidity_regime"].value_counts().sort_index()
                tprint_info(f"Refined regime distribution:\n{refined_counts}")

                # Compute regime confidence scores
                regime_confidence = self._compute_regime_confidence(
                    df=liquidity_df,
                    regime_labels=liquidity_df["liquidity_regime"],
                    feature_cols=refinement_features,
                )

                # Store raw confidence and also provide a min-max normalized
                # version in [0, 1] so thresholds remain meaningful even when
                # raw distances produce very small confidence values.
                try:
                    conf_raw = regime_confidence.astype(float)
                    liquidity_df["regime_confidence_raw"] = conf_raw

                    if conf_raw.notna().sum() > 0:
                        conf_min = float(conf_raw.min())
                        conf_max = float(conf_raw.max())
                        if conf_max > conf_min:
                            conf_norm = (conf_raw - conf_min) / (conf_max - conf_min)
                        else:
                            conf_norm = conf_raw * 0.0
                        liquidity_df["regime_confidence"] = conf_norm
                        tprint_info(
                            f"Regime confidence (normalized): mean={conf_norm.mean():.3f}, "
                            f"median={conf_norm.median():.3f}; raw_mean={conf_raw.mean():.3f}"
                        )
                    else:
                        liquidity_df["regime_confidence"] = regime_confidence
                        tprint_info("Regime confidence series empty; no normalization applied")
                except Exception as conf_exc:
                    tprint_warning(f"Regime confidence normalization failed, using raw values: {conf_exc}")
                    liquidity_df["regime_confidence"] = regime_confidence

            # ------------------------------------------------------------------
            # 4) Train XGBClassifier on liquidity regimes
            #    Optionally run an automatic feature pruning pass using
            #    regime_confidence correlations and feature importance.
            # ------------------------------------------------------------------
            auto_prune_enabled = False

            n_regimes_cfg = int(config.get("liquidity_n_regimes", 5))
            training_metrics: Dict[str, Any] = {}
            model = None
            feature_pipeline_artifacts = None
            # Optional temporal smoothing to reduce regime flicker before
            # training and quality assessment.
            use_smoothing = bool(config.get("liquidity_enable_smoothing", True))
            regime_col = "liquidity_regime"
            if use_smoothing:
                try:
                    window = int(config.get("liquidity_smoothing_window", 5))
                    min_run = int(config.get("liquidity_smoothing_min_run", 2))
                    smoothed_labels = self._smooth_regime_labels(
                        labels=liquidity_df[regime_col].astype(int),
                        window=window,
                        min_run=min_run,
                    )
                    liquidity_df["liquidity_regime_smoothed"] = smoothed_labels
                    if bool(config.get("liquidity_use_smoothed_for_training", True)):
                        regime_col = "liquidity_regime_smoothed"
                    tprint_info(
                        f"Applied liquidity regime smoothing: window={window}, min_run={min_run}"
                    )
                except Exception as smooth_exc:
                    tprint_warning(
                        f"Liquidity regime smoothing failed (non-fatal): {smooth_exc}"
                    )

            regime_labels = liquidity_df[regime_col].astype(int)

            proba_df = self._compute_regime_probabilities(
                df=liquidity_df,
                regime_labels=regime_labels,
                feature_cols=refinement_features,
                n_regimes=n_regimes_cfg,
            )

            probs_15m_path: Optional[str] = None

            for lbl in range(n_regimes_cfg):
                p_col = f"p_regime_{lbl}"
                if p_col not in proba_df.columns:
                    proba_df[p_col] = 0.0

            for lbl in range(n_regimes_cfg):
                src_col = f"p_regime_{lbl}"
                dst_col = f"liquidity_regime_{lbl}_prob"
                proba_df[dst_col] = proba_df[src_col]

            for col in proba_df.columns:
                liquidity_df[col] = proba_df[col]

            # Map 1h probabilities to the 15m grid and persist a dedicated
            # ml_liquidity_regime_probs_15m artifact for downstream
            # consumers (get_specialist_models_outputs,
            # specialist_feature_diagnostics, unified models). This keeps the
            # training dataset as-is while exposing a compact probabilities
            # view aligned to the base timeframe.
            try:
                mapped_probs = self._map_probabilities_to_15m(
                    proba_df=proba_df,
                    market_data_1h=market_data_1h,
                    symbol=symbol,
                    exchange=exchange,
                    direction=direction,
                    config=config,
                )

                if mapped_probs is not None and not mapped_probs.empty:
                    to_save_probs = mapped_probs.reset_index().rename(
                        columns={mapped_probs.index.name or "index": "timestamp"}
                    )

                    probs_15m_path = self._save_artifact(
                        data=to_save_probs,
                        artifact_name="ml_liquidity_regime_probs_15m",
                        artifact_type="data",
                        data_category="features",
                        metadata={
                            "symbol": symbol,
                            "exchange": exchange,
                            "timeframe": str(
                                config.get("liquidity_output_timeframe", "15m")
                            ),
                            "source_regime_timeframe": regime_timeframe,
                        },
                    )
            except Exception as map_exc:
                tprint_warning(
                    f"Liquidity 1h→15m probability mapping/save failed (non-fatal): {map_exc}"
                )

            # ------------------------------------------------------------------
            # 5) Assess regime quality
            #   a) Generic cluster quality via ClusterQualityAssessor (original labels)
            #   b) Generic cluster quality on XGB-implied regimes (argmax probs)
            #   c) Liquidity-specific quality via LiquidityClusterQualityAssessor
            # ------------------------------------------------------------------
            liquidity_quality_metrics: Optional[ClusterQualityMetrics] = None
            liquidity_quality_path: Optional[str] = None

            try:
                liquidity_quality_metrics, liquidity_quality_path = self._assess_liquidity_regime_quality(
                    liquidity_df=liquidity_df,
                    regime_col="liquidity_regime",
                    config=config,
                )
            except Exception as quality_exc:
                tprint_warning(f"Liquidity regime quality assessment failed: {quality_exc}")

            # Liquidity-specific quality assessment & reports
            liquidity_cluster_metrics: Optional[LiquidityClusterQualityMetrics] = None
            liquidity_cluster_md_path: Optional[str] = None
            liquidity_cluster_csv_path: Optional[str] = None
            liquidity_cluster_metrics_path: Optional[str] = None
            liquidity_tree_params_path: Optional[str] = None

            try:
                # One-step forward returns as secondary diagnostic
                forward_returns_1h = liquidity_df.get("return_1h")
                if forward_returns_1h is not None:
                    forward_returns_1h = forward_returns_1h.shift(-1)

                assessor = LiquidityClusterQualityAssessor(config=config)
                # (a) Liquidity-specific quality on original hierarchical regimes
                liquidity_cluster_metrics = assessor.assess_liquidity_clusters(
                    liquidity_df=liquidity_df,
                    regime_labels=liquidity_df["liquidity_regime"].astype(int),
                    forward_returns_1h=forward_returns_1h,
                    config=config,
                )

                # Expose CoV-based and overall quality metrics for multi-criteria selection
                training_metrics["liquidity_effort_result_cov_separation_score"] = float(
                    liquidity_cluster_metrics.effort_result_cov_separation_score
                )
                training_metrics["liquidity_returns_cov_separation_score"] = float(
                    liquidity_cluster_metrics.returns_cov_separation_score
                )
                training_metrics["liquidity_overall_quality_score"] = float(
                    liquidity_cluster_metrics.overall_quality_score
                )
                # Also expose the number of regimes discovered for reporting/HPO
                try:
                    training_metrics["n_regimes"] = int(liquidity_cluster_metrics.n_regimes)
                except Exception:
                    pass

                # Aggregate forward 1h returns across regimes as additional
                # diagnostics for HPO/analysis. We surface both a
                # sample-weighted mean across all regimes and the best
                # per-regime mean forward return.
                try:
                    per_regime = liquidity_cluster_metrics.per_regime_metrics or {}
                    weighted_sum = 0.0
                    sample_sum = 0.0
                    best_forward = None

                    for _, reg_data in per_regime.items():
                        fr = reg_data.get("forward_return_mean")
                        n = reg_data.get("n_samples")
                        if isinstance(fr, (int, float)) and isinstance(n, (int, float)):
                            fr_val = float(fr)
                            n_val = float(n)
                            weighted_sum += fr_val * n_val
                            sample_sum += n_val
                            if best_forward is None or fr_val > best_forward:
                                best_forward = fr_val

                    if sample_sum > 0.0 and best_forward is not None:
                        training_metrics["forward_return_mean_1h_weighted"] = float(
                            weighted_sum / sample_sum
                        )
                        training_metrics["forward_return_mean_1h_best_regime"] = float(best_forward)
                except Exception:
                    pass

                # Calculate HSIC for Regime Independence from Target (Validation)
                try:
                    # Use forward 1h returns as validation target
                    hsic_target = liquidity_df.get("return_1h").shift(-1)
                    hsic_regimes = liquidity_df["liquidity_regime"]

                    # Align mask
                    mask = hsic_target.notna() & hsic_regimes.notna() & (hsic_regimes >= 0)

                    if mask.sum() > 100:
                        hsic_score = calculate_hsic(
                            hsic_regimes[mask].values,
                            hsic_target[mask].values,
                            kernel_X='delta', # Categorical
                            kernel_Y='rbf'    # Continuous
                        )
                        training_metrics["liquidity_regime_hsic"] = float(hsic_score)
                        tprint_info(f"  HSIC(Regime, FwdReturn1h): {hsic_score:.6f}")
                except Exception as hsic_exc:
                    tprint_warning(f"HSIC calculation failed: {hsic_exc}")

                # Generate human-readable reports in outcomes/
                liquidity_cluster_md_path = assessor.save_markdown_report(
                    metrics=liquidity_cluster_metrics,
                    symbol=symbol,
                    output_dir="outcomes",
                )
                liquidity_cluster_csv_path = assessor.save_csv_report(
                    metrics=liquidity_cluster_metrics,
                    symbol=symbol,
                    output_dir="outcomes",
                )
                feature_distinctiveness_path = assessor.save_feature_distinctiveness_report(
                    metrics=liquidity_cluster_metrics,
                    symbol=symbol,
                    output_dir="outcomes",
                )

                # Persist metrics as versioned artifact
                try:
                    metrics_dict = {
                        "effort_result_separation_score": liquidity_cluster_metrics.effort_result_separation_score,
                        "ghost_vs_valid_contrast": liquidity_cluster_metrics.ghost_vs_valid_contrast,
                        "absorption_vs_valid_contrast": liquidity_cluster_metrics.absorption_vs_valid_contrast,
                        "effort_result_cov_separation_score": liquidity_cluster_metrics.effort_result_cov_separation_score,
                        "returns_cov_separation_score": liquidity_cluster_metrics.returns_cov_separation_score,
                        "ghost_reversal_rate": liquidity_cluster_metrics.ghost_reversal_rate,
                        "ghost_false_trend_rate": liquidity_cluster_metrics.ghost_false_trend_rate,
                        "absorption_reversal_rate": liquidity_cluster_metrics.absorption_reversal_rate,
                        "absorption_follow_through_rate": liquidity_cluster_metrics.absorption_follow_through_rate,
                        "valid_trend_follow_through": liquidity_cluster_metrics.valid_trend_follow_through,
                        "apathy_noise_fraction": liquidity_cluster_metrics.apathy_noise_fraction,
                        "class_balance_score": liquidity_cluster_metrics.class_balance_score,
                        "n_regimes": liquidity_cluster_metrics.n_regimes,
                        "n_samples": liquidity_cluster_metrics.n_samples,
                        "per_regime_metrics": liquidity_cluster_metrics.per_regime_metrics,
                        "overall_quality_score": liquidity_cluster_metrics.overall_quality_score,
                        "assessment_timestamp": liquidity_cluster_metrics.assessment_timestamp,
                    }

                    liquidity_cluster_metrics_path = self._save_artifact(
                        data=metrics_dict,
                        artifact_name="ml_liquidity_cluster_quality_metrics_15m",
                        artifact_type="data",
                        metadata={
                            "overall_quality_score": liquidity_cluster_metrics.overall_quality_score,
                            "n_regimes": liquidity_cluster_metrics.n_regimes,
                            "assessment_timestamp": liquidity_cluster_metrics.assessment_timestamp,
                        },
                    )

                    tree_params = {
                        "symbol": symbol,
                        "exchange": exchange,
                        "timeframe": regime_timeframe,
                        "n_regimes": int(liquidity_cluster_metrics.n_regimes),
                        "volume_threshold": float(config.get("liquidity_tree_volume_threshold", 0.0)),
                        "delta_threshold": float(config.get("liquidity_tree_delta_threshold", 0.0)),
                        "range_threshold": float(config.get("liquidity_tree_range_threshold", 0.0)),
                        "amihud_threshold": float(config.get("liquidity_tree_amihud_threshold", 0.0)),
                    }

                    liquidity_tree_params_path = self._save_artifact(
                        data=tree_params,
                        artifact_name="ml_liquidity_regime_tree_15m",
                        artifact_type="data",
                        metadata={
                            "symbol": symbol,
                            "exchange": exchange,
                            "timeframe": regime_timeframe,
                            "n_regimes": int(liquidity_cluster_metrics.n_regimes),
                        },
                    )
                except Exception as save_metrics_exc:
                    tprint_error(f"Failed to save liquidity cluster quality metrics artifact: {save_metrics_exc}")
                    training_metrics["liquidity_metrics_save_error"] = str(save_metrics_exc)
                    
            except Exception as liquidity_cluster_exc:
                error_msg = f"Liquidity-specific cluster quality assessment failed: {liquidity_cluster_exc}"
                tprint_error(error_msg)
                training_metrics["liquidity_quality_assessment_error"] = str(liquidity_cluster_exc)
                training_metrics["liquidity_quality_assessment_failed"] = True
                
                # Continue execution but mark as failed
                liquidity_cluster_metrics = None

            # ------------------------------------------------------------------
            # 6) Save 1h training artifacts
            # ------------------------------------------------------------------
            liquidity_to_save = liquidity_df.reset_index().rename(
                columns={liquidity_df.index.name or "index": "timestamp"}
            )

            tprint_info(
                f"💾 Saving liquidity training dataset with shape {liquidity_to_save.shape} "
                f"to versioned HDF5 store"
            )
            # Build metadata with temporal split information
            metadata = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": regime_timeframe,
                "source_market_data": market_source,
                "version": "v2_with_burnin",
                "training_start": str(split_config.training.start),
                "training_end": str(split_config.training.effective_end),
                "validation_start": str(split_config.validation.start),
                "validation_end": str(split_config.validation.effective_end),
                "test_start": str(split_config.test.start),
                "test_end": str(split_config.test.effective_end),
            }
            if split_config.burnin is not None:
                metadata["burnin_start"] = str(split_config.burnin.start)
                metadata["burnin_end"] = str(split_config.burnin.effective_end)

            training_data_path = self._save_artifact(
                data=liquidity_to_save,
                artifact_name="ml_liquidity_training_data_15m",
                artifact_type="data",
                metadata=metadata,
            )

            execution_time = time.time() - start_time
            tprint_info(
                f" {self.step_name} completed in {execution_time:.2f}s "
                f"with {len(liquidity_df)} samples"
            )

            return {
                "success": True,
                "artifacts": {
                    "liquidity_training_data": liquidity_df,
                    "liquidity_training_data_path": training_data_path,
                    "liquidity_quality_metrics": liquidity_quality_metrics,
                    "liquidity_quality_path": liquidity_quality_path,
                    "liquidity_probs_15m_path": probs_15m_path,
                    "liquidity_tree_params_path": liquidity_tree_params_path,
                },
                "metrics": training_metrics,
                "execution_time": execution_time,
            }

        except Exception as exc:
            execution_time = time.time() - start_time
            error_msg = f"{self.step_name} failed: {exc}"
            self.logger.error(error_msg, exc_info=True)
            tprint_error(error_msg)
            return {
                "success": False,
                "artifacts": {},
                "metrics": {},
                "error": str(exc),
                "execution_time": execution_time,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _smooth_regime_labels(
        self,
        labels: pd.Series,
        window: int = 5,
        min_run: int = 2,
    ) -> pd.Series:
        """Apply simple temporal smoothing to integer regime labels.

        Uses a causal moving-window majority vote and a minimum run-length
        filter to reduce one-bar flicker while preserving regime structure.
        """
        if not isinstance(labels, pd.Series) or labels.empty:
            return labels

        try:
            window = max(int(window), 1)
        except Exception:
            window = 5
        try:
            min_run = max(int(min_run), 1)
        except Exception:
            min_run = 2

        values = labels.to_numpy(copy=True)
        n = len(values)
        smoothed = values.copy()

        # Causal moving-window majority vote
        for i in range(n):
            start = 0 if i + 1 <= window else i + 1 - window
            window_vals = values[start : i + 1]
            if window_vals.size == 0:
                continue
            unique_vals, counts = np.unique(window_vals, return_counts=True)
            smoothed[i] = unique_vals[int(np.argmax(counts))]

        # Minimum run-length filter to remove short flickers
        i = 0
        while i < n:
            j = i + 1
            while j < n and smoothed[j] == smoothed[i]:
                j += 1
            run_len = j - i
            if run_len < min_run:
                if i > 0:
                    replacement = smoothed[i - 1]
                elif j < n:
                    replacement = smoothed[j]
                else:
                    replacement = smoothed[i]
                smoothed[i:j] = replacement
            i = j

        return pd.Series(smoothed, index=labels.index, dtype=labels.dtype)

    def _generate_liquidity_features_vectorized(self, market_data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Vectorized feature generation with hardware acceleration."""
        df = market_data.copy()
        
        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required OHLCV columns for liquidity features: {missing}")
        
        eps = 1e-9
        
        # Use hardware-accelerated operations if available
        if self.vectorized_ops is not None:
            return self._generate_features_with_vectorization(df, config, eps)
        else:
            return self._generate_features_fallback(df, config, eps)
    
    def _generate_features_with_vectorization(self, df: pd.DataFrame, config: Dict[str, Any], eps: float) -> pd.DataFrame:
        """Generate features using vectorized operations."""
        # Memory monitoring
        if self.hardware_manager:
            self.hardware_manager.monitor_memory_usage("feature_generation_start")
        
        # Vectorized basic calculations
        df["range"] = (df["high"] - df["low"]).astype(float)
        df["range"] = df["range"].replace(0, np.nan)
        df["return_1h"] = np.log(df["close"] / df["close"].shift(1)).astype(float)
        df["abs_return_1h"] = df["return_1h"].abs()
        df["dollar_volume"] = (df["close"] * df["volume"]).astype(float)
        
        # Vectorized rolling operations using optimized chunks
        vol_window_daily = int(config.get("liquidity_rvol_lookback_24", 96))
        vol_window_weekly = int(config.get("liquidity_rvol_lookback_168", 672))
        
        # Use vectorized rolling operations
        rolling_ops = {
            f"vol_sma_{vol_window_daily}": ("volume", vol_window_daily, "mean"),
            f"vol_sma_{vol_window_weekly}": ("volume", vol_window_weekly, "mean"),
            "vol_sma_20": ("volume", 80, "mean"),
        }
        
        for col_name, (source_col, window, operation) in rolling_ops.items():
            if self.vectorized_ops:
                df[col_name] = self.vectorized_ops.vectorized_rolling(
                    df[source_col].values, window, operation
                )
            else:
                df[col_name] = df[source_col].rolling(window, min_periods=max(1, window//5)).mean()
        
        # Vectorized ratio calculations
        df["rvol_24"] = df["volume"] / (df["vol_sma_24"] + eps)
        df["rvol_168"] = df["volume"] / (df["vol_sma_168"] + eps)
        df["rvol_20"] = df["volume"] / (df["vol_sma_20"] + eps)
        
        # Continue with vectorized implementations for all features...
        # (This is a partial implementation - full implementation would include all 60+ features)
        
        # Memory checkpoint
        if self.hardware_manager:
            self.hardware_manager.monitor_memory_usage("feature_generation_mid")
        
        # Generate remaining features using vectorized operations
        df = self._generate_volume_efficiency_features(df, eps)
        df = self._generate_volatility_features(df, config, eps)
        df = self._generate_directional_features(df, eps)
        df = self._generate_trend_persistence_features(df, eps)
        df = self._generate_interaction_features(df, eps)
        df = self._generate_tier1_features(df, eps)
        df = self._generate_tier2_features(df, eps)
        
        # Final memory checkpoint
        if self.hardware_manager:
            self.hardware_manager.monitor_memory_usage("feature_generation_end")
        
        return df
    
    def _generate_features_fallback(self, df: pd.DataFrame, config: Dict[str, Any], eps: float) -> pd.DataFrame:
        """Fallback feature generation without vectorization."""
        tprint_warning("Using fallback feature generation (vectorization unavailable)")
        return self._generate_liquidity_features_original(df, config, eps)
    
    def _generate_liquidity_features_original(self, df: pd.DataFrame, config: Dict[str, Any], eps: float) -> pd.DataFrame:
        """Original feature generation method (preserved for fallback)."""
        # Call the existing original method
        return self._generate_liquidity_features_legacy(df, config)
    
    def _generate_liquidity_features_legacy(self, market_data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Legacy feature generation method - preserves original implementation."""
        # This method contains all the original feature generation logic
        # For now, we'll implement a basic version to ensure compatibility
        df = market_data.copy()
        eps = 1e-9
        
        # Basic derived quantities (original implementation)
        df["range"] = (df["high"] - df["low"]).astype(float)
        df["range"] = df["range"].replace(0, np.nan)
        df["return_1h"] = np.log(df["close"] / df["close"].shift(1)).astype(float)
        df["abs_return_1h"] = df["return_1h"].abs()
        df["dollar_volume"] = (df["close"] * df["volume"]).astype(float)
        
        # Relative volume context (ensure rvol_* exist for downstream usage)
        vol_window_daily = int(config.get("liquidity_rvol_lookback_24", 24))
        vol_window_weekly = int(config.get("liquidity_rvol_lookback_168", 168))

        df["vol_sma_24"] = df["volume"].rolling(vol_window_daily, min_periods=5).mean()
        df["vol_sma_168"] = df["volume"].rolling(vol_window_weekly, min_periods=20).mean()
        df["vol_sma_20"] = df["volume"].rolling(20, min_periods=5).mean()

        df["rvol_24"] = df["volume"] / (df["vol_sma_24"] + eps)
        df["rvol_168"] = df["volume"] / (df["vol_sma_168"] + eps)
        df["rvol_20"] = df["volume"] / (df["vol_sma_20"] + eps)
        
        # Generate all the original features using the new vectorized methods
        # This ensures compatibility while leveraging optimizations
        df = self._generate_volume_efficiency_features(df, eps)
        df = self._generate_volatility_features(df, config, eps)
        df = self._generate_directional_features(df, eps)
        df = self._generate_trend_persistence_features(df, eps)
        df = self._generate_interaction_features(df, eps)
        df = self._generate_tier1_features(df, eps)
        df = self._generate_tier2_features(df, eps)
        
        # Add missing features that weren't covered in vectorized methods
        # Candle geometry
        df["clv"] = (df["close"] - df["low"]) / (df["range"] + eps) - 0.5
        upper_wick = (df["high"] - df[["close", "open"]].max(axis=1)).clip(lower=0)
        lower_wick = (df[["close", "open"]].min(axis=1) - df["low"]).clip(lower=0)
        df["wick_ratio"] = np.maximum(upper_wick, lower_wick) / (df["range"] + eps)
        df["body_dominance"] = (df["close"] - df["open"]) / (df["range"] + eps)
        df["gap_factor"] = (df["open"] - df["close"].shift(1)) / (df["close"].shift(1) + eps)
        
        # Intraday vs closing volatility feature
        df["intraday_close_ratio"] = df["range"] / (df["abs_return_1h"].replace(0, np.nan) + eps)
        
        # Ease of Movement (EMV)
        mid_price = (df["high"] + df["low"]) / 2.0
        mid_price_prev = mid_price.shift(1)
        df["emv"] = (mid_price - mid_price_prev) / ((df["volume"] / (df["range"] + eps)) + eps)
        
        # Winsorize heavy tails for stability
        winsor_lower = float(config.get("liquidity_winsor_lower", 0.005))
        winsor_upper = float(config.get("liquidity_winsor_upper", 0.995))
        winsor_cols = [
            "rvol_24", "rvol_168", "vol_z_24", "normalized_range",
            "ghost_ratio", "absorption_ratio", "amihud_validity",
            "amivest_efficiency", "emv", "intraday_close_ratio"
        ]
        for col in winsor_cols:
            if col in df.columns:
                series = df[col].dropna()
                if len(series) == 0:
                    continue
                lo = series.quantile(winsor_lower)
                hi = series.quantile(winsor_upper)
                df[col] = df[col].clip(lower=lo, upper=hi)
        
        # Generate additional features for completeness
        # CATEGORY 4: REVERSAL & TRAP PATTERNS (10 features)
        sign_changes = (np.sign(df["return_1h"]) != np.sign(df["return_1h"].shift(1))).astype(float)
        df["reversal_intensity"] = df["abs_return_1h"] * sign_changes
        df["reversal_intensity_ewm3"] = df["reversal_intensity"].ewm(span=3, adjust=False).mean()
        
        df["reversal_conviction"] = (
            (np.sign(df["return_1h"]) == np.sign(df["return_1h"].shift(1))).astype(float)
            .rolling(window=24, min_periods=2).sum() / 24.0
        )
        df["reversal_conviction_ewm3"] = df["reversal_conviction"].ewm(span=3, adjust=False).mean()
        
        df["whipsaw_count"] = sign_changes.rolling(window=48, min_periods=4).sum()
        df["whipsaw_count_ewm6"] = df["whipsaw_count"].ewm(span=6, adjust=False).mean()
        
        df["reversal_volume_sync"] = df["reversal_intensity"] * df["volume_direction_conviction"]
        df["reversal_volume_sync_ewm3"] = df["reversal_volume_sync"].ewm(span=3, adjust=False).mean()
        
        df["return_autocorr_lag6"] = df["return_1h"].rolling(window=12, min_periods=6).apply(
            lambda x: x.iloc[:6].corr(x.iloc[6:]) if len(x) == 12 else 0, raw=False
        )
        
        price_change_6h = (df["close"] - df["close"].shift(6)).abs()
        volatility_6h = df["abs_return_1h"].rolling(window=6, min_periods=2).sum()
        df["efficiency_ratio"] = price_change_6h / (volatility_6h + eps)
        
        # Additional features...
        # (Continue with remaining original features)
        
        return df
    
    def _generate_volume_efficiency_features(self, df: pd.DataFrame, eps: float) -> pd.DataFrame:
        """Vectorized volume efficiency features."""
        # Volume-Efficiency Ratio
        df["volume_efficiency_ratio"] = df["volume"] / (df["range"] + eps)
        
        # Vectorized volatility calculations
        vol_window = 96
        vol_mean = df["volume"].rolling(vol_window, min_periods=5).mean()
        vol_std = df["volume"].rolling(vol_window, min_periods=5).std()
        df["vol_z_24"] = (df["volume"] - vol_mean) / (vol_std.replace(0, np.nan) + eps)
        
        # Stability features (vectorized)
        window = 24
        df["volume_stddev_stability"] = (
            df["volume"].rolling(window, min_periods=3).std() /
            (df["volume"].rolling(window, min_periods=3).mean() + eps)
        )
        df["range_stddev_stability"] = (
            df["range"].rolling(window, min_periods=3).std() /
            (df["range"].rolling(window, min_periods=3).mean() + eps)
        )
        df["return_stddev_stability"] = (
            df["abs_return_1h"].rolling(window, min_periods=3).std() /
            (df["abs_return_1h"].rolling(window, min_periods=3).mean() + eps)
        )
        
        return df
    
    def _generate_volatility_features(self, df: pd.DataFrame, config: Dict[str, Any], eps: float) -> pd.DataFrame:
        """Vectorized volatility features."""
        # Normalized range
        range_std_lookback = int(config.get("liquidity_range_std_lookback", 192))
        range_std = df["range"].rolling(range_std_lookback, min_periods=10).std()
        df["normalized_range"] = df["range"] / (range_std.replace(0, np.nan) + eps)
        
        # Effort vs Result ratios
        df["normalized_volume"] = np.log1p(df["volume"])
        df["ghost_ratio"] = df["normalized_range"] / (df["normalized_volume"] + eps)
        df["absorption_ratio"] = df["normalized_volume"] / (df["normalized_range"] + eps)
        
        # Amihud and Amivest
        df["amihud_validity"] = df["abs_return_1h"] / (df["dollar_volume"] + eps)
        df["amivest_efficiency"] = df["dollar_volume"] / (df["abs_return_1h"] + eps)
        df["amihud_baseline"] = df["amihud_validity"].rolling(96, min_periods=6).median()
        df["amihud_spike_ratio"] = df["amihud_validity"] / (df["amihud_baseline"] + eps)
        
        return df
    
    def _generate_directional_features(self, df: pd.DataFrame, eps: float) -> pd.DataFrame:
        """Vectorized directional orderflow features."""
        # Close position in range
        df["close_position_range"] = (df["close"] - df["low"]) / (df["range"] + eps)
        df["close_position_range"] = df["close_position_range"].clip(0, 1)
        
        # Volume intensities
        df["volume_buyer_intensity"] = df["volume"] * df["close_position_range"]
        df["volume_seller_intensity"] = df["volume"] * (1.0 - df["close_position_range"])
        
        # Directional imbalance
        total_dir_volume = df["volume_buyer_intensity"] + df["volume_seller_intensity"]
        df["volume_direction_imbalance"] = (
            (df["volume_buyer_intensity"] - df["volume_seller_intensity"]) / (total_dir_volume + eps)
        )
        df["volume_direction_conviction"] = df["volume_direction_imbalance"].abs()
        
        # Direction changes and consistency
        df["direction_change"] = (
            (df["close"] > df["close"].shift(1)).astype(float) -
            (df["close"] < df["close"].shift(1)).astype(float)
        )
        df["volume_direction_consistency"] = (
            df["volume_direction_imbalance"] * df["direction_change"]
        )
        
        # Delta alignment
        price_direction = np.sign(df["close"] - df["open"])
        delta_direction = np.sign(df["volume_direction_imbalance"])
        df["delta_alignment"] = price_direction * delta_direction
        df["delta_alignment_3h"] = df["delta_alignment"].rolling(window=12, min_periods=1).mean()
        df["delta_regime_signal"] = (
            df["volume_direction_conviction"] * df["delta_alignment_3h"]
        )
        
        return df
    
    def _generate_trend_persistence_features(self, df: pd.DataFrame, eps: float) -> pd.DataFrame:
        """Vectorized trend persistence features."""
        direction_sign = np.sign(df["return_1h"])
        direction_sign = direction_sign.replace(0, np.nan)
        
        # Consecutive direction ratios
        same_direction_3 = (direction_sign == direction_sign.shift(1)).astype(float)
        df["consecutive_direction_ratio_3h"] = same_direction_3.rolling(window=12, min_periods=1).sum() / 12.0
        df["consecutive_direction_ratio_3h_ewm3"] = (
            df["consecutive_direction_ratio_3h"].ewm(span=3, adjust=False).mean()
        )
        
        same_direction_6 = (direction_sign == direction_sign.shift(1)).astype(float)
        df["consecutive_direction_ratio_6h"] = same_direction_6.rolling(window=24, min_periods=1).sum() / 24.0
        df["consecutive_direction_ratio_6h_ewm3"] = (
            df["consecutive_direction_ratio_6h"].ewm(span=3, adjust=False).mean()
        )
        
        # Momentum persistence
        momentum_ma_3 = df["return_1h"].rolling(window=12, min_periods=2).mean()
        momentum_ma_6 = df["return_1h"].rolling(window=24, min_periods=2).mean()
        df["momentum_persistence_3h"] = (momentum_ma_3 - momentum_ma_6) / (abs(momentum_ma_6) + eps)
        df["momentum_persistence_3h_ewm6"] = (
            df["momentum_persistence_3h"].ewm(span=6, adjust=False).mean()
        )
        
        # Trend confirmation
        df["trend_confirmation_3h"] = (
            df["consecutive_direction_ratio_3h"] * df["volume_direction_conviction"]
        )
        df["trend_confirmation_6h"] = (
            df["consecutive_direction_ratio_6h"] * df["volume_direction_conviction"]
        )
        
        return df
    
    def _generate_interaction_features(self, df: pd.DataFrame, eps: float) -> pd.DataFrame:
        """Vectorized interaction features."""
        # Core interactions
        df["volume_range_interaction"] = df["rvol_24"] * df["normalized_range"]
        df["trend_strength"] = (
            df["volume_direction_conviction"] * df["consecutive_direction_ratio_6h"]
        )
        df["trap_indicator"] = df["ghost_ratio"] * df["whipsaw_count"] if "whipsaw_count" in df.columns else 0.0
        df["absorption_signal"] = (
            df["absorption_ratio"] * (1.0 - df["delta_alignment_3h"].abs())
        )
        
        return df
    
    def _generate_tier1_features(self, df: pd.DataFrame, eps: float) -> pd.DataFrame:
        """Vectorized Tier 1 high-impact features."""
        # Roll's spread estimator
        return_autocov = -df["return_1h"].rolling(2).apply(
            lambda x: x.iloc[0] * x.iloc[1] if len(x) == 2 else 0, raw=False
        )
        df["rolls_spread"] = 2 * np.sqrt(return_autocov.clip(lower=0))
        
        # Breakout failure rate
        high_range_mask = df["normalized_range"] > df["normalized_range"].rolling(20).quantile(0.8)
        reversal_next_bar = (np.sign(df["return_1h"]) != np.sign(df["return_1h"].shift(1)))
        df["breakout_failure_rate"] = (high_range_mask & reversal_next_bar).rolling(12, min_periods=4).mean()
        
        # Cumulative delta divergence
        cumulative_volume_imbalance = df["volume_direction_imbalance"].rolling(6).sum()
        cumulative_price_change = (df["close"].diff(6) / df["close"].shift(6)).abs()
        df["cumulative_delta_divergence"] = (
            np.abs(cumulative_volume_imbalance) - cumulative_price_change
        )
        
        # Order flow persistence
        df["order_flow_persistence"] = df["volume_direction_imbalance"].rolling(3).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
        )
        
        # Volume depth ratio
        price_move_1pct = df["close"] * 0.01
        df["volume_depth_ratio"] = df["volume"] / (price_move_1pct + eps)
        
        return df
    
    def _generate_tier2_features(self, df: pd.DataFrame, eps: float) -> pd.DataFrame:
        """Vectorized Tier 2 features."""
        # Parkinson's volatility
        df["parkinsons_volatility"] = np.sqrt(
            (1 / (4 * np.log(2))) * (np.log(df["high"] / df["low"]) ** 2)
        ).rolling(24, min_periods=2).mean()
        
        # VWAP distance
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        vwap = (typical_price * df["volume"]).rolling(80, min_periods=5).sum() / (
            df["volume"].rolling(80, min_periods=5).sum() + eps
        )
        df["vwap_distance"] = (df["close"] - vwap) / (vwap + eps)
        
        # Kyle's lambda enhanced
        signed_volume = df["volume"] * np.sign(df["return_1h"])
        cumulative_signed_volume = signed_volume.rolling(24, min_periods=2).sum()
        price_change_6h = df["close"] - df["close"].shift(24)
        df["kyles_lambda_enhanced"] = price_change_6h / (cumulative_signed_volume + eps)
        
        # Trap score and vol-of-vol
        df["trap_score"] = (
            df["normalized_range"] *
            (1 / (df["rvol_20"] + eps)) *
            df["reversal_intensity"] if "reversal_intensity" in df.columns else 0.0
        )
        
        realized_vol_6h_rolling = df["return_1h"].rolling(24, min_periods=2).std()
        df["vol_of_vol"] = (
            realized_vol_6h_rolling.rolling(48, min_periods=4).std() /
            (realized_vol_6h_rolling.rolling(48, min_periods=4).mean() + eps)
        )
        
        return df
    
    def _detect_optimal_regime_count(self, df: pd.DataFrame, config: Dict[str, Any]) -> int:
        """Detect optimal number of regimes using multiple criteria."""
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            from sklearn.preprocessing import StandardScaler
            import scipy.cluster.hierarchy as sch
            from scipy.spatial.distance import pdist
        except ImportError as e:
            tprint_warning(f"Clustering libraries not available, using default 5 regimes: {e}")
            return 5
        
        # Select core features for regime detection
        core_features = [
            "rvol_24_scaled", "rvol_168_scaled", "vol_z_24", "rvol_20",
            "delta_regime_signal_scaled", "volume_direction_conviction",
            "amihud_spike_ratio_scaled", "volume_efficiency_ratio",
            "consecutive_direction_ratio_6h", "trend_confirmation_6h",
            "normalized_range", "ghost_ratio", "absorption_ratio",
            "cumulative_delta_divergence",
        ]
        
        available_features = [f for f in core_features if f in df.columns]
        if len(available_features) < 3:
            tprint_warning("Insufficient features for regime detection, using default 5")
            return 5
        
        feature_data = df[available_features].dropna()
        if len(feature_data) < 100:
            tprint_warning("Insufficient samples for regime detection, using default 5")
            return 5
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(feature_data)
        features_scaled_df = pd.DataFrame(
            features_scaled,
            index=feature_data.index,
            columns=available_features,
        )
        
        detection_method = str(config.get("liquidity_regime_detection_method", "wcv"))
        max_regimes = int(config.get("liquidity_max_regimes", 8))
        min_regime_size = float(config.get("liquidity_min_regime_size", 0.05))
        stability_threshold = float(config.get("liquidity_regime_stability_threshold", 0.7))
        
        tprint_info(f"🔍 Detecting optimal regimes using {detection_method} method...")
        
        if detection_method == "wcv":
            return self._detect_regimes_wcv(
                features_scaled_df, max_regimes, min_regime_size, stability_threshold
            )
        elif detection_method == "silhouette":
            return self._detect_regimes_silhouette(
                features_scaled_df.values, max_regimes, min_regime_size, stability_threshold
            )
        elif detection_method == "elbow":
            return self._detect_regimes_elbow(
                features_scaled_df.values, max_regimes, min_regime_size, stability_threshold
            )
        elif detection_method == "gap_statistic":
            return self._detect_regimes_gap_statistic(
                features_scaled_df.values, max_regimes, min_regime_size, stability_threshold
            )
        else:
            tprint_warning(f"Unknown detection method {detection_method}, using wcv")
            return self._detect_regimes_wcv(
                features_scaled_df, max_regimes, min_regime_size, stability_threshold
            )
    
    def _detect_regimes_silhouette(self, features_scaled: np.ndarray, max_regimes: int, 
                                 min_regime_size: float, stability_threshold: float) -> int:
        """Detect optimal regimes using silhouette analysis."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        silhouette_scores = []
        regime_counts = range(2, max_regimes + 1)
        
        for n_clusters in regime_counts:
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
                cluster_labels = kmeans.fit_predict(features_scaled)
                
                # Check minimum regime size constraint
                _, label_counts = np.unique(cluster_labels, return_counts=True)
                min_size_fraction = min(label_counts) / len(cluster_labels)
                
                if min_size_fraction < min_regime_size:
                    silhouette_scores.append(-1)  # Penalize for too small regimes
                    continue
                
                silhouette_avg = silhouette_score(features_scaled, cluster_labels)
                silhouette_scores.append(silhouette_avg)
                
            except Exception as e:
                tprint_warning(f"Silhouette analysis failed for n_clusters={n_clusters}: {e}")
                silhouette_scores.append(-1)
        
        if not silhouette_scores or max(silhouette_scores) <= 0:
            tprint_warning("Silhouette analysis failed, using default 5 regimes")
            return 5
        
        optimal_idx = np.argmax(silhouette_scores)
        optimal_n = list(regime_counts)[optimal_idx]
        
        # Apply stability threshold
        if silhouette_scores[optimal_idx] < stability_threshold:
            tprint_info(f"Silhouette score {silhouette_scores[optimal_idx]:.3f} below threshold {stability_threshold}, using default 5")
            return 5
        
        tprint_info(f"📊 Silhouette analysis: optimal_n_regimes={optimal_n}, score={silhouette_scores[optimal_idx]:.3f}")
        return optimal_n
    
    def _detect_regimes_elbow(self, features_scaled: np.ndarray, max_regimes: int,
                            min_regime_size: float, stability_threshold: float) -> int:
        """Detect optimal regimes using elbow method."""
        from sklearn.cluster import KMeans
        
        inertias = []
        regime_counts = range(1, max_regimes + 1)
        
        for n_clusters in regime_counts:
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
                kmeans.fit(features_scaled)
                inertias.append(kmeans.inertia_)
            except Exception as e:
                tprint_warning(f"Elbow analysis failed for n_clusters={n_clusters}: {e}")
                inertias.append(np.inf)
        
        if len(inertias) < 3:
            return 5
        
        # Calculate elbow point using second derivative
        inertias = np.array(inertias)
        diffs = np.diff(inertias)
        second_diffs = np.diff(diffs)
        
        if len(second_diffs) == 0:
            return 5
        
        elbow_idx = np.argmax(second_diffs) + 2  # +2 because of double diff
        optimal_n = min(elbow_idx, max_regimes)
        
        tprint_info(f"📊 Elbow analysis: optimal_n_regimes={optimal_n}")
        return max(2, optimal_n)  # Ensure at least 2 regimes
    
    def _detect_regimes_gap_statistic(self, features_scaled: np.ndarray, max_regimes: int,
                                   min_regime_size: float, stability_threshold: float) -> int:
        """Detect optimal regimes using gap statistic."""
        # Simplified gap statistic implementation
        from sklearn.cluster import KMeans
        from sklearn.metrics import pairwise_distances
        
        gap_values = []
        
        for n_clusters in range(1, max_regimes + 1):
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
                cluster_labels = kmeans.fit_predict(features_scaled)
                
                # Within-cluster dispersion
                Wk = 0
                for k in range(n_clusters):
                    cluster_points = features_scaled[cluster_labels == k]
                    if len(cluster_points) > 1:
                        Wk += np.sum(pairwise_distances(cluster_points)**2) / (2 * len(cluster_points))
                
                # Reference dispersion (uniform random data)
                np.random.seed(42)
                reference_data = np.random.uniform(
                    features_scaled.min(), features_scaled.max(), features_scaled.shape
                )
                
                ref_kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
                ref_labels = ref_kmeans.fit_predict(reference_data)
                
                Wk_ref = 0
                for k in range(n_clusters):
                    ref_cluster_points = reference_data[ref_labels == k]
                    if len(ref_cluster_points) > 1:
                        Wk_ref += np.sum(pairwise_distances(ref_cluster_points)**2) / (2 * len(ref_cluster_points))
                
                gap = np.log(Wk_ref) - np.log(Wk)
                gap_values.append(gap)
                
            except Exception as e:
                tprint_warning(f"Gap statistic failed for n_clusters={n_clusters}: {e}")
                gap_values.append(0)
        
        if len(gap_values) < 2:
            return 5
        
        # Find optimal k using one-standard-error rule
        gap_values = np.array(gap_values)
        optimal_idx = np.argmax(gap_values)
        optimal_n = optimal_idx + 1  # +1 because range starts at 1
        
        tprint_info(f"📊 Gap statistic: optimal_n_regimes={optimal_n}, gap={gap_values[optimal_idx]:.3f}")
        return min(max(2, optimal_n), max_regimes)
    def _generate_liquidity_features(self, market_data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        df = market_data.copy()

        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required OHLCV columns for liquidity features: {missing}")

        eps = 1e-9

        # Delegate core OHLCV-derived liquidity features to the feature bank
        df = generate_liquidity_regime_features(df, config)

        # Ease of Movement (EMV)
        mid_price = (df["high"] + df["low"]) / 2.0
        mid_price_prev = mid_price.shift(1)
        df["emv"] = (mid_price - mid_price_prev) / ((df["volume"] / (df["range"] + eps)) + eps)

        # Candle geometry
        df["clv"] = (df["close"] - df["low"]) / (df["range"] + eps) - 0.5
        upper_wick = (df["high"] - df[["close", "open"]].max(axis=1)).clip(lower=0)
        lower_wick = (df[["close", "open"]].min(axis=1) - df["low"]).clip(lower=0)
        df["wick_ratio"] = np.maximum(upper_wick, lower_wick) / (df["range"] + eps)
        df["body_dominance"] = (df["close"] - df["open"]) / (df["range"] + eps)
        df["gap_factor"] = (df["open"] - df["close"].shift(1)) / (df["close"].shift(1) + eps)

        # Intraday vs closing volatility feature
        df["intraday_close_ratio"] = df["range"] / (df["abs_return_1h"].replace(0, np.nan) + eps)

        # Winsorize heavy tails for stability
        winsor_lower = float(config.get("liquidity_winsor_lower", 0.005))
        winsor_upper = float(config.get("liquidity_winsor_upper", 0.995))
        winsor_cols = [
            "rvol_24",
            "rvol_168",
            "vol_z_24",
            "normalized_range",
            "ghost_ratio",
            "absorption_ratio",
            "amihud_validity",
            "amivest_efficiency",
            "emv",
            "intraday_close_ratio",
        ]
        for col in winsor_cols:
            if col in df.columns:
                series = df[col].dropna()
                if len(series) == 0:
                    continue
                lo = series.quantile(winsor_lower)
                hi = series.quantile(winsor_upper)
                df[col] = df[col].clip(lower=lo, upper=hi)

        # ============================================================================
        # LIQUIDITY REGIME FEATURES: 60 FOCUSED FEATURES FOR REGIME DISTINCTIVENESS
        # Timeframes aligned to 30m-3h trading duration:
        # - 1h: Immediate price action (current bar context)
        # - 3h: Trade-matched window (max 3h trade duration)
        # - 6h: Structural context (2× longest trade, intermediate regime)
        # ============================================================================
        # Each feature directly maximizes between-regime variance for at least one
        # regime pair (Valid Trend, Apathy, Absorption, Ghost)
        # ============================================================================

        # CATEGORY 1: DIRECTIONAL ORDERFLOW (10 features)
        # Distinguish Trend (high conviction) vs Apathy (balanced flow)

        df["close_position_range"] = (df["close"] - df["low"]) / (df["range"] + eps)
        df["close_position_range"] = df["close_position_range"].clip(0, 1)

        df["volume_buyer_intensity"] = df["volume"] * df["close_position_range"]
        df["volume_seller_intensity"] = df["volume"] * (1.0 - df["close_position_range"])

        total_dir_volume = df["volume_buyer_intensity"] + df["volume_seller_intensity"]
        df["volume_direction_imbalance"] = (
            (df["volume_buyer_intensity"] - df["volume_seller_intensity"]) / (total_dir_volume + eps)
        )

        df["volume_direction_conviction"] = df["volume_direction_imbalance"].abs()

        df["direction_change"] = (
            (df["close"] > df["close"].shift(1)).astype(float) -
            (df["close"] < df["close"].shift(1)).astype(float)
        )
        df["volume_direction_consistency"] = (
            df["volume_direction_imbalance"] * df["direction_change"]
        )

        # Delta alignment: does orderflow direction match price direction?
        price_direction = np.sign(df["close"] - df["open"])  # bar direction
        delta_direction = np.sign(df["volume_direction_imbalance"])  # volume direction

        # Alignment score: +1 (aligned), -1 (diverged), 0 (neutral)
        df["delta_alignment"] = price_direction * delta_direction

        # Rolling alignment strength over 3h window (3 bars)
        df["delta_alignment_3h"] = df["delta_alignment"].rolling(window=12, min_periods=1).mean()

        # For Valid Trend: high conviction + high alignment
        # For Absorption: high conviction + LOW/negative alignment
        df["delta_regime_signal"] = (
            df["volume_direction_conviction"] * df["delta_alignment_3h"]
        )

        # CATEGORY 2: TREND PERSISTENCE (10 features)
        # Distinguish Trend (high persistence) vs Apathy (random) vs Absorption (reversals)

        direction_sign = np.sign(df["return_1h"])
        direction_sign = direction_sign.replace(0, np.nan)

        same_direction_3 = (direction_sign == direction_sign.shift(1)).astype(float)
        df["consecutive_direction_ratio_3h"] = same_direction_3.rolling(window=12, min_periods=1).sum() / 12.0
        df["consecutive_direction_ratio_3h_ewm3"] = (
            df["consecutive_direction_ratio_3h"].ewm(span=3, adjust=False).mean()
        )

        same_direction_6 = (direction_sign == direction_sign.shift(1)).astype(float)
        df["consecutive_direction_ratio_6h"] = same_direction_6.rolling(window=24, min_periods=1).sum() / 24.0
        df["consecutive_direction_ratio_6h_ewm3"] = (
            df["consecutive_direction_ratio_6h"].ewm(span=3, adjust=False).mean()
        )

        df["return_autocorr_lag1_3h"] = df["return_1h"].rolling(window=12, min_periods=2).apply(
            lambda x: x.iloc[-1] * x.iloc[-2] if len(x) >= 2 else 0, raw=False
        )
        df["return_autocorr_lag1_6h"] = df["return_1h"].rolling(window=24, min_periods=2).apply(
            lambda x: x.iloc[-1] * x.iloc[-2] if len(x) >= 2 else 0, raw=False
        )

        momentum_ma_3 = df["return_1h"].rolling(window=12, min_periods=2).mean()
        momentum_ma_6 = df["return_1h"].rolling(window=24, min_periods=2).mean()
        df["momentum_persistence_3h"] = (momentum_ma_3 - momentum_ma_6) / (abs(momentum_ma_6) + eps)
        df["momentum_persistence_3h_ewm6"] = (
            df["momentum_persistence_3h"].ewm(span=6, adjust=False).mean()
        )

        df["trend_confirmation_3h"] = (
            df["consecutive_direction_ratio_3h"] * df["volume_direction_conviction"]
        )
        df["trend_confirmation_6h"] = (
            df["consecutive_direction_ratio_6h"] * df["volume_direction_conviction"]
        )

        # CATEGORY 3: VOLATILITY-MOMENTUM ALIGNMENT (10 features)
        # Distinguish Trend (vol + momentum sync) vs Ghost (vol spikes without momentum)

        realized_vol_1h = df["abs_return_1h"]
        realized_vol_3h = df["return_1h"].rolling(window=12, min_periods=1).std()
        realized_vol_6h = df["return_1h"].rolling(window=24, min_periods=2).std()

        df["realized_vol_1h"] = realized_vol_1h
        df["realized_vol_3h"] = realized_vol_3h
        df["realized_vol_6h"] = realized_vol_6h

        df["vol_ratio_1h_3h"] = realized_vol_1h / (realized_vol_3h + eps)
        df["vol_ratio_3h_6h"] = realized_vol_3h / (realized_vol_6h + eps)
        df["vol_ratio_1h_6h"] = realized_vol_1h / (realized_vol_6h + eps)

        vol_ma_6 = df["abs_return_1h"].rolling(window=24, min_periods=2).mean()
        df["vol_spike_ratio"] = df["abs_return_1h"] / (vol_ma_6 + eps)

        df["vol_momentum_sync"] = (
            (df["vol_spike_ratio"] > 1.5).astype(float) * df["volume_direction_conviction"]
        )
        df["vol_momentum_sync_ewm3"] = (
            df["vol_momentum_sync"].ewm(span=3, adjust=False).mean()
        )

        df["range_momentum_divergence"] = (
            df["range"] - df["abs_return_1h"]
        ) / (df["range"] + eps)

        df["momentum_vol_alignment_3h"] = df["abs_return_1h"] / (realized_vol_3h + eps)
        df["momentum_vol_alignment_3h_ewm3"] = (
            df["momentum_vol_alignment_3h"].ewm(span=3, adjust=False).mean()
        )

        # CATEGORY 4: REVERSAL & TRAP PATTERNS (10 features)
        # Distinguish Trend (few reversals) vs Absorption (reversals with substance) vs Ghost (whipsaws)

        sign_changes = (np.sign(df["return_1h"]) != np.sign(df["return_1h"].shift(1))).astype(float)
        df["reversal_intensity"] = df["abs_return_1h"] * sign_changes
        df["reversal_intensity_ewm3"] = (
            df["reversal_intensity"].ewm(span=3, adjust=False).mean()
        )

        df["reversal_conviction"] = (
            (np.sign(df["return_1h"]) == np.sign(df["return_1h"].shift(1))).astype(float)
            .rolling(window=6, min_periods=2).sum() / 6.0
        )
        df["reversal_conviction_ewm3"] = (
            df["reversal_conviction"].ewm(span=3, adjust=False).mean()
        )

        df["whipsaw_count"] = sign_changes.rolling(window=12, min_periods=4).sum()
        df["whipsaw_count_ewm6"] = (
            df["whipsaw_count"].ewm(span=6, adjust=False).mean()
        )

        df["reversal_volume_sync"] = (
            df["reversal_intensity"] * df["volume_direction_conviction"]
        )
        df["reversal_volume_sync_ewm3"] = (
            df["reversal_volume_sync"].ewm(span=3, adjust=False).mean()
        )

        df["return_autocorr_lag6"] = df["return_1h"].rolling(window=48, min_periods=6).apply(
            lambda x: x.iloc[:24].corr(x.iloc[24:]) if len(x) == 48 else 0, raw=False
        )

        price_change_6h = (df["close"] - df["close"].shift(24)).abs()
        volatility_6h = df["abs_return_1h"].rolling(window=24, min_periods=2).sum()
        df["efficiency_ratio"] = price_change_6h / (volatility_6h + eps)

        # CATEGORY 5: ORDERBOOK PRESSURE (6 features)
        # Distinguish Absorption (stacked orders) from Apathy (scattered)

        close_pct_in_range = df["close_position_range"]
        volume_concentration_3h = (
            (close_pct_in_range * df["volume"]).rolling(window=12, min_periods=1).std() /
            (df["volume"].rolling(window=12, min_periods=1).mean() + eps)
        )
        df["volume_concentration_ratio_3h"] = volume_concentration_3h
        df["volume_concentration_ratio_3h_ewm6"] = (
            df["volume_concentration_ratio_3h"].ewm(span=6, adjust=False).mean()
        )

        high_move = (df["high"] - df["close"]).abs()
        low_move = (df["close"] - df["low"]).abs()
        df["pressure_ratio"] = (
            (high_move * df["volume"]) / ((low_move * df["volume"]) + eps)
        )
        df["pressure_ratio_ewm6"] = (
            df["pressure_ratio"].ewm(span=6, adjust=False).mean()
        )

        price_move_1pct = df["close"] * 0.01
        df["kyle_lambda_proxy"] = (
            df["volume"] / price_move_1pct
        ).rolling(window=24, min_periods=2).mean()
        df["kyle_lambda_proxy_ewm6"] = (
            df["kyle_lambda_proxy"].ewm(span=6, adjust=False).mean()
        )

        # CATEGORY 6: MULTI-TIMEFRAME VOLATILITY ALIGNMENT (8 features)
        # Context for regime identification via vol profiles

        df["intra_bar_vol_estimate"] = (df["high"] - df["low"]) / (df["close"] + eps)

        upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
        lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
        df["wick_vol_contribution"] = (upper_wick + lower_wick) / (df["range"] + eps)
        df["wick_vol_contribution_ewm6"] = (
            df["wick_vol_contribution"].ewm(span=6, adjust=False).mean()
        )

        df["session_vol_percentile"] = (
            df["abs_return_1h"].rolling(window=96, min_periods=4).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1]
            )
        )
        df["session_vol_percentile_ewm6"] = (
            df["session_vol_percentile"].ewm(span=6, adjust=False).mean()
        )

        vol_above_ma = (df["abs_return_1h"] > df["abs_return_1h"].rolling(window=24).mean()).astype(float)
        df["vol_clustering"] = vol_above_ma.rolling(window=24, min_periods=2).sum() / 24.0
        df["vol_clustering_ewm6"] = (
            df["vol_clustering"].ewm(span=6, adjust=False).mean()
        )

        df["vol_regime_change"] = (
            (realized_vol_3h - realized_vol_6h) / (realized_vol_6h + eps)
        )

        # CATEGORY 7: INFORMATION EFFICIENCY (8 features)
        # Market quality and discovery efficiency

        df["efficiency_ratio_ewm6"] = (
            df["efficiency_ratio"].ewm(span=6, adjust=False).mean()
        )

        price_trend_6h = (df["close"].diff(24) > 0).astype(float)
        volume_trend_6h = (df["volume"] > df["volume"].rolling(window=24).mean()).astype(float)
        df["volume_price_trend_sync"] = (
            price_trend_6h.rolling(window=24, min_periods=2).mean() -
            volume_trend_6h.rolling(window=24, min_periods=2).mean()
        )
        df["volume_price_trend_sync_ewm6"] = (
            df["volume_price_trend_sync"].ewm(span=6, adjust=False).mean()
        )

        df["price_impact_ratio"] = (
            df["range"] / (df["volume"] + eps)
        )
        df["price_impact_ratio_ewm6"] = (
            df["price_impact_ratio"].ewm(span=6, adjust=False).mean()
        )

        df["momentum_volume_alignment"] = (
            np.sign(df["return_1h"]) * df["volume_direction_conviction"]
        )

        # ============================================================================
        # INTERACTION FEATURES: Composite signals for regime distinction
        # ============================================================================

        # Effort × Result interaction (captures regime essence)
        df["volume_range_interaction"] = df["rvol_24"] * df["normalized_range"]

        # Trend strength composite (Valid Trend signal)
        df["trend_strength"] = (
            df["volume_direction_conviction"] * df["consecutive_direction_ratio_6h"]
        )

        # Trap indicator (Ghost regime signal)
        df["trap_indicator"] = df["ghost_ratio"] * df["whipsaw_count"]

        # Absorption signal (high volume + diverged delta)
        df["absorption_signal"] = (
            df["absorption_ratio"] * (1.0 - df["delta_alignment_3h"].abs())
        )

        # ============================================================================
        # TIER 1 FEATURES: High-impact regime discriminators
        # ============================================================================

        # 1. Roll's Spread Estimator - Measures bid-ask spread from return covariance
        # High spread = Poor liquidity (Ghost), Low spread = Good liquidity (Valid Trend)
        return_autocov = -df["return_1h"].rolling(2).apply(
            lambda x: x.iloc[0] * x.iloc[1] if len(x) == 2 else 0, raw=False
        )
        df["rolls_spread"] = 2 * np.sqrt(return_autocov.clip(lower=0))

        # 2. Breakout Failure Rate - Detects Ghost regime (large move followed by reversal)
        high_range_mask = df["normalized_range"] > df["normalized_range"].rolling(20).quantile(0.8)
        reversal_next_bar = (np.sign(df["return_1h"]) != np.sign(df["return_1h"].shift(1)))
        df["breakout_failure_rate"] = (high_range_mask & reversal_next_bar).rolling(12, min_periods=4).mean()

        # 3. Cumulative Delta Divergence - Absorption identifier
        # High CDD = sustained volume/price divergence = Absorption
        cumulative_volume_imbalance = df["volume_direction_imbalance"].rolling(6).sum()
        cumulative_price_change = (df["close"].diff(6) / df["close"].shift(6)).abs()
        df["cumulative_delta_divergence"] = (
            np.abs(cumulative_volume_imbalance) - cumulative_price_change
        )

        # 4. Order Flow Persistence - Trend reliability
        # High persistence = Valid Trend, Low persistence (flips) = Ghost/Apathy
        df["order_flow_persistence"] = df["volume_direction_imbalance"].rolling(3).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
        )

        # 5. Volume Depth Ratio - Liquidity depth proxy
        # Low VDR = Shallow liquidity (Ghost/Steamroller), High VDR = Deep liquidity (Absorption/Valid Trend)
        price_move_1pct = df["close"] * 0.01
        df["volume_depth_ratio"] = df["volume"] / (price_move_1pct + eps)

        # ============================================================================
        # TIER 2 FEATURES: High value-add discriminators
        # ============================================================================

        # 6. Parkinson's Range-Based Volatility - More efficient estimator
        df["parkinsons_volatility"] = np.sqrt(
            (1 / (4 * np.log(2))) * (np.log(df["high"] / df["low"]) ** 2)
        ).rolling(6, min_periods=2).mean()

        # 7. VWAP Distance - Value area context
        # Near VWAP = fair value (Apathy/Absorption), Far = trending (Valid Trend/Ghost)
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        vwap = (typical_price * df["volume"]).rolling(20, min_periods=5).sum() / (
            df["volume"].rolling(20, min_periods=5).sum() + eps
        )
        df["vwap_distance"] = (df["close"] - vwap) / (vwap + eps)

        # 8. Kyle's Lambda (enhanced) - Measures permanent price impact
        signed_volume = df["volume"] * np.sign(df["return_1h"])
        cumulative_signed_volume = signed_volume.rolling(6, min_periods=2).sum()
        price_change_6h = df["close"] - df["close"].shift(6)
        df["kyles_lambda_enhanced"] = price_change_6h / (cumulative_signed_volume + eps)

        # 9. Trap Score - Ghost composite signal
        # Large range + low volume + subsequent reversal = Ghost
        df["trap_score"] = (
            df["normalized_range"] *
            (1 / (df["rvol_20"] + eps)) *
            df["reversal_intensity"]
        )

        # 10. Vol-of-Vol - Regime stability
        # Stable vol = Valid Trend or Apathy, Unstable vol = Ghost or Absorption
        realized_vol_6h_rolling = df["return_1h"].rolling(24, min_periods=2).std()
        df["vol_of_vol"] = (
            realized_vol_6h_rolling.rolling(48, min_periods=4).std() /
            (realized_vol_6h_rolling.rolling(48, min_periods=4).mean() + eps)
        )

        # ============================================================================
        # ROLLING NORMALIZATION: Core dimensions for regime assignment
        # Use rolling window to prevent look-ahead bias with canonical window size.
        # ============================================================================
        window_size = 500  # ~500 bars for 15m data (canonical normalization window)
        min_periods = window_size // 2

        # Volume dimension: route via adaptive normalizer with explicit volume
        # column hints so rvol_* axes receive log1p+zscore treatment.
        volume_axes = [col for col in ["rvol_24", "rvol_168"] if col in df.columns]
        if volume_axes:
            try:
                volume_scaled = rolling_adaptive_normalize(
                    df[volume_axes],
                    window=window_size,
                    min_periods=min_periods,
                    volume_columns=volume_axes,
                )
                for col in volume_axes:
                    df[f"{col}_scaled"] = volume_scaled[col]
            except Exception as e:
                tprint_warning(
                    f"Failed to apply adaptive normalization for liquidity rvol axes: {e}"
                )
                for col in volume_axes:
                    try:
                        df[f"{col}_scaled"] = log1p_zscore_normalize(
                            df[col].clip(lower=0.0),
                            window=window_size,
                            min_periods=min_periods,
                        )
                    except Exception as inner_e:
                        tprint_warning(f"Failed to log1p+zscore normalize {col}: {inner_e}")

        # Use generic rolling winsorized z-score normalization for other
        # continuous, non-pure-volume dimensions.
        df["vol_z_24_scaled"] = winsorized_zscore_normalize(
            df["vol_z_24"],
            window=window_size,
        )

        # Delta dimension (order flow alignment)
        df["delta_regime_signal_scaled"] = winsorized_zscore_normalize(
            df["delta_regime_signal"],
            window=window_size,
        )

        # Amihud dimension (illiquidity/price impact)
        df["amihud_spike_ratio_scaled"] = winsorized_zscore_normalize(
            df["amihud_spike_ratio"],
            window=window_size,
        )
        try:
            rvol24_scaled = df.get("rvol_24_scaled")
            rvol168_scaled = df.get("rvol_168_scaled")
            norm_range = df.get("normalized_range")
            vol_of_vol = df.get("vol_of_vol")
            if rvol168_scaled is not None and norm_range is not None and vol_of_vol is not None:
                long_rvol_thr = float(config.get("liquidity_structural_rvol_168_thresh", -0.5))
                long_low = rvol168_scaled < long_rvol_thr
                range_low_band = norm_range.rolling(96, min_periods=10).quantile(0.3)
                flat_range = norm_range < range_low_band
                vol_stab_band = vol_of_vol.rolling(96, min_periods=10).quantile(0.3)
                stable_vol = vol_of_vol < vol_stab_band
                df["liquidity_structural_low"] = (long_low & flat_range & stable_vol).astype(int)
            else:
                df["liquidity_structural_low"] = 0
        except Exception:
            df["liquidity_structural_low"] = 0

        try:
            rvol24_scaled = df.get("rvol_24_scaled")
            norm_range = df.get("normalized_range")
            gap_factor = df.get("gap_factor")
            if rvol24_scaled is not None and norm_range is not None and gap_factor is not None:
                range_high_band = norm_range.rolling(96, min_periods=10).quantile(0.8)
                high_range = norm_range > range_high_band
                short_rvol_thr = float(config.get("liquidity_transient_rvol_24_thresh", 0.0))
                low_rvol_short = rvol24_scaled < short_rvol_thr
                gap_thr = float(config.get("liquidity_gap_factor_thresh", 0.005))
                big_gap = gap_factor.abs() > gap_thr
                df["liquidity_transient_gap"] = (high_range & low_rvol_short & big_gap).astype(int)
            else:
                df["liquidity_transient_gap"] = 0
        except Exception:
            df["liquidity_transient_gap"] = 0

        return df

    def _compute_winsorized_cov_ratio(
        self,
        feature_df: pd.DataFrame,
        regime_labels: pd.Series,
        feature_cols: Optional[List[str]] = None,
    ) -> float:
        """
        Compute winsorized coefficient of variation ratio (between/within regimes).

        Higher ratio = better regime separation.

        Args:
            feature_df: DataFrame with features
            regime_labels: Series with regime assignments
            feature_cols: List of feature columns to use (if None, use all numeric)

        Returns:
            WCV ratio score (between-regime CoV / within-regime CoV)
        """
        if feature_cols is None:
            feature_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()

        feature_data = feature_df[feature_cols].copy()
        eps = 1e-9

        # Winsorize features (1st-99th percentile)
        for col in feature_cols:
            if col in feature_data.columns:
                series = feature_data[col].dropna()
                if len(series) > 0:
                    q01 = series.quantile(0.01)
                    q99 = series.quantile(0.99)
                    feature_data[col] = feature_data[col].clip(lower=q01, upper=q99)

        regime_ids = sorted(regime_labels.unique())
        if len(regime_ids) < 2:
            return 0.0

        # Compute between-regime CoV
        regime_means = []
        for regime_id in regime_ids:
            mask = regime_labels == regime_id
            if mask.sum() < 3:
                continue
            regime_mean = feature_data[mask].mean().mean()  # mean across features, then across samples
            regime_means.append(regime_mean)

        if len(regime_means) < 2:
            return 0.0

        between_mean = float(np.mean(regime_means))
        between_std = float(np.std(regime_means))
        between_cov = between_std / (abs(between_mean) + eps)

        # Compute within-regime CoV (average across regimes)
        within_covs = []
        for regime_id in regime_ids:
            mask = regime_labels == regime_id
            if mask.sum() < 3:
                continue
            regime_data = feature_data[mask]
            regime_mean = regime_data.mean().mean()
            regime_std = regime_data.std().mean()
            regime_cov = regime_std / (abs(regime_mean) + eps)
            within_covs.append(regime_cov)

        if len(within_covs) == 0:
            return 0.0

        within_cov = float(np.mean(within_covs))

        # WCV ratio: between / within (higher is better)
        wcv_ratio = between_cov / (within_cov + eps)

        return float(wcv_ratio)

    def _find_optimal_split(
        self,
        feature_series: pd.Series,
        other_features_df: pd.DataFrame,
        candidate_percentiles: Optional[np.ndarray] = None,
        min_regime_fraction: float = 0.03,
    ) -> Tuple[float, float]:
        """
        Find optimal threshold for a feature split that maximizes WCV ratio.

        Args:
            feature_series: Feature to split on
            other_features_df: Features to compute WCV on (Volume, Delta, Amihud)
            candidate_percentiles: Percentiles to test (default: 20th to 80th by 5)
            min_regime_fraction: Minimum fraction of samples per regime

        Returns:
            (optimal_threshold, best_wcv_score)
        """
        if candidate_percentiles is None:
            candidate_percentiles = np.arange(0.20, 0.85, 0.05)

        feature_vals = feature_series.dropna()
        if len(feature_vals) < 100:
            # Not enough samples, return median
            return float(feature_vals.median()), 0.0

        min_samples = int(len(feature_vals) * min_regime_fraction)

        best_threshold = float(feature_vals.median())
        best_score = 0.0

        for pct in candidate_percentiles:
            threshold = float(feature_vals.quantile(pct))

            # Create binary split
            high_mask = feature_series >= threshold
            low_mask = feature_series < threshold

            # Check minimum size constraint
            if high_mask.sum() < min_samples or low_mask.sum() < min_samples:
                continue

            # Create temporary regime labels (0=low, 1=high)
            temp_labels = pd.Series(0, index=feature_series.index)
            temp_labels[high_mask] = 1

            # Compute WCV ratio
            wcv_score = self._compute_winsorized_cov_ratio(
                other_features_df,
                temp_labels,
                feature_cols=other_features_df.columns.tolist(),
            )

            # Penalize very imbalanced splits
            balance = min(high_mask.sum(), low_mask.sum()) / max(high_mask.sum(), low_mask.sum())
            balance_penalty = 0.5 + 0.5 * balance  # ranges from 0.5 to 1.0

            adjusted_score = wcv_score * balance_penalty

            if adjusted_score > best_score:
                best_score = adjusted_score
                best_threshold = threshold

        tprint_info(
            f"Optimal split for {feature_series.name}: threshold={best_threshold:.4f}, "
            f"WCV score={best_score:.4f}"
        )

        return best_threshold, best_score

    def _hierarchical_regime_assignment(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Assign regimes using hierarchical decision tree optimized for WCV.

        Decision tree structure:
        Level 1: Volume Split (High vs Low)
        ├─ Low Volume Branch
        │  ├─ Level 2: Range Split
        │  │  ├─ Flat Range → Apathy (0)
        │  │  └─ Large Range → Level 3: Amihud Split
        │  │     ├─ High Amihud Spike → Ghost (3) [trap/low liquidity]
        │  │     └─ Low Amihud → Steamroller (4) [initiative momentum]
        │  │
        └─ High Volume Branch
           └─ Level 2: Delta Alignment
              ├─ Delta Aligned with Price → Valid Trend (1)
              └─ Delta Diverged from Price → Absorption (2)

        Args:
            df: DataFrame with all liquidity features
            config: Configuration dict

        Returns:
            DataFrame with 'liquidity_regime' column added
        """
        work_df = df.copy()

        config["liquidity_tree_volume_threshold"] = None
        config["liquidity_tree_delta_threshold"] = None
        config["liquidity_tree_range_threshold"] = None
        config["liquidity_tree_amihud_threshold"] = None

        # Core features for WCV computation
        core_feature_cols = ["rvol_24_scaled", "delta_regime_signal_scaled", "amihud_spike_ratio_scaled"]

        # Ensure all required features exist
        required_features = ["rvol_24_scaled", "normalized_range", "amihud_spike_ratio_scaled", "delta_regime_signal_scaled"]
        missing = [f for f in required_features if f not in work_df.columns]
        if missing:
            raise ValueError(f"Missing required features for hierarchical regime assignment: {missing}")

        core_features_df = work_df[core_feature_cols].copy()

        # Initialize regime labels (will be overwritten)
        regimes = np.full(len(work_df), -1, dtype=int)

        # ========================================================================
        # LEVEL 1: VOLUME SPLIT (High vs Low Volume)
        # ========================================================================
        volume_override = config.get("liquidity_tree_volume_threshold_override")
        if volume_override is not None:
            volume_threshold = float(volume_override)
            tprint_info(
                f"Using override volume threshold for liquidity tree: {volume_threshold:.4f}"
            )
        else:
            vol_pct_low = float(config.get("liquidity_tree_volume_pct_low", 0.30))
            vol_pct_high = float(config.get("liquidity_tree_volume_pct_high", 0.70))
            vol_pct_step = float(config.get("liquidity_tree_volume_pct_step", 0.05))

            if vol_pct_step <= 0.0:
                vol_pct_step = 0.05
            if vol_pct_high <= vol_pct_low:
                vol_pct_low, vol_pct_high = 0.30, 0.70

            volume_percentiles = np.arange(vol_pct_low, vol_pct_high, vol_pct_step)

            volume_threshold, _ = self._find_optimal_split(
                work_df["rvol_24_scaled"],
                core_features_df,
                candidate_percentiles=volume_percentiles,
            )

        config["liquidity_tree_volume_threshold"] = float(volume_threshold)

        high_volume_mask = work_df["rvol_24_scaled"] >= volume_threshold
        low_volume_mask = ~high_volume_mask

        tprint_info(
            f"Level 1 Volume Split: {high_volume_mask.sum()} high-vol samples, "
            f"{low_volume_mask.sum()} low-vol samples"
        )

        # ========================================================================
        # HIGH VOLUME BRANCH: LEVEL 2 - Delta Alignment Split
        # ========================================================================
        high_vol_indices = work_df.index[high_volume_mask]
        if len(high_vol_indices) > 0:
            # Find optimal delta alignment threshold (or use override)
            delta_override = config.get("liquidity_tree_delta_threshold_override")
            if delta_override is not None:
                delta_threshold = float(delta_override)
                tprint_info(
                    f"Using override delta threshold for liquidity tree: {delta_threshold:.4f}"
                )
            else:
                delta_pct_low = float(config.get("liquidity_tree_delta_pct_low", 0.35))
                delta_pct_high = float(config.get("liquidity_tree_delta_pct_high", 0.65))
                delta_pct_step = float(config.get("liquidity_tree_delta_pct_step", 0.05))

                if delta_pct_step <= 0.0:
                    delta_pct_step = 0.05
                if delta_pct_high <= delta_pct_low:
                    delta_pct_low, delta_pct_high = 0.35, 0.65

                delta_percentiles = np.arange(delta_pct_low, delta_pct_high, delta_pct_step)

                delta_threshold, _ = self._find_optimal_split(
                    work_df.loc[high_vol_indices, "delta_regime_signal_scaled"],
                    core_features_df.loc[high_vol_indices],
                    candidate_percentiles=delta_percentiles,
                )

            config["liquidity_tree_delta_threshold"] = float(delta_threshold)

            # Delta aligned (positive) → Valid Trend (1)
            # Delta diverged (negative) → Absorption (2)
            delta_aligned_mask = work_df["delta_regime_signal_scaled"] >= delta_threshold

            valid_trend_mask = high_volume_mask & delta_aligned_mask
            absorption_mask = high_volume_mask & ~delta_aligned_mask

            regimes[valid_trend_mask.values] = 1
            regimes[absorption_mask.values] = 2

            tprint_info(
                f"  High-vol branch: {valid_trend_mask.sum()} Valid Trend, "
                f"{absorption_mask.sum()} Absorption"
            )

        # ========================================================================
        # LOW VOLUME BRANCH: LEVEL 2 - Range Split
        # ========================================================================
        low_vol_indices = work_df.index[low_volume_mask]
        if len(low_vol_indices) > 0:
            # Find optimal range threshold (or use override)
            range_override = config.get("liquidity_tree_range_threshold_override")
            if range_override is not None:
                range_threshold = float(range_override)
                tprint_info(
                    f"Using override range threshold for liquidity tree: {range_threshold:.4f}"
                )
            else:
                range_pct_low = float(config.get("liquidity_tree_range_pct_low", 0.30))
                range_pct_high = float(config.get("liquidity_tree_range_pct_high", 0.70))
                range_pct_step = float(config.get("liquidity_tree_range_pct_step", 0.05))

                if range_pct_step <= 0.0:
                    range_pct_step = 0.05
                if range_pct_high <= range_pct_low:
                    range_pct_low, range_pct_high = 0.30, 0.70

                range_percentiles = np.arange(range_pct_low, range_pct_high, range_pct_step)

                range_threshold, _ = self._find_optimal_split(
                    work_df.loc[low_vol_indices, "normalized_range"],
                    core_features_df.loc[low_vol_indices],
                    candidate_percentiles=range_percentiles,
                )

            config["liquidity_tree_range_threshold"] = float(range_threshold)

            flat_range_mask = work_df["normalized_range"] < range_threshold
            large_range_mask = work_df["normalized_range"] >= range_threshold

            # Flat range → Apathy (0)
            apathy_mask = low_volume_mask & flat_range_mask
            regimes[apathy_mask.values] = 0

            tprint_info(
                f"  Low-vol flat-range: {apathy_mask.sum()} Apathy samples"
            )

            # ====================================================================
            # LOW VOLUME + LARGE RANGE: LEVEL 3 - Amihud Split
            # ====================================================================
            large_range_indices = work_df.index[low_volume_mask & large_range_mask]
            if len(large_range_indices) > 20:
                # Find optimal Amihud threshold (or use override)
                amihud_override = config.get("liquidity_tree_amihud_threshold_override")
                if amihud_override is not None:
                    amihud_threshold = float(amihud_override)
                    tprint_info(
                        f"Using override Amihud threshold for liquidity tree: {amihud_threshold:.4f}"
                    )
                else:
                    amihud_pct_low = float(config.get("liquidity_tree_amihud_pct_low", 0.40))
                    amihud_pct_high = float(config.get("liquidity_tree_amihud_pct_high", 0.75))
                    amihud_pct_step = float(config.get("liquidity_tree_amihud_pct_step", 0.05))

                    if amihud_pct_step <= 0.0:
                        amihud_pct_step = 0.05
                    if amihud_pct_high <= amihud_pct_low:
                        amihud_pct_low, amihud_pct_high = 0.40, 0.75

                    amihud_percentiles = np.arange(
                        amihud_pct_low,
                        amihud_pct_high,
                        amihud_pct_step,
                    )

                    amihud_threshold, _ = self._find_optimal_split(
                        work_df.loc[large_range_indices, "amihud_spike_ratio_scaled"],
                        core_features_df.loc[large_range_indices],
                        candidate_percentiles=amihud_percentiles,
                    )

                config["liquidity_tree_amihud_threshold"] = float(amihud_threshold)

                high_amihud_mask = work_df["amihud_spike_ratio_scaled"] >= amihud_threshold

                # High Amihud spike → Ghost (3) [trap/illiquidity]
                # Low Amihud → Steamroller (4) [initiative momentum]
                ghost_mask = low_volume_mask & large_range_mask & high_amihud_mask
                steamroller_mask = low_volume_mask & large_range_mask & ~high_amihud_mask

                regimes[ghost_mask.values] = 3
                regimes[steamroller_mask.values] = 4

                tprint_info(
                    f"  Low-vol large-range: {ghost_mask.sum()} Ghost, "
                    f"{steamroller_mask.sum()} Steamroller"
                )
            else:
                # Not enough samples for Amihud split, assign all to Ghost (conservative)
                ghost_mask = low_volume_mask & large_range_mask
                regimes[ghost_mask.values] = 3
                tprint_info(
                    f"  Low-vol large-range: {ghost_mask.sum()} Ghost (insufficient samples for Amihud split)"
                )

        # ========================================================================
        # FINALIZE
        # ========================================================================
        work_df["liquidity_regime"] = regimes

        # Sanity check: ensure all samples are assigned
        unassigned = (regimes == -1).sum()
        if unassigned > 0:
            tprint_warning(
                f"Warning: {unassigned} samples unassigned in hierarchical regime assignment, "
                f"assigning to Apathy (0)"
            )
            work_df.loc[work_df["liquidity_regime"] == -1, "liquidity_regime"] = 0

        # Print regime distribution
        regime_counts = work_df["liquidity_regime"].value_counts().sort_index()
        tprint_info(f"Hierarchical regime distribution:\n{regime_counts}")

        return work_df

    def _refine_regimes_with_centroids(
        self,
        df: pd.DataFrame,
        regime_labels: pd.Series,
        feature_cols: List[str],
        n_iterations: int = 3,
        min_distance_improvement: float = 0.05,
    ) -> pd.Series:
        """
        Iteratively refine regime assignments using K-means-style centroid updates.

        Args:
            df: Feature dataframe
            regime_labels: Initial regime assignments
            feature_cols: Features to use for distance calculation (top discriminative features)
            n_iterations: Max iterations
            min_distance_improvement: Stop if improvement < this threshold

        Returns:
            Refined regime labels
        """
        refined_labels = regime_labels.copy()

        # Select only available features
        available_features = [f for f in feature_cols if f in df.columns]
        if not available_features:
            tprint_warning("No features available for centroid refinement, skipping")
            return refined_labels

        feature_data = df[available_features].fillna(0).values

        for iteration in range(n_iterations):
            # 1. Compute regime centroids
            regime_ids = np.unique(refined_labels)
            centroids = {}
            for regime_id in regime_ids:
                mask = refined_labels == regime_id
                if mask.sum() < 3:
                    # Skip regimes with too few samples
                    continue
                centroids[regime_id] = feature_data[mask].mean(axis=0)

            if len(centroids) < 2:
                tprint_warning("Not enough regimes for centroid refinement")
                break

            # 2. Reassign each sample to nearest centroid
            distances = np.zeros((len(feature_data), len(centroids)))
            centroid_ids = list(centroids.keys())

            for i, regime_id in enumerate(centroid_ids):
                distances[:, i] = np.linalg.norm(
                    feature_data - centroids[regime_id], axis=1
                )

            new_labels = np.array([centroid_ids[i] for i in np.argmin(distances, axis=1)])

            # 3. Check convergence
            n_changed = (new_labels != refined_labels.values).sum()
            pct_changed = n_changed / len(refined_labels)

            tprint_info(
                f"Centroid refinement iteration {iteration+1}: {pct_changed:.2%} samples reassigned"
            )

            if pct_changed < min_distance_improvement:
                tprint_info(f"Converged after {iteration+1} iterations")
                break

            refined_labels = pd.Series(new_labels, index=refined_labels.index)

        return refined_labels

    def _compute_regime_confidence(
        self,
        df: pd.DataFrame,
        regime_labels: pd.Series,
        feature_cols: List[str],
    ) -> pd.Series:
        """
        Compute confidence score for each regime assignment.

        Confidence is higher when a point is clearly closer to one
        centroid than to any other (far from frontiers), and lower
        when it lies near decision boundaries between regimes.

        Args:
            df: Feature dataframe
            regime_labels: Regime assignments
            feature_cols: Features to use for distance calculation

        Returns:
            Series with confidence scores [0, 1]
        """
        confidence = pd.Series(0.0, index=df.index, dtype=float)

        # Select only available features
        available_features = [f for f in feature_cols if f in df.columns]
        if not available_features:
            tprint_warning("No features available for confidence computation, returning zeros")
            return confidence

        feature_data = df[available_features].fillna(0).values

        # Compute centroids
        regime_ids = np.unique(regime_labels)
        centroid_vectors: List[np.ndarray] = []
        centroid_ids: List[int] = []
        for regime_id in regime_ids:
            mask = regime_labels == regime_id
            if mask.sum() < 3:
                continue
            centroid_vectors.append(feature_data[mask].mean(axis=0))
            centroid_ids.append(int(regime_id))

        if not centroid_vectors:
            tprint_warning("No valid centroids for confidence computation, returning zeros")
            return confidence

        centroids_arr = np.vstack(centroid_vectors)  # shape: (K, D)

        # Distances from each sample to each centroid: shape (N, K)
        diffs = feature_data[:, None, :] - centroids_arr[None, :, :]
        distances = np.linalg.norm(diffs, axis=2)

        # For each sample, confidence is based on how much closer it is to the
        # nearest centroid than to the second-nearest (frontier-based margin).
        if distances.shape[1] >= 2:
            # Sort per-row distances to get nearest and second-nearest
            sorted_d = np.sort(distances, axis=1)
            d_min = sorted_d[:, 0]
            d_second = sorted_d[:, 1]

            eps = 1e-9
            # Ratio of closeness to frontier; small when clearly closer to one
            # centroid, ~1 when equidistant between two.
            ratio = d_min / (d_second + eps)
            conf_vals = 1.0 - ratio
            conf_vals = np.clip(conf_vals, 0.0, 1.0)
        else:
            # Only one centroid available; fall back to distance-based confidence
            d_min = distances[:, 0]
            conf_vals = 1.0 / (1.0 + d_min)

        confidence.iloc[:] = conf_vals
        return confidence

    def _compute_regime_probabilities(
        self,
        df: pd.DataFrame,
        regime_labels: pd.Series,
        feature_cols: List[str],
        n_regimes: int,
    ) -> pd.DataFrame:
        available_features = [f for f in feature_cols if f in df.columns]
        if not available_features:
            return pd.DataFrame(
                0.0,
                index=df.index,
                columns=[f"p_regime_{k}" for k in range(n_regimes)],
            )

        feature_data = df[available_features].fillna(0).values
        labels = regime_labels.reindex(df.index)
        unique_labels = sorted(int(x) for x in labels.dropna().unique())
        if not unique_labels:
            return pd.DataFrame(
                0.0,
                index=df.index,
                columns=[f"p_regime_{k}" for k in range(n_regimes)],
            )

        centroid_vectors: List[np.ndarray] = []
        centroid_ids: List[int] = []
        for regime_id in unique_labels:
            mask = labels == regime_id
            if mask.sum() < 3:
                continue
            centroid_vectors.append(feature_data[mask.values].mean(axis=0))
            centroid_ids.append(int(regime_id))

        if not centroid_vectors:
            return pd.DataFrame(
                0.0,
                index=df.index,
                columns=[f"p_regime_{k}" for k in range(n_regimes)],
            )

        centroids_arr = np.vstack(centroid_vectors)
        diffs = feature_data[:, None, :] - centroids_arr[None, :, :]
        distances = np.linalg.norm(diffs, axis=2)
        eps = 1e-9
        sims = 1.0 / (distances + eps)
        row_sums = sims.sum(axis=1)
        row_sums[row_sums == 0.0] = 1.0
        probs = sims / row_sums[:, None]

        proba_df = pd.DataFrame(
            0.0,
            index=df.index,
            columns=[f"p_regime_{k}" for k in range(n_regimes)],
        )

        for j, regime_id in enumerate(centroid_ids):
            if 0 <= regime_id < n_regimes:
                col = f"p_regime_{regime_id}"
                proba_df[col] = probs[:, j]

        return proba_df

    def _compute_kde_threshold(self, series: pd.Series, config: Dict[str, Any], prefix: str) -> float:
        vals = series.dropna().astype(float)
        if len(vals) < 50:
            return float(vals.median()) if len(vals) > 0 else 0.0

        q33 = vals.quantile(0.33)
        q66 = vals.quantile(0.66)
        band_vals = vals[(vals >= q33) & (vals <= q66)]
        if len(band_vals) < 20:
            return float(vals.median())

        try:
            kde = gaussian_kde(band_vals.values.astype(float))
            grid = np.linspace(q33, q66, 256)
            densities = kde(grid)
            idx_min = int(np.argmin(densities))
            thresh = float(grid[idx_min])
            tprint_info(
                f"KDE threshold for {prefix}: q33={q33:.4f}, q66={q66:.4f}, thresh={thresh:.4f}"
            )
            return thresh
        except Exception as exc:
            tprint_warning(f"KDE threshold estimation failed for {prefix}: {exc}; using median")
            return float(vals.median())

    def _assign_liquidity_regimes(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        work_df = df.copy()

        if "rvol_24" not in work_df.columns or "normalized_range" not in work_df.columns:
            raise ValueError("Missing rvol_24 or normalized_range for liquidity regime assignment")

        vol_thresh = self._compute_kde_threshold(work_df["rvol_24"], config, prefix="volume_rvol_24")
        range_thresh = self._compute_kde_threshold(work_df["normalized_range"], config, prefix="normalized_range")

        work_df["volume_state"] = np.where(work_df["rvol_24"] >= vol_thresh, 1, 0)
        work_df["move_state"] = np.where(work_df["normalized_range"] >= range_thresh, 1, 0)

        regimes = np.full(len(work_df), np.nan, dtype=float)

        high_vol = work_df["volume_state"] == 1
        high_move = work_df["move_state"] == 1
        low_move = work_df["move_state"] == 0

        # 1 = Valid Trend (high vol, high move)
        mask_valid = high_vol & high_move

        # 2 = Absorption (high vol, low move or absorption_ratio > 1)
        mask_absorption = high_vol & low_move
        if "absorption_ratio" in work_df.columns:
            absorption_ratio_thresh = float(config.get("liquidity_absorption_ratio_thresh", 1.5))
            mask_absorption = mask_absorption & (work_df["absorption_ratio"] > absorption_ratio_thresh)

        # 3 = Ghost / Drift (low vol, high move or ghost_ratio > 1)
        mask_ghost = (work_df["volume_state"] == 0) & (work_df["move_state"] == 1)
        if "ghost_ratio" in work_df.columns:
            ghost_ratio_thresh = float(config.get("liquidity_ghost_ratio_thresh", 1.5))
            mask_ghost |= work_df["ghost_ratio"] > ghost_ratio_thresh

        # 0 = Apathy (everything else)
        mask_apathy = ~(mask_valid | mask_absorption | mask_ghost)

        regimes[mask_apathy.values] = 0
        regimes[mask_absorption.values] = 2
        regimes[mask_ghost.values] = 3
        regimes[mask_valid.values] = 1

        work_df["liquidity_regime"] = regimes

        # Ambiguity: enforce nearest regime but encode low confidence via weights
        d_vol = (work_df["rvol_24"] - vol_thresh).abs()
        d_range = (work_df["normalized_range"] - range_thresh).abs()

        # Use local scale from central band for normalization
        vol_vals = work_df["rvol_24"].dropna()
        range_vals = work_df["normalized_range"].dropna()
        vol_band_width = max(vol_vals.quantile(0.66) - vol_vals.quantile(0.33), 1e-6) if len(vol_vals) > 20 else 1.0
        range_band_width = max(range_vals.quantile(0.66) - range_vals.quantile(0.33), 1e-6) if len(range_vals) > 20 else 1.0

        d_vol_norm = (d_vol / vol_band_width).clip(0.0, 1.0)
        d_range_norm = (d_range / range_band_width).clip(0.0, 1.0)
        ambiguity = 1.0 - np.minimum(d_vol_norm, d_range_norm)  # 1.0 near threshold, 0 far away

        w_min = float(config.get("liquidity_min_sample_weight", 0.3))
        sample_weight = w_min + (1.0 - w_min) * (1.0 - ambiguity)

        work_df["liquidity_sample_weight"] = sample_weight.astype(float)

        return work_df

    def _assess_liquidity_regime_quality(
        self,
        *,
        liquidity_df: pd.DataFrame,
        regime_col: Optional[str],
        config: Dict[str, Any],
    ) -> Tuple[Optional[ClusterQualityMetrics], Optional[str]]:
        # Allow callers (notably HPO workflows) to bypass the generic
        # ClusterQualityAssessor, which can be expensive on large blank-mode
        # datasets. Liquidity-specific quality assessment remains active.
        if bool(config.get("liquidity_quality_skip_generic_cluster_assessor", False)):
            tprint_info("Skipping generic ClusterQualityAssessor for liquidity regimes (HPO mode)")
            return None, None

        if regime_col is None or regime_col not in liquidity_df.columns:
            tprint_warning("No liquidity regime column provided; skipping regime quality assessment")
            return None, None

        regime_series = liquidity_df[regime_col]
        valid_mask = regime_series.notna()
        if valid_mask.sum() == 0:
            tprint_warning("No valid liquidity regime labels for quality assessment")
            return None, None

        regime_labels = np.asarray(regime_series[valid_mask].astype(int), dtype=int)

        numeric_df = liquidity_df.select_dtypes(include=[np.number])
        feature_cols = [c for c in numeric_df.columns if c != regime_col]
        if not feature_cols:
            tprint_warning("No numeric features available for liquidity regime quality assessment")
            return None, None

        feature_data = numeric_df[feature_cols].loc[valid_mask]
        timestamps = liquidity_df.index[valid_mask]

        min_regime_size = int(config.get("liquidity_min_regime_size", 3))
        temporal_mode = str(config.get("liquidity_temporal_sensitivity_mode", "regime_persistence_focused"))
        fast_mode = bool(config.get("liquidity_quality_fast_mode", False))

        try:
            metrics = self.quality_assessor.assess_quality(
                regime_labels=regime_labels,
                feature_data=feature_data,
                forward_returns=None,
                timestamps=timestamps,
                min_regime_size=min_regime_size,
                temporal_sensitivity_mode=temporal_mode,
                fast_mode=fast_mode,
                standardize_for_metrics=True,
            )
        except Exception as exc:
            tprint_warning(f"Liquidity regime quality assessment failed: {exc}")
            return None, None

        metrics_dict: Dict[str, Any]
        if hasattr(metrics, "to_dict"):
            metrics_dict = metrics.to_dict()  # type: ignore[assignment]
        elif is_dataclass(metrics) and not isinstance(metrics, type):
            metrics_dict = asdict(metrics)
        else:
            metrics_dict = {"metrics": metrics}

        quality_df = pd.DataFrame([metrics_dict])
        try:
            suffix = str(config.get("liquidity_quality_artifact_suffix", ""))
            base_artifact_name = "ml_liquidity_regime_quality_15m"
            artifact_name = f"{base_artifact_name}{suffix}"

            quality_path = self._save_artifact(
                data=quality_df,
                artifact_name=artifact_name,
                artifact_type="data",
                metadata={
                    "min_regime_size": min_regime_size,
                },
            )
        except Exception as save_exc:
            tprint_warning(f"Failed to save liquidity regime quality artifact: {save_exc}")
            quality_path = None

        return metrics, quality_path

    def _map_probabilities_to_15m(
        self,
        *,
        proba_df: Optional[pd.DataFrame],
        market_data_1h: pd.DataFrame,
        symbol: str,
        exchange: str,
        direction: str,
        config: Dict[str, Any],
    ) -> Optional[pd.DataFrame]:
        if proba_df is None or proba_df.empty:
            return None

        output_timeframe = str(config.get("liquidity_output_timeframe", "15m"))
        regime_timeframe = str(config.get("regime_timeframe", output_timeframe))
        if output_timeframe == "1h":
            return proba_df

        # Load 15m market data to get target index
        market_data_15m, _ = self.load_market_data_or_fail(
            {**config, "timeframe": output_timeframe},
            pipeline_state={},
            allow_config_override=True,
        )

        if not isinstance(market_data_15m, pd.DataFrame) or market_data_15m.empty:
            tprint_warning("15m market data unavailable; skipping 1h→15m probability mapping")
            return None

        if not isinstance(market_data_15m.index, pd.DatetimeIndex):
            market_data_15m = market_data_15m.copy()
            market_data_15m.index = pd.to_datetime(market_data_15m.index)

        # Ensure probability index is DatetimeIndex
        if not isinstance(proba_df.index, pd.DatetimeIndex):
            idx = pd.to_datetime(proba_df.index)
            proba_df = proba_df.copy()
            proba_df.index = idx

        # If regimes are already computed on the 15m grid, simply align to the
        # 15m market index via forward-fill instead of applying 1h→15m mapping.
        if regime_timeframe == output_timeframe:
            mapped = proba_df.reindex(market_data_15m.index, method="ffill")

            # Optional early backfill: if probabilities only start late in the
            # history (e.g., specialist trained on a recent window), then
            # forward-fill alone will leave leading NaNs. For downstream
            # consumers like meta-labeling and specialist_feature_diagnostics,
            # this manifests as low coverage. When enabled, backfill the
            # earliest non-null 15m row to all preceding bars so that the
            # saved 15m probabilities form a continuous series over the
            # available market-data index.
            if bool(config.get("liquidity_backfill_early_probs", True)):
                try:
                    nonnull_mask = mapped.notna().any(axis=1)
                    if nonnull_mask.any():
                        first_pos = int(np.argmax(nonnull_mask.to_numpy()))
                        if first_pos > 0:
                            mapped.iloc[:first_pos] = mapped.iloc[first_pos]
                except Exception:
                    # Best-effort only; fall back to forward-fill-only
                    # behaviour if anything goes wrong here.
                    pass

            return mapped

        mode = str(config.get("liquidity_prob_interpolation_mode", "step")).lower()

        if mode == "linear":
            # Linear interpolation between consecutive 1h bars across 15m children
            one_h_index = proba_df.index.sort_values()
            if len(one_h_index) < 2:
                return proba_df.reindex(market_data_15m.index, method="ffill")

            # Build a continuous 15m index spanning the 1h data
            full_15m_index = pd.date_range(
                start=one_h_index.min(),
                end=one_h_index.max(),
                freq=output_timeframe,
            )
            # Reindex to 15m and interpolate linearly in time
            step_reindexed = proba_df.reindex(one_h_index)
            step_resampled = step_reindexed.resample(output_timeframe).ffill()
            interp_df = step_resampled.reindex(full_15m_index).interpolate(method="time")
            # Align to actual 15m market index
            mapped = interp_df.reindex(market_data_15m.index, method="nearest")
            return mapped

        # Default: step mapping from parent 1h bar via floor
        parent_index = market_data_15m.index.floor("1H")
        mapped = proba_df.reindex(parent_index, method="ffill")
        mapped.index = market_data_15m.index
        return mapped
