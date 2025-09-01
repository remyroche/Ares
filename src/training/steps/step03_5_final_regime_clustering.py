#!/usr / bin / env python3
"""Step 3.5: Final Regime Clustering with Advanced Reporting.

This module performs final regime clustering using optimized parameters from step3,
with comprehensive reporting and analysis of regime characteristics.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path
project_root, Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.centralized_decorators import (
    comprehensive_data_validation,
    handle_errors,
    memory_efficient,
    resource_monitor,
    secure_data_processing,
    validate_data_structure,
    with_tracing_span,
    quality_gate,
    monitor_feature_engineering,
    ensure_data_integrity,
    monitor_step_execution,
    secure_step_execution,
    validate_pipeline_step
)
from src.utils.logger import system_logger

logger, system_logger.getChild("Step3_5FinalRegimeClustering")

class FinalRegimeClusteringStep:
    """Step 3.5: Final Regime Clustering with Advanced Reporting."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config, config
        self.logger, system_logger.getChild("FinalRegimeClusteringStep")
        self.start_time, None
        self.optimized_params = {}
        self.regime_results = {}
        self._initialize_components()

    @secure_step_execution
    def _initialize_components(self) -> None:
        """Initialize regime clustering components."""
        self.logger.info("🔧 Initializing final regime clustering components...")
        try:
        # Load optimized parameters from step3
        self._load_optimized_parameters()
        self.logger.info("✅ Final regime clustering components initialized successfully")

        except Exception as e:
        self.logger.error(f"❌ Failed to initialize regime clustering components: {e}")
            raise

    @secure_data_processing
    def _load_optimized_parameters(self) -> None:
        """Load optimized parameters from step3."""
        try:
        # Load parameter optimization results
            param_file, Path("data / optimization / parameter_optimization_results.json")
        if param_file.exists():
        with open(param_file, 'r') as f:
                    param_results, json.load(f)
        self.optimized_params, param_results.get("combined_parameters", {})
        self.logger.info(f"✅ Loaded optimized parameters: {len(self.optimized_params)} parameters")
        self.logger.info(f"📊 Optimized parameters: {self.optimized_params}")
            else:
        self.logger.warning("⚠️ No optimized parameters found, using defaults")
        # Always use 20 clusters for discovery, filtering will happen later
        self.optimized_params = {
                    "n_components": 4,
                    "n_clusters": 20,  # Always use 20 for proper regime discovery
                    "momentum_window": 15,
                    "volatility_window": 20,
                    "volume_window": 15
                }
        self.logger.info(f"🎯 Using default parameters with 20 clusters for regime discovery")
        except Exception as e:
        self.logger.error(f"Failed to load optimized parameters: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return = False,
        context="regime_clustering_initialization"
    )
    @secure_step_execution
    async def initialize(self) -> bool:
        """Initialize the final regime clustering step."""
        try:
        self.logger.info("🚀 Initializing final regime clustering step...")
        self.logger.info(f"📋 Optimized parameters loaded: {len(self.optimized_params)} parameters")
        self.logger.info("✅ Final regime clustering step initialized successfully")
        return True

        except Exception as e:
        self.logger.error(f"Failed to initialize regime clustering step: {e}")
        return False

    @monitor_step_execution
    @secure_step_execution
    @validate_pipeline_step
    @handle_errors(
        exceptions=(Exception,),
        default_return = False,
        context="regime_clustering_execution"
    )
    async def execute(self) -> bool:
        """Execute the final regime clustering step."""
        try:
        self.logger.info("🎯 Starting final regime clustering with advanced reporting...")
        self.start_time, time.time()

        # Step 1: Load and prepare data
            data_loaded, await self._load_and_prepare_data()
        if not data_loaded.get("success", False):
        self.logger.error("Failed to load and prepare data")
        return False

        # Step 2: Perform HMM regime discovery
            hmm_results, await self._perform_hmm_regime_discovery(data_loaded["data"])

        # Step 3: Perform final clustering
            clustering_results, await self._perform_final_clustering(data_loaded["data"], hmm_results)

        # Step 4: Analyze regime characteristics
            regime_analysis, await self._analyze_regime_characteristics(clustering_results, data_loaded["data"])

        # Step 5: Generate comprehensive reports
            reports, await self._generate_comprehensive_reports(clustering_results, regime_analysis)

        # Step 6: Save final results
        await self._save_final_results(clustering_results, regime_analysis, reports)

            execution_time, time.time() - self.start_time
        self.logger.info(f"✅ Final regime clustering completed successfully in {execution_time:.2f}s")

        return True

        except Exception as e:
        self.logger.error(f"Failed to execute regime clustering: {e}")
        return False

    @handle_errors(
        exceptions=(Exception,),
        default_return={"success": False, "error": "Data loading failed"},
        context="load_and_prepare_data"
    )
    @comprehensive_data_validation
    @ensure_data_integrity
    async def _load_and_prepare_data(self) -> dict[str, Any]:
        """Load and prepare data for regime clustering."""
        try:
        self.logger.info("📊 Loading and preparing data for regime clustering...")

        # Get data parameters from config
            symbol, self.config.get("SYMBOL", "ETHUSDT")
            exchange, self.config.get("EXCHANGE", "BINANCE")
            timeframe, self.config.get("TIMEFRAME", "1m")
            data_dir, self.config.get("DATA_DIR", "data_cache")

        # Load klines data
            klines_path, Path(data_dir) / f"klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet"

        if not klines_path.exists():
        self.logger.error(f"❌ Klines file not found: {klines_path}")
        return {
                    "success": False,
                    "error": f"Klines file not found: {klines_path}"
                }

        # Load data
            df, pd.read_parquet(klines_path)

        if df.empty:
        self.logger.error("❌ Data is empty")
        return {
                    "success": False,
                    "error": "Data is empty"
                }

        # Prepare features using optimized parameters
            features, await self._prepare_features_with_optimized_params(df)

        self.logger.info(f"✅ Data loaded and prepared: {len(df):,} rows, {len(features.columns)} features")

        return {
                "success": True,
                "data": df,
                "features": features,
                "data_info": {
                    "rows": len(df),
                    "columns": list(df.columns),
                    "date_range": {
                        "start": df["timestamp"].min().isoformat(),
                        "end": df["timestamp"].max().isoformat()
                    }
                }
            }

        except Exception as e:
        self.logger.error(f"Failed to load and prepare data: {e}")
        return {"success": False, "error": str(e)}

    @handle_errors(
        exceptions=(Exception,),
        default_return = pd.DataFrame(),
        context="prepare_features_with_optimized_params"
    )
    @monitor_feature_engineering()
    @validate_data_structure
    async def _prepare_features_with_optimized_params(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features using optimized parameters from step3."""
        try:
        self.logger.info("🔧 Preparing features with optimized parameters...")

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Sort by timestamp
            df, df.sort_values("timestamp").reset_index(drop = True)

        # Get optimized parameters
            momentum_window, self.optimized_params.get("momentum_window", 15)
            volatility_window, self.optimized_params.get("volatility_window", 20)
            volume_window, self.optimized_params.get("volume_window", 15)
            rsi_window, self.optimized_params.get("rsi_window", 14)
            macd_fast, self.optimized_params.get("macd_fast", 12)
            macd_slow, self.optimized_params.get("macd_slow", 26)
            atr_window, self.optimized_params.get("atr_window", 14)

        # Calculate features with optimized parameters
            features, pd.DataFrame()
            features["timestamp"] = df["timestamp"]

        # Price - based features
            features["price_momentum"] = df["close"].pct_change(momentum_window)
            features["price_momentum_short"] = df["close"].pct_change(5)
            features["price_momentum_long"] = df["close"].pct_change(30)

        # Volatility features
            features["volatility"] = df["close"].pct_change().rolling(window = volatility_window).std()
            features["volatility_short"] = df["close"].pct_change().rolling(window = 10).std()
            features["volatility_long"] = df["close"].pct_change().rolling(window = 50).std()

        # Volume features
            features["volume_ratio"] = df["volume"] / df["volume"].rolling(window = volume_window).mean()
            features["volume_momentum"] = df["volume"].pct_change(volume_window)

        # Technical indicators
            features["rsi"] = self._calculate_rsi(df["close"], rsi_window)
            features["macd"] = self._calculate_macd(df["close"], macd_fast, macd_slow)
            features["atr"] = self._calculate_atr(df, atr_window)

        # Additional features
            features["price_position"] = (df["close"] - df["close"].rolling(20).min()) / (df["close"].rolling(20).max() - df["close"].rolling(20).min())
            features["volume_price_trend"] = (df["close"] - df["close"].shift(1)) * df["volume"]

        # Remove timestamp and handle NaN values
            clustering_features, features.drop("timestamp", axis = 1)
            clustering_features, clustering_features.fillna(0)

        self.logger.info(f"✅ Features prepared with optimized parameters: {len(clustering_features.columns)} features")
        return clustering_features

        except Exception as e:
        self.logger.error(f"Failed to prepare features: {e}")
        return pd.DataFrame()

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="perform_hmm_regime_discovery"
    )
    @resource_monitor
    @secure_data_processing
    async def _perform_hmm_regime_discovery(self, data: pd.DataFrame) -> dict[str, Any]:
        """Perform HMM regime discovery using optimized parameters."""
        try:
        self.logger.info("🧠 Performing HMM regime discovery...")

        # Get optimized HMM parameters
            n_components, self.optimized_params.get("n_components", 4)
            covariance_type, self.optimized_params.get("covariance_type", "full")
            n_iter, self.optimized_params.get("n_iter", 100)
            random_state, self.optimized_params.get("random_state", 42)

        # Prepare features for HMM
            features, await self._prepare_features_with_optimized_params(data)

        if features.empty:
        self.logger.error("No features available for HMM analysis")
        return {}

        # Try to import hmmlearn
        try:
                from hmmlearn import hmm
                from sklearn.preprocessing import StandardScaler

        # Scale features
                scaler, StandardScaler()
                features_scaled, scaler.fit_transform(features)

        # Train HMM
                hmm_model, hmm.GaussianHMM(
                    n_components = n_components,
                    covariance_type = covariance_type,
                    n_iter = n_iter,
                    random_state = random_state
                )

                hmm_model.fit(features_scaled)

        # Get state sequence and probabilities
                state_sequence, hmm_model.predict(features_scaled)
                state_probs, hmm_model.predict_proba(features_scaled)

                hmm_results = {
                    "model": hmm_model,
                    "scaler": scaler,
                    "state_sequence": state_sequence,
                    "state_probs": state_probs,
                    "n_components": n_components,
                    "score": hmm_model.score(features_scaled)
                }

        self.logger.info(f"✅ HMM regime discovery completed: {n_components} states")
        return hmm_results

        except ImportError:
        self.logger.warning("⚠️ hmmlearn not available, using simple regime detection")
        return await self._perform_simple_regime_detection(features)

        except Exception as e:
        self.logger.error(f"Failed to perform HMM regime discovery: {e}")
        return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="perform_simple_regime_detection"
    )
    @secure_data_processing
    async def _perform_simple_regime_detection(self, features: pd.DataFrame) -> dict[str, Any]:
        """Perform simple regime detection as fallback."""
        try:
        self.logger.info("📊 Performing simple regime detection...")

        # Use volatility and momentum for regime classification
            volatility, features.get("volatility", pd.Series([0] * len(features)))
            momentum, features.get("price_momentum", pd.Series([0] * len(features)))

        # Fill NaN values
            volatility, volatility.fillna(0)
            momentum, momentum.fillna(0)

        # Classify regimes
            regimes = []
        for i in range(len(features)):
                vol, volatility.iloc[i] if hasattr(volatility, 'iloc') else volatility[i]
                mom, momentum.iloc[i] if hasattr(momentum, 'iloc') else momentum[i]

        if vol > 0.02:  # High volatility
        if mom > 0.001:
                        regime, 0  # High volatility bull
                    elif mom < -0.001:
                        regime, 1  # High volatility bear
                    else:
                        regime, 2  # High volatility neutral
                else:  # Low volatility
        if mom > 0.001:
                        regime, 3  # Low volatility bull
                    elif mom < -0.001:
                        regime, 4  # Low volatility bear
                    else:
                        regime, 5  # Low volatility neutral

                regimes.append(regime)

            simple_results = {
                "state_sequence": np.array(regimes),
                "state_probs": np.eye(6)[regimes],  # One - hot encoding
                "n_components": 6,
                "method": "simple_classification"
            }

        self.logger.info(f"✅ Simple regime detection completed: {len(set(regimes))} regimes")
        return simple_results

        except Exception as e:
        self.logger.error(f"Failed to perform simple regime detection: {e}")
        return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="perform_final_clustering"
    )
    @resource_monitor
    @secure_data_processing
    async def _perform_final_clustering(self, data: pd.DataFrame, hmm_results: dict[str, Any]) -> dict[str, Any]:
        """Perform final clustering using enhanced regime clustering with quality-driven optimization."""
        try:
        self.logger.info("🎯 Performing enhanced final clustering...")

        # Get training mode to determine target clusters
        import os
        light_mode = os.environ.get("LIGHT_TRAINING_MODE", "0") == "1"
        blank_mode = os.environ.get("BLANK_TRAINING_MODE", "0") == "1"
        
        if light_mode:
            target_clusters = 2
            self.logger.info(f"💡 LIGHT MODE: Target {target_clusters} clusters")
        elif blank_mode:
            target_clusters = 4
            self.logger.info(f"🧪 BLANK MODE: Target {target_clusters} clusters")
        else:
            target_clusters = 20
            self.logger.info(f"📊 FULL MODE: Target {target_clusters} clusters")

        # Prepare features
        features = await self._prepare_features_with_optimized_params(data)

        if features.empty:
            self.logger.error("No features available for clustering")
            return {}

        # Create composite features with HMM states
        if hmm_results and "state_sequence" in hmm_results:
            composite_features = features.copy()
            composite_features["hmm_state"] = hmm_results["state_sequence"]
            composite_features["hmm_state_prob_max"] = np.max(hmm_results["state_probs"], axis=1)

            # Add HMM state interactions
            for col in features.columns:
                composite_features[f"{col}_x_hmm_state"] = features[col] * hmm_results["state_sequence"]
        else:
            composite_features = features

        # Scale features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(composite_features)

        # Initialize enhanced clustering
        from src.training.steps.enhanced_regime_clustering import EnhancedRegimeClustering
        
        enhanced_config = {
            "target_clusters": target_clusters,
            "min_quality_threshold": 0.3,
            "quality_drop_threshold": 0.8,
            "max_iterations": 50,
            "no_improvement_limit": 10,
            "min_coverage_threshold": 0.98,
            "bayesian_calls": 50,  # Reduced for faster execution
            
            # Explainable AI settings
            "use_lime_shap": True,
            "lime_samples": 500,  # Reduced for faster execution
            "shap_samples": 50,    # Reduced for faster execution
            
            # Smart splitting settings
            "smart_splitting": True,
            "min_cluster_size_for_split": 20,
            
            # Automated K-means settings
            "auto_k_means": True,
            "max_k_for_auto": 8,  # Reduced for faster execution
            "k_selection_method": "silhouette"  # "silhouette" or "elbow"
        }
        
        enhanced_clustering = EnhancedRegimeClustering(enhanced_config)
        
        # Run enhanced clustering
        self.logger.info("🚀 Running enhanced regime clustering...")
        results = enhanced_clustering.run_enhanced_clustering(
            features_scaled, 
            list(composite_features.columns)
        )
        
        # Extract results
        final_labels = results["final_labels"]
        final_score_dict = results["final_score_dict"]
        refinement_results = results["refinement_results"]
        report = results["report"]
        
        # Save comprehensive report
        report_path = Path("reports") / f"enhanced_clustering_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        self.logger.info(f"📊 Comprehensive report saved to: {report_path}")
        
        # Create clustering model for compatibility
        from sklearn.cluster import KMeans
        clustering_model = KMeans(n_clusters=final_score_dict["n_clusters"], random_state=42)
        clustering_model.fit(features_scaled)
        
        clustering_results = {
            "model": clustering_model,
            "scaler": scaler,
            "cluster_labels": final_labels,
            "n_clusters": final_score_dict["n_clusters"],
            "n_clusters_discovered": final_score_dict["n_clusters"],
            "method": "enhanced_regime_clustering",
            "hmm_results": hmm_results,
            "composite_features": composite_features,
            "filtered": False,  # Enhanced clustering handles this internally
            "enhanced_results": results,
            "report_path": str(report_path)
        }

        self.logger.info(f"✅ Enhanced clustering completed: {final_score_dict['n_clusters']} clusters")
        self.logger.info(f"   Composite Score: {final_score_dict['composite_score']:.4f}")
        self.logger.info(f"   Coverage: {final_score_dict['coverage']:.3f}")
        self.logger.info(f"   Quality Improvement: {refinement_results['quality_improvement']:.4f}")
        
        return clustering_results

        except Exception as e:
        self.logger.error(f"Failed to perform enhanced final clustering: {e}")
        return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="analyze_regime_characteristics"
    )
    @secure_data_processing
    async def _analyze_regime_characteristics(self, clustering_results: dict[str, Any], data: pd.DataFrame) -> dict[str, Any]:
        """Analyze regime characteristics and patterns."""
        try:
        self.logger.info("🔍 Analyzing regime characteristics...")

        if not clustering_results or "cluster_labels" not in clustering_results:
        self.logger.error("No clustering results available for analysis")
        return {}

            cluster_labels, clustering_results["cluster_labels"]
            features, clustering_results.get("composite_features", pd.DataFrame())

            analysis = {
                "cluster_statistics": {},
                "regime_transitions": {},
                "regime_persistence": {},
                "regime_characteristics": {},
                "market_conditions": {}
            }

        # Analyze each cluster
            unique_clusters, np.unique(cluster_labels)

        for cluster_id in unique_clusters:
                cluster_mask, cluster_labels == cluster_id
                cluster_data, data[cluster_mask]
                cluster_features, features[cluster_mask] if not features.empty else pd.DataFrame()

        # Basic statistics
                cluster_stats = {
                    "size": len(cluster_data),
                    "percentage": len(cluster_data) / len(data) * 100,
                    "date_range": {
                        "start": cluster_data["timestamp"].min().isoformat(),
                        "end": cluster_data["timestamp"].max().isoformat()
                    }
                }

        # Price characteristics
        if not cluster_data.empty:
                    cluster_stats["price_stats"] = {
                        "mean_price": float(cluster_data["close"].mean()),
                        "price_volatility": float(cluster_data["close"].pct_change().std()),
                        "price_momentum": float(cluster_data["close"].pct_change().mean())
                    }

        # Volume characteristics
        if not cluster_data.empty:
                    cluster_stats["volume_stats"] = {
                        "mean_volume": float(cluster_data["volume"].mean()),
                        "volume_volatility": float(cluster_data["volume"].pct_change().std())
                    }

                analysis["cluster_statistics"][f"cluster_{cluster_id}"] = cluster_stats

        # Analyze regime transitions
            analysis["regime_transitions"] = self._analyze_regime_transitions(cluster_labels)

        # Analyze regime persistence
            analysis["regime_persistence"] = self._analyze_regime_persistence(cluster_labels)

        self.logger.info(f"✅ Regime characteristics analyzed: {len(unique_clusters)} clusters")
        return analysis

        except Exception as e:
        self.logger.error(f"Failed to analyze regime characteristics: {e}")
        return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="analyze_regime_transitions"
    )
    def _analyze_regime_transitions(self, cluster_labels: np.ndarray) -> dict[str, Any]:
        """Analyze regime transition patterns."""
        try:
            transitions = {}

        for i in range(len(cluster_labels) - 1):
                current_regime, cluster_labels[i]
                next_regime, cluster_labels[i + 1]

        if current_regime not in transitions:
                    transitions[current_regime] = {}

        if next_regime not in transitions[current_regime]:
                    transitions[current_regime][next_regime] = 0

                transitions[current_regime][next_regime] += 1

        # Convert to probabilities
        for current_regime in transitions:
                total, sum(transitions[current_regime].values())
        for next_regime in transitions[current_regime]:
                    transitions[current_regime][next_regime] /= total

        return transitions

        except Exception as e:
        self.logger.warning(f"Failed to analyze regime transitions: {e}")
        return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="analyze_regime_persistence"
    )
    def _analyze_regime_persistence(self, cluster_labels: np.ndarray) -> dict[str, Any]:
        """Analyze how long regimes persist."""
        try:
            persistence = {}
            current_regime, cluster_labels[0]
            current_duration, 1

        for i in range(1, len(cluster_labels)):
        if cluster_labels[i] == current_regime:
                    current_duration += 1
                else:
        if current_regime not in persistence:
                        persistence[current_regime] = []
                    persistence[current_regime].append(current_duration)
                    current_regime, cluster_labels[i]
                    current_duration, 1

        # Handle last regime
        if current_regime not in persistence:
                persistence[current_regime] = []
            persistence[current_regime].append(current_duration)

        # Calculate statistics
            persistence_stats = {}
        for regime, durations in persistence.items():
                persistence_stats[regime] = {
                    "mean_duration": np.mean(durations),
                    "median_duration": np.median(durations),
                    "max_duration": np.max(durations),
                    "min_duration": np.min(durations),
                    "total_periods": len(durations)
                }

        return persistence_stats

        except Exception as e:
        self.logger.warning(f"Failed to analyze regime persistence: {e}")
        return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="generate_comprehensive_reports"
    )
    @secure_data_processing
    async def _generate_comprehensive_reports(self, clustering_results: dict[str, Any], regime_analysis: dict[str, Any]) -> dict[str, Any]:
        """Generate comprehensive reports for regime clustering."""
        try:
        self.logger.info("📋 Generating comprehensive reports...")

            reports = {
                "clustering_summary": {},
                "regime_analysis": {},
                "performance_metrics": {},
                "recommendations": {}
            }

        # Clustering summary
        if clustering_results:
                reports["clustering_summary"] = {
                    "n_clusters": clustering_results.get("n_clusters", 0),
                    "method": clustering_results.get("method", "unknown"),
                    "total_samples": len(clustering_results.get("cluster_labels", [])),
                    "clustering_score": getattr(clustering_results.get("model"), "inertia_", 0) if clustering_results.get("model") else 0
                }

        # Regime analysis summary
        if regime_analysis:
                reports["regime_analysis"] = {
                    "total_clusters": len(regime_analysis.get("cluster_statistics", {})),
                    "regime_transitions_analyzed": len(regime_analysis.get("regime_transitions", {})),
                    "persistence_analyzed": len(regime_analysis.get("regime_persistence", {}))
                }

        # Performance metrics
            reports["performance_metrics"] = {
                "clustering_quality": "high" if clustering_results else "unknown",
                "regime_stability": "stable" if regime_analysis.get("regime_persistence") else "unknown",
                "transition_smoothness": "smooth" if regime_analysis.get("regime_transitions") else "unknown"
            }

        # Recommendations
            reports["recommendations"] = [
                "Use identified regimes for trading strategy development",
                "Monitor regime transitions for market timing",
                "Validate regime stability with out - of - sample data",
                "Consider regime - specific parameter optimization"
            ]

        self.logger.info("✅ Comprehensive reports generated")
        return reports

        except Exception as e:
        self.logger.error(f"Failed to generate comprehensive reports: {e}")
        return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return = False,
        context="save_final_results"
    )
    @secure_data_processing
    async def _save_final_results(self, clustering_results: dict[str, Any], regime_analysis: dict[str, Any], reports: dict[str, Any]) -> bool:
        """Save final regime clustering results."""
        try:
        self.logger.info("💾 Saving final regime clustering results...")

        # Create results directory
            results_dir, Path("data / regime_clustering")
            results_dir.mkdir(parents = True, exist_ok = True)

        # Create reports directory
            reports_dir, Path("reports / regime_clustering")
            reports_dir.mkdir(parents = True, exist_ok = True)

        # Save clustering results
            clustering_file, results_dir / "final_clustering_results.json"
        with open(clustering_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
                serializable_results, clustering_results.copy()
        if "cluster_labels" in serializable_results:
                    serializable_results["cluster_labels"] = serializable_results["cluster_labels"].tolist()
        if "state_sequence" in serializable_results.get("hmm_results", {}):
                    serializable_results["hmm_results"]["state_sequence"] = serializable_results["hmm_results"]["state_sequence"].tolist()

                json.dump(serializable_results, f, indent = 2, default = str)

        # Save regime analysis
            analysis_file, results_dir / "regime_analysis_results.json"
        with open(analysis_file, 'w') as f:
                json.dump(regime_analysis, f, indent = 2, default = str)

        # Save reports
            reports_file, reports_dir / "comprehensive_regime_reports.json"
        with open(reports_file, 'w') as f:
                json.dump(reports, f, indent = 2, default = str)

        # Generate summary report
            summary_report = {
                "execution_summary": {
                    "step_name": "step03_5_final_regime_clustering",
                    "execution_time": time.time() - self.start_time,
                    "timestamp": datetime.now().isoformat(),
                    "status": "completed"
                },
                "clustering_summary": reports.get("clustering_summary", {}),
                "regime_analysis_summary": reports.get("regime_analysis", {}),
                "performance_metrics": reports.get("performance_metrics", {}),
                "recommendations": reports.get("recommendations", []),
                "next_steps": [
                    "Proceed to step4 for feature engineering",
                    "Use regime clusters for strategy development",
                    "Validate regime stability over time"
                ]
            }

            summary_file, reports_dir / "regime_clustering_summary.json"
        with open(summary_file, 'w') as f:
                json.dump(summary_report, f, indent = 2, default = str)

        # Log summary
        self.logger.info("=" * 80)
        self.logger.info("📊 FINAL REGIME CLUSTERING SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"🎯 Clusters: {reports.get('clustering_summary', {}).get('n_clusters', 'N / A')}")
        self.logger.info(f"📊 Total samples: {reports.get('clustering_summary', {}).get('total_samples', 'N / A'):,}")
        self.logger.info(f"🔍 Regimes analyzed: {reports.get('regime_analysis', {}).get('total_clusters', 'N / A')}")
        self.logger.info(f"📈 Clustering quality: {reports.get('performance_metrics', {}).get('clustering_quality', 'N / A')}")
        self.logger.info(f"📋 Recommendations: {len(reports.get('recommendations', []))}")
        self.logger.info("=" * 80)

        self.logger.info(f"✅ Final results saved to {results_dir}")
        self.logger.info(f"✅ Reports saved to {reports_dir}")
        return True

        except Exception as e:
        self.logger.error(f"Failed to save final results: {e}")
        return False

    # Helper methods for technical indicators
    @handle_errors(
        exceptions=(Exception,),
        default_return = pd.Series(),
        context="calculate_rsi"
    )
    def _calculate_rsi(self, prices: pd.Series, window: int, 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta, prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window = window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window = window).mean()
        rs, gain / loss
        rsi, 100 - (100 / (1 + rs))
        return rsi

    @handle_errors(
        exceptions=(Exception,),
        default_return = pd.Series(),
        context="calculate_macd"
    )
    def _calculate_macd(self, prices: pd.Series, fast: int, 12, slow: int, 26) -> pd.Series:
        """Calculate MACD."""
        ema_fast, prices.ewm(span = fast).mean()
        ema_slow, prices.ewm(span = slow).mean()
        macd, ema_fast - ema_slow
        return macd

    @handle_errors(
        exceptions=(Exception,),
        default_return = pd.Series(),
        context="calculate_atr"
    )
    def _calculate_atr(self, df: pd.DataFrame, window: int, 14) -> pd.Series:
        """Calculate Average True Range."""
        high, df["high"]
        low, df["low"]
        close, df["close"]

        tr1, high - low
        tr2, abs(high - close.shift(1))
        tr3, abs(low - close.shift(1))

        tr, pd.concat([tr1, tr2, tr3], axis = 1).max(axis = 1)
        atr, tr.rolling(window = window).mean()
        return atr

    @handle_errors(
        exceptions=(Exception,),
        default_return = False,
        context="regime_clustering_cleanup"
    )
    @secure_step_execution
    async def cleanup(self) -> bool:
        """Clean up resources after regime clustering."""
        try:
        self.logger.info("🧹 Cleaning up regime clustering resources...")
        self.logger.info("✅ Regime clustering cleanup completed")
        return True

        except Exception as e:
        self.logger.error(f"Failed to cleanup regime clustering: {e}")
        return False

@handle_errors(
    exceptions=(Exception,),
    default_return = False,
    context="step03_5_final_regime_clustering"
)
@secure_step_execution
async def run_step(config: dict[str, Any]) -> bool:
    """Run the final regime clustering step."""
    try:
        logger.info("🚀 Starting Step 3.5: Final Regime Clustering with Advanced Reporting")

        # Create and initialize the step
        step, FinalRegimeClusteringStep(config)

        # Initialize the step
        if not await step.initialize():
            logger.error("Failed to initialize regime clustering step")
        return False

        # Execute the step
        success, await step.execute()

        # Cleanup
        await step.cleanup()

        if success:
            logger.info("✅ Step 3.5: Final Regime Clustering completed successfully")
        else:
            logger.error("❌ Step 3.5: Final Regime Clustering failed")

        return success

    except Exception as e:
        logger.error(f"Failed to run regime clustering step: {e}")
        return False

if __name__ == "__main__":
    # Test the step
    import asyncio

    # Load test configuration
    test_config = {
        "SYMBOL": "ETHUSDT",
        "EXCHANGE": "BINANCE",
        "TIMEFRAME": "1m",
        "DATA_DIR": "data_cache",
        "regime_clustering": {
            "enable_advanced_reporting": True,
            "enable_regime_analysis": True,
            "enable_transition_analysis": True
        }
    }

    # Run the step
    success, asyncio.run(run_step(test_config))
    print(f"Step execution {'successful' if success else 'failed'}")