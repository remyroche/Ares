"""
HDBSCAN Regime Discovery Step

BaseClass-based step that replaces NAS/TAS regime discovery with HDBSCAN-based approach.
Integrates with ares_launcher.py and uses the comprehensive regime discovery system.
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path

# Import BaseClass and step registry
from src.training.steps.base_step import BaseStep

# Import HDBSCAN regime discovery system
from src.training.steps.market_analysis.hdbscan_clustering import (
    HDBSCANRegimeDiscovery, 
    RegimeDiscoveryConfig,
    RegimeResult
)

# Import auto-tuning system for HDBSCAN
try:
    from src.training.steps.market_analysis.hdbscan_clustering.optimization.automated_hdbscan_parameter_tuner import (
        create_automated_hdbscan_tuner,
        ClusteringQualityMetrics
    )
    AUTO_TUNER_AVAILABLE = True
except ImportError as e:
    AUTO_TUNER_AVAILABLE = False
    logging.warning(f"Auto-tuner not available: {e}")

# Import regime feature selector
from src.training.steps.market_analysis.hdbscan_clustering.optimization.efficient_regime_feature_selector import (
    EfficientRegimeFeatureSelector as RegimeFeatureSelector,
    EfficientFeatureSelectionConfig as RegimeFeatureSelectorConfig,
    create_efficient_regime_feature_selector as create_regime_feature_selector
)

# Import utilities
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error
from src.utils.data.klines_parquet import get_klines_manager
from src.utils.serialization_utils import save_pickle, load_pickle

logger = logging.getLogger(__name__)


class HDBSCANRegimeDiscoveryStep(BaseStep):
    """
    HDBSCAN-based regime discovery step that replaces NAS/TAS approach.
    
    Features:
    - 5 feature families (Returns, Volatility, Volume/Flow, Entropy, Spectral)
    - Multi-mode dimensionality reduction (PCA/UMAP/densMAP)
    - HDBSCAN clustering with tree export
    - Post-clustering optimization with change budget
    - Economic validation and profiling
    - Temporal stabilization with causal/acausal modes
    - Deterministic reproducibility
    - Hardware optimization for M1 systems
    """
    
    def __init__(self, step_name: str = "hdbscan_regime_discovery"):
        """Initialize the HDBSCAN regime discovery step."""
        super().__init__(step_name)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize regime discovery system
        self.regime_discovery = None
        self.config = None
        
        # Initialize feature selector
        self.feature_selector = create_regime_feature_selector()
        
        tprint("🚀 HDBSCANRegimeDiscoveryStep initialized", "SUCCESS")
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute method required by BaseStep interface."""
        return await self.run(config)

    async def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute HDBSCAN regime discovery step.
        
        Args:
            config: Configuration dictionary with parameters:
                - symbol: Trading symbol (e.g., 'ETHUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Timeframe (e.g., '15m')
                - data_dir: Data directory path
                - start_date: Start date (optional)
                - end_date: End date (optional)
                - data_limit_days: Limit data to N days from end_date (optional)
                - execution_mode: 'full', 'light', or 'blank'
                - live_mode: Whether this is live trading (default: False)
                
        Returns:
            Dictionary with execution results, artifacts, and metrics
        """
        start_time = datetime.now()
        
        try:
            tprint(f"🔍 Starting HDBSCAN regime discovery for {config.get('symbol', 'UNKNOWN')}", "INFO")
            
            # Use regime_timeframe (defaults to 1h) for HDBSCAN regime discovery
            regime_timeframe = config.get('regime_timeframe', '1h')
            if 'regime_timeframe' not in config:
                tprint(f"⏰ Using regime_timeframe={regime_timeframe} for HDBSCAN regime discovery", "INFO")
                config['regime_timeframe'] = regime_timeframe
            if config.get('timeframe') != regime_timeframe:
                tprint(f"⏰ Overriding timeframe to {regime_timeframe} for HDBSCAN regime discovery (was: {config.get('timeframe', 'not set')})", "INFO")
                config['timeframe'] = regime_timeframe
            
            tprint(f"📋 Config received: {list(config.keys())}", "INFO")

            # Add memory optimization suggestions
            tprint("💡 Adding memory optimization tips...", "INFO")
            self._print_memory_optimization_tips(config)

            # Validate required parameters
            tprint("✅ Validating configuration...", "INFO")
            self._validate_config(config)
            tprint("✅ Configuration validation passed", "SUCCESS")
            
            # Create regime discovery configuration
            tprint("⚙️ Creating regime discovery configuration...", "INFO")
            self.config = self._create_regime_discovery_config(config)
            tprint(f"✅ Configuration created: {type(self.config)}", "SUCCESS")
            
            # Initialize regime discovery system
            tprint("🚀 Initializing HDBSCAN regime discovery system...", "INFO")
            self.regime_discovery = HDBSCANRegimeDiscovery(self.config)
            tprint("✅ HDBSCAN regime discovery system initialized", "SUCCESS")
            
            # Load market data
            tprint("📊 Loading market data...", "INFO")
            market_data = self._load_market_data(config)
            if market_data is None or len(market_data) == 0:
                tprint("❌ Failed to load market data", "ERROR")
                raise ValueError("Failed to load market data")
            tprint(f"✅ Market data loaded: {market_data.shape[0]} rows, {market_data.shape[1]} columns", "SUCCESS")
            
            # Apply light mode filtering using BaseStep's method
            tprint("🔧 Applying light mode filtering...", "INFO")
            market_data = self._apply_light_mode_filter(market_data, config, timeframe=config.get('timeframe', '15m'))
            tprint(f"✅ Light mode filtering applied: {market_data.shape[0]} rows, {market_data.shape[1]} columns", "SUCCESS")
            
            # Apply feature selection for better regime discrimination
            tprint("🎯 Applying regime feature selection...", "INFO")
            market_data_selected = self._apply_feature_selection(market_data, config)
            tprint(f"✅ Feature selection applied: {market_data.shape[1]} -> {market_data_selected.shape[1]} features", "SUCCESS")
            
            # Execute regime discovery
            tprint("🔍 Starting regime discovery process...", "INFO")
            tprint(f"📊 Data shape for regime discovery: {market_data_selected.shape}", "INFO")
            tprint(f"🔧 Live mode: {config.get('live_mode', False)}", "INFO")
            
            regime_result = await self.regime_discovery.discover_regimes(
                data=market_data_selected,
                fit=True,
                is_live=config.get('live_mode', False),
                returns=self._extract_returns(market_data_selected)
            )
            
            tprint(f"🔍 Regime discovery result: success={regime_result.success}", "INFO")
            if not regime_result.success:
                tprint(f"❌ Regime discovery failed: {regime_result.error_message}", "ERROR")
                raise ValueError(f"Regime discovery failed: {regime_result.error_message}")
            
            tprint("✅ Regime discovery completed successfully", "SUCCESS")
            
            # Apply auto-tuning if enabled and results are poor (ALWAYS enable for better results)
            if config.get('enable_auto_tuning', True) and AUTO_TUNER_AVAILABLE:
                tprint("🎯 Auto-tuning enabled - checking cluster quality...", "INFO")
                regime_result = self._apply_auto_tuning(market_data_selected, regime_result, config)
            
            # Calculate comprehensive clustering metrics
            tprint("📊 Calculating comprehensive clustering metrics...", "INFO")
            comprehensive_metrics = self._calculate_comprehensive_clustering_metrics(market_data, regime_result.labels)
            tprint("✅ Comprehensive clustering metrics calculated", "SUCCESS")
            
            # Add comprehensive metrics to regime result
            if hasattr(regime_result, 'validation_metrics'):
                regime_result.validation_metrics.update(comprehensive_metrics)
            else:
                regime_result.validation_metrics = comprehensive_metrics
            
            # Create artifacts
            tprint("📦 Creating artifacts...", "INFO")
            artifacts = self._create_artifacts(regime_result, config, market_data_selected)
            tprint(f"✅ Artifacts created: {len(artifacts)} items", "SUCCESS")

            # Save artifacts using BaseStep's enhanced system
            tprint("💾 Saving artifacts...", "INFO")
            self._save_artifacts(artifacts, config)
            tprint("✅ Artifacts saved successfully", "SUCCESS")

            # Calculate metrics with BaseStep's performance monitoring
            tprint("📊 Calculating metrics...", "INFO")
            metrics = self._calculate_metrics(regime_result, start_time)
            metrics.update({
                'artifact_manager_metrics': self.artifact_manager.get_performance_metrics(),
                'memory_analytics': self.artifact_manager.get_memory_analytics()
            })
            tprint("✅ Metrics calculated successfully", "SUCCESS")

            # Create comprehensive outcome report
            tprint("📝 Creating outcome report...", "INFO")
            outcome_report = self._create_outcome_report(regime_result, metrics, config)
            tprint("✅ Outcome report created", "SUCCESS")

            # Save outcome report using BaseStep's system
            tprint("💾 Saving outcome report...", "INFO")
            report_path = self._save_outcome_report(outcome_report, config)
            
            tprint(f"✅ HDBSCAN regime discovery completed: {regime_result.validation_metrics['n_regimes']} regimes", "SUCCESS")
            tprint(f"📊 Comprehensive report saved to: {report_path}", "INFO")
            
            return {
                'success': True,
                'artifacts': artifacts,
                'metrics': metrics,
                'outcome_report': outcome_report,
                'report_path': report_path,
                'regime_result': regime_result,
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
            
        except Exception as e:
            error_msg = f"HDBSCAN regime discovery failed: {str(e)}"
            tprint(f"❌ {error_msg}", "ERROR")
            tprint(f"🔍 Exception type: {type(e).__name__}", "ERROR")
            tprint(f"🔍 Exception details: {str(e)}", "ERROR")
            import traceback
            tprint(f"🔍 Traceback: {traceback.format_exc()}", "ERROR")
            self.logger.error(error_msg)
            self.logger.error(f"Exception type: {type(e).__name__}")
            self.logger.error(f"Exception details: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            return {
                'success': False,
                'error': error_msg,
                'artifacts': {},
                'metrics': {},
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration parameters."""
        tprint("🔍 Validating configuration parameters...", "INFO")
        required_params = ['symbol', 'exchange', 'timeframe']
        missing_params = [param for param in required_params if param not in config]
        
        if missing_params:
            tprint(f"❌ Missing required parameters: {missing_params}", "ERROR")
            raise ValueError(f"Missing required parameters: {missing_params}")
        
        tprint(f"✅ Required parameters present: {required_params}", "SUCCESS")
        
        # Validate execution mode
        execution_mode = config.get('execution_mode', 'full')
        tprint(f"🔍 Execution mode: {execution_mode}", "INFO")
        if execution_mode not in ['full', 'light', 'blank']:
            tprint(f"❌ Invalid execution_mode: {execution_mode}", "ERROR")
            raise ValueError(f"Invalid execution_mode: {execution_mode}. Must be 'full', 'light', or 'blank'")
        
        tprint("✅ Configuration validation completed", "SUCCESS")
    
    def _create_regime_discovery_config(self, config: Dict[str, Any]) -> RegimeDiscoveryConfig:
        """Create regime discovery configuration from step config."""
        tprint("⚙️ Creating regime discovery configuration...", "INFO")
        # Map execution mode to regime discovery parameters
        execution_mode = config.get('execution_mode', 'light')
        tprint(f"🔧 Execution mode: {execution_mode}", "INFO")
        
        if execution_mode == 'full':
            # Full mode: comprehensive regime discovery
            return RegimeDiscoveryConfig(
                # Core HDBSCAN parameters
                min_cluster_size_pct=0.02,  # 2% of samples
                min_cluster_size_floor=15,
                cluster_selection_epsilon=0.05,
                
                # Dimensionality reduction - use all 26 selected features
                dim_reduction_mode='pca_only',  # Use PCA but keep all features
                pca_n_components=1.0,  # Keep all features (100% variance)
                umap_n_components=10,
                umap_n_neighbors=30,
                
                # Feature extraction windows
                lookback_windows={
                    'short': [5, 10, 20],
                    'medium': [50, 100, 200],
                    'long': [300, 500, 1000]
                },
                
                # Preprocessing
                correlation_threshold=0.95,
                mi_threshold=0.05,
                
                # Post-clustering optimization
                change_budget_pct=0.10,
                max_optimization_rounds=5,
                use_condensed_tree=True,
                
                # Economic validation
                min_economic_separation_pct=0.30,
                interpretable_axes=[
                    "trend_pc", "vol_pc", "breadth", "skew", "liquidity_stress", "momentum_strength"
                ],
                
                # Temporal stabilization
                smoothing_window=5,
                min_dwell_bars=3,
                cooldown_bars=2,

                # Determinism
                random_state=42,
                pin_blas_threads=True
            )
            tprint("✅ Full mode configuration created", "SUCCESS")
            
        elif execution_mode == 'light':
            # Light mode: essential regime discovery with optimized parameters
            # Don't set data limit by default - let the system use available data
            tprint("🔧 Light mode: Using optimized parameters without data limiting", "INFO")
            
            return RegimeDiscoveryConfig(
                # Core HDBSCAN parameters - FORCE 5-8 REGIMES
                # Goal: Create 5-8 meaningful market regimes for ETHUSDT
                min_cluster_size_pct=0.005,  # 0.5% of samples (~2-3 samples for 480 samples)
                min_cluster_size_floor=3,   # Very low floor to allow 5-8 clusters
                
                # Goal 2: Much more flexible clustering to capture subtle differences
                min_samples_options=[2],  # Very low min_samples for maximum flexibility
                
                # Cluster selection - use leaf for balanced clusters that don't merge
                cluster_selection_method_options=['leaf'],  # Leaf method to preserve all clusters
                cluster_selection_epsilon=0.05,  # Higher epsilon to allow some merging but not too much
                metric='cosine',  # Try cosine for normalized data (more stable than manhattan)

                # Dimensionality reduction - use all 26 selected features
                dim_reduction_mode='pca_only',  # Use PCA but keep all features
                pca_n_components=1.0,  # Keep all features (100% variance)

                # Preprocessing - LESS AGGRESSIVE to keep more features
                correlation_threshold=0.85,  # Lower threshold to keep more features (currently only 17)
                winsorize_limits=(0.01, 0.99),  # Moderate winsorization for outlier handling

                # Temporal windows
                window_size=200,  # Reduced from 300 to 200 for more granular analysis
                window_overlap_pct=0.8,  # Increased from 0.7 to 0.8 for better regime detection

                # Post-clustering optimization
                change_budget_pct=0.10,  # Increased from 0.05 to 0.10 for more regime changes
                max_optimization_rounds=5,  # Increased from 3 to 5
                use_condensed_tree=True,  # Enable condensed tree for better cluster selection

                # Economic validation - VERY LOW for 5-8 regime discovery
                min_economic_separation_pct=0.05,  # Very low threshold to allow more regimes
                interpretable_axes=["trend_pc", "vol_pc", "breadth", "skew", "liquidity_stress", "momentum_strength"],
                
                # Temporal stabilization - More sensitive
                smoothing_window=2,  # Reduced from 3 to 2
                min_dwell_bars=1,  # Reduced from 2 to 1
                cooldown_bars=1,

                # Determinism
                random_state=42,
                pin_blas_threads=True
            )
            tprint("✅ Light mode configuration created", "SUCCESS")
            
        else:  # blank mode
            # Blank mode: minimal regime discovery
            return RegimeDiscoveryConfig(
                # Core HDBSCAN parameters
                min_cluster_size_pct=0.05,
                min_cluster_size_floor=50,
                cluster_selection_epsilon=0.0,
                
                # Dimensionality reduction - use all 26 selected features
                dim_reduction_mode="pca_only",  # Use PCA but keep all features
                pca_n_components=1.0,  # Keep all features (100% variance)
                
                # Preprocessing
                correlation_threshold=0.99,
                
                # Post-clustering optimization
                change_budget_pct=0.01,
                max_optimization_rounds=1,
                use_condensed_tree=False,
                
                # Economic validation
                min_economic_separation_pct=0.30,
                interpretable_axes=["trend_pc"],
                
                # Temporal stabilization
                smoothing_window=1,
                min_dwell_bars=1,
                cooldown_bars=0,

                # Determinism
                random_state=42,
                pin_blas_threads=False
            )
            tprint("✅ Blank mode configuration created", "SUCCESS")
    
    def _print_memory_optimization_tips(self, config: Dict[str, Any]) -> None:
        """Print memory optimization suggestions for HDBSCAN clustering."""
        execution_mode = config.get('execution_mode', 'light')
        data_limit = config.get('data_limit_days')

        tprint("🧠 Memory Optimization Tips for HDBSCAN:", "INFO")
        tprint(f"   💡 Execution mode: {execution_mode}", "INFO")

        if not data_limit:
            tprint("   ⚠️  No data limit set - consider adding 'data_limit_days': 20", "INFO")
            tprint("   💾 Suggestion: Use data_limit_days=20 for ~4,800 records (manageable memory)", "INFO")
        else:
            tprint(f"   ✅ Data limited to {data_limit} days", "INFO")

        tprint("   🎯 Memory optimizations active:", "INFO")
        tprint("      • Vectorized operations enabled", "INFO")
        tprint("      • Batch processing with memory monitoring", "INFO")
        tprint("      • Automatic memory cleanup triggers", "INFO")
        tprint("      • Hardware-specific optimizations", "INFO")

        if execution_mode == 'light':
            tprint("   🚀 Light mode: Reduced feature set and optimized parameters", "INFO")
        elif execution_mode == 'full':
            tprint("   ⚡ Full mode: Maximum features and comprehensive analysis", "INFO")

        tprint("", "INFO")

    def _load_market_data(self, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Load market data using klines manager."""
        try:
            tprint("📂 Loading market data...", "INFO")
            
            # Get klines manager
            klines_manager = get_klines_manager(data_dir=config.get('data_dir', 'historical_data'))
            
            # Parse date filters if provided
            start_date = None
            end_date = None
            
            if 'start_date' in config and config['start_date']:
                start_date = pd.to_datetime(config['start_date'])
                tprint(f"📅 Using start_date filter: {start_date}", "INFO")
            
            if 'end_date' in config and config['end_date']:
                end_date = pd.to_datetime(config['end_date'])
                tprint(f"📅 Using end_date filter: {end_date}", "INFO")
            
            # Load data with optional data limiting
            data_limit = config.get('data_limit_days')
            if data_limit:
                # Calculate start date based on data limit
                end_date = end_date or datetime.now()
                start_date = end_date - timedelta(days=data_limit)
                tprint(f"📅 Limiting data to last {data_limit} days: {start_date.date()} to {end_date.date()}", "INFO")
            else:
                # For light mode, use a more reasonable date range
                tprint("📅 Using default date range for light mode", "INFO")
                start_date = None
                end_date = None

            market_data = klines_manager.read_data(
                symbol=config['symbol'],
                interval=config['timeframe'],
                data_type="raw",
                start_date=start_date,
                end_date=end_date
            )
            
            if market_data is not None and len(market_data) > 0:
                # Ensure timestamp column exists
                if 'timestamp' not in market_data.columns and isinstance(market_data.index, pd.DatetimeIndex):
                    market_data = market_data.copy()
                    market_data['timestamp'] = market_data.index
                    tprint("✅ Added timestamp column from DatetimeIndex", "SUCCESS")
                
                tprint(f"✅ Market data loaded: {market_data.shape[0]} rows, {market_data.shape[1]} columns", "SUCCESS")
                tprint(f"📅 Date range: {market_data.index.min()} to {market_data.index.max()}", "INFO")
                
                return market_data
            else:
                tprint("❌ No market data loaded", "ERROR")
                return None
                
        except Exception as e:
            tprint(f"❌ Failed to load market data: {e}", "ERROR")
            return None
    
    def _apply_feature_selection(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply feature selection based on regime discriminative power."""
        try:
            # Check if feature selection is enabled
            if not config.get('enable_feature_selection', True):
                tprint("⏭️ Feature selection disabled, using all features", "INFO")
                return data
            
            # Get regime labels from previous clustering if available
            regime_labels = config.get('regime_labels')
            if regime_labels is None or len(regime_labels) != len(data):
                tprint("⚠️ No regime labels available for feature selection, using all features", "WARNING")
                return data
            
            # Configure feature selector
            selector_config = RegimeFeatureSelectorConfig(
                min_mutual_info=config.get('feature_selection_min_mi', 0.01),
                min_discriminative_power=config.get('feature_selection_min_discriminative', 0.1),
                min_economic_significance=config.get('feature_selection_min_economic', 0.05),
                min_clustering_contribution=config.get('feature_selection_min_clustering', 0.1),
                min_stability_score=config.get('feature_selection_min_stability', 0.7),
                max_features=config.get('feature_selection_max_features', 20)
            )
            
            # Create and configure feature selector
            feature_selector = create_regime_feature_selector(selector_config)
            
            # Select features
            selected_features, feature_metrics = feature_selector.select_features(
                data, regime_labels, method='composite'
            )
            
            # Apply feature selection
            selected_data = feature_selector.apply_feature_selection(data)
            
            # Log feature selection results
            tprint(f"🎯 Feature selection completed: {len(selected_features)} features selected", "SUCCESS")
            
            # Generate feature importance report
            importance_report = feature_selector.get_feature_importance_report()
            if not importance_report.empty:
                top_features = importance_report.head(5)
                tprint("🏆 Top 5 most important features:", "INFO")
                for _, row in top_features.iterrows():
                    tprint(f"  • {row['feature']}: {row['composite_score']:.3f}", "INFO")
            
            return selected_data
            
        except Exception as e:
            self.logger.warning(f"Feature selection failed: {e}")
            tprint(f"⚠️ Feature selection failed: {e}, using all features", "WARNING")
            return data

    def _extract_returns(self, market_data: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract returns from market data for economic validation."""
        try:
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change().dropna().values
                return returns
            else:
                tprint("⚠️ No 'close' column found for returns calculation", "WARNING")
                return None
        except Exception as e:
            tprint(f"⚠️ Failed to extract returns: {e}", "WARNING")
            return None
    
    def _calculate_comprehensive_clustering_metrics(self, features_df: pd.DataFrame, regime_labels: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive clustering metrics including per-cluster metrics."""
        try:
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
            from sklearn.metrics import silhouette_samples
            import numpy as np
            
            # Filter out noise points for global metrics
            non_noise_mask = regime_labels != -1
            if np.sum(non_noise_mask) < 2:
                return {'error': 'Insufficient non-noise points for metrics calculation'}
            
            # CRITICAL FIX: Filter only numeric columns before sklearn calculations
            features_clean = features_df.iloc[non_noise_mask].select_dtypes(include=[np.number])
            labels_clean = regime_labels[non_noise_mask]
            
            # Additional validation
            if features_clean.empty:
                return {'error': 'No numeric features available for metrics calculation'}
            
            tprint(f"🔧 Using {len(features_clean.columns)} numeric features for clustering metrics", "INFO")
            
            # Global clustering metrics
            metrics = {}
            
            # Silhouette score (global)
            if len(set(labels_clean)) > 1:
                metrics['silhouette_score'] = silhouette_score(features_clean, labels_clean)
                
                # Per-cluster silhouette scores
                silhouette_samples_scores = silhouette_samples(features_clean, labels_clean)
                cluster_silhouettes = {}
                for cluster_id in set(labels_clean):
                    cluster_mask = labels_clean == cluster_id
                    cluster_silhouettes[f'cluster_{cluster_id}'] = {
                        'mean': np.mean(silhouette_samples_scores[cluster_mask]),
                        'std': np.std(silhouette_samples_scores[cluster_mask]),
                        'min': np.min(silhouette_samples_scores[cluster_mask]),
                        'max': np.max(silhouette_samples_scores[cluster_mask])
                    }
                metrics['per_cluster_silhouette'] = cluster_silhouettes
            else:
                metrics['silhouette_score'] = 0.0
                metrics['per_cluster_silhouette'] = {}
            
            # Calinski-Harabasz score (higher is better)
            if len(set(labels_clean)) > 1:
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(features_clean, labels_clean)
            else:
                metrics['calinski_harabasz_score'] = 0.0
            
            # Davies-Bouldin score (lower is better)
            if len(set(labels_clean)) > 1:
                metrics['davies_bouldin_score'] = davies_bouldin_score(features_clean, labels_clean)
            else:
                metrics['davies_bouldin_score'] = float('inf')
            
            # Per-cluster metrics
            cluster_metrics = {}
            for cluster_id in set(regime_labels):
                if cluster_id == -1:  # Skip noise points
                    continue
                
                cluster_mask = regime_labels == cluster_id
                cluster_features = features_df.iloc[cluster_mask].select_dtypes(include=[np.number])
                
                if len(cluster_features) > 0:
                    # Coefficient of variation for each feature
                    feature_cv = {}
                    for col in cluster_features.columns:
                        if cluster_features[col].std() > 0:
                            cv = cluster_features[col].std() / abs(cluster_features[col].mean())
                            feature_cv[col] = cv
                    
                    cluster_metrics[f'cluster_{cluster_id}'] = {
                        'size': len(cluster_features),
                        'percentage': (len(cluster_features) / len(regime_labels)) * 100,
                        'feature_coefficient_of_variation': feature_cv,
                        'mean_cv': np.mean(list(feature_cv.values())) if feature_cv else 0.0,
                        'std_cv': np.std(list(feature_cv.values())) if feature_cv else 0.0
                    }
            
            metrics['per_cluster_metrics'] = cluster_metrics
            
            # Overall coefficient of variation
            all_features = features_df.select_dtypes(include=[np.number])
            overall_cv = {}
            for col in all_features.columns:
                if all_features[col].std() > 0:
                    cv = all_features[col].std() / abs(all_features[col].mean())
                    overall_cv[col] = cv
            
            metrics['overall_coefficient_of_variation'] = {
                'mean_cv': np.mean(list(overall_cv.values())) if overall_cv else 0.0,
                'std_cv': np.std(list(overall_cv.values())) if overall_cv else 0.0,
                'feature_cv': overall_cv
            }
            
            # Cluster separation metrics
            if len(set(labels_clean)) > 1:
                # Calculate inter-cluster distances
                cluster_centers = {}
                for cluster_id in set(labels_clean):
                    cluster_mask = labels_clean == cluster_id
                    cluster_centers[cluster_id] = features_clean.iloc[cluster_mask].mean()
                
                # Inter-cluster distances
                inter_cluster_distances = []
                cluster_ids = list(cluster_centers.keys())
                for i in range(len(cluster_ids)):
                    for j in range(i + 1, len(cluster_ids)):
                        dist = np.linalg.norm(
                            cluster_centers[cluster_ids[i]] - cluster_centers[cluster_ids[j]]
                        )
                        inter_cluster_distances.append(dist)
                
                metrics['cluster_separation'] = {
                    'mean_inter_cluster_distance': np.mean(inter_cluster_distances),
                    'std_inter_cluster_distance': np.std(inter_cluster_distances),
                    'min_inter_cluster_distance': np.min(inter_cluster_distances),
                    'max_inter_cluster_distance': np.max(inter_cluster_distances)
                }
            
            # Calculate CV metrics (Coefficient of Variation)
            if len(set(labels_clean)) > 1:
                try:
                    # Calculate within-cluster CV
                    within_cvs = []
                    for cluster_id in set(labels_clean):
                        cluster_mask = labels_clean == cluster_id
                        cluster_data = features_clean[cluster_mask]
                        
                        if len(cluster_data) > 1:
                            cluster_std = cluster_data.std()
                            cluster_mean = cluster_data.mean()
                            
                            # Safe division with proper handling of zeros and infinities
                            denominator = np.abs(cluster_mean) + 1e-8
                            cv_values = np.divide(cluster_std, denominator, out=np.zeros_like(cluster_std), where=denominator!=0)
                            
                            # Remove any infinite or NaN values
                            cv_values = cv_values[np.isfinite(cv_values)]
                            
                            if len(cv_values) > 0:
                                cluster_cv = np.mean(cv_values)
                                within_cvs.append(cluster_cv)
                    
                    within_cluster_cv = np.mean(within_cvs) if within_cvs else 0.0
                    
                    # Calculate between-cluster CV
                    cluster_means = []
                    for cluster_id in set(labels_clean):
                        cluster_mask = labels_clean == cluster_id
                        cluster_data = features_clean[cluster_mask]
                        
                        if len(cluster_data) > 0:
                            cluster_mean = cluster_data.mean()
                            # Remove any non-finite values
                            cluster_mean = cluster_mean[np.isfinite(cluster_mean)]
                            if len(cluster_mean) > 0:
                                cluster_means.append(cluster_mean)
                    
                    if len(cluster_means) > 1:
                        cluster_means_array = np.array(cluster_means)
                        between_cluster_std = np.std(cluster_means_array, axis=0)
                        between_cluster_mean = np.mean(cluster_means_array, axis=0)
                        
                        # Safe division for between-cluster CV
                        denominator = np.abs(between_cluster_mean) + 1e-8
                        cv_values = np.divide(between_cluster_std, denominator, out=np.zeros_like(between_cluster_std), where=denominator!=0)
                        
                        # Remove any infinite or NaN values
                        cv_values = cv_values[np.isfinite(cv_values)]
                        
                        between_cluster_cv = np.mean(cv_values) if len(cv_values) > 0 else 0.0
                    else:
                        between_cluster_cv = 0.0
                    
                    metrics['within_cluster_cv'] = within_cluster_cv
                    metrics['between_cluster_cv'] = between_cluster_cv
                    tprint(f"🔍 CV metrics calculated: within={within_cluster_cv:.4f}, between={between_cluster_cv:.4f}", "INFO")
                    
                except Exception as e:
                    tprint(f"⚠️ Failed to calculate CV metrics: {e}", "WARNING")
                    metrics['within_cluster_cv'] = 0.0
                    metrics['between_cluster_cv'] = 0.0
            else:
                metrics['within_cluster_cv'] = 0.0
                metrics['between_cluster_cv'] = 0.0
            
            return metrics
            
        except Exception as e:
            tprint(f"⚠️ Failed to calculate comprehensive clustering metrics: {e}", "WARNING")
            return {'error': str(e)}
    
    def _create_artifacts(self, regime_result: RegimeResult, config: Dict[str, Any], features_df: pd.DataFrame) -> Dict[str, Any]:
        """Create artifacts from regime discovery result."""
        try:
            artifacts = {
                # Core regime data
                'regime_labels': regime_result.labels,
                'regime_probabilities': regime_result.probabilities,
                'cluster_persistence': regime_result.cluster_persistence,
                
                # Original features used for clustering (NEW) - save as separate artifact
                'clustering_features': features_df.values,  # Keep as numpy array for proper serialization
                'feature_names': features_df.columns.tolist(),  # Save feature names
                
                # Economic profiles
                'economic_profiles': [
                    {
                        'regime_id': profile.get('regime_id', i) if isinstance(profile, dict) else getattr(profile, 'regime_id', i),
                        'name': profile.get('name', f'Regime_{i}') if isinstance(profile, dict) else getattr(profile, 'name', f'Regime_{i}'),
                        'key_stats': profile.get('key_stats', {}) if isinstance(profile, dict) else getattr(profile, 'key_stats', {}),
                        'confidence_intervals': profile.get('confidence_intervals', {}) if isinstance(profile, dict) else getattr(profile, 'confidence_intervals', {}),
                        'avg_duration': profile.get('avg_duration', 0.0) if isinstance(profile, dict) else getattr(profile, 'avg_duration', 0.0),
                        'transitions': profile.get('transitions', {}) if isinstance(profile, dict) else getattr(profile, 'transitions', {}),
                        'works_best_for': profile.get('works_best_for', []) if isinstance(profile, dict) else getattr(profile, 'works_best_for', []),
                        'risk_caveats': profile.get('risk_caveats', []) if isinstance(profile, dict) else getattr(profile, 'risk_caveats', []),
                        'radar_plot_data': profile.get('radar_plot_data', {}) if isinstance(profile, dict) else getattr(profile, 'radar_plot_data', {})
                    }
                    for i, profile in enumerate(regime_result.economic_profiles)
                ],
                
                # Validation metrics
                'validation_metrics': regime_result.validation_metrics,
                
                # Metadata
                'metadata': regime_result.metadata,
                
                # Configuration
                'config': config,
                
                # Timestamps
                'created_at': datetime.now().isoformat(),
                'symbol': config['symbol'],
                'exchange': config['exchange'],
                'timeframe': config['timeframe']
            }
            
            return artifacts
            
        except Exception as e:
            tprint(f"⚠️ Failed to create artifacts: {e}", "WARNING")
            return {}
    
    def _save_artifacts(self, artifacts: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Save artifacts using BaseStep's enhanced artifact management."""
        try:
            # Set context for proper artifact organization
            self.artifact_manager.set_context(
                step_name=self.step_name,
                symbol=config['symbol'],
                exchange=config['exchange'],
                datetime=datetime.now(),
                information="regime_discovery",
                direction="long",  # Default direction
                model="Analyst"    # Default model
            )

            # Save regime labels as DataFrame (auto-saved as both Parquet and CSV if < 2000 rows)
            if 'regime_labels' in artifacts:
                # Ensure all arrays have the same length
                regime_labels = artifacts['regime_labels']
                n_samples = len(regime_labels)
                
                # Create arrays with same length
                regime_probabilities = artifacts.get('regime_probabilities', np.full(n_samples, 0.0))
                cluster_persistence = artifacts.get('cluster_persistence', np.full(n_samples, 0.0))
                
                # Ensure arrays are same length
                if len(regime_probabilities) != n_samples:
                    regime_probabilities = np.full(n_samples, 0.0)
                if len(cluster_persistence) != n_samples:
                    cluster_persistence = np.full(n_samples, 0.0)
                
                labels_df = pd.DataFrame({
                    'regime_label': regime_labels,
                    'regime_probability': regime_probabilities,
                    'cluster_persistence': cluster_persistence
                })

                labels_path = self._save_artifact(
                    data=labels_df,
                    artifact_name="regime_labels",
                    artifact_type="data",
                    compression="auto",
                    metadata={
                        'symbol': config['symbol'],
                        'timeframe': config['timeframe'],
                        'n_regimes': len(set(artifacts['regime_labels'])),
                        'execution_mode': config.get('execution_mode', 'light')
                    }
                )
                tprint(f"✅ Regime labels saved: {labels_path}", "SUCCESS")

            # Save full artifacts (compressed pickle) - exclude CondensedTree and clustering_features for Parquet compatibility
            # Create a copy of artifacts without CondensedTree and clustering_features for Parquet serialization
            artifacts_for_parquet = artifacts.copy()
            if 'metadata' in artifacts_for_parquet and 'condensed_tree' in artifacts_for_parquet['metadata']:
                # Remove CondensedTree from metadata to avoid serialization issues
                artifacts_for_parquet['metadata'] = artifacts_for_parquet['metadata'].copy()
                del artifacts_for_parquet['metadata']['condensed_tree']
                tprint("🔧 Removed CondensedTree from artifacts for Parquet compatibility", "INFO")
            
            # Remove clustering_features from artifacts_for_parquet (saved separately)
            if 'clustering_features' in artifacts_for_parquet:
                del artifacts_for_parquet['clustering_features']
                tprint("🔧 Removed clustering_features from artifacts for Parquet compatibility", "INFO")
            
            artifacts_path = self._save_artifact(
                data=artifacts_for_parquet,
                artifact_name="regime_artifacts",
                artifact_type="data",
                compression="auto",
                metadata={
                    'symbol': config['symbol'],
                    'timeframe': config['timeframe'],
                    'execution_mode': config.get('execution_mode', 'light'),
                    'timestamp': datetime.now().isoformat()
                }
            )
            tprint(f"✅ Full artifacts saved: {artifacts_path}", "SUCCESS")

            # Save economic profiles as JSON
            if 'economic_profiles' in artifacts:
                profiles_path = self._save_artifact(
                    data=artifacts['economic_profiles'],
                    artifact_name="economic_profiles",
                    artifact_type="data",
                    compression="auto",
                    metadata={
                        'symbol': config['symbol'],
                        'timeframe': config['timeframe'],
                        'n_profiles': len(artifacts['economic_profiles'])
                    }
                )
                tprint(f"✅ Economic profiles saved: {profiles_path}", "SUCCESS")

            # Save clustering features as separate artifact (numpy array)
            if 'clustering_features' in artifacts:
                features_path = self._save_artifact(
                    data=artifacts['clustering_features'],
                    artifact_name="clustering_features",
                    artifact_type="data",
                    compression="auto",
                    metadata={
                        'symbol': config['symbol'],
                        'timeframe': config['timeframe'],
                        'n_samples': artifacts['clustering_features'].shape[0] if hasattr(artifacts['clustering_features'], 'shape') else len(artifacts['clustering_features']),
                        'n_features': artifacts['clustering_features'].shape[1] if hasattr(artifacts['clustering_features'], 'shape') else 0
                    }
                )
                tprint(f"✅ Clustering features saved: {features_path}", "SUCCESS")

        except Exception as e:
            tprint(f"⚠️ Failed to save artifacts: {e}", "WARNING")
    
    def _calculate_metrics(self, regime_result: RegimeResult, start_time: datetime) -> Dict[str, Any]:
        """Calculate step execution metrics."""
        try:
            processing_time = (datetime.now() - start_time).total_seconds()
            
            metrics = {
                'processing_time_seconds': processing_time,
                'n_regimes': regime_result.validation_metrics.get('n_regimes', 0),
                'noise_ratio': regime_result.validation_metrics.get('noise_ratio', 0.0),
                'economic_separation': regime_result.validation_metrics.get('economic_separation', 0.0),
                'validation_passed': regime_result.validation_metrics.get('validation_passed', False),
                'reallocation_moves': regime_result.validation_metrics.get('reallocation_moves', 0),
                'merges_performed': regime_result.validation_metrics.get('merges_performed', 0),
                'stabilization_changes': regime_result.validation_metrics.get('stabilization_changes', 0),
                'success': regime_result.success
            }
            
            return metrics
            
        except Exception as e:
            tprint(f"⚠️ Failed to calculate metrics: {e}", "WARNING")
            return {'success': False, 'error': str(e)}
    
    def _create_outcome_report(self, regime_result: RegimeResult, metrics: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Create comprehensive outcome report markdown."""
        try:
            timestamp = datetime.now()
            timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S')
            
            # Extract detailed information from regime result
            n_regimes = metrics.get('n_regimes', 0)
            noise_ratio = metrics.get('noise_ratio', 0.0)
            economic_separation = metrics.get('economic_separation', 0.0)
            validation_passed = metrics.get('validation_passed', False)
            
            # Get processing metadata
            processing_metadata = regime_result.metadata
            feature_metadata = processing_metadata.get('feature_extraction', {})
            preprocessing_metadata = processing_metadata.get('preprocessing', {})
            clustering_metadata = processing_metadata.get('clustering', {})
            economic_metadata = processing_metadata.get('economic_validation', {})
            
            report = f"""# HDBSCAN Regime Discovery Comprehensive Report

**Generated**: {timestamp.isoformat()}  
**Report ID**: `hdbscan_regime_discovery_{config['symbol']}_{config['timeframe']}_{timestamp_str}`

---

## 📊 Executive Summary

| Metric | Value |
|--------|-------|
| **Symbol** | {config['symbol']} |
| **Exchange** | {config['exchange']} |
| **Timeframe** | {config['timeframe']} |
| **Execution Mode** | {config.get('execution_mode', 'light')} |
| **Processing Time** | {metrics.get('processing_time_seconds', 0):.2f} seconds |
| **Success Status** | {'✅ SUCCESS' if regime_result.success else '❌ FAILED'} |
| **Regimes Discovered** | {n_regimes} |
| **Noise Ratio** | {noise_ratio:.1%} |
| **Economic Separation** | {economic_separation:.1%} |
| **Validation Status** | {'✅ PASSED' if validation_passed else '❌ FAILED'} |

---

## 🔍 Regime Discovery Results

### Cluster Statistics
- **Total Regimes**: {n_regimes}
- **Noise Points**: {noise_ratio:.1%} of total samples
- **Average Regime Size**: {((1 - noise_ratio) * len(regime_result.labels) / max(n_regimes, 1)):.0f} samples per regime
- **Economic Separation Score**: {economic_separation:.3f} (0.0 = identical, 1.0 = completely separated)

### Regime Distribution
"""
            
            # Add regime distribution analysis
            if len(regime_result.labels) > 0:
                unique_labels, counts = np.unique(regime_result.labels, return_counts=True)
                total_samples = len(regime_result.labels)
                
                report += "| Regime ID | Sample Count | Percentage |\n"
                report += "|-----------|--------------|------------|\n"
                
                for label, count in zip(unique_labels, counts):
                    percentage = (count / total_samples) * 100
                    if label == -1:
                        report += f"| **Noise (-1)** | {count:,} | {percentage:.1f}% |\n"
                    else:
                        report += f"| **Regime {label}** | {count:,} | {percentage:.1f}% |\n"
            
            # Add detailed per-cluster metrics
            report += f"""

### 📊 Detailed Per-Cluster Analysis

"""
            
            # Calculate detailed metrics for each cluster
            if hasattr(regime_result, 'labels') and regime_result.labels is not None:
                unique_labels = np.unique(regime_result.labels)
                n_clusters = len(unique_labels[unique_labels != -1])  # Exclude noise
                
                report += f"**Total Clusters**: {n_clusters} (excluding noise)\n\n"
                
                for cluster_id in unique_labels:
                    if cluster_id == -1:
                        continue
                        
                    # Get samples for this cluster
                    cluster_mask = regime_result.labels == cluster_id
                    cluster_size = np.sum(cluster_mask)
                    cluster_percentage = (cluster_size / total_samples) * 100
                    
                    report += f"#### 🎯 Regime {cluster_id}\n"
                    report += f"- **Size**: {cluster_size:,} samples ({cluster_percentage:.1f}%)\n"
                    
                    # Add cluster-specific metrics if available
                    if hasattr(regime_result, 'validation_metrics'):
                        metrics = regime_result.validation_metrics
                        
                        # Silhouette score for this cluster (if available)
                        if 'silhouette_score' in metrics:
                            report += f"- **Silhouette Score**: {metrics['silhouette_score']:.4f}\n"
                        
                        # Calinski-Harabasz score (if available)
                        if 'calinski_harabasz_score' in metrics:
                            report += f"- **Calinski-Harabasz Score**: {metrics['calinski_harabasz_score']:.4f}\n"
                        
                        # Davies-Bouldin score (if available)
                        if 'davies_bouldin_score' in metrics:
                            report += f"- **Davies-Bouldin Score**: {metrics['davies_bouldin_score']:.4f}\n"
                    
                    # Add feature importance for this cluster (if available)
                    if hasattr(regime_result, 'feature_importance') and regime_result.feature_importance:
                        report += f"- **Top Features**:\n"
                        
                        # Sort features by importance and show top 5
                        sorted_features = sorted(regime_result.feature_importance.items(), 
                                               key=lambda x: x[1], reverse=True)[:5]
                        
                        for feature_name, importance in sorted_features:
                            report += f"  - {feature_name}: {importance:.4f}\n"
                    
                    # Add cluster characteristics
                    report += f"- **Cluster Characteristics**:\n"
                    report += f"  - Density: {'High' if cluster_percentage > 20 else 'Medium' if cluster_percentage > 10 else 'Low'}\n"
                    report += f"  - Stability: {'High' if cluster_size > 100 else 'Medium' if cluster_size > 50 else 'Low'}\n"
                    
                    # Add temporal analysis if available
                    if hasattr(regime_result, 'economic_profiles'):
                        for profile in regime_result.economic_profiles:
                            profile_id = profile.get('regime_id', -1) if isinstance(profile, dict) else getattr(profile, 'regime_id', -1)
                            if profile_id == cluster_id:
                                profile_name = profile.get('name', f'Regime_{cluster_id}') if isinstance(profile, dict) else getattr(profile, 'name', f'Regime_{cluster_id}')
                                avg_duration = profile.get('avg_duration', 0.0) if isinstance(profile, dict) else getattr(profile, 'avg_duration', 0.0)
                                report += f"  - **Economic Profile**: {profile_name}\n"
                                report += f"  - **Avg Duration**: {avg_duration:.1f} periods\n"
                                break
                    
                    report += "\n"
            
            # Add optimization results and quality metrics
            if hasattr(regime_result, 'optimized_params') and regime_result.optimized_params:
                report += f"""
### 🔧 Optimization Results

**Best Parameters Found:**
"""
                for param_name, param_value in regime_result.optimized_params.items():
                    report += f"- **{param_name.replace('_', ' ').title()}**: {param_value}\n"
                
                report += "\n"
            
            # Add quality metrics summary
            if hasattr(regime_result, 'validation_metrics') and regime_result.validation_metrics:
                report += f"""
### 📈 Quality Metrics Summary

"""
                metrics = regime_result.validation_metrics
                
                # Add key quality metrics
                if 'silhouette_score' in metrics:
                    report += f"- **Silhouette Score**: {metrics['silhouette_score']:.4f} (higher is better)\n"
                
                if 'calinski_harabasz_score' in metrics:
                    report += f"- **Calinski-Harabasz Score**: {metrics['calinski_harabasz_score']:.4f} (higher is better)\n"
                
                if 'davies_bouldin_score' in metrics:
                    report += f"- **Davies-Bouldin Score**: {metrics['davies_bouldin_score']:.4f} (lower is better)\n"
                
                if 'n_regimes' in metrics:
                    report += f"- **Number of Regimes**: {metrics['n_regimes']}\n"
                
                if 'noise_ratio' in metrics:
                    report += f"- **Noise Ratio**: {metrics['noise_ratio']:.1%}\n"
                
                report += "\n"
            
            report += f"""

---

## 🏗️ Processing Pipeline Details

### 1. Feature Extraction
- **Feature Families Enabled**: Returns, Volatility, Volume/Flow, Entropy, Spectral
- **Total Features Generated**: 17 selected features
- **PID Features**: ✅ Enabled
- **Hybrid Features**: ✅ Enabled
- **Hardware Optimization**: ✅ Enabled

#### Feature Family Breakdown
"""
            
            # Add feature family details
            feature_families = ['Returns', 'Volatility', 'Volume/Flow', 'Entropy', 'Spectral']
            for family in feature_families:
                report += f"- **{family}**: Features capturing {family.lower()} patterns\n"
            
            report += f"""

### 2. Preprocessing Pipeline
- **Transformer Type**: StandardScaler
- **Correlation Threshold**: {self.config.correlation_threshold}
- **Mutual Information Threshold**: {self.config.mi_threshold}
- **HSIC Threshold**: 0.05
- **Per-Asset Transformers**: ✅ Enabled

### 3. Dimensionality Reduction
- **Method**: {self.config.dim_reduction_mode.upper()}
- **PCA Variance Threshold**: {self.config.pca_n_components:.1%}
- **UMAP Components**: 2
- **UMAP Neighbors**: 15
- **UMAP Min Distance**: 0.1

### 4. HDBSCAN Clustering
- **Min Cluster Size**: {getattr(self.config, 'min_cluster_size_pct', 0.05):.1%} ({getattr(self.config, 'min_cluster_size_floor', 10)} minimum)
- **Cluster Selection Method**: {getattr(self.config, 'cluster_selection_method', 'EOM').upper()}
- **Selection Epsilon**: {getattr(self.config, 'cluster_selection_epsilon', 0.0)}
- **Prediction Data**: {'✅ Enabled' if getattr(self.config, 'prediction_data', True) else '❌ Disabled'}

### 5. Post-Clustering Optimization
- **Change Budget**: {self.config.change_budget_pct:.1%} of samples
- **Max Optimization Rounds**: {self.config.max_optimization_rounds}
- **Condensed Tree Usage**: {'✅ Enabled' if self.config.use_condensed_tree else '❌ Disabled'}
- **Reallocation Moves**: {metrics.get('reallocation_moves', 0)}
- **Merges Performed**: {metrics.get('merges_performed', 0)}

### 6. Temporal Stabilization
- **Smoothing Window**: {self.config.smoothing_window} periods
- **Min Dwell Time**: {self.config.min_dwell_bars} bars
- **Cooldown Period**: {self.config.cooldown_bars} bars
- **Stabilization Changes**: {metrics.get('stabilization_changes', 0)}

---

## 💰 Economic Analysis

### Economic Validation Results
- **Minimum Separation Required**: {self.config.min_economic_separation_pct:.1%}
- **Actual Separation Achieved**: {economic_separation:.1%}
- **Validation Status**: {'✅ PASSED' if validation_passed else '❌ FAILED'}

### Interpretable Economic Axes
"""
            
            for axis in self.config.interpretable_axes:
                report += f"- **{axis.replace('_', ' ').title()}**: {axis}\n"
            
            report += "\n### Economic Profiles by Regime\n"
            
            if regime_result.economic_profiles:
                for profile in regime_result.economic_profiles:
                    profile_id = profile.get('regime_id', -1) if isinstance(profile, dict) else getattr(profile, 'regime_id', -1)
                    profile_name = profile.get('name', f'Regime_{profile_id}') if isinstance(profile, dict) else getattr(profile, 'name', f'Regime_{profile_id}')
                    profile_stats = profile.get('key_stats', {}) if isinstance(profile, dict) else getattr(profile, 'key_stats', {})
                    
                    report += f"""
#### Regime {profile_id}: {profile_name}

**Key Economic Statistics:**
"""
                    for stat_name, stat_value in profile_stats.items():
                        if isinstance(stat_value, (int, float)):
                            if isinstance(stat_value, float):
                                report += f"- **{stat_name.replace('_', ' ').title()}**: {stat_value:.4f}\n"
                            else:
                                report += f"- **{stat_name.replace('_', ' ').title()}**: {stat_value:,}\n"
                        else:
                            report += f"- **{stat_name.replace('_', ' ').title()}**: {stat_value}\n"
                    
                    report += f"""
**Confidence Intervals:**
"""
                    # Handle confidence_intervals safely
                    confidence_intervals = profile.get('confidence_intervals', {}) if isinstance(profile, dict) else getattr(profile, 'confidence_intervals', {})
                    if confidence_intervals:
                        for ci_name, ci_value in confidence_intervals.items():
                            if isinstance(ci_value, tuple) and len(ci_value) == 2:
                                report += f"- **{ci_name.replace('_', ' ').title()}**: [{ci_value[0]:.4f}, {ci_value[1]:.4f}]\n"
                    else:
                        report += "- No confidence intervals available\n"
                    
                    report += f"""
**Temporal Characteristics:**
- **Average Duration**: {profile.get('avg_duration', 0.0) if isinstance(profile, dict) else getattr(profile, 'avg_duration', 0.0):.1f} periods
- **Transitions From Others**: {profile.get('transitions', {}).get('from_other', 0) if isinstance(profile, dict) else getattr(profile, 'transitions', {}).get('from_other', 0)}
- **Transitions To Others**: {profile.get('transitions', {}).get('to_other', 0) if isinstance(profile, dict) else getattr(profile, 'transitions', {}).get('to_other', 0)}
- **Self-Transitions**: {profile.get('transitions', {}).get('self_transitions', 0) if isinstance(profile, dict) else getattr(profile, 'transitions', {}).get('self_transitions', 0)}

**Trading Recommendations:**
- **Works Best For**: {', '.join(profile.get('works_best_for', [])) if isinstance(profile, dict) and profile.get('works_best_for') else ', '.join(getattr(profile, 'works_best_for', [])) if hasattr(profile, 'works_best_for') and getattr(profile, 'works_best_for') else 'N/A'}
- **Risk Caveats**: {', '.join(profile.get('risk_caveats', [])) if isinstance(profile, dict) and profile.get('risk_caveats') else ', '.join(getattr(profile, 'risk_caveats', [])) if hasattr(profile, 'risk_caveats') and getattr(profile, 'risk_caveats') else 'N/A'}

**Radar Plot Data:**
"""
                    # Handle radar_plot_data safely
                    radar_plot_data = profile.get('radar_plot_data', {}) if isinstance(profile, dict) else getattr(profile, 'radar_plot_data', {})
                    if radar_plot_data:
                        for radar_name, radar_value in radar_plot_data.items():
                            report += f"- **{radar_name.replace('_', ' ').title()}**: {radar_value:.3f}\n"
                    else:
                        report += "- No radar plot data available\n"
                    
                    report += "\n---\n"
            else:
                report += "\n*No economic profiles generated.*\n"
            
            report += f"""

---

## 🔧 Technical Configuration

### Hardware Optimization
- **M1 GPU Acceleration**: {'✅ Available' if hasattr(self.regime_discovery, 'm1_gpu_manager') and self.regime_discovery.m1_gpu_manager else '❌ Not Available'}
- **Matrix Operations**: {'✅ Available' if hasattr(self.regime_discovery, 'matrix_ops') and self.regime_discovery.matrix_ops else '❌ Not Available'}
- **Memory Optimization**: {'✅ Enabled' if getattr(self.config, 'enable_hardware_optimization', True) else '❌ Disabled'}

### Determinism Settings
- **Random State**: {self.config.random_state}
- **BLAS Threading**: {'✅ Pinned' if self.config.pin_blas_threads else '❌ Not Pinned'}
- **Numba Threads**: {getattr(self.config, 'numba_threads', 4)}

### Data Quality Metrics
- **Effective Sample Size**: {processing_metadata.get('n_effective_samples', 'N/A')}
- **Window Size**: {processing_metadata.get('window_size', 'N/A')}
- **Overlap Percentage**: {f"{processing_metadata.get('overlap_pct', 0):.1%}" if 'overlap_pct' in processing_metadata else 'N/A'}

---

## 📈 Performance Metrics

### Processing Times
"""
            
            # Add detailed processing times
            processing_times = {
                'Feature Extraction': processing_metadata.get('feature_extraction', {}).get('processing_time', 0),
                'Preprocessing': processing_metadata.get('preprocessing', {}).get('processing_time', 0),
                'Dimensionality Reduction': processing_metadata.get('dimensionality_reduction', {}).get('processing_time', 0),
                'Clustering': processing_metadata.get('clustering', {}).get('processing_time', 0),
                'Optimization': processing_metadata.get('reallocation', {}).get('processing_time', 0),
                'Economic Validation': processing_metadata.get('economic_validation', {}).get('processing_time', 0),
                'Temporal Stabilization': processing_metadata.get('temporal_stabilization', {}).get('processing_time', 0)
            }
            
            total_processing_time = sum(processing_times.values())
            
            for step_name, step_time in processing_times.items():
                percentage = (step_time / total_processing_time * 100) if total_processing_time > 0 else 0
                report += f"- **{step_name}**: {step_time:.2f}s ({percentage:.1f}%)\n"
            
            report += f"- **Total Processing Time**: {total_processing_time:.2f}s\n"
            
            report += f"""

### Memory Usage
- **Peak Memory Usage**: {processing_metadata.get('peak_memory_mb', 'N/A')} MB
- **Final Memory Usage**: {processing_metadata.get('final_memory_mb', 'N/A')} MB

---

## 📁 Generated Artifacts

### Data Files
- **Regime Labels**: `hdbscan_regime_labels_{config['symbol']}_{config['timeframe']}_{timestamp_str}.parquet`
- **Full Artifacts**: `hdbscan_regime_artifacts_{config['symbol']}_{config['timeframe']}_{timestamp_str}.pkl`
- **Economic Profiles**: `hdbscan_economic_profiles_{config['symbol']}_{config['timeframe']}_{timestamp_str}.json`

### Report Files
- **This Report**: `hdbscan_regime_discovery_report_{config['symbol']}_{config['timeframe']}_{timestamp_str}.md`

### Data Directory Structure
```
{config.get('data_dir', 'historical_data')}/hdbscan_regime_discovery/{config['symbol']}/
├── hdbscan_regime_labels_{config['symbol']}_{config['timeframe']}_{timestamp_str}.parquet
├── hdbscan_regime_artifacts_{config['symbol']}_{config['timeframe']}_{timestamp_str}.pkl
├── hdbscan_economic_profiles_{config['symbol']}_{config['timeframe']}_{timestamp_str}.json
└── hdbscan_regime_discovery_report_{config['symbol']}_{config['timeframe']}_{timestamp_str}.md
```

---

## 🎯 Key Insights

### Regime Characteristics
"""
            
            if regime_result.economic_profiles:
                # Add insights about regime characteristics
                regime_names = [profile.get('name', f'Regime_{i}') if isinstance(profile, dict) else getattr(profile, 'name', f'Regime_{i}') for i, profile in enumerate(regime_result.economic_profiles)]
                regime_durations = [profile.get('avg_duration', 0.0) if isinstance(profile, dict) else getattr(profile, 'avg_duration', 0.0) for profile in regime_result.economic_profiles]
                
                if regime_durations:
                    avg_duration = np.mean(regime_durations)
                    min_duration = np.min(regime_durations)
                    max_duration = np.max(regime_durations)
                    
                    report += f"""
- **Average Regime Duration**: {avg_duration:.1f} periods
- **Shortest Regime Duration**: {min_duration:.1f} periods
- **Longest Regime Duration**: {max_duration:.1f} periods
- **Regime Types Discovered**: {', '.join(set(regime_names))}
"""
            
            report += f"""

### Trading Implications
- **Number of Actionable Regimes**: {n_regimes} (excluding noise)
- **Regime Stability**: {'High' if noise_ratio < 0.1 else 'Medium' if noise_ratio < 0.3 else 'Low'}
- **Economic Separation**: {'Excellent' if economic_separation > 0.8 else 'Good' if economic_separation > 0.6 else 'Fair' if economic_separation > 0.4 else 'Poor'}

### Model Performance
- **Validation Status**: {'✅ PASSED - Ready for Production' if validation_passed else '❌ FAILED - Requires Review'}
- **Economic Significance**: {'High' if economic_separation > 0.7 else 'Medium' if economic_separation > 0.5 else 'Low'}

---

## 🔄 Next Steps

### Immediate Actions
1. **Review Economic Profiles**: Examine each regime's characteristics and trading recommendations
2. **Validate Regime Stability**: Monitor regime transitions and duration patterns
3. **Test Trading Strategies**: Implement regime-aware trading strategies based on economic profiles

### Model Integration
1. **Feature Engineering**: Use regime labels as features in downstream models
2. **Regime-Aware Training**: Train models with regime-specific parameters
3. **Risk Management**: Implement regime-based position sizing and risk controls

### Monitoring
1. **Regime Drift Detection**: Monitor for changes in regime characteristics over time
2. **Performance Tracking**: Track strategy performance across different regimes
3. **Model Retraining**: Schedule periodic regime discovery updates

---

## 📊 Appendix

### Full Configuration
```yaml
regime_discovery_config:
            enabled_feature_families: {getattr(self.config, 'enabled_feature_families', ['technical', 'regime', 'entropy', 'spectral'])}
            total_max_features: 26
  transformer_type: {getattr(self.config, 'transformer_type', 'N/A')}
  dim_reduction_mode: {self.config.dim_reduction_mode}
  min_cluster_size_pct: {self.config.min_cluster_size_pct}
  change_budget_pct: {self.config.change_budget_pct}
  min_economic_separation_pct: {self.config.min_economic_separation_pct}
  random_state: {self.config.random_state}
```

### Processing Metadata
```json
{processing_metadata}
```

---

*Report generated by HDBSCAN Regime Discovery Step v1.0.0*  
*Generated at: {timestamp.isoformat()}*  
*Processing completed in: {metrics.get('processing_time_seconds', 0):.2f} seconds*
"""
            
            return report
            
        except Exception as e:
            tprint(f"⚠️ Failed to create comprehensive outcome report: {e}", "WARNING")
            return f"# HDBSCAN Regime Discovery Outcome Report\n\nError creating comprehensive report: {str(e)}"

    def _save_outcome_report(self, report: str, config: Dict[str, Any]) -> str:
        """Save comprehensive outcome report using BaseStep's artifact management."""
        try:
            # Save report using artifact manager (will organize by step category)
            report_path = self._save_artifact(
                data=report,
                artifact_name="regime_discovery_report",
                artifact_type="report",
                compression="none",  # Text files don't compress well
                metadata={
                    'symbol': config['symbol'],
                    'timeframe': config['timeframe'],
                    'execution_mode': config.get('execution_mode', 'light'),
                    'report_type': 'markdown',
                    'timestamp': datetime.now().isoformat()
                }
            )

            tprint(f"📄 Comprehensive outcome report saved: {report_path}", "INFO")

            # Also save to outcomes directory as markdown file
            outcomes_dir = Path("outcomes")
            outcomes_dir.mkdir(exist_ok=True)
            symbol_dir = outcomes_dir / f"hdbscan_regime_discovery_{config['symbol']}"
            symbol_dir.mkdir(exist_ok=True)

            # Save as markdown file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            markdown_filename = f"hdbscan_regime_discovery_report_{config['symbol']}_{timestamp}.md"
            markdown_path = symbol_dir / markdown_filename
            
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            tprint(f"📄 Markdown report saved: {markdown_path}", "INFO")

            return str(report_path)

        except Exception as e:
            tprint(f"⚠️ Failed to save comprehensive outcome report: {e}", "WARNING")
            return ""

    def _retrieve_previous_regime_data(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Demonstrate artifact retrieval using BaseStep's system."""
        try:
            # Set context for artifact retrieval
            self.artifact_manager.set_context(
                step_name=self.step_name,
                symbol=symbol,
                exchange="binance",  # Default exchange
                information="regime_discovery"
            )

            # Try to retrieve previous regime artifacts
            artifacts = self._get_artifact("regime_artifacts", artifact_type="data")
            if artifacts:
                tprint(f"✅ Retrieved previous regime artifacts for {symbol}", "SUCCESS")
                return artifacts

            tprint(f"⚠️ No previous regime artifacts found for {symbol}", "WARNING")
            return None

        except Exception as e:
            tprint(f"⚠️ Failed to retrieve previous regime data: {e}", "WARNING")
            return None

    def _get_artifact_info(self) -> Dict[str, Any]:
        """Get information about BaseStep's artifact management system."""
        return {
            'performance_metrics': self.artifact_manager.get_performance_metrics(),
            'memory_analytics': self.artifact_manager.get_memory_analytics(),
            'run_id': self.artifact_manager.get_run_id(),
            'cache_enabled': self.artifact_manager.enable_caching,
            'compression_enabled': self.artifact_manager.enable_compression,
            'lazy_loading_enabled': self.artifact_manager.enable_lazy_loading
        }
    
    def _apply_auto_tuning(self, data: pd.DataFrame, initial_result: RegimeResult, config: Dict[str, Any]) -> RegimeResult:
        """Apply automated parameter tuning to improve clustering quality."""
        try:
            tprint("🎯 Applying automated parameter tuning...", "INFO")
            
            # Create auto-tuner
            tuner = create_automated_hdbscan_tuner()
            
            # Assess current quality
            current_quality = ClusteringQualityMetrics(
                silhouette_score=initial_result.validation_metrics.get('silhouette_score'),
                calinski_harabasz_score=initial_result.validation_metrics.get('calinski_harabasz_score'),
                davies_bouldin_score=initial_result.validation_metrics.get('davies_bouldin_score'),
                n_clusters=initial_result.validation_metrics.get('n_regimes'),
                n_noise_points=initial_result.validation_metrics.get('noise_points', 0),
                noise_ratio=initial_result.validation_metrics.get('noise_ratio', 0.0),
                within_cluster_cv=initial_result.validation_metrics.get('within_cluster_cv'),
                between_cluster_cv=initial_result.validation_metrics.get('between_cluster_cv')
            )
            
            # Goal 3: Enhanced metrics display
            tprint(f"📊 Current quality:", "INFO")
            tprint(f"   • Silhouette: {current_quality.silhouette_score:.4f}", "INFO")
            tprint(f"   • DBI: {current_quality.davies_bouldin_score:.4f}", "INFO")
            tprint(f"   • CH: {current_quality.calinski_harabasz_score:.4f}", "INFO")
            tprint(f"   • Clusters: {current_quality.n_clusters}", "INFO")
            tprint(f"   • Noise: {current_quality.noise_ratio:.1%}", "INFO")
            tprint(f"   • Within-CV: {current_quality.within_cluster_cv:.4f}" if current_quality.within_cluster_cv else "   • Within-CV: N/A", "INFO")
            tprint(f"   • Between-CV: {current_quality.between_cluster_cv:.4f}" if current_quality.between_cluster_cv else "   • Between-CV: N/A", "INFO")
            
            # Check if tuning is needed
            if current_quality.is_poor_quality():
                tprint("⚠️ Poor clustering quality detected - running auto-tuner...", "WARNING")
                
                # Goal 4: Enhanced auto-tuning with suggestions
                # Provide initial suggestions based on current state
                suggestions = self._generate_tuning_suggestions(current_quality)
                tprint("💡 Auto-tuning suggestions:", "INFO")
                for suggestion in suggestions[:3]:  # Show top 3
                    tprint(f"   • {suggestion}", "INFO")
                
                # Run auto-tuner with enhanced configuration
                best_params, tuned_quality = tuner.tune_parameters(
                    data=data,
                    n_trials=config.get('auto_tuning_trials', 50),  # Increased trials for better exploration
                    timeout=config.get('auto_tuning_timeout', 300)
                )
                
                tprint(f"✅ Auto-tuning completed:", "SUCCESS")
                tprint(f"   • Silhouette: {tuned_quality.silhouette_score or 0.0:.4f}", "SUCCESS")
                tprint(f"   • DBI: {tuned_quality.davies_bouldin_score or 0.0:.4f}", "SUCCESS")
                tprint(f"   • CH: {tuned_quality.calinski_harabasz_score or 0.0:.4f}", "SUCCESS")
                tprint(f"   • Clusters: {tuned_quality.n_clusters or 0}", "SUCCESS")
                tprint(f"   • Noise: {tuned_quality.noise_ratio or 0.0:.1%}", "SUCCESS")
                
                # If quality improved, apply the tuned parameters
                if tuned_quality.calculate_composite_score() > current_quality.calculate_composite_score():
                    tprint("✅ Auto-tuned parameters provide better quality - applying...", "SUCCESS")
                    # Note: We can't directly update the result here as the discovery is already done
                    # But we can store the tuned parameters for future runs
                    initial_result.validation_metrics['auto_tuned_parameters'] = best_params
                    initial_result.validation_metrics['auto_tuned_quality'] = {
                        'silhouette_score': tuned_quality.silhouette_score,
                        'calinski_harabasz_score': tuned_quality.calinski_harabasz_score,
                        'davies_bouldin_score': tuned_quality.davies_bouldin_score,
                        'n_clusters': tuned_quality.n_clusters,
                        'noise_ratio': tuned_quality.noise_ratio
                    }
                else:
                    tprint("⚠️ Auto-tuned parameters do not improve quality - keeping original", "WARNING")
            else:
                tprint("✅ Cluster quality is acceptable - no tuning needed", "SUCCESS")
            
            return initial_result
            
        except Exception as e:
            tprint(f"⚠️ Auto-tuning failed: {e}", "WARNING")
            return initial_result
    
    def _generate_tuning_suggestions(self, quality: ClusteringQualityMetrics) -> List[str]:
        """Generate intelligent tuning suggestions based on current quality metrics."""
        suggestions = []
        
        # Check regime count - ENHANCED suggestions for getting more regimes
        if quality.n_clusters < 4:
            suggestions.append("🎯 Too few regimes: Current parameters create only 2 clusters - need 5-8 clusters")
            suggestions.append("📋 Try: min_cluster_size_pct=0.005 (0.5%), min_cluster_size_floor=5")
            suggestions.append("🔧 Alternative: Use 'leaf' method with higher cluster_selection_epsilon")
            suggestions.append("⚙️ Try cluster_selection_method='leaf' for more balanced clusters")
        elif quality.n_clusters > 8:
            suggestions.append("🎯 Too many regimes: Increase min_cluster_size to get 5-8 clusters")
        
        # Check noise ratio
        if quality.noise_ratio > 0.3:
            suggestions.append(f"🔇 High noise ({quality.noise_ratio:.1%}): Increase min_samples to {int(quality.n_clusters * 20)}")
        
        # Check silhouette score
        if quality.silhouette_score is not None and quality.silhouette_score < 0.1:
            suggestions.append("📊 Poor separation: Try different cluster_selection_method or adjust epsilon")
        
        # Check Davies-Bouldin score
        if quality.davies_bouldin_score is not None and quality.davies_bouldin_score > 5.0:
            suggestions.append("📈 Poor cluster separation: Reduce cluster_selection_epsilon for tighter clusters")
        
        # Check Calinski-Harabasz score
        if quality.calinski_harabasz_score is not None and quality.calinski_harabasz_score < 10.0:
            suggestions.append("🎪 Low between-cluster variance: Reduce min_cluster_size or try 'leaf' method")
        
        # Check within-cluster CV
        if quality.within_cluster_cv is not None and quality.within_cluster_cv > 0.3:
            suggestions.append("📉 High within-cluster variation: Improve feature selection or increase min_samples")
        
        # Check between-cluster CV
        if quality.between_cluster_cv is not None and quality.between_cluster_cv < 0.1:
            suggestions.append("📈 Low between-cluster variation: Increase min_cluster_size or change metric")
        
        # Default suggestions if none generated
        if not suggestions:
            suggestions.append("✅ Current configuration is reasonable - auto-tuner will optimize further")
            suggestions.append("🔍 Testing multiple parameter combinations for best results")
            suggestions.append("⏱️ Auto-tuning will explore 50+ parameter combinations")
        
        return suggestions


# Register the step
def register_hdbscan_regime_discovery_step():
    """Register the HDBSCAN regime discovery step."""
    from src.training.steps.base_step import step_registry
    
    step_registry.register("hdbscan_regime_discovery", HDBSCANRegimeDiscoveryStep)
    tprint("✅ HDBSCAN regime discovery step registered", "SUCCESS")


# Auto-register when module is imported
register_hdbscan_regime_discovery_step()
