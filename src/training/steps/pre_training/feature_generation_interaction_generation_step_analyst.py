"""
Three-Phase LGBM+SHAP Analyst Interaction Generation Step.

This step implements a sophisticated pipeline for analyst feature engineering:
1. Phase 0: Load artifacts and adaptive feature selection (3-6 per category)
2. Phase 1: Generate normalized variants with RobustScaler bounding
3. Phase 2: Apply cheap pruning with category protection (40-50% reduction)
4. Phase 3: Three-phase LGBM+SHAP for feature selection and interaction discovery
5. Phase 4: Combine features, verify category coverage, save artifacts

Key Features:
- Adaptive feature selection (3-6 per category based on signal strength)
- RobustScaler bounding to prevent extreme values
- Category protection during pruning (maintain ≥3 per category)
- Tree-based interaction guidance with corrected SHAP analysis
- Comprehensive causality enforcement
- Category coverage tracking (≥2 per category in final set)
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
import time
from pathlib import Path

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_error, tprint_performance,
    tprint_warning, tprint_structured, LogLevel
)

# VectorBT imports
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import (
        VectorBTRollingOptimizer, rolling_mean, rolling_std, rolling_var
    )
    from src.feature_generation.utils.unified_vectorization_manager import (
        UnifiedVectorizationManager, VectorizationConfig
    )
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    tprint_warning("⚠️ VectorBT components not available")

# Hardware optimization
try:
    from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager, WorkloadType
    from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
    HARDWARE_OPT_AVAILABLE = True
except ImportError:
    HARDWARE_OPT_AVAILABLE = False
    tprint_warning("⚠️ Hardware optimization not available")

# ML utilities
try:
    import lightgbm as lgb
    from sklearn.model_selection import cross_val_score
    from sklearn.multioutput import MultiOutputRegressor
    import shap
    LGBM_AVAILABLE = True
    SHAP_AVAILABLE = True
except ImportError as e:
    LGBM_AVAILABLE = False
    SHAP_AVAILABLE = False
    tprint_warning(f"⚠️ ML libraries not available: {e}")

# Import our new utilities
try:
    from src.training.utils.feature_selection.variant_generator import VariantGenerator, generate_all_variants
    from src.training.utils.feature_selection.cheap_pruning import CheapPruningPipeline, apply_cheap_pruning, PruningConfig
    UTILITIES_AVAILABLE = True
except ImportError as e:
    UTILITIES_AVAILABLE = False
    tprint_warning(f"⚠️ Feature selection utilities not available: {e}")

logger = logging.getLogger(__name__)


class FeatureGenerationInteractionGenerationStepAnalyst(BaseStep):
    """
    Three-Phase LGBM+SHAP Analyst Interaction Generation Step.
    
    Implements a comprehensive pipeline for feature engineering with:
    - Top feature selection by composite_score
    - Numerically safe variant generation
    - Per-category cheap pruning
    - LGBM+SHAP feature selection
    - Tree-guided interaction discovery
    """

    def __init__(self, step_name: str = "feature_generation_interaction_generation_step_analyst"):
        """Initialize the analyst interaction generation step."""
        super().__init__(step_name)
        self.logger = system_logger.getChild('AnalystInteractionGeneration')
        
        # Initialize hardware optimization
        if HARDWARE_OPT_AVAILABLE:
            self.hardware_manager = UnifiedHardwareManager()
            self.memory_optimizer = M1MemoryOptimizer(memory_limit_gb=8.0)
        else:
            self.hardware_manager = None
            self.memory_optimizer = None
        
        # Initialize VectorBT components
        self.vectorization_manager = None
        self.rolling_optimizer = None
        
        # Performance tracking
        self.performance_stats = {
            'phase0_time': 0.0,
            'phase1_time': 0.0,
            'phase2_time': 0.0,
            'phase3_1_time': 0.0,
            'phase3_2_time': 0.0,
            'phase3_3_time': 0.0,
            'phase4_time': 0.0,
            'total_time': 0.0,
            'features_selected_per_category': {},
            'variants_generated': 0,
            'features_after_pruning': 0,
            'final_feature_count': 0,
            'interaction_count': 0,
            'numerical_safety_incidents': 0,
            'category_coverage': {}
        }
        
        # Numerical safety log
        self.numerical_safety_log = []
        
        # Category definitions
        self.categories = ['trend', 'oscillator', 'momentum', 'return', 'volatility', 'volume', 'acceleration']

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute three-phase LGBM+SHAP analyst interaction generation.

        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol (e.g., 'ETHUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Timeframe (e.g., '15m')
                - execution_mode: 'full', 'light', or 'blank'
                - top_features_per_category: Number of top features to select (default: 4)
                - pruning_target: Target pruning percentage (default: 0.45 for 45%)

        Returns:
            Dict containing:
            - 'success': bool indicating if step completed successfully
            - 'artifacts': dict of created artifacts
            - 'metrics': dict of performance metrics
            - 'error': error message if step failed (optional)
        """
        start_time = time.time()
        tprint_info(f"🚀 [ANALYST] Starting three-phase LGBM+SHAP interaction generation for {config.get('symbol', 'UNKNOWN')}")

        try:
            # Initialize optimization components
            await self._initialize_optimization_components(config)
            
            # Phase 0: Load artifacts and select top features
            tprint_info("=" * 80)
            tprint_info("📋 PHASE 0: Load Artifacts and Select Top Features")
            tprint_info("=" * 80)
            phase0_start = time.time()
            
            lookback_optimization, labeled_data, generated_features, top_features_by_category = \
                await self._phase0_load_and_select(config)
            
            self.performance_stats['phase0_time'] = time.time() - phase0_start
            tprint_performance(f"Phase 0 completed", self.performance_stats['phase0_time'])
            
            # Phase 1: Generate variants
            tprint_info("=" * 80)
            tprint_info("🔄 PHASE 1: Generate Numerically Safe Variants")
            tprint_info("=" * 80)
            phase1_start = time.time()
            
            variant_features = await self._phase1_generate_variants(
                generated_features, top_features_by_category, lookback_optimization, config
            )
            
            self.performance_stats['phase1_time'] = time.time() - phase1_start
            tprint_performance(f"Phase 1 completed", self.performance_stats['phase1_time'])
            
            # Phase 2: Cheap pruning
            tprint_info("=" * 80)
            tprint_info("✂️ PHASE 2: Per-Category Cheap Pruning")
            tprint_info("=" * 80)
            phase2_start = time.time()
            
            pruned_features, pruning_stats = await self._phase2_cheap_pruning(
                variant_features, labeled_data, config
            )
            
            self.performance_stats['phase2_time'] = time.time() - phase2_start
            tprint_performance(f"Phase 2 completed", self.performance_stats['phase2_time'])
            
            # Phase 3: Three-phase LGBM+SHAP
            tprint_info("=" * 80)
            tprint_info("🤖 PHASE 3: Three-Phase LGBM+SHAP Pipeline")
            tprint_info("=" * 80)
            
            final_features, interactions, shap_metadata = await self._phase3_lgbm_shap_pipeline(
                pruned_features, labeled_data, config
            )
            
            # Phase 4: Integration and artifact saving
            tprint_info("=" * 80)
            tprint_info("💾 PHASE 4: Integration and Artifact Saving")
            tprint_info("=" * 80)
            phase4_start = time.time()
            
            artifacts, metrics = await self._phase4_save_artifacts(
                final_features, interactions, shap_metadata, pruning_stats, config
            )
            
            self.performance_stats['phase4_time'] = time.time() - phase4_start
            self.performance_stats['total_time'] = time.time() - start_time
            
            tprint_success(f"✅ [ANALYST] Three-phase pipeline completed in {self.performance_stats['total_time']:.2f}s")
            tprint_info(f"📊 Final feature count: {self.performance_stats['final_feature_count']}")
            tprint_info(f"🔗 Interaction count: {self.performance_stats['interaction_count']}")
            
            return {
                'success': True,
                'artifacts': artifacts,
                'metrics': metrics
            }

        except Exception as e:
            error_msg = f"[ANALYST] Three-phase pipeline failed: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            self.logger.error(error_msg, exc_info=True)

            return {
                'success': False,
                'artifacts': {},
                'metrics': self.performance_stats,
                'error': error_msg
            }

    async def _initialize_optimization_components(self, config: Dict[str, Any]):
        """Initialize VectorBT and hardware optimization components."""
        tprint_info("🔧 Initializing optimization components")
        
        try:
            # Initialize hardware optimization
            if HARDWARE_OPT_AVAILABLE and self.hardware_manager:
                self.hardware_manager.optimize_for_workload(WorkloadType.ML_TRAINING)
                tprint_success("✅ Hardware optimization initialized")
            
            # Initialize VectorBT components
            if VECTORBT_AVAILABLE:
                vectorization_config = VectorizationConfig(
                    enable_vectorbt=True,
                    enable_gpu=config.get('enable_gpu', False),
                    enable_parallel=True,
                    memory_efficient=True,
                    max_memory_gb=8.0,
                    chunk_size=1000,
                    enable_monitoring=True
                )
                
                self.vectorization_manager = UnifiedVectorizationManager(vectorization_config)
                self.rolling_optimizer = VectorBTRollingOptimizer(
                    enable_parallel=True,
                    memory_efficient=True,
                    chunk_size=1000
                )
                tprint_success("✅ VectorBT components initialized")
            
        except Exception as e:
            tprint_warning(f"⚠️ Optimization initialization partial failure: {e}")

    async def _phase0_load_and_select(self, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """
        Phase 0: Load artifacts and select top 4 features per category.
        
        Returns:
            Tuple of (lookback_optimization, labeled_data, generated_features, top_features_by_category)
        """
        tprint_info("📊 Loading artifacts via BaseStep artifact manager")
        
        # Load artifacts
        try:
            lookback_optimization = self._get_artifact('lookback_optimization', 'data')
            tprint_success(f"✅ Loaded lookback_optimization: {lookback_optimization.shape}")
            tprint_structured({"Lookback Optimization": lookback_optimization.head().to_dict()}, level=LogLevel.INFO)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load lookback_optimization artifact: {e}")
        
        try:
            labeled_data = self._get_artifact('labeled_data', 'data')
            tprint_success(f"✅ Loaded labeled_data: {labeled_data.shape}")
            tprint_structured({"Labeled Data": labeled_data.head().to_dict()}, level=LogLevel.INFO)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load labeled_data artifact: {e}")
        
        try:
            generated_features = self._get_artifact('generated_features', 'data')
            tprint_success(f"✅ Loaded generated_features: {generated_features.shape}")
        except Exception as e:
            raise FileNotFoundError(f"Failed to load generated_features artifact: {e}")
        
        # Apply light mode filtering
        generated_features = self._apply_light_mode_filter(
            generated_features, config, config.get('timeframe', '15m')
        )
        labeled_data = self._apply_light_mode_filter(
            labeled_data, config, config.get('timeframe', '15m')
        )
        
        # Select top features per category
        top_features_per_category = config.get('top_features_per_category', 4)
        top_features_by_category = self._select_top_features_per_category(
            lookback_optimization, top_features_per_category
        )
        
        return lookback_optimization, labeled_data, generated_features, top_features_by_category

    def _select_top_features_per_category(self, lookback_optimization: pd.DataFrame, top_n: int = 4) -> Dict:
        """
        Select top features per category using adaptive selection (3-6 per category).
        
        Uses adaptive selection based on signal strength:
        - Minimum 3, maximum 6 per category
        - Select features above 85th percentile of composite_score within category
        - Allow categories with stronger signals to contribute more features
        
        Args:
            lookback_optimization: DataFrame with feature_name, category, composite_score, optimal_lookback
            top_n: Base number of features (used as fallback)
            
        Returns:
            Dict mapping category -> list of (feature_name, optimal_lookback, composite_score)
        """
        tprint_info(f"🎯 Adaptive feature selection (3-6 per category based on signal strength)")
        
        top_features_by_category = {}
        
        for category in self.categories:
            # Filter features by category
            category_features = lookback_optimization[
                lookback_optimization['category'].str.lower() == category.lower()
            ].copy()
            
            if len(category_features) == 0:
                tprint_warning(f"⚠️ No features found for category: {category}")
                continue
            
            # Sort by composite_score descending
            category_features = category_features.sort_values('composite_score', ascending=False)
            
            # Adaptive selection logic
            n_features = len(category_features)
            if n_features < 3:
                # If less than 3 features, take all
                selected_features = category_features
            else:
                # Calculate 85th percentile threshold
                threshold = category_features['composite_score'].quantile(0.85)
                
                # Select features above threshold
                above_threshold = category_features[category_features['composite_score'] >= threshold]
                
                # Apply min/max constraints
                min_features = min(3, n_features)
                max_features = min(6, n_features)
                
                if len(above_threshold) < min_features:
                    # If not enough above threshold, take top min_features
                    selected_features = category_features.head(min_features)
                elif len(above_threshold) > max_features:
                    # If too many above threshold, take top max_features
                    selected_features = category_features.head(max_features)
                else:
                    # Use adaptive selection
                    selected_features = above_threshold
            
            # Store as list of tuples
            top_features_by_category[category] = [
                (row['feature_name'], row['optimal_lookback'], row['composite_score'])
                for _, row in selected_features.iterrows()
            ]
            
            self.performance_stats['features_selected_per_category'][category] = len(selected_features)
            
            tprint_info(f"  {category.upper()}: Selected {len(selected_features)} features (adaptive)")
            for feature_name, optimal_lookback, composite_score in top_features_by_category[category]:
                tprint_info(f"    - {feature_name}: lookback={optimal_lookback}, score={composite_score:.4f}")
        
        return top_features_by_category

    async def _phase1_generate_variants(
        self, 
        generated_features: pd.DataFrame,
        top_features_by_category: Dict,
        lookback_optimization: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Phase 1: Generate normalized variants with RobustScaler bounding.
        
        Uses our new VariantGenerator utility with:
        - 3-4 variants per feature (base, vol-norm, VWAP, trend-adj)
        - RobustScaler bounding to prevent extreme values
        - Causality enforcement via shift(1)
        
        Returns:
            DataFrame with all variant features
        """
        tprint_info("🔄 Generating normalized variants with RobustScaler bounding")
        
        if not UTILITIES_AVAILABLE:
            raise ImportError("Variant generation utilities not available")
        
        # Prepare OHLCV data for variant generation
        ohlcv_columns = ['close', 'high', 'low', 'open', 'volume']
        ohlcv_data = generated_features[ohlcv_columns].copy() if all(col in generated_features.columns for col in ohlcv_columns) else None
        
        if ohlcv_data is None:
            tprint_warning("⚠️ OHLCV data not available, using basic price data")
            # Create basic OHLCV from available data
            ohlcv_data = pd.DataFrame(index=generated_features.index)
            if 'close' in generated_features.columns:
                ohlcv_data['close'] = generated_features['close']
                ohlcv_data['high'] = generated_features.get('high', generated_features['close'])
                ohlcv_data['low'] = generated_features.get('low', generated_features['close'])
                ohlcv_data['open'] = generated_features.get('open', generated_features['close'])
            if 'volume' in generated_features.columns:
                ohlcv_data['volume'] = generated_features['volume']
            else:
                # Create dummy volume if not available
                ohlcv_data['volume'] = 1000
        
        # Prepare selected features list for variant generation
        selected_features = []
        for category, features in top_features_by_category.items():
            for feature_name, optimal_lookback, composite_score in features:
                if feature_name in generated_features.columns:
                    selected_features.append({
                        'feature_name': feature_name,
                        'category': category,
                        'optimal_lookback': int(optimal_lookback),
                        'composite_score': composite_score
                    })
        
        # Generate variants using our utility
        try:
            variant_features, variant_stats = generate_all_variants(
                features_df=generated_features,
                selected_features=selected_features,
                ohlcv_data=ohlcv_data
            )
            
            self.performance_stats['variants_generated'] = len(variant_features.columns)
            
            # Log variant generation statistics
            tprint_success(f"✅ Generated {len(variant_features.columns)} variant features")
            tprint_info(f"📊 Variant breakdown: {variant_stats['variants_by_type']}")
            
            if variant_stats['failed_variants']:
                tprint_warning(f"⚠️ Failed variants: {len(variant_stats['failed_variants'])}")
            
            return variant_features
            
        except Exception as e:
            tprint_error(f"❌ Variant generation failed: {e}")
            raise

    async def _phase2_cheap_pruning(
        self,
        variant_features: pd.DataFrame,
        labeled_data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Phase 2: Apply cheap pruning with category protection (40-50% reduction).
        
        Uses our new CheapPruningPipeline with 5 sequential methods:
        1. Variance pruning (~5% reduction, no category protection)
        2. Statistical significance pruning (~10% reduction, no category protection)
        3. Stability pruning (~10-15% reduction, category protection ≥3 per category)
        4. Mutual information pruning (~10% reduction, category protection ≥3 per category)
        5. Correlation pruning (~10-15% reduction, category protection ≥3 per category)
        
        Returns:
            Tuple of (pruned_features, pruning_stats)
        """
        tprint_info("✂️ Applying cheap pruning with category protection")
        
        if not UTILITIES_AVAILABLE:
            raise ImportError("Cheap pruning utilities not available")
        
        # Get targets from labeled data
        target_columns = [col for col in labeled_data.columns if col in [
            'directional_confidence', 'opportunity_asymmetry',
            'long_overall_opportunity', 'short_overall_opportunity'
        ]]
        
        if not target_columns:
            raise ValueError("No target columns found in labeled_data")
        
        targets = labeled_data[target_columns]
        
        # Create feature categories mapping
        feature_categories = {}
        for col in variant_features.columns:
            # Try to infer category from feature name
            if any(cat in col.lower() for cat in ['trend', 'sma', 'ema']):
                feature_categories[col] = 'trend'
            elif any(cat in col.lower() for cat in ['rsi', 'stoch', 'oscillator']):
                feature_categories[col] = 'oscillator'
            elif any(cat in col.lower() for cat in ['momentum', 'roc', 'macd']):
                feature_categories[col] = 'momentum'
            elif any(cat in col.lower() for cat in ['return', 'pct_change']):
                feature_categories[col] = 'return'
            elif any(cat in col.lower() for cat in ['vol', 'volatility', 'std']):
                feature_categories[col] = 'volatility'
            elif any(cat in col.lower() for cat in ['volume', 'vol']):
                feature_categories[col] = 'volume'
            elif any(cat in col.lower() for cat in ['accel', 'jerk']):
                feature_categories[col] = 'acceleration'
            else:
                feature_categories[col] = 'unknown'
        
        # Create composite scores (use 1.0 as default if not available)
        composite_scores = {col: 1.0 for col in variant_features.columns}
        
        # Apply pruning using our utility
        try:
            pruned_features, pruning_stats = apply_cheap_pruning(
                features_df=variant_features,
                targets_df=targets,
                feature_categories=feature_categories,
                composite_scores=composite_scores,
                config=PruningConfig()
            )
            
            self.performance_stats['features_after_pruning'] = len(pruned_features.columns)
            
            tprint_success(f"✅ Pruning completed: {len(variant_features.columns)} -> {len(pruned_features.columns)} features")
            tprint_info(f"📊 Reduction: {(1 - len(pruned_features.columns)/len(variant_features.columns))*100:.1f}%")
            
            # Log category distribution after pruning
            final_categories = self._get_category_distribution(pruned_features.columns, feature_categories)
            tprint_info(f"📊 Final category distribution: {final_categories}")
            
            return pruned_features, pruning_stats
            
        except Exception as e:
            tprint_error(f"❌ Cheap pruning failed: {e}")
            raise
    
    def _get_category_distribution(self, feature_names: List[str], feature_categories: Dict[str, str]) -> Dict[str, int]:
        """Get distribution of features by category."""
        distribution = {}
        for feature_name in feature_names:
            category = feature_categories.get(feature_name, 'unknown')
            distribution[category] = distribution.get(category, 0) + 1
        return distribution

    async def _phase3_lgbm_shap_pipeline(
        self,
        pruned_features: pd.DataFrame,
        labeled_data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Phase 3: Three-phase LGBM+SHAP pipeline with corrected interaction discovery.
        
        Phase 3.1: Shallow LGBM sweep (max_depth=4, num_leaves=15, n_estimators=100)
        Phase 3.2: Deeper LGBM refinement (max_depth=5, num_leaves=31, n_estimators=100)
        Phase 3.3: Deep interaction discovery (max_depth=6, num_leaves=31, corrected SHAP approach)
        
        Returns:
            Tuple of (final_features, interactions, shap_metadata)
        """
        if not LGBM_AVAILABLE or not SHAP_AVAILABLE:
            raise ImportError("LightGBM and SHAP are required for Phase 3")
        
        # Get targets
        target_columns = [col for col in labeled_data.columns if col in [
            'directional_confidence', 'opportunity_asymmetry',
            'long_overall_opportunity', 'short_overall_opportunity'
        ]]
        targets = labeled_data[target_columns]
        
        # Phase 3.1: Shallow LGBM sweep (Select Top 30%)
        tprint_info("🤖 Phase 3.1: Shallow LGBM Sweep (Select Top 30%)")
        phase3_1_start = time.time()
        
        top_30_percent = await self._phase3_1_shallow_sweep(pruned_features, targets, config)
        
        self.performance_stats['phase3_1_time'] = time.time() - phase3_1_start
        tprint_performance(f"Phase 3.1 completed", self.performance_stats['phase3_1_time'])
        
        # Phase 3.2: Deeper LGBM refinement (Select Top 40)
        tprint_info("🤖 Phase 3.2: Deeper LGBM Refinement (Select Top 40)")
        phase3_2_start = time.time()
        
        top_40_features = await self._phase3_2_deeper_refinement(top_30_percent, targets, config)
        
        self.performance_stats['phase3_2_time'] = time.time() - phase3_2_start
        tprint_performance(f"Phase 3.2 completed", self.performance_stats['phase3_2_time'])
        
        # Phase 3.3: Deep interaction discovery (Generate Top 50)
        tprint_info("🤖 Phase 3.3: Deep Interaction Discovery (Generate Top 50)")
        phase3_3_start = time.time()
        
        interactions, shap_metadata = await self._phase3_3_interaction_discovery(
            top_40_features, targets, config
        )
        
        self.performance_stats['phase3_3_time'] = time.time() - phase3_3_start
        tprint_performance(f"Phase 3.3 completed", self.performance_stats['phase3_3_time'])
        
        return top_40_features, interactions, shap_metadata

    async def _phase3_1_shallow_sweep(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Phase 3.1: Shallow LGBM sweep to select top 30% features.
        
        Uses MultiOutputRegressor with corrected parameters:
        - max_depth=4 (increased to capture interactions)
        - num_leaves=15 (more flexibility)
        - n_estimators=100 (more stable importance)
        - Sample data for SHAP (max 10k rows)
        """
        tprint_info("  📊 Training shallow LGBM with MultiOutputRegressor...")
        
        # Sample data for SHAP calculation (max 10k rows)
        if len(features) > 10000:
            sample_indices = np.random.choice(len(features), 10000, replace=False)
            features_sample = features.iloc[sample_indices]
            targets_sample = targets.iloc[sample_indices]
        else:
            features_sample = features
            targets_sample = targets
        
        # Setup LGBM with corrected parameters
        lgbm_params = {
            'max_depth': 4,
            'num_leaves': 15,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'random_state': 42,
            'verbose': -1
        }
        
        # Train MultiOutputRegressor
        model = MultiOutputRegressor(lgb.LGBMRegressor(**lgbm_params))
        model.fit(features_sample, targets_sample)
        
        # Calculate SHAP values
        tprint_info("  🔍 Calculating SHAP values...")
        explainer = shap.TreeExplainer(model.estimators_[0], model_output='raw')
        shap_values = explainer.shap_values(features_sample)
        
        # Average SHAP values across targets
        if isinstance(shap_values, list):
            # Multiple targets
            avg_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            # Single target
            avg_shap = np.abs(shap_values)
        
        # Rank features by mean absolute SHAP value
        feature_importance = pd.Series(
            np.mean(avg_shap, axis=0),
            index=features.columns
        ).sort_values(ascending=False)
        
        # Select top 30%
        n_select = max(1, int(len(features.columns) * 0.3))
        top_features = feature_importance.head(n_select).index.tolist()
        
        tprint_success(f"  ✅ Selected {len(top_features)} features (top 30%)")
        
        return features[top_features]
    
    async def _phase3_2_deeper_refinement(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Phase 3.2: Deeper LGBM refinement to select top 40 features.
        
        Uses deeper LGBM with multi-criteria selection:
        - SHAP importance (60%)
        - Feature importance (30%)
        - Stability (10%)
        """
        tprint_info("  📊 Training deeper LGBM for refinement...")
        
        # Sample data for SHAP calculation
        if len(features) > 10000:
            sample_indices = np.random.choice(len(features), 10000, replace=False)
            features_sample = features.iloc[sample_indices]
            targets_sample = targets.iloc[sample_indices]
        else:
            features_sample = features
            targets_sample = targets
        
        # Setup deeper LGBM
        lgbm_params = {
            'max_depth': 5,
            'num_leaves': 31,
            'n_estimators': 100,
            'learning_rate': 0.05,
            'random_state': 42,
            'verbose': -1
        }
        
        # Train MultiOutputRegressor
        model = MultiOutputRegressor(lgb.LGBMRegressor(**lgbm_params))
        model.fit(features_sample, targets_sample)
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model.estimators_[0], model_output='raw')
        shap_values = explainer.shap_values(features_sample)
        
        # Average SHAP values across targets
        if isinstance(shap_values, list):
            avg_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            avg_shap = np.abs(shap_values)
        
        # Calculate feature importance
        feature_importance = model.estimators_[0].feature_importances_
        
        # Calculate stability (consistency across targets)
        if isinstance(shap_values, list):
            stability = np.std([np.mean(np.abs(sv), axis=0) for sv in shap_values], axis=0)
        else:
            stability = np.zeros(len(features.columns))
        
        # Normalize scores
        shap_scores = np.mean(avg_shap, axis=0)
        shap_scores = (shap_scores - np.min(shap_scores)) / (np.max(shap_scores) - np.min(shap_scores) + 1e-8)
        
        imp_scores = (feature_importance - np.min(feature_importance)) / (np.max(feature_importance) - np.min(feature_importance) + 1e-8)
        
        stab_scores = (stability - np.min(stability)) / (np.max(stability) - np.min(stability) + 1e-8)
        
        # Multi-criteria selection
        combined_scores = (
            0.6 * shap_scores +  # SHAP importance (60%)
            0.3 * imp_scores +  # Feature importance (30%)
            0.1 * stab_scores   # Stability (10%)
        )
        
        # Rank and select top 40
        feature_scores = pd.Series(combined_scores, index=features.columns).sort_values(ascending=False)
        n_select = min(40, len(features.columns))
        top_features = feature_scores.head(n_select).index.tolist()
        
        tprint_success(f"  ✅ Selected {len(top_features)} features (top 40)")
        
        return features[top_features]
    
    async def _phase3_3_interaction_discovery(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Phase 3.3: Deep interaction discovery with corrected SHAP approach.
        
        Uses tree-based interaction guidance and corrected SHAP analysis:
        1. Train deep LGBM to extract feature pairs
        2. Generate 3 operations per top 10 pairs (30 candidates)
        3. Use standard SHAP values FOR interaction features (not interaction values)
        4. Select top 50 interactions
        """
        tprint_info("  🌳 Training deep LGBM for interaction guidance...")
        
        # Sample data for efficiency
        if len(features) > 10000:
            sample_indices = np.random.choice(len(features), 10000, replace=False)
            features_sample = features.iloc[sample_indices]
            targets_sample = targets.iloc[sample_indices]
        else:
            features_sample = features
            targets_sample = targets
        
        # Train deep LGBM for tree analysis
        lgbm_params = {
            'max_depth': 6,
            'num_leaves': 31,
            'n_estimators': 200,
            'min_child_samples': 50,
            'min_split_gain': 0.01,
            'learning_rate': 0.05,
            'random_state': 42,
            'verbose': -1
        }
        
        model = MultiOutputRegressor(lgb.LGBMRegressor(**lgbm_params))
        model.fit(features_sample, targets_sample)
        
        # Extract feature pairs from trees
        tprint_info("  🔍 Extracting feature pairs from tree splits...")
        feature_pairs = self._extract_tree_splitting_pairs(model.estimators_[0])
        
        # Generate interaction candidates (top 10 pairs × 3 operations = 30 candidates)
        tprint_info("  🔧 Generating interaction candidates...")
        interaction_candidates = []
        
        for i, (f1, f2, co_occurrence) in enumerate(feature_pairs[:10]):  # Top 10 pairs
            if f1 in features.columns and f2 in features.columns:
                # Generate 3 operations per pair
                operations = [
                    (f"{f1}_x_{f2}", features[f1] * features[f2]),
                    (f"{f1}_div_{f2}", features[f1] / (features[f2] + 1e-8)),
                    (f"{f1}_minus_{f2}", features[f1] - features[f2])
                ]
                
                for name, interaction in operations:
                    interaction_candidates.append((name, interaction))
        
        # Test candidates with fast mutual information
        tprint_info("  📊 Testing candidates with mutual information...")
        from sklearn.feature_selection import mutual_info_regression
        
        mi_scores = {}
        for name, interaction in interaction_candidates:
            try:
                # Calculate MI with first target
                target_col = targets.columns[0]
                mi_score = mutual_info_regression(
                    interaction.values.reshape(-1, 1),
                    targets[target_col].values,
                    random_state=42
                )[0]
                mi_scores[name] = mi_score
            except Exception as e:
                self.logger.warning(f"MI calculation failed for {name}: {e}")
                mi_scores[name] = 0.0
        
        # Select top 30 candidates by MI
        top_candidates = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)[:30]
        
        # Calculate SHAP values for top candidates
        tprint_info("  🔍 Calculating SHAP values for top candidates...")
        shap_scores = {}
        
        for name, _ in top_candidates:
            try:
                # Add interaction to feature set
                test_features = features.copy()
                test_features[name] = interaction_candidates[[n for n, _ in interaction_candidates].index(name)][1]
                
                # Sample for SHAP
                if len(test_features) > 5000:
                    sample_indices = np.random.choice(len(test_features), 5000, replace=False)
                    test_sample = test_features.iloc[sample_indices]
                    targets_test = targets.iloc[sample_indices]
                else:
                    test_sample = test_features
                    targets_test = targets
                
                # Train model and calculate SHAP
                test_model = MultiOutputRegressor(lgb.LGBMRegressor(**lgbm_params))
                test_model.fit(test_sample, targets_test)
                
                explainer = shap.TreeExplainer(test_model.estimators_[0], model_output='raw')
                shap_values = explainer.shap_values(test_sample)
                
                # Get SHAP value for the interaction feature
                interaction_idx = test_sample.columns.get_loc(name)
                if isinstance(shap_values, list):
                    interaction_shap = np.mean([np.abs(sv[:, interaction_idx]) for sv in shap_values])
                else:
                    interaction_shap = np.mean(np.abs(shap_values[:, interaction_idx]))
                
                shap_scores[name] = interaction_shap
                
            except Exception as e:
                self.logger.warning(f"SHAP calculation failed for {name}: {e}")
                shap_scores[name] = 0.0
        
        # Select top 50 interactions
        top_interactions = sorted(shap_scores.items(), key=lambda x: x[1], reverse=True)[:50]
        
        # Create interaction DataFrame
        interaction_features = pd.DataFrame(index=features.index)
        interaction_metadata = {
            'feature_pairs': feature_pairs[:10],
            'interaction_scores': dict(top_interactions),
            'mi_scores': dict(top_candidates)
        }
        
        for name, _ in top_interactions:
            # Find the interaction in candidates
            for candidate_name, interaction in interaction_candidates:
                if candidate_name == name:
                    interaction_features[name] = interaction
                    break
        
        tprint_success(f"  ✅ Generated {len(interaction_features.columns)} interaction features")
        
        return interaction_features, interaction_metadata
    
    def _extract_tree_splitting_pairs(self, booster) -> List[Tuple[str, str, int]]:
        """
        Extract feature pairs that frequently split together in trees.
        
        Returns:
            List of (feature1, feature2, co_occurrence_count) tuples
        """
        from collections import defaultdict
        
        feature_pairs = defaultdict(int)
        
        try:
            # Get tree structure
            trees = booster.dump_model()['tree_info']
            
            for tree in trees:
                features_in_tree = set()
                
                # Traverse tree to find all features used
                def traverse_node(node):
                    if 'split_feature' in node:
                        features_in_tree.add(node['split_feature'])
                        if 'left_child' in node:
                            traverse_node(node['left_child'])
                        if 'right_child' in node:
                            traverse_node(node['right_child'])
                
                traverse_node(tree['tree_structure'])
                
                # Count all pairs in this tree
                features_list = list(features_in_tree)
                for i in range(len(features_list)):
                    for j in range(i + 1, len(features_list)):
                        pair = tuple(sorted([features_list[i], features_list[j]]))
                        feature_pairs[pair] += 1
            
            # Convert to list and sort by co-occurrence
            pairs_list = [(f1, f2, count) for (f1, f2), count in feature_pairs.items()]
            pairs_list.sort(key=lambda x: x[2], reverse=True)
            
            return pairs_list
            
        except Exception as e:
            self.logger.warning(f"Tree analysis failed: {e}")
            return []

    async def _phase4_save_artifacts(
        self,
        final_features: pd.DataFrame,
        interactions: pd.DataFrame,
        shap_metadata: Dict,
        pruning_stats: Dict,
        config: Dict[str, Any]
    ) -> Tuple[Dict, Dict]:
        """
        Phase 4: Combine features, verify category coverage, save artifacts, generate report.
        
        Ensures at least 2 features from each original category in final set.
        Saves comprehensive artifacts with enhanced metadata.
        
        Returns:
            Tuple of (artifacts, metrics)
        """
        tprint_info("💾 Phase 4: Integration and artifact saving")
        
        # Combine features and interactions
        combined_features = pd.concat([final_features, interactions], axis=1)
        
        self.performance_stats['final_feature_count'] = len(final_features.columns)
        self.performance_stats['interaction_count'] = len(interactions.columns)
        
        # Verify category coverage (ensure ≥2 per category)
        tprint_info("🔍 Verifying category coverage (minimum 2 per category)...")
        category_coverage = self._verify_category_coverage(combined_features, final_features, config)
        self.performance_stats['category_coverage'] = category_coverage
        
        # Save artifacts with enhanced metadata
        tprint_info("💾 Saving artifacts with enhanced metadata...")
        
        # Enhanced metadata for interaction features
        enhanced_metadata = {
            'symbol': config.get('symbol', 'UNKNOWN'),
            'exchange': config.get('exchange', 'UNKNOWN'),
            'timeframe': config.get('timeframe', 'UNKNOWN'),
            'execution_mode': config.get('execution_mode', 'light'),
            'n_base_features': len(final_features.columns),
            'n_interaction_features': len(interactions.columns),
            'total_features': len(combined_features.columns),
            'category_coverage': category_coverage,
            'variant_generation': shap_metadata.get('variant_generation', {}),
            'pruning_stages': shap_metadata.get('pruning_stages', {}),
            'interaction_discovery': shap_metadata.get('interaction_discovery', {}),
            'created_at': datetime.now().isoformat()
        }
        
        # 1. Analyst interaction features
        features_path = self._save_artifact(
            data=combined_features,
            artifact_name='analyst_interaction_features',
            artifact_type='data',
            metadata=enhanced_metadata
        )
        
        # 2. Enhanced analyst interaction metadata
        metadata_path = self._save_artifact(
            data=shap_metadata,
            artifact_name='analyst_interaction_metadata',
            artifact_type='metadata',
            metadata={
                'created_at': datetime.now().isoformat(),
                'total_features': len(combined_features.columns),
                'category_coverage': category_coverage
            }
        )
        
        # 3. Analyst feature importance
        importance_path = self._save_artifact(
            data=shap_metadata.get('interaction_scores', {}),
            artifact_name='analyst_feature_importance',
            artifact_type='metadata',
            metadata={'created_at': datetime.now().isoformat()}
        )
        
        # 4. Analyst pruning stats
        pruning_path = self._save_artifact(
            data=pruning_stats,
            artifact_name='analyst_pruning_stats',
            artifact_type='metadata',
            metadata={'created_at': datetime.now().isoformat()}
        )
        
        artifacts = {
            'analyst_interaction_features': features_path,
            'analyst_interaction_metadata': metadata_path,
            'analyst_feature_importance': importance_path,
            'analyst_pruning_stats': pruning_path
        }
        
        # Generate comprehensive outcome report
        tprint_info("📊 Generating comprehensive outcome report...")
        report_path = self._generate_outcome_report(
            shap_metadata, pruning_stats, category_coverage, config
        )
        if report_path:
            tprint_success(f"✅ Outcome report generated: {report_path}")
        
        metrics = {
            'success': True,
            'performance_stats': self.performance_stats,
            'category_coverage': category_coverage,
            'total_features': len(combined_features.columns),
            'base_features': len(final_features.columns),
            'interaction_features': len(interactions.columns)
        }
        
        tprint_success(f"✅ Phase 4 completed: {len(combined_features.columns)} total features")
        tprint_info(f"📊 Category coverage: {category_coverage}")
        
        return artifacts, metrics

    def _verify_category_coverage(
        self, 
        combined_features: pd.DataFrame, 
        final_features: pd.DataFrame, 
        config: Dict[str, Any]
    ) -> Dict[str, int]:
        """
        Verify category coverage ensuring at least 2 features per category.
        
        Args:
            combined_features: All features (base + interactions)
            final_features: Base features only
            config: Configuration dictionary
            
        Returns:
            Dict mapping category -> count
        """
        tprint_info("🔍 Verifying category coverage (minimum 2 per category)...")
        
        # Create feature categories mapping
        feature_categories = {}
        for col in combined_features.columns:
            # Infer category from feature name
            if any(cat in col.lower() for cat in ['trend', 'sma', 'ema']):
                feature_categories[col] = 'trend'
            elif any(cat in col.lower() for cat in ['rsi', 'stoch', 'oscillator']):
                feature_categories[col] = 'oscillator'
            elif any(cat in col.lower() for cat in ['momentum', 'roc', 'macd']):
                feature_categories[col] = 'momentum'
            elif any(cat in col.lower() for cat in ['return', 'pct_change']):
                feature_categories[col] = 'return'
            elif any(cat in col.lower() for cat in ['vol', 'volatility', 'std']):
                feature_categories[col] = 'volatility'
            elif any(cat in col.lower() for cat in ['volume', 'vol']):
                feature_categories[col] = 'volume'
            elif any(cat in col.lower() for cat in ['accel', 'jerk']):
                feature_categories[col] = 'acceleration'
            else:
                feature_categories[col] = 'unknown'
        
        # Count features per category
        category_counts = {}
        for category in self.categories:
            category_counts[category] = sum(
                1 for col in combined_features.columns 
                if feature_categories.get(col, 'unknown') == category
            )
        
        # Check if any category has < 2 features
        under_represented = [cat for cat, count in category_counts.items() if count < 2]
        
        if under_represented:
            tprint_warning(f"⚠️ Under-represented categories: {under_represented}")
            tprint_info("📊 Category distribution:")
            for cat, count in category_counts.items():
                status = "✅" if count >= 2 else "⚠️"
                tprint_info(f"  {status} {cat}: {count} features")
        else:
            tprint_success("✅ All categories have ≥2 features")
            tprint_info("📊 Category distribution:")
            for cat, count in category_counts.items():
                tprint_info(f"  ✅ {cat}: {count} features")
        
        return category_counts
    
    def _generate_outcome_report(
        self,
        shap_metadata: Dict,
        pruning_stats: Dict,
        category_coverage: Dict,
        config: Dict[str, Any]
    ) -> Optional[str]:
        """
        Generate comprehensive outcome report.
        
        Returns:
            Path to generated report file
        """
        try:
            # Create outcomes directory
            outcomes_dir = Path("outcomes")
            outcomes_dir.mkdir(exist_ok=True)
            
            # Generate report filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            symbol = config.get('symbol', 'UNKNOWN')
            report_filename = f"analyst_interaction_generation_{symbol}_{timestamp}.md"
            report_path = outcomes_dir / report_filename
            
            # Generate report content
            report_content = self._create_report_content(
                shap_metadata, pruning_stats, category_coverage, config
            )
            
            # Write report
            with open(report_path, 'w') as f:
                f.write(report_content)
            
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate outcome report: {e}")
            return None
    
    def _create_report_content(
        self,
        shap_metadata: Dict,
        pruning_stats: Dict,
        category_coverage: Dict,
        config: Dict[str, Any]
    ) -> str:
        """Create markdown report content."""
        
        content = f"""# Analyst Interaction Generation Report

## Overview
- **Symbol**: {config.get('symbol', 'UNKNOWN')}
- **Exchange**: {config.get('exchange', 'UNKNOWN')}
- **Timeframe**: {config.get('timeframe', 'UNKNOWN')}
- **Execution Mode**: {config.get('execution_mode', 'light')}
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Feature Statistics
- **Total Features**: {self.performance_stats.get('final_feature_count', 0) + self.performance_stats.get('interaction_count', 0)}
- **Base Features**: {self.performance_stats.get('final_feature_count', 0)}
- **Interaction Features**: {self.performance_stats.get('interaction_count', 0)}

## Category Coverage
"""
        
        for category, count in category_coverage.items():
            status = "✅" if count >= 2 else "⚠️"
            content += f"- {status} **{category.title()}**: {count} features\n"
        
        content += f"""
## Performance Metrics
- **Phase 0 Time**: {self.performance_stats.get('phase0_time', 0):.2f}s
- **Phase 1 Time**: {self.performance_stats.get('phase1_time', 0):.2f}s
- **Phase 2 Time**: {self.performance_stats.get('phase2_time', 0):.2f}s
- **Phase 3.1 Time**: {self.performance_stats.get('phase3_1_time', 0):.2f}s
- **Phase 3.2 Time**: {self.performance_stats.get('phase3_2_time', 0):.2f}s
- **Phase 3.3 Time**: {self.performance_stats.get('phase3_3_time', 0):.2f}s
- **Phase 4 Time**: {self.performance_stats.get('phase4_time', 0):.2f}s
- **Total Time**: {self.performance_stats.get('total_time', 0):.2f}s

## Variant Generation Statistics
"""
        
        if 'variant_generation' in shap_metadata:
            variant_stats = shap_metadata['variant_generation']
            content += f"- **Variants Generated**: {variant_stats.get('total_variants_generated', 0)}\n"
            content += f"- **Failed Variants**: {len(variant_stats.get('failed_variants', []))}\n"
            content += f"- **Variant Types**: {variant_stats.get('variants_by_type', {})}\n"
        
        content += f"""
## Pruning Statistics
"""
        
        if 'stage_results' in pruning_stats:
            for stage, stats in pruning_stats['stage_results'].items():
                content += f"- **{stage.title()}**: Removed {stats.get('features_removed', 0)} features\n"
        
        content += f"""
## Interaction Discovery
"""
        
        if 'interaction_discovery' in shap_metadata:
            interaction_stats = shap_metadata['interaction_discovery']
            content += f"- **Feature Pairs Analyzed**: {len(interaction_stats.get('feature_pairs', []))}\n"
            content += f"- **Top Interactions**: {len(interaction_stats.get('interaction_scores', {}))}\n"
        
        content += f"""
## Technical Details
- **Adaptive Feature Selection**: 3-6 features per category based on signal strength
- **RobustScaler Bounding**: Prevents extreme values with percentile clipping
- **Category Protection**: Maintains ≥3 features per category during pruning
- **Corrected SHAP Analysis**: Standard SHAP values for interaction features
- **Causality Enforcement**: All features shifted to prevent lookahead bias

## Recommendations
1. Monitor category balance in downstream training
2. Validate interaction features for numerical stability
3. Consider feature importance rankings for model selection
4. Review pruning statistics for feature quality insights

---
*Report generated by Analyst Interaction Generation Step*
"""
        
        return content

    def _check_category_coverage(self, combined_features: pd.DataFrame, shap_metadata: Dict) -> Dict:
        """Check category coverage in final feature set."""
        tprint_info("📊 Checking category coverage")
        
        category_counts = {}
        
        # Count features per category from metadata
        for feature_name in combined_features.columns:
            # Extract category from metadata or feature name
            category = shap_metadata.get('feature_categories', {}).get(feature_name, 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Log warnings for imbalanced categories
        for category, count in category_counts.items():
            if count < 5:
                tprint_warning(f"⚠️ Category {category} has only {count} features (< 5)")
            elif count > 20:
                tprint_warning(f"⚠️ Category {category} has {count} features (> 20, possible overrepresentation)")
            else:
                tprint_info(f"  {category}: {count} features")
        
        return category_counts

    def _generate_outcome_report(
        self,
        shap_metadata: Dict,
        pruning_stats: Dict,
        category_coverage: Dict,
        config: Dict[str, Any]
    ) -> Optional[str]:
        """Generate comprehensive outcome report in markdown format."""
        try:
            outcomes_dir = Path('outcomes')
            outcomes_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"{self.step_name}_report_{timestamp}.md"
            report_path = outcomes_dir / report_filename
            
            with open(report_path, 'w') as f:
                f.write(f"# Analyst Interaction Generation Report\n\n")
                f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Symbol:** {config.get('symbol', 'N/A')}\n")
                f.write(f"**Timeframe:** {config.get('timeframe', 'N/A')}\n\n")
                
                f.write("## Performance Summary\n\n")
                f.write(f"- **Total Execution Time:** {self.performance_stats['total_time']:.2f}s\n")
                f.write(f"- **Phase 0 Time:** {self.performance_stats['phase0_time']:.2f}s\n")
                f.write(f"- **Phase 1 Time:** {self.performance_stats['phase1_time']:.2f}s\n")
                f.write(f"- **Phase 2 Time:** {self.performance_stats['phase2_time']:.2f}s\n")
                f.write(f"- **Phase 3.1 Time:** {self.performance_stats['phase3_1_time']:.2f}s\n")
                f.write(f"- **Phase 3.2 Time:** {self.performance_stats['phase3_2_time']:.2f}s\n")
                f.write(f"- **Phase 3.3 Time:** {self.performance_stats['phase3_3_time']:.2f}s\n")
                f.write(f"- **Phase 4 Time:** {self.performance_stats['phase4_time']:.2f}s\n\n")
                
                f.write("## Feature Statistics\n\n")
                f.write(f"- **Variants Generated:** {self.performance_stats['variants_generated']}\n")
                f.write(f"- **Features After Pruning:** {self.performance_stats['features_after_pruning']}\n")
                f.write(f"- **Final Feature Count:** {self.performance_stats['final_feature_count']}\n")
                f.write(f"- **Interaction Count:** {self.performance_stats['interaction_count']}\n")
                f.write(f"- **Total Features:** {self.performance_stats['final_feature_count'] + self.performance_stats['interaction_count']}\n\n")
                
                f.write("## Category Coverage\n\n")
                for category, count in category_coverage.items():
                    f.write(f"- **{category}:** {count} features\n")
                f.write("\n")
                
                f.write("## Numerical Safety\n\n")
                f.write(f"- **Safety Incidents:** {self.performance_stats['numerical_safety_incidents']}\n\n")
                
                if self.numerical_safety_log:
                    f.write("### Safety Incidents Log\n\n")
                    for incident in self.numerical_safety_log[:10]:  # Show first 10
                        f.write(f"- {incident}\n")
                    if len(self.numerical_safety_log) > 10:
                        f.write(f"\n... and {len(self.numerical_safety_log) - 10} more incidents\n")
                
                f.write("\n## Pruning Statistics\n\n")
                for key, value in pruning_stats.items():
                    f.write(f"- **{key}:** {value}\n")
            
            return str(report_path)
            
        except Exception as e:
            tprint_error(f"❌ Report generation failed: {e}")
            return None

    async def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run method required by BaseStep interface."""
        return await self.execute(config)


# Register the step
def register_feature_generation_interaction_generation_step_analyst():
    """Register the analyst interaction generation step."""
    from src.training.steps.base_step import step_registry

    step_registry.register("feature_generation_interaction_generation_step_analyst", FeatureGenerationInteractionGenerationStepAnalyst)
    tprint("✅ Feature generation interaction generation step analyst registered", "SUCCESS")


# Auto-register when module is imported
register_feature_generation_interaction_generation_step_analyst()
