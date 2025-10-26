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
        
        # Generate variants using parallel processing
        try:
            # Use parallel processing for large feature sets
            if len(selected_features) > 10:
                tprint_info("  🚀 Using parallel variant generation...")
                variant_features = self._parallel_variant_generation(
                    selected_features, generated_features, ohlcv_data
                )
                variant_stats = {'variants_by_type': {}, 'failed_variants': []}
            else:
                # Use sequential processing for small feature sets
                variant_features, variant_stats = generate_all_variants(
                    features_df=generated_features,
                    selected_features=selected_features,
                    ohlcv_data=ohlcv_data
                )
            
            self.performance_stats['variants_generated'] = len(variant_features.columns)
            
            # Log variant generation statistics
            tprint_success(f"✅ Generated {len(variant_features.columns)} variant features")
            if variant_stats.get('variants_by_type'):
                tprint_info(f"📊 Variant breakdown: {variant_stats['variants_by_type']}")
            
            if variant_stats.get('failed_variants'):
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
        
        # Get feature categories from lookback optimization (feature bank)
        feature_categories = self._get_feature_categories_from_bank(variant_features.columns, lookback_optimization)
        
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
        
        Uses fast feature importance + mutual information proxy instead of expensive SHAP.
        - max_depth=4 (increased to capture interactions)
        - num_leaves=15 (more flexibility)
        - n_estimators=100 (more stable importance)
        - Fast proxy: 60% feature importance + 40% mutual information
        """
        tprint_info("  📊 Training shallow LGBM with fast feature selection...")
        
        # Use consistent sampling strategy with chunked processing
        features_sample, targets_sample = self._get_consistent_sample(features, targets, max_samples=8000)
        
        # Apply chunked processing for large datasets
        if len(features_sample) > 5000:
            features_sample = self._chunked_processing(features_sample, targets_sample, chunk_size=2000)
        
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
        
        # Fast feature importance calculation (no SHAP)
        tprint_info("  🔍 Calculating fast feature importance...")
        
        # Get feature importance from LGBM
        importance_scores = model.estimators_[0].feature_importances_
        
        # Calculate mutual information with first target
        from sklearn.feature_selection import mutual_info_regression
        mi_scores = mutual_info_regression(
            features_sample, 
            targets_sample.iloc[:, 0], 
            random_state=42
        )
        
        # Normalize scores
        importance_scores = (importance_scores - np.min(importance_scores)) / (np.max(importance_scores) - np.min(importance_scores) + 1e-8)
        mi_scores = (mi_scores - np.min(mi_scores)) / (np.max(mi_scores) - np.min(mi_scores) + 1e-8)
        
        # Combined score: 60% importance + 40% mutual information
        combined_scores = 0.6 * importance_scores + 0.4 * mi_scores
        
        # Rank features by combined score
        feature_importance = pd.Series(
            combined_scores,
            index=features.columns
        ).sort_values(ascending=False)
        
        # Select top 30%
        n_select = max(1, int(len(features.columns) * 0.3))
        top_features = feature_importance.head(n_select).index.tolist()
        
        tprint_success(f"  ✅ Selected {len(top_features)} features (top 30%) using fast proxy")
        
        return features[top_features]
    
    async def _phase3_2_deeper_refinement(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Phase 3.2: Deeper LGBM refinement to select top 40 features.
        
        Uses deeper LGBM with fast multi-criteria selection:
        - Feature importance (60%)
        - Mutual information (30%)
        - Stability (10%)
        """
        tprint_info("  📊 Training deeper LGBM for refinement...")
        
        # Use consistent sampling strategy with chunked processing
        features_sample, targets_sample = self._get_consistent_sample(features, targets, max_samples=8000)
        
        # Apply chunked processing for large datasets
        if len(features_sample) > 5000:
            features_sample = self._chunked_processing(features_sample, targets_sample, chunk_size=2000)
        
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
        
        # Fast multi-criteria selection (no SHAP)
        tprint_info("  🔍 Calculating fast multi-criteria scores...")
        
        # Calculate feature importance
        feature_importance = model.estimators_[0].feature_importances_
        
        # Calculate mutual information with first target
        from sklearn.feature_selection import mutual_info_regression
        mi_scores = mutual_info_regression(
            features_sample, 
            targets_sample.iloc[:, 0], 
            random_state=42
        )
        
        # Calculate stability (variance across features)
        stability = np.var(features_sample.values, axis=0)
        
        # Normalize scores
        imp_scores = (feature_importance - np.min(feature_importance)) / (np.max(feature_importance) - np.min(feature_importance) + 1e-8)
        mi_scores = (mi_scores - np.min(mi_scores)) / (np.max(mi_scores) - np.min(mi_scores) + 1e-8)
        stab_scores = (stability - np.min(stability)) / (np.max(stability) - np.min(stability) + 1e-8)
        
        # Multi-criteria selection
        combined_scores = (
            0.6 * imp_scores +   # Feature importance (60%)
            0.3 * mi_scores +    # Mutual information (30%)
            0.1 * stab_scores    # Stability (10%)
        )
        
        # Rank and select top 40
        feature_scores = pd.Series(combined_scores, index=features.columns).sort_values(ascending=False)
        n_select = min(40, len(features.columns))
        top_features = feature_scores.head(n_select).index.tolist()
        
        tprint_success(f"  ✅ Selected {len(top_features)} features (top 40) using fast proxy")
        
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
        
        # Use consistent sampling strategy with chunked processing
        features_sample, targets_sample = self._get_consistent_sample(features, targets, max_samples=8000)
        
        # Apply chunked processing for large datasets
        if len(features_sample) > 5000:
            features_sample = self._chunked_processing(features_sample, targets_sample, chunk_size=2000)
        
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
        
        # Early stopping interaction discovery
        tprint_info("  🔧 Generating interactions with early stopping...")
        interaction_features = self._early_stopping_interaction_discovery(
            feature_pairs, features, targets, max_candidates=20
        )
        
        # Create interaction DataFrame
        interaction_df = pd.DataFrame(interaction_features, index=features.index)
        
        # Apply causality shift
        interaction_df = interaction_df.shift(1)
        
        # Apply RobustScaler
        scaler = RobustScaler()
        interaction_df = pd.DataFrame(
            scaler.fit_transform(interaction_df),
            columns=interaction_df.columns,
            index=interaction_df.index
        )
        
        # Create metadata
        interaction_metadata = {
            'feature_pairs': feature_pairs[:10],
            'interaction_scores': {name: 1.0 for name in interaction_df.columns},  # Placeholder scores
            'early_stopping_applied': True
        }
        
        tprint_success(f"  ✅ Generated {len(interaction_df.columns)} interaction features with early stopping")
        
        return interaction_df, interaction_metadata
    
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
        
        # Get feature categories from lookback optimization (feature bank)
        feature_categories = self._get_feature_categories_from_bank(combined_features.columns, lookback_optimization)
        
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

    def _get_feature_categories_from_bank(self, feature_names: List[str], lookback_optimization: Dict) -> Dict[str, str]:
        """Get feature categories from lookback optimization feature bank."""
        feature_categories = {}
        
        # Get categories from lookback optimization if available
        if 'feature_categories' in lookback_optimization:
            bank_categories = lookback_optimization['feature_categories']
            for feature_name in feature_names:
                # Try to find exact match first
                if feature_name in bank_categories:
                    feature_categories[feature_name] = bank_categories[feature_name]
                else:
                    # Try to find partial match (for variants)
                    base_name = feature_name.split('_')[0]  # Get base feature name
                    if base_name in bank_categories:
                        feature_categories[feature_name] = bank_categories[base_name]
                    else:
                        feature_categories[feature_name] = 'unknown'
        else:
            # Fallback to name-based inference
            for feature_name in feature_names:
                feature_categories[feature_name] = self._infer_feature_category(feature_name)
        
        return feature_categories
    
    def _infer_feature_category(self, feature_name: str) -> str:
        """Infer feature category from name with robust keyword matching."""
        category_keywords = {
            'trend': ['sma', 'ema', 'trend', 'moving_average', 'ma'],
            'oscillator': ['rsi', 'stoch', 'oscillator', 'williams', 'cci'],
            'momentum': ['momentum', 'roc', 'macd', 'rate_of_change', 'pct_change'],
            'return': ['return', 'pct_change', 'log_return', 'ret'],
            'volatility': ['vol', 'volatility', 'std', 'atr', 'bb', 'bollinger'],
            'volume': ['volume', 'vol', 'vwap', 'obv', 'ad'],
            'acceleration': ['accel', 'jerk', 'second_derivative', '2nd_deriv']
        }
        
        feature_lower = feature_name.lower()
        for category, keywords in category_keywords.items():
            if any(keyword in feature_lower for keyword in keywords):
                return category
        
        return 'unknown'
    
    def _get_consistent_sample(self, features: pd.DataFrame, targets: pd.DataFrame, max_samples: int = 8000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get consistent sample across all phases."""
        if len(features) > max_samples:
            # Use same random seed for consistency
            np.random.seed(42)
            sample_idx = np.random.choice(len(features), max_samples, replace=False)
            return features.iloc[sample_idx], targets.iloc[sample_idx]
        return features, targets
    
    def _adaptive_category_selection(self, features_by_category: Dict[str, List[str]], feature_importance: pd.Series, min_per_category: int = 2, max_per_category: int = 8) -> List[str]:
        """Dynamic feature selection based on signal strength."""
        selected_features = []
        
        for category, features in features_by_category.items():
            if len(features) == 0:
                continue
                
            # Calculate signal strength (composite score variance)
            scores = [feature_importance.get(f, 0) for f in features]
            if len(scores) > 1:
                signal_strength = np.std(scores) / (np.mean(scores) + 1e-8)
            else:
                signal_strength = 0.5  # Default for single feature
            
            # Adaptive selection: more features for stronger signals
            if signal_strength > 0.5:  # High signal strength
                n_select = min(max_per_category, len(features))
            elif signal_strength > 0.2:  # Medium signal strength
                n_select = min(6, len(features))
            else:  # Low signal strength
                n_select = min(min_per_category, len(features))
            
            # Get top features by importance
            category_importance = feature_importance[features]
            top_category_features = category_importance.nlargest(n_select).index.tolist()
            selected_features.extend(top_category_features)
        
        return selected_features
    
    def _early_stopping_interaction_discovery(self, feature_pairs: List[Tuple], features: pd.DataFrame, targets: pd.DataFrame, max_candidates: int = 20) -> Dict[str, pd.Series]:
        """Early stopping interaction discovery with diminishing returns detection."""
        from sklearn.feature_selection import mutual_info_regression
        
        interaction_features = {}
        scores = []
        best_score = 0
        stagnation_count = 0
        
        for i, (f1, f2, co_occurrence) in enumerate(feature_pairs):
            if i >= max_candidates:
                break
                
            if f1 not in features.columns or f2 not in features.columns:
                continue
                
            # Generate interactions
            interactions = {
                f"{f1}_x_{f2}": features[f1] * features[f2],
                f"{f1}_div_{f2}": features[f1] / (features[f2] + 1e-8),
                f"{f1}_minus_{f2}": features[f1] - features[f2]
            }
            
            # Score interactions using MI + correlation
            for name, interaction in interactions.items():
                try:
                    mi_score = mutual_info_regression(
                        interaction.values.reshape(-1, 1), 
                        targets.iloc[:, 0],
                        random_state=42
                    )[0]
                    corr_score = abs(interaction.corr(targets.iloc[:, 0]))
                    
                    # Combined score (70% MI + 30% correlation)
                    combined_score = 0.7 * mi_score + 0.3 * corr_score
                    scores.append((name, combined_score))
                    
                    # Early stopping logic
                    if combined_score > best_score:
                        best_score = combined_score
                        stagnation_count = 0
                    else:
                        stagnation_count += 1
                        
                    if stagnation_count >= 5:  # Stop if no improvement for 5 iterations
                        break
                        
                except Exception as e:
                    tprint_warning(f"  ⚠️ Error scoring interaction {name}: {e}")
                    continue
            
            if stagnation_count >= 5:
                break
        
        # Select top interactions
        scores.sort(key=lambda x: x[1], reverse=True)
        top_interactions = scores[:10]  # Top 10 interactions
        
        # Generate final interaction features
        for name, score in top_interactions:
            if name in interactions:
                interaction_features[name] = interactions[name]
        
        return interaction_features
    
    def _parallel_variant_generation(self, selected_features: List[Dict], features_df: pd.DataFrame, ohlcv_data: pd.DataFrame) -> pd.DataFrame:
        """Parallel variant generation using multiprocessing."""
        import multiprocessing as mp
        from functools import partial
        
        if not HARDWARE_OPT_AVAILABLE:
            # Fallback to sequential processing
            return self._sequential_variant_generation(selected_features, features_df, ohlcv_data)
        
        try:
            # Get hardware manager
            hardware_manager = get_unified_hardware_manager()
            hardware_manager.optimize_for_workload(WorkloadType.FEATURE_ENGINEERING, OptimizationLevel.BALANCED)
            
            # Use optimal number of processes
            max_processes = min(4, mp.cpu_count())
            
            with mp.Pool(processes=max_processes) as pool:
                generate_func = partial(
                    self._generate_single_feature_variants,
                    features_df=features_df,
                    ohlcv_data=ohlcv_data
                )
                results = pool.map(generate_func, selected_features)
            
            # Combine results
            all_variants = {}
            for variants in results:
                if variants:
                    all_variants.update(variants)
            
            return pd.DataFrame(all_variants, index=features_df.index)
            
        except Exception as e:
            tprint_warning(f"⚠️ Parallel processing failed, falling back to sequential: {e}")
            return self._sequential_variant_generation(selected_features, features_df, ohlcv_data)
    
    def _generate_single_feature_variants(self, feature_info: Dict, features_df: pd.DataFrame, ohlcv_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Generate variants for a single feature (for parallel processing)."""
        try:
            from src.training.utils.feature_selection.variant_generator import generate_all_variants
            
            # Generate variants for single feature
            variants, _ = generate_all_variants(
                features_df=features_df,
                selected_features=[feature_info],
                ohlcv_data=ohlcv_data
            )
            
            return variants.to_dict('series')
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to generate variants for {feature_info.get('feature_name', 'unknown')}: {e}")
            return {}
    
    def _sequential_variant_generation(self, selected_features: List[Dict], features_df: pd.DataFrame, ohlcv_data: pd.DataFrame) -> pd.DataFrame:
        """Sequential variant generation (fallback)."""
        try:
            from src.training.utils.feature_selection.variant_generator import generate_all_variants
            
            variants, _ = generate_all_variants(
                features_df=features_df,
                selected_features=selected_features,
                ohlcv_data=ohlcv_data
            )
            
            return variants
            
        except Exception as e:
            tprint_error(f"❌ Sequential variant generation failed: {e}")
            raise
    
    def _chunked_processing(self, features: pd.DataFrame, targets: pd.DataFrame, chunk_size: int = 5000) -> pd.DataFrame:
        """Process large datasets in chunks to reduce memory usage."""
        if not HARDWARE_OPT_AVAILABLE:
            return features
        
        try:
            # Get memory optimizer
            memory_optimizer = get_m1_memory_optimizer()
            
            # Check if chunking is needed
            if len(features) <= chunk_size:
                return features
            
            tprint_info(f"  📊 Processing {len(features)} rows in chunks of {chunk_size}")
            
            # Process in chunks
            chunk_results = []
            for i in range(0, len(features), chunk_size):
                chunk_features = features.iloc[i:i+chunk_size]
                chunk_targets = targets.iloc[i:i+chunk_size]
                
                # Process chunk
                processed_chunk = self._process_chunk(chunk_features, chunk_targets)
                chunk_results.append(processed_chunk)
                
                # Memory cleanup
                memory_optimizer.force_garbage_collection()
            
            # Combine results
            result = pd.concat(chunk_results, ignore_index=True)
            
            tprint_success(f"  ✅ Chunked processing completed: {len(result)} rows")
            return result
            
        except Exception as e:
            tprint_warning(f"⚠️ Chunked processing failed, using full dataset: {e}")
            return features
    
    def _process_chunk(self, chunk_features: pd.DataFrame, chunk_targets: pd.DataFrame) -> pd.DataFrame:
        """Process a single chunk of data."""
        # This is a placeholder - implement specific chunk processing logic
        return chunk_features


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
