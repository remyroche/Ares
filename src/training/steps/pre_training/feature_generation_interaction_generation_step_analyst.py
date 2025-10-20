"""
Feature Generation Interaction Generation Step - Analyst Mode

Three-phase LGBM+SHAP pipeline for interaction generation:
1. Phase 1: Variant generation & shallow LGBM sweep (top 40% selection)
2. Phase 2: Middle refinement with deeper LGBM (top 40 selection)
3. Phase 3: Deep interaction discovery with full LGBM (top 50 interactions)

Optimized with M1 hardware utilities, chunked processing, memory optimization,
and VectorBT acceleration for maximum performance on Apple Silicon.
"""

from __future__ import annotations

import logging
import gc
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, List, Tuple
import concurrent.futures
import threading
import time
import os

from src.utils.tprint import tprint

from src.training.steps.base_step import BaseStep



# Import utility classes
from src.training.steps.pre_training.unified_data_driven_pipeline.utils.variant_generator import (
    FeatureVariantGenerator, VariantConfig
)
from src.training.steps.pre_training.unified_data_driven_pipeline.utils.interaction_generator import (
    FeatureInteractionGenerator, InteractionConfig
)
from src.training.steps.pre_training.unified_data_driven_pipeline.utils.shap_interaction_scorer import (
    SHAPInteractionScorer, SHAPScorerConfig
)

# Import optimization integration
from src.training.steps.pre_training.unified_data_driven_pipeline.utils.optimization_integration import (
    OptimizationIntegrationManager, IntegratedOptimizationConfig
)

# M1 Hardware Optimization Imports
from src.utils.hardware.m1_gpu_utils import (
    get_m1_gpu_manager, optimize_dataframe_for_m1, create_m1_optimized_array
)
from src.utils.hardware.m1_memory_optimizer import (
    get_m1_memory_optimizer, optimize_dataframe_memory
)
from src.utils.hardware.m1_cpu_optimizer import (
    get_m1_cpu_optimizer, create_m1_optimized_thread_pool, parallel_map_m1
)

# VectorBT and Unified Vectorization Imports
from src.utils.ml_common.unified_vectorization_manager import (
    get_unified_vectorization_manager, OperationType, OptimizationStrategy
)

@dataclass
class InteractionGenerationResult:
    success: bool
    interaction_features: pd.DataFrame
    interaction_metadata: Dict[str, Any]
    generation_metrics: Dict[str, Any]
    artifacts: Dict[str, Any]
    error_message: Optional[str] = None

@dataclass
class PhaseConfig:
    """Configuration for each phase of the pipeline."""
    
    # Phase 1: Shallow LGBM sweep
    phase1_lgbm_params: Dict[str, Any] = None
    phase1_n_folds: int = 3
    phase1_selection_ratio: float = 0.4  # Top 40%
    
    # Phase 2: Middle refinement
    phase2_lgbm_params: Dict[str, Any] = None
    phase2_n_folds: int = 3
    phase2_top_k: int = 40
    
    # Phase 3: Deep interaction discovery
    phase3_lgbm_params: Dict[str, Any] = None
    phase3_n_folds: int = 5
    phase3_top_interactions: int = 50
    
    # Data sampling by mode
    light_mode_sample_size: int = 50000
    blank_mode_sample_size: int = 250000
    full_mode_sample_size: int = 250000
    
    def __post_init__(self):
        # Phase 1: Shallow LGBM
        if self.phase1_lgbm_params is None:
            self.phase1_lgbm_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'n_estimators': 50,
                'learning_rate': 0.08,
                'num_leaves': 15,
                'max_depth': 4,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.7,
                'bagging_freq': 5,
                'verbose': -1,
                'force_col_wise': True,
                'early_stopping_rounds': 10
            }
        
        # Phase 2: Middle refinement
        if self.phase2_lgbm_params is None:
            self.phase2_lgbm_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'n_estimators': 100,
                'learning_rate': 0.08,
                'num_leaves': 31,
                'max_depth': 6,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.7,
                'bagging_freq': 5,
                'verbose': -1,
                'force_col_wise': True,
                'early_stopping_rounds': 25
            }
        
        # Phase 3: Deep interaction discovery
        if self.phase3_lgbm_params is None:
            self.phase3_lgbm_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'n_estimators': 800,
                'learning_rate': 0.05,
                'num_leaves': 63,
                'max_depth': 8,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.7,
                'bagging_freq': 5,
                'verbose': -1,
                'force_col_wise': True,
                'early_stopping_rounds': 50
            }

@dataclass
class FeatureGenerationInteractionGenerationStepAnalyst(BaseStep):
    """Analyst mode interaction generation step with three-phase LGBM+SHAP pipeline."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("feature_generation_interaction_generation_step_analyst", config)
        
        # Initialize phase configuration
        self.phase_config = PhaseConfig()
        
        # Initialize M1 hardware optimizers
        tprint("🧠 Initializing M1 hardware optimizers...")
        self.m1_gpu_manager = get_m1_gpu_manager()
        self.m1_memory_optimizer = get_m1_memory_optimizer(memory_limit_gb=8.0)
        self.m1_cpu_optimizer = get_m1_cpu_optimizer()
        
        # Initialize VectorBT and unified vectorization
        tprint("🚀 Initializing VectorBT and unified vectorization...")
        self.unified_vectorization_manager = get_unified_vectorization_manager()
        
        # Initialize optimization integration manager
        tprint("🚀 Initializing optimization integration manager...")
        optimization_config = IntegratedOptimizationConfig(
            memory_mapping_threshold_gb=2.0,
            enable_gpu_acceleration=True,
            enable_int32_downcasting=True,
            enable_float32_downcasting=True,
            max_interactions=1000
        )
        self.optimization_manager = OptimizationIntegrationManager(optimization_config)
        
        # Initialize utility classes
        self.variant_generator = FeatureVariantGenerator()
        self.interaction_generator = FeatureInteractionGenerator()
        
        # Parallel processing configuration
        self.parallel_workers = 6
        self.chunk_size = 10000
        self.memory_mapped_threshold = 50000
        
        # Memory optimization settings
        self.aggressive_gc_threshold = 0.8
        self.float32_conversion = True
        
        # Performance tracking
        self.performance_stats = {
            'total_processing_time': 0.0,
            'memory_optimizations_applied': 0,
            'chunks_processed': 0,
            'gpu_accelerations_used': 0,
            'vectorbt_optimizations_used': 0,
            'phase1_time': 0.0,
            'phase2_time': 0.0,
            'phase3_time': 0.0
        }
        
        tprint("✅ M1-optimized Analyst interaction generation step initialized")

    def _optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage with M1-specific optimizations."""
        if df is None or df.empty:
            return df
            
        tprint("🧠 Applying M1 memory optimizations to DataFrame...")
        
        # Apply M1-specific memory optimization
        optimized_df = optimize_dataframe_for_m1(df)
        
        # Convert float64 to float32 where precision allows
        if self.float32_conversion:
            numeric_cols = optimized_df.select_dtypes(include=[np.float64]).columns
            for col in numeric_cols:
                if optimized_df[col].min() >= np.finfo(np.float32).min and \
                   optimized_df[col].max() <= np.finfo(np.float32).max:
                    optimized_df[col] = optimized_df[col].astype(np.float32)
                    
        # Apply pandas memory optimization
        optimized_df = self.m1_memory_optimizer.optimize_dataframe_memory(optimized_df)
        
        self.performance_stats['memory_optimizations_applied'] += 1
        tprint(f"✅ DataFrame memory optimized: {optimized_df.shape}")
        return optimized_df

    def _sample_data_by_mode(self, data: pd.DataFrame, mode: str) -> pd.DataFrame:
        """Sample data based on mode (light/blank/full)."""
        if mode == 'light':
            sample_size = self.phase_config.light_mode_sample_size
        elif mode == 'blank':
            sample_size = self.phase_config.blank_mode_sample_size
        else:  # full
            sample_size = self.phase_config.full_mode_sample_size
            
        if len(data) <= sample_size:
            tprint(f"📊 Data size ({len(data)}) <= sample size ({sample_size}), using all data")
            return data
            
        # Sample most recent data
        sampled_data = data.tail(sample_size)
        tprint(f"📊 Sampled {len(sampled_data)} rows from {len(data)} total rows for {mode} mode")
        
        return sampled_data

    def _load_timeframes_from_optimization(self, artifact_manager) -> List[str]:
        """Load optimized timeframes from period_lookback_optimization step (top2-3 for interactions)."""
        try:
            # Try to load top periods first (top2-3 for interactions)
            top_periods = artifact_manager.get_artifact('feature_generation_period_lookback_optimization_step', 'top_periods')
            if top_periods and isinstance(top_periods, list) and len(top_periods) >= 2:
                # Use top 2-3 periods for interactions
                interaction_periods = top_periods[:3] if len(top_periods) >= 3 else top_periods
                tprint(f"📊 Loaded top periods for interactions (top2-3): {interaction_periods}")
                return interaction_periods
            
            # Fallback to optimized periods
            opt_periods = artifact_manager.get_artifact('period_lookback_optimization', 'optimized_periods')
            if opt_periods and isinstance(opt_periods, list):
                # Use top 2-3 periods for interactions
                interaction_periods = opt_periods[:3] if len(opt_periods) >= 3 else opt_periods
                tprint(f"📊 Loaded optimized periods for interactions (top2-3): {interaction_periods}")
                return interaction_periods
        except Exception as e:
            tprint(f"⚠️ Failed to load optimized timeframes: {e}")
            
        # Fallback to default timeframes
        default_timeframes = ['15m', '1h', '4h']
        tprint(f"📊 Using default timeframes: {default_timeframes}")
        return default_timeframes

    def _execute_phase1(self, features_df: pd.DataFrame, targets: pd.Series, 
                       price_data: Optional[pd.DataFrame] = None,
                       volume: Optional[pd.Series] = None,
                       timeframes: List[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute Phase 1: Variant generation & shallow LGBM sweep."""
        tprint("🔧 [PHASE1] Starting Phase 1: Variant generation & shallow LGBM sweep")
        phase_start_time = time.time()
        
        try:
            # Generate variants for each timeframe
            all_variants = {}
            
            for tf in timeframes:
                tprint(f"📊 [PHASE1] Processing timeframe: {tf}")
                
                # Resample features to timeframe
                tf_features = self.variant_generator._resample_to_timeframe(features_df, tf)
                tf_price = self.variant_generator._resample_to_timeframe(price_data, tf) if price_data is not None else None
                tf_volume = self.variant_generator._resample_to_timeframe(volume, tf) if volume is not None else None
                
                if tf_features.empty:
                    tprint(f"⚠️ [PHASE1] No data for timeframe {tf}, skipping")
                    continue
                    
                # Generate variants
                tprint(f"🔄 [PHASE1] Generating variants for timeframe {tf}")
                variants = self.variant_generator.generate_all_variants(
                    tf_features, 
                    price_data=tf_price,
                    volume=tf_volume
                )
                
                # Add timeframe prefix
                for variant_type, df in variants.items():
                    prefixed_df = pd.DataFrame()
                    for col in df.columns:
                        prefixed_df[f"{tf}_{col}"] = df[col]
                    all_variants[f"{tf}_{variant_type}"] = prefixed_df
                tprint(f"✅ [PHASE1] Generated {len(variants)} variant types for timeframe {tf}")
            
            # Combine all variants
            if not all_variants:
                tprint("❌ No variants generated in Phase 1")
                return pd.DataFrame(), {}
                
            combined_variants = pd.concat(list(all_variants.values()), axis=1)
            tprint(f"📊 Phase 1: Generated {len(combined_variants.columns)} variant features")
            
            # Normalize variants
            normalized_variants = self.variant_generator.normalize_variants(
                {k: v for k, v in all_variants.items() if not v.empty}, 
                method='zscore'
            )
            
            # Combine normalized variants
            normalized_combined = pd.concat(list(normalized_variants.values()), axis=1)
            
            # Align with targets
            aligned_data = normalized_combined.join(targets.rename('target'), how='inner').dropna()
            if aligned_data.empty:
                tprint("❌ No aligned data after joining variants and targets")
                return pd.DataFrame(), {}
                
            features_aligned = aligned_data.drop(columns=['target'])
            targets_aligned = aligned_data['target']
            
            tprint(f"📊 Phase 1: Aligned data shape: {features_aligned.shape}")
            
            # SHAP scoring with shallow LGBM
            shap_config = SHAPScorerConfig(
                lgbm_params=self.phase_config.phase1_lgbm_params,
                n_folds=self.phase_config.phase1_n_folds,
                shap_weight=0.5,
                interaction_centrality_weight=0.3,
                stability_weight=0.2
            )
            
            shap_scorer = SHAPInteractionScorer(shap_config)
            shap_results = shap_scorer.score_features(features_aligned, targets_aligned)
            
            if not shap_results.get('success', False):
                tprint("❌ Phase 1 SHAP scoring failed")
                return pd.DataFrame(), {}
                
            # Select top 40% features
            n_features = len(features_aligned.columns)
            top_k = max(1, int(n_features * self.phase_config.phase1_selection_ratio))
            top_features = shap_scorer.get_top_features(shap_results, top_k, 'combined')
            
            selected_features = features_aligned[top_features]
            tprint(f"📊 Phase 1: Selected {len(selected_features.columns)} features (top {self.phase_config.phase1_selection_ratio*100}%)")
            
            # Store metadata
            phase1_metadata = {
                'total_variants_generated': len(combined_variants.columns),
                'selected_features_count': len(selected_features.columns),
                'selection_ratio': self.phase_config.phase1_selection_ratio,
                'shap_results': {
                    'feature_names': shap_results['feature_names'],
                    'shap_scores': shap_results['shap_scores'].tolist(),
                    'interaction_centrality': shap_results['interaction_centrality'].tolist(),
                    'stability_scores': shap_results['stability_scores'].tolist(),
                    'combined_scores': shap_results['combined_scores'].tolist()
                },
                'timeframes_processed': timeframes,
                'variant_types': list(all_variants.keys())
            }
            
            self.performance_stats['phase1_time'] = time.time() - phase_start_time
            tprint(f"✅ Phase 1 completed in {self.performance_stats['phase1_time']:.2f}s")
            
            return selected_features, phase1_metadata
            
        except Exception as e:
            tprint(f"❌ Phase 1 failed: {e}")
            return pd.DataFrame(), {'error': str(e)}

    def _execute_phase2(self, phase1_features: pd.DataFrame, targets: pd.Series) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute Phase 2: Middle refinement with deeper LGBM."""
        tprint("🔧 Starting Phase 2: Middle refinement")
        phase_start_time = time.time()
        
        try:
            # Align features and targets
            aligned_data = phase1_features.join(targets.rename('target'), how='inner').dropna()
            if aligned_data.empty:
                tprint("❌ No aligned data in Phase 2")
                return pd.DataFrame(), {}
                
            features_aligned = aligned_data.drop(columns=['target'])
            targets_aligned = aligned_data['target']
            
            tprint(f"📊 Phase 2: Input features: {features_aligned.shape}")
            
            # SHAP scoring with middle LGBM
            shap_config = SHAPScorerConfig(
                lgbm_params=self.phase_config.phase2_lgbm_params,
                n_folds=self.phase_config.phase2_n_folds,
                enable_top_k_filter=True,
                top_k_features=min(self.phase_config.phase2_top_k * 2, len(features_aligned.columns)),
                shap_weight=0.5,
                interaction_centrality_weight=0.3,
                stability_weight=0.2
            )
            
            shap_scorer = SHAPInteractionScorer(shap_config)
            shap_results = shap_scorer.score_features(features_aligned, targets_aligned)
            
            if not shap_results.get('success', False):
                tprint("❌ Phase 2 SHAP scoring failed")
                return pd.DataFrame(), {}
                
            # Select top 40 features
            top_features = shap_scorer.get_top_features(shap_results, self.phase_config.phase2_top_k, 'combined')
            selected_features = features_aligned[top_features]
            
            tprint(f"📊 Phase 2: Selected {len(selected_features.columns)} features")
            
            # Store metadata
            phase2_metadata = {
                'input_features_count': len(features_aligned.columns),
                'selected_features_count': len(selected_features.columns),
                'shap_results': {
                    'feature_names': shap_results['feature_names'],
                    'shap_scores': shap_results['shap_scores'].tolist(),
                    'interaction_centrality': shap_results['interaction_centrality'].tolist(),
                    'stability_scores': shap_results['stability_scores'].tolist(),
                    'combined_scores': shap_results['combined_scores'].tolist()
                },
                'interaction_centrality_pairs': shap_results.get('interaction_centrality_pairs', {})
            }
            
            self.performance_stats['phase2_time'] = time.time() - phase_start_time
            tprint(f"✅ Phase 2 completed in {self.performance_stats['phase2_time']:.2f}s")
            
            return selected_features, phase2_metadata
            
        except Exception as e:
            tprint(f"❌ Phase 2 failed: {e}")
            return pd.DataFrame(), {'error': str(e)}

    def _execute_phase3(self, phase2_features: pd.DataFrame, targets: pd.Series,
                       phase2_metadata: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute Phase 3: Deep interaction discovery."""
        tprint("🔧 Starting Phase 3: Deep interaction discovery")
        phase_start_time = time.time()
        
        try:
            # Align features and targets
            aligned_data = phase2_features.join(targets.rename('target'), how='inner').dropna()
            if aligned_data.empty:
                tprint("❌ No aligned data in Phase 3")
                return pd.DataFrame(), {}
                
            features_aligned = aligned_data.drop(columns=['target'])
            targets_aligned = aligned_data['target']
            
            tprint(f"📊 Phase 3: Input features: {features_aligned.shape}")
            
            # Get interaction centrality pairs from Phase 2
            centrality_pairs = phase2_metadata.get('interaction_centrality_pairs', {})
            
            if centrality_pairs:
                tprint(f"📊 Phase 3: Using {len(centrality_pairs)} interaction centrality pairs from Phase 2")
                
                # Generate interactions based on centrality
                interactions_df = self.interaction_generator.generate_interactions_from_centrality(
                    features_aligned,
                    centrality_pairs,
                    max_pairs=self.phase_config.phase3_top_interactions // 4  # 4 interactions per pair
                )
            else:
                tprint("📊 Phase 3: No centrality pairs, using top features for interaction generation")
                
                # Fallback: generate interactions from top features
                top_features = list(features_aligned.columns)
                interactions_df = self.interaction_generator.generate_interactions_from_top_features(
                    features_aligned,
                    top_features,
                    max_pairs=self.phase_config.phase3_top_interactions // 4
                )
            
            if interactions_df.empty:
                tprint("❌ No interactions generated in Phase 3")
                return pd.DataFrame(), {}
                
            tprint(f"📊 Phase 3: Generated {len(interactions_df.columns)} interactions")
            
            # Filter interactions by variance and correlation
            interactions_df = self.interaction_generator.filter_interactions_by_variance(interactions_df)
            interactions_df = self.interaction_generator.filter_interactions_by_correlation(interactions_df)
            
            tprint(f"📊 Phase 3: Filtered to {len(interactions_df.columns)} interactions")
            
            # SHAP scoring with deep LGBM
            shap_config = SHAPScorerConfig(
                lgbm_params=self.phase_config.phase3_lgbm_params,
                n_folds=self.phase_config.phase3_n_folds,
                use_shap_interactions=True,
                interaction_pairs_limit=25,
                shap_weight=0.6,
                interaction_centrality_weight=0.4,
                stability_weight=0.0  # Focus on SHAP and interactions in final phase
            )
            
            shap_scorer = SHAPInteractionScorer(shap_config)
            shap_results = shap_scorer.score_features(interactions_df, targets_aligned)
            
            if not shap_results.get('success', False):
                tprint("❌ Phase 3 SHAP scoring failed")
                return pd.DataFrame(), {}
                
            # Select top interactions
            top_interactions = shap_scorer.get_top_features(
                shap_results, 
                self.phase_config.phase3_top_interactions, 
                'combined'
            )
            
            selected_interactions = interactions_df[top_interactions]
            tprint(f"📊 Phase 3: Selected {len(selected_interactions.columns)} interactions")
            
            # Store metadata
            phase3_metadata = {
                'generated_interactions_count': len(interactions_df.columns),
                'selected_interactions_count': len(selected_interactions.columns),
                'shap_results': {
                    'feature_names': shap_results['feature_names'],
                    'shap_scores': shap_results['shap_scores'].tolist(),
                    'interaction_centrality': shap_results['interaction_centrality'].tolist(),
                    'stability_scores': shap_results['stability_scores'].tolist(),
                    'combined_scores': shap_results['combined_scores'].tolist()
                },
                'interaction_metadata': self.interaction_generator.get_interaction_metadata(),
                'interaction_summary': self.interaction_generator.get_interaction_summary()
            }
            
            self.performance_stats['phase3_time'] = time.time() - phase_start_time
            tprint(f"✅ Phase 3 completed in {self.performance_stats['phase3_time']:.2f}s")
            
            return selected_interactions, phase3_metadata
            
        except Exception as e:
            tprint(f"❌ Phase 3 failed: {e}")
            return pd.DataFrame(), {'error': str(e)}

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        tprint("🚀 [ANALYST] Starting three-phase LGBM+SHAP interaction generation pipeline")
        self.logger.info("🔧 Starting Analyst mode three-phase interaction generation")
        
        # Set up enhanced artifact manager with Analyst context
        symbol = training_input.get('symbol', 'ETHUSDT')
        exchange = training_input.get('exchange', 'binance')
        context = get_analyst_context(symbol, exchange)
        am = setup_enhanced_artifact_manager(**context)
        
        # Start memory monitoring
        self.m1_memory_optimizer.start_monitoring()
        
        try:
            # Extract training input parameters
            data = training_input.get('data')
            symbol = training_input.get('symbol', 'ETHUSDT')
            timeframe = training_input.get('timeframe', '15m')
            direction = training_input.get('direction', 'longs')
            intensity = training_input.get('intensity', 'blank')
            lookback_days = training_input.get('lookback_days')
            start_date = training_input.get('start_date')
            end_date = training_input.get('end_date')
            exchange = training_input.get('exchange', 'binance')
            custom_overrides = training_input.get('custom_overrides')
            
            tprint(f"📊 [ANALYST] Configuration: Symbol={symbol}, Timeframe={timeframe}, Direction={direction}, Intensity={intensity}")
            
            # Get artifact manager
            artifact_manager = get_pretraining_artifact_manager()
            
            # Load selected features
            tprint("📥 [ANALYST] Loading selected features from artifact manager")
            selected_df = artifact_manager.get_dataframe('feature_selection', ArtifactKeys.SELECTED_FEATURES)
            if selected_df is None or selected_df.empty:
                selected_df = artifact_manager.get_dataframe('feature_generation_feature_selection_step', ArtifactKeys.SELECTED_FEATURES)
            if selected_df is None or selected_df.empty:
                return InteractionGenerationResult(
                    success=False,
                    interaction_features=pd.DataFrame(),
                    interaction_metadata={},
                    generation_metrics={},
                    artifacts={},
                    error_message="Selected features not found. Run feature_selection before interaction_generation."
                )
            
            tprint(f"✅ [ANALYST] Loaded {len(selected_df.columns)} selected features with shape {selected_df.shape}")
            
            # Load targets
            tprint("📥 [ANALYST] Loading targets from labeling integration step")
            targets_series = None
            for step_name in ("feature_generation_labeling_integration_step", "labeling_integration"):
                series = artifact_manager.get_series(step_name, ArtifactKeys.TARGETS)
                if isinstance(series, pd.Series) and not series.empty:
                    targets_series = series.astype(float)
                    tprint(f"✅ [ANALYST] Loaded targets from {step_name}: {len(targets_series)} samples")
                    break
            
            if targets_series is None or targets_series.empty:
                return InteractionGenerationResult(
                    success=False,
                    interaction_features=pd.DataFrame(),
                    interaction_metadata={},
                    generation_metrics={},
                    artifacts={},
                    error_message="Targets from feature_generation_labeling_integration_step are required."
                )
            
            # Sample data by mode
            tprint(f"✂️ [ANALYST] Sampling data for {intensity} mode")
            sampled_features = self._sample_data_by_mode(selected_df, intensity)
            sampled_targets = self._sample_data_by_mode(targets_series, intensity)
            
            # Align features and targets
            tprint("🔍 [ANALYST] Aligning features and targets timestamps")
            aligned = sampled_features.join(sampled_targets.rename("target"), how="inner").dropna()
            if aligned.empty:
                tprint("❌ [ANALYST] No overlapping timestamps between features and targets")
                return InteractionGenerationResult(
                    success=False,
                    interaction_features=pd.DataFrame(),
                    interaction_metadata={},
                    generation_metrics={},
                    artifacts={},
                    error_message="No overlapping timestamps between features and targets."
                )
            
            features_df = aligned.drop(columns=['target'])
            targets = aligned['target']
            
            tprint(f"✅ [ANALYST] Final aligned data: features={features_df.shape}, targets={targets.shape}")
            
            # Load timeframes
            tprint("📊 [ANALYST] Loading optimized timeframes from period lookback optimization")
            timeframes = self._load_timeframes_from_optimization(artifact_manager)
            
            # Load price data and volume for variant generation
            tprint("📊 [ANALYST] Loading OHLCV data for variant generation")
            price_data = None
            volume = None
            try:
                # Try to load OHLCV data from artifacts
                ohlcv_data = artifact_manager.get_dataframe('feature_generation', 'ohlcv_data')
                if ohlcv_data is not None and not ohlcv_data.empty:
                    price_data = ohlcv_data[['open', 'high', 'low', 'close']]
                    volume = ohlcv_data['volume']
                    tprint(f"✅ [ANALYST] Loaded OHLCV data: {price_data.shape}")
            except Exception as e:
                tprint(f"⚠️ Could not load OHLCV data: {e}")
            
            # Execute three-phase pipeline
            tprint("🔄 [ANALYST] ========== PHASE 1: VARIANT GENERATION & SHALLOW LGBM SWEEP ==========")
            
            # Phase 1: Variant generation & shallow LGBM sweep
            phase1_features, phase1_metadata = self._execute_phase1(
                features_df, targets, price_data, volume, timeframes
            )
            tprint(f"✅ [ANALYST] Phase 1 completed: {len(phase1_features.columns)} features selected")
            
            if phase1_features.empty:
                return InteractionGenerationResult(
                    success=False,
                    interaction_features=pd.DataFrame(),
                    interaction_metadata={'phase1_error': phase1_metadata.get('error', 'Unknown error')},
                    generation_metrics={},
                    artifacts={},
                    error_message="Phase 1 failed"
                )
            
            # Phase 2: Middle refinement
            tprint("🔄 [ANALYST] ========== PHASE 2: MIDDLE REFINEMENT WITH DEEPER LGBM ==========")
            phase2_features, phase2_metadata = self._execute_phase2(phase1_features, targets)
            tprint(f"✅ [ANALYST] Phase 2 completed: {len(phase2_features.columns)} features selected")
            
            if phase2_features.empty:
                return InteractionGenerationResult(
                    success=False,
                    interaction_features=pd.DataFrame(),
                    interaction_metadata={
                        'phase1_metadata': phase1_metadata,
                        'phase2_error': phase2_metadata.get('error', 'Unknown error')
                    },
                    generation_metrics={},
                    artifacts={},
                    error_message="Phase 2 failed"
                )
            
            # Phase 3: Deep interaction discovery
            tprint("🔄 [ANALYST] ========== PHASE 3: DEEP INTERACTION DISCOVERY WITH FULL LGBM ==========")
            phase3_interactions, phase3_metadata = self._execute_phase3(phase2_features, targets, phase2_metadata)
            tprint(f"✅ [ANALYST] Phase 3 completed: {len(phase3_interactions.columns)} interaction features generated")
            
            if phase3_interactions.empty:
                return InteractionGenerationResult(
                    success=False,
                    interaction_features=pd.DataFrame(),
                    interaction_metadata={
                        'phase1_metadata': phase1_metadata,
                        'phase2_metadata': phase2_metadata,
                        'phase3_error': phase3_metadata.get('error', 'Unknown error')
                    },
                    generation_metrics={},
                    artifacts={},
                    error_message="Phase 3 failed"
                )
            
            # Apply final memory optimization
            tprint("🧹 [ANALYST] Applying final memory optimization to interaction features")
            final_interactions = self._optimize_dataframe_memory(phase3_interactions)
            
            # Store artifacts
            tprint("💾 [ANALYST] Storing interaction features in artifact manager")
            artifact_manager.store_enhanced(ArtifactKeys.INTERACTION_FEATURES, final_interactions, {
                'step': 'interaction_generation_analyst',
                'shape': final_interactions.shape,
                'created_at': datetime.now().isoformat(),
                'direction': direction,
                'intensity': intensity
            })
            
            # Combine all metadata
            combined_metadata = {
                'phase1_metadata': phase1_metadata,
                'phase2_metadata': phase2_metadata,
                'phase3_metadata': phase3_metadata,
                'pipeline_config': {
                    'direction': direction,
                    'intensity': intensity,
                    'timeframes': timeframes,
                    'sample_sizes': {
                        'light': self.phase_config.light_mode_sample_size,
                        'blank': self.phase_config.blank_mode_sample_size,
                        'full': self.phase_config.full_mode_sample_size
                    }
                }
            }
            
            tprint("💾 [ANALYST] Storing interaction metadata in artifact manager")
            artifact_manager.store_enhanced(ArtifactKeys.INTERACTION_METADATA, combined_metadata, {
                'step': 'interaction_generation_analyst',
                'created_at': datetime.now().isoformat()
            })
            
            # Calculate generation metrics
            total_time = time.time() - start_time
            generation_metrics = {
                'total_processing_time': total_time,
                'phase_times': {
                    'phase1': self.performance_stats['phase1_time'],
                    'phase2': self.performance_stats['phase2_time'],
                    'phase3': self.performance_stats['phase3_time']
                },
                'm1_optimizations': self.performance_stats.copy(),
                'final_interactions_count': len(final_interactions.columns),
                'success': True
            }
            
            tprint("💾 [ANALYST] Storing generation metrics in artifact manager")
            artifact_manager.store_enhanced(ArtifactKeys.INTERACTION_GENERATION_METRICS, generation_metrics, {
                'step': 'interaction_generation_analyst',
                'created_at': datetime.now().isoformat()
            })
            
            # Create result
            result = InteractionGenerationResult(
                success=True,
                interaction_features=final_interactions,
                interaction_metadata=combined_metadata,
                generation_metrics=generation_metrics,
                artifacts={
                    'interaction_features': final_interactions,
                    'interaction_metadata': combined_metadata,
                    'generation_metrics': generation_metrics
                },
                error_message=None
            )
            
            # Final performance summary
            tprint(f"⏱️ [ANALYST] Total processing time: {total_time:.2f}s")
            tprint(f"📊 [ANALYST] Final interactions: {len(final_interactions.columns)}")
            tprint(f"🧠 [ANALYST] Memory optimizations applied: {self.performance_stats['memory_optimizations_applied']}")
            
            # Generate comprehensive report
            tprint("📋 [ANALYST] Generating comprehensive interaction report")
            try:
                report = self._generate_comprehensive_report(
                    final_interactions, combined_metadata, symbol, timeframe, features_df
                )
                self._store_comprehensive_report(report, symbol, timeframe, direction, intensity)
                tprint("✅ [ANALYST] Comprehensive report generated and stored successfully")
            except Exception as e:
                tprint(f"⚠️ [ANALYST] Report generation failed: {e}")
                pass
            
            tprint("🎉 [ANALYST] Three-phase interaction generation completed successfully")
            return result
            
        except Exception as e:
            tprint(f"❌ [ANALYST] Interaction generation failed: {e}")
            self.logger.error(f"Analyst interaction generation failed: {e}")
            
            return InteractionGenerationResult(
                success=False,
                interaction_features=pd.DataFrame(),
                interaction_metadata={},
                generation_metrics={'error': str(e), 'processing_time_seconds': time.time() - start_time},
                artifacts={},
                error_message=str(e)
            )
        finally:
            # Cleanup and stop memory monitoring
            tprint("🧹 Cleaning up M1 optimizations...")
            try:
                self.m1_memory_optimizer.stop_monitoring()
                self.m1_memory_optimizer.force_garbage_collection()
                tprint("✅ Cleanup completed")
            except Exception as cleanup_error:
                tprint(f"⚠️ Cleanup warning: {cleanup_error}")

    # Minimal hooks for ModularComponent
    def _initialize_resources(self) -> bool:
        try:
            self.set_state('initialized', True)
            return True
        except Exception:
            return False

    def _cleanup_resources(self) -> None:
        self.set_state('initialized', False)

    def _process_data(self, data: Any, **kwargs) -> Any:
        return data

    def _get_validation_rules(self) -> Dict[str, Any]:
        return {
            'data_types': ['pandas.DataFrame'],
            'required_attributes': ['open', 'high', 'low', 'close', 'volume'],
            'min_size': 100
        }

    def _validate_component_specific(self, data: Any) -> Dict[str, Any]:
        errors, warnings, metadata = [], [], {}
        if isinstance(data, pd.DataFrame):
            missing = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c not in data.columns]
            if missing:
                errors.append(f"Missing required columns: {missing}")
            metadata['shape'] = data.shape
        return {'errors': errors, 'warnings': warnings, 'metadata': metadata}

    def _generate_comprehensive_report(self, 
                                     interactions: pd.DataFrame, 
                                     metadata: Dict[str, Any], 
                                     symbol: str, 
                                     timeframe: str,
                                     original_features: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive interaction report with comparative tables and metadata."""
        from datetime import datetime as _dt
        import numpy as _np
        import pandas as _pd

        n_rows = int(len(interactions)) if isinstance(interactions, _pd.DataFrame) else 0
        n_cols = int(len(interactions.columns)) if isinstance(interactions, _pd.DataFrame) else 0

        # Extract phase information
        phase1_meta = metadata.get('phase1_metadata', {})
        phase2_meta = metadata.get('phase2_metadata', {})
        phase3_meta = metadata.get('phase3_metadata', {})

        # Comparative table: top features per timeframe/variant
        comparative_table = []
        if phase1_meta.get('shap_results'):
            shap_results = phase1_meta['shap_results']
            for i, feature_name in enumerate(shap_results.get('feature_names', [])):
                comparative_table.append({
                    'feature': feature_name,
                    'phase1_shap_score': shap_results['shap_scores'][i] if i < len(shap_results['shap_scores']) else 0.0,
                    'phase1_centrality': shap_results['interaction_centrality'][i] if i < len(shap_results['interaction_centrality']) else 0.0,
                    'phase1_stability': shap_results['stability_scores'][i] if i < len(shap_results['stability_scores']) else 0.0,
                    'phase1_combined': shap_results['combined_scores'][i] if i < len(shap_results['combined_scores']) else 0.0,
                })

        # Interaction table: top pairs with strengths and parent importances
        interaction_table = []
        if phase3_meta.get('shap_results'):
            shap_results = phase3_meta['shap_results']
            for i, feature_name in enumerate(shap_results.get('feature_names', [])):
                # Get interaction metadata
                interaction_meta = phase3_meta.get('interaction_metadata', {})
                interaction_info = interaction_meta.get(feature_name, {})
                
                interaction_table.append({
                    'interaction': feature_name,
                    'feature1': interaction_info.get('feature1', 'Unknown'),
                    'feature2': interaction_info.get('feature2', 'Unknown'),
                    'interaction_type': interaction_info.get('interaction_type', 'Unknown'),
                    'phase3_shap_score': shap_results['shap_scores'][i] if i < len(shap_results['shap_scores']) else 0.0,
                    'phase3_centrality': shap_results['interaction_centrality'][i] if i < len(shap_results['interaction_centrality']) else 0.0,
                    'phase3_combined': shap_results['combined_scores'][i] if i < len(shap_results['combined_scores']) else 0.0,
                })

        # Diversity/cluster summary
        diversity_summary = {
            'phase1_total_variants': phase1_meta.get('total_variants_generated', 0),
            'phase1_selected_features': phase1_meta.get('selected_features_count', 0),
            'phase2_selected_features': phase2_meta.get('selected_features_count', 0),
            'phase3_final_interactions': phase3_meta.get('selected_interactions_count', 0),
            'timeframes_processed': phase1_meta.get('timeframes_processed', []),
            'variant_types': phase1_meta.get('variant_types', [])
        }

        # Performance metrics
        performance_metrics = {
            'phase1_time': self.performance_stats.get('phase1_time', 0.0),
            'phase2_time': self.performance_stats.get('phase2_time', 0.0),
            'phase3_time': self.performance_stats.get('phase3_time', 0.0),
            'total_time': self.performance_stats.get('total_processing_time', 0.0),
            'memory_optimizations': self.performance_stats.get('memory_optimizations_applied', 0),
            'gpu_accelerations': self.performance_stats.get('gpu_accelerations_used', 0),
            'vectorbt_optimizations': self.performance_stats.get('vectorbt_optimizations_used', 0)
        }

        # Final manifest with variant metadata
        final_manifest = {
            'total_interactions': len(interactions.columns),
            'interaction_types': {},
            'parent_features': set(),
            'timeframe_distribution': {},
            'variant_type_distribution': {}
        }

        # Analyze interaction metadata
        if phase3_meta.get('interaction_metadata'):
            for interaction_name, meta in phase3_meta['interaction_metadata'].items():
                interaction_type = meta.get('interaction_type', 'unknown')
                final_manifest['interaction_types'][interaction_type] = final_manifest['interaction_types'].get(interaction_type, 0) + 1
                
                parent_features = meta.get('parent_features', [])
                for parent in parent_features:
                    final_manifest['parent_features'].add(parent)
                    
                    # Extract timeframe and variant info from parent feature names
                    if '_' in parent:
                        parts = parent.split('_')
                        if len(parts) >= 2:
                            timeframe = parts[0]
                            final_manifest['timeframe_distribution'][timeframe] = final_manifest['timeframe_distribution'].get(timeframe, 0) + 1
                            
                            # Identify variant type
                            if 'vol_norm' in parent:
                                variant_type = 'vol_normalized'
                            elif 'vwap' in parent:
                                variant_type = 'vwap_based'
                            elif 'combined' in parent:
                                variant_type = 'combined'
                            else:
                                variant_type = 'raw'
                            final_manifest['variant_type_distribution'][variant_type] = final_manifest['variant_type_distribution'].get(variant_type, 0) + 1

        # Convert sets to lists for JSON serialization
        final_manifest['parent_features'] = list(final_manifest['parent_features'])

        return {
            'title': 'Comprehensive Interaction Generation Report - Analyst Mode',
            'timestamp': _dt.now().isoformat(),
            'configuration': {
                'symbol': symbol,
                'timeframe': timeframe,
                'mode': 'analyst',
                'pipeline_config': metadata.get('pipeline_config', {})
            },
            'summary': {
                'rows': n_rows,
                'columns': n_cols,
                'memory_mb': float(interactions.memory_usage(deep=True).sum() / (1024**2)) if isinstance(interactions, _pd.DataFrame) else 0.0
            },
            'phase_summary': {
                'phase1': {
                    'total_variants_generated': phase1_meta.get('total_variants_generated', 0),
                    'selected_features_count': phase1_meta.get('selected_features_count', 0),
                    'selection_ratio': phase1_meta.get('selection_ratio', 0.0),
                    'timeframes_processed': phase1_meta.get('timeframes_processed', [])
                },
                'phase2': {
                    'input_features_count': phase2_meta.get('input_features_count', 0),
                    'selected_features_count': phase2_meta.get('selected_features_count', 0)
                },
                'phase3': {
                    'generated_interactions_count': phase3_meta.get('generated_interactions_count', 0),
                    'selected_interactions_count': phase3_meta.get('selected_interactions_count', 0)
                }
            },
            'comparative_table': comparative_table[:50],  # Top 50
            'interaction_table': interaction_table[:50],  # Top 50
            'diversity_summary': diversity_summary,
            'performance_metrics': performance_metrics,
            'final_manifest': final_manifest
        }

    def _store_comprehensive_report(self, report: Dict[str, Any], symbol: str, timeframe: str, direction: str, intensity: str) -> None:
        """Store comprehensive report as markdown and JSON."""
        from datetime import datetime as _dt
        from pathlib import Path as _Path
        import json as _json
        
        out_dir = _Path('outcomes')
        out_dir.mkdir(exist_ok=True)
        ts = _dt.now().strftime('%Y%m%d_%H%M%S')
        
        # Generate markdown report
        md = self._format_comprehensive_markdown(report)
        md_path = out_dir / f"interaction_generation_comprehensive_report_{symbol}_{timeframe}_{direction}_{intensity}_{ts}.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md)
        
        # Store JSON report
        json_path = out_dir / f"interaction_generation_comprehensive_report_{symbol}_{timeframe}_{direction}_{intensity}_{ts}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            _json.dump(report, f, indent=2, ensure_ascii=False)

    def _format_comprehensive_markdown(self, report: Dict[str, Any]) -> str:
        """Format comprehensive report as markdown."""
        md = f"# {report['title']}\n\n"
        md += f"**Generated:** {report['timestamp']}\n\n"
        
        # Configuration
        cfg = report.get('configuration', {})
        md += "## 📌 Configuration\n\n"
        md += f"- Symbol: {cfg.get('symbol', '?')}\n"
        md += f"- Timeframe: {cfg.get('timeframe', '?')}\n"
        md += f"- Mode: {cfg.get('mode', '?')}\n"
        
        # Summary
        summ = report.get('summary', {})
        md += "\n## 📊 Summary\n\n"
        md += f"- Rows: {summ.get('rows', 0):,}\n"
        md += f"- Interactions: {summ.get('columns', 0)}\n"
        md += f"- Memory: {summ.get('memory_mb', 0.0):.2f} MB\n"
        
        # Phase Summary
        md += "\n## 🔄 Phase Summary\n\n"
        phase_summary = report.get('phase_summary', {})
        
        for phase_name, phase_data in phase_summary.items():
            md += f"### {phase_name.upper()}\n\n"
            for key, value in phase_data.items():
                if isinstance(value, list):
                    md += f"- {key}: {', '.join(map(str, value))}\n"
                else:
                    md += f"- {key}: {value}\n"
            md += "\n"
        
        # Performance Metrics
        md += "\n## ⏱️ Performance Metrics\n\n"
        perf = report.get('performance_metrics', {})
        md += f"- Phase 1 Time: {perf.get('phase1_time', 0.0):.2f}s\n"
        md += f"- Phase 2 Time: {perf.get('phase2_time', 0.0):.2f}s\n"
        md += f"- Phase 3 Time: {perf.get('phase3_time', 0.0):.2f}s\n"
        md += f"- Total Time: {perf.get('total_time', 0.0):.2f}s\n"
        md += f"- Memory Optimizations: {perf.get('memory_optimizations', 0)}\n"
        md += f"- GPU Accelerations: {perf.get('gpu_accelerations', 0)}\n"
        md += f"- VectorBT Optimizations: {perf.get('vectorbt_optimizations', 0)}\n"
        
        # Comparative Table
        md += "\n## 🔝 Top Features by Phase 1 Scores\n\n"
        comparative_table = report.get('comparative_table', [])
        if comparative_table:
            md += "| Feature | Phase 1 SHAP | Centrality | Stability | Combined |\n"
            md += "|---|---:|---:|---:|---:|\n"
            for row in comparative_table[:20]:  # Top 20
                md += f"| {row['feature']} | {row['phase1_shap_score']:.4f} | {row['phase1_centrality']:.4f} | {row['phase1_stability']:.4f} | {row['phase1_combined']:.4f} |\n"
        else:
            md += "_No comparative data available._\n"
        
        # Interaction Table
        md += "\n## 🔗 Top Interactions by Phase 3 Scores\n\n"
        interaction_table = report.get('interaction_table', [])
        if interaction_table:
            md += "| Interaction | Feature 1 | Feature 2 | Type | Phase 3 SHAP | Combined |\n"
            md += "|---|---|---|---|---:|---:|\n"
            for row in interaction_table[:20]:  # Top 20
                md += f"| {row['interaction']} | {row['feature1']} | {row['feature2']} | {row['interaction_type']} | {row['phase3_shap_score']:.4f} | {row['phase3_combined']:.4f} |\n"
        else:
            md += "_No interaction data available._\n"
        
        # Final Manifest
        md += "\n## 📋 Final Manifest\n\n"
        manifest = report.get('final_manifest', {})
        md += f"- Total Interactions: {manifest.get('total_interactions', 0)}\n"
        md += f"- Unique Parent Features: {len(manifest.get('parent_features', []))}\n"
        
        # Interaction Types
        md += "\n### Interaction Types\n\n"
        interaction_types = manifest.get('interaction_types', {})
        for int_type, count in interaction_types.items():
            md += f"- {int_type}: {count}\n"
        
        # Timeframe Distribution
        md += "\n### Timeframe Distribution\n\n"
        timeframe_dist = manifest.get('timeframe_distribution', {})
        for tf, count in timeframe_dist.items():
            md += f"- {tf}: {count}\n"
        
        # Variant Type Distribution
        md += "\n### Variant Type Distribution\n\n"
        variant_dist = manifest.get('variant_type_distribution', {})
        for variant, count in variant_dist.items():
            md += f"- {variant}: {count}\n"
        
        return md


# Handler for ares_launcher/sub_pipeline integration
async def handle_feature_generation_interaction_generation_step_analyst(
    symbol: str = "ETHUSDT",
    timeframe: str = "15m",
    direction: str = "longs",
    intensity: str = "blank",
    lookback_days: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    exchange: str = "binance",
    custom_overrides: Optional[Dict[str, Any]] = None,
    data: Optional[pd.DataFrame] = None,
    **kwargs
) -> InteractionGenerationResult:
    """Execute Analyst mode three-phase interaction generation."""
    start_time = time.time()
    tprint("🔧 Starting Analyst mode handle_feature_generation_interaction_generation_step_analyst")
    
    # Initialize step
    step = FeatureGenerationInteractionGenerationStepAnalyst()
    
    # Prepare training input
    training_input = {
        'symbol': symbol,
        'timeframe': timeframe,
        'direction': direction,
        'intensity': intensity,
        'lookback_days': lookback_days,
        'start_date': start_date,
        'end_date': end_date,
        'exchange': exchange,
        'custom_overrides': custom_overrides,
        'data': data
    }
    
    # Execute step
    result = await step.execute(training_input, {})
    
    tprint(f"✅ Analyst handler completed in {time.time() - start_time:.2f}s")
    return result
