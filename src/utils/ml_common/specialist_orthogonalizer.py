"""
Specialist Orthogonalizer - Performance Optimized

Forces each specialist to explain what others miss through orthogonal target generation.
XGB Macro serves as anchor, with AUC-weighted ensemble optimization.
14 specialists total with 2-core parallel processing optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import psutil
from functools import lru_cache
from pathlib import Path
import pickle

from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error


# Complete specialist categories (14 total)
SPECIALIST_CATEGORIES = {
    # Primary Anchor
    'xgb_macro': {
        'patterns': ['macro_trend_', 'xgb_macro_', 'regime_macro_'],
        'importance': 0.25,  # Anchor - highest importance
        'description': 'XGB Macro Trend - Primary trend anchor'
    },
    
    # Core Specialists
    'risk': {
        'patterns': ['risk_score', 'risk_regime_', 'risk_pred_'],
        'importance': 0.15,
        'description': 'ML Risk - Market risk and volatility assessment'
    },
    'liquidity': {
        'patterns': ['liquidity_regime_', 'liquidity_score'],
        'importance': 0.12,
        'description': 'Liquidity - Market depth and liquidity analysis'
    },
    'path': {
        'patterns': ['path_', 'path_risk_', 'path_quality_'],
        'importance': 0.10,
        'description': 'Path - Price path quality and risk assessment'
    },
    
    # Momentum & Trend Specialists
    'momentum': {
        'patterns': ['momentum_persistence_', 'momentum_'],
        'importance': 0.08,
        'description': 'Momentum - Trend persistence and momentum signals'
    },
    'xgb_meso': {
        'patterns': ['meso_trend_', 'xgb_meso_', 'regime_meso_'],
        'importance': 0.07,
        'description': 'XGB Meso Trend - Medium-term trend analysis'
    },
    
    # Volume & Flow Specialists
    'volume': {
        'patterns': ['vol_force_'],
        'importance': 0.05,
        'description': 'Volume - Volume force and flow analysis'
    },
    'microstructure': {
        'patterns': ['micro_', 'microstructure_', 'order_flow_', 'spread_'],
        'importance': 0.04,
        'description': 'Microstructure - Market microstructure analysis'
    },
    
    # Pattern & Technical Specialists
    'candlestick': {
        'patterns': ['candlestick_', 'candle_', 'doji_', 'hammer_', 'engulfing_'],
        'importance': 0.04,
        'description': 'Candlestick - Candlestick pattern analysis'
    },
    'spectral': {
        'patterns': ['spectral_', 'frequency_', 'fft_', 'wavelet_'],
        'importance': 0.03,
        'description': 'Spectral - Frequency domain analysis'
    },
    
    # Reversion & Contrarian Specialists
    'reversion': {
        'patterns': ['mr_', 'mean_reversion_'],
        'importance': 0.03,
        'description': 'Reversion - Mean reversion and contrarian signals'
    },
    'volatility': {
        'patterns': ['volatility_burst_', 'vol_spike_', 'vol_regime_'],
        'importance': 0.02,
        'description': 'Volatility - Volatility burst and regime analysis'
    },
    
    # Smart Money & Structure Specialists
    'smc': {
        'patterns': ['smc_', 'smart_money_', 'market_structure_'],
        'importance': 0.01,
        'description': 'SMC - Smart Money Concepts'
    }
}


class OptimizedSpecialistOrthogonalizer:
    """Performance-optimized orthogonalizer with 2-core parallel processing"""
    
    # Global flag for optimization components availability
    _optimization_components_available = False
    
    def __init__(self, anchor_specialist: str = 'xgb_macro', max_workers: int = 2,
                 enable_cache: bool = True, enable_feature_optimization: bool = True,
                 enable_orthogonal_hpo: bool = False, enable_conservative_pruning: bool = False,
                 enable_target_denoising: bool = False,
                 cache_dir: Path = None):
        self.anchor_specialist = anchor_specialist
        self.max_workers = max_workers
        self.specialist_categories = SPECIALIST_CATEGORIES
        self.feature_mappings = self._create_feature_mappings()
        self.lgbm_params = self._get_optimized_lgbm_params()
        
        # Initialize optimization components
        self.enable_cache = enable_cache
        self.enable_feature_optimization = enable_feature_optimization
        self.enable_orthogonal_hpo = enable_orthogonal_hpo
        self.enable_conservative_pruning = enable_conservative_pruning
        
        if enable_cache and self._optimization_components_available and SpecialistModelCache:
            self.cache = SpecialistModelCache(cache_dir=cache_dir)
        else:
            self.cache = None
        
        if enable_feature_optimization and self._optimization_components_available and FeatureOptimizer:
            self.feature_optimizer = FeatureOptimizer()
        else:
            self.feature_optimizer = None
        
        if enable_orthogonal_hpo and self._optimization_components_available and OrthogonalizationAwareHPO:
            hpo_config = HPOConfig(n_trials=30, timeout=1800)  # Narrow settings
            self.orthogonal_hpo = OrthogonalizationAwareHPO(hpo_config)
        else:
            self.orthogonal_hpo = None
        
        if enable_conservative_pruning and self._optimization_components_available and ConservativeEnsemblePruner:
            pruning_config = PruningConfig(min_ensemble_size=8, max_ensemble_size=12)
            self.pruner = ConservativeEnsemblePruner(pruning_config)
        else:
            self.pruner = None
        
        # Initialize target denoiser
        self.enable_target_denoising = enable_target_denoising
        if enable_target_denoising and self._optimization_components_available and TargetDenoiser:
            denoising_config = DenoisingConfig(method='kalman')  # Default fast method
            self.target_denoiser = TargetDenoiser(denoising_config)
        else:
            self.target_denoiser = None
        
    def _create_feature_mappings(self) -> Dict[str, List[str]]:
        """Create pattern-to-specialist mappings for feature extraction"""
        mappings = {}
        for specialist, config in self.specialist_categories.items():
            mappings[specialist] = config['patterns']
        return mappings
    
    def _get_optimized_lgbm_params(self) -> Dict[str, Any]:
        """Get LGBM parameters optimized for 2-core processing"""
        return {
            # Core parameters
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            
            # GOSS for speed (2-core friendly)
            'bagging_freq': 5,
            'bagging_fraction': 0.7,      # Reduced for 2-core
            'top_rate': 0.15,              # Reduced top rate
            'other_rate': 0.05,            # Reduced other rate
            
            # Tree structure (conservative for 2-core)
            'num_leaves': 15,              # Reduced from 31
            'max_depth': 5,                # Reduced from 7
            'min_child_samples': 50,        # Increased for stability
            'min_child_weight': 0.01,      # Increased
            
            # Feature optimization
            'feature_fraction': 0.6,        # Reduced for speed
            'max_bin': 127,                 # Reduced from 255
            'bin_construct_sample_cnt': 20000, # Reduced
            
            # Training optimization
            'learning_rate': 0.15,          # Increased for faster convergence
            'n_estimators': 80,              # Reduced from 100
            'early_stopping_rounds': 15,    # Reduced
            
            # Regularization
            'lambda_l1': 0.2,               # Increased for stability
            'lambda_l2': 0.2,               # Increased
            'min_split_gain': 0.15,         # Increased
            
            # Hardware optimization
            'n_jobs': 1,                    # Single thread per model
            'verbose': -1,                  # Silent
            'seed': 42
        }
    
    def extract_specialist_features(self, df: pd.DataFrame, specialist: str) -> pd.DataFrame:
        """Extract features for a specific specialist"""
        if specialist not in self.feature_mappings:
            raise ValueError(f"Unknown specialist: {specialist}")
        
        patterns = self.feature_mappings[specialist]
        features = []
        
        for pattern in patterns:
            matching_cols = [col for col in df.columns if col.startswith(pattern)]
            features.extend(matching_cols)
        
        if not features:
            tprint_warning(f"⚠️ No features found for specialist '{specialist}'")
            return pd.DataFrame(index=df.index)
        
        return df[features].copy()
    
    def calculate_specialist_auc(
        self, 
        features: pd.DataFrame, 
        target: pd.Series, 
        sample_weights: Optional[pd.Series] = None
    ) -> float:
        """Calculate AUC for specialist features"""
        try:
            # Handle NaN values
            X_clean = features.fillna(0)
            y_clean = target.reindex(X_clean.index).fillna(0)
            
            if len(X_clean) < 50:  # Need sufficient samples
                return 0.5
            
            # Simple logistic regression for AUC calculation
            model = LogisticRegression(random_state=42, max_iter=1000)
            
            weights_clean = None
            if sample_weights is not None:
                weights_clean = sample_weights.reindex(X_clean.index).fillna(1.0)
                model.fit(X_clean, y_clean, sample_weight=weights_clean)
            else:
                model.fit(X_clean, y_clean)
            
            predictions = model.predict_proba(X_clean)[:, 1]
            auc = roc_auc_score(y_clean, predictions, sample_weight=weights_clean)
            
            return max(0.5, min(1.0, auc))  # Clip to [0.5, 1.0]
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to calculate AUC: {e}")
            return 0.5
    
    def calculate_auc_weights(self, specialist_performance: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate AUC-based weights for specialists"""
        auc_scores = {}
        
        for specialist, performance in specialist_performance.items():
            auc = performance.get('auc', 0.5)
            # Normalize AUC to [0, 1] range (0.5 = random, 1.0 = perfect)
            normalized_auc = max(0, (auc - 0.5) * 2)
            auc_scores[specialist] = normalized_auc
        
        # Apply softmax for weight distribution
        exp_scores = {spec: np.exp(score * 3) for spec, score in auc_scores.items()}
        total_exp = sum(exp_scores.values())
        
        if total_exp > 0 and len(auc_scores) > 0:
            auc_weights = {spec: exp/total_exp for spec, exp in exp_scores.items()}
        else:
            # Fallback to equal weights
            if len(auc_scores) > 0:
                equal_weight = 1.0 / len(auc_scores)
                auc_weights = {spec: equal_weight for spec in auc_scores.keys()}
            else:
                auc_weights = {}
        
        return auc_weights
    
    def generate_auc_weighted_orthogonal_targets(
        self, 
        specialist_df: pd.DataFrame, 
        target_series: pd.Series, 
        sample_weights: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Generate orthogonal targets with AUC-based weighting"""
        
        tprint_info("🔄 Generating AUC-weighted orthogonal targets for specialist categories...")
        
        # First pass: Calculate baseline AUC for each specialist
        baseline_performance = {}
        available_specialists = []
        
        for specialist in self.specialist_categories.keys():
            features = self.extract_specialist_features(specialist_df, specialist)
            if not features.empty:
                auc = self.calculate_specialist_auc(features, target_series, sample_weights)
                baseline_performance[specialist] = {'auc': auc}
                available_specialists.append(specialist)
        
        tprint_info(f"  Found {len(available_specialists)} available specialists")
        for spec, perf in baseline_performance.items():
            tprint_info(f"    {spec}: AUC = {perf['auc']:.4f}")
        
        # Calculate AUC-based weights
        auc_weights = self.calculate_auc_weights(baseline_performance)
        
        tprint_info("  AUC-based weights:")
        for spec, weight in auc_weights.items():
            tprint_info(f"    {spec}: {weight:.3f}")
        
        # Generate orthogonal targets using AUC weights
        if len(available_specialists) > 0:
            orthogonal_targets = pd.DataFrame(index=specialist_df.index)
        else:
            orthogonal_targets = pd.DataFrame()
        
        for specialist in available_specialists:
            tprint_info(f"  Processing {specialist} specialist...")
            
            # Get features for this specialist
            specialist_features = self.extract_specialist_features(specialist_df, specialist)
            
            if specialist_features.empty:
                orthogonal_targets[f"{specialist}_orthogonal"] = target_series
                continue
            
            # Get features from all OTHER specialists
            other_specialists = [s for s in available_specialists if s != specialist]
            other_features_list = []
            
            for other_spec in other_specialists:
                other_features = self.extract_specialist_features(specialist_df, other_spec)
                if not other_features.empty:
                    other_features_list.append(other_features)
            
            if other_features_list:
                # Combine all other specialist features
                other_features_combined = pd.concat(other_features_list, axis=1)
                
                # Handle NaN values
                other_features_clean = self._optimize_memory_usage(other_features_combined)
                target_clean = target_series.reindex(other_features_clean.index).fillna(0)
                
                # Train model on other specialists to predict target
                try:
                    # Use LogisticRegression for binary target
                    if sample_weights is not None:
                        weights_clean = sample_weights.reindex(other_features_clean.index).fillna(1.0)
                        model = LogisticRegression(random_state=42, max_iter=1000)
                        model.fit(other_features_clean, target_clean, sample_weight=weights_clean)
                    else:
                        model = LogisticRegression(random_state=42, max_iter=1000)
                        model.fit(other_features_clean, target_clean)
                    
                    # Predict what other specialists can explain
                    other_prediction = model.predict_proba(other_features_clean)[:, 1]
                    
                    # Orthogonal target = true target - what others can explain
                    orthogonal_target = target_clean - other_prediction
                    
                    # Clip to reasonable range
                    orthogonal_target = np.clip(orthogonal_target, -1, 1)
                    
                    orthogonal_targets[f"{specialist}_orthogonal"] = orthogonal_target
                    
                    tprint_success(f"    ✓ Generated orthogonal target for {specialist}")
                    
                except Exception as e:
                    tprint_warning(f"    ⚠️ Failed to generate orthogonal target for {specialist}: {e}")
                    orthogonal_targets[f"{specialist}_orthogonal"] = target_series
            else:
                # No other specialists available, use original target
                orthogonal_targets[f"{specialist}_orthogonal"] = target_series
                tprint_info(f"    ✓ No other specialists, using original target for {specialist}")
        
        tprint_success(f"✅ Generated orthogonal targets for {len(available_specialists)} specialists")
        return orthogonal_targets, auc_weights
    
    def _optimize_memory_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize memory usage for large datasets"""
        
        if df.empty or len(df.columns) == 0:
            return df
        
        # Create a copy to avoid modifying original
        optimized_df = df.copy()
        
        # Downcast numeric columns
        for col in optimized_df.select_dtypes(include=['int64']).columns:
            try:
                optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='integer')
            except (TypeError, ValueError):
                # Skip if conversion fails
                pass
        
        for col in optimized_df.select_dtypes(include=['float64']).columns:
            try:
                optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
            except (TypeError, ValueError):
                # Skip if conversion fails
                pass
        
        # Convert object columns to category if low cardinality
        for col in optimized_df.select_dtypes(include=['object']).columns:
            if optimized_df[col].nunique() < 50:  # Lower threshold for 2-core
                try:
                    optimized_df[col] = optimized_df[col].astype('category')
                except (TypeError, ValueError):
                    # Skip if conversion fails
                    pass
        
        return optimized_df
    
    def _create_specialist_batches(self, specialists: List[str]) -> List[List[str]]:
        """Create balanced batches for 2-core processing"""
        
        # Group specialists by computational complexity
        light_specialists = ['smc', 'reversion', 'volatility', 'spectral']  # Fast
        medium_specialists = ['volume', 'microstructure', 'candlestick']     # Medium
        heavy_specialists = ['path', 'momentum', 'xgb_meso']                  # Heavy
        very_heavy_specialists = ['risk', 'liquidity', 'xgb_macro']          # Very heavy
        
        # Filter to available specialists
        available_light = [s for s in light_specialists if s in specialists]
        available_medium = [s for s in medium_specialists if s in specialists]
        available_heavy = [s for s in heavy_specialists if s in specialists]
        available_very_heavy = [s for s in very_heavy_specialists if s in specialists]
        
        # Create balanced batches
        batches = []
        
        # Batch 1: Very heavy + 1 light
        batch1 = available_very_heavy[:3] + available_light[:1]
        if batch1:
            batches.append(batch1)
        
        # Batch 2: Remaining very heavy + remaining light
        remaining_very_heavy = available_very_heavy[3:]
        remaining_light = available_light[1:]
        batch2 = remaining_very_heavy[:2] + remaining_light[:2]
        if batch2:
            batches.append(batch2)
        
        # Batch 3: Heavy + medium
        batch3 = available_heavy[:3] + available_medium[:1]
        if batch3:
            batches.append(batch3)
        
        # Batch 4: Remaining specialists
        remaining_heavy = available_heavy[3:]
        remaining_medium = available_medium[1:]
        remaining_light = available_light[3:]
        batch4 = remaining_heavy + remaining_medium + remaining_light
        if batch4:
            batches.append(batch4)
        
        return batches
    
    def _train_single_specialist_lgbm(
        self, 
        specialist_df: pd.DataFrame, 
        orthogonal_target: pd.Series, 
        sample_weights: Optional[pd.Series],
        specialist: str
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Train single specialist LGBM with optimizations"""
        
        start_time = time.time()
        
        # Extract features
        features = self.extract_specialist_features(specialist_df, specialist)
        if features.empty:
            return np.zeros(len(orthogonal_target)), {'auc': 0.5, 'error': 'No features'}
        
        # Optimize memory
        features = self._optimize_memory_usage(features)
        
        # Prepare data
        X = features.fillna(0)
        y = orthogonal_target.reindex(X.index).fillna(0)
        weights = sample_weights.reindex(X.index).fillna(1.0) if sample_weights is not None else None
        
        if len(X) < 50:  # Need sufficient samples
            return np.zeros(len(X)), {'auc': 0.5, 'error': 'Insufficient samples'}
        
        try:
            # Train LGBM with optimizations
            model = lgb.LGBMClassifier(**self.lgbm_params)
            
            model.fit(
                X, y, 
                sample_weight=weights,
                eval_set=[(X, y)],
                eval_metric='auc',
                callbacks=[lgb.early_stopping(15), lgb.log_evaluation(0)]
            )
            
            # Predictions
            predictions = model.predict_proba(X)[:, 1]
            
            # Performance metrics
            auc = roc_auc_score(y, predictions, sample_weight=weights)
            
            performance = {
                'auc': auc,
                'n_features': len(X.columns),
                'n_samples': len(X),
                'training_time': time.time() - start_time
            }
            
            return predictions, performance
            
        except Exception as e:
            return np.zeros(len(X)), {'auc': 0.5, 'error': str(e)}
    
    def _train_specialist_batch(
        self,
        specialist_df: pd.DataFrame,
        orthogonal_targets: pd.DataFrame,
        sample_weights: Optional[pd.Series],
        specialist_batch: List[str]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
        """Train a batch of specialists sequentially"""
        
        batch_predictions = {}
        batch_performance = {}
        
        for specialist in specialist_batch:
            if f"{specialist}_orthogonal" in orthogonal_targets.columns:
                try:
                    predictions, performance = self._train_single_specialist_lgbm(
                        specialist_df,
                        orthogonal_targets[f"{specialist}_orthogonal"],
                        sample_weights,
                        specialist
                    )
                    
                    batch_predictions[specialist] = predictions
                    batch_performance[specialist] = performance
                    
                except Exception as e:
                    tprint_warning(f"Failed to train {specialist}: {e}")
                    batch_performance[specialist] = {'auc': 0.5, 'error': str(e)}
        
        return batch_predictions, batch_performance
    
    def run_2_core_lgbm_comparison(
        self, 
        specialist_df: pd.DataFrame, 
        target_series: pd.Series, 
        sample_weights: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Run LGBM comparison with 2-core parallel processing"""
        
        tprint_info("🚀 Starting 2-core optimized specialist orthogonalization...")
        tprint_info(f"📊 Dataset: {len(specialist_df):,} samples, {len(specialist_df.columns)} features")
        tprint_info(f"🎯 Specialists: {len(self.specialist_categories)} total")
        tprint_info(f"⚡ Processing: 2-core parallel with balanced batching")
        
        # Phase 1: Orthogonal target generation
        orthogonal_targets, auc_weights = self.generate_auc_weighted_orthogonal_targets(
            specialist_df, target_series, sample_weights
        )
        
        # Phase 2: Create batches
        available_specialists = [s for s in self.specialist_categories.keys() 
                                if f"{s}_orthogonal" in orthogonal_targets.columns]
        specialist_batches = self._create_specialist_batches(available_specialists)
        
        tprint_info(f"📦 Created {len(specialist_batches)} batches for 2-core processing")
        
        # Phase 3: Parallel batch processing
        orthogonal_predictions = {}
        orthogonal_performance = {}
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            
            for batch_idx, specialist_batch in enumerate(specialist_batches):
                future = executor.submit(
                    self._train_specialist_batch,
                    specialist_df,
                    orthogonal_targets,
                    sample_weights,
                    specialist_batch
                )
                futures[future] = batch_idx
            
            for future in as_completed(futures):
                batch_idx = futures[future]
                try:
                    batch_predictions, batch_performance = future.result(timeout=900)  # 15min timeout
                    
                    orthogonal_predictions.update(batch_predictions)
                    orthogonal_performance.update(batch_performance)
                    
                    completed_specialists = len(orthogonal_predictions)
                    total_specialists = sum(len(batch) for batch in specialist_batches)
                    tprint_info(f"  ✅ Batch {batch_idx + 1} complete: {completed_specialists}/{total_specialists} specialists")
                    
                except Exception as e:
                    tprint_error(f"❌ Batch {batch_idx} failed: {e}")
        
        # Phase 4: Baseline LGBM
        baseline_results = self._run_baseline_lgbm(specialist_df, target_series, sample_weights)
        
        # Phase 5: Ensemble creation and evaluation
        ensemble_results = self._create_auc_weighted_ensemble(
            orthogonal_predictions, auc_weights, target_series, sample_weights
        )
        
        tprint_success("✅ 2-core optimization complete!")
        
        return {
            'baseline': baseline_results,
            'orthogonal': orthogonal_performance,
            'auc_weights': auc_weights,
            'ensemble': ensemble_results,
            'predictions': orthogonal_predictions
        }
    
    def _run_baseline_lgbm(
        self, 
        specialist_df: pd.DataFrame, 
        target_series: pd.Series, 
        sample_weights: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Run baseline LGBM on all specialist features"""
        
        tprint_info("📊 Running baseline LGBM comparison...")
        
        # Combine all specialist features
        all_features = []
        for specialist in self.specialist_categories.keys():
            features = self.extract_specialist_features(specialist_df, specialist)
            if not features.empty:
                all_features.append(features)
        
        if not all_features:
            return {'auc': 0.5, 'error': 'No features available'}
        
        X_baseline = pd.concat(all_features, axis=1)
        X_baseline = self._optimize_memory_usage(X_baseline)
        
        # Prepare data
        y_baseline = target_series.reindex(X_baseline.index).fillna(0)
        weights_baseline = sample_weights.reindex(X_baseline.index).fillna(1.0) if sample_weights is not None else None
        
        try:
            model = lgb.LGBMClassifier(**self.lgbm_params)
            model.fit(
                X_baseline, y_baseline,
                sample_weight=weights_baseline,
                eval_set=[(X_baseline, y_baseline)],
                eval_metric='auc',
                callbacks=[lgb.early_stopping(15), lgb.log_evaluation(0)]
            )
            
            predictions = model.predict_proba(X_baseline)[:, 1]
            auc = roc_auc_score(y_baseline, predictions, sample_weight=weights_baseline)
            
            return {
                'auc': auc,
                'n_features': len(X_baseline.columns),
                'n_samples': len(X_baseline),
                'model': model
            }
            
        except Exception as e:
            return {'auc': 0.5, 'error': str(e)}
    
    def _create_auc_weighted_ensemble(
        self,
        orthogonal_predictions: Dict[str, np.ndarray],
        auc_weights: Dict[str, float],
        target_series: pd.Series,
        sample_weights: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Create AUC-weighted ensemble from orthogonal predictions"""
        
        tprint_info("🎯 Creating AUC-weighted ensemble...")
        
        if not orthogonal_predictions:
            return {'auc': 0.5, 'error': 'No orthogonal predictions available'}
        
        # Align all predictions to common index
        common_index = None
        aligned_predictions = {}
        
        for specialist, predictions in orthogonal_predictions.items():
            if common_index is None:
                common_index = pd.Series(predictions).index
            else:
                common_index = common_index.intersection(pd.Series(predictions).index)
        
        for specialist, predictions in orthogonal_predictions.items():
            pred_series = pd.Series(predictions, index=pd.Series(predictions).index)
            aligned_predictions[specialist] = pred_series.reindex(common_index)
        
        # Create weighted ensemble
        ensemble_prediction = np.zeros(len(common_index))
        total_weight = 0
        
        for specialist, predictions in aligned_predictions.items():
            weight = auc_weights.get(specialist, 0.0)
            ensemble_prediction += predictions.values * weight
            total_weight += weight
        
        if total_weight > 0:
            ensemble_prediction /= total_weight
        
        # Calculate ensemble performance
        y_ensemble = target_series.reindex(common_index).fillna(0)
        weights_ensemble = sample_weights.reindex(common_index).fillna(1.0) if sample_weights is not None else None
        
        ensemble_auc = roc_auc_score(y_ensemble, ensemble_prediction, sample_weight=weights_ensemble)
        
        return {
            'auc': ensemble_auc,
            'predictions': ensemble_prediction,
            'n_specialists': len(aligned_predictions),
            'weights_used': auc_weights
        }
    
    def validate_specialist_coverage(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Validate which specialists have sufficient feature coverage"""
        coverage = {}
        
        for specialist in self.specialist_categories.keys():
            features = self.extract_specialist_features(df, specialist)
            coverage[specialist] = not features.empty
        
        return coverage
    
    def get_specialist_importance(self) -> Dict[str, float]:
        """Get specialist importance weights"""
        return {spec: config['importance'] for spec, config in self.specialist_categories.items()}

    def run_optimized_orthogonalization(self, specialist_df: pd.DataFrame, 
                                      target_series: pd.Series,
                                      sample_weights: Optional[pd.Series] = None,
                                      run_hpo: bool = False,
                                      run_pruning: bool = False,
                                      optimize_features: bool = False) -> Dict[str, Any]:
        """Run optimized orthogonalization with all enhancements"""
        
        tprint_info("🚀 Starting optimized specialist orthogonalization...")
        start_time = time.time()
        
        # Step 1: Feature optimization if enabled
        if optimize_features and self.feature_optimizer is not None:
            tprint_info("🔧 Optimizing specialist features...")
            self.specialist_categories, feature_analysis = self.feature_optimizer.optimize_feature_pipeline(
                self.specialist_categories, specialist_df
            )
            self.feature_mappings = self._create_feature_mappings()
        
        # Step 2: Extract specialist features with caching
        specialist_data = {}
        for specialist in self.specialist_categories.keys():
            features = self.extract_specialist_features(specialist_df, specialist)
            if not features.empty:
                specialist_data[specialist] = features
        
        tprint_info(f"📊 Extracted features for {len(specialist_data)} specialists")
        
        # Step 3: Run orthogonalization-aware HPO if enabled
        hpo_results = {}
        if run_hpo and self.orthogonal_hpo is not None and len(specialist_data) > 0:
            tprint_info("🎯 Running orthogonalization-aware HPO...")
            hpo_results = self.orthogonal_hpo.optimize_specialist_ensemble(
                list(specialist_data.keys()), specialist_data, target_series, sample_weights
            )
        
        # Step 4: Generate orthogonal targets
        if hpo_results:
            orthogonal_targets = self._generate_targets_with_hpo_params(
                specialist_data, target_series, sample_weights, hpo_results
            )
        else:
            orthogonal_targets, auc_weights = self.generate_auc_weighted_orthogonal_targets(
                specialist_df, target_series, sample_weights
            )
            auc_weights = {'auc_weights': auc_weights}
        
        # Step 5: Conservative ensemble pruning if enabled
        pruned_specialists = list(specialist_data.keys())
        if run_pruning and self.pruner is not None and hpo_results:
            tprint_info("✂️ Running conservative ensemble pruning...")
            performance_metrics = self._extract_performance_metrics(hpo_results)
            
            # Calculate diversity matrix
            specialist_predictions = {}
            for specialist in specialist_data.keys():
                # Generate placeholder predictions for diversity calculation
                specialist_predictions[specialist] = np.random.random(len(target_series))
            
            diversity_matrix = self.pruner.calculate_diversity_matrix(specialist_predictions)
            
            pruned_specialists = self.pruner.prune_ensemble_conservative(
                specialist_data, performance_metrics, diversity_matrix
            )
        
        # Step 6: Final ensemble training
        final_results = {}
        if pruned_specialists:
            tprint_info(f"🎯 Training final ensemble with {len(pruned_specialists)} specialists...")
            final_results = self._train_final_ensemble(
                pruned_specialists, specialist_data, orthogonal_targets, sample_weights
            )
        
        total_time = time.time() - start_time
        
        # Create comprehensive results
        results = {
            'optimization_time': total_time,
            'feature_analysis': feature_analysis if optimize_features else None,
            'hpo_results': hpo_results,
            'orthogonal_targets': orthogonal_targets,
            'pruned_specialists': pruned_specialists,
            'final_results': final_results,
            'performance_summary': self._create_performance_summary(final_results),
            'optimization_settings': {
                'feature_optimization': optimize_features,
                'hpo_enabled': run_hpo,
                'pruning_enabled': run_pruning,
                'cache_enabled': self.enable_cache
            }
        }
        
        tprint_success(f"✅ Optimized orthogonalization completed in {total_time:.1f}s")
        return results
    
    def _generate_targets_with_hpo_params(self, specialist_data: Dict[str, pd.DataFrame],
                                        target_series: pd.Series, sample_weights: Optional[pd.Series],
                                        hpo_results: Dict[str, Any]) -> pd.DataFrame:
        """Generate orthogonal targets using HPO-optimized parameters"""
        
        orthogonal_targets = pd.DataFrame(index=target_series.index)
        
        # Use HPO results to generate better orthogonal targets
        orthogonal_results = hpo_results.get('orthogonal_results', {})
        
        for specialist_name, specialist_features in specialist_data.items():
            if specialist_name in orthogonal_results:
                # Generate orthogonal target using optimized approach
                try:
                    # Get other specialists' data
                    other_specialists = [s for s in specialist_data.keys() if s != specialist_name]
                    other_features_list = []
                    
                    for other_spec in other_specialists:
                        other_features = specialist_data[other_spec]
                        if not other_features.empty:
                            other_features_list.append(other_features)
                    
                    if other_features_list:
                        # Combine other specialist features
                        other_features_combined = pd.concat(other_features_list, axis=1)
                        other_features_clean = self._optimize_memory_usage(other_features_combined)
                        
                        # Train regression model to remove other specialists' influence
                        specialist_features_clean = self._optimize_memory_usage(specialist_features)
                        
                        model = lgb.LGBMRegressor(
                            n_estimators=50,
                            max_depth=3,
                            learning_rate=0.1,
                            verbose=-1
                        )
                        
                        X_clean = specialist_features_clean.fillna(0)
                        other_clean = other_features_clean.fillna(0)
                        y_clean = target_series.reindex(X_clean.index).fillna(0)
                        
                        if len(X_clean) > 50:
                            model.fit(X_clean, y_clean)
                            predictions = model.predict(X_clean)
                            
                            # Calculate orthogonal target (residuals)
                            orthogonal_target = y_clean - predictions
                            orthogonal_targets[f"{specialist_name}_orthogonal"] = orthogonal_target
                        else:
                            orthogonal_targets[f"{specialist_name}_orthogonal"] = y_clean
                    else:
                        orthogonal_targets[f"{specialist_name}_orthogonal"] = target_series
                        
                except Exception as e:
                    tprint_warning(f"Failed to generate orthogonal target for {specialist_name}: {e}")
                    orthogonal_targets[f"{specialist_name}_orthogonal"] = target_series
            else:
                orthogonal_targets[f"{specialist_name}_orthogonal"] = target_series
        
        return orthogonal_targets
    
    def _extract_performance_metrics(self, hpo_results: Dict[str, Any]) -> Dict[str, Dict]:
        """Extract performance metrics from HPO results"""
        
        performance_metrics = {}
        orthogonal_results = hpo_results.get('orthogonal_results', {})
        
        for specialist_name, results in orthogonal_results.items():
            performance_metrics[specialist_name] = {
                'auc': results.get('individual_auc', 0.5),
                'orthogonality_score': results.get('orthogonality_score', 0.0),
                'stability': 0.7,  # Placeholder
                'training_speed': 1.0,  # Placeholder
                'individual_auc': results.get('individual_auc', 0.5)
            }
        
        return performance_metrics
    
    def _train_final_ensemble(self, pruned_specialists: List[str], 
                            specialist_data: Dict[str, pd.DataFrame],
                            orthogonal_targets: pd.DataFrame,
                            sample_weights: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Train final ensemble with pruned specialists"""
        
        ensemble_results = {}
        
        for specialist in pruned_specialists:
            if f"{specialist}_orthogonal" in orthogonal_targets.columns:
                orthogonal_target = orthogonal_targets[f"{specialist}_orthogonal"]
                specialist_features = specialist_data[specialist]
                
                # Train final model
                predictions, performance = self._train_single_specialist_lgbm(
                    specialist_features, orthogonal_target, sample_weights, specialist
                )
                
                ensemble_results[specialist] = {
                    'predictions': predictions,
                    'performance': performance
                }
        
        return ensemble_results
    
    def _create_performance_summary(self, final_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create performance summary for the optimized ensemble"""
        
        if not final_results:
            return {'message': 'No results available'}
        
        # Aggregate performance metrics
        all_aucs = []
        all_training_times = []
        
        for specialist, results in final_results.items():
            performance = results.get('performance', {})
            all_aucs.append(performance.get('auc', 0.5))
            all_training_times.append(performance.get('training_time', 0.0))
        
        summary = {
            'n_specialists': len(final_results),
            'mean_auc': np.mean(all_aucs) if all_aucs else 0.5,
            'std_auc': np.std(all_aucs) if all_aucs else 0.0,
            'total_training_time': sum(all_training_times),
            'mean_training_time': np.mean(all_training_times) if all_training_times else 0.0,
            'best_specialist': max(final_results.keys(), key=lambda x: final_results[x]['performance'].get('auc', 0.5)) if final_results else None,
            'worst_specialist': min(final_results.keys(), key=lambda x: final_results[x]['performance'].get('auc', 0.5)) if final_results else None
        }
        
        return summary


    def generate_denoised_orthogonal_targets(self, specialist_df: pd.DataFrame, 
                                           target_series: pd.Series,
                                           sample_weights: Optional[pd.Series] = None,
                                           denoising_method: str = 'kalman',
                                           volume_series: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Generate orthogonal targets with denoised labels"""
        
        if self.target_denoiser is None:
            tprint_warning("Target denoiser not available, using original targets")
            return self.generate_auc_weighted_orthogonal_targets(specialist_df, target_series, sample_weights)
        
        tprint_info(f"🔇 Applying target denoising with method: {denoising_method}")
        
        # Update denoiser method if different
        if self.target_denoiser.config.method != denoising_method:
            from .target_denoiser import DenoisingConfig
            self.target_denoiser.config.method = denoising_method
        
        # Denoise the target
        denoising_result = self.target_denoiser.denoise_target(
            target_series, features=specialist_df, volume_series=volume_series
        )
        
        denoised_target = denoising_result.denoised_target
        tprint_info(f"   Noise reduction: {denoising_result.denoising_stats.get('noise_reduction', 0):.1%}")
        tprint_info(f"   Agreement rate: {denoising_result.denoising_stats.get('agreement_rate', 0):.1%}")
        
        # Generate orthogonal targets with denoised labels
        orthogonal_targets = pd.DataFrame(index=specialist_df.index)
        denoising_results = {}
        
        # Validate specialist coverage
        coverage = self.validate_specialist_coverage(specialist_df)
        available_specialists = [s for s, has in coverage.items() if has]
        
        if len(available_specialists) < 2:
            tprint_warning("Need at least 2 specialists for orthogonalization")
            return orthogonal_targets, {}
        
        # Calculate baseline performance with denoised target
        baseline_performance = self._calculate_baseline_performance(
            specialist_df, denoised_target, sample_weights
        )
        
        # Calculate AUC weights
        auc_weights = self.calculate_auc_weights(baseline_performance)
        
        # Generate orthogonal targets for each specialist
        for specialist in available_specialists:
            tprint_info(f"  Processing {specialist} specialist with denoised target...")
            
            # Get features for this specialist
            specialist_features = self.extract_specialist_features(specialist_df, specialist)
            
            if specialist_features.empty:
                orthogonal_targets[f"{specialist}_orthogonal"] = denoised_target
                denoising_results[specialist] = {'method': 'none', 'reason': 'no_features'}
                continue
            
            # Get features from all OTHER specialists
            other_specialists = [s for s in available_specialists if s != specialist]
            other_features_list = []
            
            for other_spec in other_specialists:
                other_features = self.extract_specialist_features(specialist_df, other_spec)
                if not other_features.empty:
                    other_features_list.append(other_features)
            
            if other_features_list:
                # Combine all other specialist features
                other_features_combined = pd.concat(other_features_list, axis=1)
                
                # Handle NaN values
                other_features_clean = self._optimize_memory_usage(other_features_combined)
                target_clean = denoised_target.reindex(other_features_clean.index).fillna(0)
                
                # Train model on other specialists to predict denoised target
                try:
                    # Use LogisticRegression for binary target
                    if sample_weights is not None:
                        weights_clean = sample_weights.reindex(other_features_clean.index).fillna(1.0)
                        model = LogisticRegression(random_state=42, max_iter=1000)
                        model.fit(other_features_clean, target_clean, sample_weight=weights_clean)
                    else:
                        model = LogisticRegression(random_state=42, max_iter=1000)
                        model.fit(other_features_clean, target_clean)
                    
                    # Predict what other specialists think
                    other_predictions = model.predict_proba(other_features_clean)[:, 1]
                    
                    # Create orthogonal target (denoised target - other specialists' predictions)
                    orthogonal_target = target_clean - other_predictions
                    
                    # Store orthogonal target
                    orthogonal_targets[f"{specialist}_orthogonal"] = orthogonal_target
                    
                    denoising_results[specialist] = {
                        'method': 'orthogonalized',
                        'original_target': target_series,
                        'denoised_target': denoised_target,
                        'orthogonal_target': orthogonal_target,
                        'auc_weight': auc_weights.get(specialist, 0.0)
                    }
                    
                except Exception as e:
                    tprint_warning(f"Failed to orthogonalize {specialist}: {e}")
                    orthogonal_targets[f"{specialist}_orthogonal"] = denoised_target
                    denoising_results[specialist] = {'method': 'fallback', 'reason': str(e)}
            else:
                orthogonal_targets[f"{specialist}_orthogonal"] = denoised_target
                denoising_results[specialist] = {'method': 'no_others', 'reason': 'no_other_specialists'}
        
        return orthogonal_targets, {
            'denoising_result': denoising_result,
            'specialist_denoising': denoising_results,
            'auc_weights': auc_weights,
            'baseline_performance': baseline_performance,
            'denoising_method': denoising_method
        }

