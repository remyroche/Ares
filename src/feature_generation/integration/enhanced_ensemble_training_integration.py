"""
Enhanced Ensemble Training Integration

This module provides meta-features optimized for ensemble training including:
- Base model outputs (required)
- Disagreement features between base models (including CV and max pairwise)
- Entropy features derived from disagreements
- Feature interaction analysis (PCA, mutual information, complementarity)
- Overall feature disagreement measures

Target: Meta-features optimized for ensemble training
Focus: Base model disagreement, entropy analysis, and interaction patterns
"""

import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import time

# Import tprint for comprehensive logging
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_debug, tprint_warning, tprint_error, 
        tprint_success, tprint_performance, tprint_progress, tprint_structured,
        tprint_data_preview, tprint_data_format, tprint_feature_counts,
        tprint_timer, tprint_logged, LogLevel
    )
    TPRINT_AVAILABLE = True
except ImportError:
    # Fallback if tprint is not available
    def tprint(*args, **kwargs):
        print(*args, **kwargs)
    def tprint_info(*args, **kwargs):
        print(f"[INFO] {' '.join(str(arg) for arg in args)}")
    def tprint_debug(*args, **kwargs):
        print(f"[DEBUG] {' '.join(str(arg) for arg in args)}")
    def tprint_warning(*args, **kwargs):
        print(f"[WARNING] {' '.join(str(arg) for arg in args)}")
    def tprint_error(*args, **kwargs):
        print(f"[ERROR] {' '.join(str(arg) for arg in args)}")
    def tprint_success(*args, **kwargs):
        print(f"[SUCCESS] {' '.join(str(arg) for arg in args)}")
    def tprint_performance(*args, **kwargs):
        print(f"[PERFORMANCE] {' '.join(str(arg) for arg in args)}")
    def tprint_progress(*args, **kwargs):
        print(f"[PROGRESS] {' '.join(str(arg) for arg in args)}")
    def tprint_structured(*args, **kwargs):
        print(f"[STRUCTURED] {' '.join(str(arg) for arg in args)}")
    def tprint_data_preview(*args, **kwargs):
        print(f"[DATA_PREVIEW] {' '.join(str(arg) for arg in args)}")
    def tprint_data_format(*args, **kwargs):
        print(f"[DATA_FORMAT] {' '.join(str(arg) for arg in args)}")
    def tprint_feature_counts(*args, **kwargs):
        print(f"[FEATURE_COUNTS] {' '.join(str(arg) for arg in args)}")
    def tprint_timer(operation):
        from contextlib import contextmanager
        @contextmanager
        def timer():
            start = time.time()
            yield
            print(f"[TIMER] {operation} took {time.time() - start:.3f}s")
        return timer()
    def tprint_logged(level=None, include_args=False, include_result=False):
        def decorator(func):
            return func
        return decorator
    class LogLevel:
        INFO = "INFO"
        DEBUG = "DEBUG"
        WARNING = "WARNING"
        ERROR = "ERROR"
        SUCCESS = "SUCCESS"
        PERFORMANCE = "PERFORMANCE"
    TPRINT_AVAILABLE = False

# Meta-features only - no feature bank integration needed

# Import ensemble models
try:
    from sklearn.ensemble import VotingRegressor, StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. Install with: pip install scikit-learn")

# Import LGBM for base models
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    warnings.warn("LGBM not available. Install with: pip install lightgbm")


class EnhancedEnsembleTrainingIntegration:
    """
    Enhanced Ensemble Training Integration.
    
    Provides meta-features optimized for meta-learner training including:
    - Base model outputs (required)
    - Disagreement features between base models (including CV and max pairwise)
    - Entropy features derived from disagreements
    - Feature interaction analysis (PCA, mutual information, complementarity)
    - Overall feature disagreement measures
    """
    
    def __init__(self, 
                 min_features: int = 20,
                 max_features: int = 40,
                 ensemble_config: Optional[Dict[str, Any]] = None):
        tprint_info("🚀 Initializing Enhanced Ensemble Training Integration")
        tprint_structured({
            "min_features": min_features,
            "max_features": max_features,
            "ensemble_config": ensemble_config or {},
            "tprint_available": TPRINT_AVAILABLE
        })
        
        self.min_features = min_features
        self.max_features = max_features
        self.ensemble_config = ensemble_config or {}
        
        # Meta-features only - no feature bank integration needed
        self.feature_integrator = None
        
        tprint_success("✅ Enhanced Ensemble Training Integration initialized successfully")
    
    def get_comprehensive_ensemble_features(self, data: pd.DataFrame, 
                                         base_models: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get meta-features optimized for ensemble training.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            base_models: Dictionary of trained base models (required)
            
        Returns:
            Dictionary containing meta-features and metadata
        """
        tprint_info("🎯 Starting comprehensive ensemble feature generation")
        tprint_data_preview(data, "Input Data", max_rows=3, max_cols=5)
        
        if base_models is None:
            tprint_error("❌ Base models must be provided for ensemble training")
            raise ValueError("Base models must be provided for ensemble training")
        
        tprint_info(f"📊 Processing {len(base_models)} base models: {list(base_models.keys())}")
        
        with tprint_timer("Meta-feature generation"):
            # Generate only meta-features
            meta_features = self._generate_meta_features(data, {}, base_models)
        
        tprint_feature_counts(0, len(meta_features['feature_names']), "Meta-feature generation")
        
        result = {
            'features': meta_features['features'],
            'feature_names': meta_features['feature_names'],
            'feature_count': len(meta_features['feature_names']),
            'target_range': (self.min_features, self.max_features),
            'ensemble_optimized': True,
            'comprehensive_features': False,
            'meta_features_included': True,
            'base_models_included': True,
            'feature_categories': self._get_feature_category_breakdown(meta_features['features']),
            'ensemble_readiness': self._assess_ensemble_readiness(meta_features['features']),
            'description': 'Meta-features for ensemble training'
        }
        
        tprint_structured({
            "feature_count": result['feature_count'],
            "target_range": result['target_range'],
            "feature_categories": result['feature_categories'],
            "ensemble_readiness_score": result['ensemble_readiness']['score']
        })
        
        tprint_success(f"✅ Generated {result['feature_count']} meta-features for ensemble training")
        
        return result
    
    
    def _generate_meta_features(self, data: pd.DataFrame, 
                              base_features: Dict[str, np.ndarray],
                              base_models: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate meta-features for ensemble training."""
        tprint_info("🔧 Starting meta-feature generation")
        tprint_debug(f"Input data shape: {data.shape}, Base models: {len(base_models) if base_models else 0}")
        
        meta_features = {}
        meta_feature_names = []
        
        # Base model outputs (always required)
        if base_models:
            tprint_info("📊 Generating base model outputs")
            with tprint_timer("Base model outputs generation"):
                base_outputs = self._generate_base_model_outputs(data, base_models)
            meta_features.update(base_outputs['features'])
            meta_feature_names.extend(base_outputs['feature_names'])
            tprint_success(f"✅ Generated {len(base_outputs['features'])} base model outputs")
        else:
            tprint_error("❌ Base models must be provided for meta-feature generation")
            raise ValueError("Base models must be provided for meta-feature generation")
        
        # Disagreement features (using base model outputs as base_features)
        tprint_info("⚖️ Generating disagreement features")
        with tprint_timer("Disagreement features generation"):
            disagreement_features = self._generate_disagreement_features(data, meta_features, base_models)
        meta_features.update(disagreement_features['features'])
        meta_feature_names.extend(disagreement_features['feature_names'])
        tprint_success(f"✅ Generated {len(disagreement_features['features'])} disagreement features")
        
        # Entropy features (using all current features)
        tprint_info("📈 Generating entropy features")
        with tprint_timer("Entropy features generation"):
            entropy_features = self._generate_entropy_features(data, meta_features, base_models)
        meta_features.update(entropy_features['features'])
        meta_feature_names.extend(entropy_features['feature_names'])
        tprint_success(f"✅ Generated {len(entropy_features['features'])} entropy features")
        
        # Interaction features
        tprint_info("🔗 Generating interaction features")
        with tprint_timer("Interaction features generation"):
            interaction_features = self._generate_interaction_features(data, meta_features, base_models)
        meta_features.update(interaction_features['features'])
        meta_feature_names.extend(interaction_features['feature_names'])
        tprint_success(f"✅ Generated {len(interaction_features['features'])} interaction features")
        
        result = {
            'features': meta_features,
            'feature_names': meta_feature_names,
            'metadata': {
                'base_outputs': base_models is not None,
                'disagreement_features': len(disagreement_features['features']),
                'entropy_features': len(entropy_features['features']),
                'interaction_features': len(interaction_features['features']),
                'total_meta_features': len(meta_feature_names)
            }
        }
        
        tprint_structured({
            "total_meta_features": result['metadata']['total_meta_features'],
            "base_outputs": result['metadata']['base_outputs'],
            "disagreement_features": result['metadata']['disagreement_features'],
            "entropy_features": result['metadata']['entropy_features'],
            "interaction_features": result['metadata']['interaction_features']
        })
        
        tprint_success(f"🎉 Meta-feature generation completed: {len(meta_feature_names)} total features")
        
        return result
    
    def _generate_base_model_outputs(self, data: pd.DataFrame, 
                                   base_models: Dict[str, Any]) -> Dict[str, Any]:
        """Generate base model outputs as features."""
        tprint_info("🤖 Generating base model outputs")
        tprint_debug(f"Processing {len(base_models)} base models")
        
        features = {}
        feature_names = []
        
        # Prepare base features for base models
        tprint_info("🔧 Preparing base features for model prediction")
        base_features = self._prepare_base_features_for_models(data)
        tprint_data_format(base_features, "Base Features for Models", check_compatibility=True)
        
        successful_models = 0
        failed_models = 0
        
        for model_name, model in base_models.items():
            tprint_debug(f"Processing base model: {model_name}")
            try:
                # Make predictions
                with tprint_timer(f"Prediction for {model_name}"):
                    predictions = model.predict(base_features)
                
                # Store as feature
                feature_name = f'base_model_{model_name}_output'
                features[feature_name] = predictions
                feature_names.append(feature_name)
                
                tprint_success(f"✅ Generated output for {model_name}: {len(predictions)} predictions")
                successful_models += 1
                
            except Exception as e:
                tprint_error(f"❌ Failed to generate output for base model {model_name}: {e}")
                warnings.warn(f"Failed to generate output for base model {model_name}: {e}")
                failed_models += 1
        
        tprint_structured({
            "successful_models": successful_models,
            "failed_models": failed_models,
            "total_features_generated": len(features)
        })
        
        if failed_models > 0:
            tprint_warning(f"⚠️ {failed_models} out of {len(base_models)} models failed to generate outputs")
        
        return {
            'features': features,
            'feature_names': feature_names
        }
    
    def _prepare_base_features_for_models(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare base features for base model prediction."""
        # This would prepare the same features used for base model training
        # For now, return a simple implementation
        if 'close' in data.columns:
            prices = data['close']
            returns = prices.pct_change().fillna(0)
            volatility = returns.rolling(20).std().fillna(0)
            
            # Simple feature matrix
            features = np.column_stack([
                returns.values,
                volatility.values,
                prices.pct_change(5).fillna(0).values,
                prices.pct_change(10).fillna(0).values
            ])
            
            # Handle NaN values
            features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
            
            return features
        else:
            # Fallback: return random features
            return np.random.randn(len(data), 4)
    
    def _generate_disagreement_features(self, data: pd.DataFrame, 
                                      base_features: Dict[str, np.ndarray],
                                      base_models: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate disagreement features between base model outputs."""
        tprint_info("⚖️ Generating disagreement features")
        
        features = {}
        feature_names = []
        
        # Get base model outputs
        base_model_outputs = []
        base_model_names = []
        for model_name in base_models.keys():
            feature_name = f'base_model_{model_name}_output'
            if feature_name in base_features:
                base_model_outputs.append(base_features[feature_name])
                base_model_names.append(model_name)
        
        tprint_debug(f"Found {len(base_model_outputs)} base model outputs: {base_model_names}")
        
        if len(base_model_outputs) < 2:
            tprint_warning("⚠️ Need at least 2 base model outputs for disagreement features")
            return {
                'features': features,
                'feature_names': feature_names
            }
        
        # Convert to numpy array for easier computation
        base_model_array = np.array(base_model_outputs)
        tprint_data_format(base_model_array, "Base Model Outputs Array", check_compatibility=True)
        
        # Basic disagreement measures
        tprint_debug("Calculating basic disagreement measures")
        base_model_std = np.std(base_model_array, axis=0)
        features['base_model_disagreement'] = base_model_std
        feature_names.append('base_model_disagreement')
        tprint_success("✅ Generated base_model_disagreement")
        
        # Coefficient of Variation (CV) - normalized disagreement
        tprint_debug("Calculating coefficient of variation disagreement")
        mean_predictions = np.mean(base_model_array, axis=0)
        cv_disagreement = np.divide(base_model_std, np.abs(mean_predictions) + 1e-8)
        features['cv_disagreement'] = cv_disagreement
        feature_names.append('cv_disagreement')
        tprint_success("✅ Generated cv_disagreement")
        
        # Maximum pairwise disagreement
        tprint_debug("Calculating maximum pairwise disagreement")
        max_pairwise_disagreement = np.zeros(base_model_array.shape[1])
        for i in range(base_model_array.shape[1]):
            pairwise_diffs = []
            for j in range(len(base_model_names)):
                for k in range(j + 1, len(base_model_names)):
                    diff = np.abs(base_model_array[j, i] - base_model_array[k, i])
                    pairwise_diffs.append(diff)
            if pairwise_diffs:
                max_pairwise_disagreement[i] = np.max(pairwise_diffs)
        
        features['max_pairwise_disagreement'] = max_pairwise_disagreement
        feature_names.append('max_pairwise_disagreement')
        tprint_success("✅ Generated max_pairwise_disagreement")
        
        # Calculate pairwise disagreements between models
        tprint_debug("Calculating pairwise disagreements between models")
        pairwise_count = 0
        for i, name1 in enumerate(base_model_names):
            for j, name2 in enumerate(base_model_names[i+1:], i+1):
                if i < len(base_model_outputs) and j < len(base_model_outputs):
                    disagreement = np.abs(base_model_outputs[i] - base_model_outputs[j])
                    features[f'{name1}_{name2}_disagreement'] = disagreement
                    feature_names.append(f'{name1}_{name2}_disagreement')
                    pairwise_count += 1
        
        tprint_success(f"✅ Generated {pairwise_count} pairwise disagreement features")
        
        # Overall feature disagreement (using all base model outputs)
        if len(base_model_outputs) > 1:
            tprint_debug("Calculating overall feature disagreement")
            all_features = np.array(base_model_outputs)
            feature_std = np.std(all_features, axis=0)
            features['overall_feature_disagreement'] = feature_std
            feature_names.append('overall_feature_disagreement')
            tprint_success("✅ Generated overall_feature_disagreement")
        
        tprint_structured({
            "total_disagreement_features": len(features),
            "base_model_count": len(base_model_outputs),
            "pairwise_features": pairwise_count
        })
        
        tprint_success(f"🎉 Generated {len(features)} disagreement features")
        
        return {
            'features': features,
            'feature_names': feature_names
        }
    
    def _generate_entropy_features(self, data: pd.DataFrame, 
                                 base_features: Dict[str, np.ndarray],
                                 base_models: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate entropy features for ensemble training, including entropy from disagreements."""
        tprint_info("📈 Generating entropy features")
        tprint_debug(f"Processing {len(base_features)} base features for entropy calculation")
        
        features = {}
        feature_names = []
        
        # Feature entropy
        tprint_debug("Calculating individual feature entropy")
        individual_entropy_count = 0
        for name, values in base_features.items():
            if len(values) > 0:
                try:
                    # Calculate entropy
                    hist, _ = np.histogram(values, bins=20)
                    hist = hist / np.sum(hist)  # Normalize
                    hist = hist[hist > 0]  # Remove zero bins
                    if len(hist) > 0:
                        entropy = -np.sum(hist * np.log2(hist))
                        
                        features[f'{name}_entropy'] = np.full(len(values), entropy)
                        feature_names.append(f'{name}_entropy')
                        individual_entropy_count += 1
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to calculate entropy for {name}: {e}")
        
        tprint_success(f"✅ Generated {individual_entropy_count} individual entropy features")
        
        # Cross-feature entropy
        tprint_debug("Calculating cross-feature joint entropy")
        joint_entropy_count = 0
        if len(base_features) > 1:
            feature_names_list = list(base_features.keys())
            for i, name1 in enumerate(feature_names_list):
                for name2 in feature_names_list[i+1:]:
                    values1 = base_features[name1]
                    values2 = base_features[name2]
                    
                    if len(values1) == len(values2):
                        try:
                            # Calculate joint entropy
                            joint_values = np.column_stack([values1, values2])
                            hist, _ = np.histogram2d(values1, values2, bins=10)
                            hist = hist / np.sum(hist)
                            hist = hist[hist > 0]
                            if len(hist) > 0:
                                joint_entropy = -np.sum(hist * np.log2(hist))
                                
                                features[f'{name1}_{name2}_joint_entropy'] = np.full(len(values1), joint_entropy)
                                feature_names.append(f'{name1}_{name2}_joint_entropy')
                                joint_entropy_count += 1
                        except Exception as e:
                            tprint_warning(f"⚠️ Failed to calculate joint entropy for {name1} and {name2}: {e}")
        
        tprint_success(f"✅ Generated {joint_entropy_count} joint entropy features")
        
        # Entropy from disagreements
        tprint_debug("Calculating entropy from disagreement features")
        disagreement_features = ['base_model_disagreement', 'overall_feature_disagreement']
        
        # Add pairwise disagreement features
        if base_models:
            base_model_names = list(base_models.keys())
            for i, name1 in enumerate(base_model_names):
                for j, name2 in enumerate(base_model_names[i+1:], i+1):
                    disagreement_name = f'{name1}_{name2}_disagreement'
                    disagreement_features.append(disagreement_name)
        
        disagreement_entropy_count = 0
        for disagreement_name in disagreement_features:
            if disagreement_name in base_features:
                values = base_features[disagreement_name]
                if len(values) > 0:
                    try:
                        # Calculate entropy of disagreement
                        hist, _ = np.histogram(values, bins=20)
                        hist = hist / np.sum(hist)  # Normalize
                        hist = hist[hist > 0]  # Remove zero bins
                        if len(hist) > 0:
                            entropy = -np.sum(hist * np.log2(hist))
                            features[f'{disagreement_name}_entropy'] = np.full(len(values), entropy)
                            feature_names.append(f'{disagreement_name}_entropy')
                            disagreement_entropy_count += 1
                    except Exception as e:
                        tprint_warning(f"⚠️ Failed to calculate entropy for disagreement {disagreement_name}: {e}")
        
        tprint_success(f"✅ Generated {disagreement_entropy_count} disagreement entropy features")
        
        tprint_structured({
            "individual_entropy_features": individual_entropy_count,
            "joint_entropy_features": joint_entropy_count,
            "disagreement_entropy_features": disagreement_entropy_count,
            "total_entropy_features": len(features)
        })
        
        tprint_success(f"🎉 Generated {len(features)} entropy features")
        
        return {
            'features': features,
            'feature_names': feature_names
        }
    
    def _generate_interaction_features(self, data: pd.DataFrame, 
                                     base_features: Dict[str, np.ndarray],
                                     base_models: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate feature interaction analysis features."""
        tprint_info("🔗 Generating interaction features")
        
        features = {}
        feature_names = []
        
        # Get base model outputs
        base_model_outputs = []
        base_model_names = []
        for model_name in base_models.keys():
            feature_name = f'base_model_{model_name}_output'
            if feature_name in base_features:
                base_model_outputs.append(base_features[feature_name])
                base_model_names.append(model_name)
        
        tprint_debug(f"Found {len(base_model_outputs)} base model outputs for interaction analysis")
        
        if len(base_model_outputs) < 2:
            tprint_warning("⚠️ Need at least 2 base model outputs for interaction features")
            return {
                'features': features,
                'feature_names': feature_names
            }
        
        # Convert to numpy array for easier computation
        base_model_array = np.array(base_model_outputs)
        tprint_data_format(base_model_array, "Base Model Array for Interaction Analysis", check_compatibility=True)
        
        # Model interaction matrix (correlation between model outputs)
        tprint_debug("Calculating model interaction matrix")
        if base_model_array.shape[0] >= 2:
            try:
                # Calculate correlation matrix between models
                model_correlations = np.corrcoef(base_model_array)
                
                # Extract upper triangular part (excluding diagonal)
                upper_tri_indices = np.triu_indices_from(model_correlations, k=1)
                interaction_strengths = model_correlations[upper_tri_indices]
                
                # Average interaction strength
                avg_interaction_strength = np.mean(interaction_strengths)
                features['avg_interaction_strength'] = np.full(base_model_array.shape[1], avg_interaction_strength)
                feature_names.append('avg_interaction_strength')
                
                # Maximum interaction strength
                max_interaction_strength = np.max(interaction_strengths)
                features['max_interaction_strength'] = np.full(base_model_array.shape[1], max_interaction_strength)
                feature_names.append('max_interaction_strength')
                
                # Interaction diversity (standard deviation of correlations)
                interaction_diversity = np.std(interaction_strengths)
                features['interaction_diversity'] = np.full(base_model_array.shape[1], interaction_diversity)
                feature_names.append('interaction_diversity')
                
                tprint_success("✅ Generated model interaction matrix features")
                
            except Exception as e:
                tprint_warning(f"⚠️ Failed to calculate model interaction matrix: {e}")
        
        # Principal Component Analysis of disagreements
        tprint_debug("Calculating PCA of disagreement patterns")
        if base_model_array.shape[0] >= 2:
            try:
                from sklearn.decomposition import PCA
                
                # Calculate pairwise disagreements
                pairwise_disagreements = []
                for i in range(len(base_model_names)):
                    for j in range(i + 1, len(base_model_names)):
                        disagreement = np.abs(base_model_array[i] - base_model_array[j])
                        pairwise_disagreements.append(disagreement)
                
                if pairwise_disagreements:
                    disagreement_matrix = np.array(pairwise_disagreements).T
                    
                    # Apply PCA to disagreement patterns
                    pca = PCA(n_components=min(3, disagreement_matrix.shape[1]))
                    pca_components = pca.fit_transform(disagreement_matrix)
                    
                    # First principal component (main disagreement pattern)
                    features['pca_disagreement_1'] = pca_components[:, 0]
                    feature_names.append('pca_disagreement_1')
                    
                    # Second principal component (secondary disagreement pattern)
                    if pca_components.shape[1] > 1:
                        features['pca_disagreement_2'] = pca_components[:, 1]
                        feature_names.append('pca_disagreement_2')
                    
                    # Explained variance ratio
                    features['pca_explained_variance_ratio'] = np.full(
                        base_model_array.shape[1], 
                        np.sum(pca.explained_variance_ratio_)
                    )
                    feature_names.append('pca_explained_variance_ratio')
                    
                    tprint_success(f"✅ Generated PCA features with {pca_components.shape[1]} components")
                    
            except ImportError:
                tprint_warning("⚠️ sklearn not available, skipping PCA features")
            except Exception as e:
                tprint_warning(f"⚠️ Failed to calculate PCA features: {e}")
        
        # Mutual information between model outputs
        tprint_debug("Calculating mutual information between model outputs")
        if base_model_array.shape[0] >= 2:
            mutual_info_scores = []
            for i in range(len(base_model_names)):
                for j in range(i + 1, len(base_model_names)):
                    # Calculate mutual information between model outputs
                    mi_score = self._calculate_mutual_information(
                        base_model_array[i], base_model_array[j]
                    )
                    mutual_info_scores.append(mi_score)
            
            if mutual_info_scores:
                # Average mutual information
                avg_mutual_info = np.mean(mutual_info_scores)
                features['avg_mutual_information'] = np.full(base_model_array.shape[1], avg_mutual_info)
                feature_names.append('avg_mutual_information')
                
                # Maximum mutual information
                max_mutual_info = np.max(mutual_info_scores)
                features['max_mutual_information'] = np.full(base_model_array.shape[1], max_mutual_info)
                feature_names.append('max_mutual_information')
                
                tprint_success(f"✅ Generated mutual information features (avg: {avg_mutual_info:.4f}, max: {max_mutual_info:.4f})")
        
        # Model complementarity score
        tprint_debug("Calculating model complementarity score")
        if base_model_array.shape[0] >= 2:
            try:
                # Calculate how much models disagree (complementarity)
                disagreement_matrix = np.zeros((len(base_model_names), len(base_model_names)))
                for i in range(len(base_model_names)):
                    for j in range(len(base_model_names)):
                        if i != j:
                            disagreement = np.mean(np.abs(base_model_array[i] - base_model_array[j]))
                            disagreement_matrix[i, j] = disagreement
                
                # Complementarity is the average disagreement
                complementarity = np.mean(disagreement_matrix[disagreement_matrix > 0])
                features['model_complementarity'] = np.full(base_model_array.shape[1], complementarity)
                feature_names.append('model_complementarity')
                
                tprint_success(f"✅ Generated model complementarity score: {complementarity:.4f}")
                
            except Exception as e:
                tprint_warning(f"⚠️ Failed to calculate model complementarity: {e}")
        
        tprint_structured({
            "total_interaction_features": len(features),
            "base_model_count": len(base_model_outputs),
            "pca_components": len([f for f in feature_names if 'pca' in f]),
            "mutual_info_features": len([f for f in feature_names if 'mutual' in f])
        })
        
        tprint_success(f"🎉 Generated {len(features)} interaction features")
        
        return {
            'features': features,
            'feature_names': feature_names
        }
    
    def _calculate_mutual_information(self, x: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
        """Calculate mutual information between two arrays."""
        tprint_debug(f"Calculating mutual information between arrays of length {len(x)}")
        try:
            # Create histograms
            hist_2d, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
            
            # Normalize to get probabilities
            p_xy = hist_2d / np.sum(hist_2d)
            
            # Marginal probabilities
            p_x = np.sum(p_xy, axis=1)
            p_y = np.sum(p_xy, axis=0)
            
            # Calculate mutual information
            mi = 0.0
            for i in range(len(p_x)):
                for j in range(len(p_y)):
                    if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                        mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
            
            tprint_debug(f"Mutual information calculated: {mi:.4f}")
            return mi
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to calculate mutual information: {e}")
            # Fallback: return 0 if calculation fails
            return 0.0
    
    def _get_feature_category_breakdown(self, features: Dict[str, np.ndarray]) -> Dict[str, int]:
        """Get breakdown of meta-features by category."""
        tprint_debug(f"Categorizing {len(features)} features")
        
        breakdown = {
            'base_outputs': 0,
            'disagreement': 0,
            'entropy': 0,
            'interaction': 0,
            'other': 0
        }
        
        for feature_name in features.keys():
            if 'base_model' in feature_name.lower() and 'output' in feature_name.lower():
                breakdown['base_outputs'] += 1
            elif 'disagreement' in feature_name.lower():
                breakdown['disagreement'] += 1
            elif 'entropy' in feature_name.lower():
                breakdown['entropy'] += 1
            elif any(keyword in feature_name.lower() for keyword in ['interaction', 'pca', 'mutual', 'complementarity']):
                breakdown['interaction'] += 1
            else:
                breakdown['other'] += 1
        
        tprint_structured({
            "feature_categorization": breakdown,
            "total_features": len(features)
        })
        
        return breakdown
    
    def _assess_ensemble_readiness(self, features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Assess how well-suited the meta-features are for ensemble training."""
        tprint_info("🔍 Assessing ensemble readiness")
        
        if not features:
            tprint_error("❌ No features available for ensemble readiness assessment")
            return {'score': 0, 'issues': ['No features available']}
        
        issues = []
        score = 100
        
        # Check feature count
        feature_count = len(features)
        tprint_debug(f"Feature count: {feature_count}")
        if feature_count < 3:  # Minimum: base outputs + disagreement + entropy
            issues.append(f'Too few meta-features: {feature_count} < 3')
            score -= 50
            tprint_warning(f"⚠️ Too few meta-features: {feature_count} < 3")
        
        # Check meta-feature presence
        category_breakdown = self._get_feature_category_breakdown(features)
        
        # Must have base model outputs
        if category_breakdown['base_outputs'] == 0:
            issues.append('No base model outputs found')
            score -= 40
            tprint_error("❌ No base model outputs found")
        else:
            tprint_success(f"✅ Found {category_breakdown['base_outputs']} base model outputs")
        
        # Should have disagreement features
        if category_breakdown['disagreement'] == 0:
            issues.append('No disagreement features found')
            score -= 20
            tprint_warning("⚠️ No disagreement features found")
        else:
            tprint_success(f"✅ Found {category_breakdown['disagreement']} disagreement features")
        
        # Should have entropy features
        if category_breakdown['entropy'] == 0:
            issues.append('No entropy features found')
            score -= 20
            tprint_warning("⚠️ No entropy features found")
        else:
            tprint_success(f"✅ Found {category_breakdown['entropy']} entropy features")
        
        # Should have interaction features
        if category_breakdown['interaction'] == 0:
            issues.append('No interaction features found')
            score -= 15
            tprint_warning("⚠️ No interaction features found")
        else:
            tprint_success(f"✅ Found {category_breakdown['interaction']} interaction features")
        
        # Check feature quality
        tprint_debug("Checking feature quality")
        quality_issues = 0
        for name, values in features.items():
            if len(values) == 0:
                quality_issues += 1
                tprint_warning(f"⚠️ Empty feature: {name}")
            elif np.all(np.isnan(values)):
                quality_issues += 1
                tprint_warning(f"⚠️ All-NaN feature: {name}")
            elif np.all(values == values[0]):  # All same value
                quality_issues += 1
                tprint_warning(f"⚠️ Constant feature: {name}")
        
        if quality_issues > 0:
            issues.append(f'{quality_issues} features have quality issues')
            score -= quality_issues * 10
            tprint_warning(f"⚠️ {quality_issues} features have quality issues")
        else:
            tprint_success("✅ All features passed quality checks")
        
        final_score = max(0, score)
        
        if final_score >= 80:
            tprint_success(f"🎉 Excellent ensemble readiness: {final_score}/100")
        elif final_score >= 60:
            tprint_info(f"✅ Good ensemble readiness: {final_score}/100")
        elif final_score >= 40:
            tprint_warning(f"⚠️ Fair ensemble readiness: {final_score}/100")
        else:
            tprint_error(f"❌ Poor ensemble readiness: {final_score}/100")
        
        tprint_structured({
            "ensemble_readiness_score": final_score,
            "feature_count": feature_count,
            "quality_issues": quality_issues,
            "issues": issues,
            "category_breakdown": category_breakdown
        })
        
        return {
            'score': final_score,
            'issues': issues,
            'feature_count': feature_count,
            'base_outputs': category_breakdown['base_outputs'],
            'disagreement_features': category_breakdown['disagreement'],
            'entropy_features': category_breakdown['entropy'],
            'interaction_features': category_breakdown['interaction'],
            'quality_issues': quality_issues
        }
    
    def prepare_data_for_ensemble_training(self, data: pd.DataFrame, 
                                         base_models: Optional[Dict[str, Any]] = None,
                                         target_column: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
        """
        Prepare data for ensemble training with meta-features.
        
        Args:
            data: Market data DataFrame
            base_models: Dictionary of trained base models (required)
            target_column: Name of target column (if None, will create synthetic target)
            
        Returns:
            Tuple of (X, y, feature_names, metadata)
        """
        tprint_info("🔧 Preparing data for ensemble training")
        tprint_data_preview(data, "Input Data for Training", max_rows=3, max_cols=5)
        
        if base_models is None:
            tprint_error("❌ Base models must be provided for ensemble training")
            raise ValueError("Base models must be provided for ensemble training")
        
        tprint_info(f"📊 Using {len(base_models)} base models: {list(base_models.keys())}")
        
        # Get meta-features
        with tprint_timer("Meta-feature generation"):
            feature_result = self.get_comprehensive_ensemble_features(data, base_models)
        features = feature_result['features']
        feature_names = feature_result['feature_names']
        
        if not features:
            tprint_warning("⚠️ No features generated, returning empty arrays")
            return np.array([]).reshape(len(data), 0), np.array([]), [], feature_result
        
        tprint_info(f"📈 Generated {len(features)} features for training")
        
        # Convert to numpy array
        tprint_debug("Converting features to numpy array")
        X = np.column_stack([features[name] for name in feature_names])
        tprint_data_format(X, "Feature Matrix X", check_compatibility=True)
        
        # Handle NaN values
        tprint_debug("Handling NaN values in feature matrix")
        nan_count_before = np.isnan(X).sum()
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        nan_count_after = np.isnan(X).sum()
        tprint_success(f"✅ Handled NaN values: {nan_count_before} → {nan_count_after}")
        
        # Create or get target variable
        if target_column and target_column in data.columns:
            tprint_info(f"📊 Using target column: {target_column}")
            y = data[target_column].values
            tprint_data_format(y, "Target Variable y", check_compatibility=True)
        else:
            tprint_info("🎯 Creating synthetic target variable")
            y = self._create_synthetic_target(data)
            tprint_data_format(y, "Synthetic Target Variable y", check_compatibility=True)
        
        # Ensure target has same length as features
        min_length = min(len(X), len(y))
        if len(X) != len(y):
            tprint_warning(f"⚠️ Mismatched lengths: X={len(X)}, y={len(y)}, using min={min_length}")
        X = X[:min_length]
        y = y[:min_length]
        
        # Add preprocessing metadata
        metadata = feature_result.copy()
        metadata.update({
            'preprocessing': {
                'nan_handled': True,
                'feature_matrix_shape': X.shape,
                'target_length': len(y),
                'base_models_used': base_models is not None
            }
        })
        
        tprint_structured({
            "feature_matrix_shape": X.shape,
            "target_length": len(y),
            "feature_names_count": len(feature_names),
            "nan_handled": True
        })
        
        tprint_success(f"🎉 Data preparation completed: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y, feature_names, metadata
    
    def _create_synthetic_target(self, data: pd.DataFrame) -> np.ndarray:
        """Create synthetic target for ensemble training (future returns)."""
        tprint_info("🎯 Creating synthetic target variable")
        
        if 'close' in data.columns:
            tprint_debug("Using close prices to create future returns target")
            prices = data['close']
            # Create future returns as target
            future_returns = prices.pct_change().shift(-1).fillna(0)
            target_values = future_returns.values
            tprint_success(f"✅ Created future returns target: {len(target_values)} values")
            tprint_data_format(target_values, "Future Returns Target", check_compatibility=True)
            return target_values
        else:
            tprint_warning("⚠️ No 'close' column found, creating random target")
            # Fallback: create random target
            random_target = np.random.randn(len(data))
            tprint_success(f"✅ Created random target: {len(random_target)} values")
            return random_target
    
    def train_enhanced_ensemble(self, data: pd.DataFrame, 
                              base_models: Optional[Dict[str, Any]] = None,
                              target_column: Optional[str] = None,
                              ensemble_type: str = 'voting',
                              test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train enhanced ensemble with meta-features.
        
        Args:
            data: Market data DataFrame
            base_models: Dictionary of trained base models (required)
            target_column: Name of target column
            ensemble_type: Type of ensemble ('voting', 'stacking')
            test_size: Fraction of data to use for testing
            
        Returns:
            Dictionary containing trained ensemble and results
        """
        tprint_info("🚀 Starting enhanced ensemble training")
        tprint_structured({
            "ensemble_type": ensemble_type,
            "test_size": test_size,
            "target_column": target_column,
            "base_models_count": len(base_models) if base_models else 0
        })
        
        if not SKLEARN_AVAILABLE:
            tprint_error("❌ Scikit-learn not available. Install with: pip install scikit-learn")
            raise ImportError("Scikit-learn not available. Install with: pip install scikit-learn")
        
        if base_models is None:
            tprint_error("❌ Base models must be provided for ensemble training")
            raise ValueError("Base models must be provided for ensemble training")
        
        # Prepare data
        with tprint_timer("Data preparation"):
            X, y, feature_names, metadata = self.prepare_data_for_ensemble_training(data, base_models, target_column)
        
        if X.size == 0:
            tprint_error("❌ No features available for ensemble training")
            raise ValueError("No features available for ensemble training")
        
        tprint_success(f"✅ Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Split data
        tprint_info(f"📊 Splitting data: {test_size:.1%} for testing")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        tprint_success(f"✅ Data split: {len(X_train)} train, {len(X_test)} test samples")
        
        # Create ensemble
        tprint_info(f"🔧 Creating {ensemble_type} ensemble")
        if ensemble_type == 'voting':
            ensemble = self._create_voting_ensemble()
        elif ensemble_type == 'stacking':
            ensemble = self._create_stacking_ensemble()
        else:
            tprint_error(f"❌ Unknown ensemble type: {ensemble_type}")
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")
        
        tprint_success(f"✅ Created {ensemble_type} ensemble")
        
        # Train ensemble
        tprint_info("🎓 Training ensemble model")
        with tprint_timer("Ensemble training"):
            ensemble.fit(X_train, y_train)
        tprint_success("✅ Ensemble training completed")
        
        # Make predictions
        tprint_info("🔮 Making predictions")
        with tprint_timer("Prediction generation"):
            y_pred_train = ensemble.predict(X_train)
            y_pred_test = ensemble.predict(X_test)
        tprint_success("✅ Predictions generated")
        
        # Calculate metrics
        tprint_info("📊 Calculating performance metrics")
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        tprint_success(f"✅ Metrics calculated: Train R²={train_r2:.4f}, Test R²={test_r2:.4f}")
        
        # Cross-validation score
        tprint_info("🔄 Performing cross-validation")
        try:
            with tprint_timer("Cross-validation"):
                cv_scores = cross_val_score(ensemble, X_train, y_train, cv=5, scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            tprint_success(f"✅ Cross-validation completed: CV R²={cv_mean:.4f}±{cv_std:.4f}")
        except Exception as e:
            tprint_warning(f"⚠️ Cross-validation failed: {e}")
            cv_mean = 0.0
            cv_std = 0.0
        
        # Check for overfitting
        overfitting = test_r2 < train_r2 - 0.1
        if overfitting:
            tprint_warning(f"⚠️ Potential overfitting detected: Test R² ({test_r2:.4f}) < Train R² ({train_r2:.4f}) - 0.1")
        else:
            tprint_success("✅ No significant overfitting detected")
        
        results = {
            'ensemble': ensemble,
            'feature_names': feature_names,
            'metadata': metadata,
            'metrics': {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'cv_r2_mean': cv_mean,
                'cv_r2_std': cv_std,
                'overfitting': overfitting
            },
            'data_info': {
                'train_size': len(X_train),
                'test_size': len(X_test),
                'n_features': X.shape[1]
            }
        }
        
        tprint_structured({
            "ensemble_training_results": {
                "ensemble_type": ensemble_type,
                "train_r2": train_r2,
                "test_r2": test_r2,
                "cv_r2_mean": cv_mean,
                "overfitting": overfitting,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "n_features": X.shape[1]
            }
        })
        
        tprint_success(f"🎉 Enhanced ensemble training completed successfully!")
        
        return results
    
    def _create_voting_ensemble(self) -> Any:
        """Create voting ensemble."""
        tprint_info("🗳️ Creating voting ensemble")
        
        estimators = []
        
        # Add base estimators
        if LGBM_AVAILABLE:
            estimators.append(('lgbm', lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)))
            tprint_success("✅ Added LGBM regressor")
        else:
            tprint_warning("⚠️ LGBM not available, skipping LGBM regressor")
        
        estimators.extend([
            ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42)),
            ('ridge', Ridge(alpha=1.0))
        ])
        
        tprint_success(f"✅ Added {len(estimators)} estimators to voting ensemble")
        tprint_structured({
            "voting_ensemble_estimators": [name for name, _ in estimators]
        })
        
        return VotingRegressor(estimators)
    
    def _create_stacking_ensemble(self) -> Any:
        """Create stacking ensemble."""
        tprint_info("📚 Creating stacking ensemble")
        
        base_estimators = []
        
        # Add base estimators
        if LGBM_AVAILABLE:
            base_estimators.append(('lgbm', lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)))
            tprint_success("✅ Added LGBM regressor as base estimator")
        else:
            tprint_warning("⚠️ LGBM not available, skipping LGBM regressor")
        
        base_estimators.extend([
            ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42))
        ])
        
        tprint_success(f"✅ Added {len(base_estimators)} base estimators")
        
        # Meta-learner
        meta_learner = LinearRegression()
        tprint_success("✅ Created LinearRegression meta-learner")
        
        tprint_structured({
            "stacking_ensemble_config": {
                "base_estimators": [name for name, _ in base_estimators],
                "meta_learner": "LinearRegression",
                "cv_folds": 5
            }
        })
        
        return StackingRegressor(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=5
        )


# Convenience functions
def get_enhanced_ensemble_features(data: pd.DataFrame, base_models: Dict[str, Any]) -> Dict[str, Any]:
    """Get meta-features for ensemble training."""
    tprint_info("🔧 Convenience function: get_enhanced_ensemble_features")
    tprint_structured({
        "data_shape": data.shape,
        "base_models_count": len(base_models),
        "base_models": list(base_models.keys())
    })
    
    integrator = EnhancedEnsembleTrainingIntegration()
    result = integrator.get_comprehensive_ensemble_features(data, base_models)
    
    tprint_success(f"✅ Convenience function completed: {result['feature_count']} features generated")
    return result


def train_enhanced_ensemble(data: pd.DataFrame, base_models: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Train enhanced ensemble with meta-features."""
    tprint_info("🚀 Convenience function: train_enhanced_ensemble")
    tprint_structured({
        "data_shape": data.shape,
        "base_models_count": len(base_models),
        "base_models": list(base_models.keys()),
        "kwargs": kwargs
    })
    
    integrator = EnhancedEnsembleTrainingIntegration()
    result = integrator.train_enhanced_ensemble(data, base_models, **kwargs)
    
    tprint_success(f"✅ Convenience function completed: R²={result['metrics']['test_r2']:.4f}")
    return result


__all__ = [
    'EnhancedEnsembleTrainingIntegration',
    'get_enhanced_ensemble_features',
    'train_enhanced_ensemble'
]