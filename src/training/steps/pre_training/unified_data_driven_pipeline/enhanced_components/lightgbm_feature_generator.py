"""
LightGBM/CatBoost + Featuretools Deep Feature Synthesis System

This module implements a sophisticated feature generation system using LightGBM/CatBoost
for SHAP interactions and Featuretools Deep Feature Synthesis for relational and 
time-based features, replacing the RandomForest/SHAP system with a more advanced approach.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import gc

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

# Import LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    tprint_warning("⚠️ LightGBM not available, using CatBoost fallback")

# Import CatBoost
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    tprint_warning("⚠️ CatBoost not available, using LightGBM fallback")

# Import Featuretools
try:
    import featuretools as ft
    FEATURETOOLS_AVAILABLE = True
except ImportError:
    FEATURETOOLS_AVAILABLE = False
    tprint_warning("⚠️ Featuretools not available, using fallback feature generation")

# Import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    tprint_warning("⚠️ SHAP not available, using fallback feature importance")

# Import ALE (Accumulated Local Effects)
try:
    from alibi.explainers import ALE
    ALE_AVAILABLE = True
except ImportError:
    ALE_AVAILABLE = False
    tprint_warning("⚠️ ALE not available, using SHAP fallback")

# Import scikit-learn for fallbacks
try:
    from sklearn.model_selection import cross_val_score, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    tprint_warning("⚠️ Scikit-learn not available, using fallback implementations")

# Import VectorBT optimizations
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import (
        VectorBTRollingOptimizer, 
        get_vectorbt_rolling_optimizer,
        optimized_rolling_mean,
        optimized_rolling_std,
        optimized_rolling_corr
    )
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    VectorBTRollingOptimizer = None

# Import caching
try:
    from src.feature_generation.core.feature_cache import FeatureCacheService
    from src.utils.serialization_utils import UniversalSerializer, JSONSerializer, PickleSerializer
    CACHING_AVAILABLE = True
except ImportError:
    CACHING_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class FeatureGenerationConfig:
    """Configuration for LightGBM/CatBoost + Featuretools feature generation."""
    # Model parameters
    model_type: str = 'lightgbm'  # 'lightgbm' or 'catboost'
    n_estimators: int = 100
    max_depth: int = 10
    learning_rate: float = 0.1
    random_state: int = 42
    n_jobs: int = -1
    
    # Featuretools parameters
    max_features: int = 100  # Maximum total features
    max_depth_featuretools: int = 2
    max_features_per_primitive: int = 5
    primitive_types: List[str] = None
    
    # SHAP parameters
    use_shap: bool = True
    shap_sample_size: int = 1000
    shap_explainer_type: str = 'tree'  # 'tree', 'linear', 'kernel'
    
    # ALE parameters
    use_ale: bool = True
    ale_grid_size: int = 50
    
    # Feature selection parameters
    feature_importance_threshold: float = 0.01
    correlation_threshold: float = 0.95
    
    # VectorBT optimization
    enable_vectorbt: bool = True
    enable_parallel: bool = True
    max_workers: int = 4
    memory_efficient: bool = True
    
    # Caching
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    
    def __post_init__(self):
        if self.primitive_types is None:
            self.primitive_types = ['add_numeric', 'multiply_numeric', 'divide_numeric', 
                                  'subtract_numeric', 'mean', 'std', 'min', 'max', 'count']


@dataclass
class GeneratedFeature:
    """Represents a generated feature with metadata."""
    name: str
    formula: str
    feature_series: pd.Series
    importance_score: float
    shap_values: Optional[np.ndarray] = None
    ale_values: Optional[np.ndarray] = None
    parent_features: List[str] = None
    feature_type: str = 'generated'
    generation_method: str = 'lightgbm_featuretools'
    metadata: Dict[str, Any] = None


@dataclass
class FeatureGenerationResult:
    """Result of feature generation process."""
    generated_features: List[GeneratedFeature]
    feature_importance_scores: Dict[str, float]
    model_performance: Dict[str, float]
    generation_time: float
    n_features_generated: int
    n_features_selected: int
    cache_hit_rate: float
    shap_analysis_completed: bool
    ale_analysis_completed: bool
    featuretools_features: int
    metadata: Dict[str, Any]


class LightGBMFeatureGenerator:
    """
    LightGBM/CatBoost + Featuretools Deep Feature Synthesis system.
    
    This system uses LightGBM/CatBoost models for SHAP interactions and
    Featuretools Deep Feature Synthesis for relational and time-based features.
    """
    
    def __init__(self, config: Optional[FeatureGenerationConfig] = None):
        """Initialize the LightGBM feature generator."""
        self.config = config or FeatureGenerationConfig()
        
        # Initialize components
        self._initialize_components()
        
        # Performance tracking
        self.performance_stats = {
            'total_generations': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'total_execution_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'shap_analyses': 0,
            'ale_analyses': 0,
            'featuretools_features': 0,
            'vectorbt_operations': 0
        }
        
        tprint_success("✅ LightGBM Feature Generator initialized")
    
    def _initialize_components(self):
        """Initialize all generator components."""
        tprint_debug("Initializing LightGBM feature generator components")
        
        # Initialize VectorBT optimizer
        if VECTORBT_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer()
            tprint_success("✅ VectorBT optimizer initialized")
        else:
            self.vectorbt_optimizer = None
            tprint_warning("⚠️ VectorBT not available, using fallback implementations")
        
        # Initialize caching
        if CACHING_AVAILABLE:
            self.feature_cache = FeatureCacheService(subdirectory="lightgbm_features")
            self.serializer = UniversalSerializer()
            self.json_serializer = JSONSerializer()
            self.pickle_serializer = PickleSerializer()
            tprint_success("✅ Caching initialized")
        else:
            self.feature_cache = None
            self.serializer = None
            self.json_serializer = None
            self.pickle_serializer = None
            tprint_warning("⚠️ Caching not available")
        
        # Initialize scalers
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        
        # Initialize Featuretools
        if FEATURETOOLS_AVAILABLE:
            self.es = ft.EntitySet(id="financial_data")
            tprint_success("✅ Featuretools initialized")
        else:
            self.es = None
            tprint_warning("⚠️ Featuretools not available, using fallback feature generation")
        
        tprint_success("✅ LightGBM feature generator components initialized")
    
    def generate_features(
        self,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        execution_mode: str = 'full'
    ) -> FeatureGenerationResult:
        """
        Generate features using LightGBM/CatBoost and Featuretools Deep Feature Synthesis.
        
        Args:
            data: Input data with features and targets
            target_column: Name of the target column
            feature_columns: Optional list of feature columns to use
            execution_mode: Execution mode ('light', 'full', 'blank')
            
        Returns:
            FeatureGenerationResult with generated features and metadata
        """
        tprint_info(f"🚀 Starting LightGBM + Featuretools feature generation for {execution_mode} mode")
        tprint_debug(f"📊 Data shape: {data.shape}, Target: {target_column}")
        
        start_time = time.time()
        
        try:
            # Prepare data
            prepared_data = self._prepare_data(data, target_column, feature_columns)
            if prepared_data is None:
                return self._create_failed_result("Data preparation failed")
            
            # Generate base features
            tprint_info("🔧 Generating base features")
            base_features = self._generate_base_features(prepared_data, execution_mode)
            
            # Train LightGBM/CatBoost model
            tprint_info("🌲 Training LightGBM/CatBoost model")
            model, feature_importance = self._train_model(
                base_features, prepared_data[target_column]
            )
            
            # Generate Featuretools features
            tprint_info("⚡ Generating Featuretools Deep Feature Synthesis features")
            featuretools_features = self._generate_featuretools_features(
                prepared_data, target_column, execution_mode
            )
            
            # Combine base and featuretools features
            all_features = self._combine_features(base_features, featuretools_features)
            
            # SHAP analysis
            shap_values = None
            if self.config.use_shap and SHAP_AVAILABLE:
                tprint_info("🔍 Performing SHAP analysis")
                shap_values = self._perform_shap_analysis(model, all_features)
            
            # ALE analysis
            ale_values = None
            if self.config.use_ale and ALE_AVAILABLE:
                tprint_info("📊 Performing ALE analysis")
                ale_values = self._perform_ale_analysis(model, all_features)
            
            # Select best features (limit to max_features)
            tprint_info("🎯 Selecting best features")
            selected_features = self._select_best_features(
                all_features, feature_importance, shap_values, ale_values
            )
            
            # Create generated features
            generated_features = self._create_generated_features(
                selected_features, all_features, shap_values, ale_values
            )
            
            # Calculate performance metrics
            model_performance = self._calculate_model_performance(
                model, all_features, prepared_data[target_column]
            )
            
            # Create result
            generation_time = time.time() - start_time
            result = FeatureGenerationResult(
                generated_features=generated_features,
                feature_importance_scores=feature_importance,
                model_performance=model_performance,
                generation_time=generation_time,
                n_features_generated=len(all_features.columns),
                n_features_selected=len(selected_features),
                cache_hit_rate=self.performance_stats['cache_hits'] / max(1, 
                    self.performance_stats['cache_hits'] + self.performance_stats['cache_misses']),
                shap_analysis_completed=shap_values is not None,
                ale_analysis_completed=ale_values is not None,
                featuretools_features=len(featuretools_features.columns) if featuretools_features is not None else 0,
                metadata={
                    'execution_mode': execution_mode,
                    'n_base_features': len(base_features.columns),
                    'model_type': self.config.model_type,
                    'shap_available': SHAP_AVAILABLE,
                    'ale_available': ALE_AVAILABLE,
                    'featuretools_available': FEATURETOOLS_AVAILABLE,
                    'vectorbt_available': VECTORBT_AVAILABLE
                }
            )
            
            # Update performance stats
            self.performance_stats.update({
                'total_generations': 1,
                'successful_generations': 1,
                'total_execution_time': generation_time,
                'shap_analyses': 1 if shap_values is not None else 0,
                'ale_analyses': 1 if ale_values is not None else 0,
                'featuretools_features': len(featuretools_features.columns) if featuretools_features is not None else 0
            })
            
            tprint_success(f"✅ Feature generation completed in {generation_time:.3f}s")
            tprint_info(f"📊 Generated {len(generated_features)} features from {len(all_features.columns)} total features")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Feature generation failed: {e}")
            self.performance_stats['failed_generations'] += 1
            return self._create_failed_result(str(e))
    
    def _prepare_data(
        self, 
        data: pd.DataFrame, 
        target_column: str, 
        feature_columns: Optional[List[str]]
    ) -> Optional[pd.DataFrame]:
        """Prepare data for feature generation."""
        try:
            # Select feature columns
            if feature_columns:
                available_columns = [col for col in feature_columns if col in data.columns]
                if not available_columns:
                    tprint_error("❌ No valid feature columns found")
                    return None
                feature_data = data[available_columns + [target_column]]
            else:
                # Use all numeric columns except target
                numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
                if target_column in numeric_columns:
                    numeric_columns.remove(target_column)
                feature_data = data[numeric_columns + [target_column]]
            
            # Remove rows with NaN values
            feature_data = feature_data.dropna()
            
            if len(feature_data) < 10:
                tprint_error("❌ Insufficient data after cleaning")
                return None
            
            tprint_debug(f"📊 Prepared data shape: {feature_data.shape}")
            return feature_data
            
        except Exception as e:
            tprint_error(f"❌ Data preparation failed: {e}")
            return None
    
    def _generate_base_features(
        self, 
        data: pd.DataFrame, 
        execution_mode: str
    ) -> pd.DataFrame:
        """Generate base features from the data."""
        try:
            base_features = pd.DataFrame(index=data.index)
            
            # Get feature columns (exclude target)
            feature_columns = [col for col in data.columns if col != data.columns[-1]]  # Assume last column is target
            
            # Generate basic technical indicators
            for col in feature_columns:
                if col in data.columns:
                    series = data[col]
                    
                    # Moving averages
                    for window in [5, 10, 20, 50]:
                        if VECTORBT_AVAILABLE and self.vectorbt_optimizer:
                            ma = self.vectorbt_optimizer.rolling_mean(series, window=window)
                        else:
                            ma = series.rolling(window=window).mean()
                        base_features[f"{col}_ma_{window}"] = ma
                    
                    # Rolling standard deviation
                    for window in [10, 20]:
                        if VECTORBT_AVAILABLE and self.vectorbt_optimizer:
                            std = self.vectorbt_optimizer.rolling_std(series, window=window)
                        else:
                            std = series.rolling(window=window).std()
                        base_features[f"{col}_std_{window}"] = std
                    
                    # Price ratios
                    if 'close' in col.lower():
                        for window in [5, 10, 20]:
                            if VECTORBT_AVAILABLE and self.vectorbt_optimizer:
                                ma = self.vectorbt_optimizer.rolling_mean(series, window=window)
                            else:
                                ma = series.rolling(window=window).mean()
                            base_features[f"{col}_ratio_{window}"] = series / ma
            
            # Remove columns with all NaN values
            base_features = base_features.dropna(axis=1, how='all')
            
            # Limit features based on execution mode
            if execution_mode == 'light':
                base_features = base_features.iloc[:, :30]  # Limit to 30 features
            elif execution_mode == 'blank':
                base_features = base_features.iloc[:, :15]  # Limit to 15 features
            
            tprint_debug(f"📊 Generated {len(base_features.columns)} base features")
            return base_features
            
        except Exception as e:
            tprint_error(f"❌ Base feature generation failed: {e}")
            return pd.DataFrame()
    
    def _train_model(
        self, 
        features: pd.DataFrame, 
        target: pd.Series
    ) -> Tuple[Any, Dict[str, float]]:
        """Train LightGBM or CatBoost model and get feature importance."""
        try:
            # Prepare data
            X = features.fillna(0)  # Fill NaN with 0
            y = target.fillna(0)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            
            # Choose model type
            if self.config.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
                model = lgb.LGBMRegressor(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    learning_rate=self.config.learning_rate,
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs,
                    verbose=-1
                )
            elif self.config.model_type == 'catboost' and CATBOOST_AVAILABLE:
                model = cb.CatBoostRegressor(
                    iterations=self.config.n_estimators,
                    depth=self.config.max_depth,
                    learning_rate=self.config.learning_rate,
                    random_seed=self.config.random_state,
                    verbose=False
                )
            else:
                # Fallback to LightGBM or CatBoost based on availability
                if LIGHTGBM_AVAILABLE:
                    model = lgb.LGBMRegressor(
                        n_estimators=self.config.n_estimators,
                        max_depth=self.config.max_depth,
                        learning_rate=self.config.learning_rate,
                        random_state=self.config.random_state,
                        n_jobs=self.config.n_jobs,
                        verbose=-1
                    )
                elif CATBOOST_AVAILABLE:
                    model = cb.CatBoostRegressor(
                        iterations=self.config.n_estimators,
                        depth=self.config.max_depth,
                        learning_rate=self.config.learning_rate,
                        random_seed=self.config.random_state,
                        verbose=False
                    )
                else:
                    tprint_error("❌ Neither LightGBM nor CatBoost available")
                    return None, {}
            
            model.fit(X_scaled, y)
            
            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(X.columns, model.feature_importances_))
            else:
                # Fallback for models without feature_importances_
                feature_importance = {col: 1.0 for col in X.columns}
            
            tprint_success(f"✅ {self.config.model_type.upper()} model trained successfully")
            return model, feature_importance
            
        except Exception as e:
            tprint_error(f"❌ Model training failed: {e}")
            return None, {}
    
    def _generate_featuretools_features(
        self, 
        data: pd.DataFrame, 
        target_column: str,
        execution_mode: str
    ) -> Optional[pd.DataFrame]:
        """Generate features using Featuretools Deep Feature Synthesis."""
        try:
            if not FEATURETOOLS_AVAILABLE or self.es is None:
                tprint_warning("⚠️ Featuretools not available, skipping Deep Feature Synthesis")
                return None
            
            # Prepare data for Featuretools
            feature_data = data.copy()
            
            # Add time index if not present
            if 'time_index' not in feature_data.columns:
                feature_data['time_index'] = pd.date_range(start='2020-01-01', periods=len(feature_data), freq='D')
            
            # Create entity set
            es = ft.EntitySet(id="financial_data")
            
            # Add entity
            es = es.add_dataframe(
                dataframe_name="financial_entity",
                dataframe=feature_data,
                index="id",
                time_index="time_index"
            )
            
            # Define primitives
            primitives = []
            for primitive_type in self.config.primitive_types:
                try:
                    if primitive_type in ['add_numeric', 'multiply_numeric', 'divide_numeric', 'subtract_numeric']:
                        primitives.append(getattr(ft.primitives, primitive_type)())
                    elif primitive_type in ['mean', 'std', 'min', 'max', 'count']:
                        primitives.append(getattr(ft.primitives, primitive_type)())
                except AttributeError:
                    tprint_debug(f"Primitive {primitive_type} not available")
                    continue
            
            if not primitives:
                tprint_warning("⚠️ No valid primitives found, using default set")
                primitives = [
                    ft.primitives.AddNumeric(),
                    ft.primitives.MultiplyNumeric(),
                    ft.primitives.Mean(),
                    ft.primitives.Std(),
                    ft.primitives.Min(),
                    ft.primitives.Max()
                ]
            
            # Generate features
            feature_matrix, feature_defs = ft.dfs(
                entityset=es,
                target_dataframe_name="financial_entity",
                max_depth=self.config.max_depth_featuretools,
                max_features=self.config.max_features_per_primitive,
                primitive=primitives,
                verbose=False
            )
            
            # Remove target column and time index
            feature_matrix = feature_matrix.drop(columns=[target_column, 'time_index'], errors='ignore')
            
            # Limit features based on execution mode
            if execution_mode == 'light':
                feature_matrix = feature_matrix.iloc[:, :20]  # Limit to 20 features
            elif execution_mode == 'blank':
                feature_matrix = feature_matrix.iloc[:, :10]  # Limit to 10 features
            
            tprint_debug(f"📊 Generated {len(feature_matrix.columns)} Featuretools features")
            return feature_matrix
            
        except Exception as e:
            tprint_warning(f"⚠️ Featuretools feature generation failed: {e}")
            return None
    
    def _combine_features(
        self, 
        base_features: pd.DataFrame, 
        featuretools_features: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Combine base features and Featuretools features."""
        try:
            if featuretools_features is not None:
                # Align indices
                common_index = base_features.index.intersection(featuretools_features.index)
                base_features_aligned = base_features.loc[common_index]
                featuretools_features_aligned = featuretools_features.loc[common_index]
                
                # Combine features
                combined_features = pd.concat([base_features_aligned, featuretools_features_aligned], axis=1)
            else:
                combined_features = base_features
            
            # Remove duplicate columns
            combined_features = combined_features.loc[:, ~combined_features.columns.duplicated()]
            
            tprint_debug(f"📊 Combined features shape: {combined_features.shape}")
            return combined_features
            
        except Exception as e:
            tprint_error(f"❌ Feature combination failed: {e}")
            return base_features
    
    def _perform_shap_analysis(
        self, 
        model: Any, 
        features: pd.DataFrame
    ) -> Optional[np.ndarray]:
        """Perform SHAP analysis on the trained model."""
        try:
            if not SHAP_AVAILABLE or model is None:
                return None
            
            # Sample data for SHAP analysis (to avoid memory issues)
            sample_size = min(self.config.shap_sample_size, len(features))
            sample_indices = np.random.choice(len(features), size=sample_size, replace=False)
            X_sample = features.iloc[sample_indices].fillna(0)
            
            # Create SHAP explainer
            if self.config.shap_explainer_type == 'tree':
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.Explainer(model)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_sample)
            
            self.performance_stats['shap_analyses'] += 1
            tprint_success("✅ SHAP analysis completed")
            return shap_values
            
        except Exception as e:
            tprint_warning(f"⚠️ SHAP analysis failed: {e}")
            return None
    
    def _perform_ale_analysis(
        self, 
        model: Any, 
        features: pd.DataFrame
    ) -> Optional[Dict[str, np.ndarray]]:
        """Perform ALE (Accumulated Local Effects) analysis on the trained model."""
        try:
            if not ALE_AVAILABLE or model is None:
                return None
            
            # Sample data for ALE analysis
            sample_size = min(500, len(features))  # ALE is more memory intensive
            sample_indices = np.random.choice(len(features), size=sample_size, replace=False)
            X_sample = features.iloc[sample_indices].fillna(0)
            
            # Create ALE explainer
            ale_explainer = ALE(model.predict, feature_names=features.columns.tolist())
            
            # Calculate ALE values
            ale_values = ale_explainer.explain(X_sample.values)
            
            self.performance_stats['ale_analyses'] += 1
            tprint_success("✅ ALE analysis completed")
            return ale_values
            
        except Exception as e:
            tprint_warning(f"⚠️ ALE analysis failed: {e}")
            return None
    
    def _select_best_features(
        self, 
        features: pd.DataFrame,
        feature_importance: Dict[str, float],
        shap_values: Optional[np.ndarray],
        ale_values: Optional[Dict[str, np.ndarray]]
    ) -> List[Dict[str, Any]]:
        """Select the best features based on importance, SHAP, and ALE."""
        try:
            if features.empty:
                return []
            
            # Calculate feature scores
            feature_scores = {}
            
            for i, col in enumerate(features.columns):
                score = feature_importance.get(col, 0.0)
                
                # Add SHAP contribution if available
                if shap_values is not None and i < shap_values.shape[1]:
                    shap_contribution = np.abs(shap_values[:, i]).mean()
                    score += shap_contribution * 0.5  # Weight SHAP contribution
                
                # Add ALE contribution if available
                if ale_values is not None and col in ale_values:
                    ale_contribution = np.abs(ale_values[col]).mean()
                    score += ale_contribution * 0.3  # Weight ALE contribution
                
                feature_scores[col] = score
            
            # Sort features by score
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Select top features (limit to max_features)
            selected_features = []
            selected_names = set()
            
            for feature_name, score in sorted_features:
                if len(selected_features) >= self.config.max_features:
                    break
                
                if score > self.config.feature_importance_threshold:
                    # Check correlation with already selected features
                    is_correlated = False
                    for selected_name in selected_names:
                        try:
                            corr = features[feature_name].corr(features[selected_name])
                            if abs(corr) > self.config.correlation_threshold:
                                is_correlated = True
                                break
                        except:
                            continue
                    
                    if not is_correlated:
                        selected_features.append({
                            'name': feature_name,
                            'score': score,
                            'feature_series': features[feature_name],
                            'importance_score': feature_importance.get(feature_name, 0.0)
                        })
                        selected_names.add(feature_name)
            
            tprint_debug(f"📊 Selected {len(selected_features)} best features out of {len(features.columns)}")
            return selected_features
            
        except Exception as e:
            tprint_error(f"❌ Feature selection failed: {e}")
            # Fallback: return top features by importance
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            return [{'name': name, 'score': score, 'feature_series': features[name], 'importance_score': score} 
                   for name, score in top_features[:self.config.max_features]]
    
    def _create_generated_features(
        self, 
        selected_features: List[Dict[str, Any]], 
        all_features: pd.DataFrame,
        shap_values: Optional[np.ndarray],
        ale_values: Optional[Dict[str, np.ndarray]]
    ) -> List[GeneratedFeature]:
        """Create GeneratedFeature objects from selected features."""
        try:
            generated_features = []
            
            for feature_info in selected_features:
                feature = GeneratedFeature(
                    name=feature_info['name'],
                    formula=feature_info['name'],  # For now, use name as formula
                    feature_series=feature_info['feature_series'],
                    importance_score=feature_info['importance_score'],
                    parent_features=[feature_info['name']],  # Self-referential for now
                    feature_type='generated',
                    generation_method='lightgbm_featuretools',
                    metadata={
                        'score': feature_info['score'],
                        'shap_available': shap_values is not None,
                        'ale_available': ale_values is not None,
                        'total_features': len(all_features.columns)
                    }
                )
                generated_features.append(feature)
            
            return generated_features
            
        except Exception as e:
            tprint_error(f"❌ Generated feature creation failed: {e}")
            return []
    
    def _calculate_model_performance(
        self, 
        model: Any, 
        features: pd.DataFrame, 
        target: pd.Series
    ) -> Dict[str, float]:
        """Calculate model performance metrics."""
        try:
            if model is None:
                return {'r2_score': 0.0, 'mse': 0.0, 'mae': 0.0}
            
            # Prepare data
            X = features.fillna(0)
            y = target.fillna(0)
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            y_pred = model.predict(X_scaled)
            
            # Calculate metrics
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            
            return {
                'r2_score': r2,
                'mse': mse,
                'mae': mae,
                'rmse': np.sqrt(mse)
            }
            
        except Exception as e:
            tprint_warning(f"⚠️ Performance calculation failed: {e}")
            return {'r2_score': 0.0, 'mse': 0.0, 'mae': 0.0}
    
    def _create_failed_result(self, error_message: str) -> FeatureGenerationResult:
        """Create a failed result."""
        return FeatureGenerationResult(
            generated_features=[],
            feature_importance_scores={},
            model_performance={},
            generation_time=0.0,
            n_features_generated=0,
            n_features_selected=0,
            cache_hit_rate=0.0,
            shap_analysis_completed=False,
            ale_analysis_completed=False,
            featuretools_features=0,
            metadata={'error': error_message}
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()


def create_lightgbm_feature_generator(
    config: Optional[FeatureGenerationConfig] = None
) -> LightGBMFeatureGenerator:
    """Create a LightGBM feature generator with default configuration."""
    return LightGBMFeatureGenerator(config)