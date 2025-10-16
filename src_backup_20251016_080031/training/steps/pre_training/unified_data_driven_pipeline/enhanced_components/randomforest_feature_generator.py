"""
RandomForest/SHAP-based Feature Generation System

This module implements a sophisticated feature generation system using RandomForest
and SHAP for feature importance analysis, replacing the PID-based system with
a more robust machine learning approach.
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

# Import machine learning libraries
try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.model_selection import cross_val_score, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    tprint_warning("⚠️ Scikit-learn not available, using fallback implementations")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    tprint_warning("⚠️ SHAP not available, using fallback feature importance")

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
    """Configuration for RandomForest feature generation."""
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    max_features: str = 'sqrt'
    random_state: int = 42
    n_jobs: int = -1
    
    # Feature selection parameters
    max_features_to_select: int = 50
    feature_importance_threshold: float = 0.01
    correlation_threshold: float = 0.95
    
    # SHAP parameters
    use_shap: bool = True
    shap_sample_size: int = 1000
    shap_explainer_type: str = 'tree'  # 'tree', 'linear', 'kernel'
    
    # VectorBT optimization
    enable_vectorbt: bool = True
    enable_parallel: bool = True
    max_workers: int = 4
    memory_efficient: bool = True
    
    # Caching
    enable_caching: bool = True
    cache_ttl_hours: int = 24


@dataclass
class GeneratedFeature:
    """Represents a generated feature with metadata."""
    name: str
    formula: str
    feature_series: pd.Series
    importance_score: float
    shap_values: Optional[np.ndarray] = None
    parent_features: List[str] = None
    feature_type: str = 'generated'
    generation_method: str = 'randomforest'
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
    metadata: Dict[str, Any]


class RandomForestFeatureGenerator:
    """
    RandomForest/SHAP-based feature generation system.
    
    This system uses RandomForest models to identify important feature combinations
    and SHAP values to understand feature interactions and importance.
    """
    
    def __init__(self, config: Optional[FeatureGenerationConfig] = None):
        """Initialize the RandomForest feature generator."""
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
            'vectorbt_operations': 0
        }
        
        tprint_success("✅ RandomForest Feature Generator initialized")
    
    def _initialize_components(self):
        """Initialize all generator components."""
        tprint_debug("Initializing RandomForest feature generator components")
        
        # Initialize VectorBT optimizer
        if VECTORBT_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer()
            tprint_success("✅ VectorBT optimizer initialized")
        else:
            self.vectorbt_optimizer = None
            tprint_warning("⚠️ VectorBT not available, using fallback implementations")
        
        # Initialize caching
        if CACHING_AVAILABLE:
            self.feature_cache = FeatureCacheService(subdirectory="randomforest_features")
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
        
        tprint_success("✅ RandomForest feature generator components initialized")
    
    def generate_features(
        self,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        execution_mode: str = 'full'
    ) -> FeatureGenerationResult:
        """
        Generate features using RandomForest and SHAP analysis.
        
        Args:
            data: Input data with features and targets
            target_column: Name of the target column
            feature_columns: Optional list of feature columns to use
            execution_mode: Execution mode ('light', 'full', 'blank')
            
        Returns:
            FeatureGenerationResult with generated features and metadata
        """
        tprint_info(f"🚀 Starting RandomForest feature generation for {execution_mode} mode")
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
            
            # Train RandomForest model
            tprint_info("🌲 Training RandomForest model")
            model, feature_importance = self._train_randomforest_model(
                base_features, prepared_data[target_column]
            )
            
            # Generate feature combinations
            tprint_info("⚡ Generating feature combinations")
            feature_combinations = self._generate_feature_combinations(
                base_features, model, feature_importance
            )
            
            # SHAP analysis
            shap_values = None
            if self.config.use_shap and SHAP_AVAILABLE:
                tprint_info("🔍 Performing SHAP analysis")
                shap_values = self._perform_shap_analysis(model, base_features)
            
            # Select best features
            tprint_info("🎯 Selecting best features")
            selected_features = self._select_best_features(
                feature_combinations, feature_importance, shap_values
            )
            
            # Create generated features
            generated_features = self._create_generated_features(
                selected_features, base_features, shap_values
            )
            
            # Calculate performance metrics
            model_performance = self._calculate_model_performance(
                model, base_features, prepared_data[target_column]
            )
            
            # Create result
            generation_time = time.time() - start_time
            result = FeatureGenerationResult(
                generated_features=generated_features,
                feature_importance_scores=feature_importance,
                model_performance=model_performance,
                generation_time=generation_time,
                n_features_generated=len(feature_combinations),
                n_features_selected=len(selected_features),
                cache_hit_rate=self.performance_stats['cache_hits'] / max(1, 
                    self.performance_stats['cache_hits'] + self.performance_stats['cache_misses']),
                shap_analysis_completed=shap_values is not None,
                metadata={
                    'execution_mode': execution_mode,
                    'n_base_features': len(base_features.columns),
                    'model_type': 'RandomForest',
                    'shap_available': SHAP_AVAILABLE,
                    'vectorbt_available': VECTORBT_AVAILABLE
                }
            )
            
            # Update performance stats
            self.performance_stats.update({
                'total_generations': 1,
                'successful_generations': 1,
                'total_execution_time': generation_time,
                'shap_analyses': 1 if shap_values is not None else 0
            })
            
            tprint_success(f"✅ Feature generation completed in {generation_time:.3f}s")
            tprint_info(f"📊 Generated {len(generated_features)} features from {len(base_features.columns)} base features")
            
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
                base_features = base_features.iloc[:, :50]  # Limit to 50 features
            elif execution_mode == 'blank':
                base_features = base_features.iloc[:, :20]  # Limit to 20 features
            
            tprint_debug(f"📊 Generated {len(base_features.columns)} base features")
            return base_features
            
        except Exception as e:
            tprint_error(f"❌ Base feature generation failed: {e}")
            return pd.DataFrame()
    
    def _train_randomforest_model(
        self, 
        features: pd.DataFrame, 
        target: pd.Series
    ) -> Tuple[Any, Dict[str, float]]:
        """Train RandomForest model and get feature importance."""
        try:
            if not SKLEARN_AVAILABLE:
                tprint_warning("⚠️ Scikit-learn not available, using fallback")
                return None, {}
            
            # Prepare data
            X = features.fillna(0)  # Fill NaN with 0
            y = target.fillna(0)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            
            # Train RandomForest
            model = RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_split=self.config.min_samples_split,
                min_samples_leaf=self.config.min_samples_leaf,
                max_features=self.config.max_features,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs
            )
            
            model.fit(X_scaled, y)
            
            # Get feature importance
            feature_importance = dict(zip(X.columns, model.feature_importances_))
            
            tprint_success("✅ RandomForest model trained successfully")
            return model, feature_importance
            
        except Exception as e:
            tprint_error(f"❌ RandomForest training failed: {e}")
            return None, {}
    
    def _generate_feature_combinations(
        self, 
        features: pd.DataFrame, 
        model: Any, 
        feature_importance: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Generate feature combinations based on RandomForest importance."""
        try:
            combinations = []
            
            # Get top features by importance
            top_features = sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:self.config.max_features_to_select]
            
            top_feature_names = [name for name, _ in top_features]
            
            # Generate pairwise combinations
            for i, feat1 in enumerate(top_feature_names):
                for feat2 in top_feature_names[i+1:]:
                    try:
                        # Product interaction
                        product = features[feat1] * features[feat2]
                        if not product.isna().all():
                            combinations.append({
                                'name': f"product_{feat1}_{feat2}",
                                'formula': f"{feat1} * {feat2}",
                                'parent_features': [feat1, feat2],
                                'feature_series': product,
                                'importance_score': (feature_importance.get(feat1, 0) + 
                                                   feature_importance.get(feat2, 0)) / 2,
                                'interaction_type': 'product'
                            })
                        
                        # Ratio interaction
                        ratio = features[feat1] / (features[feat2] + 1e-8)
                        if not ratio.isna().all():
                            combinations.append({
                                'name': f"ratio_{feat1}_{feat2}",
                                'formula': f"{feat1} / ({feat2} + 1e-8)",
                                'parent_features': [feat1, feat2],
                                'feature_series': ratio,
                                'importance_score': (feature_importance.get(feat1, 0) + 
                                                   feature_importance.get(feat2, 0)) / 2,
                                'interaction_type': 'ratio'
                            })
                        
                        # Difference interaction
                        diff = features[feat1] - features[feat2]
                        if not diff.isna().all():
                            combinations.append({
                                'name': f"diff_{feat1}_{feat2}",
                                'formula': f"{feat1} - {feat2}",
                                'parent_features': [feat1, feat2],
                                'feature_series': diff,
                                'importance_score': (feature_importance.get(feat1, 0) + 
                                                   feature_importance.get(feat2, 0)) / 2,
                                'interaction_type': 'difference'
                            })
                            
                    except Exception as e:
                        tprint_debug(f"Error generating combination for {feat1}, {feat2}: {e}")
                        continue
            
            tprint_debug(f"📊 Generated {len(combinations)} feature combinations")
            return combinations
            
        except Exception as e:
            tprint_error(f"❌ Feature combination generation failed: {e}")
            return []
    
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
    
    def _select_best_features(
        self, 
        combinations: List[Dict[str, Any]], 
        feature_importance: Dict[str, float],
        shap_values: Optional[np.ndarray]
    ) -> List[Dict[str, Any]]:
        """Select the best features based on importance and correlation."""
        try:
            if not combinations:
                return []
            
            # Sort by importance score
            combinations.sort(key=lambda x: x['importance_score'], reverse=True)
            
            # Remove highly correlated features
            selected = []
            for combo in combinations:
                if len(selected) >= self.config.max_features_to_select:
                    break
                
                # Check correlation with already selected features
                is_correlated = False
                for selected_combo in selected:
                    try:
                        corr = combo['feature_series'].corr(selected_combo['feature_series'])
                        if abs(corr) > self.config.correlation_threshold:
                            is_correlated = True
                            break
                    except:
                        continue
                
                if not is_correlated and combo['importance_score'] > self.config.feature_importance_threshold:
                    selected.append(combo)
            
            tprint_debug(f"📊 Selected {len(selected)} best features")
            return selected
            
        except Exception as e:
            tprint_error(f"❌ Feature selection failed: {e}")
            return combinations[:self.config.max_features_to_select]
    
    def _create_generated_features(
        self, 
        selected_combinations: List[Dict[str, Any]], 
        base_features: pd.DataFrame,
        shap_values: Optional[np.ndarray]
    ) -> List[GeneratedFeature]:
        """Create GeneratedFeature objects from selected combinations."""
        try:
            generated_features = []
            
            for combo in selected_combinations:
                feature = GeneratedFeature(
                    name=combo['name'],
                    formula=combo['formula'],
                    feature_series=combo['feature_series'],
                    importance_score=combo['importance_score'],
                    parent_features=combo['parent_features'],
                    feature_type='generated',
                    generation_method='randomforest',
                    metadata={
                        'interaction_type': combo.get('interaction_type', 'unknown'),
                        'base_features_count': len(base_features.columns),
                        'shap_available': shap_values is not None
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
            metadata={'error': error_message}
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()


def create_randomforest_feature_generator(
    config: Optional[FeatureGenerationConfig] = None
) -> RandomForestFeatureGenerator:
    """Create a RandomForest feature generator with default configuration."""
    return RandomForestFeatureGenerator(config)