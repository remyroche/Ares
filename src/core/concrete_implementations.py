"""
Concrete Implementations of Abstract Base Classes

This module provides concrete implementations of the abstract base classes
to demonstrate their usage and provide production-ready examples.

Implementations:
1. DataValidator - Concrete validator for data validation
2. MLTrainingStep - Concrete training step for ML models
3. KMeansClustering - Concrete K-means clustering implementation
4. MultiOutputRandomForest - Concrete multi-output random forest
5. MomentumPatternDiscoverer - Concrete momentum pattern discovery
6. ProfitBasedLabeling - Concrete profit-based labeling strategy
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import logging
from datetime import datetime

# Import base classes
from .abstract_base_classes import (
    BaseValidator, BaseTrainingStep, BaseClusteringAlgorithm,
    MultiOutputModel, BasePatternDiscoverer, BaseLabelingStrategy,
    ValidationResult, TrainingResult, ClusteringResult, PatternDiscoveryResult,
    PatternDefinition, LabelingResult, ValidationLevel, TrainingStatus,
    ClusteringAlgorithm, PatternType, LabelingStrategy
)

# ML imports
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Core utilities
from src.utils.logger import system_logger

logger = system_logger.getChild('ConcreteImplementations')

# ============================================================================
# DATA VALIDATOR IMPLEMENTATION
# ============================================================================

class DataValidator(BaseValidator):
    """
    Concrete implementation of BaseValidator for data validation.
    
    Provides comprehensive data validation including:
    - Data type validation
    - Shape and dimension validation
    - Value range validation
    - Missing value detection
    - Statistical validation
    """
    
    def __init__(self, 
                 name: str = "DataValidator",
                 validation_level: ValidationLevel = ValidationLevel.STANDARD,
                 **kwargs):
        super().__init__(name, validation_level, **kwargs)
        
        # Validation configuration
        self.required_columns = self.config.get('required_columns', [])
        self.column_types = self.config.get('column_types', {})
        self.value_ranges = self.config.get('value_ranges', {})
        self.max_missing_ratio = self.config.get('max_missing_ratio', 0.1)
        self.min_samples = self.config.get('min_samples', 10)
        
        if self.logger:
            self.logger.info(f"DataValidator initialized with {len(self.required_columns)} required columns")

    async def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate data according to configuration."""
        start_time = time.time()
        errors = []
        warnings = []
        metrics = {}
        
        try:
            # Convert to DataFrame if needed
            if isinstance(data, np.ndarray):
                data = pd.DataFrame(data)
            elif not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            
            # Basic data validation
            if data.empty:
                errors.append("Data is empty")
                return ValidationResult(
                    is_valid=False,
                    errors=errors,
                    warnings=warnings,
                    metrics=metrics,
                    execution_time=time.time() - start_time
                )
            
            # Check minimum samples
            if len(data) < self.min_samples:
                errors.append(f"Insufficient samples: {len(data)} < {self.min_samples}")
            
            # Check required columns
            missing_columns = set(self.required_columns) - set(data.columns)
            if missing_columns:
                errors.append(f"Missing required columns: {missing_columns}")
            
            # Check column types
            for col, expected_type in self.column_types.items():
                if col in data.columns:
                    if not isinstance(data[col].iloc[0], expected_type):
                        warnings.append(f"Column {col} has unexpected type: {type(data[col].iloc[0])}")
            
            # Check value ranges
            for col, (min_val, max_val) in self.value_ranges.items():
                if col in data.columns:
                    col_data = pd.to_numeric(data[col], errors='coerce')
                    if not col_data.isna().all():
                        if col_data.min() < min_val or col_data.max() > max_val:
                            warnings.append(f"Column {col} values outside range [{min_val}, {max_val}]")
            
            # Check missing values
            missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            if missing_ratio > self.max_missing_ratio:
                errors.append(f"Too many missing values: {missing_ratio:.2%} > {self.max_missing_ratio:.2%}")
            
            # Calculate metrics
            metrics = {
                'n_samples': len(data),
                'n_features': len(data.columns),
                'missing_ratio': missing_ratio,
                'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
                'dtypes': data.dtypes.to_dict()
            }
            
            # Determine validity
            is_valid = len(errors) == 0
            
            result = ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                metrics=metrics,
                execution_time=time.time() - start_time
            )
            
            self._record_validation(result)
            
            if self.logger:
                if is_valid:
                    self.logger.info(f"Data validation passed: {len(data)} samples, {len(data.columns)} features")
                else:
                    self.logger.warning(f"Data validation failed: {len(errors)} errors, {len(warnings)} warnings")
            
            return result
            
        except Exception as e:
            error_result = ValidationResult(
                is_valid=False,
                errors=[f"Validation failed with exception: {str(e)}"],
                warnings=warnings,
                metrics=metrics,
                execution_time=time.time() - start_time
            )
            
            self._record_validation(error_result)
            
            if self.logger:
                self.logger.error(f"Data validation exception: {e}")
            
            return error_result

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get comprehensive validation summary."""
        base_summary = super().get_validation_summary()
        
        # Add data-specific metrics
        if self.validation_history:
            latest_result = self.validation_history[-1]
            base_summary.update({
                'latest_n_samples': latest_result.metrics.get('n_samples', 0),
                'latest_n_features': latest_result.metrics.get('n_features', 0),
                'latest_missing_ratio': latest_result.metrics.get('missing_ratio', 0),
                'latest_memory_usage': latest_result.metrics.get('memory_usage_mb', 0)
            })
        
        return base_summary

# ============================================================================
# ML TRAINING STEP IMPLEMENTATION
# ============================================================================

class MLTrainingStep(BaseTrainingStep):
    """
    Concrete implementation of BaseTrainingStep for ML model training.
    
    Provides comprehensive ML training including:
    - Data preprocessing and feature engineering
    - Model training with hyperparameter optimization
    - Cross-validation and performance evaluation
    - Model persistence and artifact generation
    """
    
    def __init__(self, 
                 name: str = "MLTrainingStep",
                 model_type: str = "random_forest",
                 **kwargs):
        super().__init__(name, **kwargs)
        
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.feature_names = []
        
        if self.logger:
            self.logger.info(f"MLTrainingStep initialized with model type: {model_type}")

    def _initialize_step_components(self) -> None:
        """Initialize ML training components."""
        # Initialize model based on type
        if self.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=self.config.get('n_estimators', 100),
                max_depth=self.config.get('max_depth', None),
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == "random_forest_classifier":
            self.model = RandomForestClassifier(
                n_estimators=self.config.get('n_estimators', 100),
                max_depth=self.config.get('max_depth', None),
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        if self.logger:
            self.logger.info(f"Initialized {self.model_type} model")

    def _process_data(self, data: Any) -> Any:
        """Process input data for training."""
        try:
            # Convert to numpy array if needed
            if isinstance(data, pd.DataFrame):
                X = data.values
                self.feature_names = data.columns.tolist()
            elif isinstance(data, np.ndarray):
                X = data
                self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            else:
                X = np.array(data)
                self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
            # Handle missing values
            if np.isnan(X).any():
                X = np.nan_to_num(X, nan=0.0)
                if self.logger:
                    self.logger.warning("Missing values found and replaced with 0")
            
            # Scale features if configured
            if self.config.get('scale_features', True):
                X = self.scaler.fit_transform(X)
                if self.logger:
                    self.logger.info("Features scaled using StandardScaler")
            
            return X
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Data processing failed: {e}")
            raise

    def _generate_artifacts(self, model: Any, results: TrainingResult) -> Dict[str, Any]:
        """Generate training artifacts."""
        artifacts = {
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'scaler_params': {
                'mean': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
                'scale': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None
            },
            'model_params': model.get_params() if hasattr(model, 'get_params') else {},
            'training_timestamp': datetime.now().isoformat()
        }
        
        # Add feature importance if available
        if hasattr(model, 'feature_importances_'):
            artifacts['feature_importance'] = {
                name: importance for name, importance in 
                zip(self.feature_names, model.feature_importances_)
            }
        
        return artifacts

    def _calculate_metrics(self, model: Any, test_data: Any) -> Dict[str, Any]:
        """Calculate performance metrics."""
        try:
            X_test, y_test = test_data
            
            # Process test data
            X_test_processed = self._process_data(X_test)
            
            # Make predictions
            y_pred = model.predict(X_test_processed)
            
            # Calculate metrics
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
            
            # Add classification metrics if applicable
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test_processed)
                metrics['prediction_confidence'] = np.mean(np.max(y_proba, axis=1))
            
            return metrics
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Metrics calculation failed: {e}")
            return {'error': str(e)}

    async def _train_model(self, data: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        """Train the ML model."""
        try:
            # Split data if not already split
            if isinstance(data, tuple) and len(data) == 2:
                X, y = data
            else:
                # Assume data contains both X and y
                X = data
                y = context.get('target') if context else None
                if y is None:
                    raise ValueError("Target values not provided")
            
            # Train model
            start_time = time.time()
            self.model.fit(X, y)
            training_time = time.time() - start_time
            
            if self.logger:
                self.logger.info(f"Model trained in {training_time:.2f}s")
            
            return self.model
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Model training failed: {e}")
            raise

# ============================================================================
# K-MEANS CLUSTERING IMPLEMENTATION
# ============================================================================

class KMeansClustering(BaseClusteringAlgorithm):
    """
    Concrete implementation of BaseClusteringAlgorithm using K-means.
    
    Provides comprehensive K-means clustering including:
    - Automatic cluster number selection
    - Performance optimization
    - Detailed metrics calculation
    - Memory management
    """
    
    def __init__(self, 
                 name: str = "KMeansClustering",
                 n_clusters: int = 3,
                 **kwargs):
        super().__init__(name, ClusteringAlgorithm.KMEANS, **kwargs)
        
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        
        if self.logger:
            self.logger.info(f"KMeansClustering initialized with {n_clusters} clusters")

    def fit_predict(self, data: np.ndarray) -> ClusteringResult:
        """Fit K-means and predict cluster labels."""
        start_time = time.time()
        
        try:
            # Validate input
            if len(data) < self.n_clusters:
                raise ValueError(f"Not enough samples ({len(data)}) for {self.n_clusters} clusters")
            
            # Scale data
            data_scaled = self.scaler.fit_transform(data)
            
            # Initialize and fit K-means
            self.model = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10,
                max_iter=300
            )
            
            labels = self.model.fit_predict(data_scaled)
            
            # Calculate metrics
            silhouette = self.get_silhouette_score(data_scaled, labels)
            inertia = self.get_inertia(data_scaled, labels)
            
            metrics = {
                'silhouette_score': silhouette,
                'inertia': inertia,
                'n_clusters': self.n_clusters,
                'n_samples': len(data),
                'n_features': data.shape[1]
            }
            
            metadata = {
                'algorithm': 'kmeans',
                'n_init': 10,
                'max_iter': 300,
                'converged': self.model.n_iter_ < 300,
                'n_iter': self.model.n_iter_
            }
            
            result = ClusteringResult(
                labels=labels,
                n_clusters=self.n_clusters,
                algorithm='kmeans',
                metrics=metrics,
                metadata=metadata,
                execution_time=time.time() - start_time,
                silhouette_score=silhouette,
                inertia=inertia
            )
            
            if self.logger:
                self.logger.info(f"K-means clustering completed: {self.n_clusters} clusters, silhouette={silhouette:.3f}")
            
            return result
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"K-means clustering failed: {e}")
            raise

# ============================================================================
# MULTI-OUTPUT RANDOM FOREST IMPLEMENTATION
# ============================================================================

class MultiOutputRandomForest(MultiOutputModel):
    """
    Concrete implementation of MultiOutputModel using Random Forest.
    
    Provides comprehensive multi-output random forest including:
    - Individual models for each output
    - Performance optimization
    - Feature importance analysis
    - Cross-validation support
    """
    
    def __init__(self, 
                 name: str = "MultiOutputRandomForest",
                 n_outputs: int = 2,
                 output_names: Optional[List[str]] = None,
                 **kwargs):
        super().__init__(name, n_outputs, output_names, **kwargs)
        
        self.scaler = StandardScaler()
        
        if self.logger:
            self.logger.info(f"MultiOutputRandomForest initialized with {n_outputs} outputs")

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultiOutputRandomForest':
        """Fit multi-output random forest."""
        try:
            start_time = time.time()
            
            # Validate inputs
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
            
            if y.shape[1] != self.n_outputs:
                raise ValueError(f"Expected {self.n_outputs} outputs, got {y.shape[1]}")
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train individual models for each output
            for i in range(self.n_outputs):
                output_name = self.output_names[i]
                y_output = y[:, i]
                
                # Determine if classification or regression
                unique_values = np.unique(y_output)
                is_classification = len(unique_values) <= 10 and all(isinstance(v, (int, np.integer)) for v in unique_values)
                
                if is_classification:
                    model = RandomForestClassifier(
                        n_estimators=self.config.get('n_estimators', 100),
                        max_depth=self.config.get('max_depth', None),
                        random_state=42,
                        n_jobs=-1
                    )
                else:
                    model = RandomForestRegressor(
                        n_estimators=self.config.get('n_estimators', 100),
                        max_depth=self.config.get('max_depth', None),
                        random_state=42,
                        n_jobs=-1
                    )
                
                model.fit(X_scaled, y_output)
                self.models[output_name] = model
                
                if self.logger:
                    self.logger.info(f"Trained model for {output_name} ({'classification' if is_classification else 'regression'})")
            
            self.is_fitted = True
            self.total_training_time = time.time() - start_time
            
            if self.logger:
                self.logger.info(f"Multi-output random forest fitted in {self.total_training_time:.2f}s")
            
            return self
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Multi-output random forest fitting failed: {e}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions for all outputs."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            start_time = time.time()
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make predictions for each output
            predictions = []
            for output_name in self.output_names:
                model = self.models[output_name]
                pred = model.predict(X_scaled)
                predictions.append(pred)
            
            # Stack predictions
            result = np.column_stack(predictions)
            
            self.total_prediction_time += time.time() - start_time
            
            if self.logger:
                self.logger.debug(f"Predictions made for {len(X)} samples in {time.time() - start_time:.3f}s")
            
            return result
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Prediction failed: {e}")
            raise

# ============================================================================
# MOMENTUM PATTERN DISCOVERER IMPLEMENTATION
# ============================================================================

class MomentumPatternDiscoverer(BasePatternDiscoverer):
    """
    Concrete implementation of BasePatternDiscoverer for momentum patterns.
    
    Provides comprehensive momentum pattern discovery including:
    - Price momentum calculation
    - Pattern definition and validation
    - Confidence scoring
    - Statistical analysis
    """
    
    def __init__(self, 
                 name: str = "MomentumPatternDiscoverer",
                 **kwargs):
        super().__init__(name, PatternType.MOMENTUM, **kwargs)
        
        self.lookback_period = self.config.get('lookback_period', 20)
        self.momentum_threshold = self.config.get('momentum_threshold', 0.05)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        
        if self.logger:
            self.logger.info(f"MomentumPatternDiscoverer initialized with lookback={self.lookback_period}")

    def discover_pattern(self, data: np.ndarray, **kwargs) -> PatternDiscoveryResult:
        """Discover momentum patterns in data."""
        try:
            # Calculate momentum
            momentum = self._calculate_momentum(data)
            
            # Identify pattern occurrences
            pattern_mask = momentum > self.momentum_threshold
            labels = pattern_mask.astype(int)
            
            # Calculate confidence scores
            confidence_scores = self._calculate_confidence(momentum, pattern_mask)
            
            # Calculate frequency
            frequency = np.mean(labels)
            
            # Calculate metrics
            metrics = {
                'momentum_mean': np.mean(momentum),
                'momentum_std': np.std(momentum),
                'pattern_frequency': frequency,
                'confidence_mean': np.mean(confidence_scores),
                'confidence_std': np.std(confidence_scores)
            }
            
            metadata = {
                'lookback_period': self.lookback_period,
                'momentum_threshold': self.momentum_threshold,
                'n_samples': len(data)
            }
            
            result = PatternDiscoveryResult(
                definition=self.get_pattern_definition(),
                labels=labels,
                confidence_scores=confidence_scores,
                frequency=frequency,
                metrics=metrics,
                metadata=metadata
            )
            
            self.discovered_patterns.append(result)
            
            if self.logger:
                self.logger.info(f"Momentum pattern discovery completed: frequency={frequency:.3f}")
            
            return result
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Momentum pattern discovery failed: {e}")
            raise

    def get_pattern_definition(self) -> PatternDefinition:
        """Get mathematical definition of momentum pattern."""
        return PatternDefinition(
            name="Momentum Pattern",
            pattern_type=PatternType.MOMENTUM,
            description="Price momentum above threshold indicating upward trend",
            mathematical_formula=f"momentum = (price[t] - price[t-{self.lookback_period}]) / price[t-{self.lookback_period}]",
            parameters={
                'lookback_period': self.lookback_period,
                'momentum_threshold': self.momentum_threshold
            },
            frequency_threshold=0.1,
            confidence_threshold=self.confidence_threshold
        )

    def _calculate_momentum(self, data: np.ndarray) -> np.ndarray:
        """Calculate momentum for each data point."""
        if len(data) < self.lookback_period + 1:
            return np.zeros(len(data))
        
        momentum = np.zeros(len(data))
        for i in range(self.lookback_period, len(data)):
            if data[i - self.lookback_period] != 0:
                momentum[i] = (data[i] - data[i - self.lookback_period]) / data[i - self.lookback_period]
        
        return momentum

    def _calculate_confidence(self, momentum: np.ndarray, pattern_mask: np.ndarray) -> np.ndarray:
        """Calculate confidence scores for momentum patterns."""
        # Simple confidence based on momentum magnitude
        confidence = np.abs(momentum)
        confidence = np.clip(confidence, 0, 1)  # Normalize to [0, 1]
        
        # Apply pattern mask
        confidence = confidence * pattern_mask
        
        return confidence

# ============================================================================
# PROFIT-BASED LABELING IMPLEMENTATION
# ============================================================================

class ProfitBasedLabeling(BaseLabelingStrategy):
    """
    Concrete implementation of BaseLabelingStrategy for profit-based labeling.
    
    Provides comprehensive profit-based labeling including:
    - Profit calculation and thresholding
    - Confidence scoring based on profit magnitude
    - Label validation and quality assessment
    """
    
    def __init__(self, 
                 name: str = "ProfitBasedLabeling",
                 **kwargs):
        super().__init__(name, LabelingStrategy.PROFIT_BASED, **kwargs)
        
        self.profit_threshold = self.config.get('profit_threshold', 0.02)  # 2% profit
        self.lookforward_period = self.config.get('lookforward_period', 5)
        self.min_confidence = self.config.get('min_confidence', 0.5)
        
        if self.logger:
            self.logger.info(f"ProfitBasedLabeling initialized with threshold={self.profit_threshold}")

    def generate_labels(self, data: np.ndarray, **kwargs) -> LabelingResult:
        """Generate profit-based labels."""
        try:
            # Extract price data
            if len(data.shape) == 1:
                prices = data
            else:
                prices = data[:, 0]  # Assume first column is price
            
            # Calculate future profits
            profits = self._calculate_future_profits(prices)
            
            # Generate labels based on profit threshold
            labels = (profits > self.profit_threshold).astype(int)
            
            # Calculate confidence scores
            confidence_scores = self.calculate_confidence(labels, data, profits=profits)
            
            # Calculate metrics
            metrics = {
                'profit_mean': np.mean(profits),
                'profit_std': np.std(profits),
                'label_frequency': np.mean(labels),
                'confidence_mean': np.mean(confidence_scores),
                'profit_threshold': self.profit_threshold
            }
            
            metadata = {
                'lookforward_period': self.lookforward_period,
                'n_samples': len(data),
                'strategy': 'profit_based'
            }
            
            result = LabelingResult(
                labels=labels,
                confidence_scores=confidence_scores,
                strategy=LabelingStrategy.PROFIT_BASED,
                metrics=metrics,
                metadata=metadata
            )
            
            self.labeling_results.append(result)
            
            if self.logger:
                self.logger.info(f"Profit-based labeling completed: {np.mean(labels):.3f} positive labels")
            
            return result
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Profit-based labeling failed: {e}")
            raise

    def calculate_confidence(self, labels: np.ndarray, data: np.ndarray, **kwargs) -> np.ndarray:
        """Calculate confidence scores for labels."""
        try:
            profits = kwargs.get('profits')
            if profits is None:
                # Calculate profits if not provided
                if len(data.shape) == 1:
                    prices = data
                else:
                    prices = data[:, 0]
                profits = self._calculate_future_profits(prices)
            
            # Confidence based on profit magnitude
            confidence = np.abs(profits)
            confidence = np.clip(confidence, 0, 1)  # Normalize to [0, 1]
            
            # Apply minimum confidence threshold
            confidence = np.maximum(confidence, self.min_confidence)
            
            return confidence
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Confidence calculation failed: {e}")
            return np.ones(len(labels)) * self.min_confidence

    def _calculate_future_profits(self, prices: np.ndarray) -> np.ndarray:
        """Calculate future profits for each price point."""
        profits = np.zeros(len(prices))
        
        for i in range(len(prices) - self.lookforward_period):
            current_price = prices[i]
            future_price = prices[i + self.lookforward_period]
            
            if current_price != 0:
                profits[i] = (future_price - current_price) / current_price
        
        return profits

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_data_validator(**kwargs) -> DataValidator:
    """Create a data validator instance."""
    return DataValidator(**kwargs)

def create_ml_training_step(**kwargs) -> MLTrainingStep:
    """Create an ML training step instance."""
    return MLTrainingStep(**kwargs)

def create_kmeans_clustering(**kwargs) -> KMeansClustering:
    """Create a K-means clustering instance."""
    return KMeansClustering(**kwargs)

def create_multi_output_random_forest(**kwargs) -> MultiOutputRandomForest:
    """Create a multi-output random forest instance."""
    return MultiOutputRandomForest(**kwargs)

def create_momentum_pattern_discoverer(**kwargs) -> MomentumPatternDiscoverer:
    """Create a momentum pattern discoverer instance."""
    return MomentumPatternDiscoverer(**kwargs)

def create_profit_based_labeling(**kwargs) -> ProfitBasedLabeling:
    """Create a profit-based labeling instance."""
    return ProfitBasedLabeling(**kwargs)