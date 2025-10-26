"""
Adaptive Optimization Engine for Apple Silicon.

This module provides an intelligent optimization system that learns from performance
patterns and automatically tunes hardware settings for optimal performance.
"""

import logging
import time
import threading
import json
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import pickle
from pathlib import Path
import asyncio
from collections import deque, defaultdict
import queue
import sqlite3
from datetime import datetime, timedelta

# Optional dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

from .unified_hardware_manager import UnifiedHardwareManager, WorkloadType, OptimizationLevel
from .advanced_cpu_optimizer import AdvancedM1CPUOptimizer, WorkloadProfile
from .enhanced_gpu_manager import EnhancedM1GPUManager, GPUOperationType
from .advanced_memory_optimizer import AdvancedM1MemoryOptimizer, MemoryStrategy

logger = logging.getLogger(__name__)

class LearningAlgorithm(Enum):
    """Learning algorithms for optimization."""
    LINEAR_REGRESSION = "linear_regression"
    DECISION_TREE = "decision_tree"
    NEURAL_NETWORK = "neural_network"
    GENETIC_ALGORITHM = "genetic_algorithm"
    REINFORCEMENT_LEARNING = "reinforcement_learning"

class OptimizationTarget(Enum):
    """Optimization targets."""
    PERFORMANCE = "performance"
    EFFICIENCY = "efficiency"
    POWER_CONSUMPTION = "power_consumption"
    THERMAL_MANAGEMENT = "thermal_management"
    MEMORY_USAGE = "memory_usage"
    BALANCED = "balanced"

@dataclass
class PerformanceMetrics:
    """Performance metrics for learning."""
    timestamp: float
    workload_type: str
    optimization_level: str
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    temperature: float
    power_consumption: float
    execution_time: float
    throughput: float
    error_rate: float
    optimization_target: str
    settings_hash: str
    performance_score: float

@dataclass
class OptimizationSettings:
    """Optimization settings configuration."""
    cpu_cores_performance: int
    cpu_cores_efficiency: int
    cpu_frequency_scaling: float
    memory_allocation_strategy: str
    gpu_acceleration_enabled: bool
    gpu_memory_pool_size: float
    thermal_threshold: float
    power_limit: float
    optimization_level: str
    workload_specific_settings: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LearningModel:
    """Learning model for optimization."""
    model_id: str
    algorithm: LearningAlgorithm
    target_metric: OptimizationTarget
    accuracy: float
    last_trained: float
    training_samples: int
    model_data: Any = None
    feature_importance: Dict[str, float] = field(default_factory=dict)

class PerformanceDatabase:
    """Database for storing performance metrics and learning data."""

    def __init__(self, db_path: str = "optimization_performance.db"):
        self.db_path = db_path
        self.logger = logger.getChild('PerformanceDatabase')
        self._init_database()

    def _init_database(self):
        """Initialize the performance database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create performance metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL,
                        workload_type TEXT,
                        optimization_level TEXT,
                        cpu_usage REAL,
                        memory_usage REAL,
                        gpu_usage REAL,
                        temperature REAL,
                        power_consumption REAL,
                        execution_time REAL,
                        throughput REAL,
                        error_rate REAL,
                        optimization_target TEXT,
                        settings_hash TEXT,
                        performance_score REAL
                    )
                ''')

                # Create optimization settings table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS optimization_settings (
                        settings_hash TEXT PRIMARY KEY,
                        cpu_cores_performance INTEGER,
                        cpu_cores_efficiency INTEGER,
                        cpu_frequency_scaling REAL,
                        memory_allocation_strategy TEXT,
                        gpu_acceleration_enabled BOOLEAN,
                        gpu_memory_pool_size REAL,
                        thermal_threshold REAL,
                        power_limit REAL,
                        optimization_level TEXT,
                        workload_specific_settings TEXT,
                        created_at REAL
                    )
                ''')

                # Create learning models table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS learning_models (
                        model_id TEXT PRIMARY KEY,
                        algorithm TEXT,
                        target_metric TEXT,
                        accuracy REAL,
                        last_trained REAL,
                        training_samples INTEGER,
                        model_data BLOB,
                        feature_importance TEXT
                    )
                ''')

                conn.commit()
                self.logger.info("📊 Performance database initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")

    def store_performance_metrics(self, metrics: PerformanceMetrics) -> bool:
        """Store performance metrics in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO performance_metrics (
                        timestamp, workload_type, optimization_level, cpu_usage,
                        memory_usage, gpu_usage, temperature, power_consumption,
                        execution_time, throughput, error_rate, optimization_target,
                        settings_hash, performance_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.timestamp, metrics.workload_type, metrics.optimization_level,
                    metrics.cpu_usage, metrics.memory_usage, metrics.gpu_usage,
                    metrics.temperature, metrics.power_consumption, metrics.execution_time,
                    metrics.throughput, metrics.error_rate, metrics.optimization_target,
                    metrics.settings_hash, metrics.performance_score
                ))
                conn.commit()
                return True

        except Exception as e:
            self.logger.error(f"Failed to store performance metrics: {e}")
            return False

    def store_optimization_settings(self, settings: OptimizationSettings) -> str:
        """Store optimization settings and return hash."""
        try:
            settings_hash = self._hash_settings(settings)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO optimization_settings (
                        settings_hash, cpu_cores_performance, cpu_cores_efficiency,
                        cpu_frequency_scaling, memory_allocation_strategy,
                        gpu_acceleration_enabled, gpu_memory_pool_size,
                        thermal_threshold, power_limit, optimization_level,
                        workload_specific_settings, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    settings_hash, settings.cpu_cores_performance, settings.cpu_cores_efficiency,
                    settings.cpu_frequency_scaling, settings.memory_allocation_strategy,
                    settings.gpu_acceleration_enabled, settings.gpu_memory_pool_size,
                    settings.thermal_threshold, settings.power_limit, settings.optimization_level,
                    json.dumps(settings.workload_specific_settings), time.time()
                ))
                conn.commit()
                return settings_hash

        except Exception as e:
            self.logger.error(f"Failed to store optimization settings: {e}")
            return ""

    def get_performance_history(self, workload_type: Optional[str] = None,
                              days: int = 30) -> List[PerformanceMetrics]:
        """Get performance history from database."""
        try:
            cutoff_time = time.time() - (days * 24 * 3600)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                if workload_type:
                    cursor.execute('''
                        SELECT * FROM performance_metrics
                        WHERE workload_type = ? AND timestamp > ?
                        ORDER BY timestamp DESC
                    ''', (workload_type, cutoff_time))
                else:
                    cursor.execute('''
                        SELECT * FROM performance_metrics
                        WHERE timestamp > ?
                        ORDER BY timestamp DESC
                    ''', (cutoff_time,))

                rows = cursor.fetchall()

                metrics = []
                for row in rows:
                    metrics.append(PerformanceMetrics(
                        timestamp=row[1],
                        workload_type=row[2],
                        optimization_level=row[3],
                        cpu_usage=row[4],
                        memory_usage=row[5],
                        gpu_usage=row[6],
                        temperature=row[7],
                        power_consumption=row[8],
                        execution_time=row[9],
                        throughput=row[10],
                        error_rate=row[11],
                        optimization_target=row[12],
                        settings_hash=row[13],
                        performance_score=row[14]
                    ))

                return metrics

        except Exception as e:
            self.logger.error(f"Failed to get performance history: {e}")
            return []

    def _hash_settings(self, settings: OptimizationSettings) -> str:
        """Generate hash for optimization settings."""
        settings_str = json.dumps({
            'cpu_cores_performance': settings.cpu_cores_performance,
            'cpu_cores_efficiency': settings.cpu_cores_efficiency,
            'cpu_frequency_scaling': settings.cpu_frequency_scaling,
            'memory_allocation_strategy': settings.memory_allocation_strategy,
            'gpu_acceleration_enabled': settings.gpu_acceleration_enabled,
            'gpu_memory_pool_size': settings.gpu_memory_pool_size,
            'thermal_threshold': settings.thermal_threshold,
            'power_limit': settings.power_limit,
            'optimization_level': settings.optimization_level
        }, sort_keys=True)

        return str(hash(settings_str))

class OptimizationLearner:
    """Machine learning component for optimization."""

    def __init__(self, database: PerformanceDatabase):
        self.database = database
        self.logger = logger.getChild('OptimizationLearner')
        self.models: Dict[str, LearningModel] = {}
        self.feature_columns = [
            'cpu_cores_performance', 'cpu_cores_efficiency', 'cpu_frequency_scaling',
            'memory_allocation_strategy', 'gpu_acceleration_enabled', 'gpu_memory_pool_size',
            'thermal_threshold', 'power_limit', 'optimization_level'
        ]

    def train_model(self, target_metric: OptimizationTarget,
                   algorithm: LearningAlgorithm = LearningAlgorithm.LINEAR_REGRESSION) -> bool:
        """Train a learning model for optimization."""
        try:
            model_id = f"{target_metric.value}_{algorithm.value}"

            # Get training data
            training_data = self._prepare_training_data(target_metric)
            if len(training_data) < 10:
                self.logger.info(f"Insufficient training data for {model_id}, using default configuration")
                # Create a default model with conservative settings
                return self._create_default_model(target_metric, algorithm)

            # Train model based on algorithm
            if algorithm == LearningAlgorithm.LINEAR_REGRESSION:
                model_data = self._train_linear_regression(training_data, target_metric)
            elif algorithm == LearningAlgorithm.DECISION_TREE:
                model_data = self._train_decision_tree(training_data, target_metric)
            else:
                self.logger.warning(f"Algorithm {algorithm.value} not implemented")
                return False

            # Calculate accuracy
            accuracy = self._calculate_model_accuracy(model_data, training_data, target_metric)

            # Create learning model
            model = LearningModel(
                model_id=model_id,
                algorithm=algorithm,
                target_metric=target_metric,
                accuracy=accuracy,
                last_trained=time.time(),
                training_samples=len(training_data),
                model_data=model_data,
                feature_importance=self._calculate_feature_importance(model_data, target_metric)
            )

            self.models[model_id] = model
            self._save_model_to_database(model)

            self.logger.info(f"🧠 Trained model {model_id} with accuracy {accuracy:.3f}")
            return True

        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            return False

    def _create_default_model(self, target_metric: OptimizationTarget, algorithm: LearningAlgorithm) -> bool:
        """Create a default model when insufficient training data is available."""
        try:
            model_id = f"{target_metric.value}_{algorithm.value}"
            
            # Create default model with conservative settings
            default_model = {
                'model_id': model_id,
                'target_metric': target_metric.value,
                'algorithm': algorithm.value,
                'accuracy': 0.5,  # Default accuracy
                'created_at': time.time(),
                'is_default': True,
                'coefficients': self._get_default_coefficients(target_metric),
                'intercept': self._get_default_intercept(target_metric)
            }
            
            # Store the default model
            self.models[model_id] = default_model
            self._save_model_to_database(default_model)
            
            self.logger.info(f"🧠 Created default model {model_id} for {target_metric.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Default model creation failed: {e}")
            return False

    def _get_default_coefficients(self, target_metric: OptimizationTarget) -> Dict[str, float]:
        """Get default coefficients for the model."""
        # Conservative default coefficients based on target metric
        if target_metric == OptimizationTarget.PERFORMANCE:
            return {
                'cpu_cores_performance': 0.1,
                'cpu_cores_efficiency': 0.05,
                'cpu_frequency_scaling': 0.2,
                'memory_allocation_strategy': 0.1,
                'gpu_acceleration_enabled': 0.3,
                'gpu_memory_pool_size': 0.05,
                'thermal_threshold': -0.1,
                'power_limit': 0.15,
                'optimization_level': 0.2
            }
        elif target_metric == OptimizationTarget.EFFICIENCY:
            return {
                'cpu_cores_performance': 0.05,
                'cpu_cores_efficiency': 0.2,
                'cpu_frequency_scaling': 0.1,
                'memory_allocation_strategy': 0.3,
                'gpu_acceleration_enabled': 0.1,
                'gpu_memory_pool_size': 0.2,
                'thermal_threshold': 0.2,
                'power_limit': 0.3,
                'optimization_level': 0.1
            }
        elif target_metric == OptimizationTarget.POWER_CONSUMPTION:
            return {
                'cpu_cores_performance': -0.1,
                'cpu_cores_efficiency': 0.1,
                'cpu_frequency_scaling': -0.2,
                'memory_allocation_strategy': 0.2,
                'gpu_acceleration_enabled': -0.3,
                'gpu_memory_pool_size': -0.1,
                'thermal_threshold': 0.3,
                'power_limit': 0.4,
                'optimization_level': -0.1
            }
        else:  # BALANCED
            return {
                'cpu_cores_performance': 0.1,
                'cpu_cores_efficiency': 0.1,
                'cpu_frequency_scaling': 0.1,
                'memory_allocation_strategy': 0.1,
                'gpu_acceleration_enabled': 0.1,
                'gpu_memory_pool_size': 0.1,
                'thermal_threshold': 0.1,
                'power_limit': 0.1,
                'optimization_level': 0.1
            }

    def _get_default_intercept(self, target_metric: OptimizationTarget) -> float:
        """Get default intercept for the model."""
        # Default intercept values based on target metric
        if target_metric == OptimizationTarget.PERFORMANCE:
            return 0.7  # Base performance score
        elif target_metric == OptimizationTarget.EFFICIENCY:
            return 0.6  # Base efficiency score
        elif target_metric == OptimizationTarget.POWER_CONSUMPTION:
            return 0.5  # Base power consumption score
        else:  # BALANCED
            return 0.6  # Base balanced score

    def _prepare_training_data(self, target_metric: OptimizationTarget) -> List[Dict[str, Any]]:
        """Prepare training data for model."""
        try:
            # Get performance metrics
            metrics = self.database.get_performance_history(days=90)
            if not metrics:
                return []

            training_data = []
            for metric in metrics:
                # Get corresponding settings
                settings = self._get_settings_for_hash(metric.settings_hash)
                if not settings:
                    continue

                # Prepare feature vector
                features = {
                    'cpu_cores_performance': settings.cpu_cores_performance,
                    'cpu_cores_efficiency': settings.cpu_cores_efficiency,
                    'cpu_frequency_scaling': settings.cpu_frequency_scaling,
                    'memory_allocation_strategy': 1 if settings.memory_allocation_strategy == 'aggressive' else 0,
                    'gpu_acceleration_enabled': 1 if settings.gpu_acceleration_enabled else 0,
                    'gpu_memory_pool_size': settings.gpu_memory_pool_size,
                    'thermal_threshold': settings.thermal_threshold,
                    'power_limit': settings.power_limit,
                    'optimization_level': 1 if settings.optimization_level == 'aggressive' else 0
                }

                # Add target value
                if target_metric == OptimizationTarget.PERFORMANCE:
                    features['target'] = metric.performance_score
                elif target_metric == OptimizationTarget.EFFICIENCY:
                    features['target'] = metric.throughput / max(1, metric.power_consumption)
                elif target_metric == OptimizationTarget.POWER_CONSUMPTION:
                    features['target'] = metric.power_consumption
                elif target_metric == OptimizationTarget.THERMAL_MANAGEMENT:
                    features['target'] = metric.temperature
                elif target_metric == OptimizationTarget.MEMORY_USAGE:
                    features['target'] = metric.memory_usage
                else:  # BALANCED
                    features['target'] = metric.performance_score * 0.4 + (1 - metric.memory_usage/100) * 0.3 + (1 - metric.temperature/100) * 0.3

                training_data.append(features)

            return training_data

        except Exception as e:
            self.logger.error(f"Failed to prepare training data: {e}")
            return []

    def _train_linear_regression(self, training_data: List[Dict[str, Any]],
                               target_metric: OptimizationTarget) -> Dict[str, Any]:
        """Train linear regression model."""
        try:
            if NUMPY_AVAILABLE:
                # Convert to numpy arrays
                X = np.array([[data[col] for col in self.feature_columns] for data in training_data])
                y = np.array([data['target'] for data in training_data])

                # Simple linear regression (normal equation)
                X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
                coefficients = np.linalg.lstsq(X_with_bias, y, rcond=None)[0]
            else:
                # Fallback: simple linear regression without numpy
                coefficients = [0.0] * (len(self.feature_columns) + 1)  # +1 for bias term

            return {
                'coefficients': coefficients.tolist() if hasattr(coefficients, 'tolist') else coefficients,
                'feature_columns': self.feature_columns,
                'algorithm': 'linear_regression'
            }

        except Exception as e:
            self.logger.error(f"Linear regression training failed: {e}")
            return {}

    def _train_decision_tree(self, training_data: List[Dict[str, Any]],
                           target_metric: OptimizationTarget) -> Dict[str, Any]:
        """Train decision tree model (simplified)."""
        try:
            # Simple decision tree implementation
            # In practice, would use scikit-learn or similar

            if NUMPY_AVAILABLE:
                # Calculate feature importance based on variance
                X = np.array([[data[col] for col in self.feature_columns] for data in training_data])
                y = np.array([data['target'] for data in training_data])

                feature_importance = {}
                for i, col in enumerate(self.feature_columns):
                    feature_values = X[:, i]
                    feature_importance[col] = np.var(feature_values) / np.var(y) if np.var(y) > 0 else 0
            else:
                # Fallback: simple feature importance without numpy
                feature_importance = {}
                for col in self.feature_columns:
                    feature_importance[col] = 0.1  # Default importance

            return {
                'feature_importance': feature_importance,
                'feature_columns': self.feature_columns,
                'algorithm': 'decision_tree',
                'training_samples': len(training_data)
            }

        except Exception as e:
            self.logger.error(f"Decision tree training failed: {e}")
            return {}

    def _calculate_model_accuracy(self, model_data: Dict[str, Any],
                                training_data: List[Dict[str, Any]],
                                target_metric: OptimizationTarget) -> float:
        """Calculate model accuracy."""
        try:
            if not model_data or not training_data:
                return 0.0

            # Simple accuracy calculation (R² for regression)
            predictions = []
            actuals = []

            for data in training_data:
                if model_data.get('algorithm') == 'linear_regression':
                    pred = self._predict_linear_regression(model_data, data)
                else:
                    pred = data['target']  # Simplified for decision tree

                predictions.append(pred)
                actuals.append(data['target'])

            # Calculate R²
            ss_res = sum((actual - pred) ** 2 for actual, pred in zip(actuals, predictions))

            if NUMPY_AVAILABLE:
                ss_tot = sum((actual - np.mean(actuals)) ** 2 for actual in actuals)
            else:
                mean_actual = sum(actuals) / len(actuals)
                ss_tot = sum((actual - mean_actual) ** 2 for actual in actuals)

            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            return max(0, min(1, r_squared))  # Clamp between 0 and 1

        except Exception as e:
            self.logger.error(f"Accuracy calculation failed: {e}")
            return 0.0

    def _predict_linear_regression(self, model_data: Dict[str, Any],
                                 features: Dict[str, Any]) -> float:
        """Make prediction using linear regression model."""
        try:
            coefficients = model_data.get('coefficients', [])
            if not coefficients:
                return 0.0

            # Calculate prediction
            prediction = coefficients[0]  # Bias term
            for i, col in enumerate(self.feature_columns):
                if i + 1 < len(coefficients):
                    prediction += coefficients[i + 1] * features.get(col, 0)

            return prediction

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return 0.0

    def _calculate_feature_importance(self, model_data: Dict[str, Any],
                                    target_metric: OptimizationTarget) -> Dict[str, float]:
        """Calculate feature importance."""
        if model_data.get('algorithm') == 'linear_regression':
            coefficients = model_data.get('coefficients', [])
            importance = {}
            for i, col in enumerate(self.feature_columns):
                if i + 1 < len(coefficients):
                    importance[col] = abs(coefficients[i + 1])
            return importance
        elif model_data.get('algorithm') == 'decision_tree':
            return model_data.get('feature_importance', {})
        else:
            return {}

    def _get_settings_for_hash(self, settings_hash: str) -> Optional[OptimizationSettings]:
        """Get optimization settings for hash."""
        try:
            with sqlite3.connect(self.database.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM optimization_settings WHERE settings_hash = ?', (settings_hash,))
                row = cursor.fetchone()

                if row:
                    return OptimizationSettings(
                        cpu_cores_performance=row[1],
                        cpu_cores_efficiency=row[2],
                        cpu_frequency_scaling=row[3],
                        memory_allocation_strategy=row[4],
                        gpu_acceleration_enabled=bool(row[5]),
                        gpu_memory_pool_size=row[6],
                        thermal_threshold=row[7],
                        power_limit=row[8],
                        optimization_level=row[9],
                        workload_specific_settings=json.loads(row[10]) if row[10] else {}
                    )
                return None

        except Exception as e:
            self.logger.error(f"Failed to get settings: {e}")
            return None

    def _save_model_to_database(self, model):
        """Save learning model to database."""
        try:
            with sqlite3.connect(self.database.db_path) as conn:
                cursor = conn.cursor()
                
                # Handle both dictionary and object formats
                if isinstance(model, dict):
                    model_id = model.get('model_id', 'unknown')
                    algorithm = model.get('algorithm', 'unknown')
                    target_metric = model.get('target_metric', 'unknown')
                    accuracy = model.get('accuracy', 0.0)
                    last_trained = model.get('created_at', time.time())
                    training_samples = model.get('training_samples', 0)
                    model_data = model.get('coefficients', {})
                    feature_importance = model.get('feature_importance', {})
                else:
                    # Object format
                    model_id = model.model_id
                    algorithm = model.algorithm.value
                    target_metric = model.target_metric.value
                    accuracy = model.accuracy
                    last_trained = model.last_trained
                    training_samples = model.training_samples
                    model_data = model.model_data
                    feature_importance = model.feature_importance
                
                cursor.execute('''
                    INSERT OR REPLACE INTO learning_models (
                        model_id, algorithm, target_metric, accuracy,
                        last_trained, training_samples, model_data, feature_importance
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    model_id, algorithm, target_metric,
                    accuracy, last_trained, training_samples,
                    pickle.dumps(model_data), json.dumps(feature_importance)
                ))
                conn.commit()

        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")

    def predict_optimal_settings(self, workload_type: str,
                               target_metric: OptimizationTarget) -> Optional[OptimizationSettings]:
        """Predict optimal settings for workload and target."""
        try:
            model_id = f"{target_metric.value}_linear_regression"
            if model_id not in self.models:
                self.logger.warning(f"Model {model_id} not found")
                return None

            model = self.models[model_id]
            if not model.model_data:
                return None

            # Get baseline settings for workload
            baseline_settings = self._get_baseline_settings(workload_type)

            # Optimize settings using model
            optimal_settings = self._optimize_settings_with_model(model, baseline_settings, target_metric)

            return optimal_settings

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return None

    def _get_baseline_settings(self, workload_type: str) -> OptimizationSettings:
        """Get baseline settings for workload type."""
        # Default baseline settings
        baseline = OptimizationSettings(
            cpu_cores_performance=4,
            cpu_cores_efficiency=4,
            cpu_frequency_scaling=1.0,
            memory_allocation_strategy='balanced',
            gpu_acceleration_enabled=True,
            gpu_memory_pool_size=500.0,
            thermal_threshold=85.0,
            power_limit=100.0,
            optimization_level='balanced'
        )

        # Adjust based on workload type
        if workload_type == 'backtesting':
            baseline.cpu_cores_performance = 4
            baseline.cpu_cores_efficiency = 0
            baseline.gpu_acceleration_enabled = False
        elif workload_type == 'ml_training':
            baseline.cpu_cores_performance = 3
            baseline.cpu_cores_efficiency = 1
            baseline.gpu_acceleration_enabled = True
        elif workload_type == 'data_processing':
            baseline.cpu_cores_performance = 2
            baseline.cpu_cores_efficiency = 2
            baseline.gpu_acceleration_enabled = False

        return baseline

    def _optimize_settings_with_model(self, model: LearningModel,
                                    baseline: OptimizationSettings,
                                    target_metric: OptimizationTarget) -> OptimizationSettings:
        """Optimize settings using trained model."""
        try:
            # Convert baseline to feature vector
            features = {
                'cpu_cores_performance': baseline.cpu_cores_performance,
                'cpu_cores_efficiency': baseline.cpu_cores_efficiency,
                'cpu_frequency_scaling': baseline.cpu_frequency_scaling,
                'memory_allocation_strategy': 1 if baseline.memory_allocation_strategy == 'aggressive' else 0,
                'gpu_acceleration_enabled': 1 if baseline.gpu_acceleration_enabled else 0,
                'gpu_memory_pool_size': baseline.gpu_memory_pool_size,
                'thermal_threshold': baseline.thermal_threshold,
                'power_limit': baseline.power_limit,
                'optimization_level': 1 if baseline.optimization_level == 'aggressive' else 0
            }

            # Simple optimization: try different values and pick best
            best_settings = baseline
            best_score = self._predict_linear_regression(model.model_data, features)

            # Try variations
            variations = [
                {'cpu_cores_performance': 3, 'cpu_cores_efficiency': 1},
                {'cpu_cores_performance': 2, 'cpu_cores_efficiency': 2},
                {'cpu_cores_performance': 1, 'cpu_cores_efficiency': 3},
                {'gpu_acceleration_enabled': 0},
                {'memory_allocation_strategy': 1},
                {'optimization_level': 1}
            ]

            for variation in variations:
                test_features = features.copy()
                test_features.update(variation)

                score = self._predict_linear_regression(model.model_data, test_features)

                # For some targets, lower is better
                if target_metric in [OptimizationTarget.POWER_CONSUMPTION, OptimizationTarget.THERMAL_MANAGEMENT, OptimizationTarget.MEMORY_USAGE]:
                    if score < best_score:
                        best_score = score
                        best_settings = self._apply_variation(baseline, variation)
                else:
                    if score > best_score:
                        best_score = score
                        best_settings = self._apply_variation(baseline, variation)

            return best_settings

        except Exception as e:
            self.logger.error(f"Settings optimization failed: {e}")
            return baseline

    def _apply_variation(self, baseline: OptimizationSettings,
                        variation: Dict[str, Any]) -> OptimizationSettings:
        """Apply variation to baseline settings."""
        new_settings = OptimizationSettings(
            cpu_cores_performance=variation.get('cpu_cores_performance', baseline.cpu_cores_performance),
            cpu_cores_efficiency=variation.get('cpu_cores_efficiency', baseline.cpu_cores_efficiency),
            cpu_frequency_scaling=baseline.cpu_frequency_scaling,
            memory_allocation_strategy='aggressive' if variation.get('memory_allocation_strategy') else baseline.memory_allocation_strategy,
            gpu_acceleration_enabled=variation.get('gpu_acceleration_enabled', baseline.gpu_acceleration_enabled),
            gpu_memory_pool_size=baseline.gpu_memory_pool_size,
            thermal_threshold=baseline.thermal_threshold,
            power_limit=baseline.power_limit,
            optimization_level='aggressive' if variation.get('optimization_level') else baseline.optimization_level
        )
        return new_settings

class AdaptiveOptimizationEngine:
    """Main adaptive optimization engine."""

    def __init__(self, database_path: str = "optimization_performance.db"):
        self.logger = logger.getChild('AdaptiveOptimizationEngine')

        # Initialize components
        self.database = PerformanceDatabase(database_path)
        self.learner = OptimizationLearner(self.database)

        # Hardware managers
        self.hardware_manager: Optional[UnifiedHardwareManager] = None
        self.cpu_optimizer: Optional[AdvancedM1CPUOptimizer] = None
        self.gpu_manager: Optional[EnhancedM1GPUManager] = None
        self.memory_optimizer: Optional[AdvancedM1MemoryOptimizer] = None

        # Learning state
        self.learning_enabled = True
        self.auto_tuning_enabled = True
        self.learning_interval = 3600  # 1 hour
        self.last_learning_time = 0

        # Performance tracking
        self.current_workload: Optional[WorkloadType] = None
        self.current_target: Optional[OptimizationTarget] = None
        self.performance_history: deque = deque(maxlen=1000)

        # Auto-tuning
        self.auto_tuning_thread: Optional[threading.Thread] = None
        self.auto_tuning_active = False

        self.logger.info("🧠 Adaptive Optimization Engine initialized")

    def initialize_hardware_managers(self):
        """Initialize hardware managers."""
        try:
            from .unified_hardware_manager import get_unified_hardware_manager
            from .advanced_cpu_optimizer import get_advanced_cpu_optimizer
            from .enhanced_gpu_manager import get_enhanced_gpu_manager
            from .advanced_memory_optimizer import get_advanced_memory_optimizer

            self.hardware_manager = get_unified_hardware_manager()
            self.cpu_optimizer = get_advanced_cpu_optimizer()
            self.gpu_manager = get_enhanced_gpu_manager()
            self.memory_optimizer = get_advanced_memory_optimizer()

            self.logger.info("🔧 Hardware managers initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize hardware managers: {e}")

    def start_learning(self):
        """Start the learning process."""
        if not self.learning_enabled:
            return

        self.logger.info("🧠 Starting optimization learning")

        # Train models for different targets
        targets = [
            OptimizationTarget.PERFORMANCE,
            OptimizationTarget.EFFICIENCY,
            OptimizationTarget.POWER_CONSUMPTION,
            OptimizationTarget.BALANCED
        ]

        for target in targets:
            success = self.learner.train_model(target, LearningAlgorithm.LINEAR_REGRESSION)
            if success:
                self.logger.info(f"✅ Trained model for {target.value}")
            else:
                self.logger.warning(f"⚠️ Failed to train model for {target.value}")

        self.last_learning_time = time.time()

    def start_auto_tuning(self):
        """Start automatic tuning."""
        if not self.auto_tuning_enabled:
            return

        self.auto_tuning_active = True
        self.auto_tuning_thread = threading.Thread(
            target=self._auto_tuning_loop,
            daemon=True
        )
        self.auto_tuning_thread.start()

        self.logger.info("🎛️ Auto-tuning started")

    def stop_auto_tuning(self):
        """Stop automatic tuning."""
        self.auto_tuning_active = False
        if self.auto_tuning_thread:
            self.auto_tuning_thread.join(timeout=2.0)
        self.logger.info("🛑 Auto-tuning stopped")

    def _auto_tuning_loop(self):
        """Auto-tuning loop."""
        while self.auto_tuning_active:
            try:
                # Check if learning is needed
                if time.time() - self.last_learning_time > self.learning_interval:
                    self.start_learning()

                # Perform auto-tuning if we have enough data
                if len(self.performance_history) > 10:
                    self._perform_auto_tuning()

                time.sleep(300)  # Check every 5 minutes

            except Exception as e:
                self.logger.error(f"Auto-tuning error: {e}")
                time.sleep(600)  # Wait 10 minutes on error

    def _perform_auto_tuning(self):
        """Perform automatic tuning."""
        try:
            if not self.current_workload or not self.current_target:
                return

            # Get optimal settings from learner
            optimal_settings = self.learner.predict_optimal_settings(
                self.current_workload.value, self.current_target
            )

            if optimal_settings:
                self.logger.info(f"🎯 Auto-tuning for {self.current_workload.value} ({self.current_target.value})")
                self._apply_optimization_settings(optimal_settings)

        except Exception as e:
            self.logger.error(f"Auto-tuning failed: {e}")

    def optimize_for_workload(self, workload_type: WorkloadType,
                            target: OptimizationTarget = OptimizationTarget.BALANCED) -> bool:
        """Optimize for specific workload and target."""
        try:
            self.current_workload = workload_type
            self.current_target = target

            self.logger.info(f"🎯 Optimizing for {workload_type.value} (target: {target.value})")

            # Get optimal settings from learner
            optimal_settings = self.learner.predict_optimal_settings(workload_type.value, target)

            if optimal_settings:
                success = self._apply_optimization_settings(optimal_settings)
                if success:
                    # Store settings for future learning
                    settings_hash = self.database.store_optimization_settings(optimal_settings)
                    self.logger.info(f"✅ Applied optimal settings (hash: {settings_hash})")
                    return True

            # Fallback to hardware manager optimization
            if self.hardware_manager:
                optimization_level = self._target_to_optimization_level(target)
                return self.hardware_manager.optimize_for_workload(workload_type, optimization_level)

            return False

        except Exception as e:
            self.logger.error(f"Workload optimization failed: {e}")
            return False

    def _apply_optimization_settings(self, settings: OptimizationSettings) -> bool:
        """Apply optimization settings to hardware."""
        try:
            if not self.hardware_manager:
                return False

            # Apply CPU settings
            if self.cpu_optimizer:
                self.cpu_optimizer.performance_cores = settings.cpu_cores_performance
                self.cpu_optimizer.efficiency_cores = settings.cpu_cores_efficiency

            # Apply memory settings
            if self.memory_optimizer:
                if settings.memory_allocation_strategy == 'aggressive':
                    self.memory_optimizer.set_memory_strategy(MemoryStrategy.AGGRESSIVE)
                elif settings.memory_allocation_strategy == 'conservative':
                    self.memory_optimizer.set_memory_strategy(MemoryStrategy.CONSERVATIVE)
                else:
                    self.memory_optimizer.set_memory_strategy(MemoryStrategy.BALANCED)

            # Apply GPU settings
            if self.gpu_manager and not settings.gpu_acceleration_enabled:
                # Disable GPU acceleration if needed
                pass

            return True

        except Exception as e:
            self.logger.error(f"Failed to apply settings: {e}")
            return False

    def _target_to_optimization_level(self, target: OptimizationTarget) -> OptimizationLevel:
        """Convert optimization target to optimization level."""
        if target == OptimizationTarget.PERFORMANCE:
            return OptimizationLevel.AGGRESSIVE
        elif target == OptimizationTarget.POWER_CONSUMPTION:
            return OptimizationLevel.MINIMAL
        else:
            return OptimizationLevel.BALANCED

    def record_performance(self, execution_time: float, throughput: float = 0.0,
                          error_rate: float = 0.0) -> bool:
        """Record performance metrics for learning."""
        try:
            if not self.current_workload or not self.current_target:
                return False

            # Get current hardware metrics
            cpu_info = self.cpu_optimizer.get_advanced_cpu_info() if self.cpu_optimizer else {}
            gpu_info = self.gpu_manager.get_enhanced_gpu_info() if self.gpu_manager else {}
            memory_stats = self.memory_optimizer.get_advanced_memory_stats() if self.memory_optimizer else {}

            # Calculate performance score
            performance_score = self._calculate_performance_score(
                execution_time, throughput, error_rate, memory_stats
            )

            # Create performance metrics
            metrics = PerformanceMetrics(
                timestamp=time.time(),
                workload_type=self.current_workload.value,
                optimization_level='adaptive',
                cpu_usage=cpu_info.get('cpu_usage', 0.0),
                memory_usage=memory_stats.get('memory_percent', 0.0),
                gpu_usage=gpu_info.get('gpu_usage', 0.0),
                temperature=cpu_info.get('thermal_stats', {}).get('current_temperature', 45.0),
                power_consumption=cpu_info.get('power_stats', {}).get('average_power', 10.0),
                execution_time=execution_time,
                throughput=throughput,
                error_rate=error_rate,
                optimization_target=self.current_target.value,
                settings_hash='',  # Will be filled by database
                performance_score=performance_score
            )

            # Store in database
            success = self.database.store_performance_metrics(metrics)

            # Add to performance history
            self.performance_history.append(metrics)

            if success:
                self.logger.debug(f"📊 Recorded performance: {performance_score:.3f}")

            return success

        except Exception as e:
            self.logger.error(f"Failed to record performance: {e}")
            return False

    def _calculate_performance_score(self, execution_time: float, throughput: float,
                                   error_rate: float, memory_stats: Dict[str, Any]) -> float:
        """Calculate overall performance score."""
        try:
            # Normalize metrics (higher is better for most)
            time_score = max(0, 1 - (execution_time / 100))  # Assume 100s is max reasonable time
            throughput_score = min(1, throughput / 1000)  # Assume 1000 is max throughput
            error_score = max(0, 1 - error_rate)
            memory_score = max(0, 1 - (memory_stats.get('memory_percent', 0) / 100))

            # Weighted average
            performance_score = (
                time_score * 0.3 +
                throughput_score * 0.3 +
                error_score * 0.2 +
                memory_score * 0.2
            )

            return max(0, min(1, performance_score))  # Clamp between 0 and 1

        except Exception as e:
            self.logger.error(f"Performance score calculation failed: {e}")
            return 0.0

    def get_learning_report(self) -> Dict[str, Any]:
        """Get learning and optimization report."""
        try:
            # Get model information
            model_info = {}
            for model_id, model in self.learner.models.items():
                model_info[model_id] = {
                    'algorithm': model.algorithm.value,
                    'target_metric': model.target_metric.value,
                    'accuracy': model.accuracy,
                    'training_samples': model.training_samples,
                    'last_trained': model.last_trained,
                    'feature_importance': model.feature_importance
                }

            # Get performance history
            recent_performance = list(self.performance_history)[-10:] if self.performance_history else []

            return {
                'learning_enabled': self.learning_enabled,
                'auto_tuning_enabled': self.auto_tuning_enabled,
                'current_workload': self.current_workload.value if self.current_workload else None,
                'current_target': self.current_target.value if self.current_target else None,
                'models': model_info,
                'recent_performance': [
                    {
                        'timestamp': m.timestamp,
                        'workload_type': m.workload_type,
                        'performance_score': m.performance_score,
                        'execution_time': m.execution_time
                    }
                    for m in recent_performance
                ],
                'learning_statistics': {
                    'total_performance_records': len(self.performance_history),
                    'last_learning_time': self.last_learning_time,
                    'learning_interval': self.learning_interval
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to generate learning report: {e}")
            return {'error': str(e)}

    def get_optimal_strategy(self, operation_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get optimal strategy for a given operation type and context.

        Args:
            operation_type: Type of operation (e.g., 'feature_selection', 'training', 'inference')
            context: Context information including memory pressure, hardware config, etc.

        Returns:
            Dictionary containing optimal strategy configuration
        """
        try:
            # Extract context information
            memory_pressure = context.get('memory_pressure', 0.0)
            hardware_config = context.get('hardware_config', {})

            # Determine workload type based on operation
            workload_map = {
                'feature_selection': WorkloadType.TRAINING,
                'training': WorkloadType.TRAINING,
                'inference': WorkloadType.INFERENCE,
                'data_processing': WorkloadType.DATA_PROCESSING,
                'optimization': WorkloadType.OPTIMIZATION
            }
            workload_type = workload_map.get(operation_type, WorkloadType.TRAINING)

            # Determine optimization target based on memory pressure
            if memory_pressure > 0.8:
                target = OptimizationTarget.EFFICIENCY
            elif memory_pressure < 0.3:
                target = OptimizationTarget.PERFORMANCE
            else:
                target = OptimizationTarget.BALANCED

            # Get optimal settings
            optimal_settings = self.learner.predict_optimal_settings(workload_type.value, target)

            # Build strategy dictionary
            strategy = {
                'operation_type': operation_type,
                'workload_type': workload_type.value,
                'optimization_target': target.value,
                'memory_pressure': memory_pressure,
                'use_gpu': hardware_config.get('use_gpu', False),
                'batch_size': hardware_config.get('batch_size', 1000),
                'num_threads': hardware_config.get('num_threads', 4),
                'recommended_settings': {}
            }

            # Add optimal settings if available
            if optimal_settings:
                strategy['recommended_settings'] = {
                    'cpu_cores_performance': optimal_settings.cpu_cores_performance,
                    'cpu_cores_efficiency': optimal_settings.cpu_cores_efficiency,
                    'memory_allocation_strategy': optimal_settings.memory_allocation_strategy,
                    'gpu_acceleration_enabled': optimal_settings.gpu_acceleration_enabled,
                    'optimization_level': optimal_settings.optimization_level
                }
            else:
                # Provide default recommendations based on target
                if target == OptimizationTarget.PERFORMANCE:
                    strategy['recommended_settings'] = {
                        'cpu_cores_performance': 8,
                        'cpu_cores_efficiency': 2,
                        'memory_allocation_strategy': 'aggressive',
                        'gpu_acceleration_enabled': True,
                        'optimization_level': 'aggressive'
                    }
                elif target == OptimizationTarget.EFFICIENCY:
                    strategy['recommended_settings'] = {
                        'cpu_cores_performance': 4,
                        'cpu_cores_efficiency': 4,
                        'memory_allocation_strategy': 'conservative',
                        'gpu_acceleration_enabled': False,
                        'optimization_level': 'minimal'
                    }
                else:
                    strategy['recommended_settings'] = {
                        'cpu_cores_performance': 6,
                        'cpu_cores_efficiency': 2,
                        'memory_allocation_strategy': 'balanced',
                        'gpu_acceleration_enabled': True,
                        'optimization_level': 'balanced'
                    }

            self.logger.debug(f"🎯 Optimal strategy for {operation_type}: {strategy['optimization_target']}")
            return strategy

        except Exception as e:
            self.logger.error(f"Failed to get optimal strategy: {e}")
            # Return a safe default strategy
            return {
                'operation_type': operation_type,
                'workload_type': 'training',
                'optimization_target': 'balanced',
                'memory_pressure': context.get('memory_pressure', 0.5),
                'use_gpu': False,
                'batch_size': 1000,
                'num_threads': 4,
                'recommended_settings': {
                    'cpu_cores_performance': 6,
                    'cpu_cores_efficiency': 2,
                    'memory_allocation_strategy': 'balanced',
                    'gpu_acceleration_enabled': False,
                    'optimization_level': 'balanced'
                },
                'error': str(e)
            }

# Global instance
_adaptive_optimization_engine: Optional[AdaptiveOptimizationEngine] = None

def get_adaptive_optimization_engine() -> AdaptiveOptimizationEngine:
    """Get the global adaptive optimization engine instance."""
    global _adaptive_optimization_engine

    if _adaptive_optimization_engine is None:
        _adaptive_optimization_engine = AdaptiveOptimizationEngine()
        _adaptive_optimization_engine.initialize_hardware_managers()
        _adaptive_optimization_engine.start_learning()
        _adaptive_optimization_engine.start_auto_tuning()

    return _adaptive_optimization_engine

def optimize_for_workload_adaptive(workload_type: WorkloadType,
                                 target: OptimizationTarget = OptimizationTarget.BALANCED) -> bool:
    """Convenience function for adaptive workload optimization."""
    engine = get_adaptive_optimization_engine()
    return engine.optimize_for_workload(workload_type, target)

def record_performance_adaptive(execution_time: float, throughput: float = 0.0,
                              error_rate: float = 0.0) -> bool:
    """Convenience function for recording performance."""
    engine = get_adaptive_optimization_engine()
    return engine.record_performance(execution_time, throughput, error_rate)
