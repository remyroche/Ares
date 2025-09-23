"""
Tactician NAS Integration

This module integrates Neural Architecture Search (NAS) as a base model
in the Tactician ensemble for 1m timeframe trading decisions.

Key Features:
- NAS discovers optimal neural architectures for 1m timeframe
- Integrates as base model alongside XGBoost, RandomForest, CatBoost, Elastic Net
- Regime-aware architecture discovery
- Optimized for real-time inference (1m timeframe)
- Multi-objective optimization (accuracy + efficiency + robustness)
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time
from dataclasses import dataclass
from datetime import datetime

# Import NAS
from src.utils.ml_common.optimization.neural_architecture_search import (
    search_neural_architecture, ArchitectureConfig, ArchitectureCandidate
)

# Import existing optimization tools
from src.utils.ml_common.optimization.hyperparameter_optimization import HyperparameterOptimizer
from src.utils.ml_common.optimization.regime_aware_hpo import RegimeAwareHyperparameterOptimizer
from src.utils.ml_common.optimization.bayesian_optimization import BayesianOptimizer

# Import existing feature engineering
from src.utils.ml_common.feature_engineering.feature_selection import FeatureSelector
from src.utils.ml_common.feature_engineering.feature_transformation import FeatureTransformer

# Import existing validation
from src.utils.ml_common.validation.cross_validation import CrossValidator
from src.utils.ml_common.validation.overfitting_detection import UniversalOverfittingDetector

# Import logging utilities
from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
    tprint_debug, tprint_progress, tprint_performance, tprint_structured,
    tprint_timer, LogLevel
)

logger = logging.getLogger(__name__)


@dataclass
class TacticianNASConfig:
    """Configuration for Tactician NAS integration."""
    
    # NAS search parameters
    n_trials: int = 30  # Reduced for faster training
    timeout_seconds: int = 1800  # 30 minutes max
    early_stopping_patience: int = 10
    
    # Architecture constraints for 1m timeframe
    max_layers: int = 6  # Shallow networks for speed
    max_units: int = 256  # Moderate complexity
    min_units: int = 32   # Minimum for meaningful learning
    
    # Multi-objective optimization
    objectives: List[str] = None
    objective_weights: List[float] = None
    
    # Regime awareness
    enable_regime_awareness: bool = True
    regime_adaptation_strength: float = 0.3
    
    # Performance optimization
    enable_early_stopping: bool = True
    enable_model_pruning: bool = True
    memory_limit_gb: float = 4.0
    
    def __post_init__(self):
        if self.objectives is None:
            self.objectives = ['accuracy', 'efficiency', 'robustness']
        if self.objective_weights is None:
            self.objective_weights = [0.5, 0.3, 0.2]  # Balance for 1m timeframe


class TacticianNASIntegration:
    """Integrates NAS as a base model in Tactician ensemble."""
    
    def __init__(self, config: Optional[TacticianNASConfig] = None):
        """Initialize Tactician NAS integration."""
        self.config = config or TacticianNASConfig()
        self.logger = logger.getChild('TacticianNASIntegration')
        self.nas_model = None
        self.architecture = None
        self.training_stats = {}
        
        tprint_info("🧠 Tactician NAS Integration initialized")
        tprint_info(f"📊 Configuration: {self.config.n_trials} trials, {self.config.timeout_seconds}s timeout")
    
    def integrate_nas_model(self, 
                          X_train: np.ndarray, 
                          y_train: np.ndarray,
                          X_val: np.ndarray, 
                          y_val: np.ndarray,
                          regime_labels: Optional[np.ndarray] = None,
                          regime_features: Optional[np.ndarray] = None) -> Optional[Any]:
        """
        Integrate NAS as a base model in Tactician ensemble.
        
        Args:
            X_train: Training features (1m timeframe)
            y_train: Training labels (trading signals)
            X_val: Validation features
            y_val: Validation labels
            regime_labels: Regime labels for regime-aware search (optional)
            regime_features: Regime-specific features (volatility, volume, trend, momentum) (optional)
            
        Returns:
            Trained NAS model or None if integration fails
        """
        tprint_info("🔍 Starting Tactician NAS integration...")
        start_time = time.time()
        
        try:
            # Fast fail: Validate inputs immediately
            if X_train.shape[0] < 1000:
                raise ValueError("Insufficient training data: need at least 1000 samples")
            
            if X_train.shape[1] > 200:
                tprint_warning("⚠️ High feature count detected, applying comprehensive feature selection pipeline...")
                
                # Step 1: mRMR (Minimum Redundancy Maximum Relevance)
                tprint_info("🔍 Step 1: Applying mRMR feature selection...")
                from src.utils.ml_common.feature_engineering.mrmr_selection import MRMRSelector
                mrmr_selector = MRMRSelector(k=80, method='fscore')
                X_train = mrmr_selector.fit_transform(X_train, y_train)
                X_val = mrmr_selector.transform(X_val)
                tprint_success(f"✅ mRMR: → {X_train.shape[1]} features")
                
                # Step 2: Mutual Information filtering
                tprint_info("🔍 Step 2: Applying Mutual Information filtering...")
                from src.utils.ml_common.feature_engineering.mutual_info_selection import MutualInfoSelector
                mi_selector = MutualInfoSelector(k=70, method='mutual_info')
                X_train = mi_selector.fit_transform(X_train, y_train)
                X_val = mi_selector.transform(X_val)
                tprint_success(f"✅ MI: → {X_train.shape[1]} features")
                
                # Step 3: LASSO regularization
                tprint_info("🔍 Step 3: Applying LASSO regularization...")
                from src.utils.ml_common.feature_engineering.lasso_selection import LassoSelector
                lasso_selector = LassoSelector(alpha=0.01, max_features=65)
                X_train = lasso_selector.fit_transform(X_train, y_train)
                X_val = lasso_selector.transform(X_val)
                tprint_success(f"✅ LASSO: → {X_train.shape[1]} features")
                
                # Step 4: RandomForest final selection (down to 60)
                tprint_info("🔍 Step 4: Applying RandomForest final selection to 60 features...")
                from src.utils.ml_common.feature_engineering.random_forest_selection import RandomForestSelector
                rf_selector = RandomForestSelector(n_estimators=100, max_features=60, method='importance')
                X_train = rf_selector.fit_transform(X_train, y_train)
                X_val = rf_selector.transform(X_val)
                tprint_success(f"✅ RandomForest: → {X_train.shape[1]} features")
                tprint_success(f"🎯 Final feature reduction to {X_train.shape[1]} features using comprehensive pipeline")
            
            # Integrate regime-specific features if provided
            if regime_features is not None:
                tprint_info("🧠 Integrating regime-specific features...")
                X_train = np.hstack([X_train, regime_features[:len(X_train)]])
                X_val = np.hstack([X_val, regime_features[len(X_train):len(X_train)+len(X_val)]])
                tprint_success(f"✅ Regime features integrated: {X_train.shape[1]} total features")
            
            # Configure NAS for Tactician requirements using existing optimization tools
            nas_config = ArchitectureConfig(
                n_trials=self.config.n_trials,
                timeout_seconds=self.config.timeout_seconds,
                max_layers=self.config.max_layers,
                max_units=self.config.max_units,
                min_units=self.config.min_units,
                objectives=self.config.objectives,
                objective_weights=self.config.objective_weights,
                enable_regime_awareness=self.config.enable_regime_awareness,
                early_stopping_patience=self.config.early_stopping_patience
            )
            
            # Search for optimal architecture using existing optimization framework
            tprint_info("🔍 Searching for optimal neural architecture...")
            
            # Use existing regime-aware HPO if regime labels provided
            if regime_labels is not None:
                tprint_info("🎯 Using regime-aware optimization...")
                regime_hpo = RegimeAwareHyperparameterOptimizer()
                # Integrate with existing regime-aware optimization
                self.architecture = search_neural_architecture(
                    X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val,
                    config=nas_config,
                    regime_labels=regime_labels
                )
            else:
                # Use standard optimization
                self.architecture = search_neural_architecture(
                    X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val,
                    config=nas_config
                )
            
            # Fast fail: Validate architecture discovery
            if self.architecture is None:
                raise RuntimeError("Architecture discovery failed - no valid architecture found")
            
            if self.architecture.overall_score < 0.5:
                raise RuntimeError(f"Architecture quality too low: {self.architecture.overall_score:.3f} < 0.5")
            
            # Create the discovered neural network
            tprint_info("🏗️ Creating NAS model from discovered architecture...")
            self.nas_model = self._create_nas_model(self.architecture, X_train.shape[1])
            
            # Train the NAS model
            tprint_info("🏋️ Training NAS model...")
            self.nas_model = self._train_nas_model(
                self.nas_model, X_train, y_train, X_val, y_val
            )
            
            # Fast fail: Validate model performance using existing overfitting detection
            tprint_info("🔍 Validating NAS model performance...")
            overfitting_detector = UniversalOverfittingDetector()
            overfitting_report = overfitting_detector.detect_overfitting(
                train_predictions=self.nas_model.predict(X_train),
                val_predictions=self.nas_model.predict(X_val),
                train_labels=y_train,
                val_labels=y_val,
                model_name="tactician_nas",
                model_type="neural_network"
            )
            
            # Fast fail: Check for severe overfitting
            if overfitting_report.severity == "high":
                raise RuntimeError(f"Severe overfitting detected: {overfitting_report.severity}")
            
            if overfitting_report.accuracy_gap > 0.2:
                raise RuntimeError(f"High accuracy gap detected: {overfitting_report.accuracy_gap:.3f} > 0.2")
            
            tprint_success("✅ NAS model validation passed")
            
            # Calculate training statistics
            training_time = time.time() - start_time
            self.training_stats = {
                'training_time': training_time,
                'architecture_params': self.architecture.total_params,
                'architecture_score': self.architecture.overall_score,
                'architecture_layers': len(self.architecture.layers),
                'success': True
            }
            
            tprint_success("✅ Tactician NAS integration completed successfully")
            tprint_info(f"📊 Architecture: {self.architecture.total_params} parameters")
            tprint_info(f"📊 Score: {self.architecture.overall_score:.4f}")
            tprint_info(f"⏱️ Training time: {training_time:.2f}s")
            
            return self.nas_model
            
        except Exception as e:
            tprint_error(f"❌ Tactician NAS integration failed: {e}")
            self.training_stats = {
                'success': False,
                'error': str(e),
                'training_time': time.time() - start_time
            }
            return None
    
    def _create_nas_model(self, architecture: ArchitectureCandidate, input_size: int) -> Any:
        """Create neural network model from discovered architecture."""
        tprint_info("🏗️ Creating neural network from discovered architecture...")
        
        try:
            # Try PyTorch first (preferred for NAS)
            try:
                import torch
                import torch.nn as nn
                
                class TacticianNASModel(nn.Module):
                    def __init__(self, architecture, input_size):
                        super().__init__()
                        self.layers = nn.ModuleList()
                        
                        prev_size = input_size
                        for layer_config in architecture.layers:
                            if layer_config['type'] == 'dense':
                                self.layers.append(nn.Linear(prev_size, layer_config['units']))
                                
                                # Add activation
                                if layer_config['activation'] == 'relu':
                                    self.layers.append(nn.ReLU())
                                elif layer_config['activation'] == 'tanh':
                                    self.layers.append(nn.Tanh())
                                elif layer_config['activation'] == 'swish':
                                    self.layers.append(nn.SiLU())
                                elif layer_config['activation'] == 'gelu':
                                    self.layers.append(nn.GELU())
                                
                                # Add dropout if specified
                                if layer_config['dropout'] > 0:
                                    self.layers.append(nn.Dropout(layer_config['dropout']))
                                
                                prev_size = layer_config['units']
                            
                            elif layer_config['type'] == 'lstm':
                                self.layers.append(nn.LSTM(
                                    prev_size, layer_config['units'], 
                                    batch_first=True, 
                                    dropout=layer_config['dropout']
                                ))
                                prev_size = layer_config['units']
                            
                            elif layer_config['type'] == 'gru':
                                self.layers.append(nn.GRU(
                                    prev_size, layer_config['units'], 
                                    batch_first=True, 
                                    dropout=layer_config['dropout']
                                ))
                                prev_size = layer_config['units']
                        
                        # Output layer for binary classification (trading signal)
                        self.output_layer = nn.Linear(prev_size, 1)
                        self.sigmoid = nn.Sigmoid()
                    
                    def forward(self, x):
                        for layer in self.layers:
                            if isinstance(layer, (nn.LSTM, nn.GRU)):
                                x, _ = layer(x)
                            else:
                                x = layer(x)
                        
                        x = self.output_layer(x)
                        return self.sigmoid(x)
                    
                    def predict(self, x):
                        """Sklearn-compatible predict method."""
                        self.eval()
                        with torch.no_grad():
                            if isinstance(x, np.ndarray):
                                x = torch.FloatTensor(x)
                            predictions = self.forward(x)
                            return predictions.numpy().flatten()
                    
                    def predict_proba(self, x):
                        """Sklearn-compatible predict_proba method."""
                        predictions = self.predict(x)
                        # Return probabilities for both classes
                        return np.column_stack([1 - predictions, predictions])
                
                model = TacticianNASModel(architecture, input_size)
                tprint_success("✅ PyTorch NAS model created")
                return model
                
            except ImportError:
                tprint_warning("⚠️ PyTorch not available, trying TensorFlow...")
                
                # Fallback to TensorFlow
                try:
                    import tensorflow as tf
                    from tensorflow import keras
                    from tensorflow.keras import layers
                    
                    inputs = keras.Input(shape=(input_size,))
                    x = inputs
                    
                    for layer_config in architecture.layers:
                        if layer_config['type'] == 'dense':
                            x = layers.Dense(
                                layer_config['units'], 
                                activation=layer_config['activation']
                            )(x)
                            if layer_config['dropout'] > 0:
                                x = layers.Dropout(layer_config['dropout'])(x)
                        
                        elif layer_config['type'] == 'lstm':
                            x = layers.LSTM(
                                layer_config['units'], 
                                return_sequences=layer_config.get('return_sequences', False),
                                dropout=layer_config['dropout']
                            )(x)
                        
                        elif layer_config['type'] == 'gru':
                            x = layers.GRU(
                                layer_config['units'], 
                                return_sequences=layer_config.get('return_sequences', False),
                                dropout=layer_config['dropout']
                            )(x)
                    
                    # Output layer for binary classification
                    outputs = layers.Dense(1, activation='sigmoid')(x)
                    
                    model = keras.Model(inputs, outputs)
                    tprint_success("✅ TensorFlow NAS model created")
                    return model
                    
                except ImportError:
                    raise ImportError("Neither PyTorch nor TensorFlow available for NAS model creation")
            
        except Exception as e:
            tprint_error(f"❌ NAS model creation failed: {e}")
            raise
    
    def _train_nas_model(self, model: Any, X_train: np.ndarray, y_train: np.ndarray, 
                       X_val: np.ndarray, y_val: np.ndarray) -> Any:
        """Train the NAS model with optimization for 1m timeframe."""
        tprint_info("🏋️ Training NAS model for 1m timeframe...")
        
        try:
            # Try PyTorch training
            if hasattr(model, 'parameters'):
                import torch
                import torch.optim as optim
                from torch.utils.data import DataLoader, TensorDataset
                
                # Convert to tensors
                X_train_tensor = torch.FloatTensor(X_train)
                y_train_tensor = torch.FloatTensor(y_train)
                X_val_tensor = torch.FloatTensor(X_val)
                y_val_tensor = torch.FloatTensor(y_val)
                
                # Create data loaders
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                
                # Setup training
                criterion = torch.nn.BCELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                
                # Early stopping
                best_val_loss = float('inf')
                patience_counter = 0
                
                # Training loop
                model.train()
                for epoch in range(100):  # More epochs for better training
                    epoch_loss = 0.0
                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                    
                    # Validation
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_val_tensor)
                        val_loss = criterion(val_outputs, y_val_tensor).item()
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= self.config.early_stopping_patience:
                            tprint_info(f"🛑 Early stopping at epoch {epoch}")
                            break
                    
                    if epoch % 10 == 0:
                        tprint_debug(f"Epoch {epoch}: Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                tprint_success("✅ PyTorch NAS model training completed")
                return model
            
            # TensorFlow training
            else:
                model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                # Early stopping callback
                from tensorflow.keras.callbacks import EarlyStopping
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True
                )
                
                model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=100,
                    batch_size=32,
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                tprint_success("✅ TensorFlow NAS model training completed")
                return model
                
        except Exception as e:
            tprint_error(f"❌ NAS model training failed: {e}")
            raise
    
    def get_nas_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from the NAS model."""
        if self.nas_model is None:
            raise ValueError("NAS model not trained. Call integrate_nas_model first.")
        
        try:
            return self.nas_model.predict(X)
        except Exception as e:
            tprint_error(f"❌ NAS prediction failed: {e}")
            raise
    
    def get_nas_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities from the NAS model."""
        if self.nas_model is None:
            raise ValueError("NAS model not trained. Call integrate_nas_model first.")
        
        try:
            if hasattr(self.nas_model, 'predict_proba'):
                return self.nas_model.predict_proba(X)
            else:
                # Fallback to predict method
                predictions = self.nas_model.predict(X)
                return np.column_stack([1 - predictions, predictions])
        except Exception as e:
            tprint_error(f"❌ NAS probability prediction failed: {e}")
            raise
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return self.training_stats.copy()
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """Get information about the discovered architecture."""
        if self.architecture is None:
            return {'message': 'No architecture discovered yet'}
        
        return {
            'total_params': self.architecture.total_params,
            'estimated_flops': self.architecture.estimated_flops,
            'accuracy': self.architecture.accuracy,
            'efficiency_score': self.architecture.efficiency_score,
            'robustness_score': self.architecture.robustness_score,
            'overall_score': self.architecture.overall_score,
            'n_layers': len(self.architecture.layers),
            'layer_types': [layer['type'] for layer in self.architecture.layers],
            'training_time': self.architecture.training_time,
            'convergence_epochs': self.architecture.convergence_epochs
        }


# Convenience function for easy integration
def create_tactician_nas_model(X_train: np.ndarray, 
                             y_train: np.ndarray,
                             X_val: np.ndarray, 
                             y_val: np.ndarray,
                             config: Optional[TacticianNASConfig] = None,
                             regime_labels: Optional[np.ndarray] = None,
                             regime_features: Optional[np.ndarray] = None) -> Optional[Any]:
    """
    Convenience function to create and train a NAS model for Tactician ensemble.
    
    Args:
        X_train: Training features (1m timeframe)
        y_train: Training labels (trading signals)
        X_val: Validation features
        y_val: Validation labels
        config: NAS configuration
        regime_labels: Regime labels for regime-aware search (optional)
        regime_features: Regime-specific features (volatility, volume, trend, momentum) (optional)
        
    Returns:
        Trained NAS model or None if creation fails
    """
    nas_integration = TacticianNASIntegration(config)
    return nas_integration.integrate_nas_model(X_train, y_train, X_val, y_val, regime_labels, regime_features)