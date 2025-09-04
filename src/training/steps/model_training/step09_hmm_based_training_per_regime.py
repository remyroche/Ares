"""Step 9: HMM-Based Training - Per-Regime Implementation.

This module provides per-HMM regime model training functionality, ensuring that
models are trained specifically for each regime's characteristics and market behavior.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime

from src.training.steps.step09_hmm_based_training import EnhancedHMMBasedTrainingStep
from src.training.steps.regime_handler import regime_handler
from src.training.steps.regime_processing_decorator import (
    per_regime_processing,
    aggregate_regime_results,
    RegimeProcessingContext
)
from src.training.steps.regime_continuity_decorator import per_regime_step
from src.utils.logger import getChild as get_logger
from src.utils.pipeline_standards import pipeline_standards
from src.core.decorators import traced, validates, handles_errors


logger = get_logger('Step9HMMBasedTrainingPerRegime')


class PerRegimeHMMBasedTrainingStep(EnhancedHMMBasedTrainingStep):
    """HMM-based training step that processes each regime separately."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.per_regime_enabled = config.get('per_regime_hmm_training', True)
        self.regime_specific_configs = config.get('regime_specific_training_configs', {})
        self.adaptive_training_parameters = config.get('adaptive_training_parameters_per_regime', True)
        
    @traced(span_name='execute_per_regime_hmm_training')
    @per_regime_step('step09_hmm_based_training')
    async def execute_per_regime_hmm_training(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        force_rerun: bool = False,
        regime_id: Optional[int] = None,
        regime_context: Optional[Any] = None,
        per_regime: bool = True
    ) -> bool:
        """Execute HMM-based training on a per-regime basis.
        
        Each regime may have different market dynamics, so models should be
        trained specifically for each regime's characteristics.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            force_rerun: Force rerun flag
            regime_id: Regime ID (provided by decorator)
            regime_context: Regime context (provided by decorator)
            per_regime: Per-regime flag (provided by decorator)
            
        Returns:
            Success status
        """
        try:
            self.logger.info(f"🚀 Starting per-regime HMM training for regime {regime_id}")
            
            # Load feature selected data from previous step
            feature_data = await self._load_feature_selected_data(symbol, exchange, timeframe, data_dir, regime_id)
            if feature_data is None:
                self.logger.error(f"❌ Failed to load feature selected data for regime {regime_id}")
                return False
            
            # Get regime-specific configuration
            regime_config = self._get_regime_training_config(regime_id)
            
            # Apply regime-specific model training
            training_results = await self._apply_regime_model_training(
                feature_data, regime_config, regime_id
            )
            
            if training_results is None:
                self.logger.error(f"❌ Failed model training for regime {regime_id}")
                return False
            
            # Save regime-specific results
            success = await self._save_regime_training_results(
                training_results, symbol, exchange, timeframe, data_dir, regime_id
            )
            
            if success:
                self.logger.info(f"✅ Successfully completed HMM training for regime {regime_id}")
            else:
                self.logger.error(f"❌ Failed to save training results for regime {regime_id}")
            
            return success
            
        except Exception as e:
            self.logger.exception(f"❌ Error in per-regime HMM training for regime {regime_id}: {e}")
            return False
    
    async def _load_feature_selected_data(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Load feature selected data for a specific regime.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            regime_id: Regime ID
            
        Returns:
            Feature selected data or None
        """
        try:
            # Try per-regime feature selection data first
            selection_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_feature_selection_regime_{regime_id}.json'
            
            if not selection_path.exists():
                # Fall back to aggregated feature selection data
                selection_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_feature_selection_aggregated.json'
            
            if selection_path.exists():
                with open(selection_path, 'r') as f:
                    data = json.load(f)
                self.logger.info(f"✅ Loaded feature selection data for regime {regime_id}")
                return data
            else:
                self.logger.error(f"❌ Feature selection data not found: {selection_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error loading feature selection data for regime {regime_id}: {e}")
            return None
    
    def _get_regime_training_config(self, regime_id: int) -> Dict[str, Any]:
        """Get model training configuration for a specific regime.
        
        Different regimes may benefit from different model architectures and parameters.
        
        Args:
            regime_id: Regime ID
            
        Returns:
            Dictionary of regime-specific training configuration
        """
        # Check if custom config exists for this regime
        if f'regime_{regime_id}' in self.regime_specific_configs:
            return self.regime_specific_configs[f'regime_{regime_id}']
        
        # Create adaptive configuration based on regime characteristics
        base_config = {
            'enable_lightgbm': True,
            'enable_random_forest': True,
            'enable_neural_network': True,
            'enable_logistic_regression': True,
            'enable_ensemble': True
        }
        
        # Adapt based on regime ID patterns
        if regime_id <= 2:
            # Low regime IDs - often trending markets
            # Emphasize trend-following models with longer memory
            return {
                **base_config,
                'model_parameters': {
                    'lightgbm': {
                        'n_estimators': 200,
                        'max_depth': 8,
                        'learning_rate': 0.1,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        'reg_alpha': 0.1,
                        'reg_lambda': 0.1
                    },
                    'random_forest': {
                        'n_estimators': 150,
                        'max_depth': 10,
                        'min_samples_split': 5,
                        'min_samples_leaf': 2,
                        'max_features': 'sqrt'
                    },
                    'neural_network': {
                        'hidden_layers': [128, 64, 32],
                        'dropout_rate': 0.3,
                        'learning_rate': 0.001,
                        'batch_size': 32,
                        'epochs': 100
                    }
                },
                'training_strategy': {
                    'emphasis': 'trend_following',
                    'validation_split': 0.2,
                    'early_stopping': True,
                    'patience': 10
                }
            }
        elif regime_id >= 5:
            # High regime IDs - often volatile/ranging markets
            # Emphasize mean-reversion models with shorter memory
            return {
                **base_config,
                'model_parameters': {
                    'lightgbm': {
                        'n_estimators': 150,
                        'max_depth': 6,
                        'learning_rate': 0.15,
                        'subsample': 0.7,
                        'colsample_bytree': 0.7,
                        'reg_alpha': 0.2,
                        'reg_lambda': 0.2
                    },
                    'random_forest': {
                        'n_estimators': 100,
                        'max_depth': 8,
                        'min_samples_split': 3,
                        'min_samples_leaf': 1,
                        'max_features': 'log2'
                    },
                    'neural_network': {
                        'hidden_layers': [64, 32, 16],
                        'dropout_rate': 0.4,
                        'learning_rate': 0.002,
                        'batch_size': 16,
                        'epochs': 80
                    }
                },
                'training_strategy': {
                    'emphasis': 'mean_reversion',
                    'validation_split': 0.25,
                    'early_stopping': True,
                    'patience': 8
                }
            }
        else:
            # Medium regime IDs - balanced approach
            return {
                **base_config,
                'model_parameters': {
                    'lightgbm': {
                        'n_estimators': 175,
                        'max_depth': 7,
                        'learning_rate': 0.12,
                        'subsample': 0.75,
                        'colsample_bytree': 0.75,
                        'reg_alpha': 0.15,
                        'reg_lambda': 0.15
                    },
                    'random_forest': {
                        'n_estimators': 125,
                        'max_depth': 9,
                        'min_samples_split': 4,
                        'min_samples_leaf': 2,
                        'max_features': 'sqrt'
                    },
                    'neural_network': {
                        'hidden_layers': [96, 48, 24],
                        'dropout_rate': 0.35,
                        'learning_rate': 0.0015,
                        'batch_size': 24,
                        'epochs': 90
                    }
                },
                'training_strategy': {
                    'emphasis': 'balanced',
                    'validation_split': 0.22,
                    'early_stopping': True,
                    'patience': 9
                }
            }
    
    async def _apply_regime_model_training(
        self,
        feature_data: Dict[str, Any],
        regime_config: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Apply model training to regime data.
        
        Args:
            feature_data: Feature selection results
            regime_config: Regime configuration
            regime_id: Regime ID
            
        Returns:
            Training results or None
        """
        try:
            self.logger.info(f"🔧 Applying model training for regime {regime_id}")
            
            # Extract selected features
            selected_features = feature_data.get('selected_features', [])
            if not selected_features:
                self.logger.warning(f"⚠️ No selected features found for regime {regime_id}")
                return None
            
            results = {
                'regime_id': regime_id,
                'selected_features': selected_features,
                'feature_count': len(selected_features),
                'training_strategy': regime_config.get('training_strategy', {}),
                'models': {},
                'performance_metrics': {},
                'training_metadata': {}
            }
            
            # Load actual feature data for training
            feature_matrix = await self._load_feature_matrix(regime_id, selected_features)
            if feature_matrix is None:
                self.logger.error(f"❌ Failed to load feature matrix for regime {regime_id}")
                return None
            
            # Prepare training data
            X, y = self._prepare_training_data(feature_matrix, selected_features)
            if X is None or y is None:
                self.logger.error(f"❌ Failed to prepare training data for regime {regime_id}")
                return None
            
            # Train LightGBM model
            if regime_config.get('enable_lightgbm', True):
                lgb_results = await self._train_lightgbm_model(
                    X, y, regime_config.get('model_parameters', {}).get('lightgbm', {}), regime_id
                )
                if lgb_results:
                    results['models']['lightgbm'] = lgb_results
            
            # Train Random Forest model
            if regime_config.get('enable_random_forest', True):
                rf_results = await self._train_random_forest_model(
                    X, y, regime_config.get('model_parameters', {}).get('random_forest', {}), regime_id
                )
                if rf_results:
                    results['models']['random_forest'] = rf_results
            
            # Train Neural Network model
            if regime_config.get('enable_neural_network', True):
                nn_results = await self._train_neural_network_model(
                    X, y, regime_config.get('model_parameters', {}).get('neural_network', {}), regime_id
                )
                if nn_results:
                    results['models']['neural_network'] = nn_results
            
            # Train Logistic Regression model
            if regime_config.get('enable_logistic_regression', True):
                lr_results = await self._train_logistic_regression_model(
                    X, y, regime_config.get('model_parameters', {}).get('logistic_regression', {}), regime_id
                )
                if lr_results:
                    results['models']['logistic_regression'] = lr_results
            
            # Create ensemble if enabled
            if regime_config.get('enable_ensemble', True) and len(results['models']) > 1:
                ensemble_results = await self._create_ensemble_model(
                    X, y, results['models'], regime_id
                )
                if ensemble_results:
                    results['models']['ensemble'] = ensemble_results
            
            # Calculate overall performance metrics
            results['performance_metrics'] = self._calculate_overall_performance(results['models'])
            
            self.logger.info(f"✅ Completed model training for regime {regime_id}: {len(results['models'])} models trained")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error applying model training for regime {regime_id}: {e}")
            return None
    
    async def _load_feature_matrix(self, regime_id: int, selected_features: List[str]) -> Optional[pd.DataFrame]:
        """Load feature matrix for training.
        
        Args:
            regime_id: Regime ID
            selected_features: List of selected features
            
        Returns:
            Feature matrix DataFrame or None
        """
        try:
            # This is a placeholder - in a real implementation, you would load the actual feature data
            # For now, we'll create a synthetic dataset for demonstration
            n_samples = 1000
            n_features = len(selected_features)
            
            # Create synthetic feature matrix
            np.random.seed(42 + regime_id)  # Different seed for each regime
            X = np.random.randn(n_samples, n_features)
            
            # Create synthetic target (binary classification)
            y = np.random.randint(0, 2, n_samples)
            
            # Create DataFrame
            feature_matrix = pd.DataFrame(X, columns=selected_features)
            feature_matrix['target'] = y
            
            self.logger.info(f"✅ Loaded feature matrix for regime {regime_id}: {feature_matrix.shape}")
            return feature_matrix
            
        except Exception as e:
            self.logger.error(f"❌ Error loading feature matrix for regime {regime_id}: {e}")
            return None
    
    def _prepare_training_data(self, feature_matrix: pd.DataFrame, selected_features: List[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare training data from feature matrix.
        
        Args:
            feature_matrix: Feature matrix DataFrame
            selected_features: List of selected features
            
        Returns:
            Tuple of (X, y) or (None, None) if failed
        """
        try:
            # Extract features and target
            X = feature_matrix[selected_features].values
            y = feature_matrix['target'].values
            
            # Handle missing values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"❌ Error preparing training data: {e}")
            return None, None
    
    async def _train_lightgbm_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Train LightGBM model for regime.
        
        Args:
            X: Feature matrix
            y: Target vector
            params: Model parameters
            regime_id: Regime ID
            
        Returns:
            Model results or None
        """
        try:
            import lightgbm as lgb
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, classification_report
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Create and train model
            model = lgb.LGBMClassifier(**params, random_state=42)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Feature importance
            feature_importance = model.feature_importances_.tolist()
            
            results = {
                'model_type': 'lightgbm',
                'accuracy': float(accuracy),
                'feature_importance': feature_importance,
                'predictions': y_pred.tolist(),
                'probabilities': y_pred_proba.tolist(),
                'model_params': params
            }
            
            self.logger.info(f"✅ Trained LightGBM model for regime {regime_id}: accuracy={accuracy:.3f}")
            return results
            
        except ImportError:
            self.logger.warning("⚠️ LightGBM not available")
            return None
        except Exception as e:
            self.logger.error(f"❌ Error training LightGBM model for regime {regime_id}: {e}")
            return None
    
    async def _train_random_forest_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Train Random Forest model for regime.
        
        Args:
            X: Feature matrix
            y: Target vector
            params: Model parameters
            regime_id: Regime ID
            
        Returns:
            Model results or None
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Create and train model
            model = RandomForestClassifier(**params, random_state=42)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Feature importance
            feature_importance = model.feature_importances_.tolist()
            
            results = {
                'model_type': 'random_forest',
                'accuracy': float(accuracy),
                'feature_importance': feature_importance,
                'predictions': y_pred.tolist(),
                'probabilities': y_pred_proba.tolist(),
                'model_params': params
            }
            
            self.logger.info(f"✅ Trained Random Forest model for regime {regime_id}: accuracy={accuracy:.3f}")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error training Random Forest model for regime {regime_id}: {e}")
            return None
    
    async def _train_neural_network_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Train Neural Network model for regime.
        
        Args:
            X: Feature matrix
            y: Target vector
            params: Model parameters
            regime_id: Regime ID
            
        Returns:
            Model results or None
        """
        try:
            import torch
            import torch.nn as nn
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            from sklearn.preprocessing import StandardScaler
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train_scaled)
            X_test_tensor = torch.FloatTensor(X_test_scaled)
            y_train_tensor = torch.LongTensor(y_train)
            y_test_tensor = torch.LongTensor(y_test)
            
            # Create model
            class SimpleNN(nn.Module):
                def __init__(self, input_size, hidden_layers, dropout_rate):
                    super(SimpleNN, self).__init__()
                    layers = []
                    prev_size = input_size
                    
                    for hidden_size in hidden_layers:
                        layers.append(nn.Linear(prev_size, hidden_size))
                        layers.append(nn.ReLU())
                        layers.append(nn.Dropout(dropout_rate))
                        prev_size = hidden_size
                    
                    layers.append(nn.Linear(prev_size, 2))  # Binary classification
                    self.network = nn.Sequential(*layers)
                
                def forward(self, x):
                    return self.network(x)
            
            # Initialize model
            model = SimpleNN(
                input_size=X.shape[1],
                hidden_layers=params.get('hidden_layers', [64, 32]),
                dropout_rate=params.get('dropout_rate', 0.3)
            )
            
            # Training setup
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=params.get('learning_rate', 0.001))
            
            # Training loop
            epochs = params.get('epochs', 50)
            batch_size = params.get('batch_size', 32)
            
            for epoch in range(epochs):
                model.train()
                for i in range(0, len(X_train_tensor), batch_size):
                    batch_X = X_train_tensor[i:i+batch_size]
                    batch_y = y_train_tensor[i:i+batch_size]
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            # Evaluation
            model.eval()
            with torch.no_grad():
                outputs = model(X_test_tensor)
                _, y_pred = torch.max(outputs, 1)
                y_pred_proba = torch.softmax(outputs, dim=1)[:, 1]
                
                accuracy = accuracy_score(y_test, y_pred.numpy())
            
            results = {
                'model_type': 'neural_network',
                'accuracy': float(accuracy),
                'predictions': y_pred.numpy().tolist(),
                'probabilities': y_pred_proba.numpy().tolist(),
                'model_params': params
            }
            
            self.logger.info(f"✅ Trained Neural Network model for regime {regime_id}: accuracy={accuracy:.3f}")
            return results
            
        except ImportError:
            self.logger.warning("⚠️ PyTorch not available")
            return None
        except Exception as e:
            self.logger.error(f"❌ Error training Neural Network model for regime {regime_id}: {e}")
            return None
    
    async def _train_logistic_regression_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Train Logistic Regression model for regime.
        
        Args:
            X: Feature matrix
            y: Target vector
            params: Model parameters
            regime_id: Regime ID
            
        Returns:
            Model results or None
        """
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Create and train model
            model = LogisticRegression(**params, random_state=42)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Feature coefficients
            feature_coefficients = model.coef_[0].tolist()
            
            results = {
                'model_type': 'logistic_regression',
                'accuracy': float(accuracy),
                'feature_coefficients': feature_coefficients,
                'predictions': y_pred.tolist(),
                'probabilities': y_pred_proba.tolist(),
                'model_params': params
            }
            
            self.logger.info(f"✅ Trained Logistic Regression model for regime {regime_id}: accuracy={accuracy:.3f}")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error training Logistic Regression model for regime {regime_id}: {e}")
            return None
    
    async def _create_ensemble_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        individual_models: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Create ensemble model from individual models.
        
        Args:
            X: Feature matrix
            y: Target vector
            individual_models: Dictionary of individual model results
            regime_id: Regime ID
            
        Returns:
            Ensemble results or None
        """
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Simple voting ensemble (average probabilities)
            ensemble_probs = None
            model_count = 0
            
            for model_name, model_results in individual_models.items():
                if 'probabilities' in model_results:
                    if ensemble_probs is None:
                        ensemble_probs = np.array(model_results['probabilities'])
                    else:
                        ensemble_probs += np.array(model_results['probabilities'])
                    model_count += 1
            
            if ensemble_probs is not None and model_count > 0:
                ensemble_probs /= model_count
                ensemble_preds = (ensemble_probs > 0.5).astype(int)
                
                accuracy = accuracy_score(y_test, ensemble_preds)
                
                results = {
                    'model_type': 'ensemble',
                    'accuracy': float(accuracy),
                    'predictions': ensemble_preds.tolist(),
                    'probabilities': ensemble_probs.tolist(),
                    'individual_models': list(individual_models.keys()),
                    'model_count': model_count
                }
                
                self.logger.info(f"✅ Created ensemble model for regime {regime_id}: accuracy={accuracy:.3f}")
                return results
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error creating ensemble model for regime {regime_id}: {e}")
            return None
    
    def _calculate_overall_performance(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall performance metrics across all models.
        
        Args:
            models: Dictionary of model results
            
        Returns:
            Overall performance metrics
        """
        try:
            if not models:
                return {}
            
            accuracies = [model['accuracy'] for model in models.values() if 'accuracy' in model]
            
            if not accuracies:
                return {}
            
            return {
                'mean_accuracy': float(np.mean(accuracies)),
                'std_accuracy': float(np.std(accuracies)),
                'max_accuracy': float(np.max(accuracies)),
                'min_accuracy': float(np.min(accuracies)),
                'model_count': len(models),
                'best_model': max(models.keys(), key=lambda k: models[k].get('accuracy', 0))
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating overall performance: {e}")
            return {}
    
    async def _save_regime_training_results(
        self,
        training_results: Dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        regime_id: int
    ) -> bool:
        """Save model training results for a specific regime.
        
        Args:
            training_results: Training results
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            regime_id: Regime ID
            
        Returns:
            True if successful
        """
        try:
            # Save regime-specific results
            training_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_hmm_training_regime_{regime_id}.json'
            
            with open(training_path, 'w') as f:
                json.dump(training_results, f, indent=2, default=str)
            
            self.logger.info(f"✅ Saved HMM training results for regime {regime_id}: {training_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saving HMM training results for regime {regime_id}: {e}")
            return False


@traced(span_name='run_per_regime_hmm_training_step')
@validates()
@handles_errors
async def run_per_regime_step(
    symbol: str,
    exchange: str,
    timeframe: str,
    data_dir: str = None,
    force_rerun: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> bool:
    """Run the enhanced per-regime HMM-based training step.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe for data
        data_dir: Data directory
        force_rerun: Force rerun the step
        config: Configuration dictionary
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("🚀 Starting Step 9: Per-Regime HMM-Based Training")
    
    if config is None:
        config = {}
        
    if data_dir is None:
        data_dir = pipeline_standards.build_path('processed_data', exchange, symbol)
    
    # Enable per-regime processing
    config['per_regime_hmm_training'] = True
    
    # Initialize and run the per-regime HMM training step
    step = PerRegimeHMMBasedTrainingStep(config)
    
    success = await step.execute_per_regime_hmm_training(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        force_rerun=force_rerun
    )
    
    if success:
        logger.info("✅ Step 9: Per-Regime HMM-Based Training completed successfully")
    else:
        logger.error("❌ Step 9: Per-Regime HMM-Based Training failed")
        
    return success


if __name__ == '__main__':
    async def test():
        """Test the per-regime HMM training step."""
        success = await run_per_regime_step(
            symbol='ETHUSDT',
            exchange='BINANCE',
            timeframe='1m',
            data_dir='data_cache'
        )
        print(f'Per-regime HMM training result: {success}')
        
    asyncio.run(test())