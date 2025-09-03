"""Specialist trainer component for tactician specialist training."""
import asyncio
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from src.core.decorators import handles_errors, log_execution_time
from src.utils.logger import system_logger
from copy import copy
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class SpecialistTrainer:
    """Handles training of specialist models for different tactics."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the specialist trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('specialist_training', {})
        self.logger = system_logger.getChild('specialist_trainer')
        self.max_iterations = self.config.get('max_iterations', 100)
        self.early_stopping_rounds = self.config.get('early_stopping_rounds', 10)
        self.random_state = self.config.get('random_state', 42)
        self.model_configs = self._initialize_model_configs()

    def _initialize_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize configuration for each model type."""
        return {'lightgbm': {'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': -1, 'random_state': self.random_state}, 'xgboost': {'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 100, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': self.random_state, 'verbosity': 0}, 'random_forest': {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2, 'random_state': self.random_state, 'n_jobs': -1}, 'neural_network': {'hidden_sizes': [128, 64, 32], 'learning_rate': 0.001, 'batch_size': 64, 'epochs': 50, 'dropout_rate': 0.2}}

    @handles_errors(exceptions=(Exception,), default_return={}, context='tactic model training')
    async def train_tactic_models(self, tactic_name: str, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, model_types: List[str], regime_id: str) -> Dict[str, Any]:
        """Train models for a specific tactic.
        
        Args:
            tactic_name: Name of the tactic
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            model_types: List of model types to train
            regime_id: Regime identifier
            
        Returns:
            Dictionary of trained models
        """
        self.logger.info(f'Training {tactic_name} models for regime {regime_id}')
        trained_models = {}
        for model_type in model_types:
            try:
                model = await self._train_single_model(model_type, X_train, y_train, X_val, y_val, tactic_name)
                if model is not None:
                    trained_models[model_type] = model
                    val_pred = model.predict(X_val)
                    val_score = accuracy_score(y_val, val_pred)
                    self.logger.info(f'    {model_type} trained: validation accuracy = {val_score:.4f}')
            except Exception as e:
                self.logger.error(f'Failed to train {model_type}: {str(e)}')
        return trained_models

    async def _train_single_model(self, model_type: str, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, tactic_name: str) -> Optional[Any]:
        """Train a single model of specified type.
        
        Args:
            model_type: Type of model to train
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            tactic_name: Name of the tactic
            
        Returns:
            Trained model or None
        """
        if model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            return await self._train_lightgbm(X_train, y_train, X_val, y_val, tactic_name)
        elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
            return await self._train_xgboost(X_train, y_train, X_val, y_val, tactic_name)
        elif model_type == 'random_forest':
            return await self._train_random_forest(X_train, y_train, X_val, y_val, tactic_name)
        elif model_type == 'neural_network' and TORCH_AVAILABLE:
            return await self._train_neural_network(X_train, y_train, X_val, y_val, tactic_name)
        else:
            self.logger.warning(f'Model type {model_type} not available')
            return None

    async def _train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, tactic_name: str) -> Optional[Any]:
        """Train a LightGBM model."""
        try:
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            params = self.model_configs['lightgbm'].copy()
            params['objective'] = 'binary' if len(np.unique(y_train)) == 2 else 'multiclass'
            if params['objective'] == 'multiclass':
                params['num_class'] = len(np.unique(y_train))
            model = lgb.train(params, train_data, valid_sets=[val_data], num_boost_round=self.max_iterations, callbacks=[lgb.early_stopping(self.early_stopping_rounds), lgb.log_evaluation(0)])
            return model
        except Exception as e:
            self.logger.error(f'LightGBM training failed: {str(e)}')
            return None

    async def _train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, tactic_name: str) -> Optional[Any]:
        """Train an XGBoost model."""
        try:
            params = self.model_configs['xgboost'].copy()
            if len(np.unique(y_train)) == 2:
                model = xgb.XGBClassifier(**params)
            else:
                model = xgb.XGBClassifier(**params, objective='multi:softprob', num_class=len(np.unique(y_train)))
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=self.early_stopping_rounds, verbose=False)
            return model
        except Exception as e:
            self.logger.error(f'XGBoost training failed: {str(e)}')
            return None

    async def _train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, tactic_name: str) -> Optional[Any]:
        """Train a Random Forest model."""
        try:
            params = self.model_configs['random_forest'].copy()
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            return model
        except Exception as e:
            self.logger.error(f'Random Forest training failed: {str(e)}')
            return None

    async def _train_neural_network(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, tactic_name: str) -> Optional[Any]:
        """Train a neural network model."""
        if not TORCH_AVAILABLE:
            return None
        try:
            X_train_tensor = torch.FloatTensor(X_train.values)
            y_train_tensor = torch.LongTensor(y_train.values)
            X_val_tensor = torch.FloatTensor(X_val.values)
            y_val_tensor = torch.LongTensor(y_val.values)
            config = self.model_configs['neural_network']
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
            input_size = X_train.shape[1]
            num_classes = len(np.unique(y_train))
            model = self._create_neural_network(input_size, config['hidden_sizes'], num_classes, config['dropout_rate'])
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
            criterion = nn.CrossEntropyLoss()
            best_val_loss = float('inf')
            patience_counter = 0
            for epoch in range(config['epochs']):
                model.train()
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= self.early_stopping_rounds:
                    break
            return model
        except Exception as e:
            self.logger.error(f'Neural network training failed: {str(e)}')
            return None

    def _create_neural_network(self, input_size: int, hidden_sizes: List[int], num_classes: int, dropout_rate: float) -> nn.Module:
        """Create a neural network architecture."""
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([nn.Linear(prev_size, hidden_size), nn.ReLU(), nn.Dropout(dropout_rate)])
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, num_classes))
        return nn.Sequential(*layers)

    @handles_errors(exceptions=(Exception,), default_return=None, context='tactic ensemble creation')
    async def create_tactic_ensemble(self, tactic_name: str, models: List[Tuple[str, Any]]) -> Optional[Any]:
        """Create an ensemble for a specific tactic.
        
        Args:
            tactic_name: Name of the tactic
            models: List of (name, model) tuples
            
        Returns:
            Ensemble model
        """
        if not models:
            return None
        try:
            ensemble = VotingClassifier(estimators=models, voting='soft', n_jobs=-1)
            self.logger.info(f'Created ensemble for {tactic_name} with {len(models)} models')
            return ensemble
        except Exception as e:
            self.logger.error(f'Failed to create ensemble: {str(e)}')
            return None