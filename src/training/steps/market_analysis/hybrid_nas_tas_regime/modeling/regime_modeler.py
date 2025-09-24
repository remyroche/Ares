"""
Regime Modeler

Creates coherent regime modeling with economic and financial relevance
based on TAS and NAS inputs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from ..config.hybrid_config import HybridRegimeConfig


class RegimeModeler:
    """
    Regime modeler that creates coherent regime modeling with economic and financial relevance.
    
    This component:
    1. Creates regime models based on TAS and NAS inputs
    2. Provides economic and financial relevance
    3. Tags existing data with regime information
    4. Replaces hmm_clustering functionality
    """
    
    def __init__(self, config: HybridRegimeConfig):
        """
        Initialize Regime Modeler.
        
        Args:
            config: Hybrid regime configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize model components
        self.models = {}
        self.regime_models = {}
        self.model_performance = {}
        
        # Initialize modeling algorithms
        self._initialize_models()
        
        self.logger.info("✅ Regime Modeler initialized")
        self.logger.info(f"🏛️ Economic modeling: {config.economic_modeling_enabled}")
        self.logger.info(f"💰 Financial modeling: {config.financial_modeling_enabled}")
        self.logger.info(f"📊 Number of regimes: {config.n_regimes}")
    
    def _initialize_models(self):
        """Initialize modeling algorithms."""
        try:
            # Random Forest
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            
            # Gradient Boosting
            self.models['gradient_boosting'] = GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            )
            
            # Logistic Regression
            self.models['logistic_regression'] = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
            
            # SVM
            self.models['svm'] = SVC(
                random_state=42,
                probability=True
            )
            
            self.logger.info("✅ Modeling algorithms initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Model initialization failed: {e}")
            raise
    
    def create_model(self, 
                     market_data: Union[pd.DataFrame, np.ndarray],
                     clustering_results: Dict[str, Any],
                     integrated_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create regime model from clustering results.
        
        Args:
            market_data: Market data
            clustering_results: Results from clustering
            integrated_results: Results from TAS/NAS integration
            
        Returns:
            Dictionary with regime model
        """
        start_time = time.time()
        self.logger.info("🏗️ Creating regime model")
        
        try:
            # Prepare data for modeling
            model_data = self._prepare_model_data(market_data, clustering_results, integrated_results)
            
            # Create regime models
            regime_models = self._create_regime_models(model_data)
            
            # Calculate model performance
            model_performance = self._calculate_model_performance(model_data, regime_models)
            
            # Generate regime predictions
            regime_predictions = self._generate_regime_predictions(model_data, regime_models)
            
            # Calculate regime probabilities
            regime_probabilities = self._calculate_regime_probabilities(model_data, regime_models)
            
            # Calculate regime stability
            regime_stability = self._calculate_regime_stability(regime_predictions)
            
            # Calculate economic significance
            economic_significance = self._calculate_economic_significance(
                regime_predictions, model_data
            )
            
            # Calculate financial significance
            financial_significance = self._calculate_financial_significance(
                regime_predictions, model_data
            )
            
            execution_time = time.time() - start_time
            
            self.logger.info(f"✅ Regime model created in {execution_time:.2f}s")
            self.logger.info(f"📊 Model accuracy: {model_performance.get('accuracy', 0.0):.3f}")
            self.logger.info(f"🎯 Economic significance: {np.mean(economic_significance):.3f}")
            self.logger.info(f"💰 Financial significance: {np.mean(financial_significance):.3f}")
            
            return {
                'success': True,
                'regime_models': regime_models,
                'regime_predictions': regime_predictions,
                'regime_probabilities': regime_probabilities,
                'regime_stability': regime_stability,
                'economic_significance': economic_significance,
                'financial_significance': financial_significance,
                'model_performance': model_performance,
                'execution_time': execution_time,
                'metadata': {
                    'n_regimes': len(set(regime_predictions)),
                    'n_samples': len(regime_predictions),
                    'model_types': list(regime_models.keys())
                }
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Regime model creation failed: {e}")
            
            return {
                'success': False,
                'regime_models': {},
                'regime_predictions': np.array([]),
                'regime_probabilities': np.array([]),
                'regime_stability': np.array([]),
                'economic_significance': np.array([]),
                'financial_significance': np.array([]),
                'model_performance': {},
                'execution_time': execution_time,
                'error_message': str(e)
            }
    
    def _prepare_model_data(self, 
                           market_data: Union[pd.DataFrame, np.ndarray],
                           clustering_results: Dict[str, Any],
                           integrated_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for regime modeling."""
        self.logger.info("📊 Preparing data for regime modeling")
        
        # Convert market data to numpy array
        if isinstance(market_data, pd.DataFrame):
            data_array = market_data.values
        else:
            data_array = market_data
        
        # Get clustering labels
        labels = clustering_results.get('labels', np.array([]))
        
        # Get integrated results
        tas_predictions = integrated_results.get('tas_predictions', np.array([]))
        nas_predictions = integrated_results.get('nas_predictions', np.array([]))
        integrated_predictions = integrated_results.get('integrated_predictions', np.array([]))
        
        # Combine features
        features = []
        
        # Market data features
        if len(data_array.shape) > 1:
            features.append(data_array)
        else:
            features.append(data_array.reshape(-1, 1))
        
        # Clustering features
        if len(labels) > 0:
            features.append(labels.reshape(-1, 1))
        
        # TAS features
        if len(tas_predictions) > 0:
            features.append(tas_predictions.reshape(-1, 1))
        
        # NAS features
        if len(nas_predictions) > 0:
            features.append(nas_predictions.reshape(-1, 1))
        
        # Integrated features
        if len(integrated_predictions) > 0:
            features.append(integrated_predictions.reshape(-1, 1))
        
        # Combine all features
        if features:
            combined_features = np.column_stack(features)
        else:
            combined_features = data_array
        
        # Ensure same length
        min_len = min(len(combined_features), len(labels))
        if min_len > 0:
            combined_features = combined_features[:min_len]
            labels = labels[:min_len]
        
        return {
            'features': combined_features,
            'labels': labels,
            'market_data': data_array,
            'clustering_results': clustering_results,
            'integrated_results': integrated_results
        }
    
    def _create_regime_models(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create regime models using different algorithms."""
        self.logger.info("🤖 Creating regime models")
        
        features = model_data['features']
        labels = model_data['labels']
        
        if len(features) == 0 or len(labels) == 0:
            return {}
        
        regime_models = {}
        
        for model_name, model in self.models.items():
            try:
                # Train model
                model.fit(features, labels)
                
                # Store trained model
                regime_models[model_name] = model
                
                self.logger.info(f"✅ {model_name} model trained")
                
            except Exception as e:
                self.logger.warning(f"⚠️ {model_name} model training failed: {e}")
                continue
        
        return regime_models
    
    def _calculate_model_performance(self, 
                                     model_data: Dict[str, Any],
                                     regime_models: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate model performance metrics."""
        self.logger.info("📊 Calculating model performance")
        
        features = model_data['features']
        labels = model_data['labels']
        
        if len(features) == 0 or len(labels) == 0:
            return {}
        
        performance = {}
        
        for model_name, model in regime_models.items():
            try:
                # Make predictions
                predictions = model.predict(features)
                probabilities = model.predict_proba(features) if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(labels, predictions)
                precision = precision_score(labels, predictions, average='weighted', zero_division=0)
                recall = recall_score(labels, predictions, average='weighted', zero_division=0)
                f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
                
                performance[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'predictions': predictions,
                    'probabilities': probabilities
                }
                
            except Exception as e:
                self.logger.warning(f"⚠️ Performance calculation for {model_name} failed: {e}")
                continue
        
        return performance
    
    def _generate_regime_predictions(self, 
                                    model_data: Dict[str, Any],
                                    regime_models: Dict[str, Any]) -> np.ndarray:
        """Generate regime predictions using ensemble of models."""
        self.logger.info("🎯 Generating regime predictions")
        
        features = model_data['features']
        
        if len(features) == 0:
            return np.array([])
        
        # Use ensemble of models for predictions
        predictions_list = []
        
        for model_name, model in regime_models.items():
            try:
                predictions = model.predict(features)
                predictions_list.append(predictions)
            except Exception as e:
                self.logger.warning(f"⚠️ Prediction generation for {model_name} failed: {e}")
                continue
        
        if not predictions_list:
            return np.array([])
        
        # Ensemble prediction (majority vote)
        predictions_array = np.array(predictions_list)
        ensemble_predictions = np.zeros(len(features))
        
        for i in range(len(features)):
            votes = predictions_array[:, i]
            ensemble_predictions[i] = np.bincount(votes.astype(int)).argmax()
        
        return ensemble_predictions
    
    def _calculate_regime_probabilities(self, 
                                        model_data: Dict[str, Any],
                                        regime_models: Dict[str, Any]) -> np.ndarray:
        """Calculate regime probabilities using ensemble of models."""
        self.logger.info("📊 Calculating regime probabilities")
        
        features = model_data['features']
        
        if len(features) == 0:
            return np.array([])
        
        # Use ensemble of models for probabilities
        probabilities_list = []
        
        for model_name, model in regime_models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(features)
                    probabilities_list.append(probabilities)
            except Exception as e:
                self.logger.warning(f"⚠️ Probability calculation for {model_name} failed: {e}")
                continue
        
        if not probabilities_list:
            return np.array([])
        
        # Ensemble probabilities (average)
        probabilities_array = np.array(probabilities_list)
        ensemble_probabilities = np.mean(probabilities_array, axis=0)
        
        return ensemble_probabilities
    
    def _calculate_regime_stability(self, regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate regime stability scores."""
        if len(regime_predictions) < 2:
            return np.array([1.0] * len(regime_predictions))
        
        stability_scores = np.zeros(len(regime_predictions))
        
        for i in range(len(regime_predictions)):
            # Look at surrounding regimes for stability
            window_size = min(10, len(regime_predictions) // 4)
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(regime_predictions), i + window_size // 2 + 1)
            
            window_regimes = regime_predictions[start_idx:end_idx]
            current_regime = regime_predictions[i]
            
            # Stability is based on consistency within window
            consistency = np.sum(window_regimes == current_regime) / len(window_regimes)
            stability_scores[i] = consistency
        
        return stability_scores
    
    def _calculate_economic_significance(self, 
                                         regime_predictions: np.ndarray,
                                         model_data: Dict[str, Any]) -> np.ndarray:
        """Calculate economic significance scores."""
        if not self.config.economic_modeling_enabled:
            return np.array([0.5] * len(regime_predictions))
        
        # Base economic significance on regime stability and market characteristics
        stability_scores = self._calculate_regime_stability(regime_predictions)
        
        # Adjust based on market data characteristics
        market_data = model_data['market_data']
        if len(market_data.shape) > 1 and market_data.shape[1] > 0:
            # Use last column (usually close price) for economic analysis
            price_data = market_data[:, -1]
            if len(price_data) > 1:
                returns = np.diff(price_data) / (price_data[:-1] + 1e-8)
                volatility = np.std(returns) if len(returns) > 0 else 0.0
                volatility_factor = min(1.0, volatility * 10)
            else:
                volatility_factor = 0.5
        else:
            volatility_factor = 0.5
        
        # Combine stability and volatility
        economic_significance = 0.7 * stability_scores + 0.3 * volatility_factor
        
        # Apply economic significance threshold
        economic_significance = np.where(
            economic_significance >= self.config.economic_significance_threshold,
            economic_significance,
            economic_significance * 0.5
        )
        
        return economic_significance
    
    def _calculate_financial_significance(self, 
                                           regime_predictions: np.ndarray,
                                           model_data: Dict[str, Any]) -> np.ndarray:
        """Calculate financial significance scores."""
        if not self.config.financial_modeling_enabled:
            return np.array([0.5] * len(regime_predictions))
        
        # Base financial significance on regime stability and trading viability
        stability_scores = self._calculate_regime_stability(regime_predictions)
        
        # Adjust based on market data characteristics
        market_data = model_data['market_data']
        if len(market_data.shape) > 1 and market_data.shape[1] > 0:
            # Use last column (usually close price) for financial analysis
            price_data = market_data[:, -1]
            if len(price_data) > 1:
                returns = np.diff(price_data) / (price_data[:-1] + 1e-8)
                volatility = np.std(returns) if len(returns) > 0 else 0.0
                volatility_factor = min(1.0, volatility * 10)
            else:
                volatility_factor = 0.5
        else:
            volatility_factor = 0.5
        
        # Combine stability and volatility
        financial_significance = 0.6 * stability_scores + 0.4 * volatility_factor
        
        # Apply financial significance threshold
        financial_significance = np.where(
            financial_significance >= self.config.financial_significance_threshold,
            financial_significance,
            financial_significance * 0.5
        )
        
        return financial_significance
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of regime models."""
        return {
            "n_models": len(self.regime_models),
            "model_types": list(self.regime_models.keys()),
            "model_performance": self.model_performance,
            "economic_modeling_enabled": self.config.economic_modeling_enabled,
            "financial_modeling_enabled": self.config.financial_modeling_enabled
        }
    
    def update_model_performance(self, performance: Dict[str, Any]):
        """Update model performance metrics."""
        self.model_performance = performance