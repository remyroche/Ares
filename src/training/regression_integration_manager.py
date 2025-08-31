# src/training/regression_integration_manager.py

import pandas as pd
from typing import Any, Dict, Optional, Tuple
from datetime import datetime

from src.training.regression_profit_predictor import RegressionProfitPredictor
from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors, handle_specific_errors


class RegressionIntegrationManager:
    """
    Integration manager for regression-based profit prediction with existing Analyst/Tactician systems.
    
    This module provides a hybrid approach that combines:
    1. Regression models for profit prediction
    2. Classification models for final decision making
    3. Enhanced position sizing based on predicted returns
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the regression integration manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("RegressionIntegrationManager")
        
        # Initialize regression predictors
        self.analyst_regression_predictor: Optional[RegressionProfitPredictor] = None
        self.tactician_regression_predictor: Optional[RegressionProfitPredictor] = None
        
        # Configuration
        self.integration_config = config.get("regression_integration", {})
        self.enable_analyst_regression = self.integration_config.get("enable_analyst_regression", True)
        self.enable_tactician_regression = self.integration_config.get("enable_tactician_regression", True)
        self.hybrid_threshold = self.integration_config.get("hybrid_threshold", 0.5)
        
        # Performance tracking
        self.integration_history: list[Dict[str, Any]] = []
        
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="regression integration initialization"
    )
    async def initialize(self) -> bool:
        """Initialize regression predictors for both Analyst and Tactician.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("🚀 Initializing regression integration manager")
            
            # Initialize Analyst regression predictor
            if self.enable_analyst_regression:
                analyst_config = self.config.get("analyst_regression", {
                    "model_type": "LightGBM",
                    "min_profit_threshold": 0.005,  # 0.5%
                    "max_profit_threshold": 0.03,   # 3%
                    "position_sizing_enabled": True
                })
                self.analyst_regression_predictor = RegressionProfitPredictor(analyst_config)
                self.logger.info("✅ Analyst regression predictor initialized")
            
            # Initialize Tactician regression predictor
            if self.enable_tactician_regression:
                tactician_config = self.config.get("tactician_regression", {
                    "model_type": "LightGBM",
                    "min_profit_threshold": 0.003,  # 0.3%
                    "max_profit_threshold": 0.02,   # 2%
                    "position_sizing_enabled": True
                })
                self.tactician_regression_predictor = RegressionProfitPredictor(tactician_config)
                self.logger.info("✅ Tactician regression predictor initialized")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Regression integration initialization failed: {str(e)}")
            return False
    
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="analyst regression prediction"
    )
    async def predict_analyst_profit(
        self, 
        features: pd.DataFrame,
        current_price: float,
        classification_confidence: float
    ) -> Optional[Dict[str, Any]]:
        """Predict profit for Analyst decision making.
        
        Args:
            features: Feature DataFrame
            current_price: Current market price
            classification_confidence: Confidence from existing classification model
            
        Returns:
            Dictionary with hybrid prediction results
        """
        try:
            if not self.analyst_regression_predictor or not self.analyst_regression_predictor.is_trained:
                self.logger.warning("Analyst regression predictor not available or not trained")
                return None
            
            # Get regression prediction
            regression_result = await self.analyst_regression_predictor.predict_profit(
                features, current_price, include_confidence=True
            )
            
            if not regression_result:
                return None
            
            # Combine with classification confidence
            hybrid_result = self._combine_regression_classification(
                regression_result, classification_confidence, "analyst"
            )
            
            # Store integration history
            self.integration_history.append({
                'timestamp': datetime.now(),
                'component': 'analyst',
                'regression_result': regression_result,
                'classification_confidence': classification_confidence,
                'hybrid_result': hybrid_result
            })
            
            return hybrid_result
            
        except Exception as e:
            self.logger.error(f"❌ Analyst profit prediction failed: {str(e)}")
            return None
    
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="tactician regression prediction"
    )
    async def predict_tactician_profit(
        self, 
        features: pd.DataFrame,
        current_price: float,
        classification_confidence: float
    ) -> Optional[Dict[str, Any]]:
        """Predict profit for Tactician decision making.
        
        Args:
            features: Feature DataFrame
            current_price: Current market price
            classification_confidence: Confidence from existing classification model
            
        Returns:
            Dictionary with hybrid prediction results
        """
        try:
            if not self.tactician_regression_predictor or not self.tactician_regression_predictor.is_trained:
                self.logger.warning("Tactician regression predictor not available or not trained")
                return None
            
            # Get regression prediction
            regression_result = await self.tactician_regression_predictor.predict_profit(
                features, current_price, include_confidence=True
            )
            
            if not regression_result:
                return None
            
            # Combine with classification confidence
            hybrid_result = self._combine_regression_classification(
                regression_result, classification_confidence, "tactician"
            )
            
            # Store integration history
            self.integration_history.append({
                'timestamp': datetime.now(),
                'component': 'tactician',
                'regression_result': regression_result,
                'classification_confidence': classification_confidence,
                'hybrid_result': hybrid_result
            })
            
            return hybrid_result
            
        except Exception as e:
            self.logger.error(f"❌ Tactician profit prediction failed: {str(e)}")
            return None
    
    def _combine_regression_classification(
        self, 
        regression_result: Dict[str, Any], 
        classification_confidence: float,
        component: str
    ) -> Dict[str, Any]:
        """Combine regression and classification predictions for hybrid decision making.
        
        Args:
            regression_result: Results from regression predictor
            classification_confidence: Confidence from classification model
            component: Component name ('analyst' or 'tactician')
            
        Returns:
            Dictionary with combined results
        """
        predicted_profit_pct = regression_result['predicted_profit_pct']
        position_sizing = regression_result.get('recommended_position_size', 0.0)
        
        # Calculate hybrid confidence
        regression_confidence = regression_result.get('confidence_metrics', {}).get('prediction_confidence', 0.5)
        hybrid_confidence = (classification_confidence + regression_confidence) / 2
        
        # Determine final decision based on hybrid approach
        if hybrid_confidence > self.hybrid_threshold and predicted_profit_pct > 0:
            final_decision = "enter"
            decision_confidence = hybrid_confidence
        else:
            final_decision = "skip"
            decision_confidence = 1.0 - hybrid_confidence
        
        # Calculate risk-adjusted position size
        risk_adjusted_position = position_sizing * hybrid_confidence
        
        return {
            'component': component,
            'predicted_profit_pct': predicted_profit_pct,
            'predicted_profit_abs': regression_result['predicted_profit_abs'],
            'classification_confidence': classification_confidence,
            'regression_confidence': regression_confidence,
            'hybrid_confidence': hybrid_confidence,
            'final_decision': final_decision,
            'decision_confidence': decision_confidence,
            'position_sizing': {
                'base_position_size': position_sizing,
                'risk_adjusted_position_size': risk_adjusted_position,
                'confidence_level': regression_result.get('confidence_level', 'low')
            },
            'risk_metrics': {
                'profit_threshold_met': predicted_profit_pct > 0,
                'confidence_threshold_met': hybrid_confidence > self.hybrid_threshold,
                'risk_reward_ratio': predicted_profit_pct / (1.0 - hybrid_confidence) if hybrid_confidence < 1.0 else predicted_profit_pct
            },
            'timestamp': datetime.now()
        }
    
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="regression model training"
    )
    async def train_analyst_regression(
        self, 
        features: pd.DataFrame, 
        profit_targets: pd.Series
    ) -> bool:
        """Train the Analyst regression model.
        
        Args:
            features: Feature DataFrame
            profit_targets: Profit target series
            
        Returns:
            bool: True if training successful
        """
        try:
            if not self.analyst_regression_predictor:
                self.logger.error("Analyst regression predictor not initialized")
                return False
            
            self.logger.info("🚀 Training Analyst regression model")
            success = await self.analyst_regression_predictor.train_model(features, profit_targets)
            
            if success:
                self.logger.info("✅ Analyst regression model training completed")
            else:
                self.logger.error("❌ Analyst regression model training failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Analyst regression training failed: {str(e)}")
            return False
    
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="regression model training"
    )
    async def train_tactician_regression(
        self, 
        features: pd.DataFrame, 
        profit_targets: pd.Series
    ) -> bool:
        """Train the Tactician regression model.
        
        Args:
            features: Feature DataFrame
            profit_targets: Profit target series
            
        Returns:
            bool: True if training successful
        """
        try:
            if not self.tactician_regression_predictor:
                self.logger.error("Tactician regression predictor not initialized")
                return False
            
            self.logger.info("🚀 Training Tactician regression model")
            success = await self.tactician_regression_predictor.train_model(features, profit_targets)
            
            if success:
                self.logger.info("✅ Tactician regression model training completed")
            else:
                self.logger.error("❌ Tactician regression model training failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Tactician regression training failed: {str(e)}")
            return False
    
    def get_integration_analytics(self) -> Dict[str, Any]:
        """Get analytics about the integration performance.
        
        Returns:
            Dictionary with integration analytics
        """
        if not self.integration_history:
            return {}
        
        df = pd.DataFrame(self.integration_history)
        
        analytics = {
            'total_predictions': len(df),
            'analyst_predictions': len(df[df['component'] == 'analyst']),
            'tactician_predictions': len(df[df['component'] == 'tactician']),
            'average_hybrid_confidence': df['hybrid_result'].apply(lambda x: x['hybrid_confidence']).mean(),
            'decision_distribution': df['hybrid_result'].apply(lambda x: x['final_decision']).value_counts().to_dict(),
            'profit_predictions': {
                'mean': df['hybrid_result'].apply(lambda x: x['predicted_profit_pct']).mean(),
                'std': df['hybrid_result'].apply(lambda x: x['predicted_profit_pct']).std(),
                'min': df['hybrid_result'].apply(lambda x: x['predicted_profit_pct']).min(),
                'max': df['hybrid_result'].apply(lambda x: x['predicted_profit_pct']).max()
            }
        }
        
        return analytics
    
    def save_models(self, base_path: str) -> bool:
        """Save both regression models.
        
        Args:
            base_path: Base path for saving models
            
        Returns:
            bool: True if save successful
        """
        try:
            success = True
            
            if self.analyst_regression_predictor:
                analyst_path = f"{base_path}/analyst_regression_model.joblib"
                success &= self.analyst_regression_predictor.save_model(analyst_path)
            
            if self.tactician_regression_predictor:
                tactician_path = f"{base_path}/tactician_regression_model.joblib"
                success &= self.tactician_regression_predictor.save_model(tactician_path)
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save models: {str(e)}")
            return False
    
    def load_models(self, base_path: str) -> bool:
        """Load both regression models.
        
        Args:
            base_path: Base path for loading models
            
        Returns:
            bool: True if load successful
        """
        try:
            success = True
            
            if self.analyst_regression_predictor:
                analyst_path = f"{base_path}/analyst_regression_model.joblib"
                success &= self.analyst_regression_predictor.load_model(analyst_path)
            
            if self.tactician_regression_predictor:
                tactician_path = f"{base_path}/tactician_regression_model.joblib"
                success &= self.tactician_regression_predictor.load_model(tactician_path)
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load models: {str(e)}")
            return False