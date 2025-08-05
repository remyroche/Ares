# src/training/steps/step7_analyst_ensemble_creation.py

import asyncio
import json
import os
import pandas as pd
import pickle
import numpy as np
from typing import Any, Dict, Optional, List
from datetime import datetime

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger


class AnalystEnsembleCreationStep:
    """Step 7: Analyst Ensemble Creation using StackingCV."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger
        
    async def initialize(self) -> None:
        """Initialize the analyst ensemble creation step."""
        try:
            self.logger.info("Initializing Analyst Ensemble Creation Step...")
            self.logger.info("Analyst Ensemble Creation Step initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing Analyst Ensemble Creation Step: {e}")
            raise
    
    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute analyst ensemble creation.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Dict containing ensemble creation results
        """
        try:
            self.logger.info("🔄 Executing Analyst Ensemble Creation...")
            
            # Extract parameters
            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            data_dir = training_input.get("data_dir", "data/training")
            
            # Load enhanced analyst models
            enhanced_models_dir = f"{data_dir}/enhanced_analyst_models"
            enhanced_models = {}
            
            # Load all regime model directories
            for regime_dir in os.listdir(enhanced_models_dir):
                regime_path = os.path.join(enhanced_models_dir, regime_dir)
                if os.path.isdir(regime_path):
                    regime_models = {}
                    for model_file in os.listdir(regime_path):
                        if model_file.endswith('.pkl'):
                            model_name = model_file.replace('.pkl', '')
                            model_path = os.path.join(regime_path, model_file)
                            
                            with open(model_path, 'rb') as f:
                                regime_models[model_name] = pickle.load(f)
                    
                    enhanced_models[regime_dir] = regime_models
            
            if not enhanced_models:
                raise ValueError(f"No enhanced analyst models found in {enhanced_models_dir}")
            
            # Create ensembles for each regime
            ensemble_results = {}
            
            for regime_name, regime_models in enhanced_models.items():
                self.logger.info(f"Creating ensemble for regime: {regime_name}")
                
                # Create ensemble for this regime
                regime_ensemble = await self._create_regime_ensemble(regime_models, regime_name)
                ensemble_results[regime_name] = regime_ensemble
            
            # Save ensemble models
            ensemble_models_dir = f"{data_dir}/analyst_ensembles"
            os.makedirs(ensemble_models_dir, exist_ok=True)
            
            for regime_name, ensemble_data in ensemble_results.items():
                ensemble_file = f"{ensemble_models_dir}/{regime_name}_ensemble.pkl"
                with open(ensemble_file, 'wb') as f:
                    pickle.dump(ensemble_data, f)
            
            # Save ensemble summary
            summary_file = f"{data_dir}/{exchange}_{symbol}_analyst_ensemble_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(ensemble_results, f, indent=2)
            
            self.logger.info(f"✅ Analyst ensemble creation completed. Results saved to {ensemble_models_dir}")
            
            # Update pipeline state
            pipeline_state["analyst_ensembles"] = ensemble_results
            
            return {
                "analyst_ensembles": ensemble_results,
                "ensemble_models_dir": ensemble_models_dir,
                "duration": 0.0,  # Will be calculated in actual implementation
                "status": "SUCCESS"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error in Analyst Ensemble Creation: {e}")
            return {
                "status": "FAILED",
                "error": str(e),
                "duration": 0.0
            }
    
    async def _create_regime_ensemble(self, models: Dict[str, Any], regime_name: str) -> Dict[str, Any]:
        """
        Create ensemble for a specific regime.
        
        Args:
            models: Enhanced models for the regime
            regime_name: Name of the regime
            
        Returns:
            Dict containing ensemble model
        """
        try:
            self.logger.info(f"Creating ensemble for regime: {regime_name}")
            
            # Extract models and their accuracies
            model_list = []
            model_names = []
            accuracies = []
            
            for model_name, model_data in models.items():
                model_list.append(model_data["model"])
                model_names.append(model_name)
                accuracies.append(model_data.get("accuracy", 0))
            
            # Create different ensemble types
            ensemble_results = {}
            
            # 1. StackingCV Ensemble
            stacking_ensemble = await self._create_stacking_ensemble(model_list, model_names, regime_name)
            ensemble_results["stacking_cv"] = stacking_ensemble
            
            # 2. Voting Ensemble
            voting_ensemble = await self._create_voting_ensemble(model_list, model_names, regime_name)
            ensemble_results["voting"] = voting_ensemble
            
            # 3. Weighted Average Ensemble
            weighted_ensemble = await self._create_weighted_ensemble(model_list, accuracies, model_names, regime_name)
            ensemble_results["weighted_average"] = weighted_ensemble
            
            # 4. Bagging Ensemble
            bagging_ensemble = await self._create_bagging_ensemble(model_list, model_names, regime_name)
            ensemble_results["bagging"] = bagging_ensemble
            
            return ensemble_results
            
        except Exception as e:
            self.logger.error(f"Error creating ensemble for regime {regime_name}: {e}")
            raise
    
    async def _create_stacking_ensemble(self, models: List[Any], model_names: List[str], regime_name: str) -> Dict[str, Any]:
        """Create StackingCV ensemble."""
        try:
            from sklearn.ensemble import StackingClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score
            
            # Create stacking classifier
            estimators = [(name, model) for name, model in zip(model_names, models)]
            stacking_classifier = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(),
                cv=5,
                stack_method='predict_proba'
            )
            
            # Simulate training (in real implementation, you'd have validation data)
            ensemble_data = {
                "ensemble": stacking_classifier,
                "ensemble_type": "StackingCV",
                "base_models": model_names,
                "final_estimator": "LogisticRegression",
                "cv_folds": 5,
                "regime": regime_name,
                "creation_date": datetime.now().isoformat()
            }
            
            return ensemble_data
            
        except Exception as e:
            self.logger.error(f"Error creating stacking ensemble for {regime_name}: {e}")
            raise
    
    async def _create_voting_ensemble(self, models: List[Any], model_names: List[str], regime_name: str) -> Dict[str, Any]:
        """Create voting ensemble."""
        try:
            from sklearn.ensemble import VotingClassifier
            
            # Create voting classifier
            estimators = [(name, model) for name, model in zip(model_names, models)]
            voting_classifier = VotingClassifier(
                estimators=estimators,
                voting='soft'
            )
            
            ensemble_data = {
                "ensemble": voting_classifier,
                "ensemble_type": "Voting",
                "base_models": model_names,
                "voting_method": "soft",
                "regime": regime_name,
                "creation_date": datetime.now().isoformat()
            }
            
            return ensemble_data
            
        except Exception as e:
            self.logger.error(f"Error creating voting ensemble for {regime_name}: {e}")
            raise
    
    async def _create_weighted_ensemble(self, models: List[Any], accuracies: List[float], 
                                      model_names: List[str], regime_name: str) -> Dict[str, Any]:
        """Create weighted average ensemble."""
        try:
            # Calculate weights based on accuracies
            total_accuracy = sum(accuracies)
            weights = [acc / total_accuracy for acc in accuracies]
            
            ensemble_data = {
                "ensemble_type": "WeightedAverage",
                "base_models": model_names,
                "weights": weights,
                "accuracies": accuracies,
                "regime": regime_name,
                "creation_date": datetime.now().isoformat()
            }
            
            return ensemble_data
            
        except Exception as e:
            self.logger.error(f"Error creating weighted ensemble for {regime_name}: {e}")
            raise
    
    async def _create_bagging_ensemble(self, models: List[Any], model_names: List[str], regime_name: str) -> Dict[str, Any]:
        """Create bagging ensemble."""
        try:
            from sklearn.ensemble import BaggingClassifier
            from sklearn.tree import DecisionTreeClassifier
            
            # Use the best performing model as base estimator
            best_model = models[0]  # In real implementation, select based on performance
            
            bagging_classifier = BaggingClassifier(
                base_estimator=best_model,
                n_estimators=10,
                max_samples=0.8,
                random_state=42
            )
            
            ensemble_data = {
                "ensemble": bagging_classifier,
                "ensemble_type": "Bagging",
                "base_estimator": model_names[0],
                "n_estimators": 10,
                "max_samples": 0.8,
                "regime": regime_name,
                "creation_date": datetime.now().isoformat()
            }
            
            return ensemble_data
            
        except Exception as e:
            self.logger.error(f"Error creating bagging ensemble for {regime_name}: {e}")
            raise


# For backward compatibility with existing step structure
async def run_step(
    symbol: str,
    exchange: str = "BINANCE",
    data_dir: str = "data/training",
    **kwargs
) -> bool:
    """
    Run the analyst ensemble creation step.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory path
        **kwargs: Additional parameters
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create step instance
        config = {"symbol": symbol, "exchange": exchange, "data_dir": data_dir}
        step = AnalystEnsembleCreationStep(config)
        await step.initialize()
        
        # Execute step
        training_input = {
            "symbol": symbol,
            "exchange": exchange,
            "data_dir": data_dir,
            **kwargs
        }
        
        pipeline_state = {}
        result = await step.execute(training_input, pipeline_state)
        
        return result.get("status") == "SUCCESS"
        
    except Exception as e:
        print(f"❌ Analyst ensemble creation failed: {e}")
        return False


if __name__ == "__main__":
    # Test the step
    async def test():
        result = await run_step("ETHUSDT", "BINANCE", "data/training")
        print(f"Test result: {result}")
    
    asyncio.run(test())
