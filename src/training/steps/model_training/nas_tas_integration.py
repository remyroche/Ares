"""
NAS-TAS Integration for Training Pipeline

This module integrates NAS and TAS components into the existing training pipeline,
providing enhanced regime detection and model training capabilities.

Key Features:
- NAS integration for Analyst training (5m timeframe)
- TAS integration for Tactician training (1m timeframe)
- CatBoost and XGBoost removal
- Enhanced feature engineering
- Pipeline orchestration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time
from dataclasses import dataclass
from pathlib import Path
import pickle

# Import NAS-TAS training components
from src.training.steps.model_training.nas_enhanced_analyst_training import (
    NASEnhancedAnalystTrainingStep, NASEnhancedAnalystTrainingConfig
)
from src.training.steps.model_training.tas_enhanced_tactician_training import (
    TASEnhancedTacticianTrainingStep, TASEnhancedTacticianTrainingConfig
)

# Import existing training components
from src.training.steps.model_training.analyst_models_training_refactored import AnalystModelsTrainingStep
from src.training.steps.model_training.tactician_models_training_refactored import TacticianModelsTrainingStep
from src.training.steps.model_training.tactician_dual_training_step import TacticianDualTrainingStep

# Import NAS and TAS components
from src.training.steps.market_analysis.nas_regime.core.perfect_nas_config import (
    PerfectNASConfig, NeuralArchitectureType
)
from src.training.steps.market_analysis.tas_regime.core.enhanced_tas_engine import (
    TASConfig, TreeSearchStrategy
)

from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_error

logger = logging.getLogger(__name__)

@dataclass
class NASTASIntegrationConfig:
    """Configuration for NAS-TAS Integration."""
    # NAS Configuration
    nas_config: PerfectNASConfig
    enable_nas_analyst: bool = True
    
    # TAS Configuration
    tas_config: TASConfig
    enable_tas_tactician: bool = True
    enable_tas_analyst: bool = True
    
    # Model Configuration
    remove_catboost: bool = True
    remove_xgboost: bool = True
    
    # Training Configuration
    analyst_timeframe: str = "5m"
    tactician_timeframe: str = "1m"
    enable_per_regime_training: bool = True
    enable_single_model_training: bool = True
    
    # Integration Configuration
    enable_feature_enhancement: bool = True
    enable_architecture_adaptation: bool = True
    enable_performance_monitoring: bool = True

class NASTASIntegration:
    """
    NAS-TAS Integration for Training Pipeline.
    
    This class orchestrates the integration of NAS and TAS components into the
    existing training pipeline, providing enhanced regime detection and model training.
    """
    
    def __init__(self, config: NASTASIntegrationConfig):
        """Initialize NAS-TAS Integration."""
        self.config = config
        self.logger = system_logger.getChild("NASTASIntegration")
        
        # Initialize NAS-TAS training steps
        self.nas_analyst_training = None
        self.tas_tactician_training = None
        self.tas_analyst_training = None
        
        # Initialize base training steps
        self.base_analyst_training = AnalystModelsTrainingStep()
        self.base_tactician_training = TacticianModelsTrainingStep()
        self.dual_tactician_training = TacticianDualTrainingStep()
        
        # Integration state
        self.integration_results = {}
        self.performance_metrics = {}
        self.adaptation_history = []
        
        self.logger.info("✅ NAS-TAS Integration initialized")
        self.logger.info(f"   NAS Analyst enabled: {config.enable_nas_analyst}")
        self.logger.info(f"   TAS Tactician enabled: {config.enable_tas_tactician}")
        self.logger.info(f"   TAS Analyst enabled: {config.enable_tas_analyst}")
        self.logger.info(f"   CatBoost removed: {config.remove_catboost}")
        self.logger.info(f"   XGBoost removed: {config.remove_xgboost}")
    
    async def execute_integrated_training(self, 
                                        training_input: Dict[str, Any], 
                                        pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute integrated NAS-TAS training.
        
        Args:
            training_input: Training input data
            pipeline_state: Current pipeline state
            
        Returns:
            Integrated training results
        """
        start_time = time.time()
        self.logger.info("🚀 Starting integrated NAS-TAS training...")
        
        try:
            # Step 1: Initialize NAS-TAS training steps
            await self._initialize_nas_tas_steps()
            
            # Step 2: Execute NAS-Enhanced Analyst training
            analyst_results = await self._execute_nas_analyst_training(
                training_input, pipeline_state
            )
            
            # Step 3: Execute TAS-Enhanced Tactician training
            tactician_results = await self._execute_tas_tactician_training(
                training_input, pipeline_state, analyst_results
            )
            
            # Step 4: Execute TAS-Enhanced Analyst training (5m timeframe)
            tas_analyst_results = await self._execute_tas_analyst_training(
                training_input, pipeline_state
            )
            
            # Step 5: Integrate results and generate summary
            integration_summary = await self._generate_integration_summary(
                analyst_results, tactician_results, tas_analyst_results
            )
            
            execution_time = time.time() - start_time
            
            # Compile results
            results = {
                'success': True,
                'execution_time': execution_time,
                'step_name': 'nas_tas_integration',
                'analyst_results': analyst_results,
                'tactician_results': tactician_results,
                'tas_analyst_results': tas_analyst_results,
                'integration_summary': integration_summary,
                'metadata': {
                    'nas_analyst_enabled': self.config.enable_nas_analyst,
                    'tas_tactician_enabled': self.config.enable_tas_tactician,
                    'tas_analyst_enabled': self.config.enable_tas_analyst,
                    'catboost_removed': self.config.remove_catboost,
                    'xgboost_removed': self.config.remove_xgboost,
                    'analyst_timeframe': self.config.analyst_timeframe,
                    'tactician_timeframe': self.config.tactician_timeframe
                }
            }
            
            self.logger.info(f"✅ Integrated NAS-TAS training completed in {execution_time:.2f}s")
            self._log_integration_summary(results)
            
            return results
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Integrated NAS-TAS training failed: {e}")
            
            return {
                'success': False,
                'execution_time': execution_time,
                'error': str(e),
                'step_name': 'nas_tas_integration',
                'metadata': {'error': str(e)}
            }
    
    async def _initialize_nas_tas_steps(self) -> None:
        """Initialize NAS-TAS training steps."""
        self.logger.info("🔧 Initializing NAS-TAS training steps...")
        
        try:
            # Initialize NAS-Enhanced Analyst training
            if self.config.enable_nas_analyst:
                nas_analyst_config = NASEnhancedAnalystTrainingConfig(
                    nas_config=self.config.nas_config,
                    tas_config=self.config.tas_config,
                    enable_nas_architecture_search=True,
                    enable_tas_5m=True,
                    remove_catboost=self.config.remove_catboost
                )
                self.nas_analyst_training = NASEnhancedAnalystTrainingStep(nas_analyst_config)
                self.logger.info("✅ NAS-Enhanced Analyst training step initialized")
            
            # Initialize TAS-Enhanced Tactician training
            if self.config.enable_tas_tactician:
                tas_tactician_config = TASEnhancedTacticianTrainingConfig(
                    tas_config=self.config.tas_config,
                    enable_tas_architecture_search=True,
                    remove_xgboost=self.config.remove_xgboost,
                    enable_tree_ensemble=True,
                    enable_boosting=True,
                    enable_bagging=True
                )
                self.tas_tactician_training = TASEnhancedTacticianTrainingStep(tas_tactician_config)
                self.logger.info("✅ TAS-Enhanced Tactician training step initialized")
            
            # Initialize TAS-Enhanced Analyst training (5m timeframe)
            if self.config.enable_tas_analyst:
                tas_analyst_config = NASEnhancedAnalystTrainingConfig(
                    nas_config=self.config.nas_config,
                    tas_config=self.config.tas_config,
                    enable_nas_architecture_search=False,  # Disable NAS for TAS-only
                    enable_tas_5m=True,
                    remove_catboost=self.config.remove_catboost
                )
                self.tas_analyst_training = NASEnhancedAnalystTrainingStep(tas_analyst_config)
                self.logger.info("✅ TAS-Enhanced Analyst training step initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize NAS-TAS training steps: {e}")
            raise
    
    async def _execute_nas_analyst_training(self, 
                                          training_input: Dict[str, Any], 
                                          pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute NAS-Enhanced Analyst training."""
        if not self.nas_analyst_training:
            return {
                'success': False,
                'error': 'NAS-Enhanced Analyst training not enabled',
                'step_name': 'nas_analyst_training'
            }
        
        self.logger.info("🔍 Executing NAS-Enhanced Analyst training...")
        
        try:
            # Prepare training input for Analyst
            analyst_input = {
                'X_5m': training_input.get('X_5m'),
                'y_5m': training_input.get('y_5m'),
                'regime_labels': training_input.get('regime_labels'),
                'market_data': training_input.get('market_data')
            }
            
            # Execute NAS-Enhanced Analyst training
            analyst_results = await self.nas_analyst_training.execute_training_step(
                analyst_input, pipeline_state
            )
            
            if analyst_results.get('success', False):
                self.logger.info("✅ NAS-Enhanced Analyst training completed successfully")
            else:
                self.logger.warning("⚠️ NAS-Enhanced Analyst training failed")
            
            return analyst_results
            
        except Exception as e:
            self.logger.error(f"❌ NAS-Enhanced Analyst training failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'step_name': 'nas_analyst_training'
            }
    
    async def _execute_tas_tactician_training(self, 
                                           training_input: Dict[str, Any], 
                                           pipeline_state: Dict[str, Any],
                                           analyst_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute TAS-Enhanced Tactician training."""
        if not self.tas_tactician_training:
            return {
                'success': False,
                'error': 'TAS-Enhanced Tactician training not enabled',
                'step_name': 'tas_tactician_training'
            }
        
        self.logger.info("🔍 Executing TAS-Enhanced Tactician training...")
        
        try:
            # Prepare training input for Tactician
            tactician_input = {
                'X_1m': training_input.get('X_1m'),
                'y_1m': training_input.get('y_1m'),
                'analyst_signals': training_input.get('analyst_signals'),
                'analyst_outputs': analyst_results.get('analyst_results', {}),
                'market_data': training_input.get('market_data')
            }
            
            # Execute TAS-Enhanced Tactician training
            tactician_results = await self.tas_tactician_training.execute_training_step(
                tactician_input, pipeline_state
            )
            
            if tactician_results.get('success', False):
                self.logger.info("✅ TAS-Enhanced Tactician training completed successfully")
            else:
                self.logger.warning("⚠️ TAS-Enhanced Tactician training failed")
            
            return tactician_results
            
        except Exception as e:
            self.logger.error(f"❌ TAS-Enhanced Tactician training failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'step_name': 'tas_tactician_training'
            }
    
    async def _execute_tas_analyst_training(self, 
                                          training_input: Dict[str, Any], 
                                          pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute TAS-Enhanced Analyst training (5m timeframe)."""
        if not self.tas_analyst_training:
            return {
                'success': False,
                'error': 'TAS-Enhanced Analyst training not enabled',
                'step_name': 'tas_analyst_training'
            }
        
        self.logger.info("🔍 Executing TAS-Enhanced Analyst training (5m timeframe)...")
        
        try:
            # Prepare training input for TAS Analyst
            tas_analyst_input = {
                'X_5m': training_input.get('X_5m'),
                'y_5m': training_input.get('y_5m'),
                'regime_labels': training_input.get('regime_labels'),
                'market_data': training_input.get('market_data')
            }
            
            # Execute TAS-Enhanced Analyst training
            tas_analyst_results = await self.tas_analyst_training.execute_training_step(
                tas_analyst_input, pipeline_state
            )
            
            if tas_analyst_results.get('success', False):
                self.logger.info("✅ TAS-Enhanced Analyst training completed successfully")
            else:
                self.logger.warning("⚠️ TAS-Enhanced Analyst training failed")
            
            return tas_analyst_results
            
        except Exception as e:
            self.logger.error(f"❌ TAS-Enhanced Analyst training failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'step_name': 'tas_analyst_training'
            }
    
    async def _generate_integration_summary(self, 
                                          analyst_results: Dict[str, Any],
                                          tactician_results: Dict[str, Any],
                                          tas_analyst_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate integration summary."""
        self.logger.info("📊 Generating integration summary...")
        
        try:
            # Calculate overall success rate
            success_count = 0
            total_count = 0
            
            if analyst_results.get('success', False):
                success_count += 1
            total_count += 1
            
            if tactician_results.get('success', False):
                success_count += 1
            total_count += 1
            
            if tas_analyst_results.get('success', False):
                success_count += 1
            total_count += 1
            
            overall_success_rate = success_count / total_count if total_count > 0 else 0.0
            
            # Calculate execution times
            analyst_time = analyst_results.get('execution_time', 0)
            tactician_time = tactician_results.get('execution_time', 0)
            tas_analyst_time = tas_analyst_results.get('execution_time', 0)
            total_time = analyst_time + tactician_time + tas_analyst_time
            
            # Generate summary
            summary = {
                'overall_success_rate': overall_success_rate,
                'total_execution_time': total_time,
                'analyst_execution_time': analyst_time,
                'tactician_execution_time': tactician_time,
                'tas_analyst_execution_time': tas_analyst_time,
                'analyst_success': analyst_results.get('success', False),
                'tactician_success': tactician_results.get('success', False),
                'tas_analyst_success': tas_analyst_results.get('success', False),
                'catboost_removed': self.config.remove_catboost,
                'xgboost_removed': self.config.remove_xgboost,
                'nas_analyst_enabled': self.config.enable_nas_analyst,
                'tas_tactician_enabled': self.config.enable_tas_tactician,
                'tas_analyst_enabled': self.config.enable_tas_analyst
            }
            
            self.logger.info(f"✅ Integration summary generated")
            self.logger.info(f"   Overall success rate: {overall_success_rate:.3f}")
            self.logger.info(f"   Total execution time: {total_time:.2f}s")
            self.logger.info(f"   Analyst success: {analyst_results.get('success', False)}")
            self.logger.info(f"   Tactician success: {tactician_results.get('success', False)}")
            self.logger.info(f"   TAS Analyst success: {tas_analyst_results.get('success', False)}")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate integration summary: {e}")
            return {
                'overall_success_rate': 0.0,
                'total_execution_time': 0.0,
                'error': str(e)
            }
    
    def _log_integration_summary(self, results: Dict[str, Any]):
        """Log integration summary."""
        try:
            metadata = results.get('metadata', {})
            self.logger.info("📊 NAS-TAS Integration Summary:")
            self.logger.info(f"   Success: {results.get('success', False)}")
            self.logger.info(f"   Execution time: {results.get('execution_time', 0):.2f}s")
            self.logger.info(f"   NAS Analyst enabled: {metadata.get('nas_analyst_enabled', False)}")
            self.logger.info(f"   TAS Tactician enabled: {metadata.get('tas_tactician_enabled', False)}")
            self.logger.info(f"   TAS Analyst enabled: {metadata.get('tas_analyst_enabled', False)}")
            self.logger.info(f"   CatBoost removed: {metadata.get('catboost_removed', False)}")
            self.logger.info(f"   XGBoost removed: {metadata.get('xgboost_removed', False)}")
            self.logger.info(f"   Analyst timeframe: {metadata.get('analyst_timeframe', 'unknown')}")
            self.logger.info(f"   Tactician timeframe: {metadata.get('tactician_timeframe', 'unknown')}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to log integration summary: {e}")
    
    def save_integration_results(self, filepath: str) -> bool:
        """Save integration results."""
        try:
            integration_data = {
                'integration_results': self.integration_results,
                'performance_metrics': self.performance_metrics,
                'adaptation_history': self.adaptation_history,
                'config': self.config
            }
            
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(integration_data, f)
            
            self.logger.info(f"✅ Integration results saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save integration results: {e}")
            return False
    
    def load_integration_results(self, filepath: str) -> bool:
        """Load integration results."""
        try:
            with open(filepath, 'rb') as f:
                integration_data = pickle.load(f)
            
            self.integration_results = integration_data.get('integration_results', {})
            self.performance_metrics = integration_data.get('performance_metrics', {})
            self.adaptation_history = integration_data.get('adaptation_history', [])
            
            self.logger.info(f"✅ Integration results loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load integration results: {e}")
            return False


# Factory function for creating NAS-TAS Integration
def create_nas_tas_integration(config: Optional[NASTASIntegrationConfig] = None) -> NASTASIntegration:
    """Create NAS-TAS Integration instance."""
    if config is None:
        # Default configuration
        nas_config = PerfectNASConfig(
            primary_architecture=NeuralArchitectureType.HYBRID,
            n_regimes=8,
            primary_timeframe="5m",
            enable_neural_odes=True,
            enable_vision_transformers=True,
            enable_state_space_models=True,
            enable_micro_regime_detection=True,
            population_size=30,
            generations=50
        )
        
        tas_config = TASConfig(
            search_strategy=TreeSearchStrategy.ENHANCED_BAYESIAN,
            population_size=25,
            max_generations=40,
            max_evaluations=150,
            enable_multi_objective=True,
            objective_weights={
                'performance': 1.0,
                'complexity': 0.3,
                'efficiency': 0.4,
                'interpretability': 0.5
            },
            max_trees=30,
            max_tree_depth=12,
            allow_boosting=True,
            allow_bagging=True,
            allow_ensemble_methods=True
        )
        
        config = NASTASIntegrationConfig(
            nas_config=nas_config,
            tas_config=tas_config,
            enable_nas_analyst=True,
            enable_tas_tactician=True,
            enable_tas_analyst=True,
            remove_catboost=True,
            remove_xgboost=True
        )
    
    return NASTASIntegration(config)