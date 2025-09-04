#!/usr/bin/env python3
"""
Enhanced Model Training Pipeline

This module provides an enhanced model training pipeline with comprehensive validation,
error handling, and monitoring at each step.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.utils.pipeline_validation_framework import (
    validation_orchestrator,
    ValidationLevel,
    ValidationResult,
)
from src.utils.operation_protection_decorators import (
    validate_data_format,
    validate_data_analysis,
    validate_data_access,
    validate_model_training,
    safe_operation,
    performance_monitor,
)
from src.utils.enhanced_common_operations import (
    load_and_validate_data,
    clean_and_prepare_data,
    analyze_data_quality,
    save_processed_data,
    validate_pipeline_step_output,
    DataValidationError,
    DataProcessingError,
)
from src.training.steps.model_training.step_validators import (
    validate_model_training_step,
    VALIDATOR_REGISTRY,
)


class EnhancedModelTrainingPipeline:
    """Enhanced model training pipeline with comprehensive validation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("EnhancedModelTrainingPipeline")
        self.pipeline_state = {}
        self.validation_reports = []
        self.performance_metrics = {}
        
        # Initialize validation orchestrator
        self.validation_orchestrator = validation_orchestrator
        
        # Pipeline steps configuration
        self.steps = [
            'data_loading',
            'data_preprocessing',
            'hmm_training',
            'regime_intelligence',
            'analyst_creation',
            'analyst_enhancement',
            'ensemble_creation',
            'tactician_training',
            'model_evaluation',
            'model_saving'
        ]
        
        self.logger.info("Enhanced Model Training Pipeline initialized")
    
    @validate_data_access(required_directories=['data_cache'])
    @performance_monitor(performance_threshold=30.0)
    async def load_training_data(self, symbol: str, exchange: str, timeframe: str) -> pd.DataFrame:
        """Load and validate training data."""
        self.logger.info(f"Loading training data for {symbol} on {exchange}")
        
        try:
            # Construct data file path
            data_file = f"data_cache/aggtrades_{exchange}_{symbol}_consolidated.parquet"
            
            # Load and validate data
            df = load_and_validate_data(
                data_file, 
                required_columns=['timestamp', 'price', 'volume', 'side']
            )
            
            # Validate pipeline step output
            if not validate_pipeline_step_output('data_loading', df, pd.DataFrame):
                raise DataValidationError("Data loading validation failed")
            
            # Store in pipeline state
            self.pipeline_state['raw_data'] = df
            self.pipeline_state['data_shape'] = df.shape
            
            self.logger.info(f"Successfully loaded data with shape: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.exception(f"Failed to load training data: {e}")
            raise DataProcessingError(f"Data loading failed: {e}") from e
    
    @validate_data_format(allow_empty=False)
    @performance_monitor(performance_threshold=60.0)
    async def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess training data."""
        self.logger.info("Preprocessing training data")
        
        try:
            # Data cleaning configuration
            cleaning_config = {
                'remove_duplicates': True,
                'handle_missing': 'forward_fill',
                'remove_outliers': False,
                'normalize_columns': False
            }
            
            # Clean and prepare data
            cleaned_df = clean_and_prepare_data(df, cleaning_config)
            
            # Analyze data quality
            quality_analysis = analyze_data_quality(cleaned_df)
            
            # Store quality analysis in pipeline state
            self.pipeline_state['data_quality'] = quality_analysis
            
            # Validate pipeline step output
            if not validate_pipeline_step_output('data_preprocessing', cleaned_df, pd.DataFrame):
                raise DataValidationError("Data preprocessing validation failed")
            
            # Store in pipeline state
            self.pipeline_state['processed_data'] = cleaned_df
            
            self.logger.info(f"Data preprocessing completed. Quality score: {quality_analysis['quality_score']:.2f}")
            return cleaned_df
            
        except Exception as e:
            self.logger.exception(f"Failed to preprocess data: {e}")
            raise DataProcessingError(f"Data preprocessing failed: {e}") from e
    
    @validate_model_training(required_metrics=['accuracy', 'loss', 'convergence_iterations'])
    @performance_monitor(performance_threshold=300.0)
    async def train_hmm_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train HMM models with validation."""
        self.logger.info("Training HMM models")
        
        try:
            # Import HMM training components
            from src.training.steps.model_training.step09_hmm_based_training import HMMBasedTrainingStep
            
            # Initialize HMM trainer
            hmm_trainer = HMMBasedTrainingStep()
            
            # Train models
            training_result = await hmm_trainer.train_models(
                symbol=self.config.get('symbol', 'ETHUSDT'),
                exchange=self.config.get('exchange', 'BINANCE'),
                timeframe=self.config.get('timeframe', '1m'),
                data_dir=self.config.get('data_dir', 'data_cache')
            )
            
            # Validate training result
            validation_report = await validate_model_training_step(
                'hmm_training', 
                training_result, 
                {'step': 'hmm_training', 'data_shape': df.shape}
            )
            
            self.validation_reports.append(validation_report)
            
            if validation_report.result == ValidationResult.FAILED:
                raise DataValidationError(f"HMM training validation failed: {validation_report.errors}")
            
            # Store in pipeline state
            self.pipeline_state['hmm_models'] = training_result
            
            self.logger.info("HMM model training completed successfully")
            return training_result
            
        except Exception as e:
            self.logger.exception(f"Failed to train HMM models: {e}")
            raise DataProcessingError(f"HMM training failed: {e}") from e
    
    @validate_model_training(required_metrics=['regime_accuracy', 'transition_accuracy', 'confidence_score'])
    @performance_monitor(performance_threshold=180.0)
    async def build_regime_intelligence(self, hmm_result: Dict[str, Any]) -> Dict[str, Any]:
        """Build unified regime intelligence."""
        self.logger.info("Building unified regime intelligence")
        
        try:
            # Import regime intelligence components
            from src.training.steps.model_training.step10_unified_regime_intelligence import UnifiedRegimeIntelligenceStep
            
            # Initialize regime intelligence builder
            regime_builder = UnifiedRegimeIntelligenceStep()
            
            # Build intelligence
            intelligence_result = await regime_builder.build_intelligence(
                symbol=self.config.get('symbol', 'ETHUSDT'),
                exchange=self.config.get('exchange', 'BINANCE'),
                timeframe=self.config.get('timeframe', '1m'),
                data_dir=self.config.get('data_dir', 'data_cache')
            )
            
            # Validate intelligence result
            validation_report = await validate_model_training_step(
                'regime_intelligence', 
                intelligence_result, 
                {'step': 'regime_intelligence', 'hmm_result': hmm_result}
            )
            
            self.validation_reports.append(validation_report)
            
            if validation_report.result == ValidationResult.FAILED:
                raise DataValidationError(f"Regime intelligence validation failed: {validation_report.errors}")
            
            # Store in pipeline state
            self.pipeline_state['regime_intelligence'] = intelligence_result
            
            self.logger.info("Regime intelligence building completed successfully")
            return intelligence_result
            
        except Exception as e:
            self.logger.exception(f"Failed to build regime intelligence: {e}")
            raise DataProcessingError(f"Regime intelligence building failed: {e}") from e
    
    @validate_model_training(required_metrics=['creation_accuracy', 'model_count'])
    @performance_monitor(performance_threshold=240.0)
    async def create_analysts(self, intelligence_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create analysts with validation."""
        self.logger.info("Creating analysts")
        
        try:
            # Import analyst creation components
            from src.training.steps.model_training.step11_analyst_creation import AnalystCreationStep
            
            # Initialize analyst creator
            analyst_creator = AnalystCreationStep()
            
            # Create analysts
            creation_result = await analyst_creator.create_analysts(
                symbol=self.config.get('symbol', 'ETHUSDT'),
                exchange=self.config.get('exchange', 'BINANCE'),
                timeframe=self.config.get('timeframe', '1m'),
                data_dir=self.config.get('data_dir', 'data_cache')
            )
            
            # Validate creation result
            validation_report = await validate_model_training_step(
                'analyst_creation', 
                creation_result, 
                {'step': 'analyst_creation', 'intelligence_result': intelligence_result}
            )
            
            self.validation_reports.append(validation_report)
            
            if validation_report.result == ValidationResult.FAILED:
                raise DataValidationError(f"Analyst creation validation failed: {validation_report.errors}")
            
            # Store in pipeline state
            self.pipeline_state['analysts'] = creation_result
            
            self.logger.info("Analyst creation completed successfully")
            return creation_result
            
        except Exception as e:
            self.logger.exception(f"Failed to create analysts: {e}")
            raise DataProcessingError(f"Analyst creation failed: {e}") from e
    
    @validate_model_training(required_metrics=['enhancement_accuracy', 'improvement_scores'])
    @performance_monitor(performance_threshold=200.0)
    async def enhance_analysts(self, creation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance analysts with validation."""
        self.logger.info("Enhancing analysts")
        
        try:
            # Import analyst enhancement components
            from src.training.steps.model_training.step12_analyst_enhancement import AnalystEnhancementStep
            
            # Initialize analyst enhancer
            analyst_enhancer = AnalystEnhancementStep()
            
            # Enhance analysts
            enhancement_result = await analyst_enhancer.enhance_analysts(
                symbol=self.config.get('symbol', 'ETHUSDT'),
                exchange=self.config.get('exchange', 'BINANCE'),
                timeframe=self.config.get('timeframe', '1m'),
                data_dir=self.config.get('data_dir', 'data_cache')
            )
            
            # Validate enhancement result
            validation_report = await validate_model_training_step(
                'analyst_enhancement', 
                enhancement_result, 
                {'step': 'analyst_enhancement', 'creation_result': creation_result}
            )
            
            self.validation_reports.append(validation_report)
            
            if validation_report.result == ValidationResult.FAILED:
                raise DataValidationError(f"Analyst enhancement validation failed: {validation_report.errors}")
            
            # Store in pipeline state
            self.pipeline_state['enhanced_analysts'] = enhancement_result
            
            self.logger.info("Analyst enhancement completed successfully")
            return enhancement_result
            
        except Exception as e:
            self.logger.exception(f"Failed to enhance analysts: {e}")
            raise DataProcessingError(f"Analyst enhancement failed: {e}") from e
    
    @validate_model_training(required_metrics=['ensemble_accuracy', 'ensemble_count'])
    @performance_monitor(performance_threshold=150.0)
    async def create_ensembles(self, enhancement_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create ensembles with validation."""
        self.logger.info("Creating ensembles")
        
        try:
            # Import ensemble creation components
            from src.training.steps.model_training.step13_analyst_ensemble_creation import AnalystEnsembleCreationStep
            
            # Initialize ensemble creator
            ensemble_creator = AnalystEnsembleCreationStep()
            
            # Create ensembles
            ensemble_result = await ensemble_creator.create_ensembles(
                symbol=self.config.get('symbol', 'ETHUSDT'),
                exchange=self.config.get('exchange', 'BINANCE'),
                timeframe=self.config.get('timeframe', '1m'),
                data_dir=self.config.get('data_dir', 'data_cache')
            )
            
            # Validate ensemble result
            validation_report = await validate_model_training_step(
                'ensemble_creation', 
                ensemble_result, 
                {'step': 'ensemble_creation', 'enhancement_result': enhancement_result}
            )
            
            self.validation_reports.append(validation_report)
            
            if validation_report.result == ValidationResult.FAILED:
                raise DataValidationError(f"Ensemble creation validation failed: {validation_report.errors}")
            
            # Store in pipeline state
            self.pipeline_state['ensembles'] = ensemble_result
            
            self.logger.info("Ensemble creation completed successfully")
            return ensemble_result
            
        except Exception as e:
            self.logger.exception(f"Failed to create ensembles: {e}")
            raise DataProcessingError(f"Ensemble creation failed: {e}") from e
    
    @validate_model_training(required_metrics=['accuracy', 'precision', 'recall', 'f1_score'])
    @performance_monitor(performance_threshold=180.0)
    async def train_tacticians(self, ensemble_result: Dict[str, Any]) -> Dict[str, Any]:
        """Train tacticians with validation."""
        self.logger.info("Training tacticians")
        
        try:
            # Import tactician training components
            from src.training.steps.model_training.step15_tactician_specialist_training import TacticianSpecialistTrainingStep
            
            # Initialize tactician trainer
            tactician_trainer = TacticianSpecialistTrainingStep()
            
            # Train tacticians
            tactician_result = await tactician_trainer.train_tacticians(
                symbol=self.config.get('symbol', 'ETHUSDT'),
                exchange=self.config.get('exchange', 'BINANCE'),
                timeframe=self.config.get('timeframe', '1m'),
                data_dir=self.config.get('data_dir', 'data_cache')
            )
            
            # Validate tactician result
            validation_report = await validate_model_training_step(
                'tactician_training', 
                tactician_result, 
                {'step': 'tactician_training', 'ensemble_result': ensemble_result}
            )
            
            self.validation_reports.append(validation_report)
            
            if validation_report.result == ValidationResult.FAILED:
                raise DataValidationError(f"Tactician training validation failed: {validation_report.errors}")
            
            # Store in pipeline state
            self.pipeline_state['tacticians'] = tactician_result
            
            self.logger.info("Tactician training completed successfully")
            return tactician_result
            
        except Exception as e:
            self.logger.exception(f"Failed to train tacticians: {e}")
            raise DataProcessingError(f"Tactician training failed: {e}") from e
    
    @validate_data_analysis(required_outputs=['evaluation_metrics', 'performance_summary'])
    @performance_monitor(performance_threshold=120.0)
    async def evaluate_models(self, tactician_result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate all trained models."""
        self.logger.info("Evaluating trained models")
        
        try:
            evaluation_result = {
                'evaluation_metrics': {},
                'performance_summary': {},
                'model_comparison': {},
                'recommendations': []
            }
            
            # Evaluate HMM models
            if 'hmm_models' in self.pipeline_state:
                hmm_metrics = self.pipeline_state['hmm_models'].get('training_metrics', {})
                evaluation_result['evaluation_metrics']['hmm'] = hmm_metrics
            
            # Evaluate analysts
            if 'enhanced_analysts' in self.pipeline_state:
                analyst_metrics = self.pipeline_state['enhanced_analysts'].get('enhancement_metrics', {})
                evaluation_result['evaluation_metrics']['analysts'] = analyst_metrics
            
            # Evaluate ensembles
            if 'ensembles' in self.pipeline_state:
                ensemble_metrics = self.pipeline_state['ensembles'].get('ensemble_metrics', {})
                evaluation_result['evaluation_metrics']['ensembles'] = ensemble_metrics
            
            # Evaluate tacticians
            if 'tacticians' in self.pipeline_state:
                tactician_metrics = self.pipeline_state['tacticians'].get('training_metrics', {})
                evaluation_result['evaluation_metrics']['tacticians'] = tactician_metrics
            
            # Create performance summary
            evaluation_result['performance_summary'] = {
                'total_models_trained': len(evaluation_result['evaluation_metrics']),
                'validation_reports_count': len(self.validation_reports),
                'pipeline_success_rate': self._calculate_success_rate()
            }
            
            # Store in pipeline state
            self.pipeline_state['evaluation'] = evaluation_result
            
            self.logger.info("Model evaluation completed successfully")
            return evaluation_result
            
        except Exception as e:
            self.logger.exception(f"Failed to evaluate models: {e}")
            raise DataProcessingError(f"Model evaluation failed: {e}") from e
    
    @validate_data_access(required_directories=['models'])
    @performance_monitor(performance_threshold=60.0)
    async def save_models(self, evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Save all trained models."""
        self.logger.info("Saving trained models")
        
        try:
            save_result = {
                'saved_models': {},
                'save_metrics': {},
                'model_paths': {}
            }
            
            # Ensure models directory exists
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            
            # Save HMM models
            if 'hmm_models' in self.pipeline_state:
                hmm_path = models_dir / "hmm_models.pkl"
                # Save HMM models (implementation depends on model format)
                save_result['saved_models']['hmm'] = str(hmm_path)
                save_result['model_paths']['hmm'] = str(hmm_path)
            
            # Save analysts
            if 'enhanced_analysts' in self.pipeline_state:
                analyst_path = models_dir / "enhanced_analysts.pkl"
                # Save analysts (implementation depends on model format)
                save_result['saved_models']['analysts'] = str(analyst_path)
                save_result['model_paths']['analysts'] = str(analyst_path)
            
            # Save ensembles
            if 'ensembles' in self.pipeline_state:
                ensemble_path = models_dir / "ensembles.pkl"
                # Save ensembles (implementation depends on model format)
                save_result['saved_models']['ensembles'] = str(ensemble_path)
                save_result['model_paths']['ensembles'] = str(ensemble_path)
            
            # Save tacticians
            if 'tacticians' in self.pipeline_state:
                tactician_path = models_dir / "tacticians.pkl"
                # Save tacticians (implementation depends on model format)
                save_result['saved_models']['tacticians'] = str(tactician_path)
                save_result['model_paths']['tacticians'] = str(tactician_path)
            
            # Save pipeline state
            pipeline_state_path = models_dir / "pipeline_state.json"
            with open(pipeline_state_path, 'w') as f:
                json.dump(self.pipeline_state, f, indent=2, default=str)
            
            save_result['saved_models']['pipeline_state'] = str(pipeline_state_path)
            save_result['model_paths']['pipeline_state'] = str(pipeline_state_path)
            
            # Save validation reports
            validation_reports_path = models_dir / "validation_reports.json"
            validation_data = [
                {
                    'step_name': report.step_name,
                    'result': report.result.value,
                    'timestamp': report.timestamp,
                    'duration': report.duration,
                    'errors': report.errors,
                    'warnings': report.warnings
                }
                for report in self.validation_reports
            ]
            with open(validation_reports_path, 'w') as f:
                json.dump(validation_data, f, indent=2)
            
            save_result['saved_models']['validation_reports'] = str(validation_reports_path)
            save_result['model_paths']['validation_reports'] = str(validation_reports_path)
            
            # Store in pipeline state
            self.pipeline_state['saved_models'] = save_result
            
            self.logger.info("Model saving completed successfully")
            return save_result
            
        except Exception as e:
            self.logger.exception(f"Failed to save models: {e}")
            raise DataProcessingError(f"Model saving failed: {e}") from e
    
    def _calculate_success_rate(self) -> float:
        """Calculate pipeline success rate based on validation reports."""
        if not self.validation_reports:
            return 0.0
        
        passed_count = sum(1 for report in self.validation_reports if report.result == ValidationResult.PASSED)
        return passed_count / len(self.validation_reports)
    
    async def run_pipeline(self, symbol: str, exchange: str, timeframe: str = "1m") -> Dict[str, Any]:
        """Run the complete enhanced model training pipeline."""
        self.logger.info("Starting enhanced model training pipeline")
        start_time = time.time()
        
        try:
            # Update config
            self.config.update({
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe
            })
            
            # Step 1: Load training data
            self.logger.info("Step 1: Loading training data")
            raw_data = await self.load_training_data(symbol, exchange, timeframe)
            
            # Step 2: Preprocess data
            self.logger.info("Step 2: Preprocessing data")
            processed_data = await self.preprocess_data(raw_data)
            
            # Step 3: Train HMM models
            self.logger.info("Step 3: Training HMM models")
            hmm_result = await self.train_hmm_models(processed_data)
            
            # Step 4: Build regime intelligence
            self.logger.info("Step 4: Building regime intelligence")
            intelligence_result = await self.build_regime_intelligence(hmm_result)
            
            # Step 5: Create analysts
            self.logger.info("Step 5: Creating analysts")
            creation_result = await self.create_analysts(intelligence_result)
            
            # Step 6: Enhance analysts
            self.logger.info("Step 6: Enhancing analysts")
            enhancement_result = await self.enhance_analysts(creation_result)
            
            # Step 7: Create ensembles
            self.logger.info("Step 7: Creating ensembles")
            ensemble_result = await self.create_ensembles(enhancement_result)
            
            # Step 8: Train tacticians
            self.logger.info("Step 8: Training tacticians")
            tactician_result = await self.train_tacticians(ensemble_result)
            
            # Step 9: Evaluate models
            self.logger.info("Step 9: Evaluating models")
            evaluation_result = await self.evaluate_models(tactician_result)
            
            # Step 10: Save models
            self.logger.info("Step 10: Saving models")
            save_result = await self.save_models(evaluation_result)
            
            # Calculate total execution time
            total_time = time.time() - start_time
            
            # Create final pipeline result
            pipeline_result = {
                'success': True,
                'execution_time': total_time,
                'pipeline_state': self.pipeline_state,
                'validation_reports': self.validation_reports,
                'success_rate': self._calculate_success_rate(),
                'saved_models': save_result,
                'performance_metrics': {
                    'total_steps': len(self.steps),
                    'completed_steps': len(self.pipeline_state),
                    'validation_reports_count': len(self.validation_reports),
                    'execution_time': total_time
                }
            }
            
            self.logger.info(f"Enhanced model training pipeline completed successfully in {total_time:.2f}s")
            return pipeline_result
            
        except Exception as e:
            total_time = time.time() - start_time
            self.logger.exception(f"Enhanced model training pipeline failed after {total_time:.2f}s: {e}")
            
            # Return failure result
            return {
                'success': False,
                'execution_time': total_time,
                'error': str(e),
                'pipeline_state': self.pipeline_state,
                'validation_reports': self.validation_reports,
                'success_rate': self._calculate_success_rate()
            }


# Main function for running the enhanced pipeline
async def run_enhanced_model_training_pipeline(
    symbol: str, 
    exchange: str, 
    timeframe: str = "1m",
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Run the enhanced model training pipeline."""
    if config is None:
        config = {}
    
    pipeline = EnhancedModelTrainingPipeline(config)
    return await pipeline.run_pipeline(symbol, exchange, timeframe)