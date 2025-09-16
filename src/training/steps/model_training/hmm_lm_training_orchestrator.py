"""
HMM LM Models Training Orchestrator

This module orchestrates the complete HMM LM models training pathway:
1. HMM Base Models Training (market_analysis)
2. HMM Ensemble Models Training (market_analysis)
3. Analyst Ensemble Training (with HMM integration)
4. Tactician Ensemble Training (with HMM integration)

Provides a unified interface for the complete HMM → Analyst → Tactician training pipeline.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from src.utils.logger import system_logger
from src.utils.tprint import tprint

# Import HMM training components
from ..market_analysis.hmm_models_training import (
    HMMModelsTrainingEnhanced,
    create_enhanced_hmm_models_training,
    execute_enhanced_hmm_models_training,
    HMMEnsembleTrainingComponent,
    create_hmm_ensemble_training_component,
    execute_hmm_ensemble_training
)

# Import Analyst and Tactician training components
from .analyst_ensemble_training import (
    AnalystEnsembleTrainingStep,
    create_analyst_ensemble_training_step,
    execute_analyst_ensemble_training
)

from .tactician_ensemble_training import (
    TacticianEnsembleTrainingStep,
    create_tactician_ensemble_training_step,
    execute_tactician_ensemble_training
)

logger = system_logger.getChild('HMMLMTrainingOrchestrator')


class TrainingPhase(Enum):
    """Training phases for progress tracking."""
    HMM_BASE_TRAINING = "hmm_base_training"
    HMM_ENSEMBLE_TRAINING = "hmm_ensemble_training"
    ANALYST_ENSEMBLE_TRAINING = "analyst_ensemble_training"
    TACTICIAN_ENSEMBLE_TRAINING = "tactician_ensemble_training"


@dataclass
class TrainingPhaseResult:
    """Result of a training phase."""
    phase: TrainingPhase
    success: bool
    start_time: float
    end_time: float
    duration: float
    artifacts: Dict[str, Any]
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    
    @property
    def is_successful(self) -> bool:
        """Check if phase was successful."""
        return self.success and self.error_message is None


@dataclass
class HMMLMTrainingConfig:
    """Configuration for HMM LM models training orchestrator."""
    # HMM Base Models Training
    hmm_base_config: Optional[Dict[str, Any]] = None
    
    # HMM Ensemble Training
    hmm_ensemble_config: Optional[Dict[str, Any]] = None
    
    # Analyst Ensemble Training
    analyst_ensemble_config: Optional[Dict[str, Any]] = None
    
    # Tactician Ensemble Training
    tactician_ensemble_config: Optional[Dict[str, Any]] = None
    
    # General settings
    symbol: str = "BTCUSDT"
    exchange: str = "binance"
    base_timeframe: str = "1h"
    analyst_timeframe: str = "5m"
    tactician_timeframe: str = "1m"
    data_dir: str = "historical_data"
    save_models: bool = True
    enable_vectorization: bool = True
    validation_enabled: bool = True


class HMMLMTrainingOrchestrator:
    """
    Orchestrates the complete HMM LM models training pathway.
    
    Training Sequence:
    1. HMM Base Models Training (1h timeframe)
    2. HMM Ensemble Models Training (1h timeframe)
    3. Analyst Ensemble Training (5m timeframe, with HMM integration)
    4. Tactician Ensemble Training (1m timeframe, with HMM integration)
    """
    
    def __init__(self, config: Optional[HMMLMTrainingConfig] = None):
        """Initialize the HMM LM training orchestrator."""
        self.config = config or HMMLMTrainingConfig()
        self.logger = logger.getChild('HMMLMTrainingOrchestrator')
        
        # Training phase results
        self.phase_results: List[TrainingPhaseResult] = []
        self.current_phase: Optional[TrainingPhase] = None
        
        # Artifacts from each phase
        self.hmm_base_artifacts: Optional[Dict[str, Any]] = None
        self.hmm_ensemble_artifacts: Optional[Dict[str, Any]] = None
        self.analyst_ensemble_artifacts: Optional[Dict[str, Any]] = None
        self.tactician_ensemble_artifacts: Optional[Dict[str, Any]] = None
        
        # Initialize training components
        self._initialize_training_components()
        
        tprint("✅ HMM LM Models Training Orchestrator initialized")
    
    def _initialize_training_components(self) -> None:
        """Initialize all training components."""
        try:
            # Initialize HMM base models training
            self.hmm_base_training = create_enhanced_hmm_models_training(
                self.config.hmm_base_config
            )
            
            # Initialize HMM ensemble training
            self.hmm_ensemble_training = create_hmm_ensemble_training_component(
                self.config.hmm_ensemble_config
            )
            
            # Initialize Analyst ensemble training
            self.analyst_ensemble_training = create_analyst_ensemble_training_step(
                self.config.analyst_ensemble_config
            )
            
            # Initialize Tactician ensemble training
            self.tactician_ensemble_training = create_tactician_ensemble_training_step(
                self.config.tactician_ensemble_config
            )
            
            self.logger.info("✅ All training components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize training components: {e}")
            raise RuntimeError(f"Training component initialization failed: {e}") from e
    
    def _start_phase(self, phase: TrainingPhase) -> TrainingPhaseResult:
        """Start a training phase."""
        self.current_phase = phase
        start_time = time.time()
        
        tprint(f"🚀 Starting {phase.value.replace('_', ' ').title()}")
        
        return TrainingPhaseResult(
            phase=phase,
            success=False,
            start_time=start_time,
            end_time=start_time,
            duration=0.0,
            artifacts={}
        )
    
    def _complete_phase(self, phase_result: TrainingPhaseResult, success: bool = True, 
                       artifacts: Optional[Dict[str, Any]] = None, 
                       error_message: Optional[str] = None,
                       metrics: Optional[Dict[str, Any]] = None) -> None:
        """Complete a training phase."""
        phase_result.success = success
        phase_result.end_time = time.time()
        phase_result.duration = phase_result.end_time - phase_result.start_time
        phase_result.artifacts = artifacts or {}
        phase_result.error_message = error_message
        phase_result.metrics = metrics
        
        self.phase_results.append(phase_result)
        
        if success:
            tprint(f"✅ Completed {phase_result.phase.value.replace('_', ' ').title()} in {phase_result.duration:.2f}s")
        else:
            tprint(f"❌ Failed {phase_result.phase.value.replace('_', ' ').title()}: {error_message}")
    
    def execute_complete_training(
        self,
        X_hmm: np.ndarray,
        y_hmm: np.ndarray,
        regime_labels_hmm: np.ndarray,
        X_analyst: np.ndarray,
        y_analyst: np.ndarray,
        regime_labels_analyst: np.ndarray,
        X_tactician: np.ndarray,
        y_tactician: np.ndarray,
        regime_labels_tactician: np.ndarray,
        feature_names_hmm: Optional[List[str]] = None,
        feature_names_analyst: Optional[List[str]] = None,
        feature_names_tactician: Optional[List[str]] = None,
        hmm_states: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Execute the complete HMM LM models training pathway.
        
        Args:
            X_hmm: HMM training features (1h timeframe)
            y_hmm: HMM training targets
            regime_labels_hmm: HMM regime labels
            X_analyst: Analyst training features (5m timeframe)
            y_analyst: Analyst training targets
            regime_labels_analyst: Analyst regime labels
            X_tactician: Tactician training features (1m timeframe)
            y_tactician: Tactician training targets
            regime_labels_tactician: Tactician regime labels
            feature_names_hmm: HMM feature names
            feature_names_analyst: Analyst feature names
            feature_names_tactician: Tactician feature names
            hmm_states: HMM cluster/regime states
            
        Returns:
            Dictionary containing complete training results and artifacts
        """
        overall_start_time = time.time()
        tprint("🎯 Starting Complete HMM LM Models Training Pathway")
        tprint("=" * 60)
        
        try:
            # Phase 1: HMM Base Models Training
            phase1_result = self._start_phase(TrainingPhase.HMM_BASE_TRAINING)
            try:
                hmm_base_results = self.hmm_base_training.execute(
                    X_hmm, y_hmm, regime_labels_hmm, feature_names_hmm, hmm_states
                )
                self.hmm_base_artifacts = hmm_base_results.get('artifacts', {})
                self._complete_phase(phase1_result, True, hmm_base_results, 
                                   metrics=hmm_base_results.get('metadata', {}))
            except Exception as e:
                self._complete_phase(phase1_result, False, error_message=str(e))
                raise
            
            # Phase 2: HMM Ensemble Models Training
            phase2_result = self._start_phase(TrainingPhase.HMM_ENSEMBLE_TRAINING)
            try:
                hmm_ensemble_results = self.hmm_ensemble_training.execute(
                    X_hmm, y_hmm, regime_labels_hmm, feature_names_hmm, hmm_states,
                    self.hmm_base_artifacts.get('hmm_base_models', {}),
                    self.hmm_base_artifacts.get('hmm_training_metrics', {})
                )
                self.hmm_ensemble_artifacts = hmm_ensemble_results.get('artifacts', {})
                self._complete_phase(phase2_result, True, hmm_ensemble_results,
                                   metrics=hmm_ensemble_results.get('metadata', {}))
            except Exception as e:
                self._complete_phase(phase2_result, False, error_message=str(e))
                raise
            
            # Phase 3: Analyst Ensemble Training (with HMM integration)
            phase3_result = self._start_phase(TrainingPhase.ANALYST_ENSEMBLE_TRAINING)
            try:
                analyst_ensemble_results = self.analyst_ensemble_training.execute(
                    X_analyst, y_analyst, regime_labels_analyst, feature_names_analyst, hmm_states,
                    None,  # base_analyst_models (will be created if None)
                    None,  # analyst_training_metrics
                    self.hmm_base_artifacts.get('hmm_base_models', {}),
                    self.hmm_base_artifacts.get('hmm_training_metrics', {})
                )
                self.analyst_ensemble_artifacts = analyst_ensemble_results.get('artifacts', {})
                self._complete_phase(phase3_result, True, analyst_ensemble_results,
                                   metrics=analyst_ensemble_results.get('metadata', {}))
            except Exception as e:
                self._complete_phase(phase3_result, False, error_message=str(e))
                raise
            
            # Phase 4: Tactician Ensemble Training (with HMM integration)
            phase4_result = self._start_phase(TrainingPhase.TACTICIAN_ENSEMBLE_TRAINING)
            try:
                # Prepare HMM data for tactician
                hmm_data = {
                    'regime_features': self.hmm_base_artifacts.get('regime_features'),
                    'hmm_base_models': self.hmm_base_artifacts.get('hmm_base_models', {}),
                    'hmm_ensemble_models': self.hmm_ensemble_artifacts.get('hmm_ensemble_models', {}),
                    'metrics': self.hmm_base_artifacts.get('hmm_training_metrics', {})
                }
                
                tactician_ensemble_results = self.tactician_ensemble_training.execute(
                    X_tactician, y_tactician, regime_labels_tactician, feature_names_tactician, hmm_states,
                    None,  # base_tactician_models (will be created if None)
                    None,  # tactician_training_metrics
                    None,  # analyst_models
                    self.analyst_ensemble_artifacts.get('analyst_ensembles', {}),
                    self.analyst_ensemble_artifacts.get('analyst_ensemble_metrics', {}),
                    hmm_data,
                    self.hmm_base_artifacts.get('hmm_base_models', {}),
                    self.hmm_ensemble_artifacts.get('hmm_ensemble_models', {})
                )
                self.tactician_ensemble_artifacts = tactician_ensemble_results.get('artifacts', {})
                self._complete_phase(phase4_result, True, tactician_ensemble_results,
                                   metrics=tactician_ensemble_results.get('metadata', {}))
            except Exception as e:
                self._complete_phase(phase4_result, False, error_message=str(e))
                raise
            
            # Generate comprehensive results
            overall_duration = time.time() - overall_start_time
            results = self._generate_comprehensive_results(overall_duration)
            
            tprint("=" * 60)
            tprint(f"🎉 Complete HMM LM Models Training Pathway completed in {overall_duration:.2f}s")
            tprint("=" * 60)
            
            return results
            
        except Exception as e:
            overall_duration = time.time() - overall_start_time
            error_msg = f"Complete HMM LM training pathway failed after {overall_duration:.2f}s: {e}"
            tprint(f"❌ {error_msg}")
            
            return {
                'success': False,
                'error': error_msg,
                'duration': overall_duration,
                'phase_results': [phase.__dict__ for phase in self.phase_results],
                'artifacts': {
                    'hmm_base_artifacts': self.hmm_base_artifacts,
                    'hmm_ensemble_artifacts': self.hmm_ensemble_artifacts,
                    'analyst_ensemble_artifacts': self.analyst_ensemble_artifacts,
                    'tactician_ensemble_artifacts': self.tactician_ensemble_artifacts
                }
            }
    
    def _generate_comprehensive_results(self, overall_duration: float) -> Dict[str, Any]:
        """Generate comprehensive training results."""
        try:
            # Calculate success rates
            successful_phases = [p for p in self.phase_results if p.is_successful]
            success_rate = len(successful_phases) / len(self.phase_results) if self.phase_results else 0
            
            # Calculate total artifacts
            total_artifacts = 0
            artifact_breakdown = {}
            
            if self.hmm_base_artifacts:
                total_artifacts += len(self.hmm_base_artifacts)
                artifact_breakdown['hmm_base_models'] = len(self.hmm_base_artifacts)
            
            if self.hmm_ensemble_artifacts:
                total_artifacts += len(self.hmm_ensemble_artifacts)
                artifact_breakdown['hmm_ensemble_models'] = len(self.hmm_ensemble_artifacts)
            
            if self.analyst_ensemble_artifacts:
                total_artifacts += len(self.analyst_ensemble_artifacts)
                artifact_breakdown['analyst_ensemble_models'] = len(self.analyst_ensemble_artifacts)
            
            if self.tactician_ensemble_artifacts:
                total_artifacts += len(self.tactician_ensemble_artifacts)
                artifact_breakdown['tactician_ensemble_models'] = len(self.tactician_ensemble_artifacts)
            
            # Generate comprehensive report
            comprehensive_report = {
                'training_summary': {
                    'overall_success': success_rate == 1.0,
                    'success_rate': success_rate,
                    'total_duration': overall_duration,
                    'phases_completed': len(successful_phases),
                    'phases_total': len(self.phase_results),
                    'total_artifacts': total_artifacts
                },
                'phase_breakdown': [
                    {
                        'phase': phase.phase.value,
                        'success': phase.is_successful,
                        'duration': phase.duration,
                        'artifacts_count': len(phase.artifacts),
                        'error_message': phase.error_message
                    }
                    for phase in self.phase_results
                ],
                'artifact_breakdown': artifact_breakdown,
                'integration_status': {
                    'hmm_base_to_ensemble': self.hmm_ensemble_artifacts is not None,
                    'hmm_to_analyst': self.analyst_ensemble_artifacts is not None,
                    'hmm_to_tactician': self.tactician_ensemble_artifacts is not None,
                    'analyst_to_tactician': self.tactician_ensemble_artifacts is not None
                },
                'recommendations': self._generate_recommendations()
            }
            
            return {
                'success': True,
                'duration': overall_duration,
                'comprehensive_report': comprehensive_report,
                'phase_results': [phase.__dict__ for phase in self.phase_results],
                'artifacts': {
                    'hmm_base_artifacts': self.hmm_base_artifacts,
                    'hmm_ensemble_artifacts': self.hmm_ensemble_artifacts,
                    'analyst_ensemble_artifacts': self.analyst_ensemble_artifacts,
                    'tactician_ensemble_artifacts': self.tactician_ensemble_artifacts
                },
                'metadata': {
                    'config': self.config,
                    'timestamp': datetime.now().isoformat(),
                    'orchestrator_version': '1.0.0'
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate comprehensive results: {e}")
            return {
                'success': False,
                'error': f"Results generation failed: {e}",
                'duration': overall_duration,
                'phase_results': [phase.__dict__ for phase in self.phase_results]
            }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on training results."""
        recommendations = []
        
        # Check success rate
        if len(self.phase_results) > 0:
            success_rate = len([p for p in self.phase_results if p.is_successful]) / len(self.phase_results)
            if success_rate < 1.0:
                recommendations.append(f"Address {len(self.phase_results) - len([p for p in self.phase_results if p.is_successful])} failed phase(s)")
        
        # Check integration status
        if not self.hmm_base_artifacts:
            recommendations.append("HMM base models training failed - review data quality and configuration")
        
        if not self.hmm_ensemble_artifacts:
            recommendations.append("HMM ensemble models training failed - review base models integration")
        
        if not self.analyst_ensemble_artifacts:
            recommendations.append("Analyst ensemble training failed - review HMM integration and data quality")
        
        if not self.tactician_ensemble_artifacts:
            recommendations.append("Tactician ensemble training failed - review all model integrations")
        
        # Check performance
        if self.phase_results:
            avg_duration = sum(p.duration for p in self.phase_results) / len(self.phase_results)
            if avg_duration > 300:  # 5 minutes
                recommendations.append("Consider optimizing training performance - phases taking longer than expected")
        
        if not recommendations:
            recommendations.append("✅ All training phases completed successfully - system ready for deployment")
        
        return recommendations


# Convenience functions
def create_hmm_lm_training_orchestrator(
    config: Optional[HMMLMTrainingConfig] = None
) -> HMMLMTrainingOrchestrator:
    """Create HMM LM training orchestrator."""
    return HMMLMTrainingOrchestrator(config)


def execute_complete_hmm_lm_training(
    X_hmm: np.ndarray,
    y_hmm: np.ndarray,
    regime_labels_hmm: np.ndarray,
    X_analyst: np.ndarray,
    y_analyst: np.ndarray,
    regime_labels_analyst: np.ndarray,
    X_tactician: np.ndarray,
    y_tactician: np.ndarray,
    regime_labels_tactician: np.ndarray,
    config: Optional[HMMLMTrainingConfig] = None,
    feature_names_hmm: Optional[List[str]] = None,
    feature_names_analyst: Optional[List[str]] = None,
    feature_names_tactician: Optional[List[str]] = None,
    hmm_states: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Execute complete HMM LM models training pathway."""
    orchestrator = create_hmm_lm_training_orchestrator(config)
    return orchestrator.execute_complete_training(
        X_hmm, y_hmm, regime_labels_hmm,
        X_analyst, y_analyst, regime_labels_analyst,
        X_tactician, y_tactician, regime_labels_tactician,
        feature_names_hmm, feature_names_analyst, feature_names_tactician,
        hmm_states
    )


# Example usage
if __name__ == "__main__":
    print("HMM LM Models Training Orchestrator")
    print("=" * 50)
    
    # Create configuration
    config = HMMLMTrainingConfig(
        symbol="BTCUSDT",
        exchange="binance",
        base_timeframe="1h",
        analyst_timeframe="5m",
        tactician_timeframe="1m",
        save_models=True,
        enable_vectorization=True
    )
    
    # Create orchestrator
    orchestrator = create_hmm_lm_training_orchestrator(config)
    
    print(f"✅ Created HMM LM training orchestrator")
    print(f"📊 Symbol: {config.symbol}")
    print(f"📊 Exchange: {config.exchange}")
    print(f"📊 Timeframes: HMM={config.base_timeframe}, Analyst={config.analyst_timeframe}, Tactician={config.tactician_timeframe}")
    
    print("\n🎯 Training Pathway:")
    print("1. HMM Base Models Training (1h timeframe)")
    print("2. HMM Ensemble Models Training (1h timeframe)")
    print("3. Analyst Ensemble Training (5m timeframe, with HMM integration)")
    print("4. Tactician Ensemble Training (1m timeframe, with HMM integration)")
    
    print("\n🔄 Integration Flow:")
    print("HMM Base → HMM Ensemble → Analyst → Tactician")
    print("All models are properly integrated for comprehensive market intelligence")