"""
Analyst Pre-ML Orchestration - 15m Timeframe Feature Engineering

This orchestrator applies the complete pre-training pipeline for Analyst models:
1. Multi-horizon profit labeling with differentiated horizons
2. Feature lookback period optimization per regime/cluster
3. PID-based feature generation (interaction, polynomial, cross-timeframe)
4. Final feature selection (multi-stage: 120→100→80→60)

ANALYST CONFIGURATION:
- Timeframe: 15m (higher timeframe for strategic IF-to-trade decisions)
- Training Data: ALL market data (not filtered)
- Output: Features optimized for Analyst model training
- Per-regime optimization: Yes, using regime assignments from market_analysis
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import traceback

# Import pre-training sub-pipeline
try:
    from ..pre_training.sub_pipeline import (
        PreTrainingSubPipeline, SubPipelineConfig, SubPipelineResult, SubPipelineStatus
    )
    PRE_TRAINING_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL: Failed to import PreTrainingSubPipeline: {e}")
    PRE_TRAINING_AVAILABLE = False

# Enhanced imports
try:
    from src.utils.logger import system_logger
    from src.utils.tprint import (
        tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
        tprint_debug, tprint_progress, tprint_timer
    )
    UTILS_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL: Failed to import utilities: {e}")
    UTILS_AVAILABLE = False


class OrchestrationPhase(Enum):
    """Orchestration execution phases."""
    HORIZON_LABELING = "horizon_labeling"
    LOOKBACK_OPTIMIZATION = "lookback_optimization"
    PID_GENERATION = "pid_generation"
    FEATURE_SELECTION = "feature_selection"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AnalystPreMLConfig:
    """Configuration for Analyst pre-ML orchestration."""
    # Data configuration
    symbol: str = "ETHUSDT"
    exchange: str = "binance"
    timeframe: str = "15m"  # ANALYST USES 15m TIMEFRAME
    data_dir: str = "historical_data"
    
    # Execution parameters
    enable_per_regime_optimization: bool = False  # Disabled - using regime probabilities as features instead
    enable_per_cluster_optimization: bool = True
    
    # Output configuration
    output_directory: str = "generated/analyst_pre_ml"
    save_intermediate_results: bool = True
    
    # Hardware optimization
    enable_parallel_processing: bool = True
    memory_limit_gb: float = 8.0
    
    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalystPreMLResult:
    """Result of Analyst pre-ML orchestration."""
    # Execution metadata
    success: bool = False
    execution_time: float = 0.0
    phase: OrchestrationPhase = OrchestrationPhase.HORIZON_LABELING
    
    # Step results
    horizon_labeling_result: Optional[Dict[str, Any]] = None
    lookback_optimization_result: Optional[Dict[str, Any]] = None
    pid_generation_result: Optional[Dict[str, Any]] = None
    feature_selection_result: Optional[Dict[str, Any]] = None
    
    # Output data
    final_features: Optional[pd.DataFrame] = None
    selected_feature_names: Optional[List[str]] = None
    
    # Metadata
    total_samples: int = 0
    total_features_generated: int = 0
    final_feature_count: int = 0
    error_message: Optional[str] = None


class AnalystPreMLOrchestrator:
    """
    Analyst Pre-ML Orchestration.
    
    Orchestrates the complete pre-training pipeline for Analyst models on 15m timeframe.
    Applies per-regime/cluster optimization for all feature engineering steps.
    """
    
    def __init__(self, config: Optional[AnalystPreMLConfig] = None):
        """Initialize the Analyst pre-ML orchestrator."""
        try:
            self.config = config or AnalystPreMLConfig()
            self.logger = system_logger.getChild('AnalystPreMLOrchestrator')
            
            # Initialize pre-training pipeline
            if PRE_TRAINING_AVAILABLE:
                self.pre_training_pipeline = PreTrainingSubPipeline()
                tprint_success("✅ Pre-training pipeline initialized for Analyst")
            else:
                self.pre_training_pipeline = None
                tprint_error("❌ Pre-training pipeline not available")
            
            tprint_success(f"✅ AnalystPreMLOrchestrator initialized (timeframe: {self.config.timeframe})")
            
        except Exception as e:
            tprint_error(f"❌ Failed to initialize AnalystPreMLOrchestrator: {e}")
            raise
    
    async def orchestrate(
        self,
        training_data: pd.DataFrame,
        regime_assignments: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> AnalystPreMLResult:
        """
        Execute the complete pre-ML orchestration for Analyst models.
        
        Args:
            training_data: Input DataFrame with market data (15m timeframe)
            regime_assignments: Optional regime assignments for per-regime optimization
            **kwargs: Additional parameters
            
        Returns:
            AnalystPreMLResult with orchestrated features and metadata
        """
        start_time = tprint_timer()
        tprint_info("🚀 Starting Analyst Pre-ML Orchestration (15m timeframe)...")
        tprint_info(f"📊 Input data shape: {training_data.shape}")
        
        result = AnalystPreMLResult()
        result.total_samples = len(training_data)
        
        try:
            # Validate pre-training pipeline availability
            if not self.pre_training_pipeline:
                raise RuntimeError("Pre-training pipeline not available")
            
            # Create sub-pipeline configuration
            sub_config = SubPipelineConfig(
                symbol=self.config.symbol,
                exchange=self.config.exchange,
                timeframe=self.config.timeframe,  # 15m for Analyst
                data_dir=self.config.data_dir,
                parallel_processing=self.config.enable_parallel_processing,
                custom_params={
                    **self.config.custom_params,
                    'enable_per_regime_optimization': self.config.enable_per_regime_optimization,
                    'enable_per_cluster_optimization': self.config.enable_per_cluster_optimization,
                    'regime_assignments': regime_assignments,
                    'role': 'analyst',  # Mark as Analyst orchestration
                    **kwargs
                }
            )
            
            tprint_info("📋 Configuration:")
            tprint_info(f"  - Timeframe: {self.config.timeframe}")
            tprint_info(f"  - Per-regime optimization: {self.config.enable_per_regime_optimization}")
            tprint_info(f"  - Per-cluster optimization: {self.config.enable_per_cluster_optimization}")
            
            # Step 1: Multi-Horizon Profit Labeling
            tprint_info("📈 Step 1/4: Multi-Horizon Profit Labeling...")
            result.phase = OrchestrationPhase.HORIZON_LABELING
            horizon_result = await self.pre_training_pipeline._execute_multi_horizon_profit_labeler(sub_config)
            
            if not horizon_result.success:
                raise RuntimeError(f"Horizon labeling failed: {horizon_result.error_message}")
            
            result.horizon_labeling_result = horizon_result.artifacts
            tprint_success("✅ Horizon labeling completed")
            
            # Step 2: Feature Lookback Optimization (per-regime/cluster)
            tprint_info("⚙️ Step 2/4: Feature Lookback Optimization (per-regime/cluster)...")
            result.phase = OrchestrationPhase.LOOKBACK_OPTIMIZATION
            lookback_result = await self.pre_training_pipeline._execute_feature_lookback_optimization(sub_config)
            
            if not lookback_result.success:
                raise RuntimeError(f"Lookback optimization failed: {lookback_result.error_message}")
            
            result.lookback_optimization_result = lookback_result.artifacts
            tprint_success("✅ Lookback optimization completed")
            
            # Step 3: PID-Based Feature Generation
            tprint_info("🔧 Step 3/4: PID-Based Feature Generation...")
            result.phase = OrchestrationPhase.PID_GENERATION
            pid_result = await self.pre_training_pipeline._execute_pid_based_feature_generation(sub_config)
            
            if not pid_result.success:
                raise RuntimeError(f"PID generation failed: {pid_result.error_message}")
            
            result.pid_generation_result = pid_result.artifacts
            result.total_features_generated = pid_result.artifacts.get('total_features', 0)
            tprint_success(f"✅ PID generation completed ({result.total_features_generated} features)")
            
            # Step 4: Final Feature Selection
            tprint_info("🎯 Step 4/4: Final Feature Selection (multi-stage)...")
            result.phase = OrchestrationPhase.FEATURE_SELECTION
            selection_result = await self.pre_training_pipeline._execute_final_feature_selection(sub_config)
            
            if not selection_result.success:
                raise RuntimeError(f"Feature selection failed: {selection_result.error_message}")
            
            result.feature_selection_result = selection_result.artifacts
            result.final_features = selection_result.artifacts.get('final_features')
            result.selected_feature_names = selection_result.artifacts.get('selected_features', [])
            result.final_feature_count = len(result.selected_feature_names) if result.selected_feature_names else 0
            tprint_success(f"✅ Feature selection completed ({result.final_feature_count} final features)")
            
            # Mark as completed
            result.success = True
            result.phase = OrchestrationPhase.COMPLETED
            result.execution_time = tprint_timer(start_time)
            
            tprint_success(f"✅ Analyst Pre-ML Orchestration completed in {result.execution_time:.2f}s")
            tprint_info(f"📊 Final feature count: {result.final_feature_count}")
            
            return result
            
        except Exception as e:
            result.success = False
            result.phase = OrchestrationPhase.FAILED
            result.error_message = str(e)
            result.execution_time = tprint_timer(start_time)
            
            tprint_error(f"❌ Analyst Pre-ML Orchestration failed: {e}")
            tprint_error(f"Error details: {traceback.format_exc()}")
            raise
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the orchestrator."""
        return {
            'config': {
                'timeframe': self.config.timeframe,
                'enable_per_regime_optimization': self.config.enable_per_regime_optimization,
                'enable_per_cluster_optimization': self.config.enable_per_cluster_optimization,
                'output_directory': self.config.output_directory
            },
            'component_availability': {
                'pre_training_pipeline': self.pre_training_pipeline is not None
            }
        }


# Convenience function for external usage
async def execute_analyst_pre_ml_orchestration(
    training_data: pd.DataFrame,
    regime_assignments: Optional[pd.DataFrame] = None,
    config: Optional[AnalystPreMLConfig] = None,
    **kwargs
) -> AnalystPreMLResult:
    """
    Execute Analyst pre-ML orchestration.
    
    Args:
        training_data: Input DataFrame with market data (15m timeframe)
        regime_assignments: Optional regime assignments for per-regime optimization
        config: Optional configuration
        **kwargs: Additional parameters
        
    Returns:
        AnalystPreMLResult with orchestrated features and metadata
    """
    orchestrator = AnalystPreMLOrchestrator(config)
    return await orchestrator.orchestrate(training_data, regime_assignments, **kwargs)
