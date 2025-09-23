"""
Tactician Training Adapter

This module provides a tactician-specific adapter that separates long & short signals
from the Analyst and adapts the training logic for differentiation between longs & shorts.
This is used on 1m timeframe for the Tactician to train 2 separate models (long and short).
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Import the existing training components
from .multi_horizon_profit_labeler import MultiHorizonProfitLabeler, MultiHorizonConfig
from .pid_based_feature_generation.pid_based_feature_generation_component import PIDBasedFeatureGenerationComponent
from .feature_lookback_optimization.feature_lookback_optimization import FeatureLookbackOptimizationComponent
from .final_feature_selection_step import FinalFeatureSelectionStep

# Import base components
from .components.base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult

# Import logger
try:
    from src.utils.logger import system_logger
    logger = system_logger.getChild('TacticianTrainingAdapter')
except ImportError:
    logger = logging.getLogger('TacticianTrainingAdapter')
    logger.setLevel(logging.INFO)


class TacticianTrainingStatus(Enum):
    """Status of tactician training process."""
    PENDING = "pending"
    SEPARATING_SIGNALS = "separating_signals"
    TRAINING_LONG = "training_long"
    TRAINING_SHORT = "training_short"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TacticianTrainingResult:
    """Result of tactician training process."""
    long_model_result: Optional[Dict[str, Any]] = None
    short_model_result: Optional[Dict[str, Any]] = None
    long_features: Optional[List[str]] = None
    short_features: Optional[List[str]] = None
    long_lookback_periods: Optional[Dict[str, int]] = None
    short_lookback_periods: Optional[Dict[str, int]] = None
    training_status: TacticianTrainingStatus = TacticianTrainingStatus.PENDING
    execution_time: float = 0.0
    error_message: Optional[str] = None


class TacticianTrainingAdapter(BaseMarketAnalysisComponent):
    """
    Tactician Training Adapter.
    
    This adapter separates long & short signals from the Analyst and adapts
    the training logic for differentiation between longs & shorts on 1m timeframe.
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the tactician training adapter."""
        super().__init__(config)
        self.logger = logger.getChild('TacticianTrainingAdapter')
        
        # Ensure we're in tactician mode (1m timeframe)
        if self.config.timeframe != '1m':
            raise ValueError(f"TacticianTrainingAdapter requires 1m timeframe, got {self.config.timeframe}")
        
        # Initialize training status
        self.training_status = TacticianTrainingStatus.PENDING
        self.start_time: Optional[float] = None
        
        # Initialize components for long and short training
        self._initialize_components()
        
        self.logger.info("🔧 TacticianTrainingAdapter initialized")
        self.logger.info(f"📊 Symbol: {self.config.symbol}")
        self.logger.info(f"📊 Exchange: {self.config.exchange}")
        self.logger.info(f"📊 Timeframe: {self.config.timeframe}")
    
    def _initialize_components(self):
        """Initialize required components for tactician training."""
        # Initialize multi-horizon profit labeler (tactician mode - with long/short differentiation)
        self.multi_horizon_labeler = MultiHorizonProfitLabeler(
            MultiHorizonConfig(analyst_mode=False)  # Tactician mode
        )
        
        # Initialize PID-based feature generation components
        self.long_pid_generator = PIDBasedFeatureGenerationComponent(self.config)
        self.short_pid_generator = PIDBasedFeatureGenerationComponent(self.config)
        
        # Initialize feature lookback optimization components
        self.long_lookback_optimizer = FeatureLookbackOptimizationComponent(self.config)
        self.short_lookback_optimizer = FeatureLookbackOptimizationComponent(self.config)
        
        # Initialize final feature selection components
        self.long_feature_selector = FinalFeatureSelectionStep({
            'timeframe': self.config.timeframe,
            'model_type': 'long'
        })
        self.short_feature_selector = FinalFeatureSelectionStep({
            'timeframe': self.config.timeframe,
            'model_type': 'short'
        })
        
        self.logger.info("✅ Tactician training components initialized")
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['tactician_training_result']
    
    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute tactician training with long/short signal separation.
        
        Args:
            data: Market data for training
            pipeline_state: Current pipeline state
            
        Returns:
            ComponentResult with tactician training results
        """
        self.start_time = time.time()
        self.training_status = TacticianTrainingStatus.SEPARATING_SIGNALS
        
        self.logger.info('🔧 Starting Tactician Training with Long/Short Separation')
        self._report_checkpoint('start', 'training_started', {
            'symbol': self.config.symbol,
            'exchange': self.config.exchange,
            'timeframe': self.config.timeframe
        })
        
        try:
            # Step 1: Separate long & short signals from Analyst
            self.logger.info('📊 Separating long & short signals from Analyst...')
            separated_signals = await self._separate_analyst_signals(data, pipeline_state)
            self._report_checkpoint('signal_separation', 'completed', {
                'long_samples': len(separated_signals.get('long_data', [])),
                'short_samples': len(separated_signals.get('short_data', []))
            })
            
            # Step 2: Train long model
            self.logger.info('🚀 Training Long Tactician Model...')
            self.training_status = TacticianTrainingStatus.TRAINING_LONG
            long_result = await self._train_directional_model(
                separated_signals['long_data'], 
                'long',
                pipeline_state
            )
            self._report_checkpoint('long_training', 'completed', {
                'long_features': len(long_result.get('features', [])),
                'long_lookback_periods': len(long_result.get('lookback_periods', {}))
            })
            
            # Step 3: Train short model
            self.logger.info('🚀 Training Short Tactician Model...')
            self.training_status = TacticianTrainingStatus.TRAINING_SHORT
            short_result = await self._train_directional_model(
                separated_signals['short_data'], 
                'short',
                pipeline_state
            )
            self._report_checkpoint('short_training', 'completed', {
                'short_features': len(short_result.get('features', [])),
                'short_lookback_periods': len(short_result.get('lookback_periods', {}))
            })
            
            # Step 4: Create comprehensive artifacts
            artifacts = await self._create_tactician_artifacts(long_result, short_result)
            
            # Step 5: Generate final report
            final_report = self._generate_tactician_report(long_result, short_result)
            self._report_checkpoint('completion', 'success', {
                'long_features': len(long_result.get('features', [])),
                'short_features': len(short_result.get('features', [])),
                'execution_time': time.time() - self.start_time
            })
            
            self.training_status = TacticianTrainingStatus.COMPLETED
            
            self.logger.info(f'✅ Tactician Training completed: Long model with {len(long_result.get("features", []))} features, Short model with {len(short_result.get("features", []))} features')
            return ComponentResult(
                success=True,
                artifacts=artifacts,
                metadata={
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    'long_features_count': len(long_result.get('features', [])),
                    'short_features_count': len(short_result.get('features', [])),
                    'training_status': self.training_status.value,
                    'final_report': final_report,
                    'execution_time': time.time() - self.start_time
                }
            )
            
        except Exception as e:
            self.training_status = TacticianTrainingStatus.FAILED
            
            self.logger.error(f'❌ Tactician Training failed: {e}')
            
            # Generate failure report
            failure_report = self._generate_failure_report(str(e))
            self._report_checkpoint('completion', 'failed', {
                'error_type': type(e).__name__,
                'execution_time': time.time() - self.start_time if self.start_time else 0
            })
            
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=str(e),
                metadata={
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    'training_status': self.training_status.value,
                    'failure_report': failure_report,
                    'execution_time': time.time() - self.start_time if self.start_time else 0
                }
            )
    
    async def _separate_analyst_signals(self, data: Any, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Separate long & short signals from Analyst results."""
        try:
            # Get Analyst results from pipeline state
            analyst_results = pipeline_state.get('analyst_results', {})
            
            if not analyst_results:
                # If no analyst results, use the multi-horizon labeler to generate them
                self.logger.info("📊 No analyst results found, generating multi-horizon labels...")
                labeled_data = self.multi_horizon_labeler.generate_labels(data)
                
                # Extract long and short signals
                long_data = self._extract_directional_data(labeled_data, 'long')
                short_data = self._extract_directional_data(labeled_data, 'short')
            else:
                # Use existing analyst results
                long_data = analyst_results.get('long_data', [])
                short_data = analyst_results.get('short_data', [])
            
            return {
                'long_data': long_data,
                'short_data': short_data,
                'separation_method': 'analyst_results' if analyst_results else 'multi_horizon_labeling'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to separate analyst signals: {e}")
            raise
    
    def _extract_directional_data(self, labeled_data: Any, direction: str) -> Any:
        """Extract directional data for long or short signals."""
        try:
            if hasattr(labeled_data, 'columns'):
                # DataFrame case
                direction_columns = [col for col in labeled_data.columns if f'_{direction}_' in col]
                if direction_columns:
                    return labeled_data[direction_columns]
                else:
                    # Fallback to all data if no directional columns
                    return labeled_data
            else:
                # Other data types
                return labeled_data
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error extracting {direction} data: {e}")
            return labeled_data
    
    async def _train_directional_model(self, directional_data: Any, direction: str, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Train a directional model (long or short) using the existing training logic."""
        try:
            self.logger.info(f"🔧 Training {direction} model with {len(directional_data) if hasattr(directional_data, '__len__') else 'unknown'} samples")
            
            # Step 1: Feature lookback optimization
            self.logger.info(f"⚙️ Optimizing feature lookback periods for {direction} model...")
            lookback_optimizer = self.long_lookback_optimizer if direction == 'long' else self.short_lookback_optimizer
            lookback_result = await lookback_optimizer.execute(directional_data, pipeline_state)
            
            # Step 2: PID-based feature generation
            self.logger.info(f"🚀 Generating PID-based features for {direction} model...")
            pid_generator = self.long_pid_generator if direction == 'long' else self.short_pid_generator
            pid_result = await pid_generator.execute(directional_data, pipeline_state)
            
            # Step 3: Final feature selection
            self.logger.info(f"📊 Performing final feature selection for {direction} model...")
            feature_selector = self.long_feature_selector if direction == 'long' else self.short_feature_selector
            selection_result = await feature_selector.execute_final_feature_selection(
                self.config.symbol,
                self.config.exchange,
                self.config.timeframe,
                "historical_data"
            )
            
            return {
                'direction': direction,
                'features': pid_result.artifacts.get('pid_based_feature_generation_result', {}).get('combined_feature_names', []),
                'lookback_periods': lookback_result.artifacts.get('feature_lookback_optimization_result', {}).get('optimized_lookback_periods', {}),
                'selected_features': selection_result,
                'pid_result': pid_result.artifacts,
                'lookback_result': lookback_result.artifacts,
                'training_successful': True
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to train {direction} model: {e}")
            return {
                'direction': direction,
                'features': [],
                'lookback_periods': {},
                'selected_features': None,
                'pid_result': {},
                'lookback_result': {},
                'training_successful': False,
                'error': str(e)
            }
    
    async def _create_tactician_artifacts(self, long_result: Dict[str, Any], short_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive artifacts for tactician training."""
        return {
            'tactician_training_result': {
                'long_model': {
                    'features': long_result.get('features', []),
                    'lookback_periods': long_result.get('lookback_periods', {}),
                    'selected_features': long_result.get('selected_features'),
                    'training_successful': long_result.get('training_successful', False),
                    'pid_result': long_result.get('pid_result', {}),
                    'lookback_result': long_result.get('lookback_result', {})
                },
                'short_model': {
                    'features': short_result.get('features', []),
                    'lookback_periods': short_result.get('lookback_periods', {}),
                    'selected_features': short_result.get('selected_features'),
                    'training_successful': short_result.get('training_successful', False),
                    'pid_result': short_result.get('pid_result', {}),
                    'lookback_result': short_result.get('lookback_result', {})
                },
                'training_summary': {
                    'long_features_count': len(long_result.get('features', [])),
                    'short_features_count': len(short_result.get('features', [])),
                    'long_training_successful': long_result.get('training_successful', False),
                    'short_training_successful': short_result.get('training_successful', False),
                    'execution_time': time.time() - self.start_time if self.start_time else 0
                },
                'metadata': {
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    'training_timestamp': datetime.now().isoformat(),
                    'component_version': '1.0.0'
                }
            }
        }
    
    def _generate_tactician_report(self, long_result: Dict[str, Any], short_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive report for tactician training."""
        return {
            'execution_summary': {
                'total_time': time.time() - self.start_time if self.start_time else 0,
                'success': True,
                'long_features': len(long_result.get('features', [])),
                'short_features': len(short_result.get('features', [])),
                'long_training_successful': long_result.get('training_successful', False),
                'short_training_successful': short_result.get('training_successful', False)
            },
            'model_breakdown': {
                'long_model': {
                    'features_count': len(long_result.get('features', [])),
                    'lookback_periods_count': len(long_result.get('lookback_periods', {})),
                    'training_successful': long_result.get('training_successful', False)
                },
                'short_model': {
                    'features_count': len(short_result.get('features', [])),
                    'lookback_periods_count': len(short_result.get('lookback_periods', {})),
                    'training_successful': short_result.get('training_successful', False)
                }
            },
            'recommendations': self._generate_tactician_recommendations(long_result, short_result)
        }
    
    def _generate_tactician_recommendations(self, long_result: Dict[str, Any], short_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations for tactician training."""
        recommendations = []
        
        long_successful = long_result.get('training_successful', False)
        short_successful = short_result.get('training_successful', False)
        
        if not long_successful:
            recommendations.append("Long model training failed - review long signal data quality and availability")
        
        if not short_successful:
            recommendations.append("Short model training failed - review short signal data quality and availability")
        
        if long_successful and short_successful:
            long_features = len(long_result.get('features', []))
            short_features = len(short_result.get('features', []))
            
            if long_features == 0:
                recommendations.append("No long features generated - check long signal data and PID thresholds")
            if short_features == 0:
                recommendations.append("No short features generated - check short signal data and PID thresholds")
            
            if long_features > 0 and short_features > 0:
                recommendations.append(f"Successfully trained both models: Long ({long_features} features), Short ({short_features} features)")
        
        return recommendations
    
    def _generate_failure_report(self, error_message: str) -> Dict[str, Any]:
        """Generate failure report."""
        return {
            'execution_summary': {
                'total_time': time.time() - self.start_time if self.start_time else 0,
                'success': False,
                'long_features': 0,
                'short_features': 0,
                'training_status': self.training_status.value
            },
            'error_details': {
                'error_message': error_message,
                'error_type': 'tactician_training_failed'
            },
            'recommendations': [
                "Review error logs for detailed failure information",
                "Check analyst signal data quality and availability",
                "Verify tactician training configuration parameters",
                "Ensure required dependencies are available"
            ]
        }
    
    def _report_checkpoint(self, step: str, status: str, details: Dict[str, Any]):
        """Report progress at key checkpoints."""
        self.logger.info(f"📊 [{step}] {status} - {details}")


# Convenience function for pipeline integration
async def run_tactician_training(symbol: str, 
                               exchange: str, 
                               timeframe: str = '1m', 
                               data: Any = None,
                               pipeline_state: Optional[Dict[str, Any]] = None) -> TacticianTrainingResult:
    """Run tactician training with long/short separation."""
    
    config = ComponentConfig(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe
    )
    
    adapter = TacticianTrainingAdapter(config)
    result = await adapter.execute(data or {}, pipeline_state or {})
    
    if result.success:
        return TacticianTrainingResult(
            long_model_result=result.artifacts.get('tactician_training_result', {}).get('long_model'),
            short_model_result=result.artifacts.get('tactician_training_result', {}).get('short_model'),
            long_features=result.artifacts.get('tactician_training_result', {}).get('long_model', {}).get('features'),
            short_features=result.artifacts.get('tactician_training_result', {}).get('short_model', {}).get('features'),
            long_lookback_periods=result.artifacts.get('tactician_training_result', {}).get('long_model', {}).get('lookback_periods'),
            short_lookback_periods=result.artifacts.get('tactician_training_result', {}).get('short_model', {}).get('lookback_periods'),
            training_status=TacticianTrainingStatus.COMPLETED,
            execution_time=result.metadata.get('execution_time', 0.0)
        )
    else:
        return TacticianTrainingResult(
            training_status=TacticianTrainingStatus.FAILED,
            execution_time=result.metadata.get('execution_time', 0.0),
            error_message=result.error_message
        )