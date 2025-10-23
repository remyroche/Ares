from typing import Dict, Optional, Any
"""Enhanced core training manager with comprehensive monitoring and error detection.

This module provides the main training manager that coordinates
the training pipeline execution with enhanced error detection,
monitoring, and reporting capabilities.
"""
from ..simplified_training_manager import SimplifiedTrainingManager
from src.utils.logger import system_logger
from src.core.decorators.errors import handles_errors
from src.utils.ml_common.utils.base_safeguards import MLTrainingSafeguards
import logging
from datetime import datetime

class TrainingManager:
    """Enhanced main training manager for the ML pipeline with comprehensive monitoring.

    This is a facade that provides a simple interface to the training pipeline
    while delegating to specialized components and providing enhanced error detection,
    monitoring, and reporting capabilities.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize enhanced training manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild('TrainingManager')
        self.pipeline_manager = SimplifiedTrainingManager(config)

        # Enhanced monitoring and error detection
        self.safeguards = MLTrainingSafeguards(config.get('safeguards', {}))
        self.training_history = []

        # Execution tracking
        self.is_initialized = False
        self.current_execution = None
        self.execution_start_time = None

        self.logger.info("🔍 Enhanced Training Manager initialized with monitoring capabilities")

    @handles_errors(Exception, fallback = False)
    async def initialize(self) -> bool:
        """Initialize the enhanced training manager with monitoring.

        Returns:
            True if initialization successful
        """
        try:
            self.logger.info('🔧 Initializing Enhanced Training Manager...')
            self.execution_start_time = datetime.now()

            # Initialize pipeline manager
            if not await self.pipeline_manager.initialize():
                error_context = {
                    'component': 'training_manager',
                    'function': 'initialize',
                    'error_type': 'initialization_failure'
                }
                self.safeguards.detect_and_classify_error(
                    Exception("Pipeline manager initialization failed"),
                    error_context
                )
                self.logger.error('❌ Failed to initialize pipeline manager')
                return False

            self.is_initialized = True
            self.logger.info('✅ Enhanced Training Manager initialized successfully')
            return True

        except Exception as e:
            error_context = {
                'component': 'training_manager',
                'function': 'initialize',
                'error_type': 'initialization_exception'
            }
            self.safeguards.detect_and_classify_error(e, error_context)
            self.logger.exception(f'❌ Initialization failed: {e}')
            return False

    def track_training_execution(self, execution_id: str, status: str,
                               metrics: Optional[Dict[str, Any]] = None):
        """Track training execution with enhanced monitoring."""
        try:
            execution_record = {
                'execution_id': execution_id,
                'timestamp': datetime.now(),
                'status': status,
                'metrics': metrics or {},
                'duration': None
            }

            if self.execution_start_time:
                execution_record['duration'] = (
                    datetime.now() - self.execution_start_time
                ).total_seconds()

            self.training_history.append(execution_record)

            # Keep only recent history (last 50 executions)
            if len(self.training_history) > 50:
                self.training_history = self.training_history[-50:]

            self.logger.debug(f"📊 Training execution tracked: {execution_id} - {status}")

        except Exception as e:
            self.logger.error(f"❌ Failed to track training execution: {e}")

    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary with monitoring data."""
        try:
            # Calculate execution statistics
            total_executions = len(self.training_history)
            successful_executions = sum(1 for e in self.training_history if e['status'] == 'completed')
            failed_executions = sum(1 for e in self.training_history if e['status'] == 'failed')

            # Calculate average duration
            durations = [e['duration'] for e in self.training_history if e['duration'] is not None]
            avg_duration = sum(durations) / len(durations) if durations else 0

            # Get error summary from safeguards
            error_summary = self.safeguards.get_error_summary()

            return {
                'training_summary': {
                    'total_executions': total_executions,
                    'successful_executions': successful_executions,
                    'failed_executions': failed_executions,
                    'success_rate': successful_executions / max(1, total_executions),
                    'average_duration': avg_duration,
                    'is_initialized': self.is_initialized,
                    'current_execution': self.current_execution
                },
                'error_summary': error_summary,
                'recent_executions': self.training_history[-10:] if self.training_history else []
            }

        except Exception as e:
            self.logger.error(f"❌ Failed to get training summary: {e}")
            return {'error': str(e)}

    def check_health_status(self) -> Dict[str, Any]:
        """Check overall health status of the training system."""
        try:
            health_status = {
                'overall_health': 'good',
                'issues': [],
                'recommendations': [],
                'risk_level': 'low'
            }

            # Check error rates
            error_summary = self.safeguards.get_error_summary()
            if error_summary['recent_errors_1h'] > 10:
                health_status['issues'].append(f"High error rate: {error_summary['recent_errors_1h']} errors in last hour")
                health_status['recommendations'].append("Investigate recent errors and check system resources")
                health_status['risk_level'] = 'high'

            # Check execution success rate
            if self.training_history:
                recent_executions = self.training_history[-10:]
                success_rate = sum(1 for e in recent_executions if e['status'] == 'completed') / len(recent_executions)
                if success_rate < 0.8:
                    health_status['issues'].append(f"Low success rate: {success_rate:.2%}")
                    health_status['recommendations'].append("Review training pipeline and error logs")
                    if health_status['risk_level'] == 'low':
                        health_status['risk_level'] = 'medium'

            # Check for critical errors
            if error_summary['severity_distribution'].get('critical', 0) > 0:
                health_status['issues'].append("Critical errors detected")
                health_status['recommendations'].append("Immediate attention required for critical errors")
                health_status['risk_level'] = 'critical'

            # Determine overall health
            if health_status['risk_level'] == 'critical':
                health_status['overall_health'] = 'critical'
            elif health_status['risk_level'] == 'high':
                health_status['overall_health'] = 'poor'
            elif health_status['risk_level'] == 'medium':
                health_status['overall_health'] = 'fair'

            return health_status

        except Exception as e:
            self.logger.error(f"❌ Health check failed: {e}")
            return {
                'overall_health': 'unknown',
                'issues': ['Health check failed'],
                'recommendations': ['Manual review required'],
                'risk_level': 'unknown'
            }

    async def train(self, symbol: str, exchange: str, start_step: Optional[str]=None, end_step: Optional[str]=None, force_rerun: bool = False) -> Dict[str, Any]:
        """Execute the training pipeline.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            exchange: Exchange name (e.g., "binance")
            start_step: Optional starting step
            end_step: Optional ending step
            force_rerun: Force re-execution of completed steps

        Returns:
            Training results
        """
        if not self.is_initialized:
            await self.initialize()
        self.logger.info(f'🚀 Starting training for {symbol} on {exchange}')
        self.pipeline_manager.symbol = symbol
        self.pipeline_manager.exchange = exchange
        result = await self.pipeline_manager.execute_pipeline(start_step = start_step, end_step = end_step, force_rerun = force_rerun)
        if result['success']:
            self.logger.info('✅ Training completed successfully')
        else:
            self.logger.error(f"❌ Training failed: {result.get('error', 'Unknown error')}")
        return result

    async def get_status(self) -> Dict[str, Any]:
        """Get current training status.

        Returns:
            Status dictionary
        """
        return self.pipeline_manager.get_pipeline_status()

    async def cleanup(self) -> None:
        """Clean up resources."""
        await self.pipeline_manager.cleanup()
        self.logger.info('🧹 Training Manager cleaned up')

async def create_training_manager(config: Dict[str, Any]) -> TrainingManager:
    """Create and initialize a training manager.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized TrainingManager
    """
    manager = TrainingManager(config)
    if await manager.initialize():
        return manager
    else:
        raise RuntimeError('Failed to initialize training manager')
