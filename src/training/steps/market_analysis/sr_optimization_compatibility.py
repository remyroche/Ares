"""Backward compatibility wrapper for SROptimizationStep using MarketAnalysisSubPipeline."""

import asyncio
from typing import Any, Dict, Optional
from .sub_pipeline import MarketAnalysisSubPipeline, SubPipelineConfig, ExecutionMode


class SROptimizationStep(MarketAnalysisSubPipeline):
    """
    Backward compatibility wrapper for SROptimizationStep.
    
    This class provides the same interface as the original SROptimizationStep
    while using the new MarketAnalysisSubPipeline infrastructure internally.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with backward compatible interface."""
        # Convert the old config format to SubPipelineConfig
        sub_pipeline_config = self._convert_config(config)
        super().__init__(sub_pipeline_config)
        
        # Store original config for compatibility
        self.original_config = config
        
        # Set up logging
        self.logger = self.logger.getChild('SROptimizationStep')
        self.logger.info("🎯 SROptimizationStep initialized with backward compatibility")
    
    def _convert_config(self, config: Dict[str, Any]) -> SubPipelineConfig:
        """Convert old config format to SubPipelineConfig."""
        # Extract relevant configuration
        sr_config = config.get('sr_optimization', {})
        training_mode = config.get('training_mode', 'full')
        
        # Determine execution mode
        if training_mode == 'light':
            mode = ExecutionMode.LIGHT
        elif training_mode == 'blank':
            mode = ExecutionMode.BLANK
        else:
            mode = ExecutionMode.FULL
        
        # Create SubPipelineConfig
        sub_config = SubPipelineConfig(
            mode=mode,
            symbol=config.get('symbol', 'BTCUSDT'),
            exchange=config.get('exchange', 'binance'),
            timeframe=config.get('timeframe', '1m'),
            data_dir=config.get('data_dir', './data'),
            output_dir=config.get('output_dir', './output')
        )
        
        return sub_config
    
    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the SR optimization pipeline with backward compatible interface.
        
        This method provides the same interface as the original SROptimizationStep
        while orchestrating the three SR stages internally.
        """
        self.logger.info('🎯 Starting SROptimizationStep execution with backward compatibility')
        
        try:
            # Extract data from pipeline state
            data = pipeline_state.get('dataframe')
            if data is None:
                raise ValueError("No dataframe found in pipeline state")
            
            # Update config with data information
            self.config.symbol = training_input.get('symbol', 'BTCUSDT')
            self.config.exchange = training_input.get('exchange', 'binance')
            self.config.timeframe = training_input.get('timeframe', '1m')
            
            # Execute the three SR stages in sequence
            results = {}
            
            # Stage 1: SR Detection
            self.logger.info('🎯 Executing Stage 1: SR Detection')
            detection_result = await self.execute_sub_pipeline('sr_detection', self.config)
            if detection_result.success:
                results['sr_levels'] = detection_result.artifacts.get('sr_levels', [])
                results['sr_metrics'] = detection_result.artifacts.get('sr_metrics', {})
                self.logger.info(f"✅ SR Detection completed: {len(results['sr_levels'])} levels detected")
            else:
                self.logger.error(f"❌ SR Detection failed: {detection_result.error}")
                return {
                    'success': False,
                    'error': f"SR Detection failed: {detection_result.error}",
                    'stage': 'sr_detection'
                }
            
            # Stage 2: SR Clustering
            self.logger.info('🚀 Executing Stage 2: SR Clustering')
            clustering_result = await self.execute_sub_pipeline('sr_clustering', self.config)
            if clustering_result.success:
                results['clustered_levels'] = clustering_result.artifacts.get('clustered_levels', [])
                results['cluster_metrics'] = clustering_result.artifacts.get('cluster_metrics', {})
                self.logger.info(f"✅ SR Clustering completed: {len(results['clustered_levels'])} clusters")
            else:
                self.logger.error(f"❌ SR Clustering failed: {clustering_result.error}")
                return {
                    'success': False,
                    'error': f"SR Clustering failed: {clustering_result.error}",
                    'stage': 'sr_clustering'
                }
            
            # Stage 3: SR ML Learning
            self.logger.info('🤖 Executing Stage 3: SR ML Learning')
            ml_result = await self.execute_sub_pipeline('sr_ml_learning', self.config)
            if ml_result.success:
                results['ml_models'] = ml_result.artifacts.get('ml_models', [])
                results['ml_metrics'] = ml_result.artifacts.get('ml_metrics', {})
                self.logger.info(f"✅ SR ML Learning completed: {len(results['ml_models'])} models")
            else:
                self.logger.error(f"❌ SR ML Learning failed: {ml_result.error}")
                return {
                    'success': False,
                    'error': f"SR ML Learning failed: {ml_result.error}",
                    'stage': 'sr_ml_learning'
                }
            
            # Calculate total execution time
            total_time = (
                detection_result.execution_time + 
                clustering_result.execution_time + 
                ml_result.execution_time
            )
            
            self.logger.info('🎯 SROptimizationStep execution completed successfully')
            self.logger.info(f"📊 Total execution time: {total_time:.2f} seconds")
            
            return {
                'success': True,
                'sr_levels': results['sr_levels'],
                'clustered_levels': results['clustered_levels'],
                'ml_models': results['ml_models'],
                'sr_metrics': results['sr_metrics'],
                'cluster_metrics': results['cluster_metrics'],
                'ml_metrics': results['ml_metrics'],
                'execution_time': total_time,
                'stage_times': {
                    'detection': detection_result.execution_time,
                    'clustering': clustering_result.execution_time,
                    'ml_learning': ml_result.execution_time
                },
                'stage': 'complete_sr_optimization'
            }
            
        except Exception as e:
            self.logger.error(f'❌ SROptimizationStep execution failed: {e}')
            import traceback
            self.logger.error(f'❌ Error details: {traceback.format_exc()}')
            return {
                'success': False,
                'error': str(e),
                'stage': 'complete_sr_optimization'
            }
    
    def validate_config(self):
        """Validate configuration for backward compatibility."""
        self.logger.info('🔍 Validating SROptimizationStep configuration...')
        
        # Validate required configuration
        required_keys = ['sr_optimization']
        for key in required_keys:
            if key not in self.original_config:
                self.logger.warning(f"⚠️ Missing configuration key: {key}")
        
        # Validate SR optimization config
        sr_config = self.original_config.get('sr_optimization', {})
        required_sr_keys = ['min_touches', 'tolerance_pct', 'lookback_periods']
        for key in required_sr_keys:
            if key not in sr_config:
                self.logger.warning(f"⚠️ Missing SR optimization key: {key}")
        
        self.logger.info('✅ SROptimizationStep configuration validation completed')
        return True
    
    def get_status(self):
        """Get status for backward compatibility."""
        return {
            'stage': 'sr_optimization',
            'status': 'ready',
            'config': self.original_config,
            'sub_pipeline_status': 'initialized'
        }