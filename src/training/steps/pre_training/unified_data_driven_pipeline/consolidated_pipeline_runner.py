"""
Consolidated Pipeline Runner

This module provides functions to run the consolidated pipeline up to specific steps,
allowing the step files to call the consolidated pipeline at the proper places.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from .consolidated_pipeline import (
    UnifiedDataDrivenPipeline,
    create_unified_pipeline,
    ConsolidatedPipelineResult
)
from .core.config import UnifiedPipelineConfig, create_default_config
from .core.simplified_config import (
    create_full_config,
    create_blank_config, 
    create_light_config,
    create_config_by_intensity,
    PipelineIntensity
)


class ConsolidatedPipelineRunner:
    """Runner for executing consolidated pipeline up to specific steps."""
    
    def __init__(self, config: Optional[UnifiedPipelineConfig] = None):
        """Initialize the pipeline runner."""
        self.config = config or create_default_config()
        self.pipeline = create_unified_pipeline(self.config)
        self.logger = logging.getLogger(__name__)
    
    async def run_data_validation_step(self, 
                                     data: pd.DataFrame,
                                     symbol: str = "ETHUSDT",
                                     timeframe: str = "15m",
                                     direction: str = "longs",
                                     intensity: str = "blank",
                                     lookback_days: Optional[int] = None,
                                     start_date: Optional[str] = None,
                                     end_date: Optional[str] = None,
                                     exchange: str = "binance",
                                     custom_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run pipeline up to data validation step."""
        try:
            # Configure pipeline based on intensity
            config = self._create_config_from_intensity(intensity, custom_overrides)
            self.pipeline = create_unified_pipeline(config)
            
            # Create pipeline state
            pipeline_state = {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'exchange': exchange,
                'step': 'data_validation'
            }
            
            # Run pipeline up to data validation
            result = await self.pipeline.process(data, timeframe=timeframe, pipeline_state=pipeline_state)
            
            # Extract validation results
            validation_result = {
                'success': result.success,
                'data_quality_score': getattr(result, 'data_quality_score', 0.0),
                'validation_metadata': getattr(result, 'validation_metadata', {}),
                'artifacts': result.artifacts or {},
                'error_message': result.error_message if not result.success else None
            }
            
            # Generate human-readable report
            await self._generate_data_validation_report(validation_result, data)
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Data validation step failed: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'artifacts': {},
                'data_quality_score': 0.0,
                'validation_metadata': {}
            }
    
    async def run_feature_generation_step(self,
                                        data: pd.DataFrame,
                                        symbol: str = "ETHUSDT",
                                        timeframe: str = "15m",
                                        direction: str = "longs",
                                        intensity: str = "blank",
                                        lookback_days: Optional[int] = None,
                                        start_date: Optional[str] = None,
                                        end_date: Optional[str] = None,
                                        exchange: str = "binance",
                                        custom_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run pipeline up to feature generation step."""
        try:
            # Configure pipeline based on intensity
            config = self._create_config_from_intensity(intensity, custom_overrides)
            self.pipeline = create_unified_pipeline(config)
            
            # Create pipeline state
            pipeline_state = {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'exchange': exchange,
                'step': 'feature_generation'
            }
            
            # Run pipeline up to feature generation
            result = await self.pipeline.process(data, timeframe=timeframe, pipeline_state=pipeline_state)
            
            # Extract feature generation results
            feature_result = {
                'success': result.success,
                'generated_features': getattr(result, 'generated_features', pd.DataFrame()),
                'feature_metadata': getattr(result, 'feature_metadata', {}),
                'generation_metrics': getattr(result, 'generation_metrics', {}),
                'artifacts': result.artifacts or {},
                'error_message': result.error_message if not result.success else None
            }
            
            # Generate human-readable report
            await self._generate_feature_generation_report(feature_result, data)
            
            return feature_result
            
        except Exception as e:
            self.logger.error(f"Feature generation step failed: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'artifacts': {},
                'generated_features': pd.DataFrame(),
                'feature_metadata': {},
                'generation_metrics': {}
            }
    
    async def run_feature_selection_step(self,
                                       data: pd.DataFrame,
                                       symbol: str = "ETHUSDT",
                                       timeframe: str = "15m",
                                       direction: str = "longs",
                                       intensity: str = "blank",
                                       lookback_days: Optional[int] = None,
                                       start_date: Optional[str] = None,
                                       end_date: Optional[str] = None,
                                       exchange: str = "binance",
                                       custom_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run pipeline up to feature selection step."""
        try:
            # Configure pipeline based on intensity
            config = self._create_config_from_intensity(intensity, custom_overrides)
            self.pipeline = create_unified_pipeline(config)
            
            # Create pipeline state
            pipeline_state = {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'exchange': exchange,
                'step': 'feature_selection'
            }
            
            # Run pipeline up to feature selection
            result = await self.pipeline.process(data, timeframe=timeframe, pipeline_state=pipeline_state)
            
            # Extract feature selection results
            selection_result = {
                'success': result.success,
                'selected_features': getattr(result, 'selected_features', pd.DataFrame()),
                'selection_metadata': getattr(result, 'selection_metadata', {}),
                'selection_metrics': getattr(result, 'selection_metrics', {}),
                'artifacts': result.artifacts or {},
                'error_message': result.error_message if not result.success else None
            }
            
            # Generate human-readable report
            await self._generate_feature_selection_report(selection_result, data)
            
            return selection_result
            
        except Exception as e:
            self.logger.error(f"Feature selection step failed: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'artifacts': {},
                'selected_features': pd.DataFrame(),
                'selection_metadata': {},
                'selection_metrics': {}
            }
    
    async def run_period_optimization_step(self,
                                         data: pd.DataFrame,
                                         symbol: str = "ETHUSDT",
                                         timeframe: str = "15m",
                                         direction: str = "longs",
                                         intensity: str = "blank",
                                         lookback_days: Optional[int] = None,
                                         start_date: Optional[str] = None,
                                         end_date: Optional[str] = None,
                                         exchange: str = "binance",
                                         custom_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run pipeline up to period optimization step."""
        try:
            # Configure pipeline based on intensity
            config = self._create_config_from_intensity(intensity, custom_overrides)
            self.pipeline = create_unified_pipeline(config)
            
            # Create pipeline state
            pipeline_state = {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'exchange': exchange,
                'step': 'period_optimization'
            }
            
            # Run pipeline up to period optimization
            result = await self.pipeline.process(data, timeframe=timeframe, pipeline_state=pipeline_state)
            
            # Extract period optimization results
            optimization_result = {
                'success': result.success,
                'optimal_periods': getattr(result, 'optimal_periods', {}),
                'optimization_metrics': getattr(result, 'optimization_metrics', {}),
                'artifacts': result.artifacts or {},
                'error_message': result.error_message if not result.success else None
            }
            
            # Generate human-readable report
            await self._generate_period_optimization_report(optimization_result, data)
            
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"Period optimization step failed: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'artifacts': {},
                'optimal_periods': {},
                'optimization_metrics': {}
            }
    
    async def run_lookback_optimization_step(self,
                                           data: pd.DataFrame,
                                           symbol: str = "ETHUSDT",
                                           timeframe: str = "15m",
                                           direction: str = "longs",
                                           intensity: str = "blank",
                                           lookback_days: Optional[int] = None,
                                           start_date: Optional[str] = None,
                                           end_date: Optional[str] = None,
                                           exchange: str = "binance",
                                           custom_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run pipeline up to lookback optimization step."""
        try:
            # Configure pipeline based on intensity
            config = self._create_config_from_intensity(intensity, custom_overrides)
            self.pipeline = create_unified_pipeline(config)
            
            # Create pipeline state
            pipeline_state = {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'exchange': exchange,
                'step': 'lookback_optimization'
            }
            
            # Run pipeline up to lookback optimization
            result = await self.pipeline.process(data, timeframe=timeframe, pipeline_state=pipeline_state)
            
            # Extract lookback optimization results
            optimization_result = {
                'success': result.success,
                'optimal_lookbacks': getattr(result, 'optimal_lookbacks', {}),
                'optimization_metrics': getattr(result, 'optimization_metrics', {}),
                'artifacts': result.artifacts or {},
                'error_message': result.error_message if not result.success else None
            }
            
            # Generate human-readable report
            await self._generate_lookback_optimization_report(optimization_result, data)
            
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"Lookback optimization step failed: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'artifacts': {},
                'optimal_lookbacks': {},
                'optimization_metrics': {}
            }
    
    async def run_interaction_generation_step(self,
                                            data: pd.DataFrame,
                                            symbol: str = "ETHUSDT",
                                            timeframe: str = "15m",
                                            direction: str = "longs",
                                            intensity: str = "blank",
                                            lookback_days: Optional[int] = None,
                                            start_date: Optional[str] = None,
                                            end_date: Optional[str] = None,
                                            exchange: str = "binance",
                                            custom_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run pipeline up to interaction generation step."""
        try:
            # Configure pipeline based on intensity
            config = self._create_config_from_intensity(intensity, custom_overrides)
            self.pipeline = create_unified_pipeline(config)
            
            # Create pipeline state
            pipeline_state = {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'exchange': exchange,
                'step': 'interaction_generation'
            }
            
            # Run pipeline up to interaction generation
            result = await self.pipeline.process(data, timeframe=timeframe, pipeline_state=pipeline_state)
            
            # Extract interaction generation results
            interaction_result = {
                'success': result.success,
                'interaction_features': getattr(result, 'interaction_features', pd.DataFrame()),
                'interaction_metadata': getattr(result, 'interaction_metadata', {}),
                'generation_metrics': getattr(result, 'generation_metrics', {}),
                'artifacts': result.artifacts or {},
                'error_message': result.error_message if not result.success else None
            }
            
            # Generate human-readable report
            await self._generate_interaction_generation_report(interaction_result, data)
            
            return interaction_result
            
        except Exception as e:
            self.logger.error(f"Interaction generation step failed: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'artifacts': {},
                'interaction_features': pd.DataFrame(),
                'interaction_metadata': {},
                'generation_metrics': {}
            }
    
    async def run_vectorization_step(self,
                                   data: pd.DataFrame,
                                   symbol: str = "ETHUSDT",
                                   timeframe: str = "15m",
                                   direction: str = "longs",
                                   intensity: str = "blank",
                                   lookback_days: Optional[int] = None,
                                   start_date: Optional[str] = None,
                                   end_date: Optional[str] = None,
                                   exchange: str = "binance",
                                   custom_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run pipeline up to vectorization step."""
        try:
            # Configure pipeline based on intensity
            config = self._create_config_from_intensity(intensity, custom_overrides)
            self.pipeline = create_unified_pipeline(config)
            
            # Create pipeline state
            pipeline_state = {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'exchange': exchange,
                'step': 'vectorization'
            }
            
            # Run pipeline up to vectorization
            result = await self.pipeline.process(data, timeframe=timeframe, pipeline_state=pipeline_state)
            
            # Extract vectorization results
            vectorization_result = {
                'success': result.success,
                'vectorized_features': getattr(result, 'vectorized_features', pd.DataFrame()),
                'vectorization_metadata': getattr(result, 'vectorization_metadata', {}),
                'performance_metrics': getattr(result, 'performance_metrics', {}),
                'artifacts': result.artifacts or {},
                'error_message': result.error_message if not result.success else None
            }
            
            # Generate human-readable report
            await self._generate_vectorization_report(vectorization_result, data)
            
            return vectorization_result
            
        except Exception as e:
            self.logger.error(f"Vectorization step failed: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'artifacts': {},
                'vectorized_features': pd.DataFrame(),
                'vectorization_metadata': {},
                'performance_metrics': {}
            }
    
    async def run_labeling_integration_step(self,
                                          data: pd.DataFrame,
                                          symbol: str = "ETHUSDT",
                                          timeframe: str = "15m",
                                          direction: str = "longs",
                                          intensity: str = "blank",
                                          lookback_days: Optional[int] = None,
                                          start_date: Optional[str] = None,
                                          end_date: Optional[str] = None,
                                          exchange: str = "binance",
                                          custom_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run pipeline up to labeling integration step."""
        try:
            # Configure pipeline based on intensity
            config = self._create_config_from_intensity(intensity, custom_overrides)
            self.pipeline = create_unified_pipeline(config)
            
            # Create pipeline state
            pipeline_state = {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'exchange': exchange,
                'step': 'labeling_integration'
            }
            
            # Run pipeline up to labeling integration
            result = await self.pipeline.process(data, timeframe=timeframe, pipeline_state=pipeline_state)
            
            # Extract labeling integration results
            labeling_result = {
                'success': result.success,
                'labeled_data': getattr(result, 'labeled_data', pd.DataFrame()),
                'labeling_metadata': getattr(result, 'labeling_metadata', {}),
                'quality_metrics': getattr(result, 'quality_metrics', {}),
                'artifacts': result.artifacts or {},
                'error_message': result.error_message if not result.success else None
            }
            
            # Generate human-readable report
            await self._generate_labeling_integration_report(labeling_result, data)
            
            return labeling_result
            
        except Exception as e:
            self.logger.error(f"Labeling integration step failed: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'artifacts': {},
                'labeled_data': pd.DataFrame(),
                'labeling_metadata': {},
                'quality_metrics': {}
            }
    
    async def run_final_validation_step(self,
                                      data: pd.DataFrame,
                                      symbol: str = "ETHUSDT",
                                      timeframe: str = "15m",
                                      direction: str = "longs",
                                      intensity: str = "blank",
                                      lookback_days: Optional[int] = None,
                                      start_date: Optional[str] = None,
                                      end_date: Optional[str] = None,
                                      exchange: str = "binance",
                                      custom_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run pipeline up to final validation step."""
        try:
            # Configure pipeline based on intensity
            config = self._create_config_from_intensity(intensity, custom_overrides)
            self.pipeline = create_unified_pipeline(config)
            
            # Create pipeline state
            pipeline_state = {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'exchange': exchange,
                'step': 'final_validation'
            }
            
            # Run pipeline up to final validation
            result = await self.pipeline.process(data, timeframe=timeframe, pipeline_state=pipeline_state)
            
            # Extract final validation results
            validation_result = {
                'success': result.success,
                'final_dataset': getattr(result, 'final_dataset', pd.DataFrame()),
                'validation_summary': getattr(result, 'validation_summary', {}),
                'quality_metrics': getattr(result, 'quality_metrics', {}),
                'pipeline_summary': getattr(result, 'pipeline_summary', {}),
                'artifacts': result.artifacts or {},
                'error_message': result.error_message if not result.success else None
            }
            
            # Generate human-readable report
            await self._generate_final_validation_report(validation_result, data)
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Final validation step failed: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'artifacts': {},
                'final_dataset': pd.DataFrame(),
                'validation_summary': {},
                'quality_metrics': {},
                'pipeline_summary': {}
            }
    
    def _create_config_from_intensity(self, intensity: str, custom_overrides: Optional[Dict[str, Any]] = None) -> UnifiedPipelineConfig:
        """Create configuration based on intensity."""
        if intensity == "full":
            config = create_full_config()
        elif intensity == "blank":
            config = create_blank_config()
        elif intensity == "light":
            config = create_light_config()
        else:
            config = create_config_by_intensity(PipelineIntensity.BLANK)
        
        # Apply custom overrides if provided
        if custom_overrides:
            for key, value in custom_overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        return config
    
    async def _generate_data_validation_report(self, result: Dict[str, Any], data: pd.DataFrame) -> None:
        """Generate human-readable report for data validation step."""
        # Create outcomes directory
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(exist_ok=True)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        report_filename = f"data_validation_report_{timestamp}.md"
        report_path = outcomes_dir / report_filename
        
        # Generate report content
        report_content = f"""# Data Validation Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
- **Status**: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}
- **Data Shape**: {data.shape[0]} rows × {data.shape[1]} columns
- **Quality Score**: {result.get('data_quality_score', 0.0):.3f}

## Validation Results
- **Success**: {result['success']}
- **Error Message**: {result.get('error_message', 'None')}
- **Artifacts Generated**: {len(result.get('artifacts', {}))}

## Next Steps
1. Review validation results
2. Address any issues if present
3. Proceed to feature generation step

---
*Report generated by Consolidated Pipeline Runner*
"""
        
        # Write report
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        # Add report to artifacts
        result['artifacts']['human_readable_report'] = str(report_path)
        
        self.logger.info(f"📊 Human-readable report saved: {report_path}")
    
    async def _generate_feature_generation_report(self, result: Dict[str, Any], data: pd.DataFrame) -> None:
        """Generate human-readable report for feature generation step."""
        # Create outcomes directory
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(exist_ok=True)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        report_filename = f"feature_generation_report_{timestamp}.md"
        report_path = outcomes_dir / report_filename
        
        # Generate report content
        report_content = f"""# Feature Generation Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
- **Status**: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}
- **Data Shape**: {data.shape[0]} rows × {data.shape[1]} columns
- **Generated Features**: {len(result.get('generated_features', pd.DataFrame()).columns)}

## Generation Results
- **Success**: {result['success']}
- **Error Message**: {result.get('error_message', 'None')}
- **Artifacts Generated**: {len(result.get('artifacts', {}))}

## Next Steps
1. Review generated features
2. Proceed to feature selection step

---
*Report generated by Consolidated Pipeline Runner*
"""
        
        # Write report
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        # Add report to artifacts
        result['artifacts']['human_readable_report'] = str(report_path)
        
        self.logger.info(f"📊 Human-readable report saved: {report_path}")
    
    async def _generate_feature_selection_report(self, result: Dict[str, Any], data: pd.DataFrame) -> None:
        """Generate human-readable report for feature selection step."""
        # Create outcomes directory
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(exist_ok=True)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        report_filename = f"feature_selection_report_{timestamp}.md"
        report_path = outcomes_dir / report_filename
        
        # Generate report content
        report_content = f"""# Feature Selection Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
- **Status**: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}
- **Data Shape**: {data.shape[0]} rows × {data.shape[1]} columns
- **Selected Features**: {len(result.get('selected_features', pd.DataFrame()).columns)}

## Selection Results
- **Success**: {result['success']}
- **Error Message**: {result.get('error_message', 'None')}
- **Artifacts Generated**: {len(result.get('artifacts', {}))}

## Next Steps
1. Review selected features
2. Proceed to period optimization step

---
*Report generated by Consolidated Pipeline Runner*
"""
        
        # Write report
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        # Add report to artifacts
        result['artifacts']['human_readable_report'] = str(report_path)
        
        self.logger.info(f"📊 Human-readable report saved: {report_path}")
    
    async def _generate_period_optimization_report(self, result: Dict[str, Any], data: pd.DataFrame) -> None:
        """Generate human-readable report for period optimization step."""
        # Create outcomes directory
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(exist_ok=True)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        report_filename = f"period_optimization_report_{timestamp}.md"
        report_path = outcomes_dir / report_filename
        
        # Generate report content
        report_content = f"""# Period Optimization Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
- **Status**: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}
- **Data Shape**: {data.shape[0]} rows × {data.shape[1]} columns
- **Optimized Periods**: {len(result.get('optimal_periods', {}))}

## Optimization Results
- **Success**: {result['success']}
- **Error Message**: {result.get('error_message', 'None')}
- **Artifacts Generated**: {len(result.get('artifacts', {}))}

## Next Steps
1. Review optimized periods
2. Proceed to lookback optimization step

---
*Report generated by Consolidated Pipeline Runner*
"""
        
        # Write report
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        # Add report to artifacts
        result['artifacts']['human_readable_report'] = str(report_path)
        
        self.logger.info(f"📊 Human-readable report saved: {report_path}")
    
    async def _generate_lookback_optimization_report(self, result: Dict[str, Any], data: pd.DataFrame) -> None:
        """Generate human-readable report for lookback optimization step."""
        # Create outcomes directory
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(exist_ok=True)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        report_filename = f"lookback_optimization_report_{timestamp}.md"
        report_path = outcomes_dir / report_filename
        
        # Generate report content
        report_content = f"""# Lookback Optimization Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
- **Status**: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}
- **Data Shape**: {data.shape[0]} rows × {data.shape[1]} columns
- **Optimized Lookbacks**: {len(result.get('optimal_lookbacks', {}))}

## Optimization Results
- **Success**: {result['success']}
- **Error Message**: {result.get('error_message', 'None')}
- **Artifacts Generated**: {len(result.get('artifacts', {}))}

## Next Steps
1. Review optimized lookbacks
2. Proceed to interaction generation step

---
*Report generated by Consolidated Pipeline Runner*
"""
        
        # Write report
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        # Add report to artifacts
        result['artifacts']['human_readable_report'] = str(report_path)
        
        self.logger.info(f"📊 Human-readable report saved: {report_path}")
    
    async def _generate_interaction_generation_report(self, result: Dict[str, Any], data: pd.DataFrame) -> None:
        """Generate human-readable report for interaction generation step."""
        # Create outcomes directory
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(exist_ok=True)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        report_filename = f"interaction_generation_report_{timestamp}.md"
        report_path = outcomes_dir / report_filename
        
        # Generate report content
        report_content = f"""# Interaction Generation Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
- **Status**: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}
- **Data Shape**: {data.shape[0]} rows × {data.shape[1]} columns
- **Interaction Features**: {len(result.get('interaction_features', pd.DataFrame()).columns)}

## Generation Results
- **Success**: {result['success']}
- **Error Message**: {result.get('error_message', 'None')}
- **Artifacts Generated**: {len(result.get('artifacts', {}))}

## Next Steps
1. Review interaction features
2. Proceed to vectorization step

---
*Report generated by Consolidated Pipeline Runner*
"""
        
        # Write report
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        # Add report to artifacts
        result['artifacts']['human_readable_report'] = str(report_path)
        
        self.logger.info(f"📊 Human-readable report saved: {report_path}")
    
    async def _generate_vectorization_report(self, result: Dict[str, Any], data: pd.DataFrame) -> None:
        """Generate human-readable report for vectorization step."""
        # Create outcomes directory
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(exist_ok=True)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        report_filename = f"vectorization_report_{timestamp}.md"
        report_path = outcomes_dir / report_filename
        
        # Generate report content
        report_content = f"""# Vectorization Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
- **Status**: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}
- **Data Shape**: {data.shape[0]} rows × {data.shape[1]} columns
- **Vectorized Features**: {len(result.get('vectorized_features', pd.DataFrame()).columns)}

## Vectorization Results
- **Success**: {result['success']}
- **Error Message**: {result.get('error_message', 'None')}
- **Artifacts Generated**: {len(result.get('artifacts', {}))}

## Next Steps
1. Review vectorized features
2. Proceed to labeling integration step

---
*Report generated by Consolidated Pipeline Runner*
"""
        
        # Write report
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        # Add report to artifacts
        result['artifacts']['human_readable_report'] = str(report_path)
        
        self.logger.info(f"📊 Human-readable report saved: {report_path}")
    
    async def _generate_labeling_integration_report(self, result: Dict[str, Any], data: pd.DataFrame) -> None:
        """Generate human-readable report for labeling integration step."""
        # Create outcomes directory
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(exist_ok=True)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        report_filename = f"labeling_integration_report_{timestamp}.md"
        report_path = outcomes_dir / report_filename
        
        # Generate report content
        report_content = f"""# Labeling Integration Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
- **Status**: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}
- **Data Shape**: {data.shape[0]} rows × {data.shape[1]} columns
- **Labeled Data**: {len(result.get('labeled_data', pd.DataFrame()).columns)}

## Labeling Results
- **Success**: {result['success']}
- **Error Message**: {result.get('error_message', 'None')}
- **Artifacts Generated**: {len(result.get('artifacts', {}))}

## Next Steps
1. Review labeled data
2. Proceed to final validation step

---
*Report generated by Consolidated Pipeline Runner*
"""
        
        # Write report
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        # Add report to artifacts
        result['artifacts']['human_readable_report'] = str(report_path)
        
        self.logger.info(f"📊 Human-readable report saved: {report_path}")
    
    async def _generate_final_validation_report(self, result: Dict[str, Any], data: pd.DataFrame) -> None:
        """Generate human-readable report for final validation step."""
        # Create outcomes directory
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(exist_ok=True)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        report_filename = f"final_validation_report_{timestamp}.md"
        report_path = outcomes_dir / report_filename
        
        # Generate report content
        report_content = f"""# Final Validation Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
- **Status**: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}
- **Data Shape**: {data.shape[0]} rows × {data.shape[1]} columns
- **Final Dataset**: {len(result.get('final_dataset', pd.DataFrame()).columns)}

## Validation Results
- **Success**: {result['success']}
- **Error Message**: {result.get('error_message', 'None')}
- **Artifacts Generated**: {len(result.get('artifacts', {}))}

## Next Steps
1. Review final dataset
2. Use dataset for model training

---
*Report generated by Consolidated Pipeline Runner*
"""
        
        # Write report
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        # Add report to artifacts
        result['artifacts']['human_readable_report'] = str(report_path)
        
        self.logger.info(f"📊 Human-readable report saved: {report_path}")


# Convenience functions for each step
async def run_data_validation_step(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Run data validation step using consolidated pipeline."""
    runner = ConsolidatedPipelineRunner()
    return await runner.run_data_validation_step(data, **kwargs)

async def run_feature_generation_step(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Run feature generation step using consolidated pipeline."""
    runner = ConsolidatedPipelineRunner()
    return await runner.run_feature_generation_step(data, **kwargs)

async def run_feature_selection_step(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Run feature selection step using consolidated pipeline."""
    runner = ConsolidatedPipelineRunner()
    return await runner.run_feature_selection_step(data, **kwargs)

async def run_period_optimization_step(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Run period optimization step using consolidated pipeline."""
    runner = ConsolidatedPipelineRunner()
    return await runner.run_period_optimization_step(data, **kwargs)

async def run_lookback_optimization_step(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Run lookback optimization step using consolidated pipeline."""
    runner = ConsolidatedPipelineRunner()
    return await runner.run_lookback_optimization_step(data, **kwargs)

async def run_interaction_generation_step(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Run interaction generation step using consolidated pipeline."""
    runner = ConsolidatedPipelineRunner()
    return await runner.run_interaction_generation_step(data, **kwargs)

async def run_vectorization_step(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Run vectorization step using consolidated pipeline."""
    runner = ConsolidatedPipelineRunner()
    return await runner.run_vectorization_step(data, **kwargs)

async def run_labeling_integration_step(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Run labeling integration step using consolidated pipeline."""
    runner = ConsolidatedPipelineRunner()
    return await runner.run_labeling_integration_step(data, **kwargs)

async def run_final_validation_step(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Run final validation step using consolidated pipeline."""
    runner = ConsolidatedPipelineRunner()
    return await runner.run_final_validation_step(data, **kwargs)