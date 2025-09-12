"""
Step 06: SR Feature Integration

This step integrates SR-specific features into the existing feature engineering pipeline,
adding only SR proximity and strength features to avoid redundancy with existing features.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import time

from src.utils.logger import system_logger
from src.utils.ml_common.sr_feature_integration import SRFeatureIntegration
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler

logger = system_logger.getChild('SRFeatureIntegration')

class SRFeatureIntegrationStep:
    """Step for integrating SR features into feature engineering pipeline."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize SR feature integration step."""
        self.config = config or {}
        self.logger = logger.getChild('SRFeatureIntegrationStep')
        
        # Initialize SR feature integration
        self.sr_integration = SRFeatureIntegration(self.config)
        
        # Feature configuration
        self.feature_config = self.config.get('sr_features', {
            'enabled': True,
            'proximity_threshold': 0.05,
            'strength_weights': {
                'touch_count': 0.4,
                'volume_confirmation': 0.3,
                'time_decay': 0.2,
                'confluence': 0.1
            }
        })
    
    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute SR feature integration step."""
        self.logger.info('🎯 Starting SR Feature Integration Step')
        start_time = time.time()
        
        try:
            # Check if SR feature integration is enabled
            if not self.feature_config.get('enabled', True):
                self.logger.info('SR feature integration disabled, skipping')
                return {
                    'success': True,
                    'sr_features_added': 0,
                    'execution_time': time.time() - start_time,
                    'stage': 'sr_feature_integration'
                }
            
            # Get existing features from pipeline state
            existing_features = pipeline_state.get('features', {})
            if not existing_features:
                self.logger.warning('No existing features found in pipeline state')
                existing_features = {}
            
            # Check for SR data availability
            sr_levels = pipeline_state.get('sr_levels', [])
            if not sr_levels:
                self.logger.warning('No SR levels found in pipeline state')
                return {
                    'success': True,
                    'sr_features_added': 0,
                    'execution_time': time.time() - start_time,
                    'stage': 'sr_feature_integration',
                    'warning': 'No SR data available'
                }
            
            self.logger.info(f'📊 Processing {len(sr_levels)} SR levels for feature integration')
            
            # Integrate SR features into existing feature set
            enhanced_features = self.sr_integration.integrate_sr_features_into_pipeline(
                existing_features=existing_features,
                pipeline_state=pipeline_state
            )
            
            # Calculate integration metrics
            original_feature_count = len(existing_features)
            enhanced_feature_count = len(enhanced_features)
            sr_features_added = enhanced_feature_count - original_feature_count
            
            # Update pipeline state
            pipeline_state['features'] = enhanced_features
            pipeline_state['sr_features_integrated'] = True
            pipeline_state['sr_feature_count'] = sr_features_added
            
            # Get SR feature names for documentation
            sr_feature_names = self.sr_integration.get_sr_feature_names()
            
            # Log results
            self.logger.info(f'✅ SR Feature Integration completed')
            self.logger.info(f'   📊 Original features: {original_feature_count}')
            self.logger.info(f'   📊 Enhanced features: {enhanced_feature_count}')
            self.logger.info(f'   📊 SR features added: {sr_features_added}')
            self.logger.info(f'   📋 SR feature names: {sr_feature_names}')
            
            execution_time = time.time() - start_time
            
            return {
                'success': True,
                'sr_features_added': sr_features_added,
                'original_feature_count': original_feature_count,
                'enhanced_feature_count': enhanced_feature_count,
                'sr_feature_names': sr_feature_names,
                'execution_time': execution_time,
                'stage': 'sr_feature_integration'
            }
            
        except Exception as e:
            self.logger.error(f'❌ SR Feature Integration failed: {e}')
            import traceback
            self.logger.error(f'Traceback: {traceback.format_exc()}')
            
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time,
                'stage': 'sr_feature_integration'
            }
    
    def validate_config(self) -> bool:
        """Validate configuration for SR feature integration."""
        try:
            # Validate SR feature configuration
            sr_config = self.config.get('sr_features', {})
            
            # Check proximity threshold
            proximity_threshold = sr_config.get('proximity_threshold', 0.05)
            if not isinstance(proximity_threshold, (int, float)) or proximity_threshold <= 0:
                self.logger.error('Invalid proximity_threshold in sr_features config')
                return False
            
            # Check strength weights
            strength_weights = sr_config.get('strength_weights', {})
            if strength_weights:
                total_weight = sum(strength_weights.values())
                if abs(total_weight - 1.0) > 0.01:  # Allow small floating point errors
                    self.logger.warning(f'Strength weights sum to {total_weight}, expected 1.0')
            
            return True
            
        except Exception as e:
            self.logger.error(f'Configuration validation failed: {e}')
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of SR feature integration."""
        return {
            'sr_feature_integration_enabled': self.feature_config.get('enabled', True),
            'proximity_threshold': self.feature_config.get('proximity_threshold', 0.05),
            'strength_weights': self.feature_config.get('strength_weights', {}),
            'sr_feature_names': self.sr_integration.get_sr_feature_names(),
            'total_sr_features': len(self.sr_integration.get_sr_feature_names())
        }