"""Regime Continuity Manager for Per-HMM Regime Pipeline.

This module ensures that regime information flows consistently through all pipeline steps,
maintaining regime context and metadata throughout the entire training process.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

from src.utils.logger import getChild as get_logger
from src.utils.common_operations import ensure_directory, safe_json_dump, safe_json_load
from src.utils.pipeline_standards import pipeline_standards
from src.core.decorators import traced, cached, validates, handles_errors, log_execution_time


logger = get_logger('RegimeContinuityManager')


class RegimeStatus(Enum):
    """Status of regime processing in each step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class RegimeMetadata:
    """Metadata for a specific regime."""
    regime_id: int
    regime_name: str
    market_characteristics: Dict[str, Any]
    data_points: int
    time_range: Tuple[datetime, datetime]
    created_at: datetime
    updated_at: datetime
    status: RegimeStatus = RegimeStatus.PENDING
    step_status: Dict[str, RegimeStatus] = None
    
    def __post_init__(self):
        if self.step_status is None:
            self.step_status = {}


@dataclass
class StepRegimeContext:
    """Context for regime processing in a specific step."""
    step_name: str
    regime_id: int
    input_data_path: str
    output_data_path: str
    configuration: Dict[str, Any]
    processing_start: datetime
    processing_end: Optional[datetime] = None
    status: RegimeStatus = RegimeStatus.PENDING
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RegimeContinuityManager:
    """Manages regime continuity throughout the entire pipeline."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the regime continuity manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = get_logger('RegimeContinuityManager')
        self.standards = pipeline_standards
        
        # Regime tracking
        self.regime_metadata: Dict[int, RegimeMetadata] = {}
        self.step_contexts: Dict[str, Dict[int, StepRegimeContext]] = {}
        self.continuity_tracking: Dict[str, Any] = {}
        
        # Pipeline steps that should maintain regime continuity
        self.regime_aware_steps = [
            'step05_labeling',
            'step06_feature_engineering', 
            'step07_enhanced_matrix_operations',
            'step08_advanced_feature_selection',
            'step09_hmm_based_training',
            'step10_unified_regime_intelligence',
            'step11_analyst_creation',
            'step12_analyst_enhancement',
            'step13_analyst_ensemble_creation',
            'step14_tactician_labeling',
            'step15_tactician_specialist_training',
            'step16_confidence_calibration',
            'step17_final_parameters_optimization',
            'step18_walk_forward_validation',
            'step19_monte_carlo_validation',
            'step20_ab_testing',
            'step21_saving'
        ]
        
    @traced(span_name='initialize_regime_continuity')
    async def initialize_regime_continuity(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str
    ) -> bool:
        """Initialize regime continuity tracking for a training session.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            
        Returns:
            True if initialization successful
        """
        try:
            self.logger.info(f"🚀 Initializing regime continuity for {exchange}_{symbol}_{timeframe}")
            
            # Load regime data from step 4
            regime_data = await self._load_regime_data(symbol, exchange, timeframe, data_dir)
            if regime_data is None:
                self.logger.error("❌ Failed to load regime data")
                return False
            
            # Extract regime metadata
            await self._extract_regime_metadata(regime_data, symbol, exchange, timeframe)
            
            # Initialize step contexts
            await self._initialize_step_contexts(symbol, exchange, timeframe, data_dir)
            
            # Save continuity tracking
            await self._save_continuity_tracking(symbol, exchange, timeframe, data_dir)
            
            self.logger.info(f"✅ Regime continuity initialized for {len(self.regime_metadata)} regimes")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Error initializing regime continuity: {e}")
            return False
    
    @traced(span_name='load_regime_data')
    async def _load_regime_data(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str
    ) -> Optional[pd.DataFrame]:
        """Load regime data from step 4.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            
        Returns:
            Regime DataFrame or None
        """
        try:
            training_dir = Path(data_dir) / 'training'
            regime_file = training_dir / f'{exchange}_{symbol}_{timeframe}_unified_regime_data.parquet'
            
            if not regime_file.exists():
                self.logger.error(f"❌ Regime data not found: {regime_file}")
                return None
                
            data = pd.read_parquet(regime_file)
            self.logger.info(f"✅ Loaded regime data: {len(data)} rows")
            return data
            
        except Exception as e:
            self.logger.exception(f"❌ Error loading regime data: {e}")
            return None
    
    @traced(span_name='extract_regime_metadata')
    async def _extract_regime_metadata(
        self,
        regime_data: pd.DataFrame,
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> None:
        """Extract metadata for each regime.
        
        Args:
            regime_data: Regime DataFrame
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
        """
        try:
            if 'composite_cluster_id' not in regime_data.columns:
                self.logger.error("❌ No composite_cluster_id column found")
                return
            
            regime_ids = sorted(regime_data['composite_cluster_id'].unique())
            
            for regime_id in regime_ids:
                regime_mask = regime_data['composite_cluster_id'] == regime_id
                regime_subset = regime_data[regime_mask]
                
                # Extract regime characteristics
                characteristics = self._analyze_regime_characteristics(regime_subset)
                
                # Create regime metadata
                metadata = RegimeMetadata(
                    regime_id=regime_id,
                    regime_name=f"Regime_{regime_id}",
                    market_characteristics=characteristics,
                    data_points=len(regime_subset),
                    time_range=(
                        regime_subset['timestamp'].min(),
                        regime_subset['timestamp'].max()
                    ),
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                
                self.regime_metadata[regime_id] = metadata
                self.logger.info(f"📊 Extracted metadata for regime {regime_id}: {len(regime_subset)} points")
                
        except Exception as e:
            self.logger.exception(f"❌ Error extracting regime metadata: {e}")
    
    def _analyze_regime_characteristics(self, regime_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze characteristics of a specific regime.
        
        Args:
            regime_data: Data for a specific regime
            
        Returns:
            Dictionary of regime characteristics
        """
        try:
            characteristics = {
                'volatility': float(regime_data['close'].std()) if 'close' in regime_data.columns else 0.0,
                'mean_return': float(regime_data['close'].pct_change().mean()) if 'close' in regime_data.columns else 0.0,
                'volume_profile': float(regime_data['volume'].mean()) if 'volume' in regime_data.columns else 0.0,
                'price_range': {
                    'min': float(regime_data['close'].min()) if 'close' in regime_data.columns else 0.0,
                    'max': float(regime_data['close'].max()) if 'close' in regime_data.columns else 0.0
                }
            }
            
            # Add trend analysis if possible
            if 'close' in regime_data.columns and len(regime_data) > 1:
                price_changes = regime_data['close'].pct_change().dropna()
                characteristics['trend_strength'] = float(abs(price_changes.mean()) / price_changes.std()) if price_changes.std() > 0 else 0.0
                characteristics['trend_direction'] = 'up' if price_changes.mean() > 0 else 'down'
            
            return characteristics
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing regime characteristics: {e}")
            return {}
    
    @traced(span_name='initialize_step_contexts')
    async def _initialize_step_contexts(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str
    ) -> None:
        """Initialize step contexts for each regime.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
        """
        try:
            for step_name in self.regime_aware_steps:
                self.step_contexts[step_name] = {}
                
                for regime_id in self.regime_metadata.keys():
                    context = StepRegimeContext(
                        step_name=step_name,
                        regime_id=regime_id,
                        input_data_path=f"{data_dir}/training/{exchange}_{symbol}_{timeframe}_step_{step_name}_input_regime_{regime_id}.parquet",
                        output_data_path=f"{data_dir}/training/{exchange}_{symbol}_{timeframe}_step_{step_name}_output_regime_{regime_id}.parquet",
                        configuration=self._get_step_configuration(step_name, regime_id),
                        processing_start=datetime.now()
                    )
                    
                    self.step_contexts[step_name][regime_id] = context
                    
            self.logger.info(f"✅ Initialized step contexts for {len(self.regime_aware_steps)} steps")
            
        except Exception as e:
            self.logger.exception(f"❌ Error initializing step contexts: {e}")
    
    def _get_step_configuration(self, step_name: str, regime_id: int) -> Dict[str, Any]:
        """Get configuration for a specific step and regime.
        
        Args:
            step_name: Name of the step
            regime_id: Regime ID
            
        Returns:
            Configuration dictionary
        """
        # Base configuration
        config = {
            'per_regime_processing': True,
            'regime_id': regime_id,
            'regime_aware': True
        }
        
        # Add regime-specific configuration based on regime characteristics
        if regime_id in self.regime_metadata:
            regime_meta = self.regime_metadata[regime_id]
            characteristics = regime_meta.market_characteristics
            
            # Adapt configuration based on regime characteristics
            if characteristics.get('volatility', 0) > 0.02:  # High volatility
                config['volatility_handling'] = 'high'
                config['lookback_periods'] = [5, 10, 20, 30]
            elif characteristics.get('volatility', 0) < 0.01:  # Low volatility
                config['volatility_handling'] = 'low'
                config['lookback_periods'] = [20, 50, 100, 200]
            else:  # Medium volatility
                config['volatility_handling'] = 'medium'
                config['lookback_periods'] = [10, 20, 50, 100]
        
        return config
    
    @traced(span_name='update_step_status')
    async def update_step_status(
        self,
        step_name: str,
        regime_id: int,
        status: RegimeStatus,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update the status of a step for a specific regime.
        
        Args:
            step_name: Name of the step
            regime_id: Regime ID
            status: New status
            error_message: Error message if failed
            metadata: Additional metadata
        """
        try:
            if step_name in self.step_contexts and regime_id in self.step_contexts[step_name]:
                context = self.step_contexts[step_name][regime_id]
                context.status = status
                context.processing_end = datetime.now()
                
                if error_message:
                    context.error_message = error_message
                
                if metadata:
                    context.metadata.update(metadata)
                
                # Update regime metadata
                if regime_id in self.regime_metadata:
                    self.regime_metadata[regime_id].step_status[step_name] = status
                    self.regime_metadata[regime_id].updated_at = datetime.now()
                
                self.logger.info(f"📊 Updated {step_name} status for regime {regime_id}: {status.value}")
                
        except Exception as e:
            self.logger.error(f"❌ Error updating step status: {e}")
    
    @traced(span_name='get_regime_context')
    async def get_regime_context(
        self,
        step_name: str,
        regime_id: int
    ) -> Optional[StepRegimeContext]:
        """Get the context for a specific step and regime.
        
        Args:
            step_name: Name of the step
            regime_id: Regime ID
            
        Returns:
            Step context or None
        """
        try:
            if step_name in self.step_contexts and regime_id in self.step_contexts[step_name]:
                return self.step_contexts[step_name][regime_id]
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error getting regime context: {e}")
            return None
    
    @traced(span_name='validate_regime_continuity')
    async def validate_regime_continuity(
        self,
        step_name: str,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str
    ) -> bool:
        """Validate that regime continuity is maintained for a step.
        
        Args:
            step_name: Name of the step
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            
        Returns:
            True if continuity is maintained
        """
        try:
            self.logger.info(f"🔍 Validating regime continuity for {step_name}")
            
            # Check if step should maintain regime continuity
            if step_name not in self.regime_aware_steps:
                self.logger.info(f"ℹ️ {step_name} is not regime-aware, skipping validation")
                return True
            
            # Check if all regimes have been processed
            if step_name not in self.step_contexts:
                self.logger.error(f"❌ No step contexts found for {step_name}")
                return False
            
            step_contexts = self.step_contexts[step_name]
            total_regimes = len(self.regime_metadata)
            processed_regimes = len([ctx for ctx in step_contexts.values() if ctx.status == RegimeStatus.COMPLETED])
            
            if processed_regimes != total_regimes:
                self.logger.warning(f"⚠️ Only {processed_regimes}/{total_regimes} regimes processed for {step_name}")
                return False
            
            # Check for failed regimes
            failed_regimes = [ctx.regime_id for ctx in step_contexts.values() if ctx.status == RegimeStatus.FAILED]
            if failed_regimes:
                self.logger.error(f"❌ Failed regimes in {step_name}: {failed_regimes}")
                return False
            
            self.logger.info(f"✅ Regime continuity validated for {step_name}")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Error validating regime continuity: {e}")
            return False
    
    @traced(span_name='save_continuity_tracking')
    async def _save_continuity_tracking(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str
    ) -> None:
        """Save continuity tracking data.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
        """
        try:
            training_dir = Path(data_dir) / 'training'
            training_dir.mkdir(parents=True, exist_ok=True)
            
            # Save regime metadata
            regime_metadata_file = training_dir / f'{exchange}_{symbol}_{timeframe}_regime_continuity_metadata.json'
            regime_data = {
                regime_id: {
                    **asdict(metadata),
                    'status': metadata.status.value,
                    'step_status': {step: status.value for step, status in metadata.step_status.items()}
                }
                for regime_id, metadata in self.regime_metadata.items()
            }
            
            safe_json_dump(regime_data, regime_metadata_file)
            
            # Save step contexts
            step_contexts_file = training_dir / f'{exchange}_{symbol}_{timeframe}_step_contexts.json'
            contexts_data = {}
            for step_name, contexts in self.step_contexts.items():
                contexts_data[step_name] = {
                    regime_id: {
                        **asdict(context),
                        'status': context.status.value,
                        'processing_start': context.processing_start.isoformat(),
                        'processing_end': context.processing_end.isoformat() if context.processing_end else None
                    }
                    for regime_id, context in contexts.items()
                }
            
            safe_json_dump(contexts_data, step_contexts_file)
            
            self.logger.info(f"✅ Saved continuity tracking data")
            
        except Exception as e:
            self.logger.exception(f"❌ Error saving continuity tracking: {e}")
    
    @traced(span_name='get_continuity_report')
    async def get_continuity_report(
        self,
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> Dict[str, Any]:
        """Generate a continuity report for the pipeline.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            
        Returns:
            Continuity report dictionary
        """
        try:
            report = {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'generated_at': datetime.now().isoformat(),
                'total_regimes': len(self.regime_metadata),
                'regime_summary': {},
                'step_summary': {},
                'continuity_issues': []
            }
            
            # Regime summary
            for regime_id, metadata in self.regime_metadata.items():
                report['regime_summary'][f'regime_{regime_id}'] = {
                    'data_points': metadata.data_points,
                    'status': metadata.status.value,
                    'completed_steps': len([s for s in metadata.step_status.values() if s == RegimeStatus.COMPLETED]),
                    'total_steps': len(self.regime_aware_steps)
                }
            
            # Step summary
            for step_name in self.regime_aware_steps:
                if step_name in self.step_contexts:
                    contexts = self.step_contexts[step_name]
                    completed = len([c for c in contexts.values() if c.status == RegimeStatus.COMPLETED])
                    failed = len([c for c in contexts.values() if c.status == RegimeStatus.FAILED])
                    
                    report['step_summary'][step_name] = {
                        'completed_regimes': completed,
                        'failed_regimes': failed,
                        'total_regimes': len(contexts),
                        'completion_rate': completed / len(contexts) if contexts else 0
                    }
                    
                    if failed > 0:
                        report['continuity_issues'].append(f"{step_name}: {failed} failed regimes")
            
            return report
            
        except Exception as e:
            self.logger.exception(f"❌ Error generating continuity report: {e}")
            return {}


# Global instance
regime_continuity_manager = RegimeContinuityManager()