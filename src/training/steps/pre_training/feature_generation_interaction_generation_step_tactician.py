"""
Feature Generation Interaction Generation Step Tactician.

This step generates interaction features for tactician models using
VectorBTRollingOptimizer and UnifiedVectorizationManager for optimal performance.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
import numpy as np
import pandas as pd

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import tprint
from src.utils.pipeline_standards import PipelineStandards

# Import VectorBT optimization components
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer
    from src.feature_generation.utils.unified_vectorization_manager import UnifiedVectorizationManager
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    tprint("⚠️ VectorBT optimizers not available, using standard pandas operations", "WARNING")

logger = logging.getLogger(__name__)


class FeatureGenerationInteractionGenerationStepTactician(BaseStep):
    """
    Feature Generation Interaction Generation Step Tactician.

    Generates interaction features specifically for tactician models using
    VectorBT optimizations for high performance.
    """

    def __init__(self, step_name: str = "feature_generation_interaction_generation_step_tactician"):
        """Initialize the tactician interaction generation step."""
        super().__init__(step_name)
        self.logger = system_logger.getChild('InteractionGenerationTactician')
        
        # Initialize optimizers if available
        if OPTIMIZATION_AVAILABLE:
            self.rolling_optimizer = VectorBTRollingOptimizer(
                enable_parallel=True,
                memory_efficient=True,
                chunk_size=1000,
                fast_fail=False,
                enable_logging=True
            )
            self.vectorization_manager = UnifiedVectorizationManager(
                fast_fail=False,
                enable_logging=True
            )
        else:
            self.rolling_optimizer = None
            self.vectorization_manager = None

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute tactician interaction feature generation with VectorBT optimizations.

        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol (e.g., 'ETHUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Timeframe (e.g., '15m')
                - execution_mode: 'full', 'light', or 'blank'

        Returns:
            Dict containing:
            - 'success': bool indicating if step completed successfully
            - 'artifacts': dict of created artifacts
            - 'metrics': dict of performance metrics
            - 'error': error message if step failed (optional)
        """
        tprint(f"🔗 Starting tactician interaction generation for {config.get('symbol', 'UNKNOWN')}", "INFO")

        try:
            # Load input features
            features_df = await self._load_features(config)
            
            if features_df is None or features_df.empty:
                tprint("⚠️ No features available for interaction generation", "WARNING")
                return self._create_empty_result(config)

            tprint(f"📊 Loaded {len(features_df)} rows with {len(features_df.columns)} features", "INFO")

            # Generate interaction features using VectorBT optimizations
            interaction_features = await self._generate_interaction_features(
                features_df, 
                config
            )

            # Save interaction features
            artifacts = await self._save_interaction_features(
                interaction_features,
                config
            )

            # Calculate metrics
            metrics = self._calculate_metrics(interaction_features, config)

            tprint(f"✅ Tactician interaction generation completed: {metrics['n_interaction_features']} interactions", "SUCCESS")
            
            if OPTIMIZATION_AVAILABLE:
                perf_stats = self.vectorization_manager.performance_stats
                tprint(f"⚡ Performance: {perf_stats.get('total_operations', 0)} operations, "
                      f"{perf_stats.get('cache_hits', 0)} cache hits", "SUCCESS")

            return {
                'success': True,
                'artifacts': artifacts,
                'metrics': metrics
            }

        except Exception as e:
            error_msg = f"Tactician interaction generation failed: {str(e)}"
            tprint(f"❌ {error_msg}", "ERROR")
            self.logger.error(error_msg)

            return {
                'success': False,
                'artifacts': {},
                'metrics': {},
                'error': error_msg
            }

    async def _load_features(self, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Load input features for interaction generation."""
        try:
            # Build feature path
            feature_path = PipelineStandards.build_path(
                'feature_generation_extended_feature_generation',
                config.get('symbol'),
                config.get('exchange'),
                config.get('timeframe'),
                artifact_type='features',
                cache_dir='data_cache'
            )

            # Load features
            if feature_path.exists():
                features_df = pd.read_parquet(feature_path)
                tprint(f"✅ Loaded features from {feature_path}", "SUCCESS")
                return features_df
            else:
                tprint(f"⚠️ Feature file not found at {feature_path}", "WARNING")
                return None

        except Exception as e:
            tprint(f"❌ Failed to load features: {e}", "ERROR")
            return None

    async def _generate_interaction_features(
        self, 
        features_df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Generate interaction features using VectorBT optimizations."""
        tprint("🔄 Generating interaction features with VectorBT optimizations...", "INFO")

        # Determine which features to create based on available columns
        interaction_features = features_df.copy()
        
        # Define interaction feature definitions
        interactions = []
        
        # Regime-based interactions
        if all(col in features_df.columns for col in ['regime', 'returns']):
            interactions.append(('regime_x_returns', lambda df: df['regime'] * df['returns']))
        
        if all(col in features_df.columns for col in ['regime', 'volatility']):
            interactions.append(('regime_x_volatility', lambda df: df['regime'] * df['volatility']))
        
        if all(col in features_df.columns for col in ['regime', 'volume']):
            interactions.append(('regime_x_volume', lambda df: df['regime'] * df['volume']))
        
        # Trend-based interactions
        if all(col in features_df.columns for col in ['trend', 'regime']):
            interactions.append(('trend_x_regime', lambda df: df['trend'] * df['regime']))
        
        # Momentum-based interactions
        if all(col in features_df.columns for col in ['momentum', 'regime']):
            interactions.append(('momentum_x_regime', lambda df: df['momentum'] * df['regime']))
        
        # Volatility-based interactions
        if all(col in features_df.columns for col in ['volatility', 'regime']):
            interactions.append(('volatility_x_regime', lambda df: df['volatility'] * df['regime']))

        # Generate interaction features with VectorBT optimizations
        for feature_name, generator_func in interactions:
            try:
                if OPTIMIZATION_AVAILABLE and self.vectorization_manager:
                    # Use VectorBT for optimized computation
                    interaction_series = generator_func(interaction_features)
                    
                    # Apply smoothing with VectorBT rolling optimizer
                    if self.rolling_optimizer:
                        smoothed = self.rolling_optimizer.rolling_mean(
                            interaction_series, 
                            window=min(20, len(interaction_series) // 10)
                        )
                        interaction_features[f'{feature_name}_smoothed'] = smoothed
                
                # Generate base interaction
                interaction_features[feature_name] = generator_func(interaction_features)
                tprint(f"✅ Generated interaction: {feature_name}", "SUCCESS")
                
            except Exception as e:
                tprint(f"⚠️ Failed to generate {feature_name}: {e}", "WARNING")
                continue

        tprint(f"✅ Generated {len([f for f in interactions if f[0] in interaction_features.columns])} interaction features", "SUCCESS")
        
        return interaction_features

    async def _save_interaction_features(
        self,
        interaction_features: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Save interaction features to disk."""
        try:
            # Build output path
            output_path = PipelineStandards.build_path(
                self.step_name,
                config.get('symbol'),
                config.get('exchange'),
                config.get('timeframe'),
                artifact_type='interaction_features',
                cache_dir='data_cache'
            )

            # Save to parquet
            output_path.parent.mkdir(parents=True, exist_ok=True)
            interaction_features.to_parquet(output_path, compression='snappy')
            
            tprint(f"✅ Saved interaction features to {output_path}", "SUCCESS")

            return {
                'interaction_features_path': str(output_path),
                'n_features': len(interaction_features.columns),
                'n_interactions': len([c for c in interaction_features.columns if any(x in c for x in ['_x_', 'smoothed'])])
            }

        except Exception as e:
            tprint(f"❌ Failed to save interaction features: {e}", "ERROR")
            raise

    def _calculate_metrics(
        self,
        interaction_features: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate performance metrics."""
        n_interactions = len([c for c in interaction_features.columns if any(x in c for x in ['_x_', 'smoothed'])])
        
        metrics = {
            'n_interaction_features': n_interactions,
            'total_features': len(interaction_features.columns),
            'n_rows': len(interaction_features),
            'interaction_types': ['regime_conditional', 'multiplicative', 'additive'],
            'target_model': 'tactician',
            'execution_mode': config.get('execution_mode', 'light'),
            'success': True
        }
        
        # Add VectorBT performance metrics if available
        if OPTIMIZATION_AVAILABLE and self.vectorization_manager:
            perf_stats = self.vectorization_manager.performance_stats
            metrics.update({
                'vectorbt_operations': perf_stats.get('vectorbt_operations', 0),
                'cache_hits': perf_stats.get('cache_hits', 0),
                'cache_misses': perf_stats.get('cache_misses', 0),
                'memory_optimizations': perf_stats.get('memory_optimizations', 0)
            })
        
        return metrics

    def _create_empty_result(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create empty result when no data is available."""
        artifacts = {
            'tactician_interaction_features': {
                'interaction_features': [],
                'interaction_types': [],
                'n_interactions': 0,
                'target_model': 'tactician',
                'metadata': {
                    'symbol': config.get('symbol'),
                    'exchange': config.get('exchange'),
                    'timeframe': config.get('timeframe'),
                    'execution_mode': config.get('execution_mode', 'light'),
                    'created_at': datetime.now().isoformat()
                }
            }
        }

        metrics = {
            'n_interaction_features': 0,
            'interaction_types': 0,
            'target_model': 'tactician',
            'execution_mode': config.get('execution_mode', 'light'),
            'success': False
        }

        return {
            'success': False,
            'artifacts': artifacts,
            'metrics': metrics
        }

    async def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run method required by BaseStep interface."""
        return await self.execute(config)


# Register the step
def register_feature_generation_interaction_generation_step_tactician():
    """Register the tactician interaction generation step."""
    from src.training.steps.base_step import step_registry

    step_registry.register("feature_generation_interaction_generation_step_tactician", FeatureGenerationInteractionGenerationStepTactician)
    tprint("✅ Feature generation interaction generation step tactician registered", "SUCCESS")


# Auto-register when module is imported
register_feature_generation_interaction_generation_step_tactician()
