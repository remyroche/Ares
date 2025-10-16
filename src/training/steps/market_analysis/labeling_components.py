from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
"""Labeling Components for Step05.

This module provides the core labeling logic components including regime-aware labeling,
meta-labeling, and composite labeling strategies.
"""
import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation
from src.utils.tprint import tprint

tprint("🔧 Loading labeling components...")

class RegimeAwareLabeling:
    """Regime-aware triple barrier labeling component."""

    @log_important_calls
    def __init__(self, config: Dict[str, Any], logger: Any = None):
        tprint("🔧 Initializing RegimeAwareLabeling...")
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.regime_barrier_optimizer = None
        self.regime_col = None
        self.time_barrier_minutes = None
        self.max_lookahead = None
        self._initialize_components()
        tprint("✅ RegimeAwareLabeling initialized")

    @log_all_calls
    def _initialize_components(self) -> None:
        """Initialize regime-aware labeling components."""
        try:
            labeling_cfg = self.config.get('vectorized_labelling_orchestrator', {})
            self.time_barrier_minutes = int(labeling_cfg.get('time_barrier_minutes', 30))
            self.max_lookahead = int(labeling_cfg.get('max_lookahead', 100))

            # Detect regime column
            try:
                from src.utils.regime_data_access import get_regime_column
                detected = get_regime_column(pd.DataFrame(columns=['composite_cluster_id'])) or 'hmm_regime'
            except ImportError:
                detected = 'hmm_regime'

            self.regime_col = str(labeling_cfg.get('hmm_barrier_regime_column', detected))

            # Initialize regime barrier optimizer
            try:
                from src.feature_generation.utils.step06_labeling_components.regime_specific_triple_barrier_optimizer import RegimeSpecificTripleBarrierOptimizer
                self.regime_barrier_optimizer = RegimeSpecificTripleBarrierOptimizer(self.config)
                self.logger.info('✅ RegimeSpecificTripleBarrierOptimizer initialized successfully')
            except ImportError as e:
                self.logger.warning(f'⚠️ Could not initialize RegimeSpecificTripleBarrierOptimizer: {e}')
                self.regime_barrier_optimizer = None

            self.logger.info(f'📋 Regime-aware labeling configuration:')
            self.logger.info(f'   - HMM regime column: {self.regime_col}')
            self.logger.info(f'   - Time barrier minutes: {self.time_barrier_minutes}')
            self.logger.info(f'   - Max lookahead: {self.max_lookahead}')

        except Exception as e:
            self.logger.error(f'❌ Failed to initialize regime-aware labeling components: {e}')
    @log_step_functions

    def validate_inputs(self, data: pd.DataFrame) -> bool:
        """Validate inputs for regime-aware labeling."""
        if self.regime_barrier_optimizer is None:
            self.logger.error('❌ Regime barrier optimizer not available')
            return False

        if self.regime_col not in data.columns:
            self.logger.error(f"❌ Regime column '{self.regime_col}' not found in data")
            return False

        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            self.logger.error(f'❌ Missing required columns for triple barrier labeling: {missing_columns}')
            return False

        return True

    def create_regime_labeler(self):
        """Create and configure the regime labeler."""
        try:
            from ..pre_training.multi_horizon_profit_labeler import MultiHorizonProfitLabeler, MultiHorizonConfig
            config = MultiHorizonConfig(
                profit_take_multiplier = 0.002,
                stop_loss_multiplier = 0.001,
                time_barrier_minutes = self.time_barrier_minutes,
                max_lookahead = self.max_lookahead,
                regime_aware = True
            )
            return MultiHorizonProfitLabeler(config)
        except ImportError as e:
            self.logger.error(f'❌ Failed to import MultiHorizonProfitLabeler: {e}')
            return None

    def generate_labels(self, data: pd.DataFrame, symbol: str, exchange: str, timeframe: str) -> Optional[pd.Series]:
        """Generate regime-aware triple barrier labels."""
        try:
            self.logger.info('🔧 Generating regime-aware triple barrier labels...')

            # Validate inputs
            if not self.validate_inputs(data):
                return None

            # Create regime labeler
            regime_labeler = self.create_regime_labeler()
            if regime_labeler is None:
                return None

            # Generate labels
            labels = regime_labeler.generate_labels(
                data,
                regime_column = self.regime_col,
                time_barrier_minutes = self.time_barrier_minutes,
                max_lookahead = self.max_lookahead
            )

            if labels is not None:
                self.logger.info(f'✅ Generated {len(labels)} regime-aware labels')
                return labels
            else:
                self.logger.error('❌ Regime-aware labeling returned None')
                return None

        except Exception as e:
            self.logger.exception(f'❌ Error in regime-aware labeling: {e}')
            return None

class MetaLabeling:
    """Meta-labeling system component."""
    @log_important_calls

    def __init__(self, config: Dict[str, Any], logger: Any = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.meta_labeling_system = None
        self._initialize_components()
    @log_all_calls

    def _initialize_components(self) -> None:
        """Initialize meta-labeling components."""
        try:
            # Try to import meta-labeling system
            try:
                from src.analyst.meta_labeling_system import MetaLabelingSystem
                self.meta_labeling_system = MetaLabelingSystem(self.config)
                self.logger.info('✅ Meta-labeling system initialized successfully')
            except ImportError as e:
                self.logger.warning(f'⚠️ Could not initialize MetaLabelingSystem: {e}')
                self.meta_labeling_system = None
        except Exception as e:
            self.logger.error(f'❌ Failed to initialize meta-labeling components: {e}')

    async def generate_analyst_labels(self, data: pd.DataFrame, symbol: str, exchange: str, timeframe: str) -> Optional[pd.Series]:
        """Generate analyst labels."""
        if self.meta_labeling_system is None:
            self.logger.warning('⚠️ Meta-labeling system not available')
            return None

        try:
            await self.meta_labeling_system.initialize()
            analyst_labels = await self.meta_labeling_system._generate_analyst_labels(data, symbol, exchange, timeframe)
            if analyst_labels is not None:
                self.logger.info('✅ Generated analyst labels')
                return analyst_labels
            else:
                self.logger.warning('⚠️ Analyst labeling returned None')
                return None
        except Exception as e:
            self.logger.warning(f'⚠️ Analyst labeling failed: {e}')
            return None

    async def generate_tactician_labels(self, data: pd.DataFrame, symbol: str, exchange: str, timeframe: str) -> Optional[pd.Series]:
        """Generate tactician labels."""
        if self.meta_labeling_system is None:
            self.logger.warning('⚠️ Meta-labeling system not available')
            return None

        try:
            await self.meta_labeling_system.initialize()
            tactician_labels = await self.meta_labeling_system._generate_tactician_labels(data, symbol, exchange, timeframe)
            if tactician_labels is not None:
                self.logger.info('✅ Generated tactician labels')
                return tactician_labels
            else:
                self.logger.warning('⚠️ Tactician labeling returned None')
                return None
        except Exception as e:
            self.logger.warning(f'⚠️ Tactician labeling failed: {e}')
            return None

class CompositeLabeling:
    """Composite labeling strategy component."""
    @log_important_calls

    def __init__(self, config: Dict[str, Any], logger: Any = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    def create_composite_label(self, data: pd.DataFrame) -> pd.Series:
        """Create composite label from multiple labeling strategies."""
        try:
            # Start with triple barrier labels as base
            if 'triple_barrier_label' not in data.columns:
                self.logger.error('❌ Triple barrier labels not found for composite labeling')
                return pd.Series(dtype = float)

            composite_label = data['triple_barrier_label'].copy()

            # Override with analyst labels where available and non-zero
            if 'analyst_label' in data.columns:
                analyst_override_mask = (data['analyst_label'] != 0) & (data['triple_barrier_label'] == 0)
                composite_label[analyst_override_mask] = data['analyst_label'][analyst_override_mask]
                self.logger.info(f'✅ Applied {analyst_override_mask.sum()} analyst label overrides')

            return composite_label

        except Exception as e:
            self.logger.warning(f'⚠️ Error creating composite label: {e}')
            return data.get('triple_barrier_label', pd.Series(dtype = float))

    def calculate_label_confidence(self, data: pd.DataFrame) -> pd.Series:
        """Calculate confidence scores for labels."""
        try:
            confidence = np.ones(len(data), dtype = np.float32)

            # Boost confidence when multiple labeling strategies agree
            if 'analyst_label' in data.columns and 'label' in data.columns:
                agreement_mask = (data['label'] == data['analyst_label']) & (data['analyst_label'] != 0)
                confidence[agreement_mask] += 0.2
                self.logger.info(f'✅ Boosted confidence for {agreement_mask.sum()} agreeing labels')

            # Cap confidence at 1.0
            confidence = np.minimum(confidence, 1.0)
            return pd.Series(confidence, index = data.index)

        except Exception as e:
            self.logger.warning(f'⚠️ Error calculating label confidence: {e}')
            return pd.Series(1.0, index = data.index)

    def determine_label_source(self, data: pd.DataFrame) -> pd.Series:
        """Determine the source of each label."""
        try:
            sources = []
            for idx in range(len(data)):
                if 'label' not in data.columns or 'triple_barrier_label' not in data.columns:
                    sources.append('unknown')
                    continue

                if data['label'].iloc[idx] == data['triple_barrier_label'].iloc[idx]:
                    if 'analyst_label' in data.columns and data['label'].iloc[idx] == data['analyst_label'].iloc[idx]:
                        sources.append('triple_barrier+analyst')
                    else:
                        sources.append('triple_barrier')
                elif 'analyst_label' in data.columns and data['label'].iloc[idx] == data['analyst_label'].iloc[idx]:
                    sources.append('analyst')
                else:
                    sources.append('composite')

            return pd.Series(sources, index = data.index)

        except Exception as e:
            self.logger.warning(f'⚠️ Error determining label source: {e}')
            return pd.Series('unknown', index = data.index)

class ComprehensiveLabeling:
    """Comprehensive labeling orchestrator that combines all labeling strategies."""
    @log_important_calls

    def __init__(self, config: Dict[str, Any], logger: Any = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # Initialize components
        self.regime_aware_labeling = RegimeAwareLabeling(config, logger)
        self.meta_labeling = MetaLabeling(config, logger)
        self.composite_labeling = CompositeLabeling(config, logger)

        # Configuration
        self.auto_recalculate_hmm_barriers = bool(
            config.get('vectorized_labelling_orchestrator', {}).get('auto_recalculate_hmm_barriers', True)
        )

    async def generate_comprehensive_labels(self, data: pd.DataFrame, symbol: str, exchange: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Generate comprehensive labels combining multiple labeling strategies."""
        try:
            self.logger.info('🚀 Generating comprehensive labels...')
            result_data = data.copy()

            # Step 1: Generate triple barrier labels if not present
            if 'triple_barrier_label' not in result_data.columns:
                self.logger.info('🔄 Triple barrier labels not found, generating them...')

                if self.regime_aware_labeling.regime_barrier_optimizer is not None and self.auto_recalculate_hmm_barriers:
                    try:
                        self.logger.info('🚀 Attempting regime-aware triple barrier labeling...')

                        if self.regime_aware_labeling.regime_col in result_data.columns:
                            self.logger.info(f'✅ Found regime column: {self.regime_aware_labeling.regime_col}')

                            regime_labels = self.regime_aware_labeling.generate_labels(result_data, symbol, exchange, timeframe)
                            if regime_labels is not None:
                                result_data['triple_barrier_label'] = regime_labels
                                result_data['labeling_method'] = 'regime_aware'
                                self.logger.info('✅ Generated regime-aware triple barrier labels')
                            else:
                                raise Exception('Regime-aware labeling failed')
                        else:
                            self.logger.warning(f"⚠️ Regime column '{self.regime_aware_labeling.regime_col}' not found")
                            raise Exception('Regime column not found')
                    except Exception as e:
                        self.logger.error(f'❌ Regime-aware labeling failed: {e}')
                        self.logger.error('❌ No fallback labeling method available - regime-aware labeling is required')
                        return None
                else:
                    if not self.auto_recalculate_hmm_barriers:
                        self.logger.error('❌ Auto-calculation disabled for regime-aware labeling')
                    if self.regime_aware_labeling.regime_barrier_optimizer is None:
                        self.logger.error('❌ Regime barrier optimizer not available')
                    self.logger.error('❌ Regime-aware labeling is required - no fallback available')
                    return None

            # Step 2: Generate meta-labels (analyst and tactician)
            analyst_labels = await self.meta_labeling.generate_analyst_labels(result_data, symbol, exchange, timeframe)
            if analyst_labels is not None:
                result_data['analyst_label'] = analyst_labels

            tactician_labels = await self.meta_labeling.generate_tactician_labels(result_data, symbol, exchange, timeframe)
            if tactician_labels is not None:
                result_data['tactician_label'] = tactician_labels

            # Step 3: Create composite label
            composite_label = self.composite_labeling.create_composite_label(result_data)
            result_data['label'] = composite_label

            # Step 4: Calculate confidence and source
            result_data['label_confidence'] = self.composite_labeling.calculate_label_confidence(result_data)
            result_data['label_source'] = self.composite_labeling.determine_label_source(result_data)

            # Log results
            self.logger.info(f'✅ Generated comprehensive labels with {len(result_data.columns)} columns')
            self.logger.info(f"   - Label distribution: {result_data['label'].value_counts().to_dict()}")
            self.logger.info(f"   - Labeling method used: {result_data.get('labeling_method', 'unknown')}")

            return result_data

        except Exception as e:
            self.logger.exception(f'❌ Error generating comprehensive labels: {e}')
            return None

__all__ = [
    "RegimeAwareLabeling",
    "MetaLabeling",
    "CompositeLabeling",
    "ComprehensiveLabeling",
]
