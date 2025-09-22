"""
HMM Ensemble Training Component

This component wraps the HMM ensemble training functionality as a pipeline component
that follows the BaseMarketAnalysisComponent interface.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from datetime import datetime
import time
import logging

from .base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult
from src.utils.tprint import tprint

# Import the HMM ensemble training functionality
from ..hmm_models_training.hmm_ensemble_training import (
    HMMEnsembleTrainingComponent as HMMEnsembleTrainingCore,
    execute_hmm_ensemble_training
)

logger = logging.getLogger(__name__)

class HMMEnsembleTrainingComponent(BaseMarketAnalysisComponent):
    """
    HMM Ensemble Training Component.

    Wraps the HMM ensemble training functionality as a pipeline component
    that follows the BaseMarketAnalysisComponent interface.
    """

    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the HMM ensemble training component."""
        super().__init__(config)
        self.core_component = HMMEnsembleTrainingCore(config=None, enable_vectorization=True)
        try:
            # Enforce 15m timeframe for ensemble runtime
            if hasattr(self.core_component, 'config'):
                setattr(self.core_component.config, 'timeframe', '15m')
                if getattr(self.core_component.config, 'timeframe', None) != '15m':
                    tprint("⚠️ HMM Ensemble: Non-15m timeframe supplied; overriding to 15m for consistency")
        except Exception:
            pass

    def get_required_artifacts(self) -> list[str]:
        """Get list of required artifacts this component must produce."""
        return ['hmm_ensemble_training_result']

    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute HMM ensemble training component.

        Args:
            data: Input data containing features, targets, and regime information
            pipeline_state: Current pipeline state

        Returns:
            ComponentResult with training results
        """
        execution_start_time = time.time()
        tprint("🚀 Starting HMM Ensemble Training Component")

        try:
            # Extract required data from pipeline state
            if 'dataframe' not in pipeline_state:
                raise ValueError("No dataframe found in pipeline state")

            dataframe = pipeline_state['dataframe']

            # Extract features and targets from dataframe
            # Look for features that were created by previous components
            feature_columns = [col for col in dataframe.columns if col.startswith('feature_') or col in [
                'close', 'high', 'low', 'open', 'volume', 'regime', 'regime_label'
            ]]

            if len(feature_columns) == 0:
                raise ValueError("No suitable feature columns found in dataframe")

            # Extract target variable (regime labels)
            if 'regime_label' in dataframe.columns:
                y = dataframe['regime_label'].values
            elif 'regime' in dataframe.columns:
                y = dataframe['regime'].values
            else:
                raise ValueError("No regime labels found in dataframe")

            # Extract feature matrix
            X = dataframe[feature_columns].values

            # Ensure X and y have the same length
            if X.shape[0] != y.shape[0]:
                raise ValueError(f"Feature matrix and target have different lengths: X={X.shape[0]}, y={y.shape[0]}")

            tprint(f"📊 Data extracted from dataframe: X={X.shape}, y={y.shape}, features={feature_columns}")

            # Extract regime labels for ensemble training - will be set to HMM state assignments
            regime_labels = None

            # Extract base HMM models from previous pipeline results
            base_hmm_models = {}
            hmm_training_metrics = {}

            # Look for previous HMM model results in pipeline state
            if 'hmm_models' in pipeline_state:
                base_hmm_models = pipeline_state['hmm_models']
            if 'hmm_metrics' in pipeline_state:
                hmm_training_metrics = pipeline_state['hmm_metrics']

            # Load cluster assignments from HMM training input file (parameterized)
            cluster_assignments = None
            try:
                import glob
                import pickle
                import os

                symbol = getattr(self.config, 'symbol', pipeline_state.get('symbol', 'ETHUSDT'))
                exchange = getattr(self.config, 'exchange', pipeline_state.get('exchange', 'BINANCE')).lower()
                timeframe = getattr(self.config, 'timeframe', pipeline_state.get('timeframe', '15m'))

                # Build pattern using parameters
                pattern = f"optimal_clusters/{exchange.lower()}/{symbol}/{timeframe}/market_analysis_hmm_training_input_{symbol}_{exchange.upper()}_{timeframe}_*.pkl"
                hmm_input_files = glob.glob(pattern)

                if hmm_input_files:
                    # Choose most recent by modification time
                    latest_file = max(hmm_input_files, key=lambda p: os.path.getmtime(p))
                    tprint(f"🔍 Loading cluster assignments from latest HMM training input file: {latest_file}")

                    with open(latest_file, 'rb') as f:
                        hmm_input_data = pickle.load(f)

                    if 'cluster_assignments' in hmm_input_data:
                        cluster_assignments = hmm_input_data['cluster_assignments']
                        tprint(f"✅ Loaded {len(cluster_assignments)} cluster assignments from HMM training input file")
                        try:
                            uniques = len(set(cluster_assignments))
                            tprint(f"📊 Cluster assignments shape: {getattr(cluster_assignments, 'shape', ('n/a',))}, Unique clusters: {uniques}")
                        except Exception:
                            pass
                    else:
                        raise ValueError("No cluster_assignments found in HMM training input file contents")
                else:
                    raise ValueError(f"No HMM training input files found matching pattern: {pattern}")

            except Exception as e:
                tprint(f"❌ Error loading cluster assignments from HMM training input file: {e}")
                raise ValueError(f"Failed to load cluster assignments: {e}")

            # Validate cluster_assignments length matches X
            if cluster_assignments is not None:
                if len(cluster_assignments) != X.shape[0]:
                    tprint(f"❌ CRITICAL: Data shape mismatch - FAILING FAST")
                    tprint(f"    X={X.shape}, cluster_assignments={cluster_assignments.shape}")
                    raise ValueError(f"Data shape mismatch: X={X.shape}, cluster_assignments={cluster_assignments.shape}")
                else:
                    tprint(f"✅ Cluster assignments loaded: {len(cluster_assignments)} samples")
            else:
                tprint(f"❌ CRITICAL: No cluster assignments available - FAILING FAST")
                raise ValueError("No cluster assignments available from HMM training input file")

            # For HMM state recognition, set both target y and regime_labels to the HMM state assignments
            y = cluster_assignments
            regime_labels = cluster_assignments

            tprint(f"📊 Final data shapes: X={X.shape}, y={y.shape}, regime_labels={len(regime_labels)}")

            # Execute HMM ensemble training with aligned data
            tprint(f"🔄 Training HMM ensemble with {len(feature_columns)} features and {len(dataframe)} samples")

            # Call the core HMM ensemble training function with aligned data
            training_result = execute_hmm_ensemble_training(
                X=X,
                y=y,
                regime_labels=regime_labels,
                feature_names=feature_columns,
                base_hmm_models=base_hmm_models,
                hmm_training_metrics=hmm_training_metrics,
                enable_vectorization=True
            )

            # Extract results
            artifacts = {
                'hmm_ensemble_training_result': training_result,
                'models': training_result.get('models', []),
                'metrics': training_result.get('metrics', {}),
                'performance': training_result.get('performance', {}),
                'feature_names': feature_columns,
                'training_time': training_result.get('execution_time', 0.0)
            }

            # Add any additional artifacts from the training result
            artifacts.update(training_result.get('artifacts', {}))

            execution_time = time.time() - execution_start_time

            tprint(f"✅ HMM ensemble training completed in {execution_time:.2f} seconds")
            tprint(f"📊 Generated {len(artifacts)} types of artifacts")

            return ComponentResult(
                success=True,
                artifacts=artifacts,
                execution_time=execution_time,
                metadata={
                    'component': 'hmm_ensemble_training',
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    'feature_count': len(feature_columns),
                    'sample_count': len(dataframe),
                    'execution_time': execution_time
                }
            )

        except Exception as e:
            execution_time = time.time() - execution_start_time
            error_msg = f"HMM ensemble training failed: {e}"
            tprint(f"❌ {error_msg}")

            logger.error(f"HMM ensemble training failed: {e}")

            return ComponentResult(
                success=False,
                artifacts={},
                error_message=error_msg,
                execution_time=execution_time,
                metadata={
                    'component': 'hmm_ensemble_training',
                    'error': str(e),
                    'execution_time': execution_time
                }
            )
