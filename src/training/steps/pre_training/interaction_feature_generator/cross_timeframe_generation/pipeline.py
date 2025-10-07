"""
Main Cross-Timeframe Pipeline

Orchestrates the complete HTF feature generation and optimization pipeline.
Implements the high-level DAG per asset with sessionization, alignment, and causal joins.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

from .phase1_probe import Phase1HTFProbe
from .phase2_optimization import Phase2Optimization
from .regime_segmentation import RegimeSegmentation
from .scoring_system import AdaptiveScoringSystem
from .ehu_rih_assignment import EHU_RIH_Assignment
from .knapsack_selection import KnapsackSelection
from .htf_materialization import HTFMaterialization
from .interaction_templates import HTFInteractionTemplates
from .statistical_selection import StatisticalSelection
from .evaluation import WalkForwardEvaluation
from .monitoring import MonitoringSystem

# Import existing feature generation components
import sys
sys.path.append('src/training/steps/pre_training/interaction_feature_generator/feature_interaction_generation')
from feature_engineering.feature_registry import FeatureRegistry, FeatureFamily
from feature_engineering.transforms import TransformRouter, create_default_transform_config


@dataclass
class PipelineConfig:
    """Configuration for the cross-timeframe pipeline."""
    
    # Data configuration
    base_timeframe_minutes: int = 5  # Base timeframe (5m or 15m)
    session_start_hour: int = 9  # Market session start hour
    session_end_hour: int = 16  # Market session end hour
    dst_handling: bool = True  # Handle DST transitions
    
    # Phase-1 configuration
    coarse_grid_min: int = 15  # Minimum HTF lookback (minutes)
    coarse_grid_max: int = 298  # Maximum HTF lookback (minutes)
    adaptive_refinement_threshold: float = 0.75  # Top-quartile threshold for refinement
    
    # Phase-2 configuration
    local_grid_factor: float = 0.5  # Local grid around shortlisted B
    ic_surface_smoothing: str = 'spline'  # 'spline' or 'gp'
    
    # Regime segmentation
    change_point_method: str = 'PELT'  # 'PELT' or 'CUSUM'
    regime_vol_quantile: float = 0.6  # Q60 for vol regime classification
    bocpd_hazard: float = 1/200  # BOCPD hazard rate
    
    # Scoring configuration
    lambda_unc: float = 0.10  # Uncertainty penalty
    lambda_cost: float = 0.05  # Cost penalty
    lambda_stale: float = 0.05  # Staleness penalty
    meta_learning_range: float = 0.05  # Meta-learner adjustment range
    
    # EHU/RIH assignment
    rih_threshold: float = 0.01  # ΔIC/Δms threshold for RIH
    hybrid_mode: bool = True  # Allow runtime switching
    
    # Knapsack constraints
    max_cost_ms: float = 25.0  # Maximum total cost in ms
    max_features: int = 120  # Maximum number of features
    max_correlation: float = 0.8  # Maximum partial correlation
    
    # Selection configuration
    stability_resamples: int = 80  # Bootstrap resamples for stability selection
    fdr_q: float = 0.1  # FDR threshold
    min_conditional_ic: float = 0.25  # Minimum conditional IC improvement
    
    # Evaluation configuration
    embargo_minutes: int = 60  # Minimum embargo period
    walk_forward_folds: int = 5  # Number of walk-forward folds
    spa_test: bool = True  # Include SPA test
    
    # Monitoring
    adaptive_penalties: bool = True  # Enable meta-learning of penalties
    dashboard_enabled: bool = True  # Enable monitoring dashboard


class CrossTimeframePipeline:
    """
    Main pipeline for cross-timeframe feature generation and optimization.
    
    Implements the complete DAG:
    1. Sessionize & align data
    2. Phase-1 HTF probes (coarse grids)
    3. Phase-2 optimization (local grids)
    4. EHU/RIH assignment
    5. Knapsack selection
    6. Materialize HTFs
    7. Generate interactions
    8. Statistical selection
    9. Walk-forward evaluation
    10. Monitoring & automation
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.feature_registry = FeatureRegistry()
        self.regime_segmentation = RegimeSegmentation(config)
        self.phase1_probe = Phase1HTFProbe(config)
        self.phase2_optimization = Phase2Optimization(config)
        self.scoring_system = AdaptiveScoringSystem(config)
        self.ehu_rih_assignment = EHU_RIH_Assignment(config)
        self.knapsack_selection = KnapsackSelection(config)
        self.htf_materialization = HTFMaterialization(config)
        self.interaction_templates = HTFInteractionTemplates(config)
        self.statistical_selection = StatisticalSelection(config)
        self.evaluation = WalkForwardEvaluation(config)
        self.monitoring = MonitoringSystem(config)
        
        # Pipeline state
        self.sessionized_data = None
        self.regime_segments = None
        self.phase1_results = None
        self.phase2_results = None
        self.selected_htfs = None
        self.materialized_htfs = None
        self.interactions = None
        self.final_features = None
        self.evaluation_results = None
        
    def run_pipeline(self, 
                    ohlcv_data: pd.DataFrame,
                    optional_data: Optional[Dict[str, pd.DataFrame]] = None,
                    targets: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Run the complete cross-timeframe pipeline.
        
        Args:
            ohlcv_data: OHLCV data with datetime index
            optional_data: Optional data (top-of-book, trades, index prices)
            targets: Target variables (log-returns or signs)
            
        Returns:
            Dictionary containing all pipeline results
        """
        self.logger.info("Starting cross-timeframe pipeline")
        
        try:
            # Step 0: Sessionize & align data
            self.logger.info("Step 0: Sessionizing and aligning data")
            self.sessionized_data = self._sessionize_and_align(ohlcv_data, optional_data)
            
            # Step 1: Regime segmentation
            self.logger.info("Step 1: Performing regime segmentation")
            self.regime_segments = self.regime_segmentation.segment_regimes(
                self.sessionized_data, targets
            )
            
            # Step 2: Phase-1 HTF probe stage
            self.logger.info("Step 2: Phase-1 HTF probe stage")
            self.phase1_results = self.phase1_probe.run_probe_stage(
                self.sessionized_data, self.regime_segments, targets
            )
            
            # Step 3: Phase-2 optimization
            self.logger.info("Step 3: Phase-2 optimization")
            self.phase2_results = self.phase2_optimization.optimize_lookbacks(
                self.sessionized_data, self.phase1_results, self.regime_segments, targets
            )
            
            # Step 4: EHU/RIH assignment
            self.logger.info("Step 4: EHU/RIH assignment")
            ehu_rih_assignments = self.ehu_rih_assignment.assign_htf_features(
                self.phase2_results, self.sessionized_data
            )
            
            # Step 5: Knapsack selection
            self.logger.info("Step 5: Knapsack selection")
            self.selected_htfs = self.knapsack_selection.select_features(
                self.phase2_results, ehu_rih_assignments
            )
            
            # Step 6: Materialize HTFs
            self.logger.info("Step 6: Materializing HTFs")
            self.materialized_htfs = self.htf_materialization.materialize_htfs(
                self.sessionized_data, self.selected_htfs
            )
            
            # Step 7: Generate interactions
            self.logger.info("Step 7: Generating HTF-aware interactions")
            self.interactions = self.interaction_templates.generate_interactions(
                self.materialized_htfs, self.sessionized_data
            )
            
            # Step 8: Statistical selection
            self.logger.info("Step 8: Statistical selection")
            self.final_features = self.statistical_selection.select_final_features(
                self.materialized_htfs, self.interactions, targets
            )
            
            # Step 9: Walk-forward evaluation
            self.logger.info("Step 9: Walk-forward evaluation")
            self.evaluation_results = self.evaluation.evaluate_features(
                self.final_features, targets, self.regime_segments
            )
            
            # Step 10: Monitoring & automation
            self.logger.info("Step 10: Setting up monitoring")
            self.monitoring.setup_monitoring(
                self.final_features, self.evaluation_results, self.regime_segments
            )
            
            # Compile results
            results = {
                'sessionized_data': self.sessionized_data,
                'regime_segments': self.regime_segments,
                'phase1_results': self.phase1_results,
                'phase2_results': self.phase2_results,
                'selected_htfs': self.selected_htfs,
                'materialized_htfs': self.materialized_htfs,
                'interactions': self.interactions,
                'final_features': self.final_features,
                'evaluation_results': self.evaluation_results,
                'pipeline_config': self.config
            }
            
            self.logger.info("Cross-timeframe pipeline completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def _sessionize_and_align(self, 
                            ohlcv_data: pd.DataFrame,
                            optional_data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, Any]:
        """
        Sessionize and align data with DST handling and causal joins.
        
        Args:
            ohlcv_data: OHLCV data with datetime index
            optional_data: Optional additional data sources
            
        Returns:
            Dictionary containing sessionized and aligned data
        """
        # Create session boundaries
        sessions = self._create_sessions(ohlcv_data)
        
        # Handle DST transitions
        if self.config.dst_handling:
            sessions = self._handle_dst_transitions(sessions)
        
        # Align data to sessions
        aligned_data = self._align_to_sessions(ohlcv_data, sessions)
        
        # Add optional data if provided
        if optional_data:
            for name, data in optional_data.items():
                aligned_data[name] = self._align_to_sessions(data, sessions)
        
        return {
            'sessions': sessions,
            'aligned_data': aligned_data,
            'base_timeframe': self.config.base_timeframe_minutes
        }
    
    def _create_sessions(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create trading sessions from data."""
        sessions = []
        current_session = None
        
        for timestamp in data.index:
            hour = timestamp.hour
            
            # Check if we're in trading hours
            if self.config.session_start_hour <= hour < self.config.session_end_hour:
                if current_session is None:
                    # Start new session
                    current_session = {
                        'session_id': len(sessions),
                        'open_dt': timestamp,
                        'close_dt': None,
                        'bars': []
                    }
                current_session['bars'].append(timestamp)
            else:
                if current_session is not None:
                    # End current session
                    current_session['close_dt'] = current_session['bars'][-1]
                    sessions.append(current_session)
                    current_session = None
        
        # Close final session if open
        if current_session is not None:
            current_session['close_dt'] = current_session['bars'][-1]
            sessions.append(current_session)
        
        return sessions
    
    def _handle_dst_transitions(self, sessions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Handle DST transitions in session data."""
        # This is a simplified implementation
        # In practice, you'd need to handle the actual DST transition dates
        # and adjust session boundaries accordingly
        return sessions
    
    def _align_to_sessions(self, data: pd.DataFrame, sessions: List[Dict[str, Any]]) -> pd.DataFrame:
        """Align data to session boundaries."""
        # For now, return the data as-is
        # In practice, you'd align timestamps to session boundaries
        return data
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and progress."""
        return {
            'sessionized_data': self.sessionized_data is not None,
            'regime_segments': self.regime_segments is not None,
            'phase1_results': self.phase1_results is not None,
            'phase2_results': self.phase2_results is not None,
            'selected_htfs': self.selected_htfs is not None,
            'materialized_htfs': self.materialized_htfs is not None,
            'interactions': self.interactions is not None,
            'final_features': self.final_features is not None,
            'evaluation_results': self.evaluation_results is not None
        }
    
    def save_pipeline_state(self, filepath: str):
        """Save pipeline state to disk."""
        state = {
            'config': self.config,
            'sessionized_data': self.sessionized_data,
            'regime_segments': self.regime_segments,
            'phase1_results': self.phase1_results,
            'phase2_results': self.phase2_results,
            'selected_htfs': self.selected_htfs,
            'materialized_htfs': self.materialized_htfs,
            'interactions': self.interactions,
            'final_features': self.final_features,
            'evaluation_results': self.evaluation_results
        }
        
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load_pipeline_state(self, filepath: str):
        """Load pipeline state from disk."""
        import pickle
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.config = state['config']
        self.sessionized_data = state['sessionized_data']
        self.regime_segments = state['regime_segments']
        self.phase1_results = state['phase1_results']
        self.phase2_results = state['phase2_results']
        self.selected_htfs = state['selected_htfs']
        self.materialized_htfs = state['materialized_htfs']
        self.interactions = state['interactions']
        self.final_features = state['final_features']
        self.evaluation_results = state['evaluation_results']