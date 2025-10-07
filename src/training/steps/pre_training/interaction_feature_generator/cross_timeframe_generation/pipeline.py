"""
Main Cross-Timeframe Pipeline

Orchestrates the complete HTF feature generation and optimization pipeline.
Implements the high-level DAG per asset with sessionization, alignment, and causal joins.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from zoneinfo import ZoneInfo

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
from .config import PipelineConfig

# Import existing feature generation components
import sys
sys.path.append('src/training/steps/pre_training/interaction_feature_generator/feature_interaction_generation')
from feature_engineering.feature_registry import FeatureRegistry, FeatureFamily
from feature_engineering.transforms import TransformRouter, create_default_transform_config


class CrossTimeframePipeline:
    """
    Main pipeline for cross-timeframe feature generation and optimization.
    
    Implements the complete DAG:
    1. Sessionize & align data
    2. Phase-1 HTF probes (coarse grids)
    3. Phase-2 optimization (local grids)
    4. EHU/RIH assignment
    5. Knapsack selection (produces ``KnapsackSelectionResult``)
    6. Materialize HTFs
    7. Generate interactions
    8. Statistical selection (produces ``StatisticalSelectionResult``)
    9. Walk-forward evaluation
    10. Monitoring & automation
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.feature_registry = FeatureRegistry()
        self.regime_segmentation = RegimeSegmentation(config.regime)
        self.scoring_system = AdaptiveScoringSystem(config.scoring, config.session)
        self.phase1_probe = Phase1HTFProbe(
            session_config=config.session,
            probe_config=config.probe,
            scoring_system=self.scoring_system,
        )
        self.phase2_optimization = Phase2Optimization(config.optimization)
        self.ehu_rih_assignment = EHU_RIH_Assignment(config.assignment)
        self.knapsack_selection = KnapsackSelection(config.selection)
        self.htf_materialization = HTFMaterialization(config.session)
        self.interaction_templates = HTFInteractionTemplates(config.selection)
        self.statistical_selection = StatisticalSelection(config.selection)
        self.evaluation = WalkForwardEvaluation(config.evaluation)
        self.monitoring = MonitoringSystem(
            monitoring_config=config.monitoring,
            scoring_config=config.scoring,
            regime_config=config.regime,
        )
        
        # Pipeline state
        self.sessionized_data = None
        self.regime_segments = None
        self.phase1_results = None
        self.phase2_results = None
        # Holds the resource allocation output (KnapsackSelectionResult)
        # before statistical pruning narrows the candidate set.
        self.selected_htfs = None
        self.materialized_htfs = None
        self.interactions = None
        # Holds the final statistical pruning output (StatisticalSelectionResult)
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
            Dictionary containing all pipeline results. The knapsack stage
            returns a ``KnapsackSelectionResult`` under ``selected_htfs`` while
            the statistical pruning stage returns a
            ``StatisticalSelectionResult`` under ``final_features`` so callers
            can differentiate between the two selection phases.
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
            
            # Step 5: Knapsack selection (stage-one resource allocation)
            self.logger.info("Step 5: Knapsack selection")
            self.selected_htfs = self.knapsack_selection.select_features(
                self.phase2_results, ehu_rih_assignments, self.sessionized_data
            )
            
            # Step 6: Materialize HTFs
            self.logger.info("Step 6: Materializing HTFs")
            selected_feature_candidates: List[Any]
            if self.selected_htfs is None:
                selected_feature_candidates = []
            elif isinstance(self.selected_htfs, list):
                selected_feature_candidates = self.selected_htfs
            else:
                selected_feature_candidates = getattr(
                    self.selected_htfs, 'selected_features', []
                ) or []

            self.materialized_htfs = self.htf_materialization.materialize_htfs(
                self.sessionized_data, selected_feature_candidates
            )
            
            # Step 7: Generate interactions
            self.logger.info("Step 7: Generating HTF-aware interactions")

            base_feature_series: Dict[str, pd.Series] = {}
            base_feature_structure: Optional[Union[pd.DataFrame, Dict[str, pd.Series]]] = None

            if isinstance(self.sessionized_data, dict):
                aligned_base_features = self.sessionized_data.get('aligned_data')

                if isinstance(aligned_base_features, pd.DataFrame):
                    base_feature_structure = aligned_base_features
                    base_feature_series = {
                        column: aligned_base_features[column]
                        for column in aligned_base_features.columns
                    }
                elif isinstance(aligned_base_features, dict):
                    filtered_series = {
                        name: series
                        for name, series in aligned_base_features.items()
                        if isinstance(series, pd.Series)
                    }
                    base_feature_structure = filtered_series
                    base_feature_series = filtered_series

            if not base_feature_series:
                self.logger.warning(
                    "No base features available from sessionized data; interaction grouping may be limited."
                )

            self.interactions = self.interaction_templates.generate_interactions(
                self.materialized_htfs,
                base_feature_structure if base_feature_structure is not None else base_feature_series,
                targets,
            )
            
            # Step 8: Statistical selection (stage-two statistical pruning)
            self.logger.info("Step 8: Statistical selection")
            self.final_features = self.statistical_selection.select_final_features(
                self.materialized_htfs, self.interactions, targets
            )
            
            # Step 9: Walk-forward evaluation
            self.logger.info("Step 9: Walk-forward evaluation")
            final_feature_list = (
                self.final_features.selected_features
                if self.final_features is not None
                else []
            )

            if targets is not None:
                self.evaluation_results = self.evaluation.evaluate_features(
                    final_feature_list,
                    targets,
                    self.regime_segments,
                    materialized_htfs=self.materialized_htfs,
                    interactions=self.interactions,
                )
            else:
                self.logger.warning("Targets not provided – skipping evaluation stage")
                self.evaluation_results = None

            evaluation_summary = None
            if self.evaluation_results is not None and hasattr(self.evaluation, "get_evaluation_summary"):
                try:
                    evaluation_summary = self.evaluation.get_evaluation_summary(self.evaluation_results)
                except Exception as summary_error:
                    self.logger.warning(
                        "Failed to summarize evaluation results: %s", summary_error
                    )
                    evaluation_summary = None

            # Step 10: Monitoring & automation
            self.logger.info("Step 10: Setting up monitoring")
            self.monitoring.setup_monitoring(
                final_feature_list,
                evaluation_summary,
                self.regime_segments,
            )

            if self.config.adaptive_penalties:
                updated_penalties = self.monitoring.get_penalty_parameters()
                if updated_penalties:
                    self.scoring_system.apply_penalty_parameters(updated_penalties)
                    self.logger.info(
                        "Applied refreshed adaptive penalties to scoring system: %s",
                        updated_penalties,
                    )

            # Compile results keeping both selection stages visible to callers
            results = {
                'sessionized_data': self.sessionized_data,
                'regime_segments': self.regime_segments,
                'phase1_results': self.phase1_results,
                'phase2_results': self.phase2_results,
                'selected_htfs': self.selected_htfs,
                'materialized_htfs': self.materialized_htfs,
                'interactions': self.interactions,
                'final_features': self.final_features,
                'selected_feature_list': final_feature_list,
                'evaluation_results': self.evaluation_results,
                'evaluation_summary': evaluation_summary,
                'pipeline_config': self.config
            }
            
            self.logger.info("Cross-timeframe pipeline completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise

    def _collect_selected_feature_matrix(self, selected_features: List[str]) -> pd.DataFrame:
        """Materialize the selected feature series into a DataFrame."""
        if not selected_features:
            return pd.DataFrame()

        feature_data: Dict[str, pd.Series] = {}

        # Materialized HTFs
        if isinstance(self.materialized_htfs, dict):
            for name, htf in self.materialized_htfs.items():
                if name in selected_features and hasattr(htf, 'feature_series'):
                    series = getattr(htf, 'feature_series')
                    if isinstance(series, pd.Series):
                        feature_data[name] = series

        # Interaction features
        if isinstance(self.interactions, list):
            for interaction in self.interactions:
                interaction_name = getattr(interaction, 'name', None)
                if interaction_name in selected_features and hasattr(interaction, 'feature_series'):
                    series = getattr(interaction, 'feature_series')
                    if isinstance(series, pd.Series):
                        feature_data[interaction_name] = series

        if not feature_data:
            return pd.DataFrame()

        feature_matrix = pd.DataFrame(feature_data)
        ordered_columns = [name for name in selected_features if name in feature_matrix.columns]
        return feature_matrix.loc[:, ordered_columns]

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
        aligned_base = self._align_to_sessions(ohlcv_data, sessions)
        aligned_data = aligned_base

        # Add optional data if provided
        if optional_data:
            for name, data in optional_data.items():
                aligned_optional = self._align_to_sessions(data, sessions)

                if aligned_optional.empty:
                    continue

                if 'session_id' in aligned_optional.columns:
                    aligned_optional = aligned_optional.drop(columns=['session_id'])

                if isinstance(aligned_data, pd.DataFrame):
                    opt_columns = list(aligned_optional.columns)
                    renamed = {
                        col: col if col.startswith(f"{name}_") or col == name else f"{name}_{col}"
                        for col in opt_columns
                    }
                    aligned_optional = aligned_optional.rename(columns=renamed)
                    aligned_data = aligned_data.join(aligned_optional, how='left')
                else:
                    aligned_data[name] = aligned_optional

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
        if not sessions:
            return sessions

        tz_candidates = []
        for session in sessions:
            if session.get('bars'):
                first_bar = pd.Timestamp(session['bars'][0])
                if first_bar.tzinfo is not None:
                    tz_candidates.append(first_bar.tzinfo)
        if not tz_candidates:
            open_dt = pd.Timestamp(sessions[0]['open_dt'])
            if open_dt.tzinfo is not None:
                tz_candidates.append(open_dt.tzinfo)

        market_tz = getattr(self.config, 'market_timezone', None)
        if market_tz:
            timezone = ZoneInfo(market_tz)
        elif tz_candidates:
            tz_candidate = tz_candidates[0]
            tz_name = getattr(tz_candidate, 'key', None) or getattr(tz_candidate, 'zone', None)
            timezone = ZoneInfo(tz_name) if tz_name else tz_candidate
        else:
            timezone = ZoneInfo('America/New_York')

        def ensure_timezone(ts: Any) -> pd.Timestamp:
            ts = pd.Timestamp(ts)
            if ts.tzinfo is None:
                return ts.tz_localize(timezone)
            return ts.tz_convert(timezone)

        start_ts = ensure_timezone(sessions[0]['open_dt'])
        end_ts = ensure_timezone(sessions[-1]['close_dt'])

        dst_transition_dates = self._compute_dst_transition_dates(timezone, start_ts, end_ts)

        adjusted_sessions: List[Dict[str, Any]] = []
        for session in sessions:
            adjusted_session = session.copy()
            bars = [ensure_timezone(bar) for bar in session.get('bars', [])]
            adjusted_session['bars'] = bars

            if not bars:
                open_ts = ensure_timezone(session['open_dt'])
                close_ts = ensure_timezone(session['close_dt'])
            else:
                open_ts = bars[0]
                close_ts = bars[-1]

            session_date = open_ts.date()
            if session_date in dst_transition_dates:
                open_time = open_ts.time()
                close_time = close_ts.time()
                open_ts = pd.Timestamp(datetime.combine(session_date, open_time), tz=timezone)
                close_ts = pd.Timestamp(datetime.combine(session_date, close_time), tz=timezone)
                adjusted_session['dst_transition'] = True
            else:
                open_ts = open_ts.tz_convert(timezone)
                close_ts = close_ts.tz_convert(timezone)

            adjusted_session['open_dt'] = open_ts
            adjusted_session['close_dt'] = close_ts
            adjusted_sessions.append(adjusted_session)

        return adjusted_sessions

    def _compute_dst_transition_dates(self, timezone: ZoneInfo, start: pd.Timestamp, end: pd.Timestamp) -> set:
        """Return set of local dates where DST transitions occur within the window."""
        check_start = start.normalize() - pd.Timedelta(days=2)
        check_end = end.normalize() + pd.Timedelta(days=2)
        hourly_index = pd.date_range(start=check_start, end=check_end, freq='h', tz=timezone)

        transition_dates = set()
        previous_offset: Optional[timedelta] = None
        for ts in hourly_index:
            offset = ts.utcoffset()
            if previous_offset is not None and offset != previous_offset:
                transition_dates.add(ts.date())
            previous_offset = offset

        return transition_dates

    def _align_to_sessions(self, data: pd.DataFrame, sessions: List[Dict[str, Any]]) -> pd.DataFrame:
        """Align data to session boundaries."""
        if data is None or data.empty or not sessions:
            return pd.DataFrame()

        if isinstance(data, pd.Series):
            data = data.to_frame(name=data.name or 'value')

        data = data.sort_index()
        base_freq = f"{self.config.base_timeframe_minutes}min"
        aligned_frames: List[pd.DataFrame] = []
        last_values: Optional[pd.Series] = None

        for session in sessions:
            open_dt = pd.Timestamp(session['open_dt'])
            close_dt = pd.Timestamp(session['close_dt'])
            if pd.isna(open_dt) or pd.isna(close_dt):
                continue

            session_mask = (data.index >= open_dt) & (data.index <= close_dt)
            session_data = data.loc[session_mask].copy()

            session_index = pd.date_range(start=open_dt, end=close_dt, freq=base_freq, tz=open_dt.tzinfo)
            if session_index.empty:
                continue

            session_data = session_data.reindex(session_index)

            session_data = session_data.ffill()

            if session_data.isna().any().any() and last_values is not None:
                session_data = session_data.fillna(last_values)
                session_data = session_data.ffill()

            session_data['session_id'] = session['session_id']

            if not session_data.empty:
                last_values = session_data.iloc[-1][data.columns]
            aligned_frames.append(session_data)

        if not aligned_frames:
            return pd.DataFrame()

        aligned_data = pd.concat(aligned_frames)
        aligned_data = aligned_data[~aligned_data.index.duplicated(keep='first')]
        return aligned_data
    
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