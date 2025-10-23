"""
Human-in-the-loop Review Interface
Streamlit app for reviewing and correcting extracted data
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from literature.extraction.extractor import ExtractedExperiment, ExtractedParameter
from literature.normalization.validator import NormalizedExperiment, NormalizedParameter

logger = logging.getLogger(__name__)

class ReviewApp:
    """Main review application class"""
    
    def __init__(self):
        self.setup_page_config()
        self.setup_session_state()
    
    def setup_page_config(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="Truffle Literature Review",
            page_icon="🍄",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def setup_session_state(self):
        """Setup session state variables"""
        if 'experiments' not in st.session_state:
            st.session_state.experiments = []
        if 'current_experiment_idx' not in st.session_state:
            st.session_state.current_experiment_idx = 0
        if 'reviewed_experiments' not in st.session_state:
            st.session_state.reviewed_experiments = set()
        if 'corrections' not in st.session_state:
            st.session_state.corrections = {}
        if 'review_stats' not in st.session_state:
            st.session_state.review_stats = {
                'total': 0,
                'reviewed': 0,
                'corrected': 0,
                'accepted': 0
            }
    
    def run(self):
        """Run the review application"""
        st.title("🍄 Truffle Literature Review Interface")
        st.markdown("Review and correct extracted experimental data from literature")
        
        # Sidebar
        self.render_sidebar()
        
        # Main content
        if st.session_state.experiments:
            self.render_main_content()
        else:
            self.render_upload_section()
    
    def render_sidebar(self):
        """Render sidebar with navigation and stats"""
        with st.sidebar:
            st.header("📊 Review Progress")
            
            # Progress bar
            if st.session_state.review_stats['total'] > 0:
                progress = st.session_state.review_stats['reviewed'] / st.session_state.review_stats['total']
                st.progress(progress)
                st.write(f"Reviewed: {st.session_state.review_stats['reviewed']}/{st.session_state.review_stats['total']}")
            
            # Statistics
            st.metric("Accepted", st.session_state.review_stats['accepted'])
            st.metric("Corrected", st.session_state.review_stats['corrected'])
            
            # Navigation
            if st.session_state.experiments:
                st.header("🧭 Navigation")
                
                # Experiment selector
                experiment_options = [f"Exp {i+1}: {exp.fungus_name} + {exp.host_name}" 
                                    for i, exp in enumerate(st.session_state.experiments)]
                
                selected_idx = st.selectbox(
                    "Select Experiment",
                    range(len(experiment_options)),
                    format_func=lambda x: experiment_options[x]
                )
                
                st.session_state.current_experiment_idx = selected_idx
                
                # Quick navigation buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("⬅️ Previous"):
                        if st.session_state.current_experiment_idx > 0:
                            st.session_state.current_experiment_idx -= 1
                            st.rerun()
                
                with col2:
                    if st.button("Next ➡️"):
                        if st.session_state.current_experiment_idx < len(st.session_state.experiments) - 1:
                            st.session_state.current_experiment_idx += 1
                            st.rerun()
            
            # Filters
            st.header("🔍 Filters")
            
            confidence_threshold = st.slider(
                "Min Confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1
            )
            
            show_reviewed = st.checkbox("Show Reviewed", value=False)
            
            # Apply filters
            if st.button("Apply Filters"):
                self.apply_filters(confidence_threshold, show_reviewed)
    
    def render_upload_section(self):
        """Render file upload section"""
        st.header("📁 Upload Data")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload extracted experiments JSON file",
            type=['json'],
            help="Upload a JSON file containing extracted experiments"
        )
        
        if uploaded_file is not None:
            try:
                data = json.load(uploaded_file)
                experiments = self.load_experiments_from_json(data)
                st.session_state.experiments = experiments
                st.session_state.review_stats['total'] = len(experiments)
                st.success(f"Loaded {len(experiments)} experiments")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading file: {e}")
        
        # Sample data button
        if st.button("Load Sample Data"):
            self.load_sample_data()
            st.rerun()
    
    def load_experiments_from_json(self, data: Dict) -> List[NormalizedExperiment]:
        """Load experiments from JSON data"""
        experiments = []
        
        for exp_data in data:
            # Convert parameters
            parameters = []
            for param_data in exp_data.get('parameters', []):
                param = NormalizedParameter(
                    parameter_name=param_data['parameter_name'],
                    original_value=param_data['original_value'],
                    original_unit=param_data['original_unit'],
                    normalized_value=param_data['normalized_value'],
                    normalized_unit=param_data['normalized_unit'],
                    confidence=param_data['confidence'],
                    validation_result=None,  # Will be set later
                    metadata=param_data.get('metadata', {})
                )
                parameters.append(param)
            
            # Create experiment
            experiment = NormalizedExperiment(
                experiment_id=exp_data['experiment_id'],
                paper_id=exp_data['paper_id'],
                fungus_taxon_id=exp_data.get('fungus_taxon_id'),
                fungus_name=exp_data['fungus_name'],
                host_taxon_id=exp_data.get('host_taxon_id'),
                host_name=exp_data['host_name'],
                inoculum_form=exp_data['inoculum_form'],
                plant_age_d=exp_data.get('plant_age_d'),
                chamber_type=exp_data['chamber_type'],
                flow_regime=exp_data['flow_regime'],
                volume_L=exp_data.get('volume_L'),
                duration_d=exp_data.get('duration_d'),
                replicates=exp_data.get('replicates'),
                colonization_pct=exp_data.get('colonization_pct'),
                time_to_colonization_d=exp_data.get('time_to_colonization_d'),
                fruiting=exp_data.get('fruiting'),
                yield_g=exp_data.get('yield_g'),
                notes=exp_data['notes'],
                confidence_0_1=exp_data['confidence_0_1'],
                parameters=parameters
            )
            experiments.append(experiment)
        
        return experiments
    
    def load_sample_data(self):
        """Load sample data for demonstration"""
        sample_experiments = [
            NormalizedExperiment(
                experiment_id="sample_001",
                paper_id="paper_001",
                fungus_name="Tuber melanosporum",
                host_name="Quercus ilex",
                inoculum_form="mycelium",
                chamber_type="hydroponic",
                flow_regime="recirculating",
                volume_L=2.0,
                duration_d=90,
                replicates=5,
                colonization_pct=85.0,
                notes="Sample experiment with high confidence",
                confidence_0_1=0.9,
                parameters=[
                    NormalizedParameter(
                        parameter_name="ph",
                        original_value=6.2,
                        original_unit="pH",
                        normalized_value=6.2,
                        normalized_unit="dimensionless",
                        confidence=0.95,
                        validation_result=None
                    ),
                    NormalizedParameter(
                        parameter_name="ec",
                        original_value=1.5,
                        original_unit="mS/cm",
                        normalized_value=1.5,
                        normalized_unit="mS/cm",
                        confidence=0.9,
                        validation_result=None
                    )
                ]
            ),
            NormalizedExperiment(
                experiment_id="sample_002",
                paper_id="paper_002",
                fungus_name="Tuber magnatum",
                host_name="Corylus avellana",
                inoculum_form="spore",
                chamber_type="aeroponic",
                flow_regime="mist",
                volume_L=1.0,
                duration_d=120,
                replicates=3,
                colonization_pct=65.0,
                notes="Sample experiment with medium confidence",
                confidence_0_1=0.6,
                parameters=[
                    NormalizedParameter(
                        parameter_name="ph",
                        original_value=5.8,
                        original_unit="pH",
                        normalized_value=5.8,
                        normalized_unit="dimensionless",
                        confidence=0.8,
                        validation_result=None
                    ),
                    NormalizedParameter(
                        parameter_name="temperature",
                        original_value=22.0,
                        original_unit="°C",
                        normalized_value=22.0,
                        normalized_unit="°C",
                        confidence=0.85,
                        validation_result=None
                    )
                ]
            )
        ]
        
        st.session_state.experiments = sample_experiments
        st.session_state.review_stats['total'] = len(sample_experiments)
    
    def render_main_content(self):
        """Render main content area"""
        if not st.session_state.experiments:
            return
        
        current_exp = st.session_state.experiments[st.session_state.current_experiment_idx]
        
        # Header
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.header(f"Experiment: {current_exp.experiment_id}")
        with col2:
            confidence_color = "green" if current_exp.confidence_0_1 > 0.8 else "orange" if current_exp.confidence_0_1 > 0.5 else "red"
            st.metric("Confidence", f"{current_exp.confidence_0_1:.2f}", delta=None)
        with col3:
            status = "✅ Reviewed" if current_exp.experiment_id in st.session_state.reviewed_experiments else "⏳ Pending"
            st.metric("Status", status)
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "🔬 Parameters", "📊 Validation", "✏️ Corrections"])
        
        with tab1:
            self.render_overview_tab(current_exp)
        
        with tab2:
            self.render_parameters_tab(current_exp)
        
        with tab3:
            self.render_validation_tab(current_exp)
        
        with tab4:
            self.render_corrections_tab(current_exp)
    
    def render_overview_tab(self, experiment: NormalizedExperiment):
        """Render overview tab"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🧬 Biological Information")
            
            # Fungus and host
            fungus_col, host_col = st.columns(2)
            with fungus_col:
                st.metric("Fungus", experiment.fungus_name)
            with host_col:
                st.metric("Host", experiment.host_name)
            
            # Inoculum and chamber
            inoculum_col, chamber_col = st.columns(2)
            with inoculum_col:
                st.metric("Inoculum", experiment.inoculum_form)
            with chamber_col:
                st.metric("Chamber", experiment.chamber_type)
            
            # Flow regime
            st.metric("Flow Regime", experiment.flow_regime)
        
        with col2:
            st.subheader("📏 Experimental Setup")
            
            # Volume and duration
            volume_col, duration_col = st.columns(2)
            with volume_col:
                st.metric("Volume (L)", experiment.volume_L or "N/A")
            with duration_col:
                st.metric("Duration (days)", experiment.duration_d or "N/A")
            
            # Replicates
            st.metric("Replicates", experiment.replicates or "N/A")
            
            # Outcomes
            outcomes_col1, outcomes_col2 = st.columns(2)
            with outcomes_col1:
                st.metric("Colonization (%)", experiment.colonization_pct or "N/A")
            with outcomes_col2:
                st.metric("Yield (g)", experiment.yield_g or "N/A")
        
        # Notes
        if experiment.notes:
            st.subheader("📝 Notes")
            st.text(experiment.notes)
    
    def render_parameters_tab(self, experiment: NormalizedExperiment):
        """Render parameters tab"""
        if not experiment.parameters:
            st.warning("No parameters found for this experiment")
            return
        
        # Parameters table
        param_data = []
        for param in experiment.parameters:
            param_data.append({
                'Parameter': param.parameter_name,
                'Original Value': param.original_value,
                'Original Unit': param.original_unit,
                'Normalized Value': param.normalized_value,
                'Normalized Unit': param.normalized_unit,
                'Confidence': f"{param.confidence:.2f}",
                'Status': "✅" if param.confidence > 0.8 else "⚠️" if param.confidence > 0.5 else "❌"
            })
        
        df = pd.DataFrame(param_data)
        st.dataframe(df, use_container_width=True)
        
        # Parameter confidence chart
        fig = px.bar(
            df, 
            x='Parameter', 
            y='Confidence',
            title="Parameter Confidence Scores",
            color='Confidence',
            color_continuous_scale=['red', 'orange', 'green']
        )
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    
    def render_validation_tab(self, experiment: NormalizedExperiment):
        """Render validation tab"""
        if not experiment.validation_result:
            st.warning("No validation results available")
            return
        
        validation = experiment.validation_result
        
        # Validation status
        col1, col2, col3 = st.columns(3)
        with col1:
            status_color = "green" if validation.is_valid else "red"
            st.metric("Valid", "✅ Yes" if validation.is_valid else "❌ No", delta=None)
        with col2:
            st.metric("Confidence", f"{validation.confidence:.2f}", delta=None)
        with col3:
            st.metric("Issues", len(validation.errors) + len(validation.warnings), delta=None)
        
        # Errors
        if validation.errors:
            st.subheader("❌ Errors")
            for error in validation.errors:
                st.error(error)
        
        # Warnings
        if validation.warnings:
            st.subheader("⚠️ Warnings")
            for warning in validation.warnings:
                st.warning(warning)
        
        # Suggestions
        if validation.suggestions:
            st.subheader("💡 Suggestions")
            for suggestion in validation.suggestions:
                st.info(suggestion)
    
    def render_corrections_tab(self, experiment: NormalizedExperiment):
        """Render corrections tab"""
        st.subheader("✏️ Make Corrections")
        
        # Basic experiment info corrections
        with st.expander("Basic Information", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                new_fungus = st.text_input("Fungus Name", value=experiment.fungus_name)
                new_host = st.text_input("Host Name", value=experiment.host_name)
                new_inoculum = st.selectbox(
                    "Inoculum Form",
                    ["spore", "mycelium", "ectomycorrhizal_rootlets", "unknown"],
                    index=["spore", "mycelium", "ectomycorrhizal_rootlets", "unknown"].index(experiment.inoculum_form)
                )
            
            with col2:
                new_chamber = st.selectbox(
                    "Chamber Type",
                    ["hydroponic", "aeroponic", "bioreactor", "petri_dish", "unknown"],
                    index=["hydroponic", "aeroponic", "bioreactor", "petri_dish", "unknown"].index(experiment.chamber_type)
                )
                new_flow = st.selectbox(
                    "Flow Regime",
                    ["static", "recirculating", "aeroponic", "mist", "unknown"],
                    index=["static", "recirculating", "aeroponic", "mist", "unknown"].index(experiment.flow_regime)
                )
        
        # Parameter corrections
        with st.expander("Parameters", expanded=True):
            for i, param in enumerate(experiment.parameters):
                st.write(f"**{param.parameter_name}**")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    new_value = st.number_input(
                        f"Value {i}",
                        value=float(param.normalized_value),
                        key=f"param_value_{i}"
                    )
                with col2:
                    new_unit = st.text_input(
                        f"Unit {i}",
                        value=param.normalized_unit,
                        key=f"param_unit_{i}"
                    )
                with col3:
                    new_confidence = st.slider(
                        f"Confidence {i}",
                        min_value=0.0,
                        max_value=1.0,
                        value=param.confidence,
                        key=f"param_confidence_{i}"
                    )
        
        # Action buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("✅ Accept", type="primary"):
                self.accept_experiment(experiment)
        
        with col2:
            if st.button("✏️ Save Corrections"):
                self.save_corrections(experiment)
        
        with col3:
            if st.button("🔄 Reset"):
                self.reset_corrections(experiment)
        
        with col4:
            if st.button("❌ Reject"):
                self.reject_experiment(experiment)
    
    def accept_experiment(self, experiment: NormalizedExperiment):
        """Accept an experiment"""
        st.session_state.reviewed_experiments.add(experiment.experiment_id)
        st.session_state.review_stats['accepted'] += 1
        st.session_state.review_stats['reviewed'] += 1
        st.success("Experiment accepted!")
        st.rerun()
    
    def save_corrections(self, experiment: NormalizedExperiment):
        """Save corrections for an experiment"""
        # Store corrections in session state
        st.session_state.corrections[experiment.experiment_id] = {
            'timestamp': datetime.now().isoformat(),
            'corrections': 'saved'  # In a real app, this would store the actual corrections
        }
        st.session_state.review_stats['corrected'] += 1
        st.success("Corrections saved!")
    
    def reset_corrections(self, experiment: NormalizedExperiment):
        """Reset corrections for an experiment"""
        if experiment.experiment_id in st.session_state.corrections:
            del st.session_state.corrections[experiment.experiment_id]
        st.success("Corrections reset!")
    
    def reject_experiment(self, experiment: NormalizedExperiment):
        """Reject an experiment"""
        st.session_state.reviewed_experiments.add(experiment.experiment_id)
        st.session_state.review_stats['reviewed'] += 1
        st.warning("Experiment rejected!")
        st.rerun()
    
    def apply_filters(self, confidence_threshold: float, show_reviewed: bool):
        """Apply filters to experiments"""
        filtered_experiments = []
        
        for exp in st.session_state.experiments:
            # Confidence filter
            if exp.confidence_0_1 < confidence_threshold:
                continue
            
            # Reviewed filter
            if not show_reviewed and exp.experiment_id in st.session_state.reviewed_experiments:
                continue
            
            filtered_experiments.append(exp)
        
        st.session_state.experiments = filtered_experiments
        st.session_state.current_experiment_idx = 0
        st.success(f"Filtered to {len(filtered_experiments)} experiments")
        st.rerun()

def main():
    """Run the review application"""
    app = ReviewApp()
    app.run()

if __name__ == "__main__":
    main()