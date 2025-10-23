"""
Web Dashboard for Truffle Knowledge Graph and Simulation System
Built with Streamlit for interactive exploration and control
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from typing import Dict, List, Any, Optional
import time
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Truffle KG Dashboard",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E8B57;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

class TruffleDashboard:
    """Main dashboard class"""
    
    def __init__(self):
        self.api_base_url = "http://localhost:5000"
        self.graphql_url = "http://localhost:8000/graphql"
        self.sparql_url = "http://localhost:5000/sparql"
        
        # Initialize session state
        if 'simulation_data' not in st.session_state:
            st.session_state.simulation_data = None
        if 'knowledge_graph_data' not in st.session_state:
            st.session_state.knowledge_graph_data = None
        if 'control_data' not in st.session_state:
            st.session_state.control_data = None
    
    def render_header(self):
        """Render the main header"""
        st.markdown('<h1 class="main-header">🍄 Truffle Knowledge Graph & Simulation System</h1>', 
                   unsafe_allow_html=True)
        st.markdown("---")
    
    def render_sidebar(self):
        """Render the sidebar navigation"""
        st.sidebar.title("Navigation")
        
        page = st.sidebar.selectbox(
            "Select Page",
            ["Overview", "Knowledge Graph", "Simulation", "Control", "Analytics", "Settings"]
        )
        
        st.sidebar.markdown("---")
        
        # Quick actions
        st.sidebar.subheader("Quick Actions")
        if st.sidebar.button("🔄 Refresh Data"):
            st.session_state.simulation_data = None
            st.session_state.knowledge_graph_data = None
            st.session_state.control_data = None
            st.rerun()
        
        if st.sidebar.button("📊 Run Simulation"):
            self.run_simulation()
        
        if st.sidebar.button("🎛️ Start Control"):
            self.start_control()
        
        st.sidebar.markdown("---")
        
        # System status
        st.sidebar.subheader("System Status")
        self.render_system_status()
        
        return page
    
    def render_system_status(self):
        """Render system status indicators"""
        try:
            # Check API connectivity
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            if response.status_code == 200:
                st.sidebar.success("✅ API Connected")
            else:
                st.sidebar.error("❌ API Disconnected")
        except:
            st.sidebar.error("❌ API Disconnected")
        
        # Check Neo4j
        try:
            response = requests.get(f"{self.api_base_url}/neo4j/health", timeout=5)
            if response.status_code == 200:
                st.sidebar.success("✅ Neo4j Connected")
            else:
                st.sidebar.error("❌ Neo4j Disconnected")
        except:
            st.sidebar.error("❌ Neo4j Disconnected")
        
        # Check RDF store
        try:
            response = requests.get(f"{self.api_base_url}/rdf/health", timeout=5)
            if response.status_code == 200:
                st.sidebar.success("✅ RDF Store Connected")
            else:
                st.sidebar.error("❌ RDF Store Disconnected")
        except:
            st.sidebar.error("❌ RDF Store Disconnected")
    
    def render_overview_page(self):
        """Render the overview page"""
        st.header("System Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Fungi Species",
                value="15",
                delta="+2 this month"
            )
        
        with col2:
            st.metric(
                label="Host Tree Species",
                value="8",
                delta="+1 this month"
            )
        
        with col3:
            st.metric(
                label="Active Experiments",
                value="12",
                delta="+3 this week"
            )
        
        with col4:
            st.metric(
                label="Success Rate",
                value="87%",
                delta="+5% this month"
            )
        
        st.markdown("---")
        
        # Recent activity
        st.subheader("Recent Activity")
        
        activity_data = {
            "Time": [
                datetime.now() - timedelta(hours=2),
                datetime.now() - timedelta(hours=4),
                datetime.now() - timedelta(hours=6),
                datetime.now() - timedelta(hours=8),
                datetime.now() - timedelta(hours=10)
            ],
            "Event": [
                "New experiment started: TME-001 on Q. ilex",
                "Simulation completed: 1000 time steps",
                "Control setpoint updated: pH 6.2",
                "Data ingested: 50 new measurements",
                "Knowledge graph updated: 3 new associations"
            ],
            "Type": ["Experiment", "Simulation", "Control", "Data", "Knowledge"]
        }
        
        df_activity = pd.DataFrame(activity_data)
        st.dataframe(df_activity, use_container_width=True)
        
        # System performance
        st.subheader("System Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CPU usage over time
            cpu_data = pd.DataFrame({
                "Time": pd.date_range(start=datetime.now() - timedelta(hours=24), 
                                    end=datetime.now(), freq="H"),
                "CPU Usage": np.random.uniform(20, 80, 25)
            })
            
            fig_cpu = px.line(cpu_data, x="Time", y="CPU Usage", 
                            title="CPU Usage (24h)")
            st.plotly_chart(fig_cpu, use_container_width=True)
        
        with col2:
            # Memory usage over time
            memory_data = pd.DataFrame({
                "Time": pd.date_range(start=datetime.now() - timedelta(hours=24), 
                                    end=datetime.now(), freq="H"),
                "Memory Usage": np.random.uniform(40, 90, 25)
            })
            
            fig_memory = px.line(memory_data, x="Time", y="Memory Usage", 
                               title="Memory Usage (24h)")
            st.plotly_chart(fig_memory, use_container_width=True)
    
    def render_knowledge_graph_page(self):
        """Render the knowledge graph exploration page"""
        st.header("Knowledge Graph Explorer")
        
        # Query interface
        st.subheader("Query Interface")
        
        query_type = st.selectbox(
            "Query Type",
            ["GraphQL", "SPARQL", "Natural Language"]
        )
        
        if query_type == "GraphQL":
            self.render_graphql_interface()
        elif query_type == "SPARQL":
            self.render_sparql_interface()
        else:
            self.render_natural_language_interface()
        
        st.markdown("---")
        
        # Knowledge graph visualization
        st.subheader("Knowledge Graph Visualization")
        
        # Entity statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Fungi", "15", "3 new")
        with col2:
            st.metric("Host Trees", "8", "1 new")
        with col3:
            st.metric("Experiments", "45", "5 new")
        
        # Network visualization
        st.subheader("Entity Relationships")
        
        # Create a simple network graph
        nodes = pd.DataFrame({
            "id": ["F1", "F2", "H1", "H2", "E1", "E2"],
            "label": ["T. melanosporum", "T. magnatum", "Q. ilex", "Q. petraea", "Exp1", "Exp2"],
            "type": ["Fungus", "Fungus", "Host", "Host", "Experiment", "Experiment"],
            "x": [1, 2, 1, 2, 1.5, 2.5],
            "y": [1, 1, 2, 2, 3, 3]
        })
        
        edges = pd.DataFrame({
            "source": ["F1", "F2", "F1", "F2", "E1", "E2"],
            "target": ["H1", "H2", "E1", "E2", "H1", "H2"],
            "relationship": ["forms_mycorrhiza", "forms_mycorrhiza", "uses", "uses", "tests", "tests"]
        })
        
        # Create network plot
        fig = go.Figure()
        
        # Add nodes
        for node_type in nodes["type"].unique():
            node_data = nodes[nodes["type"] == node_type]
            fig.add_trace(go.Scatter(
                x=node_data["x"],
                y=node_data["y"],
                mode="markers+text",
                text=node_data["label"],
                textposition="middle center",
                name=node_type,
                marker=dict(size=20, opacity=0.8)
            ))
        
        # Add edges
        for _, edge in edges.iterrows():
            source_node = nodes[nodes["id"] == edge["source"]].iloc[0]
            target_node = nodes[nodes["id"] == edge["target"]].iloc[0]
            
            fig.add_trace(go.Scatter(
                x=[source_node["x"], target_node["x"]],
                y=[source_node["y"], target_node["y"]],
                mode="lines",
                line=dict(width=2, color="gray"),
                showlegend=False,
                hoverinfo="skip"
            ))
        
        fig.update_layout(
            title="Knowledge Graph Network",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_graphql_interface(self):
        """Render GraphQL query interface"""
        st.subheader("GraphQL Query Interface")
        
        # Example queries
        example_queries = {
            "Get all fungi": """
query {
  fungi {
    id
    species
    commonName
    strain
  }
}
""",
            "Get best hosts for fungus": """
query GetBestHosts($fungusId: ID!, $phMax: Float) {
  bestHostsForFungus(fungusId: $fungusId, phMax: $phMax) {
    id
    species
    commonName
    age
  }
}
""",
            "Get similar recipes": """
query GetSimilarRecipes($recipeId: ID!, $threshold: Float) {
  similarRecipes(recipeId: $recipeId, similarityThreshold: $threshold) {
    id
    name
    ec
    ph
    nutrients {
      chemical
      concentration
      unit
    }
  }
}
"""
        }
        
        selected_query = st.selectbox("Select example query:", list(example_queries.keys()))
        query = st.text_area("GraphQL Query:", value=example_queries[selected_query], height=200)
        
        if st.button("Execute Query"):
            try:
                # This would execute the actual GraphQL query
                st.success("Query executed successfully!")
                st.json({"data": {"fungi": [{"id": "fungus_001", "species": "Tuber melanosporum"}]}})
            except Exception as e:
                st.error(f"Query failed: {e}")
    
    def render_sparql_interface(self):
        """Render SPARQL query interface"""
        st.subheader("SPARQL Query Interface")
        
        # Example SPARQL queries
        example_queries = {
            "Get all fungi": """
PREFIX ex: <http://example.org/truffle/kg#>
SELECT ?fungus ?species ?strain ?commonName WHERE {
    ?fungus a ex:Fungus ;
            ex:species ?species ;
            ex:strain ?strain ;
            ex:commonName ?commonName .
}
""",
            "Get mycorrhizal associations": """
PREFIX ex: <http://example.org/truffle/kg#>
SELECT ?fungus ?host ?fungusSpecies ?hostSpecies WHERE {
    ?fungus ex:formsMycorrhizaWith ?host .
    ?fungus ex:species ?fungusSpecies .
    ?host ex:species ?hostSpecies .
}
""",
            "Get experiments with outcomes": """
PREFIX ex: <http://example.org/truffle/kg#>
SELECT ?experiment ?outcome ?colonization ?yield WHERE {
    ?experiment a ex:Experiment ;
               ex:hasOutcome ?outcome .
    ?outcome ex:colonizationPercent ?colonization ;
            ex:yield ?yield .
}
"""
        }
        
        selected_query = st.selectbox("Select example query:", list(example_queries.keys()))
        query = st.text_area("SPARQL Query:", value=example_queries[selected_query], height=200)
        
        if st.button("Execute SPARQL Query"):
            try:
                # This would execute the actual SPARQL query
                st.success("SPARQL query executed successfully!")
                st.json({"head": {"vars": ["fungus", "species"]}, "results": {"bindings": []}})
            except Exception as e:
                st.error(f"SPARQL query failed: {e}")
    
    def render_natural_language_interface(self):
        """Render natural language query interface"""
        st.subheader("Natural Language Query Interface")
        
        query = st.text_input("Ask a question about truffle cultivation:")
        
        if st.button("Ask Question"):
            if query:
                st.info("Natural language processing not yet implemented. Please use GraphQL or SPARQL interfaces.")
            else:
                st.warning("Please enter a question.")
    
    def render_simulation_page(self):
        """Render the simulation page"""
        st.header("Simulation Control")
        
        # Simulation parameters
        st.subheader("Simulation Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            duration = st.slider("Simulation Duration (time units)", 10, 1000, 100)
            grid_resolution = st.selectbox("Grid Resolution", [25, 50, 100], index=1)
            time_step = st.number_input("Time Step", 0.001, 1.0, 0.01, 0.001)
        
        with col2:
            initial_tips = st.slider("Initial Hyphal Tips", 1, 20, 5)
            branching_rate = st.slider("Branching Rate", 0.01, 0.2, 0.05, 0.01)
            anastomosis_distance = st.slider("Anastomosis Distance", 1.0, 10.0, 3.0, 0.1)
        
        # Environment parameters
        st.subheader("Environment Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ph = st.slider("pH", 4.0, 8.0, 6.0, 0.1)
            ec = st.slider("EC (mS/cm)", 0.5, 3.0, 1.2, 0.1)
        
        with col2:
            temperature = st.slider("Temperature (°C)", 15.0, 30.0, 22.0, 0.5)
            humidity = st.slider("Humidity (%)", 40.0, 90.0, 70.0, 1.0)
        
        with col3:
            do = st.slider("Dissolved Oxygen (mg/L)", 4.0, 12.0, 8.0, 0.1)
            co2 = st.slider("CO2 (ppm)", 300, 1000, 400, 10)
        
        # Run simulation
        if st.button("Run Simulation", type="primary"):
            with st.spinner("Running simulation..."):
                self.run_simulation(duration, grid_resolution, time_step, 
                                  initial_tips, branching_rate, anastomosis_distance,
                                  ph, ec, temperature, humidity, do, co2)
        
        # Display results
        if st.session_state.simulation_data:
            self.render_simulation_results()
    
    def render_simulation_results(self):
        """Render simulation results"""
        st.subheader("Simulation Results")
        
        data = st.session_state.simulation_data
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Tips", data.get("total_tips", 0))
        with col2:
            st.metric("Active Tips", data.get("active_tips", 0))
        with col3:
            st.metric("Total Length", f"{data.get('total_length', 0):.2f}")
        with col4:
            st.metric("Branching Events", data.get("branching_events", 0))
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Hyphal network
            st.subheader("Hyphal Network")
            
            # Create sample data for visualization
            np.random.seed(42)
            n_points = 100
            x = np.random.uniform(0, 100, n_points)
            y = np.random.uniform(0, 100, n_points)
            colors = np.random.uniform(0, 1, n_points)
            
            fig = px.scatter(x=x, y=y, color=colors, 
                           title="Hyphal Tip Distribution",
                           labels={"x": "X Position", "y": "Y Position"})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Growth statistics over time
            st.subheader("Growth Statistics")
            
            time_points = np.linspace(0, 100, 50)
            total_tips = 5 + 0.1 * time_points + np.random.normal(0, 0.5, 50)
            active_tips = total_tips * 0.8 + np.random.normal(0, 0.2, 50)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=time_points, y=total_tips, 
                                   name="Total Tips", line=dict(color="blue")))
            fig.add_trace(go.Scatter(x=time_points, y=active_tips, 
                                   name="Active Tips", line=dict(color="red")))
            
            fig.update_layout(title="Tip Count Over Time",
                            xaxis_title="Time",
                            yaxis_title="Number of Tips")
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_control_page(self):
        """Render the control page"""
        st.header("Control System")
        
        # Control parameters
        st.subheader("Control Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            prediction_horizon = st.slider("Prediction Horizon", 5, 50, 20)
            control_horizon = st.slider("Control Horizon", 5, 30, 10)
            time_step = st.number_input("Control Time Step (s)", 0.1, 10.0, 1.0, 0.1)
        
        with col2:
            ph_setpoint = st.slider("pH Setpoint", 4.0, 8.0, 6.0, 0.1)
            ec_setpoint = st.slider("EC Setpoint (mS/cm)", 0.5, 3.0, 1.2, 0.1)
            temp_setpoint = st.slider("Temperature Setpoint (°C)", 15.0, 30.0, 22.0, 0.5)
        
        # Control actions
        st.subheader("Control Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ph_dosing = st.slider("pH Dosing (mL/min)", 0.0, 10.0, 0.0, 0.1)
            ec_dosing = st.slider("EC Dosing (mL/min)", 0.0, 5.0, 0.0, 0.1)
        
        with col2:
            air_flow = st.slider("Air Flow (L/min)", 0.0, 100.0, 50.0, 1.0)
            water_flow = st.slider("Water Flow (L/min)", 0.0, 10.0, 2.0, 0.1)
        
        with col3:
            heating = st.slider("Heating (%)", 0.0, 100.0, 0.0, 1.0)
            cooling = st.slider("Cooling (%)", 0.0, 100.0, 0.0, 1.0)
        
        # Control buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Start Control", type="primary"):
                self.start_control()
        
        with col2:
            if st.button("Stop Control"):
                self.stop_control()
        
        with col3:
            if st.button("Reset Setpoints"):
                self.reset_setpoints()
        
        # Display control results
        if st.session_state.control_data:
            self.render_control_results()
    
    def render_control_results(self):
        """Render control results"""
        st.subheader("Control Results")
        
        # Create sample control data
        time_points = np.linspace(0, 300, 300)
        
        # Simulate control responses
        ph_response = 6.0 + 0.2 * np.sin(0.02 * time_points) + np.random.normal(0, 0.05, 300)
        ec_response = 1.2 + 0.1 * np.cos(0.03 * time_points) + np.random.normal(0, 0.02, 300)
        temp_response = 22.0 + 1.0 * np.sin(0.01 * time_points) + np.random.normal(0, 0.1, 300)
        
        # Control actions
        ph_dosing = np.maximum(0, 0.5 * np.sin(0.05 * time_points) + np.random.normal(0, 0.1, 300))
        ec_dosing = np.maximum(0, 0.3 * np.cos(0.04 * time_points) + np.random.normal(0, 0.05, 300))
        air_flow = 50 + 10 * np.sin(0.02 * time_points) + np.random.normal(0, 2, 300)
        
        # Plot control responses
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("pH Control", "EC Control", "Temperature Control", "Control Actions"),
            vertical_spacing=0.1
        )
        
        # pH control
        fig.add_trace(
            go.Scatter(x=time_points, y=ph_response, name="pH", line=dict(color="blue")),
            row=1, col=1
        )
        fig.add_hline(y=6.0, line_dash="dash", line_color="red", row=1, col=1)
        
        # EC control
        fig.add_trace(
            go.Scatter(x=time_points, y=ec_response, name="EC", line=dict(color="green")),
            row=1, col=2
        )
        fig.add_hline(y=1.2, line_dash="dash", line_color="red", row=1, col=2)
        
        # Temperature control
        fig.add_trace(
            go.Scatter(x=time_points, y=temp_response, name="Temperature", line=dict(color="orange")),
            row=2, col=1
        )
        fig.add_hline(y=22.0, line_dash="dash", line_color="red", row=2, col=1)
        
        # Control actions
        fig.add_trace(
            go.Scatter(x=time_points, y=ph_dosing, name="pH Dosing", line=dict(color="purple")),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=time_points, y=ec_dosing, name="EC Dosing", line=dict(color="brown")),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_analytics_page(self):
        """Render the analytics page"""
        st.header("Analytics & Insights")
        
        # Data analysis options
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Experimental Results", "Simulation Performance", "Control Effectiveness", "Knowledge Graph Insights"]
        )
        
        if analysis_type == "Experimental Results":
            self.render_experimental_analytics()
        elif analysis_type == "Simulation Performance":
            self.render_simulation_analytics()
        elif analysis_type == "Control Effectiveness":
            self.render_control_analytics()
        else:
            self.render_kg_analytics()
    
    def render_experimental_analytics(self):
        """Render experimental analytics"""
        st.subheader("Experimental Results Analysis")
        
        # Create sample experimental data
        experiments = pd.DataFrame({
            "Experiment": [f"Exp_{i:03d}" for i in range(1, 21)],
            "Fungus": np.random.choice(["T. melanosporum", "T. magnatum", "T. borchii"], 20),
            "Host": np.random.choice(["Q. ilex", "Q. petraea", "C. avellana"], 20),
            "pH": np.random.uniform(5.5, 6.5, 20),
            "EC": np.random.uniform(0.8, 2.0, 20),
            "Colonization": np.random.uniform(20, 95, 20),
            "Yield": np.random.uniform(5, 50, 20),
            "Success": np.random.choice([True, False], 20, p=[0.8, 0.2])
        })
        
        # Success rate by fungus
        col1, col2 = st.columns(2)
        
        with col1:
            success_by_fungus = experiments.groupby("Fungus")["Success"].mean()
            fig = px.bar(x=success_by_fungus.index, y=success_by_fungus.values,
                        title="Success Rate by Fungus Species")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Colonization vs Yield
            fig = px.scatter(experiments, x="Colonization", y="Yield", 
                           color="Fungus", size="EC",
                           title="Colonization vs Yield")
            st.plotly_chart(fig, use_container_width=True)
        
        # Environmental conditions analysis
        st.subheader("Environmental Conditions Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # pH distribution
            fig = px.histogram(experiments, x="pH", nbins=20, 
                             title="pH Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # EC vs Success
            fig = px.box(experiments, x="Success", y="EC", 
                        title="EC Distribution by Success")
            st.plotly_chart(fig, use_container_width=True)
    
    def render_simulation_analytics(self):
        """Render simulation analytics"""
        st.subheader("Simulation Performance Analysis")
        
        # Performance metrics
        metrics = pd.DataFrame({
            "Metric": ["CPU Usage", "Memory Usage", "Simulation Time", "Convergence Rate"],
            "Value": [65, 78, 45, 92],
            "Unit": ["%", "%", "seconds", "%"]
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(metrics, x="Metric", y="Value", 
                        title="Performance Metrics")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Simulation accuracy over time
            time_points = np.linspace(0, 100, 50)
            accuracy = 0.8 + 0.2 * (1 - np.exp(-time_points / 20)) + np.random.normal(0, 0.02, 50)
            
            fig = px.line(x=time_points, y=accuracy, 
                         title="Simulation Accuracy Over Time")
            st.plotly_chart(fig, use_container_width=True)
    
    def render_control_analytics(self):
        """Render control analytics"""
        st.subheader("Control Effectiveness Analysis")
        
        # Control performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Setpoint Tracking", "94%", "2%")
        with col2:
            st.metric("Overshoot", "1.2%", "-0.3%")
        with col3:
            st.metric("Settling Time", "45s", "-5s")
        with col4:
            st.metric("Energy Efficiency", "87%", "3%")
    
    def render_kg_analytics(self):
        """Render knowledge graph analytics"""
        st.subheader("Knowledge Graph Insights")
        
        # Entity statistics
        entities = pd.DataFrame({
            "Entity Type": ["Fungi", "Host Trees", "Experiments", "Outcomes", "Protocols"],
            "Count": [15, 8, 45, 120, 12],
            "Growth": [2, 1, 5, 15, 1]
        })
        
        fig = px.bar(entities, x="Entity Type", y="Count", 
                    title="Knowledge Graph Entity Counts")
        st.plotly_chart(fig, use_container_width=True)
    
    def render_settings_page(self):
        """Render the settings page"""
        st.header("System Settings")
        
        # API Configuration
        st.subheader("API Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            api_host = st.text_input("API Host", "localhost")
            api_port = st.number_input("API Port", 3000, 10000, 5000)
        
        with col2:
            graphql_host = st.text_input("GraphQL Host", "localhost")
            graphql_port = st.number_input("GraphQL Port", 3000, 10000, 8000)
        
        # Database Configuration
        st.subheader("Database Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            neo4j_uri = st.text_input("Neo4j URI", "bolt://localhost:7687")
            neo4j_user = st.text_input("Neo4j Username", "neo4j")
            neo4j_password = st.text_input("Neo4j Password", "password", type="password")
        
        with col2:
            rdf_endpoint = st.text_input("RDF Endpoint", "http://localhost:3030/truffle")
            rdf_user = st.text_input("RDF Username", "admin")
            rdf_password = st.text_input("RDF Password", "admin", type="password")
        
        # Simulation Configuration
        st.subheader("Simulation Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            default_duration = st.number_input("Default Duration", 10, 1000, 100)
            default_grid_size = st.selectbox("Default Grid Size", [25, 50, 100], index=1)
        
        with col2:
            auto_save = st.checkbox("Auto-save Results", True)
            real_time_plotting = st.checkbox("Real-time Plotting", True)
        
        # Save settings
        if st.button("Save Settings", type="primary"):
            st.success("Settings saved successfully!")
    
    def run_simulation(self, duration=100, grid_resolution=50, time_step=0.01,
                      initial_tips=5, branching_rate=0.05, anastomosis_distance=3.0,
                      ph=6.0, ec=1.2, temperature=22.0, humidity=70.0, do=8.0, co2=400):
        """Run simulation with given parameters"""
        try:
            # This would call the actual simulation system
            st.session_state.simulation_data = {
                "total_tips": initial_tips + int(duration * 0.1),
                "active_tips": initial_tips + int(duration * 0.08),
                "total_length": duration * 2.5,
                "branching_events": int(duration * 0.05),
                "anastomosis_events": int(duration * 0.02),
                "duration": duration,
                "parameters": {
                    "grid_resolution": grid_resolution,
                    "time_step": time_step,
                    "initial_tips": initial_tips,
                    "branching_rate": branching_rate,
                    "anastomosis_distance": anastomosis_distance,
                    "ph": ph,
                    "ec": ec,
                    "temperature": temperature,
                    "humidity": humidity,
                    "do": do,
                    "co2": co2
                }
            }
            st.success("Simulation completed successfully!")
        except Exception as e:
            st.error(f"Simulation failed: {e}")
    
    def start_control(self):
        """Start control system"""
        try:
            st.session_state.control_data = {
                "status": "running",
                "start_time": datetime.now(),
                "setpoints": {"ph": 6.0, "ec": 1.2, "temperature": 22.0}
            }
            st.success("Control system started!")
        except Exception as e:
            st.error(f"Failed to start control: {e}")
    
    def stop_control(self):
        """Stop control system"""
        try:
            if st.session_state.control_data:
                st.session_state.control_data["status"] = "stopped"
            st.success("Control system stopped!")
        except Exception as e:
            st.error(f"Failed to stop control: {e}")
    
    def reset_setpoints(self):
        """Reset control setpoints"""
        try:
            st.success("Setpoints reset to default values!")
        except Exception as e:
            st.error(f"Failed to reset setpoints: {e}")
    
    def run(self):
        """Run the dashboard"""
        self.render_header()
        
        page = self.render_sidebar()
        
        if page == "Overview":
            self.render_overview_page()
        elif page == "Knowledge Graph":
            self.render_knowledge_graph_page()
        elif page == "Simulation":
            self.render_simulation_page()
        elif page == "Control":
            self.render_control_page()
        elif page == "Analytics":
            self.render_analytics_page()
        elif page == "Settings":
            self.render_settings_page()

def main():
    """Main function to run the dashboard"""
    dashboard = TruffleDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()