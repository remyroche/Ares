"""
Main entry point for the Truffle Knowledge Graph and Simulation System
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import yaml
from typing import Dict, Any

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from knowledge_graph.ingestion.etl_pipeline import TruffleKGIngestion
from simulation.pde.rad_solver import create_truffle_system
from simulation.abm.hyphal_growth import HyphalGrowthABM, GrowthParameters
from simulation.control.mpc_controller import TruffleMPCController
from api.graphql.schema import get_schema
from api.sparql.endpoint import create_sparql_endpoint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('truffle_kg.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TruffleKGSystem:
    """Main system class for the Truffle Knowledge Graph and Simulation System"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize the system
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "config/config.yaml"
        self.config = self._load_config()
        
        # Initialize components
        self.etl_pipeline = None
        self.pde_solver = None
        self.abm_simulator = None
        self.mpc_controller = None
        self.sparql_endpoint = None
        
        logger.info("Truffle KG System initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return {}
    
    def setup_etl_pipeline(self):
        """Setup the ETL pipeline"""
        logger.info("Setting up ETL pipeline")
        self.etl_pipeline = TruffleKGIngestion(self.config)
        logger.info("ETL pipeline setup complete")
    
    def setup_simulation_components(self):
        """Setup simulation components"""
        logger.info("Setting up simulation components")
        
        # Setup PDE solver
        domain = tuple(self.config['simulation']['pde']['domain_size'])
        nx, ny, nz = self.config['simulation']['pde']['grid_resolution']
        dt = self.config['simulation']['pde']['time_step']
        
        self.pde_solver = create_truffle_system()
        logger.info("PDE solver setup complete")
        
        # Setup ABM simulator
        abm_domain = tuple(self.config['simulation']['abm']['domain_size'])
        growth_params = GrowthParameters(
            base_growth_rate=1.0,
            branching_rate=self.config['simulation']['abm']['branching_rate'],
            anastomosis_distance=self.config['simulation']['abm']['anastomosis_distance']
        )
        
        self.abm_simulator = HyphalGrowthABM(abm_domain, growth_params)
        logger.info("ABM simulator setup complete")
        
        # Setup MPC controller
        pred_horizon = self.config['simulation']['mpc']['prediction_horizon']
        control_horizon = self.config['simulation']['mpc']['control_horizon']
        dt_mpc = self.config['simulation']['mpc']['time_step']
        
        self.mpc_controller = TruffleMPCController(pred_horizon, control_horizon, dt_mpc)
        logger.info("MPC controller setup complete")
    
    def setup_api_endpoints(self):
        """Setup API endpoints"""
        logger.info("Setting up API endpoints")
        
        # Setup SPARQL endpoint
        rdf_file = self.config.get('databases', {}).get('rdf', {}).get('file')
        self.sparql_endpoint = create_sparql_endpoint(rdf_file)
        logger.info("SPARQL endpoint setup complete")
    
    def run_etl_pipeline(self, input_directory: str):
        """Run the ETL pipeline"""
        if not self.etl_pipeline:
            self.setup_etl_pipeline()
        
        logger.info(f"Running ETL pipeline on {input_directory}")
        self.etl_pipeline.run_pipeline(input_directory)
        logger.info("ETL pipeline completed")
    
    def run_simulation(self, duration: float = 100.0):
        """Run the simulation"""
        if not self.abm_simulator:
            self.setup_simulation_components()
        
        logger.info(f"Running simulation for {duration} time units")
        
        # Add initial hyphal tips
        import random
        for i in range(5):
            x = random.uniform(20, 30)
            y = random.uniform(20, 30)
            z = random.uniform(10, 15)
            self.abm_simulator.add_hyphal_tip(x, y, z)
        
        # Run ABM simulation
        n_steps = int(duration / self.abm_simulator.dt)
        self.abm_simulator.run(n_steps)
        
        # Get hyphal density for PDE coupling
        hyphal_density = self.abm_simulator.get_hyphal_density()
        self.pde_solver.set_hyphal_density(hyphal_density)
        
        # Run PDE simulation
        initial_concentrations = {
            "NO3": self.pde_solver.species["NO3"].initial_concentration * np.ones((self.pde_solver.ny, self.pde_solver.nx)),
            "H2PO4": self.pde_solver.species["H2PO4"].initial_concentration * np.ones((self.pde_solver.ny, self.pde_solver.nx)),
            "K": self.pde_solver.species["K"].initial_concentration * np.ones((self.pde_solver.ny, self.pde_solver.nx)),
            "Ca": self.pde_solver.species["Ca"].initial_concentration * np.ones((self.pde_solver.ny, self.pde_solver.nx)),
            "O2": self.pde_solver.species["O2"].initial_concentration * np.ones((self.pde_solver.ny, self.pde_solver.nx))
        }
        
        t_points, solution = self.pde_solver.solve(initial_concentrations, duration)
        
        logger.info("Simulation completed")
        return {
            'abm_results': self.abm_simulator,
            'pde_results': (t_points, solution),
            'hyphal_density': hyphal_density
        }
    
    def run_mpc_control(self, duration: float = 300.0):
        """Run MPC control simulation"""
        if not self.mpc_controller:
            self.setup_simulation_components()
        
        logger.info(f"Running MPC control for {duration} minutes")
        
        # Define setpoint changes
        setpoint_changes = [
            (0, {'pH': 6.0, 'EC': 1.2, 'temperature': 22.0}),
            (duration/3, {'pH': 6.2, 'EC': 1.5, 'temperature': 24.0}),
            (2*duration/3, {'pH': 5.8, 'EC': 1.0, 'temperature': 20.0})
        ]
        
        self.mpc_controller.run_closed_loop_simulation(duration, setpoint_changes)
        logger.info("MPC control completed")
    
    def start_api_servers(self):
        """Start API servers"""
        logger.info("Starting API servers")
        
        if not self.sparql_endpoint:
            self.setup_api_endpoints()
        
        # Start SPARQL endpoint
        import threading
        sparql_thread = threading.Thread(
            target=self.sparql_endpoint.run,
            kwargs={
                'host': self.config['api']['sparql']['host'],
                'port': self.config['api']['sparql']['port'],
                'debug': self.config['api']['sparql']['debug']
            }
        )
        sparql_thread.daemon = True
        sparql_thread.start()
        
        logger.info("API servers started")
    
    def create_dashboard(self):
        """Create a simple dashboard"""
        logger.info("Creating dashboard")
        
        # This would create a web dashboard
        # For now, just plot some results
        if self.abm_simulator:
            self.abm_simulator.plot_hyphae()
            self.abm_simulator.plot_statistics()
        
        if self.mpc_controller:
            self.mpc_controller.plot_results()
        
        logger.info("Dashboard created")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Truffle Knowledge Graph and Simulation System')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Configuration file path')
    parser.add_argument('--mode', type=str, choices=['etl', 'simulation', 'control', 'api', 'dashboard'], 
                       default='simulation', help='Operation mode')
    parser.add_argument('--input-dir', type=str, help='Input directory for ETL')
    parser.add_argument('--duration', type=float, default=100.0, help='Simulation duration')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host for API servers')
    parser.add_argument('--port', type=int, default=5000, help='Port for API servers')
    
    args = parser.parse_args()
    
    # Create system
    system = TruffleKGSystem(args.config)
    
    try:
        if args.mode == 'etl':
            if not args.input_dir:
                logger.error("Input directory required for ETL mode")
                sys.exit(1)
            system.run_etl_pipeline(args.input_dir)
        
        elif args.mode == 'simulation':
            results = system.run_simulation(args.duration)
            logger.info("Simulation completed successfully")
        
        elif args.mode == 'control':
            system.run_mpc_control(args.duration)
            logger.info("MPC control completed successfully")
        
        elif args.mode == 'api':
            system.start_api_servers()
            logger.info("API servers started. Press Ctrl+C to stop.")
            try:
                while True:
                    import time
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Shutting down API servers")
        
        elif args.mode == 'dashboard':
            system.setup_simulation_components()
            system.run_simulation(args.duration)
            system.create_dashboard()
        
        else:
            logger.error(f"Unknown mode: {args.mode}")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Error in {args.mode} mode: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()