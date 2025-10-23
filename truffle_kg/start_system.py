#!/usr/bin/env python3
"""
Startup script for the Truffle Knowledge Graph and Simulation System
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path
import yaml
import signal
import threading
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TruffleSystemManager:
    """Manages the startup and shutdown of all system components"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.processes: List[subprocess.Popen] = []
        self.running = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
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
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()
        sys.exit(0)
    
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are available"""
        logger.info("Checking dependencies...")
        
        required_packages = [
            'numpy', 'scipy', 'pandas', 'matplotlib', 'plotly',
            'flask', 'graphene', 'rdflib', 'neo4j', 'streamlit'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing required packages: {missing_packages}")
            logger.error("Please install missing packages with: pip install -r requirements.txt")
            return False
        
        logger.info("All dependencies are available")
        return True
    
    def check_databases(self) -> bool:
        """Check if required databases are running"""
        logger.info("Checking database connectivity...")
        
        # Check Neo4j
        try:
            import neo4j
            driver = neo4j.GraphDatabase.driver(
                self.config['databases']['neo4j']['uri'],
                auth=(self.config['databases']['neo4j']['user'], 
                      self.config['databases']['neo4j']['password'])
            )
            with driver.session() as session:
                session.run("RETURN 1")
            driver.close()
            logger.info("Neo4j is running")
        except Exception as e:
            logger.warning(f"Neo4j not available: {e}")
            logger.warning("Please start Neo4j: neo4j start")
        
        # Check RDF store (optional)
        try:
            import requests
            response = requests.get(self.config['databases']['rdf']['endpoint'], timeout=5)
            if response.status_code == 200:
                logger.info("RDF store is running")
            else:
                logger.warning("RDF store not responding")
        except Exception as e:
            logger.warning(f"RDF store not available: {e}")
            logger.warning("Please start your RDF store (e.g., Apache Jena Fuseki)")
        
        return True
    
    def start_sparql_endpoint(self):
        """Start the SPARQL endpoint"""
        logger.info("Starting SPARQL endpoint...")
        
        cmd = [
            sys.executable, "-m", "api.sparql.endpoint",
            "--host", self.config['api']['sparql']['host'],
            "--port", str(self.config['api']['sparql']['port'])
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path(__file__).parent
        )
        
        self.processes.append(process)
        logger.info(f"SPARQL endpoint started on {self.config['api']['sparql']['host']}:{self.config['api']['sparql']['port']}")
    
    def start_graphql_endpoint(self):
        """Start the GraphQL endpoint"""
        logger.info("Starting GraphQL endpoint...")
        
        cmd = [
            sys.executable, "-m", "api.graphql.server",
            "--host", self.config['api']['graphql']['host'],
            "--port", str(self.config['api']['graphql']['port'])
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path(__file__).parent
        )
        
        self.processes.append(process)
        logger.info(f"GraphQL endpoint started on {self.config['api']['graphql']['host']}:{self.config['api']['graphql']['port']}")
    
    def start_dashboard(self):
        """Start the Streamlit dashboard"""
        logger.info("Starting dashboard...")
        
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            "dashboard/frontend/app.py",
            "--server.port", str(self.config['visualization']['dashboard']['port']),
            "--server.address", self.config['visualization']['dashboard']['host']
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path(__file__).parent
        )
        
        self.processes.append(process)
        logger.info(f"Dashboard started on {self.config['visualization']['dashboard']['host']}:{self.config['visualization']['dashboard']['port']}")
    
    def start_monitoring(self):
        """Start monitoring services"""
        logger.info("Starting monitoring services...")
        
        # Start Prometheus metrics endpoint (if configured)
        if self.config.get('monitoring', {}).get('metrics', {}).get('enabled', False):
            cmd = [
                sys.executable, "-m", "monitoring.metrics_server",
                "--port", str(self.config['monitoring']['metrics']['port'])
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path(__file__).parent
            )
            
            self.processes.append(process)
            logger.info("Monitoring services started")
    
    def initialize_knowledge_graph(self):
        """Initialize the knowledge graph with sample data"""
        logger.info("Initializing knowledge graph...")
        
        try:
            from knowledge_graph.ingestion.etl_pipeline import TruffleKGIngestion
            
            # Create sample data directory if it doesn't exist
            sample_data_dir = Path("data/sample")
            sample_data_dir.mkdir(parents=True, exist_ok=True)
            
            # Create sample data files
            self._create_sample_data(sample_data_dir)
            
            # Run ETL pipeline
            pipeline = TruffleKGIngestion(self.config)
            pipeline.run_pipeline(str(sample_data_dir))
            
            logger.info("Knowledge graph initialized with sample data")
        except Exception as e:
            logger.error(f"Failed to initialize knowledge graph: {e}")
    
    def _create_sample_data(self, data_dir: Path):
        """Create sample data for initialization"""
        # Create sample lab data
        lab_data_dir = data_dir / "lab_data"
        lab_data_dir.mkdir(exist_ok=True)
        
        import pandas as pd
        import numpy as np
        
        # Sample experiment data
        experiments = pd.DataFrame({
            "experiment_id": [f"exp_{i:03d}" for i in range(1, 11)],
            "fungus_species": np.random.choice(["Tuber melanosporum", "Tuber magnatum", "Tuber borchii"], 10),
            "host_species": np.random.choice(["Quercus ilex", "Quercus petraea", "Corylus avellana"], 10),
            "ph": np.random.uniform(5.5, 6.5, 10),
            "ec": np.random.uniform(0.8, 2.0, 10),
            "temperature": np.random.uniform(20, 25, 10),
            "colonization_percent": np.random.uniform(30, 95, 10),
            "yield": np.random.uniform(5, 50, 10),
            "success": np.random.choice([True, False], 10, p=[0.8, 0.2])
        })
        
        experiments.to_csv(lab_data_dir / "experiments.csv", index=False)
        
        # Sample sensor data
        sensor_data_dir = data_dir / "sensor_data"
        sensor_data_dir.mkdir(exist_ok=True)
        
        time_points = pd.date_range(start="2024-01-01", end="2024-01-07", freq="H")
        sensor_data = pd.DataFrame({
            "timestamp": time_points,
            "ph": 6.0 + 0.2 * np.sin(2 * np.pi * np.arange(len(time_points)) / 24),
            "ec": 1.2 + 0.1 * np.cos(2 * np.pi * np.arange(len(time_points)) / 24),
            "temperature": 22.0 + 2.0 * np.sin(2 * np.pi * np.arange(len(time_points)) / 24),
            "humidity": 70.0 + 10.0 * np.sin(2 * np.pi * np.arange(len(time_points)) / 24),
            "do": 8.0 + 1.0 * np.sin(2 * np.pi * np.arange(len(time_points)) / 24)
        })
        
        sensor_data.to_csv(sensor_data_dir / "sensor_readings.csv", index=False)
        
        logger.info("Sample data created")
    
    def start(self):
        """Start all system components"""
        logger.info("Starting Truffle Knowledge Graph and Simulation System...")
        
        # Check dependencies
        if not self.check_dependencies():
            logger.error("Dependency check failed")
            return False
        
        # Check databases
        self.check_databases()
        
        # Initialize knowledge graph
        self.initialize_knowledge_graph()
        
        # Start services
        try:
            self.start_sparql_endpoint()
            time.sleep(2)  # Give services time to start
            
            self.start_graphql_endpoint()
            time.sleep(2)
            
            self.start_dashboard()
            time.sleep(2)
            
            self.start_monitoring()
            
            self.running = True
            logger.info("All services started successfully!")
            logger.info("Dashboard available at: http://localhost:8080")
            logger.info("SPARQL endpoint available at: http://localhost:5000/sparql")
            logger.info("GraphQL endpoint available at: http://localhost:8000/graphql")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start services: {e}")
            self.shutdown()
            return False
    
    def shutdown(self):
        """Shutdown all system components"""
        logger.info("Shutting down system...")
        
        self.running = False
        
        # Terminate all processes
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            except Exception as e:
                logger.warning(f"Error terminating process: {e}")
        
        self.processes.clear()
        logger.info("System shutdown complete")
    
    def run(self):
        """Run the system manager"""
        if not self.start():
            logger.error("Failed to start system")
            return False
        
        try:
            # Keep running until interrupted
            while self.running:
                time.sleep(1)
                
                # Check if any processes have died
                for i, process in enumerate(self.processes):
                    if process.poll() is not None:
                        logger.warning(f"Process {i} has died, restarting...")
                        # Restart logic could be added here
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        finally:
            self.shutdown()
        
        return True

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Truffle Knowledge Graph and Simulation System Manager')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Configuration file path')
    parser.add_argument('--init-only', action='store_true', help='Only initialize the knowledge graph')
    parser.add_argument('--no-dashboard', action='store_true', help='Start without dashboard')
    
    args = parser.parse_args()
    
    manager = TruffleSystemManager(args.config)
    
    if args.init_only:
        manager.initialize_knowledge_graph()
        return
    
    if args.no_dashboard:
        # Modify config to skip dashboard
        manager.config['visualization']['dashboard']['enabled'] = False
    
    success = manager.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()