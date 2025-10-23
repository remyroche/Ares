# Truffle Knowledge Graph and Simulation System

A comprehensive IT support system for R&D regarding growing truffles hydroponically, featuring a knowledge graph, simulation software, and AI-assisted control systems.

## 🌟 Features

### Knowledge Graph (KG)
- **Unified Data Model**: Integrates scattered facts about fungi, hosts, nutrients, environment, protocols, and outcomes
- **Causal Queries**: Ask compositional questions like "Which Tuber strains colonize Quercus ilex below pH 6.2 under low N, and what inoculation methods worked?"
- **Dual Storage**: RDF/OWL for interoperability + Neo4j property graph for fast exploration
- **Ontology Alignment**: Compatible with NCBI Taxon, ChEBI, ENVO, PO/PATO, PROV-O, and QUDT/OM

### Simulation Software
- **Reaction-Advection-Diffusion (RAD) PDEs**: Simulates nutrient and signal transport
- **Agent-Based Model (ABM)**: Models individual hyphal tips with growth, branching, and tropism
- **Model Predictive Control (MPC)**: Controls pH, EC, DO, temperature, and flow rates
- **Surrogate Models**: AI-accelerated design-space search and control optimization

### API Layer
- **GraphQL Interface**: Unified query interface for both RDF and Neo4j data
- **SPARQL Endpoint**: Direct RDF querying with reasoning capabilities
- **REST API**: Standard REST endpoints for integration

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd truffle_kg
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Install additional dependencies**:
```bash
# Install spaCy model
python -m spacy download en_core_sci_sm

# Install Neo4j (if not already installed)
# Follow Neo4j installation guide for your OS
```

4. **Setup databases**:
```bash
# Start Neo4j
neo4j start

# Start RDF store (e.g., Apache Jena Fuseki)
# Follow Fuseki installation guide
```

### Configuration

1. **Edit configuration**:
```bash
cp config/config.yaml.example config/config.yaml
# Edit config/config.yaml with your database credentials
```

2. **Initialize knowledge graph**:
```bash
python main.py --mode etl --input-dir data/input
```

### Running the System

1. **Run simulation**:
```bash
python main.py --mode simulation --duration 100
```

2. **Run control system**:
```bash
python main.py --mode control --duration 300
```

3. **Start API servers**:
```bash
python main.py --mode api
```

4. **Create dashboard**:
```bash
python main.py --mode dashboard --duration 100
```

## 📊 System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   ETL Pipeline  │    │  Knowledge Graph│
│                 │    │                 │    │                 │
│ • PDF Papers    │───▶│ • NLP Processing│───▶│ • RDF/OWL Store │
│ • Lab CSVs      │    │ • Normalization │    │ • Neo4j Graph   │
│ • Sensor Data   │    │ • Validation    │    │ • Provenance    │
│ • Images        │    │ • Integration   │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Simulation    │    │   Control       │    │   API Layer     │
│                 │    │                 │    │                 │
│ • RAD PDEs      │◀───│ • MPC Controller│◀───│ • GraphQL       │
│ • ABM Hyphae    │    │ • Setpoint Mgmt │    │ • SPARQL        │
│ • Root Models   │    │ • Optimization  │    │ • REST          │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔬 Core Components

### Knowledge Graph Schema

The system uses a comprehensive ontology with the following main entities:

- **Fungus**: Species, strain, genotype, mating type, culture history
- **HostTree**: Species, age, rootstock, root architecture
- **Mycorrhiza**: Association between fungus and host tree
- **NutrientRecipe**: Macro/micro nutrients, chelators, carbon sources
- **Environment**: pH, EC, DO, CO₂, temperature, humidity, light, flow
- **Protocol**: Inoculation, sterilization, biofilm/scaffold methods
- **Experiment**: Design, replicates, sensors, measurement schedule
- **Outcome**: Colonization %, hyphal density, yield, primordia
- **Evidence**: Provenance, calibration, confidence, methods

### Simulation Models

#### 1. Reaction-Advection-Diffusion (RAD) PDEs
```python
∂S/∂t = D_S ∇²S - u·∇S - k_uptake(x,t)S + R(S)
```
- Simulates nutrient transport in hydroponic systems
- Couples with hyphal density from ABM
- Handles multiple species (NO₃⁻, H₂PO₄⁻, K⁺, Ca²⁺, O₂)

#### 2. Agent-Based Model (ABM)
```python
x_{t+Δt} = x_t + v(∇C)ẑΔt + σN(0,I)
```
- Individual hyphal tips as agents
- Chemotaxis, thigmotaxis, and pH tropism
- Branching and anastomosis behaviors

#### 3. Model Predictive Control (MPC)
- Controls pH, EC, DO, temperature, flow rates
- Optimizes setpoints while minimizing stress
- Handles constraints and disturbances

### API Queries

#### GraphQL Examples
```graphql
# Get best hosts for a fungus
query GetBestHosts($fungusId: ID!, $phMax: Float) {
  bestHostsForFungus(fungusId: $fungusId, phMax: $phMax) {
    id
    species
    commonName
    age
  }
}

# Find similar nutrient recipes
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
```

#### SPARQL Examples
```sparql
# Find best hosts for Tuber melanosporum at pH ≤ 6.2
PREFIX ex: <http://example.org/truffle/kg#>
SELECT ?host ?method (AVG(?colonPct) AS ?avgColon) WHERE {
  ?myc ex:ofFungus ex:Tuber_melanosporum ;
       ex:withHost ?host ;
       ex:observedUnder ?env ;
       ex:measuredBy ?outcome .
  ?env ex:pH ?pH .
  FILTER(?pH <= 6.2)
  ?outcome ex:colonizationPercent ?colonPct ;
           ex:fromProtocol ?prot .
  ?prot ex:inoculationMethod ?method .
}
GROUP BY ?host ?method
ORDER BY DESC(?avgColon)
```

## 🧪 Example Workflows

### 1. Data Ingestion and Knowledge Graph Population

```python
from knowledge_graph.ingestion.etl_pipeline import TruffleKGIngestion

# Initialize ETL pipeline
pipeline = TruffleKGIngestion(config)

# Ingest data from various sources
pipeline.run_pipeline("data/input")
```

### 2. Simulation and Control

```python
from simulation.abm.hyphal_growth import HyphalGrowthABM
from simulation.pde.rad_solver import create_truffle_system
from simulation.control.mpc_controller import TruffleMPCController

# Setup simulation components
abm = HyphalGrowthABM(domain_size=(100, 100, 50))
pde_solver = create_truffle_system()
mpc = TruffleMPCController(prediction_horizon=20, control_horizon=10)

# Run coupled simulation
abm.run(1000)  # Run ABM
hyphal_density = abm.get_hyphal_density()
pde_solver.set_hyphal_density(hyphal_density)
t_points, solution = pde_solver.solve(initial_concentrations, 100.0)

# Run control simulation
mpc.run_closed_loop_simulation(duration=300)
```

### 3. Querying the Knowledge Graph

```python
# GraphQL query
query = """
query {
  bestHostsForFungus(fungusId: "fungus_001", phMax: 6.2) {
    id
    species
    commonName
  }
}
"""

# SPARQL query
sparql_query = """
PREFIX ex: <http://example.org/truffle/kg#>
SELECT ?fungus ?host WHERE {
  ?fungus ex:formsMycorrhizaWith ?host .
  ?fungus ex:species "Tuber melanosporum" .
}
"""
```

## 📈 Performance and Scaling

### System Requirements
- **Minimum**: 8GB RAM, 4 CPU cores, 50GB storage
- **Recommended**: 32GB RAM, 16 CPU cores, 500GB SSD storage
- **Production**: 64GB RAM, 32 CPU cores, 1TB NVMe storage

### Performance Metrics
- **Knowledge Graph**: 10,000+ entities, <100ms query response
- **Simulation**: 1,000+ hyphal tips, real-time PDE solving
- **Control**: <1s MPC optimization, <100ms setpoint updates

### Scaling Options
- **Horizontal**: Multiple API instances behind load balancer
- **Vertical**: Increase resources for single instance
- **Database**: Neo4j clustering, RDF store federation

## 🔧 Configuration

### Environment Variables
```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password"
export RDF_ENDPOINT="http://localhost:3030/truffle"
export LOG_LEVEL="INFO"
```

### Configuration File
See `config/config.yaml` for detailed configuration options including:
- Database connections
- API endpoints
- Simulation parameters
- ML model settings
- Security and monitoring

## 🧪 Testing

### Unit Tests
```bash
pytest tests/unit/
```

### Integration Tests
```bash
pytest tests/integration/
```

### End-to-End Tests
```bash
pytest tests/e2e/
```

### Coverage Report
```bash
pytest --cov=truffle_kg --cov-report=html
```

## 📚 Documentation

### API Documentation
- **GraphQL**: Available at `http://localhost:8000/graphql`
- **SPARQL**: Available at `http://localhost:5000/sparql`
- **REST**: Available at `http://localhost:3000/api`

### Code Documentation
```bash
# Generate documentation
sphinx-build -b html docs/ docs/_build/html
```

### Example Queries
See `api/graphql/examples.py` and `api/sparql/examples.py` for comprehensive query examples.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
black truffle_kg/
flake8 truffle_kg/
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Scientific Community**: For truffle cultivation research and data
- **Open Source Libraries**: RDFLib, Neo4j, CasADi, SciPy, and many others
- **Research Partners**: Collaborating institutions and researchers

## 📞 Support

- **Documentation**: [Wiki](https://github.com/your-org/truffle-kg/wiki)
- **Issues**: [GitHub Issues](https://github.com/your-org/truffle-kg/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/truffle-kg/discussions)
- **Email**: support@truffle-kg.com

## 🔮 Roadmap

### Phase 1 (Current)
- [x] Core knowledge graph schema
- [x] Basic simulation components
- [x] API endpoints
- [x] ETL pipeline

### Phase 2 (Next 3 months)
- [ ] Advanced ML models
- [ ] Web dashboard
- [ ] Real-time monitoring
- [ ] Mobile app

### Phase 3 (Next 6 months)
- [ ] Cloud deployment
- [ ] Multi-tenant support
- [ ] Advanced analytics
- [ ] Integration with lab equipment

---

**Built with ❤️ for the future of truffle cultivation**