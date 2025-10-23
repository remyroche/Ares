// Neo4j Property Graph Schema for Truffle Knowledge Graph
// This mirrors the RDF/OWL schema for fast exploration and path queries

// Create constraints and indexes
CREATE CONSTRAINT fungus_id IF NOT EXISTS FOR (f:Fungus) REQUIRE f.id IS UNIQUE;
CREATE CONSTRAINT host_tree_id IF NOT EXISTS FOR (h:HostTree) REQUIRE h.id IS UNIQUE;
CREATE CONSTRAINT mycorrhiza_id IF NOT EXISTS FOR (m:Mycorrhiza) REQUIRE m.id IS UNIQUE;
CREATE CONSTRAINT nutrient_recipe_id IF NOT EXISTS FOR (r:NutrientRecipe) REQUIRE r.id IS UNIQUE;
CREATE CONSTRAINT nutrient_id IF NOT EXISTS FOR (n:Nutrient) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT environment_id IF NOT EXISTS FOR (e:Environment) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT protocol_id IF NOT EXISTS FOR (p:Protocol) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT experiment_id IF NOT EXISTS FOR (e:Experiment) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT outcome_id IF NOT EXISTS FOR (o:Outcome) REQUIRE o.id IS UNIQUE;
CREATE CONSTRAINT evidence_id IF NOT EXISTS FOR (e:Evidence) REQUIRE e.id IS UNIQUE;

// Create indexes for common query patterns
CREATE INDEX fungus_species IF NOT EXISTS FOR (f:Fungus) ON (f.species);
CREATE INDEX host_tree_species IF NOT EXISTS FOR (h:HostTree) ON (h.species);
CREATE INDEX environment_ph IF NOT EXISTS FOR (e:Environment) ON (e.pH);
CREATE INDEX environment_ec IF NOT EXISTS FOR (e:Environment) ON (e.EC);
CREATE INDEX outcome_colonization IF NOT EXISTS FOR (o:Outcome) ON (o.colonizationPercent);
CREATE INDEX outcome_yield IF NOT EXISTS FOR (o:Outcome) ON (o.yield);

// Create node labels and relationships
// Fungus nodes
CREATE (f:Fungus {
    id: 'fungus_001',
    species: 'Tuber melanosporum',
    strain: 'TME-001',
    genotype: 'TME-001-G1',
    matingType: 'MAT1-1',
    cultureHistory: 'Lab cultured from wild isolate',
    commonName: 'Black Truffle'
});

CREATE (f2:Fungus {
    id: 'fungus_002',
    species: 'Tuber magnatum',
    strain: 'TMA-001',
    genotype: 'TMA-001-G1',
    matingType: 'MAT1-2',
    cultureHistory: 'Lab cultured from wild isolate',
    commonName: 'White Truffle'
});

CREATE (f3:Fungus {
    id: 'fungus_003',
    species: 'Tuber borchii',
    strain: 'TBO-001',
    genotype: 'TBO-001-G1',
    matingType: 'MAT1-1',
    cultureHistory: 'Lab cultured from wild isolate',
    commonName: 'Bianchetto Truffle'
});

// Host Tree nodes
CREATE (h:HostTree {
    id: 'host_001',
    species: 'Quercus ilex',
    commonName: 'Holm Oak',
    age: 2,
    rootstock: 'Q. ilex seedling',
    rootArchitecture: 'Taproot with lateral branches'
});

CREATE (h2:HostTree {
    id: 'host_002',
    species: 'Quercus petraea',
    commonName: 'Sessile Oak',
    age: 2,
    rootstock: 'Q. petraea seedling',
    rootArchitecture: 'Taproot with lateral branches'
});

CREATE (h3:HostTree {
    id: 'host_003',
    species: 'Corylus avellana',
    commonName: 'Hazel',
    age: 1,
    rootstock: 'C. avellana seedling',
    rootArchitecture: 'Fibrous root system'
});

// Nutrient Recipe nodes
CREATE (r:NutrientRecipe {
    id: 'recipe_001',
    name: 'Base_Recipe_A',
    description: 'Standard hydroponic nutrient solution for truffle cultivation',
    EC: 1.2,
    pH: 6.0,
    totalVolume: 1000
});

CREATE (r2:NutrientRecipe {
    id: 'recipe_002',
    name: 'Low_EC_Recipe',
    description: 'Low electrical conductivity recipe for sensitive strains',
    EC: 0.8,
    pH: 5.8,
    totalVolume: 1000
});

// Nutrient nodes
CREATE (n1:Nutrient {
    id: 'nutrient_001',
    chemical: 'NO3-',
    concentration: 150.0,
    unit: 'mg/L',
    form: 'KNO3'
});

CREATE (n2:Nutrient {
    id: 'nutrient_002',
    chemical: 'H2PO4-',
    concentration: 50.0,
    unit: 'mg/L',
    form: 'KH2PO4'
});

CREATE (n3:Nutrient {
    id: 'nutrient_003',
    chemical: 'K+',
    concentration: 200.0,
    unit: 'mg/L',
    form: 'KNO3, KH2PO4'
});

CREATE (n4:Nutrient {
    id: 'nutrient_004',
    chemical: 'Ca2+',
    concentration: 150.0,
    unit: 'mg/L',
    form: 'Ca(NO3)2'
});

CREATE (n5:Nutrient {
    id: 'nutrient_005',
    chemical: 'Mg2+',
    concentration: 50.0,
    unit: 'mg/L',
    form: 'MgSO4'
});

CREATE (n6:Nutrient {
    id: 'nutrient_006',
    chemical: 'Fe2+',
    concentration: 2.0,
    unit: 'mg/L',
    form: 'Fe-EDTA'
});

// Environment nodes
CREATE (e:Environment {
    id: 'env_001',
    pH: 6.2,
    EC: 1.2,
    DO: 8.5,
    CO2: 400,
    temperature: 22.0,
    humidity: 70.0,
    lightSpectrum: 'Full spectrum LED',
    flowRate: 2.0,
    description: 'Standard hydroponic environment'
});

CREATE (e2:Environment {
    id: 'env_002',
    pH: 5.8,
    EC: 0.8,
    DO: 9.0,
    CO2: 350,
    temperature: 20.0,
    humidity: 75.0,
    lightSpectrum: 'Blue-red LED',
    flowRate: 1.5,
    description: 'Low EC environment for sensitive strains'
});

// Protocol nodes
CREATE (p:Protocol {
    id: 'protocol_001',
    name: 'Standard Inoculation',
    inoculationMethod: 'Root dip in spore suspension',
    sterilizationMethod: 'Autoclave at 121°C for 20 min',
    biofilmMethod: 'Natural biofilm formation',
    hydroModule: 'NFT (Nutrient Film Technique)',
    description: 'Standard protocol for truffle inoculation'
});

CREATE (p2:Protocol {
    id: 'protocol_002',
    name: 'Advanced Inoculation',
    inoculationMethod: 'Mycelial fragment injection',
    sterilizationMethod: 'UV sterilization + autoclave',
    biofilmMethod: 'Pre-formed biofilm scaffold',
    hydroModule: 'DWC (Deep Water Culture)',
    description: 'Advanced protocol with biofilm scaffold'
});

// Experiment nodes
CREATE (exp:Experiment {
    id: 'exp_001',
    name: 'TME-001 Colonization Study',
    design: 'Randomized block design',
    replicates: 12,
    duration: 90,
    measurementSchedule: 'Weekly',
    sensors: 'pH, EC, DO, temperature, humidity',
    description: 'Colonization study with T. melanosporum on Q. ilex'
});

CREATE (exp2:Experiment {
    id: 'exp_002',
    name: 'Low EC Tolerance Study',
    design: 'Factorial design',
    replicates: 18,
    duration: 120,
    measurementSchedule: 'Bi-weekly',
    sensors: 'pH, EC, DO, temperature, humidity, CO2',
    description: 'Study of truffle tolerance to low EC conditions'
});

// Outcome nodes
CREATE (o:Outcome {
    id: 'outcome_001',
    colonizationPercent: 85.5,
    hyphalDensity: 2.3,
    yield: 15.2,
    primordiaCount: 8,
    measurementDate: '2024-01-15',
    success: true
});

CREATE (o2:Outcome {
    id: 'outcome_002',
    colonizationPercent: 92.1,
    hyphalDensity: 3.1,
    yield: 22.8,
    primordiaCount: 12,
    measurementDate: '2024-01-15',
    success: true
});

CREATE (o3:Outcome {
    id: 'outcome_003',
    colonizationPercent: 45.2,
    hyphalDensity: 1.1,
    yield: 3.5,
    primordiaCount: 2,
    measurementDate: '2024-01-15',
    success: false
});

// Evidence nodes
CREATE (ev:Evidence {
    id: 'evidence_001',
    evidenceCode: 'in_planta',
    method: 'Microscopy',
    device: 'Confocal microscope',
    calibration: 'Standardized protocol',
    operator: 'Dr. Smith',
    date: '2024-01-15',
    confidence: 0.95
});

CREATE (ev2:Evidence {
    id: 'evidence_002',
    evidenceCode: 'in_vitro',
    method: 'Plate assay',
    device: 'Incubator',
    calibration: 'Temperature calibrated',
    operator: 'Lab Tech A',
    date: '2024-01-10',
    confidence: 0.88
});

// Create relationships
// Mycorrhizal associations
CREATE (f)-[:FORMS_MYCORRHIZA_WITH]->(h);
CREATE (f2)-[:FORMS_MYCORRHIZA_WITH]->(h2);
CREATE (f3)-[:FORMS_MYCORRHIZA_WITH]->(h3);

// Nutrient recipe relationships
CREATE (r)-[:HAS_NUTRIENT]->(n1);
CREATE (r)-[:HAS_NUTRIENT]->(n2);
CREATE (r)-[:HAS_NUTRIENT]->(n3);
CREATE (r)-[:HAS_NUTRIENT]->(n4);
CREATE (r)-[:HAS_NUTRIENT]->(n5);
CREATE (r)-[:HAS_NUTRIENT]->(n6);

CREATE (r2)-[:HAS_NUTRIENT]->(n1);
CREATE (r2)-[:HAS_NUTRIENT]->(n2);
CREATE (r2)-[:HAS_NUTRIENT]->(n3);
CREATE (r2)-[:HAS_NUTRIENT]->(n4);
CREATE (r2)-[:HAS_NUTRIENT]->(n5);
CREATE (r2)-[:HAS_NUTRIENT]->(n6);

// Experiment relationships
CREATE (exp)-[:USES_PROTOCOL]->(p);
CREATE (exp)-[:USES_NUTRIENT_RECIPE]->(r);
CREATE (exp)-[:HAS_OUTCOME]->(o);
CREATE (exp)-[:HAS_OUTCOME]->(o2);

CREATE (exp2)-[:USES_PROTOCOL]->(p2);
CREATE (exp2)-[:USES_NUTRIENT_RECIPE]->(r2);
CREATE (exp2)-[:HAS_OUTCOME]->(o3);

// Evidence relationships
CREATE (o)-[:SUPPORTED_BY]->(ev);
CREATE (o2)-[:SUPPORTED_BY]->(ev);
CREATE (o3)-[:SUPPORTED_BY]->(ev2);

// Environment relationships (simplified - in practice these would be more complex)
CREATE (exp)-[:OBSERVED_UNDER]->(e);
CREATE (exp2)-[:OBSERVED_UNDER]->(e2);