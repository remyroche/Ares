"""
GraphQL Schema for Truffle Knowledge Graph API
Provides unified interface for querying both RDF and Neo4j data
"""

import graphene
from graphene import ObjectType, String, Float, Int, List, Field, ID
from graphene_sqlalchemy import SQLAlchemyObjectType
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# GraphQL Types
class MeasurementType(graphene.ObjectType):
    """Represents a quantitative measurement"""
    value = Float(required=True)
    unit = String(required=True)
    uncertainty = Float()
    confidence = Float()
    method = String()
    device = String()

class EvidenceType(graphene.ObjectType):
    """Represents evidence supporting measurements"""
    evidence_code = String(required=True)
    method = String(required=True)
    device = String(required=True)
    operator = String(required=True)
    confidence = Float(required=True)
    date = String(required=True)

class FungusType(graphene.ObjectType):
    """Represents a truffle fungus"""
    id = ID(required=True)
    species = String(required=True)
    strain = String()
    genotype = String()
    mating_type = String()
    culture_history = String()
    common_name = String()

class HostTreeType(graphene.ObjectType):
    """Represents a host tree"""
    id = ID(required=True)
    species = String(required=True)
    common_name = String()
    age = Int()
    rootstock = String()
    root_architecture = String()

class NutrientType(graphene.ObjectType):
    """Represents a nutrient compound"""
    id = ID(required=True)
    chemical = String(required=True)
    concentration = Float(required=True)
    unit = String(required=True)
    form = String()

class NutrientRecipeType(graphene.ObjectType):
    """Represents a nutrient recipe"""
    id = ID(required=True)
    name = String(required=True)
    description = String()
    ec = Float()
    ph = Float()
    total_volume = Float()
    nutrients = List(NutrientType)

class EnvironmentType(graphene.ObjectType):
    """Represents environmental conditions"""
    id = ID(required=True)
    ph = Float()
    ec = Float()
    do = Float()
    co2 = Float()
    temperature = Float()
    humidity = Float()
    light_spectrum = String()
    flow_rate = Float()
    description = String()

class ProtocolType(graphene.ObjectType):
    """Represents an experimental protocol"""
    id = ID(required=True)
    name = String(required=True)
    inoculation_method = String()
    sterilization_method = String()
    biofilm_method = String()
    hydro_module = String()
    description = String()

class ExperimentType(graphene.ObjectType):
    """Represents an experiment"""
    id = ID(required=True)
    name = String(required=True)
    design = String()
    replicates = Int()
    duration = Int()
    measurement_schedule = String()
    sensors = String()
    description = String()
    protocol = Field(ProtocolType)
    nutrient_recipe = Field(NutrientRecipeType)
    environment = Field(EnvironmentType)

class OutcomeType(graphene.ObjectType):
    """Represents experimental outcomes"""
    id = ID(required=True)
    colonization_percent = Field(MeasurementType)
    hyphal_density = Field(MeasurementType)
    yield_measurement = Field(MeasurementType)
    primordia_count = Int()
    success = Boolean()
    evidence = List(EvidenceType)

class MycorrhizaType(graphene.ObjectType):
    """Represents a mycorrhizal association"""
    id = ID(required=True)
    fungus = Field(FungusType)
    host_tree = Field(HostTreeType)
    environment = Field(EnvironmentType)
    outcomes = List(OutcomeType)

# Query Types
class Query(ObjectType):
    """Root query type"""
    
    # Fungus queries
    fungi = List(FungusType, species=String())
    fungus = Field(FungusType, id=ID(required=True))
    
    # Host tree queries
    host_trees = List(HostTreeType, species=String())
    host_tree = Field(HostTreeType, id=ID(required=True))
    
    # Nutrient queries
    nutrients = List(NutrientType)
    nutrient = Field(NutrientType, id=ID(required=True))
    
    # Recipe queries
    nutrient_recipes = List(NutrientRecipeType, ec_max=Float(), ph_min=Float(), ph_max=Float())
    nutrient_recipe = Field(NutrientRecipeType, id=ID(required=True))
    
    # Environment queries
    environments = List(EnvironmentType, ph_min=Float(), ph_max=Float(), ec_min=Float(), ec_max=Float())
    environment = Field(EnvironmentType, id=ID(required=True))
    
    # Protocol queries
    protocols = List(ProtocolType, method=String())
    protocol = Field(ProtocolType, id=ID(required=True))
    
    # Experiment queries
    experiments = List(ExperimentType, fungus_id=ID(), host_tree_id=ID(), success=Boolean())
    experiment = Field(ExperimentType, id=ID(required=True))
    
    # Outcome queries
    outcomes = List(OutcomeType, min_colonization=Float(), min_yield=Float())
    outcome = Field(OutcomeType, id=ID(required=True))
    
    # Mycorrhiza queries
    mycorrhizae = List(MycorrhizaType, fungus_id=ID(), host_tree_id=ID())
    mycorrhiza = Field(MycorrhizaType, id=ID(required=True))
    
    # Complex queries
    best_hosts_for_fungus = List(HostTreeType, fungus_id=ID(required=True), ph_max=Float())
    similar_recipes = List(NutrientRecipeType, recipe_id=ID(required=True), similarity_threshold=Float())
    successful_protocols = List(ProtocolType, fungus_id=ID(), host_tree_id=ID())
    
    def resolve_fungi(self, info, species=None):
        """Resolve fungi query"""
        # This would query the actual data source
        # For now, return mock data
        return [
            FungusType(
                id="fungus_001",
                species="Tuber melanosporum",
                strain="TME-001",
                genotype="TME-001-G1",
                mating_type="MAT1-1",
                culture_history="Lab cultured from wild isolate",
                common_name="Black Truffle"
            ),
            FungusType(
                id="fungus_002",
                species="Tuber magnatum",
                strain="TMA-001",
                genotype="TMA-001-G1",
                mating_type="MAT1-2",
                culture_history="Lab cultured from wild isolate",
                common_name="White Truffle"
            )
        ]
    
    def resolve_fungus(self, info, id):
        """Resolve single fungus query"""
        # Mock implementation
        if id == "fungus_001":
            return FungusType(
                id="fungus_001",
                species="Tuber melanosporum",
                strain="TME-001",
                genotype="TME-001-G1",
                mating_type="MAT1-1",
                culture_history="Lab cultured from wild isolate",
                common_name="Black Truffle"
            )
        return None
    
    def resolve_host_trees(self, info, species=None):
        """Resolve host trees query"""
        return [
            HostTreeType(
                id="host_001",
                species="Quercus ilex",
                common_name="Holm Oak",
                age=2,
                rootstock="Q. ilex seedling",
                root_architecture="Taproot with lateral branches"
            ),
            HostTreeType(
                id="host_002",
                species="Quercus petraea",
                common_name="Sessile Oak",
                age=2,
                rootstock="Q. petraea seedling",
                root_architecture="Taproot with lateral branches"
            )
        ]
    
    def resolve_nutrient_recipes(self, info, ec_max=None, ph_min=None, ph_max=None):
        """Resolve nutrient recipes with filtering"""
        recipes = [
            NutrientRecipeType(
                id="recipe_001",
                name="Base_Recipe_A",
                description="Standard hydroponic nutrient solution",
                ec=1.2,
                ph=6.0,
                total_volume=1000.0,
                nutrients=[
                    NutrientType(
                        id="nutrient_001",
                        chemical="NO3-",
                        concentration=150.0,
                        unit="mg/L",
                        form="KNO3"
                    ),
                    NutrientType(
                        id="nutrient_002",
                        chemical="H2PO4-",
                        concentration=50.0,
                        unit="mg/L",
                        form="KH2PO4"
                    )
                ]
            ),
            NutrientRecipeType(
                id="recipe_002",
                name="Low_EC_Recipe",
                description="Low electrical conductivity recipe",
                ec=0.8,
                ph=5.8,
                total_volume=1000.0,
                nutrients=[]
            )
        ]
        
        # Apply filters
        filtered_recipes = recipes
        if ec_max is not None:
            filtered_recipes = [r for r in filtered_recipes if r.ec <= ec_max]
        if ph_min is not None:
            filtered_recipes = [r for r in filtered_recipes if r.ph >= ph_min]
        if ph_max is not None:
            filtered_recipes = [r for r in filtered_recipes if r.ph <= ph_max]
        
        return filtered_recipes
    
    def resolve_best_hosts_for_fungus(self, info, fungus_id, ph_max=None):
        """Find best host trees for a specific fungus"""
        # This would query the knowledge graph for actual data
        # For now, return mock data
        hosts = [
            HostTreeType(
                id="host_001",
                species="Quercus ilex",
                common_name="Holm Oak",
                age=2,
                rootstock="Q. ilex seedling",
                root_architecture="Taproot with lateral branches"
            )
        ]
        
        if ph_max is not None:
            # Filter by pH tolerance (mock implementation)
            hosts = [h for h in hosts if True]  # Simplified
        
        return hosts
    
    def resolve_similar_recipes(self, info, recipe_id, similarity_threshold=0.8):
        """Find similar nutrient recipes"""
        # This would use similarity algorithms on the actual data
        # For now, return mock data
        return [
            NutrientRecipeType(
                id="recipe_002",
                name="Similar_Recipe",
                description="Similar to requested recipe",
                ec=1.1,
                ph=5.9,
                total_volume=1000.0,
                nutrients=[]
            )
        ]

# Mutation Types
class CreateExperimentInput(graphene.InputObjectType):
    """Input for creating a new experiment"""
    name = String(required=True)
    design = String()
    replicates = Int()
    duration = Int()
    fungus_id = ID(required=True)
    host_tree_id = ID(required=True)
    protocol_id = ID()
    nutrient_recipe_id = ID()
    environment_id = ID()

class CreateExperiment(graphene.Mutation):
    """Mutation to create a new experiment"""
    
    class Arguments:
        input = CreateExperimentInput(required=True)
    
    experiment = Field(ExperimentType)
    success = Boolean()
    message = String()
    
    def mutate(self, info, input):
        """Create a new experiment"""
        # This would create the experiment in the database
        # For now, return mock data
        experiment = ExperimentType(
            id="exp_new",
            name=input.name,
            design=input.design or "Randomized design",
            replicates=input.replicates or 10,
            duration=input.duration or 90,
            measurement_schedule="Weekly",
            sensors="pH, EC, DO, temperature",
            description="New experiment"
        )
        
        return CreateExperiment(
            experiment=experiment,
            success=True,
            message="Experiment created successfully"
        )

class UpdateSetpointInput(graphene.InputObjectType):
    """Input for updating control setpoints"""
    ph = Float()
    ec = Float()
    temperature = Float()
    do = Float()

class UpdateSetpoint(graphene.Mutation):
    """Mutation to update control setpoints"""
    
    class Arguments:
        input = UpdateSetpointInput(required=True)
    
    success = Boolean()
    message = String()
    
    def mutate(self, info, input):
        """Update control setpoints"""
        # This would update the MPC controller setpoints
        # For now, return success
        return UpdateSetpoint(
            success=True,
            message="Setpoints updated successfully"
        )

class Mutation(ObjectType):
    """Root mutation type"""
    create_experiment = CreateExperiment.Field()
    update_setpoint = UpdateSetpoint.Field()

# Schema
schema = graphene.Schema(query=Query, mutation=Mutation)

# Example queries
EXAMPLE_QUERIES = {
    "get_fungi": """
        query {
            fungi {
                id
                species
                commonName
                strain
            }
        }
    """,
    
    "get_best_hosts": """
        query GetBestHosts($fungusId: ID!, $phMax: Float) {
            bestHostsForFungus(fungusId: $fungusId, phMax: $phMax) {
                id
                species
                commonName
                age
            }
        }
    """,
    
    "get_similar_recipes": """
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
    """,
    
    "create_experiment": """
        mutation CreateExperiment($input: CreateExperimentInput!) {
            createExperiment(input: $input) {
                experiment {
                    id
                    name
                    design
                    replicates
                }
                success
                message
            }
        }
    """,
    
    "update_setpoints": """
        mutation UpdateSetpoints($input: UpdateSetpointInput!) {
            updateSetpoint(input: $input) {
                success
                message
            }
        }
    """
}

def get_schema():
    """Get the GraphQL schema"""
    return schema

def get_example_queries():
    """Get example GraphQL queries"""
    return EXAMPLE_QUERIES