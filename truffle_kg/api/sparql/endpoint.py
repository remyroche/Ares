"""
SPARQL Endpoint for Truffle Knowledge Graph
Provides SPARQL query interface for RDF data
"""

import logging
from typing import Dict, List, Any, Optional
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD
from rdflib.plugins.sparql import prepareQuery
from rdflib.plugins.sparql.results import ResultRow
import json
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import traceback

logger = logging.getLogger(__name__)

class TruffleSPARQLEndpoint:
    """SPARQL endpoint for truffle knowledge graph"""
    
    def __init__(self, rdf_file: str = None):
        """
        Initialize the SPARQL endpoint
        
        Args:
            rdf_file: Path to RDF file to load
        """
        self.graph = Graph()
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Load namespaces
        self.ex = Namespace("http://example.org/truffle/kg#")
        self.prov = Namespace("http://www.w3.org/ns/prov#")
        self.qudt = Namespace("http://qudt.org/schema/qudt/")
        
        # Load RDF data
        if rdf_file:
            self.load_rdf_file(rdf_file)
        
        # Setup routes
        self._setup_routes()
        
        logger.info("SPARQL endpoint initialized")
    
    def load_rdf_file(self, rdf_file: str):
        """Load RDF data from file"""
        try:
            self.graph.parse(rdf_file, format="turtle")
            logger.info(f"Loaded RDF data from {rdf_file}")
        except Exception as e:
            logger.error(f"Error loading RDF file {rdf_file}: {e}")
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/sparql', methods=['GET', 'POST'])
        def sparql_query():
            """Handle SPARQL queries"""
            try:
                if request.method == 'GET':
                    query = request.args.get('query', '')
                    format_type = request.args.get('format', 'json')
                else:
                    data = request.get_json()
                    query = data.get('query', '')
                    format_type = data.get('format', 'json')
                
                if not query:
                    return jsonify({'error': 'No query provided'}), 400
                
                # Execute SPARQL query
                results = self.execute_sparql_query(query)
                
                if format_type == 'json':
                    return jsonify(results)
                elif format_type == 'xml':
                    return self._format_xml_results(results)
                else:
                    return jsonify(results)
                    
            except Exception as e:
                logger.error(f"SPARQL query error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/sparql/query', methods=['POST'])
        def sparql_query_post():
            """Handle SPARQL queries via POST with form data"""
            try:
                query = request.form.get('query', '')
                format_type = request.form.get('format', 'json')
                
                if not query:
                    return jsonify({'error': 'No query provided'}), 400
                
                results = self.execute_sparql_query(query)
                
                if format_type == 'json':
                    return jsonify(results)
                else:
                    return jsonify(results)
                    
            except Exception as e:
                logger.error(f"SPARQL query error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/sparql/explore')
        def sparql_explore():
            """SPARQL query explorer interface"""
            return render_template('sparql_explorer.html')
        
        @self.app.route('/sparql/examples')
        def sparql_examples():
            """Get example SPARQL queries"""
            return jsonify(self.get_example_queries())
    
    def execute_sparql_query(self, query: str) -> Dict[str, Any]:
        """Execute a SPARQL query and return results"""
        try:
            # Prepare query
            prepared_query = prepareQuery(query)
            
            # Execute query
            results = self.graph.query(prepared_query)
            
            # Convert results to dictionary
            if results.type == 'SELECT':
                return self._format_select_results(results)
            elif results.type == 'ASK':
                return self._format_ask_results(results)
            elif results.type == 'CONSTRUCT':
                return self._format_construct_results(results)
            elif results.type == 'DESCRIBE':
                return self._format_describe_results(results)
            else:
                return {'error': 'Unsupported query type'}
                
        except Exception as e:
            logger.error(f"Error executing SPARQL query: {e}")
            raise e
    
    def _format_select_results(self, results) -> Dict[str, Any]:
        """Format SELECT query results"""
        bindings = []
        for row in results:
            binding = {}
            for var in results.vars:
                if var in row:
                    value = row[var]
                    if isinstance(value, Literal):
                        binding[str(var)] = {
                            'value': str(value),
                            'type': 'literal',
                            'datatype': str(value.datatype) if value.datatype else None
                        }
                    elif isinstance(value, URIRef):
                        binding[str(var)] = {
                            'value': str(value),
                            'type': 'uri'
                        }
                    else:
                        binding[str(var)] = {
                            'value': str(value),
                            'type': 'unknown'
                        }
            bindings.append(binding)
        
        return {
            'head': {
                'vars': [str(var) for var in results.vars]
            },
            'results': {
                'bindings': bindings
            }
        }
    
    def _format_ask_results(self, results) -> Dict[str, Any]:
        """Format ASK query results"""
        return {
            'head': {},
            'boolean': bool(results)
        }
    
    def _format_construct_results(self, results) -> Dict[str, Any]:
        """Format CONSTRUCT query results"""
        # Convert constructed graph to turtle
        constructed_graph = Graph()
        for triple in results:
            constructed_graph.add(triple)
        
        return {
            'head': {},
            'results': {
                'turtle': constructed_graph.serialize(format='turtle')
            }
        }
    
    def _format_describe_results(self, results) -> Dict[str, Any]:
        """Format DESCRIBE query results"""
        # Convert described graph to turtle
        described_graph = Graph()
        for triple in results:
            described_graph.add(triple)
        
        return {
            'head': {},
            'results': {
                'turtle': described_graph.serialize(format='turtle')
            }
        }
    
    def _format_xml_results(self, results: Dict[str, Any]) -> str:
        """Format results as XML (simplified)"""
        # This is a simplified XML formatter
        # In practice, you'd use a proper SPARQL XML formatter
        xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
        xml += '<sparql xmlns="http://www.w3.org/2005/sparql-results#">\n'
        
        if 'head' in results:
            xml += '  <head>\n'
            if 'vars' in results['head']:
                for var in results['head']['vars']:
                    xml += f'    <variable name="{var}"/>\n'
            xml += '  </head>\n'
        
        if 'results' in results and 'bindings' in results['results']:
            xml += '  <results>\n'
            for binding in results['results']['bindings']:
                xml += '    <result>\n'
                for var, value in binding.items():
                    xml += f'      <binding name="{var}">\n'
                    if value['type'] == 'literal':
                        xml += f'        <literal>{value["value"]}</literal>\n'
                    elif value['type'] == 'uri':
                        xml += f'        <uri>{value["value"]}</uri>\n'
                    xml += '      </binding>\n'
                xml += '    </result>\n'
            xml += '  </results>\n'
        
        xml += '</sparql>'
        return xml
    
    def get_example_queries(self) -> Dict[str, str]:
        """Get example SPARQL queries"""
        return {
            "get_all_fungi": """
                PREFIX ex: <http://example.org/truffle/kg#>
                SELECT ?fungus ?species ?strain ?commonName WHERE {
                    ?fungus a ex:Fungus ;
                            ex:species ?species ;
                            ex:strain ?strain ;
                            ex:commonName ?commonName .
                }
            """,
            
            "get_fungi_by_species": """
                PREFIX ex: <http://example.org/truffle/kg#>
                SELECT ?fungus ?species ?strain WHERE {
                    ?fungus a ex:Fungus ;
                            ex:species ?species ;
                            ex:strain ?strain .
                    FILTER(?species = "Tuber melanosporum")
                }
            """,
            
            "get_mycorrhizal_associations": """
                PREFIX ex: <http://example.org/truffle/kg#>
                SELECT ?fungus ?host ?fungusSpecies ?hostSpecies WHERE {
                    ?fungus ex:formsMycorrhizaWith ?host .
                    ?fungus ex:species ?fungusSpecies .
                    ?host ex:species ?hostSpecies .
                }
            """,
            
            "get_experiments_with_outcomes": """
                PREFIX ex: <http://example.org/truffle/kg#>
                SELECT ?experiment ?outcome ?colonization ?yield WHERE {
                    ?experiment a ex:Experiment ;
                               ex:hasOutcome ?outcome .
                    ?outcome ex:colonizationPercent ?colonization ;
                            ex:yield ?yield .
                }
            """,
            
            "get_nutrient_recipes_by_ec": """
                PREFIX ex: <http://example.org/truffle/kg#>
                SELECT ?recipe ?name ?ec ?ph WHERE {
                    ?recipe a ex:NutrientRecipe ;
                            ex:name ?name ;
                            ex:EC ?ec ;
                            ex:pH ?ph .
                    FILTER(?ec <= 1.5)
                }
                ORDER BY ?ec
            """,
            
            "get_best_hosts_for_fungus": """
                PREFIX ex: <http://example.org/truffle/kg#>
                SELECT ?host ?hostSpecies (AVG(?colonization) AS ?avgColonization) WHERE {
                    ?fungus ex:species "Tuber melanosporum" ;
                            ex:formsMycorrhizaWith ?host .
                    ?host ex:species ?hostSpecies .
                    ?myc ex:ofFungus ?fungus ;
                         ex:withHost ?host ;
                         ex:observedUnder ?env ;
                         ex:measuredBy ?outcome .
                    ?env ex:pH ?pH .
                    FILTER(?pH <= 6.2)
                    ?outcome ex:colonizationPercent ?colonization .
                }
                GROUP BY ?host ?hostSpecies
                ORDER BY DESC(?avgColonization)
            """,
            
            "get_successful_protocols": """
                PREFIX ex: <http://example.org/truffle/kg#>
                SELECT ?protocol ?method (COUNT(?outcome) AS ?successCount) WHERE {
                    ?experiment ex:usesProtocol ?protocol ;
                               ex:hasOutcome ?outcome .
                    ?protocol ex:inoculationMethod ?method .
                    ?outcome ex:success true .
                }
                GROUP BY ?protocol ?method
                ORDER BY DESC(?successCount)
            """,
            
            "get_environmental_conditions": """
                PREFIX ex: <http://example.org/truffle/kg#>
                SELECT ?env ?ph ?ec ?temperature ?humidity WHERE {
                    ?env a ex:Environment ;
                         ex:pH ?ph ;
                         ex:EC ?ec ;
                         ex:temperature ?temperature ;
                         ex:humidity ?humidity .
                    FILTER(?ph >= 5.5 && ?ph <= 6.5)
                    FILTER(?ec >= 0.8 && ?ec <= 2.0)
                }
            """,
            
            "get_evidence_for_outcomes": """
                PREFIX ex: <http://example.org/truffle/kg#>
                PREFIX prov: <http://www.w3.org/ns/prov#>
                SELECT ?outcome ?evidence ?method ?confidence WHERE {
                    ?outcome ex:supportedBy ?evidence .
                    ?evidence ex:method ?method ;
                              ex:confidence ?confidence .
                }
                ORDER BY DESC(?confidence)
            """,
            
            "get_measurement_statistics": """
                PREFIX ex: <http://example.org/truffle/kg#>
                SELECT ?measurementType (AVG(?value) AS ?avgValue) (MIN(?value) AS ?minValue) (MAX(?value) AS ?maxValue) WHERE {
                    ?outcome ex:hasMeasurement ?measurement .
                    ?measurement ex:value ?value ;
                                 ex:unit ?unit .
                    BIND(?unit AS ?measurementType)
                }
                GROUP BY ?measurementType
            """
        }
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the SPARQL endpoint server"""
        logger.info(f"Starting SPARQL endpoint on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

def create_sparql_endpoint(rdf_file: str = None) -> TruffleSPARQLEndpoint:
    """Create a SPARQL endpoint instance"""
    return TruffleSPARQLEndpoint(rdf_file)

def main():
    """Main function to run the SPARQL endpoint"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Truffle Knowledge Graph SPARQL Endpoint')
    parser.add_argument('--rdf-file', type=str, help='Path to RDF file to load')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Create and run endpoint
    endpoint = create_sparql_endpoint(args.rdf_file)
    endpoint.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()