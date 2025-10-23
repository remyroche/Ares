"""
Literature Review Publisher
Generates HTML/PDF reports and loads data into knowledge graph
"""

import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import jinja2
from dataclasses import asdict
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from literature.normalization.validator import NormalizedExperiment, NormalizedParameter
from knowledge_graph.ingestion.etl_pipeline import TruffleKGIngestion

logger = logging.getLogger(__name__)

class LiteratureReviewPublisher:
    """Publishes literature review reports and loads data into knowledge graph"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'data/literature_review'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Jinja2 templates
        self.template_dir = Path(__file__).parent / 'templates'
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.template_dir)),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
    
    def publish_review(self, experiments: List[NormalizedExperiment], 
                      papers_metadata: List[Dict] = None) -> Dict[str, Any]:
        """Publish complete literature review"""
        logger.info(f"Publishing literature review for {len(experiments)} experiments")
        
        # Generate data dumps
        data_dumps = self.generate_data_dumps(experiments)
        
        # Generate quality report
        quality_report = self.generate_quality_report(experiments)
        
        # Generate narrative report
        narrative_report = self.generate_narrative_report(experiments, papers_metadata)
        
        # Generate visualizations
        visualizations = self.generate_visualizations(experiments)
        
        # Load into knowledge graph
        kg_results = self.load_into_knowledge_graph(experiments)
        
        # Generate summary
        summary = {
            'total_experiments': len(experiments),
            'valid_experiments': sum(1 for exp in experiments if exp.validation_result.is_valid),
            'high_confidence_experiments': sum(1 for exp in experiments if exp.confidence_0_1 > 0.8),
            'data_dumps': data_dumps,
            'quality_report': quality_report,
            'narrative_report': narrative_report,
            'visualizations': visualizations,
            'kg_results': kg_results,
            'generated_at': datetime.now().isoformat()
        }
        
        # Save summary
        with open(self.output_dir / 'review_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info("Literature review published successfully")
        return summary
    
    def generate_data_dumps(self, experiments: List[NormalizedExperiment]) -> Dict[str, str]:
        """Generate CSV data dumps"""
        data_dumps = {}
        
        # Experiments CSV
        experiments_data = []
        for exp in experiments:
            exp_dict = {
                'experiment_id': exp.experiment_id,
                'paper_id': exp.paper_id,
                'fungus_taxon_id': exp.fungus_taxon_id,
                'fungus_name': exp.fungus_name,
                'host_taxon_id': exp.host_taxon_id,
                'host_name': exp.host_name,
                'inoculum_form': exp.inoculum_form,
                'plant_age_d': exp.plant_age_d,
                'chamber_type': exp.chamber_type,
                'flow_regime': exp.flow_regime,
                'volume_L': exp.volume_L,
                'duration_d': exp.duration_d,
                'replicates': exp.replicates,
                'colonization_pct': exp.colonization_pct,
                'time_to_colonization_d': exp.time_to_colonization_d,
                'fruiting': exp.fruiting,
                'yield_g': exp.yield_g,
                'notes': exp.notes,
                'confidence_0_1': exp.confidence_0_1,
                'is_valid': exp.validation_result.is_valid if exp.validation_result else False
            }
            experiments_data.append(exp_dict)
        
        experiments_df = pd.DataFrame(experiments_data)
        experiments_file = self.output_dir / 'experiments.csv'
        experiments_df.to_csv(experiments_file, index=False)
        data_dumps['experiments'] = str(experiments_file)
        
        # Parameters CSV
        parameters_data = []
        for exp in experiments:
            for param in exp.parameters:
                param_dict = {
                    'experiment_id': exp.experiment_id,
                    'parameter_name': param.parameter_name,
                    'original_value': param.original_value,
                    'original_unit': param.original_unit,
                    'normalized_value': param.normalized_value,
                    'normalized_unit': param.normalized_unit,
                    'confidence': param.confidence,
                    'is_valid': param.validation_result.is_valid if param.validation_result else False
                }
                parameters_data.append(param_dict)
        
        parameters_df = pd.DataFrame(parameters_data)
        parameters_file = self.output_dir / 'parameters.csv'
        parameters_df.to_csv(parameters_file, index=False)
        data_dumps['parameters'] = str(parameters_file)
        
        # Nutrient recipes CSV
        recipes_data = []
        for exp in experiments:
            if exp.nutrient_recipe:
                recipe_dict = {
                    'experiment_id': exp.experiment_id,
                    'base_medium': exp.nutrient_recipe.get('base_medium', ''),
                    'salts_json': json.dumps(exp.nutrient_recipe.get('salts_json', {})),
                    'ions_json': json.dumps(exp.nutrient_recipe.get('ions_json', {})),
                    'sugar_gL': exp.nutrient_recipe.get('sugar_gL'),
                    'pgr_json': json.dumps(exp.nutrient_recipe.get('pgr_json', {})),
                    'micronutrients_json': json.dumps(exp.nutrient_recipe.get('micronutrients_json', {})),
                    'chelators_json': json.dumps(exp.nutrient_recipe.get('chelators_json', {})),
                    'recipe_confidence': exp.nutrient_recipe.get('recipe_confidence', 0.0)
                }
                recipes_data.append(recipe_dict)
        
        if recipes_data:
            recipes_df = pd.DataFrame(recipes_data)
            recipes_file = self.output_dir / 'nutrient_recipes.csv'
            recipes_df.to_csv(recipes_file, index=False)
            data_dumps['nutrient_recipes'] = str(recipes_file)
        
        # Outcomes CSV
        outcomes_data = []
        for exp in experiments:
            outcome_dict = {
                'experiment_id': exp.experiment_id,
                'colonization_pct': exp.colonization_pct,
                'time_to_colonization_d': exp.time_to_colonization_d,
                'fruiting': exp.fruiting,
                'yield_g': exp.yield_g,
                'success': exp.colonization_pct > 50.0 if exp.colonization_pct else None
            }
            outcomes_data.append(outcome_dict)
        
        outcomes_df = pd.DataFrame(outcomes_data)
        outcomes_file = self.output_dir / 'outcomes.csv'
        outcomes_df.to_csv(outcomes_file, index=False)
        data_dumps['outcomes'] = str(outcomes_file)
        
        logger.info(f"Generated data dumps: {list(data_dumps.keys())}")
        return data_dumps
    
    def generate_quality_report(self, experiments: List[NormalizedExperiment]) -> Dict[str, Any]:
        """Generate data quality report"""
        if not experiments:
            return {'error': 'No experiments to analyze'}
        
        # Basic statistics
        total_experiments = len(experiments)
        valid_experiments = sum(1 for exp in experiments if exp.validation_result.is_valid)
        high_confidence_experiments = sum(1 for exp in experiments if exp.confidence_0_1 > 0.8)
        
        # Parameter coverage
        all_params = set()
        for exp in experiments:
            for param in exp.parameters:
                all_params.add(param.parameter_name)
        
        param_coverage = {}
        for param_name in all_params:
            param_experiments = sum(1 for exp in experiments 
                                  if any(p.parameter_name == param_name for p in exp.parameters))
            param_coverage[param_name] = {
                'count': param_experiments,
                'percentage': (param_experiments / total_experiments) * 100
            }
        
        # Confidence distribution
        confidences = [exp.confidence_0_1 for exp in experiments]
        confidence_stats = {
            'mean': float(np.mean(confidences)),
            'std': float(np.std(confidences)),
            'min': float(np.min(confidences)),
            'max': float(np.max(confidences)),
            'median': float(np.median(confidences))
        }
        
        # Validation issues
        validation_issues = {
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        for exp in experiments:
            if exp.validation_result:
                validation_issues['errors'].extend(exp.validation_result.errors)
                validation_issues['warnings'].extend(exp.validation_result.warnings)
                validation_issues['suggestions'].extend(exp.validation_result.suggestions)
        
        # Count unique issues
        validation_issues['error_counts'] = {error: validation_issues['errors'].count(error) 
                                           for error in set(validation_issues['errors'])}
        validation_issues['warning_counts'] = {warning: validation_issues['warnings'].count(warning) 
                                             for warning in set(validation_issues['warnings'])}
        
        # Species distribution
        fungus_species = {}
        host_species = {}
        for exp in experiments:
            fungus_species[exp.fungus_name] = fungus_species.get(exp.fungus_name, 0) + 1
            host_species[exp.host_name] = host_species.get(exp.host_name, 0) + 1
        
        quality_report = {
            'total_experiments': total_experiments,
            'valid_experiments': valid_experiments,
            'high_confidence_experiments': high_confidence_experiments,
            'validation_rate': (valid_experiments / total_experiments) * 100,
            'high_confidence_rate': (high_confidence_experiments / total_experiments) * 100,
            'parameter_coverage': param_coverage,
            'confidence_statistics': confidence_stats,
            'validation_issues': validation_issues,
            'fungus_species_distribution': fungus_species,
            'host_species_distribution': host_species
        }
        
        # Save quality report
        quality_file = self.output_dir / 'quality_report.json'
        with open(quality_file, 'w') as f:
            json.dump(quality_report, f, indent=2, default=str)
        
        logger.info("Generated quality report")
        return quality_report
    
    def generate_narrative_report(self, experiments: List[NormalizedExperiment], 
                                papers_metadata: List[Dict] = None) -> Dict[str, str]:
        """Generate narrative literature review report"""
        
        # Calculate statistics
        total_experiments = len(experiments)
        valid_experiments = sum(1 for exp in experiments if exp.validation_result.is_valid)
        
        # Species analysis
        fungus_species = {}
        host_species = {}
        for exp in experiments:
            fungus_species[exp.fungus_name] = fungus_species.get(exp.fungus_name, 0) + 1
            host_species[exp.host_name] = host_species.get(exp.host_name, 0) + 1
        
        # Parameter analysis
        param_stats = {}
        for exp in experiments:
            for param in exp.parameters:
                if param.parameter_name not in param_stats:
                    param_stats[param.parameter_name] = []
                param_stats[param.parameter_name].append(param.normalized_value)
        
        param_summary = {}
        for param_name, values in param_stats.items():
            param_summary[param_name] = {
                'count': len(values),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
        
        # Success rate analysis
        success_experiments = sum(1 for exp in experiments 
                                if exp.colonization_pct and exp.colonization_pct > 50.0)
        success_rate = (success_experiments / total_experiments) * 100 if total_experiments > 0 else 0
        
        # Generate HTML report
        html_template = self.jinja_env.get_template('literature_review.html')
        html_content = html_template.render(
            total_experiments=total_experiments,
            valid_experiments=valid_experiments,
            success_rate=success_rate,
            fungus_species=fungus_species,
            host_species=host_species,
            param_summary=param_summary,
            experiments=experiments[:10],  # Show first 10 experiments
            generated_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        html_file = self.output_dir / 'literature_review.html'
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Generate PDF report (if weasyprint is available)
        pdf_file = None
        try:
            import weasyprint
            pdf_file = self.output_dir / 'literature_review.pdf'
            weasyprint.HTML(string=html_content).write_pdf(str(pdf_file))
            logger.info("Generated PDF report")
        except ImportError:
            logger.warning("weasyprint not available, PDF generation skipped")
        
        narrative_report = {
            'html_file': str(html_file),
            'pdf_file': str(pdf_file) if pdf_file else None,
            'total_experiments': total_experiments,
            'valid_experiments': valid_experiments,
            'success_rate': success_rate
        }
        
        logger.info("Generated narrative report")
        return narrative_report
    
    def generate_visualizations(self, experiments: List[NormalizedExperiment]) -> Dict[str, str]:
        """Generate visualization plots"""
        visualizations = {}
        
        if not experiments:
            return visualizations
        
        # 1. Parameter distribution plots
        param_data = []
        for exp in experiments:
            for param in exp.parameters:
                param_data.append({
                    'parameter': param.parameter_name,
                    'value': param.normalized_value,
                    'confidence': param.confidence,
                    'fungus': exp.fungus_name,
                    'host': exp.host_name
                })
        
        if param_data:
            param_df = pd.DataFrame(param_data)
            
            # Parameter value distributions
            fig = px.box(param_df, x='parameter', y='value', 
                        title='Parameter Value Distributions',
                        color='parameter')
            fig.update_layout(xaxis_tickangle=-45)
            param_dist_file = self.output_dir / 'parameter_distributions.html'
            fig.write_html(str(param_dist_file))
            visualizations['parameter_distributions'] = str(param_dist_file)
            
            # Confidence vs parameter scatter
            fig = px.scatter(param_df, x='value', y='confidence', 
                           color='parameter', facet_col='parameter',
                           title='Parameter Values vs Confidence')
            conf_scatter_file = self.output_dir / 'confidence_scatter.html'
            fig.write_html(str(conf_scatter_file))
            visualizations['confidence_scatter'] = str(conf_scatter_file)
        
        # 2. Species analysis
        fungus_counts = {}
        host_counts = {}
        for exp in experiments:
            fungus_counts[exp.fungus_name] = fungus_counts.get(exp.fungus_name, 0) + 1
            host_counts[exp.host_name] = host_counts.get(exp.host_name, 0) + 1
        
        # Fungus species pie chart
        if fungus_counts:
            fig = px.pie(values=list(fungus_counts.values()), 
                        names=list(fungus_counts.keys()),
                        title='Fungus Species Distribution')
            fungus_pie_file = self.output_dir / 'fungus_species.html'
            fig.write_html(str(fungus_pie_file))
            visualizations['fungus_species'] = str(fungus_pie_file)
        
        # Host species pie chart
        if host_counts:
            fig = px.pie(values=list(host_counts.values()), 
                        names=list(host_counts.keys()),
                        title='Host Species Distribution')
            host_pie_file = self.output_dir / 'host_species.html'
            fig.write_html(str(host_pie_file))
            visualizations['host_species'] = str(host_pie_file)
        
        # 3. Success rate analysis
        success_data = []
        for exp in experiments:
            if exp.colonization_pct is not None:
                success_data.append({
                    'fungus': exp.fungus_name,
                    'host': exp.host_name,
                    'colonization': exp.colonization_pct,
                    'success': exp.colonization_pct > 50.0
                })
        
        if success_data:
            success_df = pd.DataFrame(success_data)
            
            # Colonization rate by fungus-host combination
            fig = px.box(success_df, x='fungus', y='colonization', 
                        color='host', title='Colonization Rates by Species Combination')
            fig.update_layout(xaxis_tickangle=-45)
            colonization_file = self.output_dir / 'colonization_rates.html'
            fig.write_html(str(colonization_file))
            visualizations['colonization_rates'] = str(colonization_file)
        
        # 4. Quality metrics dashboard
        quality_metrics = []
        for exp in experiments:
            quality_metrics.append({
                'experiment_id': exp.experiment_id,
                'confidence': exp.confidence_0_1,
                'is_valid': exp.validation_result.is_valid if exp.validation_result else False,
                'parameter_count': len(exp.parameters),
                'fungus': exp.fungus_name,
                'host': exp.host_name
            })
        
        if quality_metrics:
            quality_df = pd.DataFrame(quality_metrics)
            
            # Confidence distribution
            fig = px.histogram(quality_df, x='confidence', 
                             title='Experiment Confidence Distribution',
                             nbins=20)
            conf_hist_file = self.output_dir / 'confidence_distribution.html'
            fig.write_html(str(conf_hist_file))
            visualizations['confidence_distribution'] = str(conf_hist_file)
            
            # Parameter count vs confidence
            fig = px.scatter(quality_df, x='parameter_count', y='confidence',
                           color='is_valid', title='Parameter Count vs Confidence')
            param_conf_file = self.output_dir / 'parameter_confidence.html'
            fig.write_html(str(param_conf_file))
            visualizations['parameter_confidence'] = str(param_conf_file)
        
        logger.info(f"Generated {len(visualizations)} visualizations")
        return visualizations
    
    def load_into_knowledge_graph(self, experiments: List[NormalizedExperiment]) -> Dict[str, Any]:
        """Load experiments into knowledge graph"""
        try:
            # Initialize ETL pipeline
            etl_pipeline = TruffleKGIngestion(self.config)
            
            # Convert experiments to ETL format
            etl_data = []
            for exp in experiments:
                # Create experiment record
                exp_record = {
                    'type': 'experiment',
                    'experiment_id': exp.experiment_id,
                    'paper_id': exp.paper_id,
                    'fungus_name': exp.fungus_name,
                    'host_name': exp.host_name,
                    'inoculum_form': exp.inoculum_form,
                    'chamber_type': exp.chamber_type,
                    'flow_regime': exp.flow_regime,
                    'volume_L': exp.volume_L,
                    'duration_d': exp.duration_d,
                    'replicates': exp.replicates,
                    'colonization_pct': exp.colonization_pct,
                    'time_to_colonization_d': exp.time_to_colonization_d,
                    'fruiting': exp.fruiting,
                    'yield_g': exp.yield_g,
                    'notes': exp.notes,
                    'confidence': exp.confidence_0_1,
                    'is_valid': exp.validation_result.is_valid if exp.validation_result else False
                }
                etl_data.append(exp_record)
                
                # Create parameter records
                for param in exp.parameters:
                    param_record = {
                        'type': 'parameter',
                        'experiment_id': exp.experiment_id,
                        'parameter_name': param.parameter_name,
                        'original_value': param.original_value,
                        'original_unit': param.original_unit,
                        'normalized_value': param.normalized_value,
                        'normalized_unit': param.normalized_unit,
                        'confidence': param.confidence,
                        'is_valid': param.validation_result.is_valid if param.validation_result else False
                    }
                    etl_data.append(param_record)
            
            # Run ETL pipeline
            etl_pipeline.normalize_data(etl_data)
            etl_pipeline.validate_data(etl_data)
            etl_pipeline.store_rdf(etl_data)
            etl_pipeline.store_neo4j(etl_data)
            
            kg_results = {
                'status': 'success',
                'experiments_loaded': len(experiments),
                'parameters_loaded': sum(len(exp.parameters) for exp in experiments),
                'message': 'Data successfully loaded into knowledge graph'
            }
            
        except Exception as e:
            logger.error(f"Error loading data into knowledge graph: {e}")
            kg_results = {
                'status': 'error',
                'error': str(e),
                'message': 'Failed to load data into knowledge graph'
            }
        
        # Save KG results
        kg_file = self.output_dir / 'kg_loading_results.json'
        with open(kg_file, 'w') as f:
            json.dump(kg_results, f, indent=2, default=str)
        
        logger.info("Knowledge graph loading completed")
        return kg_results

def main():
    """Example usage of the literature review publisher"""
    config = {
        'output_dir': 'data/literature_review',
        'neo4j': {
            'uri': 'bolt://localhost:7687',
            'user': 'neo4j',
            'password': 'password'
        },
        'rdf': {
            'endpoint': 'http://localhost:3030/ds'
        }
    }
    
    publisher = LiteratureReviewPublisher(config)
    
    # Load sample experiments
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
            notes="Sample experiment",
            confidence_0_1=0.9,
            parameters=[],
            validation_result=None
        )
    ]
    
    # Publish review
    results = publisher.publish_review(sample_experiments)
    print(f"Published review with {results['total_experiments']} experiments")

if __name__ == "__main__":
    main()