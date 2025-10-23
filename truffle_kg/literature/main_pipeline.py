"""
Main Literature Processing Pipeline
Orchestrates the complete workflow from discovery to knowledge graph loading
"""

import logging
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import sys

# Import our modules
sys.path.append(str(Path(__file__).parent))

from discovery.downloader import LiteratureDownloader
from parsing.parser import DocumentParser
from extraction.extractor import ExperimentExtractor
from normalization.validator import DataNormalizer
from review.review_app import ReviewApp
from publishing.review_publisher import LiteratureReviewPublisher

logger = logging.getLogger(__name__)

class LiteratureProcessingPipeline:
    """Main pipeline for processing literature data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'data/literature_processing'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.downloader = LiteratureDownloader(config)
        self.parser = DocumentParser(config)
        self.extractor = ExperimentExtractor()
        self.normalizer = DataNormalizer()
        self.publisher = LiteratureReviewPublisher(config)
        
        # Pipeline state
        self.papers_metadata = []
        self.parsed_documents = []
        self.extracted_experiments = []
        self.normalized_experiments = []
        
    def run_full_pipeline(self, queries: Dict[str, str], max_results_per_query: int = 1000) -> Dict[str, Any]:
        """Run the complete literature processing pipeline"""
        logger.info("Starting full literature processing pipeline...")
        
        pipeline_results = {
            'start_time': datetime.now().isoformat(),
            'steps_completed': [],
            'errors': [],
            'warnings': []
        }
        
        try:
            # Step 1: Discover and Download
            logger.info("Step 1: Discovering and downloading literature...")
            papers, patents = self.downloader.run_discovery(queries, max_results_per_query)
            self.papers_metadata = papers
            pipeline_results['steps_completed'].append('discovery_download')
            pipeline_results['papers_found'] = len(papers)
            pipeline_results['patents_found'] = len(patents)
            
            # Step 2: Parse and Segment
            logger.info("Step 2: Parsing and segmenting documents...")
            parsed_docs = self.parse_documents(papers)
            self.parsed_documents = parsed_docs
            pipeline_results['steps_completed'].append('parse_segment')
            pipeline_results['documents_parsed'] = len(parsed_docs)
            
            # Step 3: Extract Data
            logger.info("Step 3: Extracting experimental data...")
            extracted_experiments = self.extract_experiments(parsed_docs)
            self.extracted_experiments = extracted_experiments
            pipeline_results['steps_completed'].append('extract_data')
            pipeline_results['experiments_extracted'] = len(extracted_experiments)
            
            # Step 4: Normalize and Validate
            logger.info("Step 4: Normalizing and validating data...")
            normalized_experiments = self.normalize_experiments(extracted_experiments)
            self.normalized_experiments = normalized_experiments
            pipeline_results['steps_completed'].append('normalize_validate')
            pipeline_results['experiments_normalized'] = len(normalized_experiments)
            
            # Step 5: Human Review (optional)
            if self.config.get('enable_human_review', False):
                logger.info("Step 5: Human review interface...")
                self.launch_review_interface()
                pipeline_results['steps_completed'].append('human_review')
            
            # Step 6: Publish Review
            logger.info("Step 6: Publishing literature review...")
            review_results = self.publisher.publish_review(normalized_experiments, papers)
            pipeline_results['steps_completed'].append('publish_review')
            pipeline_results['review_results'] = review_results
            
            # Final summary
            pipeline_results['end_time'] = datetime.now().isoformat()
            pipeline_results['status'] = 'success'
            pipeline_results['total_experiments'] = len(normalized_experiments)
            pipeline_results['valid_experiments'] = sum(1 for exp in normalized_experiments 
                                                      if exp.validation_result.is_valid)
            
            logger.info(f"Pipeline completed successfully: {len(normalized_experiments)} experiments processed")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            pipeline_results['status'] = 'error'
            pipeline_results['error'] = str(e)
            pipeline_results['end_time'] = datetime.now().isoformat()
        
        # Save pipeline results
        results_file = self.output_dir / 'pipeline_results.json'
        with open(results_file, 'w') as f:
            json.dump(pipeline_results, f, indent=2, default=str)
        
        return pipeline_results
    
    def parse_documents(self, papers: List[Any]) -> List[Any]:
        """Parse downloaded documents"""
        parsed_docs = []
        
        for paper in papers:
            try:
                # Check if PDF is available
                if hasattr(paper, 'sections_json') and 'pdf_path' in paper.sections_json:
                    pdf_path = Path(paper.sections_json['pdf_path'])
                    if pdf_path.exists():
                        parsed_doc = self.parser.parse_document(pdf_path)
                        if parsed_doc:
                            parsed_docs.append(parsed_doc)
                else:
                    # Create basic document from metadata
                    parsed_doc = self.create_basic_document(paper)
                    if parsed_doc:
                        parsed_docs.append(parsed_doc)
                        
            except Exception as e:
                logger.error(f"Error parsing document {paper.paper_id}: {e}")
                continue
        
        return parsed_docs
    
    def create_basic_document(self, paper: Any) -> Optional[Any]:
        """Create basic document from paper metadata"""
        try:
            from parsing.parser import ParsedDocument, DocumentSection
            
            # Create basic sections
            sections = []
            if hasattr(paper, 'abstract') and paper.abstract:
                sections.append(DocumentSection(
                    section_type='abstract',
                    content=paper.abstract,
                    page_number=0,
                    start_char=0,
                    end_char=len(paper.abstract),
                    confidence=0.5,
                    metadata={}
                ))
            
            # Create parsed document
            parsed_doc = ParsedDocument(
                document_id=paper.paper_id,
                document_type='paper',
                title=paper.title,
                authors=paper.authors,
                abstract=paper.abstract or '',
                sections=sections,
                tables=[],
                figures=[],
                references=[],
                metadata={'source': 'metadata_only'},
                parsed_at=datetime.now()
            )
            
            return parsed_doc
            
        except Exception as e:
            logger.error(f"Error creating basic document: {e}")
            return None
    
    def extract_experiments(self, parsed_docs: List[Any]) -> List[Any]:
        """Extract experiments from parsed documents"""
        all_experiments = []
        
        for doc in parsed_docs:
            try:
                experiments = self.extractor.extract_experiments(doc)
                all_experiments.extend(experiments)
            except Exception as e:
                logger.error(f"Error extracting experiments from {doc.document_id}: {e}")
                continue
        
        return all_experiments
    
    def normalize_experiments(self, experiments: List[Any]) -> List[Any]:
        """Normalize and validate experiments"""
        normalized_experiments = []
        
        for exp in experiments:
            try:
                normalized_exp = self.normalizer.normalize_experiment(exp)
                normalized_experiments.append(normalized_exp)
            except Exception as e:
                logger.error(f"Error normalizing experiment {exp.experiment_id}: {e}")
                continue
        
        # Remove duplicates
        normalized_experiments = self.normalizer.deduplicate_experiments(normalized_experiments)
        
        return normalized_experiments
    
    def launch_review_interface(self):
        """Launch human review interface"""
        try:
            import streamlit.web.cli as stcli
            import sys
            import subprocess
            
            # Save current experiments for review
            review_file = self.output_dir / 'experiments_for_review.json'
            experiments_data = []
            for exp in self.normalized_experiments:
                exp_dict = {
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
                    'confidence_0_1': exp.confidence_0_1,
                    'parameters': [
                        {
                            'parameter_name': param.parameter_name,
                            'original_value': param.original_value,
                            'original_unit': param.original_unit,
                            'normalized_value': param.normalized_value,
                            'normalized_unit': param.normalized_unit,
                            'confidence': param.confidence,
                            'metadata': param.metadata
                        }
                        for param in exp.parameters
                    ]
                }
                experiments_data.append(exp_dict)
            
            with open(review_file, 'w') as f:
                json.dump(experiments_data, f, indent=2, default=str)
            
            logger.info(f"Review interface data saved to {review_file}")
            logger.info("To launch review interface, run: streamlit run literature/review/review_app.py")
            
        except Exception as e:
            logger.error(f"Error launching review interface: {e}")
    
    def run_step(self, step_name: str, **kwargs) -> Dict[str, Any]:
        """Run a specific pipeline step"""
        logger.info(f"Running pipeline step: {step_name}")
        
        try:
            if step_name == 'discover_download':
                queries = kwargs.get('queries', {})
                max_results = kwargs.get('max_results_per_query', 1000)
                papers, patents = self.downloader.run_discovery(queries, max_results)
                return {'papers': papers, 'patents': patents}
            
            elif step_name == 'parse_segment':
                papers = kwargs.get('papers', [])
                parsed_docs = self.parse_documents(papers)
                return {'parsed_documents': parsed_docs}
            
            elif step_name == 'extract_data':
                parsed_docs = kwargs.get('parsed_documents', [])
                experiments = self.extract_experiments(parsed_docs)
                return {'experiments': experiments}
            
            elif step_name == 'normalize_validate':
                experiments = kwargs.get('experiments', [])
                normalized_experiments = self.normalize_experiments(experiments)
                return {'normalized_experiments': normalized_experiments}
            
            elif step_name == 'publish_review':
                experiments = kwargs.get('normalized_experiments', [])
                papers = kwargs.get('papers', [])
                review_results = self.publisher.publish_review(experiments, papers)
                return {'review_results': review_results}
            
            else:
                raise ValueError(f"Unknown step: {step_name}")
                
        except Exception as e:
            logger.error(f"Error running step {step_name}: {e}")
            return {'error': str(e)}

def main():
    """Main entry point for the literature processing pipeline"""
    parser = argparse.ArgumentParser(description='Truffle Literature Processing Pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Configuration file path')
    parser.add_argument('--step', type=str, choices=[
        'discover_download', 'parse_segment', 'extract_data', 
        'normalize_validate', 'human_review', 'publish_review', 'full'
    ], default='full', help='Pipeline step to run')
    parser.add_argument('--queries', type=str, default='literature/queries.json',
                       help='Queries file path')
    parser.add_argument('--max-results', type=int, default=1000,
                       help='Maximum results per query')
    parser.add_argument('--enable-review', action='store_true',
                       help='Enable human review interface')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add command line arguments to config
    config['enable_human_review'] = args.enable_review
    
    # Load queries
    with open(args.queries, 'r') as f:
        queries = json.load(f)
    
    # Initialize pipeline
    pipeline = LiteratureProcessingPipeline(config)
    
    if args.step == 'full':
        # Run full pipeline
        results = pipeline.run_full_pipeline(queries, args.max_results)
        print(f"Pipeline completed: {results['status']}")
        print(f"Total experiments: {results.get('total_experiments', 0)}")
        print(f"Valid experiments: {results.get('valid_experiments', 0)}")
    else:
        # Run specific step
        results = pipeline.run_step(args.step, queries=queries, max_results_per_query=args.max_results)
        print(f"Step {args.step} completed")
        print(f"Results: {results}")

if __name__ == "__main__":
    main()