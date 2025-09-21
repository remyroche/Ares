#!/usr/bin/env python3
"""
Clustering Pipeline

This module provides a complete pipeline for regime clustering that integrates
HMM discovery outputs with regime consolidation and ML output generation.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
import json

from .regime_consolidator import RegimeConsolidator, ConsolidationConfig, ConsolidationResult
from .hmm_integration import HMMDiscoveryIntegration, HMMDiscoveryData
from .ml_output_generator import MLOutputGenerator, MLTrainingDataset, ClusterProfile

logger = logging.getLogger(__name__)

class ClusteringPipeline:
    """
    Complete pipeline for regime clustering from HMM discovery to ML-ready outputs.
    
    This pipeline:
    1. Loads HMM discovery results
    2. Validates data quality
    3. Consolidates regimes into balanced clusters
    4. Generates ML-ready training datasets
    5. Provides comprehensive outputs and validation
    """
    
    def __init__(self, 
                 consolidation_config: Optional[ConsolidationConfig] = None,
                 output_base_dir: str = "training/steps/market_analysis/clustering"):
        """
        Initialize the clustering pipeline.
        
        Args:
            consolidation_config: Configuration for regime consolidation
            output_base_dir: Base directory for all outputs
        """
        self.consolidation_config = consolidation_config or ConsolidationConfig()
        self.output_base_dir = Path(output_base_dir)
        
        # Initialize components
        self.hmm_integration = HMMDiscoveryIntegration()
        self.consolidator = RegimeConsolidator(self.consolidation_config)
        self.ml_generator = MLOutputGenerator(
            output_dir=str(self.output_base_dir / "ml_outputs")
        )
        
        self.logger = logger.getChild("ClusteringPipeline")
        self.logger.info("ClusteringPipeline initialized")
    
    def run_complete_pipeline(self, 
                            hmm_results_file: Union[str, Path],
                            symbol: str,
                            timeframe: str,
                            save_outputs: bool = True) -> Dict[str, Any]:
        """
        Run the complete clustering pipeline.
        
        Args:
            hmm_results_file: Path to HMM discovery results JSON file
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Data timeframe (e.g., '1h', '15m')
            save_outputs: Whether to save all outputs to files
            
        Returns:
            Dictionary containing all pipeline results and outputs
        """
        
        pipeline_start = datetime.now()
        self.logger.info(f"Starting complete clustering pipeline for {symbol} {timeframe}")
        
        try:
            # Step 1: Load HMM discovery results
            self.logger.info("Step 1: Loading HMM discovery results")
            hmm_data = self.hmm_integration.load_hmm_discovery_results(hmm_results_file)
            
            # Step 2: Validate data quality
            self.logger.info("Step 2: Validating data quality")
            validation_results = self.hmm_integration.validate_for_clustering(hmm_data)
            
            if not validation_results['is_valid']:
                self.logger.warning("Data validation warnings found:")
                for warning in validation_results['warnings']:
                    self.logger.warning(f"  - {warning}")
            
            # Step 3: Create clustering input
            self.logger.info("Step 3: Creating clustering input")
            clustering_input = self.hmm_integration.create_clustering_input(hmm_data)
            
            # Step 4: Consolidate regimes
            self.logger.info("Step 4: Consolidating regimes")
            consolidation_result = self.consolidator.consolidate_regimes(clustering_input)
            
            # Step 5: Generate ML training dataset
            self.logger.info("Step 5: Generating ML training dataset")
            training_dataset = self.ml_generator.generate_training_dataset(
                consolidation_result, clustering_input
            )
            
            # Step 6: Generate cluster profiles
            self.logger.info("Step 6: Generating cluster profiles")
            cluster_profiles = self.ml_generator.generate_cluster_profiles(consolidation_result)
            
            # Step 7: Validate training dataset
            self.logger.info("Step 7: Validating training dataset")
            dataset_validation = self.ml_generator.validate_training_dataset(training_dataset)
            
            # Step 8: Generate training recommendations
            self.logger.info("Step 8: Generating training recommendations")
            training_recommendations = self.ml_generator.generate_training_recommendations(
                training_dataset, cluster_profiles
            )
            
            # Step 9: Generate feature importance report
            self.logger.info("Step 9: Generating feature importance report")
            feature_importance = self.ml_generator.create_feature_importance_report(training_dataset)
            
            # Calculate total processing time
            total_time = (datetime.now() - pipeline_start).total_seconds()
            
            # Compile results
            pipeline_results = {
                'metadata': {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'pipeline_start': pipeline_start.isoformat(),
                    'pipeline_end': datetime.now().isoformat(),
                    'total_processing_time': total_time,
                    'config': {
                        'target_clusters': self.consolidation_config.target_clusters,
                        'min_cluster_size_pct': self.consolidation_config.min_cluster_size_pct,
                        'max_cluster_size_pct': self.consolidation_config.max_cluster_size_pct,
                        'coverage_target': self.consolidation_config.coverage_target
                    }
                },
                'hmm_data': {
                    'regime_count': hmm_data.regime_count,
                    'total_samples': hmm_data.total_samples,
                    'original_data_points': hmm_data.original_data_points
                },
                'validation_results': validation_results,
                'consolidation_result': {
                    'original_regime_count': consolidation_result.original_regime_count,
                    'final_cluster_count': consolidation_result.final_cluster_count,
                    'coverage_percentage': consolidation_result.coverage_percentage,
                    'top_clusters_coverage': consolidation_result.top_clusters_coverage,
                    'balance_score': consolidation_result.balance_score,
                    'processing_time': consolidation_result.processing_time
                },
                'training_dataset': {
                    'n_samples': training_dataset.n_samples,
                    'n_features': training_dataset.n_features,
                    'n_clusters': training_dataset.n_clusters,
                    'feature_names': training_dataset.feature_names
                },
                'cluster_profiles': [
                    {
                        'cluster_id': profile.cluster_id,
                        'market_regime': profile.market_regime,
                        'sample_count': profile.sample_count,
                        'sample_percentage': profile.sample_percentage,
                        'is_trainable': profile.is_trainable
                    }
                    for profile in cluster_profiles
                ],
                'dataset_validation': dataset_validation,
                'training_recommendations': training_recommendations,
                'feature_importance': feature_importance.to_dict('records')
            }
            
            # Save outputs if requested
            saved_files = {}
            if save_outputs:
                self.logger.info("Saving pipeline outputs")
                saved_files = self.ml_generator.save_ml_outputs(
                    training_dataset, cluster_profiles, consolidation_result, symbol, timeframe
                )
                
                # Save complete pipeline results
                pipeline_results_file = self.output_base_dir / f"pipeline_results_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(pipeline_results_file, 'w') as f:
                    json.dump(pipeline_results, f, indent=2, default=str)
                
                saved_files['pipeline_results'] = pipeline_results_file
                pipeline_results['saved_files'] = saved_files
            
            self.logger.info(f"Pipeline completed successfully in {total_time:.2f}s")
            self.logger.info(f"Final results: {consolidation_result.final_cluster_count} clusters, "
                           f"{consolidation_result.top_clusters_coverage:.2%} coverage")
            
            return pipeline_results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
    
    def run_from_outcomes_dir(self,
                            outcomes_dir: Union[str, Path],
                            symbol: str,
                            timeframe: str,
                            save_outputs: bool = True) -> Dict[str, Any]:
        """
        Run pipeline using the latest HMM discovery results from outcomes directory.
        
        Args:
            outcomes_dir: Directory containing HMM discovery outcome files
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Data timeframe (e.g., '1h', '15m')
            save_outputs: Whether to save all outputs to files
            
        Returns:
            Dictionary containing all pipeline results and outputs
        """
        
        self.logger.info(f"Running pipeline from outcomes directory for {symbol} {timeframe}")
        
        # Load latest HMM results
        hmm_data = self.hmm_integration.load_latest_hmm_results(outcomes_dir, symbol, timeframe)
        
        # Create temporary results file
        temp_results_file = self.output_base_dir / f"temp_hmm_results_{symbol}_{timeframe}.json"
        
        # Save HMM data to temporary file for pipeline processing
        temp_results_data = {
            'metadata': hmm_data.metadata,
            'artifacts': {
                'hmm_regime_discovery_result': hmm_data.regime_characteristics
            }
        }
        
        with open(temp_results_file, 'w') as f:
            json.dump(temp_results_data, f, indent=2, default=str)
        
        try:
            # Run pipeline
            results = self.run_complete_pipeline(temp_results_file, symbol, timeframe, save_outputs)
            return results
        
        finally:
            # Clean up temporary file
            if temp_results_file.exists():
                temp_results_file.unlink()
    
    def validate_pipeline_results(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate pipeline results for quality and completeness.
        
        Args:
            pipeline_results: Results from run_complete_pipeline()
            
        Returns:
            Dictionary with validation results and recommendations
        """
        
        validation = {
            'is_valid': True,
            'warnings': [],
            'recommendations': [],
            'quality_metrics': {}
        }
        
        # Check consolidation quality
        consolidation = pipeline_results['consolidation_result']
        
        if consolidation['top_clusters_coverage'] < 0.90:
            validation['warnings'].append(f"Low top clusters coverage: {consolidation['top_clusters_coverage']:.2%}")
            validation['recommendations'].append("Consider adjusting consolidation parameters")
        
        if consolidation['balance_score'] < 0.6:
            validation['warnings'].append(f"Low balance score: {consolidation['balance_score']:.2f}")
            validation['recommendations'].append("Consider adjusting cluster size constraints")
        
        # Check training dataset quality
        dataset = pipeline_results['training_dataset']
        
        if dataset['n_samples'] < 1000:
            validation['warnings'].append(f"Small training dataset: {dataset['n_samples']} samples")
            validation['recommendations'].append("Consider collecting more data or adjusting parameters")
        
        # Check cluster trainability
        trainable_clusters = sum(1 for profile in pipeline_results['cluster_profiles'] 
                               if profile['is_trainable'])
        
        if trainable_clusters < 10:
            validation['warnings'].append(f"Few trainable clusters: {trainable_clusters}")
            validation['recommendations'].append("Consider focusing on top clusters for training")
        
        # Calculate quality metrics
        validation['quality_metrics'] = {
            'consolidation_quality': consolidation['balance_score'],
            'coverage_quality': consolidation['top_clusters_coverage'],
            'dataset_size': dataset['n_samples'],
            'trainable_clusters': trainable_clusters,
            'total_clusters': consolidation['final_cluster_count']
        }
        
        # Overall validation
        if validation['warnings']:
            validation['is_valid'] = False
        
        return validation
    
    def get_pipeline_summary(self, pipeline_results: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of pipeline results.
        
        Args:
            pipeline_results: Results from run_complete_pipeline()
            
        Returns:
            Formatted summary string
        """
        
        metadata = pipeline_results['metadata']
        hmm_data = pipeline_results['hmm_data']
        consolidation = pipeline_results['consolidation_result']
        dataset = pipeline_results['training_dataset']
        
        summary = f"""
=== REGIME CLUSTERING PIPELINE SUMMARY ===

Symbol: {metadata['symbol']}
Timeframe: {metadata['timeframe']}
Processing Time: {metadata['total_processing_time']:.2f}s

INPUT DATA:
- Original HMM Regimes: {hmm_data['regime_count']:,}
- Total Samples: {hmm_data['total_samples']:,}
- Original Data Points: {hmm_data['original_data_points']:,}

CONSOLIDATION RESULTS:
- Final Clusters: {consolidation['final_cluster_count']}
- Coverage: {consolidation['coverage_percentage']:.2%}
- Top Clusters Coverage: {consolidation['top_clusters_coverage']:.2%}
- Balance Score: {consolidation['balance_score']:.2f}

ML TRAINING DATASET:
- Samples: {dataset['n_samples']:,}
- Features: {dataset['n_features']}
- Clusters: {dataset['n_clusters']}
- Feature Names: {', '.join(dataset['feature_names'])}

TOP CLUSTERS:
"""
        
        # Add top 10 clusters
        for i, profile in enumerate(pipeline_results['cluster_profiles'][:10]):
            summary += f"  {i+1}. Cluster {profile['cluster_id']}: {profile['market_regime']} "
            summary += f"({profile['sample_percentage']:.1%}, {profile['sample_count']:,} samples)\n"
        
        # Add recommendations
        recommendations = pipeline_results['training_recommendations']
        if recommendations['model_types']:
            summary += f"\nRECOMMENDED MODELS: {', '.join(recommendations['model_types'][:3])}\n"
        
        summary += "\n" + "=" * 50
        
        return summary


def create_clustering_pipeline(
    target_clusters: int = 20,
    min_cluster_size_pct: float = 0.03,
    max_cluster_size_pct: float = 0.08,
    coverage_target: float = 0.95,
    output_base_dir: str = "training/steps/market_analysis/clustering"
) -> ClusteringPipeline:
    """
    Create and return a new ClusteringPipeline instance with custom configuration.
    
    Args:
        target_clusters: Number of target clusters
        min_cluster_size_pct: Minimum cluster size as percentage
        max_cluster_size_pct: Maximum cluster size as percentage
        coverage_target: Target coverage by top clusters
        output_base_dir: Base directory for outputs
        
    Returns:
        Configured ClusteringPipeline instance
    """
    
    from .regime_consolidator import create_consolidation_config
    
    config = create_consolidation_config(
        target_clusters=target_clusters,
        min_cluster_size_pct=min_cluster_size_pct,
        max_cluster_size_pct=max_cluster_size_pct,
        coverage_target=coverage_target
    )
    
    return ClusteringPipeline(config, output_base_dir)