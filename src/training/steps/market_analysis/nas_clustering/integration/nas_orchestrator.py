"""
NAS Orchestrator for pipeline integration.

This module provides the main orchestrator for NAS-driven clustering
with full pipeline compatibility and enhanced regime detection.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.nas_clusterer import NASClusterer, NASClusteringResult
from ..core.nas_config import NASConfig, NASClusteringConfig
from ..core.nas_feature_extractor import NASFeatureExtractor
from ..core.micro_regime_detector import MicroRegimeDetector
from ..components.nas_clustering_component import NASClusteringComponent
from ..components.nas_output_formatter import NASOutputFormatter

logger = logging.getLogger(__name__)


class NASOrchestrator:
    """Main orchestrator for NAS-driven clustering."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize NAS orchestrator.
        
        Args:
            config: Orchestrator configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize NAS configuration
        self.nas_config = NASConfig.create_default_config()
        if 'nas_config' in config:
            self.nas_config.update_config(config['nas_config'])
        
        # Initialize components
        self.nas_clusterer = NASClusterer(self.nas_config.clustering_config)
        self.feature_extractor = NASFeatureExtractor(self.nas_config.clustering_config.get_feature_config())
        self.micro_regime_detector = MicroRegimeDetector(self.nas_config.clustering_config.get_micro_regime_config())
        self.output_formatter = NASOutputFormatter(self.nas_config.clustering_config.__dict__)
        
        # Pipeline integration
        self.pipeline_compatible = True
        self.hmm_compatible = True
        
        self.logger.info(f"✅ NAS Orchestrator initialized for {self.nas_config.clustering_config.timeframe} timeframe")
    
    async def run_nas_clustering(self, data: Union[pd.DataFrame, np.ndarray],
                               timestamps: Optional[np.ndarray] = None,
                               symbol: str = "BTCUSDT",
                               exchange: str = "binance",
                               timeframe: str = "15m") -> Dict[str, Any]:
        """Run complete NAS clustering pipeline.
        
        Args:
            data: Market data for clustering
            timestamps: Optional timestamps array
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            
        Returns:
            Dictionary with NAS clustering results
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"🚀 Starting NAS clustering pipeline for {symbol} on {exchange} ({timeframe})")
            
            # Step 1: Extract NAS features
            self.logger.info("📊 Step 1: Extracting NAS features")
            feature_result = self.feature_extractor.extract_features(data, timestamps)
            
            if feature_result.features.size == 0:
                raise ValueError("No features extracted for NAS clustering")
            
            # Step 2: Detect micro-regimes
            self.logger.info("🔍 Step 2: Detecting micro-regimes")
            micro_regime_result = self.micro_regime_detector.detect_micro_regimes(
                data, timestamps, feature_result.features
            )
            
            # Step 3: Perform NAS clustering
            self.logger.info("🧠 Step 3: Performing NAS clustering")
            clustering_result = self.nas_clusterer.cluster(
                data, timestamps, optimize_parameters=True, generate_report=True
            )
            
            if not clustering_result.success:
                raise RuntimeError(f"NAS clustering failed: {clustering_result.error_message}")
            
            # Step 4: Format output for pipeline compatibility
            self.logger.info("📋 Step 4: Formatting output for pipeline compatibility")
            formatted_result = self.output_formatter.format_clustering_result(
                clustering_result, feature_result
            )
            
            # Step 5: Create timestamped regime data for LM training
            self.logger.info("🤖 Step 5: Creating timestamped regime data for LM training")
            timestamped_data = self._create_timestamped_regime_data(
                clustering_result, feature_result, micro_regime_result
            )
            
            # Step 6: Generate comprehensive report
            self.logger.info("📊 Step 6: Generating comprehensive report")
            comprehensive_report = self._generate_comprehensive_report(
                clustering_result, feature_result, micro_regime_result, timestamped_data
            )
            
            execution_time = time.time() - start_time
            
            # Create final result
            final_result = {
                'success': True,
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat(),
                'method': 'nas_clustering_pipeline',
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                
                # Core clustering results
                'clustering_result': clustering_result,
                'feature_result': feature_result,
                'micro_regime_result': micro_regime_result,
                
                # Formatted output
                'formatted_result': formatted_result,
                'timestamped_data': timestamped_data,
                'comprehensive_report': comprehensive_report,
                
                # Pipeline compatibility
                'pipeline_compatible': True,
                'hmm_compatible': True,
                'regime_data_available': True,
                'lm_training_ready': True
            }
            
            self.logger.info(f"✅ NAS clustering pipeline completed in {execution_time:.2f}s")
            return final_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ NAS clustering pipeline failed after {execution_time:.2f}s: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat()
            }
    
    def _create_timestamped_regime_data(self, clustering_result: NASClusteringResult,
                                       feature_result: Any,
                                       micro_regime_result: Any) -> Dict[str, Any]:
        """Create timestamped regime data for LM model training.
        
        Args:
            clustering_result: NAS clustering result
            feature_result: Feature extraction result
            micro_regime_result: Micro-regime detection result
            
        Returns:
            Dictionary with timestamped regime data
        """
        try:
            timestamped_data = {
                'regime_data': {
                    'regime_labels': clustering_result.labels.tolist(),
                    'regime_centers': clustering_result.cluster_centers.tolist(),
                    'regime_statistics': clustering_result.statistics,
                    'regime_quality_metrics': clustering_result.quality_metrics,
                    'regime_validation': clustering_result.validation,
                    'regime_metadata': clustering_result.metadata
                },
                
                'nas_data': {
                    'nas_architectures': clustering_result.nas_architectures,
                    'nas_score': clustering_result.quality_metrics.get('nas_score', 0.0),
                    'nas_architecture_type': clustering_result.metadata.get('nas_architecture_type', 'hybrid')
                },
                
                'micro_regime_data': {
                    'micro_regimes': micro_regime_result.micro_regimes.tolist() if micro_regime_result else [],
                    'micro_regime_types': [t.value for t in micro_regime_result.micro_regime_types] if micro_regime_result else [],
                    'micro_regime_scores': micro_regime_result.micro_regime_scores.tolist() if micro_regime_result else [],
                    'micro_regime_detection_accuracy': micro_regime_result.detection_accuracy if micro_regime_result else 0.0
                },
                
                'economic_data': {
                    'economic_significance_scores': clustering_result.economic_significance_scores.tolist(),
                    'trading_viability_scores': clustering_result.trading_viability_scores.tolist(),
                    'regime_transitions': clustering_result.regime_transitions.tolist() if clustering_result.regime_transitions is not None else []
                },
                
                'feature_data': {
                    'feature_names': feature_result.feature_names,
                    'feature_count': len(feature_result.feature_names),
                    'feature_metadata': feature_result.feature_metadata
                },
                
                'lm_training_data': {
                    'regime_sequences': clustering_result.labels.tolist(),
                    'regime_transitions': clustering_result.regime_transitions.tolist() if clustering_result.regime_transitions is not None else [],
                    'economic_significance': clustering_result.economic_significance_scores.tolist(),
                    'trading_viability': clustering_result.trading_viability_scores.tolist(),
                    'micro_regime_sequences': micro_regime_result.micro_regimes.tolist() if micro_regime_result else [],
                    'micro_regime_types': [t.value for t in micro_regime_result.micro_regime_types] if micro_regime_result else [],
                    'regime_statistics': clustering_result.statistics,
                    'regime_quality_metrics': clustering_result.quality_metrics,
                    'regime_metadata': clustering_result.metadata,
                    'nas_architectures': clustering_result.nas_architectures,
                    'timestamp': clustering_result.timestamp,
                    'execution_time': clustering_result.execution_time
                },
                
                'pipeline_metadata': {
                    'pipeline_compatible': True,
                    'hmm_compatible': True,
                    'regime_data_available': True,
                    'lm_training_ready': True,
                    'nas_enhanced': True,
                    'micro_regime_detection': True,
                    'economic_significance': True,
                    'trading_viability': True
                }
            }
            
            return timestamped_data
            
        except Exception as e:
            self.logger.error(f"❌ Timestamped regime data creation failed: {e}")
            return {}
    
    def _generate_comprehensive_report(self, clustering_result: NASClusteringResult,
                                     feature_result: Any,
                                     micro_regime_result: Any,
                                     timestamped_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive NAS clustering report.
        
        Args:
            clustering_result: NAS clustering result
            feature_result: Feature extraction result
            micro_regime_result: Micro-regime detection result
            timestamped_data: Timestamped regime data
            
        Returns:
            Dictionary with comprehensive report
        """
        try:
            report = {
                'executive_summary': {
                    'method': 'nas_clustering',
                    'timeframe': clustering_result.metadata.get('timeframe', '15m'),
                    'n_regimes': clustering_result.metadata.get('n_regimes', 0),
                    'nas_architecture_type': clustering_result.metadata.get('nas_architecture_type', 'hybrid'),
                    'success': clustering_result.success,
                    'execution_time': clustering_result.execution_time,
                    'timestamp': clustering_result.timestamp
                },
                
                'clustering_performance': {
                    'silhouette_score': clustering_result.quality_metrics.get('silhouette_score', 0.0),
                    'nas_score': clustering_result.quality_metrics.get('nas_score', 0.0),
                    'calinski_harabasz_score': clustering_result.quality_metrics.get('calinski_harabasz_score', 0.0),
                    'validation': clustering_result.validation
                },
                
                'regime_analysis': {
                    'regime_distribution': clustering_result.statistics.get('regime_distribution', {}),
                    'regime_percentages': clustering_result.statistics.get('regime_percentages', {}),
                    'n_clusters': clustering_result.statistics.get('n_clusters', 0),
                    'total_samples': clustering_result.statistics.get('total_samples', 0)
                },
                
                'nas_analysis': {
                    'nas_architectures': clustering_result.nas_architectures,
                    'nas_architecture_type': clustering_result.metadata.get('nas_architecture_type', 'hybrid'),
                    'nas_score': clustering_result.quality_metrics.get('nas_score', 0.0),
                    'nas_optimization': {
                        'economic_significance_threshold': clustering_result.metadata.get('economic_significance_threshold', 0.7),
                        'trading_viability_threshold': clustering_result.metadata.get('trading_viability_threshold', 0.6),
                        'regime_transition_cost': clustering_result.metadata.get('regime_transition_cost', 0.05)
                    }
                },
                
                'micro_regime_analysis': {
                    'micro_regimes_detected': len(micro_regime_result.micro_regime_types) if micro_regime_result else 0,
                    'micro_regime_types': [t.value for t in micro_regime_result.micro_regime_types] if micro_regime_result else [],
                    'detection_accuracy': micro_regime_result.detection_accuracy if micro_regime_result else 0.0,
                    'micro_regime_scores': micro_regime_result.micro_regime_scores.tolist() if micro_regime_result else []
                },
                
                'economic_analysis': {
                    'economic_significance_scores': clustering_result.economic_significance_scores.tolist(),
                    'mean_economic_significance': float(np.mean(clustering_result.economic_significance_scores)),
                    'trading_viability_scores': clustering_result.trading_viability_scores.tolist(),
                    'mean_trading_viability': float(np.mean(clustering_result.trading_viability_scores)),
                    'regime_transitions': clustering_result.regime_transitions.tolist() if clustering_result.regime_transitions is not None else []
                },
                
                'feature_analysis': {
                    'feature_count': len(feature_result.feature_names),
                    'feature_names': feature_result.feature_names,
                    'feature_metadata': feature_result.feature_metadata,
                    'execution_time': feature_result.execution_time
                },
                
                'pipeline_integration': {
                    'pipeline_compatible': True,
                    'hmm_compatible': True,
                    'regime_data_available': True,
                    'lm_training_ready': True,
                    'nas_enhanced': True,
                    'micro_regime_detection': True,
                    'economic_significance': True,
                    'trading_viability': True
                },
                
                'recommendations': {
                    'regime_quality': 'High' if clustering_result.quality_metrics.get('nas_score', 0.0) > 0.7 else 'Medium' if clustering_result.quality_metrics.get('nas_score', 0.0) > 0.5 else 'Low',
                    'trading_viability': 'High' if np.mean(clustering_result.trading_viability_scores) > 0.7 else 'Medium' if np.mean(clustering_result.trading_viability_scores) > 0.5 else 'Low',
                    'economic_significance': 'High' if np.mean(clustering_result.economic_significance_scores) > 0.7 else 'Medium' if np.mean(clustering_result.economic_significance_scores) > 0.5 else 'Low',
                    'micro_regime_detection': 'Good' if micro_regime_result.detection_accuracy > 0.7 else 'Fair' if micro_regime_result.detection_accuracy > 0.5 else 'Poor' if micro_regime_result else 'N/A'
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Comprehensive report generation failed: {e}")
            return {}
    
    def save_results(self, results: Dict[str, Any], output_dir: str) -> bool:
        """Save NAS clustering results to files.
        
        Args:
            results: NAS clustering results
            output_dir: Output directory path
            
        Returns:
            Success status
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save main results
            main_output_path = output_path / "nas_clustering_results.json"
            with open(main_output_path, 'w') as f:
                import json
                json.dump(results, f, indent=2, default=str)
            
            # Save timestamped regime data
            if 'timestamped_data' in results:
                regime_data_path = output_path / "timestamped_regime_data.json"
                with open(regime_data_path, 'w') as f:
                    json.dump(results['timestamped_data'], f, indent=2, default=str)
            
            # Save comprehensive report
            if 'comprehensive_report' in results:
                report_path = output_path / "comprehensive_report.json"
                with open(report_path, 'w') as f:
                    json.dump(results['comprehensive_report'], f, indent=2, default=str)
            
            self.logger.info(f"✅ NAS clustering results saved to {output_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save NAS clustering results: {e}")
            return False
    
    def load_results(self, input_dir: str) -> Dict[str, Any]:
        """Load NAS clustering results from files.
        
        Args:
            input_dir: Input directory path
            
        Returns:
            Loaded results dictionary
        """
        try:
            input_path = Path(input_dir)
            
            # Load main results
            main_input_path = input_path / "nas_clustering_results.json"
            if main_input_path.exists():
                with open(main_input_path, 'r') as f:
                    results = json.load(f)
                
                self.logger.info(f"✅ NAS clustering results loaded from {input_dir}")
                return results
            else:
                self.logger.warning(f"⚠️ No results found in {input_dir}")
                return {}
                
        except Exception as e:
            self.logger.error(f"❌ Failed to load NAS clustering results: {e}")
            return {}