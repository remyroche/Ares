#!/usr/bin/env python3
"""Data persistence functionality for HMM clustering."""

import json
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from src.utils.logger import system_logger
from src.utils.common_operations import ensure_directory, safe_json_dump

logger = system_logger.getChild("HMMDataPersistence")


class DataPersistenceMixin:
    """Mixin class for data persistence in HMM clustering."""
    
    def save_regime_results(self, 
                          results: Dict[str, Any],
                          symbol: str,
                          exchange: str,
                          timeframe: str,
                          data_dir: str = "data") -> Dict[str, Path]:
        """Save regime discovery results to disk.
        
        Args:
            results: Results dictionary from regime discovery
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Base data directory
            
        Returns:
            Dictionary of saved file paths
        """
        saved_files = {}
        
        try:
            # Create output directory
            out_dir = ensure_directory(Path(data_dir) / "hmm_regimes")
            
            # Save composite clusters DataFrame
            if 'composite_df' in results or 'regime_states' in results:
                # Create composite DataFrame if not present
                if 'composite_df' not in results and 'regime_states' in results:
                    regime_states = results['regime_states']
                    composite_df = pd.DataFrame({
                        'composite_cluster_id': regime_states,
                        'timestamp': pd.date_range(
                            start='2024-01-01',
                            periods=len(regime_states),
                            freq='1min'
                        )
                    })
                else:
                    composite_df = results['composite_df']
                
                # Save to parquet
                filename = f"{exchange}_{symbol}_{timeframe}_composite_clusters.parquet"
                filepath = out_dir / filename
                composite_df.to_parquet(filepath, index=False)
                saved_files['composite_clusters'] = filepath
                logger.info(f"✅ Saved composite clusters to: {filepath}")
                
                # Also save with alternate name for compatibility
                alt_filename = f"{exchange}_{symbol}_{timeframe}_hmm_regimes.parquet"
                alt_filepath = out_dir / alt_filename
                composite_df.to_parquet(alt_filepath, index=False)
                saved_files['hmm_regimes'] = alt_filepath
            
            # Save regime transitions
            if 'regime_transitions' in results:
                transitions_file = out_dir / f"{exchange}_{symbol}_{timeframe}_transitions.json"
                safe_json_dump(results['regime_transitions'], transitions_file)
                saved_files['transitions'] = transitions_file
                logger.info(f"✅ Saved transitions to: {transitions_file}")
            
            # Save metrics and metadata
            metadata = {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'n_regimes': results.get('n_regimes', 0),
                'total_periods': len(results.get('regime_states', [])),
                'regime_distribution': results.get('regime_distribution', {}),
                'execution_time': results.get('execution_time', 0),
                'economic_significance': results.get('economic_significance', False),
                'ensemble_quality': results.get('ensemble_quality', {}),
                'overall_quality_score': results.get('overall_quality_score', 0)
            }
            
            metadata_file = out_dir / f"{exchange}_{symbol}_{timeframe}_metadata.json"
            safe_json_dump(metadata, metadata_file, indent=2)
            saved_files['metadata'] = metadata_file
            logger.info(f"✅ Saved metadata to: {metadata_file}")
            
            # Save optimization results if present
            if 'optimized_params' in results:
                opt_file = out_dir / f"{exchange}_{symbol}_{timeframe}_optimized_params.json"
                safe_json_dump(results['optimized_params'], opt_file, indent=2)
                saved_files['optimized_params'] = opt_file
                logger.info(f"✅ Saved optimized parameters to: {opt_file}")
            
            # Save ML models if present
            if 'transition_models' in results and results['transition_models']:
                models_dir = ensure_directory(out_dir / "models")
                
                # Save feature list
                if 'selected_features' in results['transition_models']:
                    features_file = models_dir / f"{exchange}_{symbol}_{timeframe}_features.json"
                    safe_json_dump(
                        results['transition_models']['selected_features'],
                        features_file
                    )
                    saved_files['selected_features'] = features_file
                
                # Note: Actual model serialization would require joblib/pickle
                # This is just saving the model metadata
                model_info = {
                    'feature_selection_completed': results['transition_models'].get('feature_selection_completed', False),
                    'lgb_training_completed': results['transition_models'].get('lgb_training_completed', False),
                    'best_performance': results['transition_models'].get('best_performance', 0),
                    'final_performance': results['transition_models'].get('final_performance', {})
                }
                
                model_info_file = models_dir / f"{exchange}_{symbol}_{timeframe}_model_info.json"
                safe_json_dump(model_info, model_info_file)
                saved_files['model_info'] = model_info_file
            
            # Create summary report
            summary = {
                'execution_summary': {
                    'timestamp': datetime.now().isoformat(),
                    'success': True,
                    'files_saved': len(saved_files),
                    'file_paths': {k: str(v) for k, v in saved_files.items()}
                },
                'regime_summary': {
                    'n_regimes': results.get('n_regimes', 0),
                    'distribution': results.get('regime_distribution', {}),
                    'transitions': results.get('regime_transitions', {}).get('total_transitions', 0),
                    'quality_score': results.get('overall_quality_score', 0)
                }
            }
            
            summary_file = out_dir / f"{exchange}_{symbol}_{timeframe}_summary.json"
            safe_json_dump(summary, summary_file, indent=2)
            saved_files['summary'] = summary_file
            
            logger.info(f"✅ Saved {len(saved_files)} files successfully")
            return saved_files
            
        except Exception as e:
            logger.error(f"❌ Error saving regime results: {e}")
            return saved_files
    
    def load_regime_results(self,
                          symbol: str,
                          exchange: str,
                          timeframe: str,
                          data_dir: str = "data") -> Optional[Dict[str, Any]]:
        """Load previously saved regime results.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Base data directory
            
        Returns:
            Dictionary of loaded results or None
        """
        try:
            out_dir = Path(data_dir) / "hmm_regimes"
            
            results = {}
            
            # Load composite clusters
            clusters_file = out_dir / f"{exchange}_{symbol}_{timeframe}_composite_clusters.parquet"
            if clusters_file.exists():
                results['composite_df'] = pd.read_parquet(clusters_file)
                results['regime_states'] = results['composite_df']['composite_cluster_id'].tolist()
                logger.info(f"✅ Loaded composite clusters from: {clusters_file}")
            
            # Load metadata
            metadata_file = out_dir / f"{exchange}_{symbol}_{timeframe}_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                results.update(metadata)
                logger.info(f"✅ Loaded metadata from: {metadata_file}")
            
            # Load transitions
            transitions_file = out_dir / f"{exchange}_{symbol}_{timeframe}_transitions.json"
            if transitions_file.exists():
                with open(transitions_file, 'r') as f:
                    results['regime_transitions'] = json.load(f)
                logger.info(f"✅ Loaded transitions from: {transitions_file}")
            
            # Load optimized parameters
            opt_file = out_dir / f"{exchange}_{symbol}_{timeframe}_optimized_params.json"
            if opt_file.exists():
                with open(opt_file, 'r') as f:
                    results['optimized_params'] = json.load(f)
                logger.info(f"✅ Loaded optimized parameters from: {opt_file}")
            
            if results:
                logger.info(f"✅ Successfully loaded regime results for {symbol}")
                return results
            else:
                logger.warning(f"⚠️ No saved results found for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error loading regime results: {e}")
            return None