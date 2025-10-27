"""
Enhanced Regime Data Splitting Step with BaseStep Integration.

This step provides comprehensive regime data splitting with:
- BaseStep inheritance for autonomous pipeline execution
- Multi-timeframe support (1h and 15m minimum)
- Regime probability tagging for each timeframe
- Regime-specific data splitting with comprehensive tags
- Enhanced metadata and statistics
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
import json
import pickle

# Handle optional dependencies gracefully
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import tprint
from src.utils.pipeline_standards import pipeline_standards

logger = logging.getLogger(__name__)


class EnhancedRegimeDataSplittingStep(BaseStep):
    """
    Enhanced Regime Data Splitting Step with BaseStep integration.
    
    Features:
    - Inherits from BaseStep for autonomous pipeline execution
    - Multi-timeframe support (1h and 15m minimum)
    - Regime probability tagging for each timeframe
    - Regime-specific data splitting with comprehensive tags
    - Enhanced metadata and statistics
    """

    def __init__(self, step_name: str = "enhanced_regime_data_splitting"):
        """Initialize the enhanced regime data splitting step."""
        super().__init__(step_name)
        self.logger = system_logger.getChild('EnhancedRegimeDataSplitting')
        self.standards = pipeline_standards
        
        # Supported timeframes
        self.supported_timeframes = ['1h', '15m']
        
        # Regime configuration
        self.regime_config = {
            'min_samples_per_regime': 100,
            'max_regimes': 10,
            'probability_threshold': 0.1
        }

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute enhanced regime data splitting with multi-timeframe support.

        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol (e.g., 'ETHUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframes: List of timeframes (e.g., ['1h', '15m'])
                - execution_mode: 'full', 'light', or 'blank'

        Returns:
            Dict containing:
            - 'success': bool indicating if step completed successfully
            - 'artifacts': dict of created artifacts
            - 'metrics': dict of performance metrics
            - 'error': error message if step failed (optional)
        """
        tprint(f"📊 Starting enhanced regime data splitting for {config.get('symbol', 'UNKNOWN')}", "INFO")

        try:
            # Extract configuration
            symbol = config.get('symbol')
            exchange = config.get('exchange')
            timeframes = config.get('timeframes', self.supported_timeframes)
            execution_mode = config.get('execution_mode', 'light')
            
            # Validate required parameters
            if not all([symbol, exchange]):
                error_msg = "Missing required parameters: symbol, exchange"
                tprint(f"❌ {error_msg}", "ERROR")
                return {
                    'success': False,
                    'error': error_msg,
                    'artifacts': {},
                    'metrics': {}
                }

            # Validate timeframes
            invalid_timeframes = [tf for tf in timeframes if tf not in self.supported_timeframes]
            if invalid_timeframes:
                error_msg = f"Unsupported timeframes: {invalid_timeframes}. Supported: {self.supported_timeframes}"
                tprint(f"❌ {error_msg}", "ERROR")
                return {
                    'success': False,
                    'error': error_msg,
                    'artifacts': {},
                    'metrics': {}
                }

            # Execute regime data splitting for each timeframe
            results = {}
            all_artifacts = {}
            all_metrics = {}
            
            for timeframe in timeframes:
                tprint(f"🔄 Processing timeframe: {timeframe}", "INFO")
                
                try:
                    timeframe_result = await self._process_timeframe(
                        symbol, exchange, timeframe, execution_mode
                    )
                    results[timeframe] = timeframe_result
                    
                    if timeframe_result['success']:
                        all_artifacts.update(timeframe_result['artifacts'])
                        all_metrics.update(timeframe_result['metrics'])
                    else:
                        tprint(f"⚠️ Failed to process timeframe {timeframe}: {timeframe_result.get('error', 'Unknown error')}", "WARNING")
                        
                except Exception as e:
                    error_msg = f"Error processing timeframe {timeframe}: {str(e)}"
                    tprint(f"❌ {error_msg}", "ERROR")
                    results[timeframe] = {
                        'success': False,
                        'error': error_msg,
                        'artifacts': {},
                        'metrics': {}
                    }

            # Create comprehensive summary
            successful_timeframes = [tf for tf, result in results.items() if result['success']]
            failed_timeframes = [tf for tf, result in results.items() if not result['success']]
            
            if not successful_timeframes:
                error_msg = f"Failed to process any timeframes. Failed: {failed_timeframes}"
                tprint(f"❌ {error_msg}", "ERROR")
                return {
                    'success': False,
                    'error': error_msg,
                    'artifacts': all_artifacts,
                    'metrics': all_metrics
                }

            # Create cross-timeframe analysis
            cross_timeframe_analysis = await self._create_cross_timeframe_analysis(results)
            all_artifacts['cross_timeframe_analysis'] = cross_timeframe_analysis

            # Generate comprehensive metrics
            comprehensive_metrics = self._generate_comprehensive_metrics(results, all_metrics)
            
            tprint(f"✅ Enhanced regime data splitting completed for {len(successful_timeframes)} timeframes", "SUCCESS")
            if failed_timeframes:
                tprint(f"⚠️ Failed timeframes: {failed_timeframes}", "WARNING")

            return {
                'success': True,
                'artifacts': all_artifacts,
                'metrics': comprehensive_metrics,
                'timeframe_results': results,
                'successful_timeframes': successful_timeframes,
                'failed_timeframes': failed_timeframes
            }

        except Exception as e:
            error_msg = f"Enhanced regime data splitting failed: {str(e)}"
            tprint(f"❌ {error_msg}", "ERROR")
            self.logger.exception(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'artifacts': {},
                'metrics': {}
            }

    async def _process_timeframe(self, symbol: str, exchange: str, timeframe: str, execution_mode: str) -> Dict[str, Any]:
        """
        Process regime data splitting for a specific timeframe.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe (e.g., '1h', '15m')
            execution_mode: Execution mode
            
        Returns:
            Dict with success status, artifacts, and metrics
        """
        try:
            tprint(f"🔄 Processing {symbol} on {exchange} ({timeframe})", "INFO")
            
            # Load market data
            market_data = await self._load_market_data(symbol, exchange, timeframe)
            if market_data is None or market_data.empty:
                return {
                    'success': False,
                    'error': f"No market data found for {symbol} on {exchange} ({timeframe})",
                    'artifacts': {},
                    'metrics': {}
                }

            # Load regime data
            regime_data = await self._load_regime_data(symbol, exchange, timeframe)
            if regime_data is None or regime_data.empty:
                return {
                    'success': False,
                    'error': f"No regime data found for {symbol} on {exchange} ({timeframe})",
                    'artifacts': {},
                    'metrics': {}
                }

            # Merge market data with regime information
            merged_data = await self._merge_market_and_regime_data(market_data, regime_data, timeframe)
            
            # Create regime-specific splits with tags
            regime_splits = await self._create_regime_splits(merged_data, timeframe)
            
            # Generate regime probability tags
            tagged_data = await self._generate_regime_probability_tags(merged_data, regime_data, timeframe)
            
            # Create artifacts
            artifacts = await self._create_timeframe_artifacts(
                tagged_data, regime_splits, symbol, exchange, timeframe
            )
            
            # Generate metrics
            metrics = self._generate_timeframe_metrics(
                tagged_data, regime_splits, symbol, exchange, timeframe
            )
            
            return {
                'success': True,
                'artifacts': artifacts,
                'metrics': metrics
            }
            
        except Exception as e:
            error_msg = f"Error processing timeframe {timeframe}: {str(e)}"
            self.logger.exception(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'artifacts': {},
                'metrics': {}
            }

    async def _load_market_data(self, symbol: str, exchange: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load market data for the specified symbol, exchange, and timeframe."""
        try:
            # Build data path using pipeline standards
            data_path = self.standards.build_path('historical_data', exchange, symbol)
            processed_path = Path(data_path) / 'processed' / f'{symbol.lower()}_{timeframe}'
            
            # Look for parquet files
            parquet_files = list(processed_path.glob('*.parquet'))
            if not parquet_files:
                # Fallback to unified data path
                unified_path = self.standards.build_path('unified_data', exchange, symbol)
                processed_path = Path(unified_path) / timeframe
                parquet_files = list(processed_path.glob('*.parquet'))
            
            if not parquet_files:
                self.logger.warning(f"No parquet files found for {symbol} on {exchange} ({timeframe})")
                return None
            
            # Load the most recent file
            latest_file = max(parquet_files, key=lambda x: x.stat().st_mtime)
            market_data = pd.read_parquet(latest_file)
            
            self.logger.info(f"Loaded market data: {len(market_data)} rows from {latest_file}")
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error loading market data for {symbol} on {exchange} ({timeframe}): {e}")
            return None

    async def _load_regime_data(self, symbol: str, exchange: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load regime data for the specified symbol, exchange, and timeframe."""
        try:
            # Build regime data path
            regime_path = self.standards.build_path('hmm_clusters', exchange, symbol)
            regime_file = Path(regime_path) / f'hmm_composite_clusters_{exchange}_{symbol}_{timeframe}.parquet'
            
            if not regime_file.exists():
                self.logger.warning(f"Regime file not found: {regime_file}")
                return None
            
            regime_data = pd.read_parquet(regime_file)
            self.logger.info(f"Loaded regime data: {len(regime_data)} rows from {regime_file}")
            return regime_data
            
        except Exception as e:
            self.logger.error(f"Error loading regime data for {symbol} on {exchange} ({timeframe}): {e}")
            return None

    async def _merge_market_and_regime_data(self, market_data: pd.DataFrame, regime_data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Merge market data with regime information."""
        try:
            # Ensure both DataFrames have a common index (timestamp)
            if 'timestamp' not in market_data.columns:
                if market_data.index.name == 'timestamp' or market_data.index.name is None:
                    market_data = market_data.reset_index()
            
            if 'timestamp' not in regime_data.columns:
                if regime_data.index.name == 'timestamp' or regime_data.index.name is None:
                    regime_data = regime_data.reset_index()
            
            # Convert timestamps to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(market_data['timestamp']):
                market_data['timestamp'] = pd.to_datetime(market_data['timestamp'])
            
            if not pd.api.types.is_datetime64_any_dtype(regime_data['timestamp']):
                regime_data['timestamp'] = pd.to_datetime(regime_data['timestamp'])
            
            # Merge on timestamp
            merged_data = pd.merge(
                market_data, 
                regime_data, 
                on='timestamp', 
                how='inner',
                suffixes=('_market', '_regime')
            )
            
            # Add timeframe information
            merged_data['timeframe'] = timeframe
            
            self.logger.info(f"Merged data: {len(merged_data)} rows for timeframe {timeframe}")
            return merged_data
            
        except Exception as e:
            self.logger.error(f"Error merging market and regime data for timeframe {timeframe}: {e}")
            raise

    async def _create_regime_splits(self, data: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """Create regime-specific data splits with comprehensive tags."""
        try:
            # Identify regime columns
            regime_columns = [col for col in data.columns if 'regime' in col.lower() and 'probability' not in col.lower()]
            
            if not regime_columns:
                self.logger.warning(f"No regime columns found for timeframe {timeframe}")
                return {}
            
            # Use the first regime column as primary
            primary_regime_col = regime_columns[0]
            regime_values = data[primary_regime_col].unique()
            
            regime_splits = {
                'timeframe': timeframe,
                'total_samples': len(data),
                'regimes': {},
                'splits': {
                    'train': {},
                    'validation': {},
                    'test': {}
                }
            }
            
            # Create splits for each regime
            for regime in regime_values:
                regime_mask = data[primary_regime_col] == regime
                regime_data = data[regime_mask]
                
                if len(regime_data) < self.regime_config['min_samples_per_regime']:
                    self.logger.warning(f"Regime {regime} has insufficient samples: {len(regime_data)}")
                    continue
                
                # Create temporal splits (70% train, 20% validation, 10% test)
                sorted_data = regime_data.sort_values('timestamp')
                n_samples = len(sorted_data)
                
                train_end = int(0.7 * n_samples)
                val_end = int(0.9 * n_samples)
                
                train_data = sorted_data.iloc[:train_end]
                val_data = sorted_data.iloc[train_end:val_end]
                test_data = sorted_data.iloc[val_end:]
                
                # Store regime information
                regime_splits['regimes'][str(regime)] = {
                    'total_samples': n_samples,
                    'train_samples': len(train_data),
                    'validation_samples': len(val_data),
                    'test_samples': len(test_data),
                    'start_time': str(sorted_data['timestamp'].min()),
                    'end_time': str(sorted_data['timestamp'].max()),
                    'regime_probability_columns': [col for col in data.columns if 'regime' in col.lower() and 'probability' in col.lower()]
                }
                
                # Store splits
                regime_splits['splits']['train'][str(regime)] = {
                    'data': train_data,
                    'samples': len(train_data),
                    'timeframe': timeframe
                }
                regime_splits['splits']['validation'][str(regime)] = {
                    'data': val_data,
                    'samples': len(val_data),
                    'timeframe': timeframe
                }
                regime_splits['splits']['test'][str(regime)] = {
                    'data': test_data,
                    'samples': len(test_data),
                    'timeframe': timeframe
                }
            
            self.logger.info(f"Created regime splits for {len(regime_splits['regimes'])} regimes in timeframe {timeframe}")
            return regime_splits
            
        except Exception as e:
            self.logger.error(f"Error creating regime splits for timeframe {timeframe}: {e}")
            raise

    async def _generate_regime_probability_tags(self, data: pd.DataFrame, regime_data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Generate comprehensive regime probability tags for the data."""
        try:
            tagged_data = data.copy()
            
            # Find probability columns
            prob_columns = [col for col in regime_data.columns if 'probability' in col.lower()]
            
            if not prob_columns:
                self.logger.warning(f"No probability columns found for timeframe {timeframe}")
                return tagged_data
            
            # Add probability columns to tagged data
            for prob_col in prob_columns:
                tagged_data[f'regime_prob_{prob_col}'] = regime_data[prob_col].values
            
            # Calculate additional probability metrics
            if len(prob_columns) > 0:
                # Get probability matrix
                prob_matrix = regime_data[prob_columns].values
                
                # Add confidence scores (max probability for each row)
                tagged_data['regime_confidence'] = np.max(prob_matrix, axis=1)
                
                # Add uncertainty (1 - confidence)
                tagged_data['regime_uncertainty'] = 1.0 - tagged_data['regime_confidence']
                
                # Add regime dominance (difference between highest and second highest probability)
                if prob_matrix.shape[1] > 1:
                    sorted_probs = np.sort(prob_matrix, axis=1)
                    tagged_data['regime_dominance'] = sorted_probs[:, -1] - sorted_probs[:, -2]
                else:
                    tagged_data['regime_dominance'] = tagged_data['regime_confidence']
                
                # Add regime stability (variance of probabilities)
                tagged_data['regime_stability'] = np.var(prob_matrix, axis=1)
            
            # Add timeframe-specific tags
            tagged_data['timeframe'] = timeframe
            tagged_data['regime_tag_version'] = '1.0'
            tagged_data['regime_tag_timestamp'] = datetime.now().isoformat()
            
            self.logger.info(f"Generated regime probability tags for {len(tagged_data)} samples in timeframe {timeframe}")
            return tagged_data
            
        except Exception as e:
            self.logger.error(f"Error generating regime probability tags for timeframe {timeframe}: {e}")
            raise

    async def _create_timeframe_artifacts(self, tagged_data: pd.DataFrame, regime_splits: Dict[str, Any], 
                                        symbol: str, exchange: str, timeframe: str) -> Dict[str, Any]:
        """Create artifacts for a specific timeframe."""
        try:
            artifacts = {}
            
            # Create artifact directory
            artifact_dir = Path("artifacts") / "enhanced_regime_data_splitting" / f"{exchange}_{symbol}_{timeframe}"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            
            # Save tagged data
            tagged_data_path = artifact_dir / f"tagged_data_{timeframe}.parquet"
            tagged_data.to_parquet(tagged_data_path)
            artifacts[f'tagged_data_{timeframe}'] = str(tagged_data_path)
            
            # Save regime splits (without data to avoid large files)
            splits_metadata = {
                'timeframe': timeframe,
                'total_samples': regime_splits.get('total_samples', 0),
                'regimes': regime_splits.get('regimes', {}),
                'split_summary': {
                    'train': {k: v['samples'] for k, v in regime_splits.get('splits', {}).get('train', {}).items()},
                    'validation': {k: v['samples'] for k, v in regime_splits.get('splits', {}).get('validation', {}).items()},
                    'test': {k: v['samples'] for k, v in regime_splits.get('splits', {}).get('test', {}).items()}
                }
            }
            
            splits_path = artifact_dir / f"regime_splits_{timeframe}.json"
            with open(splits_path, 'w') as f:
                json.dump(splits_metadata, f, indent=2, default=str)
            artifacts[f'regime_splits_{timeframe}'] = str(splits_path)
            
            # Save regime statistics
            regime_stats = self._calculate_regime_statistics(tagged_data, timeframe)
            stats_path = artifact_dir / f"regime_statistics_{timeframe}.json"
            with open(stats_path, 'w') as f:
                json.dump(regime_stats, f, indent=2, default=str)
            artifacts[f'regime_statistics_{timeframe}'] = str(stats_path)
            
            return artifacts
            
        except Exception as e:
            self.logger.error(f"Error creating artifacts for timeframe {timeframe}: {e}")
            raise

    def _calculate_regime_statistics(self, data: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """Calculate comprehensive regime statistics."""
        try:
            stats = {
                'timeframe': timeframe,
                'total_samples': len(data),
                'regime_columns': [col for col in data.columns if 'regime' in col.lower()],
                'probability_columns': [col for col in data.columns if 'regime_prob' in col.lower()],
                'statistics': {}
            }
            
            # Calculate statistics for each regime column
            regime_columns = [col for col in data.columns if 'regime' in col.lower() and 'probability' not in col.lower()]
            
            for col in regime_columns:
                if col in data.columns:
                    regime_values = data[col].value_counts()
                    stats['statistics'][col] = {
                        'unique_values': len(regime_values),
                        'value_counts': regime_values.to_dict(),
                        'most_common': regime_values.index[0] if len(regime_values) > 0 else None,
                        'least_common': regime_values.index[-1] if len(regime_values) > 0 else None
                    }
            
            # Calculate probability statistics
            prob_columns = [col for col in data.columns if 'regime_prob' in col.lower()]
            if prob_columns:
                prob_data = data[prob_columns]
                stats['probability_statistics'] = {
                    'mean_probabilities': prob_data.mean().to_dict(),
                    'std_probabilities': prob_data.std().to_dict(),
                    'min_probabilities': prob_data.min().to_dict(),
                    'max_probabilities': prob_data.max().to_dict()
                }
            
            # Calculate confidence statistics
            if 'regime_confidence' in data.columns:
                stats['confidence_statistics'] = {
                    'mean_confidence': data['regime_confidence'].mean(),
                    'std_confidence': data['regime_confidence'].std(),
                    'min_confidence': data['regime_confidence'].min(),
                    'max_confidence': data['regime_confidence'].max()
                }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating regime statistics for timeframe {timeframe}: {e}")
            return {'error': str(e)}

    def _generate_timeframe_metrics(self, tagged_data: pd.DataFrame, regime_splits: Dict[str, Any], 
                                   symbol: str, exchange: str, timeframe: str) -> Dict[str, Any]:
        """Generate comprehensive metrics for a specific timeframe."""
        try:
            metrics = {
                'timeframe': timeframe,
                'symbol': symbol,
                'exchange': exchange,
                'total_samples': len(tagged_data),
                'regime_count': len(regime_splits.get('regimes', {})),
                'data_quality': {
                    'completeness': 1.0 - (tagged_data.isnull().sum().sum() / (len(tagged_data) * len(tagged_data.columns))),
                    'duplicate_rows': tagged_data.duplicated().sum(),
                    'memory_usage_mb': tagged_data.memory_usage(deep=True).sum() / 1024 / 1024
                }
            }
            
            # Add regime-specific metrics
            if regime_splits.get('regimes'):
                regime_metrics = {}
                for regime_id, regime_info in regime_splits['regimes'].items():
                    regime_metrics[regime_id] = {
                        'total_samples': regime_info.get('total_samples', 0),
                        'train_samples': regime_info.get('train_samples', 0),
                        'validation_samples': regime_info.get('validation_samples', 0),
                        'test_samples': regime_info.get('test_samples', 0),
                        'sample_distribution': {
                            'train_pct': regime_info.get('train_samples', 0) / max(regime_info.get('total_samples', 1), 1),
                            'validation_pct': regime_info.get('validation_samples', 0) / max(regime_info.get('total_samples', 1), 1),
                            'test_pct': regime_info.get('test_samples', 0) / max(regime_info.get('total_samples', 1), 1)
                        }
                    }
                metrics['regime_metrics'] = regime_metrics
            
            # Add probability metrics
            prob_columns = [col for col in tagged_data.columns if 'regime_prob' in col.lower()]
            if prob_columns:
                prob_data = tagged_data[prob_columns]
                metrics['probability_metrics'] = {
                    'mean_probability': prob_data.mean().mean(),
                    'std_probability': prob_data.std().mean(),
                    'confidence_mean': tagged_data.get('regime_confidence', pd.Series([0])).mean(),
                    'uncertainty_mean': tagged_data.get('regime_uncertainty', pd.Series([0])).mean()
                }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error generating metrics for timeframe {timeframe}: {e}")
            return {'error': str(e)}

    async def _create_cross_timeframe_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create cross-timeframe analysis and comparison."""
        try:
            analysis = {
                'cross_timeframe_summary': {
                    'total_timeframes': len(results),
                    'successful_timeframes': len([r for r in results.values() if r.get('success', False)]),
                    'failed_timeframes': len([r for r in results.values() if not r.get('success', False)])
                },
                'timeframe_comparison': {},
                'regime_consistency': {}
            }
            
            # Compare metrics across timeframes
            successful_results = {tf: result for tf, result in results.items() if result.get('success', False)}
            
            if len(successful_results) > 1:
                # Compare regime counts
                regime_counts = {}
                for tf, result in successful_results.items():
                    metrics = result.get('metrics', {})
                    regime_counts[tf] = metrics.get('regime_count', 0)
                
                analysis['timeframe_comparison']['regime_counts'] = regime_counts
                
                # Compare sample counts
                sample_counts = {}
                for tf, result in successful_results.items():
                    metrics = result.get('metrics', {})
                    sample_counts[tf] = metrics.get('total_samples', 0)
                
                analysis['timeframe_comparison']['sample_counts'] = sample_counts
                
                # Calculate consistency metrics
                if regime_counts:
                    regime_values = list(regime_counts.values())
                    analysis['regime_consistency'] = {
                        'regime_count_std': np.std(regime_values) if NUMPY_AVAILABLE else 0,
                        'regime_count_mean': np.mean(regime_values) if NUMPY_AVAILABLE else 0,
                        'regime_count_cv': np.std(regime_values) / max(np.mean(regime_values), 1) if NUMPY_AVAILABLE else 0
                    }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error creating cross-timeframe analysis: {e}")
            return {'error': str(e)}

    def _generate_comprehensive_metrics(self, results: Dict[str, Any], all_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive metrics across all timeframes."""
        try:
            comprehensive_metrics = {
                'execution_summary': {
                    'total_timeframes': len(results),
                    'successful_timeframes': len([r for r in results.values() if r.get('success', False)]),
                    'failed_timeframes': len([r for r in results.values() if not r.get('success', False)]),
                    'execution_timestamp': datetime.now().isoformat()
                },
                'timeframe_metrics': all_metrics,
                'aggregate_statistics': {}
            }
            
            # Calculate aggregate statistics
            successful_results = [r for r in results.values() if r.get('success', False)]
            if successful_results:
                total_samples = sum(r.get('metrics', {}).get('total_samples', 0) for r in successful_results)
                total_regimes = sum(r.get('metrics', {}).get('regime_count', 0) for r in successful_results)
                
                comprehensive_metrics['aggregate_statistics'] = {
                    'total_samples_across_timeframes': total_samples,
                    'average_samples_per_timeframe': total_samples / len(successful_results),
                    'total_regimes_across_timeframes': total_regimes,
                    'average_regimes_per_timeframe': total_regimes / len(successful_results)
                }
            
            return comprehensive_metrics
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive metrics: {e}")
            return {'error': str(e)}
