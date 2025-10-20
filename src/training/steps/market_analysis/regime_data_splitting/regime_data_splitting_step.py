"""
Regime Data Splitting Step

BaseStep-based implementation for regime data splitting.
Migrated from the old pipeline pattern to use the new BaseStep architecture.
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path

from src.training.steps.base_step import BaseStep
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error
from src.utils.logger import system_logger

logger = logging.getLogger(__name__)


class RegimeDataSplittingStep(BaseStep):
    """
    Regime Data Splitting Step using BaseStep pattern.
    
    Splits market data into regime-specific datasets for training.
    """
    
    def __init__(self, step_name: str = "regime_data_splitting"):
        """Initialize the regime data splitting step."""
        super().__init__(step_name)
        self.logger = system_logger.getChild('RegimeDataSplittingStep')
        
        tprint("✅ RegimeDataSplittingStep initialized", "SUCCESS")
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute regime data splitting step.
        
        Args:
            config: Configuration dictionary with parameters:
                - symbol: Trading symbol (e.g., 'ETHUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Timeframe (e.g., '15m')
                - data_dir: Data directory path
                - start_date: Start date (optional)
                - end_date: End date (optional)
                - regime_labels: Regime labels to split on
                - split_ratios: Train/validation/test ratios (default: [0.7, 0.15, 0.15])
                
        Returns:
            Dictionary with execution results and split datasets
        """
        start_time = datetime.now()
        
        try:
            tprint(f"🔍 Starting regime data splitting for {config.get('symbol', 'UNKNOWN')}", "INFO")
            
            # Set context for artifact management
            self._set_context(
                symbol=config.get('symbol', 'ETHUSDT'),
                exchange=config.get('exchange', 'binance'),
                direction=config.get('direction', 'both'),
                model=config.get('model', 'default')
            )
            
            # Load market data
            market_data = self._load_dataframe('market_data')
            if market_data is None:
                # Try alternative artifact names
                market_data = self._load_dataframe('processed_data') or self._load_dataframe('data')
                
            if market_data is None:
                raise ValueError("No market data found in artifacts")
            
            tprint(f"✅ Loaded market data: {market_data.shape[0]} rows, {market_data.shape[1]} columns", "SUCCESS")
            
            # Load regime labels
            regime_labels = self._load_regime_labels(config)
            if regime_labels is None:
                raise ValueError("No regime labels found")
            
            tprint(f"✅ Loaded regime labels: {len(regime_labels)} labels", "SUCCESS")
            
            # Validate data alignment
            if len(market_data) != len(regime_labels):
                raise ValueError(f"Data length mismatch: {len(market_data)} vs {len(regime_labels)}")
            
            # Perform regime data splitting
            split_datasets = self._split_data_by_regimes(market_data, regime_labels, config)
            
            # Save split datasets
            self._save_split_datasets(split_datasets, config)
            
            # Calculate metrics
            metrics = self._calculate_split_metrics(split_datasets, start_time, config)
            
            # Create outcome report
            outcome_report = self._create_outcome_report(split_datasets, metrics, config)
            
            tprint(f"✅ Regime data splitting completed: {len(split_datasets)} datasets", "SUCCESS")
            
            return {
                'success': True,
                'split_datasets': split_datasets,
                'metrics': metrics,
                'outcome_report': outcome_report,
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
            
        except Exception as e:
            error_msg = f"Regime data splitting failed: {str(e)}"
            tprint(f"❌ {error_msg}", "ERROR")
            self.logger.error(error_msg)
            
            return {
                'success': False,
                'error': error_msg,
                'split_datasets': {},
                'metrics': {},
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
    
    def _load_regime_labels(self, config: Dict[str, Any]) -> Optional[np.ndarray]:
        """Load regime labels from artifacts or config."""
        try:
            # Try to load from artifacts first
            regime_data = self._get_artifact('regime_labels')
            if regime_data is not None:
                if isinstance(regime_data, dict) and 'labels' in regime_data:
                    return np.array(regime_data['labels'])
                elif isinstance(regime_data, (list, np.ndarray)):
                    return np.array(regime_data)
            
            # Try to load from config
            if 'regime_labels' in config:
                return np.array(config['regime_labels'])
            
            # Try to load from file
            regime_file = config.get('regime_labels_file')
            if regime_file and Path(regime_file).exists():
                regime_data = pd.read_parquet(regime_file)
                if 'regime_label' in regime_data.columns:
                    return regime_data['regime_label'].values
                elif 'labels' in regime_data.columns:
                    return regime_data['labels'].values
            
            return None
            
        except Exception as e:
            tprint(f"⚠️ Failed to load regime labels: {e}", "WARNING")
            return None
    
    def _split_data_by_regimes(self, data: pd.DataFrame, regime_labels: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
        """Split data by regime labels."""
        try:
            split_ratios = config.get('split_ratios', [0.7, 0.15, 0.15])
            if len(split_ratios) != 3 or abs(sum(split_ratios) - 1.0) > 1e-6:
                split_ratios = [0.7, 0.15, 0.15]
                tprint("⚠️ Invalid split ratios, using default [0.7, 0.15, 0.15]", "WARNING")
            
            # Get unique regimes
            unique_regimes = np.unique(regime_labels)
            unique_regimes = unique_regimes[unique_regimes >= 0]  # Remove noise (-1)
            
            tprint(f"📊 Found {len(unique_regimes)} regimes: {unique_regimes}", "INFO")
            
            split_datasets = {
                'train': {},
                'validation': {},
                'test': {},
                'regime_stats': {},
                'metadata': {
                    'total_samples': len(data),
                    'n_regimes': len(unique_regimes),
                    'split_ratios': split_ratios,
                    'regime_distribution': {}
                }
            }
            
            # Split each regime
            for regime_id in unique_regimes:
                regime_mask = regime_labels == regime_id
                regime_data = data[regime_mask].copy()
                
                if len(regime_data) < 10:  # Skip regimes with too few samples
                    tprint(f"⚠️ Skipping regime {regime_id}: only {len(regime_data)} samples", "WARNING")
                    continue
                
                # Calculate split sizes
                n_samples = len(regime_data)
                n_train = int(n_samples * split_ratios[0])
                n_val = int(n_samples * split_ratios[1])
                n_test = n_samples - n_train - n_val
                
                # Shuffle data
                regime_data = regime_data.sample(frac=1.0, random_state=42)
                
                # Split data
                train_data = regime_data.iloc[:n_train]
                val_data = regime_data.iloc[n_train:n_train + n_val]
                test_data = regime_data.iloc[n_train + n_val:]
                
                # Store splits
                split_datasets['train'][f'regime_{regime_id}'] = train_data
                split_datasets['validation'][f'regime_{regime_id}'] = val_data
                split_datasets['test'][f'regime_{regime_id}'] = test_data
                
                # Store regime stats
                split_datasets['regime_stats'][f'regime_{regime_id}'] = {
                    'total_samples': n_samples,
                    'train_samples': len(train_data),
                    'val_samples': len(val_data),
                    'test_samples': len(test_data),
                    'train_ratio': len(train_data) / n_samples,
                    'val_ratio': len(val_data) / n_samples,
                    'test_ratio': len(test_data) / n_samples
                }
                
                split_datasets['metadata']['regime_distribution'][f'regime_{regime_id}'] = n_samples
            
            return split_datasets
            
        except Exception as e:
            tprint(f"❌ Failed to split data by regimes: {e}", "ERROR")
            raise
    
    def _save_split_datasets(self, split_datasets: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Save split datasets using artifact manager."""
        try:
            # Save each split
            for split_name in ['train', 'validation', 'test']:
                if split_name in split_datasets:
                    self._save_artifact(f'{split_name}_data', split_datasets[split_name])
            
            # Save regime stats
            self._save_artifact('regime_stats', split_datasets['regime_stats'])
            
            # Save metadata
            self._save_metadata(split_datasets['metadata'])
            
            tprint("✅ Split datasets saved to artifacts", "SUCCESS")
            
        except Exception as e:
            tprint(f"⚠️ Failed to save split datasets: {e}", "WARNING")
    
    def _calculate_split_metrics(self, split_datasets: Dict[str, Any], start_time: datetime, config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate splitting metrics."""
        try:
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate basic metrics
            total_samples = split_datasets['metadata']['total_samples']
            n_regimes = split_datasets['metadata']['n_regimes']
            
            # Calculate split distribution
            train_samples = sum(len(data) for data in split_datasets['train'].values())
            val_samples = sum(len(data) for data in split_datasets['validation'].values())
            test_samples = sum(len(data) for data in split_datasets['test'].values())
            
            metrics = {
                'processing_time_seconds': processing_time,
                'total_samples': total_samples,
                'n_regimes': n_regimes,
                'train_samples': train_samples,
                'validation_samples': val_samples,
                'test_samples': test_samples,
                'train_ratio': train_samples / total_samples if total_samples > 0 else 0,
                'validation_ratio': val_samples / total_samples if total_samples > 0 else 0,
                'test_ratio': test_samples / total_samples if total_samples > 0 else 0,
                'regime_distribution': split_datasets['metadata']['regime_distribution'],
                'success': True
            }
            
            return metrics
            
        except Exception as e:
            tprint(f"⚠️ Failed to calculate metrics: {e}", "WARNING")
            return {'success': False, 'error': str(e)}
    
    def _create_outcome_report(self, split_datasets: Dict[str, Any], metrics: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Create outcome report markdown."""
        try:
            report = f"""# Regime Data Splitting Outcome Report

## Execution Summary
- **Symbol**: {config.get('symbol', 'UNKNOWN')}
- **Exchange**: {config.get('exchange', 'UNKNOWN')}
- **Timeframe**: {config.get('timeframe', 'UNKNOWN')}
- **Processing Time**: {metrics.get('processing_time_seconds', 0):.2f} seconds
- **Success**: {'✅ Yes' if metrics.get('success', False) else '❌ No'}

## Data Splitting Results
- **Total Samples**: {metrics.get('total_samples', 0):,}
- **Number of Regimes**: {metrics.get('n_regimes', 0)}
- **Train Samples**: {metrics.get('train_samples', 0):,} ({metrics.get('train_ratio', 0):.1%})
- **Validation Samples**: {metrics.get('validation_samples', 0):,} ({metrics.get('validation_ratio', 0):.1%})
- **Test Samples**: {metrics.get('test_samples', 0):,} ({metrics.get('test_ratio', 0):.1%})

## Regime Distribution
"""
            
            regime_dist = metrics.get('regime_distribution', {})
            for regime, count in regime_dist.items():
                report += f"- **{regime}**: {count:,} samples\n"
            
            report += f"""
## Configuration
- **Split Ratios**: {config.get('split_ratios', [0.7, 0.15, 0.15])}
- **Random State**: 42 (for reproducibility)

## Generated Artifacts
- Train data (by regime)
- Validation data (by regime)
- Test data (by regime)
- Regime statistics
- Metadata

---
*Generated by Regime Data Splitting Step at {datetime.now().isoformat()}*
"""
            
            return report
            
        except Exception as e:
            tprint(f"⚠️ Failed to create outcome report: {e}", "WARNING")
            return f"# Regime Data Splitting Outcome Report\n\nError creating report: {str(e)}"


# Register the step
def register_regime_data_splitting_step():
    """Register the regime data splitting step."""
    from src.training.steps.base_step import step_registry
    
    step_registry.register("regime_data_splitting", RegimeDataSplittingStep)
    tprint("✅ Regime data splitting step registered", "SUCCESS")


# Auto-register when module is imported
register_regime_data_splitting_step()