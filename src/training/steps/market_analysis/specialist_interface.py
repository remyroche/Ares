"""
Specialist Data Interface - Standardization Framework

This module provides standardized interfaces for all specialist models to ensure
consistent data structures, naming conventions, and ensemble compatibility.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Standard column naming conventions
STANDARD_PREDICTION_COLUMNS = {
    'primary_prediction': 'specialist_prediction',
    'primary_probability': 'specialist_probability', 
    'primary_score': 'specialist_score',
    'target_column': 'target_label',
    'timestamp': 'timestamp',
    'confidence': 'prediction_confidence',
    'regime': 'market_regime'
}

# Standard metadata structure
STANDARD_METADATA_FIELDS = {
    'specialist_name': str,
    'symbol': str,
    'exchange': str,
    'timeframe': str,
    'direction': str,
    'model_type': str,
    'target_range': str,
    'n_samples': int,
    'mi_score': float,
    'hsic_score': float,
    'binary_output': bool,
    'orthogonal_features': int,
    'high_correlation_pairs': int,
    'requirements_met': int,
    'timestamp': str
}


class SpecialistDataInterface:
    """Standard interface for all specialist data structures."""
    
    @staticmethod
    def standardize_prediction_data(df: pd.DataFrame, specialist_name: str) -> pd.DataFrame:
        """
        Standardize prediction data format for all specialists.
        
        Args:
            df: Original prediction DataFrame
            specialist_name: Name of the specialist
            
        Returns:
            Standardized DataFrame with consistent column names
        """
        # Start with all original columns to preserve features
        standardized = df.copy()
        
        # Ensure timestamp column exists
        if 'timestamp' not in df.columns:
            standardized['timestamp'] = df.index
            
        # Identify prediction columns with priority order
        prediction_candidates = [
            'specialist_prediction', 'prediction', 'probability', 'score',
            'momentum_persistence_prediction', 'vol_force_breakout', 
            'vol_force_volatility', 'vol_force_trend', 'smc_prediction'
        ]
        
        pred_col = None
        for candidate in prediction_candidates:
            matching_cols = [col for col in df.columns if candidate in col.lower()]
            if matching_cols:
                pred_col = matching_cols[0]
                break
                
        # Fallback to any column that looks like a prediction
        if not pred_col:
            fallback_cols = [col for col in df.columns if any(x in col.lower() 
                           for x in ['prediction', 'probability', 'score', 'force', 'regime'])]
            if fallback_cols:
                pred_col = fallback_cols[0]
        
        if pred_col:
            standardized['specialist_prediction'] = df[pred_col]
            standardized['specialist_probability'] = df[pred_col]  # Use same for now
            logger.info(f"Using {pred_col} as primary prediction for {specialist_name}")
        else:
            logger.warning(f"No prediction column found for {specialist_name}, using defaults")
            standardized['specialist_prediction'] = 0.5
            standardized['specialist_probability'] = 0.5
            
        # Standard target identification
        target_candidates = ['target_label', 'target', 'label', 'momentum_persistence_label']
        target_col = None
        
        for candidate in target_candidates:
            matching_cols = [col for col in df.columns if candidate in col.lower()]
            if matching_cols:
                target_col = matching_cols[0]
                break
                
        if target_col:
            standardized['target_label'] = df[target_col]
        else:
            # Create synthetic target from prediction distribution
            threshold = standardized['specialist_prediction'].median()
            standardized['target_label'] = (standardized['specialist_prediction'] > threshold).astype(int)
            logger.info(f"Created synthetic target for {specialist_name} with threshold {threshold:.4f}")
            
        return standardized
    
    @staticmethod
    def create_standard_metadata(specialist_name: str, config: Dict[str, Any], 
                              metrics: Dict[str, Any], mi_score: float = 0.0,
                              hsic_score: float = 0.0) -> Dict[str, Any]:
        """
        Create standardized metadata for specialist artifacts.
        
        Args:
            specialist_name: Name of the specialist
            config: Configuration dictionary
            metrics: Performance metrics
            mi_score: Mutual Information score
            hsic_score: HSIC score
            
        Returns:
            Standardized metadata dictionary
        """
        metadata = {
            'specialist_name': specialist_name,
            'symbol': config.get('symbol', 'ETHUSDT'),
            'exchange': config.get('exchange', 'binance'),
            'timeframe': config.get('timeframe', '15m'),
            'direction': config.get('direction', 'long'),
            'model_type': config.get('model_type', specialist_name),
            'target_range': config.get('target_range', '1.5-3%'),
            'n_samples': metrics.get('n_samples', 0),
            'mi_score': mi_score,
            'hsic_score': hsic_score,
            'binary_output': metrics.get('binary_output', True),
            'orthogonal_features': metrics.get('orthogonal_features', 0),
            'high_correlation_pairs': metrics.get('high_correlation_pairs', 0),
            'requirements_met': metrics.get('requirements_met', 0),
            'timestamp': datetime.utcnow().isoformat(),
            **metrics  # Include all original metrics
        }
        
        return metadata
    
    @staticmethod
    def validate_standard_structure(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that DataFrame follows standard structure.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        required_columns = ['specialist_prediction', 'specialist_probability', 'target_label']
        
        for col in required_columns:
            if col not in df.columns:
                issues.append(f"Missing required column: {col}")
        
        # Check for binary output
        if 'specialist_prediction' in df.columns:
            unique_vals = df['specialist_prediction'].nunique()
            if unique_vals > 10:
                issues.append(f"Prediction column not binary: {unique_vals} unique values")
        
        # Check for target validity
        if 'target_label' in df.columns:
            unique_targets = df['target_label'].nunique()
            if unique_targets > 2:
                issues.append(f"Target column not binary: {unique_targets} unique values")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def convert_to_binary_output(df: pd.DataFrame, threshold_method: str = 'median') -> pd.DataFrame:
        """
        Convert predictions to binary 0/1 output.
        
        Args:
            df: Input DataFrame
            threshold_method: Method for threshold determination ('median', 'mean', 'quantile_75')
            
        Returns:
            DataFrame with binary predictions
        """
        result = df.copy()
        
        if 'specialist_prediction' in result.columns:
            pred_col = result['specialist_prediction']
            
            if threshold_method == 'median':
                threshold = pred_col.median()
            elif threshold_method == 'mean':
                threshold = pred_col.mean()
            elif threshold_method == 'quantile_75':
                threshold = pred_col.quantile(0.75)
            else:
                threshold = 0.5
            
            # Convert to binary
            result['specialist_prediction_binary'] = (pred_col > threshold).astype(int)
            result['binary_threshold_used'] = threshold
            
            logger.info(f"Converted to binary using {threshold_method} threshold: {threshold:.4f}")
        
        return result


class SpecialistEnsembleInterface:
    """Interface for preparing specialists for ensemble construction."""
    
    @staticmethod
    def prepare_for_ensemble(specialist_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Prepare multiple specialist outputs for ensemble construction.
        
        Args:
            specialist_data: Dictionary of specialist_name -> DataFrame
            
        Returns:
            Combined DataFrame ready for ensemble
        """
        ensemble_data = None
        
        for specialist_name, df in specialist_data.items():
            # Standardize each specialist
            std_df = SpecialistDataInterface.standardize_prediction_data(df, specialist_name)
            
            # Convert to binary
            binary_df = SpecialistDataInterface.convert_to_binary_output(std_df)
            
            # Rename columns to include specialist name
            binary_df = binary_df.rename(columns={
                'specialist_prediction_binary': f'{specialist_name}_prediction',
                'specialist_probability': f'{specialist_name}_probability',
                'prediction_confidence': f'{specialist_name}_confidence'
            })
            
            # Keep only relevant columns for ensemble
            ensemble_cols = ['timestamp', f'{specialist_name}_prediction', 
                           f'{specialist_name}_probability']
            if f'{specialist_name}_confidence' in binary_df.columns:
                ensemble_cols.append(f'{specialist_name}_confidence')
                
            specialist_ensemble = binary_df[ensemble_cols]
            
            if ensemble_data is None:
                ensemble_data = specialist_ensemble
            else:
                # Merge on timestamp
                ensemble_data = pd.merge(ensemble_data, specialist_ensemble, 
                                       on='timestamp', how='outer')
        
        return ensemble_data
    
    @staticmethod
    def analyze_ensemble_diversity(ensemble_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze diversity and orthogonality of specialists for ensemble.
        
        Args:
            ensemble_data: Combined specialist data
            
        Returns:
            Diversity analysis results
        """
        prediction_cols = [col for col in ensemble_data.columns if '_prediction' in col]
        
        diversity_analysis = {
            'specialist_count': len(prediction_cols),
            'pairwise_correlations': {},
            'high_correlation_pairs': [],
            'orthogonal_pairs': [],
            'average_correlation': 0.0
        }
        
        correlations = []
        
        for i, col1 in enumerate(prediction_cols):
            for j, col2 in enumerate(prediction_cols[i+1:], i+1):
                # Calculate correlation
                valid_mask = ~(ensemble_data[col1].isna() | ensemble_data[col2].isna())
                if valid_mask.sum() > 100:
                    corr = np.corrcoef(ensemble_data[col1][valid_mask], 
                                     ensemble_data[col2][valid_mask])[0, 1]
                    
                    diversity_analysis['pairwise_correlations'][f"{col1}_vs_{col2}"] = corr
                    correlations.append(corr)
                    
                    if corr > 0.5:
                        diversity_analysis['high_correlation_pairs'].append((col1, col2, corr))
                    elif corr < 0.1:
                        diversity_analysis['orthogonal_pairs'].append((col1, col2, corr))
        
        if correlations:
            diversity_analysis['average_correlation'] = np.mean(correlations)
        
        return diversity_analysis
