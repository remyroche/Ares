"""
Feature Generation Interaction Generation Step - Analyst

This step generates interaction features specifically for the Analyst model.
Analyst focuses on profit prediction and longer-term patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
from itertools import combinations

from src.training.steps.base_step import BaseStep


class FeatureGenerationInteractionGenerationStepAnalyst(BaseStep):
    """
    Generates interaction features for Analyst model.
    
    Analyst-specific interactions:
    - Momentum × Volatility
    - Volume × Price patterns
    - Multi-timeframe relationships
    - Risk-reward indicators
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Analyst interaction generation step.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(
            step_name="feature_generation_interaction_generation_step_analyst",
            config=config
        )
        self.logger = logging.getLogger(__name__)
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Analyst interaction generation.
        
        Args:
            config: Configuration containing:
                - max_interactions: Maximum number of interactions to generate
                - interaction_types: Types of interactions to create
                - model: Should be 'Analyst'
        
        Returns:
            Dictionary containing:
                - success: bool
                - interaction_features_path: str
                - interaction_list: List[str]
                - artifacts: list
                - metrics: dict
        """
        start_time = datetime.now()
        
        try:
            self.logger.info("🔄 Starting Analyst interaction generation")
            
            # Load selected features
            features_df = self._load_dataframe('selected_features')
            if features_df is None:
                return {
                    'success': False,
                    'error': 'No selected features found',
                    'artifacts': [],
                    'metrics': {}
                }
            
            # Get label column
            label_column = config.get('label_column', 'label')
            if label_column not in features_df.columns:
                return {
                    'success': False,
                    'error': f'Label column {label_column} not found',
                    'artifacts': [],
                    'metrics': {}
                }
            
            # Separate features and labels
            y = features_df[label_column]
            X = features_df.drop(columns=[label_column])
            
            # Generate interactions
            interaction_df = X.copy()
            interaction_list = list(X.columns)
            
            max_interactions = config.get('max_interactions', 50)
            interaction_types = config.get(
                'interaction_types',
                ['multiply', 'divide', 'ratio']
            )
            
            # Identify feature groups for targeted interactions
            momentum_features = [col for col in X.columns if 'momentum' in col or 'returns' in col]
            volatility_features = [col for col in X.columns if 'volatility' in col or 'std' in col]
            volume_features = [col for col in X.columns if 'volume' in col]
            
            # Generate Analyst-specific interactions
            generated_count = 0
            
            # 1. Momentum × Volatility (risk-adjusted returns)
            if 'multiply' in interaction_types:
                for mom_feat in momentum_features[:5]:  # Top 5 momentum features
                    for vol_feat in volatility_features[:3]:  # Top 3 volatility features
                        if generated_count >= max_interactions:
                            break
                        
                        interaction_name = f'analyst_risk_adjusted_{mom_feat}_{vol_feat}'
                        interaction_df[interaction_name] = (
                            X[mom_feat] / (X[vol_feat] + 1e-10)
                        )
                        interaction_list.append(interaction_name)
                        generated_count += 1
            
            # 2. Volume-weighted momentum
            if 'multiply' in interaction_types and volume_features:
                for mom_feat in momentum_features[:5]:
                    if generated_count >= max_interactions:
                        break
                    
                    for vol_feat in volume_features[:2]:
                        if generated_count >= max_interactions:
                            break
                        
                        interaction_name = f'analyst_vol_weighted_{mom_feat}_{vol_feat}'
                        interaction_df[interaction_name] = X[mom_feat] * X[vol_feat]
                        interaction_list.append(interaction_name)
                        generated_count += 1
            
            # 3. Feature ratios (relative strength)
            if 'ratio' in interaction_types:
                feature_pairs = list(combinations(momentum_features[:8], 2))
                for feat1, feat2 in feature_pairs[:min(10, max_interactions - generated_count)]:
                    interaction_name = f'analyst_ratio_{feat1}_{feat2}'
                    interaction_df[interaction_name] = (
                        X[feat1] / (X[feat2] + 1e-10)
                    )
                    interaction_list.append(interaction_name)
                    generated_count += 1
            
            # 4. Cross-feature products (momentum combinations)
            if 'multiply' in interaction_types:
                feature_pairs = list(combinations(momentum_features[:6], 2))
                for feat1, feat2 in feature_pairs[:min(10, max_interactions - generated_count)]:
                    interaction_name = f'analyst_product_{feat1}_{feat2}'
                    interaction_df[interaction_name] = X[feat1] * X[feat2]
                    interaction_list.append(interaction_name)
                    generated_count += 1
            
            # Add label back
            interaction_df[label_column] = y
            
            # Remove any NaN or inf values
            interaction_df = interaction_df.replace([np.inf, -np.inf], np.nan)
            initial_rows = len(interaction_df)
            interaction_df = interaction_df.dropna()
            dropped_rows = initial_rows - len(interaction_df)
            
            if dropped_rows > 0:
                self.logger.warning(f"Dropped {dropped_rows} rows with invalid values")
            
            # Interaction generation results
            interaction_results = {
                'base_features': len(X.columns),
                'interactions_generated': generated_count,
                'total_features': len(interaction_list),
                'interaction_types_used': interaction_types,
                'rows_dropped': dropped_rows
            }
            
            # Save interaction features
            interaction_path = self._save_dataframe(
                interaction_df,
                'analyst_interaction_features',
                metadata=interaction_results
            )
            
            # Save interaction list
            list_path = self._save_metadata(
                {'interactions': interaction_list},
                'analyst_interaction_list'
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'success': True,
                'interaction_features_path': interaction_path,
                'interaction_list': interaction_list,
                'artifacts': [interaction_path, list_path],
                'metrics': {
                    'interactions_generated': generated_count,
                    'total_features': len(interaction_list),
                    'execution_time': execution_time
                }
            }
            
        except Exception as e:
            self.logger.error(f"Analyst interaction generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'artifacts': [],
                'metrics': {}
            }
