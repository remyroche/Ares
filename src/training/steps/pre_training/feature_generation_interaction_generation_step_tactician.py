"""
Feature Generation Interaction Generation Step - Tactician

This step generates interaction features specifically for the Tactician model.
Tactician focuses on entry timing and short-term patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
from itertools import combinations

from src.training.steps.base_step import BaseStep


class FeatureGenerationInteractionGenerationStepTactician(BaseStep):
    """
    Generates interaction features for Tactician model.
    
    Tactician-specific interactions:
    - Microstructure patterns
    - Short-term momentum shifts
    - Entry signal combinations
    - Volatility-adjusted timing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Tactician interaction generation step.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(
            step_name="feature_generation_interaction_generation_step_tactician",
            config=config
        )
        self.logger = logging.getLogger(__name__)
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Tactician interaction generation.
        
        Args:
            config: Configuration containing:
                - max_interactions: Maximum number of interactions to generate
                - interaction_types: Types of interactions to create
                - model: Should be 'Tactician'
        
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
            self.logger.info("🔄 Starting Tactician interaction generation")
            
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
                ['multiply', 'divide', 'difference']
            )
            
            # Identify feature groups for targeted interactions
            momentum_features = [col for col in X.columns if 'momentum' in col or 'returns' in col]
            volatility_features = [col for col in X.columns if 'volatility' in col or 'std' in col]
            rsi_features = [col for col in X.columns if 'rsi' in col]
            
            # Generate Tactician-specific interactions
            generated_count = 0
            
            # 1. Short-term momentum divergence (for entry timing)
            if 'difference' in interaction_types and len(momentum_features) >= 2:
                # Compare short vs medium term momentum
                short_term_feats = [f for f in momentum_features if any(str(p) in f for p in ['5', '10'])]
                medium_term_feats = [f for f in momentum_features if any(str(p) in f for p in ['20', '30'])]
                
                for short_feat in short_term_feats[:3]:
                    for medium_feat in medium_term_feats[:3]:
                        if generated_count >= max_interactions:
                            break
                        
                        interaction_name = f'tactician_momentum_divergence_{short_feat}_{medium_feat}'
                        interaction_df[interaction_name] = X[short_feat] - X[medium_feat]
                        interaction_list.append(interaction_name)
                        generated_count += 1
            
            # 2. Volatility-adjusted momentum (timing precision)
            if 'divide' in interaction_types:
                for mom_feat in momentum_features[:5]:
                    for vol_feat in volatility_features[:3]:
                        if generated_count >= max_interactions:
                            break
                        
                        interaction_name = f'tactician_vol_adjusted_{mom_feat}_{vol_feat}'
                        interaction_df[interaction_name] = (
                            X[mom_feat] / (X[vol_feat] + 1e-10)
                        )
                        interaction_list.append(interaction_name)
                        generated_count += 1
            
            # 3. RSI combinations (overbought/oversold patterns)
            if 'multiply' in interaction_types and len(rsi_features) >= 2:
                feature_pairs = list(combinations(rsi_features[:4], 2))
                for feat1, feat2 in feature_pairs[:min(8, max_interactions - generated_count)]:
                    interaction_name = f'tactician_rsi_combo_{feat1}_{feat2}'
                    interaction_df[interaction_name] = (
                        (X[feat1] - 50) * (X[feat2] - 50)  # Centered RSI interaction
                    )
                    interaction_list.append(interaction_name)
                    generated_count += 1
            
            # 4. Feature acceleration (rate of change of indicators)
            if 'difference' in interaction_types:
                for feat in momentum_features[:min(10, max_interactions - generated_count)]:
                    interaction_name = f'tactician_acceleration_{feat}'
                    interaction_df[interaction_name] = X[feat].diff()
                    interaction_list.append(interaction_name)
                    generated_count += 1
            
            # 5. Cross-feature products (signal combinations)
            if 'multiply' in interaction_types:
                feature_pairs = list(combinations(momentum_features[:6], 2))
                for feat1, feat2 in feature_pairs[:min(10, max_interactions - generated_count)]:
                    interaction_name = f'tactician_signal_{feat1}_{feat2}'
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
                'tactician_interaction_features',
                metadata=interaction_results
            )
            
            # Save interaction list
            list_path = self._save_metadata(
                {'interactions': interaction_list},
                'tactician_interaction_list'
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
            self.logger.error(f"Tactician interaction generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'artifacts': [],
                'metrics': {}
            }
