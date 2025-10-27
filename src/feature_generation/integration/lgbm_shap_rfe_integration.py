"""
LGBM-SHAP RFE Integration for Enhanced Models Training

This module integrates the LGBM-SHAP RFE selector with the enhanced models training
integration to provide a complete feature selection pipeline for regime models training.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime

# Import the LGBM-SHAP RFE selector
from ..feature_selection.lgbm_shap_rfe_selector import (
    LGBMSHAPRFESelector, 
    LGBMSHAPRFEConfig,
    create_lgbm_shap_rfe_selector
)

# Import enhanced models training integration
from .enhanced_models_training_integration import EnhancedModelsTrainingIntegration

# Import project utilities
from src.utils.tprint import tprint, tprint_success, tprint_warning, tprint_performance, tprint_debug

logger = logging.getLogger(__name__)

class LGBMSHAPRFEIntegration:
    """
    Integration class that combines enhanced models training with LGBM-SHAP RFE selection.
    
    This class provides a complete pipeline for:
    1. Generating comprehensive features using the enhanced models training integration
    2. Selecting optimal features using LGBM-SHAP RFE
    3. Generating detailed reports with global and per-feature metrics
    """
    
    def __init__(self, 
                 target_features: int = 60,
                 removal_percentage: float = 0.25,
                 lgbm_params: Optional[Dict[str, Any]] = None,
                 enable_detailed_logging: bool = True):
        """
        Initialize the LGBM-SHAP RFE integration.
        
        Args:
            target_features: Target number of features to select
            removal_percentage: Percentage of features to remove per iteration (0.25 = 25%)
            lgbm_params: Custom LGBM parameters
            enable_detailed_logging: Enable detailed logging with tprint
        """
        self.target_features = target_features
        self.removal_percentage = removal_percentage
        self.enable_detailed_logging = enable_detailed_logging
        
        # Initialize enhanced models training integration
        self.enhanced_integration = EnhancedModelsTrainingIntegration()
        
        # Configure LGBM-SHAP RFE selector
        rfe_config = LGBMSHAPRFEConfig(
            target_features=target_features,
            removal_percentage=removal_percentage,
            enable_detailed_logging=enable_detailed_logging
        )
        
        # Override LGBM parameters if provided
        if lgbm_params:
            rfe_config.lgb_params.update(lgbm_params)
        
        # Initialize selector
        self.rfe_selector = create_lgbm_shap_rfe_selector(rfe_config)
        
        tprint_success("🔧 LGBM-SHAP RFE Integration initialized")
    
    def select_features_for_regime_training(self, 
                                          data: pd.DataFrame,
                                          target_column: Optional[str] = None,
                                          custom_feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Select optimal features for regime models training using LGBM-SHAP RFE.
        
        Args:
            data: Input data with OHLCV columns
            target_column: Target column name (if None, will create synthetic target)
            custom_feature_names: Custom feature names (if None, will use generated names)
            
        Returns:
            Dictionary containing selection results and detailed report
        """
        tprint("🚀 Starting LGBM-SHAP RFE feature selection for regime training")
        
        # Step 1: Generate comprehensive features
        tprint("📊 Step 1: Generating comprehensive features")
        feature_result = self.enhanced_integration.get_comprehensive_training_features(data)
        
        if not feature_result['success']:
            raise ValueError(f"Feature generation failed: {feature_result.get('error', 'Unknown error')}")
        
        features_df = feature_result['features']
        feature_names = feature_result['feature_names']
        
        tprint(f"✅ Generated {len(feature_names)} features")
        
        # Step 2: Prepare target variable
        tprint("🎯 Step 2: Preparing target variable")
        if target_column is None:
            # Create synthetic target (future returns)
            target = self._create_synthetic_target(data)
            tprint("📈 Created synthetic target (future returns)")
        else:
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            target = data[target_column].values
            tprint(f"📈 Using provided target column: {target_column}")
        
        # Step 3: Align features and target
        tprint("🔗 Step 3: Aligning features and target")
        aligned_data = self._align_features_and_target(features_df, target, data)
        
        if aligned_data is None:
            raise ValueError("Failed to align features and target")
        
        X_aligned, y_aligned, feature_names_aligned = aligned_data
        
        tprint(f"✅ Aligned data: {X_aligned.shape[0]} samples, {X_aligned.shape[1]} features")
        
        # Step 4: Run LGBM-SHAP RFE selection
        tprint("🔍 Step 4: Running LGBM-SHAP RFE selection")
        selection_result = self.rfe_selector.select_features(
            X_aligned, 
            y_aligned, 
            feature_names_aligned,
            target_features=self.target_features
        )
        
        if not selection_result['success']:
            raise ValueError(f"Feature selection failed: {selection_result.get('error', 'Unknown error')}")
        
        # Step 5: Prepare final results
        tprint("📋 Step 5: Preparing final results")
        final_result = self._prepare_final_results(
            selection_result, 
            feature_result, 
            data,
            target_column
        )
        
        tprint_success("🎉 LGBM-SHAP RFE feature selection completed successfully!")
        
        return final_result
    
    def _create_synthetic_target(self, data: pd.DataFrame, 
                               future_periods: int = 1) -> np.ndarray:
        """Create synthetic target variable (future returns)."""
        if 'close' not in data.columns:
            raise ValueError("'close' column not found in data")
        
        close_prices = data['close'].values
        future_returns = np.zeros_like(close_prices)
        
        # Calculate future returns
        for i in range(len(close_prices) - future_periods):
            current_price = close_prices[i]
            future_price = close_prices[i + future_periods]
            future_returns[i] = (future_price - current_price) / current_price
        
        # Set last few values to NaN (no future data)
        future_returns[-future_periods:] = np.nan
        
        return future_returns
    
    def _align_features_and_target(self, 
                                 features_df: pd.DataFrame,
                                 target: np.ndarray,
                                 original_data: pd.DataFrame) -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
        """Align features and target, handling NaN values."""
        tprint_debug("🔗 Aligning features and target")
        
        # Convert features to numpy array
        X = features_df.values
        feature_names = features_df.columns.tolist()
        
        # Find valid indices (no NaN in features or target)
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(target))
        
        if not np.any(valid_mask):
            tprint_warning("⚠️ No valid samples after alignment")
            return None
        
        # Filter valid samples
        X_valid = X[valid_mask]
        y_valid = target[valid_mask]
        
        tprint(f"📊 Valid samples after alignment: {len(y_valid)}")
        
        return X_valid, y_valid, feature_names
    
    def _prepare_final_results(self, 
                             selection_result: Dict[str, Any],
                             feature_result: Dict[str, Any],
                             original_data: pd.DataFrame,
                             target_column: Optional[str]) -> Dict[str, Any]:
        """Prepare final results with comprehensive information."""
        tprint_debug("📋 Preparing final results")
        
        # Extract key information
        selected_indices = selection_result['selected_features']
        selected_feature_names = selection_result['selected_feature_names']
        removed_features = selection_result['removed_features']
        selection_history = selection_result['selection_history']
        detailed_report = selection_result['report']
        
        # Create feature importance summary
        importance_summary = self._create_importance_summary(selection_history)
        
        # Create performance summary
        performance_summary = self._create_performance_summary(selection_history)
        
        # Prepare final results
        final_result = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'target_features': self.target_features,
            'removal_percentage': self.removal_percentage,
            'selected_features': {
                'indices': selected_indices,
                'names': selected_feature_names,
                'count': len(selected_feature_names)
            },
            'removed_features': {
                'names': removed_features,
                'count': len(removed_features)
            },
            'selection_process': {
                'total_iterations': len(selection_history),
                'history': selection_history,
                'importance_summary': importance_summary,
                'performance_summary': performance_summary
            },
            'detailed_report': detailed_report,
            'original_data_info': {
                'shape': original_data.shape,
                'columns': original_data.columns.tolist(),
                'target_column': target_column
            },
            'feature_generation_info': {
                'total_features_generated': len(feature_result['feature_names']),
                'feature_categories': feature_result.get('feature_categories', {}),
                'generation_metadata': feature_result.get('metadata', {})
            }
        }
        
        return final_result
    
    def _create_importance_summary(self, selection_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create importance summary from selection history."""
        if not selection_history:
            return {}
        
        # Extract importance scores from all iterations
        all_importance_scores = []
        for iteration in selection_history:
            if 'importance_scores' in iteration:
                all_importance_scores.extend(iteration['importance_scores'])
        
        if not all_importance_scores:
            return {}
        
        importance_array = np.array(all_importance_scores)
        
        return {
            'mean_importance': float(np.mean(importance_array)),
            'std_importance': float(np.std(importance_array)),
            'min_importance': float(np.min(importance_array)),
            'max_importance': float(np.max(importance_array)),
            'median_importance': float(np.median(importance_array))
        }
    
    def _create_performance_summary(self, selection_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create performance summary from selection history."""
        if not selection_history:
            return {}
        
        # Extract performance scores
        performances = [iteration['performance'] for iteration in selection_history if 'performance' in iteration]
        
        if not performances:
            return {}
        
        performance_array = np.array(performances)
        
        return {
            'mean_performance': float(np.mean(performance_array)),
            'std_performance': float(np.std(performance_array)),
            'min_performance': float(np.min(performance_array)),
            'max_performance': float(np.max(performance_array)),
            'final_performance': float(performances[-1]) if performances else None,
            'performance_trend': 'improving' if len(performances) > 1 and performances[-1] > performances[0] else 'stable'
        }


def create_lgbm_shap_rfe_integration(target_features: int = 60,
                                    removal_percentage: float = 0.25,
                                    lgbm_params: Optional[Dict[str, Any]] = None,
                                    enable_detailed_logging: bool = True) -> LGBMSHAPRFEIntegration:
    """Create and return a LGBM-SHAP RFE integration instance."""
    return LGBMSHAPRFEIntegration(
        target_features=target_features,
        removal_percentage=removal_percentage,
        lgbm_params=lgbm_params,
        enable_detailed_logging=enable_detailed_logging
    )


# Example usage
if __name__ == "__main__":
    # Create sample OHLCV data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic OHLCV data
    base_price = 100
    returns = np.random.normal(0, 0.02, n_samples)
    prices = base_price * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_samples)
    })
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    # Create integration
    integration = create_lgbm_shap_rfe_integration(
        target_features=60,
        removal_percentage=0.25,
        enable_detailed_logging=True
    )
    
    # Run feature selection
    result = integration.select_features_for_regime_training(data)
    
    print(f"Selected {result['selected_features']['count']} features")
    print(f"Removed {result['removed_features']['count']} features")
    print(f"Total iterations: {result['selection_process']['total_iterations']}")