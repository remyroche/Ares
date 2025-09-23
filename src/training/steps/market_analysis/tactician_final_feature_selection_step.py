"""
Tactician Final Feature Selection Step - With Long/Short Differentiation

This module provides the final feature selection step for Tactician models on 1m timeframe
with long/short differentiation. Uses separate selection for long and short opportunities.

Key features:
- Long/short differentiation (separate selection)
- Optimized for 1m timeframe
- Enhanced feature selection process
- Focus on separate long and short opportunity assessment
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
import asyncio
import json
import time
from dataclasses import dataclass

# Import logger
from src.utils.logger import get_logger
from src.utils.comprehensive_function_logger import log_all_calls
from src.core.decorators import handles_errors, traced, log_execution_time, validates

@dataclass
class TacticianFeatureSelectionConfig:
    """Configuration for Tactician final feature selection."""
    # Feature selection parameters
    initial_features: int = 200  # Higher for 1m data
    stage_1_target: int = 150
    stage_2_target: int = 100
    stage_3_target: int = 80
    final_target: int = 60
    
    # Model parameters
    rf_n_estimators: int = 100
    cv_folds: int = 5
    
    # Quality thresholds
    min_feature_importance: float = 0.01
    min_correlation_threshold: float = 0.1
    max_correlation_threshold: float = 0.9
    
    # Timeframe specific (1m for Tactician)
    timeframe_minutes: int = 1
    
    # Long/short differentiation parameters
    enable_long_short_differentiation: bool = True
    long_short_balance_weight: float = 0.5  # Weight for balancing long/short features
    directional_confidence_threshold: float = 0.1  # Minimum confidence for directional bias
    
    # Output settings
    save_analysis: bool = True
    output_directory: str = "outcomes/market_analysis/tactician"
    verbose: bool = True

@dataclass
class TacticianFeatureSelectionResult:
    """Result of Tactician final feature selection."""
    # Selected features (long/short differentiated)
    final_features: List[str]
    long_final_features: List[str]
    short_final_features: List[str]
    
    # Stage results
    stage_1_features: List[str]
    stage_2_features: List[str]
    stage_3_features: List[str]
    
    # Feature importance scores
    final_importance_scores: Dict[str, float]
    long_importance_scores: Dict[str, float]
    short_importance_scores: Dict[str, float]
    
    # Quality metrics
    overall_quality_score: float
    long_quality_score: float
    short_quality_score: float
    long_short_balance_score: float
    
    # Performance metrics
    selection_time: float
    total_features_processed: int
    final_feature_count: int
    long_feature_count: int
    short_feature_count: int
    
    # Status
    success: bool
    error_message: Optional[str] = None

class TacticianFinalFeatureSelector:
    """
    Tactician Final Feature Selector - WITH LONG/SHORT DIFFERENTIATION.
    
    Performs final feature selection for Tactician models on 1m timeframe
    with long/short differentiation.
    """
    
    def __init__(self, config: Optional[TacticianFeatureSelectionConfig] = None):
        """Initialize the Tactician final feature selector."""
        self.config = config or TacticianFeatureSelectionConfig()
        self.logger = get_logger("TacticianFinalFeatureSelector")
        
        # Initialize output directory
        self.output_dir = Path(self.config.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("🚀 Tactician Final Feature Selector initialized (WITH LONG/SHORT DIFFERENTIATION)")
        self.logger.info(f"   → Initial features: {self.config.initial_features}")
        self.logger.info(f"   → Final target: {self.config.final_target}")
        self.logger.info(f"   → Timeframe: {self.config.timeframe_minutes}m")
        self.logger.info(f"   → Long/Short differentiation: {'Enabled' if self.config.enable_long_short_differentiation else 'Disabled'}")
        self.logger.info(f"   → Output directory: {self.output_dir}")
    
    async def select_features(self, 
                            feature_data: pd.DataFrame,
                            target_data: Optional[pd.Series] = None,
                            long_target_data: Optional[pd.Series] = None,
                            short_target_data: Optional[pd.Series] = None) -> TacticianFeatureSelectionResult:
        """
        Perform final feature selection for Tactician (LONG/SHORT DIFFERENTIATED).
        
        Args:
            feature_data: Feature data for selection
            target_data: Combined target variable (optional)
            long_target_data: Long-specific target variable (optional)
            short_target_data: Short-specific target variable (optional)
            
        Returns:
            TacticianFeatureSelectionResult with selected features
        """
        start_time = time.time()
        self.logger.info("🔍 Starting Tactician final feature selection (LONG/SHORT DIFFERENTIATED)")
        
        try:
            # Step 1: Validate input data
            validation_result = await self._validate_input_data(feature_data, target_data, long_target_data, short_target_data)
            if not validation_result['is_valid']:
                return TacticianFeatureSelectionResult(
                    final_features=[],
                    long_final_features=[],
                    short_final_features=[],
                    stage_1_features=[],
                    stage_2_features=[],
                    stage_3_features=[],
                    final_importance_scores={},
                    long_importance_scores={},
                    short_importance_scores={},
                    overall_quality_score=0.0,
                    long_quality_score=0.0,
                    short_quality_score=0.0,
                    long_short_balance_score=0.0,
                    selection_time=time.time() - start_time,
                    total_features_processed=0,
                    final_feature_count=0,
                    long_feature_count=0,
                    short_feature_count=0,
                    success=False,
                    error_message=validation_result['error_message']
                )
            
            # Step 2: Prepare data for selection
            prepared_data = await self._prepare_data_for_selection(feature_data, target_data, long_target_data, short_target_data)
            
            # Step 3: Stage 1 - Initial feature filtering
            stage_1_result = await self._stage_1_selection(prepared_data)
            
            # Step 4: Stage 2 - Correlation-based selection
            stage_2_result = await self._stage_2_selection(stage_1_result, prepared_data)
            
            # Step 5: Stage 3 - Importance-based selection
            stage_3_result = await self._stage_3_selection(stage_2_result, prepared_data)
            
            # Step 6: Long/Short differentiated final selection
            final_result = await self._final_selection_differentiated(stage_3_result, prepared_data)
            
            # Step 7: Calculate quality metrics
            quality_metrics = await self._calculate_quality_metrics(final_result, prepared_data)
            
            # Step 8: Save results
            if self.config.save_analysis:
                await self._save_selection_results(final_result, quality_metrics)
            
            # Step 9: Create final result
            result = TacticianFeatureSelectionResult(
                final_features=final_result['features'],
                long_final_features=final_result['long_features'],
                short_final_features=final_result['short_features'],
                stage_1_features=stage_1_result['features'],
                stage_2_features=stage_2_result['features'],
                stage_3_features=stage_3_result['features'],
                final_importance_scores=final_result['importance_scores'],
                long_importance_scores=final_result['long_importance_scores'],
                short_importance_scores=final_result['short_importance_scores'],
                overall_quality_score=quality_metrics['overall'],
                long_quality_score=quality_metrics['long'],
                short_quality_score=quality_metrics['short'],
                long_short_balance_score=quality_metrics['long_short_balance'],
                selection_time=time.time() - start_time,
                total_features_processed=len(feature_data.columns),
                final_feature_count=len(final_result['features']),
                long_feature_count=len(final_result['long_features']),
                short_feature_count=len(final_result['short_features']),
                success=True
            )
            
            self.logger.info(f"✅ Tactician feature selection completed: {result.final_feature_count} features selected")
            self.logger.info(f"   → Long features: {result.long_feature_count}")
            self.logger.info(f"   → Short features: {result.short_feature_count}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Tactician feature selection failed: {e}")
            return TacticianFeatureSelectionResult(
                final_features=[],
                long_final_features=[],
                short_final_features=[],
                stage_1_features=[],
                stage_2_features=[],
                stage_3_features=[],
                final_importance_scores={},
                long_importance_scores={},
                short_importance_scores={},
                overall_quality_score=0.0,
                long_quality_score=0.0,
                short_quality_score=0.0,
                long_short_balance_score=0.0,
                selection_time=time.time() - start_time,
                total_features_processed=0,
                final_feature_count=0,
                long_feature_count=0,
                short_feature_count=0,
                success=False,
                error_message=str(e)
            )
    
    async def _validate_input_data(self, feature_data: pd.DataFrame, 
                                 target_data: Optional[pd.Series],
                                 long_target_data: Optional[pd.Series],
                                 short_target_data: Optional[pd.Series]) -> Dict[str, Any]:
        """Validate input data for feature selection."""
        try:
            # Check feature data
            if feature_data is None or feature_data.empty:
                return {'is_valid': False, 'error_message': 'Feature data is empty or None'}
            
            if len(feature_data.columns) < self.config.final_target:
                return {'is_valid': False, 'error_message': f'Insufficient features: {len(feature_data.columns)} columns'}
            
            if len(feature_data) < 100:
                return {'is_valid': False, 'error_message': f'Insufficient samples: {len(feature_data)} rows'}
            
            # Check target data
            has_target = target_data is not None
            has_long_target = long_target_data is not None
            has_short_target = short_target_data is not None
            
            if not (has_target or (has_long_target and has_short_target)):
                return {'is_valid': False, 'error_message': 'No target data provided'}
            
            # Validate target data lengths
            if target_data is not None and len(target_data) != len(feature_data):
                return {'is_valid': False, 'error_message': 'Target data length mismatch'}
            
            if long_target_data is not None and len(long_target_data) != len(feature_data):
                return {'is_valid': False, 'error_message': 'Long target data length mismatch'}
            
            if short_target_data is not None and len(short_target_data) != len(feature_data):
                return {'is_valid': False, 'error_message': 'Short target data length mismatch'}
            
            # Check for NaN values in targets
            if target_data is not None and target_data.isna().all():
                return {'is_valid': False, 'error_message': 'Target data contains only NaN values'}
            
            if long_target_data is not None and long_target_data.isna().all():
                return {'is_valid': False, 'error_message': 'Long target data contains only NaN values'}
            
            if short_target_data is not None and short_target_data.isna().all():
                return {'is_valid': False, 'error_message': 'Short target data contains only NaN values'}
            
            return {'is_valid': True, 'error_message': None}
            
        except Exception as e:
            return {'is_valid': False, 'error_message': f'Validation error: {e}'}
    
    async def _prepare_data_for_selection(self, feature_data: pd.DataFrame, 
                                         target_data: Optional[pd.Series],
                                         long_target_data: Optional[pd.Series],
                                         short_target_data: Optional[pd.Series]) -> Dict[str, Any]:
        """Prepare data for feature selection."""
        try:
            # Clean and prepare feature data
            prepared_features = feature_data.copy()
            
            # Handle missing values
            prepared_features = prepared_features.fillna(prepared_features.median())
            
            # Remove infinite values
            prepared_features = prepared_features.replace([np.inf, -np.inf], np.nan)
            prepared_features = prepared_features.fillna(prepared_features.median())
            
            # Remove constant columns
            constant_columns = prepared_features.columns[prepared_features.nunique() <= 1]
            if len(constant_columns) > 0:
                prepared_features = prepared_features.drop(columns=constant_columns)
                self.logger.info(f"Removed {len(constant_columns)} constant columns")
            
            # Prepare target data
            prepared_target = None
            if target_data is not None:
                prepared_target = target_data.copy()
                prepared_target = prepared_target.fillna(prepared_target.median())
            
            prepared_long_target = None
            if long_target_data is not None:
                prepared_long_target = long_target_data.copy()
                prepared_long_target = prepared_long_target.fillna(prepared_long_target.median())
            
            prepared_short_target = None
            if short_target_data is not None:
                prepared_short_target = short_target_data.copy()
                prepared_short_target = prepared_short_target.fillna(prepared_short_target.median())
            
            return {
                'features': prepared_features,
                'target': prepared_target,
                'long_target': prepared_long_target,
                'short_target': prepared_short_target,
                'original_feature_count': len(feature_data.columns),
                'cleaned_feature_count': len(prepared_features.columns)
            }
            
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            raise
    
    async def _stage_1_selection(self, prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 1: Initial feature filtering based on variance and basic quality."""
        try:
            features = prepared_data['features']
            target = prepared_data['target']
            long_target = prepared_data['long_target']
            short_target = prepared_data['short_target']
            
            # Calculate variance for each feature
            variances = features.var()
            
            # Calculate correlation with targets if available
            correlations = {}
            long_correlations = {}
            short_correlations = {}
            
            if target is not None:
                for col in features.columns:
                    try:
                        corr = features[col].corr(target)
                        correlations[col] = abs(corr) if not np.isnan(corr) else 0.0
                    except:
                        correlations[col] = 0.0
            
            if long_target is not None:
                for col in features.columns:
                    try:
                        corr = features[col].corr(long_target)
                        long_correlations[col] = abs(corr) if not np.isnan(corr) else 0.0
                    except:
                        long_correlations[col] = 0.0
            
            if short_target is not None:
                for col in features.columns:
                    try:
                        corr = features[col].corr(short_target)
                        short_correlations[col] = abs(corr) if not np.isnan(corr) else 0.0
                    except:
                        short_correlations[col] = 0.0
            
            # Select features based on variance and correlation
            selected_features = []
            scores = {}
            
            for col in features.columns:
                variance_score = variances[col] if not np.isnan(variances[col]) else 0.0
                correlation_score = correlations.get(col, 0.0)
                
                # Combined score (variance + correlation if target available)
                if target is not None:
                    combined_score = variance_score * 0.5 + correlation_score * 0.5
                else:
                    combined_score = variance_score
                
                if combined_score > self.config.min_feature_importance:
                    selected_features.append(col)
                    scores[col] = combined_score
            
            # Sort by score and limit to target
            if len(selected_features) > self.config.stage_1_target:
                sorted_features = sorted(selected_features, key=lambda x: scores[x], reverse=True)
                selected_features = sorted_features[:self.config.stage_1_target]
            
            self.logger.info(f"Stage 1: Selected {len(selected_features)} features from {len(features.columns)}")
            
            return {
                'features': selected_features,
                'scores': {col: scores[col] for col in selected_features}
            }
            
        except Exception as e:
            self.logger.error(f"Stage 1 selection failed: {e}")
            return {'features': [], 'scores': {}}
    
    async def _stage_2_selection(self, stage_1_result: Dict[str, Any], 
                                prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 2: Correlation-based selection to remove redundant features."""
        try:
            stage_1_features = stage_1_result['features']
            if len(stage_1_features) == 0:
                return {'features': [], 'scores': {}}
            
            features = prepared_data['features'][stage_1_features]
            
            # Calculate correlation matrix
            correlation_matrix = features.corr().abs()
            
            # Remove highly correlated features
            selected_features = []
            scores = {}
            
            for i, col1 in enumerate(stage_1_features):
                is_redundant = False
                
                for j, col2 in enumerate(stage_1_features):
                    if i != j and col2 in selected_features:
                        corr = correlation_matrix.loc[col1, col2]
                        if corr > self.config.max_correlation_threshold:
                            is_redundant = True
                            break
                
                if not is_redundant:
                    selected_features.append(col1)
                    scores[col1] = stage_1_result['scores'].get(col1, 0.0)
            
            # If we removed too many features, add back some with lower correlation
            if len(selected_features) < self.config.stage_2_target:
                remaining_features = [f for f in stage_1_features if f not in selected_features]
                remaining_features.sort(key=lambda x: stage_1_result['scores'].get(x, 0.0), reverse=True)
                
                for feature in remaining_features:
                    if len(selected_features) >= self.config.stage_2_target:
                        break
                    
                    # Check if adding this feature would create too much redundancy
                    max_corr = 0.0
                    for selected_feature in selected_features:
                        corr = correlation_matrix.loc[feature, selected_feature]
                        max_corr = max(max_corr, corr)
                    
                    if max_corr < self.config.max_correlation_threshold:
                        selected_features.append(feature)
                        scores[feature] = stage_1_result['scores'].get(feature, 0.0)
            
            self.logger.info(f"Stage 2: Selected {len(selected_features)} features from {len(stage_1_features)}")
            
            return {
                'features': selected_features,
                'scores': scores
            }
            
        except Exception as e:
            self.logger.error(f"Stage 2 selection failed: {e}")
            return {'features': stage_1_result['features'], 'scores': stage_1_result['scores']}
    
    async def _stage_3_selection(self, stage_2_result: Dict[str, Any], 
                                prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 3: Importance-based selection using Random Forest."""
        try:
            stage_2_features = stage_2_result['features']
            if len(stage_2_features) == 0:
                return {'features': [], 'scores': {}}
            
            features = prepared_data['features'][stage_2_features]
            target = prepared_data['target']
            long_target = prepared_data['long_target']
            short_target = prepared_data['short_target']
            
            # Use Random Forest for feature importance if target is available
            if target is not None:
                try:
                    from sklearn.ensemble import RandomForestRegressor
                    
                    # Prepare data
                    X = features.fillna(0)
                    y = target.fillna(0)
                    
                    # Train Random Forest
                    rf = RandomForestRegressor(
                        n_estimators=self.config.rf_n_estimators,
                        random_state=42,
                        n_jobs=-1
                    )
                    rf.fit(X, y)
                    
                    # Get feature importance
                    importance_scores = dict(zip(stage_2_features, rf.feature_importances_))
                    
                except Exception as e:
                    self.logger.warning(f"Random Forest failed: {e}, using correlation-based selection")
                    importance_scores = {}
                    for col in stage_2_features:
                        try:
                            corr = features[col].corr(target)
                            importance_scores[col] = abs(corr) if not np.isnan(corr) else 0.0
                        except:
                            importance_scores[col] = 0.0
            else:
                # Use variance as importance if no target
                importance_scores = {}
                for col in stage_2_features:
                    variance = features[col].var()
                    importance_scores[col] = variance if not np.isnan(variance) else 0.0
            
            # Select features based on importance
            selected_features = []
            scores = {}
            
            for feature in stage_2_features:
                importance = importance_scores.get(feature, 0.0)
                if importance > self.config.min_feature_importance:
                    selected_features.append(feature)
                    scores[feature] = importance
            
            # Sort by importance and limit to target
            if len(selected_features) > self.config.stage_3_target:
                sorted_features = sorted(selected_features, key=lambda x: scores[x], reverse=True)
                selected_features = sorted_features[:self.config.stage_3_target]
            
            self.logger.info(f"Stage 3: Selected {len(selected_features)} features from {len(stage_2_features)}")
            
            return {
                'features': selected_features,
                'scores': {col: scores[col] for col in selected_features}
            }
            
        except Exception as e:
            self.logger.error(f"Stage 3 selection failed: {e}")
            return {'features': stage_2_result['features'], 'scores': stage_2_result['scores']}
    
    async def _final_selection_differentiated(self, stage_3_result: Dict[str, Any], 
                                            prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Final selection using long/short differentiated approach."""
        try:
            stage_3_features = stage_3_result['features']
            if len(stage_3_features) == 0:
                return {
                    'features': [],
                    'long_features': [],
                    'short_features': [],
                    'importance_scores': {},
                    'long_importance_scores': {},
                    'short_importance_scores': {}
                }
            
            features = prepared_data['features'][stage_3_features]
            target = prepared_data['target']
            long_target = prepared_data['long_target']
            short_target = prepared_data['short_target']
            
            # Final feature selection based on cross-validation performance
            if target is not None:
                try:
                    from sklearn.ensemble import RandomForestRegressor
                    from sklearn.model_selection import cross_val_score
                    
                    # Prepare data
                    X = features.fillna(0)
                    y = target.fillna(0)
                    
                    # Test different feature subsets
                    best_features = []
                    best_score = -np.inf
                    
                    # Start with top features and add more if performance improves
                    sorted_features = sorted(stage_3_features, 
                                           key=lambda x: stage_3_result['scores'].get(x, 0.0), 
                                           reverse=True)
                    
                    for i in range(1, min(len(sorted_features) + 1, self.config.final_target + 5)):
                        current_features = sorted_features[:i]
                        
                        try:
                            # Train model with current features
                            rf = RandomForestRegressor(
                                n_estimators=self.config.rf_n_estimators,
                                random_state=42,
                                n_jobs=-1
                            )
                            
                            # Cross-validation
                            cv_scores = cross_val_score(
                                rf, X[current_features], y, 
                                cv=self.config.cv_folds, 
                                scoring='neg_mean_squared_error'
                            )
                            
                            mean_score = cv_scores.mean()
                            
                            if mean_score > best_score:
                                best_score = mean_score
                                best_features = current_features.copy()
                            
                        except Exception as e:
                            self.logger.warning(f"CV failed for {len(current_features)} features: {e}")
                            continue
                    
                    # Limit to final target
                    if len(best_features) > self.config.final_target:
                        best_features = best_features[:self.config.final_target]
                    
                    # Get final importance scores
                    if best_features:
                        rf = RandomForestRegressor(
                            n_estimators=self.config.rf_n_estimators,
                            random_state=42,
                            n_jobs=-1
                        )
                        rf.fit(X[best_features], y)
                        importance_scores = dict(zip(best_features, rf.feature_importances_))
                    else:
                        importance_scores = {}
                    
                except Exception as e:
                    self.logger.warning(f"Final CV selection failed: {e}, using importance-based selection")
                    # Fallback to importance-based selection
                    sorted_features = sorted(stage_3_features, 
                                           key=lambda x: stage_3_result['scores'].get(x, 0.0), 
                                           reverse=True)
                    best_features = sorted_features[:self.config.final_target]
                    importance_scores = {col: stage_3_result['scores'].get(col, 0.0) for col in best_features}
            else:
                # No target - use importance-based selection
                sorted_features = sorted(stage_3_features, 
                                       key=lambda x: stage_3_result['scores'].get(x, 0.0), 
                                       reverse=True)
                best_features = sorted_features[:self.config.final_target]
                importance_scores = {col: stage_3_result['scores'].get(col, 0.0) for col in best_features}
            
            # Long/short differentiated selection
            long_features = []
            short_features = []
            long_importance_scores = {}
            short_importance_scores = {}
            
            if long_target is not None and short_target is not None:
                # Separate selection for long and short features
                for feature in best_features:
                    try:
                        long_corr = features[feature].corr(long_target)
                        short_corr = features[feature].corr(short_target)
                        
                        long_corr = abs(long_corr) if not np.isnan(long_corr) else 0.0
                        short_corr = abs(short_corr) if not np.isnan(short_corr) else 0.0
                        
                        if long_corr > short_corr:
                            long_features.append(feature)
                            long_importance_scores[feature] = long_corr
                        else:
                            short_features.append(feature)
                            short_importance_scores[feature] = short_corr
                            
                    except Exception as e:
                        self.logger.warning(f"Error in long/short selection for {feature}: {e}")
                        # Default to long if error
                        long_features.append(feature)
                        long_importance_scores[feature] = 0.0
            else:
                # No long/short targets - use combined features
                long_features = best_features
                short_features = best_features
                long_importance_scores = importance_scores
                short_importance_scores = importance_scores
            
            self.logger.info(f"Final: Selected {len(best_features)} features from {len(stage_3_features)}")
            self.logger.info(f"   → Long features: {len(long_features)}")
            self.logger.info(f"   → Short features: {len(short_features)}")
            
            return {
                'features': best_features,
                'long_features': long_features,
                'short_features': short_features,
                'importance_scores': importance_scores,
                'long_importance_scores': long_importance_scores,
                'short_importance_scores': short_importance_scores
            }
            
        except Exception as e:
            self.logger.error(f"Final selection failed: {e}")
            return {
                'features': stage_3_result['features'],
                'long_features': stage_3_result['features'],
                'short_features': stage_3_result['features'],
                'importance_scores': stage_3_result['scores'],
                'long_importance_scores': stage_3_result['scores'],
                'short_importance_scores': stage_3_result['scores']
            }
    
    async def _calculate_quality_metrics(self, final_result: Dict[str, Any], 
                                        prepared_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate quality metrics for selected features."""
        try:
            final_features = final_result['features']
            long_features = final_result['long_features']
            short_features = final_result['short_features']
            
            if len(final_features) == 0:
                return {
                    'overall': 0.0,
                    'long': 0.0,
                    'short': 0.0,
                    'long_short_balance': 0.0
                }
            
            features = prepared_data['features'][final_features]
            
            # Calculate overall quality
            overall_quality = len(final_features) / max(1, len(prepared_data['features'].columns))
            
            # Calculate long quality
            long_quality = len(long_features) / max(1, len(final_features))
            
            # Calculate short quality
            short_quality = len(short_features) / max(1, len(final_features))
            
            # Calculate long/short balance
            long_short_balance = 1.0 - abs(len(long_features) - len(short_features)) / max(len(long_features) + len(short_features), 1)
            
            return {
                'overall': overall_quality,
                'long': long_quality,
                'short': short_quality,
                'long_short_balance': long_short_balance
            }
            
        except Exception as e:
            self.logger.warning(f"Quality metrics calculation failed: {e}")
            return {
                'overall': 0.5,
                'long': 0.5,
                'short': 0.5,
                'long_short_balance': 0.5
            }
    
    async def _save_selection_results(self, final_result: Dict[str, Any], 
                                    quality_metrics: Dict[str, float]) -> None:
        """Save feature selection results."""
        try:
            # Save final features
            final_features_file = self.output_dir / "tactician_final_features.json"
            with open(final_features_file, 'w') as f:
                json.dump({
                    'final_features': final_result['features'],
                    'long_features': final_result['long_features'],
                    'short_features': final_result['short_features'],
                    'feature_count': len(final_result['features']),
                    'long_feature_count': len(final_result['long_features']),
                    'short_feature_count': len(final_result['short_features']),
                    'importance_scores': final_result['importance_scores'],
                    'long_importance_scores': final_result['long_importance_scores'],
                    'short_importance_scores': final_result['short_importance_scores'],
                    'quality_metrics': quality_metrics,
                    'selection_timestamp': datetime.now().isoformat()
                }, f, indent=2)
            
            self.logger.info(f"💾 Final features saved to: {final_features_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save selection results: {e}")

# Convenience functions
def create_tactician_final_feature_selector(config: Optional[TacticianFeatureSelectionConfig] = None) -> TacticianFinalFeatureSelector:
    """Create Tactician final feature selector."""
    return TacticianFinalFeatureSelector(config)

async def select_tactician_final_features(feature_data: pd.DataFrame,
                                        target_data: Optional[pd.Series] = None,
                                        long_target_data: Optional[pd.Series] = None,
                                        short_target_data: Optional[pd.Series] = None,
                                        config: Optional[TacticianFeatureSelectionConfig] = None) -> TacticianFeatureSelectionResult:
    """Select Tactician final features."""
    selector = TacticianFinalFeatureSelector(config)
    return await selector.select_features(feature_data, target_data, long_target_data, short_target_data)