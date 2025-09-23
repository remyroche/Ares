"""
Analyst Final Feature Selection Step - No Long/Short Differentiation

This module provides the final feature selection step for Analyst models on 5m timeframe
without long/short differentiation. Uses unified approach for overall opportunity assessment.

Key features:
- No long/short differentiation (unified approach)
- Optimized for 5m timeframe
- Simplified feature selection process
- Focus on overall opportunity assessment
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
import asyncio
import json

# Import logger
from src.utils.logger import get_logger
from src.utils.comprehensive_function_logger import log_all_calls
from src.core.decorators import handles_errors, traced, log_execution_time, validates

@dataclass
class AnalystFeatureSelectionConfig:
    """Configuration for Analyst final feature selection."""
    # Feature selection parameters
    initial_features: int = 100
    stage_1_target: int = 80
    stage_2_target: int = 60
    stage_3_target: int = 40
    final_target: int = 30
    
    # Model parameters
    rf_n_estimators: int = 100
    cv_folds: int = 5
    
    # Quality thresholds
    min_feature_importance: float = 0.01
    min_correlation_threshold: float = 0.1
    max_correlation_threshold: float = 0.9
    
    # Timeframe specific (5m for Analyst)
    timeframe_minutes: int = 5
    
    # Output settings
    save_analysis: bool = True
    output_directory: str = "outcomes/market_analysis/analyst"
    verbose: bool = True

@dataclass
class AnalystFeatureSelectionResult:
    """Result of Analyst final feature selection."""
    # Selected features
    final_features: List[str]
    stage_1_features: List[str]
    stage_2_features: List[str]
    stage_3_features: List[str]
    
    # Feature importance scores
    final_importance_scores: Dict[str, float]
    stage_1_scores: Dict[str, float]
    stage_2_scores: Dict[str, float]
    stage_3_scores: Dict[str, float]
    
    # Quality metrics
    overall_quality_score: float
    feature_diversity_score: float
    redundancy_score: float
    stability_score: float
    
    # Performance metrics
    selection_time: float
    total_features_processed: int
    final_feature_count: int
    
    # Status
    success: bool
    error_message: Optional[str] = None

class AnalystFinalFeatureSelector:
    """
    Analyst Final Feature Selector - NO LONG/SHORT DIFFERENTIATION.
    
    Performs final feature selection for Analyst models on 5m timeframe
    without long/short differentiation.
    """
    
    def __init__(self, config: Optional[AnalystFeatureSelectionConfig] = None):
        """Initialize the Analyst final feature selector."""
        self.config = config or AnalystFeatureSelectionConfig()
        self.logger = get_logger("AnalystFinalFeatureSelector")
        
        # Initialize output directory
        self.output_dir = Path(self.config.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("🚀 Analyst Final Feature Selector initialized (NO LONG/SHORT DIFFERENTIATION)")
        self.logger.info(f"   → Initial features: {self.config.initial_features}")
        self.logger.info(f"   → Final target: {self.config.final_target}")
        self.logger.info(f"   → Timeframe: {self.config.timeframe_minutes}m")
        self.logger.info(f"   → Output directory: {self.output_dir}")
    
    async def select_features(self, 
                            feature_data: pd.DataFrame,
                            target_data: Optional[pd.Series] = None) -> AnalystFeatureSelectionResult:
        """
        Perform final feature selection for Analyst (UNIFIED APPROACH).
        
        Args:
            feature_data: Feature data for selection
            target_data: Target variable (optional, will use unsupervised selection if not provided)
            
        Returns:
            AnalystFeatureSelectionResult with selected features
        """
        start_time = time.time()
        self.logger.info("🔍 Starting Analyst final feature selection (UNIFIED APPROACH)")
        
        try:
            # Step 1: Validate input data
            validation_result = await self._validate_input_data(feature_data, target_data)
            if not validation_result['is_valid']:
                return AnalystFeatureSelectionResult(
                    final_features=[],
                    stage_1_features=[],
                    stage_2_features=[],
                    stage_3_features=[],
                    final_importance_scores={},
                    stage_1_scores={},
                    stage_2_scores={},
                    stage_3_scores={},
                    overall_quality_score=0.0,
                    feature_diversity_score=0.0,
                    redundancy_score=1.0,
                    stability_score=0.0,
                    selection_time=time.time() - start_time,
                    total_features_processed=0,
                    final_feature_count=0,
                    success=False,
                    error_message=validation_result['error_message']
                )
            
            # Step 2: Prepare data for selection
            prepared_data = await self._prepare_data_for_selection(feature_data, target_data)
            
            # Step 3: Stage 1 - Initial feature filtering
            stage_1_result = await self._stage_1_selection(prepared_data)
            
            # Step 4: Stage 2 - Correlation-based selection
            stage_2_result = await self._stage_2_selection(stage_1_result, prepared_data)
            
            # Step 5: Stage 3 - Importance-based selection
            stage_3_result = await self._stage_3_selection(stage_2_result, prepared_data)
            
            # Step 6: Final selection
            final_result = await self._final_selection(stage_3_result, prepared_data)
            
            # Step 7: Calculate quality metrics
            quality_metrics = await self._calculate_quality_metrics(final_result, prepared_data)
            
            # Step 8: Save results
            if self.config.save_analysis:
                await self._save_selection_results(final_result, quality_metrics)
            
            # Step 9: Create final result
            result = AnalystFeatureSelectionResult(
                final_features=final_result['features'],
                stage_1_features=stage_1_result['features'],
                stage_2_features=stage_2_result['features'],
                stage_3_features=stage_3_result['features'],
                final_importance_scores=final_result['importance_scores'],
                stage_1_scores=stage_1_result['scores'],
                stage_2_scores=stage_2_result['scores'],
                stage_3_scores=stage_3_result['scores'],
                overall_quality_score=quality_metrics['overall'],
                feature_diversity_score=quality_metrics['diversity'],
                redundancy_score=quality_metrics['redundancy'],
                stability_score=quality_metrics['stability'],
                selection_time=time.time() - start_time,
                total_features_processed=len(feature_data.columns),
                final_feature_count=len(final_result['features']),
                success=True
            )
            
            self.logger.info(f"✅ Analyst feature selection completed: {result.final_feature_count} features selected")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Analyst feature selection failed: {e}")
            return AnalystFeatureSelectionResult(
                final_features=[],
                stage_1_features=[],
                stage_2_features=[],
                stage_3_features=[],
                final_importance_scores={},
                stage_1_scores={},
                stage_2_scores={},
                stage_3_scores={},
                overall_quality_score=0.0,
                feature_diversity_score=0.0,
                redundancy_score=1.0,
                stability_score=0.0,
                selection_time=time.time() - start_time,
                total_features_processed=0,
                final_feature_count=0,
                success=False,
                error_message=str(e)
            )
    
    async def _validate_input_data(self, feature_data: pd.DataFrame, 
                                 target_data: Optional[pd.Series]) -> Dict[str, Any]:
        """Validate input data for feature selection."""
        try:
            # Check feature data
            if feature_data is None or feature_data.empty:
                return {'is_valid': False, 'error_message': 'Feature data is empty or None'}
            
            if len(feature_data.columns) < self.config.final_target:
                return {'is_valid': False, 'error_message': f'Insufficient features: {len(feature_data.columns)} columns'}
            
            if len(feature_data) < 100:
                return {'is_valid': False, 'error_message': f'Insufficient samples: {len(feature_data)} rows'}
            
            # Check target data if provided
            if target_data is not None:
                if len(target_data) != len(feature_data):
                    return {'is_valid': False, 'error_message': 'Target data length mismatch'}
                
                if target_data.isna().all():
                    return {'is_valid': False, 'error_message': 'Target data contains only NaN values'}
            
            return {'is_valid': True, 'error_message': None}
            
        except Exception as e:
            return {'is_valid': False, 'error_message': f'Validation error: {e}'}
    
    async def _prepare_data_for_selection(self, feature_data: pd.DataFrame, 
                                        target_data: Optional[pd.Series]) -> Dict[str, Any]:
        """Prepare data for feature selection."""
        try:
            # Clean feature data
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
            
            return {
                'features': prepared_features,
                'target': prepared_target,
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
            
            # Calculate variance for each feature
            variances = features.var()
            
            # Calculate correlation with target if available
            correlations = {}
            if target is not None:
                for col in features.columns:
                    try:
                        corr = features[col].corr(target)
                        correlations[col] = abs(corr) if not np.isnan(corr) else 0.0
                    except:
                        correlations[col] = 0.0
            
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
    
    async def _final_selection(self, stage_3_result: Dict[str, Any], 
                              prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Final selection using cross-validation."""
        try:
            stage_3_features = stage_3_result['features']
            if len(stage_3_features) == 0:
                return {'features': [], 'importance_scores': {}}
            
            features = prepared_data['features'][stage_3_features]
            target = prepared_data['target']
            
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
            
            self.logger.info(f"Final: Selected {len(best_features)} features from {len(stage_3_features)}")
            
            return {
                'features': best_features,
                'importance_scores': importance_scores
            }
            
        except Exception as e:
            self.logger.error(f"Final selection failed: {e}")
            return {'features': stage_3_result['features'], 'importance_scores': stage_3_result['scores']}
    
    async def _calculate_quality_metrics(self, final_result: Dict[str, Any], 
                                       prepared_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate quality metrics for selected features."""
        try:
            final_features = final_result['features']
            if len(final_features) == 0:
                return {
                    'overall': 0.0,
                    'diversity': 0.0,
                    'redundancy': 1.0,
                    'stability': 0.0
                }
            
            features = prepared_data['features'][final_features]
            
            # Calculate diversity score
            diversity_score = len(final_features) / max(1, len(prepared_data['features'].columns))
            
            # Calculate redundancy score
            correlation_matrix = features.corr().abs()
            redundancy_score = correlation_matrix.mean().mean()
            
            # Calculate stability score
            stability_scores = []
            for col in final_features:
                col_values = features[col].dropna()
                if len(col_values) > 1:
                    stability = 1.0 - (col_values.std() / (col_values.mean() + 1e-8))
                    stability_scores.append(max(0.0, stability))
            
            stability_score = np.mean(stability_scores) if stability_scores else 0.0
            
            # Calculate overall quality score
            overall_score = (diversity_score + (1.0 - redundancy_score) + stability_score) / 3
            
            return {
                'overall': overall_score,
                'diversity': diversity_score,
                'redundancy': redundancy_score,
                'stability': stability_score
            }
            
        except Exception as e:
            self.logger.warning(f"Quality metrics calculation failed: {e}")
            return {
                'overall': 0.5,
                'diversity': 0.5,
                'redundancy': 0.5,
                'stability': 0.5
            }
    
    async def _save_selection_results(self, final_result: Dict[str, Any], 
                                    quality_metrics: Dict[str, float]) -> None:
        """Save feature selection results."""
        try:
            # Save final features
            final_features_file = self.output_dir / "analyst_final_features.json"
            with open(final_features_file, 'w') as f:
                json.dump({
                    'final_features': final_result['features'],
                    'feature_count': len(final_result['features']),
                    'importance_scores': final_result['importance_scores'],
                    'quality_metrics': quality_metrics,
                    'selection_timestamp': datetime.now().isoformat()
                }, f, indent=2)
            
            self.logger.info(f"💾 Final features saved to: {final_features_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save selection results: {e}")

# Convenience functions
def create_analyst_final_feature_selector(config: Optional[AnalystFeatureSelectionConfig] = None) -> AnalystFinalFeatureSelector:
    """Create Analyst final feature selector."""
    return AnalystFinalFeatureSelector(config)

async def select_analyst_final_features(feature_data: pd.DataFrame,
                                      target_data: Optional[pd.Series] = None,
                                      config: Optional[AnalystFeatureSelectionConfig] = None) -> AnalystFeatureSelectionResult:
    """Select Analyst final features."""
    selector = AnalystFinalFeatureSelector(config)
    return await selector.select_features(feature_data, target_data)