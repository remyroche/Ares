"""
Enhanced Specialist Diagnostics Mixin v2 - MI Improvement & Standardization

This enhanced mixin provides comprehensive diagnostic capabilities with focus on:
- Mutual Information (MI) improvement monitoring
- Data structure standardization
- Feature orthogonality enforcement
- Binary output standardization
- Ensemble compatibility
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
import logging

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
import re

from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.versioned_artifacts import VersionedArtifactStore
from .specialist_interface import SpecialistDataInterface, SpecialistEnsembleInterface
from .enhanced_feature_generators import EnhancedFeaturePipeline
from .specialist_data_standard import (
    SpecialistRequirements, SpecialistMetrics, SpecialistArtifact,
    SpecialistDataValidator, SpecialistStandardFactory, SpecialistType
)
from src.training.steps.labeling.specialist_feature_diagnostics import SpecialistFeatureDiagnostics


class SpecialistDiagnosticsMixinEnhancedV2:
    """
    Enhanced mixin providing comprehensive diagnostic capabilities for specialist models.
    
    Key enhancements:
    - MI improvement monitoring and optimization
    - Standardized data structure enforcement
    - Enhanced feature generation integration
    - Ensemble compatibility validation
    - Comprehensive compliance reporting
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize mixin components while being tolerant to kwargs intended for BaseStep.
        """
        # Some enhanced specialists pass BaseStep kwargs (e.g., use_versioned_artifacts)
        # even though the next class in the MRO may not accept them. Strip unsupported
        # kwargs here to avoid TypeError while preserving recognized parameters.
        passthrough_kwargs = dict(kwargs)  # make a shallow copy
        passthrough_kwargs.pop("use_versioned_artifacts", None)

        # Call parent __init__ if available to preserve step_name/context
        if args or any(key in passthrough_kwargs for key in ['step_name']):
            super().__init__(*args, **passthrough_kwargs)
        
        self.requirements = SpecialistRequirements()
        self.validator = SpecialistDataValidator(self.requirements)
        self.factory = SpecialistStandardFactory(self.requirements)
        self.feature_pipeline = EnhancedFeaturePipeline()
        self.feature_diagnostics = SpecialistFeatureDiagnostics()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _create_standardized_output(self, features: pd.DataFrame, labels: pd.Series,
                                  predictions: np.ndarray, probabilities: np.ndarray,
                                  symbol: str, exchange: str, timeframe: str, direction: str) -> pd.DataFrame:
        """Create standardized output structure."""
        output_df = pd.DataFrame(index=features.index)
        output_df['timestamp'] = features.index
        output_df['specialist_prediction'] = predictions
        output_df['specialist_probability'] = probabilities
        output_df['target_label'] = labels
        # Save top 20 features for diagnostics
        for col in features.columns[:20]:
            output_df[f'feature_{col}'] = features[col]
        return output_df

    def save_specialist_results(self,
                              config: Dict[str, Any],
                              feature_df: pd.DataFrame,
                              labels: pd.Series,
                              predictions: np.ndarray,
                              probabilities: np.ndarray,
                              model: Any,
                              metrics: Dict[str, Any],
                              specialist_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Centralized method to save specialist results, artifacts, and run diagnostics.
        """
        if specialist_name is None:
            specialist_name = self.__class__.__name__

        symbol = config.get('symbol', 'UNKNOWN')
        exchange = config.get('exchange', 'UNKNOWN')
        timeframe = config.get('timeframe', 'UNKNOWN')
        direction = config.get('direction', 'long')

        # 1. Standardized Output
        output_df = self._create_standardized_output(
            feature_df, labels, predictions, probabilities,
            symbol, exchange, timeframe, direction
        )

        # 2. Metadata
        metadata = SpecialistDataInterface.create_standard_metadata(
            specialist_name=specialist_name,
            config=config,
            metrics=metrics,
            mi_score=metrics.get('mi_score', 0.0),
            hsic_score=metrics.get('hsic_score', 0.0)
        )

        # 3. Artifact Naming
        # Convert CamelCase to snake_case for artifact name
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', specialist_name)
        snake_case_name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        if snake_case_name.endswith('_step'):
            snake_case_name = snake_case_name[:-5]

        artifact_name = f"{snake_case_name}_prediction_{timeframe}"

        # 4. Save Data Artifact
        self._save_artifact(data=output_df, artifact_name=artifact_name,
                          artifact_type="data", data_category="predictions", metadata=metadata)

        # 5. Versioned Store
        try:
            if hasattr(self, 'versioned_store') and self.versioned_store:
                self.versioned_store.add_data(output_df, version_name=artifact_name)
                tprint_success(f"💾 Saved predictions to versioned store as '{artifact_name}'")
        except Exception as ve:
            tprint_warning(f"Versioned store save failed: {ve}")

        # 6. Save Model Artifact
        model_artifact_name = f"{snake_case_name}_model_{timeframe}"
        self._save_artifact(data=model, artifact_name=model_artifact_name,
                          artifact_type="model", data_category="models", metadata=metadata)

        # 7. Diagnostics
        diagnostics_result = self.run_enhanced_diagnostics(symbol, exchange, timeframe, direction)

        if diagnostics_result.get('success', False):
             if 'compliance_report' in diagnostics_result:
                metrics.update({
                    'enhanced_mi_score': diagnostics_result['compliance_report']['metrics'].get('mi_score', 0.0),
                    'enhanced_requirements_met': diagnostics_result['compliance_report'].get('requirements_met', False),
                })
             if 'ensemble_compatibility' in diagnostics_result:
                 metrics.update({
                     'ensemble_ready': diagnostics_result['ensemble_compatibility'].get('ensemble_ready', False),
                 })

        return {
            "success": True,
            "metrics": metrics,
            "n_samples": len(output_df),
            "artifact_name": artifact_name,
            "diagnostics": diagnostics_result
        }

    def _load_self_artifacts_enhanced(self, symbol: str, exchange: str, timeframe: str, direction: str) -> Dict[str, Any]:
        """Load this specialist's own artifacts with enhanced validation."""
        try:
            # Ensure context-aware versioned store is initialized
            store = self.versioned_store
            if store is None:
                return {'error': 'Versioned store unavailable'}
            
            specialist_name = self.__class__.__name__.replace('Step', '').lower()
            
            # Get latest prediction view
            views = store.list_versions()
            pred_views = [v for v in views if 'prediction' in v.lower()]
            
            if not pred_views:
                self.logger.debug(
                    "No prediction-tagged versions found; falling back to latest available view."
                )
                pred_views = views
            
            if not pred_views:
                return {'error': 'No prediction views found'}
            
            latest_view = pred_views[-1]
            artifact_view = store.get_view(latest_view)
            
            # Convert to DataFrame and standardize
            df = artifact_view.to_pandas()
            if not isinstance(df, pd.DataFrame) or len(df) == 0:
                return {'error': 'Invalid artifact data'}
            
            # Standardize data structure
            standardized_df = SpecialistDataInterface.standardize_prediction_data(df, specialist_name)
            
            # Validate structure
            is_valid, issues = self.validator.validate_prediction_data(standardized_df, specialist_name)
            if not is_valid:
                self.logger.warning(f"Data validation issues: {issues}")
            
            return {
                'data': standardized_df,
                'view_name': latest_view,
                'validation_issues': issues,
                'is_valid': is_valid
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load artifacts: {e}")
            return {'error': str(e)}
    
    def _generate_enhanced_features(self, df: pd.DataFrame, specialist_type: SpecialistType) -> pd.DataFrame:
        """Generate enhanced features for MI improvement."""
        try:
            # Determine specialist type from class name
            if 'volume_force' in self.__class__.__name__.lower():
                specialist_type = SpecialistType.VOLUME_FORCE
            elif 'momentum_persistence' in self.__class__.__name__.lower():
                specialist_type = SpecialistType.MOMENTUM_PERSISTENCE
            elif 'smc_regime' in self.__class__.__name__.lower():
                specialist_type = SpecialistType.SMC_REGIME
            elif 'volatility_burst' in self.__class__.__name__.lower():
                specialist_type = SpecialistType.VOLATILITY_BURST
            else:
                specialist_type = SpecialistType.VOLUME_FORCE  # Default
            
            # Generate enhanced features
            enhanced_features = self.feature_pipeline.generate_enhanced_features(
                df, specialist_type.value
            )
            
            return enhanced_features
            
        except Exception as e:
            self.logger.error(f"Enhanced feature generation failed: {e}")
            return pd.DataFrame(index=df.index)
    
    def _compute_mi_improvement_metrics(self, features: pd.DataFrame, labels: pd.Series, 
                                      predictions: np.ndarray) -> Dict[str, Any]:
        """Compute comprehensive MI improvement metrics."""
        metrics = {}
        
        try:
            # 1. Prediction MI to target
            pred_mi = mutual_info_regression(
                predictions.reshape(-1, 1), labels.values
            )[0]
            metrics['prediction_mi_to_target'] = pred_mi
            
            # 2. Feature MI analysis
            if len(features.columns) > 0:
                feature_mi_scores = []
                for col in features.columns:
                    if features[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                        mi_score = mutual_info_regression(
                            features[col].values.reshape(-1, 1), labels.values
                        )[0]
                        feature_mi_scores.append(mi_score)
                
                if feature_mi_scores:
                    metrics['avg_feature_mi'] = np.mean(feature_mi_scores)
                    metrics['max_feature_mi'] = np.max(feature_mi_scores)
                    metrics['mi_improvement_potential'] = metrics['max_feature_mi'] - metrics['avg_feature_mi']
                    
                    # Count high-MI features
                    high_mi_features = sum(1 for mi in feature_mi_scores if mi > self.requirements.min_mi_score)
                    metrics['high_mi_features_count'] = high_mi_features
                    metrics['high_mi_features_ratio'] = high_mi_features / len(feature_mi_scores)
            
            # 3. MI improvement assessment
            metrics['mi_target_met'] = pred_mi >= self.requirements.min_mi_score
            metrics['mi_improvement_needed'] = self.requirements.min_mi_score - pred_mi if pred_mi < self.requirements.min_mi_score else 0
            
        except Exception as e:
            self.logger.error(f"MI computation failed: {e}")
            metrics.update({
                'prediction_mi_to_target': 0.0,
                'avg_feature_mi': 0.0,
                'max_feature_mi': 0.0,
                'mi_target_met': False,
                'mi_improvement_needed': self.requirements.min_mi_score
            })
        
        return metrics
    
    def _enforce_orthogonality_enhanced(self, features: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Enhanced feature orthogonality enforcement with detailed analysis."""
        orthogonality_metrics = {
            'original_features': len(features.columns),
            'dropped_features': 0,
            'high_correlation_pairs': 0,
            'orthogonal_features': 0,
            'correlation_reduction': 0.0
        }
        
        if len(features.columns) < 2:
            return features, orthogonality_metrics
        
        try:
            # Compute correlation matrix
            corr_matrix = features.corr().abs()
            
            # Find high correlations
            high_corr_mask = (corr_matrix > self.requirements.max_correlation_threshold) & (corr_matrix < 1.0)
            high_corr_pairs = np.where(high_corr_mask)
            
            orthogonality_metrics['high_correlation_pairs'] = len(high_corr_pairs[0])
            
            # Calculate average correlation before
            avg_corr_before = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
            
            # Remove highly correlated features
            features_to_drop = set()
            for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
                if i < j:  # Avoid duplicates
                    col1, col2 = corr_matrix.index[i], corr_matrix.index[j]
                    # Keep the feature with higher variance (more information)
                    if features[col1].var() > features[col2].var():
                        features_to_drop.add(col2)
                    else:
                        features_to_drop.add(col1)
            
            # Drop features
            orthogonal_features = features.drop(columns=features_to_drop)
            orthogonality_metrics['dropped_features'] = len(features_to_drop)
            orthogonality_metrics['orthogonal_features'] = len(orthogonal_features.columns)
            
            # Calculate correlation reduction
            if len(orthogonal_features.columns) > 1:
                new_corr_matrix = orthogonal_features.corr().abs()
                avg_corr_after = new_corr_matrix.values[np.triu_indices_from(new_corr_matrix.values, k=1)].mean()
                orthogonality_metrics['correlation_reduction'] = avg_corr_before - avg_corr_after
            
            if features_to_drop:
                self.logger.info(f"Orthogonalization: dropped {len(features_to_drop)} features, "
                               f"correlation reduced by {orthogonality_metrics['correlation_reduction']:.3f}")
            
            return orthogonal_features, orthogonality_metrics
            
        except Exception as e:
            self.logger.error(f"Orthogonality enforcement failed: {e}")
            return features, orthogonality_metrics
    
    def _standardize_binary_output(self, predictions: np.ndarray, probabilities: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Standardize binary output with optimal threshold selection."""
        binary_metrics = {
            'threshold_used': 0.5,
            'binary_conversion_method': 'median',
            'unique_values_before': len(np.unique(predictions)),
            'unique_values_after': 2,
            'binary_compliance': True
        }
        
        try:
            # Determine optimal threshold
            if len(np.unique(probabilities)) > 2:
                # Use median as default threshold for robustness
                threshold = np.median(probabilities)
                binary_metrics['threshold_used'] = threshold
                binary_metrics['binary_conversion_method'] = 'median'
            else:
                threshold = 0.5
                binary_metrics['threshold_used'] = threshold
                binary_metrics['binary_conversion_method'] = 'fixed'
            
            # Convert to binary
            binary_predictions = (probabilities >= threshold).astype(int)
            binary_metrics['unique_values_after'] = len(np.unique(binary_predictions))
            
            # Validate binary output
            binary_metrics['binary_compliance'] = len(np.unique(binary_predictions)) == 2
            
            return binary_predictions, binary_metrics
            
        except Exception as e:
            self.logger.error(f"Binary standardization failed: {e}")
            # Fallback to simple threshold
            binary_predictions = (probabilities >= 0.5).astype(int)
            return binary_predictions, binary_metrics
    
    def _create_standardized_artifact(self, data: pd.DataFrame, metadata: Dict[str, Any], 
                                    metrics: Dict[str, Any]) -> SpecialistArtifact:
        """Create standardized specialist artifact."""
        try:
            # Determine specialist type
            specialist_name = self.__class__.__name__
            if 'volume_force' in specialist_name.lower():
                specialist_type = SpecialistType.VOLUME_FORCE
            elif 'momentum_persistence' in specialist_name.lower():
                specialist_type = SpecialistType.MOMENTUM_PERSISTENCE
            elif 'smc_regime' in specialist_name.lower():
                specialist_type = SpecialistType.SMC_REGIME
            else:
                specialist_type = SpecialistType.VOLUME_FORCE
            
            # Create standardized artifact
            artifact = self.factory.create_standard_artifact(
                specialist_name=specialist_name,
                specialist_type=specialist_type,
                data=data,
                metadata=metadata,
                metrics=metrics
            )
            
            return artifact
            
        except Exception as e:
            self.logger.error(f"Artifact creation failed: {e}")
            raise
    
    def _analyze_ensemble_compatibility(self, artifact: SpecialistArtifact) -> Dict[str, Any]:
        """Analyze ensemble compatibility of the specialist."""
        compatibility_metrics = {
            'ensemble_ready': False,
            'data_structure_compliant': False,
            'mi_compliant': False,
            'orthogonality_compliant': False,
            'binary_output_compliant': False,
            'overall_compliance_score': 0.0
        }
        
        try:
            # Check data structure compliance
            data_structure_compliant = artifact.validate_structure()[0]
            compatibility_metrics['data_structure_compliant'] = data_structure_compliant
            
            # Check MI compliance
            mi_compliant = artifact.metrics.mi_score >= self.requirements.min_mi_score
            compatibility_metrics['mi_compliant'] = mi_compliant
            
            # Check orthogonality compliance
            orthogonality_compliant = artifact.metrics.high_correlation_pairs <= self.requirements.max_high_correlation_pairs
            compatibility_metrics['orthogonality_compliant'] = orthogonality_compliant
            
            # Check binary output compliance
            binary_compliant = not self.requirements.binary_output_required or artifact.metrics.binary_output
            compatibility_metrics['binary_output_compliant'] = binary_compliant
            
            # Calculate overall compliance score
            compliance_factors = [
                data_structure_compliant,
                mi_compliant,
                orthogonality_compliant,
                binary_compliant
            ]
            compatibility_metrics['overall_compliance_score'] = sum(compliance_factors) / len(compliance_factors)
            
            # Determine ensemble readiness
            compatibility_metrics['ensemble_ready'] = compatibility_metrics['overall_compliance_score'] >= 0.75
            
            return compatibility_metrics
            
        except Exception as e:
            self.logger.error(f"Ensemble compatibility analysis failed: {e}")
            return compatibility_metrics
    
    def run_enhanced_diagnostics(self, symbol: str, exchange: str, timeframe: str, direction: str) -> Dict[str, Any]:
        """Run comprehensive enhanced diagnostics."""
        # Ensure BaseStep context matches the artifacts we are about to load
        if hasattr(self, "set_context"):
            self.set_context(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                direction=direction,
                model=self.step_name,
            )
        
        self.logger.info(f"🚀 Running enhanced diagnostics for {self.__class__.__name__}")
        
        try:
            # 1. Load artifacts
            artifact_data = self._load_self_artifacts_enhanced(symbol, exchange, timeframe, direction)
            if 'error' in artifact_data:
                return {'success': False, 'error': artifact_data['error']}
            
            df = artifact_data['data']
            
            # 2. Extract existing features from prediction data
            print(f"DEBUG: DataFrame columns: {list(df.columns)[:10]}...")
            feature_cols = [col for col in df.columns if col.startswith('feature_')]
            print(f"DEBUG: Found {len(feature_cols)} feature columns")
            if feature_cols:
                enhanced_features = df[feature_cols]
                print(f"DEBUG: Using existing features with shape: {enhanced_features.shape}")
            else:
                # Fallback: try to generate features if OHLCV data is available
                print("WARNING: No feature columns found, attempting to generate features")
                try:
                    enhanced_features = self._generate_enhanced_features(df, SpecialistType.VOLUME_FORCE)
                except KeyError as e:
                    return {'success': False, 'error': f'Cannot generate features: {e}'}
            
            # 3. Extract labels and predictions
            if 'target_label' in df.columns and 'specialist_prediction' in df.columns:
                labels = df['target_label']
                predictions = df['specialist_prediction']
                probabilities = df['specialist_probability']
            else:
                return {'success': False, 'error': 'Missing required columns'}
            
            # 4. Run comprehensive feature diagnostics
            tprint_info(f"🔬 Running comprehensive feature analysis for {self.__class__.__name__}")
            feature_diagnostics_results = self.feature_diagnostics.comprehensive_feature_analysis(
                features=enhanced_features,
                labels=labels,
                predictions=predictions,
                specialist_name=self.__class__.__name__
            )

            # 5. Compute MI improvement metrics
            mi_metrics = self._compute_mi_improvement_metrics(enhanced_features, labels, predictions)

            # 6. Enforce orthogonality
            orthogonal_features, orthogonality_metrics = self._enforce_orthogonality_enhanced(enhanced_features)

            # 7. Run orthogonalization diagnostics
            if 'error' not in feature_diagnostics_results:
                tprint_info(f"🎯 Running orthogonalization diagnostics for {self.__class__.__name__}")
                orthogonalization_diagnostics = self.feature_diagnostics.advanced_orthogonalization_diagnostics(
                    original_features=enhanced_features,
                    orthogonal_features=orthogonal_features,
                    labels=labels,
                    dropped_features=orthogonality_metrics.get('dropped_features', []),
                    specialist_name=self.__class__.__name__
                )
            else:
                orthogonalization_diagnostics = {'error': 'Feature diagnostics failed'}
            
            # 6. Standardize binary output
            binary_predictions, binary_metrics = self._standardize_binary_output(predictions, probabilities)
            
            # 7. Create comprehensive metrics
            comprehensive_metrics = {
                **mi_metrics,
                **orthogonality_metrics,
                **binary_metrics,
                'requirements_met': sum([
                    mi_metrics['mi_target_met'],
                    orthogonality_metrics['high_correlation_pairs'] <= self.requirements.max_high_correlation_pairs,
                    binary_metrics['binary_compliance']
                ])
            }
            
            # 8. Create standardized artifact
            metadata = {
                'specialist_name': self.__class__.__name__,
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'direction': direction,
                'timestamp': datetime.utcnow().isoformat(),
                'enhanced_diagnostics': True
            }
            
            artifact = self._create_standardized_artifact(df, metadata, comprehensive_metrics)
            
            # 9. Analyze ensemble compatibility
            compatibility = self._analyze_ensemble_compatibility(artifact)
            
            # 10. Generate compliance report
            compliance_report = artifact.get_compliance_summary()
            
            # 11. Generate comprehensive diagnostics report
            comprehensive_diagnostics = {
                'feature_analysis': feature_diagnostics_results,
                'orthogonalization_diagnostics': orthogonalization_diagnostics,
                'legacy_metrics': comprehensive_metrics
            }

            # Save comprehensive diagnostics report
            diagnostics_filename = f"{self.__class__.__name__}_comprehensive_diagnostics"
            self.feature_diagnostics.save_diagnostics_report(
                comprehensive_diagnostics, diagnostics_filename
            )

            return {
                'success': True,
                'artifact': artifact,
                'compliance_report': compliance_report,
                'ensemble_compatibility': compatibility,
                'enhanced_features_count': len(enhanced_features.columns),
                'orthogonal_features_count': len(orthogonal_features.columns),
                'mi_improvement_needed': mi_metrics.get('mi_improvement_needed', 0),
                'comprehensive_diagnostics': comprehensive_diagnostics,
                'feature_quality_score': feature_diagnostics_results.get('overall_quality_score', 0.0),
                'orthogonality_score': orthogonalization_diagnostics.get('orthogonality_score', 0.0) if 'error' not in orthogonalization_diagnostics else 0.0,
                'recommendations': self._generate_comprehensive_recommendations(
                    comprehensive_metrics, compatibility, feature_diagnostics_results, orthogonalization_diagnostics
                )
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced diagnostics failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_diagnostics(self, symbol: str, exchange: str, timeframe: str, direction: str) -> Dict[str, Any]:
        """Compatibility shim so diagnostics scripts can rely on run_diagnostics."""
        return self.run_enhanced_diagnostics(symbol, exchange, timeframe, direction)
    
    def _generate_comprehensive_recommendations(self, metrics: Dict[str, Any], compatibility: Dict[str, Any],
                                              feature_diagnostics: Dict[str, Any],
                                              orthogonalization_diagnostics: Dict[str, Any]) -> List[str]:
        """Generate comprehensive improvement recommendations from all diagnostic sources."""
        recommendations = []

        # 1. Legacy recommendations (MI, orthogonality, binary output)
        recommendations.extend(self._generate_legacy_recommendations(metrics, compatibility))

        # 2. Feature analysis recommendations
        if 'recommendations' in feature_diagnostics and feature_diagnostics['recommendations']:
            recommendations.extend([
                f"📊 Feature Analysis: {rec}" for rec in feature_diagnostics['recommendations']
            ])

        # 3. Orthogonalization diagnostics recommendations
        if 'recommendations' in orthogonalization_diagnostics and orthogonalization_diagnostics['recommendations']:
            recommendations.extend([
                f"🎯 Orthogonalization: {rec}" for rec in orthogonalization_diagnostics['recommendations']
            ])

        # 4. Quality score-based recommendations
        feature_quality = feature_diagnostics.get('overall_quality_score', 0.0)
        orthogonality_score = orthogonalization_diagnostics.get('orthogonality_score', 0.0)

        if feature_quality < 0.7:
            recommendations.append(
                f"🚨 CRITICAL: Feature quality score ({feature_quality:.2f}) is poor. "
                f"Address missing data and weak predictive features immediately."
            )

        if orthogonality_score < 0.8:
            recommendations.append(
                f"⚠️ WARNING: Orthogonality score ({orthogonality_score:.2f}) indicates redundant features. "
                f"Consider stricter orthogonalization thresholds."
            )

        return recommendations

    def _generate_legacy_recommendations(self, metrics: Dict[str, Any], compatibility: Dict[str, Any]) -> List[str]:
        """Generate recommendations from legacy diagnostic metrics."""
        recommendations = []

        # MI improvement recommendations
        if not metrics.get('mi_target_met', False):
            recommendations.append(
                f"🔧 MI Improvement needed: target {self.requirements.min_mi_score}, "
                f"current {metrics.get('prediction_mi_to_target', 0):.4f}. "
                f"Add non-linear features and market regime indicators."
            )

        # Orthogonality recommendations
        if metrics.get('high_correlation_pairs', 0) > self.requirements.max_high_correlation_pairs:
            recommendations.append(
                f"🔄 Orthogonality needed: {metrics.get('high_correlation_pairs', 0)} "
                f"high correlation pairs found. Remove redundant features."
            )

        # Binary output recommendations
        if not metrics.get('binary_compliance', False):
            recommendations.append(
                "🔢 Binary output required: Convert predictions to 0/1 scalar using threshold optimization."
            )

        # Ensemble readiness recommendations
        if not compatibility.get('ensemble_ready', False):
            recommendations.append(
                f"🚀 Ensemble preparation needed: Current compliance score "
                f"{compatibility.get('overall_compliance_score', 0):.2f}. "
                f"Focus on improving {compatibility.get('overall_compliance_score', 0) * 4:.0f}/4 requirements."
            )

        # Feature improvement recommendations
        if metrics.get('enhanced_features_count', 0) < self.requirements.min_features:
            recommendations.append(
                f"➕ Feature expansion needed: Only {metrics.get('enhanced_features_count', 0)} "
                f"features, minimum {self.requirements.min_features} required."
            )

        return recommendations
