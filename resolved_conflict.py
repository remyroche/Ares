"""
RESOLVED CONFLICT CONTENT

Replace the entire conflict section with this resolved content:

def get_overfitting_detector(config: Optional[OverfittingConfig] = None) -> UniversalOverfittingDetector:
    \"\"\"Get overfitting detector instance.\"\"\"
    if config is None:
        return DEFAULT_OVERFITTING_DETECTOR
    return UniversalOverfittingDetector(config)

def detect_overfitting_with_learning_curves(model: Any,
                                          X_train: np.ndarray,
                                          X_val: np.ndarray,
                                          y_train: np.ndarray,
                                          y_val: np.ndarray,
                                          X_test: Optional[np.ndarray] = None,
                                          y_test: Optional[np.ndarray] = None,
                                          model_name: str = "unknown",
                                          model_type: str = "unknown",
                                          fold_number: Optional[int] = None,
                                          config: Optional[OverfittingConfig] = None) -> OverfittingReport:
    \"\"\"
    Enhanced overfitting detection with learning curve analysis.
    
    This function integrates learning curve analysis into the existing overfitting detection
    to provide more comprehensive overfitting assessment.
    \"\"\"
    try:
        # Import learning curve analyzer
        from ..evaluation.enhanced_learning_curve_analysis import EnhancedLearningCurveAnalyzer
        
        # Initialize detector
        detector = get_overfitting_detector(config)
        
        # Get basic predictions
        train_predictions = model.predict(X_train)
        val_predictions = model.predict(X_val)
        
        # Get probabilities if available
        train_probabilities = None
        val_probabilities = None
        if hasattr(model, 'predict_proba'):
            try:
                train_probabilities = model.predict_proba(X_train)
                val_probabilities = model.predict_proba(X_val)
            except Exception as e:
                logger.warning(f"Could not get probabilities: {e}")
        
        # Get feature importance if available
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            feature_importance = np.abs(model.coef_).flatten()
        
        # Perform basic overfitting detection
        basic_report = detector.detect_overfitting(
            train_predictions=train_predictions,
            val_predictions=val_predictions,
            train_labels=y_train,
            val_labels=y_val,
            train_probabilities=train_probabilities,
            val_probabilities=val_probabilities,
            feature_importance=feature_importance,
            model_name=model_name,
            model_type=model_type,
            fold_number=fold_number
        )
        
        # Perform learning curve analysis
        try:
            learning_curve_analyzer = EnhancedLearningCurveAnalyzer()
            
            # Combine train and val for learning curve analysis
            X_combined = np.vstack([X_train, X_val])
            y_combined = np.concatenate([y_train, y_val])
            
            # Determine scoring metric
            is_classification = len(np.unique(y_combined)) <= 10
            scoring = 'accuracy' if is_classification else 'r2'
            
            # Perform learning curve analysis
            learning_curve_result = learning_curve_analyzer.analyze_learning_curve(
                model=model,
                X_train=X_combined,
                y_train=y_combined,
                X_test=X_test if X_test is not None else X_val,
                y_test=y_test if y_test is not None else y_val,
                scoring=scoring
            )
            
            # Add learning curve indicators to the report
            if learning_curve_result.overfitting_risk in ["high", "severe"]:
                basic_report.indicators.append("learning_curve_overfitting")
                basic_report.warnings.append("Learning curve analysis indicates overfitting risk")
            
            if learning_curve_result.convergence_stability == "poor":
                basic_report.recommendations.append("Poor convergence stability - consider learning rate adjustment")
            
            if learning_curve_result.training_efficiency == "low":
                basic_report.recommendations.append("Low training efficiency - consider model simplification")
            
            # Update severity based on learning curve analysis
            if learning_curve_result.overfitting_risk == "severe":
                if basic_report.severity == "moderate":
                    basic_report.severity = "high"
                elif basic_report.severity == "none":
                    basic_report.severity = "moderate"
            
            logger.info(f"✅ Learning curve analysis integrated for {model_name}")
            
        except Exception as e:
            logger.warning(f"Learning curve analysis failed: {e}")
            basic_report.warnings.append("Learning curve analysis unavailable")
        
        return basic_report
        
    except Exception as e:
        logger.error(f"Enhanced overfitting detection with learning curves failed: {e}")
        # Fallback to basic detection
        detector = get_overfitting_detector(config)
        return detector.detect_overfitting(
            train_predictions=model.predict(X_train),
            val_predictions=model.predict(X_val),
            train_labels=y_train,
            val_labels=y_val,
            model_name=model_name,
            model_type=model_type,
            fold_number=fold_number
        )


class ModelEnhancementDetector:
    \"\"\"Detect models that could benefit from parameter tuning and optimization.\"\"\"

    def __init__(self):
        \"\"\"Initialize model enhancement detector.\"\"\"
        self.logger = logging.getLogger('ModelEnhancementDetector')

    def detect_enhancement_opportunities(self,
                                       model,
                                       X_train: np.ndarray,
                                       X_val: np.ndarray,
                                       y_train: np.ndarray,
                                       y_val: np.ndarray,
                                       model_name: str = "unknown",
                                       model_type: str = "unknown") -> Dict[str, Any]:
        \"\"\"
        Detect opportunities for model enhancement and parameter tuning.

        Args:
            model: Trained ML model
            X_train: Training features
            X_val: Validation features
            y_train: Training labels
            y_val: Validation labels
            model_name: Name of the model
            model_type: Type of model

        Returns:
            Dict: Enhancement opportunities and recommendations
        \"\"\"
        opportunities = {
            'model_name': model_name,
            'model_type': model_type,
            'enhancement_opportunities': [],
            'parameter_tuning_suggestions': [],
            'performance_issues': [],
            'data_issues': [],
            'confidence_level': 0.0,
            'priority': 'low',  # low, medium, high, critical
            'estimated_improvement_potential': 0.0
        }

        try:
            # 1. Check if model is underfitting (too simple)
            underfitting_score = self._check_underfitting(model, X_train, X_val, y_train, y_val)
            if underfitting_score > 0.7:
                opportunities['enhancement_opportunities'].append('model_complexity_increase')
                opportunities['parameter_tuning_suggestions'].append({
                    'warning': 'Model appears to be underfitting - may be too simple for the data',
                    'reason': 'High training and validation errors suggest insufficient model capacity',
                    'improvement_potential': 'Investigate increasing model complexity'
                })

            # 2. Check for parameter sensitivity
            sensitivity_analysis = self._analyze_parameter_sensitivity(model, X_train, y_train)
            if sensitivity_analysis['high_sensitivity']:
                opportunities['enhancement_opportunities'].append('parameter_tuning_needed')
                opportunities['parameter_tuning_suggestions'].append({
                    'warning': 'Model shows parameter sensitivity - room for improvement through tuning',
                    'reason': f'{model_type} models typically benefit from parameter optimization',
                    'improvement_potential': 'Consider hyperparameter optimization for better performance'
                })

            # 3. Check for feature importance imbalance
            importance_analysis = self._analyze_feature_importance(model, X_train)
            if importance_analysis['imbalanced']:
                opportunities['enhancement_opportunities'].append('feature_engineering')
                opportunities['parameter_tuning_suggestions'].append({
                    'warning': 'Feature importance is heavily imbalanced',
                    'reason': f'{importance_analysis["concentration_ratio"]:.2%} of importance in top 10% of features',
                    'improvement_potential': 'Review feature selection and consider feature engineering'
                })

            # 4. Check for overfitting potential
            overfitting_potential = self._check_overfitting_potential(model, X_train, X_val, y_train, y_val)
            if overfitting_potential > 0.6:
                opportunities['enhancement_opportunities'].append('regularization_increase')
                opportunities['parameter_tuning_suggestions'].append({
                    'warning': 'Model shows signs of potential overfitting',
                    'reason': f'Overfitting potential score: {overfitting_potential:.2f}',
                    'improvement_potential': 'Consider increasing regularization to prevent overfitting'
                })

            # 5. Check for optimization opportunities
            optimization_opportunities = self._check_optimization_opportunities(model, model_type)
            opportunities['enhancement_opportunities'].extend(optimization_opportunities)

            # Add warnings for optimization opportunities instead of specific recommendations
            for opportunity in optimization_opportunities:
                opportunities['parameter_tuning_suggestions'].append({
                    'warning': f'Model-specific optimization opportunity detected: {opportunity.replace("_", " ")}',
                    'reason': f'{model_type} models can benefit from {opportunity.replace("_", " ")}',
                    'improvement_potential': f'Consider model-specific optimizations for {opportunity.replace("_", " ")}'
                })

            # Calculate overall enhancement potential
            opportunities['confidence_level'] = self._calculate_enhancement_confidence(opportunities)
            opportunities['priority'] = self._determine_priority(opportunities)
            opportunities['estimated_improvement_potential'] = self._estimate_improvement_potential(opportunities)

            # Generate detailed recommendations
            opportunities['detailed_recommendations'] = self._generate_detailed_recommendations(opportunities)

        except Exception as e:
            self.logger.error(f"Model enhancement detection failed: {e}")
            opportunities['error'] = str(e)

        return opportunities

    def _check_underfitting(self, model, X_train, X_val, y_train, y_val) -> float:
        \"\"\"Check if model is underfitting (score from 0.0 to 1.0).\"\"\"
        try:
            # Get predictions
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

            # Calculate metrics
            train_mse = np.mean((y_train - train_pred) ** 2)
            val_mse = np.mean((y_val - val_pred) ** 2)

            # Normalize by target variance
            target_var = np.var(y_train)
            if target_var == 0:
                return 0.0

            train_normalized_error = train_mse / target_var
            val_normalized_error = val_mse / target_var

            # Underfitting score: higher when both train and val errors are high
            underfitting_score = min(1.0, (train_normalized_error + val_normalized_error) / 2.0)

            return underfitting_score

        except Exception as e:
            self.logger.warning(f"Underfitting check failed: {e}")
            return 0.0

    def _analyze_parameter_sensitivity(self, model, X_train, y_train) -> Dict[str, Any]:
        \"\"\"Analyze parameter sensitivity to determine tuning needs.\"\"\"
        analysis = {
            'high_sensitivity': False,
            'model_type': model.__class__.__name__.lower()
        }

        try:
            # Simple parameter sensitivity check based on model type
            model_type = model.__class__.__name__.lower()

            # All these model types typically benefit from parameter tuning
            if ('xgb' in model_type or 'xgboost' in model_type or
                'lgbm' in model_type or 'lightgbm' in model_type or
                'catboost' in model_type or
                'randomforest' in model_type or 'neural' in model_type or
                'torch' in model_type or 'keras' in model_type or
                'deepscaler' in model_type or 'mamba' in model_type or
                'linear' in model_type or 'ridge' in model_type or
                'lasso' in model_type or 'elasticnet' in model_type):

                analysis['high_sensitivity'] = True

        except Exception as e:
            self.logger.warning(f"Parameter sensitivity analysis failed: {e}")

        return analysis

    def _analyze_feature_importance(self, model, X_train) -> Dict[str, Any]:
        \"\"\"Analyze feature importance distribution.\"\"\"
        analysis = {
            'imbalanced': False,
            'concentration_ratio': 0.0,
            'top_features': []
        }

        try:
            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_).flatten()
            else:
                return analysis  # No feature importance available

            # Calculate concentration
            sorted_importance = np.sort(importance)[::-1]
            top_10_percent = sorted_importance[:max(1, len(sorted_importance) // 10)]
            analysis['concentration_ratio'] = np.sum(top_10_percent) / np.sum(sorted_importance)

            if analysis['concentration_ratio'] > 0.8:  # 80% of importance in top 10% features
                analysis['imbalanced'] = True

            # Get top feature indices
            top_indices = np.argsort(importance)[::-1][:10]
            analysis['top_features'] = top_indices.tolist()

        except Exception as e:
            self.logger.warning(f"Feature importance analysis failed: {e}")

        return analysis

    def _check_overfitting_potential(self, model, X_train, X_val, y_train, y_val) -> float:
        \"\"\"Check potential for overfitting (score from 0.0 to 1.0).\"\"\"
        try:
            # Get predictions
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

            # Calculate train vs validation performance gap
            train_mse = np.mean((y_train - train_pred) ** 2)
            val_mse = np.mean((y_val - val_pred) ** 2)

            # Calculate overfitting potential
            if train_mse == 0:
                return 1.0  # Perfect training fit = high overfitting risk

            performance_ratio = val_mse / train_mse
            overfitting_potential = min(1.0, max(0.0, 1.0 - 1.0 / (1.0 + performance_ratio)))

            return overfitting_potential

        except Exception as e:
            self.logger.warning(f"Overfitting potential check failed: {e}")
            return 0.0

    def _check_optimization_opportunities(self, model, model_type: str) -> List[str]:
        \"\"\"Check for model-specific optimization opportunities.\"\"\"
        opportunities = []

        try:
            # Model-specific optimization opportunities
            if ('neural' in model_type.lower() or 'torch' in model_type.lower() or
                'keras' in model_type.lower() or 'deepscaler' in model_type.lower() or
                'mamba' in model_type.lower()):
                opportunities.extend([
                    'learning_rate_scheduling',
                    'batch_normalization',
                    'gradient_clipping',
                    'early_stopping_optimization',
                    'architecture_optimization',
                    'attention_mechanism_tuning'
                ])

                # Add specific optimizations for advanced architectures
                if 'deepscaler' in model_type.lower():
                    opportunities.extend([
                        'scaling_factor_optimization',
                        'time_series_preprocessing_tuning',
                        'multi_scale_feature_integration'
                    ])
                elif 'mamba' in model_type.lower():
                    opportunities.extend([
                        'state_space_optimization',
                        'selective_scan_tuning',
                        'hardware_aware_optimization'
                    ])

            elif 'xgb' in model_type.lower() or 'xgboost' in model_type.lower() or 'lgbm' in model_type.lower() or 'lightgbm' in model_type.lower() or 'catboost' in model_type.lower():
                opportunities.extend([
                    'tree_structure_optimization',
                    'feature_interaction_constraints',
                    'monotone_constraints',
                    'categorical_feature_handling',
                    'boosting_round_optimization'
                ])

            elif 'linear' in model_type.lower() or 'ridge' in model_type.lower() or 'lasso' in model_type.lower() or 'elasticnet' in model_type.lower():
                opportunities.extend([
                    'regularization_optimization',
                    'feature_scaling_check',
                    'multicollinearity_analysis'
                ])

            elif 'randomforest' in model_type.lower() or 'extratrees' in model_type.lower():
                opportunities.extend([
                    'ensemble_diversity_optimization',
                    'feature_sampling_optimization',
                    'bootstrap_optimization'
                ])

            elif 'svm' in model_type.lower() or 'svc' in model_type.lower():
                opportunities.extend([
                    'kernel_optimization',
                    'gamma_parameter_tuning',
                    'class_weight_optimization'
                ])

            elif 'knn' in model_type.lower():
                opportunities.extend([
                    'distance_metric_optimization',
                    'neighbor_count_optimization',
                    'weight_function_optimization'
                ])

            elif 'bayesian' in model_type.lower() or 'naive' in model_type.lower():
                opportunities.extend([
                    'prior_optimization',
                    'smoothing_parameter_tuning',
                    'feature_independence_assumptions'
                ])

            # Default opportunities for unknown model types
            else:
                opportunities.extend([
                    'general_hyperparameter_tuning',
                    'ensemble_methods',
                    'cross_validation_optimization'
                ])

        except Exception as e:
            self.logger.warning(f"Optimization opportunities check failed: {e}")

        return opportunities

    def _calculate_enhancement_confidence(self, opportunities: Dict[str, Any]) -> float:
        \"\"\"Calculate confidence level for enhancement recommendations.\"\"\"
        confidence_factors = []

        # Base confidence
        base_confidence = 0.5

        # Factor based on number of opportunities found
        n_opportunities = len(opportunities['enhancement_opportunities'])
        opportunity_factor = min(0.3, n_opportunities * 0.1)

        # Factor based on parameter tuning suggestions
        n_suggestions = len(opportunities['parameter_tuning_suggestions'])
        suggestion_factor = min(0.2, n_suggestions * 0.05)

        total_confidence = base_confidence + opportunity_factor + suggestion_factor

        return min(1.0, total_confidence)

    def _determine_priority(self, opportunities: Dict[str, Any]) -> str:
        \"\"\"Determine priority level for enhancement.\"\"\"
        n_opportunities = len(opportunities['enhancement_opportunities'])
        confidence = opportunities['confidence_level']

        if n_opportunities >= 3 and confidence > 0.8:
            return 'critical'
        elif n_opportunities >= 2 and confidence > 0.6:
            return 'high'
        elif n_opportunities >= 1 and confidence > 0.4:
            return 'medium'
        else:
            return 'low'

    def _estimate_improvement_potential(self, opportunities: Dict[str, Any]) -> float:
        \"\"\"Estimate potential improvement from enhancements.\"\"\"
        improvement_factors = {
            'model_complexity_increase': 0.15,
            'parameter_tuning_needed': 0.20,
            'feature_engineering': 0.10,
            'regularization_increase': 0.05,
            'learning_rate_scheduling': 0.08,
            'tree_structure_optimization': 0.12,
            'categorical_feature_handling': 0.10,
            'boosting_round_optimization': 0.09,
            'architecture_optimization': 0.14,
            'attention_mechanism_tuning': 0.11,
            'scaling_factor_optimization': 0.13,
            'time_series_preprocessing_tuning': 0.12,
            'multi_scale_feature_integration': 0.11,
            'state_space_optimization': 0.15,
            'selective_scan_tuning': 0.13,
            'hardware_aware_optimization': 0.10
        }

        total_potential = 0.0
        for opportunity in opportunities['enhancement_opportunities']:
            if opportunity in improvement_factors:
                total_potential += improvement_factors[opportunity]

        return min(0.5, total_potential)  # Cap at 50% potential improvement

    def _generate_detailed_recommendations(self, opportunities: Dict[str, Any]) -> List[str]:
        \"\"\"Generate detailed recommendations based on analysis.\"\"\"
        recommendations = []

        if opportunities['priority'] == 'critical':
            recommendations.append("🚨 CRITICAL: Immediate model enhancement required")
        elif opportunities['priority'] == 'high':
            recommendations.append("⚠️ HIGH: Strong enhancement opportunities identified")
        elif opportunities['priority'] == 'medium':
            recommendations.append("📊 MEDIUM: Moderate enhancement opportunities available")
        else:
            recommendations.append("✅ LOW: Minimal enhancement opportunities found")

        # Add specific recommendations based on opportunities
        for opportunity in opportunities['enhancement_opportunities']:
            if opportunity == 'model_complexity_increase':
                recommendations.append("🔧 Consider increasing model complexity (deeper trees, more estimators, additional layers)")
            elif opportunity == 'parameter_tuning_needed':
                recommendations.append("🔧 Perform comprehensive hyperparameter optimization")
            elif opportunity == 'feature_engineering':
                recommendations.append("🔧 Review feature selection and consider feature engineering")
            elif opportunity == 'regularization_increase':
                recommendations.append("🔧 Increase regularization to prevent overfitting")

        return recommendations


def detect_overfitting_for_model(model,
                                X_train: np.ndarray,
                                X_val: np.ndarray,
                                y_train: np.ndarray,
                                y_val: np.ndarray,
                                model_name: str = "unknown",
                                model_type: str = "unknown",
                                fold_number: Optional[int] = None,
                                config: Optional[OverfittingConfig] = None) -> OverfittingReport:
    \"\"\"
    Convenience function to detect overfitting for any ML model.
    
    Args:
        model: Trained ML model
        X_train: Training features
        X_val: Validation features
        y_train: Training labels
        y_val: Validation labels
        model_name: Name of the model
        model_type: Type of model
        fold_number: Fold number for cross-validation
        config: Overfitting detection configuration
        
    Returns:
        OverfittingReport: Comprehensive overfitting analysis
    \"\"\"
    detector = get_overfitting_detector(config)
    
    # Get predictions
    train_predictions = model.predict(X_train)
    val_predictions = model.predict(X_val)
    
    # Get probabilities if available
    train_probabilities = None
    val_probabilities = None
    if hasattr(model, 'predict_proba'):
        try:
            train_probabilities = model.predict_proba(X_train)
            val_probabilities = model.predict_proba(X_val)
        except Exception as e:
            logger.warning(f"Could not get probabilities from model: {e}")
            # Continue without probabilities - they're optional
    
    # Get feature importance if available
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        feature_importance = np.abs(model.coef_).flatten()
    
    # Detect overfitting
    return detector.detect_overfitting(
        train_predictions=train_predictions,
        val_predictions=val_predictions,
        train_labels=y_train,
        val_labels=y_val,
        train_probabilities=train_probabilities,
        val_probabilities=val_probabilities,
        feature_importance=feature_importance,
        model_name=model_name,
        model_type=model_type,
        fold_number=fold_number
    )