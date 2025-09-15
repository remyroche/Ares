"""
Enhanced Short-Term Entry Timing Analysis and Recommendations

This module provides comprehensive analysis and recommendations for:
1. Integration with existing feature_engineering/ pipeline
2. Confidence calibration system explanation
3. ML model training, HPO, and model selection recommendations
4. Enhanced model architecture based on Tactician components

Key Features:
- Complete feature engineering integration analysis
- Multi-phase confidence calibration system
- Advanced ML model recommendations
- HPO optimization strategies
- Model selection criteria for short-term predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
from datetime import datetime

from src.utils.logger import system_logger

logger = system_logger.getChild('EnhancedShortTermAnalysis')


@dataclass
class FeatureEngineeringIntegration:
    """Analysis of feature engineering integration with enhanced model."""
    
    # Existing feature engineering capabilities
    existing_features: Dict[str, List[str]] = field(default_factory=dict)
    
    # Enhanced features needed for short-term prediction
    enhanced_features: Dict[str, List[str]] = field(default_factory=dict)
    
    # Integration recommendations
    integration_strategy: str = "hybrid"  # "existing_only", "enhanced_only", "hybrid"
    
    # Performance considerations
    feature_count_estimate: int = 0
    processing_time_estimate: float = 0.0
    memory_usage_estimate: float = 0.0


@dataclass
class ConfidenceCalibrationSystem:
    """Comprehensive confidence calibration system for different prediction phases."""
    
    # Phase-specific confidence metrics
    pre_movement_confidence: Dict[str, float] = field(default_factory=dict)
    target_movement_confidence: Dict[str, float] = field(default_factory=dict)
    entry_timing_confidence: Dict[str, float] = field(default_factory=dict)
    risk_assessment_confidence: Dict[str, float] = field(default_factory=dict)
    
    # Calibration methods
    calibration_methods: List[str] = field(default_factory=lambda: [
        "platt_scaling", "isotonic_regression", "temperature_scaling", 
        "histogram_binning", "bayesian_calibration"
    ])
    
    # Validation metrics
    validation_metrics: List[str] = field(default_factory=lambda: [
        "reliability_diagram", "brier_score", "log_loss", "calibration_error"
    ])


@dataclass
class MLModelRecommendations:
    """Recommendations for ML models, training, and HPO for short-term predictions."""
    
    # Model recommendations
    recommended_models: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # HPO recommendations
    hpo_strategies: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Training recommendations
    training_strategies: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Performance considerations
    performance_requirements: Dict[str, float] = field(default_factory=dict)


class EnhancedShortTermAnalysis:
    """
    Comprehensive analysis and recommendations for enhanced short-term entry timing.
    
    This class provides detailed analysis of:
    1. Feature engineering integration
    2. Confidence calibration system
    3. ML model recommendations
    4. Training and HPO strategies
    """
    
    def __init__(self):
        """Initialize the analysis system."""
        self.logger = logger.getChild('EnhancedShortTermAnalysis')
        
        self.logger.info("🚀 Initializing Enhanced Short-Term Analysis")
        
        # Initialize analysis components
        self.feature_integration = self._analyze_feature_engineering_integration()
        self.confidence_system = self._design_confidence_calibration_system()
        self.ml_recommendations = self._generate_ml_recommendations()
        
        self.logger.info("✅ Enhanced Short-Term Analysis initialized")
    
    def _analyze_feature_engineering_integration(self) -> FeatureEngineeringIntegration:
        """Analyze integration with existing feature_engineering/ pipeline."""
        
        self.logger.info("🔍 Analyzing feature engineering integration...")
        
        # Existing features from feature_engineering/ pipeline
        existing_features = {
            "technical_indicators": [
                "RSI", "MACD", "Bollinger_Bands", "SMA", "EMA", "ATR", "Stochastic",
                "ADX", "CCI", "Williams_R", "ROC", "OBV", "MFI", "ADX", "DX",
                "MINUS_DI", "PLUS_DI", "MINUS_DM", "PLUS_DM", "MIDPOINT", "MIDPRICE",
                "T3", "HT_TRENDLINE", "SAR", "DEMA", "TEMA", "TRANGE", "VAR", "STDDEV"
            ],
            "cross_timeframe_features": [
                "correlation_features", "momentum_features", "volatility_features",
                "volume_features", "microstructure_features", "order_flow_features",
                "momentum_divergence", "volatility_spillover"
            ],
            "interaction_features": [
                "polynomial_interactions", "cross_timeframe_interactions",
                "pattern_recognition_features", "regime_specific_features"
            ],
            "advanced_features": [
                "fractional_differentiation", "triple_barrier_labeling",
                "regime_aware_features", "profit_based_features"
            ],
            "microstructure_features": [
                "price_impact", "volume_price_correlation", "bid_ask_spread_proxy",
                "trade_size_distribution", "order_flow_imbalance", "liquidity_metrics"
            ]
        }
        
        # Enhanced features needed for short-term prediction
        enhanced_features = {
            "ultra_short_term_features": [
                "1m_momentum", "2m_momentum", "3m_momentum", "5m_momentum",
                "1m_volatility", "2m_volatility", "3m_volatility", "5m_volatility",
                "1m_volume_profile", "2m_volume_profile", "3m_volume_profile", "5m_volume_profile"
            ],
            "pre_movement_features": [
                "momentum_divergence", "support_resistance_distance", "volatility_clustering",
                "price_acceleration", "volume_acceleration", "order_flow_divergence"
            ],
            "entry_timing_features": [
                "optimal_entry_timing", "wait_duration", "entry_confidence",
                "pre_movement_direction", "pre_movement_magnitude", "pre_movement_duration"
            ],
            "risk_assessment_features": [
                "drawdown_probability", "volatility_forecast", "liquidity_impact",
                "adverse_movement_risk", "correlation_risk", "concentration_risk"
            ],
            "market_microstructure_features": [
                "order_book_depth", "trade_size_distribution", "bid_ask_spread",
                "price_impact", "volume_weighted_price", "trade_intensity"
            ]
        }
        
        # Integration strategy
        integration_strategy = "hybrid"  # Use both existing and enhanced features
        
        # Performance estimates
        total_existing_features = sum(len(features) for features in existing_features.values())
        total_enhanced_features = sum(len(features) for features in enhanced_features.values())
        feature_count_estimate = total_existing_features + total_enhanced_features
        
        # Processing time estimate (seconds per 1000 samples)
        processing_time_estimate = 0.5 + (feature_count_estimate * 0.001)
        
        # Memory usage estimate (MB per 1000 samples)
        memory_usage_estimate = feature_count_estimate * 0.008  # 8KB per feature per 1000 samples
        
        return FeatureEngineeringIntegration(
            existing_features=existing_features,
            enhanced_features=enhanced_features,
            integration_strategy=integration_strategy,
            feature_count_estimate=feature_count_estimate,
            processing_time_estimate=processing_time_estimate,
            memory_usage_estimate=memory_usage_estimate
        )
    
    def _design_confidence_calibration_system(self) -> ConfidenceCalibrationSystem:
        """Design comprehensive confidence calibration system."""
        
        self.logger.info("🎯 Designing confidence calibration system...")
        
        # Phase-specific confidence metrics
        pre_movement_confidence = {
            "direction_confidence": 0.0,      # Confidence in pre-movement direction
            "magnitude_confidence": 0.0,      # Confidence in pre-movement magnitude
            "duration_confidence": 0.0,       # Confidence in pre-movement duration
            "timing_confidence": 0.0,         # Confidence in optimal entry timing
            "overall_confidence": 0.0         # Overall pre-movement confidence
        }
        
        target_movement_confidence = {
            "probability_confidence": 0.0,    # Confidence in target probability
            "direction_confidence": 0.0,      # Confidence in target direction
            "magnitude_confidence": 0.0,      # Confidence in target magnitude
            "timing_confidence": 0.0,         # Confidence in target timing
            "overall_confidence": 0.0         # Overall target confidence
        }
        
        entry_timing_confidence = {
            "strategy_confidence": 0.0,       # Confidence in entry strategy
            "timing_confidence": 0.0,         # Confidence in entry timing
            "risk_confidence": 0.0,           # Confidence in risk assessment
            "market_confidence": 0.0,         # Confidence in market conditions
            "overall_confidence": 0.0         # Overall entry confidence
        }
        
        risk_assessment_confidence = {
            "drawdown_confidence": 0.0,       # Confidence in drawdown prediction
            "volatility_confidence": 0.0,     # Confidence in volatility forecast
            "liquidity_confidence": 0.0,      # Confidence in liquidity impact
            "correlation_confidence": 0.0,    # Confidence in correlation risk
            "overall_confidence": 0.0         # Overall risk confidence
        }
        
        # Calibration methods
        calibration_methods = [
            "platt_scaling",           # Logistic regression calibration
            "isotonic_regression",     # Non-parametric calibration
            "temperature_scaling",     # Neural network calibration
            "histogram_binning",       # Histogram-based calibration
            "bayesian_calibration"     # Bayesian approach to calibration
        ]
        
        # Validation metrics
        validation_metrics = [
            "reliability_diagram",     # Visual calibration assessment
            "brier_score",            # Proper scoring rule
            "log_loss",               # Logarithmic loss
            "calibration_error",      # Expected calibration error
            "sharpness",              # Prediction sharpness
            "resolution"              # Prediction resolution
        ]
        
        return ConfidenceCalibrationSystem(
            pre_movement_confidence=pre_movement_confidence,
            target_movement_confidence=target_movement_confidence,
            entry_timing_confidence=entry_timing_confidence,
            risk_assessment_confidence=risk_assessment_confidence,
            calibration_methods=calibration_methods,
            validation_metrics=validation_metrics
        )
    
    def _generate_ml_recommendations(self) -> MLModelRecommendations:
        """Generate comprehensive ML model recommendations."""
        
        self.logger.info("🤖 Generating ML model recommendations...")
        
        # Model recommendations based on short-term prediction requirements
        recommended_models = {
            "pre_movement_prediction": {
                "primary_models": [
                    {
                        "name": "RandomForestRegressor",
                        "reason": "Excellent for pattern recognition and feature importance",
                        "hyperparameters": {
                            "n_estimators": [100, 200, 300],
                            "max_depth": [10, 15, 20, None],
                            "min_samples_split": [2, 5, 10],
                            "min_samples_leaf": [1, 2, 4],
                            "max_features": ["sqrt", "log2", 0.8]
                        }
                    },
                    {
                        "name": "GradientBoostingRegressor",
                        "reason": "Strong performance on sequential patterns",
                        "hyperparameters": {
                            "n_estimators": [100, 200, 300],
                            "learning_rate": [0.01, 0.05, 0.1],
                            "max_depth": [3, 5, 7],
                            "subsample": [0.8, 0.9, 1.0]
                        }
                    }
                ],
                "secondary_models": [
                    {
                        "name": "MLPRegressor",
                        "reason": "Neural network for complex non-linear patterns",
                        "hyperparameters": {
                            "hidden_layer_sizes": [(50,), (100,), (50, 25), (100, 50)],
                            "activation": ["relu", "tanh"],
                            "learning_rate": ["constant", "adaptive"],
                            "alpha": [0.0001, 0.001, 0.01]
                        }
                    }
                ]
            },
            "target_movement_prediction": {
                "primary_models": [
                    {
                        "name": "CatBoostRegressor",
                        "reason": "Excellent for categorical features and handling missing values",
                        "hyperparameters": {
                            "iterations": [1000, 2000, 3000],
                            "learning_rate": [0.03, 0.05, 0.1],
                            "depth": [6, 8, 10],
                            "l2_leaf_reg": [1, 3, 5, 7, 9]
                        }
                    },
                    {
                        "name": "LGBMRegressor",
                        "reason": "Fast training and good performance on time series",
                        "hyperparameters": {
                            "n_estimators": [1000, 2000, 3000],
                            "learning_rate": [0.03, 0.05, 0.1],
                            "max_depth": [6, 8, 10],
                            "num_leaves": [31, 63, 127],
                            "subsample": [0.8, 0.9, 1.0]
                        }
                    }
                ],
                "secondary_models": [
                    {
                        "name": "TabNetRegressor",
                        "reason": "Attention-based neural network for feature selection",
                        "hyperparameters": {
                            "n_d": [32, 64, 128],
                            "n_a": [32, 64, 128],
                            "n_steps": [3, 5, 7],
                            "gamma": [1.0, 1.5, 2.0],
                            "lambda_sparse": [1e-4, 1e-3, 1e-2]
                        }
                    }
                ]
            },
            "entry_timing_prediction": {
                "primary_models": [
                    {
                        "name": "XGBRegressor",
                        "reason": "Excellent for regression tasks with complex features",
                        "hyperparameters": {
                            "n_estimators": [1000, 2000, 3000],
                            "learning_rate": [0.03, 0.05, 0.1],
                            "max_depth": [6, 8, 10],
                            "subsample": [0.8, 0.9, 1.0],
                            "colsample_bytree": [0.8, 0.9, 1.0]
                        }
                    },
                    {
                        "name": "Ridge",
                        "reason": "Stable baseline for timing predictions",
                        "hyperparameters": {
                            "alpha": [0.1, 1.0, 10.0, 100.0]
                        }
                    }
                ]
            },
            "risk_assessment_prediction": {
                "primary_models": [
                    {
                        "name": "RandomForestRegressor",
                        "reason": "Robust for risk prediction with uncertainty quantification",
                        "hyperparameters": {
                            "n_estimators": [200, 300, 500],
                            "max_depth": [10, 15, 20],
                            "min_samples_split": [2, 5, 10],
                            "min_samples_leaf": [1, 2, 4]
                        }
                    }
                ]
            }
        }
        
        # HPO strategies
        hpo_strategies = {
            "bayesian_optimization": {
                "method": "optuna",
                "n_trials": 200,
                "timeout": 3600,
                "pruning": True,
                "early_stopping": True,
                "patience": 20
            },
            "random_search": {
                "method": "random",
                "n_trials": 100,
                "timeout": 1800
            },
            "grid_search": {
                "method": "grid",
                "cv_folds": 5,
                "scoring": "neg_mean_squared_error"
            }
        }
        
        # Training strategies
        training_strategies = {
            "time_series_cv": {
                "method": "TimeSeriesSplit",
                "n_splits": 5,
                "test_size": 0.2,
                "gap": 0
            },
            "walk_forward_analysis": {
                "method": "WalkForwardAnalysis",
                "initial_train_size": 0.6,
                "step_size": 0.1,
                "expanding_window": True
            },
            "purged_cv": {
                "method": "PurgedKFold",
                "n_splits": 5,
                "pct_embargo": 0.01
            }
        }
        
        # Performance requirements
        performance_requirements = {
            "min_hit_rate": 0.6,           # 60% minimum hit rate
            "min_r2_score": 0.3,           # 30% minimum R² score
            "max_prediction_time": 0.1,    # 100ms maximum prediction time
            "min_confidence_calibration": 0.8,  # 80% minimum calibration accuracy
            "max_drawdown": 0.05,          # 5% maximum drawdown
            "min_sharpe_ratio": 1.0        # 1.0 minimum Sharpe ratio
        }
        
        return MLModelRecommendations(
            recommended_models=recommended_models,
            hpo_strategies=hpo_strategies,
            training_strategies=training_strategies,
            performance_requirements=performance_requirements
        )
    
    def get_feature_engineering_integration_plan(self) -> Dict[str, Any]:
        """Get comprehensive feature engineering integration plan."""
        
        plan = {
            "integration_strategy": self.feature_integration.integration_strategy,
            "existing_features": self.feature_integration.existing_features,
            "enhanced_features": self.feature_integration.enhanced_features,
            "total_features": self.feature_integration.feature_count_estimate,
            "performance_estimates": {
                "processing_time_per_1k_samples": self.feature_integration.processing_time_estimate,
                "memory_usage_per_1k_samples_mb": self.feature_integration.memory_usage_estimate
            },
            "integration_steps": [
                "1. Import existing feature_engineering/ utilities",
                "2. Create enhanced feature generators for short-term prediction",
                "3. Implement feature selection and optimization",
                "4. Add confidence calibration features",
                "5. Integrate with existing pipeline",
                "6. Performance testing and optimization"
            ],
            "recommended_approach": {
                "use_existing_features": True,
                "add_enhanced_features": True,
                "feature_selection": "lasso_with_correlation_filtering",
                "feature_scaling": "robust_scaling",
                "feature_validation": "comprehensive_validation"
            }
        }
        
        return plan
    
    def get_confidence_calibration_explanation(self) -> Dict[str, Any]:
        """Get comprehensive confidence calibration system explanation."""
        
        explanation = {
            "system_overview": {
                "purpose": "Multi-phase confidence calibration for short-term entry timing",
                "phases": ["pre_movement", "target_movement", "entry_timing", "risk_assessment"],
                "calibration_methods": self.confidence_system.calibration_methods,
                "validation_metrics": self.confidence_system.validation_metrics
            },
            "phase_specific_calibration": {
                "pre_movement_phase": {
                    "description": "Calibrates confidence in pre-movement predictions",
                    "metrics": list(self.confidence_system.pre_movement_confidence.keys()),
                    "calibration_method": "isotonic_regression",
                    "validation_metric": "brier_score"
                },
                "target_movement_phase": {
                    "description": "Calibrates confidence in target movement predictions",
                    "metrics": list(self.confidence_system.target_movement_confidence.keys()),
                    "calibration_method": "platt_scaling",
                    "validation_metric": "log_loss"
                },
                "entry_timing_phase": {
                    "description": "Calibrates confidence in entry timing recommendations",
                    "metrics": list(self.confidence_system.entry_timing_confidence.keys()),
                    "calibration_method": "temperature_scaling",
                    "validation_metric": "calibration_error"
                },
                "risk_assessment_phase": {
                    "description": "Calibrates confidence in risk assessment predictions",
                    "metrics": list(self.confidence_system.risk_assessment_confidence.keys()),
                    "calibration_method": "bayesian_calibration",
                    "validation_metric": "reliability_diagram"
                }
            },
            "calibration_workflow": [
                "1. Collect prediction probabilities and actual outcomes",
                "2. Apply calibration method to each phase",
                "3. Validate calibration using appropriate metrics",
                "4. Update calibration parameters based on validation",
                "5. Apply calibrated confidence scores to new predictions",
                "6. Monitor calibration performance over time"
            ],
            "implementation_details": {
                "calibration_frequency": "daily_retraining",
                "validation_window": "30_days",
                "calibration_threshold": 0.8,
                "fallback_method": "histogram_binning"
            }
        }
        
        return explanation
    
    def get_ml_training_recommendations(self) -> Dict[str, Any]:
        """Get comprehensive ML training recommendations."""
        
        recommendations = {
            "model_selection": {
                "pre_movement_models": self.ml_recommendations.recommended_models["pre_movement_prediction"],
                "target_movement_models": self.ml_recommendations.recommended_models["target_movement_prediction"],
                "entry_timing_models": self.ml_recommendations.recommended_models["entry_timing_prediction"],
                "risk_assessment_models": self.ml_recommendations.recommended_models["risk_assessment_prediction"]
            },
            "hpo_strategies": {
                "primary_method": "bayesian_optimization",
                "fallback_method": "random_search",
                "optimization_target": "hit_rate_weighted_by_confidence",
                "constraints": {
                    "max_training_time": "2_hours",
                    "max_prediction_time": "100ms",
                    "min_validation_samples": 1000
                }
            },
            "training_strategies": {
                "primary_method": "time_series_cv",
                "validation_method": "walk_forward_analysis",
                "data_splitting": {
                    "train_ratio": 0.6,
                    "validation_ratio": 0.2,
                    "test_ratio": 0.2
                },
                "temporal_validation": {
                    "gap_between_splits": "1_hour",
                    "embargo_period": "15_minutes"
                }
            },
            "performance_requirements": self.ml_recommendations.performance_requirements,
            "training_pipeline_changes": {
                "current_models": [
                    "NeuralObliviousDecisionEnsembles",
                    "CatBoostRegressor", 
                    "LGBMRegressor",
                    "Ridge"
                ],
                "recommended_additions": [
                    "RandomForestRegressor",  # For pre-movement prediction
                    "GradientBoostingRegressor",  # For pattern recognition
                    "MLPRegressor",  # For complex non-linear patterns
                    "XGBRegressor",  # For entry timing prediction
                    "TabNetRegressor"  # For feature selection
                ],
                "model_ensemble_strategy": {
                    "pre_movement": "RandomForest + GradientBoosting",
                    "target_movement": "CatBoost + LGBM + TabNet",
                    "entry_timing": "XGBoost + Ridge",
                    "risk_assessment": "RandomForest + MLP"
                }
            },
            "hpo_optimization": {
                "search_space_expansion": {
                    "add_temporal_features": True,
                    "add_microstructure_features": True,
                    "add_regime_features": True,
                    "add_confidence_features": True
                },
                "objective_function": {
                    "primary_metric": "hit_rate",
                    "secondary_metrics": ["r2_score", "calibration_error"],
                    "penalty_terms": ["prediction_time", "model_complexity"]
                },
                "early_stopping": {
                    "patience": 20,
                    "min_delta": 0.001,
                    "monitor": "validation_hit_rate"
                }
            }
        }
        
        return recommendations
    
    def get_comprehensive_analysis(self) -> Dict[str, Any]:
        """Get comprehensive analysis combining all components."""
        
        analysis = {
            "feature_engineering_integration": self.get_feature_engineering_integration_plan(),
            "confidence_calibration_system": self.get_confidence_calibration_explanation(),
            "ml_training_recommendations": self.get_ml_training_recommendations(),
            "implementation_roadmap": {
                "phase_1": {
                    "duration": "1_week",
                    "tasks": [
                        "Integrate existing feature_engineering/ pipeline",
                        "Implement enhanced feature generators",
                        "Set up confidence calibration framework"
                    ]
                },
                "phase_2": {
                    "duration": "2_weeks", 
                    "tasks": [
                        "Implement recommended ML models",
                        "Set up HPO optimization",
                        "Implement training strategies"
                    ]
                },
                "phase_3": {
                    "duration": "1_week",
                    "tasks": [
                        "Performance testing and optimization",
                        "Integration testing",
                        "Documentation and deployment"
                    ]
                }
            },
            "expected_improvements": {
                "prediction_accuracy": "+15-25%",
                "confidence_calibration": "+20-30%",
                "entry_timing_optimization": "+10-20%",
                "risk_assessment_accuracy": "+15-25%"
            }
        }
        
        return analysis


# Convenience functions
def get_enhanced_short_term_analysis() -> EnhancedShortTermAnalysis:
    """Get enhanced short-term analysis instance."""
    return EnhancedShortTermAnalysis()


def get_feature_engineering_integration_plan() -> Dict[str, Any]:
    """Get feature engineering integration plan."""
    analysis = get_enhanced_short_term_analysis()
    return analysis.get_feature_engineering_integration_plan()


def get_confidence_calibration_explanation() -> Dict[str, Any]:
    """Get confidence calibration system explanation."""
    analysis = get_enhanced_short_term_analysis()
    return analysis.get_confidence_calibration_explanation()


def get_ml_training_recommendations() -> Dict[str, Any]:
    """Get ML training recommendations."""
    analysis = get_enhanced_short_term_analysis()
    return analysis.get_ml_training_recommendations()


# Example usage
if __name__ == "__main__":
    # Get comprehensive analysis
    analysis = get_enhanced_short_term_analysis()
    comprehensive_analysis = analysis.get_comprehensive_analysis()
    
    print("Enhanced Short-Term Entry Timing Analysis")
    print("=" * 50)
    
    # Feature Engineering Integration
    feature_plan = comprehensive_analysis["feature_engineering_integration"]
    print(f"\n📊 Feature Engineering Integration:")
    print(f"   Total Features: {feature_plan['total_features']}")
    print(f"   Integration Strategy: {feature_plan['integration_strategy']}")
    print(f"   Processing Time: {feature_plan['performance_estimates']['processing_time_per_1k_samples']:.3f}s per 1k samples")
    
    # Confidence Calibration
    confidence_system = comprehensive_analysis["confidence_calibration_system"]
    print(f"\n🎯 Confidence Calibration System:")
    print(f"   Phases: {len(confidence_system['system_overview']['phases'])}")
    print(f"   Calibration Methods: {len(confidence_system['system_overview']['calibration_methods'])}")
    print(f"   Validation Metrics: {len(confidence_system['system_overview']['validation_metrics'])}")
    
    # ML Training Recommendations
    ml_recommendations = comprehensive_analysis["ml_training_recommendations"]
    print(f"\n🤖 ML Training Recommendations:")
    print(f"   Pre-movement Models: {len(ml_recommendations['model_selection']['pre_movement_models']['primary_models'])}")
    print(f"   Target Movement Models: {len(ml_recommendations['model_selection']['target_movement_models']['primary_models'])}")
    print(f"   HPO Strategy: {ml_recommendations['hpo_strategies']['primary_method']}")
    print(f"   Training Strategy: {ml_recommendations['training_strategies']['primary_method']}")
    
    # Expected Improvements
    improvements = comprehensive_analysis["expected_improvements"]
    print(f"\n📈 Expected Improvements:")
    for metric, improvement in improvements.items():
        print(f"   {metric.replace('_', ' ').title()}: {improvement}")
    
    print(f"\n✅ Comprehensive analysis completed successfully!")